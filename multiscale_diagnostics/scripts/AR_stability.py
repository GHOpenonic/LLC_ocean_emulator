"""AR stability of multiscale emulators vs LLC truth: area-weighted mean / std / RMSE per AR step over coarse outside N,
coarse inside N and sharp; (time step, sharp row band) tasks run in parallel."""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import multiprocessing as mp
import sys
import time as _time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.lines import Line2D

sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics')
from predictions_io import open_predictions

COARSE_CACHE = '/orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_cache_0p33deg.zarr'
FACE_CACHE = '/orcd/scratch/seedfund/001/llc_emulator/LLC4320_face1_i0-4320_j0-4320.zarr'
OUT_ROOT = '/home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/figs/AR-stability'
FIELDS = ['U', 'V', 'Theta', 'Salt']  # both caches: channel blocks of 51 levels in this order; Eta is channel 204
UNITS = {'Theta': '°C', 'Salt': 'psu', 'U': 'm s$^{-1}$', 'V': 'm s$^{-1}$', 'Eta': 'm'}
BOX = 16  # sharp cells per coarse cell
SCOPES = {'coarse-out': 'coarse outside N', 'coarse-in': 'coarse inside N', 'sharp': 'sharp'}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--emulator', action='append', required=True, metavar='NAME=DIR',
               help='repeat per emulator; DIR holds sharp_pred.zarr and coarse_pred.zarr (a path to either works too)')
p.add_argument('--vars', nargs='+', choices=list(UNITS), default=['Theta', 'Salt', 'U', 'V', 'Eta'])
p.add_argument('--stats', nargs='+', choices=['mean', 'std', 'rmse'], default=['mean', 'std', 'rmse'])
p.add_argument('--mode', choices=['FULL', 'COLUMN_MEAN_ONLY', 'SURFACE'], default='FULL',
               help='FULL (k x var grid) | COLUMN_MEAN_ONLY | SURFACE (k=0); the last two stack vars vertically')
p.add_argument('--depth-average', type=int, nargs=3, default=[0, 51, 5], metavar=('START', 'STOP', 'STEP'),
               help='column-mean levels')
p.add_argument('--depth-levels', type=int, nargs='*', default=[10, 25, 50], help='extra single-level rows in FULL mode')
p.add_argument('--time-range', type=int, nargs=2, default=[0, 744], metavar=('START', 'END'),
               help='inclusive hours since the IC')
p.add_argument('--stepping', type=int, default=3,
               help='hours; each store steps every max(stepping, its save interval) hours from the IC')
p.add_argument('--band', type=int, default=2160, help='sharp rows per task (multiple of both stores\' chunk rows)')
p.add_argument('--sharp-truth', default=FACE_CACHE, help='face cache holding the sharp domain')
p.add_argument('--workers', type=int, default=None)
p.add_argument('--out-dir', default=None)
args = p.parse_args()

N_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
N_WORKERS = args.workers or int(os.environ.get('AR_N_WORKERS', N_CPUS))
EMULATORS = [spec.split('=', 1) for spec in args.emulator]
EMU_NAMES = [name for name, _ in EMULATORS]
VARS, STATS, MODE, BAND = args.vars, args.stats, args.mode, args.band
DEPTH_AVERAGE = slice(*args.depth_average)
DEPTH_LEVELS = args.depth_levels
_t0 = _time.time()


def log(msg):
    print(f"[{_time.time() - _t0:6.0f}s] {msg}", flush=True)


ROW_KEYS = {'FULL': ['column', 'surface', *DEPTH_LEVELS], 'COLUMN_MEAN_ONLY': ['column'], 'SURFACE': ['surface']}[MODE]
_ALL_ROWS = {'column': list(range(*DEPTH_AVERAGE.indices(51))), 'surface': [0], **{k: [k] for k in DEPTH_LEVELS}}
LEVELS_TO_READ = sorted({k for r in ROW_KEYS for k in _ALL_ROWS[r]})
ROW_POS = {r: [LEVELS_TO_READ.index(k) for k in _ALL_ROWS[r]] for r in ROW_KEYS}

# Each var's channels in both caches, and its slice of the CH list read at once.
_CH = {v: [204] if v == 'Eta' else [51 * FIELDS.index(v) + k for k in LEVELS_TO_READ] for v in VARS}
CH = [c for v in VARS for c in _CH[v]]
_ends = np.cumsum([len(_CH[v]) for v in VARS])
CH_SLICE = {v: slice(e - len(_CH[v]), e) for v, e in zip(VARS, _ends)}


# ============== LOAD (lazy; workers read their own steps) ==============
def emulator_dir(path):
    d = Path(path)
    return d.parent if d.suffix == '.zarr' else d


COARSE_EMU, SHARP_EMU = {}, {}
for name, path in EMULATORS:
    d = emulator_dir(path)
    COARSE_EMU[name] = open_predictions(str(d / 'coarse_pred.zarr'))
    SHARP_EMU[name] = open_predictions(str(d / 'sharp_pred.zarr'))
starts = {name: ds.attrs['start'] for name, ds in COARSE_EMU.items()}
if len(set(starts.values())) > 1:
    raise ValueError(f"emulators start from different ICs: {starts}")
IC = np.datetime64(next(iter(starts.values())))

ct = xr.open_zarr(COARSE_CACHE, consolidated=False, chunks=None)
ft = xr.open_zarr(args.sharp_truth, consolidated=False, chunks=None)
FACE = int(ft.attrs['llc_face'])

# N: the sharp domain's coarse footprint, as validate_multiscale's tile_placement builds it.
lat, lon = SHARP_EMU[EMU_NAMES[0]].lat.values, SHARP_EMU[EMU_NAMES[0]].lon.values
for name in EMU_NAMES[1:]:
    if not (np.array_equal(SHARP_EMU[name].lat.values, lat) and np.array_equal(SHARP_EMU[name].lon.values, lon)):
        raise ValueError(f"{name}'s sharp domain differs from {EMU_NAMES[0]}'s")
J0, J1, I0, I1 = int(lat[0]), int(lat[-1]) + 1, int(lon[0]), int(lon[-1]) + 1
if FACE >= 7 or any(v % BOX for v in (J0, J1, I0, I1)):
    raise ValueError(f"face {FACE} j[{J0}:{J1}) i[{I0}:{I1}) isn't a native-orientation box of whole coarse cells")
NJ = J1 - J0
INSIDE = np.zeros((13, 270, 270), bool)
INSIDE[FACE, J0 // BOX:J1 // BOX, I0 // BOX:I1 // BOX] = True
INSIDE = INSIDE.ravel()
FY = slice(J0 - int(ft.attrs['llc_j_start']), J1 - int(ft.attrs['llc_j_start']))
FX = slice(I0 - int(ft.attrs['llc_i_start']), I1 - int(ft.attrs['llc_i_start']))

# Each emulator store steps every max(STEPPING, its save interval) hours from the IC; TIMES is their union.
STORES = {'coarse': COARSE_EMU, 'sharp': SHARP_EMU}
SCOPE_GRID = {'coarse-out': 'coarse', 'coarse-in': 'coarse', 'sharp': 'sharp'}


def _hours(times):
    return (times - IC) // np.timedelta64(1, 'h')


EVERY, PICKED = {}, {}
for g, stores in STORES.items():
    for n, ds in stores.items():
        h = _hours(ds.time.values)
        saved = int(np.diff(h).min())
        EVERY[g, n] = max(args.stepping, saved)
        if EVERY[g, n] % saved:
            raise ValueError(f"STEPPING {args.stepping} h isn't a multiple of {n}'s {g} save interval ({saved} h)")
        PICKED[g, n] = ds.time.values[(h >= args.time_range[0]) & (h <= args.time_range[1]) & (h % EVERY[g, n] == 0)]
TIMES = np.unique(np.concatenate(list(PICKED.values())))
N_T = len(TIMES)
STEPS = _hours(TIMES).astype('f8')

# Store rows per TIMES entry, -1 where that store isn't stepped; truth is read wherever any emulator of its grid is.
IDX = {key: np.where(np.isin(TIMES, t), pd.Index(STORES[key[0]][key[1]].time.values).get_indexer(TIMES), -1)
       for key, t in PICKED.items()}
ANY = {g: np.any([IDX[g, n] >= 0 for n in EMU_NAMES], axis=0) for g in STORES}
TRUTH_IDX = {'coarse': pd.Index(ct.time.values).get_indexer(TIMES), 'sharp': pd.Index(ft.time.values).get_indexer(TIMES)}
for g, path in (('coarse', COARSE_CACHE), ('sharp', args.sharp_truth)):
    if (TRUTH_IDX[g][ANY[g]] < 0).any():
        raise ValueError(f"{path} lacks {g} steps {TIMES[ANY[g] & (TRUTH_IDX[g] < 0)]}")
AVAIL = {(s, n): ANY[g] if n in ('llc', 'persistence') else IDX[g, n] >= 0
         for s, g in SCOPE_GRID.items() for n in ['llc', 'persistence', *EMU_NAMES]}

# Unnormalization: (mean, std, wet) per CH row; coarse land is 0 in its cache, the face cache's is NaN.
_native = zarr.open_group(ct.attrs['native_store'], mode='r')
_lvl = {v: [0] if v == 'Eta' else LEVELS_TO_READ for v in VARS}
COARSE_NORM = (np.concatenate([np.ravel(_native[f'{v}_mean'][:])[_lvl[v]] for v in VARS])[:, None],
               np.concatenate([np.ravel(_native[f'{v}_std'][:])[_lvl[v]] for v in VARS])[:, None],
               ct.wet.isel(channel=CH).values.astype(bool).transpose(1, 0, 2, 3).reshape(len(CH), -1))
SHARP_NORM = (ft.prognostic_mean.values[CH].astype('f8')[:, None], ft.prognostic_std.values[CH].astype('f8')[:, None],
              None)
COARSE_W = ct.rA.values.astype('f8').ravel()
SHARP_W = ft.rA.isel(y=FY, x=FX).values.astype('f8')


def _coarse_raw(ti):
    """The coarse truth's normalized CH at cache row ti, (C, face * y * x)."""
    a = np.asarray(ct.prognostic.isel(time=ti, channel=CH).values)
    return a.transpose(1, 0, 2, 3).reshape(len(CH), -1)


def _sharp_raw(ti, rows=slice(None)):
    """The sharp truth's normalized CH at cache row ti over domain rows `rows`, (C, ny, nx)."""
    y = slice(FY.start + (rows.start or 0), FY.start + (NJ if rows.stop is None else rows.stop))
    return np.asarray(ft.prognostic.isel(time=ti, prognostic_channel=CH, y=y, x=FX).values)


COARSE_IC = _coarse_raw(int(pd.Index(ct.time.values).get_loc(IC)))  # persistence: the IC held fixed (read before forking)
SHARP_IC = _sharp_raw(int(pd.Index(ft.time.values).get_loc(IC)))


def _phys(raw, var, mean, std, wet):
    """One var's physical (nk, ncell) float64 from a cache's normalized CH rows."""
    s = CH_SLICE[var]
    x = raw[s].astype('f8') * std[s] + mean[s]
    return x if wet is None else np.where(wet[s], x, np.nan)


def _emu(ds, ti, var, **space):
    """One var of an emulator store at its row ti, (nk, ncell) float64."""
    da = ds[var].isel(time=ti, **space)
    if var != 'Eta':
        da = da.sel(k=LEVELS_TO_READ)
    a = np.asarray(da.values, dtype='f8')
    return a.reshape(1 if var == 'Eta' else a.shape[0], -1)


def _dmean(a, pos, ref=None):
    """Depth-average over positions `pos` ignoring NaN levels; with `ref`, of the squared error against it (squared
    per level so opposite-sign errors cannot cancel). One level at a time, so temporaries stay one level big."""
    total = count = 0
    for k in pos:
        x = a[k] if ref is None else (a[k] - ref[k]) ** 2
        ok = np.isfinite(x)
        total, count = total + np.where(ok, x, 0.0), count + ok
    with np.errstate(invalid='ignore', divide='ignore'):
        return total / count


def _sums(x, ok, w):
    """[sum w, sum w x, sum w x^2] over cells valid in `ok` and x."""
    ok = ok & np.isfinite(x)
    w, x = np.where(ok, w, 0.0), np.where(ok, x, 0.0)
    wx = w * x
    return np.array([w.sum(), wx.sum(), (wx * x).sum()])


def _stat(s, stat):
    """mean / std / rmse from _sums; rmse's x is the squared error."""
    with np.errstate(invalid='ignore', divide='ignore'):
        m = s[..., 1] / s[..., 0]
        if stat == 'mean':
            return m
        if stat == 'rmse':
            return np.sqrt(m)
        return np.sqrt(np.maximum(s[..., 2] / s[..., 0] - m * m, 0.0))


def task(job):
    """Sums of step i over the coarse globe (band None), or over sharp rows [band, band + BAND)."""
    i, band = job
    if band is None:
        truth_raw, ic_raw, norm = _coarse_raw(TRUTH_IDX['coarse'][i]), COARSE_IC, COARSE_NORM
        emus = {n: (COARSE_EMU[n], IDX['coarse', n][i], {}) for n in EMU_NAMES if IDX['coarse', n][i] >= 0}
        scopes = [('coarse-out', ~INSIDE, COARSE_W), ('coarse-in', INSIDE, COARSE_W)]
    else:
        rows = slice(band, min(band + BAND, NJ))
        truth_raw = _sharp_raw(TRUTH_IDX['sharp'][i], rows).reshape(len(CH), -1)
        ic_raw, norm = SHARP_IC[:, rows].reshape(len(CH), -1), SHARP_NORM
        emus = {n: (SHARP_EMU[n], IDX['sharp', n][i], {'lat': rows}) for n in EMU_NAMES if IDX['sharp', n][i] >= 0}
        scopes = [('sharp', True, SHARP_W[rows].ravel())]
    out = {}
    for var in VARS:
        truth, ic = _phys(truth_raw, var, *norm), _phys(ic_raw, var, *norm)
        xs = {n: _emu(ds, ti, var, **space) for n, (ds, ti, space) in emus.items()}
        for row, pos in (ROW_POS if var != 'Eta' else {ROW_KEYS[0]: [0]}).items():
            tbar = _dmean(truth, pos)
            fields = {'llc': tbar, **{n: _dmean(x, pos) for n, x in xs.items()}}
            errs = {'persistence': _dmean(ic, pos, truth), **{n: _dmean(x, pos, truth) for n, x in xs.items()}}
            for scope, cells, w in scopes:
                ok = cells & np.isfinite(tbar)  # truth's ocean, shared by every source
                for n, f in fields.items():
                    out[('field', var, row, scope, n)] = _sums(f, ok, w)
                for n, e in errs.items():
                    out[('err', var, row, scope, n)] = _sums(e, ok, w)
    return i, out


# ============== PLOTTER ==============
ROW_LABELS = {'column': f"column mean\nk[{DEPTH_AVERAGE.start}:{DEPTH_AVERAGE.stop}:{DEPTH_AVERAGE.step}]",
              'surface': 'surface\nk = 0', **{k: f'k = {k}' for k in DEPTH_LEVELS}}
_PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']
LINE_STYLE = {'llc': dict(color='k', lw=2.0, zorder=5), 'persistence': dict(color='k', lw=1.5, zorder=4),
              **{k: dict(color=_PALETTE[i % len(_PALETTE)], lw=1.3) for i, k in enumerate(EMU_NAMES)}}
SCOPE_LS = {'sharp': '-', 'coarse-in': '--', 'coarse-out': ':'}
LABEL = {'llc': 'LLC4320 (truth)', 'persistence': 'persistence (IC held)', **{k: k for k in EMU_NAMES}}
FIGURES = {
    'mean': ('Domain mean (area-weighted by rA) vs AR step', 'AR-stability_domain-mean', 'mean [{u}]'),
    'std': ('Domain standard deviation (area-weighted) vs AR step', 'AR-stability_domain-std', 'std [{u}]'),
    'rmse': ('RMSE against LLC4320 truth (area-weighted, per-cell) vs AR step', 'AR-stability_rmse', 'RMSE [{u}]'),
}
TAG = f"N = face {FACE} j[{J0}:{J1}) i[{I0}:{I1})"
OUT_DIR = args.out_dir or (f"{OUT_ROOT}/{'+'.join(EMU_NAMES)}_f{FACE}"
                           f"_h{args.time_range[0]}-{args.time_range[1]}s{args.stepping}h")


def plot_stat(sums, stat, title, fname, ylabel_fmt):
    """Time series of `stat`: a row x var grid (FULL), or vars stacked vertically (COLUMN_MEAN_ONLY / SURFACE)."""
    names = ['persistence', *EMU_NAMES] if stat == 'rmse' else ['llc', *EMU_NAMES]
    kind = 'err' if stat == 'rmse' else 'field'
    stacked = MODE != 'FULL'
    if stacked:
        panels = [(c, 0, var, ROW_KEYS[0]) for c, var in enumerate(VARS)]
        nr, nc, size = len(VARS), 1, (7.0, 2.4 * len(VARS))
    else:  # Eta only fills the first row
        panels = [(r, c, var, row) for r, row in enumerate(ROW_KEYS) for c, var in enumerate(VARS)
                  if r == 0 or var != 'Eta']
        nr, nc, size = len(ROW_KEYS), len(VARS), (4.2 * len(VARS), 2.7 * len(ROW_KEYS))
    last_row = {}
    for gr, gc, _, _ in panels:
        last_row[gc] = max(last_row.get(gc, -1), gr)

    fig, axes = plt.subplots(nr, nc, figsize=size, squeeze=False, sharex=True)
    for ax in axes.flat:
        ax.axis('off')
    for gr, gc, var, row in panels:
        ax = axes[gr][gc]
        ax.axis('on')
        for name in names:
            for scope in SCOPES:
                key, m = (kind, var, row, scope, name), AVAIL[(scope, name)]
                if key in sums:  # a store with no step in TIME_RANGE has none
                    ax.plot(STEPS[m], _stat(sums[key], stat)[m], ls=SCOPE_LS[scope], **LINE_STYLE[name])
        ylab = ylabel_fmt.format(u=UNITS[var])
        if stacked:
            ax.set_ylabel(f"{var}\n{ylab}", fontsize=9)
        else:
            if gr == 0:
                ax.set_title(var, fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{ROW_LABELS[row]}\n{ylab}" if gc == 0 else ylab, fontsize=9)
        if gr == last_row[gc]:
            ax.set_xlabel('AR step (hours since rollout start)', fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
    handles = ([Line2D([], [], label=LABEL[n], **LINE_STYLE[n]) for n in names]
               + [Line2D([], [], color='0.4', lw=1.3, ls=SCOPE_LS[s], label=label) for s, label in SCOPES.items()])
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=10)
    if stacked:
        title += f" — {ROW_LABELS[ROW_KEYS[0]].replace(chr(10), ' ')}"
        fname += '_column-mean' if MODE == 'COLUMN_MEAN_ONLY' else '_surface'
    fig.suptitle(f'{title} ({TAG})', fontsize=14, y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = f"{OUT_DIR}/{fname}.png"
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    log(f"saved {out}")


def step_line(sums, i):
    """The first var and row's headline stat at step i, per emulator: coarse outside N / inside N / sharp."""
    stat = 'rmse' if 'rmse' in STATS else 'mean'
    kind, var, row = 'err' if stat == 'rmse' else 'field', VARS[0], ROW_KEYS[0]
    def val(s, n):
        key = (kind, var, row, s, n)
        return f"{_stat(sums[key][i], stat):.4g}" if key in sums and AVAIL[s, n][i] else "-"
    parts = [f"{n} " + "/".join(val(s, n) for s in SCOPES) for n in EMU_NAMES]
    return f"{var} {row} {stat} out/in/sharp: " + "  ".join(parts)


def main():
    jobs = [(i, b) for i in range(N_T)
            for b in ([None] if ANY['coarse'][i] else []) + (list(range(0, NJ, BAND)) if ANY['sharp'][i] else [])]
    left = Counter(i for i, _ in jobs)
    n_workers = max(1, min(N_WORKERS, len(jobs)))
    log(f"{N_T} steps {TIMES[0]} -> {TIMES[-1]} (+{STEPS[0]:.0f} to +{STEPS[-1]:.0f} h from IC {IC}); "
        f"{TAG}; {len(jobs)} tasks (sharp bands of {BAND} rows) on {n_workers} workers")
    for n in EMU_NAMES:
        log(f"  {n}: " + ", ".join(f"{g} every {EVERY[g, n]} h ({int((IDX[g, n] >= 0).sum())} steps)" for g in STORES))
    log(f"  vars={VARS} stats={STATS} mode={MODE} levels={LEVELS_TO_READ}")
    sums = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('fork')) as pool:
        for i, out in pool.map(task, jobs):
            for key, s in out.items():
                sums.setdefault(key, np.zeros((N_T, 3)))[i] += s
            left[i] -= 1
            if not left[i]:
                log(f"step {i + 1:4d}/{N_T} {str(TIMES[i])[:13]} (+{STEPS[i]:.0f} h)  {step_line(sums, i)}")

    log("valid fraction of truth's ocean (1.000 = no NaN/inf anywhere):")
    for var in VARS:
        for scope in SCOPES:
            for n in EMU_NAMES:
                key, m = ('field', var, ROW_KEYS[0], scope, n), AVAIL[(scope, n)]
                if key not in sums:
                    continue
                v = np.nanmin(sums[key][m, 0] / sums[('field', var, ROW_KEYS[0], scope, 'llc')][m, 0])
                log(f"    {var:6s} {scope:10s} {n:16s} min {v:.4f}" + ("" if v > 0.9999 else "   <-- NaN/inf present"))

    os.makedirs(OUT_DIR, exist_ok=True)
    series = {'|'.join([stat, *map(str, key[1:])]): _stat(s, stat)
              for key, s in sums.items() for stat in (['rmse'] if key[0] == 'err' else ['mean', 'std'])}
    np.savez(f"{OUT_DIR}/stats_{MODE}.npz", steps=STEPS, times=TIMES, **series,
             **{f"avail|{s}|{n}": m for (s, n), m in AVAIL.items()})
    for s in STATS:
        plot_stat(sums, s, *FIGURES[s])
    log("done")


if __name__ == '__main__':
    main()
