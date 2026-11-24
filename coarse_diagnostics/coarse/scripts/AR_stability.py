"""AR stability of coarse emulators vs LLC truth: area-weighted mean / std / RMSE time series per var and depth row; time blocks run in parallel."""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import multiprocessing as mp
import sys
import time as _time
import warnings
from concurrent.futures import ProcessPoolExecutor

import dask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics')
from predictions_io import open_predictions

CACHE = '/orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_cache_0p33deg.zarr'
OUT_ROOT = '/home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/figs/AR-stability'
FIELDS = ['U', 'V', 'Theta', 'Salt']  # cache channel blocks of 51 levels, in this order; Eta is channel 204
UNITS = {'Theta': '°C', 'Salt': 'psu', 'U': 'm s$^{-1}$', 'V': 'm s$^{-1}$', 'Eta': 'm'}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--emulator', action='append', required=True, metavar='NAME=PATH', help='repeat per emulator')
p.add_argument('--vars', nargs='+', choices=list(UNITS), default=['Theta', 'Salt', 'U', 'V', 'Eta'])
p.add_argument('--stats', nargs='+', choices=['mean', 'std', 'rmse'], default=['rmse'])
p.add_argument('--mode', choices=['FULL', 'COLUMN_MEAN_ONLY', 'SURFACE'], default='FULL',
               help='FULL (k x var grid) | COLUMN_MEAN_ONLY | SURFACE (k=0); the last two stack vars vertically')
p.add_argument('--depth-average', type=int, nargs=3, default=[0, 51, 5], metavar=('START', 'STOP', 'STEP'),
               help='column-mean levels')
p.add_argument('--depth-levels', type=int, nargs='*', default=[10, 25, 50], help='extra single-level rows in FULL mode')
p.add_argument('--time-block', type=int, default=4, help='time steps per worker task')
p.add_argument('--face', default='1', help="0-12 for one face, 'all' for the globe")
p.add_argument('--x-extent', type=int, nargs=2, default=[0, 270], metavar=('START', 'STOP'))
p.add_argument('--y-extent', type=int, nargs=2, default=[0, 270], metavar=('START', 'STOP'))
p.add_argument('--time-range', type=int, nargs=2, default=[0, 744], metavar=('START', 'END'),
               help='inclusive indices into the times truth and emulators share')
p.add_argument('--stepping', type=int, default=1)
p.add_argument('--workers', type=int, default=None)
p.add_argument('--out-dir', default=None)
args = p.parse_args()

N_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
N_WORKERS = args.workers or int(os.environ.get('AR_N_WORKERS', N_CPUS))
dask.config.set(scheduler='synchronous')  # parallelism comes from the process pool, one time block per task
GLOBE = args.face == 'all'
SPACE = dict(face=slice(0, 13) if GLOBE else int(args.face), x=slice(*args.x_extent), y=slice(*args.y_extent))
SP_DIMS = ['face', 'y', 'x'] if GLOBE else ['y', 'x']
EMULATORS = [spec.split('=', 1) for spec in args.emulator]
VARS, STATS, MODE = args.vars, args.stats, args.mode
DEPTH_AVERAGE = slice(*args.depth_average)
DEPTH_LEVELS = args.depth_levels
_t0 = _time.time()


def log(msg):
    print(f"[{_time.time() - _t0:6.0f}s] {msg}", flush=True)


ROW_KEYS = {'FULL': ['column', 'surface', *DEPTH_LEVELS], 'COLUMN_MEAN_ONLY': ['column'], 'SURFACE': ['surface']}[MODE]
_ALL_ROWS = {'column': list(range(*DEPTH_AVERAGE.indices(51))), 'surface': [0], **{k: [k] for k in DEPTH_LEVELS}}
ROWS = {r: _ALL_ROWS[r] for r in ROW_KEYS}
LEVELS_TO_READ = sorted({k for ks in ROWS.values() for k in ks})
ROW_POS = {r: [LEVELS_TO_READ.index(k) for k in ks] for r, ks in ROWS.items()}


# ============== LOAD (lazy; workers read their own time blocks) ==============
def open_truth():
    """Physical fields at LEVELS_TO_READ from the normalized coarse cache, land NaN, faces 7-12 north-up."""
    cache = xr.open_zarr(CACHE, consolidated=False).drop_vars('boundary')
    native = zarr.open_group(cache.attrs['native_store'], mode='r')
    fields = {}
    for v in VARS:
        if v == 'Eta':
            ch, lvl = [204], [0]
        else:
            ch, lvl = [51 * FIELDS.index(v) + k for k in LEVELS_TO_READ], LEVELS_TO_READ
        mean = xr.DataArray(np.ravel(native[f'{v}_mean'][:])[lvl], dims='channel')
        std = xr.DataArray(np.ravel(native[f'{v}_std'][:])[lvl], dims='channel')
        da = (cache.prognostic.isel(channel=ch) * std + mean).where(cache.wet.isel(channel=ch).astype(bool))
        fields[v] = da.isel(channel=0, drop=True) if v == 'Eta' else da.rename(channel='k')
    return xr.Dataset(fields).assign_coords(face=np.arange(13)), cache.rA.assign_coords(face=np.arange(13))


def open_emulator(path):
    ds = open_predictions(path)[VARS]
    ds = xr.Dataset({v: ds[v].isel(k=LEVELS_TO_READ) if 'k' in ds[v].dims else ds[v] for v in VARS})
    return ds.rename(lat='y', lon='x').drop_vars(['k', 'y', 'x'], errors='ignore')


truth, rA = open_truth()
raw = {'llc': truth, **{name: open_emulator(path) for name, path in EMULATORS}}
common = truth.time.values
for name, _ in EMULATORS:
    common = np.intersect1d(common, raw[name].time.values)
TIMES = common[args.time_range[0]:args.time_range[1] + 1:args.stepping]
SOURCES = {k: ds.sel(time=TIMES).isel(**SPACE) for k, ds in raw.items()}
N_T = len(TIMES)
STEPS = (TIMES - TIMES[0]) / np.timedelta64(1, 'h')  # hours since rollout start
_rA = rA.isel(**SPACE).values.astype('f8').ravel()
AREA_W = _rA / _rA.sum()


def _read(ds, tsl):
    """All vars for one time block as float64 (time, [k,] ncell)."""
    out = {}
    for v in VARS:
        da = ds[v].isel(time=tsl)
        da = da.transpose('time', *(['k'] if 'k' in da.dims else []), *SP_DIMS)
        out[v] = np.asarray(da.values, dtype='f8').reshape(*da.shape[:-len(SP_DIMS)], -1)
    return out


def _wstats(x, ok):
    """Area-weighted (mean, std, weight sum) over cells valid in `ok` and x. x: (nb, ncell)."""
    ok = ok & np.isfinite(x)
    w = np.where(ok, AREA_W, 0.0)
    wsum = w.sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = (np.where(ok, x, 0.0) * w).sum(-1) / wsum
        var = (np.where(ok, x - mean[:, None], 0.0) ** 2 * w).sum(-1) / wsum
    return mean, np.sqrt(var), wsum


def _rms(d2):
    """sqrt of the area-weighted mean of a per-cell squared error."""
    ok = np.isfinite(d2)
    w = np.where(ok, AREA_W, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt((np.where(ok, d2, 0.0) * w).sum(-1) / w.sum(-1))


def _dmean(a, pos):
    """Depth-average over `pos` (None = 2D var), ignoring NaN levels."""
    if pos is None:
        return a
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(a[:, pos], axis=1)


def _sq_err(x, truth, pos):
    """Per-cell squared error, squared per level before depth-averaging so opposite-sign errors cannot cancel."""
    return _dmean((x - truth) ** 2, pos)


TRUTH0 = _read(SOURCES['llc'], slice(0, 1))  # persistence: hold the first truth state fixed (read before forking)


def block_stats(i0):
    """Every requested series over time steps [i0, i0 + time_block)."""
    tsl = slice(i0, min(i0 + args.time_block, N_T))
    blocks = {name: _read(ds, tsl) for name, ds in SOURCES.items()}
    out = {}
    for var in VARS:
        truth = blocks['llc'][var]
        rows = {ROW_KEYS[0]: None} if truth.ndim == 2 else ROW_POS
        for row, pos in rows.items():
            valid = np.isfinite(_dmean(truth, pos))
            for name in SOURCES:
                x = blocks[name][var]
                mean, std, wsum = _wstats(_dmean(x, pos), valid)
                if name == 'llc':
                    truth_wsum = wsum
                out[('valid_frac', var, row, name)] = wsum / truth_wsum
                out[('mean', var, row, name)] = mean
                out[('std', var, row, name)] = std
                if 'rmse' in STATS and name != 'llc':
                    out[('rmse', var, row, name)] = _rms(_sq_err(x, truth, pos))
            if 'rmse' in STATS:
                out[('rmse', var, row, 'persistence')] = _rms(_sq_err(TRUTH0[var], truth, pos))
    return i0, out


# ============== PLOTTER ==============
emulator_keys = [name for name, _ in EMULATORS]
ROW_LABELS = {'column': f"column mean\nk[{DEPTH_AVERAGE.start}:{DEPTH_AVERAGE.stop}:{DEPTH_AVERAGE.step}]",
              'surface': 'surface\nk = 0', **{k: f'k = {k}' for k in DEPTH_LEVELS}}
_PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']
LINE_STYLE = {'llc': dict(color='k', lw=2.0, zorder=5), 'persistence': dict(color='k', lw=1.5, zorder=4),
              **{k: dict(color=_PALETTE[i % len(_PALETTE)], lw=1.3) for i, k in enumerate(emulator_keys)}}
LABEL = {'llc': 'LLC4320 coarse (truth)', 'persistence': 'persistence', **{k: k for k in emulator_keys}}
FIGURES = {
    'mean': ('Domain-mean (area-weighted by rA) vs AR step', 'AR-stability_domain-mean', 'mean [{u}]'),
    'std': ('Domain standard deviation (area-weighted) vs AR step', 'AR-stability_domain-std', 'std [{u}]'),
    'rmse': ('RMSE against LLC4320 coarse truth (area-weighted, per-cell) vs AR step', 'AR-stability_rmse', 'RMSE [{u}]'),
}
TAG = 'globe' if GLOBE else f'face{args.face}'
OUT_DIR = args.out_dir or (f"{OUT_ROOT}/{TAG}/{'+'.join(emulator_keys)}_x{args.x_extent[0]}-{args.x_extent[1]}"
                           f"_y{args.y_extent[0]}-{args.y_extent[1]}_t{args.time_range[0]}-{args.time_range[1]}s{args.stepping}")


def plot_stat(stats, stat, title, fname, ylabel_fmt):
    """Time series of `stat`: a row x var grid (FULL), or vars stacked vertically (COLUMN_MEAN_ONLY / SURFACE)."""
    names = ['persistence', *emulator_keys] if stat == 'rmse' else list(SOURCES)
    stacked = MODE != 'FULL'
    if stacked:
        panels = [(c, 0, var, ROW_KEYS[0]) for c, var in enumerate(VARS)]
        nr, nc, size = len(VARS), 1, (7.0, 2.4 * len(VARS))
    else:  # 2D vars (Eta) only fill the first row
        panels = [(r, c, var, row) for r, row in enumerate(ROW_KEYS) for c, var in enumerate(VARS)
                  if r == 0 or 'k' in truth[var].dims]
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
            ax.plot(STEPS, stats[(stat, var, row, name)], label=LABEL[name], **LINE_STYLE[name])
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
    handles, labels = axes[panels[0][0]][panels[0][1]].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)
    if stacked:
        title += f" — {ROW_LABELS[ROW_KEYS[0]].replace(chr(10), ' ')}"
        fname += '_column-mean' if MODE == 'COLUMN_MEAN_ONLY' else '_surface'
    fig.suptitle(f'{title} ({TAG})', fontsize=14, y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = f"{OUT_DIR}/{fname}.png"
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    log(f"saved {out}")


def main():
    starts = list(range(0, N_T, args.time_block))
    n_workers = max(1, min(N_WORKERS, len(starts)))
    log(f"{N_T} times, {len(starts)} blocks of {args.time_block} on {n_workers} workers; vars={VARS} stats={STATS} "
        f"mode={MODE} levels={LEVELS_TO_READ} face={args.face}")
    stats = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('fork')) as pool:
        for n, (i0, out) in enumerate(pool.map(block_stats, starts), 1):
            for key, vals in out.items():
                stats.setdefault(key, np.full(N_T, np.nan))[i0:i0 + len(vals)] = vals
            if n % max(1, len(starts) // 20) == 0 or n == len(starts):
                log(f"  blocks done: {n}/{len(starts)}")

    log("valid fraction of truth's ocean (1.000 = no NaN/inf anywhere):")
    for var in VARS:
        for name in emulator_keys:
            v = stats[('valid_frac', var, ROW_KEYS[0], name)]
            log(f"    {var:6s} {name:12s} min {np.nanmin(v):.4f}" + ("" if np.nanmin(v) > 0.9999 else "   <-- NaN/inf present"))

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(f"{OUT_DIR}/stats_{MODE}.npz", steps=STEPS, **{'|'.join(map(str, k)): v for k, v in stats.items()})
    for s in STATS:
        plot_stat(stats, s, *FIGURES[s])
    log("done")


if __name__ == '__main__':
    main()
