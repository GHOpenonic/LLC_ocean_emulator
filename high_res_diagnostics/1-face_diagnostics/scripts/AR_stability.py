"""AR-stability: domain mean, std and RMSE vs LLC truth, per AR step."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import concurrent.futures as cf
import time as _time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diag_common import base_parser, emulator_configs_from, load_patches, mem_budget_bytes, n_cpus, normalize_times, out_dir

p = base_parser(__doc__)
p.add_argument('--depth-average', type=int, nargs=3, default=[0, 51, 5], metavar=('START', 'STOP', 'STEP'),
               help='row 0: depth-average over these levels')
p.add_argument('--depth-levels', type=int, nargs='+', default=[0, 10, 25, 50], help='one row per level')
args = p.parse_args()

llc_patch, emulator_patches, emulator_info, all_patches = load_patches(
    emulator_configs_from(args), slice(*args.i_extent), slice(*args.j_extent),
    args.time_range, args.stepping)

DEPTH_AVERAGE = slice(*args.depth_average)
DEPTH_LEVELS = args.depth_levels
PLOT_VARS = args.vars
READ_WORKERS = min(32, n_cpus())  # NFS throughput scales with concurrent readers

UNITS = {'Theta': '°C', 'Salt': 'psu', 'U': 'm s$^{-1}$', 'V': 'm s$^{-1}$', 'Eta': 'm'}
OUT_DIR = out_dir(args, 'AR-stability')

ROW_LABELS = [f"column mean\nk[{DEPTH_AVERAGE.start}:{DEPTH_AVERAGE.stop}"
              + (f":{DEPTH_AVERAGE.step}]" if DEPTH_AVERAGE.step else "]")] + \
             [f"k = {k}" for k in DEPTH_LEVELS]
N_ROWS, N_COLS = len(ROW_LABELS), len(PLOT_VARS)
SOURCES = [('llc', llc_patch)] + [(k, emulator_patches[k]) for _, k in emulator_info]
_PATCH = dict(SOURCES)

# Read only the union of levels any row needs, once per time block; a lazy dask graph kept whole columns resident.
_AVG = list(range(*DEPTH_AVERAGE.indices(51)))
LEVELS_TO_READ = sorted(set(_AVG) | set(DEPTH_LEVELS))
_POS = {k: i for i, k in enumerate(LEVELS_TO_READ)}
_AVG_POS = [_POS[k] for k in _AVG]

_rA = np.asarray(llc_patch['rA'].values, dtype='f8')
AREA_W = _rA / _rA.sum()

_block_bytes = 8 * len(LEVELS_TO_READ) * _rA.size * len(SOURCES) * 4  # x4 for per-row temporaries
TIME_BLOCK = int(max(1, min(48, mem_budget_bytes() // _block_bytes)))
print(f"TIME_BLOCK = {TIME_BLOCK}, READ_WORKERS = {READ_WORKERS}")


# Stats use finite cells only (land and blow-ups are NaN); truth's mask is shared by every source.
def _wstats(x, ok=None):
    """Area-weighted (mean, std, weight sum) over valid cells. x: (nb, ny, nx)."""
    ok = np.isfinite(x) if ok is None else (ok & np.isfinite(x))
    w = np.where(ok, AREA_W, 0.0)
    wsum = w.sum(axis=(-2, -1))
    xv = np.where(ok, x, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(wsum > 0, (xv * w).sum(axis=(-2, -1)) / wsum, np.nan)
        dev = np.where(ok, x - mean[:, None, None], 0.0)
        var = np.where(wsum > 0, (dev ** 2 * w).sum(axis=(-2, -1)) / wsum, np.nan)
    return mean, np.sqrt(var), wsum


def _rms(d2):
    """sqrt of the area-weighted mean of a per-cell squared error."""
    ok = np.isfinite(d2)
    w = np.where(ok, AREA_W, 0.0)
    wsum = w.sum(axis=(-2, -1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(
            wsum > 0,
            np.sqrt((np.where(ok, d2, 0.0) * w).sum(axis=(-2, -1)) / wsum),
            np.nan,
        )


def _dmean(a, pos):
    """Depth-average over `pos`, ignoring NaN levels."""
    if pos is None:
        return a
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(a[:, pos], axis=1)


def _read(job):
    name, var, tsl = job
    dv = _PATCH[name][var]
    if 'k' not in dv.dims:
        return job, np.asarray(dv.isel(time=tsl).values, dtype='f8')
    return job, np.asarray(dv.isel(time=tsl, k=LEVELS_TO_READ).values, dtype='f8')


N_T = llc_patch.sizes['time']
# Hours since rollout start, from the time coordinate so stepping is labelled correctly.
_tt = normalize_times(llc_patch.time.values)
STEPS = ((_tt - _tt[0]).total_seconds() / 3600.0).to_numpy()
stats = {}


def _put(key, i0, vals):
    stats.setdefault(key, np.full(N_T, np.nan))[i0:i0 + len(vals)] = vals


_t0 = _time.time()
_pool = cf.ThreadPoolExecutor(max_workers=READ_WORKERS)
try:
    for var in PLOT_VARS:
        is_2d = 'k' not in llc_patch[var].dims
        rows = [(0, None)] if is_2d else \
               [(0, _AVG_POS)] + [(r + 1, [_POS[k]]) for r, k in enumerate(DEPTH_LEVELS)]
        for i0 in range(0, N_T, TIME_BLOCK):
            tsl = slice(i0, min(i0 + TIME_BLOCK, N_T))
            jobs = [(n, var, tsl) for n, _ in SOURCES]
            blocks = {j[0]: b for j, b in _pool.map(_read, jobs)}
            truth = blocks['llc']
            for row, pos in rows:
                valid = np.isfinite(_dmean(truth, pos))
                for name, _ in SOURCES:
                    x = blocks[name]
                    xbar = _dmean(x, pos)
                    mean, std, wsum = _wstats(xbar, ok=valid)
                    _put(('mean', var, row, name), i0, mean)
                    _put(('std', var, row, name), i0, std)
                    _put(('valid_frac', var, row, name), i0, wsum)
                    if name != 'llc':
                        # Square per level before depth-averaging so opposite-sign errors cannot cancel.
                        if pos is None:
                            d2 = (x - truth) ** 2
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', RuntimeWarning)
                                d2 = np.nanmean((x[:, pos] - truth[:, pos]) ** 2, axis=1)
                        _put(('rmse', var, row, name), i0, _rms(d2))
            del blocks, truth
            print(f"  {var:6s} t[{tsl.start}:{tsl.stop}] of {N_T}  ({_time.time() - _t0:5.0f}s)", flush=True)
finally:
    _pool.shutdown()

print(f"\ndone: {len(stats)} series of length {N_T} in {(_time.time() - _t0) / 60:.1f} min")
print("usable domain fraction (1.000 = no NaN/inf anywhere):")
for var in PLOT_VARS:
    for name, _ in SOURCES:
        k = ('valid_frac', var, 0, name)
        if k in stats:
            v = stats[k]
            flag = "" if np.nanmin(v) > 0.9999 else "   <-- NaN/inf present"
            print(f"    {var:6s} {name:12s} min {np.nanmin(v):.4f}{flag}")

# ============== PLOTTING ==============
LINE_STYLE = {'llc': dict(color='k', lw=2.0, zorder=5)}
_PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']
for _i, (_n, _k) in enumerate(emulator_info):
    LINE_STYLE[_k] = dict(color=_PALETTE[_i % len(_PALETTE)], lw=1.3)
LABEL = {'llc': 'LLC4320 (truth)'}
LABEL.update({k: n for n, k in emulator_info})


def plot_stat(stat, title, fname, ylabel_fmt):
    """One k x var grid of time series for `stat` in {'mean','std','rmse'}."""
    last_row = {}  # lowest row with content per column (Eta stops at row 0)
    for c, var in enumerate(PLOT_VARS):
        rows = [r for r in range(N_ROWS)
                if any((stat, var, r, n) in stats
                       for n, _ in SOURCES if not (stat == 'rmse' and n == 'llc'))]
        last_row[c] = max(rows) if rows else -1

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4.2 * N_COLS, 2.7 * N_ROWS),
                             squeeze=False, sharex=True)
    for r in range(N_ROWS):
        for c, var in enumerate(PLOT_VARS):
            ax = axes[r][c]
            drawn = False
            for name, _ in SOURCES:
                if stat == 'rmse' and name == 'llc':
                    continue
                key = (stat, var, r, name)
                if key not in stats:
                    continue
                ax.plot(STEPS, stats[key], label=LABEL[name], **LINE_STYLE[name])
                drawn = True
            if not drawn:
                ax.axis('off')
                continue
            if r == 0:
                ax.set_title(var, fontsize=12, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f"{ROW_LABELS[r]}\n{ylabel_fmt.format(u=UNITS.get(var, ''))}", fontsize=9)
            else:
                ax.set_ylabel(ylabel_fmt.format(u=UNITS.get(var, '')), fontsize=9)
            if r == last_row[c]:
                ax.set_xlabel('AR step (hours since rollout start)', fontsize=9)
            ax.grid(alpha=0.3, lw=0.5)
            ax.tick_params(labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=14, y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = f"{OUT_DIR}/{fname}.png"
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}")


plot_stat('mean', 'Domain-mean (area-weighted by rA) vs AR step', 'AR-stability_domain-mean', 'mean [{u}]')
plot_stat('std', 'Domain standard deviation (area-weighted) vs AR step', 'AR-stability_domain-std', 'std [{u}]')
plot_stat('rmse', 'RMSE against LLC4320 truth (area-weighted, per-cell) vs AR step', 'AR-stability_rmse', 'RMSE [{u}]')
