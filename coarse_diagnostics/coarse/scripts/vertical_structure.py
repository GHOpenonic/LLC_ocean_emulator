"""Depth x lat/lon transects of coarse LLC truth vs emulators: one figure per var, rows = times, cols = sources."""
import argparse
import os
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics')
from predictions_io import open_predictions

CACHE = '/orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_cache_0p33deg.zarr'
OUT_ROOT = '/home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/figs/vertical_structure'
FIELDS = ['U', 'V', 'Theta', 'Salt']  # cache channel blocks of 51 levels, in this order
CMAPS = {'Theta': 'Spectral_r', 'Salt': 'viridis', 'U': 'bwr', 'V': 'bwr'}
UNITS = {'Theta': '°C', 'Salt': 'PSU', 'U': 'm/s', 'V': 'm/s'}
# LLC4320 cell-centre depths [m] (ocean_emulators.constants.DEPTH_LEVELS)
DEPTH_LEVELS = np.array([0.5, 1.57, 2.79, 4.185, 5.78, 7.595, 9.66, 12.01, 14.68, 17.705, 21.125, 24.99, 29.345,
                         34.24, 39.725, 45.855, 52.69, 60.28, 68.685, 77.965, 88.175, 99.37, 111.6, 124.915, 139.365,
                         154.99, 171.825, 189.9, 209.235, 229.855, 251.77, 274.985, 299.505, 325.32, 352.42, 380.79,
                         410.41, 441.255, 473.305, 506.54, 540.935, 576.465, 613.11, 650.855, 689.685, 729.595,
                         770.585, 812.66, 855.835, 900.135, 945.595])

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--emulator', action='append', required=True, metavar='NAME=PATH', help='repeat per emulator')
p.add_argument('--vars', nargs='+', choices=list(CMAPS), default=['Theta', 'Salt', 'U', 'V'])
p.add_argument('--k', type=int, nargs=2, default=[0, 51], metavar=('START', 'STOP'))
p.add_argument('--lat', type=float, nargs='+', required=True, help='START STOP for the transect axis, or one constant')
p.add_argument('--lon', type=float, nargs='+', required=True, help='START STOP for the transect axis, or one constant')
p.add_argument('--res', type=float, default=1 / 3, help='transect sample spacing [deg]')
p.add_argument('--time-range', type=int, nargs=2, default=[0, 731], metavar=('START', 'END'),
               help='inclusive indices into the times truth and emulators share')
p.add_argument('--stepping', type=int, default=93)
p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
args = p.parse_args()

if (len(args.lat) == 2) == (len(args.lon) == 2):
    sys.exit('give exactly one of --lat / --lon as START STOP; the other is the constant')
EMULATORS = [spec.split('=', 1) for spec in args.emulator]
KS = slice(*args.k)
_t0 = _time.time()


def log(msg):
    print(f"[{_time.time() - _t0:6.0f}s] {msg}", flush=True)


# ============== OPEN (lazy, whole globe) ==============
cache = xr.open_zarr(CACHE, consolidated=False).drop_vars('boundary')
native = zarr.open_group(cache.attrs['native_store'], mode='r')


def truth_var(v):
    ch = [51 * FIELDS.index(v) + k for k in range(51)][KS]
    mean = xr.DataArray(np.ravel(native[f'{v}_mean'][:])[KS], dims='channel')
    std = xr.DataArray(np.ravel(native[f'{v}_std'][:])[KS], dims='channel')
    da = (cache.prognostic.isel(channel=ch) * std + mean).where(cache.wet.isel(channel=ch).astype(bool))
    return da.rename(channel='k').transpose('time', 'k', 'face', 'y', 'x')


sources = {'llc': {v: truth_var(v) for v in args.vars}}  # all sources (time, k, face, y, x)
for name, path in EMULATORS:
    ds = open_predictions(path)
    sources[name] = {v: ds[v].isel(k=KS).transpose('time', 'k', 'face', 'lat', 'lon') for v in args.vars}
common = cache.time.values
for name, _ in EMULATORS:
    common = np.intersect1d(common, sources[name][args.vars[0]].time.values)
TIMES = common[args.time_range[0]:args.time_range[1] + 1:args.stepping]
log(f"{len(common)} common times -> {len(TIMES)} plotted")

# ============== TRANSECT POINTS -> nearest (face, y, x) ==============
if len(args.lat) == 2:
    along = np.arange(args.lat[0], args.lat[1] + args.res / 2, args.res)
    pt_lat, pt_lon, const_name, const, xlabel = along, np.full_like(along, args.lon[0]), 'lon', args.lon[0], 'Latitude [°]'
else:
    along = np.arange(args.lon[0], args.lon[1] + args.res / 2, args.res)
    pt_lat, pt_lon, const_name, const, xlabel = np.full_like(along, args.lat[0]), along, 'lat', args.lat[0], 'Longitude [°]'


def _xyz(lo, la):
    lo, la = np.deg2rad(lo), np.deg2rad(la)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


XC, YC = cache.XC.values, cache.YC.values
dist, idx = cKDTree(_xyz(XC.ravel(), YC.ravel())).query(_xyz(pt_lon, pt_lat))
FAR = dist > np.deg2rad(1)  # > 1 deg from any cell stays blank
F, Y, X = np.unravel_index(idx, XC.shape)
depth_m = DEPTH_LEVELS[KS]

# Faces a source has no data on (e.g. a 1-face run in a 13-face store is all-NaN elsewhere): drawn gray, not as land
NODATA = {}
for src, fields in sources.items():
    surf = np.asarray(fields[args.vars[0]].sel(time=TIMES[0]).isel(k=0).values)
    covered = np.isfinite(surf).any(axis=(-2, -1))  # (face,)
    NODATA[src] = ~covered[F]
    if not covered.all():
        log(f"  {src}: data only on faces {np.flatnonzero(covered).tolist()}; rest drawn gray")


# ============== READ: one whole-globe time step at a time, then pick the points ==============
def read(job):
    src, v, t = job
    block = np.asarray(sources[src][v].sel(time=t).values, dtype='f4')  # (k, face, y, x), ~200 MB
    out = block[:, F, Y, X]
    out[:, FAR] = np.nan
    return job, out


jobs = [(s, v, t) for s in sources for v in args.vars for t in TIMES]
log(f"reading {len(jobs)} (source, var, time) blocks on {args.threads} threads...")
with ThreadPoolExecutor(args.threads) as pool:
    DATA = dict(pool.map(read, jobs))
log("read done")

# ============== PLOT ==============
tag = (f"{const_name}{const:g}_{'lat' if const_name == 'lon' else 'lon'}{along[0]:g}-{along[-1]:g}"
       f"_k{KS.start}-{KS.stop}_t{args.time_range[0]}-{args.time_range[1]}s{args.stepping}")
out_dir = f"{OUT_ROOT}/{'+'.join(n for n, _ in EMULATORS)}"
os.makedirs(out_dir, exist_ok=True)
labels = {'llc': 'LLC', **{n: n for n, _ in EMULATORS}}

for v in args.vars:
    fig, axes = plt.subplots(len(TIMES), len(sources), figsize=(4.5 * len(sources), 3 * len(TIMES)), dpi=200,
                             squeeze=False, sharey=True, constrained_layout=True)
    for row, t in enumerate(TIMES):
        fields = [DATA[(s, v, t)] for s in sources]
        if not any(np.isfinite(f).any() for f in fields):
            log(f"  skipping {v} {t}: no finite values")
            continue
        if v in ('U', 'V'):
            vmax = max(np.nanmax(np.abs(f)) for f in fields); vmin = -vmax
        else:
            vmin, vmax = min(np.nanmin(f) for f in fields), max(np.nanmax(f) for f in fields)
        time_str = pd.Timestamp(t).strftime('%d/%m/%Y:%Hh')
        for col, (s, field) in enumerate(zip(sources, fields)):
            ax = axes[row, col]
            im = ax.pcolormesh(along, depth_m, field, cmap=CMAPS[v], vmin=vmin, vmax=vmax, shading='nearest')
            if NODATA[s].any():
                gray = np.where(NODATA[s], 1.0, np.nan)[None, :].repeat(len(depth_m), 0)
                ax.pcolormesh(along, depth_m, gray, cmap=matplotlib.colors.ListedColormap(['0.6']), shading='nearest')
            ax.set_ylim(depth_m[-1], depth_m[0])  # surface on top
            ax.set_title(f'{labels[s]} {v} {time_str}', fontsize=8)
            ax.text(0.98, 0.96, f'{const_name} = {const:g}°', transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, bbox=dict(fc='white', alpha=0.7, lw=0))
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel('Depth [m]', fontsize=7)
            if row == len(TIMES) - 1:
                ax.set_xlabel(xlabel, fontsize=7)
        fig.colorbar(im, ax=axes[row, :], pad=0.01, fraction=0.03, label=f'{v} [{UNITS[v]}]')
    out = f"{out_dir}/transect-{v}-{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log(f"saved {out}")

log("done")
