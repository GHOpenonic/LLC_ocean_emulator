"""KE, EKE and buoyancy of the coarse LLC truth vs emulators at one level: maps (+ emulator - truth error) and area-mean time series."""
import argparse
import os
import sys
import time as _time

import dask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from fastjmd95 import jmd95numba
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics')
from predictions_io import open_predictions

CACHE = '/orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_cache_0p33deg.zarr'
OUT_ROOT = '/home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/figs/KE_EKE_B'
FIELDS = ['U', 'V', 'Theta', 'Salt']  # cache channel blocks of 51 levels, in this order
G, RHO0 = 9.81, 1025
UNITS = {'KE': 'm$^2$ s$^{-2}$', 'EKE': 'm$^2$ s$^{-2}$', 'B': 'm s$^{-2}$'}
CMAPS = {'KE': 'magma', 'EKE': 'magma', 'B': 'viridis'}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--emulator', action='append', required=True, metavar='NAME=PATH', help='repeat per emulator')
p.add_argument('--vars', nargs='+', choices=['KE', 'EKE', 'B'], default=['KE', 'EKE', 'B'])
p.add_argument('--mode', nargs='+', choices=['map', 'time_series'], default=['map', 'time_series'])
p.add_argument('--face', default='1', help="0-12 for one face, 'all' for the globe")
p.add_argument('--x-extent', type=int, nargs=2, default=[0, 270], metavar=('START', 'STOP'))
p.add_argument('--y-extent', type=int, nargs=2, default=[0, 270], metavar=('START', 'STOP'))
p.add_argument('--depth', type=int, default=0, help='k level')
p.add_argument('--time-range', type=int, nargs=2, default=[0, 743], metavar=('START', 'END'),
               help='plotted times: inclusive indices into the times truth and emulators share')
p.add_argument('--stepping', type=int, default=48)
p.add_argument('--mean-range', type=int, nargs=2, default=[0, 743], metavar=('START', 'END'),
               help='EKE time-mean window (inclusive indices, same axis as --time-range)')
p.add_argument('--mean-stepping', type=int, default=1)
p.add_argument('--out-dir', default=None)
args = p.parse_args()

N_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
dask.config.set(scheduler='threads', num_workers=N_CPUS)
GLOBE = args.face == 'all'
SPACE = dict(face=slice(0, 13) if GLOBE else int(args.face), x=slice(*args.x_extent), y=slice(*args.y_extent))
EMULATORS = [spec.split('=', 1) for spec in args.emulator]
_t0 = _time.time()


def log(msg):
    print(f"[{_time.time() - _t0:6.0f}s] {msg}", flush=True)


def open_truth():
    """Coarse cache at `depth`: physical U, V, Theta, Salt (time, face, y, x), land NaN, faces 7-12 north-up."""
    cache = xr.open_zarr(CACHE, consolidated=False).drop_vars('boundary')
    native = zarr.open_group(cache.attrs['native_store'], mode='r')
    fields = {}
    for n, v in enumerate(FIELDS):
        c = 51 * n + args.depth
        mean, std = native[f'{v}_mean'][args.depth], native[f'{v}_std'][args.depth]
        wet = cache.wet.isel(channel=c).astype(bool)
        fields[v] = (cache.prognostic.isel(channel=c).astype('f4') * std + mean).where(wet)
    grid = cache[['XC', 'YC', 'rA']]
    return xr.Dataset(fields).assign_coords(face=np.arange(13)), grid.assign_coords(face=np.arange(13))


def open_emulator(path):
    """Emulator at `depth` as dask, with positional face/y/x like the truth."""
    ds = open_predictions(path, chunks={})[FIELDS].isel(k=args.depth).rename(lat='y', lon='x')
    return ds.drop_vars(['k', 'y', 'x'])


truth, grid = open_truth()
raw = {'llc': truth, **{name: open_emulator(path) for name, path in EMULATORS}}
common = truth.time.values
for name, _ in EMULATORS:
    common = np.intersect1d(common, raw[name].time.values)
sources = {k: ds.sel(time=common).isel(**SPACE) for k, ds in raw.items()}
grid = grid.isel(**SPACE).compute()
plot_t = common[args.time_range[0]:args.time_range[1] + 1:args.stepping]
mean_t = common[args.mean_range[0]:args.mean_range[1] + 1:args.mean_stepping]
HOURS = (plot_t - common[0]) / np.timedelta64(1, 'h')
SP_DIMS = [d for d in ('face', 'y', 'x') if d in grid.rA.dims]
LABELS = {'llc': 'LLC (truth)', **{name: name for name, _ in EMULATORS}}
log(f"{len(common)} common times; plotting {len(plot_t)}, EKE mean over {len(mean_t)}; "
    f"face={args.face} x={args.x_extent} y={args.y_extent} k={args.depth}; {N_CPUS} dask threads")

uv_mean = {}
if 'EKE' in args.vars:
    keys = list(sources)
    means = dask.compute(*[sources[k][['U', 'V']].sel(time=mean_t).mean('time') for k in keys])
    uv_mean = dict(zip(keys, means))
    log("EKE time means done")


def quantity(key, var):
    """Lazy (time, [face,] y, x) KE, EKE or B at the plotted times."""
    ds = sources[key].sel(time=plot_t)
    if var == 'KE':
        q = 0.5 * (ds.U ** 2 + ds.V ** 2)
    elif var == 'EKE':
        q = 0.5 * ((ds.U - uv_mean[key].U) ** 2 + (ds.V - uv_mean[key].V) ** 2)
    else:
        rho = xr.apply_ufunc(jmd95numba.rho, ds.Salt, ds.Theta, np.float32(0), dask='parallelized',
                             output_dtypes=['f4'])
        q = -G * (rho - RHO0) / RHO0
    return q.transpose('time', *SP_DIMS)


tag = (f"{'+'.join(n for n, _ in EMULATORS)}_f{args.face}_x{args.x_extent[0]}-{args.x_extent[1]}"
       f"_y{args.y_extent[0]}-{args.y_extent[1]}_k{args.depth}_t{args.time_range[0]}-{args.time_range[1]}s{args.stepping}"
       f"_m{args.mean_range[0]}-{args.mean_range[1]}s{args.mean_stepping}")
OUT_DIR = args.out_dir or f"{OUT_ROOT}/{tag}"
os.makedirs(OUT_DIR, exist_ok=True)

# ============== TIME SERIES: area-weighted spatial mean per plotted time ==============
if 'time_series' in args.mode:
    def area_mean(q):
        w = grid.rA.where(q.notnull())
        return (q * grid.rA).sum(SP_DIMS) / w.sum(SP_DIMS)

    jobs = [(var, key) for var in args.vars for key in sources]
    series = dict(zip(jobs, dask.compute(*[area_mean(quantity(key, var)) for var, key in jobs])))
    log("time series computed")
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for var in args.vars:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for i, key in enumerate(sources):
            style = dict(color='k', lw=2.0, zorder=5) if key == 'llc' else dict(color=palette[(i - 1) % len(palette)], lw=1.3)
            ax.plot(HOURS, series[(var, key)].values, label=LABELS[key], **style)
        ax.set_xlabel('hours since rollout start', fontsize=9)
        ax.set_ylabel(f'area-mean {var} [{UNITS[var]}]', fontsize=9)
        ax.set_title(f'{var} k={args.depth} area mean (face {args.face})', fontsize=11)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
        out = f"{OUT_DIR}/time-series_{var}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f"saved {out}")

# ============== MAPS: truth | emulators | emulator - truth ==============
if 'map' in args.mode:
    if GLOBE:  # nearest-neighbour regrid onto 1/3 deg lon/lat; cells > 1 deg from any wet-or-land cell stay blank
        def _xyz(lon, lat):
            lon, lat = np.deg2rad(lon), np.deg2rad(lat)
            return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
        lon_g, lat_g = np.meshgrid(np.arange(-180, 180, 1 / 3) + 1 / 6, np.arange(-90, 90, 1 / 3) + 1 / 6)
        tree = cKDTree(_xyz(grid.XC.values.ravel(), grid.YC.values.ravel()))
        _, regrid_idx = tree.query(_xyz(lon_g.ravel(), lat_g.ravel()), distance_upper_bound=np.deg2rad(1))
        to_plane = lambda a: np.append(a.ravel(), np.nan)[regrid_idx].reshape(lon_g.shape)
        extent, panel = [-180, 180, -90, 90], (6, 3)
    else:
        to_plane = lambda a: a
        extent, panel = None, (3.6, 3)

    jobs = [(var, key) for var in args.vars for key in sources]
    fields = dict(zip(jobs, dask.compute(*[quantity(key, var) for var, key in jobs])))
    log("map fields computed")
    emu_keys = [name for name, _ in EMULATORS]
    ncols = 1 + 2 * len(emu_keys)
    for var in args.vars:
        fig, axes = plt.subplots(len(plot_t), ncols, figsize=(panel[0] * ncols, panel[1] * len(plot_t)),
                                 dpi=150, squeeze=False)
        for row, t in enumerate(plot_t):
            time_str = pd.Timestamp(t).strftime('%d/%m/%Y:%Hh')
            planes = {key: to_plane(fields[(var, key)].values[row]) for key in sources}
            errors = {key: planes[key] - planes['llc'] for key in emu_keys}
            vals = np.concatenate([a[np.isfinite(a)] for a in planes.values()])
            vmin, vmax = (0, np.percentile(vals, 99)) if var != 'B' else np.percentile(vals, [1, 99])
            elim = np.percentile(np.abs(np.concatenate([a[np.isfinite(a)] for a in errors.values()])), 99)
            panels = [(LABELS[k], planes[k], CMAPS[var], vmin, vmax) for k in sources] + \
                     [(f'{k} - LLC', errors[k], 'RdBu_r', -elim, elim) for k in emu_keys]
            for col, (label, a, cmap, lo, hi) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(a, origin='lower', extent=extent, cmap=cmap, vmin=lo, vmax=hi,
                               aspect='auto', interpolation='nearest')
                ax.set_title(f'{label} {var} k={args.depth} {time_str}', fontsize=8)
                ax.tick_params(labelsize=7)
                if GLOBE:
                    ax.set_ylabel('Latitude', fontsize=7)
                    if row == len(plot_t) - 1:
                        ax.set_xlabel('Longitude', fontsize=7)
                if col == len(sources) - 1:
                    fig.colorbar(im, ax=axes[row, :len(sources)], pad=0.01, fraction=0.03, label=f'{var} [{UNITS[var]}]')
                elif col == ncols - 1:
                    fig.colorbar(im, ax=axes[row, len(sources):], pad=0.01, fraction=0.03, label=f'error [{UNITS[var]}]')
        out = f"{OUT_DIR}/map_{var}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f"saved {out}")

log("done")
