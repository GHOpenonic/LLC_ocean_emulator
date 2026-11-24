"""Coarse LLC truth (left) vs emulators (right) video at one level, one row per variable; frames rendered in parallel, then ffmpeg."""
import argparse
import os

# keep each render process single-threaded (must precede numpy import)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import multiprocessing as mp
import subprocess
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import dask
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics')
from predictions_io import open_predictions
import KE_EKE_B as keb  # KE / buoyancy formulas, colormaps and units

CACHE = '/orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_cache_0p33deg.zarr'
OUT_ROOT = '/home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/figs/video'
FIELDS = ['U', 'V', 'Theta', 'Salt']  # cache channel blocks of 51 levels, in this order; Eta is channel 204
DERIVED = {'KE': (keb.kinetic_energy, ('U', 'V')), 'B': (keb.buoyancy, ('Salt', 'Theta'))}  # var: (fn, its inputs)
CMAPS = {'Theta': 'Spectral_r', 'Salt': 'viridis', 'U': 'bwr', 'V': 'bwr', 'Eta': cmocean.cm.deep_r,
         **{v: keb.CMAPS[v] for v in DERIVED}}
UNITS = {'Theta': '°C', 'Salt': 'PSU', 'U': 'm/s', 'V': 'm/s', 'Eta': 'm', **{v: keb.UNITS[v] for v in DERIVED}}
SYMMETRIC = {'U', 'V'}  # colour limits centred on 0

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--emulator', action='append', required=True, metavar='NAME=PATH', help='repeat per emulator')
p.add_argument('--vars', nargs='+', choices=list(CMAPS), default=['Theta'])
p.add_argument('--face', default='1', help="0-12 for one face, 'all' for the globe")
p.add_argument('--x-extent', type=int, nargs=2, default=[0, 270], metavar=('START', 'STOP'))
p.add_argument('--y-extent', type=int, nargs=2, default=[0, 270], metavar=('START', 'STOP'))
p.add_argument('--depth', type=int, default=0, help='k level (ignored for Eta)')
p.add_argument('--time-range', type=int, nargs=2, default=[0, 744], metavar=('START', 'END'),
               help='inclusive indices into the times truth and emulators share')
p.add_argument('--stepping', type=int, default=1)
p.add_argument('--fps', type=int, default=24)
p.add_argument('--mode', choices=('full', 'fill'), default='full', help='fill: render only missing frames')
p.add_argument('--workers', type=int, default=None)
p.add_argument('--clip-pct', type=float, default=0.5, help='colour-limit percentile clip on LLC')
p.add_argument('--colorscale', nargs='+', choices=('log', 'reg'), default=None,
               help='one per --vars entry, in order (default: all reg)')
args = p.parse_args()
if args.colorscale and len(args.colorscale) != len(args.vars):
    p.error(f"--colorscale needs one entry per var ({len(args.vars)}), got {len(args.colorscale)}")
LOG = {v: s == 'log' for v, s in zip(args.vars, args.colorscale or ['reg'] * len(args.vars))}

N_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
N_WORKERS = args.workers or int(os.environ.get('VIDEO_N_WORKERS', N_CPUS))
FRAMES_PER_WORKER = int(os.environ.get('VIDEO_FRAMES_PER_WORKER', '6'))
dask.config.set(scheduler='threads', num_workers=N_CPUS)
GLOBE = args.face == 'all'
SPACE = dict(face=slice(0, 13) if GLOBE else int(args.face), x=slice(*args.x_extent), y=slice(*args.y_extent))
EMULATORS = [spec.split('=', 1) for spec in args.emulator]
LABELS = {'llc': 'LLC (truth)', **{name: name for name, _ in EMULATORS}}
BASE = list(dict.fromkeys(b for v in args.vars for b in (DERIVED[v][1] if v in DERIVED else (v,))))  # stored vars to load
_t0 = _time.time()


def log(msg):
    print(f"[{_time.time() - _t0:6.0f}s] {msg}", flush=True)


# ============== LOAD (truth + emulators at one level, positional face/y/x) ==============
def open_truth():
    """Physical fields from the normalized coarse cache, land NaN, faces 7-12 north-up."""
    cache = xr.open_zarr(CACHE, consolidated=False).drop_vars('boundary')
    native = zarr.open_group(cache.attrs['native_store'], mode='r')
    fields = {}
    for v in BASE:
        c, lvl = (204, 0) if v == 'Eta' else (51 * FIELDS.index(v) + args.depth, args.depth)
        mean = np.ravel(native[f'{v}_mean'][:])[lvl]
        std = np.ravel(native[f'{v}_std'][:])[lvl]
        wet = cache.wet.isel(channel=c).astype(bool)
        fields[v] = (cache.prognostic.isel(channel=c).astype('f4') * std + mean).where(wet)
    return xr.Dataset(fields).assign_coords(face=np.arange(13)), cache[['XC', 'YC']].assign_coords(face=np.arange(13))


def open_emulator(path):
    ds = open_predictions(path, chunks={})[BASE]
    ds = xr.Dataset({v: ds[v].isel(k=args.depth) if 'k' in ds[v].dims else ds[v] for v in BASE})
    return ds.rename(lat='y', lon='x').drop_vars(['k', 'y', 'x'], errors='ignore')


def with_derived(ds):
    """The requested vars, computing KE / B lazily from their stored inputs."""
    return xr.Dataset({v: DERIVED[v][0](*(ds[b] for b in DERIVED[v][1])) if v in DERIVED else ds[v] for v in args.vars})


truth, grid = open_truth()
raw = {k: with_derived(ds) for k, ds in {'llc': truth, **{name: open_emulator(path) for name, path in EMULATORS}}.items()}
common = truth.time.values
for name, _ in EMULATORS:
    common = np.intersect1d(common, raw[name].time.values)
times = common[args.time_range[0]:args.time_range[1] + 1:args.stepping]
grid = grid.isel(**SPACE).compute()
SP_DIMS = [d for d in ('face', 'y', 'x') if d in grid.XC.dims]

keys = list(raw)
jobs = [(v, k) for v in args.vars for k in keys]
lazy = [raw[k][v].sel(time=times).isel(**SPACE).transpose('time', *SP_DIMS).astype('f4').data for v, k in jobs]
log(f"loading {len(jobs)} fields x {len(times)} frames (face={args.face}, k={args.depth}) on {N_CPUS} threads...")
DATA = dict(zip(jobs, dask.compute(*lazy)))  # shared copy-on-write with forked workers
n_times = len(times)
log(f"loaded {sum(a.nbytes for a in DATA.values()) / 2**30:.2f} GiB")

# ============== PLANE MAPPING (single face as is; globe -> 1/3 deg lon/lat) ==============
if GLOBE:  # nearest neighbour; cells > 1 deg from any grid cell stay blank
    def _xyz(lon, lat):
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    lon_g, lat_g = np.meshgrid(np.arange(-180, 180, 1 / 3) + 1 / 6, np.arange(-90, 90, 1 / 3) + 1 / 6)
    tree = cKDTree(_xyz(grid.XC.values.ravel(), grid.YC.values.ravel()))
    _, regrid_idx = tree.query(_xyz(lon_g.ravel(), lat_g.ravel()), distance_upper_bound=np.deg2rad(1))
    to_plane = lambda a: np.append(a.ravel(), np.nan)[regrid_idx].reshape(lon_g.shape)
    EXTENT, PANEL = [-180, 180, -90, 90], (6.0, 3.2)
else:
    to_plane = lambda a: a
    EXTENT, PANEL = None, (4.0, 3.6)

# ============== COLOUR LIMITS (from LLC, fixed for the whole video) ==============
NORMS = {}
for v in args.vars:
    vals = DATA[(v, 'llc')][np.linspace(0, n_times - 1, min(20, n_times)).astype(int)]
    vals = vals[np.isfinite(vals)]
    pct = lambda a, q: float(np.percentile(a, q))
    if v in SYMMETRIC:
        hi = pct(np.abs(vals), 100 - args.clip_pct)
        NORMS[v] = (mcolors.SymLogNorm(pct(np.abs(vals), 10), vmin=-hi, vmax=hi) if LOG[v]  # linear inside |x| < 10th pct
                    else mcolors.Normalize(-hi, hi))
    elif LOG[v]:  # non-positive values render as blank
        if not (pos := vals[vals > 0]).size:
            raise SystemExit(f"--colorscale log for {v}: LLC has no positive values")
        NORMS[v] = mcolors.LogNorm(pct(pos, args.clip_pct), pct(pos, 100 - args.clip_pct))
    else:
        NORMS[v] = mcolors.Normalize(pct(vals, args.clip_pct), pct(vals, 100 - args.clip_pct))
    log(f"  {v} ({'log' if LOG[v] else 'reg'}): [{NORMS[v].vmin:.4g}, {NORMS[v].vmax:.4g}]")

tag = (f"{'+'.join(n for n, _ in EMULATORS)}_{'-'.join(('log' if LOG[v] else '') + v for v in args.vars)}_f{args.face}_x{args.x_extent[0]}-{args.x_extent[1]}"
       f"_y{args.y_extent[0]}-{args.y_extent[1]}_k{args.depth}_t{args.time_range[0]}-{args.time_range[1]}s{args.stepping}")
FRAME_DIR = f"{OUT_ROOT}/frames_{tag}"
os.makedirs(FRAME_DIR, exist_ok=True)


# ============== PER-WORKER FIGURE (built once, only data swapped per frame) ==============
_WORKER = {}


def _init_worker():
    fig, axes = plt.subplots(len(args.vars), len(keys), figsize=(PANEL[0] * len(keys), PANEL[1] * len(args.vars)),
                             dpi=100, squeeze=False, constrained_layout=True)
    ims = {}
    for r, v in enumerate(args.vars):
        for c, k in enumerate(keys):
            ax = axes[r, c]
            ims[(v, k)] = ax.imshow(to_plane(DATA[(v, k)][0]), origin='lower', extent=EXTENT, cmap=CMAPS[v],
                                    norm=NORMS[v], aspect='auto' if GLOBE else 'equal', interpolation='nearest')
            ax.set_title(f"{LABELS[k]}: {v}" + ('' if v == 'Eta' else f" k={args.depth}"), fontsize=10)
            ax.tick_params(labelsize=7)
        fig.colorbar(ims[(v, keys[-1])], ax=axes[r, :], shrink=0.85, pad=0.01, label=f"{v} [{UNITS[v]}]")
    _WORKER.update(fig=fig, ims=ims, title=fig.suptitle('', fontsize=12))


def render_frame(t):
    for (v, k), im in _WORKER['ims'].items():
        im.set_data(to_plane(DATA[(v, k)][t]))
    _WORKER['title'].set_text(f"face {args.face}  —  {pd.Timestamp(times[t]).strftime('%d/%m/%Y:%Hh')}")
    _WORKER['fig'].savefig(f"{FRAME_DIR}/frame_{t:05d}.png", dpi=100, pil_kwargs={"compress_level": 1})
    return t


def missing_frames():
    return [t for t in range(n_times) if not os.path.isfile(f := f"{FRAME_DIR}/frame_{t:05d}.png") or os.path.getsize(f) == 0]


# ============== DISPATCH + ENCODE ==============
def main():
    frames = missing_frames() if args.mode == 'fill' else list(range(n_times))
    log(f"rendering {len(frames)}/{n_times} frames on {N_WORKERS} workers -> {FRAME_DIR}")
    ctx = mp.get_context('fork')
    per_round = N_WORKERS * FRAMES_PER_WORKER  # fresh pool per round bounds copy-on-write growth
    done, t0 = 0, _time.time()
    for start in range(0, len(frames), per_round):
        batch = frames[start:start + per_round]
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(batch)), mp_context=ctx, initializer=_init_worker) as pool:
            try:
                for _ in pool.map(render_frame, batch, chunksize=max(1, len(batch) // (2 * N_WORKERS))):
                    done += 1
            except BrokenProcessPool:
                log("a render worker died (likely OOM): lower VIDEO_N_WORKERS or raise --mem, then rerun with fill")
                raise
        log(f"  frames done: {done}/{len(frames)} ({done / max(_time.time() - t0, 1e-9):.2f} frame/s)")

    if (left := missing_frames()):
        raise RuntimeError(f"{len(left)} frames still missing in {FRAME_DIR}")
    out = f"{OUT_ROOT}/video_{tag}_fps{args.fps}.mp4"
    subprocess.run(["ffmpeg", "-y", "-threads", str(N_WORKERS), "-framerate", str(args.fps),
                    "-i", f"{FRAME_DIR}/frame_%05d.png", "-frames:v", str(n_times),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even dims
                    "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)
    log(f"saved {out} ({n_times} frames, {n_times / args.fps:.1f}s)")


if __name__ == '__main__':
    main()
