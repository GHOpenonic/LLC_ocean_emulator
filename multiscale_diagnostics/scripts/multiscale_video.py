"""Multiscale emulator vs LLC truth video at one level on a lat/lon map, sharp inside N and coarse outside: one row per
variable, columns truth | emulator [| emulator - truth]; frames rendered in parallel, then ffmpeg."""
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
from pathlib import Path

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
sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/scripts')
from predictions_io import open_predictions
import KE_EKE_B as keb  # KE / buoyancy formulas, colormaps and units

COARSE_CACHE = '/orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_cache_0p33deg.zarr'
FACE_CACHE = '/orcd/scratch/seedfund/001/llc_emulator/LLC4320_face1_i0-4320_j0-4320.zarr'
OUT_ROOT = '/home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/figs/video'
FIELDS = ['U', 'V', 'Theta', 'Salt']  # both caches: channel blocks of 51 levels in this order; Eta is channel 204
BOX = 16  # sharp cells per coarse cell
DERIVED = {'KE': (keb.kinetic_energy, ('U', 'V')), 'B': (keb.buoyancy, ('Salt', 'Theta'))}  # var: (fn, its inputs)
CMAPS = {'Theta': 'Spectral_r', 'Salt': 'viridis', 'U': 'bwr', 'V': 'bwr', 'Eta': cmocean.cm.deep_r,
         **{v: keb.CMAPS[v] for v in DERIVED}}
UNITS = {'Theta': '°C', 'Salt': 'PSU', 'U': 'm/s', 'V': 'm/s', 'Eta': 'm', **{v: keb.UNITS[v] for v in DERIVED}}
SYMMETRIC = {'U', 'V'}  # colour limits centred on 0
DPI = 100

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--emulator', required=True, metavar='NAME=DIR',
               help='DIR holds sharp_pred.zarr and coarse_pred.zarr (a path to either works too)')
p.add_argument('--vars', nargs='+', choices=list(CMAPS), default=['Theta'])
p.add_argument('--colorscale', nargs='+', choices=('log', 'reg'), default=None,
               help='one per --vars entry, in order (default: all reg)')
p.add_argument('--depth', type=int, default=0, help='k level (Eta is always k=0)')
p.add_argument('--time-range', type=int, nargs=2, default=[0, 744], metavar=('START', 'END'),
               help='inclusive hours since the IC')
p.add_argument('--stepping', type=int, default=3,
               help='hours; each store steps every max(stepping, its save interval) hours and holds its last step between')
p.add_argument('--lat-extent', type=float, nargs=2, default=[-90, 90], metavar=('SOUTH', 'NORTH'))
p.add_argument('--lon-extent', type=float, nargs=2, default=[-180, 180], metavar=('WEST', 'EAST'),
               help='degrees east; EAST may pass 180 to cross the dateline')
p.add_argument('--pixels', type=int, default=1000, help='map width per panel in pixels')
p.add_argument('--error', action='store_true', help='add an emulator - truth column')
p.add_argument('--min-ocean-frac', type=float, default=0.25,
               help="mask cells under this ocean fraction at the level (coarse 1 - land_frac, sharp hFacC)")
p.add_argument('--fps', type=int, default=24)
p.add_argument('--mode', choices=('full', 'fill'), default='full', help='fill: render only missing frames')
p.add_argument('--workers', type=int, default=None)
p.add_argument('--clip-pct', type=float, default=0.5, help='colour-limit percentile clip on truth')
p.add_argument('--sharp-truth', default=FACE_CACHE, help='face cache holding the sharp domain')
args = p.parse_args()
args.lat_extent = sorted(args.lat_extent)  # either order
if args.lon_extent[1] <= args.lon_extent[0]:
    p.error(f"--lon-extent {args.lon_extent}: EAST must exceed WEST (pass 180 to cross the dateline, e.g. 170 190)")
if args.colorscale and len(args.colorscale) != len(args.vars):
    p.error(f"--colorscale needs one entry per var ({len(args.vars)}), got {len(args.colorscale)}")
LOG = {v: s == 'log' for v, s in zip(args.vars, args.colorscale or ['reg'] * len(args.vars))}

N_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
N_WORKERS = args.workers or int(os.environ.get('VIDEO_N_WORKERS', N_CPUS))
FRAMES_PER_WORKER = int(os.environ.get('VIDEO_FRAMES_PER_WORKER', '6'))
VARS, K, THR = args.vars, args.depth, args.min_ocean_frac
BASE = list(dict.fromkeys(b for v in VARS for b in (DERIVED[v][1] if v in DERIVED else (v,))))  # stored vars to load
LEVEL = {v: 0 if v == 'Eta' else K for v in BASE}
CH = [204 if v == 'Eta' else 51 * FIELDS.index(v) + K for v in BASE]
_t0 = _time.time()


def log(msg):
    print(f"[{_time.time() - _t0:6.0f}s] {msg}", flush=True)


# ============== LOAD (lazy; workers read their own frames) ==============
NAME, _path = args.emulator.split('=', 1)
_dir = Path(_path).parent if Path(_path).suffix == '.zarr' else Path(_path)
EMU = {'coarse': open_predictions(str(_dir / 'coarse_pred.zarr')), 'sharp': open_predictions(str(_dir / 'sharp_pred.zarr'))}
IC = np.datetime64(EMU['coarse'].attrs['start'])
ct = xr.open_zarr(COARSE_CACHE, consolidated=False, chunks=None)
ft = xr.open_zarr(args.sharp_truth, consolidated=False, chunks=None)
CT_TIME, FT_TIME = pd.Index(ct.time.values), pd.Index(ft.time.values)


def _hours(times):
    return (times - IC) // np.timedelta64(1, 'h')


# Each store steps every max(STEPPING, its save interval) hours; frames are their union, and each grid shows its latest
# step at or before the frame (the IC before its first), as the rollout carries it.
PICKED, EVERY = {}, {}
for g, ds in EMU.items():
    h = _hours(ds.time.values)
    saved = int(np.diff(h).min())
    EVERY[g] = max(args.stepping, saved)
    if EVERY[g] % saved:
        raise ValueError(f"STEPPING {args.stepping} h isn't a multiple of the {g} save interval ({saved} h)")
    PICKED[g] = ds.time.values[(h >= args.time_range[0]) & (h <= args.time_range[1]) & (h % EVERY[g] == 0)]
FRAMES = np.unique(np.concatenate(list(PICKED.values())))
N_F = len(FRAMES)
HELD = {g: np.concatenate([np.array([IC], dtype=t.dtype), t])[np.searchsorted(t, FRAMES, side='right')]
        for g, t in PICKED.items()}
for g, idx in (('coarse', CT_TIME), ('sharp', FT_TIME)):
    if (gap := ~np.isin(HELD[g], idx)).any():
        raise ValueError(f"the {g} truth lacks steps {HELD[g][gap]}")

# N: the sharp domain's coarse footprint, as validate_multiscale's tile_placement builds it.
FACE = int(ft.attrs['llc_face'])
lat, lon = EMU['sharp'].lat.values, EMU['sharp'].lon.values
J0, J1, I0, I1 = int(lat[0]), int(lat[-1]) + 1, int(lon[0]), int(lon[-1]) + 1
if FACE >= 7 or any(v % BOX for v in (J0, J1, I0, I1)):
    raise ValueError(f"face {FACE} j[{J0}:{J1}) i[{I0}:{I1}) isn't a native-orientation box of whole coarse cells")
INSIDE = np.zeros((13, 270, 270), bool)
INSIDE[FACE, J0 // BOX:J1 // BOX, I0 // BOX:I1 // BOX] = True
INSIDE = INSIDE.ravel()
FY = slice(J0 - int(ft.attrs['llc_j_start']), J1 - int(ft.attrs['llc_j_start']))
FX = slice(I0 - int(ft.attrs['llc_i_start']), I1 - int(ft.attrs['llc_i_start']))
NI = I1 - I0

# ============== LAT/LON MAPPING (nearest cell; in N the sharp one) ==============
(LAT0, LAT1), (LON0, LON1) = args.lat_extent, args.lon_extent
W = args.pixels
H = max(1, round(W * (LAT1 - LAT0) / (LON1 - LON0)))
LON_C, LAT_C = LON0 + (LON1 - LON0) * (np.arange(W) + 0.5) / W, LAT0 + (LAT1 - LAT0) * (np.arange(H) + 0.5) / H
EXTENT = [LON0, LON1, LAT0, LAT1]


def _xyz(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


_px = _xyz(*np.meshgrid(LON_C, LAT_C)).reshape(-1, 3)
_, _cidx = cKDTree(_xyz(ct.XC.values.ravel(), ct.YC.values.ravel())).query(_px, distance_upper_bound=np.deg2rad(1))
_has = _cidx < INSIDE.size
IN_N = np.zeros(H * W, bool)
IN_N[_has] = INSIDE[_cidx[_has]]
OUT_PX, OUT_CELL = np.flatnonzero(_has & ~IN_N), _cidx[_has & ~IN_N]
IN_PX, IN_CELL = np.flatnonzero(IN_N), np.zeros(0, int)
SY = SX = None  # the sharp window read each frame: rows / cols of the face cache
if IN_PX.size:
    _pts = _xyz(ft.XC.isel(y=FY, x=FX).values.ravel(), ft.YC.isel(y=FY, x=FX).values.ravel())
    _, _sidx = cKDTree(_pts, balanced_tree=False, compact_nodes=False).query(_px[IN_PX],
                                                                             distance_upper_bound=np.deg2rad(0.1))
    _ok = _sidx < len(_pts)
    IN_PX, _sidx = IN_PX[_ok], _sidx[_ok]
    _jj, _ii = np.divmod(_sidx, NI)
    WJ, WI = slice(int(_jj.min()), int(_jj.max()) + 1), slice(int(_ii.min()), int(_ii.max()) + 1)  # domain positions
    IN_CELL = (_jj - WJ.start) * (WI.stop - WI.start) + (_ii - WI.start)
    SY, SX = slice(FY.start + WJ.start, FY.start + WJ.stop), slice(FX.start + WI.start, FX.start + WI.stop)
    del _pts, _sidx
del _px, _cidx

# Small-ocean cells are NaN in truth and emulator alike; land is too (0 in the coarse cache, NaN in the face cache).
_lf = {k: ct.land_frac.isel(k=k).values.ravel() for k in set(LEVEL.values())}
COARSE_OK = {v: (_lf[LEVEL[v]] < 1) & (1 - _lf[LEVEL[v]] >= THR) for v in BASE}
if SY is not None:
    _hf = {k: ft.hFacC.isel(k=k, y=SY, x=SX).values.ravel() for k in set(LEVEL.values())}
    SHARP_OK = {v: (_hf[LEVEL[v]] > 0) & (_hf[LEVEL[v]] >= THR) for v in BASE}
_native = zarr.open_group(ct.attrs['native_store'], mode='r')
C_MEAN = np.array([np.ravel(_native[f'{v}_mean'][:])[LEVEL[v]] for v in BASE], 'f4')[:, None]
C_STD = np.array([np.ravel(_native[f'{v}_std'][:])[LEVEL[v]] for v in BASE], 'f4')[:, None]
S_MEAN, S_STD = (ft[n].values[CH].astype('f4')[:, None] for n in ('prognostic_mean', 'prognostic_std'))
CO_CELLS, SH_CELLS = np.unique(OUT_CELL), np.unique(IN_CELL)  # the cells shown, for the error column's RMSE
CO_W = ct.rA.values.ravel().astype('f8')[CO_CELLS]
SH_W = ft.rA.isel(y=SY, x=SX).values.ravel().astype('f8')[SH_CELLS] if SY is not None else None


def _emu(ds, v, tau, **space):
    da = ds[v].sel(time=tau)
    if v != 'Eta':
        da = da.sel(k=K)
    return np.asarray(da.isel(**space).values, dtype='f4').ravel()


def _fields(grid, tau, truth):
    """VARS on `grid` at tau as flat physical float32, small-ocean cells NaN; an emulator at the IC is the truth."""
    if grid == 'coarse':
        if truth or tau == IC:
            a = np.asarray(ct.prognostic.isel(time=CT_TIME.get_loc(tau), channel=CH).values)
            a = a.transpose(1, 0, 2, 3).reshape(len(CH), -1).astype('f4') * C_STD + C_MEAN
            base = dict(zip(BASE, a))
        else:
            base = {v: _emu(EMU['coarse'], v, tau) for v in BASE}
        ok = COARSE_OK
    else:
        if truth or tau == IC:
            a = np.asarray(ft.prognostic.isel(time=FT_TIME.get_loc(tau), prognostic_channel=CH, y=SY, x=SX).values)
            base = dict(zip(BASE, a.reshape(len(CH), -1).astype('f4') * S_STD + S_MEAN))
        else:
            base = {v: _emu(EMU['sharp'], v, tau, lat=WJ, lon=WI) for v in BASE}
        ok = SHARP_OK
    base = {v: np.where(ok[v], x, np.nan) for v, x in base.items()}
    return {v: np.asarray(DERIVED[v][0](*(base[b] for b in DERIVED[v][1])), 'f4') if v in DERIVED else base[v]
            for v in VARS}


def _image(coarse, sharp):
    img = np.full(H * W, np.nan, 'f4')
    if coarse is not None:
        img[OUT_PX] = coarse[OUT_CELL]
    if sharp is not None:
        img[IN_PX] = sharp[IN_CELL]
    return img.reshape(H, W)


def _rmse(x, truth, cells, w):
    """Area-weighted RMSE over `cells` valid in both."""
    d = x[cells] - truth[cells]
    ok = np.isfinite(d)
    return float(np.sqrt((w[ok] * d[ok] ** 2).sum() / w[ok].sum())) if ok.any() else np.nan


def load_frame(f):
    """Frame f's {(var, 'llc' | NAME): (H, W) map} and, with --error, {var: (coarse, sharp) RMSE over the shown cells}."""
    src = {key: (_fields('coarse', HELD['coarse'][f], truth) if OUT_PX.size else None,
                 _fields('sharp', HELD['sharp'][f], truth) if IN_PX.size else None)
           for key, truth in (('llc', True), (NAME, False))}
    imgs = {(v, key): _image(None if c is None else c[v], None if s is None else s[v])
            for key, (c, s) in src.items() for v in VARS}
    rmse = {}
    if args.error:
        (tc, ts), (ec, es) = src['llc'], src[NAME]
        rmse = {v: (np.nan if ec is None else _rmse(ec[v], tc[v], CO_CELLS, CO_W),
                    np.nan if es is None else _rmse(es[v], ts[v], SH_CELLS, SH_W)) for v in VARS}
    return imgs, rmse


# ============== COLOUR LIMITS (from truth, fixed for the whole video; set in main before the render pools fork) ==============
NORMS, ERR_NORMS = {}, {}


def _norm(vals, symmetric, log_scale):
    vals = vals[np.isfinite(vals)]
    pct = lambda a, q: float(np.percentile(a, q))
    if symmetric:
        a = np.abs(vals)
        hi = pct(a, 100 - args.clip_pct) or 1e-12
        # linear inside |x| < 10th pct
        return mcolors.SymLogNorm(pct(a, 10) or hi / 100, vmin=-hi, vmax=hi) if log_scale else mcolors.Normalize(-hi, hi)
    if log_scale:  # non-positive values render as blank
        if not (pos := vals[vals > 0]).size:
            raise SystemExit("--colorscale log: truth has no positive values")
        return mcolors.LogNorm(pct(pos, args.clip_pct), pct(pos, 100 - args.clip_pct))
    return mcolors.Normalize(pct(vals, args.clip_pct), pct(vals, 100 - args.clip_pct))


ocean = f"_ocean{THR:g}" if THR else ""
tag = (f"{NAME}_{'-'.join(('log' if LOG[v] else '') + v for v in VARS)}_k{K}_lat{LAT0:g}_{LAT1:g}_lon{LON0:g}_{LON1:g}"
       f"_h{args.time_range[0]}-{args.time_range[1]}s{args.stepping}h_px{W}{ocean}{'_err' if args.error else ''}")
FRAME_DIR = f"{OUT_ROOT}/frames_{tag}"
COLS = ['llc', NAME] + (['err'] if args.error else [])
TITLES = {'llc': 'LLC4320 (truth)', NAME: NAME, 'err': f"{NAME} − truth"}


def _level(v):
    return '' if v == 'Eta' else f" k={K}"


# ============== PER-WORKER FIGURE (built once, only data swapped per frame) ==============
_WORKER = {}


def _init_worker():
    panel = W / DPI
    fig, axes = plt.subplots(len(VARS), len(COLS), figsize=(1.15 * panel * len(COLS), (panel * H / W + 0.6) * len(VARS)),
                             dpi=DPI, squeeze=False, constrained_layout=True)
    blank, ims, titles = np.full((H, W), np.nan, 'f4'), {}, {}
    seam = IN_N.reshape(H, W).astype('f4') if IN_PX.size and OUT_PX.size else None
    for r, v in enumerate(VARS):
        for c, col in enumerate(COLS):
            ax = axes[r, c]
            cmap, norm = ('RdBu_r', ERR_NORMS[v]) if col == 'err' else (CMAPS[v], NORMS[v])
            ims[v, col] = ax.imshow(blank, origin='lower', extent=EXTENT, cmap=cmap, norm=norm, interpolation='nearest')
            if seam is not None:  # N's edge: sharp inside, coarse outside
                ax.contour(LON_C, LAT_C, seam, levels=[0.5], colors='k', linewidths=0.6, linestyles='--')
            titles[v, col] = ax.set_title(f"{TITLES[col]}: {v}{_level(v)}", fontsize=11)
            ax.tick_params(labelsize=8)
        fig.colorbar(ims[v, NAME], ax=axes[r, :2], shrink=0.85, pad=0.01, label=f"{v} [{UNITS[v]}]")
        if args.error:
            fig.colorbar(ims[v, 'err'], ax=axes[r, 2], shrink=0.85, pad=0.01, label=f"Δ{v} [{UNITS[v]}]")
    _WORKER.update(fig=fig, ims=ims, titles=titles, suptitle=fig.suptitle('', fontsize=13))


def _when(f):
    """Frame f's title: its time, and the held step of any grid that didn't step there."""
    h = int(_hours(FRAMES[f]))
    held = [f"{g} at +{int(_hours(HELD[g][f]))} h" for g, n in (('coarse', OUT_PX.size), ('sharp', IN_PX.size))
            if n and HELD[g][f] != FRAMES[f]]
    return f"{pd.Timestamp(FRAMES[f]).strftime('%d/%m/%Y:%Hh')} (+{h} h)" + (f"  [{', '.join(held)}]" if held else "")


def render_frame(f):
    imgs, rmse = load_frame(f)
    for (v, col), im in _WORKER['ims'].items():
        im.set_data(imgs[v, NAME] - imgs[v, 'llc'] if col == 'err' else imgs[v, col])
        if col == 'err':
            _WORKER['titles'][v, col].set_text(f"{TITLES[col]}: {v}{_level(v)}\nRMSE coarse {rmse[v][0]:.3g} | "
                                               f"sharp {rmse[v][1]:.3g} {UNITS[v]}")
    _WORKER['suptitle'].set_text(f"{NAME}  —  {_when(f)}")
    _WORKER['fig'].savefig(f"{FRAME_DIR}/frame_{f:05d}.png", dpi=DPI, pil_kwargs={"compress_level": 1})
    return f, rmse


def missing_frames():
    return [f for f in range(N_F) if not os.path.isfile(x := f"{FRAME_DIR}/frame_{f:05d}.png") or os.path.getsize(x) == 0]


# ============== DISPATCH + ENCODE ==============
def main():
    ctx = mp.get_context('fork')
    log(f"{NAME}: {N_F} frames {FRAMES[0]} -> {FRAMES[-1]} (+{_hours(FRAMES[0])} to +{_hours(FRAMES[-1])} h from IC {IC}); "
        f"coarse every {EVERY['coarse']} h, sharp every {EVERY['sharp']} h")
    log(f"map lat [{LAT0:g}, {LAT1:g}] lon [{LON0:g}, {LON1:g}] at {W}x{H} px ({(LON1 - LON0) / W:.3g} deg/px): "
        f"{OUT_PX.size} coarse px, {IN_PX.size} sharp px (N = face {FACE} j[{J0}:{J1}) i[{I0}:{I1}))"
        + (f", sharp window y[{SY.start}:{SY.stop}) x[{SX.start}:{SX.stop})" if SY is not None else ""))
    v0 = BASE[0]
    masked = lambda wet, ok: f"{(wet & ~ok).sum() / max(wet.sum(), 1):.2%}"
    log(f"vars={VARS} k={K}; min ocean frac {THR:g} masks {masked(_lf[LEVEL[v0]] < 1, COARSE_OK[v0])} of wet coarse cells"
        + (f", {masked(_hf[LEVEL[v0]] > 0, SHARP_OK[v0])} of the sharp window's" if SY is not None else "")
        + f" at k={LEVEL[v0]}")

    sample = np.linspace(0, N_F - 1, min(12, N_F)).astype(int)
    with ProcessPoolExecutor(max_workers=max(1, min(N_WORKERS, len(sample))), mp_context=ctx) as pool:
        loaded = list(pool.map(load_frame, sample))
    for v in VARS:
        NORMS[v] = _norm(np.concatenate([im[v, 'llc'].ravel() for im, _ in loaded]), v in SYMMETRIC, LOG[v])
        log(f"  {v} ({'log' if LOG[v] else 'reg'}): [{NORMS[v].vmin:.4g}, {NORMS[v].vmax:.4g}]")
        if args.error:
            ERR_NORMS[v] = _norm(np.concatenate([(im[v, NAME] - im[v, 'llc']).ravel() for im, _ in loaded]), True, LOG[v])
            log(f"  {v} error: [{ERR_NORMS[v].vmin:.4g}, {ERR_NORMS[v].vmax:.4g}]")
    del loaded

    os.makedirs(FRAME_DIR, exist_ok=True)
    frames = missing_frames() if args.mode == 'fill' else list(range(N_F))
    log(f"rendering {len(frames)}/{N_F} frames on {N_WORKERS} workers -> {FRAME_DIR}")
    per_round = N_WORKERS * FRAMES_PER_WORKER  # fresh pool per round bounds copy-on-write growth
    for start in range(0, len(frames), per_round):
        batch = frames[start:start + per_round]
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(batch)), mp_context=ctx, initializer=_init_worker) as pool:
            try:
                for f, rmse in pool.map(render_frame, batch):
                    log(f"frame {f + 1:4d}/{N_F} {_when(f)}"
                        + "".join(f"  {v} rmse coarse {c:.3g} sharp {s:.3g}" for v, (c, s) in rmse.items()))
            except BrokenProcessPool:
                log("a render worker died: see any traceback above, else likely OOM (lower VIDEO_N_WORKERS or raise "
                    "--mem, then rerun with fill)")
                raise

    if (left := missing_frames()):
        raise RuntimeError(f"{len(left)} frames still missing in {FRAME_DIR}")
    out = f"{OUT_ROOT}/video_{tag}_fps{args.fps}.mp4"
    subprocess.run(["ffmpeg", "-y", "-threads", str(N_WORKERS), "-framerate", str(args.fps),
                    "-i", f"{FRAME_DIR}/frame_%05d.png", "-frames:v", str(N_F),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even dims
                    "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)
    log(f"saved {out} ({N_F} frames, {N_F / args.fps:.1f}s)")


if __name__ == '__main__':
    main()
