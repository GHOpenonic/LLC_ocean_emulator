# dependencies
import argparse
import os

# ── Threading hygiene: keep each render process single-threaded ──
# (must be set BEFORE numpy / mkl import to take effect)
# With ~100 workers, letting each spawn its own BLAS pool oversubscribes the
# node by 100x and the resulting context switching dominates the run.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# glibc gives every thread its own malloc arena (64 MiB of address space each);
# across ~100 forked workers that inflates RSS for no benefit.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import dask
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import time

# ── Headless backend, MUST be set before pyplot is used ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def _cpu_budget():
    """CPUs this process may use, honouring cgroup/Slurm limits."""
    override = os.environ.get("VIDEO_N_WORKERS")
    if override:
        return max(1, int(override))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return max(1, int(slurm))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, mp.cpu_count() or 1)


N_CPUS = _cpu_budget()
# Frames each render worker draws before its pool is replaced (bounds memory creep).
FRAMES_PER_WORKER = int(os.environ.get("VIDEO_FRAMES_PER_WORKER", "6"))
logger.info(f"CPU budget: {N_CPUS}")

# The preload below is the only phase that benefits from threads: it is pure
# zarr decompression, so give it the whole node before the workers fork.
dask.config.set(scheduler="threads", num_workers=N_CPUS)

# ============================================================
# ============== LOAD LLC + EMULATORS (shared loader) ========
# ============================================================
import sys
sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/scripts')
from diag_common import LLC_STORE, format_time, load_patches

def env_list(name, default=""):
    """Parse a space-separated environment variable."""
    return os.environ.get(name, default).split()


# Domain/time/emulator knobs come from video_gen.sh
emulator_names = env_list(
    "VIDEO_EMULATOR_NAME",
    "1-face-ckpt4",
)

emulator_paths = env_list(
    "VIDEO_EMULATOR_PATH",
    "/orcd/data/abodner/002/cody/inference_patch/"
    "1-face/epoch4-ema-quintic-20121014-20121021-25047717/"
    "predictions.zarr",
)

if len(emulator_names) != len(emulator_paths):
    raise ValueError(
        "VIDEO_EMULATOR_NAME and VIDEO_EMULATOR_PATH must contain "
        f"the same number of entries. "
        f"Got {len(emulator_names)} names and {len(emulator_paths)} paths."
    )


I_EXTENT = [
    int(x)
    for x in os.environ.get("VIDEO_I_EXTENT", "2880 3600").split()
]

J_EXTENT = [
    int(x)
    for x in os.environ.get("VIDEO_J_EXTENT", "720 1440").split()
]

selected_time_range = [
    int(x)
    for x in os.environ.get("VIDEO_TIME_RANGE", "0 71").split()
]  # inclusive

stepping = int(os.environ.get("VIDEO_STEPPING", "1"))

REGION_NAME = os.environ.get(
    "VIDEO_REGION_NAME",
    f"i[{I_EXTENT[0]}:{I_EXTENT[1]}] "
    f"j[{J_EXTENT[0]}:{J_EXTENT[1]}]",
)


# Build one configuration dictionary per emulator
emulator_configs = [
    {
        "name": name,
        "key": f"emulator_{idx + 1}",
        "path": path,
        "desc": "",
    }
    for idx, (name, path) in enumerate(
        zip(emulator_names, emulator_paths)
    )
]


llc_patch, emulator_patches, emulator_info, all_patches = load_patches(
    emulator_configs,
    slice(*I_EXTENT),
    slice(*J_EXTENT),
    selected_time_range,
    stepping,
    emulator_chunks={},
)

n_emulators = len(emulator_info)

# ============================================================
# ============== CONFIG ======================================
# ============================================================

# ── Model (column) ordering: LLC first, then emulators ─────────────
model_order = [('LLC4320', 'llc')] + emulator_info      # [(display_name, key), ...]
n_models = len(model_order)

# ── Variable (row) selection ───────────────────────────────────────
selected_variables = os.environ.get("VIDEO_VARS", "Salt").split()
n_vars = len(selected_variables)

cmaps = {
    'Theta': plt.cm.Spectral_r,
    'Salt':  plt.cm.viridis,
    'U':     plt.cm.bwr,
    'V':     plt.cm.bwr,
    'W':     plt.cm.magma
}
units = {
    'Theta': 'Θ [°C]',
    'Salt':  'S [PSU]',
    'U':     'U [m/s]',
    'V':     'V [m/s]',
    'W':     'W [m/s]'
}

# ── Extra panel knobs (from video_gen.sh) ───────────────────────────
GRADIENTS = os.environ.get("VIDEO_GRADIENTS", "0") == "1"
ERRORS = os.environ.get("VIDEO_ERRORS", "0") == "1"

# Columns per row: (kind, model_key, title); kind is 'field', 'grad' or 'err'
columns = []
for mname, mkey in model_order:
    columns.append(('field', mkey, mname))
    if GRADIENTS:
        columns.append(('grad', mkey, f"{mname} gradients"))
if ERRORS:
    columns += [('err', mkey, f"{mname} − LLC") for mname, mkey in emulator_info]
n_cols = len(columns)
TAG = f"{n_vars}x{n_cols}" + ("-grad" if GRADIENTS else "") + ("-err" if ERRORS else "")

# float16 rounds away neighbour/emulator differences, so derived panels need float32
FACE_DTYPE = np.float32 if (GRADIENTS or ERRORS) else np.float16

# ── Params ──────────────────────────────────────────────────────────
# Offsets INTO the loaded patch (the domain itself is VIDEO_I/J_EXTENT).
i_0, i_1 = 0, llc_patch.sizes['i'] - 1
j_0, j_1 = 0, llc_patch.sizes['j'] - 1
k_min = 0
k_max = 51
n_times = llc_patch.sizes['time']
fps = 24

# Surface-mesh decimation.
# The TOP surface is 720x720 -> expensive, safe to decimate.
# The WALLS are only 51 (k) x 720 -> cheap, and decimating the k-axis
# causes vertical banding, so keep k at full resolution (stride 1).
# At dpi=100 a panel is ~550x500 px, so a 719x719 top surface is already
# ~2 quads per pixel; stride 2 halves the mesh in each direction for no
# visible change and is the single biggest remaining lever if you need the
# run shorter. Overridable so you don't have to edit the file.
SURF_RSTRIDE = int(os.environ.get("VIDEO_SURF_STRIDE", 1))   # top surface (j, i)
SURF_CSTRIDE = SURF_RSTRIDE
WALL_KSTRIDE = 1                     # vertical (k) on walls -> keep full res
WALL_HSTRIDE = int(os.environ.get("VIDEO_WALL_STRIDE", 1))   # horizontal (i or j)

path_out = '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos/figs'
FRAME_DIR = f"{path_out}/frames_{TAG}"
os.makedirs(FRAME_DIR, exist_ok=True)

# ── Face cache ──────────────────────────────────────────────────────
# Extracting the faces costs a few hundred GiB of reads (the walls need every
# k level, and a chunk is a whole 720x720 plane), but the result is only a few
# GiB. So write it out once and reuse it: a rerun, a --mode fill, or a tweak
# to anything downstream of the load then starts in seconds instead of hours.
# The cache is keyed by a hash of everything the arrays depend on, so it can
# never be silently reused after a config change; set VIDEO_FACE_CACHE=0 to
# bypass it entirely.
FACE_CACHE_ENABLED = os.environ.get("VIDEO_FACE_CACHE", "1") != "0"
FACE_CACHE_ROOT = f"{path_out}/face_cache"


def _face_cache_dir():
    """Directory for this exact configuration's cached faces."""
    import hashlib, json
    times = llc_patch.time.values
    manifest = {
        "version": 1,
        "llc_store": LLC_STORE,
        "emulators": [[c["key"], c["path"]] for c in emulator_configs],
        "model_order": [[n, k] for n, k in model_order],
        "variables": list(selected_variables),
        "extents": [I_EXTENT, J_EXTENT],
        "bounds": [i_0, i_1, j_0, j_1, k_min, k_max],
        "time_range": list(selected_time_range),
        "stepping": stepping,
        "n_times": int(len(times)),
        "time_first": str(times[0]),
        "time_last": str(times[-1]),
    }
    if FACE_DTYPE != np.float16:
        manifest["dtype"] = "float32"   # only added here so float16 cache keys stay valid
    blob = json.dumps(manifest, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return f"{FACE_CACHE_ROOT}/{digest}", blob

# ============================================================
# ====== PRELOAD ONLY THE 3 VISIBLE FACES (memory-lean) ======
# ============================================================
# NOTE: LLC's data vars use (time, k, j, i) but the emulators use
# (time, k, lat, lon). The datasets ALSO carry i/j coords from the copied
# grid_vars, which is misleading. So we detect each variable's own
# horizontal dims instead of hardcoding i/j.

# LLC keeps W on `k_p1` -- 52 cell INTERFACES -- while Theta/Salt/U/V sit on
# `k`, 51 cell CENTRES, and the emulator writes W back onto `k`. So the vertical
# axis has to be looked up per variable: hardcoding 'k' both leaves LLC's W at
# full depth and makes `k_p1` look like a third horizontal dimension.
VERTICAL_DIMS = ('k', 'k_p1', 'k_l', 'k_u', 'lev')


def _vertical_dim(da):
    """Return the vertical dimension of `da`, or None for a surface field."""
    for dim in VERTICAL_DIMS:
        if dim in da.dims:
            return dim
    return None


def _horizontal_dims(da):
    """Return (y_dim, x_dim) for a (time, <vertical>, y, x)-style DataArray."""
    vdim = _vertical_dim(da)
    hdims = [d for d in da.dims if d not in ('time', vdim)]
    if len(hdims) != 2:
        raise ValueError(f"Expected 2 horizontal dims, got {da.dims}")
    return hdims[0], hdims[1]   # (y, x): j/lat first, i/lon second


def _depth_values(patch, var, k_slice):
    """Depth in metres for each level of `var` within `k_slice`.

    The store carries `Z` for cell centres but no coordinate for the interfaces
    W sits on, so derive them: face[0] = 0, face[i+1] = 2*centre[i] - face[i].
    """
    centres = np.asarray(patch['Z'].values, dtype=float)
    if _vertical_dim(patch[var]) != 'k_p1':
        return centres[k_slice]
    faces = [0.0]
    for centre in centres:
        faces.append(2.0 * centre - faces[-1])
    return np.asarray(faces)[k_slice]

CACHE_DIR, _cache_blob = _face_cache_dir()
FACE_NAMES = ("surface", "south", "east")


def _face_paths(model_key, var):
    return {f: f"{CACHE_DIR}/{model_key}__{var}__{f}.npy" for f in FACE_NAMES}


def _load_cached_faces(model_key, var):
    """Cached faces for one (model, variable), or None if unusable.

    Cached per variable rather than per run so that a job killed part way
    through the load -- a time limit, say -- keeps what it already paid for.
    """
    if not FACE_CACHE_ENABLED:
        return None
    paths = _face_paths(model_key, var)
    if not all(os.path.isfile(pth) for pth in paths.values()):
        return None
    try:
        # Read into RAM rather than mmap. It is only a few GiB, the parent
        # loads it before forking so every worker shares it copy-on-write, and
        # unlike mapped pages it cannot be evicted and re-fetched over the
        # shared filesystem while 100 renderers compete for memory.
        return {f: np.load(pth) for f, pth in paths.items()}
    except (OSError, ValueError) as exc:
        logger.warning(f"  {model_key}/{var}: cache unreadable ({exc}); re-reading.")
        return None


def _store_cached_faces(model_key, var, per_face):
    """Persist one variable's faces. A cache we cannot write is not fatal."""
    if not FACE_CACHE_ENABLED:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        for fname, arr in per_face.items():
            final = f"{CACHE_DIR}/{model_key}__{var}__{fname}.npy"
            # Write through a handle: np.save(path, ...) would append another
            # ".npy" to a temp name that does not end in it, and the rename
            # would then miss the file it just wrote.
            tmp = f"{final}.tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, final)   # atomic: never leaves a half-written .npy
        with open(f"{CACHE_DIR}/manifest.json", "w") as fh:
            fh.write(_cache_blob)
    except OSError as exc:
        logger.warning(f"  {model_key}/{var}: could not cache ({exc}); continuing.")


logger.info(f"Loading visible faces ({np.dtype(FACE_DTYPE).name}); cache: "
            f"{CACHE_DIR if FACE_CACHE_ENABLED else 'disabled'}")
faces = {}
_n_cached = 0

for _, model_key in model_order:
    ds = all_patches[model_key]
    faces[model_key] = {}

    for var in selected_variables:
        cached = _load_cached_faces(model_key, var)
        if cached is not None:
            faces[model_key][var] = cached
            _n_cached += 1
            logger.info(f"  {model_key}/{var}: from cache "
                        f"surface{cached['surface'].shape}")
            continue

        k_dim = _vertical_dim(ds[var])
        v = ds[var].isel({k_dim: slice(k_min, k_max)})

        y_dim, x_dim = _horizontal_dims(v)

        v = v.isel({
            y_dim: slice(j_0, j_1),
            x_dim: slice(i_0, i_1),
        })

        ny = v.sizes[y_dim]
        nx = v.sizes[x_dim]

        # One dask.compute for all three faces, not three .load() calls.
        # The walls need every k level, and a chunk here is a whole 720x720
        # plane, so the three faces come out of the same chunk grid; asking
        # for them together lets dask read that grid once and fan the slices
        # out of it. This only works because the stores are dask-backed --
        # see the open_zarr note above.
        t_var = time.monotonic()
        surface, south, east = dask.compute(
            v.isel({k_dim: 0}).data,
            v.isel({y_dim: 0}).data,
            v.isel({x_dim: nx - 1}).data,
        )

        faces[model_key][var] = {
            "surface": np.asarray(surface, dtype=FACE_DTYPE),
            "south": np.asarray(south, dtype=FACE_DTYPE),
            "east": np.asarray(east, dtype=FACE_DTYPE),
        }
        del surface, south, east
        _store_cached_faces(model_key, var, faces[model_key][var])

        logger.info(
            f"  {model_key}/{var} [{k_dim}={v.sizes[k_dim]}, {y_dim}={ny}, {x_dim}={nx}]: "
            f"surface{faces[model_key][var]['surface'].shape} "
            f"south{faces[model_key][var]['south'].shape} "
            f"east{faces[model_key][var]['east'].shape} "
            f"in {time.monotonic() - t_var:.1f}s"
        )

_preload_bytes = sum(
    a.nbytes
    for per_var in faces.values()
    for per_face in per_var.values()
    for a in per_face.values()
)
logger.info(
    f"Faces: {_preload_bytes / 2**30:.2f} GiB in RAM "
    f"({_n_cached}/{len(model_order) * len(selected_variables)} from cache; "
    f"shared with every worker copy-on-write, not duplicated)"
)

# Time labels (small, safe to keep in memory)
time_values = llc_patch.time.values

# ============================================================
# ============== COLOR LIMITS (from LLC truth) ===============
# ============================================================
# Clip the limits to a percentile rather than the outright min/max. LLC4320
# carries a handful of junk cells -- the 4-tile Salt window has 3 cells out of
# 5.4M sitting at ~43 PSU against a real range of 34.2-36.6 -- and a straight
# nanmax hands those 3 cells 60% of the colormap, leaving the actual field as
# one flat block of colour. Set VIDEO_CLIP_PCT=0 for true min/max.
CLIP_PCT = float(os.environ.get("VIDEO_CLIP_PCT", 0.1))

logger.info(f"Computing global color limits from LLC faces "
            f"(clip: {CLIP_PCT}%)...")
norms = {}
for var in selected_variables:
    arrs = [
        faces['llc'][var]['surface'],
        faces['llc'][var]['south'],
        faces['llc'][var]['east'],
    ]
    gmin = min(float(np.nanmin(a)) for a in arrs)
    gmax = max(float(np.nanmax(a)) for a in arrs)
    if CLIP_PCT > 0:
        # float32 for the percentile: np.percentile on float16 accumulates
        # visible error, and these arrays are tens of millions of cells.
        finite = np.concatenate([
            a[np.isfinite(a)].ravel().astype(np.float32) for a in arrs
        ])
        lo = float(np.percentile(finite, CLIP_PCT))
        hi = float(np.percentile(finite, 100.0 - CLIP_PCT))
        del finite
        if hi > lo:
            if (gmax - gmin) > 1.5 * (hi - lo):
                logger.info(f"  {var}: outliers clipped, "
                            f"[{gmin:.4f}, {gmax:.4f}] -> [{lo:.4f}, {hi:.4f}]")
            gmin, gmax = lo, hi
    norms[var] = mcolors.Normalize(vmin=gmin, vmax=gmax)
    logger.info(f"  {var}: [{gmin:.4f}, {gmax:.4f}]")

# ============================================================
# ============== SHARED GRID GEOMETRY ========================
# ============================================================
i_vals = np.arange(i_0, i_1 + 1)
j_vals = np.arange(j_0, j_1 + 1)

# One depth axis per variable: W is rendered on its own interface depths rather
# than borrowed tracer centres, so its box lines up with what it represents.
z_vals_by_var = {
    var: _depth_values(llc_patch, var, slice(k_min, k_max))
    for var in selected_variables
}
z_vals = z_vals_by_var[selected_variables[0]]

I_surf, J_surf = np.meshgrid(i_vals, j_vals)
Z_surf_by_var = {
    var: np.full_like(I_surf, zv[0], dtype=float)
    for var, zv in z_vals_by_var.items()
}
Z_surf = Z_surf_by_var[selected_variables[0]]

Z_jwall_by_var, Z_iwall_by_var = {}, {}
for var, zv in z_vals_by_var.items():
    _, Z_jwall_by_var[var] = np.meshgrid(i_vals, zv)
    _, Z_iwall_by_var[var] = np.meshgrid(j_vals, zv)

I_jwall, _ = np.meshgrid(i_vals, z_vals)
J_jwall = np.full_like(I_jwall, j_0, dtype=float)

J_iwall, _ = np.meshgrid(j_vals, z_vals)
I_iwall = np.full_like(J_iwall, i_1, dtype=float)

z_bounds_by_var = {
    var: (float(zv[0]), float(zv[-1])) for var, zv in z_vals_by_var.items()
}
z_top, z_bot = z_bounds_by_var[selected_variables[0]]
edge_kw = dict(color="k", linewidth=0.5)

# ============================================================
# ============== GRADIENT / ERROR PANELS =====================
# ============================================================
# Computed per frame inside the workers; only the colour limits are global.
DXY = np.sqrt(llc_patch['rA'].isel(j=slice(j_0, j_1), i=slice(i_0, i_1)).values).astype(np.float32)


def hgrad(s):
    """|∇h| of surface field(s) (..., j, i), per metre."""
    gy, gx = np.gradient(s.astype(np.float32), axis=(-2, -1))
    return np.hypot(gx, gy) / DXY


def vgrad(w, var):
    """|∂z| of wall field(s) (..., k, x), per metre."""
    return np.abs(np.gradient(w.astype(np.float32), z_vals_by_var[var], axis=-2))


def panel_faces(kind, mkey, var, t):
    """(surface, south, east) for one panel at frame(s) t."""
    f = faces[mkey][var]
    if kind == 'field':
        return f['surface'][t], f['south'][t], f['east'][t]
    if kind == 'grad':
        return hgrad(f['surface'][t]), vgrad(f['south'][t], var), vgrad(f['east'][t], var)
    llc = faces['llc'][var]
    return tuple(f[n][t].astype(np.float32) - llc[n][t] for n in FACE_NAMES)   # emulator - llc


def panel_keys(kind, var):
    """Colour-scale key for each of (surface, south, east)."""
    return {'field': (var,) * 3,
            'grad': ((var, 'h'), (var, 'v'), (var, 'v')),
            'err': ((var, 'err'),) * 3}[kind]


def _hi(arrs):
    """Upper colour limit over the given arrays (same clip as the fields)."""
    return float(np.nanpercentile(np.concatenate([np.ravel(a) for a in arrs]), 100.0 - CLIP_PCT))


# Fixed limits for the whole video, from frames spread over the full time range
sample_t = np.linspace(0, n_times - 1, min(20, n_times)).astype(int)
for var in selected_variables:
    if GRADIENTS:
        gs, gso, ge = panel_faces('grad', 'llc', var, sample_t)
        norms[(var, 'h')] = mcolors.Normalize(0.0, _hi([gs]))
        norms[(var, 'v')] = mcolors.Normalize(0.0, _hi([gso, ge]))
        cmaps[(var, 'h')] = cmaps[(var, 'v')] = plt.cm.Purples
        units[(var, 'h')], units[(var, 'v')] = f"|∇h {var}| [/m] (top)", f"|∂z {var}| [/m] (walls)"
    if ERRORS:
        e = max(_hi([np.abs(a) for a in panel_faces('err', k, var, sample_t)]) for _, k in emulator_info)
        norms[(var, 'err')] = mcolors.Normalize(-e, e)
        cmaps[(var, 'err')] = plt.cm.RdBu_r
        units[(var, 'err')] = f"Δ{units[var]}"
for key in norms:
    if key not in selected_variables:
        logger.info(f"  {key}: [{norms[key].vmin:.3e}, {norms[key].vmax:.3e}]")

# ── Colour lookup ───────────────────────────────────────────────────
# `cmap(norm(data))` walks a Normalize + Colormap call chain for every one of
# the 518_400 cells of a face, every frame. A Colormap is a 256-entry table, so
# quantise once and fancy-index instead: same 256 output colours, one pass.
_COLOR_LUT = {}
for _var in norms:
    _cmap = cmaps[_var]
    _COLOR_LUT[_var] = (
        np.asarray(_cmap(np.arange(_cmap.N)), dtype=np.float32),
        np.asarray(_cmap.get_bad(), dtype=np.float32),
        float(norms[_var].vmin),
        float(_cmap.N) / (float(norms[_var].vmax) - float(norms[_var].vmin)),
    )


def to_fc(data_2d, var):
    """RGBA for `data_2d` under this variable's colormap and global limits."""
    lut, bad, vmin, scale = _COLOR_LUT[var]
    x = (np.asarray(data_2d, dtype=np.float32) - vmin) * scale
    finite = np.isfinite(x)
    # clip to [0, N-1]: matches Colormap's under/over fallback to the end
    # colours, which is what the LLC-derived limits already imply.
    np.clip(x, 0.0, lut.shape[0] - 0.001, out=x, where=finite)
    idx = np.where(finite, x, 0.0).astype(np.uint16)
    out = lut[idx]
    if not finite.all():
        out[~finite] = bad
    return out


def draw_edges(ax, var=None):
    ax.plot([i_0, i_1], [j_0, j_0], [z_top, z_top], **edge_kw)
    ax.plot([i_0, i_1], [j_1, j_1], [z_top, z_top], **edge_kw)
    ax.plot([i_0, i_0], [j_0, j_1], [z_top, z_top], **edge_kw)
    ax.plot([i_1, i_1], [j_0, j_1], [z_top, z_top], **edge_kw)
    ax.plot([i_0, i_1], [j_0, j_0], [z_bot, z_bot], **edge_kw)
    ax.plot([i_1, i_1], [j_0, j_1], [z_bot, z_bot], **edge_kw)
    ax.plot([i_0, i_0], [j_0, j_0], [z_top, z_bot], **edge_kw)
    ax.plot([i_1, i_1], [j_0, j_0], [z_top, z_bot], **edge_kw)
    ax.plot([i_1, i_1], [j_1, j_1], [z_top, z_bot], **edge_kw)

# ============================================================
# ====== FAST QUAD MESH (replaces per-frame plot_surface) ====
# ============================================================
# `ax.plot_surface(..., facecolors=...)` is the wrong tool for a 720x720 face.
# Because `facecolors` is set it takes matplotlib's non-vectorised branch and
# runs a Python loop over all 518_400 cells, calling `_array_perimeter` and
# building a small array per cell -- then `Poly3DCollection.do_3d_projection`
# runs a *second* 518_400-element Python loop (a `sorted()` with a
# `np.ma.column_stack` per item) on every single draw.
#
# Two observations kill both loops:
#   1. The geometry never changes -- only the colours do. So tessellate once
#      per worker and thereafter only swap facecolors.
#   2. The camera never changes either, so the back-to-front draw order is one
#      fixed permutation. Compute it once at setup and bake it into the vertex
#      array, instead of re-deriving it on every draw of every frame.


def _quad_verts(X, Y, Z, rstride=1, cstride=1):
    """(N, 4, 3) corner array for a grid, matching plot_surface's cell order.

    Cells are emitted row-major and wound (r0,c0) -> (r0,c1) -> (r1,c1) ->
    (r1,c0), which is what `cbook._array_perimeter` yields for a 2x2 block, so
    a facecolor array flattened the same way lines up cell for cell.
    """
    rows, cols = np.shape(Z)
    r = np.r_[np.arange(0, rows - 1, rstride), rows - 1]
    c = np.r_[np.arange(0, cols - 1, cstride), cols - 1]
    r0, r1, c0, c1 = r[:-1], r[1:], c[:-1], c[1:]

    def corners(A):
        A = np.asarray(A, dtype=float)
        return np.stack(
            [A[np.ix_(r0, c0)], A[np.ix_(r0, c1)],
             A[np.ix_(r1, c1)], A[np.ix_(r1, c0)]],
            axis=-1,
        )

    quads = np.stack([corners(X), corners(Y), corners(Z)], axis=-1)
    return quads.reshape(-1, 4, 3), (r0, c0)


class _FastQuadCollection(Poly3DCollection):
    """Poly3DCollection specialised for a static grid of equal-sized quads.

    `do_3d_projection` becomes one vectorised projection: no Python loop over
    polygons, and no re-sort. The back-to-front order still matters -- with
    edgecolor == facecolor, adjacent quads overlap in their seam -- but the
    camera and the geometry are both fixed, so the order is a constant
    permutation. `freeze_view_order` computes it once, in exactly the way
    Poly3DCollection would (descending mean projected depth, stable), and
    bakes it into the vertex array.
    """

    def __init__(self, quads, **kwargs):
        super().__init__([], **kwargs)
        self._quads = np.ascontiguousarray(quads, dtype=float)
        self._set_vec(self._quads)
        self._segslices = None
        self._codes3d = None
        self._closed = True

    def _set_vec(self, quads):
        self._n_quads = quads.shape[0]
        flat = quads.reshape(-1, 3)
        self._vec = np.vstack([flat.T, np.ones(flat.shape[0])])

    def freeze_view_order(self):
        """Sort quads back-to-front for the current view; return the permutation.

        Must be called after the axis limits and `view_init` are final, since
        those are what the projection matrix is built from.
        """
        self.axes.M = self.axes.get_proj()
        _, _, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        depth = np.asarray(tzs).reshape(self._n_quads, 4).mean(axis=1)
        order = np.argsort(-depth, kind="stable")
        self._set_vec(np.ascontiguousarray(self._quads[order]))
        del self._quads          # ~50 MB per 720x720 face, no longer needed
        return order.astype(np.int32, copy=False)

    def do_3d_projection(self):
        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        verts2d = np.stack([txs, tys], axis=-1).reshape(self._n_quads, 4, 2)
        PolyCollection.set_verts(self, verts2d, self._closed)
        self._facecolors2d = self._facecolor3d
        self._edgecolors2d = self._edgecolor3d
        return np.min(tzs) if tzs.size else np.nan

    def set_frame_colors(self, rgba):
        """Point the mesh at a new (N, 4) colour array for the next draw."""
        # edge == face is what plot_surface does; it is what hides the seams.
        self.set_facecolor(rgba)
        self.set_edgecolor(rgba)


def _add_face(ax, X, Y, Z, rstride, cstride, zorder):
    """Attach one static quad mesh to `ax`; return (mesh, colour selector).

    The axes must already have their final limits and view angle.
    """
    quads, (r0, c0) = _quad_verts(X, Y, Z, rstride, cstride)
    mesh = _FastQuadCollection(quads, shade=False, zorder=zorder)
    ax.add_collection(mesh)
    order = mesh.freeze_view_order()
    sel = np.ix_(r0, c0)
    n_quads = quads.shape[0]
    del quads

    def colors(data_2d, var):
        return to_fc(data_2d, var)[sel].reshape(n_quads, 4)[order]

    return mesh, colors

# ============================================================
# ============== PER-WORKER FIGURE (built once) ==============
# ============================================================
_WORKER = {}   # module-global per-process state: {'fig':..., 'meshes':...}


def _init_worker():
    """Build ONE reusable figure per worker, with static geometry baked in.

    Everything that does not change between frames -- tessellation, axis
    limits, labels, titles, box edges, colorbars -- is done here exactly once,
    so `render_frame` is reduced to colour swaps plus the rasterisation.
    """
    fig = plt.figure(figsize=(5.5 * n_cols, 5.0 * n_vars), dpi=100)
    meshes = {}
    axes = {}
    for r, var in enumerate(selected_variables):
        for c, (kind, mkey, mname) in enumerate(columns):
            idx = r * n_cols + c + 1
            ax = fig.add_subplot(n_vars, n_cols, idx, projection='3d')
            axes[(var, c)] = ax

            # Fixed limits and view: we skip plot_surface, so nothing
            # autoscales for us -- and the mesh sort order below is only valid
            # once the projection these define is final.
            ax.auto_scale_xyz([i_0, i_1], [j_0, j_1],
                              list(z_bounds_by_var[var]), False)
            ax.view_init(elev=35, azim=-45)
            draw_edges(ax, var)

            surf = _add_face(ax, I_surf, J_surf, Z_surf_by_var[var],
                             SURF_RSTRIDE, SURF_CSTRIDE, zorder=3)
            south = _add_face(ax, I_jwall, J_jwall, Z_jwall_by_var[var],
                              WALL_KSTRIDE, WALL_HSTRIDE, zorder=2)
            east = _add_face(ax, I_iwall, J_iwall, Z_iwall_by_var[var],
                             WALL_KSTRIDE, WALL_HSTRIDE, zorder=1)
            meshes[(var, c)] = (surf, south, east)

            ax.set_xlabel("i", labelpad=6, fontsize=7)
            ax.set_ylabel("j", labelpad=6, fontsize=7)
            ax.set_zlabel("Depth [m]", labelpad=6, fontsize=7)
            ax.tick_params(labelsize=6)

            if r == 0:
                ax.set_title(mname, fontsize=11, pad=12)
            if c == 0:
                ax.text2D(-0.15, 0.5, var, transform=ax.transAxes,
                          fontsize=12, rotation=90, va='center', ha='center')

        # one colorbar per colour scale, on the last column that uses it (|∇h| on the first, so it doesn't stack with |∂z|)
        for key in dict.fromkeys(k for kind, _, _ in columns for k in panel_keys(kind, var)):
            pick = min if key == (var, 'h') else max
            c = pick(c for c, (kind, _, _) in enumerate(columns) if key in panel_keys(kind, var))
            sm = plt.cm.ScalarMappable(cmap=cmaps[key], norm=norms[key])
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[(var, c)], fraction=0.026, pad=0.10, shrink=0.55)
            cbar.set_label(units[key], fontsize=8)

    fig.subplots_adjust(wspace=0.10, hspace=0.15)
    # The only text that varies per frame -- keep the artist, retitle in place.
    suptitle = fig.suptitle("", fontsize=13, y=0.97)

    _WORKER['fig'] = fig
    _WORKER['meshes'] = meshes
    _WORKER['suptitle'] = suptitle


def render_frame(frame_idx):
    """Render a single frame to PNG using this worker's reusable figure."""
    fig = _WORKER['fig']
    meshes = _WORKER['meshes']

    for var in selected_variables:
        for c, (kind, mkey, _) in enumerate(columns):
            # surface (j, i), south (k, i), east (k, j)
            for (mesh, colors), data, key in zip(meshes[(var, c)],
                                                 panel_faces(kind, mkey, var, frame_idx),
                                                 panel_keys(kind, var)):
                mesh.set_frame_colors(colors(data, key))

    _WORKER['suptitle'].set_text(
        f"LLC4320 {REGION_NAME}  —  {format_time(time_values[frame_idx])}")

    out = f"{FRAME_DIR}/frame_{frame_idx:05d}.png"
    # compress_level 1 instead of 6: these PNGs exist only to be handed to
    # ffmpeg seconds later, so CPU spent shrinking them is CPU wasted.
    fig.savefig(out, dpi=100, pil_kwargs={"compress_level": 1})
    return frame_idx

# ============================================================
# ============== DISPATCH + ENCODE ===========================
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Render LLC 3D video frames and encode MP4/GIF outputs."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "fill"),
        default="full",
        help=(
            "full: render every frame before encoding (default); "
            "fill: render only missing or empty PNG frames before encoding"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="render processes (default: $VIDEO_N_WORKERS, else "
             "$SLURM_CPUS_PER_TASK, else the CPU affinity mask)",
    )
    parser.add_argument(
        "--preload-only",
        action="store_true",
        help="extract and cache the faces, then exit without rendering. Use "
             "this to prime the cache in its own job: the extraction is I/O "
             "bound and needs few cores but a long wall clock, whereas the "
             "render wants every core and, with a warm cache, minutes",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="also write a GIF. Off by default: it needs two extra full "
             "passes over every PNG and comes out far larger than the MP4",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help=argparse.SUPPRESS,   # now the default; kept so old commands run
    )
    return parser.parse_args()


def missing_frame_indices():
    """Return expected frame indices whose PNG is absent or empty."""
    return [
        frame_idx
        for frame_idx in range(n_times)
        if not os.path.isfile(f"{FRAME_DIR}/frame_{frame_idx:05d}.png")
        or os.path.getsize(f"{FRAME_DIR}/frame_{frame_idx:05d}.png") == 0
    ]


def main():
    args = parse_args()
    n_workers = args.workers or N_CPUS

    # The faces are extracted at import time, so by here the cache is written.
    if args.preload_only:
        logger.info(f"--preload-only: faces ready in {CACHE_DIR}; not rendering.")
        return
    if args.mode == "fill":
        frame_indices = missing_frame_indices()
        logger.info(
            f"Fill mode: found {len(frame_indices)} missing or empty frames "
            f"in {FRAME_DIR} (expected {n_times} total)."
        )
    else:
        frame_indices = list(range(n_times))

    if frame_indices:
        n_workers = max(1, min(n_workers, len(frame_indices)))
        logger.info(f"Rendering {len(frame_indices)} frames on {n_workers} workers "
                    f"(surface stride={SURF_RSTRIDE}/{SURF_CSTRIDE}, "
                    f"wall stride={WALL_KSTRIDE}/{WALL_HSTRIDE})...")

        # Free the lazy xarray/zarr graphs before forking: they are useless to
        # the workers (everything they read lives in `faces`) and every worker
        # would otherwise inherit and dirty their pages.
        global llc_patch_full, emulator_patches_raw, emulator_patches_subset
        llc_patch_full = emulator_patches_raw = emulator_patches_subset = None

        # 'fork' (Linux default) lets workers share the preloaded `faces` arrays
        # copy-on-write, so we do NOT duplicate the data per worker in RAM.
        ctx = mp.get_context("fork")
        # Frames are rendered in rounds of FRAMES_PER_WORKER per worker, each round
        # on a fresh pool: workers exit before copy-on-write growth can add up
        # (the fork-safe stand-in for max_tasks_per_child). A worker killed by
        # the OOM killer raises BrokenProcessPool instead of hanging the job.
        per_round = n_workers * FRAMES_PER_WORKER
        t0 = time.monotonic()
        done = 0
        step = max(1, len(frame_indices) // 20)
        for start in range(0, len(frame_indices), per_round):
            batch = frame_indices[start:start + per_round]
            chunksize = max(1, len(batch) // (n_workers * 2))
            with ProcessPoolExecutor(max_workers=min(n_workers, len(batch)), mp_context=ctx,
                                     initializer=_init_worker) as pool:
                try:
                    for _ in pool.map(render_frame, batch, chunksize=chunksize):
                        done += 1
                        if done % step == 0 or done == len(frame_indices):
                            rate = done / max(time.monotonic() - t0, 1e-9)
                            eta = (len(frame_indices) - done) / max(rate, 1e-9)
                            logger.info(f"  frames done: {done}/{len(frame_indices)} "
                                        f"({rate:.2f} frame/s, ETA {eta / 60:.1f} min)")
                except BrokenProcessPool:
                    logger.error("A render worker died (most likely OOM-killed). Lower "
                                 "VIDEO_N_WORKERS or raise --mem, then rerun in fill mode.")
                    raise
        logger.info(f"Rendering took {(time.monotonic() - t0) / 60:.1f} min")
    else:
        logger.info("All expected frames already exist; skipping rendering.")

    remaining = missing_frame_indices()
    if remaining:
        preview = ", ".join(f"{idx:05d}" for idx in remaining[:10])
        suffix = "..." if len(remaining) > 10 else ""
        raise RuntimeError(
            f"Cannot encode: {len(remaining)} frames are still missing or empty "
            f"in {FRAME_DIR} (first: {preview}{suffix})"
        )

    # ── Stitch PNGs → MP4 (fast, high quality) ──
    # PNG decode is the bottleneck here, not x264, so thread the decoder too.
    ff_threads = ["-threads", str(n_workers)]
    mp4_out = f"{path_out}/llc_3D_{TAG}-full.mp4"
    logger.info(f"Encoding MP4 with ffmpeg → {mp4_out}")
    subprocess.run([
        "ffmpeg", "-y",
        *ff_threads,
        "-framerate", str(fps),
        "-i", f"{FRAME_DIR}/frame_%05d.png",
        # FRAME_DIR is keyed only on the panel layout, so a shorter run can
        # land in a directory a longer one already filled. Without this the
        # %05d pattern keeps reading past our last frame and silently splices
        # the previous run's leftovers onto the end.
        "-frames:v", str(n_times),
        "-c:v", "libx264",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        *ff_threads,
        mp4_out,
    ], check=True)

    if not args.gif:
        logger.info(f"Done! {n_times} frames → {n_times/fps:.1f}s "
                    f"(MP4: {mp4_out}, GIF skipped; pass --gif for one)")
        return

    # ── Also produce a GIF (palette method = better quality) ──
    # Read the MP4 rather than the PNG sequence: two more decode passes over
    # ~700 full-size PNGs cost far more than decoding the h264 we just wrote.
    gif_out = f"{path_out}/llc_3D_{TAG}-full.gif"
    logger.info(f"Encoding GIF with ffmpeg → {gif_out}")
    palette = f"{FRAME_DIR}/palette.png"
    subprocess.run([
        "ffmpeg", "-y", *ff_threads,
        "-i", mp4_out,
        "-vf", "palettegen",
        palette,
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", *ff_threads,
        "-i", mp4_out,
        "-i", palette,
        "-lavfi", "paletteuse",
        gif_out,
    ], check=True)

    logger.info(f"Done! {n_times} frames → {n_times/fps:.1f}s "
                f"(MP4: {mp4_out}, GIF: {gif_out})")

if __name__ == "__main__":
    main()
