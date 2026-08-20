# dependencies
import os

# ── Threading hygiene: keep each of the 60 processes single-threaded ──
# (must be set BEFORE numpy / mkl import to take effect)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import dask
import subprocess
import multiprocessing as mp

# ── Headless backend, MUST be set before pyplot is used ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# ============== LOAD LLC ====================================
# ============================================================
llc_patch_full = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320', consolidated=False).isel(
    face=1,
    i=slice(2880, 3600),
    i_g=slice(2880, 3600),
    j=slice(720, 1440),
    j_g=slice(720, 1440),
)
# quad: i=[2880:4320), j=[0:1440)
# Agulhas: i=[2880:3600), j=[720:1440)

# ============================================================
# ============== LOAD EMULATORS ==============================
# ============================================================
emulator_configs = [
    # {
    #     'name': 'rb-pred-resid-field-ckpt-50',
    #     'key': 'emulator_1',
    #     'path': '/orcd/data/abodner/002/cody/inference_patch/2026-07-20-eval:Samudra_LLC:rb-Agulhas-pred_field-eager-ckpt50-fixed-18406515/predictions_4d.zarr',
    #     'desc': ''
    # },
        {
        'name': 'rb-pred-resid-eager-ckpt-50',
        'key': 'emulator_1',
        'path': '/orcd/data/abodner/002/cody/inference_patch/2026-07-20-eval:Samudra_LLC:rb-Agulhas-pred_resid-eager-ckpt50-fixed-3weeks-18427424/predictions_4d_extended-18444486.zarr',
        'desc': ''
    },
    # {
    #     'name': 'rb-Agulhas-strides=1-pred_field-ckpt-25',
    #     'key': 'emulator_2',
    #     'path': '/orcd/data/abodner/002/cody/inference_patch/rb/2026-07-06-eval:Samudra_LLC:rb-Agulhas-strides=1-pred_field-ckpt-25-17335589/predictions_4d.zarr',
    #     'desc': ''   
    # },

]

# ============== OPEN EMULATOR DATASETS ==============
emulator_patches_raw = {}
for cfg in emulator_configs:
    emulator_patches_raw[cfg['key']] = xr.open_dataset(cfg['path'], consolidated=True)
    logger.info(f"Loaded {cfg['name']}: {cfg['desc']}")

# ============== TIME MATCHING ==============
def normalize_times(times):
    return pd.DatetimeIndex([
        pd.Timestamp(
            int(t.year), int(t.month), int(t.day),
            int(t.hour), int(t.minute), int(t.second)
        )
        if hasattr(t, "year")
        else pd.Timestamp(t).floor("s")
        for t in times
    ])

llc_times_norm = normalize_times(llc_patch_full.time.values)

common_times = llc_times_norm
for cfg in emulator_configs:
    emulator_times_norm = normalize_times(emulator_patches_raw[cfg['key']].time.values)
    common_times = common_times.intersection(emulator_times_norm)

common_times = common_times.sort_values()

llc_mask = llc_times_norm.isin(common_times)
llc_patch = llc_patch_full.isel(time=llc_mask)

logger.info(f"LLC subset to {len(common_times)} common times")

# ============== PORT GRID VARS & BUILD UNIFIED STRUCTURE ==============
grid_vars = ['XC', 'YC', 'rA', 'Z', 'dxC', 'dyC', 'dxG', 'dyG']

emulator_patches = {}
for cfg in emulator_configs:
    patch_raw = emulator_patches_raw[cfg['key']]
    patch_times_norm = normalize_times(patch_raw.time.values)

    patch_mask = patch_times_norm.isin(common_times)
    patch = patch_raw.isel(time=patch_mask)

    for gv in grid_vars:
        patch[gv] = llc_patch[gv]

    emulator_patches[cfg['key']] = patch

# ============== UNIFIED REFERENCE LISTS ==============
emulator_info = [(cfg['name'], cfg['key']) for cfg in emulator_configs]
n_emulators = len(emulator_info)

all_patches = {'llc': llc_patch}
all_patches.update(emulator_patches)

logger.info(f"=== Setup complete: LLC + {n_emulators} emulators ===")
for name, key in emulator_info:
    logger.info(f"  {name} ({key})")

# ============== TIME SUBSET / SYNC ==============
selected_time_range = [0, 48]   # inclusive indices
stepping = 1 

start_idx, end_idx = selected_time_range

# First subset LLC
llc_patch = llc_patch.isel(time=slice(start_idx, end_idx + 1, stepping))

# Then subset each emulator safely (handles shorter emulator runs)
emulator_patches_subset = {}
for key, patch in emulator_patches.items():
    max_time = patch.sizes['time']
    safe_end_idx = min(end_idx, max_time - 1)
    patch_subset = patch.isel(time=slice(start_idx, safe_end_idx + 1, stepping))
    emulator_patches_subset[key] = patch_subset

emulator_patches = emulator_patches_subset

# Match LLC length to shortest emulator
min_time_len = min(
    [llc_patch.sizes['time']] +
    [patch.sizes['time'] for patch in emulator_patches.values()]
)

llc_patch = llc_patch.isel(time=slice(0, min_time_len))
emulator_patches = {
    key: patch.isel(time=slice(0, min_time_len))
    for key, patch in emulator_patches.items()
}

# Rebuild combined dict
all_patches = {'llc': llc_patch}
all_patches.update(emulator_patches)

# Diagnostics
logger.info(f"Subset to time indices {start_idx}:{end_idx}")
logger.info(f"Stepping = {stepping}")
logger.info(f"Final synchronized length = {min_time_len}")
logger.info(f"LLC now has {llc_patch.sizes['time']} times")
for name, key in emulator_info:
    logger.info(f"{name} ({key}) now has {emulator_patches[key].sizes['time']} times")

def format_time(t_val):
    """Format a time value to DD/MM/YYYY:HH regardless of cftime or datetime64."""
    try:
        return f"{t_val.day:02d}/{t_val.month:02d}/{t_val.year}:{t_val.hour:02d}h"
    except AttributeError:
        t_pd = pd.Timestamp(t_val)
        return f"{t_pd.day:02d}/{t_pd.month:02d}/{t_pd.year}:{t_pd.hour:02d}h"

# ============================================================
# ============== CONFIG ======================================
# ============================================================

# ── Model (column) ordering: LLC first, then emulators ─────────────
model_order = [('LLC4320', 'llc')] + emulator_info      # [(display_name, key), ...]
n_models = len(model_order)

# ── Variable (row) selection ───────────────────────────────────────
selected_variables = ['Theta', 'Salt', 'U', 'V']
n_vars = len(selected_variables)

cmaps = {
    'Theta': plt.cm.Spectral_r,
    'Salt':  plt.cm.viridis,
    'U':     plt.cm.bwr,
    'V':     plt.cm.bwr,
}
units = {
    'Theta': 'Θ [°C]',
    'Salt':  'S [PSU]',
    'U':     'U [m/s]',
    'V':     'V [m/s]',
}

# ── Params ──────────────────────────────────────────────────────────
i_0, i_1 = 200, 220
j_0, j_1 = 200, 220
k_min = 0
k_max = 51
n_times = llc_patch.sizes['time']
fps = 1

# Surface-mesh decimation.
# The TOP surface is 720x720 -> expensive, safe to decimate.
# The WALLS are only 51 (k) x 720 -> cheap, and decimating the k-axis
# causes vertical banding, so keep k at full resolution (stride 1).
SURF_RSTRIDE, SURF_CSTRIDE = 1, 1    # top surface (j, i)
WALL_KSTRIDE = 1                     # vertical (k) on walls -> keep full res
WALL_HSTRIDE = 1                     # horizontal (i or j) on walls

# Surface-mesh decimation. rstride/cstride=1 is brutally slow on 720x720.
# 4 is nearly indistinguishable in a video and MUCH faster. Bump to 1 if
# you really need full resolution.
RSTRIDE = 1
CSTRIDE = 1

path_out = '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos/figs'
FRAME_DIR = f"{path_out}/frames_{n_vars}x{n_models}"
os.makedirs(FRAME_DIR, exist_ok=True)

# ============================================================
# ====== PRELOAD ONLY THE 3 VISIBLE FACES (memory-lean) ======
# ============================================================
# NOTE: LLC's data vars use (time, k, j, i) but the emulators use
# (time, k, lat, lon). The datasets ALSO carry i/j coords from the copied
# grid_vars, which is misleading. So we detect each variable's own
# horizontal dims instead of hardcoding i/j.

def _horizontal_dims(da):
    """Return (y_dim, x_dim) for a (time, k, y, x)-style DataArray."""
    hdims = [d for d in da.dims if d not in ('time', 'k')]
    if len(hdims) != 2:
        raise ValueError(f"Expected 2 horizontal dims, got {da.dims}")
    return hdims[0], hdims[1]   # (y, x): j/lat first, i/lon second

logger.info("Preloading visible faces into RAM (float16)...")
faces = {}

for _, model_key in model_order:
    ds = all_patches[model_key]
    faces[model_key] = {}

    for var in selected_variables:
        v = ds[var].isel(k=slice(k_min, k_max))

        y_dim, x_dim = _horizontal_dims(v)

        v = v.isel({
            y_dim: slice(j_0, j_1),
            x_dim: slice(i_0, i_1),
        })

        ny = v.sizes[y_dim]
        nx = v.sizes[x_dim]

        faces[model_key][var] = {
            "surface": v.isel(k=k_min).load().values.astype(np.float16),
            "south": v.isel({y_dim: 0}).load().values.astype(np.float16),
            "east": v.isel({x_dim: nx - 1}).load().values.astype(np.float16),
        }

        logger.info(
            f"  {model_key}/{var} [{y_dim}={ny}, {x_dim}={nx}]: "
            f"surface{faces[model_key][var]['surface'].shape} "
            f"south{faces[model_key][var]['south'].shape} "
            f"east{faces[model_key][var]['east'].shape}"
        )

# Time labels (small, safe to keep in memory)
time_values = llc_patch.time.values

# ============================================================
# ============== COLOR LIMITS (from LLC truth) ===============
# ============================================================
logger.info("Computing global color limits from LLC faces...")
norms = {}
for var in selected_variables:
    arrs = [
        faces['llc'][var]['surface'],
        faces['llc'][var]['south'],
        faces['llc'][var]['east'],
    ]
    gmin = min(float(np.nanmin(a)) for a in arrs)
    gmax = max(float(np.nanmax(a)) for a in arrs)
    norms[var] = mcolors.Normalize(vmin=gmin, vmax=gmax)
    logger.info(f"  {var}: [{gmin:.4f}, {gmax:.4f}]")

# ============================================================
# ============== SHARED GRID GEOMETRY ========================
# ============================================================
i_vals = np.arange(i_0, i_1 + 1)
j_vals = np.arange(j_0, j_1 + 1)
z_vals = llc_patch.isel(k=slice(k_min, k_max))["Z"].values

I_surf, J_surf = np.meshgrid(i_vals, j_vals)
Z_surf = np.full_like(I_surf, z_vals[0], dtype=float)

I_jwall, Z_jwall = np.meshgrid(i_vals, z_vals)
J_jwall = np.full_like(I_jwall, j_0, dtype=float)

J_iwall, Z_iwall = np.meshgrid(j_vals, z_vals)
I_iwall = np.full_like(J_iwall, i_1, dtype=float)

z_top, z_bot = float(z_vals[0]), float(z_vals[-1])
edge_kw = dict(color="k", linewidth=0.5)

def to_fc(data_2d, var):
    return cmaps[var](norms[var](data_2d))

def draw_edges(ax):
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
# ============== PER-WORKER FIGURE (built once) ==============
# ============================================================
_WORKER = {}   # module-global per-process state: {'fig':..., 'axes':...}

def _init_worker():
    """Build ONE reusable figure per worker process."""
    fig = plt.figure(figsize=(5.5 * n_models, 5.0 * n_vars), dpi=100)
    axes = {}
    for r, var in enumerate(selected_variables):
        for c, (mname, mkey) in enumerate(model_order):
            idx = r * n_models + c + 1
            axes[(var, mkey)] = fig.add_subplot(
                n_vars, n_models, idx, projection='3d')
        # one colorbar per row, on the rightmost axis (persists across cla())
        last_ax = axes[(var, model_order[-1][1])]
        sm = plt.cm.ScalarMappable(cmap=cmaps[var], norm=norms[var])
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=last_ax, fraction=0.026, pad=0.10, shrink=0.55)
        cbar.set_label(units[var], fontsize=8)

    fig.subplots_adjust(wspace=0.10, hspace=0.15)
    _WORKER['fig'] = fig
    _WORKER['axes'] = axes

def render_frame(frame_idx):
    """Render a single frame to PNG using this worker's reusable figure."""
    fig = _WORKER['fig']
    axes = _WORKER['axes']

    for r, var in enumerate(selected_variables):
        for c, (model_name, model_key) in enumerate(model_order):
            ax = axes[(var, model_key)]
            ax.cla()

            f = faces[model_key][var]
            surface = f['surface'][frame_idx]   # (j, i)
            south   = f['south'][frame_idx]     # (k, i)
            east    = f['east'][frame_idx]      # (k, j)

            # top surface: 720x720, decimate freely
            ax.plot_surface(I_surf, J_surf, Z_surf,
                            facecolors=to_fc(surface, var),
                            shade=False,
                            rstride=SURF_RSTRIDE, cstride=SURF_CSTRIDE, zorder=3)

            # south wall (k, i): rstride = k (keep=1), cstride = i
            ax.plot_surface(I_jwall, J_jwall, Z_jwall,
                            facecolors=to_fc(south, var),
                            shade=False,
                            rstride=WALL_KSTRIDE, cstride=WALL_HSTRIDE, zorder=2)

            # east wall (k, j): rstride = k (keep=1), cstride = j
            ax.plot_surface(I_iwall, J_iwall, Z_iwall,
                            facecolors=to_fc(east, var),
                            shade=False,
                            rstride=WALL_KSTRIDE, cstride=WALL_HSTRIDE, zorder=1)

            draw_edges(ax)
            ax.view_init(elev=35, azim=-45)
            ax.set_xlabel("i", labelpad=6, fontsize=7)
            ax.set_ylabel("j", labelpad=6, fontsize=7)
            ax.set_zlabel("Depth [m]", labelpad=6, fontsize=7)
            ax.tick_params(labelsize=6)

            if r == 0:
                ax.set_title(model_name, fontsize=11, pad=12)
            if c == 0:
                ax.text2D(-0.15, 0.5, var, transform=ax.transAxes,
                          fontsize=12, rotation=90, va='center', ha='center')

    fig.suptitle(f"LLC4320 Agulhas  —  {format_time(time_values[frame_idx])}",
                 fontsize=13, y=0.97)

    out = f"{FRAME_DIR}/frame_{frame_idx:05d}.png"
    fig.savefig(out, dpi=100)
    return frame_idx

# ============================================================
# ============== DISPATCH + ENCODE ===========================
# ============================================================
def main():
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count() or 1))
    n_workers = max(1, min(n_workers, n_times))
    logger.info(f"Rendering {n_times} frames on {n_workers} workers "
                f"(rstride/cstride={RSTRIDE}/{CSTRIDE})...")

    # 'fork' (Linux default) lets workers share the preloaded `faces` arrays
    # copy-on-write, so we do NOT duplicate the data 60x in RAM.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        done = 0
        for _ in pool.imap_unordered(render_frame, range(n_times), chunksize=1):
            done += 1
            if done % 10 == 0 or done == n_times:
                logger.info(f"  frames done: {done}/{n_times}")

    # ── Stitch PNGs → MP4 (fast, high quality) ──
    mp4_out = f"{path_out}/llc_3D_{n_vars}x{n_models}-full.mp4"
    logger.info(f"Encoding MP4 with ffmpeg → {mp4_out}")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{FRAME_DIR}/frame_%05d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        mp4_out,
    ], check=True)

    # ── Optional: also produce a GIF (palette method = better quality) ──
    gif_out = f"{path_out}/llc_3D_{n_vars}x{n_models}-full.gif"
    logger.info(f"Encoding GIF with ffmpeg → {gif_out}")
    palette = f"{FRAME_DIR}/palette.png"
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{FRAME_DIR}/frame_%05d.png",
        "-vf", "palettegen",
        palette,
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{FRAME_DIR}/frame_%05d.png",
        "-i", palette,
        "-lavfi", "paletteuse",
        gif_out,
    ], check=True)

    logger.info(f"Done! {n_times} frames → {n_times/fps:.1f}s "
                f"(MP4: {mp4_out}, GIF: {gif_out})")

if __name__ == "__main__":
    main()