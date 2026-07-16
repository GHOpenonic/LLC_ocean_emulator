# dependencies
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import dask
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.animation as animation

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

import os
import multiprocessing as mp
from functools import partial

# IMPORTANT: headless backend, must be set before pyplot heavy use
import matplotlib
matplotlib.use("Agg")

FRAME_DIR = f"{path_out}/frames_{n_vars}x{n_models}"
os.makedirs(FRAME_DIR, exist_ok=True)

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
    {
        'name': 'rb-pred-resid-eager-ckpt-40',
        'key': 'emulator_1',
        'path': '/orcd/data/abodner/002/cody/inference_patch/2026-07-15-eval:Samudra_LLC:rb-Agulhas-pred_resid-reg-ckpt40-17993072/predictions_4d.zarr',
        'desc': ''
    },
    # {
    #     'name': 'rb-pred-resid-reg-ckpt-25',
    #     'key': 'emulator_2',
    #     'path': '/orcd/data/abodner/002/cody/inference_patch/2026-07-13-eval:Samudra_LLC:rb-Agulhas-pred_resid-reg-ckpt25-17850330/predictions_4d.zarr',
    #     'desc': ''
    # },
]

# ============== OPEN EMULATOR DATASETS ==============
emulator_patches_raw = {}
for cfg in emulator_configs:
    emulator_patches_raw[cfg['key']] = xr.open_dataset(cfg['path'], consolidated=True)
    print(f"Loaded {cfg['name']}: {cfg['desc']}")

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

print(f"LLC subset to {len(common_times)} common times")

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

print(f"\n=== Setup complete: LLC + {n_emulators} emulators ===")
for name, key in emulator_info:
    print(f"  {name} ({key})")

# ============== TIME SUBSET / SYNC ==============
selected_time_range = [0, 336]   # inclusive indices
stepping = 1                     # 1 = every timestep, 4 = every 4th timestep

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
print(f"Subset to time indices {start_idx}:{end_idx}")
print(f"Stepping = {stepping}")
print(f"Final synchronized length = {min_time_len}")
print(f"LLC now has {llc_patch.sizes['time']} times")
for name, key in emulator_info:
    print(f"{name} ({key}) now has {emulator_patches[key].sizes['time']} times")

def format_time(t_val):
    """Format a time value to DD/MM/YYYY:HH regardless of cftime or datetime64."""
    try:
        return f"{t_val.day:02d}/{t_val.month:02d}/{t_val.year}:{t_val.hour:02d}h"
    except AttributeError:
        t_pd = pd.Timestamp(t_val)
        return f"{t_pd.day:02d}/{t_pd.month:02d}/{t_pd.year}:{t_pd.hour:02d}h"

# ============================================================
# ============== VIDEO GENERATION ============================
# ============================================================

# ── Model (column) ordering: LLC first, then emulators ─────────────
model_order = [('LLC4320', 'llc')] + emulator_info      # [(display_name, key), ...]
n_models = len(model_order)                             # n = 1 + n_emulators

# ── Variable (row) selection ───────────────────────────────────────
# options: 'Theta', 'Salt', 'U', 'V'   (default: just Theta)
selected_variables = ['Theta']
# e.g. selected_variables = ['Theta', 'Salt', 'U', 'V']
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
i_0, i_1 = 0, 720
j_0, j_1 = 0, 720
k_max = 51
n_times = llc_patch.sizes['time']
fps = 24

# ── Pre-compute global color limits per variable (from LLC truth) ──
logger.info("Computing global color limits...")
sample_idx = np.linspace(0, n_times - 1, min(20, n_times), dtype=int)

norms = {}
for var in selected_variables:
    gmin, gmax = np.inf, -np.inf
    for t in sample_idx:
        tmp = llc_patch.isel(
            time=int(t), i=slice(i_0, i_1 + 1), j=slice(j_0, j_1 + 1), k=slice(0, k_max)
        )[var].values
        gmin = min(gmin, float(np.nanmin(tmp)))
        gmax = max(gmax, float(np.nanmax(tmp)))
    norms[var] = mcolors.Normalize(vmin=gmin, vmax=gmax)
    logger.info(f"  {var}: [{gmin:.4f}, {gmax:.4f}]")

# ── Pre-compute grid coordinates (shared across all patches) ───────
i_vals = np.arange(i_0, i_1 + 1)
j_vals = np.arange(j_0, j_1 + 1)
z_vals = llc_patch.isel(k=slice(0, k_max))["Z"].values

I_surf, J_surf = np.meshgrid(i_vals, j_vals)
Z_surf = np.full_like(I_surf, z_vals[0], dtype=float)

I_jwall, Z_jwall = np.meshgrid(i_vals, z_vals)
J_jwall = np.full_like(I_jwall, j_0, dtype=float)

J_iwall, Z_iwall = np.meshgrid(j_vals, z_vals)
I_iwall = np.full_like(J_iwall, i_1, dtype=float)

z_top, z_bot = float(z_vals[0]), float(z_vals[-1])
edge_kw = dict(color="k", linewidth=0.5)

# ── Set up v×n figure with 3D axes ─────────────────────────────────
fig = plt.figure(figsize=(5.5 * n_models, 5.0 * n_vars), dpi=100)

# ax_grid[(var, model_key)] = axes
ax_grid = {}
for r, var in enumerate(selected_variables):
    for c, (model_name, model_key) in enumerate(model_order):
        idx = r * n_models + c + 1
        ax = fig.add_subplot(n_vars, n_models, idx, projection='3d')
        ax_grid[(var, model_key)] = ax

    # one colorbar per row (variable), attached to rightmost column axis
    last_ax = ax_grid[(var, model_order[-1][1])]
    sm = plt.cm.ScalarMappable(cmap=cmaps[var], norm=norms[var])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=last_ax, fraction=0.026, pad=0.10, shrink=0.55)
    cbar.set_label(units[var], fontsize=8)

fig.suptitle('', fontsize=13, y=0.97)
fig.subplots_adjust(wspace=0.10, hspace=0.15)

# ── Helper: data → RGBA face colors ────────────────────────────────
def to_fc(data_2d, var):
    return cmaps[var](norms[var](data_2d))

# ── Draw box edges ──────────────────────────────────────────────────
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

# ── Animation update ───────────────────────────────────────────────
def update(frame_idx):

    # pre-slice each patch once per frame
    subsets = {}
    for _, model_key in model_order:
        subsets[model_key] = all_patches[model_key].isel(
            time=frame_idx,
            i=slice(i_0, i_1 + 1),
            j=slice(j_0, j_1 + 1),
            k=slice(0, k_max)
        )

    for r, var in enumerate(selected_variables):
        for c, (model_name, model_key) in enumerate(model_order):
            ax = ax_grid[(var, model_key)]
            ax.cla()

            data = subsets[model_key][var].values   # (k, j, i)
            surface = data[0, :, :]                  # top layer
            south   = data[:, 0, :]                  # j = j_0 wall
            east    = data[:, :, -1]                 # i = i_1 wall

            ax.plot_surface(I_surf, J_surf, Z_surf,
                            facecolors=to_fc(surface, var),
                            shade=False, rstride=1, cstride=1, zorder=3)
            ax.plot_surface(I_jwall, J_jwall, Z_jwall,
                            facecolors=to_fc(south, var),
                            shade=False, rstride=1, cstride=1, zorder=2)
            ax.plot_surface(I_iwall, J_iwall, Z_iwall,
                            facecolors=to_fc(east, var),
                            shade=False, rstride=1, cstride=1, zorder=1)

            draw_edges(ax)
            ax.view_init(elev=35, azim=-45)
            ax.set_xlabel("i", labelpad=6, fontsize=7)
            ax.set_ylabel("j", labelpad=6, fontsize=7)
            ax.set_zlabel("Depth [m]", labelpad=6, fontsize=7)
            ax.tick_params(labelsize=6)

            # column title (model name) on the top row
            if r == 0:
                ax.set_title(model_name, fontsize=11, pad=12)

            # row label (variable) on the leftmost column
            if c == 0:
                ax.text2D(-0.15, 0.5, var, transform=ax.transAxes,
                          fontsize=12, rotation=90,
                          va='center', ha='center')

    t_val = llc_patch.time.values[frame_idx]
    fig.suptitle(f"LLC4320 Agulhas  —  {format_time(t_val)}",
                 fontsize=13, y=0.97)

    logger.info(f"Frame {frame_idx + 1}/{n_times}")

# ── Render & save ──────────────────────────────────────────────────
logger.info(f"Rendering {n_times} frames at {fps} fps...")
ani = animation.FuncAnimation(fig, update, frames=n_times, interval=1000 / fps)
path_out = '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos/figs'
out_name = f'{path_out}/llc_3D_{n_vars}x{n_models}-full.gif'
ani.save(out_name, writer='pillow', fps=fps)
plt.close(fig)
logger.info(f"\nDone! Saved {n_times} frames → {n_times/fps:.1f}s GIF ({out_name})")