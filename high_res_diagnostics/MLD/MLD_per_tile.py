"""
This script takes mixed layer depth (MLD) per pixel calculations previously performed in MLD_per_pixel.py 
and calculates and plots time-averaged MLD heatmaps of LLC4320 data.

The methods are as follows:

0. Import dependencies, define tile/box helper functions
1. Initialize dask
2. Set params
3. Open MLD_per_pixel dataset
4. temporally coarsen, subset into tiles
7. Compute plotting reqs before figure production
8. Produce figure: time-averaged MLD heatmap with pixels=tile_width

"""

"""
0. Import dependencies, define helper functions
"""
# dependencies
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
import dask
from dask.distributed import Client, LocalCluster
from fastjmd95 import jmd95numba 
from scalene import scalene_profiler
import os
import pathlib
import matplotlib.pyplot as plt
from pathlib import Path
from xhistogram.xarray import histogram

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

# define functions
# first define a helper function to rotate plots for N-S/vertical alignment
def rotate_axes_90_clockwise(ax):
    """
    Robustly rotate a Matplotlib Axes 90 degrees clockwise by:
      - rendering the figure to an RGBA buffer,
      - cropping the pixels belonging to the given Axes,
      - rotating the image 90 deg clockwise,
      - placing the rotated image back into the figure at the same axes position,
      - removing the original Axes.

    Notes:
    - This rasterizes the axis contents (the result is an image, not vector art).
    - Colorbars or other axes that live outside the target `ax` are left alone.
    - Works reliably for full grids or arbitrary subsets.
    """
    fig = ax.figure

    # Force draw so renderer & sizes are correct
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get pixel bbox of the axes (in display coordinates) via renderer-aware call
    bbox = ax.get_window_extent(renderer=renderer)  # Bbox in display (pixel) coords

    # Grab the full figure RGBA buffer (H x W x 4)
    buf = np.asarray(renderer.buffer_rgba())  # returns an (H,W,4) uint8 array

    H, W = buf.shape[0], buf.shape[1]

    # Convert bbox to integer pixel coords and crop.
    # Note: Bbox y coords are in display coordinates with origin at lower-left.
    x0, x1 = int(np.floor(bbox.x0)), int(np.ceil(bbox.x1))
    y0, y1 = int(np.floor(bbox.y0)), int(np.ceil(bbox.y1))

    # Convert to numpy row indices (origin at top-left)
    row0 = max(0, H - y1)
    row1 = min(H, H - y0)
    col0 = max(0, x0)
    col1 = min(W, x1)

    if row1 <= row0 or col1 <= col0:
        raise RuntimeError("Calculated zero-size axes crop — renderer/coords inconsistent.")

    axes_img = buf[row0:row1, col0:col1, :].copy()  # copy to be safe

    # Rotate 90 degrees clockwise. np.rot90 rotates counterclockwise, so use k=-1 (or k=3).
    rotated = np.rot90(axes_img, k=-1)

    # Create a new axes in the same figure position (figure coords) and show the rotated image.
    # We must compute the original axes position in figure coordinates:
    fig_x0, fig_y0, fig_w, fig_h = ax.get_position().bounds

    # Add overlaid axes and show the rotated image
    new_ax = fig.add_axes([fig_x0, fig_y0, fig_w, fig_h], anchor='C')  # same place
    new_ax.imshow(rotated, origin='upper', aspect='auto')
    new_ax.set_axis_off()

    # Remove the original axes (so only rotated image remains)
    fig.delaxes(ax)

    # Force redraw
    fig.canvas.draw()
    return new_ax

def main():

    logger.info('Initializing Dask')

    """
    1. Initialize dask, scalene if flagged
    """

    # get SLURM environment variables, flags
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "job")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "0")
    SLURM_CPUS_PER_TASK = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    scalene_flag = os.environ.get("SCALENE_PROFILE", "True").lower() in ("True")

    if scalene_flag:
        # begin memory profiling
        scalene_profiler.start()

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker = 8,
        memory_limit="120GB",
        dashboard_address=None)
    client = Client(cluster)
    logger.info(client)

    """
    2. Set params
    """
    logger.info('Set params')

    # set tile width
    tile_width = 0.5

    time_window = int(24*30) # temporal resolution, currently set to 1 month

    # exp name, data dir, and data
    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'
    data = 'MLD__Kuroshio_1.0_(0,720)'
    exp_name = str(data) + f'_{tile_width}' + f'_{time_window}'

    logger.info(f'Experiment: {exp_name}')

    """
    3. Open MLD_per_pixel dataset
    """

    # open MLD_per_pixel
    MLD_per_pixel = xr.open_zarr(f'{data_dir}/{data}.zarr',consolidated=False)

    # chunk
    MLD_per_pixel = MLD_per_pixel.chunk({'time':time_window,'i':96,'j':96})
    """
    4. Temporally coarsen, subset into tiles
    """
    MLD_per_pixel = MLD_per_pixel.coarsen(time=time_window, boundary="trim").mean()
    area = MLD_per_pixel.rA

    # subset into tiles, weight by surface area
    YC = MLD_per_pixel.YC
    XC = MLD_per_pixel.XC
    area = MLD_per_pixel.rA

    lat_edges = np.arange(YC.values.min(), YC.values.max() + tile_width, tile_width)
    lon_edges = np.arange(XC.values.min(), XC.values.max() + tile_width, tile_width)

    num = histogram(
        YC,
        XC,
        bins=[lat_edges, lon_edges],
        weights=MLD_per_pixel['MLD_pixels'] * area,
        dim=("j", "i"))

    den = histogram(
        YC,
        XC,
        bins=[lat_edges, lon_edges],
        weights=area,
        dim=("j", "i"))

    MLD_tiles = num / den

    """
    5. Compute plotting reqs before figure production
    """

    logger.info('Compute plotting reqs')
    MLD_tiles = MLD_tiles.compute()

    """
    6. Produce figures: time-averaged MLD heatmap with pixels=tile_width
    """
    logger.info('Produce figure')

    pre_outdir = Path(__file__).resolve().parent

    
    outdir = pathlib.Path(f"figs/{exp_name}")  # or whatever path you want
    outdir.mkdir(parents=True, exist_ok=True)

    for t in MLD_tiles.time.values:
        fig, ax = plt.subplots(figsize=(6,5))

        MLD_tiles_sel = MLD_tiles.sel(time=t)

        contours = plt.imshow(MLD_tiles_sel)#ax.contourf(MLD_map_sel.i, MLD_map_sel.j, vals, cmap="Spectral_r",vmin=np.min(vals),vmax=np.max(vals), levels = 5)

        plt.colorbar(contours, ax = ax, label="MLD (m)")

        ax.set_title(f"{exp_name}" + f"_t-{t}", fontsize=14)

       # rotate_axes_90_clockwise(ax)

        fig.savefig(outdir / str(f"_t-{t}.png"), dpi=200, bbox_inches="tight")
        plt.close()


    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()