"""
This script calculates mixed layer depth (MLD) time-averaged MLD heatmaps of LLC4320 data using MLD methods in:
https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py

The methods are as follows:

0. Import dependencies, define tile/box helper functions
1. Initialize dask
2. Set params
3. Open and subset LLC4320
4. Subset into tiles
5. Follow code to calculate the MLD
6. group by tiles, weight by surface areas, take temporal means, and map i,j coordinate extents to tiles
7. Compute plotting reqs before figure production
8. Produce figure: time-averaged MLD heatmap with pixels=tile_width
9. Optionally save MLD-tile dataarray as netcdf for further analysis 

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

# build i,j,face index for a lat/lon spatial box about central lat/lon coord for llc4320
def llc_latlon_box_indices(
    LLC,
    lat_center,
    lon_center,
    degree_extent):

    half = degree_extent / 2.0

    lat_min = lat_center - half
    lat_max = lat_center + half
    lon_min = lon_center - half
    lon_max = lon_center + half

    XC = LLC["XC"]
    YC = LLC["YC"]

    face_boxes = {}

    for face in XC.face.values:
        xc = XC.sel(face=face)
        yc = YC.sel(face=face)

        # mask points inside the lat/lon box
        mask = (
            (yc >= lat_min) & (yc <= lat_max) &
            (xc >= lon_min) & (xc <= lon_max))

        if not mask.any():
            continue

        # get i/j indices where mask is True
        jj, ii = np.where(mask.values)

        j_start = int(jj.min())
        j_end   = int(jj.max()) + 1
        i_start = int(ii.min())
        i_end   = int(ii.max()) + 1

        face_boxes[int(face)] = (j_start, j_end, i_start, i_end)

    return face_boxes

# function to select i-j slices per box by lat/lon boxes
def i_j_slices(LLC, spacing):
    i_start, i_end = [], []
    j_start, j_end = [], []

    XC, YC = LLC['XC'].values, LLC['YC'].values

    # determine lat/lon min/max
    lon_min, lon_max = np.nanmin(XC), np.nanmax(XC)
    lat_min, lat_max = np.nanmin(YC), np.nanmax(YC)

    # number of boxes along lat/lon
    num_boxes_lon = int(np.ceil((lon_max - lon_min) / spacing))
    num_boxes_lat = int(np.ceil((lat_max - lat_min) / spacing))

    # loop over boxes in lat/lon
    for b_lon in range(num_boxes_lon):
        for b_lat in range(num_boxes_lat):
            # start/end lat/lon for this box
            lon_start = lon_min + b_lon * spacing
            lon_end   = lon_start + spacing
            lat_start = lat_min + b_lat * spacing
            lat_end   = lat_start + spacing

            # compute the distance to box corners
            dist_start = np.sqrt((XC - lon_start)**2 + (YC - lat_start)**2)
            dist_end   = np.sqrt((XC - lon_end)**2   + (YC - lat_end)**2)

            # nearest pixel indices
            j_s, i_s = np.unravel_index(np.nanargmin(dist_start), XC.shape)
            j_e, i_e = np.unravel_index(np.nanargmin(dist_end),   XC.shape)

            # sort to ensure proper i/j order
            i0, i1 = sorted([i_s, i_e])
            j0, j1 = sorted([j_s, j_e])

            i_start.append(i0)
            i_end.append(i1)
            j_start.append(j0)
            j_end.append(j1)

    return i_start[::-1], i_end[::-1], j_start, j_end, num_boxes_lon * num_boxes_lat


def iterate_boxes(LLC, resolution_deg):

    i_start, i_end, j_start, j_end, num_boxes = i_j_slices(LLC, resolution_deg)

    box_id = 0
    for i0, i1 in zip(i_start, i_end):
        for j0, j1 in zip(j_start, j_end):
            # only yield boxes with valid size
            if (i1 - i0) > 0 and (j1 - j0) > 0:
                yield box_id, (i0, i1, j0, j1)
                box_id += 1
                
# function to build a box index
def build_sub_index(LLC, resolution_deg,type):
    box = xr.full_like(LLC["XC"], fill_value=-1).astype(int)
    box = box.rename([type])

    for box_id, (i0, i1, j0, j1) in iterate_boxes(LLC, resolution_deg):
        box[j0:j1, i0:i1] = box_id
    return box


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

    # set location
    # ------------ 1 deg Kuroshio Extension centered @ 39°N, 158°E
    loc = 'Kuroshio'
    lat_center = 39
    lon_center = 158
    degree_extent = 10.0
    tile_width = 0.5


    # ------------ 1 deg Agulhas Current centered @ 43°S, 14°E
    # loc = 'Agulhas'
    # lat_center = -43
    # lon_center = 14
    # degree_extent = 1.0
    # tile_width = 0.5

    # ------------ 1 deg Gulf Stream centered @ 43°S, 14°E
    # loc = 'Gulf'
    # lat_center = 39
    # lon_center = -66
    # degree_extent = 1.0
    # tile_width = 0.5

    # set temporal params
    t_0 = 0
    t_1 = int(24*30)#int(429 * 24)
    time_window = int(24*30) # temporal resolution, currently set to 1 month

    # exp name
    exp_name = str(slurm_job_name) + f'_{loc}' + f'_{degree_extent}' + f'_{tile_width}' + f'_{time_window}' + f'_({t_0},{t_1})'

    logger.info(f'Experiment: {exp_name}')

    """
    3. open and subset LLC4320
    """

    # open LLC4320
    LLC_full = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

    # select [i,j] spatial box, face, temporal subset
    boxes = llc_latlon_box_indices(
    LLC_full,
    lat_center=lat_center,
    lon_center=lon_center,
    degree_extent=degree_extent)

    subs = []
    for face, (j0, j1, i0, i1) in boxes.items():
        subs.append(
            LLC_full.isel(face=face, j=slice(j0, j1), i=slice(i0, i1))
        )

    LLC_sub = xr.concat(subs, dim="face").isel(face=0) # can't handle multiple faces currently :(

    # select temporal extent, chunk: k should be full-column per chunk for .min(dim="k"
    LLC_sub = LLC_sub.isel(time=slice(t_0,t_1)).chunk({'time': time_window, 'k': -1,'i': 96, 'j': 96})

    """
    4. Subset into tiles
    """
    # select dims
    LLC_sub = LLC_sub[['Salt','Theta','XC','YC','Z','rA']]

    # break into half-deg tiles:
    # build tile index
    tile_width = 0.5
    tile_index = build_sub_index(LLC_sub, tile_width,"tile").compute()
    LLC_tiles = LLC_sub.assign_coords(tile=tile_index)
    # mask invalid pixels (the purple edges in the fig below)
    LLC_tiles = LLC_tiles.where(LLC_tiles.tile >= 0)
    

    """
    5. Follow code from https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py
        to calculate the MLD
    """
    # reference density 
    rho0 = 1025 #kg/m^3

    # potential density anomaly 
    # with the reference pressure of 0 dbar and ρ0 = 1000 kg m−3
    sigma0 = jmd95numba.rho(LLC_tiles.Salt, LLC_tiles.Theta,0) - rho0

    # sigma0 at 10m depth for reference, no broadcasting
    sigma0_10m = sigma0.isel(k=6)
    delta_sigma = sigma0 - sigma0_10m

    MLD_pixels = LLC_tiles.Z.broadcast_like(sigma0).where(delta_sigma<=0.03).min(dim="k",skipna=True)
    LLC_tiles['MLD'] = MLD_pixels

    """
    6. group by tiles, weight by surface areas, take temporal means, and map i,j coordinate extents to tiles
    """
    # divide into tiles, weight by surface areas
    area = LLC_tiles['rA']

    MLD_tiles = (
        (LLC_tiles.MLD * area)
        .groupby("tile")
        .sum()
        / area.groupby("tile").sum())

    # take temporal means
    MLD_timeavg_tiles = MLD_tiles.coarsen(time=time_window, boundary="trim").mean()
    # map i,j to tiles
    MLD_map = MLD_timeavg_tiles.sel(tile=LLC_tiles.tile)

    """
    7. Compute plotting reqs before figure production
    """

    logger.info('Compute plotting reqs')
    MLD_map = MLD_map.compute()

    """
    8. Produce figure: time-averaged MLD heatmap with pixels=tile_width
    """
    logger.info('Produce figure')

    outdir = Path(__file__).resolve().parent

    count = 0
    for t in MLD_map.time:
        count += 1
        fig, ax = plt.subplots(figsize=(6,5))

        MLD_map_sel = MLD_map.sel(time=t)

        contours = plt.imshow(MLD_map_sel)#ax.contourf(MLD_map_sel.i, MLD_map_sel.j, vals, cmap="Spectral_r",vmin=np.min(vals),vmax=np.max(vals), levels = 5)

        plt.colorbar(contours, ax = ax, label="MLD (m)")

        ax.set_title(f"{exp_name}" + f"_t-{count}", fontsize=14)

       # rotate_axes_90_clockwise(ax)

        fig.savefig(outdir / str(f"{exp_name}"+ f"_t-{count}.png"), dpi=200, bbox_inches="tight")
        plt.close()


    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()