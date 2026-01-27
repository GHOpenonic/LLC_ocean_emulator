"""
This script calculates mixed layer depth (MLD) per pixel of a spatiotemporal segment of LLC4320 data using MLD methods in:
https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py
It is the first of two to calculate temporally averaged MLD in spatial tiles

The methods are as follows:

0. Import dependencies, define tile/box helper functions
1. Initialize dask
2. Set params
3. Open and subset LLC4320
4. Follow code to calculate the MLD per pixel
5. save as zarr

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
    degree_extent = 1.0
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

    # exp name, data_dir
    exp_name = str(slurm_job_name) + f'_{loc}' + f'_{degree_extent}' + f'_{tile_width}' + f'_{time_window}' + f'_({t_0},{t_1})'
    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'

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
    4. Follow code from https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py
        to calculate the MLD per pixel
    """
    # reference density 
    rho0 = 1025 #kg/m^3

    # potential density anomaly 
    # with the reference pressure of 0 dbar and ρ0 = 1000 kg m−3
    sigma0 = jmd95numba.rho(LLC_sub.Salt, LLC_sub.Theta,0) - rho0

    # sigma0 at 10m depth for reference, no broadcasting
    sigma0_10m = sigma0.isel(k=6)
    delta_sigma = sigma0 - sigma0_10m

    MLD_pixels = LLC_sub.Z.broadcast_like(sigma0).where(delta_sigma<=0.03).min(dim="k",skipna=True)

    """
    5. Save as zarr
    """

    MLD_pixels.to_dataset(name="MLD_pixels").to_zarr(store = f"{data_dir}/{exp_name}.zarr",mode="w")

    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()