"""
This script calculates mixed layer depth (MLD) per spatially averaged tile, then averages temporally, on a face of LLC4320 data using MLD methods in:
https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py

The methods are as follows:

0. Import dependencies, define tile/box helper functions
1. Initialize dask
2. Set params
3. Open and subset LLC4320
 subset into tiles
4. Follow code to calculate the MLD per pixel
5. Temporally coarsen, 
6. save as zarr

"""

"""
0. Import dependencies, define helper functions
"""
# dependencies
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import zarr
from dask.distributed import Client, LocalCluster
from fastjmd95 import jmd95numba 
from scalene import scalene_profiler
import os
from xhistogram.xarray import histogram
from pathlib import Path
import time


# calculate mld per column
rho0 = 1025 #ref den in kg/m^3
kref = 6 # 10m
dens_thres = 0.03 
def calc_MLD_col(theta, salt, z, rho0=rho0, kref=kref, dens_thres=dens_thres):
    # theta, salt, z are (k,)

    rho = jmd95numba.rho(salt, theta, 0) - rho0
    drho = rho - rho[kref]

    mask = drho <= dens_thres

    if not np.any(mask):
        return np.nan
    return np.min(z[mask])

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

# build i,j,face index for a lat/lon spatial box about central lat/lon coord for llc4320
def llc_latlon_box_indices(
    LLC,
    lat_center,
    lon_center,
    degree_extent
):

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
            (xc >= lon_min) & (xc <= lon_max)
        )

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

        
    t0 = time.perf_counter()
    logger.info(f"time elapsed: {(time.perf_counter() - t0)/60:.3f} minutes")

    logger.info('Initializing Dask')

    """
    1. Initialize dask, scalene if flagged
    """


    # get SLURM environment variables, flags
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "job")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "0")
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    slurm_mem = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    scalene_flag = os.environ.get("SCALENE_PROFILE", "True").lower() in ("True")

    if scalene_flag:
        # begin memory profiling
        scalene_profiler.start()
    

    n_workers=1
    mem_gb = slurm_mem / 1024
    logger.info(f'{mem_gb}GB')
    worker_mem = f"{0.9 * mem_gb / n_workers:.1f}GB"
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker = slurm_cpus // n_workers,
        memory_limit=worker_mem,
        dashboard_address=None,
        local_directory="/tmp")
    client = Client(cluster)
    logger.info(f'{client}')

    """
    2. Set params
    """
    logger.info('Set params')

                        # # set location
                        # # face = 7
                        # # set temporal params
                        # t_0 = 432
                        # t_1 = t_0 + int(365*24) #yr 

                        
                        # # set tile width, temporal averaging
                        # tile_width = 0.25

                        # # horizontal subsets
                        # # h_0 = 2000
                        # # h_1 = h_0 + 540

                        # # AGULHAS:
                        # j_0 = 800
                        # j_1 = j_0 + 540
                        # i_0 = 2500
                        # i_1 = i_0 + 540

                        # face = 1


     # set size of tile in degrees lat/lon, sets FFT tile sizes, 
    # set size of sub-tile boxes in lat/lon, set i,j extents of the spatial box
    # ------------ 1 deg Kuroshio Extension centered @ 39°N, 158°E
    # loc = 'Kuroshio'
    # lat_center = 39
    # lon_center = 158
    # extent = 1.0
    # buffer = 0 # a little greater than 1 allows tile_width to trim to 4 sub-panels of exactly 0.5 x 0.5 deg^2 = 1 x 1 deg^2
    # degree_extent = extent + buffer
    # tile_width = 0.5

    # ------------ 1 deg Agulhas Current centered @ 43°S, 14°E
    loc = 'Agulhas'
    lat_center = -43
    lon_center = 14
    extent = 1.0
    buffer = 0 # a little greater than 1 allows tile_width to trim to 4 sub-panels of exactly 0.5 x 0.5 deg^2 = 1 x 1 deg^2
    degree_extent = extent + buffer
    tile_width = 0.25

    # ------------ 1 deg Gulf Stream centered @ 43°S, 14°E
    # loc = 'Gulf'
    # lat_center = 39
    # lon_center = -66
    # extent = 1.0
    # buffer = 0 # a little greater than 1 allows tile_width to trim to 4 sub-panels of exactly 0.5 x 0.5 deg^2 = 1 x 1 deg^2
    # degree_extent = extent + buffer
    # tile_width = 0.5



    # temporal extent of the calculation/time series
    t_0 =  0 #4000
    t_1 =  24#int(429 * 24)# t_0 +  24 * 4 * 4#

    # exp name, data_dir
    exp_name = str(slurm_job_name) + f'_({t_0},{t_1})'+f'_{loc}'
    #data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'

    logger.info(f'Experiment: {exp_name}')

    """
    3. open and subset LLC4320
    """


    # open LLC4320 and chunk: k should be full-column per chunk for .min(dim="k")
    LLC_full = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False, chunks={"time": 96,"k": -1,"i": -1,"j": -1,},)

                        # # select temporal extent, select face
                        # LLC_sub = LLC_face.isel(time=slice(t_0,t_1), face = face, i = slice(i_0,i_1), j = slice(j_0,j_1))[['Theta','Salt','Z','XC','YC','rA']]


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

    LLC_sub = xr.concat(subs, dim="face")

    # select temporal extent
    LLC_sub = LLC_sub.isel(time=slice(t_0,t_1))[['Theta','Salt','Z','XC','YC','rA']]


    """
    SUBSET INTO TILES
    """
    # subset into tiles, weight by surface area    
    YC = LLC_sub.YC 
    XC = LLC_sub.XC 
    area = LLC_sub.rA.chunk({'i':384,'j':384}) 
    lat_min = float(YC.min()) 
    lat_max = float(YC.max()) 
    lon_min = float(XC.min()) 
    lon_max = float(XC.max())
    
    # compute tile labels eagerly (small arrays)
    tile_lat = ((LLC_sub.YC - lat_min) / tile_width).astype("int32").compute()
    tile_lon = ((LLC_sub.XC - lon_min) / tile_width).astype("int32").compute()

   # stack dataset
    LLC_tile = LLC_sub.assign_coords(
        tile_lat=tile_lat,
        tile_lon=tile_lon,
    ).stack(cell=("j","i"))

    # stack area
    area_cell = LLC_sub.rA.stack(cell=("j","i"))

    # attach tile labels to area
    area_cell = area_cell.assign_coords(
        tile_lat=LLC_tile.tile_lat,
        tile_lon=LLC_tile.tile_lon,
    )

    num = (LLC_tile * area_cell).groupby(["tile_lat","tile_lon"]).sum("cell")
    den = area_cell.groupby(["tile_lat","tile_lon"]).sum("cell")

    LLC_tile = num / den


    """
    4. Calculate MLD per pixel
    """

    MLD_pixels = xr.apply_ufunc( # use ufunc along single columns to manage memory
        calc_MLD_col,
        LLC_tile.Theta,
        LLC_tile.Salt,
        LLC_tile.Z,
        input_core_dims=[["k"], ["k"], ["k"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],)

    LLC_MLD = LLC_tile.copy()
    LLC_MLD['MLD_pixels'] = MLD_pixels
    
    LLC_MLD = LLC_MLD.resample(time="MS").mean() # divide into monthly, take the mean



    logger.info(f"code time elapsed: {(time.perf_counter() - t0)/60:.3f} minutes")


    """
    6. Produce figures: time-averaged MLD heatmap with pixels=tile_width
    """

    logger.info('Produce figure')
    
    outdir = Path(f"figs/{exp_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    for t in LLC_MLD.time.values:
        logger.info("t")
        # select and compute month
        MLD_tiles_sel = LLC_MLD.sel(time=t).compute()

        fig, ax = plt.subplots(figsize=(8,5))

        mld = ax.imshow(MLD_tiles_sel['MLD_pixels'],
        extent=[
                float(LLC_MLD.XC.min()), float(LLC_MLD.XC.max()),
                float(LLC_MLD.YC.min()), float(LLC_MLD.YC.max()),],
            origin="lower",cmap=cmocean.cm.deep_r)

        plt.colorbar(mld, ax=ax, label="MLD (m)")
        month_str = pd.to_datetime(t).strftime('%m-%Y')

        ax.set_title(f"{exp_name} – {month_str}", fontsize=14)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.savefig(outdir / f"{month_str}.png", dpi=200, bbox_inches="tight")
        plt.close()


    logger.info(f"code + figure time elapsed: {(time.perf_counter() - t0)/60:.3f} minutes")



    """
    7. Save as zarr
    """
    logger.info(f'Save as zarr')\
    

    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'
    LLC_MLD.to_zarr(store = f"{data_dir}/{exp_name}.zarr",mode="w")#, encoding = encoding)
    logger.info(f"zarr storage time elapsed: {(time.perf_counter() - t1)/60:.3f} minutes")
    logger.info(f"total time elapsed: {(time.perf_counter() - t0)/60:.3f} minutes")

    logger.info(f'data out: {data_dir}/{exp_name}.zarr"')





    


    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()