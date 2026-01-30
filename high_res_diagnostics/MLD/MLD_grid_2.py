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

def main():

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
    

    n_workers=2
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
    logger.info(client)

    """
    2. Set params
    """
    logger.info('Set params')

    # set location
    face = 7
    # set temporal params
    t_0 = 2640#432
    t_1 = 3384 #t_0 + 24#(365*24) #JANUARY 

    
    # set tile width, temporal averaging
    tile_width = 0.25

    # horizontal subsets
    h_0 = 2000
    h_1 = h_0 + 540

    # exp name, data_dir
    exp_name = str(slurm_job_name) + f'_face{face}' + f'_({t_0},{t_1})'+f'_{tile_width}'+f'_({h_0,h_1})'
    #data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'

    logger.info(f'Experiment: {exp_name}')

    """
    3. open and subset LLC4320
    """


    # open LLC4320 and chunk: k should be full-column per chunk for .min(dim="k")
    LLC_face = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False, chunks={"time": 96,"k": -1,"i": 384,"j": 384,},)

    # select temporal extent, select face
    LLC_sub = LLC_face.isel(time=slice(t_0,t_1), face = face, i = slice(h_0,h_1), j = slice(h_0,h_1))[['Theta','Salt','Z','XC','YC','rA']]

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

    """
    7. Save as zarr
    """
    logger.info(f'Save as zarr')

    # define chunk encoding for zarr - match MDL_pixel chunking
 #   LLC_MLD = LLC_MLD.chunk({'time': 1, 'i': 384, 'j': 384})
 #   encoding = {"MLD_pixels": {"chunks": (1, 384, 384)},}


    ds_out = xr.Dataset({
        "MLD_pixels": LLC_MLD['MLD_pixels'],
        "rA": area,
        "XC": LLC_MLD["XC"],
        "YC": LLC_MLD["YC"],})
    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'
    ds_out.to_zarr(store = f"{data_dir}/{exp_name}.zarr",mode="w")#, encoding = encoding)

    logger.info(f'data out: {data_dir}/{exp_name}.zarr"')

    """
    6. Produce figures: time-averaged MLD heatmap with pixels=tile_width
    """

    # logger.info('Produce figure')
    
    # outdir = Path(f"figs/{exp_name}")
    # outdir.mkdir(parents=True, exist_ok=True)

    # for t in LLC_MLD.time.values:
    #     logger.info("t")
    #     # select and compute month
    #     MLD_tiles_sel = LLC_MLD.sel(time=t).compute()

    #     fig, ax = plt.subplots(figsize=(8,5))

    #     mld = ax.imshow(MLD_tiles_sel['MLD_pixels'],
    #     extent=[
    #             float(LLC_MLD.XC.min()), float(LLC_MLD.XC.max()),
    #             float(LLC_MLD.YC.min()), float(LLC_MLD.YC.max()),],
    #         origin="lower",cmap=cmocean.cm.deep_r)

    #     plt.colorbar(mld, ax=ax, label="MLD (m)")
    #     month_str = pd.to_datetime(t).strftime('%m-%Y')

    #     ax.set_title(f"{exp_name} – {month_str}", fontsize=14)

    #     ax.set_xlabel("Longitude")
    #     ax.set_ylabel("Latitude")

    #     fig.savefig(outdir / f"{month_str}.png", dpi=200, bbox_inches="tight")
    #     plt.close()



    


    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()