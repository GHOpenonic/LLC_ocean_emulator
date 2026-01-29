"""
This script calculates mixed layer depth (MLD) per tile, then plots, on a face of LLC4320 data using MLD methods in:
https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py

The methods are as follows:

0. Import dependencies, define tile/box helper functions
1. Initialize dask
2. Set params
3. Open and subset LLC4320
4. Follow code to calculate the MLD per pixel
5. Temporally coarsen, subset into tiles
6. Produce figs

"""

"""
0. Import dependencies, define helper functions
"""
# dependencies
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
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
    t_0 = 432
    t_1 = t_0 + 24 #(365*24) 

    
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
    LLC_sub = LLC_face.isel(time=slice(t_0,t_1), face = face, i = slice(h_0,h_1), j = slice(h_0,h_1))[['Theta','Salt','Z','XC','YC']]

    """
    4. Calculate MLD per pixel
    """

    MLD_pixels = xr.apply_ufunc( # use ufunc along single columns to manage memory
        calc_MLD_col,
        LLC_sub.Theta,
        LLC_sub.Salt,
        LLC_sub.Z,
        input_core_dims=[["k"], ["k"], ["k"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],)

    LLC_MLD = LLC_sub.copy()
    LLC_MLD['MLD_pixels'] = MLD_pixels
    
    LLC_MLD = LLC_MLD.resample(time="MS").mean() # divide into monthly, take the mean

    # rechunk
    LLC_MLD = LLC_MLD.chunk({'time':1,'i':384,'j':384}) # monthly chunks

    # subset into tiles, weight by surface area
    YC = LLC_MLD.YC
    XC = LLC_MLD.XC
    area = LLC_MLD.rA.chunk({'i':384,'j':384})

    lat_min = float(YC.min())
    lat_max = float(YC.max())
    lon_min = float(XC.min())
    lon_max = float(XC.max())

    lat_edges = np.arange(lat_min, lat_max + tile_width, tile_width)
    lon_edges = np.arange(lon_min, lon_max + tile_width, tile_width)

    num = histogram(
        YC,
        XC,
        bins=[lat_edges, lon_edges],
        weights=llc_MLD['MLD_pixels'] * area,
        dim=("j", "i"))

    den = histogram(
        YC,
        XC,
        bins=[lat_edges, lon_edges],
        weights=area,
        dim=("j", "i"))

    MLD_tiles = num / den

    """
    6. Produce figures: time-averaged MLD heatmap with pixels=tile_width
    """

    logger.info('Produce figure')
    
    outdir = Path(f"figs/{exp_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    for t in MLD_tiles.time.values:
        # select and compute month
        MLD_tiles_sel = MLD_tiles.sel(time=t).compute()

        fig, ax = plt.subplots(figsize=(6,5))

        contours = ax.imshow(MLD_tiles_sel)#ax.contourf(MLD_map_sel.i, MLD_map_sel.j, vals, cmap="Spectral_r",vmin=np.min(vals),vmax=np.max(vals), levels = 5)

        plt.colorbar(contours, ax = ax, label="MLD (m)")

        ax.set_title(f"{exp_name} – {pd.to_datetime(t).strftime('%B %Y')}", fontsize=14)

       # rotate_axes_90_clockwise(ax)

        month_str = pd.to_datetime(t).strftime('%m-%Y')
        fig.savefig(outdir / f"{month_str}.png", dpi=200, bbox_inches="tight")
        plt.close()

    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()