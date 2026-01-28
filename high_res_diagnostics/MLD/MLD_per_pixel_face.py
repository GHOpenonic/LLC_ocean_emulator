"""
This script calculates mixed layer depth (MLD) per pixel on a face of LLC4320 data using MLD methods in:
https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py
It is the first of two to calculate temporally averaged MLD on llc faces

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
import numpy as np
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster
from fastjmd95 import jmd95numba 
from scalene import scalene_profiler
import os

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
    

    n_workers=4
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
    t_1 = t_0 + (365*24) 

    # exp name, data_dir
    exp_name = str(slurm_job_name) + f'_face{face}' + f'_({t_0},{t_1})'
    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'

    logger.info(f'Experiment: {exp_name}')

    """
    3. open and subset LLC4320
    """

    # open LLC4320 and chunk: k should be full-column per chunk for .min(dim="k")
    LLC_face = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False, chunks={"time": 240,"k": -1,"i": 540,"j": 540,},)

    # select temporal extent, select face
    LLC_sub = LLC_face.isel(time=slice(t_0,t_1), face = face)

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

    MLD_pixels = LLC_sub.Z.where(delta_sigma <= 0.03).min("k")
    area = LLC_sub.rA

    """
    5. Save as zarr
    """
    logger.info(f'Save as zarr')

    encoding = {"MLD_pixels": {"chunks": (240, 540, 540)},}

    #rechunk
    MLD_pixels = MLD_pixels.chunk({"time": 240,"i": 540,"j": 540,})

    ds_out = xr.Dataset({
        "MLD_pixels": MLD_pixels,
        "rA": area,
        "XC": LLC_sub["XC"],
        "YC": LLC_sub["YC"],})

    ds_out.to_zarr(store = f"{data_dir}/{exp_name}.zarr",mode="w", encoding = encoding)

    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()