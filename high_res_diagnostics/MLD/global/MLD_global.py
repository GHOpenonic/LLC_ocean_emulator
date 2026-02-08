"""
This script calculates mixed layer depth (MLD) per pixel on a spatiotemporal chunk of LLC4320 data
It will be used iteratively to compute and save MLD per pixel for a test subset of the LLC4320 dataset

The methods are as follows:

0. Import dependencies
1. Initialize dask
2. Set params
3. Open and subset LLC4320
4. Calculate the MLD per pixel
5. Append to zarr

"""

"""
0. Import dependencies, define helper functions
"""
# dependencies
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster
from fastjmd95 import jmd95numba 
import numba
from scalene import scalene_profiler
from pathlib import Path
import time

# calculate mld per column
rho0 = 1025 #ref den in kg/m^3
kref = 6 # 10m
dens_thres = 0.03 
@numba.guvectorize([(numba.float32[:], numba.float32[:], numba.float32[:], numba.float32[:])], 
                 '(k),(k),(k)->()', nopython=True)
def calc_MLD_col(theta, salt, z, out):
    # Hardcoded constants for the Numba kernel
    _rho0 = 1025.0
    _kref = 6 
    _dens_thres = 0.03

    # Calculate density at reference depth once
    rho_ref = jmd95numba.rho(salt[_kref], theta[_kref], 0.0) - _rho0
    
    # Initialize output with NaN
    out[0] = np.nan
    
    # Manual loop for Numba efficiency
    for k in range(len(z)):
        rho_k = jmd95numba.rho(salt[k], theta[k], 0.0) - _rho0
        drho = rho_k - rho_ref
        
        if drho <= _dens_thres:
            if np.isnan(out[0]) or z[k] < out[0]:
                out[0] = z[k]

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)


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
    

    n_workers=7
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
    # write exp_n ---------------------------------------------
    exp_n = 1


    # temporal extent of the calculation/time series
    t_llc_offset = 432
    t_iter = 1460#(365 * 24)
    t_0 = t_llc_offset + exp_n * t_iter
    t_1 = t_0 + t_iter 

    # face
    #face = 7

    # FOR TESTING: horizontal slices:
    h_0, h_1  = 0, 4320

    # exp name
    exp_name = str(slurm_job_name) +f'_exp:{exp_n}'+ f'_({t_0},{t_1})'+f'_all_faces' #+ f'_{h_0,},{h_1}'

    logger.info(f'Experiment: {exp_name}')

    """
    3. open and subset LLC4320
    """

    # open LLC4320 and slice to correct time slice and face
    LLC_sub = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)[['Theta', 'Salt', 'Z','XC','YC']].isel(time=slice(t_0,t_1), i = slice(h_0, h_1),j=slice(h_0,h_1))

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
        vectorize = False,
        dask="parallelized",
        output_dtypes=[np.float32],)

    logger.info(f"MLD calculation time elapsed: {(time.perf_counter() - t0)/60:.3f} minutes")

    """
    5. Append to zarr
    """
    t1 = time.perf_counter()
    logger.info(f'Append to zarr')

    mld_4d = MLD_pixels#.expand_dims(face=[face]) 

    # Reorder to match: (time, face, j, i)
    mld_4d = mld_4d.transpose("time", "face", "j", "i")

    MLD_intermediary = xr.Dataset({"MLD": mld_4d.drop_vars(['XC', 'YC', 'Z'], errors='ignore')}).chunk({"time": 730})


    MLD_intermediary.to_zarr(
        "/orcd/data/abodner/002/cody/MLD_llc4320/MLD_ds.zarr",
        region={
            "time": slice(t_0 - t_llc_offset, t_1 - t_llc_offset),
            "face": slice(0,13),
            "j": slice(0, h_1 - h_0),
            "i": slice(0, h_1 - h_0),
        },
        zarr_format = 2
    )

    logger.info(f"zarr storage time elapsed: {(time.perf_counter() - t1)/60:.3f} minutes")
    logger.info(f"total time elapsed: {(time.perf_counter() - t0)/60:.3f} minutes")

    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()