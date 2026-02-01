"""
This script calculates mixed layer depth (MLD) per pixel on a chunk of LLC4320 data using MLD methods in:
https://github.com/abodner/submeso_param_net/blob/main/scripts/preprocess_llc4320/preprocess.py
It is the first of two to calculate temporally averaged MLD on llc chunk

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
    # face = 7
    # set temporal params
    t_0 = 432
    t_1 = t_0 + (365*24) 

    # horizontal subsets
    # h_0 = 2000
    # h_1 = h_0 + 540

    # AGULHAS:
    j_0 = 800
    j_1 = j_0 + 540
    i_0 = 2500
    i_1 = i_0 + 540

    face = 1

    # exp name, data_dir
    exp_name = str(slurm_job_name) + f'_face{face}' + f'_({t_0},{t_1})'+f'_(Agulhas)'
    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'

    logger.info(f'Experiment: {exp_name}')

    """
    3. open and subset LLC4320
    """

    # open LLC4320 and chunk: k should be full-column per chunk for .min(dim="k")
    LLC_face = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False, chunks={"time": 96,"k": -1,"i": 384,"j": 384,},)

    # select temporal extent, select face
    LLC_sub = LLC_face.isel(time=slice(t_0,t_1), face = face, i = slice(i_0,i_1), j = slice(j_0,j_1))[['Theta','Salt','Z','XC','YC','rA']]

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

    area = LLC_sub.rA.chunk({"i": 384,"j": 384,}) 

    """
    5. Save as zarr
    """
    logger.info(f'Save as zarr')

    #rechunk
    MLD_pixels = MLD_pixels.chunk({"time": 96,"i": 384,"j": 384,})

    # define chunk encoding for zarr - match MDL_pixel chunking
    encoding = {"MLD_pixels": {"chunks": (96, 384, 384)},}


    ds_out = xr.Dataset({
        "MLD_pixels": MLD_pixels,
        "rA": area,
        "XC": LLC_sub["XC"].chunk({"i": 384,"j": 384,}),
        "YC": LLC_sub["YC"].chunk({"i": 384,"j": 384,}),})

    ds_out.to_zarr(store = f"{data_dir}/{exp_name}.zarr",mode="w", encoding = encoding)

    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()