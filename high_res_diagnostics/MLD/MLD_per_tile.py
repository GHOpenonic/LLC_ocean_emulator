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

    n_workers=3
    mem_gb = slurm_mem / 1024
    worker_mem = f"{0.9 * mem_gb / n_workers:.1f}GB"
    logger.info(f'{mem_gb}')
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker = slurm_cpus // n_workers,
        memory_limit=worker_mem,
        dashboard_address=None)
    client = Client(cluster)
    logger.info(client)

    """
    2. Set params
    """
    logger.info('Set params')

    # set tile width, temporal averaging
    tile_width = 0.5
    time_window = int(24*30) # temporal resolution, currently set to 1 month

    # exp name, data dir, and data
    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'
    data = 'MLD_Kuroshio_15.0_(0,8640)'
    exp_name = str(data) + f'_{tile_width}' + f'_{time_window}'

    logger.info(f'Experiment: {exp_name}')

    """
    3. Open MLD_per_pixel dataset
    """

    # open MLD_per_pixel
    MLD_per_pixel = xr.open_zarr(f'{data_dir}/{data}.zarr',consolidated=False)

    # chunk
    MLD_per_pixel = MLD_per_pixel.chunk({'time':time_window,'i':384,'j':384})

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