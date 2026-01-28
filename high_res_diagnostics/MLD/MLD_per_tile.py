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
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
from scalene import scalene_profiler
import os
import pathlib
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

    n_workers=8
    mem_gb = slurm_mem / 1024
    worker_mem = f"{0.9 * mem_gb / n_workers:.1f}GB"
    logger.info(f'{mem_gb}GB')
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

    # exp name, data dir, and data
    data_dir = '/orcd/data/abodner/002/cody/MLD_per_pixel'
    data = 'MLD_face7_(432,9192)'
    exp_name = str(data) + f'_{tile_width}'

    logger.info(f'Experiment: {exp_name}')

    """
    3. Open MLD_per_pixel dataset
    """

    # open MLD_per_pixel
    MLD_per_pixel = xr.open_zarr(f'{data_dir}/{data}.zarr',consolidated=False)

    """
    4. Temporally coarsen, subset into tiles
    """

    MLD_per_pixel = MLD_per_pixel.resample(time="MS").mean() # divide into monthly, take the mean

    # chunk
    MLD_per_pixel = MLD_per_pixel.chunk({'time':1,'i':384,'j':384}) # monthly chunks

    # subset into tiles, weight by surface area
    YC = MLD_per_pixel.YC
    XC = MLD_per_pixel.XC
    area = MLD_per_pixel.rA.chunk({'i':384,'j':384})

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
    5. Produce figures: time-averaged MLD heatmap with pixels=tile_width
    """
    logger.info('Produce figure')

    pre_outdir = Path(__file__).resolve().parent

    
    outdir = pathlib.Path(f"figs/{exp_name}")  # or whatever path you want
    outdir.mkdir(parents=True, exist_ok=True)

    for t in MLD_tiles.time.values:
        # select and compute month
        MLD_tiles_sel = MLD_tiles.sel(time=t).compute()

        fig, ax = plt.subplots(figsize=(6,5))

        contours = ax.imshow(MLD_tiles_sel)#ax.contourf(MLD_map_sel.i, MLD_map_sel.j, vals, cmap="Spectral_r",vmin=np.min(vals),vmax=np.max(vals), levels = 5)

        plt.colorbar(contours, ax = ax, label="MLD (m)")

        ax.set_title(f"{exp_name} – {pd.to_datetime(t).strftime('%B %Y')}", fontsize=14)

       # rotate_axes_90_clockwise(ax)

        month_str = pd.to_datetime(t).strftime('%Y-%m')
        fig.savefig(outdir / f"{month_str}.png", dpi=200, bbox_inches="tight")
        plt.close()


    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()