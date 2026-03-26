"""
This script computes tiled mixed layer depth (MLD) for a spatial subset of LLC4320.

Default behavior is data-first diagnostics:
1. Compute MLD from area-weighted tile-mean Theta/Salt.
2. Optionally apply temporal averaging.
3. Save compact zarr output for downstream statistics and plotting.
4. Plotting is optional and off by default.
"""

import logging
import os
import time
from pathlib import Path

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster
from fastjmd95 import jmd95numba
from scalene import scalene_profiler

# MLD settings
rho0 = 1025.0
kref = 6
DENS_THRES = 0.03

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def calc_MLD_col(theta, salt, z, rho0=rho0, kref=kref, dens_thres=DENS_THRES):
    """Calculate MLD for one vertical column."""
    rho = jmd95numba.rho(salt, theta, 0) - rho0
    drho = rho - rho[kref]

    mask = drho <= dens_thres
    if not np.any(mask):
        return np.nan
    return np.min(z[mask])


def area_weighted_tile_mean(var, area_cell, den, tile_lat, tile_lon):
    """Area-weighted mean over (j, i) into (tile_lat, tile_lon)."""
    var_cell = (
        var.assign_coords(tile_lat=tile_lat, tile_lon=tile_lon)
        .stack(cell=("j", "i"))
        .reset_index("cell")
    )

    num = (var_cell * area_cell).groupby(["tile_lat", "tile_lon"]).sum("cell")
    valid = den > 0
    return (num / den).where(valid)


def main():
    t0 = time.perf_counter()

    # SLURM + runtime flags
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "job")
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    slurm_mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))

    scalene_flag = env_bool("SCALENE_PROFILE", False)
    make_figures = env_bool("MLD_MAKE_FIGURES", False)

    # Spatial/temporal processing settings
    tile_width = float(os.environ.get("MLD_TILE_WIDTH", "0.25"))
    temporal_avg = os.environ.get("MLD_TEMPORAL_AVG", "1MS").strip()

    face = int(os.environ.get("MLD_FACE", "1"))
    i_0 = int(os.environ.get("MLD_I0", "2880"))
    i_1 = int(os.environ.get("MLD_I1", "3600"))
    j_0 = int(os.environ.get("MLD_J0", "720"))
    j_1 = int(os.environ.get("MLD_J1", "1440"))

    t_0 = int(os.environ.get("MLD_T0", "9216"))
    t_1 = int(os.environ.get("MLD_T1", "9960"))

    source_zarr = os.environ.get(
        "MLD_SOURCE_ZARR", "/orcd/data/abodner/003/LLC4320/LLC4320"
    )

    # Output settings
    data_dir = os.environ.get(
        "MLD_DATA_DIR", "/orcd/data/abodner/002/cody/MLD_diagnostic_data"
    )
    write_time_chunk = max(1, int(os.environ.get("MLD_WRITE_TIME_CHUNK", "24")))
    write_tile_chunk = max(1, int(os.environ.get("MLD_WRITE_TILE_CHUNK", "64")))
    max_figures = max(1, int(os.environ.get("MLD_MAX_FIGURES", "200")))

    # Dask worker settings
    n_workers = max(1, min(slurm_cpus, int(os.environ.get("MLD_N_WORKERS", "1"))))
    threads_per_worker = max(1, slurm_cpus // n_workers)

    if scalene_flag:
        scalene_profiler.start()

    logger.info("Initializing Dask")
    mem_gb = slurm_mem_mb / 1024 if slurm_mem_mb > 0 else 0.0
    worker_mem = None
    if mem_gb > 0:
        worker_mem = f"{0.8 * mem_gb / n_workers:.1f}GB"

    cluster_kwargs = {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "processes": True,
        "dashboard_address": ":0",
        "local_directory": os.environ.get("TMPDIR", "/tmp"),
    }
    if worker_mem is not None:
        cluster_kwargs["memory_limit"] = worker_mem

    cluster = LocalCluster(**cluster_kwargs)
    client = Client(cluster)
    logger.info("%s", client)

    exp_name = (
        f"{slurm_job_name},time=({t_0},{t_1}),"
        f"loc=i({i_0},{i_1}),j({j_0},{j_1}),"
        f"dt={temporal_avg},dxy={tile_width}"
    )
    logger.info("Experiment: %s", exp_name)
    logger.info(
        "Run settings: workers=%d, threads/worker=%d, figures=%s, temporal_avg=%s",
        n_workers,
        threads_per_worker,
        make_figures,
        temporal_avg,
    )
    logger.info("Source zarr: %s", source_zarr)

    try:
        # 1) Open subset
        llc = xr.open_zarr(source_zarr, consolidated=False)[
            ["Theta", "Salt", "Z", "XC", "YC", "rA"]
        ]

        subset_indexers = {
            "time": slice(t_0, t_1),
            "i": slice(i_0, i_1),
            "j": slice(j_0, j_1),
        }
        if "face" in llc.dims:
            subset_indexers["face"] = face

        llc_sub = llc.isel(**subset_indexers)

        # 2) Build tile labels from geographic coordinates
        lat_min = float(llc_sub.YC.min())
        lon_min = float(llc_sub.XC.min())

        tile_lat = ((llc_sub.YC - lat_min) / tile_width).astype("int32").compute()
        tile_lon = ((llc_sub.XC - lon_min) / tile_width).astype("int32").compute()

        # 3) Area-weighted spatial averaging into tiles
        area_cell = (
            llc_sub.rA.assign_coords(tile_lat=tile_lat, tile_lon=tile_lon)
            .stack(cell=("j", "i"))
            .reset_index("cell")
        )
        den = area_cell.groupby(["tile_lat", "tile_lon"]).sum("cell")

        theta_tile = area_weighted_tile_mean(
            llc_sub.Theta, area_cell, den, tile_lat, tile_lon
        ).transpose("time", "k", "tile_lat", "tile_lon")

        salt_tile = area_weighted_tile_mean(
            llc_sub.Salt, area_cell, den, tile_lat, tile_lon
        ).transpose("time", "k", "tile_lat", "tile_lon")

        # Static tile coordinates for downstream comparison/plotting
        xc_tile = area_weighted_tile_mean(
            llc_sub.XC, area_cell, den, tile_lat, tile_lon
        ).transpose("tile_lat", "tile_lon")
        yc_tile = area_weighted_tile_mean(
            llc_sub.YC, area_cell, den, tile_lat, tile_lon
        ).transpose("tile_lat", "tile_lon")

        # 4) MLD per tile/time
        mld_pixels = xr.apply_ufunc(
            calc_MLD_col,
            theta_tile,
            salt_tile,
            llc_sub.Z,
            input_core_dims=[["k"], ["k"], ["k"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
        ).rename("MLD_pixels")

        mld_pixels = mld_pixels.transpose("time", "tile_lat", "tile_lon")

        # Optional temporal averaging; use MLD_TEMPORAL_AVG=none for native hourly output
        if temporal_avg.lower() in {"none", "raw"}:
            mld_out = mld_pixels
        else:
            mld_out = mld_pixels.resample(time=temporal_avg).mean()

        mld_out = mld_out.astype(np.float32)

        time_chunk_out = min(write_time_chunk, mld_out.sizes["time"])
        tile_lat_chunk_out = min(write_tile_chunk, mld_out.sizes["tile_lat"])
        tile_lon_chunk_out = min(write_tile_chunk, mld_out.sizes["tile_lon"])

        mld_out = mld_out.chunk(
            {
                "time": time_chunk_out,
                "tile_lat": tile_lat_chunk_out,
                "tile_lon": tile_lon_chunk_out,
            }
        )

        xc_tile = xc_tile.astype(np.float32).chunk(
            {"tile_lat": tile_lat_chunk_out, "tile_lon": tile_lon_chunk_out}
        )
        yc_tile = yc_tile.astype(np.float32).chunk(
            {"tile_lat": tile_lat_chunk_out, "tile_lon": tile_lon_chunk_out}
        )

        ds_out = xr.Dataset({"MLD_pixels": mld_out, "XC": xc_tile, "YC": yc_tile})

        logger.info(
            "Compute graph prepared in %.3f minutes",
            (time.perf_counter() - t0) / 60,
        )

        # 5) Save compact zarr output
        t_write = time.perf_counter()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        store = f"{data_dir}/{exp_name}.zarr"

        compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
        encoding = {
            "MLD_pixels": {
                "dtype": "float32",
                "compressor": compressor,
                "chunks": (time_chunk_out, tile_lat_chunk_out, tile_lon_chunk_out),
            },
            "XC": {
                "dtype": "float32",
                "compressor": compressor,
                "chunks": (tile_lat_chunk_out, tile_lon_chunk_out),
            },
            "YC": {
                "dtype": "float32",
                "compressor": compressor,
                "chunks": (tile_lat_chunk_out, tile_lon_chunk_out),
            },
        }

        ds_out.to_zarr(store=store, mode="w", encoding=encoding)

        logger.info(
            "zarr storage time elapsed: %.3f minutes",
            (time.perf_counter() - t_write) / 60,
        )
        logger.info("data out: %s", store)

        # 6) Optional figures from saved output
        if make_figures:
            ds_plot = xr.open_zarr(store, consolidated=False)
            n_times = ds_plot.sizes["time"]

            if n_times > max_figures:
                logger.warning(
                    "Skipping figures: time length %d exceeds MLD_MAX_FIGURES=%d",
                    n_times,
                    max_figures,
                )
            else:
                outdir = Path(f"figs/{exp_name}")
                outdir.mkdir(parents=True, exist_ok=True)

                xc_min = float(ds_plot.XC.min())
                xc_max = float(ds_plot.XC.max())
                yc_min = float(ds_plot.YC.min())
                yc_max = float(ds_plot.YC.max())

                if temporal_avg.lower() in {"none", "raw"} or "H" in temporal_avg.upper():
                    time_fmt = "%Y-%m-%d %H:%M"
                elif "D" in temporal_avg.upper():
                    time_fmt = "%Y-%m-%d"
                else:
                    time_fmt = "%Y-%m"

                logger.info("Produce figures")
                for t in ds_plot.time.values:
                    label = pd.to_datetime(t).strftime(time_fmt)
                    logger.info("%s", label)
                    mld_sel = ds_plot["MLD_pixels"].sel(time=t).compute()

                    fig, ax = plt.subplots(figsize=(8, 5))
                    im = ax.imshow(
                        mld_sel,
                        extent=[xc_min, xc_max, yc_min, yc_max],
                        origin="lower",
                        cmap=cmocean.cm.deep_r,
                    )
                    plt.colorbar(im, ax=ax, label="MLD (m)")
                    ax.set_title(f"{exp_name} - {label}", fontsize=14)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    fig.savefig(outdir / f"{label}.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

        logger.info("total time elapsed: %.3f minutes", (time.perf_counter() - t0) / 60)

    finally:
        client.close()
        cluster.close()
        if scalene_flag:
            scalene_profiler.stop()


if __name__ == "__main__":
    main()
