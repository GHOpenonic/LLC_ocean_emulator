"""
This diagnostic script calculates the vertical heat flux (VHF) through a layer of water 
in low-frequency (LF), high-frequency (HF), and total (LF + HF) components, 
then produces a time series figure of these VHF components. These calculations 
are performed in a specified spatial box in the LLC4320 dataset. 
The methods are as follows:

0. Import dependencies, define tile/box helper functions
1. Initialize dask
2. Set params
3. Open and subset LLC4320
4. tiling and vertical velocity interpolation w/ grid_3d.interp, this is necessary as Theta values are spatially centered within LLC pixels while W (vertical velocity) 
    values are horizontally centered, but vertically shifted to the edges of LLC pixels. Tiling spatially enables spatial mean calculation and breaks fft computation up to fit in memory
5. calculate the total VHF with $C_p \rho <W'T'>$ for each time t, where <W'T'> = bar(WT) - bar(W) bar(T), where <W'T'> is the heat transport due to correlated fluctuations in W and T,
    bar(WT) is the total vertical heat transport (both mean and correlated fluctuations), and bar(W) bar(T) is the heat transport due to the mean vertical motion carrying mean temperature.
6. define LF/HF masks
7. LF, HF VHF calculation
8. calculate sum of LF and HF, apply a rolling mean for plotting
9. Compute plotting reqs before figure production
10. Produce figure
11. Optionally save VHF time series as netcdf for further analysis 

Note on fft(W,T) fields:
filtering W and T fields separately before calculating <WT> isolates the VHF carried by the low and high frequency motions themselves,
which allows for explicit separation of the transport mechanism by scale.
"""

"""
0. Import dependencies, define helper functions
"""

import xarray as xr
import numpy as np
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import zarr
import xgcm 
from scipy.signal.windows import tukey

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

# function to select i-j slices per box by lat/lon boxes
def i_j_slices(LLC, spacing):
    i_start, i_end = [], []
    j_start, j_end = [], []

    XC, YC= LLC['XC'].values, LLC['YC'].values

  #  i_min, j_min = np.min(LLC['i'].values), np.min(LLC['j'].values)

    extent = (np.nanmax(XC)) - (np.nanmin(XC))
    num_boxes = extent/spacing 

    lon_min, lat_min= np.nanmin(XC), np.nanmin(YC)

    for box in range(int(num_boxes)):
        # start lat/lon
        lon_start, lat_start = lon_min + spacing * box, lat_min + spacing * box

        # end lat/lon
        lon_end, lat_end = lon_min + spacing * box + spacing, lat_min + spacing * box + spacing

        # convert to i,j
        lon_dist_s, lat_dist_s = np.nanargmin(np.abs(XC - lon_start)),  np.nanargmin(np.abs(YC - lat_start))
        (_,i_s), (j_s,_) = np.unravel_index(lat_dist_s, YC.shape), np.unravel_index(lon_dist_s, XC.shape)

        lon_dist_e, lat_dist_e = np.nanargmin(np.abs(XC - lon_end)),  np.nanargmin(np.abs(YC - lat_end))
        (_,i_e), (j_e,_) = np.unravel_index(lat_dist_e, YC.shape), np.unravel_index(lon_dist_e, XC.shape)


        i_start.append(int(i_e)), i_end.append(int(i_s)), j_start.append(int(j_s)), j_end.append(int(j_e))
   # print (f'i: {i_start[::-1]}, {i_end[::-1]}, j: {j_start}, {j_end}, {int(num_boxes)}')
    return i_start[::-1], i_end[::-1], j_start, j_end, int(num_boxes)


def iterate_boxes(LLC, resolution_deg):

    i_start, i_end, j_start, j_end, num_boxes = i_j_slices(LLC, resolution_deg)

    box_id = 0
    for i0, i1 in zip(i_start, i_end):
        for j0, j1 in zip(j_start, j_end):
            # only yield boxes with valid size
            if (i1 - i0) > 0 and (j1 - j0) > 0:
                yield box_id, (i0, i1, j0, j1)
                box_id += 1
                
# function to build a box index
def build_sub_index(LLC, resolution_deg,type):
    box = xr.full_like(LLC["XC"], fill_value=-1).astype(int)
    box = box.rename([type])

    for box_id, (i0, i1, j0, j1) in iterate_boxes(LLC, resolution_deg):
        box[j0:j1, i0:i1] = box_id
    return box

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
        n_workers=3,
        threads_per_worker = 8,
        memory_limit="128GB",
        dashboard_address=None)
    client = Client(cluster)
    logger.info(client)

    """
    2. Set params
    """
    logger.info('Set params')
    # exp name
    exp_name = str(slurm_job_name)

    # set depth
    depth_ind = 14 #14 is 40m, 21 is 100m, 30 is 250m, 39 is 500m 

    
    # set size of tile in degrees lat/lon, sets FFT tile sizes, 
    # set size of sub-tile boxes in lat/lon, set i,j extents of the spatial box
    # ------------ 1 deg Kuroshio Extension centered @ 39°N, 158°E
    lat_center = 39
    lon_center = 158
    degree_extent = 1.2 # a little greater than 1 allows tile_width to trim to 4 sub-panels of exactly 0.5 x 0.5 deg^2 = 1 x 1 deg^2
    tile_width = 0.5

    # ------------- 2 deg Kuroshio Extension centered @ 39°N, 158°E
    # lat_center = 39
    # lon_center = 158
    # degree_extent = 2.2
    # tile_width = 1.0

    # ------------ 1 deg Agulhas Current centered @ 43°S, 14°E
    # lat_center = -43
    # lon_center = 14
    # degree_extent = 1.2
    # tile_width = 0.5

    # ------------- 2 deg Agulhas Current centered @ 43°S, 14°E
    # lat_center = -43
    # lon_center = 14
    # degree_extent = 2.2
    # tile_width = 1.0

    # rolling mean temporal extent for plotting
    rolling_mean_l = 24 * 5

    # temporal extent of the calculation/time series
    t_0 =  0 #4000
    t_1 =  int(429 * 24)# t_0 +  24 * 4 * 4#

    # LF average: define the LF/HF temporal cutoff
    LF_cutoff  = 24 * 3
    dt = 1.0 # hourly data

    # define seawater heat capacity and density
    C_p, rho = 3900, 1025

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

    for face, (j0, j1, i0, i1) in boxes.items():
        LLC_sub = LLC_full.isel(face=face, j=slice(j0, j1), i=slice(i0, i1))

    # select temporal extent, chunk
    LLC_sub = LLC_sub.isel(time=slice(t_0,t_1)).chunk({'time': -1, 'i': 96, 'j': 96})


    """
    4. vertical velocity interpolation, tiling
    """

    # define and interpolate vertical velocity, operate on all k to correctly weight adjacent faces (xgcm needs full vertical grid information)
    grid_3d = xgcm.Grid(
        LLC_sub,
        coords={
            'Z': {'center': 'k', 'left': 'k_p1'}},
        metrics={'Z': ['drF']},
        autoparse_metadata=False)

    # define vertical velocity
    LLC_sub['W_interp'] = grid_3d.interp(LLC_sub['W'], 'Z', boundary='extend')

    # select depth
    LLC_sub = LLC_sub.isel(k=depth_ind)

    # subset to only relevant vars
    LLC_sub = LLC_sub[['W_interp','Theta','XC','YC', 'rA']]

    # build tile index
    tile_index = build_sub_index(LLC_sub, tile_width,"tile").compute()
    LLC_sub = LLC_sub.assign_coords(tile=tile_index)
    # mask invalid pixels (the purple edges in the fig below)
    LLC_sub = LLC_sub.where(LLC_sub.tile >= 0)

    # define surface area, W, T
    area = LLC_sub['rA'] # surface areas
    W = LLC_sub['W_interp']
    T = LLC_sub['Theta']

    """
    5. calculate the total VHF with $C_p \rho <W'T'>$ for each time t, where <W'T'> = bar(WT) - bar(W) bar(T), 
    """

    # sum by tile, preserving area weighting
    WT_m = ((W * T * area).groupby("tile").sum()) / area.groupby("tile").sum()
    W_m  = ((W * area).groupby("tile").sum()) / area.groupby("tile").sum()
    T_m  = ((T * area).groupby("tile").sum()) / area.groupby("tile").sum()


    # calculate <W'T'>, avg by tile
    total_WT_ = (WT_m.mean(dim='tile') - (W_m.mean(dim='tile') * T_m.mean(dim='tile')))

    VHF_total = C_p * rho * total_WT_

    """
    6. define LF/HF masks
    """ 

    dt = 1.0  # hours
    Nt = total_WT_.sizes["time"]

    freq = np.fft.fftfreq(Nt, d=dt)
    df = np.abs(freq[1] - freq[0])

    fc = 1 / LF_cutoff  # cutoff frequency (cycles/hour)

    # number of frequency bins below cutoff
    n_cut = int(np.floor(fc / df))

    # build symmetric Tukey window around zero frequency
    win_len = 2 * n_cut + 1
    win = tukey(win_len, alpha=0.3)  # alpha controls smoothness

    LF_mask = np.zeros(Nt)

    # positive frequencies (including zero)
    LF_mask[: n_cut + 1] = win[n_cut:]

    # negative frequencies
    LF_mask[-n_cut:] = win[:n_cut]

    HF_mask = 1.0 - LF_mask

    # expand dims for broadcasting to (time, j, i)
    LF_mask_nd = LF_mask[:, None, None]
    HF_mask_nd = HF_mask[:, None, None]


    """
    7. LF, HF VHF calculation
    """

    # FFT in time, apply LF / HF
    time_axis = W.get_axis_num("time")

    # fft
    W_f = da.fft.fft(W.data, axis=time_axis)
    T_f = da.fft.fft(T.data, axis=time_axis)

    # apply LF / HF masks (broadcast over j, i)
    LF_W = W_f * LF_mask_nd
    HF_W = W_f * HF_mask_nd

    LF_T = T_f * LF_mask_nd
    HF_T = T_f * HF_mask_nd

    # inverse FFT
    W_LF = da.fft.ifft(LF_W, axis=time_axis).real
    W_HF = da.fft.ifft(HF_W, axis=time_axis).real

    T_LF = da.fft.ifft(LF_T, axis=time_axis).real
    T_HF = da.fft.ifft(HF_T, axis=time_axis).real

    # back to xarray
    coords = W.coords
    dims = W.dims

    W_LF = xr.DataArray(W_LF, coords=coords, dims=dims, name="W_LF")
    W_HF = xr.DataArray(W_HF, coords=coords, dims=dims, name="W_HF")
    T_LF = xr.DataArray(T_LF, coords=coords, dims=dims, name="T_LF")
    T_HF = xr.DataArray(T_HF, coords=coords, dims=dims, name="T_HF")

    # VHF calculation

    # ---------- LF ----------
    WT_LF = ((W_LF * T_LF * area).groupby("tile").sum()) / area.groupby("tile").sum()
    Wm_LF = ((W_LF * area).groupby("tile").sum()) / area.groupby("tile").sum()
    Tm_LF = ((T_LF * area).groupby("tile").sum()) / area.groupby("tile").sum()

    LF_VHF = C_p * rho * (WT_LF.mean(dim='tile') - Wm_LF.mean(dim='tile') * Tm_LF.mean(dim='tile'))

    # ---------- HF ----------
    WT_HF = ((W_HF * T_HF * area).groupby("tile").sum()) / area.groupby("tile").sum()
    Wm_HF = ((W_HF * area).groupby("tile").sum()) / area.groupby("tile").sum()
    Tm_HF = ((T_HF * area).groupby("tile").sum()) / area.groupby("tile").sum()

    HF_VHF = C_p * rho * (WT_HF.mean(dim='tile') - Wm_HF.mean(dim='tile') * Tm_HF.mean(dim='tile'))

    """
    8. calculate sum of LF and HF, apply a rolling mean for plotting
    """

    VHF_total_sum = LF_VHF + HF_VHF

    # rolling means for better visualization
    LF_VHF = LF_VHF.rolling(time=rolling_mean_l).mean()
    HF_VHF = HF_VHF.rolling(time=rolling_mean_l).mean()
    VHF_total_sum = VHF_total_sum.rolling(time=rolling_mean_l).mean()
    VHF_total = VHF_total.rolling(time=rolling_mean_l).mean()

    """
    9. Compute plotting reqs before figure production
    """

    logger.info('Compute plotting reqs')
    VHF_total, LF_VHF, HF_VHF, VHF_total_sum = dask.compute(
        VHF_total, LF_VHF, HF_VHF, VHF_total_sum)

    """
    10. Produce figure
    """

    logger.info('Produce figure')
    outdir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(VHF_total_sum.time, VHF_total_sum, color="black", label="LF + HF", linewidth=2)
    ax.plot(VHF_total.time, VHF_total, color="green", label="Total VHF", linewidth=2)
    ax.plot(HF_VHF.time, HF_VHF, color="blue", label="HF VHF", linewidth=1.5)
    ax.plot(LF_VHF.time, LF_VHF, color="red", label=f"LF VHF, (<{LF_cutoff } hours) ", linewidth=2)

    ax.set_title(f"{exp_name}", fontsize=18)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("W m$^{-2}$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    fig.savefig(outdir / f"{exp_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    """
    11. Optionally save VHF time series as netcdf for further analysis
    """
    #logger.info('Save VHF time series')
    #LF_VHF.to_netcdf(outdir / "timeseries" / "LF_VHF_ts.nc")
    #HF_VHF_ts.to_netcdf(outdir / "timeseries" / "HF_VHF_ts.nc")
    #total_VHF_ts.to_netcdf(outdir / "timeseries" / "total_VHF_ts.nc")

    if scalene_flag:
        # stop memory profiling
        scalene_profiler.stop()

if __name__ == "__main__":
    main()