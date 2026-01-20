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
6. loop over each tile -> VHF per tile 
    a. split each tile into sub-boxes, needed for calculating spatial averages 
    b. take the temporal fourier transform of W and T fields, define LF and HF by applying butterworth filters to W and T fields, inverse fourier transform back into time space
        paper on butterworth filters in oceanography: Roberts, J., & Roberts, T. D. (1978). Use of the Butterworth low‐pass filter for oceanographic data. Journal of Geophysical Research: Oceans, 83(C11), 5510-5514.
    c. calculate VHF with the same equation as in 5
7. concat tiles together, apply a rolling mean for plotting
8. Compute plotting reqs before figure production
9. Produce figure
10. Optionally save VHF time series as netcdf for further analysis 

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
from dask.distributed import Client, LocalCluster, get_client
import zarr
import xgcm 
from scipy.signal import butter, freqz

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


def main():
    logger.info('Initializing Dask')

    """
    1. Initialize dask, scalene if flagged
    """

    # get SLURM environment variables, flags
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "job")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "0")
    scalene_flag = os.environ.get("SCALENE_PROFILE", "True").lower() in ("True")

    if scalene_flag:
        # begin memory profiling
        scalene_profiler.start()

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=8,
        memory_limit="80GB", # 480, 530 for 2 deg
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
    depth_ind = 39 #14 is 40m, 21 is 100m, 30 is 250m, 39 is 500m 

    
    # set size of tile in degrees lat/lon, sets FFT tile sizes, 
    # set size of sub-tile boxes in lat/lon, set i,j extents of the spatial box
    # ------------ 1 deg Kuroshio Extension centered @ 39°N, 158°E
    i_0 = 2790
    i_1 = 2860
    j_0 = 745
    j_1 = 795
    tile_width = .51 #4 tiles at 1 deg
    box_width = 0.23 # 4 box
    face = 7

    # ------------- 2 deg Kuroshio Extension centered @ 39°N, 158°E
    # i_0 = 2755
    # i_1 = 2895
    # j_0 = 720
    # j_1 = 820
    # tile_width = 1 # 4 tiles at 2 deg
    # box_width = 0.45# 4 box
    # face = 7

    # ------------ 1 deg Agulhas Current centered @ 43°S, 14°E
    # i_0 = 2790
    # i_1 = 2860
    # j_0 = 745
    # j_1 = 795
    # tile_width = .51 #4 tiles at 1 deg
    # box_width = 0.23 # 4 box
    # face = ?

    # ------------- 2 deg Agulhas Current centered @ 43°S, 14°E
    # i_0 = 2755
    # i_1 = 2895
    # j_0 = 720
    # j_1 = 820
    # tile_width = 1 # 4 tiles at 2 deg
    # box_width = 0.45# 4 box
    # face = ?

    # rolling mean temporal extent for plotting
    rolling_mean_l = 48 

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

    # select [i,j] spatial box, face 7, temporal subset, chunk
    LLC_sub = LLC_full.isel(time = slice(t_0,t_1), \
        i = slice(i_0, i_1), j = slice(j_0, j_1), \
            face = face).chunk({'time': -1, 'i': -1, 'j': -1})


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

    W = LLC_sub['W_interp']
    T = LLC_sub['Theta']
    A = LLC_sub['rA']

    # trim
    valid = W.tile >= 0
    W = W.where(valid)
    T = T.where(valid)
    A = A.where(valid)

    """
    5. calculate the total VHF with $C_p \rho <W'T'>$ for each time t, where <W'T'> = bar(WT) - bar(W) bar(T), 
    """


    # define WT, Wm, Tm, weighting by surface cell areas
    WT_m = ((W * T * A).groupby("tile").sum() / A.groupby("tile").sum())
    W_m = ((W * A).groupby("tile").sum() / A.groupby("tile").sum())
    T_m = ((T * A).groupby("tile").sum() / A.groupby("tile").sum())


    # calculate <W'T'>, avg by box
    total_WT_ = (WT_m.mean(dim="tile") - (W_m.mean(dim="tile") * T_m.mean(dim="tile")))

    VHF_total = C_p * rho * total_WT_

    """
    6. define LF/HF masks, heat capacity, density, and loop over each tile, 
    """ 

    time_axis = total_WT_.get_axis_num("time")

    # define filter cutoff
    fs = 1 / dt # sampling freq
    fc = 1 / LF_cutoff # cutoff freq
    Nt = LLC_sub.sizes['time']
    
    # define 4th-order lowpass Butterworth filter
    N_order = 4
    b_LF, a_LF = butter(N_order, Wn=fc, btype='low', fs=fs)
    b_HF, a_HF = butter(N_order, Wn=fc, btype='high', fs=fs)

    # freq for FFT mask
    freq = np.fft.fftfreq(Nt, d=dt)
    w, h_LF = freqz(b_LF, a_LF, worN=Nt, fs=fs) # apply butterworth
    _, h_HF = freqz(b_HF, a_HF, worN=Nt, fs=fs)

    # define LF and HF masks using butter
    LF_mask_nd = np.abs(h_LF)[:, None, None]  # expand dims for broadcasting
    HF_mask_nd = np.abs(h_HF)[:, None, None]

    # define heat capacity and density
    C_p = 3900
    rho = 1025

    # loop over each tile, calculate VHF
    LF_VHF_list = []
    HF_VHF_list = []
    for tile in np.unique(LLC_sub.tile.values):
        if tile >= 0:
            # select current tile
            LLC_tile = LLC_sub.where(LLC_sub.tile == tile, drop=True).persist() # materialize tile, give unique key

            """
            6 a. split each tile into sub-boxes, needed for calculating spatial averages 
            """

            # break tile into sub-boxes
            box_index = build_sub_index(LLC_tile, box_width,"box").compute()
            LLC_tile = LLC_tile.assign_coords(box=box_index)

            # mask invalid pixels
            LLC_tile = LLC_tile.where(LLC_tile.box >= 0)
            
            # define W, T, A
            W = LLC_tile['W_interp']
            T = LLC_tile['Theta']
            A = LLC_tile['rA']

            # trim
            valid = W.box >= 0
            W = W.where(valid)
            T = T.where(valid)
            A = A.where(valid)
            """
            6 b. take the temporal fourier transform of W and T fields, define LF and HF by applying low/high pass filters to W and T fields, inverse fourier transform back into time space
            """
            # take the temporal fft of the W and T fields
            time_axis = W.get_axis_num("time")
            W_f = da.fft.fft(W.data, axis=time_axis)
            T_f = da.fft.fft(T.data, axis=time_axis)

            # apply masks (broadcasts over i,j)
            LF_W = W_f * LF_mask_nd
            HF_W = W_f * HF_mask_nd

            LF_T = T_f * LF_mask_nd
            HF_T = T_f * HF_mask_nd

            # inverse fourier transform back into time space, take the real
            W_LF = da.fft.ifft(LF_W, axis=time_axis).real
            W_HF = da.fft.ifft(HF_W, axis=time_axis).real 

            T_LF = da.fft.ifft(LF_T, axis=time_axis).real
            T_HF = da.fft.ifft(HF_T, axis=time_axis).real 

            # return to xarray
            coords = {"time": W.time, "i": W.i, "j": W.j}
            dims = ("time", "j", "i")
            
            W_LF = xr.DataArray(W_LF, coords=coords, dims=dims, name="W_LF")
            W_HF = xr.DataArray(W_HF, coords=coords, dims=dims, name="W_HF")

            T_LF = xr.DataArray(T_LF, coords=coords, dims=dims, name="T_LF")
            T_HF = xr.DataArray(T_HF, coords=coords, dims=dims, name="T_HF")

            """
            6 c. calculate VHF with the same equation as in 5
            """

            # define WT, Wm, Tm, weighting by surface cell areas ================ box === tile
            spatial_divider = "box"
            # LF
            WT_LF = (W_LF * T_LF * A).groupby(f'{spatial_divider}').sum() / A.groupby(f'{spatial_divider}').sum()
            Wm_LF = (W_LF * A).groupby(f'{spatial_divider}').sum() / A.groupby(f'{spatial_divider}').sum()
            Tm_LF = (T_LF * A).groupby(f'{spatial_divider}').sum() / A.groupby(f'{spatial_divider}').sum()

            # HF
            WT_HF = (W_HF * T_HF * A).groupby(f'{spatial_divider}').sum() / A.groupby(f'{spatial_divider}').sum()
            Wm_HF = (W_HF * A).groupby(f'{spatial_divider}').sum() / A.groupby(f'{spatial_divider}').sum()
            Tm_HF = (T_HF * A).groupby(f'{spatial_divider}').sum() / A.groupby(f'{spatial_divider}').sum()

            # define VHF
            LF_VHF =  C_p * rho * (WT_LF.mean(dim=f'{spatial_divider}') - Wm_LF.mean(dim=f'{spatial_divider}') * Tm_LF.mean(dim=f'{spatial_divider}'))
            HF_VHF =  C_p * rho * (WT_HF.mean(dim=f'{spatial_divider}') - Wm_HF.mean(dim=f'{spatial_divider}') * Tm_HF.mean(dim=f'{spatial_divider}'))

            # compute VHF per tile, append to lists
            LF_VHF = LF_VHF.compute()
            HF_VHF = HF_VHF.compute()

            LF_VHF_list.append(LF_VHF)
            HF_VHF_list.append(HF_VHF)

            # clear per-tile graphs
            client = get_client()
            client.cancel([
                W_LF, W_HF, T_LF, T_HF, WT_LF, WT_HF, Wm_LF, Wm_HF, Tm_LF, Tm_HF])



            # logging
        #    logger.info(f'tile {tile} complete')

    """
    7. concat tiles together, apply a rolling mean for plotting
    """

    # concat lists together
    LF_VHF = xr.concat(LF_VHF_list, dim="tile").mean("tile")
    HF_VHF = xr.concat(HF_VHF_list, dim="tile").mean("tile")
    VHF_total_sum = LF_VHF + HF_VHF

    # rolling means for better visualization
    LF_VHF = LF_VHF.rolling(time=rolling_mean_l).mean()
    HF_VHF = HF_VHF.rolling(time=rolling_mean_l).mean()
    VHF_total_sum = VHF_total_sum.rolling(time=rolling_mean_l).mean()
    VHF_total = VHF_total.rolling(time=rolling_mean_l).mean()

    """
    8. Compute plotting reqs before figure production
    """

    logger.info('Compute plotting reqs')
    VHF_total.compute()
    LF_VHF = LF_VHF.compute()
    HF_VHF = HF_VHF.compute()
    VHF_total_sum = VHF_total_sum.compute()

    """
    9. Produce figure
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
    10. Optionally save VHF time series as netcdf for further analysis
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