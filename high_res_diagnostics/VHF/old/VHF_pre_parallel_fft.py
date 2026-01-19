"""
This script calculates the vertical heat flux (VHF) through a layer of water 
in low-frequency (LF), high-frequency (HF), and total (LF + HF) components, 
then produces a time series figure of these VHF components. These calculations 
are performed in a specified spatial box in the LLC4320 dataset. 
The methods are as follows:

0. Import dependencies, define helper functions
1. Initialize dask
2. Set params
3. Open and subset LLC4320
4. vertical velocity interpolation w/ grid_3d.interp, This is necessary as Theta values are spatially centered within LLC pixels while W (vertical velocity) values are horizontally centered, but vertically shifted to the edges of LLC pixels. 
5. Box subsetting, coarse grain spatially by box
6. calculate the total VHF with $C_p \rho <W'T'>$ for each time t, where <W'T'> = bar(WT) - bar(W) bar(T)
7. take the temporal fourier transform of W and T fields
8. define LF and HF by respectively applying a low and high pass filter to W and T using a mask and inverse fourier transform back into time space
9. define LF and HF VHF respectively with $C_p \rho <W_{lowpass} T_{lowpass}>$ and $C_p \rho <W_{highpass} T_{highpass}>$
10. Compute plotting reqs before figure production
11. Produce figure

Note on fft(W,T) fields:
filtering W and T fields separately before calculating <WT> isolates the VHF carried by the low and high frequency motions themselves,
which explicitly separates the transport mechanism by scale.
"""

"""
0. Import dependencies, define helper functions
"""

import xarray as xr
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
import zarr
import xgcm 


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


def iterate_boxes(LLC, box_width):

    i_start, i_end, j_start, j_end, num_boxes = i_j_slices(LLC, box_width)

    box_id = 0
    for i0, i1 in zip(i_start, i_end):
        for j0, j1 in zip(j_start, j_end):
            # only yield boxes with valid size
            if (i1 - i0) > 0 and (j1 - j0) > 0:
                yield box_id, (i0, i1, j0, j1)
                box_id += 1

# function to build a box index
def build_box_index(LLC, box_width):
    box = xr.full_like(LLC["XC"], fill_value=-1).astype(int)

    for box_id, (i0, i1, j0, j1) in iterate_boxes(LLC, box_width):
        box[j0:j1, i0:i1] = box_id
    return box


def main():
    logger.info('Initializing Dask')

    """
    1. Initialize dask
    """

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=12,
        memory_limit="520GB",
        dashboard_address=None)
    client = Client(cluster)
    logger.info(client)

    """
    2. Set params
    """
    logger.info('Set params')

    # set size of tile in degrees lat/lon, sets FFT tile sizes (originally box)
    tile_width = 1

    # set depth
    depth_ind = 14 #40m

    # set i,j extents of the spatial box
    # these are 1x1 deg:
    i_0 = 2790
    i_1 = 2860
    j_0 = 745
    j_1 = 795
    chunk_factor = 1 # rechunk graphs by ndeg

    # calculate FFT of fluctuations or just fields?
    fluctuations = False # just fields

    # how much to take the rolling mean for plotting
    rolling_mean_l = 48 

    # temporal extent of the calculation/time series
    t_0 =  0#4000
    t_1 = 429 * 24#t_0 +  24 * 4 * 4#

    # LF average: define the LF/HF temporal cutoff
    LF_cutoff  = 24 * 3

    # define seawater heat capacity and density
    C_p, rho = 3900, 1025

    """
    3. open and subset LLC4320
    """

    # open LLC4320
    LLC_full = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

    # select [i,j] spatial box, face 7, temporal subset, chunk
    LLC_sub = LLC_full.isel(time = slice(t_0,t_1), i = slice(i_0, i_1), j = slice(j_0, j_1), face = 7).chunk({'time': -1, 'i': int((i_1-i_0)/chunk_factor), 'j': int((j_1-j_0)/chunk_factor)})


    """
    4. vertical velocity interpolation
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

    # build, assign box index
    box_index = build_box_index(LLC_sub, tile_width).compute()
    LLC_sub = LLC_sub.assign_coords(box=box_index)

    W = LLC_sub['W_interp']
    T = LLC_sub['Theta']

    """
    5. coarse grain spatially by box
    """

    # group into boxes, define WT, Wm, Tm, weighting by surface cell areas
    A = LLC_sub['rA']

    WT = (W * T * A).groupby("box").sum() / A.groupby("box").sum()
    Wm = (W * A).groupby("box").sum() / A.groupby("box").sum()
    Tm = (T * A).groupby("box").sum() / A.groupby("box").sum()

    valid = (WT.box != -1)
    WT = WT.where(valid)
    Wm = Wm.where(valid)
    Tm = Tm.where(valid)

    """
    6. calculate the total VHF with $C_p \rho <W'T'>$ for each time t, where <W'T'> = bar(WT) - bar(W) bar(T)
    """

    # calculate <W'T'>, avg by box
    total_WT_ = (WT.mean(dim="box") - Wm.mean(dim="box") * Tm.mean(dim="box"))

    VHF_total = C_p * rho * total_WT_

    """
    7. take the temporal fourier transform of W and T fields
    """

    time_axis = total_WT_.get_axis_num("time")

    W = W.persist() 
    T = T.persist()

    if fluctuations == True:
        f = 2
    else:
        f = 1

    for iter in range(f):
        if iter == 0:
            exp_name = "FFT_FIELDS"
            # FFT of fields
            W_f = np.fft.fft(W.data, axis=time_axis)
            T_f = np.fft.fft(T.data, axis=time_axis)
        if iter == 1:
            exp_name = "FFT_FLUCTUATIONS"
            # decompose into fluctuations of the fields
            W_prime = W - W.mean(dim="time")
            T_prime = T - T.mean(dim="time")

            # FFT of fluctuations of the fields
            W_f = np.fft.fft(W_prime.data, axis=time_axis)
            T_f = np.fft.fft(T_prime.data, axis=time_axis)



        """
        8. define LF and HF by respectively applying a low and high pass filter to W and T using a mask and inverse fourier transform back into time space
        """
        dt = 1.0  # data is hourly
        Nt = total_WT_.sizes["time"]

        freq = np.fft.fftfreq(Nt, d=dt)  # in cycles per hour

        # define cutoff
        fc = dt / LF_cutoff # cycles per hour

        # frequency mask: shape (Nt,)
        LF_mask = np.abs(freq) <= fc
        HF_mask = np.abs(freq) > fc

        # expand dims for broadcasting: (Nt, 1, 1)
        LF_mask_nd = LF_mask[:, None, None]
        HF_mask_nd = HF_mask[:, None, None]

        # apply masks (broadcasts over i,j)
        LF_W = W_f * LF_mask_nd
        HF_W = W_f * HF_mask_nd

        LF_T = T_f * LF_mask_nd
        HF_T = T_f * HF_mask_nd

        # inverse fourier transform back into time space, take the real
        W_LF = np.fft.ifft(LF_W, axis=time_axis).real
        W_HF = np.fft.ifft(HF_W, axis=time_axis).real 

        T_LF = np.fft.ifft(LF_T, axis=time_axis).real
        T_HF = np.fft.ifft(HF_T, axis=time_axis).real 

        # convert to xr.dataarray
        coords = {"time": T.time, "i": T.i, "j": T.j}
        dims = ("time", "j", "i")

        W_LF = xr.DataArray(W_LF, coords=coords, dims=dims, name="W_LF")
        W_HF = xr.DataArray(W_HF, coords=coords, dims=dims, name="W_HF")

        T_LF = xr.DataArray(T_LF, coords=coords, dims=dims, name="T_LF")
        T_HF = xr.DataArray(T_HF, coords=coords, dims=dims, name="T_HF")

        """
        9. define LF and HF VHF respectively with $C_p \rho W_{lowpass} T_{lowpass}$ and $C_p \rho W_{highpass} T_{highpass}$
        """
        # group into boxes, define WT, Wm, Tm, weighting by surface cell areas
        A = LLC_sub['rA']

        # LF
        WT_LF = (W_LF * T_LF * A).groupby("box").sum() / A.groupby("box").sum()
        Wm_LF = (W_LF * A).groupby("box").sum() / A.groupby("box").sum()
        Tm_LF = (T_LF * A).groupby("box").sum() / A.groupby("box").sum()

        valid = (WT_LF.box != -1)
        WT_LF = WT_LF.where(valid)
        Wm_LF = Wm_LF.where(valid)
        Tm_LF = Tm_LF.where(valid)

        # HF
        WT_HF = (W_HF * T_HF * A).groupby("box").sum() / A.groupby("box").sum()
        Wm_HF = (W_HF * A).groupby("box").sum() / A.groupby("box").sum()
        Tm_HF = (T_HF * A).groupby("box").sum() / A.groupby("box").sum()

        valid = (WT_HF.box != -1)
        WT_HF = WT_HF.where(valid)
        Wm_HF = Wm_HF.where(valid)
        Tm_HF = Tm_HF.where(valid)

        # define VHF
        LF_VHF =  C_p * rho * (WT_LF.mean(dim="box") - Wm_LF.mean(dim="box") * Tm_LF.mean(dim="box"))
        HF_VHF =  C_p * rho * (WT_HF.mean(dim="box") - Wm_HF.mean(dim="box") * Tm_HF.mean(dim="box"))

        VHF_total_sum = LF_VHF + HF_VHF

        # rolling means for better visualization
        LF_VHF = LF_VHF.rolling(time=rolling_mean_l).mean()
        HF_VHF = HF_VHF.rolling(time=rolling_mean_l).mean()
        VHF_total_sum = VHF_total_sum.rolling(time=rolling_mean_l).mean()
        VHF_total = VHF_total.rolling(time=rolling_mean_l).mean()
    
        """
        10. Compute plotting reqs before figure production
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

        ax.set_title(f"VHF, {exp_name}", fontsize=18)
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("W m$^{-2}$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        fig.savefig(outdir / f"VHF_timeseries_{LF_cutoff }, {exp_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    """
    Save VHF time series as netcdf for further exploration
    """
    #logger.info('Save VHF time series')
    #LF_VHF.to_netcdf(outdir / "timeseries" / "LF_VHF_ts.nc")
    #HF_VHF_ts.to_netcdf(outdir / "timeseries" / "HF_VHF_ts.nc")
    #total_VHF_ts.to_netcdf(outdir / "timeseries" / "total_VHF_ts.nc")

if __name__ == "__main__":
    main()