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
7. take the temporal fourier transform of <W'T'>
8. define LF and HF by respectively applying a low and high pass filter to <W'T'> using a mask and inverse fourier transform back into time space
9. define LF and HF VHF respectively with $C_p \rho W_{lowpass} T_{lowpass}$ and $C_p \rho W_{highpass} T_{highpass}$
10. Compute plotting reqs before figure production
11. Produce figure
Note on fft(<WT>):
Filtering <WT> directly isolates the amount of total VHF variance at low vs high freqs, without separating by the driving motions' themselves.
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

    cluster = LocalCluster( # n_workers * threads_per_workers should be <= cpus requested in slurm job
    n_workers=3,
    threads_per_worker=5,
    memory_limit='100GB',#'250GB', # memory_limit * n_workers should be ~85% or sth of --mem in slurm job ------- 
    dashboard_address=None#, 
    # local_directory="/scratch/codycruz/dask_tmp" 
    ) 
    client = Client(cluster) 
    logger.info(client)

    """
    2. Set params
    """
    logger.info('Set params')

    # set size of box in degrees lat/lon
    box_width = 0.5

    # set depth
    depth_ind = 14 #40m

    # set i,j extents of the spatial box
    i_0 = 2790
    i_1 = 2860
    j_0 = 745
    j_1 = 795

    # temporal extent of the calculation/time series
    t_0 = 4000 #0
    t_1 = t_0 +  24 * 4 * 4#429 * 24

    # LF average: define the number of hours to be averaged
    LF_cutoff  = 128

    # define seawater heat capacity and density
    C_p, rho = 3900, 1025

    logger.info('Prescribing calculations')

    """
    3. open and subset LLC4320
    """

    # open LLC4320
    LLC_full = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

    # select [i,j] spatial box, face 7, temporal subset, chunk
    LLC_sub = LLC_full.isel(time = slice(t_0,t_1), i = slice(i_0, i_1), j = slice(j_0, j_1), face = 7).chunk({'time': LF_cutoff  * 4, 'i': i_1 - i_0, 'j': j_1 - j_0})


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
    LLC_sub = LLC_sub.isel(k=14)

    # subset to only relevant vars
    LLC_sub = LLC_sub[['W_interp','Theta','XC','YC', 'rA']]

    # build, assign box index
    box_index = build_box_index(LLC_sub, box_width).compute()
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
    7. take the temporal fourier transform of <W'T'>
    """

    time_axis = total_WT_.get_axis_num("time")
    total_hat = np.fft.fft(total_WT_, axis=time_axis)

    """
    8. define LF and HF by respectively applying a low and high pass filter to <W'T'> using a mask and inverse fourier transform back into time space
    """
    dt = 1.0  # data is hourly
    Nt = total_WT_.sizes["time"]

    freq = np.fft.fftfreq(Nt, d=dt)  # in cycles per hour

    # define cutoff
    fc = dt / LF_cutoff 

    # define masks
    LF_mask = np.abs(freq) <= fc
    HF_mask = np.abs(freq) > fc
    # reshape masks into [time]
    shape = [1] * total_WT_.ndim
    shape[time_axis] = Nt

    LF_mask_nd = LF_mask.reshape(shape)
    HF_mask_nd = HF_mask.reshape(shape)

    # apply masks
    LF_hat = total_hat * LF_mask_nd
    HF_hat = total_hat * HF_mask_nd

    # inverse fourier transform back into time space, take the real
    WT_LF = np.fft.ifft(LF_hat, axis=time_axis).real
    WT_HF = np.fft.ifft(HF_hat, axis=time_axis).real 

    """
    9. define LF and HF VHF respectively with $C_p \rho W_{lowpass} T_{lowpass}$ and $C_p \rho W_{highpass} T_{highpass}$
    """
    # define VHF
    LF_VHF=  C_p * rho * WT_LF
    HF_VHF=  C_p * rho * WT_HF

    # convert to xr.dataarray
    LF_VHF = xr.DataArray(LF_VHF, coords={"time": VHF_total.time}, dims=["time"], name="LF_VHF")
    HF_VHF = xr.DataArray(HF_VHF, coords={"time": VHF_total.time}, dims=["time"], name="HF_VHF")
   
    """
    10. Compute plotting reqs before figure production
    """

    logger.info('Compute plotting reqs')
    VHF_total = VHF_total.compute()
    LF_VHF = LF_VHF.compute()
    HF_VHF = HF_VHF.compute()

    """
    9. Produce figure
    """

    logger.info('Produce figure')
    outdir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(VHF_total.time, VHF_total, color="black", label="Total VHF", linewidth=2)
    ax.plot(HF_VHF.time, HF_VHF, color="blue", label="HF VHF", linewidth=1.5)
    ax.plot(LF_VHF.time, LF_VHF, color="red", label=f"LF VHF, (<{LF_cutoff } hours) ", linewidth=2)

    ax.set_title("VHF Decomposition — Time Series", fontsize=18)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("W m$^{-2}$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    fig.savefig(outdir / f"VHF_timeseries_{LF_cutoff }.png", dpi=200, bbox_inches="tight")
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