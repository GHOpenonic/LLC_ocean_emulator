"""
This script calculates the vertical heat flux (VHF) in low-frequency (LF),
high-frequency (HF), and total (LF + HF) components, then produces a time
series figure of these VHF components. These calculations are performed
in a specified spatial box in the LLC4320 dataset. The methods are as follows:

1. define W (vertical velocity) and T (Theta, potential temperature) within a 1 deg x 1 deg box centered at 39N, 158E in the LLC4320 dataset. Split into 4 n deg x n deg sub-boxes.
2. interpolate W using grid_3d.interp along the z axis. This is necessary as Theta values are spatially centered within LLC pixels while W (vertical velocity) values are horizontally centered, but vertically shifted to the edges of LLC pixels.
3. calculate W and T spatial means in each n deg x n deg box for each hourly time t
4. calculate W' and T' by subtracting the box spatial mean from each within-box pixel's W and T value along time (so each time has a W' and T') 
5. calculate the LF+HF VHF with $C_p \rho W' T'$ for each time t 
6. calculate the LF VHF with $C_p \rho <W' T'>$ for each time t where <> denotes taking the time average of each W' and T' set to 1 day averages
$<W' T'> = \bar{WT} - \bar{W} \bar{T}$ 
7. calculate the HF VHF by subtracting LF from LF + HF: $HF = (LF + HF) - LF = C_p \rho W' T' - C_p \rho <W' T'>$ at each time step t (hourly).
8. calculate the LF, HF, and total VHF mean across all i,j for each time to produce a time series, weighting by cell surface areas
"""



"""
Notes, 12/8/25
define  W'T'_{total} same as LF method without daily averaging. New LF and HF: 
define W', T' with coarse graining.
define HF, LF w/ fourier transform as high pass, low pass filter, daily as cutoff. 
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


# define functions!

# function to get lat/lon vals from i, j indices
def lat_lon_from_i_j(LLC):
    lats = []
    lons = []
    lats.append(LLC['YC'][0,:].values)
    lons.append(LLC['XC'][:,0].values)

    return lats, lons

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


# function to iterate through boxes in space and time, applying some f
#def box_means(LLC, var, box_width):
#    i_start, i_end, j_start, j_end, num_boxes = i_j_slices(LLC, box_width)
#    vals = []
#    for i0, i1 in zip(i_start, i_end):
#        for j0, j1 in zip(j_start, j_end):
#            vals.append(
#                LLC[var]
#                .isel(i=slice(i0, i1), j=slice(j0, j1))
#                .mean(dim=['i', 'j']))

#    LLC[f'{var}_means'] = xr.concat(vals, dim='box')
#    return LLC

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

def calc_deviances(LLC, var, box_index):
    valid = box_index.where(box_index != -1)
    valid = valid.rename("box_id")

    # rename means to match groupby dimension names
    means = LLC[var].groupby(valid).mean().rename("means")

    # subtract box means from each cell
    anomalies = LLC[var].groupby(valid) - means

    LLC[f"{var}_prime"] = anomalies
    return LLC

# define function to calculate LF components
#def calc_LF(LLC, LF_average, box_width, box_index):
#    # compute WT, W< T box means hourly
#    i_start, i_end, j_start, j_end, num_boxes = i_j_slices(LLC, box_width)
#    WT_list = []
#    W_list = []
#    T_list = []

#    for i0, i1 in zip(i_start, i_end):
#        for j0, j1 in zip(j_start, j_end):

#            W_box = LLC["W40"].isel(i=slice(i0, i1), j = slice(j0, j1))
#            T_box = LLC["T40"].isel(i=slice(i0, i1), j = slice(j0, j1))

#            WT_box_mean = (W_box * T_box).mean(dim = ["i", "j"])
#            W_box_mean  =  W_box.mean(dim = ["i", "j"])
#            T_box_mean  =  T_box.mean(dim = ["i", "j"])

#            WT_list.append(WT_box_mean)
#            W_list.append(W_box_mean)
#            T_list.append(T_box_mean)

    # stack into shape (box, time)
#    WT = xr.concat(WT_list, dim = "box")
#    Wm = xr.concat(W_list,  dim = "box")
#    Tm = xr.concat(T_list,  dim = "box")

    # coarsen in time for LF average
#   WT_LF = WT.coarsen(time = LF_average, boundary="trim").mean()
#    Wm_LF = Wm.coarsen(time = LF_average, boundary="trim").mean()
#    Tm_LF = Tm.coarsen(time = LF_average, boundary="trim").mean()

    # calc covariance
#    LF_cov = WT_LF - (Wm_LF * Tm_LF)

#    LF_cov.name = "Wprime_Tprime_LF"

#    return LF_cov

def main():
    logger.info('Initializing Dask')

    cluster = LocalCluster( # n_workers * threads_per_workers should be <= cpus requested in slurm job
    n_workers=3,
    threads_per_worker=5,
    memory_limit='250GB', # memory_limit * n_workers should be ~85% or sth of --mem in slurm job ------- 
    dashboard_address=None#, 
    # local_directory="/scratch/codycruz/dask_tmp" 
    ) 
    client = Client(cluster) 
    logger.info(client)

    """
    Set params
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
    t_0 = 0
    t_1 = 429 * 24

    # LF average: define the number of hours to be averaged
    LF_average = 24

    # define seawater heat capacity and density
    C_p, rho = 3900, 1025

    """
    Prescribe calculations, should be lazy until persist later
    """
    logger.info('Prescribe calculations')

    # open LLC4320
    LLC_full = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

    # select large i,j spatial box, face 7
    LLC_sub = LLC_full.isel(i = slice(i_0, i_1), j = slice(j_0, j_1), face = 7)

    # select temporal subset, chunk in LF_average time and the whole i,j box, select subset of k around k=14=40m for interpolation
    LLC_sub = LLC_sub.isel(time = slice(t_0,t_1), k = slice(depth_ind-1, depth_ind+1)).chunk({'time': LF_average * 10, 'i': i_1 - i_0, 'j': j_1 - j_0})

    # define and interpolate vertical velocity
    grid_3d = xgcm.Grid(
        LLC_sub,
        coords={
            'Z': {'center': 'k', 'left': 'k_p1'}},
        autoparse_metadata=False)

    # define vertical velocity
    LLC_sub['W40'] = grid_3d.interp(LLC_sub['W'], 'Z', boundary='extend')

    # define potential temp
    LLC_sub['T40'] = LLC_sub['Theta']

    # select depth
    LLC_sub = LLC_sub.isel(k=1) # we already subset to a slice +/- depth_ind, so we have k = 0,1,2 to select from, k=1 corresponds to depth_ind

    # subset to only relevant vars
    LLC_sub = LLC_sub[['W40','T40','XC','YC','rA']]

    # build box index
    box_index = build_box_index(LLC_sub, box_width).compute()

    # calculate means
    #LLC_sub = LLC_sub['W40'].groupby(box_index).mean()
    #LLC_sub = LLC_sub['T40'].groupby(box_index).mean()

    # calculate deviances from the means
    LLC_sub = calc_deviances(LLC_sub, 'W40', box_index)
    LLC_sub = calc_deviances(LLC_sub, 'T40', box_index)

    # calculate the LF + HF VHF for each time t
    total_VHF = C_p * rho * LLC_sub['W40_prime'] * LLC_sub['T40_prime']

    # calculate the LF VHF for each time LF_average
    #LF_VHF = C_p * rho * calc_LF(LLC_sub, LF_average, box_width, box_index)
    WT = (LLC_sub['W40'] * LLC_sub['T40']).groupby(box_index).mean()
    Wm = LLC_sub['W40'].groupby(box_index).mean()
    Tm = LLC_sub['T40'].groupby(box_index).mean()
    # drop values with box index = bad
    valid = (WT.XC != -1)
    # take the mean across all boxes
    WT = WT.where(valid, drop=True).mean(dim='XC')
    Wm = Wm.where(valid, drop=True).mean(dim='XC')
    Tm = Tm.where(valid, drop=True).mean(dim='XC')
    logger.info('Compute intermediates WT, Wm, Tm')
    WT = WT.compute()
    Wm = Wm.compute()
    Tm = Tm.compute()
    # calculate covariance
    LF_cov = WT.coarsen(time=LF_average, boundary='trim').mean() - \
         (Wm.coarsen(time=LF_average, boundary="trim").mean() *
          Tm.coarsen(time=LF_average, boundary="trim").mean())
    logger.info('Compute intermediate LF_cov')
    LF_cov = LF_cov.compute()
    # calculate LF VHF
    LF_VHF = C_p * rho * LF_cov

    # calculate the HF VHF by subtacting LF from total (HF = HF + LF - LF)
    # expand LF_VHF to match the shape of total_VHF to calculate HF_VHF
    nt = total_VHF.sizes["time"]
    nLF = LF_VHF.sizes["time"]
    # expect nt = LF_average * nLF
    assert nt == LF_average * nLF, "Time dimensions must satisfy total = LF_average * LF"
    # build the mapping index
    idx = np.repeat(np.arange(nLF), LF_average)
    # now index LF_VHF using integer indexing
    LF_expanded = LF_VHF.isel(time = xr.DataArray(idx, dims = "time"))
    LF_expanded = LF_expanded.assign_coords(time = total_VHF.time)
    # finally calculate HF VHF:
    HF_VHF = total_VHF - LF_expanded

    # take means across i,j, weighting by cell surface areas to produce a time series
    weights = LLC_sub['rA']
    HF_VHF_ts = (HF_VHF * weights).sum(dim = ("i","j")) / weights.sum(dim = ("i","j"))
    total_VHF_ts = (total_VHF * weights).sum(dim = ("i","j")) / weights.sum(dim = ("i","j"))

    # expand LF to hourly resolution for nice plotting
    idx = np.repeat(np.arange(LF_VHF.sizes["time"]), LF_average)
    LF_expanded_ts = LF_VHF.isel(time=xr.DataArray(idx, dims="time"))
    LF_expanded_ts = LF_expanded_ts.assign_coords(time=HF_VHF_ts.time)

    """
    Compute plotting reqs before figure production
    """
    logger.info('Compute plotting reqs')
    total_VHF_ts = total_VHF_ts.compute()
    HF_VHF_ts = HF_VHF_ts.compute()
    LF_expanded_ts = LF_expanded_ts.compute()

    """
    Produce figure
    """
    logger.info('Produce figure')
    outdir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(total_VHF_ts.time, total_VHF_ts, color="black", label="Total VHF", linewidth=2)
    ax.plot(HF_VHF_ts.time, HF_VHF_ts, color="blue", label="HF VHF", linewidth=1.5)
    ax.plot(LF_expanded_ts.time, LF_expanded_ts, color="red", label="LF VHF", linewidth=2)

    ax.set_title("VHF Decomposition — Time Series", fontsize=18)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("W m$^{-2}$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    fig.savefig(outdir / "VHF_timeseries.png", dpi=200, bbox_inches="tight")
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