"""Shared LLC + emulator loading for the 1-face diagnostics (mirrors the notebooks)."""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, '/home/codycruz/LLC_ocean_emulator/high_res_diagnostics')
from predictions_io import open_predictions

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

LLC_STORE = '/orcd/data/abodner/003/LLC4320/LLC4320'
GRID_VARS = ['XC', 'YC', 'rA', 'Z', 'dxC', 'dyC', 'dxG', 'dyG']
FIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs')


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--emulator', action='append', required=True, metavar='NAME=PATH',
                   help='repeat for several emulators')
    p.add_argument('--face', type=int, default=1, help='LLC face of the truth (must match the emulator run)')
    p.add_argument('--i-extent', type=int, nargs=2, required=True, metavar=('START', 'STOP'))
    p.add_argument('--j-extent', type=int, nargs=2, required=True, metavar=('START', 'STOP'))
    p.add_argument('--time-range', type=int, nargs=2, required=True, metavar=('START', 'END'),
                   help='inclusive indices')
    p.add_argument('--stepping', type=int, default=1)
    p.add_argument('--vars', nargs='+', required=True)
    p.add_argument('--out-dir', default=None)
    return p


def emulator_configs_from(args):
    configs = []
    for n, spec in enumerate(args.emulator, start=1):
        name, path = spec.split('=', 1)
        configs.append({'name': name, 'key': f'emulator_{n}', 'path': path, 'desc': ''})
    return configs


def out_dir(args, subdir):
    if args.out_dir:
        d = args.out_dir
    else:
        names = '+'.join(spec.split('=', 1)[0] for spec in args.emulator)
        tag = (f"{names}_f{args.face}_i{args.i_extent[0]}-{args.i_extent[1]}_j{args.j_extent[0]}-{args.j_extent[1]}"
               f"_t{args.time_range[0]}-{args.time_range[1]}s{args.stepping}")
        d = os.path.join(FIG_ROOT, subdir, tag)
    os.makedirs(d, exist_ok=True)
    return d


def normalize_times(times):
    return pd.DatetimeIndex([
        pd.Timestamp(
            int(t.year), int(t.month), int(t.day),
            int(t.hour), int(t.minute), int(t.second)
        )
        if hasattr(t, "year")
        else pd.Timestamp(t).floor("s")
        for t in times
    ])


def format_time(t_val):
    """DD/MM/YYYY:HH for cftime or datetime64."""
    try:
        return f"{t_val.day:02d}/{t_val.month:02d}/{t_val.year}:{t_val.hour:02d}h"
    except AttributeError:
        t_pd = pd.Timestamp(t_val)
        return f"{t_pd.day:02d}/{t_pd.month:02d}/{t_pd.year}:{t_pd.hour:02d}h"


def load_patches(emulator_configs, i_extent, j_extent, selected_time_range, stepping,
                 emulator_chunks=None, face=1):
    """Returns (llc_patch, emulator_patches, emulator_info, all_patches); emulator_chunks={} makes them dask-backed."""
    # ============== LOAD LLC ==============
    i_g_extent = i_extent
    j_g_extent = j_extent
    llc_patch_full = xr.open_zarr(LLC_STORE, consolidated=False).isel(
        face=face,
        i=i_extent,
        i_g=i_g_extent,
        j=j_extent,
        j_g=j_g_extent
    )

    # ============== OPEN EMULATOR DATASETS ==============
    emulator_patches_raw = {}
    for cfg in emulator_configs:
        emu_ds = open_predictions(cfg['path'], chunks=emulator_chunks)
        dim_mapping = {}
        if 'lat' in emu_ds.dims and 'j' not in emu_ds.dims:
            dim_mapping['lat'] = 'j'
        if 'lon' in emu_ds.dims and 'i' not in emu_ds.dims:
            dim_mapping['lon'] = 'i'
        if dim_mapping:
            emu_ds = emu_ds.rename(dim_mapping)
        emu_ds = emu_ds.isel(i=i_extent, j=j_extent)
        emu_ds = emu_ds.assign_coords({
            'i': llc_patch_full.i,
            'j': llc_patch_full.j
        })
        emulator_patches_raw[cfg['key']] = emu_ds
        print(f"Loaded {cfg['name']}: {cfg['path']}")

    # ============== TIME MATCHING ==============
    llc_times_norm = normalize_times(llc_patch_full.time.values)
    common_times = llc_times_norm
    for cfg in emulator_configs:
        emulator_times_norm = normalize_times(emulator_patches_raw[cfg['key']].time.values)
        common_times = common_times.intersection(emulator_times_norm)
    common_times = common_times.sort_values()

    llc_mask = llc_times_norm.isin(common_times)
    llc_patch = llc_patch_full.isel(time=llc_mask)
    print(f"LLC subset to {len(common_times)} common times")

    # ============== PORT GRID VARS ==============
    emulator_patches = {}
    for cfg in emulator_configs:
        patch_raw = emulator_patches_raw[cfg['key']]
        patch_times_norm = normalize_times(patch_raw.time.values)
        patch = patch_raw.isel(time=patch_times_norm.isin(common_times))
        for gv in GRID_VARS:
            if gv in llc_patch:
                patch[gv] = llc_patch[gv]
        emulator_patches[cfg['key']] = patch

    emulator_info = [(cfg['name'], cfg['key']) for cfg in emulator_configs]

    # ============== TIME SUBSET ==============
    start_idx, end_idx = selected_time_range
    llc_patch = llc_patch.isel(time=slice(start_idx, end_idx + 1, stepping))

    emulator_patches_subset = {}
    for key, patch in emulator_patches.items():
        safe_end_idx = min(end_idx, patch.sizes['time'] - 1)  # handles shorter emulator runs
        emulator_patches_subset[key] = patch.isel(time=slice(start_idx, safe_end_idx + 1, stepping))
    emulator_patches = emulator_patches_subset

    min_time_len = min(
        [llc_patch.sizes['time']] +
        [patch.sizes['time'] for patch in emulator_patches.values()]
    )
    llc_patch = llc_patch.isel(time=slice(0, min_time_len))
    emulator_patches = {
        key: patch.isel(time=slice(0, min_time_len))
        for key, patch in emulator_patches.items()
    }

    all_patches = {'llc': llc_patch}
    all_patches.update(emulator_patches)

    print(f"Subset to time indices {start_idx}:{end_idx}, stepping {stepping}: "
          f"{min_time_len} synchronized times")
    print(f"  {format_time(llc_patch.time.values[0])} -> {format_time(llc_patch.time.values[-1])}")
    return llc_patch, emulator_patches, emulator_info, all_patches


VERTICAL_DIMS = ("k", "k_p1", "k_l", "k_u", "lev")


def vertical_dim_of(da):
    for dim in VERTICAL_DIMS:
        if dim in da.dims:
            return dim
    return None


def pick(patch, var, *, time=None, k=None):
    """`patch[var]` with `time`/`k` applied to the axes it actually has (LLC W is on k_p1)."""
    da = patch[var]
    if time is not None:
        da = da.isel(time=time)
    if k is not None:
        dim = vertical_dim_of(da)
        if dim is not None:
            da = da.isel({dim: k})
    return da


def n_cpus():
    return int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))


def mem_budget_bytes(fraction=0.5):
    mb = os.environ.get('SLURM_MEM_PER_NODE')
    total = int(mb) * 1024 ** 2 if mb else os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    return int(total * fraction)
