"""Coarsen a 1/48-degree sharp rollout (tiled_eval predictions.zarr, one face) to the 1/3-degree native faces.

The same operator as the truth (Ocean_Emulator_Multiscale/scripts/coarsen_llc.py, native product):
- U and V are centred from the C grid (the face's last column/row copies itself; land is 0 first) and
  rotated to east/north with the 1/48-degree CS/SN;
- every field is the rA-weighted mean over the wet 1/48-degree cells of each 16x16 box
  (mask_c per level for U, V, Theta and Salt; hFacC_2d > 0 for Eta); all-land boxes are NaN.

Output, next to the source as <name>_third_coarse.zarr, in the layout of the coarse emulator's predictions.zarr:
U_k, V_k, Theta_k, Salt_k and Eta as (time, face, lat, lon), 13 faces of 270 x 270 with only `--face` filled.
Faces 0-5 are already north-up, so no rotation is needed for face 1.
"""
import argparse
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr

sys.path.insert(0, "/orcd/home/002/codycruz/Ocean_Emulator_Multiscale/src")
from ocean_emulators.llc_coarse import box_sum, center_inplace, ratio, rotate_to_east_north  # noqa: E402

RAW = "/orcd/data/abodner/003/LLC4320/LLC4320"
HFACC_2D = "/orcd/data/abodner/002/cody/LLC_boundary_rechunk/LLC4320_2d_720.zarr/hFacC_2d"
BOX, LEVELS, N_FACES, N = 16, 51, 13, 270
F32 = np.float32

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--predictions", required=True, help="tiled_eval predictions.zarr (time, lat=j, lon=i) of one face")
p.add_argument("--face", type=int, default=1)
p.add_argument("--threads", type=int, default=16, help="time steps coarsened at once")
p.add_argument("--time-stride", type=int, default=1, help="keep every Nth time step")
args = p.parse_args()
if args.face >= 7:
    raise SystemExit("faces 7-12 need the north-up rotation; only faces 0-6 are handled")

src = zarr.open_group(args.predictions, mode="r")
out_path = Path(args.predictions).with_name(Path(args.predictions).stem + "_third_coarse.zarr")
times = xr.open_zarr(args.predictions, chunks=None).time.values[:: args.time_stride]
steps = np.arange(src["Eta"].shape[0])[:: args.time_stride]
print(f"{args.predictions}: {len(steps)} times, face {args.face} -> {out_path}", flush=True)

# The truth's grid and masks on this face.
raw = zarr.open_group(RAW, mode="r")
f = args.face
cs, sn, ra = (raw[name][f].astype(F32) for name in ("CS", "SN", "rA"))
wet3d = raw["mask_c"][:, f] > 0
wet2d = zarr.open_array(HFACC_2D, mode="r")[f] > 0
wet_area3d = np.stack([box_sum(np.where(wet3d[k], ra, F32(0)), BOX) for k in range(LEVELS)])
wet_area2d = box_sum(np.where(wet2d, ra, F32(0)), BOX)
print("grid and masks loaded", flush=True)


def box_mean(x: np.ndarray, wet: np.ndarray, wet_area: np.ndarray) -> np.ndarray:
    return ratio(box_sum(np.where(wet, x, F32(0)) * ra, BOX), wet_area)


# Output: the coarse emulator's layout, NaN off `--face`.
compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
out = zarr.open_group(str(out_path), mode="w")
names = [f"{v}_{k}" for v in ("U", "V", "Theta", "Salt") for k in range(LEVELS)] + ["Eta"]
for name in names:
    a = out.create_dataset(name, shape=(len(steps), N_FACES, N, N), chunks=(1, N_FACES, N, N), dtype="f4",
                           fill_value=np.nan, compressor=compressor)
    a.attrs["_ARRAY_DIMENSIONS"] = ["time", "face", "lat", "lon"]
for name, values in (("face", np.arange(N_FACES)), ("lat", np.arange(N)), ("lon", np.arange(N))):
    out[name] = values
    out[name].attrs["_ARRAY_DIMENSIONS"] = [name]
# Times as proleptic-gregorian hours, like the coarse emulator; the source's date labels are kept as they are.
times = [pd.Timestamp(t) if isinstance(t, np.datetime64) else t for t in times]
dates = [cftime.DatetimeProlepticGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in times]
units = f"hours since {dates[0].strftime('%Y-%m-%dT%H:%M:%S')}"
out["time"] = cftime.date2num(dates, units, calendar="proleptic_gregorian")
out["time"].attrs.update(_ARRAY_DIMENSIONS=["time"], units=units, calendar="proleptic_gregorian")
out.attrs.update(source=str(args.predictions), face=f, source_attrs=dict(src.attrs),
                 operator="coarsen_llc.py native: centred, east/north U/V; rA-weighted wet 16x16 means; land NaN")


def coarsen(index: int) -> None:
    t = int(steps[index])
    for k in range(LEVELS):
        u = center_inplace(src[f"U_{k}"][t].astype(F32), axis=-1)
        v = center_inplace(src[f"V_{k}"][t].astype(F32), axis=-2)
        east, north = rotate_to_east_north(u, v, cs, sn)
        for name, x in (("U", east), ("V", north), ("Theta", src[f"Theta_{k}"][t]), ("Salt", src[f"Salt_{k}"][t])):
            out[f"{name}_{k}"][index, f] = box_mean(x, wet3d[k], wet_area3d[k])
    out["Eta"][index, f] = box_mean(src["Eta"][t], wet2d, wet_area2d)


start = _time.perf_counter()
with ThreadPoolExecutor(args.threads) as pool:
    for count, _ in enumerate(pool.map(coarsen, range(len(steps))), 1):
        if count % 10 == 0 or count == len(steps):
            elapsed = _time.perf_counter() - start
            print(f"{count}/{len(steps)} times, {elapsed / count:.1f} s each", flush=True)
zarr.consolidate_metadata(str(out_path))
print(f"Done: {out_path}", flush=True)
