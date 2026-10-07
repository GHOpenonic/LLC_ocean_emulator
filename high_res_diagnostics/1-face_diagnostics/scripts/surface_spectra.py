"""Surface frequency-wavenumber spectra of KE = 1/2 (U^2 + V^2) and b = -g sigma0 / rho0."""
import concurrent.futures as cf

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xrft
from fastjmd95 import jmd95numba
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter

from diag_common import base_parser, emulator_configs_from, load_patches, mem_budget_bytes, n_cpus, out_dir, pick

p = base_parser(__doc__)
args = p.parse_args()
bad = set(args.vars) - {'KE', 'B'}
if bad:
    raise SystemExit(f"--vars must be KE and/or B, got {sorted(bad)}")

llc_patch, emulator_patches, emulator_info, all_patches = load_patches(
    emulator_configs_from(args), slice(*args.i_extent), slice(*args.j_extent),
    args.time_range, args.stepping)

SPECTRA_VARS = args.vars
OUT_DIR = out_dir(args, 'KE_B/spectra')
G = 9.81
rho0 = 1025
NEEDS = {'KE': ('U', 'V'), 'B': ('Salt', 'Theta')}


def surface(patch, name):
    return pick(patch, name, k=0).values


def surface_field(patch, var):
    """(time, j, i) DataArray of KE or B at k=0; land (NaN) filled with 0 so the FFT stays finite."""
    if var == 'KE':
        U, V = surface(patch, 'U'), surface(patch, 'V')
        data = 0.5 * (U ** 2 + V ** 2)
    else:
        Salt, Theta = surface(patch, 'Salt'), surface(patch, 'Theta')
        sigma0 = jmd95numba.rho(Salt, Theta, 0) - rho0
        data = -G * sigma0 / rho0
    data = np.nan_to_num(data, nan=0.0)
    return xr.DataArray(data, dims=('time', 'j', 'i'),
                        coords={'time': patch.time, 'j': patch.j, 'i': patch.i})


def build_metric_coords(patch):
    """Approximate uniform x1/y1 coords in metres from the mean rA."""
    dx = float(np.sqrt(np.nanmean(patch['rA'].values)))
    nj = patch.sizes['j']
    ni = patch.sizes['i']
    x1 = (np.arange(ni) - ni / 2 + 0.5) * dx
    y1 = (np.arange(nj) - nj / 2 + 0.5) * dx
    return x1, y1, dx


def azimuthal_avg(k, l, f, N, nfactor=4.0):
    """Azimuthally average 2-D spectrum slice f(l,k) -> 1-D in kr."""
    kk, ll = np.meshgrid(k, l)
    K = np.sqrt(kk ** 2 + ll ** 2)
    nbins = int(N / nfactor)
    if k.max() > l.max():
        ki = np.linspace(0., l.max(), nbins)
    else:
        ki = np.linspace(0., k.max(), nbins)
    kidx = np.digitize(K.ravel(), ki)
    area = np.bincount(kidx)
    kr = np.bincount(kidx, weights=K.ravel()) / np.maximum(area, 1)
    iso_f = np.ma.masked_invalid(
        np.bincount(kidx, weights=f.ravel()) / np.maximum(area, 1)
    ) * kr
    return kr, iso_f


def compute_iso_spectrum(da, x1, y1, time_s):
    """3-D power spectrum -> azimuthal average -> (omega, kr) in SI."""
    da = da.assign_coords(
        x1=('i', x1), y1=('j', y1), time=('time', time_s)
    ).swap_dims({'i': 'x1', 'j': 'y1'})
    drop_list = [c for c in ['i', 'j', 'XC', 'YC', 'rA'] if c in da.coords]
    if drop_list:
        da = da.drop_vars(drop_list)
    da['x1'].attrs['units'] = 'm'
    da['y1'].attrs['units'] = 'm'
    da['time'].attrs['units'] = 's'
    da = da.chunk({'time': -1, 'y1': -1, 'x1': -1})

    ps3d = xrft.power_spectrum(
        da, dim=['x1', 'y1', 'time'],
        window='hann', window_correction=True,
    ).compute(scheduler='synchronous')

    kx_vals = ps3d.freq_x1.values
    ky_vals = ps3d.freq_y1.values
    omega_vals = ps3d.freq_time.values
    nomega = len(omega_vals)
    nfactor = 4.0

    _kr, _ = azimuthal_avg(kx_vals, ky_vals, ps3d.isel(freq_time=0).values, len(kx_vals), nfactor)
    ps_iso = np.ma.zeros((nomega, len(_kr)))
    for j in range(nomega):
        _, ps_iso[j, :] = azimuthal_avg(kx_vals, ky_vals, ps3d.isel(freq_time=j).values,
                                        len(kx_vals), nfactor)
    _kr[0] = 0.0
    ps_iso_xr = xr.DataArray(np.array(ps_iso), coords={'freq_time': omega_vals, 'kr': _kr},
                             dims=['freq_time', 'kr'])
    ps_iso_xr['freq_time'].attrs['units'] = 'cycles/s'
    ps_iso_xr['kr'].attrs['units'] = 'cycles/m'
    return ps_iso_xr


x1, y1, dx_m = build_metric_coords(llc_patch)
print(f"Approx dx = {dx_m:.1f} m")
time_vals = llc_patch.time.values
time_s = (time_vals - time_vals[0]) / np.timedelta64(1, 's') \
    if np.issubdtype(time_vals.dtype, np.datetime64) else \
    np.array([(pd.Timestamp(t.isoformat()) - pd.Timestamp(time_vals[0].isoformat())).total_seconds()
              for t in time_vals])


def _spectrum(job):
    var, patch_name = job
    print(f"  spectra: {var} {patch_name}", flush=True)
    return job, compute_iso_spectrum(surface_field(all_patches[patch_name], var), x1, y1, time_s)


jobs = [(var, name) for var in SPECTRA_VARS for name in all_patches]
_per_job = 16 * llc_patch.sizes['time'] * len(x1) * len(y1) * 6  # complex FFT plus window/copies
n_workers = int(max(1, min(len(jobs), n_cpus(), mem_budget_bytes() // _per_job)))
print(f"{len(jobs)} spectra on {n_workers} worker(s)")
spectra_cache = {var: {} for var in SPECTRA_VARS}
with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
    for (var, name), ps in pool.map(_spectrum, jobs):
        spectra_cache[var][name] = ps
print("All spectra computed.")

# ============== PLOTTING ==============
dt_hours = float(np.median(np.diff(time_s))) / 3600.0  # follows stepping
nyquist_cph = 1.0 / (2.0 * dt_hours)
SPEC_XLIM = [0.01, 0.25]
SPEC_YLIM = [0.018, nyquist_cph]
cmap_dict = {'KE': 'magma', 'B': 'viridis'}
panel_keys = ['llc'] + [key for _, key in emulator_info]
panel_labels = ['LLC'] + [name for name, _ in emulator_info]


def get_plot_arrays(ps):
    """kr_km(+), omega_cph(+), Z_vp = |w_cph|*P/1e3 in the positive quadrant."""
    kr = ps.kr.values * 1e3
    om = ps.freq_time.values * 3600.0
    iso_km = ps.values / 1e3
    pos_o = om > 0
    pos_k = kr > 0
    Z = iso_km[np.ix_(pos_o, pos_k)]
    krp = kr[pos_k]
    omp = om[pos_o]
    Z_vp = np.abs(omp)[:, None] * Z
    return krp, omp, Z_vp


nrows, ncols = 1, len(panel_keys)
for var in SPECTRA_VARS:
    cmap = cmap_dict.get(var, 'magma')
    vals = [get_plot_arrays(spectra_cache[var][pkey])[2] for pkey in panel_keys]
    flat = np.concatenate([np.ravel(z) for z in vals])
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if flat.size == 0:
        print(f"Skipping {var}: no finite data found.")
        continue
    vmin = np.percentile(flat, 1)
    vmax = np.percentile(flat, 99)
    levels = np.power(10., np.linspace(np.log10(vmin), np.log10(vmax), 16))[:15]
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(5.5 * ncols, 3.5 * nrows), dpi=150)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.1, hspace=0.3)
    axes = [fig.add_subplot(gs[0, c]) for c in range(ncols)]
    for c, (pkey, plabel) in enumerate(zip(panel_keys, panel_labels)):
        ax = axes[c]
        krp, omp, Z_vp = get_plot_arrays(spectra_cache[var][pkey])
        ax.contourf(krp, omp, Z_vp, levels=levels, norm=norm, cmap=cmap, extend='both')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(SPEC_XLIM)
        ax.set_ylim(SPEC_YLIM)
        ax.set_xlabel('wavenumber (cpkm)', fontsize=14)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if c == 0:
            ax.set_ylabel('frequency (cph)', fontsize=14)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', labelleft=False)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        ax.set_title(plabel, fontsize=12, fontweight='bold')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='neither')
    lo = int(np.ceil(np.log10(vmin)))
    hi = int(np.floor(np.log10(vmax)))
    exps = np.arange(lo, hi + 1)
    if len(exps) >= 2:
        cbar.set_ticks([10.0 ** e for e in exps])
        cbar.set_ticklabels([rf'$10^{{{e}}}$' for e in exps])
    else:
        ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.1e}' for t in ticks])
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(rf'$|\omega|\,k_r\,P$ ({var})', fontsize=14, labelpad=5)

    out = f'{OUT_DIR}/spectra_surface_{var}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ saved {out}')


# ============== SPECTRAL MAGNITUDE MATCH: normalized RMSE ==============
def spectral_nrmse(Z_llc, Z_emu):
    """NRMSE (%) between two variance-preserving spectra on shared finite bins."""
    m = np.isfinite(Z_llc) & np.isfinite(Z_emu)
    if not np.any(m):
        return np.nan
    a, b = Z_llc[m], Z_emu[m]
    denom = np.sqrt(np.sum(a ** 2))
    if denom == 0:
        return np.nan
    return np.sqrt(np.sum((a - b) ** 2)) / denom * 100.0


print("\nSpectral magnitude match at the surface (normalized RMSE, %):")
for var in SPECTRA_VARS:
    _, _, Z_llc = get_plot_arrays(spectra_cache[var]['llc'])
    for emu_name, emu_key in emulator_info:
        _, _, Z_emu = get_plot_arrays(spectra_cache[var][emu_key])
        val = spectral_nrmse(np.ma.filled(Z_llc, np.nan), np.ma.filled(Z_emu, np.nan))
        print(f"  {var:>3s} | {emu_name} : {val:6.2f}% NRMSE")
