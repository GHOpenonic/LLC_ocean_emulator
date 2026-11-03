"""Surface frequency-wavenumber spectra and E(k) of KE = 1/2 (U^2 + V^2) and b = -g sigma0 / rho0, for two time ranges."""
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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, NullLocator

from diag_common import base_parser, emulator_configs_from, format_time, load_patches, mem_budget_bytes, n_cpus, out_dir, pick

p = base_parser(__doc__)
p.add_argument('--time-range-2', type=int, nargs=2, required=True, metavar=('START', 'END'),
               help='second time range (inclusive indices), plotted as row 2')
args = p.parse_args()
bad = set(args.vars) - {'KE', 'B'}
if bad:
    raise SystemExit(f"--vars must be KE and/or B, got {sorted(bad)}")

TIME_RANGES = [tuple(args.time_range), tuple(args.time_range_2)]
LOADED = [load_patches(emulator_configs_from(args), slice(*args.i_extent), slice(*args.j_extent),
                       tr, args.stepping, face=args.face) for tr in TIME_RANGES]
llc_patch, _, emulator_info, _ = LOADED[0]

SPECTRA_VARS = args.vars
OUT_DIR = out_dir(args, 'KE_B/spectra-two-ranges')
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




def seconds_since_start(time_vals):
    if np.issubdtype(time_vals.dtype, np.datetime64):
        return (time_vals - time_vals[0]) / np.timedelta64(1, 's')
    t0 = pd.Timestamp(time_vals[0].isoformat())
    return np.array([(pd.Timestamp(t.isoformat()) - t0).total_seconds() for t in time_vals])


x1, y1, dx_m = build_metric_coords(llc_patch)
print(f"Approx dx = {dx_m:.1f} m")

# spectra[r][var][source], one entry per time range
spectra, time_s_all, row_labels = [], [], []
for r, (llc_r, _, _, patches_r) in enumerate(LOADED):
    time_s = seconds_since_start(llc_r.time.values)
    time_s_all.append(time_s)
    row_labels.append(f"{format_time(llc_r.time.values[0])} –\n{format_time(llc_r.time.values[-1])}")

    def _spectrum(job, patches_r=patches_r, time_s=time_s):
        var, name = job
        print(f"  spectra: range {r + 1} {var} {name}", flush=True)
        return job, compute_iso_spectrum(surface_field(patches_r[name], var), x1, y1, time_s)

    jobs = [(var, name) for var in SPECTRA_VARS for name in patches_r]
    _per_job = 16 * llc_r.sizes['time'] * len(x1) * len(y1) * 6  # complex FFT plus window/copies
    n_workers = int(max(1, min(len(jobs), n_cpus(), mem_budget_bytes() // _per_job)))
    print(f"range {r + 1}: {len(jobs)} spectra on {n_workers} worker(s)")
    cache = {var: {} for var in SPECTRA_VARS}
    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for (var, name), ps in pool.map(_spectrum, jobs):
            cache[var][name] = ps
    spectra.append(cache)
del LOADED
print("All spectra computed.")

# ============== PLOTTING ==============
SPEC_XLIM = [0.01, 0.25]
# Frequencies both ranges resolve, so no empty bands at either edge.
_om = [sp[v]['llc'].freq_time.values * 3600.0 for sp in spectra for v in SPECTRA_VARS]
SPEC_YLIM = [max(0.018, max(o[o > 0].min() for o in _om)), min(o.max() for o in _om)]
WAVELENGTH_TICKS_KM = [100, 30, 10, 5]
PERIOD_TICKS_H = [48, 24, 12, 6, 3, 2]
MARK_KM = 30  # labelled R_d
M2_HOURS = 12.42
FIELD_UNITS = {'KE': r'm$^2$ s$^{-2}$', 'B': r'm s$^{-2}$'}
RANGE_STYLES = ['-', '--', ':', '-.']  # Week 1 solid, Week 2 dashed
N_LEVELS = 12
cmap_dict = {'KE': 'magma', 'B': 'viridis'}
panel_keys = ['llc'] + [key for _, key in emulator_info]
panel_labels = ['LLC4320 (truth)'] + [name for name, _ in emulator_info]
_EMU_COLORS = ['tab:green', 'tab:orange', 'tab:purple', 'tab:red', 'tab:blue']
LINE_STYLE = {'llc': dict(color='k', lw=1.8, zorder=5)}
LINE_STYLE.update({k: dict(color=_EMU_COLORS[i % len(_EMU_COLORS)], lw=1.4) for i, (_, k) in enumerate(emulator_info)})


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


def wavenumber_spectrum(ps):
    """E(k) per cpkm: the iso spectrum integrated over all frequencies."""
    kr = ps.kr.values * 1e3
    d_omega = float(np.median(np.diff(ps.freq_time.values)))
    E = np.asarray(ps.values).sum(axis=0) * d_omega / 1e3
    pos = kr > 0
    return kr[pos], E[pos]


def frequency_spectrum(ps):
    """One-sided E(omega) per cph: the iso spectrum integrated over all wavenumbers."""
    om = ps.freq_time.values * 3600.0
    dk = float(np.median(np.diff(ps.kr.values)))
    E = np.asarray(ps.values).sum(axis=1) * dk / 3600.0
    pos = om > 0
    return om[pos], 2.0 * E[pos]  # field is real, so E(-omega) = E(omega)


def _inv(x):
    """1/x for the wavelength/period secondary axes; 0 maps to inf."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore'):
        return 1.0 / x


def _in_window(krp, omp, Z):
    return Z[np.ix_((omp >= SPEC_YLIM[0]) & (omp <= SPEC_YLIM[1]), (krp >= SPEC_XLIM[0]) & (krp <= SPEC_XLIM[1]))]


def shared_levels(var):
    """Log levels from the truth's visible range (1st percentile to max), pooled over both time ranges."""
    flat = np.concatenate([np.ravel(np.ma.filled(_in_window(*get_plot_arrays(sp[var]['llc'])), np.nan))
                           for sp in spectra])
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if flat.size == 0:
        return None
    return np.logspace(np.log10(np.percentile(flat, 1)), np.log10(flat.max()), N_LEVELS)


def _log_axis(axis):
    axis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis.set_minor_formatter(NullFormatter())


def _int_ticks(sec_axis, ticks, lo, hi):
    """Fixed integer ticks on a 1/x secondary axis, kept inside [lo, hi] of the primary axis."""
    sec_axis.set_major_locator(FixedLocator([t for t in ticks if lo <= 1.0 / t <= hi]))
    sec_axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    sec_axis.set_minor_locator(NullLocator())


n2d = len(panel_keys)
nrows = len(TIME_RANGES)
fig = plt.figure(figsize=(4.3 * (n2d + 1) + 0.8, 3.6 * nrows * len(SPECTRA_VARS) + 1.6), dpi=150)
outer = gridspec.GridSpec(len(SPECTRA_VARS), 1, hspace=0.55, figure=fig)


def _ylim_in(curves, lo, hi):
    """Log y-range covering every curve inside [lo, hi] on x."""
    vis = np.concatenate([y[(x >= lo) & (x <= hi)] for x, y in curves])
    vis = vis[np.isfinite(vis) & (vis > 0)]
    return vis.min() / 2, vis.max() * 2


def _line_panel(ax, curves, xlim, mark, mark_label, xlabel, sec_label, sec_ticks, ylabel):
    """1-D spectra: colour = source, line style = time range; dotted marker line; 1/x axis on top."""
    for (r, pkey), (x, y) in curves.items():
        ax.loglog(x, y, ls=RANGE_STYLES[r % len(RANGE_STYLES)], **LINE_STYLE[pkey])
    ax.axvline(mark, color='0.3', ls=':', lw=1.2)
    ax.text(mark, 0.03, f' {mark_label}', transform=ax.get_xaxis_transform(), fontsize=10, color='0.3')
    ax.set_xlim(xlim)
    ax.set_ylim(_ylim_in(curves.values(), *xlim))
    _log_axis(ax.xaxis)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.tick_params(labelsize=10, which='both', left=False)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    sec = ax.secondary_xaxis('top', functions=(_inv, _inv))
    _int_ticks(sec.xaxis, sec_ticks, *xlim)
    sec.tick_params(labelsize=10)
    sec.set_xlabel(sec_label, fontsize=12)
    ax.grid(alpha=0.3, which='major', linestyle='--', linewidth=0.5)


for v, var in enumerate(SPECTRA_VARS):
    levels = shared_levels(var)
    if levels is None:
        print(f"Skipping {var}: no finite truth data in the plot window.")
        continue
    norm = LogNorm(vmin=levels[0], vmax=levels[-1])
    cmap = plt.get_cmap(cmap_dict.get(var, 'magma')).copy()
    cmap.set_under('black')  # below the truth's 1st percentile reads as clearly empty
    cmap.set_bad('black')
    cmap.set_over(cmap(1.0))

    # [2-D spectra block | narrow gap for the period axis | E(k) over E(omega)]
    blocks = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[v], wspace=0, width_ratios=[n2d, 0.16, 1])
    gs2d = gridspec.GridSpecFromSubplotSpec(nrows, n2d, subplot_spec=blocks[0], hspace=0.07, wspace=0.07)
    gs1d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=blocks[2], hspace=0.5)
    axes = [[fig.add_subplot(gs2d[r, c]) for c in range(n2d)] for r in range(nrows)]

    for r in range(nrows):
        top, bottom = r == 0, r == nrows - 1
        for c, (pkey, plabel) in enumerate(zip(panel_keys, panel_labels)):
            ax = axes[r][c]
            krp, omp, Z_vp = get_plot_arrays(spectra[r][var][pkey])
            ax.set_facecolor('black')  # masked bins match the "empty" colour
            cs = ax.contourf(krp, omp, np.ma.filled(Z_vp, np.nan), levels=levels, norm=norm, cmap=cmap, extend='both')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(SPEC_XLIM)
            ax.set_ylim(SPEC_YLIM)
            _log_axis(ax.xaxis)
            _log_axis(ax.yaxis)
            ax.tick_params(labelsize=10, labelbottom=bottom, labelleft=(c == 0))
            if bottom:
                ax.set_xlabel('wavenumber (cpkm)', fontsize=12)
            if c == 0:
                ax.set_ylabel('frequency (cph)', fontsize=12)
                ax.annotate(f"Week {r + 1}\n{row_labels[r]}", xy=(-0.36, 0.5), xycoords='axes fraction',
                            rotation=90, ha='center', va='center', fontsize=11, fontweight='bold')
            if c == n2d - 1:
                sec = ax.secondary_yaxis('right', functions=(_inv, _inv))
                _int_ticks(sec.yaxis, PERIOD_TICKS_H, *SPEC_YLIM)
                sec.tick_params(labelsize=10)
                sec.set_ylabel('period (h)', fontsize=12)
            if top:
                sec = ax.secondary_xaxis('top', functions=(_inv, _inv))
                _int_ticks(sec.xaxis, WAVELENGTH_TICKS_KM, *SPEC_XLIM)
                sec.tick_params(labelsize=10)
                sec.set_xlabel('wavelength (km)', fontsize=12)
                ax.set_title(f"{plabel} — {var}", fontsize=12, fontweight='bold', pad=10)
            ax.grid(alpha=0.25, which='major', linestyle='--', linewidth=0.5, color='w')

    u = FIELD_UNITS.get(var, '')
    ax_k = fig.add_subplot(gs1d[0])
    _line_panel(ax_k, {(r, k): wavenumber_spectrum(spectra[r][var][k]) for r in range(nrows) for k in panel_keys},
                SPEC_XLIM, 1.0 / MARK_KM, r'$R_d$', 'wavenumber (cpkm)', 'wavelength (km)', WAVELENGTH_TICKS_KM,
                rf'E(k) [({u})$^2$ / cpkm]')
    ax_k.set_title(f"E(k), E(ω) — {var}", fontsize=12, fontweight='bold', pad=10)
    handles = [Line2D([], [], label=lab, **LINE_STYLE[k]) for k, lab in zip(panel_keys, panel_labels)] + \
              [Line2D([], [], color='0.4', ls=RANGE_STYLES[r % len(RANGE_STYLES)], label=f'Week {r + 1}')
               for r in range(nrows)]
    ax_k.legend(handles=handles, fontsize=9, frameon=False, loc='upper right')

    ax_w = fig.add_subplot(gs1d[1])
    _line_panel(ax_w, {(r, k): frequency_spectrum(spectra[r][var][k]) for r in range(nrows) for k in panel_keys},
                SPEC_YLIM, 1.0 / M2_HOURS, 'M2 tide', 'frequency (cph)', 'period (h)', PERIOD_TICKS_H,
                rf'E(ω) [({u})$^2$ / cph]')

    # Horizontal colorbar under the spectra columns, shared by both rows.
    p0, p1 = axes[-1][0].get_position(), axes[-1][n2d - 1].get_position()
    h_in = fig.get_figheight()  # place it a fixed 0.8 in below the axes, clear of the tick labels
    cax = fig.add_axes([p0.x0, p0.y0 - 0.8 / h_in, p1.x1 - p0.x0, 0.13 / h_in])
    cbar = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='both')
    exps = np.arange(np.ceil(np.log10(levels[0])), np.floor(np.log10(levels[-1])) + 1).astype(int)
    if len(exps) >= 2:
        cbar.set_ticks([10.0 ** e for e in exps], labels=[rf'$10^{{{e}}}$' for e in exps])
    cbar.minorticks_off()
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(rf'$|\omega|\,k_r\,P$ ({var})', fontsize=12)

tags = '_'.join(f"t{a}-{b}" for a, b in TIME_RANGES)
out = f"{OUT_DIR}/spectra_surface_{'+'.join(SPECTRA_VARS)}_{tags}.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
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
for r, (a, b) in enumerate(TIME_RANGES):
    for var in SPECTRA_VARS:
        _, _, Z_llc = get_plot_arrays(spectra[r][var]['llc'])
        for emu_name, emu_key in emulator_info:
            _, _, Z_emu = get_plot_arrays(spectra[r][var][emu_key])
            val = spectral_nrmse(np.ma.filled(Z_llc, np.nan), np.ma.filled(Z_emu, np.nan))
            print(f"  t[{a}:{b}] {var:>3s} | {emu_name} : {val:6.2f}% NRMSE")
