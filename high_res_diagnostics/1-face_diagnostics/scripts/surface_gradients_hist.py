"""High-gradient drift at one depth, plus |grad| histograms (all px and top 2.5%) of LLC and every emulator."""
import concurrent.futures as cf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diag_common import base_parser, emulator_configs_from, format_time, load_patches, n_cpus, out_dir, pick

p = base_parser(__doc__)
p.add_argument('--depth', type=int, default=0, help='k level (W uses k_p1)')
args = p.parse_args()

llc_patch, emulator_patches, emulator_info, all_patches = load_patches(
    emulator_configs_from(args), slice(*args.i_extent), slice(*args.j_extent),
    args.time_range, args.stepping, face=args.face)
n_emulators = len(emulator_info)

grad_vars = args.vars
depth = args.depth
PERCENTILE = 97.5
N_BINS_ALL, N_BINS_TOP = 100, 50  # shared |grad| bins across LLC + emulators, per time
OUT_DIR = out_dir(args, 'simple')

metric_vars = {
    'Theta': (None, None),
    'Salt': (None, None),
    'U': ('dxC', 'dyG'),   # U is on (j, i_g)
    'V': ('dxG', 'dyC'),   # V is on (j_g, i)
}
hist_colours = plt.get_cmap('tab10').colors
drift_colours = {
    'llc_only': '#FC9065',
    'emu_only': '#BE3977',
    'overlap': '#FCFCBE',
}


def _metric_values(patch, var):
    dx_name, dy_name = metric_vars.get(var, (None, None))
    if dx_name is None or dy_name is None:
        dx = np.sqrt(patch['rA'].values)  # tracers are on (j, i)
        dy = dx
    else:
        dx = patch[dx_name].values
        dy = patch[dy_name].values
    return dx, dy


def _coord_block(patch, name, n_j, n_i):
    coord = patch[name]
    j_dim = 'j' if 'j' in coord.dims else ('lat' if 'lat' in coord.dims else 'y')
    i_dim = 'i' if 'i' in coord.dims else ('lon' if 'lon' in coord.dims else 'x')
    sel = coord.isel({j_dim: slice(0, n_j), i_dim: slice(0, n_i)})
    for extra_dim in set(sel.dims) - {j_dim, i_dim}:
        sel = sel.isel({extra_dim: 0})
    return sel.transpose(j_dim, i_dim).values


def _nice_lon_lat_ticks(coord_2d, axis_vals, axis):
    idx = np.linspace(0, len(axis_vals) - 1, 5).round().astype(int)
    pos = axis_vals[idx]
    if axis == 'i':
        coord_vals = coord_2d[coord_2d.shape[0] // 2, idx]
    else:
        coord_vals = coord_2d[idx, coord_2d.shape[1] // 2]
    return pos, [f'{v:.1f}' for v in coord_vals]


xc_block = yc_block = None
for var in grad_vars:
    print(f"Computing grad_{var} at k={depth}...")
    fields, gradient_masks, grads = {}, {}, {}

    def _load(item):
        name, patch = item
        return name, pick(patch, var, k=depth).values  # (time, j, i)

    with cf.ThreadPoolExecutor(max_workers=min(len(all_patches), n_cpus())) as pool:
        loaded = dict(pool.map(_load, all_patches.items()))

    for patch_name, patch in all_patches.items():
        data = loaded[patch_name]
        dx, dy = _metric_values(patch, var)
        if dx.shape != data.shape[-2:] or dy.shape != data.shape[-2:]:
            raise ValueError(f"{patch_name} {var}: data {data.shape[-2:]} vs dx {dx.shape} / dy {dy.shape}")
        d_di = (np.roll(data, -1, axis=-1) - np.roll(data, 1, axis=-1)) / (2 * dx[np.newaxis, :, :])
        d_dj = (np.roll(data, -1, axis=-2) - np.roll(data, 1, axis=-2)) / (2 * dy[np.newaxis, :, :])
        grad_mag = np.sqrt(d_di ** 2 + d_dj ** 2)

        mask = np.zeros_like(grad_mag, dtype=bool)
        for t in range(grad_mag.shape[0]):
            mask[t] = grad_mag[t] >= np.nanpercentile(grad_mag[t], PERCENTILE)
        fields[patch_name] = data
        grads[patch_name] = grad_mag
        gradient_masks[patch_name] = mask
        print(f"  ✓ {patch_name}: {mask.sum()} high-gradient pixels ({mask.sum() / mask.size * 100:.1f}%)")

    print(f"Generating gradient drift figure for {var}...")
    ref_patch = emulator_patches[emulator_info[0][1]]
    n_times = len(ref_patch.time)
    nrows, ncols = n_times, n_emulators + 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), dpi=150, squeeze=False)

    for row in range(n_times):
        time_str = format_time(ref_patch.time.values[row])
        llc_mask_surface = gradient_masks['llc'][row]
        n_j, n_i = llc_mask_surface.shape
        j_vals, i_vals = np.arange(n_j), np.arange(n_i)
        if xc_block is None:
            xc_block = _coord_block(llc_patch, 'XC', n_j, n_i)
            yc_block = _coord_block(llc_patch, 'YC', n_j, n_i)
        x_tick_pos, x_tick_labels = _nice_lon_lat_ticks(xc_block, i_vals, 'i')
        y_tick_pos, y_tick_labels = _nice_lon_lat_ticks(yc_block, j_vals, 'j')

        top_mean = {k: grads[k][row][gradient_masks[k][row]].mean() for k in grads}
        for col, (emu_name, emu_key) in enumerate(emulator_info):
            ax = axes[row, col]
            emu_mask_surface = gradient_masks[emu_key][row]
            ax.imshow(fields[emu_key][row], cmap='Greys', aspect='auto', origin='lower')

            overlap_mask = llc_mask_surface & emu_mask_surface
            llc_only_mask = llc_mask_surface & ~emu_mask_surface
            emu_only_mask = emu_mask_surface & ~llc_mask_surface
            llc_j, llc_i = np.where(llc_only_mask)
            emu_j, emu_i = np.where(emu_only_mask)
            ovl_j, ovl_i = np.where(overlap_mask)
            ax.scatter(llc_i, llc_j, c=drift_colours['llc_only'], s=1, alpha=0.5, label='LLC top 2.5%', rasterized=True)
            ax.scatter(emu_i, emu_j, c=drift_colours['emu_only'], s=1, alpha=0.5, label='Emu top 2.5%', rasterized=True)
            ax.scatter(ovl_i, ovl_j, c=drift_colours['overlap'], s=1, alpha=0.7, label='Overlap', rasterized=True)

            pix = int((100 - PERCENTILE) / 100 * n_j * n_i)
            ax.set_title(f'{emu_name} {var} k={depth} {time_str} overlap={np.sum(overlap_mask)}/{pix}\n'
                         f'top-2.5% mean |grad| {top_mean[emu_key]:.2e} (LLC {top_mean["llc"]:.2e})', fontsize=8)
            ax.set_xticks(x_tick_pos)
            ax.set_xticklabels(x_tick_labels, fontsize=7)
            ax.set_yticks(y_tick_pos)
            ax.set_yticklabels(y_tick_labels, fontsize=7)
            ax.set_xlabel('Longitude', fontsize=7)
            ax.set_ylabel('Latitude', fontsize=7)

        # |grad| histograms on bins shared by LLC + emulators: all px, then each patch's own top 2.5%
        hist_patches = [('LLC', 'llc')] + list(emulator_info)
        all_px = {k: grads[k][row][np.isfinite(grads[k][row])] for _, k in hist_patches}
        top_px = {k: grads[k][row][gradient_masks[k][row]] for _, k in hist_patches}
        for ax, px, n_bins, label in [(axes[row, -2], all_px, N_BINS_ALL, 'all px'),
                                      (axes[row, -1], top_px, N_BINS_TOP, 'top 2.5%')]:
            lo = 0 if label == 'all px' else min(v.min() for v in px.values())
            edges = np.linspace(lo, max(v.max() for v in px.values()), n_bins + 1)
            for colour, (name, key) in zip(hist_colours, hist_patches):
                counts, _ = np.histogram(px[key], bins=edges)
                ax.stairs(counts, edges, color=colour, alpha=0.2, fill=True)
                ax.stairs(counts, edges, color=colour, linewidth=1, label=name)
            ax.set_title(f'{var} k={depth} {time_str} |grad| {label}', fontsize=8)
            ax.set_xlabel(f'|grad {var}|', fontsize=7)
            ax.set_ylabel('# px', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)
            ax.set_yscale('log')

    plt.tight_layout()
    out = f'{OUT_DIR}/gradients-hist-{var}-k{depth}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved {out}")
