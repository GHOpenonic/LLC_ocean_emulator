"""Prognostic fields at one depth: LLC + emulators, one row per time."""
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
    args.time_range, args.stepping)
n_emulators = len(emulator_info)

prog_vars = args.vars
depth = args.depth
colormaps = {'Theta': 'Spectral_r', 'Salt': 'viridis', 'U': 'bwr', 'V': 'bwr', 'W': 'magma', 'Eta': 'viridis'}
OUT_DIR = out_dir(args, 'simple')

for var in prog_vars:
    print(f"Generating plots for {var} at k={depth}...")

    ref_patch = emulator_patches[emulator_info[0][1]]
    n_times = len(ref_patch.time)
    time_indices = list(range(n_times))
    nrows = len(time_indices)
    ncols_fields = 1 + n_emulators
    cmap = colormaps.get(var, 'viridis')

    # Read each source's (time, j, i) slab once, concurrently.
    sources = [llc_patch] + [emulator_patches[k] for _, k in emulator_info]
    with cf.ThreadPoolExecutor(max_workers=min(len(sources), n_cpus())) as pool:
        slabs = list(pool.map(lambda s: pick(s, var, k=depth), sources))
        values = list(pool.map(lambda s: s.values, slabs))

    fig, axes = plt.subplots(nrows, ncols_fields, figsize=(3.6 * ncols_fields, 3 * nrows), dpi=200)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols_fields == 1:
        axes = axes.reshape(-1, 1)

    labels = ['LLC'] + [name for name, _ in emulator_info]
    for row, t in enumerate(time_indices):
        time_str = format_time(ref_patch.time.values[t])
        fields = [v[t] for v in values]

        finite_fields = [f[np.isfinite(f)] for f in fields]
        if not any(vals.size for vals in finite_fields):
            print(f"  Skipping {var} {time_str}: no finite values")
            continue
        vmin = np.min([vals.min() for vals in finite_fields if vals.size])
        vmax = np.max([vals.max() for vals in finite_fields if vals.size])

        cf_ = None
        for col, (field, slab, label) in enumerate(zip(fields, slabs, labels)):
            ax = axes[row, col]
            cf_ = ax.contourf(slab.coords.get('i', np.arange(field.shape[-1])),
                              slab.coords.get('j', np.arange(field.shape[-2])),
                              np.ma.masked_invalid(field),
                              cmap=cmap, vmin=vmin, vmax=vmax, levels=30)
            ax.set_title(f'{label} {var} k={depth} {time_str}', fontsize=8)
        fig.colorbar(cf_, ax=axes[row, :], pad=0.02, fraction=0.046)

    out = f'{OUT_DIR}/fields-{var}-k{depth}.png'
    plt.savefig(out)
    plt.close(fig)
    print(f"✓ Saved {out}")
