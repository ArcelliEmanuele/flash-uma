import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data ──────────────────────────────────────────────────────────────────────

sweep_atoms = [100, 200, 500, 1000, 2000]
sweep_edges = [10, 20, 30, 40, 50, 60, 70]

# TF32 = True
pt_atoms_time   = [11.45, 12.96, 25.33, 47.58, 93.55]
gpu_atoms_time  = [12.97, 12.98, 23.72, 44.67, 87.97]
cust_atoms_time = [15.27, 16.88, 16.92, 23.00, 43.51]

pt_edges_time   = [13.03, 22.20, 30.35, 39.05, 47.54, 56.26, 65.15]
gpu_edges_time  = [12.91, 20.89, 28.57, 36.71, 44.68, 52.82, 61.06]
cust_edges_time = [16.83, 16.96, 17.70, 20.41, 23.04, 26.83, 30.78]

pt_atoms_mem   = [1184.64, 2234.94, 5371.10, 10610.59, 21102.33]
gpu_atoms_mem  = [1206.86, 2279.18, 5486.53, 10842.95, 21566.87]
cust_atoms_mem = [536.13,  954.66,  2201.08, 4295.91,  8477.45]

pt_edges_mem   = [2323.03, 4400.12, 6474.44, 8550.06, 10608.84, 12676.13, 14740.02]
gpu_edges_mem  = [2380.54, 4500.77, 6616.12, 8737.23, 10842.00, 12951.71, 15059.43]
cust_edges_mem = [1027.55, 1852.37, 2660.95, 3483.90,  4295.88,  5120.19,  5923.15]

# ── Style ─────────────────────────────────────────────────────────────────────

C_PT   = '#9DB4C0'   # muted blue-gray   – UMA (PyTorch)
C_GPU  = '#507DBC'   # medium blue        – UMA (GPU backend)
C_OURS = '#E05C2A'   # warm orange-red    – FlashUMA

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          15,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'axes.grid':          True,
    'grid.color':         '#e0e0e0',
    'grid.linewidth':     0.6,
    'grid.linestyle':     '-',
    'xtick.major.size':   0,
    'ytick.major.size':   0,
    'legend.frameon':     False,
    'legend.fontsize':    14,
})

KW = dict(linewidth=2.8, markersize=8, markeredgewidth=0)

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.subplots_adjust(hspace=0.60, wspace=0.40)

def gb(vals):
    return [v / 1024 for v in vals]

panels = [
    # (ax, x, ys, xlabel, ylabel, title)
    (axes[0, 0], sweep_atoms,
     [pt_atoms_time, gpu_atoms_time, cust_atoms_time],
     'Number of atoms', 'Time (ms)',
     'Runtime: system size\n(~50 edges / atom)'),

    (axes[0, 1], sweep_edges,
     [pt_edges_time, gpu_edges_time, cust_edges_time],
     'Edges per atom', 'Time (ms)',
     'Runtime: graph density\n(1 000 atoms)'),

    (axes[1, 0], sweep_atoms,
     [gb(pt_atoms_mem), gb(gpu_atoms_mem), gb(cust_atoms_mem)],
     'Number of atoms', 'Peak VRAM (GB)',
     'Memory: system size\n(~50 edges / atom)'),

    (axes[1, 1], sweep_edges,
     [gb(pt_edges_mem), gb(gpu_edges_mem), gb(cust_edges_mem)],
     'Edges per atom', 'Peak VRAM (GB)',
     'Memory: graph density\n(1 000 atoms)'),
]

markers = ['o', 's', 'D']
colors  = [C_PT, C_GPU, C_OURS]
labels  = ['UMA official (PyTorch)', 'UMA official (GPU)', 'FlashUMA (ours)']

for ax, x, ys, xlabel, ylabel, title in panels:
    for y, m, c, lbl in zip(ys, markers, colors, labels):
        ax.plot(x, y, marker=m, color=c, label=lbl, **KW)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14.5, fontweight='semibold', pad=8)
    ax.tick_params(labelsize=13)

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_xscale('log')
    ax.set_xticks(sweep_atoms)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
axes[0, 1].set_xticks(sweep_edges)
axes[1, 1].set_xticks(sweep_edges)

# Single legend above the figure
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls,
           loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.03),
           fontsize=14)


plt.savefig('benchmark.png', dpi=160, bbox_inches='tight', facecolor='white')
print("Saved benchmark.png")
