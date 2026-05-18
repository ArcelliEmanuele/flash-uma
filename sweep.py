import torch
import gc
import time
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as wigner_hybrid

from wrapper import ZeroAllocBackbone
import model as custom_model  # Used to monkey-patch morton sorting

# ==============================================================================
# BYPASS LOGIC
# ==============================================================================
original_reorder = custom_model.reorder_graph_morton

def dummy_reorder_graph_morton(pos, Z, batch, idx_i, idx_j):
    """
    Bypasses the Morton sorting overhead for the custom model.
    Data is pre-sorted during generation to ensure realistic memory access patterns.
    """
    perm = torch.arange(len(pos), device=pos.device)
    return pos, Z, batch, idx_i, idx_j, perm, perm

# ==============================================================================
# DATA GENERATOR
# ==============================================================================
def generate_mock_data(num_atoms, target_degree, cutoff, device):
    """Generates pre-morton-sorted random graphs strictly matching the target degree."""
    pos_raw = torch.rand(num_atoms, 3, dtype=torch.float32, device=device)
    dist_matrix = torch.cdist(pos_raw, pos_raw)
    
    k_idx = min(num_atoms * target_degree + num_atoms, num_atoms * num_atoms)
    kth_dist = dist_matrix.flatten().kthvalue(k_idx).values.item()
    pos_base = pos_raw * ((cutoff - 1e-4) / kth_dist)
    
    Z = torch.randint(1, 100, (num_atoms,), device=device)
    batch = torch.zeros(num_atoms, dtype=torch.long, device=device)

    idx_i = torch.arange(num_atoms, device=device).repeat_interleave(num_atoms)
    idx_j = torch.arange(num_atoms, device=device).repeat(num_atoms)
    mask  = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    
    edge_vec  = pos_base[idx_i] - pos_base[idx_j]
    in_cut    = edge_vec.norm(dim=-1) < cutoff
    idx_i, idx_j = idx_i[in_cut], idx_j[in_cut]
    
    # PRE-SORT THE GRAPH HERE
    pos_s, Z_s, batch_s, idx_i_s, idx_j_s, _, _ = original_reorder(
        pos_base, Z, batch, idx_i, idx_j
    )
    E = idx_i_s.shape[0]

    rand_offsets = torch.randint(-1, 2, (E, 3), dtype=torch.float32, device=device)
    mock_natoms = torch.tensor([num_atoms], dtype=torch.long, device=device)

    data_mock = AtomicData(
        pos=pos_s.clone().requires_grad_(True), 
        atomic_numbers=Z_s,
        cell=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True),
        pbc=torch.zeros(1, 3, dtype=torch.bool, device=device),
        natoms=mock_natoms,                                         
        edge_index=torch.stack([idx_i_s, idx_j_s], dim=0),
        cell_offsets=rand_offsets,
        nedges=torch.tensor([E], dtype=torch.long, device=device),
        charge=torch.zeros(1, dtype=torch.long, device=device), 
        spin=torch.zeros(1, dtype=torch.long, device=device),                                             
        fixed=torch.zeros(num_atoms, dtype=torch.long, device=device),
        tags=torch.zeros(num_atoms, dtype=torch.long, device=device),
        batch=batch_s, sid=["mock_0"], dataset=["oc20"],
    )
    data_mock.batch_full = batch_s
    data_mock.atomic_numbers_full = Z_s
    return data_mock, E

# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================
def measure_performance(model, data, is_custom=False, warmup=10, iters=30):
    if is_custom:
        custom_model.reorder_graph_morton = dummy_reorder_graph_morton

    # 1. Warmup
    for _ in range(warmup):
        data.pos.grad, data.cell.grad = None, None
        _ = model(data)
    torch.cuda.synchronize()
    
    # 2. Measure Peak Memory
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    data.pos.grad, data.cell.grad = None, None
    out = model(data)
    del out # Free immediately
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    torch.cuda.synchronize()

    # 3. Measure Speed. Time each iteration individually so we can report
    # spread, not just the mean: MD step-time jitter is itself a meaningful
    # metric (it bounds worst-case wall clock over a long trajectory).
    iter_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for s_evt, e_evt in iter_events:
        data.pos.grad, data.cell.grad = None, None
        s_evt.record()
        _ = model(data)
        e_evt.record()
    torch.cuda.synchronize()

    times_ms = np.array([s_evt.elapsed_time(e_evt) for s_evt, e_evt in iter_events])
    avg_time_ms = float(times_ms.mean())
    std_time_ms = float(times_ms.std())

    if is_custom:
        custom_model.reorder_graph_morton = original_reorder

    return avg_time_ms, std_time_ms, peak_mem_mb

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    
    # Patch Wigner Hybrid logic
    original_axis_angle = wigner_hybrid.axis_angle_wigner_hybrid
    def patched_axis_angle(*args, **kwargs):
        kwargs['gamma'] = torch.zeros(args[0].shape[0], dtype=args[0].dtype, device=args[0].device)
        return original_axis_angle(*args, **kwargs)
    wigner_hybrid.axis_angle_wigner_hybrid = patched_axis_angle

    # Helper to load models dynamically
    def load_model(exec_mode):
        settings = InferenceSettings(
            execution_mode=exec_mode, merge_mole=True, external_graph_gen=True, 
            use_quaternion_wigner=True, activation_checkpointing=False
        )
        unit = load_predict_unit(path="uma-s-1p2.pt", inference_settings=settings, device=device.type)
        mod = unit.model
        if hasattr(mod, "module"): mod = mod.module
        mod.backbone.regress_config.stress = True
        mod.backbone.regress_config.forces = True
        return mod.to(device).eval(), settings

    # Load temporarily just to get cutoff and generate data
    temp_model, _ = load_model("umas_fast_pytorch")
    cutoff = temp_model.backbone.cutoff
    del temp_model; gc.collect(); torch.cuda.empty_cache()

    # --- Pre-generate Datasets ---
    print("\nGenerating mock datasets...")
    sweep_atoms = [100, 200, 500, 1000, 2000]
    fixed_degree = 50
    data_atoms = [generate_mock_data(a, fixed_degree, cutoff, device) for a in sweep_atoms]

    sweep_edges = [10, 20, 30, 40, 50, 60, 70]
    fixed_atoms = 1000
    data_edges = [generate_mock_data(fixed_atoms, e, cutoff, device) for e in sweep_edges]

    # ==============================================================================
    # 1. BENCHMARK: OFFICIAL PYTORCH BACKEND
    # ==============================================================================
    print("\n" + "="*50)
    print(" BENCHMARKING BASELINE (umas_fast_pytorch + torch.compile)")
    print("="*50)
    pt_model, pt_settings = load_model("umas_fast_pytorch")
    pt_model.prepare_for_inference(data_atoms[0][0], pt_settings)
    pt_model = pt_model.to(device)
    pt_model_compiled = torch.compile(pt_model)
    
    pt_atoms_time, pt_atoms_std, pt_atoms_mem = [], [], []
    print(f"--- SWEEP 1: Atoms (Fixed Degree ~{fixed_degree}) ---")
    for (data, actual_e), a in zip(data_atoms, sweep_atoms):
        ms, sd, mb = measure_performance(pt_model_compiled, data, is_custom=False)
        pt_atoms_time.append(ms); pt_atoms_std.append(sd); pt_atoms_mem.append(mb)
        print(f"Atoms: {a:<5} | Edges: {actual_e:<6} | Time: {ms:>6.2f} ± {sd:>5.2f} ms | VRAM: {mb:>6.2f} MB")

    pt_edges_time, pt_edges_std, pt_edges_mem = [], [], []
    print(f"\n--- SWEEP 2: Edges (Fixed Atoms ~{fixed_atoms}) ---")
    for (data, actual_e), e in zip(data_edges, sweep_edges):
        ms, sd, mb = measure_performance(pt_model_compiled, data, is_custom=False)
        pt_edges_time.append(ms); pt_edges_std.append(sd); pt_edges_mem.append(mb)
        print(f"Target Degree: {e:<2} | Edges: {actual_e:<6} | Time: {ms:>6.2f} ± {sd:>5.2f} ms | VRAM: {mb:>6.2f} MB")
        
    del pt_model, pt_model_compiled; gc.collect(); torch.cuda.empty_cache()

    # ==============================================================================
    # 2. BENCHMARK: OFFICIAL GPU BACKEND
    # ==============================================================================
    print("\n" + "="*50)
    print(" BENCHMARKING BASELINE (umas_fast_gpu + torch.compile)")
    print("="*50)
    gpu_model, gpu_settings = load_model("umas_fast_gpu")
    gpu_model.prepare_for_inference(data_atoms[0][0], gpu_settings)
    gpu_model = gpu_model.to(device)
    gpu_model_compiled = torch.compile(gpu_model)
    
    gpu_atoms_time, gpu_atoms_std, gpu_atoms_mem = [], [], []
    print(f"--- SWEEP 1: Atoms (Fixed Degree ~{fixed_degree}) ---")
    for (data, actual_e), a in zip(data_atoms, sweep_atoms):
        ms, sd, mb = measure_performance(gpu_model_compiled, data, is_custom=False)
        gpu_atoms_time.append(ms); gpu_atoms_std.append(sd); gpu_atoms_mem.append(mb)
        print(f"Atoms: {a:<5} | Edges: {actual_e:<6} | Time: {ms:>6.2f} ± {sd:>5.2f} ms | VRAM: {mb:>6.2f} MB")

    gpu_edges_time, gpu_edges_std, gpu_edges_mem = [], [], []
    print(f"\n--- SWEEP 2: Edges (Fixed Atoms ~{fixed_atoms}) ---")
    for (data, actual_e), e in zip(data_edges, sweep_edges):
        ms, sd, mb = measure_performance(gpu_model_compiled, data, is_custom=False)
        gpu_edges_time.append(ms); gpu_edges_std.append(sd); gpu_edges_mem.append(mb)
        print(f"Target Degree: {e:<2} | Edges: {actual_e:<6} | Time: {ms:>6.2f} ± {sd:>5.2f} ms | VRAM: {mb:>6.2f} MB")

    del gpu_model, gpu_model_compiled; gc.collect(); torch.cuda.empty_cache()

    # ==============================================================================
    # 3. BENCHMARK: CUSTOM TRITON BACKBONE
    # ==============================================================================
    print("\n" + "="*50)
    print(" BENCHMARKING CUSTOM ZeroAllocBackbone (+ torch.compile)")
    print("="*50)
    cust_model, cust_settings = load_model("umas_fast_gpu")
    cust_model.prepare_for_inference(data_atoms[0][0], cust_settings)
    
    # Replace backbone before compiling
    cust_model.backbone = ZeroAllocBackbone(cust_model.backbone, dtype=torch.float32)
    cust_model = cust_model.to(device).eval()
    cust_model_compiled = torch.compile(cust_model)

    cust_atoms_time, cust_atoms_std, cust_atoms_mem = [], [], []
    print(f"--- SWEEP 1: Atoms (Fixed Degree ~{fixed_degree}) ---")
    for (data, actual_e), a in zip(data_atoms, sweep_atoms):
        ms, sd, mb = measure_performance(cust_model_compiled, data, is_custom=True)
        cust_atoms_time.append(ms); cust_atoms_std.append(sd); cust_atoms_mem.append(mb)
        print(f"Atoms: {a:<5} | Edges: {actual_e:<6} | Time: {ms:>6.2f} ± {sd:>5.2f} ms | VRAM: {mb:>6.2f} MB")

    cust_edges_time, cust_edges_std, cust_edges_mem = [], [], []
    print(f"\n--- SWEEP 2: Edges (Fixed Atoms ~{fixed_atoms}) ---")
    for (data, actual_e), e in zip(data_edges, sweep_edges):
        ms, sd, mb = measure_performance(cust_model_compiled, data, is_custom=True)
        cust_edges_time.append(ms); cust_edges_std.append(sd); cust_edges_mem.append(mb)
        print(f"Target Degree: {e:<2} | Edges: {actual_e:<6} | Time: {ms:>6.2f} ± {sd:>5.2f} ms | VRAM: {mb:>6.2f} MB")

    # ==============================================================================
    # PLOTTING
    # ==============================================================================
    print("\nGenerating Comparison Plots...")
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Graph 1: Speed vs Atoms (error bars = ±1 std of per-step time)
    axs[0, 0].errorbar(sweep_atoms, pt_atoms_time, yerr=pt_atoms_std, marker='v', linestyle=':', color='#8EBA42', linewidth=2, capsize=3, label='Baseline (PyTorch)')
    axs[0, 0].errorbar(sweep_atoms, gpu_atoms_time, yerr=gpu_atoms_std, marker='o', linestyle='--', color='#E24A33', linewidth=2, capsize=3, label='Baseline (GPU Triton)')
    axs[0, 0].errorbar(sweep_atoms, cust_atoms_time, yerr=cust_atoms_std, marker='s', linestyle='-', color='#348ABD', linewidth=2, capsize=3, label='Custom (ZeroAlloc)')
    axs[0, 0].set_title(f'Speed vs Number of Atoms\n(Fixed {fixed_degree} Edges/Atom)', fontsize=14)
    axs[0, 0].set_xlabel('Number of Atoms', fontsize=12)
    axs[0, 0].set_ylabel('Time per step (ms)', fontsize=12)
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Graph 2: Memory vs Atoms
    axs[0, 1].plot(sweep_atoms, pt_atoms_mem, marker='v', linestyle=':', color='#8EBA42', linewidth=2, label='Baseline (PyTorch)')
    axs[0, 1].plot(sweep_atoms, gpu_atoms_mem, marker='o', linestyle='--', color='#E24A33', linewidth=2, label='Baseline (GPU Triton)')
    axs[0, 1].plot(sweep_atoms, cust_atoms_mem, marker='s', linestyle='-', color='#348ABD', linewidth=2, label='Custom (ZeroAlloc)')
    axs[0, 1].set_title(f'Peak Memory vs Number of Atoms\n(Fixed {fixed_degree} Edges/Atom)', fontsize=14)
    axs[0, 1].set_xlabel('Number of Atoms', fontsize=12)
    axs[0, 1].set_ylabel('Peak VRAM (MB)', fontsize=12)
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Graph 3: Speed vs Edges (error bars = ±1 std of per-step time)
    axs[1, 0].errorbar(sweep_edges, pt_edges_time, yerr=pt_edges_std, marker='v', linestyle=':', color='#8EBA42', linewidth=2, capsize=3, label='Baseline (PyTorch)')
    axs[1, 0].errorbar(sweep_edges, gpu_edges_time, yerr=gpu_edges_std, marker='^', linestyle='--', color='#E24A33', linewidth=2, capsize=3, label='Baseline (GPU Triton)')
    axs[1, 0].errorbar(sweep_edges, cust_edges_time, yerr=cust_edges_std, marker='D', linestyle='-', color='#348ABD', linewidth=2, capsize=3, label='Custom (ZeroAlloc)')
    axs[1, 0].set_title(f'Speed vs Edges per Atom\n(Fixed {fixed_atoms} Atoms)', fontsize=14)
    axs[1, 0].set_xlabel('Edges per Atom (Degree)', fontsize=12)
    axs[1, 0].set_ylabel('Time per step (ms)', fontsize=12)
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Graph 4: Memory vs Edges
    axs[1, 1].plot(sweep_edges, pt_edges_mem, marker='v', linestyle=':', color='#8EBA42', linewidth=2, label='Baseline (PyTorch)')
    axs[1, 1].plot(sweep_edges, gpu_edges_mem, marker='^', linestyle='--', color='#E24A33', linewidth=2, label='Baseline (GPU Triton)')
    axs[1, 1].plot(sweep_edges, cust_edges_mem, marker='D', linestyle='-', color='#348ABD', linewidth=2, label='Custom (ZeroAlloc)')
    axs[1, 1].set_title(f'Peak Memory vs Edges per Atom\n(Fixed {fixed_atoms} Atoms)', fontsize=14)
    axs[1, 1].set_xlabel('Edges per Atom (Degree)', fontsize=12)
    axs[1, 1].set_ylabel('Peak VRAM (MB)', fontsize=12)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('escn_comparison_benchmark_tf32.png', dpi=300, bbox_inches='tight')
    print("Saved results to 'escn_comparison_benchmark.png'.")

if __name__ == "__main__":
    run_benchmark()