import torch
import gc
import torch.cuda.nvtx as nvtx

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as wigner_hybrid

from wrapper import ZeroAllocBackbone

# ==============================================================================
# EVALUATION HELPERS
# ==============================================================================

def get_tensor(out_dict, key_suffix):
    for k, v in out_dict.items():
        if key_suffix in k:
            if isinstance(v, dict): return list(v.values())[0]
            return v
    return None

def report_errors(name, actual, reference):
    diff = (actual.detach().float() - reference.detach().float()).abs()
    max_abs = diff.max().item(); mean_abs = diff.mean().item()
    ref_abs = reference.detach().float().abs() + 1e-7
    max_rel = (diff / ref_abs).max().item(); mean_rel = (diff / ref_abs).mean().item()
    print(f"{name:<35s} | MaxAbs: {max_abs:.4e} | MeanAbs: {mean_abs:.4e} | MaxRel: {max_rel:.4e} | MeanRel: {mean_rel:.4e}")

def benchmark_end_to_end(name, model, data, warmup=5, iters=20):
    for _ in range(warmup):
        data.pos.grad = None
        data.cell.grad = None
        _ = model(data)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    nvtx.range_push(f"{name} - End to End")
    for _ in range(iters):
        data.pos.grad = None
        data.cell.grad = None
        _ = model(data)
    nvtx.range_pop()
    end.record()
    torch.cuda.synchronize()
    
    avg_time = start.elapsed_time(end) / iters
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"{name:<25s} | Time/Step: {avg_time:>6.2f} ms | Peak VRAM: {peak_mem:>7.2f} MB")


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)

    print("=== eSCN Benchmark: Official API via load_predict_unit ===")
    
    # Patch Wigner Hybrid logic
    original_axis_angle = wigner_hybrid.axis_angle_wigner_hybrid
    def patched_axis_angle(*args, **kwargs):
        kwargs['gamma'] = torch.zeros(args[0].shape[0], dtype=args[0].dtype, device=args[0].device)
        return original_axis_angle(*args, **kwargs)
    wigner_hybrid.axis_angle_wigner_hybrid = patched_axis_angle

    CKPT_PATH = "uma-s-1p2.pt"
    print(f"Loading {CKPT_PATH} via load_predict_unit...")
    
    settings = InferenceSettings(
        execution_mode="umas_fast_gpu",
        merge_mole=True,
        external_graph_gen=True, 
        use_quaternion_wigner=True,
        activation_checkpointing=False
    )
    
    predict_unit = load_predict_unit(path=CKPT_PATH, inference_settings=settings, device=device.type)
    official_model = predict_unit.model
    if hasattr(official_model, "module"): official_model = official_model.module
        
    official_model = official_model.to(device)
    official_model.eval()

    # FORCE STRESS TRACKING FOR THE BENCHMARK
    official_model.backbone.regress_config.stress = True
    official_model.backbone.regress_config.forces = True
    official_model.backbone.regress_config.direct_forces = False
    official_model.backbone.regress_config.direct_stress = False

    num_atoms = 600
    target_degree = 40
    cutoff = official_model.backbone.cutoff
    
    pos_raw = torch.rand(num_atoms, 3, dtype=torch.float32, device=device)
    dist_matrix = torch.cdist(pos_raw, pos_raw)
    kth_dist = dist_matrix.flatten().kthvalue(num_atoms * target_degree + num_atoms).values.item()
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
    E = idx_i.shape[0]

    rand_offsets = torch.randint(-1, 2, (E, 3), dtype=torch.float32, device=device)
    mock_charge = torch.zeros(1, dtype=torch.long, device=device)
    mock_spin = torch.zeros(1, dtype=torch.long, device=device)
    mock_natoms = torch.tensor([num_atoms], dtype=torch.long, device=device)

    data_mock = AtomicData(
        pos=pos_base.clone().requires_grad_(True), atomic_numbers=Z,
        cell=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True),
        pbc=torch.zeros(1, 3, dtype=torch.bool, device=device),
        natoms=mock_natoms,                                         
        edge_index=torch.stack([idx_i, idx_j], dim=0),
        cell_offsets=rand_offsets,
        nedges=torch.tensor([E], dtype=torch.long, device=device),
        charge=mock_charge, spin=mock_spin,                                             
        fixed=torch.zeros(num_atoms, dtype=torch.long, device=device),
        tags=torch.zeros(num_atoms, dtype=torch.long, device=device),
        batch=batch, sid=["mock_0"], dataset=["oc20"],
    )
    data_mock.batch_full = batch
    data_mock.atomic_numbers_full = Z

    official_model.prepare_for_inference(data_mock, settings)
    official_model = official_model.to(device)

    print("\n--- STAGE 1: OFFICIAL MODEL EXECUTION ---")
    out_off = official_model(data_mock)
    
    e_off_ref = get_tensor(out_off, "energy")
    f_off_ref = get_tensor(out_off, "forces")
    s_off_ref = get_tensor(out_off, "stress")

    benchmark_end_to_end("Official Compiled", official_model, data_mock)

    print("\n--- STAGE 2: CUSTOM BACKBONE EXECUTION ---")
    old_bb = official_model.backbone
    official_model.backbone = ZeroAllocBackbone(old_bb, dtype=torch.float32)
    
    del old_bb
    gc.collect(); torch.cuda.empty_cache()

    data_mock.pos.grad = None
    data_mock.cell.grad = None
    out_cust = official_model(data_mock)
    
    e_cust = get_tensor(out_cust, "energy")
    f_cust = get_tensor(out_cust, "forces")
    s_cust = get_tensor(out_cust, "stress")

    print("\n" + "=" * 80)
    print("VERIFYING NUMERICAL EQUIVALENCE (CPU Compare)")
    print("=" * 80)
    report_errors("Energy Scalar", e_cust.cpu(), e_off_ref.cpu())
    report_errors("Forces (dE/dPos)", f_cust.cpu(), f_off_ref.cpu())
    if s_off_ref is not None:
        report_errors("Stress Tensor", s_cust.cpu(), s_off_ref.cpu())

    print("\n" + "=" * 80)
    print("BENCHMARKING")
    print("=" * 80)
    benchmark_end_to_end("ZeroAllocBackbone", official_model, data_mock)

if __name__ == "__main__":
    run_benchmarks()