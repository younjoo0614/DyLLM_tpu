import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from dyllm.model_executor.layers.attention import Attention
from dyllm.utils.context import set_context
from dyllm.utils.metadata import set_metadata

def get_reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, num_heads, num_kv_heads, head_dim, scale, max_seqlen_q, max_seqlen_k, device):
    """Compute reference attention using pure PyTorch."""
    o_refs = []
    # q, k, v are assumed to be [total_seqlen, num_heads, head_dim]
    device = q.device
    
    for i in range(len(cu_seqlens_q) - 1):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        
        q_i = q[start_q:end_q].unsqueeze(0).transpose(1, 2) # [1, heads, seq, dim]
        k_i = k[start_k:end_k].unsqueeze(0).transpose(1, 2)
        v_i = v[start_k:end_k].unsqueeze(0).transpose(1, 2)
        
        o_i = F.scaled_dot_product_attention(
            q_i, 
            k_i, 
            v_i,
            is_causal=False
        )
        o_refs.append(o_i.transpose(1, 2).squeeze(0)) # back to [seq, heads, dim]
        
    return torch.cat(o_refs, dim=0)

def test_attention_forward(rank=0):
    device = torch_xla.device() if HAS_XLA else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{rank}] Testing on device: {device}")
    
    if HAS_XLA:
        import torch_xla.runtime as xr
        world_size = xr.world_size() # using the updated api
    else:
        world_size = 1
    rank_idx = rank

    local_num_heads = 4
    local_num_kv_heads = 2
    head_dim = 64
    scale = 1.0 / (head_dim ** 0.5)

    global_num_heads = local_num_heads * world_size
    global_num_kv_heads = local_num_kv_heads * world_size

    # Create dummy varying sequence lengths
    # Batch size 3
    seqlens_q = [256, 512, 378]
    seqlens_k = [412, 567, 812] # q seqlen != k seqlen

    cu_seqlens_q = [0] + [sum(seqlens_q[:i+1]) for i in range(len(seqlens_q))]
    cu_seqlens_k = [0] + [sum(seqlens_k[:i+1]) for i in range(len(seqlens_k))]
    
    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]

    # Model definition
    attention = Attention(
        num_heads=local_num_heads,
        head_dim=head_dim,
        scale=scale,
        num_kv_heads=local_num_kv_heads,
        threshold=0.99
    ).to(device)
    
    # Cache managers base their allocations off this dynamically, match our flattened kv tensor setups.
    attention.v_cache.hidden_dim = local_num_kv_heads * head_dim
    attention.context_cache.hidden_dim = local_num_heads * head_dim

    # Dummy inputs
    # Generate random inputs on CPU first to guarantee identical tensors across all processes
    q_list, k_list, v_list = [], [], []
    for i in range(world_size):
        torch.manual_seed(42 + i)
        q_list.append(torch.randn(total_q, local_num_heads, head_dim, device="cpu", dtype=torch.bfloat16).to(device))
        k_list.append(torch.randn(total_k, local_num_kv_heads, head_dim, device="cpu", dtype=torch.bfloat16).to(device))
        v_list.append(torch.randn(total_k, local_num_kv_heads, head_dim, device="cpu", dtype=torch.bfloat16).to(device))

    q = q_list[rank_idx]
    k = k_list[rank_idx]
    v = v_list[rank_idx]

    global_q = torch.cat(q_list, dim=1)
    global_k = torch.cat(k_list, dim=1)
    global_v = torch.cat(v_list, dim=1)

    if rank == 0:
        print(f"[{rank}] Local q: {q.shape}, k: {k.shape}, v: {v.shape}")
        print(f"[{rank}] Global q: {global_q.shape}, k: {global_k.shape}, v: {global_v.shape}")

    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    
    set_context(
        is_full=True,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_k=cu_seqlens_k_t,
        max_seqlen_q=max(seqlens_q),
        max_seqlen_k=max(seqlens_k),
        total_seqlen=total_q,
        total_seqlen_k=total_k,
        context_lens_cpu=seqlens_q,
        context_lens=torch.tensor(seqlens_q, dtype=torch.int32, device=device)
    )
    
    running_seqs_list = list(range(len(seqlens_q)))
    running_seqs_tensor = torch.arange(len(seqlens_q), dtype=torch.int32, device=device)
    finished_seqs_tensor = torch.tensor([], dtype=torch.int32, device=device)

    set_metadata(
        all_seqs=running_seqs_list,
        running_seqs=running_seqs_list,
        finished_seqs=finished_seqs_tensor,
    )
    # Manually inject the tensor since set_metadata might not take it directly
    from dyllm.utils.metadata import get_metadata
    get_metadata().running_seqs_tensor = running_seqs_tensor

    # Forward pass (use 'k' and 'v' without artificial expansions inside test layer scope, the layer handles internally via num_kv_heads properly)
    out = attention(q, k, v)
    
    if HAS_XLA:
        torch_xla.sync()
    
    global_k_ref = global_k.repeat_interleave(global_num_heads // global_num_kv_heads, dim=1) if global_q.shape[1] != global_k.shape[1] else global_k
    global_v_ref = global_v.repeat_interleave(global_num_heads // global_num_kv_heads, dim=1) if global_q.shape[1] != global_v.shape[1] else global_v
    
    # Check correctness
    out_ref = get_reference_attention(
        global_q, global_k_ref, global_v_ref,
        cu_seqlens_q_t, cu_seqlens_k_t,
        global_num_heads, global_num_kv_heads, head_dim, scale, max(seqlens_q), max(seqlens_k), device
    )
    
    # Compute difference for the local slice
    out_ref_local_slice = out_ref[:, rank_idx * local_num_heads : (rank_idx + 1) * local_num_heads, :]
    
    if rank == 0:
        print(f"[{rank}] Output local: {out.shape}, Output global ref: {out_ref.shape}, sliced ref: {out_ref_local_slice.shape}")
        
    diff = torch.abs(out.float() - out_ref_local_slice.float()).max().item()
    
    print(f"[{rank}] Max difference between implementation and reference: {diff}")
    
    if diff < 1e-2:
        print(f"[{rank}] Test Passed!")
    else:
        print(f"[{rank}] Test Failed! Max diff is too large.")

def _mp_fn(rank, flags):
    test_attention_forward(rank)

if __name__ == '__main__':
    if HAS_XLA:
        print("Running distributed TPU test...")
        xmp.spawn(_mp_fn, args=({},), nprocs=None, start_method='fork')
    else:
        print("Running single-device fallback test...")
        test_attention_forward(0)
