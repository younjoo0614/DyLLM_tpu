import torch


_LIB_DEF = torch.library.Library("dyllm_tpu", "DEF")
try:
    _LIB_DEF.define(
        "varlen_attention(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, Tensor cu_seqlens_k, float scale) -> Tensor"
    )
    _LIB_DEF.define(
        "cache_get_seqs(Tensor cache, Tensor seq_starts, Tensor cu_seqlens, Tensor seq_ids, int total_seqlen) -> Tensor"
    )
    _LIB_DEF.define(
        "cache_reset_full(Tensor cache, Tensor src, Tensor seq_starts, Tensor cu_seqlens, Tensor seq_ids) -> Tensor"
    )
    _LIB_DEF.define(
        "cache_scatter_update(Tensor cache, Tensor src, Tensor row_idx, Tensor cu_seqlens, Tensor seq_starts, Tensor seq_ids) -> Tensor"
    )
    _LIB_DEF.define(
        "cache_get_block(Tensor cache, Tensor seq_starts, Tensor cu_seqlens_k, Tensor cu_seqlens_q, Tensor seq_ids, int total_seqlen) -> Tensor"
    )
    _LIB_DEF.define(
        "cache_reset_block(Tensor cache, Tensor src, Tensor seq_starts, Tensor cu_seqlens_k, Tensor cu_seqlens_q, Tensor seq_ids) -> Tensor"
    )
except Exception:
    pass


@torch.library.impl("dyllm_tpu::varlen_attention", "CompositeExplicitAutograd")
def _varlen_attention_impl(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    outs = []
    batch = cu_seqlens_q.numel() - 1
    for i in range(batch):
        q0 = int(cu_seqlens_q[i].item())
        q1 = int(cu_seqlens_q[i + 1].item())
        k0 = int(cu_seqlens_k[i].item())
        k1 = int(cu_seqlens_k[i + 1].item())
        qi = q[q0:q1]
        ki = k[k0:k1]
        vi = v[k0:k1]
        scores = torch.einsum("qhd,khd->hqk", qi.float(), ki.float()) * scale
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        outs.append(torch.einsum("hqk,khd->qhd", probs, vi))
    return torch.cat(outs, dim=0) if outs else q.new_empty((0, q.size(1), q.size(2)))


def _row_to_seq_idx(rows: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    # cu_seqlens has length num_seqs + 1.  Find the segment each global row
    # belongs to in O(N log B) via searchsorted on the upper bounds.
    cu_long = cu_seqlens.to(dtype=torch.int64)
    bounds = cu_long[1:].contiguous()
    seg = torch.searchsorted(bounds, rows, right=False)
    # Clamp in case of trailing rows past the last bound (shouldn't happen,
    # but keeps the gather safe).
    seg = seg.clamp(max=bounds.numel() - 1)
    return seg, cu_long


@torch.library.impl("dyllm_tpu::cache_get_seqs", "CompositeExplicitAutograd")
def _cache_get_seqs_impl(cache, seq_starts, cu_seqlens, seq_ids, total_seqlen):
    total = int(total_seqlen)
    if total == 0:
        return cache.new_empty((0, cache.size(1)))
    rows = torch.arange(total, dtype=torch.int64, device=cache.device)
    seg, cu_long = _row_to_seq_idx(rows, cu_seqlens)
    seq_id_per_row = seq_ids.to(dtype=torch.int64).index_select(0, seg)
    local = rows - cu_long.index_select(0, seg)
    cache_row = seq_starts.to(dtype=torch.int64).index_select(0, seq_id_per_row) + local
    return cache.index_select(0, cache_row)


@torch.library.impl("dyllm_tpu::cache_reset_full", "CompositeExplicitAutograd")
def _cache_reset_full_impl(cache, src, seq_starts, cu_seqlens, seq_ids):
    total = src.size(0)
    if total == 0:
        return cache
    rows = torch.arange(total, dtype=torch.int64, device=cache.device)
    seg, cu_long = _row_to_seq_idx(rows, cu_seqlens)
    seq_id_per_row = seq_ids.to(dtype=torch.int64).index_select(0, seg)
    local = rows - cu_long.index_select(0, seg)
    target_row = seq_starts.to(dtype=torch.int64).index_select(0, seq_id_per_row) + local
    return cache.index_copy(0, target_row, src.to(dtype=cache.dtype))


@torch.library.impl("dyllm_tpu::cache_scatter_update", "CompositeExplicitAutograd")
def _cache_scatter_update_impl(cache, src, row_idx, cu_seqlens, seq_starts, seq_ids):
    out = cache.clone()
    row_idx_cpu = row_idx.detach().cpu()
    cu_cpu = cu_seqlens.detach().cpu()
    seq_ids_cpu = seq_ids.detach().cpu()
    for i in range(row_idx_cpu.numel()):
        pos = int(row_idx_cpu[i].item())
        seq_idx = int(torch.searchsorted(cu_cpu, torch.tensor(pos + 1)) - 1)
        seq_id = int(seq_ids_cpu[seq_idx].item())
        local = pos - int(cu_cpu[seq_idx].item())
        out[int(seq_starts[seq_id].item()) + local] = src[i]
    return out


def _block_target_rows(seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids, total_q, device):
    # For each q-arranged row r, owning seg s satisfies cu_q[s] <= r < cu_q[s+1].
    # Cache slot is start_seq[s] + (k_len_s - q_len_s) + (r - cu_q[s])
    #                            = start_seq[s] + cu_k[s+1] - cu_q[s+1] + r.
    rows = torch.arange(total_q, dtype=torch.int64, device=device)
    seg, cu_q_long = _row_to_seq_idx(rows, cu_seqlens_q)
    cu_k_long = cu_seqlens_k.to(dtype=torch.int64)
    seq_id_per_row = seq_ids.to(dtype=torch.int64).index_select(0, seg)
    base = (
        seq_starts.to(dtype=torch.int64).index_select(0, seq_id_per_row)
        + cu_k_long.index_select(0, seg + 1)
        - cu_q_long.index_select(0, seg + 1)
    )
    return base + rows


@torch.library.impl("dyllm_tpu::cache_get_block", "CompositeExplicitAutograd")
def _cache_get_block_impl(cache, seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids, total_seqlen):
    total = int(total_seqlen)
    if total == 0:
        return cache.new_empty((0, cache.size(1)))
    cache_row = _block_target_rows(
        seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids, total, cache.device
    )
    return cache.index_select(0, cache_row)


@torch.library.impl("dyllm_tpu::cache_reset_block", "CompositeExplicitAutograd")
def _cache_reset_block_impl(cache, src, seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids):
    total = src.size(0)
    if total == 0:
        return cache
    cache_row = _block_target_rows(
        seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids, total, cache.device
    )
    return cache.index_copy(0, cache_row, src.to(dtype=cache.dtype))


def varlen_attention_op(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    return torch.ops.dyllm_tpu.varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, float(scale))


def cache_get_seqs_op(cache, seq_starts, cu_seqlens, seq_ids, total_seqlen):
    return torch.ops.dyllm_tpu.cache_get_seqs(cache, seq_starts, cu_seqlens, seq_ids, int(total_seqlen))


def cache_reset_full_op(cache, src, seq_starts, cu_seqlens, seq_ids):
    return torch.ops.dyllm_tpu.cache_reset_full(cache, src, seq_starts, cu_seqlens, seq_ids)


def cache_scatter_update_op(cache, src, row_idx, cu_seqlens, seq_starts, seq_ids):
    return torch.ops.dyllm_tpu.cache_scatter_update(cache, src, row_idx, cu_seqlens, seq_starts, seq_ids)


def cache_get_block_op(cache, seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids, total_seqlen):
    return torch.ops.dyllm_tpu.cache_get_block(cache, seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids, int(total_seqlen))


def cache_reset_block_op(cache, src, seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids):
    return torch.ops.dyllm_tpu.cache_reset_block(cache, src, seq_starts, cu_seqlens_k, cu_seqlens_q, seq_ids)
