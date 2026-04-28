import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Union

from dyllm.utils.context import Context

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _expand_indices_kernel(output_ptr, cu_counts_ptr, n_groups, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        if pid >= n_groups:
            return
        start = tl.load(cu_counts_ptr + pid)
        end = tl.load(cu_counts_ptr + pid + 1)
        count = end - start
        if count == 0:
            return
        output_offset = output_ptr + start
        for i in range(0, count, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < count
            tl.store(output_offset + offsets, pid, mask=mask)


def triton_repeat_interleave(cu_filtered: torch.Tensor, total_tokens: int, out_tensor: torch.Tensor):
    if not HAS_TRITON or cu_filtered.device.type != "cuda":
        counts = (cu_filtered[1:] - cu_filtered[:-1]).to(torch.long)
        out_tensor.copy_(torch.repeat_interleave(torch.arange(counts.numel(), device=out_tensor.device), counts))
        return out_tensor
    bsz = cu_filtered.shape[0] - 1
    _expand_indices_kernel[(bsz,)](out_tensor, cu_filtered, bsz, BLOCK_SIZE=256)
    return out_tensor


if HAS_TRITON:
    @triton.jit
    def filter_and_count_kernel(scores_ptr, tokens_ptr, pos_ptr, k_values_ptr, thresholds_ptr, out_tokens_ptr, out_pos_ptr, out_counts_ptr, stride_b, stride_l, bsz, max_len, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        k_val = tl.load(k_values_ptr + pid)
        thr_val = tl.load(thresholds_ptr + pid)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < max_len
        row_start = pid * stride_b
        s_val = tl.load(scores_ptr + row_start + offs, mask=mask, other=float("-inf"))
        t_val = tl.load(tokens_ptr + row_start + offs, mask=mask, other=0)
        p_val = tl.load(pos_ptr + row_start + offs, mask=mask, other=0)
        keep = (offs < k_val) & (s_val >= thr_val) & (s_val > float("-inf"))
        tl.store(out_tokens_ptr + row_start + offs, tl.where(keep, t_val, -1), mask=mask)
        tl.store(out_pos_ptr + row_start + offs, tl.where(keep, p_val, 0), mask=mask)
        tl.store(out_counts_ptr + pid, tl.sum(tl.where(keep, 1, 0)))


def launch_filter_and_count(dense_scores, dense_tokens, dense_pos, k_values, thresholds, bsz, max_len):
    if not HAS_TRITON or dense_scores.device.type != "cuda":
        keep = (
            (torch.arange(max_len, device=dense_scores.device).unsqueeze(0) < k_values.unsqueeze(1))
            & (dense_scores >= thresholds.unsqueeze(1))
            & (dense_scores > float("-inf"))
        )
        out_tokens = torch.where(keep, dense_tokens, torch.full_like(dense_tokens, -1))
        out_pos = torch.where(keep, dense_pos, torch.zeros_like(dense_pos))
        out_counts = keep.to(torch.int32).sum(dim=1)
        return out_pos, out_tokens, out_counts

    out_tokens = torch.empty_like(dense_tokens)
    out_pos = torch.empty_like(dense_pos)
    out_counts = torch.empty((bsz,), dtype=torch.int32, device=dense_scores.device)
    block_size = triton.next_power_of_2(max_len)
    filter_and_count_kernel[(bsz,)](
        dense_scores,
        dense_tokens,
        dense_pos,
        k_values,
        thresholds,
        out_tokens,
        out_pos,
        out_counts,
        dense_scores.stride(0),
        dense_scores.stride(1),
        bsz,
        max_len,
        BLOCK_SIZE=block_size,
    )
    return out_pos, out_tokens, out_counts


if HAS_TRITON:
    @triton.jit
    def ragged_to_dense_kernel(scores_ptr, tokens_ptr, pos_ptr, cu_seqlens_ptr, dense_scores_ptr, dense_tokens_ptr, dense_pos_ptr, max_len, n_total_tokens, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        start_idx = tl.load(cu_seqlens_ptr + pid)
        end_idx = tl.load(cu_seqlens_ptr + pid + 1)
        actual_len = end_idx - start_idx
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < actual_len
        ragged_offs = start_idx + offs
        ragged_mask = mask & (ragged_offs < n_total_tokens)
        s_val = tl.load(scores_ptr + ragged_offs, mask=ragged_mask, other=float("-inf"))
        t_val = tl.load(tokens_ptr + ragged_offs, mask=ragged_mask, other=0)
        p_val = tl.load(pos_ptr + ragged_offs, mask=ragged_mask, other=0)
        dense_offs = pid * max_len + offs
        tl.store(dense_scores_ptr + dense_offs, tl.where(mask, s_val, float("-inf")))
        tl.store(dense_tokens_ptr + dense_offs, tl.where(mask, t_val, 0))
        tl.store(dense_pos_ptr + dense_offs, tl.where(mask, p_val, 0))


def launch_ragged_to_dense(scores, tokens, pos, cu_seqlens, bsz, max_len=32):
    device = scores.device
    dense_scores = torch.empty((bsz, max_len), dtype=scores.dtype, device=device)
    dense_tokens = torch.empty((bsz, max_len), dtype=tokens.dtype, device=device)
    dense_pos = torch.empty((bsz, max_len), dtype=pos.dtype, device=device)

    if not HAS_TRITON or device.type != "cuda":
        dense_scores = torch.full((bsz, max_len), float("-inf"), dtype=scores.dtype, device=device)
        dense_tokens = torch.zeros((bsz, max_len), dtype=tokens.dtype, device=device)
        dense_pos = torch.zeros((bsz, max_len), dtype=pos.dtype, device=device)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        idx = torch.arange(max_len, device=device).unsqueeze(0).expand(bsz, max_len)
        mask = idx < lengths.unsqueeze(1)

        flat_idx = cu_seqlens[:-1].unsqueeze(1) + idx
        flat_idx = torch.where(mask, flat_idx, torch.zeros_like(flat_idx))

        dense_scores = torch.where(mask, scores[flat_idx], dense_scores)
        dense_tokens = torch.where(mask, tokens[flat_idx], dense_tokens)
        dense_pos = torch.where(mask, pos[flat_idx], dense_pos)
        
        return dense_scores, dense_tokens, dense_pos

    block_size = triton.next_power_of_2(max_len)
    ragged_to_dense_kernel[(bsz,)](
        scores,
        tokens,
        pos,
        cu_seqlens,
        dense_scores,
        dense_tokens,
        dense_pos,
        max_len,
        scores.numel(),
        BLOCK_SIZE=block_size,
    )
    return dense_scores, dense_tokens, dense_pos


def top_p_logits(logits: torch.Tensor, top_p: Union[float, torch.Tensor, None] = None) -> torch.Tensor:
    if top_p is None:
        return logits
    p_val = top_p if isinstance(top_p, torch.Tensor) else torch.tensor([top_p], device=logits.device)
    p_val = p_val.view(-1, 1)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p_val
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: torch.Tensor, top_k: Union[int, torch.Tensor, None] = None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k_t = top_k if isinstance(top_k, torch.Tensor) else torch.tensor([top_k], device=logits.device)
    max_k = min(int(top_k_t.max().item()), logits.size(-1))
    top_k_vals, _ = torch.topk(logits, max_k, dim=-1)
    k_indices = (top_k_t - 1).clamp(min=0).long().unsqueeze(-1)
    thresholds = top_k_vals.gather(1, k_indices)
    return logits.masked_fill(logits < thresholds, torch.finfo(logits.dtype).min)


class BaseSampler(nn.Module):
    def __init__(self, algorithm: str = "confidence"):
        super().__init__()
        self.algorithm = algorithm

    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def sample_token(self, probs: torch.Tensor, top_tokens: torch.Tensor, temperatures: Optional[torch.Tensor]) -> torch.Tensor:
        return top_tokens

    def compute_scores(self, probs: torch.Tensor, top_probs: torch.Tensor) -> torch.Tensor:
        if self.algorithm == "confidence":
            return top_probs
        if self.algorithm == "margin_confidence":
            top2_probs, _ = probs.topk(k=2, dim=-1)
            return top2_probs[:, 0] - top2_probs[:, 1]
        if self.algorithm == "random":
            return torch.rand_like(top_probs)
        raise ValueError(f"Unsupported algorithm for LLaDA: {self.algorithm}")

    def forward(
        self,
        input_logits: torch.Tensor,
        ctx: Context,
        input_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        temperatures: Optional[torch.Tensor] = None,
        num_transfer: Optional[torch.Tensor] = None,
        thresholds: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        block_size: int = 32,
    ):
        logits = self.adjust_logits(input_logits)
        device = logits.device
        relative_idx, batch_offsets, cu_filtered = input_indices
        bsz = cu_filtered.shape[0] - 1

        total_tokens = relative_idx.shape[0]
        group_ids = torch.empty(total_tokens, device=device, dtype=torch.long)
        triton_repeat_interleave(cu_filtered, total_tokens, out_tensor=group_ids)

        k_values = num_transfer.flatten().clamp_min_(0).to(device) if num_transfer is not None else torch.zeros(bsz, device=device, dtype=torch.long)

        base_offsets = ctx.cu_seqlens_q[:-1]
        global_rows = base_offsets.index_select(0, group_ids) + relative_idx
        cand_logits = logits.index_select(0, global_rows)

        scaled_logits = cand_logits / temperatures.unsqueeze(-1) if temperatures is not None else cand_logits
        if top_k is not None:
            scaled_logits = top_k_logits(scaled_logits, top_k)
        if top_p is not None:
            scaled_logits = top_p_logits(scaled_logits, top_p)

        probs = F.softmax(scaled_logits, dim=-1)
        top_probs, top_tokens = probs.max(dim=-1)
        scores = self.compute_scores(probs, top_probs)
        sampled_all = self.sample_token(probs, top_tokens, temperatures)

        abs_idx = relative_idx + batch_offsets.gather(0, group_ids)
        dense_scores, dense_tokens, dense_pos = launch_ragged_to_dense(scores, sampled_all, abs_idx, cu_filtered, bsz, block_size)

        sorted_scores, sorted_indices = torch.sort(dense_scores, dim=1, descending=True)
        sorted_tokens = torch.gather(dense_tokens, 1, sorted_indices)
        sorted_pos = torch.gather(dense_pos, 1, sorted_indices)

        if thresholds is None:
            thresholds = torch.full((bsz,), float("-inf"), device=device)

        out_pos, out_tokens, out_counts = launch_filter_and_count(sorted_scores, sorted_tokens, sorted_pos, k_values, thresholds, bsz, block_size)
        return out_pos, out_tokens, out_counts


class LLaDASampler(BaseSampler):
    pass


class DreamSampler(BaseSampler):
    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits if logits.size(0) <= 1 else torch.cat([logits[:1], logits[:-1]], dim=0)

    def sample_token(self, probs: torch.Tensor, top_tokens: torch.Tensor, temperatures: Optional[torch.Tensor]) -> torch.Tensor:
        if temperatures is not None and torch.any(temperatures > 0):
            try:
                return dists.Categorical(probs=probs).sample()
            except Exception:
                return top_tokens
        return top_tokens

    def compute_scores(self, probs: torch.Tensor, top_probs: torch.Tensor) -> torch.Tensor:
        if self.algorithm == "entropy":
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            return torch.sum(probs * log_probs, dim=-1)
        if self.algorithm == "origin":
            return torch.rand_like(top_probs)
        raise ValueError(f"Unsupported algorithm for Dream: {self.algorithm}")
