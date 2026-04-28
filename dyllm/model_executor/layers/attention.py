import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from dyllm.utils.context import get_context, set_context
from dyllm.utils.metadata import get_metadata
from dyllm.engine.cache_manager import CacheManager


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        threshold: float = 0.99,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.threshold = threshold
        self.context_cache = CacheManager(self.num_heads * self.head_dim)
        self.v_cache = CacheManager(self.num_kv_heads * self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_context()
        metadata = get_metadata()
        num_repeat = self.num_heads // self.num_kv_heads

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        cu_q_cpu = ctx.cu_seqlens_q_cpu  
        cu_k_cpu = ctx.cu_seqlens_k_cpu
       
        total_q = q.shape[0]
        total_k = k.shape[0]
        bounds_q = torch.tensor(cu_q_cpu[1:], dtype=torch.int64, device=q.device)
        bounds_k = torch.tensor(cu_k_cpu[1:], dtype=torch.int64, device=q.device)
        rows_q = torch.arange(total_q, dtype=torch.int64, device=q.device)
        rows_k = torch.arange(total_k, dtype=torch.int64, device=q.device)
        seg_q = torch.searchsorted(bounds_q, rows_q, right=True)
        seg_k = torch.searchsorted(bounds_k, rows_k, right=True)

        def _run_sdpa(q_, k_, v_, seg_q_=None):
            # Pass ``seg_q_`` when ``q_`` is row-pruned; otherwise mask
            # broadcasting silently expands the output back to total_q.
            if seg_q_ is None:
                seg_q_ = seg_q

            # GQA: replicate KV heads to match Q head count.
            n_rep = q_.shape[1] // k_.shape[1]
            if n_rep > 1:
                k_ = k_.repeat_interleave(n_rep, dim=1)
                v_ = v_.repeat_interleave(n_rep, dim=1)

            allow_mask = seg_q_.unsqueeze(1) == seg_k.unsqueeze(0)
            q_h = q_.transpose(0, 1)  # [H, L, D]
            k_h = k_.transpose(0, 1)
            v_h = v_.transpose(0, 1)
            out = F.scaled_dot_product_attention(
                q_h, k_h, v_h,
                attn_mask=allow_mask.unsqueeze(0),
                scale=self.scale,
                is_causal=False,
            )
            return out.transpose(0, 1).to(q_.dtype)

        if ctx.is_full:
            
            o = _run_sdpa(q, k, v) # [L, H, D]
            
            c_cache = o.flatten(-2, -1).contiguous()  # sum(L), H * D
            self.context_cache.reset_full(
                c_cache, seq_ids=metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs
            )
            v_flat = v.flatten(-2, -1).contiguous()
            self.v_cache.reset_full(v_flat, metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs)

            self.context_cache.finish(metadata.finished_seqs)
            self.v_cache.finish(metadata.finished_seqs)

            return o

        else:
            is_q_pruned = ctx.idx_salient_row_k is not None

            v_cache = self.v_cache.get_seqs(metadata.running_seqs_tensor).view(-1, self.num_kv_heads, self.head_dim)
            v_delta_sparse = torch.zeros_like(k)

            if is_q_pruned:
                v_delta = v - v_cache[ctx.idx_salient_row_k]
                v_delta_sparse[ctx.idx_salient_row_k] = v_delta
                v_cache = v_cache.index_copy(0, ctx.idx_salient_row_k, v)
                self.v_cache.scatter_update(metadata.running_seqs_tensor, ctx.idx_salient_row_k, v.flatten(-2, -1))
            else:
                v_delta = v - v_cache[ctx.idx_salient_row]
                v_delta_sparse[ctx.idx_salient_row] = v_delta
                v_cache = v_cache.index_copy(0, ctx.idx_salient_row, v)
                self.v_cache.scatter_update(metadata.running_seqs_tensor, ctx.idx_salient_row, v.flatten(-2, -1))

            # Salient-subset Q requires matching segment ids.
            seg_q_salient = seg_q.index_select(0, ctx.idx_salient_row.to(torch.long))
            o_salient = _run_sdpa(q[ctx.idx_salient_row], k, v_cache, seg_q_=seg_q_salient)  # [num_salient, H, D]

            c_cache = (
                self.context_cache.get_seqs_block(metadata.running_seqs_tensor).view(-1, self.num_heads, self.head_dim)
                if is_q_pruned
                else self.context_cache.get_seqs(metadata.running_seqs_tensor).view(-1, self.num_heads, self.head_dim)
            )

            delta_c = _run_sdpa(q, k, v_delta_sparse)

            c_flat = c_cache.flatten(-2, -1)
            new_c_flat = c_flat + delta_c.flatten(-2, -1)
            new_c_flat[ctx.idx_salient_row] = o_salient.flatten(-2, -1)

            if dist.is_initialized():
                num = (c_flat * new_c_flat).sum(dim=-1)
                den_c = (c_flat * c_flat).sum(dim=-1)
                den_new_c = (new_c_flat * new_c_flat).sum(dim=-1)

                from dyllm.tpu.collectives import all_reduce_sum
                num = all_reduce_sum(num)
                den_c = all_reduce_sum(den_c)
                den_new_c = all_reduce_sum(den_new_c)

                cos_sim = num / (torch.sqrt(den_c) * torch.sqrt(den_new_c) + 1e-8)
            else:
                cos_sim = F.cosine_similarity(c_flat, new_c_flat, dim=-1)  # [L]

            idxs = torch.nonzero(cos_sim < self.threshold, as_tuple=False).squeeze(-1).contiguous()

            if dist.is_initialized():
                num_valid_tensor = torch.tensor([idxs.numel()], dtype=torch.long, device=idxs.device)
                if num_valid_tensor.device.type == "xla":
                    pass
                else:
                    dist.broadcast(num_valid_tensor, src=0)
                    if dist.get_rank() != 0:
                        idxs = torch.empty(num_valid_tensor.item(), dtype=torch.long, device=idxs.device)
                    dist.broadcast(idxs, src=0)

            num_valid = idxs.numel()

            # Bucket-pad idxs to next power-of-two per sequence so XLA sees
            # only O(log N) shapes across steps. Padding repeats the
            # segment's last valid index (idempotent for writes; reads land
            # on a valid row we discard). Per-sequence padding is required
            # to avoid leaking indices into another sequence's KV range.
            if num_valid > 0 and ctx.cu_seqlens_q_cpu is not None:
                num_segments = len(ctx.cu_seqlens_q_cpu) - 1
                seg_bounds = ctx.cu_seqlens_q_cpu
                idxs_cpu = idxs.detach().cpu().tolist()
                seg_counts = [0] * num_segments
                seg_last_idx = [None] * num_segments
                j = 0
                for i, x in enumerate(idxs_cpu):
                    while j < num_segments and x >= seg_bounds[j + 1]:
                        j += 1
                    if j == num_segments:
                        break
                    seg_counts[j] += 1
                    seg_last_idx[j] = x

                # Bucket size per segment (>=32, capped by segment length).
                seg_buckets = []
                for s in range(num_segments):
                    cnt = seg_counts[s]
                    if cnt == 0:
                        seg_buckets.append(0)
                        continue
                    seg_len = int(seg_bounds[s + 1] - seg_bounds[s])
                    bucket = 1 << max(5, (cnt - 1).bit_length())
                    if bucket > seg_len:
                        bucket = seg_len
                    seg_buckets.append(bucket)

                total_pad = sum(b - c for b, c in zip(seg_buckets, seg_counts))
                if total_pad > 0:
                    # Build padded idxs on CPU; one transfer to device.
                    padded = []
                    cursor = 0
                    for s in range(num_segments):
                        cnt = seg_counts[s]
                        if cnt > 0:
                            padded.extend(idxs_cpu[cursor : cursor + cnt])
                            cursor += cnt
                            extra = seg_buckets[s] - cnt
                            if extra > 0:
                                padded.extend([seg_last_idx[s]] * extra)
                    idxs = torch.tensor(
                        padded, dtype=idxs.dtype, device=idxs.device
                    )
                    num_valid = len(padded)

            cu_salientlens = torch.searchsorted(idxs, ctx.cu_seqlens_q).to(torch.int32)

            if ctx.idx_salient_row_k is not None:
                self.context_cache.reset_block(new_c_flat, metadata.running_seqs_tensor)
                tids = torch.arange(num_valid, device=idxs.device)
                group_idx = torch.bucketize(tids, cu_salientlens[1:], right=True)
                offset_val = ctx.cu_promptlens[group_idx + 1].to(idxs.dtype)
                ctx.idx_salient_row_k = idxs + offset_val
            else:
                self.context_cache.reset_full(
                    new_c_flat, metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs
                )

            ctx.idx_salient_row = idxs.to(torch.long)
            ctx.cu_salientlens = cu_salientlens

            self.v_cache.finish(metadata.finished_seqs)
            self.context_cache.finish(metadata.finished_seqs)

            return new_c_flat.view(-1, self.num_heads, self.head_dim)
