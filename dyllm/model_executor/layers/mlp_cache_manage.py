import torch
from torch import nn

from dyllm.utils.context import get_context
from dyllm.utils.metadata import get_metadata
from dyllm.engine.cache_manager import CacheManager


class MLPcache(nn.Module):

    def __init__(self, hidden_dim: int = 0):
        super().__init__()
        self.cache_manager = CacheManager(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor):
        ctx = get_context()
        metadata = get_metadata()

        if ctx.is_full:
            # Cache full hidden state and return x directly: the cache
            # read-back can corrupt batched per-sequence ordering on XLA.
            self.cache_manager.reset_full(x, metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs)
            self.cache_manager.finish(metadata.finished_seqs)
            return x
        else:
            # Sparse step: merge prev hidden state with new salient rows
            # locally to avoid an XLA cache write→read round-trip.
            # idx_salient_row   → q-arranged buffer rows
            # idx_salient_row_k → k-arranged cache rows
            local_idx = ctx.idx_salient_row.to(torch.long)
            cache_idx = (
                ctx.idx_salient_row_k if ctx.idx_salient_row_k is not None else ctx.idx_salient_row
            )
            if ctx.idx_salient_row_k is not None:
                prev = self.cache_manager.get_seqs_block(metadata.running_seqs_tensor)
            else:
                prev = self.cache_manager.get_seqs(metadata.running_seqs_tensor)
            out = prev.index_copy(0, local_idx, x)
            # Persist salient values; write-only so XLA can't reorder.
            self.cache_manager.scatter_update(
                metadata.running_seqs_tensor,
                cache_idx,
                x,
            )
            self.cache_manager.finish(metadata.finished_seqs)
            return out
