"""Cross-process collectives for XLA tensors using the gloo backend.

PJRT multi-process with TPU_VISIBLE_CHIPS isolates each process to a
single TPU chip.  The native XLA collectives (xm.all_reduce, xm.all_gather)
only operate within a single process and do NOT communicate across processes.

This module provides replacements that:
  1. Materialize the XLA tensor (torch_xla.sync)
  2. Copy to CPU
  3. Perform the collective via gloo (torch.distributed)
  4. Copy back to the XLA device
"""

import torch
import torch.distributed as dist


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce (SUM) across ranks. Works for XLA, CPU and CUDA tensors."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return tensor

    if tensor.device.type == "xla":
        import torch_xla
        torch_xla.sync()
        cpu_t = tensor.cpu()
        dist.all_reduce(cpu_t, op=dist.ReduceOp.SUM)
        return cpu_t.to(tensor.device)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_cat(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather and concatenate along *dim*."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return tensor

    world_size = dist.get_world_size()

    if tensor.device.type == "xla":
        import torch_xla
        torch_xla.sync()
        cpu_t = tensor.cpu()
        gather_list = [torch.empty_like(cpu_t) for _ in range(world_size)]
        dist.all_gather(gather_list, cpu_t)
        gathered = torch.cat(gather_list, dim=dim)
        return gathered.to(tensor.device)

    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)
