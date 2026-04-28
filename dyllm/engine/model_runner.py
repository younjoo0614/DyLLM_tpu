import pickle
import torch
import os
from datetime import timedelta
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from dyllm.config import Config
from dyllm.engine.sequence import Sequence
from dyllm.engine.cache_manager import CacheManager
from dyllm.model_executor.models import LLaDAForDLM, DreamForDLM

from dyllm.model_executor.layers.sampler import LLaDASampler, DreamSampler

from dyllm.utils.context import set_context, get_context, reset_context
from dyllm.utils.weight_loader import load_model
from dyllm.utils.metadata import get_metadata

xm = None


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.backend = config.backend
        self.xm = None
        if self.backend == "xla":
            try:
                import torch_xla.core.xla_model as _xm

                self.xm = _xm
            except Exception:
                self.xm = None

        if self.backend == "xla":
            if self.xm is None:
                raise RuntimeError("XLA backend selected but torch_xla is not importable")
            # With PJRT multiprocess, each process should bind to its local XLA device.
            self.device = self.xm.xla_device()
        elif self.backend == "cuda":
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")
        self.is_cuda = self.device.type == "cuda"

        dist_backend = "nccl" if self.backend == "cuda" else "gloo"
        dist_timeout_s = int(os.environ.get("DYLLM_DIST_TIMEOUT", "120"))
        dist_timeout = timedelta(seconds=max(10, dist_timeout_s))
        if not dist.is_initialized():
            if self.world_size == 1:
                rendezvous_file = f"/tmp/dyllm_dist_{os.getpid()}"
                dist.init_process_group(
                    backend=dist_backend,
                    init_method=f"file://{rendezvous_file}",
                    world_size=1,
                    rank=0,
                    timeout=dist_timeout,
                )
            else:
                port = config.dist_port if config.dist_port is not None else int(os.environ.get("DYLLM_DIST_PORT", "29500"))
                dist.init_process_group(
                    backend=dist_backend,
                    init_method=f"tcp://127.0.0.1:{port}",
                    world_size=self.world_size,
                    rank=rank,
                    timeout=dist_timeout,
                )

        if self.is_cuda:
            torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        if hf_config.model_type == "Dream":
            self.model = DreamForDLM(hf_config, config.threshold)
            self.sampler = DreamSampler("entropy")
        elif hf_config.model_type == "llada":
            self.model = LLaDAForDLM(hf_config, config.threshold)
            self.sampler = LLaDASampler("confidence")
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        load_model(self.model, config.model)
        self.model.to(self.device)
        if self.backend == "xla" and self.xm is not None:
            import torch_xla
            torch_xla.sync()

        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="dyllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="dyllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.device.type == "xla" and self.xm is not None:
            import torch_xla
            torch_xla.sync()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _to_device(self, t: torch.Tensor | None):
        if t is None:
            return None
        return t.to(device=self.device)

    def _pre_allocate_caches(self, is_full: bool):
        """Pre-allocate CacheManager storage before forward so the per-layer
        XLA graph stays identical across steps and avoids recompilation."""
        if not is_full:
            return
        metadata = get_metadata()
        ctx = get_context()
        for module in self.model.modules():
            if isinstance(module, CacheManager):
                new_ids = []
                new_lens = []
                for i, seq_id in enumerate(metadata.running_seqs):
                    if seq_id not in module._seq_to_start:
                        new_ids.append(seq_id)
                        new_lens.append(ctx.context_lens_cpu[i])
                if new_ids:
                    module._vectorized_allocate_new_seqs(new_ids, new_lens)

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def prepare_full(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        idx_salient_row = []
        context_lens = []
        seq_ids = []
        metadata = get_metadata()
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq)
            positions.extend(list(range(seqlen)))
            seqlen_q = seqlen
            seqlen_k = seqlen
            context_lens.append(seqlen)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            idx_salient_row.extend(list(range(seqlen)))
            seq_ids.append(seq.seq_id)
        input_ids = self._to_device(torch.tensor(input_ids, dtype=torch.int64))
        positions = self._to_device(torch.tensor(positions, dtype=torch.int64))
        cu_seqlens_q_list = cu_seqlens_q
        cu_seqlens_k_list = cu_seqlens_k
        cu_seqlens_q = self._to_device(torch.tensor(cu_seqlens_q, dtype=torch.int32))
        cu_seqlens_k = self._to_device(torch.tensor(cu_seqlens_k, dtype=torch.int32))
        context_lens_list = context_lens
        context_lens = self._to_device(torch.tensor(context_lens, dtype=torch.int32))
        metadata.running_seqs = seq_ids
        metadata.running_seqs_tensor = self._to_device(torch.tensor(seq_ids, dtype=torch.long))
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_cpu=cu_seqlens_q_list,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_k_cpu=cu_seqlens_k_list,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            context_lens=context_lens,
            context_lens_cpu=context_lens_list,
            total_seqlen=cu_seqlens_q_list[-1],
        )
        return input_ids, positions

    def prepare_sparse(self, seqs: list[Sequence]):
        metadata = get_metadata()
        input_ids = []
        idx_updated_rows = []
        idx_salient_rows = []
        idx_salient_rows_k = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        cu_promptlens = [0]
        cu_updatedlens = [0]
        cu_salientlens = [0]
        positions = []
        positions_k = []
        seq_ids = []
        context_lens = []
        max_seqlen_q = 0
        max_seqlen_k = 0
        for seq in seqs:
            seqlen = len(seq)
            seqlen_q = seq.max_new_tokens
            cu_promptlens.append(seq.num_prompt_tokens + cu_promptlens[-1])
            if seq.processed_steps > 4 and seq.processed_steps % 8 != 0:
                input_ids.extend(seq[-seq.max_new_tokens :])
                positions.extend(list(range(seqlen - seq.max_new_tokens, seqlen)))
                idx_salient_rows.extend(
                    [
                        i - seq.num_prompt_tokens + cu_seqlens_q[-1]
                        for i in seq.last_token_pos
                        if i >= seqlen - seq.max_new_tokens
                    ]
                )
                idx_salient_rows_k.extend([i + cu_seqlens_k[-1] for i in seq.last_token_pos])
                seqlen_q = seq.max_new_tokens
            else:
                input_ids.extend(seq)
                positions.extend(list(range(seqlen)))
                idx_salient_rows.extend([i + cu_seqlens_q[-1] for i in seq.last_token_pos])
                idx_salient_rows_k = None
                seqlen_q = seqlen
            positions_k.extend([i for i in seq.last_token_pos])
            idx_updated_rows.extend([i + cu_seqlens_k[-1] for i in seq.last_token_pos])
            cu_updatedlens.append(len(seq.last_tokens) + cu_updatedlens[-1])
            cu_salientlens.append(len(seq.last_tokens) + cu_salientlens[-1])
            seq_ids.append(seq.seq_id)
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            context_lens.append(seqlen)

        def _cpu_tensor(data, t_dtype):
            t = torch.tensor(data, dtype=t_dtype, device="cpu")
            return t.pin_memory() if self.is_cuda else t

        input_ids_cpu = _cpu_tensor(input_ids, torch.int64)
        positions_cpu = _cpu_tensor(positions, torch.int64)
        positions_k_cpu = _cpu_tensor(positions_k, torch.int64)

        cu_seqlens_q_list = cu_seqlens_q
        cu_seqlens_k_list = cu_seqlens_k

        cu_seqlens_q_cpu = _cpu_tensor(cu_seqlens_q, torch.int32)
        cu_seqlens_k_cpu = _cpu_tensor(cu_seqlens_k, torch.int32)
        cu_promptlens_cpu = _cpu_tensor(cu_promptlens, torch.int32)
        cu_updatedlens_cpu = _cpu_tensor(cu_updatedlens, torch.int32)
        cu_salientlens_cpu = _cpu_tensor(cu_salientlens, torch.int32)

        idx_updated_rows_cpu = _cpu_tensor(idx_updated_rows, torch.int32)
        idx_salient_rows_cpu = _cpu_tensor(idx_salient_rows, torch.int64)

        idx_salient_rows_k_cpu = (
            _cpu_tensor(idx_salient_rows_k, torch.int64)
            if idx_salient_rows_k is not None
            else None
        )

        context_lens_list = context_lens
        context_lens_cpu = _cpu_tensor(context_lens, torch.int32)

        metadata.running_seqs = seq_ids
        seq_ids_cpu = _cpu_tensor(seq_ids, torch.long)

        input_ids = input_ids_cpu.to(device=self.device, non_blocking=True)
        positions = positions_cpu.to(device=self.device, non_blocking=True)
        positions_k = positions_k_cpu.to(device=self.device, non_blocking=True)
        cu_seqlens_q = cu_seqlens_q_cpu.to(device=self.device, non_blocking=True)
        cu_seqlens_k = cu_seqlens_k_cpu.to(device=self.device, non_blocking=True)
        cu_promptlens = cu_promptlens_cpu.to(device=self.device, non_blocking=True)
        cu_updatedlens = cu_updatedlens_cpu.to(device=self.device, non_blocking=True)
        cu_salientlens = cu_salientlens_cpu.to(device=self.device, non_blocking=True)
        idx_updated_rows = idx_updated_rows_cpu.to(device=self.device, non_blocking=True)

        # XLA advanced indexing requires int64 for cache row gathers.
        idx_salient_rows = idx_salient_rows_cpu.to(dtype=torch.int64, device=self.device, non_blocking=True)
        idx_salient_rows_k = (
            idx_salient_rows_k_cpu.to(dtype=torch.int64, device=self.device, non_blocking=True) 
            if idx_salient_rows_k_cpu is not None else None
        )
        context_lens = context_lens_cpu.to(device=self.device, non_blocking=True)
        metadata.running_seqs_tensor = seq_ids_cpu.to(device=self.device, non_blocking=True)

        set_context(
            False,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_cpu=cu_seqlens_q_list,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_k_cpu=cu_seqlens_k_list,
            cu_promptlens=cu_promptlens,
            cu_updatedlens=cu_updatedlens,
            cu_salientlens=cu_salientlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            idx_updated_row=idx_updated_rows,
            idx_salient_row=idx_salient_rows,
            idx_salient_row_k=idx_salient_rows_k,
            context_lens=context_lens,
            context_lens_cpu=context_lens_list,
            total_seqlen_k=cu_seqlens_k_list[-1],
            total_seqlen=cu_seqlens_q_list[-1],
            positions_k=positions_k,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], sample_on_cpu: bool = False):
        mask_id = self.config.mask_id
        rels, counts, temps, transfers = [], [], [], []
        top_ps, top_ks, thrs = [], [], []
        batch_offsets = []
        has_p = has_k = has_thr = False

        is_sparse_k = get_context().idx_salient_row_k is not None

        for seq in seqs:
            limit = min(len(seq.token_ids), seq.block_size * (seq.block_idx + 1) + seq.num_prompt_tokens)
            mask_pos = [i for i, t in enumerate(seq.token_ids[:limit]) if t == mask_id]

            current_offset = 0
            if is_sparse_k:
                current_offset = len(seq) - seq.max_new_tokens
                mask_pos = [p - current_offset for p in mask_pos if p >= current_offset]

            rels.extend(mask_pos)
            batch_offsets.append(current_offset)
            counts.append(len(mask_pos))
            (
                temps.extend([seq.temperature] * len(mask_pos))
                if seq.temperature is not None and seq.temperature > 1e-6
                else temps.extend([None] * len(mask_pos))
            )
            transfers.append(seq.num_transfer_tokens)
            p_val = seq.top_p if seq.top_p is not None else None
            top_ps.extend([p_val] * len(mask_pos))
            k_val = seq.top_k if seq.top_k is not None else None
            top_ks.extend([k_val] * len(mask_pos))
            t_val = seq.threshold if seq.threshold is not None else None
            thrs.extend([t_val] * len(mask_pos))
            has_p |= seq.top_p is not None
            has_k |= seq.top_k is not None
            has_thr |= seq.threshold is not None

        block_size = max([seq.block_size if seq.block_size is not None else seq.max_new_tokens for seq in seqs])

        cu_cpu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device="cpu")
        cu_cpu[1:] = torch.cumsum(torch.tensor(counts, dtype=torch.int32, device="cpu"), dim=0)

        if all(t is None for t in temps):
            temps_cpu = None
        else:
            safe_temps = [t if t is not None else 0.0 for t in temps]
            temps_cpu = torch.tensor(safe_temps, dtype=torch.float32, device="cpu")

        transfers_cpu = torch.tensor(transfers, dtype=torch.int32, device="cpu")
        p_cpu = torch.tensor(top_ps, dtype=torch.float32, device="cpu") if has_p else None
        k_cpu = torch.tensor(top_ks, dtype=torch.int32, device="cpu") if has_k else None
        thr_cpu = torch.tensor(thrs, dtype=torch.float32, device="cpu") if has_thr else None

        rels_cpu = torch.tensor(rels, dtype=torch.long, device="cpu")
        offsets_cpu = torch.tensor(batch_offsets, dtype=torch.long, device="cpu")

        # When sampling on CPU (XLA path), skip device transfer to keep
        # variable-size tensors out of the XLA graph.
        _dev = (lambda t: t) if sample_on_cpu else self._to_device

        cu_filtered = _dev(cu_cpu)
        temperatures = _dev(temps_cpu) if temps_cpu is not None else None
        num_transfer_tokens = _dev(transfers_cpu)
        top_p = _dev(p_cpu) if p_cpu is not None else None
        top_k = _dev(k_cpu) if k_cpu is not None else None
        thresholds = _dev(thr_cpu) if thr_cpu is not None else None

        rel = _dev(rels_cpu)
        batch_offsets_gpu = _dev(offsets_cpu)

        return (
            (temperatures, thresholds, top_k, top_p, block_size),
            (rel, batch_offsets_gpu, cu_filtered),
            num_transfer_tokens,
        )

    @torch.no_grad()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_full: bool):
        if is_full or self.enforce_eager or input_ids.size(0) > 512 or self.device.type == "xla":
            out = self.model.compute_logits(self.model(input_ids, positions))
            
            if self.device.type == "xla":
                import torch_xla
                torch_xla.sync()
                if self.world_size > 1:
                    dist.barrier()
            return out
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_full: bool) -> list[int]:
        input_ids, positions = self.prepare_full(seqs) if is_full else self.prepare_sparse(seqs)

        # Pre-allocate cache storage so the forward XLA graph stays
        # shape-stable across steps (no torch.cat in the trace).
        self._pre_allocate_caches(is_full)
        if self.device.type == "xla":
            import torch_xla
            torch_xla.sync()

        logits = self.run_model(input_ids, positions, is_full)
        if self.rank == 0:
            # Sampler runs on CPU under XLA: mask counts change every step
            # and would otherwise trigger per-shape recompilation.
            if self.device.type == "xla":
                import torch_xla
                torch_xla.sync()
                logits_sample = logits.cpu()
            else:
                logits_sample = logits

            sampler_params, input_indices, num_transfer_tokens = self.prepare_sample(
                seqs, sample_on_cpu=(self.device.type == "xla"),
            )

            # Lightweight CPU context for the sampler (only cu_seqlens_q used).
            ctx = get_context()
            if self.device.type == "xla":
                from dyllm.utils.context import Context
                sample_ctx = Context(
                    is_full=ctx.is_full,
                    cu_seqlens_q=torch.tensor(ctx.cu_seqlens_q_cpu, dtype=torch.int32),
                )
            else:
                sample_ctx = ctx

            pos, token_ids, counts = self.sampler(
                input_logits=logits_sample,
                ctx=sample_ctx,
                input_indices=input_indices,
                temperatures=sampler_params[0],
                num_transfer=num_transfer_tokens,
                thresholds=sampler_params[1],
                top_k=sampler_params[2],
                top_p=sampler_params[3],
                block_size=sampler_params[4],
            )
        else:
            pos, token_ids, counts = None, None, None

        # Flush pending XLA ops on all ranks so per-step traces stay
        # identical (otherwise rank-0 sampler ops leak into the next
        # step and break cross-replica collectives).
        if self.device.type == "xla":
            import torch_xla
            torch_xla.sync()

        reset_context()
        return pos, token_ids, counts
