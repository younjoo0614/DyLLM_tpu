import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


def _pjrt_enabled() -> bool:
    return bool(os.environ.get("PJRT_DEVICE") or os.environ.get("PJRT_DEVICES"))


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs: int = 16
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    mask_id: int = -1
    num_full_steps: int = 8
    threshold: float = 0.99
    dist_port: int | None = None
    runtime_device: str = "auto"
    backend: str = "auto"
    
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.mask_id = self.hf_config.mask_token_id
        if hasattr(self.hf_config, "max_position_embeddings"):
            max_context_length = self.hf_config.max_position_embeddings
        elif hasattr(self.hf_config, "max_sequence_length"):
            max_context_length = self.hf_config.max_sequence_length
        self.max_model_len = min(self.max_model_len, max_context_length)
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.backend == "auto":
            if self.runtime_device.startswith("xla") or _pjrt_enabled():
                self.backend = "xla"
            elif torch.cuda.is_available():
                self.backend = "cuda"
            else:
                self.backend = "cpu"

        if self.runtime_device == "auto":
            if self.backend == "xla":
                self.runtime_device = "xla"
            elif self.backend == "cuda":
                self.runtime_device = "cuda"
            else:
                self.runtime_device = "cpu"
