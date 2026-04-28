import torch
import torch_xla.core.xla_model as xm
from torch_xla.experimental.custom_kernel import flash_attention


# 1. Define the input tensors (Query, Key, Value)
# Shape is generally: (batch_size, num_heads, sequence_length, head_dim)
q = torch.randn(3, 32, 64, 128).to('xla').
k = torch.randn(3, 32, 128, 128).to('xla')
v = torch.randn(3, 32, 128, 128).to('xla')


# 2. Run the built-in Pallas Flash Attention kernel
output = flash_attention(q, k, v)


print("Flash Attention Output Shape:", output.shape)
