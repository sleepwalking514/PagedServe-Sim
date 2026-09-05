#!/usr/bin/env python3
"""Compile the CUDA extension and compare it with a PyTorch reference."""

import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def pytorch_reference(query, key_cache, value_cache, block_table, sequence_lengths):
    batch_size, query_heads, head_dim = query.shape
    kv_heads = key_cache.shape[1]
    block_tokens = key_cache.shape[2]
    group_size = query_heads // kv_heads
    output = torch.empty_like(query)

    for batch in range(batch_size):
        sequence_length = int(sequence_lengths[batch])
        block_count = math.ceil(sequence_length / block_tokens)
        physical_blocks = block_table[batch, :block_count]
        keys = key_cache[physical_blocks].permute(0, 2, 1, 3).reshape(-1, kv_heads, head_dim)[:sequence_length]
        values = value_cache[physical_blocks].permute(0, 2, 1, 3).reshape(-1, kv_heads, head_dim)[:sequence_length]
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            scores = query[batch, query_head].float() @ keys[:, kv_head].float().T
            weights = torch.softmax(scores / math.sqrt(head_dim), dim=0)
            output[batch, query_head] = weights @ values[:, kv_head].float()
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch build is required")

    root = Path(__file__).resolve().parent.parent
    extension = load(
        name="paged_attention_torch_extension",
        sources=[str(root / "cuda" / "paged_attention_torch.cu")],
        extra_include_paths=[str(root / "include")],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

    torch.manual_seed(123)
    device = "cuda"
    batch_size, query_heads, kv_heads, head_dim = 2, 4, 2, 32
    block_tokens, physical_blocks = 4, 6
    sequence_lengths = torch.tensor([7, 10], device=device, dtype=torch.int32)
    block_table = torch.tensor([[3, 1, 5], [0, 4, 2]], device=device, dtype=torch.int32)
    query = torch.randn(batch_size, query_heads, head_dim, device=device, dtype=torch.float16) * 0.1
    key_cache = torch.randn(physical_blocks, kv_heads, block_tokens, head_dim, device=device, dtype=torch.float16) * 0.1
    value_cache = torch.randn_like(key_cache)

    actual = extension.paged_attention_decode(query, key_cache, value_cache, block_table, sequence_lengths)
    expected = pytorch_reference(query, key_cache, value_cache, block_table, sequence_lengths)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-3, atol=3e-3)
    max_error = (actual.float() - expected.float()).abs().max().item()
    print(f"PagedAttention validation passed; max absolute error: {max_error:.6f}")


if __name__ == "__main__":
    main()
