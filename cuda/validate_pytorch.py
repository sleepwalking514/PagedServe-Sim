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

    invalid_lengths = sequence_lengths.clone()
    invalid_lengths[0] = block_table.shape[1] * block_tokens + 1
    try:
        extension.paged_attention_decode(query, key_cache, value_cache, block_table, invalid_lengths)
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-range sequence length was not rejected")

    invalid_table = block_table.clone()
    invalid_table[0, 0] = physical_blocks
    try:
        extension.paged_attention_decode(query, key_cache, value_cache, invalid_table, sequence_lengths)
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-range physical block id was not rejected")

    # Check the 8B-like GQA attention dimensions without loading model weights.
    # Twelve physical KV blocks use well below 1 MiB of FP16 KV on this GPU.
    query_heads, kv_heads, head_dim = 32, 8, 128
    block_tokens, physical_blocks = 16, 12
    sequence_lengths = torch.tensor([119, 130], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [[9, 2, 11, 1, 6, 0, 8, 4, 7], [3, 10, 5, 7, 0, 11, 2, 8, 1]],
        device=device,
        dtype=torch.int32,
    )
    query = torch.randn(2, query_heads, head_dim, device=device, dtype=torch.float16) * 0.1
    key_cache = torch.randn(physical_blocks, kv_heads, block_tokens, head_dim, device=device, dtype=torch.float16) * 0.1
    value_cache = torch.randn_like(key_cache)
    actual = extension.paged_attention_decode(query, key_cache, value_cache, block_table, sequence_lengths)
    expected = pytorch_reference(query, key_cache, value_cache, block_table, sequence_lengths)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-3, atol=3e-3)
    model_shape_error = (actual.float() - expected.float()).abs().max().item()

    # Exercise the same 1,024-token context limit as the CUDA profile on an
    # irregular table. Even 70 one-layer KV blocks occupy only about 4.4 MiB.
    physical_blocks = 70
    sequence_lengths = torch.tensor([1024], device=device, dtype=torch.int32)
    block_table = torch.randperm(physical_blocks, device=device)[:64].to(torch.int32).unsqueeze(0)
    query = torch.randn(1, query_heads, head_dim, device=device, dtype=torch.float16) * 0.1
    key_cache = torch.randn(physical_blocks, kv_heads, block_tokens, head_dim, device=device, dtype=torch.float16) * 0.1
    value_cache = torch.randn_like(key_cache)
    actual = extension.paged_attention_decode(query, key_cache, value_cache, block_table, sequence_lengths)
    expected = pytorch_reference(query, key_cache, value_cache, block_table, sequence_lengths)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-3, atol=3e-3)
    long_context_error = (actual.float() - expected.float()).abs().max().item()

    print(
        "PagedAttention validation passed; "
        f"irregular-small max absolute error: {max_error:.6f}; "
        f"8B-like GQA-shape max absolute error: {model_shape_error:.6f}; "
        f"1,024-token irregular max absolute error: {long_context_error:.6f}"
    )


if __name__ == "__main__":
    main()
