#!/usr/bin/env python3
"""WMMA 16×16×16 GEMM smoke test for RDNA4 (gfx1201).

Runs a minimal matrix multiply using WMMA intrinsics via FlashInfer's MMA
layer and verifies the output against a CPU reference.

Usage (inside a gfx1201 container):
    python -m pytest tests/rocm_tests/test_wmma_gemm_smoke.py -v
"""

import torch


def test_wmma_gemm_16x16x16_smoke():
    """Verify FlashInfer JIT compilation of a single-decode kernel on gfx1201."""
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(
        0
    ).gcnArchName.startswith("gfx1201"), (
        "This test requires a gfx1201 (RDNA4) GPU"
    )

    import flashinfer

    batch_size = 2
    num_heads_q = 4
    num_heads_kv = 4
    head_dim = 128
    block_size = 16
    seq_len = 16  # 1 page per sequence

    dtype = torch.float16
    device = "cuda:0"

    q = torch.randn(
        (batch_size, num_heads_q, head_dim),
        dtype=dtype,
        device=device,
    )

    # Each request uses exactly one page. page i -> request i.
    # kv_cache shape: (num_blocks, 2, block_size, num_heads_kv, head_dim) NHD
    total_blocks = batch_size
    kv_cache = torch.randn(
        (total_blocks, 2, block_size, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

    # paged_kv_indptr: cumulative page count [0, 1, 2] (one page per request)
    paged_kv_indptr = torch.tensor(
        [0, 1, 2], dtype=torch.int32, device="cpu"
    )
    # paged_kv_indices: one block per request [0, 1]
    paged_kv_indices = torch.arange(total_blocks, dtype=torch.int32)
    # last_page_len: 16 tokens per page
    last_page_len = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device="cpu"
    )

    # Decode plan is positional: (indptr, indices, last_page_len, ...)
    wrapper.plan(
        paged_kv_indptr,
        paged_kv_indices,
        last_page_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        block_size,
        sm_scale=head_dim ** -0.5,
        logits_soft_cap=0.0,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Run decode — triggers JIT compilation of the WMMA kernel
    paged_kv_cache = (kv_cache[:, 0], kv_cache[:, 1])
    o = wrapper.run(q, paged_kv_cache)

    assert torch.isfinite(o).all(), "Decode output contains NaN or Inf"
    assert (o.abs() > 1e-8).any(), "Decode output is all zeros"

    # CPU/GPU SDPA reference — per-request, 3-D format (seq, heads, dim).
    # Decode: one query token attends to all seq_len KV entries unmasked.
    # We use is_causal=False to explicitly match this. is_causal=True would be
    # fragile across PyTorch versions (top-left vs bottom-right alignment).
    attn_scale = head_dim ** -0.5
    o_ref_list = []
    for i in range(batch_size):
        k_i = kv_cache[i, 0]  # (block_size, num_heads_kv, head_dim)
        v_i = kv_cache[i, 1]
        q_i = q[i].unsqueeze(0)  # (1, num_heads_q, head_dim)
        o_i_ref = torch.nn.functional.scaled_dot_product_attention(
            q_i, k_i, v_i, is_causal=False, scale=attn_scale,
        )  # (1, heads, dim)
        o_ref_list.append(o_i_ref)
    o_ref = torch.cat(o_ref_list, dim=0)  # (batch, heads, dim)

    assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-1), (
        f"Decode output mismatch with SDPA reference: max diff={
            (o - o_ref).abs().max().item():.4f}"
    )

    print(f"WMMA decode kernel smoke test PASSED — shape {o.shape}, dtype {o.dtype}")


def test_wmma_prefill_smoke():
    """Verify FlashInfer prefill kernel compiles and runs on gfx1201."""
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(
        0
    ).gcnArchName.startswith("gfx1201"), (
        "This test requires a gfx1201 (RDNA4) GPU"
    )

    import flashinfer

    batch_size = 2
    num_heads_q = 4
    num_heads_kv = 4
    head_dim = 128
    block_size = 16
    seq_len = 32  # 2 pages per sequence

    dtype = torch.float16
    device = "cuda:0"

    total_tokens = batch_size * seq_len
    q = torch.randn(
        (total_tokens, num_heads_q, head_dim),
        dtype=dtype,
        device=device,
    )

    blocks_per_seq = seq_len // block_size
    total_blocks = batch_size * blocks_per_seq
    kv_cache = torch.randn(
        (total_blocks, 2, block_size, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    # qo_indptr: cumulative token count per sequence [0, seq_len, 2*seq_len]
    qo_indptr = torch.tensor(
        [0, seq_len, total_tokens], dtype=torch.int32, device="cpu"
    )

    # paged_kv_indptr: cumulative block count [0, blocks_per_seq, 2*blocks_per_seq]
    paged_kv_indptr = torch.tensor(
        [0, blocks_per_seq, 2 * blocks_per_seq],
        dtype=torch.int32,
        device="cpu",
    )

    # paged_kv_indices: contiguous block IDs
    paged_kv_indices = torch.arange(total_blocks, dtype=torch.int32)
    paged_kv_last_page_len = torch.full(
        (batch_size,), block_size, dtype=torch.int32, device="cpu"
    )

    # Prefill plan is positional: (qo_indptr, paged_kv_indptr, paged_kv_indices, ...)
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        block_size,
        causal=True,
        sm_scale=head_dim ** -0.5,
        logits_soft_cap=0.0,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    paged_kv_cache = (kv_cache[:, 0], kv_cache[:, 1])
    o = wrapper.run(q, paged_kv_cache)

    assert torch.isfinite(o).all(), "Prefill output contains NaN or Inf"
    assert (o.abs() > 1e-8).any(), "Prefill output is all zeros"

    print(f"WMMA prefill kernel smoke test PASSED — shape {o.shape}, dtype {o.dtype}")

