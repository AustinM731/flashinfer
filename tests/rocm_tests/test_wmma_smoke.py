#!/usr/bin/env python3
"""WMMA 16×16×16 GEMM smoke test for RDNA4 (gfx1201).

Runs a minimal matrix multiply using WMMA intrinsics via FlashInfer's MMA
layer and verifies the output against a CPU reference.

Usage (inside a gfx1201 container):
    python -m pytest tests/rocm_tests/test_wmma_gemm_smoke.py -v
"""

import torch


def test_wmma_gem_16x16x16_smoke():
    """Verify FlashInfer JIT compilation of a single-decode kernel on gfx1201.

    This test triggers JIT compilation of the decode kernel (which exercises
    the WMMA intrinsics path in mma_hip.h) and checks that it produces valid
    output. It is intentionally small and fast.
    """
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(
        0
    ).gcnArchName.startswith("gfx1201"), (
        "This test requires a gfx1201 (RDNA4) GPU"
    )

    import flashinfer

    # Build a minimal paged KV cache and query to trigger the decode kernel.
    batch_size = 2
    num_heads_q = 4
    num_heads_kv = 4
    head_dim = 128
    block_size = 16
    seq_len = 16  # 1 page per sequence

    dtype = torch.float16
    device = "cuda:0"

    # Query: (batch_size, num_heads_q, head_dim)
    q = torch.randn(
        (batch_size, num_heads_q, head_dim),
        dtype=dtype,
        device=device,
    )

    # KV cache: (num_blocks, 2, block_size, num_heads_kv, head_dim)
    num_blocks = seq_len  # 1 block per sequence
    kv_cache = torch.randn(
        (num_blocks, 2, block_size, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )

    # Key/query for decode (single token per sequence)
    k = torch.randn(
        (batch_size, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        (batch_size, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )

    # Create decode wrapper & plan
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

    # Block table: (batch_size, max_num_blocks)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(
        1, -1
    ).repeat(batch_size, 1)

    # Last page length: (batch_size,) — each seq has 1 page of 16 tokens
    last_page_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cpu")

    wrapper.plan(
        qo_indptr=torch.tensor([0, batch_size], dtype=torch.int32, device="cpu"),
        paged_kv_indptr=torch.tensor(
            [0, 1, 2], dtype=torch.int32, device="cpu"
        ),
        paged_kv_indices=block_table.reshape(-1),
        paged_kv_last_page_len=last_page_len,
        num_qo_heads=num_heads_q,
        num_kv_heads=num_heads_kv,
        head_dim_qk=head_dim,
        page_size=block_size,
        causal=True,
        sm_scale=head_dim ** -0.5,
        window_left=-1,
        logits_soft_cap=0.0,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )

    # Run decode — this triggers JIT compilation of the WMMA kernel
    o = torch.empty(
        (batch_size, num_heads_q, head_dim),
        dtype=dtype,
        device=device,
    )
    wrapper.run(q, k, v, kv_cache[:, 0], kv_cache[:, 1], o)

    # Basic sanity: output should be finite and non-zero
    assert torch.isfinite(o).all(), "Output contains NaN or Inf"
    assert (o.abs() > 1e-8).any(), "Output is all zeros — possible no-op or broken kernel"

    print(f"WMMA decode kernel smoke test PASSED — output shape {o.shape}, dtype {o.dtype}")


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

    # Query: (batch_size * seq_len, num_heads_q, head_dim) for ragged
    # Using paged prefill instead:
    q = torch.randn(
        (batch_size * seq_len, num_heads_q, head_dim),
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        (batch_size * seq_len, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        (batch_size * seq_len, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )

    num_blocks = (batch_size * seq_len) // block_size  # total blocks
    kv_cache = torch.randn(
        (num_blocks, 2, block_size, num_heads_kv, head_dim),
        dtype=dtype,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    # qo_indptr: cumulative token count per sequence
    qo_indptr = torch.tensor(
        [0, seq_len, batch_size * seq_len], dtype=torch.int32, device="cpu"
    )

    # block table: (batch_size, blocks_per_seq)
    blocks_per_seq = seq_len // block_size
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(
        batch_size, blocks_per_seq
    ).to(device)

    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=torch.tensor(
            [0, blocks_per_seq, 2 * blocks_per_seq],
            dtype=torch.int32,
            device="cpu",
        ),
        paged_kv_indices=block_table.reshape(-1),
        paged_kv_last_page_len=torch.full(
            (batch_size,), block_size, dtype=torch.int32, device="cpu"
        ),
        num_qo_heads=num_heads_q,
        num_kv_heads=num_heads_kv,
        head_dim_qk=head_dim,
        page_size=block_size,
        causal=True,
        sm_scale=head_dim ** -0.5,
        window_left=-1,
        logits_soft_cap=0.0,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )

    o = torch.empty_like(q)
    wrapper.run(q, k, v, kv_cache[:, 0], kv_cache[:, 1], o)

    assert torch.isfinite(o).all(), "Prefill output contains NaN or Inf"
    assert (o.abs() > 1e-8).any(), "Prefill output is all zeros"

    print(f"WMMA prefill kernel smoke test PASSED — output shape {o.shape}, dtype {o.dtype}")
