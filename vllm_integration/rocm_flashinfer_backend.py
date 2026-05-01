# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ROCm FlashInfer attention backend adapter for vLLM on RDNA4.

Thin wrapper around the AMD-ported FlashInfer package that implements the
vLLM AttentionBackend protocol. Only supports the core decode + prefill path
(FP16/BF16, no TRTLLM, no cascade, no FP4).
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

import triton
import triton.language as tl

from vllm.config.cache import CacheDType
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# FlashInfer wrappers (lazy import at first use)
# ---------------------------------------------------------------------------

def _get_flashinfer_decode_wrapper():
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    return BatchDecodeWithPagedKVCacheWrapper


def _get_flashinfer_prefill_wrapper():
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    return BatchPrefillWithPagedKVCacheWrapper


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class RocmFlashInferMetadata:
    num_actual_tokens: int
    slot_mapping: torch.Tensor
    q_data_type: torch.dtype
    kv_cache_dtype: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # Prefill
    prefill_wrapper: object | None = None

    # Decode
    decode_wrapper: object | None = None


# ---------------------------------------------------------------------------
# Metadata Builder
# ---------------------------------------------------------------------------

class RocmFlashInferMetadataBuilder(
    AttentionMetadataBuilder[RocmFlashInferMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config,
        )
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_dim = kv_cache_spec.head_size
        self.page_size = kv_cache_spec.block_size

        self.kv_cache_dtype = kv_cache_spec.dtype
        self.cache_dtype_str = (
            self.cache_config.cache_dtype
            if self.cache_config else "auto"
        )
        if self.cache_dtype_str == "auto":
            self.kv_cache_dtype = self.model_config.dtype

        # Global hyperparameters (all layers must share these for FlashInfer)
        self.sm_scale = self.head_dim ** -0.5
        self.window_left = -1
        self.logits_soft_cap = None

        # Workspace buffer
        self._workspace_buffer: torch.Tensor | None = None

        # Prefill wrapper (shared across shapes)
        self._prefill_wrapper: object | None = None

        # Decode wrapper for non-cudagraph
        self._decode_wrapper: object | None = None

        # Decode wrappers for cudagraph (one per batch size)
        self._decode_wrappers_cudagraph: dict[int, object] = {}
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode()
            == CUDAGraphMode.FULL
        )

        # Persistent buffers for paged metadata
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len,
            kv_cache_spec.block_size,
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        self.pin_memory = False
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
            self.paged_kv_indptr.cpu,
            pin_memory=self.pin_memory,
        )
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)
        self.qo_indptr_gpu: torch.Tensor | None = None

    def _make_buffer(
        self,
        *size: int,
        dtype: torch.dtype = torch.int32,
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=True,
        )

    def _get_workspace_buffer(self) -> torch.Tensor:
        if self._workspace_buffer is None:
            size = 2048 * 1024 * 1024  # 2 GB
            self._workspace_buffer = torch.zeros(
                size,
                dtype=torch.uint8,
                device=self.device,
            )
        return self._workspace_buffer

    def _get_prefill_wrapper(self):
        if self._prefill_wrapper is None:
            WrapperCls = _get_flashinfer_prefill_wrapper()
            self._prefill_wrapper = WrapperCls(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
            )
        return self._prefill_wrapper

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        if use_cudagraph:
            wrapper = self._decode_wrappers_cudagraph.get(batch_size)
        else:
            wrapper = self._decode_wrapper

        if wrapper is None:
            WrapperCls = _get_flashinfer_decode_wrapper()
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None

            wrapper = WrapperCls(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                use_tensor_cores=True,
            )

            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = wrapper
            else:
                self._decode_wrapper = wrapper

        return wrapper

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np,
        seq_lens_np,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        import numpy as np

        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1:num_reqs + 1],
        )
        self.paged_kv_indptr_cpu_buffer[:num_reqs + 1] = self.paged_kv_indptr.cpu[
            :num_reqs + 1
        ]
        paged_kv_indptr = self.paged_kv_indptr.gpu[:num_reqs + 1]
        paged_kv_indptr.copy_(
            self.paged_kv_indptr_cpu_buffer[:num_reqs + 1],
            non_blocking=True,
        )

        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            self.paged_kv_last_page_len.cpu[:num_reqs],
            non_blocking=True,
        )
        return paged_kv_indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmFlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu
        page_size = self.page_size

        attn_metadata = RocmFlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type=self.model_config.dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill_wrapper=None,
            decode_wrapper=None,
        )

        if common_prefix_len > 0:
            raise NotImplementedError(
                "Cascade attention is not supported in ROCm FlashInfer backend."
            )

        # FlashInfer native path (no TRTLLM on ROCm)
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens_np = seq_lens_cpu.numpy()
        num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
        paged_kv_indices = self._compute_flashinfer_kv_metadata(
            num_blocks_np,
            seq_lens_np,
            block_table_tensor,
            num_reqs,
            page_size,
        )

        # ---- PREFILL ----
        if num_prefills > 0:
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                prefill_start:num_reqs + 1
            ]
            paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                prefill_start:num_reqs
            ]

            prefill_wrapper = self._get_prefill_wrapper()
            prefill_wrapper.plan(
                qo_indptr_prefill_cpu,
                paged_kv_indptr_prefill_cpu,
                paged_kv_indices,
                paged_kv_last_page_len_prefill_cpu,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                page_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap or 0.0,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
            )
            attn_metadata.prefill_wrapper = prefill_wrapper

        # ---- DECODE ----
        if num_decodes > 0:
            pure_decode = num_prefills == 0
            use_cudagraph = self.enable_cuda_graph and pure_decode

            decode_wrapper = self._get_decode_wrapper(
                num_decode_tokens,
                use_cudagraph,
            )
            fast_plan_decode(
                decode_wrapper,
                indptr_cpu=self.paged_kv_indptr.cpu[:num_decode_tokens + 1],
                indices=paged_kv_indices,
                last_page_len_cpu=self.paged_kv_last_page_len.cpu[:num_decode_tokens],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=page_size,
                pos_encoding_mode="NONE",
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap or 0.0,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
            )
            attn_metadata.decode_wrapper = decode_wrapper

        return attn_metadata


@triton.jit
def _copy_page_indices_kernel(
    paged_kv_indices,
    block_table,
    block_table_stride,
    paged_kv_indptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(paged_kv_indptr + req_idx)
    end_idx = tl.load(paged_kv_indptr + req_idx + 1)
    num_pages = end_idx - start_idx

    offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(tl.cdiv(num_pages, BLOCK_SIZE)):
        block_offsets = i * BLOCK_SIZE + offsets
        mask_block = block_offsets < num_pages
        block_idx = tl.load(
            block_table + req_idx * block_table_stride + block_offsets,
            mask=mask_block,
        )
        tl.store(
            paged_kv_indices + start_idx + block_offsets,
            block_idx,
            mask=mask_block,
        )

def fast_plan_decode(
    decode_wrapper: Any,
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str,
    sm_scale: float,
    window_left: int,
    logits_soft_cap: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    decode_wrapper.plan(
        indptr_cpu,
        indices,
        last_page_len_cpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        sm_scale=sm_scale,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
    )


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class RocmFlashInferBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_ROCM"

    @staticmethod
    def get_impl_cls() -> type["RocmFlashInferImpl"]:
        return RocmFlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["RocmFlashInferMetadataBuilder"]:
        return RocmFlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (1, 2, 4, 0, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # RDNA4 (gfx12xx) has capability (12, 0)
        return capability >= DeviceCapability(12, 0)

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: DeviceCapability,
        attn_type: str,
        use_non_causal: bool = False,
        use_batch_invariant: bool = False,
    ) -> list[str]:
        reasons = super().validate_configuration(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            use_per_head_quant_scales,
            device_capability,
            attn_type,
            use_non_causal,
            use_batch_invariant,
        )
        # Extra ROCm-specific checks
        if kv_cache_dtype not in ("auto", "float16", "bfloat16", None):
            reasons.append(
                f"kv_cache_dtype '{kv_cache_dtype}' not supported "
                "(ROCm FlashInfer only supports auto/fp16/bf16)"
            )
        return reasons


# ---------------------------------------------------------------------------
# Impl class
# ---------------------------------------------------------------------------

class RocmFlashInferImpl(AttentionImpl):
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = (
            torch.tensor(alibi_slopes, dtype=torch.float32)
            if alibi_slopes is not None
            else None
        )
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder attention not supported in ROCm FlashInfer backend."
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmFlashInferMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        # FlashInfer run() takes (q, paged_kv_cache, *, out=o)
        # where paged_kv_cache = (k_cache, v_cache), each with shape
        # (num_blocks, block_size, num_kv_heads, head_size) for NHD.
        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]
        paged_kv_cache = (k_cache, v_cache)

        # ---- PREFILL ----
        if attn_metadata.num_prefills > 0:
            prefill_wrapper: Any = attn_metadata.prefill_wrapper
            prefill_start = attn_metadata.num_decode_tokens
            num_actual_tokens = attn_metadata.num_actual_tokens

            q_prefill = query[prefill_start:num_actual_tokens]
            o_prefill = output[prefill_start:num_actual_tokens]

            prefill_wrapper.run(q_prefill, paged_kv_cache, out=o_prefill)

        # ---- DECODE ----
        if attn_metadata.num_decodes > 0:
            decode_wrapper: Any = attn_metadata.decode_wrapper
            num_decode = attn_metadata.num_decode_tokens

            q_decode = query[:num_decode]
            o_decode = output[:num_decode]

            decode_wrapper.run(q_decode, paged_kv_cache, out=o_decode)

        return output
