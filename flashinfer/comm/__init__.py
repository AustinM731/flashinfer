import torch as _torch

# The flashinfer.comm submodule is upstream NVIDIA-only: TRT-LLM custom
# all-reduce, MNNVL multicast, and libcudart-backed IPC handles. None of
# this exists on ROCm — there is no libcudart, no NVSwitch, and no TRT-LLM
# kernels. Importing the submodule on a HIP build crashes at module load
# (cuda_ipc.py instantiates CudaRTLibrary() which asserts on missing libcudart).
#
# vLLM's flashinfer_all_reduce.py probes this namespace via
# `hasattr(flashinfer_comm, "allreduce_fusion")` and falls back to NCCL/RCCL
# when False. So on ROCm we expose an empty namespace and let the probe fail
# cleanly. Setting VLLM_ALLREDUCE_USE_FLASHINFER=0 in the deployment env is
# the belt-and-suspenders companion to this.
_IS_ROCM = hasattr(_torch.version, "hip") and _torch.version.hip is not None

if not _IS_ROCM:
    from .cuda_ipc import CudaRTLibrary, create_shared_buffer, free_shared_buffer
    from .dlpack_utils import pack_strided_memory
    from .mapping import Mapping
    from .trtllm_ar import AllReduceFusionOp as AllReduceFusionOp
    from .trtllm_ar import AllReduceFusionPattern as AllReduceFusionPattern
    from .trtllm_ar import AllReduceStrategyConfig as AllReduceStrategyConfig
    from .trtllm_ar import AllReduceStrategyType as AllReduceStrategyType
    from .trtllm_ar import QuantizationSFLayout as QuantizationSFLayout
    from .trtllm_ar import (
        compute_fp4_swizzled_layout_sf_size as compute_fp4_swizzled_layout_sf_size,
    )
    from .trtllm_ar import gen_trtllm_comm_module as gen_trtllm_comm_module
    from .trtllm_ar import trtllm_allreduce_fusion as trtllm_allreduce_fusion
    from .trtllm_ar import (
        trtllm_create_ipc_workspace_for_all_reduce as trtllm_create_ipc_workspace_for_all_reduce,
    )
    from .trtllm_ar import (
        trtllm_create_ipc_workspace_for_all_reduce_fusion as trtllm_create_ipc_workspace_for_all_reduce_fusion,
    )
    from .trtllm_ar import trtllm_custom_all_reduce as trtllm_custom_all_reduce
    from .trtllm_ar import (
        trtllm_destroy_ipc_workspace_for_all_reduce as trtllm_destroy_ipc_workspace_for_all_reduce,
    )
    from .trtllm_ar import (
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion as trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
    )
    from .trtllm_ar import trtllm_lamport_initialize as trtllm_lamport_initialize
    from .trtllm_ar import trtllm_lamport_initialize_all as trtllm_lamport_initialize_all
    from .trtllm_ar import trtllm_moe_allreduce_fusion as trtllm_moe_allreduce_fusion
    from .trtllm_ar import (
        trtllm_moe_finalize_allreduce_fusion as trtllm_moe_finalize_allreduce_fusion,
    )
    from .vllm_ar import all_reduce as vllm_all_reduce
    from .vllm_ar import dispose as vllm_dispose
    from .vllm_ar import gen_vllm_comm_module as gen_vllm_comm_module
    from .vllm_ar import get_graph_buffer_ipc_meta as vllm_get_graph_buffer_ipc_meta
    from .vllm_ar import init_custom_ar as vllm_init_custom_ar
    from .vllm_ar import meta_size as vllm_meta_size
    from .vllm_ar import register_buffer as vllm_register_buffer
    from .vllm_ar import register_graph_buffers as vllm_register_graph_buffers

    # from .mnnvl import MnnvlMemory, MnnvlMoe, MoEAlltoallInfo
