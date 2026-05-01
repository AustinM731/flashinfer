// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "gpu_runtime_compat.hpp"
#include "macros.hpp"

namespace flashinfer {
namespace gpu_iface {

// Platform-agnostic stream type
#if defined(PLATFORM_CUDA_DEVICE)
constexpr int kWarpSize = 32;

#elif defined(PLATFORM_HIP_DEVICE)
#if defined(__gfx1201__)
// RDNA4 (gfx1201) uses 32-thread wavefronts with WMMA intrinsics
// gfx1200/gfx1202 not yet supported — add when WMMA port is verified.
constexpr int kWarpSize = 32;
#else
// CDNA (gfx90a, gfx908, gfx942, gfx950, etc.) uses 64-thread wavefronts with MFMA
constexpr int kWarpSize = 64;
#endif

#endif

}  // namespace gpu_iface
}  // namespace flashinfer
