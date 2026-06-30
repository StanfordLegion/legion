/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Registers ReductionOpAdd with only the 4 basic CUfunctions
// (apply_excl, apply_nonexcl, fold_excl, fold_nonexcl).  The advanced and
// transpose slots in CudaRedOpDesc are left null.  This exercises the path
// where resolve_kernel_slot returns false for a null host_proxy and
// progress_xd skips fast_reduction_kernel_mode via the has_fast_kernels guard.
//
// ReductionOpAdd has LHS == RHS (both uint32_t), so sizeof_lhs == sizeof_rhs
// and the fast-path condition is actually reached before the guard fires.

#include "realm.h"
#include "realm/cuda/cuda_module.h"
#include "cuda_reduce_basic.h"

extern Realm::Logger log_app;

static void check_cudart(cudaError_t err, const char *tok, const char *file, int line)
{
  if(err != cudaSuccess) {
    log_app.fatal("%s(%d): Error in %s = %d", file, line, tok, (int)err);
    abort();
  }
}

#define CHECK_CURT(x) check_cudart((x), #x, __FILE__, __LINE__)

void register_basic_gpu_reduction(Realm::Runtime &realm, Realm::ReductionOpID redop_id)
{
  if(!realm.register_reduction<ReductionOpAdd>(redop_id)) {
    log_app.fatal("Failed to register reduction op");
    abort();
  }

  Realm::Cuda::CudaModule *cuda = realm.get_module<Realm::Cuda::CudaModule>("cuda");
  if(cuda == nullptr)
    return;

  Realm::Machine::ProcessorQuery pq =
      Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
          .only_kind(Realm::Processor::TOC_PROC);

  std::vector<Realm::Cuda::CudaRedOpDesc> descs;
  for(Realm::Processor proc : pq) {
    int devid = 0;
    if(!cuda->get_cuda_device_id(proc, &devid))
      continue;

    CHECK_CURT(cudaSetDevice(devid));

    Realm::Cuda::CudaRedOpDesc desc;
    desc.proc = proc;
    desc.redop_id = redop_id;

    // Register only the 4 basic kernels.  Advanced and transpose slots remain
    // null (their CudaRedOpDesc defaults), so the dispatch falls back to the
    // basic stride-based path via the has_fast_kernels guard.
    CHECK_CURT(cudaGetFuncBySymbol(
        static_cast<cudaFunction_t *>(&desc.apply_excl),
        reinterpret_cast<const void *>(
            Realm::Cuda::ReductionKernels::apply_cuda_kernel<ReductionOpAdd, true>)));
    CHECK_CURT(cudaGetFuncBySymbol(
        static_cast<cudaFunction_t *>(&desc.apply_nonexcl),
        reinterpret_cast<const void *>(
            Realm::Cuda::ReductionKernels::apply_cuda_kernel<ReductionOpAdd, false>)));
    CHECK_CURT(cudaGetFuncBySymbol(
        static_cast<cudaFunction_t *>(&desc.fold_excl),
        reinterpret_cast<const void *>(
            Realm::Cuda::ReductionKernels::fold_cuda_kernel<ReductionOpAdd, true>)));
    CHECK_CURT(cudaGetFuncBySymbol(
        static_cast<cudaFunction_t *>(&desc.fold_nonexcl),
        reinterpret_cast<const void *>(
            Realm::Cuda::ReductionKernels::fold_cuda_kernel<ReductionOpAdd, false>)));

    descs.push_back(desc);
  }

  if(descs.empty())
    return;

  Realm::Event e = Realm::Event::NO_EVENT;
  if(!cuda->register_reduction(e, descs.data(), descs.size())) {
    log_app.fatal("Failed to register basic GPU reduction kernels");
    abort();
  }
  e.wait();
}
