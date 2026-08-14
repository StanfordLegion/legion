/*
 * Copyright 2025 Stanford University, NVIDIA Corporation, Los Alamos National Laboratory
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

// Registers ReductionOpAdd with only the 4 basic HIP kernels
// (apply_excl, apply_nonexcl, fold_excl, fold_nonexcl). The advanced and
// transpose slots in HipRedOpDesc are left null. This exercises the path
// where has_fast_kernels is false and progress_xd skips
// fast_reduction_kernel_mode, falling back to the basic stride-based loop.
//
// ReductionOpAdd has LHS == RHS (both uint32_t), so sizeof_lhs == sizeof_rhs
// and the fast-path condition is actually reached before the has_fast_kernels
// guard fires.

#include "realm.h"
#include "realm/hip/hip_module.h"
#include "realm/hip/hip_redop.h"
#include "hip_reduce_basic.h"

extern Realm::Logger log_app;

void register_basic_gpu_reduction(Realm::Runtime &realm, Realm::ReductionOpID redop_id)
{
  if(!realm.register_reduction<ReductionOpAdd>(redop_id)) {
    log_app.fatal("Failed to register reduction op");
    abort();
  }

#if defined(REALM_USE_HIP) && !defined(REALM_USE_HIP_HIJACK)
  Realm::Hip::HipModule *hip = realm.get_module<Realm::Hip::HipModule>("hip");
  if(hip == nullptr)
    return;

  Realm::Machine::ProcessorQuery pq =
      Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
          .only_kind(Realm::Processor::TOC_PROC);

  std::vector<Realm::Hip::HipRedOpDesc> descs;
  for(Realm::Processor p : pq) {
    Realm::Hip::HipRedOpDesc desc;
    desc.proc = p;
    desc.redop_id = redop_id;
    // Register only the 4 basic kernels. Advanced and transpose slots stay
    // null (HipRedOpDesc defaults), so dispatch falls back to the basic
    // stride-based path via the has_fast_kernels guard.
    desc.apply_excl = reinterpret_cast<void *>(
        &Realm::Hip::ReductionKernels::apply_hip_kernel<ReductionOpAdd, true>);
    desc.apply_nonexcl = reinterpret_cast<void *>(
        &Realm::Hip::ReductionKernels::apply_hip_kernel<ReductionOpAdd, false>);
    desc.fold_excl = reinterpret_cast<void *>(
        &Realm::Hip::ReductionKernels::fold_hip_kernel<ReductionOpAdd, true>);
    desc.fold_nonexcl = reinterpret_cast<void *>(
        &Realm::Hip::ReductionKernels::fold_hip_kernel<ReductionOpAdd, false>);
    descs.push_back(desc);
  }

  if(descs.empty())
    return;

  Realm::Event e = Realm::Event::NO_EVENT;
  if(!hip->register_reduction(e, descs.data(), descs.size())) {
    log_app.fatal("Failed to register basic HIP reduction kernels");
    abort();
  }
  e.wait();
#endif
}
