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

#include "realm.h"
#include "realm/hip/hip_module.h"
#include "realm/hip/hip_redop.h"
#include "simple_reduce.h"

extern Realm::Logger log_app;

void register_gpu_reduction(Realm::Runtime &realm, Realm::ReductionOpID redop_id)
{
  // Register the redop the normal way (this will _not_ register the hip redops, we'll do
  // that later)
  if(!realm.register_reduction<ReductionOpMixedAdd>(redop_id)) {
    log_app.fatal("Failed to register reduction op");
  }

#if defined(REALM_USE_HIP) && !defined(REALM_USE_HIP_HIJACK)
  Realm::Hip::HipModule *hip = realm.get_module<Realm::Hip::HipModule>("hip");
  if(hip != nullptr) {
    Realm::Machine::ProcessorQuery pq =
        Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
            .only_kind(Realm::Processor::TOC_PROC);
    std::vector<Realm::Hip::HipRedOpDesc> descs;
    for(Realm::Processor p : pq) {
      Realm::Hip::HipRedOpDesc desc;
      desc.proc = p;
      desc.redop_id = redop_id;
      // For HIP without runtime hijack, hipLaunchKernel accepts host-side void*
      // function pointers directly - no per-GPU translation needed
      desc.apply_excl = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernels::apply_hip_kernel<ReductionOpMixedAdd, true>);
      desc.apply_nonexcl = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernels::apply_hip_kernel<ReductionOpMixedAdd, false>);
      desc.fold_excl = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernels::fold_hip_kernel<ReductionOpMixedAdd, true>);
      desc.fold_nonexcl = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernels::fold_hip_kernel<ReductionOpMixedAdd, false>);

      desc.apply_excl_advanced = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsAdvanced::apply_hip_kernel<ReductionOpMixedAdd,
                                                                   true>);
      desc.apply_nonexcl_advanced = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsAdvanced::apply_hip_kernel<ReductionOpMixedAdd,
                                                                   false>);
      desc.fold_excl_advanced = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsAdvanced::fold_hip_kernel<ReductionOpMixedAdd,
                                                                  true>);
      desc.fold_nonexcl_advanced = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsAdvanced::fold_hip_kernel<ReductionOpMixedAdd,
                                                                  false>);

      desc.apply_excl_transpose = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsTranspose::apply_hip_kernel<ReductionOpMixedAdd,
                                                                    true>);
      desc.apply_nonexcl_transpose = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsTranspose::apply_hip_kernel<ReductionOpMixedAdd,
                                                                    false>);
      desc.fold_excl_transpose = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsTranspose::fold_hip_kernel<ReductionOpMixedAdd,
                                                                   true>);
      desc.fold_nonexcl_transpose = reinterpret_cast<void *>(
          &Realm::Hip::ReductionKernelsTranspose::fold_hip_kernel<ReductionOpMixedAdd,
                                                                   false>);
      descs.push_back(desc);
    }
    Realm::Event e = Realm::Event::NO_EVENT;
    if(!hip->register_reduction(e, descs.data(), descs.size())) {
      assert(0 && "Failed to register reduction directly with hip module");
    }
    e.wait();
  }
#endif
}
