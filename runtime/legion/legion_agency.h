/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_AGENCY_H__
#define __LEGION_AGENCY_H__

/**
 * \file legion_agency.h
 * Support for using Agency inside of Legion tasks
 */

#include "legion.h"
#include <agency/agency.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/sequenced_executor.hpp>
#include <agency/execution/executor/vector_executor.hpp>
#include <agency/execution/executor/variant_executor.hpp>
// Agency doesn't know that we hijacked OpenMP so fake it
#ifndef _OPENMP
#define _OPENMP 201307
#endif
#include <agency/omp/execution.hpp>
#if defined (__CUDACC__) || defined (__HIPCC__)
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#endif

namespace Legion {

  namespace Internal {
    using VariantExecutor = agency::variant_executor<
                                  agency::sequenced_executor,
                                  agency::vector_executor,
                                  agency::omp::parallel_for_executor
#if defined (__CUDACC__) || defined (__HIPCC__)
                                  , agency::cuda_parallel_executor
#endif
                                  >;
  };

  class LegionExecutor : public Internal::VariantExecutor {
  public:
    // Optional task name for error messages
    LegionExecutor(const char *task_name = NULL);
    // Optionally pass in the Task* for the executor to get error messages
    LegionExecutor(const Task *task);
  public:
    static inline Internal::VariantExecutor select_local_executor(
                                              const char *task_name);
  };

}; // namespace Legion

#include "legion/legion_agency.inl"

#endif // __LEGION_AGENCY_H__

