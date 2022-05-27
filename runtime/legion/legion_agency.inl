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

// Useful for IDEs 
#include "legion/legion_agency.h"

namespace Legion {

    /////////////////////////////////////////////////////////////
    // Legion Executor 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LegionExecutor::LegionExecutor(const char *task_name)
      : Internal::VariantExecutor(select_local_executor(task_name))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionExecutor::LegionExecutor(const Task *task)
      : Internal::VariantExecutor(select_local_executor(task->get_task_name()))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    inline Internal::VariantExecutor LegionExecutor::select_local_executor(
                                                          const char *task_name)
    //--------------------------------------------------------------------------
    {
      Processor p = Processor::get_executing_processor();
      switch (p.kind())
      {
        case Processor::LOC_PROC:
          {
            // Assume our CPUs have vector intrinsics, if
            // not the vector executor just falls back to
            // normal sequential execution
            return Internal::VariantExecutor(agency::vector_executor());
          }
        case Processor::OMP_PROC:
          {
            return Internal::VariantExecutor(
                      agency::omp::parallel_for_executor());
          }
        case Processor::TOC_PROC:
          {
#if defined (__CUDACC__) || defined (__HIPCC__)
            return Internal::VariantExecutor(agency::cuda::parallel_executor());
#else
            // This is an error because we need to be compiled
            // with nvcc if we are going to be running on GPU processors
            if (task_name != NULL)
              fprintf(stderr,"ERROR: using a LegionExecutor in an Agency "
                             "variant of task %s on a GPU processor, but "
                             "file %s was not compiled with nvcc! Please "
                             "make sure that %s is compiled with nvcc and "
                             "then try again.", task_name, __FILE__, __FILE__);
            else
              fprintf(stderr,"ERROR: using a LegionExecutor in an Agency "
                             "variant of an unknown task on a GPU processor, "
                             "but file %s was not compiled with nvcc! Please "
                             "make sure that %s is compiled with nvcc and "
                             "then try again. Note you can figure out which "
                             "task is responsible for this error by using the "
                             "LegionExecutor(const Task*) constructor.", 
                             __FILE__, __FILE__);
            assert(false);
#endif
            break;
          }
        case Processor::IO_PROC:
          {
            // I/O processors should just be sequential
            return Internal::VariantExecutor(agency::sequenced_executor());
          }
        default:
          {
            if (task_name != NULL)
              fprintf(stderr,"WARNING: LegionExecutor used in an Agency "
                             "variant of task %s is being used on an unknown "
                             "processor type %d. Falling back to the "
                             "sequential executor.", task_name, p.kind());
            else
              fprintf(stderr,"WARNING: LegionExecutor used in an Agency "
                             "variant of unknown task in file %s is being "
                             "used on an unknown processor type %d. Falling "
                             "back to the sequential executor.", 
                             __FILE__, p.kind());
            return Internal::VariantExecutor(agency::sequenced_executor());
          }
      }
      return Internal::VariantExecutor();
    }

}; // namespace Legion

