/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#include "runtime.h"
#include <cuda.h>

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    uintptr_t MemoryManager::allocate_legion_instance_gpu(size_t footprint,
                                                          bool needs_deferral)
    //--------------------------------------------------------------------------
    {
      uintptr_t result = 0;
      switch (memory.kind())
      {
        case Z_COPY_MEM:
        case GPU_FB_MEM:
          {
            if (needs_deferral)
            {
              MallocInstanceArgs args(this, footprint, &result);
              const RtEvent wait_on = runtime->issue_runtime_meta_task(args,
                  LG_LATENCY_WORK_PRIORITY, RtEvent::NO_RT_EVENT, local_gpu);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
              return result;
            }
            else
            {
              // Use the driver API here to avoid the CUDA hijack
              if (memory.kind() == Memory::GPU_FB_MEM)
              {
                CUdeviceptr ptr;
                if (cuMemAlloc(&ptr, footprint) == CUDA_SUCCESS)
                  result = (uintptr_t)ptr;
                else
                  result = 0;
              }
              else
              {
                void *ptr = NULL;
                if (cuMemHostAlloc(&ptr, footprint, CU_MEMHOSTALLOC_PORTABLE |
                      CU_MEMHOSTALLOC_DEVICEMAP) == CUDA_SUCCESS)
                {
                  result = (uintptr_t)ptr;
                  // Check that the device pointer is the same as the host
                  CUdeviceptr gpuptr;
                  if (cuMemHostGetDevicePointer(&gpuptr,ptr,0) == CUDA_SUCCESS)
                  {
                    if (ptr != (void*)gpuptr)
                      result = 0;
                  }
                  else
                    result = 0;
                }
                else
                  result = 0;
              }
            }
            break;
          }
        default:
          assert(false);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::free_external_allocation_gpu(uintptr_t ptr, size_t size)
    //--------------------------------------------------------------------------
    {
      switch (memory.kind())
      {
        case Memory::GPU_FB_MEM:
          {
            cuMemFree((CUdeviceptr)ptr);
            break;
          }
        case Memory::Z_COPY_MEM:
          {
            cuMemFreeHost((void*)ptr);
            break;
          }
        default:
          assert(false);
      }
    }

  }; // namespace Internal
}; // namespace Legion

// EOF
