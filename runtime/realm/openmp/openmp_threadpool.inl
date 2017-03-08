/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// OpenMP (or similar) thread pool for Realm

// NOP, but helps IDEs
#include "openmp_threadpool.h"

namespace Realm {

  namespace ThreadLocal {
    extern __thread ThreadPool::WorkerInfo *threadpool_workerinfo;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadPool

  /*static*/ inline ThreadPool::WorkerInfo *ThreadPool::get_worker_info(void)
  {
    return ThreadLocal::threadpool_workerinfo;
  }

}; // namespace Realm
