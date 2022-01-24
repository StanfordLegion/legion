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

// OpenMP (or similar) thread pool for Realm

// NOP, but helps IDEs
#include "realm/openmp/openmp_threadpool.h"

namespace Realm {

  extern Logger log_omp;

  namespace ThreadLocal {
    extern REALM_THREAD_LOCAL ThreadPool::WorkerInfo *threadpool_workerinfo;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadPool

  /*static*/ inline ThreadPool::WorkerInfo *ThreadPool::get_worker_info(bool warn_if_missing)
  {
    ThreadPool::WorkerInfo *info = ThreadLocal::threadpool_workerinfo;
    if(REALM_UNLIKELY(!info && warn_if_missing))
      log_omp.warning() << "OpenMP runtime call from non-OpenMP-enabled Realm processor!";
    return info;
  }

}; // namespace Realm
