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

#ifndef REALM_OPENMP_INTERNAL_H
#define REALM_OPENMP_INTERNAL_H

#include "realm/realm_config.h"
#include "realm/proc_impl.h"
#include "realm/tasks.h"

#include <vector>

namespace Realm {

  class ThreadPool;

  // this is nearly identical to a LocalCPUProcessor, but it asks for its thread(s)
  //  to run on the specified numa domain

  class LocalOpenMPProcessor : public LocalTaskProcessor {
  public:
    LocalOpenMPProcessor(Processor _me, int _numa_node,
			 int _num_threads, bool _fake_cpukind,
			 CoreReservationSet& crs, size_t _stack_size,
			 bool _force_kthreads);
    virtual ~LocalOpenMPProcessor(void);

    virtual void shutdown(void);

  protected:
    int numa_node;
    int num_threads;
    CoreReservation *core_rsrv;
  public:
    ThreadPool *pool;
  };


  // we want to subclass the scheduler to replace the execute_task method, but we also want to
  //  allow the use of user or kernel threads, so we apply a bit of template magic (which only works
  //  because the constructors for the KernelThreadTaskScheduler and UserThreadTaskScheduler classes
  //  have the same prototypes)

  template <typename T>
  class OpenMPTaskScheduler : public T {
  public:
    OpenMPTaskScheduler(Processor _proc, Realm::CoreReservation& _core_rsrv,
			LocalOpenMPProcessor *_omp_proc);

    virtual ~OpenMPTaskScheduler(void);

  protected:
    virtual bool execute_task(Task *task);
    virtual void execute_internal_task(InternalTask *task);
    
    // might also need to override the thread-switching methods to keep TLS up to date

    LocalOpenMPProcessor *omp_proc;
  };

};

#endif
