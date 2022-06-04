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

#ifndef REALM_OPENMP_SYSTEM_RUNTIME
  class ThreadPool;
#endif

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
    class OpenMPContextManager : public TaskContextManager {
    public:
      OpenMPContextManager(LocalOpenMPProcessor *_proc);

      virtual void *create_context(Task *task) const;
      virtual void destroy_context(Task *task, void *context) const;

    protected:
      LocalOpenMPProcessor *proc;
    };

    int numa_node;
    int num_threads;
    CoreReservation *core_rsrv;
    OpenMPContextManager ctxmgr;
#ifndef REALM_OPENMP_SYSTEM_RUNTIME
    ThreadPool *pool;
#endif
  };

};

#endif
