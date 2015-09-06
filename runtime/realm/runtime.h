/* Copyright 2015 Stanford University, NVIDIA Corporation
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

// Realm runtime object

#ifndef REALM_RUNTIME_H
#define REALM_RUNTIME_H

#include "processor.h"
#include "redop.h"

#include "lowlevel_config.h"

namespace Realm {

    class Runtime {
    protected:
      void *impl;  // hidden internal implementation - this is NOT a transferrable handle

    public:
      Runtime(void);
      Runtime(const Runtime& r) : impl(r.impl) {}
      Runtime& operator=(const Runtime& r) { impl = r.impl; return *this; }

      ~Runtime(void) {}

      static Runtime get_runtime(void);

      bool init(int *argc, char ***argv);

      bool register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr);
      bool register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop);

      // there are three potentially interesting ways to start the initial
      // tasks:
      enum RunStyle {
	ONE_TASK_ONLY,  // a single task on a single node of the machine
	ONE_TASK_PER_NODE, // one task running on one proc of each node
	ONE_TASK_PER_PROC, // a task for every processor in the machine
      };

      void run(Processor::TaskFuncID task_id = 0, RunStyle style = ONE_TASK_ONLY,
	       const void *args = 0, size_t arglen = 0, bool background = false);

      // requests a shutdown of the runtime
      void shutdown(void);

      void wait_for_shutdown(void);
    };
	
}; // namespace Realm

//include "runtime.inl"

#endif // ifndef REALM_RUNTIME_H

