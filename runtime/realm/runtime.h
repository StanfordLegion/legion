/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "realm/processor.h"
#include "realm/redop.h"
#include "realm/custom_serdez.h"

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

      // performs any network initialization and, critically, makes sure
      //  *argc and *argv contain the application's real command line
      //  (instead of e.g. mpi spawner information)
      bool network_init(int *argc, char ***argv);

      // configures the runtime from the provided command line - after this 
      //  call it is possible to create user events/reservations/etc, 
      //  perform registrations and query the machine model, but not spawn
      //  tasks or create instances
      bool configure_from_command_line(int argc, char **argv);
      bool configure_from_command_line(std::vector<std::string> &cmdline,
				       bool remove_realm_args = false);

      // starts up the runtime, allowing task/instance creation
      void start(void);

      // single-call version of the above three calls
      bool init(int *argc, char ***argv);

      // this is now just a wrapper around Processor::register_task - consider switching to
      //  that
      bool register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr);

      bool register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop);
      template <typename REDOP>
      bool register_reduction(ReductionOpID redop_id)
      {
	const ReductionOp<REDOP> redop;
	return register_reduction(redop_id, &redop);
      }

      bool register_custom_serdez(CustomSerdezID serdez_id, const CustomSerdezUntyped *serdez);
      template <typename SERDEZ>
      bool register_custom_serdez(CustomSerdezID serdez_id)
      {
	const CustomSerdezWrapper<SERDEZ> serdez;
	return register_custom_serdez(serdez_id, &serdez);
      }

      Event collective_spawn(Processor target_proc, Processor::TaskFuncID task_id, 
			     const void *args, size_t arglen,
			     Event wait_on = Event::NO_EVENT, int priority = 0);

      Event collective_spawn_by_kind(Processor::Kind target_kind, Processor::TaskFuncID task_id, 
				     const void *args, size_t arglen,
				     bool one_per_node = false,
				     Event wait_on = Event::NO_EVENT, int priority = 0);

      // there are three potentially interesting ways to start the initial
      // tasks:
      enum RunStyle {
	ONE_TASK_ONLY,  // a single task on a single node of the machine
	ONE_TASK_PER_NODE, // one task running on one proc of each node
	ONE_TASK_PER_PROC, // a task for every processor in the machine
      };

      void run(Processor::TaskFuncID task_id = 0, RunStyle style = ONE_TASK_ONLY,
	       const void *args = 0, size_t arglen = 0, bool background = false)
	__attribute__((deprecated
#ifndef __ICC // Apparently icc doesn't support strings for deprecated warnings
              ("use collect_spawn calls instead")
#endif
              ));

      // requests a shutdown of the runtime
      void shutdown(Event wait_on = Event::NO_EVENT, int result_code = 0);

      // returns the result_code passed to shutdown()
      int wait_for_shutdown(void);
    };
	
}; // namespace Realm

//include "runtime.inl"

#endif // ifndef REALM_RUNTIME_H

