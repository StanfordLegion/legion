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

// processors for Realm

#ifndef REALM_PROCESSOR_H
#define REALM_PROCESSOR_H

#include "realm/realm_c.h"

#include "realm/event.h"

#include <vector>
#include <map>

namespace Realm {

    typedef ::realm_address_space_t AddressSpace;

    class ProfilingRequestSet;
    class CodeDescriptor;

    class Processor {
    public:
      typedef ::realm_id_t id_t;

      id_t id;
      bool operator<(const Processor& rhs) const { return id < rhs.id; }
      bool operator==(const Processor& rhs) const { return id == rhs.id; }
      bool operator!=(const Processor& rhs) const { return id != rhs.id; }

      static const Processor NO_PROC;

      bool exists(void) const { return id != 0; }

      typedef ::realm_task_func_id_t TaskFuncID;
      typedef void (*TaskFuncPtr)(const void *args, size_t arglen,
				  const void *user_data, size_t user_data_len,
				  Processor proc);

      // Different Processor types (defined in realm_c.h)
      // can't just typedef the kind because of C/C++ enum scope rules
      enum Kind {
#define C_ENUMS(name, desc) name,
  REALM_PROCESSOR_KINDS(C_ENUMS)
#undef C_ENUMS
      };

      // Return what kind of processor this is
      Kind kind(void) const;
      // Return the address space for this processor
      AddressSpace address_space(void) const;

      static Processor create_group(const std::vector<Processor>& members);
      void get_group_members(std::vector<Processor>& members);

      int get_num_cores(void) const;

      // special task IDs
      enum {
        // Save ID 0 for the force shutdown function
	TASK_ID_PROCESSOR_NOP      = 0,
	TASK_ID_PROCESSOR_INIT     = 1,
	TASK_ID_PROCESSOR_SHUTDOWN = 2,
	TASK_ID_FIRST_AVAILABLE    = 4,
      };

      Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
		  Event wait_on = Event::NO_EVENT, int priority = 0) const;

      // Same as the above but with requests for profiling
      Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
                  const ProfilingRequestSet &requests,
                  Event wait_on = Event::NO_EVENT, int priority = 0) const;

      static Processor get_executing_processor(void);

      // changes the priority of the currently running task
      static void set_current_task_priority(int new_priority);

      // returns the finish event for the currently running task
      static Event get_current_finish_event(void);

      // dynamic task registration - this may be done for:
      //  1) a specific processor/group (anywhere in the system)
      //  2) for all processors of a given type, either in the local address space/process,
      //       or globally
      //
      // in both cases, an Event is returned, and any tasks launched that expect to use the
      //  newly-registered task IDs must include that event as a precondition

      Event register_task(TaskFuncID func_id,
			  const CodeDescriptor& codedesc,
			  const ProfilingRequestSet& prs,
			  const void *user_data = 0, size_t user_data_len = 0) const;

      static Event register_task_by_kind(Kind target_kind, bool global,
					 TaskFuncID func_id,
					 const CodeDescriptor& codedesc,
					 const ProfilingRequestSet& prs,
					 const void *user_data = 0, size_t user_data_len = 0);

      // reports an execution fault in the currently running task
      static void report_execution_fault(int reason,
					 const void *reason_data, size_t reason_size);

      // reports a problem with a processor in general (this is primarily for fault injection)
      void report_processor_fault(int reason,
				  const void *reason_data, size_t reason_size) const;

      static const char* get_kind_name(Kind kind);
    };

    inline std::ostream& operator<<(std::ostream& os, Processor p) { return os << std::hex << p.id << std::dec; }
	
}; // namespace Realm

#include "realm/processor.inl"

#endif // ifndef REALM_PROCESSOR_H

