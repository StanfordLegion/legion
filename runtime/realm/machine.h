/* Copyright 2016 Stanford University, NVIDIA Corporation
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

// machine model for Realm

#ifndef REALM_MACHINE_H
#define REALM_MACHINE_H

#include "lowlevel_config.h"

#include "processor.h"
#include "memory.h"

namespace Realm {

    class Runtime;

    class Machine {
    protected:
      void *impl;  // hidden internal implementation - this is NOT a transferrable handle

      friend class Runtime;
      explicit Machine(void *_impl) : impl(_impl) {}

    public:
      Machine(const Machine& m) : impl(m.impl) {}
      Machine& operator=(const Machine& m) { impl = m.impl; return *this; }
      ~Machine(void) {}

      static Machine get_machine(void);

      void get_all_memories(std::set<Memory>& mset) const;
      void get_all_processors(std::set<Processor>& pset) const;

      void get_local_processors(std::set<Processor>& pset) const;
      void get_local_processors_by_kind(std::set<Processor>& pset,
					Processor::Kind kind) const;

      // Return the set of memories visible from a processor
      void get_visible_memories(Processor p, std::set<Memory>& mset) const;

      // Return the set of memories visible from a memory
      void get_visible_memories(Memory m, std::set<Memory>& mset) const;

      // Return the set of processors which can all see a given memory
      void get_shared_processors(Memory m, std::set<Processor>& pset) const;

      size_t get_address_space_count(void) const;

    public:
      struct ProcessorMemoryAffinity {
	Processor p;
	Memory m;
	unsigned bandwidth; // TODO: consider splitting read vs. write?
	unsigned latency;
      };

      struct MemoryMemoryAffinity {
	Memory m1, m2;
	unsigned bandwidth;
	unsigned latency;
      };

      int get_proc_mem_affinity(std::vector<ProcessorMemoryAffinity>& result,
				Processor restrict_proc = Processor::NO_PROC,
				Memory restrict_memory = Memory::NO_MEMORY) const;

      int get_mem_mem_affinity(std::vector<MemoryMemoryAffinity>& result,
			       Memory restrict_mem1 = Memory::NO_MEMORY,
			       Memory restrict_mem2 = Memory::NO_MEMORY) const;

      // subscription interface for dynamic machine updates
      class MachineUpdateSubscriber {
      public:
       virtual ~MachineUpdateSubscriber(void) {}

       enum UpdateType { THING_ADDED,
                         THING_REMOVED,
                         THING_UPDATED
       };

       // callbacks occur on a thread that belongs to the runtime - please defer any
       //  complicated processing if possible
       virtual void processor_updated(Processor p, UpdateType update_type, 
                                      const void *payload, size_t payload_size) = 0;

       virtual void memory_updated(Memory m, UpdateType update_type,
                                   const void *payload, size_t payload_size) = 0;
      };

      // currently, all subscriptions are global - we expect updates to be rare enough that
      //  subscribers can do filtering themselves
      void add_subscription(MachineUpdateSubscriber *subscriber);
      void remove_subscription(MachineUpdateSubscriber *subscriber);
    };
	
}; // namespace Realm

//include "machine.inl"

#endif // ifndef REALM_MACHINE_H

