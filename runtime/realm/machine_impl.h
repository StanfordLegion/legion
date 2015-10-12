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

// implementation for Realm Machine class

#ifndef REALM_MACHINE_IMPL_H
#define REALM_MACHINE_IMPL_H

#include "machine.h"

#include "legion_types.h"
#include "legion_utilities.h"
#include "activemsg.h"

#include <vector>
#include <set>

namespace Realm {

    class MachineImpl {
    public:
      MachineImpl(void);
      ~MachineImpl(void);

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

      int get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				Processor restrict_proc = Processor::NO_PROC,
				Memory restrict_memory = Memory::NO_MEMORY) const;

      int get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
			       Memory restrict_mem1 = Memory::NO_MEMORY,
			       Memory restrict_mem2 = Memory::NO_MEMORY) const;
      
      void parse_node_announce_data(int node_id,
				    unsigned num_procs, unsigned num_memories,
				    const void *args, size_t arglen,
				    bool remote);

      void add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma);
      void add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma);

    protected:
      GASNetHSL mutex;
      std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
      std::vector<Machine::MemoryMemoryAffinity> mem_mem_affinities;
    };

    extern MachineImpl *machine_singleton;
    inline MachineImpl *get_machine(void) { return machine_singleton; }

  // active messages

  enum {
    NODE_ANNOUNCE_DONE = 0,
    NODE_ANNOUNCE_PROC, // PROC id kind
    NODE_ANNOUNCE_MEM,  // MEM id size
    NODE_ANNOUNCE_PMA,  // PMA proc_id mem_id bw latency
    NODE_ANNOUNCE_MMA,  // MMA mem1_id mem2_id bw latency
  };

  struct NodeAnnounceMessage {
    struct RequestArgs : public BaseMedium {
      gasnet_node_t node_id;
      unsigned num_procs;
      unsigned num_memories;
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<NODE_ANNOUNCE_MSGID,
				       RequestArgs,
				       handle_request> Message;

    static void send_request(gasnet_node_t target,
			     unsigned num_procs, unsigned num_memories,
			     const void *data, size_t datalen, int payload_mode);

    static void await_all_announcements(void);
  };

	
}; // namespace Realm

//include "machine_impl.inl"

#endif // ifndef REALM_MACHINE_IMPL_H

