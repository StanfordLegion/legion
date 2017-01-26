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

      bool has_affinity(Processor p, Memory m, Machine::AffinityDetails *details = 0) const;
      bool has_affinity(Memory m1, Memory m2, Machine::AffinityDetails *details = 0) const;

      int get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				Processor restrict_proc = Processor::NO_PROC,
				Memory restrict_memory = Memory::NO_MEMORY) const;

      int get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
			       Memory restrict_mem1 = Memory::NO_MEMORY,
			       Memory restrict_mem2 = Memory::NO_MEMORY) const;
      
      void parse_node_announce_data(int node_id, unsigned num_procs,
				    unsigned num_memories, unsigned num_ib_memories,
				    const void *args, size_t arglen,
				    bool remote);

      void add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma);
      void add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma);

      void add_subscription(Machine::MachineUpdateSubscriber *subscriber);
      void remove_subscription(Machine::MachineUpdateSubscriber *subscriber);

      mutable GASNetHSL mutex;
      std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
      std::vector<Machine::MemoryMemoryAffinity> mem_mem_affinities;
      std::set<Machine::MachineUpdateSubscriber *> subscribers;
    };

    template <typename T>
    class QueryPredicate {
    public:
      virtual ~QueryPredicate(void) {};

      virtual QueryPredicate<T> *clone(void) const = 0;

      virtual bool matches_predicate(MachineImpl *machine, T thing) const = 0;
    };

    class ProcessorKindPredicate : public QueryPredicate<Processor> {
    public:
      ProcessorKindPredicate(Processor::Kind _kind);

      virtual QueryPredicate<Processor> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Processor thing) const;

    protected:
      Processor::Kind kind;
    };

    class ProcessorHasAffinityPredicate : public QueryPredicate<Processor> {
    public:
      ProcessorHasAffinityPredicate(Memory _memory, unsigned _min_bandwidth, unsigned _max_latency);

      virtual QueryPredicate<Processor> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Processor thing) const;

    protected:
      Memory memory;
      unsigned min_bandwidth;
      unsigned max_latency;
    };

    class ProcessorBestAffinityPredicate : public QueryPredicate<Processor> {
    public:
      ProcessorBestAffinityPredicate(Memory _memory, int _bandwidth_weight, int _latency_weight);

      virtual QueryPredicate<Processor> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Processor thing) const;

    protected:
      Memory memory;
      int bandwidth_weight;
      int latency_weight;
    };

    class ProcessorQueryImpl {
    public:
      ProcessorQueryImpl(const Machine& _machine);
     
    protected:
      // these things are refcounted and copied-on-write
      ProcessorQueryImpl(const ProcessorQueryImpl& copy_from);
      ~ProcessorQueryImpl(void);

    public:
      void add_reference(void);
      void remove_reference(void);
      // makes and returns if a copy if more than one reference is held
      ProcessorQueryImpl *writeable_reference(void);

      void restrict_to_node(int new_node_id);
      void add_predicate(QueryPredicate<Processor> *pred);

      Processor first_match(void) const;
      Processor next_match(Processor after) const;
      size_t count_matches(void) const;
      Processor random_match(void) const;

    protected:
      int references;
      MachineImpl *machine;
      bool is_restricted;
      int restricted_node_id;
      std::vector<QueryPredicate<Processor> *> predicates;     
    };            

    class MemoryKindPredicate : public QueryPredicate<Memory> {
    public:
      MemoryKindPredicate(Memory::Kind _kind);

      virtual QueryPredicate<Memory> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Memory thing) const;

    protected:
      Memory::Kind kind;
    };

    class MemoryHasProcAffinityPredicate : public QueryPredicate<Memory> {
    public:
      MemoryHasProcAffinityPredicate(Processor _proc, unsigned _min_bandwidth, unsigned _max_latency);

      virtual QueryPredicate<Memory> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Memory thing) const;

    protected:
      Processor proc;
      unsigned min_bandwidth;
      unsigned max_latency;
    };

    class MemoryHasMemAffinityPredicate : public QueryPredicate<Memory> {
    public:
      MemoryHasMemAffinityPredicate(Memory _memory, unsigned _min_bandwidth, unsigned _max_latency);

      virtual QueryPredicate<Memory> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Memory thing) const;

    protected:
      Memory memory;
      unsigned min_bandwidth;
      unsigned max_latency;
    };

    class MemoryBestProcAffinityPredicate : public QueryPredicate<Memory> {
    public:
      MemoryBestProcAffinityPredicate(Processor _proc, int _bandwidth_weight, int _latency_weight);

      virtual QueryPredicate<Memory> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Memory thing) const;

    protected:
      Processor proc;
      int bandwidth_weight;
      int latency_weight;
    };

    class MemoryBestMemAffinityPredicate : public QueryPredicate<Memory> {
    public:
      MemoryBestMemAffinityPredicate(Memory _memory, int _bandwidth_weight, int _latency_weight);

      virtual QueryPredicate<Memory> *clone(void) const;

      virtual bool matches_predicate(MachineImpl *machine, Memory thing) const;

    protected:
      Memory memory;
      int bandwidth_weight;
      int latency_weight;
    };

    class MemoryQueryImpl {
    public:
      MemoryQueryImpl(const Machine& _machine);
     
    protected:
      // these things are refcounted and copied-on-write
      MemoryQueryImpl(const MemoryQueryImpl& copy_from);
      ~MemoryQueryImpl(void);

    public:
      void add_reference(void);
      void remove_reference(void);
      // makes and returns if a copy if more than one reference is held
      MemoryQueryImpl *writeable_reference(void);

      void restrict_to_node(int new_node_id);
      void add_predicate(QueryPredicate<Memory> *pred);

      Memory first_match(void) const;
      Memory next_match(Memory after) const;
      size_t count_matches(void) const;
      Memory random_match(void) const;

    protected:
      int references;
      MachineImpl *machine;
      bool is_restricted;
      int restricted_node_id;
      std::vector<QueryPredicate<Memory> *> predicates;     
    };            

    extern MachineImpl *machine_singleton;
    inline MachineImpl *get_machine(void) { return machine_singleton; }

  // active messages

  enum {
    NODE_ANNOUNCE_DONE = 0,
    NODE_ANNOUNCE_PROC, // PROC id kind
    NODE_ANNOUNCE_MEM,  // MEM id size
    NODE_ANNOUNCE_IB_MEM, // IB_MEM id size
    NODE_ANNOUNCE_PMA,  // PMA proc_id mem_id bw latency
    NODE_ANNOUNCE_MMA,  // MMA mem1_id mem2_id bw latency
  };

  struct NodeAnnounceMessage {
    struct RequestArgs : public BaseMedium {
      gasnet_node_t node_id;
      unsigned num_procs;
      unsigned num_memories;
      unsigned num_ib_memories;
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<NODE_ANNOUNCE_MSGID,
				       RequestArgs,
				       handle_request> Message;

    static void send_request(gasnet_node_t target, unsigned num_procs,
			     unsigned num_memories, unsigned num_ib_memories,
			     const void *data, size_t datalen, int payload_mode);

    static void await_all_announcements(void);
  };

	
}; // namespace Realm

//include "machine_impl.inl"

#endif // ifndef REALM_MACHINE_IMPL_H

