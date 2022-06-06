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

// machine model for Realm

#ifndef REALM_MACHINE_H
#define REALM_MACHINE_H

#include "realm/processor.h"
#include "realm/memory.h"

#include <iterator>

namespace Realm {

    class Runtime;

    class REALM_PUBLIC_API Machine {
    protected:
      friend class Runtime;
      explicit Machine(void *_impl) : impl(_impl) {}

    public:
      Machine(const Machine& m) : impl(m.impl) {}
      Machine& operator=(const Machine& m) { impl = m.impl; return *this; }
      ~Machine(void) {}

      static Machine get_machine(void);

      class ProcessorQuery;
      class MemoryQuery;

      struct AffinityDetails {
	unsigned bandwidth;
	unsigned latency;
      };

      bool has_affinity(Processor p, Memory m, AffinityDetails *details = 0) const;
      bool has_affinity(Memory m1, Memory m2, AffinityDetails *details = 0) const;

      // older queries, to be deprecated

      void get_all_memories(std::set<Memory>& mset) const;
      void get_all_processors(std::set<Processor>& pset) const;

      void get_local_processors(std::set<Processor>& pset) const;
      void get_local_processors_by_kind(std::set<Processor>& pset,
					Processor::Kind kind) const;

      // Return the set of memories visible from a processor
      void get_visible_memories(Processor p, std::set<Memory>& mset,
				bool local_only = true) const;

      // Return the set of memories visible from a memory
      void get_visible_memories(Memory m, std::set<Memory>& mset,
				bool local_only = true) const;

      // Return the set of processors which can all see a given memory
      void get_shared_processors(Memory m, std::set<Processor>& pset,
				 bool local_only = true) const;

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
				Memory restrict_memory = Memory::NO_MEMORY,
				bool local_only = true) const;

      int get_mem_mem_affinity(std::vector<MemoryMemoryAffinity>& result,
			       Memory restrict_mem1 = Memory::NO_MEMORY,
			       Memory restrict_mem2 = Memory::NO_MEMORY,
			       bool local_only = true) const;

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

      // subscriptions are encouraged to use a query which filters which processors or
      //  memories cause notifications
      void add_subscription(MachineUpdateSubscriber *subscriber);
      void add_subscription(MachineUpdateSubscriber *subscriber,
			    const ProcessorQuery &query);
      void add_subscription(MachineUpdateSubscriber *subscriber,
			    const MemoryQuery &query);

      void remove_subscription(MachineUpdateSubscriber *subscriber);

      void *impl;  // hidden internal implementation - this is NOT a transferrable handle
    };

    template <typename QT, typename RT> class MachineQueryIterator;

    class REALM_PUBLIC_API Machine::ProcessorQuery {
    public:
      explicit ProcessorQuery(const Machine& m);
      ProcessorQuery(const ProcessorQuery& q);

      ~ProcessorQuery(void);

      ProcessorQuery& operator=(const ProcessorQuery& q);

      bool operator==(const ProcessorQuery& compare_to) const;
      bool operator!=(const ProcessorQuery& compare_to) const;

      // filter predicates (returns self-reference for chaining)
      // if multiple predicates are used, they must all match (i.e. the intersection is returned)

      // restrict to just those of the specified 'kind'
      ProcessorQuery& only_kind(Processor::Kind kind);

      // restrict to those managed by this address space
      ProcessorQuery& local_address_space(void);

      // restrict to those in same address space as specified Processor or Memory
      ProcessorQuery& same_address_space_as(Processor p);
      ProcessorQuery& same_address_space_as(Memory m);
      
      // restrict to those that have affinity to a given memory
      ProcessorQuery& has_affinity_to(Memory m, unsigned min_bandwidth = 0, unsigned max_latency = 0);

      // restrict to those whose best affinity is to the given memory
      ProcessorQuery& best_affinity_to(Memory m, int bandwidth_weight = 1, int latency_weight = 0);

      // results - a query may be executed multiple times - when the machine model is
      //  dynamic, there is no guarantee that the results of any two executions will be consistent

      // return the number of matched processors
      size_t count(void) const;

      // return the first matched processor, or NO_PROC
      Processor first(void) const;

      // return the next matched processor after the one given, or NO_PROC
      Processor next(Processor after) const;

      // return a random matched processor, or NO_PROC if none exist
      Processor random(void) const;

      typedef MachineQueryIterator<ProcessorQuery, Processor> iterator;

      // return an iterator that allows enumerating all matched processors
      iterator begin(void) const;
      iterator end(void) const;
      
    protected:
      void *impl;
    };

    class REALM_PUBLIC_API Machine::MemoryQuery {
    public:
      explicit MemoryQuery(const Machine& m);
      MemoryQuery(const MemoryQuery& q);

      ~MemoryQuery(void);

      MemoryQuery& operator=(const MemoryQuery& q);

      bool operator==(const MemoryQuery& compare_to) const;
      bool operator!=(const MemoryQuery& compare_to) const;

      // filter predicates (returns self-reference for chaining)
      // if multiple predicates are used, they must all match (i.e. the intersection is returned)

      // restrict to just those of the specified 'kind'
      MemoryQuery& only_kind(Memory::Kind kind);

      // restrict to those managed by this address space
      MemoryQuery& local_address_space(void);

      // restrict to those in same address space as specified Processor or Memory
      MemoryQuery& same_address_space_as(Processor p);
      MemoryQuery& same_address_space_as(Memory m);
      
      // restrict to those that have affinity to a given processor or memory
      MemoryQuery& has_affinity_to(Processor p, unsigned min_bandwidth = 0, unsigned max_latency = 0);
      MemoryQuery& has_affinity_to(Memory m, unsigned min_bandwidth = 0, unsigned max_latency = 0);

      // restrict to those whose best affinity is to the given processor or memory
      MemoryQuery& best_affinity_to(Processor p, int bandwidth_weight = 1, int latency_weight = 0);
      MemoryQuery& best_affinity_to(Memory m, int bandwidth_weight = 1, int latency_weight = 0);

      // restrict to those whose total capacity is at least 'min_size' bytes
      MemoryQuery& has_capacity(size_t min_bytes);

      // results - a query may be executed multiple times - when the machine model is
      //  dynamic, there is no guarantee that the results of any two executions will be consistent

      // return the number of matched processors
      size_t count(void) const;

      // return the first matched processor, or NO_PROC
      Memory first(void) const;

      // return the next matched processor after the one given, or NO_PROC
      Memory next(Memory after) const;

      // return a random matched processor, or NO_PROC if none exist
      Memory random(void) const;

      typedef MachineQueryIterator<MemoryQuery, Memory> iterator;

      // return an iterator that allows enumerating all matched processors
      iterator begin(void) const;
      iterator end(void) const;
      
    protected:
      void *impl;
    };

    template <typename QT, typename RT>
    class REALM_PUBLIC_API MachineQueryIterator {
    public:
      // explicitly set iterator traits
      typedef std::input_iterator_tag iterator_category;
      typedef RT value_type;
      typedef std::ptrdiff_t difference_type;
      typedef RT *pointer;
      typedef RT& reference;

      // would like this constructor to be protected and have QT be a friend, but that requires
      //  C++11 (or a compiler like g++ that supports it even without -std=c++11)
      //  The CUDA compiler also seems to be a little dense here as well
#if (REALM_CXX_STANDARD >= 11) && (!defined(__CUDACC__) && !defined(__HIPCC__))
    protected:
      friend QT;
#else
    public:
#endif
      MachineQueryIterator(const QT& _query, RT _result);

    protected:
      QT query;
      RT result;

    public:
      MachineQueryIterator(const MachineQueryIterator<QT,RT>& copy_from);
      ~MachineQueryIterator(void);
	
      MachineQueryIterator<QT,RT>& operator=(const MachineQueryIterator<QT,RT>& copy_from);
	
      bool operator==(const MachineQueryIterator<QT,RT>& compare_to) const;
      bool operator!=(const MachineQueryIterator<QT,RT>& compare_to) const;
	
      RT operator*(void);
      const RT *operator->(void);
	
      MachineQueryIterator<QT,RT>& operator++(/*prefix*/);
      MachineQueryIterator<QT,RT> operator++(int /*postfix*/);

      // in addition to testing an iterator against .end(), you can also cast to bool, allowing
      // for(iterator it = q.begin(); q; ++q) ...
      operator bool(void) const;
    };
      
	
}; // namespace Realm

#include "realm/machine.inl"

#endif // ifndef REALM_MACHINE_H

