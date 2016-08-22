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

#include "machine_impl.h"

#include "logging.h"
#include "proc_impl.h"
#include "mem_impl.h"
#include "runtime_impl.h"

#include "activemsg.h"

namespace Realm {

  Logger log_machine("machine");
  Logger log_annc("announce");

  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine
  //

    /*static*/ Machine Machine::get_machine(void) 
    {
      return Machine(get_runtime()->machine);
    }

    size_t Machine::get_address_space_count(void) const
    {
      return gasnet_nodes();
    }

    void Machine::get_all_memories(std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_all_memories(mset);
    }
    
    void Machine::get_all_processors(std::set<Processor>& pset) const
    {
      return ((MachineImpl *)impl)->get_all_processors(pset);
    }

    void Machine::get_local_processors(std::set<Processor>& pset) const
    {
      return ((MachineImpl *)impl)->get_local_processors(pset);
    }

    void Machine::get_local_processors_by_kind(std::set<Processor>& pset,
					       Processor::Kind kind) const
    {
      return ((MachineImpl *)impl)->get_local_processors_by_kind(pset, kind);
    }

    // Return the set of memories visible from a processor
    void Machine::get_visible_memories(Processor p, std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(p, mset);
    }

    // Return the set of memories visible from a memory
    void Machine::get_visible_memories(Memory m, std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(m, mset);
    }

    // Return the set of processors which can all see a given memory
    void Machine::get_shared_processors(Memory m, std::set<Processor>& pset) const
    {
      return ((MachineImpl *)impl)->get_shared_processors(m, pset);
    }

    bool Machine::has_affinity(Processor p, Memory m, AffinityDetails *details /*= 0*/) const
    {
      return ((MachineImpl *)impl)->has_affinity(p, m, details);
    }

    bool Machine::has_affinity(Memory m1, Memory m2, AffinityDetails *details /*= 0*/) const
    {
      return ((MachineImpl *)impl)->has_affinity(m1, m2, details);
    }

    int Machine::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				       Processor restrict_proc /*= Processor::NO_PROC*/,
				       Memory restrict_memory /*= Memory::NO_MEMORY*/) const
    {
      return ((MachineImpl *)impl)->get_proc_mem_affinity(result, restrict_proc, restrict_memory);
    }

    int Machine::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
				      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
				      Memory restrict_mem2 /*= Memory::NO_MEMORY*/) const
    {
      return ((MachineImpl *)impl)->get_mem_mem_affinity(result, restrict_mem1, restrict_mem2);
    }

    void Machine::add_subscription(MachineUpdateSubscriber *subscriber)
    {
      ((MachineImpl *)impl)->add_subscription(subscriber);
    }

    void Machine::remove_subscription(MachineUpdateSubscriber *subscriber)
    {
      ((MachineImpl *)impl)->remove_subscription(subscriber);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineImpl
  //

    MachineImpl *machine_singleton = 0;

  MachineImpl::MachineImpl(void)
  {
    assert(machine_singleton == 0);
    machine_singleton = this;
  }

  MachineImpl::~MachineImpl(void)
  {
    assert(machine_singleton == this);
    machine_singleton = 0;
  }

    void MachineImpl::parse_node_announce_data(int node_id,
					       unsigned num_procs, unsigned num_memories,
					       const void *args, size_t arglen,
					       bool remote)
    {
      AutoHSLLock al(mutex);

      const size_t *cur = (const size_t *)args;
#ifndef NDEBUG
      const size_t *limit = (const size_t *)(((const char *)args)+arglen);
#endif
      while(1) {
	assert(cur < limit);
	if(*cur == NODE_ANNOUNCE_DONE) break;
	switch(*cur++) {
	case NODE_ANNOUNCE_PROC:
	  {
	    ID id((ID::IDType)*cur++);
	    Processor p = id.convert<Processor>();
	    assert(id.proc.proc_idx < num_procs);
	    Processor::Kind kind = (Processor::Kind)(*cur++);
            int num_cores = (int)(*cur++);
            log_annc.debug() << "adding proc " << p << " (kind = " << kind << 
                                " num_cores = " << num_cores << ")";
	    if(remote) {
	      RemoteProcessor *proc = new RemoteProcessor(p, kind, num_cores);
	      get_runtime()->nodes[id.proc.owner_node].processors[id.proc.proc_idx] = proc;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_MEM:
	  {
	    ID id((ID::IDType)*cur++);
	    Memory m = id.convert<Memory>();
	    assert(id.memory.mem_idx < num_memories);
            Memory::Kind kind = (Memory::Kind)(*cur++);
	    size_t size = *cur++;
	    void *regbase = (void *)(*cur++);
	    log_annc.debug() << "adding memory " << m << " (kind = " << kind
			     << ", size = " << size << ", regbase = " << regbase << ")";
	    if(remote) {
	      RemoteMemory *mem = new RemoteMemory(m, size, kind, regbase);
	      get_runtime()->nodes[id.memory.owner_node].memories[id.memory.mem_idx] = mem;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_PMA:
	  {
	    Machine::ProcessorMemoryAffinity pma;
	    pma.p = ID((ID::IDType)*cur++).convert<Processor>();
	    pma.m = ID((ID::IDType)*cur++).convert<Memory>();
	    pma.bandwidth = *cur++;
	    pma.latency = *cur++;
	    log_annc.debug() << "adding affinity " << pma.p << " -> " << pma.m
			     << " (bw = " << pma.bandwidth << ", latency = " << pma.latency << ")";

	    proc_mem_affinities.push_back(pma);
	  }
	  break;

	case NODE_ANNOUNCE_MMA:
	  {
	    Machine::MemoryMemoryAffinity mma;
	    mma.m1 = ID((ID::IDType)*cur++).convert<Memory>();
	    mma.m2 = ID((ID::IDType)*cur++).convert<Memory>();
	    mma.bandwidth = *cur++;
	    mma.latency = *cur++;
	    log_annc.debug() << "adding affinity " << mma.m1 << " <-> " << mma.m2
			     << " (bw = " << mma.bandwidth << ", latency = " << mma.latency << ")";

	    mem_mem_affinities.push_back(mma);
	  }
	  break;

	default:
	  assert(0);
	}
      }
    }

    void MachineImpl::get_all_memories(std::set<Memory>& mset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	mset.insert((*it).m);
      }
    }

    void MachineImpl::get_all_processors(std::set<Processor>& pset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	pset.insert((*it).p);
      }
    }

    void MachineImpl::get_local_processors(std::set<Processor>& pset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	Processor p = (*it).p;
	if(ID(p).proc.owner_node == gasnet_mynode())
	  pset.insert(p);
      }
    }

    void MachineImpl::get_local_processors_by_kind(std::set<Processor>& pset,
						   Processor::Kind kind) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	Processor p = (*it).p;
	if((ID(p).proc.owner_node == gasnet_mynode()) && (p.kind() == kind))
	  pset.insert(p);
      }
    }

    // Return the set of memories visible from a processor
    void MachineImpl::get_visible_memories(Processor p, std::set<Memory>& mset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if((*it).p == p && (*it).m.capacity() > 0)
	  mset.insert((*it).m);
      }
    }

    // Return the set of memories visible from a memory
    void MachineImpl::get_visible_memories(Memory m, std::set<Memory>& mset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if((*it).m1 == m && (*it).m2.capacity() > 0)
	  mset.insert((*it).m2);
	
	if((*it).m2 == m && (*it).m1.capacity() > 0)
	  mset.insert((*it).m1);
      }
    }

    // Return the set of processors which can all see a given memory
    void MachineImpl::get_shared_processors(Memory m, std::set<Processor>& pset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if((*it).m == m)
	  pset.insert((*it).p);
      }
    }

    bool MachineImpl::has_affinity(Processor p, Memory m, Machine::AffinityDetails *details /*= 0*/) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if(it->p != p) continue;
	if(it->m != m) continue;
	if(details) {
	  details->bandwidth = it->bandwidth;
	  details->latency = it->latency;
	}
	return true;
      }
      return false;
    }

    bool MachineImpl::has_affinity(Memory m1, Memory m2, Machine::AffinityDetails *details /*= 0*/) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if(it->m1 != m1) continue;
	if(it->m2 != m2) continue;
	if(details) {
	  details->bandwidth = it->bandwidth;
	  details->latency = it->latency;
	}
	return true;
      }
      return false;
    }

    int MachineImpl::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
					     Processor restrict_proc /*= Processor::NO_PROC*/,
					     Memory restrict_memory /*= Memory::NO_MEMORY*/) const
    {
      int count = 0;

      {
	// TODO: consider using a reader/writer lock here instead
	AutoHSLLock al(mutex);
	for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	    it != proc_mem_affinities.end();
	    it++) {
	  if(restrict_proc.exists() && ((*it).p != restrict_proc)) continue;
	  if(restrict_memory.exists() && ((*it).m != restrict_memory)) continue;
	  result.push_back(*it);
	  count++;
	}
      }

      return count;
    }

    int MachineImpl::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
					    Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
					    Memory restrict_mem2 /*= Memory::NO_MEMORY*/) const
    {
      // Handle the case for same memories
      if (restrict_mem1.exists() && (restrict_mem1 == restrict_mem2))
      {
	Machine::MemoryMemoryAffinity affinity;
        affinity.m1 = restrict_mem1;
        affinity.m2 = restrict_mem1;
        affinity.bandwidth = 100;
        affinity.latency = 1;
        result.push_back(affinity);
        return 1;
      }

      int count = 0;

      {
	// TODO: consider using a reader/writer lock here instead
	AutoHSLLock al(mutex);
	for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	    it != mem_mem_affinities.end();
	    it++) {
	  if(restrict_mem1.exists() && 
	     ((*it).m1 != restrict_mem1)) continue;
	  if(restrict_mem2.exists() && 
	     ((*it).m2 != restrict_mem2)) continue;
	  result.push_back(*it);
	  count++;
	}
      }

      return count;
    }

    void MachineImpl::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma)
    {
      AutoHSLLock al(mutex);
      proc_mem_affinities.push_back(pma);
    }

    void MachineImpl::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma)
    {
      AutoHSLLock al(mutex);
      mem_mem_affinities.push_back(mma);
    }

    void MachineImpl::add_subscription(Machine::MachineUpdateSubscriber *subscriber)
    {
      AutoHSLLock al(mutex);
      subscribers.insert(subscriber);
    }

    void MachineImpl::remove_subscription(Machine::MachineUpdateSubscriber *subscriber)
    {
      AutoHSLLock al(mutex);
      subscribers.erase(subscriber);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine::ProcessorQuery
  //

  Machine::ProcessorQuery::ProcessorQuery(const Machine& m)
  {
    impl = new ProcessorQueryImpl(m);
  }

  Machine::ProcessorQuery::ProcessorQuery(const ProcessorQuery& q)
  {
    ((ProcessorQueryImpl *)(q.impl))->add_reference();
    impl = q.impl;
  }

  Machine::ProcessorQuery::~ProcessorQuery(void)
  {
    ((ProcessorQueryImpl *)impl)->remove_reference();
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::operator=(const ProcessorQuery& q)
  {
    if(impl != q.impl) {
      ((ProcessorQueryImpl *)impl)->remove_reference();
      ((ProcessorQueryImpl *)(q.impl))->add_reference();
      impl = q.impl;
    }
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::only_kind(Processor::Kind kind)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->add_predicate(new ProcessorKindPredicate(kind));
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::local_address_space(void)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(gasnet_mynode());
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::same_address_space_as(Processor p)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(ID(p).proc.owner_node);
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::same_address_space_as(Memory m)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(ID(m).proc.owner_node);
    return *this;
  }
      
  Machine::ProcessorQuery& Machine::ProcessorQuery::has_affinity_to(Memory m,
								    unsigned min_bandwidth /*= 0*/,
								    unsigned max_latency /*= 0*/)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->add_predicate(new ProcessorHasAffinityPredicate(m, min_bandwidth, max_latency));
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::best_affinity_to(Memory m,
								     int bandwidth_weight /*= 1*/,
								     int latency_weight /*= 0*/)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->add_predicate(new ProcessorBestAffinityPredicate(m, bandwidth_weight, latency_weight));
    return *this;
  }

  size_t Machine::ProcessorQuery::count(void) const
  {
    return ((ProcessorQueryImpl *)impl)->count_matches();
  }

  Processor Machine::ProcessorQuery::first(void) const
  {
    return ((ProcessorQueryImpl *)impl)->first_match();
  }

  Processor Machine::ProcessorQuery::next(Processor after) const
  {
    return ((ProcessorQueryImpl *)impl)->next_match(after);
  }

  Processor Machine::ProcessorQuery::random(void) const
  {
    return ((ProcessorQueryImpl *)impl)->random_match();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine::MemoryQuery
  //

  Machine::MemoryQuery::MemoryQuery(const Machine& m)
  {
    impl = new MemoryQueryImpl(m);
  }

  Machine::MemoryQuery::MemoryQuery(const MemoryQuery& q)
  {
    ((MemoryQueryImpl *)(q.impl))->add_reference();
    impl = q.impl;
  }

  Machine::MemoryQuery::~MemoryQuery(void)
  {
    ((MemoryQueryImpl *)impl)->remove_reference();
  }

  Machine::MemoryQuery& Machine::MemoryQuery::operator=(const MemoryQuery& q)
  {
    if(impl != q.impl) {
      ((MemoryQueryImpl *)impl)->remove_reference();
      ((MemoryQueryImpl *)(q.impl))->add_reference();
      impl = q.impl;
    }
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::only_kind(Memory::Kind kind)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->add_predicate(new MemoryKindPredicate(kind));
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::local_address_space(void)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(gasnet_mynode());
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::same_address_space_as(Processor p)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(ID(p).proc.owner_node);
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::same_address_space_as(Memory m)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(ID(m).memory.owner_node);
    return *this;
  }
      
  Machine::MemoryQuery& Machine::MemoryQuery::has_affinity_to(Memory m,
							      unsigned min_bandwidth /*= 0*/,
							      unsigned max_latency /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->add_predicate(new MemoryHasMemAffinityPredicate(m, min_bandwidth, max_latency));
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::best_affinity_to(Memory m,
							       int bandwidth_weight /*= 1*/,
							       int latency_weight /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->add_predicate(new MemoryBestMemAffinityPredicate(m, bandwidth_weight, latency_weight));
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::has_affinity_to(Processor p,
							      unsigned min_bandwidth /*= 0*/,
							      unsigned max_latency /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->add_predicate(new MemoryHasProcAffinityPredicate(p, min_bandwidth, max_latency));
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::best_affinity_to(Processor p,
							       int bandwidth_weight /*= 1*/,
							       int latency_weight /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->add_predicate(new MemoryBestProcAffinityPredicate(p, bandwidth_weight, latency_weight));
    return *this;
  }

  size_t Machine::MemoryQuery::count(void) const
  {
    return ((MemoryQueryImpl *)impl)->count_matches();
  }

  Memory Machine::MemoryQuery::first(void) const
  {
    return ((MemoryQueryImpl *)impl)->first_match();
  }

  Memory Machine::MemoryQuery::next(Memory after) const
  {
    return ((MemoryQueryImpl *)impl)->next_match(after);
  }

  Memory Machine::MemoryQuery::random(void) const
  {
    return ((MemoryQueryImpl *)impl)->random_match();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorKindPredicate
  //

  ProcessorKindPredicate::ProcessorKindPredicate(Processor::Kind _kind)
    : kind(_kind)
  {}

  QueryPredicate<Processor> *ProcessorKindPredicate::clone(void) const
  {
    return new ProcessorKindPredicate(kind);
  }

  bool ProcessorKindPredicate::matches_predicate(MachineImpl *machine, Processor thing) const
  {
    return (thing.kind() == kind);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorHasAffinityPredicate
  //

  ProcessorHasAffinityPredicate::ProcessorHasAffinityPredicate(Memory _memory,
							       unsigned _min_bandwidth,
							       unsigned _max_latency)
    : memory(_memory)
    , min_bandwidth(_min_bandwidth)
    , max_latency(_max_latency)
  {}

  QueryPredicate<Processor> *ProcessorHasAffinityPredicate::clone(void) const
  {
    return new ProcessorHasAffinityPredicate(memory, min_bandwidth, max_latency);
  }

  bool ProcessorHasAffinityPredicate::matches_predicate(MachineImpl *machine, Processor thing) const
  {
    Machine::AffinityDetails details;
    if(!machine->has_affinity(thing, memory, &details)) return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (details.latency > max_latency)) return false;
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorBestAffinityPredicate
  //

  ProcessorBestAffinityPredicate::ProcessorBestAffinityPredicate(Memory _memory,
								 int _bandwidth_weight,
								 int _latency_weight)
    : memory(_memory)
    , bandwidth_weight(_bandwidth_weight)
    , latency_weight(_latency_weight)
  {}

  QueryPredicate<Processor> *ProcessorBestAffinityPredicate::clone(void) const
  {
    return new ProcessorBestAffinityPredicate(memory, bandwidth_weight, latency_weight);
  }

  bool ProcessorBestAffinityPredicate::matches_predicate(MachineImpl *machine, Processor thing) const
  {
    Memory best = Memory::NO_MEMORY;
    int best_aff = INT_MIN;
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine->get_proc_mem_affinity(affinities, thing);
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = affinities.begin();
	it != affinities.end();
	it++) {
      int aff = (it->bandwidth * bandwidth_weight) + (it->latency * latency_weight);
      if(aff > best_aff) {
	best_aff = aff;
	best = it->m;
      }
    }
    return (best == memory);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorQueryImpl
  //

  ProcessorQueryImpl::ProcessorQueryImpl(const Machine& _machine)
    : references(1)
    , machine((MachineImpl *)_machine.impl)
    , is_restricted(false)
    , restricted_node_id(-1)
  {}
     
  ProcessorQueryImpl::ProcessorQueryImpl(const ProcessorQueryImpl& copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted(copy_from.is_restricted)
    , restricted_node_id(copy_from.restricted_node_id)
  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<QueryPredicate<Processor> *>::const_iterator it = copy_from.predicates.begin();
	it != copy_from.predicates.end();
	it++)
      predicates.push_back((*it)->clone());
  }

  ProcessorQueryImpl::~ProcessorQueryImpl(void)
  {
    assert(references == 0);
    for(std::vector<QueryPredicate<Processor> *>::iterator it = predicates.begin();
	it != predicates.end();
	it++)
      delete *it;
  }

  void ProcessorQueryImpl::add_reference(void)
  {
    __sync_fetch_and_add(&references, 1);
  }

  void ProcessorQueryImpl::remove_reference(void)
  {
    int left = __sync_sub_and_fetch(&references, 1);
    if(left == 0)
      delete this;
  }

  ProcessorQueryImpl *ProcessorQueryImpl::writeable_reference(void)
  {
    // safe to test without an atomic because we are a reference, and if the count is 1,
    //  there can be no others
    if(references == 1) {
      return this;
    } else {
      ProcessorQueryImpl *copy = new ProcessorQueryImpl(*this);
      remove_reference();
      return copy;
    }
  }

  void ProcessorQueryImpl::restrict_to_node(int new_node_id)
  {
    // attempts to restrict to two different nodes results in no possible match
    if(is_restricted && (new_node_id != restricted_node_id)) {
      restricted_node_id = -1;
    } else {
      is_restricted = true;
      restricted_node_id = new_node_id;
    }
  }

  void ProcessorQueryImpl::add_predicate(QueryPredicate<Processor> *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
  }

  Processor ProcessorQueryImpl::first_match(void) const
  {
    if(is_restricted && (restricted_node_id < 0)) return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Processor> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, p);
	if(ok && (!lowest.exists() || (p.id < lowest.id)))
	  lowest = p;
      }
    }
    return lowest;
  }

  Processor ProcessorQueryImpl::next_match(Processor after) const
  {
    if(is_restricted && (restricted_node_id < 0)) return Processor::NO_PROC;
    if(!after.exists()) return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(p.id <= after.id) continue;
	if(is_restricted && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Processor> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, p);
	if(ok && (!lowest.exists() || (p.id < lowest.id)))
	  lowest = p;
      }
    }
    return lowest;
  }

  size_t ProcessorQueryImpl::count_matches(void) const
  {
    if(is_restricted && (restricted_node_id < 0)) return 0;
    std::set<Processor> pset;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Processor> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, p);
	if(ok)
	  pset.insert((*it).p);
      }
    }
    return pset.size();
  }

  Processor ProcessorQueryImpl::random_match(void) const
  {
    if(is_restricted && (restricted_node_id < 0)) return Processor::NO_PROC;
    Processor chosen = Processor::NO_PROC;
    int count = 0;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Processor> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, p);
	if(ok) {
	  count++;
	  if((count == 1) || ((lrand48() % count) == 0))
	    chosen = p;
	}
      }
    }
    return chosen;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryKindPredicate
  //

  MemoryKindPredicate::MemoryKindPredicate(Memory::Kind _kind)
    : kind(_kind)
  {}

  QueryPredicate<Memory> *MemoryKindPredicate::clone(void) const
  {
    return new MemoryKindPredicate(kind);
  }

  bool MemoryKindPredicate::matches_predicate(MachineImpl *machine, Memory thing) const
  {
    return (thing.kind() == kind);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryHasProcAffinityPredicate
  //

  MemoryHasProcAffinityPredicate::MemoryHasProcAffinityPredicate(Processor _proc,
								 unsigned _min_bandwidth,
								 unsigned _max_latency)
    : proc(_proc)
    , min_bandwidth(_min_bandwidth)
    , max_latency(_max_latency)
  {}

  QueryPredicate<Memory> *MemoryHasProcAffinityPredicate::clone(void) const
  {
    return new MemoryHasProcAffinityPredicate(proc, min_bandwidth, max_latency);
  }

  bool MemoryHasProcAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing) const
  {
    Machine::AffinityDetails details;
    if(!machine->has_affinity(proc, thing, &details)) return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (details.latency > max_latency)) return false;
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryHasMemAffinityPredicate
  //

  MemoryHasMemAffinityPredicate::MemoryHasMemAffinityPredicate(Memory _memory,
							       unsigned _min_bandwidth,
							       unsigned _max_latency)
    : memory(_memory)
    , min_bandwidth(_min_bandwidth)
    , max_latency(_max_latency)
  {}

  QueryPredicate<Memory> *MemoryHasMemAffinityPredicate::clone(void) const
  {
    return new MemoryHasMemAffinityPredicate(memory, min_bandwidth, max_latency);
  }

  bool MemoryHasMemAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing) const
  {
    Machine::AffinityDetails details;
    if(!machine->has_affinity(memory, thing, &details)) return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (details.latency > max_latency)) return false;
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryBestProcAffinityPredicate
  //

  MemoryBestProcAffinityPredicate::MemoryBestProcAffinityPredicate(Processor _proc,
								   int _bandwidth_weight,
								   int _latency_weight)
    : proc(_proc)
    , bandwidth_weight(_bandwidth_weight)
    , latency_weight(_latency_weight)
  {}

  QueryPredicate<Memory> *MemoryBestProcAffinityPredicate::clone(void) const
  {
    return new MemoryBestProcAffinityPredicate(proc, bandwidth_weight, latency_weight);
  }

  bool MemoryBestProcAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing) const
  {
    Processor best = Processor::NO_PROC;
    int best_aff = INT_MIN;
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine->get_proc_mem_affinity(affinities, Processor::NO_PROC, thing);
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = affinities.begin();
	it != affinities.end();
	it++) {
      int aff = (it->bandwidth * bandwidth_weight) + (it->latency * latency_weight);
      if(aff > best_aff) {
	best_aff = aff;
	best = it->p;
      }
    }
    return (best == proc);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryBestMemAffinityPredicate
  //

  MemoryBestMemAffinityPredicate::MemoryBestMemAffinityPredicate(Memory _memory,
								 int _bandwidth_weight,
								 int _latency_weight)
    : memory(_memory)
    , bandwidth_weight(_bandwidth_weight)
    , latency_weight(_latency_weight)
  {}

  QueryPredicate<Memory> *MemoryBestMemAffinityPredicate::clone(void) const
  {
    return new MemoryBestMemAffinityPredicate(memory, bandwidth_weight, latency_weight);
  }

  bool MemoryBestMemAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing) const
  {
    Memory best = Memory::NO_MEMORY;
    int best_aff = INT_MIN;
    std::vector<Machine::MemoryMemoryAffinity> affinities;
    machine->get_mem_mem_affinity(affinities, thing);
    for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = affinities.begin();
	it != affinities.end();
	it++) {
      int aff = (it->bandwidth * bandwidth_weight) + (it->latency * latency_weight);
      if(aff > best_aff) {
	best_aff = aff;
	best = it->m2;
      }
    }
    return (best == memory);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryQueryImpl
  //

  MemoryQueryImpl::MemoryQueryImpl(const Machine& _machine)
    : references(1)
    , machine((MachineImpl *)_machine.impl)
    , is_restricted(false)
    , restricted_node_id(-1)
  {}
     
  MemoryQueryImpl::MemoryQueryImpl(const MemoryQueryImpl& copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted(copy_from.is_restricted)
    , restricted_node_id(copy_from.restricted_node_id)
  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<QueryPredicate<Memory> *>::const_iterator it = copy_from.predicates.begin();
	it != copy_from.predicates.end();
	it++)
      predicates.push_back((*it)->clone());
  }

  MemoryQueryImpl::~MemoryQueryImpl(void)
  {
    assert(references == 0);
    for(std::vector<QueryPredicate<Memory> *>::iterator it = predicates.begin();
	it != predicates.end();
	it++)
      delete *it;
  }

  void MemoryQueryImpl::add_reference(void)
  {
    __sync_fetch_and_add(&references, 1);
  }

  void MemoryQueryImpl::remove_reference(void)
  {
    int left = __sync_sub_and_fetch(&references, 1);
    if(left == 0)
      delete this;
  }

  MemoryQueryImpl *MemoryQueryImpl::writeable_reference(void)
  {
    // safe to test without an atomic because we are a reference, and if the count is 1,
    //  there can be no others
    if(references == 1) {
      return this;
    } else {
      MemoryQueryImpl *copy = new MemoryQueryImpl(*this);
      remove_reference();
      return copy;
    }
  }

  void MemoryQueryImpl::restrict_to_node(int new_node_id)
  {
    // attempts to restrict to two different nodes results in no possible match
    if(is_restricted && (new_node_id != restricted_node_id)) {
      restricted_node_id = -1;
    } else {
      is_restricted = true;
      restricted_node_id = new_node_id;
    }
  }

  void MemoryQueryImpl::add_predicate(QueryPredicate<Memory> *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
  }

  Memory MemoryQueryImpl::first_match(void) const
  {
    if(is_restricted && (restricted_node_id < 0)) return Memory::NO_MEMORY;
    Memory lowest = Memory::NO_MEMORY;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Memory> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, m);
	if(ok && (!lowest.exists() || (m.id < lowest.id)))
	  lowest = m;
      }
    }
    return lowest;
  }

  Memory MemoryQueryImpl::next_match(Memory after) const
  {
    if(is_restricted && (restricted_node_id < 0)) return Memory::NO_MEMORY;
    if(!after.exists()) return Memory::NO_MEMORY;
    Memory lowest = Memory::NO_MEMORY;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(m.id <= after.id) continue;
	if(is_restricted && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Memory> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, m);
	if(ok && (!lowest.exists() || (m.id < lowest.id)))
	  lowest = m;
      }
    }
    return lowest;
  }

  size_t MemoryQueryImpl::count_matches(void) const
  {
    if(is_restricted && (restricted_node_id < 0)) return 0;
    std::set<Memory> pset;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Memory> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, m);
	if(ok)
	  pset.insert((*it).m);
      }
    }
    return pset.size();
  }

  Memory MemoryQueryImpl::random_match(void) const
  {
    if(is_restricted && (restricted_node_id < 0)) return Memory::NO_MEMORY;
    Memory chosen = Memory::NO_MEMORY;
    int count = 0;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	bool ok = true;
	for(std::vector<QueryPredicate<Memory> *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, m);
	if(ok) {
	  count++;
	  if((count == 1) || ((lrand48() % count) == 0))
	    chosen = m;
	}
      }
    }
    return chosen;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class NodeAnnounceMessage
  //

  static int announcements_received = 0;

  /*static*/ void NodeAnnounceMessage::handle_request(RequestArgs args,
						      const void *data,
						      size_t datalen)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    log_annc.info("%d: received announce from %d (%d procs, %d memories)\n",
		  gasnet_mynode(),
		  args.node_id,
		  args.num_procs,
		  args.num_memories);
    
    Node *n = &(get_runtime()->nodes[args.node_id]);
    n->processors.resize(args.num_procs);
    n->memories.resize(args.num_memories);

    // do the parsing of this data inside a mutex because it touches common
    //  data structures
    {
      get_machine()->parse_node_announce_data(args.node_id,
					      args.num_procs, args.num_memories,
					      data, datalen, true);

      __sync_fetch_and_add(&announcements_received, 1);
    }
  }

  /*static*/ void NodeAnnounceMessage::send_request(gasnet_node_t target,
						    unsigned num_procs,
						    unsigned num_memories,
						    const void *data,
						    size_t datalen,
						    int payload_mode)
  {
    RequestArgs args;

    args.node_id = gasnet_mynode();
    args.num_procs = num_procs;
    args.num_memories = num_memories;
    Message::request(target, args, data, datalen, payload_mode);
  }

  /*static*/ void NodeAnnounceMessage::await_all_announcements(void)
  {
    // wait until we hear from everyone else?
    while((int)announcements_received < (int)(gasnet_nodes() - 1))
      do_some_polling();

    log_annc.info("node %d has received all of its announcements\n", gasnet_mynode());
  }

}; // namespace Realm
