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
	    assert(id.index() < num_procs);
	    Processor::Kind kind = (Processor::Kind)(*cur++);
	    log_annc.debug() << "adding proc " << p << " (kind = " << kind << ")";
	    if(remote) {
	      RemoteProcessor *proc = new RemoteProcessor(p, kind);
	      get_runtime()->nodes[ID(p).node()].processors[ID(p).index()] = proc;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_MEM:
	  {
	    ID id((ID::IDType)*cur++);
	    Memory m = id.convert<Memory>();
	    assert(id.index_h() < num_memories);
            Memory::Kind kind = (Memory::Kind)(*cur++);
	    unsigned size = *cur++;
	    void *regbase = (void *)(*cur++);
	    log_annc.debug() << "adding memory " << m << " (kind = " << kind
			     << ", size = " << size << ", regbase = " << regbase << ")";
	    if(remote) {
	      RemoteMemory *mem = new RemoteMemory(m, size, kind, regbase);
	      get_runtime()->nodes[ID(m).node()].memories[ID(m).index_h()] = mem;
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
	if(ID(p).node() == gasnet_mynode())
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
	if((ID(p).node() == gasnet_mynode()) && (p.kind() == kind))
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
	if((*it).p == p)
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
	if((*it).m1 == m)
	  mset.insert((*it).m2);
	
	if((*it).m2 == m)
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
