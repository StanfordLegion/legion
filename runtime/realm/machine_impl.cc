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

#include "machine_impl.h"

#include "logging.h"
#include "proc_impl.h"
#include "mem_impl.h"
#include "runtime_impl.h"

#include "activemsg.h"

namespace Realm {

  Logger log_machine("machine");
  Logger log_annc("announce");

  template <typename KT, typename VT>
  inline void delete_map_contents(std::map<KT,VT *>& m)
  {
    for(typename std::map<KT,VT *>::const_iterator it = m.begin();
	it != m.end();
	++it)
      delete it->second;
  }

  static inline bool is_local_affinity(const Machine::ProcessorMemoryAffinity& pma)
  {
    return ID(pma.p).proc.owner_node == ID(pma.m).memory.owner_node;
  }

  static inline bool is_local_affinity(const Machine::MemoryMemoryAffinity& mma)
  {
    return ID(mma.m1).memory.owner_node == ID(mma.m2).memory.owner_node;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineAffinityInfo<KT,AT>
  //

  template <typename KT, typename AT>
  MachineAffinityInfo<KT,AT>::MachineAffinityInfo(void)
  {}

  template <typename KT, typename AT>
  MachineAffinityInfo<KT,AT>::~MachineAffinityInfo(void)
  {
    delete_map_contents(all);
#if 0
    for(std::map<KT, AT *>::const_iterator it = all.begin();
	it != all.end();
	++it)
      delete it->second;
#endif
  }

  template <typename KT, typename AT>
  bool MachineAffinityInfo<KT,AT>::add_affinity(KT key, const AT& aff,
						bool is_local)
  {
    // look up existing affinity
    AT *& ptr = all[key];
    if(ptr == 0) {
      // create a new entry
      ptr = new AT(aff);
      if(is_local)
	local[key] = ptr;
    } else {
      // see if it's an exact match
      if((ptr->bandwidth == aff.bandwidth) && (ptr->latency == aff.latency))
	return false; // no change

      ptr->bandwidth = aff.bandwidth;
      ptr->latency = aff.latency;
    }

    // maybe update the best set
    if(best.empty()) {
      // easy case - we're best by default
      best[key] = ptr;
    } else {
      // all existing ones are equal so just look at the first
      AT *cur_best = best.begin()->second;
      if(cur_best->bandwidth > aff.bandwidth) {
	// better ones exist - do nothing
      } else if(cur_best->bandwidth < aff.bandwidth) {
	// this is better than existing - erase those and replace with us
	best.clear();
	best[key] = ptr;
      } else {
	// equal to existing ones - add ourselves
	best[key] = ptr;
      }
    }

    return true; // something changed
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineProcInfo
  //

  MachineProcInfo::MachineProcInfo(Processor _p)
    : p(_p)
  {}

  MachineProcInfo::~MachineProcInfo(void)
  {}

  bool MachineProcInfo::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma)
  {
    bool is_local = is_local_affinity(pma);
    return pmas.add_affinity(pma.m, pma, is_local);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineMemInfo
  //

  MachineMemInfo::MachineMemInfo(Memory _m)
    : m(_m)
  {}
    
  MachineMemInfo::~MachineMemInfo(void)
  {}

  bool MachineMemInfo::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma)
  {
    bool is_local = is_local_affinity(pma);
    return pmas.add_affinity(pma.p, pma, is_local);
  }

  bool MachineMemInfo::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma)
  {
    bool is_local = is_local_affinity(mma);

    if(mma.m1 == m)
      return mmas_out.add_affinity(mma.m2, mma, is_local);

    if(mma.m2 == m)
      return mmas_in.add_affinity(mma.m1, mma, is_local);

    // shouldn't have been called!
    assert(0);
    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineNodeInfo
  //

  MachineNodeInfo::MachineNodeInfo(int _node)
    : node(_node)
  {}

  MachineNodeInfo::~MachineNodeInfo(void)
  {
    delete_map_contents(procs);
    delete_map_contents(mems);
  }

  bool MachineNodeInfo::add_processor(Processor p)
  {
    assert(node == ID(p).proc.owner_node);
    MachineProcInfo *& ptr = procs[p];
    // TODO: see if anything changed?
    if(ptr != 0)
      return false;

    ptr = new MachineProcInfo(p);

    Processor::Kind k = p.kind();
    proc_by_kind[k][p] = ptr;
    return true;
  }
  
  bool MachineNodeInfo::add_memory(Memory m)
  {
    assert(node == ID(m).memory.owner_node);
    MachineMemInfo *& ptr = mems[m];
    // TODO: see if anything changed?
    if(ptr != 0)
      return false;

    ptr = new MachineMemInfo(m);

    Memory::Kind k = m.kind();
    mem_by_kind[k][m] = ptr;
    return true;
  }

  bool MachineNodeInfo::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma)
  {
    bool changed = false;

    if(ID(pma.p).proc.owner_node == node) {
      MachineProcInfo *mpi = procs[pma.p];
      assert(mpi != 0);
      if(mpi->add_proc_mem_affinity(pma))
	changed = true;
    }

    if(ID(pma.m).memory.owner_node == node) {
      MachineMemInfo *mmi = mems[pma.m];
      assert(mmi != 0);
      if(mmi->add_proc_mem_affinity(pma))
	changed = true;
    }

    return changed;
  }

  bool MachineNodeInfo::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma)
  {
    bool changed = false;

    if(ID(mma.m1).memory.owner_node == node) {
      MachineMemInfo *mmi = mems[mma.m1];
      assert(mmi != 0);
      if(mmi->add_mem_mem_affinity(mma))
	changed = true;
    }

    if(ID(mma.m2).memory.owner_node == node) {
      MachineMemInfo *mmi = mems[mma.m2];
      assert(mmi != 0);
      if(mmi->add_mem_mem_affinity(mma))
	changed = true;
    }

    return changed;
  }


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
    void Machine::get_visible_memories(Processor p, std::set<Memory>& mset,
				       bool local_only /*= true*/) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(p, mset, local_only);
    }

    // Return the set of memories visible from a memory
    void Machine::get_visible_memories(Memory m, std::set<Memory>& mset,
				       bool local_only /*= true*/) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(m, mset, local_only);
    }

    // Return the set of processors which can all see a given memory
    void Machine::get_shared_processors(Memory m, std::set<Processor>& pset,
					bool local_only /*= true*/) const
    {
      return ((MachineImpl *)impl)->get_shared_processors(m, pset, local_only);
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
				       Memory restrict_memory /*= Memory::NO_MEMORY*/,
				       bool local_only /*= true*/) const
    {
      return ((MachineImpl *)impl)->get_proc_mem_affinity(result, restrict_proc, restrict_memory, local_only);
    }

    int Machine::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
				      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
				      Memory restrict_mem2 /*= Memory::NO_MEMORY*/,
				      bool local_only /*= true*/) const
    {
      return ((MachineImpl *)impl)->get_mem_mem_affinity(result, restrict_mem1, restrict_mem2, local_only);
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
    delete_map_contents(nodeinfos);
  }

#ifndef REALM_SKIP_INTERNODE_AFFINITIES
  static bool allows_internode_copies(Memory::Kind kind,
				      bool& send_ok, bool& recv_ok,
				      bool& is_reg)
  {
    send_ok = false;
    recv_ok = false;
    is_reg = false;
    switch(kind) {
      // these can be the source of a RemoteWrite, but it's non-ideal
    case Memory::SYSTEM_MEM:
    case Memory::Z_COPY_MEM:
      {
	send_ok = true;
	recv_ok = true;
	break;
      }

      // these can be the target of a RemoteWrite message, but
      //  not the source
    case Memory::GPU_FB_MEM:
    case Memory::DISK_MEM:
    case Memory::HDF_MEM:
    case Memory::FILE_MEM:
      {
	recv_ok = true;
	break;
      }

    case Memory::REGDMA_MEM:
      {
	send_ok = true;
	recv_ok = true;
	is_reg = true;
	break;
      }

    default: break;
    }

    return (send_ok || recv_ok);
  }
#endif

    void MachineImpl::parse_node_announce_data(int node_id, unsigned num_procs,
					       unsigned num_memories, unsigned num_ib_memories,
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

#ifndef REALM_SKIP_INTERNODE_AFFINITIES
	      {
		// manufacture affinities for remote writes
		// acceptable local sources: SYSTEM, Z_COPY, REGDMA (bonus)
		// acceptable remote targets: SYSTEM, Z_COPY, GPU_FB, DISK, HDF, FILE, REGDMA (bonus)
		int bw = 6;
		int latency = 1000;
		bool rem_send, rem_recv, rem_reg;
		if(!allows_internode_copies(kind,
					    rem_send, rem_recv, rem_reg))
		  continue;

		// iterate over local memories and check their kinds
		Node *n = &(get_runtime()->nodes[gasnet_mynode()]);
		for(std::vector<MemoryImpl *>::const_iterator it = n->memories.begin();
		    it != n->memories.end();
		    ++it) {		    
		  Machine::MemoryMemoryAffinity mma;
		  mma.bandwidth = bw + (rem_reg ? 1 : 0);
		  mma.latency = latency - (rem_reg ? 100 : 0);

		  bool lcl_send, lcl_recv, lcl_reg;
		  if(!allows_internode_copies((*it)->get_kind(),
					      lcl_send, lcl_recv, lcl_reg))
		    continue;

		  if(lcl_reg) {
		    mma.bandwidth += 1;
		    mma.latency -= 100;
		  }

		  if(lcl_send && rem_recv) {
		    mma.m1 = (*it)->me;
		    mma.m2 = m;
		    log_annc.debug() << "adding inter-node affinity "
				     << mma.m1 << " -> " << mma.m2
				     << " (bw = " << mma.bandwidth << ", latency = " << mma.latency << ")";
		    add_mem_mem_affinity(mma, true /*lock held*/);
		    //mem_mem_affinities.push_back(mma);
		  }
		  if(rem_send && lcl_recv) {
		    mma.m1 = m;
		    mma.m2 = (*it)->me;
		    log_annc.debug() << "adding inter-node affinity "
				     << mma.m1 << " -> " << mma.m2
				     << " (bw = " << mma.bandwidth << ", latency = " << mma.latency << ")";
		    add_mem_mem_affinity(mma, true /*lock held*/);
		    //mem_mem_affinities.push_back(mma);
		  }
		}
	      }
#endif
	    }
	  }
	  break;

	case NODE_ANNOUNCE_IB_MEM:
	  {
	    ID id((ID::IDType)*cur++);
	    Memory m = id.convert<Memory>();
	    assert(id.memory.mem_idx < num_ib_memories);
	    Memory::Kind kind = (Memory::Kind)(*cur++);
	    size_t size = *cur++;
	    void *regbase = (void *)(*cur++);
	    log_annc.debug() << "adding ib memory " << m << " (kind = " << kind
			     << ", size = " << size << ", regbase = " << regbase << ")";
	    if(remote) {
	      RemoteMemory *mem = new RemoteMemory(m, size, kind, regbase);
              get_runtime()->nodes[id.memory.owner_node].ib_memories[id.memory.mem_idx] = mem;
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

	    add_proc_mem_affinity(pma, true /*lock held*/);
	    //proc_mem_affinities.push_back(pma);
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

	    add_mem_mem_affinity(mma, true /*lock held*/);
	    //mem_mem_affinities.push_back(mma);
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
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	mset.insert((*it).m);
      }
#else
      for(std::map<int, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
	  it != nodeinfos.end();
	  ++it)
	for(std::map<Memory, MachineMemInfo *>::const_iterator it2 = it->second->mems.begin();
	    it2 != it->second->mems.end();
	    ++it2)
	  mset.insert(it2->first);
#endif
    }

    void MachineImpl::get_all_processors(std::set<Processor>& pset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	pset.insert((*it).p);
      }
#else
      for(std::map<int, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
	  it != nodeinfos.end();
	  ++it)
	for(std::map<Processor, MachineProcInfo *>::const_iterator it2 = it->second->procs.begin();
	    it2 != it->second->procs.end();
	    ++it2)
	  pset.insert(it2->first);
#endif
    }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(int node) const
  {
    std::map<int, MachineNodeInfo *>::const_iterator it = nodeinfos.find(node);
    if(it != nodeinfos.end())
      return it->second;
    else
      return 0;
  }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(Processor p) const
  {
    return get_nodeinfo(ID(p).proc.owner_node);
  }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(Memory m) const
  {
    return get_nodeinfo(ID(m).memory.owner_node);
  }

    void MachineImpl::get_local_processors(std::set<Processor>& pset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	Processor p = (*it).p;
	if(ID(p).proc.owner_node == gasnet_mynode())
	  pset.insert(p);
      }
#else
      const MachineNodeInfo *mynode = get_nodeinfo(gasnet_mynode());
      assert(mynode != 0);
      for(std::map<Processor, MachineProcInfo *>::const_iterator it = mynode->procs.begin();
	  it != mynode->procs.end();
	  ++it)
	pset.insert(it->first);
#endif
    }

    void MachineImpl::get_local_processors_by_kind(std::set<Processor>& pset,
						   Processor::Kind kind) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	Processor p = (*it).p;
	if((ID(p).proc.owner_node == gasnet_mynode()) && (p.kind() == kind))
	  pset.insert(p);
      }
#else
      const MachineNodeInfo *mynode = get_nodeinfo(gasnet_mynode());
      assert(mynode != 0);
      std::map<Processor::Kind, std::map<Processor, MachineProcInfo *> >::const_iterator it = mynode->proc_by_kind.find(kind);
      if(it != mynode->proc_by_kind.end())
	for(std::map<Processor, MachineProcInfo *>::const_iterator it2 = it->second.begin();
	    it2 != it->second.end();
	    ++it2)
	  pset.insert(it2->first);
#endif
    }

    // Return the set of memories visible from a processor
    void MachineImpl::get_visible_memories(Processor p, std::set<Memory>& mset, bool local_only) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if((*it).p == p && (*it).m.capacity() > 0 &&
	   (!local_only || is_local_affinity(*it)))
	  mset.insert((*it).m);
      }
#else
      const MachineNodeInfo *ni = get_nodeinfo(p);
      if(!ni) return;
      std::map<Processor, MachineProcInfo *>::const_iterator it = ni->procs.find(p);
      if(it == ni->procs.end()) return;
      const std::map<Memory, Machine::ProcessorMemoryAffinity *>& pmas = (local_only ?
									    it->second->pmas.local :
									    it->second->pmas.all);
      for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 = pmas.begin();
	  it2 != pmas.end();
	  ++it2)
	mset.insert(it2->first);
#endif
    }

    // Return the set of memories visible from a memory
    void MachineImpl::get_visible_memories(Memory m, std::set<Memory>& mset,
					   bool local_only) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if(local_only && !is_local_affinity(*it)) continue;

	if((*it).m1 == m && (*it).m2.capacity() > 0)
	  mset.insert((*it).m2);
	
	if((*it).m2 == m && (*it).m1.capacity() > 0)
	  mset.insert((*it).m1);
      }
#else
      const MachineNodeInfo *ni = get_nodeinfo(m);
      if(!ni) return;
      std::map<Memory, MachineMemInfo *>::const_iterator it = ni->mems.find(m);
      if(it == ni->mems.end()) return;
      // handle both directions for now
      {
	const std::map<Memory, Machine::MemoryMemoryAffinity *>& mmas = (local_only ?
									   it->second->mmas_out.local :
  									   it->second->mmas_out.all);
	for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 = mmas.begin();
	    it2 != mmas.end();
	    ++it2)
	  mset.insert(it2->first);
      }
      {
	const std::map<Memory, Machine::MemoryMemoryAffinity *>& mmas = (local_only ?
									   it->second->mmas_in.local :
  									   it->second->mmas_in.all);
	for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 = mmas.begin();
	    it2 != mmas.end();
	    ++it2)
	  mset.insert(it2->first);
      }
#endif
    }

    // Return the set of processors which can all see a given memory
    void MachineImpl::get_shared_processors(Memory m, std::set<Processor>& pset,
					    bool local_only) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoHSLLock al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if(((*it).m == m) && (!local_only || is_local_affinity(*it)))
	  pset.insert((*it).p);
      }
#else
      const MachineNodeInfo *ni = get_nodeinfo(m);
      if(!ni) return;
      std::map<Memory, MachineMemInfo *>::const_iterator it = ni->mems.find(m);
      if(it == ni->mems.end()) return;
      const std::map<Processor, Machine::ProcessorMemoryAffinity *>& pmas = (local_only ?
									       it->second->pmas.local :
									       it->second->pmas.all);
      for(std::map<Processor, Machine::ProcessorMemoryAffinity *>::const_iterator it2 = pmas.begin();
	  it2 != pmas.end();
	    ++it2)
	pset.insert(it2->first);
#endif
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
					     Memory restrict_memory /*= Memory::NO_MEMORY*/,
					     bool local_only /*= true*/) const
    {
      int count = 0;

      {
	// TODO: consider using a reader/writer lock here instead
	AutoHSLLock al(mutex);
#ifdef USE_OLD_AFFINITIES
	for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	    it != proc_mem_affinities.end();
	    it++) {
	  if(restrict_proc.exists() && ((*it).p != restrict_proc)) continue;
	  if(restrict_memory.exists() && ((*it).m != restrict_memory)) continue;
	  if(local_only && !is_local_affinity(*it)) continue;
	  result.push_back(*it);
	  count++;
	}
#else
	if(restrict_proc.exists()) {
	  const MachineNodeInfo *np = get_nodeinfo(restrict_proc);
	  if(!np) return 0;
	  std::map<Processor, MachineProcInfo *>::const_iterator it = np->procs.find(restrict_proc);
	  if(it == np->procs.end()) return 0;
	  const MachineProcInfo *mpi = it->second;
	  const std::map<Memory, Machine::ProcessorMemoryAffinity *>& pmas = (local_only ? mpi->pmas.local : mpi->pmas.all);

	  if(restrict_memory.exists()) {
	    std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 = pmas.find(restrict_memory);
	    if(it2 != pmas.end()) {
	      result.push_back(*(it2->second));
	      count++;
	    }
	  } else {
	    for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 = pmas.begin();
		it2 != pmas.end();
		++it2) {
	      result.push_back(*(it2->second));
	      count++;
	    }
	  }
	} else {
	  if(restrict_memory.exists()) {
	    const MachineNodeInfo *nm = get_nodeinfo(restrict_memory);
	    if(!nm) return 0;
	    std::map<Memory, MachineMemInfo *>::const_iterator it = nm->mems.find(restrict_memory);
	    if(it == nm->mems.end()) return 0;
	    const MachineMemInfo *mmi = it->second;
	    const std::map<Processor, Machine::ProcessorMemoryAffinity *>& pmas = (local_only ? mmi->pmas.local : mmi->pmas.all);

	    for(std::map<Processor, Machine::ProcessorMemoryAffinity *>::const_iterator it2 = pmas.begin();
		it2 != pmas.end();
		++it2) {
	      result.push_back(*(it2->second));
	      count++;
	    }
	  } else {
	    // lookup of every single affinity - blech
	    for(std::map<int, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
		it != nodeinfos.end();
		++it)
	      for(std::map<Processor, MachineProcInfo *>::const_iterator it2 = it->second->procs.begin();
		  it2 != it->second->procs.end();
		  ++it2) {
		const MachineProcInfo *mpi = it2->second;
		const std::map<Memory, Machine::ProcessorMemoryAffinity *>& pmas = (local_only ? mpi->pmas.local : mpi->pmas.all);

		for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 = pmas.begin();
		    it2 != pmas.end();
		    ++it2) {
		  result.push_back(*(it2->second));
		  count++;
		}
	      }
	  }
	}
#endif
      }

      return count;
    }

    int MachineImpl::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
					  Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
					  Memory restrict_mem2 /*= Memory::NO_MEMORY*/,
					  bool local_only /*= true*/) const
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
#ifdef USE_OLD_AFFINITIES
	for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	    it != mem_mem_affinities.end();
	    it++) {
	  if(restrict_mem1.exists() && 
	     ((*it).m1 != restrict_mem1)) continue;
	  if(restrict_mem2.exists() && 
	     ((*it).m2 != restrict_mem2)) continue;
	  if(local_only && !is_local_affinity(*it)) continue;
	  result.push_back(*it);
	  count++;
	}
#else
	if(restrict_mem1.exists()) {
	  const MachineNodeInfo *nm1 = get_nodeinfo(restrict_mem1);
	  if(!nm1) return 0;
	  std::map<Memory, MachineMemInfo *>::const_iterator it = nm1->mems.find(restrict_mem1);
	  if(it == nm1->mems.end()) return 0;
	  const MachineMemInfo *mmi = it->second;
	  const std::map<Memory, Machine::MemoryMemoryAffinity *>& mmas = (local_only ? mmi->mmas_out.local : mmi->mmas_out.all);

	  if(restrict_mem2.exists()) {
	    std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 = mmas.find(restrict_mem2);
	    if(it2 != mmas.end()) {
	      result.push_back(*(it2->second));
	      count++;
	    }
	  } else {
	    for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 = mmas.begin();
		it2 != mmas.end();
		++it2) {
	      result.push_back(*(it2->second));
	      count++;
	    }
	  }
	} else {
	  if(restrict_mem2.exists()) {
	    const MachineNodeInfo *nm2 = get_nodeinfo(restrict_mem2);
	    if(!nm2) return 0;
	    std::map<Memory, MachineMemInfo *>::const_iterator it = nm2->mems.find(restrict_mem2);
	    if(it == nm2->mems.end()) return 0;
	    const MachineMemInfo *mmi = it->second;
	    const std::map<Memory, Machine::MemoryMemoryAffinity *>& mmas = (local_only ? mmi->mmas_in.local : mmi->mmas_in.all);

	    for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 = mmas.begin();
		it2 != mmas.end();
		++it2) {
	      result.push_back(*(it2->second));
	      count++;
	    }
	  } else {
	    // lookup of every single affinity - blech
	    for(std::map<int, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
		it != nodeinfos.end();
		++it)
	      for(std::map<Memory, MachineMemInfo *>::const_iterator it2 = it->second->mems.begin();
		  it2 != it->second->mems.end();
		  ++it2) {
		const MachineMemInfo *mmi = it2->second;
		const std::map<Memory, Machine::MemoryMemoryAffinity *>& mmas = (local_only ? mmi->mmas_out.local : mmi->mmas_out.all);

		for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 = mmas.begin();
		    it2 != mmas.end();
		    ++it2) {
		  result.push_back(*(it2->second));
		  count++;
		}
	      }
	  }
	}
#endif
      }

      return count;
    }

  void MachineImpl::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma,
					  bool lock_held /*= false*/)
  {
    if(!lock_held) mutex.lock();

    proc_mem_affinities.push_back(pma);

    int np = ID(pma.p).proc.owner_node;
    int mp = ID(pma.m).memory.owner_node;
    {
      MachineNodeInfo *& ptr = nodeinfos[np];
      if(!ptr) ptr = new MachineNodeInfo(np);
      ptr->add_processor(pma.p);
      if(np == mp)
	ptr->add_memory(pma.m);
      ptr->add_proc_mem_affinity(pma);
    }
    if(np != mp) {
      MachineNodeInfo *& ptr = nodeinfos[mp];
      if(!ptr) ptr = new MachineNodeInfo(mp);
      ptr->add_memory(pma.m);
      ptr->add_proc_mem_affinity(pma);
    }

    if(!lock_held) mutex.unlock();
  }

  void MachineImpl::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma,
					 bool lock_held /*= false*/)
  {
    if(!lock_held) mutex.lock();

    mem_mem_affinities.push_back(mma);

    int m1p = ID(mma.m1).memory.owner_node;
    int m2p = ID(mma.m2).memory.owner_node;
    {
      MachineNodeInfo *& ptr = nodeinfos[m1p];
      if(!ptr) ptr = new MachineNodeInfo(m1p);
      ptr->add_memory(mma.m1);
      if(m1p == m2p)
	ptr->add_memory(mma.m2);
      ptr->add_mem_mem_affinity(mma);
    }
    if(m1p != m2p) {
      MachineNodeInfo *& ptr = nodeinfos[m2p];
      if(!ptr) ptr = new MachineNodeInfo(m2p);
      ptr->add_memory(mma.m2);
      ptr->add_mem_mem_affinity(mma);
    }

    if(!lock_held) mutex.unlock();
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
    ((ProcessorQueryImpl *)impl)->restrict_to_kind(kind);
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
    ((MemoryQueryImpl *)impl)->restrict_to_kind(kind);
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
  // class ProcessorHasAffinityPredicate
  //

  ProcessorHasAffinityPredicate::ProcessorHasAffinityPredicate(Memory _memory,
							       unsigned _min_bandwidth,
							       unsigned _max_latency)
    : memory(_memory)
    , min_bandwidth(_min_bandwidth)
    , max_latency(_max_latency)
  {}

  ProcQueryPredicate *ProcessorHasAffinityPredicate::clone(void) const
  {
    return new ProcessorHasAffinityPredicate(memory, min_bandwidth, max_latency);
  }

  bool ProcessorHasAffinityPredicate::matches_predicate(MachineImpl *machine, Processor thing,
							const MachineProcInfo *info) const
  {
#ifdef USE_OLD_AFFINITIES
    Machine::AffinityDetails details;
    if(!machine->has_affinity(thing, memory, &details)) return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (details.latency > max_latency)) return false;
    return true;
#else
    assert(info != 0);
    std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it = info->pmas.all.find(memory);
    if(it == info->pmas.all.end()) return false;
    if((min_bandwidth != 0) && (it->second->bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (it->second->latency > max_latency)) return false;
    return true;
#endif
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

  ProcQueryPredicate *ProcessorBestAffinityPredicate::clone(void) const
  {
    return new ProcessorBestAffinityPredicate(memory, bandwidth_weight, latency_weight);
  }

  bool ProcessorBestAffinityPredicate::matches_predicate(MachineImpl *machine, Processor thing,
							 const MachineProcInfo *info) const
  {
#ifndef USE_OLD_AFFINITIES
    if((bandwidth_weight == 1) && (latency_weight == 0)) {
      assert(info != 0);
      std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it = info->pmas.best.find(memory);
      return(it != info->pmas.best.end());
    }
#endif

    // old way
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
    , is_restricted_node(false)
    , is_restricted_kind(false)
  {}
     
  ProcessorQueryImpl::ProcessorQueryImpl(const ProcessorQueryImpl& copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted_node(copy_from.is_restricted_node)
    , restricted_node_id(copy_from.restricted_node_id)
    , is_restricted_kind(copy_from.is_restricted_kind)
    , restricted_kind(copy_from.restricted_kind)
  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<ProcQueryPredicate *>::const_iterator it = copy_from.predicates.begin();
	it != copy_from.predicates.end();
	it++)
      predicates.push_back((*it)->clone());
  }

  ProcessorQueryImpl::~ProcessorQueryImpl(void)
  {
    assert(references == 0);
    for(std::vector<ProcQueryPredicate *>::iterator it = predicates.begin();
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
    if(is_restricted_node && (new_node_id != restricted_node_id)) {
      restricted_node_id = -1;
    } else {
      is_restricted_node = true;
      restricted_node_id = new_node_id;
    }
  }

  void ProcessorQueryImpl::restrict_to_kind(Processor::Kind new_kind)
  {
    // attempts to restrict to two different kind results in no possible match
    // (use node restriction to enforce this)
    if(is_restricted_kind && (new_kind != restricted_kind)) {
      is_restricted_node = true;
      restricted_node_id = -1;
    } else {
      is_restricted_kind = true;
      restricted_kind = new_kind;
    }
  }

  void ProcessorQueryImpl::add_predicate(ProcQueryPredicate *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
  }

  Processor ProcessorQueryImpl::first_match(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted_node && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (p.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, p);
	if(ok && (!lowest.exists() || (p.id < lowest.id)))
	  lowest = p;
      }
    }
    return lowest;
#else
    std::map<int, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Processor, MachineProcInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Processor::Kind, std::map<Processor, MachineProcInfo *> >::const_iterator it2 = it->second->proc_by_kind.find(restricted_kind);
	if(it2 != it->second->proc_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->procs);

      if(plist) {
	std::map<Processor, MachineProcInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok)
	    return it2->first;

	  // try next processor (if it exists)
	  ++it2;
	}
      }

      // try the next node (if it exists)
      ++it;
    }
    return Processor::NO_PROC;
#endif
  }

  Processor ProcessorQueryImpl::next_match(Processor after) const
  {
    if(!after.exists()) return Processor::NO_PROC;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(p.id <= after.id) continue;
	if(is_restricted_node && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (p.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, p);
	if(ok && (!lowest.exists() || (p.id < lowest.id)))
	  lowest = p;
      }
    }
    return lowest;
#else
    std::map<int, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).proc.owner_node);
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Processor, MachineProcInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Processor::Kind, std::map<Processor, MachineProcInfo *> >::const_iterator it2 = it->second->proc_by_kind.find(restricted_kind);
	if(it2 != it->second->proc_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->procs);

      if(plist) {
        std::map<Processor, MachineProcInfo *>::const_iterator it2;
	// same node?  if so, skip past ones we've done
	if(it->first == ID(after).proc.owner_node)
	  it2 = plist->upper_bound(after);
	else
	  it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok)
	    return it2->first;

	  // try next processor (if it exists)
	  ++it2;
	}
      }

      // try the next node (if it exists)
      ++it;
    }
    return Processor::NO_PROC;
#endif
  }

  size_t ProcessorQueryImpl::count_matches(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return 0;
    std::set<Processor> pset;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted_node && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (p.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, p);
	if(ok)
	  pset.insert((*it).p);
      }
    }
    return pset.size();
#else
    size_t count = 0;
    std::map<int, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Processor, MachineProcInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Processor::Kind, std::map<Processor, MachineProcInfo *> >::const_iterator it2 = it->second->proc_by_kind.find(restricted_kind);
	if(it2 != it->second->proc_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->procs);

      if(plist) {
	std::map<Processor, MachineProcInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok)
	    count += 1;

	  // continue to next processor (if it exists)
	  ++it2;
	}
      }

      // continue to the next node (if it exists)
      ++it;
    }
    return count;
#endif
  }

  Processor ProcessorQueryImpl::random_match(void) const
  {
    Processor chosen = Processor::NO_PROC;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Processor::NO_PROC;
    int count = 0;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted_node && (ID(p).proc.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (p.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
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
#else
    size_t count = 0;
    std::map<int, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Processor, MachineProcInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Processor::Kind, std::map<Processor, MachineProcInfo *> >::const_iterator it2 = it->second->proc_by_kind.find(restricted_kind);
	if(it2 != it->second->proc_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->procs);

      if(plist) {
	std::map<Processor, MachineProcInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok) {
	    count++;
	    if((count == 1) || ((lrand48() % count) == 0))
	      chosen = it2->first;
	  }

	  // continue to next processor (if it exists)
	  ++it2;
	}
      }

      // continue to the next node (if it exists)
      ++it;
    }
#endif
    return chosen;
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

  MemoryQueryPredicate *MemoryHasProcAffinityPredicate::clone(void) const
  {
    return new MemoryHasProcAffinityPredicate(proc, min_bandwidth, max_latency);
  }

  bool MemoryHasProcAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing,
					      const MachineMemInfo *info) const
  {
#ifdef USE_OLD_AFFINITIES
    Machine::AffinityDetails details;
    if(!machine->has_affinity(proc, thing, &details)) return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (details.latency > max_latency)) return false;
    return true;
#else
    assert(info != 0);
    std::map<Processor, Machine::ProcessorMemoryAffinity *>::const_iterator it = info->pmas.all.find(proc);
    if(it == info->pmas.all.end()) return false;
    if((min_bandwidth != 0) && (it->second->bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (it->second->latency > max_latency)) return false;
    return true;
#endif
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

  MemoryQueryPredicate *MemoryHasMemAffinityPredicate::clone(void) const
  {
    return new MemoryHasMemAffinityPredicate(memory, min_bandwidth, max_latency);
  }

  bool MemoryHasMemAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing,
							const MachineMemInfo *info) const
  {
#ifdef USE_OLD_AFFINITIES
    Machine::AffinityDetails details;
    if(!machine->has_affinity(memory, thing, &details)) return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (details.latency > max_latency)) return false;
    return true;
#else
    assert(info != 0);
    std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it = info->mmas_out.all.find(memory);
    if(it == info->mmas_out.all.end()) return false;
    if((min_bandwidth != 0) && (it->second->bandwidth < min_bandwidth)) return false;
    if((max_latency != 0) && (it->second->latency > max_latency)) return false;
    return true;
#endif
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

  MemoryQueryPredicate *MemoryBestProcAffinityPredicate::clone(void) const
  {
    return new MemoryBestProcAffinityPredicate(proc, bandwidth_weight, latency_weight);
  }

  bool MemoryBestProcAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing,
					      const MachineMemInfo *info) const
  {
#ifndef USE_OLD_AFFINITIES
    if((bandwidth_weight == 1) && (latency_weight == 0)) {
      assert(info != 0);
      std::map<Processor, Machine::ProcessorMemoryAffinity *>::const_iterator it = info->pmas.best.find(proc);
      return(it != info->pmas.best.end());
    }
#endif

    // old way
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

  MemoryQueryPredicate *MemoryBestMemAffinityPredicate::clone(void) const
  {
    return new MemoryBestMemAffinityPredicate(memory, bandwidth_weight, latency_weight);
  }

  bool MemoryBestMemAffinityPredicate::matches_predicate(MachineImpl *machine, Memory thing,
					      const MachineMemInfo *info) const
  {
#ifndef USE_OLD_AFFINITIES
    if((bandwidth_weight == 1) && (latency_weight == 0)) {
      assert(info != 0);
      std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it = info->mmas_out.best.find(memory);
      return(it != info->mmas_out.best.end());
    }
#endif

    // old way
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
    , is_restricted_node(false)
    , is_restricted_kind(false)
  {}
     
  MemoryQueryImpl::MemoryQueryImpl(const MemoryQueryImpl& copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted_node(copy_from.is_restricted_node)
    , restricted_node_id(copy_from.restricted_node_id)
    , is_restricted_kind(copy_from.is_restricted_kind)
    , restricted_kind(copy_from.restricted_kind)
  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<MemoryQueryPredicate *>::const_iterator it = copy_from.predicates.begin();
	it != copy_from.predicates.end();
	it++)
      predicates.push_back((*it)->clone());
  }

  MemoryQueryImpl::~MemoryQueryImpl(void)
  {
    assert(references == 0);
    for(std::vector<MemoryQueryPredicate *>::iterator it = predicates.begin();
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
    if(is_restricted_node && (new_node_id != restricted_node_id)) {
      restricted_node_id = -1;
    } else {
      is_restricted_node = true;
      restricted_node_id = new_node_id;
    }
  }

  void MemoryQueryImpl::restrict_to_kind(Memory::Kind new_kind)
  {
    // attempts to restrict to two different kind results in no possible match
    // (use node restriction to enforce this)
    if(is_restricted_kind && (new_kind != restricted_kind)) {
      is_restricted_node = true;
      restricted_node_id = -1;
    } else {
      is_restricted_kind = true;
      restricted_kind = new_kind;
    }
  }

  void MemoryQueryImpl::add_predicate(MemoryQueryPredicate *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
  }

  Memory MemoryQueryImpl::first_match(void) const
  {
#if USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Memory::NO_MEMORY;
    Memory lowest = Memory::NO_MEMORY;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted_node && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, m);
	if(ok && (!lowest.exists() || (m.id < lowest.id)))
	  lowest = m;
      }
    }
    return lowest;
#else
    std::map<int, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Memory::Kind, std::map<Memory, MachineMemInfo *> >::const_iterator it2 = it->second->mem_by_kind.find(restricted_kind);
	if(it2 != it->second->mem_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->mems);

      if(plist) {
	std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<MemoryQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok)
	    return it2->first;

	  // try next memory (if it exists)
	  ++it2;
	}
      }

      // try the next node (if it exists)
      ++it;
    }
    return Memory::NO_MEMORY;
#endif
  }

  Memory MemoryQueryImpl::next_match(Memory after) const
  {
    if(!after.exists()) return Memory::NO_MEMORY;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Memory::NO_MEMORY;
    Memory lowest = Memory::NO_MEMORY;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(m.id <= after.id) continue;
	if(is_restricted_node && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, m);
	if(ok && (!lowest.exists() || (m.id < lowest.id)))
	  lowest = m;
      }
    }
    return lowest;
#else
    std::map<int, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).memory.owner_node);
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Memory::Kind, std::map<Memory, MachineMemInfo *> >::const_iterator it2 = it->second->mem_by_kind.find(restricted_kind);
	if(it2 != it->second->mem_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->mems);

      if(plist) {
        std::map<Memory, MachineMemInfo *>::const_iterator it2;
	// same node?  if so, skip past ones we've done
	if(it->first == ID(after).memory.owner_node)
	  it2 = plist->upper_bound(after);
	else
	  it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<MemoryQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok)
	    return it2->first;

	  // try next memory (if it exists)
	  ++it2;
	}
      }

      // try the next node (if it exists)
      ++it;
    }
    return Memory::NO_MEMORY;
#endif
  }

  size_t MemoryQueryImpl::count_matches(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return 0;
    std::set<Memory> pset;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted_node && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
	    ok && (it2 != predicates.end());
	    it2++)
	  ok &= (*it2)->matches_predicate(machine, m);
	if(ok)
	  pset.insert((*it).m);
      }
    }
    return pset.size();
#else
    size_t count = 0;
    std::map<int, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Memory::Kind, std::map<Memory, MachineMemInfo *> >::const_iterator it2 = it->second->mem_by_kind.find(restricted_kind);
	if(it2 != it->second->mem_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->mems);

      if(plist) {
	std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<MemoryQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok)
	    count += 1;

	  // continue to next memory (if it exists)
	  ++it2;
	}
      }

      // continue to the next node (if it exists)
      ++it;
    }
    return count;
#endif
  }

  Memory MemoryQueryImpl::random_match(void) const
  {
    Memory chosen = Memory::NO_MEMORY;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Memory::NO_MEMORY;
    int count = 0;
    {
      // problem with nested locks here...
      //AutoHSLLock al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted_node && (ID(m).memory.owner_node != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
	bool ok = true;
	for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
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
#else
    size_t count = 0;
    std::map<int, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
	std::map<Memory::Kind, std::map<Memory, MachineMemInfo *> >::const_iterator it2 = it->second->mem_by_kind.find(restricted_kind);
	if(it2 != it->second->mem_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->mems);

      if(plist) {
	std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
	  bool ok = true;
	  for(std::vector<MemoryQueryPredicate *>::const_iterator it3 = predicates.begin();
	      ok && (it3 != predicates.end());
	      it3++)
	    ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
	  if(ok) {
	    count++;
	    if((count == 1) || ((lrand48() % count) == 0))
	      chosen = it2->first;
	  }

	  // continue to next memory (if it exists)
	  ++it2;
	}
      }

      // continue to the next node (if it exists)
      ++it;
    }
#endif
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
    n->ib_memories.resize(args.num_ib_memories);

    // do the parsing of this data inside a mutex because it touches common
    //  data structures
    {
      get_machine()->parse_node_announce_data(args.node_id, args.num_procs,
					      args.num_memories, args.num_ib_memories,
					      data, datalen, true);

      __sync_fetch_and_add(&announcements_received, 1);
    }
  }

  /*static*/ void NodeAnnounceMessage::send_request(gasnet_node_t target,
						    unsigned num_procs,
						    unsigned num_memories,
						    unsigned num_ib_memories,
						    const void *data,
						    size_t datalen,
						    int payload_mode)
  {
    RequestArgs args;

    args.node_id = gasnet_mynode();
    args.num_procs = num_procs;
    args.num_memories = num_memories;
    args.num_ib_memories = num_ib_memories;
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
