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

#include "realm/machine_impl.h"

#include "realm/logging.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/runtime_impl.h"

#include "realm/activemsg.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

#ifdef REALM_ON_WINDOWS
// TODO: clean up query cache code and renable!
#define REALM_DISABLE_MACHINE_QUERY_CACHE

static int lrand48() { return rand(); }
#endif

TYPE_IS_SERIALIZABLE(Realm::NodeAnnounceTag);
TYPE_IS_SERIALIZABLE(Realm::Memory);
TYPE_IS_SERIALIZABLE(Realm::Memory::Kind);
TYPE_IS_SERIALIZABLE(Realm::Channel::SupportedPath);
TYPE_IS_SERIALIZABLE(Realm::XferDesKind);

namespace Realm {

  Logger log_machine("machine");
  Logger log_annc("announce");
  Logger log_query("query");

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
    return ID(pma.p).proc_owner_node() == ID(pma.m).memory_owner_node();
  }

  static inline bool is_local_affinity(const Machine::MemoryMemoryAffinity& mma)
  {
    return ID(mma.m1).memory_owner_node() == ID(mma.m2).memory_owner_node();
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
    assert(node == NodeID(ID(p).proc_owner_node()));
    MachineProcInfo *& ptr = procs[p];
    // TODO: see if anything changed?
    if(ptr != 0)
      return false;

    ptr = new MachineProcInfo(p);

    return true;
  }
  
  bool MachineNodeInfo::add_memory(Memory m)
  {
    assert(node == NodeID(ID(m).memory_owner_node()));
    MachineMemInfo *& ptr = mems[m];
    // TODO: see if anything changed?
    if(ptr != 0)
      return false;

    ptr = new MachineMemInfo(m);

    return true;
  }

  bool MachineNodeInfo::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma)
  {
    bool changed = false;

    if(NodeID(ID(pma.p).proc_owner_node()) == node) {
      MachineProcInfo *mpi = procs[pma.p];
      assert(mpi != 0);
      if(mpi->add_proc_mem_affinity(pma))
	changed = true;
    }

    if(NodeID(ID(pma.m).memory_owner_node()) == node) {
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

    if(NodeID(ID(mma.m1).memory_owner_node()) == node) {
      MachineMemInfo *mmi = mems[mma.m1];
      assert(mmi != 0);
      if(mmi->add_mem_mem_affinity(mma))
	changed = true;
    }

    if(NodeID(ID(mma.m2).memory_owner_node()) == node) {
      MachineMemInfo *mmi = mems[mma.m2];
      assert(mmi != 0);
      if(mmi->add_mem_mem_affinity(mma))
	changed = true;
    }

    return changed;
  }

  void MachineNodeInfo::update_kind_maps()
  {
    proc_by_kind.clear();
    for(std::map<Processor, MachineProcInfo *>::const_iterator it = procs.begin();
        it != procs.end();
        ++it)
      proc_by_kind[it->first.kind()][it->first] = it->second;

    mem_by_kind.clear();
    for(std::map<Memory, MachineMemInfo *>::const_iterator it = mems.begin();
        it != mems.end();
        ++it)
      mem_by_kind[it->first.kind()][it->first] = it->second;
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
      return Network::max_node_id + 1;
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


  namespace Config {
    bool use_machine_query_cache = true;
  };

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

    void MachineImpl::parse_node_announce_data(int node_id,
					       const void *args, size_t arglen,
					       bool remote)
    {
      AutoLock<> al(mutex);

      assert(node_id <= Network::max_node_id);
      Node& n = get_runtime()->nodes[node_id];

      Serialization::FixedBufferDeserializer fbd(args, arglen);
      bool ok = true;
      while(ok && (fbd.bytes_left() > 0)) {
	NodeAnnounceTag tag;
	if(!(fbd >> tag)) {
	  log_annc.fatal() << "unexpected end of input";
	  assert(0);
	}

	switch(tag) {
	case NODE_ANNOUNCE_PROC:
	  {
	    Processor p;
	    Processor::Kind kind;
	    int num_cores;
	    ok = (ok &&
		  (fbd >> p) &&
		  (fbd >> kind) &&
		  (fbd >> num_cores));
	    if(ok) {
	      assert(NodeID(ID(p).proc_owner_node()) == node_id);
	      log_annc.debug() << "adding proc " << p << " (kind = " << kind
			       << " num_cores = " << num_cores << ")";
	      if(remote) {
		RemoteProcessor *proc = new RemoteProcessor(p, kind, num_cores);
                if(n.processors.size() <= ID(p).proc_proc_idx())
                  n.processors.resize(ID(p).proc_proc_idx() + 1, 0);
		n.processors[ID(p).proc_proc_idx()] = proc;
	      }
	    }
	  }
	  break;

	case NODE_ANNOUNCE_MEM:
	  {
	    Memory m;
	    Memory::Kind kind;
	    size_t size;
	    bool has_rdma_info = false;
	    ByteArray rdma_info;
	    ok = (ok &&
		  (fbd >> m) &&
		  (fbd >> kind) &&
		  (fbd >> size) &&
		  (fbd >> has_rdma_info));
	    if(has_rdma_info)
	      ok = ok && (fbd >> rdma_info);
	    if(ok) {
	      assert(NodeID(ID(m).memory_owner_node()) == node_id);
	      log_annc.debug() << "adding memory " << m << " (kind = " << kind
			       << ", size = " << size << ", has_rdma = " << has_rdma_info << ")";
	      if(remote) {
		MemoryImpl *mem;
		if(has_rdma_info) {
		  mem = Network::get_network(node_id)->create_remote_memory(m,
									    size,
									    kind,
									    rdma_info);
		} else {
		  mem = new RemoteMemory(m, size, kind);
		}
                if(n.memories.size() >= ID(m).memory_mem_idx())
                  n.memories.resize(ID(m).memory_mem_idx() + 1, 0);
		n.memories[ID(m).memory_mem_idx()] = mem;

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
		  Node *mynode = &(get_runtime()->nodes[Network::my_node_id]);
		  for(std::vector<MemoryImpl *>::const_iterator it = mynode->memories.begin();
		      it != mynode->memories.end();
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
	  }
	  break;

	case NODE_ANNOUNCE_IB_MEM:
	  {
	    Memory m;
	    Memory::Kind kind;
	    size_t size;
	    bool has_rdma_info = false;
	    ByteArray rdma_info;
	    ok = (ok &&
		  (fbd >> m) &&
		  (fbd >> kind) &&
		  (fbd >> size) &&
		  (fbd >> has_rdma_info));
	    if(has_rdma_info)
	      ok = ok && (fbd >> rdma_info);
	    if(ok) {
	      assert(NodeID(ID(m).memory_owner_node()) == node_id);
	      log_annc.debug() << "adding ib memory " << m << " (kind = " << kind
			       << ", size = " << size << ", has_rdma = " << has_rdma_info << ")";
	      if(remote) {
		IBMemory *ibmem;
		if(has_rdma_info) {
		  ibmem = Network::get_network(node_id)->create_remote_ib_memory(m,
										 size,
										 kind,
										 rdma_info);
		} else {
		  ibmem = new IBMemory(m, size, MemoryImpl::MKIND_REMOTE, kind, 0, 0);
		}

                if(n.ib_memories.size() >= ID(m).memory_mem_idx())
                  n.ib_memories.resize(ID(m).memory_mem_idx() + 1, 0);
		n.ib_memories[ID(m).memory_mem_idx()] = ibmem;
	      }
	    }
	  }
	  break;

	case NODE_ANNOUNCE_PMA:
	  {
	    Machine::ProcessorMemoryAffinity pma;
	    ok = (ok &&
		  (fbd >> pma.p) &&
		  (fbd >> pma.m) &&
		  (fbd >> pma.bandwidth) &&
		  (fbd >> pma.latency));
	    if(ok) {
	      log_annc.debug() << "adding affinity " << pma.p << " -> " << pma.m
			       << " (bw = " << pma.bandwidth << ", latency = " << pma.latency << ")";

	      add_proc_mem_affinity(pma, true /*lock held*/);
	      //proc_mem_affinities.push_back(pma);
	    }
	  }
	  break;

	case NODE_ANNOUNCE_MMA:
	  {
	    Machine::MemoryMemoryAffinity mma;
	    ok = (ok &&
		  (fbd >> mma.m1) &&
		  (fbd >> mma.m2) &&
		  (fbd >> mma.bandwidth) &&
		  (fbd >> mma.latency));
	    if(ok) {
	      log_annc.debug() << "adding affinity " << mma.m1 << " <-> " << mma.m2
			       << " (bw = " << mma.bandwidth << ", latency = " << mma.latency << ")";

	      add_mem_mem_affinity(mma, true /*lock held*/);
	      //mem_mem_affinities.push_back(mma);
	    }
	  }
	  break;

	case NODE_ANNOUNCE_DMA_CHANNEL:
	  {
	    RemoteChannelInfo *rci = RemoteChannelInfo::deserialize_new(fbd);
	    if(rci) {
              RemoteChannel *rc = rci->create_remote_channel();
              delete rci;

	      log_annc.debug() << "adding channel: " << *rc;
	      assert(rc->node == node_id);
	      if(remote)
		get_runtime()->add_dma_channel(rc);
	      else
		delete rc; // don't actually need it
	    }
	  }
	  break;

	default:
	  log_annc.fatal() << "unknown tag: " << tag;
	  assert(0);
	}
      }

      assert(ok && (fbd.bytes_left() == 0));
    }

    void MachineImpl::get_all_memories(std::set<Memory>& mset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoLock<> al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	mset.insert((*it).m);
      }
#else
      for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
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
      AutoLock<> al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	pset.insert((*it).p);
      }
#else
      for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
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
    std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.find(node);
    if(it != nodeinfos.end())
      return it->second;
    else
      return 0;
  }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(Processor p) const
  {
    return get_nodeinfo(ID(p).proc_owner_node());
  }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(Memory m) const
  {
    return get_nodeinfo(ID(m).memory_owner_node());
  }

    void MachineImpl::get_local_processors(std::set<Processor>& pset) const
    {
      // TODO: consider using a reader/writer lock here instead
      AutoLock<> al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	Processor p = (*it).p;
	if(ID(p).proc_owner_node() == Network::my_node_id)
	  pset.insert(p);
      }
#else
      const MachineNodeInfo *mynode = get_nodeinfo(Network::my_node_id);
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
      AutoLock<> al(mutex);
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	Processor p = (*it).p;
	if((ID(p).proc_owner_node() == Network::my_node_id) && (p.kind() == kind))
	  pset.insert(p);
      }
#else
      const MachineNodeInfo *mynode = get_nodeinfo(Network::my_node_id);
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
      AutoLock<> al(mutex);
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
      AutoLock<> al(mutex);
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
      AutoLock<> al(mutex);
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
      AutoLock<> al(mutex);
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
      AutoLock<> al(mutex);
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
	AutoLock<> al(mutex);
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
	    for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
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
	AutoLock<> al(mutex);
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
	    for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
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

    int np = ID(pma.p).proc_owner_node();
    int mp = ID(pma.m).memory_owner_node();
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
    invalidate_query_caches();
    if(!lock_held) mutex.unlock();
  }

  void MachineImpl::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma,
					 bool lock_held /*= false*/)
  {
    if(!lock_held) mutex.lock();

    mem_mem_affinities.push_back(mma);

    int m1p = ID(mma.m1).memory_owner_node();
    int m2p = ID(mma.m2).memory_owner_node();
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
    invalidate_query_caches();
    if(!lock_held) mutex.unlock();
  }

    void MachineImpl::add_subscription(Machine::MachineUpdateSubscriber *subscriber)
    {
      AutoLock<> al(mutex);
      subscribers.insert(subscriber);
    }

    void MachineImpl::remove_subscription(Machine::MachineUpdateSubscriber *subscriber)
    {
      AutoLock<> al(mutex);
      subscribers.erase(subscriber);
    }

    void MachineImpl::invalidate_query_caches()
    {
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
      while (!__sync_bool_compare_and_swap(&MemoryQueryImpl::init,0,1))
        continue;
      __sync_fetch_and_add(&MemoryQueryImpl::cache_invalid_count,1);
      (void)__sync_val_compare_and_swap(&MemoryQueryImpl::global_valid_cache,1,0);
      __sync_sub_and_fetch(&MemoryQueryImpl::init,1);
      log_query.debug("invalidate_query_caches MemoryQueryImpl::cache_invalid_count = %d \n", MemoryQueryImpl::cache_invalid_count);
      while (!__sync_bool_compare_and_swap(&ProcessorQueryImpl::init,0,1))
        continue;
      __sync_fetch_and_add(&ProcessorQueryImpl::cache_invalid_count,1);
      (void)__sync_val_compare_and_swap(&ProcessorQueryImpl::global_valid_cache,1,0);
      __sync_sub_and_fetch(&ProcessorQueryImpl::init,1);
       log_query.debug("invalidate_query_caches complete ProcessorQueryImpl::cache_invalid_count = %d \n", ProcessorQueryImpl::cache_invalid_count);
#endif
    }

  void cleanup_query_caches()
  {
    MemoryQueryImpl::_mem_cache.clear();
    ProcessorQueryImpl::_proc_cache.clear();
    ProcessorQueryImpl::_proc_cache_affinity.clear();
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
    ((ProcessorQueryImpl *)impl)->restrict_to_node(Network::my_node_id);
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::same_address_space_as(Processor p)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(ID(p).proc_owner_node());
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::same_address_space_as(Memory m)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(ID(m).proc_owner_node());
    return *this;
  }
      
  Machine::ProcessorQuery& Machine::ProcessorQuery::has_affinity_to(Memory m,
								    unsigned min_bandwidth /*= 0*/,
								    unsigned max_latency /*= 0*/)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    // we have a cached version of the map, record it
    ((ProcessorQueryImpl *)impl)->add_predicate(new ProcessorHasAffinityPredicate(m, min_bandwidth, max_latency));
    // query type can use one of the cached maps
    if (!min_bandwidth && !max_latency)
      ((ProcessorQueryImpl*)impl)->set_cached_mem(m);
    else
      ((ProcessorQueryImpl*)impl)->reset_cached_mem();
    return *this;
  }

  Machine::ProcessorQuery& Machine::ProcessorQuery::best_affinity_to(Memory m,
								     int bandwidth_weight /*= 1*/,
								     int latency_weight /*= 0*/)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->add_predicate(new ProcessorBestAffinityPredicate(m, bandwidth_weight, latency_weight));
    ((ProcessorQueryImpl *)impl)->reset_cached_mem();
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
    if (Config::use_machine_query_cache)
      return ((ProcessorQueryImpl *)impl)->cache_next(after);
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
    ((MemoryQueryImpl *)impl)->restrict_to_node(Network::my_node_id);
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::same_address_space_as(Processor p)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(ID(p).proc_owner_node());
    return *this;
  }

  Machine::MemoryQuery& Machine::MemoryQuery::same_address_space_as(Memory m)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(ID(m).memory_owner_node());
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

  Machine::MemoryQuery& Machine::MemoryQuery::has_capacity(size_t min_bytes)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_by_capacity(min_bytes);
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
    if (Config::use_machine_query_cache)
     return ((MemoryQueryImpl *)impl)->cache_next(after);
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
  unsigned int ProcessorQueryImpl::init=0;
  unsigned int ProcessorQueryImpl::cache_invalid_count=0;
  bool ProcessorQueryImpl::global_valid_cache=true;


  std::map<Processor::Kind, std::vector<Processor> > ProcessorQueryImpl::_proc_cache;
  std::map<Processor::Kind, std::map<Memory, std::vector<Processor> > > ProcessorQueryImpl::_proc_cache_affinity;

  ProcessorQueryImpl::ProcessorQueryImpl(const Machine& _machine)
    : references(1)
    , machine((MachineImpl *)_machine.impl)
    , is_restricted_node(false)
    , is_restricted_kind(false)
    , cached_mem(Memory::NO_MEMORY)
    , is_cached_mem(false)
    , shared_cached_list(false)
    , valid_cache(false)
    , cur_cached_list(NULL)
    , invalid_count(cache_invalid_count)

  {}

  ProcessorQueryImpl::ProcessorQueryImpl(const ProcessorQueryImpl& copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted_node(copy_from.is_restricted_node)
    , restricted_node_id(copy_from.restricted_node_id)
    , is_restricted_kind(copy_from.is_restricted_kind)
    , restricted_kind(copy_from.restricted_kind)
    , cached_mem(copy_from.cached_mem)
    , is_cached_mem(copy_from.is_cached_mem)
    , shared_cached_list(copy_from.shared_cached_list)
    , valid_cache(copy_from.valid_cache)
    , cur_cached_list(copy_from.cur_cached_list)
    , invalid_count(copy_from.invalid_count)
  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<ProcQueryPredicate *>::const_iterator it = copy_from.predicates.begin();
	it != copy_from.predicates.end();
	it++)
      predicates.push_back((*it)->clone());

    if (!shared_cached_list) {
      cur_cached_list = NULL;
      valid_cache = false;
    }
  }

  ProcessorQueryImpl::~ProcessorQueryImpl(void)
  {
    assert(references.load() == 0);
    for(std::vector<ProcQueryPredicate *>::iterator it = predicates.begin();
	it != predicates.end();
	it++)
      delete *it;

    if (!shared_cached_list && cur_cached_list) {
      delete cur_cached_list;
      cur_cached_list = NULL;
    }
  }

  void ProcessorQueryImpl::add_reference(void)
  {
    references.fetch_add(1);
  }

  void ProcessorQueryImpl::remove_reference(void)
  {
    int left = references.fetch_sub(1) - 1;
    if(left == 0)
      delete this;
  }

  ProcessorQueryImpl *ProcessorQueryImpl::writeable_reference(void)
  {
    // safe to test without an atomic because we are a reference, and if the count is 1,
    //  there can be no others
    if(references.load() == 1) {
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
    // any constraint to processor query will invalidate cached list
    if (valid_cache && cur_cached_list) {
      delete cur_cached_list; cur_cached_list = NULL;
    }
    valid_cache = false;
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
    if (valid_cache && cur_cached_list) {
      delete cur_cached_list; cur_cached_list = NULL;
    }
    valid_cache = false;
  }

  void ProcessorQueryImpl::add_predicate(ProcQueryPredicate *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
    if (valid_cache && cur_cached_list) {
      delete cur_cached_list; cur_cached_list = NULL;
    }
    valid_cache = false;
  }

  Processor ProcessorQueryImpl::next(Processor after)
  {
    Processor nextp = Processor::NO_PROC;

    if (cur_cached_list == NULL) {
      log_query.fatal() << "cur_cached_list is null";
      assert(0);
    }
    if (!cur_cached_list) return nextp;
    if (!cur_cached_list->size()) return nextp;
    if ((*cur_cached_list)[0] == after)
      cur_index = 1;
    else {
      if (((*cur_cached_list)[cur_index] != after) && (cur_index < cur_cached_list->size())) {
        log_query.fatal() << "cur_cached_list: inconsistent state";
        assert(0);
      }
      ++cur_index;
    }
    if (cur_index < cur_cached_list->size())
      nextp =  (*cur_cached_list)[cur_index];
    return nextp;
  }

  std::vector<Processor>* ProcessorQueryImpl::cached_list() const
  {

    if ((invalid_count == cache_invalid_count) && valid_cache) {
      log_query.debug("processor cached_list: [valid_cache] \n"); 
      return cur_cached_list;
    }

    log_query.debug("processor cached_list: is_restricted_kind= %d, is_restricted_node = %d, is_cached_mem = %d \n", is_restricted_kind, is_restricted_node, is_cached_mem);
    // shared cache, not mutated query
    if (is_restricted_kind && (!is_restricted_node) && (!predicates.size() || is_cached_mem)) {
      // if the caches are invalid and not in the middle of a query, reset
      if (!global_valid_cache) {
        _proc_cache.clear();
        _proc_cache_affinity.clear();
        global_valid_cache = true;
      }
      bool found=false;
      if (!is_cached_mem) {
        std::map<Processor::Kind, std::vector<Processor> >::const_iterator it;
        it = _proc_cache.find(restricted_kind);
        if (it != _proc_cache.end()) {
          found=true;
        }
      }
      // proc-mem affinity
      else {
        std::map<Processor::Kind, std::map<Memory, std::vector<Processor> > >::const_iterator it2;
        it2 = _proc_cache_affinity.find(restricted_kind);
        if (it2 != _proc_cache_affinity.end()) {
          std::map<Memory, std::vector<Processor> >::const_iterator it3 = it2->second.find(cached_mem);
          if (it3 != it2->second.end()) {
            found = true;
          }
        }
      }
      // if not found - dynamically create the cache
      if (!found) {
        if (is_cached_mem) {
          std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
          machine->get_proc_mem_affinity(proc_mem_affinities);
          for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
            Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
            _proc_cache_affinity[affinity.p.kind()][affinity.m].push_back(affinity.p);
            if (affinity.p.kind() == restricted_kind)
              found = true;
          }
        }
        else  {
          std::map<NodeID, MachineNodeInfo *>::const_iterator it;
          it = machine->nodeinfos.begin();
          // iterate over all the nodes
          while(it != machine->nodeinfos.end()) {
            std::map<Processor::Kind, std::map<Processor, MachineProcInfo *> >::const_iterator it2 =
              it->second->proc_by_kind.find(restricted_kind);
            const std::map<Processor, MachineProcInfo *> *plist;
            // if the list is not empty
            if(it2 != it->second->proc_by_kind.end())
              plist = &(it2->second);
            else
              plist = 0;
            if (plist) {
              found=true;
              for (std::map<Processor, MachineProcInfo* >::const_iterator it3 =  plist->begin(); it3 != plist->end(); ++it3)
                (_proc_cache)[restricted_kind].push_back(it3->first);
            }
            it++;
          }
        }
      }
      if (found) {
        if (!is_cached_mem)
          return  &((_proc_cache)[restricted_kind]);
        else
          return  &(((_proc_cache_affinity)[restricted_kind])[cached_mem]);
      }
    }
    return NULL;
  }


  bool ProcessorQueryImpl::cached_query(Processor& pval, QueryType q)  const
  {
    bool is_valid = false;
    if (!Config::use_machine_query_cache)
      return is_valid;

#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    while (true) {
      if (__sync_bool_compare_and_swap(&init,0,1)) {
        std::vector<Processor>* clist = NULL;
        if ((clist = cached_list()) != NULL) {
          is_valid = true;
          switch (q) {
          case QUERY_FIRST:
            pval = clist->empty() ? Processor::NO_PROC : (*clist)[0];
            break;
          case QUERY_RANDOM:
            pval =  (*clist)[lrand48() % clist->size()];
            break;
          default:
            assert(false); // invalid query
            break;
          }
        }
        __sync_sub_and_fetch(&init,1);
        return is_valid;
      }
    }
#else
    return is_valid;
#endif
  }

  bool ProcessorQueryImpl::cached_query(Processor p, Processor& pval)
  {
    bool is_valid = false;
    if (!Config::use_machine_query_cache)
      return is_valid;

#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    while (true) {
      if (__sync_bool_compare_and_swap(&init,0,1)) {
        if (invalid_count != cache_invalid_count) {
          log_query.debug("processor cached_query: invalid cache  %u\n", cache_invalid_count);
          return is_valid;
        }
        is_valid = true;
        std::vector<Processor>* clist = NULL;
        if ((clist = cached_list()) != NULL) {
          if (!valid_cache)shared_cached_list = true;
          cur_cached_list = clist;
          pval = next(p);
        }
        else
          pval = mutated_cached_query(p);
        __sync_sub_and_fetch(&init,1);
        return is_valid;
      }
    }
#else
    return is_valid;
#endif
  }

 bool ProcessorQueryImpl::cached_query(size_t &count) const
  {
    bool is_valid = false;
    if (!Config::use_machine_query_cache)
      return is_valid;

#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    while (true) {
      if (__sync_bool_compare_and_swap(&init,0,1)) {
        std::vector<Processor>* clist = NULL;
        if ((clist = cached_list()) != NULL) {
          count =  clist->size();
          is_valid=true;
        }
        __sync_sub_and_fetch(&init,1);
        return is_valid;
      }
    }
#else
    return is_valid;
#endif
  }

  Processor ProcessorQueryImpl::mutated_cached_query(Processor after)
  {
    // if we have a valid_mutated_cache list
    Processor pval = Processor::NO_PROC;
    bool first_time = true;
    if (valid_cache) {
      pval = next(after);
      log_query.debug("mutated_cached_query: processor output id: [valid cache] = %llx\n", pval.id);
      return pval;
    }
    valid_cache=true;

    // if this is not pointing to a persistent list
    if (cur_cached_list && !shared_cached_list) {
      delete cur_cached_list;
      cur_cached_list=NULL;
    }

    shared_cached_list = false;
    // general mutated query
    cur_cached_list = new std::vector<Processor>();
    // enter the first element i.e. after
    cur_cached_list->push_back(after);
    cur_index = 1;
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).proc_owner_node());
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
        if(it->first == NodeID(ID(after).proc_owner_node()))
          it2 = plist->upper_bound(after);
        else
          it2 = plist->begin();
        while(it2 != plist->end()) {
          bool ok = true;
          for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
              ok && (it3 != predicates.end());
              it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok) {
            if (first_time) {
              pval = it2->first;
              first_time = false;
            }
            cur_cached_list->push_back(it2->first);
          }
          // try next processor (if it exists)
          ++it2;
        }
      }
      // try the next node (if it exists)
      ++it;
    }
    log_query.debug("mutated_cached_query processor output id =  %llx\n", pval.id);
    return pval;
  }

  Processor ProcessorQueryImpl::first_match(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted_node && (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
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

    // optimize if restricted kind without predicates and restricted_node attached to the query
    Processor pval = Processor::NO_PROC;
    if (cached_query(pval, QUERY_FIRST))
      return pval;

    // general case where restricted_node_id or predicates are defined
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
	break;

      const std::map<Processor, MachineProcInfo *> *plist = 0;
      if(is_restricted_kind) {
	std::map<Processor::Kind, std::map<Processor, MachineProcInfo *> >::const_iterator it2 = it->second->proc_by_kind.find(restricted_kind);
	if(it2 != it->second->proc_by_kind.end())
	  plist = &(it2->second);
	else
	  plist = 0;
      } else
	plist = &(it->second->procs);

      if(plist) {
	  // if restricted node id is set or is_restricted_kind && predicates.size() == 0
	if ((is_restricted_node || is_restricted_kind)  && (!predicates.size()))
	  return plist->begin()->first;

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

  Processor ProcessorQueryImpl::next_match(Processor after)
  {
    if(!after.exists()) return Processor::NO_PROC;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(p.id <= after.id) continue;
	if(is_restricted_node && (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
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
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).proc_owner_node());
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
	if(it->first == NodeID(ID(after).proc_owner_node()))
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

  // cache the set of valid processors if required and return next
  Processor ProcessorQueryImpl::cache_next(Processor after)
  {

    log_query.debug("cache_next: processor input id =  %llx\n", after.id);
    Processor pval = Processor::NO_PROC;
    if (cached_query(after, pval))
      return pval;
    else
      // caches may be invalidated in the middle of a query
      return next_match(after);
  }

  size_t ProcessorQueryImpl::count_matches(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return 0;
    std::set<Processor> pset;
    {
      // problem with nested locks here...
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted_node && (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
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
    size_t count=0;
    if (cached_query(count))
      return count;

    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
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
	if(it2 != it->second->proc_by_kind.end()) {
	  plist = &(it2->second);
	  if (!predicates.size())
	    return it2->second.size();
	}
	else
	  plist = 0;
      } else
	plist = &(it->second->procs);

      if(plist) {
	if ((is_restricted_node) && (!predicates.size()))
	  return plist->size();

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
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Processor p =(*it).p;
	if(is_restricted_node && (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
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
    // optimize if restricted kind without predicates and restricted_node attached to the query
    Processor pval = Processor::NO_PROC;
    if (cached_query(pval, QUERY_RANDOM))
      return pval;
    int count = 0;
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
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
  unsigned int MemoryQueryImpl::init=0;
  unsigned int MemoryQueryImpl::cache_invalid_count=0;
  bool MemoryQueryImpl::global_valid_cache=true;

  std::map<Memory::Kind, std::vector<Memory> > MemoryQueryImpl::_mem_cache;

  MemoryQueryImpl::MemoryQueryImpl(const Machine& _machine)
    : references(1)
    , machine((MachineImpl *)_machine.impl)
    , is_restricted_node(false)
    , is_restricted_kind(false)
    , restricted_min_capacity(0)
    , shared_cached_list(false)
    , valid_cache(false)
    , cur_cached_list(NULL)
    , invalid_count(cache_invalid_count)

  {
  }

  MemoryQueryImpl::MemoryQueryImpl(const MemoryQueryImpl& copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted_node(copy_from.is_restricted_node)
    , restricted_node_id(copy_from.restricted_node_id)
    , is_restricted_kind(copy_from.is_restricted_kind)
    , restricted_kind(copy_from.restricted_kind)
    , restricted_min_capacity(copy_from.restricted_min_capacity)
    , shared_cached_list(copy_from.shared_cached_list)
    , valid_cache(copy_from.valid_cache)
    , cur_cached_list(copy_from.cur_cached_list)
    , invalid_count(cache_invalid_count)

  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<MemoryQueryPredicate *>::const_iterator it = copy_from.predicates.begin();
	it != copy_from.predicates.end();
	it++)
      predicates.push_back((*it)->clone());

    if (!shared_cached_list) {
      cur_cached_list = NULL;
      valid_cache = false;
    }
  }

  MemoryQueryImpl::~MemoryQueryImpl(void)
  {
    assert(references.load() == 0);
    for(std::vector<MemoryQueryPredicate *>::iterator it = predicates.begin();
	it != predicates.end();
	it++)
      delete *it;
    if (!shared_cached_list && cur_cached_list)
      delete cur_cached_list;
  }

  void MemoryQueryImpl::add_reference(void)
  {
    references.fetch_add(1);
  }

  void MemoryQueryImpl::remove_reference(void)
  {
    int left = references.fetch_sub(1) - 1;
    if(left == 0)
      delete this;
  }

  MemoryQueryImpl *MemoryQueryImpl::writeable_reference(void)
  {
    // safe to test without an atomic because we are a reference, and if the count is 1,
    //  there can be no others
    if(references.load() == 1) {
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
    if (valid_cache && cur_cached_list) {
      delete cur_cached_list; cur_cached_list = NULL;
    }
    valid_cache = false;
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
    if (valid_cache && cur_cached_list) {
      delete cur_cached_list; cur_cached_list = NULL;
    }
    valid_cache = false;
  }

  void MemoryQueryImpl::restrict_by_capacity(size_t new_min_bytes)
  {
    restricted_min_capacity = std::max(restricted_min_capacity,
                                       new_min_bytes);
    if (valid_cache && cur_cached_list) {
      delete cur_cached_list; cur_cached_list = NULL;
    }
    valid_cache = false;
  }

  void MemoryQueryImpl::add_predicate(MemoryQueryPredicate *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
    if (valid_cache && cur_cached_list) {
      delete cur_cached_list; cur_cached_list = NULL;
    }
    valid_cache = false;
  }



  std::vector<Memory>* MemoryQueryImpl::cached_list() const
  {
    if ((invalid_count == cache_invalid_count) && valid_cache) {
      return cur_cached_list;
    }
    bool found = false;
    // shared cache, not mutated query
    if (is_restricted_kind && (!is_restricted_node) && (!predicates.size()) &&
        (restricted_min_capacity == 0)) {
      // if the caches are invalid and not in the middle of a query, reset
      if (!global_valid_cache) {
        _mem_cache.clear();
        global_valid_cache = true;
      }
      // if cache is not valid for this query
      std::map<Memory::Kind, std::vector<Memory> >::const_iterator it;
      it = _mem_cache.find(restricted_kind);
      if (it != _mem_cache.end()) {
        found=true;
      }
      // if not found - dynamically create the cache
      // mem_cache may also be cleared/reset when dealing with resilience/elasticity
      if (!found) {
        std::map<NodeID, MachineNodeInfo *>::const_iterator it;
        it = machine->nodeinfos.begin();
        // iterate over all the nodes
        while(it != machine->nodeinfos.end()) {
          std::map<Memory::Kind, std::map<Memory, MachineMemInfo *> >::const_iterator it2 = it->second->mem_by_kind.find(restricted_kind);
          // find all the memories by memory kind
          const std::map<Memory, MachineMemInfo *> *plist;
          // if the list is not empty
          if(it2 != it->second->mem_by_kind.end())
            plist = &(it2->second);
          else
            plist = 0;
          if (plist) {
            found=true;
            for (std::map<Memory, MachineMemInfo* >::const_iterator it3 =  plist->begin(); it3 != plist->end(); ++it3)
              (_mem_cache)[restricted_kind].push_back(it3->first);
          }
          it++;
        }
      }
    }
    if (found)
      return &((_mem_cache)[restricted_kind]);
    return NULL;
  }



  bool MemoryQueryImpl::cached_query(Memory& mval, QueryType q)  const
  {
    bool is_valid = false;
    if (!Config::use_machine_query_cache)
      return is_valid;

#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    while (true) {
      if (__sync_bool_compare_and_swap(&init,0,1)) {
        std::vector<Memory>* clist = NULL;
        if ((clist = cached_list()) != NULL) {
          is_valid = true;
          switch (q) {
          case QUERY_FIRST:
            mval = clist->empty() ? Memory::NO_MEMORY : (*clist)[0];
            break;
          case QUERY_RANDOM:
            mval =  (*clist)[lrand48() % clist->size()];
            break;
          default:
            assert(false);  // invalid query
            break;
          }
        }
        __sync_sub_and_fetch(&init,1);
        return is_valid;
      }
    }
#else
    return is_valid;
#endif
  }

  bool MemoryQueryImpl::cached_query(Memory m, Memory& mval)
  {
    bool is_valid = false;
    if (!Config::use_machine_query_cache)
      return is_valid;

#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    while (true) {
      if (__sync_bool_compare_and_swap(&init,0,1)) {
        if (invalid_count != cache_invalid_count) {
          log_query.debug("memory cache_query: invalid cache  %u\n", cache_invalid_count);
          return is_valid;
        }
        is_valid = true;
        std::vector<Memory>* clist = NULL;
        if ((clist = cached_list()) != NULL) {
          if(!valid_cache) shared_cached_list = true;
          cur_cached_list = clist;
          mval = next(m);
        }
        else
          mval = mutated_cached_query(m);
        __sync_sub_and_fetch(&init,1);
        return is_valid;
      }
    }
#else
    return is_valid;
#endif
  }

 bool MemoryQueryImpl::cached_query(size_t &count) const
  {
    bool is_valid = false;
    if (!Config::use_machine_query_cache)
      return is_valid;

#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    while (true) {
      if (__sync_bool_compare_and_swap(&init,0,1)) {
        std::vector<Memory>* clist = NULL;
        if ((clist = cached_list()) != NULL) {
          count =  clist->size();
          is_valid=true;
        }
        __sync_sub_and_fetch(&init,1);
        return is_valid;
      }
    }
#else
    return is_valid;
#endif
  }

  Memory MemoryQueryImpl::mutated_cached_query(Memory after)
  {
    // if we have a valid_mutated_cache list
    Memory mval = Memory::NO_MEMORY;
    bool first_time = true;
    if (valid_cache) {
      mval = next(after);
      log_query.debug("mutated_cached_query: memory output id: [valid cache] = %llx\n", mval.id);
      return mval;
    }
    valid_cache=true;
    // if this is not pointing to a persistent list
    if (cur_cached_list && !shared_cached_list) {
      delete cur_cached_list;
      cur_cached_list=NULL;
    }
    shared_cached_list = false;
    // general mutated query
    cur_cached_list = new std::vector<Memory>();
    // enter the first element
    cur_cached_list->push_back(after);
    cur_index = 1;
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).memory_owner_node());
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
        if(it->first == NodeID(ID(after).memory_owner_node()))
          it2 = plist->upper_bound(after);
        else
          it2 = plist->begin();
        while(it2 != plist->end()) {
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2->first.capacity() >= restricted_min_capacity));
          for(std::vector<MemoryQueryPredicate *>::const_iterator it3 = predicates.begin();
              ok && (it3 != predicates.end());
              it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok) {
            if (first_time) {
              mval = it2->first;
              first_time = false;
            }
            cur_cached_list->push_back(it2->first);
          }
          // try next memory (if it exists)
          ++it2;
        }
      }
      // try the next node (if it exists)
      ++it;
    }
    return mval;
  }

  Memory MemoryQueryImpl::first_match(void) const
  {
#if USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return Memory::NO_MEMORY;
    Memory lowest = Memory::NO_MEMORY;
    {
      // problem with nested locks here...
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted_node && (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
        if((restricted_min_capacity > 0) && (m.capacity() < restricted_min_capacity))
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

    Memory mval = Memory::NO_MEMORY;
    if (cached_query(mval, QUERY_FIRST))
      return mval;

    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
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
        if ((is_restricted_node) && (!predicates.size()))
	  return plist->begin()->first;

	std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2->first.capacity() >= restricted_min_capacity));
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
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(m.id <= after.id) continue;
	if(is_restricted_node && (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
        if((restricted_min_capacity > 0) && (m.capacity() < restricted_min_capacity))
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
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).memory_owner_node());
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
	if(it->first == NodeID(ID(after).memory_owner_node()))
	  it2 = plist->upper_bound(after);
	else
	  it2 = plist->begin();
	while(it2 != plist->end()) {
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2->first.capacity() >= restricted_min_capacity));
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

  Memory MemoryQueryImpl::next(Memory after) {
    Memory nextp = Memory::NO_MEMORY;
    if (!cur_cached_list) return nextp;
    if (!cur_cached_list->size()) return nextp;
    if ((*cur_cached_list)[0] == after)
      cur_index = 1;
    else
      ++cur_index;
    if (cur_index < cur_cached_list->size())
      nextp =  (*cur_cached_list)[cur_index];
    return nextp;
  }

  Memory MemoryQueryImpl::cache_next(Memory after) {
    log_query.debug("cache_next: memory input id =  %llx\n", after.id);
    Memory mval = Memory::NO_MEMORY;
    if (cached_query(after, mval))
      return mval;
    else
      // caches may be invalidated in the middle of a query
      return next_match(after);
  }

  size_t MemoryQueryImpl::count_matches(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0)) return 0;
    std::set<Memory> pset;
    {
      // problem with nested locks here...
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted_node && (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
        if((restricted_min_capacity > 0) && (m.capacity() < restricted_min_capacity))
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
    if (cached_query(count))
      return count;
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
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
        // if predicates is empty then increment count by size of plist
        if (!predicates.size()) {
          count =  count + plist->size();
        }
        // else iterate over all relevant memories on this node and match the predicates
        else  {
	std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
	while(it2 != plist->end()) {
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2->first.capacity() >= restricted_min_capacity));
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
      //AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = machine->proc_mem_affinities.begin();
	  it != machine->proc_mem_affinities.end();
	  it++) {
	Memory m =(*it).m;
	if(is_restricted_node && (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
	  continue;
	if(is_restricted_kind && (m.kind() != restricted_kind))
	  continue;
        if((restricted_min_capacity > 0) && (m.capacity() < restricted_min_capacity))
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

    Memory mval = Memory::NO_MEMORY;
    if (cached_query(mval, QUERY_RANDOM))
      return mval;

    size_t count = 0;
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
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
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2->first.capacity() >= restricted_min_capacity));
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

  static atomic<int> announcements_received(0);
  static atomic<int> announce_fragments_expected(0);

  /*static*/ void NodeAnnounceMessage::handle_message(NodeID sender, const NodeAnnounceMessage &args,
						      const void *data, size_t datalen)

  {
    log_annc.info() << "received fragment from " << sender
                    << ": num_fragments=" << args.num_fragments;
    
    if(args.num_fragments > 0) {
      // update the remaining fragment count and then mark that we've got
      //  this sender's contribution
      announce_fragments_expected.fetch_add(args.num_fragments);
      announcements_received.fetch_add_acqrel(1);
    }

    get_machine()->parse_node_announce_data(sender,
                                            data, datalen, true);

    // this fragment has been handled
    announce_fragments_expected.fetch_sub_acqrel(1);
  }

  /*static*/ void NodeAnnounceMessage::await_all_announcements(void)
  {
    // two steps:

    // 1) wait until we've got the fragment-count-bearing message from every
    //  other node
    while(announcements_received.load() < Network::max_node_id) {
      Thread::yield();
      //do_some_polling();
    }

    // 2) then wait for the expected fragment count to drop to zero
    while(announce_fragments_expected.load_acquire() > 0) {
      Thread::yield();
    }

    log_annc.info("node %d has received all of its announcements", Network::my_node_id);

    // 3) go ahead and build the by-kind maps in each node info
    MachineImpl *impl = get_machine();
    for(std::map<int, MachineNodeInfo *>::iterator it = impl->nodeinfos.begin();
        it != impl->nodeinfos.end();
        ++it)
      it->second->update_kind_maps();
  }
  

  ActiveMessageHandlerReg<NodeAnnounceMessage> node_announce_message;

}; // namespace Realm
