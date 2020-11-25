/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#include "realm/mem_impl.h"

#include "realm/proc_impl.h"
#include "realm/logging.h"
#include "realm/serialize.h"
#include "realm/inst_impl.h"
#include "realm/runtime_impl.h"
#include "realm/profiling.h"
#include "realm/utils.h"
#include "realm/activemsg.h"

namespace Realm {

  Logger log_malloc("malloc");
  Logger log_copy("copy");
  extern Logger log_inst; // in inst_impl.cc


  namespace Config {
    // if true, Realm memories attempt to satisfy instance allocation requests
    //  on the basis of deferred instance destructions
    bool deferred_instance_allocation = true;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class Memory
  //

    AddressSpace Memory::address_space(void) const
    {
      // this is a hack for the Legion runtime
      ID id(*this);
      unsigned n = id.memory_owner_node();
      if(n <= ID::MAX_NODE_ID)
        return n;
      else
        return 0;  // claim node 0 owns "global" things
    }

    Memory::Kind Memory::kind(void) const
    {
      return get_runtime()->get_memory_impl(*this)->get_kind();
    }

    size_t Memory::capacity(void) const
    {
      return get_runtime()->get_memory_impl(*this)->size;
    }

    // reports a problem with a memory in general (this is primarily for fault injection)
    void Memory::report_memory_fault(int reason,
				     const void *reason_data,
				     size_t reason_size) const
    {
      assert(0);
    }

    /*static*/ const Memory Memory::NO_MEMORY = { 0 };


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryImpl
  //

    MemoryImpl::MemoryImpl(Memory _me, size_t _size,
			   MemoryKind _kind, Memory::Kind _lowlevel_kind,
			   NetworkSegment *_segment)
      : me(_me), size(_size), kind(_kind), lowlevel_kind(_lowlevel_kind)
      , segment(_segment)
    {}

    MemoryImpl::~MemoryImpl(void)
    {
      for(std::vector<RegionInstanceImpl *>::iterator it = local_instances.instances.begin();
	  it != local_instances.instances.end();
	  ++it)
	if(*it)
	  delete *it;

      for(std::map<NodeID, InstanceList *>::const_iterator it = instances_by_creator.begin();
	  it != instances_by_creator.end();
	  ++it) {
	for(std::vector<RegionInstanceImpl *>::iterator it2 = it->second->instances.begin();
	    it2 != it->second->instances.end();
	    ++it2)
	  if(*it2)
	    delete *it2;
	delete it->second;
      }
    }

    // default implementation handles deferral, but falls through to
    //  allocate_storage_immediate for any actual allocation
    MemoryImpl::AllocationResult MemoryImpl::allocate_storage_deferrable(RegionInstanceImpl *inst,
									 bool need_alloc_result,
									 Event precondition)
    {
      // all allocation requests are handled by the memory's owning node for
      //  now - local caching might be possible though
      NodeID target = ID(me).memory_owner_node();
      assert(target == Network::my_node_id);

      // check precondition on allocation
      bool alloc_poisoned = false;
      if(precondition.has_triggered_faultaware(alloc_poisoned)) {
	// attempt immediate allocation (it'll handle poison)
	return allocate_storage_immediate(inst, need_alloc_result,
					  alloc_poisoned,
					  TimeLimit::responsive());
      } else {
	// defer allocation attempt
	inst->metadata.inst_offset = RegionInstanceImpl::INSTOFFSET_DELAYEDALLOC;
	inst->deferred_create.defer(inst, this,
				    need_alloc_result,
				    precondition);
	return ALLOC_DEFERRED /*asynchronous notification*/;
      }
    }

    void MemoryImpl::release_storage_deferrable(RegionInstanceImpl *inst,
						Event precondition)
    {
      // all allocation requests are handled by the memory's owning node for
      //  now - local caching might be possible though
      NodeID target = ID(me).memory_owner_node();
      assert(target == Network::my_node_id);

      bool poisoned = false;
      if(precondition.has_triggered_faultaware(poisoned)) {
	// fall through to immediate storage release
	release_storage_immediate(inst, poisoned,
				  TimeLimit::responsive());
      } else {
	// ask the instance to tell us when the precondition is satisified
	inst->deferred_destroy.defer(inst, this, precondition);
      }
    }

#if 0
    off_t MemoryImpl::alloc_bytes_local(size_t size)
    {
      assert(0);
      return 0;
    }

    void MemoryImpl::free_bytes_local(off_t offset, size_t size)
    {
      assert(0);
    }

    // make bad offsets really obvious (+1 PB)
    static const off_t ZERO_SIZE_INSTANCE_OFFSET = 1ULL << ((sizeof(off_t) == 8) ? 50 : 30);

    off_t MemoryImpl::alloc_bytes_local(size_t size)
    {
      AutoLock<> al(mutex);

      // for zero-length allocations, return a special "offset"
      if(size == 0) {
	return this->size + ZERO_SIZE_INSTANCE_OFFSET;
      }

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding allocation from %zd to %zd",
			  size, (size_t)(size + (alignment - leftover)));
	  size += (alignment - leftover);
	}
      }
      // HACK: pad the size by a bit to see if we have people falling off
      //  the end of their allocations
      size += 0;

      // try to minimize footprint by allocating at the highest address possible
      if(!free_blocks.empty()) {
	std::map<off_t, off_t>::iterator it = free_blocks.end();
	do {
	  --it;  // predecrement since we started at the end

	  if(it->second == (off_t)size) {
	    // perfect match
	    off_t retval = it->first;
	    free_blocks.erase(it);
	    log_malloc.info("alloc full block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, (ssize_t)retval);
	    usage += size;
	    if(usage > peak_usage) peak_usage = usage;
	    size_t footprint = this->size - retval;
	    if(footprint > peak_footprint) peak_footprint = footprint;
	    return retval;
	  }
	
	  if(it->second > (off_t)size) {
	    // some left over
	    off_t leftover = it->second - size;
	    off_t retval = it->first + leftover;
	    it->second = leftover;
	    log_malloc.info("alloc partial block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, (ssize_t)retval);
	    usage += size;
	    if(usage > peak_usage) peak_usage = usage;
	    size_t footprint = this->size - retval;
	    if(footprint > peak_footprint) peak_footprint = footprint;
	    return retval;
	  }
	} while(it != free_blocks.begin());
      }

      // no blocks large enough - boo hoo
      log_malloc.info("alloc FAILED: mem=" IDFMT " size=%zd", me.id, size);
      return -1;
    }

    void MemoryImpl::free_bytes_local(off_t offset, size_t size)
    {
      log_malloc.info() << "free block: mem=" << me << " size=" << size << " ofs=" << offset;
      AutoLock<> al(mutex);

      // frees of zero bytes should have the special offset
      if(size == 0) {
	assert((size_t)offset == this->size + ZERO_SIZE_INSTANCE_OFFSET);
	return;
      }

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding free from %zd to %zd",
			  size, (size_t)(size + (alignment - leftover)));
	  size += (alignment - leftover);
	}
      }

      usage -= size;
      // only made things smaller, so can't impact the peak usage

      if(free_blocks.size() > 0) {
	// find the first existing block that comes _after_ us
	std::map<off_t, off_t>::iterator after = free_blocks.lower_bound(offset);
	if(after != free_blocks.end()) {
	  // found one - is it the first one?
	  if(after == free_blocks.begin()) {
	    // yes, so no "before"
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }
	    free_blocks[offset] = size;
	  } else {
	    // no, get range that comes before us too
	    std::map<off_t, off_t>::iterator before = after; before--;

	    // if we're adjacent to the after, merge with it
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }

	    // if we're adjacent with the before, grow it instead of adding
	    //  a new range
	    assert((before->first + before->second) <= offset);
	    if((before->first + before->second) == offset) {
	      before->second += size;
	    } else {
	      free_blocks[offset] = size;
	    }
	  }
	} else {
	  // nothing's after us, so just see if we can merge with the range
	  //  that's before us

	  std::map<off_t, off_t>::iterator before = after; before--;

	  // if we're adjacent with the before, grow it instead of adding
	  //  a new range
	  assert((before->first + before->second) <= offset);
	  if((before->first + before->second) == offset) {
	    before->second += size;
	  } else {
	    free_blocks[offset] = size;
	  }
	}
      } else {
	// easy case - nothing was free, so now just our block is
	free_blocks[offset] = size;
      }
    }
#endif

    Memory::Kind MemoryImpl::get_kind(void) const
    {
      return lowlevel_kind;
    }

    RegionInstanceImpl *MemoryImpl::get_instance(RegionInstance i)
    {
      ID id(i);
      assert(id.is_instance());

      NodeID cnode = id.instance_creator_node();
      unsigned idx = id.instance_inst_idx();
      if(cnode == Network::my_node_id) {
	// if it was locally created, we can directly access the local_instances list
	//  and it's a fatal error if it doesn't exist
	AutoLock<> al(local_instances.mutex);
	assert(idx < local_instances.instances.size());
	assert(local_instances.instances[idx] != 0);
	return local_instances.instances[idx];
      } else {
	// figure out which instance list to look in - non-local creators require a 
	//  protected lookup
	InstanceList *ilist;
	{
	  AutoLock<> al(instance_map_mutex);
	  // this creates a new InstanceList if needed
	  InstanceList *& iref = instances_by_creator[cnode];
	  if(!iref)
	    iref = new InstanceList;
	  ilist = iref;
	}

	// now look up (and possibly create) the instance in the right list
	{
	  AutoLock<> al(ilist->mutex);

	  if(idx >= ilist->instances.size())
	    ilist->instances.resize(idx + 1, 0);

	  if(ilist->instances[idx] == 0) {
	    log_inst.info() << "creating proxy for remotely-created instance: " << i;
	    ilist->instances[idx] = new RegionInstanceImpl(i, me);
	  }

	  return ilist->instances[idx];
	}
      }
    }

    // adds a new instance to this memory, to be filled in by caller
    RegionInstanceImpl *MemoryImpl::new_instance(void)
    {
      // selecting a slot requires holding the mutex
      unsigned inst_idx;
      RegionInstanceImpl *inst_impl;
      {
	AutoLock<> al(local_instances.mutex);
	  
	if(local_instances.free_list.empty()) {
	  // need to grow the list - do it in chunks
	  const size_t chunk_size = 8;
	  size_t old_size = local_instances.instances.size();
	  size_t new_size = old_size + chunk_size;
	  if(new_size > (1 << ID::INSTANCE_INDEX_WIDTH)) {
	    new_size = (1 << ID::INSTANCE_INDEX_WIDTH);
	    if(old_size == new_size) {
	      // completely out of slots - nothing we can do
	      return 0;
	    }
	  }
	  local_instances.instances.resize(new_size, 0);
	  local_instances.free_list.resize(chunk_size - 1);
	  for(size_t i = 0; i < chunk_size - 1; i++)
	    local_instances.free_list[i] = new_size - 1 - i;
	  inst_idx = old_size;
	  inst_impl = 0;
	} else {
	  inst_idx = local_instances.free_list.back();
	  local_instances.free_list.pop_back();
	  inst_impl = local_instances.instances[inst_idx];
	}
      }

      // we've got a slot and possibly an object to reuse - if not, allocate
      //  it now (only retaking the lock to add it back to the list)
      if(!inst_impl) {
	ID mem_id(me);
	RegionInstance i = ID::make_instance(mem_id.memory_owner_node(),
					     Network::my_node_id /*creator*/,
					     mem_id.memory_mem_idx(),
					     inst_idx).convert<RegionInstance>();
	log_inst.info() << "creating new local instance: " << i;
	inst_impl = new RegionInstanceImpl(i, me);
	{
	  AutoLock<> al(local_instances.mutex);
	  local_instances.instances[inst_idx] = inst_impl;
	}
      } else
	log_inst.info() << "reusing local instance: " << inst_impl->me;

      return inst_impl;
    }

    // releases a deleted instance so that it can be reused
    void MemoryImpl::release_instance(RegionInstance inst)
    {
      int inst_idx = ID(inst).instance_inst_idx();

      log_inst.info() << "releasing local instance: " << inst;
      {
	AutoLock<> al(local_instances.mutex);
	local_instances.free_list.push_back(inst_idx);
      }
    }

    void *MemoryImpl::get_inst_ptr(RegionInstanceImpl *inst,
				   off_t offset, size_t size)
    {
      // fall through to old memory-wide implementation
      return get_direct_ptr(inst->metadata.inst_offset + offset, size);
    }

    const ByteArray *MemoryImpl::get_rdma_info(NetworkModule *network) const
    {
      return (segment ? segment->get_rdma_info(network) : 0);
    }

    bool MemoryImpl::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
    {
      // any ability to convert to remote addresses will come from subclasses
      return false;
    }

    NetworkSegment *MemoryImpl::get_network_segment()
    {
      return segment;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IBMemory
  //

    IBMemory::IBMemory(Memory _me, size_t _size,
		       MemoryKind _kind, Memory::Kind _lowlevel_kind,
		       void *prealloc_base, NetworkSegment *_segment)
      : MemoryImpl(_me, _size, _kind, _lowlevel_kind, _segment)
      , base(static_cast<char *>(prealloc_base))
    {
      free_blocks[0] = _size;
    }

    IBMemory::~IBMemory()
    {
    }

    // old-style allocation used by IB memories
    // make bad offsets really obvious (+1 PB)
    static const off_t ZERO_SIZE_INSTANCE_OFFSET = 1ULL << ((sizeof(off_t) == 8) ? 50 : 30);

    off_t IBMemory::alloc_bytes_local(size_t size)
    {
      AutoLock<> al(mutex);

      // for zero-length allocations, return a special "offset"
      if(size == 0) {
	return this->size + ZERO_SIZE_INSTANCE_OFFSET;
      }

      const size_t alignment = 256;

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding allocation from %zd to %zd",
			  size, (size_t)(size + (alignment - leftover)));
	  size += (alignment - leftover);
	}
      }
      // HACK: pad the size by a bit to see if we have people falling off
      //  the end of their allocations
      size += 0;

      // try to minimize footprint by allocating at the highest address possible
      if(!free_blocks.empty()) {
	std::map<off_t, off_t>::iterator it = free_blocks.end();
	do {
	  --it;  // predecrement since we started at the end

	  if(it->second == (off_t)size) {
	    // perfect match
	    off_t retval = it->first;
	    free_blocks.erase(it);
	    log_malloc.info("alloc full block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, (ssize_t)retval);
#if 0
	    usage += size;
	    if(usage > peak_usage) peak_usage = usage;
	    size_t footprint = this->size - retval;
	    if(footprint > peak_footprint) peak_footprint = footprint;
#endif
	    return retval;
	  }
	
	  if(it->second > (off_t)size) {
	    // some left over
	    off_t leftover = it->second - size;
	    off_t retval = it->first + leftover;
	    it->second = leftover;
	    log_malloc.info("alloc partial block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, (ssize_t)retval);
#if 0
	    usage += size;
	    if(usage > peak_usage) peak_usage = usage;
	    size_t footprint = this->size - retval;
	    if(footprint > peak_footprint) peak_footprint = footprint;
#endif
	    return retval;
	  }
	} while(it != free_blocks.begin());
      }

      // no blocks large enough - boo hoo
      log_malloc.info("alloc FAILED: mem=" IDFMT " size=%zd", me.id, size);
      return -1;
    }

    void IBMemory::free_bytes_local(off_t offset, size_t size)
    {
      log_malloc.info() << "free block: mem=" << me << " size=" << size << " ofs=" << offset;
      AutoLock<> al(mutex);

      // frees of zero bytes should have the special offset
      if(size == 0) {
	assert((size_t)offset == this->size + ZERO_SIZE_INSTANCE_OFFSET);
	return;
      }

      const size_t alignment = 256;

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding free from %zd to %zd",
			  size, (size_t)(size + (alignment - leftover)));
	  size += (alignment - leftover);
	}
      }

#if 0
      usage -= size;
      // only made things smaller, so can't impact the peak usage
#endif

      if(free_blocks.size() > 0) {
	// find the first existing block that comes _after_ us
	std::map<off_t, off_t>::iterator after = free_blocks.lower_bound(offset);
	if(after != free_blocks.end()) {
	  // found one - is it the first one?
	  if(after == free_blocks.begin()) {
	    // yes, so no "before"
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }
	    free_blocks[offset] = size;
	  } else {
	    // no, get range that comes before us too
	    std::map<off_t, off_t>::iterator before = after; before--;

	    // if we're adjacent to the after, merge with it
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }

	    // if we're adjacent with the before, grow it instead of adding
	    //  a new range
	    assert((before->first + before->second) <= offset);
	    if((before->first + before->second) == offset) {
	      before->second += size;
	    } else {
	      free_blocks[offset] = size;
	    }
	  }
	} else {
	  // nothing's after us, so just see if we can merge with the range
	  //  that's before us

	  std::map<off_t, off_t>::iterator before = after; before--;

	  // if we're adjacent with the before, grow it instead of adding
	  //  a new range
	  assert((before->first + before->second) <= offset);
	  if((before->first + before->second) == offset) {
	    before->second += size;
	  } else {
	    free_blocks[offset] = size;
	  }
	}
      } else {
	// easy case - nothing was free, so now just our block is
	free_blocks[offset] = size;
      }
    }

    void *IBMemory::get_direct_ptr(off_t offset, size_t size)
    {
      assert(NodeID(ID(me).memory_owner_node()) == Network::my_node_id);
      assert((offset >= 0) && ((size_t)(offset + size) <= this->size));
      return (base + offset);
    }

    // not used by IB memories
    MemoryImpl::AllocationResult IBMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
							bool need_alloc_result,
							bool poisoned,
							TimeLimit work_until)
    {
      abort();
    }

    void IBMemory::release_storage_immediate(RegionInstanceImpl *inst,
					     bool poisoned,
					     TimeLimit work_until)
    {
      abort();
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalManagedMemory
  //

    LocalManagedMemory::LocalManagedMemory(Memory _me, size_t _size,
					   MemoryKind _kind, size_t _alignment,
					   Memory::Kind _lowlevel_kind,
					   NetworkSegment *_segment)
      : MemoryImpl(_me, _size, _kind, _lowlevel_kind, _segment)
      , alignment(_alignment)
      , cur_release_seqid(0)
      , usage(stringbuilder() << "realm/mem " << _me << "/usage")
      , peak_usage(stringbuilder() << "realm/mem " << _me << "/peak_usage")
      , peak_footprint(stringbuilder() << "realm/mem " << _me << "/peak_footprint")
    {
      current_allocator.add_range(0, _size);
    }

    LocalManagedMemory::~LocalManagedMemory(void)
    {
#ifdef REALM_PROFILE_MEMORY_USAGE
      printf("Memory " IDFMT " usage: peak=%zd (%.1f MB) footprint=%zd (%.1f MB)\n",
	     me.id, 
	     (size_t)peak_usage, peak_usage / 1048576.0,
	     (size_t)peak_footprint, peak_footprint / 1048576.0);
#endif
    }

    // attempt to allocate storage for the specified instance
    MemoryImpl::AllocationResult LocalManagedMemory::allocate_storage_deferrable(RegionInstanceImpl *inst,
										 bool need_alloc_result,
										 Event precondition)
    {
      // all allocation requests are handled by the memory's owning node for
      //  now - local caching might be possible though
      NodeID target = ID(me).memory_owner_node();
      assert(target == Network::my_node_id);

      // check precondition on allocation
      bool alloc_poisoned = false;
      if(precondition.has_triggered_faultaware(alloc_poisoned)) {
	if(alloc_poisoned) {
	  // a poisoned creation works a lot like a failed creation
	  inst->notify_allocation(ALLOC_CANCELLED,
				  RegionInstanceImpl::INSTOFFSET_FAILED,
				  TimeLimit::responsive());
	  return ALLOC_INSTANT_FAILURE;
	} else {
	  // attempt allocation below
	}
      } else {
	// defer allocation attempt
	inst->metadata.inst_offset = RegionInstanceImpl::INSTOFFSET_DELAYEDALLOC;
	inst->deferred_create.defer(inst, this,
				    need_alloc_result,
				    precondition);
	return ALLOC_DEFERRED /*asynchronous notification*/;
      }

      AllocationResult result;
      size_t inst_offset = 0;
      if(inst->metadata.ext_resource != 0) {
	// this is an external allocation - it had better be a memory resource
	ExternalMemoryResource *res = dynamic_cast<ExternalMemoryResource *>(inst->metadata.ext_resource);
	if(res != 0) {
	  // automatic success - make the "offset" be the difference between the
	  //  base address we were given and our own allocation's base
	  void *mem_base = get_direct_ptr(0, 0); // only our subclasses know this
	  assert(mem_base != 0);
	  // underflow is ok here - it'll work itself out when we add the mem_base
	  //  back in on accesses
	  inst_offset = res->base - reinterpret_cast<uintptr_t>(mem_base);
	  result = ALLOC_INSTANT_SUCCESS;
	} else {
	  log_inst.warning() << "attempt to register non-memory resource: mem=" << me << " resource=" << *(inst->metadata.ext_resource);
	  result = ALLOC_INSTANT_FAILURE;
	}
      } else {
	// normal allocation from our managed pool
	AutoLock<> al(allocator_mutex);

	result = attempt_deferrable_allocation(inst,
					       inst->metadata.layout->bytes_used,
					       inst->metadata.layout->alignment_reqd,
					       inst_offset);
      }

      // if we needed an alloc result, send deferred responses too
      if((result != ALLOC_DEFERRED) || need_alloc_result) {
	inst->notify_allocation(result, inst_offset,
				TimeLimit::responsive());
      }

      return result;
    }

    // for internal use by allocation routines - must be called with
    //  allocator_mutex held!
    MemoryImpl::AllocationResult LocalManagedMemory::attempt_deferrable_allocation(RegionInstanceImpl *inst,
										   size_t bytes,
										   size_t alignment,
										   size_t& inst_offset)
    {
      // as long as there aren't any pending allocations, we can attempt to
      //  satisfy the allocation based on the current state
      if(pending_allocs.empty()) {
	bool ok = current_allocator.allocate(inst->me,
					     bytes, alignment, inst_offset);
	if(ok) {
	  return ALLOC_INSTANT_SUCCESS;
	} else {
	  // doesn't currently fit - are there any pending deletes that
	  //  might allow it to fit in the future?
	  // also, check that deferred allocations are even permitted
	  if(pending_releases.empty() ||
	     !Config::deferred_instance_allocation) {
	    // nope - this allocation can't succeed based on what we know
	    //  right now
	    return ALLOC_INSTANT_FAILURE;
	  } else {
	    // build the future state based on those deletes and try again
	    future_allocator = current_allocator;
	    for(std::vector<PendingRelease>::iterator it = pending_releases.begin();
		it != pending_releases.end();
		++it) {
	      // shouldn't have any ready ones here
	      assert(!it->is_ready);
	      // due to network delays, it's possible for multiple
	      //  deallocations of the same instance to be in our list,
	      //  so ignore failures to deallocate from the future state
	      //  (this is conservative, because it can only cause a
	      //  false-failure of a future allocation)
	      future_allocator.deallocate(it->inst->me, true /*missing ok*/);
	    }

	    bool ok = future_allocator.allocate(inst->me,
						bytes, alignment, inst_offset);
	    if(ok) {
	      pending_allocs.push_back(PendingAlloc(inst, bytes, alignment,
						    cur_release_seqid));
	      // now that we have pending allocs, we need release_allocator
	      //  to be valid
	      release_allocator = current_allocator;
	      return ALLOC_DEFERRED;
	    } else {
	      return ALLOC_INSTANT_FAILURE; /*immediate notification*/
	      // NOTE: future_allocator becomes invalid
	    }
	  }
	}
      } else {
	// with other pending allocs, we can only tentatively allocate based
	//  on future state
	bool ok = future_allocator.allocate(inst->me,
					    bytes, alignment, inst_offset);
	if(ok) {
	  pending_allocs.push_back(PendingAlloc(inst, bytes, alignment,
						cur_release_seqid));
	  return ALLOC_DEFERRED;
	} else {
	  return ALLOC_INSTANT_FAILURE; /*immediate notification*/
	}
      }
    }
  
    // release storage associated with an instance
    void LocalManagedMemory::release_storage_deferrable(RegionInstanceImpl *inst,
							Event precondition)
    {
      // all allocation requests are handled by the memory's owning node for
      //  now - local caching might be possible though
      NodeID target = ID(me).memory_owner_node();
      assert(target == Network::my_node_id);

      bool poisoned = false;
      bool triggered = precondition.has_triggered_faultaware(poisoned);

      // a poisoned precondition silently cancels the deletion - up to
      //  requestor to realize this has occurred since the deletion does
      //  not have its own completion event
      if(triggered && poisoned)
	return;

      // ignore external instances here - we can't reuse their memory for
      //  future allocations
      if(inst->metadata.ext_resource == 0) {
	// this release may satisfy pending allocation requests
	std::vector<std::pair<RegionInstanceImpl *, size_t> > successful_allocs;

	do { // so we can 'break' out early below
	  AutoLock<> al(allocator_mutex);

	  // special case: we can get a (deferred only!) destruction of an
	  //  instance whose creation precondition hasn't even been satisfied -
	  //  it won't be represented in the heap state, so we can't do a
	  //  "future release of it" - wait until the creation precondition is
	  //  satisfied
	  if(inst->metadata.inst_offset == RegionInstanceImpl::INSTOFFSET_DELAYEDALLOC) {
	    assert(!triggered);
	    inst->metadata.inst_offset = RegionInstanceImpl::INSTOFFSET_DELAYEDDESTROY;
	    break;
	  }

	  if(pending_allocs.empty()) {
	    if(triggered) {
	      // we can apply the destruction directly to current state
	      if(inst->metadata.inst_offset != RegionInstanceImpl::INSTOFFSET_FAILED)
		current_allocator.deallocate(inst->me);
	    } else {
	      // push the op, but we're not maintaining a future state yet
	      pending_releases.push_back(PendingRelease(inst,
							false /*!triggered*/,
							++cur_release_seqid));
	    }
	  } else {
	    // even if this destruction is ready, we can't update current
	    //  state because older pending ops exist
	    // TODO: pushing past a single pending alloc should always be safe
	    if(triggered) {
	      if(inst->metadata.inst_offset == RegionInstanceImpl::INSTOFFSET_FAILED) {
		// (exception: a ready destruction of a failed allocation need not
		//  be deferred)
	      } else {
		// event is known to have triggered, so these must not fail
		release_allocator.deallocate(inst->me);
		future_allocator.deallocate(inst->me);
		// see if we can reorder this (and maybe other) releases to
		//  satisfy the pending allocs
		if(attempt_release_reordering(successful_allocs)) {
		  // we'll notify the successful allocations below, after we've
		  //  released the mutex
		} else {
		  // nope, stick ourselves on the back of the (unreordered)
		  //  pending release list
		  pending_releases.push_back(PendingRelease(inst,
							    true /*triggered*/,
							    ++cur_release_seqid));
		}
	      }
	    } else {
	      // TODO: is it safe to test for failedness yet?
	      if(inst->metadata.inst_offset != RegionInstanceImpl::INSTOFFSET_FAILED)
		future_allocator.deallocate(inst->me, true /*missing ok*/);
	      pending_releases.push_back(PendingRelease(inst,
							false /*!triggered*/,
							++cur_release_seqid));
	    }
	  }
	} while(0);

	if(!successful_allocs.empty()) {
	  for(std::vector<std::pair<RegionInstanceImpl *, size_t> >::iterator it = successful_allocs.begin();
	      it != successful_allocs.end();
	      ++it) {
	    it->first->notify_allocation(ALLOC_EVENTUAL_SUCCESS,
					 it->second,
					 TimeLimit::responsive());
	  }
	}
      }

      // even if we don't apply the destruction to the heap state right away,
      //  we always ack a triggered destruction
      if(triggered) {
	inst->notify_deallocation();
      } else {
	// ask the instance to tell us when the precondition is satisified
	inst->deferred_destroy.defer(inst, this, precondition);
      }
    }

    // should only be called by RegionInstance::DeferredCreate
    MemoryImpl::AllocationResult LocalManagedMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
							 bool need_alloc_result,
							 bool poisoned,
							 TimeLimit work_until)
    {
      AllocationResult result;
      size_t inst_offset = 0;
      {
	AutoLock<> al(allocator_mutex);

	// with the lock held, check the state of the instance to see if a
	//  deferred destruction has also been received - if so, we'll have to
	//  add that to the allocator history too
	assert((inst->metadata.inst_offset == RegionInstanceImpl::INSTOFFSET_DELAYEDALLOC) ||
	       (inst->metadata.inst_offset == RegionInstanceImpl::INSTOFFSET_DELAYEDDESTROY));	       
	bool deferred_destroy_exists = (inst->metadata.inst_offset == RegionInstanceImpl::INSTOFFSET_DELAYEDDESTROY);
	inst->metadata.inst_offset = RegionInstanceImpl::INSTOFFSET_UNALLOCATED;

	if(poisoned) {
	  result = ALLOC_CANCELLED;
	  inst_offset = RegionInstanceImpl::INSTOFFSET_FAILED;
	} else {
	  if(inst->metadata.ext_resource != 0) {
	    // this is an external allocation - it had better be a memory resource
	    ExternalMemoryResource *res = dynamic_cast<ExternalMemoryResource *>(inst->metadata.ext_resource);
	    if(res != 0) {
	      // automatic success - make the "offset" be the difference between the
	      //  base address we were given and our own allocation's base
	      void *mem_base = get_direct_ptr(0, 0); // only our subclasses know this
	      assert(mem_base != 0);
	      // underflow is ok here - it'll work itself out when we add the mem_base
	      //  back in on accesses
	      inst_offset = res->base - reinterpret_cast<uintptr_t>(mem_base);
	      result = ALLOC_INSTANT_SUCCESS;
	    } else {
	      log_inst.warning() << "attempt to register non-memory resource: mem=" << me << " resource=" << *res;
	      result = ALLOC_INSTANT_FAILURE;
	    }
	  } else {
	    result = attempt_deferrable_allocation(inst,
						   inst->metadata.layout->bytes_used,
						   inst->metadata.layout->alignment_reqd,
						   inst_offset);
	  }
	}

	// success or fail, if a deferred destruction is in flight, we have to
	//  add it to our list so that we can find it later
	if(deferred_destroy_exists && (inst->metadata.ext_resource == 0)) {
	  // a successful (now or later) allocation should update the future
	  //  state, if we have one
	  if(((result == ALLOC_INSTANT_SUCCESS) || (result == ALLOC_DEFERRED)) &&
	     !pending_allocs.empty())
	    future_allocator.deallocate(inst->me, true /*missing ok*/);

	  pending_releases.push_back(PendingRelease(inst,
						    false /*!ready*/,
						    ++cur_release_seqid));
	}
      }
	    
      // if we needed an alloc result, send deferred responses too
      if((result != ALLOC_DEFERRED) || need_alloc_result) {
	inst->notify_allocation(result, inst_offset, work_until);
      }

      return result;
    }

  //define DEBUG_DEFERRED_ALLOCATIONS
#ifdef DEBUG_DEFERRED_ALLOCATIONS
  Logger log_defalloc("defalloc");

  std::ostream& operator<<(std::ostream& os, const MemoryImpl::PendingAlloc& p)
  {
    os << p.inst->me << "(" << ((void *)(p.inst)) << "," << p.bytes << "," << p.alignment << "," << p.last_release_seqid << ")";
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const MemoryImpl::PendingRelease& p)
  {
    os << p.inst->me << "(" << ((void *)(p.inst)) << "," << p.is_ready << "," << p.seqid << ")";
    return os;
  }

  template <typename T1, typename T2>
  std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& p)
  {
    os << "<" << p.first << "," << p.second << ">";
    return os;
  }
#endif

  // attempts to satisfy pending allocations based on reordering releases to
  //  move the ready ones first - assumes 'release_allocator' has been
  //  properly maintained
  bool LocalManagedMemory::attempt_release_reordering(std::vector<std::pair<RegionInstanceImpl *, size_t> >& successful_allocs)
  {
    PendingAlloc& oldest = pending_allocs.front();
    if(!release_allocator.can_allocate(oldest.inst->me,
				       oldest.bytes,
				       oldest.alignment)) {
      // nope - oldest allocation still is stuck
      return false;
    }

    std::vector<PendingAlloc>::iterator a_now = pending_allocs.begin();
    BasicRangeAllocator<size_t, RegionInstance> test_allocator = release_allocator;
    while(a_now != pending_allocs.end()) {
      size_t offset = 0;
      bool ok = test_allocator.allocate(a_now->inst->me,
					a_now->bytes,
					a_now->alignment,
					offset);
      if(ok) {
	successful_allocs.push_back(std::make_pair(a_now->inst,
						   offset));
	++a_now;
      } else
	break;
    }
    // should have gotten at least one
    assert(a_now != pending_allocs.begin());

    // did we get them all?
    if(a_now == pending_allocs.end()) {
      // life is easy - test becomes current, and we remove
      //  all ready deallocs
      current_allocator.swap(test_allocator);
      pending_allocs.clear();
      std::vector<PendingRelease>::iterator it3 = pending_releases.begin();
      while(it3 != pending_releases.end())
	if(it3->is_ready)
	  it3 = pending_releases.erase(it3);
	else
	  ++it3;

      return true;
    } else {
      // see if we can still tentatively satisfy remaining allocs
      BasicRangeAllocator<size_t, RegionInstance> test_future_allocator = test_allocator;
      std::vector<PendingAlloc>::iterator a_future = a_now;
      std::vector<PendingRelease>::iterator it3 = pending_releases.begin();
      while(a_future != pending_allocs.end()) {
	// first apply any non-ready releases older than this alloc
	while((it3 != pending_releases.end()) &&
	      (it3->seqid <= a_future->last_release_seqid)) {
	  if(!it3->is_ready)
	    test_future_allocator.deallocate(it3->inst->me,
					     true /*missing ok*/);
	  ++it3;
	}

	// and now try allocation
	size_t offset;
	bool ok = test_future_allocator.allocate(a_future->inst->me,
						 a_future->bytes,
						 a_future->alignment,
						 offset);
	if(ok)
	  ++a_future;
	else
	  break;
      }

      // did we get all the way through?
      if(a_future == pending_allocs.end()) {
	// yes - this is a viable alternate timeline

	// don't forget to apply any remaining pending releases
	while(it3 != pending_releases.end()) {
	  if(!it3->is_ready)
	    test_future_allocator.deallocate(it3->inst->me,
					     true /*missing ok*/);
	  ++it3;
	}

	// now go back through and erase any ready ones
	it3 = pending_releases.begin();
	while(it3 != pending_releases.end())
	  if(it3->is_ready)
	    it3 = pending_releases.erase(it3);
	  else
	    ++it3;

	// erase the allocations we succeeded on
	pending_allocs.erase(pending_allocs.begin(), a_now);

	current_allocator.swap(test_allocator);
	future_allocator.swap(test_future_allocator);
	// we applied all the ready releases, so current == release
	release_allocator = current_allocator;
	return true;
      } else {
	// nope - it didn't work - unwind everything and clear out
	//  the allocations we thought we could do
	successful_allocs.clear();
	return false;
      }
    }
  }
  
    // should only be called by RegionInstance::DeferredDestroy
    void LocalManagedMemory::release_storage_immediate(RegionInstanceImpl *inst,
						       bool poisoned,
						       TimeLimit work_until)
    {
      // for external instances, all we have to do is ack the destruction (assuming
      //  it wasn't poisoned)
      if(inst->metadata.ext_resource != 0) {
	if(!poisoned)
	  inst->notify_deallocation();
	return;
      }

      std::vector<std::pair<RegionInstanceImpl *, size_t> > successful_allocs;
      std::vector<RegionInstanceImpl *> failed_allocs;
      //std::vector<RegionInstanceImpl *> deallocs;
      {
	AutoLock<> al(allocator_mutex);

#ifdef DEBUG_DEFERRED_ALLOCATIONS
	log_defalloc.print() << "deferred destruction: m=" << me
			     << " inst=" << inst->me << " poisoned=" << poisoned
			     << " allocs=" << PrettyVector<PendingAlloc>(pending_allocs)
			     << " release=" << PrettyVector<PendingRelease>(pending_releases);
#endif

	// this destruction should be somewhere in the pending ops list
	assert(!pending_releases.empty());
	
	std::vector<PendingRelease>::iterator it = pending_releases.begin();

	if(!poisoned) {
	  // special case: if we're the oldest pending item (and we're not
	  //  poisoned), we unclog things in the order we planned
	  if(it->inst == inst) {
	    if(!pending_allocs.empty())
	      release_allocator.deallocate(it->inst->me);

	    // catch up the current state
	    do {
	      current_allocator.deallocate(it->inst->me);
	      //deallocs.push_back(it->inst);

	      // did this unblock any allocations?
	      std::vector<PendingAlloc>::iterator it2 = pending_allocs.begin();
	      while(it2 != pending_allocs.end()) {
		// if this alloc depends on further pending releases, we can't
		//  be sure it'll work
		{
		  std::vector<PendingRelease>::iterator next_rel = it + 1;
		  if((next_rel != pending_releases.end()) &&
		     (it2->last_release_seqid >= next_rel->seqid))
		    break;
		}

#ifdef DEBUG_REALM
		// but it should never be older than the current release
		assert(it2->last_release_seqid >= it->seqid);
#endif
		
		// all older release are done, so this alloc had better work
		//  against the current state
		size_t offset = 0;
		bool ok = current_allocator.allocate(it2->inst->me, it2->bytes,
						     it2->alignment, offset);
		assert(ok);
#ifdef DEBUG_REALM
		// it should also be where we thought it was in the future
		//  allocator state (unless it's already been future-deleted)
		size_t f_first, f_size;
		if(future_allocator.lookup(it2->inst->me, f_first, f_size)) {
		  assert((f_first == offset) && (f_size == it2->bytes));
		} else {
		  // find in future deletion list
		  std::vector<PendingRelease>::const_iterator it3 = pending_releases.begin();
		  while(true) {
		    // should not run off end of list
		    assert(it3 != pending_releases.end());
		    if(it3->inst != it2->inst) {
		      ++it3;
		    } else {
		      // found it - make sure it's not already deleted
		      assert(!it3->is_ready);
		      break;
		    }
		  }
		}
#endif
		successful_allocs.push_back(std::make_pair(it2->inst, offset));
		++it2;
	      }

	      pending_allocs.erase(pending_allocs.begin(), it2);

	      ++it;
	    } while((it != pending_releases.end()) && (it->is_ready));
	    
	    pending_releases.erase(pending_releases.begin(), it);

	    // if we did any allocations but some remain, we need to rebuild
	    //  the release_allocator state (TODO: incremental updates?)
	    if(!successful_allocs.empty() && !pending_allocs.empty()) {
	      release_allocator = current_allocator;
	      for(std::vector<PendingRelease>::iterator it = pending_releases.begin();
		  it != pending_releases.end();
		  ++it) {
		// actually, include all pending releases
		//if(it->seqid > pending_allocs.front().last_release_seqid)
		//  break;
		if(it->is_ready)
 		  release_allocator.deallocate(it->inst->me);
	      }
	    }
	  } else {
	    // find this destruction in the list and mark it ready
	    do {
	      ++it;
	      assert(it != pending_releases.end());  // can't fall off end
	    } while(it->inst != inst);
	    it->is_ready = true;

	    if(pending_allocs.empty()) {
	      // we can apply this delete to the current state
	      current_allocator.deallocate(inst->me);
              //deallocs.push_back(inst);
	      it = pending_releases.erase(it);
	    } else {
	      // apply this free to the release_allocator - we'll check below
	      //  to see if it unblocks one or more allocations AND leaves the
	      //  rest possible
	      release_allocator.deallocate(inst->me);
	    }
	  }
	
	  // a couple different ways to get to a state where the ready releases
	  //  allow allocations to proceed but we could not be sure above, so
	  //  check now
	  if(!pending_allocs.empty())
	    attempt_release_reordering(successful_allocs);
	} else {
	  // special case: if there are no pending allocation requests, we
	  //  just forget this destruction request ever happened - there is
	  //  no future state to fix up
	  if(pending_allocs.empty()) {
	    while(it->inst != inst) {
	      ++it;
	      assert(it != pending_releases.end());  // can't fall off end
	    }

	    it = pending_releases.erase(it);
	  } else {
	    // rewrite our (future) history without this allocation
	    future_allocator = current_allocator;
	    release_allocator = current_allocator;

	    std::vector<PendingAlloc>::iterator it2 = pending_allocs.begin();

	    bool found = false;
	    while(it != pending_releases.end()) {
	      // save the sequence id in case we delete the entry
	      unsigned seqid = it->seqid;
	      
	      // only eat the first matching release that we find
	      if((it->inst == inst) && !found) {
		// alternate universe starts now
		found = true;
		it = pending_releases.erase(it);
	      } else {
		// replay this destuction on the future state
		future_allocator.deallocate(it->inst->me,
					    true /*missing ok*/);
		++it;
	      }

	      // check any allocs that were waiting just on the releases we
	      //  have seen so far
	      while((it2 != pending_allocs.end()) &&
		    (it2->last_release_seqid <= seqid)) {
		size_t offset;
		bool ok = future_allocator.allocate(it2->inst->me,
						    it2->bytes, it2->alignment,
						    offset);
		if(ok) {
		  ++it2;
		} else {
		  // this should only happen if we've seen the poisoned release
		  assert(found);
		  
		  // this alloc is no longer possible - remove from the list
		  //  and notify of the failure
		  failed_allocs.push_back(it2->inst);
		  it2 = pending_allocs.erase(it2);
		}
	      }
	    }
	  }
	}

#ifdef DEBUG_DEFERRED_ALLOCATIONS
	log_defalloc.print() << "deferred destruction done: m=" << me
			     << " inst=" << inst->me << " poisoned=" << poisoned
			     << " allocs=" << PrettyVector<PendingAlloc>(pending_allocs)
			     << " release=" << PrettyVector<PendingRelease>(pending_releases)
			     << " success=" << PrettyVector<std::pair<RegionInstanceImpl *, size_t> >(successful_allocs)
			     << " failure=" << PrettyVector<RegionInstanceImpl *>(failed_allocs);
#endif
      }

      // now go through and do notifications
      if(!successful_allocs.empty())
	for(std::vector<std::pair<RegionInstanceImpl *, size_t> >::iterator it = successful_allocs.begin();
	    it != successful_allocs.end();
	    ++it) {
	  it->first->notify_allocation(ALLOC_EVENTUAL_SUCCESS, it->second,
				       work_until);
	}

      if(!failed_allocs.empty())
	for(std::vector<RegionInstanceImpl *>::iterator it = failed_allocs.begin();
	    it != failed_allocs.end();
	    ++it) {
	  (*it)->notify_allocation(ALLOC_EVENTUAL_FAILURE,
				   RegionInstanceImpl::INSTOFFSET_FAILED,
				   work_until);
	}

      // even if we don't apply the destruction to the heap state right away,
      //  we always ack a triggered (but not poisoned) destruction now
      if(!poisoned) {
	inst->notify_deallocation();
      }
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalManagedMemory::PendingAlloc
  //

  LocalManagedMemory::PendingAlloc::PendingAlloc(RegionInstanceImpl *_inst,
						 size_t _bytes, size_t _align,
						 unsigned _release_seqid)
    : inst(_inst), bytes(_bytes), alignment(_align)
    , last_release_seqid(_release_seqid)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalManagedMemory::PendingRelease
  //

  LocalManagedMemory::PendingRelease::PendingRelease(RegionInstanceImpl *_inst,
						     bool _ready, unsigned _seqid)
    : inst(_inst), is_ready(_ready), seqid(_seqid)
  {}

    
  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalCPUMemory
  //

  LocalCPUMemory::LocalCPUMemory(Memory _me, size_t _size, 
                                 int _numa_node, Memory::Kind _lowlevel_kind,
				 void *prealloc_base /*= 0*/,
				 NetworkSegment *_segment /*= 0*/)
    : LocalManagedMemory(_me, _size, MKIND_SYSMEM, ALIGNMENT,
			 _lowlevel_kind, _segment),
      numa_node(_numa_node)
  {
    if(prealloc_base) {
      base = (char *)prealloc_base;
      prealloced = true;
    } else {
      // allocate our own space
      // enforce alignment on the whole memory range
      base_orig = static_cast<char *>(malloc(_size + ALIGNMENT - 1));
      if(!base_orig) {
	log_malloc.fatal() << "insufficient system memory: "
			   << size << " bytes needed (from -ll:csize)";
	abort();
      }
      size_t ofs = reinterpret_cast<size_t>(base_orig) % ALIGNMENT;
      if(ofs > 0) {
	base = base_orig + (ALIGNMENT - ofs);
      } else {
	base = base_orig;
      }
      prealloced = false;

      // we should not have been given a NetworkSegment by our caller
      assert(!segment);
      // advertise our allocation in case the network can register it
      local_segment.assign(NetworkSegmentInfo::HostMem,
			   base, _size);
      segment = &local_segment;
    }
    log_malloc.debug("CPU memory at %p, size = %zd%s%s", base, _size, 
		     prealloced ? " (prealloced)" : "",
		     (segment && segment->single_network) ? " (registered)" : "");
  }

  LocalCPUMemory::~LocalCPUMemory(void)
  {
    if(!prealloced)
      free(base_orig);
  }

  void LocalCPUMemory::get_bytes(off_t offset, void *dst, size_t size)
  {
    memcpy(dst, base+offset, size);
  }

  void LocalCPUMemory::put_bytes(off_t offset, const void *src, size_t size)
  {
    memcpy(base+offset, src, size);
  }

  void *LocalCPUMemory::get_direct_ptr(off_t offset, size_t size)
  {
//    assert((offset >= 0) && ((size_t)(offset + size) <= this->size));
    return (base + offset);
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMemory
  //

    RemoteMemory::RemoteMemory(Memory _me, size_t _size, Memory::Kind k,
			       MemoryKind mk /*= MKIND_REMOTE */)
      : MemoryImpl(_me, _size, mk, k, nullptr /*no segment*/)
    {}

    RemoteMemory::~RemoteMemory(void)
    {}

    // forward the requests immediately and let the owner node handle deferrals
    MemoryImpl::AllocationResult RemoteMemory::allocate_storage_deferrable(RegionInstanceImpl *inst,
									   bool need_alloc_result,
									   Event precondition)
    {
      NodeID target = ID(me).memory_owner_node();
      assert(target != Network::my_node_id);

      // we need to send the layout information to the memory's owner node - see
      //  how big that'll be
      Serialization::ByteCountSerializer bcs;
      bool ok = bcs << *inst->metadata.layout;
      if(ok && (inst->metadata.ext_resource != 0))
	ok = bcs << *inst->metadata.ext_resource;
      assert(ok);
      size_t layout_bytes = bcs.bytes_used();
      
      ActiveMessage<MemStorageAllocRequest> amsg(target, layout_bytes);
      amsg->memory = me;
      amsg->inst = inst->me;
      amsg->need_alloc_result = need_alloc_result;
      amsg->precondition = precondition;
      amsg << *inst->metadata.layout;
      if(inst->metadata.ext_resource != 0)
	amsg << *inst->metadata.ext_resource;
      amsg.commit();
      return ALLOC_DEFERRED /*asynchronous notification*/;
    }

    // release storage associated with an instance
    void RemoteMemory::release_storage_deferrable(RegionInstanceImpl *inst,
						  Event precondition)
    {
      NodeID target = ID(me).memory_owner_node();
      assert(target != Network::my_node_id);
      
      ActiveMessage<MemStorageReleaseRequest> amsg(target);
      amsg->memory = me;
      amsg->inst = inst->me;
      amsg->precondition = precondition;
      amsg.commit();
    }

    MemoryImpl::AllocationResult RemoteMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
						   bool need_alloc_result,
						   bool poisoned, TimeLimit work_until)
    {
      // actual allocation/release should always happen on the owner node
      abort();
    }

    void RemoteMemory::release_storage_immediate(RegionInstanceImpl *inst,
						 bool poisoned, TimeLimit work_until)
    {
      // actual allocation/release should always happen on the owner node
      abort();
    }

    off_t RemoteMemory::alloc_bytes_local(size_t size)
    {
      assert(0);
      return 0;
    }

    void RemoteMemory::free_bytes_local(off_t offset, size_t size)
    {
      assert(0);
    }

    void RemoteMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // can't read/write a remote memory
      assert(0);
    }

    void RemoteMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // can't read/write a remote memory
      assert(0);
    }

    void *RemoteMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageAllocRequest
  //

  /*static*/ void MemStorageAllocRequest::handle_message(NodeID sender, const MemStorageAllocRequest &args,
							 const void *data, size_t datalen)
  {
    MemoryImpl *impl = get_runtime()->get_memory_impl(args.memory);
    RegionInstanceImpl *inst = impl->get_instance(args.inst);

    // deserialize the layout
    Serialization::FixedBufferDeserializer fbd(data, datalen);
    InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::deserialize_new(fbd);
    assert(ilg != 0);
    ExternalInstanceResource *res = 0;
    if(fbd.bytes_left() > 0) {
      res = ExternalInstanceResource::deserialize_new(fbd);
      assert((res != 0) && (fbd.bytes_left() == 0));
    }
    inst->metadata.layout = ilg;  // TODO: mark metadata valid?
    inst->metadata.ext_resource = res;

    impl->allocate_storage_deferrable(inst,
				      args.need_alloc_result,
				      args.precondition);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageAllocResponse
  //

  /*static*/ void MemStorageAllocResponse::handle_message(NodeID sender, const MemStorageAllocResponse &args,
							  const void *data, size_t datalen,
							  TimeLimit work_until)
  {
    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.inst);

    impl->notify_allocation(args.result, args.offset, work_until);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageReleaseRequest
  //

  /*static*/ void MemStorageReleaseRequest::handle_message(NodeID sender, const MemStorageReleaseRequest &args,
							   const void *data, size_t datalen)
  {
    MemoryImpl *impl = get_runtime()->get_memory_impl(args.memory);
    RegionInstanceImpl *inst = impl->get_instance(args.inst);

    impl->release_storage_deferrable(inst,
				     args.precondition);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageReleaseResponse
  //

  /*static*/ void MemStorageReleaseResponse::handle_message(NodeID sender, const MemStorageReleaseResponse &args,
							    const void *data, size_t datalen)
  {
    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.inst);

    impl->notify_deallocation();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteWriteMessage
  //

  struct PartialWriteKey {
    unsigned sender;
    unsigned sequence_id;
    bool operator<(const PartialWriteKey& rhs) const
    {
      if(sender < rhs.sender) return true;
      if(sender > rhs.sender) return false;
      return sequence_id < rhs.sequence_id;
    }
  };

  struct PartialWriteEntry {
    RemoteWriteFence *fence;
    int remaining_count;
  };

  typedef std::map<PartialWriteKey, PartialWriteEntry> PartialWriteMap;
  static PartialWriteMap partial_remote_writes;
  static Mutex partial_remote_writes_lock;

  /*static*/ void RemoteWriteMessage::handle_message(NodeID sender, const RemoteWriteMessage &args,
						     const void *data,
						     size_t datalen)
  {
    MemoryImpl *impl = get_runtime()->get_memory_impl(args.mem);

    log_copy.debug() << "received remote write request: mem=" << args.mem
		     << ", offset=" << args.offset << ", size=" << datalen
		     << ", seq=" << sender << '/' << args.sequence_id;
#ifdef DEBUG_REMOTE_WRITES
    printf("received remote write request: mem=" IDFMT ", offset=%zd, size=%zd, seq=%d/%d",
	   args.mem.id, args.offset, datalen,
	   sender, args.sequence_id);
    printf("  data[%p]: %08x %08x %08x %08x %08x %08x %08x %08x\n",
	   data,
	   ((unsigned *)(data))[0], ((unsigned *)(data))[1],
	   ((unsigned *)(data))[2], ((unsigned *)(data))[3],
	   ((unsigned *)(data))[4], ((unsigned *)(data))[5],
	   ((unsigned *)(data))[6], ((unsigned *)(data))[7]);
#endif

    // determine if the active message already wrote the data to the
    //  right place in a registered memory
    bool was_written_directly = false;
    if(impl->kind == MemoryImpl::MKIND_SYSMEM) {
      LocalCPUMemory *cpumem = (LocalCPUMemory *)impl;
      if(data == (cpumem->base + args.offset)) {
	// copy is in right spot - yay!
	was_written_directly = true;
      } else {
#if 0
	// TODO: restore this error check?
	if(!cpumem->registered) {
	  log_copy.error() << "received remote write to registered memory in wrong spot: "
			   << data << " != "
			   << ((void *)(cpumem->base)) << "+" << args.offset
			   << " = " << ((void *)(cpumem->base + args.offset));
	}
#endif
      }
    }

    // if it wasn't directly written, we have to copy it
    if(!was_written_directly)
      impl->put_bytes(args.offset, data, datalen);

    // track the sequence ID to know when the full RDMA is done
    if(args.sequence_id > 0) {
      PartialWriteKey key;
      key.sender = sender;
      key.sequence_id = args.sequence_id;
      partial_remote_writes_lock.lock();
      PartialWriteMap::iterator it = partial_remote_writes.find(key);
      if(it == partial_remote_writes.end()) {
	// first reference to this one
	PartialWriteEntry entry;
	entry.fence = 0;
	entry.remaining_count = -1;
	partial_remote_writes[key] = entry;
#ifdef DEBUG_PWT
	printf("PWT: %d: new entry for %d/%d: %p, %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence,
	       entry.remaining_count, entry.remaining_count - 1);
#endif
	entry.remaining_count--;
	if(entry.remaining_count == 0) {
	  // we're the last write, and we've already got the fence, so 
	  //  respond
	  ActiveMessage<RemoteWriteFenceAckMessage> amsg(sender);
	  amsg->fence = entry.fence;
	  amsg.commit();
	  partial_remote_writes.erase(it);
	  partial_remote_writes_lock.unlock();
	  return;
	}
      }
      partial_remote_writes_lock.unlock();
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteSerdezMessage
  //
  /*static*/ void RemoteSerdezMessage::handle_message(NodeID sender, const RemoteSerdezMessage &args,
                                                      const void *data,
                                                      size_t datalen)
  {
    log_copy.debug() << "received remote serdez request: mem=" << args.mem
		     << ", offset=" << args.offset << ", size=" << datalen
		     << ", seq=" << sender << '/' << args.sequence_id;

    const CustomSerdezUntyped *serdez_op = get_runtime()->custom_serdez_table.get(args.serdez_id, 0);
    assert(serdez_op != 0);
    size_t field_size = serdez_op->sizeof_field_type;
    char* pos = (char*)get_runtime()->get_memory_impl(args.mem)->get_direct_ptr(args.offset, args.count * serdez_op->sizeof_field_type);
    const char* buffer = (const char*) data;
    for(size_t i = 0; i < args.count; i++) {
      size_t elemnt_size = serdez_op->deserialize(pos, buffer);
      buffer += elemnt_size;
      pos+= field_size;
      datalen -= elemnt_size;
    }
    assert(datalen == 0);

    // track the sequence ID to know when the full RDMA is done
    if(args.sequence_id > 0) {
      PartialWriteKey key;
      key.sender = sender;
      key.sequence_id = args.sequence_id;
      partial_remote_writes_lock.lock();
      PartialWriteMap::iterator it = partial_remote_writes.find(key);
      if(it == partial_remote_writes.end()) {
	// first reference to this one
	PartialWriteEntry entry;
	entry.fence = 0;
	entry.remaining_count = -1;
	partial_remote_writes[key] = entry;
#ifdef DEBUG_PWT
	printf("PWT: %d: new entry for %d/%d: %p, %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence,
	       entry.remaining_count, entry.remaining_count - 1);
#endif
	entry.remaining_count--;
	if(entry.remaining_count == 0) {
	  // we're the last write, and we've already got the fence, so
	  //  respond
	  ActiveMessage<RemoteWriteFenceAckMessage> amsg(sender);
	  amsg->fence = entry.fence;
	  amsg.commit();
	  partial_remote_writes.erase(it);
	  partial_remote_writes_lock.unlock();
	  return;
	}
      }
      partial_remote_writes_lock.unlock();
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteReduceMessage
  //

  /*static*/ void RemoteReduceMessage::handle_message(NodeID sender, const RemoteReduceMessage &args,
						      const void *data,
						      size_t datalen)
  {
    ReductionOpID redop_id;
    bool red_fold;
    if(args.redop_id > 0) {
      redop_id = args.redop_id;
      red_fold = false;
    } else if(args.redop_id < 0) {
      redop_id = -args.redop_id;
      red_fold = true;
    } else {
      assert(args.redop_id != 0);
      return;
    }

    log_copy.debug("received remote reduce request: mem=" IDFMT ", offset=%zd+%zd, size=%zd, redop=%d(%s), seq=%d/%d",
		   args.mem.id, (ssize_t)args.offset, (ssize_t)args.stride, datalen,
		   redop_id, (red_fold ? "fold" : "apply"),
		   sender, args.sequence_id);

    const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
    assert(redop != 0);

    size_t count = datalen / redop->sizeof_rhs;

    MemoryImpl *m_impl = get_runtime()->get_memory_impl(args.mem);
    void *lhs = m_impl->get_direct_ptr(args.offset, args.stride * count);
    if(lhs && (m_impl->kind != MemoryImpl::MKIND_GPUFB)) {
      // directly apply/fold to memory contents
      if(red_fold)
	redop->fold_strided(lhs, data,
			    args.stride, redop->sizeof_rhs,
			    count, false /*not exclusive*/);
      else
	redop->apply_strided(lhs, data, 
			     args.stride, redop->sizeof_rhs,
			     count, false /*not exclusive*/);
    } else {
      // need to use an intermediate buffer in sysmem
      size_t MAX_BUFFER_SIZE = 1 << 20; // 1MB
      size_t lhs_size = (red_fold ? redop->sizeof_rhs : redop->sizeof_lhs);
      size_t max_chunk_count = ((lhs_size >= MAX_BUFFER_SIZE) ?
				  1 :
				(  MAX_BUFFER_SIZE / lhs_size));
      char *lhs_buffer = reinterpret_cast<char *>(malloc(lhs_size *
							 max_chunk_count));
      assert(lhs_buffer != 0);
      size_t chunk_start = 0;
      while(chunk_start < count) {
	size_t chunk_size = std::min(count - chunk_start, max_chunk_count);
	if(size_t(args.stride) == lhs_size) {
	  m_impl->get_bytes(args.offset + (chunk_start * args.stride),
			    lhs_buffer,
			    lhs_size * chunk_size);
	} else {
	  // no 2d version of get_bytes
	  for(size_t i = 0; i < chunk_size; i++)
	    m_impl->get_bytes(args.offset + ((chunk_start + i) * args.stride),
			      lhs_buffer + (i * lhs_size),
			      lhs_size);
	}

	// apply/fold to local buffer
	if(red_fold)
	  redop->fold_strided(lhs_buffer,
			      reinterpret_cast<const char *>(data) + (chunk_start * redop->sizeof_rhs),
			      lhs_size, redop->sizeof_rhs,
			      chunk_size, true /*exclusive*/);
	else
	  redop->apply_strided(lhs_buffer,
			       reinterpret_cast<const char *>(data) + (chunk_start * redop->sizeof_rhs),
			       lhs_size, redop->sizeof_rhs,
			       chunk_size, true /*exclusive*/);

	// and copy updated bytes back
	if(size_t(args.stride) == lhs_size) {
	  m_impl->put_bytes(args.offset + (chunk_start * args.stride),
			    lhs_buffer,
			    lhs_size * chunk_size);
	} else {
	  // no 2d version of put_bytes
	  for(size_t i = 0; i < chunk_size; i++)
	    m_impl->put_bytes(args.offset + ((chunk_start + i) * args.stride),
			      lhs_buffer + (i * lhs_size),
			      lhs_size);
	}

	chunk_start += chunk_size;
      }

      free(lhs_buffer);
    }

    // track the sequence ID to know when the full RDMA is done
    if(args.sequence_id > 0) {
      PartialWriteKey key;
      key.sender = sender;
      key.sequence_id = args.sequence_id;
      partial_remote_writes_lock.lock();
      PartialWriteMap::iterator it = partial_remote_writes.find(key);
      if(it == partial_remote_writes.end()) {
	// first reference to this one
	PartialWriteEntry entry;
	entry.fence = 0;
	entry.remaining_count = -1;
	partial_remote_writes[key] = entry;
#ifdef DEBUG_PWT
	printf("PWT: %d: new entry for %d/%d: %p, %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence,
	       entry.remaining_count, entry.remaining_count - 1);
#endif
	entry.remaining_count--;
	if(entry.remaining_count == 0) {
	  // we're the last write, and we've already got the fence, so 
	  //  respond
	  ActiveMessage<RemoteWriteFenceAckMessage> amsg(sender);
	  amsg->fence = entry.fence;
	  amsg.commit();
	  partial_remote_writes.erase(it);
	  partial_remote_writes_lock.unlock();
	  return;
	}
      }
      partial_remote_writes_lock.unlock();
    }
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteWriteFence
  //

  RemoteWriteFence::RemoteWriteFence(Operation *op)
    : Operation::AsyncWorkItem(op)
  {}

  void RemoteWriteFence::request_cancellation(void)
  {
    // ignored
  }

  void RemoteWriteFence::print(std::ostream& os) const
  {
    os << "RemoteWriteFence";
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteWriteFenceMessage
  //

  /*static*/ void RemoteWriteFenceMessage::handle_message(NodeID sender, const RemoteWriteFenceMessage &args,
							  const void *data, size_t datalen)
  {
    log_copy.debug("remote write fence (mem = " IDFMT ", seq = %d/%d, count = %d, fence = %p",
		   args.mem.id, sender, args.sequence_id, args.num_writes, args.fence);
    
    assert(args.sequence_id != 0);
    // track the sequence ID to know when the full RDMA is done
    if(args.sequence_id > 0) {
      PartialWriteKey key;
      key.sender = sender;
      key.sequence_id = args.sequence_id;
      partial_remote_writes_lock.lock();
      PartialWriteMap::iterator it = partial_remote_writes.find(key);
      if(it == partial_remote_writes.end()) {
	// first reference to this one
	PartialWriteEntry entry;
	entry.fence = args.fence;
	entry.remaining_count = args.num_writes;
	partial_remote_writes[key] = entry;
#ifdef DEBUG_PWT
	printf("PWT: %d: new entry for %d/%d: %p, %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (previous writes)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p -> %p, %d -> %d\n",
	       Network::my_node_id, key.sender, key.sequence_id,
	       entry.fence, args.fence,
	       entry.remaining_count, entry.remaining_count + args.num_writes);
#endif
        assert(entry.fence == 0);
	entry.fence = args.fence;
	entry.remaining_count += args.num_writes;
	// a negative remaining count means we got too many writes!
	assert(entry.remaining_count >= 0);
	if(entry.remaining_count == 0) {
	  // this fence came after all the writes, so respond
	  ActiveMessage<RemoteWriteFenceAckMessage> amsg(sender);
	  amsg->fence = entry.fence;
	  amsg.commit();
	  partial_remote_writes.erase(it);
	  partial_remote_writes_lock.unlock();
	  return;
	}
      }
      partial_remote_writes_lock.unlock();
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteWriteFenceAckMessage
  //

  /*static*/ void RemoteWriteFenceAckMessage::handle_message(NodeID sender,
							     const RemoteWriteFenceAckMessage &args,
							     const void *data, size_t datalen)
  {
    log_copy.debug("remote write fence ack: fence = %p",
		   args.fence);

    args.fence->mark_finished(true /*successful*/);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // do_remote_*
  //

#if 0
    unsigned do_remote_write(Memory mem, off_t offset,
			     const void *data, size_t datalen,
			     unsigned sequence_id,
			     bool make_copy /*= false*/)
    {
      log_copy.debug("sending remote write request: mem=" IDFMT ", offset=%zd, size=%zd",
		     mem.id, (ssize_t)offset, datalen);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(mem);
      char *dstptr;
      if(m_impl->kind == MemoryImpl::MKIND_RDMA) {
	dstptr = ((char *)(((RemoteMemory *)m_impl)->regbase)) + offset;
	//printf("remote mem write to rdma'able memory: dstptr = %p\n", dstptr);
      } else
	dstptr = 0;
      assert(datalen > 0);

      // if we don't have a destination pointer, we need to use the LMB, which
      //  may require chopping this request into pieces
      if(!dstptr) {
	size_t max_xfer_size = get_lmb_size(ID(mem).memory_owner_node());

	if(datalen > max_xfer_size) {
	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;

	  int count = 1;
	  while(datalen > max_xfer_size) {
	    ActiveMessage<RemoteWriteMessage> amsg(ID(mem).memory_owner_node());
	    amsg->mem = mem;
	    amsg->offset = offset;
	    amsg->sequence_id = sequence_id;
	    amsg.add_payload(pos,max_xfer_size, make_copy? PAYLOAD_COPY: PAYLOAD_KEEP);
	    amsg.commit();
	    offset += max_xfer_size;
	    pos += max_xfer_size;
	    datalen -= max_xfer_size;
	    count++;
	  }

	  // last send includes whatever's left
	  ActiveMessage<RemoteWriteMessage> amsg(ID(mem).memory_owner_node());
	  amsg->mem = mem;
	  amsg->offset = offset;
	  amsg->sequence_id = sequence_id;
	  amsg.add_payload(pos,datalen,make_copy? PAYLOAD_COPY: PAYLOAD_KEEP);
	  amsg.commit();
	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	ActiveMessage<RemoteWriteMessage> amsg(ID(mem).memory_owner_node(), datalen, dstptr);
	amsg->mem = mem;
	amsg->offset = offset;
	amsg->sequence_id = sequence_id;
	amsg.add_payload(data,datalen,make_copy? PAYLOAD_COPY: PAYLOAD_KEEP);
	amsg.commit();
	return 1;
      }
    }

    unsigned do_remote_write(Memory mem, off_t offset,
			     const void *data, size_t datalen,
			     off_t stride, size_t lines,
			     unsigned sequence_id,
			     bool make_copy /*= false*/)
    {
      log_copy.debug("sending remote write request: mem=" IDFMT ", offset=%zd, size=%zdx%zd",
		     mem.id, (ssize_t)offset, datalen, lines);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(mem);
      char *dstptr;
      if(m_impl->kind == MemoryImpl::MKIND_RDMA) {
	dstptr = ((char *)(((RemoteMemory *)m_impl)->regbase)) + offset;
	//printf("remote mem write to rdma'able memory: dstptr = %p\n", dstptr);
      } else
	dstptr = 0;

      // if we don't have a destination pointer, we need to use the LMB, which
      //  may require chopping this request into pieces
      if(!dstptr) {
	size_t max_xfer_size = get_lmb_size(ID(mem).memory_owner_node());
	size_t max_lines_per_xfer = max_xfer_size / datalen;
	assert(max_lines_per_xfer > 0);

	if(lines > max_lines_per_xfer) {
	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;
	  int count = 1;
	  while(lines > max_lines_per_xfer) {
	    size_t payload_size = datalen*max_lines_per_xfer;
	    ActiveMessage<RemoteWriteMessage> amsg(ID(mem).memory_owner_node(), payload_size, dstptr);
	    amsg->mem = mem;
	    amsg->offset = offset;
	    amsg->sequence_id = sequence_id;
	    if (payload_size) {
	      amsg.add_payload(pos, datalen, max_lines_per_xfer, stride,
			       (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP));
	    }
	    amsg.commit();
	    offset += datalen * max_lines_per_xfer;
	    pos += stride * max_lines_per_xfer;
	    lines -= max_lines_per_xfer;
	    count++;
	  }

	  // last send includes whatever's left
	  size_t payload_size = datalen*lines;
	  ActiveMessage<RemoteWriteMessage> amsg(ID(mem).memory_owner_node(), payload_size, dstptr);
	  amsg->mem = mem;
	  amsg->offset = offset;
	  amsg->sequence_id = sequence_id;
	  if (payload_size) {
	    amsg.add_payload(pos, datalen, lines, stride,
			     (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP));
	  }
	  amsg.commit();
	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	size_t payload_size = datalen*lines;
	ActiveMessage<RemoteWriteMessage> amsg(ID(mem).memory_owner_node(), payload_size, dstptr);
	amsg->mem = mem;
	amsg->offset = offset;
	amsg->sequence_id = sequence_id;
	if (payload_size) {
	  amsg.add_payload(data, datalen, lines, stride,
			   (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP));
	}
	amsg.commit();
	return 1;
      }
    }

#if 0
    unsigned do_remote_write(Memory mem, off_t offset,
			     const SpanList &spans, size_t datalen,
			     unsigned sequence_id,
			     bool make_copy /*= false*/)
    {
      log_copy.debug("sending remote write request: mem=" IDFMT ", offset=%zd, size=%zd(%zd spans)",
		     mem.id, (ssize_t)offset, datalen, spans.size());

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(mem);
      char *dstptr;
      if(m_impl->kind == MemoryImpl::MKIND_RDMA) {
	dstptr = ((char *)(((RemoteMemory *)m_impl)->regbase)) + offset;
	//printf("remote mem write to rdma'able memory: dstptr = %p\n", dstptr);
      } else
	dstptr = 0;

      // if we don't have a destination pointer, we need to use the LMB, which
      //  may require chopping this request into pieces
      if(!dstptr) {
	size_t max_xfer_size = get_lmb_size(ID(mem).memory_owner_node());

	if(datalen > max_xfer_size) {
	  log_copy.info("breaking large send into pieces");

	  int count = 0;
	  // this is trickier because we don't actually know how much will fit
	  //  in each transfer
	  SpanList::const_iterator it = spans.begin();
	  while(datalen > 0) {
	    // possible special case - if the first span is too big to fit at
	    //   all, chop it up and send it
	    assert(it != spans.end());
	    if(it->second > max_xfer_size) {
              const char *pos = (const char *)(it->first);
              size_t left = it->second;
              while(left > max_xfer_size) {
		ActiveMessage<RemoteWriteMessage> amsg1(ID(mem).memory_owner_node(), max_xfer_size);
		amsg1->mem = mem;
		amsg1->offset = offset;
		amsg1->sequence_id = sequence_id;
		amsg1.add_payload(pos, max_xfer_size, make_copy? PAYLOAD_COPY: PAYLOAD_KEEP);
		amsg1.commit();
		offset += max_xfer_size;
		pos += max_xfer_size;
		left -= max_xfer_size;
		count++;
	      }
	      ActiveMessage<RemoteWriteMessage> amsg(ID(mem).memory_owner_node(), left);
	      amsg->mem = mem;
	      amsg->offset = offset;
	      amsg->sequence_id = sequence_id;
	      amsg.add_payload(pos, left, make_copy? PAYLOAD_COPY: PAYLOAD_KEEP);
	      amsg.commit();
	      offset += left;
	      count++;
	      datalen -= it->second;
	      it++;
	      continue;
	    }

	    // take spans in order until we run out of space or spans
	    SpanList subspans;
	    size_t xfer_size = 0;
	    while(it != spans.end()) {
	      // can we fit the next one?
	      if((xfer_size + it->second) > max_xfer_size) break;

	      subspans.push_back(*it);
	      xfer_size += it->second;
	      it++;
	    }
	    // if we didn't get at least one span, we won't make forward progress
	    assert(!subspans.empty());
	    ActiveMessage<RemoteWriteMessage> amsg2(ID(mem).memory_owner_node(), xfer_size);
	    amsg2->mem = mem;
	    amsg2->offset = offset;
	    amsg2->sequence_id = sequence_id;
	    if (xfer_size) {
	      PayloadSource *payload_src = new SpanPayload(subspans, xfer_size,PAYLOAD_COPY);
	      payload_src->copy_data(amsg2.payload_ptr(xfer_size));
	    }
	    amsg2.commit();
	    offset += xfer_size;
	    datalen -= xfer_size;
	    count++;
	  }

	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	ActiveMessage<RemoteWriteMessage> amsg2(ID(mem).memory_owner_node(), datalen, dstptr);
	amsg2->mem = mem;
	amsg2->offset = offset;
	amsg2->sequence_id = sequence_id;
	if (datalen) {
	  PayloadSource *payload_src = new SpanPayload(spans, datalen, PAYLOAD_COPY);
	  payload_src->copy_data(amsg2.payload_ptr(datalen));
	}
	amsg2.commit();
	return 1;
      }
    }
#endif

    unsigned do_remote_serdez(Memory mem, off_t offset,
                             CustomSerdezID serdez_id,
                             const void *data, size_t count,
                             unsigned sequence_id)
    {
      const CustomSerdezUntyped *serdez_op = get_runtime()->custom_serdez_table.get(serdez_id, 0);
      assert(serdez_op != 0);
      size_t field_size = serdez_op->sizeof_field_type;
      log_copy.debug("sending remote serdez request: mem=" IDFMT ", offset=%zd, size=%zdx%zd, serdez_id=%d",
                     mem.id, (ssize_t)offset, field_size, count, serdez_id);
      size_t max_xfer_size = get_lmb_size(ID(mem).memory_owner_node());
      // create a intermediate buf with same size as max_xfer_size
      char* buffer_start = (char*) malloc(max_xfer_size);
      const char *pos = (const char *)data;
      unsigned xfers = 0;

      size_t element_size = 0;
      while (count > 0) {
        size_t cur_size = 0;
        size_t cur_count = 0;
        char* buffer = buffer_start;
        off_t new_offset = offset;
        while (count > 0) {
          element_size = serdez_op->serialized_size(pos);
          // break if including this element exceeds max_xfer_size
          if (element_size + cur_size > max_xfer_size)
            break;
          count--;
          cur_count++;
          serdez_op->serialize(pos, buffer);
          pos += field_size;
          new_offset += field_size;
          buffer += element_size;
          cur_size += element_size;
        }
        if (cur_size == 0) {
          if (count == 0) {
            // No elements to serialize
            log_copy.error() << "In performing remote serdez request "
                             << "(serdez_id=" << serdez_id << "): "
                             << "No elements to serialize";
          } else if (cur_count == 0) {
            // Individual serialized element size greater than lmb buffer
            log_copy.error() << "In performing remote serdez request "
                             << "(serdez_id=" << serdez_id << "): "
                             << "Serialized size of custom serdez type (" << element_size << " bytes) "
                             << "exceeds size of the LMB buffer (" << max_xfer_size << " bytes). Try "
                             << "increasing the LMB buffer size using "
                             << "-ll:lmbsize <kbytes>";
          } else {
            // No element wrote data
            log_copy.error() << "In performing remote serdez request "
                             << "(serdez_id=" << serdez_id << "): "
                             << "No serialized element wrote data";
          }
          assert(cur_size > 0);
        }
	ActiveMessage<RemoteSerdezMessage> amsg(ID(mem).memory_owner_node(), cur_size);
	amsg->mem = mem;
	amsg->offset = offset;
	amsg->count = cur_count;
	amsg->serdez_id = serdez_id;
	amsg->sequence_id = sequence_id;
	amsg.add_payload(buffer_start, cur_size, PAYLOAD_COPY); // TODO: copy directly
	amsg.commit();
        offset = new_offset;
        xfers ++;
      }
      free(buffer_start);
      return xfers;
    }
#endif
    unsigned do_remote_reduce(Memory mem, off_t offset,
			      ReductionOpID redop_id, bool red_fold,
			      const void *data, size_t count,
			      off_t src_stride, off_t dst_stride,
			      unsigned sequence_id,
			      bool make_copy /*= false*/)
    {
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      assert(redop != 0);
      size_t rhs_size = redop->sizeof_rhs;

      log_copy.debug("sending remote reduction request: mem=" IDFMT ", offset=%zd+%zd, size=%zdx%zd, redop=%d(%s)",
		     mem.id, (ssize_t)offset, (ssize_t)dst_stride, rhs_size, count,
		     redop_id, (red_fold ? "fold" : "apply"));

      // reductions always have to bounce off an intermediate buffer, so are subject to
      //  LMB limits
      {
	// HACK: we don't expose get_lmb_size any more, but reductions should
	//  go through dma channels soon anyway
	size_t max_xfer_size = 16384;
	//size_t max_xfer_size = get_lmb_size(ID(mem).memory_owner_node());
	size_t max_elmts_per_xfer = max_xfer_size / rhs_size;
	assert(max_elmts_per_xfer > 0);

	if(count > max_elmts_per_xfer) {
	  log_copy.info("breaking large reduction into pieces");
	  const char *pos = (const char *)data;
	  int xfers = 1;
	  while(count > max_elmts_per_xfer) {
	    size_t payload_size = rhs_size*max_elmts_per_xfer;
	    ActiveMessage<RemoteReduceMessage> amsg(ID(mem).memory_owner_node(), payload_size);
	    if (payload_size) {
	      amsg.add_payload(pos, rhs_size, max_elmts_per_xfer, src_stride,
			       (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP));
	    }
	    amsg->mem = mem;
	    amsg->offset = offset;
	    amsg->stride = dst_stride;
	    amsg->redop_id = red_fold ? -redop_id : redop_id;
	    amsg->sequence_id = sequence_id;
	    amsg.commit();
	    offset += dst_stride * max_elmts_per_xfer;
	    pos += src_stride * max_elmts_per_xfer;
	    count -= max_elmts_per_xfer;
	    xfers++;
	  }

	  // last send includes whatever's left
	  size_t payload_size = rhs_size*count;
	  ActiveMessage<RemoteReduceMessage> amsg1(ID(mem).memory_owner_node(), payload_size);
	  amsg1->mem = mem;
	  amsg1->offset = offset;
	  amsg1->stride = dst_stride;
	  amsg1->redop_id = red_fold ? -redop_id : redop_id;
	  amsg1->sequence_id = sequence_id;
	  if (payload_size) {
	    amsg1.add_payload(pos, rhs_size, count, src_stride,
			      (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP));
	  }
	  amsg1.commit();
	  return xfers;
	}
      }

      // we get here with a write smaller than the LMB
      {
	size_t payload_size = rhs_size*count;
	ActiveMessage<RemoteReduceMessage> amsg1(ID(mem).memory_owner_node(), payload_size);
	amsg1->mem = mem;
	amsg1->offset = offset;
	amsg1->stride = dst_stride;
	amsg1->redop_id = red_fold ? -redop_id : redop_id;
	amsg1->sequence_id = sequence_id;
	if (payload_size) {
	  amsg1.add_payload(data, rhs_size, count, src_stride,
			    (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP));
	}
	amsg1.commit();
	return 1;
      }
    }

    void do_remote_fence(Memory mem, unsigned sequence_id, unsigned num_writes,
                         RemoteWriteFence *fence)
    {
      // technically we could handle a num_writes == 0 case, but since it's
      //  probably indicative of badness elsewhere, barf on it for now
      assert(num_writes > 0);

      ActiveMessage<RemoteWriteFenceMessage> amsg(ID(mem).memory_owner_node());
      amsg->mem = mem;
      amsg->sequence_id = sequence_id;
      amsg->num_writes = num_writes;
      amsg->fence = fence;
      amsg.commit();
    }

  ActiveMessageHandlerReg<RemoteWriteMessage> remote_write_message_handler;
  ActiveMessageHandlerReg<RemoteReduceMessage> remote_reduce_message_handler;
  ActiveMessageHandlerReg<RemoteSerdezMessage> remote_serdez_message_handler;
  ActiveMessageHandlerReg<RemoteWriteFenceMessage> remote_write_fence_message_handler;
  ActiveMessageHandlerReg<RemoteWriteFenceAckMessage> remote_write_fence_ack_message_handler;

  ActiveMessageHandlerReg<MemStorageAllocRequest> mem_storage_alloc_request_handler;
  ActiveMessageHandlerReg<MemStorageAllocResponse> mem_storage_alloc_response_handler;
  ActiveMessageHandlerReg<MemStorageReleaseRequest> mem_storage_release_request_handler;
  ActiveMessageHandlerReg<MemStorageReleaseResponse> mem_storage_release_response_handler;

}; // namespace Realm
