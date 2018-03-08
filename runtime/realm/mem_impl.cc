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

#include "realm/mem_impl.h"

#include "realm/proc_impl.h"
#include "realm/logging.h"
#include "realm/serialize.h"
#include "realm/inst_impl.h"
#include "realm/runtime_impl.h"
#include "realm/profiling.h"
#include "realm/utils.h"

#ifdef USE_GASNET
#ifndef GASNET_PAR
#define GASNET_PAR
#endif
#include <gasnet.h>
// eliminate GASNet warnings for unused static functions
static const void *ignore_gasnet_warning1 __attribute__((unused)) = (void *)_gasneti_threadkey_init;
#ifdef _INCLUDED_GASNET_TOOLS_H
static const void *ignore_gasnet_warning2 __attribute__((unused)) = (void *)_gasnett_trace_printf_noop;
#endif
#endif

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

namespace Realm {

  Logger log_malloc("malloc");
  Logger log_copy("copy");
  extern Logger log_inst; // in inst_impl.cc



  ////////////////////////////////////////////////////////////////////////
  //
  // class Memory
  //

    AddressSpace Memory::address_space(void) const
    {
      // this is a hack for the Legion runtime
      ID id(*this);
      unsigned n = id.memory.owner_node;
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

    MemoryImpl::MemoryImpl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment, Memory::Kind _lowlevel_kind)
      : me(_me), size(_size), kind(_kind), alignment(_alignment), lowlevel_kind(_lowlevel_kind)
      , usage(stringbuilder() << "realm/mem " << _me << "/usage")
      , peak_usage(stringbuilder() << "realm/mem " << _me << "/peak_usage")
      , peak_footprint(stringbuilder() << "realm/mem " << _me << "/peak_footprint")
    {
      allocator.add_range(0, _size);
    }

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

#ifdef REALM_PROFILE_MEMORY_USAGE
      printf("Memory " IDFMT " usage: peak=%zd (%.1f MB) footprint=%zd (%.1f MB)\n",
	     me.id, 
	     (size_t)peak_usage, peak_usage / 1048576.0,
	     (size_t)peak_footprint, peak_footprint / 1048576.0);
#endif
    }

    // make bad offsets really obvious (+1 PB)
    static const off_t ZERO_SIZE_INSTANCE_OFFSET = 1ULL << ((sizeof(off_t) == 8) ? 50 : 30);

    off_t MemoryImpl::alloc_bytes_local(size_t size)
    {
      AutoHSLLock al(mutex);

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
      AutoHSLLock al(mutex);

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

    off_t MemoryImpl::alloc_bytes_remote(size_t size)
    {
      // RPC over to owner's node for allocation
      return RemoteMemAllocRequest::send_request(ID(me).memory.owner_node, me, size);
    }

    void MemoryImpl::free_bytes_remote(off_t offset, size_t size)
    {
      assert(0);
    }

    Memory::Kind MemoryImpl::get_kind(void) const
    {
      return lowlevel_kind;
    }

    RegionInstanceImpl *MemoryImpl::get_instance(RegionInstance i)
    {
      ID id(i);
      assert(id.is_instance());

      NodeID cnode = id.instance.creator_node;
      unsigned idx = id.instance.inst_idx;
      if(cnode == my_node_id) {
	// if it was locally created, we can directly access the local_instances list
	//  and it's a fatal error if it doesn't exist
	AutoHSLLock al(local_instances.mutex);
	assert(idx < local_instances.instances.size());
	assert(local_instances.instances[idx] != 0);
	return local_instances.instances[idx];
      } else {
	// figure out which instance list to look in - non-local creators require a 
	//  protected lookup
	InstanceList *ilist;
	{
	  AutoHSLLock al(instance_map_mutex);
	  // this creates a new InstanceList if needed
	  InstanceList *& iref = instances_by_creator[cnode];
	  if(!iref)
	    iref = new InstanceList;
	  ilist = iref;
	}

	// now look up (and possibly create) the instance in the right list
	{
	  AutoHSLLock al(ilist->mutex);

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
	AutoHSLLock al(local_instances.mutex);
	  
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
	RegionInstance i = ID::make_instance(mem_id.memory.owner_node,
					     my_node_id /*creator*/,
					     mem_id.memory.mem_idx,
					     inst_idx).convert<RegionInstance>();
	log_inst.info() << "creating new local instance: " << i;
	inst_impl = new RegionInstanceImpl(i, me);
	{
	  AutoHSLLock al(local_instances.mutex);
	  local_instances.instances[inst_idx] = inst_impl;
	}
      } else
	log_inst.info() << "reusing local instance: " << inst_impl->me;

      return inst_impl;
    }

    // releases a deleted instance so that it can be reused
    void MemoryImpl::release_instance(RegionInstance inst)
    {
      int inst_idx = ID(inst).instance.inst_idx;

      log_inst.info() << "releasing local instance: " << inst;
      {
	AutoHSLLock al(local_instances.mutex);
	local_instances.free_list.push_back(inst_idx);
      }
    }

    // attempt to allocate storage for the specified instance
    bool MemoryImpl::allocate_instance_storage(RegionInstance i,
					       size_t bytes, size_t alignment,
					       Event precondition, size_t offset /*=0*/)
    {
      // all allocation requests are handled by the memory's owning node for
      //  now - local caching might be possible though
      NodeID target = ID(me).memory.owner_node;
      if(target != my_node_id) {
	MemStorageAllocRequest::send_request(target,
					     me, i,
					     bytes, alignment,
					     precondition, offset);
	return false /*asynchronous notification*/;
      }

      if(!precondition.has_triggered()) {
	// TODO: queue things up?
	precondition.wait();
      }

      bool ok;
      {
	AutoHSLLock al(allocator_mutex);
	ok = allocator.allocate(i, bytes, alignment, offset);
      }

      if(ID(i).instance.creator_node == my_node_id) {
	// local notification of result
	get_instance(i)->notify_allocation(ok, offset);
      } else {
	// remote notification
	MemStorageAllocResponse::send_request(ID(i).instance.creator_node,
					      i,
					      offset,
					      ok);
      }

      return true /*immediate notification*/;
    }

    // release storage associated with an instance
    void MemoryImpl::release_instance_storage(RegionInstance i,
					      Event precondition)
    {
      // all allocation requests are handled by the memory's owning node for
      //  now - local caching might be possible though
      NodeID target = ID(me).memory.owner_node;
      if(target != my_node_id) {
	MemStorageReleaseRequest::send_request(target,
					       me, i,
					       precondition);
	return;
      }

      // TODO: memory needs to handle non-ready releases
      assert(precondition.has_triggered());

      RegionInstanceImpl *impl = get_instance(i);

      // better not be in the unallocated state...
      assert(impl->metadata.inst_offset != size_t(-1));
      // deallocate unless the allocation had failed
      if(impl->metadata.inst_offset != size_t(-2)) {
	AutoHSLLock al(allocator_mutex);
	allocator.deallocate(i);
      }

      if(ID(i).instance.creator_node == my_node_id) {
	// local notification of result
	get_instance(i)->notify_deallocation();
      } else {
	// remote notification
	MemStorageReleaseResponse::send_request(ID(i).instance.creator_node,
						i);
      }
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalCPUMemory
  //

  LocalCPUMemory::LocalCPUMemory(Memory _me, size_t _size,
				 void *prealloc_base /*= 0*/, bool _registered /*= false*/) 
    : MemoryImpl(_me, _size, MKIND_SYSMEM, ALIGNMENT, 
		 (_registered ? Memory::REGDMA_MEM : Memory::SYSTEM_MEM))
  {
    if(prealloc_base) {
      base = (char *)prealloc_base;
      prealloced = true;
      registered = _registered;
    } else {
      // allocate our own space
      // enforce alignment on the whole memory range
      base_orig = new char[_size + ALIGNMENT - 1];
      size_t ofs = reinterpret_cast<size_t>(base_orig) % ALIGNMENT;
      if(ofs > 0) {
	base = base_orig + (ALIGNMENT - ofs);
      } else {
	base = base_orig;
      }
      prealloced = false;
      assert(!_registered);
      registered = false;
    }
    log_malloc.debug("CPU memory at %p, size = %zd%s%s", base, _size, 
		     prealloced ? " (prealloced)" : "", registered ? " (registered)" : "");
    free_blocks[0] = _size;
  }

  LocalCPUMemory::~LocalCPUMemory(void)
  {
    if(!prealloced)
      delete[] base_orig;
  }

  off_t LocalCPUMemory::alloc_bytes(size_t size)
  {
    return alloc_bytes_local(size);
  }
  
  void LocalCPUMemory::free_bytes(off_t offset, size_t size)
  {
    free_bytes_local(offset, size);
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

  int LocalCPUMemory::get_home_node(off_t offset, size_t size)
  {
    return my_node_id;
  }

  void *LocalCPUMemory::local_reg_base(void)
  {
    return registered ? base : 0;
  };
  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMemory
  //

    RemoteMemory::RemoteMemory(Memory _me, size_t _size, Memory::Kind k, void *_regbase)
      : MemoryImpl(_me, _size, _regbase ? MKIND_RDMA : MKIND_REMOTE, 0, k), regbase(_regbase)
    {
    }

    RemoteMemory::~RemoteMemory(void)
    {
    }

    off_t RemoteMemory::alloc_bytes(size_t size)
    {
      return alloc_bytes_remote(size);
    }

    void RemoteMemory::free_bytes(off_t offset, size_t size)
    {
      free_bytes_remote(offset, size);
    }

    void RemoteMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // can't read/write a remote memory
#define ALLOW_REMOTE_MEMORY_WRITES
#ifdef ALLOW_REMOTE_MEMORY_WRITES
      // THIS IS BAD - no fence means no consistency!
      do_remote_write(me, offset, src, size, 0, true /* make copy! */);
#else
      assert(0);
#endif
    }

    void RemoteMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // this better be an RDMA-able memory
#ifdef USE_GASNET
      assert(kind == MemoryImpl::MKIND_RDMA);
      void *srcptr = ((char *)regbase) + offset;
      gasnet_get(dst, ID(me).memory.owner_node, srcptr, size);
#else
      assert(0 && "no remote get_bytes without GASNET");
#endif
    }

    void *RemoteMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0;
    }

    int RemoteMemory::get_home_node(off_t offset, size_t size)
    {
      return ID(me).memory.owner_node;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetMemory
  //

    GASNetMemory::GASNetMemory(Memory _me, size_t size_per_node)
      : MemoryImpl(_me, 0 /* we'll calculate it below */, MKIND_GLOBAL,
		     MEMORY_STRIDE, Memory::GLOBAL_MEM)
    {
      num_nodes = max_node_id + 1;
      segbases.resize(num_nodes);
#ifdef USE_GASNET
      gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[num_nodes];
      CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );
      
      for(int i = 0; i < num_nodes; i++) {
	assert(seginfos[i].size >= size_per_node);
	segbases[i] = (char *)(seginfos[i].addr);
      }
      delete[] seginfos;
#else
      for(int i = 0; i < num_nodes; i++) {
	segbases[i] = (char *)(malloc(size_per_node));
	assert(segbases[i] != 0);
      }
#endif

      size = size_per_node * num_nodes;
      memory_stride = MEMORY_STRIDE;
      
      free_blocks[0] = size;
      // tell new allocator about the available memory too
      allocator.add_range(0, size);
    }

    GASNetMemory::~GASNetMemory(void)
    {
    }

    off_t GASNetMemory::alloc_bytes(size_t size)
    {
      if(my_node_id == 0) {
	return alloc_bytes_local(size);
      } else {
	return alloc_bytes_remote(size);
      }
    }

    void GASNetMemory::free_bytes(off_t offset, size_t size)
    {
      if(my_node_id == 0) {
	free_bytes_local(offset, size);
      } else {
	free_bytes_remote(offset, size);
      }
    }

    void GASNetMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      char *dst_c = (char *)dst;
      while(size > 0) {
	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;
#ifdef USE_GASNET
	gasnet_get(dst_c, node, segbases[node]+(blkid * memory_stride)+blkoffset, chunk_size);
#else
	memcpy(dst_c, segbases[node]+(blkid * memory_stride)+blkoffset, chunk_size);
#endif
	offset += chunk_size;
	dst_c += chunk_size;
	size -= chunk_size;
      }
    }

    void GASNetMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      char *src_c = (char *)src; // dropping const on purpose...
      while(size > 0) {
	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;
#ifdef USE_GASNET
	gasnet_put(node, segbases[node]+(blkid * memory_stride)+blkoffset, src_c, chunk_size);
#else
	memcpy(segbases[node]+(blkid * memory_stride)+blkoffset, src_c, chunk_size);
#endif
	offset += chunk_size;
	src_c += chunk_size;
	size -= chunk_size;
      }
    }

    void GASNetMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					    size_t count, const void *entry_buffer)
    {
      assert(0);
#ifdef NEED_TO_FIX_REDUCTION_LISTS_FOR_DEPPART
      const char *entry = (const char *)entry_buffer;
      unsigned ptr;

      for(size_t i = 0; i < count; i++)
      {
	redop->get_list_pointers(&ptr, entry, 1);
	//printf("ptr[%d] = %d\n", i, ptr);
	off_t elem_offset = offset + ptr * redop->sizeof_lhs;
	off_t blkid = (elem_offset / memory_stride / num_nodes);
	off_t node = (elem_offset / memory_stride) % num_nodes;
	off_t blkoffset = elem_offset % memory_stride;
	assert(node == my_node_id);
	char *tgt_ptr = ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset;
	redop->apply_list_entry(tgt_ptr, entry, 1, ptr);
	entry += redop->sizeof_list_entry;
      }
#endif
    }

    void *GASNetMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0;  // can't give a pointer to the caller - have to use RDMA
    }

    int GASNetMemory::get_home_node(off_t offset, size_t size)
    {
      off_t start_blk = offset / memory_stride;
      off_t end_blk = (offset + size - 1) / memory_stride;
      if(start_blk != end_blk) return -1;

      return start_blk % num_nodes;
    }

    void GASNetMemory::get_batch(size_t batch_size,
				 const off_t *offsets, void * const *dsts, 
				 const size_t *sizes)
    {
#define NO_USE_NBI_ACCESSREGION
#ifdef USE_GASNET
#ifdef USE_NBI_ACCESSREGION
      gasnet_begin_nbi_accessregion();
#endif
#endif
      DetailedTimer::push_timer(10);
      for(size_t i = 0; i < batch_size; i++) {
	off_t offset = offsets[i];
	char *dst_c = (char *)(dsts[i]);
	size_t size = sizes[i];

	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;

	while(size > 0) {
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;

	  char *src_c = (segbases[node] +
			 (blkid * memory_stride) + blkoffset);
#ifdef USE_GASNET
	  if(node != my_node_id) {
	    gasnet_get_nbi(dst_c, node, src_c, chunk_size);
	  } else
#endif
	  {
	    memcpy(dst_c, src_c, chunk_size);
	  }

	  dst_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

#ifdef USE_GASNET
#ifdef USE_NBI_ACCESSREGION
      DetailedTimer::push_timer(11);
      gasnet_handle_t handle = gasnet_end_nbi_accessregion();
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(12);
      gasnet_wait_syncnb(handle);
      DetailedTimer::pop_timer();
#else
      DetailedTimer::push_timer(13);
      gasnet_wait_syncnbi_gets();
      DetailedTimer::pop_timer();
#endif
#endif
    }

    void GASNetMemory::put_batch(size_t batch_size,
				 const off_t *offsets,
				 const void * const *srcs, 
				 const size_t *sizes)
    {
#ifdef USE_GASNET
      gasnet_begin_nbi_accessregion();
#endif

      DetailedTimer::push_timer(14);
      for(size_t i = 0; i < batch_size; i++) {
	off_t offset = offsets[i];
	const char *src_c = (char *)(srcs[i]);
	size_t size = sizes[i];

	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;

	while(size > 0) {
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;

	  char *dst_c = (segbases[node] +
			 (blkid * memory_stride) + blkoffset);
#ifdef USE_GASNET
	  if(node != my_node_id) {
	    gasnet_put_nbi(node, dst_c, (void *)src_c, chunk_size);
	  } else
#endif
	  {
	    memcpy(dst_c, src_c, chunk_size);
	  }

	  src_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

#ifdef USE_GASNET
      DetailedTimer::push_timer(15);
      gasnet_handle_t handle = gasnet_end_nbi_accessregion();
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(16);
      gasnet_wait_syncnb(handle);
      DetailedTimer::pop_timer();
#endif
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageAllocRequest
  //

  /*static*/ void MemStorageAllocRequest::handle_request(RequestArgs args)
  {
    MemoryImpl *impl = get_runtime()->get_memory_impl(args.memory);

    impl->allocate_instance_storage(args.inst,
				    args.bytes, args.alignment,
				    args.precondition, args.offset);
  }

  /*static*/ void MemStorageAllocRequest::send_request(NodeID target,
						       Memory memory, RegionInstance inst,
						       size_t bytes, size_t alignment,
						       Event precondition, size_t offset)
  {
    RequestArgs args;

    args.memory = memory;
    args.inst = inst;
    args.bytes = bytes;
    args.alignment = alignment;
    args.precondition = precondition;
    args.offset = offset;

    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageAllocResponse
  //

  /*static*/ void MemStorageAllocResponse::handle_request(RequestArgs args)
  {
    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.inst);

    impl->notify_allocation(args.success, args.offset);
  }

  /*static*/ void MemStorageAllocResponse::send_request(NodeID target,
							RegionInstance inst,
							size_t offset,
							bool success)
  {
    RequestArgs args;

    args.inst = inst;
    args.offset = offset;
    args.success = success;

    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageReleaseRequest
  //

  /*static*/ void MemStorageReleaseRequest::handle_request(RequestArgs args)
  {
    MemoryImpl *impl = get_runtime()->get_memory_impl(args.memory);

    impl->release_instance_storage(args.inst,
				   args.precondition);
  }

  /*static*/ void MemStorageReleaseRequest::send_request(NodeID target,
							 Memory memory,
							 RegionInstance inst,
							 Event precondition)
  {
    RequestArgs args;

    args.memory = memory;
    args.inst = inst;
    args.precondition = precondition;

    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemStorageReleaseResponse
  //

  /*static*/ void MemStorageReleaseResponse::handle_request(RequestArgs args)
  {
    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.inst);

    impl->notify_deallocation();
  }

  /*static*/ void MemStorageReleaseResponse::send_request(NodeID target,
							  RegionInstance inst)
  {
    RequestArgs args;

    args.inst = inst;

    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMemAllocRequest
  //

  /*static*/ void RemoteMemAllocRequest::handle_request(RequestArgs args)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    //printf("[%d] handling remote alloc of size %zd\n", my_node_id, args.size);
    off_t offset = get_runtime()->get_memory_impl(args.memory)->alloc_bytes(args.size);
    //printf("[%d] remote alloc will return %d\n", my_node_id, result);

    ResponseArgs r_args;
    r_args.resp_ptr = args.resp_ptr;
    r_args.offset = offset;
    Response::request(args.sender, r_args);
  }
  
  /*static*/ void RemoteMemAllocRequest::handle_response(ResponseArgs args)
  {
    HandlerReplyFuture<off_t> *f = static_cast<HandlerReplyFuture<off_t> *>(args.resp_ptr);
    f->set(args.offset);
  }

  /*static*/ off_t RemoteMemAllocRequest::send_request(NodeID target,
						       Memory memory,
						       size_t size)
  {
    HandlerReplyFuture<off_t> result;

    RequestArgs args;
    args.sender = my_node_id;
    args.resp_ptr = &result;
    args.memory = memory;
    args.size = size;

    Request::request(target, args);

    // wait for result to come back
    result.wait();
    return result.get();
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
  static GASNetHSL partial_remote_writes_lock;

  /*static*/ void RemoteWriteMessage::handle_request(RequestArgs args,
						     const void *data,
						     size_t datalen)
  {
    MemoryImpl *impl = get_runtime()->get_memory_impl(args.mem);

    log_copy.debug() << "received remote write request: mem=" << args.mem
		     << ", offset=" << args.offset << ", size=" << datalen
		     << ", seq=" << args.sender << '/' << args.sequence_id;
#ifdef DEBUG_REMOTE_WRITES
    printf("received remote write request: mem=" IDFMT ", offset=%zd, size=%zd, seq=%d/%d",
	   args.mem.id, args.offset, datalen,
	   args.sender, args.sequence_id);
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
      if(cpumem->registered) {
	if(data == (cpumem->base + args.offset)) {
	  // copy is in right spot - yay!
	  was_written_directly = true;
	} else {
	  log_copy.error() << "received remote write to registered memory in wrong spot: "
			   << data << " != "
			   << ((void *)(cpumem->base)) << "+" << args.offset
			   << " = " << ((void *)(cpumem->base + args.offset));
	}
      }
    }

    // if it wasn't directly written, we have to copy it
    if(!was_written_directly)
      impl->put_bytes(args.offset, data, datalen);

    // track the sequence ID to know when the full RDMA is done
    if(args.sequence_id > 0) {
      PartialWriteKey key;
      key.sender = args.sender;
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
	       my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       my_node_id, key.sender, key.sequence_id,
	       entry.fence,
	       entry.remaining_count, entry.remaining_count - 1);
#endif
	entry.remaining_count--;
	if(entry.remaining_count == 0) {
	  // we're the last write, and we've already got the fence, so 
	  //  respond
          RemoteWriteFenceAckMessage::send_request(args.sender,
                                                   entry.fence);
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
  /*static*/ void RemoteSerdezMessage::handle_request(RequestArgs args,
                                                      const void *data,
                                                      size_t datalen)
  {
    log_copy.debug() << "received remote serdez request: mem=" << args.mem
		     << ", offset=" << args.offset << ", size=" << datalen
		     << ", seq=" << args.sender << '/' << args.sequence_id;

    const CustomSerdezUntyped *serdez_op = get_runtime()->custom_serdez_table[args.serdez_id];
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
      key.sender = args.sender;
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
	       my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       my_node_id, key.sender, key.sequence_id,
	       entry.fence,
	       entry.remaining_count, entry.remaining_count - 1);
#endif
	entry.remaining_count--;
	if(entry.remaining_count == 0) {
	  // we're the last write, and we've already got the fence, so
	  //  respond
          RemoteWriteFenceAckMessage::send_request(args.sender,
                                                   entry.fence);
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

  /*static*/ void RemoteReduceMessage::handle_request(RequestArgs args,
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
		   args.sender, args.sequence_id);

    const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];

    size_t count = datalen / redop->sizeof_rhs;

    void *lhs = get_runtime()->get_memory_impl(args.mem)->get_direct_ptr(args.offset, args.stride * count);
    assert(lhs);

    if(red_fold)
      redop->fold_strided(lhs, data,
			  args.stride, redop->sizeof_rhs, count, false /*not exclusive*/);
    else
      redop->apply_strided(lhs, data, 
			   args.stride, redop->sizeof_rhs, count, false /*not exclusive*/);

    // track the sequence ID to know when the full RDMA is done
    if(args.sequence_id > 0) {
      PartialWriteKey key;
      key.sender = args.sender;
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
	       my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       my_node_id, key.sender, key.sequence_id,
	       entry.fence,
	       entry.remaining_count, entry.remaining_count - 1);
#endif
	entry.remaining_count--;
	if(entry.remaining_count == 0) {
	  // we're the last write, and we've already got the fence, so 
	  //  respond
          RemoteWriteFenceAckMessage::send_request(args.sender,
                                                   entry.fence);
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
  // class RemoteReduceListMessage
  //

  /*static*/ void RemoteReduceListMessage::handle_request(RequestArgs args,
							  const void *data,
							  size_t datalen)
  {
    MemoryImpl *impl = get_runtime()->get_memory_impl(args.mem);
    
    log_copy.debug("received remote reduction list request: mem=" IDFMT ", offset=%zd, size=%zd, redopid=%d",
		   args.mem.id, (ssize_t)args.offset, datalen, args.redopid);

    switch(impl->kind) {
    case MemoryImpl::MKIND_SYSMEM:
    case MemoryImpl::MKIND_ZEROCOPY:
    case MemoryImpl::MKIND_GPUFB:
    default:
      assert(0);

    case MemoryImpl::MKIND_GLOBAL:
      {
	const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[args.redopid];
	assert((datalen % redop->sizeof_list_entry) == 0);
	impl->apply_reduction_list(args.offset,
				   redop,
				   datalen / redop->sizeof_list_entry,
				   data);
      }
    }
  }

  /*static*/ void RemoteReduceListMessage::send_request(NodeID target,
							Memory mem,
							off_t offset,
							ReductionOpID redopid,
							const void *data,
							size_t datalen,
							int payload_mode)
  {
    RequestArgs args;

    args.mem = mem;
    args.offset = offset;
    args.redopid = redopid;
    Message::request(target, args, data, datalen, payload_mode);
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

  /*static*/ void RemoteWriteFenceMessage::handle_request(RequestArgs args)
  {
    log_copy.debug("remote write fence (mem = " IDFMT ", seq = %d/%d, count = %d, fence = %p",
		   args.mem.id, args.sender, args.sequence_id, args.num_writes, args.fence);
    
    assert(args.sequence_id != 0);
    // track the sequence ID to know when the full RDMA is done
    if(args.sequence_id > 0) {
      PartialWriteKey key;
      key.sender = args.sender;
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
	       my_node_id, key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (previous writes)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p -> %p, %d -> %d\n",
	       my_node_id, key.sender, key.sequence_id,
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
          RemoteWriteFenceAckMessage::send_request(args.sender,
                                                   entry.fence);
	  partial_remote_writes.erase(it);
	  partial_remote_writes_lock.unlock();
	  return;
	}
      }
      partial_remote_writes_lock.unlock();
    }
  }

  /*static*/ void RemoteWriteFenceMessage::send_request(NodeID target,
							Memory memory,
							unsigned sequence_id,
							unsigned num_writes,
                                                        RemoteWriteFence *fence)
  {
    RequestArgs args;

    args.mem = memory;
    args.sender = my_node_id;
    args.sequence_id = sequence_id;
    args.num_writes = num_writes;
    args.fence = fence;
    Message::request(target, args);
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteWriteFenceAckMessage
  //

  /*static*/ void RemoteWriteFenceAckMessage::handle_request(RequestArgs args)
  {
    log_copy.debug("remote write fence ack: fence = %p",
		   args.fence);

    args.fence->mark_finished(true /*successful*/);
  }

  /*static*/ void RemoteWriteFenceAckMessage::send_request(NodeID target,
                                                           RemoteWriteFence *fence)
  {
    RequestArgs args;

    args.fence = fence;
    Message::request(target, args);
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // do_remote_*
  //

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
	size_t max_xfer_size = get_lmb_size(ID(mem).memory.owner_node);

	if(datalen > max_xfer_size) {
	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;
	  RemoteWriteMessage::RequestArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.sender = my_node_id;
	  args.sequence_id = sequence_id;

	  int count = 1;
	  while(datalen > max_xfer_size) {
	    RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
						 pos, max_xfer_size,
						 make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += max_xfer_size;
	    pos += max_xfer_size;
	    datalen -= max_xfer_size;
	    count++;
	  }

	  // last send includes whatever's left
	  RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
					       pos, datalen, 
					       make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	RemoteWriteMessage::RequestArgs args;
	args.mem = mem;
	args.offset = offset;
        args.sender = my_node_id;
	args.sequence_id = sequence_id;
	RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
					     data, datalen,
					     (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP),
					     dstptr);
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
	size_t max_xfer_size = get_lmb_size(ID(mem).memory.owner_node);
	size_t max_lines_per_xfer = max_xfer_size / datalen;
	assert(max_lines_per_xfer > 0);

	if(lines > max_lines_per_xfer) {
	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;
	  RemoteWriteMessage::RequestArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.sender = my_node_id;
	  args.sequence_id = sequence_id;

	  int count = 1;
	  while(lines > max_lines_per_xfer) {
	    RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
						 pos, datalen,
						 stride, max_lines_per_xfer,
						 make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += datalen * max_lines_per_xfer;
	    pos += stride * max_lines_per_xfer;
	    lines -= max_lines_per_xfer;
	    count++;
	  }

	  // last send includes whatever's left
	  RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
					       pos, datalen, stride, lines,
					       make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	RemoteWriteMessage::RequestArgs args;
	args.mem = mem;
	args.offset = offset;
        args.sender = my_node_id;
	args.sequence_id = sequence_id;

	RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
					     data, datalen, stride, lines,
					     make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP,
					     dstptr);

	return 1;
      }
    }

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
	size_t max_xfer_size = get_lmb_size(ID(mem).memory.owner_node);

	if(datalen > max_xfer_size) {
	  log_copy.info("breaking large send into pieces");
	  RemoteWriteMessage::RequestArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.sender = my_node_id;
	  args.sequence_id = sequence_id;

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
		RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
						     pos, max_xfer_size,
						     make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
		args.offset += max_xfer_size;
		pos += max_xfer_size;
		left -= max_xfer_size;
		count++;
	      }
	      RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
						   pos, left,
						   make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	      args.offset += left;
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

	    RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
						 subspans, xfer_size,
						 make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += xfer_size;
	    datalen -= xfer_size;
	    count++;
	  }

	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	RemoteWriteMessage::RequestArgs args;
	args.mem = mem;
	args.offset = offset;
        args.sender = my_node_id;
	args.sequence_id = sequence_id;

	RemoteWriteMessage::Message::request(ID(mem).memory.owner_node, args,
					     spans, datalen,
					     make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP,
					     dstptr);
	
	return 1;
      }
    }

    unsigned do_remote_serdez(Memory mem, off_t offset,
                             CustomSerdezID serdez_id,
                             const void *data, size_t count,
                             unsigned sequence_id)
    {
      const CustomSerdezUntyped *serdez_op = get_runtime()->custom_serdez_table[serdez_id];
      size_t field_size = serdez_op->sizeof_field_type;
      log_copy.debug("sending remote serdez request: mem=" IDFMT ", offset=%zd, size=%zdx%zd, serdez_id=%d",
                     mem.id, (ssize_t)offset, field_size, count, serdez_id);
      size_t max_xfer_size = get_lmb_size(ID(mem).memory.owner_node);
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
        RemoteSerdezMessage::RequestArgs args;
        args.mem = mem;
        args.offset = offset;
        offset = new_offset;
        args.count = cur_count;
        args.serdez_id = serdez_id;
        args.sender = my_node_id;
        args.sequence_id = sequence_id;
        RemoteSerdezMessage::Message::request(ID(mem).memory.owner_node, args,
                                              buffer_start, cur_size, PAYLOAD_COPY);
        xfers ++;
      }
      free(buffer_start);
      return xfers;
    }

    unsigned do_remote_reduce(Memory mem, off_t offset,
			      ReductionOpID redop_id, bool red_fold,
			      const void *data, size_t count,
			      off_t src_stride, off_t dst_stride,
			      unsigned sequence_id,
			      bool make_copy /*= false*/)
    {
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];
      size_t rhs_size = redop->sizeof_rhs;

      log_copy.debug("sending remote reduction request: mem=" IDFMT ", offset=%zd+%zd, size=%zdx%zd, redop=%d(%s)",
		     mem.id, (ssize_t)offset, (ssize_t)dst_stride, rhs_size, count,
		     redop_id, (red_fold ? "fold" : "apply"));

      // reductions always have to bounce off an intermediate buffer, so are subject to
      //  LMB limits
      {
	size_t max_xfer_size = get_lmb_size(ID(mem).memory.owner_node);
	size_t max_elmts_per_xfer = max_xfer_size / rhs_size;
	assert(max_elmts_per_xfer > 0);

	if(count > max_elmts_per_xfer) {
	  log_copy.info("breaking large reduction into pieces");
	  const char *pos = (const char *)data;
	  RemoteReduceMessage::RequestArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.stride = dst_stride;
	  assert(((off_t)(args.stride)) == dst_stride); // did it fit?
	  // fold encoded as a negation of the redop_id
	  args.redop_id = red_fold ? -redop_id : redop_id;
	  //args.red_fold = red_fold;
	  args.sender = my_node_id;
	  args.sequence_id = sequence_id;

	  int xfers = 1;
	  while(count > max_elmts_per_xfer) {
	    RemoteReduceMessage::Message::request(ID(mem).memory.owner_node, args,
						  pos, rhs_size,
						  src_stride, max_elmts_per_xfer,
						  make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += dst_stride * max_elmts_per_xfer;
	    pos += src_stride * max_elmts_per_xfer;
	    count -= max_elmts_per_xfer;
	    xfers++;
	  }

	  // last send includes whatever's left
	  RemoteReduceMessage::Message::request(ID(mem).memory.owner_node, args,
						pos, rhs_size, src_stride, count,
						make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	  return xfers;
	}
      }

      // we get here with a write smaller than the LMB
      {
	RemoteReduceMessage::RequestArgs args;
	args.mem = mem;
	args.offset = offset;
	args.stride = dst_stride;
	assert(((off_t)(args.stride)) == dst_stride); // did it fit?
	// fold encoded as a negation of the redop_id
	args.redop_id = red_fold ? -redop_id : redop_id;
	//args.red_fold = red_fold;
	args.sender = my_node_id;
	args.sequence_id = sequence_id;

	RemoteReduceMessage::Message::request(ID(mem).memory.owner_node, args,
					      data, rhs_size, src_stride, count,
					      make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);

	return 1;
      }
    }

    void do_remote_apply_red_list(int node, Memory mem, off_t offset,
				  ReductionOpID redopid,
				  const void *data, size_t datalen)
    {
      RemoteReduceListMessage::send_request(node, mem, offset, redopid,
					    data, datalen, PAYLOAD_COPY);
    }

    void do_remote_fence(Memory mem, unsigned sequence_id, unsigned num_writes,
                         RemoteWriteFence *fence)
    {
      // technically we could handle a num_writes == 0 case, but since it's
      //  probably indicative of badness elsewhere, barf on it for now
      assert(num_writes > 0);

      RemoteWriteFenceMessage::send_request(ID(mem).memory.owner_node, mem, sequence_id,
					    num_writes, fence);
    }
  
}; // namespace Realm
