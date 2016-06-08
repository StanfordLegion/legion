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

#include "mem_impl.h"

#include "proc_impl.h"
#include "logging.h"
#include "serialize.h"
#include "inst_impl.h"
#include "runtime_impl.h"
#include "profiling.h"
#include "utils.h"

namespace Realm {

  Logger log_malloc("malloc");
  extern Logger log_copy; // in idx_impl.cc
  extern Logger log_inst; // in inst_impl.cc



  ////////////////////////////////////////////////////////////////////////
  //
  // class Memory
  //

    AddressSpace Memory::address_space(void) const
    {
      // limitations in the current newmap implementation require that all memories
      //  be managed by the HLR instance on the first node
      return 0; //ID(id).node();
    }

    ID::IDType Memory::local_id(void) const
    {
      return ID(id).index();
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

    // make bad offsets really obvious (+1 PB)
    static const off_t ZERO_SIZE_INSTANCE_OFFSET = 1ULL << 50;

    MemoryImpl::MemoryImpl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment, Memory::Kind _lowlevel_kind)
      : me(_me), size(_size), kind(_kind), alignment(_alignment), lowlevel_kind(_lowlevel_kind)
      , usage(stringbuilder() << "realm/mem " << _me << "/usage")
      , peak_usage(stringbuilder() << "realm/mem " << _me << "/peak_usage")
      , peak_footprint(stringbuilder() << "realm/mem " << _me << "/peak_footprint")
    {
    }

    MemoryImpl::~MemoryImpl(void)
    {
#ifdef REALM_PROFILE_MEMORY_USAGE
      printf("Memory " IDFMT " usage: peak=%zd (%.1f MB) footprint=%zd (%.1f MB)\n",
	     me.id, 
	     (size_t)peak_usage, peak_usage / 1048576.0,
	     (size_t)peak_footprint, peak_footprint / 1048576.0);
#endif
    }

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
			  size, size + (alignment - leftover));
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
	    log_malloc.info("alloc full block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, retval);
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
	    log_malloc.info("alloc partial block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, retval);
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
			  size, size + (alignment - leftover));
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
      return RemoteMemAllocRequest::send_request(ID(me).node(), me, size);
    }

    void MemoryImpl::free_bytes_remote(off_t offset, size_t size)
    {
      assert(0);
    }

    Memory::Kind MemoryImpl::get_kind(void) const
    {
      return lowlevel_kind;
    }

    RegionInstance MemoryImpl::create_instance_local(IndexSpace r,
						       const int *linearization_bits,
						       size_t bytes_needed,
						       size_t block_size,
						       size_t element_size,
						       const std::vector<size_t>& field_sizes,
						       ReductionOpID redopid,
						       off_t list_size,
                                                       const ProfilingRequestSet &reqs,
						       RegionInstance parent_inst)
    {
      off_t inst_offset = alloc_bytes(bytes_needed);
      if(inst_offset < 0) {
        return RegionInstance::NO_INST;
      }

      off_t count_offset = -1;
      if(list_size > 0) {
	count_offset = alloc_bytes(sizeof(size_t));
	if(count_offset < 0) {
	  return RegionInstance::NO_INST;
	}

	size_t zero = 0;
	put_bytes(count_offset, &zero, sizeof(zero));
      }

      // SJT: think about this more to see if there are any race conditions
      //  with an allocator temporarily having the wrong ID
      RegionInstance i = ID(ID::ID_INSTANCE, 
			    ID(me).node(),
			    ID(me).index_h(),
			    0).convert<RegionInstance>();


      //RegionMetaDataImpl *r_impl = get_runtime()->get_metadata_impl(r);
      DomainLinearization linear;
      linear.deserialize(linearization_bits);

      RegionInstanceImpl *i_impl = new RegionInstanceImpl(i, r, me, inst_offset, 
                                                              bytes_needed, redopid,
							      linear, block_size, 
                                                              element_size, field_sizes, reqs,
							      count_offset, list_size, 
                                                              parent_inst);

      // find/make an available index to store this in
      ID::IDType index;
      {
	AutoHSLLock al(mutex);

	size_t size = instances.size();
	for(index = 0; index < size; index++)
	  if(!instances[index]) {
	    i.id |= index;
	    i_impl->me = i;
	    instances[index] = i_impl;
	    break;
	  }
        assert(index < (((ID::IDType)1) << ID::INDEX_L_BITS));

	i.id |= index;
	i_impl->me = i;
	i_impl->lock.me = ID(i.id).convert<Reservation>(); // have to change the lock's ID too!
	if(index >= size) instances.push_back(i_impl);
      }

      log_inst.info("local instance " IDFMT " created in memory " IDFMT " at offset %zd+%zd (redop=%d list_size=%zd parent_inst=" IDFMT " block_size=%zd)",
		    i.id, me.id, inst_offset, bytes_needed, redopid, list_size,
                    parent_inst.id, block_size);

      return i;
    }

    RegionInstance MemoryImpl::create_instance_remote(IndexSpace r,
							const int *linearization_bits,
							size_t bytes_needed,
							size_t block_size,
							size_t element_size,
							const std::vector<size_t>& field_sizes,
							ReductionOpID redopid,
							off_t list_size,
                                                        const ProfilingRequestSet &reqs,
							RegionInstance parent_inst)
    {
      CreateInstanceRequest::Result resp;

      CreateInstanceRequest::send_request(&resp,
					  ID(me).node(), me, r,
					  parent_inst, bytes_needed,
					  block_size, element_size,
					  list_size, redopid,
					  linearization_bits,
					  field_sizes,
					  &reqs);

      // Only do this if the response succeeds
      if (resp.i.exists()) {
        log_inst.debug("created remote instance: inst=" IDFMT " offset=%zd", resp.i.id, resp.inst_offset);

        DomainLinearization linear;
        linear.deserialize(linearization_bits);

        RegionInstanceImpl *i_impl = new RegionInstanceImpl(resp.i, r, me, resp.inst_offset, bytes_needed, redopid,
                                                                linear, block_size, element_size, field_sizes, reqs,
                                                                resp.count_offset, list_size, parent_inst);

        unsigned index = ID(resp.i).index_l();
        // resize array if needed
        if(index >= instances.size()) {
          AutoHSLLock a(mutex);
          if(index >= instances.size()) {
            log_inst.debug("resizing instance array: mem=" IDFMT " old=%zd new=%d",
                     me.id, instances.size(), index+1);
            for(unsigned i = instances.size(); i <= index; i++)
              instances.push_back(0);
          }
        }
        instances[index] = i_impl;
      }
      return resp.i;
    }

    RegionInstanceImpl *MemoryImpl::get_instance(RegionInstance i)
    {
      ID id(i);

      // have we heard of this one before?  if not, add it
      unsigned index = id.index_l();
      if(index >= instances.size()) { // lock not held - just for early out
	AutoHSLLock a(mutex);
	if(index >= instances.size()) // real check
	  instances.resize(index + 1);
      }

      if(!instances[index]) {
	//instances[index] = new RegionInstanceImpl(id.node());
	assert(0);
      }

      return instances[index];
    }

    void MemoryImpl::destroy_instance_local(RegionInstance i, 
					      bool local_destroy)
    {
      log_inst.info("destroying local instance: mem=" IDFMT " inst=" IDFMT "", me.id, i.id);

      // all we do for now is free the actual data storage
      unsigned index = ID(i).index_l();
      assert(index < instances.size());

      RegionInstanceImpl *iimpl = instances[index];

      free_bytes(iimpl->metadata.alloc_offset, iimpl->metadata.size);

      if(iimpl->metadata.count_offset >= 0)
	free_bytes(iimpl->metadata.count_offset, sizeof(size_t));

      // begin recovery of metadata
      if(iimpl->metadata.initiate_cleanup(i.id)) {
	// no remote copies exist, so we can reclaim instance immediately
	//log_metadata.info("no remote copies of metadata for " IDFMT, i.id);
	// TODO
      }
      
      // handle any profiling requests
      iimpl->finalize_instance();
      
      return; // TODO: free up actual instance record?
      ID id(i);

      // TODO: actually free corresponding storage
    }

    void MemoryImpl::destroy_instance_remote(RegionInstance i, 
					       bool local_destroy)
    {
      // if we're the original destroyer of the instance, tell the owner
      if(local_destroy) {
	int owner = ID(me).node();

	log_inst.debug("destroying remote instance: node=%d inst=" IDFMT "", owner, i.id);

	DestroyInstanceMessage::send_request(owner, me, i);
      }

      // and right now, we leave the instance itself untouched
      return;
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

  RegionInstance LocalCPUMemory::create_instance(IndexSpace r,
						 const int *linearization_bits,
						 size_t bytes_needed,
						 size_t block_size,
						 size_t element_size,
						 const std::vector<size_t>& field_sizes,
						 ReductionOpID redopid,
						 off_t list_size,
						 const ProfilingRequestSet &reqs,
						 RegionInstance parent_inst)
  {
    return create_instance_local(r, linearization_bits, bytes_needed,
				 block_size, element_size, field_sizes, redopid,
				 list_size, reqs, parent_inst);
  }

  void LocalCPUMemory::destroy_instance(RegionInstance i, 
					bool local_destroy)
  {
    destroy_instance_local(i, local_destroy);
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
    return (base + offset);
  }

  int LocalCPUMemory::get_home_node(off_t offset, size_t size)
  {
    return gasnet_mynode();
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
    
    RegionInstance RemoteMemory::create_instance(IndexSpace r,
						 const int *linearization_bits,
						 size_t bytes_needed,
						 size_t block_size,
						 size_t element_size,
						 const std::vector<size_t>& field_sizes,
						 ReductionOpID redopid,
						 off_t list_size,
						 const ProfilingRequestSet &reqs,
						 RegionInstance parent_inst)
    {
      return create_instance_remote(r, linearization_bits, bytes_needed,
				    block_size, element_size, field_sizes, redopid,
				    list_size, reqs, parent_inst);
    }

    void RemoteMemory::destroy_instance(RegionInstance i, 
					bool local_destroy)
    {
      destroy_instance_remote(i, local_destroy);
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
      gasnet_get(dst, ID(me).node(), srcptr, size);
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
      return ID(me).node();
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetMemory
  //

    GASNetMemory::GASNetMemory(Memory _me, size_t size_per_node)
      : MemoryImpl(_me, 0 /* we'll calculate it below */, MKIND_GLOBAL,
		     MEMORY_STRIDE, Memory::GLOBAL_MEM)
    {
      num_nodes = gasnet_nodes();
      seginfos = new gasnet_seginfo_t[num_nodes];
      CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );
      
      for(int i = 0; i < num_nodes; i++) {
	assert(seginfos[i].size >= size_per_node);
      }

      size = size_per_node * num_nodes;
      memory_stride = MEMORY_STRIDE;
      
      free_blocks[0] = size;
    }

    GASNetMemory::~GASNetMemory(void)
    {
    }

    RegionInstance GASNetMemory::create_instance(IndexSpace r,
						 const int *linearization_bits,
						 size_t bytes_needed,
						 size_t block_size,
						 size_t element_size,
						 const std::vector<size_t>& field_sizes,
						 ReductionOpID redopid,
						 off_t list_size,
                                                 const ProfilingRequestSet &reqs,
						 RegionInstance parent_inst)
    {
      if(gasnet_mynode() == 0) {
	return create_instance_local(r, linearization_bits, bytes_needed,
				     block_size, element_size, field_sizes, redopid,
				     list_size, reqs, parent_inst);
      } else {
	return create_instance_remote(r, linearization_bits, bytes_needed,
				      block_size, element_size, field_sizes, redopid,
				      list_size, reqs, parent_inst);
      }
    }

    void GASNetMemory::destroy_instance(RegionInstance i, 
					bool local_destroy)
    {
      if(gasnet_mynode() == 0) {
	destroy_instance_local(i, local_destroy);
      } else {
	destroy_instance_remote(i, local_destroy);
      }
    }

    off_t GASNetMemory::alloc_bytes(size_t size)
    {
      if(gasnet_mynode() == 0) {
	return alloc_bytes_local(size);
      } else {
	return alloc_bytes_remote(size);
      }
    }

    void GASNetMemory::free_bytes(off_t offset, size_t size)
    {
      if(gasnet_mynode() == 0) {
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
	gasnet_get(dst_c, node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, chunk_size);
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
	gasnet_put(node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, src_c, chunk_size);
	offset += chunk_size;
	src_c += chunk_size;
	size -= chunk_size;
      }
    }

    void GASNetMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					    size_t count, const void *entry_buffer)
    {
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
	assert(node == gasnet_mynode());
	char *tgt_ptr = ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset;
	redop->apply_list_entry(tgt_ptr, entry, 1, ptr);
	entry += redop->sizeof_list_entry;
      }
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
#ifdef USE_NBI_ACCESSREGION
      gasnet_begin_nbi_accessregion();
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

	  char *src_c = (((char *)seginfos[node].addr) +
			 (blkid * memory_stride) + blkoffset);
	  if(node == gasnet_mynode()) {
	    memcpy(dst_c, src_c, chunk_size);
	  } else {
	    gasnet_get_nbi(dst_c, node, src_c, chunk_size);
	  }

	  dst_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

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
    }

    void GASNetMemory::put_batch(size_t batch_size,
				 const off_t *offsets,
				 const void * const *srcs, 
				 const size_t *sizes)
    {
      gasnet_begin_nbi_accessregion();

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

	  char *dst_c = (((char *)seginfos[node].addr) +
			 (blkid * memory_stride) + blkoffset);
	  if(node == gasnet_mynode()) {
	    memcpy(dst_c, src_c, chunk_size);
	  } else {
	    gasnet_put_nbi(node, dst_c, (void *)src_c, chunk_size);
	  }

	  src_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(15);
      gasnet_handle_t handle = gasnet_end_nbi_accessregion();
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(16);
      gasnet_wait_syncnb(handle);
      DetailedTimer::pop_timer();
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMemAllocRequest
  //

  /*static*/ void RemoteMemAllocRequest::handle_request(RequestArgs args)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    //printf("[%d] handling remote alloc of size %zd\n", gasnet_mynode(), args.size);
    off_t offset = get_runtime()->get_memory_impl(args.memory)->alloc_bytes(args.size);
    //printf("[%d] remote alloc will return %d\n", gasnet_mynode(), result);

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

  /*static*/ off_t RemoteMemAllocRequest::send_request(gasnet_node_t target,
						       Memory memory,
						       size_t size)
  {
    HandlerReplyFuture<off_t> result;

    RequestArgs args;
    args.sender = gasnet_mynode();
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
  // class CreateInstanceRequest
  //

  /*static*/ void CreateInstanceRequest::handle_request(RequestArgs args,
							const void *msgdata, size_t msglen)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

    const Payload *payload = (const Payload *)msgdata;

    std::vector<size_t> field_sizes(payload->num_fields);
    for(size_t i = 0; i < payload->num_fields; i++)
      field_sizes[i] = payload->field_size(i);

    ProfilingRequestSet prs;
    // TODO: unbreak once the serialization stuff is repaired
    //size_t req_offset = sizeof(CreateInstancePayload) + sizeof(size_t) * payload->num_fields;
    //requests.deserialize(((const char*)msgdata)+req_offset);

    MemoryImpl *m_impl = get_runtime()->get_memory_impl(args.m);
    RegionInstance inst = m_impl->create_instance(args.r, 
						  payload->linearization_bits,
						  payload->bytes_needed,
						  payload->block_size,
						  payload->element_size,
						  field_sizes,
						  payload->redopid,
						  payload->list_size,
						  prs,
						  args.parent_inst);


    // send the response
    ResponseArgs r_args;

    r_args.resp_ptr = args.resp_ptr;
    r_args.i = inst;

    if (inst.exists()) {
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(inst);

      r_args.inst_offset = i_impl->metadata.alloc_offset;
      r_args.count_offset = i_impl->metadata.count_offset;
    } else {
      r_args.inst_offset = -1;
      r_args.count_offset = -1;
    }

    Response::request(args.sender, r_args);
  }

  /*static*/ void CreateInstanceRequest::handle_response(ResponseArgs args)
  {
    HandlerReplyFuture<Result> *f = static_cast<HandlerReplyFuture<Result> *>(args.resp_ptr);

    Result r;
    r.i = args.i;
    r.inst_offset = args.inst_offset;
    r.count_offset = args.count_offset;

    f->set(r);
  }

  /*static*/ void CreateInstanceRequest::send_request(Result *result,
						      gasnet_node_t target, Memory memory, IndexSpace ispace,
						      RegionInstance parent_inst, size_t bytes_needed,
						      size_t block_size, size_t element_size,
						      off_t list_size, ReductionOpID redopid,
						      const int *linearization_bits,
						      const std::vector<size_t>& field_sizes,
						      const ProfilingRequestSet *prs)
  {
    size_t req_offset = sizeof(Payload) + sizeof(size_t) * field_sizes.size();
    // TODO: unbreak once the serialization stuff is repaired
    //if(prs)
    //  assert(prs->empty());
    size_t payload_size = req_offset + 0;//reqs.compute_size();
    Payload *payload = (Payload *)malloc(payload_size);

    payload->bytes_needed = bytes_needed;
    payload->block_size = block_size;
    payload->element_size = element_size;
    //payload->adjust = ?
    payload->list_size = list_size;
    payload->redopid = redopid;

    for(unsigned i = 0; i < RegionInstanceImpl::MAX_LINEARIZATION_LEN; i++)
      payload->linearization_bits[i] = linearization_bits[i];

    payload->num_fields = field_sizes.size();
    for(unsigned i = 0; i < field_sizes.size(); i++)
      payload->field_size(i) = field_sizes[i];

    //reqs.serialize(((char*)payload)+req_offset);

    RequestArgs args;
    args.m = memory;
    args.r = ispace;
    args.parent_inst = parent_inst;
    log_inst.debug("creating remote instance: node=%d", ID(memory).node());

    HandlerReplyFuture<Result> result_future;
    args.resp_ptr = &result_future;
    args.sender = gasnet_mynode();

    Request::request(target, args, payload, payload_size, PAYLOAD_FREE);

    result_future.wait();
    *result = result_future.get();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DestroyInstanceRequest
  //

  /*static*/ void DestroyInstanceMessage::handle_request(RequestArgs args)
  {
    MemoryImpl *m_impl = get_runtime()->get_memory_impl(args.m);
    m_impl->destroy_instance(args.i, false);
  }

  /*static*/ void DestroyInstanceMessage::send_request(gasnet_node_t target,
						       Memory memory,
						       RegionInstance inst)
  {
    RequestArgs args;

    args.m = memory;
    args.i = inst;
    Message::request(target, args);
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

    switch(impl->kind) {
    case MemoryImpl::MKIND_SYSMEM:
      {
	LocalCPUMemory *cpumem = (LocalCPUMemory *)impl;
	if(cpumem->registered) {
	  if(data == (cpumem->base + args.offset)) {
	    // copy is in right spot - yay!
	  } else {
	    printf("%d: received remote write to registered memory in wrong spot: %p != %p+%zd = %p\n",
		   gasnet_mynode(), data, cpumem->base, args.offset, cpumem->base + args.offset);
	    impl->put_bytes(args.offset, data, datalen);
	  }
	} else {
	  impl->put_bytes(args.offset, data, datalen);
	}
	    
	break;
      }

    case MemoryImpl::MKIND_ZEROCOPY:
    case MemoryImpl::MKIND_GPUFB:
      {
	impl->put_bytes(args.offset, data, datalen);
	
	break;
      }

    default:
      assert(0);
    }

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
	       gasnet_mynode(), key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       gasnet_mynode(), key.sender, key.sequence_id,
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
	       gasnet_mynode(), key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       gasnet_mynode(), key.sender, key.sequence_id,
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

    log_copy.debug("received remote reduce request: mem=" IDFMT ", offset=%zd+%d, size=%zd, redop=%d(%s), seq=%d/%d",
		   args.mem.id, args.offset, args.stride, datalen,
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
	       gasnet_mynode(), key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (either another write or the fence)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p, %d -> %d\n",
	       gasnet_mynode(), key.sender, key.sequence_id,
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
		   args.mem.id, args.offset, datalen, args.redopid);

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

  /*static*/ void RemoteReduceListMessage::send_request(gasnet_node_t target,
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
	       gasnet_mynode(), key.sender, key.sequence_id,
	       entry.fence, entry.remaining_count);
#endif
      } else {
	// have an existing entry (previous writes)
	PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	printf("PWT: %d: have entry for %d/%d: %p -> %p, %d -> %d\n",
	       gasnet_mynode(), key.sender, key.sequence_id,
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

  /*static*/ void RemoteWriteFenceMessage::send_request(gasnet_node_t target,
							Memory memory,
							unsigned sequence_id,
							unsigned num_writes,
                                                        RemoteWriteFence *fence)
  {
    RequestArgs args;

    args.mem = memory;
    args.sender = gasnet_mynode();
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

  /*static*/ void RemoteWriteFenceAckMessage::send_request(gasnet_node_t target,
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
		     mem.id, offset, datalen);

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
	size_t max_xfer_size = get_lmb_size(ID(mem).node());

	if(datalen > max_xfer_size) {
	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;
	  RemoteWriteMessage::RequestArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.sender = gasnet_mynode();
	  args.sequence_id = sequence_id;

	  int count = 1;
	  while(datalen > max_xfer_size) {
	    RemoteWriteMessage::Message::request(ID(mem).node(), args,
						 pos, max_xfer_size,
						 make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += max_xfer_size;
	    pos += max_xfer_size;
	    datalen -= max_xfer_size;
	    count++;
	  }

	  // last send includes whatever's left
	  RemoteWriteMessage::Message::request(ID(mem).node(), args,
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
        args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;
	RemoteWriteMessage::Message::request(ID(mem).node(), args,
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
		     mem.id, offset, datalen, lines);

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
	size_t max_xfer_size = get_lmb_size(ID(mem).node());
	size_t max_lines_per_xfer = max_xfer_size / datalen;
	assert(max_lines_per_xfer > 0);

	if(lines > max_lines_per_xfer) {
	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;
	  RemoteWriteMessage::RequestArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.sender = gasnet_mynode();
	  args.sequence_id = sequence_id;

	  int count = 1;
	  while(lines > max_lines_per_xfer) {
	    RemoteWriteMessage::Message::request(ID(mem).node(), args,
						 pos, datalen,
						 stride, max_lines_per_xfer,
						 make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += datalen * max_lines_per_xfer;
	    pos += stride * max_lines_per_xfer;
	    lines -= max_lines_per_xfer;
	    count++;
	  }

	  // last send includes whatever's left
	  RemoteWriteMessage::Message::request(ID(mem).node(), args,
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
        args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;

	RemoteWriteMessage::Message::request(ID(mem).node(), args,
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
		     mem.id, offset, datalen, spans.size());

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
	size_t max_xfer_size = get_lmb_size(ID(mem).node());

	if(datalen > max_xfer_size) {
	  log_copy.info("breaking large send into pieces");
	  RemoteWriteMessage::RequestArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.sender = gasnet_mynode();
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
		RemoteWriteMessage::Message::request(ID(mem).node(), args,
						     pos, max_xfer_size,
						     make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
		args.offset += max_xfer_size;
		pos += max_xfer_size;
		left -= max_xfer_size;
		count++;
	      }
	      RemoteWriteMessage::Message::request(ID(mem).node(), args,
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

	    RemoteWriteMessage::Message::request(ID(mem).node(), args,
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
        args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;

	RemoteWriteMessage::Message::request(ID(mem).node(), args,
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
                     mem.id, offset, field_size, count, serdez_id);
      size_t max_xfer_size = get_lmb_size(ID(mem).node());
      // create a intermediate buf with same size as max_xfer_size
      char* buffer_start = (char*) malloc(max_xfer_size);
      const char *pos = (const char *)data;
      unsigned xfers = 0;
      while (count > 0) {
        size_t cur_size = 0;
        size_t cur_count = 0;
        char* buffer = buffer_start;
        off_t new_offset = offset;
        while (count > 0) {
          size_t elemnt_size = serdez_op->serialized_size(pos);
          // break if including this element exceeds max_xfer_size
          if (elemnt_size + cur_size > max_xfer_size)
            break;
          count--;
          cur_count++;
          serdez_op->serialize(pos, buffer); 
          pos += field_size;
          new_offset += field_size;
          buffer += elemnt_size;
          cur_size += elemnt_size;
        }
        assert(cur_size > 0);
        RemoteSerdezMessage::RequestArgs args;
        args.mem = mem;
        args.offset = offset;
        offset = new_offset;
        args.count = cur_count;
        args.serdez_id = serdez_id;
        args.sender = gasnet_mynode();
        args.sequence_id = sequence_id;
        RemoteSerdezMessage::Message::request(ID(mem).node(), args,
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
		     mem.id, offset, dst_stride, rhs_size, count,
		     redop_id, (red_fold ? "fold" : "apply"));

      // reductions always have to bounce off an intermediate buffer, so are subject to
      //  LMB limits
      {
	size_t max_xfer_size = get_lmb_size(ID(mem).node());
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
	  args.sender = gasnet_mynode();
	  args.sequence_id = sequence_id;

	  int xfers = 1;
	  while(count > max_elmts_per_xfer) {
	    RemoteReduceMessage::Message::request(ID(mem).node(), args,
						  pos, rhs_size,
						  src_stride, max_elmts_per_xfer,
						  make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += dst_stride * max_elmts_per_xfer;
	    pos += src_stride * max_elmts_per_xfer;
	    count -= max_elmts_per_xfer;
	    xfers++;
	  }

	  // last send includes whatever's left
	  RemoteReduceMessage::Message::request(ID(mem).node(), args,
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
	args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;

	RemoteReduceMessage::Message::request(ID(mem).node(), args,
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

      RemoteWriteFenceMessage::send_request(ID(mem).node(), mem, sequence_id,
					    num_writes, fence);
    }
  
}; // namespace Realm
