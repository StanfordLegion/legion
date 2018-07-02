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

#include "realm/inst_impl.h"

#include "realm/event_impl.h"
#include "realm/mem_impl.h"
#include "realm/logging.h"
#include "realm/runtime_impl.h"
#include "realm/deppart/inst_helper.h"
#ifdef USE_HDF
#include "realm/hdf5/hdf5_access.h"
#endif

TYPE_IS_SERIALIZABLE(Realm::InstanceLayoutGeneric::FieldLayout);

namespace Realm {

  Logger log_inst("inst");

  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredInstDestroy
  //

    class DeferredInstDestroy : public EventWaiter {
    public:
      DeferredInstDestroy(RegionInstance _inst) : inst(_inst) { }
      virtual ~DeferredInstDestroy(void) { }
    public:
      virtual bool event_triggered(Event e, bool poisoned)
      {
	// if input event is poisoned, do not attempt to destroy the lock
	// we don't have an output event here, so this may result in a leak if nobody is
	//  paying attention
	if(poisoned) {
	  log_poison.info() << "poisoned deferred instance destruction skipped - POSSIBLE LEAK - inst=" << inst;
	} else {
	  inst.destroy();
	  //get_runtime()->get_memory_impl(impl->memory)->destroy_instance(impl->me, true); 
	}
        return true;
      }

      virtual void print(std::ostream& os) const
      {
        os << "deferred instance destruction";
      }

      virtual Event get_finish_event(void) const
      {
	return Event::NO_EVENT;
      }

    protected:
      RegionInstance inst;
    };

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance
  //

    AddressSpace RegionInstance::address_space(void) const
    {
      return ID(id).instance.owner_node;
    }

    Memory RegionInstance::get_location(void) const
    {
      return ID::make_memory(ID(id).instance.owner_node,
			     ID(id).instance.mem_idx).convert<Memory>();
    }

    /*static*/ Event RegionInstance::create_instance(RegionInstance& inst,
						     Memory memory,
						     InstanceLayoutGeneric *ilg,
						     const ProfilingRequestSet& prs,
						     Event wait_on)
    {
      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
      RegionInstanceImpl *impl = m_impl->new_instance();
      // we can fail to get a valid pointer if we are out of instance slots
      if(!impl) {
	inst = RegionInstance::NO_INST;
	// import the profiling requests to see if anybody is paying attention to
	//  failure
	ProfilingMeasurementCollection pmc;
	pmc.import_requests(prs);
	if(pmc.wants_measurement<ProfilingMeasurements::InstanceStatus>()) {
	  ProfilingMeasurements::InstanceStatus stat;
	  stat.result = ProfilingMeasurements::InstanceStatus::INSTANCE_COUNT_EXCEEDED;
	  stat.error_code = 0;
	  pmc.add_measurement(stat);
	} else {
	  // fatal error
	  log_inst.fatal() << "FATAL: instance count exceeded for memory " << memory;
	  assert(0);
	}
	// generate a poisoned event for completion
	GenEventImpl *ev = GenEventImpl::create_genevent();
	Event ready_event = ev->current_event();
	GenEventImpl::trigger(ready_event, true /*poisoned*/);
	return ready_event;
      }

      impl->metadata.layout = ilg;
      
      if (!prs.empty()) {
        impl->requests = prs;
        impl->measurements.import_requests(impl->requests);
        if(impl->measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>())
          impl->timeline.record_create_time();
      }

      // request allocation of storage - a true response means it was serviced right
      //  away
      Event ready_event;
      if(m_impl->allocate_instance_storage(impl->me,
					   ilg->bytes_used,
					   ilg->alignment_reqd,
					   wait_on)) {
	assert(impl->metadata.inst_offset != (size_t)-1);
	if(impl->metadata.inst_offset != (size_t)-2) {
	  // successful allocation
	  ready_event = Event::NO_EVENT;
	  if(impl->measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>())
	    impl->timeline.record_ready_time();
	} else {
	  // generate a poisoned event for completion
	  GenEventImpl *ev = GenEventImpl::create_genevent();
	  ready_event = ev->current_event();
	  GenEventImpl::trigger(ready_event, true /*poisoned*/);
	}
      } else {
	// we will probably need an event to track when it is ready
	GenEventImpl *ev = GenEventImpl::create_genevent();
	ready_event = ev->current_event();
	bool alloc_done, alloc_successful;
	// use mutex to avoid race on allocation callback
	{
	  AutoHSLLock al(impl->mutex);
	  if(impl->metadata.inst_offset != (size_t)-1) {
	    alloc_done = true;
	    alloc_successful = (impl->metadata.inst_offset != (size_t)-2);
	  } else {
	    alloc_done = false;
	    alloc_successful = false;
	    impl->metadata.ready_event = ready_event;
	  }
	}
	if(alloc_done) {
	  // lost the race to the notification callback, so we trigger the
	  //  ready event ourselves
	  if(alloc_successful) {
	    if(impl->measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>())
	      impl->timeline.record_ready_time();
	    GenEventImpl::trigger(ready_event, false /*!poisoned*/);
	    ready_event = Event::NO_EVENT;
	  } else {
	    // poison the ready event and still return it
	    GenEventImpl::trigger(ready_event, true /*poisoned*/);
	  }
	}
      }

      inst = impl->me;
      log_inst.info() << "instance created: inst=" << inst << " bytes=" << ilg->bytes_used << " ready=" << ready_event;
      log_inst.debug() << "instance layout: inst=" << inst << " layout=" << *ilg;
      return ready_event;
    }

    /*static*/ Event RegionInstance::create_external(RegionInstance &inst,
                                                     Memory memory, uintptr_t base,
                                                     InstanceLayoutGeneric *ilg,
						     const ProfilingRequestSet& prs,
						     Event wait_on)
    {
      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
      RegionInstanceImpl *impl = m_impl->new_instance();

      // This actually doesn't have any bytes used in realm land
      ilg->bytes_used = 0;
      impl->metadata.layout = ilg;
      
      if (!prs.empty()) {
        impl->requests = prs;
        impl->measurements.import_requests(impl->requests);
        if(impl->measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>())
          impl->timeline.record_create_time();
      }

      // This is a little scary because the result could be negative, but we know
      // that unsigned undeflow produces correct results mod 2^64 so its ok
      // Pray that we never have to debug this
      unsigned char *impl_base = 
        (unsigned char*)m_impl->get_direct_ptr(0/*offset*/, 0/*size*/);
      size_t inst_offset = (size_t)(((unsigned char*)base) - impl_base);
#ifndef NDEBUG
      bool ok = 
#endif
        m_impl->allocate_instance_storage(impl->me,
					  ilg->bytes_used,
					  ilg->alignment_reqd,
					  wait_on, 
                                          inst_offset);
      assert(ok);

      inst = impl->me;
      log_inst.info() << "external instance created: inst=" << inst;
      log_inst.debug() << "external instance layout: inst=" << inst << " layout=" << *ilg;
      return Event::NO_EVENT;
    }

    void RegionInstance::destroy(Event wait_on /*= Event::NO_EVENT*/) const
    {
      // we can immediately turn this into a (possibly-preconditioned) request to
      //  deallocate the instance's storage - the eventual callback from that
      //  will be what actually destroys the instance
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      // TODO: send destruction request through so memory can see it, even
      //  if it's not ready
      bool poisoned = false;
      if(!wait_on.has_triggered_faultaware(poisoned)) {
	EventImpl::add_waiter(wait_on, new DeferredInstDestroy(*this));
        return;
      }
      // a poisoned precondition silently cancels the deletion - up to
      //  requestor to realize this has occurred since the deletion does
      //  not have its own completion event
      if(poisoned)
	return;

      log_inst.info() << "instance destroyed: inst=" << *this;

      // this does the right thing even though we're using an instance ID
      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(*this);
      mem_impl->release_instance_storage(*this, wait_on);
    }

    void RegionInstance::destroy(const std::vector<DestroyedField>& destroyed_fields,
				 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: actually call destructor
      if(!destroyed_fields.empty()) {
	log_inst.warning() << "WARNING: field destructors ignored - inst=" << *this;
      }
      destroy(wait_on);
    }

    /*static*/ const RegionInstance RegionInstance::NO_INST = { 0 };

    // a generic accessor just holds a pointer to the impl and passes all 
    //  requests through
    LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic> RegionInstance::get_accessor(void) const
    {
      // request metadata (if needed), but don't block on it yet
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(*this);
      // have to stall on metadata if it's not available...
      if(!i_impl->metadata.is_valid())
	i_impl->request_metadata().wait();
      assert(i_impl->metadata.layout);
	
      return LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic>(LegionRuntime::Accessor::AccessorType::Generic::Untyped(*this));
    }

    const InstanceLayoutGeneric *RegionInstance::get_layout(void) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // have to stall on metadata if it's not available...
      if(!r_impl->metadata.is_valid())
	r_impl->request_metadata().wait();
      assert(r_impl->metadata.layout);
      return r_impl->metadata.layout;
    }

    void RegionInstance::read_untyped(size_t offset, void *data, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // have to stall on metadata if it's not available...
      if(!r_impl->metadata.is_valid())
	r_impl->request_metadata().wait();
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      mem->get_bytes(r_impl->metadata.inst_offset + offset, data, datalen);
    }

    void RegionInstance::write_untyped(size_t offset, const void *data, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // have to stall on metadata if it's not available...
      if(!r_impl->metadata.is_valid())
	r_impl->request_metadata().wait();
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      mem->put_bytes(r_impl->metadata.inst_offset + offset, data, datalen);
    }

    void RegionInstance::reduce_apply_untyped(size_t offset, ReductionOpID redop_id,
					      const void *data, size_t datalen,
					      bool exclusive /*= false*/) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // have to stall on metadata if it's not available...
      if(!r_impl->metadata.is_valid())
	r_impl->request_metadata().wait();
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];
      assert(redop);
      // data should match RHS size
      assert(datalen == redop->sizeof_rhs);
      // can we run the reduction op directly on the memory location?
      void *ptr = mem->get_direct_ptr(r_impl->metadata.inst_offset + offset,
				      redop->sizeof_lhs);
      if(ptr) {
	redop->apply(ptr, data, 1, exclusive);
      } else {
	// we have to do separate get/put, which means we cannot supply
	//  atomicity in the !exclusive case
	assert(exclusive);
	void *lhs_copy = alloca(redop->sizeof_lhs);
	mem->get_bytes(r_impl->metadata.inst_offset + offset,
		       lhs_copy, redop->sizeof_lhs);
	redop->apply(lhs_copy, data, 1, true /*always exclusive*/);
	mem->put_bytes(r_impl->metadata.inst_offset + offset,
		       lhs_copy, redop->sizeof_lhs);
      }
    }

    void RegionInstance::reduce_fold_untyped(size_t offset, ReductionOpID redop_id,
					     const void *data, size_t datalen,
					     bool exclusive /*= false*/) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // have to stall on metadata if it's not available...
      if(!r_impl->metadata.is_valid())
	r_impl->request_metadata().wait();
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];
      assert(redop);
      // data should match RHS size
      assert(datalen == redop->sizeof_rhs);
      // can we run the reduction op directly on the memory location?
      void *ptr = mem->get_direct_ptr(r_impl->metadata.inst_offset + offset,
				      redop->sizeof_rhs);
      if(ptr) {
	redop->fold(ptr, data, 1, exclusive);
      } else {
	// we have to do separate get/put, which means we cannot supply
	//  atomicity in the !exclusive case
	assert(exclusive);
	void *rhs1_copy = alloca(redop->sizeof_rhs);
	mem->get_bytes(r_impl->metadata.inst_offset + offset,
		       rhs1_copy, redop->sizeof_rhs);
	redop->fold(rhs1_copy, data, 1, true /*always exclusive*/);
	mem->put_bytes(r_impl->metadata.inst_offset + offset,
		       rhs1_copy, redop->sizeof_rhs);
      }
    }

    // returns a null pointer if the instance storage cannot be directly
    //  accessed via load/store instructions
    void *RegionInstance::pointer_untyped(size_t offset, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // have to stall on metadata if it's not available...
      if(!r_impl->metadata.is_valid())
	r_impl->request_metadata().wait();
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      void *ptr = mem->get_direct_ptr(r_impl->metadata.inst_offset + offset,
				      datalen);
      return ptr;
    }

    void RegionInstance::get_strided_access_parameters(size_t start, size_t count,
						       ptrdiff_t field_offset, size_t field_size,
						       intptr_t& base, ptrdiff_t& stride)
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);

      // TODO: make sure we're in range

      void *orig_base = 0;
      size_t orig_stride = 0;
      bool ok = r_impl->get_strided_parameters(orig_base, orig_stride, field_offset);
      assert(ok);
      base = reinterpret_cast<intptr_t>(orig_base);
      stride = orig_stride;
    }

    void RegionInstance::report_instance_fault(int reason,
					       const void *reason_data,
					       size_t reason_size) const
    {
      assert(0);
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstanceImpl
  //

    RegionInstanceImpl::RegionInstanceImpl(RegionInstance _me, Memory _memory)
      : me(_me), memory(_memory) //, lis(0)
    {
      lock.init(ID(me).convert<Reservation>(), ID(me).instance.creator_node);
      lock.in_use = true;

      metadata.inst_offset = (size_t)-1;
      metadata.ready_event = Event::NO_EVENT;
      metadata.layout = 0;
      
      // Initialize this in case the user asks for profiling information
      timeline.instance = _me;
    }

    RegionInstanceImpl::~RegionInstanceImpl(void)
    {
      if(metadata.is_valid())
	delete metadata.layout;
    }

    void RegionInstanceImpl::notify_allocation(bool success, size_t offset)
    {
      if(!success) {
	// if somebody is listening to profiling measurements, we report
	//  a failed allocation through that channel - if not, we explode
	bool report_failure = (measurements.wants_measurement<ProfilingMeasurements::InstanceStatus>() ||
			       measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>());
	if(report_failure) {
	  log_inst.info() << "allocation failed: inst=" << me;
	  if(measurements.wants_measurement<ProfilingMeasurements::InstanceStatus>()) {
	    ProfilingMeasurements::InstanceStatus stat;
	    stat.result = ProfilingMeasurements::InstanceStatus::FAILED_ALLOCATION;
	    stat.error_code = 0;
	    measurements.add_measurement(stat);
	  }

	  if(measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>()) {
	    ProfilingMeasurements::InstanceAllocResult result;
	    result.success = false;
	    measurements.add_measurement(result);
	  }
	  
	  // send any remaining incomplete profiling responses
	  measurements.send_responses(requests);

          // clear the measurments after we send the response
          measurements.clear();

	  // poison the completion event, if it exists
	  Event ready_event = Event::NO_EVENT;
	  {
	    AutoHSLLock al(mutex);
	    ready_event = metadata.ready_event;
	    metadata.ready_event = Event::NO_EVENT;
	    metadata.inst_offset = (size_t)-2;
	  }
	  if(ready_event.exists())
	    GenEventImpl::trigger(ready_event, true /*poisoned*/);
	  return;
	} else {
	  log_inst.fatal() << "instance allocation failed - out of memory in mem " << memory;
	  exit(1);
	}
      }

      log_inst.debug() << "allocation completed: inst=" << me << " offset=" << offset;

      // before we publish the offset, we need to update the layout
      // SJT: or not?  that might be part of RegionInstance::get_base_address?
      //metadata.layout->relocate(offset);

      // update must be performed with the metadata mutex held to make sure there
      //  are no races between it and getting the ready event 
      Event ready_event;
      {
	AutoHSLLock al(mutex);
	ready_event = metadata.ready_event;
	metadata.ready_event = Event::NO_EVENT;
	metadata.inst_offset = offset;
      }
      if(ready_event.exists())
	GenEventImpl::trigger(ready_event, false /*!poisoned*/);

      // metadata is now valid and can be shared
      NodeSet early_reqs;
      metadata.mark_valid(early_reqs);
      if(!early_reqs.empty()) {
	log_inst.debug() << "sending instance metadata to early requestors: isnt=" << me;
	size_t datalen = 0;
	void *data = metadata.serialize(datalen);
	MetadataResponseMessage::broadcast_request(early_reqs, ID(me).id, data, datalen);
	free(data);
      }

      if(measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>()) {
	ProfilingMeasurements::InstanceAllocResult result;
	result.success = true;
	measurements.add_measurement(result);
      }

      if(measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>()) {
	timeline.record_ready_time();
      }

      // the InstanceMemoryUsage measurement is added at creation time for
      //  profilers that want that before instance deletion occurs
      if(measurements.wants_measurement<ProfilingMeasurements::InstanceMemoryUsage>()) {
	ProfilingMeasurements::InstanceMemoryUsage usage;
	usage.instance = me;
	usage.memory = memory;
	usage.bytes = metadata.layout->bytes_used;
	measurements.add_measurement(usage);
      }
    }

    void RegionInstanceImpl::notify_deallocation(void)
    {
      log_inst.debug() << "deallocation completed: inst=" << me;

      if (measurements.wants_measurement<ProfilingMeasurements::InstanceStatus>()) {
	ProfilingMeasurements::InstanceStatus stat;
	stat.result = ProfilingMeasurements::InstanceStatus::DESTROYED_SUCCESSFULLY;
	stat.error_code = 0;
	measurements.add_measurement(stat);
      }

      if (measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>()) {
	timeline.record_delete_time();
	measurements.add_measurement(timeline);
      }

      // send any remaining incomplete profiling responses
      measurements.send_responses(requests);

      // was this a successfully allocatated instance?
      if(metadata.inst_offset != size_t(-2)) {
	// send any required invalidation messages for metadata
	bool recycle_now = metadata.initiate_cleanup(me.id);
	if(recycle_now)
	  recycle_instance();
      } else {
	// failed allocations never had valid metadata - recycle immediately
	recycle_instance();
      }
    }

    void RegionInstanceImpl::recycle_instance(void)
    {
      // delete an existing layout, if present
      if(metadata.layout) {
	delete metadata.layout;
	metadata.layout = 0;
      }

      // set the offset back to the "unallocated" value
      metadata.inst_offset = size_t(-1);

      measurements.clear();

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
      m_impl->release_instance(me);
    }

    // helper function to figure out which field we're in
    void find_field_start(const std::vector<size_t>& field_sizes, off_t byte_offset,
			  size_t size, off_t& field_start, int& field_size)
    {
      off_t start = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++) {
	assert((*it) > 0);
	if(byte_offset < (off_t)(*it)) {
	  if ((off_t)(byte_offset + size) > (off_t)(*it)) {
      log_inst.error("Requested field does not match the expected field size");
      assert(false);
    }
	  field_start = start;
	  field_size = (*it);
	  return;
	}
	start += (*it);
	byte_offset -= (*it);
      }
      assert(0);
    }

    bool RegionInstanceImpl::get_strided_parameters(void *&base, size_t &stride,
						      off_t field_offset)
    {
      MemoryImpl *mem = get_runtime()->get_memory_impl(memory);

      // this exists for compatibility and assumes N=1, T=long long
      const InstanceLayout<1,long long> *inst_layout = dynamic_cast<const InstanceLayout<1,long long> *>(metadata.layout);
      assert(inst_layout != 0);

      // look up the right field
      std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(field_offset);
      assert(it != inst_layout->fields.end());

      // hand out a null pointer for empty instances (stride can be whatever
      //  the caller wants)
      if(inst_layout->piece_lists[it->second.list_idx].pieces.empty()) {
	base = 0;
	return true;
      }

      // also only works for a single piece
      assert(inst_layout->piece_lists[it->second.list_idx].pieces.size() == 1);
      const InstanceLayoutPiece<1,long long> *piece = inst_layout->piece_lists[it->second.list_idx].pieces[0];
      assert((piece->layout_type == InstanceLayoutPiece<1,long long>::AffineLayoutType));
      const AffineLayoutPiece<1,long long> *affine = static_cast<const AffineLayoutPiece<1,long long> *>(piece);

      // if the caller wants a particular stride and we differ (and have more
      //  than one element), fail
      if(stride != 0) {
        if((affine->bounds.hi[0] > affine->bounds.lo[0]) &&
           (affine->strides[0] != stride))
          return false;
      } else {
        stride = affine->strides[0];
      }

      // find the offset of the first and last elements and then try to
      //  turn that into a direct memory pointer
      size_t start_offset = (metadata.inst_offset +
                             affine->offset +
                             affine->strides.dot(affine->bounds.lo) +
                             it->second.rel_offset);
      size_t total_bytes = (it->second.size_in_bytes + 
                            affine->strides[0] * (affine->bounds.hi -
                                                  affine->bounds.lo));
 
      base = mem->get_direct_ptr(start_offset, total_bytes);
      if (!base) return false;

      // now adjust the base pointer so that we can use absolute indexing
      //  again
      // careful - have to use 'stride' instead of 'affine->strides' in
      //  case we agreed to the caller's incorrect stride when size == 1
      base = ((char *)base) - (stride * affine->bounds.lo[0]);
     
      return true;
    }

    void *RegionInstanceImpl::Metadata::serialize(size_t& out_size) const
    {
      Serialization::DynamicBufferSerializer dbs(128);

      bool ok = ((dbs << alloc_offset) &&
		 (dbs << size) &&
		 (dbs << redopid) &&
		 (dbs << count_offset) &&
		 (dbs << red_list_size) &&
		 (dbs << block_size) &&
		 (dbs << elmt_size) &&
		 (dbs << field_sizes) &&
		 (dbs << parent_inst) &&
		 (dbs << inst_offset) &&
		 (dbs << filename) &&
		 (dbs << *layout));
      assert(ok);

      out_size = dbs.bytes_used();
      return dbs.detach_buffer(0 /*trim*/);
    }

    void RegionInstanceImpl::Metadata::deserialize(const void *in_data, size_t in_size)
    {
      Serialization::FixedBufferDeserializer fbd(in_data, in_size);

      bool ok = ((fbd >> alloc_offset) &&
		 (fbd >> size) &&
		 (fbd >> redopid) &&
		 (fbd >> count_offset) &&
		 (fbd >> red_list_size) &&
		 (fbd >> block_size) &&
		 (fbd >> elmt_size) &&
		 (fbd >> field_sizes) &&
		 (fbd >> parent_inst) &&
		 (fbd >> inst_offset) &&
		 (fbd >> filename));
      if(ok)
	layout = InstanceLayoutGeneric::deserialize_new(fbd);
      assert(ok && (layout != 0) && (fbd.bytes_left() == 0));
    }

#ifdef POINTER_CHECKS
    void RegionInstanceImpl::verify_access(unsigned ptr)
    {
      StaticAccess<RegionInstanceImpl> data(this);
      const ElementMask &mask = data->is.get_valid_mask();
      if (!mask.is_set(ptr))
      {
        fprintf(stderr,"ERROR: Accessing invalid pointer %d in logical region " IDFMT "\n",ptr,data->is.id);
        assert(false);
      }
    }
#endif

    static inline off_t calc_mem_loc(off_t alloc_offset, off_t field_start, int field_size, int elmt_size,
				     int block_size, int index)
    {
      return (alloc_offset +                                      // start address
	      ((index / block_size) * block_size * elmt_size) +   // full blocks
	      (field_start * block_size) +                        // skip other fields
	      ((index % block_size) * field_size));               // some some of our fields within our block
    }

    void RegionInstanceImpl::get_bytes(int index, off_t byte_offset, void *dst, size_t size)
    {
      // must have valid data by now - block if we have to
      metadata.await_data();
      off_t o;
      if(metadata.block_size == 1) {
	// no blocking - don't need to know about field boundaries
	o = metadata.alloc_offset + (index * metadata.elmt_size) + byte_offset;
      } else {
	off_t field_start=0;
	int field_size=0;
	find_field_start(metadata.field_sizes, byte_offset, size, field_start, field_size);
        o = calc_mem_loc(metadata.alloc_offset, field_start, field_size,
                         metadata.elmt_size, metadata.block_size, index);

      }
      MemoryImpl *m = get_runtime()->get_memory_impl(memory);
      m->get_bytes(o, dst, size);
    }

    void RegionInstanceImpl::put_bytes(int index, off_t byte_offset, const void *src, size_t size)
    {
      // must have valid data by now - block if we have to
      metadata.await_data();
      off_t o;
      if(metadata.block_size == 1) {
	// no blocking - don't need to know about field boundaries
	o = metadata.alloc_offset + (index * metadata.elmt_size) + byte_offset;
      } else {
	off_t field_start=0;
	int field_size=0;
	find_field_start(metadata.field_sizes, byte_offset, size, field_start, field_size);
        o = calc_mem_loc(metadata.alloc_offset, field_start, field_size,
                         metadata.elmt_size, metadata.block_size, index);
      }
      MemoryImpl *m = get_runtime()->get_memory_impl(memory);
      m->put_bytes(o, src, size);
    }

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, AffineLayoutPiece<N,T> > AffineLayoutPiece<N,T>::serdez_subclass;

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutGeneric, InstanceLayout<N,T> > InstanceLayout<N,T>::serdez_subclass;

#define DOIT(N,T) \
  template class AffineLayoutPiece<N,T>; \
  template class InstanceLayout<N,T>;
  FOREACH_NT(DOIT)
#undef DOIT

#ifdef USE_HDF
  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, HDF5LayoutPiece<N,T> > HDF5LayoutPiece<N,T>::serdez_subclass;

#define DOIT(N,T) \
  template class HDF5LayoutPiece<N,T>;
  FOREACH_NT(DOIT)
#undef DOIT
#endif

}; // namespace Realm
