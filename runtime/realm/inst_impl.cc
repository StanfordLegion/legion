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

#include "realm/inst_impl.h"

#include "realm/event_impl.h"
#include "realm/mem_impl.h"
#include "realm/logging.h"
#include "realm/runtime_impl.h"
#include "realm/deppart/inst_helper.h"
#ifdef REALM_USE_HDF5
#include "realm/hdf5/hdf5_access.h"
#endif

TYPE_IS_SERIALIZABLE(Realm::InstanceLayoutGeneric::FieldLayout);

namespace Realm {

  Logger log_inst("inst");

  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstanceImpl::DeferredCreate
  //

  void RegionInstanceImpl::DeferredCreate::defer(RegionInstanceImpl *_inst,
						 MemoryImpl *_mem,
						 size_t _bytes,
						 size_t _align,
						 bool _need_alloc_result,
 						 Event wait_on)
  {
    inst = _inst;
    mem = _mem;
    bytes = _bytes;
    align = _align;
    need_alloc_result = _need_alloc_result;
    EventImpl::add_waiter(wait_on, this);
  }

  void RegionInstanceImpl::DeferredCreate::event_triggered(bool poisoned)
  {
    if(poisoned)
      log_poison.info() << "poisoned deferred instance creation skipped - inst=" << inst;
    
    mem->deferred_creation_triggered(inst, bytes, align,
				     need_alloc_result, poisoned);
  }

  void RegionInstanceImpl::DeferredCreate::print(std::ostream& os) const
  {
    os << "deferred instance creation";
  }

  Event RegionInstanceImpl::DeferredCreate::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstanceImpl::DeferredDestroy
  //

  void RegionInstanceImpl::DeferredDestroy::defer(RegionInstanceImpl *_inst,
						  MemoryImpl *_mem,
						  Event wait_on)
  {
    inst = _inst;
    mem = _mem;
    EventImpl::add_waiter(wait_on, this);
  }

  void RegionInstanceImpl::DeferredDestroy::event_triggered(bool poisoned)
  {
    if(poisoned)
      log_poison.info() << "poisoned deferred instance destruction skipped - POSSIBLE LEAK - inst=" << inst;
    
    mem->deferred_destruction_triggered(inst, poisoned);
  }

  void RegionInstanceImpl::DeferredDestroy::print(std::ostream& os) const
  {
    os << "deferred instance destruction";
  }

  Event RegionInstanceImpl::DeferredDestroy::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance
  //

    AddressSpace RegionInstance::address_space(void) const
    {
      return ID(id).instance_owner_node();
    }

    Memory RegionInstance::get_location(void) const
    {
      return (exists() ?
	        ID::make_memory(ID(id).instance_owner_node(),
				ID(id).instance_mem_idx()).convert<Memory>() :
	        Memory::NO_MEMORY);
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
	bool reported = false;
	if(pmc.wants_measurement<ProfilingMeasurements::InstanceStatus>()) {
	  ProfilingMeasurements::InstanceStatus stat;
	  stat.result = ProfilingMeasurements::InstanceStatus::INSTANCE_COUNT_EXCEEDED;
	  stat.error_code = 0;
	  pmc.add_measurement(stat);
	  reported = true;
	}
	if(pmc.wants_measurement<ProfilingMeasurements::InstanceAbnormalStatus>()) {
	  ProfilingMeasurements::InstanceAbnormalStatus stat;
	  stat.result = ProfilingMeasurements::InstanceStatus::INSTANCE_COUNT_EXCEEDED;
	  stat.error_code = 0;
	  pmc.add_measurement(stat);
	  reported = true;
	}
	if(pmc.wants_measurement<ProfilingMeasurements::InstanceAllocResult>()) {
	  ProfilingMeasurements::InstanceAllocResult result;
	  result.success = false;
	  pmc.add_measurement(result);
	}
	if(!reported) {
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

      // set this handle before we do anything that can result in a
      //  profiling callback containing this instance handle
      inst = impl->me;

      impl->metadata.layout = ilg;

      bool need_alloc_result = false;
      if (!prs.empty()) {
        impl->requests = prs;
        impl->measurements.import_requests(impl->requests);
        if(impl->measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>())
          impl->timeline.record_create_time();
	need_alloc_result = impl->measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>();
      }

      impl->metadata.need_alloc_result = need_alloc_result;
      impl->metadata.need_notify_dealloc = false;

      log_inst.debug() << "instance layout: inst=" << inst << " layout=" << *ilg;

      // request allocation of storage - note that due to the asynchronous
      //  nature of any profiling responses, it is not safe to refer to the
      //  instance metadata (whether the allocation succeeded or not) after
      //  this point)
      Event ready_event;
      switch(m_impl->allocate_instance_storage(impl->me,
					       ilg->bytes_used,
					       ilg->alignment_reqd,
					       need_alloc_result,
					       wait_on)) {
      case MemoryImpl::ALLOC_INSTANT_SUCCESS:
	{
	  // successful allocation
	  assert(impl->metadata.inst_offset <= RegionInstanceImpl::INSTOFFSET_MAXVALID);
	  ready_event = Event::NO_EVENT;
	  break;
	}

      case MemoryImpl::ALLOC_INSTANT_FAILURE:
      case MemoryImpl::ALLOC_CANCELLED:
	{
	  // generate a poisoned event for completion
	  // NOTE: it is unsafe to look at the impl->metadata or the 
	  //  passed-in instance layout at this point due to the possibility
	  //  of an asynchronous destruction of the instance in a profiling
	  //  handler
	  GenEventImpl *ev = GenEventImpl::create_genevent();
	  ready_event = ev->current_event();
	  GenEventImpl::trigger(ready_event, true /*poisoned*/);
	  break;
	}

      case MemoryImpl::ALLOC_DEFERRED:
	{
	  // we will probably need an event to track when it is ready
	  GenEventImpl *ev = GenEventImpl::create_genevent();
	  ready_event = ev->current_event();
	  bool alloc_done, alloc_successful;
	  // use mutex to avoid race on allocation callback
	  {
	    AutoLock<> al(impl->mutex);
	    switch(impl->metadata.inst_offset) {
	    case RegionInstanceImpl::INSTOFFSET_UNALLOCATED:
	    case RegionInstanceImpl::INSTOFFSET_DELAYEDALLOC:
	    case RegionInstanceImpl::INSTOFFSET_DELAYEDDESTROY:
	      {
		alloc_done = false;
		alloc_successful = false;
		impl->metadata.ready_event = ready_event;
		break;
	      }
	    case RegionInstanceImpl::INSTOFFSET_FAILED:
	      {
		alloc_done = true;
		alloc_successful = false;
		break;
	      }
	    default:
	      {
		alloc_done = true;
		alloc_successful = true;
		break;
	      }
	    }
	  }
	  if(alloc_done) {
	    // lost the race to the notification callback, so we trigger the
	    //  ready event ourselves
	    if(alloc_successful) {
	      GenEventImpl::trigger(ready_event, false /*!poisoned*/);
	      ready_event = Event::NO_EVENT;
	    } else {
	      // poison the ready event and still return it
	      GenEventImpl::trigger(ready_event, true /*poisoned*/);
	    }
	  }
	  break;
	}

      case MemoryImpl::ALLOC_EVENTUAL_SUCCESS:
      case MemoryImpl::ALLOC_EVENTUAL_FAILURE:
	// should not occur
	assert(0);
      }

      log_inst.info() << "instance created: inst=" << inst << " bytes=" << ilg->bytes_used << " ready=" << ready_event;
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
      
      bool need_alloc_result = false;
      if (!prs.empty()) {
        impl->requests = prs;
        impl->measurements.import_requests(impl->requests);
        if(impl->measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>())
          impl->timeline.record_create_time();
	need_alloc_result = impl->measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>();
      }

      impl->metadata.need_alloc_result = need_alloc_result;
      impl->metadata.need_notify_dealloc = false;

      // This is a little scary because the result could be negative, but we know
      // that unsigned undeflow produces correct results mod 2^64 so its ok
      // Pray that we never have to debug this
      unsigned char *impl_base = 
        (unsigned char*)m_impl->get_direct_ptr(0/*offset*/, 0/*size*/);
      size_t inst_offset = (size_t)(((unsigned char*)base) - impl_base);
#ifndef NDEBUG
      MemoryImpl::AllocationResult result =
#endif
        m_impl->allocate_instance_storage(impl->me,
					  ilg->bytes_used,
					  ilg->alignment_reqd,
					  need_alloc_result,
					  wait_on, 
                                          inst_offset);
      assert(result == MemoryImpl::ALLOC_INSTANT_SUCCESS);

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

      log_inst.info() << "instance destroyed: inst=" << *this << " wait_on=" << wait_on;

      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(*this);
      RegionInstanceImpl *inst_impl = mem_impl->get_instance(*this);
      mem_impl->release_instance_storage(inst_impl, wait_on);
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

    // before you can get an instance's index space or construct an accessor for
    //  a given processor, the necessary metadata for the instance must be
    //  available on to that processor
    // this can require network communication and/or completion of the actual
    //  allocation, so an event is returned and (as always) the application
    //  must decide when/where to handle this precondition
    Event RegionInstance::fetch_metadata(Processor target) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);

      NodeID target_node = ID(target).proc_owner_node();
      if(target_node == Network::my_node_id) {
	// local metadata request
	return r_impl->request_metadata();
      } else {
	// prefetch on other node's behalf
	return r_impl->prefetch_metadata(target_node);
      }
    }

    const InstanceLayoutGeneric *RegionInstance::get_layout(void) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      return r_impl->metadata.layout;
    }

    void RegionInstance::read_untyped(size_t offset, void *data, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      mem->get_bytes(r_impl->metadata.inst_offset + offset, data, datalen);
    }

    void RegionInstance::write_untyped(size_t offset, const void *data, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      mem->put_bytes(r_impl->metadata.inst_offset + offset, data, datalen);
    }

    void RegionInstance::reduce_apply_untyped(size_t offset, ReductionOpID redop_id,
					      const void *data, size_t datalen,
					      bool exclusive /*= false*/) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      if(redop == 0) {
	log_inst.fatal() << "no reduction op registered for ID " << redop_id;
	abort();
      }
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
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      if(redop == 0) {
	log_inst.fatal() << "no reduction op registered for ID " << redop_id;
	abort();
      }
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
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
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
      lock.init(ID(me).convert<Reservation>(), ID(me).instance_creator_node());
      lock.in_use = true;

      metadata.inst_offset = INSTOFFSET_UNALLOCATED;
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

  void RegionInstanceImpl::notify_allocation(MemoryImpl::AllocationResult result,
					     size_t offset)
    {
      using namespace ProfilingMeasurements;

      if((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ||
	 (result == MemoryImpl::ALLOC_EVENTUAL_FAILURE) ||
	 (result == MemoryImpl::ALLOC_CANCELLED)) {
	// if somebody is listening to profiling measurements, we report
	//  a failed allocation through that channel - if not, we explode
	// exception: InstanceAllocResult is not enough for EVENTUAL_FAILURE,
	//  since we would have already said we thought it would succeed
	bool report_failure = (measurements.wants_measurement<InstanceStatus>() ||
			       measurements.wants_measurement<InstanceAbnormalStatus>() ||
			       (measurements.wants_measurement<InstanceAllocResult>() &&
				(result != MemoryImpl::ALLOC_EVENTUAL_FAILURE)));
	if(!report_failure) {
	  if((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ||
	     (result == MemoryImpl::ALLOC_EVENTUAL_FAILURE)) {
	    log_inst.fatal() << "instance allocation failed - out of memory in mem " << memory;
	    abort();
	  }

	  // exception: allocations that were cancelled would have had some
	  //  error response reported further up the chain, so let this
	  //  one slide
	  assert(result == MemoryImpl::ALLOC_CANCELLED);
	}
	
	log_inst.info() << "allocation failed: inst=" << me;

	// poison the completion event, if it exists
	Event ready_event = Event::NO_EVENT;
	{
	  AutoLock<> al(mutex);
	  ready_event = metadata.ready_event;
	  metadata.ready_event = Event::NO_EVENT;
	  metadata.inst_offset = (size_t)-2;

	  // adding measurements is not thread safe w.r.t. a deferral
	  //  message, so do it with lock held
	  if(measurements.wants_measurement<InstanceStatus>()) {
	    InstanceStatus stat;
	    stat.result = ((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ?
                           InstanceStatus::FAILED_ALLOCATION :
			   InstanceStatus::CANCELLED_ALLOCATION);
	    stat.error_code = 0;
	    measurements.add_measurement(stat);
	  }

	  if(measurements.wants_measurement<InstanceAbnormalStatus>()) {
	    InstanceAbnormalStatus stat;
	    stat.result = ((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ?
                             InstanceStatus::FAILED_ALLOCATION :
			     InstanceStatus::CANCELLED_ALLOCATION);
	    stat.error_code = 0;
	    measurements.add_measurement(stat);
	  }

	  if(metadata.need_alloc_result) {
#ifdef DEBUG_REALM
	    assert(measurements.wants_measurement<InstanceAllocResult>());
	    assert(!metadata.need_notify_dealloc);
#endif

	    // this is either the only result we will get or has raced ahead of
	    //  the deferral message
	    metadata.need_alloc_result = false;

	    InstanceAllocResult result;
	    result.success = false;
	    measurements.add_measurement(result);
	  }
	}
	  
	// send any remaining incomplete profiling responses
	measurements.send_responses(requests);

	// clear the measurments after we send the response
	measurements.clear();

	if(ready_event.exists())
	  GenEventImpl::trigger(ready_event, true /*poisoned*/);
	return;
      }

      if(result == MemoryImpl::ALLOC_DEFERRED) {
	// this should only be received if an InstanceAllocRequest measurement
	//  was requested, but we have to be careful about recording the
	//  expected-future-success because it may race with the actual success
	//  (or unexpected failure), so use the mutex
	bool need_notify_dealloc = false;
	{
	  AutoLock<> al(mutex);

	  if(metadata.need_alloc_result) {
#ifdef DEBUG_REALM
	    assert(measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>());
#endif
	    ProfilingMeasurements::InstanceAllocResult result;
	    result.success = true;
	    measurements.add_measurement(result);

	    metadata.need_alloc_result = false;

	    // if we were super-slow, notification of the subsequent
	    //  deallocation may have been delayed
	    need_notify_dealloc = metadata.need_notify_dealloc;
	    metadata.need_notify_dealloc = false;
	  }
	}

	if(need_notify_dealloc)
	  notify_deallocation();

	return;
      }

      log_inst.debug() << "allocation completed: inst=" << me << " offset=" << offset;

      // before we publish the offset, we need to update the layout
      // SJT: or not?  that might be part of RegionInstance::get_base_address?
      //metadata.layout->relocate(offset);

      // update must be performed with the metadata mutex held to make sure there
      //  are no races between it and getting the ready event 
      Event ready_event;
      {
	AutoLock<> al(mutex);
	ready_event = metadata.ready_event;
	metadata.ready_event = Event::NO_EVENT;
	metadata.inst_offset = offset;

	// adding measurements is not thread safe w.r.t. a deferral
	//  message, so do it with lock held
	if(metadata.need_alloc_result) {
#ifdef DEBUG_REALM
	  assert(measurements.wants_measurement<InstanceAllocResult>());
	  assert(!metadata.need_notify_dealloc);
#endif

	  // this is either the only result we will get or has raced ahead of
	  //  the deferral message
	  metadata.need_alloc_result = false;

	  ProfilingMeasurements::InstanceAllocResult result;
	  result.success = true;
	  measurements.add_measurement(result);
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
      if(ready_event.exists())
	GenEventImpl::trigger(ready_event, false /*!poisoned*/);

      // metadata is now valid and can be shared
      NodeSet early_reqs;
      metadata.mark_valid(early_reqs);
      if(!early_reqs.empty()) {
	log_inst.debug() << "sending instance metadata to early requestors: isnt=" << me;
	ActiveMessage<MetadataResponseMessage> amsg(early_reqs,65536);
	metadata.serialize_msg(amsg);
	amsg->id = ID(me).id;
	amsg.commit();
      }

      if(measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>()) {
	timeline.record_ready_time();
      }

    }

    void RegionInstanceImpl::notify_deallocation(void)
    {
      // handle race with a slow DEFERRED notification
      bool notification_delayed = false;
      {
	AutoLock<> al(mutex);
	if(metadata.need_alloc_result) {
	  metadata.need_notify_dealloc = true;
	  notification_delayed = true;
	}
      }
      if(notification_delayed) return;

      log_inst.debug() << "deallocation completed: inst=" << me;

      // our instance better not be in the unallocated state...
      assert(metadata.inst_offset != INSTOFFSET_UNALLOCATED);
      assert(metadata.inst_offset != INSTOFFSET_DELAYEDALLOC);
      assert(metadata.inst_offset != INSTOFFSET_DELAYEDDESTROY);

      // was this a successfully allocatated instance?
      if(metadata.inst_offset != INSTOFFSET_FAILED) {
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

	// flush the remote prefetch cache
	{
	  AutoLock<> al(mutex);
	  prefetch_events.clear();
	}

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
      metadata.inst_offset = INSTOFFSET_UNALLOCATED;

      measurements.clear();

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
      m_impl->release_instance(me);
    }

    Event RegionInstanceImpl::prefetch_metadata(NodeID target_node)
    {
      assert(target_node != Network::my_node_id);

      Event e = Event::NO_EVENT;
      {
	AutoLock<> al(mutex);
	std::map<NodeID, Event>::iterator it = prefetch_events.find(target_node);
	if(it != prefetch_events.end())
	  return it->second;

	// have to make a new one
	e = GenEventImpl::create_genevent()->current_event();
	prefetch_events.insert(std::make_pair(target_node, e));
      }

      // send a message to the target node to fetch metadata
      // TODO: save a hop by sending request to owner directly?
      ActiveMessage<InstanceMetadataPrefetchRequest> amsg(target_node, 0);
      amsg->inst = me;
      amsg->valid_event = e;
      amsg.commit();

      return e;
    }

    /*static*/ void InstanceMetadataPrefetchRequest::handle_message(NodeID sender,
								    const InstanceMetadataPrefetchRequest& msg,
								    const void *data,
								    size_t datalen)
    {
      // make a local request and trigger the remote event based on the local one
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(msg.inst);
      Event e = r_impl->request_metadata();
      log_inst.info() << "metadata prefetch: inst=" << msg.inst
		      << " local=" << e << " remote=" << msg.valid_event;

      if(e.exists()) {
	GenEventImpl *e_impl = get_runtime()->get_genevent_impl(msg.valid_event);
	EventMerger *m = &(e_impl->merger);
	m->prepare_merger(msg.valid_event, false/*!ignore faults*/, 1);
	m->add_precondition(e);
	m->arm_merger();
      } else {
	GenEventImpl::trigger(msg.valid_event, false /*!poisoned*/);
      }	
    }

    ActiveMessageHandlerReg<InstanceMetadataPrefetchRequest> inst_prefetch_msg_handler;

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

  template <typename T>
  void RegionInstanceImpl::Metadata::serialize_msg(T& fbs) const
    {
      bool ok = ((fbs << alloc_offset) &&
		 (fbs << size) &&
		 (fbs << redopid) &&
		 (fbs << count_offset) &&
		 (fbs << red_list_size) &&
		 (fbs << block_size) &&
		 (fbs << elmt_size) &&
		 (fbs << field_sizes) &&
		 (fbs << parent_inst) &&
		 (fbs << inst_offset) &&
		 (fbs << filename) &&
		 (fbs << *layout));
      assert(ok);
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

    void RegionInstanceImpl::Metadata::do_invalidate(void)
    {
      // delete an existing layout, if present
      if(layout) {
	delete layout;
	layout = 0;
      }

      // set the offset back to the "unallocated" value
      inst_offset = INSTOFFSET_UNALLOCATED;
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

#ifdef REALM_USE_HDF5
  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, HDF5LayoutPiece<N,T> > HDF5LayoutPiece<N,T>::serdez_subclass;

#define DOIT(N,T) \
  template class HDF5LayoutPiece<N,T>;
  FOREACH_NT(DOIT)
#undef DOIT
#endif

}; // namespace Realm
