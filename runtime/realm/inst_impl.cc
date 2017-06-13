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

#include "inst_impl.h"

#include "event_impl.h"
#include "mem_impl.h"
#include "logging.h"
#include "runtime_impl.h"

namespace Realm {

  Logger log_inst("inst");

  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredInstDestroy
  //

    class DeferredInstDestroy : public EventWaiter {
    public:
      DeferredInstDestroy(RegionInstanceImpl *i) : impl(i) { }
      virtual ~DeferredInstDestroy(void) { }
    public:
      virtual bool event_triggered(Event e, bool poisoned)
      {
	// if input event is poisoned, do not attempt to destroy the lock
	// we don't have an output event here, so this may result in a leak if nobody is
	//  paying attention
	if(poisoned) {
	  log_poison.info() << "poisoned deferred instance destruction skipped - POSSIBLE LEAK - inst=" << impl->me;
	} else {
	  log_inst.info("instance destroyed: space=" IDFMT " id=" IDFMT "",
			impl->metadata.is.id, impl->me.id);
	  get_runtime()->get_memory_impl(impl->memory)->destroy_instance(impl->me, true); 
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
      RegionInstanceImpl *impl;
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
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(*this);
      return i_impl->memory;
    }

    void RegionInstance::destroy(Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(*this);
      if (!wait_on.has_triggered())
      {
	EventImpl::add_waiter(wait_on, new DeferredInstDestroy(i_impl));
        return;
      }

      log_inst.info("instance destroyed: space=" IDFMT " id=" IDFMT "",
	       i_impl->metadata.is.id, this->id);
      get_runtime()->get_memory_impl(i_impl->memory)->destroy_instance(*this, true);
    }

    void RegionInstance::destroy(const std::vector<DestroyedField>& destroyed_fields,
				 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: actually call destructor
      assert(destroyed_fields.empty());
      destroy(wait_on);
    }

    /*static*/ const RegionInstance RegionInstance::NO_INST = { 0 };

    // a generic accessor just holds a pointer to the impl and passes all 
    //  requests through
    LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic> RegionInstance::get_accessor(void) const
    {
      // request metadata (if needed), but don't block on it yet
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(*this);
      Event e = i_impl->metadata.request_data(ID(id).instance.owner_node, id);
      if(!e.has_triggered())
	log_inst.info("requested metadata in accessor creation: " IDFMT, id);
	
      return LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic>(LegionRuntime::Accessor::AccessorType::Generic::Untyped((void *)i_impl));
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

    RegionInstanceImpl::RegionInstanceImpl(RegionInstance _me, IndexSpace _is, Memory _memory, 
					   off_t _offset, size_t _size, ReductionOpID _redopid,
					   const DomainLinearization& _linear, size_t _block_size,
					   size_t _elmt_size, const std::vector<size_t>& _field_sizes,
					   const ProfilingRequestSet &reqs,
					   off_t _count_offset /*= 0*/, off_t _red_list_size /*= 0*/,
					   RegionInstance _parent_inst /*= NO_INST*/)
      : me(_me), memory(_memory)
    {
      metadata.linearization = _linear;

      metadata.block_size = _block_size;
      metadata.elmt_size = _elmt_size;

      metadata.field_sizes = _field_sizes;

      metadata.is = _is;
      metadata.alloc_offset = _offset;
      //metadata.access_offset = _offset + _adjust;
      metadata.size = _size;
      
      //StaticAccess<IndexSpaceImpl> rdata(_is.impl());
      //locked_data.first_elmt = rdata->first_elmt;
      //locked_data.last_elmt = rdata->last_elmt;

      metadata.redopid = _redopid;
      metadata.count_offset = _count_offset;
      metadata.red_list_size = _red_list_size;
      metadata.parent_inst = _parent_inst;

      metadata.mark_valid();

      lock.init(ID(me).convert<Reservation>(), ID(me).instance.owner_node);
      lock.in_use = true;

      if (!reqs.empty()) {
        requests = reqs;
        measurements.import_requests(requests);
        if (measurements.wants_measurement<
                          ProfilingMeasurements::InstanceTimeline>()) {
          timeline.record_create_time();
        }
      }
    }

    // when we auto-create a remote instance, we don't know region/offset
    RegionInstanceImpl::RegionInstanceImpl(RegionInstance _me, Memory _memory)
      : me(_me), memory(_memory)
    {
      lock.init(ID(me).convert<Reservation>(), ID(me).instance.owner_node);
      lock.in_use = true;
    }

    RegionInstanceImpl::~RegionInstanceImpl(void) {}

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

    void RegionInstanceImpl::record_instance_usage(void)
    {
      // can't do this in the constructor because our ID isn't right yet...
      if(measurements.wants_measurement<ProfilingMeasurements::InstanceMemoryUsage>()) {
	ProfilingMeasurements::InstanceMemoryUsage usage;
	usage.instance = me;
	usage.memory = memory;
	usage.bytes = metadata.size;
	measurements.add_measurement(usage);
      }
    }

    bool RegionInstanceImpl::get_strided_parameters(void *&base, size_t &stride,
						      off_t field_offset)
    {
      MemoryImpl *mem = get_runtime()->get_memory_impl(memory);

      // must have valid data by now - block if we have to
      metadata.await_data();

      off_t offset = metadata.alloc_offset;
      size_t elmt_stride;
      
      if (metadata.block_size == 1) {
        offset += field_offset;
        elmt_stride = metadata.elmt_size;
      } else {
        off_t field_start=0;
        int field_size=0;
        find_field_start(metadata.field_sizes, field_offset, 1, field_start, field_size);

        offset += (field_start * metadata.block_size) + (field_offset - field_start);
	elmt_stride = field_size;
      }

      base = mem->get_direct_ptr(offset, 0);
      if (!base) return false;

      // if the caller wants a particular stride and we differ (and have more
      //  than one element), fail
      if(stride != 0) {
        if((stride != elmt_stride) && (metadata.size > metadata.elmt_size))
          return false;
      } else {
        stride = elmt_stride;
      }

      // if there's a per-element offset, apply it after we've agreed with the caller on 
      //  what we're pretending the stride is
      const DomainLinearization& dl = metadata.linearization;
      if(dl.get_dim() > 0) {
	// make sure this instance uses a 1-D linearization
	assert(dl.get_dim() == 1);

	LegionRuntime::Arrays::Mapping<1, 1> *mapping = dl.get_mapping<1>();
	LegionRuntime::Arrays::Rect<1> preimage = mapping->preimage((coord_t)0);
	assert(preimage.lo == preimage.hi);
	// double-check that whole range maps densely
	preimage.hi.x[0] += 1; // not perfect, but at least detects non-unit-stride case
	assert(mapping->image_is_dense(preimage));
	coord_t inst_first_elmt = preimage.lo[0];
	//printf("adjusting base by %d * %zd\n", inst_first_elmt, stride);
	base = ((char *)base) - inst_first_elmt * stride;
      }

      return true;
    }

    void RegionInstanceImpl::finalize_instance(void)
    {
      if (!requests.empty()) {
        if (measurements.wants_measurement<
                          ProfilingMeasurements::InstanceTimeline>()) {
	  // set the instance ID correctly now - it wasn't available at construction time
          timeline.instance = me;
          timeline.record_delete_time();
          measurements.add_measurement(timeline);
        }
        measurements.send_responses(requests);
        requests.clear();
      }
    }

    void *RegionInstanceImpl::Metadata::serialize(size_t& out_size) const
    {
      // figure out how much space we need
      out_size = (sizeof(IndexSpace) +
		  sizeof(off_t) +
		  sizeof(size_t) +
		  sizeof(ReductionOpID) +
		  sizeof(off_t) +
		  sizeof(off_t) +
		  sizeof(size_t) +
		  sizeof(size_t) +
		  sizeof(size_t) + (field_sizes.size() * sizeof(size_t)) +
		  sizeof(RegionInstance) +
		  (MAX_LINEARIZATION_LEN * sizeof(int)));
      void *data = malloc(out_size);
      char *pos = (char *)data;
#define S(val) do { memcpy(pos, &(val), sizeof(val)); pos += sizeof(val); } while(0)
      S(is);
      S(alloc_offset);
      S(size);
      S(redopid);
      S(count_offset);
      S(red_list_size);
      S(block_size);
      S(elmt_size);
      size_t l = field_sizes.size();
      S(l);
      for(size_t i = 0; i < l; i++) S(field_sizes[i]);
      S(parent_inst);
      linearization.serialize((int *)pos);
#undef S
      return data;
    }

    void RegionInstanceImpl::Metadata::deserialize(const void *in_data, size_t in_size)
    {
      const char *pos = (const char *)in_data;
#define S(val) do { memcpy(&(val), pos, sizeof(val)); pos += sizeof(val); } while(0)
      S(is);
      S(alloc_offset);
      S(size);
      S(redopid);
      S(count_offset);
      S(red_list_size);
      S(block_size);
      S(elmt_size);
      size_t l;
      S(l);
      field_sizes.resize(l);
      for(size_t i = 0; i < l; i++) S(field_sizes[i]);
      S(parent_inst);
      linearization.deserialize((const int *)pos);
#undef S
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

}; // namespace Realm
