/* Copyright 2019 Stanford University, NVIDIA Corporation
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

// RegionInstance implementations for Realm

#ifndef REALM_INST_IMPL_H
#define REALM_INST_IMPL_H

#include "realm/instance.h"
#include "realm/id.h"
#include "realm/inst_layout.h"

#include "realm/mutex.h"

#include "realm/rsrv_impl.h"
#include "realm/metadata.h"
#include "realm/event_impl.h"
#include "realm/profiling.h"
#include "realm/mem_impl.h"

namespace Realm {

    class RegionInstanceImpl {
    protected:
      // RegionInstanceImpl creation/deletion is handled by MemoryImpl
      friend class MemoryImpl;
      RegionInstanceImpl(RegionInstance _me, Memory _memory);
      ~RegionInstanceImpl(void);

      class DeferredCreate : public EventWaiter {
      public:
	void defer(RegionInstanceImpl *_inst, MemoryImpl *_mem,
		   size_t _bytes, size_t _align, Event wait_on);
	virtual void event_triggered(bool poisoned);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	RegionInstanceImpl *inst;
	MemoryImpl *mem;
	size_t bytes, align;
      };
      DeferredCreate deferred_create;

      class DeferredDestroy : public EventWaiter {
      public:
	void defer(RegionInstanceImpl *_inst, MemoryImpl *_mem, Event wait_on);
	virtual void event_triggered(bool poisoned);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	RegionInstanceImpl *inst;
	MemoryImpl *mem;
      };
      DeferredDestroy deferred_destroy;

    public:
      // the life cycle of an instance is defined in part by when the
      //  allocation and deallocation of storage occurs, but that is managed
      //  by the memory, which uses these callbacks to notify us
      void notify_allocation(MemoryImpl::AllocationResult result, size_t offset);
      void notify_deallocation(void);

#ifdef POINTER_CHECKS
      void verify_access(unsigned ptr);
      const ElementMask& get_element_mask(void);
#endif
      void get_bytes(int index, off_t byte_offset, void *dst, size_t size);
      void put_bytes(int index, off_t byte_offset, const void *src, size_t size);

#if 0
      static Event copy(RegionInstance src, 
			RegionInstance target,
			IndexSpace isegion,
			size_t elmt_size,
			size_t bytes_to_copy,
			Event after_copy = Event::NO_EVENT);
#endif

      bool get_strided_parameters(void *&base, size_t &stride,
				  off_t field_offset);

      Event request_metadata(void) { return metadata.request_data(ID(me).instance_creator_node(), me.id); }

      // called once storage has been released and all remote metadata is invalidated
      void recycle_instance(void);

    public: //protected:
      friend class RegionInstance;

      RegionInstance me;
      Memory memory; // not part of metadata because it's determined from ID alone
      // Profiling info only needed on creation node
      ProfilingRequestSet requests;
      ProfilingMeasurementCollection measurements;
      ProfilingMeasurements::InstanceTimeline timeline;

      // several special values exist for the 'inst_offset' field below
      static const size_t INSTOFFSET_UNALLOCATED = size_t(-1);
      static const size_t INSTOFFSET_FAILED = size_t(-2);
      static const size_t INSTOFFSET_DELAYEDALLOC = size_t(-3);
      static const size_t INSTOFFSET_DELAYEDDESTROY = size_t(-4);
      static const size_t INSTOFFSET_MAXVALID = size_t(-5);
      class Metadata : public MetadataBase {
      public:
	void *serialize(size_t& out_size) const;

	template<typename T>
	  void serialize_msg(T &fbd) const;

	void deserialize(const void *in_data, size_t in_size);

	off_t alloc_offset;
	size_t size;
	ReductionOpID redopid;
	off_t count_offset;
	off_t red_list_size;
	size_t block_size, elmt_size;
	std::vector<size_t> field_sizes;
	RegionInstance parent_inst;

	size_t inst_offset;
	Event ready_event;
	InstanceLayoutGeneric *layout;
	std::string filename; // temp hack for attached files
      };

      // used for atomic access to metadata
      Mutex mutex;
      Metadata metadata;

      // used for serialized application access to contents of instance
      ReservationImpl lock;
    };

    // helper function to figure out which field we're in
    void find_field_start(const std::vector<size_t>& field_sizes, off_t byte_offset,
			  size_t size, off_t& field_start, int& field_size);
    
}; // namespace Realm

#endif // ifndef REALM_INST_IMPL_H
