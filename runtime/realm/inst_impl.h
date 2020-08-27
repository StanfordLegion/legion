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

    class CompiledInstanceLayout : public PieceLookup::CompiledProgram {
    public:
      CompiledInstanceLayout();
      ~CompiledInstanceLayout();

      virtual void *allocate_memory(size_t bytes);

      void reset();

      void *program_base;
      size_t program_size;
    };

    class RegionInstanceImpl {
    protected:
      // RegionInstanceImpl creation/deletion is handled by MemoryImpl
      friend class MemoryImpl;
      friend class LocalManagedMemory;
      RegionInstanceImpl(RegionInstance _me, Memory _memory);
      ~RegionInstanceImpl(void);

      class DeferredCreate : public EventWaiter {
      public:
	void defer(RegionInstanceImpl *_inst, MemoryImpl *_mem,
		   bool _need_alloc_result, Event wait_on);
	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	RegionInstanceImpl *inst;
	MemoryImpl *mem;
	bool need_alloc_result;
      };
      DeferredCreate deferred_create;

      class DeferredDestroy : public EventWaiter {
      public:
	void defer(RegionInstanceImpl *_inst, MemoryImpl *_mem, Event wait_on);
	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	RegionInstanceImpl *inst;
	MemoryImpl *mem;
      };
      DeferredDestroy deferred_destroy;

    public:
      // entry point for both create_instance and create_external_instance
      static Event create_instance(RegionInstance& inst,
				   Memory memory,
				   InstanceLayoutGeneric *ilg,
				   const ExternalInstanceResource *res,
				   const ProfilingRequestSet& prs,
				   Event wait_on);
      
      // the life cycle of an instance is defined in part by when the
      //  allocation and deallocation of storage occurs, but that is managed
      //  by the memory, which uses these callbacks to notify us
      void notify_allocation(MemoryImpl::AllocationResult result, size_t offset,
			     TimeLimit work_until);
      void notify_deallocation(void);

      bool get_strided_parameters(void *&base, size_t &stride,
				  off_t field_offset);

      Event request_metadata(void) { return metadata.request_data(int(ID(me).instance_creator_node()), me.id); }

      // ensures metadata is available on the specified node
      Event prefetch_metadata(NodeID target);

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

      protected:
	virtual void do_invalidate(void);

      public:
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
	bool need_alloc_result, need_notify_dealloc;
	InstanceLayoutGeneric *layout;
	ExternalInstanceResource *ext_resource;
	void *mem_specific;  // a place for memory's to hang info
	std::string filename; // temp hack for attached files

	CompiledInstanceLayout lookup_program;
      };

      // used for atomic access to metadata
      Mutex mutex;
      Metadata metadata;
      // cache of prefetch events for other nodes
      std::map<NodeID, Event> prefetch_events;

      // used for serialized application access to contents of instance
      ReservationImpl lock;
    };

    // active messages

    struct InstanceMetadataPrefetchRequest {
      RegionInstance inst;
      Event valid_event;

      static void handle_message(NodeID sender,
				 const InstanceMetadataPrefetchRequest& msg,
				 const void *data, size_t datalen,
				 TimeLimit work_until);
    };
      
}; // namespace Realm

#endif // ifndef REALM_INST_IMPL_H
