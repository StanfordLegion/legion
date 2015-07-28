/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#ifndef LOWLEVEL_IMPL_H
#define LOWLEVEL_IMPL_H

// For doing bit masks for maximum number of nodes
#include "legion_types.h"
#include "legion_utilities.h"

#define NODE_MASK_TYPE uint64_t
#define NODE_MASK_SHIFT 6
#define NODE_MASK_MASK 0x3F

#ifndef MAX_NUM_THREADS
#define MAX_NUM_THREADS 32
#endif

#include "lowlevel.h"

#define NO_USE_REALMS_NODESET
#ifdef USE_REALMS_NODESET
#include "realm/dynamic_set.h"

namespace LegionRuntime {
  namespace LowLevel {
#if MAX_NUM_NODES <= 65536
    typedef DynamicSet<unsigned short> NodeSet;
#else
    // possibly unnecessary future-proofing...
    typedef DynamicSet<unsigned int> NodeSet;
#endif
  };
};
#else
namespace LegionRuntime {
  namespace LowLevel {
    typedef LegionRuntime::HighLevel::NodeSet NodeSet;
    //typedef LegionRuntime::HighLevel::BitMask<NODE_MASK_TYPE,MAX_NUM_NODES,
    //                                          NODE_MASK_SHIFT,NODE_MASK_MASK> NodeMask;
  };
};
namespace Realm {
  typedef LegionRuntime::HighLevel::NodeSet NodeSet;
};
#endif

#include "realm/operation.h"
#include "realm/dynamic_table.h"
#include "realm/id.h"

#include <assert.h>

#include "activemsg.h"
#include <pthread.h>

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DECLARE(cur_thread);

extern pthread_key_t thread_timer_key;

#include <string.h>

#include <vector>
#include <deque>
#include <queue>
#include <set>
#include <list>
#include <map>
#include <aio.h>
#include <greenlet>

#if __cplusplus >= 201103L
#define typeof decltype
#endif

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)


namespace Realm {
  class Module;
  class Operation;
  class ProfilingRequestSet;
};

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"
#include "realm/machine_impl.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"

// namespace importing for backwards compatibility
namespace LegionRuntime {
  namespace LowLevel {
    typedef Realm::ID ID;
    typedef Realm::EventWaiter EventWaiter;
    typedef Realm::EventImpl EventImpl;
    typedef Realm::GenEventImpl GenEventImpl;
    typedef Realm::BarrierImpl BarrierImpl;
    typedef Realm::ReservationImpl ReservationImpl;
    typedef Realm::MachineImpl MachineImpl;
    typedef Realm::ProcessorImpl ProcessorImpl;
    typedef Realm::ProcessorGroup ProcessorGroup;
    typedef Realm::Task Task;
    typedef Realm::MemoryImpl MemoryImpl;
  };
};

namespace LegionRuntime {
  namespace LowLevel {

    extern Logger::Category log_mutex;

#ifdef EVENT_TRACING
    // For event tracing
    struct EventTraceItem {
    public:
      enum Action {
        ACT_CREATE = 0,
        ACT_QUERY = 1,
        ACT_TRIGGER = 2,
        ACT_WAIT = 3,
      };
    public:
      unsigned time_units, event_id, event_gen, action;
    };
#endif

#ifdef LOCK_TRACING
    // For lock tracing
    struct LockTraceItem {
    public:
      enum Action {
        ACT_LOCAL_REQUEST = 0, // request for a lock where the owner is local
        ACT_REMOTE_REQUEST = 1, // request for a lock where the owner is not local
        ACT_FORWARD_REQUEST = 2, // for forwarding of requests
        ACT_LOCAL_GRANT = 3, // local grant of the lock
        ACT_REMOTE_GRANT = 4, // remote grant of the lock (change owners)
        ACT_REMOTE_RELEASE = 5, // remote release of a shared lock
      };
    public:
      unsigned time_units, lock_id, owner, action;
    };
#endif

    // prioritized list that maintains FIFO order within a priority level
    template <typename T>
    class pri_list : public std::list<T> {
    public:
      void pri_insert(T to_add) {
        // Common case: if the guy on the back has our priority or higher then just
        // put us on the back too.
        if (this->empty() || (this->back()->priority >= to_add->priority))
          this->push_back(to_add);
        else
        {
          // Uncommon case: go through the list until we find someone
          // who has a priority lower than ours.  We know they
          // exist since we saw them on the back.
          bool inserted = false;
          for (typename std::list<T>::iterator it = this->begin();
                it != this->end(); it++)
          {
            if ((*it)->priority < to_add->priority)
            {
              this->insert(it, to_add);
              inserted = true;
              break;
            }
          }
          // Technically we shouldn't need this, but just to be safe
          assert(inserted);
        }
      }
    };
     
    struct ElementMaskImpl {
      //int count, offset;
      typedef unsigned long long uint64;
      uint64_t dummy;
      uint64_t bits[0];

      static size_t bytes_needed(off_t offset, off_t count)
      {
	size_t need = sizeof(ElementMaskImpl) + (((count + 63) >> 6) << 3);
	return need;
      }
	
    };

#ifdef USE_GASNET 
    class HandlerThread : public PreemptableThread {
    public:
      HandlerThread(IncomingMessageManager *m) : manager(m) { }
      virtual ~HandlerThread(void) { }
    public:
      virtual Processor get_processor(void) const 
        { assert(false); return Processor::NO_PROC; }
    public:
      virtual void thread_main(void);
      virtual void sleep_on_event(Event wait_for);
    public:
      void join(void);
    private:
      IncomingMessage *current_msg, *next_msg;
      IncomingMessageManager *const manager;
    };
#endif

    class RegionInstanceImpl;

    class MetadataBase {
    public:
      MetadataBase(void);
      ~MetadataBase(void);

      enum State { STATE_INVALID,
		   STATE_VALID,
		   STATE_REQUESTED,
		   STATE_INVALIDATE,  // if invalidate passes normal request response
		   STATE_CLEANUP };

      bool is_valid(void) const { return state == STATE_VALID; }

      void mark_valid(void); // used by owner
      void handle_request(int requestor);

      // returns an Event for when data will be valid
      Event request_data(int owner, IDType id);
      void await_data(bool block = true);  // request must have already been made
      void handle_response(void);
      void handle_invalidate(void);

      // these return true once all remote copies have been invalidated
      bool initiate_cleanup(IDType id);
      bool handle_inval_ack(int sender);

    protected:
      GASNetHSL mutex;
      State state;  // current state
      GenEventImpl *valid_event_impl; // event to track receipt of in-flight request (if any)
      NodeSet remote_copies;
    };

    class RegionInstanceImpl {
    public:
      RegionInstanceImpl(RegionInstance _me, IndexSpace _is, Memory _memory, off_t _offset, size_t _size, 
			 ReductionOpID _redopid,
			 const DomainLinearization& _linear, size_t _block_size, size_t _elmt_size, 
			 const std::vector<size_t>& _field_sizes,
			 const Realm::ProfilingRequestSet &reqs,
			 off_t _count_offset = -1, off_t _red_list_size = -1, 
			 RegionInstance _parent_inst = RegionInstance::NO_INST);

      // when we auto-create a remote instance, we don't know region/offset/linearization
      RegionInstanceImpl(RegionInstance _me, Memory _memory);

      ~RegionInstanceImpl(void);

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

      Event request_metadata(void) { return metadata.request_data(ID(me).node(), me.id); }

      void finalize_instance(void);

    public: //protected:
      friend class Realm::RegionInstance;

      RegionInstance me;
      Memory memory; // not part of metadata because it's determined from ID alone
      // Profiling info only needed on creation node
      Realm::ProfilingRequestSet requests;
      Realm::ProfilingMeasurementCollection measurements;
      Realm::ProfilingMeasurements::InstanceTimeline timeline;

      class Metadata : public MetadataBase {
      public:
	void *serialize(size_t& out_size) const;
	void deserialize(const void *in_data, size_t in_size);

	IndexSpace is;
	off_t alloc_offset;
	size_t size;
	ReductionOpID redopid;
	off_t count_offset;
	off_t red_list_size;
	size_t block_size, elmt_size;
	std::vector<size_t> field_sizes;
	RegionInstance parent_inst;
	DomainLinearization linearization;
      };

      Metadata metadata;

      static const unsigned MAX_LINEARIZATION_LEN = 16;

      ReservationImpl lock;
    };

    class IndexSpaceImpl {
    public:
      IndexSpaceImpl(void);
      ~IndexSpaceImpl(void);

      void init(IndexSpace _me, unsigned _init_owner);

      void init(IndexSpace _me, IndexSpace _parent,
		size_t _num_elmts,
		const ElementMask *_initial_valid_mask = 0, bool _frozen = false);

      static const ID::ID_Types ID_TYPE = ID::ID_INDEXSPACE;

      bool is_parent_of(IndexSpace other);

      size_t instance_size(const ReductionOpUntyped *redop = 0,
			   off_t list_size = -1);

      off_t instance_adjust(const ReductionOpUntyped *redop = 0);

      Event request_valid_mask(void);

      IndexSpace me;
      ReservationImpl lock;
      IndexSpaceImpl *next_free;

      struct StaticData {
	IndexSpace parent;
	bool frozen;
	size_t num_elmts;
        size_t first_elmt, last_elmt;
        // This had better damn well be the last field
        // in the struct in order to avoid race conditions!
	bool valid;
      };
      struct CoherentData : public StaticData {
	unsigned valid_mask_owners;
	int avail_mask_owner;
      };

      CoherentData locked_data;
      GASNetHSL valid_mask_mutex;
      ElementMask *valid_mask;
      int valid_mask_count;
      bool valid_mask_complete;
      Event valid_mask_event;
      GenEventImpl *valid_mask_event_impl;
      int valid_mask_first, valid_mask_last;
      bool valid_mask_contig;
      ElementMask *avail_mask;
    };

    template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
    class DynamicTableAllocator {
    public:
      typedef _ET ET;
      static const size_t INNER_BITS = _INNER_BITS;
      static const size_t LEAF_BITS = _LEAF_BITS;

      typedef GASNetHSL LT;
      typedef int IT;
      typedef Realm::DynamicTableNode<Realm::DynamicTableNodeBase<LT, IT> *, 1 << INNER_BITS, LT, IT> INNER_TYPE;
      typedef Realm::DynamicTableNode<ET, 1 << LEAF_BITS, LT, IT> LEAF_TYPE;
      typedef Realm::DynamicTableFreeList<DynamicTableAllocator<ET, _INNER_BITS, _LEAF_BITS> > FreeList;
      
      static LEAF_TYPE *new_leaf_node(IT first_index, IT last_index, 
				      int owner, FreeList *free_list)
      {
	LEAF_TYPE *leaf = new LEAF_TYPE(0, first_index, last_index);
	IT last_ofs = (((IT)1) << LEAF_BITS) - 1;
	for(IT i = 0; i <= last_ofs; i++)
	  leaf->elems[i].init(ID(ET::ID_TYPE, owner, first_index + i).convert<typeof(leaf->elems[0].me)>(), owner);

	if(free_list) {
	  // stitch all the new elements into the free list
	  free_list->lock.lock();

	  for(IT i = 0; i <= last_ofs; i++)
	    leaf->elems[i].next_free = ((i < last_ofs) ? 
					  &(leaf->elems[i+1]) :
					  free_list->first_free);

	  free_list->first_free = &(leaf->elems[first_index ? 0 : 1]);

	  free_list->lock.unlock();
	}

	return leaf;
      }
    };

    typedef DynamicTableAllocator<GenEventImpl, 10, 8> EventTableAllocator;
    typedef DynamicTableAllocator<BarrierImpl, 10, 4> BarrierTableAllocator;
    typedef DynamicTableAllocator<ReservationImpl, 10, 8> ReservationTableAllocator;
    typedef DynamicTableAllocator<IndexSpaceImpl, 10, 4> IndexSpaceTableAllocator;
    typedef DynamicTableAllocator<ProcessorGroup, 10, 4> ProcessorGroupTableAllocator;

    // for each of the ID-based runtime objects, we're going to have an
    //  implementation class and a table to look them up in
    struct Node {
      Node(void);

      // not currently resizable
      std::vector<MemoryImpl *> memories;
      std::vector<ProcessorImpl *> processors;

      Realm::DynamicTable<EventTableAllocator> events;
      Realm::DynamicTable<BarrierTableAllocator> barriers;
      Realm::DynamicTable<ReservationTableAllocator> reservations;
      Realm::DynamicTable<IndexSpaceTableAllocator> index_spaces;
      Realm::DynamicTable<ProcessorGroupTableAllocator> proc_groups;
    };

    struct NodeAnnounceData;

    class RuntimeImpl {
    public:
      RuntimeImpl(void);
      ~RuntimeImpl(void);

      bool init(int *argc, char ***argv);

      bool register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr);
      bool register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop);

      void run(Processor::TaskFuncID task_id = 0, 
	       Runtime::RunStyle style = Runtime::ONE_TASK_ONLY,
	       const void *args = 0, size_t arglen = 0, bool background = false);

      // requests a shutdown of the runtime
      void shutdown(bool local_request);

      void wait_for_shutdown(void);

      // three event-related impl calls - get_event_impl() will give you either
      //  a normal event or a barrier, but you won't be able to do specific things
      //  (e.g. trigger a GenEventImpl or adjust a BarrierImpl)
      EventImpl *get_event_impl(Event e);
      GenEventImpl *get_genevent_impl(Event e);
      BarrierImpl *get_barrier_impl(Event e);

      ReservationImpl *get_lock_impl(ID id);
      MemoryImpl *get_memory_impl(ID id);
      ProcessorImpl *get_processor_impl(ID id);
      ProcessorGroup *get_procgroup_impl(ID id);
      IndexSpaceImpl *get_index_space_impl(ID id);
      RegionInstanceImpl *get_instance_impl(ID id);
#ifdef DEADLOCK_TRACE
      void add_thread(const pthread_t *thread);
#endif

    protected:
    public:
      MachineImpl *machine;

      Processor::TaskIDTable task_table;
      std::map<ReductionOpID, const ReductionOpUntyped *> reduce_op_table;

#ifdef NODE_LOGGING
      static const char *prefix;
#endif

      std::vector<Realm::Module *> modules;
      Node *nodes;
      MemoryImpl *global_memory;
      EventTableAllocator::FreeList *local_event_free_list;
      BarrierTableAllocator::FreeList *local_barrier_free_list;
      ReservationTableAllocator::FreeList *local_reservation_free_list;
      IndexSpaceTableAllocator::FreeList *local_index_space_free_list;
      ProcessorGroupTableAllocator::FreeList *local_proc_group_free_list;

      pthread_t *background_pthread;
#ifdef DEADLOCK_TRACE
      unsigned next_thread;
      unsigned signaled_threads;
      pthread_t all_threads[MAX_NUM_THREADS];
      unsigned thread_counts[MAX_NUM_THREADS];
#endif
    };

    extern RuntimeImpl *runtime_singleton;
    inline RuntimeImpl *get_runtime(void) { return runtime_singleton; }

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
