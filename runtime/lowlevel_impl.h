/* Copyright 2013 Stanford University
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

#include "lowlevel.h"

#include <assert.h>

#ifndef NO_INCLUDE_GASNET
#include "activemsg.h"
#endif

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DECLARE(cur_thread);

#include <pthread.h>
#include <string.h>

#include <vector>
#include <deque>
#include <queue>
#include <set>
#include <list>
#include <map>

// Comment this out to disable shared task queue

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

// GASnet helper stuff

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

    class AutoHSLLock {
    public:
      AutoHSLLock(gasnet_hsl_t &mutex) : mutexp(&mutex), held(true)
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      AutoHSLLock(gasnet_hsl_t *_mutexp) : mutexp(_mutexp), held(true)
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      ~AutoHSLLock(void) 
      {
	if(held)
	  gasnet_hsl_unlock(mutexp);
	log_mutex(LEVEL_SPEW, "MUTEX LOCK OUT %p", mutexp);
	//printf("[%d] MUTEX LOCK OUT %p\n", gasnet_mynode(), mutexp);
      }
      void release(void)
      {
	assert(held);
	gasnet_hsl_unlock(mutexp);
	held = false;
      }
      void reacquire(void)
      {
	assert(!held);
	gasnet_hsl_lock(mutexp);
	held = true;
      }
    protected:
      gasnet_hsl_t *mutexp;
      bool held;
    };

    typedef LegionRuntime::HighLevel::BitMask<NODE_MASK_TYPE,MAX_NUM_NODES,
                                              NODE_MASK_SHIFT,NODE_MASK_MASK> NodeMask;

    // for each of the ID-based runtime objects, we're going to have an
    //  implementation class and a table to look them up in
    struct Node {
      Node(void);

      gasnet_hsl_t mutex;  // used to cover resizing activities on vectors below
      std::vector<Event::Impl> events;
      size_t num_events;
      std::vector<Reservation::Impl> locks;
      size_t num_locks;
      std::vector<Memory::Impl *> memories;
      std::vector<Processor::Impl *> processors;
      std::vector<IndexSpace::Impl *> index_spaces;
    };

    template <class T>
    class Atomic {
    public:
      Atomic(T _value) : value(_value)
      {
	gasnet_hsl_init(&mutex);
	//printf("%d: atomic %p = %d\n", gasnet_mynode(), this, value);
      }

      T get(void) const { return (*((volatile T*)(&value))); }

      void decrement(void)
      {
	AutoHSLLock a(mutex);
	//T old_value(value);
	value--;
	//printf("%d: atomic %p %d -> %d\n", gasnet_mynode(), this, old_value, value);
      }

    protected:
      T value;
      gasnet_hsl_t mutex;
    };

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
     
    class ID {
    public:
      // two forms of bit pack for IDs:
      //
      //  3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
      //  1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
      // +-----+---------------------------------------------------------+
      // | TYP |   NODE  |           INDEX                               |
      // | TYP |   NODE  |   INDEX_H     |           INDEX_L             |
      // +-----+---------------------------------------------------------+

      enum {
	TYPE_BITS = 3,
	INDEX_H_BITS = 8,
	INDEX_L_BITS = 16,
	INDEX_BITS = INDEX_H_BITS + INDEX_L_BITS,
	NODE_BITS = 32 - TYPE_BITS - INDEX_BITS
      };

      enum ID_Types {
	ID_SPECIAL,
	ID_EVENT,
	ID_LOCK,
	ID_MEMORY,
	ID_PROCESSOR,
	ID_INDEXSPACE,
	ID_ALLOCATOR,
	ID_INSTANCE,
      };

      enum ID_Specials {
	ID_INVALID = 0,
	ID_GLOBAL_MEM = (1U << INDEX_H_BITS) - 1,
      };

      ID(unsigned _value) : value(_value) {}

      template <class T>
      ID(T thing_to_get_id_from) : value(thing_to_get_id_from.id) {}

      ID(ID_Types _type, unsigned _node, unsigned _index)
	: value((((unsigned)_type) << (NODE_BITS + INDEX_BITS)) |
		(_node << INDEX_BITS) |
		_index) {}

      ID(ID_Types _type, unsigned _node, unsigned _index_h, unsigned _index_l)
	: value((((unsigned)_type) << (NODE_BITS + INDEX_BITS)) |
		(_node << INDEX_BITS) |
		(_index_h << INDEX_L_BITS) |
		_index_l) {}

      unsigned id(void) const { return value; }
      ID_Types type(void) const { return (ID_Types)(value >> (NODE_BITS + INDEX_BITS)); }
      unsigned node(void) const { return ((value >> INDEX_BITS) & ((1U << NODE_BITS)-1)); }
      unsigned index(void) const { return (value & ((1U << INDEX_BITS) - 1)); }
      unsigned index_h(void) const { return ((value >> INDEX_L_BITS) & ((1U << INDEX_H_BITS)-1)); }
      unsigned index_l(void) const { return (value & ((1U << INDEX_L_BITS) - 1)); }

      template <class T>
      T convert(void) const { T thing_to_return = { value }; return thing_to_return; }
      
    protected:
      unsigned value;
    };
    
    class Runtime {
    public:
      static Runtime *get_runtime(void) { return runtime; }

      Event::Impl *get_event_impl(ID id);
      Reservation::Impl *get_lock_impl(ID id);
      Memory::Impl *get_memory_impl(ID id);
      Processor::Impl *get_processor_impl(ID id);
      IndexSpace::Impl *get_index_space_impl(ID id);
      RegionInstance::Impl *get_instance_impl(ID id);

    protected:
    public:
      static Runtime *runtime;

      Node *nodes;
      Memory::Impl *global_memory;
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

    class Reservation::Impl {
    public:
      Impl(void);

      void init(Reservation _me, unsigned _init_owner, size_t _data_size = 0);

      template <class T>
      void set_local_data(T *data)
      {
	local_data = data;
	local_data_size = sizeof(T);
        own_local = false;
      }

      //protected:
      Reservation me;
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      bool in_use;

      enum { MODE_EXCL = 0, ZERO_COUNT = 0x11223344 };

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      uint64_t remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      std::map<unsigned, std::deque<Event> > local_waiters;
      bool requested; // do we have a request for the lock in flight?

      // local data protected by lock
      void *local_data;
      size_t local_data_size;
      bool own_local;

      static gasnet_hsl_t freelist_mutex;
      static Reservation::Impl *first_free;
      Reservation::Impl *next_free;

      Event acquire(unsigned new_mode, bool exclusive,
		 Event after_lock = Event::NO_EVENT);

      bool select_local_waiters(std::deque<Event>& to_wake);

      void release(void);

      bool is_locked(unsigned check_mode, bool excl_ok);

      void release_reservation(void);
    };

    template <class T>
    class StaticAccess {
    public:
      typedef typename T::StaticData StaticData;

      // if already_valid, just check that data is already valid
      StaticAccess(T* thing_with_data, bool already_valid = false)
	: data(&thing_with_data->locked_data)
      {
	if(already_valid) {
	  assert(data->valid);
	} else {
	  if(!data->valid) {
	    // get a valid copy of the static data by taking and then releasing
	    //  a shared lock
	    thing_with_data->lock.acquire(1, false).wait(true);// TODO: must this be blocking?
	    thing_with_data->lock.release();
	    assert(data->valid);
	  }
	}
      }

      ~StaticAccess(void) {}

      const StaticData *operator->(void) { return data; }

    protected:
      StaticData *data;
    };

    template <class T>
    class SharedAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      // if already_held, just check that it's held (if in debug mode)
      SharedAccess(T* thing_with_data, bool already_held = false)
	: data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
      {
	if(already_held) {
	  assert(lock->is_locked(1, true));
	} else {
	  lock->acquire(1, false).wait();
	}
      }

      ~SharedAccess(void)
      {
	lock->release();
      }

      const CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      Reservation::Impl *lock;
    };

    template <class T>
    class ExclusiveAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      // if already_held, just check that it's held (if in debug mode)
      ExclusiveAccess(T* thing_with_data, bool already_held = false)
	: data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
      {
	if(already_held) {
	  assert(lock->is_locked(0, true));
	} else {
	  lock->acquire(0, true).wait();
	}
      }

      ~ExclusiveAccess(void)
      {
	lock->release();
      }

      CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      Reservation::Impl *lock;
    };

    class ProcessorAssignment {
    public:
      ProcessorAssignment(int _num_local_procs);

      // binds a thread to the right set of cores based (-1 = not a local proc)
      void bind_thread(int core_id, pthread_attr_t *attr, const char *debug_name = 0);

    protected:
      // physical configuration of processors
      typedef std::map<int, std::vector<int> > NodeProcMap;
      typedef std::map<int, NodeProcMap> SystemProcMap;

      int num_local_procs;
      bool valid;
      std::vector<int> local_proc_assignments;
      cpu_set_t leftover_procs;
    };
    extern ProcessorAssignment *proc_assignment;

    extern Processor::TaskIDTable task_id_table;

    class UtilityProcessor;

    class Processor::Impl {
    public:
      Impl(Processor _me, Processor::Kind _kind, Processor _util = Processor::NO_PROC)
	: me(_me), kind(_kind), util(_util), util_proc(0), run_counter(0) {}

      void run(Atomic<int> *_run_counter)
      {
	run_counter = _run_counter;
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority) = 0;

      void finished(void)
      {
	if(run_counter)
	  run_counter->decrement();
      }

      void set_utility_processor(UtilityProcessor *_util_proc);

      virtual void enable_idle_task(void) { assert(0); }
      virtual void disable_idle_task(void) { assert(0); }
      virtual bool is_idle_task_enabled(void) { return(false); }

    public:
      Processor me;
      Processor::Kind kind;
      Processor util;
      UtilityProcessor *util_proc;
      Atomic<int> *run_counter;
    };
    
    class PreemptableThread {
    public:
      PreemptableThread(void) {}
      virtual ~PreemptableThread(void) {}

      void start_thread(size_t stack_size, int core_id, const char *debug_name);

      static bool preemptable_sleep(Event wait_for, bool block = false);

      virtual Processor get_processor(void) const = 0;

    protected:
      static void *thread_entry(void *data);

      virtual void thread_main(void) = 0;

      virtual void sleep_on_event(Event wait_for, bool block = false) = 0;

      pthread_t thread;
    };

#define SHARED_UTILITY_QUEUE
#ifdef SHARED_UTILITY_QUEUE
    class UtilityQueue;
#endif

    class UtilityProcessor : public Processor::Impl {
    public:
      UtilityProcessor(Processor _me, 
#ifdef SHARED_UTILITY_QUEUE
                       UtilityQueue *_shared_queue,
#endif
                       int core_id = -1, 
                       int _num_worker_threads = 1);

      virtual ~UtilityProcessor(void);

      void start_worker_threads(size_t stack_size);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);

      void request_shutdown(void);

      void enable_idle_task(Processor::Impl *proc);
      void disable_idle_task(Processor::Impl *proc);

      void wait_for_shutdown(void);

#ifdef SHARED_UTILITY_QUEUE
      void shared_tasks_available(void);
#endif

      class UtilityThread;
      class UtilityTask;

    protected:
      //friend class UtilityThread;
      //friend class UtilityTask;

      void enqueue_runnable_task(UtilityTask *task);
      int core_id;
      int num_worker_threads;
      bool shutdown_requested;
      //Event shutdown_event;

      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;

      UtilityTask *idle_task;

#ifdef SHARED_UTILITY_QUEUE
      UtilityQueue *shared_queue;
#endif

      std::set<UtilityThread *> threads;
      std::list<UtilityTask *> tasks;
      std::set<Processor::Impl *> idle_procs;
      std::set<Processor::Impl *> procs_in_idle_task;
    };

    class Memory::Impl {
    public:
      enum MemoryKind {
	MKIND_SYSMEM,  // directly accessible from CPU
	MKIND_GLOBAL,  // accessible via GASnet (spread over all nodes)
	MKIND_RDMA,    // remote, but accessible via RDMA
	MKIND_REMOTE,  // not accessible
	MKIND_GPUFB,   // GPU framebuffer memory (accessible via cudaMemcpy)
	MKIND_ZEROCOPY, // CPU memory, pinned for GPU access
      };

    Impl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment, Kind _lowlevel_kind)
      : me(_me), size(_size), kind(_kind), alignment(_alignment), lowlevel_kind(_lowlevel_kind)
      {
	gasnet_hsl_init(&mutex);
      }

      unsigned add_instance(RegionInstance::Impl *i);

      RegionInstance::Impl *get_instance(RegionInstance i);

      RegionInstance create_instance_local(IndexSpace is,
					   const int *linearization_bits,
					   size_t bytes_needed,
					   size_t block_size,
					   size_t element_size,
					   const std::vector<size_t>& field_sizes,
					   ReductionOpID redopid,
					   off_t list_size,
					   RegionInstance parent_inst);

      RegionInstance create_instance_remote(IndexSpace is,
					    const int *linearization_bits,
					    size_t bytes_needed,
					    size_t block_size,
					    size_t element_size,
					    const std::vector<size_t>& field_sizes,
					    ReductionOpID redopid,
					    off_t list_size,
					    RegionInstance parent_inst);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
					     RegionInstance parent_inst) = 0;

      void destroy_instance_local(RegionInstance i, bool local_destroy);
      void destroy_instance_remote(RegionInstance i, bool local_destroy);

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy) = 0;

      off_t alloc_bytes_local(size_t size);
      void free_bytes_local(off_t offset, size_t size);

      off_t alloc_bytes_remote(size_t size);
      void free_bytes_remote(off_t offset, size_t size);

      virtual off_t alloc_bytes(size_t size) = 0;
      virtual void free_bytes(off_t offset, size_t size) = 0;

      virtual void get_bytes(off_t offset, void *dst, size_t size) = 0;
      virtual void put_bytes(off_t offset, const void *src, size_t size) = 0;

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					size_t count, const void *entry_buffer)
      {
	assert(0);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size) = 0;
      virtual int get_home_node(off_t offset, size_t size) = 0;

      Memory::Kind get_kind(void) const;

    public:
      Memory me;
      size_t size;
      MemoryKind kind;
      size_t alignment;
      Kind lowlevel_kind;
      gasnet_hsl_t mutex; // protection for resizing vectors
      std::vector<RegionInstance::Impl *> instances;
      std::map<off_t, off_t> free_blocks;
    };

    class GASNetMemory : public Memory::Impl {
    public:
      static const size_t MEMORY_STRIDE = 1024;

      GASNetMemory(Memory _me, size_t size_per_node);

      virtual ~GASNetMemory(void);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
					     RegionInstance parent_inst);

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					size_t count, const void *entry_buffer);

      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

      void get_batch(size_t batch_size,
		     const off_t *offsets, void * const *dsts, 
		     const size_t *sizes);

      void put_batch(size_t batch_size,
		     const off_t *offsets, const void * const *srcs, 
		     const size_t *sizes);

    protected:
      int num_nodes;
      off_t memory_stride;
      gasnet_seginfo_t *seginfos;
      //std::map<off_t, off_t> free_blocks;
    };

    class RegionInstance::Impl {
    public:
      Impl(RegionInstance _me, IndexSpace _is, Memory _memory, off_t _offset, size_t _size, ReductionOpID _redopid,
	   const DomainLinearization& _linear, size_t _block_size, size_t _elmt_size, const std::vector<size_t>& _field_sizes,
	   off_t _count_offset = -1, off_t _red_list_size = -1, RegionInstance _parent_inst = NO_INST);

      // when we auto-create a remote instance, we don't know region/offset/linearization
      Impl(RegionInstance _me, Memory _memory);

      ~Impl(void);

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

    public: //protected:
      friend class RegionInstance;

      RegionInstance me;
      Memory memory;
      DomainLinearization linearization;

      static const unsigned MAX_FIELDS_PER_INST = 2048;
      static const unsigned MAX_LINEARIZATION_LEN = 16;

      struct StaticData {
	IndexSpace is;
	off_t alloc_offset; //, access_offset;
	size_t size;
	//size_t first_elmt, last_elmt;
	//bool is_reduction;
	ReductionOpID redopid;
	off_t count_offset;
	off_t red_list_size;
	size_t block_size, elmt_size;
	int field_sizes[MAX_FIELDS_PER_INST];
	RegionInstance parent_inst;
	int linearization_bits[MAX_LINEARIZATION_LEN];
        // This had better damn well be the last field
        // in the struct in order to avoid race conditions!
	bool valid;
      } locked_data;

      Reservation::Impl lock;
    };

    class Event::Impl {
    public:
      Impl(void);

      void init(Event _me, unsigned _init_owner);

      static Event create_event(void);

      // test whether an event has triggered without waiting
      bool has_triggered(Event::gen_t needed_gen);

      // causes calling thread to block until event has occurred
      //void wait(Event::gen_t needed_gen);

      void external_wait(Event::gen_t needed_gen);

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);

      // record that the event has triggered and notify anybody who cares
      void trigger(Event::gen_t gen_triggered, int trigger_node, Event wait_on = NO_EVENT);

      // used to adjust a barrier's arrival count either up or down
      // if delta > 0, timestamp is current time (on requesting node)
      // if delta < 0, timestamp says which positive adjustment this arrival must wait for
      void adjust_arrival(Event::gen_t barrier_gen, int delta, 
			  Barrier::timestamp_t timestamp, Event wait_on = NO_EVENT);

      class EventWaiter {
      public:
	virtual bool event_triggered(void) = 0;
	virtual void print_info(void) = 0;
      };

      void add_waiter(Event event, EventWaiter *waiter, bool pre_subscribed = false);

    public: //protected:
      Event me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed, free_generation;
      Event::Impl *next_free;
      static Event::Impl *first_free;
      static gasnet_hsl_t freelist_mutex;

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible event)

      //uint64_t remote_waiters; // bitmask of which remote nodes are waiting on the event
      NodeMask remote_waiters;
      std::map<Event::gen_t, std::vector<EventWaiter *> > local_waiters; // set of local threads that are waiting on event (keyed by generation)

      // for barriers
      unsigned base_arrival_count, current_arrival_count;

      class PendingUpdates;
      std::map<Event::gen_t, PendingUpdates *> pending_updates;
    };

    class IndexSpace::Impl {
    public:
      Impl(IndexSpace _me, IndexSpace _parent,
	   size_t _num_elmts,
	   const ElementMask *_initial_valid_mask = 0, bool _frozen = false);

      // this version is called when we create a proxy for a remote region
      Impl(IndexSpace _me);

      ~Impl(void);

      bool is_parent_of(IndexSpace other);

      size_t instance_size(const ReductionOpUntyped *redop = 0,
			   off_t list_size = -1);

      off_t instance_adjust(const ReductionOpUntyped *redop = 0);

      Event request_valid_mask(void);

      IndexSpace me;
      Reservation::Impl lock;

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
      gasnet_hsl_t valid_mask_mutex;
      ElementMask *valid_mask;
      int valid_mask_count;
      bool valid_mask_complete;
      Event valid_mask_event;
      int valid_mask_first, valid_mask_last;
      bool valid_mask_contig;
      ElementMask *avail_mask;
    };

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
