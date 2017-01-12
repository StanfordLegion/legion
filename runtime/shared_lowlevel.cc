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


#include "lowlevel.h"
#include "accessor.h"
#include "realm/profiling.h"
#include "realm/timers.h"
#include "realm/custom_serdez.h"

#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

using namespace LegionRuntime::Accessor;

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>

#include <map>
#include <set>
#include <list>
#include <deque>
#include <vector>

#include <pthread.h>
#include <errno.h>

#include <signal.h>
#include <unistd.h>
#ifdef LEGION_BACKTRACE
#include <execinfo.h>
#include <cxxabi.h>
#endif

#define BASE_EVENTS	  1024	
#define BASE_RESERVATIONS 64	
#define BASE_METAS	  64
#define BASE_ALLOCATORS	  64
#define BASE_INSTANCES	  64

// The number of threads for this version
#define NUM_PROCS	4
#define NUM_UTIL_PROCS  1
#define NUM_DMA_THREADS 1
// Maximum memory in global
#define GLOBAL_MEM      4096   // (MB)	
#define LOCAL_MEM       16384  // (KB)
// Default Pthreads stack size
#define STACK_SIZE      2      // (MB) 

#ifdef DEBUG_REALM
#define PTHREAD_SAFE_CALL(cmd)			\
	{					\
		int ret = (cmd);		\
		if (ret != 0) {			\
			fprintf(stderr,"PTHREAD error: %s = %d (%s)\n", #cmd, ret, strerror(ret));	\
			assert(false);		\
		}				\
	}
#else
#define PTHREAD_SAFE_CALL(cmd)			\
	(cmd);
#endif

#ifdef DEBUG_PRINT
#define DPRINT1(str,arg)						\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg);				\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

#define DPRINT2(str,arg1,arg2)						\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg1,arg2);				\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

#define DPRINT3(str,arg1,arg2,arg3)					\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg1,arg2,arg3);			\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

#define DPRINT4(str,arg1,arg2,arg3,arg4)				\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg1,arg2,arg3,arg4);		\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

// Declration for the debug mutex
pthread_mutex_t debug_mutex;
#endif // DEBUG_PRINT

// Local processor id
// Instead of using __thread, we'll try and make this
// code more portable by using pthread thread local storage
//__thread unsigned local_proc_id;
pthread_key_t local_thread_key;

namespace LegionRuntime {
  namespace LowLevel {

    // bring Realm's DetailedTimer into this namespace
    typedef Realm::DetailedTimer DetailedTimer;
    
// MAC OSX doesn't support pthread barrier type
#ifdef __MACH__
    typedef UtilityBarrier pthread_barrier_t;
#endif
    
    // Implementation for each of the runtime objects
    class EventImpl;
    class ReservationImpl;
    class MemoryImpl;
    class RegionInstanceImpl;
    class IndexSpaceImpl;
    class IndexSpaceAllocatorImpl;
    class ProcessorImpl;
    class ProcessorGroup;
    class DMAQueue;
    class CopyOperation;

    class RuntimeImpl;

    class MachineImpl {
    public:
      void get_all_memories(std::set<Memory>& mset) const;
      void get_all_processors(std::set<Processor>& pset) const;

      // Return the set of memories visible from a processor
      void get_visible_memories(Processor p, std::set<Memory>& mset) const;

      // Return the set of memories visible from a memory
      void get_visible_memories(Memory m, std::set<Memory>& mset) const;

      // Return the set of processors which can all see a given memory
      void get_shared_processors(Memory m, std::set<Processor>& pset) const;

      int get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				Processor restrict_proc /*= Processor::NO_PROC*/,
				Memory restrict_memory /*= Memory::NO_MEMORY*/) const;

      int get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
			       Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
			       Memory restrict_mem2 /*= Memory::NO_MEMORY*/) const;
      
    protected:
      friend class RuntimeImpl;
      std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
      std::vector<Machine::MemoryMemoryAffinity> mem_mem_affinities;
    };

    class RuntimeImpl {
    public:
      RuntimeImpl(MachineImpl *m);
    public:

      bool init(int *argc, char ***argv);

      void run(Processor::TaskFuncID task_id = 0, 
	       Runtime::RunStyle style = Runtime::ONE_TASK_ONLY,
	       const void *args = 0, size_t arglen = 0, bool background = false);

      void shutdown(void);
      void wait_for_shutdown(void);

      static RuntimeImpl* get_runtime(void) { return runtime; } 
      static DMAQueue *get_dma_queue(void) { return dma_queue; }

      EventImpl*           get_event_impl(Event e);
      ReservationImpl*     get_reservation_impl(Reservation r);
      MemoryImpl*          get_memory_impl(Memory m);
      ProcessorImpl*       get_processor_impl(Processor p);
      IndexSpaceImpl*  get_metadata_impl(IndexSpace is);
      RegionInstanceImpl*  get_instance_impl(RegionInstance i);

      EventImpl*           get_free_event(void);
      ReservationImpl*     get_free_reservation(size_t data_size = 0);
      IndexSpaceImpl*  get_free_metadata(size_t num_elmts);
      IndexSpaceImpl*  get_free_metadata(const ElementMask &mask);
      IndexSpaceImpl*  get_free_metadata(IndexSpaceImpl *par);
      IndexSpaceImpl*  get_free_metadata(IndexSpaceImpl *par, const ElementMask &mask);
      RegionInstanceImpl*  get_free_instance(Memory m, 
                                               size_t num_elmts, size_t alloc_size, 
					       const std::vector<size_t>& field_sizes,
					       size_t elmt_size, size_t block_size,
					       const DomainLinearization& linearization,
					       char *ptr, const ReductionOpUntyped *redop,
					       RegionInstanceImpl *parent,
                                               const Realm::ProfilingRequestSet &reqs);
      ProcessorGroup *get_free_proc_group(const std::vector<Processor>& members);

      const ReductionOpUntyped* get_reduction_op(ReductionOpID redop);

      // Return events that are free
      void free_event(EventImpl *event);
      void free_reservation(ReservationImpl *reservation);
      void free_metadata(IndexSpaceImpl *impl);
      void free_instance(RegionInstanceImpl *impl);
    public:
      // A nice helper method for debugging events
      void print_event_waiters(void);
    protected:
      static RuntimeImpl *runtime;
      static DMAQueue *dma_queue;
    protected:
      friend class Realm::Machine;
      friend class Realm::Runtime;
    public:
      struct TaskTableEntry {
	Processor::TaskFuncPtr func_ptr;
	void *userdata;
	size_t userlen;
      };
      typedef std::map<Processor::TaskFuncID, TaskTableEntry> TaskTable;
      bool register_task(Processor::TaskFuncID func_id, Processor::TaskFuncPtr func_ptr,
			 const void *userdata, size_t userlen);
    protected:
      TaskTable task_table;
      std::map<ReductionOpID, const ReductionOpUntyped *> redop_table;
      std::map<CustomSerdezID, const CustomSerdezUntyped *> custom_serdez_table;
      std::set<Processor> procs;
      std::vector<EventImpl*> events;
      std::deque<EventImpl*> free_events; 
      std::vector<ReservationImpl*> reservations;
      std::deque<ReservationImpl*> free_reservations;
      std::vector<MemoryImpl*> memories;
      std::vector<ProcessorImpl*> processors;
      std::vector<ProcessorGroup*> proc_groups;
      std::vector<IndexSpaceImpl*> metadatas;
      std::deque<IndexSpaceImpl*> free_metas;
      std::vector<RegionInstanceImpl*> instances;
      std::deque<RegionInstanceImpl*> free_instances;
      MachineImpl *machine;
      pthread_t *background_pthread;
      pthread_rwlock_t event_lock;
      pthread_mutex_t  free_event_lock;
      pthread_rwlock_t reservation_lock;
      pthread_mutex_t  free_reservation_lock;
      pthread_rwlock_t proc_group_lock;
      pthread_rwlock_t metadata_lock;
      pthread_mutex_t  free_metas_lock;
      pthread_rwlock_t allocator_lock;
      pthread_mutex_t  free_alloc_lock;
      pthread_rwlock_t instance_lock;
      pthread_mutex_t  free_inst_lock;
    };

    /* static */
    RuntimeImpl *RuntimeImpl::runtime = NULL;
    DMAQueue *RuntimeImpl::dma_queue = NULL;

    class EventWaiter {
    public:
      virtual ~EventWaiter(void) { }
      virtual bool event_triggered(void) = 0;
      virtual void print_info(FILE *f) = 0;
    };

    class ExternalWaiter : public EventWaiter {
    public:
      ExternalWaiter(void)
        : ready(false)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex, NULL));
        PTHREAD_SAFE_CALL(pthread_cond_init(&cond, NULL));
      }
      virtual ~ExternalWaiter(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_destroy(&mutex));
        PTHREAD_SAFE_CALL(pthread_cond_destroy(&cond));
      }
    public:
      virtual bool event_triggered(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
        ready = true;
        PTHREAD_SAFE_CALL(pthread_cond_signal(&cond));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
        // Don't delete us, we're on the stack
        return false;
      }
      virtual void print_info(FILE *f) 
      {
        fprintf(f,"External waiter");
      }
    public:
      void wait(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
        if (!ready) {
          PTHREAD_SAFE_CALL(pthread_cond_wait(&cond, &mutex));
        }
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
      }
    protected:
      pthread_mutex_t mutex;
      pthread_cond_t cond;
      bool ready;
    };

    class DMAOperation : public EventWaiter {
    public:
      DMAOperation(void) : capture_timeline(false), capture_usage(false) { }
      DMAOperation(const Realm::ProfilingRequestSet &reqs);
      virtual ~DMAOperation(void);
      virtual bool event_triggered(void);
      virtual void print_info(FILE *f) = 0;
      virtual void perform(void) = 0;
    public:
      Realm::ProfilingRequestSet requests;
      Realm::ProfilingMeasurementCollection measurements;
      Realm::ProfilingMeasurements::OperationTimeline timeline;
      Realm::ProfilingMeasurements::OperationMemoryUsage usage;
      bool capture_timeline;
      bool capture_usage;
    };

    class DMAQueue {
    public:
      DMAQueue(unsigned num_threads);
    public:
      void start(void);
      void shutdown(void);
      void run_dma_loop(void);
      void enqueue_dma(DMAOperation *op);
    public:
      static void* start_dma_thread(void *args);
    public:
      const unsigned num_dma_threads;
    protected:
      bool dma_shutdown;
      pthread_mutex_t dma_lock;
      pthread_cond_t dma_cond;
      std::vector<pthread_t> dma_threads;
      std::deque<DMAOperation*> ready_ops;
    };
    
    struct TimerStackEntry {
    public:
      int timer_kind;
      double start_time;
      double accum_child_time;
    };

    struct PerThreadTimerData {
    public:
      PerThreadTimerData(void)
      {
        //unsigned *local_proc_id = (unsigned*)pthread_getspecific(local_proc_key);
        //thread = *local_proc_id; 
        mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
        PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
      }
      ~PerThreadTimerData(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
        free(mutex);
      }

      //unsigned thread;
      std::list<TimerStackEntry> timer_stack;
      std::map<int, double> timer_accum;
      pthread_mutex_t *mutex;
    };

    static size_t find_field(const std::vector<size_t>& field_sizes,
			     size_t offset, size_t size,
			     size_t& field_start, size_t& field_size, size_t& within_field)
    {
      size_t start = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end(); 
	  it++) {
	if(offset < *it) {
	  // we're in this field
	  field_start = start;
	  field_size = *it;
	  within_field = offset;
	  // if size is nonzero, make sure it fits in the field
	  if(size && ((offset + size) <= *it)) {
	    return size;
	  } else {
	    return (*it - offset);
	  }
	} else {
	  // try the next field
	  start += *it;
	  offset -= *it;
	}
      }
      // fall through means there is no field
      return 0;
    }

    ////////////////////////////////////////////////////////
    // Event Impl (up here since we need it in Processor Impl) 
    ////////////////////////////////////////////////////////

    class EventImpl {
    public:
	typedef unsigned EventIndex;
	typedef unsigned EventGeneration;
    public:
        class WaiterInfo {
        public:
          WaiterInfo(void)
            : waiter(NULL), gen_needed(0) { }
          WaiterInfo(EventWaiter *w, EventGeneration g)
            : waiter(w), gen_needed(g) { }
        public:
          EventWaiter *waiter;
          EventGeneration gen_needed;
        };
    public:
        class DeferredTrigger : public EventWaiter {
        public:
          DeferredTrigger(EventImpl *t, unsigned c = 1)
            : target(t), count(c) { }
          virtual ~DeferredTrigger(void) { }
        public:
          virtual bool event_triggered(void) {
            target->trigger(count);
            return true;
          }
          virtual void print_info(FILE *f) {
            Event e = target->get_event();
            fprintf(f,"deferred trigger: after=" IDFMT "/%d\n",
                    e.id, e.gen+1);
          }
        protected:
          EventImpl *target;
          unsigned count;
        };
        class EventMerger : public EventWaiter {
        public:
          EventMerger(EventImpl *t, unsigned p, unsigned c = 1)
            : target(t), pending(p+1), count(c) { }
          virtual ~EventMerger(void) { }
        public:
          virtual bool event_triggered(void) {
            unsigned remaining = __sync_sub_and_fetch(&pending,1);
            bool last = (remaining == 0);
            if (last)
              target->trigger(count);
            return last;
          }
          virtual void print_info(FILE *f) {
            Event e = target->get_event();
            fprintf(f,"event merger: after=" IDFMT "/%d\n",
                    e.id, e.gen+1);
          }
          void register_waiter(EventImpl *source, Event wait_for) {
            source->add_waiter(wait_for.gen, this);
          }
          bool arm(void) {
            return event_triggered();
          }
        protected:
          EventImpl *target;
          unsigned pending;
          unsigned count;
        };
        class DeferredArrival : public EventWaiter {
        public:
          DeferredArrival(EventImpl *t, unsigned c, EventGeneration g)
            : target(t), count(c), apply_gen(g) { }
          virtual ~DeferredArrival(void) { }
        public:
          virtual bool event_triggered(void) {
            target->perform_arrival(count, Event::NO_EVENT, apply_gen);
            return true;
          }
          virtual void print_info(FILE *f) {
            Event e = target->get_event();
            fprintf(f,"deferred arrival of %d for gen %d: after=" IDFMT "/%d\n",
                    count, apply_gen, e.id, e.gen+1);
          }
        protected:
          EventImpl *target;
          unsigned count;
          EventGeneration apply_gen;
        };
    public:
	EventImpl(EventIndex idx, bool activate=false) 
		: index(idx)
	{
	  in_use = activate;
	  generation = 0;
          free_generation = 0;
	  sources = 0;
          mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
          wait_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
	  PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
	  PTHREAD_SAFE_CALL(pthread_cond_init(wait_cond,NULL));
	  if (in_use)
	  {
	    // Always initialize the current event to hand out to
	    // generation + 1, so the event will have triggered
	    // when the event matches the generation
	    current.id = index;
	    current.gen = generation+1;
            free_generation = generation+1;
	    sources = 1;
#ifdef DEBUG_REALM
	    assert(current.exists());
#endif
          }
	}
        ~EventImpl(void)
        {
          PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
          PTHREAD_SAFE_CALL(pthread_cond_destroy(wait_cond));
          free(mutex);
          free(wait_cond);

	  // free barrier-related data (if present)
	  if(initial_value)
	    free(initial_value);
	  for(std::map<Event::gen_t, void *>::iterator it = final_values.begin();
	      it != final_values.end();
	      it++)
	    free(it->second);
        }
	
	// test whether an event has triggered without waiting
	bool has_triggered(EventGeneration needed_gen);
	// block until event has triggered
	void wait(EventGeneration needed_gen);
        // defer triggering of an event on another event
        void defer_trigger(Event wait_for);
	// create an event that won't trigger until all input events have
	Event merge_events(const std::map<EventImpl*,Event> &wait_for);
	// Trigger the event
	void trigger(unsigned count = 1);
	// Check to see if the event is active, if not activate it (return true), otherwise false
	bool activate(void);	
	// Register a dependent event, return true if event had not been triggered and was registered
        void add_waiter(EventGeneration needed_gen, EventWaiter *waiter);
	// Return an event for this EventImplementation
	Event get_event();
        // Return a user event for this EventImplementation
        UserEvent get_user_event();
        // Return a barrier for this EventImplementation
        Barrier get_barrier(unsigned expected_arrivals, ReductionOpID _redop_id,
			    const void *_initial_value, size_t _initial_value_size);
        // Alter the arrival count for the barrier
        void alter_arrival_count(int delta, EventGeneration alter_gen);
        void perform_arrival(int count, Event wait_on,
                             EventGeneration apply_gen);
        bool get_result(Event::gen_t needed_gen, void *value, size_t value_size);
        void apply_reduction(Event::gen_t apply_gen, const void *reduce_value, size_t reduce_value_size);

        void add_happens_before_set(Event::gen_t gen, const std::set<Event>& happens_before, bool all_must_trigger);
    public:
        // A debug helper method
        void print_waiters(void);
    private: 
	bool in_use;
	unsigned sources;
        unsigned arrivals; // for use with barriers
	const EventIndex index;
	EventGeneration generation;
        EventGeneration free_generation;
	// The version of the event to hand out (i.e. with generation+1)
	// so we can detect when the event has triggered with testing
	// generational equality
	Event current; 
	pthread_mutex_t *mutex;
	pthread_cond_t *wait_cond;
        std::list<WaiterInfo> waiters;
        const ReductionOpUntyped *redop;
        void *initial_value;
        std::map<Event::gen_t, void *> final_values;
    private:
        struct PendingArrival {
        public:
          PendingArrival(void)
            : count(0), wait_on(Event::NO_EVENT) { }
          PendingArrival(int c, Event e)
            : count(c), wait_on(e) { }
        public:
          int count;
          Event wait_on;
        };
        // for use with barriers
        std::map<EventGeneration,int/*alterations*/> pending_alterations;
        std::map<EventGeneration,std::deque<PendingArrival> > pending_arrivals;

        // happens before advice from app
        struct HappensBeforePair {
	  std::set<Event> events;
	  bool all_must_trigger;
	};
        std::map<EventGeneration,std::deque<HappensBeforePair> > happens_before_sets;
    }; 

    ////////////////////////////////////////////////////////
    // Processor Impl (up here since we need it in Event) 
    ////////////////////////////////////////////////////////

    class ProcessorImpl {
    public:
        // For creation of normal processors when there is no utility processors
        ProcessorImpl(pthread_barrier_t *init, const RuntimeImpl::TaskTable &table, 
                      Processor p, size_t stacksize, Processor::Kind kind) 
          : init_bar(init), task_table(table), proc(p), 
            proc_kind(kind), utility_proc(this),
            shutdown(false), shutdown_trigger(false), 
            stack_size(stacksize), running_thread(NULL)
        {
          utility.id = p.id; // we are our own utility processor
          initialize_state();
        }
        // For the creation of normal processors when there are utility processors
        ProcessorImpl(pthread_barrier_t *init, const RuntimeImpl::TaskTable &table,
                      Processor p, size_t stacksize, ProcessorImpl *util)
          : init_bar(init), task_table(table), proc(p), 
            utility(util->get_utility_processor()), 
            proc_kind(Processor::LOC_PROC), utility_proc(util),
            shutdown(false), shutdown_trigger(false), 
            stack_size(stacksize), running_thread(NULL)
        {
          initialize_state();
        }
        virtual ~ProcessorImpl(void)
        {
          PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
          PTHREAD_SAFE_CALL(pthread_cond_destroy(wait_cond));
          free(mutex);
          free(wait_cond);
        }
    protected:
        void initialize_state(void);
    public:
        // Operations for utility processors
        Processor get_utility_processor(void) const;
        Processor get_id(void) const { return proc; }
        Processor::Kind get_proc_kind(void) const { return proc_kind; }
    public:
        void add_to_group(ProcessorGroup *grp) { groups.push_back(grp); }
        virtual void get_group_members(std::vector<Processor>& members);
    public:
        virtual Event spawn(Processor::TaskFuncID func_id, const void * args,
                            size_t arglen, Event wait_on, int priority);
        virtual Event spawn(Processor::TaskFuncID func_id, const void * args,
                            size_t arglen, const Realm::ProfilingRequestSet &requests,
                            Event wait_on, int priority);
    protected:
	class TaskDesc {
        public:
          TaskDesc(Processor::TaskFuncID id, const void *_args, size_t _arglen,
                   EventImpl *_complete, int _priority,
                   int _start_arrivals, int _finish_arrivals, int _expected)
            : func_id(id), args(0), arglen(_arglen),
              complete(_complete), priority(_priority), 
              start_arrivals(_start_arrivals), finish_arrivals(_finish_arrivals),
              expected(_expected), capture_timeline(false), capture_usage(false)
          {
            if (arglen > 0)
            {
              args = malloc(arglen);
              memcpy(args, _args, arglen);
            }
          }
          TaskDesc(Processor::TaskFuncID id, const void *_args, size_t _arglen,
                   EventImpl *_complete, int _priority,
                   int _start_arrivals, int _finish_arrivals, int _expected,
                   const Realm::ProfilingRequestSet &reqs)
            : func_id(id), args(0), arglen(_arglen),
              complete(_complete), priority(_priority), 
              start_arrivals(_start_arrivals), finish_arrivals(_finish_arrivals),
              expected(_expected), requests(reqs)
          {
            if (arglen > 0)
            {
              args = malloc(arglen);
              memcpy(args, _args, arglen);
            }
            measurements.import_requests(requests);
            capture_timeline = measurements.wants_measurement<
                                Realm::ProfilingMeasurements::OperationTimeline>(); 
            capture_usage = measurements.wants_measurement<
                                Realm::ProfilingMeasurements::OperationProcessorUsage>();
            if (capture_timeline)
              timeline.record_create_time();
          }
          ~TaskDesc(void)
          {
            if (requests.request_count() > 0) {
              if (capture_timeline)
                measurements.add_measurement(timeline);
              if (capture_usage)
                measurements.add_measurement(usage);
              measurements.send_responses(requests);
            }
            if (args)
              free(args);
          }
	public:
          Processor::TaskFuncID func_id;
          void * args;
          size_t arglen;
          EventImpl *complete;
          int priority;
          // Used for shared tasks assigned to processor groups
          int start_arrivals;
          int finish_arrivals;
          int expected;
          Realm::ProfilingRequestSet requests;
          Realm::ProfilingMeasurementCollection measurements;
          Realm::ProfilingMeasurements::OperationTimeline timeline;
          Realm::ProfilingMeasurements::OperationProcessorUsage usage;
          bool capture_timeline;
          bool capture_usage;
	};
        class DeferredTask : public EventWaiter {
        public:
          DeferredTask(ProcessorImpl *t, TaskDesc *d)
            : target(t), task(d) { }
          virtual ~DeferredTask(void) { }
        public:
          virtual bool event_triggered(void) {
            target->enqueue_task(task, Event::NO_EVENT);
            return true;
          }
          virtual void print_info(FILE *f) {
            Event e = task->complete->get_event();
            fprintf(f,"deferred task: after=" IDFMT "/%d\n",
                    e.id, e.gen+1);
          }
        protected:
          ProcessorImpl *target;
          TaskDesc *task;
        };
      public:
        class ProcessorThread : public EventWaiter {
        public:
          enum ThreadState {
            RUNNING_STATE,
            PAUSED_STATE,
            RESUMABLE_STATE,
            SLEEPING_STATE, // about to sleep
            SLEEP_STATE,
          };
        public:
          ProcessorThread(ProcessorImpl *impl, size_t stack_size);
          ~ProcessorThread(void);
        public:
          void do_initialize(void) { initialize = true; }
          void do_finalize(void) { finalize = true; }
        public:
          static void* entry(void *arg);
          void run(void);
          void awake(void);
          void sleep(void);
          void prepare_to_sleep(void);
          void preempt(EventImpl *event, EventImpl::EventGeneration needed); 
          void resume(void);
          void shutdown(void);
          void start(void);
          inline Processor get_processor(void) const { return proc->get_id(); } 
        public:
          virtual bool event_triggered(void);
          virtual void print_info(FILE *f) {
            fprintf(f, "Paused thread %lx\n", (unsigned long)thread);
          }
        public:
          ProcessorImpl *const proc;
        protected:
          pthread_t thread;
          pthread_attr_t attr;
          ThreadState state;
          pthread_mutex_t *thread_mutex;
          pthread_cond_t *thread_cond;
          bool initialize;
          bool finalize;
        };
    public:
        void start_processor(void);
        void shutdown_processor(void);
        void initialize_processor(void);
        void finalize_processor(void);
        void process_kill(void);
        bool is_idle(void);
        void pause_thread(ProcessorThread *thread);
        void resume_thread(ProcessorThread *thread);
    public:
        void enqueue_task(TaskDesc *task, Event wait_on);
    protected:
        void add_to_ready_queue(TaskDesc *desc);
	bool execute_task(ProcessorThread *thread);
        void run_task(TaskDesc *task);
    public:
        pthread_attr_t attr; // For setting pthread parameters when starting the thread
    protected:
        pthread_barrier_t *init_bar;
        const RuntimeImpl::TaskTable& task_table;
	Processor proc;
        Processor utility;
        Processor::Kind proc_kind;
        ProcessorImpl *utility_proc;
	std::list<TaskDesc*> ready_queue;
	pthread_mutex_t *mutex;
	pthread_cond_t *wait_cond;
	// Used for detecting the shutdown condition
	bool shutdown, shutdown_trigger;
        const size_t stack_size;
        std::vector<ProcessorGroup *> groups;  // groups this proc is a member of
    protected:
        ProcessorThread               *running_thread;
        std::set<ProcessorThread*>    paused_threads;
        std::deque<ProcessorThread*>  resumable_threads;
        std::vector<ProcessorThread*> available_threads;
    };

    class ProcessorGroup : public ProcessorImpl {
    public:
      static const Processor::id_t FIRST_PROC_GROUP_ID = 1000;

      ProcessorGroup(Processor p) 
	: ProcessorImpl(0 /*init*/, RuntimeImpl::TaskTable(), p, 0 /*stacksize*/, 
                        Processor::PROC_GROUP), next_target(0)
      {
      }

      void add_member(ProcessorImpl *new_member) {
	members.push_back(new_member);
	new_member->add_to_group(this);
      }
      
      virtual void get_group_members(std::vector<Processor>& members);

      virtual Event spawn(Processor::TaskFuncID func_id, const void * args,
			  size_t arglen, Event wait_on, int priority);
      virtual Event spawn(Processor::TaskFuncID func_id, const void * args,
                            size_t arglen, const Realm::ProfilingRequestSet &requests,
                            Event wait_on, int priority);
    protected:
      std::vector<ProcessorImpl *> members;
      size_t next_target;
    };

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    ////////////////////////////////////////////////////////
    // Events 
    ////////////////////////////////////////////////////////
 
    /* static */ const Event Event::NO_EVENT = { 0, 0 };
    // Take this you POS c++ type system
    /* static */ const UserEvent UserEvent::NO_USER_EVENT = 
      *(static_cast<UserEvent*>(const_cast<Event*>(&Event::NO_EVENT)));

    bool Event::has_triggered(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	if (!id) return true;
	EventImpl *e = RuntimeImpl::get_runtime()->get_event_impl(*this);
	return e->has_triggered(gen);
    }

    void Event::wait(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL); 
	if (!id) return;
	EventImpl *e = RuntimeImpl::get_runtime()->get_event_impl(*this);
	e->wait(gen);
    }

    // used by non-legion threads to wait on an event - always blocking
    void Event::external_wait(void) const
    {
      if (!id) return;
      EventImpl *e = RuntimeImpl::get_runtime()->get_event_impl(*this);
      ExternalWaiter waiter;
      e->add_waiter(gen, &waiter);
      waiter.wait();
    }

    Event Event::merge_events(Event ev1, Event ev2, Event ev3,
                              Event ev4, Event ev5, Event ev6)
    {
      std::set<Event> wait_for;
      if (ev1.exists()) wait_for.insert(ev1);
      if (ev2.exists()) wait_for.insert(ev2);
      if (ev3.exists()) wait_for.insert(ev3);
      if (ev4.exists()) wait_for.insert(ev4);
      if (ev5.exists()) wait_for.insert(ev5);
      if (ev6.exists()) wait_for.insert(ev6);

      if (wait_for.empty())
        return Event::NO_EVENT;
      else if (wait_for.size() == 1)
        return *(wait_for.begin());
      else
        return merge_events(wait_for);
    }

    Event Event::merge_events(const std::set<Event>& wait_for)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        size_t wait_for_size = wait_for.size();
        // Ignore any no-events
        // Fast-outs for cases where there is 0 or 1 existing events
        if (wait_for.find(Event::NO_EVENT) != wait_for.end())
        {
          // Ignore the no event
          wait_for_size--;
          if (wait_for_size == 1)
          {
            Event result = Event::NO_EVENT;
            // Find the actual event
            for (std::set<Event>::const_iterator it = wait_for.begin();
                  it != wait_for.end(); it++)
            {
              result = *it;
              if (result.exists())
              {
                break;
              }
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(result.exists());
#endif
            return result;
          }
        }
        else if (wait_for_size == 1)
        {
          // wait for size is 1, which means there is only one event
          Event result = *(wait_for.begin());
#ifdef DEBUG_HIGH_LEVEL
          assert(result.exists());
#endif
          return result;
        }
        // Check to make sure we have valid events
        if (wait_for_size == 0)
        {
          return Event::NO_EVENT;
        }
        // Get a new event
	EventImpl *e = RuntimeImpl::get_runtime()->get_free_event();
        // Get the implementations for all the wait_for events
        // Do this to avoid calling get_event_impl while holding the event lock
        std::map<EventImpl*,Event> wait_for_impl;
        for (std::set<Event>::const_iterator it = wait_for.begin();
              it != wait_for.end(); it++)
        {
          assert(wait_for_impl.size() < wait_for.size());
          if (!(*it).exists())
            continue;
          EventImpl *src_impl = RuntimeImpl::get_runtime()->get_event_impl(*it);
          std::pair<EventImpl*,Event> made_pair(src_impl,*it);
          wait_for_impl.insert(std::pair<EventImpl*,Event>(src_impl,*it));
        }
	return e->merge_events(wait_for_impl);
    }

    /*static*/ void Event::advise_event_ordering(Event happens_before,
						 Event happens_after)
    {
      std::set<Event> s;
      s.insert(happens_before);
      EventImpl *e = RuntimeImpl::get_runtime()->get_event_impl(happens_after);
      e->add_happens_before_set(happens_after.gen, s, true);
    }

    /*static*/ void Event::advise_event_ordering(const std::set<Event>& happens_before,
						 Event happens_after,
						 bool all_must_trigger /*= true*/)
    {
      EventImpl *e = RuntimeImpl::get_runtime()->get_event_impl(happens_after);
      e->add_happens_before_set(happens_after.gen, happens_before, all_must_trigger);
    }

    ////////////////////////////////////////////////////////
    // User Events (just use base event impl) 
    ////////////////////////////////////////////////////////

    UserEvent UserEvent::create_user_event(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      EventImpl *impl = RuntimeImpl::get_runtime()->get_free_event();
      return impl->get_user_event();
    }

    void UserEvent::trigger(Event wait_on /*= NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return;
      EventImpl *impl = RuntimeImpl::get_runtime()->get_event_impl(*this);
      if (wait_on.exists())
        impl->defer_trigger(wait_on);
      else
        impl->trigger();
    }

    ////////////////////////////////////////////////////////
    // Barrier Events (have to use same base impl)
    ////////////////////////////////////////////////////////
    
    Barrier Barrier::create_barrier(unsigned expected_arrivals, ReductionOpID redop_id /*= 0*/,
				    const void *initial_value /*= 0*/, size_t initial_value_size /*= 0*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      EventImpl *impl = RuntimeImpl::get_runtime()->get_free_event();
      Barrier b = impl->get_barrier(expected_arrivals, redop_id, initial_value, initial_value_size);
      //log_barrier.info("barrier " IDFMT ".%d - create %d", b.id, b.gen, expected_arrivals);
      
      return b;
    }

    void Barrier::destroy_barrier(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      EventImpl *impl = RuntimeImpl::get_runtime()->get_event_impl(*this);
      RuntimeImpl::get_runtime()->free_event(impl);
    }

    Barrier Barrier::advance_barrier(void) const
    {
      Barrier next(*this);
      next.gen++;
      return next;
    }

    Barrier Barrier::alter_arrival_count(int delta) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return *this;
      EventImpl *impl = RuntimeImpl::get_runtime()->get_event_impl(*this);
      impl->alter_arrival_count(delta, gen);
      return *this;
    }

    Barrier Barrier::get_previous_phase(void) const
    {
      Barrier result = *this;
      result.gen--;
      return result;
    }

    bool Barrier::get_result(void *value, size_t value_size) const
    {
      EventImpl *impl = RuntimeImpl::get_runtime()->get_event_impl(*this);
      return impl->get_result(gen, value, value_size);
    }

    void Barrier::arrive(unsigned count /*=1*/, Event wait_on /*= Event::NO_EVENT*/,
			 const void *reduce_value /*= 0*/, size_t reduce_value_size /*= 0*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return;

      EventImpl *impl = RuntimeImpl::get_runtime()->get_event_impl(*this);
      // Do this before the arrival to avoid a race
      if(reduce_value_size > 0)
	impl->apply_reduction(gen, reduce_value, reduce_value_size);
      impl->perform_arrival(count, wait_on, gen);
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    bool EventImpl::has_triggered(EventGeneration needed_gen)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	result = (needed_gen <= generation);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void EventImpl::wait(EventGeneration needed_gen)
    {
      ProcessorImpl::ProcessorThread *thread = 
        (ProcessorImpl::ProcessorThread*)pthread_getspecific(local_thread_key);
      thread->preempt(this, needed_gen);
    }

    void EventImpl::defer_trigger(Event wait_for)
    {
      DeferredTrigger *thunk = new DeferredTrigger(this);
      EventImpl *src_impl = RuntimeImpl::get_runtime()->get_event_impl(wait_for);
      src_impl->add_waiter(wait_for.gen, thunk);
    }

    Event EventImpl::merge_events(const std::map<EventImpl*,Event> &wait_for)
    {
      Event ret = current;
      EventMerger *merger = new EventMerger(this, wait_for.size());
      for (std::map<EventImpl*,Event>::const_iterator it = 
            wait_for.begin(); it != wait_for.end(); it++)
      {
        merger->register_waiter(it->first, it->second);
      }
      bool nuke = merger->arm();
      if (nuke)
        delete merger;
      return ret;
    } 

    void EventImpl::trigger(unsigned count)
    {
      // Update the generation
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_REALM
      assert(in_use);
      assert(sources >= count);
#endif
      sources -= count;
      bool finished = false;
      // These is only used by barriers that have pending operations
      bool trigger_again = false;
      std::deque<PendingArrival> pending_copy;
      if (sources == 0)
      {
#ifdef DEBUG_PRINT
        //DPRINT2("Event %u triggered for generation %u\n",index,generation);
#endif
        // Increment the generation so that nobody can register a triggerable
        // with this event, but keep event in_use so no one can use the event
        generation++;
#ifdef DEBUG_REALM
        assert(generation == current.gen);
#endif
        // Get the set of people to trigger
        std::vector<EventWaiter*> to_trigger;
        for (std::list<WaiterInfo>::iterator it = waiters.begin();
              it != waiters.end(); /*nothing*/)
        {
          if (it->gen_needed == generation)
          {
            to_trigger.push_back(it->waiter);
            it = waiters.erase(it);
          }
          else
            it++;
        }
        // double check that happens_before_sets were correct
        {
          std::map<EventGeneration,std::deque<HappensBeforePair> >::iterator
            finder = happens_before_sets.find(generation);
          if (finder != happens_before_sets.end()) {
            for(std::deque<HappensBeforePair>::iterator it = finder->second.begin();
                it != finder->second.end();
                it++) {
              // count how many of the events in this set have triggered
              bool any_triggered = false;
              for(std::set<Event>::iterator it2 = it->events.begin();
                  it2 != it->events.end();
                  it2++) {
                if (it2->has_triggered())
                  any_triggered = true;
                else
                  if (it->all_must_trigger) {
                    assert(false);  // debugger trap for now
                  }
              }
              if (!any_triggered) {
                assert(false);  // debugger trap for now
              }
            }
            happens_before_sets.erase(finder);
          }
        }
        finished = (generation == free_generation);
        if (finished)
        {
          in_use = false;
          assert(waiters.empty());
        }
        else
        {
          // Otherwise we are a barrier so update the state
          sources = arrivals;
          current.gen++;
          // Also check to see if we have any pending operations 
          {
            std::map<EventGeneration,int>::iterator finder = 
              pending_alterations.find(current.gen);
            if (finder != pending_alterations.end()) {
              sources += finder->second;
              if (sources == 0)
                trigger_again = true;
              pending_alterations.erase(finder);
            }
          }
          // Also see if we have any pending arrivals
          {
            std::map<EventGeneration,std::deque<PendingArrival> >::iterator
              finder = pending_arrivals.find(current.gen);
            if (finder != pending_arrivals.end()) {
              pending_copy = finder->second;
              pending_arrivals.erase(finder);
            }
          }
        }
        // Wake up any waiters
        PTHREAD_SAFE_CALL(pthread_cond_broadcast(wait_cond));
        // Can't be holding the lock when triggering other triggerables
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Trigger any dependent events for this generation
        for (std::vector<EventWaiter*>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
        {
          bool nuke = (*it)->event_triggered();
          if (nuke)
            delete (*it);
        }
      }
      else
      {
        // Not done so release the lock
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));	
      }
      // tell the runtime that we're free
      if (finished)
        RuntimeImpl::get_runtime()->free_event(this);
      // Do some extra work for barriers
      if (!pending_copy.empty()) {
        for (std::deque<PendingArrival>::const_iterator it = 
              pending_copy.begin(); it != pending_copy.end(); it++)
        {
          // Not the most efficient way of doing this but it works
          for (int idx = 0; idx < it->count; idx++)
            defer_trigger(it->wait_on);
        }
      }
      if (trigger_again)
        trigger(0);
    }

    bool EventImpl::activate(void)
    {
	bool result = false;
        // Try acquiring the lock, if we don't get it then just move on
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!in_use)
	{
		in_use = true;
		result = true;
		sources = 1;
		// Set generation to generation+1, see 
		// comment in constructor
		current.id = index;
		current.gen = generation+1;
                free_generation = generation+1;
#ifdef DEBUG_REALM
		assert(current.exists());
#endif
	}	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void EventImpl::add_waiter(Event::gen_t gen_needed, EventWaiter *waiter)
    {
      bool trigger_now = false;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (gen_needed > generation)
        waiters.push_back(WaiterInfo(waiter, gen_needed));
      else
        trigger_now = true;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      if (trigger_now) {
        bool nuke = waiter->event_triggered();
        if (nuke)
          delete waiter;
      }
    }

    Event EventImpl::get_event() 
    {
#ifdef DEBUG_REALM
        assert(in_use);
#endif
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	Event result = current;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    UserEvent EventImpl::get_user_event()
    {
#ifdef DEBUG_REALM
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      UserEvent result; 
      result.id = current.id;
      result.gen = current.gen;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    Logger::Category log_barrier("barrier");

    Barrier EventImpl::get_barrier(unsigned expected_arrivals, ReductionOpID _redop_id,
				   const void *_initial_value, size_t _initial_value_size)
    {
#ifdef DEBUG_REALM
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      Barrier result;
      result.id = current.id;
      result.gen = current.gen;
      // Set the number of expected arrivals
      sources = expected_arrivals;
      arrivals = expected_arrivals;
      // Make sure we don't prematurely free this event
      free_generation = (unsigned)-1;

      if(_redop_id) {
	redop = RuntimeImpl::get_runtime()->get_reduction_op(_redop_id);
	assert(redop->sizeof_lhs == _initial_value_size);
	initial_value = malloc(_initial_value_size);
	memcpy(initial_value, _initial_value, _initial_value_size);
      } else {
	assert(_initial_value_size == 0);
	redop = 0;
	initial_value = 0;
      }

      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    void EventImpl::alter_arrival_count(int delta, EventGeneration alter_gen)
    {
#ifdef DEBUG_REALM
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      // Check to see if we are on the right generation
      if (alter_gen > (generation+1)) {
        // Add this to the list of pending alterations for future generation
        std::map<EventGeneration,int>::iterator finder = 
          pending_alterations.find(alter_gen);
        if (finder != pending_alterations.end())
          finder->second += delta;
        else
          pending_alterations[alter_gen] = delta;
      } else {
        // We're working on the current generation
#ifdef DEBUG_REALM
        if (delta < 0) // If we're deleting, make sure nothing weird happens
          assert(int(sources) > (-delta));
#endif
        sources += delta;
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      //log_barrier.info("barrier " IDFMT ".%d - adjust %d + %d = %d",
      //	       current.id, current.gen, old_sources, delta, old_sources + delta);
    }

    void EventImpl::perform_arrival(int count, Event wait_on,
                                    EventGeneration apply_gen)
    {
#ifdef DEBUG_REALM
      assert(in_use);
#endif
      if (wait_on.exists()) {
        EventImpl *src_impl = RuntimeImpl::get_runtime()->get_event_impl(wait_on);
        DeferredArrival *waiter = new DeferredArrival(this, count, apply_gen);
        src_impl->add_waiter(wait_on.gen, waiter);
        return;
      }
      bool trigger_now = true;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      // Check to see if we are on the right generation
      if (apply_gen > (generation+1)) {
        trigger_now = false;
        // If there is no wait on, then we can just update
        // the pending alteration count
        std::map<EventGeneration,int>::iterator finder = 
          pending_alterations.find(apply_gen);
        if (finder != pending_alterations.end())
          finder->second -= count;
        else
          pending_alterations[apply_gen] = -count;
      } 
      // This is the right generation, so see we can trigger the count 
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      if (trigger_now)
        trigger(count);
    }

    bool EventImpl::get_result(Event::gen_t needed_gen, void *value, size_t value_size)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      bool result = (needed_gen <= generation);
      if(result) {
	assert(redop != 0);
	assert(value_size == redop->sizeof_lhs);
	std::map<Event::gen_t, void *>::iterator it = final_values.find(needed_gen);
	assert(it != final_values.end());

	memcpy(value, it->second, value_size);
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));

      return result;
    } 
      
    void EventImpl::apply_reduction(Event::gen_t apply_gen, const void *reduce_value, size_t reduce_value_size)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));

      assert(redop != 0);
      assert(reduce_value_size == redop->sizeof_rhs);

      std::map<Event::gen_t, void *>::iterator it = final_values.find(apply_gen);
      if(it == final_values.end()) {
	// new entry
	void *newbuf = malloc(redop->sizeof_lhs);
	memcpy(newbuf, initial_value, redop->sizeof_lhs);
	redop->apply(newbuf, reduce_value, 1);
	final_values[apply_gen] = newbuf;
      } else {
	redop->apply(it->second, reduce_value, 1);
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void EventImpl::add_happens_before_set(Event::gen_t gen, 
					   const std::set<Event>& happens_before,
					   bool all_must_trigger)
    {
      // hold lock to access happens_before_sets
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      happens_before_sets[gen].push_back(HappensBeforePair());
      HappensBeforePair &p = happens_before_sets[gen].back();
      p.events.insert(happens_before.begin(), happens_before.end());
      p.all_must_trigger = all_must_trigger;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void EventImpl::print_waiters(void)
    {
      // No need to hold the lock because this method
      // will only ever be called from a debugger
      if (in_use && !waiters.empty())
      {
        fprintf(stdout,"Event %d, Generation %d has %ld waiters\n",
            index, generation, waiters.size());
        for (std::list<WaiterInfo>::const_iterator it = waiters.begin();
              it != waiters.end(); it++)
        {
          fprintf(stdout,"  gen=%d Waiter %p:", it->gen_needed, it->waiter);
          it->waiter->print_info(stdout);
          fprintf(stdout,"\n");
        }
        fflush(stdout);
      }
    }

    ////////////////////////////////////////////////////////
    // Reservation 
    ////////////////////////////////////////////////////////

    Logger::Category log_reservation("reservation");

    class ReservationImpl {
    public:
      class DeferredAcquire : public EventWaiter {
      public:
        DeferredAcquire(ReservationImpl *t, unsigned m, bool e, Event a)
          : target(t), mode(m), exclusive(e), acquire(a) { }
        virtual ~DeferredAcquire(void) { }
      public:
        virtual bool event_triggered(void) {
          target->acquire(mode, exclusive, Event::NO_EVENT, acquire);
          return true;
        }
        virtual void print_info(FILE *f) {
          fprintf(f,"deferred acquire: after=" IDFMT "/%d\n",
                  acquire.id, acquire.gen);
        }
      protected:
        ReservationImpl *target;
        unsigned mode;
        bool exclusive;
        Event acquire;
      };
      class DeferredRelease : public EventWaiter {
      public:
        DeferredRelease(ReservationImpl *t)
          : target(t) { }
        virtual ~DeferredRelease(void) { }
      public:
        virtual bool event_triggered(void) {
          target->release(Event::NO_EVENT);
          return true;
        }
        virtual void print_info(FILE *f) {
          fprintf(f,"deferred release");
        }
      protected:
        ReservationImpl *target;
      };
    public:
	ReservationImpl(int idx, bool activate = false, size_t dsize = 0) : index(idx) {
		active = activate;
		taken = false;
		mode = 0;
		holders = 0;
		waiters = false;
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
                if (activate)
                {
                    if (dsize > 0)
                    {
                        data_size = dsize;
                        data = malloc(data_size);
#ifdef DEBUG_REALM
                        assert(data != NULL);
#endif
                    }
                    else
                    {
                        data_size = 0;
                        data = NULL;
                    }
                }
                else
                {
#ifdef DEBUG_REALM
                    assert(dsize == 0);
#endif
                    data_size = 0;
                    data = NULL;
                }
	}	
        ~ReservationImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
                if (data_size != 0)
                {
#ifdef DEBUG_REALM
                    assert(data != NULL);
#endif
                    free(data);
                    data = NULL;
                    data_size = 0;
                }
        }

	Event acquire(unsigned mode, bool exclusive, 
                      Event wait_on, Event post = Event::NO_EVENT);
	void release(Event wait_on);

	bool activate(size_t data_size);
	void deactivate(void);
	Reservation get_reservation(void) const;
        size_t get_data_size(void) const;
        void* get_data_ptr(void) const;
    private:
	Event register_request(unsigned m, bool exc, Event post);
	void perform_release(std::set<EventImpl*> &to_trigger);
    private:
	class ReservationRecord {
	public:
          unsigned mode;
          bool exclusive;
          Event event;
          bool handled;
	};
    private:
	const int index;
	bool active;
	bool taken;
	bool exclusive;
	bool waiters;
	unsigned mode;
	unsigned holders;
	std::list<ReservationRecord> requests;
	pthread_mutex_t *mutex;
        void *data;
        size_t data_size;
    };

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    /*static*/ const Reservation Reservation::NO_RESERVATION = { 0 };

    Event Reservation::acquire(unsigned mode, bool exclusive, Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	ReservationImpl *l = RuntimeImpl::get_runtime()->get_reservation_impl(*this);
	return l->acquire(mode,exclusive, wait_on);
    }

    void Reservation::release(Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	ReservationImpl *l = RuntimeImpl::get_runtime()->get_reservation_impl(*this);
	l->release(wait_on);
    }

    Reservation Reservation::create_reservation(size_t data_size)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	return RuntimeImpl::get_runtime()->get_free_reservation(data_size)->get_reservation();
    }

    void Reservation::destroy_reservation(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	ReservationImpl *l = RuntimeImpl::get_runtime()->get_reservation_impl(*this);
	l->deactivate();
    }

    size_t Reservation::data_size(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ReservationImpl *l = RuntimeImpl::get_runtime()->get_reservation_impl(*this);
        return l->get_data_size();
    }

    void* Reservation::data_ptr(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ReservationImpl *l = RuntimeImpl::get_runtime()->get_reservation_impl(*this);
        return l->get_data_ptr();
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    Event ReservationImpl::acquire(unsigned m, bool exc, Event wait_on, Event post)
    {
      if (wait_on.exists()) {
        EventImpl *e = RuntimeImpl::get_runtime()->get_free_event();
        Event result = e->get_event();
        DeferredAcquire *waiter = new DeferredAcquire(this, m, exc, result);
        EventImpl *source = RuntimeImpl::get_runtime()->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, waiter);
        return result;
      }
      Event result = Event::NO_EVENT;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      log_reservation.debug("reservation request: reservation=%x mode=%d "
                                  "excl=%d event=" IDFMT "/%d count=%d",
               index, m, exc, wait_on.id, wait_on.gen, holders); 
      // check to see if we have to wait on event first
      bool trigger_post = false; 
      if (taken)
      {
        // If either is exclusive we have to register the request
        if (exclusive || exc)
        {
          result = register_request(m,exc,post);
        }
        else
        {
          if ((mode == m) && !waiters)
          {
            // Not exclusive and modes are equal
            // and there are no waiters
            // Can still acquire the reservation 
            holders++;
            trigger_post = true;
          }
          else
          {
            result = register_request(m,exc,post);	
          }
        }
      }
      else
      {
        // Nobody has the reservation, grab it
        taken = true;
        trigger_post = true;
        exclusive = exc;
        mode = m;
        holders = 1;
#ifdef DEBUG_PRINT
        DPRINT3("Granting reservation %d in mode %d with exclusive %d\n",index,mode,exclusive);
#endif
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      if (trigger_post && post.exists()) {
        EventImpl *post_impl = RuntimeImpl::get_runtime()->get_event_impl(post); 
        post_impl->trigger();
      }
      return result;
    }

    // Always called while holding the mutex 
    Event ReservationImpl::register_request(unsigned m, bool exc, Event post)
    {
      ReservationRecord req;
      req.mode = m;
      req.exclusive = exc;
      if (!post.exists()) {
        // If we didn't have one yet, then make a post event
        EventImpl *e = RuntimeImpl::get_runtime()->get_free_event();
        req.event = e->get_event();
      } else {
        req.event = post;
      }
      req.handled = false;
      // Add this to the list of requests
      requests.push_back(req);

      // Finally set waiters to true if it's already true
      // or there are now threads waiting
      waiters = true;
      
      return req.event;
    }

    void ReservationImpl::release(Event wait_on)
    {
      if (wait_on.exists()) {
        DeferredRelease *waiter = new DeferredRelease(this);
        EventImpl *source = RuntimeImpl::get_runtime()->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, waiter);
        return;
      }
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      log_reservation.debug("release request: reservation=%x mode=%d excl=%d event=" IDFMT "/%d count=%d",
               index, mode, exclusive, wait_on.id, wait_on.gen, holders);
      std::set<EventImpl*> to_trigger;
      perform_release(to_trigger);
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      // Don't perform any triggers while holding the reservation's mutex 
      for (std::set<EventImpl*>::const_iterator it = to_trigger.begin();
            it != to_trigger.end(); it++)
      {
        (*it)->trigger();
      }
    }
    
    // Always called while holding the reservations's mutex
    void ReservationImpl::perform_release(std::set<EventImpl*> &to_trigger)
    {
      holders--;	
      // If the holders are zero, get the next request out of the queue and trigger it
      if (holders==0)
      {
#ifdef DEBUG_PRINT
        DPRINT1("Releasing reservation %d\n",index);
#endif
        // Clean out all the handled requests
        {
          std::list<ReservationRecord>::iterator it = requests.begin();
          while (it != requests.end())
          {
            if (it->handled)
              it = requests.erase(it);
            else
              it++;
          }
        }
        // Check to see if there are any waiters
        if (requests.empty())
        {
          waiters= false;
          taken = false;
          return;
        }
        ReservationRecord req = requests.front();
        requests.pop_front();
        // Set the mode and exclusivity
        exclusive = req.exclusive;
        mode = req.mode;
        holders = 1;
#ifdef DEBUG_PRINT
        DPRINT3("Issuing reservation %d in mode %d with exclusivity %d\n",index,mode,exclusive);
#endif
        // Trigger the event
        to_trigger.insert(RuntimeImpl::get_runtime()->get_event_impl(req.event));
        // If this isn't an exclusive mode, see if there are any other
        // requests with the same mode that aren't exclusive that we can handle
        if (!exclusive)
        {
          waiters = false;
          for (std::list<ReservationRecord>::iterator it = requests.begin();
                  it != requests.end(); it++)
          {
            if ((it->mode == mode) && (!it->exclusive) && (!it->handled))
            {
              it->handled = true;
              to_trigger.insert(RuntimeImpl::get_runtime()->get_event_impl(it->event));
              holders++;
            }
            else
            {
              // There is at least one thread still waiting
              waiters = true;
            }
          }	
        }
        else
        {
          waiters = (requests.size()>0);
        }
      }
    }

    bool ReservationImpl::activate(size_t dsize)
    {
	bool result = false;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{
		active = true;
		result = true;
		waiters = false;
                if (dsize > 0)
                {
                    data_size = dsize;
                    data = malloc(data_size);
#ifdef DEBUG_REALM
                    assert(data != NULL);
#endif
                }
                else
                {
                    data_size = 0;
                    data = NULL;
                }
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void ReservationImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;	
        if (data_size > 0)
        {
#ifdef DEBUG_REALM
            assert(data != NULL);
#endif
            free(data);
            data = NULL;
            data_size = 0;
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        RuntimeImpl::get_runtime()->free_reservation(this);
    }

    Reservation ReservationImpl::get_reservation(void) const
    {
#ifdef DEBUG_LOWL_LEVEL
        assert(index != 0);
#endif
	Reservation r;
        r.id = static_cast<id_t>(index);
	return r;
    }

    size_t ReservationImpl::get_data_size(void) const
    {
        return data_size;
    }

    void* ReservationImpl::get_data_ptr(void) const
    {
        return data;
    }

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    ////////////////////////////////////////////////////////
    // Processor 
    ////////////////////////////////////////////////////////

    /*static*/ const Processor Processor::NO_PROC = { 0 };

    // Processor Impl at top due to use in event
    
    /*static*/ Processor Processor::get_executing_processor(void)
    {
      ProcessorImpl::ProcessorThread *thread = 
        (ProcessorImpl::ProcessorThread*)pthread_getspecific(local_thread_key);
      return thread->get_processor();
    }

    Processor::Kind Processor::kind(void) const
    {
      return RuntimeImpl::get_runtime()->get_processor_impl(*this)->get_proc_kind();
    }
    
    Event Processor::spawn(Processor::TaskFuncID func_id, const void * args,
                            size_t arglen, Event wait_on, int priority) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ProcessorImpl *p = RuntimeImpl::get_runtime()->get_processor_impl(*this);
      return p->spawn(func_id, args, arglen, wait_on, priority);
    }

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
                           const Realm::ProfilingRequestSet &requests,
                           Event wait_on, int priority) const
    {
      ProcessorImpl *p = RuntimeImpl::get_runtime()->get_processor_impl(*this);
      return p->spawn(func_id, args, arglen, requests, wait_on, priority);
    }

    AddressSpace Processor::address_space(void) const
    {
      return 0;
    }

    IDType Processor::local_id(void) const
    {
      return id;
    }

    /*static*/ Processor Processor::create_group(const std::vector<Processor>& members)
    {
      return RuntimeImpl::get_runtime()->get_free_proc_group(members)->get_id();
    }

    void Processor::get_group_members(std::vector<Processor>& members)
    {
      RuntimeImpl::get_runtime()->get_processor_impl(*this)->get_group_members(members);
    }

    Logger log_taskreg("taskreg");
  

    Event Processor::register_task(TaskFuncID func_id,
				   const CodeDescriptor& codedesc,
				   const ProfilingRequestSet& prs,
				   const void *user_data /*= 0*/,
				   size_t user_data_len /*= 0*/) const
    {
      // some sanity checks first
      if(codedesc.type() != TypeConv::from_cpp_type<TaskFuncPtr>()) {
	log_taskreg.fatal() << "attempt to register a task function of improper type: " << codedesc.type();
	assert(0);
      }

      const FunctionPointerImplementation *fpi = codedesc.find_impl<FunctionPointerImplementation>();
      if(!fpi) {
	log_taskreg.fatal() << "shared lowlevel cannot register a task with no function pointer";
	assert(0);
      }

      // HACK - just fall back to global registration for now
      RuntimeImpl::get_runtime()->register_task(func_id, (TaskFuncPtr)(fpi->fnptr),
						user_data, user_data_len);

      return Event::NO_EVENT;
    }

    /*static*/ Event Processor::register_task_by_kind(Kind target_kind, bool global,
						      TaskFuncID func_id,
						      const CodeDescriptor& codedesc,
						      const ProfilingRequestSet& prs,
						      const void *user_data /*= 0*/,
						      size_t user_data_len /*= 0*/)
    {
      // some sanity checks first
      if(codedesc.type() != TypeConv::from_cpp_type<TaskFuncPtr>()) {
	log_taskreg.fatal() << "attempt to register a task function of improper type: " << codedesc.type();
	assert(0);
      }

      const FunctionPointerImplementation *fpi = codedesc.find_impl<FunctionPointerImplementation>();
      if(!fpi) {
	log_taskreg.fatal() << "shared lowlevel cannot register a task with no function pointer";
	assert(0);
      }

      // HACK - just fall back to global registration for now
      RuntimeImpl::get_runtime()->register_task(func_id, (TaskFuncPtr)(fpi->fnptr),
						user_data, user_data_len);

      return Event::NO_EVENT;
    }
};

namespace LegionRuntime {
  namespace LowLevel {

    void ProcessorImpl::initialize_state(void)
    {
        mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
        wait_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
        PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
        PTHREAD_SAFE_CALL(pthread_cond_init(wait_cond,NULL));
    }

    void ProcessorImpl::get_group_members(std::vector<Processor>& members)
    {
        // only member of the "group" is us
        members.push_back(proc);
    }

    Event ProcessorImpl::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on, int priority)
    {
      TaskDesc *task = new TaskDesc(func_id, args, arglen,
                                    RuntimeImpl::get_runtime()->get_free_event(),
                                    priority, 0, 0, 1);
      Event result = task->complete->get_event();

      enqueue_task(task, wait_on);	
      return result;
    }

    Event ProcessorImpl::spawn(Processor::TaskFuncID func_id, const void * args,
			       size_t arglen, const Realm::ProfilingRequestSet &reqs,
                               Event wait_on, int priority)
    {
      TaskDesc *task = new TaskDesc(func_id, args, arglen,
                                    RuntimeImpl::get_runtime()->get_free_event(),
                                    priority, 0, 0, 1, reqs);
      Event result = task->complete->get_event();

      enqueue_task(task, wait_on);	
      return result;
    }

    void ProcessorImpl::enqueue_task(TaskDesc *task, Event wait_on)
    {
      if (wait_on.exists()) {
        EventImpl *wait_impl = RuntimeImpl::get_runtime()->get_event_impl(wait_on);
        DeferredTask *waiter = new DeferredTask(this, task);
        wait_impl->add_waiter(wait_on.gen, waiter);
        return;
      }
      // Put it on the ready queue
      add_to_ready_queue(task);
    }

    void ProcessorImpl::add_to_ready_queue(TaskDesc *task)
    {
      if (task->capture_timeline)
        task->timeline.record_ready_time();
      ProcessorThread *to_wake = NULL;
      ProcessorThread *to_start = NULL;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      // Common case
      if (ready_queue.empty() || 
          (ready_queue.back()->priority >= task->priority))
        ready_queue.push_back(task);
      else
      {
        bool inserted = false;
        for (std::list<TaskDesc*>::iterator it = ready_queue.begin();
              it != ready_queue.end(); it++)
        {
          if ((*it)->priority < task->priority)
          {
            ready_queue.insert(it, task);
            inserted = true;
            break;
          }
        }
        // Technically we shouldn't need this but whatever
        if (!inserted)
          ready_queue.push_back(task);
      }
      // Figure out if we need to wake someone up
      if (running_thread == NULL) {
        if (!available_threads.empty()) {
          to_wake = available_threads.back();
          available_threads.pop_back();
          running_thread = to_wake;
        } else {
          to_start = new ProcessorThread(this, stack_size);
          running_thread = to_start;
        }
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      if (to_wake)
        to_wake->awake();
      if (to_start)
        to_start->start();
    }

    Processor ProcessorImpl::get_utility_processor(void) const
    {
      return utility;
    }

    bool ProcessorImpl::execute_task(ProcessorThread *thread)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      // Sanity check, we should be the running thread if we are in here
      assert(thread == running_thread);
      // First check to see if there are any resumable threads
      // If there are then we will switch onto those
      if (!resumable_threads.empty())
      {
        // Move this thread on to the available threads and wake
        // up one of the resumable threads
        thread->prepare_to_sleep();
        available_threads.push_back(thread);
        // Pull the first thread off the resumable threads
        ProcessorThread *to_resume = resumable_threads.front();
        resumable_threads.pop_front();
        // Make this the running thread
        running_thread = to_resume;
        // Release the lock
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Wake up the resumable thread
        to_resume->resume();
        // Put ourselves to sleep
        thread->sleep();
      }
      else if (ready_queue.empty())
      {
        // If there are no tasks to run, then we should go to sleep
        thread->prepare_to_sleep();
        available_threads.push_back(thread);
        running_thread = NULL;
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        thread->sleep();
      }
      else
      {
        // Pull a task off the queue and execute it
        TaskDesc *task = ready_queue.front();
        ready_queue.pop_front();
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        run_task(task);
      }
      // This value is monotonic so once it becomes true, then we should exit
      return shutdown;
    }

    void ProcessorImpl::run_task(TaskDesc *task)
    {
      int start_count = __sync_fetch_and_add(&(task->start_arrivals),1);
      if (start_count == 0)
      {
        if (task->func_id != 0)
        {
          RuntimeImpl::TaskTable::const_iterator it = 
                              task_table.find(task->func_id);
#ifdef DEBUG_REALM
          assert(it != task_table.end());
#endif
          Processor::TaskFuncPtr func = it->second.func_ptr;
          if (task->capture_timeline)
            task->timeline.record_start_time();
          if (task->capture_usage)
            task->usage.proc = proc;
          func(task->args, task->arglen,
	       it->second.userdata, it->second.userlen, proc);
          if (task->capture_timeline)
	  {
            task->timeline.record_end_time();
	    task->timeline.record_complete_time();
	  }
        } else {
          process_kill();
        }
        // Trigger the event indicating that the task has been run
        task->complete->trigger();
      }
      int expected_finish = task->expected;
      int finish_count = __sync_add_and_fetch(&(task->finish_arrivals),1);
      if (finish_count == expected_finish)
        delete task;
    }

    void ProcessorImpl::start_processor(void)
    {
      // Make a new thread and tell it to run the start-up routine
      assert(running_thread == NULL);
      running_thread = new ProcessorThread(this, stack_size);
      running_thread->do_initialize();
      running_thread->start();
    }

    void ProcessorImpl::shutdown_processor(void)
    {
      // First check to make sure that we received the kill
      // pill. If we didn't then wait for it. This is how
      // we distinguish deadlock from just normal termination
      // from all the processors being idle 
      std::vector<ProcessorThread*> to_shutdown;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (!shutdown_trigger)
        PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond, mutex));
      assert(shutdown_trigger);
      shutdown = true;
      to_shutdown = available_threads;
      if (running_thread != NULL)
        to_shutdown.push_back(running_thread);
      assert(resumable_threads.empty());
      assert(paused_threads.empty());
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      //printf("Processor " IDFMT " needed %ld threads\n", 
      //        proc.id, to_shutdown.size());
      // We can now read this outside the lock since we know
      // that the threads are all asleep and are all about to exit
      assert(!to_shutdown.empty());
      for (unsigned idx = 0; idx < to_shutdown.size(); idx++)
      {
        if (idx == 0)
          to_shutdown[idx]->do_finalize();
        to_shutdown[idx]->shutdown();
        delete to_shutdown[idx];
      }
    }

    void ProcessorImpl::initialize_processor(void)
    {
      //fprintf(stdout,"This is processor %d\n",proc.id);
      //fflush(stdout);
      // Check to see if there is an initialization task
      RuntimeImpl::TaskTable::const_iterator it = 
        task_table.find(Processor::TASK_ID_PROCESSOR_INIT);
      if (it != task_table.end())
      {	  
        Processor::TaskFuncPtr func = it->second.func_ptr;
        func(NULL, 0, NULL, 0, proc);
      }
      // Wait for all the processors to be ready to go
#ifndef __MACH__
      int bar_result = pthread_barrier_wait(init_bar);
      if (bar_result == PTHREAD_BARRIER_SERIAL_THREAD)
      {
        // Free the barrier
        PTHREAD_SAFE_CALL(pthread_barrier_destroy(init_bar));
        free(init_bar);
      }
#if DEBUG_REALM
      else
      {
        PTHREAD_SAFE_CALL(bar_result);
      }
#endif
#else // MAC OSX case
      bool delete_barrier = init_bar->arrive();
      if (delete_barrier)
        delete init_bar;
#endif
      init_bar = NULL;
    }

    void ProcessorImpl::finalize_processor(void)
    {
      // Check to see if there is a shutdown method
      RuntimeImpl::TaskTable::const_iterator it = 
        task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
      if (it != task_table.end())
      {	  
        Processor::TaskFuncPtr func = it->second.func_ptr;
        func(NULL, 0, NULL, 0, proc);
      }
    }
    
    void ProcessorImpl::process_kill(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      shutdown_trigger = true;
      PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    bool ProcessorImpl::is_idle(void)
    {
      // This processor is idle if all its threads are idle
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      bool result = (running_thread == NULL) && resumable_threads.empty()
                      && paused_threads.empty();
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    void ProcessorImpl::pause_thread(ProcessorThread *thread)
    {
      ProcessorThread *to_wake = NULL;
      ProcessorThread *to_start = NULL;
      ProcessorThread *to_resume = NULL;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      assert(running_thread == thread);
      // Put this on the list of paused threads
      paused_threads.insert(thread);
      // Now see if we have other work to do
      if (!resumable_threads.empty()) {
        to_resume = resumable_threads.front();
        resumable_threads.pop_front();
        running_thread = to_resume;
      } else if (!ready_queue.empty()) {
        // Note we might need to make a new thread here
        if (!available_threads.empty()) {
          to_wake = available_threads.back();
          available_threads.pop_back();
          running_thread = to_wake;
        } else {
          // Make a new thread to run
          to_start = new ProcessorThread(this, stack_size);
          running_thread = to_start;
        }
      } else {
        // Nothing else to do, so mark that no one is running
        running_thread = NULL;
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      // Wake up any threads while not holding the lock
      if (to_wake)
        to_wake->awake();
      if (to_start)
        to_start->start();
      if (to_resume)
        to_resume->resume();
    }

    void ProcessorImpl::resume_thread(ProcessorThread *thread)
    {
      bool resume_now = false; 
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      std::set<ProcessorThread*>::iterator finder = 
        paused_threads.find(thread);
      assert(finder != paused_threads.end());
      paused_threads.erase(finder);
      if (running_thread == NULL) {
        // No one else is running now, so resume the thread
        running_thread = thread;
        resume_now = true;
      } else {
        // Easy case, just add it to the list of resumable threads
        resumable_threads.push_back(thread);
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      if (resume_now)
        thread->resume();
    }

    ProcessorImpl::ProcessorThread::ProcessorThread(ProcessorImpl *p,
                                                    size_t stack_size)
      : proc(p), state(RUNNING_STATE), 
        initialize(false), finalize(false)
    {
      thread_mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
      thread_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
      PTHREAD_SAFE_CALL(pthread_mutex_init(thread_mutex,NULL));
      PTHREAD_SAFE_CALL(pthread_cond_init(thread_cond,NULL));
      PTHREAD_SAFE_CALL(pthread_attr_init(&attr));
      PTHREAD_SAFE_CALL(pthread_attr_setstacksize(&attr, stack_size));
    }

    ProcessorImpl::ProcessorThread::~ProcessorThread(void)
    {
      free(thread_mutex);
      free(thread_cond);
    }

    /*static*/ void* ProcessorImpl::ProcessorThread::entry(void *arg)
    {
      ProcessorThread *thread = (ProcessorThread*)arg; 
      PTHREAD_SAFE_CALL(pthread_setspecific(local_thread_key, thread));
      // Also set the value of thread timer key
      PTHREAD_SAFE_CALL( pthread_setspecific(thread_timer_key, NULL) );
      thread->run();
      return NULL;
    }

    void ProcessorImpl::ProcessorThread::run(void)
    {
      if (initialize)
        proc->initialize_processor();
      while (true)
      {
        assert(state == RUNNING_STATE);
        bool quit = proc->execute_task(this);
        if (quit) break;
      }
      if (finalize)
        proc->finalize_processor();
    }

    void ProcessorImpl::ProcessorThread::awake(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_mutex));
      assert((state == SLEEPING_STATE) || (state == SLEEP_STATE));
      // Only need to signal if the thread is actually asleep
      if (state == SLEEP_STATE)
        PTHREAD_SAFE_CALL(pthread_cond_signal(thread_cond));
      state = RUNNING_STATE;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_mutex));
    }

    void ProcessorImpl::ProcessorThread::sleep(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_mutex));
      assert((state == SLEEPING_STATE) || (state == RUNNING_STATE));
      // If we haven't been told to stay awake, then go to sleep
      if (state == SLEEPING_STATE) {
        state = SLEEP_STATE;
        PTHREAD_SAFE_CALL(pthread_cond_wait(thread_cond, thread_mutex));
      }
      assert(state == RUNNING_STATE);
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_mutex));
    }

    void ProcessorImpl::ProcessorThread::prepare_to_sleep(void)
    {
      // Don't need the lock since we are running
      assert(state == RUNNING_STATE);
      state = SLEEPING_STATE;
    }

    void ProcessorImpl::ProcessorThread::preempt(EventImpl *event,
                                                 EventImpl::EventGeneration needed)
    {
      // First set our state to paused 
      assert(state == RUNNING_STATE);
      // Mark that this thread is paused
      state = PAUSED_STATE;
      // Then tell the processor to pause the thread
      proc->pause_thread(this);
      // Now register ourselves with the event
      event->add_waiter(needed, this);
      // Take our lock and see if we are still in the paused state
      // It's possible we've already been woken up so check before
      // going to sleep
      PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_mutex));
      // If we are in the paused state or the resumable state
      // then we actually do need to go to sleep so we can be woken
      // up by the processor later
      if ((state == PAUSED_STATE) || (state == RESUMABLE_STATE))
      {
        PTHREAD_SAFE_CALL(pthread_cond_wait(thread_cond, thread_mutex));
      }
      assert(state == RUNNING_STATE);
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_mutex));
    }

    bool ProcessorImpl::ProcessorThread::event_triggered(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_mutex));
      assert(state == PAUSED_STATE);
      state = RESUMABLE_STATE;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_mutex));
      // Now tell the processor that this thread is resumable
      proc->resume_thread(this);
      return false;
    }

    void ProcessorImpl::ProcessorThread::resume(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_mutex));
      assert(state == RESUMABLE_STATE);
      state = RUNNING_STATE;
      PTHREAD_SAFE_CALL(pthread_cond_signal(thread_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_mutex));
    }

    void ProcessorImpl::ProcessorThread::shutdown(void)
    {
      // Wake up the thread
      PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_mutex));
      state = RUNNING_STATE;
      PTHREAD_SAFE_CALL(pthread_cond_signal(thread_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_mutex));
      // Now wait to join with the thread
      void *result;
      PTHREAD_SAFE_CALL(pthread_join(thread, &result));
    }

    void ProcessorImpl::ProcessorThread::start(void)
    {
      PTHREAD_SAFE_CALL(pthread_create(&thread, &attr,
                          ProcessorImpl::ProcessorThread::entry, (void*)this));
    }

    void ProcessorGroup::get_group_members(std::vector<Processor>& members)
    {
      for(std::vector<ProcessorImpl *>::const_iterator it = this->members.begin();
	  it != this->members.end();
	  it++)
	members.push_back((*it)->get_id());
    }

    Event ProcessorGroup::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on, int priority)
    {
      // Create a new task description and enqueue it for all the members
      TaskDesc *task = new TaskDesc(func_id, args, arglen,
                                    RuntimeImpl::get_runtime()->get_free_event(),
                                    priority, 0, 0, members.size());
      Event result = task->complete->get_event();

      for (std::vector<ProcessorImpl*>::const_iterator it = members.begin();
            it != members.end(); it++)
      {
        (*it)->enqueue_task(task, wait_on);
      }
      return result;
    }

    Event ProcessorGroup::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, const Realm::ProfilingRequestSet &reqs,
                                Event wait_on, int priority)
    {
      // Create a new task description and enqueue it for all the members
      TaskDesc *task = new TaskDesc(func_id, args, arglen,
                                    RuntimeImpl::get_runtime()->get_free_event(),
                                    priority, 0, 0, members.size(), reqs);
      Event result = task->complete->get_event();

      for (std::vector<ProcessorImpl*>::const_iterator it = members.begin();
            it != members.end(); it++)
      {
        (*it)->enqueue_task(task, wait_on);
      }
      return result;
    }

    ////////////////////////////////////////////////////////
    // Memory 
    ////////////////////////////////////////////////////////
    
    class MemoryImpl {
    public:
	MemoryImpl(size_t max, Memory::Kind k) 
		: max_size(max), remaining(max), kind(k)
	{
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
	}
        ~MemoryImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	size_t remaining_bytes(void);
	void* allocate_space(size_t size);
	void free_space(void *ptr, size_t size);
        size_t total_space(void) const;  
        Memory::Kind get_kind(void) const;
    private:
	const size_t max_size;
	size_t remaining;
	pthread_mutex_t *mutex;
        const Memory::Kind kind;
    };

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    const Memory Memory::NO_MEMORY = { 0 };

    Memory::Kind Memory::kind(void) const
    {
      return RuntimeImpl::get_runtime()->get_memory_impl(*this)->get_kind();
    }

    size_t Memory::capacity(void) const
    {
      return RuntimeImpl::get_runtime()->get_memory_impl(*this)->total_space();
    }

    AddressSpace Memory::address_space(void) const
    {
      return 0;
    }

    IDType Memory::local_id(void) const
    {
      return id;
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    size_t MemoryImpl::remaining_bytes(void) 
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	size_t result = remaining;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void* MemoryImpl::allocate_space(size_t size)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	void *ptr = NULL;
	if (size < remaining)
	{
		remaining -= size;
		ptr = malloc(size);
#ifdef DEBUG_REALM
		assert(ptr != NULL);
#endif
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return ptr;
    }

    void MemoryImpl::free_space(void *ptr, size_t size)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_REALM
	assert(ptr != NULL);
#endif
	remaining += size;
	free(ptr);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    size_t MemoryImpl::total_space(void) const
    {
      return max_size;
    }

    Memory::Kind MemoryImpl::get_kind(void) const
    {
      return kind;
    }

    ////////////////////////////////////////////////////////
    // Element Masks
    ////////////////////////////////////////////////////////

    struct ElementMaskImpl {
      //int count, offset;
      int dummy;
      unsigned bits[0];

      static size_t bytes_needed(int offset, int count)
      {
	size_t need = sizeof(ElementMaskImpl) + (((count + 31) >> 5) << 2);
	return need;
      }
	
    };

  };
};

namespace Realm {

    ElementMask::ElementMask(void)
      : first_element(-1), num_elements(-1), memory(Memory::NO_MEMORY), offset(-1),
	raw_data(0), first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
    }

    ElementMask::ElementMask(int _num_elements, int _first_element /*= 0*/)
      : first_element(_first_element), num_elements(_num_elements), memory(Memory::NO_MEMORY), offset(-1),
        first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = (char *)calloc(1, bytes_needed);
      //((ElementMaskImpl *)raw_data)->count = num_elements;
      //((ElementMaskImpl *)raw_data)->offset = first_element;
    }

    ElementMask::ElementMask(const ElementMask &copy_from, 
			     int _num_elements, int _first_element /*= -1*/)
    {
      first_element = (_first_element >= 0) ? _first_element : copy_from.first_element;
      num_elements = _num_elements;
      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      // if we have bounds, make sure they're trimmed to what we actually cover
      if((first_enabled_elmt >= 0) && (first_enabled_elmt < first_element)) {
	first_enabled_elmt = first_element;
      }
      if((last_enabled_elmt >= 0) && (last_enabled_elmt >= (first_element + num_elements))) {
	last_enabled_elmt = first_element + num_elements - 1;
      }
      // figure out the copy offset - must be an integral number of bytes
      ptrdiff_t copy_byte_offset = (first_element - copy_from.first_element);
      assert((copy_from.first_element + (copy_byte_offset << 3)) == first_element);

      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = (char *)calloc(1, bytes_needed);  // sets initial values to 0

      // how much to copy?
      size_t bytes_avail = (ElementMaskImpl::bytes_needed(copy_from.first_element, 
							  copy_from.num_elements) -
			    copy_byte_offset);
      size_t bytes_to_copy = (bytes_needed <= bytes_avail) ? bytes_needed : bytes_avail;

      if(copy_from.raw_data) {
	if(copy_byte_offset >= 0) {
	  memcpy(raw_data, copy_from.raw_data + copy_byte_offset, bytes_to_copy);
	} else {
	  // we start before the input mask, so offset is applied to our pointer
	  memcpy(raw_data + (-copy_byte_offset), copy_from.raw_data, bytes_to_copy);
	}
      } else {
        assert(false);
      }
    }

    ElementMask::ElementMask(const ElementMask &copy_from, bool trim /*= false*/)
    {
      first_element = copy_from.first_element;
      num_elements = copy_from.num_elements;
      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      ptrdiff_t copy_byte_offset = 0;
      if(trim) {
	// trimming from the end is easy - just reduce num_elements
	if(last_enabled_elmt >= 0) {
	  assert(last_enabled_elmt < (first_element + num_elements));
	  num_elements = last_enabled_elmt + 1 - first_element;
	}

	// trimming from the beginning requires stepping by units of 8 so that we can copy bytes
	if(first_enabled_elmt > first_element) {
	  assert(first_enabled_elmt < (first_element + num_elements));
	  copy_byte_offset = (first_enabled_elmt - first_element) >> 3;  // truncates
	  first_element += (copy_byte_offset << 3); // convert back to bits
	  num_elements -= (copy_byte_offset << 3);
	}
      }
      assert(num_elements >= 0);
	
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = (char *)calloc(1, bytes_needed);

      if(copy_from.raw_data) {
	memcpy(raw_data, copy_from.raw_data + copy_byte_offset, bytes_needed);
      } else {
	assert(false);
      }
    }

    ElementMask::~ElementMask(void)
    {
      if (raw_data != 0)
      {
        free(raw_data);
        raw_data = NULL;
      }
    }

    ElementMask& ElementMask::operator=(const ElementMask &rhs)
    {
      first_element = rhs.first_element;
      num_elements = rhs.num_elements;
      first_enabled_elmt = rhs.first_enabled_elmt;
      last_enabled_elmt = rhs.last_enabled_elmt;
      size_t bytes_needed = rhs.raw_size();
      if (raw_data)
        free(raw_data);
      raw_data = (char *)calloc(1, bytes_needed);
      if (rhs.raw_data)
      {
        memcpy(raw_data, rhs.raw_data, bytes_needed);
      }
      else
      {
        assert(false);
      }
      return *this;
    }

    void ElementMask::enable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("ENABLE %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
	int pos = start - first_element;
	assert(pos < num_elements);
	for(int i = 0; i < count; i++) {
	  unsigned *ptr = &(impl->bits[pos >> 5]);
	  *ptr |= (1U << (pos & 0x1f));
	  pos++;
	}
	//printf("ENABLED %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
      } else {
	assert(0);
      }
      if((first_enabled_elmt < 0) || (start < first_enabled_elmt))
	first_enabled_elmt = start;

      if((last_enabled_elmt < 0) || ((start+count-1) > last_enabled_elmt))
	last_enabled_elmt = start + count - 1;
    }

    void ElementMask::disable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned *ptr = &(impl->bits[pos >> 5]);
	  *ptr &= ~(1U << (pos & 0x1f));
	  pos++;
	}
      } else {
	assert(0);
      }
      // if the first_enabled_elmt was in this range, then disable it
      if ((start <= first_enabled_elmt) && (first_enabled_elmt < (start+count)))
        first_enabled_elmt = -1;
    }

    int ElementMask::find_enabled(int count /*= 1 */, int start /*= 0*/) const
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("FIND_ENABLED %p %d %d %x\n", raw_data, first_element, count, impl->bits[0]);
	if(start < first_enabled_elmt)
	  start = first_enabled_elmt;
	for(int pos = start; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    unsigned bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	assert(0);
      }
      return -1;
    }

    int ElementMask::find_disabled(int count /*= 1 */, int start /*= 0*/) const
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	if(start < first_enabled_elmt)
	  start = first_enabled_elmt;
	for(int pos = start; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    unsigned bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	    if(bit != 0) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	assert(0);
      }
      return -1;
    }

    bool ElementMask::is_set(int ptr) const
    {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        unsigned bit = ((impl->bits[ptr >> 5] >> (ptr & 0x1f))) & 1;
        return (bit == 1);
    }

    size_t ElementMask::pop_count(bool enabled) const
    {
      size_t count = 0;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        const int max_full = (num_elements >> 5);
        bool remainder = (num_elements % 32) != 0;
        for (int index = 0; index < max_full; index++)
          count += __builtin_popcount(impl->bits[index]);
        if (remainder)
          count += __builtin_popcount(impl->bits[max_full]);
        if (!enabled)
          count = num_elements - count;
      } else {
        assert(0);
      }
      return count;
    }

    bool ElementMask::operator!(void) const
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++) {
          if (impl->bits[index])
            return false;
        }
      } else {
        assert(false);
      }
      return true;
    }

    bool ElementMask::operator==(const ElementMask &other) const
    {
      if (num_elements != other.num_elements)
        return false;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        assert(other.raw_data != 0);
        ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++)
        {
          if (impl->bits[index] != other_impl->bits[index])
            return false;
        }
      } else {
        assert(false);
      }
      return true;
    }

    bool ElementMask::operator!=(const ElementMask &other) const
    {
      return !((*this) == other);
    }

    ElementMask ElementMask::operator|(const ElementMask &other) const
    {
      ElementMask result(num_elements); 
      ElementMaskImpl *target = (ElementMaskImpl *)result.raw_data;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        assert(other.raw_data != 0);
        ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
        assert(num_elements == other.num_elements);
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++)
        {
          target->bits[index] = impl->bits[index] | other_impl->bits[index];
        }
      } else {
        assert(false);
      }
      return result;
    }

    ElementMask ElementMask::operator&(const ElementMask &other) const
    {
      ElementMask result(num_elements);
      ElementMaskImpl *target = (ElementMaskImpl *)result.raw_data;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        assert(other.raw_data != 0);
        ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
        assert(num_elements == other.num_elements);
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++)
        {
          target->bits[index] = impl->bits[index] & other_impl->bits[index];
        }
      } else {
        assert(false);
      }
      return result;
    }

    ElementMask ElementMask::operator-(const ElementMask &other) const
    {
      ElementMask result(num_elements);
      ElementMaskImpl *target = (ElementMaskImpl *)result.raw_data;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        assert(other.raw_data != 0);
        ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
        assert(num_elements == other.num_elements);
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++)
        {
          target->bits[index] = impl->bits[index] & ~(other_impl->bits[index]);  
        }
      } else {
        assert(false);
      }
      return result;
    }

    ElementMask& ElementMask::operator|=(const ElementMask &other)
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        assert(other.raw_data != 0);
        ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
        assert(num_elements == other.num_elements);
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++)
        {
          impl->bits[index] |= other_impl->bits[index];
        }
      } else {
        assert(false);
      }
      return *this;
    }

    ElementMask& ElementMask::operator&=(const ElementMask &other)
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        assert(other.raw_data != 0);
        ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
        assert(num_elements == other.num_elements);
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++)
        {
          impl->bits[index] &= other_impl->bits[index];
        }
      } else {
        assert(false);
      }
      return *this;
    }

    ElementMask& ElementMask::operator-=(const ElementMask &other)
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        assert(other.raw_data != 0);
        ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
        assert(num_elements == other.num_elements);
        const int max_full = ((num_elements+31) >> 5);
        for (int index = 0; index < max_full; index++)
        {
          impl->bits[index] &= ~(other_impl->bits[index]);
        }
      } else {
        assert(false);
      }
      return *this;
    }

    size_t ElementMask::raw_size(void) const
    {
      return ElementMaskImpl::bytes_needed(offset,num_elements);
    }

    const void *ElementMask::get_raw(void) const
    {
      return raw_data;
    }

    void ElementMask::set_raw(const void *data)
    {
      assert(0);
    }

    ElementMask::OverlapResult ElementMask::overlaps_with(const ElementMask &other,
                                             off_t max_effort) const
    {
#ifdef DEBUG_REALM
      assert(raw_size() == other.raw_size());
#endif
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *) raw_data;
        const void *other_raw = other.get_raw();
        if (other_raw == 0)
          assert(false);
        ElementMaskImpl *other_impl = (ElementMaskImpl *) other_raw;
        int max_full = (num_elements >> 5); 
        for (int index = 0; index < max_full; index++)
        {
          if (impl->bits[index] & other_impl->bits[index])
            return OVERLAP_YES;
        }
        if (((num_elements % 32) != 0) &&
            (impl->bits[max_full] & other_impl->bits[max_full]))
          return OVERLAP_YES;
      } else {
        assert(false);
      }
      return OVERLAP_NO;
    }

    ElementMask::Enumerator *ElementMask::enumerate_enabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 1);
    }

    ElementMask::Enumerator *ElementMask::enumerate_disabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 0);
    }

    ElementMask::Enumerator::Enumerator(const ElementMask& _mask, int _start, int _polarity)
      : mask(_mask), pos(_start), polarity(_polarity) {}

    ElementMask::Enumerator::~Enumerator(void) {}

    bool ElementMask::Enumerator::get_next(int &position, int &length)
    {
      if(mask.raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)(mask.raw_data);

	// scan until we find a bit set with the right polarity
	while(pos < mask.num_elements) {
	  int bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	  if(bit != polarity) {
	    pos++;
	    continue;
	  }

	  // ok, found one bit with the right polarity - now see how many
	  //  we have in a row
	  position = pos++;
	  while(pos < mask.num_elements) {
	    int bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	    if(bit != polarity) break;
            pos++;
	  }
	  // we get here either because we found the end of the run or we 
	  //  hit the end of the mask
	  length = pos - position;
	  return true;
	}

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      } else {
	assert(0);

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      }
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    ////////////////////////////////////////////////////////
    // CopyOperation (Declaration Only) 
    ////////////////////////////////////////////////////////

    class CopyOperation : public DMAOperation {
    public:
      CopyOperation(const std::vector<Domain::CopySrcDstField>& _srcs,
                    const std::vector<Domain::CopySrcDstField>& _dsts,
                    const Domain _domain,
                    ReductionOpID _redop_id, bool _red_fold,
                    EventImpl *_done_event)
        : DMAOperation(), srcs(_srcs), dsts(_dsts), domain(_domain),
          redop_id(_redop_id), red_fold(_red_fold),
          done_event(_done_event)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));    
        // If we don't have a done event, make one
        if (!done_event)
          done_event = RuntimeImpl::get_runtime()->get_free_event();
      }

      CopyOperation(const std::vector<Domain::CopySrcDstField>& _srcs,
                    const std::vector<Domain::CopySrcDstField>& _dsts,
                    const Realm::ProfilingRequestSet &reqs,
                    const Domain _domain, ReductionOpID _redop_id, bool _red_fold,
                    EventImpl *_done_event)
        : DMAOperation(reqs), srcs(_srcs), dsts(_dsts), domain(_domain),
          redop_id(_redop_id), red_fold(_red_fold),
          done_event(_done_event)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));    
        // If we don't have a done event, make one
        if (!done_event)
          done_event = RuntimeImpl::get_runtime()->get_free_event();
      }

      virtual ~CopyOperation(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_destroy(&mutex));
      }

      virtual void perform(void);

      virtual void print_info(FILE *f) {
        Event e = done_event->get_event();
        fprintf(f,"deferred copy: after=" IDFMT "/%d\n",
                    e.id, e.gen+1);
      }

      Event register_copy(Event wait_on);

    protected:
      std::vector<Domain::CopySrcDstField> srcs;
      std::vector<Domain::CopySrcDstField> dsts;
      Domain domain;
      ReductionOpID redop_id;
      bool red_fold;
      EventImpl *done_event;
      pthread_mutex_t mutex; 
    };

    class FillOperation : public DMAOperation {
    public:
      FillOperation(const std::vector<Domain::CopySrcDstField> &_dsts,
                    const void *value, size_t size, 
                    const Domain &dom, EventImpl *_done_event)
        : dsts(_dsts), fill_value_size(size), domain(dom), done_event(_done_event)
      {
        fill_value = malloc(fill_value_size);
        memcpy(fill_value, value, fill_value_size);
      }
      FillOperation(const std::vector<Domain::CopySrcDstField> &_dsts,
                    const Realm::ProfilingRequestSet &reqs,
                    const void *value, size_t size, 
                    const Domain &dom, EventImpl *_done_event)
        : DMAOperation(reqs), dsts(_dsts), fill_value_size(size), 
          domain(dom), done_event(_done_event)
      {
        fill_value = malloc(fill_value_size);
        memcpy(fill_value, value, fill_value_size);
      }
      virtual ~FillOperation(void)
      {
        free(fill_value);
      }

      virtual void perform(void);

      virtual void print_info(FILE *f) {
        Event e = done_event->get_event();
        fprintf(f,"deferred fill: after=" IDFMT "/%d\n",
                e.id, e.gen+1);
      }
    protected:
      std::vector<Domain::CopySrcDstField> dsts;
      void *fill_value;
      size_t fill_value_size;
      Domain domain;
      EventImpl *done_event;
    };

    class ComputeIndexSpaces : public DMAOperation {
    public:
      ComputeIndexSpaces(const std::vector<IndexSpace::BinaryOpDescriptor> &p,
                         EventImpl *d)
        : pairs(p), done_event(d) { }
      ComputeIndexSpaces(const std::vector<IndexSpace::BinaryOpDescriptor> &p,
                         const Realm::ProfilingRequestSet &reqs,
                         EventImpl *d)
        : DMAOperation(reqs), pairs(p), done_event(d) { }
      virtual ~ComputeIndexSpaces(void) { }
    public:
      virtual void perform(void);  
      virtual void print_info(FILE *f);
    protected:
      std::vector<IndexSpace::BinaryOpDescriptor> pairs;
      EventImpl *done_event;
    };

    class ReduceIndexSpaces : public DMAOperation {
    public:
      ReduceIndexSpaces(IndexSpace::IndexSpaceOperation o,
                        const std::vector<IndexSpace> &s,
                        IndexSpaceImpl *r,
                        EventImpl *d)
        : op(o), spaces(s), result(r), done_event(d) { }
      ReduceIndexSpaces(IndexSpace::IndexSpaceOperation o,
                        const std::vector<IndexSpace> &s,
                        const Realm::ProfilingRequestSet &reqs,
                        IndexSpaceImpl *r,
                        EventImpl *d)
        : DMAOperation(reqs), op(o), spaces(s), result(r), done_event(d) { }
      virtual ~ReduceIndexSpaces(void) { }
    public:
      virtual void perform(void);
      virtual void print_info(FILE *f);
    protected:
      IndexSpace::IndexSpaceOperation op;
      std::vector<IndexSpace> spaces;
      IndexSpaceImpl *result;
      EventImpl *done_event;
    };

    class DeferredEqualSpaces : public DMAOperation {
    public:
      DeferredEqualSpaces(const std::vector<IndexSpaceImpl*> &subs,
                          IndexSpaceImpl *t, size_t g, EventImpl *d)
        : target(t), subspaces(subs), granularity(g), done_event(d) { }
      DeferredEqualSpaces(const std::vector<IndexSpaceImpl*> &subs,
                          const Realm::ProfilingRequestSet &reqs,
                          IndexSpaceImpl *t, size_t g, EventImpl *d)
        : DMAOperation(reqs), target(t), subspaces(subs), granularity(g), done_event(d) { }
      virtual ~DeferredEqualSpaces(void) { }
    public:
      virtual void perform(void);
      virtual void print_info(FILE *f);
    protected:
      IndexSpaceImpl *target;
      std::vector<IndexSpaceImpl*> subspaces;
      size_t granularity;
      EventImpl *done_event;
    };

    class DeferredWeightedSpaces : public DMAOperation {
    public:
      DeferredWeightedSpaces(const std::vector<IndexSpaceImpl*> &subs,
                             IndexSpaceImpl *t, size_t g, EventImpl *d,
                             const std::vector<int> &w)
        : target(t), subspaces(subs), granularity(g), done_event(d), weights(w) { }
      DeferredWeightedSpaces(const std::vector<IndexSpaceImpl*> &subs,
                             const Realm::ProfilingRequestSet &reqs,
                             IndexSpaceImpl *t, size_t g, EventImpl *d,
                             const std::vector<int> &w)
        : DMAOperation(reqs), target(t), subspaces(subs), granularity(g), 
          done_event(d), weights(w) { }
      virtual ~DeferredWeightedSpaces(void) { }
    public:
      virtual void perform(void);
      virtual void print_info(FILE *f);
    protected:
      IndexSpaceImpl *target;
      std::vector<IndexSpaceImpl*> subspaces;
      size_t granularity;
      EventImpl *done_event;
      std::vector<int> weights;
    };

    class DeferredFieldSpaces : public DMAOperation {
    public:
      DeferredFieldSpaces(const std::vector<IndexSpace::FieldDataDescriptor> &f,
                          const std::map<DomainPoint,IndexSpace> &s,
                          EventImpl *d, IndexSpaceImpl *t)
        : target(t), field_data(f), subspaces(s), done_event(d) { }
      DeferredFieldSpaces(const std::vector<IndexSpace::FieldDataDescriptor> &f,
                          const std::map<DomainPoint,IndexSpace> &s,
                          const Realm::ProfilingRequestSet &reqs,
                          EventImpl *d, IndexSpaceImpl *t)
        : DMAOperation(reqs), target(t), field_data(f), subspaces(s), done_event(d) { }
      virtual ~DeferredFieldSpaces(void) { }
    public:
      virtual void perform(void);
      virtual void print_info(FILE *f);
    protected:
      IndexSpaceImpl *target;
      std::vector<IndexSpace::FieldDataDescriptor> field_data;
      std::map<DomainPoint,IndexSpace> subspaces;
      EventImpl *done_event;
    };

    class DeferredImageSpaces : public DMAOperation {
    public:
      DeferredImageSpaces(const std::vector<IndexSpace::FieldDataDescriptor> &f,
                          const std::map<IndexSpace,IndexSpace> &s,
                          EventImpl *d, IndexSpaceImpl *t)
        : target(t), field_data(f), subspaces(s), done_event(d) { }
      DeferredImageSpaces(const std::vector<IndexSpace::FieldDataDescriptor> &f,
                          const std::map<IndexSpace,IndexSpace> &s,
                          const Realm::ProfilingRequestSet &reqs,
                          EventImpl *d, IndexSpaceImpl *t)
        : DMAOperation(reqs), target(t), field_data(f), subspaces(s), done_event(d) { }
      virtual ~DeferredImageSpaces(void) { }
    public:
      virtual void perform(void);
      virtual void print_info(FILE *f);
    protected:
      IndexSpaceImpl *target;
      std::vector<IndexSpace::FieldDataDescriptor> field_data;
      std::map<IndexSpace,IndexSpace> subspaces;
      EventImpl *done_event;
    };

    class DeferredPreimageSpaces : public DMAOperation {
    public:
      DeferredPreimageSpaces(const std::vector<IndexSpace::FieldDataDescriptor> &f,
                             const std::map<IndexSpace,IndexSpace> &s,
                             EventImpl *d, IndexSpaceImpl *t)
        : target(t), field_data(f), subspaces(s), done_event(d) { }
      DeferredPreimageSpaces(const std::vector<IndexSpace::FieldDataDescriptor> &f,
                             const std::map<IndexSpace,IndexSpace> &s,
                             const Realm::ProfilingRequestSet &reqs,
                             EventImpl *d, IndexSpaceImpl *t)
        : DMAOperation(reqs), target(t), field_data(f), subspaces(s), done_event(d) { }
      virtual ~DeferredPreimageSpaces(void) { }
    public:
      virtual void perform(void);
      virtual void print_info(FILE *f);
    protected:
      IndexSpaceImpl *target;
      std::vector<IndexSpace::FieldDataDescriptor> field_data;
      std::map<IndexSpace,IndexSpace> subspaces;
      EventImpl *done_event;
    };

    ////////////////////////////////////////////////////////
    // IndexSpaceImpl (Declaration Only) 
    ////////////////////////////////////////////////////////

    class IndexSpaceImpl {
    public:
      
    public:
	IndexSpaceImpl(int idx, size_t num, bool activate = false) {
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		index = idx;
		if (activate)
		{
			num_elmts = num;
			reservation = RuntimeImpl::get_runtime()->get_free_reservation();
                        mask = ElementMask(num_elmts);
                        parent = NULL;
		}
	}

        IndexSpaceImpl(int idx, IndexSpaceImpl *par, const ElementMask &m, bool activate = false) {
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
                PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		index = idx;
		if (activate)
		{
			num_elmts = m.get_num_elmts();
	                // Since we have a parent, use the parent's master allocator	
			reservation = RuntimeImpl::get_runtime()->get_free_reservation();
                        mask = m;
                        parent = par;
		}
        }

        ~IndexSpaceImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	bool activate(size_t num_elmts);
        bool activate(const ElementMask &m);
        bool activate(IndexSpaceImpl *par);
        bool activate(IndexSpaceImpl *par, const ElementMask &m);
	void deactivate(void);	
	IndexSpace get_metadata(void);

        IndexSpaceAllocatorImpl *create_allocator(void);

        static RegionInstance create_instance(Memory m, 
				       const std::vector<size_t>& field_sizes,
				       size_t block_size, 
				       const DomainLinearization& dl,
				       size_t num_elements,
                                       const Realm::ProfilingRequestSet &reqs,
				       ReductionOpID redopid = 0);

	void destroy_instance(RegionInstance i);

	Reservation get_reservation(void);

        ElementMask& get_element_mask(void);
        const ElementMask& get_element_mask(void) const;

        size_t get_num_elmts(void) const { return num_elmts; }

        void create_equal_subspaces(const std::vector<IndexSpaceImpl*> &subspaces,
                                    size_t granularity);
        void create_weighted_subspaces(const std::vector<IndexSpaceImpl*> &subspaces,
                                    size_t granularity,
                                    std::vector<int> &weights);

        void create_subspaces_by_field(const std::vector<IndexSpace::FieldDataDescriptor> &field_data,
                                       const std::map<DomainPoint,IndexSpace> &subspaces);
        void create_subspaces_by_image(const std::vector<IndexSpace::FieldDataDescriptor> &field_data,
                                       const std::map<IndexSpace,IndexSpace> &subspaces);
        void create_subspaces_by_preimage(const std::vector<IndexSpace::FieldDataDescriptor> &field_data,
                                          const std::map<IndexSpace,IndexSpace> &subspaces);

        static Event fill(const std::vector<Domain::CopySrcDstField> &dsts,
                          const void *fill_value, size_t fill_value_size,
                          Event wait_on, const Domain &domain);

        static Event fill(const std::vector<Domain::CopySrcDstField> &dsts,
                          const Realm::ProfilingRequestSet &reqs,
                          const void *fill_value, size_t fill_value_size,
                          Event wait_on, const Domain &domain);

        static Event copy(RegionInstance src_inst, RegionInstance dst_inst,
		   size_t elem_size, const Domain domain, Event wait_on = Event::NO_EVENT,
		   ReductionOpID redop_id = 0, bool red_fold = false);

        static Event copy(const std::vector<Domain::CopySrcDstField>& srcs,
		   const std::vector<Domain::CopySrcDstField>& dsts,
		   const Domain domain,
		   Event wait_on,
		   ReductionOpID redop_id = 0, bool red_fold = false);
        static Event copy(const std::vector<Domain::CopySrcDstField>& srcs,
		   const std::vector<Domain::CopySrcDstField>& dsts,
                   const Realm::ProfilingRequestSet &requests,
		   const Domain domain, Event wait_on,
		   ReductionOpID redop_id = 0, bool red_fold = false);
    public:
        // Traverse up the tree to the parent region that owns the master allocator
        // Peform the operation and then update the element mask on the way back down
        unsigned allocate_space(unsigned count);
        void     free_space(unsigned ptr, unsigned count);
    private:
	ReservationImpl *reservation;
	pthread_mutex_t *mutex;
	bool active;
	int index;
	size_t num_elmts;
        ElementMask mask;
        IndexSpaceImpl *parent;
    };

    
    ////////////////////////////////////////////////////////
    // Region Allocator 
    ////////////////////////////////////////////////////////

    class IndexSpaceAllocatorImpl {
    public:
      IndexSpaceAllocatorImpl(IndexSpace is)
        : is_impl(RuntimeImpl::get_runtime()->get_metadata_impl(is))
      {
	mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
      }
      
      ~IndexSpaceAllocatorImpl(void)
      {
	PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
	::free(mutex);
      }

    public:
      unsigned alloc_elmt(size_t num_elmts = 1);
      void free_elmt(unsigned ptr, unsigned count);
      bool activate(IndexSpaceImpl *owner);
      void deactivate();
      //IndexSpaceAllocator get_allocator(void) const;

    private:
      IndexSpaceImpl *is_impl;
      pthread_mutex_t *mutex;
    }; 

  };
};

namespace Realm {

    unsigned IndexSpaceAllocator::alloc(unsigned count /*= 1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return ((IndexSpaceAllocatorImpl *)impl)->alloc_elmt(count);
    }

    void IndexSpaceAllocator::free(unsigned ptr, unsigned count /*= 1 */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ((IndexSpaceAllocatorImpl *)impl)->free_elmt(ptr, count);
    }

    void IndexSpaceAllocator::destroy(void)
    {
      if (impl != NULL)
      {
        delete ((IndexSpaceAllocatorImpl *)impl);
        // Avoid double frees
        impl = NULL;
      }
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    unsigned IndexSpaceAllocatorImpl::alloc_elmt(size_t num_elmts)
    {
      // No need to hold the lock since we're just reading
      return is_impl->allocate_space(num_elmts);
    }

    void IndexSpaceAllocatorImpl::free_elmt(unsigned ptr, unsigned count)
    {
      // No need to hold the lock since we're just reading
      is_impl->free_space(ptr,count);
    }
    
    ////////////////////////////////////////////////////////
    // Region Instance 
    ////////////////////////////////////////////////////////

    class RegionInstanceImpl { 
    public:
        RegionInstanceImpl(int idx, Memory m, size_t num, size_t alloc, 
	     const std::vector<size_t>& _field_sizes,
	     size_t elem_size, size_t _block_size,
	     const DomainLinearization& _dl,
	     bool activate = false, char *base = NULL, const ReductionOpUntyped *op = NULL,
	     RegionInstanceImpl *parent = NULL)
	  : elmt_size(elem_size), num_elmts(num), allocation_size(alloc), 
            field_sizes(_field_sizes), block_size(_block_size), linearization(_dl),
	    reduction((op!=NULL)), list((parent!=NULL)), redop(op), 
            parent_impl(parent), cur_entry(0), index(idx)
	{
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		if (active)
		{
			memory = m;
			// Use the memory to allocate the space, fail if there is none
			//MemoryImpl *mem = RuntimeImpl::get_runtime()->get_memory_impl(m);
			base_ptr = base; //(char*)mem->allocate_space(num_elmts*elem_size);	
#ifdef DEBUG_REALM
			assert(base_ptr != NULL);
#endif
			reservation = RuntimeImpl::get_runtime()->get_free_reservation();
		}
	}

        ~RegionInstanceImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	const void* read(unsigned ptr);
	void write(unsigned ptr, const void* newval);	
        bool activate(Memory m, size_t num_elmts, size_t alloc,
		      const std::vector<size_t>& _field_sizes, size_t elem_size, size_t _block_size,
		      const DomainLinearization& _dl,
                      char *base, const ReductionOpUntyped *op, RegionInstanceImpl *parent,
                      const Realm::ProfilingRequestSet &reqs);
	void deactivate(void);
	RegionInstance get_instance(void) const;
        Reservation get_reservation(void);
        void perform_copy_operation(RegionInstanceImpl *target, const ElementMask &src_mask, const ElementMask &dst_mask);
        void apply_list(RegionInstanceImpl *target);
        void append_list(RegionInstanceImpl *target);
        void verify_access(unsigned ptr);
        bool is_reduction(void) const { return reduction; }
        bool is_list_reduction(void) const { return list; }
        void* get_base_ptr(void) const { return base_ptr; }
        void* get_address(int index, size_t field_start, size_t field_Size, size_t within_field);
        size_t get_elmt_size(void) const { return elmt_size; }
        const std::vector<size_t>& get_field_sizes(void) const { return field_sizes; }
        size_t get_num_elmts(void) const { return num_elmts; }
        size_t get_block_size(void) const { return block_size; }
        size_t* get_cur_entry(void) { return &cur_entry; }
        const DomainLinearization& get_linearization(void) const { return linearization; }
        void fill_field(unsigned offset, unsigned size, const void *fill_value,
                        size_t fill_value_size, const Domain &domain);
        inline Memory get_location(void) const { return memory; }
    private:
	char *base_ptr;	
	size_t elmt_size;
	size_t num_elmts;
        size_t allocation_size;
        std::vector<size_t> field_sizes;
        size_t block_size;
        DomainLinearization linearization;
	Memory memory;
	pthread_mutex_t *mutex;
        bool reduction; // reduction fold
        bool list; // reduction list
        const ReductionOpUntyped *redop; // for all reductions
        RegionInstanceImpl *parent_impl; // for lists
        size_t cur_entry; // for lists
	bool active;
	const int index;
	// Fields for the copy operation
	ReservationImpl *reservation;
        Realm::ProfilingRequestSet requests;
        Realm::ProfilingMeasurementCollection measurements;
        Realm::ProfilingMeasurements::InstanceTimeline timeline;
        bool capture_timeline;
    };

    class DeferredInstDestroy : public EventWaiter {
    public:
      DeferredInstDestroy(RegionInstanceImpl *i) : impl(i) { }
      virtual ~DeferredInstDestroy(void) { }
    public:
      virtual bool event_triggered(void) {
        impl->deactivate();
        return true;
      }
      virtual void print_info(FILE *f) {
        fprintf(f,"deferred instance destroy");
      }
    private:
      RegionInstanceImpl *impl;
    };

  };
};

namespace Realm {

    /*static*/ const RegionInstance RegionInstance::NO_INST = { 0 };

    RegionAccessor<AccessorType::Generic> RegionInstance::get_accessor(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstanceImpl *impl = RuntimeImpl::get_runtime()->get_instance_impl(*this);
      return RegionAccessor<AccessorType::Generic>(AccessorType::Generic::Untyped(impl));
    }

    AddressSpace RegionInstance::address_space(void) const 
    {
      return 0;
    }

    IDType RegionInstance::local_id(void) const
    {
      return id;
    }

    Memory RegionInstance::get_location(void) const
    {
      RegionInstanceImpl *impl = RuntimeImpl::get_runtime()->get_instance_impl(*this);
      return impl->get_location();
    }

    void RegionInstance::destroy(Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstanceImpl *impl = RuntimeImpl::get_runtime()->get_instance_impl(*this);
      if (!wait_on.has_triggered())
      {
        EventImpl *wait_impl = RuntimeImpl::get_runtime()->get_event_impl(wait_on);
        DeferredInstDestroy *waiter = new DeferredInstDestroy(impl);
        wait_impl->add_waiter(wait_on.gen, waiter);
        return;
      }
      impl->deactivate();
    }

    void RegionInstance::destroy(const std::vector<DestroyedField>& destroyed_fields,
				 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: actually call destructor
      assert(destroyed_fields.empty());
      destroy(wait_on);
    }
};

namespace LegionRuntime {
  namespace LowLevel {

#ifdef OLD_INTFC
    Event RegionInstance::copy_to_untyped(RegionInstance target, Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return RuntimeImpl::get_runtime()->get_instance_impl(*this)->copy_to(target,wait_on);
    }

    Event RegionInstance::copy_to_untyped(RegionInstance target, const ElementMask &mask,
                                                Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return RuntimeImpl::get_runtime()->get_instance_impl(*this)->copy_to(target,mask,wait_on);
    }

    Event RegionInstance::copy_to_untyped(RegionInstance target, IndexSpace region,
                                                 Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return RuntimeImpl::get_runtime()->get_instance_impl(*this)->copy_to(target,region,wait_on);
    }
#endif

    const void* RegionInstanceImpl::read(unsigned ptr)
    {
      // 'ptr' has already been multiplied by elmt_size
      return ((void*)(base_ptr + ptr));
    }

    void RegionInstanceImpl::write(unsigned ptr, const void* newval)
    {
      // 'ptr' has already been multiplied by elmt_size
      memcpy((base_ptr + ptr),newval,elmt_size);
    }

    bool RegionInstanceImpl::activate(Memory m, size_t num, size_t alloc, 
					const std::vector<size_t>& _field_sizes,
					size_t elem_size, size_t _block_size,
					const DomainLinearization& _dl,
					char *base, const ReductionOpUntyped *op, 
                                        RegionInstanceImpl *parent,
                                        const Realm::ProfilingRequestSet &reqs)
    {
	bool result = false;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{
		active = true;
		result = true;
		memory = m;
		num_elmts = num;
                allocation_size = alloc;
		field_sizes = _field_sizes;
		elmt_size = elem_size;
		block_size = _block_size;
		linearization = _dl;
		//MemoryImpl *mem = RuntimeImpl::get_runtime()->get_memory_impl(m);
		base_ptr = base; //(char*)mem->allocate_space(num_elmts*elmt_size);
                redop = op;
                reduction = (redop != NULL);
                parent_impl = parent;
                list = (parent != NULL);
                cur_entry = 0;
#ifdef DEBUG_REALM
		assert(base_ptr != NULL);
#endif
		reservation = RuntimeImpl::get_runtime()->get_free_reservation();
                if (!reqs.empty()) {
                    requests = reqs;
                    measurements.import_requests(requests);
                    if (measurements.wants_measurement<
                                      Realm::ProfilingMeasurements::InstanceTimeline>()) {
                      capture_timeline = true;
                      timeline.instance.id = index;
                      timeline.record_create_time();
                    } else {
                      capture_timeline = false;
                    }
                    if (measurements.wants_measurement<
                                      Realm::ProfilingMeasurements::InstanceMemoryUsage>()) {
                      Realm::ProfilingMeasurements::InstanceMemoryUsage usage;
                      usage.instance.id = index;
                      usage.memory = memory;
                      usage.bytes = allocation_size;
                      measurements.add_measurement(usage);
                    }
                } else {
                  capture_timeline = false;
                }
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void RegionInstanceImpl::deactivate(void)
    {
        if (!requests.empty()) {
          if (capture_timeline) {
            timeline.record_delete_time();
            measurements.add_measurement(timeline);
          }
          measurements.send_responses(requests);
        }
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;
	MemoryImpl *mem = RuntimeImpl::get_runtime()->get_memory_impl(memory);
	mem->free_space(base_ptr,allocation_size);
        allocation_size = 0;
	num_elmts = 0;
	field_sizes.clear();
	elmt_size = 0;
	block_size = 0;
	base_ptr = NULL;	
        redop = NULL;
        reduction = false;
        parent_impl = NULL;
        list = false;
	reservation->deactivate();
	reservation = NULL;
        requests.clear();
        measurements.clear();
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        RuntimeImpl::get_runtime()->free_instance(this);
    }

    Logger::Category log_copy("copy");

    namespace RangeExecutors {
      class Memcpy {
      public:
        Memcpy(void *_dst_base, const void *_src_base, size_t _elmt_size)
          : dst_base((char*)_dst_base), src_base((const char*)_src_base), 
            elmt_size(_elmt_size) { }

        void do_span(int offset, int count)
        {
          off_t byte_offset = offset * elmt_size;
          size_t byte_count = count  * elmt_size;
          memcpy(dst_base + byte_offset,
                 src_base + byte_offset,
                 byte_count);
        }

      protected:
        char *dst_base;
        const char *src_base;
        size_t elmt_size;
      };

      class RedopApply {
      public:
        RedopApply(const ReductionOpUntyped *_redop, void *_dst_base,
                   const void *_src_base, size_t _elmt_size)
          : redop(_redop), dst_base((char*)_dst_base),
            src_base((const char*)_src_base), elmt_size(_elmt_size) { }

        void do_span(int offset, int count)
        {
          off_t src_offset = offset * redop->sizeof_rhs; 
          off_t dst_offset = offset * elmt_size;
          redop->apply(dst_base + dst_offset,
                       src_base + src_offset,
                       count, false/*exclusive*/);
        }

      protected:
        const ReductionOpUntyped *redop;
        char *dst_base;
        const char *src_base;
        size_t elmt_size;
      };

      class RedopFold {
      public:
        RedopFold(const ReductionOpUntyped *_redop, void *_dst_base,
                  const void *_src_base)
          : redop(_redop), dst_base((char*)_dst_base),
            src_base((const char*)_src_base) { }

        void do_span(int offset, int count)
        {
          off_t byte_offset = offset * redop->sizeof_rhs; 
          redop->fold(dst_base + byte_offset,
                      src_base + byte_offset,
                      count, false/*exclusive*/);
        }

      protected:
        const ReductionOpUntyped *redop;
        char *dst_base;
        const char *src_base;
      };
    }; // Namespace RangeExecutors

    void RegionInstanceImpl::perform_copy_operation(RegionInstanceImpl *target, const ElementMask &src_mask, const ElementMask &dst_mask)
    {
        DetailedTimer::ScopedPush sp(TIME_COPY); 
        const void *src_ptr = base_ptr;
        void       *tgt_ptr = target->base_ptr;
#ifdef DEBUG_REALM
        assert((src_ptr != NULL) && (tgt_ptr != NULL));
#endif
        if (!reduction)
        {
#ifdef DEBUG_REALM
          if (target->reduction)
          {
             fprintf(stderr,"Cannot copy from non-reduction instance %d to reduction instance %d\n",
                      this->index, target->index);
             exit(1);
          }
#endif
          // This is a normal copy
	  // but it assumes AOS!
	  assert((block_size == 1) && (target->block_size == 1));
          RangeExecutors::Memcpy rexec(tgt_ptr, src_ptr, elmt_size);
          ElementMask::forall_ranges(rexec, dst_mask, src_mask);
        }
        else
        {
          // See if this is a list reduction or a fold reduction
          if (list)
          {
            if (!target->reduction)
            {
              // We need to apply the reductions to the actual buffer 
              apply_list(target);
            }
            else
            {
              // Reduction-to-reduction copy 
#ifdef DEBUG_REALM
              // Make sure they are the same kind of reduction
              if (this->redop != target->redop)
              {
                fprintf(stderr,"Illegal copy between reduction instances %d and %d with different reduction operations\n",
                          this->index, target->index);
                exit(1);
              }
#endif
              if (target->list)
              {
                // Append the list
                append_list(target);
              }
              else
              {
                // Otherwise just apply it to its target 
                apply_list(target);
              }
            }
          }
          else
          {
            // This is a reduction instance, see if we are doing a reduction-to-normal copy 
            // or a reduction-to-reduction copy
            if (!target->reduction)
            {
              // Reduction-to-normal copy  
              RangeExecutors::RedopApply rexec(redop, tgt_ptr, src_ptr, elmt_size);
              ElementMask::forall_ranges(rexec, dst_mask, src_mask);
            }
            else
            {
#ifdef DEBUG_REALM
              // Make sure its a reduction fold copy
              if (target->list)
              {
                  fprintf(stderr,"Cannot copy from fold reduction instance %d to list reduction instance %d\n",
                          this->index, target->index);
                  exit(1);
              }
              // Make sure they have the same reduction op
              if (this->redop != target->redop)
              {
                fprintf(stderr,"Illegal copy between reduction instances %d and %d with different reduction operations\n",
                          this->index, target->index);
                exit(1);
              }
#endif
              // Reduction-to-reduction copy
              RangeExecutors::RedopFold rexec(redop, tgt_ptr, src_ptr);
              ElementMask::forall_ranges(rexec, dst_mask, src_mask);
            }
          }
        }
    }

    void RegionInstanceImpl::apply_list(RegionInstanceImpl *target)
    {
#ifdef DEBUG_REALM
        assert(this->list);
        assert(!target->list);
        assert(cur_entry <= num_elmts);
#endif
        // Get the current end of the list
        // Don't use any atomics or anything else, assume that
        // race conditions are handled at the user level above
        if (target->reduction)
        {
          this->redop->fold_list_entry(target->base_ptr, this->base_ptr, cur_entry, 0);
        }
        else
        {
          this->redop->apply_list_entry(target->base_ptr, this->base_ptr, cur_entry, 0);
        }
    }

    void RegionInstanceImpl::append_list(RegionInstanceImpl *target)
    {
#ifdef DEBUG_REALM
        assert(this->list);
        assert(target->list);
#endif
        // TODO: Implement this
        assert(false);
    }

    RegionInstance RegionInstanceImpl::get_instance(void) const
    {
	RegionInstance inst;
	inst.id = index;
	return inst;
    }

    Reservation RegionInstanceImpl::get_reservation(void)
    {
	return reservation->get_reservation();
    }

    void RegionInstanceImpl::verify_access(unsigned ptr)
    {
#if 0
      const ElementMask &mask = region.get_valid_mask();
      if (!mask.is_set(ptr))
      {
        fprintf(stderr,"ERROR: Accessing invalid pointer %d in logical region %d\n",ptr,index);
	assert(0);
      }
#endif
    }

    void* RegionInstanceImpl::get_address(int index, size_t field_start, size_t field_size,
					    size_t within_field)
    {
      if(block_size == 1) {
	// simple AOS case:
	return (base_ptr + (index * elmt_size) + field_start + within_field);
      } else {
	int num_blocks = index / block_size;
	int within_block = index % block_size;

	return (base_ptr + 
	 	(num_blocks * block_size * elmt_size) +
	 	(field_start * block_size) +
		(within_block * field_size) +
		within_field);
      }
    }

    void RegionInstanceImpl::fill_field(unsigned fill_offset, unsigned fill_size, 
                                          const void *fill_value, size_t fill_value_size,
                                          const Domain &domain)
    {
      assert(fill_size == fill_value_size);
      size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
      size_t bytes =
#endif
	find_field(get_field_sizes(), fill_offset, fill_size,
		   field_start, field_size, within_field);
      assert(bytes == fill_size);
      if (domain.get_dim() == 0) {
        if (linearization.get_dim() == 1) {
          Arrays::Mapping<1, 1> *dst_linearization = 
            linearization.get_mapping<1>();
          for (Domain::DomainPointIterator itr(domain); itr; itr++)
          {
            int index = dst_linearization->image(itr.p.get_index());
            void *raw_addr = get_address(index, field_start,
                                         field_size, within_field);
            memcpy(raw_addr, fill_value, fill_value_size);
          }
        } else {
          for (Domain::DomainPointIterator itr(domain); itr; itr++)
          {
            int index = itr.p.get_index();
            void *raw_addr = get_address(index, field_start,
                                         field_size, within_field);
            memcpy(raw_addr, fill_value, fill_value_size);
          }
        }
      } else {
        for (Domain::DomainPointIterator itr(domain); itr; itr++)
        {
          int index = linearization.get_image(itr.p);
          void *raw_addr = get_address(index, field_start,
                                       field_size, within_field);
          memcpy(raw_addr, fill_value, fill_value_size);
        }
      }
    }

#if 0
    RegionAccessor<AccessorGeneric> RegionAccessor<AccessorGeneric>::get_field_accessor(off_t offset, size_t size) const
    {
      return RegionAccessor<AccessorGeneric>(internal_data, field_offset + offset);
    }

    // Acessor Generic (can convert)
    template <>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorGeneric>(void) const
    { return true; }

    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorArray>(void) const
    { 
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      if (impl->is_reduction())
      {
        return false;
      }
      else
      {
        return true;
      }
    }

    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      if (impl->is_reduction() && !impl->is_list_reduction())
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorReductionList>(void) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      if (impl->is_reduction() && impl->is_list_reduction())
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorGPU>(void) const
    {
      return false;
    }

    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorGPUReductionFold>(void) const
    {
      return false;
    }

    bool RegionAccessor<AccessorGeneric>::is_reduction_only(void) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      return impl->is_reduction();
    }

    // Accessor Generic (convert)
    template <>
    RegionAccessor<AccessorGeneric> RegionAccessor<AccessorGeneric>::convert<AccessorGeneric>(void) const
    { return *this; }

    template<>
    RegionAccessor<AccessorArray> RegionAccessor<AccessorGeneric>::convert<AccessorArray>(void) const
    { 
#ifdef DEBUG_REALM
      assert(!this->is_reduction_only());
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      RegionAccessor<AccessorArray> ret(impl->get_base_ptr()); 
#ifdef POINTER_CHECKS
      ret.impl_ptr = impl;
#endif
      return ret;
    }

    template<>
    RegionAccessor<AccessorArrayReductionFold> RegionAccessor<AccessorGeneric>::convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
#ifdef DEBUG_REALM
      assert(impl->is_reduction() && !impl->is_list_reduction());
#endif
      return RegionAccessor<AccessorArrayReductionFold>(impl->get_base_ptr());
    }

    template<>
    RegionAccessor<AccessorReductionList> RegionAccessor<AccessorGeneric>::convert<AccessorReductionList>(void) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data; 
#ifdef DEBUG_REALM
      assert(impl->is_reduction() && impl->is_list_reduction());
#endif
      return RegionAccessor<AccessorReductionList>(impl,impl->get_num_elmts(),impl->get_elmt_size());
    }

    template<>
    RegionAccessor<AccessorGPU> RegionAccessor<AccessorGeneric>::convert<AccessorGPU>(void) const
    {
      assert(false);
      return RegionAccessor<AccessorGPU>();
    }

    template<>
    RegionAccessor<AccessorGPUReductionFold> RegionAccessor<AccessorGeneric>::convert<AccessorGPUReductionFold>(void) const
    {
      assert(false);
      return RegionAccessor<AccessorGPUReductionFold>();
    }

    RegionAccessor<AccessorReductionList>::RegionAccessor(void *_internal_data,
                                                                                        size_t _num_entries,
                                                                                        size_t _elmt_size)
    {
      internal_data = _internal_data;

      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      cur_size = impl->get_cur_entry(); 
      max_size = _num_entries;
      entry_list = impl->get_base_ptr();
    }

    void RegionAccessor<AccessorReductionList>::flush(void) const
    {
      assert(false);
    }

    void RegionAccessor<AccessorReductionList>::reduce_slow_case(size_t my_pos, unsigned ptrvalue,
                                                                const void *entry, size_t sizeof_entry) const
    {
      assert(false);
    }

#ifdef POINTER_CHECKS
    void RegionAccessor<AccessorGeneric>::verify_access(unsigned ptr) const
    {
        ((RegionInstanceImpl*)internal_data)->verify_access(ptr);
    }

#if 0
    void RegionAccessor<AccessorArray>::verify_access(unsigned ptr) const
    {
        ((RegionInstanceImpl*)impl_ptr)->verify_access(ptr);
    }
#endif
#endif
#endif

    Logger::Category log_region("region");

  };
};

namespace Realm {

    ////////////////////////////////////////////////////////
    // IndexSpace 
    ////////////////////////////////////////////////////////

    /*static*/ const IndexSpace IndexSpace::NO_SPACE = { 0 };
    /*static*/ const Domain Domain::NO_DOMAIN = Domain();

    // Lifting Declaration of IndexSpaceImpl above allocator so we can call it in allocator
    
    IndexSpace IndexSpace::create_index_space(size_t num_elmts)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_free_metadata(num_elmts);	
	//log_region.info("index space created: id=%x num=%zd",
        //		   r->get_metadata().id, num_elmts);
	return r->get_metadata();
    }

    IndexSpace IndexSpace::create_index_space(const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_free_metadata(mask);
      return r->get_metadata();
    }

    IndexSpace IndexSpace::create_index_space(IndexSpace parent, 
                                              const ElementMask &mask,
                                              bool allocable /*= true*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceImpl *par = RuntimeImpl::get_runtime()->get_metadata_impl(parent);
      IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_free_metadata(par, mask);
      //log_region.info("index space created: id=%x parent=%x",
      //		 r->get_metadata().id, parent.id);
      return r->get_metadata();
    }

    Event IndexSpace::create_equal_subspaces(size_t count, size_t granularity,
                                             std::vector<IndexSpace>& subspaces,
                                             bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Make each of the resulting subspaces
      subspaces.resize(count);
      std::vector<IndexSpaceImpl*> subspace_impls(count);
      for (unsigned idx = 0; idx < count; idx++) {
        subspace_impls[idx] = rt->get_free_metadata(impl);
        subspaces[idx] = subspace_impls[idx]->get_metadata();
      }
      DeferredEqualSpaces *op = new DeferredEqualSpaces(subspace_impls, impl,
                                                        granularity, done_impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_equal_subspaces(size_t count, size_t granularity,
                                             std::vector<IndexSpace>& subspaces,
                                             const Realm::ProfilingRequestSet &reqs,
                                             bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Make each of the resulting subspaces
      subspaces.resize(count);
      std::vector<IndexSpaceImpl*> subspace_impls(count);
      for (unsigned idx = 0; idx < count; idx++) {
        subspace_impls[idx] = rt->get_free_metadata(impl);
        subspaces[idx] = subspace_impls[idx]->get_metadata();
      }
      DeferredEqualSpaces *op = new DeferredEqualSpaces(subspace_impls, reqs, impl,
                                                        granularity, done_impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_weighted_subspaces(size_t count, size_t granularity,
                                                const std::vector<int> &weights,
                                                std::vector<IndexSpace> &subspaces,
                                                bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Make each of the resulting subspaces
      subspaces.resize(count);
      std::vector<IndexSpaceImpl*> subspace_impls(count);
      for (unsigned idx = 0; idx < count; idx++) {
        subspace_impls[idx] = rt->get_free_metadata(impl);
        subspaces[idx] = subspace_impls[idx]->get_metadata();
      }
      DeferredWeightedSpaces *op = new DeferredWeightedSpaces(subspace_impls, impl,
                                                  granularity, done_impl, weights);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_weighted_subspaces(size_t count, size_t granularity,
                                                const std::vector<int> &weights,
                                                std::vector<IndexSpace> &subspaces,
                                                const Realm::ProfilingRequestSet &reqs,
                                                bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Make each of the resulting subspaces
      subspaces.resize(count);
      std::vector<IndexSpaceImpl*> subspace_impls(count);
      for (unsigned idx = 0; idx < count; idx++) {
        subspace_impls[idx] = rt->get_free_metadata(impl);
        subspaces[idx] = subspace_impls[idx]->get_metadata();
      }
      DeferredWeightedSpaces *op = new DeferredWeightedSpaces(subspace_impls, reqs, impl,
                                                  granularity, done_impl, weights);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_subspaces_by_field(
                                        const std::vector<FieldDataDescriptor> &field_data,
                                        std::map<DomainPoint, IndexSpace> &subspaces,
                                        bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Fill in the subspaces
      for (std::map<DomainPoint,IndexSpace>::iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *child = rt->get_free_metadata(impl);
        it->second = child->get_metadata();
      }
      DeferredFieldSpaces *op = new DeferredFieldSpaces(field_data, subspaces,
                                                        done_impl, impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_subspaces_by_field(
                                        const std::vector<FieldDataDescriptor> &field_data,
                                        std::map<DomainPoint, IndexSpace> &subspaces,
                                        const Realm::ProfilingRequestSet &reqs,
                                        bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Fill in the subspaces
      for (std::map<DomainPoint,IndexSpace>::iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *child = rt->get_free_metadata(impl);
        it->second = child->get_metadata();
      }
      DeferredFieldSpaces *op = new DeferredFieldSpaces(field_data, subspaces, reqs,
                                                        done_impl, impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_subspaces_by_image(
                                        const std::vector<FieldDataDescriptor> &field_data,
                                        std::map<IndexSpace, IndexSpace> &subspaces,
                                        bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Fill in the subspaces
      for (std::map<IndexSpace,IndexSpace>::iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *child = rt->get_free_metadata(impl);
        it->second = child->get_metadata();
      }
      DeferredImageSpaces *op = new DeferredImageSpaces(field_data, subspaces,
                                                        done_impl, impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_subspaces_by_image(
                                        const std::vector<FieldDataDescriptor> &field_data,
                                        std::map<IndexSpace, IndexSpace> &subspaces,
                                        const Realm::ProfilingRequestSet &reqs,
                                        bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Fill in the subspaces
      for (std::map<IndexSpace,IndexSpace>::iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *child = rt->get_free_metadata(impl);
        it->second = child->get_metadata();
      }
      DeferredImageSpaces *op = new DeferredImageSpaces(field_data, subspaces, reqs,
                                                        done_impl, impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_subspaces_by_preimage(
                                        const std::vector<FieldDataDescriptor> &field_data,
                                        std::map<IndexSpace, IndexSpace> &subspaces,
                                        bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Fill in the subspaces
      for (std::map<IndexSpace,IndexSpace>::iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *child = rt->get_free_metadata(impl);
        it->second = child->get_metadata();
      }
      DeferredPreimageSpaces *op = new DeferredPreimageSpaces(field_data, subspaces,
                                                              done_impl, impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    Event IndexSpace::create_subspaces_by_preimage(
                                        const std::vector<FieldDataDescriptor> &field_data,
                                        std::map<IndexSpace, IndexSpace> &subspaces,
                                        const Realm::ProfilingRequestSet &reqs,
                                        bool mutable_results, Event wait_on) const
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_impl = rt->get_free_event();
      Event result = done_impl->get_event();
      IndexSpaceImpl *impl = rt->get_metadata_impl(*this);
      // Fill in the subspaces
      for (std::map<IndexSpace,IndexSpace>::iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *child = rt->get_free_metadata(impl);
        it->second = child->get_metadata();
      }
      DeferredPreimageSpaces *op = new DeferredPreimageSpaces(field_data, subspaces, reqs,
                                                              done_impl, impl);
      if (wait_on.exists()) {
        EventImpl *source = rt->get_event_impl(wait_on);
        source->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    /*static*/ Event IndexSpace::compute_index_spaces(
                                          std::vector<BinaryOpDescriptor> &pairs,
                                          bool mutable_results, Event wait_on)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      // Fill in the index space output
      for (unsigned idx = 0; idx < pairs.size(); idx++) {
        IndexSpaceImpl *parent = rt->get_metadata_impl(pairs[idx].parent);
        IndexSpaceImpl *result = rt->get_free_metadata(parent);
        pairs[idx].result = result->get_metadata();
      }
      // Construct an operation to compute the result
      EventImpl *done_event = rt->get_free_event();
      Event result = done_event->get_event();
      ComputeIndexSpaces *op = new ComputeIndexSpaces(pairs, done_event);
      if (wait_on.exists()) {
        EventImpl *event_impl = rt->get_event_impl(wait_on);
        event_impl->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    /*static*/ Event IndexSpace::compute_index_spaces(
                                          std::vector<BinaryOpDescriptor> &pairs,
                                          const Realm::ProfilingRequestSet &reqs,
                                          bool mutable_results, Event wait_on)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      // Fill in the index space output
      for (unsigned idx = 0; idx < pairs.size(); idx++) {
        IndexSpaceImpl *parent = rt->get_metadata_impl(pairs[idx].parent);
        IndexSpaceImpl *result = rt->get_free_metadata(parent);
        pairs[idx].result = result->get_metadata();
      }
      // Construct an operation to compute the result
      EventImpl *done_event = rt->get_free_event();
      Event result = done_event->get_event();
      ComputeIndexSpaces *op = new ComputeIndexSpaces(pairs, reqs, done_event);
      if (wait_on.exists()) {
        EventImpl *event_impl = rt->get_event_impl(wait_on);
        event_impl->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    /*static*/ Event IndexSpace::reduce_index_spaces(IndexSpaceOperation op,
                                      const std::vector<IndexSpace> &spaces,
                                      IndexSpace &result, bool mutable_results,
                                      IndexSpace parent, Event wait_on)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      IndexSpaceImpl *parent_impl = rt->get_metadata_impl(parent);
      IndexSpaceImpl *result_impl = rt->get_free_metadata(parent_impl);
      result = result_impl->get_metadata();
      EventImpl *done_event = rt->get_free_event();
      Event ready = done_event->get_event();
      ReduceIndexSpaces *reduce_op = 
        new ReduceIndexSpaces(op, spaces, result_impl, done_event);
      if (wait_on.exists()) {
        EventImpl *event_impl = rt->get_event_impl(wait_on);
        event_impl->add_waiter(wait_on.gen, reduce_op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(reduce_op);
      }
      return ready;
    }

    /*static*/ Event IndexSpace::reduce_index_spaces(IndexSpaceOperation op,
                                      const std::vector<IndexSpace> &spaces,
                                      const Realm::ProfilingRequestSet &reqs,
                                      IndexSpace &result, bool mutable_results,
                                      IndexSpace parent, Event wait_on)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      IndexSpaceImpl *parent_impl = rt->get_metadata_impl(parent);
      IndexSpaceImpl *result_impl = rt->get_free_metadata(parent_impl);
      result = result_impl->get_metadata();
      EventImpl *done_event = rt->get_free_event();
      Event ready = done_event->get_event();
      ReduceIndexSpaces *reduce_op = 
        new ReduceIndexSpaces(op, spaces, reqs, result_impl, done_event);
      if (wait_on.exists()) {
        EventImpl *event_impl = rt->get_event_impl(wait_on);
        event_impl->add_waiter(wait_on.gen, reduce_op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(reduce_op);
      }
      return ready;
    }

    IndexSpaceAllocator IndexSpace::create_allocator(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(*this);
	return IndexSpaceAllocator(r->create_allocator());
    }

    RegionInstance Domain::create_instance(Memory m, size_t elmt_size,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      std::vector<size_t> field_sizes(1);
      field_sizes[0] = elmt_size;
      // for an instance with a single field, block size should be a don't care
      return create_instance(m, field_sizes, 1, redop_id);
    }

    RegionInstance Domain::create_instance(Memory m, size_t elmt_size,
                                           const Realm::ProfilingRequestSet &reqs,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      std::vector<size_t> field_sizes(1);
      field_sizes[0] = elmt_size;
      // for an instance with a single field, block size should be a don't care
      return create_instance(m, field_sizes, 1, reqs, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
					   ReductionOpID redop_id) const
    {
      Realm::ProfilingRequestSet requests;
      return create_instance(memory, field_sizes, block_size, requests, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
                                           const Realm::ProfilingRequestSet &reqs,
					   ReductionOpID redop_id) const
    {
        if (!memory.exists())
        {
          return RegionInstance::NO_INST;
        }

        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	if(get_dim() > 0) {
	  // we have a rectangle - figure out its volume and create based on that
	  DomainLinearization dl;
	  Arrays::Rect<1> inst_extent;
	  switch(get_dim()) {
	  case 1:
	    {
	      Arrays::FortranArrayLinearization<1> cl(get_rect<1>(), 0);
	      dl = DomainLinearization::from_mapping<1>(Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
	      inst_extent = cl.image_convex(get_rect<1>());
	      break;
	    }

	  case 2:
	    {
	      Arrays::FortranArrayLinearization<2> cl(get_rect<2>(), 0);
	      dl = DomainLinearization::from_mapping<2>(Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
	      inst_extent = cl.image_convex(get_rect<2>());
	      break;
	    }

	  case 3:
	    {
	      Arrays::FortranArrayLinearization<3> cl(get_rect<3>(), 0);
	      dl = DomainLinearization::from_mapping<3>(Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
	      inst_extent = cl.image_convex(get_rect<3>());
	      break;
	    }

	  default: assert(0);
	  }
	  return IndexSpaceImpl::create_instance(memory, field_sizes, block_size, dl, 
                                                   int(inst_extent.hi) + 1, reqs, redop_id);
	} else {
	  IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(get_index_space());

	  DomainLinearization dl;
	  size_t count = r->get_num_elmts();
#ifndef FULL_SIZE_INSTANCES
	  // if we know that we just need a subset of the elements, make a smaller instance
	  {
	    int first_elmt = r->get_element_mask().first_enabled();
	    int last_elmt = r->get_element_mask().last_enabled();

	    // not 64-bit clean, but the shared LLR probably doesn't have to worry about that
	    if((first_elmt >= 0) && (last_elmt >= first_elmt) &&
	       ((first_elmt > 0) || ((size_t)last_elmt < count-1))) {
	      // reduce instance size, and block size if necessary
	      count = last_elmt - first_elmt + 1;
	      if(block_size > count)
		block_size = count;
	      Translation<1> inst_offset(-first_elmt);
	      dl = DomainLinearization::from_mapping<1>(Mapping<1,1>::new_dynamic_mapping(inst_offset));
	    }
	  }
#endif
	  return IndexSpaceImpl::create_instance(memory, field_sizes, block_size, 
				                   dl, count, reqs, redop_id);
	}
    }

    RegionInstance Domain::create_hdf5_instance(const char *file_name,
                                                const std::vector<size_t> &field_sizes,
                                                const std::vector<const char*> &field_files,
                                                bool ready_only) const
    {
      // TODO: Implement this
      assert(false);
      return RegionInstance::NO_INST;
    }

    RegionInstance Domain::create_file_instance(const char *file_name,
                                                const std::vector<size_t> &field_sizes,
                                                legion_lowlevel_file_mode_t file_mode) const
    {
      // TODO: Implement this
      assert(false);
      return RegionInstance::NO_INST;
    }

#if 0
    RegionInstance IndexSpace::create_instance(Memory m, ReductionOpID redop) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(*this);
        return r->create_instance(m, redop);
    }

    RegionInstance IndexSpace::create_instance(Memory m, ReductionOpID redop,
                                                  off_t list_size, RegionInstance parent_inst) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(*this);
        return r->create_instance(m, redop, list_size, parent_inst);
    }
#endif

    void IndexSpace::destroy(Event wait_on) const
    {
        // TODO: figure out how to wait
        assert(false);
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(*this);
        r->deactivate();
    }

#if 0
    void IndexSpace::destroy_allocator(IndexSpaceAllocator a) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(*this);
	r->destroy_allocator(a);
    }

    void IndexSpace::destroy_instance(RegionInstance i) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(*this);
	r->destroy_instance(i);
    }
#endif

    const ElementMask &IndexSpace::get_valid_mask(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(*this);
      return r->get_element_mask();
    }

    Event Domain::fill(const std::vector<CopySrcDstField> &dsts,
                       const void *fill_value, size_t fill_value_size,
                       Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return IndexSpaceImpl::fill(dsts, fill_value, fill_value_size,
                                    wait_on, *this);
    }

    Event Domain::copy(RegionInstance src_inst, RegionInstance dst_inst, size_t elem_size,
		       Event wait_on /*= Event::NO_EVENT*/,
		       ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(get_index_space());
      //return r->copy(src_inst, dst_inst, elem_size, *this, wait_on, redop_id, red_fold);
      return IndexSpaceImpl::copy(src_inst, dst_inst, elem_size, *this,
                                    wait_on, redop_id, red_fold);
    }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       Event wait_on,
		       ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(get_index_space());
      //return r->copy(srcs, dsts, *this, wait_on, redop_id, red_fold);
      return IndexSpaceImpl::copy(srcs, dsts, *this, wait_on, redop_id, red_fold);
    }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       const ElementMask& mask, Event wait_on,
		       ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(get_index_space());
      assert(0);
      //return r->copy(srcs, dsts, *this, wait_on, redop_id, red_fold);
      return IndexSpaceImpl::copy(srcs, dsts, *this, wait_on, redop_id, red_fold);
    }

    Event Domain::copy_indirect(const CopySrcDstField &idx,
				const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				Event wait_on /*= Event::NO_EVENT*/,
				ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      // TODO: Sean needs to implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event Domain::copy_indirect(const CopySrcDstField &idx,
				const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				const ElementMask &mask,
				Event wait_on /*= Event::NO_EVENT*/,
				ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      // TODO: Sean needs to implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event Domain::fill(const std::vector<CopySrcDstField> &dsts,
                       const Realm::ProfilingRequestSet &reqs,
                       const void *fill_value, size_t fill_value_size,
                       Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return IndexSpaceImpl::fill(dsts, reqs, fill_value, fill_value_size,
                                    wait_on, *this);
    }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
                       const std::vector<CopySrcDstField>& dsts,
                       const Realm::ProfilingRequestSet &requests,
                       Event wait_on, ReductionOpID redop_id, bool red_fold) const
    {
      return IndexSpaceImpl::copy(srcs, dsts, requests, *this,
                                    wait_on, redop_id, red_fold);
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    bool IndexSpaceImpl::activate(size_t num)
    {
	bool result = false;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{ 
		active = true;
		result = true;
		num_elmts = num;
		reservation = RuntimeImpl::get_runtime()->get_free_reservation();
                mask = ElementMask(num_elmts);
                parent = NULL;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    bool IndexSpaceImpl::activate(const ElementMask &m)
    {
      bool result = false;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (!active)
      {
        active = true;
        result = true;
        num_elmts = m.get_num_elmts();
        reservation = RuntimeImpl::get_runtime()->get_free_reservation();
        mask = m;
        parent = NULL;
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    bool IndexSpaceImpl::activate(IndexSpaceImpl *par)
    {
      bool result = false;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (!active)
      {
        active = true;
        result = true;
        parent = par;
        num_elmts = parent->get_element_mask().get_num_elmts();
        reservation = RuntimeImpl::get_runtime()->get_free_reservation();
        mask = ElementMask(num_elmts);
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    bool IndexSpaceImpl::activate(IndexSpaceImpl *par, const ElementMask &m)
    {
      bool result = false;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (!active)
      {
        active = true;
        result = true;
        num_elmts = m.get_num_elmts();
        reservation = RuntimeImpl::get_runtime()->get_free_reservation();
        mask = m;
        parent = par;
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    void IndexSpaceImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;
	num_elmts = 0;
	reservation->deactivate();
	reservation = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        RuntimeImpl::get_runtime()->free_metadata(this);
    }

    unsigned IndexSpaceImpl::allocate_space(unsigned count)
    {
        int result = 0;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        if (parent == NULL)
        {
            // Do the allocation ourselves
          result = mask.find_disabled(count);
          if (result == -1)
          {
              // Allocation failure, didn't work
              fprintf(stderr,"Allocation failure in shared low level runtime. "
                  "No available space for %d elements in region %d.\n",count, index);
              exit(1);
          }
          //printf("Allocating element %d in region %d\n",result,index);
        }
        else
        {
            // Make the parent do it and intercept the returning value
            result = parent->allocate_space(count);
        }
#ifdef DEBUG_REALM
        assert(result >= 0);
#endif
        // Update the mask to reflect the allocation
        mask.enable(result,count);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        return unsigned(result);
    }

    void IndexSpaceImpl::free_space(unsigned ptr, unsigned count)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_REALM
        // Some sanity checks
        assert(int(ptr) < mask.get_num_elmts());
        assert(int(ptr+count) < mask.get_num_elmts());
        assert(mask.is_set(ptr));
#endif
        if (parent == NULL)
        {
           // No need to do anything here 
        }
        else
        {
            // Tell the parent to do it
            parent->free_space(ptr,count);
        }
        // Update our mask no matter what
        mask.disable(ptr,count);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    IndexSpace IndexSpaceImpl::get_metadata(void)
    {
	IndexSpace meta;
	meta.id = index;
	return meta;
    }

    ElementMask& IndexSpaceImpl::get_element_mask(void)
    {
#ifdef DEBUG_REALM
      assert(active);
#endif
      return mask;
    }

    const ElementMask& IndexSpaceImpl::get_element_mask(void) const
    {
#ifdef DEBUG_REALM
      assert(active);
#endif
      return mask;
    }

    void IndexSpaceImpl::create_equal_subspaces(
              const std::vector<IndexSpaceImpl*> &subspaces, size_t granularity)
    {
      // First count how many elements we have in our mask
      size_t elem_count = mask.pop_count();
      // Compute how many elements need to go in each sub-space
      // Round up to full elements
      size_t subspace_count = (elem_count + subspaces.size() - 1) / subspaces.size();
      // Clamp to granularity if necessary
      if (subspace_count < granularity)
        subspace_count = granularity;
      // Iterate over the enabled elements and assign them 
      int current = 0;
      for (std::vector<IndexSpaceImpl*>::const_iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        bool done_early = false;
        ElementMask &child_mask = (*it)->get_element_mask();
        for (unsigned idx = 0; idx < subspace_count; idx++)
        {
          int next = mask.find_enabled(1, current); 
          if (next != -1) {
            child_mask.enable(next);
            // Start the search for the next element one past our current place
            current = next + 1;
          } else {
            // Couldn't find anymore elements so we are done
            done_early = true;
            break;
          }
        }
        // If we are out of elements then we are done
        if (done_early)
          break;
      }
    }

    void IndexSpaceImpl::create_weighted_subspaces(
              const std::vector<IndexSpaceImpl*> &subspaces, size_t granularity,
              std::vector<int> &weights)
    {
      assert(weights.size() == subspaces.size());
      // Count the sum of all the weights
      int total_weight = 0;
      for (std::vector<int>::const_iterator it = weights.begin();
            it != weights.end(); it++)
        total_weight += *it;
      if (total_weight == 0) {
        // If total granularity is zero fall back to equal partition
        create_equal_subspaces(subspaces, granularity);
        return;
      }
      // Count how many elements we have in our mask
      size_t elem_count = mask.pop_count();
#if 0
      if ((elem_count <= size_t(total_weight)) && (granularity > 0)) {
        // More weight than elements, if granularity is greater than zero
        // then scale any non-zero weights so they will at least get the
        // minimum number of elements ensured by granularity
        size_t weight_per_element = (total_weight + elem_count - 1) / elem_count;
        size_t minimum_weight = granularity * weight_per_element;
        for (std::vector<int>::iterator it = weights.begin(); 
              it != weights.end(); it++)
        {
          if (size_t(*it) < minimum_weight) {
            total_weight += (minimum_weight - (*it));
            (*it) = minimum_weight;
          }
        }
      }
#endif
      int current = 0;
      unsigned weight_idx = 0;
      float float_count = elem_count; // convert to floating point
      float float_total_inv = 1.f / total_weight;
      // Keep track of the accumulated rounding error, and once
      // it gets to be larger than 1.0 add an element to the next value
      float accumulated_error = 0.f;
      for (std::vector<IndexSpaceImpl*>::const_iterator it = subspaces.begin();
            it != subspaces.end(); it++, weight_idx++)
      {
        // Skip any entries with zero weight
        if (weights[weight_idx] == 0)
          continue;
        float local_elems = float_count * float(weights[weight_idx]) * float_total_inv;
        // Convert back to an integer by rounding
        float rounded = (local_elems + 0.5);
        size_t subspace_count = rounded; // truncate back to integer
        // See the error that we got
        accumulated_error += (local_elems - float(subspace_count));
        // If we have accumulated too much round-off error, add an extra element
        if (accumulated_error >= 1.f) {
          subspace_count++;
          accumulated_error -= 1.f;
        }
        // If we are on the last iteration and we have round-off error
        // then increment our count by one
        if ((weight_idx == (subspaces.size()-1)) && (accumulated_error > 0.f)) {
          subspace_count++;
        }
        // If we are less than the minimum granularity, increase that now
        if (subspace_count < granularity)
          subspace_count = granularity;
        bool done_early = false;
        ElementMask &child_mask = (*it)->get_element_mask();
        for (unsigned idx = 0; idx < subspace_count; idx++)
        {
          int next = mask.find_enabled(1, current);
          if (next != -1) {
            child_mask.enable(next);
            // Start the search for the next element one past our current place
            current = next + 1;
          } else {
            // Couldn't find any more elements so we are done
            done_early = true;
            break;
          }
        }
        if (done_early)
          break;
      }
    }

    void IndexSpaceImpl::create_subspaces_by_field(
			    const std::vector<IndexSpace::FieldDataDescriptor> &field_data,
                            const std::map<DomainPoint,IndexSpace> &subspaces)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      // First convert all the index spaces to impls 
      std::map<DomainPoint,ElementMask*> element_masks;
      int dim = -1;
      for (std::map<DomainPoint,IndexSpace>::const_iterator it = 
            subspaces.begin(); it != subspaces.end(); it++)
      {
        // Make sure all the dimensions are the same
        if (dim == -1)
          dim = it->first.get_dim();
        else
          assert(dim == it->first.get_dim());
        IndexSpaceImpl *impl = rt->get_metadata_impl(it->second);
        element_masks[it->first] = &(impl->get_element_mask());
      }
      // Now iterate over all the field data and assign the points
      for (std::vector<IndexSpace::FieldDataDescriptor>::const_iterator it = field_data.begin();
            it != field_data.end(); it++)
      {
        // Make sure that the dim aligns with the field size
        assert((((dim == 0) ? 1 : dim) * sizeof(int)) == it->field_size);
        RegionInstanceImpl *inst = rt->get_instance_impl(it->inst);
        // Find the field data for this field
        size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
        size_t bytes =
#endif
	  find_field(inst->get_field_sizes(), it->field_offset,
		     it->field_size, field_start, field_size, within_field);
        // Should have at least enough bytes to read
        assert(bytes >= it->field_size);
        // Now iterate over all the points in the element space 
        IndexSpaceImpl *source = rt->get_metadata_impl(it->index_space);
        ElementMask::Enumerator enumerator(source->get_element_mask(), 0, 1/*enabled*/);
        switch (dim)
        {
          case 0:
            {
              int pos, len;
              while (enumerator.get_next(pos,len)) {
                for (int i = 0; i < len; i++) {
                  void *raw_addr = 
                    inst->get_address(pos+i, field_start, field_size, within_field);
                  // Interpret this as color
                  DomainPoint dp(*((int*)raw_addr));
                  std::map<DomainPoint,ElementMask*>::const_iterator finder = 
                    element_masks.find(dp);
                  assert(finder != element_masks.end());
                  finder->second->enable(pos+i);
                }
              }
              break;
            }
          case 1:
            {
              int pos, len;
              while (enumerator.get_next(pos,len)) {
                for (int i = 0; i < len; i++) {
                  void *raw_addr = 
                    inst->get_address(pos+i, field_start, field_size, within_field);
                  // Interpret this as a 1-D point
                  DomainPoint dp = 
                    DomainPoint::from_point<1>(Arrays::Point<1>((int*)raw_addr));
                  std::map<DomainPoint,ElementMask*>::const_iterator finder = 
                    element_masks.find(dp);
                  assert(finder != element_masks.end());
                  finder->second->enable(pos+i);
                }
              }
              break;
            }
          case 2:
            {
              int pos, len;
              while (enumerator.get_next(pos,len)) {
                for (int i = 0; i < len; i++) {
                  void *raw_addr = 
                    inst->get_address(pos+i, field_start, field_size, within_field);
                  // Interpret this as a 2-D point
                  DomainPoint dp = 
                    DomainPoint::from_point<2>(Arrays::Point<2>((int*)raw_addr));
                  std::map<DomainPoint,ElementMask*>::const_iterator finder = 
                    element_masks.find(dp);
                  assert(finder != element_masks.end());
                  finder->second->enable(pos+i);
                }
              }
              break;
            }
          case 3:
            {
              int pos, len;
              while (enumerator.get_next(pos,len)) {
                for (int i = 0; i < len; i++) {
                  void *raw_addr = 
                    inst->get_address(pos+i, field_start, field_size, within_field);
                  // Interpret this as a 3-D point
                  DomainPoint dp = 
                    DomainPoint::from_point<3>(Arrays::Point<3>((int*)raw_addr));
                  std::map<DomainPoint,ElementMask*>::const_iterator finder = 
                    element_masks.find(dp);
                  assert(finder != element_masks.end());
                  finder->second->enable(pos+i);
                }
              }
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
    }

    void IndexSpaceImpl::create_subspaces_by_image(
                            const std::vector<IndexSpace::FieldDataDescriptor> &field_data,
                            const std::map<IndexSpace,IndexSpace> &subspaces)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      // First convert all the index spaces to element masks
      std::map<const ElementMask*,ElementMask*> element_masks;
      for (std::map<IndexSpace,IndexSpace>::const_iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *key = rt->get_metadata_impl(it->first);
        IndexSpaceImpl *value = rt->get_metadata_impl(it->second);
        const ElementMask *key_mask = &(key->get_element_mask());
        ElementMask *value_mask = &(value->get_element_mask());
        element_masks[key_mask] = value_mask;
      }
      // Iterate over the field data
      for (std::vector<IndexSpace::FieldDataDescriptor>::const_iterator it = field_data.begin();
            it != field_data.end(); it++)
      {
        // Make sure that the dim aligns with the field size
        assert(sizeof(ptr_t) == it->field_size);
        RegionInstanceImpl *inst = rt->get_instance_impl(it->inst);
        // Find the field data for this field
        size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
        size_t bytes =
#endif
	  find_field(inst->get_field_sizes(), it->field_offset,
		     it->field_size, field_start, field_size, within_field);
        // Should have at least enough bytes to read
        assert(bytes >= it->field_size);
        IndexSpaceImpl *source = rt->get_metadata_impl(it->index_space);
        const ElementMask &source_mask = source->get_element_mask();
        // Iterate over all the index spaces and find intersections
        for (std::map<const ElementMask*,ElementMask*>::const_iterator mask_it =
              element_masks.begin(); mask_it != element_masks.end(); mask_it++)
        {
          ElementMask overlap = source_mask & *(mask_it->first);
          // If there is no overlap, keep going
          if (!overlap)
            continue;
          // Otherwise do the projection
          ElementMask::Enumerator enumerator(overlap, 0, 1/*enabled*/);  
          int pos, len;
          while (enumerator.get_next(pos,len)) {   
            for (int i = 0; i < len; i++) {
              void *raw_addr = 
                    inst->get_address(pos+i, field_start, field_size, within_field);
              int ptr = *((int*)raw_addr);
              // Check to make sure that the pointer is in our set
              // If it's not then we ignore it
              if (mask.is_set(ptr))
                mask_it->second->enable(ptr);
            }
          }
        }
      }
    }

    void IndexSpaceImpl::create_subspaces_by_preimage(
                            const std::vector<IndexSpace::FieldDataDescriptor> &field_data,
                            const std::map<IndexSpace,IndexSpace> &subspaces)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      // First convert all the index spaces to element masks
      std::map<const ElementMask*,ElementMask*> element_masks;
      for (std::map<IndexSpace,IndexSpace>::const_iterator it = subspaces.begin();
            it != subspaces.end(); it++)
      {
        IndexSpaceImpl *key = rt->get_metadata_impl(it->first);
        IndexSpaceImpl *value = rt->get_metadata_impl(it->second);
        const ElementMask *key_mask = &(key->get_element_mask());
        ElementMask *value_mask = &(value->get_element_mask());
        element_masks[key_mask] = value_mask;
      }
      // Iterate over the field data
      for (std::vector<IndexSpace::FieldDataDescriptor>::const_iterator it = field_data.begin();
            it != field_data.end(); it++)
      {
        // Make sure that the dim aligns with the field size
        assert(sizeof(ptr_t) == it->field_size);
        RegionInstanceImpl *inst = rt->get_instance_impl(it->inst);
        // Find the field data for this field
        size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
        size_t bytes =
#endif
	  find_field(inst->get_field_sizes(), it->field_offset,
		     it->field_size, field_start, field_size, within_field);
        // Should have at least enough bytes to read
        assert(bytes >= it->field_size);
        IndexSpaceImpl *source = rt->get_metadata_impl(it->index_space);
        ElementMask::Enumerator enumerator(source->get_element_mask(), 0, 1/*enabled*/);
        // Iterate over all the points
        int pos, len;
        while (enumerator.get_next(pos,len)) {   
          for (int i = 0; i < len; i++) {
            void *raw_addr = 
                  inst->get_address(pos+i, field_start, field_size, within_field);
            int ptr = *((int*)raw_addr);
            // Check to 
            // Now for the expensive part, figure out which subspaces
            // this pointer is a part of and set the corresponding
            // points in the right element masks
            for (std::map<const ElementMask*,ElementMask*>::const_iterator mask_it = 
                  element_masks.begin(); mask_it != element_masks.end(); mask_it++)
            {
              if (mask_it->first->is_set(ptr))
                mask_it->second->enable(pos+i);
            }
          }
        }
      }
    }

    IndexSpaceAllocatorImpl *IndexSpaceImpl::create_allocator(void)
    {
      IndexSpaceAllocatorImpl *alloc_impl = new IndexSpaceAllocatorImpl(get_metadata());
      return alloc_impl;
    }

    /*static*/ RegionInstance IndexSpaceImpl::create_instance(Memory m,
						     const std::vector<size_t>& field_sizes,
						     size_t block_size, 
						     const DomainLinearization& dl,
						     size_t num_elements,
                                                     const Realm::ProfilingRequestSet &reqs,
						     ReductionOpID redop_id /*=0*/)
    {
        if (!m.exists())
        {
          return RegionInstance::NO_INST;
        }
        // First try to create the location in the memory, if there is no space
        // don't bother trying to make the data
        MemoryImpl *mem = RuntimeImpl::get_runtime()->get_memory_impl(m);

	size_t elmt_size = 0;
	for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	    it != field_sizes.end();
	    it++)
	  elmt_size += *it;

	// also have to round num_elmts up to block size
	size_t rounded_num_elmts = num_elements;
	if(block_size > 1) {
	  size_t leftover = rounded_num_elmts % block_size;
	  if(leftover)
	    rounded_num_elmts += block_size - leftover;
	}

	char *ptr = (char*)mem->allocate_space(rounded_num_elmts * elmt_size);
	if (ptr == NULL) {
	  return RegionInstance::NO_INST;
	}

	// if a redop was provided, fill the new memory with the op's identity
	const ReductionOpUntyped *redop = 0;
	if(redop_id) {
	  redop = RuntimeImpl::get_runtime()->get_reduction_op(redop_id);
          // We no longer do reduction initialization in the low-level runtime
	}

	RegionInstanceImpl* impl = RuntimeImpl::get_runtime()->get_free_instance(m,
									       num_elements, 
									       rounded_num_elmts*elmt_size,
									       field_sizes,
									       elmt_size, 
									       block_size, dl,
									       ptr, 
									       redop,
                                                                     NULL/*parent instance*/,
                                                                               reqs);
	RegionInstance inst = impl->get_instance();
	return inst;
    }

#if 0
    RegionInstance IndexSpaceImpl::create_instance(Memory m, ReductionOpID redopid, off_t list_size,
                                                              RegionInstance parent_inst) 
    {
        if (!m.exists())
        {
            return RegionInstance::NO_INST; 
        }
        MemoryImpl *mem = RuntimeImpl::get_runtime()->get_memory_impl(m);
 // There must be a reduction operation for a list instance
#ifdef DEBUG_REALM
        assert(redopid > 0);
#endif
        const ReductionOpUntyped *op = RuntimeImpl::get_runtime()->get_reduction_op(redopid); 
        char *ptr = (char*)mem->allocate_space(list_size * (op->sizeof_rhs + sizeof(utptr_t)));
        if (ptr == NULL)
        {
            return RegionInstance::NO_INST;
        }
        // Set everything up
        RegionInstanceImpl *parent_impl = RuntimeImpl::get_runtime()->get_instance_impl(parent_inst);
#ifdef DEBUG_REALM
        assert(parent_impl != NULL);
#endif
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        IndexSpace r = { index };
        RegionInstanceImpl *impl = RuntimeImpl::get_runtime()->get_free_instance(r,m,list_size,op->sizeof_rhs, ptr, op, parent_impl);
        RegionInstance inst = impl->get_instance();
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        return inst;
    }
#endif

    void IndexSpaceImpl::destroy_instance(RegionInstance inst)
    {
	RegionInstanceImpl *impl = RuntimeImpl::get_runtime()->get_instance_impl(inst);
	impl->deactivate();
    }

    Reservation IndexSpaceImpl::get_reservation(void)
    {
	return reservation->get_reservation();
    }	  
      
    namespace RangeExecutors {
      class GatherScatter {
      public:
	GatherScatter(const std::vector<Domain::CopySrcDstField>& _srcs,
		      const std::vector<Domain::CopySrcDstField>& _dsts)
	  : srcs(_srcs), dsts(_dsts)
	{
	  // determine element size
	  elem_size = 0;
	  for(std::vector<Domain::CopySrcDstField>::const_iterator i = srcs.begin(); i != srcs.end(); i++)
	    elem_size += i->size;

	  buffer = new char[elem_size];
	}

	~GatherScatter(void)
	{
	  delete[] buffer;
	}

        void do_span(int start, int count)
        {
	  for(int index = start; index < (start + count); index++) {
	    // gather data from source
	    int write_offset = 0;
	    for(std::vector<Domain::CopySrcDstField>::const_iterator i = srcs.begin(); i != srcs.end(); i++) {
	      RegionInstanceImpl *inst = RuntimeImpl::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start = 0, field_size = 0, within_field = 0;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
                                          field_start, field_size, within_field);
		// printf("RD(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(index, field_start, within_field));
		int xl_index = index; // translated index
		if(inst->get_linearization().get_dim() == 1)
		  xl_index = inst->get_linearization().get_mapping<1>()->image(index);
		assert(bytes > 0);
		memcpy(buffer + write_offset, 
		       inst->get_address(xl_index, field_start, field_size, within_field),
		       bytes);
		offset += bytes;
		size -= bytes;
		write_offset += bytes;
	      }
	    }

	    // now scatter to destination
	    int read_offset = 0;
	    for(std::vector<Domain::CopySrcDstField>::const_iterator i = dsts.begin(); i != dsts.end(); i++) {
	      RegionInstanceImpl *inst = RuntimeImpl::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start = 0, field_size = 0, within_field = 0;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
		                          field_start, field_size, within_field);
		// printf("WR(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(index, field_start, within_field));
		int xl_index = index; // translated index
		if(inst->get_linearization().get_dim() == 1)
		  xl_index = inst->get_linearization().get_mapping<1>()->image(index);
		assert(bytes > 0);
		memcpy(inst->get_address(xl_index, field_start, field_size, within_field),
		       buffer + read_offset, 
		       bytes);
		offset += bytes;
		size -= bytes;
		read_offset += bytes;
	      }
	    }
	  }
	}

        void do_domain(const Domain domain)
        {
	  for(Domain::DomainPointIterator dpi(domain); dpi; dpi++) {
	    DomainPoint dp = dpi.p;

	    // gather data from source
	    int write_offset = 0;
	    for(std::vector<Domain::CopySrcDstField>::const_iterator i = srcs.begin(); i != srcs.end(); i++) {
	      RegionInstanceImpl *inst = RuntimeImpl::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start = 0, field_size = 0, within_field = 0;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
					  field_start, field_size, within_field);
		// printf("RD(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(inst->get_linearization().get_image(dp), field_start, field_size, within_field));
		assert(bytes > 0);
		memcpy(buffer + write_offset, 
		       inst->get_address(inst->get_linearization().get_image(dp), 
					 field_start, field_size, within_field),
		       bytes);
		offset += bytes;
		size -= bytes;
		write_offset += bytes;
	      }
	    }

	    // now scatter to destination
	    int read_offset = 0;
	    for(std::vector<Domain::CopySrcDstField>::const_iterator i = dsts.begin(); i != dsts.end(); i++) {
	      RegionInstanceImpl *inst = RuntimeImpl::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start = 0, field_size = 0, within_field = 0;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
					  field_start, field_size, within_field);
		// printf("WR(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(inst->get_linearization().get_image(dp), field_start, field_size, within_field));
		assert(bytes > 0);
		memcpy(inst->get_address(inst->get_linearization().get_image(dp),
					 field_start, field_size, within_field),
		       buffer + read_offset, 
		       bytes);
		offset += bytes;
		size -= bytes;
		read_offset += bytes;
	      }
	    }
	  }
	}

      protected:
	std::vector<Domain::CopySrcDstField> srcs;
	std::vector<Domain::CopySrcDstField> dsts;
	size_t elem_size;
	char *buffer;
      };

      class ReductionFold {
      public:
        ReductionFold(const std::vector<Domain::CopySrcDstField>& _srcs,
		      const std::vector<Domain::CopySrcDstField>& _dsts,
                      const ReductionOpUntyped *_redop)
	  : srcs(_srcs), dsts(_dsts), redop(_redop) 
        { 
          // Assume reductions can only be applied to a single field at a time
          assert(srcs.size() == 1);
          assert(dsts.size() == 1);
        }
      public:
        void do_span(int start, int count)
        {
          RegionInstanceImpl *src_inst = RuntimeImpl::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstanceImpl *dst_inst = RuntimeImpl::get_runtime()->get_instance_impl(dsts[0].inst);
          // This should be from one reduction fold instance to another
          for (int index = start; index < (start+count); index++)
	  {
	    int src_index = index;
	    if(src_inst->get_linearization().get_dim() == 1)
	      src_index = src_inst->get_linearization().get_mapping<1>()->image(index);
	    int dst_index = index;
	    if(dst_inst->get_linearization().get_dim() == 1)
	      dst_index = src_inst->get_linearization().get_mapping<1>()->image(index);
            // Assume that there is only one field and they are contiguous
            void *src_ptr = src_inst->get_address(src_index, 0, redop->sizeof_rhs, 0);
            void *dst_ptr = dst_inst->get_address(dst_index, 0, redop->sizeof_rhs, 0);
            redop->fold(dst_ptr, src_ptr, 1, false/*exclusive*/);
          }
        }
        void do_domain(const Domain domain)
        {
          RegionInstanceImpl *src_inst = RuntimeImpl::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstanceImpl *dst_inst = RuntimeImpl::get_runtime()->get_instance_impl(dsts[0].inst);
          for(Domain::DomainPointIterator dpi(domain); dpi; dpi++) {
	    DomainPoint dp = dpi.p;
            void *src_ptr = src_inst->get_address(src_inst->get_linearization().get_image(dp), 0, redop->sizeof_rhs, 0);
            void *dst_ptr = dst_inst->get_address(dst_inst->get_linearization().get_image(dp), 0, redop->sizeof_rhs, 0);
            redop->fold(dst_ptr, src_ptr, 1, false/*exclusive*/);
          }
        }
      protected:
        std::vector<Domain::CopySrcDstField> srcs;
        std::vector<Domain::CopySrcDstField> dsts;
        const ReductionOpUntyped *redop;
      };

      class ReductionApply {
      public:
        ReductionApply(const std::vector<Domain::CopySrcDstField>& _srcs,
		       const std::vector<Domain::CopySrcDstField>& _dsts,
                       const ReductionOpUntyped *_redop)
	  : srcs(_srcs), dsts(_dsts), redop(_redop) 
        { 
          // Assume reductions can only be applied to a single field at a time
          assert(srcs.size() == 1);
          assert(dsts.size() == 1);
        }
      public:
        void do_span(int start, int count)
        {
          RegionInstanceImpl *src_inst = RuntimeImpl::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstanceImpl *dst_inst = RuntimeImpl::get_runtime()->get_instance_impl(dsts[0].inst);
          size_t offset = dsts[0].offset;
          size_t size = dsts[0].size;
          size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
          size_t bytes = 
#endif
            find_field(dst_inst->get_field_sizes(), offset, size,
                       field_start, field_size, within_field);
	  assert(bytes == size);
          for (int index = start; index < (start+count); index++)
          {
	    int src_index = index;
	    if(src_inst->get_linearization().get_dim() == 1)
	      src_index = src_inst->get_linearization().get_mapping<1>()->image(index);
	    int dst_index = index;
	    if(dst_inst->get_linearization().get_dim() == 1)
	      dst_index = dst_inst->get_linearization().get_mapping<1>()->image(index);
            void *src_ptr = src_inst->get_address(src_index, 0, redop->sizeof_rhs, 0);  
            void *dst_ptr = dst_inst->get_address(dst_index, field_start, field_size, within_field);
            redop->apply(dst_ptr, src_ptr, 1, false/*exclusive*/);
          }
        }
        void do_domain(const Domain domain)
        {
          RegionInstanceImpl *src_inst = RuntimeImpl::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstanceImpl *dst_inst = RuntimeImpl::get_runtime()->get_instance_impl(dsts[0].inst);
          size_t offset = dsts[0].offset;
          size_t size = dsts[0].size;
          size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
          size_t bytes = 
#endif
            find_field(dst_inst->get_field_sizes(), offset, size,
                       field_start, field_size, within_field);
	  assert(bytes == size);
          for (Domain::DomainPointIterator dpi(domain); dpi; dpi++) {
            DomainPoint dp = dpi.p;
            void *src_ptr = src_inst->get_address(src_inst->get_linearization().get_image(dp), 0, redop->sizeof_rhs, 0);
            void *dst_ptr = dst_inst->get_address(dst_inst->get_linearization().get_image(dp),
                                                  field_start, field_size, within_field);
            redop->apply(dst_ptr, src_ptr, 1, false/*exclusive*/);
          }
        }
      protected:
        std::vector<Domain::CopySrcDstField> srcs;
        std::vector<Domain::CopySrcDstField> dsts;
        const ReductionOpUntyped *redop;
      };
    };

    DMAOperation::DMAOperation(const Realm::ProfilingRequestSet &reqs)
      : requests(reqs)
    {
      measurements.import_requests(requests);
      capture_timeline = measurements.wants_measurement<
                          Realm::ProfilingMeasurements::OperationTimeline>(); 
      capture_usage = measurements.wants_measurement<
                          Realm::ProfilingMeasurements::OperationMemoryUsage>();
      if (capture_timeline)
        timeline.record_create_time();
    }

    DMAOperation::~DMAOperation(void)
    {
      if (requests.request_count() > 0) {
        if (capture_timeline)
          measurements.add_measurement(timeline);
        if (capture_usage)
          measurements.add_measurement(usage);
        measurements.send_responses(requests);
      }
    }

    bool DMAOperation::event_triggered(void)
    {
      RuntimeImpl::get_dma_queue()->enqueue_dma(this);
      // Don't delete yet, we still have to do the operation
      return false;
    }

    Event CopyOperation::register_copy(Event wait_on)
    {
      Event result = done_event->get_event();
      if (wait_on.exists()) {
        EventImpl *event_impl = RuntimeImpl::get_runtime()->get_event_impl(wait_on);
        event_impl->add_waiter(wait_on.gen, this);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(this);
      }
      return result;
    }

    void CopyOperation::perform(void)
    {
      DetailedTimer::ScopedPush sp(TIME_COPY); 
      // A little bit of a hack for the shared lowlevel profiling
      if (capture_usage && !srcs.empty() && !dsts.empty()) {
        usage.source = srcs[0].inst.get_location();
        usage.target = dsts[0].inst.get_location();
      }

      if (redop_id == 0)
      {
        RangeExecutors::GatherScatter rexec(srcs, dsts);

        if(domain.get_dim() == 0) {
          // This is an index space copy
          IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(domain.get_index_space());
          const ElementMask& mask = r->get_element_mask();
          ElementMask::forall_ranges(rexec, mask, mask);
        } else {
          rexec.do_domain(domain);
        }
      }
      else // This is a reduction operation
      {
        // Get the reduction operation that we are doing
        const ReductionOpUntyped *redop = RuntimeImpl::get_runtime()->get_reduction_op(redop_id);
        // See if we're doing a fold or not 
        if (red_fold)
        {
          RangeExecutors::ReductionFold rexec(srcs,dsts,redop);
          if (domain.get_dim() == 0) {
            IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(domain.get_index_space());
            const ElementMask& mask = r->get_element_mask();
            ElementMask::forall_ranges(rexec, mask, mask);
          } else {
            rexec.do_domain(domain);
          }
        }
        else
        {
          RangeExecutors::ReductionApply rexec(srcs,dsts,redop);
          if (domain.get_dim() == 0) {
            IndexSpaceImpl *r = RuntimeImpl::get_runtime()->get_metadata_impl(domain.get_index_space());
            const ElementMask& mask = r->get_element_mask();
            ElementMask::forall_ranges(rexec, mask, mask);
          } else {
            rexec.do_domain(domain);
          }
        }
      }
      // Trigger the event indicating that we are done
      done_event->trigger();
    }

    void FillOperation::perform(void)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      for (std::vector<Domain::CopySrcDstField>::const_iterator it = 
            dsts.begin(); it != dsts.end(); it++)
      {
        RegionInstanceImpl *impl = rt->get_instance_impl(it->inst); 
        impl->fill_field(it->offset, it->size, fill_value,
                         fill_value_size, domain);
      }
      done_event->trigger();
    }

    void ComputeIndexSpaces::perform(void)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      for (std::vector<IndexSpace::BinaryOpDescriptor>::const_iterator it = 
            pairs.begin(); it != pairs.end(); it++)
      {
        IndexSpaceImpl *target = rt->get_metadata_impl(it->result);
        IndexSpaceImpl *left = rt->get_metadata_impl(it->left_operand);
        IndexSpaceImpl *right = rt->get_metadata_impl(it->right_operand);
        ElementMask &target_mask = target->get_element_mask();
        const ElementMask &left_mask = left->get_element_mask();
        const ElementMask &right_mask = right->get_element_mask();
        switch (it->op)
        {
          case IndexSpace::ISO_UNION:
            {
              target_mask = left_mask | right_mask;
              break;
            }
          case IndexSpace::ISO_INTERSECT:
            {
              target_mask = left_mask & right_mask;
              break;
            }
          case IndexSpace::ISO_SUBTRACT:
            {
              target_mask = left_mask - right_mask;
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
      done_event->trigger();
    }

    void ComputeIndexSpaces::print_info(FILE *f)
    {
      Event e = done_event->get_event();
      fprintf(f,"deferred compute index spaces: after=" IDFMT "/%d\n",
              e.id, e.gen+1);
    }

    void ReduceIndexSpaces::perform(void)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      ElementMask &target = result->get_element_mask();
      switch (op)
      {
        case IndexSpace::ISO_UNION:
          {
            for (std::vector<IndexSpace>::const_iterator it = 
                  spaces.begin(); it != spaces.end(); it++)
            {
              IndexSpaceImpl *space = rt->get_metadata_impl(*it);
              const ElementMask &mask = space->get_element_mask();
              target |= mask;
            }
            break;
          }
        case IndexSpace::ISO_INTERSECT:
          {
            assert(!spaces.empty());
            IndexSpaceImpl *space = rt->get_metadata_impl(spaces[0]);
            target = space->get_element_mask();
            for (unsigned idx = 1; idx < spaces.size(); idx++)
            {
              space = rt->get_metadata_impl(spaces[idx]);
              const ElementMask &mask = space->get_element_mask();
              target &= mask;
            }
            break;
          }
        case IndexSpace::ISO_SUBTRACT:
          {
            assert(!spaces.empty());
            IndexSpaceImpl *space = rt->get_metadata_impl(spaces[0]);
            target = space->get_element_mask();
            for (unsigned idx = 1; idx < spaces.size(); idx++)
            {
              space = rt->get_metadata_impl(spaces[idx]);
              const ElementMask &mask = space->get_element_mask();
              target -= mask;
            }
            break;
          }
        default:
          assert(false); // should never get here
      }
      done_event->trigger();
    }

    void ReduceIndexSpaces::print_info(FILE *f)
    {
      Event e = done_event->get_event();
      fprintf(f,"deferred reduce index spaces: after=" IDFMT "/%d\n",
              e.id, e.gen+1);
    }

    void DeferredEqualSpaces::perform(void)
    {
      target->create_equal_subspaces(subspaces, granularity);
      done_event->trigger();
    }

    void DeferredEqualSpaces::print_info(FILE *f)
    {
      Event e = done_event->get_event();
      fprintf(f,"deferred create equal subspaces: after=" IDFMT "/%d\n",
              e.id, e.gen+1);
    }

    void DeferredWeightedSpaces::perform(void)
    {
      target->create_weighted_subspaces(subspaces, granularity, weights);
      done_event->trigger();
    }

    void DeferredWeightedSpaces::print_info(FILE *f)
    {
      Event e = done_event->get_event();
      fprintf(f,"deferred create weighted subspaces: after=" IDFMT "/%d\n",
              e.id, e.gen+1);
    }

    void DeferredFieldSpaces::perform(void)
    {
      target->create_subspaces_by_field(field_data, subspaces);
      done_event->trigger();
    }

    void DeferredFieldSpaces::print_info(FILE *f)
    {
      Event e = done_event->get_event();
      fprintf(f,"deferred create subspaces by field: after=" IDFMT "/%d\n",
              e.id, e.gen+1);
    }

    void DeferredImageSpaces::perform(void)
    {
      target->create_subspaces_by_image(field_data, subspaces);
      done_event->trigger();
    }

    void DeferredImageSpaces::print_info(FILE *f)
    {
      Event e = done_event->get_event();
      fprintf(f,"deferred create subspaces by image: after=" IDFMT "/%d\n",
              e.id, e.gen+1);
    }

    void DeferredPreimageSpaces::perform(void)
    {
      target->create_subspaces_by_preimage(field_data, subspaces);
      done_event->trigger();
    }

    void DeferredPreimageSpaces::print_info(FILE *f)
    {
      Event e = done_event->get_event();
      fprintf(f,"deferred create subspaces by preimage: after=" IDFMT "/%d\n",
              e.id, e.gen+1);
    }

    /*static*/
    Event IndexSpaceImpl::fill(const std::vector<Domain::CopySrcDstField> &dsts,
                                 const void *fill_value, size_t fill_value_size,
                                 Event wait_on, const Domain &domain)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_event = rt->get_free_event();
      Event result = done_event->get_event();
      FillOperation *op = new FillOperation(dsts, fill_value, fill_value_size,
                                            domain, done_event);
      if (wait_on.exists()) {
        EventImpl *event_impl = rt->get_event_impl(wait_on);
        event_impl->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    /*static*/
    Event IndexSpaceImpl::fill(const std::vector<Domain::CopySrcDstField> &dsts,
                                 const Realm::ProfilingRequestSet &requests,
                                 const void *fill_value, size_t fill_value_size,
                                 Event wait_on, const Domain &domain)
    {
      RuntimeImpl *rt = RuntimeImpl::get_runtime();
      EventImpl *done_event = rt->get_free_event();
      Event result = done_event->get_event();
      FillOperation *op = new FillOperation(dsts, requests, fill_value, 
                                            fill_value_size, domain, done_event);
      if (wait_on.exists()) {
        EventImpl *event_impl = rt->get_event_impl(wait_on);
        event_impl->add_waiter(wait_on.gen, op);
      } else {
        RuntimeImpl::get_dma_queue()->enqueue_dma(op);
      }
      return result;
    }

    /*static*/
    Event IndexSpaceImpl::copy(RegionInstance src_inst, RegionInstance dst_inst, size_t elem_size,
				 const Domain domain, Event wait_on /*= Event::NO_EVENT*/,
				 ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/)
    {
      std::vector<Domain::CopySrcDstField> srcs, dsts;

      srcs.push_back(Domain::CopySrcDstField(src_inst, 0, elem_size));
      dsts.push_back(Domain::CopySrcDstField(dst_inst, 0, elem_size));

      return copy(srcs, dsts, domain, wait_on, redop_id, red_fold);
    }
    
    /*static*/
    Event IndexSpaceImpl::copy(const std::vector<Domain::CopySrcDstField>& srcs,
				 const std::vector<Domain::CopySrcDstField>& dsts,
				 Domain domain, Event wait_on,
				 ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/)
    {
      EventImpl *done_event = NULL;
      CopyOperation *co = new CopyOperation(srcs, dsts, 
					    domain, //get_element_mask(), get_element_mask(),
					    redop_id, red_fold,
					    done_event);
      return co->register_copy(wait_on);
    }

    /*static*/
    Event IndexSpaceImpl::copy(const std::vector<Domain::CopySrcDstField>& srcs,
                                 const std::vector<Domain::CopySrcDstField>& dsts,
                                 const Realm::ProfilingRequestSet &requests,
                                 const Domain domain, Event wait_on,
                                 ReductionOpID redop_id, bool red_fold)
    {
      EventImpl *done_event = NULL;
      CopyOperation *co = new CopyOperation(srcs, dsts, requests, 
					    domain, redop_id, red_fold,
					    done_event);
      return co->register_copy(wait_on);
    }

    ////////////////////////////////////////////////////////
    // DMA Queue 
    ////////////////////////////////////////////////////////

    DMAQueue::DMAQueue(unsigned num_threads)
      : num_dma_threads(num_threads), dma_shutdown(false)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_init(&dma_lock,NULL));
      PTHREAD_SAFE_CALL(pthread_cond_init(&dma_cond,NULL));
      dma_threads.resize(num_dma_threads);
    }

    void DMAQueue::start(void)
    {
      pthread_attr_t attr;
      PTHREAD_SAFE_CALL(pthread_attr_init(&attr));
      for (unsigned idx = 0; idx < num_dma_threads; idx++)
      {
        PTHREAD_SAFE_CALL(pthread_create(&dma_threads[idx], &attr,
                                         DMAQueue::start_dma_thread, (void*)this));
      }
      PTHREAD_SAFE_CALL(pthread_attr_destroy(&attr));
    }

    void DMAQueue::shutdown(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&dma_lock));
      dma_shutdown = true;
      PTHREAD_SAFE_CALL(pthread_cond_broadcast(&dma_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&dma_lock));
      // Now join on all the threads
      for (unsigned idx = 0; idx < num_dma_threads; idx++)
      {
        void *result;
        PTHREAD_SAFE_CALL(pthread_join(dma_threads[idx],&result));
      }
    }

    void DMAQueue::run_dma_loop(void)
    {
      while (true)
      {
        DMAOperation *op = NULL;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&dma_lock));
        if (ready_ops.empty() && !dma_shutdown)
        {
          // Go to sleep
          PTHREAD_SAFE_CALL(pthread_cond_wait(&dma_cond, &dma_lock));
        }
        // When we wake up see if there is anything
        // to do or see if we are done
        if (!ready_ops.empty())
        {
          op = ready_ops.front();
          ready_ops.pop_front();
        }
        else if (dma_shutdown)
        {
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&dma_lock));
          // Break out of the loop
          break;
        }
        // Release our lock
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&dma_lock));
        // If we have a copy perform it and then delete it
        if (op != NULL)
        {
          if (op->capture_timeline)
            op->timeline.record_start_time();
          op->perform();
          if (op->capture_timeline)
	  {
            op->timeline.record_end_time();
	    op->timeline.record_complete_time();
	  }
          delete op;
        }
      }
    }

    void DMAQueue::enqueue_dma(DMAOperation *op)
    {
      if (op->capture_timeline)
        op->timeline.record_ready_time();
      if (num_dma_threads > 0)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&dma_lock));
        ready_ops.push_back(op);
        PTHREAD_SAFE_CALL(pthread_cond_signal(&dma_cond));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&dma_lock));
      }
      else
      {
        // If we don't have any dma threads, just do the copy now
        if (op->capture_timeline)
          op->timeline.record_start_time();
        op->perform();
        if (op->capture_timeline) 
	{
          op->timeline.record_end_time();
	  op->timeline.record_complete_time();
	}
        delete op;
      }
    }

    /*static*/ void* DMAQueue::start_dma_thread(void *args)
    {
      DMAQueue *dma_queue = (DMAQueue*)args;
      dma_queue->run_dma_loop();
      pthread_exit(NULL);
    }

#ifdef LEGION_BACKTRACE
    static void legion_backtrace(int signal)
    {
      assert((signal == SIGTERM) || (signal == SIGINT) || 
             (signal == SIGABRT) || (signal == SIGSEGV) ||
             (signal == SIGFPE));
      void *bt[256];
      int bt_size = backtrace(bt, 256);
      char **bt_syms = backtrace_symbols(bt, bt_size);
      size_t buffer_size = 2048; // default buffer size
      char *buffer = (char*)malloc(buffer_size);
      size_t offset = 0;
      size_t funcnamesize = 256;
      char *funcname = (char*)malloc(funcnamesize);
      for (int i = 0; i < bt_size; i++) {
        // Modified from https://panthema.net/2008/0901-stacktrace-demangled/ under WTFPL 2.0
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;
        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = bt_syms[i]; *p; ++p) {
          if (*p == '(')
            begin_name = p;
          else if (*p == '+')
            begin_offset = p;
          else if (*p == ')' && begin_offset) {
            end_offset = p;
            break;
          }
        }
        // If offset is within half of the buffer size, double the buffer
        if (offset >= (buffer_size / 2)) {
          buffer_size *= 2;
          buffer = (char*)realloc(buffer, buffer_size);
        }
        if (begin_name && begin_offset && end_offset &&
            (begin_name < begin_offset)) {
          *begin_name++ = '\0';
          *begin_offset++ = '\0';
          *end_offset = '\0';
          // mangled name is now in [begin_name, begin_offset) and caller
          // offset in [begin_offset, end_offset). now apply __cxa_demangle():
          int status;
          char* demangled_name = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
          if (status == 0) {
            funcname = demangled_name; // use possibly realloc()-ed string
            offset += snprintf(buffer+offset,buffer_size-offset,
                               "  %s : %s+%s\n", bt_syms[i], funcname, begin_offset);
          } else {
            // demangling failed. Output function name as a C function with no arguments.
            offset += snprintf(buffer+offset,buffer_size-offset,
                               "  %s : %s()+%s\n", bt_syms[i], begin_name, begin_offset);
          }
        } else {
          // Who knows just print the whole line
          offset += snprintf(buffer+offset,buffer_size-offset,
                             "%s\n",bt_syms[i]);
        }
      }
      fprintf(stderr,"BACKTRACE\n----------\n%s\n----------\n", buffer);
      fflush(stderr);
      free(buffer);
      free(funcname);
    }
#endif

    static void legion_freeze(int signal)
    {
      assert((signal == SIGINT) || (signal == SIGABRT) ||
             (signal == SIGSEGV) || (signal == SIGFPE));
      int process_id = getpid(); 
      fprintf(stderr,"Legion process received signal %d: %s\n",
                      signal, strsignal(signal));
      fprintf(stderr,"Process %d is frozen!\n", process_id);
      fflush(stderr);
      while (true)
        sleep(1);
    }

    ////////////////////////////////////////////////////////
    // Machine 
    ////////////////////////////////////////////////////////

    struct MachineRunArgs {
      RuntimeImpl *r;
      Processor::TaskFuncID task_id;
      Runtime::RunStyle style;
      const void *args;
      size_t arglen;
    };  

    static void *background_run_thread(void *data)
    {
      MachineRunArgs *args = (MachineRunArgs *)data;
      args->r->run(args->task_id, args->style, args->args, args->arglen,
		   false /* foreground from this thread's perspective */);
      delete args;
      return 0;
    }

    Logger::Category log_machine("machine");

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    size_t Machine::get_address_space_count(void) const
    {
        return 1;
    }

    /*static*/ Machine Machine::get_machine(void)
    {
      return Machine(RuntimeImpl::get_runtime()->machine);
    }

    void Machine::get_all_memories(std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_all_memories(mset);
    }
    
    void Machine::get_all_processors(std::set<Processor>& pset) const
    {
      return ((MachineImpl *)impl)->get_all_processors(pset);
    }

    // Return the set of memories visible from a processor
    void Machine::get_visible_memories(Processor p, std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(p, mset);
    }

    // Return the set of memories visible from a memory
    void Machine::get_visible_memories(Memory m, std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(m, mset);
    }

    // Return the set of processors which can all see a given memory
    void Machine::get_shared_processors(Memory m, std::set<Processor>& pset) const
    {
      return ((MachineImpl *)impl)->get_shared_processors(m, pset);
    }

    int Machine::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				       Processor restrict_proc /*= Processor::NO_PROC*/,
				       Memory restrict_memory /*= Memory::NO_MEMORY*/) const
    {
      return ((MachineImpl *)impl)->get_proc_mem_affinity(result, restrict_proc, restrict_memory);
    }

    int Machine::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
				      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
				      Memory restrict_mem2 /*= Memory::NO_MEMORY*/) const
    {
      return ((MachineImpl *)impl)->get_mem_mem_affinity(result, restrict_mem1, restrict_mem2);
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    void MachineImpl::get_all_memories(std::set<Memory>& mset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	mset.insert((*it).m);
      }
    }

    void MachineImpl::get_all_processors(std::set<Processor>& pset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	pset.insert((*it).p);
      }
    }

    // Return the set of memories visible from a processor
    void MachineImpl::get_visible_memories(Processor p, std::set<Memory>& mset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if((*it).p == p)
	  mset.insert((*it).m);
      }
    }

    // Return the set of memories visible from a memory
    void MachineImpl::get_visible_memories(Memory m, std::set<Memory>& mset) const
    {
      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if((*it).m1 == m)
	  mset.insert((*it).m2);

	if((*it).m2 == m)
	  mset.insert((*it).m1);
      }
    }

    // Return the set of processors which can all see a given memory
    void MachineImpl::get_shared_processors(Memory m, std::set<Processor>& pset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if((*it).m == m)
	  pset.insert((*it).p);
      }
    }

    int MachineImpl::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
					     Processor restrict_proc /*= Processor::NO_PROC*/,
					     Memory restrict_memory /*= Memory::NO_MEMORY*/) const
    {
      int count = 0;

      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if(restrict_proc.exists() && ((*it).p != restrict_proc)) continue;
	if(restrict_memory.exists() && ((*it).m != restrict_memory)) continue;
	result.push_back(*it);
	count++;
      }

      return count;
    }

    int MachineImpl::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
					    Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
					    Memory restrict_mem2 /*= Memory::NO_MEMORY*/) const
    {
      // Handle the case for same memories
      if (restrict_mem1.exists() && (restrict_mem1 == restrict_mem2))
      {
	Machine::MemoryMemoryAffinity affinity;
        affinity.m1 = restrict_mem1;
        affinity.m2 = restrict_mem1;
        affinity.bandwidth = 100;
        affinity.latency = 1;
        result.push_back(affinity);
        return 1;
      }

      int count = 0;

      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if(restrict_mem1.exists() && 
	   ((*it).m1 != restrict_mem1)) continue;
	if(restrict_mem2.exists() && 
	   ((*it).m2 != restrict_mem2)) continue;
	result.push_back(*it);
	count++;
      }

      return count;
    }

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    ////////////////////////////////////////////////////////
    // Runtime 
    ////////////////////////////////////////////////////////

    Runtime::Runtime(void)
      : impl(0)
    {
      // ok to construct extra ones - we will make sure only one calls init() though
    }

    /*static*/ Runtime Runtime::get_runtime(void)
    {
      Runtime r;
      r.impl = RuntimeImpl::get_runtime();
      return r;
    }

    bool Runtime::init(int *argc, char ***argv)
    {
      if(RuntimeImpl::get_runtime() != 0) {
	fprintf(stderr, "ERROR: cannot initialize more than one runtime at a time!\n");
	return false;
      }

      MachineImpl *m = new MachineImpl;
      impl = new RuntimeImpl(m);
      return ((RuntimeImpl *)impl)->init(argc, argv);
    }
    
    bool Runtime::register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr)
    {
      assert(impl != 0);

      return ((RuntimeImpl *)impl)->register_task(taskid, taskptr, 0, 0);
    }

    bool Runtime::register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->redop_table.count(redop_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->redop_table[redop_id] = redop;
      return true;
    }

    bool Runtime::register_custom_serdez(CustomSerdezID serdez_id, const CustomSerdezUntyped *serdez)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->custom_serdez_table.count(serdez_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->custom_serdez_table[serdez_id] = serdez;
      return true;
    }

    void Runtime::run(Processor::TaskFuncID task_id /*= 0*/,
	              RunStyle style /*= ONE_TASK_ONLY*/,
   	              const void *args /*= 0*/, size_t arglen /*= 0*/,
		      bool background /*= false*/)
    {
      assert(impl != 0);
      ((RuntimeImpl *)impl)->run(task_id, style, args, arglen, background);
    }

    void Runtime::shutdown(void)
    {
      assert(impl != 0);
      ((RuntimeImpl *)impl)->shutdown();
    }

    void Runtime::wait_for_shutdown(void)
    {
      assert(impl != 0);
      ((RuntimeImpl *)impl)->wait_for_shutdown();

      // delete the impl once it's shut down
      RuntimeImpl::runtime = 0;
      delete ((RuntimeImpl *)impl);
      impl = 0;
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    RuntimeImpl::RuntimeImpl(MachineImpl *m)
      : machine(m), background_pthread(0)
    {
	for (unsigned i=0; i<BASE_EVENTS; i++)
        {
            EventImpl *event = new EventImpl(i);
            events.push_back(event);
            if (i != 0) // Don't hand out the NO_EVENT event
              free_events.push_back(event);
        }

	for (unsigned i=0; i<BASE_RESERVATIONS; i++)
        {
		reservations.push_back(new ReservationImpl(i));
                if (i != 0)
                  free_reservations.push_back(reservations.back());
        }

	for (unsigned i=0; i<BASE_METAS; i++)
	{
		metadatas.push_back(new IndexSpaceImpl(i,0,0));
                if (i != 0)
                  free_metas.push_back(metadatas.back());
	}

	for (unsigned i=0; i<BASE_INSTANCES; i++)
	{
		Memory m;
		m.id = 0;
		instances.push_back(new RegionInstanceImpl(i,
							     m,
							     0,
                                                             0,
							     std::vector<size_t>(),
							     0,
							     0,
							     DomainLinearization()));
                if (i != 0)
                  free_instances.push_back(instances.back());
	}

	PTHREAD_SAFE_CALL(pthread_rwlock_init(&event_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_event_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&reservation_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_reservation_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&proc_group_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&metadata_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_metas_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&allocator_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_alloc_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&instance_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_inst_lock,NULL));
    }

    bool RuntimeImpl::register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr,
				    const void *userdata, size_t userlen)
    {
      if (taskid == 0)
      {
	fprintf(stderr,"Using task_id 0 in the task table is illegal!  Task_id 0 is the shutdown task\n");
	fflush(stderr);
	exit(1);
      }

      TaskTable::iterator it = task_table.find(taskid);
      if(it != task_table.end()) {
	// ignore re-registration of the same task
	if(it->second.func_ptr != taskptr) {
	  fprintf(stderr, "Attempt to change function pointer for task %d\n", taskid);
	  assert(0);
	  exit(1);
	}
	if((it->second.userlen != userlen) ||
	   (userlen && memcmp(it->second.userdata, userdata, userlen))) {
	  fprintf(stderr, "Attempt to change user data for task %d\n", taskid);
	  assert(0);
	  exit(1);
	}
	return true;
      }

      TaskTableEntry& tte = task_table[taskid];
      tte.func_ptr = taskptr;
      if(userlen) {
	tte.userdata = malloc(userlen);
	memcpy(tte.userdata, userdata, userlen);
	tte.userlen = userlen;
      } else {
	tte.userdata = 0;
	tte.userlen = 0;
      }

      return true;
    }

    bool RuntimeImpl::init(int *argc, char ***argv)
    {
        // make all timestamps use relative time from now
        Realm::Clock::set_zero_time();

        unsigned num_cpus = NUM_PROCS;
        unsigned num_utility_cpus = NUM_UTIL_PROCS;
        unsigned num_dma_threads = NUM_DMA_THREADS;
        size_t cpu_mem_size_in_mb = GLOBAL_MEM;
        size_t cpu_l1_size_in_kb = LOCAL_MEM;
        size_t cpu_stack_size = STACK_SIZE;

#ifdef DEBUG_PRINT
	PTHREAD_SAFE_CALL(pthread_mutex_init(&debug_mutex,NULL));
#endif
        // Create the pthread keys for thread local data
        PTHREAD_SAFE_CALL( pthread_key_create(&local_thread_key, NULL) );
        DetailedTimer::init_timers();

        for (int i=1; i < *argc; i++)
        {
#define INT_ARG(argname, varname) do { \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = atoi((*argv)[++i]);		\
	    continue;					\
	  } } while(0)
          
          INT_ARG("-ll:csize", cpu_mem_size_in_mb);
          INT_ARG("-ll:l1size", cpu_l1_size_in_kb);
          INT_ARG("-ll:cpu", num_cpus);
          INT_ARG("-ll:util", num_utility_cpus);
          INT_ARG("-ll:dma", num_dma_threads);
          INT_ARG("-ll:stack",cpu_stack_size);
#undef INT_ARG
        }
        cpu_stack_size = cpu_stack_size * (1 << 20);

        if (num_utility_cpus > num_cpus)
        {
            fprintf(stderr,"The number of processor groups (%d) cannot be "
                    "greater than the number of cpus (%d)\n",
                    num_utility_cpus,num_cpus);
            fflush(stderr);
            exit(1);
        }

	// Create the runtime and initialize with this machine
	runtime = this;
        dma_queue = new DMAQueue(num_dma_threads);

        // Initialize the logger
	// this now wants std::vector<std::string> instead of argc/argv
	{
	  std::vector<std::string> cmdline(*argc - 1);
	  for(int i = 1; i < *argc; i++)
	    cmdline[i - 1] = (*argv)[i];
	  Realm::Logger::configure_from_cmdline(cmdline);
	}
	
        // Fill in the tables
        // find in proc 0 with NULL
        processors.push_back(NULL);
#ifndef __MACH__
        pthread_barrier_t *init_barrier = 
          (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
        PTHREAD_SAFE_CALL(pthread_barrier_init(init_barrier,NULL,
                                    (num_cpus+num_utility_cpus)));
#else
        pthread_barrier_t *init_barrier = 
          new pthread_barrier_t(num_cpus+num_utility_cpus);
#endif
        if (num_utility_cpus > 0)
        {
          // This is the case where we have actual utility processors  

          // Keep track of the number of users of each utility cpu
          std::vector<ProcessorImpl*> temp_utils(num_utility_cpus);
          for (unsigned idx = 0; idx < num_utility_cpus; idx++)
          {
            Processor p;
            p.id = num_cpus+idx+1;
            procs.insert(p);
            temp_utils[idx] = new ProcessorImpl(init_barrier, task_table, 
                                p, cpu_stack_size, Processor::UTIL_PROC);
          }
          // Now we can make the processors themselves
          for (unsigned idx = 0; idx < num_cpus; idx++)
          {
            Processor p;
            p.id = idx + 1;
            procs.insert(p);
            // Figure out which utility processor this guy gets
#ifdef SPECIALIZED_UTIL_PROCS
            unsigned utility_idx = 0;
#else
            unsigned utility_idx = idx % num_utility_cpus;
#endif
#ifdef DEBUG_REALM
            assert(utility_idx < num_utility_cpus);
#endif
            ProcessorImpl *impl = new ProcessorImpl(init_barrier, task_table, p, 
                                        cpu_stack_size, temp_utils[utility_idx]);
            // Add this processor as utility user
            processors.push_back(impl);
          }
          // Finally we can add the utility processors to the set of processors
          for (unsigned idx = 0; idx < num_utility_cpus; idx++)
          {
            processors.push_back(temp_utils[idx]);
          }
        }
        else
        {
          // This is the case where everyone processor is its own utility processor
          for (unsigned idx = 0; idx < num_cpus; idx++)
          {
            Processor p;
            p.id = idx + 1;
            procs.insert(p);
            ProcessorImpl *impl = new ProcessorImpl(init_barrier, task_table, p, 
                                            cpu_stack_size, Processor::LOC_PROC);
            processors.push_back(impl);
          }
        }
	
        if (cpu_mem_size_in_mb > 0)
	{
                // Make the first memory null
	        memories.push_back(NULL);
                // Do the global memory
		//Memory global;
		//global.id = 1;
		//memories.insert(global);
		MemoryImpl *impl = new MemoryImpl(cpu_mem_size_in_mb*1024*1024, Memory::SYSTEM_MEM);
		memories.push_back(impl);
	}
        else
        {
                fprintf(stderr,"SYSTEM MEMORY is not allowed to be empty "
                        "for the shared low-level runtime. Use '-ll:csize' "
                        "to adjust the memory to a positive value.\n");
                fflush(stderr);
                exit(1);
        }
        if (cpu_l1_size_in_kb > 0)
        {
          for (unsigned id=2; id<=(num_cpus+1); id++)
          {
	          //Memory m;
                  //m.id = id;
                  //memories.insert(m);
                  MemoryImpl *impl = new MemoryImpl(cpu_l1_size_in_kb*1024, Memory::LEVEL1_CACHE);
                  memories.push_back(impl);
          }
        }

        // Now set up the affinities for each of the different processors and memories
        for (std::set<Processor>::iterator it = procs.begin(); it != procs.end(); it++)
        {
          // Give all processors 32 GB/s to the global memory
          if (cpu_mem_size_in_mb > 0)
          {
	    Machine::ProcessorMemoryAffinity global_affin;
            global_affin.p = *it;
            global_affin.m.id = 1;
            global_affin.bandwidth = 32;
            global_affin.latency = 50; /* higher latency */
            machine->proc_mem_affinities.push_back(global_affin);
          }
          // Give the processor good affinity to its L1, but not to other L1
          if (cpu_l1_size_in_kb > 0)
          {
            for (unsigned id = 2; id <= (num_cpus+1); id++)
            {
              if (id == (it->id+1))
              {
                // Our L1, high bandwidth with low latency
                Machine::ProcessorMemoryAffinity local_affin;
                local_affin.p = *it;
                local_affin.m.id = id;
                local_affin.bandwidth = 100;
                local_affin.latency = 1; /* small latency */
                machine->proc_mem_affinities.push_back(local_affin);
              }
              else
              {
                // Other L1, low bandwidth with long latency
                Machine::ProcessorMemoryAffinity other_affin;
                other_affin.p = *it;
                other_affin.m.id = id;
                other_affin.bandwidth = 10;
                other_affin.latency = 100; /* high latency */
                machine->proc_mem_affinities.push_back(other_affin);
              }
            }
          }
        }
        // Set up the affinities between the different memories
        {
          // Global to all others
          if ((cpu_mem_size_in_mb > 0) && (cpu_l1_size_in_kb > 0))
          {
            for (unsigned id = 2; id <= (num_cpus+1); id++)
            {
              Machine::MemoryMemoryAffinity global_affin;
              global_affin.m1.id = 1;
              global_affin.m2.id = id;
              global_affin.bandwidth = 32;
              global_affin.latency = 50;
              machine->mem_mem_affinities.push_back(global_affin);
            }
          }

          // From any one to any other one
          if (cpu_l1_size_in_kb > 0)
          {
            for (unsigned id = 2; id <= (num_cpus+1); id++)
            {
              for (unsigned other=id+1; other <= (num_cpus+1); other++)
              {
                Machine::MemoryMemoryAffinity pair_affin;
                pair_affin.m1.id = id;
                pair_affin.m2.id = other;
                pair_affin.bandwidth = 10;
                pair_affin.latency = 100;
                machine->mem_mem_affinities.push_back(pair_affin);
              }
            }
          }
        }
	// Now start the threads for each of the processors
	// except for processor 0 which is this thread
#ifdef DEBUG_REALM
        assert(processors.size() == (num_cpus+num_utility_cpus+1));
#endif
		
#ifdef LEGION_BACKTRACE
        signal(SIGSEGV, legion_backtrace);
        signal(SIGTERM, legion_backtrace);
        signal(SIGINT, legion_backtrace);
        signal(SIGABRT, legion_backtrace);
        signal(SIGFPE, legion_backtrace);
#endif
        if (getenv("LEGION_FREEZE_ON_ERROR") != NULL)
        {
          signal(SIGSEGV, legion_freeze);
          signal(SIGINT,  legion_freeze);
          signal(SIGABRT, legion_freeze);
          signal(SIGFPE,  legion_freeze);
        }

	return true; // successful initialization
    }

    void RuntimeImpl::run(Processor::TaskFuncID task_id /*= 0*/,
			  Runtime::RunStyle style /*= ONE_TASK_ONLY*/,
			  const void *args /*= 0*/, size_t arglen /*= 0*/,
			  bool background /*= false*/)
    { 

      if(background) {
        log_machine.info("background operation requested\n");
	fflush(stdout);
	MachineRunArgs *margs = new MachineRunArgs;
	margs->r = this;
	margs->task_id = task_id;
	margs->style = style;
	margs->args = args;
	margs->arglen = arglen;
	
	pthread_t *threadp = (pthread_t*)malloc(sizeof(pthread_t));
	pthread_attr_t attr;
	PTHREAD_SAFE_CALL( pthread_attr_init(&attr) );
	PTHREAD_SAFE_CALL( pthread_create(threadp, &attr, 
                            &background_run_thread, (void *)margs) );
	PTHREAD_SAFE_CALL( pthread_attr_destroy(&attr) );
        // Save this pointer in the background thread
        background_pthread = threadp;
	return;
      }

      // Start the threads for each of the processors (including the utility processors)
      for (unsigned id=1; id < processors.size(); id++)
        processors[id]->start_processor();
      dma_queue->start();

      if(task_id != 0) { // no need to check ONE_TASK_ONLY here, since 1 node
	for(int id = 1; id <= NUM_PROCS; id++) {
	  Processor p;
          p.id = static_cast<id_t>(id);
	  p.spawn(task_id,args,arglen);
	  if(style != Runtime::ONE_TASK_PER_PROC) break;
	}
      }
      
      // Poll the processors to see if they are done
      while (true)
      {
        size_t idle_processors = 1; // 0 proc is always done
        for (unsigned id=1; id < processors.size(); id++)
          if (processors[id]->is_idle())
            idle_processors++;
        if (idle_processors == processors.size())
          break;
        // Otherwise sleep for a whole second
        sleep(1);
      }
      for (unsigned id=1; id < processors.size(); id++)
        processors[id]->shutdown_processor();
      dma_queue->shutdown();

      // Once we're done with this, then we can exit with a successful error code
      exit(0);
    }

    void RuntimeImpl::shutdown(void)
    {
      for (std::set<Processor>::const_iterator it = procs.begin();
	   it != procs.end(); it++)
      {
	// Kill pill
	it->spawn(0, NULL, 0);
      }
    }

    void RuntimeImpl::wait_for_shutdown(void)
    {
      if (background_pthread != NULL)
      {
        pthread_t *background_thread = (pthread_t*)background_pthread;
        void *result;
        PTHREAD_SAFE_CALL(pthread_join(*background_thread, &result));
        free(background_thread);
        // Set this to null so we don't need to do wait anymore
        background_pthread = NULL;
      }
    }

    EventImpl* RuntimeImpl::get_event_impl(Event e)
    {
        EventImpl::EventIndex i = e.id;
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&event_lock));
#ifdef DEBUG_REALM
	assert(i != 0);
	assert(i < events.size());
#endif
        EventImpl *result = events[i];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
	return result;
    }

    void RuntimeImpl::free_event(EventImpl *e)
    {
      // Put this event back on the list of free events
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_event_lock));
      free_events.push_back(e);
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_event_lock));
    }

    void RuntimeImpl::print_event_waiters(void)
    {
      // No need to hold the lock here since we'll only
      // ever call this method from the debugger
      for (unsigned idx = 0; idx < events.size(); idx++)
      {
        events[idx]->print_waiters();
      }
    }

    ReservationImpl* RuntimeImpl::get_reservation_impl(Reservation r)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&reservation_lock));
#ifdef DEBUG_REALM
	assert(r.id != 0);
	assert(r.id < reservations.size());
#endif
        ReservationImpl *result = reservations[r.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&reservation_lock));
	return result;
    }

    void RuntimeImpl::free_reservation(ReservationImpl *r)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_reservation_lock));
      free_reservations.push_back(r);
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_reservation_lock));
    }

    MemoryImpl* RuntimeImpl::get_memory_impl(Memory m)
    {
	if (m.id < memories.size())
		return memories[m.id];
	else
        {
                assert(false);
		return NULL;
        }
    }

    ProcessorImpl* RuntimeImpl::get_processor_impl(Processor p)
    {
      if(p.id >= ProcessorGroup::FIRST_PROC_GROUP_ID) {
	IDType id = p.id - ProcessorGroup::FIRST_PROC_GROUP_ID;
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&proc_group_lock));
#ifdef DEBUG_REALM
	assert(id < proc_groups.size());
#endif
	ProcessorGroup *grp = proc_groups[id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&proc_group_lock));
	return grp;
      }

#ifdef DEBUG_REALM
        assert(p.exists());
	assert(p.id < processors.size());
#endif
	return processors[p.id];
    }

    IndexSpaceImpl* RuntimeImpl::get_metadata_impl(IndexSpace m)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&metadata_lock));
#ifdef DEBUG_REALM
	assert(m.id != 0);
	assert(m.id < metadatas.size());
#endif
        IndexSpaceImpl *result = metadatas[m.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
	return result;
    }

    void RuntimeImpl::free_metadata(IndexSpaceImpl *impl)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        free_metas.push_back(impl);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
    }

    RegionInstanceImpl* RuntimeImpl::get_instance_impl(RegionInstance i)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&instance_lock));
#ifdef DEBUG_REALM
	assert(i.id != 0);
	assert(i.id < instances.size());
#endif
        RegionInstanceImpl *result = instances[i.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
	return result;
    }

    void RuntimeImpl::free_instance(RegionInstanceImpl *impl)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_inst_lock));
        free_instances.push_back(impl);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_inst_lock));
    }

    EventImpl* RuntimeImpl::get_free_event()
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_event_lock));
        if (!free_events.empty())
        {
          EventImpl *result = free_events.front();
          free_events.pop_front();
          // Release the lock
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_event_lock));
          // Activate this event
          bool activated = result->activate();
#ifdef DEBUG_REALM
          assert(activated);
#else
	  (void)activated; // eliminate compiler warning
#endif
          return result;
        }
        // We weren't able to get a new event, get the writer lock
        // for the vector of event implementations and add some more
        PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&event_lock));
        unsigned index = events.size();
        EventImpl *result = new EventImpl(index,true);
        events.push_back(result);
        // Make a whole bunch of other events while we're here
        for (unsigned idx=1; idx < BASE_EVENTS; idx++)
        {
          EventImpl *temp = new EventImpl(index+idx,false);
          events.push_back(temp);
          free_events.push_back(temp);
        }
        // Release the lock on events
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
        // Release the lock on free events
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_event_lock));
        return result;
    }

    ReservationImpl* RuntimeImpl::get_free_reservation(size_t data_size/*= 0*/)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_reservation_lock));
        if (!free_reservations.empty())
        {
          ReservationImpl *result = free_reservations.front();
          free_reservations.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_reservation_lock));
          bool activated = result->activate(data_size);
#ifdef DEBUG_REALM
          assert(activated);
#else
	  (void)activated; // eliminate compiler warning
#endif
          return result;
        }
        // We weren't able to get a new event, get the writer lock
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&reservation_lock));
	unsigned index = reservations.size();
	reservations.push_back(new ReservationImpl(index,true,data_size));
	ReservationImpl *result = reservations[index];
        // Create a whole bunch of other reservations too while we're here
        for (unsigned idx=1; idx < BASE_RESERVATIONS; idx++)
        {
          reservations.push_back(new ReservationImpl(index+idx,false));
          free_reservations.push_back(reservations.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&reservation_lock));	
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_reservation_lock));
	return result;
    }

    ProcessorGroup *RuntimeImpl::get_free_proc_group(const std::vector<Processor>& members)
    {
      // this adds to the list of proc groups, so take the write lock
      PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&proc_group_lock));
      unsigned index = proc_groups.size();
      Processor p;
      p.id = index + ProcessorGroup::FIRST_PROC_GROUP_ID;
      ProcessorGroup *grp = new ProcessorGroup(p);
      proc_groups.push_back(grp);
      PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&proc_group_lock));

      // we can add the members without holding the lock
      for(std::vector<Processor>::const_iterator it = members.begin();
	  it != members.end();
	  it++)
	grp->add_member(get_processor_impl(*it));

      return grp;
    }

    IndexSpaceImpl* RuntimeImpl::get_free_metadata(size_t num_elmts)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          IndexSpaceImpl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(num_elmts);
#ifdef DEBUG_REALM
          assert(activated);
#else
	  (void)activated; // eliminate compiler warning
#endif
          return result;
        }
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new IndexSpaceImpl(index,num_elmts,true));
	IndexSpaceImpl *result = metadatas[index];
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new IndexSpaceImpl(index+idx,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
	return result;
    }

    IndexSpaceImpl* RuntimeImpl::get_free_metadata(const ElementMask &mask)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          IndexSpaceImpl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(mask);
#ifdef DEBUG_REALM
          assert(activated);
#else
	  (void)activated; // eliminate compiler warning
#endif
          return result;
        }
        // Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new IndexSpaceImpl(index,0,false));
	IndexSpaceImpl *result = metadatas[index];
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new IndexSpaceImpl(index+idx,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
        result->activate(mask);
	return result;
    }

    IndexSpaceImpl* RuntimeImpl::get_free_metadata(IndexSpaceImpl *parent)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          IndexSpaceImpl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(parent);
#ifdef DEBUG_REALM
          assert(activated);
#else
	  (void)activated; // eliminate compiler warning
#endif
          return result;
        }
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new IndexSpaceImpl(index,0,false));
	IndexSpaceImpl *result = metadatas[index];
        result->activate(parent);
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new IndexSpaceImpl(index+idx,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
	return result;
    }

    IndexSpaceImpl* RuntimeImpl::get_free_metadata(IndexSpaceImpl *parent, const ElementMask &mask)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          IndexSpaceImpl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(parent,mask);
#ifdef DEBUG_REALM
          assert(activated);
#else
	  (void)activated; // eliminate compiler warning
#endif
          return result;
        }
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new IndexSpaceImpl(index,parent,mask,true));
	IndexSpaceImpl *result = metadatas[index];
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new IndexSpaceImpl(index+idx,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
	return result;
    }


    RegionInstanceImpl* RuntimeImpl::get_free_instance(Memory m, 
                                                     size_t num_elmts, size_t alloc_size,
						     const std::vector<size_t>& field_sizes,
						     size_t elmt_size, size_t block_size,
						     const DomainLinearization& linearization,
						     char *ptr, const ReductionOpUntyped *redop,
						     RegionInstanceImpl *parent,
                                                     const Realm::ProfilingRequestSet &reqs)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_inst_lock));
        if (!free_instances.empty())
        {
          RegionInstanceImpl *result = free_instances.front();
          free_instances.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_inst_lock));
          bool activated = result->activate(m, num_elmts, alloc_size, 
                                            field_sizes, elmt_size, block_size, 
                                            linearization, ptr, redop, parent, reqs);
#ifdef DEBUG_REALM
          assert(activated);
#else
	  (void)activated; // eliminate compiler warning
#endif
          return result;
        }
	// Nothing free so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&instance_lock));
	unsigned int index = instances.size();
	instances.push_back(new RegionInstanceImpl(index, m, num_elmts, alloc_size,
                                                     field_sizes,
						     elmt_size, block_size, linearization,
						     true, ptr, redop, parent));
	RegionInstanceImpl *result = instances[index];
        // Create a whole bunch of other instances while we're here
        for (unsigned idx=1; idx < BASE_INSTANCES; idx++)
        {
          instances.push_back(new RegionInstanceImpl(index+idx,
						       m,
						       0,
                                                       0,
						       std::vector<size_t>(),
						       0,
						       0,
						       DomainLinearization(),
						       false));
          free_instances.push_back(instances.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_inst_lock));
	return result;
    }

    const ReductionOpUntyped* RuntimeImpl::get_reduction_op(ReductionOpID redop)
    {
#ifdef DEBUG_REALM
      assert(redop_table.find(redop) != redop_table.end());
#endif
      return redop_table[redop];
    }

  };

};

namespace LegionRuntime {
  namespace Accessor {
    using namespace LegionRuntime::LowLevel;

    void AccessorType::Generic::Untyped::read_untyped(ptr_t ptr, void *dst, size_t bytes, off_t offset) const
    {
#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      int index = ((impl->get_linearization().get_dim() == 1) ?
		     (int)(impl->get_linearization().get_mapping<1>()->image(ptr.value)) :
		     ptr.value);
      size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
      size_t bytes2 = 
#endif
        find_field(impl->get_field_sizes(), field_offset + offset, bytes,
 		   field_start, field_size, within_field);
      assert(bytes == bytes2);
      const char *src = (const char *)(impl->get_address(index, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    void AccessorType::Generic::Untyped::read_untyped(const DomainPoint& dp, void *dst, size_t bytes, off_t offset) const
    {
#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, dp);
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      int index = impl->get_linearization().get_image(dp);
      size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
      size_t bytes2 = 
#endif
        find_field(impl->get_field_sizes(), field_offset + offset, bytes,
		   field_start, field_size, within_field);
      assert(bytes == bytes2);
      const char *src = (const char *)(impl->get_address(index, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    void AccessorType::Generic::Untyped::write_untyped(ptr_t ptr, const void *src, size_t bytes, off_t offset) const
    {
#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      int index = ((impl->get_linearization().get_dim() == 1) ?
		     (int)(impl->get_linearization().get_mapping<1>()->image(ptr.value)) :
		     ptr.value);
      size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
      size_t bytes2 = 
#endif
        find_field(impl->get_field_sizes(), field_offset + offset, bytes,
		   field_start, field_size, within_field);
      assert(bytes == bytes2);
      char *dst = (char *)(impl->get_address(index, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    void AccessorType::Generic::Untyped::write_untyped(const DomainPoint& dp, const void *src, size_t bytes, off_t offset) const
    {
#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, dp);
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      int index = impl->get_linearization().get_image(dp);
      size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
      size_t bytes2 = 
#endif
        find_field(impl->get_field_sizes(), field_offset + offset, bytes,
    		   field_start, field_size, within_field);
      assert(bytes == bytes2);
      char *dst = (char *)(impl->get_address(index, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    void *AccessorType::Generic::Untyped::raw_span_ptr(ptr_t ptr, size_t req_count, size_t& act_count, ByteOffset& stride)
    {
#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      int index = ((impl->get_linearization().get_dim() == 1) ?
		     (int)(impl->get_linearization().get_mapping<1>()->image(ptr.value)) :
		     ptr.value);
      size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
      size_t bytes = 
#endif
        find_field(impl->get_field_sizes(), field_offset, 0,
		   field_start, field_size, within_field);
      assert(bytes > 0);
      char *dst = (char *)(impl->get_address(index, field_start, field_size, within_field));
      // actual count and stride depend on whether instance is blocked
      if(impl->get_block_size() == 1) {
	// AOS
	stride.offset = impl->get_elmt_size();
	act_count = impl->get_num_elmts() - index;
      } else {
	stride.offset = field_size;
	act_count = impl->get_block_size() - (index % impl->get_block_size());
      }
      return dst;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset *offsets)
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      Arrays::Mapping<DIM, 1> *mapping = impl->get_linearization().get_mapping<DIM>();
      Point<1> strides[DIM];
      int index = mapping->image_linear_subrect(r, subrect, strides);
      // TODO: trim subrect in HybridSOA case
      size_t field_start = 0, field_size = 0, within_field = 0;
#ifndef NDEBUG
      size_t bytes2 = 
#endif
        find_field(impl->get_field_sizes(), field_offset, 1,
                   field_start, field_size, within_field);
      assert(bytes2 == 1);
      char *dst = (char *)(impl->get_address(index, field_start, field_size, within_field));
      for(int i = 0; i < DIM; i++)
	offsets[i].offset = (strides[i][0] * 
			     ((impl->get_block_size() > 1) ? field_size : impl->get_elmt_size()));
      return dst;
    }

    template void *AccessorType::Generic::Untyped::raw_rect_ptr<1>(const Rect<1>& r, Rect<1>& subrect, ByteOffset *offset);
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<2>(const Rect<2>& r, Rect<2>& subrect, ByteOffset *offset);
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<3>(const Rect<3>& r, Rect<3>& subrect, ByteOffset *offset);

    //static const void *(AccessorType::Generic::Untyped::*dummy_ptr)(const Rect<3>&, Rect<3>&, ByteOffset*) = AccessorType::Generic::Untyped::raw_rect_ptr<3>;

    bool AccessorType::Generic::Untyped::get_aos_parameters(void *& base, size_t& stride) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      if (impl->get_block_size() != 1) return false;
      if(base != 0) return false;
      size_t elem_size = impl->get_elmt_size();
      if((stride != 0) && (stride != elem_size)) return false;
      stride = elem_size;
      // Compute the offset based on any trimming
      int index_offset = (impl->get_linearization().get_dim() == 1) ?
                          (int)impl->get_linearization().get_mapping<1>()->image(0) : 0;
      base = ((char *)(impl->get_base_ptr())) + field_offset + (index_offset * elem_size);

      return true;
    }

    bool AccessorType::Generic::Untyped::get_soa_parameters(void *& base, size_t& stride) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      int inst_first_elmt = 0;
      const DomainLinearization& dl = impl->get_linearization();
      if(dl.get_dim() > 0) {
	// make sure this instance uses a 1-D linearization
	assert(dl.get_dim() == 1);

	Arrays::Mapping<1, 1> *mapping = dl.get_mapping<1>();
	Rect<1> image(0, impl->get_num_elmts()-1);
	Rect<1> preimage = mapping->preimage(image.lo);
	assert(preimage.lo == preimage.hi);
	// double-check that whole range maps densely
	preimage.hi.x[0] += impl->get_num_elmts() - 1;
	assert(mapping->image_is_dense(preimage));
	inst_first_elmt = preimage.lo[0];
      }

      // don't handle fixed base addresses yet
      if (base != 0) return false;

      size_t field_start = 0, field_size = 0, within_field = 0;
      find_field(impl->get_field_sizes(), field_offset, 1,
                 field_start, field_size, within_field);

      size_t block_size = impl->get_block_size();

      if(block_size == 1) {
	// AOS, which might be ok if there's only a single field or strides match
	if((impl->get_num_elmts() > 1) &&
	   (stride != 0) && (stride != impl->get_elmt_size()))
	  return false;

	base = (((char *)(impl->get_base_ptr()))
		+ field_offset
		- (impl->get_elmt_size() * inst_first_elmt) // adjust for the first element not being #0
		);

	if(stride == 0)
	  stride = impl->get_elmt_size();
      } else
	if(block_size == impl->get_num_elmts()) {
	  // SOA
	  base = (((char *)(impl->get_base_ptr()))
		  + (field_start * impl->get_block_size())
		  + (field_offset - field_start)
		  - (field_size * inst_first_elmt)  // adjust for the first element not being #0
		  );
	    
	  if ((stride != 0) && (stride != field_size)) return false;
	  stride = field_size;
	} else {
	  // hybrid SOA, we lose
	  return false;
	}

      return true;
    }

    bool AccessorType::Generic::Untyped::get_hybrid_soa_parameters(void *& base, size_t& stride,
								   size_t& block_size, size_t& block_stride) const
    {
      // TODO: implement this
      return false;
    }

    bool AccessorType::Generic::Untyped::get_redfold_parameters(void *& base) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // make sure this is a reduction fold instance
      if(!impl->is_reduction() || impl->is_list_reduction()) return false;

      if(base != 0) return false;
      base = impl->get_base_ptr();

      int inst_first_elmt = 0;
      const DomainLinearization& dl = impl->get_linearization();
      if(dl.get_dim() > 0) {
	// make sure this instance uses a 1-D linearization
	assert(dl.get_dim() == 1);

	Arrays::Mapping<1, 1> *mapping = dl.get_mapping<1>();
	Rect<1> image(0, impl->get_num_elmts()-1);
	Rect<1> preimage = mapping->preimage(image.lo);
	assert(preimage.lo == preimage.hi);
	// double-check that whole range maps densely
	preimage.hi.x[0] += impl->get_num_elmts() - 1;
	assert(mapping->image_is_dense(preimage));
	inst_first_elmt = preimage.lo[0];
	base = ((char *)base) - inst_first_elmt * impl->get_elmt_size();
      }

      return true;
    }

    bool AccessorType::Generic::Untyped::get_redlist_parameters(void *& base, ptr_t *& next_ptr) const
    {
      // TODO: implement this
      return false;
    }

  };

  namespace Arrays {
    //template<> class Mapping<1,1>;
    template <unsigned IDIM, unsigned ODIM>
    MappingRegistry<IDIM, ODIM> Mapping<IDIM, ODIM>::registry;
  };
};
