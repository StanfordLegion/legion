/* Copyright 2015 Stanford University, NVIDIA Corporation
 * Copyright 2015 Los Alamos National Laboratory
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
#endif

#include "realm/operation.h"
#include "realm/dynamic_table.h"
#include "realm/id.h"

#include <assert.h>

#include "activemsg.h"

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

namespace LegionRuntime {
  namespace LowLevel {

    typedef Realm::ID ID;

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

    template <typename LT>
    class AutoLock {
    public:
      AutoLock(LT &mutex) : mutex(mutex), held(true)
      { 
	log_mutex.spew("MUTEX LOCK IN %p", &mutex);
	mutex.lock();
	log_mutex.spew("MUTEX LOCK HELD %p", &mutex);
      }

      ~AutoLock(void) 
      {
	if(held)
	  mutex.unlock();
	log_mutex.spew("MUTEX LOCK OUT %p", &mutex);
      }

      void release(void)
      {
	assert(held);
	mutex.unlock();
	held = false;
      }

      void reacquire(void)
      {
	assert(!held);
	mutex.lock();
	held = true;
      }
    protected:
      LT &mutex;
      bool held;
    };

    typedef AutoLock<GASNetHSL> AutoHSLLock;

    typedef LegionRuntime::HighLevel::BitMask<NODE_MASK_TYPE,MAX_NUM_NODES,
                                              NODE_MASK_SHIFT,NODE_MASK_MASK> NodeMask;

#ifdef USE_REALMS_NODESET
#if MAX_NUM_NODES <= 65536
    typedef DynamicSet<unsigned short> NodeSet;
#else
    // possibly unnecessary future-proofing...
    typedef DynamicSet<unsigned int> NodeSet;
#endif
#else
    typedef LegionRuntime::HighLevel::NodeSet NodeSet;
#endif

    template <class T>
    class Atomic {
    public:
      Atomic(T _value) : value(_value)
      {
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
      GASNetHSL mutex;
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
     
    class EventWaiter {
    public:
      virtual ~EventWaiter(void) {}
      virtual bool event_triggered(void) = 0;
      virtual void print_info(FILE *f) = 0;
    };

    // parent class of GenEventImpl and BarrierImpl
    class EventImpl {
    public:
      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen) = 0;

      // causes calling thread to block until event has occurred
      //void wait(Event::gen_t needed_gen);

      virtual void external_wait(Event::gen_t needed_gen) = 0;

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/) = 0;

      static bool add_waiter(Event needed, EventWaiter *waiter);
    };

    class GenEventImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_EVENT;

      GenEventImpl(void);

      void init(ID _me, unsigned _init_owner);

      static GenEventImpl *create_genevent(void);

      // get the Event (id+generation) for the current (i.e. untriggered) generation
      Event current_event(void) const { Event e = me.convert<Event>(); e.gen = generation+1; return e; }

      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen);

      virtual void external_wait(Event::gen_t needed_gen);

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/);

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = Event::NO_EVENT, Event ev4 = Event::NO_EVENT,
				Event ev5 = Event::NO_EVENT, Event ev6 = Event::NO_EVENT);

      // record that the event has triggered and notify anybody who cares
      void trigger(Event::gen_t gen_triggered, int trigger_node, Event wait_on = Event::NO_EVENT);

      // if you KNOW you want to trigger the current event (which by definition cannot
      //   have already been triggered) - this is quicker:
      void trigger_current(void);

      void check_for_catchup(Event::gen_t implied_trigger_gen);

    public: //protected:
      ID me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed;
      GenEventImpl *next_free;

      GASNetHSL mutex; // controls which local thread has access to internal data (not runtime-visible event)

      NodeSet remote_waiters;
      std::vector<EventWaiter *> local_waiters; // set of local threads that are waiting on event
    };

    class BarrierImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_BARRIER;

      BarrierImpl(void);

      void init(ID _me, unsigned _init_owner);

      static BarrierImpl *create_barrier(unsigned expected_arrivals, ReductionOpID redopid,
					 const void *initial_value = 0, size_t initial_value_size = 0);

      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen);

      virtual void external_wait(Event::gen_t needed_gen);

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/);

      // used to adjust a barrier's arrival count either up or down
      // if delta > 0, timestamp is current time (on requesting node)
      // if delta < 0, timestamp says which positive adjustment this arrival must wait for
      void adjust_arrival(Event::gen_t barrier_gen, int delta, 
			  Barrier::timestamp_t timestamp, Event wait_on,
			  const void *reduce_value, size_t reduce_value_size);

      bool get_result(Event::gen_t result_gen, void *value, size_t value_size);

    public: //protected:
      ID me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed;
      Event::gen_t first_generation, free_generation;
      BarrierImpl *next_free;

      GASNetHSL mutex; // controls which local thread has access to internal data (not runtime-visible event)

      // class to track per-generation status
      class Generation;

      std::map<Event::gen_t, Generation *> generations;

      // a list of remote waiters and the latest generation they're interested in
      // also the latest generation that each node (that has ever subscribed) has been told about
      std::map<unsigned, Event::gen_t> remote_subscribe_gens, remote_trigger_gens;
      std::map<Event::gen_t, Event::gen_t> held_triggers;

      unsigned base_arrival_count;
      ReductionOpID redop_id;
      const ReductionOpUntyped *redop;
      char *initial_value;  // for reduction barriers

      unsigned value_capacity; // how many values the two allocations below can hold
      char *final_values;   // results of completed reductions
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

    class ReservationImpl {
    public:
      ReservationImpl(void);

      static const ID::ID_Types ID_TYPE = ID::ID_LOCK;

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

      GASNetHSL mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      NodeSet remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      std::map<unsigned, std::deque<GenEventImpl *> > local_waiters;
      bool requested; // do we have a request for the lock in flight?

      // local data protected by lock
      void *local_data;
      size_t local_data_size;
      bool own_local;

      static GASNetHSL freelist_mutex;
      static ReservationImpl *first_free;
      ReservationImpl *next_free;

      // created a GenEventImpl if needed to describe when reservation is granted
      Event acquire(unsigned new_mode, bool exclusive,
		    GenEventImpl *after_lock = 0);

      bool select_local_waiters(std::deque<GenEventImpl *>& to_wake);

      void release(void);

      bool is_locked(unsigned check_mode, bool excl_ok);

      void release_reservation(void);

      struct PackFunctor {
      public:
        PackFunctor(int *p) : pos(p) { }
      public:
        inline void apply(int target) { *pos++ = target; }
      public:
        int *pos;
      };
    };

    template <typename T>
    class StaticAccess {
    public:
      typedef typename T::StaticData StaticData;

      StaticAccess(T* thing_with_data, bool already_valid = false);

      ~StaticAccess(void) {}

      const StaticData *operator->(void) { return data; }

    protected:
      StaticData *data;
    };

    template <typename T>
    class SharedAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      SharedAccess(T* thing_with_data, bool already_held = false);

      ~SharedAccess(void)
      {
	lock->release();
      }

      const CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      ReservationImpl *lock;
    };

    template <class T>
    class ExclusiveAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      ExclusiveAccess(T* thing_with_data, bool already_held = false);

      ~ExclusiveAccess(void)
      {
	lock->release();
      }

      CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      ReservationImpl *lock;
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
#ifndef __MACH__
      cpu_set_t leftover_procs;
#endif
    };
    extern ProcessorAssignment *proc_assignment;

    extern Processor::TaskIDTable task_id_table;

    class ProcessorGroup;

    // information for a task launch
    class Task : public Realm::Operation {
    public:
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
	   Event _finish_event, int _priority,
           int expected_count);
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
           const Realm::ProfilingRequestSet &reqs,
	   Event _finish_event, int _priority,
           int expected_count);

      virtual ~Task(void);

      Processor proc;
      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
      Event finish_event;
      int priority;
      int run_count, finish_count;
      bool capture_proc;
    };

    class ProcessorImpl {
    public:
      ProcessorImpl(Processor _me, Processor::Kind _kind);

      virtual ~ProcessorImpl(void);

      void run(Atomic<int> *_run_counter)
      {
	run_counter = _run_counter;
      }

      virtual void start_processor(void) = 0;
      virtual void shutdown_processor(void) = 0;
      virtual void initialize_processor(void) = 0;
      virtual void finalize_processor(void) = 0;

      virtual void enqueue_task(Task *task) = 0;

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority) = 0;

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const Realm::ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority) = 0;

      void finished(void)
      {
	if(run_counter)
	  run_counter->decrement();
      }

    public:
      Processor me;
      Processor::Kind kind;
      Atomic<int> *run_counter;
    }; 

    class DeferredTaskSpawn : public EventWaiter {
    public:
      DeferredTaskSpawn(ProcessorImpl *_proc, Task *_task) 
        : proc(_proc), task(_task) {}

      virtual ~DeferredTaskSpawn(void)
      {
        // we do _NOT_ own the task - do not free it
      }

      virtual bool event_triggered(void);
      virtual void print_info(FILE *f);

    protected:
      ProcessorImpl *proc;
      Task *task;
    };

    // generic way of keeping a prioritized queue of stuff to do
    // Needs to be protected by owner lock
    template <typename JOBTYPE>
    class JobQueue {
    public:
      JobQueue(void);

      bool empty(void) const;

      void insert(JOBTYPE *job, int priority);

      JOBTYPE *pop(void);

      struct WaitingJob : public EventWaiter {
	JOBTYPE *job;
	int priority;
	JobQueue *queue;

	virtual bool event_triggered(void);
	virtual void print_info(FILE *f);
      };

      std::map<int, std::deque<JOBTYPE*> > ready;
    };

    template <typename JOBTYPE>
    JobQueue<JOBTYPE>::JobQueue(void)
    {
    }

    template<typename JOBTYPES>
    bool JobQueue<JOBTYPES>::empty(void) const
    {
      return ready.empty();
    }

    template <typename JOBTYPE>
    void JobQueue<JOBTYPE>::insert(JOBTYPE *job, int priority)
    {
      std::deque<JOBTYPE *>& dq = ready[-priority];
      dq.push_back(job);
    }

    template <typename JOBTYPE>
    JOBTYPE *JobQueue<JOBTYPE>::pop(void)
    {
      if(ready.empty()) return 0;

      // get the sublist with the highest priority (remember, we negate before lookup)
      typename std::map<int, std::deque<JOBTYPE *> >::iterator it = ready.begin();

      // any deque that's present better be non-empty
      assert(!(it->second.empty()));
      JOBTYPE *job = it->second.front();
      it->second.pop_front();

      // if the list is now empty, remove it and update the new max priority
      if(it->second.empty()) {
	ready.erase(it);
      }

      return job;
    }

    class ProcessorGroup : public ProcessorImpl {
    public:
      ProcessorGroup(void);

      virtual ~ProcessorGroup(void);

      static const ID::ID_Types ID_TYPE = ID::ID_PROCGROUP;

      void init(Processor _me, int _owner);

      void set_group_members(const std::vector<Processor>& member_list);

      void get_group_members(std::vector<Processor>& member_list);

      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);

      virtual void enqueue_task(Task *task);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const Realm::ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);


    public: //protected:
      bool members_valid;
      bool members_requested;
      std::vector<ProcessorImpl *> members;
      ReservationImpl lock;
      ProcessorGroup *next_free;

      void request_group_members(void);
    };
    
    class PreemptableThread {
    public:
      PreemptableThread(void) 
      {
#ifdef EVENT_GRAPH_TRACE
        enclosing_stack.push_back(Event::NO_EVENT);
#endif
      }
      virtual ~PreemptableThread(void) {}

      void start_thread(size_t stack_size, int core_id, const char *debug_name);

      static bool preemptable_sleep(Event wait_for);

      virtual Processor get_processor(void) const = 0;

      void run_task(Task *task, Processor actual_proc = Processor::NO_PROC);

#ifdef EVENT_GRAPH_TRACE
      inline Event find_enclosing(void) 
      { assert(!enclosing_stack.empty()); return enclosing_stack.back(); }
      inline void start_enclosing(const Event &term_event)
      { enclosing_stack.push_back(term_event); }
      inline void finish_enclosing(void)
      { assert(enclosing_stack.size() > 1); enclosing_stack.pop_back(); }
#endif

    protected:
      static void *thread_entry(void *data);

      virtual void thread_main(void) = 0;

      virtual void sleep_on_event(Event wait_for) = 0;

      pthread_t thread;

#ifdef EVENT_GRAPH_TRACE
      std::deque<Event> enclosing_stack; 
#endif
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

    // Forward declaration
    class LocalProcessor;

    class LocalThread : public PreemptableThread, EventWaiter {
    public:
      enum ThreadState {
        RUNNING_STATE,
        PAUSED_STATE,
        RESUMABLE_STATE,
        SLEEPING_STATE, // about to sleep
        SLEEP_STATE,
      };
    public:
      LocalThread(LocalProcessor *proc);
      virtual ~LocalThread(void);
    public:
      inline void do_initialize(void) { initialize = true; }
      inline void do_finalize(void) { finalize = true; }
    public:
      virtual Processor get_processor(void) const;
    protected:
      virtual void thread_main(void);
      virtual void sleep_on_event(Event wait_for);
      virtual bool event_triggered(void);
      virtual void print_info(FILE *f);
    public:
      void awake(void);
      void sleep(void);
      void prepare_to_sleep(void);
      void resume(void);
      void shutdown(void);
    public:
      LocalProcessor *const proc;
    protected:
      ThreadState state;
      GASNetHSL thread_mutex;
      GASNetCondVar thread_cond;
      bool initialize;
      bool finalize;
    };

    class LocalProcessor : public ProcessorImpl {
    public:
      LocalProcessor(Processor _me, Processor::Kind _kind, 
                     size_t stack_size, const char *name,
                     int core_id = -1);
      virtual ~LocalProcessor(void);
    public:
      // Make these virtual so they can be modified if necessary
      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);
      virtual LocalThread* create_new_thread(void);
    public:
      bool execute_task(LocalThread *thread);
      void pause_thread(LocalThread *thread);
      void resume_thread(LocalThread *thread);
    public:
      virtual void enqueue_task(Task *task);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const Realm::ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);
    protected:
      const int core_id;
      const size_t stack_size;
      const char *const processor_name;
      GASNetHSL mutex;
      GASNetCondVar condvar;
      JobQueue<Task> task_queue;
      bool shutdown, shutdown_trigger;
    protected:
      LocalThread               *running_thread;
      std::set<LocalThread*>    paused_threads;
      std::deque<LocalThread*>  resumable_threads;
      std::vector<LocalThread*> available_threads;
    };

    // Forward declarations
    class GreenletThread;
    class GreenletProcessor;

    class GreenletTask : public greenlet, public EventWaiter {
    public:
      GreenletTask(Task *task, GreenletProcessor *proc,
                   void *stack, long *stack_size);
      virtual ~GreenletTask(void);
    public:
      virtual bool event_triggered(void);
      virtual void print_info(FILE *f);
    public:
      virtual void* run(void *arg);
    protected:
      Task *const task;
      GreenletProcessor *const proc;
    };

    class GreenletThread : public PreemptableThread {
    public:
      GreenletThread(GreenletProcessor *proc);
      virtual ~GreenletThread(void);
    public:
      virtual Processor get_processor(void) const;
    public:
      virtual void thread_main(void);
      virtual void sleep_on_event(Event wait_for);
    public:
      void start_task(GreenletTask *task);
      void resume_task(GreenletTask *task);
      void return_to_root(void);
      void wait_for_shutdown(void);
    public:
      GreenletProcessor *const proc;
    protected:
      GreenletTask *current_task;
    };

    class GreenletProcessor : public ProcessorImpl {
    public:
      enum GreenletState {
        GREENLET_IDLE,
        GREENLET_RUNNING,
      };
      struct GreenletStack {
      public:
        void *stack;
        long stack_size;
      };
    public:
      GreenletProcessor(Processor _me, Processor::Kind _kind,
                        size_t stack_size, int init_stack_count,
                        const char *name, int core_id = -1);
      virtual ~GreenletProcessor(void);
    public:
      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);
    public:
      virtual void enqueue_task(Task *task);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event,
                              int priority);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const Realm::ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);
    public:
      bool execute_task(void);
      void pause_task(GreenletTask *paused_task);
      void unpause_task(GreenletTask *paused_task);
    public:
      bool allocate_stack(GreenletStack &stack);
      void create_stack(GreenletStack &stack);
      void complete_greenlet(GreenletTask *greenlet); 
    public:
      const int core_id;
      const size_t proc_stack_size;
      const char *const processor_name;
    protected:
      GASNetHSL mutex;
      GASNetCondVar condvar;
      JobQueue<Task> task_queue;
      bool shutdown, shutdown_trigger;
    protected:
      GreenletThread             *greenlet_thread;
      GreenletState              thread_state;
      std::set<GreenletTask*>    paused_tasks; 
      std::list<GreenletTask*>   resumable_tasks;
      std::vector<GreenletStack> greenlet_stacks;
      std::vector<GreenletTask*> complete_greenlets;
    };

    class RegionInstanceImpl;

    class MemoryImpl {
    public:
      enum MemoryKind {
	MKIND_SYSMEM,  // directly accessible from CPU
	MKIND_GLOBAL,  // accessible via GASnet (spread over all nodes)
	MKIND_RDMA,    // remote, but accessible via RDMA
	MKIND_REMOTE,  // not accessible
#ifdef USE_CUDA
	MKIND_GPUFB,   // GPU framebuffer memory (accessible via cudaMemcpy)
#endif
	MKIND_ZEROCOPY, // CPU memory, pinned for GPU access
	MKIND_DISK,    // disk memory accessible by owner node
#ifdef USE_HDF
	MKIND_HDF      // HDF memory accessible by owner node
#endif
      };

      MemoryImpl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment, Memory::Kind _lowlevel_kind);

      virtual ~MemoryImpl(void);

      unsigned add_instance(RegionInstanceImpl *i);

      RegionInstanceImpl *get_instance(RegionInstance i);

      RegionInstance create_instance_local(IndexSpace is,
					   const int *linearization_bits,
					   size_t bytes_needed,
					   size_t block_size,
					   size_t element_size,
					   const std::vector<size_t>& field_sizes,
					   ReductionOpID redopid,
					   off_t list_size,
                                           const Realm::ProfilingRequestSet &reqs,
					   RegionInstance parent_inst);

      RegionInstance create_instance_remote(IndexSpace is,
					    const int *linearization_bits,
					    size_t bytes_needed,
					    size_t block_size,
					    size_t element_size,
					    const std::vector<size_t>& field_sizes,
					    ReductionOpID redopid,
					    off_t list_size,
                                            const Realm::ProfilingRequestSet &reqs,
					    RegionInstance parent_inst);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
                                             const Realm::ProfilingRequestSet &reqs,
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
      Memory::Kind lowlevel_kind;
      GASNetHSL mutex; // protection for resizing vectors
      std::vector<RegionInstanceImpl *> instances;
      std::map<off_t, off_t> free_blocks;
#ifdef REALM_PROFILE_MEMORY_USAGE
      size_t usage, peak_usage, peak_footprint;
#endif
    };

    class GASNetMemory : public MemoryImpl {
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
                                             const Realm::ProfilingRequestSet &reqs,
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

    class DiskMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      DiskMemory(Memory _me, size_t _size, std::string _file);

      virtual ~DiskMemory(void);

      virtual RegionInstance create_instance(IndexSpace is,
                                            const int *linearization_bits,
                                            size_t bytes_needed,
                                            size_t block_size,
                                            size_t element_size,
                                            const std::vector<size_t>& field_sizes,
                                            ReductionOpID redopid,
                                            off_t list_size,
                                            const Realm::ProfilingRequestSet &reqs,
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

    public:
      int fd; // file descriptor
      std::string file;  // file name
    };

#ifdef USE_HDF
    class HDFMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      HDFMemory(Memory _me);

      virtual ~HDFMemory(void);

      
      virtual RegionInstance create_instance(IndexSpace is,
                                             const int *linearization_bits,
                                             size_t bytes_needed,
                                             size_t block_size,
                                             size_t element_size,
                                             const std::vector<size_t>& field_sizes,
                                             ReductionOpID redopid,
                                             off_t list_size,
                                             const Realm::ProfilingRequestSet &reqs,
                                             RegionInstance parent_inst);
      
      RegionInstance create_instance(IndexSpace is,
                                     const int *linearization_bits,
                                     size_t bytes_needed,
                                     size_t block_size,
                                     size_t element_size,
                                     const std::vector<size_t>& field_sizes,
                                     ReductionOpID redopid,
                                     off_t list_size,
                                     const Realm::ProfilingRequestSet &reqs,
                                     RegionInstance parent_inst,
                                     const char* file,
                                     const std::vector<const char*>& path_names,
                                     Domain domain,
                                     bool read_only);

      virtual void destroy_instance(RegionInstance i,
                                    bool local_destroy);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);
      void get_bytes(IDType inst_id, const DomainPoint& dp, int fid, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);
      void put_bytes(IDType inst_id, const DomainPoint& dp, int fid, const void *src, size_t size);

      virtual void apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                                       size_t count, const void *entry_buffer);

      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

    public:
      struct HDFMetadata {
        int lo[3];
        hsize_t dims[3];
        int ndims;
        hid_t file_id;
        std::vector<hid_t> dataset_ids;
        std::vector<hid_t> datatype_ids;
        HDFMemory* hdf_memory;
      };
      std::vector<HDFMetadata*> hdf_metadata;
    };
#endif

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
      
      void parse_node_announce_data(const void *args, size_t arglen,
				    const NodeAnnounceData& annc_data,
				    bool remote);
    protected:
      std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
      std::vector<Machine::MemoryMemoryAffinity> mem_mem_affinities;
    };

    extern MachineImpl *machine_singleton;
    inline MachineImpl *get_machine(void) { return machine_singleton; }

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

    template <typename T>
    StaticAccess<T>::StaticAccess(T* thing_with_data, bool already_valid /*= false*/)
      : data(&thing_with_data->locked_data)
    {
      // if already_valid, just check that data is already valid
      if(already_valid) {
	assert(data->valid);
      } else {
	if(!data->valid) {
	  // get a valid copy of the static data by taking and then releasing
	  //  a shared lock
	  Event e = thing_with_data->lock.acquire(1, false);
	  if(!e.has_triggered()) 
            e.wait();
	  thing_with_data->lock.release();
	  assert(data->valid);
	}
      }
    }

    template <typename T>
    SharedAccess<T>::SharedAccess(T* thing_with_data, bool already_held /*= false*/)
      : data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
    {
      // if already_held, just check that it's held (if in debug mode)
      if(already_held) {
	assert(lock->is_locked(1, true));
      } else {
	Event e = thing_with_data->lock.acquire(1, false);
	if(!e.has_triggered())
          e.wait();
      }
    }

    template <typename T>
    ExclusiveAccess<T>::ExclusiveAccess(T* thing_with_data, bool already_held /*= false*/)
      : data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
    {
      // if already_held, just check that it's held (if in debug mode)
      if(already_held) {
	assert(lock->is_locked(0, true));
      } else {
	Event e = thing_with_data->lock.acquire(0, true);
	if(!e.has_triggered())
          e.wait();
      }
    }

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
