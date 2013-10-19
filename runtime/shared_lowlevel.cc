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


#include "lowlevel.h"
#include "accessor.h"
#include "legion_logging.h"

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
#include <vector>

#include <pthread.h>
#include <errno.h>

#define BASE_EVENTS	1024	
#define BASE_LOCKS	64	
#define BASE_METAS	64
#define BASE_ALLOCATORS	64
#define BASE_INSTANCES	64

// The number of threads for this version
#define NUM_PROCS	4
#define NUM_UTIL_PROCS  0
// Maximum memory in global
#define GLOBAL_MEM      4096   // (MB)	
#define LOCAL_MEM       16384  // (KB)
// Default Pthreads stack size
#define STACK_SIZE      2      // (MB) 

#ifdef DEBUG_LOW_LEVEL
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
pthread_key_t local_proc_key;
static void thread_proc_free(void *arg)
{
  assert(arg != NULL);
  free(arg);
}

namespace LegionRuntime {
  namespace LowLevel {

// MAC OSX doesn't support pthread barrier type
#ifdef __MACH__
    typedef UtilityBarrier pthread_barrier_t;
#endif
    
    // Implementation for each of the runtime objects
    class EventImpl;
    class LockImpl;
    class MemoryImpl;
    class ProcessorImpl;

    class Runtime {
    public:
      Runtime(Machine *m, const ReductionOpTable &table);
    public:
      static Runtime* get_runtime(void) { return runtime; } 

      EventImpl*           get_event_impl(Event e);
      LockImpl*            get_lock_impl(Lock l);
      MemoryImpl*          get_memory_impl(Memory m);
      ProcessorImpl*       get_processor_impl(Processor p);
      IndexSpace::Impl*  get_metadata_impl(IndexSpace is);
      RegionInstance::Impl*  get_instance_impl(RegionInstance i);

      EventImpl*           get_free_event(void);
      LockImpl*            get_free_lock(size_t data_size = 0);
      IndexSpace::Impl*  get_free_metadata(size_t num_elmts);
      IndexSpace::Impl*  get_free_metadata(IndexSpace::Impl *par, const ElementMask &mask);
      RegionInstance::Impl*  get_free_instance(IndexSpace is, Memory m, size_t num_elmts, 
					       const std::vector<size_t>& field_sizes,
					       size_t elmt_size, size_t block_size,
					       const DomainLinearization& linearization,
					       char *ptr, const ReductionOpUntyped *redop,
					       RegionInstance::Impl *parent);

      const ReductionOpUntyped* get_reduction_op(ReductionOpID redop);

      // Return events that are free
      void free_event(EventImpl *event);
    protected:
      static Runtime *runtime;
    protected:
      friend class Machine;
      ReductionOpTable redop_table;
      std::vector<EventImpl*> events;
      std::list<EventImpl*> free_events; // Keep a free list of events since this seems to dominate perf
      std::vector<LockImpl*> locks;
      std::list<LockImpl*> free_locks;
      std::vector<MemoryImpl*> memories;
      std::vector<ProcessorImpl*> processors;
      std::vector<IndexSpace::Impl*> metadatas;
      std::list<IndexSpace::Impl*> free_metas;
      std::vector<RegionInstance::Impl*> instances;
      std::list<RegionInstance::Impl*> free_instances;
      Machine *machine;
      pthread_rwlock_t event_lock;
      pthread_mutex_t  free_event_lock;
      pthread_rwlock_t lock_lock;
      pthread_mutex_t  free_lock_lock;
      pthread_rwlock_t metadata_lock;
      pthread_mutex_t  free_metas_lock;
      pthread_rwlock_t allocator_lock;
      pthread_mutex_t  free_alloc_lock;
      pthread_rwlock_t instance_lock;
      pthread_mutex_t  free_inst_lock;
    };

    /* static */
    Runtime *Runtime::runtime = NULL;
    
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
        unsigned *local_proc_id = (unsigned*)pthread_getspecific(local_proc_key);
        thread = *local_proc_id; 
        mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
        PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
      }
      ~PerThreadTimerData(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
        free(mutex);
      }

      unsigned thread;
      std::list<TimerStackEntry> timer_stack;
      std::map<int, double> timer_accum;
      pthread_mutex_t *mutex;
    };

    pthread_mutex_t global_timer_mutex = PTHREAD_MUTEX_INITIALIZER;
    std::vector<PerThreadTimerData*> timer_data;
    //__thread PerThreadTimerData *thread_timer_data;
    pthread_key_t thread_timer_key;
    static void thread_timer_free(void *arg)
    {
      assert(arg != NULL);
      PerThreadTimerData *ptr = (PerThreadTimerData*)arg;
      delete ptr;
    }

#ifdef DETAILED_TIMING
    /*static*/ void DetailedTimer::clear_timers(bool all_nodes /*=true*/)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&global_timer_mutex));
      for (std::vector<PerThreadTimerData*>::iterator it = timer_data.begin();
            it != timer_data.end(); it++)
      {
        // Take each thread's data lock as well
        PTHREAD_SAFE_CALL(pthread_mutex_lock(((*it)->mutex)));
        (*it)->timer_accum.clear();
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(((*it)->mutex)));
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&global_timer_mutex));
    }

    /*static*/ void DetailedTimer::push_timer(int timer_kind)
    {
      PerThreadTimerData *thread_timer_data = 
        (PerThreadTimerData*) pthread_getspecific(thread_timer_key);
      if (!thread_timer_data)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&global_timer_mutex));
        thread_timer_data = new PerThreadTimerData();
        PTHREAD_SAFE_CALL(pthread_setspecific(thread_timer_key,thread_timer_data));
        timer_data.push_back(thread_timer_data);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&global_timer_mutex));
      }

      // no lock required here - only our thread touches the stack
      TimerStackEntry entry;
      entry.timer_kind = timer_kind;
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      entry.start_time = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec);
      entry.accum_child_time = 0;
      thread_timer_data->timer_stack.push_back(entry);
    }

    /*static*/ void DetailedTimer::pop_timer(void)
    {
      PerThreadTimerData *thread_timer_data =
        (PerThreadTimerData*) pthread_getspecific(thread_timer_key);
      if (!thread_timer_data)
      {
        printf("Got pop without initialized thread data !?\n");
        exit(1);
      }

      // no conflicts on stack
      TimerStackEntry old_top = thread_timer_data->timer_stack.back();
      thread_timer_data->timer_stack.pop_back();

      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      double elapsed = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec) - old_top.start_time;

      // all the elapsed time is added to the new top as child time
      if (!thread_timer_data->timer_stack.empty())
        thread_timer_data->timer_stack.back().accum_child_time += elapsed;

      // only the elapsed time minus our own child time goes into the timer accumulator
      elapsed -= old_top.accum_child_time;

      // We do need a lock to touch the accumulator
      if (old_top.timer_kind > 0)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_timer_data->mutex));
        
        std::map<int,double>::iterator it = thread_timer_data->timer_accum.find(old_top.timer_kind);
        if (it != thread_timer_data->timer_accum.end())
          it->second += elapsed;
        else
          thread_timer_data->timer_accum.insert(std::make_pair(old_top.timer_kind,elapsed));

        PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_timer_data->mutex));
      }
    }

    /*static*/ void DetailedTimer::roll_up_timers(std::map<int,double> &timers, bool local_only)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&global_timer_mutex));

      for (std::vector<PerThreadTimerData*>::iterator it = timer_data.begin();
            it != timer_data.end(); it++)
      {
        // Take the local lock for each thread's data too
        PTHREAD_SAFE_CALL(pthread_mutex_lock(((*it)->mutex)));

        for (std::map<int,double>::iterator it2 = (*it)->timer_accum.begin();
              it2 != (*it)->timer_accum.end(); it2++)
        {
          std::map<int,double>::iterator it3 = timers.find(it2->first);
          if (it3 != timers.end())
            it3->second += it2->second;
          else
            timers.insert(*it2);
        }

        PTHREAD_SAFE_CALL(pthread_mutex_unlock(((*it)->mutex)));
      }

      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&global_timer_mutex));
    }

    /*static*/ void DetailedTimer::report_timers(bool local_only /* = false*/)
    {
      std::map<int, double> timers;

      roll_up_timers(timers, local_only);

      printf("DETAILED_TIMING_SUMMARY:\n");
      for (std::map<int,double>::iterator it = timers.begin();
            it != timers.end(); it++)
      {
        printf("%12s - %7.3f s\n", stringify(it->first), it->second);
      }
      printf("END OF DETAILED TIMING SUMMARY\n");
    }
#endif
    
    

    // Any object which can be triggered should be able to triggered
    // This will include Events and Locks
    class Triggerable {
    public:
        typedef unsigned TriggerHandle;
	virtual void trigger(unsigned count = 1, TriggerHandle = 0) = 0;
	// make the warnings go away
	virtual ~Triggerable() { }
    };

    ////////////////////////////////////////////////////////
    // Event Impl (up here since we need it in Processor Impl) 
    ////////////////////////////////////////////////////////

    class EventImpl : public Triggerable {
    public:
	typedef unsigned EventIndex;
	typedef unsigned EventGeneration;
    public:
	EventImpl(EventIndex idx, bool activate=false) 
		: index(idx)
	{
	  in_use = activate;
	  generation = 0;
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
	    sources = 1;
#ifdef DEBUG_LOW_LEVEL
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
        }
	
	// test whether an event has triggered without waiting
	bool has_triggered(EventGeneration needed_gen);
	// block until event has triggered
	void wait(EventGeneration needed_gen, bool block);
	// create an event that won't trigger until all input events have
	Event merge_events(const std::map<EventImpl*,Event> &wait_for);
	// Trigger the event
	void trigger(unsigned count = 1, TriggerHandle handle = 0);
	// Check to see if the event is active, if not activate it (return true), otherwise false
	bool activate(void);	
	// Register a dependent event, return true if event had not been triggered and was registered
	bool register_dependent(Triggerable *target, EventGeneration needed_gen, TriggerHandle handle = 0);
	// Return an event for this EventImplementation
	Event get_event();
        // Return a user event for this EventImplementation
        UserEvent get_user_event();
        // Return a barrier for this EventImplementation
        Barrier get_barrier(unsigned expected_arrivals);
        // Alter the arrival count for the barrier
        Barrier alter_arrival_count(int delta);
    private: 
	bool in_use;
	unsigned sources;
	const EventIndex index;
	EventGeneration generation;
	// The version of the event to hand out (i.e. with generation+1)
	// so we can detect when the event has triggered with testing
	// generational equality
	Event current; 
	pthread_mutex_t *mutex;
	pthread_cond_t *wait_cond;
	std::vector<Triggerable*> triggerables;
        std::vector<TriggerHandle> trigger_handles;
    }; 

    ////////////////////////////////////////////////////////
    // Processor Impl (up here since we need it in Event) 
    ////////////////////////////////////////////////////////

    class ProcessorImpl : public Triggerable {
    public:
        // For creation of normal processors when there is no utility processors
        ProcessorImpl(pthread_barrier_t *init, const Processor::TaskIDTable &table, 
                      Processor p, size_t stacksize, bool return_finish = false) 
          : init_bar(init), task_table(table), proc(p), 
            proc_kind(Processor::LOC_PROC), utility_proc(this),
            has_scheduler(table.find(Processor::TASK_ID_PROCESSOR_IDLE) != table.end()),
            is_utility_proc(true), return_on_finish(return_finish), remaining_stops(0),
            scheduler_invoked(false), util_shutdown(true)
        {
          utility.id = p.id; // we are our own utility processor
          initialize_state(stacksize);
	  idle_task_enabled = true;
          num_idle_tasks = 0;
          // Add ourselves as the processor that we're managing 
          util_users.insert(p);
        }
        // For the creation of normal processors when there are utility processors
        ProcessorImpl(pthread_barrier_t *init, const Processor::TaskIDTable &table,
                      Processor p, size_t stacksize, ProcessorImpl *util, bool return_finish = false)
          : init_bar(init), task_table(table), proc(p), 
            utility(util->get_utility_processor()), 
            proc_kind(Processor::LOC_PROC), utility_proc(util), 
            has_scheduler(table.find(Processor::TASK_ID_PROCESSOR_IDLE) != table.end()),
            is_utility_proc(false), return_on_finish(return_finish), remaining_stops(0),
            scheduler_invoked(false), util_shutdown(false)
        {
          initialize_state(stacksize);
	  idle_task_enabled = true;
          num_idle_tasks = 0;
        }
        // For the creation of explicit utility processors
        ProcessorImpl(pthread_barrier_t *init, const Processor::TaskIDTable &table,
                      Processor p, size_t stacksize, unsigned num_owners, bool return_finish = false)
          : init_bar(init), task_table(table), proc(p), 
            proc_kind(Processor::UTIL_PROC), utility_proc(this),
            has_scheduler(table.find(Processor::TASK_ID_PROCESSOR_IDLE) != table.end()),
            is_utility_proc(true), return_on_finish(return_finish), remaining_stops(num_owners),
            scheduler_invoked(false), util_shutdown(true)
        {
          utility.id = p.id;
          initialize_state(stacksize);
	  idle_task_enabled = false;  // no idle task for util procs
          num_idle_tasks = 0;
        }
        ~ProcessorImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                PTHREAD_SAFE_CALL(pthread_cond_destroy(wait_cond));
                PTHREAD_SAFE_CALL(pthread_attr_destroy(&attr));
                free(mutex);
                free(wait_cond);
        }
    private:
        void initialize_state(size_t stacksize);
    public:
        // Operations for utility processors
        Processor get_utility_processor(void) const;
        Processor::Kind get_proc_kind(void) const { return proc_kind; }
        void release_user(void);
        void utility_finish(void);
        const std::set<Processor>& get_utility_users(void) const;
        void add_utility_user(Processor p, ProcessorImpl *impl);
    public:
	Event spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on);
        void run(void);
	void trigger(unsigned count = 1, TriggerHandle handle = 0);
	static void* start(void *proc);
	void preempt(EventImpl *event, EventImpl::EventGeneration needed);
    public:
        void enable_idle_task(void);
        void disable_idle_task(void);
        void increment_utility(void);
        void decrement_utility(void);
    private:
	void execute_task(bool permit_shutdown);
        bool perform_scheduling(bool need_lock);
    private:
	class TaskDesc {
	public:
		Processor::TaskFuncID func_id;
		void * args;
		size_t arglen;
		Event wait;
		EventImpl *complete;
	};
    public:
        pthread_attr_t attr; // For setting pthread parameters when starting the thread
    private:
        pthread_barrier_t *init_bar;
	Processor::TaskIDTable task_table;
	Processor proc;
        Processor utility;
        const Processor::Kind proc_kind;
        ProcessorImpl *utility_proc;
	std::list<TaskDesc> ready_queue;
	std::list<TaskDesc> waiting_queue;
	pthread_mutex_t *mutex;
	pthread_cond_t *wait_cond;
	// Used for detecting the shutdown condition
	bool shutdown;
        bool idle_task_enabled;
	EventImpl *shutdown_trigger;
        const bool has_scheduler;
        const bool is_utility_proc;
        const bool return_on_finish;
        unsigned remaining_stops; // for utility processor knowing when to stop
        bool scheduler_invoked;   // for traking if we've invoked the scheduler
        bool util_shutdown;       // for knowing when our utility processor is done
        unsigned num_idle_tasks; // number of idle tasks enabled for the utility processor
        std::set<Processor> util_users;// Users of the utility processor to know when it's safe to finish
        std::vector<ProcessorImpl*> constituents; // User impls of the utility processor
    };

    ////////////////////////////////////////////////////////
    // Events 
    ////////////////////////////////////////////////////////

    /* static */ const Event Event::NO_EVENT = { 0, 0 };

    bool Event::has_triggered(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	if (!id) return true;
	EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
	return e->has_triggered(gen);
    }

    void Event::wait(bool block) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL); 
	if (!id) return;
	EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
	e->wait(gen,block);
    }

    // used by non-legion threads to wait on an event - always blocking
    void Event::external_wait(void) const
    {
      // shared lowlevel does pthreads for everybody
      wait(true);
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
	EventImpl *e = Runtime::get_runtime()->get_free_event();
        // Get the implementations for all the wait_for events
        // Do this to avoid calling get_event_impl while holding the event lock
        std::map<EventImpl*,Event> wait_for_impl;
        for (std::set<Event>::const_iterator it = wait_for.begin();
              it != wait_for.end(); it++)
        {
          assert(wait_for_impl.size() < wait_for.size());
          if (!(*it).exists())
            continue;
          EventImpl *src_impl = Runtime::get_runtime()->get_event_impl(*it);
          std::pair<EventImpl*,Event> made_pair(src_impl,*it);
          wait_for_impl.insert(std::pair<EventImpl*,Event>(src_impl,*it));
        }
	return e->merge_events(wait_for_impl);
    }

    bool EventImpl::has_triggered(EventGeneration needed_gen)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	result = (needed_gen <= generation);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void EventImpl::wait(EventGeneration needed_gen, bool block)
    {
        if (block)
        {
            // First check to see if the event has triggered
            PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));	
            // Wait until the generation indicates that the event has occurred
            while (needed_gen > generation) 
            {
                    DetailedTimer::ScopedPush sp(TIME_NONE);
                    PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond,mutex));
            }
            PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        }
        else
        {
            // Try preempting the process
            unsigned *local_proc_id = (unsigned*)pthread_getspecific(local_proc_key);
            Processor local = { *local_proc_id };
            ProcessorImpl *impl = Runtime::get_runtime()->get_processor_impl(local);
            // This call will only return once the event has triggered
            impl->preempt(this,needed_gen);
        }
    }

    Event EventImpl::merge_events(const std::map<EventImpl*,Event> &wait_for)
    {
	// We need the lock here so that events we've already registered
	// can't trigger this event before sources is set
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_PRINT
	//DPRINT2("Mering events into event %u generation %u\n",index,generation);
#endif
	sources = 0;
	for (std::map<EventImpl*,Event>::const_iterator it = wait_for.begin();
		it != wait_for.end(); it++)
	{
		EventImpl *src_impl = (it->first);
		// Handle the special case where this event is an older generation
		// of the same event implementation.  In this case we know it
		// already triggered.
		if (src_impl == this)
			continue;
		if (src_impl->register_dependent(this,(it->second).gen))
			sources++;
	}
	Event ret;
        // Handle the case where there are no events, or all the waiting events
        // have already triggered
        if (sources > 0)
        {
          ret = current;
        }
        else
        {
#ifdef DEBUG_LOW_LEVEL
          assert(in_use); // event should be in use
          assert(triggerables.size() == 0); // there should be no triggerables
#endif
          in_use = false;
          // return no event since all the preceding events have already triggered
          ret = Event::NO_EVENT;
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // If ret does not exist, put this back on the list of free events
        if (!ret.exists())
          Runtime::get_runtime()->free_event(this);
	return ret;
    } 

    void EventImpl::trigger(unsigned count, TriggerHandle handle)
    {
	// Update the generation
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
        assert(in_use);
	assert(sources >= count);
#endif
	sources -= count;
        bool finished = false;
	if (sources == 0)
	{
#ifdef DEBUG_PRINT
		//DPRINT2("Event %u triggered for generation %u\n",index,generation);
#endif
		// Increment the generation so that nobody can register a triggerable
		// with this event, but keep event in_use so no one can use the event
		generation++;
#ifdef DEBUG_LOW_LEVEL
		assert(generation == current.gen);
#endif
                // Wake up any waiters
		PTHREAD_SAFE_CALL(pthread_cond_broadcast(wait_cond));
		// Can't be holding the lock when triggering other triggerables
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
		// Trigger any dependent events
		while (!triggerables.empty())
		{
#ifdef DEBUG_LOW_LEVEL
                        assert(triggerables.size() == trigger_handles.size());
#endif
			triggerables.back()->trigger(1, trigger_handles.back());
			triggerables.pop_back();
                        trigger_handles.pop_back();
		}
		// Reacquire the lock and mark that in_use is false
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
		in_use = false;
                finished = true;
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));	
        // tell the runtime that we're free
        if (finished)
          Runtime::get_runtime()->free_event(this);
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
#ifdef DEBUG_LOW_LEVEL
		assert(current.exists());
#endif
	}	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    bool EventImpl::register_dependent(Triggerable *target, EventGeneration gen, TriggerHandle handle)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	// Make sure they're asking for the right generation, otherwise it's already triggered
	if (gen > generation)
	{
		result = true;
		// Enqueue it
		triggerables.push_back(target);	
                trigger_handles.push_back(handle);
#ifdef DEBUG_LOW_LEVEL
                assert(triggerables.size() == trigger_handles.size());
#endif
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));	
	return result;
    }

    Event EventImpl::get_event() 
    {
#ifdef DEBUG_LOW_LEVEL
        assert(in_use);
#endif
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	Event result = current;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    UserEvent EventImpl::get_user_event()
    {
#ifdef DEBUG_LOW_LEVEL
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      UserEvent result; 
      result.id = current.id;
      result.gen = current.gen;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    Barrier EventImpl::get_barrier(unsigned expected_arrivals)
    {
#ifdef DEBUG_LOW_LEVEL
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      Barrier result;
      result.id = current.id;
      result.gen = current.gen;
      // Set the number of expected arrivals
      sources = expected_arrivals;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    Barrier EventImpl::alter_arrival_count(int delta)
    {
#ifdef DEBUG_LOW_LEVEL
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
      if (delta < 0) // If we're deleting, make sure nothing weird happens
        assert(int(sources) > (-delta));
#endif
      sources += delta;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      Barrier result;
      result.id = current.id;
      result.gen = current.gen;
      return result;
    }

    ////////////////////////////////////////////////////////
    // User Events (just use base event impl) 
    ////////////////////////////////////////////////////////

    UserEvent UserEvent::create_user_event(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      EventImpl *impl = Runtime::get_runtime()->get_free_event();
      return impl->get_user_event();
    }

    void UserEvent::trigger(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return;
      EventImpl *impl = Runtime::get_runtime()->get_event_impl(*this);
      impl->trigger();
    }

    ////////////////////////////////////////////////////////
    // Barrier Events (have to use same base impl)
    ////////////////////////////////////////////////////////
    
    Barrier Barrier::create_barrier(unsigned expected_arrivals)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      EventImpl *impl = Runtime::get_runtime()->get_free_event();
      return impl->get_barrier(expected_arrivals);
    }

    void Barrier::destroy_barrier(void)
    {
      // TODO: Implement barrier destruction
      assert(false);
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
      EventImpl *impl = Runtime::get_runtime()->get_event_impl(*this);
      impl->alter_arrival_count(delta);
      return *this;
    }

    void Barrier::arrive(unsigned count /*=1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return;
      EventImpl *impl = Runtime::get_runtime()->get_event_impl(*this);
      impl->trigger(count);
    }

    ////////////////////////////////////////////////////////
    // Lock 
    ////////////////////////////////////////////////////////

    /*static*/ const Lock Lock::NO_LOCK = Lock();

    Logger::Category log_lock("lock");

    class LockImpl : public Triggerable {
    public:
	LockImpl(int idx, bool activate = false, size_t dsize = 0) : index(idx) {
		active = activate;
		taken = false;
		mode = 0;
		holders = 0;
		waiters = false;
                next_handle = 1;
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
                if (activate)
                {
                    if (dsize > 0)
                    {
                        data_size = dsize;
                        data = malloc(data_size);
#ifdef DEBUG_LOW_LEVEL
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
#ifdef DEBUG_LOW_LEVEL
                    assert(dsize == 0);
#endif
                    data_size = 0;
                    data = NULL;
                }
	}	
        ~LockImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
                if (data_size != 0)
                {
#ifdef DEBUG_LOW_LEVEL
                    assert(data != NULL);
#endif
                    free(data);
                    data = NULL;
                    data_size = 0;
                }
        }

	Event lock(unsigned mode, bool exclusive, Event wait_on);
	void unlock(Event wait_on);
	void trigger(unsigned count = 1, TriggerHandle handle = 0);

	bool activate(size_t data_size);
	void deactivate(void);
	Lock get_lock(void) const;
        size_t get_data_size(void) const;
        void* get_data_ptr(void) const;
    private:
	Event register_request(unsigned m, bool exc, TriggerHandle handle = 0);
	void perform_unlock(std::set<EventImpl*> &to_trigger);
    private:
	class LockRecord {
	public:
		unsigned mode;
		bool exclusive;
		Event event;
		bool handled;
                bool ready; // If this lock waits on a event, see if it's ready
                TriggerHandle id; // If it's not ready this is the trigger handle
	};
    private:
	const int index;
	bool active;
	bool taken;
	bool exclusive;
	bool waiters;
	unsigned mode;
	unsigned holders;
        TriggerHandle next_handle; // all numbers >0 are lock requests, 0 is unlock trigger handle
	std::list<LockRecord> requests;
	pthread_mutex_t *mutex;
        void *data;
        size_t data_size;
    };

    Event Lock::lock(unsigned mode, bool exclusive, Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	return l->lock(mode,exclusive, wait_on);
    }

    void Lock::unlock(Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	l->unlock(wait_on);
    }

    Lock Lock::create_lock(size_t data_size)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	return Runtime::get_runtime()->get_free_lock(data_size)->get_lock();
    }

    void Lock::destroy_lock(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	l->deactivate();
    }

    size_t Lock::data_size(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
        return l->get_data_size();
    }

    void* Lock::data_ptr(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
        return l->get_data_ptr();
    }

    Event LockImpl::lock(unsigned m, bool exc, Event wait_on)
    {
	Event result = Event::NO_EVENT;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        log_lock(LEVEL_DEBUG,"lock request: lock=%x mode=%d excl=%d event=%x/%d count=%d",
                 index, m, exc, wait_on.id, wait_on.gen, holders); 
        // check to see if we have to wait on event first
        bool must_wait = false;
        if (wait_on.exists())
        {
          // Try registering the lock
          EventImpl *impl = Runtime::get_runtime()->get_event_impl(wait_on);
          if (impl->register_dependent(this, wait_on.gen, next_handle))
          {
            // Successfully registered with the event, register the request as asleep
            must_wait = true;
          }
        }
        if (must_wait)
        {
          result = register_request(m, exc, next_handle);
          // Increment the next handle since we used it
          next_handle++;
        }
        else // Didn't have to wait for anything do the normal thing
        {
          if (taken)
          {
                  // If either is exclusive we have to register the request
                  if (exclusive || exc)
                  {
                          result = register_request(m,exc);
                  }
                  else
                  {
                          if ((mode == m) && !waiters)
                          {
                                  // Not exclusive and modes are equal
                                  // and there are no waiters
                                  // Can still acquire the lock
                                  holders++;
                          }
                          else
                          {
                                  result = register_request(m,exc);	
                          }
                  }
          }
          else
          {
                  // Nobody has the lock, grab it
                  taken = true;
                  exclusive = exc;
                  mode = m;
                  holders = 1;
#ifdef DEBUG_PRINT
                  DPRINT3("Granting lock %d in mode %d with exclusive %d\n",index,mode,exclusive);
#endif
          }
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    // Always called while holding the lock
    Event LockImpl::register_request(unsigned m, bool exc, TriggerHandle handle)
    {
	EventImpl *e = Runtime::get_runtime()->get_free_event();
	LockRecord req;
	req.mode = m;
	req.exclusive = exc;
	req.event = e->get_event();
	req.handled = false;
        req.id = handle;
        // If handle is 0 then the request is already awake, otherwise wait for the trigger to occur
        req.ready = (handle == 0);
	// Add this to the list of requests
	requests.push_back(req);

	// Finally set waiters to true if it's already true
	// or there are now threads waiting
	waiters = waiters || req.ready;
	
	return req.event;
    }

    void LockImpl::unlock(Event wait_on)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        log_lock(LEVEL_DEBUG,"unlock request: lock=%x mode=%d excl=%d event=%x/%d count=%d",
                 index, mode, exclusive, wait_on.id, wait_on.gen, holders);
        std::set<EventImpl*> to_trigger;
	if (wait_on.exists())
	{
		// Register this lock to be unlocked when the even triggers	
		EventImpl *e = Runtime::get_runtime()->get_event_impl(wait_on);
                // Use default handle 0 to indicate unlock event
		if (!(e->register_dependent(this,wait_on.gen)))
		{
			// The event didn't register which means it already triggered
			// so go ahead and perform the unlock operation
			perform_unlock(to_trigger);
		}	
	}
	else
	{
		// No need to wait to perform the unlock
		perform_unlock(to_trigger);		
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Don't perform any triggers while holding the lock's lock
        for (std::set<EventImpl*>::const_iterator it = to_trigger.begin();
              it != to_trigger.end(); it++)
        {
          (*it)->trigger();
        }
    }

    void LockImpl::trigger(unsigned count, TriggerHandle handle)
    {
        std::set<EventImpl*> to_trigger;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        // If the trigger handle is 0 then unlock the lock, 
        // otherwise find the lock request to wake up
        if (handle == 0)
        {
          perform_unlock(to_trigger);
        }
        else
        {
          bool found = false;
          // Go through the list and mark the matching request as being ready
          for (std::list<LockRecord>::iterator it = requests.begin();
                it != requests.end(); it++)
          {
            if (it->id == handle)
            {
              found = true;
#ifdef DEBUG_LOW_LEVEL
              assert(!it->ready);
#endif
              it->ready = true;
              // Try acquiring this lock just in case it is available,
              // otherwise we can just leave this request on the queue
              if (taken)
              {
                if (!exclusive && !it->exclusive && (mode == it->mode) && !waiters)
                {
                  holders++;
                  // Trigger the event saying we have the lock
                  to_trigger.insert(Runtime::get_runtime()->get_event_impl(it->event));
                  // Remove the request
                  requests.erase(it);
                }
                else
                {
                  // There are now definitely waiters
                  waiters = true;
                }
              }
              else // Nobody else has it, grab it!
              {
                taken = true;
                exclusive = it->exclusive;
                mode = it->mode; 
                holders = 1;
                // Trigger the event saying we have the lock
                to_trigger.insert(Runtime::get_runtime()->get_event_impl(it->event));
                // Remove this request
                requests.erase(it);
#ifdef DEBUG_PRINT
                  DPRINT3("Granting lock %d in mode %d with exclusive %d\n",index,mode,exclusive);
#endif
              }
              break;
            }
          }
#ifdef DEBUG_LOW_LEVEL
          assert(found);
#endif
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Don't perform any triggers while holding the lock's lock
        for (std::set<EventImpl*>::const_iterator it = to_trigger.begin();
              it != to_trigger.end(); it++)
        {
          (*it)->trigger();
        }
    }

    // Always called while holding the lock's mutex
    void LockImpl::perform_unlock(std::set<EventImpl*> &to_trigger)
    {
	holders--;	
	// If the holders are zero, get the next request out of the queue and trigger it
	if (holders==0)
	{
#ifdef DEBUG_PRINT
		DPRINT1("Unlocking lock %d\n",index);
#endif
		// Check to see if there are any waiters
		if (requests.empty())
		{
			waiters= false;
			taken = false;
			return;
		}
                // Clean out all the handled requests
                {
                  std::list<LockRecord>::iterator it = requests.begin();
                  while (it != requests.end())
                  {
                    if (it->handled)
                      it = requests.erase(it);
                    else
                      it++;
                  }
                }
		LockRecord req;
                bool found = false;
                for (std::list<LockRecord>::iterator it = requests.begin();
                      it != requests.end(); it++)
                {
                  if (it->ready)
                  {
                    req = *it;
                    it->handled = true;
                    found = true;
                    break;
                  }
                }
                // Check to see if we found a new candidate
                if (!found)
		{
			waiters = false;
			taken = false;
			return;
		}
		// Set the mode and exclusivity
		exclusive = req.exclusive;
		mode = req.mode;
		holders = 1;
#ifdef DEBUG_PRINT
		DPRINT3("Issuing lock %d in mode %d with exclusivity %d\n",index,mode,exclusive);
#endif
		// Trigger the event
                to_trigger.insert(Runtime::get_runtime()->get_event_impl(req.event));
		// If this isn't an exclusive mode, see if there are any other
		// requests with the same mode that aren't exclusive that we can handle
		if (!exclusive)
		{
			waiters = false;
			for (std::list<LockRecord>::iterator it = requests.begin();
				it != requests.end(); it++)
			{
                          if (it->ready)
                          {
				if ((it->mode == mode) && (!it->exclusive) && (!it->handled))
				{
					it->handled = true;
                                        to_trigger.insert(Runtime::get_runtime()->get_event_impl(it->event));
					holders++;
				}
				else
				{
					// There is at least one thread still waiting
					waiters = true;
				}
                          }
			}	
		}
		else
		{
			waiters = (requests.size()>0);
		}
	}
    }

    bool LockImpl::activate(size_t dsize)
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
#ifdef DEBUG_LOW_LEVEL
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

    void LockImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;	
        if (data_size > 0)
        {
#ifdef DEBUG_LOW_LEVEL
            assert(data != NULL);
#endif
            free(data);
            data = NULL;
            data_size = 0;
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    Lock LockImpl::get_lock(void) const
    {
#ifdef DEBUG_LOWL_LEVEL
        assert(index != 0);
#endif
	Lock l = { static_cast<id_t>(index) };
	return l;
    }

    size_t LockImpl::get_data_size(void) const
    {
        return data_size;
    }

    void* LockImpl::get_data_ptr(void) const
    {
        return data;
    }

    ////////////////////////////////////////////////////////
    // Processor 
    ////////////////////////////////////////////////////////

    /*static*/ const Processor Processor::NO_PROC = { 0 };

    // Processor Impl at top due to use in event
    
    Event Processor::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
	return p->spawn(func_id, args, arglen, wait_on);
    }

    Processor Processor::get_utility_processor(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
        return p->get_utility_processor();
    }

    const std::set<Processor>& Machine::get_local_processors(Processor p) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ProcessorImpl *pp = Runtime::get_runtime()->get_processor_impl(p);
        return pp->get_utility_users();
    }

    void Processor::enable_idle_task(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
        p->enable_idle_task();
    }

    void Processor::disable_idle_task(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
        p->disable_idle_task();
    }

    void ProcessorImpl::initialize_state(size_t stacksize)
    {
        mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
        wait_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
        PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
        PTHREAD_SAFE_CALL(pthread_cond_init(wait_cond,NULL));
        PTHREAD_SAFE_CALL(pthread_attr_init(&attr));
        PTHREAD_SAFE_CALL(pthread_attr_setstacksize(&attr,stacksize));
        shutdown = false;
        shutdown_trigger = NULL;
    }

    Event ProcessorImpl::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on)
    {
	TaskDesc task;
	task.func_id = func_id;
	task.args = malloc(arglen);
	memcpy(task.args,args,arglen);
	task.arglen = arglen;
	task.wait = wait_on;
	task.complete = Runtime::get_runtime()->get_free_event();
	Event result = task.complete->get_event();

	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (wait_on.exists())
	{
		// Try registering this processor with the event
		EventImpl *wait_impl = Runtime::get_runtime()->get_event_impl(wait_on);
		if (!wait_impl->register_dependent(this, wait_on.gen, wait_on.id))
		{
#ifdef DEBUG_PRINT
			DPRINT2("Registering task %d on processor %d ready queue\n",func_id,proc.id);
#endif
			// Failed to register which means it is ready to execute
			ready_queue.push_back(task);
			// If it wasn't registered, then the event triggered
			// Notify the processor thread in case it is waiting
			PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
		}	
		else
		{
#ifdef DEBUG_PRINT
			DPRINT2("Registering task %d on processor %d waiting queue\n",func_id,proc.id);
#endif
			// Successfully registered, put the task on the waiting queue
			waiting_queue.push_back(task);
		}
	}
	else
	{
#ifdef DEBUG_PRINT
		DPRINT2("Putting task %d on processor %d ready queue\n",func_id,proc.id);
#endif
		// Put it on the ready queue
		ready_queue.push_back(task);
		// Signal the thread there is a task to run in case it is waiting
		PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    Processor ProcessorImpl::get_utility_processor(void) const
    {
        return utility;
    }

    void ProcessorImpl::release_user()
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
      assert(remaining_stops > 0);
#endif
      remaining_stops--;
      // If we've had all our users released, we can shutdown
      if (remaining_stops == 0)
      {
        shutdown = true;
      }
      // Signal in case the utility processor is waiting on work
      PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::utility_finish(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
      assert(!is_utility_proc);
      assert(!util_shutdown);
#endif
      // Set util shutdown to true
      util_shutdown = true;
      // send a signal in case the processor was waiting
      PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    const std::set<Processor>& ProcessorImpl::get_utility_users(void) const
    {
      if (is_utility_proc)
        return util_users;
      else
        return utility_proc->get_utility_users();
    }

    void ProcessorImpl::add_utility_user(Processor p, ProcessorImpl *impl)
    {
      util_users.insert(p);
      constituents.push_back(impl);
      // Increment the number of idle tasks since processors start
      // off with their idle tasks enabled
      num_idle_tasks++;
    }

    void ProcessorImpl::enable_idle_task(void)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        if (!idle_task_enabled && (utility_proc != this))
          utility_proc->increment_utility();
        idle_task_enabled = true;
        // Wake up thread so it can run the idle task
        PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::disable_idle_task(void)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        if (idle_task_enabled && (utility_proc != this))
          utility_proc->decrement_utility();
        idle_task_enabled = false;    
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::increment_utility(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (num_idle_tasks == 0)
        PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
      num_idle_tasks++;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::decrement_utility(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
      assert(num_idle_tasks > 0);
#endif
      num_idle_tasks--;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::run(void)
    {
        //fprintf(stdout,"This is processor %d\n",proc.id);
        //fflush(stdout);
        // Check to see if there is an initialization task
        if (task_table.find(Processor::TASK_ID_PROCESSOR_INIT) != task_table.end())
        {
          Processor::TaskFuncPtr func = task_table[Processor::TASK_ID_PROCESSOR_INIT];
          func(NULL, 0, proc);
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
#if DEBUG_LOW_LEVEL
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
        //fprintf(stdout,"Processor %d is starting\n",proc.id);
        //fflush(stdout);
	// Processors run forever and permit shutdowns
	while (true)
	{
		// Make sure we're holding the lock
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
		// This task will perform the unlock
		execute_task(true);
	}
    }

    void ProcessorImpl::preempt(EventImpl *event, EventImpl::EventGeneration needed)
    {
	// Try registering this processor with the event in case it goes to sleep
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!(event->register_dependent(this, needed)))
	{
		// The even triggered, release the lock and return
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
		return;
	}
        // have to hold the lock here when testing this
        // so we don't accidentally miss a wake-up when
        // going to sleep.
	while (!(event->has_triggered(needed)))
	{
                // Don't permit shutdowns since there is still a task waiting
		execute_task(false);
                // Relock the task for our next attempt
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	}
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    bool ProcessorImpl::perform_scheduling(bool need_lock)
    {
      if (need_lock)
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      // See if we should invoke the scheduler
      if (has_scheduler && idle_task_enabled && !scheduler_invoked
          && !shutdown && ready_queue.empty())
      {
        scheduler_invoked = true;
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        Processor::TaskFuncPtr scheduler = task_table[Processor::TASK_ID_PROCESSOR_IDLE];
        scheduler(NULL, 0, proc);
        // Return from the scheduler, so we can reevaluate status
        scheduler_invoked = false;
        // Lock released
        return true;
      }
      // If we acquired the lock then we should always unlock it
      if (need_lock)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        return true;
      }
      // Otherwise the scheduler wasn't invoked so we still hold the lock
      return false;
    }

    // Must always be holding the lock when calling this task
    // This task will always unlock it
    void ProcessorImpl::execute_task(bool permit_shutdown)
    {
        // Look through the waiting queue, to see if any tasks
        // have been woken up	
        	
        // If we don't have any work to do, check to see
        // if we can run the idle task.  Also if we're the utility
        // processor, and we don't have anything to do then try
        // running the idle task for each of our constituents.
        if (ready_queue.empty() && perform_scheduling(false/*need lock*/))
        {
          // If we return true then we've release the lock so we
          // have to go back around the loop when we're done
          // Now we need to return since we no longer hold the lock
          return;
        }
        else if (is_utility_proc && !shutdown && ready_queue.empty())
        {
          if (num_idle_tasks == 0)
          {
            PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond,mutex));
          }
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
          // Note we don't need the lock to read these
          // since they don't change after the constructor is called
          for (std::vector<ProcessorImpl*>::const_iterator it = constituents.begin();
                it != constituents.end(); it++)
          {
            // We should never be in our own list of constituents
#ifdef DEBUG_LOW_LEVEL
            assert((*it) != this);
#endif
            (*it)->perform_scheduling(true/*need lock*/);
          }
          // Return since we no longer hold the lock
          return;
        }
	if (ready_queue.empty())
	{	
		if (shutdown && permit_shutdown && waiting_queue.empty())
		{
                        // Check to see if we have to wait for our utility processor to finish
                        if (!util_shutdown)
                        {
                          DetailedTimer::ScopedPush sp(TIME_NONE);
                          // Wait for our utility processor to indicate that its done
                          PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond,mutex));
                        }
                        // unlock the lock, just in case someone else decides they want to tell us something
                        // to do even though we've already exited
                        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
                        // Check to see if there is a shutdown method
                        if (task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN) != task_table.end())
                        {
                          // If there is, call the shutdown method before triggering
                          Processor::TaskFuncPtr func = task_table[Processor::TASK_ID_PROCESSOR_SHUTDOWN];
                          func(NULL, 0, proc);
                        }
                        // If we don't have any utility users or we are our own
                        // utility processor, then we are done
                        if (util_users.empty() || 
                            ((util_users.size() == 1) && (util_users.find(proc) != util_users.end())))
                        {
                          shutdown_trigger->trigger();
                        }
                        else
                        {
                          // Send shutdown messages to all our users that aren't us
                          for (std::set<Processor>::const_iterator it = util_users.begin();
                                it != util_users.end(); it++)
                          {
                            // Skip ourselves
                            if ((*it) == proc)
                              continue;
                            ProcessorImpl *orig = Runtime::get_runtime()->get_processor_impl(*it);
                            orig->utility_finish();
                          }
                        }
                        pthread_exit(NULL);	
		}
		
		// Wait until someone tells us there is work to do unless we've been told to shutdown
                if (!shutdown)
                {
                  DetailedTimer::ScopedPush sp(TIME_NONE);
                  PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond,mutex));
                }
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	}
        else if (scheduler_invoked)
        {
                // Don't allow other tasks to be run while running the idle task
                PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
                return;
        }
	else
	{
		// Pop a task off the queue and run it
		TaskDesc task = ready_queue.front();
		ready_queue.pop_front();
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));	
		// Check for the shutdown function
		if (task.func_id == 0)
		{
                        shutdown = true;
                        shutdown_trigger = task.complete;
                        // Check to see if we have a utility processor, if so mark that we're done
                        // and then set the flag to indicate when the utility processor has drained
                        // its tasks
                        if (!is_utility_proc && (utility_proc != this))
                        {
                          util_shutdown = false;
                          // Tell our utility processor to tell us when it's done
                          utility_proc->release_user();
                        }
                        else
                        {
                          // We didn't have a utility processor to shutdown
                          util_shutdown = true;
                        }
			// Continue going around until all tasks are run
			return;
		}
#ifdef DEBUG_LOW_LEVEL
		assert(task_table.find(task.func_id) != task_table.end());
#endif
		Processor::TaskFuncPtr func = task_table[task.func_id];	
		func(task.args, task.arglen, proc);
		// Trigger the event indicating that the task has been run
		task.complete->trigger();
		// Clean up the mess
		if (task.arglen > 0)
			free(task.args);
	}
    }

    void ProcessorImpl::trigger(unsigned count, TriggerHandle handle)
    {
	// We're not sure which task is ready, but at least one of them is
	// so wake up the processor thread if it is waiting
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        // The trigger handle is the ID of the event we were
        // waiting on.  Move any tasks in the waiting queue
        // waiting on that event over to the ready queue
        // and then wake up the processor.
        for (std::list<TaskDesc>::iterator it = waiting_queue.begin();
              it != waiting_queue.end(); /*nothing*/)
        {
          if ((it->wait.id == handle) && it->wait.has_triggered())
          {
            ready_queue.push_back(*it);
            it = waiting_queue.erase(it);
          }	
          else
            it++;
        }
	PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    // The static method used to start the processor running
    void* ProcessorImpl::start(void *p)
    {
	ProcessorImpl *proc = (ProcessorImpl*)p;
	// Set the thread local variable processor id
	//local_proc_id = proc->proc.id;
        {
          unsigned *thread_id = (unsigned*)malloc(sizeof(unsigned));
          *thread_id = proc->proc.id;
          PTHREAD_SAFE_CALL( pthread_setspecific(local_proc_key, thread_id) );
        }
        // Also set the value of thread timer key
        PTHREAD_SAFE_CALL( pthread_setspecific(thread_timer_key, NULL) );
	// Will never return from this call
	proc->run();
        if (!proc->return_on_finish)
          pthread_exit(NULL);	
        return NULL;
    }

    ////////////////////////////////////////////////////////
    // Memory 
    ////////////////////////////////////////////////////////
    
    const Memory Memory::NO_MEMORY = {0};

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
#ifdef DEBUG_LOW_LEVEL
		assert(ptr != NULL);
#endif
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return ptr;
    }

    void MemoryImpl::free_space(void *ptr, size_t size)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
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
      raw_data = calloc(1, bytes_needed);
      //((ElementMaskImpl *)raw_data)->count = num_elements;
      //((ElementMaskImpl *)raw_data)->offset = first_element;
    }

    ElementMask::ElementMask(const ElementMask &copy_from, 
			     int _num_elements /*= -1*/, int _first_element /*= 0*/)
    {
      first_element = copy_from.first_element;
      num_elements = copy_from.num_elements;
      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);

      if(copy_from.raw_data) {
	memcpy(raw_data, copy_from.raw_data, bytes_needed);
      } else {
        assert(false);
      }
    }

    ElementMask& ElementMask::operator=(const ElementMask &rhs)
    {
      first_element = rhs.first_element;
      num_elements = rhs.num_elements;
      first_enabled_elmt = rhs.first_enabled_elmt;
      last_enabled_elmt = rhs.last_enabled_elmt;
      size_t bytes_needed = rhs.raw_size();
      raw_data = calloc(1, bytes_needed);
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
        int max_full = (num_elements >> 5);
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
#ifdef DEBUG_LOW_LEVEL
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


    ////////////////////////////////////////////////////////
    // IndexSpace::Impl (Declaration Only) 
    ////////////////////////////////////////////////////////

    class IndexSpace::Impl {
    public:
	Impl(int idx, size_t num, bool activate = false) {
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		index = idx;
		if (activate)
		{
			num_elmts = num;
			lock = Runtime::get_runtime()->get_free_lock();
                        mask = ElementMask(num_elmts);
                        parent = NULL;
		}
	}

        Impl(int idx, IndexSpace::Impl *par, const ElementMask &m, bool activate = false) {
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
                PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		index = idx;
		if (activate)
		{
			num_elmts = m.get_num_elmts();
	                // Since we have a parent, use the parent's master allocator	
			lock = Runtime::get_runtime()->get_free_lock();
                        mask = m;
                        parent = par;
		}
        }

        ~Impl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	bool activate(size_t num_elmts);
        bool activate(IndexSpace::Impl *par, const ElementMask &m);
	void deactivate(void);	
	IndexSpace get_metadata(void);

        IndexSpaceAllocator create_allocator(void);

        RegionInstance create_instance(Memory m, 
				       const std::vector<size_t>& field_sizes,
				       size_t block_size, 
				       const DomainLinearization& dl,
				       size_t num_elements,
				       ReductionOpID redopid = 0);

	void destroy_instance(RegionInstance i);

	Lock get_lock(void);

        const ElementMask& get_element_mask(void);

        size_t get_num_elmts(void) const { return num_elmts; }

        Event copy(RegionInstance src_inst, RegionInstance dst_inst,
		   size_t elem_size, const Domain domain, Event wait_on = Event::NO_EVENT,
		   ReductionOpID redop_id = 0, bool red_fold = false);

        Event copy(const std::vector<Domain::CopySrcDstField>& srcs,
		   const std::vector<Domain::CopySrcDstField>& dsts,
		   const Domain domain,
		   Event wait_on,
		   ReductionOpID redop_id = 0, bool red_fold = false);

        class CopyOperation : public Triggerable {
	public:
	  CopyOperation(const std::vector<Domain::CopySrcDstField>& _srcs,
			const std::vector<Domain::CopySrcDstField>& _dsts,
			const Domain _domain,
			ReductionOpID _redop_id, bool _red_fold,
			EventImpl *_done_event)
	    : srcs(_srcs), dsts(_dsts), 
	      domain(_domain),
	      redop_id(_redop_id), red_fold(_red_fold),
	      done_event(_done_event) 
          {
            PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));    
          }

          ~CopyOperation(void)
          {
            PTHREAD_SAFE_CALL(pthread_mutex_destroy(&mutex));
          }

	  virtual void trigger(unsigned count = 1, TriggerHandle handle = 0)
	  {
#ifdef LEGION_LOGGING
            LegionRuntime::HighLevel::LegionLogging::log_timing_event(
                                          Machine::get_executing_processor(),
                                          done_event->get_event(), COPY_READY);
#endif
	    perform_copy_operation();
            // Trigger the done event if it exists when we're done
            // Hold the lock when reading done event since we need to make
            // sure it has been set if it needs to be set
            PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
            EventImpl *done_clone = done_event;
            PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
            if (done_clone)
              done_clone->trigger();
	    delete this;
	  }

	  // registers the copy event with the before_event - returns true if successful, false if not
	  //  (in which case the copy is performed immediately)
	  Event register_copy(Event wait_on)
	  {
            Event result = Event::NO_EVENT;
            EventImpl *done_clone = NULL;
	    if (wait_on.exists()) {
	      // Try registering this as a triggerable with the event	
	      EventImpl *event_impl = Runtime::get_runtime()->get_event_impl(wait_on);
              // Need to hold the mutex here in case we have to set the done_event
              // to make sure it gets set before trigger is called
              PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	      if (event_impl->register_dependent(this, wait_on.gen, 0)) {
		// make sure we have a completion event
		if (!done_event)
		  done_event = Runtime::get_runtime()->get_free_event();
                result = done_event->get_event();
                PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
                return result;
	      }
              done_clone = done_event;
              PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	    }
#ifdef LEGION_LOGGING
            LegionRuntime::HighLevel::LegionLogging::log_timing_event(
                                          Machine::get_executing_processor(),
                                          done_event->get_event(), COPY_INIT);
#endif

	    // either there was no wait event or it has already fired
	    perform_copy_operation();
            if (done_clone)
            {
#ifdef LEGION_LOGGING
              if (!result.exists())
                result = done_clone->get_event();
#endif
              done_clone->trigger();
            }
            // We're done, so we can free ourselves
            delete this;
            return result;
	  }

	protected:
	  void perform_copy_operation(void);
	  std::vector<Domain::CopySrcDstField> srcs;
	  std::vector<Domain::CopySrcDstField> dsts;
	  Domain domain;
	  ReductionOpID redop_id;
	  bool red_fold;
	  EventImpl *done_event;
          pthread_mutex_t mutex;
	};

    public:
        // Traverse up the tree to the parent region that owns the master allocator
        // Peform the operation and then update the element mask on the way back down
        unsigned allocate_space(unsigned count);
        void     free_space(unsigned ptr, unsigned count);
    private:
	//std::set<RegionAllocatorUntyped> allocators;
	std::set<RegionInstance> instances;
	LockImpl *lock;
	pthread_mutex_t *mutex;
	bool active;
	int index;
	size_t num_elmts;
        ElementMask mask;
        IndexSpace::Impl *parent;
    };

    
    ////////////////////////////////////////////////////////
    // Region Allocator 
    ////////////////////////////////////////////////////////

    class IndexSpaceAllocator::Impl {
    public:
      Impl(IndexSpace is)
        : is_impl(Runtime::get_runtime()->get_metadata_impl(is))
      {
	mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
      }
      
      ~Impl(void)
      {
	PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
	::free(mutex);
      }

    public:
      unsigned alloc_elmt(size_t num_elmts = 1);
      void free_elmt(unsigned ptr, unsigned count);
      bool activate(IndexSpace::Impl *owner);
      void deactivate();
      //IndexSpaceAllocator get_allocator(void) const;
      //Lock get_lock(void);

    private:
      IndexSpace::Impl *is_impl;
      pthread_mutex_t *mutex;
    }; 

    unsigned IndexSpaceAllocator::alloc(unsigned count /*= 1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return impl->alloc_elmt(count);
    }

    void IndexSpaceAllocator::free(unsigned ptr, unsigned count /*= 1 */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      impl->free_elmt(ptr, count);
    }

    void IndexSpaceAllocator::destroy(void)
    {
      if (impl != NULL)
      {
        delete impl;
        // Avoid double frees
        impl = NULL;
      }
    }

    unsigned IndexSpaceAllocator::Impl::alloc_elmt(size_t num_elmts)
    {
      // No need to hold the lock since we're just reading
      return is_impl->allocate_space(num_elmts);
    }

    void IndexSpaceAllocator::Impl::free_elmt(unsigned ptr, unsigned count)
    {
      // No need to hold the lock since we're just reading
      is_impl->free_space(ptr,count);
    }
    
    ////////////////////////////////////////////////////////
    // Region Instance 
    ////////////////////////////////////////////////////////

    class RegionInstance::Impl : public Triggerable { 
    public:
        Impl(int idx, IndexSpace r, Memory m, size_t num, 
	     const std::vector<size_t>& _field_sizes,
	     size_t elem_size, size_t _block_size,
	     const DomainLinearization& _dl,
	     bool activate = false, char *base = NULL, const ReductionOpUntyped *op = NULL,
	     RegionInstance::Impl *parent = NULL)
	  : elmt_size(elem_size), num_elmts(num), field_sizes(_field_sizes), block_size(_block_size), linearization(_dl),
	    reduction((op!=NULL)), list((parent!=NULL)), redop(op), parent_impl(parent), cur_entry(0), index(idx), next_handle(1)
	{
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		if (active)
		{
		        region = r;
			memory = m;
			// Use the memory to allocate the space, fail if there is none
			//MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
			base_ptr = base; //(char*)mem->allocate_space(num_elmts*elem_size);	
#ifdef DEBUG_LOW_LEVEL
			assert(base_ptr != NULL);
#endif
			lock = Runtime::get_runtime()->get_free_lock();
		}
	}

        ~Impl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	const void* read(unsigned ptr);
	void write(unsigned ptr, const void* newval);	
        bool activate(IndexSpace r, Memory m, size_t num_elmts, 
		      const std::vector<size_t>& _field_sizes, size_t elem_size, size_t _block_size,
		      const DomainLinearization& _dl,
                      char *base, const ReductionOpUntyped *op, RegionInstance::Impl *parent);
	void deactivate(void);
	Event copy_to(RegionInstance target, Event wait_on);
        Event copy_to(RegionInstance target, const ElementMask &mask, Event wait_on);
        Event copy_to(RegionInstance target, IndexSpace src_region, Event wait_on);
	RegionInstance get_instance(void) const;
	void trigger(unsigned count, TriggerHandle handle);
	Lock get_lock(void);
        void perform_copy_operation(RegionInstance::Impl *target, const ElementMask &src_mask, const ElementMask &dst_mask);
        void apply_list(RegionInstance::Impl *target);
        void append_list(RegionInstance::Impl *target);
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
    private:
        class CopyOperation2 {
        public:
          RegionInstance::Impl *target;
          EventImpl *complete;
          TriggerHandle id;
          const ElementMask &src_mask;
          const ElementMask &dst_mask;
        public:
          CopyOperation2(RegionInstance::Impl *t, EventImpl *c, TriggerHandle i, 
                        const ElementMask &s, const ElementMask &d)
            : target(t), complete(c), id(i), src_mask(s), dst_mask(d) { }
        };
    private:
        IndexSpace region;
	char *base_ptr;	
	size_t elmt_size;
	size_t num_elmts;
        std::vector<size_t> field_sizes;
        size_t block_size;
        DomainLinearization linearization;
	Memory memory;
	pthread_mutex_t *mutex;
        bool reduction; // reduction fold
        bool list; // reduction list
        const ReductionOpUntyped *redop; // for all reductions
        RegionInstance::Impl *parent_impl; // for lists
        size_t cur_entry; // for lists
	bool active;
	const int index;
	// Fields for the copy operation
	LockImpl *lock;
        TriggerHandle next_handle;
        std::list<CopyOperation2> pending_copies;
    };

    /*static*/ const RegionInstance RegionInstance::NO_INST = { 0 };

    RegionAccessor<AccessorType::Generic> RegionInstance::get_accessor(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstance::Impl *impl = Runtime::get_runtime()->get_instance_impl(*this);
      return RegionAccessor<AccessorType::Generic>(AccessorType::Generic::Untyped(impl));
    }

    void RegionInstance::destroy(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstance::Impl *impl = Runtime::get_runtime()->get_instance_impl(*this);
      impl->deactivate();
    }

#ifdef OLD_INTFC
    Event RegionInstance::copy_to_untyped(RegionInstance target, Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Runtime::get_runtime()->get_instance_impl(*this)->copy_to(target,wait_on);
    }

    Event RegionInstance::copy_to_untyped(RegionInstance target, const ElementMask &mask,
                                                Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Runtime::get_runtime()->get_instance_impl(*this)->copy_to(target,mask,wait_on);
    }

    Event RegionInstance::copy_to_untyped(RegionInstance target, IndexSpace region,
                                                 Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Runtime::get_runtime()->get_instance_impl(*this)->copy_to(target,region,wait_on);
    }
#endif

    const void* RegionInstance::Impl::read(unsigned ptr)
    {
      // 'ptr' has already been multiplied by elmt_size
      return ((void*)(base_ptr + ptr));
    }

    void RegionInstance::Impl::write(unsigned ptr, const void* newval)
    {
      // 'ptr' has already been multiplied by elmt_size
      memcpy((base_ptr + ptr),newval,elmt_size);
    }

    bool RegionInstance::Impl::activate(IndexSpace r, Memory m, size_t num, 
					const std::vector<size_t>& _field_sizes,
					size_t elem_size, size_t _block_size,
					const DomainLinearization& _dl,
					char *base, const ReductionOpUntyped *op, RegionInstance::Impl *parent)
    {
	bool result = false;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{
		active = true;
		result = true;
		region = r;
		memory = m;
		num_elmts = num;
		field_sizes = _field_sizes;
		elmt_size = elem_size;
		block_size = _block_size;
		linearization = _dl;
		//MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
		base_ptr = base; //(char*)mem->allocate_space(num_elmts*elmt_size);
                redop = op;
                reduction = (redop != NULL);
                parent_impl = parent;
                list = (parent != NULL);
                cur_entry = 0;
#ifdef DEBUG_LOW_LEVEL
		assert(base_ptr != NULL);
#endif
		lock = Runtime::get_runtime()->get_free_lock();
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void RegionInstance::Impl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;
	MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(memory);
	mem->free_space(base_ptr,num_elmts*elmt_size);
	num_elmts = 0;
	field_sizes.clear();
	elmt_size = 0;
	block_size = 0;
	base_ptr = NULL;	
        redop = NULL;
        reduction = false;
        parent_impl = NULL;
        list = false;
	lock->deactivate();
	lock = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    Logger::Category log_copy("copy");

    Event RegionInstance::Impl::copy_to(RegionInstance target, Event wait_on)
    {
      return copy_to(target,region,wait_on);
    }

    Event RegionInstance::Impl::copy_to(RegionInstance target, IndexSpace src_region, Event wait_on)
    {
      const ElementMask &mask = src_region.get_valid_mask();
      return copy_to(target,mask,wait_on);
    }

    Event RegionInstance::Impl::copy_to(RegionInstance target, const ElementMask &mask, Event wait_on)
    {
	RegionInstance::Impl *target_impl = Runtime::get_runtime()->get_instance_impl(target);
        const ElementMask &target_mask = target_impl->region.get_valid_mask();
	//log_copy(LEVEL_INFO, "copy %x/%p/%x -> %x/%p/%x", index, this, region.id, target.id, target_impl, target_impl->region.id);
#ifdef DEBUG_LOW_LEVEL
	assert(target_impl->num_elmts == num_elmts);
	assert(target_impl->elmt_size == elmt_size);
#endif
	// Check to see if the event exists
	if (wait_on.exists())
	{
		// Try registering this as a triggerable with the event	
		EventImpl *event_impl = Runtime::get_runtime()->get_event_impl(wait_on);
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
		if (event_impl->register_dependent(this,wait_on.gen,next_handle))
		{
                        CopyOperation2 op(target_impl,Runtime::get_runtime()->get_free_event(),
                                          next_handle,mask,target_mask);
                        // Put it in the list of copy operations
                        pending_copies.push_back(op);
                        next_handle++;
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
			return op.complete->get_event();
		}
		else
		{
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
                        // Nothing to wait for
                        // Fall through and perform the copy
		}
	}
        perform_copy_operation(target_impl,mask,target_mask);
        return Event::NO_EVENT;
    }

    void RegionInstance::Impl::trigger(unsigned count, TriggerHandle handle)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        // Find the copy operation in the set
        bool found = false;
        EventImpl *complete = NULL; 
        for (std::list<CopyOperation2>::iterator it = pending_copies.begin();
              it != pending_copies.end(); it++)
        {
          if (it->id == handle)
          {
            found = true;
            perform_copy_operation(it->target,it->src_mask,it->dst_mask);
            complete = it->complete;
            // Remove it from the list
            pending_copies.erase(it);
            break;
          }
        }
#ifdef DEBUG_LOW_LEVEL
        assert(found);
#endif
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Trigger the event saying we're done while not holding the lock!
        complete->trigger();
    }

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

    void RegionInstance::Impl::perform_copy_operation(RegionInstance::Impl *target, const ElementMask &src_mask, const ElementMask &dst_mask)
    {
        DetailedTimer::ScopedPush sp(TIME_COPY); 
        const void *src_ptr = base_ptr;
        void       *tgt_ptr = target->base_ptr;
#ifdef DEBUG_LOW_LEVEL
        assert((src_ptr != NULL) && (tgt_ptr != NULL));
#endif
        if (!reduction)
        {
#ifdef DEBUG_LOW_LEVEL
          if (target->reduction)
          {
             fprintf(stderr,"Cannot copy from non-reduction instance %d to reduction instance %d\n",
                      this->index, target->index);
             exit(1);
          }
#endif
          // This is a normal copy
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
#ifdef DEBUG_LOW_LEVEL
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
#ifdef DEBUG_LOW_LEVEL
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

    void RegionInstance::Impl::apply_list(RegionInstance::Impl *target)
    {
#ifdef DEBUG_LOW_LEVEL
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

    void RegionInstance::Impl::append_list(RegionInstance::Impl *target)
    {
#ifdef DEBUG_LOW_LEVEL
        assert(this->list);
        assert(target->list);
#endif
        // TODO: Implement this
        assert(false);
    }

    RegionInstance RegionInstance::Impl::get_instance(void) const
    {
	RegionInstance inst;
	inst.id = index;
	return inst;
    }

    Lock RegionInstance::Impl::get_lock(void)
    {
	return lock->get_lock();
    }

    void RegionInstance::Impl::verify_access(unsigned ptr)
    {
      const ElementMask &mask = region.get_valid_mask();
      if (!mask.is_set(ptr))
      {
        fprintf(stderr,"ERROR: Accessing invalid pointer %d in logical region %d\n",ptr,index);
	assert(0);
      }
    }

    void* RegionInstance::Impl::get_address(int index, size_t field_start, size_t field_size,
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
      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data;
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
      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data;
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
      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data;
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
      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data;
      return impl->is_reduction();
    }

    // Accessor Generic (convert)
    template <>
    RegionAccessor<AccessorGeneric> RegionAccessor<AccessorGeneric>::convert<AccessorGeneric>(void) const
    { return *this; }

    template<>
    RegionAccessor<AccessorArray> RegionAccessor<AccessorGeneric>::convert<AccessorArray>(void) const
    { 
#ifdef DEBUG_LOW_LEVEL
      assert(!this->is_reduction_only());
#endif
      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data;
      RegionAccessor<AccessorArray> ret(impl->get_base_ptr()); 
#ifdef POINTER_CHECKS
      ret.impl_ptr = impl;
#endif
      return ret;
    }

    template<>
    RegionAccessor<AccessorArrayReductionFold> RegionAccessor<AccessorGeneric>::convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data;
#ifdef DEBUG_LOW_LEVEL
      assert(impl->is_reduction() && !impl->is_list_reduction());
#endif
      return RegionAccessor<AccessorArrayReductionFold>(impl->get_base_ptr());
    }

    template<>
    RegionAccessor<AccessorReductionList> RegionAccessor<AccessorGeneric>::convert<AccessorReductionList>(void) const
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data; 
#ifdef DEBUG_LOW_LEVEL
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

      RegionInstance::Impl *impl = (RegionInstance::Impl*)internal_data;
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
        ((RegionInstance::Impl*)internal_data)->verify_access(ptr);
    }

#if 0
    void RegionAccessor<AccessorArray>::verify_access(unsigned ptr) const
    {
        ((RegionInstance::Impl*)impl_ptr)->verify_access(ptr);
    }
#endif
#endif
#endif

    ////////////////////////////////////////////////////////
    // IndexSpace 
    ////////////////////////////////////////////////////////

    /*static*/ const IndexSpace IndexSpace::NO_SPACE = IndexSpace();
    /*static*/ const Domain Domain::NO_DOMAIN = Domain();

    // Lifting Declaration of IndexSpace::Impl above allocator so we can call it in allocator
    
    Logger::Category log_region("region");

    IndexSpace IndexSpace::create_index_space(size_t num_elmts)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpace::Impl *r = Runtime::get_runtime()->get_free_metadata(num_elmts);	
	//log_region(LEVEL_INFO, "index space created: id=%x num=%zd",
        //		   r->get_metadata().id, num_elmts);
	return r->get_metadata();
    }

    IndexSpace IndexSpace::create_index_space(IndexSpace parent, const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpace::Impl *par = Runtime::get_runtime()->get_metadata_impl(parent);
      IndexSpace::Impl *r = Runtime::get_runtime()->get_free_metadata(par, mask);
      //log_region(LEVEL_INFO, "index space created: id=%x parent=%x",
      //		 r->get_metadata().id, parent.id);
      return r->get_metadata();
    }

    IndexSpaceAllocator IndexSpace::create_allocator(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_allocator();
    }

    RegionInstance Domain::create_instance(Memory m, size_t elmt_size,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      std::vector<size_t> field_sizes(1);
      field_sizes[0] = elmt_size;
      return create_instance(m, field_sizes, 1, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
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
	      Arrays::CArrayLinearization<1> cl(get_rect<1>());
	      dl = DomainLinearization::from_mapping<1>(Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
	      inst_extent = cl.image_convex(get_rect<1>());
	      break;
	    }

	  case 2:
	    {
	      Arrays::CArrayLinearization<2> cl(get_rect<2>());
	      dl = DomainLinearization::from_mapping<2>(Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
	      inst_extent = cl.image_convex(get_rect<2>());
	      break;
	    }

	  case 3:
	    {
	      Arrays::CArrayLinearization<3> cl(get_rect<3>());
	      dl = DomainLinearization::from_mapping<3>(Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
	      inst_extent = cl.image_convex(get_rect<3>());
	      break;
	    }

	  default: assert(0);
	  }
	  IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(get_index_space());
	  // HACK - override block size to force SOA for now
	  block_size = inst_extent.volume();
	  return r->create_instance(memory, field_sizes, block_size, dl, int(inst_extent.hi) + 1, redop_id);
	} else {
	  IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(get_index_space());
	  return r->create_instance(memory, field_sizes, block_size, 
				    DomainLinearization(), r->get_num_elmts(),
				    redop_id);
	}
    }

#if 0
    RegionInstance IndexSpace::create_instance(Memory m, ReductionOpID redop) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(*this);
        return r->create_instance(m, redop);
    }

    RegionInstance IndexSpace::create_instance(Memory m, ReductionOpID redop,
                                                  off_t list_size, RegionInstance parent_inst) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(*this);
        return r->create_instance(m, redop, list_size, parent_inst);
    }
#endif

    void IndexSpace::destroy(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(*this);
        r->deactivate();
    }

#if 0
    void IndexSpace::destroy_allocator(IndexSpaceAllocator a) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_allocator(a);
    }

    void IndexSpace::destroy_instance(RegionInstance i) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_instance(i);
    }
#endif

    const ElementMask &IndexSpace::get_valid_mask(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(*this);
      return r->get_element_mask();
    }

    Event Domain::copy(RegionInstance src_inst, RegionInstance dst_inst, size_t elem_size,
		       Event wait_on /*= Event::NO_EVENT*/,
		       ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(get_index_space());
      return r->copy(src_inst, dst_inst, elem_size, *this, wait_on, redop_id, red_fold);
    }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       Event wait_on,
		       ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(get_index_space());
      return r->copy(srcs, dsts, *this, wait_on, redop_id, red_fold);
    }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       const ElementMask& mask, Event wait_on,
		       ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(get_index_space());
      assert(0);
      return r->copy(srcs, dsts, *this, wait_on, redop_id, red_fold);
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

    bool IndexSpace::Impl::activate(size_t num)
    {
	bool result = false;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{ 
		active = true;
		result = true;
		num_elmts = num;
		lock = Runtime::get_runtime()->get_free_lock();
                mask = ElementMask(num_elmts);
                parent = NULL;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    bool IndexSpace::Impl::activate(IndexSpace::Impl *par, const ElementMask &m)
    {
      bool result = false;
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (!active)
      {
        active = true;
        result = true;
        num_elmts = m.get_num_elmts();
        lock = Runtime::get_runtime()->get_free_lock();
        mask = m;
        parent = par;
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    void IndexSpace::Impl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;
	num_elmts = 0;
        // Mike: The High Level Runtime is responsible for deleting instances!
#if 0
	for (std::set<RegionInstance>::iterator it = instances.begin();
		it != instances.end(); it++)
	{
		RegionInstance::Impl *instance = Runtime::get_runtime()->get_instance_impl(*it);
		instance->deactivate();
	}	
#endif
	instances.clear();
	lock->deactivate();
	lock = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    unsigned IndexSpace::Impl::allocate_space(unsigned count)
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
#ifdef DEBUG_LOW_LEVEL
        assert(result >= 0);
#endif
        // Update the mask to reflect the allocation
        mask.enable(result,count);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        return unsigned(result);
    }

    void IndexSpace::Impl::free_space(unsigned ptr, unsigned count)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
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

    IndexSpace IndexSpace::Impl::get_metadata(void)
    {
	IndexSpace meta;
	meta.id = index;
	return meta;
    }

    const ElementMask& IndexSpace::Impl::get_element_mask(void)
    {
#ifdef DEBUG_LOW_LEVEL
      assert(active);
#endif
      return mask;
    }

    IndexSpaceAllocator IndexSpace::Impl::create_allocator(void)
    {
      IndexSpaceAllocator::Impl *alloc_impl = new IndexSpaceAllocator::Impl(get_metadata());
      return IndexSpaceAllocator(alloc_impl);
    }

    RegionInstance IndexSpace::Impl::create_instance(Memory m,
						     const std::vector<size_t>& field_sizes,
						     size_t block_size, 
						     const DomainLinearization& dl,
						     size_t num_elements,
						     ReductionOpID redop_id /*=0*/)
    {
        if (!m.exists())
        {
          return RegionInstance::NO_INST;
        }
        // First try to create the location in the memory, if there is no space
        // don't bother trying to make the data
        MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);

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
	if(redop_id) {
	  const ReductionOpUntyped *redop = Runtime::get_runtime()->get_reduction_op(redop_id);
	  assert(redop->has_identity);
	  assert(elmt_size == redop->sizeof_rhs);
	  redop->init(ptr, rounded_num_elmts);
	}

	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	IndexSpace r = { static_cast<id_t>(index) };
	RegionInstance::Impl* impl = Runtime::get_runtime()->get_free_instance(r, m,
									       num_elements, 
									       field_sizes,
									       elmt_size, 
									       block_size, dl,
									       ptr, NULL/*redop*/, NULL/*parent instance*/);
	RegionInstance inst = impl->get_instance();
	instances.insert(inst);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return inst;
    }

#if 0
    RegionInstance IndexSpace::Impl::create_instance(Memory m, ReductionOpID redopid, off_t list_size,
                                                              RegionInstance parent_inst) 
    {
        if (!m.exists())
        {
            return RegionInstance::NO_INST; 
        }
        MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
 // There must be a reduction operation for a list instance
#ifdef DEBUG_LOW_LEVEL
        assert(redopid > 0);
#endif
        const ReductionOpUntyped *op = Runtime::get_runtime()->get_reduction_op(redopid); 
        char *ptr = (char*)mem->allocate_space(list_size * (op->sizeof_rhs + sizeof(utptr_t)));
        if (ptr == NULL)
        {
            return RegionInstance::NO_INST;
        }
        // Set everything up
        RegionInstance::Impl *parent_impl = Runtime::get_runtime()->get_instance_impl(parent_inst);
#ifdef DEBUG_LOW_LEVEL
        assert(parent_impl != NULL);
#endif
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        IndexSpace r = { index };
        RegionInstance::Impl *impl = Runtime::get_runtime()->get_free_instance(r,m,list_size,op->sizeof_rhs, ptr, op, parent_impl);
        RegionInstance inst = impl->get_instance();
        instances.insert(inst);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        return inst;
    }
#endif

    void IndexSpace::Impl::destroy_instance(RegionInstance inst)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	std::set<RegionInstance>::iterator it = instances.find(inst);
#ifdef DEBUG_LOW_LEVEL
	assert(it != instances.end());
#endif	
	instances.erase(it);
	RegionInstance::Impl *impl = Runtime::get_runtime()->get_instance_impl(inst);
	impl->deactivate();
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    Lock IndexSpace::Impl::get_lock(void)
    {
	return lock->get_lock();
    }

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
	  if((offset + size) <= *it) {
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
	      RegionInstance::Impl *inst = Runtime::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start, field_size, within_field;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
					  field_start, field_size, within_field);
		// printf("RD(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(index, field_start, within_field));
		assert(bytes > 0);
		memcpy(buffer + write_offset, 
		       inst->get_address(index, field_start, field_size, within_field),
		       bytes);
		offset += bytes;
		size -= bytes;
		write_offset += bytes;
	      }
	    }

	    // now scatter to destination
	    int read_offset = 0;
	    for(std::vector<Domain::CopySrcDstField>::const_iterator i = dsts.begin(); i != dsts.end(); i++) {
	      RegionInstance::Impl *inst = Runtime::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start, field_size, within_field;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
					  field_start, field_size, within_field);
		// printf("WR(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(index, field_start, within_field));
		assert(bytes > 0);
		memcpy(inst->get_address(index, field_start, field_size, within_field),
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
	      RegionInstance::Impl *inst = Runtime::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start, field_size, within_field;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
					  field_start, field_size, within_field);
		// printf("RD(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(index, field_start, within_field));
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
	      RegionInstance::Impl *inst = Runtime::get_runtime()->get_instance_impl(i->inst);
	      size_t offset = i->offset;
	      size_t size = i->size;
	      while(size > 0) {
		size_t field_start, field_size, within_field;
		size_t bytes = find_field(inst->get_field_sizes(), offset, size,
					  field_start, field_size, within_field);
		// printf("WR(%d,%d,%d)(%zd,%zd,%zd,%zd,%zd)(%p,%p)\n",
		//        i->inst.id, i->offset, i->size, offset, size, field_start, within_field, bytes,
		//        inst->get_base_ptr(),
		//        inst->get_address(index, field_start, within_field));
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
          RegionInstance::Impl *src_inst = Runtime::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstance::Impl *dst_inst = Runtime::get_runtime()->get_instance_impl(dsts[0].inst);
          // This should be from one reduction fold instance to another
          for (int index = start; index < (start+count); index++)
          {
            // Assume that there is only one field and they are contiguous
            void *src_ptr = src_inst->get_address(index, 0, redop->sizeof_rhs, 0);
            void *dst_ptr = dst_inst->get_address(index, 0, redop->sizeof_rhs, 0);
            redop->fold(dst_ptr, src_ptr, 1, false/*exclusive*/);
          }
        }
        void do_domain(const Domain domain)
        {
          RegionInstance::Impl *src_inst = Runtime::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstance::Impl *dst_inst = Runtime::get_runtime()->get_instance_impl(dsts[0].inst);
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
          RegionInstance::Impl *src_inst = Runtime::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstance::Impl *dst_inst = Runtime::get_runtime()->get_instance_impl(dsts[0].inst);
          size_t offset = dsts[0].offset;
          size_t size = dsts[0].size;
          size_t field_start, field_size, within_field;
          size_t bytes = find_field(dst_inst->get_field_sizes(), offset, size,
                                    field_start, field_size, within_field);
          for (int index = start; index < (start+count); index++)
          {
            void *src_ptr = src_inst->get_address(index, 0, redop->sizeof_rhs, 0);  
            void *dst_ptr = dst_inst->get_address(index, field_start, field_size, within_field);
            redop->apply(dst_ptr, src_ptr, 1, false/*exclusive*/);
          }
        }
        void do_domain(const Domain domain)
        {
          RegionInstance::Impl *src_inst = Runtime::get_runtime()->get_instance_impl(srcs[0].inst);
          RegionInstance::Impl *dst_inst = Runtime::get_runtime()->get_instance_impl(dsts[0].inst);
          size_t offset = dsts[0].offset;
          size_t size = dsts[0].size;
          size_t field_start, field_size, within_field;
          size_t bytes = find_field(dst_inst->get_field_sizes(), offset, size,
                                    field_start, field_size, within_field);
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

    void IndexSpace::Impl::CopyOperation::perform_copy_operation(void)
    {
      DetailedTimer::ScopedPush sp(TIME_COPY); 
#ifdef LEGION_LOGGING
      LegionRuntime::HighLevel::LegionLogging::log_timing_event(
                                    Machine::get_executing_processor(),
                                    done_event->get_event(), COPY_BEGIN);
#endif

      if (redop_id == 0)
      {
        RangeExecutors::GatherScatter rexec(srcs, dsts);

        if(domain.get_dim() == 0) {
          // This is an index space copy
          IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(domain.get_index_space());
          const ElementMask& mask = r->get_element_mask();
          ElementMask::forall_ranges(rexec, mask, mask);
        } else {
          rexec.do_domain(domain);
        }
      }
      else // This is a reduction operation
      {
        // Get the reduction operation that we are doing
        const ReductionOpUntyped *redop = Runtime::get_runtime()->get_reduction_op(redop_id);
        // See if we're doing a fold or not 
        if (red_fold)
        {
          RangeExecutors::ReductionFold rexec(srcs,dsts,redop);
          if (domain.get_dim() == 0) {
            IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(domain.get_index_space());
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
            IndexSpace::Impl *r = Runtime::get_runtime()->get_metadata_impl(domain.get_index_space());
            const ElementMask& mask = r->get_element_mask();
            ElementMask::forall_ranges(rexec, mask, mask);
          } else {
            rexec.do_domain(domain);
          }
        }
      }
#ifdef LEGION_LOGGING
      LegionRuntime::HighLevel::LegionLogging::log_timing_event(
                                      Machine::get_executing_processor(),
                                      done_event->get_event(), COPY_END);
#endif
    }

    Event IndexSpace::Impl::copy(RegionInstance src_inst, RegionInstance dst_inst, size_t elem_size,
				 const Domain domain, Event wait_on /*= Event::NO_EVENT*/,
				 ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/)
    {
      std::vector<Domain::CopySrcDstField> srcs, dsts;

      srcs.push_back(Domain::CopySrcDstField(src_inst, 0, elem_size));
      dsts.push_back(Domain::CopySrcDstField(dst_inst, 0, elem_size));

      return copy(srcs, dsts, domain, wait_on, redop_id, red_fold);
    }
    
    Event IndexSpace::Impl::copy(const std::vector<Domain::CopySrcDstField>& srcs,
				 const std::vector<Domain::CopySrcDstField>& dsts,
				 Domain domain, Event wait_on,
				 ReductionOpID redop_id /*= 0*/, bool red_fold /*= false*/)
    {
      EventImpl *done_event = NULL;
#ifdef LEGION_LOGGING
      done_event = Runtime::get_runtime()->get_free_event(); 
      LegionRuntime::HighLevel::LegionLogging::log_timing_event(
                                      Machine::get_executing_processor(),
                                      done_event->get_event(), COPY_INIT);
#endif
      CopyOperation *co = new CopyOperation(srcs, dsts, 
					    domain, //get_element_mask(), get_element_mask(),
					    redop_id, red_fold,
					    done_event);
      return co->register_copy(wait_on);
    }

#if 0
    Event IndexSpace::Impl::copy(const std::vector<Domain::CopySrcDstField>& srcs,
				 const std::vector<Domain::CopySrcDstField>& dsts,
				 const ElementMask& mask, Event wait_on,
				 ReductionOpID redop_id, bool red_fold)
    {
      CopyOperation *co = new CopyOperation(srcs, dsts, 
					    get_element_mask(), mask,
					    redop_id, red_fold,
					    0);
      return co->register_copy(wait_on);
    }
#endif

    ////////////////////////////////////////////////////////
    // Machine 
    ////////////////////////////////////////////////////////

    Machine::Machine(int *argc, char ***argv,
			const Processor::TaskIDTable &task_table,
                        const ReductionOpTable &redop_table,
			bool cps_style, Processor::TaskFuncID init_id)
      : background_pthread(NULL)
    {
	// Default nobody can use task id 0 since that is the shutdown id
	if (task_table.find(0) != task_table.end())
	{
		fprintf(stderr,"Using task_id 0 in the task table is illegal!  Task_id 0 is the shutdown task\n");
		fflush(stderr);
		exit(1);
	}

        unsigned num_cpus = NUM_PROCS;
        unsigned num_utility_cpus = NUM_UTIL_PROCS;
        size_t cpu_mem_size_in_mb = GLOBAL_MEM;
        size_t cpu_l1_size_in_kb = LOCAL_MEM;
        size_t cpu_stack_size = STACK_SIZE;

#ifdef DEBUG_PRINT
	PTHREAD_SAFE_CALL(pthread_mutex_init(&debug_mutex,NULL));
#endif
        // Create the pthread keys for thread local data
        PTHREAD_SAFE_CALL( pthread_key_create(&local_proc_key, thread_proc_free) );
        PTHREAD_SAFE_CALL( pthread_key_create(&thread_timer_key, thread_timer_free) );

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
          INT_ARG("-ll:util",num_utility_cpus);
          INT_ARG("-ll:stack",cpu_stack_size);
#undef INT_ARG
        }
        cpu_stack_size = cpu_stack_size * (1 << 20);

        if (num_utility_cpus > num_cpus)
        {
            fprintf(stderr,"The number of processor groups (%d) cannot be greater than the number of cpus (%d)\n",num_utility_cpus,num_cpus);
            fflush(stderr);
            exit(1);
        }

	// Create the runtime and initialize with this machine
	Runtime::runtime = new Runtime(this, redop_table);

        // Initialize the logger
        Logger::init(*argc, (const char**)*argv);
	
        // Fill in the tables
        // find in proc 0 with NULL
        Runtime::runtime->processors.push_back(NULL);
#ifndef __MACH__
        pthread_barrier_t *init_barrier = (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
        PTHREAD_SAFE_CALL(pthread_barrier_init(init_barrier,NULL,(num_cpus+num_utility_cpus)));
#else
        pthread_barrier_t *init_barrier = new pthread_barrier_t(num_cpus+num_utility_cpus);
#endif
        explicit_utility_procs = (num_utility_cpus > 0);
        if (num_utility_cpus > 0)
        {
          // This is the case where we have actual utility processors  

          // Keep track of the number of users of each utility cpu
          std::vector<ProcessorImpl*> temp_utils(num_utility_cpus);
          for (unsigned idx = 0; idx < num_utility_cpus; idx++)
          {
            Processor p;
            p.id = num_cpus+idx+1;
            // Figure out how many users this guy will have
            unsigned num_users = (num_cpus/num_utility_cpus) + (idx < (num_cpus%num_utility_cpus) ? 1 : 0);
            temp_utils[idx] = new ProcessorImpl(init_barrier, task_table, p, cpu_stack_size, num_users);
          }
          // Now we can make the processors themselves
          for (unsigned idx = 0; idx < num_cpus; idx++)
          {
            Processor p;
            p.id = idx + 1;
            procs.insert(p);
            // Figure out which utility processor this guy gets
            unsigned utility_idx = idx % num_utility_cpus;
#ifdef DEBUG_LOW_LEVEL
            assert(utility_idx < num_utility_cpus);
#endif
            ProcessorImpl *impl = new ProcessorImpl(init_barrier, task_table, p, cpu_stack_size, temp_utils[utility_idx], (idx==0));
            // Add this processor as utility user
            temp_utils[utility_idx]->add_utility_user(p, impl);
            Runtime::runtime->processors.push_back(impl);
          }
          // Finally we can add the utility processors to the set of processors
          for (unsigned idx = 0; idx < num_utility_cpus; idx++)
          {
            Runtime::runtime->processors.push_back(temp_utils[idx]);
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
            ProcessorImpl *impl = new ProcessorImpl(init_barrier, task_table, p, cpu_stack_size);
            Runtime::runtime->processors.push_back(impl);
          }
        }
	
#if 0
	for (unsigned id=1; id<=num_cpus; id++)
	{
		Processor p;
		p.id = id;
		procs.insert(p);
                // Compute its utility processor (if any)
		ProcessorImpl *impl;
                if (num_utility_cpus > 0)
                {
                  unsigned util = id % num_utility_cpus;
                  Processor utility;
                  utility.id = num_cpus + 1 + util;
                  //fprintf(stdout,"Processor %d has utility processor %d\n",id,utility.id);
                  //fflush(stdout);
                  impl = new ProcessorImpl(init_barrier,task_table, p, utility, cpu_stack_size);
                  utility_users[util]++;
                }
                else
                {
                  impl = new ProcessorImpl(init_barrier,task_table, p, cpu_stack_size);
                }
		Runtime::runtime->processors.push_back(impl);
	}	
        // Also create the utility processors
        for (unsigned id=1; id<=num_utility_cpus; id++)
        {
                Processor p;
                p.id = num_cpus + id;
#ifdef DEBUG_LOW_LEVEL
                assert(utility_users[id-1] > 0);
#endif
                //fprintf(stdout,"Utility processor %d has %d users\n",p.id,utility_users[id-1]);
                //fflush(stdout);
                // This processor is a utility processor so it is be default its own utility
                ProcessorImpl *impl = new ProcessorImpl(init_barrier,task_table, p, cpu_stack_size, true/*utility*/, utility_users[id-1]);
                Runtime::runtime->processors.push_back(impl);
        }
#endif
	{
                // Make the first memory null
                Runtime::runtime->memories.push_back(NULL);
                // Do the global memory
		Memory global;
		global.id = 1;
		memories.insert(global);
		MemoryImpl *impl = new MemoryImpl(cpu_mem_size_in_mb*1024*1024, Memory::SYSTEM_MEM);
		Runtime::runtime->memories.push_back(impl);
	}
	for (unsigned id=2; id<=(num_cpus+1); id++)
	{
		Memory m;
		m.id = id;
		memories.insert(m);
		MemoryImpl *impl = new MemoryImpl(cpu_l1_size_in_kb*1024, Memory::LEVEL1_CACHE);
		Runtime::runtime->memories.push_back(impl);
	}
	// All memories are visible from each processor
	for (unsigned id=1; id<=num_cpus; id++)
	{
		Processor p;
		p.id = id;
		visible_memories_from_procs.insert(std::pair<Processor,std::set<Memory> >(p,memories));
	}	
	// All memories are visible from all memories, all processors are visible from all memories
	for (unsigned id=1; id<=(num_cpus+1); id++)
	{
		Memory m;
		m.id = id;
		visible_memories_from_memory.insert(std::pair<Memory,std::set<Memory> >(m,memories));
		visible_procs_from_memory.insert(std::pair<Memory,std::set<Processor> >(m,procs));
	}

        // Now set up the affinities for each of the different processors and memories
        for (std::set<Processor>::iterator it = procs.begin(); it != procs.end(); it++)
        {
          // Give all processors 32 GB/s to the global memory
          {
            ProcessorMemoryAffinity global_affin = { *it, {1}, 32, 50/* higher latency */ };
            proc_mem_affinities.push_back(global_affin);
          }
          // Give the processor good affinity to its L1, but not to other L1
          for (unsigned id = 2; id <= (num_cpus+1); id++)
          {
            if (id == (it->id+1))
            {
              // Our L1, high bandwidth with low latency
              ProcessorMemoryAffinity local_affin = { *it, {id}, 100, 1/* small latency */};
              proc_mem_affinities.push_back(local_affin);
            }
            else
            {
              // Other L1, low bandwidth with long latency
              ProcessorMemoryAffinity other_affin = { *it, {id}, 10, 100 /*high latency*/ };
              proc_mem_affinities.push_back(other_affin);
            }
          }
        }
        // Set up the affinities between the different memories
        {
          // Global to all others
          for (unsigned id = 2; id <= (num_cpus+1); id++)
          {
            MemoryMemoryAffinity global_affin = { {1}, {id}, 32, 50 };
            mem_mem_affinities.push_back(global_affin);
          }

          // From any one to any other one
          for (unsigned id = 2; id <= (num_cpus+1); id++)
          {
            for (unsigned other=id+1; other <= (num_cpus+1); other++)
            {
              MemoryMemoryAffinity pair_affin = { {id}, {other}, 10, 100 };
              mem_mem_affinities.push_back(pair_affin);
            }
          }
        }
	// Now start the threads for each of the processors
	// except for processor 0 which is this thread
#ifdef DEBUG_LOW_LEVEL
        assert(Runtime::runtime->processors.size() == (num_cpus+num_utility_cpus+1));
#endif
		
	// If we're doing CPS style set up the inital task and run the scheduler
	if (cps_style)
	{
		Processor p;
		p.id = 1;
		p.spawn(init_id,**argv,*argc);
		// Now run the scheduler, we'll never return from this
		ProcessorImpl *impl = Runtime::runtime->processors[1];
		impl->start((void*)impl);
	}
	// Finally do the initialization for thread 0
        unsigned *local_proc_id = (unsigned*)malloc(sizeof(unsigned));
	*local_proc_id = 1;
        PTHREAD_SAFE_CALL( pthread_setspecific(local_proc_key, local_proc_id) );
        PTHREAD_SAFE_CALL( pthread_setspecific(thread_timer_key, NULL) );
    }

    Machine::~Machine()
    {
    }

    struct MachineRunArgs {
      Machine *m;
      Processor::TaskFuncID task_id;
      Machine::RunStyle style;
      const void *args;
      size_t arglen;
    };  

    static void *background_run_thread(void *data)
    {
      MachineRunArgs *args = (MachineRunArgs *)data;
      args->m->run(args->task_id, args->style, args->args, args->arglen,
		   false /* foreground from this thread's perspective */);
      delete args;
      return 0;
    }

    Logger::Category log_machine("machine");

    void Machine::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/,
                      bool background /*= false*/)
    { 

      if(background) {
        log_machine.info("background operation requested\n");
	fflush(stdout);
	MachineRunArgs *margs = new MachineRunArgs;
	margs->m = this;
	margs->task_id = task_id;
	margs->style = style;
	margs->args = args;
	margs->arglen = arglen;
	
	pthread_t *threadp = (pthread_t*)malloc(sizeof(pthread_t));
	pthread_attr_t attr;
	PTHREAD_SAFE_CALL( pthread_attr_init(&attr) );
	PTHREAD_SAFE_CALL( pthread_create(threadp, &attr, &background_run_thread, (void *)margs) );
	PTHREAD_SAFE_CALL( pthread_attr_destroy(&attr) );
        // Save this pointer in the background thread
        background_pthread = threadp;
	return;
      }

      if(task_id != 0) { // no need to check ONE_TASK_ONLY here, since 1 node
	for(int id = 1; id <= NUM_PROCS; id++) {
	  Processor p = { static_cast<id_t>(id) };
	  p.spawn(task_id,args,arglen);
	  if(style != ONE_TASK_PER_PROC) break;
	}
      }
      // Start the threads for each of the processors (including the utility processors)
      std::vector<pthread_t> other_threads(Runtime::runtime->processors.size());
      for (unsigned id=2; id<Runtime::runtime->processors.size(); id++)
      {
              ProcessorImpl *impl = Runtime::runtime->processors[id];
              PTHREAD_SAFE_CALL(pthread_create(&(other_threads[id]), &(impl->attr), ProcessorImpl::start, (void*)impl));
      }

      // Now run the scheduler
      ProcessorImpl *impl = Runtime::runtime->processors[1];
      ProcessorImpl::start((void*)impl);
      // When we return join on all the other threads and then exit
      for (unsigned id=2; id<other_threads.size(); id++)
      {
          void *result;
          PTHREAD_SAFE_CALL(pthread_join(other_threads[id],&result));
      }
#ifdef ORDERED_LOGGING 
      Logger::finalize();
#endif
      // Once we're done with this, then we can exit
      pthread_exit(NULL);
    }

    void Machine::shutdown(bool local_request /*= true*/)
    {
      for (std::set<Processor>::const_iterator it = procs.begin();
	   it != procs.end(); it++)
      {
	// Kill pill
	it->spawn(0, NULL, 0);
      }
    }

    void Machine::wait_for_shutdown(void)
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

    Processor::Kind Machine::get_processor_kind(Processor p) const
    {
        return Runtime::get_runtime()->get_processor_impl(p)->get_proc_kind();        
    }

    Memory::Kind Machine::get_memory_kind(Memory m) const
    {
        return Runtime::get_runtime()->get_memory_impl(m)->get_kind();
    }

    size_t Machine::get_memory_size(const Memory m) const
    {
        return Runtime::runtime->get_memory_impl(m)->total_space();
    }

    /*static*/ Machine* Machine::get_machine(void)
    {
	return Runtime::get_runtime()->machine;
    }

    /*static*/ Processor Machine::get_executing_processor(void)
    {
      unsigned *local_proc_id = (unsigned*)pthread_getspecific(local_proc_key);
      Processor local = { *local_proc_id };
      return local;
    }
    
    int Machine::get_proc_mem_affinity(std::vector<ProcessorMemoryAffinity> &result,
                                        Processor restrict_proc /*= Processor::NO_PROC*/,
                                        Memory restrict_memory /*= Memory::NO_MEMORY*/)
    {
      int count = 0;

      for (std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin(); it != proc_mem_affinities.end(); it++)
      {
        if (restrict_proc.exists() && ((*it).p != restrict_proc)) continue;
        if (restrict_memory.exists() && ((*it).m != restrict_memory)) continue;
        result.push_back(*it);
        count++;
      }

      return count;
    }

    int Machine::get_mem_mem_affinity(std::vector<MemoryMemoryAffinity> &result,
                                      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
                                      Memory restrict_mem2 /*= Memory::NO_MEMORY*/)
    {
      // Handle the case for same memories
      if (restrict_mem1.exists() && (restrict_mem1 == restrict_mem2))
      {
        MemoryMemoryAffinity affinity;
        affinity.m1 = restrict_mem1;
        affinity.m2 = restrict_mem1;
        affinity.bandwidth = 100;
        affinity.latency = 1;
        result.push_back(affinity);
        return 1;
      }

      int count = 0;
      for (std::vector<Machine::MemoryMemoryAffinity>::const_iterator it =
            mem_mem_affinities.begin(); it != mem_mem_affinities.end(); it++)
      {
        if (restrict_mem1.exists() &&
            ((*it).m1 != restrict_mem1)) continue;
        if (restrict_mem2.exists() &&
            ((*it).m2 != restrict_mem2)) continue;
        result.push_back(*it);
        count++;
      }

      return count;
    }

    void Machine::parse_node_announce_data(const void *args, size_t arglen,
                                           const NodeAnnounceData &annc_data,
                                           bool remote)
    {
      // Should never be called in this version of the low level runtime
      assert(false);
    }

    ////////////////////////////////////////////////////////
    // Runtime 
    ////////////////////////////////////////////////////////

    Runtime::Runtime(Machine *m, const ReductionOpTable &table)
	: redop_table(table), machine(m)
    {
	for (unsigned i=0; i<BASE_EVENTS; i++)
        {
            EventImpl *event = new EventImpl(i);
            events.push_back(event);
            if (i != 0) // Don't hand out the NO_EVENT event
              free_events.push_back(event);
        }

	for (unsigned i=0; i<BASE_LOCKS; i++)
        {
		locks.push_back(new LockImpl(i));
                if (i != 0)
                  free_locks.push_back(locks.back());
        }

	for (unsigned i=0; i<BASE_METAS; i++)
	{
		metadatas.push_back(new IndexSpace::Impl(i,0,0));
                if (i != 0)
                  free_metas.push_back(metadatas.back());
	}

	for (unsigned i=0; i<BASE_INSTANCES; i++)
	{
		Memory m;
		m.id = 0;
		instances.push_back(new RegionInstance::Impl(i,
							     IndexSpace::NO_SPACE,
							     m,
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
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&lock_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_lock_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&metadata_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_metas_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&allocator_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_alloc_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&instance_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_inst_lock,NULL));
    }

    EventImpl* Runtime::get_event_impl(Event e)
    {
        EventImpl::EventIndex i = e.id;
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&event_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(i != 0);
	assert(i < events.size());
#endif
        EventImpl *result = events[i];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
	return result;
    }

    void Runtime::free_event(EventImpl *e)
    {
      // Put this event back on the list of free events
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_event_lock));
      free_events.push_back(e);
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_event_lock));
    }

    LockImpl* Runtime::get_lock_impl(Lock l)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&lock_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(l.id != 0);
	assert(l.id < locks.size());
#endif
        LockImpl *result = locks[l.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&lock_lock));
	return result;
    }

    MemoryImpl* Runtime::get_memory_impl(Memory m)
    {
	if (m.id < memories.size())
		return memories[m.id];
	else
        {
                assert(false);
		return NULL;
        }
    }

    ProcessorImpl* Runtime::get_processor_impl(Processor p)
    {
#ifdef DEBUG_LOW_LEVEL
        assert(p.exists());
	assert(p.id < processors.size());
#endif
	return processors[p.id];
    }

    IndexSpace::Impl* Runtime::get_metadata_impl(IndexSpace m)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&metadata_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(m.id != 0);
	assert(m.id < metadatas.size());
#endif
        IndexSpace::Impl *result = metadatas[m.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
	return result;
    }

    RegionInstance::Impl* Runtime::get_instance_impl(RegionInstance i)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&instance_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(i.id != 0);
	assert(i.id < instances.size());
#endif
        RegionInstance::Impl *result = instances[i.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
	return result;
    }

    EventImpl* Runtime::get_free_event()
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
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
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

    LockImpl* Runtime::get_free_lock(size_t data_size/*= 0*/)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_lock_lock));
        if (!free_locks.empty())
        {
          LockImpl *result = free_locks.front();
          free_locks.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_lock_lock));
          bool activated = result->activate(data_size);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
        // We weren't able to get a new event, get the writer lock
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&lock_lock));
	unsigned index = locks.size();
	locks.push_back(new LockImpl(index,true,data_size));
	LockImpl *result = locks[index];
        // Create a whole bunch of other locks too while we're here
        for (unsigned idx=1; idx < BASE_LOCKS; idx++)
        {
          locks.push_back(new LockImpl(index+idx,false));
          free_locks.push_back(locks.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&lock_lock));	
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_lock_lock));
	return result;
    }

    IndexSpace::Impl* Runtime::get_free_metadata(size_t num_elmts)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          IndexSpace::Impl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(num_elmts);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new IndexSpace::Impl(index,num_elmts,true));
	IndexSpace::Impl *result = metadatas[index];
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new IndexSpace::Impl(index+idx,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
	return result;
    }

    IndexSpace::Impl* Runtime::get_free_metadata(IndexSpace::Impl *parent, const ElementMask &mask)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          IndexSpace::Impl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(parent,mask);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new IndexSpace::Impl(index,parent,mask,true));
	IndexSpace::Impl *result = metadatas[index];
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new IndexSpace::Impl(index+idx,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
	return result;
    }


    RegionInstance::Impl* Runtime::get_free_instance(IndexSpace r, Memory m, size_t num_elmts, 
						     const std::vector<size_t>& field_sizes,
						     size_t elmt_size, size_t block_size,
						     const DomainLinearization& linearization,
						     char *ptr, const ReductionOpUntyped *redop,
						     RegionInstance::Impl *parent)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_inst_lock));
        if (!free_instances.empty())
        {
          RegionInstance::Impl *result = free_instances.front();
          free_instances.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_inst_lock));
          bool activated = result->activate(r, m, num_elmts, field_sizes, elmt_size, block_size, linearization,
					    ptr, redop, parent);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
	// Nothing free so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&instance_lock));
	unsigned int index = instances.size();
	instances.push_back(new RegionInstance::Impl(index, r, m, num_elmts, field_sizes,
						     elmt_size, block_size, linearization,
						     true, ptr, redop, parent));
	RegionInstance::Impl *result = instances[index];
        // Create a whole bunch of other instances while we're here
        for (unsigned idx=1; idx < BASE_INSTANCES; idx++)
        {
          instances.push_back(new RegionInstance::Impl(index+idx,
						       IndexSpace::NO_SPACE,
						       m,
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

    const ReductionOpUntyped* Runtime::get_reduction_op(ReductionOpID redop)
    {
#ifdef DEBUG_LOW_LEVEL
      assert(redop_table.find(redop) != redop_table.end());
#endif
      return redop_table[redop];
    }

  };

#ifdef ORDERED_LOGGING 
  /*static*/ void Logger::finalize(void)
  {
    // Flush the buffer
    assert(write(get_log_file(), get_logging_buffer(), *get_logging_location()) >= 0);
    // Close the file
    close(get_log_file());
    // Free the memory
    free(get_logging_buffer());
  }
#endif

  // Machine specific implementation of logvprintf
  /*static*/ void Logger::logvprintf(LogLevel level, int category, const char *fmt, va_list args)
  {
    char buffer[400];
    unsigned *local_proc_id = (unsigned*)pthread_getspecific(local_proc_key);
    sprintf(buffer, "[%d - %lx] {%s}{%s}: ",
            0, /*pthread_self()*/long(*local_proc_id), Logger::stringify(level), Logger::get_categories_by_id()[category].c_str());
    int len = strlen(buffer);
    vsnprintf(buffer+len, 399-len, fmt, args);
    strcat(buffer, "\n");
#ifdef ORDERED_LOGGING 
    // Update the length to reflect the newline character
    len = strlen(buffer);
    long long loc = __sync_fetch_and_add(get_logging_location(),len);
    // Check to see if we can actually fit
    // If we can't keep looping until we're allowed to write
    int possible_end = (loc <= logging_buffer_size) ? int(loc) : -1;
    // There is an implicit assumption in this loop that the spinning
    // threads can't count to 2^64 before the buffer is written out
    // to the file.  Pretty sure this is a safe assumption...
    while ((loc+len) >= logging_buffer_size)
    {
      // Wait until all the writers have finished
      if (possible_end == *((volatile int*)get_written_location()))
      {
        // We're the first ones to update and not fit, so do
        // the write back of the buffer, and then mark that it
        // is ready to use
        assert(write(get_log_file(), get_logging_buffer(), possible_end) >= 0);
        // Reset the end written location first
        *((volatile int*)get_written_location()) = 0;
        // Then use compare and swap to reset the logging location
        *((volatile long long*)get_logging_location()) = 0;
      }
      // Now get a new location and see if it works
      long long new_loc = __sync_fetch_and_add(get_logging_location(),len);
      // If new_loc is less than the old_loc need to reset the possible end
      // since the buffer was reset
      if (new_loc < loc)
        possible_end = (loc <= logging_buffer_size) ? int(loc) : -1;
      loc = new_loc;
    }
    // Once we're here, we can just do our write into the buffer and then
    // mark that we did our write
    memcpy(get_logging_buffer()+loc,buffer,len);
    __sync_fetch_and_add(get_written_location(),len);
#else
    fflush(stdout);
    fputs(buffer, stderr);
#endif
  }
};

namespace LegionRuntime {
  namespace Accessor {
    using namespace LegionRuntime::LowLevel;

    void AccessorType::Generic::Untyped::read_untyped(ptr_t ptr, void *dst, size_t bytes, off_t offset) const
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      size_t field_start, field_size, within_field;
      size_t bytes2 = find_field(impl->get_field_sizes(), field_offset + offset, bytes,
				 field_start, field_size, within_field);
      assert(bytes == bytes2);
      const char *src = (const char *)(impl->get_address(ptr.value, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    void AccessorType::Generic::Untyped::read_untyped(const DomainPoint& dp, void *dst, size_t bytes, off_t offset) const
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      int index = impl->get_linearization().get_image(dp);
      size_t field_start, field_size, within_field;
      size_t bytes2 = find_field(impl->get_field_sizes(), field_offset + offset, bytes,
				 field_start, field_size, within_field);
      assert(bytes == bytes2);
      const char *src = (const char *)(impl->get_address(index, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    void AccessorType::Generic::Untyped::write_untyped(ptr_t ptr, const void *src, size_t bytes, off_t offset) const
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      size_t field_start, field_size, within_field;
      size_t bytes2 = find_field(impl->get_field_sizes(), field_offset + offset, bytes,
				 field_start, field_size, within_field);
      assert(bytes == bytes2);
      char *dst = (char *)(impl->get_address(ptr.value, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    void AccessorType::Generic::Untyped::write_untyped(const DomainPoint& dp, const void *src, size_t bytes, off_t offset) const
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      int index = impl->get_linearization().get_image(dp);
      size_t field_start, field_size, within_field;
      size_t bytes2 = find_field(impl->get_field_sizes(), field_offset + offset, bytes,
				 field_start, field_size, within_field);
      assert(bytes == bytes2);
      char *dst = (char *)(impl->get_address(index, field_start, field_size, within_field));
      memcpy(dst, src, bytes);
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset *offsets)
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      Arrays::Mapping<DIM, 1> *mapping = impl->get_linearization().get_mapping<DIM>();
      Point<1> strides[DIM];
      int index = mapping->image_linear_subrect(r, subrect, strides);
      // TODO: trim subrect in HybridSOA case
      size_t field_start, field_size, within_field;
      size_t bytes2 = find_field(impl->get_field_sizes(), field_offset, 1,
				 field_start, field_size, within_field);
      assert(bytes2 == 1);
      char *dst = (char *)(impl->get_address(index, field_start, field_size, within_field));
      for(int i = 0; i < DIM; i++)
	offsets[i].offset = (strides[i] * 
			     ((impl->get_block_size() > 1) ? field_size : impl->get_elmt_size()));
      return dst;
    }

    template void *AccessorType::Generic::Untyped::raw_rect_ptr<3>(const Rect<3>& r, Rect<3>& subrect, ByteOffset *offset);

    //static const void *(AccessorType::Generic::Untyped::*dummy_ptr)(const Rect<3>&, Rect<3>&, ByteOffset*) = AccessorType::Generic::Untyped::raw_rect_ptr<3>;

    bool AccessorType::Generic::Untyped::get_aos_parameters(void *& base, size_t& stride) const
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;

      if(base != 0) return false;
      base = ((char *)(impl->get_base_ptr())) + field_offset;

      size_t elem_size = impl->get_elmt_size();
      if((stride != 0) && (stride != elem_size)) return false;
      stride = elem_size;

      return true;
    }

    bool AccessorType::Generic::Untyped::get_soa_parameters(void *& base, size_t& stride) const
    {
      // TODO: implement this
      return false;
    }

    bool AccessorType::Generic::Untyped::get_hybrid_soa_parameters(void *& base, size_t& stride,
								   size_t& block_size, size_t& block_stride) const
    {
      // TODO: implement this
      return false;
    }

    bool AccessorType::Generic::Untyped::get_redfold_parameters(void *& base) const
    {
      // TODO: actually check that we're a reduction?
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;

      if(base != 0) return false;
      base = impl->get_base_ptr();

      return true;
    }

    bool AccessorType::Generic::Untyped::get_redlist_parameters(void *& base, ptr_t *& next_ptr) const
    {
      // TODO: implement this
      return false;
    }

#ifdef POINTER_CHECKS
    void AccessorType::verify_access(void *impl_ptr, unsigned ptr) 
    {
      assert(impl_ptr != NULL);
      RegionInstance::Impl *impl = (RegionInstance::Impl *) impl_ptr;
      impl->verify_access(ptr);
    }

    void AccessorType::verify_access(void *impl_ptr, const LowLevel::DomainPoint& dp)
    {
        assert(impl_ptr != NULL);
        RegionInstance::Impl *impl = (RegionInstance::Impl *) impl_ptr;
	int index = impl->get_linearization().get_image(dp);
	impl->verify_access(index);
    }
#endif
  };

  namespace Arrays {
    //template<> class Mapping<1,1>;
    template <unsigned IDIM, unsigned ODIM>
    MappingRegistry<IDIM, ODIM> Mapping<IDIM, ODIM>::registry;
  };
};
