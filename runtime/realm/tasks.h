/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// tasks and task scheduling for Realm

#ifndef REALM_TASKS_H
#define REALM_TASKS_H

#include "realm/processor.h"
#include "realm/id.h"

#include "realm/operation.h"
#include "realm/profiling.h"

#include "realm/threads.h"
#include "realm/pri_queue.h"
#include "realm/bytearray.h"
#include "realm/atomics.h"
#include "realm/mutex.h"
#include "realm/bgwork.h"

namespace Realm {

    class ProcessorImpl;
  
    // information for a task launch
    class Task : public Operation {
    public:
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
           const ProfilingRequestSet &reqs,
	   Event _before_event,
	   GenEventImpl *_finish_event, EventImpl::gen_t _finish_gen,
	   int _priority);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~Task(void);

    public:
      virtual bool mark_ready(void);
      virtual bool mark_started(void);

      virtual void print(std::ostream& os) const;

      virtual bool attempt_cancellation(int error_code, const void *reason_data, size_t reason_size);

      virtual void set_priority(int new_priority);
      
      void execute_on_processor(Processor p);

      Processor proc;
      Processor::TaskFuncID func_id;

      // "small-vector" optimization for task args
      char *argdata;
      size_t arglen;
      static const size_t SHORT_ARGLEN_MAX = 64;
      char short_argdata[SHORT_ARGLEN_MAX];
      bool free_argdata;
      //ByteArray args;

      Event before_event;
      int priority;

      // intrusive task list - used for pending, ready, and suspended tasks
      IntrusivePriorityListLink<Task> tl_link;
      REALM_PMTA_DEFN(Task,IntrusivePriorityListLink<Task>,tl_link);
      REALM_PMTA_DEFN(Task,int,priority);
      typedef IntrusivePriorityList<Task, int, REALM_PMTA_USE(Task,tl_link), REALM_PMTA_USE(Task,priority), DummyLock> TaskList;

      class DeferredSpawn : public EventWaiter {
      public:
	DeferredSpawn(void);
	void setup(ProcessorImpl *_proc, Task *_task, Event _wait_on);
        void defer(EventImpl *_wait_impl, EventImpl::gen_t _wait_gen);
	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

	// attempts to add another task to the this deferred spawn group -
	// returns true on success, or false if the event has already
	//  triggered, in which case 'poisoned' is set appropriately
	bool add_task(Task *to_add, bool& poisoned);

      protected:
	ProcessorImpl *proc;
	Task *task;
	Event wait_on;
	Mutex pending_list_mutex;
	TaskList pending_list;
	bool is_triggered, is_poisoned;
	size_t list_length;
      };
      DeferredSpawn deferred_spawn;
      
    protected:
      virtual void mark_completed(void);

      virtual Status::Result get_state(void);

      Thread *executing_thread;

      // to spread out the cost of marking a long list of tasks ready, we
      //  keep a 'marked_ready' bit in the head task of the list and the rest
      //  have a pointer to the head task (which uses a uintptr_t so we can
      //  borrow the bottom bit for avoiding races)
      atomic<bool> marked_ready;
    public: // HACK for debug - should be protected
      atomic<uintptr_t> pending_head;
    };

    class TaskQueue {
    public:
      TaskQueue(void);

      // we used most of the signed integer range for priorities - we do borrow a 
      //  few of the extreme values to make sure we have "infinity" and "negative infinity"
      //  and that we don't run into problems with -INT_MIN
      typedef int priority_t;
      static const priority_t PRI_MAX_FINITE = INT_MAX - 1;
      static const priority_t PRI_MIN_FINITE = -(INT_MAX - 1);
      static const priority_t PRI_POS_INF = PRI_MAX_FINITE + 1;
      static const priority_t PRI_NEG_INF = PRI_MIN_FINITE - 1;

      class NotificationCallback {
      public:
	virtual void item_available(priority_t item_priority) = 0;
      };

      // starvation seems to be a problem on shared task queues
      FIFOMutex mutex;
      Task::TaskList ready_task_list;
      std::vector<NotificationCallback *> callbacks;
      std::vector<priority_t> callback_priorities;
      ProfilingGauges::AbsoluteRangeGauge<int> *task_count_gauge;

      void add_subscription(NotificationCallback *callback, priority_t higher_than = PRI_NEG_INF);

      void remove_subscription(NotificationCallback *callback);

      void set_gauge(ProfilingGauges::AbsoluteRangeGauge<int> *new_gauge);

      void free_gauge();
      // gets highest priority task available from any task queue in list
      static Task *get_best_task(const std::vector<TaskQueue *>& queues,
				 int& task_priority);

      void enqueue_task(Task *task);
      void enqueue_tasks(Task::TaskList& tasks, size_t num_tasks);
    };

    // an internal task is an arbitrary blob of work that needs to happen on
    //  a processor's actual thread(s)
    class InternalTask {
    protected:
      // cannot be destroyed directly
      virtual ~InternalTask() {}

    public:
      virtual void execute_on_processor(Processor p) = 0;

      IntrusiveListLink<InternalTask> tl_link;
      REALM_PMTA_DEFN(InternalTask,IntrusiveListLink<InternalTask>,tl_link);
      typedef IntrusiveList<InternalTask, REALM_PMTA_USE(InternalTask,tl_link), Mutex> TaskList;
    };

    // a common extension for processors is to provide some context for
    //  running tasks - this can be done by subclassing and overriding
    //  `execute_task`, but simple cases can be handled with
    //  TaskContextManagers
    class TaskContextManager {
    public:
      // create a context for the specified task - the value returned will
      //  be provided to the call to destroy_context
      virtual void *create_context(Task *task) const = 0;

      virtual void destroy_context(Task *task, void *context) const = 0;
    };

    // a task scheduler in which one or more worker threads execute tasks from one
    //  or more task queues
    // once given a task, a worker must complete it before taking on new work
    // if a worker needs to suspend, a new worker may be spun up to start a new task
    // this parent version tries to be agnostic to whether the threads are
    //  user or kernel threads
    class ThreadedTaskScheduler : public ThreadScheduler {
    public:
      ThreadedTaskScheduler(void);

      virtual ~ThreadedTaskScheduler(void);

      virtual void add_task_queue(TaskQueue *queue);

      virtual void remove_task_queue(TaskQueue *queue);

      virtual void configure_bgworker(BackgroundWorkManager *manager,
				      long long max_timeslice,
				      int numa_domain);

      // add a context manager - each new one "wraps" the previous ones,
      //  constructing its context after them and destroying before
      void add_task_context(const TaskContextManager *_manager);

      virtual void start(void) = 0;
      virtual void shutdown(void) = 0;

      // called when thread status changes
      virtual void thread_blocking(Thread *thread);
      virtual void thread_ready(Thread *thread);

      virtual void set_thread_priority(Thread *thread, int new_priority);

      void add_internal_task(InternalTask *itask);

    public:
      // the main scheduler loop - lock should be held before calling
      void scheduler_loop(void);
      // an entry point that takes the scheduler lock explicitly
      void scheduler_loop_wlock(void);

    protected:
      // returns true if everything went well, false if running thread
      //   may have been left in a bad state
      virtual bool execute_task(Task *task) = 0;

      virtual void execute_internal_task(InternalTask *task) = 0;

      virtual Thread *worker_create(bool make_active) = 0;
      virtual void worker_sleep(Thread *switch_to) = 0;
      virtual void worker_wake(Thread *to_wake) = 0;
      virtual void worker_terminate(Thread *switch_to) = 0;

      // gets highest priority task available from any task queue
      Task *get_best_ready_task(int& task_priority);

      // TODO: switch this to DelegatingMutex - goal is that callers of
      //  things like thread_ready() should not have to block on
      //  contention
      FIFOMutex lock;
      std::vector<TaskQueue *> task_queues;
      std::vector<Thread *> idle_workers;
      std::set<Thread *> blocked_workers;
      // threads that block while holding a scheduler lock go here instead
      std::set<Thread *> spinning_workers;

      std::vector<const TaskContextManager *> context_managers;

      // internal task list is NOT guarded by the main mutex
      InternalTask::TaskList internal_tasks;

      typedef PriorityQueue<Thread *, DummyLock> ResumableQueue;
      ResumableQueue resumable_workers;
      std::map<Thread *, int> worker_priorities;
      atomic<bool> shutdown_flag;
      int active_worker_count;  // workers that are awake (i.e. using a core)
      int unassigned_worker_count;  // awake but unassigned workers

      // helper for tracking/sanity-checking worker counts
      void update_worker_count(int active_delta, int unassigned_delta, bool check = true);

      // workers that are unassigned and cannot find any work would often (but not
      //  always) like to suspend until work is available - this is done via a "work counter"
      //  that monotonically increments whenever any kind of new work is available and a 
      //  "suspended on" value that indicates if any threads are suspended on a particular
      //  count and need to be signalled
      // this model allows the provider of new work to update the counter in a lock-free way
      //  and only do the condition variable broadcast if somebody is probably sleeping

      class WorkCounter {
      public:
	WorkCounter(void);
	~WorkCounter(void);

	void set_interrupt_flag(atomic<bool> *_interrupt_flag);

	// called whenever new work is available
	void increment_counter(void);

	uint64_t read_counter(void) const;

	// returns true if there is new work since the old_counter value was read
	// this is non-blocking, and may be called while holding another lock
	bool check_for_work(uint64_t old_counter);

	// waits until new work arrives - this will possibly go to sleep,
	//  so should not be called while holding another lock
	void wait_for_work(uint64_t old_counter);

      protected:
	// 64-bit counter is used to avoid dealing with wrap-around cases
        // bottom bits count the number of sleepers, but a max of 2^56 operations
        //   is still a lot
        static const unsigned SLEEPER_BITS = 8;
	atomic<uint64_t> counter;
	atomic<bool> *interrupt_flag;

        // doorbell list popping is protected with a lock-free delegating mutex
        DelegatingMutex db_mutex;
        DoorbellList db_list;
      };
	
      WorkCounter work_counter;

      virtual void wait_for_work(uint64_t old_work_counter);

      // most of our work counter updates are going to come from priority queues, so a little
      //  template-fu here...
      template <typename PQ>
      class WorkCounterUpdater : public PQ::NotificationCallback {
      public:
        WorkCounterUpdater(ThreadedTaskScheduler *sched)
	  : work_counter(&sched->work_counter)
	{}

	// TaskQueue-style
	virtual void item_available(typename PQ::priority_t)
	{
	  work_counter->increment_counter();
	}

	// PriorityQueue-style
	virtual bool item_available(Thread *, typename PQ::priority_t) 
	{ 
	  work_counter->increment_counter();
	  return false;  // never consumes the work
	}
      protected:
	WorkCounter *work_counter;
      };

      WorkCounterUpdater<TaskQueue> wcu_task_queues;
      WorkCounterUpdater<ResumableQueue> wcu_resume_queue;

      BackgroundWorkManager::Worker bgworker;
      atomic<bool> bgworker_interrupt;
      long long max_bgwork_timeslice;

    public:
      // various configurable settings
      bool cfg_reuse_workers;
      int cfg_max_idle_workers;
      int cfg_min_active_workers;
      int cfg_max_active_workers;
    };

    inline uint64_t ThreadedTaskScheduler::WorkCounter::read_counter(void) const
    {
      // just return the counter value with the sleeper bits removed
      return (counter.load_acquire() >> SLEEPER_BITS);
    }

    // returns true if there is new work since the old_counter value was read
    // this is non-blocking, and may be called while holding another lock
    inline bool ThreadedTaskScheduler::WorkCounter::check_for_work(uint64_t old_counter)
    {
      // test the counter value without synchronization
      return (read_counter() != old_counter);
    }


    // an implementation of ThreadedTaskScheduler that uses kernel threads
    //  for workers
    class KernelThreadTaskScheduler : public ThreadedTaskScheduler {
    public:
      KernelThreadTaskScheduler(Processor _proc, CoreReservation& _core_rsrv);

      virtual ~KernelThreadTaskScheduler(void);

      virtual void add_task_queue(TaskQueue *queue);

      virtual void remove_task_queue(TaskQueue *queue);

      virtual void start(void);

      virtual void shutdown(void);

      virtual void thread_starting(Thread *thread);

      virtual void thread_terminating(Thread *thread);

    protected:
      virtual bool execute_task(Task *task);

      virtual void execute_internal_task(InternalTask *task);

      virtual Thread *worker_create(bool make_active);
      virtual void worker_sleep(Thread *switch_to);
      virtual void worker_wake(Thread *to_wake);
      virtual void worker_terminate(Thread *switch_to);

      virtual void wait_for_work(uint64_t old_work_counter);

      Processor proc;
      CoreReservation &core_rsrv;

      std::set<Thread *> all_workers;
      std::set<Thread *> active_workers;
      std::set<Thread *> terminating_workers;
      std::map<Thread *, FIFOMutex::CondVar *> sleeping_threads;
      FIFOMutex::CondVar shutdown_condvar;
    };

#ifdef REALM_USE_USER_THREADS
    // an implementation of ThreadedTaskScheduler that uses user threads
    //  for workers (and one or more kernel threads for hosts
    class UserThreadTaskScheduler : public ThreadedTaskScheduler {
    public:
      UserThreadTaskScheduler(Processor _proc, CoreReservation& _core_rsrv);

      virtual ~UserThreadTaskScheduler(void);

      virtual void add_task_queue(TaskQueue *queue);

      virtual void remove_task_queue(TaskQueue *queue);

      virtual void start(void);
      virtual void shutdown(void);

      virtual void thread_starting(Thread *thread);

      virtual void thread_terminating(Thread *thread);

    protected:
      virtual bool execute_task(Task *task);

      virtual void execute_internal_task(InternalTask *task);

      void host_thread_loop(void);
      
      // you can't delete a user thread until you've switched off of it, so
      //  use TLS to mark when that should happen
      void request_user_thread_cleanup(Thread *thread);
      void do_user_thread_cleanup(void);
      
      virtual Thread *worker_create(bool make_active);
      virtual void worker_sleep(Thread *switch_to);
      virtual void worker_wake(Thread *to_wake);
      virtual void worker_terminate(Thread *switch_to);

      virtual void wait_for_work(uint64_t old_work_counter);

      Processor proc;
      CoreReservation &core_rsrv;

      std::set<Thread *> all_hosts;
      std::set<Thread *> all_workers;

      int host_startups_remaining;
      FIFOMutex::CondVar host_startup_condvar;

    public:
      int cfg_num_host_threads;
    };
#endif

}; // namespace Realm

#endif // ifndef REALM_TASKS_H
