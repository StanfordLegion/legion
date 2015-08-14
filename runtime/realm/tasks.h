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

// tasks and task scheduling for Realm

#ifndef REALM_TASKS_H
#define REALM_TASKS_H

#include "processor.h"
#include "id.h"

#include "operation.h"
#include "profiling.h"

#include "threads.h"
#include "pri_queue.h"

namespace Realm {

    // information for a task launch
    class Task : public Operation {
    public:
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
	   Event _finish_event, int _priority,
           int expected_count);
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
           const ProfilingRequestSet &reqs,
	   Event _finish_event, int _priority,
           int expected_count);

      virtual ~Task(void);

      void execute_on_processor(Processor p);

      Processor proc;
      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
      Event finish_event;
      int priority;
      int run_count, finish_count;
      bool capture_proc;
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

      typedef PriorityQueue<Task *, GASNetHSL> TaskQueue;

      virtual void add_task_queue(TaskQueue *queue);

      virtual void start(void) = 0;
      virtual void shutdown(void) = 0;

      // called when thread status changes
      virtual void thread_blocking(Thread *thread);
      virtual void thread_ready(Thread *thread);

    public:
      // the main scheduler loop - lock should be held before calling
      void scheduler_loop(void);
      // an entry point that takes the scheduler lock explicitly
      void scheduler_loop_wlock(void);

    protected:
      // returns true if everything went well, false if running thread
      //   may have been left in a bad state
      virtual bool execute_task(Task *task) = 0;

      virtual Thread *worker_create(bool make_active) = 0;
      virtual void worker_sleep(Thread *switch_to) = 0;
      virtual void worker_wake(Thread *to_wake) = 0;
      virtual void worker_terminate(Thread *switch_to) = 0;

      GASNetHSL lock;
      std::vector<TaskQueue *> task_queues;
      std::vector<Thread *> idle_workers;

      typedef PriorityQueue<Thread *, DummyLock> ResumableQueue;
      ResumableQueue resumable_workers;
      std::map<Thread *, int> worker_priorities;
      bool shutdown_flag;
      int active_worker_count;  // workers that are awake (i.e. using a core)
      int unassigned_worker_count;  // awake but unassigned workers

      // helper for tracking/sanity-checking worker counts
      void update_worker_count(int active_delta, int unassigned_delta, bool check = true);

      // workers that are unassigned and cannot find any work would often (but not
      //  always) like to suspend until work is available - this is done via a "work counter"
      //  that monotonically increments whenever any kind of new work is available and a 
      //  "suspended on" value that indicates if any threads are suspended on a particular
      //  count and need to be signalled
      // this model allows the provided of new work to update the counter in a lock-free way
      //  and only do the condition variable broadcast if somebody is probably sleeping
      //
      // 64-bit counters are used to avoid dealing with wrap-around cases
      volatile long long work_counter, work_counter_wait_value;
      GASNetCondVar work_counter_condvar;

      void increment_work_counter(void);
      virtual void wait_for_work(long long old_work_counter);

      // most of our work counter updates are going to come from priority queues, so a little
      //  template-fu here...
      template <typename PQ>
      class WorkCounterUpdater : public PQ::NotificationCallback {
      public:
        WorkCounterUpdater(ThreadedTaskScheduler *_sched) : sched(_sched) {}
	virtual bool item_available(typename PQ::ITEMTYPE, typename PQ::priority_t) 
	{ 
	  sched->increment_work_counter();
	  return false;  // never consumes the work
	}
      protected:
	ThreadedTaskScheduler *sched;
      };

      WorkCounterUpdater<TaskQueue> wcu_task_queues;
      WorkCounterUpdater<ResumableQueue> wcu_resume_queue;

    public:
      // various configurable settings
      bool cfg_reuse_workers;
      int cfg_max_idle_workers;
      int cfg_min_active_workers;
      int cfg_max_active_workers;
    };

    // an implementation of ThreadedTaskScheduler that uses kernel threads
    //  for workers
    class KernelThreadTaskScheduler : public ThreadedTaskScheduler {
    public:
      KernelThreadTaskScheduler(Processor _proc, CoreReservation& _core_rsrv);

      virtual ~KernelThreadTaskScheduler(void);

      virtual void add_task_queue(TaskQueue *queue);

      virtual void start(void);
      virtual void shutdown(void);

      virtual void thread_starting(Thread *thread);

      virtual void thread_terminating(Thread *thread);

    protected:
      virtual bool execute_task(Task *task);

      virtual Thread *worker_create(bool make_active);
      virtual void worker_sleep(Thread *switch_to);
      virtual void worker_wake(Thread *to_wake);
      virtual void worker_terminate(Thread *switch_to);

      virtual void wait_for_work(long long old_work_counter);

      Processor proc;
      CoreReservation &core_rsrv;

      std::set<Thread *> all_workers;
      std::set<Thread *> active_workers;
      std::map<Thread *, GASNetCondVar *> sleeping_threads;
      GASNetCondVar shutdown_condvar;
    };

#ifdef REALM_USE_USER_THREADS
    // an implementation of ThreadedTaskScheduler that uses user threads
    //  for workers (and one or more kernel threads for hosts
    class UserThreadTaskScheduler : public ThreadedTaskScheduler {
    public:
      UserThreadTaskScheduler(Processor _proc, CoreReservation& _core_rsrv);

      virtual ~UserThreadTaskScheduler(void);

      virtual void add_task_queue(TaskQueue *queue);

      virtual void start(void);
      virtual void shutdown(void);

      virtual void thread_starting(Thread *thread);

      virtual void thread_terminating(Thread *thread);

    protected:
      virtual bool execute_task(Task *task);

      void host_thread_loop(void);
      
      // you can't delete a user thread until you've switched off of it, so
      //  use TLS to mark when that should happen
      void request_user_thread_cleanup(Thread *thread);
      void do_user_thread_cleanup(void);
      
      virtual Thread *worker_create(bool make_active);
      virtual void worker_sleep(Thread *switch_to);
      virtual void worker_wake(Thread *to_wake);
      virtual void worker_terminate(Thread *switch_to);

      virtual void wait_for_work(long long old_work_counter);

      Processor proc;
      CoreReservation &core_rsrv;

      std::set<Thread *> all_hosts;
      std::set<Thread *> all_workers;

    public:
      int cfg_num_host_threads;
    };
#endif

}; // namespace Realm

#endif // ifndef REALM_TASKS_H
