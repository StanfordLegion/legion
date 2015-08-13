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

#include "tasks.h"

#include "runtime_impl.h"

namespace Realm {

  Logger log_task("task");
  Logger log_util("util");

  ////////////////////////////////////////////////////////////////////////
  //
  // class Task
  //

    Task::Task(Processor _proc, Processor::TaskFuncID _func_id,
	       const void *_args, size_t _arglen,
	       Event _finish_event, int _priority, int expected_count)
      : Operation(), proc(_proc), func_id(_func_id), arglen(_arglen),
	finish_event(_finish_event), priority(_priority),
        run_count(0), finish_count(expected_count), capture_proc(false)
    {
      if(arglen) {
	args = malloc(arglen);
	memcpy(args, _args, arglen);
      } else
	args = 0;
    }

    Task::Task(Processor _proc, Processor::TaskFuncID _func_id,
	       const void *_args, size_t _arglen,
               const ProfilingRequestSet &reqs,
	       Event _finish_event, int _priority, int expected_count)
      : Operation(reqs), proc(_proc), func_id(_func_id), arglen(_arglen),
	finish_event(_finish_event), priority(_priority),
        run_count(0), finish_count(expected_count)
    {
      if(arglen) {
	args = malloc(arglen);
	memcpy(args, _args, arglen);
      } else
	args = 0;
      capture_proc = measurements.wants_measurement<
                        ProfilingMeasurements::OperationProcessorUsage>();
    }

    Task::~Task(void)
    {
      free(args);
      if (capture_proc) {
        ProfilingMeasurements::OperationProcessorUsage usage;
        usage.proc = proc;
        measurements.add_measurement(usage);
      }
    }

  void Task::execute_on_processor(Processor p)
  {
    // if the processor isn't specified, use what's in the task object
    if(!p.exists())
      p = this->proc;

    Processor::TaskFuncPtr fptr = get_runtime()->task_table[func_id];
#if 0
    char argstr[100];
    argstr[0] = 0;
    for(size_t i = 0; (i < arglen) && (i < 40); i++)
      sprintf(argstr+2*i, "%02x", ((unsigned char *)(args))[i]);
    if(arglen > 40) strcpy(argstr+80, "...");
    log_util(((func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
	     "task start: %d (%p) (%s)", func_id, fptr, argstr);
#endif
#ifdef EVENT_GRAPH_TRACE
    start_enclosing(finish_event);
    unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif
    log_task.info("thread running ready task %p for proc " IDFMT "",
		  this, p.id);

    // does the profiler want to know where it was run?
    if(measurements.wants_measurement<ProfilingMeasurements::OperationProcessorUsage>()) {
      ProfilingMeasurements::OperationProcessorUsage opu;
      opu.proc = p;
      measurements.add_measurement(opu);
    }

    mark_started();

    // make sure the current processor is set during execution of the task
    ThreadLocal::current_processor = p;

    (*fptr)(args, arglen, p);

    // and clear the TLS when we're done
    ThreadLocal::current_processor = Processor::NO_PROC;

    mark_completed();

    log_task.info("thread finished running task %p for proc " IDFMT "",
		  this, proc.id);
#ifdef EVENT_GRAPH_TRACE
    unsigned long long stop = TimeStamp::get_current_time_in_micros();
    finish_enclosing();
    log_event_graph.debug("Task Time: (" IDFMT ",%d) %lld",
			  finish_event.id, finish_event.gen,
			  (stop - start));
#endif
#if 0
    log_util(((func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
	     "task end: %d (%p) (%s)", func_id, fptr, argstr);
#endif
    if(finish_event.exists())
      get_runtime()->get_genevent_impl(finish_event)->
	trigger(finish_event.gen, gasnet_mynode());
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadedTaskScheduler
  //

  ThreadedTaskScheduler::ThreadedTaskScheduler(void)
    : shutdown_flag(false)
    , active_worker_count(0)
    , unassigned_worker_count(0)
    , cfg_reuse_workers(true)
    , cfg_max_idle_workers(1)
    , cfg_min_active_workers(1)
    , cfg_max_active_workers(1)
  {}

  ThreadedTaskScheduler::~ThreadedTaskScheduler(void)
  {
    // make sure everything got cleaned up right
    assert(active_worker_count == 0);
    assert(unassigned_worker_count == 0);
    assert(idle_workers.empty());
  }

  void ThreadedTaskScheduler::add_task_queue(TaskQueue *queue)
  {
    AutoHSLLock al(lock);

    task_queues.push_back(queue);
  }

  // helper for tracking/sanity-checking worker counts
  inline void ThreadedTaskScheduler::update_worker_count(int active_delta,
							 int unassigned_delta,
							 bool check /*= true*/)
  {
    //define DEBUG_THREAD_SCHEDULER
#ifdef DEBUG_THREAD_SCHEDULER
    printf("UWC: %p a=%d%+d u=%d%+d\n", Thread::self(),
	   active_worker_count, active_delta,
	   unassigned_worker_count, unassigned_delta);
#endif

    active_worker_count += active_delta;
    unassigned_worker_count += unassigned_delta;

    if(check) {
      // active worker count should always be in bounds
      assert((active_worker_count >= cfg_min_active_workers) &&
	     (active_worker_count <= cfg_max_active_workers));

      // should always have an unassigned worker if there's room
      assert((unassigned_worker_count > 0) ||
	     (active_worker_count == cfg_max_active_workers));
    }
  }

  void ThreadedTaskScheduler::thread_blocking(Thread *thread)
  {
    // there's a potential race between a thread blocking and being reawakened,
    //  so take the scheduler lock and THEN try to mark the thread as blocked
    AutoHSLLock al(lock);

    bool really_blocked = try_update_thread_state(thread,
						  Thread::STATE_BLOCKING,
						  Thread::STATE_BLOCKED);

    // TODO: if the thread is already ready again, we might still choose to suspend it if
    //  higher-priority work is pending
    if(!really_blocked)
      return;

    while(true) {
      // let's try to find something better to do than spin our wheels

      // first choice - is there a resumable worker we can yield to?
      if(!resumable_workers.empty()) {
	Thread *yield_to = resumable_workers.get(0); // we don't care about priority
	// this preserves active and unassigned counts
	update_worker_count(0, 0);
	worker_sleep(yield_to);  // returns only when we're ready
	break;
      }

      // next choice - if we're above the min active count AND there's at least one
      //  unassigned worker active, we can just sleep
      if((active_worker_count > cfg_min_active_workers) &&
	 (unassigned_worker_count > 0)) {
	// this reduces the active worker count by one
	update_worker_count(-1, 0);
	worker_sleep(0);  // returns only when we're ready
	break;
      }

      // next choice - is there an idle worker we can yield to?
      if(!idle_workers.empty()) {
	Thread *yield_to = idle_workers.back();
	idle_workers.pop_back();
	// this preserves the active count, increased unassigned by 1
	update_worker_count(0, +1);
	worker_sleep(yield_to);  // returns only when we're ready
	break;
      }
	
      // last choice - create a new worker to mind the store
      // TODO: consider not doing this until we know there's work for it?
      if(true) {
	Thread *yield_to = worker_create(false);
	// this preserves the active count, increased unassigned by 1
	update_worker_count(0, +1);
	worker_sleep(yield_to);  // returns only when we're ready
	break;
      }

      // TODO: some sort of cpu yield here to at least prevent a tight spin
    }
  }

  void ThreadedTaskScheduler::thread_ready(Thread *thread)
  {
    // TODO: might be nice to do this in a lock-free way, since this is called by
    //  some other thread
    AutoHSLLock al(lock);

    // look up the priority of this thread and then add it to the resumable workers
    std::map<Thread *, int>::const_iterator it = worker_priorities.find(thread);
    assert(it != worker_priorities.end());
    // adding to the priority queue should wake up any sleeping workers if needed
    resumable_workers.put(thread, it->second);  // TODO: round-robin for now
  }

  // the main scheduler loop
  void ThreadedTaskScheduler::scheduler_loop(void)
  {
    // the entire body of this method, except for when running an actual task, is
    //   a critical section - lock should be taken by caller
    {
      //AutoHSLLock al(lock);

      // we're a new, and initially unassigned, worker - counters have already been updated

      while(true) {
	// first rule - always yield to a resumable worker
	while(!resumable_workers.empty()) {
	  Thread *yield_to = resumable_workers.get(0); // priority is irrelevant

	  // this should only happen if we're at the max active worker count (otherwise
	  //  somebody should have just woken this guy up earlier), and reduces the 
	  // unassigned worker count by one
	  update_worker_count(0, -1);

	  idle_workers.push_back(Thread::self());
	  worker_sleep(yield_to);

	  // we're awake again, but still looking for work...
	}

	// try to get a new task then
	// remember where a task has come from in case we want to put it back
	Task *task = 0;
	TaskQueue *task_source = 0;
	int task_priority = TaskQueue::PRI_NEG_INF;
	for(std::vector<TaskQueue *>::const_iterator it = task_queues.begin();
	    it != task_queues.end();
	    it++) {
	  int new_priority;
	  Task *new_task = (*it)->get(&new_priority, task_priority);
	  if(new_task) {
	    // if we got something better, put back the old thing (if any)
	    if(task)
	      task_source->put(task, task_priority, false); // back on front of list
	  
	    task = new_task;
	    task_source = *it;
	    task_priority = new_priority;
	  }
	}

	// did we find work to do?
	if(task) {
	  // we've now got some assigned work, so fire up a new idle worker if we were the last
	  //  and there's a room for another active worker
	  if((unassigned_worker_count == 1) && (active_worker_count < cfg_max_active_workers)) {
	    // create an active worker, net zero change in unassigned workers
	    update_worker_count(+1, 0);
	    worker_create(true); // start it running right away
	  } else {
	    // one fewer unassigned worker
	    update_worker_count(0, -1);
	  }

	  // we'll run the task after letting go of the lock, but update this thread's
	  //  priority here
	  worker_priorities[Thread::self()] = task_priority;

	  // release the lock while we run the task
	  lock.unlock();

	  bool ok = execute_task(task);
	  assert(ok);  // no fault recovery yet
	  // TODO: let operation table manage lifetime eventually
	  delete task;

	  lock.lock();

	  worker_priorities.erase(Thread::self());

	  // and we're back to being unassigned
	  update_worker_count(0, +1);

	  // are we allowed to reuse this worker for another task?
	  if(!cfg_reuse_workers) break;
	} else {
	  // no?  thumb twiddling time

	  // are we shutting down?
	  if(shutdown_flag) {
	    // yes, we can terminate - wake up an idler (if any) first though
	    if(!idle_workers.empty()) {
	      Thread *to_wake = idle_workers.back();
	      idle_workers.pop_back();
	      // no net change in worker counts
	      worker_terminate(to_wake);
	      break;
	    } else {
	      // nobody to wake, so -1 active/unassigned worker
	      update_worker_count(-1, -1, false); // ok to drop below mins
	      worker_terminate(0);
	      break;
	    }
	  }

	  // do we have more unassigned and idle tasks than we need?
	  int total_idle_count = (unassigned_worker_count +
				  (int)(idle_workers.size()));
	  if(total_idle_count > cfg_max_idle_workers) {
	    // if there are sleeping idlers, terminate in favor of one of those - keeps
	    //  worker counts constant
	    if(!idle_workers.empty()) {
	      Thread *to_wake = idle_workers.back();
	      idle_workers.pop_back();
	      // no net change in worker counts
	      worker_terminate(to_wake);
	      break;
	    } else {
	      // nobody to take our place, but if there's at least one other unassigned worker
	      //  we can just terminate
	      if((unassigned_worker_count > 1) &&
		 (active_worker_count > cfg_min_active_workers)) {
		update_worker_count(-1, -1, false);
		worker_terminate(0);
		break;
	      } else {
		// no, stay awake but yield the CPU for a bit
		idle_thread_yield();
	      }
	    }
	  } else {
	    // we don't want to terminate, but sleeping is still a possibility
	    if((unassigned_worker_count > 1) &&
	       (active_worker_count > cfg_min_active_workers)) {
	      update_worker_count(-1, -1, false);
	      idle_workers.push_back(Thread::self());
	      worker_sleep(0);
	    } else {
	      // no, stay awake but yield the CPU for a bit
	      idle_thread_yield();
	    }
	  }
	}
      }
    }
  }

  // an entry point that takes the scheduler lock explicitly
  void ThreadedTaskScheduler::scheduler_loop_wlock(void)
  {
    AutoHSLLock al(lock);
    scheduler_loop();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class KernelThreadTaskScheduler
  //

  KernelThreadTaskScheduler::KernelThreadTaskScheduler(Processor _proc,
						       CoreReservation& _core_rsrv)
    : proc(_proc)
    , core_rsrv(_core_rsrv)
    , shutdown_condvar(lock)
  {
  }

  KernelThreadTaskScheduler::~KernelThreadTaskScheduler(void)
  {
    // cleanup should happen before destruction
    assert(all_workers.empty());
  }

  void  KernelThreadTaskScheduler::add_task_queue(TaskQueue *queue)
  {
    // call the parent implementation first
    ThreadedTaskScheduler::add_task_queue(queue);
  }

  void KernelThreadTaskScheduler::start(void)
  {
    // fire up the minimum number of workers
    {
      AutoHSLLock al(lock);

      update_worker_count(cfg_min_active_workers, cfg_min_active_workers);

      for(int i = 0; i < cfg_min_active_workers; i++)
	worker_create(true);
    }
  }

  void KernelThreadTaskScheduler::shutdown(void)
  {
    shutdown_flag = true;

    // wait for all workers to finish
    {
      AutoHSLLock al(lock);

      while(!all_workers.empty())
	shutdown_condvar.wait();
    }

#ifdef DEBUG_THREAD_SCHEDULER
    printf("sched shutdown complete\n");
#endif
  }

  void KernelThreadTaskScheduler::thread_starting(Thread *thread)
  {
#ifdef DEBUG_THREAD_SCHEDULER
    printf("worker starting: %p\n", thread);
#endif

    // see if we're supposed to be active yet
    {
      AutoHSLLock al(lock);

      if(active_workers.count(thread) == 0) {
	// nope, sleep on a CV until we are
	GASNetCondVar my_cv(lock);
	sleeping_threads[thread] = &my_cv;

	while(active_workers.count(thread) == 0)
	  my_cv.wait();
    
	sleeping_threads.erase(thread);
      }
    }
  }

  void KernelThreadTaskScheduler::thread_terminating(Thread *thread)
  {
    AutoHSLLock al(lock);

    // if the thread is still in our all_workers list, this was unexpected
    if(all_workers.count(thread) > 0) {
      printf("unexpected worker termination: %p\n", thread);

      // if this was our last worker, and we're not shutting down,
      //  something bad probably happened - fire up a new worker and
      //  hope things work themselves out
      if((all_workers.size() == 1) && !shutdown_flag) {
	printf("HELP!  Lost last worker for proc " IDFMT "!", proc.id);
	worker_terminate(worker_create(false));
      } else {
	// just let it die
	worker_terminate(0);
      }
    }
  }

  bool KernelThreadTaskScheduler::execute_task(Task *task)
  {
    task->execute_on_processor(proc);
    return true;
  }

  Thread *KernelThreadTaskScheduler::worker_create(bool make_active)
  {
    // lock is held by caller
    ThreadLaunchParameters tlp;
    Thread *t = Thread::create_kernel_thread<ThreadedTaskScheduler,
					     &ThreadedTaskScheduler::scheduler_loop_wlock>(this,
											 tlp,
											 core_rsrv,
											 this);
    all_workers.insert(t);
    if(make_active)
      active_workers.insert(t);
    return t;
  }

  void KernelThreadTaskScheduler::worker_sleep(Thread *switch_to)
  {
    // lock is held by caller

#ifdef DEBUG_THREAD_SCHEDULER
    printf("switch: %p -> %p\n", Thread::self(), switch_to);
#endif

    // take ourself off the active list
    size_t count = active_workers.erase(Thread::self());
    assert(count == 1);

    GASNetCondVar my_cv(lock);
    sleeping_threads[Thread::self()] = &my_cv;

    // with kernel threads, sleeping and waking are separable actions
    if(switch_to)
      worker_wake(switch_to);

    // now sleep until we're active again
    while(active_workers.count(Thread::self()) == 0)
      my_cv.wait();
    
    // awake again, unregister our (stack-allocated) CV
    sleeping_threads.erase(Thread::self());
  }

  void KernelThreadTaskScheduler::worker_wake(Thread *to_wake)
  {
    // make sure target is actually asleep and mark active
    assert(active_workers.count(to_wake) == 0);
    active_workers.insert(to_wake);

    // if they have a CV (they might not yet), poke that
    std::map<Thread *, GASNetCondVar *>::const_iterator it = sleeping_threads.find(to_wake);
    if(it != sleeping_threads.end())
      it->second->signal();
  }

  void KernelThreadTaskScheduler::worker_terminate(Thread *switch_to)
  {
    // caller holds lock

    Thread *me = Thread::self();

#ifdef DEBUG_THREAD_SCHEDULER
    printf("terminate: %p -> %p\n", me, switch_to);
#endif

    // take ourselves off the active list (FOREVER...)
    size_t count = active_workers.erase(me);
    assert(count == 1);

    // also off the all workers list
    all_workers.erase(me);

    // and wake up whoever we're switching to (if any)
    if(switch_to)
      worker_wake(switch_to);

    // detach and delete the worker thread
    me->detach();
    delete me;
    
    // if this was the last thread, we'd better be in shutdown...
    if(all_workers.empty()) {
      assert(shutdown_flag);
      shutdown_condvar.signal();
    }
  }
  
  void KernelThreadTaskScheduler::idle_thread_yield(void)
  {
    // don't sleep, but let go of lock and other threads run
    lock.unlock();
    Thread::yield();
    lock.lock();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UserThreadTaskScheduler
  //

#ifdef REALM_USE_USER_THREADS
  UserThreadTaskScheduler::UserThreadTaskScheduler(Processor _proc,
						   CoreReservation& _core_rsrv)
    : proc(_proc)
    , core_rsrv(_core_rsrv)
    , cfg_num_host_threads(1)
  {
  }

  UserThreadTaskScheduler::~UserThreadTaskScheduler(void)
  {
    // cleanup should happen before destruction
    assert(all_workers.empty());
    assert(all_hosts.empty());
  }

  void  UserThreadTaskScheduler::add_task_queue(TaskQueue *queue)
  {
    // call the parent implementation first
    ThreadedTaskScheduler::add_task_queue(queue);
  }

  void UserThreadTaskScheduler::start(void)
  {
    // with user threading, active must always match the number of host threads
    cfg_min_active_workers = cfg_num_host_threads;
    cfg_max_active_workers = cfg_num_host_threads;

    // fire up the host threads (which will fire up initial workers)
    {
      AutoHSLLock al(lock);

      update_worker_count(cfg_num_host_threads, cfg_num_host_threads);

      ThreadLaunchParameters tlp;
      tlp.set_stack_size(4096);  // really small stack is fine here

      for(int i = 0; i < cfg_num_host_threads; i++) {
	Thread *t = Thread::create_kernel_thread<UserThreadTaskScheduler,
						 &UserThreadTaskScheduler::host_thread_loop>(this,
											tlp,
											core_rsrv,
											0);
	all_hosts.insert(t);
      }
    }
  }

  void UserThreadTaskScheduler::shutdown(void)
  {
    // set the shutdown flag and wait for all the host threads to exit
    AutoHSLLock al(lock);

    shutdown_flag = true;

    while(!all_hosts.empty()) {
      // pick an arbitrary host and join on it
      Thread *t = *all_hosts.begin();
      al.release();
      t->join();  // can't hold lock while waiting
      al.reacquire();
      all_hosts.erase(t);
      delete t;
    }
  }

  namespace ThreadLocal {
    // you can't delete a user thread until you've switched off of it, so
    //  use TLS to mark when that should happen
    static __thread Thread *terminated_user_thread = 0;
  };

  inline void UserThreadTaskScheduler::request_user_thread_cleanup(Thread *thread)
  {
    // make sure we haven't forgotten some other thread
    assert(ThreadLocal::terminated_user_thread == 0);
    ThreadLocal::terminated_user_thread = thread;
  }

  inline void UserThreadTaskScheduler::do_user_thread_cleanup(void)
  {
    if(ThreadLocal::terminated_user_thread != 0) {
      delete ThreadLocal::terminated_user_thread;
      ThreadLocal::terminated_user_thread = 0;
    }
  }

  void UserThreadTaskScheduler::host_thread_loop(void)
  {
    AutoHSLLock al(lock);

    while(!shutdown_flag) {
      // create a user worker thread - it won't start right away
      Thread *worker = worker_create(false);

      // for user ctx switching, lock is HELD during thread switches
      Thread::user_switch(worker);
      do_user_thread_cleanup();

      if(!shutdown_flag) {
	printf("HELP!  Lost a user worker thread - making a new one...\n");
	update_worker_count(+1, +1);
      }
    }
  }

  void UserThreadTaskScheduler::thread_starting(Thread *thread)
  {
    // nothing to do here
  }

  void UserThreadTaskScheduler::thread_terminating(Thread *thread)
  {
    // these threads aren't supposed to terminate
    assert(0);
  }

  bool UserThreadTaskScheduler::execute_task(Task *task)
  {
    task->execute_on_processor(proc);
    return true;
  }

  Thread *UserThreadTaskScheduler::worker_create(bool make_active)
  {
    // lock held by caller

    // user threads can never start active
    assert(!make_active);

    ThreadLaunchParameters tlp;
    Thread *t = Thread::create_user_thread<ThreadedTaskScheduler,
					   &ThreadedTaskScheduler::scheduler_loop>(this,
										   tlp,
										   this);
    all_workers.insert(t);
    //if(make_active)
    //  active_workers.insert(t);
    return t;
  }
    
  void UserThreadTaskScheduler::worker_sleep(Thread *switch_to)
  {
    // lock is held by caller

    // a user thread may not sleep without transferring control to somebody else
    assert(switch_to != 0);

#ifdef DEBUG_THREAD_SCHEDULER
    printf("switch: %p -> %p\n", Thread::self(), switch_to);
#endif

    // take ourself off the active list
    //size_t count = active_workers.erase(Thread::self());
    //assert(count == 1);

    Thread::user_switch(switch_to);

    do_user_thread_cleanup();

    // put ourselves back on the active list
    //active_workers.insert(Thread::self());
  }

  void UserThreadTaskScheduler::worker_wake(Thread *to_wake)
  {
    // in a user-threading environment, can't just wake a thread up out of nowhere
    assert(0);
  }

  void UserThreadTaskScheduler::worker_terminate(Thread *switch_to)
  {
    // lock is held by caller

    // terminating is like sleeping, except you are allowed to terminate to "nobody" when
    //  shutting down
    assert((switch_to != 0) || shutdown_flag);

#ifdef DEBUG_THREAD_SCHEDULER
    printf("terminate: %p -> %p\n", Thread::self(), switch_to);
#endif

    // take ourself off the active and all worker lists
    //size_t count = active_workers.erase(Thread::self());
    //assert(count == 1);

    size_t count = all_workers.erase(Thread::self());
    assert(count == 1);

    // whoever we switch to should delete us
    request_user_thread_cleanup(Thread::self());

    Thread::user_switch(switch_to);

    // we don't expect to ever get control back
    assert(0);
  }

  void UserThreadTaskScheduler::idle_thread_yield(void)
  {
    // don't sleep, but let go of lock and other threads run
    lock.unlock();
    Thread::yield();
    lock.lock();
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class Task
  //

}; // namespace Realm
