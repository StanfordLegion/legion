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

#include "realm/tasks.h"

#include "realm/runtime_impl.h"

namespace Realm {

  Logger log_task("task");
  Logger log_sched("sched");

  ////////////////////////////////////////////////////////////////////////
  //
  // class Task
  //

  Task::Task(Processor _proc, Processor::TaskFuncID _func_id,
	     const void *_args, size_t _arglen,
	     const ProfilingRequestSet &reqs,
	     Event _before_event,
	     GenEventImpl *_finish_event, EventImpl::gen_t _finish_gen,
	     int _priority)
    : Operation(_finish_event, _finish_gen, reqs)
    , proc(_proc), func_id(_func_id),
      before_event(_before_event), priority(_priority),
      executing_thread(0), marked_ready(false), pending_head(0)
  {
    // clamp task priority to "finite" range
    if(priority < TaskQueue::PRI_MIN_FINITE) priority = TaskQueue::PRI_MIN_FINITE;
    if(priority > TaskQueue::PRI_MAX_FINITE) priority = TaskQueue::PRI_MAX_FINITE;

    arglen = _arglen;
    if(arglen <= SHORT_ARGLEN_MAX) {
      if(arglen) {
	memcpy(short_argdata, _args, arglen);
        argdata = short_argdata;
      } else
	argdata = 0;
      free_argdata = false;
    } else {
      argdata = static_cast<char *>(malloc(arglen));
      assert(argdata != 0);
      memcpy(argdata, _args, arglen);
      free_argdata = true;
    }
    log_task.info() << "task " << (void *)this << " created: func=" << func_id
		    << " proc=" << _proc << " arglen=" << _arglen
		    << " before=" << _before_event << " after=" << get_finish_event();
  }

  Task::~Task(void)
  {
    if(free_argdata)
      free(argdata);

    // if we're still holding a reference to a pending_head, something is wrong
    assert(pending_head.load() == 0);
  }

  void Task::print(std::ostream& os) const
  {
    os << "task(proc=" << proc << ", func=" << func_id << ")";
  }

  bool Task::mark_ready(void)
  {
    log_task.info() << "task " << (void *)this << " ready: func=" << func_id
		    << " proc=" << proc << " arglen=" << arglen
		    << " before=" << before_event << " after=" << get_finish_event();
    bool success = Operation::mark_ready();
    if(!success) {
      // might still want the timeline info
      if(wants_timeline)
	timeline.record_ready_time();
    }
    // unconditionally set the 'marked_ready' bit for the benefit of any
    //  tasks that were pending with this one (they can't use the state of this
    //  task as a cancellation would obscure whether it had become ready first)
    marked_ready.store_release(true);
    return success;
  }

  bool Task::mark_started(void)
  {
    log_task.info() << "task " << (void *)this << " started: func=" << func_id
		    << " proc=" << proc << " arglen=" << arglen
		    << " before=" << before_event << " after=" << get_finish_event();
    // perform a delayed WAITING->READY update now if we were part of a chain
    //  of tasks - don't actually look at the head (e.g. to grab the ready
    //  time) until we're sure we got in ahead of any cancellations tohugh
    bool did_pending_update = false;
    if(pending_head.load() != 0) {
      Status::Result expected = Status::WAITING;
      if(state.compare_exchange(expected, Status::READY))
	did_pending_update = true;
    }
    bool success = Operation::mark_started();
    // if this succeeded, it's our job to release our hold on a pending_head
    if(success) {
      uintptr_t phead = pending_head.load() & ~uintptr_t(1);
      if(phead != 0) {
	// still have to avoid races with get_state calls though
	while(true) {
	  uintptr_t prev = phead;
	  if(pending_head.compare_exchange(prev, 0)) break;
	  assert(prev == (phead + 1));
	}
	Task *headptr = reinterpret_cast<Task *>(phead);
//#ifdef DEBUG_REALM
	// no need for load_acquire here, because the pushing and popping of
	//  this task (and the head task) involved a mutex
	{
	  bool head_ready = headptr->marked_ready.load();
	  if(!head_ready) {
	    log_task.fatal() << "HELP: headptr=" << std::hex << phead << std::dec
			     << " ready=" << head_ready << " state=" << headptr->state.load();
	    abort();
	  }
	}
	//assert(headptr->marked_ready.load());
//#endif
	if(did_pending_update && wants_timeline)
	  timeline.ready_time = headptr->timeline.ready_time;
	headptr->remove_reference();
      }
    }
    return success;
  }

  void Task::mark_completed(void)
  {
    log_task.info() << "task " << (void *)this << " completed: func=" << func_id
		    << " proc=" << proc << " arglen=" << arglen
		    << " before=" << before_event << " after=" << get_finish_event();
    Operation::mark_completed();
  }

  Operation::Status::Result Task::get_state(void)
  {
    Status::Result lazy_state = state.load();

    // if we're in pending but part of a chain, check our head task
    if((lazy_state == Status::WAITING) && (pending_head.load() != 0)) {
      // try to set the bottom bit of pending_head to prevent races with
      //  calls to mark_started or attempt_cancellation that clear pending_head
      while(true) {
	uintptr_t phead = pending_head.fetch_or(1);
	if(phead == 0) {
	  // nope, already been cleared, so nothing for us to look at
	  pending_head.store(0);
	  break;
	}
	if((phead & 1) == 1) {
	  // somebody else is already holding the lock, so try again
	  continue;
	}
	// success - we can safely look at marked_ready and the timeline now
	Task *headptr = reinterpret_cast<Task *>(phead);
	bool success = false;
	if(headptr->marked_ready.load_acquire()) {
	  Status::Result expected = Status::WAITING;
	  success = state.compare_exchange(expected, Status::READY);
	}
	if(success && wants_timeline)
	  timeline.ready_time = headptr->timeline.ready_time;

	// clear the LSB to release our hold - clearing of the whole pointer
	//  will be done by mark_started or attempt_cancellation
	pending_head.fetch_sub(1);

	return (success ? Status::READY : lazy_state);
      }
    }

    // in any other case, our state was actually current
    return lazy_state;
  }

  bool Task::attempt_cancellation(int error_code, const void *reason_data, size_t reason_size)
  {
    // let the base class handle the easy cases
    if(Operation::attempt_cancellation(error_code, reason_data, reason_size)) {
      // if we had a reference to a head pending task, let go of it
      uintptr_t phead = pending_head.load() & ~uintptr_t(1);
      if(phead != 0) {
	// still have to avoid races with get_state calls though
	while(true) {
	  uintptr_t prev = phead;
	  if(pending_head.compare_exchange(prev, 0)) break;
	  assert(prev == (phead + 1));
	}
	Task *headptr = reinterpret_cast<Task *>(phead);
	if(wants_timeline)
	  timeline.ready_time = headptr->timeline.ready_time;
	headptr->remove_reference();
      }
      return true;
    }

    // for a running task, see if we can signal it to stop
    Status::Result prev = Status::RUNNING;
    if(state.compare_exchange(prev, Status::INTERRUPT_REQUESTED)) {
      status.error_code = error_code;
      status.error_details.set(reason_data, reason_size);
      Thread *t = executing_thread;
      assert(t != 0);
      t->signal(Thread::TSIG_INTERRUPT, true /*async*/);
      return true;
    }

    // let our caller try more ideas if it has any
    return false;
  }

  void Task::set_priority(int new_priority)
  {
    priority = new_priority;
    Thread *t = executing_thread;
    if(t)
      t->set_priority(new_priority);
  }

  void Task::execute_on_processor(Processor p)
  {
    // if the processor isn't specified, use what's in the task object
    if(!p.exists())
      p = this->proc;

    //Processor::TaskFuncPtr fptr = get_runtime()->task_table[func_id];
#if 0
    char argstr[100];
    argstr[0] = 0;
    for(size_t i = 0; (i < arglen) && (i < 40); i++)
      sprintf(argstr+2*i, "%02x", ((unsigned char *)(args))[i]);
    if(arglen > 40) strcpy(argstr+80, "...");
    log_util(((func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
	     "task start: %d (%p) (%s)", func_id, fptr, argstr);
#endif

    // does the profiler want to know where it was run?
    if(measurements.wants_measurement<ProfilingMeasurements::OperationProcessorUsage>()) {
      ProfilingMeasurements::OperationProcessorUsage opu;
      opu.proc = p;
      measurements.add_measurement(opu);
    }

    // indicate which thread will be running this task before we mark it running
    Thread *thread = Thread::self();
    executing_thread = thread;
    thread->start_operation(this);

    // set up any requested performance counters
    thread->setup_perf_counters(measurements);

    // mark that we're starting the task, checking for cancellation
    bool ok_to_run = mark_started();

    if(ok_to_run) {
      // make sure the current processor is set during execution of the task
      ThreadLocal::current_processor = p;

#ifdef REALM_USE_EXCEPTIONS
      // even if exceptions are enabled, we only install handlers if somebody is paying
      //  attention to the OperationStatus
      if(measurements.wants_measurement<ProfilingMeasurements::OperationStatus>() ||
	 measurements.wants_measurement<ProfilingMeasurements::OperationAbnormalStatus>()) {	 
	try {
	  Thread::ExceptionHandlerPresence ehp;
	  thread->start_perf_counters();
	  get_runtime()->get_processor_impl(p)->execute_task(func_id,
							     ByteArrayRef(argdata, arglen));
	  thread->stop_perf_counters();
	  thread->stop_operation(this);
	  thread->record_perf_counters(measurements);
	  mark_finished(true /*successful*/);
	}
	catch (const ExecutionException& e) {
	  e.populate_profiling_measurements(measurements);
	  thread->stop_operation(this);
	  mark_terminated(e.error_code, e.details);
	}
      } else
#endif
      {
	// just run the task - if it completes, we assume it was successful
	thread->start_perf_counters();
	get_runtime()->get_processor_impl(p)->execute_task(func_id,
							   ByteArrayRef(argdata, arglen));
	thread->stop_perf_counters();
	thread->stop_operation(this);
	thread->record_perf_counters(measurements);
	mark_finished(true /*successful*/);
      }

      // and clear the TLS when we're done
      // TODO: get this right when using user threads
      //ThreadLocal::current_processor = Processor::NO_PROC;
    } else {
      // !ok_to_run
      thread->stop_operation(this);
      mark_finished(false /*!successful*/);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Task::DeferredSpawn
  //

  Task::DeferredSpawn::DeferredSpawn(void)
    : is_triggered(false)
    , list_length(0)
  {}

  void Task::DeferredSpawn::setup(ProcessorImpl *_proc, Task *_task,
				  Event _wait_on)
  {
    proc = _proc;
    task = _task;
    wait_on = _wait_on;
  }

  void Task::DeferredSpawn::defer(EventImpl *_wait_impl,
                                  EventImpl::gen_t _wait_gen)
  {
    {
      AutoLock<> al(pending_list_mutex);
      // insert ourselves in the pending list
      pending_list.push_back(task);
      list_length++;
    }
    _wait_impl->add_waiter(_wait_gen, this);
  }
    
  void Task::DeferredSpawn::event_triggered(bool poisoned, TimeLimit work_until)
  {
    // record the triggering, which closes the pending_list
    {
      AutoLock<> al(pending_list_mutex);
      is_poisoned = poisoned;
      is_triggered = true;
    }
    if(poisoned) {
      // hold a reference on this task while we cancel a bunch of tasks including ourself
      task->add_reference();
      while(!pending_list.empty(INT_MIN)) {
        Task *to_cancel = pending_list.pop_front(INT_MIN);
        // cancel the task - this has to work
        log_poison.info() << "cancelling poisoned task - task=" << to_cancel << " after=" << to_cancel->get_finish_event();
        to_cancel->handle_poisoned_precondition(wait_on);
      }
      task->remove_reference();
    } else {
      //log_task.print() << "enqueuing " << pending_list.size() << " tasks";
      proc->enqueue_tasks(pending_list, list_length);
    }
  }

  void Task::DeferredSpawn::print(std::ostream& os) const
  {
    os << "deferred task: func=" << task->func_id << " proc=" << task->proc << " finish=" << task->get_finish_event();
  }

  Event Task::DeferredSpawn::get_finish_event(void) const
  {
    // TODO: change this interface to return multiple finish events
    return task->get_finish_event();
  }

  // attempts to add another task to the this deferred spawn group -
  // returns true on success, or false if the event has already
  //  triggered, in which case 'poisoned' is set appropriately
  bool Task::DeferredSpawn::add_task(Task *to_add, bool& poisoned)
  {
    assert(to_add->before_event == wait_on);
    bool ok;
    {
      AutoLock<> al(pending_list_mutex);
      if(is_triggered) {
	ok = false;
	poisoned = is_poisoned;
      } else {
	ok = true;
	pending_list.push_back(to_add);
	list_length++;
	// task we added will hold a reference this head task
	task->add_reference();
	to_add->pending_head.store(reinterpret_cast<uintptr_t>(task));
	// make sure to record our ready time if anybody in the chain needs it
	if(to_add->wants_timeline)
	  task->wants_timeline = true;
      }
    }
    return ok;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TaskQueue
  //

  TaskQueue::TaskQueue(void)
    : task_count_gauge(0)
  {}

  void TaskQueue::add_subscription(NotificationCallback *callback,
				   priority_t higher_than /*= PRI_NEG_INF*/)
  {
    AutoLock<FIFOMutex> al(mutex);
    callbacks.push_back(callback);
    callback_priorities.push_back(higher_than);
  }

  void TaskQueue::remove_subscription(NotificationCallback *callback)
  {
    AutoLock<FIFOMutex> al(mutex);
    std::vector<NotificationCallback *>::iterator cit = callbacks.begin();
    std::vector<priority_t>::iterator cpit = callback_priorities.begin();
    while (cit != callbacks.end()) {
      if( *cit == callback) {
        callbacks.erase(cit);
        callback_priorities.erase(cpit);
        break;
      }
      ++cit; ++cpit;
    }
  }
  
  void TaskQueue::set_gauge(ProfilingGauges::AbsoluteRangeGauge<int> *new_gauge)
  {
    task_count_gauge = new_gauge;
  }

  void TaskQueue::free_gauge()
  {
    delete task_count_gauge;
    task_count_gauge = 0;
  }

  // gets highest priority task available from any task queue
  /*static*/ Task *TaskQueue::get_best_task(const std::vector<TaskQueue *>& queues,
					    int& task_priority)
  {
    // remember where a task has come from in case we want to put it back
    Task *task = 0;
    TaskQueue *task_source = 0;

    for(std::vector<TaskQueue *>::const_iterator it = queues.begin();
	it != queues.end();
	it++) {
      Task *new_task;
      {
	AutoLock<FIFOMutex> al((*it)->mutex);
	new_task = (*it)->ready_task_list.pop_front(task_priority+1);
      }
      if(new_task) {
	if((*it)->task_count_gauge)
	  *((*it)->task_count_gauge) -= 1;

	// if we got something better, put back the old thing (if any)
	if(task) {
	  {
	    AutoLock<FIFOMutex> al(task_source->mutex);
	    task_source->ready_task_list.push_front(task);
	  }
	  if(task_source->task_count_gauge)
	    (*task_source->task_count_gauge) += 1;
	}
	  
	task = new_task;
	task_source = *it;
	task_priority = task->priority;
      }
    }

    return task;
  }

  void TaskQueue::enqueue_task(Task *task)
  {
    priority_t notify_priority = PRI_NEG_INF;

    // just jam it into the task queue
    if(task->mark_ready()) {
//#ifdef DEBUG_REALM
      // we should not be directly pushing a task that was part of a chain
      assert(task->pending_head.load() == 0);
//#endif

      {
	AutoLock<FIFOMutex> al(mutex);
	if(ready_task_list.empty(task->priority))
	  notify_priority = task->priority;
	ready_task_list.push_back(task);
      }

      if(task_count_gauge)
	*task_count_gauge += 1;

      if(notify_priority > PRI_NEG_INF)
	for(size_t i = 0; i < callbacks.size(); i++)
	  if(notify_priority >= callback_priorities[i])
	    callbacks[i]->item_available(notify_priority);
    } else
      task->mark_finished(false /*!successful*/);
  }

  void TaskQueue::enqueue_tasks(Task::TaskList& tasks, size_t num_tasks)
  {
    // early out if there are no tasks to add
    if(tasks.empty(PRI_NEG_INF)) return;

    // mark just the first task as ready - the remaining tasks will lazily
    //  pick up the update when mark_started gets called or when the state
    //  is explictly queried
    // we lose the ability to filter out tasks that have been cancelled, but
    //  that'll happen in mark_started too, and it saves us from messing with
    //  the list structure here
    // the catch is that the "first" task isn't necessarily the one at the
    //  front of the list (since higher-priority tasks jump ahead in the list)
    Task *first_task = tasks.front();
    uintptr_t phead = first_task->pending_head.load();
    if(phead != 0) {
      // LSB needs to be masked off to make a valid pointer
      first_task = reinterpret_cast<Task *>(phead & ~uintptr_t(1));
//#ifdef DEBUG_REALM
      // this task had better not also have a pending_head link
      assert(first_task->pending_head.load() == 0);
//#endif
    }
    first_task->mark_ready();

    // we'll tentatively notify based on the highest priority task to be
    //  added
    priority_t notify_priority = tasks.front()->priority;

    {
      AutoLock<FIFOMutex> al(mutex);
      // cancel notification if we already have equal/higher priority tasks
      if(!ready_task_list.empty(notify_priority))
	notify_priority = PRI_NEG_INF;
      // absorb new list into ours
      ready_task_list.absorb_append(tasks);
    }

    if(task_count_gauge)
      *task_count_gauge += num_tasks;

    if(notify_priority > PRI_NEG_INF)
      for(size_t i = 0; i < callbacks.size(); i++)
	if(notify_priority >= callback_priorities[i])
	  callbacks[i]->item_available(notify_priority);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadedTaskScheduler::WorkCounter
  //

  ThreadedTaskScheduler::WorkCounter::WorkCounter(void)
    : counter(0)
    , interrupt_flag(0)
  {}

  ThreadedTaskScheduler::WorkCounter::~WorkCounter(void)
  {}

  void ThreadedTaskScheduler::WorkCounter::set_interrupt_flag(atomic<bool> *_interrupt_flag)
  {
    interrupt_flag = _interrupt_flag;
  }

  void ThreadedTaskScheduler::WorkCounter::increment_counter(void)
  {
    // set the interrupt flag if we have one
    if(interrupt_flag)
      interrupt_flag->store(true);

    // bump the counter and atomically see if any sleepers exist
    uint64_t old_value = counter.fetch_add_acqrel(1 << SLEEPER_BITS);
    uint64_t sleepers = (old_value & ((1 << SLEEPER_BITS) - 1));

    // no sleepers?  all done
    if(REALM_LIKELY(sleepers == 0))
      return;

    // try to claim sleepers to wake up
    uint64_t expected = old_value + (1 << SLEEPER_BITS);
    while(true) {
      uint64_t newval = expected - sleepers;

      if(counter.compare_exchange(expected, newval)) {
        // success - we must now notify exactly that many sleepers on the
        //   doorbell list, but we have to get exclusive popping access first

        uint64_t tstate;
        uint64_t act_pops = db_mutex.attempt_enter(sleepers, tstate);
        while(act_pops != 0) {
          db_list.notify_newest(act_pops, true /*prefer_spinning*/);

          // try to release mutex, but loop in case work was delegated to us
          act_pops = db_mutex.attempt_exit(tstate);
        }

        break;
      } else {
        // two failure cases:
        //  1) somebody else has bumped the work counter in the mean time -
        //      this is great, because waking the sleepers is now their problem
        //  2) additional sleepers (on the _post-incremented_) count have shown
        //      up - it's probably futile to wake the older sleepers, but let's
        //      try it anyway
        if((expected >> SLEEPER_BITS) != ((old_value >> SLEEPER_BITS) + 1)) {
          // case 1
          break;
        } else {
          // case 2
          continue;
        }
      }
    }
  }

  // waits until new work arrives - this will possibly go to sleep,
  //  so should not be called while holding another lock
  void ThreadedTaskScheduler::WorkCounter::wait_for_work(uint64_t old_counter)
  {
    // use a CAS to add to the sleeper count, but only if the counter hasn't
    //  moved on
    uint64_t oldval = counter.load_acquire();
    while(true) {
      if((oldval >> SLEEPER_BITS) != old_counter) {
        // counter moved - no need to wait
        return;
      }

      if(counter.compare_exchange(oldval, oldval + 1)) {
        // success - we can add our doorbell to the list and wait on it below
        break;
      }
    }

    Doorbell *db = Doorbell::get_thread_doorbell();
    db->prepare();
    if(db_list.add_doorbell(db)) {
      db->wait();
    } else {
      // signal came first, so cancel our wait
      db->cancel();
    }
#ifdef DEBUG_REALM
    // sanity-check that counter did move
    uint64_t checkval = counter.load();
    assert((checkval >> SLEEPER_BITS) != old_counter);
    (void)checkval;
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadedTaskScheduler
  //

  ThreadedTaskScheduler::ThreadedTaskScheduler(void)
    : shutdown_flag(false)
    , active_worker_count(0)
    , unassigned_worker_count(0)
    , wcu_task_queues(this)
    , wcu_resume_queue(this)
    , bgworker_interrupt(false)
    , max_bgwork_timeslice(0)
    , cfg_reuse_workers(true)
    , cfg_max_idle_workers(1)
    , cfg_min_active_workers(1)
    , cfg_max_active_workers(1)
  {
    // hook up the work counter updates for the resumable worker queue
    resumable_workers.add_subscription(&wcu_resume_queue);
  }

  ThreadedTaskScheduler::~ThreadedTaskScheduler(void)
  {
    // make sure everything got cleaned up right
    assert(active_worker_count == 0);
    assert(unassigned_worker_count == 0);
    assert(idle_workers.empty());
  }

  void ThreadedTaskScheduler::add_task_queue(TaskQueue *queue)
  {
    AutoLock<FIFOMutex> al(lock);

    task_queues.push_back(queue);

    // hook up the work counter updates for this queue
    queue->add_subscription(&wcu_task_queues);
  }

  void ThreadedTaskScheduler::remove_task_queue(TaskQueue *queue)
  {
    AutoLock<FIFOMutex> al(lock);
    for (std::vector<TaskQueue *>::iterator it = task_queues.begin(); it != task_queues.end();++it) {
      if (*it == queue) {
        //found; we erase and exit
        task_queues.erase(it);
        break;
      }
    }
    
    // un-hook up the work counter updates for this queue
    queue->remove_subscription(&wcu_task_queues);
  }

  void ThreadedTaskScheduler::add_task_context(const TaskContextManager *_manager)
  {
    AutoLock<FIFOMutex> al(lock);

    context_managers.push_back(_manager);
  }

  void ThreadedTaskScheduler::configure_bgworker(BackgroundWorkManager *manager,
						 long long max_timeslice,
						 int numa_domain)
  {
    if(max_timeslice > 0) {
      bgworker.set_manager(manager);
      // convert from us to ns
      bgworker.set_max_timeslice(max_timeslice * 1000);
      bgworker.set_numa_domain(numa_domain);

      work_counter.set_interrupt_flag(&bgworker_interrupt);
    }

    max_bgwork_timeslice = max_timeslice;
  }

  // helper for tracking/sanity-checking worker counts
  void ThreadedTaskScheduler::update_worker_count(int active_delta,
						  int unassigned_delta,
						  bool check /*= true*/)
  {
    //define DEBUG_THREAD_SCHEDULER
#ifdef DEBUG_THREAD_SCHEDULER
    printf("UWC: %p %p a=%d%+d u=%d%+d\n", Thread::self(), this,
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
    AutoLock<FIFOMutex> al(lock);

    bool really_blocked = try_update_thread_state(thread,
						  Thread::STATE_BLOCKING,
						  Thread::STATE_BLOCKED);

    // TODO: if the thread is already ready again, we might still choose to suspend it if
    //  higher-priority work is pending
    if(!really_blocked)
      return;

    // if this thread has enabled the scheduler lock, we're not going to
    //  yield to anybody - we'll spin until we're marked as being ready
    //  again
    if(ThreadLocal::scheduler_lock > 0) {
      log_sched.debug() << "thread w/ scheduler lock spinning: " << thread;
      spinning_workers.insert(thread);
      while(true) {
	uint64_t old_work_counter = work_counter.read_counter();
	switch(thread->get_state()) {
	case Thread::STATE_READY:
	  {
	    log_sched.debug() << "thread w/ scheduler lock ready: " << thread;
	    return;
	  }

	case Thread::STATE_ALERTED:
	  {
	    log_sched.debug() << "thread w/ scheduler lock alerted: " << thread;
	    thread->process_signals();
	    bool resuspended = try_update_thread_state(thread,
						       Thread::STATE_ALERTED,
						       Thread::STATE_BLOCKED);
	    if(!resuspended) {
	      assert(thread->get_state() == Thread::STATE_READY);
	      return;
	    }
	    break;
	  }

	case Thread::STATE_BLOCKED:
	  {
	    // twiddle our thumbs until something happens
	    wait_for_work(old_work_counter);
	    break;
	  }

	default:
	  assert(0);
	};
      }
    }

#ifdef DEBUG_THREAD_SCHEDULER
    assert(blocked_workers.count(thread) == 0);
#endif
    blocked_workers.insert(thread);

    log_sched.debug() << "scheduler worker blocking: sched=" << this << " worker=" << thread;

    while(true) {//thread->get_state() != Thread::STATE_READY) {
      bool alerted = try_update_thread_state(thread,
					     Thread::STATE_ALERTED,
					     Thread::STATE_BLOCKED);
      if(alerted) {
	log_sched.debug() << "thread alerted while blocked: sched=" << this << " worker=" << thread;
	thread->process_signals();
      }

      // let's try to find something better to do than spin our wheels

      // remember the work counter value before we start so that we don't iterate
      //   unnecessarily
      //uint64_t old_work_counter = work_counter.read_counter();

      // first choice - is there a resumable worker we can yield to?
      if(!resumable_workers.empty()) {
	Thread *yield_to = resumable_workers.get(0); // we don't care about priority
	// first check - is this US?  if so, we're done
	if(yield_to == thread) {
	  printf("resuming ourselves! (%d)\n", thread->get_state());
	  if(thread->get_state() != Thread::STATE_READY) continue;
	  break;
	}
	// this preserves active and unassigned counts
	update_worker_count(0, 0);
	worker_sleep(yield_to);
	// go back around in case this was just an alert
	if(thread->get_state() != Thread::STATE_READY) continue;
	break;
      }

      // next choice - if we're above the min active count AND there's at least one
      //  unassigned worker active, we can just sleep
      if((active_worker_count > cfg_min_active_workers) &&
	 (unassigned_worker_count > 0)) {
	// this reduces the active worker count by one
	update_worker_count(-1, 0);
	worker_sleep(0);
	// go back around in case this was just an alert
	if(thread->get_state() != Thread::STATE_READY) continue;
	break;
      }

      // next choice - is there an idle worker we can yield to?
      if(!idle_workers.empty()) {
	Thread *yield_to = idle_workers.back();
	idle_workers.pop_back();
	// this preserves the active count, increased unassigned by 1
	update_worker_count(0, +1);
	worker_sleep(yield_to);
	// go back around in case this was just an alert
	if(thread->get_state() != Thread::STATE_READY) continue;
	break;
      }
	
      // last choice - create a new worker to mind the store
      // TODO: consider not doing this until we know there's work for it?
      if(true) {
	Thread *yield_to = worker_create(false);
	// this preserves the active count, increased unassigned by 1
	update_worker_count(0, +1);
	worker_sleep(yield_to);
	// go back around in case this was just an alert
	if(thread->get_state() != Thread::STATE_READY) continue;
	break;
      }

      // wait at least until some new work shows up (even if we don't end up getting it,
      //  it's better than a tight spin)
      //wait_for_work(old_work_counter);
    }

#ifdef DEBUG_THREAD_SCHEDULER
    assert(blocked_workers.count(thread) == 1);
#endif
    blocked_workers.erase(thread);
  }

  void ThreadedTaskScheduler::thread_ready(Thread *thread)
  {
    log_sched.debug() << "scheduler worker ready: sched=" << this << " worker=" << thread;

    // TODO: might be nice to do this in a lock-free way, since this is called by
    //  some other thread
    AutoLock<FIFOMutex> al(lock);

    // if this was a spinning thread, remove it from the list and poke the
    //  work counter in cases its execution resource is napping
    if(!spinning_workers.empty()) {
      std::set<Thread *>::iterator it = spinning_workers.find(thread);
      if(it != spinning_workers.end()) {
	spinning_workers.erase(it);
	work_counter.increment_counter();
	return;
      }
    }

    // it may be that the thread has noticed that it is ready already, in
    //  which case it'll no longer be blocked and we don't want to resume it
    if(blocked_workers.count(thread) == 0)
      return;

    // this had better not be after shutdown was initiated
    if(shutdown_flag.load()) {
      log_sched.fatal() << "scheduler worker awakened during shutdown: sched=" << this << " worker=" << thread;
      abort();
    }

    // look up the priority of this thread and then add it to the resumable workers
    std::map<Thread *, int>::const_iterator it = worker_priorities.find(thread);
    assert(it != worker_priorities.end());
    int priority = it->second;

    // if this worker is higher priority than any other resumable workers and we're
    //  not at the max active thread count, we can immediately wake up the thread
    if((active_worker_count < cfg_max_active_workers) &&
       resumable_workers.empty(priority-1)) {
      update_worker_count(+1, 0);
      worker_wake(thread);
    } else {
      resumable_workers.put(thread, priority);  // TODO: round-robin for now
    }
  }

  void ThreadedTaskScheduler::set_thread_priority(Thread *thread, int new_priority)
  {
    int old_priority;

    {
      AutoLock<FIFOMutex> al(lock);
      std::map<Thread *, int>::iterator it = worker_priorities.find(thread);
      assert(it != worker_priorities.end());
      old_priority = it->second;
      it->second = new_priority;
    }

    log_sched.debug() << "thread priority change: thread=" << (void *)thread << " old=" << old_priority << " new=" << new_priority;
  }

  void ThreadedTaskScheduler::add_internal_task(InternalTask *itask)
  {
    // no need to take the main mutex - just add to the internal task list and
    //   bump the work counter
    internal_tasks.push_back(itask);
    work_counter.increment_counter();
  }

  // the main scheduler loop
  void ThreadedTaskScheduler::scheduler_loop(void)
  {
    // the entire body of this method, except for when running an actual task, is
    //   a critical section - lock should be taken by caller
    {
      //AutoLock<> al(lock);

      // we're a new, and initially unassigned, worker - counters have already been updated

      while(true) {
	// remember the work counter value before we start so that we don't iterate
	//   unnecessarily
	uint64_t old_work_counter = work_counter.read_counter();

	// internal tasks always take precedence
	while(!internal_tasks.empty()) {
	  InternalTask *itask = internal_tasks.pop_front();
	  // if somebody else popped the task first, we've nothing to do
	  if(!itask) break;

	  // one fewer unassigned worker
	  update_worker_count(0, -1);
	
	  // we'll run the task after letting go of the lock, but update this thread's
	  //  priority here
	  worker_priorities[Thread::self()] = TaskQueue::PRI_POS_INF;

	  // drop scheduler lock while we execute the internal task
	  lock.unlock();

	  // internal tasks are not allowed to context switch, so engage the
	  //  scheduler lock
	  ThreadLocal::scheduler_lock++;

	  // if we have any context managers, create the necessary contexts
	  std::vector<void *> contexts(context_managers.size(), 0);
	  if(!context_managers.empty())
	    for(size_t i = 0; i < context_managers.size(); i++)
	      contexts[i] = context_managers[i]->create_context(0 /*task*/);

	  execute_internal_task(itask);

          // destroy contexts in reverse order
	  if(!context_managers.empty())
	    for(size_t i = context_managers.size(); i > 0; i--)
	      context_managers[i-1]->destroy_context(0 /*task*/, contexts[i-1]);

          ThreadLocal::scheduler_lock--;

	  // we don't delete the internal task object - it can do that itself
	  //  if it wants, or the requestor of the operation can do it once
	  //  completion has been determined

	  lock.lock();

	  worker_priorities.erase(Thread::self());

	  // and we're back to being unassigned
	  update_worker_count(0, +1);
	}
	
	// if we have both resumable and new ready tasks, we want the one that
	//  is the highest priority, with ties going to resumable tasks - we
	//  can do this cleanly by taking advantage of the fact that the
	//  resumable_workers queue uses the scheduler lock, so can't change
	//  during this call
	// peek at the top thing (if any) in that queue, and then try to find
	//  a ready task with higher priority
	int resumable_priority = ResumableQueue::PRI_NEG_INF;
	resumable_workers.peek(&resumable_priority);

	// try to get a new task then
	int task_priority = resumable_priority;
	Task *task = TaskQueue::get_best_task(task_queues, task_priority);

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

	  // if we have any context managers, create the necessary contexts
	  std::vector<void *> contexts(context_managers.size(), 0);
	  if(!context_managers.empty()) {
	    for(size_t i = 0; i < context_managers.size(); i++)
	      contexts[i] = context_managers[i]->create_context(task);
	    // add a reference to the task to prevent it from being deleted
	    //  before we destroy our contexts
	    // NOTE: this does NOT prevent the task from finishing - a context
	    //  should add an AsyncWorkItem to the task if that is needed
	    task->add_reference();
	  }

#ifndef NDEBUG
	  bool ok =
#endif
	    execute_task(task);
	  assert(ok);  // no fault recovery yet

	  if(!context_managers.empty()) {
	    // destroy contexts in reverse order
	    for(size_t i = context_managers.size(); i > 0; i--)
	      context_managers[i-1]->destroy_context(task, contexts[i-1]);
	    // and only then remove the extra reference we were holding
	    task->remove_reference();
	  }

	  lock.lock();

	  worker_priorities.erase(Thread::self());

	  // and we're back to being unassigned
	  update_worker_count(0, +1);

	  // are we allowed to reuse this worker for another task?
	  if(cfg_reuse_workers) continue;

	  // if not, terminate
	  break;
	}

	// having checked for higher-priority ready tasks, we can always
	//  take the highest-priority resumable task, if any, and run it
	if(!resumable_workers.empty()) {
	  Thread *yield_to = resumable_workers.get(0); // priority is irrelevant
	  assert(yield_to != Thread::self());

	  // this should only happen if we're at the max active worker count (otherwise
	  //  somebody should have just woken this guy up earlier), and reduces the 
	  // unassigned worker count by one
	  update_worker_count(0, -1);

	  idle_workers.push_back(Thread::self());
	  worker_sleep(yield_to);

	  // loop around and check both queues again
	  continue;
	}

	{
	  // no ready or resumable tasks?  thumb twiddling time

	  // are we shutting down?
	  if(shutdown_flag.load()) {
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
	      assert(to_wake != Thread::self());
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
		// no, stay awake but suspend until there's a chance that the next iteration
		//  of this loop would turn out different
		wait_for_work(old_work_counter);
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
	      // no, stay awake but suspend until there's a chance that the next iteration
	      //  of this loop would turn out different
	      wait_for_work(old_work_counter);
	    }
	  }
	}
      }
    }
  }

  // an entry point that takes the scheduler lock explicitly
  void ThreadedTaskScheduler::scheduler_loop_wlock(void)
  {
    AutoLock<FIFOMutex> al(lock);
    scheduler_loop();
  }

  void ThreadedTaskScheduler::wait_for_work(uint64_t old_work_counter)
  {
    // clear the interrupt flag (if we use it) before we check the work counter
    //  to avoid a missed interrupt
    if(max_bgwork_timeslice > 0)
      bgworker_interrupt.store(false);

    // try a check without letting go of our lock first
    if(work_counter.check_for_work(old_work_counter))
      return;

    // drop our scheduler lock while we wait
    lock.unlock();

    if(max_bgwork_timeslice > 0) {
      // try to be productive while we're waiting
      bgworker.do_work(max_bgwork_timeslice, &bgworker_interrupt);
    } else {
      // just let the work counter wake us up when there's stuff to do
      work_counter.wait_for_work(old_work_counter);
    }

    lock.lock();
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

  void  KernelThreadTaskScheduler::remove_task_queue(TaskQueue *queue)
  {
    // call the parent implementation first
    ThreadedTaskScheduler::remove_task_queue(queue);
  }

  void KernelThreadTaskScheduler::start(void)
  {
    // fire up the minimum number of workers
    {
      AutoLock<FIFOMutex> al(lock);

      update_worker_count(cfg_min_active_workers, cfg_min_active_workers);

      for(int i = 0; i < cfg_min_active_workers; i++)
	worker_create(true);
    }
  }

  void KernelThreadTaskScheduler::shutdown(void)
  {
    log_sched.info() << "scheduler shutdown requested: sched=" << this;
    shutdown_flag.store(true);
    // setting the shutdown flag adds "work" to the system
    work_counter.increment_counter();

    // wait for all workers to finish
    {
      AutoLock<FIFOMutex> al(lock);

      while(!all_workers.empty() || !terminating_workers.empty())
	shutdown_condvar.wait();
    }

    log_sched.info() << "scheduler shutdown complete: sched=" << this;
  }

  void KernelThreadTaskScheduler::thread_starting(Thread *thread)
  {
    log_sched.info() << "scheduler worker started: sched=" << this << " worker=" << thread;

    // see if we're supposed to be active yet
    {
      AutoLock<FIFOMutex> al(lock);

      if(active_workers.count(thread) == 0) {
	// nope, sleep on a CV until we are
        FIFOMutex::CondVar my_cv(lock);
	sleeping_threads[thread] = &my_cv;

	while(active_workers.count(thread) == 0)
	  my_cv.wait();
    
	sleeping_threads.erase(thread);
      }
    }
  }

  void KernelThreadTaskScheduler::thread_terminating(Thread *thread)
  {
    log_sched.info() << "scheduler worker terminating: sched=" << this << " worker=" << thread;

    AutoLock<FIFOMutex> al(lock);

    // if the thread is still in our all_workers list, this was unexpected
    if(all_workers.count(thread) > 0) {
      printf("unexpected worker termination: %p\n", thread);

      // if this was our last worker, and we're not shutting down,
      //  something bad probably happened - fire up a new worker and
      //  hope things work themselves out
      if((all_workers.size() == 1) && !shutdown_flag.load()) {
	printf("HELP!  Lost last worker for proc " IDFMT "!", proc.id);
	worker_terminate(worker_create(false));
      } else {
	// just let it die
	worker_terminate(0);
      }
    }

    // detach and delete the worker thread - better be expected now
    assert(terminating_workers.count(thread) > 0);
    terminating_workers.erase(thread);
    thread->detach();
    delete thread;

    // if this was the last thread, we'd better be in shutdown...
    if(all_workers.empty() && terminating_workers.empty()) {
      assert(shutdown_flag.load());
      shutdown_condvar.signal();
    }
  }

  bool KernelThreadTaskScheduler::execute_task(Task *task)
  {
    task->execute_on_processor(proc);
    return true;
  }

  void KernelThreadTaskScheduler::execute_internal_task(InternalTask *task)
  {
    task->execute_on_processor(proc);
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
#ifndef NDEBUG
    size_t count =
#endif
      active_workers.erase(Thread::self());
    assert(count == 1);

    FIFOMutex::CondVar my_cv(lock);
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
    std::map<Thread *, FIFOMutex::CondVar *>::const_iterator it = sleeping_threads.find(to_wake);
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
#ifndef NDEBUG
    size_t count =
#endif
      active_workers.erase(me);
    assert(count == 1);

    // also off the all workers list
    all_workers.erase(me);
    terminating_workers.insert(me);

    // and wake up whoever we're switching to (if any)
    if(switch_to)
      worker_wake(switch_to);
  }
  
  void KernelThreadTaskScheduler::wait_for_work(uint64_t old_work_counter)
  {
    // if we have a dedicated core and we don't care about power, we can spin-wait here
    bool spin_wait = false;
    if(spin_wait) {
      while(!work_counter.check_for_work(old_work_counter)) {
	// don't sleep, but let go of lock and other threads run
	lock.unlock();
	Thread::yield();
	lock.lock();
      }
      return;
    }

    // otherwise fall back to the base (sleeping) implementation
    ThreadedTaskScheduler::wait_for_work(old_work_counter);
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
    , host_startup_condvar(lock)
    , cfg_num_host_threads(1)
  {
  }

  UserThreadTaskScheduler::~UserThreadTaskScheduler(void)
  {
    // cleanup should happen before destruction
    assert(all_workers.empty());
    assert(all_hosts.empty());
    assert(active_worker_count == 0);
  }

  void UserThreadTaskScheduler::add_task_queue(TaskQueue *queue)
  {
    // call the parent implementation first
    ThreadedTaskScheduler::add_task_queue(queue);
  }

  void UserThreadTaskScheduler::remove_task_queue(TaskQueue *queue)
  {
    // call the parent implementation first
    ThreadedTaskScheduler::remove_task_queue(queue);
  }

  void UserThreadTaskScheduler::start(void)
  {
    // with user threading, active must always match the number of host threads
    cfg_min_active_workers = cfg_num_host_threads;
    cfg_max_active_workers = cfg_num_host_threads;

    // fire up the host threads (which will fire up initial workers)
    {
      AutoLock<FIFOMutex> al(lock);

      update_worker_count(cfg_num_host_threads, cfg_num_host_threads);

      ThreadLaunchParameters tlp;
      tlp.set_stack_size(32768);  // really small stack is fine here (4KB is too small)

      host_startups_remaining = cfg_num_host_threads;

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
    log_sched.info() << "scheduler shutdown requested: sched=" << this;
    // set the shutdown flag and wait for all the host threads to exit
    AutoLock<FIFOMutex> al(lock);

    // make sure everybody actually started before we tell them to shut down
    while(host_startups_remaining > 0) {
      printf("wait\n");
      host_startup_condvar.wait();
    }

    shutdown_flag.store(true);
    // setting the shutdown flag adds "work" to the system
    work_counter.increment_counter();

    while(!all_hosts.empty()) {
      // pick an arbitrary host and join on it
      Thread *t = *all_hosts.begin();
      al.release();
      t->join();  // can't hold lock while waiting
      al.reacquire();
      all_hosts.erase(t);
      delete t;
    }
    log_sched.info() << "scheduler shutdown complete: sched=" << this;
  }

  namespace ThreadLocal {
    // you can't delete a user thread until you've switched off of it, so
    //  use TLS to mark when that should happen
    static REALM_THREAD_LOCAL Thread *terminated_user_thread = 0;
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
    log_sched.debug() << "host thread started: sched=" << this << " thread=" << Thread::self();
    AutoLock<FIFOMutex> al(lock);

    // create a user worker thread - it won't start right away
    Thread *worker = worker_create(false);

    // now signal that we've started
    int left = --host_startups_remaining;
    if(left == 0)
      host_startup_condvar.broadcast();

    while(true) {
      // for user ctx switching, lock is HELD during thread switches
      Thread::user_switch(worker);
      do_user_thread_cleanup();

      if(shutdown_flag.load())
	break;

      // getting here is unexpected
      printf("HELP!  Lost a user worker thread - making a new one...\n");
      update_worker_count(+1, +1);
      worker = worker_create(false);
    }
    log_sched.debug() << "host thread finished: sched=" << this << " thread=" << Thread::self();
  }

  void UserThreadTaskScheduler::thread_starting(Thread *thread)
  {
    // nothing to do here
    log_sched.info() << "scheduler worker created: sched=" << this << " worker=" << thread;
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

  void UserThreadTaskScheduler::execute_internal_task(InternalTask *task)
  {
    task->execute_on_processor(proc);
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
										   &core_rsrv,
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
    assert((switch_to != 0) || shutdown_flag.load());

#ifdef DEBUG_THREAD_SCHEDULER
    printf("terminate: %p -> %p\n", Thread::self(), switch_to);
#endif

    // take ourself off the active and all worker lists
    //size_t count = active_workers.erase(Thread::self());
    //assert(count == 1);

#ifndef NDEBUG
    size_t count =
#endif
      all_workers.erase(Thread::self());
    assert(count == 1);

    // whoever we switch to should delete us
    Thread *t = Thread::self();
    request_user_thread_cleanup(Thread::self());
    update_thread_state(t, Thread::STATE_DELETED);

    if(switch_to != 0)
      assert(switch_to->get_state() != Thread::STATE_DELETED);

    Thread::user_switch(switch_to);

    // we don't expect to ever get control back
    assert(0);
  }

  void UserThreadTaskScheduler::wait_for_work(uint64_t old_work_counter)
  {
    // if we have a dedicated core and we don't care about power, we can spin-wait here
    bool spin_wait = false;
    if(spin_wait) {
      while(!work_counter.check_for_work(old_work_counter)) {
	// don't sleep, but let go of lock and other threads run
	lock.unlock();
	Thread::yield();
	lock.lock();
      }
      return;
    }

    // otherwise fall back to the base (sleeping) implementation
    ThreadedTaskScheduler::wait_for_work(old_work_counter);
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class Task
  //

}; // namespace Realm
