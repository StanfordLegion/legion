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

#include "proc_impl.h"

#include "timers.h"
#include "runtime_impl.h"
#include "logging.h"
#include "serialize.h"

GASNETT_THREADKEY_DEFINE(cur_preemptable_thread);

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

namespace Realm {

  Logger log_task("task");
  Logger log_util("util");


  ////////////////////////////////////////////////////////////////////////
  //
  // class Processor
  //

    /*static*/ const Processor Processor::NO_PROC = { 0 }; 

    /*static*/ Processor Processor::get_executing_processor(void) 
    { 
      void *tls_val = gasnett_threadkey_get(cur_preemptable_thread);
      if (tls_val != NULL)
      {
        PreemptableThread *me = (PreemptableThread *)tls_val;
        return me->get_processor();
      }
      // Otherwise this better be a GPU processor 
#ifdef USE_CUDA
      return GPUProcessor::get_processor();
#else
      assert(0);
#endif
    }

    Processor::Kind Processor::kind(void) const
    {
      return get_runtime()->get_processor_impl(*this)->kind;
    }

    /*static*/ Processor Processor::create_group(const std::vector<Processor>& members)
    {
      // are we creating a local group?
      if((members.size() == 0) || (ID(members[0]).node() == gasnet_mynode())) {
	ProcessorGroup *grp = get_runtime()->local_proc_group_free_list->alloc_entry();
	grp->set_group_members(members);
#ifdef EVENT_GRAPH_TRACE
        {
          const int base_size = 1024;
          char base_buffer[base_size];
          char *buffer;
          int buffer_size = (members.size() * 20);
          if (buffer_size >= base_size)
            buffer = (char*)malloc(buffer_size+1);
          else
            buffer = base_buffer;
          buffer[0] = '\0';
          int offset = 0;
          for (std::vector<Processor>::const_iterator it = members.begin();
                it != members.end(); it++)
          {
            int written = snprintf(buffer+offset,buffer_size-offset,
                                   " " IDFMT, it->id);
            assert(written < (buffer_size-offset));
            offset += written;
          }
          log_event_graph.info("Group: " IDFMT " %ld%s",
                                grp->me.id, members.size(), buffer); 
          if (buffer_size >= base_size)
            free(buffer);
        }
#endif
	return grp->me;
      }

      assert(0);
    }

    void Processor::get_group_members(std::vector<Processor>& members)
    {
      // if we're a plain old processor, the only member of our "group" is ourself
      if(ID(*this).type() == ID::ID_PROCESSOR) {
	members.push_back(*this);
	return;
      }

      assert(ID(*this).type() == ID::ID_PROCGROUP);

      ProcessorGroup *grp = get_runtime()->get_procgroup_impl(*this);
      grp->get_group_members(members);
    }

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
			   //std::set<RegionInstance> instances_needed,
			   Event wait_on, int priority) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ProcessorImpl *p = get_runtime()->get_processor_impl(*this);

      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      Event e = finish_event->current_event();
#ifdef EVENT_GRAPH_TRACE
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.info("Task Request: %d " IDFMT 
                            " (" IDFMT ",%d) (" IDFMT ",%d)"
                            " (" IDFMT ",%d) %d %p %ld",
                            func_id, id, e.id, e.gen,
                            wait_on.id, wait_on.gen,
                            enclosing.id, enclosing.gen,
                            priority, args, arglen);
#endif

      p->spawn_task(func_id, args, arglen, //instances_needed, 
		    wait_on, e, priority);
      return e;
    }

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
                           const ProfilingRequestSet &reqs,
			   Event wait_on, int priority) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ProcessorImpl *p = get_runtime()->get_processor_impl(*this);

      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      Event e = finish_event->current_event();
#ifdef EVENT_GRAPH_TRACE
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.info("Task Request: %d " IDFMT 
                            " (" IDFMT ",%d) (" IDFMT ",%d)"
                            " (" IDFMT ",%d) %d %p %ld",
                            func_id, id, e.id, e.gen,
                            wait_on.id, wait_on.gen,
                            enclosing.id, enclosing.gen,
                            priority, args, arglen);
#endif

      p->spawn_task(func_id, args, arglen, reqs,
		    wait_on, e, priority);
      return e;
    }

    AddressSpace Processor::address_space(void) const
    {
      return ID(id).node();
    }

    ID::IDType Processor::local_id(void) const
    {
      return ID(id).index();
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorImpl
  //

    ProcessorImpl::ProcessorImpl(Processor _me, Processor::Kind _kind)
      : me(_me), kind(_kind), run_counter(0)
    {
    }

    ProcessorImpl::~ProcessorImpl(void)
    {
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorGroup
  //

    ProcessorGroup::ProcessorGroup(void)
      : ProcessorImpl(Processor::NO_PROC, Processor::PROC_GROUP),
	members_valid(false), members_requested(false), next_free(0)
    {
    }

    ProcessorGroup::~ProcessorGroup(void)
    {
    }

    void ProcessorGroup::init(Processor _me, int _owner)
    {
      assert(ID(_me).node() == (unsigned)_owner);

      me = _me;
      lock.init(ID(me).convert<Reservation>(), ID(me).node());
    }

    void ProcessorGroup::set_group_members(const std::vector<Processor>& member_list)
    {
      // can only be perform on owner node
      assert(ID(me).node() == gasnet_mynode());
      
      // can only be done once
      assert(!members_valid);

      for(std::vector<Processor>::const_iterator it = member_list.begin();
	  it != member_list.end();
	  it++) {
	ProcessorImpl *m_impl = get_runtime()->get_processor_impl(*it);
	members.push_back(m_impl);
      }

      members_requested = true;
      members_valid = true;
    }

    void ProcessorGroup::get_group_members(std::vector<Processor>& member_list)
    {
      assert(members_valid);

      for(std::vector<ProcessorImpl *>::const_iterator it = members.begin();
	  it != members.end();
	  it++)
	member_list.push_back((*it)->me);
    }

    void ProcessorGroup::start_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::shutdown_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::initialize_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::finalize_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::enqueue_task(Task *task)
    {
      for (std::vector<ProcessorImpl *>::const_iterator it = members.begin();
            it != members.end(); it++)
      {
        (*it)->enqueue_task(task);
      }
    }

    /*virtual*/ void ProcessorGroup::spawn_task(Processor::TaskFuncID func_id,
						const void *args, size_t arglen,
						//std::set<RegionInstance> instances_needed,
						Event start_event, Event finish_event,
						int priority)
    {
      // create a task object and insert it into the queue
      Task *task = new Task(me, func_id, args, arglen, 
                            finish_event, priority, members.size());

      if (start_event.has_triggered())
        enqueue_task(task);
      else
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }

    /*virtual*/ void ProcessorGroup::spawn_task(Processor::TaskFuncID func_id,
						const void *args, size_t arglen,
                                                const ProfilingRequestSet &reqs,
						Event start_event, Event finish_event,
						int priority)
    {
      // create a task object and insert it into the queue
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            finish_event, priority, members.size());

      if (start_event.has_triggered())
        enqueue_task(task);
      else
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }


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


  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredTaskSpawn
  //

      bool DeferredTaskSpawn::event_triggered(void)
    {
      log_task.debug("deferred task now ready: func=%d finish=" IDFMT "/%d",
                 task->func_id, 
                 task->finish_event.id, task->finish_event.gen);
      proc->enqueue_task(task);
      return true;
    }

    void DeferredTaskSpawn::print_info(FILE *f)
    {
      fprintf(f,"deferred task: func=%d proc=" IDFMT " finish=" IDFMT "/%d\n",
             task->func_id, task->proc.id, task->finish_event.id, task->finish_event.gen);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SpawnTaskMessage
  //

  /*static*/ void SpawnTaskMessage::handle_request(RequestArgs args,
						   const void *data,
						   size_t datalen)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    ProcessorImpl *p = get_runtime()->get_processor_impl(args.proc);
    log_task.debug("remote spawn request: proc_id=" IDFMT " task_id=%d event=" IDFMT "/%d",
		   args.proc.id, args.func_id, args.start_id, args.start_gen);
    Event start_event, finish_event;
    start_event.id = args.start_id;
    start_event.gen = args.start_gen;
    finish_event.id = args.finish_id;
    finish_event.gen = args.finish_gen;

    Serialization::FixedBufferDeserializer fbd(data, datalen);
    fbd.extract_bytes(0, args.user_arglen);  // skip over task args - we'll access those directly

    // profiling requests are optional - extract only if there's data
    ProfilingRequestSet prs;
    if(fbd.bytes_left() > 0)
      fbd >> prs;
      
    p->spawn_task(args.func_id, data, args.user_arglen, prs,
		  start_event, finish_event, args.priority);
  }

  /*static*/ void SpawnTaskMessage::send_request(gasnet_node_t target, Processor proc,
						 Processor::TaskFuncID func_id,
						 const void *args, size_t arglen,
						 const ProfilingRequestSet *prs,
						 Event start_event, Event finish_event,
						 int priority)
  {
    RequestArgs r_args;

    r_args.proc = proc;
    r_args.func_id = func_id;
    r_args.start_id = start_event.id;
    r_args.start_gen = start_event.gen;
    r_args.finish_id = finish_event.id;
    r_args.finish_gen = finish_event.gen;
    r_args.priority = priority;
    r_args.user_arglen = arglen;
    
    if(!prs) {
      // no profiling, so task args are the only payload
      Message::request(target, r_args, args, arglen, PAYLOAD_COPY);
    } else {
      // need to serialize both the task args and the profiling request
      //  into a single payload
      Serialization::DynamicBufferSerializer dbs(arglen + 4096);  // assume profiling requests are < 4K

      dbs.append_bytes(args, arglen);
      dbs << *prs;

      size_t datalen = dbs.bytes_used();
      void *data = dbs.detach_buffer(-1);  // don't trim - this buffer has a short life
      Message::request(target, r_args, data, datalen, PAYLOAD_FREE);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteProcessor
  //

    RemoteProcessor::RemoteProcessor(Processor _me, Processor::Kind _kind)
      : ProcessorImpl(_me, _kind)
    {
    }

    RemoteProcessor::~RemoteProcessor(void)
    {
    }

    void RemoteProcessor::start_processor(void)
    {
      assert(0);
    }

    void RemoteProcessor::shutdown_processor(void)
    {
      assert(0);
    }

    void RemoteProcessor::initialize_processor(void)
    {
      assert(0);
    }

    void RemoteProcessor::finalize_processor(void)
    {
      assert(0);
    }

    void RemoteProcessor::enqueue_task(Task *task)
    {
      // should never be called
      assert(0);
    }

    void RemoteProcessor::tasks_available(int priority)
    {
      log_task.warning("remote processor " IDFMT " being told about local tasks ready?",
		       me.id);
    }

    void RemoteProcessor::spawn_task(Processor::TaskFuncID func_id,
				     const void *args, size_t arglen,
				     //std::set<RegionInstance> instances_needed,
				     Event start_event, Event finish_event,
				     int priority)
    {
      log_task.debug("spawning remote task: proc=" IDFMT " task=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
		     me.id, func_id, 
		     start_event.id, start_event.gen,
		     finish_event.id, finish_event.gen);

      SpawnTaskMessage::send_request(ID(me).node(), me, func_id,
				     args, arglen, 0 /* no profiling requests */,
				     start_event, finish_event, priority);
    }

    void RemoteProcessor::spawn_task(Processor::TaskFuncID func_id,
				     const void *args, size_t arglen,
				     const Realm::ProfilingRequestSet &reqs,
				     Event start_event, Event finish_event,
				     int priority)
    {
      log_task.debug("spawning remote task: proc=" IDFMT " task=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
		     me.id, func_id, 
		     start_event.id, start_event.gen,
		     finish_event.id, finish_event.gen);

      SpawnTaskMessage::send_request(ID(me).node(), me, func_id,
				     args, arglen, &reqs,
				     start_event, finish_event, priority);
    }

  
  ///////////////////////////////////////////////////////////////////////////
    //
    // SPAGHETTI CODE BELOW THIS POINT
    
    LocalThread::LocalThread(LocalProcessor *p)
      : PreemptableThread(), proc(p), state(RUNNING_STATE),
	thread_cond(thread_mutex),
        initialize(false), finalize(false)
    {
    }

    LocalThread::~LocalThread(void)
    {
    }

    Processor LocalThread::get_processor(void) const
    {
      return proc->me;
    }

    void LocalThread::thread_main(void)
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

    void LocalThread::sleep_on_event(Event wait_for)
    {
#ifdef EVENT_GRAPH_TRACE
      unsigned long long start = TimeStamp::get_current_time_in_micros(); 
#endif
      assert(state == RUNNING_STATE);
      // First mark that we are this thread is now paused
      state = PAUSED_STATE;
      // Then tell the processor to pause the thread
      proc->pause_thread(this);
      // Now register ourselves with the event
      EventImpl::add_waiter(wait_for, this);
      // Take our lock and see if we are still in the paused state
      // It's possible that we've already been woken up so check before
      // going to sleep
      thread_mutex.lock();
      // If we are in the paused state or the resumable state then we actually
      // do need to go to sleep so we can be woken up by the processor later
      if ((state == PAUSED_STATE) || (state == RESUMABLE_STATE))
      {
	thread_cond.wait();
      }
      assert(state == RUNNING_STATE);
      thread_mutex.unlock();
#ifdef EVENT_GRAPH_TRACE
      unsigned long long stop = TimeStamp::get_current_time_in_micros();
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.debug("Task Wait: (" IDFMT ",%d) (" IDFMT ",%d) %lld",
                            enclosing.id, enclosing.gen,
                            wait_for.id, wait_for.gen, (stop - start));
#endif
    }

    bool LocalThread::event_triggered(void)
    {
      thread_mutex.lock();
      assert(state == PAUSED_STATE);
      state = RESUMABLE_STATE;
      thread_mutex.unlock();
      // Now tell the processor that this thread is resumable
      proc->resume_thread(this);
      return false;
    }

    void LocalThread::print_info(FILE *f)
    {
      fprintf(f, "Waiting thread %lx\n", (unsigned long)thread); 
    }

    void LocalThread::awake(void)
    {
      thread_mutex.lock();
      assert((state == SLEEPING_STATE) || (state == SLEEP_STATE));
      // Only need to signal if the thread is actually asleep
      if (state == SLEEP_STATE)
	thread_cond.signal();
      state = RUNNING_STATE;
      thread_mutex.unlock();
    }

    void LocalThread::sleep(void)
    {
      thread_mutex.lock();
      assert((state == SLEEPING_STATE) || (state == RUNNING_STATE));
      // If we haven't been told to stay awake, then go to sleep
      if (state == SLEEPING_STATE) {
        state = SLEEP_STATE;
	thread_cond.wait();
      }
      assert(state == RUNNING_STATE);
      thread_mutex.unlock();
    }

    void LocalThread::prepare_to_sleep(void)
    {
      // Don't need the lock since we are running
      assert(state == RUNNING_STATE);
      state = SLEEPING_STATE;
    }

    void LocalThread::resume(void)
    {
      thread_mutex.lock();
      assert(state == RESUMABLE_STATE);
      state = RUNNING_STATE;
      thread_cond.signal();
      thread_mutex.unlock();
    }

    void LocalThread::shutdown(void)
    {
      // wake up the thread
      thread_mutex.lock();
      state = RUNNING_STATE;
      thread_cond.signal();
      thread_mutex.unlock();
      // Now wait to join with the thread
      void *result;
      pthread_join(thread, &result);
    }

    LocalProcessor::LocalProcessor(Processor _me, Processor::Kind _kind, 
                                   size_t stacksize, const char *name, int _core)
      : ProcessorImpl(_me, _kind), core_id(_core), 
        stack_size(stacksize), processor_name(name),
	condvar(mutex), done_initialization(false),
        shutdown(false), shutdown_trigger(false), running_thread(0)
    {
    }

    LocalProcessor::~LocalProcessor(void)
    {
    }

    void LocalProcessor::start_processor(void)
    {
      assert(running_thread == 0);
      running_thread = create_new_thread();
      running_thread->do_initialize();
      running_thread->start_thread(stack_size, core_id, processor_name);
      mutex.lock();
      if(!done_initialization)
	condvar.wait();
      mutex.unlock();
    }

    void LocalProcessor::shutdown_processor(void)
    {
      // First check to make sure that we received the kill
      // pill. If we didn't then wait for it. This is how
      // we distinguish deadlock from just normal termination
      // from all the processors being idle
      std::vector<LocalThread*> to_shutdown;
      mutex.lock();
      if (!shutdown_trigger)
	condvar.wait();
      assert(shutdown_trigger);
      shutdown = true;
      to_shutdown = available_threads;
      if (running_thread)
        to_shutdown.push_back(running_thread);
      assert(resumable_threads.empty());
      assert(paused_threads.empty());
      mutex.unlock();
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

    void LocalProcessor::initialize_processor(void)
    {
      mutex.lock();
      done_initialization = true;
      condvar.signal();
      mutex.unlock();
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_INIT);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor init task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor init task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor init task: proc=" IDFMT "", me.id);
      }
    }

    void LocalProcessor::finalize_processor(void)
    {
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor shutdown task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor shutdown task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor shutdown task: proc=" IDFMT "", me.id);
      }
    }

    LocalThread* LocalProcessor::create_new_thread(void)
    {
      return new LocalThread(this);
    }

    bool LocalProcessor::execute_task(LocalThread *thread)
    {
      mutex.lock();
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
        LocalThread *to_resume = resumable_threads.front();
        resumable_threads.pop_front();
        // Make this the running thread
        running_thread = to_resume;
        // Release the lock
	mutex.unlock();
        // Wake up the resumable thread
        to_resume->resume();
        // Put ourselves to sleep
        thread->sleep();
      }
      else if (task_queue.empty())
      {
        // If there are no tasks to run, then we should go to sleep
        thread->prepare_to_sleep();
        available_threads.push_back(thread);
        running_thread = NULL;
	mutex.unlock();
        thread->sleep();
      }
      else
      {
        // Pull a task off the queue and execute it
        Task *task = task_queue.pop();
        if (task->func_id == 0) {
          // This is the kill pill so we need to handle it special
          finished();
          // Mark that we received the shutdown trigger
          shutdown_trigger = true;
	  condvar.signal();
	  mutex.unlock();
          // Trigger the completion task
          if (__sync_fetch_and_add(&(task->run_count),1) == 0)
            get_runtime()->get_genevent_impl(task->finish_event)->
                          trigger(task->finish_event.gen, gasnet_mynode());
          // Delete the task
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        } else {
	  mutex.unlock();
          // Common case: just run the task
          if (__sync_fetch_and_add(&task->run_count,1) == 0)
            thread->run_task(task, me);
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        }
      }
      // This value is monotonic so once it becomes true, then we should exit
      return shutdown;
    }

    void LocalProcessor::pause_thread(LocalThread *thread)
    {
      LocalThread *to_wake = 0;
      LocalThread *to_start = 0;
      LocalThread *to_resume = 0;
      mutex.lock();
      assert(running_thread == thread);
      // Put this on the list of paused threads
      paused_threads.insert(thread);
      // Now see if we have other work to do
      if (!resumable_threads.empty()) {
        to_resume = resumable_threads.front();
        resumable_threads.pop_front();
        running_thread = to_resume;
      } else if (!task_queue.empty()) {
        // Note we might need to make a new thread here
        if (!available_threads.empty()) {
          to_wake = available_threads.back();
          available_threads.pop_back();
          running_thread = to_wake;
        } else {
          // Make a new thread to run
          to_start = create_new_thread();
          running_thread = to_start;
        }
      } else {
        // Nothing else to do, so mark that no one is running
        running_thread = 0;
      }
      mutex.unlock();
      // Wake up any threads while not holding the lock
      if (to_wake)
        to_wake->awake();
      if (to_start)
        to_start->start_thread(stack_size, core_id, processor_name);
      if (to_resume)
        to_resume->resume();
    }

    void LocalProcessor::resume_thread(LocalThread *thread)
    {
      bool resume_now = false;
      mutex.lock();
      std::set<LocalThread*>::iterator finder = 
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
      mutex.unlock();
      if (resume_now)
        thread->resume();
    }

    void LocalProcessor::enqueue_task(Task *task)
    {
      // Mark this task as ready
      task->mark_ready();
      LocalThread *to_wake = 0;
      LocalThread *to_start = 0;
      mutex.lock();
      task_queue.insert(task, task->priority);
      // Figure out if we need to wake someone up
      if (running_thread == NULL) {
        if (!available_threads.empty()) {
          to_wake = available_threads.back();
          available_threads.pop_back();
          running_thread = to_wake;
        } else {
          to_start = create_new_thread(); 
          running_thread = to_start;
        }
      }
      mutex.unlock();
      if (to_wake)
        to_wake->awake();
      if (to_start)
        to_start->start_thread(stack_size, core_id, processor_name);
    }

    void LocalProcessor::spawn_task(Processor::TaskFuncID func_id,
                                    const void *args, size_t arglen,
                                    Event start_event, Event finish_event,
                                    int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, finish_event, 
                            priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    void LocalProcessor::spawn_task(Processor::TaskFuncID func_id,
                                    const void *args, size_t arglen,
                                    const Realm::ProfilingRequestSet &reqs,
                                    Event start_event, Event finish_event,
                                    int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            finish_event, priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    ProcessorAssignment *proc_assignment = 0;

    void PreemptableThread::start_thread(size_t stack_size, int core_id, 
                                         const char *debug_name)
    {
      pthread_attr_t attr;
      CHECK_PTHREAD( pthread_attr_init(&attr) );
      CHECK_PTHREAD( pthread_attr_setstacksize(&attr,stack_size) );
      if(proc_assignment)
	proc_assignment->bind_thread(core_id, &attr, debug_name);
      CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_entry, (void *)this) );
      CHECK_PTHREAD( pthread_attr_destroy(&attr) );
#ifdef DEADLOCK_TRACE
      get_runtime()->add_thread(&thread);
#endif
    }

    void PreemptableThread::run_task(Task *task, Processor actual_proc /*=NO_PROC*/)
    {
      Processor::TaskFuncPtr fptr = get_runtime()->task_table[task->func_id];
#if 0
      char argstr[100];
      argstr[0] = 0;
      for(size_t i = 0; (i < task->arglen) && (i < 40); i++)
        sprintf(argstr+2*i, "%02x", ((unsigned char *)(task->args))[i]);
      if(task->arglen > 40) strcpy(argstr+80, "...");
      log_util(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
               "utility task start: %d (%p) (%s)", task->func_id, fptr, argstr);
#endif
#ifdef EVENT_GRAPH_TRACE
      start_enclosing(task->finish_event);
      unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif
      log_task.info("thread running ready task %p for proc " IDFMT "",
                              task, task->proc.id);
      task->mark_started();
      (*fptr)(task->args, task->arglen, 
              (actual_proc.exists() ? actual_proc : task->proc));
      task->mark_completed();
      // Capture the actual processor if necessary
      if (task->capture_proc && actual_proc.exists())
        task->proc = actual_proc;
      log_task.info("thread finished running task %p for proc " IDFMT "",
                              task, task->proc.id);
#ifdef EVENT_GRAPH_TRACE
      unsigned long long stop = TimeStamp::get_current_time_in_micros();
      finish_enclosing();
      log_event_graph.debug("Task Time: (" IDFMT ",%d) %lld",
                            task->finish_event.id, task->finish_event.gen,
                            (stop - start));
#endif
#if 0
      log_util(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
               "utility task end: %d (%p) (%s)", task->func_id, fptr, argstr);
#endif
      if(task->finish_event.exists())
        get_runtime()->get_genevent_impl(task->finish_event)->
                        trigger(task->finish_event.gen, gasnet_mynode());
    }

    /*static*/ bool PreemptableThread::preemptable_sleep(Event wait_for)
    {
      // check TLS to see if we're really a preemptable thread
      void *tls_val = gasnett_threadkey_get(cur_preemptable_thread);
      if(!tls_val) return false;

      PreemptableThread *me = (PreemptableThread *)tls_val;

      me->sleep_on_event(wait_for);
      return true;
    }
    
    /*static*/ void *PreemptableThread::thread_entry(void *data)
    {
      PreemptableThread *me = (PreemptableThread *)data;

      // set up TLS variable so we can remember who we are way down the call
      //  stack
      gasnett_threadkey_set(cur_preemptable_thread, me);

      // Initialize this value to NULL, it will get filled in the first time it is used
      CHECK_PTHREAD( pthread_setspecific(thread_timer_key, NULL) );

      // and then just call the virtual thread_main
      me->thread_main();

      return 0;
    }
 
    GreenletTask::GreenletTask(Task *t, GreenletProcessor *p,
                               void *s, long *ssize)
      : greenlet(NULL, s, ssize), task(t), proc(p)
    {
    }

    GreenletTask::~GreenletTask(void)
    {
      // Make sure we are dead
      assert(isdead());
      // Remove our reference on our task
      if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
        delete task;
    }

    bool GreenletTask::event_triggered(void)
    {
      // Tell the processor we're awake
      proc->unpause_task(this);
      // Don't delete
      return false;
    }

    void GreenletTask::print_info(FILE *f)
    {
      fprintf(f,"Waiting greenlet %p of processor %s\n",
              this, proc->processor_name);
    }

    void* GreenletTask::run(void *arg)
    {
      GreenletThread *thread = static_cast<GreenletThread*>(arg);
      thread->run_task(task, proc->me);
      proc->complete_greenlet(this);
      return NULL;
    }

    GreenletThread::GreenletThread(GreenletProcessor *p)
      : proc(p)
    {
      current_task = NULL;
    }

    GreenletThread::~GreenletThread(void)
    {
    }

    Processor GreenletThread::get_processor(void) const
    {
      return proc->me;
    }

    void GreenletThread::thread_main(void)
    {
      greenlet::init_greenlet_thread();
      proc->initialize_processor();
      while (true)
      {
        bool quit = proc->execute_task();
        if (quit) break;
      }
      proc->finalize_processor();
    }

    void GreenletThread::sleep_on_event(Event wait_for)
    {
      assert(current_task != NULL);
      // Register ourselves as the waiter
      EventImpl::add_waiter(wait_for, current_task);
      GreenletTask *paused_task = current_task;
      // Tell the processor to pause us
      proc->pause_task(paused_task);
      // When we return the event has triggered
      assert(paused_task == current_task);
    }

    void GreenletThread::start_task(GreenletTask *task)
    {
      current_task = task;
      task->switch_to(this);
    }

    void GreenletThread::resume_task(GreenletTask *task)
    {
      current_task = task;
      task->switch_to(this);
    }

    void GreenletThread::return_to_root(void)
    {
      current_task = NULL;
      greenlet *root = greenlet::root();
      root->switch_to(NULL);
    }

    void GreenletThread::wait_for_shutdown(void)
    {
      void *result;
      pthread_join(thread, &result);
    }

    GreenletProcessor::GreenletProcessor(Processor _me, Processor::Kind _kind,
                                         size_t _stack_size, int init_stack_size,
                                         const char *name, int _core_id)
      : ProcessorImpl(_me, _kind), core_id(_core_id), proc_stack_size(_stack_size), 
        processor_name(name),
	condvar(mutex), done_initialization(false),
	shutdown(false), shutdown_trigger(false), 
        greenlet_thread(0), thread_state(GREENLET_RUNNING)
    {
    }

    GreenletProcessor::~GreenletProcessor(void)
    {
    }

    void GreenletProcessor::start_processor(void)
    {
      assert(greenlet_thread == 0);
      greenlet_thread = new GreenletThread(this);
      greenlet_thread->start_thread(proc_stack_size, core_id, processor_name);
      mutex.lock();
      if(!done_initialization)
	condvar.wait();
      mutex.unlock();
    }

    void GreenletProcessor::shutdown_processor(void)
    {
      mutex.lock();
      if (!shutdown_trigger)
	condvar.wait();
      assert(shutdown_trigger);
      shutdown = true;
      // Signal our thread in case it is asleep
      condvar.signal();
      mutex.unlock();
      greenlet_thread->wait_for_shutdown();
    }

    void GreenletProcessor::initialize_processor(void)
    {
      mutex.lock();
      done_initialization = true;
      condvar.signal();
      mutex.unlock();
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_INIT);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor init task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor init task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor init task: proc=" IDFMT "", me.id);
      }
    }

    void GreenletProcessor::finalize_processor(void)
    {
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor shutdown task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor shutdown task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor shutdown task: proc=" IDFMT "", me.id);
      }
    }

    void GreenletProcessor::enqueue_task(Task *task)
    {
      // Mark this task as ready
      task->mark_ready();
      mutex.lock();
      task_queue.insert(task, task->priority); 
      // Wake someone up if we aren't running
      if (thread_state == GREENLET_IDLE)
      {
        thread_state = GREENLET_RUNNING;
	condvar.signal();
      }
      mutex.unlock();
    }

    void GreenletProcessor::spawn_task(Processor::TaskFuncID func_id,
                                       const void *args, size_t arglen,
                                       Event start_event, Event finish_event,
                                       int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, finish_event, 
                            priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    void GreenletProcessor::spawn_task(Processor::TaskFuncID func_id,
                                       const void *args, size_t arglen,
                                       const Realm::ProfilingRequestSet &reqs,
                                       Event start_event, Event finish_event,
                                       int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            finish_event, priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    bool GreenletProcessor::execute_task(void)
    {
      mutex.lock();
      // We should be running
      assert(thread_state == GREENLET_RUNNING);
      if (!resumable_tasks.empty())
      {
        // If we have tasks that are ready to resume, run them
        GreenletTask *to_resume = resumable_tasks.front();
        resumable_tasks.pop_front();
	mutex.unlock();
        greenlet_thread->resume_task(to_resume);
      }
      else if (task_queue.empty())
      {
        // Nothing to do, so let's go to sleep
        thread_state = GREENLET_IDLE;
	condvar.wait();
        if (!shutdown)
          assert(thread_state == GREENLET_RUNNING);
	mutex.unlock();
      }
      else
      {
        // Pull a task off the queue and execute it
        Task *task = task_queue.pop();
        if (task->func_id == 0) {
          // This is the kill pill so we need to handle it special
          finished();
          // Mark that we received the shutdown trigger
          shutdown_trigger = true;
	  condvar.signal();
	  mutex.unlock();
          // Trigger the completion task
          if (__sync_fetch_and_add(&(task->run_count),1) == 0)
            get_runtime()->get_genevent_impl(task->finish_event)->
                          trigger(task->finish_event.gen, gasnet_mynode());
          // Delete the task
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        } else {
	  mutex.unlock();
          if (__sync_fetch_and_add(&(task->run_count),1) == 0) {
            GreenletStack stack;
            if (!allocate_stack(stack))
              create_stack(stack);
            GreenletTask *green_task = new GreenletTask(task, this,
                                            stack.stack, &stack.stack_size);
            greenlet_thread->start_task(green_task);
          } else {
            // Remove our deletion reference
            if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
              delete task;
          }
        }
      }
      // If we have any complete greenlets, clean them up
      if (!complete_greenlets.empty())
      {
        for (std::vector<GreenletTask*>::const_iterator it = 
              complete_greenlets.begin(); it != complete_greenlets.end(); it++)
        {
          delete (*it);
        }
        complete_greenlets.clear();
      }
      if (shutdown)
        return (task_queue.empty() && resumable_tasks.empty());
      return false;
    }

    void GreenletProcessor::pause_task(GreenletTask *paused_task)
    {
      mutex.lock();
      bool found = false;
      // Go through and see if the task is already ready
      for (std::list<GreenletTask*>::reverse_iterator it = 
            resumable_tasks.rbegin(); it != resumable_tasks.rend(); it++)
      {
        if ((*it) == paused_task)
        {
          found = true;
          // Reverse iterator conversion requires adding 1 first
          resumable_tasks.erase((++it).base());
          break;
        }
      }
      // If we found it we're already ready so just return
      if (found)
      {
	mutex.unlock();
        return;
      }
      // Add it to the list of paused tasks
      paused_tasks.insert(paused_task);
      // Now figure out what we want to do
      if (!resumable_tasks.empty())
      {
        // Pick a task to resume and run it
        GreenletTask *to_resume = resumable_tasks.front();
        resumable_tasks.pop_front();
	mutex.unlock();
        greenlet_thread->resume_task(to_resume);
      }
      else if (!task_queue.empty())
      {
        // Pull a task off the queue and execute it
        Task *task = task_queue.pop();
        if (task->func_id == 0) {
          // This is the kill pill so we need to handle it special
          finished();
          // Mark that we received the shutdown trigger
          shutdown_trigger = true;
	  condvar.signal();
	  mutex.unlock();
          // Trigger the completion task
          if (__sync_fetch_and_add(&(task->run_count),1) == 0)
            get_runtime()->get_genevent_impl(task->finish_event)->
                          trigger(task->finish_event.gen, gasnet_mynode());
          // Delete the task
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        } else {
	  mutex.unlock();
          if (__sync_fetch_and_add(&(task->run_count),1) == 0) {
            GreenletStack stack;
            if (!allocate_stack(stack))
              create_stack(stack);
            GreenletTask *green_task = new GreenletTask(task, this,
                                            stack.stack, &stack.stack_size);
            greenlet_thread->start_task(green_task);
          } else {
            // Remove our deletion reference
            if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
              delete task;
          }
        }
      }
      else
      {
	mutex.unlock();
        // Nothing to do, send us back to the root at which
        // point we'll likely go to sleep
        greenlet_thread->return_to_root(); 
      }
    }

    void GreenletProcessor::unpause_task(GreenletTask *paused_task)
    {
      mutex.lock();
      paused_tasks.erase(paused_task);
      resumable_tasks.push_back(paused_task);
      if (thread_state == GREENLET_IDLE)
      {
        thread_state = GREENLET_RUNNING;
	condvar.signal();
      }
      mutex.unlock();
    }

    bool GreenletProcessor::allocate_stack(GreenletStack &stack)
    {
      // No need to hold the lock since only one thread is here
      if (!greenlet_stacks.empty())
      {
        stack = greenlet_stacks.back();
        greenlet_stacks.pop_back();
        return true; // succeeded
      }
      return false; // failed
    }

    void GreenletProcessor::create_stack(GreenletStack &stack)
    {
      // We need to make a stack
      // Set the suggested stack size
      stack.stack_size = proc_stack_size;
      // Then call the greenlet library
      stack.stack = greenlet::alloc_greenlet_stack(&stack.stack_size);
    }

    void GreenletProcessor::complete_greenlet(GreenletTask *greenlet)
    {
      // No need for the lock here, only one thread 
      complete_greenlets.push_back(greenlet);
      // Tricky optimization here, we can actually release
      // the stack now because we know there is only one thread
      // and we are guaranteed to exit after this call so by the
      // time this thread will try to re-use the stack we are
      // guaranteed to have finished using it.
      greenlet_stacks.push_back(GreenletStack());
      GreenletStack &last = greenlet_stacks.back();
      last.stack = greenlet->release_stack(&last.stack_size);
    }

    ProcessorAssignment::ProcessorAssignment(int _num_local_procs)
      : num_local_procs(_num_local_procs)
    {
      valid = false;

#ifdef __MACH__
      //printf("thread affinity not supported on Mac OS X\n");
      return;
#else
      cpu_set_t cset;
      int ret = sched_getaffinity(0, sizeof(cset), &cset);
      if(ret < 0) {
	printf("failed to get affinity info - binding disabled\n");
	return;
      }

      SystemProcMap proc_map;
      {
	DIR *nd = opendir("/sys/devices/system/node");
	if(!nd) {
	  printf("can't open /sys/devices/system/node - binding disabled\n");
	  return;
	}
	for(struct dirent *ne = readdir(nd); ne; ne = readdir(nd)) {
	  if(strncmp(ne->d_name, "node", 4)) continue;  // not a node directory
	  int node_id = atoi(ne->d_name + 4);
	  
	  char per_node_path[1024];
	  sprintf(per_node_path, "/sys/devices/system/node/%s", ne->d_name);
	  DIR *cd = opendir(per_node_path);
	  if(!cd) {
	    printf("can't open %s - skipping\n", per_node_path);
	    continue;
	  }

	  for(struct dirent *ce = readdir(cd); ce; ce = readdir(cd)) {
	    if(strncmp(ce->d_name, "cpu", 3)) continue; // definitely not a cpu
	    char *pos;
	    int cpu_id = strtol(ce->d_name + 3, &pos, 10);
	    if(pos && *pos) continue;  // doesn't match cpu[0-9]+
	    
	    // is this a cpu we're allowed to use?
	    if(!CPU_ISSET(cpu_id, &cset)) {
	      printf("cpu %d not available - skipping\n", cpu_id);
	      continue;
	    }

	    // figure out which physical core it is
	    char core_id_path[1024];
	    sprintf(core_id_path, "/sys/devices/system/node/%s/%s/topology/core_id", ne->d_name, ce->d_name);
	    FILE *f = fopen(core_id_path, "r");
	    if(!f) {
	      printf("can't read %s - skipping\n", core_id_path);
	      continue;
	    }
	    int core_id;
	    int count = fscanf(f, "%d", &core_id);
	    fclose(f);
	    if(count != 1) {
	      printf("can't find core id in %s - skipping\n", core_id_path);
	      continue;
	    }
	    
	    //printf("found: %d %d %d\n", node_id, cpu_id, core_id);
	    proc_map[node_id][core_id].push_back(cpu_id);
	  }
	  closedir(cd);
	}
	closedir(nd);
      }
      
#if 0
      printf("Available cores:\n");
      for(SystemProcMap::const_iterator it1 = proc_map.begin(); it1 != proc_map.end(); it1++) {
	printf("  Node %d:", it1->first);
	for(NodeProcMap::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
	  if(it2->second.size() == 1) {
	    printf(" %d", it2->second[0]);
	  } else {
	    printf(" {");
	    for(size_t i = 0; i < it2->second.size(); i++)
	      printf("%s%d", (i ? " " : ""), it2->second[i]);
	    printf("}");
	  }
	}
	printf("\n");
      }
#endif

      // count how many actual cores we have
      int core_count = 0;
      for(SystemProcMap::const_iterator it1 = proc_map.begin(); it1 != proc_map.end(); it1++)
	core_count += it1->second.size();
      
      if(core_count <= num_local_procs) {
	//printf("not enough cores (%zd) to support %d local processors - skipping binding\n", core_count, num_local_procs);
	return;
      }
      
      // pick cores for each local proc - try to round-robin across nodes
      SystemProcMap::iterator curnode = proc_map.end();
      memcpy(&leftover_procs, &cset, sizeof(cset));  // subtract from cset to get leftovers
      for(int i = 0; i < num_local_procs; i++) {
	// pick the next node with any cores left
	do {
	  if(curnode != proc_map.end())
	    curnode++;
	  if(curnode == proc_map.end())
	    curnode = proc_map.begin();
	} while(curnode->second.size() == 0);
	
	NodeProcMap::iterator curcore = curnode->second.begin();
	assert(curcore != curnode->second.end());
	assert(curcore->second.size() > 0);
	
	// take the first cpu id for this core and add it to the local proc assignments
	local_proc_assignments.push_back(curcore->second[0]);
	
	// and remove ALL cpu ids for this core from the leftover set
	for(std::vector<int>::const_iterator it = curcore->second.begin(); it != curcore->second.end(); it++)
	  CPU_CLR(*it, &leftover_procs);
	
	// and now remove this core from the node's list of available cores
	curnode->second.erase(curcore);
      }

      // we now have a valid set of bindings
      valid = true;

      // set the process' default affinity to just the leftover nodes
      bool override_default_affinity = false;
      if(override_default_affinity) {
	int ret = sched_setaffinity(0, sizeof(leftover_procs), &leftover_procs);
	if(ret < 0) {
	  printf("failed to set default affinity info!\n");
	}
      }
	
#if 0
      {
	printf("Local Proc Assignments:");
	for(std::vector<int>::const_iterator it = local_proc_assignments.begin(); it != local_proc_assignments.end(); it++)
	  printf(" %d", *it);
	printf("\n");
	
	printf("Leftover Processors   :");
	for(int i = 0; i < CPU_SETSIZE; i++)
	  if(CPU_ISSET(i, &leftover_procs))
	    printf(" %d", i);
	printf("\n");
      }
#endif
#endif
    }

    // binds a thread to the right set of cores based (-1 = not a local proc)
    void ProcessorAssignment::bind_thread(int core_id, pthread_attr_t *attr, const char *debug_name /*= 0*/)
    {
      if(!valid) {
	//printf("no processor assignment for %s %d (%p)\n", debug_name ? debug_name : "unknown", core_id, attr);
	return;
      }

#ifndef __MACH__
      if((core_id >= 0) && (core_id < num_local_procs)) {
	int cpu_id = local_proc_assignments[core_id];

	//printf("processor assignment for %s %d (%p) = %d\n", debug_name ? debug_name : "unknown", core_id, attr, cpu_id);

	cpu_set_t cset;
	CPU_ZERO(&cset);
	CPU_SET(cpu_id, &cset);
	if(attr)
	  CHECK_PTHREAD( pthread_attr_setaffinity_np(attr, sizeof(cset), &cset) );
	else
	  CHECK_PTHREAD( pthread_setaffinity_np(pthread_self(), sizeof(cset), &cset) );
      } else {
	//printf("processor assignment for %s %d (%p) = leftovers\n", debug_name ? debug_name : "unknown", core_id, attr);

	if(attr)
	  CHECK_PTHREAD( pthread_attr_setaffinity_np(attr, sizeof(leftover_procs), &leftover_procs) );
	else
	  CHECK_PTHREAD( pthread_setaffinity_np(pthread_self(), sizeof(leftover_procs), &leftover_procs) );
      }
#endif
    }

}; // namespace Realm
