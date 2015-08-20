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
#include "profiling.h"

#ifdef USE_CUDA
#include "lowlevel_gpu.h"
#endif

#include <sys/types.h>
#include <dirent.h>

GASNETT_THREADKEY_DEFINE(cur_preemptable_thread);

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

namespace Realm {

  extern Logger log_task;  // defined in tasks.cc
  extern Logger log_util;  // defined in tasks.cc


  ////////////////////////////////////////////////////////////////////////
  //
  // class Processor
  //

    /*static*/ const Processor Processor::NO_PROC = { 0 }; 

  namespace ThreadLocal {
    __thread Processor current_processor;
  };

#if 0
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
      return LegionRuntime::LowLevel::GPUProcessor::get_processor();
#else
      assert(0);
#endif
    }
#endif

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
				     const ProfilingRequestSet &reqs,
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

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalTaskProcessor
  //

  LocalTaskProcessor::LocalTaskProcessor(Processor _me, Processor::Kind _kind)
    : ProcessorImpl(_me, _kind), sched(0)
  {
    // nothing really happens until we get a scheduler
  }

  LocalTaskProcessor::~LocalTaskProcessor(void)
  {
    delete sched;
  }

  void LocalTaskProcessor::set_scheduler(ThreadedTaskScheduler *_sched)
  {
    sched = _sched;

    // add our task queue to the scheduler
    sched->add_task_queue(&task_queue);

    // this should be requested from outside now
#if 0
    // if we have an init task, queue that up (with highest priority)
    Processor::TaskIDTable::iterator it = 
      get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_INIT);
    if(it != get_runtime()->task_table.end()) {
      Task *t = new Task(me, Processor::TASK_ID_PROCESSOR_INIT,
			 0, 0,
			 Event::NO_EVENT, 0, 1);
      task_queue.put(t, task_queue.PRI_MAX_FINITE);
    } else {
      log_task.info("no processor init task: proc=" IDFMT "", me.id);
    }
#endif

    // finally, fire up the scheduler
    sched->start();
  }

  // old methods to delete
  void LocalTaskProcessor::start_processor(void) { assert(0); }
  void LocalTaskProcessor::shutdown_processor(void) { assert(0); }
  void LocalTaskProcessor::initialize_processor(void) { assert(0); }
  void LocalTaskProcessor::finalize_processor(void) { assert(0); }

  void LocalTaskProcessor::enqueue_task(Task *task)
  {
    // just jam it into the task queue
    task->mark_ready();
    task_queue.put(task, task->priority);
  }

  void LocalTaskProcessor::spawn_task(Processor::TaskFuncID func_id,
				     const void *args, size_t arglen,
				     //std::set<RegionInstance> instances_needed,
				     Event start_event, Event finish_event,
				     int priority)
  {
    spawn_task(func_id, args, arglen, ProfilingRequestSet(),
	       start_event, finish_event, priority);
  }

  void LocalTaskProcessor::spawn_task(Processor::TaskFuncID func_id,
				     const void *args, size_t arglen,
				     const ProfilingRequestSet &reqs,
				     Event start_event, Event finish_event,
				     int priority)
  {
    assert(func_id != 0);
    // create a task object for this
    Task *task = new Task(me, func_id, args, arglen, reqs, finish_event, priority, 1);

    // if the start event has already triggered, we can enqueue right away
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

  // blocks until things are cleaned up
  void LocalTaskProcessor::shutdown(void)
  {
    // this should be requested from outside now
#if 0
    // enqueue a shutdown task, if it exists
    Processor::TaskIDTable::iterator it = 
      get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
    if(it != get_runtime()->task_table.end()) {
      Task *t = new Task(me, Processor::TASK_ID_PROCESSOR_SHUTDOWN,
			 0, 0,
			 Event::NO_EVENT, 0, 1);
      task_queue.put(t, task_queue.PRI_MIN_FINITE);
    } else {
      log_task.info("no processor shutdown task: proc=" IDFMT "", me.id);
    }
#endif

    sched->shutdown();
  }
  

  class stringbuilder {
  public:
    operator std::string(void) const { return ss.str(); }
    template <typename T>
    stringbuilder& operator<<(T data) { ss << data; return *this; }
  protected:
    std::stringstream ss;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalCPUProcessor
  //

  LocalCPUProcessor::LocalCPUProcessor(Processor _me, size_t _stack_size)
    : LocalTaskProcessor(_me, Processor::LOC_PROC)
  {
    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "CPU proc " << _me;

    core_rsrv = new CoreReservation(name, params);

#ifdef REALM_USE_USER_THREADS
    UserThreadTaskScheduler *sched = new UserThreadTaskScheduler(me, *core_rsrv);
    // no config settings we want to tweak yet
#else
    KernelThreadTaskScheduler *sched = new KernelThreadTaskScheduler(me, *core_rsrv);
    sched->cfg_max_idle_workers = 3; // keep a few idle threads around
#endif
    set_scheduler(sched);
  }

  LocalCPUProcessor::~LocalCPUProcessor(void)
  {
    delete core_rsrv;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalUtilityProcessor
  //

  LocalUtilityProcessor::LocalUtilityProcessor(Processor _me, size_t _stack_size)
    : LocalTaskProcessor(_me, Processor::UTIL_PROC)
  {
    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_alu_usage(params.CORE_USAGE_SHARED);
    params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "utility proc " << _me;

    core_rsrv = new CoreReservation(name, params);

#ifdef REALM_USE_USER_THREADS
    UserThreadTaskScheduler *sched = new UserThreadTaskScheduler(me, *core_rsrv);
    // no config settings we want to tweak yet
#else
    KernelThreadTaskScheduler *sched = new KernelThreadTaskScheduler(me, *core_rsrv);
    // no config settings we want to tweak yet
#endif
    set_scheduler(sched);
  }

  LocalUtilityProcessor::~LocalUtilityProcessor(void)
  {
    delete core_rsrv;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalIOProcessor
  //

  LocalIOProcessor::LocalIOProcessor(Processor _me, size_t _stack_size,
                                     int _concurrent_io_threads)
    : LocalTaskProcessor(_me, Processor::IO_PROC)
  {
    CoreReservationParameters params;
    params.set_alu_usage(params.CORE_USAGE_SHARED);
    params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "IO proc " << _me;

    core_rsrv = new CoreReservation(name, params);

    // IO processors always use kernel threads
    ThreadedTaskScheduler *sched = new KernelThreadTaskScheduler(me, *core_rsrv);

    // allow concurrent IO threads
    sched->cfg_max_active_workers = _concurrent_io_threads;

    set_scheduler(sched);
  }

  LocalIOProcessor::~LocalIOProcessor(void)
  {
    delete core_rsrv;
  }


}; // namespace Realm
