/* Copyright 2016 Stanford University, NVIDIA Corporation
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
#include "utils.h"

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
  Logger log_taskreg("taskreg");

  ////////////////////////////////////////////////////////////////////////
  //
  // class Processor
  //

    /*static*/ const Processor Processor::NO_PROC = { 0 }; 

  namespace ThreadLocal {
    __thread Processor current_processor;
  };

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

      p->spawn_task(func_id, args, arglen, ProfilingRequestSet(),
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

      // TODO: special case - registration on a local processor with a raw function pointer and no
      //  profiling requests - can be done immediately and return NO_EVENT

      Event finish_event = GenEventImpl::create_genevent()->current_event();

      TaskRegistration *tro = new TaskRegistration(codedesc, 
						   ByteArrayRef(user_data, user_data_len),
						   finish_event, prs);
      tro->mark_ready();
      tro->mark_started();

      std::vector<Processor> local_procs;
      std::map<gasnet_node_t, std::vector<Processor> > remote_procs;
      // is the target a single processor or a group?
      if(ID(*this).type() == ID::ID_PROCESSOR) {
	gasnet_node_t n = ID(*this).node();
	if(n == gasnet_mynode())
	  local_procs.push_back(*this);
	else
	  remote_procs[n].push_back(*this);
      } else {
	// assume we're a group
	ProcessorGroup *grp = get_runtime()->get_procgroup_impl(*this);
	std::vector<Processor> members;
	grp->get_group_members(members);
	for(std::vector<Processor>::const_iterator it = members.begin();
	    it != members.end();
	    it++) {
	  Processor p = *it;
	  gasnet_node_t n = ID(p).node();
	  if(n == gasnet_mynode())
	    local_procs.push_back(p);
	  else
	    remote_procs[n].push_back(p);
	}
      }

      // remote processors need a portable implementation available
      if(!remote_procs.empty()) {
	if(!tro->codedesc.has_portable_implementations()) {
	  // try converting a function pointer into a DSO reference
	  const FunctionPointerImplementation *fpi = tro->codedesc.find_impl<FunctionPointerImplementation>();
	  if(!fpi) {
	    log_taskreg.fatal() << "remote proc needs portable code: no function pointer available either";
	    assert(0);
	  }
	  DSOReferenceImplementation *dso = cvt_fnptr_to_dsoref(fpi);
	  if(!dso) {
	    log_taskreg.fatal() << "couldn't generate DSO reference for remote task registration";
	    assert(0);
	  }
	  tro->codedesc.add_implementation(dso);
	}
      }
	 
      // local processor(s) can be called directly
      if(!local_procs.empty()) {
	// for now, always need a function pointer implementation
	if(!tro->codedesc.find_impl<FunctionPointerImplementation>()) {
	  // try to make one from a dso reference, if available
	  const DSOReferenceImplementation *dso = tro->codedesc.find_impl<DSOReferenceImplementation>();
	  if(!dso) {
	    log_taskreg.fatal() << "local task registration needs fnptr or DSO reference!";
	    assert(0);
	  }
	  FunctionPointerImplementation *fpi = cvt_dsoref_to_fnptr(dso);
	  if(!fpi) {
	    log_taskreg.fatal() << "failed to convert DSO reference to function pointer";
	    assert(0);
	  }
	}

	for(std::vector<Processor>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  ProcessorImpl *p = get_runtime()->get_processor_impl(*it);
	  p->register_task(func_id, tro->codedesc, tro->userdata);
	}
      }

      if(!remote_procs.empty()) {
	// TODO: remote proc case
	assert(0);
      }

      tro->mark_finished();
      return finish_event;
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

      // TODO: special case - registration on local processord with a raw function pointer and no
      //  profiling requests - can be done immediately and return NO_EVENT

      Event finish_event = GenEventImpl::create_genevent()->current_event();

      TaskRegistration *tro = new TaskRegistration(codedesc, 
						   ByteArrayRef(user_data, user_data_len),
						   finish_event, prs);
      tro->mark_ready();
      tro->mark_started();

      // do local processors first
      std::set<Processor> local_procs;
      get_runtime()->machine->get_local_processors_by_kind(local_procs, target_kind);
      if(!local_procs.empty()) {
	// for now, always need a function pointer implementation
	if(!tro->codedesc.find_impl<FunctionPointerImplementation>()) {
	  // try to make one from a dso reference, if available
	  const DSOReferenceImplementation *dso = tro->codedesc.find_impl<DSOReferenceImplementation>();
	  if(!dso) {
	    log_taskreg.fatal() << "local task registration needs fnptr or DSO reference!";
	    assert(0);
	  }
	  FunctionPointerImplementation *fpi = cvt_dsoref_to_fnptr(dso);
	  if(!fpi) {
	    log_taskreg.fatal() << "failed to convert DSO reference to function pointer";
	    assert(0);
	  }
	}

	for(std::set<Processor>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  ProcessorImpl *p = get_runtime()->get_processor_impl(*it);
	  p->register_task(func_id, tro->codedesc, tro->userdata);
	}
      }

      if(global) {
	// remote processors need a portable implementation available
	if(!tro->codedesc.has_portable_implementations()) {
	  // try converting a function pointer into a DSO reference
	  const FunctionPointerImplementation *fpi = tro->codedesc.find_impl<FunctionPointerImplementation>();
	  if(!fpi) {
	    log_taskreg.fatal() << "remote proc needs portable code: no function pointer available either";
	    assert(0);
	  }
	  DSOReferenceImplementation *dso = cvt_fnptr_to_dsoref(fpi);
	  if(!dso) {
	    log_taskreg.fatal() << "couldn't generate DSO reference for remote task registration";
	    assert(0);
	  }
	  tro->codedesc.add_implementation(dso);
	}

	// TODO: remote proc case
	assert(0);
      }

      tro->mark_finished();
      return finish_event;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorImpl
  //

    ProcessorImpl::ProcessorImpl(Processor _me, Processor::Kind _kind)
      : me(_me), kind(_kind)
    {
    }

    ProcessorImpl::~ProcessorImpl(void)
    {
    }

    void ProcessorImpl::shutdown(void)
    {
    }

    void ProcessorImpl::execute_task(Processor::TaskFuncID func_id,
				     const ByteArrayRef& task_args)
    {
      // should never be called
      assert(0);
    }

    void ProcessorImpl::register_task(Processor::TaskFuncID func_id,
				      const CodeDescriptor& codedesc,
				      const ByteArrayRef& user_data)
    {
      // should never be called
      assert(0);
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
	m_impl->add_to_group(this);
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

    void ProcessorGroup::enqueue_task(Task *task)
    {
      // put it into the task queue - one of the member procs will eventually grab it
      task->mark_ready();
      task_queue.put(task, task->priority);
    }

    void ProcessorGroup::add_to_group(ProcessorGroup *group)
    {
      // recursively add all of our members
      assert(members_valid);

      for(std::vector<ProcessorImpl *>::const_iterator it = members.begin();
	  it != members.end();
	  it++)
	(*it)->add_to_group(group);
    }

    /*virtual*/ void ProcessorGroup::spawn_task(Processor::TaskFuncID func_id,
						const void *args, size_t arglen,
                                                const ProfilingRequestSet &reqs,
						Event start_event, Event finish_event,
						int priority)
    {
      // create a task object and insert it into the queue
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            start_event, finish_event, priority);

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
      proc->enqueue_task(task);
      return true;
    }

    void DeferredTaskSpawn::print_info(FILE *f)
    {
      fprintf(f,"deferred task: func=%d proc=" IDFMT " finish=" IDFMT "/%d\n",
             task->func_id, task->proc.id, task->get_finish_event().id, task->get_finish_event().gen);
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

    Event start_event, finish_event;
    start_event.id = args.start_id;
    start_event.gen = args.start_gen;
    finish_event.id = args.finish_id;
    finish_event.gen = args.finish_gen;

    log_task.debug() << "received remote spawn request:"
		     << " func=" << args.func_id
		     << " proc=" << args.proc
		     << " finish=" << finish_event;

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

    void RemoteProcessor::enqueue_task(Task *task)
    {
      // should never be called
      assert(0);
    }

    void RemoteProcessor::add_to_group(ProcessorGroup *group)
    {
      // not currently supported
      assert(0);
    }

    void RemoteProcessor::spawn_task(Processor::TaskFuncID func_id,
				     const void *args, size_t arglen,
				     const ProfilingRequestSet &reqs,
				     Event start_event, Event finish_event,
				     int priority)
    {
      log_task.debug() << "sending remote spawn request:"
		       << " func=" << func_id
		       << " proc=" << me
		       << " finish=" << finish_event;

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
			 Event::NOEVENT, Event::NO_EVENT, 0);
      task_queue.put(t, task_queue.PRI_MAX_FINITE);
    } else {
      log_proc.info("no processor init task: proc=" IDFMT "", me.id);
    }
#endif

    // finally, fire up the scheduler
    sched->start();
  }

  void LocalTaskProcessor::add_to_group(ProcessorGroup *group)
  {
    // add the group's task queue to our scheduler too
    sched->add_task_queue(&group->task_queue);
  }

  void LocalTaskProcessor::enqueue_task(Task *task)
  {
    // just jam it into the task queue
    task->mark_ready();
    task_queue.put(task, task->priority);
  }

  void LocalTaskProcessor::spawn_task(Processor::TaskFuncID func_id,
				     const void *args, size_t arglen,
				     const ProfilingRequestSet &reqs,
				     Event start_event, Event finish_event,
				     int priority)
  {
    assert(func_id != 0);
    // create a task object for this
    Task *task = new Task(me, func_id, args, arglen, reqs,
			  start_event, finish_event, priority);

    // if the start event has already triggered, we can enqueue right away
    if(start_event.has_triggered()) {
      enqueue_task(task);
    } else {
      EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }
  }

  void LocalTaskProcessor::register_task(Processor::TaskFuncID func_id,
					 const CodeDescriptor& codedesc,
					 const ByteArrayRef& user_data)
  {
    // first, make sure we haven't seen this task id before
    assert(task_table.count(func_id) == 0);

    // next, get see if we have a function pointer to register
    Processor::TaskFuncPtr fnptr;
    const FunctionPointerImplementation *fpi = codedesc.find_impl<FunctionPointerImplementation>();
    if(fpi) {
      fnptr = (Processor::TaskFuncPtr)(fpi->fnptr);
    } else {
      assert(0);
    }

    log_taskreg.info() << "task " << func_id << " registered on " << me << ": " << fnptr;

    TaskTableEntry &tte = task_table[func_id];
    tte.fnptr = fnptr;
    tte.user_data = user_data;
  }

  void LocalTaskProcessor::execute_task(Processor::TaskFuncID func_id,
					const ByteArrayRef& task_args)
  {
    std::map<Processor::TaskFuncID, TaskTableEntry>::const_iterator it = task_table.find(func_id);
    if(it == task_table.end()) {
      // TODO: remove this hack once the tools are available to the HLR to call these directly
      if(func_id < Processor::TASK_ID_FIRST_AVAILABLE) {
	log_taskreg.warning() << "task " << func_id << " not registered on " << me << ": ignoring missing legacy setup/shutdown task";
	return;
      }
      log_taskreg.fatal() << "task " << func_id << " not registered on " << me;
      assert(0);
    }

    const TaskTableEntry& tte = it->second;

    log_taskreg.debug() << "task " << func_id << " executing on " << me << ": " << tte.fnptr;

    (tte.fnptr)(task_args.base(), task_args.size(),
		tte.user_data.base(), tte.user_data.size(),
		me);
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
			 Event::NO_EVENT, Event::NO_EVENT, 0);
      task_queue.put(t, task_queue.PRI_MIN_FINITE);
    } else {
      log_proc.info("no processor shutdown task: proc=" IDFMT "", me.id);
    }
#endif

    sched->shutdown();
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalCPUProcessor
  //

  LocalCPUProcessor::LocalCPUProcessor(Processor _me, CoreReservationSet& crs,
				       size_t _stack_size)
    : LocalTaskProcessor(_me, Processor::LOC_PROC)
  {
    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "CPU proc " << _me;

    core_rsrv = new CoreReservation(name, crs, params);

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

  LocalUtilityProcessor::LocalUtilityProcessor(Processor _me, CoreReservationSet& crs,
					       size_t _stack_size)
    : LocalTaskProcessor(_me, Processor::UTIL_PROC)
  {
    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_alu_usage(params.CORE_USAGE_SHARED);
    params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "utility proc " << _me;

    core_rsrv = new CoreReservation(name, crs, params);

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

  LocalIOProcessor::LocalIOProcessor(Processor _me, CoreReservationSet& crs,
				     size_t _stack_size, int _concurrent_io_threads)
    : LocalTaskProcessor(_me, Processor::IO_PROC)
  {
    CoreReservationParameters params;
    params.set_alu_usage(params.CORE_USAGE_SHARED);
    params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "IO proc " << _me;

    core_rsrv = new CoreReservation(name, crs, params);

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


  ////////////////////////////////////////////////////////////////////////
  //
  // class TaskRegistration
  //

  TaskRegistration::TaskRegistration(const CodeDescriptor& _codedesc,
				     const ByteArrayRef& _userdata,
				     Event _finish_event, const ProfilingRequestSet &_requests)
    : Operation(_finish_event, _requests)
    , codedesc(_codedesc), userdata(_userdata)
  {}


}; // namespace Realm
