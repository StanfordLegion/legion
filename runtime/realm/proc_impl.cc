/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "realm/proc_impl.h"

#include "realm/timers.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"
#include "realm/serialize.h"
#include "realm/profiling.h"
#include "realm/utils.h"

#include <sys/types.h>
#include <dirent.h>

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
      if((members.size() == 0) || (ID(members[0]).proc.owner_node == my_node_id)) {
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
      return Processor::NO_PROC;
    }

    void Processor::get_group_members(std::vector<Processor>& members)
    {
      // if we're a plain old processor, the only member of our "group" is ourself
      if(ID(*this).is_processor()) {
	members.push_back(*this);
	return;
      }

      assert(ID(*this).is_procgroup());

      ProcessorGroup *grp = get_runtime()->get_procgroup_impl(*this);
      grp->get_group_members(members);
    }

    int Processor::get_num_cores(void) const
    {
      return get_runtime()->get_processor_impl(*this)->num_cores;
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

    // changes the priority of the currently running task
    /*static*/ void Processor::set_current_task_priority(int new_priority)
    {
      // set the priority field in the task object and it'll update the thread
      Operation *op = Thread::self()->get_operation();
      assert(op != 0);
      op->set_priority(new_priority);
    }

    // returns the finish event for the currently running task
    /*static*/ Event Processor::get_current_finish_event(void)
    {
      Operation *op = Thread::self()->get_operation();
      assert(op != 0);
      return op->get_finish_event();
    }

    AddressSpace Processor::address_space(void) const
    {
      ID id(*this);
      return id.proc.owner_node;
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
      get_runtime()->optable.add_local_operation(finish_event, tro);
      // we haven't told anybody about this operation yet, so cancellation really shouldn't
      //  be possible
#ifndef NDEBUG
      bool ok_to_run =
#endif
	(tro->mark_ready() && tro->mark_started());
      assert(ok_to_run);

      std::vector<Processor> local_procs;
      std::map<NodeID, std::vector<Processor> > remote_procs;
      // is the target a single processor or a group?
      ID id(*this);
      if(id.is_processor()) {
	NodeID n = id.proc.owner_node;
	if(n == my_node_id)
	  local_procs.push_back(*this);
	else
	  remote_procs[n].push_back(*this);
      } else {
	// assume we're a group
	assert(id.is_procgroup());
	ProcessorGroup *grp = get_runtime()->get_procgroup_impl(*this);
	std::vector<Processor> members;
	grp->get_group_members(members);
	for(std::vector<Processor>::const_iterator it = members.begin();
	    it != members.end();
	    it++) {
	  Processor p = *it;
	  NodeID n = ID(p).proc.owner_node;
	  if(n == my_node_id)
	    local_procs.push_back(p);
	  else
	    remote_procs[n].push_back(p);
	}
      }

      // remote processors need a portable implementation available
      if(!remote_procs.empty()) {
	if(!tro->codedesc.has_portable_implementations()) {
	  log_taskreg.fatal() << "cannot remotely register a task with no portable implementations";
	  assert(0);
	}
      }
	 
      // local processor(s) can be called directly
      if(!local_procs.empty()) {
	for(std::vector<Processor>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  ProcessorImpl *p = get_runtime()->get_processor_impl(*it);
	  p->register_task(func_id, tro->codedesc, tro->userdata);
	}
      }

      for(std::map<NodeID, std::vector<Processor> >::const_iterator it = remote_procs.begin();
	  it != remote_procs.end();
	  it++) {
	NodeID target = it->first;
	RemoteTaskRegistration *reg_op = new RemoteTaskRegistration(tro, target);
	tro->add_async_work_item(reg_op);
	RegisterTaskMessage::send_request(target, func_id, NO_KIND, it->second,
					  tro->codedesc,
					  tro->userdata.base(), tro->userdata.size(),
					  reg_op);
      }

      tro->mark_finished(true /*successful*/);
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
      get_runtime()->optable.add_local_operation(finish_event, tro);
      // we haven't told anybody about this operation yet, so cancellation really shouldn't
      //  be possible
#ifndef NDEBUG
      bool ok_to_run =
#endif
	(tro->mark_ready() && tro->mark_started());
      assert(ok_to_run);

      // do local processors first
      std::set<Processor> local_procs;
      get_runtime()->machine->get_local_processors_by_kind(local_procs, target_kind);
      if(!local_procs.empty()) {
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
	  log_taskreg.fatal() << "cannot remotely register a task with no portable implementations";
	  assert(0);
	}

	for(NodeID target = 0; target <= max_node_id; target++) {
	  // skip ourselves
	  if(target == my_node_id)
	    continue;

	  RemoteTaskRegistration *reg_op = new RemoteTaskRegistration(tro, target);
	  tro->add_async_work_item(reg_op);
	  RegisterTaskMessage::send_request(target, func_id, target_kind, std::vector<Processor>(),
					    tro->codedesc,
					    tro->userdata.base(), tro->userdata.size(),
					    reg_op);
	}
      }

      tro->mark_finished(true /*successful*/);
      return finish_event;
    }

    // reports an execution fault in the currently running task
    /*static*/ void Processor::report_execution_fault(int reason,
						      const void *reason_data,
						      size_t reason_size)
    {
#ifdef REALM_USE_EXCEPTIONS
      if(Thread::self()->exceptions_permitted()) {
	throw ApplicationException(reason, reason_data, reason_size);
      } else
#endif
      {
	Processor p = get_executing_processor();
	assert(p.exists());
	log_poison.fatal() << "FATAL: no handler for reported processor fault: proc=" << p
			   << " reason=" << reason;
	assert(0);
      }
    }

    // reports a problem with a processor in general (this is primarily for fault injection)
    void Processor::report_processor_fault(int reason,
					   const void *reason_data,
					   size_t reason_size) const
    {
      assert(0);
    }

    /*static*/ const char* Processor::get_kind_name(Kind kind)
    {
      switch (kind)
      {
        case NO_KIND:
          return "NO_KIND";
        case TOC_PROC:
          return "TOC_PROC";
        case LOC_PROC:
          return "LOC_PROC";
        case UTIL_PROC:
          return "UTIL_PROC";
        case IO_PROC:
          return "IO_PROC";
        case PROC_GROUP:
          return "PROC_GROUP";
        case PROC_SET:
          return "PROC_SET";
        default:
          assert(0);
      }
      return NULL;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorImpl
  //

    ProcessorImpl::ProcessorImpl(Processor _me, Processor::Kind _kind,
                                 int _num_cores)
      : me(_me), kind(_kind), num_cores(_num_cores)
    {
    }

    ProcessorImpl::~ProcessorImpl(void)
    {
    }

    void ProcessorImpl::start_threads(void)
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
				      CodeDescriptor& codedesc,
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
      , ready_task_count(0)
    {
    }

    ProcessorGroup::~ProcessorGroup(void)
    {
      delete ready_task_count;
    }

    void ProcessorGroup::init(Processor _me, int _owner)
    {
      assert(ID(_me).pgroup.owner_node == (unsigned)_owner);

      me = _me;
      lock.init(ID(me).convert<Reservation>(), ID(me).pgroup.owner_node);
    }

    void ProcessorGroup::set_group_members(const std::vector<Processor>& member_list)
    {
      // can only be performed on owner node
      assert(ID(me).pgroup.owner_node == my_node_id);
      
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

      // now that we exist, profile our queue depth
      std::string gname = stringbuilder() << "realm/proc " << me << "/ready tasks";
      ready_task_count = new ProfilingGauges::AbsoluteRangeGauge<int>(gname);
      task_queue.set_gauge(ready_task_count);
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
      if(task->mark_ready())
	task_queue.put(task, task->priority);
      else
	task->mark_finished(false /*!successful*/);
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
      // check for spawn to remote processor group
      NodeID target = ID(me).pgroup.owner_node;
      if(target != my_node_id) {
	log_task.debug() << "sending remote spawn request:"
			 << " func=" << func_id
			 << " proc=" << me
			 << " finish=" << finish_event;

	get_runtime()->optable.add_remote_operation(finish_event, target);

	SpawnTaskMessage::send_request(target, me, func_id,
				       args, arglen, &reqs,
				       start_event, finish_event, priority);
	return;
      }

      // create a task object and insert it into the queue
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            start_event, finish_event, priority);
      get_runtime()->optable.add_local_operation(finish_event, task);

      bool poisoned = false;
      if (start_event.has_triggered_faultaware(poisoned)) {
	if(poisoned) {
	  log_poison.info() << "cancelling poisoned task - task=" << task << " after=" << task->get_finish_event();
	  task->handle_poisoned_precondition(start_event);
	} else
	  enqueue_task(task);
      } else
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredTaskSpawn
  //

    bool DeferredTaskSpawn::event_triggered(Event e, bool poisoned)
    {
      if(poisoned) {
	// cancel the task - this has to work
	log_poison.info() << "cancelling poisoned task - task=" << task << " after=" << task->get_finish_event();
	task->handle_poisoned_precondition(e);
	return true;
      }

      proc->enqueue_task(task);
      return true;
    }

    void DeferredTaskSpawn::print(std::ostream& os) const
    {
      os << "deferred task: func=" << task->func_id << " proc=" << task->proc << " finish=" << task->get_finish_event();
    }

    Event DeferredTaskSpawn::get_finish_event(void) const
    {
      return task->get_finish_event();
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

    log_task.debug() << "received remote spawn request:"
		     << " func=" << args.func_id
		     << " proc=" << args.proc
		     << " finish=" << args.finish_event;

    Serialization::FixedBufferDeserializer fbd(data, datalen);
    fbd.extract_bytes(0, args.user_arglen);  // skip over task args - we'll access those directly

    // profiling requests are optional - extract only if there's data
    ProfilingRequestSet prs;
    if(fbd.bytes_left() > 0)
      fbd >> prs;
      
    p->spawn_task(args.func_id, data, args.user_arglen, prs,
		  args.start_event, args.finish_event, args.priority);
  }

  /*static*/ void SpawnTaskMessage::send_request(NodeID target, Processor proc,
						 Processor::TaskFuncID func_id,
						 const void *args, size_t arglen,
						 const ProfilingRequestSet *prs,
						 Event start_event, Event finish_event,
						 int priority)
  {
    RequestArgs r_args;

    r_args.proc = proc;
    r_args.func_id = func_id;
    r_args.start_event = start_event;
    r_args.finish_event = finish_event;
    r_args.priority = priority;
    r_args.user_arglen = arglen;
    
    if(!prs || prs->empty()) {
      // no profiling, so task args are the only payload
      Message::request(target, r_args, args, arglen, PAYLOAD_COPY);
    } else {
      // need to serialize both the task args and the profiling request
      //  into a single payload
      // allocate a little extra initial space for the profiling requests, but not too
      //  much in case the copy sits around outside the srcdatapool for a long time
      // (if we need more than this, we'll pay for a realloc during serialization, but it
      //  will still work correctly)
      Serialization::DynamicBufferSerializer dbs(arglen + 512);

      dbs.append_bytes(args, arglen);
      dbs << *prs;

      size_t datalen = dbs.bytes_used();
      void *data = dbs.detach_buffer(-1);  // don't trim - this buffer has a short life
      Message::request(target, r_args, data, datalen, PAYLOAD_FREE);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegisterTaskMessage
  //

  /*static*/ void RegisterTaskMessage::handle_request(RequestArgs args, const void *data, size_t datalen)
  {
    std::vector<Processor> procs;
    CodeDescriptor codedesc;
    ByteArray userdata;

    Serialization::FixedBufferDeserializer fbd(data, datalen);
#ifndef NDEBUG
    bool ok =
#endif
      ((fbd >> procs) && (fbd >> codedesc) && (fbd >> userdata));
    assert(ok && (fbd.bytes_left() == 0));

    if(procs.empty()) {
      // use the supplied kind and find all procs of that kind
      std::set<Processor> local_procs;
      get_runtime()->machine->get_local_processors_by_kind(local_procs, args.kind);
    
      for(std::set<Processor>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++) {
	ProcessorImpl *p = get_runtime()->get_processor_impl(*it);
	p->register_task(args.func_id, codedesc, userdata);
      }
    } else {
      for(std::vector<Processor>::const_iterator it = procs.begin();
	  it != procs.end();
	  it++) {
	ProcessorImpl *p = get_runtime()->get_processor_impl(*it);
	p->register_task(args.func_id, codedesc, userdata);
      }
    }

    // TODO: include status/profiling eventually
    RegisterTaskCompleteMessage::send_request(args.sender, args.reg_op,
					      true /*successful*/);
  }

  /*static*/ void RegisterTaskMessage::send_request(NodeID target,
						    Processor::TaskFuncID func_id,
						    Processor::Kind kind,
						    const std::vector<Processor>& procs,
						    const CodeDescriptor& codedesc,
						    const void *userdata, size_t userlen,
						    RemoteTaskRegistration *reg_op)
  {
    RequestArgs args;

    args.sender = my_node_id;
    args.func_id = func_id;
    args.kind = kind;
    args.reg_op = reg_op;

    Serialization::DynamicBufferSerializer dbs(1024);
    dbs << procs;
    dbs << codedesc;
    dbs << ByteArrayRef(userdata, userlen);

    size_t datalen = dbs.bytes_used();
    void *data = dbs.detach_buffer(-1 /*no trim*/);
    Message::request(target, args, data, datalen, PAYLOAD_FREE);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegisterTaskCompleteMessage
  //

  /*static*/ void RegisterTaskCompleteMessage::handle_request(RequestArgs args)
  {
    args.reg_op->mark_finished(args.successful);
  }

  /*static*/ void RegisterTaskCompleteMessage::send_request(NodeID target,
							    RemoteTaskRegistration *reg_op,
							    bool successful)
  {
    RequestArgs args;

    args.sender = my_node_id;
    args.reg_op = reg_op;
    args.successful = successful;

    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteProcessor
  //

    RemoteProcessor::RemoteProcessor(Processor _me, Processor::Kind _kind,
                                     int _num_cores)
      : ProcessorImpl(_me, _kind, _num_cores)
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

      ID id(me);
      NodeID target = 0;
      if(id.is_processor())
	target = id.proc.owner_node;
      else if(id.is_procgroup())
	target = id.pgroup.owner_node;
      else {
	assert(0);
      }

      get_runtime()->optable.add_remote_operation(finish_event, target);

      SpawnTaskMessage::send_request(target, me, func_id,
				     args, arglen, &reqs,
				     start_event, finish_event, priority);
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalTaskProcessor
  //

  LocalTaskProcessor::LocalTaskProcessor(Processor _me, Processor::Kind _kind,
                                         int _num_cores)
    : ProcessorImpl(_me, _kind, _num_cores)
    , sched(0)
    , ready_task_count(stringbuilder() << "realm/proc " << me << "/ready tasks")
  {
    task_queue.set_gauge(&ready_task_count);
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
  }

  void LocalTaskProcessor::add_to_group(ProcessorGroup *group)
  {
    // add the group's task queue to our scheduler too
    sched->add_task_queue(&group->task_queue);
  }

  void LocalTaskProcessor::enqueue_task(Task *task)
  {
    // just jam it into the task queue
    if(task->mark_ready())
      task_queue.put(task, task->priority);
    else
      task->mark_finished(false /*!successful*/);
  }

  void LocalTaskProcessor::spawn_task(Processor::TaskFuncID func_id,
				     const void *args, size_t arglen,
				     const ProfilingRequestSet &reqs,
				     Event start_event, Event finish_event,
				     int priority)
  {
    // create a task object for this
    Task *task = new Task(me, func_id, args, arglen, reqs,
			  start_event, finish_event, priority);
    get_runtime()->optable.add_local_operation(finish_event, task);

    // if the start event has already triggered, we can enqueue right away
    bool poisoned = false;
    if (start_event.has_triggered_faultaware(poisoned)) {
      if(poisoned) {
	log_poison.info() << "cancelling poisoned task - task=" << task << " after=" << task->get_finish_event();
	task->handle_poisoned_precondition(start_event);
      } else
	enqueue_task(task);
    } else {
      EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }
  }

  void LocalTaskProcessor::register_task(Processor::TaskFuncID func_id,
					 CodeDescriptor& codedesc,
					 const ByteArrayRef& user_data)
  {
    // first, make sure we haven't seen this task id before
    if(task_table.count(func_id) > 0) {
      log_taskreg.fatal() << "duplicate task registration: proc=" << me << " func=" << func_id;
      assert(0);
    }

    // next, get see if we have a function pointer to register
    Processor::TaskFuncPtr fnptr;
    const FunctionPointerImplementation *fpi = codedesc.find_impl<FunctionPointerImplementation>();

    // if we don't have a function pointer implementation, see if we can make one
    if(!fpi) {
      const std::vector<CodeTranslator *>& translators = get_runtime()->get_code_translators();
      for(std::vector<CodeTranslator *>::const_iterator it = translators.begin();
	  it != translators.end();
	  it++)
	if((*it)->can_translate<FunctionPointerImplementation>(codedesc)) {
	  FunctionPointerImplementation *newfpi = (*it)->translate<FunctionPointerImplementation>(codedesc);
	  if(newfpi) {
	    log_taskreg.info() << "function pointer created: trans=" << (*it)->name << " fnptr=" << (void *)(newfpi->fnptr);
	    codedesc.add_implementation(newfpi);
	    fpi = newfpi;
	    break;
	  }
	}
    }

    assert(fpi != 0);

    fnptr = (Processor::TaskFuncPtr)(fpi->fnptr);

    log_taskreg.info() << "task " << func_id << " registered on " << me << ": " << codedesc;

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
	log_taskreg.info() << "task " << func_id << " not registered on " << me << ": ignoring missing legacy setup/shutdown task";
	return;
      }
      log_taskreg.fatal() << "task " << func_id << " not registered on " << me;
      assert(0);
    }

    const TaskTableEntry& tte = it->second;

    log_taskreg.debug() << "task " << func_id << " executing on " << me << ": " << ((void *)(tte.fnptr));

    (tte.fnptr)(task_args.base(), task_args.size(),
		tte.user_data.base(), tte.user_data.size(),
		me);
  }

  // starts worker threads and performs any per-processor initialization
  void LocalTaskProcessor::start_threads(void)
  {
    // finally, fire up the scheduler
    sched->start();
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
				       size_t _stack_size, bool _force_kthreads)
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
    if(!_force_kthreads) {
      UserThreadTaskScheduler *sched = new UserThreadTaskScheduler(me, *core_rsrv);
      // no config settings we want to tweak yet
      set_scheduler(sched);
    } else
#endif
    {
      KernelThreadTaskScheduler *sched = new KernelThreadTaskScheduler(me, *core_rsrv);
      sched->cfg_max_idle_workers = 3; // keep a few idle threads around
      set_scheduler(sched);
    }
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
					       size_t _stack_size, bool _force_kthreads, bool _pin_util_proc)
    : LocalTaskProcessor(_me, Processor::UTIL_PROC)
  {
    CoreReservationParameters params;
    params.set_num_cores(1);
    if (_pin_util_proc)
    {
      params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
      params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
    }
    else
    {
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
    }
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "utility proc " << _me;

    core_rsrv = new CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS
    if(!_force_kthreads) {
      UserThreadTaskScheduler *sched = new UserThreadTaskScheduler(me, *core_rsrv);
      // no config settings we want to tweak yet
      set_scheduler(sched);
    } else
#endif
    {
      KernelThreadTaskScheduler *sched = new KernelThreadTaskScheduler(me, *core_rsrv);
      // no config settings we want to tweak yet
      set_scheduler(sched);
    }
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
  {
    log_taskreg.debug() << "task registration created: op=" << (void *)this << " finish=" << _finish_event;
  }

  TaskRegistration::~TaskRegistration(void)
  {
    log_taskreg.debug() << "task registration destroyed: op=" << (void *)this;
  }

  void TaskRegistration::print(std::ostream& os) const
  {
    os << "TaskRegistration";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteTaskRegistration
  //

  RemoteTaskRegistration::RemoteTaskRegistration(TaskRegistration *reg_op, int _target_node)
    : Operation::AsyncWorkItem(reg_op)
    , target_node(_target_node)
  {}

  void RemoteTaskRegistration::request_cancellation(void)
  {
    // ignored
  }

  void RemoteTaskRegistration::print(std::ostream& os) const
  {
    os << "RemoteTaskRegistration(node=" << target_node << ")";
  }


}; // namespace Realm
