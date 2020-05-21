/* Copyright 2020 Stanford University, NVIDIA Corporation
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
#include "realm/activemsg.h"

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
    REALM_THREAD_LOCAL Processor current_processor;
    
    // if nonzero, prevents application thread from yielding execution
    //  resources on an Event wait
    REALM_THREAD_LOCAL int scheduler_lock = 0;
  };

    Processor::Kind Processor::kind(void) const
    {
      return get_runtime()->get_processor_impl(*this)->kind;
    }

    /*static*/ Processor Processor::create_group(const std::vector<Processor>& members)
    {
      // are we creating a local group?
      if((members.size() == 0) || (NodeID(ID(members[0]).proc_owner_node()) == Network::my_node_id)) {
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

    /*static*/ void Processor::free_group(const Processor& pg)
    {
      ProcessorGroup *grp = get_runtime()->get_procgroup_impl(pg);
      //first we remove members and properly clean up task queues from members list
      grp->remove_group_members();
      
      //return group to free list
      get_runtime()->local_proc_group_free_list->free_entry( grp );
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

    void ProcessorGroup::remove_group_members()
    {
      // can only be performed on owner node
      assert(NodeID(ID(me).pgroup_owner_node()) == Network::my_node_id); // TODO check with Sean
      for(std::vector<ProcessorImpl*>::iterator it = members.begin();
          it != members.end();
          it++) {
        ProcessorImpl *m_impl = *it;
        m_impl->remove_group(this);
      }
      members.clear(); //if reused we clear previous members
      members_requested = false;
      members_valid = false;
      task_queue.free_gauge();
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
		    wait_on, finish_event, ID(e).event_generation(), priority);
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
		    wait_on, finish_event, ID(e).event_generation(), priority);
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
      return id.proc_owner_node();
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

      GenEventImpl *finish_event_impl = GenEventImpl::create_genevent();
      Event finish_event = finish_event_impl->current_event();

      TaskRegistration *tro = new TaskRegistration(codedesc, 
						   ByteArrayRef(user_data, user_data_len),
						   finish_event_impl,
						   ID(finish_event).event_generation(),
						   prs);
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
	NodeID n = id.proc_owner_node();
	if(n == Network::my_node_id)
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
	  NodeID n = ID(p).proc_owner_node();
	  if(n == Network::my_node_id)
	    local_procs.push_back(p);
	  else
	    remote_procs[n].push_back(p);
	}
      }

      // remote processors need a portable implementation available
      if(!remote_procs.empty()) {
	// try to create one if we don't have one
	if(!tro->codedesc.has_portable_implementations() &&
	   !tro->codedesc.create_portable_implementation()) {
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
	ActiveMessage<RegisterTaskMessage> amsg(target, 65536);
	amsg->func_id = func_id;
	amsg->kind = NO_KIND;
	amsg->reg_op = reg_op;
	amsg << it->second;
	amsg << tro->codedesc;
	amsg << ByteArrayRef(tro->userdata.base(), tro->userdata.size());
	amsg.commit();
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

      GenEventImpl *finish_event_impl = GenEventImpl::create_genevent();
      Event finish_event = finish_event_impl->current_event();

      TaskRegistration *tro = new TaskRegistration(codedesc, 
						   ByteArrayRef(user_data, user_data_len),
						   finish_event_impl,
						   ID(finish_event).event_generation(),
						   prs);
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
	if(!tro->codedesc.has_portable_implementations() &&
	   !tro->codedesc.create_portable_implementation()) {
	  log_taskreg.fatal() << "cannot remotely register a task with no portable implementations";
	  assert(0);
	}

	for(NodeID target = 0; target <= Network::max_node_id; target++) {
	  // skip ourselves
	  if(target == Network::my_node_id)
	    continue;

	  RemoteTaskRegistration *reg_op = new RemoteTaskRegistration(tro, target);
	  tro->add_async_work_item(reg_op);
	  ActiveMessage<RegisterTaskMessage> amsg(target, 65536);
	  amsg->func_id = func_id;
	  amsg->kind = target_kind;
	  amsg->reg_op = reg_op;
	  amsg << std::vector<Processor>();
	  amsg << tro->codedesc;
	  amsg << ByteArrayRef(tro->userdata.base(), tro->userdata.size());
	  amsg.commit();
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

  /*static*/ void Processor::enable_scheduler_lock(void)
  {
#ifdef DEBUG_REALM
    assert(ThreadLocal::current_processor.exists());
#endif
    ThreadLocal::scheduler_lock++;
  }

  /*static*/ void Processor::disable_scheduler_lock(void)
  {
#ifdef DEBUG_REALM
    assert(ThreadLocal::scheduler_lock > 0);
#endif
    ThreadLocal::scheduler_lock--;
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

    // helper function for spawn implementations
    void ProcessorImpl::enqueue_or_defer_task(Task *task, Event start_event,
					      DeferredSpawnCache *cache)
    {
      // case 1: no precondition
      if(!start_event.exists()) {
	enqueue_task(task);
	return;
      }

      // case 2: precondition is triggered or poisoned
      EventImpl *start_impl = get_runtime()->get_event_impl(start_event);
      EventImpl::gen_t start_gen = ID(start_event).event_generation();
      bool poisoned = false;
      if(!start_impl->has_triggered(start_gen, poisoned)) {
	// we'll create a new deferral unless we can tack it on to an existing
	//  one
	bool new_deferral = true;
        // we might hit in the cache below, but set up the deferral before to
        //  avoid race conditions with other tasks being added
	task->deferred_spawn.setup(this, task, start_event);

	if(cache) {
	  Task *leader = 0;
	  Task *evicted = 0;
	  {
	    AutoLock<> al(cache->mutex);
	    size_t i = 0;
	    while((i < DeferredSpawnCache::MAX_ENTRIES) &&
		  (cache->events[i] != start_event)) i++;
	    if(i < DeferredSpawnCache::MAX_ENTRIES) {
	      // cache hit
	      cache->counts[i]++;
	      leader = cache->tasks[i];
	      leader->add_reference();  // keep it alive until we use it below
	    } else {
	      // miss - see if any counts are at 0
	      i = 0;
	      while((i < DeferredSpawnCache::MAX_ENTRIES) &&
		    (cache->counts[i] > 0)) i++;
	      // no? decrement them all and see if one goes to 0 now
	      if(i < DeferredSpawnCache::MAX_ENTRIES) {
		i = 0;
		while((i < DeferredSpawnCache::MAX_ENTRIES) &&
		      (--cache->counts[i] > 0)) i++;
		// decrement the rest too
		for(size_t j = i+1; j < DeferredSpawnCache::MAX_ENTRIES; j++)
		  cache->counts[j]--;
	      }

	      // if we've got a candidate now, do a replacement
	      if(i < DeferredSpawnCache::MAX_ENTRIES) {
		evicted = cache->tasks[i];
		cache->events[i] = start_event;
		cache->tasks[i] = task;
		cache->counts[i] = 1;
		task->add_reference(); // cache holds a reference now too
	      }
	    }
	  }
	  // decrement the refcount on a task we evicted (if any)
	  if(evicted)
	    evicted->remove_reference();

	  // if we found a leader, try to add ourselves to their list
	  if(leader) {
	    bool added = leader->deferred_spawn.add_task(task, poisoned);
            leader->remove_reference();  // safe to let go of this now
            if(added) {
	      // success - nothing more needs to be done here
	      return;
	    } else {
	      // failure, so no deferral is needed - fall through to
	      //  enqueue-or-cancel code below
	      new_deferral = false;
	    }
	  }
	}

	if(new_deferral) {
	  task->deferred_spawn.defer(start_impl, start_gen);
	  return;
	}
      }

      // precondition is either triggered or poisoned
      if(poisoned) {
	log_poison.info() << "cancelling poisoned task - task=" << task << " after=" << task->get_finish_event();
	task->handle_poisoned_precondition(start_event);
      } else {
	enqueue_task(task);
      }
    }

    // runs an internal Realm operation on this processor
    void ProcessorImpl::add_internal_task(InternalTask *task)
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
      deferred_spawn_cache.clear();
    }

    ProcessorGroup::~ProcessorGroup(void)
    {
      deferred_spawn_cache.flush();
      delete ready_task_count;
    }

    void ProcessorGroup::init(Processor _me, int _owner)
    {
      assert(NodeID(ID(_me).pgroup_owner_node()) == _owner);

      me = _me;
      lock.init(ID(me).convert<Reservation>(), ID(me).pgroup_owner_node());
    }

    void ProcessorGroup::set_group_members(const std::vector<Processor>& member_list)
    {
      // can only be performed on owner node
      assert(NodeID(ID(me).pgroup_owner_node()) == Network::my_node_id);
      
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

    void ProcessorGroup::remove_group(ProcessorGroup *group) {
      assert(0);
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
      task_queue.enqueue_task(task);
    }

    void ProcessorGroup::enqueue_tasks(Task::TaskList& tasks)
    {
      task_queue.enqueue_tasks(tasks);
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
						Event start_event,
						GenEventImpl *finish_event,
						EventImpl::gen_t finish_gen,
						int priority)
    {
      // check for spawn to remote processor group
      NodeID target = ID(me).pgroup_owner_node();
      if(target != Network::my_node_id) {
	Event e = finish_event->make_event(finish_gen);
	log_task.debug() << "sending remote spawn request:"
			 << " func=" << func_id
			 << " proc=" << me
			 << " finish=" << e;

	get_runtime()->optable.add_remote_operation(e, target);
	// if profiling data need DynamicBufferSerializer?
	size_t len = reqs.empty() ? arglen: arglen+512;
	ActiveMessage<SpawnTaskMessage> amsg(target, len);
	amsg->proc = me;
	amsg->start_event = start_event;
	amsg->finish_event = e;
	amsg->user_arglen = arglen;
	amsg->priority = priority;
	amsg->func_id = func_id;
	amsg.add_payload(args, arglen);
	// if profiling data
	if (!reqs.empty()) {
	  amsg << reqs;
	}
	amsg.commit();
	return;
      }

      // create a task object and insert it into the queue
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            start_event, finish_event, finish_gen, priority);
      get_runtime()->optable.add_local_operation(finish_event->make_event(finish_gen),
						 task);

      enqueue_or_defer_task(task, start_event, &deferred_spawn_cache);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegisterTaskMessage
  //

  /*static*/ void RegisterTaskMessage::handle_message(NodeID sender, const RegisterTaskMessage &args,
						      const void *data, size_t datalen)
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
    ActiveMessage<RegisterTaskCompleteMessage> amsg(sender);
    amsg->reg_op = args.reg_op;
    amsg->successful = true;
    amsg.commit();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RegisterTaskCompleteMessage
  //

  /*static*/ void RegisterTaskCompleteMessage::handle_message(NodeID sender,
							      const RegisterTaskCompleteMessage &args,
							      const void *data, size_t datalen)
  {
    args.reg_op->mark_finished(args.successful);
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

    void RemoteProcessor::enqueue_tasks(Task::TaskList& tasks)
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
				     Event start_event,
				     GenEventImpl *finish_event,
				     EventImpl::gen_t finish_gen,
				     int priority)
    {
      Event e = finish_event->make_event(finish_gen);
      log_task.debug() << "sending remote spawn request:"
		       << " func=" << func_id
		       << " proc=" << me
		       << " finish=" << e;

      ID id(me);
      NodeID target = 0;
      if(id.is_processor())
	target = id.proc_owner_node();
      else if(id.is_procgroup())
	target = id.pgroup_owner_node();
      else {
	assert(0);
      }

      get_runtime()->optable.add_remote_operation(e, target);
      // if profiling data need DynamicBufferSerializer?
      size_t len = reqs.empty() ? arglen: arglen+512;
      ActiveMessage<SpawnTaskMessage> amsg(target,len);
      amsg->proc = me;
      amsg->start_event = start_event;
      amsg->finish_event = e;
      amsg->user_arglen = arglen;
      amsg->priority = priority;
      amsg->func_id = func_id;
      amsg.add_payload(args, arglen);
      // if profiling data
      if (!reqs.empty()) {
	amsg << reqs;
      }
      amsg.commit();
    }

    void RemoteProcessor::remove_group(ProcessorGroup *group)
    {
      // not currently supported
      assert(0);
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
    deferred_spawn_cache.clear();
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

  void LocalTaskProcessor::remove_group(ProcessorGroup *group)
  {
    // remove the group's task queue from our scheduler
    sched->remove_task_queue(&group->task_queue);
  }

  void LocalTaskProcessor::enqueue_task(Task *task)
  {
    task_queue.enqueue_task(task);
  }

  void LocalTaskProcessor::enqueue_tasks(Task::TaskList& tasks)
  {
    task_queue.enqueue_tasks(tasks);
  }

  void LocalTaskProcessor::spawn_task(Processor::TaskFuncID func_id,
				      const void *args, size_t arglen,
				      const ProfilingRequestSet &reqs,
				      Event start_event,
				      GenEventImpl *finish_event,
				      EventImpl::gen_t finish_gen,
				      int priority)
  {
    // create a task object for this
    Task *task = new Task(me, func_id, args, arglen, reqs,
			  start_event, finish_event, finish_gen, priority);
    get_runtime()->optable.add_local_operation(finish_event->make_event(finish_gen),
					       task);

    enqueue_or_defer_task(task, start_event, &deferred_spawn_cache);
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
    if(func_id == Processor::TASK_ID_PROCESSOR_NOP)
      return;

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
    deferred_spawn_cache.flush();
  }
  
  // runs an internal Realm operation on this processor
  void LocalTaskProcessor::add_internal_task(InternalTask *task)
  {
    sched->add_internal_task(task);
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
				     GenEventImpl *_finish_event,
				     EventImpl::gen_t _finish_gen,
				     const ProfilingRequestSet &_requests)
    : Operation(_finish_event, _finish_gen, _requests)
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


  ////////////////////////////////////////////////////////////////////////
  //
  // class SpawnTaskMessage
  //
  /*static*/ void SpawnTaskMessage::handle_message(NodeID sender,
						      const SpawnTaskMessage &args,
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

    GenEventImpl *finish_impl = get_runtime()->get_genevent_impl(args.finish_event);
    p->spawn_task(args.func_id, data, args.user_arglen, prs,
		  args.start_event,
		  finish_impl, ID(args.finish_event).event_generation(),
		  args.priority);
  }

  ActiveMessageHandlerReg<SpawnTaskMessage> spawn_task_message_handler;
  ActiveMessageHandlerReg<RegisterTaskMessage> register_task_message_handler;
  ActiveMessageHandlerReg<RegisterTaskCompleteMessage> register_task_complete_message_handler;

}; // namespace Realm
