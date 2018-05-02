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

// Runtime implementation for Realm

#include "realm/runtime_impl.h"

#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/inst_impl.h"

#include "realm/activemsg.h"
#include "realm/deppart/preimage.h"

#include "realm/cmdline.h"

#include "realm/codedesc.h"

#include "realm/utils.h"

// For doing backtraces
#include <execinfo.h> // symbols
#include <cxxabi.h>   // demangling

#ifdef USE_GASNET
#ifndef GASNET_PAR
#define GASNET_PAR
#endif
#include <gasnet.h>
#include <gasnet_coll.h>
// eliminate GASNet warnings for unused static functions
static const void *ignore_gasnet_warning1 __attribute__((unused)) = (void *)_gasneti_threadkey_init;
#ifdef _INCLUDED_GASNET_TOOLS_H
static const void *ignore_gasnet_warning2 __attribute__((unused)) = (void *)_gasnett_trace_printf_noop;
#endif
#endif

#ifndef USE_GASNET
/*extern*/ void *fake_gasnet_mem_base = 0;
/*extern*/ size_t fake_gasnet_mem_size = 0;
#endif

// remote copy active messages from from lowlevel_dma.h for now
#include "realm/transfer/lowlevel_dma.h"

// create xd message and update bytes read/write messages
#include "realm/transfer/channel.h"

#include <unistd.h>
#include <signal.h>

#include <fstream>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#ifdef USE_GASNET
#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)
#endif

TYPE_IS_SERIALIZABLE(Realm::NodeAnnounceTag);
TYPE_IS_SERIALIZABLE(Realm::Memory);
TYPE_IS_SERIALIZABLE(Realm::Memory::Kind);
TYPE_IS_SERIALIZABLE(Realm::Channel::SupportedPath);
TYPE_IS_SERIALIZABLE(Realm::XferDes::XferKind);

namespace LegionRuntime {
  namespace Accessor {
    namespace DebugHooks {
      // these are calls that can be implemented by a higher level (e.g. Legion) to
      //  perform privilege/bounds checks on accessor reference and produce more useful
      //  information for debug

      /*extern*/ void (*check_bounds_ptr)(void *region, ptr_t ptr) = 0;
      /*extern*/ void (*check_bounds_dpoint)(void *region, const Legion::DomainPoint &dp) = 0;

      /*extern*/ const char *(*find_privilege_task_name)(void *region) = 0;
    };
  };
};

namespace Realm {

  Logger log_runtime("realm");
  Logger log_collective("collective");
  extern Logger log_task; // defined in proc_impl.cc
  extern Logger log_taskreg; // defined in proc_impl.cc
  
  ////////////////////////////////////////////////////////////////////////
  //
  // signal handlers
  //

    static void realm_freeze(int signal)
    {
      assert((signal == SIGINT) || (signal == SIGABRT) ||
             (signal == SIGSEGV) || (signal == SIGFPE) ||
             (signal == SIGBUS));
      int process_id = getpid();
      char hostname[128];
      gethostname(hostname, 127);
      fprintf(stderr,"Legion process received signal %d: %s\n",
                      signal, strsignal(signal));
      fprintf(stderr,"Process %d on node %s is frozen!\n", 
                      process_id, hostname);
      fflush(stderr);
      while (true)
        sleep(1);
    }

  // not static so that it can be invoked manually from gdb
  void show_event_waiters(std::ostream& os)
  {
    os << "PRINTING ALL PENDING EVENTS:\n";
    for(NodeID i = 0; i <= max_node_id; i++) {
      Node *n = &get_runtime()->nodes[i];
      // Iterate over all the events and get their implementations
      for (unsigned long j = 0; j < n->events.max_entries(); j++) {
	if (!n->events.has_entry(j))
	  continue;
	GenEventImpl *e = n->events.lookup_entry(j, i/*node*/);
	AutoHSLLock a2(e->mutex);
	
	// print anything with either local or remote waiters
	if(e->current_local_waiters.empty() &&
	   e->future_local_waiters.empty() &&
	   e->remote_waiters.empty())
	  continue;

	os << "Event " << e->me <<": gen=" << e->generation
	   << " subscr=" << e->gen_subscribed
	   << " local=" << e->current_local_waiters.size()
	   << "+" << e->future_local_waiters.size()
	   << " remote=" << e->remote_waiters.size() << "\n";
	for(std::vector<EventWaiter *>::const_iterator it = e->current_local_waiters.begin();
	    it != e->current_local_waiters.end();
	    it++) {
	  os << "  [" << (e->generation+1) << "] L:" << (*it) << " - ";
	  (*it)->print(os);
	  os << "\n";
	}
	for(std::map<EventImpl::gen_t, std::vector<EventWaiter *> >::const_iterator it = e->future_local_waiters.begin();
	    it != e->future_local_waiters.end();
	    it++) {
	  for(std::vector<EventWaiter *>::const_iterator it2 = it->second.begin();
	      it2 != it->second.end();
	      it2++) {
	    os << "  [" << (it->first) << "] L:" << (*it2) << " - ";
	    (*it2)->print(os);
	    os << "\n";
	  }
	}
	// for(std::map<Event::gen_t, NodeMask>::const_iterator it = e->remote_waiters.begin();
	//     it != e->remote_waiters.end();
	//     it++) {
	//   fprintf(f, "  [%d] R:", it->first);
	//   for(int k = 0; k < MAX_NUM_NODES; k++)
	//     if(it->second.is_set(k))
	// 	fprintf(f, " %d", k);
	//   fprintf(f, "\n");
	// }
      }
      for (unsigned long j = 0; j < n->barriers.max_entries(); j++) {
	if (!n->barriers.has_entry(j))
	  continue;
	BarrierImpl *b = n->barriers.lookup_entry(j, i/*node*/); 
	AutoHSLLock a2(b->mutex);
	// skip any barriers with no waiters
	if (b->generations.empty())
	  continue;

	os << "Barrier " << b->me << ": gen=" << b->generation
	   << " subscr=" << b->gen_subscribed << "\n";
	for (std::map<EventImpl::gen_t, BarrierImpl::Generation*>::const_iterator git = 
	       b->generations.begin(); git != b->generations.end(); git++) {
	  const std::vector<EventWaiter*> &waiters = git->second->local_waiters;
	  for (std::vector<EventWaiter*>::const_iterator it = 
		 waiters.begin(); it != waiters.end(); it++) {
	    os << "  [" << (git->first) << "] L:" << (*it) << " - ";
	    (*it)->print(os);
	    os << "\n";
	  }
	}
      }
    }

    // TODO - pending barriers
#if 0
    // // convert from events to barriers
    // fprintf(f,"PRINTING ALL PENDING EVENTS:\n");
    // for(int i = 0; i <= max_node_id; i++) {
    // 	Node *n = &get_runtime()->nodes[i];
    //   // Iterate over all the events and get their implementations
    //   for (unsigned long j = 0; j < n->events.max_entries(); j++) {
    //     if (!n->events.has_entry(j))
    //       continue;
    // 	  EventImpl *e = n->events.lookup_entry(j, i/*node*/);
    // 	  AutoHSLLock a2(e->mutex);
    
    // 	  // print anything with either local or remote waiters
    // 	  if(e->local_waiters.empty() && e->remote_waiters.empty())
    // 	    continue;

    //     fprintf(f,"Event " IDFMT ": gen=%d subscr=%d local=%zd remote=%zd\n",
    // 		  e->me.id, e->generation, e->gen_subscribed, 
    // 		  e->local_waiters.size(), e->remote_waiters.size());
    // 	  for(std::map<Event::gen_t, std::vector<EventWaiter *> >::iterator it = e->local_waiters.begin();
    // 	      it != e->local_waiters.end();
    // 	      it++) {
    // 	    for(std::vector<EventWaiter *>::iterator it2 = it->second.begin();
    // 		it2 != it->second.end();
    // 		it2++) {
    // 	      fprintf(f, "  [%d] L:%p ", it->first, *it2);
    // 	      (*it2)->print_info(f);
    // 	    }
    // 	  }
    // 	  for(std::map<Event::gen_t, NodeMask>::const_iterator it = e->remote_waiters.begin();
    // 	      it != e->remote_waiters.end();
    // 	      it++) {
    // 	    fprintf(f, "  [%d] R:", it->first);
    // 	    for(int k = 0; k < MAX_NUM_NODES; k++)
    // 	      if(it->second.is_set(k))
    // 		fprintf(f, " %d", k);
    // 	    fprintf(f, "\n");
    // 	  }
    // 	}
    // }
#endif

    os << "DONE\n";
    os.flush();
  }

  static void realm_show_events(int signal)
  {
    const char *filename = getenv("REALM_SHOW_EVENT_FILENAME");
    if(filename) {
      std::ofstream f(filename);
      get_runtime()->optable.print_operations(f);
      show_event_waiters(f);
    } else {
      get_runtime()->optable.print_operations(std::cout);
      show_event_waiters(std::cout);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Runtime
  //

    Runtime::Runtime(void)
      : impl(0)
    {
      // ok to construct extra ones - we will make sure only one calls init() though
    }

    /*static*/ Runtime Runtime::get_runtime(void)
    {
      Runtime r;
      // explicit namespace qualifier here due to name collision
      r.impl = Realm::get_runtime();
      return r;
    }

    // performs any network initialization and, critically, makes sure
    //  *argc and *argv contain the application's real command line
    //  (instead of e.g. mpi spawner information)
    bool Runtime::network_init(int *argc, char ***argv)
    {
      if(runtime_singleton != 0) {
	fprintf(stderr, "ERROR: cannot initialize more than one runtime at a time!\n");
	return false;
      }

      impl = new RuntimeImpl;
      runtime_singleton = static_cast<RuntimeImpl *>(impl);
      return static_cast<RuntimeImpl *>(impl)->network_init(argc, argv);
    }

    // configures the runtime from the provided command line - after this 
    //  call it is possible to create user events/reservations/etc, 
    //  perform registrations and query the machine model, but not spawn
    //  tasks or create instances
    bool Runtime::configure_from_command_line(int argc, char **argv)
    {
      assert(impl != 0);
      std::vector<std::string> cmdline;
      cmdline.reserve(argc);
      for(int i = 1; i < argc; i++)
	cmdline.push_back(argv[i]);
      return static_cast<RuntimeImpl *>(impl)->configure_from_command_line(cmdline);
    }

    bool Runtime::configure_from_command_line(std::vector<std::string> &cmdline,
					      bool remove_realm_args /*= false*/)
    {
      assert(impl != 0);
      if(remove_realm_args) {
	return static_cast<RuntimeImpl *>(impl)->configure_from_command_line(cmdline);
      } else {
	// pass in a copy so we don't mess up the original
	std::vector<std::string> cmdline_copy(cmdline);
	return static_cast<RuntimeImpl *>(impl)->configure_from_command_line(cmdline_copy);
      }
    }

    // starts up the runtime, allowing task/instance creation
    void Runtime::start(void)
    {
      assert(impl != 0);
      static_cast<RuntimeImpl *>(impl)->start();
    }

    // single-call version of the above three calls
    bool Runtime::init(int *argc, char ***argv)
    {
      if(!network_init(argc, argv)) return false;
      if(!configure_from_command_line(*argc, *argv)) return false;
      start();
      return true;
    }
    
    // this is now just a wrapper around Processor::register_task - consider switching to
    //  that
    bool Runtime::register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr)
    {
      assert(impl != 0);

      CodeDescriptor codedesc(taskptr);
      ProfilingRequestSet prs;
      std::set<Event> events;
      std::vector<ProcessorImpl *>& procs = ((RuntimeImpl *)impl)->nodes[my_node_id].processors;
      for(std::vector<ProcessorImpl *>::iterator it = procs.begin();
	  it != procs.end();
	  it++) {
	Event e = (*it)->me.register_task(taskid, codedesc, prs);
	events.insert(e);
      }

      Event merged = Event::merge_events(events);
      log_taskreg.info() << "waiting on event: " << merged;
      merged.wait();
      return true;
#if 0
      if(((RuntimeImpl *)impl)->task_table.count(taskid) > 0)
	return false;

      ((RuntimeImpl *)impl)->task_table[taskid] = taskptr;
      return true;
#endif
    }

    bool Runtime::register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->reduce_op_table.count(redop_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->reduce_op_table[redop_id] = redop->clone();
      return true;
    }

    bool Runtime::register_custom_serdez(CustomSerdezID serdez_id, const CustomSerdezUntyped *serdez)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->custom_serdez_table.count(serdez_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->custom_serdez_table[serdez_id] = serdez->clone();
      return true;
    }

    Event Runtime::collective_spawn(Processor target_proc, Processor::TaskFuncID task_id, 
				    const void *args, size_t arglen,
				    Event wait_on /*= Event::NO_EVENT*/, int priority /*= 0*/)
    {
      return ((RuntimeImpl *)impl)->collective_spawn(target_proc, task_id, args, arglen,
						     wait_on, priority);
    }

    Event Runtime::collective_spawn_by_kind(Processor::Kind target_kind, Processor::TaskFuncID task_id, 
					    const void *args, size_t arglen,
					    bool one_per_node /*= false*/,
					    Event wait_on /*= Event::NO_EVENT*/, int priority /*= 0*/)
    {
      return ((RuntimeImpl *)impl)->collective_spawn_by_kind(target_kind, task_id,
							     args, arglen,
							     one_per_node,
							     wait_on, priority);
    }

    void Runtime::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/,
                      bool background /*= false*/)
    {
      ((RuntimeImpl *)impl)->run(task_id, style, args, arglen, background);
    }

    class DeferredShutdown : public EventWaiter {
    public:
      DeferredShutdown(RuntimeImpl *_runtime, int _result_code);
      virtual ~DeferredShutdown(void);

      virtual bool event_triggered(Event e, bool poisoned);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      RuntimeImpl *runtime;
      int result_code;
    };

    DeferredShutdown::DeferredShutdown(RuntimeImpl *_runtime, int _result_code)
      : runtime(_runtime)
      , result_code(_result_code)
    {}

    DeferredShutdown::~DeferredShutdown(void)
    {}

    bool DeferredShutdown::event_triggered(Event e, bool poisoned)
    {
      // no real good way to deal with a poisoned shutdown precondition
      if(poisoned) {
	log_poison.fatal() << "HELP!  poisoned precondition for runtime shutdown";
	assert(false);
      }
      log_runtime.info() << "triggering deferred shutdown";
      runtime->shutdown(true, result_code);
      return true; // go ahead and delete us
    }

    void DeferredShutdown::print(std::ostream& os) const
    {
      os << "deferred shutdown";
    }

    Event DeferredShutdown::get_finish_event(void) const
    {
      return Event::NO_EVENT;
    }

    void Runtime::shutdown(Event wait_on /*= Event::NO_EVENT*/,
			   int result_code /*= 0*/)
    {
      log_runtime.info() << "shutdown requested - wait_on=" << wait_on;
      if(wait_on.has_triggered())
	((RuntimeImpl *)impl)->shutdown(true, result_code); // local request
      else
	EventImpl::add_waiter(wait_on,
			      new DeferredShutdown((RuntimeImpl *)impl,
						   result_code));
    }

    int Runtime::wait_for_shutdown(void)
    {
      int result = ((RuntimeImpl *)impl)->wait_for_shutdown();

      // after the shutdown, we nuke the RuntimeImpl
      delete ((RuntimeImpl *)impl);
      impl = 0;
      runtime_singleton = 0;

      return result;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteIDAllocator
  //

  Logger log_remote_id("remoteid");

  RemoteIDAllocator::RemoteIDAllocator(void)
  {}

  RemoteIDAllocator::~RemoteIDAllocator(void)
  {}

  void RemoteIDAllocator::set_request_size(ID::ID_Types id_type, int batch_size, int low_water_mark)
  {
    batch_sizes[id_type] = batch_size;
    low_water_marks[id_type] = low_water_mark;
  }

  void RemoteIDAllocator::make_initial_requests(void)
  {
    AutoHSLLock al(mutex);

    for(std::map<ID::ID_Types, int>::const_iterator it = batch_sizes.begin();
	it != batch_sizes.end();
	it++) {
      ID::ID_Types id_type = it->first;
      int batch_size = it->second;

      for(NodeID i = 0; i <= max_node_id; i++) {
	if(i == my_node_id) continue;

	reqs_in_flight[id_type].insert(i);

	RemoteIDRequestMessage::send_request(i, id_type, batch_size);
      }
    }
  }

  ID::IDType RemoteIDAllocator::get_remote_id(NodeID target, ID::ID_Types id_type)
  {
    assert(batch_sizes.count(id_type) > 0);

    ID::IDType id;
    bool request_more = false;
    {
      AutoHSLLock al(mutex);
      std::vector<std::pair<ID::IDType, ID::IDType> >& tgt_ranges = id_ranges[id_type][target];
      assert(!tgt_ranges.empty());
      id = tgt_ranges[0].first;
      if(tgt_ranges[0].first == tgt_ranges[0].second) {
	tgt_ranges.erase(tgt_ranges.begin());
      } else {
	tgt_ranges[0].first++;
      }
      if(tgt_ranges.empty() || 
	 ((tgt_ranges.size() == 1) && 
	  ((tgt_ranges[0].second - tgt_ranges[0].first) < (ID::IDType)low_water_marks[id_type]))) {
	// want to request more ids, as long as a request isn't already in flight
	if(reqs_in_flight[id_type].count(target) == 0) {
	  reqs_in_flight[id_type].insert(target);
	  request_more = true;
	}
      }
    }

    if(request_more)
      RemoteIDRequestMessage::send_request(target,
					   id_type,
					   batch_sizes[id_type]);

    log_remote_id.debug() << "assigned remote ID: target=" << target << " type=" << id_type << " id=" << id;
    return id;
  }

  void RemoteIDAllocator::add_id_range(NodeID target, ID::ID_Types id_type, ID::IDType first, ID::IDType last)
  {
    AutoHSLLock al(mutex);

    std::set<NodeID>::iterator it = reqs_in_flight[id_type].find(target);
    assert(it != reqs_in_flight[id_type].end());
    reqs_in_flight[id_type].erase(it);

    id_ranges[id_type][target].push_back(std::make_pair(first, last));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteIDRequestMessage
  //

  /*static*/ void RemoteIDRequestMessage::handle_request(RequestArgs args)
  {
    log_remote_id.debug() << "received remote id request: sender=" << args.sender
			  << " type=" << args.id_type << " count=" << args.count;

    int first = 0;
    int last = 0;
    ID first_id;
    ID last_id;

    switch(args.id_type) {
    case ID::ID_SPARSITY: {
      assert(0);
      //get_runtime()->local_sparsity_map_free_list->alloc_range(args.count, first, last);
      first_id = ID::make_sparsity(my_node_id, 0, first);
      last_id = ID::make_sparsity(my_node_id, 0, last);
      break;
    }
    default: assert(0);
    }

    RemoteIDResponseMessage::send_request(args.sender, args.id_type, first_id.id, last_id.id);
  }

  /*static*/ void RemoteIDRequestMessage::send_request(NodeID target, ID::ID_Types id_type, int count)
  {
    RequestArgs args;

    log_remote_id.debug() << "sending remote id request: target=" << target << " type=" << id_type << " count=" << count;
    args.sender = my_node_id;
    args.id_type = id_type;
    args.count = count;
    Message::request(target, args);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteIDResponseMessage
  //

  /*static*/ void RemoteIDResponseMessage::handle_request(RequestArgs args)
  {
    log_remote_id.debug() << "received remote id response: responder=" << args.responder
			  << " type=" << args.id_type
			  << " first=" << std::hex << args.first_id << " last=" << args.last_id << std::dec;

    get_runtime()->remote_id_allocator.add_id_range(args.responder,
						    args.id_type,
						    args.first_id,
						    args.last_id);
  }

  /*static*/ void RemoteIDResponseMessage::send_request(NodeID target, ID::ID_Types id_type,
							ID::IDType first_id, ID::IDType last_id)
  {
    RequestArgs args;

    log_remote_id.debug() << "sending remote id response: target=" << target
			 << " type=" << id_type 
			 << " first=" << std::hex << first_id << " last=" << last_id << std::dec;

    args.responder = my_node_id;
    args.id_type = id_type;
    args.first_id = first_id;
    args.last_id = last_id;

    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreModule
  //

  namespace Config {
    // if true, worker threads that might have used user-level thread switching
    //  fall back to kernel threading
    bool force_kernel_threads = false;
  };

  CoreModule::CoreModule(void)
    : Module("core")
    , num_cpu_procs(1), num_util_procs(1), num_io_procs(0)
    , concurrent_io_threads(1)  // Legion does not support values > 1 right now
    , sysmem_size_in_mb(512), stack_size_in_mb(2)
  {}

  CoreModule::~CoreModule(void)
  {}

  /*static*/ Module *CoreModule::create_module(RuntimeImpl *runtime,
					       std::vector<std::string>& cmdline)
  {
    CoreModule *m = new CoreModule;

    // parse command line arguments
    CommandLineParser cp;
    cp.add_option_int("-ll:cpu", m->num_cpu_procs)
      .add_option_int("-ll:util", m->num_util_procs)
      .add_option_int("-ll:io", m->num_io_procs)
      .add_option_int("-ll:concurrent_io", m->concurrent_io_threads)
      .add_option_int("-ll:csize", m->sysmem_size_in_mb)
      .add_option_int("-ll:stacksize", m->stack_size_in_mb, true /*keep*/)
      .parse_command_line(cmdline);

    return m;
  }

  // create any memories provided by this module (default == do nothing)
  //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
  void CoreModule::create_memories(RuntimeImpl *runtime)
  {
    Module::create_memories(runtime);

    if(sysmem_size_in_mb > 0) {
      Memory m = runtime->next_local_memory_id();
      MemoryImpl *mi = new LocalCPUMemory(m, sysmem_size_in_mb << 20);
      runtime->add_memory(mi);
    }
  }

  // create any processors provided by the module (default == do nothing)
  //  (each new ProcessorImpl should use a Processor from
  //   RuntimeImpl::next_local_processor_id)
  void CoreModule::create_processors(RuntimeImpl *runtime)
  {
    Module::create_processors(runtime);

    for(int i = 0; i < num_util_procs; i++) {
      Processor p = runtime->next_local_processor_id();
      ProcessorImpl *pi = new LocalUtilityProcessor(p, runtime->core_reservation_set(),
						    stack_size_in_mb << 20,
						    Config::force_kernel_threads);
      runtime->add_processor(pi);
    }

    for(int i = 0; i < num_io_procs; i++) {
      Processor p = runtime->next_local_processor_id();
      ProcessorImpl *pi = new LocalIOProcessor(p, runtime->core_reservation_set(),
					       stack_size_in_mb << 20,
					       concurrent_io_threads);
      runtime->add_processor(pi);
    }

    for(int i = 0; i < num_cpu_procs; i++) {
      Processor p = runtime->next_local_processor_id();
      ProcessorImpl *pi = new LocalCPUProcessor(p, runtime->core_reservation_set(),
						stack_size_in_mb << 20,
						Config::force_kernel_threads);
      runtime->add_processor(pi);
    }
  }

  // create any DMA channels provided by the module (default == do nothing)
  void CoreModule::create_dma_channels(RuntimeImpl *runtime)
  {
    Module::create_dma_channels(runtime);

    // no dma channels
  }

  // create any code translators provided by the module (default == do nothing)
  void CoreModule::create_code_translators(RuntimeImpl *runtime)
  {
    Module::create_code_translators(runtime);

#ifdef REALM_USE_DLFCN
    runtime->add_code_translator(new DSOCodeTranslator);
#endif
  }

  // clean up any common resources created by the module - this will be called
  //  after all memories/processors/etc. have been shut down and destroyed
  void CoreModule::cleanup(void)
  {
    // nothing to clean up

    Module::cleanup();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RuntimeImpl
  //

    RuntimeImpl *runtime_singleton = 0;

  // these should probably be member variables of RuntimeImpl?
    static size_t stack_size_in_mb;
  
    RuntimeImpl::RuntimeImpl(void)
      : machine(0), 
#ifdef NODE_LOGGING
	prefix("."),
#endif
	nodes(0), global_memory(0),
	local_event_free_list(0), local_barrier_free_list(0),
	local_reservation_free_list(0),
	local_proc_group_free_list(0),
	//local_sparsity_map_free_list(0),
	run_method_called(false),
	shutdown_requested(false),
	shutdown_result_code(0),
	shutdown_condvar(shutdown_mutex),
	core_map(0), core_reservations(0),
	sampling_profiler(true /*system default*/),
	num_local_memories(0), num_local_ib_memories(0),
	num_local_processors(0),
#ifndef USE_GASNET
	nongasnet_regmem_base(0),
	nongasnet_reg_ib_mem_base(0),
#endif
	module_registrar(this)
    {
      machine = new MachineImpl;
    }

    RuntimeImpl::~RuntimeImpl(void)
    {
      delete machine;
      delete core_reservations;
      delete core_map;

      delete_container_contents(reduce_op_table);
      delete_container_contents(custom_serdez_table);
    }

    Memory RuntimeImpl::next_local_memory_id(void)
    {
      Memory m = ID::make_memory(my_node_id,
				 num_local_memories++).convert<Memory>();
      return m;
    }

    Memory RuntimeImpl::next_local_ib_memory_id(void)
    {
      Memory m = ID::make_ib_memory(my_node_id,
                                    num_local_ib_memories++).convert<Memory>();
      return m;
    }

    Processor RuntimeImpl::next_local_processor_id(void)
    {
      Processor p = ID::make_processor(my_node_id, 
				       num_local_processors++).convert<Processor>();
      return p;
    }

    void RuntimeImpl::add_memory(MemoryImpl *m)
    {
      // right now expect this to always be for the current node and the next memory ID
      ID id(m->me);
      assert(id.memory.owner_node == my_node_id);
      assert(id.memory.mem_idx == nodes[my_node_id].memories.size());

      nodes[my_node_id].memories.push_back(m);
    }

    void RuntimeImpl::add_ib_memory(MemoryImpl *m)
    {
      // right now expect this to always be for the current node and the next memory ID
      ID id(m->me);
      assert(id.memory.owner_node == my_node_id);
      assert(id.memory.mem_idx == nodes[my_node_id].ib_memories.size());

      nodes[my_node_id].ib_memories.push_back(m);
    }

    void RuntimeImpl::add_processor(ProcessorImpl *p)
    {
      // right now expect this to always be for the current node and the next processor ID
      ID id(p->me);
      assert(id.proc.owner_node == my_node_id);
      assert(id.proc.proc_idx == nodes[my_node_id].processors.size());

      nodes[my_node_id].processors.push_back(p);
    }

    void RuntimeImpl::add_dma_channel(DMAChannel *c)
    {
      nodes[c->node].dma_channels.push_back(c);
    }

    void RuntimeImpl::add_code_translator(CodeTranslator *t)
    {
      code_translators.push_back(t);
    }

    void RuntimeImpl::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity& pma)
    {
      machine->add_proc_mem_affinity(pma);
    }

    void RuntimeImpl::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity& mma)
    {
      machine->add_mem_mem_affinity(mma);
    }

    CoreReservationSet& RuntimeImpl::core_reservation_set(void)
    {
      assert(core_reservations);
      return *core_reservations;
    }

    const std::vector<CodeTranslator *>& RuntimeImpl::get_code_translators(void) const
    {
      return code_translators;
    }

    static void add_proc_mem_affinities(MachineImpl *machine,
					const std::set<Processor>& procs,
					const std::set<Memory>& mems,
					int bandwidth,
					int latency)
    {
      for(std::set<Processor>::const_iterator it1 = procs.begin();
	  it1 != procs.end();
	  it1++) 
	for(std::set<Memory>::const_iterator it2 = mems.begin();
	    it2 != mems.end();
	    it2++) {
	  std::vector<Machine::ProcessorMemoryAffinity> pmas;
	  machine->get_proc_mem_affinity(pmas, *it1, *it2);
	  if(!pmas.empty()) continue;
	  log_runtime.debug() << "adding missing affinity: " << *it1 << " " << *it2 << " " << bandwidth << " " << latency;
	  Machine::ProcessorMemoryAffinity pma;
	  pma.p = *it1;
	  pma.m = *it2;
	  pma.bandwidth = bandwidth;
	  pma.latency = latency;
	  machine->add_proc_mem_affinity(pma);
	}
    }

    static void add_mem_mem_affinities(MachineImpl *machine,
				       const std::set<Memory>& mems1,
				       const std::set<Memory>& mems2,
				       int bandwidth,
				       int latency)
    {
      for(std::set<Memory>::const_iterator it1 = mems1.begin();
	  it1 != mems1.end();
	  it1++) 
	for(std::set<Memory>::const_iterator it2 = mems2.begin();
	    it2 != mems2.end();
	    it2++) {
	  std::vector<Machine::MemoryMemoryAffinity> mmas;
	  machine->get_mem_mem_affinity(mmas, *it1, *it2);
	  if(!mmas.empty()) continue;
	  log_runtime.debug() << "adding missing affinity: " << *it1 << " " << *it2 << " " << bandwidth << " " << latency;
	  Machine::MemoryMemoryAffinity mma;
	  mma.m1 = *it1;
	  mma.m2 = *it2;
	  mma.bandwidth = bandwidth;
	  mma.latency = latency;
	  machine->add_mem_mem_affinity(mma);
	}
    }

    bool RuntimeImpl::network_init(int *argc, char ***argv)
    {
      DetailedTimer::init_timers();

      // gasnet_init() must be called before parsing command line arguments, as some
      //  spawners (e.g. the ssh spawner for gasnetrun_ibv) start with bogus args and
      //  fetch the real ones from somewhere during gasnet_init()

#ifdef USE_GASNET
      // SJT: WAR for issue on Titan with duplicate cookies on Gemini
      //  communication domains
      char *orig_pmi_gni_cookie = getenv("PMI_GNI_COOKIE");
      if(orig_pmi_gni_cookie) {
	char new_pmi_gni_cookie[32];
	snprintf(new_pmi_gni_cookie, 32, "%d", 1+atoi(orig_pmi_gni_cookie));
	setenv("PMI_GNI_COOKIE", new_pmi_gni_cookie, 1 /*overwrite*/);
      }
      // SJT: another GASNET workaround - if we don't have GASNET_IB_SPAWNER set, assume it was MPI
      // (This is called GASNET_IB_SPAWNER for versions <= 1.24 and GASNET_SPAWNER for versions >= 1.26)
      if(!getenv("GASNET_IB_SPAWNER") && !getenv("GASNET_SPAWNER")) {
	setenv("GASNET_IB_SPAWNER", "mpi", 0 /*no overwrite*/);
	setenv("GASNET_SPAWNER", "mpi", 0 /*no overwrite*/);
      }

      // and one more... disable GASNet's probing of pinnable memory - it's
      //  painfully slow on most systems (the gemini conduit doesn't probe
      //  at all, so it's ok)
      // we can do this because in gasnet_attach() we will ask for exactly as
      //  much as we need, and we can detect failure there if that much memory
      //  doesn't actually exist
      // inconveniently, we have to set a PHYSMEM_MAX before we call
      //  gasnet_init and we don't have our argc/argv until after, so we can't
      //  set PHYSMEM_MAX correctly, but setting it to something really big to
      //  prevent all the early checks from failing gets us to that final actual
      //  alloc/pin in gasnet_attach ok
      {
	// the only way to control this is with environment variables, so set
	//  them unless the user has already set them (in which case, we assume
	//  they know what they're doing)
	// do handle the case where NOPROBE is set to 1, but PHYSMEM_MAX isn't
	const char *e = getenv("GASNET_PHYSMEM_NOPROBE");
	if(!e || (atoi(e) > 0)) {
	  if(!e)
	    setenv("GASNET_PHYSMEM_NOPROBE", "1", 0 /*no overwrite*/);
	  if(!getenv("GASNET_PHYSMEM_MAX")) {
	    // just because it's fun to read things like this 20 years later:
	    // "nobody will ever build a system with more than 1 TB of RAM..."
	    setenv("GASNET_PHYSMEM_MAX", "1T", 0 /*no overwrite*/);
	  }
	}
      }

      // and yet another GASNet workaround: the Infiniband conduit seems to
      //  have a problem with AMRDMA mode, consuming receive buffers even for
      //  request targets that are in AMRDMA mode - disable the mode by default
#ifdef GASNET_CONDUIT_IBV
      if(!getenv("GASNET_AMRDMA_MAX_PEERS"))
        setenv("GASNET_AMRDMA_MAX_PEERS", "0", 0 /*no overwrite*/);
#endif

#ifdef DEBUG_REALM_STARTUP
      { // we don't have rank IDs yet, so everybody gets to spew
        char s[80];
        gethostname(s, 79);
        strcat(s, " enter gasnet_init");
        TimeStamp ts(s, false);
        fflush(stdout);
      }
#endif
      CHECK_GASNET( gasnet_init(argc, argv) );
      my_node_id = gasnet_mynode();
      max_node_id = gasnet_nodes() - 1;
#ifdef DEBUG_REALM_STARTUP
      { // once we're convinced there isn't skew here, reduce this to rank 0
        char s[80];
        gethostname(s, 79);
        strcat(s, " exit gasnet_init");
        TimeStamp ts(s, false);
        fflush(stdout);
      }
#endif
#endif

      // TODO: this is here to match old behavior, but it'd probably be
      //  better to have REALM_DEFAULT_ARGS only be visible to Realm...

      // if the REALM_DEFAULT_ARGS environment variable is set, these arguments
      //  are inserted at the FRONT of the command line (so they may still be
      //  overridden by actual command line args)
      {
	const char *e = getenv("REALM_DEFAULT_ARGS");
	if(e) {
	  // find arguments first, then construct new argv of right size
	  std::vector<const char *> starts, ends;
	  while(*e) {
	    if(isspace(*e)) { e++; continue; }
	    if(*e == '\'') {
	      // single quoted string
	      e++; assert(*e);
	      starts.push_back(e);
	      // read until next single quote
	      while(*e && (*e != '\'')) e++;
	      ends.push_back(e++);
	      assert(!*e || isspace(*e));
	      continue;
	    }
	    if(*e == '\"') {
	      // double quoted string
	      e++; assert(*e);
	      starts.push_back(e);
	      // read until next double quote
	      while(*e && (*e != '\"')) e++;
	      ends.push_back(e++);
	      assert(!*e || isspace(*e));
	      continue;
	    }
	    // no quotes - just take until next whitespace
	    starts.push_back(e);
	    while(*e && !isspace(*e)) e++;
	    ends.push_back(e);
	  }
	  if(!starts.empty()) {
	    int new_argc = *argc + starts.size();
	    char **new_argv = (char **)(malloc((new_argc + 1) * sizeof(char *)));
	    // new args go after argv[0] and anything that looks like a
	    //  positional argument (i.e. doesn't start with -)
	    int before_new = 0;
	    while(before_new < *argc) {
	      if((before_new > 0) && ((*argv)[before_new][0] == '-'))
		break;
	      new_argv[before_new] = (*argv)[before_new];
	      before_new++;
	    }
	    for(size_t i = 0; i < starts.size(); i++)
	      new_argv[i + before_new] = strndup(starts[i], ends[i] - starts[i]);
	    for(int i = before_new; i < *argc; i++)
	      new_argv[i + starts.size()] = (*argv)[i];
	    new_argv[new_argc] = 0;

	    *argc = new_argc;
	    *argv = new_argv;
	  }
	}
      }

      return true;
    }

    bool RuntimeImpl::configure_from_command_line(std::vector<std::string> &cmdline)
    {
      // very first thing - let the logger initialization happen
      Logger::configure_from_cmdline(cmdline);

      // start up the threading subsystem - modules will likely want threads
      if(!Threading::initialize()) exit(1);

      // now load modules
      module_registrar.create_static_modules(cmdline, modules);
      module_registrar.create_dynamic_modules(cmdline, modules);

      PartitioningOpQueue::configure_from_cmdline(cmdline);

      // low-level runtime parameters
#ifdef USE_GASNET
      size_t gasnet_mem_size_in_mb = 256;
      size_t reg_ib_mem_size_in_mb = 256;
#else
      size_t gasnet_mem_size_in_mb = 0;
      size_t reg_ib_mem_size_in_mb = 64; // for transposes/serdez
#endif
      size_t reg_mem_size_in_mb = 0;
      size_t disk_mem_size_in_mb = 0;
      // Static variable for stack size since we need to 
      // remember it when we launch threads in run 
      stack_size_in_mb = 2;
      //unsigned cpu_worker_threads = 1;
      unsigned dma_worker_threads = 1;
      unsigned active_msg_worker_threads = 1;
      unsigned active_msg_handler_threads = 1;
#ifdef EVENT_TRACING
      size_t   event_trace_block_size = 1 << 20;
      double   event_trace_exp_arrv_rate = 1e3;
#endif
#ifdef LOCK_TRACING
      size_t   lock_trace_block_size = 1 << 20;
      double   lock_trace_exp_arrv_rate = 1e2;
#endif
      // should local proc threads get dedicated cores?
      bool dummy_reservation_ok = true;
      bool show_reservations = false;
      // are hyperthreads considered to share a physical core
      bool hyperthread_sharing = true;
      bool pin_dma_threads = false;

      CommandLineParser cp;
      cp.add_option_int("-ll:gsize", gasnet_mem_size_in_mb)
	.add_option_int("-ll:rsize", reg_mem_size_in_mb)
	.add_option_int("-ll:ib_rsize", reg_ib_mem_size_in_mb)
	.add_option_int("-ll:dsize", disk_mem_size_in_mb)
	.add_option_int("-ll:stacksize", stack_size_in_mb)
	.add_option_int("-ll:dma", dma_worker_threads)
        .add_option_bool("-ll:pin_dma", pin_dma_threads)
	.add_option_int("-ll:amsg", active_msg_worker_threads)
	.add_option_int("-ll:ahandlers", active_msg_handler_threads)
	.add_option_int("-ll:dummy_rsrv_ok", dummy_reservation_ok)
	.add_option_bool("-ll:show_rsrv", show_reservations)
	.add_option_int("-ll:ht_sharing", hyperthread_sharing);

      std::string event_trace_file, lock_trace_file;

      cp.add_option_string("-ll:eventtrace", event_trace_file)
	.add_option_string("-ll:locktrace", lock_trace_file);

#ifdef NODE_LOGGING
      cp.add_option_string("-ll:prefix", RuntimeImpl::prefix);
#else
      std::string dummy_prefix;
      cp.add_option_string("-ll:prefix", dummy_prefix);
#endif

      cp.add_option_int("-realm:eventloopcheck", Config::event_loop_detection_limit);
      cp.add_option_bool("-ll:force_kthreads", Config::force_kernel_threads);
      cp.add_option_bool("-ll:frsrv_fallback", Config::use_fast_reservation_fallback);

      bool cmdline_ok = cp.parse_command_line(cmdline);

      if(!cmdline_ok) {
	fprintf(stderr, "ERROR: failure parsing command line options\n");
	exit(1);
      }

#ifndef EVENT_TRACING
      if(!event_trace_file.empty()) {
	fprintf(stderr, "WARNING: event tracing requested, but not enabled at compile time!\n");
      }
#endif

#ifndef LOCK_TRACING
      if(!lock_trace_file.empty()) {
          fprintf(stderr, "WARNING: lock tracing requested, but not enabled at compile time!\n");
      }
#endif

#ifndef NODE_LOGGING
      if(!dummy_prefix.empty()) {
	fprintf(stderr,"WARNING: prefix set, but NODE_LOGGING not enabled at compile time!\n");
      }
#endif

      // Check that we have enough resources for the number of nodes we are using
      if (max_node_id >= MAX_NUM_NODES)
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but runtime is configured "
                       "for at most %d nodes. Update the 'MAX_NUM_NODES' macro "
                       "in legion_config.h", max_node_id+1, MAX_NUM_NODES);
        exit(1);
      }
      if (max_node_id > (NodeID)(ID::MAX_NODE_ID))
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes. Update the allocation "
		       "of bits in ID", max_node_id+1, (ID::MAX_NODE_ID + 1));
        exit(1);
      }

      // if compiled in and not explicitly disabled, check our user threading
      //  support
#ifdef REALM_USE_USER_THREADS
      if(!Config::force_kernel_threads) {
        bool ok = Thread::test_user_switch_support();
        if(!ok) {
          log_runtime.warning() << "user switching not working - falling back to kernel threads";
          Config::force_kernel_threads = true;
        }
      }
#endif

      core_map = CoreMap::discover_core_map(hyperthread_sharing);
      core_reservations = new CoreReservationSet(core_map);

      sampling_profiler.configure_from_cmdline(cmdline, *core_reservations);

      // initialize barrier timestamp
      BarrierImpl::barrier_adjustment_timestamp = (((Barrier::timestamp_t)(my_node_id)) << BarrierImpl::BARRIER_TIMESTAMP_NODEID_SHIFT) + 1;

      NodeAnnounceMessage::Message::add_handler_entries("Node Announce AM");
      SpawnTaskMessage::Message::add_handler_entries("Spawn Task AM");
      LockRequestMessage::Message::add_handler_entries("Lock Request AM");
      LockReleaseMessage::Message::add_handler_entries("Lock Release AM");
      LockGrantMessage::Message::add_handler_entries("Lock Grant AM");
      EventSubscribeMessage::Message::add_handler_entries("Event Subscribe AM");
      EventTriggerMessage::Message::add_handler_entries("Event Trigger AM");
      EventUpdateMessage::Message::add_handler_entries("Event Update AM");
      RemoteMemAllocRequest::Request::add_handler_entries("Remote Memory Allocation Request AM");
      RemoteMemAllocRequest::Response::add_handler_entries("Remote Memory Allocation Response AM");
      //CreateInstanceRequest::Request::add_handler_entries("Create Instance Request AM");
      //CreateInstanceRequest::Response::add_handler_entries("Create Instance Response AM");
      RemoteCopyMessage::add_handler_entries("Remote Copy AM");
      RemoteFillMessage::add_handler_entries("Remote Fill AM");
#ifdef DETAILED_TIMING
      TimerDataRequestMessage::Message::add_handler_entries("Roll-up Request AM");
      TimerDataResponseMessage::Message::add_handler_entries("Roll-up Data AM");
      ClearTimersMessage::Message::add_handler_entries("Clear Timer Request AM");
#endif
      //DestroyInstanceMessage::Message::add_handler_entries("Destroy Instance AM");
      RemoteWriteMessage::Message::add_handler_entries("Remote Write AM");
      RemoteReduceMessage::Message::add_handler_entries("Remote Reduce AM");
      RemoteSerdezMessage::Message::add_handler_entries("Remote Serdez AM");
      RemoteWriteFenceMessage::Message::add_handler_entries("Remote Write Fence AM");
      RemoteWriteFenceAckMessage::Message::add_handler_entries("Remote Write Fence Ack AM");
      DestroyLockMessage::Message::add_handler_entries("Destroy Lock AM");
      RemoteReduceListMessage::Message::add_handler_entries("Remote Reduction List AM");
      RuntimeShutdownMessage::Message::add_handler_entries("Machine Shutdown AM");
      BarrierAdjustMessage::Message::add_handler_entries("Barrier Adjust AM");
      BarrierSubscribeMessage::Message::add_handler_entries("Barrier Subscribe AM");
      BarrierTriggerMessage::Message::add_handler_entries("Barrier Trigger AM");
      BarrierMigrationMessage::Message::add_handler_entries("Barrier Migration AM");
      MetadataRequestMessage::Message::add_handler_entries("Metadata Request AM");
      MetadataResponseMessage::Message::add_handler_entries("Metadata Response AM");
      MetadataInvalidateMessage::Message::add_handler_entries("Metadata Invalidate AM");
      MetadataInvalidateAckMessage::Message::add_handler_entries("Metadata Inval Ack AM");
      XferDesRemoteWriteMessage::Message::add_handler_entries("XferDes Remote Write AM");
      XferDesRemoteWriteAckMessage::Message::add_handler_entries("XferDes Remote Write Ack AM");
      XferDesCreateMessage::Message::add_handler_entries("Create XferDes Request AM");
      XferDesDestroyMessage::Message::add_handler_entries("Destroy XferDes Request AM");
      NotifyXferDesCompleteMessage::Message::add_handler_entries("Notify XferDes Completion Request AM");
      UpdateBytesWriteMessage::Message::add_handler_entries("Update Bytes Write AM");
      UpdateBytesReadMessage::Message::add_handler_entries("Update Bytes Read AM");
      RegisterTaskMessage::Message::add_handler_entries("Register Task AM");
      RegisterTaskCompleteMessage::Message::add_handler_entries("Register Task Complete AM");
      RemoteMicroOpMessage::Message::add_handler_entries("Remote Micro Op AM");
      RemoteMicroOpCompleteMessage::Message::add_handler_entries("Remote Micro Op Complete AM");
      RemoteSparsityContribMessage::Message::add_handler_entries("Remote Sparsity Contrib AM");
      RemoteSparsityRequestMessage::Message::add_handler_entries("Remote Sparsity Request AM");
      ApproxImageResponseMessage::Message::add_handler_entries("Approx Image Response AM");
      SetContribCountMessage::Message::add_handler_entries("Set Contrib Count AM");
      RemoteIDRequestMessage::Message::add_handler_entries("Remote ID Request AM");
      RemoteIDResponseMessage::Message::add_handler_entries("Remote ID Response AM");
      RemoteIBAllocRequestAsync::Message::add_handler_entries("Remote IB Alloc Request AM");
      RemoteIBAllocResponseAsync::Message::add_handler_entries("Remote IB Alloc Response AM");
      RemoteIBFreeRequestAsync::Message::add_handler_entries("Remote IB Free Request AM");
      MemStorageAllocRequest::Message::add_handler_entries("Memory Storage Alloc Request");
      MemStorageAllocResponse::Message::add_handler_entries("Memory Storage Alloc Response");
      MemStorageReleaseRequest::Message::add_handler_entries("Memory Storage Release Request");
      MemStorageReleaseResponse::Message::add_handler_entries("Memory Storage Release Response");
      //TestMessage::add_handler_entries("Test AM");
      //TestMessage2::add_handler_entries("Test 2 AM");

      nodes = new Node[max_node_id + 1];

      // create allocators for local node events/locks/index spaces - do this before we start handling
      //  active messages
      {
	Node& n = nodes[my_node_id];
	local_event_free_list = new EventTableAllocator::FreeList(n.events, my_node_id);
	local_barrier_free_list = new BarrierTableAllocator::FreeList(n.barriers, my_node_id);
	local_reservation_free_list = new ReservationTableAllocator::FreeList(n.reservations, my_node_id);
	local_proc_group_free_list = new ProcessorGroupTableAllocator::FreeList(n.proc_groups, my_node_id);

	local_sparsity_map_free_lists.resize(max_node_id + 1);
	for(NodeID i = 0; i <= max_node_id; i++) {
	  nodes[i].sparsity_maps.resize(max_node_id + 1, 0);
	  DynamicTable<SparsityMapTableAllocator> *m = new DynamicTable<SparsityMapTableAllocator>;
	  nodes[i].sparsity_maps[my_node_id] = m;
	  local_sparsity_map_free_lists[i] = new SparsityMapTableAllocator::FreeList(*m, i /*owner_node*/);
	}
      }

      init_endpoints(gasnet_mem_size_in_mb, reg_mem_size_in_mb, reg_ib_mem_size_in_mb,
		     *core_reservations,
		     cmdline);

      // now that we've done all of our argument parsing, scan through what's
      //  left and see if anything starts with -ll: - probably a misspelled
      //  argument
      for(std::vector<std::string>::const_iterator it = cmdline.begin();
	  it != cmdline.end();
	  it++)
	if(it->compare(0, 4, "-ll:") == 0) {
	  fprintf(stderr, "ERROR: unrecognized lowlevel option: %s\n", it->c_str());
          assert(0);
	}

#ifndef USE_GASNET
      // network initialization is also responsible for setting the "zero_time"
      //  for relative timing - no synchronization necessary in non-gasnet case
      Realm::Clock::set_zero_time();
#endif

#ifdef USE_GASNET
      // Put this here so that it complies with the GASNet specification and
      // doesn't make any calls between gasnet_init and gasnet_attach
      gasnet_set_waitmode(GASNET_WAIT_BLOCK);
#endif

      //remote_id_allocator.set_request_size(ID::ID_SPARSITY, 4096, 3072);
      remote_id_allocator.make_initial_requests();

#ifdef DEADLOCK_TRACE
      next_thread = 0;
      signaled_threads = 0;
      signal(SIGTERM, deadlock_catch);
      signal(SIGINT, deadlock_catch);
#endif
      if ((getenv("LEGION_FREEZE_ON_ERROR") != NULL) ||
          (getenv("REALM_FREEZE_ON_ERROR") != NULL)) {
        signal(SIGSEGV, realm_freeze);
        signal(SIGABRT, realm_freeze);
        signal(SIGFPE,  realm_freeze);
        signal(SIGILL,  realm_freeze);
        signal(SIGBUS,  realm_freeze);
      } else if ((getenv("REALM_BACKTRACE") != NULL) ||
                 (getenv("LEGION_BACKTRACE") != NULL)) {
        signal(SIGSEGV, realm_backtrace);
        signal(SIGABRT, realm_backtrace);
        signal(SIGFPE,  realm_backtrace);
        signal(SIGILL,  realm_backtrace);
        signal(SIGBUS,  realm_backtrace);
      }

      // debugging tool to dump realm event graphs after a fixed delay
      //  (easier than actually detecting a hang)
      {
	const char *e = getenv("REALM_SHOW_EVENT_WAITERS");
	if(e) {
	  const char *pos;
	  int delay = strtol(e, (char **)&pos, 10);
	  assert(delay > 0);
	  if(*pos == '+')
	    delay += my_node_id * atoi(pos + 1);
	  log_runtime.info() << "setting show_event alarm for " << delay << " seconds";
	  signal(SIGALRM, realm_show_events);
	  alarm(delay);
	}
      }
      
      start_polling_threads(active_msg_worker_threads);

      start_handler_threads(active_msg_handler_threads,
			    *core_reservations,
			    stack_size_in_mb << 20);

#ifdef USE_GASNET
      // this needs to happen after init_endpoints
      gasnet_coll_init(0, 0, 0, 0, 0);
#endif

      start_dma_worker_threads(dma_worker_threads,
			       *core_reservations);

      PartitioningOpQueue::start_worker_threads(*core_reservations);

#ifdef EVENT_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save event info
      Tracer<EventTraceItem>::init_trace(event_trace_block_size,
                                         event_trace_exp_arrv_rate);
#endif
#ifdef LOCK_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save lock info
      Tracer<LockTraceItem>::init_trace(lock_trace_block_size,
                                        lock_trace_exp_arrv_rate);
#endif
	
      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->initialize(this);

      //gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
      //CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

      if(gasnet_mem_size_in_mb > 0)
	// use an 'owner_node' of all 1's for this
        // SJT: actually, go back to an owner node of 0 and memory_idx of all 1's for now
	global_memory = new GASNetMemory(ID::make_memory(0, -1U).convert<Memory>(), gasnet_mem_size_in_mb << 20);
      else
	global_memory = 0;

      Node *n = &nodes[my_node_id];

      // create memories and processors for all loaded modules
      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_memories(this);

      LocalCPUMemory *regmem;
      if(reg_mem_size_in_mb > 0) {
#ifdef USE_GASNET
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[max_node_id + 1];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, max_node_id + 1) );
	char *regmem_base = ((char *)(seginfos[my_node_id].addr)) + (gasnet_mem_size_in_mb << 20);
	delete[] seginfos;
#else
	nongasnet_regmem_base = malloc(reg_mem_size_in_mb << 20);
	assert(nongasnet_regmem_base != 0);
	char *regmem_base = static_cast<char *>(nongasnet_regmem_base);
#endif
	Memory m = get_runtime()->next_local_memory_id();
	regmem = new LocalCPUMemory(m,
				    reg_mem_size_in_mb << 20,
				    regmem_base,
				    true);
	get_runtime()->add_memory(regmem);
      } else
	regmem = 0;

      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_processors(this);

      LocalCPUMemory *reg_ib_mem;
      if(reg_ib_mem_size_in_mb > 0) {
#ifdef USE_GASNET
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[max_node_id + 1];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, max_node_id + 1) );
	char *reg_ib_mem_base = ((char *)(seginfos[my_node_id].addr)) + (gasnet_mem_size_in_mb << 20)
                                + (reg_mem_size_in_mb << 20);
	delete[] seginfos;
#else
	nongasnet_reg_ib_mem_base = malloc(reg_ib_mem_size_in_mb << 20);
	assert(nongasnet_reg_ib_mem_base != 0);
	char *reg_ib_mem_base = static_cast<char *>(nongasnet_reg_ib_mem_base);
#endif
	Memory m = get_runtime()->next_local_ib_memory_id();
	reg_ib_mem = new LocalCPUMemory(m,
				        reg_ib_mem_size_in_mb << 20,
				        reg_ib_mem_base,
				        true);
	get_runtime()->add_ib_memory(reg_ib_mem);
      } else
        reg_ib_mem = 0;

      // create local disk memory
      DiskMemory *diskmem;
      if(disk_mem_size_in_mb > 0) {
        char file_name[30];
        sprintf(file_name, "disk_file%d.tmp", my_node_id);
        Memory m = get_runtime()->next_local_memory_id();
        diskmem = new DiskMemory(m,
                                 disk_mem_size_in_mb << 20,
                                 std::string(file_name));
        get_runtime()->add_memory(diskmem);
      } else
        diskmem = 0;

      FileMemory *filemem;
      filemem = new FileMemory(get_runtime()->next_local_memory_id());
      get_runtime()->add_memory(filemem);

      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_dma_channels(this);

      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_code_translators(this);
      
      // start dma system at the very ending of initialization
      // since we need list of local gpus to create channels
      start_dma_system(dma_worker_threads,
		       pin_dma_threads, 100
		       ,*core_reservations);

      // now that we've created all the processors/etc., we can try to come up with core
      //  allocations that satisfy everybody's requirements - this will also start up any
      //  threads that have already been requested
      bool ok = core_reservations->satisfy_reservations(dummy_reservation_ok);
      if(ok) {
	if(show_reservations) {
	  std::cout << *core_map << std::endl;
	  core_reservations->report_reservations(std::cout);
	}
      } else {
	printf("HELP!  Could not satisfy all core reservations!\n");
	exit(1);
      }

      {
        // iterate over all local processors and add affinities for them
	// all of this should eventually be moved into appropriate modules
	std::map<Processor::Kind, std::set<Processor> > procs_by_kind;

	for(std::vector<ProcessorImpl *>::const_iterator it = n->processors.begin();
	    it != n->processors.end();
	    it++)
	  if(*it) {
	    Processor p = (*it)->me;
	    Processor::Kind k = (*it)->me.kind();

	    procs_by_kind[k].insert(p);
	  }

	// now iterate over memories too
	std::map<Memory::Kind, std::set<Memory> > mems_by_kind;
	for(std::vector<MemoryImpl *>::const_iterator it = n->memories.begin();
	    it != n->memories.end();
	    it++)
	  if(*it) {
	    Memory m = (*it)->me;
	    Memory::Kind k = (*it)->me.kind();

	    mems_by_kind[k].insert(m);
	  }

	if(global_memory)
	  mems_by_kind[Memory::GLOBAL_MEM].insert(global_memory->me);

	std::set<Processor::Kind> local_cpu_kinds;
	local_cpu_kinds.insert(Processor::LOC_PROC);
	local_cpu_kinds.insert(Processor::UTIL_PROC);
	local_cpu_kinds.insert(Processor::IO_PROC);
	local_cpu_kinds.insert(Processor::PROC_SET);

	for(std::set<Processor::Kind>::const_iterator it = local_cpu_kinds.begin();
	    it != local_cpu_kinds.end();
	    it++) {
	  Processor::Kind k = *it;

	  add_proc_mem_affinities(machine,
				  procs_by_kind[k],
				  mems_by_kind[Memory::SYSTEM_MEM],
				  100, // "large" bandwidth
				  5   // "small" latency
				  );

	  add_proc_mem_affinities(machine,
				  procs_by_kind[k],
				  mems_by_kind[Memory::REGDMA_MEM],
				  80,  // "large" bandwidth
				  10   // "small" latency
				  );

	  add_proc_mem_affinities(machine,
				  procs_by_kind[k],
				  mems_by_kind[Memory::DISK_MEM],
				  5,   // "low" bandwidth
				  100 // "high" latency
				  );

	  add_proc_mem_affinities(machine,
				  procs_by_kind[k],
				  mems_by_kind[Memory::HDF_MEM],
				  5,   // "low" bandwidth
				  100 // "high" latency
				  );

	  add_proc_mem_affinities(machine,
                  procs_by_kind[k],
                  mems_by_kind[Memory::FILE_MEM],
                  5,    // low bandwidth
                  100   // high latency)
                  );

	  add_proc_mem_affinities(machine,
				  procs_by_kind[k],
				  mems_by_kind[Memory::GLOBAL_MEM],
				  10,  // "lower" bandwidth
				  50  // "higher" latency
				  );
	}

	add_mem_mem_affinities(machine,
			       mems_by_kind[Memory::SYSTEM_MEM],
			       mems_by_kind[Memory::GLOBAL_MEM],
			       30,  // "lower" bandwidth
			       25  // "higher" latency
			       );

	add_mem_mem_affinities(machine,
			       mems_by_kind[Memory::SYSTEM_MEM],
			       mems_by_kind[Memory::DISK_MEM],
			       15,  // "low" bandwidth
			       50  // "high" latency
			       );

	add_mem_mem_affinities(machine,
			       mems_by_kind[Memory::SYSTEM_MEM],
			       mems_by_kind[Memory::FILE_MEM],
			       15,  // "low" bandwidth
			       50  // "high" latency
			       );

	for(std::set<Processor::Kind>::const_iterator it = local_cpu_kinds.begin();
	    it != local_cpu_kinds.end();
	    it++) {
	  Processor::Kind k = *it;

	  add_proc_mem_affinities(machine,
				  procs_by_kind[k],
				  mems_by_kind[Memory::Z_COPY_MEM],
				  40,  // "large" bandwidth
				  3   // "small" latency
				  );
	}
      }
      {
	Serialization::DynamicBufferSerializer dbs(4096);

	unsigned num_procs = 0;
	unsigned num_memories = 0;
	unsigned num_ib_memories = 0;
	bool ok = true;

	// announce each processor
	for(std::vector<ProcessorImpl *>::const_iterator it = n->processors.begin();
	    it != n->processors.end();
	    it++)
	  if(*it) {
	    Processor p = (*it)->me;
	    Processor::Kind k = (*it)->me.kind();
	    int num_cores = (*it)->num_cores;

	    num_procs++;
	    ok = (ok &&
		  (dbs << NODE_ANNOUNCE_PROC) &&
		  (dbs << p) &&
		  (dbs << k) &&
		  (dbs << num_cores));
	  }

	// now each memory
	for(std::vector<MemoryImpl *>::const_iterator it = n->memories.begin();
	    it != n->memories.end();
	    it++)
	  if(*it) {
	    Memory m = (*it)->me;
	    Memory::Kind k = (*it)->me.kind();
	    size_t size = (*it)->size;
	    intptr_t regptr = reinterpret_cast<intptr_t>((*it)->local_reg_base());

	    num_memories++;
	    ok = (ok &&
		  (dbs << NODE_ANNOUNCE_MEM) &&
		  (dbs << m) &&
		  (dbs << k) &&
		  (dbs << size) &&
		  (dbs << regptr));
	  }

        for (std::vector<MemoryImpl *>::const_iterator it = n->ib_memories.begin();
             it != n->ib_memories.end();
             it++)
          if(*it) {
            Memory m = (*it)->me;
            Memory::Kind k = (*it)->me.kind();
	    size_t size = (*it)->size;
	    intptr_t regptr = reinterpret_cast<intptr_t>((*it)->local_reg_base());

            num_ib_memories++;
	    ok = (ok &&
		  (dbs << NODE_ANNOUNCE_IB_MEM) &&
		  (dbs << m) &&
		  (dbs << k) &&
		  (dbs << size) &&
		  (dbs << regptr));
          }

	// announce each processor's affinities
	for(std::vector<ProcessorImpl *>::const_iterator it = n->processors.begin();
	    it != n->processors.end();
	    it++)
	  if(*it) {
	    Processor p = (*it)->me;
	    std::vector<Machine::ProcessorMemoryAffinity> pmas;
	    machine->get_proc_mem_affinity(pmas, p);

	    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it2 = pmas.begin();
		it2 != pmas.end();
		it2++) {
	      ok = (ok &&
		    (dbs << NODE_ANNOUNCE_PMA) &&
		    (dbs << it2->p) &&
		    (dbs << it2->m) &&
		    (dbs << it2->bandwidth) &&
		    (dbs << it2->latency));
	    }
	  }

	// now each memory's affinities with other memories
	for(std::vector<MemoryImpl *>::const_iterator it = n->memories.begin();
	    it != n->memories.end();
	    it++)
	  if(*it) {
	    Memory m = (*it)->me;
	    std::vector<Machine::MemoryMemoryAffinity> mmas;
	    machine->get_mem_mem_affinity(mmas, m);

	    for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it2 = mmas.begin();
		it2 != mmas.end();
		it2++) {
	      // only announce intra-node ones and only those with this memory as m1 to avoid
	      //  duplicates
	      if((it2->m1 != m) || ((NodeID)(it2->m2.address_space()) != my_node_id))
		continue;

	      ok = (ok &&
		    (dbs << NODE_ANNOUNCE_MMA) &&
		    (dbs << it2->m1) &&
		    (dbs << it2->m2) &&
		    (dbs << it2->bandwidth) &&
		    (dbs << it2->latency));
	    }
	  }

	for(std::vector<Channel *>::const_iterator it = n->dma_channels.begin();
	    it != n->dma_channels.end();
	    ++it)
	  if(*it) {
	    ok = (ok &&
		  (dbs << NODE_ANNOUNCE_DMA_CHANNEL) &&
		  (*it)->serialize_remote_info(dbs));
	  }

	ok = (ok && (dbs << NODE_ANNOUNCE_DONE));
	assert(ok);

#ifdef DEBUG_REALM_STARTUP
	if(my_node_id == 0) {
	  TimeStamp ts("sending announcements", false);
	  fflush(stdout);
	}
#endif

	// now announce ourselves to everyone else
	for(NodeID i = 0; i <= max_node_id; i++)
	  if(i != my_node_id)
	    NodeAnnounceMessage::send_request(i,
					      num_procs,
					      num_memories,
					      num_ib_memories,
					      dbs.get_buffer(),
					      dbs.bytes_used(),
					      PAYLOAD_COPY);

	NodeAnnounceMessage::await_all_announcements();

#ifdef DEBUG_REALM_STARTUP
	if(my_node_id == 0) {
	  TimeStamp ts("received all announcements", false);
	  fflush(stdout);
	}
#endif
      }

      return true;
    }

    void RuntimeImpl::start(void)
    {
      // all we have to do here is tell the processors to start up their
      //  threads...
      for(std::vector<ProcessorImpl *>::const_iterator it = nodes[my_node_id].processors.begin();
	  it != nodes[my_node_id].processors.end();
	  ++it)
	(*it)->start_threads();
    }

  template <typename T>
  void spawn_on_all(const T& container_of_procs,
		    Processor::TaskFuncID func_id,
		    const void *args, size_t arglen,
		    Event start_event = Event::NO_EVENT,
		    int priority = 0)
  {
    for(typename T::const_iterator it = container_of_procs.begin();
	it != container_of_procs.end();
	it++)
      (*it)->me.spawn(func_id, args, arglen, ProfilingRequestSet(), start_event, priority);
  }

  struct CollectiveSpawnInfo {
    Processor target_proc;
    Processor::TaskFuncID task_id;
    Event wait_on;
    int priority;
  };

#define DEBUG_COLLECTIVES

#if defined(USE_GASNET) && defined(DEBUG_COLLECTIVES)
  static const int GASNET_COLL_FLAGS = GASNET_COLL_IN_MYSYNC | GASNET_COLL_OUT_MYSYNC | GASNET_COLL_LOCAL;
  
  template <typename T>
  static void broadcast_check(const T& val, const char *name)
  {
    T bval;
    gasnet_coll_broadcast(GASNET_TEAM_ALL, &bval, 0, const_cast<T *>(&val), sizeof(T), GASNET_COLL_FLAGS);
    if(val != bval) {
      log_collective.fatal() << "collective mismatch on node " << my_node_id << " for " << name << ": " << val << " != " << bval;
      assert(false);
    }
  }
#endif

    Event RuntimeImpl::collective_spawn(Processor target_proc, Processor::TaskFuncID task_id, 
					const void *args, size_t arglen,
					Event wait_on /*= Event::NO_EVENT*/, int priority /*= 0*/)
    {
      log_collective.info() << "collective spawn: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " before=" << wait_on;

#ifdef USE_GASNET
#ifdef DEBUG_COLLECTIVES
      broadcast_check(target_proc, "target_proc");
      broadcast_check(task_id, "task_id");
      broadcast_check(priority, "priority");
#endif

      // root node will be whoever owns the target proc
      int root = ID(target_proc).proc.owner_node;

      if((int)my_node_id == root) {
	// ROOT NODE

	// step 1: receive wait_on from every node
	Event *all_events = 0;
	all_events = new Event[max_node_id + 1];
	gasnet_coll_gather(GASNET_TEAM_ALL, root, all_events, &wait_on, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: merge all the events
	std::set<Event> event_set;
	for(NodeID i = 0; i <= max_node_id; i++) {
	  //log_collective.info() << "ev " << i << ": " << all_events[i];
	  if(all_events[i].exists())
	    event_set.insert(all_events[i]);
	}
	delete[] all_events;

	Event merged_event = Event::merge_events(event_set);
	log_collective.info() << "merged precondition: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " before=" << merged_event;

	// step 3: run the task
	Event finish_event = target_proc.spawn(task_id, args, arglen, merged_event, priority);

	// step 4: broadcast the finish event to everyone
	gasnet_coll_broadcast(GASNET_TEAM_ALL, &finish_event, root, &finish_event, sizeof(Event), GASNET_COLL_FLAGS);

	log_collective.info() << "collective spawn: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " after=" << finish_event;

	return finish_event;
      } else {
	// NON-ROOT NODE

	// step 1: send our wait_on to the root for merging
	gasnet_coll_gather(GASNET_TEAM_ALL, root, 0, &wait_on, sizeof(Event), GASNET_COLL_FLAGS);

	// steps 2 and 3: twiddle thumbs

	// step 4: receive finish event
	Event finish_event;
	gasnet_coll_broadcast(GASNET_TEAM_ALL, &finish_event, root, 0, sizeof(Event), GASNET_COLL_FLAGS);

	log_collective.info() << "collective spawn: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " after=" << finish_event;

	return finish_event;
      }
#else
      // no GASNet, so a collective spawn is the same as a regular spawn
      Event finish_event = target_proc.spawn(task_id, args, arglen, wait_on, priority);

      log_collective.info() << "collective spawn: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " after=" << finish_event;

      return finish_event;
#endif
    }

    Event RuntimeImpl::collective_spawn_by_kind(Processor::Kind target_kind, Processor::TaskFuncID task_id, 
						const void *args, size_t arglen,
						bool one_per_node /*= false*/,
						Event wait_on /*= Event::NO_EVENT*/, int priority /*= 0*/)
    {
      log_collective.info() << "collective spawn: kind=" << target_kind << " func=" << task_id << " priority=" << priority << " before=" << wait_on;

#ifdef USE_GASNET
#ifdef DEBUG_COLLECTIVES
      broadcast_check(target_kind, "target_kind");
      broadcast_check(task_id, "task_id");
      broadcast_check(one_per_node, "one_per_node");
      broadcast_check(priority, "priority");
#endif

      // every node is involved in this one, so the root is arbitrary - we'll pick node 0

      Event merged_event;

      if(my_node_id == 0) {
	// ROOT NODE

	// step 1: receive wait_on from every node
	Event *all_events = 0;
	all_events = new Event[max_node_id + 1];
	gasnet_coll_gather(GASNET_TEAM_ALL, 0, all_events, &wait_on, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: merge all the events
	std::set<Event> event_set;
	for(NodeID i = 0; i <= max_node_id; i++) {
	  //log_collective.info() << "ev " << i << ": " << all_events[i];
	  if(all_events[i].exists())
	    event_set.insert(all_events[i]);
	}
	delete[] all_events;

	merged_event = Event::merge_events(event_set);

	// step 3: broadcast the merged event back to everyone
	gasnet_coll_broadcast(GASNET_TEAM_ALL, &merged_event, 0, &merged_event, sizeof(Event), GASNET_COLL_FLAGS);
      } else {
	// NON-ROOT NODE

	// step 1: send our wait_on to the root for merging
	gasnet_coll_gather(GASNET_TEAM_ALL, 0, 0, &wait_on, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: twiddle thumbs

	// step 3: receive merged wait_on event
	gasnet_coll_broadcast(GASNET_TEAM_ALL, &merged_event, 0, 0, sizeof(Event), GASNET_COLL_FLAGS);
      }
#else
      // no GASNet, so our precondition is the only one
      Event merged_event = wait_on;
#endif

      // now spawn 0 or more local tasks
      std::set<Event> event_set;

      const std::vector<ProcessorImpl *>& local_procs = nodes[my_node_id].processors;

      for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
	if((target_kind == Processor::NO_KIND) || ((*it)->kind == target_kind)) {
	  Event e = (*it)->me.spawn(task_id, args, arglen, ProfilingRequestSet(),
				    merged_event, priority);
	  log_collective.info() << "spawn by kind: proc=" << (*it)->me << " func=" << task_id << " before=" << merged_event << " after=" << e;
	  if(e.exists())
	    event_set.insert(e);

	  if(one_per_node)
	    break;
	}

      // local merge
      Event my_finish = Event::merge_events(event_set);

#ifdef USE_GASNET
      if(my_node_id == 0) {
	// ROOT NODE

	// step 1: receive wait_on from every node
	Event *all_events = 0;
	all_events = new Event[max_node_id + 1];
	gasnet_coll_gather(GASNET_TEAM_ALL, 0, all_events, &my_finish, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: merge all the events
	std::set<Event> event_set;
	for(NodeID i = 0; i <= max_node_id; i++) {
	  //log_collective.info() << "ev " << i << ": " << all_events[i];
	  if(all_events[i].exists())
	    event_set.insert(all_events[i]);
	}
	delete[] all_events;

	Event merged_finish = Event::merge_events(event_set);

	// step 3: broadcast the merged event back to everyone
	gasnet_coll_broadcast(GASNET_TEAM_ALL, &merged_finish, 0, &merged_finish, sizeof(Event), GASNET_COLL_FLAGS);

	log_collective.info() << "collective spawn: kind=" << target_kind << " func=" << task_id << " priority=" << priority << " after=" << merged_finish;

	return merged_finish;
      } else {
	// NON-ROOT NODE

	// step 1: send our wait_on to the root for merging
	gasnet_coll_gather(GASNET_TEAM_ALL, 0, 0, &my_finish, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: twiddle thumbs

	// step 3: receive merged wait_on event
	Event merged_finish;
	gasnet_coll_broadcast(GASNET_TEAM_ALL, &merged_finish, 0, 0, sizeof(Event), GASNET_COLL_FLAGS);

	log_collective.info() << "collective spawn: kind=" << target_kind << " func=" << task_id << " priority=" << priority << " after=" << merged_finish;

	return merged_finish;
      }
#else
      // no GASNet, so just return our locally merged event
      log_collective.info() << "collective spawn: kind=" << target_kind << " func=" << task_id << " priority=" << priority << " after=" << my_finish;

      return my_finish;
#endif
    }

#if 0
    struct MachineRunArgs {
      RuntimeImpl *r;
      Processor::TaskFuncID task_id;
      Runtime::RunStyle style;
      const void *args;
      size_t arglen;
    };  

    static bool running_as_background_thread = false;

    static void *background_run_thread(void *data)
    {
      MachineRunArgs *args = (MachineRunArgs *)data;
      running_as_background_thread = true;
      args->r->run(args->task_id, args->style, args->args, args->arglen,
		   false /* foreground from this thread's perspective */);
      delete args;
      return 0;
    }
#endif

    void RuntimeImpl::run(Processor::TaskFuncID task_id /*= 0*/,
			  Runtime::RunStyle style /*= ONE_TASK_ONLY*/,
			  const void *args /*= 0*/, size_t arglen /*= 0*/,
			  bool background /*= false*/)
    { 
      // trigger legacy behavior (e.g. calling shutdown task on all processors)
      run_method_called = true;
#if 0
      if(background) {
        log_runtime.info("background operation requested\n");
	fflush(stdout);
	MachineRunArgs *margs = new MachineRunArgs;
	margs->r = this;
	margs->task_id = task_id;
	margs->style = style;
	margs->args = args;
	margs->arglen = arglen;
	
        pthread_t *threadp = (pthread_t*)malloc(sizeof(pthread_t));
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	CHECK_PTHREAD( pthread_create(threadp, &attr, &background_run_thread, (void *)margs) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
        background_pthread = threadp;
#ifdef DEADLOCK_TRACE
        this->add_thread(threadp); 
#endif
	return;
      }
#endif

      // step 1: a collective spawn to run the init task on all processors that care
      Event init_event = collective_spawn_by_kind(Processor::NO_KIND, Processor::TASK_ID_PROCESSOR_INIT, 0, 0,
						  false /*run on all procs*/,
						  Event::NO_EVENT,
						  INT_MAX); // runs with max priority
      
      Event main_event;
      if(task_id != 0) {
	if(style == Runtime::ONE_TASK_ONLY) {
	  // everybody needs to agree on this...
	  Processor p = nodes[0].processors[0]->me;
	  main_event = collective_spawn(p, task_id, args, arglen, init_event);
	} else {
	  main_event = collective_spawn_by_kind(Processor::NO_KIND, task_id, args, arglen,
						(style == Runtime::ONE_TASK_PER_NODE),
						init_event, 0 /*priority*/);
	}
      } else {
	// no main task!?
	main_event = init_event;
      }

      // if we're in background mode, we just return to the caller now
      if(background)
	return;

      // otherwise, sleep until shutdown has been requested by somebody
      {
	AutoHSLLock al(shutdown_mutex);
	while(!shutdown_requested)
	  shutdown_condvar.wait();
	log_runtime.info("shutdown request received - terminating\n");
      }

      int result = wait_for_shutdown();
      exit(result);
    }

    // this is not member data of RuntimeImpl because we don't want use-after-free problems
    static int shutdown_count = 0;

    void RuntimeImpl::shutdown(bool local_request, int result_code)
    {
      // filter out duplicate requests
      bool already_started = (__sync_fetch_and_add(&shutdown_count, 1) > 0);
      if(already_started)
	return;

      if(local_request) {
	log_runtime.info("shutdown request - notifying other nodes");
	for(NodeID i = 0; i <= max_node_id; i++)
	  if(i != my_node_id)
	    RuntimeShutdownMessage::send_request(i, result_code);
      }

      log_runtime.info("shutdown request - cleaning up local processors");

      if(run_method_called) {
	// legacy shutdown - call shutdown task on processors
	log_task.info("spawning processor shutdown task on local cpus");

	const std::vector<ProcessorImpl *>& local_procs = nodes[my_node_id].processors;

	spawn_on_all(local_procs, Processor::TASK_ID_PROCESSOR_SHUTDOWN, 0, 0,
		     Event::NO_EVENT,
		     INT_MIN); // runs with lowest priority
      }

      {
	AutoHSLLock al(shutdown_mutex);
	shutdown_result_code = result_code;
	shutdown_requested = true;
	shutdown_condvar.broadcast();
      }
    }

    int RuntimeImpl::wait_for_shutdown(void)
    {
#if 0
      bool exit_process = true;
      if (background_pthread != 0)
      {
        pthread_t *background_thread = (pthread_t*)background_pthread;
        void *result;
        pthread_join(*background_thread, &result);
        free(background_thread);
        // Set this to null so we don't wait anymore
        background_pthread = 0;
        exit_process = false;
      }
#endif

      // sleep until shutdown has been requested by somebody
      {
	AutoHSLLock al(shutdown_mutex);
	while(!shutdown_requested)
	  shutdown_condvar.wait();
	log_runtime.info("shutdown request received - terminating");
      }

#ifdef USE_GASNET
      // don't start tearing things down until all processes agree
      gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
      gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
#endif

      // Shutdown all the threads

      // threads that cause inter-node communication have to stop first
      PartitioningOpQueue::stop_worker_threads();
      stop_dma_worker_threads();
      stop_dma_system();
      stop_activemsg_threads();

      sampling_profiler.shutdown();

      {
	std::vector<ProcessorImpl *>& local_procs = nodes[my_node_id].processors;
	for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++)
	  (*it)->shutdown();
      }

#ifdef EVENT_TRACING
      if(event_trace_file) {
	printf("writing event trace to %s\n", event_trace_file);
        Tracer<EventTraceItem>::dump_trace(event_trace_file, false);
	free(event_trace_file);
	event_trace_file = 0;
      }
#endif
#ifdef LOCK_TRACING
      if (lock_trace_file)
      {
        printf("writing lock trace to %s\n", lock_trace_file);
        Tracer<LockTraceItem>::dump_trace(lock_trace_file, false);
        free(lock_trace_file);
        lock_trace_file = 0;
      }
#endif

#ifdef REPORT_REALM_RESOURCE_USAGE
      {
        RuntimeImpl *rt = get_runtime();
        printf("node %d realm resource usage: ev=%d, rsrv=%d, idx=%d, pg=%d\n",
               my_node_id,
               rt->local_event_free_list->next_alloc,
               rt->local_reservation_free_list->next_alloc,
               rt->local_index_space_free_list->next_alloc,
               rt->local_proc_group_free_list->next_alloc);
      }
#endif
#ifdef EVENT_GRAPH_TRACE
      {
        //FILE *log_file = Logger::get_log_file();
        show_event_waiters(/*log_file*/);
      }
#endif

      // delete processors, memories, nodes, etc.
      {
	for(NodeID i = 0; i <= max_node_id; i++) {
	  Node& n = nodes[i];

	  delete_container_contents(n.memories);
	  delete_container_contents(n.processors);
	  delete_container_contents(n.ib_memories);
	  delete_container_contents(n.dma_channels);
	  delete_container_contents(n.sparsity_maps);
	}
	
	delete[] nodes;
	delete global_memory;
	delete local_event_free_list;
	delete local_barrier_free_list;
	delete local_reservation_free_list;
	delete local_proc_group_free_list;
	delete_container_contents(local_sparsity_map_free_lists);

	// same for code translators
	delete_container_contents(code_translators);

	for(std::vector<Module *>::iterator it = modules.begin();
	    it != modules.end();
	    it++) {
	  (*it)->cleanup();
	  delete (*it);
	}

	module_registrar.unload_module_sofiles();
      }

#ifndef USE_GASNET
      if(nongasnet_regmem_base != 0)
	free(nongasnet_regmem_base);
      if(nongasnet_reg_ib_mem_base != 0)
	free(nongasnet_reg_ib_mem_base);
#endif

      if(!Threading::cleanup()) exit(1);

      return shutdown_result_code;
    }

    EventImpl *RuntimeImpl::get_event_impl(Event e)
    {
      ID id(e);
      if(id.is_event())
	return get_genevent_impl(e);
      if(id.is_barrier())
	return get_barrier_impl(e);

      log_runtime.fatal() << "invalid event handle: id=" << id;
      assert(0 && "invalid event handle");
      return 0;
    }

    GenEventImpl *RuntimeImpl::get_genevent_impl(Event e)
    {
      ID id(e);
      assert(id.is_event());

      Node *n = &nodes[id.event.creator_node];
      GenEventImpl *impl = n->events.lookup_entry(id.event.gen_event_idx, id.event.creator_node);
      {
	ID check(impl->me);
	assert(check.event.creator_node == id.event.creator_node);
	assert(check.event.gen_event_idx == id.event.gen_event_idx);
      }
      return impl;
    }

    BarrierImpl *RuntimeImpl::get_barrier_impl(Event e)
    {
      ID id(e);
      assert(id.is_barrier());

      Node *n = &nodes[id.barrier.creator_node];
      BarrierImpl *impl = n->barriers.lookup_entry(id.barrier.barrier_idx, id.barrier.creator_node);
      {
	ID check(impl->me);
	assert(check.barrier.creator_node == id.barrier.creator_node);
	assert(check.barrier.barrier_idx == id.barrier.barrier_idx);
      }
      return impl;
    }

    SparsityMapImplWrapper *RuntimeImpl::get_sparsity_impl(ID id)
    {
      if(!id.is_sparsity()) {
	log_runtime.fatal() << "invalid index space sparsity handle: id=" << id;
	assert(0 && "invalid index space sparsity handle");
      }

      Node *n = &nodes[id.sparsity.owner_node];
      DynamicTable<SparsityMapTableAllocator> *& m = n->sparsity_maps[id.sparsity.creator_node];
      // might need to construct this (in a lock-free way)
      if(m == 0) {
	// construct one and try to swap it in
	DynamicTable<SparsityMapTableAllocator> *newm = new DynamicTable<SparsityMapTableAllocator>;
	if(!__sync_bool_compare_and_swap(&m, 0, newm))
	  delete newm;  // somebody else made it faster
      }
      SparsityMapImplWrapper *impl = m->lookup_entry(id.sparsity.sparsity_idx,
						     id.sparsity.owner_node);
      // creator node isn't always right, so try to fix it
      if(impl->me != id) {
	if(impl->me.sparsity.creator_node == 0)
	  impl->me.sparsity.creator_node = id.sparsity.creator_node;
	assert(impl->me == id);
      }
      return impl;
    }
  
    SparsityMapImplWrapper *RuntimeImpl::get_available_sparsity_impl(NodeID target_node)
    {
      SparsityMapImplWrapper *wrap = local_sparsity_map_free_lists[target_node]->alloc_entry();
      wrap->me.sparsity.creator_node = my_node_id;
      return wrap;
    }

    ReservationImpl *RuntimeImpl::get_lock_impl(ID id)
    {
      if(id.is_reservation()) {
	Node *n = &nodes[id.rsrv.creator_node];
	ReservationImpl *impl = n->reservations.lookup_entry(id.rsrv.rsrv_idx, id.rsrv.creator_node);
	assert(impl->me == id.convert<Reservation>());
	return impl;
      }

      if(id.is_instance())
	return &(get_instance_impl(id)->lock);

      if(id.is_procgroup())
	return &(get_procgroup_impl(id)->lock);

      log_runtime.fatal() << "invalid reservation handle: id=" << id;
      assert(0 && "invalid reservation handle");
      return 0;
    }

    template <class T>
    inline T *null_check(T *ptr)
    {
      assert(ptr != 0);
      return ptr;
    }

    MemoryImpl *RuntimeImpl::get_memory_impl(ID id)
    {
      if(id.is_memory()) {
        // support old encoding for global memory too
	if((id.memory.owner_node > ID::MAX_NODE_ID) || (id.memory.mem_idx == ((1U << 12) - 1)))
	  return global_memory;
	else
	  return null_check(nodes[id.memory.owner_node].memories[id.memory.mem_idx]);
      }

      if(id.is_ib_memory()) {
        return null_check(nodes[id.ib_memory.owner_node].ib_memories[id.ib_memory.mem_idx]);
      }
#ifdef TODO
      if(id.is_allocator()) {
	if(id.allocator.owner_node > ID::MAX_NODE_ID)
	  return global_memory;
	else
	  return null_check(nodes[id.allocator.owner_node].memories[id.allocator.mem_idx]);
      }
#endif

      if(id.is_instance()) {
        // support old encoding for global memory too
	if((id.instance.owner_node > ID::MAX_NODE_ID) || (id.instance.mem_idx == ((1U << 12) - 1)))
	  return global_memory;
	else
	  return null_check(nodes[id.instance.owner_node].memories[id.instance.mem_idx]);
      }

      log_runtime.fatal() << "invalid memory handle: id=" << id;
      assert(0 && "invalid memory handle");
      return 0;
    }

    ProcessorImpl *RuntimeImpl::get_processor_impl(ID id)
    {
      if(id.is_procgroup())
	return get_procgroup_impl(id);

      if(!id.is_processor()) {
	log_runtime.fatal() << "invalid processor handle: id=" << id;
	assert(0 && "invalid processor handle");
      }

      return null_check(nodes[id.proc.owner_node].processors[id.proc.proc_idx]);
    }

    ProcessorGroup *RuntimeImpl::get_procgroup_impl(ID id)
    {
      if(!id.is_procgroup()) {
	log_runtime.fatal() << "invalid processor group handle: id=" << id;
	assert(0 && "invalid processor group handle");
      }

      Node *n = &nodes[id.pgroup.owner_node];
      ProcessorGroup *impl = n->proc_groups.lookup_entry(id.pgroup.pgroup_idx,
							 id.pgroup.owner_node);
      assert(impl->me == id.convert<Processor>());
      return impl;
    }

    RegionInstanceImpl *RuntimeImpl::get_instance_impl(ID id)
    {
      if(!id.is_instance()) {
	log_runtime.fatal() << "invalid instance handle: id=" << id;
	assert(0 && "invalid instance handle");
      }

      MemoryImpl *mem = get_memory_impl(id);

      return mem->get_instance(id.convert<RegionInstance>());
#if 0
      AutoHSLLock al(mem->mutex);

      // TODO: factor creator_node into lookup!
      if(id.instance.inst_idx >= mem->instances.size()) {
	assert(id.instance.owner_node != my_node_id);

	size_t old_size = mem->instances.size();
	if(id.instance.inst_idx >= old_size) {
	  // still need to grow (i.e. didn't lose the race)
	  mem->instances.resize(id.instance.inst_idx + 1);

	  // don't have region/offset info - will have to pull that when
	  //  needed
	  for(unsigned i = old_size; i <= id.instance.inst_idx; i++) 
	    mem->instances[i] = 0;
	}
      }

      if(!mem->instances[id.instance.inst_idx]) {
	if(!mem->instances[id.instance.inst_idx]) {
	  //printf("[%d] creating proxy instance: inst=" IDFMT "\n", my_node_id, id.id());
	  mem->instances[id.instance.inst_idx] = new RegionInstanceImpl(id.convert<RegionInstance>(), mem->me);
	}
      }
	  
      return mem->instances[id.instance.inst_idx];
#endif
    }

    /*static*/
    void RuntimeImpl::realm_backtrace(int signal)
    {
      assert((signal == SIGILL) || (signal == SIGFPE) || 
             (signal == SIGABRT) || (signal == SIGSEGV) ||
             (signal == SIGBUS));
#if 0
      void *bt[256];
      int bt_size = backtrace(bt, 256);
      char **bt_syms = backtrace_symbols(bt, bt_size);
      size_t buffer_size = 2048; // default buffer size
      char *buffer = (char*)malloc(buffer_size);
      size_t offset = 0;
      size_t funcnamesize = 256;
      char *funcname = (char*)malloc(funcnamesize);
      for (int i = 0; i < bt_size; i++) {
        // Modified from https://panthema.net/2008/0901-stacktrace-demangled/ 
        // under WTFPL 2.0
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;
        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = bt_syms[i]; *p; ++p) {
          if (*p == '(')
            begin_name = p;
          else if (*p == '+')
            begin_offset = p;
          else if (*p == ')' && begin_offset) {
            end_offset = p;
            break;
          }
        }
        // If offset is within half of the buffer size, double the buffer
        if (offset >= (buffer_size / 2)) {
          buffer_size *= 2;
          buffer = (char*)realloc(buffer, buffer_size);
        }
        if (begin_name && begin_offset && end_offset &&
            (begin_name < begin_offset)) {
          *begin_name++ = '\0';
          *begin_offset++ = '\0';
          *end_offset = '\0';
          // mangled name is now in [begin_name, begin_offset) and caller
          // offset in [begin_offset, end_offset). now apply __cxa_demangle():
          int status;
          char* demangled_name = 
            abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
          if (status == 0) {
            funcname = demangled_name; // use possibly realloc()-ed string
            offset += snprintf(buffer+offset,buffer_size-offset,
                         "  %s : %s+%s\n", bt_syms[i], funcname, begin_offset);
          } else {
            // demangling failed. Output function name as a C function 
            // with no arguments.
            offset += snprintf(buffer+offset,buffer_size-offset,
                     "  %s : %s()+%s\n", bt_syms[i], begin_name, begin_offset);
          }
        } else {
          // Who knows just print the whole line
          offset += snprintf(buffer+offset,buffer_size-offset,
                             "%s\n",bt_syms[i]);
        }
      }
      fprintf(stderr,"BACKTRACE (%d, %lx)\n----------\n%s\n----------\n", 
              my_node_id, (unsigned long)pthread_self(), buffer);
      fflush(stderr);
      free(buffer);
      free(funcname);
#endif
      Backtrace bt;
      bt.capture_backtrace(1 /* skip this handler */);
      bt.lookup_symbols();
      fflush(stdout);
      fflush(stderr);
      std::cout << std::flush;
      std::cerr << "Signal " << signal << " received by process " << getpid()
                << " (thread "  << std::hex << uintptr_t(pthread_self())
                << std::dec << ") at: " << bt << std::flush;
      // returning would almost certainly cause this signal to be raised again,
      //  so sleep for a second in case other threads also want to chronicle
      //  their own deaths, and then exit
      sleep(1);
      // don't bother trying to clean things up
      _exit(1);
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class Node
  //

    Node::Node(void)
    {
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RuntimeShutdownMessage
  //

  /*static*/ void RuntimeShutdownMessage::handle_request(RequestArgs args)
  {
    log_runtime.info() << "shutdown request received: sender=" << args.initiating_node << " code=" << args.result_code;

    get_runtime()->shutdown(false, args.result_code);
  }

  /*static*/ void RuntimeShutdownMessage::send_request(NodeID target,
						       int result_code)
  {
    RequestArgs args;

    args.initiating_node = my_node_id;
    args.result_code = result_code;
    Message::request(target, args);
  }

  
}; // namespace Realm
