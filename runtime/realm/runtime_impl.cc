/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include "runtime_impl.h"

#include "proc_impl.h"
#include "mem_impl.h"
#include "inst_impl.h"

#include "activemsg.h"

#include "cmdline.h"

#include "codedesc.h"

#include "utils.h"

// For doing backtraces
#include <execinfo.h> // symbols
#include <cxxabi.h>   // demangling

#ifndef USE_GASNET
/*extern*/ void *fake_gasnet_mem_base = 0;
/*extern*/ size_t fake_gasnet_mem_size = 0;
#endif

// remote copy active messages from from lowlevel_dma.h for now
#include "lowlevel_dma.h"
namespace Realm {
  typedef LegionRuntime::LowLevel::RemoteCopyMessage RemoteCopyMessage;
  typedef LegionRuntime::LowLevel::RemoteFillMessage RemoteFillMessage;
};

// create xd message and update bytes read/write messages
#include "channel.h"
namespace Realm {
  typedef LegionRuntime::LowLevel::XferDesRemoteWriteMessage XferDesRemoteWriteMessage;
  typedef LegionRuntime::LowLevel::XferDesRemoteWriteAckMessage XferDesRemoteWriteAckMessage;
  typedef LegionRuntime::LowLevel::XferDesCreateMessage XferDesCreateMessage;
  typedef LegionRuntime::LowLevel::XferDesDestroyMessage XferDesDestroyMessage;
  typedef LegionRuntime::LowLevel::NotifyXferDesCompleteMessage NotifyXferDesCompleteMessage;
  typedef LegionRuntime::LowLevel::UpdateBytesWriteMessage UpdateBytesWriteMessage;
  typedef LegionRuntime::LowLevel::UpdateBytesReadMessage UpdateBytesReadMessage;
  typedef LegionRuntime::LowLevel::RemoteIBAllocRequestAsync RemoteIBAllocRequestAsync;
  typedef LegionRuntime::LowLevel::RemoteIBAllocResponseAsync RemoteIBAllocResponseAsync;
  typedef LegionRuntime::LowLevel::RemoteIBFreeRequestAsync RemoteIBFreeRequestAsync;
}

#include <unistd.h>
#include <signal.h>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

namespace LegionRuntime {
  namespace LowLevel {
    extern void show_event_waiters(std::ostream& os);
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

  static void realm_show_events(int signal)
  {
    LegionRuntime::LowLevel::show_event_waiters(std::cout);
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

    bool Runtime::init(int *argc, char ***argv)
    {
      if(runtime_singleton != 0) {
	fprintf(stderr, "ERROR: cannot initialize more than one runtime at a time!\n");
	return false;
      }

      impl = new RuntimeImpl;
      runtime_singleton = ((RuntimeImpl *)impl);
      return ((RuntimeImpl *)impl)->init(argc, argv);
    }
    
    // this is now just a wrapper around Processor::register_task - consider switching to
    //  that
    bool Runtime::register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr)
    {
      assert(impl != 0);

      CodeDescriptor codedesc(taskptr);
      ProfilingRequestSet prs;
      std::set<Event> events;
      std::vector<ProcessorImpl *>& procs = ((RuntimeImpl *)impl)->nodes[gasnet_mynode()].processors;
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

      ((RuntimeImpl *)impl)->reduce_op_table[redop_id] = redop;
      return true;
    }

    bool Runtime::register_custom_serdez(CustomSerdezID serdez_id, const CustomSerdezUntyped *serdez)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->custom_serdez_table.count(serdez_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->custom_serdez_table[serdez_id] = serdez;
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
      DeferredShutdown(RuntimeImpl *_runtime);
      virtual ~DeferredShutdown(void);

      virtual bool event_triggered(Event e, bool poisoned);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      RuntimeImpl *runtime;
    };

    DeferredShutdown::DeferredShutdown(RuntimeImpl *_runtime)
      : runtime(_runtime)
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
      runtime->shutdown(true);
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

    void Runtime::shutdown(Event wait_on /*= Event::NO_EVENT*/)
    {
      log_runtime.info() << "shutdown requested - wait_on=" << wait_on;
      if(wait_on.has_triggered())
	((RuntimeImpl *)impl)->shutdown(true); // local request
      else
	EventImpl::add_waiter(wait_on, new DeferredShutdown((RuntimeImpl *)impl));
    }

    void Runtime::wait_for_shutdown(void)
    {
      ((RuntimeImpl *)impl)->wait_for_shutdown();

      // after the shutdown, we nuke the RuntimeImpl
      delete ((RuntimeImpl *)impl);
      impl = 0;
      runtime_singleton = 0;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreModule
  //

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
						    stack_size_in_mb << 20);
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
						stack_size_in_mb << 20);
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
	local_reservation_free_list(0), local_index_space_free_list(0),
	local_proc_group_free_list(0), run_method_called(false),
	shutdown_requested(false), shutdown_condvar(shutdown_mutex),
	core_map(0), core_reservations(0),
	sampling_profiler(true /*system default*/),
	num_local_memories(0), num_local_ib_memories(0),
	num_local_processors(0),
	module_registrar(this)
    {
      machine = new MachineImpl;
    }

    RuntimeImpl::~RuntimeImpl(void)
    {
      delete machine;
      delete core_reservations;
      delete core_map;
    }

    Memory RuntimeImpl::next_local_memory_id(void)
    {
      Memory m = ID::make_memory(gasnet_mynode(),
				 num_local_memories++).convert<Memory>();
      return m;
    }

    Memory RuntimeImpl::next_local_ib_memory_id(void)
    {
      Memory m = ID::make_ib_memory(gasnet_mynode(),
                                    num_local_ib_memories++).convert<Memory>();
      return m;
    }

    Processor RuntimeImpl::next_local_processor_id(void)
    {
      Processor p = ID::make_processor(gasnet_mynode(), 
				       num_local_processors++).convert<Processor>();
      return p;
    }

    void RuntimeImpl::add_memory(MemoryImpl *m)
    {
      // right now expect this to always be for the current node and the next memory ID
      ID id(m->me);
      assert(id.memory.owner_node == gasnet_mynode());
      assert(id.memory.mem_idx == nodes[gasnet_mynode()].memories.size());

      nodes[gasnet_mynode()].memories.push_back(m);
    }

    void RuntimeImpl::add_ib_memory(MemoryImpl *m)
    {
      // right now expect this to always be for the current node and the next memory ID
      ID id(m->me);
      assert(id.memory.owner_node == gasnet_mynode());
      assert(id.memory.mem_idx == nodes[gasnet_mynode()].ib_memories.size());

      nodes[gasnet_mynode()].ib_memories.push_back(m);
    }

    void RuntimeImpl::add_processor(ProcessorImpl *p)
    {
      // right now expect this to always be for the current node and the next processor ID
      ID id(p->me);
      assert(id.proc.owner_node == gasnet_mynode());
      assert(id.proc.proc_idx == nodes[gasnet_mynode()].processors.size());

      nodes[gasnet_mynode()].processors.push_back(p);
    }

    void RuntimeImpl::add_dma_channel(DMAChannel *c)
    {
      dma_channels.push_back(c);
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

    const std::vector<DMAChannel *>& RuntimeImpl::get_dma_channels(void) const
    {
      return dma_channels;
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
	  Machine::MemoryMemoryAffinity mma;
	  mma.m1 = *it1;
	  mma.m2 = *it2;
	  mma.bandwidth = bandwidth;
	  mma.latency = latency;
	  machine->add_mem_mem_affinity(mma);
	}
    }

    bool RuntimeImpl::init(int *argc, char ***argv)
    {
      // have to register domain mappings too
      LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Arrays::CArrayLinearization<1> >();
      LegionRuntime::Arrays::Mapping<2,1>::register_mapping<LegionRuntime::Arrays::CArrayLinearization<2> >();
      LegionRuntime::Arrays::Mapping<3,1>::register_mapping<LegionRuntime::Arrays::CArrayLinearization<3> >();
      LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Arrays::FortranArrayLinearization<1> >();
      LegionRuntime::Arrays::Mapping<2,1>::register_mapping<LegionRuntime::Arrays::FortranArrayLinearization<2> >();
      LegionRuntime::Arrays::Mapping<3,1>::register_mapping<LegionRuntime::Arrays::FortranArrayLinearization<3> >();
      LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Arrays::Translation<1> >();
      // we also register split dim linearization
      //LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Layouts::SplitDimLinearization<1> >();
      //LegionRuntime::Arrays::Mapping<2,1>::register_mapping<LegionRuntime::Layouts::SplitDimLinearization<2> >();
      //LegionRuntime::Arrays::Mapping<3,1>::register_mapping<LegionRuntime::Layouts::SplitDimLinearization<3> >();


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
#ifdef DEBUG_REALM_STARTUP
      { // once we're convinced there isn't skew here, reduce this to rank 0
        char s[80];
        gethostname(s, 79);
        strcat(s, " exit gasnet_init");
        TimeStamp ts(s, false);
        fflush(stdout);
      }
#endif

      // new command-line parsers will work from a vector<string> representation of the
      //  command line
      std::vector<std::string> cmdline;
      if(*argc > 1) {
	cmdline.resize(*argc - 1);
	for(int i = 1; i < *argc; i++)
	  cmdline[i - 1] = (*argv)[i];
      }

      // very first thing - let the logger initialization happen
      Logger::configure_from_cmdline(cmdline);

      // start up the threading subsystem - modules will likely want threads
      if(!Threading::initialize()) exit(1);

      // now load modules
      module_registrar.create_static_modules(cmdline, modules);
      module_registrar.create_dynamic_modules(cmdline, modules);

      // low-level runtime parameters
#ifdef USE_GASNET
      size_t gasnet_mem_size_in_mb = 256;
      size_t reg_ib_mem_size_in_mb = 256;
#else
      size_t gasnet_mem_size_in_mb = 0;
      size_t reg_ib_mem_size_in_mb = 0;
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

      CommandLineParser cp;
      cp.add_option_int("-ll:gsize", gasnet_mem_size_in_mb)
	.add_option_int("-ll:rsize", reg_mem_size_in_mb)
	.add_option_int("-ll:ib_rsize", reg_ib_mem_size_in_mb)
	.add_option_int("-ll:dsize", disk_mem_size_in_mb)
	.add_option_int("-ll:stacksize", stack_size_in_mb)
	.add_option_int("-ll:dma", dma_worker_threads)
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

      // these are actually parsed in activemsg.cc, but consume them here for now
      size_t dummy = 0;
      cp.add_option_int("-ll:numlmbs", dummy)
	.add_option_int("-ll:lmbsize", dummy)
	.add_option_int("-ll:forcelong", dummy)
	.add_option_int("-ll:sdpsize", dummy)
	.add_option_int("-ll:spillwarn", dummy)
	.add_option_int("-ll:spillstep", dummy)
	.add_option_int("-ll:spillstall", dummy);

      bool cmdline_ok = cp.parse_command_line(cmdline);

      if(!cmdline_ok) {
	fprintf(stderr, "ERROR: failure parsing command line options\n");
	gasnet_exit(1);
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

      // scan through what's left and see if anything starts with -ll: - probably a misspelled argument
      for(std::vector<std::string>::const_iterator it = cmdline.begin();
	  it != cmdline.end();
	  it++)
	if(it->compare(0, 4, "-ll:") == 0) {
	  fprintf(stderr, "ERROR: unrecognized lowlevel option: %s\n", it->c_str());
          assert(0);
	}

      // Check that we have enough resources for the number of nodes we are using
      if (gasnet_nodes() > MAX_NUM_NODES)
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but runtime is configured "
                       "for at most %d nodes. Update the 'MAX_NUM_NODES' macro "
                       "in legion_config.h", gasnet_nodes(), MAX_NUM_NODES);
        gasnet_exit(1);
      }
      if (gasnet_nodes() > (ID::MAX_NODE_ID + 1))
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes. Update the allocation "
		       "of bits in ID", gasnet_nodes(), (ID::MAX_NODE_ID + 1));
        gasnet_exit(1);
      }

      core_map = CoreMap::discover_core_map(hyperthread_sharing);
      core_reservations = new CoreReservationSet(core_map);

      sampling_profiler.configure_from_cmdline(cmdline, *core_reservations);

      // initialize barrier timestamp
      BarrierImpl::barrier_adjustment_timestamp = (((Barrier::timestamp_t)(gasnet_mynode())) << BarrierImpl::BARRIER_TIMESTAMP_NODEID_SHIFT) + 1;

      gasnet_handlerentry_t handlers[128];
      int hcount = 0;
      hcount += NodeAnnounceMessage::Message::add_handler_entries(&handlers[hcount], "Node Announce AM");
      hcount += SpawnTaskMessage::Message::add_handler_entries(&handlers[hcount], "Spawn Task AM");
      hcount += LockRequestMessage::Message::add_handler_entries(&handlers[hcount], "Lock Request AM");
      hcount += LockReleaseMessage::Message::add_handler_entries(&handlers[hcount], "Lock Release AM");
      hcount += LockGrantMessage::Message::add_handler_entries(&handlers[hcount], "Lock Grant AM");
      hcount += EventSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Event Subscribe AM");
      hcount += EventTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Event Trigger AM");
      hcount += EventUpdateMessage::Message::add_handler_entries(&handlers[hcount], "Event Update AM");
      hcount += RemoteMemAllocRequest::Request::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Request AM");
      hcount += RemoteMemAllocRequest::Response::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Response AM");
      hcount += CreateInstanceRequest::Request::add_handler_entries(&handlers[hcount], "Create Instance Request AM");
      hcount += CreateInstanceRequest::Response::add_handler_entries(&handlers[hcount], "Create Instance Response AM");
      hcount += RemoteCopyMessage::add_handler_entries(&handlers[hcount], "Remote Copy AM");
      hcount += RemoteFillMessage::add_handler_entries(&handlers[hcount], "Remote Fill AM");
      hcount += ValidMaskRequestMessage::Message::add_handler_entries(&handlers[hcount], "Valid Mask Request AM");
      hcount += ValidMaskDataMessage::Message::add_handler_entries(&handlers[hcount], "Valid Mask Data AM");
      hcount += ValidMaskFetchMessage::Message::add_handler_entries(&handlers[hcount], "Valid Mask Fetch AM");
#ifdef DETAILED_TIMING
      hcount += TimerDataRequestMessage::Message::add_handler_entries(&handlers[hcount], "Roll-up Request AM");
      hcount += TimerDataResponseMessage::Message::add_handler_entries(&handlers[hcount], "Roll-up Data AM");
      hcount += ClearTimersMessage::Message::add_handler_entries(&handlers[hcount], "Clear Timer Request AM");
#endif
      hcount += DestroyInstanceMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Instance AM");
      hcount += RemoteWriteMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write AM");
      hcount += RemoteReduceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Reduce AM");
      hcount += RemoteSerdezMessage::Message::add_handler_entries(&handlers[hcount], "Remote Serdez AM");
      hcount += RemoteWriteFenceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write Fence AM");
      hcount += RemoteWriteFenceAckMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write Fence Ack AM");
      hcount += DestroyLockMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Lock AM");
      hcount += RemoteReduceListMessage::Message::add_handler_entries(&handlers[hcount], "Remote Reduction List AM");
      hcount += RuntimeShutdownMessage::Message::add_handler_entries(&handlers[hcount], "Machine Shutdown AM");
      hcount += BarrierAdjustMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Adjust AM");
      hcount += BarrierSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Subscribe AM");
      hcount += BarrierTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Trigger AM");
      hcount += BarrierMigrationMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Migration AM");
      hcount += MetadataRequestMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Request AM");
      hcount += MetadataResponseMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Response AM");
      hcount += MetadataInvalidateMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Invalidate AM");
      hcount += MetadataInvalidateAckMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Inval Ack AM");
      hcount += XferDesRemoteWriteMessage::Message::add_handler_entries(&handlers[hcount], "XferDes Remote Write AM");
      hcount += XferDesRemoteWriteAckMessage::Message::add_handler_entries(&handlers[hcount], "XferDes Remote Write Ack AM");
      hcount += XferDesCreateMessage::Message::add_handler_entries(&handlers[hcount], "Create XferDes Request AM");
      hcount += XferDesDestroyMessage::Message::add_handler_entries(&handlers[hcount], "Destroy XferDes Request AM");
      hcount += NotifyXferDesCompleteMessage::Message::add_handler_entries(&handlers[hcount], "Notify XferDes Completion Request AM");
      hcount += UpdateBytesWriteMessage::Message::add_handler_entries(&handlers[hcount], "Update Bytes Write AM");
      hcount += UpdateBytesReadMessage::Message::add_handler_entries(&handlers[hcount], "Update Bytes Read AM");
      hcount += RegisterTaskMessage::Message::add_handler_entries(&handlers[hcount], "Register Task AM");
      hcount += RegisterTaskCompleteMessage::Message::add_handler_entries(&handlers[hcount], "Register Task Complete AM");
      hcount += RemoteIBAllocRequestAsync::Message::add_handler_entries(&handlers[hcount], "Remote IB Alloc Request AM");
      hcount += RemoteIBAllocResponseAsync::Message::add_handler_entries(&handlers[hcount], "Remote IB Alloc Response AM");
      hcount += RemoteIBFreeRequestAsync::Message::add_handler_entries(&handlers[hcount], "Remote IB Free Request AM");
      //hcount += TestMessage::add_handler_entries(&handlers[hcount], "Test AM");
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount], "Test 2 AM");

      init_endpoints(handlers, hcount, 
		     gasnet_mem_size_in_mb, reg_mem_size_in_mb, reg_ib_mem_size_in_mb,
		     *core_reservations,
		     *argc, (const char **)*argv);

#ifndef USE_GASNET
      // network initialization is also responsible for setting the "zero_time"
      //  for relative timing - no synchronization necessary in non-gasnet case
      Realm::Clock::set_zero_time();
#endif

      // Put this here so that it complies with the GASNet specification and
      // doesn't make any calls between gasnet_init and gasnet_attach
      gasnet_set_waitmode(GASNET_WAIT_BLOCK);

      nodes = new Node[gasnet_nodes()];

      // create allocators for local node events/locks/index spaces
      {
	Node& n = nodes[gasnet_mynode()];
	local_event_free_list = new EventTableAllocator::FreeList(n.events, gasnet_mynode());
	local_barrier_free_list = new BarrierTableAllocator::FreeList(n.barriers, gasnet_mynode());
	local_reservation_free_list = new ReservationTableAllocator::FreeList(n.reservations, gasnet_mynode());
	local_index_space_free_list = new IndexSpaceTableAllocator::FreeList(n.index_spaces, gasnet_mynode());
	local_proc_group_free_list = new ProcessorGroupTableAllocator::FreeList(n.proc_groups, gasnet_mynode());
      }

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
	    delay += gasnet_mynode() * atoi(pos + 1);
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

      LegionRuntime::LowLevel::create_builtin_dma_channels(this);

      LegionRuntime::LowLevel::start_dma_worker_threads(dma_worker_threads,
                                                        *core_reservations);

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

      Node *n = &nodes[gasnet_mynode()];

      // create memories and processors for all loaded modules
      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_memories(this);

      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_processors(this);

      LocalCPUMemory *regmem;
      if(reg_mem_size_in_mb > 0) {
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );
	char *regmem_base = ((char *)(seginfos[gasnet_mynode()].addr)) + (gasnet_mem_size_in_mb << 20);
	delete[] seginfos;
	Memory m = get_runtime()->next_local_memory_id();
	regmem = new LocalCPUMemory(m,
				    reg_mem_size_in_mb << 20,
				    regmem_base,
				    true);
	get_runtime()->add_memory(regmem);
      } else
	regmem = 0;

      LocalCPUMemory *reg_ib_mem;
      if(reg_ib_mem_size_in_mb > 0) {
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );
	char *reg_ib_mem_base = ((char *)(seginfos[gasnet_mynode()].addr)) + (gasnet_mem_size_in_mb << 20)
                                + (reg_mem_size_in_mb << 20);
	delete[] seginfos;
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
        sprintf(file_name, "disk_file%d.tmp", gasnet_mynode());
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
      LegionRuntime::LowLevel::start_dma_system(dma_worker_threads, 100
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
	const unsigned ADATA_SIZE = 4096;
	size_t adata[ADATA_SIZE];
	unsigned apos = 0;

	unsigned num_procs = 0;
	unsigned num_memories = 0;
	unsigned num_ib_memories = 0;

	// announce each processor
	for(std::vector<ProcessorImpl *>::const_iterator it = n->processors.begin();
	    it != n->processors.end();
	    it++)
	  if(*it) {
	    Processor p = (*it)->me;
	    Processor::Kind k = (*it)->me.kind();
	    int num_cores = (*it)->num_cores;

	    num_procs++;
	    adata[apos++] = NODE_ANNOUNCE_PROC;
	    adata[apos++] = p.id;
	    adata[apos++] = k;
	    adata[apos++] = num_cores;
	  }

	// now each memory
	for(std::vector<MemoryImpl *>::const_iterator it = n->memories.begin();
	    it != n->memories.end();
	    it++)
	  if(*it) {
	    Memory m = (*it)->me;
	    Memory::Kind k = (*it)->me.kind();

	    num_memories++;
	    adata[apos++] = NODE_ANNOUNCE_MEM;
	    adata[apos++] = m.id;
	    adata[apos++] = k;
	    adata[apos++] = (*it)->size;
	    adata[apos++] = reinterpret_cast<size_t>((*it)->local_reg_base());
	  }

        for (std::vector<MemoryImpl *>::const_iterator it = n->ib_memories.begin();
             it != n->ib_memories.end();
             it++)
          if(*it) {
            Memory m = (*it)->me;
            Memory::Kind k = (*it)->me.kind();

            num_ib_memories++;
            adata[apos++] = NODE_ANNOUNCE_IB_MEM;
            adata[apos++] = m.id;
            adata[apos++] = k;
            adata[apos++] = (*it)->size;
            adata[apos++] = reinterpret_cast<size_t>((*it)->local_reg_base());
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
	      adata[apos++] = NODE_ANNOUNCE_PMA;
	      adata[apos++] = it2->p.id;
	      adata[apos++] = it2->m.id;
	      adata[apos++] = it2->bandwidth;
	      adata[apos++] = it2->latency;
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
	      if((it2->m1 != m) || (it2->m2.address_space() != gasnet_mynode()))
		continue;

	      adata[apos++] = NODE_ANNOUNCE_MMA;
	      adata[apos++] = it2->m1.id;
	      adata[apos++] = it2->m2.id;
	      adata[apos++] = it2->bandwidth;
	      adata[apos++] = it2->latency;
	    }
	  }

	adata[apos++] = NODE_ANNOUNCE_DONE;
	assert(apos < ADATA_SIZE);

#ifdef DEBUG_REALM_STARTUP
	if(gasnet_mynode() == 0) {
	  TimeStamp ts("sending announcements", false);
	  fflush(stdout);
	}
#endif

	// now announce ourselves to everyone else
	for(unsigned i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    NodeAnnounceMessage::send_request(i,
						     num_procs,
						     num_memories,
						     num_ib_memories,
						     adata, apos*sizeof(adata[0]),
						     PAYLOAD_COPY);

	NodeAnnounceMessage::await_all_announcements();

#ifdef DEBUG_REALM_STARTUP
	if(gasnet_mynode() == 0) {
	  TimeStamp ts("received all announcements", false);
	  fflush(stdout);
	}
#endif
      }

      return true;
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
      log_collective.fatal() << "collective mismatch on node " << gasnet_mynode() << " for " << name << ": " << val << " != " << bval;
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

      if((int)gasnet_mynode() == root) {
	// ROOT NODE

	// step 1: receive wait_on from every node
	Event *all_events = 0;
	all_events = new Event[gasnet_nodes()];
	gasnet_coll_gather(GASNET_TEAM_ALL, root, all_events, &wait_on, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: merge all the events
	std::set<Event> event_set;
	for(int i = 0; i < (int)gasnet_nodes(); i++) {
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

      if(gasnet_mynode() == 0) {
	// ROOT NODE

	// step 1: receive wait_on from every node
	Event *all_events = 0;
	all_events = new Event[gasnet_nodes()];
	gasnet_coll_gather(GASNET_TEAM_ALL, 0, all_events, &wait_on, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: merge all the events
	std::set<Event> event_set;
	for(int i = 0; i < (int)gasnet_nodes(); i++) {
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

      const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;

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
      if(gasnet_mynode() == 0) {
	// ROOT NODE

	// step 1: receive wait_on from every node
	Event *all_events = 0;
	all_events = new Event[gasnet_nodes()];
	gasnet_coll_gather(GASNET_TEAM_ALL, 0, all_events, &my_finish, sizeof(Event), GASNET_COLL_FLAGS);

	// step 2: merge all the events
	std::set<Event> event_set;
	for(int i = 0; i < (int)gasnet_nodes(); i++) {
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

      wait_for_shutdown();
      exit(0);
    }

    // this is not member data of RuntimeImpl because we don't want use-after-free problems
    static int shutdown_count = 0;

    void RuntimeImpl::shutdown(bool local_request /*= true*/)
    {
      // filter out duplicate requests
      bool already_started = (__sync_fetch_and_add(&shutdown_count, 1) > 0);
      if(already_started)
	return;

      if(local_request) {
	log_runtime.info("shutdown request - notifying other nodes");
	for(unsigned i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    RuntimeShutdownMessage::send_request(i);
      }

      log_runtime.info("shutdown request - cleaning up local processors");

      if(run_method_called) {
	// legacy shutdown - call shutdown task on processors
	log_task.info("spawning processor shutdown task on local cpus");

	const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;

	spawn_on_all(local_procs, Processor::TASK_ID_PROCESSOR_SHUTDOWN, 0, 0,
		     Event::NO_EVENT,
		     INT_MIN); // runs with lowest priority
      }

      {
	AutoHSLLock al(shutdown_mutex);
	shutdown_requested = true;
	shutdown_condvar.broadcast();
      }
    }

    void RuntimeImpl::wait_for_shutdown(void)
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
      LegionRuntime::LowLevel::stop_dma_worker_threads();
      LegionRuntime::LowLevel::stop_dma_system();
      stop_activemsg_threads();

      sampling_profiler.shutdown();

      {
	std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;
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
               gasnet_mynode(),
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
	for(gasnet_node_t i = 0; i < gasnet_nodes(); i++) {
	  Node& n = nodes[i];

	  delete_container_contents(n.memories);
	  delete_container_contents(n.processors);
	}
	
	delete[] nodes;
	delete global_memory;
	delete local_event_free_list;
	delete local_barrier_free_list;
	delete local_reservation_free_list;
	delete local_index_space_free_list;
	delete local_proc_group_free_list;

	// delete all the DMA channels that we were given
	delete_container_contents(dma_channels);

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

      if(!Threading::cleanup()) exit(1);

      // this terminates the process, so control never gets back to caller
      // would be nice to fix this...
      //if (exit_process)
      //  gasnet_exit(0);
    }

    EventImpl *RuntimeImpl::get_event_impl(Event e)
    {
      ID id(e);
      if(id.is_event())
	return get_genevent_impl(e);
      if(id.is_barrier())
	return get_barrier_impl(e);
      assert(0);
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

    ReservationImpl *RuntimeImpl::get_lock_impl(ID id)
    {
      if(id.is_reservation()) {
	Node *n = &nodes[id.rsrv.creator_node];
	ReservationImpl *impl = n->reservations.lookup_entry(id.rsrv.rsrv_idx, id.rsrv.creator_node);
	assert(impl->me == id.convert<Reservation>());
	return impl;
      }

      if(id.is_idxspace())
	return &(get_index_space_impl(id)->lock);

      if(id.is_instance())
	return &(get_instance_impl(id)->lock);

      if(id.is_procgroup())
	return &(get_procgroup_impl(id)->lock);

      assert(0);
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
      assert(0);
      return 0;
    }

    ProcessorImpl *RuntimeImpl::get_processor_impl(ID id)
    {
      if(id.is_procgroup())
	return get_procgroup_impl(id);

      assert(id.is_processor());
      return null_check(nodes[id.proc.owner_node].processors[id.proc.proc_idx]);
    }

    ProcessorGroup *RuntimeImpl::get_procgroup_impl(ID id)
    {
      assert(id.is_procgroup());

      Node *n = &nodes[id.pgroup.owner_node];
      ProcessorGroup *impl = n->proc_groups.lookup_entry(id.pgroup.pgroup_idx,
							 id.pgroup.owner_node);
      assert(impl->me == id.convert<Processor>());
      return impl;
    }

    IndexSpaceImpl *RuntimeImpl::get_index_space_impl(ID id)
    {
      assert(id.is_idxspace());

      Node *n = &nodes[id.idxspace.owner_node];
      IndexSpaceImpl *impl = n->index_spaces.lookup_entry(id.idxspace.idxspace_idx,
							  id.idxspace.owner_node);
      assert(impl->me == id.convert<IndexSpace>());
      return impl;
    }

    RegionInstanceImpl *RuntimeImpl::get_instance_impl(ID id)
    {
      assert(id.is_instance());
      MemoryImpl *mem = get_memory_impl(id);
      
      AutoHSLLock al(mem->mutex);

      // TODO: factor creator_node into lookup!
      if(id.instance.inst_idx >= mem->instances.size()) {
	assert(id.instance.owner_node != gasnet_mynode());

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
	  //printf("[%d] creating proxy instance: inst=" IDFMT "\n", gasnet_mynode(), id.id());
	  mem->instances[id.instance.inst_idx] = new RegionInstanceImpl(id.convert<RegionInstance>(), mem->me);
	}
      }
	  
      return mem->instances[id.instance.inst_idx];
    }

    /*static*/
    void RuntimeImpl::realm_backtrace(int signal)
    {
      assert((signal == SIGILL) || (signal == SIGFPE) || 
             (signal == SIGABRT) || (signal == SIGSEGV) ||
             (signal == SIGBUS));
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
              gasnet_mynode(), (unsigned long)pthread_self(), buffer);
      fflush(stderr);
      free(buffer);
      free(funcname);
      // returning would almost certainly cause this signal to be raised again,
      //  so sleep for a second in case other threads also want to chronicle
      //  their own deaths, and then exit
      sleep(1);
      exit(1);
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
    log_runtime.info("received shutdown request from node %d", args.initiating_node);

    get_runtime()->shutdown(false);
  }

  /*static*/ void RuntimeShutdownMessage::send_request(gasnet_node_t target)
  {
    RequestArgs args;

    args.initiating_node = gasnet_mynode();
    args.dummy = 0;
    Message::request(target, args);
  }

  
}; // namespace Realm
