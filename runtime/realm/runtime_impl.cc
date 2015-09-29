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

// Runtime implementation for Realm

#include "runtime_impl.h"

#include "proc_impl.h"
#include "mem_impl.h"
#include "inst_impl.h"

#include "activemsg.h"

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

#ifdef USE_CUDA
#include "lowlevel_gpu.h"
namespace Realm {
  typedef LegionRuntime::LowLevel::GPUProcessor GPUProcessor;
  typedef LegionRuntime::LowLevel::GPUFBMemory GPUFBMemory;
  typedef LegionRuntime::LowLevel::GPUZCMemory GPUZCMemory;
  typedef LegionRuntime::LowLevel::GPUWorker GPUWorker;
};
#endif

#include <unistd.h>
#include <signal.h>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

namespace Realm {

  Logger log_runtime("realm");
  extern Logger log_task; // defined in proc_impl.cc
  
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
    
    bool Runtime::register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->task_table.count(taskid) > 0)
	return false;

      ((RuntimeImpl *)impl)->task_table[taskid] = taskptr;
      return true;
    }

    bool Runtime::register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->reduce_op_table.count(redop_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->reduce_op_table[redop_id] = redop;
      return true;
    }

    void Runtime::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/,
                      bool background /*= false*/)
    {
      ((RuntimeImpl *)impl)->run(task_id, style, args, arglen, background);
    }

    void Runtime::shutdown(void)
    {
      ((RuntimeImpl *)impl)->shutdown(true); // local request
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

  REGISTER_REALM_MODULE(CoreModule);

  CoreModule::CoreModule(void)
    : Module("core")
    , num_cpu_procs(1), num_util_procs(1), num_io_procs(1)
    , concurrent_io_threads(1)  // Legion does not support values > 1 right now
    , sysmem_size_in_mb(512), stack_size_in_mb(2)
  {}

  CoreModule::~CoreModule(void)
  {}

  /*static*/ Module *CoreModule::create_module(RuntimeImpl *runtime,
					       std::vector<std::string>& cmdline)
  {
    Module *m = new CoreModule;

#if 0
    // parse command line arguments
    CommandLineParser cp;
    cp.add_option_int("-ll:cpu", m->num_cpu_procs)
      .add_option_int("-ll:util", m->num_util_procs)
      .add_option_int("-ll:io", m->num_io_procs)
      .add_option_int("-ll:concurrent_io", m->concurrent_io_threads)
      .add_option_int("-ll:csize", m->sysmem_size_in_mb)
      .add_option_int("-ll:stacksize", m->stack_size_in_mb, true /*keep*/)
      .parse_cmdline(cmdline);
#endif

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

    // no code translators
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
#ifdef OLD_INIT
    static std::vector<LocalTaskProcessor *> local_cpus;
    static std::vector<LocalTaskProcessor *> local_util_procs;
    static std::vector<LocalTaskProcessor *> local_io_procs;
#endif
    static size_t stack_size_in_mb;
#ifdef USE_CUDA
    static std::vector<GPUProcessor *> local_gpus;
    static std::map<GPUProcessor *, GPUFBMemory *> gpu_fbmems;
    static std::map<GPUProcessor *, GPUZCMemory *> gpu_zcmems;
#endif
  
    RuntimeImpl::RuntimeImpl(void)
      : machine(0), nodes(0), global_memory(0),
	local_event_free_list(0), local_barrier_free_list(0),
	local_reservation_free_list(0), local_index_space_free_list(0),
	local_proc_group_free_list(0), background_pthread(0),
	shutdown_requested(false), shutdown_condvar(shutdown_mutex),
	num_local_memories(0), num_local_processors(0),
	module_registrar(this)
    {
      machine = new MachineImpl;
    }

    RuntimeImpl::~RuntimeImpl(void)
    {
      delete machine;
    }

    Memory RuntimeImpl::next_local_memory_id(void)
    {
      Memory m = ID(ID::ID_MEMORY, 
		    gasnet_mynode(), 
		    num_local_memories++, 0).convert<Memory>();
      return m;
    }

    Processor RuntimeImpl::next_local_processor_id(void)
    {
      Processor p = ID(ID::ID_PROCESSOR, 
		       gasnet_mynode(), 
		       num_local_processors++).convert<Processor>();
      return p;
    }

    void RuntimeImpl::add_memory(MemoryImpl *m)
    {
      // right now expect this to always be for the current node and the next memory ID
      assert((ID(m->me).node() == gasnet_mynode()) &&
	     (ID(m->me).index_h() == nodes[gasnet_mynode()].memories.size()));

      nodes[gasnet_mynode()].memories.push_back(m);
    }

    void RuntimeImpl::add_processor(ProcessorImpl *p)
    {
      // right now expect this to always be for the current node and the next processor ID
      assert((ID(p->me).node() == gasnet_mynode()) &&
	     (ID(p->me).index() == nodes[gasnet_mynode()].processors.size()));

      nodes[gasnet_mynode()].processors.push_back(p);
    }

    CoreReservationSet& RuntimeImpl::core_reservation_set(void)
    {
      return core_reservations;
    }

    static void add_proc_mem_affinity(const std::set<Processor>& procs,
				      const std::set<Memory>& mems,
				      int bandwidth,
				      int latency,
				      size_t *adata,
				      unsigned& apos)
    {
      for(std::set<Processor>::const_iterator it1 = procs.begin();
	  it1 != procs.end();
	  it1++) 
	for(std::set<Memory>::const_iterator it2 = mems.begin();
	    it2 != mems.end();
	    it2++) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it1).id;
	  adata[apos++] = (*it2).id;
	  adata[apos++] = bandwidth;
	  adata[apos++] = latency;
	}
    }

    static void add_mem_mem_affinity(const std::set<Memory>& mems1,
				     const std::set<Memory>& mems2,
				     int bandwidth,
				     int latency,
				     size_t *adata,
				     unsigned& apos)
    {
      for(std::set<Memory>::const_iterator it1 = mems1.begin();
	  it1 != mems1.end();
	  it1++) 
	for(std::set<Memory>::const_iterator it2 = mems2.begin();
	    it2 != mems2.end();
	    it2++) {
	  adata[apos++] = NODE_ANNOUNCE_MMA;
	  adata[apos++] = (*it1).id;
	  adata[apos++] = (*it2).id;
	  adata[apos++] = bandwidth;
	  adata[apos++] = latency;
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

      DetailedTimer::init_timers();

      // gasnet_init() must be called before parsing command line arguments, as some
      //  spawners (e.g. the ssh spawner for gasnetrun_ibv) start with bogus args and
      //  fetch the real ones from somewhere during gasnet_init()

      //GASNetNode::my_node = new GASNetNode(argc, argv, this);
      // SJT: WAR for issue on Titan with duplicate cookies on Gemini
      //  communication domains
      char *orig_pmi_gni_cookie = getenv("PMI_GNI_COOKIE");
      if(orig_pmi_gni_cookie) {
        char *new_pmi_gni_cookie = (char *)malloc(256);
        sprintf(new_pmi_gni_cookie, "PMI_GNI_COOKIE=%d", 1+atoi(orig_pmi_gni_cookie));
        //printf("changing PMI cookie to: '%s'\n", new_pmi_gni_cookie);
        putenv(new_pmi_gni_cookie);  // libc now owns the memory
      }
      // SJT: another GASNET workaround - if we don't have GASNET_IB_SPAWNER set, assume it was MPI
      if(!getenv("GASNET_IB_SPAWNER"))
	putenv(strdup("GASNET_IB_SPAWNER=mpi"));
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
      std::vector<std::string> cmdline(*argc - 1);
      for(int i = 1; i < *argc; i++)
	cmdline[i - 1] = (*argv)[i];

      // very first thing - let the logger initialization happen
      Logger::configure_from_cmdline(*argc, (const char **)*argv);

      // now load modules
      module_registrar.create_static_modules(cmdline, modules);
      module_registrar.create_dynamic_modules(cmdline, modules);

      // low-level runtime parameters
#ifdef USE_GASNET
      size_t gasnet_mem_size_in_mb = 256;
#else
      size_t gasnet_mem_size_in_mb = 0;
#endif
#ifdef OLD_INIT
      size_t cpu_mem_size_in_mb = 512;
#endif
      size_t reg_mem_size_in_mb = 0;
      size_t disk_mem_size_in_mb = 0;
      // Static variable for stack size since we need to 
      // remember it when we launch threads in run 
      stack_size_in_mb = 2;
#ifdef OLD_INIT
      unsigned num_local_cpus = 1;
      unsigned num_util_procs = 1;
      unsigned num_io_procs = 0;
      unsigned concurrent_io_threads = 1; // Legion does not support values > 1 right now
#endif
      //unsigned cpu_worker_threads = 1;
      unsigned dma_worker_threads = 1;
      unsigned active_msg_worker_threads = 1;
      unsigned active_msg_handler_threads = 1;
#ifdef USE_CUDA
      size_t zc_mem_size_in_mb = 64;
      size_t fb_mem_size_in_mb = 256;
      unsigned num_local_gpus = 0;
      unsigned num_gpu_streams = 12;
      bool     gpu_worker_thread = true;
      bool pin_sysmem_for_gpu = true;
#endif
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

      for(int i = 1; i < *argc; i++) {
#define INT_ARG(argname, varname)                       \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = atoi((*argv)[++i]);		\
	    continue;					\
	  }

#define BOOL_ARG(argname, varname)                      \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = true;				\
	    continue;					\
	  }

	INT_ARG("-ll:gsize", gasnet_mem_size_in_mb);
#ifdef OLD_INIT
	INT_ARG("-ll:csize", cpu_mem_size_in_mb);
#endif
	INT_ARG("-ll:rsize", reg_mem_size_in_mb);
        INT_ARG("-ll:dsize", disk_mem_size_in_mb);
        INT_ARG("-ll:stacksize", stack_size_in_mb);
#ifdef OLD_INIT
	INT_ARG("-ll:cpu", num_local_cpus);
	INT_ARG("-ll:util", num_util_procs);
        INT_ARG("-ll:io", num_io_procs);
	INT_ARG("-ll:concurrent_io", concurrent_io_threads);
#endif
	//INT_ARG("-ll:workers", cpu_worker_threads);
	INT_ARG("-ll:dma", dma_worker_threads);
	INT_ARG("-ll:amsg", active_msg_worker_threads);
	INT_ARG("-ll:ahandlers", active_msg_handler_threads);
	INT_ARG("-ll:dummy_rsrv_ok", dummy_reservation_ok);
	INT_ARG("-ll:show_rsrv", show_reservations);
#ifdef USE_CUDA
	INT_ARG("-ll:fsize", fb_mem_size_in_mb);
	INT_ARG("-ll:zsize", zc_mem_size_in_mb);
	INT_ARG("-ll:gpu", num_local_gpus);
        INT_ARG("-ll:streams", num_gpu_streams);
        BOOL_ARG("-ll:gpuworker", gpu_worker_thread);
        INT_ARG("-ll:pin", pin_sysmem_for_gpu);
#endif

	if(!strcmp((*argv)[i], "-ll:eventtrace")) {
#ifdef EVENT_TRACING
	  event_trace_file = strdup((*argv)[++i]);
#else
	  fprintf(stderr, "WARNING: event tracing requested, but not enabled at compile time!\n");
#endif
	  continue;
	}

        if (!strcmp((*argv)[i], "-ll:locktrace"))
        {
#ifdef LOCK_TRACING
          lock_trace_file = strdup((*argv)[++i]);
#else
          fprintf(stderr, "WARNING: lock tracing requested, but not enabled at compile time!\n");
#endif
          continue;
        }

        if (!strcmp((*argv)[i], "-ll:prefix"))
        {
#ifdef NODE_LOGGING
          RuntimeImpl::prefix = strdup((*argv)[++i]);
#else
          fprintf(stderr,"WARNING: prefix set, but NODE_LOGGING not enabled at compile time!\n");
#endif
          continue;
        }

        // Skip arguments that parsed in activemsg.cc
        if (!strcmp((*argv)[i], "-ll:numlmbs") || !strcmp((*argv)[i],"-ll:lmbsize") ||
            !strcmp((*argv)[i], "-ll:forcelong") || !strcmp((*argv)[i],"-ll:sdpsize"))
        {
          i++;
          continue;
        }

        if (strncmp((*argv)[i], "-ll:", 4) == 0)
        {
	  fprintf(stderr, "ERROR: unrecognized lowlevel option: %s\n", (*argv)[i]);
          assert(0);
	}
      }

      // Check that we have enough resources for the number of nodes we are using
      if (gasnet_nodes() > MAX_NUM_NODES)
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but runtime is configured "
                       "for at most %d nodes. Update the 'MAX_NUM_NODES' macro "
                       "in legion_types.h", gasnet_nodes(), MAX_NUM_NODES);
        gasnet_exit(1);
      }
      if (gasnet_nodes() > (1 << ID::NODE_BITS))
      {
#ifdef LEGION_IDS_ARE_64BIT
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes. Update the allocation "
                       "of bits in ID", gasnet_nodes(), (1 << ID::NODE_BITS));
#else
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes.  Update the allocation "
                       "of bits in ID or switch to 64-bit IDs with the "
                       "-DLEGION_IDS_ARE_64BIT compile-time flag",
                       gasnet_nodes(), (1 << ID::NODE_BITS));
#endif
        gasnet_exit(1);
      }

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
      hcount += RemoteMemAllocRequest::Request::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Request AM");
      hcount += RemoteMemAllocRequest::Response::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Response AM");
      hcount += CreateInstanceRequest::Request::add_handler_entries(&handlers[hcount], "Create Instance Request AM");
      hcount += CreateInstanceRequest::Response::add_handler_entries(&handlers[hcount], "Create Instance Response AM");
      hcount += RemoteCopyMessage::add_handler_entries(&handlers[hcount], "Remote Copy AM");
      hcount += RemoteFillMessage::add_handler_entries(&handlers[hcount], "Remote Fill AM");
      hcount += ValidMaskRequestMessage::Message::add_handler_entries(&handlers[hcount], "Valid Mask Request AM");
      hcount += ValidMaskDataMessage::Message::add_handler_entries(&handlers[hcount], "Valid Mask Data AM");
#ifdef DETAILED_TIMING
      hcount += TimerDataRequestMessage::Message::add_handler_entries(&handlers[hcount], "Roll-up Request AM");
      hcount += TimerDataResponseMessage::Message::add_handler_entries(&handlers[hcount], "Roll-up Data AM");
      hcount += ClearTimersMessage::Message::add_handler_entries(&handlers[hcount], "Clear Timer Request AM");
#endif
      hcount += DestroyInstanceMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Instance AM");
      hcount += RemoteWriteMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write AM");
      hcount += RemoteReduceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Reduce AM");
      hcount += RemoteWriteFenceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write Fence AM");
      hcount += RemoteWriteFenceAckMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write Fence Ack AM");
      hcount += DestroyLockMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Lock AM");
      hcount += RemoteReduceListMessage::Message::add_handler_entries(&handlers[hcount], "Remote Reduction List AM");
      hcount += RuntimeShutdownMessage::Message::add_handler_entries(&handlers[hcount], "Machine Shutdown AM");
      hcount += BarrierAdjustMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Adjust AM");
      hcount += BarrierSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Subscribe AM");
      hcount += BarrierTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Trigger AM");
      hcount += MetadataRequestMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Request AM");
      hcount += MetadataResponseMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Response AM");
      hcount += MetadataInvalidateMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Invalidate AM");
      hcount += MetadataInvalidateAckMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Inval Ack AM");
      //hcount += TestMessage::add_handler_entries(&handlers[hcount], "Test AM");
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount], "Test 2 AM");

      init_endpoints(handlers, hcount, 
		     gasnet_mem_size_in_mb, reg_mem_size_in_mb,
		     core_reservations,
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
#if defined(REALM_BACKTRACE) || defined(LEGION_BACKTRACE)
      signal(SIGSEGV, realm_backtrace);
      signal(SIGABRT, realm_backtrace);
      signal(SIGFPE,  realm_backtrace);
      signal(SIGILL,  realm_backtrace);
      signal(SIGBUS,  realm_backtrace);
#endif
      if ((getenv("LEGION_FREEZE_ON_ERROR") != NULL) ||
          (getenv("REALM_FREEZE_ON_ERROR") != NULL))
      {
        signal(SIGSEGV, realm_freeze);
        signal(SIGABRT, realm_freeze);
        signal(SIGFPE,  realm_freeze);
        signal(SIGILL,  realm_freeze);
        signal(SIGBUS,  realm_freeze);
      }
      
      start_polling_threads(active_msg_worker_threads);

      start_handler_threads(active_msg_handler_threads,
			    core_reservations,
			    stack_size_in_mb << 20);

      LegionRuntime::LowLevel::start_dma_worker_threads(dma_worker_threads,
							core_reservations);

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
	
      //gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
      //CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

      if(gasnet_mem_size_in_mb > 0)
	global_memory = new GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>(), gasnet_mem_size_in_mb << 20);
      else
	global_memory = 0;

      Node *n = &nodes[gasnet_mynode()];

      printf("%zd modules\n", modules.size());

      // create memories and processors for all loaded modules
      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_memories(this);

      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_processors(this);

#ifdef OLD_INIT
      // create utility processors (if any)
      if (num_util_procs > 0)
      {
        for(unsigned i = 0; i < num_util_procs; i++) {
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
          LocalUtilityProcessor *up = new LocalUtilityProcessor(p, core_reservations,
								stack_size_in_mb << 20);
          n->processors.push_back(up);
          local_util_procs.push_back(up);
        }
      }
      // create i/o processors (if any)
      if (num_io_procs > 0)
      {
        for (unsigned i = 0; i < num_io_procs; i++) {
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
	  LocalIOProcessor *io = new LocalIOProcessor(p, core_reservations,
						      stack_size_in_mb << 20,
						      concurrent_io_threads);
          n->processors.push_back(io);
          local_io_procs.push_back(io);
        }
      }
#endif

#ifdef USE_CUDA
      // Keep track of the local system memories so we can pin them
      // after we've initialized the GPU
      std::vector<LocalCPUMemory*> local_mems;
      // Figure out which GPUs support peer access (if any)
      // and prioritize them so they are used first
      std::vector<int> peer_gpus;
      std::vector<int> dumb_gpus;
      // only do this if gpus have been requested
      if(num_local_gpus > 0) {
	// Initialize the driver API
	CHECK_CU( cuInit(0) );
	{
	  int num_devices;
	  CHECK_CU( cuDeviceGetCount(&num_devices) );
	  for (int i = 0; i < num_devices; i++)
	  {
	    CUdevice device;
	    CHECK_CU( cuDeviceGet(&device, i) );
	    bool has_peer = false;
	    // Go through all the other devices and see
	    // if we have peer access to them
	    for (int j = 0; j < num_devices; j++)
            {
	      if (i == j) continue;
	      CUdevice peer;
	      CHECK_CU( cuDeviceGet(&peer, j) );
	      int can_access;
	      CHECK_CU( cuDeviceCanAccessPeer(&can_access, device, peer) );
	      if (can_access)
	      {
		has_peer = true;
		break;
	      }
	    }
	    if (has_peer)
	      peer_gpus.push_back(i);
	    else
	      dumb_gpus.push_back(i);
	  }
	}
      }
#endif

#ifdef OLD_INIT
      // create local processors
      for(unsigned i = 0; i < num_local_cpus; i++) {
	Processor p = ID(ID::ID_PROCESSOR, 
			 gasnet_mynode(), 
			 n->processors.size()).convert<Processor>();
        LocalTaskProcessor *lp;
	lp = new LocalCPUProcessor(p, core_reservations, stack_size_in_mb << 20);
	n->processors.push_back(lp);
	local_cpus.push_back(lp);
      }

      // create local memory
      LocalCPUMemory *cpumem;
      if(cpu_mem_size_in_mb > 0) {
	cpumem = new LocalCPUMemory(ID(ID::ID_MEMORY, 
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    cpu_mem_size_in_mb << 20);
	n->memories.push_back(cpumem);
#ifdef USE_CUDA
        local_mems.push_back(cpumem);
#endif
      } else
	cpumem = 0;
#endif

      LocalCPUMemory *regmem;
      if(reg_mem_size_in_mb > 0) {
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );
	char *regmem_base = ((char *)(seginfos[gasnet_mynode()].addr)) + (gasnet_mem_size_in_mb << 20);
	delete[] seginfos;
	regmem = new LocalCPUMemory(ID(ID::ID_MEMORY,
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    reg_mem_size_in_mb << 20,
				    regmem_base,
				    true);
	n->memories.push_back(regmem);
#ifdef USE_CUDA
        local_mems.push_back(regmem);
#endif
      } else
	regmem = 0;

      // create local disk memory
      DiskMemory *diskmem;
      if(disk_mem_size_in_mb > 0) {
        diskmem = new DiskMemory(ID(ID::ID_MEMORY,
                                    gasnet_mynode(),
                                    n->memories.size(), 0).convert<Memory>(),
                                 disk_mem_size_in_mb << 20,
                                 "disk_file.tmp");
        n->memories.push_back(diskmem);
      } else
        diskmem = 0;

#ifdef USE_HDF
      // create HDF memory
      HDFMemory *hdfmem;
      hdfmem = new HDFMemory(ID(ID::ID_MEMORY,
                                gasnet_mynode(),
                                n->memories.size(), 0).convert<Memory>());
      n->memories.push_back(hdfmem);
#endif



#ifdef USE_CUDA
      if(num_local_gpus > 0) {
        if (num_local_gpus > (peer_gpus.size() + dumb_gpus.size()))
        {
          printf("Requested %d GPUs, but only %ld GPUs exist on node %d\n",
            num_local_gpus, peer_gpus.size()+dumb_gpus.size(), gasnet_mynode());
          assert(false);
        }

	// TODO: let the CUDA module actually parse config variables
	assert(gpu_worker_thread == true);

	for(unsigned i = 0; i < num_local_gpus; i++) {
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
	  //printf("GPU's ID is " IDFMT "\n", p.id);
 	  GPUProcessor *gp = new GPUProcessor(p, core_reservations,
                                              (i < peer_gpus.size() ?
                                                peer_gpus[i] : 
                                                dumb_gpus[i-peer_gpus.size()]), 
                                              zc_mem_size_in_mb << 20,
                                              fb_mem_size_in_mb << 20,
                                              stack_size_in_mb << 20,
                                              num_gpu_streams);
	  n->processors.push_back(gp);
	  local_gpus.push_back(gp);

	  Memory m = ID(ID::ID_MEMORY,
			gasnet_mynode(),
			n->memories.size(), 0).convert<Memory>();
	  GPUFBMemory *fbm = new GPUFBMemory(m, gp);
	  n->memories.push_back(fbm);

	  gpu_fbmems[gp] = fbm;

	  Memory m2 = ID(ID::ID_MEMORY,
			 gasnet_mynode(),
			 n->memories.size(), 0).convert<Memory>();
	  GPUZCMemory *zcm = new GPUZCMemory(m2, gp);
	  n->memories.push_back(zcm);

	  gpu_zcmems[gp] = zcm;
	}
        // Now pin any CPU memories
        if(pin_sysmem_for_gpu)
          for (unsigned idx = 0; idx < local_mems.size(); idx++)
            local_mems[idx]->pin_memory(local_gpus[0]);

        // Register peer access for any GPUs which support it
        if ((num_local_gpus > 1) && (peer_gpus.size() > 1))
        {
          unsigned peer_count = (num_local_gpus < peer_gpus.size()) ? 
                                  num_local_gpus : peer_gpus.size();
          // Needs to go both ways so register in all directions
          for (unsigned i = 0; i < peer_count; i++)
          {
            CUdevice device;
            CHECK_CU( cuDeviceGet(&device, peer_gpus[i]) );
            for (unsigned j = 0; j < peer_count; j++)
            {
              if (i == j) continue;
              CUdevice peer;
              CHECK_CU( cuDeviceGet(&peer, peer_gpus[j]) );
              int can_access;
              CHECK_CU( cuDeviceCanAccessPeer(&can_access, device, peer) );
              if (can_access)
                local_gpus[i]->enable_peer_access(local_gpus[j]);
            }
          }
        }
      }
#endif

      // now that we've created all the processors/etc., we can try to come up with core
      //  allocations that satisfy everybody's requirements - this will also start up any
      //  threads that have already been requested
      bool ok = core_reservations.satisfy_reservations(dummy_reservation_ok);
      if(ok) {
	if(show_reservations) {
	  std::cout << *core_reservations.get_core_map() << std::endl;
	  core_reservations.report_reservations(std::cout);
	}
      } else {
	printf("HELP!  Could not satisfy all core reservations!\n");
	exit(1);
      }

      {
	const unsigned ADATA_SIZE = 4096;
	size_t adata[ADATA_SIZE];
	unsigned apos = 0;

	unsigned num_procs = 0;
	unsigned num_memories = 0;

        // iterate over all local processors and add announcements for them
	std::map<Processor::Kind, std::set<Processor> > procs_by_kind;

	for(std::vector<ProcessorImpl *>::const_iterator it = n->processors.begin();
	    it != n->processors.end();
	    it++)
	  if(*it) {
	    Processor p = (*it)->me;
	    Processor::Kind k = (*it)->me.kind();

	    num_procs++;
	    adata[apos++] = NODE_ANNOUNCE_PROC;
	    adata[apos++] = p.id;
	    adata[apos++] = k;

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

	    num_memories++;
	    adata[apos++] = NODE_ANNOUNCE_MEM;
	    adata[apos++] = m.id;
	    adata[apos++] = k;
	    adata[apos++] = (*it)->size;
	    adata[apos++] = reinterpret_cast<size_t>((*it)->local_reg_base());

	    mems_by_kind[k].insert(m);
	  }

	if(global_memory)
	  mems_by_kind[Memory::GLOBAL_MEM].insert(global_memory->me);

	// affinities for now are a big hack
	std::set<Processor::Kind> local_cpu_kinds;
	local_cpu_kinds.insert(Processor::LOC_PROC);
	local_cpu_kinds.insert(Processor::UTIL_PROC);
	local_cpu_kinds.insert(Processor::IO_PROC);

	for(std::set<Processor::Kind>::const_iterator it = local_cpu_kinds.begin();
	    it != local_cpu_kinds.end();
	    it++) {
	  Processor::Kind k = *it;

	  add_proc_mem_affinity(procs_by_kind[k],
				mems_by_kind[Memory::SYSTEM_MEM],
				100, // "large" bandwidth
				1,   // "small" latency
				adata, apos);

	  add_proc_mem_affinity(procs_by_kind[k],
				mems_by_kind[Memory::REGDMA_MEM],
				80,  // "large" bandwidth
				5,   // "small" latency
				adata, apos);

	  add_proc_mem_affinity(procs_by_kind[k],
				mems_by_kind[Memory::DISK_MEM],
				5,   // "low" bandwidth
				100, // "high" latency
				adata, apos);
	  
	  add_proc_mem_affinity(procs_by_kind[k],
				mems_by_kind[Memory::HDF_MEM],
				5,   // "low" bandwidth
				100, // "high" latency
				adata, apos);

	  add_proc_mem_affinity(procs_by_kind[k],
				mems_by_kind[Memory::GLOBAL_MEM],
				10,  // "lower" bandwidth
				50,  // "higher" latency
				adata, apos);
	}

	add_mem_mem_affinity(mems_by_kind[Memory::SYSTEM_MEM],
			     mems_by_kind[Memory::GLOBAL_MEM],
			     30,  // "lower" bandwidth
			     25,  // "higher" latency
			     adata, apos);

	add_mem_mem_affinity(mems_by_kind[Memory::SYSTEM_MEM],
			     mems_by_kind[Memory::DISK_MEM],
			     15,  // "low" bandwidth
			     50,  // "high" latency
			     adata, apos);

#ifdef USE_CUDA
	// TODO: actually get gpu<->fb affinity right for multiple gpus
	add_proc_mem_affinity(procs_by_kind[Processor::TOC_PROC],
			      mems_by_kind[Memory::FB_MEM],
			      200, // "big" bandwidth
			      5,   // "ok" latency
			      adata, apos);

	add_proc_mem_affinity(procs_by_kind[Processor::TOC_PROC],
			      mems_by_kind[Memory::Z_COPY_MEM],
			      20,  // "medium" bandwidth
			      200, // "bad" latency
			      adata, apos);

	for(std::set<Processor::Kind>::const_iterator it = local_cpu_kinds.begin();
	    it != local_cpu_kinds.end();
	    it++) {
	  Processor::Kind k = *it;

	  add_proc_mem_affinity(procs_by_kind[k],
				mems_by_kind[Memory::Z_COPY_MEM],
				40,  // "large" bandwidth
				3,   // "small" latency
				adata, apos);
	}
#endif

	adata[apos++] = NODE_ANNOUNCE_DONE;
	assert(apos < ADATA_SIZE);

	// parse our own data (but don't create remote proc/mem objects)
	machine->parse_node_announce_data(gasnet_mynode(),
					  num_procs,
					  num_memories,
					  adata, apos*sizeof(adata[0]), 
					  false);

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

    struct MachineRunArgs {
      RuntimeImpl *r;
      Processor::TaskFuncID task_id;
      Runtime::RunStyle style;
      const void *args;
      size_t arglen;
    };  

    static bool running_as_background_thread = false;

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

    static void *background_run_thread(void *data)
    {
      MachineRunArgs *args = (MachineRunArgs *)data;
      running_as_background_thread = true;
      args->r->run(args->task_id, args->style, args->args, args->arglen,
		   false /* foreground from this thread's perspective */);
      delete args;
      return 0;
    }

    void RuntimeImpl::run(Processor::TaskFuncID task_id /*= 0*/,
			  Runtime::RunStyle style /*= ONE_TASK_ONLY*/,
			  const void *args /*= 0*/, size_t arglen /*= 0*/,
			  bool background /*= false*/)
    { 
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

      const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;

      // now that we've got the machine description all set up, we can start
      //  the worker threads for local processors, which'll probably ask the
      //  high-level runtime to set itself up
      if(task_table.count(Processor::TASK_ID_PROCESSOR_INIT) > 0) {
	log_task.info("spawning processor init task on local cpus");

	spawn_on_all(local_procs, Processor::TASK_ID_PROCESSOR_INIT, 0, 0,
		     Event::NO_EVENT,
		     INT_MAX); // runs with max priority

#ifdef OLD_INIT
	spawn_on_all(local_util_procs,Processor::TASK_ID_PROCESSOR_INIT, 0, 0,
		     Event::NO_EVENT,
		     INT_MAX); // runs with max priority

	spawn_on_all(local_cpus,Processor::TASK_ID_PROCESSOR_INIT, 0, 0,
		     Event::NO_EVENT,
		     INT_MAX); // runs with max priority

	spawn_on_all(local_io_procs,Processor::TASK_ID_PROCESSOR_INIT, 0, 0,
		     Event::NO_EVENT,
		     INT_MAX); // runs with max priority

#ifdef USE_CUDA
	spawn_on_all(local_gpus, Processor::TASK_ID_PROCESSOR_INIT, 0, 0,
		     Event::NO_EVENT,
		     INT_MAX); // runs with max priority
#endif
#endif
      } else {
	log_task.info("no processor init task");
      }

      if(task_id != 0 && 
	 ((style != Runtime::ONE_TASK_ONLY) || 
	  (gasnet_mynode() == 0))) {//(gasnet_nodes()-1)))) {
	for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  (*it)->me.spawn(task_id, args, arglen, ProfilingRequestSet(),
			  Event::NO_EVENT, 0/*priority*/);
	  if(style != Runtime::ONE_TASK_PER_PROC) break;
	}
      }

#ifdef TRACE_RESOURCES
      RuntimeImpl *rt = get_runtime();
#endif
#ifdef OLD_WAIT_LOOP
      // wait for idle-ness somehow?
      int timeout = -1;
      while(running_proc_count.get() > 0) {
	if(timeout >= 0) {
	  timeout--;
	  if(timeout == 0) {
	    printf("TIMEOUT!\n");
	    exit(1);
	  }
	}
	fflush(stdout);
	sleep(1);

#ifdef TRACE_RESOURCES
        log_runtime.info("total events: %d", rt->local_event_free_list->next_alloc);
        log_runtime.info("total reservations: %d", rt->local_reservation_free_list->next_alloc);
        log_runtime.info("total index spaces: %d", rt->local_index_space_free_list->next_alloc);
        log_runtime.info("total proc groups: %d", rt->local_proc_group_free_list->next_alloc);
#endif
      }
      log_runtime.info("running proc count is now zero - terminating\n");
#endif
      // sleep until shutdown has been requested by somebody
      {
	AutoHSLLock al(shutdown_mutex);
	while(!shutdown_requested)
	  shutdown_condvar.wait();
	log_runtime.info("shutdown request received - terminating\n");
      }

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

      // Shutdown all the threads
      for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
	(*it)->shutdown();

#ifdef OLD_INIT
      for(std::vector<LocalTaskProcessor *>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++)
	(*it)->shutdown();

      for(std::vector<LocalTaskProcessor *>::iterator it = local_io_procs.begin();
          it != local_io_procs.end();
          it++)
	(*it)->shutdown();

      for(std::vector<LocalTaskProcessor *>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++)
	(*it)->shutdown();

#ifdef USE_CUDA
      for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	  it != local_gpus.end();
	  it++)
	(*it)->shutdown();
#endif
#endif

      // delete processors, memories, nodes, etc.
      {
	for(gasnet_node_t i = 0; i < gasnet_nodes(); i++) {
	  Node& n = nodes[i];

	  for(std::vector<MemoryImpl *>::iterator it = n.memories.begin();
	      it != n.memories.end();
	      it++)
	    delete (*it);

	  for(std::vector<ProcessorImpl *>::iterator it = n.processors.begin();
	      it != n.processors.end();
	      it++)
	    delete (*it);
	}
	
	delete[] nodes;
	delete global_memory;
	delete local_event_free_list;
	delete local_barrier_free_list;
	delete local_reservation_free_list;
	delete local_index_space_free_list;
	delete local_proc_group_free_list;

	for(std::vector<Module *>::iterator it = modules.begin();
	    it != modules.end();
	    it++) {
	  (*it)->cleanup();
	  delete (*it);
	}

	module_registrar.unload_module_sofiles();
      }

      // need to kill other threads too so we can actually terminate process
      // Exit out of the thread
      LegionRuntime::LowLevel::stop_dma_worker_threads();
      stop_activemsg_threads();

      // if we are running as a background thread, just terminate this thread
      // if not, do a full process exit - gasnet may have started some threads we don't have handles for,
      //   and if they're left running, the app will hang
      if(running_as_background_thread) {
	pthread_exit(0);
      } else {
	// not strictly necessary, but helps us find memory leaks
	runtime_singleton = 0;
	delete this;
	exit(0);
      }
    }

    void RuntimeImpl::shutdown(bool local_request /*= true*/)
    {
      if(local_request) {
	log_runtime.info("shutdown request - notifying other nodes\n");
	for(unsigned i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    RuntimeShutdownMessage::send_request(i);
      }

      log_runtime.info("shutdown request - cleaning up local processors\n");

      if(task_table.count(Processor::TASK_ID_PROCESSOR_SHUTDOWN) > 0) {
	log_task.info("spawning processor shutdown task on local cpus");

	const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;

	spawn_on_all(local_procs, Processor::TASK_ID_PROCESSOR_SHUTDOWN, 0, 0,
		     Event::NO_EVENT,
		     INT_MIN); // runs with lowest priority
#ifdef OLD_INIT
	spawn_on_all(local_cpus, Processor::TASK_ID_PROCESSOR_SHUTDOWN, 0, 0,
		     Event::NO_EVENT,
		     INT_MIN); // runs with lowest priority
	spawn_on_all(local_util_procs, Processor::TASK_ID_PROCESSOR_SHUTDOWN, 0, 0,
		     Event::NO_EVENT,
		     INT_MIN); // runs with lowest priority
	spawn_on_all(local_io_procs, Processor::TASK_ID_PROCESSOR_SHUTDOWN, 0, 0,
		     Event::NO_EVENT,
		     INT_MIN); // runs with lowest priority
#ifdef USE_CUDA
	spawn_on_all(local_gpus, Processor::TASK_ID_PROCESSOR_SHUTDOWN, 0, 0,
		     Event::NO_EVENT,
		     INT_MIN); // runs with lowest priority
#endif
#endif
      } else {
	log_task.info("no processor shutdown task");
      }

      {
	AutoHSLLock al(shutdown_mutex);
	shutdown_requested = true;
	shutdown_condvar.broadcast();
      }
    }

    void RuntimeImpl::wait_for_shutdown(void)
    {
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

      // this terminates the process, so control never gets back to caller
      // would be nice to fix this...
      if (exit_process)
        gasnet_exit(0);
    }

    EventImpl *RuntimeImpl::get_event_impl(Event e)
    {
      ID id(e);
      switch(id.type()) {
      case ID::ID_EVENT:
	return get_genevent_impl(e);
      case ID::ID_BARRIER:
	return get_barrier_impl(e);
      default:
	assert(0);
      }
    }

    GenEventImpl *RuntimeImpl::get_genevent_impl(Event e)
    {
      ID id(e);
      assert(id.type() == ID::ID_EVENT);

      Node *n = &nodes[id.node()];
      GenEventImpl *impl = n->events.lookup_entry(id.index(), id.node());
      assert(impl->me == id);

      // check to see if this is for a generation more than one ahead of what we
      //  know of - this should only happen for remote events, but if it does it means
      //  there are some generations we don't know about yet, so we can catch up (and
      //  notify any local waiters right away)
      impl->check_for_catchup(e.gen - 1);

      return impl;
    }

    BarrierImpl *RuntimeImpl::get_barrier_impl(Event e)
    {
      ID id(e);
      assert(id.type() == ID::ID_BARRIER);

      Node *n = &nodes[id.node()];
      BarrierImpl *impl = n->barriers.lookup_entry(id.index(), id.node());
      assert(impl->me == id);
      return impl;
    }

    ReservationImpl *RuntimeImpl::get_lock_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_LOCK:
	{
	  Node *n = &nodes[id.node()];
	  ReservationImpl *impl = n->reservations.lookup_entry(id.index(), id.node());
	  assert(impl->me == id.convert<Reservation>());
	  return impl;
#if 0
	  std::vector<ReservationImpl>& locks = nodes[id.node()].locks;

	  unsigned index = id.index();
	  if(index >= n->num_locks) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->locks.size();
	    assert(oldsize < MAX_LOCAL_LOCKS);
	    if(index >= oldsize) { // only it's still too small
              assert((index+1) < MAX_LOCAL_LOCKS);
	      n->locks.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->locks[i].init(ID(ID::ID_LOCK, id.node(), i).convert<Reservation>(),
				 id.node());
	      n->num_locks = index + 1;
	    }
	  }
	  return &(locks[index]);
#endif
	}

      case ID::ID_INDEXSPACE:
	return &(get_index_space_impl(id)->lock);

      case ID::ID_INSTANCE:
	return &(get_instance_impl(id)->lock);

      case ID::ID_PROCGROUP:
	return &(get_procgroup_impl(id)->lock);

      default:
	assert(0);
      }
    }

    template <class T>
    inline T *null_check(T *ptr)
    {
      assert(ptr != 0);
      return ptr;
    }

    MemoryImpl *RuntimeImpl::get_memory_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_MEMORY:
      case ID::ID_ALLOCATOR:
      case ID::ID_INSTANCE:
	if(id.index_h() == ID::ID_GLOBAL_MEM)
	  return global_memory;
	return null_check(nodes[id.node()].memories[id.index_h()]);

      default:
	assert(0);
      }
    }

    ProcessorImpl *RuntimeImpl::get_processor_impl(ID id)
    {
      if(id.type() == ID::ID_PROCGROUP)
	return get_procgroup_impl(id);

      assert(id.type() == ID::ID_PROCESSOR);
      return null_check(nodes[id.node()].processors[id.index()]);
    }

    ProcessorGroup *RuntimeImpl::get_procgroup_impl(ID id)
    {
      assert(id.type() == ID::ID_PROCGROUP);

      Node *n = &nodes[id.node()];
      ProcessorGroup *impl = n->proc_groups.lookup_entry(id.index(), id.node());
      assert(impl->me == id.convert<Processor>());
      return impl;
    }

    IndexSpaceImpl *RuntimeImpl::get_index_space_impl(ID id)
    {
      assert(id.type() == ID::ID_INDEXSPACE);

      Node *n = &nodes[id.node()];
      IndexSpaceImpl *impl = n->index_spaces.lookup_entry(id.index(), id.node());
      assert(impl->me == id.convert<IndexSpace>());
      return impl;
#if 0
      unsigned index = id.index();
      if(index >= n->index_spaces.size()) {
	AutoHSLLock a(n->mutex); // take lock before we actually resize

	if(index >= n->index_spaces.size())
	  n->index_spaces.resize(index + 1);
      }

      if(!n->index_spaces[index]) { // haven't seen this metadata before?
	//printf("UNKNOWN METADATA " IDFMT "\n", id.id());
	AutoHSLLock a(n->mutex); // take lock before we actually allocate
	if(!n->index_spaces[index]) {
	  n->index_spaces[index] = new IndexSpaceImpl(id.convert<IndexSpace>());
	} 
      }

      return n->index_spaces[index];
#endif
    }

    RegionInstanceImpl *RuntimeImpl::get_instance_impl(ID id)
    {
      assert(id.type() == ID::ID_INSTANCE);
      MemoryImpl *mem = get_memory_impl(id);
      
      AutoHSLLock al(mem->mutex);

      if(id.index_l() >= mem->instances.size()) {
	assert(id.node() != gasnet_mynode());

	size_t old_size = mem->instances.size();
	if(id.index_l() >= old_size) {
	  // still need to grow (i.e. didn't lose the race)
	  mem->instances.resize(id.index_l() + 1);

	  // don't have region/offset info - will have to pull that when
	  //  needed
	  for(unsigned i = old_size; i <= id.index_l(); i++) 
	    mem->instances[i] = 0;
	}
      }

      if(!mem->instances[id.index_l()]) {
	if(!mem->instances[id.index_l()]) {
	  //printf("[%d] creating proxy instance: inst=" IDFMT "\n", gasnet_mynode(), id.id());
	  mem->instances[id.index_l()] = new RegionInstanceImpl(id.convert<RegionInstance>(), mem->me);
	}
      }
	  
      return mem->instances[id.index_l()];
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
