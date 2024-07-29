/* Copyright 2024 Stanford University, NVIDIA Corporation
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
#include "realm/transfer/ib_memory.h"

#include "realm/activemsg.h"
#include "realm/deppart/preimage.h"

#include "realm/cmdline.h"

#include "realm/codedesc.h"

#include "realm/utils.h"

// remote copy active messages from from lowlevel_dma.h for now
#include "realm/transfer/lowlevel_dma.h"

// create xd message and update bytes read/write messages
#include "realm/transfer/channel.h"
#include "realm/transfer/channel_disk.h"

#ifdef REALM_USE_KOKKOS
#include "realm/kokkos_interop.h"
#endif

#ifdef REALM_USE_NVTX
#include "realm/nvtx.h"
#endif

#include <string.h>
#include <thread>
#include <sstream>
#include <fstream>

#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#endif

// for cpu resource discovery
#if defined(REALM_ON_LINUX) || defined(REALM_ON_FREEBSD)
#include <sys/sysinfo.h>
#endif

#ifdef REALM_ON_MACOS
#include <sys/sysctl.h>
#endif

#ifdef REALM_ON_WINDOWS
#include <winsock2.h>
#include <windows.h>
#include <processthreadsapi.h>
#include <synchapi.h>
#include <sysinfoapi.h>

#pragma comment(lib, "ws2_32.lib")

static void sleep(int seconds)
{
  Sleep(seconds * 1000);
}

static char *strndup(const char *src, size_t maxlen)
{
  size_t actlen = strnlen(src, maxlen);
  char *dst = (char *)malloc(actlen + 1);
  strncpy(dst, src, actlen);
  return dst;
}
#endif

#include <fstream>

#define CHECK_LIBC(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "error: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

TYPE_IS_SERIALIZABLE(Realm::NodeAnnounceTag);
TYPE_IS_SERIALIZABLE(Realm::Memory);
TYPE_IS_SERIALIZABLE(Realm::Memory::Kind);
TYPE_IS_SERIALIZABLE(Realm::Channel::SupportedPath);
TYPE_IS_SERIALIZABLE(Realm::XferDesKind);
TYPE_IS_SERIALIZABLE(Realm::Machine::ProcessorMemoryAffinity);
TYPE_IS_SERIALIZABLE(Realm::Machine::ProcessInfo);

namespace Realm {

  Logger log_runtime("realm");
  Logger log_collective("collective");
  extern Logger log_task; // defined in proc_impl.cc
  extern Logger log_taskreg; // defined in proc_impl.cc
  
  ////////////////////////////////////////////////////////////////////////
  //
  // hacks to force linkage of things
  //

  extern int force_utils_cc_linkage;

  int *linkage_forcing[] = { &force_utils_cc_linkage };


  ////////////////////////////////////////////////////////////////////////
  //
  // signal handlers
  //

  namespace ThreadLocal {
    static REALM_THREAD_LOCAL int error_signal_value = 0;
  };

  static void register_error_signal_handler(void (*handler)(int))
  {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
    // register our handler for the standard error signals - set SA_ONSTACK
    //  so that any thread with an alt stack uses it
    struct sigaction action;
    action.sa_handler = handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_ONSTACK;

    CHECK_LIBC( sigaction(SIGINT, &action, 0) );
    CHECK_LIBC( sigaction(SIGABRT, &action, 0) );
    CHECK_LIBC( sigaction(SIGSEGV, &action, 0) );
    CHECK_LIBC( sigaction(SIGFPE, &action, 0) );
    CHECK_LIBC( sigaction(SIGBUS, &action, 0) );
    CHECK_LIBC( sigaction(SIGILL, &action, 0) );
#endif
  }

  static void unregister_error_signal_handler(void)
  {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
    // set standard error signals back to default handler
    struct sigaction action;
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;

    CHECK_LIBC( sigaction(SIGINT, &action, 0) );
    CHECK_LIBC( sigaction(SIGABRT, &action, 0) );
    CHECK_LIBC( sigaction(SIGSEGV, &action, 0) );
    CHECK_LIBC( sigaction(SIGFPE, &action, 0) );
    CHECK_LIBC( sigaction(SIGBUS, &action, 0) );
    CHECK_LIBC( sigaction(SIGILL, &action, 0) );
#endif
  }

    static void realm_freeze(int signal)
    {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
      assert((signal == SIGINT) || (signal == SIGABRT) ||
             (signal == SIGSEGV) || (signal == SIGFPE) ||
             (signal == SIGBUS) || (signal == SIGILL));
      int process_id = getpid();
      char hostname[128];
      gethostname(hostname, 127);
      fprintf(stderr,"Legion process received signal %d: %s\n",
                      signal, strsignal(signal));
      fprintf(stderr,"Process %d on node %s is frozen!\n", 
                      process_id, hostname);
      fflush(stderr);

      // now that we've stopped, don't catch any further SIGINTs
      struct sigaction action;
      action.sa_handler = SIG_DFL;
      sigemptyset(&action.sa_mask);
      action.sa_flags = 0;

      CHECK_LIBC( sigaction(SIGINT, &action, 0) );
#endif

      while (true)
        sleep(1);
    }

  template <typename TABLE>
  void show_event_table(std::ostream& os, NodeID nodeid, TABLE& events)
  {
    // Iterate over all the events and get their implementations
    for (unsigned long j = 0; j < events.max_entries(); j++) {
      if (!events.has_entry(j))
	continue;
      GenEventImpl *e = events.lookup_entry(j, nodeid);
      AutoLock<> a2(e->mutex);
	
      // print anything with either local or remote waiters
      if(e->current_local_waiters.empty() &&
	 e->future_local_waiters.empty() &&
	 e->remote_waiters.empty())
	continue;

      size_t clw_size = 0;
      for(EventWaiter *pos = e->current_local_waiters.head.next;
	  pos;
	  pos = pos->ew_list_link.next)
	clw_size++;
      EventImpl::gen_t gen = e->generation.load();
      os << "Event " << e->me <<": gen=" << gen
	 << " subscr=" << e->gen_subscribed.load()
	 << " local=" << clw_size //e->current_local_waiters.size()
	 << "+" << e->future_local_waiters.size()
	 << " remote=" << e->remote_waiters.size() << "\n";
      for(EventWaiter *pos = e->current_local_waiters.head.next;
	  pos;
	  pos = pos->ew_list_link.next) {
	os << "  [" << (gen+1) << "] L:" << pos/*(*it)*/ << " - ";
	pos/*(*it)*/->print(os);
	os << "\n";
      }
      for(std::map<EventImpl::gen_t, EventWaiter::EventWaiterList>::const_iterator it = e->future_local_waiters.begin();
	  it != e->future_local_waiters.end();
	  it++) {
	for(EventWaiter *pos = it->second.head.next;
	    pos;
	    pos = pos->ew_list_link.next) {
	  os << "  [" << (it->first) << "] L:" << pos/*(*it2)*/ << " - ";
	  pos/*(*it2)*/->print(os);
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
  }

  // not static so that it can be invoked manually from gdb
  void show_event_waiters(std::ostream& os)
  {
    os << "PRINTING ALL PENDING EVENTS:\n";
    for(NodeID i = 0; i <= Network::max_node_id; i++) {
      Node *n = &get_runtime()->nodes[i];

      if(i == Network::my_node_id)
	show_event_table(os, i, get_runtime()->local_events);
      else
	show_event_table(os, i, n->remote_events);

      for (unsigned long j = 0; j < n->barriers.max_entries(); j++) {
	if (!n->barriers.has_entry(j))
	  continue;
	BarrierImpl *b = n->barriers.lookup_entry(j, i/*node*/); 
	AutoLock<> a2(b->mutex);
	// skip any barriers with no waiters
	if (b->generations.empty())
	  continue;

	os << "Barrier " << b->me << ": gen=" << b->generation.load()
	   << " subscr=" << b->gen_subscribed.load() << "\n";
	for (std::map<EventImpl::gen_t, BarrierImpl::Generation*>::const_iterator git = 
	       b->generations.begin(); git != b->generations.end(); git++) {
	  const EventWaiter::EventWaiterList &waiters = git->second->local_waiters;
	  for(EventWaiter *pos = waiters.head.next;
	      pos;
	      pos = pos->ew_list_link.next) {
	    os << "  [" << (git->first) << "] L:" << pos/*(*it)*/ << " - ";
	    pos/*(*it)*/->print(os);
	    os << "\n";
	  }
	}
      }
    }

    // TODO - pending barriers
#if 0
    // // convert from events to barriers
    // fprintf(f,"PRINTING ALL PENDING EVENTS:\n");
    // for(int i = 0; i <= Network::max_node_id; i++) {
    // 	Node *n = &get_runtime()->nodes[i];
    //   // Iterate over all the events and get their implementations
    //   for (unsigned long j = 0; j < n->events.max_entries(); j++) {
    //     if (!n->events.has_entry(j))
    //       continue;
    // 	  EventImpl *e = n->events.lookup_entry(j, i/*node*/);
    // 	  AutoLock<> a2(e->mutex);
    
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
  // struct ReductionOpUntyped
  //

  /*static*/ ReductionOpUntyped *ReductionOpUntyped::clone_reduction_op(const ReductionOpUntyped *redop)
  {
    void *ptr = malloc(redop->sizeof_this);
    assert(ptr);
    memcpy(ptr, redop, redop->sizeof_this);
    ReductionOpUntyped *cloned = static_cast<ReductionOpUntyped *>(ptr);
    // fix up identity and userdata fields, if non-null
    if(redop->identity)
      cloned->identity = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(redop->identity) -
                                                  reinterpret_cast<uintptr_t>(redop) +
                                                  reinterpret_cast<uintptr_t>(cloned));
    if(redop->userdata)
      cloned->userdata = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(redop->userdata) -
                                                  reinterpret_cast<uintptr_t>(redop) +
                                                  reinterpret_cast<uintptr_t>(cloned));
    return cloned;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Runtime
  //

    Runtime::Runtime(void)
      : impl(0)
    {
      // ok to construct extra ones - we will make sure only one calls init() though
      if (runtime_singleton) {
        impl = runtime_singleton;
      } else {
        impl = new RuntimeImpl;
        runtime_singleton = static_cast<RuntimeImpl *>(impl);
      }
    }

    /*static*/ Runtime Runtime::get_runtime(void)
    {
      Runtime r;
      // explicit namespace qualifier here due to name collision
      r.impl = Realm::get_runtime();
      return r;
    }

    // returns a valid (but possibly empty) string pointer describing the
    //  version of the Realm library - this can be compared against
    //  REALM_VERSION in application code to detect a header/library mismatch
    const char *realm_library_version = REALM_VERSION;

    /*static*/ const char *Runtime::get_library_version()
    {
      return realm_library_version;
    }

#if defined(REALM_USE_UCX) || defined(REALM_USE_MPI) || defined(REALM_USE_GASNET1) || defined(REALM_USE_GASNETEX) || defined(REALM_USE_KOKKOS)
    // global flag that tells us if a realm runtime has already been
    //  initialized in this process - some underlying libraries (e.g. mpi,
    //  gasnet, kokkos) do not permit reinitialization
    static bool runtime_initialized = false;
#endif

    // performs any network initialization and, critically, makes sure
    //  *argc and *argv contain the application's real command line
    //  (instead of e.g. mpi spawner information)
    bool Runtime::network_init(int *argc, char ***argv)
    {
#if defined(REALM_USE_UCX) || defined(REALM_USE_MPI) || defined(REALM_USE_GASNET1) || defined(REALM_USE_GASNETEX) || defined(REALM_USE_KOKKOS)
      if(runtime_initialized) {
        fprintf(stderr, "ERROR: reinitialization not supported by these Realm components:"
#ifdef REALM_USE_UCX
                " ucx"
#endif
#ifdef REALM_USE_MPI
                " mpi"
#endif
#ifdef REALM_USE_GASNET1
                " gasnet1"
#endif
#ifdef REALM_USE_GASNETEX
                " gasnetex"
#endif
#ifdef REALM_USE_KOKKOS
                " kokkos"
#endif
                "\n");
        return false;
      }
      runtime_initialized = true;
#endif

      assert(runtime_singleton != 0);
      return static_cast<RuntimeImpl *>(impl)->network_init(argc, argv);
    }

        void Runtime::parse_command_line(int argc, char **argv)
    {
      assert(impl != 0);
      std::vector<std::string> cmdline;
      cmdline.reserve(argc);
      for(int i = 1; i < argc; i++)
        cmdline.push_back(argv[i]);
      static_cast<RuntimeImpl *>(impl)->parse_command_line(cmdline);
    }

    void Runtime::parse_command_line(std::vector<std::string> &cmdline,
                                                bool remove_realm_args /*= false*/)
    {
      assert(impl != 0);
      if(remove_realm_args) {
        static_cast<RuntimeImpl *>(impl)->parse_command_line(cmdline);
      } else {
        // pass in a copy so we don't mess up the original
        std::vector<std::string> cmdline_copy(cmdline);
        static_cast<RuntimeImpl *>(impl)->parse_command_line(cmdline_copy);
      }
    }

    void Runtime::finish_configure(void)
    {
      assert(impl != 0);
      static_cast<RuntimeImpl *>(impl)->finish_configure();
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
      // if we get null pointers for argc and argv, use a local version so
      //  any changes from network_init are seen in configure_from_command_line
      int my_argc = 0;
      char **my_argv = 0;
      if(!argc) argc = &my_argc;
      if(!argv) argv = &my_argv;

      if(!network_init(argc, argv)) return false;
      if(!create_configs(*argc, *argv))
        return false;
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
      std::vector<Event> events;
      std::vector<ProcessorImpl *>& procs = ((RuntimeImpl *)impl)->nodes[Network::my_node_id].processors;
      for (std::vector<ProcessorImpl *>::iterator it = procs.begin();
           it != procs.end(); it++) {
        Event e = (*it)->me.register_task(taskid, codedesc, prs);
        events.push_back(e);
      }

      Event merged = Event::merge_events(events);
      log_taskreg.info() << "waiting on event: " << merged;
      merged.external_wait();
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

      ReductionOpUntyped *cloned = ReductionOpUntyped::clone_reduction_op(redop);
      bool conflict = ((RuntimeImpl *)impl)->reduce_op_table.put(redop_id, cloned);
      if(conflict) {
	log_runtime.error() << "duplicate registration of reduction op " << redop_id;
	free(cloned);
	return false;
      }

      return true;
    }

    bool Runtime::register_custom_serdez(CustomSerdezID serdez_id, const CustomSerdezUntyped *serdez)
    {
      assert(impl != 0);

      CustomSerdezUntyped *cloned = serdez->clone();
      bool conflict = ((RuntimeImpl *)impl)->custom_serdez_table.put(serdez_id, cloned);
      if(conflict) {
	log_runtime.error() << "duplicate registration of custom serdez " << serdez_id;
	delete cloned;
	return false;
      }

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
  
    void RuntimeImpl::DeferredShutdown::defer(RuntimeImpl *_runtime,
					      Event wait_on)
    {
      runtime = _runtime;
      EventImpl::add_waiter(wait_on, this);
    }

    void RuntimeImpl::DeferredShutdown::event_triggered(bool poisoned,
							TimeLimit work_until)
    {
      // no real good way to deal with a poisoned shutdown precondition
      if(poisoned) {
	log_poison.fatal() << "HELP!  poisoned precondition for runtime shutdown";
	assert(false);
      }
      log_runtime.info() << "triggering deferred shutdown";
      runtime->initiate_shutdown();
    }

    void RuntimeImpl::DeferredShutdown::print(std::ostream& os) const
    {
      os << "deferred shutdown";
    }

    Event RuntimeImpl::DeferredShutdown::get_finish_event(void) const
    {
      return Event::NO_EVENT;
    }

    void Runtime::shutdown(Event wait_on /*= Event::NO_EVENT*/,
			   int result_code /*= 0*/)
    {
      // if we're called from inside a task, automatically include the
      //  task's finish event as well
      if(Thread::self()) {
	Operation *op = Thread::self()->get_operation();
	if(op != 0) {
	  log_runtime.debug() << "shutdown merging finish event=" << op->get_finish_event();
	  wait_on = Event::merge_events(wait_on, op->get_finish_event());
	}
      }

      log_runtime.info() << "shutdown requested - wait_on=" << wait_on
			 << " code=" << result_code;

      // send a message to the shutdown master if it's not us
      NodeID shutdown_master_node = 0;
      if(Network::my_node_id != shutdown_master_node) {
	ActiveMessage<RuntimeShutdownRequest> amsg(shutdown_master_node);
	amsg->wait_on = wait_on;
	amsg->result_code = result_code;
	amsg.commit();
	return;
      }

      RuntimeImpl *r_impl = static_cast<RuntimeImpl *>(impl);
      bool duplicate = r_impl->request_shutdown(wait_on, result_code);
      if(!duplicate) {
	if(wait_on.has_triggered())
	  r_impl->initiate_shutdown();
	else
	  r_impl->deferred_shutdown.defer(r_impl, wait_on);
      }
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

    bool Runtime::create_configs(int argc, char **argv)
    {
      return ((RuntimeImpl *)impl)->create_configs(argc, argv);
    }

    ModuleConfig* Runtime::get_module_config(const std::string name)
    {
      return ((RuntimeImpl *)impl)->get_module_config(name);
    }

    Module *Runtime::get_module_untyped(const char *name)
    {
      if(runtime_singleton) {
	return runtime_singleton->get_module_untyped(name);
      } else {
	// modules don't exist if we're not initialized yet
	return 0;
      }
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreModule
  //

  namespace Config {
    // if true, worker threads that might have used user-level thread switching
    //  fall back to kernel threading
    bool force_kernel_threads = false;
    unsigned long long job_id = 0;
  };

  CoreModuleConfig::CoreModuleConfig(void)
    : ModuleConfig("core")
  {
    config_map.insert({"cpu", &num_cpu_procs});
    config_map.insert({"util", &num_util_procs});
    config_map.insert({"io", &num_io_procs});
    config_map.insert({"sysmem", &sysmem_size});
    config_map.insert({"stack_size", &stack_size});
    config_map.insert({"pin_util_procs", &pin_util_procs});
    config_map.insert({"use_ext_sysmem", &use_ext_sysmem});
    config_map.insert({"regmem", &reg_mem_size});
    config_map.insert({"enable_sparsity_refcount", &enable_sparsity_refcount});

    resource_map.insert({"cpu", &res_num_cpus});
    resource_map.insert({"sysmem", &res_sysmem_size});
  }

#ifdef REALM_ON_WINDOWS
static DWORD CountSetBits(ULONG_PTR bitMask)
{
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    DWORD bitSetCount = 0;
    ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;    
    DWORD i;
    
    for (i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest)?1:0);
        bitTest/=2;
    }

    return bitSetCount;
}
#endif

  bool CoreModuleConfig::discover_resource(void)
  {
#ifdef REALM_ON_LINUX
    // system memory
    struct sysinfo memInfo;
    sysinfo (&memInfo);
    res_sysmem_size = memInfo.totalram;
    // phyical cores
    std::ifstream infile("/proc/cpuinfo");
    if (infile.fail()) return false;
    std::string line;
    int cpu_id = 0;
    int cpu_cores = 0;
    std::map<int, int> cpus;
    while (std::getline(infile, line)) {
      if (strncmp(line.c_str(), "physical id", 11) == 0) {
        std::istringstream iss(line);
        std::string x, y, z;
        if (!(iss >> x >> y >> z >> cpu_id)) {
          infile.close();
          return false;
        };
      }
      if (strncmp(line.c_str(), "cpu cores", 9) == 0) {
        std::istringstream iss(line);
        std::string x, y, z;
        if (!(iss >> x >> y >> z >> cpu_cores)) {
          infile.close();
          return false;
        }
        std::map<int, int>::iterator it = cpus.find(cpu_id);
        if (it == cpus.end()) {
          cpus.insert({cpu_id, cpu_cores});
        } else {
          assert(it->second == cpu_cores);
        }
      }
    }
    for (std::map<int, int>::iterator it = cpus.begin(); it != cpus.end(); it++) {
      res_num_cpus += it->second;
    }
    infile.close();
#endif
#ifdef REALM_ON_WINDOWS
    // system memory
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    res_sysmem_size = static_cast<size_t>(memInfo.ullTotalPageFile);
    // physical cores
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = nullptr;
    DWORD buffer_size = 0;

    // Get the required buffer size
    if(!GetLogicalProcessorInformationEx(RelationAll, nullptr, &buffer_size)) {
      DWORD last_err = GetLastError();
      if(last_err != ERROR_INSUFFICIENT_BUFFER) {
        return false;
      }
      assert(buffer_size != 0);
    }

    buffer = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(new char[buffer_size]);
    if (!GetLogicalProcessorInformationEx(RelationAll, buffer, &buffer_size)) {
      delete[] buffer;
      return false;
    }
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX ptr = buffer;
    while (reinterpret_cast<LPBYTE>(ptr) < reinterpret_cast<LPBYTE>(buffer) + buffer_size) {
      if (ptr->Relationship == RelationProcessorCore) {
        DWORD logical_processor_count = 0;
        for (DWORD i = 0; i < ptr->Processor.GroupCount; i++) {
          logical_processor_count += CountSetBits(ptr->Processor.GroupMask[i].Mask);
        }
        if (logical_processor_count == 1) {
          res_num_cpus++;
        }
      }
      ptr = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<LPBYTE>(ptr) + ptr->Size);
    }
    delete[] buffer;
#endif
#ifdef REALM_ON_FREEBSD
    // system memory
    size_t buflen = sizeof(size_t);
    sysctlbyname("hw.physmem", &res_sysmem_size, &buflen, NULL, 0);
    // phyical cores
    buflen = sizeof(int);
    sysctlbyname("hw.ncpu", &res_num_cpus, &buflen, NULL, 0);
#endif
#ifdef REALM_ON_MACOS
    // system memory
    size_t buflen = sizeof(size_t);
    sysctlbyname("hw.memsize", &res_sysmem_size, &buflen, NULL, 0);
    // phyical cores
    buflen = sizeof(int);
    sysctlbyname("hw.physicalcpu", &res_num_cpus, &buflen, NULL, 0);
#endif
    log_runtime.info("Discover resource cpu cores %d, sysmem %zu", res_num_cpus, res_sysmem_size);
    resource_discover_finished = true;
    return resource_discover_finished;
  }

  void CoreModuleConfig::configure_from_cmdline(std::vector<std::string>& cmdline)
  {
    assert(finish_configured == false);
    // parse command line arguments
    CommandLineParser cp;

    // config for CoreModule
    cp.add_option_int("-ll:cpu", num_cpu_procs)
      .add_option_int("-ll:util", num_util_procs)
      .add_option_int("-ll:io", num_io_procs)
      .add_option_int("-ll:concurrent_io", concurrent_io_threads)
      .add_option_int_units("-ll:csize", sysmem_size, 'm')
      .add_option_int_units("-ll:stacksize", stack_size, 'm')
      .add_option_bool("-ll:pin_util", pin_util_procs)
      .add_option_int("-ll:cpu_bgwork", cpu_bgwork_timeslice)
      .add_option_int("-ll:util_bgwork", util_bgwork_timeslice)
      .add_option_int("-ll:ext_sysmem", use_ext_sysmem);

    // config for RuntimeImpl
    // low-level runtime parameters
    if(Network::max_node_id > 0)
      reg_ib_mem_size = 256 << 20; // for inter-node copies
    else
      reg_ib_mem_size = 64 << 20; // for local transposes/serdez


    // This dummy network list is actually handled in network_init()
    // this is just here to help verify low-level arguement
    std::vector<std::string> dummy_network_list;

    cp.add_option_int_units("-ll:rsize", reg_mem_size, 'm')
      .add_option_int_units("-ll:ib_rsize", reg_ib_mem_size, 'm')
      .add_option_int_units("-ll:dsize", disk_mem_size, 'm')
      .add_option_int("-ll:dma", dma_worker_threads)
      .add_option_bool("-ll:pin_dma", pin_dma_threads)
      .add_option_int("-ll:dummy_rsrv_ok", dummy_reservation_ok)
      .add_option_bool("-ll:show_rsrv", show_reservations)
      .add_option_int("-ll:ht_sharing", hyperthread_sharing)
      .add_option_int_units("-ll:bitset_chunk", bitset_chunk_size, 'k')
      .add_option_int("-ll:bitset_twolevel", bitset_twolevel);


    cp.add_option_string("-ll:eventtrace", event_trace_file)
      .add_option_string("-ll:locktrace", lock_trace_file);

#ifdef NODE_LOGGING
    cp.add_option_string("-ll:prefix", RuntimeImpl::prefix);
#else
    std::string dummy_prefix;
    cp.add_option_string("-ll:prefix", dummy_prefix);
#endif

    cp.add_option_int("-ll:ahandlers", active_msg_handler_threads);
    cp.add_option_int("-ll:handler_bgwork", active_msg_handler_bgwork);
    cp.add_option_stringlist("-ll:networks", dummy_network_list);
    cp.add_option_int_units("-ll:replheap", replheap_size);

    // The default of path_cache_size is 0, when it is set to non-zero, the caching is enabled.
    cp.add_option_int("-ll:path_cache_size", Config::path_cache_lru_size);

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
  }

  CoreModule::CoreModule(void)
    : Module("core")
    , config(nullptr)
  {}

  CoreModule::~CoreModule(void)
  {
    assert(config != nullptr);
    config = nullptr;
  }

  /*static*/ ModuleConfig *CoreModule::create_module_config(RuntimeImpl *runtime)
  {
    CoreModuleConfig *config = new CoreModuleConfig();
    config->discover_resource();
    return config;
  }

  /*static*/ Module *CoreModule::create_module(RuntimeImpl *runtime)
  {
    CoreModule *m = new CoreModule;

    CoreModuleConfig *config =
        checked_cast<CoreModuleConfig *>(runtime->get_module_config("core"));
    assert(config != nullptr);
    assert(config->finish_configured);
    assert(m->name == config->get_name());
    assert(m->config == nullptr);
    m->config = config;

    return m;
  }

  // create any memories provided by this module (default == do nothing)
  //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
  void CoreModule::create_memories(RuntimeImpl *runtime)
  {
    Module::create_memories(runtime);

    MemoryImpl *sysmem;
    if(config->sysmem_size > 0) {
      Memory m = runtime->next_local_memory_id();
      sysmem = new LocalCPUMemory(m, config->sysmem_size,
                                  -1/*don't care numa domain*/,
                                  Memory::SYSTEM_MEM);
      runtime->add_memory(sysmem);
    } else
      sysmem = 0;

    // create a memory that will hold external instances (the sysmem above
    //  might get registered with network and/or gpus, but external instances
    //  usually won't have those affinities)
    if(config->use_ext_sysmem || !sysmem) {
      Memory m = runtime->next_local_memory_id();
      ext_sysmem = new LocalCPUMemory(m, 0 /*size*/,
                                      -1 /*don't care numa domain*/,
                                      Memory::SYSTEM_MEM);
      runtime->add_memory(ext_sysmem);
    } else
      ext_sysmem = sysmem;
  }

  // create any processors provided by the module (default == do nothing)
  //  (each new ProcessorImpl should use a Processor from
  //   RuntimeImpl::next_local_processor_id)
  void CoreModule::create_processors(RuntimeImpl *runtime)
  {
    Module::create_processors(runtime);

    for(int i = 0; i < config->num_util_procs; i++) {
      Processor p = runtime->next_local_processor_id();
      ProcessorImpl *pi = new LocalUtilityProcessor(p, runtime->core_reservation_set(),
						    config->stack_size,
						    Config::force_kernel_threads,
                                                    config->pin_util_procs,
						    &runtime->bgwork,
						    config->util_bgwork_timeslice);
      runtime->add_processor(pi);
    }

    for(int i = 0; i < config->num_io_procs; i++) {
      Processor p = runtime->next_local_processor_id();
      ProcessorImpl *pi = new LocalIOProcessor(p, runtime->core_reservation_set(),
					       config->stack_size,
					       config->concurrent_io_threads);
      runtime->add_processor(pi);
    }

    for(int i = 0; i < config->num_cpu_procs; i++) {
      Processor p = runtime->next_local_processor_id();
      ProcessorImpl *pi = new LocalCPUProcessor(p, runtime->core_reservation_set(),
						config->stack_size,
						Config::force_kernel_threads,
						&runtime->bgwork,
						config->cpu_bgwork_timeslice);
      runtime->add_processor(pi);
    }
  }

  // create any DMA channels provided by the module (default == do nothing)
  void CoreModule::create_dma_channels(RuntimeImpl *runtime)
  {
    Module::create_dma_channels(runtime);

    // create the standard set of channels here
    runtime->add_dma_channel(new MemcpyChannel(&runtime->bgwork));
    runtime->add_dma_channel(new MemfillChannel(&runtime->bgwork));
    runtime->add_dma_channel(new MemreduceChannel(&runtime->bgwork));
    runtime->add_dma_channel(new RemoteWriteChannel(&runtime->bgwork));
    runtime->add_dma_channel(new AddressSplitChannel(&runtime->bgwork));
    runtime->add_dma_channel(new FileChannel(&runtime->bgwork));
    runtime->add_dma_channel(new DiskChannel(&runtime->bgwork));
    // "GASNet" means global memory here
    runtime->add_dma_channel(new GASNetChannel(&runtime->bgwork,
					       XFER_GASNET_READ));
    runtime->add_dma_channel(new GASNetChannel(&runtime->bgwork,
					       XFER_GASNET_WRITE));
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
  
    RuntimeImpl::RuntimeImpl(void)
      : machine(0), 
        num_untriggered_events(0),
	nodes(0),
	local_event_free_list(0), local_barrier_free_list(0),
	local_reservation_free_list(0),
	local_compqueue_free_list(0),
	//local_sparsity_map_free_list(0),
	run_method_called(false),
	shutdown_condvar(shutdown_mutex),
	shutdown_request_received(false),
	shutdown_result_code(0),
	shutdown_initiated(false),
	shutdown_in_progress(false),
	core_map(0), core_reservations(0),
	message_manager(0),
	sampling_profiler(true /*system default*/),
	num_local_memories(0), num_local_ib_memories(0),
	num_local_processors(0),
	module_registrar(this),
        modules_created(false),
	module_configs_created(false)
    {
      machine = new MachineImpl;
    }

    RuntimeImpl::~RuntimeImpl(void)
    {
      delete machine;
      delete core_reservations;
      delete core_map;

      delete_container_contents_free(reduce_op_table.map);
      delete_container_contents(custom_serdez_table.map);
    }

    Memory RuntimeImpl::next_local_memory_id(void)
    {
      Memory m = ID::make_memory(Network::my_node_id,
				 num_local_memories++).convert<Memory>();
      return m;
    }

    Memory RuntimeImpl::next_local_ib_memory_id(void)
    {
      Memory m = ID::make_ib_memory(Network::my_node_id,
                                    num_local_ib_memories++).convert<Memory>();
      return m;
    }

    Processor RuntimeImpl::next_local_processor_id(void)
    {
      Processor p = ID::make_processor(Network::my_node_id, 
				       num_local_processors++).convert<Processor>();
      return p;
    }

    void RuntimeImpl::add_memory(MemoryImpl *m)
    {
      // right now expect this to always be for the current node and the next memory ID
      ID id(m->me);
      assert(NodeID(id.memory_owner_node()) == Network::my_node_id);
      assert(id.memory_mem_idx() == nodes[Network::my_node_id].memories.size());

      nodes[Network::my_node_id].memories.push_back(m);
    }

    void RuntimeImpl::add_ib_memory(IBMemory *m)
    {
      // right now expect this to always be for the current node and the next memory ID
      ID id(m->me);
      assert(NodeID(id.memory_owner_node()) == Network::my_node_id);
      assert(id.memory_mem_idx() == nodes[Network::my_node_id].ib_memories.size());

      nodes[Network::my_node_id].ib_memories.push_back(m);
    }

    void RuntimeImpl::add_processor(ProcessorImpl *p)
    {
      // right now expect this to always be for the current node and the next processor ID
      ID id(p->me);
      assert(NodeID(id.proc_owner_node()) == Network::my_node_id);
      assert(id.proc_proc_idx() == nodes[Network::my_node_id].processors.size());

      nodes[Network::my_node_id].processors.push_back(p);
    }

    void RuntimeImpl::add_dma_channel(Channel *c)
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

    CoreReservationSet& RuntimeImpl::core_reservation_set(void)
    {
      assert(core_reservations);
      return *core_reservations;
    }

    const std::vector<CodeTranslator *>& RuntimeImpl::get_code_translators(void) const
    {
      return code_translators;
    }

    Module *RuntimeImpl::get_module_untyped(const char *name) const
    {
      if(!modules_created) {
        log_runtime.fatal() << "request for '" << name
                            << "' module before all modules have been created";
        abort();
      }

      // TODO: worth building a map here instead?
      for(std::vector<Module *>::const_iterator it = modules.begin();
          it != modules.end();
          ++it)
        if(!strcmp(name, (*it)->get_name().c_str()))
          return *it;

      for(std::vector<NetworkModule *>::const_iterator it = network_modules.begin();
          it != network_modules.end();
          ++it)
        if(!strcmp(name, (*it)->get_name().c_str()))
          return *it;

      return 0;
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

    bool RuntimeImpl::network_init(int *argc, char ***argv)
    {
      // if we're given empty or non-existent argc/argv, start from a
      //  dummy command line with a single string (which is supposed to be
      //  the name of the binary) so that the network module and/or the
      //  REALM_DEFAULT_ARGS code below can safely modify it - (we can only
      //  copy it back if the pointers are not null
      int local_argc;
      const char **local_argv;
      if(argc && *argc && argv) {
        local_argc = *argc;
        local_argv = *const_cast<const char ***>(argv);
      } else {
        static const char *dummy_cmdline_args[] = { "unknown-binary", 0 };

        local_argc = 1;
        local_argv = dummy_cmdline_args;
      }

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
	    int new_argc = local_argc + starts.size();
	    const char **new_argv = (const char **)(malloc((new_argc + 1) * sizeof(char *)));
	    // new args go after argv[0] and anything that looks like a
	    //  positional argument (i.e. doesn't start with -)
	    int before_new = 0;
	    while(before_new < local_argc) {
	      if((before_new > 0) && (local_argv[before_new][0] == '-'))
		break;
	      new_argv[before_new] = local_argv[before_new];
	      before_new++;
	    }
	    for(size_t i = 0; i < starts.size(); i++)
	      new_argv[i + before_new] = strndup(starts[i], ends[i] - starts[i]);
	    for(int i = before_new; i < local_argc; i++)
	      new_argv[i + starts.size()] = local_argv[i];
	    new_argv[new_argc] = 0;

	    local_argc = new_argc;
	    local_argv = new_argv;
	  }
	}
      }

      module_registrar.create_network_modules(network_modules, &local_argc, &local_argv);

      if(argc) *argc = local_argc;
      if(argv) *argv = const_cast<char **>(local_argv);

      return true;
    }

    template <typename T>
    static bool serialize_announce(T &serializer,
                                   const Machine::ProcessInfo *process_info,
                                   NetworkModule *net)
    {
      bool ok =
          ((serializer << NODE_ANNOUNCE_PROCESS_INFO) && (serializer << *process_info));
      return ok;
    }

    template <typename T>
    static bool serialize_announce(T &serializer, const ProcessorImpl *impl,
                                   NetworkModule *net)
    {
      Processor p = impl->me;
      Processor::Kind k = impl->me.kind();
      int num_cores = impl->num_cores;

      bool ok = ((serializer << NODE_ANNOUNCE_PROC) &&
                 (serializer << p) &&
                 (serializer << k) &&
                 (serializer << num_cores));
      return ok;
    }

    template <typename T>
    static bool serialize_announce(T &serializer, const MemoryImpl *impl,
                                   NetworkModule *net)
    {
      Memory m = impl->me;
      Memory::Kind k = impl->me.kind();
      size_t size = impl->size;
      const ByteArray *rdma_info = impl->get_rdma_info(net);

      bool ok = ((serializer << NODE_ANNOUNCE_MEM) &&
                 (serializer << m) &&
                 (serializer << k) &&
                 (serializer << size) &&
                 (serializer << (rdma_info != 0)));
      if(rdma_info != 0)
        ok = ok && (serializer << *rdma_info);
      return ok;
    }

    template <typename T>
    static bool serialize_announce(T &serializer, const IBMemory *ibmem,
                                   NetworkModule *net)
    {
      Memory m = ibmem->me;
      Memory::Kind k = ibmem->me.kind();
      size_t size = ibmem->size;
      const ByteArray *rdma_info = ibmem->get_rdma_info(net);

      bool ok = ((serializer << NODE_ANNOUNCE_IB_MEM) &&
                 (serializer << m) &&
                 (serializer << k) &&
                 (serializer << size) &&
                 (serializer << (rdma_info != 0)));
      if(rdma_info != 0)
        ok = ok && (serializer << *rdma_info);
      return ok;
    }

    template <typename T>
    static bool serialize_announce(T& serializer,
                                   const Machine::ProcessorMemoryAffinity& pma,
                                   NetworkModule *net)
    {
      bool ok = ((serializer << NODE_ANNOUNCE_PMA) && (serializer << pma));
      return ok;
    }

    template <typename T>
    static bool serialize_announce(T &serializer, const Channel *ch, NetworkModule *net)
    {
      RemoteChannelInfo *rci = ch->construct_remote_info();
      bool ok = ((serializer << NODE_ANNOUNCE_DMA_CHANNEL) &&
                 (serializer << *rci));
      delete rci;
      return ok;
    }

    template <typename T, typename Elem>
    static bool serialize_announce(T &serializer, const std::vector<Elem> &elements,
                                   NetworkModule *net)
    {
      // TODO: rather than have each element push a tag, we can instead push a tag and
      // size once here, compacting the announcement data
      bool ok = true;
      for(const Elem &element : elements) {
        ok = serialize_announce(serializer, element, net);
        if(!ok) {
          break;
        }
      }
      return ok;
    }

    template <typename T>
    static bool serialize_announce(T &serializer, const Node *node,
                                   const MachineImpl *machine_impl, NetworkModule *net)
    {
      bool ok = true;
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      ok = serialize_announce(serializer, node->processors, net);
      if(!ok) {
        return ok;
      }
      for(ProcessorImpl *proc : node->processors) {
        get_machine()->get_proc_mem_affinity(pmas, proc->me);
        ok = serialize_announce(serializer, pmas, net);
        if(!ok) {
          return ok;
        }
      }
      ok = serialize_announce(serializer, node->dma_channels, net);
      if(!ok) {
        return ok;
      }
      ok = serialize_announce(
          serializer, (machine_impl->nodeinfos.at(Network::my_node_id))->process_info,
          net);
      return ok;
    }

    /// Internal auxilary class to handle active messages for sharing CPU memory objects
    struct ShareableMemoryMessageHandler {
      struct Payload {
        realm_id_t memory_id; // The memory which is being shared
        size_t sz;            // The size of the memory
      };
      static Mutex mutex;
      static Mutex::CondVar cond_var;
      static size_t num_msgs_handled;
      static void handle_message(NodeID sender, const ShareableMemoryMessageHandler &args,
                                 const void *data, size_t len)
      {
        const Payload *msg = reinterpret_cast<const Payload *>(data);
        const Payload *end_msg = msg + len / sizeof(*msg);
        // Iterate all the payloads passed here.
        AutoLock<> al(ShareableMemoryMessageHandler::mutex);
        for (; msg != end_msg; ++msg) {
          assert(ID(msg->memory_id).is_memory() && "Parsed id is not a memory id");
          assert((sender == NodeID(ID(msg->memory_id).memory_owner_node())) &&
                 "Sender is not owner of sent memory");
          SharedMemoryInfo shm;
          std::string path = get_shm_name(msg->memory_id);

          if(!SharedMemoryInfo::open(shm, path, msg->sz)) {
            log_runtime.warning() << "Failed to open shared memory " << ID(msg->memory_id)
                                  << ':' << msg->sz << ' ' << path;
          }
          else {
            get_runtime()->remote_shared_memory_mappings.emplace(msg->memory_id, std::move(shm));
          }
        }
        // Count the number of messages handled, there should be one for every shareable
        // peer
        num_msgs_handled++;
        if (num_msgs_handled == Network::shared_peers.size()) {
          // All done opening shared memory regions for all the shared peers, wake up the
          // runtime initialization
          ShareableMemoryMessageHandler::cond_var.signal();
        }
      }
    };
    Mutex ShareableMemoryMessageHandler::mutex;
    Mutex::CondVar ShareableMemoryMessageHandler::cond_var(ShareableMemoryMessageHandler::mutex);
    size_t ShareableMemoryMessageHandler::num_msgs_handled = 0;

    static ActiveMessageHandlerReg<ShareableMemoryMessageHandler> shareable_memory_message_handler;

#if defined(REALM_USE_SHM) && defined(REALM_USE_ANONYMOUS_SHARED_MEMORY)
    static std::string get_mailbox_name(NodeID id) {
      return std::to_string(id) + '.' + std::to_string(Config::job_id);
    }
#endif

    void RuntimeImpl::create_shared_peers(void)
    {
#if defined(REALM_USE_SHM) and defined(REALM_USE_ANONYMOUS_SHARED_MEMORY)
      std::vector<OsHandle> handles;
      OsHandle all_node_mailbox =
          Realm::ipc_mailbox_create(get_mailbox_name(Network::my_node_id));
      Network::barrier(); // Wait for everyone to create their ipc mailboxes
      if(all_node_mailbox != Realm::INVALID_OS_HANDLE) {
        NodeSet send_nodes;
        for(NodeID node_id : Network::all_peers) {
          std::string slot_name = get_mailbox_name(node_id);
          if(!Realm::ipc_mailbox_send(all_node_mailbox, slot_name, handles,
                                      &(Network::my_node_id), sizeof(NodeID))) {
            log_runtime.info("Create shared_peers using ipc mailbox, but unable to "
                             "send msg to node %u, skipping",
                             (unsigned)node_id);
            continue;
          }
          send_nodes.add(node_id);
        }
        for(NodeID node_id : send_nodes) {
          NodeID recv_data = 0;
          size_t data_sz;
          std::string slot_name = get_mailbox_name(node_id);
          if(!Realm::ipc_mailbox_recv(all_node_mailbox, slot_name, handles, &recv_data,
                                      data_sz, sizeof(NodeID))) {
            log_runtime.warning("Create shared_peers using ipc mailbox, but unable to "
                                "recv msg from node %u, skipping",
                                (unsigned)node_id);
            continue;
          }
          assert(send_nodes.contains(recv_data) && "Received from unexpected node");
          Network::shared_peers.add(node_id);
        }
        shared_peers_use_network_module = false;
      } else {
        log_runtime.warning("Failed to create ipc mailbox for building shared_peers, so "
                            "fall back to use network modules");
      }

      // Wait for everyone to complete coordinating their shared_peers
      Network::barrier();
      close_handle(all_node_mailbox);
#endif
      if(shared_peers_use_network_module) {
        Network::shared_peers.empty();
        for(NetworkModule *module : network_modules) {
          module->get_shared_peers(Network::shared_peers);
        }
      }
    }

    bool RuntimeImpl::share_memories(void)
    {
#if defined(REALM_USE_SHM)
#if defined(REALM_USE_ANONYMOUS_SHARED_MEMORY)
      // Temporary structure defining the layout of the data sent along in the mailbox
      struct MessagePayload {
        realm_id_t mem;
        size_t size;
        MessagePayload() {}
        MessagePayload(realm_id_t _mem, size_t _size)
          : mem(_mem)
          , size(_size)
        {}
      };
      std::vector<OsHandle> my_handles, peer_handles;
      std::vector<MessagePayload> my_mem_ids, peer_mem_ids;
      static const size_t MAX_PEER_MEMORIES = 1024;
      OsHandle intra_node_mailbox =
          Realm::ipc_mailbox_create(get_mailbox_name(Network::my_node_id));
      if(intra_node_mailbox == Realm::INVALID_OS_HANDLE) {
        log_runtime.error("Failed to create ipc mailbox");
        return false;
      }
      {
        size_t idx = 0;
        my_handles.resize(local_shared_memory_mappings.size(), Realm::INVALID_OS_HANDLE);
        my_mem_ids.resize(local_shared_memory_mappings.size());
        for(std::unordered_map<realm_id_t, SharedMemoryInfo>::iterator it =
                local_shared_memory_mappings.begin();
            it != local_shared_memory_mappings.end(); ++it) {
          realm_id_t id = it->first;
          SharedMemoryInfo &shm = it->second;
          assert(shm.get_handle() != Realm::INVALID_OS_HANDLE);
          my_handles[idx] = shm.get_handle();
          my_mem_ids[idx] = MessagePayload(id, shm.get_size());
          idx++;
        }
      }

      log_runtime.info() << "Found " << my_handles.size() << " shared memories";

      Network::barrier(); // Wait for everyone to create their ipc mailboxes
                          // (TODO: only do on shared peers)
      // Share all the sharable memories.  Do this AFTER the network barrier to ensure
      // all the intra_node_mailboxes have been created
      for(NodeSetIterator it = Network::shared_peers.begin();
          it != Network::shared_peers.end(); ++it) {
        size_t data_sz;
        std::string slot_name = get_mailbox_name(*it);
        if(!Realm::ipc_mailbox_send(intra_node_mailbox, slot_name, my_handles,
                                    my_mem_ids.data(),
                                    my_mem_ids.size() * sizeof(my_mem_ids[0]))) {
          log_runtime.warning(
              "Unable to send shared memory information to node %u, skipping",
              (unsigned)*it);
          continue;
        }
        // TODO: modify ipc_mailbox_recv to peek and resize the data buffer the same as
        // the peer handles
        peer_mem_ids.resize(MAX_PEER_MEMORIES);
        if(!Realm::ipc_mailbox_recv(intra_node_mailbox, slot_name, peer_handles,
                                    peer_mem_ids.data(), data_sz,
                                    peer_mem_ids.size() * sizeof(peer_mem_ids[0]))) {
          log_runtime.warning(
              "Unable to recv shared memory information from node %u, skipping",
              (unsigned)*it);
          continue;
        }
        assert(peer_handles.size() * sizeof(peer_mem_ids[0]) == data_sz &&
               "Mismatch in received handles and ids");
        for(size_t p = 0; p < peer_handles.size(); p++) {
          SharedMemoryInfo peer_shm;
          if(!SharedMemoryInfo::open(peer_shm, peer_handles[p], peer_mem_ids[p].size)) {
            log_runtime.warning()
                << "Failed to open shared memory " << peer_mem_ids[p].mem;
            close_handle(peer_handles[p]);
          } else {
            peer_shm.unlink(); // No need to keep the OS resources around
            remote_shared_memory_mappings.emplace(peer_mem_ids[p].mem, std::move(peer_shm));
          }
        }
      }
      // Wait for everyone to complete opening their sharing
      // TODO: only do on shared peers
      Network::barrier();
      close_handle(intra_node_mailbox);
      return true;
#else  // REALM_USE_ANONYMOUS_SHARED_MEMORY
      ActiveMessage<ShareableMemoryMessageHandler> msg(
          Network::shared_peers, local_shared_memory_mappings.size() *
                                     sizeof(ShareableMemoryMessageHandler::Payload));
      for(std::unordered_map<realm_id_t, SharedMemoryInfo>::iterator it =
              local_shared_memory_mappings.begin();
          it != local_shared_memory_mappings.end(); ++it) {
        ShareableMemoryMessageHandler::Payload payload;
        payload.memory_id = it->first;
        payload.sz = it->second.get_size();
        msg.add_payload(&payload, sizeof(payload));
      }
      msg.commit();
      { // Wait for everyone to send their mappings
        AutoLock<> al(ShareableMemoryMessageHandler::mutex);
        while(ShareableMemoryMessageHandler::num_msgs_handled !=
              Network::shared_peers.size()) {
          ShareableMemoryMessageHandler::cond_var.wait();
        }
      }
      Network::barrier(); // Wait for everyone to get all their mappings from us
      return true;
#endif // REALM_USE_ANONYMOUS_MEMORY
#else  // REALM_USE_SHM
      return true;
#endif
    }

    static void allgather_announcement(Realm::Serialization::DynamicBufferSerializer &dbs,
                                       const NodeSet &targets, MachineImpl *machine,
                                       NetworkModule *network_module)
    {
      std::vector<char> all_announcements;
      std::vector<size_t> lengths(targets.size() + 1);
      char *buffer = nullptr;
      size_t rank = 0;

      // Use the networking module to exchange all the announcement information, by
      // whatever optimal path is available.  We assume a non-symmetric machine here,
      // so we use allgatherv.
      network_module->allgatherv(reinterpret_cast<const char *>(dbs.get_buffer()),
                                 dbs.bytes_used(), all_announcements, lengths);
      buffer = all_announcements.data();
      // Traverse the nodes _in-order_, as their data is laid out in the same order
      for(NodeID node_id = 0; node_id <= Network::max_node_id; node_id++) {
        if(node_id != Network::my_node_id) {
          if(!targets.contains(node_id)) {
            // Not a node that's collaborating here, so skip it and don't update the
            // buffer pointer
            continue;
          }
          machine->parse_node_announce_data(node_id, buffer, lengths[rank], true);
        }
        // Increment to the next section of the buffer with data for the next node id
        buffer += lengths[rank];
        rank++;
      }
    }

    void RuntimeImpl::parse_command_line(std::vector<std::string> &cmdline)
    {
      // very first thing - let the logger initialization happen
      Logger::configure_from_cmdline(cmdline);

      // calibrate timers
      int use_cpu_tsc = -1; // dont care
      uint64_t force_cpu_tsq_freq = 0; // no force
      {
        CommandLineParser cp;
        cp.add_option_int("-ll:cputsc", use_cpu_tsc);
        cp.add_option_int_units("-ll:tscfreq", force_cpu_tsq_freq, 'm', false/*!binary*/);
        bool ok = cp.parse_command_line(cmdline);
        assert(ok);
      }
      Clock::calibrate(use_cpu_tsc, force_cpu_tsq_freq);

#ifdef REALM_USE_NVTX
      // need to init nvtx at the very beginning
      std::vector<std::string> nvtx_module_list;
      {
        CommandLineParser cp;
        // modules are defined as the key of the nvtx_categories_predefined in nvtx.cc
        //   if all is passed, all modules will be enabled. 
        cp.add_option_stringlist("-ll:nvtx_modules", nvtx_module_list);
        bool ok = cp.parse_command_line(cmdline);
        assert(ok);
      }
      init_nvtx(nvtx_module_list);
#endif

      // configure network modules
      for(std::vector<NetworkModule *>::const_iterator it = network_modules.begin();
          it != network_modules.end();
          it++)
        (*it)->parse_command_line(this, cmdline);

      // configure module configs
      std::map<std::string, ModuleConfig*>::iterator it;
      for (it = module_configs.begin(); it != module_configs.end(); it++) {
        ModuleConfig *module_config = it->second;
        module_config->configure_from_cmdline(cmdline);
      }

      PartitioningOpQueue::configure_from_cmdline(cmdline);

      // parse the global Config
      {
        CommandLineParser cp;
        cp.add_option_int("-realm:eventloopcheck", Config::event_loop_detection_limit);
        cp.add_option_bool("-ll:force_kthreads", Config::force_kernel_threads);
        cp.add_option_bool("-ll:frsrv_fallback", Config::use_fast_reservation_fallback);
        cp.add_option_int("-ll:machine_query_cache", Config::use_machine_query_cache);
        cp.add_option_int("-ll:defalloc", Config::deferred_instance_allocation);
        cp.add_option_int("-ll:amprofile", Config::profile_activemsg_handlers);
        cp.add_option_int("-ll:aminline", Config::max_inline_message_time);
        bool cmdline_ok = cp.parse_command_line(cmdline);
        if(!cmdline_ok) {
          fprintf(stderr, "ERROR: failure parsing command line options for Config\n");
          exit(1);
        }
      }

      // load the CoreModuleConfig
      CoreModuleConfig *config =
          checked_cast<CoreModuleConfig *>(get_module_config("core"));
      assert(config != nullptr);

      core_map = CoreMap::discover_core_map(config->hyperthread_sharing);
      core_reservations = new CoreReservationSet(core_map);

      sampling_profiler.configure_from_cmdline(cmdline, *core_reservations);

      bgwork.configure_from_cmdline(cmdline);

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
    }

    void RuntimeImpl::finish_configure(void)
    {
      // mark all module config as finished
      std::map<std::string, ModuleConfig*>::iterator it;
      for (it = module_configs.begin(); it != module_configs.end(); it++) {
        ModuleConfig *module_config = it->second;
        module_config->finish_configure();
      }

      // start up the threading subsystem - modules will likely want threads
      if(!Threading::initialize()) exit(1);

      // now load modules
      module_registrar.create_static_modules(modules);
      module_registrar.create_dynamic_modules(modules);
      modules_created = true;

      // load the CoreModuleConfig
      CoreModuleConfig *config = dynamic_cast<CoreModuleConfig *>(get_module_config("core"));
      assert(config != nullptr);
      assert(config->finish_configured);

      // Check that we have enough resources for the number of nodes we are using
      if (Network::max_node_id > (NodeID)(ID::MAX_NODE_ID))
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes. Update the allocation "
		       "of bits in ID", Network::max_node_id+1, (ID::MAX_NODE_ID + 1));
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

      event_triggerer.add_to_manager(&bgwork);

      // initialize barrier timestamp
      BarrierImpl::barrier_adjustment_timestamp.store((((Barrier::timestamp_t)(Network::my_node_id)) << BarrierImpl::BARRIER_TIMESTAMP_NODEID_SHIFT) + 1);

      nodes = new Node[Network::max_node_id + 1];

      // configure the bit sets used by NodeSet
      {
	// choose a chunk size that's roughly the requested size, but
	//  clamp to [8,1024]
	size_t bitsets_per_chunk = ((config->bitset_chunk_size << 3) /
				    (Network::max_node_id + 1));
	if(bitsets_per_chunk < 8)
	  bitsets_per_chunk = 8;
	if(bitsets_per_chunk > 1024)
	  bitsets_per_chunk = 1024;
	// negative values of bitset_twolevel are a threshold
	bool use_twolevel = ((config->bitset_twolevel > 0) ||
			     ((config->bitset_twolevel < 0) &&
			      (Network::max_node_id >= -config->bitset_twolevel)));
	NodeSetBitmask::configure_allocator(Network::max_node_id,
					    bitsets_per_chunk,
					    use_twolevel);
      }

      // create allocators for local node events/locks/index spaces - do this before we start handling
      //  active messages
      {
	Node& n = nodes[Network::my_node_id];
	local_event_free_list = new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
	local_barrier_free_list = new BarrierTableAllocator::FreeList(n.barriers, Network::my_node_id);
	local_reservation_free_list = new ReservationTableAllocator::FreeList(n.reservations, Network::my_node_id);
	local_compqueue_free_list = new CompQueueTableAllocator::FreeList(n.compqueues, Network::my_node_id);

	local_sparsity_map_free_lists.resize(Network::max_node_id + 1);
	for(NodeID i = 0; i <= Network::max_node_id; i++) {
	  nodes[i].sparsity_maps.resize(Network::max_node_id + 1,
					atomic<DynamicTable<SparsityMapTableAllocator> *>(0));
	  DynamicTable<SparsityMapTableAllocator> *m = new DynamicTable<SparsityMapTableAllocator>;
	  nodes[i].sparsity_maps[Network::my_node_id].store(m);
	  local_sparsity_map_free_lists[i] = new SparsityMapTableAllocator::FreeList(*m, i /*owner_node*/);
	}

	local_subgraph_free_lists.resize(Network::max_node_id + 1);
	for(NodeID i = 0; i <= Network::max_node_id; i++) {
	  nodes[i].subgraphs.resize(Network::max_node_id + 1,
				    atomic<DynamicTable<SubgraphTableAllocator> *>(0));
	  DynamicTable<SubgraphTableAllocator> *m = new DynamicTable<SubgraphTableAllocator>;
	  nodes[i].subgraphs[Network::my_node_id].store(m);
	  local_subgraph_free_lists[i] = new SubgraphTableAllocator::FreeList(*m, i /*owner_node*/);
	}

	local_proc_group_free_lists.resize(Network::max_node_id + 1);
	for(NodeID i = 0; i <= Network::max_node_id; i++) {
	  nodes[i].proc_groups.resize(Network::max_node_id + 1,
				    atomic<DynamicTable<ProcessorGroupTableAllocator> *>(0));
	  DynamicTable<ProcessorGroupTableAllocator> *m = new DynamicTable<ProcessorGroupTableAllocator>;
	  nodes[i].proc_groups[Network::my_node_id].store(m);
	  local_proc_group_free_lists[i] = new ProcessorGroupTableAllocator::FreeList(*m, i /*owner_node*/);
	}
      }

      // form requests for network-registered memory
      if(config->reg_ib_mem_size > 0) {
	reg_ib_mem_segment.request(NetworkSegmentInfo::HostMem,
				   config->reg_ib_mem_size, 64);
	network_segments.push_back(&reg_ib_mem_segment);
      }
      if(config->reg_mem_size > 0) {
	reg_mem_segment.request(NetworkSegmentInfo::HostMem,
				config->reg_mem_size, 64);
	network_segments.push_back(&reg_mem_segment);
      }

      // construct active message handler table once before any network(s) init
      activemsg_handler_table.construct_handler_table();

      // and also our incoming active message manager
      message_manager = new IncomingMessageManager(Network::max_node_id + 1,
						   config->active_msg_handler_threads,
						   *core_reservations);
      if(config->active_msg_handler_bgwork)
	message_manager->add_to_manager(&bgwork);
      else
	assert(config->active_msg_handler_threads > 0);

        // Coordinate a job identifer across all the nodes in order to use it for
        // generating names in the system namespace (like files or sockets).  This needs
        // to come before the modules make their memories, but after the network is
        // initialized.  This cannot be currently done if GASNET1 is enabled, as the
        // broadcast function is not available until after Module::attach
#if !defined(REALM_USE_GASNET1)
      {
        Config::job_id =
            Network::broadcast(0, Clock::current_time_in_nanoseconds(true) + rand());
      }
#endif

      // create shared_peers either using ipc mailbox or network modules
      create_shared_peers();

      // initialize modules and create memories before we do network attach
      //  so that we have a chance to register these other memories for
      //  RDMA transfers
      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->initialize(this);

    for(std::vector<Module *>::const_iterator it = modules.begin(); it != modules.end();
        it++)
      (*it)->create_memories(this);

    Node *n = &nodes[Network::my_node_id];
    for(MemoryImpl *mem : n->memories) {
      NetworkSegment *seg = mem->get_network_segment();
      if(seg)
        network_segments.push_back(seg);
    }
    for(IBMemory *ibm : n->ib_memories) {
      NetworkSegment *seg = ibm->get_network_segment();
      if(seg)
        network_segments.push_back(seg);
    }

    // attach to the network
    for(std::vector<NetworkModule *>::const_iterator it = network_modules.begin();
        it != network_modules.end(); it++)
      (*it)->attach(this, network_segments);

    {
      // try to get all nodes to have roughly the same idea of the "zero
      //  "time" by using network barriers
      Network::barrier();
      Realm::Clock::set_zero_time();
      Network::barrier();
    }

#ifdef DEADLOCK_TRACE
      next_thread = 0;
      signaled_threads = 0;
      signal(SIGTERM, deadlock_catch);
      signal(SIGINT, deadlock_catch);
#endif
      const char *realm_freeze_env = getenv("REALM_FREEZE_ON_ERROR");
      const char *legion_freeze_env = getenv("LEGION_FREEZE_ON_ERROR"); 
      if (((legion_freeze_env != NULL) && (atoi(legion_freeze_env) != 0)) ||
          ((realm_freeze_env != NULL) && (atoi(realm_freeze_env) != 0))) {
	register_error_signal_handler(realm_freeze);
      } else {
        const char *realm_backtrace_env = getenv("REALM_BACKTRACE");
        const char *legion_backtrace_env = getenv("LEGION_BACKTRACE"); 
        if (((realm_backtrace_env != NULL) && (atoi(realm_backtrace_env) != 0)) ||
            ((legion_backtrace_env != NULL) && (atoi(legion_backtrace_env) != 0)))
          register_error_signal_handler(realm_backtrace);
      }

      // debugging tool to dump realm event graphs after a fixed delay
      //  (easier than actually detecting a hang)
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
      {
	const char *e = getenv("REALM_SHOW_EVENT_WAITERS");
	if(e) {
	  const char *pos;
	  int delay = strtol(e, (char **)&pos, 10);
	  assert(delay > 0);
	  if(*pos == '+')
	    delay += Network::my_node_id * atoi(pos + 1);
	  log_runtime.info() << "setting show_event alarm for " << delay << " seconds";
	  signal(SIGALRM, realm_show_events);
	  alarm(delay);
	}
      }
#endif
      
      bgwork.start_dedicated_workers(*core_reservations);

      PartitioningOpQueue::start_worker_threads(*core_reservations, &bgwork);

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

      // network-specific memories are created after attachment
      for(NetworkModule *module : network_modules) {
        module->create_memories(this);
      }
      for(NodeID node_id : Network::shared_peers) {
        log_runtime.debug() << Network::my_node_id << " is shareable with " << node_id;
      }

      LocalCPUMemory *regmem;
      if(config->reg_mem_size > 0) {
	void *regmem_base = reg_mem_segment.base;
	assert(regmem_base != 0);
	Memory m = get_runtime()->next_local_memory_id();
	regmem = new LocalCPUMemory(m,
				    config->reg_mem_size,
                                    -1/*don't care numa domain*/,
                                    Memory::REGDMA_MEM,
				    regmem_base,
				    &reg_mem_segment);
	get_runtime()->add_memory(regmem);
      } else
	regmem = 0;

      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_processors(this);

      IBMemory *reg_ib_mem;
      if(config->reg_ib_mem_size > 0) {
	void *reg_ib_mem_base = reg_ib_mem_segment.base;
	assert(reg_ib_mem_base != 0);
	Memory m = get_runtime()->next_local_ib_memory_id();
	reg_ib_mem = new IBMemory(m,
				  config->reg_ib_mem_size,
				  MemoryImpl::MKIND_SYSMEM,
				  Memory::REGDMA_MEM,
				  reg_ib_mem_base,
				  &reg_ib_mem_segment);
	get_runtime()->add_ib_memory(reg_ib_mem);
      } else
        reg_ib_mem = 0;

      // create local disk memory
      DiskMemory *diskmem;
      if(config->disk_mem_size > 0) {
        char file_name[30];
        snprintf(file_name, sizeof file_name, "disk_file%d.tmp", Network::my_node_id);
        Memory m = get_runtime()->next_local_memory_id();
        diskmem = new DiskMemory(m,
                                 config->disk_mem_size,
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
	(*it)->create_code_translators(this);
      
      // start dma system at the very ending of initialization
      // since we need list of local gpus to create channels
      if(config->dma_worker_threads > 0) {
        // warn about use of old flags
        log_runtime.warning() << "-ll:dma specified on command line no longer has effect - use -ll:bgwork to control background worker threads (which include dma work)";
      }
      start_dma_system(&bgwork);

      // now that we've created all the processors/etc., we can try to come up with core
      //  allocations that satisfy everybody's requirements - this will also start up any
      //  threads that have already been requested
      bool ok = core_reservations->satisfy_reservations(config->dummy_reservation_ok);
      if(ok) {
	if(config->show_reservations) {
	  std::cout << *core_map << std::endl;
	  core_reservations->report_reservations(std::cout);
	}
      } else {
	printf("HELP!  Could not satisfy all core reservations!\n");
	exit(1);
      }

      // create the "replicated heap" that puts instance layouts and sparsity
      //  maps where non-CPU devices can see them
      repl_heap.init(config->replheap_size, 1 /*chunks*/);

      if (!Network::shared_peers.empty() && local_shared_memory_mappings.size() > 0) {
        if (!share_memories()) {
          log_runtime.fatal("Failed to share memories with peers");
          abort();
        }
        for(std::unordered_map<realm_id_t, SharedMemoryInfo>::iterator it =
                local_shared_memory_mappings.begin();
            it != local_shared_memory_mappings.end(); ++it) {
          // We're done communicating our shared memories, so there's no need to keep the
          // sharing handles active, so unlink them now.
          it->second.unlink();
        }
      }

      for(std::vector<Module *>::const_iterator it = modules.begin();
	  it != modules.end();
	  it++)
	(*it)->create_dma_channels(this);

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

	std::set<Processor::Kind> local_cpu_kinds;
	local_cpu_kinds.insert(Processor::LOC_PROC);
	local_cpu_kinds.insert(Processor::UTIL_PROC);
	local_cpu_kinds.insert(Processor::IO_PROC);
	local_cpu_kinds.insert(Processor::PROC_SET);

      for(std::set<Processor::Kind>::const_iterator it = local_cpu_kinds.begin();
          it != local_cpu_kinds.end(); it++) {
        Processor::Kind k = *it;

        add_proc_mem_affinities(machine, procs_by_kind[k],
                                mems_by_kind[Memory::SYSTEM_MEM],
                                100, // "large" bandwidth
                                5    // "small" latency
        );

        add_proc_mem_affinities(machine, procs_by_kind[k],
                                mems_by_kind[Memory::REGDMA_MEM],
                                80, // "large" bandwidth
                                10  // "small" latency
        );

        add_proc_mem_affinities(machine, procs_by_kind[k],
                                mems_by_kind[Memory::SOCKET_MEM],
                                100, // "large" bandwidth
                                5    // "small" latency
        );

        add_proc_mem_affinities(machine, procs_by_kind[k], mems_by_kind[Memory::DISK_MEM],
                                5,  // "low" bandwidth
                                100 // "high" latency
        );

        add_proc_mem_affinities(machine, procs_by_kind[k], mems_by_kind[Memory::HDF_MEM],
                                5,  // "low" bandwidth
                                100 // "high" latency
        );

        add_proc_mem_affinities(machine, procs_by_kind[k], mems_by_kind[Memory::FILE_MEM],
                                5,  // low bandwidth
                                100 // high latency)
        );

        add_proc_mem_affinities(machine, procs_by_kind[k],
                                mems_by_kind[Memory::GLOBAL_MEM],
                                10, // "lower" bandwidth
                                50  // "higher" latency
        );
      }

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

      // retrieve process info
      {
        Machine::ProcessInfo process_info;
        int errcode =
            gethostname(process_info.hostname, Machine::ProcessInfo::MAX_HOSTNAME_LENGTH);
        if(errcode != 0) {
          log_runtime.warning() << "gethostname failed with " << errno;
        }
#ifdef REALM_ON_WINDOWS
        process_info.processid = GetCurrentProcessId();
        std::hash<std::string> hostname_hasher;
        process_info.hostid = hostname_hasher(process_info.hostname);
#else
        process_info.processid = getpid();
        process_info.hostid = gethostid();
#endif
        machine->add_process_info(Network::my_node_id, process_info);
      }

      // announce by network type
      Serialization::DynamicBufferSerializer dbs(4096);

      for(NetworkModule *module : network_modules) {
        // first, build the set of nodes we'll talk to
        NodeSet targets;
        for(NodeID i = 0; i <= Network::max_node_id; i++) {
          if((i != Network::my_node_id) && (Network::get_network(i) == module)) {
            targets.add(i);
          }
        }
        if(targets.empty()) {
          continue;
        }

        // Announcement needs to happen in two stages in order to ensure all memory
        // information is available for later serialization information (like remote
        // channels)
        // Stage 1: Announce all the memories attributes
        dbs.reset();
        ok = serialize_announce(dbs, n->memories, module);
        assert(ok && "Failed to serialize memories");
        ok = serialize_announce(dbs, n->ib_memories, module);
        assert(ok && "Failed to serialize ib memories");
        allgather_announcement(dbs, targets, machine, module);

        // Stage 2: Announce everything else.
        dbs.reset();
        ok = serialize_announce(dbs, n, machine, module);
        assert(ok && "Failed to serialize node for announcement");
        allgather_announcement(dbs, targets, machine, module);
      }

      // Now that we have full knowledge of the machine, update the machine model's
      // internal representation.  Start with the kind maps
      machine->update_kind_maps();
      // and the mem_mem affinities
      machine->enumerate_mem_mem_affinities();

      // Then update the path caches
      if (Config::path_cache_lru_size) {
        assert(Config::path_cache_lru_size > 0);
        init_path_cache();
      }
    }

    bool RuntimeImpl::configure_from_command_line(std::vector<std::string> &cmdline)
    {
      parse_command_line(cmdline);
      finish_configure();
      return true;
    }

    void RuntimeImpl::start(void)
    {
      // all we have to do here is tell the processors to start up their
      //  threads...
      for(std::vector<ProcessorImpl *>::const_iterator it = nodes[Network::my_node_id].processors.begin();
	  it != nodes[Network::my_node_id].processors.end();
	  ++it)
	(*it)->start_threads();

#ifdef REALM_USE_KOKKOS
      // now that the threads are started up, we can spin up the kokkos runtime
      KokkosInterop::kokkos_initialize(nodes[Network::my_node_id].processors);
#endif
    }

  template <typename T>
  Event spawn_on_all(const T& container_of_procs,
		     Processor::TaskFuncID func_id,
		     const void *args, size_t arglen,
		     Event start_event, int priority)
  {
    std::vector<Event> events;
    for (typename T::const_iterator it = container_of_procs.begin();
         it != container_of_procs.end(); it++) {
        Event e = (*it)->me.spawn(func_id, args, arglen, ProfilingRequestSet(),
                                  start_event, priority);
        events.push_back(e);
    }
    return Event::merge_events(events);
  }

  struct CollectiveSpawnInfo {
    Processor target_proc;
    Processor::TaskFuncID task_id;
    Event wait_on;
    int priority;
  };

#define DEBUG_COLLECTIVES

#ifdef DEBUG_COLLECTIVES
  template <typename T>
  static void broadcast_check(const T& val, const char *name)
  {
    T bval = Network::broadcast(0 /*root*/, val);
    if(val != bval) {
      log_collective.fatal() << "collective mismatch on node " << Network::my_node_id << " for " << name << ": " << val << " != " << bval;
      assert(false);
    }
  }
#endif

    Event RuntimeImpl::collective_spawn(Processor target_proc, Processor::TaskFuncID task_id, 
					const void *args, size_t arglen,
					Event wait_on /*= Event::NO_EVENT*/, int priority /*= 0*/)
    {
      log_collective.info() << "collective spawn: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " before=" << wait_on;

#ifdef DEBUG_COLLECTIVES
      broadcast_check(target_proc, "target_proc");
      broadcast_check(task_id, "task_id");
      broadcast_check(priority, "priority");
#endif

      // root node will be whoever owns the target proc
      NodeID root = ID(target_proc).proc_owner_node();

      if(Network::my_node_id == root) {
	// ROOT NODE

	// step 1: receive wait_on from every node
	std::vector<Event> all_events;
	Network::gather(root, wait_on, all_events);

	// step 2: merge all the events
	std::vector<Event> events;
	for(NodeID i = 0; i <= Network::max_node_id; i++) {
	  //log_collective.info() << "ev " << i << ": " << all_events[i];
	  if(all_events[i].exists())
	    events.push_back(all_events[i]);
	}

	Event merged_event = Event::merge_events(events);
	log_collective.info() << "merged precondition: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " before=" << merged_event;

	// step 3: run the task
	Event finish_event = target_proc.spawn(task_id, args, arglen, merged_event, priority);

	// step 4: broadcast the finish event to everyone else
	(void) Network::broadcast(root, finish_event);

	log_collective.info() << "collective spawn: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " after=" << finish_event;

	return finish_event;
      } else {
	// NON-ROOT NODE

	// step 1: send our wait_on to the root for merging
	Network::gather(root, wait_on);

	// steps 2 and 3: twiddle thumbs

	// step 4: receive finish event
	Event finish_event = Network::broadcast(root, Event::NO_EVENT);

	log_collective.info() << "collective spawn: proc=" << target_proc << " func=" << task_id << " priority=" << priority << " after=" << finish_event;

	return finish_event;
      }
    }

    Event RuntimeImpl::collective_spawn_by_kind(Processor::Kind target_kind, Processor::TaskFuncID task_id, 
						const void *args, size_t arglen,
						bool one_per_node /*= false*/,
						Event wait_on /*= Event::NO_EVENT*/, int priority /*= 0*/)
    {
      log_collective.info()
          << "collective spawn: kind=" << target_kind << " func=" << task_id
          << " priority=" << priority << " before=" << wait_on;

#ifdef DEBUG_COLLECTIVES
      broadcast_check(target_kind, "target_kind");
      broadcast_check(task_id, "task_id");
      broadcast_check(one_per_node, "one_per_node");
      broadcast_check(priority, "priority");
#endif

      // every node is involved in this one, so the root is arbitrary - we'll
      // pick node 0

      Event merged_event;

      if (Network::my_node_id == 0) {
        // ROOT NODE

        // step 1: receive wait_on from every node
        std::vector<Event> all_events;
        Network::gather(0 /*root*/, wait_on, all_events);

        // step 2: merge all the events
        // Remove all the non-existant events
        all_events.erase(std::remove_if(all_events.begin(), all_events.end(),
                                        [](Event e) { return !e.exists(); }),
                         all_events.end());

        merged_event = Event::merge_events(all_events);

        // step 3: broadcast the merged event back to everyone else
        (void)Network::broadcast(0 /*root*/, merged_event);
      } else {
        // NON-ROOT NODE

        // step 1: send our wait_on to the root for merging
        Network::gather(0 /*root*/, wait_on);

        // step 2: twiddle thumbs

        // step 3: receive merged wait_on event
        merged_event = Network::broadcast(0 /*root*/, Event::NO_EVENT);
      }

      // now spawn 0 or more local tasks
      std::vector<Event> events;

      const std::vector<ProcessorImpl *> &local_procs =
          nodes[Network::my_node_id].processors;

      for (std::vector<ProcessorImpl *>::const_iterator it =
               local_procs.begin();
           it != local_procs.end(); it++)
        if ((target_kind == Processor::NO_KIND) ||
            ((*it)->kind == target_kind)) {
          Event e =
              (*it)->me.spawn(task_id, args, arglen, ProfilingRequestSet(),
                              merged_event, priority);
          log_collective.info()
              << "spawn by kind: proc=" << (*it)->me << " func=" << task_id
              << " before=" << merged_event << " after=" << e;
          if (e.exists())
            events.push_back(e);

          if (one_per_node)
            break;
        }

      // local merge
      Event my_finish = Event::merge_events(events);

      if (Network::my_node_id == 0) {
        // ROOT NODE

        // step 1: receive wait_on from every node
        std::vector<Event> all_events;
        Network::gather(0 /*root*/, my_finish, all_events);
        // Remove all the non-existant events
        all_events.erase(std::remove_if(all_events.begin(), all_events.end(),
                                        [](Event e) { return !e.exists(); }),
                         all_events.end());

        Event merged_finish = Event::merge_events(all_events);

        // step 3: broadcast the merged event back to everyone
        (void)Network::broadcast(0 /*root*/, merged_finish);

        log_collective.info()
            << "collective spawn: kind=" << target_kind << " func=" << task_id
            << " priority=" << priority << " after=" << merged_finish;

        return merged_finish;
      } else {
        // NON-ROOT NODE

        // step 1: send our wait_on to the root for merging
        Network::gather(0 /*root*/, my_finish);

        // step 2: twiddle thumbs

        // step 3: receive merged wait_on event
        Event merged_finish = Network::broadcast(0 /*root*/, Event::NO_EVENT);

        log_collective.info()
            << "collective spawn: kind=" << target_kind << " func=" << task_id
            << " priority=" << priority << " after=" << merged_finish;

        return merged_finish;
      }
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
      
      if(task_id != 0) {
	if(style == Runtime::ONE_TASK_ONLY) {
	  // everybody needs to agree on this...
	  Processor p = nodes[0].processors[0]->me;
	  collective_spawn(p, task_id, args, arglen, init_event);
	} else {
	  collective_spawn_by_kind(Processor::NO_KIND, task_id, args, arglen,
				   (style == Runtime::ONE_TASK_PER_NODE),
				   init_event, 0 /*priority*/);
	}
      } else {
	// no main task!?
	assert(0);
      }

      // if we're in background mode, we just return to the caller now
      if(background)
	return;

      // otherwise, sleep until shutdown has been requested by somebody
      int result = wait_for_shutdown();
      exit(result);
    }

    bool RuntimeImpl::request_shutdown(Event wait_on, int result_code)
    {
      AutoLock<> al(shutdown_mutex);
      // if this is a duplicate request, it has to match exactly
      if(shutdown_request_received) {
	if((wait_on != shutdown_precondition) ||
	   (result_code != shutdown_result_code)) {
	  log_runtime.fatal() << "inconsistent shutdown requests:"
			      << " old=" << shutdown_precondition << "/" << shutdown_result_code
			      << " new=" << wait_on << "/" << result_code;
	  abort();
	}
	
	return true;
      } else {
	shutdown_precondition = wait_on;
	shutdown_result_code = result_code;
	shutdown_request_received = true;
	
	return false;
      }
    }

    void RuntimeImpl::initiate_shutdown(void)
    {
      // if we're the master, we need to notify everyone else first
      NodeID shutdown_master_node = 0;
      if((Network::my_node_id == shutdown_master_node) &&
	 (Network::max_node_id > 0)) {
      	NodeSet targets;
	for(NodeID i = 0; i <= Network::max_node_id; i++)
	  if(i != Network::my_node_id)
	    targets.add(i);

	ActiveMessage<RuntimeShutdownMessage> amsg(targets);
	amsg->result_code = shutdown_result_code;
	amsg.commit();
      }

      {
	AutoLock<> al(shutdown_mutex);
	assert(shutdown_request_received);
	shutdown_initiated = true;
	shutdown_condvar.broadcast();
      }
    }

    int RuntimeImpl::wait_for_shutdown(void)
    {
      // sleep until shutdown has been requested by somebody
      {
        AutoLock<> al(shutdown_mutex);
        while (!shutdown_initiated)
          shutdown_condvar.wait();
      }
      log_runtime.info("shutdown request received - terminating");

      // we need a task to run on each processor to ensure anything that was
      //  running when the shutdown was initiated (e.g. the task that initiated
      //  the shutdown) has finished - in legacy mode this is the "shutdown"
      //  task, otherwise it's just a NOP (task 0)
      {
        Processor::TaskFuncID flush_task_id =
            (run_method_called ? Processor::TASK_ID_PROCESSOR_SHUTDOWN
                               : Processor::TASK_ID_PROCESSOR_NOP);

        // legacy shutdown - call shutdown task on processors
        log_runtime.info() << "local processor shutdown tasks initiated";

        const std::vector<ProcessorImpl *> &local_procs =
            nodes[Network::my_node_id].processors;
        Event e =
            spawn_on_all(local_procs, flush_task_id, 0, 0, Event::NO_EVENT,
                         INT_MIN); // runs with lowest priority
        e.external_wait();
        log_runtime.info() << "local processor shutdown tasks complete";
      }

      {
        size_t n = num_untriggered_events.load();
        if (n != 0) {
          log_runtime.fatal() << n << " pending operations during shutdown!";
          abort();
        }
      }

      // the operation tables on every rank should be clear of work
      optable.shutdown_check();

      // make sure the network is completely quiescent
      if (Network::max_node_id > 0) {
        int tries = 0;
        while (true) {
          tries++;
          bool done = Network::check_for_quiescence(message_manager);
          if (done) {
            if (Network::my_node_id == 0)
              log_runtime.info() << "quiescent after " << tries << " attempts";
            break;
          }

          if (tries >= 10) {
            log_runtime.fatal()
                << "network still not quiescent after " << tries << " attempts";
            abort();
          }
        }
      }

      // mark that a shutdown is in progress so that we can hopefully catch
      //  things that try to run during teardown
      shutdown_in_progress.store(true);

#ifdef REALM_USE_KOKKOS
      // finalize the kokkos runtime
      KokkosInterop::kokkos_finalize(nodes[Network::my_node_id].processors);
#endif

      // Shutdown all the threads

      // stop processors before most other things, as they may be helping with
      //  background work
      {
        std::vector<ProcessorImpl *> &local_procs =
            nodes[Network::my_node_id].processors;
        for (std::vector<ProcessorImpl *>::const_iterator it =
                 local_procs.begin();
             it != local_procs.end(); it++)
          (*it)->shutdown();
      }

      // threads that cause inter-node communication have to stop first
      PartitioningOpQueue::stop_worker_threads();

      for (std::vector<Channel *>::iterator it =
               nodes[Network::my_node_id].dma_channels.begin();
           it != nodes[Network::my_node_id].dma_channels.end(); ++it)
        (*it)->shutdown();
      stop_dma_system();

      repl_heap.cleanup();

      // let network-dependent cleanup happen before we detach
      for (std::vector<Module *>::iterator it = modules.begin();
           it != modules.end(); it++) {
        (*it)->pre_detach_cleanup();
      }

      // detach from the network
      for (std::vector<NetworkModule *>::const_iterator it =
               network_modules.begin();
           it != network_modules.end(); it++)
        (*it)->detach(this, network_segments);

#ifdef DEBUG_REALM
      event_triggerer.shutdown_work_item();
#endif
      bgwork.stop_dedicated_workers();

      // tear down the active message manager
      message_manager->shutdown();
      delete message_manager;

      sampling_profiler.shutdown();

      if (Config::profile_activemsg_handlers)
        activemsg_handler_table.report_message_handler_stats();

#ifdef EVENT_TRACING
      if (event_trace_file) {
        printf("writing event trace to %s\n", event_trace_file);
        Tracer<EventTraceItem>::dump_trace(event_trace_file, false);
        free(event_trace_file);
        event_trace_file = 0;
      }
#endif
#ifdef LOCK_TRACING
      if (lock_trace_file) {
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
               Network::my_node_id, rt->local_event_free_list->next_alloc,
               rt->local_reservation_free_list->next_alloc,
               rt->local_index_space_free_list->next_alloc,
               rt->local_proc_group_free_list->next_alloc);
      }
#endif
      cleanup_query_caches();
      {
        // Clean up all the modules before tearing down the runtime state.
        for (std::vector<Module *>::iterator it = modules.begin();
             it != modules.end(); it++) {
          (*it)->cleanup();
          delete (*it);
        }

        for (std::vector<NetworkModule *>::iterator it =
                 network_modules.begin();
             it != network_modules.end(); it++) {
          (*it)->cleanup();
          delete (*it);
        }
        Network::single_network = 0;

        // clean up all the module configs
        for(std::map<std::string, ModuleConfig *>::iterator it = module_configs.begin();
            it != module_configs.end(); it++) {
          delete(it->second);
          it->second = nullptr;
        }

        // dlclose all dynamic module handles
        module_registrar.unload_module_sofiles();

        delete[] nodes;
        delete local_event_free_list;
        delete local_barrier_free_list;
        delete local_reservation_free_list;
        delete local_compqueue_free_list;
        delete_container_contents(local_sparsity_map_free_lists);
        delete_container_contents(local_subgraph_free_lists);
        delete_container_contents(local_proc_group_free_lists);

        // same for code translators
        delete_container_contents(code_translators);

        // Clear the global nodesets that potentially reference dynamic bitmasks that will
        // be free'd when we free the allocations.
        // TODO: properly manage the life-time of the nodeset bitmask allocations to avoid
        // this issue for future nodesets
        Network::all_peers.clear();
        Network::shared_peers.clear();

        NodeSetBitmask::free_allocations();
      }

      if (!Threading::cleanup())
        exit(1);

      // very last step - unregister our signal handlers
      unregister_error_signal_handler();

      if (Config::path_cache_lru_size) {
        finalize_path_cache();
      }

#ifdef REALM_USE_NVTX
      // finalize nvtx
      finalize_nvtx();
#endif

      return shutdown_result_code;
    }

    bool RuntimeImpl::create_configs(int argc, char **argv)
    {
      if (!module_configs_created) {
        std::vector<std::string> cmdline;
        cmdline.reserve(argc);
        for(int i = 1; i < argc; i++)
          cmdline.push_back(argv[i]);
        module_registrar.create_static_module_configs(module_configs);
        module_registrar.create_dynamic_module_configs(cmdline, module_configs);
        module_configs_created = true;
      }
      return true;
    }

    ModuleConfig* RuntimeImpl::get_module_config(const std::string name)
    {
      std::map<std::string, ModuleConfig*>::iterator it;
      it = module_configs.find(name);
      if (it == module_configs.end()) {
        return NULL;
      } else {
        return it->second;
      }
    }

    EventImpl *RuntimeImpl::get_event_impl(Event e)
    {
      if(shutdown_in_progress.load()) {
	log_runtime.fatal() << "looking up event after shutdown: " << e;
	abort();
      }
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

      GenEventImpl *impl;
      if(NodeID(id.event_creator_node()) == Network::my_node_id) {
	// use our shallower local event table
	impl = local_events.lookup_entry(id.event_gen_event_idx(),
					 id.event_creator_node());
      } else {
	Node *n = &nodes[id.event_creator_node()];
	impl = n->remote_events.lookup_entry(id.event_gen_event_idx(),
					     id.event_creator_node());
      }
      {
	ID check(impl->me);
	assert(check.event_creator_node() == id.event_creator_node());
	assert(check.event_gen_event_idx() == id.event_gen_event_idx());
      }
      return impl;
    }

    BarrierImpl *RuntimeImpl::get_barrier_impl(Event e)
    {
      ID id(e);
      assert(id.is_barrier());

      Node *n = &nodes[id.barrier_creator_node()];
      BarrierImpl *impl = n->barriers.lookup_entry(id.barrier_barrier_idx(),
						   id.barrier_creator_node());
      {
	ID check(impl->me);
	assert(check.barrier_creator_node() == id.barrier_creator_node());
	assert(check.barrier_barrier_idx() == id.barrier_barrier_idx());
      }
      return impl;
    }

    SparsityMapImplWrapper *RuntimeImpl::get_sparsity_impl(ID id)
    {
      if(!id.is_sparsity()) {
	log_runtime.fatal() << "invalid index space sparsity handle: id=" << id;
	assert(0 && "invalid index space sparsity handle");
      }

      Node *n = &nodes[id.sparsity_owner_node()];
      atomic<DynamicTable<SparsityMapTableAllocator> *>& m = n->sparsity_maps[id.sparsity_creator_node()];
      // might need to construct this (in a lock-free way)
      DynamicTable<SparsityMapTableAllocator> *mptr = m.load();
      if(mptr == 0) {
	// construct one and try to swap it in
	DynamicTable<SparsityMapTableAllocator> *newm = new DynamicTable<SparsityMapTableAllocator>;
	if(m.compare_exchange(mptr, newm))
	  mptr = newm;  // we're using the one we made
	else
	  delete newm;  // somebody else made it faster (mptr has winner)
      }
      SparsityMapImplWrapper *impl = mptr->lookup_entry(id.sparsity_sparsity_idx(),
							id.sparsity_owner_node());
      // creator node isn't always right, so try to fix it
      if(impl->me != id) {
	if(impl->me.sparsity_creator_node() == 0)
	  impl->me.sparsity_creator_node() = NodeID(id.sparsity_creator_node());
	assert(impl->me == id);
      }
      return impl;
    }
  
    SparsityMapImplWrapper *RuntimeImpl::get_available_sparsity_impl(NodeID target_node)
    {
      SparsityMapImplWrapper *wrap = local_sparsity_map_free_lists[target_node]->alloc_entry();
      wrap->me.sparsity_creator_node() = Network::my_node_id;
      return wrap;
    }

    void RuntimeImpl::free_sparsity_impl(SparsityMapImplWrapper *impl)
    {
      assert(
          local_sparsity_map_free_lists[impl->me.sparsity_owner_node()]->table.has_entry(
              impl->me.sparsity_sparsity_idx()));
      local_sparsity_map_free_lists[impl->me.sparsity_owner_node()]->free_entry(impl);
    }

    SubgraphImpl *RuntimeImpl::get_subgraph_impl(ID id)
    {
      if(!id.is_subgraph()) {
	log_runtime.fatal() << "invalid subgraph handle: id=" << id;
	assert(0 && "invalid subgraph handle");
      }

      Node *n = &nodes[id.subgraph_owner_node()];
      atomic<DynamicTable<SubgraphTableAllocator> *>& m = n->subgraphs[id.subgraph_creator_node()];
      // might need to construct this (in a lock-free way)
      DynamicTable<SubgraphTableAllocator> *mptr = m.load();
      if(mptr == 0) {
	// construct one and try to swap it in
	DynamicTable<SubgraphTableAllocator> *newm = new DynamicTable<SubgraphTableAllocator>;
	if(m.compare_exchange(mptr, newm))
	  mptr = newm;  // we're using the one we made
	else
	  delete newm;  // somebody else made it faster (mptr has winner)
      }
      SubgraphImpl *impl = mptr->lookup_entry(id.subgraph_subgraph_idx(),
					      id.subgraph_owner_node());
      // creator node isn't always right, so try to fix it
      if(impl->me != id) {
	if(impl->me.subgraph_creator_node() == 0)
	  impl->me.subgraph_creator_node() = NodeID(id.subgraph_creator_node());
	assert(impl->me == id);
      }
      return impl;
    }
  
    ReservationImpl *RuntimeImpl::get_lock_impl(ID id)
    {
      if(id.is_reservation()) {
	Node *n = &nodes[id.rsrv_creator_node()];
	ReservationImpl *impl = n->reservations.lookup_entry(id.rsrv_rsrv_idx(),
							     id.rsrv_creator_node());
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
	return null_check(nodes[id.memory_owner_node()].memories[id.memory_mem_idx()]);
      }

      if(id.is_ib_memory()) {
        return null_check(nodes[id.memory_owner_node()].ib_memories[id.memory_mem_idx()]);
      }
#ifdef TODO
      if(id.is_allocator()) {
	return null_check(nodes[id.allocator.owner_node].memories[id.allocator.mem_idx]);
      }
#endif

      if(id.is_instance()) {
	return null_check(nodes[id.instance_owner_node()].memories[id.instance_mem_idx()]);
      }

      log_runtime.fatal() << "invalid memory handle: id=" << id;
      assert(0 && "invalid memory handle");
      return 0;
    }

    IBMemory *RuntimeImpl::get_ib_memory_impl(ID id)
    {
      if(id.is_ib_memory()) {
        return null_check(nodes[id.memory_owner_node()].ib_memories[id.memory_mem_idx()]);
      }

      log_runtime.fatal() << "invalid ib memory handle: id=" << id;
      assert(0 && "invalid ib memory handle");
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

      return null_check(nodes[id.proc_owner_node()].processors[id.proc_proc_idx()]);
    }

    ProcessorGroupImpl *RuntimeImpl::get_procgroup_impl(ID id)
    {
      if(!id.is_procgroup()) {
	log_runtime.fatal() << "invalid processor group handle: id=" << id;
	assert(0 && "invalid processor group handle");
      }

      Node *n = &nodes[id.pgroup_owner_node()];
      atomic<DynamicTable<ProcessorGroupTableAllocator> *>& m = n->proc_groups[id.pgroup_creator_node()];
      // might need to construct this (in a lock-free way)
      DynamicTable<ProcessorGroupTableAllocator> *mptr = m.load();
      if(mptr == 0) {
	// construct one and try to swap it in
	DynamicTable<ProcessorGroupTableAllocator> *newm = new DynamicTable<ProcessorGroupTableAllocator>;
	if(m.compare_exchange(mptr, newm))
	  mptr = newm;  // we're using the one we made
	else
	  delete newm;  // somebody else made it faster (mptr has winner)
      }
      ProcessorGroupImpl *impl = mptr->lookup_entry(id.pgroup_pgroup_idx(),
						    id.pgroup_owner_node());
      // creator node isn't always right, so try to fix it
      if(ID(impl->me) != id) {
	ID fixed = impl->me;
	if(fixed.pgroup_creator_node() == 0) {
	  fixed.pgroup_creator_node() = NodeID(id.pgroup_creator_node());
	  impl->me = fixed.convert<Processor>();
	}
	assert(impl->me == id.convert<Processor>());
      }
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
      AutoLock<> al(mem->mutex);

      // TODO: factor creator_node into lookup!
      if(id.instance.inst_idx >= mem->instances.size()) {
	assert(id.instance.owner_node != Network::my_node_id);

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
	  //printf("[%d] creating proxy instance: inst=" IDFMT "\n", Network::my_node_id, id.id());
	  mem->instances[id.instance.inst_idx] = new RegionInstanceImpl(id.convert<RegionInstance>(), mem->me);
	}
      }
	  
      return mem->instances[id.instance.inst_idx];
#endif
    }

    CompQueueImpl *RuntimeImpl::get_compqueue_impl(ID id)
    {
      if(!id.is_compqueue()) {
	log_runtime.fatal() << "invalid completion queue handle: id=" << id;
	assert(0 && "invalid completion queue handle");
      }

      Node *n = &nodes[id.pgroup_owner_node()];
      CompQueueImpl *impl = n->compqueues.lookup_entry(id.compqueue_cq_idx(),
						       id.compqueue_owner_node());
      assert(impl->me == id.convert<CompletionQueue>());
      return impl;
    }

    /*static*/
    void RuntimeImpl::realm_backtrace(int signal)
    {
      // the signal handler has been called before, it is called again because
      // an error is occured during printing the trace, to avoid handling signals 
      // recursively, we just exit.
      if (ThreadLocal::error_signal_value != 0) {
        std::cerr << "Signal " << signal 
                  << " raised inside realm signal handler, previous caught signal " << ThreadLocal::error_signal_value
                  << std::endl;
        unregister_error_signal_handler();
        abort();
      }
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
      assert((signal == SIGINT) || (signal == SIGFPE) ||
             (signal == SIGABRT) || (signal == SIGSEGV) ||
             (signal == SIGBUS) || (signal == SIGILL));
#endif
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
              Network::my_node_id, (unsigned long)pthread_self(), buffer);
      fflush(stderr);
      free(buffer);
      free(funcname);
#endif
      ThreadLocal::error_signal_value = signal;
      std::cerr << "Signal " << signal << " received by node " << Network::my_node_id
#ifdef REALM_ON_WINDOWS
                << ", process " << GetCurrentProcessId()
                << " (thread " << GetCurrentThreadId()
#else
                << ", process " << getpid()
                << " (thread "  << std::hex << uintptr_t(pthread_self())
#endif
                << std::dec << ") - obtaining backtrace\n" << std::flush;

      Backtrace bt;
      bt.capture_backtrace(1 /* skip this handler */);
      bt.lookup_symbols();
      fflush(stdout);
      fflush(stderr);
      std::cout << std::flush;
      std::cerr << "Signal " << signal
#ifdef REALM_ON_WINDOWS
                << " received by process " << GetCurrentProcessId()
                << " (thread " << GetCurrentThreadId()
#else
                << " received by process " << getpid()
                << " (thread " << std::hex << uintptr_t(pthread_self())
#endif
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

    Node::Node(void) {}

    Node::~Node(void)
    {
      // delete processors, memories, nodes, etc.
      delete_container_contents(memories);
      delete_container_contents(processors);
      delete_container_contents(ib_memories);
      delete_container_contents(dma_channels);

      for(atomic<DynamicTable<SparsityMapTableAllocator> *> &atomic_sparsity :
          sparsity_maps) {
        delete atomic_sparsity.load();
      }

      for(atomic<DynamicTable<SubgraphTableAllocator> *> &atomic_subgraph : subgraphs) {
        delete atomic_subgraph.load();
      }

      for(atomic<DynamicTable<ProcessorGroupTableAllocator> *> &atomic_proc_group :
          proc_groups) {
        delete atomic_proc_group.load();
      }
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RuntimeShutdownMessage
  //

  /*static*/ void RuntimeShutdownRequest::handle_message(NodeID sender,
							 const RuntimeShutdownRequest &args,
							 const void *data, size_t datalen)
  {
    log_runtime.info() << "shutdown request received: sender=" << sender
		       << " wait_on=" << args.wait_on
		       << " code=" << args.result_code;

    RuntimeImpl *r_impl = runtime_singleton;
    bool duplicate = r_impl->request_shutdown(args.wait_on, args.result_code);
    if(!duplicate) {
      if(args.wait_on.has_triggered())
	r_impl->initiate_shutdown();
      else
	r_impl->deferred_shutdown.defer(r_impl, args.wait_on);
    }
  }

  /*static*/ void RuntimeShutdownMessage::handle_message(NodeID sender,
							 const RuntimeShutdownMessage &args,
							 const void *data, size_t datalen)
  {
    log_runtime.info() << "shutdown initiation received: sender=" << sender
		       << " code=" << args.result_code;

    RuntimeImpl *r_impl = runtime_singleton;
    bool duplicate = r_impl->request_shutdown(Event::NO_EVENT, args.result_code);
    assert(!duplicate);
    r_impl->initiate_shutdown();
  }

  ActiveMessageHandlerReg<RuntimeShutdownRequest> runtime_shutdown_request_handler;
  ActiveMessageHandlerReg<RuntimeShutdownMessage> runtime_shutdown_message_handler;

}; // namespace Realm
