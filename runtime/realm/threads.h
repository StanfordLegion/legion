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

// generic Realm interface to threading libraries (e.g. pthreads)

#ifndef REALM_THREADS_H
#define REALM_THREADS_H

#include "realm/realm_config.h"
#include "realm/activemsg.h"

#ifdef REALM_USE_USER_THREADS
#ifdef __MACH__
#define _XOPEN_SOURCE
#endif
#endif

#include <stddef.h>

#include <string>
#include <list>
#include <set>
#include <map>
#include <deque>
#include <iostream>

#ifdef REALM_USE_PAPI
// string.h is not used here, but if this is included by somebody else after
//  we include papi.h, mismatches occur because ffsll() is declared with __THROW!?
#include <string.h>
#include <papi.h>
#endif

namespace Realm {

  namespace Threading {
    // calls to initialize and cleanup any global state for the threading subsystem
    bool initialize(void);
    bool cleanup(void);
  };

  class Operation;

  // ALL work inside Realm (i.e. both tasks and internal Realm work) should be done
  //  inside a Thread, which comes in (at least) two flavors:
  // a) KernelThread - a kernel-managed thread, supporting preemptive multitasking
  //     and possibly-blocking system calls
  // b) UserThread - a userspace-managed thread, supporting only cooperative multitasking
  //     but with (hopefully) much lower switching overhead
  //
  // Cooperative multitasking is handled with the help of a "scheduler" that the thread
  //  calls into when it wishes to sleep
  //
  // Threads return a void * on completion, available to any object that "joins" on it.
  // A thread may also be sent a "signal", which can be delivered either synchronously (i.e.
  //  upon interaction with the scheduler) or asynchronously (e.g. via POSIX signals).

  class ThreadLaunchParameters;
  class ThreadScheduler;
  class CoreReservation;

  // from profiling.h
  class ProfilingMeasurementCollection;

#ifdef REALM_USE_PAPI
  class PAPICounters;
#endif

  //template <class CONDTYPE> class ThreadWaker;

  class Thread {
  protected:
    // thread objects are not constructed directly
    Thread(ThreadScheduler *_scheduler);    

    template <typename T, void (T::*START_MTHD)(void)>
    static void thread_entry_wrapper(void *obj);
 
    static Thread *create_kernel_thread_untyped(void *target, void (*entry_wrapper)(void *),
						const ThreadLaunchParameters& params,
						CoreReservation& rsrv,
						ThreadScheduler *_scheduler);
   
#ifdef REALM_USE_USER_THREADS
    static Thread *create_user_thread_untyped(void *target, void (*entry_wrapper)(void *),
					      const ThreadLaunchParameters& params,
					      const CoreReservation *rsrv,
					      ThreadScheduler *_scheduler);
#endif
   
  public:
    // for kernel threads, the scheduler is optional - however, a thread with no scheduler
    //  is not allowed to wait on a Realm Event or any internal object
    // a kernel thread also requires a core reservation that tells it which core(s) it may
    //  use when executing
    template <typename T, void (T::*START_MTHD)(void)>
    static Thread *create_kernel_thread(T *target,
					const ThreadLaunchParameters& params,
					CoreReservation& rsrv,
					ThreadScheduler *_scheduler = 0);

#ifdef REALM_USE_USER_THREADS
    // user threads must specify a scheduler - the whole point is that the OS isn't
    //  controlling them...
    template <typename T, void (T::*START_MTHD)(void)>
    static Thread *create_user_thread(T *target,
				      const ThreadLaunchParameters& params,
				      const CoreReservation *rsrv,
				      ThreadScheduler *_scheduler);
#endif

    virtual ~Thread(void);

    enum State { STATE_CREATED,
		 STATE_STARTUP,
		 STATE_RUNNING,
		 STATE_BLOCKING,
		 STATE_BLOCKED,
		 STATE_ALERTED,
		 STATE_READY,
		 STATE_FINISHED,
		 STATE_DELETED,
                 };

    State get_state(void);

    enum Signal { TSIG_NONE,
		  TSIG_SHOW_BACKTRACE,
		  TSIG_INTERRUPT,
    };

    // adds a signal to the thread's queue, triggering an asynchronous notification
    //  if 'asynchronous' is true
    void signal(Signal sig, bool asynchronous);

    // returns the next signal in the queue, if any
    Signal pop_signal(void);

    // pops and handles any signals in the queue
    void process_signals(void);

    virtual void join(void) = 0; // BLOCKS until the thread completes
    virtual void detach(void) = 0;

    // called from within a thread
    static Thread *self(void);
    static void abort(void);
    static void yield(void);

    // called from within thread to indicate the association of an Operation with the thread
    //  for cancellation reasons
    void start_operation(Operation *op);
    void stop_operation(Operation *op);
    Operation *get_operation(void) const;

    // changes the priority of the thread (and, by extension, the operation it
    //   is working on)
    void set_priority(int new_priority);

#ifdef REALM_USE_USER_THREADS
    // perform a user-level thread switch
    // if called from a kernel thread, that thread becomes the "host" for the user thread
    // if called by a user thread with 'switch_to'==0, control returns to host
    static void user_switch(Thread *switch_to);

    // some systems do not appear to support user thread switching for
    //  reasons unknown, so allow code to test to see if it's working first
    static bool test_user_switch_support(size_t stack_size = 1 << 20);
#endif

    template <typename CONDTYPE>
    static void wait_for_condition(const CONDTYPE& cond, bool& poisoned);

    // does this thread have exception handlers installed?
    bool exceptions_permitted(void) const;

    // put one of these in a try {} block to indicate that exceptions are allowed
    class ExceptionHandlerPresence {
    public:
      ExceptionHandlerPresence(void);
      ~ExceptionHandlerPresence(void);
    };

    // per-thread performance counters
    void setup_perf_counters(const ProfilingMeasurementCollection& pmc);
    void start_perf_counters(void);
    void stop_perf_counters(void);
    void record_perf_counters(ProfilingMeasurementCollection& pmc);

  protected:
    friend class ThreadScheduler;

    template <class CONDTYPE>
    friend class ThreadWaker;

    // atomically updates the thread's state, returning the old state
    Thread::State update_state(Thread::State new_state);

    // updates the thread's state, but only if it's in the specified 'old_state' (i.e. an
    //  atomic compare and swap) - returns true on success and false on failure
    bool try_update_state(Thread::State old_state, Thread::State new_state);

    // send an asynchronous notification to the thread
    virtual void alert_thread(void) = 0;

    State state;
    ThreadScheduler *scheduler;
    Operation *current_op;
    int exception_handler_count;
    int signal_count;
    GASNetHSL signal_mutex;
    std::deque<Signal> signal_queue;

#ifdef REALM_USE_PAPI
    PAPICounters *papi_counters;
#endif
  };

  // Finally, a Thread may operate as a co-routine, "yielding" intermediate values and 
  //  suspending until it is "resumed" (with an optional value)

  template <typename YT, typename RT = int>
  class Coroutine : public Thread {
  public:
    virtual ~Coroutine(void);

    YT get_yield_value(void);
    void resume(RT value = RT());

    static RT yield(YT value);

  protected:
    YT yield_value;
    RT resume_value;
  };

  class ThreadScheduler {
  public:
    virtual ~ThreadScheduler(void);

    // this can be used for logging or to hold a thread before it starts running
    virtual void thread_starting(Thread *thread) = 0;

    // callbacks from a thread when it wants to sleep (i.e. yielding on a co-routine interaction
    //  or blocking on some condition) or terminate - either will generally result in some other
    //  thread being woken up)
    //virtual void thread_yielding(Thread *thread) = 0;
    virtual void thread_blocking(Thread *thread) = 0;
    virtual void thread_terminating(Thread *thread) = 0;

    // notification that a thread is ready (this will generally come from some thread other
    //  than the one that's now ready)
    virtual void thread_ready(Thread *thread) = 0;

    virtual void set_thread_priority(Thread *thread, int new_priority) = 0;

  protected:
    // delegates friendship of Thread with subclasses
    Thread::State update_thread_state(Thread *thread, Thread::State new_state);
    bool try_update_thread_state(Thread *thread, Thread::State old_state, Thread::State new_state);
  };

  template <typename T, T _DEFAULT>
  struct WithDefault {
  public:
    static const T DEFAULT_VALUE = _DEFAULT;

    WithDefault(void); // uses default
    WithDefault(T _val);

    operator T(void) const;
    WithDefault<T,_DEFAULT>& operator=(T newval);

  protected:
    T val;
  };

  // any thread (user or kernel) will have its own stack and heap - the size of which can
  //  be controlled when the thread is launched - defaults are provided for all
  //  values, along with convenient mutators
  class ThreadLaunchParameters {
  public:
    static const ptrdiff_t STACK_SIZE_DEFAULT = -1;
    static const ptrdiff_t HEAP_SIZE_DEFAULT = -1;

    WithDefault<ptrdiff_t, STACK_SIZE_DEFAULT> stack_size;
    WithDefault<ptrdiff_t, HEAP_SIZE_DEFAULT> heap_size;

    ThreadLaunchParameters(void);

    ThreadLaunchParameters& set_stack_size(ptrdiff_t new_stack_size);
    ThreadLaunchParameters& set_heap_size(ptrdiff_t new_heap_size);
  };

  // Kernel threads will generally be much happier if they decide up front which core(s)
  //  each of them are going to use.  Since this is a global optimization problem, we allow
  //  different parts of the system to create "reservations", which the runtime will then
  //  attempt to satisfy.  A thread can be launched before this happens, but will not actually
  //  run until the reservations are satisfied.
  
  // A reservation can request one or more cores, optionally restricted to a particular NUMA
  //  domain (as numbered by the OS).  The reservation should also indicate how heavily (if at
  //  all) it intends to use the integer, floating-point, and load/store datapaths of the core(s).
  //  A reservation with EXCLUSIVE use is compatible with those expecting MINIMAL use of the
  //  same datapath, but not with any other reservation desiring EXCLUSIVE or SHARED access.
  class CoreReservationParameters {
  public:
    enum CoreUsage { CORE_USAGE_NONE,
		     CORE_USAGE_MINIMAL,
		     CORE_USAGE_SHARED,
		     CORE_USAGE_EXCLUSIVE };

    static const int NUMA_DOMAIN_DONTCARE = -1;
    static const ptrdiff_t STACK_SIZE_DEFAULT = -1;
    static const ptrdiff_t HEAP_SIZE_DEFAULT = -1;

    WithDefault<int, 1>                        num_cores;   // how many cores are requested
    WithDefault<int, NUMA_DOMAIN_DONTCARE>     numa_domain; // which NUMA domain the cores should come from
    WithDefault<CoreUsage, CORE_USAGE_SHARED>  alu_usage;   // "integer" datapath usage
    WithDefault<CoreUsage, CORE_USAGE_MINIMAL> fpu_usage;   // floating-point usage
    WithDefault<CoreUsage, CORE_USAGE_SHARED>  ldst_usage;  // "memory" datapath usage
    WithDefault<ptrdiff_t, STACK_SIZE_DEFAULT> max_stack_size;
    WithDefault<ptrdiff_t, HEAP_SIZE_DEFAULT>  max_heap_size;

    CoreReservationParameters(void);

    CoreReservationParameters& set_num_cores(int new_num_cores);
    CoreReservationParameters& set_numa_domain(int new_numa_domain);
    CoreReservationParameters& set_alu_usage(CoreUsage new_alu_usage);
    CoreReservationParameters& set_fpu_usage(CoreUsage new_fpu_usage);
    CoreReservationParameters& set_ldst_usage(CoreUsage new_ldst_usage);
    CoreReservationParameters& set_max_stack_size(ptrdiff_t new_max_stack_size);
    CoreReservationParameters& set_max_heap_size(ptrdiff_t new_max_heap_size);
  };

  class CoreReservationSet;

  class CoreReservation {
  public:
    CoreReservation(const std::string& _name, CoreReservationSet &crs,
		    const CoreReservationParameters& _params);

    // eventually we'll get an Allocation, which is an opaque type because it's OS-dependent :(
    struct Allocation;

    // to be informed of the eventual allocation, you supply one of these:
    class NotificationListener {
    public:
      virtual ~NotificationListener(void) {}
      virtual void notify_allocation(const CoreReservation& rsrv) = 0;
    };

    void add_listener(NotificationListener *listener);

  public:
    std::string name;
    CoreReservationParameters params;

    // no locks needed here because we aren't multi-threaded until the allocation exists
    Allocation *allocation;
  protected:
    friend class CoreReservationSet;
    void notify_listeners(void);

    std::list<NotificationListener *> listeners;
  };    

  // a description of the actual (host, for now) processor cores available in the system
  // we are most interested in the enumeration of them and the ways in which the cores
  //   share datapaths, which will impact how we assign reservations to cores
  class CoreMap {
  public:
    CoreMap(void);
    ~CoreMap(void);

    void clear(void);

    friend std::ostream& operator<<(std::ostream& os, const CoreMap& cm);

    // in general, you'll want to discover the core map rather than set it up yourself
    // hyperthread_sharing - if true, hyperthreads are considered to share a core, which prevents
    //                         them both from being if a reservation asks for exclusive access to the
    //                         core
    //                       if false, hyperthreads are considered to be separate cores, making more
    //                         cores available, but potentially exposing the app to contention issues
    static CoreMap *discover_core_map(bool hyperthread_sharing);

    // creates a simple synthetic core map - it is symmetric and hierarchical:
    //   numa domains -> cores -> fp clusters (shared fpu) -> hyperthreads (shared alu/ldst)
    static CoreMap *create_synthetic(int num_domains, int cores_per_domain,
				     int hyperthreads = 1, int fp_cluster_size = 1);

    struct Proc {
      int id;      // a unique integer id
      int domain;  // which (NUMA) domain is it in
      std::set<int> kernel_proc_ids;  // set of kernel processor IDs (might be empty)
      std::set<Proc *> shares_alu;    // which other procs does this share an ALU with
      std::set<Proc *> shares_fpu;    // which other procs does this share an FPU with
      std::set<Proc *> shares_ldst;   // which other procs does this share an LD/ST path with
    };

    typedef std::map<int, Proc *> ProcMap;
    typedef std::map<int, ProcMap> DomainMap;

    ProcMap all_procs;
    DomainMap by_domain;
  };

  // manages a set of core reservations and if/how they are satisfied
  class CoreReservationSet {
  public:
    // if constructed without a CoreMap, it'll attempt to discover one itself
    CoreReservationSet(bool hyperthread_sharing);
    CoreReservationSet(const CoreMap* _cm);

    ~CoreReservationSet(void);

    const CoreMap *get_core_map(void) const;

    void add_reservation(CoreReservation& rsrv);

    // if 'dummy_reservation_ok' is set, a failed reservation will be "satisfied" with
    //  one that uses dummy (i.e. no assign cores) reservations
    bool satisfy_reservations(bool dummy_reservation_ok = false);

    void report_reservations(std::ostream& os) const;

  protected:
    bool owns_coremap;
    const CoreMap *cm;
    std::map<CoreReservation *, CoreReservation::Allocation *> allocations;
  };

#if 0
  class ThreadScheduler;

  class KernelThread : public Thread {
  public:
    KernelThread(CoreReservation& _rsrv, ThreadScheduler *scheduler);
    virtual ~KernelThread(void);

    virtual State state(void) = 0;
    virtual void await_state(State desired, bool invert = false) = 0;
    virtual void *yield_value(void) = 0;
    virtual void *finish_value(void) = 0;

    virtual void resume(void *value = 0) = 0;
    virtual void signal(int sig, bool asynchronous) = 0;
    virtual void detach(void) = 0;
  };

  class Thread;
  class ThreadLaunchParameters;
  class ThreadReservation;
  class ThreadReservationParameters;
  class ThreadFactory;

  // basically a struct with nice defaults

  // also a struct with defaults
  class ThreadReservationParameters {

    ThreadReservationParameters(void)
      : num_cores(1)
      , numa_domain(NUMA_DOMAIN_DONTCARE)
      , core_usage(CORE_USAGE_SHARED)
      , fpu_usage(CORE_USAGE_MINIMAL)
      , max_stack_size(STACK_SIZE_DEFAULT)
      , max_heap_size(HEAP_SIZE_DEFAULT)
    {}
    virtual ~ThreadReservationParameters(void) {}
  };

  // object that describes a thread from the "outside" - operations performed
  //  by the thread itself are static methods on this class (i.e. you don't need
  //  to carry around your Thread * inside the thread)
  class ThreadFactory {
  public:
    virtual ~ThreadFactory(void) = 0;

    virtual ThreadReservation *add_reservation(const ThreadReservationParameters& params) = 0;
    virtual bool satisfy_reservations(void) = 0;

    template <typename T, void *(T::*MTHD)(void *)>
    Thread *create_thread(T *target, void *data, 
			  ThreadReservation *rsrv, const ThreadLaunchParameters& params);

    typedef void *(* ThreadEntryUntypedFnptr)(void *, void *);

  protected:
    template <typename T, void *(T::*MTHD)(void *)>
    static void *thread_entry_untyped(void *target, void *data) {
      return (((T *)target)->*MTHD)(data);
    }

    virtual Thread *create_thread_untyped(ThreadEntryUntypedFnptr fnptr,
					  void *target, void *data, 
					  ThreadReservation *rsrv,
					  const ThreadLaunchParameters& params) = 0;
  };

  template <typename T, void *(T::*MTHD)(void *)>
  Thread *ThreadFactory::create_thread(T *target, void *data, 
				       ThreadReservation *rsrv, const ThreadLaunchParameters& params)
  {
    return create_thread_untyped(thread_entry_untyped<T, MTHD>,
				 target, data, rsrv, params);
  }

  // ThreadReservations cannot be created directly - they are subclassesed and created/destroyed
  //  by actual ThreadFactory implementations
  class ThreadReservation {
  protected:
    ThreadReservation(void) {}
    ~ThreadReservation(void) {}
  };
#endif

#ifdef REALM_USE_PAPI
  class PAPICounters {
  protected:
    PAPICounters(void);
    ~PAPICounters(void);

  public:
    static PAPICounters *setup_counters(const ProfilingMeasurementCollection& pmc);
    void cleanup(void);

    void start(void);
    void suspend(void);
    void resume(void);
    void stop(void);
    void record(ProfilingMeasurementCollection& pmc);

  protected:
    int papi_event_set;
    std::map<int, size_t> event_codes;
    std::vector<long long> event_counts;
  };
#endif

  // move this somewhere else

  class DummyLock {
  public:
    void lock(void) {}
    void unlock(void) {}
  };

} // namespace Realm

#include "realm/threads.inl"

#endif // REALM_THREADS_H
