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

#include "realm/threads.h"

#include "realm/logging.h"
#include "realm/faults.h"
#include "realm/operation.h"

#ifdef DEBUG_USWITCH
#include <stdio.h>
#endif

#include <pthread.h>
#include <errno.h>
// for PTHREAD_STACK_MIN
#include <limits.h>
#ifdef __MACH__
// for sched_yield
#include <sched.h>
#endif

#ifdef REALM_USE_USER_THREADS
#include <ucontext.h>
#ifdef __MACH__
// MacOS has (loudly) deprecated set/get/make/swapcontext,
//  despite there being no POSIX replacement for them...
// this check is on the use, not the declaration, so we wrap them here

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
inline int getcontext_wrap(ucontext_t *u) { return getcontext(u); }
inline int swapcontext_wrap(ucontext_t *u1, const ucontext_t *u2) { return swapcontext(u1, u2); }
inline void makecontext_wrap(ucontext_t *u, void (*fn)(), int args, ...) { makecontext(u, fn, 0); }
#pragma GCC diagnostic pop
#define getcontext getcontext_wrap
#define swapcontext swapcontext_wrap
#define makecontext makecontext_wrap
#endif
#endif

#ifdef REALM_USE_HWLOC
#include <hwloc.h>
#endif

#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <string>
#include <map>

#ifdef __linux__
// needed for scanning Linux's /sys
#include <dirent.h>
#include <stdio.h>
#ifdef REALM_USE_HWLOC
#include <hwloc/linux.h>
#endif
#endif

#ifndef CHECK_LIBC
#define CHECK_LIBC(cmd) do { \
  errno = 0; \
  int ret = (cmd); \
  if(ret != 0) { \
    std::cerr << "ERROR: " __FILE__ ":" << __LINE__ << ": " #cmd " = " << ret << " (" << strerror(errno) << ")" << std::endl;	\
    assert(0); \
  } \
} while(0)
#endif

#ifndef CHECK_PTHREAD
#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    std::cerr << "PTHREAD: " #cmd " = " << ret << " (" << strerror(ret) << ")" << std::endl;	\
    assert(0); \
  } \
} while(0)
#endif

namespace Realm {

  Logger log_thread("threads");

#ifdef REALM_USE_PAPI
  Logger log_papi("papi");
  namespace PAPI {
    bool papi_available = false;
  };
#endif

  namespace ThreadLocal {
    /*extern*/ __thread Thread *current_thread = 0;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreReservation

  // we keep a global map of reservations to their allocations (this is inherently a global problem)
  std::map<CoreReservation *, CoreReservation::Allocation *> allocations;

  CoreReservation::CoreReservation(const std::string& _name, CoreReservationSet &crs,
				   const CoreReservationParameters& _params)
    : name(_name), params(_params), allocation(0)
  {
    // reservations automatically add themselves to the set
    crs.add_reservation(*this);

    log_thread.info() << "reservation created: " << name;
  }

  // with pthreads on Linux, an allocation is the cpu_set used for affinity
  struct CoreReservation::Allocation {
    bool exclusive_ownership;
    std::set<int> proc_ids;
#ifndef __MACH__
    bool restrict_cpus;  // if true, thread is confined to set below
    cpu_set_t allowed_cpus;
#endif
  };

  void CoreReservation::add_listener(NotificationListener *listener)
  {
    if(allocation) {
      // already have allocation - just call back immediately
      listener->notify_allocation(*this);
    } else {
      listeners.push_back(listener);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreReservationSet

  CoreReservationSet::CoreReservationSet(bool hyperthread_sharing)
    : owns_coremap(true), cm(0)
  {
    cm = CoreMap::discover_core_map(hyperthread_sharing);
  }

  CoreReservationSet::CoreReservationSet(const CoreMap *_cm)
    : owns_coremap(false), cm(_cm)
  {
  }

  CoreReservationSet::~CoreReservationSet(void)
  {
    if(owns_coremap)
      delete const_cast<CoreMap *>(cm);
    
    // we don't own the CoreReservation *'s in the allocation map, but we do own the 
    //  allocations
    for(std::map<CoreReservation *, CoreReservation::Allocation *>::iterator it = allocations.begin();
	it != allocations.end();
	it++)
      delete it->second;
    allocations.clear();
  }

  const CoreMap *CoreReservationSet::get_core_map(void) const
  {
    return cm;
  }

  void CoreReservationSet::add_reservation(CoreReservation& rsrv)
  {
    assert(allocations.count(&rsrv) == 0);
    allocations[&rsrv] = 0;
  }

  static bool can_add_usage(CoreReservationParameters::CoreUsage current,
			    CoreReservationParameters::CoreUsage reqd)
  {
    switch(current) {
    case CoreReservationParameters::CORE_USAGE_EXCLUSIVE:
      {
	// exclusive cannot coexist with exclusive or shared
	if(reqd == CoreReservationParameters::CORE_USAGE_EXCLUSIVE) return false;
	if(reqd == CoreReservationParameters::CORE_USAGE_SHARED) return false;
	return true;
      }

    case CoreReservationParameters::CORE_USAGE_SHARED:
      {
	// shared cannot coexist with exclusive
	if(reqd == CoreReservationParameters::CORE_USAGE_EXCLUSIVE) return false;
	return true;
      }

    default:
      {
	// NONE and MINIMAL are fine
	return true;
      }
    }
  }

  static void add_usage(CoreReservationParameters::CoreUsage& current,
			CoreReservationParameters::CoreUsage reqd)
  {
    // this ends up being a simple max
    if(reqd > current)
      current = reqd;
  }		

  // versions of the above that understand shared cores
  static bool can_add_usage(const std::map<const CoreMap::Proc *,
			                   CoreReservationParameters::CoreUsage>& current,
			    CoreReservationParameters::CoreUsage reqd,
			    const CoreMap::Proc *p,
			    const std::set<CoreMap::Proc *>& shared)
  {
    std::map<const CoreMap::Proc *, CoreReservationParameters::CoreUsage>::const_iterator it;
    it = current.find(p);
    if((it != current.end()) && !can_add_usage(it->second, reqd)) return false;

    for(std::set<CoreMap::Proc *>::const_iterator it2 = shared.begin();
	it2 != shared.end();
	it2++) {
      it = current.find(*it2);
      if((it != current.end()) && !can_add_usage(it->second, reqd)) return false;
    }

    return true;
  }

  static void add_usage(std::map<const CoreMap::Proc *,
			         CoreReservationParameters::CoreUsage>& current,
			CoreReservationParameters::CoreUsage reqd,
			const CoreMap::Proc *p,
			const std::set<CoreMap::Proc *>& shared)
  {
    std::map<const CoreMap::Proc *, CoreReservationParameters::CoreUsage>::iterator it;
    it = current.find(p);
    if(it != current.end())
      add_usage(it->second, reqd);
    else
      current.insert(std::make_pair(p, reqd));

    for(std::set<CoreMap::Proc *>::const_iterator it2 = shared.begin();
	it2 != shared.end();
	it2++) {
      it = current.find(*it2);
      if(it != current.end())
	add_usage(it->second, reqd);
      else
	current.insert(std::make_pair(*it2, reqd));
    }
  }

  // attempts to find an allocation satisfying all the reservation requests in 'allocs' -
  //  if any allocations are already present, those are preserved (possibly causing the
  //  allocation attempt to fail)
  static bool attempt_allocation(const CoreMap& cm,
				 std::map<CoreReservation *, CoreReservation::Allocation *>& allocs)
  {
    // we'll need to keep track of the usage level of each core
    std::map<const CoreMap::Proc *, CoreReservationParameters::CoreUsage> alu_usage, fpu_usage, ldst_usage;
    std::map<const CoreMap::Proc *, int> user_count;

    // iterate through the requests and sort them by whether or not they have any exclusivity
    //  demands and whether they're limited to a particular numa domain
    // also record pre-allocated reservations and their usage
    std::map<std::pair<bool, int>, std::set<CoreReservation *> > to_satisfy;
    for(std::map<CoreReservation *, CoreReservation::Allocation *>::iterator it = allocs.begin();
	it != allocs.end();
	it++) {
      CoreReservation *rsrv = it->first;
      CoreReservation::Allocation *alloc = it->second;
      if(alloc) {
	for(std::set<int>::const_iterator it = alloc->proc_ids.begin();
	    it != alloc->proc_ids.end();
	    it++) {
	  // get the corresponding CoreMap::Proc
	  CoreMap::ProcMap::const_iterator it2 = cm.all_procs.find(*it);
	  if(it2 == cm.all_procs.end()) {
	    log_thread.error() << "existing allocation ('" << rsrv->name << "') has an unknown proc id (" << *it << ")";
	    return false; // no way to fix this
	  }
	  const CoreMap::Proc *p = it2->second;

	  // update/check user_count
	  if(alloc->exclusive_ownership && (user_count.count(p) > 0)) {
	    log_thread.error() << "existing allocation ('" << rsrv->name << "') has unsatisfiable exclusivity on proc id (" << p->id << ")";
	    return false; // no way to fix this
	  }
	  user_count[p]++;

	  // update/check usage
	  if(!(can_add_usage(alu_usage, rsrv->params.alu_usage, p, p->shares_alu) &&
	       can_add_usage(fpu_usage, rsrv->params.fpu_usage, p, p->shares_fpu) &&
	       can_add_usage(ldst_usage, rsrv->params.ldst_usage, p, p->shares_ldst))) {
	    log_thread.error() << "existing allocation ('" << rsrv->name << "') has unsatisfiable usage on proc id (" << p->id << ")";
	    return false; // no way to fix this
	  }
	  add_usage(alu_usage, rsrv->params.alu_usage, p, p->shares_alu);
	  add_usage(fpu_usage, rsrv->params.fpu_usage, p, p->shares_fpu);
	  add_usage(ldst_usage, rsrv->params.ldst_usage, p, p->shares_ldst);
	}
      } else {
	std::pair<bool, int> key = std::make_pair(((rsrv->params.alu_usage == CoreReservationParameters::CORE_USAGE_EXCLUSIVE) ||
						   (rsrv->params.fpu_usage == CoreReservationParameters::CORE_USAGE_EXCLUSIVE) ||
						   (rsrv->params.ldst_usage == CoreReservationParameters::CORE_USAGE_EXCLUSIVE)),
						  rsrv->params.numa_domain);
	to_satisfy[key].insert(rsrv);
      }
    }

    // ok, now attempt to satisfy all the requests
    // by _reverse_ iterating over to_satisfy, we consider exclusive requests before shared
    //  and those that want a particular numa domain before those that don't care
    std::map<CoreReservation *, std::set<const CoreMap::Proc *> > assigned_procs;
    for(std::map<std::pair<bool, int>, std::set<CoreReservation *> >::reverse_iterator it = to_satisfy.rbegin();
	it != to_satisfy.rend();
	it++) {
      bool has_exclusive = it->first.first;
      int req_domain = it->first.second;

      std::vector<const CoreMap::Proc *> pm;
      if(req_domain >= 0) {
	CoreMap::DomainMap::const_iterator it2 = cm.by_domain.find(req_domain);
	if(it2 == cm.by_domain.end()) {
	  log_thread.error() << "one or more reservations requiring unknown domain (" << req_domain << ")";
	  return false;
	}
	for(CoreMap::ProcMap::const_iterator it3 = it2->second.begin(); it3 != it2->second.end(); it3++)
	  pm.push_back(it3->second);
      } else {
	// shuffle the procs from the different domains to get a roughly-even distribution
	std::list<std::pair<const CoreMap::ProcMap *, CoreMap::ProcMap::const_iterator> > rr;
	for(CoreMap::DomainMap::const_iterator it2 = cm.by_domain.begin();
	    it2 != cm.by_domain.end();
	    it2++)
	  if(!(it2->second.empty()))
	    rr.push_back(std::make_pair(&(it2->second), it2->second.begin()));
	while(!(rr.empty())) {
	  std::pair<const CoreMap::ProcMap *, CoreMap::ProcMap::const_iterator> x = rr.front();
	  rr.pop_front();
	  pm.push_back(x.second->second);
	  if(++x.second != x.first->end())
	    rr.push_back(x);
	}
      }

      for(std::set<CoreReservation *>::iterator it2 = it->second.begin();
	  it2 != it->second.end();
	  it2++) {
	CoreReservation *rsrv = *it2;
	std::set<const CoreMap::Proc *>& procs = assigned_procs[rsrv];

	// iterate over all the possibly available processors and see if any fit
	for(std::vector<const CoreMap::Proc *>::iterator it3 = pm.begin();
	    it3 != pm.end();
	    it3++)
	{
	  const CoreMap::Proc *p = *it3;

	  // is there already conflicting usage?
	  if(!(can_add_usage(alu_usage, rsrv->params.alu_usage, p, p->shares_alu) &&
	       can_add_usage(fpu_usage, rsrv->params.fpu_usage, p, p->shares_fpu) &&
	       can_add_usage(ldst_usage, rsrv->params.ldst_usage, p, p->shares_ldst)))
	    continue;

	  // yes, do so and add this to the assigned procs
	  add_usage(alu_usage, rsrv->params.alu_usage, p, p->shares_alu);
	  add_usage(fpu_usage, rsrv->params.fpu_usage, p, p->shares_fpu);
	  add_usage(ldst_usage, rsrv->params.ldst_usage, p, p->shares_ldst);
	  procs.insert(p);

	  // an exclusive reservation request stops as soon as we have enough, while
	  //  a shared reservation will use any/all compatible processors
	  if(has_exclusive && ((int)(procs.size()) >= rsrv->params.num_cores))
	    break;
	}

	// if we didn't get enough, we've failed this allocation
	if((int)(procs.size()) < rsrv->params.num_cores) {
	  log_thread.warning() << "reservation ('" << rsrv->name << "') cannot be satisfied";
	  return false;
	}
      }
    }

    // if we got all the way through, we're successful and can now fill in the new allocations
    for(std::map<CoreReservation *, std::set<const CoreMap::Proc *> >::iterator it = assigned_procs.begin();
	it != assigned_procs.end();
	it++) {
      CoreReservation *rsrv = it->first;
      CoreReservation::Allocation *alloc = new CoreReservation::Allocation;

      alloc->exclusive_ownership = true;  // unless we set it false below
#ifndef __MACH__
      alloc->restrict_cpus = false; // unless we set it to true below
      CPU_ZERO(&alloc->allowed_cpus);
#endif

      for(std::set<const CoreMap::Proc *>::iterator it2 = it->second.begin();
	  it2 != it->second.end();
	  it2++) {
	const CoreMap::Proc *p = *it2;

	alloc->proc_ids.insert(p->id);
	if(user_count[p] > 1)
	  alloc->exclusive_ownership = false;
#ifndef __MACH__
	if(!(p->kernel_proc_ids.empty())) {
	  alloc->restrict_cpus = true;
	  for(std::set<int>::const_iterator it3 = p->kernel_proc_ids.begin();
	      it3 != p->kernel_proc_ids.end();
	      it3++)
	    CPU_SET(*it3, &alloc->allowed_cpus);
	}
#endif
      }

      if(rsrv->allocation) {
	log_thread.info() << "replacing allocation for reservation '" << rsrv->name << "'";
	CoreReservation::Allocation *old_alloc = rsrv->allocation;
	rsrv->allocation = alloc;
	delete old_alloc; // TODO: reference count once we allow updates
      } else
	rsrv->allocation = alloc;

      allocs[rsrv] = alloc;
    }

    return true;
  }

  bool CoreReservationSet::satisfy_reservations(bool dummy_reservation_ok /*= false*/)
  {
    // remember who is missing an allocation - we'll need to notify them
    std::set<CoreReservation *> missing;
    for(std::map<CoreReservation *, CoreReservation::Allocation *>::iterator it = allocations.begin();
	it != allocations.end();
	it++)
      if(!it->second)
	missing.insert(it->first);

    // one shot for now - eventually allow a reservation to say it's willing to be
    //  adjusted if needed
    bool ok = attempt_allocation(*cm,
				 allocations);
    if(!ok) {
      if(!dummy_reservation_ok)
	return false;

      // dummy allocations for everyone!
      for(std::set<CoreReservation *>::iterator it = missing.begin();
	  it != missing.end();
	  it++) {
	CoreReservation *rsrv = *it;

	CoreReservation::Allocation *alloc = new CoreReservation::Allocation;

	alloc->exclusive_ownership = true;  // unless we set it false below
#ifndef __MACH__
	alloc->restrict_cpus = false; // unless we set it to true below
	CPU_ZERO(&alloc->allowed_cpus);
#endif
	rsrv->allocation = alloc;
	allocations[rsrv] = alloc;
      }
    }      

    // for all the reservations that were missing allocations, notify any registered listeners
    for(std::set<CoreReservation *>::iterator it = missing.begin();
	it != missing.end();
	it++) {
      CoreReservation *rsrv = *it;
      for(std::list<CoreReservation::NotificationListener *>::iterator it = rsrv->listeners.begin();
	  it != rsrv->listeners.end();
	  it++)
	(*it)->notify_allocation(*rsrv);
    }

    return true;
  }

  template <typename T>
  static std::ostream& operator<<(std::ostream &os, const std::set<T>& s)
  {
    os << '<';
    if(!(s.empty())) {
      typename std::set<T>::const_iterator it = s.begin();
      while(true) {
	os << *it;
	if(++it == s.end()) break;
	os << ',';
      }
    }
    os << '>';
    return os;
  }

  void CoreReservationSet::report_reservations(std::ostream& os) const
  {
    // iterate over the allocation map and print stuff out
    for(std::map<CoreReservation *, CoreReservation::Allocation *>::const_iterator it = allocations.begin();
	it != allocations.end();
	it++) {
      const CoreReservation *rsrv = it->first;
      const CoreReservation::Allocation *alloc = it->second;
      os << rsrv->name << ": ";
      if(alloc) {
	os << "allocated " << alloc->proc_ids;
      } else {
	os << "not allocated";
      }
      os << std::endl;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Thread

  static bool handler_registered = false;
  // Valgrind uses SIGUSR2 on Darwin
  static int handler_signal = SIGUSR1;

  static void signal_handler(int signal, siginfo_t *info, void *context)
  {
    if(signal == handler_signal) {
      // somebody pinged us to look at our signals
      Thread *t = ThreadLocal::current_thread;
      assert(t);
      t->process_signals();
      return;
    }
    
    Backtrace bt;
    bt.capture_backtrace();
    bt.lookup_symbols();
    log_thread.error() << "received unexpected signal " << signal << " backtrace=" << bt;
  }

  static void register_handler(void)
  {
    if(!__sync_bool_compare_and_swap(&handler_registered, false, true)) return;

    struct sigaction act;
    bzero(&act, sizeof(act));
    act.sa_sigaction = &signal_handler;
    act.sa_flags = SA_SIGINFO;

    CHECK_LIBC( sigaction(handler_signal, &act, 0) );
  }

  void Thread::signal(Signal sig, bool asynchronous)
  {
    log_thread.info() << "sending signal: target=" << (void *)this << " signal=" << sig << " async=" << asynchronous;
    {
      AutoHSLLock a(signal_mutex);
      signal_queue.push_back(sig);
    }
    int prev = __sync_fetch_and_add(&signal_count, 1);
    if((prev == 0) && asynchronous)
      alert_thread();
  }

  Thread::Signal Thread::pop_signal(void)
  {
    if(signal_count) {
      Signal sig;
      __sync_fetch_and_sub(&signal_count, 1);
      AutoHSLLock a(signal_mutex);
      sig = signal_queue.front();
      signal_queue.pop_front();
      return sig;
    } else
      return TSIG_NONE;
  }

  void Thread::process_signals(void)
  {
    // should only be called from the thread itself
    assert(this == Thread::self());

    while(signal_count > 0) {
      Signal sig;
      {
	__sync_fetch_and_sub(&signal_count, 1);
	AutoHSLLock a(signal_mutex);
	// should never be empty, as there's no race conditions on emptying the queue
	assert(!signal_queue.empty());
	sig = signal_queue.front();
	signal_queue.pop_front();
      }

      switch(sig) {
      case Thread::TSIG_INTERRUPT: 
	{
	  Operation *op = current_op;
	  if(op && op->cancellation_requested()) {
#ifdef REALM_USE_EXCEPTIONS
	    if(exceptions_permitted()) {
  	      throw CancellationException();
	    } else
#endif
	    {
	      log_thread.fatal() << "no handler for TSIG_INTERRUPT: thread=" << this << " op=" << op;
	      assert(0);
	    }
	  } else
	    log_thread.warning() << "unwanted TSIG_INTERRUPT: thread=" << this << " op=" << op;
	  break;
	}
      default: 
	{
	  assert(0);
	}
      }
    }
  }

  // changes the priority of the thread (and, by extension, the operation it
  //   is working on)
  void Thread::set_priority(int new_priority)
  {
    assert(scheduler != 0);
    scheduler->set_thread_priority(this, new_priority);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class KernelThread

  class KernelThread : public Thread {
  public:
    KernelThread(void *_target, void (*_entry_wrapper)(void *),
		 ThreadScheduler *_scheduler);

    virtual ~KernelThread(void);

    void start_thread(const ThreadLaunchParameters& params,
		      const CoreReservation& rsrv);

    virtual void join(void);
    virtual void detach(void);

  protected:
    static void *pthread_entry(void *data);

    virtual void alert_thread(void);

    void *target;
    void (*entry_wrapper)(void *);
    pthread_t thread;
    bool ok_to_delete;
  };

  KernelThread::KernelThread(void *_target, void (*_entry_wrapper)(void *),
			     ThreadScheduler *_scheduler)
    : Thread(_scheduler), target(_target), entry_wrapper(_entry_wrapper)
    , ok_to_delete(false)
  {
  }

  KernelThread::~KernelThread(void)
  {
    // insist that a thread be join()'d or detach()'d before deletion
    assert(ok_to_delete);
  }

  /*static*/ void *KernelThread::pthread_entry(void *data)
  {
    KernelThread *thread = (KernelThread *)data;

    // set up TLS so people can find us
    ThreadLocal::current_thread = thread;

    log_thread.info() << "thread " << thread << " started";
    thread->update_state(STATE_RUNNING);

    if(thread->scheduler)
      thread->scheduler->thread_starting(thread);
    
    // call the actual thread body
    (*thread->entry_wrapper)(thread->target);

    // on return, we update our status and terminate
    log_thread.info() << "thread " << thread << " finished";
    thread->update_state(STATE_FINISHED);

    // this is last so that the scheduler can delete us if it wants to
    if(thread->scheduler)
      thread->scheduler->thread_terminating(thread);
    
    return 0;
  }

  void KernelThread::start_thread(const ThreadLaunchParameters& params,
				  const CoreReservation& rsrv)
  {
    // before we create any threads, make sure we have our signal handler registered
    register_handler();

    pthread_attr_t attr;

    CHECK_PTHREAD( pthread_attr_init(&attr) );

    // allocation better exist...
    assert(rsrv.allocation);

#ifndef __MACH__
    if(rsrv.allocation->restrict_cpus)
      CHECK_PTHREAD( pthread_attr_setaffinity_np(&attr, 
						 sizeof(rsrv.allocation->allowed_cpus),
						 &(rsrv.allocation->allowed_cpus)) );
#endif

    // pthreads also has min stack size, except that you have to guess what it
    //  is when using glibc - the PTHREAD_STACK_MIN value is the min size of
    //  the stack _AFTER_ any space needed for thread-local storage has been 
    //  subtracted (see https://sourceware.org/bugzilla/show_bug.cgi?id=11787)
    // glibc won't tell you how much to add, but anecdotally, 128KB seems to
    //  be enough, so we'll do the greater of 256KB and twice
    //  PTHREAD_STACK_MIN and hope we never have to debug this again...
    const ptrdiff_t MIN_STACK_SIZE = ((PTHREAD_STACK_MIN > (128 << 10)) ?
				        (PTHREAD_STACK_MIN * 2) :
				        (256 << 10));

    ptrdiff_t stack_size = 0;  // 0 == "pthread default"

    if(params.stack_size != params.STACK_SIZE_DEFAULT) {
      // make sure it's not too large
      assert((rsrv.params.max_stack_size == rsrv.params.STACK_SIZE_DEFAULT) ||
	     (params.stack_size <= rsrv.params.max_stack_size));

      stack_size = std::max<ptrdiff_t>(params.stack_size, MIN_STACK_SIZE);
    } else {
      // does the entire core reservation have a non-standard stack size?
      if(rsrv.params.max_stack_size != rsrv.params.STACK_SIZE_DEFAULT) {
	stack_size = std::max<ptrdiff_t>(rsrv.params.max_stack_size,
					 MIN_STACK_SIZE);
      }
    }
    if(stack_size > 0)
      CHECK_PTHREAD( pthread_attr_setstacksize(&attr, stack_size) );

    // TODO: actually use heap size

    update_state(STATE_STARTUP);

    // time to actually create the thread
    CHECK_PTHREAD( pthread_create(&thread, &attr, pthread_entry, this) );

    CHECK_PTHREAD( pthread_attr_destroy(&attr) );

    log_thread.info() << "thread created:" << this << " (" << rsrv.name << ") - pthread " << std::hex << thread << std::dec;
    log_thread.debug() << "thread stack: " << this << " size=" << stack_size;
  }

  void KernelThread::join(void)
  {
    CHECK_PTHREAD( pthread_join(thread, 0 /* ignore retval */) );
    ok_to_delete = true;
  }

  void KernelThread::detach(void)
  {
    CHECK_PTHREAD( pthread_detach(thread) );
    ok_to_delete = true;
  }

  void KernelThread::alert_thread(void)
  {
    // are we alerting ourself?
    if(this->thread == pthread_self()) {
      // just process the signals right here and now
      process_signals();
    } else {
      pthread_kill(this->thread, handler_signal);
    }
  }

  // used when we don't have an allocation yet
  template <typename T>
  class DeferredThreadStart : public CoreReservation::NotificationListener {
  public:
    DeferredThreadStart(T *_thread,
                        const ThreadLaunchParameters& _params);

    virtual ~DeferredThreadStart(void);

    virtual void notify_allocation(const CoreReservation& rsrv);

  protected:
    T *thread;
    ThreadLaunchParameters params;
  };

  template <typename T>
  DeferredThreadStart<T>::DeferredThreadStart(T *_thread,
                                              const ThreadLaunchParameters& _params)
    : thread(_thread), params(_params)
  {
  }

  template <typename T>
  DeferredThreadStart<T>::~DeferredThreadStart(void)
  {
  }

  template <typename T>
  void DeferredThreadStart<T>::notify_allocation(const CoreReservation& rsrv)
  {
    // thread is allowed to start now
    thread->start_thread(params, rsrv);
    delete this;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UserThread

#ifdef REALM_USE_USER_THREADS
  namespace {
    int uswitch_test_check_flag = 1;
    ucontext_t uswitch_test_ctx1, uswitch_test_ctx2;

    void uswitch_test_entry(int arg)
    {
      log_thread.debug() << "uswitch test: adding: " << uswitch_test_check_flag << " " << arg;
      __sync_fetch_and_add(&uswitch_test_check_flag, arg);
      errno = 0;
      int ret = swapcontext(&uswitch_test_ctx2, &uswitch_test_ctx1);
      if(ret != 0) {
	log_thread.fatal() << "uswitch test: swap out failed: " << ret << " " << errno;
	assert(0);
      }
    }
  }

  // some systems do not appear to support user thread switching for
  //  reasons unknown, so allow code to test to see if it's working first
  /*static*/ bool Thread::test_user_switch_support(size_t stack_size /*= 1 << 20*/)
  {
    errno = 0;
    int ret;
    ret = getcontext(&uswitch_test_ctx2);
    if(ret != 0) {
      log_thread.info() << "uswitch test: getcontext failed: " << ret << " " << errno;
      return false;
    }
    void *stack_base = malloc(stack_size);
    if(!stack_base) {
      log_thread.info() << "uswitch test: stack malloc failed";
      return false;
    }
    uswitch_test_ctx2.uc_link = 0; // we don't expect it to ever fall through
    uswitch_test_ctx2.uc_stack.ss_sp = stack_base;
    uswitch_test_ctx2.uc_stack.ss_size = stack_size;
    uswitch_test_ctx2.uc_stack.ss_flags = 0;
    makecontext(&uswitch_test_ctx2,
		reinterpret_cast<void(*)()>(uswitch_test_entry),
		1, 66);

    // now try to swap and back
    errno = 0;
    ret = swapcontext(&uswitch_test_ctx1, &uswitch_test_ctx2);
    if(ret != 0) {
      log_thread.info() << "uswitch test: swap in failed: " << ret << " " << errno;
      free(stack_base);
      return false;
    }

    int val = __sync_fetch_and_add(&uswitch_test_check_flag, 0);
    if(val != 67) {
      log_thread.info() << "uswitch test: val mismatch: " << val << " != 67";
      free(stack_base);
      return false;
    }

    log_thread.debug() << "uswitch test: check succeeded";
    free(stack_base);
    return true;
  }

  class UserThread : public Thread {
  public:
    UserThread(void *_target, void (*_entry_wrapper)(void *),
	       ThreadScheduler *_scheduler);

    virtual ~UserThread(void);

    void start_thread(const ThreadLaunchParameters& params,
		      const CoreReservation *rsrv);

    virtual void join(void);
    virtual void detach(void);

    static void user_switch(UserThread *switch_to);

  protected:
    static void uthread_entry(void) __attribute__((noreturn));

    virtual void alert_thread(void);

    static const int MAGIC_VALUE = 0x11223344;

    void *target;
    void (*entry_wrapper)(void *);
    int magic;
#ifndef __MACH__
    ucontext_t ctx;
#else
    // valgrind says Darwin's getcontext is writing past the end of ctx?
    ucontext_t ctx;
    int padding[512];
#endif
    void *stack_base;
    size_t stack_size;
    bool ok_to_delete;
    bool running;
    pthread_t host_pthread;
  };

  UserThread::UserThread(void *_target, void (*_entry_wrapper)(void *),
			 ThreadScheduler *_scheduler)
    : Thread(_scheduler), target(_target), entry_wrapper(_entry_wrapper)
    , magic(MAGIC_VALUE), stack_base(0), stack_size(0), ok_to_delete(false)
    , running(false)
  {
  }

  UserThread::~UserThread(void)
  {
    // cannot delete an active thread...
    assert(!running);

    if(stack_base != 0)
      free(stack_base);
  }

  namespace ThreadLocal {
    __thread ucontext_t *host_context = 0;
    // current_user_thread is redundant with current_thread, but kept for debugging
    //  purposes for now
    __thread UserThread *current_user_thread = 0;
    __thread Thread *current_host_thread = 0;
  };

  /*static*/ void UserThread::uthread_entry(void)
  {
    UserThread *thread = ThreadLocal::current_user_thread;
    assert(thread != 0);

    thread->host_pthread = pthread_self();
    thread->running = true;

    log_thread.info() << "thread " << thread << " started";
    thread->update_state(STATE_RUNNING);

    if(thread->scheduler)
      thread->scheduler->thread_starting(thread);
    
    // call the actual thread body
    (*thread->entry_wrapper)(thread->target);

    if(thread->scheduler)
      thread->scheduler->thread_terminating(thread);
    
    // on return, we update our status and terminate
    log_thread.info() << "thread " << thread << " finished";
    thread->update_state(STATE_FINISHED);

    // returning from this call is lethal, so hand control back to the host
    //  thread and hope for the best
    while(true) {
      user_switch(0);
      log_thread.warning() << "HELP!  switched to a terminated thread " << thread;
    }
  }

  void UserThread::start_thread(const ThreadLaunchParameters& params,
				const CoreReservation *rsrv)
  {
    // it turns out MacOS behaves REALLY strangely with a stack < 32KB, and there
    //  make be some lower limit in Linux-land too, so clamp to 64KB to be safe
    const ptrdiff_t MIN_STACK_SIZE = 64 << 10;

    if(params.stack_size != params.STACK_SIZE_DEFAULT) {
      // make sure it's not too large
      if(rsrv)
	assert((rsrv->params.max_stack_size == rsrv->params.STACK_SIZE_DEFAULT) ||
	       (params.stack_size <= rsrv->params.max_stack_size));

      stack_size = std::max<ptrdiff_t>(params.stack_size, MIN_STACK_SIZE);
    } else {
      // does the entire core reservation have a non-standard stack size?
      if(rsrv &&
	 (rsrv->params.max_stack_size != rsrv->params.STACK_SIZE_DEFAULT)) {
	stack_size = std::max<ptrdiff_t>(rsrv->params.max_stack_size,
					 MIN_STACK_SIZE);
      }
    }

    stack_base = malloc(stack_size);
    assert(stack_base != 0);

    CHECK_LIBC( getcontext(&ctx) );

    ctx.uc_link = 0; // we don't expect it to ever fall through
    ctx.uc_stack.ss_sp = stack_base;
    ctx.uc_stack.ss_size = stack_size;
    ctx.uc_stack.ss_flags = 0;

    // grr...  entry point takes int's, which might not hold a void *
    // we'll just fish our UserThread * out of TLS
    makecontext(&ctx, uthread_entry, 0);

    update_state(STATE_STARTUP);    

    log_thread.info() << "thread created:" << this << " (" << (rsrv ? rsrv->name : "??") << ") - user thread";
    log_thread.debug() << "thread stack: " << this << " size=" << stack_size << " base=" << stack_base;
  }

  void UserThread::join(void)
  {
    assert(0); // not supported yet
  }

  void UserThread::detach(void)
  {
    assert(0); // not supported yet
  }

  /*static*/ void UserThread::user_switch(UserThread *switch_to)
  {
#ifdef DEBUG_USWITCH
    printf("uswitch: %p: %p -> %p\n",
	   ThreadLocal::current_host_thread ? ThreadLocal::current_host_thread : ThreadLocal::current_thread,
	   ThreadLocal::current_user_thread,
	   switch_to);
#endif

    if(ThreadLocal::current_user_thread == 0) {
      // called from a kernel thread, which will be used as the host

      assert(switch_to != 0);
      assert(switch_to->magic == MAGIC_VALUE);
      assert(ThreadLocal::host_context == 0);

      // this holds the host's state
      ucontext_t host_ctx;

      ThreadLocal::host_context = &host_ctx;
      ThreadLocal::current_user_thread = switch_to;
      ThreadLocal::current_host_thread = ThreadLocal::current_thread;
      ThreadLocal::current_thread = switch_to;

      CHECK_LIBC( swapcontext(&host_ctx, &switch_to->ctx) );

      assert(ThreadLocal::current_user_thread == 0);
      assert(ThreadLocal::host_context == &host_ctx);
      ThreadLocal::host_context = 0;
    } else {
      UserThread *switch_from = ThreadLocal::current_user_thread;
      ThreadLocal::current_user_thread = switch_to;

      assert(switch_from->running == true);
      switch_from->running = false;

      if(switch_to != 0) {
	assert(switch_to->magic == MAGIC_VALUE);
	assert(switch_to->running == false);

	ThreadLocal::current_thread = switch_to;

	// a switch between two user contexts - nice and simple
	CHECK_LIBC( swapcontext(&switch_from->ctx, &switch_to->ctx) );

	assert(switch_from->running == false);
	switch_from->host_pthread = pthread_self();
	switch_from->running = true;
      } else {
	// a return of control to the host thread
	assert(ThreadLocal::host_context != 0);

	ThreadLocal::current_thread = ThreadLocal::current_host_thread;
	ThreadLocal::current_host_thread = 0;

	CHECK_LIBC( swapcontext(&switch_from->ctx, ThreadLocal::host_context) );

	// if we get control back
	assert(switch_from->running == false);
	switch_from->host_pthread = pthread_self();
	switch_from->running = true;
      }
    }
  }

  void UserThread::alert_thread(void)
  {
    if(ThreadLocal::current_thread == this) {
      // just process the signals right here and now
      process_signals();
    } else {
      // TODO: work out the race conditions inherent in this process
      if(running) {
	pthread_kill(host_pthread, handler_signal);
      } else {
        assert(scheduler != 0);
	if(try_update_state(STATE_BLOCKED, STATE_ALERTED)) {
          scheduler->thread_ready(this);
        } else {
	  log_thread.fatal()  << "HELP! couldn't alert: thread=" << this << " state=" << get_state();
	  assert(0);
	}
      }
    }
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class Thread

  /*static*/ Thread *Thread::create_kernel_thread_untyped(void *target, void (*entry_wrapper)(void *),
							  const ThreadLaunchParameters& params,
							  CoreReservation& rsrv,
							  ThreadScheduler *_scheduler)
  {
    KernelThread *t = new KernelThread(target, entry_wrapper, _scheduler);

    // if we have an allocation, we can start the thread immediately
    if(rsrv.allocation) {
      t->start_thread(params, rsrv);
    } else {
      rsrv.add_listener(new DeferredThreadStart<KernelThread>(t, params));
    }

    return t;
  }

  /*static*/ void Thread::yield(void)
  {
#ifdef __MACH__
    sched_yield();
#else
    pthread_yield();
#endif
  }

#ifdef REALM_USE_USER_THREADS
  /*static*/ Thread *Thread::create_user_thread_untyped(void *target, void (*entry_wrapper)(void *),
							const ThreadLaunchParameters& params,
							const CoreReservation *rsrv,
							ThreadScheduler *_scheduler)
  {
    UserThread *t = new UserThread(target, entry_wrapper, _scheduler);

    // no need to wait on an allocation - the host thread will take care of that
    t->start_thread(params, rsrv);

    return t;
  }

  /*static*/ void Thread::user_switch(Thread *switch_to)
  {
    // just cast 'switch_to' to a UserThread - UserThread::user_switch will do a bit
    //   of sanity-checking
    UserThread::user_switch((UserThread *)switch_to);
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreMap

  CoreMap::CoreMap(void)
  {
  }

  CoreMap::~CoreMap(void)
  {
    clear();
  }

  void CoreMap::clear(void)
  {
    // delete all the processor entries
    for(ProcMap::iterator it = all_procs.begin();
	it != all_procs.end();
	it++)
      delete it->second;

    // now clear out the two maps
    all_procs.clear();
    by_domain.clear();
  }

  /*static*/ CoreMap *CoreMap::create_synthetic(int num_domains,
						int cores_per_domain,
						int hyperthreads /*= 1*/,
						int fp_cluster_size /*= 1*/)
  {
    CoreMap *cm = new CoreMap;

    // processor ids will just be monotonically increasing
    int next_id = 0;

    for(int d = 0; d < num_domains; d++) {
      for(int c = 0; c < cores_per_domain; c++) {
	std::set<Proc *> fp_procs;

	for(int f = 0; f < fp_cluster_size; f++) {
	  std::set<Proc *> ht_procs;

	  for(int h = 0; h < hyperthreads; h++) {
	    int id = next_id++;

	    Proc *p = new Proc;

	    p->id = id;
	    p->domain = d;
	    // kernel proc id list is empty - this is synthetic

	    cm->all_procs[id] = p;
	    cm->by_domain[d][id] = p;

	    ht_procs.insert(p);
	    fp_procs.insert(p);
	  }

	  // ALU and LD/ST shared with all other hyperthreads
	  for(std::set<Proc *>::iterator it1 = ht_procs.begin(); it1 != ht_procs.end(); it1++)
	    for(std::set<Proc *>::iterator it2 = ht_procs.begin(); it2 != ht_procs.end(); it2++)
	      if(it1 != it2) {
		(*it1)->shares_alu.insert(*it2);
		(*it1)->shares_ldst.insert(*it2);
	      }
	}

	// FPU shared with all other procs in cluster
	for(std::set<Proc *>::iterator it1 = fp_procs.begin(); it1 != fp_procs.end(); it1++)
	  for(std::set<Proc *>::iterator it2 = fp_procs.begin(); it2 != fp_procs.end(); it2++)
	    if(it1 != it2) {
	      (*it1)->shares_fpu.insert(*it2);
	    }
      }
    }

    return cm;
  }

  // this function is templated on the map key, since we don't really care 
  //  what was used to make the equivalence classes
  template <typename K>
  void update_core_sharing(const std::map<K, std::set<CoreMap::Proc *> >& core_sets,
			   bool share_alu, bool share_fpu, bool share_ldst)
  {
    for(typename std::map<K, std::set<CoreMap::Proc *> >::const_iterator it = core_sets.begin();
        it != core_sets.end();
        it++) {
      const std::set<CoreMap::Proc *>& cset = it->second;
      if(cset.size() == 1) continue;  // singleton set - no sharing

      // all pairs dependencies
      for(std::set<CoreMap::Proc *>::const_iterator it1 = cset.begin(); it1 != cset.end(); it1++) {
        for(std::set<CoreMap::Proc *>::const_iterator it2 = cset.begin(); it2 != cset.end(); it2++) {
          if(it1 != it2) {
            CoreMap::Proc *p = *it1;
            if(share_alu)
	      p->shares_alu.insert(*it2);
	    if(share_fpu)
	      p->shares_fpu.insert(*it2);
	    if(share_ldst)
	      p->shares_ldst.insert(*it2);
          }
        }
      }
    }
  }

#ifdef __linux__
  static CoreMap *extract_core_map_from_linux_sys(bool hyperthread_sharing)
  {
    cpu_set_t cset;
    int ret = sched_getaffinity(0, sizeof(cset), &cset);
    if(ret < 0) {
      log_thread.warning() << "failed to get affinity info";
      return 0;
    }

    DIR *nd = opendir("/sys/devices/system/node");
    if(!nd) {
      log_thread.warning() << "can't open /sys/devices/system/node";
      return 0;
    }

    CoreMap *cm = new CoreMap;
    // hyperthreading sets are cores with the same node ID and physical core ID
    //  they share ALU, FPU, and LDST units
    std::map<std::pair<int, int>, std::set<CoreMap::Proc *> > ht_sets;

    // "thread_siblings" can be used to detect Bulldozer's core pairs that 
    //  share the same FPU (this will also catch hyperthreads, but that's ok)
    std::map<std::string, std::set<CoreMap::Proc *> > sibling_sets;

    // look for entries named /sys/devices/system/node/node<N>
    for(struct dirent *ne = readdir(nd); ne; ne = readdir(nd)) {
      if(strncmp(ne->d_name, "node", 4)) continue;  // not a node directory
      char *pos;
      int node_id = strtol(ne->d_name + 4, &pos, 10);
      if(pos && *pos) continue;  // doesn't match node[0-9]+
	  
      char per_node_path[1024];
      sprintf(per_node_path, "/sys/devices/system/node/%s", ne->d_name);
      DIR *cd = opendir(per_node_path);
      if(!cd) {
	log_thread.warning() << "can't open '" << per_node_path << "' - skipping";
	continue;
      }

      // look for entries named /sys/devices/system/node/node<N>/cpu<N>
      for(struct dirent *ce = readdir(cd); ce; ce = readdir(cd)) {
	if(strncmp(ce->d_name, "cpu", 3)) continue; // definitely not a cpu
	char *pos;
	int cpu_id = strtol(ce->d_name + 3, &pos, 10);
	if(pos && *pos) continue;  // doesn't match cpu[0-9]+
	    
	// is this a cpu we're allowed to use?
	if(!CPU_ISSET(cpu_id, &cset)) {
	  log_thread.info() << "cpu " << cpu_id << " not available - skipping";
	  continue;
	}

	// figure out which physical core it is (i.e. detect hyperthreads)
	char core_id_path[1024];
	sprintf(core_id_path, "/sys/devices/system/node/%s/%s/topology/core_id", ne->d_name, ce->d_name);
	FILE *f = fopen(core_id_path, "r");
	if(!f) {
	  log_thread.warning() << "can't read '" << core_id_path << "' - skipping";
	  continue;
	}
	int core_id;
	int count = fscanf(f, "%d", &core_id);
	fclose(f);
	if(count != 1) {
	  log_thread.warning() << "can't find core id in '" << core_id_path << "' - skipping";
	  continue;
	}

	CoreMap::Proc *p = new CoreMap::Proc;

	p->id = cpu_id;
	p->domain = node_id;
	p->kernel_proc_ids.insert(cpu_id);

	cm->all_procs[cpu_id] = p;
	cm->by_domain[node_id][cpu_id] = p;

	// add to HT sets to deal with in a bit
	ht_sets[std::make_pair(node_id, core_id)].insert(p);

	// read the sibling set, if we can - no need to parse it because we
	//  expect symmetry across all cores in the same set
	{
	  char sibling_path[1024];
	  sprintf(sibling_path, "/sys/devices/system/node/%s/%s/topology/thread_siblings_list", ne->d_name, ce->d_name);
	  FILE *f = fopen(sibling_path, "r");
	  if(f) {
	    char line[256];
	    if(fgets(line, 255, f)) {
	      if(*line)
		sibling_sets[line].insert(p);
	    } else
	      log_thread.warning() << "error reading '" << sibling_path << "' - no contents?";
	    fclose(f);
	  } else
	    log_thread.warning() << "can't read '" << sibling_path << "' - skipping";
	}
      }
      closedir(cd);
    }
    closedir(nd);

    if(hyperthread_sharing) {
      update_core_sharing(ht_sets, true /*alu*/, true /*fpu*/, true /*ldst*/);
      update_core_sharing(sibling_sets,
			  false /*!alu*/, true /*fpu*/, false /*!ldst*/);
    }

    // all done!
    return cm;
  }
#endif

#ifdef REALM_USE_HWLOC
#ifdef __linux__
  // find bulldozer cpus that share fpu
  static bool get_bd_sibling_id(int cpu_id, int core_id,
				std::set<int>& sibling_ids) {
    char str[1024];
    sprintf(str, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings", cpu_id);
    FILE *f = fopen(str, "r");
    if(!f) {
      log_thread.warning() << "can't read '" << str << "' - skipping";
      return false;
    }
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    hwloc_linux_parse_cpumap_file(f, set);

    fclose(f);

    // loop over all siblings (except ourselves)
    for(int siblingid = hwloc_bitmap_first(set);
	siblingid != -1;
	siblingid = hwloc_bitmap_next(set, siblingid)) {
      if(siblingid == cpu_id) continue;

      // don't filter siblings with the same core ID - this catches
      //  hyperthreads too
#if 0
      sprintf(str, "/sys/devices/system/cpu/cpu%d/topology/core_id", siblingid);
      f = fopen(str, "r");
      if(!f) {
	log_thread.warning() << "can't read '" << str << "' - skipping";
	continue;
      }
      int sib_core_id;
      int count = fscanf(f, "%d", &sib_core_id);
      fclose(f);
      if(count != 1) {
	log_thread.warning() << "can't find core id in '" << str << "' - skipping";
	continue;
      }
      if(sib_core_id == core_id) continue;
#endif
      sibling_ids.insert(siblingid);
    }

    hwloc_bitmap_free(set);

    return true;
  }
#endif

  static CoreMap *extract_core_map_from_hwloc(bool hyperthread_sharing)
  {
    CoreMap *cm = new CoreMap;

    // hyperthreading sets are cores with the same node ID and physical core ID
    std::map<std::pair<int, int>, std::set<CoreMap::Proc *> > ht_sets;
    // bulldozer specific sets
    std::map<int, std::set<CoreMap::Proc *> > bd_sets;

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    hwloc_obj_t obj = NULL;
    int cpu_id, node_id, core_id;
    while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_CORE, obj)) != NULL) {
      if(obj->online_cpuset && obj->allowed_cpuset) {
        cpu_id = hwloc_bitmap_first(obj->cpuset);
        while(cpu_id != -1) {

          node_id = hwloc_bitmap_first(obj->nodeset);
          core_id = obj->os_index;

          CoreMap::Proc *p = new CoreMap::Proc;

          p->id = cpu_id;
          p->domain = node_id;
          p->kernel_proc_ids.insert(cpu_id);

          cm->all_procs[cpu_id] = p;
          cm->by_domain[node_id][cpu_id] = p;

          // add to HT sets to deal with in a bit
          ht_sets[std::make_pair(node_id, core_id)].insert(p);

#ifdef __linux__
          // add bulldozer sets
	  std::set<int> sibling_ids;
	  if(get_bd_sibling_id(cpu_id, core_id, sibling_ids) &&
	     !sibling_ids.empty()) {
	    bd_sets[cpu_id].insert(p);
	    for(std::set<int>::const_iterator it = sibling_ids.begin();
		it != sibling_ids.end();
		++it)
	      bd_sets[*it].insert(p);
	  }
#endif

          cpu_id = hwloc_bitmap_next(obj->cpuset, cpu_id);
        }
      }
    }
    hwloc_topology_destroy(topology);

    if(hyperthread_sharing) {
      update_core_sharing(ht_sets, true /*alu*/, true /*fpu*/, true /*ldst*/);
      update_core_sharing(bd_sets,
			  false /*!alu*/, true /*fpu*/, false /*!ldst*/);
    }

    // all done!
    return cm;
  }
#endif

  /*static*/ CoreMap *CoreMap::discover_core_map(bool hyperthread_sharing)
  {
    // we'll try a number of different strategies to discover the local cores:
    // 1) a user-defined synthetic map, if REALM_SYNTHETIC_CORE_MAP is set
    if(getenv("REALM_SYNTHETIC_CORE_MAP")) {
      const char *p = getenv("REALM_SYNTHETIC_CORE_MAP");
      int num_domains = 1;
      int num_cores = 1;
      int hyperthreads = 1;
      int fp_cluster_size = 1;
      while(true) {
	if(!(p[0] && (p[1] == '=') && isdigit(p[2]))) break;

	const char *p2;
	int x = strtol(p+2, (char **)&p2, 10);
	if(x == 0) { p+=2; break; }  // zero of anything is bad
	if(p[0] == 'd') num_domains = x; else
	if(p[0] == 'c') num_cores = x; else
	if(p[0] == 'h') hyperthreads = x; else
	if(p[0] == 'f') fp_cluster_size = x; else
	  break;
	p = p2;

	// now we want a comma (to continue) or end-of-string
	if(*p != ',') break;
	p++;
      }
      // if parsing reached the end of string, we're good
      if(*p == 0) {
	return CoreMap::create_synthetic(num_domains, num_cores, hyperthreads, fp_cluster_size);
      } else {
	const char *orig = getenv("REALM_SYNTHETIC_CORE_MAP");
	log_thread.error("Error parsing REALM_SYNTHETIC_CORE_MAP: '%.*s(^)%s'",
			 (int)(p-orig), orig, p);
      }
    }

    // 2) extracted from hwloc information
#ifdef REALM_USE_HWLOC
    {
      CoreMap *cm = extract_core_map_from_hwloc(hyperthread_sharing);
      if(cm) return cm;
    }
#endif

    // 3) extracted from Linux's /sys
#ifdef __linux__
    {
      CoreMap *cm = extract_core_map_from_linux_sys(hyperthread_sharing);
      if(cm) return cm;
    }
#endif

    // 4) as a final fallback a single-core synthetic map
    {
      CoreMap *cm = create_synthetic(1, 1);
      return cm;
    }
  }
  
  static void show_share_set(std::ostream& os, const char *name,
			     const std::set<CoreMap::Proc *>& sset)
  {
    if(sset.empty()) return;

    os << ' ' << name << "=<";
    std::set<CoreMap::Proc *>::const_iterator it = sset.begin();
    while(true) {
      os << (*it)->id;
      if(++it == sset.end()) break;
      os << ',';
    }
    os << ">";
  }

  /*friend*/ std::ostream& operator<<(std::ostream& os, const CoreMap& cm)
  {
    os << "core map {" << std::endl;
    for(CoreMap::DomainMap::const_iterator it = cm.by_domain.begin();
	it != cm.by_domain.end();
	it++) {
      os << "  domain " << it->first << " {" << std::endl;
      for(CoreMap::ProcMap::const_iterator it2 = it->second.begin();
	  it2 != it->second.end();
	  it2++) {
	os << "    core " << it2->first << " {";
	const CoreMap::Proc *p = it2->second;
	if(!p->kernel_proc_ids.empty()) {
	  os << " ids=<";
	  std::set<int>::const_iterator it3 = p->kernel_proc_ids.begin();
	  while(true) {
	    os << *it3;
	    if(++it3 == p->kernel_proc_ids.end()) break;
	    os << ',';
	  }
	  os << ">";
	}

	show_share_set(os, "alu", p->shares_alu);
	show_share_set(os, "fpu", p->shares_fpu);
	show_share_set(os, "ldst", p->shares_ldst);

	os << " }" << std::endl;
      }
      os << "  }" << std::endl;
    }
    os << "}";
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PAPICounters


#ifdef REALM_USE_PAPI
  PAPICounters::PAPICounters(void)
    : papi_event_set(PAPI_NULL)
  {}

  PAPICounters::~PAPICounters(void)
  {
    if(papi_event_set != PAPI_NULL) {
      int ret = PAPI_cleanup_eventset(papi_event_set);
      assert(ret == PAPI_OK);
      int orig_event_set = papi_event_set;
      ret = PAPI_destroy_eventset(&papi_event_set);
      log_papi.debug() << "destroy_eventset: " << orig_event_set << " (" << ret << ")";
      assert(ret == PAPI_OK);
      assert(papi_event_set == PAPI_NULL);
    }
  }

  /*static*/ PAPICounters *PAPICounters::setup_counters(const ProfilingMeasurementCollection& pmc)
  {
    // if we didn't successfully initialize PAPI, don't try to use it...
    if(!PAPI::papi_available)
      return 0;

    // first, check for all the things we know how to translate into PAPI events
    std::vector<int> desired_events;

    if(pmc.wants_measurement<ProfilingMeasurements::IPCPerfCounters>()) {
      desired_events.push_back(PAPI_TOT_INS);
      desired_events.push_back(PAPI_TOT_CYC);
      desired_events.push_back(PAPI_FP_INS);
      desired_events.push_back(PAPI_LD_INS);
      desired_events.push_back(PAPI_SR_INS);
      desired_events.push_back(PAPI_BR_INS);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L1ICachePerfCounters>()) {
      desired_events.push_back(PAPI_L1_ICA);
      desired_events.push_back(PAPI_L1_ICM);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L1DCachePerfCounters>()) {
      desired_events.push_back(PAPI_L1_DCA);
      desired_events.push_back(PAPI_L1_DCM);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L2CachePerfCounters>()) {
      desired_events.push_back(PAPI_L2_TCA);
      desired_events.push_back(PAPI_L2_TCM);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L3CachePerfCounters>()) {
      desired_events.push_back(PAPI_L3_TCA);
      desired_events.push_back(PAPI_L3_TCM);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::TLBPerfCounters>()) {
      desired_events.push_back(PAPI_TLB_IM);
      desired_events.push_back(PAPI_TLB_DM);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::BranchPredictionPerfCounters>()) {
      desired_events.push_back(PAPI_BR_CN);
      desired_events.push_back(PAPI_BR_TKN);
      desired_events.push_back(PAPI_BR_MSP);
    }

    // exit early if none present
    if(desired_events.empty()) return 0;

    // otherwise create an event set and add as many of them as we can
    PAPICounters *ctrs = new PAPICounters;

    {
      int ret = PAPI_create_eventset(&(ctrs->papi_event_set));
      log_papi.debug() << "create_eventset: " << ctrs->papi_event_set << " (" << ret << ")";
      assert(ret == PAPI_OK);
    }

    size_t count = 0;
    for(std::vector<int>::const_iterator it = desired_events.begin();
	it != desired_events.end();
	++it) {
      // event might already have been added?
      if(ctrs->event_codes.count(*it) > 0)
	continue;

      int ret = PAPI_add_event(ctrs->papi_event_set, *it);
      if(ret == PAPI_OK) {
	ctrs->event_codes[*it] = count;
	ctrs->event_counts.push_back(0);
	count++;
	continue;
      }
      // two kinds of tolerable error
      if(ret == PAPI_ENOEVNT) {
	log_papi.debug() << "event " << *it << " not available on hardware - skipping";
	continue;
      }
      if(ret == PAPI_ECNFLCT) {
	log_papi.debug() << "event " << *it << " conflicts with previously added events - skipping";
	continue;
      }
      // anything else is a real problem
      log_papi.fatal() << "add event: " << PAPI_strerror(ret) << " (" << ret << ")";
      assert(false);
    }

    return ctrs;
  }

  void PAPICounters::cleanup(void)
  {
    // TODO: try to reuse these things?
    delete this;
  }

  void PAPICounters::start(void)
  {
    int ret = PAPI_start(papi_event_set);
    log_papi.debug() << "start counters: " << papi_event_set << " (" << ret << ")";
    assert(ret == PAPI_OK);
  }

  void PAPICounters::stop(void)
  {
    int ret = PAPI_accum(papi_event_set, &event_counts[0]);
    assert(ret == PAPI_OK);
    ret = PAPI_stop(papi_event_set, 0 /* don't read values again */);
    log_papi.debug() << "stop counters: " << papi_event_set << " (" << ret << ")";
    assert(ret == PAPI_OK);
  }

  void PAPICounters::resume(void)
  {
    // same as start for now
    start();
  }

  void PAPICounters::suspend(void)
  {
    // same as stop for now
    stop();
  }

  // little helper to get a counter if present, or -1 if not
  static inline long long get_counter_val(int code,
					  const std::map<int, size_t>& event_codes,
					  const std::vector<long long>& event_counts,
					  int& found_count)
  {
    std::map<int, size_t>::const_iterator it = event_codes.find(code);
    if(it != event_codes.end()) {
      found_count++;
      return event_counts[it->second];
    } else
      return -1;
  }

  void PAPICounters::record(ProfilingMeasurementCollection& pmc)
  {
    if(pmc.wants_measurement<ProfilingMeasurements::IPCPerfCounters>()) {
      ProfilingMeasurements::IPCPerfCounters ctrs;
      int found_count = 0;
      ctrs.total_insts  = get_counter_val(PAPI_TOT_INS, event_codes, event_counts, found_count);
      ctrs.total_cycles = get_counter_val(PAPI_TOT_CYC, event_codes, event_counts, found_count);
      ctrs.fp_insts     = get_counter_val(PAPI_FP_INS , event_codes, event_counts, found_count);
      ctrs.ld_insts     = get_counter_val(PAPI_LD_INS , event_codes, event_counts, found_count);
      ctrs.st_insts     = get_counter_val(PAPI_SR_INS , event_codes, event_counts, found_count);
      ctrs.br_insts     = get_counter_val(PAPI_BR_INS , event_codes, event_counts, found_count);
      if(found_count > 0)
	pmc.add_measurement(ctrs);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L1ICachePerfCounters>()) {
      ProfilingMeasurements::L1ICachePerfCounters ctrs;
      int found_count = 0;
      ctrs.accesses = get_counter_val(PAPI_L1_ICA, event_codes, event_counts, found_count);
      ctrs.misses   = get_counter_val(PAPI_L1_ICM, event_codes, event_counts, found_count);
      if(found_count > 0)
	pmc.add_measurement(ctrs);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L1DCachePerfCounters>()) {
      ProfilingMeasurements::L1DCachePerfCounters ctrs;
      int found_count = 0;
      ctrs.accesses = get_counter_val(PAPI_L1_DCA, event_codes, event_counts, found_count);
      ctrs.misses   = get_counter_val(PAPI_L1_DCM, event_codes, event_counts, found_count);
      if(found_count > 0)
	pmc.add_measurement(ctrs);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L2CachePerfCounters>()) {
      ProfilingMeasurements::L2CachePerfCounters ctrs;
      int found_count = 0;
      ctrs.accesses = get_counter_val(PAPI_L2_TCA, event_codes, event_counts, found_count);
      ctrs.misses   = get_counter_val(PAPI_L2_TCM, event_codes, event_counts, found_count);
      if(found_count > 0)
	pmc.add_measurement(ctrs);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::L3CachePerfCounters>()) {
      ProfilingMeasurements::L3CachePerfCounters ctrs;
      int found_count = 0;
      ctrs.accesses = get_counter_val(PAPI_L3_TCA, event_codes, event_counts, found_count);
      ctrs.misses   = get_counter_val(PAPI_L3_TCM, event_codes, event_counts, found_count);
      if(found_count > 0)
	pmc.add_measurement(ctrs);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::TLBPerfCounters>()) {
      ProfilingMeasurements::TLBPerfCounters ctrs;
      int found_count = 0;
      ctrs.inst_misses = get_counter_val(PAPI_TLB_IM, event_codes, event_counts, found_count);
      ctrs.data_misses = get_counter_val(PAPI_TLB_DM, event_codes, event_counts, found_count);
      if(found_count > 0)
	pmc.add_measurement(ctrs);
    }
    if(pmc.wants_measurement<ProfilingMeasurements::BranchPredictionPerfCounters>()) {
      ProfilingMeasurements::BranchPredictionPerfCounters ctrs;
      int found_count = 0;
      ctrs.total_branches = get_counter_val(PAPI_BR_CN , event_codes, event_counts, found_count);
      ctrs.taken_branches = get_counter_val(PAPI_BR_TKN, event_codes, event_counts, found_count);
      ctrs.mispredictions = get_counter_val(PAPI_BR_MSP, event_codes, event_counts, found_count);
      if(found_count > 0)
	pmc.add_measurement(ctrs);
    }

#ifdef REALM_PAPI_DEBUG
    for(std::map<int, size_t>::const_iterator it = event_codes.begin();
	it != event_codes.end();
	++it) {
      log_papi.error() << "counter[" << (it->first & 0x7fffffff) << "] = " << event_counts[it->second];
    }
#endif
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // initialize/cleanup

  namespace Threading {

    bool initialize(void)
    {
#ifdef REALM_USE_PAPI
      {
	int ret = PAPI_library_init(PAPI_VER_CURRENT);
	if(ret == PAPI_VER_CURRENT) {
	  // initialized - now tell it we have threads
	  ret = PAPI_thread_init(pthread_self);
	  if(ret == PAPI_OK) {
	    PAPI::papi_available = true;
	    int numctrs = PAPI_get_opt(PAPI_MAX_HWCTRS, 0);
	    log_papi.debug() << "initalized successfully - " << numctrs << " counters";
	  } else {
	    log_papi.warning() << "thread init error: " << PAPI_strerror(ret) << " (" << ret << ")";
	  }
	} else {
	  // failure could be due to a version mismatch or some other error
	  if(ret > 0) {
	    log_papi.warning() << "version mismatch - wanted: " << PAPI_VER_CURRENT << ", got: " << ret;
	  } else {
	    log_papi.warning() << "initialization error: " << PAPI_strerror(ret) << " (" << ret << ")";
	  }
	}
      }
#endif
      return true;
    }

    bool cleanup(void)
    {
#ifdef REALM_USE_PAPI
      if(PAPI::papi_available) {
	PAPI_shutdown();
      }
#endif
      return true;
    }

  };


}; // namespace Realm
