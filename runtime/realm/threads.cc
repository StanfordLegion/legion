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

// generic Realm interface to threading libraries (e.g. pthreads)

#include "threads.h"

#include "logging.h"

#ifdef DEBUG_USWITCH
#include <stdio.h>
#endif

#include <pthread.h>
// for PTHREAD_STACK_MIN
#include <limits.h>
#ifdef __MACH__
// for sched_yield
#include <sched.h>
#endif

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

#include <string.h>
#include <stdlib.h>
#include <string>
#include <map>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    std::cerr << "PTHREAD: " #cmd " = " << ret << " (" << strerror(ret) << ")" << std::endl;	\
    assert(0); \
  } \
} while(0)

namespace Realm {

  Logger log_thread("threads");

  namespace ThreadLocal {
    /*extern*/ __thread Thread *current_thread = 0;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreReservation

  // we keep a global map of reservations to their allocations (this is inherently a global problem)
  std::map<CoreReservation *, CoreReservation::Allocation *> allocations;

  CoreReservation::CoreReservation(const std::string& _name, const CoreReservationParameters& _params)
    : name(_name), params(_params), allocation(0)
  {
    // reservations automatically add themselves to the map
    assert(allocations.count(this) == 0);
    allocations[this] = 0;

    log_thread.info() << "reservation created: " << name;
  }

  // with pthreads on Linux, an allocation is the cpu_set used for affinity
  struct CoreReservation::Allocation {
#ifndef __MACH__
    bool restrict_cpus;  // if true, thread is confined to set below
    cpu_set_t allowed_cpus;
#endif
  };

  /*static*/ bool CoreReservation::satisfy_reservations(void)
  {
    std::list<CoreReservation *> satisfied;

    // satisfy everybody with a dummy version for now
    for(std::map<CoreReservation *, CoreReservation::Allocation *>::iterator it = allocations.begin();
	it != allocations.end();
	it++) {
      // already satisfied?
      if(it->second)
	continue;

      CoreReservation::Allocation *alloc = new CoreReservation::Allocation;
#ifndef __MACH__
      alloc->restrict_cpus = false;
#endif

      it->second = alloc;
      it->first->allocation = alloc;
      satisfied.push_back(it->first);
    }

    // for all the threads we've satisfied, notify any registered listeners
    for(std::list<CoreReservation *>::iterator it = satisfied.begin();
	it != satisfied.end();
	it++) {
      CoreReservation *rsrv = *it;
      for(std::list<CoreReservation::NotificationListener *>::iterator it = rsrv->listeners.begin();
	  it != rsrv->listeners.end();
	  it++)
	(*it)->notify_allocation(*rsrv);
    }

    return true;
  }

  /*static*/ void CoreReservation::report_reservations(std::ostream& os)
  {
    // iterate over the allocation map and print stuff out
    for(std::map<CoreReservation *, CoreReservation::Allocation *>::const_iterator it = allocations.begin();
	it != allocations.end();
	it++) {
      const CoreReservation *rsrv = it->first;
      const CoreReservation::Allocation *alloc = it->second;
      os << rsrv->name << ": ";
      if(alloc) {
	os << "allocated";
      } else {
	os << "not allocated";
      }
      os << std::endl;
    }
  }

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

    if(thread->scheduler)
      thread->scheduler->thread_terminating(thread);
    
    // on return, we update our status and terminate
    log_thread.info() << "thread " << thread << " finished";
    thread->update_state(STATE_FINISHED);

    return 0;
  }

  void KernelThread::start_thread(const ThreadLaunchParameters& params,
				  const CoreReservation& rsrv)
  {
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

    if(params.stack_size != params.STACK_SIZE_DEFAULT) {
      // make sure it's not too large
      assert((rsrv.params.max_stack_size == rsrv.params.STACK_SIZE_DEFAULT) ||
	     (params.stack_size <= rsrv.params.max_stack_size));

      // pthreads also has a limit
      if(params.stack_size < PTHREAD_STACK_MIN)
	CHECK_PTHREAD( pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN) );
      else
	CHECK_PTHREAD( pthread_attr_setstacksize(&attr, params.stack_size) );
    } else {
      // does the entire core reservation have a non-standard stack size?
      if(rsrv.params.max_stack_size != rsrv.params.STACK_SIZE_DEFAULT) {
	if(rsrv.params.max_stack_size < PTHREAD_STACK_MIN)
	  CHECK_PTHREAD( pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN) );
	else
	  CHECK_PTHREAD( pthread_attr_setstacksize(&attr, rsrv.params.max_stack_size) );
      }
    }

    // TODO: actually use heap size

    update_state(STATE_STARTUP);

    // time to actually create the thread
    CHECK_PTHREAD( pthread_create(&thread, &attr, pthread_entry, this) );

    log_thread.info() << "thread created:" << this << " (" << rsrv.name << ") - pthread " << thread;
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

  class UserThread : public Thread {
  public:
    UserThread(void *_target, void (*_entry_wrapper)(void *),
	       ThreadScheduler *_scheduler);

    virtual ~UserThread(void);

    void start_thread(const ThreadLaunchParameters& params);

    virtual void join(void);
    virtual void detach(void);

    static void user_switch(UserThread *switch_to);

  protected:
    static void uthread_entry(void) __attribute__((noreturn));

    static const int MAGIC_VALUE = 0x11223344;

    void *target;
    void (*entry_wrapper)(void *);
    int magic;
    ucontext_t ctx;
    void *stack_base;
    size_t stack_size;
    bool ok_to_delete;
    bool running;
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

  void UserThread::start_thread(const ThreadLaunchParameters& params)
  {
    // figure out how big the stack should be
    if(params.stack_size != params.STACK_SIZE_DEFAULT) {
      stack_size = params.stack_size;
    } else {
      stack_size = 2 << 20; // pick something - 2MB ?
    }

    stack_base = malloc(stack_size);
    assert(stack_base != 0);

    getcontext(&ctx);

    ctx.uc_link = 0; // we don't expect it to ever fall through
    ctx.uc_stack.ss_sp = stack_base;
    ctx.uc_stack.ss_size = stack_size;
    ctx.uc_stack.ss_flags = 0;

    // grr...  entry point takes int's, which might not hold a void *
    // we'll just fish our UserThread * out of TLS
    makecontext(&ctx, uthread_entry, 0);

    update_state(STATE_STARTUP);    
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

      int ret = swapcontext(&host_ctx, &switch_to->ctx);

      // if we return with a value of 0, that means we were (eventually) given control
      //  back, as we hoped
      assert(ret == 0);

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
	int ret = swapcontext(&switch_from->ctx, &switch_to->ctx);
	assert(ret == 0);

	assert(switch_from->running == false);
	switch_from->running = true;
      } else {
	// a return of control to the host thread
	assert(ThreadLocal::host_context != 0);

	ThreadLocal::current_thread = ThreadLocal::current_host_thread;
	ThreadLocal::current_host_thread = 0;

	int ret = swapcontext(&switch_from->ctx, ThreadLocal::host_context);
	assert(ret == 0);

	// if we get control back
	assert(switch_from->running == false);
	switch_from->running = true;
      }
    }
  }


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

  /*static*/ Thread *Thread::create_user_thread_untyped(void *target, void (*entry_wrapper)(void *),
							const ThreadLaunchParameters& params,
							ThreadScheduler *_scheduler)
  {
    UserThread *t = new UserThread(target, entry_wrapper, _scheduler);

    // no need to wait on an allocation - the host thread will take care of that
    t->start_thread(params);

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

  /*static*/ void Thread::user_switch(Thread *switch_to)
  {
    // just cast 'switch_to' to a UserThread - UserThread::user_switch will do a bit
    //   of sanity-checking
    UserThread::user_switch((UserThread *)switch_to);
  }

}; // namespace Realm
