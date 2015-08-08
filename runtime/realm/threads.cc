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

#include <pthread.h>

#include <string.h>
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
    /*extern*/ __thread Thread *current_thread;
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
      alloc->restrict_cpus = false;

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

    void start_thread(const ThreadLaunchParameters& params,
		      const CoreReservation& rsrv);

    virtual void join(void);

  protected:
    static void *pthread_entry(void *data);

    void *target;
    void (*entry_wrapper)(void *);
    pthread_t thread;
  };

  KernelThread::KernelThread(void *_target, void (*_entry_wrapper)(void *),
			     ThreadScheduler *_scheduler)
    : Thread(_scheduler), target(_target), entry_wrapper(_entry_wrapper)
  {
  }

  /*static*/ void *KernelThread::pthread_entry(void *data)
  {
    KernelThread *thread = (KernelThread *)data;

    // set up TLS so people can find us
    ThreadLocal::current_thread = thread;

    log_thread.info() << "thread " << thread << " started";
    thread->update_state(STATE_RUNNING);
    
    // call the actual thread body
    (*thread->entry_wrapper)(thread->target);

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

    if(rsrv.allocation->restrict_cpus)
      CHECK_PTHREAD( pthread_attr_setaffinity_np(&attr, 
						 sizeof(rsrv.allocation->allowed_cpus),
						 &(rsrv.allocation->allowed_cpus)) );

    if(params.stack_size != params.STACK_SIZE_DEFAULT) {
      // make sure it's not too large
      assert((rsrv.params.max_stack_size == rsrv.params.STACK_SIZE_DEFAULT) ||
	     (params.stack_size <= rsrv.params.max_stack_size));
      CHECK_PTHREAD( pthread_attr_setstacksize(&attr, params.stack_size) );
    } else {
      // does the entire core reservation have a non-standard stack size?
      if(rsrv.params.max_stack_size != rsrv.params.STACK_SIZE_DEFAULT)
	CHECK_PTHREAD( pthread_attr_setstacksize(&attr, rsrv.params.max_stack_size) );
    }

    // TODO: actually use heap size

    // time to actually create the thread
    CHECK_PTHREAD( pthread_create(&thread, &attr, pthread_entry, this) );

    log_thread.info() << "thread created:" << this << " (" << rsrv.name << ") - pthread " << thread;
  }

  void KernelThread::join(void)
  {
    CHECK_PTHREAD( pthread_join(thread, 0 /* ignore retval */) );
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
  // class Thread

  /*static*/ Thread *Realm::Thread::create_kernel_thread_untyped(void *target, void (*entry_wrapper)(void *),
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


}; // namespace Realm
