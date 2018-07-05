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

// nop, but helps IDEs
#include "realm/threads.h"

#include <assert.h>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class WithDefault<T, _DEFAULT>

  template <typename T, T _DEFAULT>
  inline WithDefault<T, _DEFAULT>::WithDefault(void)
    : val(_DEFAULT)
  {}

  template <typename T, T _DEFAULT>
  inline WithDefault<T, _DEFAULT>::WithDefault(T _val)
    : val(_val)
  {}

  template <typename T, T _DEFAULT>
  inline WithDefault<T, _DEFAULT>::operator T(void) const
  {
    return val;
  }

  template <typename T, T _DEFAULT>
  inline WithDefault<T,_DEFAULT>& WithDefault<T, _DEFAULT>::operator=(T newval)
  {
    val = newval;
    return *this;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadLaunchParameters

  inline ThreadLaunchParameters::ThreadLaunchParameters(void)
  {
    // default constructors on fields do all the work
  }
  
  inline ThreadLaunchParameters& ThreadLaunchParameters::set_stack_size(ptrdiff_t new_stack_size)
  {
    this->stack_size = new_stack_size;
    return *this;
  }

  inline ThreadLaunchParameters& ThreadLaunchParameters::set_heap_size(ptrdiff_t new_heap_size)
  {
    this->heap_size = new_heap_size;
    return *this;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Thread

  inline Thread::Thread(ThreadScheduler *_scheduler)
    : state(STATE_CREATED)
    , scheduler(_scheduler)
    , current_op(0)
    , exception_handler_count(0)
    , signal_count(0)
  {
  }

  inline Thread::~Thread(void)
  {
  }    

  inline Thread::State Thread::get_state(void)
  {
    return state;
  }

  // atomically updates the thread's state, returning the old state
  inline Thread::State Thread::update_state(Thread::State new_state)
  {
    // strangely, there's no standard exchange - just compare-and-exchange
    State old_state;
    do {
      old_state = state;
    } while(!__sync_bool_compare_and_swap(&state, old_state, new_state));
    return old_state;
  }

  // updates the thread's state, but only if it's in the specified 'old_state' (i.e. an
  //  atomic compare and swap) - returns true on success and false on failure
  inline bool Thread::try_update_state(Thread::State old_state, Thread::State new_state)
  {
    // this one maps directly to compiler atomics
    return __sync_bool_compare_and_swap(&state, old_state, new_state);
  }

  // use compiler-provided TLS for quickly finding our thread - stick this in another
  //  namespace to make it obvious
  namespace ThreadLocal {
    extern __thread Thread *current_thread;
  };
  
  inline /*static*/ Thread *Thread::self(void)
  {
    return ThreadLocal::current_thread;
  }

  template <typename T, void (T::*START_MTHD)(void)>
  /*static*/ void Thread::thread_entry_wrapper(void *obj)
  {
    // just calls the actual method on the passed-in object
    (((T *)obj)->*START_MTHD)();
  }
    
  // for kernel threads, the scheduler is optional - the default is one that lets
  //  the OS manage all runnable threads however it wants
  template <typename T, void (T::*START_MTHD)(void)>
  /*static*/ Thread *Thread::create_kernel_thread(T *target,
						  const ThreadLaunchParameters& params,
						  CoreReservation& rsrv,
						  ThreadScheduler *_scheduler /*= 0*/)
  {
    return create_kernel_thread_untyped(target, thread_entry_wrapper<T, START_MTHD>,
					params, rsrv, _scheduler);
  }

#ifdef REALM_USE_USER_THREADS
  // user threads must specify a scheduler - the whole point is that the OS isn't
  //  controlling them...
  template <typename T, void (T::*START_MTHD)(void)>
  /*static*/ Thread *Thread::create_user_thread(T *target,
						const ThreadLaunchParameters& params,
						const CoreReservation *rsrv,
						ThreadScheduler *_scheduler)
  {
    return create_user_thread_untyped(target, thread_entry_wrapper<T, START_MTHD>,
				      params, rsrv, _scheduler);
  }
#endif

  template <typename CONDTYPE>
  class ThreadWaker : public CONDTYPE::Callback {
  public:
    ThreadWaker(const CONDTYPE& cond, Thread *_thread);
    void operator()(bool _poisoned);

    Thread *thread;
    bool poisoned;
  };

  template <typename CONDTYPE>
  ThreadWaker<CONDTYPE>::ThreadWaker(const CONDTYPE& cond, Thread *_thread)
    : CONDTYPE::Callback(cond), thread(_thread)
  {
  }

  template <typename CONDTYPE>
  void ThreadWaker<CONDTYPE>::operator()(bool _poisoned)
  {
    // just store the poison state here - the thread will have to check it
    //  once it starts back up
    poisoned = _poisoned;

    // mark the thread as ready and notify the thread's scheduler if it has already gone to sleep
    Thread::State old_state = thread->update_state(Thread::STATE_READY);
    switch(old_state) {
    case Thread::STATE_BLOCKING:
      {
	// we caught it before it went to sleep, so nothing to do
	break;
      }

    case Thread::STATE_ALERTED:
      {
	// it was asleep, but it's being awakened due to an alert, and will 
	// notice that we've set the state to READY
	break;
      }

    case Thread::STATE_BLOCKED:
      {
	// it's asleep - tell the scheduler to wake it upt
	assert(thread->scheduler);
	thread->scheduler->thread_ready(thread);
	break;
      }

    default:
      {
	// all other cases are illegal
	assert(0);
      };
    }
  }

  template <typename CONDTYPE>
  /*static*/ void Thread::wait_for_condition(const CONDTYPE& cond, bool& poisoned)
  {
    Thread *thread = self();

    // suspend any performance counters before we do internal stuff
#ifdef REALM_USE_PAPI
    if(thread->papi_counters) thread->papi_counters->suspend();
#endif

    // we're interacting with the scheduler, so check for signals first
    if(thread->signal_count > 0)
      thread->process_signals();

    // first, indicate our intent to sleep
    thread->update_state(STATE_BLOCKING);
    // now create the callback so we know when to wake up
    ThreadWaker<CONDTYPE> cb(cond, thread);
    cond.add_callback(cb);

    // now tell the scheduler we are blocking
    //  (it will update our status if we succeed in blocking)
    assert(thread->scheduler != 0);
    thread->scheduler->thread_blocking(thread);

    // poison propagates to caller
    poisoned = cb.poisoned;

    // check signals again on the way out (async ones should have woken us already,
    //  but synchronous ones do not)
    if(thread->signal_count > 0)
      thread->process_signals();

    // finally, resume any performance counters
#ifdef REALM_USE_PAPI
    if(thread->papi_counters) thread->papi_counters->resume();
#endif
  }

  inline bool Thread::exceptions_permitted(void) const
  {
    return (exception_handler_count > 0);
  }

  inline void Thread::start_operation(Operation *op)
  {
    assert(current_op == 0);
    current_op = op;
  }

  inline void Thread::stop_operation(Operation *op)
  {
    assert(current_op == op);
    current_op = 0;
  }

  inline Operation *Thread::get_operation(void) const
  {
    return current_op;
  }

  inline void Thread::setup_perf_counters(const ProfilingMeasurementCollection& pmc)
  {
#ifdef REALM_USE_PAPI
    papi_counters = PAPICounters::setup_counters(pmc);
#endif
  }

  inline void Thread::start_perf_counters(void)
  {
#ifdef REALM_USE_PAPI
    if(papi_counters) papi_counters->start();
#endif
  }

  inline void Thread::stop_perf_counters(void)
  {
#ifdef REALM_USE_PAPI
    if(papi_counters) papi_counters->stop();
#endif
  }

  inline void Thread::record_perf_counters(ProfilingMeasurementCollection& pmc)
  {
#ifdef REALM_USE_PAPI
    if(papi_counters) {
      papi_counters->record(pmc);
      papi_counters->cleanup();
      papi_counters = 0; // cleanup call might delete, or save it for later
    }
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Thread::ExceptionHandlerPresence

  inline Thread::ExceptionHandlerPresence::ExceptionHandlerPresence(void)
  {
    // no need for locks - only called within the thread
    Thread::self()->exception_handler_count++;
  }

  inline Thread::ExceptionHandlerPresence::~ExceptionHandlerPresence(void)
  {
    // no need for locks - only called within the thread
    Thread::self()->exception_handler_count--;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Coroutine<YT, RT>
#if DO_I_REALLY_WANT_COROUTINES
  template <typename YT, typename RT>
  inline Coroutine<YT, RT>::~Coroutine(void)
  {
  }

  template <typename YT, typename RT>
  inline YT Coroutine<YT, RT>::get_yield_value(void)
  {
    // illegal unless we're actually yielded...
    assert(state == STATE_YIELDED);
    return yield_value;
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadScheduler

  inline ThreadScheduler::~ThreadScheduler(void)
  {
  }

  inline Thread::State ThreadScheduler::update_thread_state(Thread *thread, 
							    Thread::State new_state)
  {
    return thread->update_state(new_state);
  }

  inline bool ThreadScheduler::try_update_thread_state(Thread *thread,
						       Thread::State old_state,
						       Thread::State new_state)
  {
    return thread->try_update_state(old_state, new_state);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CoreReservationParameters

  inline CoreReservationParameters::CoreReservationParameters(void)
  {
    // default constructors on fields do all the work
  }
  
  inline CoreReservationParameters& CoreReservationParameters::set_num_cores(int new_num_cores)
  {
    this->num_cores = new_num_cores;
    return *this;
  }

  inline CoreReservationParameters& CoreReservationParameters::set_numa_domain(int new_numa_domain)
  {
    this->numa_domain = new_numa_domain;
    return *this;
  }

  inline CoreReservationParameters& CoreReservationParameters::set_alu_usage(CoreUsage new_alu_usage)
  {
    this->alu_usage = new_alu_usage;
    return *this;
  }

  inline CoreReservationParameters& CoreReservationParameters::set_fpu_usage(CoreUsage new_fpu_usage)
  {
    this->fpu_usage = new_fpu_usage;
    return *this;
  }

  inline CoreReservationParameters& CoreReservationParameters::set_ldst_usage(CoreUsage new_ldst_usage)
  {
    this->ldst_usage = new_ldst_usage;
    return *this;
  }

  inline CoreReservationParameters& CoreReservationParameters::set_max_stack_size(ptrdiff_t new_max_stack_size)
  {
    this->max_stack_size = new_max_stack_size;
    return *this;
  }

  inline CoreReservationParameters& CoreReservationParameters::set_max_heap_size(ptrdiff_t new_max_heap_size)
  {
    this->max_heap_size = new_max_heap_size;
    return *this;
  }


};

