/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// mutexes, condition variables, etc.

// NOP, but helps IDEs
#include "realm/mutex.h"

#ifdef DEBUG_REALM
#include <assert.h>
#endif

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Doorbell
  //

  inline void Doorbell::prepare()
  {
    // uncontested transition from IDLE to PENDING_AWAKE
#ifdef DEBUG_REALM
    uint32_t prev = state.exchange(STATE_PENDING_AWAKE);
    assert(prev == STATE_IDLE);
    (void)prev;
#else
    state.store(STATE_PENDING_AWAKE);
    #endif
  }

  inline void Doorbell::cancel()
  {
    // uncontested transition from PENDING_AWAKE to IDLE
#ifdef DEBUG_REALM
    uint32_t prev = state.exchange(STATE_IDLE);
    assert(prev == STATE_PENDING_AWAKE);
    (void)prev;
#else
    state.store(STATE_IDLE);
    #endif
  }

  inline bool Doorbell::satisfied()
  {
    uint32_t val = state.load_acquire();
#ifdef DEBUG_REALM
    // should not be asking about an idle doorbell
    assert(val != STATE_IDLE);
#endif
    return ((val & STATE_SATISFIED_BIT) != 0);
  }

  inline uint32_t Doorbell::wait()
  {
    uint32_t val = state.load_acquire();
    // mark satisfied path as "likely" because other is slow anyway
    if(REALM_LIKELY((val & STATE_SATISFIED_BIT) != 0)) {
      // uncontested transition back to IDLE
#ifdef DEBUG_REALM
      uint32_t prev = state.exchange(STATE_IDLE);
      assert(prev == val);
      (void)prev;
#else
      state.store(STATE_IDLE);
#endif
      return (val >> STATE_SATISFIED_VAL_SHIFT);
    } else
      return wait_slow();
  }

  inline bool Doorbell::is_sleeping() const
  {
    uint32_t val = state.load();
#ifdef DEBUG_REALM
    // should not be asking about an idle or satisfied doorbell
    assert((val != STATE_IDLE) && ((val & STATE_SATISFIED_BIT) == 0));
#endif
    return (val != STATE_PENDING_AWAKE);
  }

  inline void Doorbell::notify(uint32_t data)
  {
    // unconditional write of SATISFIED_BIT+data, but we need to know what
    //  the old state was to see if a kernel wakeup is required
    uint32_t newval = STATE_SATISFIED_BIT + (data << STATE_SATISFIED_VAL_SHIFT);
    uint32_t oldval = state.exchange(newval);
    if(REALM_UNLIKELY(oldval == STATE_PENDING_ASLEEP))
      notify_slow();
  }

  inline void Doorbell::prewake()
  {
    // unconditional write of STATE_PENDING_PREWAKE, expecting to see only
    //  AWAKE or ASLEEP as the previous state - if ASLEEP, take slow path to
    //  wake thread
    uint32_t oldval = state.exchange(STATE_PENDING_PREWAKE);
#ifdef DEBUG_REALM
    assert((oldval == STATE_PENDING_AWAKE) || (oldval == STATE_PENDING_ASLEEP));
#endif
    if(REALM_UNLIKELY(oldval == STATE_PENDING_ASLEEP))
      prewake_slow();
  }

  inline void Doorbell::cancel_prewake()
  {
    // uncontested transition from PREWAKE back to AWAKE
#ifdef DEBUG_REALM
    uint32_t prev = state.exchange(STATE_PENDING_AWAKE);
    assert(prev == STATE_PENDING_PREWAKE);
    (void)prev;
#else
    state.store(STATE_PENDING_AWAKE);
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DoorbellList
  //

  inline bool DoorbellList::add_doorbell(Doorbell *db)
  {
#ifdef REALM_ENABLE_STARVATION_CHECKS
    db->starvation_count = 0;
#endif
    uintptr_t oldval = head_or_count.load();
    while(true) {
      if((oldval & 1) == 0) {
        // appears to be the head of a list - try to stick 'db' on the front
        db->next_doorbell = reinterpret_cast<Doorbell *>(oldval);
        uintptr_t newval = reinterpret_cast<uintptr_t>(db);
        if(head_or_count.compare_exchange(oldval, newval))
          return true;  // added to list
      } else {
        // appears to be a nonzero "extra count" - try to take one
        uintptr_t newval = ((oldval == 1) ? 0 : (oldval - 2));
        if(head_or_count.compare_exchange(oldval, newval))
          return false;  // consumed token
      }
      // CAS on one path or the other failed - try again...
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UnfairMutex
  //

  inline UnfairMutex::UnfairMutex()
    : state(0)
  {}

  inline void UnfairMutex::lock()
  {
    // uncontended path is changing state from 0 to 1 - use atomic OR
    uint32_t prev = state.fetch_or_acqrel(1);
    if(REALM_LIKELY((prev & 1) == 0)) {
      return;
    } else {
      // fall back to slow path
      lock_slow();
    }
  }

  inline bool UnfairMutex::trylock()
  {
    // uncontended path is changing state from 0 to 1 - use atomic OR
    uint32_t prev = state.fetch_or_acqrel(1);
    // no slow path here - either the bit wasn't set and we have the lock,
    //  or it was and we don't
    return ((prev & 1) == 0);
  }

  inline void UnfairMutex::unlock()
  {
    // fast case is transition from 1 to 0 - use CAS to avoid releasing
    //  lock if there are waiters that we should give the lock to instead
    uint32_t expected = 1;
    if(REALM_LIKELY(state.compare_exchange(expected, 0))) {
      return;
    } else {
      // fall back to slow path
      unlock_slow();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FIFOMutex
  //

  inline FIFOMutex::FIFOMutex()
    : state(0)
  {}

  inline void FIFOMutex::lock()
  {
    // uncontended path is changing state from 0 to 1 - use atomic OR
    uint32_t prev = state.fetch_or_acqrel(1);
    if(REALM_LIKELY((prev & 1) == 0)) {
      return;
    } else {
      // fall back to slow path
      lock_slow();
    }
  }

  inline bool FIFOMutex::trylock()
  {
    // uncontended path is changing state from 0 to 1 - use atomic OR
    uint32_t prev = state.fetch_or_acqrel(1);
    // no slow path here - either the bit wasn't set and we have the lock,
    //  or it was and we don't
    return ((prev & 1) == 0);
  }

  inline void FIFOMutex::unlock()
  {
    // fast case is transition from 1 to 0 - use CAS to avoid releasing
    //  lock if there are waiters that we should give the lock to instead
    uint32_t expected = 1;
    if(REALM_LIKELY(state.compare_exchange(expected, 0))) {
      return;
    } else {
      // fall back to slow path
      unlock_slow();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MutexChecker
  //

  inline MutexChecker::MutexChecker(const char *_name, void *_object /*= 0*/,
                                    int _limit /*= 1*/)
    : name(_name)
    , object(_object)
    , limit(_limit)
    , cur_count(0)
  {}

  inline MutexChecker::~MutexChecker()
  {
    // count should be 0 on destruction
    int actval = cur_count.load();
    if(REALM_UNLIKELY(actval != 0))
      lock_fail(actval, 0);
  }

  inline void MutexChecker::lock(CheckedScope *cs /*= 0*/)
  {
    // unconditional increment of count - if it exceeds the limit we've
    //  violated the supposed invariant
    int actval = cur_count.fetch_add(1);  // NOTE: intentionally relaxed MO
    if(REALM_UNLIKELY((actval < 0) || (actval >= limit)))
      lock_fail(actval, cs);
  }

  inline bool MutexChecker::trylock(CheckedScope *cs /*= 0*/)
  {
    // contention is not allowed, so just fall through to lock
    lock(cs);
    return true;
  }

  inline void MutexChecker::unlock(CheckedScope *cs /*= 0*/)
  {
    // use a (relaxed) CAS here because if the count was too high, we want
    //  to try to catch all the threads that were in (or about to enter) the
    //  guarded region
    int expval = cur_count.load();
    while(true) {
      if(REALM_UNLIKELY((expval <= 0) || (expval > limit)))
        unlock_fail(expval, cs);
      if(cur_count.compare_exchange_relaxed(expval, expval - 1))
        break;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MutexChecker::CheckedScope
  //

  inline MutexChecker::CheckedScope::CheckedScope(MutexChecker& _checker,
                                                  const char *_name,
                                                  void *_object /*= 0*/)
    : checker(_checker)
    , name(_name)
    , object(_object)
  {
    checker.lock(this);
  }

  inline MutexChecker::CheckedScope::~CheckedScope()
  {
    checker.unlock(this);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class AutoLock<LT>
  //

  template <typename LT>
  inline AutoLock<LT>::AutoLock(LT& mutex)
    : mutex(mutex), held(true)
  { 
    mutex.lock();
  }

  template <typename LT>
  inline AutoLock<LT>::~AutoLock(void) 
  {
    if(held)
      mutex.unlock();
  }

  template <typename LT>
  inline void AutoLock<LT>::release(void)
  {
#ifdef DEBUG_REALM
    assert(held);
#endif
    mutex.unlock();
    held = false;
  }

  template <typename LT>
  inline void AutoLock<LT>::reacquire(void)
  {
#ifdef DEBUG_REALM
    assert(!held);
#endif
    mutex.lock();
    held = true;
  }

  
};
