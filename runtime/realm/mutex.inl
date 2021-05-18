/* Copyright 2021 Stanford University, NVIDIA Corporation
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
