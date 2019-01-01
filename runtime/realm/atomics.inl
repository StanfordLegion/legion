/* Copyright 2019 Stanford University, NVIDIA Corporation
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

// atomics for Realm - this is a simple wrapper around C++11's std::atomic
//  if available and uses gcc's __sync_* primitives otherwise

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/atomics.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class atomic<T>
  //

  template <typename T>
  inline atomic<T>::atomic(void)
  {}

  template <typename T>
  inline atomic<T>::atomic(T _value)
    : value(_value)
  {}

  template <typename T>
  inline T atomic<T>::load(void) const
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.load(std::memory_order_relaxed);
#else
    return __sync_fetch_and_add(const_cast<T *>(&value), 0);
#endif
  }

  template <typename T>
  inline T atomic<T>::load_acquire(void) const
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.load(std::memory_order_acquire);
#else
    T val = __sync_fetch_and_add(const_cast<T *>(&value), 0);
    __sync_synchronize();  // full memory fence is all we've got
    return val;
#endif
  }

  template <typename T>
  inline void atomic<T>::store(T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    value.store(newval, std::memory_order_relaxed);
#else
    value = newval;
#endif
  }

  template <typename T>
  inline void atomic<T>::store_release(T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    value.store(newval, std::memory_order_release);
#else
    __sync_synchronize();  // full memory fence is all we've got
    value = newval;
#endif
  }

  template <typename T>
  inline T atomic<T>::exchange(T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.exchange(newval);
#else
    while(true) {
      T oldval = value;
      if(__sync_vool_compare_and_swap(&value, oldval, newval))
	return oldval;
    }
#endif
  }

  template <typename T>
  inline bool atomic<T>::compare_exchange(T& expected, T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    // do we need a relaxed version of this?
    return value.compare_exchange_strong(expected, newval,
                                         std::memory_order_acq_rel);
#else
    T oldval = __sync_val_compare_and_swap(&value, expected, newval);
    if(oldval == expected) {
      return true;
    } else {
      expected = oldval;
      return false;
    }
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_add(T to_add)
  {
#ifdef REALM_USE_STD_ATOMIC
    // do we need a release version?
    return value.fetch_add(to_add, std::memory_order_relaxed);
#else
    return __sync_fetch_and_add(&value, to_add);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_sub(T to_sub)
  {
#ifdef REALM_USE_STD_ATOMIC
    // do we need a release version?
    return value.fetch_sub(to_sub, std::memory_order_relaxed);
#else
    return __sync_fetch_and_sub(&value, to_sub);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_and(T to_and)
  {
#ifdef REALM_USE_STD_ATOMIC
    // do we need a release version?
    return value.fetch_and(to_and, std::memory_order_relaxed);
#else
    return __sync_fetch_and_and(&value, to_and);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_or(T to_or)
  {
#ifdef REALM_USE_STD_ATOMIC
    // do we need a release version?
    return value.fetch_or(to_or, std::memory_order_relaxed);
#else
    return __sync_fetch_and_or(&value, to_or);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_xor(T to_xor)
  {
#ifdef REALM_USE_STD_ATOMIC
    // do we need a release version?
    return value.fetch_xor(to_xor, std::memory_order_relaxed);
#else
    return __sync_fetch_and_xor(&value, to_xor);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_min(T to_min)
  {
    // no builtins for this, so use compare_exchange
    T prev = load();
    while(prev > to_min)
      if(compare_exchange(prev, to_min))
	break;
    return prev;
  }

  template <typename T>
  inline T atomic<T>::fetch_max(T to_max)
  {
    // no builtins for this, so use compare_exchange
    T prev = load();
    while(prev < to_max)
      if(compare_exchange(prev, to_max))
	break;
    return prev;
  }


};
