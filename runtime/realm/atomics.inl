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
  inline atomic<T>::atomic(const atomic<T>& copy_from)
#ifdef REALM_USE_STD_ATOMIC
    : value(copy_from.value.load())
#else
    : value(copy_from.value)
#endif
  {}

  template <typename T>
  inline atomic<T>& atomic<T>::operator=(const atomic<T>& copy_from)
  {
#ifdef REALM_USE_STD_ATOMIC
    value.store(copy_from.value.load());
#else
    value = copy_from.value;
#endif
    return *this;
  }

  template <typename T>
  inline T atomic<T>::load(void) const
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.load(std::memory_order_relaxed);
#else
    return *static_cast<volatile const T*>(&value);
    //return __sync_fetch_and_add(const_cast<T *>(&value), 0);
#endif
  }

  template <typename T>
  inline T atomic<T>::load_acquire(void) const
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.load(std::memory_order_acquire);
#else
    T val = *static_cast<volatile const T*>(&value);
    //T val = __sync_fetch_and_add(const_cast<T *>(&value), 0);
    __sync_synchronize();  // full memory fence is all we've got
    return val;
#endif
  }

  // a fenced load is an load_acquire that cannot be reordered with earlier stores
  template <typename T>
  inline T atomic<T>::load_fenced(void) const
  {
#ifdef REALM_USE_STD_ATOMIC
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return value.load(std::memory_order_acquire);
#else
    __sync_synchronize();  // full memory fence is all we've got
    T val = *static_cast<volatile const T*>(&value);
    //T val = __sync_fetch_and_add(const_cast<T *>(&value), 0);
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
#ifdef TSAN_ENABLED
    // use an exchange to keep tsan happy
    this->exchange(newval);
#else
    value = newval;
#endif
#endif
  }

  template <typename T>
  inline void atomic<T>::store_release(T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    value.store(newval, std::memory_order_release);
#else
#ifdef TSAN_ENABLED
    // use an exchange to keep tsan happy
    this->exchange(newval);
#else
    __sync_synchronize();  // full memory fence is all we've got
    value = newval;
#endif
#endif
  }

  template <typename T>
  inline T atomic<T>::exchange(T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.exchange(newval);
#else
    while(true) {
#ifdef TSAN_ENABLED
      // thread sanitizer doesn't like the naked read even with the CAS after
      T oldval = __sync_val_compare_and_swap(&value, newval, newval);
#else
      T oldval = value;
#endif
      if(__sync_bool_compare_and_swap(&value, oldval, newval))
	return oldval;
    }
#endif
  }

  template <typename T>
  inline bool atomic<T>::compare_exchange(T& expected, T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.compare_exchange_strong(expected, newval,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
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
  inline bool atomic<T>::compare_exchange_relaxed(T& expected, T newval)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.compare_exchange_strong(expected, newval,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed);
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
    return value.fetch_add(to_add, std::memory_order_relaxed);
#else
    return __sync_fetch_and_add(&value, to_add);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_add_acqrel(T to_add)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_add(to_add, std::memory_order_acq_rel);
#else
    return __sync_fetch_and_add(&value, to_add);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_sub(T to_sub)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_sub(to_sub, std::memory_order_relaxed);
#else
    return __sync_fetch_and_sub(&value, to_sub);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_sub_acqrel(T to_sub)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_sub(to_sub, std::memory_order_acq_rel);
#else
    return __sync_fetch_and_sub(&value, to_sub);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_and(T to_and)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_and(to_and, std::memory_order_relaxed);
#else
    return __sync_fetch_and_and(&value, to_and);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_and_acqrel(T to_and)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_and(to_and, std::memory_order_acq_rel);
#else
    return __sync_fetch_and_and(&value, to_and);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_or(T to_or)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_or(to_or, std::memory_order_relaxed);
#else
    return __sync_fetch_and_or(&value, to_or);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_or_acqrel(T to_or)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_or(to_or, std::memory_order_acq_rel);
#else
    return __sync_fetch_and_or(&value, to_or);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_xor(T to_xor)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_xor(to_xor, std::memory_order_relaxed);
#else
    return __sync_fetch_and_xor(&value, to_xor);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_xor_acqrel(T to_xor)
  {
#ifdef REALM_USE_STD_ATOMIC
    return value.fetch_xor(to_xor, std::memory_order_acq_rel);
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
  inline T atomic<T>::fetch_min_acqrel(T to_min)
  {
    // "relaxed" version is already acq_rel
    return fetch_min(to_min);
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

  template <typename T>
  inline T atomic<T>::fetch_max_acqrel(T to_max)
  {
    // "relaxed" version is already acq_rel
    return fetch_max(to_max);
  }
  

};
