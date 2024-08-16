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
  inline atomic<T>::atomic(const atomic<T> &copy_from)
    : value(copy_from.value.load())
  {}

  template <typename T>
  inline atomic<T> &atomic<T>::operator=(const atomic<T> &copy_from)
  {
    value.store(copy_from.value.load());
    return *this;
  }

  template <typename T>
  inline T atomic<T>::load(void) const
  {
    return value.load(std::memory_order_relaxed);
  }

  template <typename T>
  inline T atomic<T>::load_acquire(void) const
  {
    return value.load(std::memory_order_acquire);
  }

  // a fenced load is an load_acquire that cannot be reordered with earlier stores
  template <typename T>
  inline T atomic<T>::load_fenced(void) const
  {
    return value.load(std::memory_order_seq_cst);
  }

  template <typename T>
  inline void atomic<T>::store(T newval)
  {
    value.store(newval, std::memory_order_relaxed);
  }

  template <typename T>
  inline void atomic<T>::store_release(T newval)
  {
    value.store(newval, std::memory_order_release);
  }

  template <typename T>
  inline T atomic<T>::exchange(T newval)
  {
    return value.exchange(newval);
  }

  template <typename T>
  inline bool atomic<T>::compare_exchange(T &expected, T newval)
  {
    return value.compare_exchange_strong(expected, newval, std::memory_order_acq_rel,
                                         std::memory_order_acquire);
  }

  template <typename T>
  inline bool atomic<T>::compare_exchange_relaxed(T &expected, T newval)
  {
    return value.compare_exchange_strong(expected, newval, std::memory_order_relaxed,
                                         std::memory_order_relaxed);
  }
  template <typename T>
  inline bool atomic<T>::compare_exchange_weak(T &expected, T newval)
  {
#if !defined(TSAN_ENABLED)
    // TSAN generates a false positive with compare_exchange_weak, so use the strong
    // version instead
    return value.compare_exchange_weak(expected, newval, std::memory_order_acq_rel,
                                       std::memory_order_acquire);
#else
    return compare_exchange(expected, newval);
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_add(T to_add)
  {
    return value.fetch_add(to_add, std::memory_order_relaxed);
  }

  template <typename T>
  inline T atomic<T>::fetch_add_acqrel(T to_add)
  {
    return value.fetch_add(to_add, std::memory_order_acq_rel);
  }

  template <typename T>
  inline T atomic<T>::fetch_sub(T to_sub)
  {
    return value.fetch_sub(to_sub, std::memory_order_relaxed);
  }

  template <typename T>
  inline T atomic<T>::fetch_sub_acqrel(T to_sub)
  {
    return value.fetch_sub(to_sub, std::memory_order_acq_rel);
  }

  template <typename T>
  inline T atomic<T>::fetch_and(T to_and)
  {
    return value.fetch_and(to_and, std::memory_order_relaxed);
  }

  template <typename T>
  inline T atomic<T>::fetch_and_acqrel(T to_and)
  {
    return value.fetch_and(to_and, std::memory_order_acq_rel);
  }

  template <typename T>
  inline T atomic<T>::fetch_or(T to_or)
  {
    return value.fetch_or(to_or, std::memory_order_relaxed);
  }

  template <typename T>
  inline T atomic<T>::fetch_or_acqrel(T to_or)
  {
    return value.fetch_or(to_or, std::memory_order_acq_rel);
  }

  template <typename T>
  inline T atomic<T>::fetch_xor(T to_xor)
  {
    return value.fetch_xor(to_xor, std::memory_order_relaxed);
  }

  template <typename T>
  inline T atomic<T>::fetch_xor_acqrel(T to_xor)
  {
    return value.fetch_xor(to_xor, std::memory_order_acq_rel);
  }

  template <typename T>
  inline T atomic<T>::fetch_min(T to_min)
  {
#if __cplusplus >= 202602ULL
    return value.fetch_min(to_min, std::memory_order_relaxed);
#else
    // no builtins for this, so use compare_exchange
    T prev = load();
    while(prev > to_min && !compare_exchange_weak(prev, to_min))
      ;
    return prev;
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_min_acqrel(T to_min)
  {
#if __cplusplus >= 202602ULL
    return value.fetch_min(to_min, std::memory_order_acq_rel);
#else
    // no builtins for this, so use compare_exchange
    T prev = load();
    while(prev > to_min && !compare_exchange_weak(prev, to_min))
      ;
    return prev;
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_max(T to_max)
  {
#if __cplusplus >= 202602ULL
    return value.fetch_max(to_max, std::memory_order_relaxed);
#else
    // no builtins for this, so use compare_exchange
    T prev = load();
    while(prev < to_max && !compare_exchange_weak(prev, to_max))
      ;
    return prev;
#endif
  }

  template <typename T>
  inline T atomic<T>::fetch_max_acqrel(T to_max)
  {
#if __cplusplus >= 202602ULL
    return value.fetch_max(to_max, std::memory_order_acq_rel);
#else
    // no builtins for this, so use compare_exchange
    T prev = load();
    while(prev < to_max && !compare_exchange_weak(prev, to_max))
      ;
    return prev;
#endif
  }

}; // namespace Realm
