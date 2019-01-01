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

#ifndef REALM_ATOMICS_H
#define REALM_ATOMICS_H

#if __cplusplus >= 201103L
#define REALM_USE_STD_ATOMIC
#endif

#ifdef REALM_USE_STD_ATOMIC
#include <atomic>
#endif

namespace Realm {

  template <typename T>
  class atomic {
  public:
    atomic(void);
    atomic(T _value);

    T load(void) const;
    T load_acquire(void) const;

    void store(T newval);
    void store_release(T newval);

    // atomic ops
    T exchange(T newval);
    bool compare_exchange(T& expected, T newval);

    T fetch_add(T to_add);
    T fetch_sub(T to_sub);
    T fetch_and(T to_and);
    T fetch_or(T to_or);
    T fetch_xor(T to_xor);
    T fetch_min(T to_min);
    T fetch_max(T to_max);

  protected:
#ifdef REALM_USE_STD_ATOMIC
    std::atomic<T> value;
#else
    T value;
#endif
  };

};

#include "realm/atomics.inl"

#endif
