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

#ifndef REALM_ATOMICS_H
#define REALM_ATOMICS_H

#include "realm/realm_config.h"

#if (REALM_CXX_STANDARD >= 11) && !defined(REALM_NO_USE_STD_ATOMIC)
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

    // this is explicit to disallow assignment of base value (must use store())
    explicit atomic(T _value);

    // copy constructors are needed if atomics are put in a container
    atomic(const atomic<T>& copy_from);
    atomic<T>& operator=(const atomic<T>& copy_from);

    T load(void) const;
    T load_acquire(void) const;
    // a fenced load is an load_acquire that cannot be reordered with earlier stores
    T load_fenced(void) const;

    void store(T newval);
    void store_release(T newval);

    // atomic ops
    T exchange(T newval);
    bool compare_exchange(T& expected, T newval);
    bool compare_exchange_relaxed(T& expected, T newval);

    // these updates use relaxed semantics, guaranteeing atomicity, but
    //  imposing no constraints on other loads and stores - use *_acqrel
    //  variants below for fencing behavior
    T fetch_add(T to_add);
    T fetch_sub(T to_sub);
    T fetch_and(T to_and);
    T fetch_or(T to_or);
    T fetch_xor(T to_xor);
    T fetch_min(T to_min);
    T fetch_max(T to_max);

    T fetch_add_acqrel(T to_add);
    T fetch_sub_acqrel(T to_sub);
    T fetch_and_acqrel(T to_and);
    T fetch_or_acqrel(T to_or);
    T fetch_xor_acqrel(T to_xor);
    T fetch_min_acqrel(T to_min);
    T fetch_max_acqrel(T to_max);

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
