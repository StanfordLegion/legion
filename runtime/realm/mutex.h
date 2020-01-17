/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#ifndef REALM_MUTEX_H
#define REALM_MUTEX_H

#include <stdint.h>

namespace Realm {

  class Mutex {
  public:
    Mutex(void);
    ~Mutex(void);

  private:
    // Should never be copied
    Mutex(const Mutex &rhs);
    Mutex& operator=(const Mutex &rhs);

  public:
    void lock(void);
    void unlock(void);
    bool trylock(void);

  protected:
    friend class CondVar;

    // the actual implementation of the mutex is hidden
    //  but if we don't define it here, we run afoul of
    //  rules type-punning, so use macros to let mutex.cc's inclusion
    //  of this file behave a little differently
    union {
      uint64_t placeholder[8]; // 64 bytes, at least 8 byte aligned
#ifdef REALM_MUTEX_IMPL
      REALM_MUTEX_IMPL;
#endif
    };
  };

  class CondVar {
  public:
    CondVar(Mutex &_mutex);
    ~CondVar(void);

    // these require that you hold the lock when you call
    void signal(void);

    void broadcast(void);

    void wait(void);

    // wait with a timeout - returns true if awakened by a signal and
    //  false if the timeout expires first
    bool timedwait(long long max_nsec);

    Mutex &mutex;

  protected:
    // the actual implementation of the condition variable is hidden
    //  but if we don't define it here, we run afoul of
    //  rules type-punning, so use macros to let mutex.cc's inclusion
    //  of this file behave a little differently
    union {
      uint64_t placeholder[8]; // 64 bytes, at least 8 byte aligned
#ifdef REALM_CONDVAR_IMPL
      REALM_CONDVAR_IMPL;
#endif
    };
  };

  template <typename LT = Mutex>
  class AutoLock {
  public:
    AutoLock(LT& _mutex);
    ~AutoLock();

    void release();
    void reacquire();
    
  protected:
    LT& mutex;
    bool held;
  };

  
};

#include "realm/mutex.inl"

#endif
