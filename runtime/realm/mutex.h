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

#include "realm/realm_config.h"

#include <stdint.h>

namespace Realm {

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE Mutex {
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

  // a reader/writer lock allows one or more readers OR one writer at a time
  class RWLock {
  public:
    RWLock();
    ~RWLock();

    // should never be copied (or moved)
#if REALM_CXX_STANDARD >= 11
    RWLock(const RWLock&) = delete;
    RWLock(RWLock&&) = delete;
    RWLock& operator=(const RWLock&) = delete;
    RWLock& operator=(RWLock&&) = delete;
#else
  private:
    RWLock(const RWLock&) : writer(*this), reader(*this) {}
    RWLock& operator=(const RWLock&) { return *this; }
  public:
#endif

    void wrlock();
    bool trywrlock();
    void rdlock();
    bool tryrdlock();
    void unlock();

    // to allow use with AutoLock<T>, the RWLock provides "aspects" that
    //  look like normal locks
    struct Writer {
      Writer(RWLock& _rwlock) : rwlock(_rwlock) {}
      void lock() { rwlock.wrlock(); }
      void trylock() { rwlock.trywrlock(); }
      void unlock() { rwlock.unlock(); }
    protected:
      RWLock& rwlock;
    };

    struct Reader {
      Reader(RWLock& _rwlock) : rwlock(_rwlock) {}
      void lock() { rwlock.rdlock(); }
      void trylock() { rwlock.tryrdlock(); }
      void unlock() { rwlock.unlock(); }
    protected:
      RWLock& rwlock;
    };

    typedef AutoLock<Writer> AutoWriterLock;
    typedef AutoLock<Reader> AutoReaderLock;

    // allow free coercion to these aspects
    operator Writer&() { return writer; }
    operator Reader&() { return reader; }

  protected:
    Writer writer;
    Reader reader;
    // the actual implementation of the rwlock is hidden
    //  but if we don't define it here, we run afoul of
    //  rules type-punning, so use macros to let mutex.cc's inclusion
    //  of this file behave a little differently
    union {
#ifdef REALM_ON_MACOS
      // apparently pthread_rwlock_t's are LARGE on macOS
      uint64_t placeholder[32]; // 256 bytes, at least 8 byte aligned
#else
      uint64_t placeholder[8]; // 64 bytes, at least 8 byte aligned
#endif
#ifdef REALM_RWLOCK_IMPL
      REALM_RWLOCK_IMPL;
#endif
    };
  };
};

#include "realm/mutex.inl"

#endif
