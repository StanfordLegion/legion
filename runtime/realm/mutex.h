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

#ifndef REALM_MUTEX_H
#define REALM_MUTEX_H

#include "realm/realm_config.h"

#include "realm/utils.h"
#include "realm/atomics.h"

#include <stdint.h>

// if enabled, we count how many times a doorbell is passed over and
//  print warnings if it seems like a lot - this has known false positives
//  (e.g. an UnfairCondVar used to wake up any one of a pool of sleeping
//  threads or an UnfairMutex under heavy contention), so do not enable
//  by default
// TODO: control with command-line switch?
//define REALM_ENABLE_STARVATION_CHECKS

namespace Realm {

  // Realm mutexes come in a few different flavors:
  // a) UnfairMutex - provides mutual exclusion, using doorbells to pass the
  //     mutex on to a waiting thread on release, favoring the most recent
  //     waiters who are still spinning in order to avoid kernel-space
  //     operations as much as possible.  This is wildly unfair, but if your
  //     need is just to protect a short critical section with minimal
  //     overhead, this should provide the best throughput
  // b) FIFOMutex - also uses doorbells, but favors the oldest waiter, even if
  //     they've gone to sleep (and therefore need kernel assistance to wake),
  //     in order to preserve fairness
  // c) KernelMutex - directly uses the system library/kernel mutex support
  //     (e.g. pthread_mutex), and is at the mercy of whatever scheduling
  //     algorithms those use (which are almost certainly not aware of which
  //     Realm threads have dedicated cores) - use only as a last resort

  class UnfairMutex;
  class FIFOMutex;
  class KernelMutex;

  // each has a matching CondVar - we don't set a default, use
  //  (whatever)Mutex::CondVar to get the right one
  class UnfairCondVar;
  class FIFOCondVar;
  class KernelCondVar;

#ifdef REALM_DEFAULT_MUTEX
  typedef REALM_DEFAULT_MUTEX Mutex;
#else
  typedef UnfairMutex Mutex;
#endif

  // a mutual exclusion checker does not guarantee mutual exclusion, but
  //  instead tests dynamically whether mutual exclusion has been achieved
  //  through other means
  // unlike real mutexes where "lock" has memory acquire semantics and
  //  "unlock" has memory release semantics, the checker tries to impose
  //  as little memory ordering as possible
  // finally, although the MutexChecker is compatible with the templated
  //  AutoLock, it's encouraged to use 'MutexChecker::CheckedScope' for
  //  the better diagnostic info
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE MutexChecker : public noncopyable {
  public:
    MutexChecker(const char *_name, void *_object = 0, int _limit = 1);
    ~MutexChecker();

    // no associated CondVar

    class CheckedScope {
    public:
      CheckedScope(MutexChecker& _checker, const char *_name, void *_object = 0);
      ~CheckedScope();

    protected:
      friend class MutexChecker;

      MutexChecker& checker;
      const char *name;
      void *object;
    };

    void lock(CheckedScope *cs = 0);
    bool trylock(CheckedScope *cs = 0);
    void unlock(CheckedScope *cs = 0);

  protected:
    void lock_fail(int actval, CheckedScope *cs);
    void unlock_fail(int actval, CheckedScope *cs);

    const char *name;
    void *object;
    int limit;
    atomic<int> cur_count;
  };

  // a doorbell is used to notify a thread that whatever condition it has been
  //  waiting for has been satisfied - all operations are lock-free in user space,
  //  but wait and notify may cross into kernel space if the doorbell owner goes
  //  (or tries to go) to sleep
  // the notification process allows 31 bits worth of information to be
  //  delivered as well
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE Doorbell {
  protected:
    // not constructed directly - use get_thread_doorbell
    Doorbell();
    ~Doorbell();

  public:
    // gets the doorbell for the current thread (you can't lookup anybody else's)
    static Doorbell *get_thread_doorbell();

    // each doorbell can be separately configured for how long it should spin
    //  before going to sleep
    static const long long DOORBELL_SLEEP_IMMEDIATE = 0;
    static const long long DOORBELL_SLEEP_NEVER = -1;
    static const long long DOORBELL_SLEEP_DEFAULT = 10000;  // 10 us

    void set_sleep_timeout(long long timeout_in_ns);

    // waiter interface
    void prepare();
    void cancel();
    bool satisfied();  // indicates satisfaction - wait must be called but is guaranteed to be immediate
    uint32_t wait();

    // waker interface
    bool is_sleeping() const;
    void notify(uint32_t data);
    // prewake does not ring the doorbell, but it does pre-wake the waiter
    //  if needed so that some latency of kernel scheduling can be hidden
    void prewake();
    void cancel_prewake();

  protected:
    // "slow" versions of the above when the fast case isn't met
    uint32_t wait_slow();
    void notify_slow();
    void prewake_slow();

    static const uint32_t STATE_IDLE = 0;
    static const uint32_t STATE_PENDING_AWAKE = 2;
    static const uint32_t STATE_PENDING_ASLEEP = 4;
    static const uint32_t STATE_PENDING_PREWAKE = 6;
    static const uint32_t STATE_SATISFIED_BIT = 1; // LSB
    static const unsigned STATE_SATISFIED_VAL_SHIFT = 1;
    atomic<uint32_t> state;

    long long sleep_timeout;
    long long next_sleep_time;

    // useful for debugging
    uintptr_t owner_tid;
#ifdef REALM_ENABLE_STARVATION_CHECKS
    // lower bound on number of times this doorbell has been passed over
    int starvation_count;

    static atomic<int> starvation_limit;
    void increase_starvation_count(int to_add, void *db_list);
#endif

    friend class DoorbellList;
    Doorbell *next_doorbell;
  };

  // a list of doorbells that are waiting to be run - like an up/down
  //  semaphore, a doorbell list can go "negative" - a pop that preceeds
  //  a push will immediately satisfy the push rather than delaying the popper
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE DoorbellList {
  public:
    DoorbellList();
    ~DoorbellList();

    // returns true if the doorbell is added, false if it consumed an available
    //  "doorbell to be named later" token
    // adding doorbells is always lock-free and can be called concurrently
    bool add_doorbell(Doorbell *db);

    // extracts the a doorbell from the list and returns it, or increments
    //  the "extra notifies" count (if 'allow_extra') and returns nullptr
    // this is lock-free w.r.t. adders, but only one thread may be performing
    //  extraction/notification at a time
    // comes in two flavors, so choose wisely:
    //  a) oldest-first - perhaps more "fair" but more expensive
    //  b) newest-first - easier, and maybe better for cache locality?
    Doorbell *extract_oldest(bool prefer_spinning, bool allow_extra);
    Doorbell *extract_newest(bool prefer_spinning, bool allow_extra);

    // helper routines to extract and notify exactly 'count' doorbells -
    //  TODO: make slightly cheaper than repeated calls to extract
    void notify_oldest(unsigned count, bool prefer_spinning);
    void notify_newest(unsigned count, bool prefer_spinning);

  protected:
    // this field is either:
    //  a) the first Doorbell in the waiting stack, or
    //  b) 2*extra_notifies - 1, if notifies are waiting for matching doorbells
    atomic<uintptr_t> head_or_count;

#ifdef DEBUG_REALM
    MutexChecker mutex_check;
#endif
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE UnfairMutex : public noncopyable {
  public:
    UnfairMutex();

    typedef UnfairCondVar CondVar;

    // these are all inlined and very fast in the non-contended case
    void lock();
    bool trylock();
    void unlock();

  protected:
    // slower lock/unlock paths when contention exists
    void lock_slow();
    void unlock_slow();

    friend class UnfairCondVar;

    // internal state consists of an LSB indicating a lock and the rest of
    //  the bits indicating how many waiters exist - the actual waiters are
    //  (or will eventually be) in the doorbell list
    atomic<uint32_t> state;
    DoorbellList db_list;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE FIFOMutex : public noncopyable {
  public:
    FIFOMutex();

    typedef FIFOCondVar CondVar;

    // these are all inlined and very fast in the non-contended case
    void lock();
    bool trylock();
    void unlock();

  protected:
    // slower lock/unlock paths when contention exists
    void lock_slow();
    void unlock_slow();

    friend class FIFOCondVar;

    // internal state consists of an LSB indicating a lock and the rest of
    //  the bits indicating how many waiters exist - the actual waiters are
    //  (or will eventually be) in the doorbell list
    atomic<uint32_t> state;
    DoorbellList db_list;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE KernelMutex : public noncopyable {
  public:
    KernelMutex();
    ~KernelMutex();

    typedef KernelCondVar CondVar;

    // these call into libpthread (or the equivalent), so YMMV
    void lock(void);
    bool trylock(void);
    void unlock(void);

  protected:
    friend class KernelCondVar;

    // the actual implementation of the mutex is hidden
    //  but if we don't define it here, we run afoul of
    //  rules type-punning, so use macros to let mutex.cc's inclusion
    //  of this file behave a little differently
    union {
      uint64_t placeholder[8]; // 64 bytes, at least 8 byte aligned
#ifdef REALM_KERNEL_MUTEX_IMPL
      REALM_KERNEL_MUTEX_IMPL;
#endif
    };
  };


  // a delegating mutex allows lock-free mutual exclusion between threads that
  //  want to work on a shared resource - a lock attempt either succeeds
  //  immediately or delegates the intended work "units" to some other thread
  //  that already holds the mutex
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE DelegatingMutex {
  public:
    DelegatingMutex();

    // attempts to enter the mutual exclusion zone in order to do 'work_units'
    //  units of work - a non-zero return indicates success with a potentially-
    //  different number of units of work that need to be performed, while a
    //  return of zero indicates contention but a guarantee that some other
    //  thread will perform the work instead
    // a small amount of temporary state must be preserved by the caller across
    //  the call to attempt_enter and any/all calls to attempt_exit (this is
    //  not stored in the object to avoid false sharing)
    uint64_t attempt_enter(uint64_t work_units, uint64_t& tstate);

    // attempts to exit the mutual exclusion zone once all known work has been
    //  performed - a zero return indicates success, while a non-zero return
    //  indicates the caller is still in the mutex and has been given more work
    //  to do
    uint64_t attempt_exit(uint64_t& tstate);

  protected:
    // LSB indicates some thread is in the mutex, while the rest of the bits
    //  accumulate delegated work
    atomic<uint64_t> state;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE UnfairCondVar : public noncopyable {
  public:
    UnfairCondVar(UnfairMutex &_mutex);

    UnfairMutex &mutex;

    // these require that you hold the underlying mutex when you call
    void signal();
    void broadcast();
    void wait();

  protected:
    // no need for atomics here - we're protected by the mutex
    unsigned num_waiters;
    DoorbellList db_list;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE FIFOCondVar : public noncopyable {
  public:
    FIFOCondVar(FIFOMutex &_mutex);

    FIFOMutex &mutex;

    // these require that you hold the underlying mutex when you call
    void signal();
    void broadcast();
    void wait();

  protected:
    // no need for atomics here - we're protected by the mutex
    unsigned num_waiters;
    DoorbellList db_list;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE KernelCondVar : public noncopyable {
  public:
    KernelCondVar(KernelMutex &_mutex);
    ~KernelCondVar(void);

    KernelMutex &mutex;

    // these require that you hold the underlying mutex when you call
    void signal(void);
    void broadcast(void);
    void wait(void);

    // wait with a timeout - returns true if awakened by a signal and
    //  false if the timeout expires first
    bool timedwait(long long max_nsec);

  protected:
    // the actual implementation of the condition variable is hidden
    //  but if we don't define it here, we run afoul of
    //  rules type-punning, so use macros to let mutex.cc's inclusion
    //  of this file behave a little differently
    union {
      uint64_t placeholder[8]; // 64 bytes, at least 8 byte aligned
#ifdef REALM_KERNEL_CONDVAR_IMPL
      REALM_KERNEL_CONDVAR_IMPL;
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
