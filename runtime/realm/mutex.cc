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

#include "realm/realm_config.h"

// include gasnet header files before mutex.h to make sure we have
//  definitions for gasnet_hsl_t and gasnett_cond_t
#ifdef REALM_USE_GASNET1

// so OpenMPI borrowed gasnet's platform-detection code and didn't change
//  the define names - work around it by undef'ing anything set via mpi.h
//  before we include gasnet.h
#undef __PLATFORM_COMPILER_GNU_VERSION_STR

#ifndef GASNET_PAR
#define GASNET_PAR
#endif
#include <gasnet.h>

#ifndef GASNETT_THREAD_SAFE
#define GASNETT_THREAD_SAFE
#endif
#include <gasnet_tools.h>

// eliminate GASNet warnings for unused static functions
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning1) = (void *)_gasneti_threadkey_init;
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning2) = (void *)_gasnett_trace_printf_noop;

// can't use gasnet_hsl_t in debug mode
// actually, don't use gasnet_hsl_t at all right now...
//#ifndef GASNET_DEBUG
#if 0
#define USE_GASNET_HSL_T
#endif
#endif

#ifdef USE_GASNET_HSL_T

#define REALM_KERNEL_MUTEX_IMPL   gasnet_hsl_t mutex
#define REALM_KERNEL_CONDVAR_IMPL gasnett_cond_t condvar
// gasnet doesn't provide an rwlock?
#define REALM_RWLOCK_IMPL  pthread_rwlock_t rwlock

#else

#ifdef REALM_ON_WINDOWS
#include <windows.h>
#include <synchapi.h>
#include <processthreadsapi.h>

#define REALM_KERNEL_MUTEX_IMPL   CRITICAL_SECTION mutex
#define REALM_KERNEL_CONDVAR_IMPL CONDITION_VARIABLE condvar
struct RWLockImpl {
  SRWLOCK rwlock;
  bool exclusive;
};
#define REALM_RWLOCK_IMPL  RWLockImpl rwlock
#else
// use pthread mutex/condvar
#include <pthread.h>
#include <errno.h>

#define REALM_KERNEL_MUTEX_IMPL   pthread_mutex_t mutex
#define REALM_KERNEL_CONDVAR_IMPL pthread_cond_t  condvar
#define REALM_RWLOCK_IMPL  pthread_rwlock_t rwlock
#endif
#endif

#include "realm/mutex.h"
#include "realm/timers.h"
#include "realm/faults.h"

#include <assert.h>
#include <new>

#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#include <unistd.h>
#endif

#if defined(REALM_ON_LINUX) && !defined(REALM_NO_USE_FUTEX)
#include <linux/futex.h>
#include <sys/time.h>
#include <sys/syscall.h>
#endif

#ifdef __SSE2__
// technically pause is an "SSE2" instruction, but it's defined in xmmintrin
//  (it'd be fine to call it on older CPUs, but since we're doing conditional
//  compilation, only use it if it does something)
#include <xmmintrin.h>
#endif

#ifdef REALM_ON_WINDOWS
static void sleep(long seconds) { Sleep(seconds * 1000); }
#endif

namespace Realm {

  Logger log_mutex("mutex");


  ////////////////////////////////////////////////////////////////////////
  //
  // class MutexChecker
  //

  namespace {
    // only have one of the lock_fail or unlock_fail threads call abort so
    //  that we don't mess up attempts to dump stack traces (which can take
    //  quite a while)
    atomic<int> abort_count(0);
  };

  void MutexChecker::lock_fail(int actval, CheckedScope *cs)
  {
    {
      LoggerMessage msg = log_mutex.fatal();
      msg << "over limit on entry into MutexChecker("
          << (name ? name : "") << "," << object << ") limit="
          << limit << " actval=" << actval;
      if(cs)
        msg << " on scope(" << (cs->name ? cs->name : "") << "," << cs->object << ")";
      Backtrace bt;
      bt.capture_backtrace();
      bt.lookup_symbols();
      msg << " at " << bt;
    }
    // wait a couple seconds so that threads in the guarded section hopefully
    //  try to leave and declare who they are too
    sleep(2);
    while(abort_count.fetch_add(1) > 0) {
      // if we're in this loop we'll never actually leave
      sleep(60);
    }
    abort();
  }

  void MutexChecker::unlock_fail(int actval, CheckedScope *cs)
  {
    {
      LoggerMessage msg = log_mutex.fatal();
      msg << "over limit on exit of MutexChecker("
          << (name ? name : "") << "," << object << ") limit="
          << limit << " actval=" << actval;
      if(cs)
        msg << " on scope(" << (cs->name ? cs->name : "") << "," << cs->object << ")";
      Backtrace bt;
      bt.capture_backtrace();
      bt.lookup_symbols();
      msg << " at " << bt;
    }
    // wait a couple seconds so that threads in the guarded section hopefully
    //  try to leave and declare who they are too
    sleep(2);
    while(abort_count.fetch_add(1) > 0) {
      // if we're in this loop we'll never actually leave
      sleep(60);
    }
    abort();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Doorbell
  //

  class DoorbellImpl : public Doorbell {
  public:
#if defined(REALM_ON_LINUX) && !defined(REALM_NO_USE_FUTEX)
    // don't need anything for futex beyond Doorbell::state
#elif defined(REALM_ON_WINDOWS)
    // use Win32 Event object
    DoorbellImpl()
    {
      event = CreateEvent(NULL /*default security*/,
                          FALSE /*!manual reset*/,
                          FALSE /*initial state*/,
                          NULL /*unnamed*/);
      assert(event);
    }

    ~DoorbellImpl()
    {
      CloseHandle(event);
    }

    HANDLE event;
#else
    // generic fallback using kernel mutex/condvar pair
    DoorbellImpl() : condvar(mutex), asleep(false) {}

    KernelMutex mutex;
    KernelCondVar condvar;
    bool asleep;
#endif
  };

  namespace ThreadLocal {
    REALM_THREAD_LOCAL Doorbell *my_doorbell = 0;
    REALM_THREAD_LOCAL char doorbell_storage[sizeof(DoorbellImpl)];
  };

  Doorbell::Doorbell()
    : state(STATE_IDLE)
    , sleep_timeout(DOORBELL_SLEEP_DEFAULT)
    , next_sleep_time(-1)
    , owner_tid(0)
#ifdef REALM_ENABLE_STARVATION_CHECKS
    , starvation_count(0)
#endif
    , next_doorbell(nullptr)
  {
#ifdef REALM_USE_PTHREADS
    owner_tid = pthread_self();
#endif
#ifdef REALM_ON_WINDOWS
    owner_tid = reinterpret_cast<uintptr_t>(GetCurrentThread());
#endif
  }

  Doorbell::~Doorbell()
  {
#ifdef DEBUG_REALM
    assert(state.load() == STATE_IDLE);
#endif
  }

  /*static*/ Doorbell *Doorbell::get_thread_doorbell()
  {
    if(!ThreadLocal::my_doorbell) {
      Doorbell *db = new((void *)ThreadLocal::doorbell_storage) DoorbellImpl;
      ThreadLocal::my_doorbell = db;
      return db;
    } else
      return ThreadLocal::my_doorbell;
  }

  uint32_t Doorbell::wait_slow()
  {
    DoorbellImpl *dbi = static_cast<DoorbellImpl *>(this);

    // make sure we're not applying a sleep timer from the previous time
    next_sleep_time = -1;

    uint32_t val = dbi->state.load_acquire();
    while((val & STATE_SATISFIED_BIT) == 0) {
      // decide whether to spin or sleep
      bool spin;
      switch(val) {
      case STATE_PENDING_PREWAKE:
        {
          // we've been told that a signal is imminent, so no sleeping
          spin = true;
          break;
        }

      case STATE_PENDING_AWAKE:
        {
          if(sleep_timeout == DOORBELL_SLEEP_IMMEDIATE) {
            spin = false;
          } else if(sleep_timeout == DOORBELL_SLEEP_NEVER) {
            spin = true;
          } else {
            // on the first iteration, set the next_sleep_time - on others,
            //  check it
            if(next_sleep_time == -1) {
              next_sleep_time = (Clock::current_time_in_nanoseconds() +
                                 sleep_timeout);
              spin = true;
            } else {
              spin = (Clock::current_time_in_nanoseconds() >= next_sleep_time);
            }
          }
          break;
        }

      default:
        {
          fprintf(stderr, "FATAL: unexpected doorbell state: this=%p val=%d\n", this, val);
          abort();
        }
      }

      if(spin) {
        // x86 wants you to insert a PAUSE in any spin loop
        // TODO: see if other processors need something similar
#ifdef __SSE2__
        _mm_pause();
#endif

        val = dbi->state.load_acquire();
        continue;
      }

#if defined(REALM_ON_LINUX) && !defined(REALM_NO_USE_FUTEX)
      // if we can transition from AWAKE->ASLEEP, we ask the kernel to
      //  wait for someone to wake us (if val changes before we get there,
      //   we'll get EAGAIN)
      if(dbi->state.compare_exchange(val, STATE_PENDING_ASLEEP)) {
        errno = 0;
        int ret = syscall(SYS_futex,
                          &dbi->state, FUTEX_WAIT, STATE_PENDING_ASLEEP,
                          nullptr, nullptr, 0);
        // acceptable results are:
        //  ret==0 (_probably_ woken up by another thread, but have to check)
        //  errno=EINTR (_probably_ a spurious wakeup, have to check though)
        //  errno=EAGAIN (FUTEX_WAKE preceded FUTEX_WAIT, no need to check
        //                 state but no harm in it either)
        if(ret < 0) {
          if((errno != EINTR) && (errno != EAGAIN)) {
            log_mutex.fatal() << "unexpected futex_wait return: ret=" << ret << " errno=" << errno;
            abort();
          }
          errno = 0;
        }
        val = state.load_acquire();
        // if for some reason we woke up without state changing from
        //  ASLEEP, try to update it to indicate that we're awake
        if((val == STATE_PENDING_ASLEEP) &&
           (dbi->state.compare_exchange(val, STATE_PENDING_AWAKE))) {
          // update our cached copy too
          val = STATE_PENDING_AWAKE;
        }
      }
#elif defined(REALM_ON_WINDOWS)
      // if we can transition from AWAKE->ASLEEP, we will wait on our
      //  auto-resetting Event object, guaranteeing one signal results in
      //  exactly one wake, no matter which order they happen in
      if(dbi->state.compare_exchange(val, STATE_PENDING_ASLEEP)) {
        DWORD result = WaitForSingleObject(dbi->event, INFINITE);
        assert(result == WAIT_OBJECT_0);
      }
#else
      // attempt to sleep, but the AWAKE->ASLEEP transition has to happen with
      //  the mutex held
      dbi->mutex.lock();
      if(dbi->state.compare_exchange(val, STATE_PENDING_ASLEEP)) {
        dbi->asleep = true;
        while(dbi->asleep)
          dbi->condvar.wait();
        val = dbi->state.load_acquire();
      }
      dbi->mutex.unlock();
#endif
    }
    // uncontested transition back to IDLE
#ifdef DEBUG_REALM
    uint32_t prev = state.exchange(STATE_IDLE);
    assert(prev == val);
    (void)prev;
#else
    state.store(STATE_IDLE);
#endif
    return (val >> STATE_SATISFIED_VAL_SHIFT);
  }

  void Doorbell::notify_slow()
  {
    DoorbellImpl *dbi = static_cast<DoorbellImpl *>(this);

#if defined(REALM_ON_LINUX) && !defined(REALM_NO_USE_FUTEX)
    // tell kernel to wake up sleeper (or not, if we get there first)
    int ret = syscall(SYS_futex, &dbi->state, FUTEX_WAKE, 1,
                      nullptr, nullptr, 0);
    assert(ret >= 0);
#elif defined(REALM_ON_WINDOWS)
    // set the sleeper's event
    DWORD result = SetEvent(dbi->event);
    assert(result);
#else
    // we know there was a sleeper, but condvar rules say we need to take mutex
    //  first
    dbi->mutex.lock();
    assert(dbi->asleep);
    dbi->asleep = false;
    dbi->condvar.signal();
    dbi->mutex.unlock();
#endif
  }

  void Doorbell::prewake_slow()
  {
    // actually the same as notify_slow because we're just interested in
    //  getting the waiter back onto a CPU core
    notify_slow();
  }

#ifdef REALM_ENABLE_STARVATION_CHECKS
  /*static*/ atomic<int> Doorbell::starvation_limit(1000);

  void Doorbell::increase_starvation_count(int to_add, void *db_list)
  {
    starvation_count += to_add;
    int cur_limit = starvation_limit.load();
    if(starvation_count > cur_limit) {
      // only print a warning if we can double the limit
      if(starvation_limit.compare_exchange_relaxed(cur_limit, 2*cur_limit)) {
        Backtrace bt;
        bt.capture_backtrace();
        bt.lookup_symbols();
        log_mutex.warning() << "doorbell starvation limit reached: list="
                            << db_list << " db=" << this << " owner="
                            << std::hex << owner_tid << std::dec
                            << " count=" << starvation_count
                            << " at " << bt;
      }
    }
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class DoorbellList
  //

  DoorbellList::DoorbellList()
    : head_or_count(0)
#ifdef DEBUG_REALM
    , mutex_check("DoorbellList", this)
#endif
  {}

  DoorbellList::~DoorbellList()
  {
#ifdef DEBUG_REALM
    assert(head_or_count.load() == 0);
#endif
  }

  Doorbell *DoorbellList::extract_oldest(bool prefer_spinning, bool allow_extra)
  {
    uintptr_t hoc = head_or_count.load_acquire();
    while((hoc == 0) || ((hoc & 1) != 0)) {
      // list appears to be empty

      // if we're not allowed to add extra entries, we're done
      if(!allow_extra)
        return nullptr;

      // try to increment the count with a compare-and-swap
      uintptr_t newval = ((hoc == 0) ? 1 : (hoc + 2));
      if(head_or_count.compare_exchange(hoc, newval)) {
        // success
        return nullptr;
      } else {
        // failure, but we re-read hoc and will try again
      }
    }

#ifdef DEBUG_REALM
    // would like to guard code above as well, but a successful cmpxchg
    //  immediately enables another thread to enter that code, and we can't
    //  disable the mutex checker fast enough, so we wait until here to
    //  look for illegal concurrence
    MutexChecker::CheckedScope cs(mutex_check, "extract_newest");
#endif

    // if we get here, we know 'hoc' points at a valid doorbell, and even if
    //  other doorbells get added concurrently, we have exclusive ownership of
    //  the "tail" of the list pointed to by 'hoc'
    Doorbell *head = reinterpret_cast<Doorbell *>(hoc);
    Doorbell *chosen;
    Doorbell *chosen_prev = nullptr;

    if(!head->next_doorbell) {
      // no list to walk and no preferences to worry about
      chosen = head;
    } else {
      // walk list to the end, remembering the last spinning one we saw, if
      //  requested
      Doorbell *spinning_prev = nullptr;
      Doorbell *spinning = nullptr;
      Doorbell *cur = head;
      Doorbell *prev = nullptr;
      while(true) {
        if(prefer_spinning && !cur->is_sleeping()) {
          spinning = cur;
          spinning_prev = prev;
        }
        if(cur->next_doorbell) {
          prev = cur;
          cur = cur->next_doorbell;
        } else {
          // end of the line
          break;
        }
      }
      // if we preferred a spinning one, and found one, use it instead of the
      //  tail we walked to
      if(spinning) {
        chosen = spinning;
        chosen_prev = spinning_prev;
#ifdef REALM_ENABLE_STARVATION_CHECKS
        // if we chose something that wasn't oldest, update the starvation
        //  count for the oldest
        if(chosen->next_doorbell)
          chosen->next_doorbell->increase_starvation_count(1 + chosen->starvation_count,
                                                           this);
#endif
      } else {
        chosen = cur;
        chosen_prev = prev;
      }
    }

    if(chosen == head) {
      // removing the head of our list is tricky because more things might have
      //  been added
      uintptr_t expected = hoc;
      uintptr_t newhead = reinterpret_cast<uintptr_t>(head->next_doorbell);
      if(!head_or_count.compare_exchange(expected, newhead)) {
        // more stuff added, but we now own that too, so walk to where 'head'
        //  appears in a next pointer and disconnect it
        Doorbell *cur = reinterpret_cast<Doorbell *>(expected);
        while(cur->next_doorbell != head) {
          assert(cur->next_doorbell);
          cur = cur->next_doorbell;
        }
        cur->next_doorbell = head->next_doorbell;
      }
    } else {
      // we're modifying something inside the list we own, so no need to mess
      //  with the atomic head pointer
      assert(chosen_prev->next_doorbell == chosen);
      chosen_prev->next_doorbell = chosen->next_doorbell;
    }

    chosen->next_doorbell = nullptr;
    return chosen;
  }

  Doorbell *DoorbellList::extract_newest(bool prefer_spinning, bool allow_extra)
  {
    uintptr_t hoc = head_or_count.load_acquire();
    while((hoc == 0) || ((hoc & 1) != 0)) {
      // list appears to be empty

      // if we're not allowed to add extra entries, we're done
      if(!allow_extra)
        return nullptr;

      // try to increment the count with a compare-and-swap
      uintptr_t newval = ((hoc == 0) ? 1 : (hoc + 2));
      if(head_or_count.compare_exchange(hoc, newval)) {
        // success
        return nullptr;
      } else {
        // failure, but we re-read hoc and will try again
      }
    }

#ifdef DEBUG_REALM
    // would like to guard code above as well, but a successful cmpxchg
    //  immediately enables another thread to enter that code, and we can't
    //  disable the mutex checker fast enough, so we wait until here to
    //  look for illegal concurrence
    MutexChecker::CheckedScope cs(mutex_check, "extract_newest");
#endif

    // if we get here, we know 'hoc' points at a valid doorbell, and even if
    //  other doorbells get added concurrently, we have exclusive ownership of
    //  the "tail" of the list pointed to by 'hoc'
    Doorbell *head = reinterpret_cast<Doorbell *>(hoc);

    // if a spinning thread is preferred, we might need a scan
    if(prefer_spinning && head->next_doorbell && head->is_sleeping()) {
      Doorbell *prev = head;
      Doorbell *cur = head->next_doorbell;
      while(cur) {
        if(!cur->is_sleeping()) {
          // we'll take this one, and we can snip it out of the list without
          //  messing with the atomic head pointer
          prev->next_doorbell = cur->next_doorbell;
          cur->next_doorbell = nullptr;
#ifdef REALM_ENABLE_STARVATION_CHECKS
          // we passed over the head of the list (and everything older)
          head->increase_starvation_count(1, this);
#endif
          return cur;
        }
        prev = cur;
        cur = cur->next_doorbell;
      }
    }

    // just take the head off the list, dealing with races with other adders
    uintptr_t expected = hoc;
    uintptr_t newhead = reinterpret_cast<uintptr_t>(head->next_doorbell);
    if(!head_or_count.compare_exchange(expected, newhead)) {
      // more stuff added, but we now own that too, so walk to where 'head'
      //  appears in a next pointer and disconnect it
      Doorbell *cur = reinterpret_cast<Doorbell *>(expected);
      while(cur->next_doorbell != head) {
        assert(cur->next_doorbell);
        cur = cur->next_doorbell;
      }
      cur->next_doorbell = head->next_doorbell;
    }
#ifdef REALM_ENABLE_STARVATION_CHECKS
    if(head->next_doorbell)
      head->next_doorbell->increase_starvation_count(1 + head->starvation_count,
                                                     this);
#endif
    head->next_doorbell = nullptr;
    return head;
  }

  void DoorbellList::notify_oldest(unsigned count, bool prefer_spinning)
  {
    // TODO: do this with a single modification of the head pointer
    for(unsigned i = 0; i < count; i++) {
      Doorbell *db = extract_oldest(prefer_spinning, true /*allow extra*/);
      if(db)
        db->notify(0);
    }
  }

  void DoorbellList::notify_newest(unsigned count, bool prefer_spinning)
  {
    // TODO: do this with a single modification of the head pointer
    for(unsigned i = 0; i < count; i++) {
      Doorbell *db = extract_newest(prefer_spinning, true /*allow extra*/);
      if(db)
        db->notify(0);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DelegatingMutex
  //

  DelegatingMutex::DelegatingMutex()
    : state(0)
  {}

  uint64_t DelegatingMutex::attempt_enter(uint64_t work_units, uint64_t& tstate)
  {
    // step 1: attempt to set LSB - if successful, we've entered mutual
    //  exclusion zone - maintain "tstate" to be the expected contents of
    //  the "state" when we try to exit
    uint64_t orig_state = state.fetch_or_acqrel(1);
    if((orig_state & 1) == 0) {
      tstate = orig_state | 1;
      return (work_units + (orig_state >> 1));
    }

    // step 2: attempt to delegate our work to the current holder of the mutex
    //  if successful (i.e. we observe the LSB set), we return with a count of 0
    orig_state = state.fetch_add_acqrel(work_units << 1);
    if((orig_state & 1) != 0) {
      return 0;
    }

    // step 3: lock wasn't held when we delegated the work, so we need to try
    //  to set the LSB again - if it's already set, delegation was succesful
    //  after all
    orig_state = state.fetch_or_acqrel(1);
    if((orig_state & 1) != 0) {
      return 0;
    }

    // step 4: if we set the bit ourselves, and the count is nonzero, there's
    //  work assigned to us (which may or may not include the work we originally
    //  intended to do)
    if(orig_state != 0) {
      tstate = orig_state | 1;
      return (orig_state >> 1);
    }

    // step 5: count was zero, so try to let go of LSB before anybody gives
    //  us work
    orig_state = 1;
    if(state.compare_exchange(orig_state, 0)) {
      return 0;
    }

    // nope, stuck with some work to do
    tstate = orig_state;
    return (orig_state >> 1);
  }

  uint64_t DelegatingMutex::attempt_exit(uint64_t& tstate)
  {
    // tstate is tracking the last value we observed (or put) in 'state' - if
    //   we can CAS that back to 0, we're done
    uint64_t expected = tstate;
    if(state.compare_exchange(expected, 0)) {
      return 0;
    }

    // nope, new work has been delegated to us - return the difference between
    //  what we knew of before and the new total
#ifdef DEBUG_REALM
    assert(expected > tstate);
#endif
    uint64_t new_work = (expected - tstate) >> 1;
    tstate = expected;
    return new_work;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UnfairMutex
  //

  void UnfairMutex::lock_slow()
  {
    // already tried and failed to take the lock, so plan on incrementing
    //  the waiter count, but only do it if the lock is held by somebody
    // (if we observe that the lock might not be held at all, try to grab
    //  it)
    uint32_t val = state.load();
    while(true) {
      if((val & 1) != 0) {
        if(state.compare_exchange(val, val+2)) {
          // successfully added ourselves as a waiter - add our doorbell
          //  to the list
          Doorbell *db = Doorbell::get_thread_doorbell();
          db->prepare();
          if(db_list.add_doorbell(db)) {
            // if we went into the list, the result value of wait tells us
            //  whether we have been given the lock directly by the previous
            //  holder or if we have just been awakened to try again
            int lock_transfer = db->wait();
            if(lock_transfer) {
#ifdef DEBUG_REALM
              // lock better be held!
              val = state.load();
              assert((val & 1) != 0);
#endif
              return;
            } else {
              val = state.fetch_or_acqrel(1);
              if((val & 1) == 0) {
                // success!
                return;
              }
            }
          } else {
            // we consumed an "extra notify token", which ALWAYS carries a lock
            //  grant
            db->cancel();
            return;
          }
        }
      } else {
        // NOTE: it's actually ok for val to be nonzero here (i.e. not locked
        //  but with pending waiters), because there's at least one thread
        //  (i.e. us) that's trying to get in and will then be responsible for
        //  waking up the existing waiters
        // this means we need to use an atomic OR instead of CAS
        val = state.fetch_or_acqrel(1);
        if((val & 1) == 0) {
          // managed to grab the lock without waiting
          return;
        }
      }
    }
  }

  void UnfairMutex::unlock_slow()
  {
    // we know there's at least one waiter, but we don't know if they are
    //  awake (in which case we want to transfer mutex ownership) or if they
    //  are asleep (in which case we don't), so try to grab a doorbell off the
    //  list but do NOT add an extra token yet, since we have to update the
    //  state before the ownership can be transferred
    Doorbell *db = db_list.extract_newest(true /*prefer_spinning*/,
                                          false /*!allow_extra*/);

    // if we got a sleeping waiter, we're not actually going to transfer the
    //  mutex ownership (because it might take them a while to wake up), so
    //  decrement both the waiter count AND the lock held LSB
    if(db && db->is_sleeping()) {
      // since we're releasing the lock to an unknown thread, this needs to
      //  have release semantics
      uint32_t prev = state.fetch_sub_acqrel(3);
      assert(((prev & 1) != 0) && (prev >= 3));
      (void)prev;

      db->notify(0 /*!lock_transfer*/);
    } else {
      // if we got a spinning waiter or none at all (i.e. extra token case),
      //  we'll transfer the lock to them, so just decrement the waiter count
      uint32_t prev = state.fetch_sub(2);
      assert(((prev & 1) != 0) && (prev >= 3));
      (void)prev;

      // reattempt extract if we didn't get one before
      if(!db)
        db = db_list.extract_newest(true /*prefer_spinning*/,
                                    true /*allow_extra*/);
      // an extra token implicitly carried the lock_transfer bit
      if(db)
        db->notify(1 /*lock_transfer*/);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FIFOMutex
  //

  void FIFOMutex::lock_slow()
  {
    // already tried and failed to take the lock, so plan on incrementing
    //  the waiter count, but only do it if the lock is held by somebody
    // (if we observe that the lock might not be held at all, try to grab
    //  it)
    uint32_t val = state.load();
    while(true) {
      if((val & 1) != 0) {
        if(state.compare_exchange(val, val+2)) {
          // successfully added ourselves as a waiter - add our doorbell
          //  to the list, and when we wake, we have the lock
          Doorbell *db = Doorbell::get_thread_doorbell();
          db->prepare();
          if(db_list.add_doorbell(db))
            db->wait();
          else
            db->cancel();
          return;
        }
      } else {
        assert(val == 0);
        if(state.compare_exchange(val, 1)) {
          // managed to grab the lock without waiting
          return;
        }
      }
    }
  }

  void FIFOMutex::unlock_slow()
  {
    // decrement the waiter count and then pass the lock on to whoever we
    //  get from the doorbell list
    uint32_t prev = state.fetch_sub(2);
    assert(((prev & 1) != 0) && (prev >= 3));
    (void)prev;

    Doorbell *db = db_list.extract_oldest(false /*!prefer_spinning*/,
                                          true /*allow_extra*/);
    if(db)
      db->notify(0);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class KernelMutex
  //

  KernelMutex::KernelMutex(void)
  {
    assert(sizeof(mutex) <= sizeof(placeholder));
#ifdef USE_GASNET_HSL_T
    gasnet_hsl_init(&mutex);
#else
#ifdef REALM_ON_WINDOWS
    InitializeCriticalSection(&mutex);
#else
    pthread_mutex_init(&mutex, 0);
#endif
#endif
  }

  KernelMutex::~KernelMutex(void)
  {
#ifdef USE_GASNET_HSL_T
    gasnet_hsl_destroy(&mutex);
#else
#ifdef REALM_ON_WINDOWS
    DeleteCriticalSection(&mutex);
#else
    pthread_mutex_destroy(&mutex);
#endif
#endif
  }

  void KernelMutex::lock(void)
  {
#ifdef USE_GASNET_HSL_T
    gasnet_hsl_lock(&mutex);
#else
#ifdef REALM_ON_WINDOWS
    EnterCriticalSection(&mutex);
#else
    pthread_mutex_lock(&mutex);
#endif
#endif
  }

  void KernelMutex::unlock(void)
  {
#ifdef USE_GASNET_HSL_T
    gasnet_hsl_unlock(&mutex);
#else
#ifdef REALM_ON_WINDOWS
    LeaveCriticalSection(&mutex);
#else
    pthread_mutex_unlock(&mutex);
#endif
#endif
  }

  bool KernelMutex::trylock(void)
  {
#ifdef USE_GASNET_HSL_T
    int ret = gasnet_hsl_trylock(&mutex);
    return (ret == 0);
#else
#ifdef REALM_ON_WINDOWS
    BOOL ret = TryEnterCriticalSection(&mutex);
    return (ret != 0);
#else
    int ret = pthread_mutex_trylock(&mutex);
    return (ret == 0);
#endif
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UnfairCondVar
  //

  UnfairCondVar::UnfairCondVar(UnfairMutex &_mutex)
    : mutex(_mutex)
    , num_waiters(0)
  {}

  void UnfairCondVar::signal()
  {
    // we hold the lock, so both 'num_waiters' are stable
    if(num_waiters > 0) {
      num_waiters--;
      Doorbell *db = db_list.extract_newest(true /*prefer_spinning*/,
                                            false /*!allow_extra*/);
      // a waiter had to add themselves before letting go of the lock, so
      //  this should never fail
      assert(db);
      // now that we have the signal-ee, don't actually ring the doorbell -
      //  just re-insert this doorbell on the mutex's waiter list
      uint32_t mutex_prev = mutex.state.fetch_add(2);
      assert((mutex_prev & 1) != 0); // i.e. we hold the lock
      // similarly, this should never see a wake-before-wait case
      bool ok = mutex.db_list.add_doorbell(db);
      assert(ok);
      (void)ok;
    } else {
      // no waiters, so signal() is a nop
    }
  }

  void UnfairCondVar::broadcast()
  {
    // we hold the lock, so both 'num_waiters' are stable - requeue every
    //  thread waitingon the condvar on the mutex instead
    while(num_waiters > 0) {
      num_waiters--;
      // NOTE: dequeue oldest-first so that the newest thing in our list
      //  becomes the newest thing in the mutex's list
      Doorbell *db = db_list.extract_oldest(false /*!prefer_spinning*/,
                                            false /*!allow_extra*/);
      // a waiter had to add themselves before letting go of the lock, so
      //  this should never fail
      assert(db);
      // now that we have the signal-ee, don't actually ring the doorbell -
      //  just re-insert this doorbell on the mutex's waiter list
      uint32_t mutex_prev = mutex.state.fetch_add(2);
      assert((mutex_prev & 1) != 0); // i.e. we hold the lock
      // similarly, this should never see a wake-before-wait case
      bool ok = mutex.db_list.add_doorbell(db);
      assert(ok);
      (void)ok;
    }
  }

  void UnfairCondVar::wait()
  {
    // we currently hold the lock, so we can safely increase the waiter
    //  count and add our doorbell to the list
    Doorbell *db = Doorbell::get_thread_doorbell();
    db->prepare();
    num_waiters++;
    bool ok = db_list.add_doorbell(db);
    // condvar doorbell list should never go negative
    assert(ok);
    (void)ok;

    // now release the mutex and wait on our doorbell - we will not be
    //  awoken until we've been moved to the mutex list and then maybe given
    //  the mutex
    mutex.unlock();
    int lock_transfer = db->wait();
    if(!lock_transfer) {
      // have to explicitly reacquire the lock
      mutex.lock();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FIFOCondVar
  //

  FIFOCondVar::FIFOCondVar(FIFOMutex &_mutex)
    : mutex(_mutex)
    , num_waiters(0)
  {}

  void FIFOCondVar::signal()
  {
    // we hold the lock, so both 'num_waiters' are stable
    if(num_waiters > 0) {
      num_waiters--;
      Doorbell *db = db_list.extract_oldest(false /*!prefer_spinning*/,
                                            false /*!allow_extra*/);
      // a waiter had to add themselves before letting go of the lock, so
      //  this should never fail
      assert(db);
      // now that we have the signal-ee, don't actually ring the doorbell -
      //  just re-insert this doorbell on the mutex's waiter list
      uint32_t mutex_prev = mutex.state.fetch_add(2);
      assert((mutex_prev & 1) != 0); // i.e. we hold the lock
      // similarly, this should never see a wake-before-wait case
      bool ok = mutex.db_list.add_doorbell(db);
      assert(ok);
      (void)ok;
    } else {
      // no waiters, so signal() is a nop
    }
  }

  void FIFOCondVar::broadcast()
  {
    // we hold the lock, so both 'num_waiters' are stable - requeue every
    //  thread waitingon the condvar on the mutex instead
    while(num_waiters > 0) {
      num_waiters--;
      Doorbell *db = db_list.extract_oldest(false /*!prefer_spinning*/,
                                            false /*!allow_extra*/);
      // a waiter had to add themselves before letting go of the lock, so
      //  this should never fail
      assert(db);
      // now that we have the signal-ee, don't actually ring the doorbell -
      //  just re-insert this doorbell on the mutex's waiter list
      uint32_t mutex_prev = mutex.state.fetch_add(2);
      assert((mutex_prev & 1) != 0); // i.e. we hold the lock
      // similarly, this should never see a wake-before-wait case
      bool ok = mutex.db_list.add_doorbell(db);
      assert(ok);
      (void)ok;
    }
  }

  void FIFOCondVar::wait()
  {
    // we currently hold the lock, so we can safely increase the waiter
    //  count and add our doorbell to the list
    Doorbell *db = Doorbell::get_thread_doorbell();
    db->prepare();
    num_waiters++;
    bool ok = db_list.add_doorbell(db);
    // condvar doorbell list should never go negative
    assert(ok);
    (void)ok;

    // now release the mutex and wait on our doorbell - we will not be
    //  awoken until we've been moved to the mutex list and then given
    //  the mutex
    mutex.unlock();
    db->wait();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class KernelCondVar
  //

  KernelCondVar::KernelCondVar(KernelMutex &_mutex)
    : mutex(_mutex)
  {
    assert(sizeof(condvar) <= sizeof(placeholder));
#ifdef USE_GASNET_HSL_T
    gasnett_cond_init(&condvar);
#else
#ifdef REALM_ON_WINDOWS
    InitializeConditionVariable(&condvar);
#else
    pthread_cond_init(&condvar, 0);
#endif
#endif
  }

  KernelCondVar::~KernelCondVar(void)
  {
#ifdef USE_GASNET_HSL_T
    gasnett_cond_destroy(&condvar);
#else
#ifdef REALM_ON_WINDOWS
    // no destructor on windows?
#else
    pthread_cond_destroy(&condvar);
#endif
#endif
  }

  // these require that you hold the lock when you call
  void KernelCondVar::signal(void)
  {
#ifdef USE_GASNET_HSL_T
    gasnett_cond_signal(&condvar);
#else
#ifdef REALM_ON_WINDOWS
    WakeConditionVariable(&condvar);
#else
    pthread_cond_signal(&condvar);
#endif
#endif
  }

  void KernelCondVar::broadcast(void)
  {
#ifdef USE_GASNET_HSL_T
    gasnett_cond_broadcast(&condvar);
#else
#ifdef REALM_ON_WINDOWS
    WakeAllConditionVariable(&condvar);
#else
    pthread_cond_broadcast(&condvar);
#endif
#endif
  }

  void KernelCondVar::wait(void)
  {
#ifdef USE_GASNET_HSL_T
    gasnett_cond_wait(&condvar, &(mutex.mutex.lock));
#else
#ifdef REALM_ON_WINDOWS
    SleepConditionVariableCS(&condvar, &mutex.mutex, INFINITE);
#else
    pthread_cond_wait(&condvar, &mutex.mutex);
#endif
#endif
  }

  // wait with a timeout - returns true if awakened by a signal and
  //  false if the timeout expires first
  bool KernelCondVar::timedwait(long long max_nsec)
  {
#ifdef USE_GASNET_HSL_T
    assert(0 && "gasnett_cond_t doesn't have timedwait!");
#else
#ifdef REALM_ON_WINDOWS
    BOOL ret = SleepConditionVariableCS(&condvar, &mutex.mutex, max_nsec / 1000000LL);
    return (ret != 0);
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += (ts.tv_nsec + max_nsec) / 1000000000LL;
    ts.tv_nsec = (ts.tv_nsec + max_nsec) % 1000000000LL;
    int ret = pthread_cond_timedwait(&condvar, &mutex.mutex, &ts);
    if(ret == ETIMEDOUT) return false;
    // TODO: check other error codes?
    return true;
#endif
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RWLock
  //

  RWLock::RWLock(void)
    : writer(*this)
    , reader(*this)
  {
    assert(sizeof(rwlock) <= sizeof(placeholder));
#ifdef REALM_ON_WINDOWS
    InitializeSRWLock(&rwlock.rwlock);
#else
    pthread_rwlock_init(&rwlock, 0);
#endif
  }

  RWLock::~RWLock(void)
  {
#ifdef REALM_ON_WINDOWS
    // no destructor on windows?
#else
    pthread_rwlock_destroy(&rwlock);
#endif
  }

  void RWLock::wrlock(void)
  {
#ifdef REALM_ON_WINDOWS
    AcquireSRWLockExclusive(&rwlock.rwlock);
    rwlock.exclusive = true;
#else
    pthread_rwlock_wrlock(&rwlock);
#endif
  }

  void RWLock::rdlock(void)
  {
#ifdef REALM_ON_WINDOWS
    AcquireSRWLockShared(&rwlock.rwlock);
    rwlock.exclusive = false;
#else
    pthread_rwlock_rdlock(&rwlock);
#endif
  }

  void RWLock::unlock(void)
  {
#ifdef REALM_ON_WINDOWS
    if(rwlock.exclusive)
      ReleaseSRWLockExclusive(&rwlock.rwlock);
    else
      ReleaseSRWLockShared(&rwlock.rwlock);
#else
    pthread_rwlock_unlock(&rwlock);
#endif
  }

  bool RWLock::trywrlock(void)
  {
#ifdef REALM_ON_WINDOWS
    BOOL ret = TryAcquireSRWLockExclusive(&rwlock.rwlock);
    if(ret != 0) {
      rwlock.exclusive = true;
      return true;
    } else
      return false;
#else
    int ret = pthread_rwlock_trywrlock(&rwlock);
    return (ret == 0);
#endif
  }

  bool RWLock::tryrdlock(void)
  {
#ifdef REALM_ON_WINDOWS
    BOOL ret = TryAcquireSRWLockShared(&rwlock.rwlock);
    if(ret != 0) {
      rwlock.exclusive = false;
      return true;
    } else
      return false;
#else
    int ret = pthread_rwlock_trywrlock(&rwlock);
    return (ret == 0);
#endif
  }


};
