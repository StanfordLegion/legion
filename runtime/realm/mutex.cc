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

#define REALM_MUTEX_IMPL   gasnet_hsl_t mutex
#define REALM_CONDVAR_IMPL gasnett_cond_t condvar
// gasnet doesn't provide an rwlock?
#define REALM_RWLOCK_IMPL  pthread_rwlock_t rwlock

#else

#ifdef REALM_ON_WINDOWS
#include <windows.h>
#include <synchapi.h>

#define REALM_MUTEX_IMPL   CRITICAL_SECTION mutex
#define REALM_CONDVAR_IMPL CONDITION_VARIABLE condvar
struct RWLockImpl {
  SRWLOCK rwlock;
  bool exclusive;
};
#define REALM_RWLOCK_IMPL  RWLockImpl rwlock
#else
// use pthread mutex/condvar
#include <pthread.h>
#include <errno.h>

#define REALM_MUTEX_IMPL   pthread_mutex_t mutex
#define REALM_CONDVAR_IMPL pthread_cond_t  condvar
#define REALM_RWLOCK_IMPL  pthread_rwlock_t rwlock
#endif
#endif

#include "realm/mutex.h"

#include <assert.h>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Mutex
  //

  Mutex::Mutex(void)
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

  Mutex::~Mutex(void)
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

  void Mutex::lock(void)
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

  void Mutex::unlock(void)
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

  bool Mutex::trylock(void)
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
  // class CondVar
  //

  CondVar::CondVar(Mutex &_mutex) 
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

  CondVar::~CondVar(void)
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
  void CondVar::signal(void)
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

  void CondVar::broadcast(void)
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

  void CondVar::wait(void)
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
  bool CondVar::timedwait(long long max_nsec)
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
