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

// NOP, but helps IDEs
#include "realm/mutex.h"

#ifdef DEBUG_REALM
#include <assert.h>
#endif

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class AutoLock<LT>
  //

  template <typename LT>
  inline AutoLock<LT>::AutoLock(LT& mutex)
    : mutex(mutex), held(true)
  { 
    mutex.lock();
  }

  template <typename LT>
  inline AutoLock<LT>::~AutoLock(void) 
  {
    if(held)
      mutex.unlock();
  }

  template <typename LT>
  inline void AutoLock<LT>::release(void)
  {
#ifdef DEBUG_REALM
    assert(held);
#endif
    mutex.unlock();
    held = false;
  }

  template <typename LT>
  inline void AutoLock<LT>::reacquire(void)
  {
#ifdef DEBUG_REALM
    assert(!held);
#endif
    mutex.lock();
    held = true;
  }

  
};
