
/* Copyright 2024 NVIDIA Corporation
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

#ifndef SPINLOCK_H
#define SPINLOCK_H

#include <pthread.h>

namespace Realm {
namespace UCP {

  class SpinLock {
  public:
    SpinLock()
    {
      int ret = pthread_spin_init(&pth_spinlock, PTHREAD_PROCESS_PRIVATE);
      assert(ret == 0);
    }

    ~SpinLock()
    {
      int ret = pthread_spin_destroy(&pth_spinlock);
      assert(ret == 0);
    }

    int lock()
    {
      return pthread_spin_lock(&pth_spinlock);
    }

    int unlock()
    {
      return pthread_spin_unlock(&pth_spinlock);
    }

    int trylock()
    {
      if (pthread_spin_trylock(&pth_spinlock) == 0) return 1;
      return 0;
    }

  private:
    pthread_spinlock_t pth_spinlock;
  };

}; // namespace UCP

}; // namespace Realm

#endif
