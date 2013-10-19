/* Copyright 2013 Stanford University
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


#ifndef __LEGION_ATOMICS__
#define __LEGION_ATOMICS__

// These are implementation of common atomics for integers
// that we use in the runtime and are supported by GNU compilers.
// Unfortunately not all of the compilers we use support them
// (suck it PGI and LLVM)!

namespace LegionRuntime {
  namespace LowLevel {

    inline int __sync_fetch_and_add(int *ptr, int value)
    {
      int result;
      asm volatile("lock; xadd %0, %1"
          : "=r"(result), "=m"(*ptr)
          : "0"(value), "m"(*ptr)
          : "memory");
      return result;
    }

    inline long __sync_fetch_and_add(long *ptr, long value)
    {
      long result;
      asm volatile("lock; xadd %0, %1"
          : "=r"(result), "=m"(*ptr)
          : "0"(value), "m"(*ptr)
          : "memory");
      return result;
    }

    inline long long __sync_fetch_and_add(long long *ptr, long long value)
    {
      long result;
      asm volatile("lock; xaddq %0, %1"
          : "=r"(result), "=m"(*ptr)
          : "0"(value), "m"(*ptr)
          : "memory");
      return result;
    }

    inline unsigned int __sync_fetch_and_add(unsigned int *ptr, unsigned int value)
    {
      unsigned int result;
      asm volatile("lock; xadd %0, %1"
          : "=r"(result), "=m"(*ptr)
          : "0"(value), "m"(*ptr)
          : "memory");
      return result;
    }

    inline unsigned long __sync_fetch_and_add(unsigned long *ptr, unsigned long value)
    {
      unsigned long result;
      asm volatile("lock; xadd %0, %1"
          : "=r"(result), "=m"(*ptr)
          : "0"(value), "m"(*ptr)
          : "memory");
      return result;
    }

    inline unsigned long long __sync_fetch_and_add(unsigned long long *ptr, unsigned long long value)
    {
      unsigned long long result;
      asm volatile("lock; xaddq %0, %1"
          : "=r"(result), "=m"(*ptr)
          : "0"(value), "m"(*ptr)
          : "memory");
      return result;
    }

    inline int __sync_add_and_fetch(int *ptr, int value)
    {
      return (__sync_fetch_and_add(ptr, value) + value);
    }

    inline long __sync_add_and_fetch(long *ptr, long value)
    {
      return (__sync_fetch_and_add(ptr, value) + value);
    }

    inline long long __sync_add_and_fetch(long long *ptr, long long value)
    {
      return (__sync_fetch_and_add(ptr, value) + value);
    }

    inline unsigned int __sync_add_and_fetch(unsigned int *ptr, unsigned int value)
    {
      return (__sync_fetch_and_add(ptr, value) + value);
    }

    inline unsigned long __sync_add_and_fetch(unsigned long *ptr, unsigned long value)
    {
      return (__sync_fetch_and_add(ptr, value) + value);
    }

    inline unsigned long long __sync_add_and_fetch(unsigned long long *ptr, unsigned long long value)
    {
      return (__sync_fetch_and_add(ptr, value) + value);
    }

  }; // LowLevel namespace
}; // LegionRuntime namespace 

#endif
