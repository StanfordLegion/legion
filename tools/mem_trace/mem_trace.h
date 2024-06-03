/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#ifndef __MEM_TRACE_H__
#define __MEM_TRACE_H__

#include <inttypes.h>

enum AllocKind
{
  MALLOC_KIND = 1,
  CALLOC_KIND = 2,
  REALLOC_KIND = 3,
  MEMALIGN_KIND = 4,
  POSIXMEMALIGN_KIND = 5,
  VALLOC_KIND = 6,
  FREE_KIND = 7
};

typedef struct Alloc {
  float time;
  AllocKind kind;
  size_t size;
  void *ptr;
  uintptr_t hash;
  Alloc() {};
  Alloc(float time, AllocKind kind, size_t size, void *ptr, uintptr_t hash)
    : time(time), kind(kind), size(size), ptr(ptr), hash(hash)
  {};
} Alloc;

#endif
