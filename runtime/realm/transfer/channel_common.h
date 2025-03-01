/* Copyright 2024 Stanford University
 * Copyright 2024 Los Alamos National Laboratory
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
#ifndef CHANNEL_COMMON
#define CHANNEL_COMMON

#include <vector>
#include <cstddef>
#include <cstdint>

#include "realm/memory.h"

namespace Realm {

  struct Node;

  void memcpy_1d(uintptr_t dst_base, uintptr_t src_base, size_t bytes);

  void memcpy_2d(uintptr_t dst_base, uintptr_t dst_lstride, uintptr_t src_base,
                 uintptr_t src_lstride, size_t bytes, size_t lines);

  void memcpy_3d(uintptr_t dst_base, uintptr_t dst_lstride, uintptr_t dst_pstride,
                 uintptr_t src_base, uintptr_t src_lstride, uintptr_t src_pstride,
                 size_t bytes, size_t lines, size_t planes);

  void memset_1d(uintptr_t dst_base, size_t bytes, const void *fill_data,
                 size_t fill_size);

  void memset_2d(uintptr_t dst_base, uintptr_t dst_lstride, size_t bytes, size_t lines,
                 const void *fill_data, size_t fill_size);

  void memset_3d(uintptr_t dst_base, uintptr_t dst_lstride, uintptr_t dst_pstride,
                 size_t bytes, size_t lines, size_t planes, const void *fill_data,
                 size_t fill_size);

  void enumerate_local_cpu_memories(const Node *node, std::vector<Memory> &mems);

}; // namespace Realm

#endif
