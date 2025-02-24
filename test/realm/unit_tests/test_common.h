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

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "realm/inst_impl.h"
#include "realm/inst_layout.h"
#include <utility>

namespace Realm {

  template <int N, typename T>
  RegionInstanceImpl *create_inst(Rect<N, T> bounds,
                                  const std::vector<FieldID> &field_ids,
                                  const std::vector<size_t> &field_sizes);

template <typename Func, size_t... Is>
void dispatch_for_dimension(int dim, Func &&func, std::index_sequence<Is...>);

} // namespace Realm

#include "test_common.inl"

#endif // TEST_COMMON_H
