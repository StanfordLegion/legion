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

// implementation of sparsity maps

#ifndef REALM_DEPPART_SPARSITY_IMPL_INL
#define REALM_DEPPART_SPARSITY_IMPL_INL

// NOP, but useful for IDEs
#include "realm/deppart/sparsity_impl.h"

#include "realm/runtime_impl.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMap<N,T>

  // not inlined due to circular include file hell
#if 0
  // looks up the public subset of the implementation object
  template <int N, typename T>
  SparsityMapPublicImpl<N,T> *SparsityMap<N,T>::impl(void) const
  {
    SparsityMapImplWrapper *wrapper = get_runtime()->get_sparsity_impl(*this);
    return wrapper->get_or_create<N,T>(*this);
  }
#endif

}; // namespace Realm

#endif // REALM_DEPPART_SPARSITY_IMPL_INL
