/* Copyright 2015 Stanford University, NVIDIA Corporation
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

// sparsity maps for Realm

// nop, but helps IDEs
#include "sparsity.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMap<N,T>

  template <int N, typename T>
  inline bool SparsityMap<N,T>::operator<(const SparsityMap<N,T> &rhs) const
  {
    return id < rhs.id;
  }

  template <int N, typename T>
  inline bool SparsityMap<N,T>::operator==(const SparsityMap<N,T> &rhs) const
  {
    return id == rhs.id;
  }

  template <int N, typename T>
  inline bool SparsityMap<N,T>::operator!=(const SparsityMap<N,T> &rhs) const
  {
    return id != rhs.id;
  }

  template <int N, typename T>
  inline bool SparsityMap<N,T>::exists(void) const
  {
    return id != 0;
  }

  // looks up the public subset of the implementation object
  template <int N, typename T>
  inline SparsityMapPublicImpl<N,T> *SparsityMap<N,T>::impl(void) const
  {
    return SparsityMapImplBase::lookup(*this);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImplBase

  // cannot be constructed directly
  inline SparsityMapImplBase::SparsityMapImplBase(int _dim, int _idxtype)
    : dim(_dim), idxtype(_idxtype)
  {}
    
  template <int N, typename T>
  /*static*/ SparsityMapPublicImpl<N,T> *SparsityMapImplBase::lookup(SparsityMap<N,T> sparsity)
  {
    SparsityMapImplBase *i = lookup(sparsity.id);
    assert((i->dim == N) && (i->idxtype == (int)(sizeof(T))));
    return static_cast<SparsityMapPublicImpl<N,T> *>(i);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapPublicImpl<N,T>

  template <int N, typename T>
  inline SparsityMapPublicImpl<N,T>::SparsityMapPublicImpl(void)
    : SparsityMapImplBase(N, (int)(sizeof(T)))
  {}


}; // namespace Realm

