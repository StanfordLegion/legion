/* Copyright 2018 Stanford University, NVIDIA Corporation
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
#include "realm/sparsity.h"

#include "realm/serialize.h"

TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::SparsityMap<N,T>);

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

  template <int N, typename T> __CUDA_HD__
  inline bool SparsityMap<N,T>::exists(void) const
  {
    return id != 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapPublicImpl<N,T>

  template <int N, typename T>
  inline bool SparsityMapPublicImpl<N,T>::is_valid(bool precise /*= true*/)
  {
    return (precise ? entries_valid : approx_valid);
  }

  template <int N, typename T>
  inline const std::vector<SparsityMapEntry<N,T> >& SparsityMapPublicImpl<N,T>::get_entries(void)
  {
    if(!entries_valid) {
      // TODO: warn here?
      make_valid(true /*precise*/).wait();
    }
    return entries;
  }
    
  template <int N, typename T>
  inline const std::vector<Rect<N,T> >& SparsityMapPublicImpl<N,T>::get_approx_rects(void)
  {
    if(!approx_valid) {
      // TODO: warn here?
      make_valid(false /*!precise*/).wait();
    }
    return approx_rects;
  }


}; // namespace Realm

