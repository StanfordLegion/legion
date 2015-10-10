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

// index spaces for Realm

// nop, but helps IDEs
#include "indexspace.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ZPoint<N,T>

  template <int N, typename T>
  inline ZPoint<N,T>::ZPoint(void)
  {}

  template <int N, typename T>
  inline T& ZPoint<N,T>::operator[](int index)
  {
    return (&x)[index];
  }

  template <int N, typename T>
  inline const T& ZPoint<N,T>::operator[](int index) const
  {
    return (&x)[index];
  }

  // specializations for N <= 4
  template <typename T>
  struct ZPoint<1,T> {
    T x;
    ZPoint(void) {}
    ZPoint(T _x) : x(_x) {}
    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }
  };

  template <typename T>
  struct ZPoint<2,T> {
    T x, y;
    ZPoint(void) {}
    ZPoint(T _x, T _y) : x(_x), y(_y) {}
    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }
  };

  template <typename T>
  struct ZPoint<3,T> {
    T x, y, z;
    ZPoint(void) {}
    ZPoint(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }
  };

  template <typename T>
  struct ZPoint<4,T> {
    T x, y, z, w;
    ZPoint(void) {}
    ZPoint(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }
  };

  // component-wise operators defined on Point<N,T>
  template <int N, typename T> 
  inline bool operator==(const ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    for(int i = 0; i < N; i++) if(lhs[i] != rhs[i]) return false;
    return true;
  }
    
  template <int N, typename T>
  inline bool operator!=(const ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    for(int i = 0; i < N; i++) if(lhs[i] != rhs[i]) return true;
    return false;
  }

  template <int N, typename T>
  inline ZPoint<N,T> operator+(const ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] + rhs[i];
    return out;
  }

  template <int N, typename T>
  inline ZPoint<N,T>& operator+=(ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] += rhs[i];
  }

  template <int N, typename T>
  inline ZPoint<N,T> operator-(const ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] - rhs[i];
    return out;
  }

  template <int N, typename T>
  inline ZPoint<N,T>& operator-=(ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] -= rhs[i];
  }

  template <int N, typename T>
  inline ZPoint<N,T> operator*(const ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] * rhs[i];
    return out;
  }

  template <int N, typename T>
  inline ZPoint<N,T>& operator*=(ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] *= rhs[i];
  }

  template <int N, typename T>
  inline ZPoint<N,T> operator/(const ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] / rhs[i];
    return out;
  }

  template <int N, typename T>
  inline ZPoint<N,T>& operator/=(ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] /= rhs[i];
  }

  template <int N, typename T>
  inline ZPoint<N,T> operator%(const ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] % rhs[i];
    return out;
  }

  template <int N, typename T>
  inline ZPoint<N,T>& operator%=(ZPoint<N,T>& lhs, const ZPoint<N,T>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] %= rhs[i];
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZRect<N,T>

  template <int N, typename T>
  inline ZRect<N,T>::ZRect(void)
  {}

  template <int N, typename T>
  inline ZRect<N,T>::ZRect(const ZPoint<N,T>& _lo, const ZPoint<N,T>& _hi)
    : lo(_lo), hi(_hi)
  {}

  template <int N, typename T>
  inline bool ZRect<N,T>::empty(void) const
  {
    for(int i = 0; i < N; i++) if(lo[i] > hi[i]) return true;
    return false;
  }

  template <int N, typename T>
  size_t ZRect<N,T>::volume(void) const
  {
    size_t v = 1;
    for(int i = 0; i < N; i++)
      if(lo[i] > hi[i])
	return 0;
      else
	v *= (size_t)(hi[i] - lo[i] + 1);
    return v;
  }

  // true if all points in other are in this rectangle
  template <int N, typename T>
  bool ZRect<N,T>::contains(const ZRect<N,T>& other) const
  {
    // containment is weird w.r.t. emptiness: if other is empty, the answer is
    //  always true - if we're empty, the answer is false, unless other was empty
    // this means we can early-out with true if other is empty, but have to remember
    //  our emptiness separately
    bool ctns = true;
    for(int i = 0; i < N; i++) {
      if(other.lo[i] > other.hi[i]) return true; // other is empty
      // now that we know the other is nonempty, we need:
      //  lo[i] <= other.lo[i] ^ other.hi[i] <= hi[i]
      // (which can only be satisfied if we're non-empty)
      if((lo[i] > other.lo[i]) || (other.hi[i] > hi[i]))
	ctns = false;
    }
    return ctns;
  }

  // true if there are any points in the intersection of the two rectangles
  template <int N, typename T>
  bool ZRect<N,T>::overlaps(const ZRect<N,T>& other) const
  {
    // overlapping requires there be an element that lies in both ranges, which
    //  is equivalent to saying that both lo's are <= both hi's - this catches
    //  cases where either rectangle is empty
    for(int i = 0; i < N; i++)
      if((lo[i] > hi[i]) || (lo[i] > other.hi[i]) ||
	 (other.lo[i] > hi[i]) || (other.lo[i] > other.hi[i])) return false;
    return true;
  }

  template <int N, typename T>
  ZRect<N,T> ZRect<N,T>::intersection(const ZRect<N,T>& other) const
  {
    ZRect<N,T> out;
    for(int i = 0; i < N; i++) {
      out.lo[i] = std::max(lo[i], other.lo[i]);
      out.hi[i] = std::min(hi[i], other.hi[i]);
    }
    return out;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZIndexSpace<N,T>

  template <int N, typename T>
  inline ZIndexSpace<N,T>::ZIndexSpace(void)
  {}

  template <int N, typename T>
  inline ZIndexSpace<N,T>::ZIndexSpace(const ZRect<N,T>& _bounds)
    : bounds(_bounds)
  {
    sparsity.id = 0;
  }

  template <int N, typename T>
  inline ZIndexSpace<N,T>::ZIndexSpace(const ZRect<N,T>& _bounds, SparsityMap<N,T> _sparsity)
    : bounds(_bounds), sparsity(_sparsity)
  {}

  // true if we're SURE that there are no points in the space (may be imprecise due to
  //  lazy loading of sparsity data)
  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::empty(void) const
  {
    return bounds.empty();
  }
    
  // true if there is no sparsity map (i.e. the bounds fully define the domain)
  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::dense(void) const
  {
    return !sparsity.exists();
  }


}; // namespace Realm

