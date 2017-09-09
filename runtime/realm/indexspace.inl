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

//include "instance.h"
//include "inst_layout.h"

#include "serialize.h"
#include "logging.h"

TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::ZPoint<N,T>);
TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::ZRect<N,T>);
TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::ZIndexSpace<N,T>);

namespace Realm {

  extern Logger log_dpops;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZPoint<N,T>

  template <int N, typename T> __CUDA_HD__
  inline ZPoint<N,T>::ZPoint(void)
  {}

  template <int N, typename T> __CUDA_HD__
  inline ZPoint<N,T>::ZPoint(const T vals[N])
  {
    for(int i = 0; i < N; i++)
      (&x)[i] = vals[i];
  }

  template <int N, typename T>
  template <typename T2> __CUDA_HD__
  inline ZPoint<N,T>::ZPoint(const ZPoint<N,T2>& copy_from)
  {
    for(int i = 0; i < N; i++)
      (&x)[i] = (&copy_from.x)[i];
  }

  template <int N, typename T>
  template <typename T2> __CUDA_HD__
  inline ZPoint<N,T>& ZPoint<N,T>::operator=(const ZPoint<N,T2>& copy_from)
  {
    for(int i = 0; i < N; i++)
      (&x)[i] = (&copy_from.x)[i];
    return *this;
  }

  template <int N, typename T> __CUDA_HD__
  inline T& ZPoint<N,T>::operator[](int index)
  {
    return (&x)[index];
  }

  template <int N, typename T> __CUDA_HD__
  inline const T& ZPoint<N,T>::operator[](int index) const
  {
    return (&x)[index];
  }

  template <int N, typename T>
  template <typename T2> __CUDA_HD__
  inline T ZPoint<N,T>::dot(const ZPoint<N,T2>& rhs) const
  {
    T acc = x * rhs.x;
    for(int i = 1; i < N; i++)
      acc += (&x)[i] * (&rhs.x)[i];
    return acc;
  }

  // specializations for N <= 4
  template <typename T>
  struct ZPoint<1,T> {
    T x;
    __CUDA_HD__
    ZPoint(void) {}
    // No need for a static array constructor here
    __CUDA_HD__
    ZPoint(T _x) : x(_x) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    ZPoint(const ZPoint<1, T2>& copy_from) : x(copy_from.x) {}
    template <typename T2> __CUDA_HD__
    ZPoint<1,T>& operator=(const ZPoint<1, T2>& copy_from)
    {
      x = copy_from.x;
      return *this;
    }

    __CUDA_HD__
    T& operator[](int index) { return (&x)[index]; }
    __CUDA_HD__
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2> __CUDA_HD__
    T dot(const ZPoint<1, T2>& rhs) const
    {
      return (x * rhs.x);
    }

    // special case: for N == 1, we're willing to coerce to T
    __CUDA_HD__
    operator T(void) const { return x; }
  };

  template <typename T>
  struct ZPoint<2,T> {
    T x, y;
    __CUDA_HD__
    ZPoint(void) {}
    __CUDA_HD__
    explicit ZPoint(const T vals[2]) : x(vals[0]), y(vals[1]) {}
    __CUDA_HD__
    ZPoint(T _x, T _y) : x(_x), y(_y) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    ZPoint(const ZPoint<2, T2>& copy_from)
      : x(copy_from.x), y(copy_from.y) {}
    template <typename T2> __CUDA_HD__
    ZPoint<2,T>& operator=(const ZPoint<2,T2>& copy_from)
    {
      x = copy_from.x;
      y = copy_from.y;
      return *this;
    }

    __CUDA_HD__
    T& operator[](int index) { return (&x)[index]; }
    __CUDA_HD__
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2> __CUDA_HD__
    T dot(const ZPoint<2, T2>& rhs) const
    {
      return (x * rhs.x) + (y * rhs.y);
    }
  };

  template <typename T>
  struct ZPoint<3,T> {
    T x, y, z;
    __CUDA_HD__
    ZPoint(void) {}
    __CUDA_HD__
    explicit ZPoint(const T vals[3]) : x(vals[0]), y(vals[1]), z(vals[2]) {}
    __CUDA_HD__
    ZPoint(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    ZPoint(const ZPoint<3, T2>& copy_from)
      : x(copy_from.x), y(copy_from.y), z(copy_from.z) {}
    template <typename T2> __CUDA_HD__
    ZPoint<3,T>& operator=(const ZPoint<3,T2>& copy_from)
    {
      x = copy_from.x;
      y = copy_from.y;
      z = copy_from.z;
      return *this;
    }

    __CUDA_HD__
    T& operator[](int index) { return (&x)[index]; }
    __CUDA_HD__
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2> __CUDA_HD__
    T dot(const ZPoint<3, T2>& rhs) const
    {
      return (x * rhs.x) + (y * rhs.y) + (z * rhs.z);
    }
  };

  template <typename T>
  struct ZPoint<4,T> {
    T x, y, z, w;
    __CUDA_HD__
    ZPoint(void) {}
    __CUDA_HD__
    explicit ZPoint(const T vals[4]) : x(vals[0]), y(vals[1]), z(vals[2]), w(vals[3]) {}
    __CUDA_HD__
    ZPoint(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    ZPoint(const ZPoint<4, T2>& copy_from)
      : x(copy_from.x), y(copy_from.y), z(copy_from.z), w(copy_from.w) {}
    template <typename T2> __CUDA_HD__
    ZPoint<4,T>& operator=(const ZPoint<4,T2>& copy_from)
    {
      x = copy_from.x;
      y = copy_from.y;
      z = copy_from.z;
      w = copy_from.w;
      return *this;
    }

    __CUDA_HD__
    T& operator[](int index) { return (&x)[index]; }
    __CUDA_HD__
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2> __CUDA_HD__
    T dot(const ZPoint<4, T2>& rhs) const
    {
      return (x * rhs.x) + (y * rhs.y) + (z * rhs.z) + (w * rhs.w);
    }
  };

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const ZPoint<N,T>& p)
  {
    os << '<' << p[0];
    for(int i = 1; i < N; i++)
      os << ',' << p[i];
    os << '>';
    return os;
  }

  // component-wise operators defined on Point<N,T> (with optional coercion)
  template <int N, typename T, typename T2> __CUDA_HD__
  inline bool operator==(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) if(lhs[i] != rhs[i]) return false;
    return true;
  }
    
  template <int N, typename T, typename T2> __CUDA_HD__
  inline bool operator!=(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) if(lhs[i] != rhs[i]) return true;
    return false;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T> operator+(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] + rhs[i];
    return out;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T>& operator+=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] += rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T> operator-(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] - rhs[i];
    return out;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T>& operator-=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] -= rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T> operator*(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] * rhs[i];
    return out;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T>& operator*=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] *= rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T> operator/(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] / rhs[i];
    return out;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T>& operator/=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] /= rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T> operator%(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] % rhs[i];
    return out;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<N,T>& operator%=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] %= rhs[i];
    return lhs;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZRect<N,T>

  template <int N, typename T> __CUDA_HD__
  inline ZRect<N,T>::ZRect(void)
  {}

  template <int N, typename T> __CUDA_HD__
  inline ZRect<N,T>::ZRect(const ZPoint<N,T>& _lo, const ZPoint<N,T>& _hi)
    : lo(_lo), hi(_hi)
  {}

  template <int N, typename T>
  template <typename T2> __CUDA_HD__
  inline ZRect<N,T>::ZRect(const ZRect<N, T2>& copy_from)
    : lo(copy_from.lo), hi(copy_from.hi)
  {}

  template <int N, typename T>
  template <typename T2> __CUDA_HD__
  inline ZRect<N,T>& ZRect<N,T>::operator=(const ZRect<N, T2>& copy_from)
  {
    lo = copy_from.lo;
    hi = copy_from.hi;
    return *this;
  }

  template <int N, typename T> __CUDA_HD__
  inline /*static*/ ZRect<N,T> ZRect<N,T>::make_empty(void)
  {
    ZRect<N,T> r;
    T v = T(); // assume any user-defined default constructor initializes things
    for(int i = 0; i < N; i++) r.hi[i] = v;
    ++v;
    for(int i = 0; i < N; i++) r.lo[i] = v;
    return r;
  }

  template <int N, typename T> __CUDA_HD__
  inline bool ZRect<N,T>::empty(void) const
  {
    for(int i = 0; i < N; i++) if(lo[i] > hi[i]) return true;
    return false;
  }

  template <int N, typename T> __CUDA_HD__
  inline size_t ZRect<N,T>::volume(void) const
  {
    size_t v = 1;
    for(int i = 0; i < N; i++)
      if(lo[i] > hi[i])
	return 0;
      else
	v *= (size_t)(hi[i] - lo[i] + 1);
    return v;
  }

  template <int N, typename T> __CUDA_HD__
  inline bool ZRect<N,T>::contains(const ZPoint<N,T>& p) const
  {
    for(int i = 0; i < N; i++)
      if((p[i] < lo[i]) || (p[i] > hi[i])) return false;
    return true;
  }

  // true if all points in other are in this rectangle
  template <int N, typename T> __CUDA_HD__
  inline bool ZRect<N,T>::contains(const ZRect<N,T>& other) const
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

  // true if all points in other are in this rectangle
  // FIXME: the bounds of an index space aren't necessarily tight - is that ok?
  template <int N, typename T> __CUDA_HD__
  inline bool ZRect<N,T>::contains(const ZIndexSpace<N,T>& other) const
  {
    return contains(other.bounds);
  }

  // true if there are any points in the intersection of the two rectangles
  template <int N, typename T> __CUDA_HD__
  inline bool ZRect<N,T>::overlaps(const ZRect<N,T>& other) const
  {
    // overlapping requires there be an element that lies in both ranges, which
    //  is equivalent to saying that both lo's are <= both hi's - this catches
    //  cases where either rectangle is empty
    for(int i = 0; i < N; i++)
      if((lo[i] > hi[i]) || (lo[i] > other.hi[i]) ||
	 (other.lo[i] > hi[i]) || (other.lo[i] > other.hi[i])) return false;
    return true;
  }

  template <int N, typename T> __CUDA_HD__
  inline ZRect<N,T> ZRect<N,T>::intersection(const ZRect<N,T>& other) const
  {
    ZRect<N,T> out;
    for(int i = 0; i < N; i++) {
      out.lo[i] = (lo[i] < other.lo[i]) ? other.lo[i] : lo[i]; // max
      out.hi[i] = (hi[i] < other.hi[i]) ? hi[i] : other.hi[i]; // min
    }
    return out;
  };

  template <int N, typename T> __CUDA_HD__
  inline ZRect<N,T> ZRect<N,T>::union_bbox(const ZRect<N,T>& other) const
  {
    if(empty()) return other;
    if(other.empty()) return *this;
    // the code below only works if both rectangles are non-empty
    ZRect<N,T> out;
    for(int i = 0; i < N; i++) {
      out.lo[i] = (lo[i] < other.lo[i]) ? lo[i] : other.lo[i]; // min
      out.hi[i] = (hi[i] < other.hi[i]) ? other.hi[i] : hi[i]; // max
    }
    return out;
  };

  // copy and fill operations (wrappers for ZIndexSpace versions)
  template <int N, typename T>
  inline Event ZRect<N,T>::fill(const std::vector<CopySrcDstField> &dsts,
				const ProfilingRequestSet &requests,
				const void *fill_value, size_t fill_value_size,
				Event wait_on /*= Event::NO_EVENT*/) const
  {
    return ZIndexSpace<N,T>(*this).fill(dsts, requests,
					fill_value, fill_value_size,
					wait_on);
  }

  template <int N, typename T>
  inline Event ZRect<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				const ProfilingRequestSet &requests,
				Event wait_on /*= Event::NO_EVENT*/,
				ReductionOpID redop_id /*= 0*/,
				bool red_fold /*= false*/) const
  {
    return ZIndexSpace<N,T>(*this).copy(srcs, dsts,
					requests, wait_on,
					redop_id, red_fold);
  }

  template <int N, typename T>
  inline Event ZRect<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				const ZIndexSpace<N,T> &mask,
				const ProfilingRequestSet &requests,
				Event wait_on /*= Event::NO_EVENT*/,
				ReductionOpID redop_id /*= 0*/,
				bool red_fold /*= false*/) const
  {
    return ZIndexSpace<N,T>(*this).copy(srcs, dsts, mask,
					requests, wait_on,
					redop_id, red_fold);
  }

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const ZRect<N,T>& p)
  {
    os << p.lo << ".." << p.hi;
    return os;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline bool operator==(const ZRect<N,T>& lhs, const ZRect<N,T2>& rhs)
  {
    return (lhs.lo == rhs.lo) && (lhs.hi == rhs.hi);
  }
    
  template <int N, typename T, typename T2> __CUDA_HD__
  inline bool operator!=(const ZRect<N,T>& lhs, const ZRect<N,T2>& rhs)
  {
    return (lhs.lo != rhs.lo) || (lhs.hi != rhs.hi);
  }

  // rectangles may be displaced by a vector (i.e. point)
  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZRect<N,T> operator+(const ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    return ZRect<N,T>(lhs.lo + rhs, lhs.hi + rhs);
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZRect<N,T>& operator+=(ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    lhs.lo += rhs;
    lhs.hi += rhs;
    return lhs;
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZRect<N,T> operator-(const ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    return ZRect<N,T>(lhs.lo - rhs, lhs.hi - rhs);
  }

  template <int N, typename T, typename T2> __CUDA_HD__
  inline ZRect<N,T>& operator-=(ZRect<N,T>& lhs, const ZRect<N,T2>& rhs)
  {
    lhs.lo -= rhs;
    lhs.hi -= rhs;
    return lhs;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZMatrix<M,N,T>

  template <int M, int N, typename T> __CUDA_HD__
  inline ZMatrix<M,N,T>::ZMatrix(void)
  {}

  // copies allow type coercion (assuming the underlying type does)
  template <int M, int N, typename T>
  template <typename T2> __CUDA_HD__
  inline ZMatrix<M,N,T>::ZMatrix(const ZMatrix<M, N, T2>& copy_from)
  {
    for(int i = 0; i < M; i++)
      rows[i] = copy_from[i];
  }
  
  template <int M, int N, typename T>
  template <typename T2> __CUDA_HD__
  inline ZMatrix<M, N, T>& ZMatrix<M,N,T>::operator=(const ZMatrix<M, N, T2>& copy_from)
  {
    for(int i = 0; i < M; i++)
      rows[i] = copy_from[i];
    return *this;
  }

  template <int M, int N, typename T, typename T2> __CUDA_HD__
  inline ZPoint<M, T> operator*(const ZMatrix<M, N, T>& m, const ZPoint<N, T2>& p)
  {
    ZPoint<M,T> out;
    for(int j = 0; j < M; j++)
      out[j] = m.rows[j].dot(p);
    return out;
  }

  template <int M, int N, typename T> __CUDA_HD__
  inline ZPoint<N, T>& ZMatrix<M,N,T>::operator[](int index)
  {
    return rows[index];
  }

  template <int M, int N, typename T> __CUDA_HD__
  inline const ZPoint<N, T>& ZMatrix<M,N,T>::operator[](int index) const
  {
    return rows[index];
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ZPointInRectIterator<N,T>
  
  template <int N, typename T> __CUDA_HD__
  inline ZPointInRectIterator<N,T>::ZPointInRectIterator(void)
    : valid(false)
  {}

  template <int N, typename T> __CUDA_HD__
  inline ZPointInRectIterator<N,T>::ZPointInRectIterator(const ZRect<N,T>& _r,
							 bool _fortran_order /*= true*/)
    : p(_r.lo), valid(!_r.empty()), rect(_r), fortran_order(_fortran_order)
  {}

  template <int N, typename T> __CUDA_HD__
  inline bool ZPointInRectIterator<N,T>::step(void)
  {
    assert(valid);  // can't step an iterator that's already done
    if(N == 1) {
      // 1-D doesn't care about fortran/C order
      if(p.x < rect.hi.x) {
	p.x++;
	return true;
      }
    } else {
      if(fortran_order) {
	// do dimensions in increasing order
	for(int i = 0; i < N; i++) {
	  if(p[i] < rect.hi[i]) {
	    p[i]++;
	    return true;
	  }
	  p[i] = rect.lo[i];
	}
      } else {
	// do dimensions in decreasing order
	for(int i = N - 1; i >= 0; i--) {
	  if(p[i] < rect.hi[i]) {
	    p[i]++;
	    return true;
	  }
	  p[i] = rect.lo[i];
	}
      }
    }
    // if we fall through, we're out of points
    valid = false;
    return false;
  }


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

  // construct an index space from a list of points or rects
  template <int N, typename T>
  inline ZIndexSpace<N,T>::ZIndexSpace(const std::vector<ZPoint<N,T> >& points)
  {
    if(points.empty()) {
      sparsity.id = 0;
      for(int i = 0; i < N; i++) {
	bounds.lo[i] = 1;
	bounds.hi[i] = 0;
      }
    } else {
      bounds.lo = points[0];
      bounds.hi = points[0];
      if(points.size() == 1) {
	// single point can easily be stored precisely
	sparsity.id = 0;
      } else {
	// more than one point may need a sparsity mask
	for(size_t i = 1; i < points.size(); i++)
	  bounds = bounds.union_bbox(ZRect<N,T>(points[i], points[i]));
	sparsity = SparsityMap<N,T>::construct(points, false /*!always_create*/);
      }
    }
    log_dpops.info() << "construct: " << *this;
  }

  template <int N, typename T>
  inline ZIndexSpace<N,T>::ZIndexSpace(const std::vector<ZRect<N,T> >& rects)
  {
    if(rects.empty()) {
      sparsity.id = 0;
      for(int i = 0; i < N; i++) {
	bounds.lo[i] = 1;
	bounds.hi[i] = 0;
      }
    } else {
      bounds = rects[0];
      if(rects.size() == 1) {
	// single rect can easily be stored precisely
	sparsity.id = 0;
      } else {
	// more than one rect may need a sparsity mask
	for(size_t i = 1; i < rects.size(); i++)
	  bounds = bounds.union_bbox(rects[i]);
	sparsity = SparsityMap<N,T>::construct(rects, false /*!always_create*/);
      }
    }
    log_dpops.info() << "construct: " << *this;
  }

  // constructs a guaranteed-empty index space
  template <int N, typename T>
  inline /*static*/ ZIndexSpace<N,T> ZIndexSpace<N,T>::make_empty(void)
  {
    return ZIndexSpace<N,T>(ZRect<N,T>::make_empty());
  }

  // reclaim any physical resources associated with this index space
  //  will clear the sparsity map of this index space if it exists
  template <int N, typename T>
  inline void ZIndexSpace<N,T>::destroy(Event wait_on /*= Event::NO_EVENT*/)
  {}

  // true if we're SURE that there are no points in the space (may be imprecise due to
  //  lazy loading of sparsity data)
  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::empty(void) const
  {
    return bounds.empty();
  }
    
  // true if there is no sparsity map (i.e. the bounds fully define the domain)
  template <int N, typename T>  __CUDA_HD__
  inline bool ZIndexSpace<N,T>::dense(void) const
  {
    return !sparsity.exists();
  }

  // kicks off any operation needed to get detailed sparsity information - asking for
  //  approximate data can be a lot quicker for complicated index spaces
  template <int N, typename T>
  inline Event ZIndexSpace<N,T>::make_valid(bool precise /*= true*/) const
  {
    if(sparsity.exists())
      return sparsity.impl()->make_valid(precise);
    else
      return Event::NO_EVENT;
  }

  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::is_valid(bool precise /*= true*/) const
  {
    if(sparsity.exists())
      return sparsity.impl()->is_valid(precise);
    else
      return true;
  }

  // returns the tightest description possible of the index space
  // if 'precise' is false, the sparsity map may be preserved even for dense
  //  spaces
  template <int N, typename T>
  ZIndexSpace<N,T> ZIndexSpace<N,T>::tighten(bool precise /*= true*/) const
  {
    if(sparsity.exists()) {
      SparsityMapPublicImpl<N,T> *impl = sparsity.impl();

      // if we don't have the data, we need to wait for it
      if(!impl->is_valid(precise)) {
	impl->make_valid(precise).wait();
      }

      // always use precise info if it's available
      if(impl->is_valid(true /*precise*/)) {
	ZIndexSpace<N,T> result;
	const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
	// three cases:
	// 1) empty index space
	if(entries.empty()) {
	  result = ZIndexSpace<N,T>::make_empty();
	} else

	// 2) single dense rectangle
	if((entries.size() == 1) &&
	   !entries[0].sparsity.exists() && (entries[0].bitmap == 0)) {
	  result = ZIndexSpace<N,T>(bounds.intersection(entries[0].bounds));
	} else

	// 3) anything else - walk rectangles and count/union those that
	//   overlap our bounds - if only 1, we can drop the sparsity map
	{
	  size_t overlap_count = 0;
	  bool need_sparsity = false;
	  result = ZIndexSpace<N,T>::make_empty();
	  for(size_t i = 0; i < entries.size(); i++) {
	    ZRect<N,T> isect = bounds.intersection(entries[i].bounds);
	    if(!isect.empty()) {
	      overlap_count++;
	      result.bounds = result.bounds.union_bbox(isect);
	      if(entries[i].sparsity.exists() || (entries[i].bitmap != 0))
		need_sparsity = true;
	    }
	  }
	  if((overlap_count > 1) || need_sparsity)
	    result.sparsity = sparsity;
	}

	log_dpops.info() << "tighten: " << *this << " = " << result;
	return result;
      } else {
	const std::vector<ZRect<N,T> >& approx_rects = impl->get_approx_rects();

	// two cases:
	// 1) empty index space
	if(approx_rects.empty()) {
	  ZRect<N,T> empty;
	  empty.hi = bounds.lo;
	  for(int i = 0; i < N; i++)
	    empty.lo[i] = empty.hi[i] + 1;
	  return ZIndexSpace<N,T>(empty);
	}

	// 2) anything else - keep the sparsity map but tighten the bounds,
	//   respecting the previous bounds
	ZRect<N,T> bbox = bounds.intersection(approx_rects[0]);
	for(size_t i = 1; i < approx_rects.size(); i++)
	  bbox = bbox.union_bbox(bounds.intersection(approx_rects[i]));
	return ZIndexSpace<N,T>(bbox, sparsity);
      }
    } else
      return *this;
  }


  // helper function that binary searches a (1-D) sparsity map entry list and returns
  //  the index of the entry that contains the point, or the first one to appear after
  //  that point
  template <int N, typename T>
  static int bsearch_map_entries(const std::vector<SparsityMapEntry<N,T> >& entries,
				 const ZPoint<N,T>& p)
  {
    assert(N == 1);
    // search range at any given time is [lo, hi)
    int lo = 0;
    int hi = entries.size();
    while(lo < hi) {
      int mid = (lo + hi) >> 1;  // rounding down keeps up from picking hi
      if(p.x < entries[mid].bounds.lo.x) 
	hi = mid;
      else if(p.x > entries[mid].bounds.hi.x)
	lo = mid + 1;
      else
	return mid;
    }
    return lo;
  }   

  // queries for individual points or rectangles
  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::contains(const ZPoint<N,T>& p) const
  {
    // test on bounding box first
    if(!bounds.contains(p))
      return false;

    // if it's a dense rectangle, no further tests
    if(dense())
      return true;

    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
    if(N == 1) {
      // binary search to find the element we want
      int idx = bsearch_map_entries<N,T>(entries, p);
      if(idx >= (int)(entries.size())) return false;

      const SparsityMapEntry<N,T>& e = entries[idx];

      // the search guaranteed we're below the upper bound of the returned entry,
      //  but we might be below the lower bound
      if(p.x < e.bounds.lo.x)
	return false;

      if(e.sparsity.exists()) {
	assert(0);
      }
      if(e.bitmap != 0) {
	assert(0);
      }
      return true;
    } else {
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	  it != entries.end();
	  it++) {
	if(!it->bounds.contains(p)) continue;
	if(it->sparsity.exists()) {
	  assert(0);
	} else if(it->bitmap != 0) {
	  assert(0);
	} else {
	  return true;
	}
      }
    }

    // no entries matched, so the point is not contained in this space
    return false;
  }

  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::contains_all(const ZRect<N,T>& r) const
  {
    // test on bounding box first
    if(!bounds.contains(r))
      return false;

    if(!dense()) {
      // test against sparsity map too
      assert(0);
    }

    return true;
  }

  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::contains_any(const ZRect<N,T>& r) const
  {
    // test on bounding box first
    if(!bounds.overlaps(r))
      return false;

    if(!dense()) {
      // test against sparsity map too
      SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
      const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	  it != entries.end();
	  it++) {
	if(!it->bounds.overlaps(r)) continue;
	if(it->sparsity.exists()) {
	  assert(0);
	} else if(it->bitmap != 0) {
	  assert(0);
	} else {
	  return true;
	}
      }
      
      return false;
    }

    return true;
  }

  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::overlaps(const ZIndexSpace<N,T>& other) const
  {
    if(dense()) {
      if(other.dense()) {
	// just test bounding boxes
	return bounds.overlaps(other.bounds);
      } else {
	// have the other guy test against our bounding box
	return other.contains_any(bounds);
      }
    } else {
      if(other.dense()) {
	return contains_any(other.bounds);
      } else {
	// nasty case - both sparse
	assert(0);
	return true;
      }
    }
  }

  // actual number of points in index space (may be less than volume of bounding box)
  template <int N, typename T>
  inline size_t ZIndexSpace<N,T>::volume(void) const
  {
    if(dense())
      return bounds.volume();

    size_t total = 0;
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
    for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	it != entries.end();
	it++) {
      ZRect<N,T> isect = bounds.intersection(it->bounds);
      if(isect.empty())
	continue;
      if(it->sparsity.exists()) {
	assert(0);
      } else if(it->bitmap != 0) {
	assert(0);
      } else {
	total += isect.volume();
      }
    }

    return total;
  }

  // approximate versions of the above queries - the approximation is guaranteed to be a supserset,
  //  so if contains_approx returns false, contains would too
  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::contains_approx(const ZPoint<N,T>& p) const
  {
    // test on bounding box first
    if(!bounds.contains(p))
      return false;

    // if it's a dense rectangle, no further tests
    if(dense())
      return true;

    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<ZRect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<ZRect<N,T> >::const_iterator it = approx_rects.begin();
	it != approx_rects.end();
	it++)
      if(it->contains(p))
	return true;

    // no entries matched, so the point is definitely not contained in this space
    return false;
  }

  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::contains_all_approx(const ZRect<N,T>& r) const
  {
    // test on bounding box first
    if(!bounds.contains(r))
      return false;

    // if it's a dense rectangle, no further tests
    if(dense())
      return true;

    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<ZRect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<ZRect<N,T> >::const_iterator it = approx_rects.begin();
	it != approx_rects.end();
	it++) {
      if(it->contains(r))
	return true;
      if(it->overlaps(r))
	assert(0);
    }

    // no entries matched, so the point is definitely not contained in this space
    return false;
  }

  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::contains_any_approx(const ZRect<N,T>& r) const
  {
    // test on bounding box first
    if(!bounds.overlaps(r))
      return false;

    // if it's a dense rectangle, no further tests
    if(dense())
      return true;

    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<ZRect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<ZRect<N,T> >::const_iterator it = approx_rects.begin();
	it != approx_rects.end();
	it++) {
      if(it->overlaps(r))
	return true;
    }

    // no entries matched, so the point is definitely not contained in this space
    return false;
  }

  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::overlaps_approx(const ZIndexSpace<N,T>& other) const
  {
    if(dense()) {
      if(other.dense()) {
	// just test bounding boxes
	return bounds.overlaps(other.bounds);
      } else {
	// have the other guy test against our bounding box
	return other.contains_any_approx(bounds);
      }
    } else {
      if(other.dense()) {
	return contains_any_approx(other.bounds);
      } else {
	// nasty case - both sparse
	assert(0);
	return true;
      }
    }
  }

  // approximage number of points in index space (may be less than volume of bounding box, but larger than
  //   actual volume)
  template <int N, typename T>
  inline size_t ZIndexSpace<N,T>::volume_approx(void) const
  {
    if(dense())
      return bounds.volume();

    size_t total = 0;
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<ZRect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<ZRect<N,T> >::const_iterator it = approx_rects.begin();
	it != approx_rects.end();
	it++)
      total += it->volume();

    return total;
  }

  // copy and fill operations

  template <int N, typename T>
  inline Event ZIndexSpace<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
				      const std::vector<CopySrcDstField> &dsts,
				      const ZIndexSpace<N,T> &mask,
				      const ProfilingRequestSet &requests,
				      Event wait_on /*= Event::NO_EVENT*/,
				      ReductionOpID redop_id /*= 0*/,
				      bool red_fold /*= false*/) const
  {
    assert(0);
    return wait_on;
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <typename FT>
  inline Event ZIndexSpace<N,T>::create_subspace_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
							  FT color,
							  ZIndexSpace<N,T>& subspace,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<FT> colors(1, color);
    std::vector<ZIndexSpace<N,T> > subspaces;
    Event e = create_subspaces_by_field(field_data, colors, subspaces, reqs, wait_on);
    subspace = subspaces[0];
    return e;
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event ZIndexSpace<N,T>::create_subspace_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& field_data,
							  const ZIndexSpace<N2,T2>& source,
							  ZIndexSpace<N,T>& image,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<ZIndexSpace<N2,T2> > sources(1, source);
    std::vector<ZIndexSpace<N,T> > images;
    Event e = create_subspaces_by_image(field_data, sources, images, reqs, wait_on);
    image = images[0];
    return e;
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event ZIndexSpace<N,T>::create_subspace_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZRect<N,T> > >& field_data,
							  const ZIndexSpace<N2,T2>& source,
							  ZIndexSpace<N,T>& image,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<ZIndexSpace<N2,T2> > sources(1, source);
    std::vector<ZIndexSpace<N,T> > images;
    Event e = create_subspaces_by_image(field_data, sources, images, reqs, wait_on);
    image = images[0];
    return e;
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event ZIndexSpace<N,T>::create_subspace_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > >& field_data,
							     const ZIndexSpace<N2,T2>& target,
							     ZIndexSpace<N,T>& preimage,
							     const ProfilingRequestSet &reqs,
							     Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<ZIndexSpace<N2,T2> > targets(1, target);
    std::vector<ZIndexSpace<N,T> > preimages;
    Event e = create_subspaces_by_preimage(field_data, targets, preimages, reqs, wait_on);
    preimage = preimages[0];
    return e;
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event ZIndexSpace<N,T>::create_subspace_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZRect<N2,T2> > >& field_data,
							     const ZIndexSpace<N2,T2>& target,
							     ZIndexSpace<N,T>& preimage,
							     const ProfilingRequestSet &reqs,
							     Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<ZIndexSpace<N2,T2> > targets(1, target);
    std::vector<ZIndexSpace<N,T> > preimages;
    Event e = create_subspaces_by_preimage(field_data, targets, preimages, reqs, wait_on);
    preimage = preimages[0];
    return e;
  }

  // simple wrappers for the multiple subspace version
  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_union(const ZIndexSpace<N,T>& lhs,
							  const ZIndexSpace<N,T>& rhs,
							  ZIndexSpace<N,T>& result,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > lhss(1, lhs);
    std::vector<ZIndexSpace<N,T> > rhss(1, rhs);
    std::vector<ZIndexSpace<N,T> > results;
    Event e = compute_unions(lhss, rhss, results, reqs, wait_on);
    result = results[0];
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_unions(const ZIndexSpace<N,T>& lhs,
							   const std::vector<ZIndexSpace<N,T> >& rhss,
							   std::vector<ZIndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > lhss(1, lhs);
    Event e = compute_unions(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_unions(const std::vector<ZIndexSpace<N,T> >& lhss,
							   const ZIndexSpace<N,T>& rhs,
							   std::vector<ZIndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > rhss(1, rhs);
    Event e = compute_unions(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  // simple wrappers for the multiple subspace version
  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_intersection(const ZIndexSpace<N,T>& lhs,
								 const ZIndexSpace<N,T>& rhs,
								 ZIndexSpace<N,T>& result,
								 const ProfilingRequestSet &reqs,
								 Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > lhss(1, lhs);
    std::vector<ZIndexSpace<N,T> > rhss(1, rhs);
    std::vector<ZIndexSpace<N,T> > results;
    Event e = compute_intersections(lhss, rhss, results, reqs, wait_on);
    result = results[0];
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_intersections(const ZIndexSpace<N,T>& lhs,
								  const std::vector<ZIndexSpace<N,T> >& rhss,
								  std::vector<ZIndexSpace<N,T> >& results,
								  const ProfilingRequestSet &reqs,
								  Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > lhss(1, lhs);
    Event e = compute_intersections(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_intersections(const std::vector<ZIndexSpace<N,T> >& lhss,
								  const ZIndexSpace<N,T>& rhs,
								  std::vector<ZIndexSpace<N,T> >& results,
								  const ProfilingRequestSet &reqs,
								  Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > rhss(1, rhs);
    Event e = compute_intersections(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  // simple wrappers for the multiple subspace version
  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_difference(const ZIndexSpace<N,T>& lhs,
							       const ZIndexSpace<N,T>& rhs,
							       ZIndexSpace<N,T>& result,
							       const ProfilingRequestSet &reqs,
							       Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > lhss(1, lhs);
    std::vector<ZIndexSpace<N,T> > rhss(1, rhs);
    std::vector<ZIndexSpace<N,T> > results;
    Event e = compute_differences(lhss, rhss, results, reqs, wait_on);
    result = results[0];
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_differences(const ZIndexSpace<N,T>& lhs,
								const std::vector<ZIndexSpace<N,T> >& rhss,
								std::vector<ZIndexSpace<N,T> >& results,
								const ProfilingRequestSet &reqs,
								Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > lhss(1, lhs);
    Event e = compute_differences(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event ZIndexSpace<N,T>::compute_differences(const std::vector<ZIndexSpace<N,T> >& lhss,
								const ZIndexSpace<N,T>& rhs,
								std::vector<ZIndexSpace<N,T> >& results,
								const ProfilingRequestSet &reqs,
								Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<ZIndexSpace<N,T> > rhss(1, rhs);
    Event e = compute_differences(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const ZIndexSpace<N,T>& is)
  {
    os << "IS:" << is.bounds;
    if(is.dense()) {
      os << ",dense";
    } else {
      os << ",sparse(" << is.sparsity << ")";
    }
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZIndexSpaceIterator<N,T>

  template <int N, typename T>
  inline ZIndexSpaceIterator<N,T>::ZIndexSpaceIterator(void)
    : valid(false)
  {}

  template <int N, typename T>
  inline ZIndexSpaceIterator<N,T>::ZIndexSpaceIterator(const ZIndexSpace<N,T>& _space)
    : valid(false)
  {
    reset(_space);
  }

  template <int N, typename T>
  inline ZIndexSpaceIterator<N,T>::ZIndexSpaceIterator(const ZIndexSpace<N,T>& _space,
						       const ZRect<N,T>& _restrict)
    : valid(false)
  {
    reset(_space, _restrict);
  }

  template <int N, typename T>
  inline void ZIndexSpaceIterator<N,T>::reset(const ZIndexSpace<N,T>& _space)
  {
    space = _space;
    restriction = space.bounds;
    if(restriction.empty()) {
      valid = false;
      return;
    }
    if(space.dense()) {
      valid = true;
      rect = restriction;
      s_impl = 0;
    } else {
      s_impl = space.sparsity.impl();
      const std::vector<SparsityMapEntry<N,T> >& entries = s_impl->get_entries();
      // find the first entry that overlaps our restriction - speed this up with a
      //  binary search on the low end of the restriction if we're 1-D
      if(N == 1)
	cur_entry = bsearch_map_entries(entries, restriction.lo);
      else
	cur_entry = 0;

      while(cur_entry < entries.size()) {
	const SparsityMapEntry<N,T>& e = entries[cur_entry];
	rect = restriction.intersection(e.bounds);
	if(!rect.empty()) {
	  assert(!e.sparsity.exists());
	  assert(e.bitmap == 0);
	  valid = true;
	  return;
	}
	cur_entry++;
      }
      // if we fall through, there was no intersection
      valid = false;
    }
  }

  template <int N, typename T>
  inline void ZIndexSpaceIterator<N,T>::reset(const ZIndexSpace<N,T>& _space,
					      const ZRect<N,T>& _restrict)
  {
    space = _space;
    restriction = space.bounds.intersection(_restrict);
    if(restriction.empty()) {
      valid = false;
      return;
    }
    if(space.dense()) {
      valid = true;
      rect = restriction;
      s_impl = 0;
    } else {
      s_impl = space.sparsity.impl();
      const std::vector<SparsityMapEntry<N,T> >& entries = s_impl->get_entries();
      // find the first entry that overlaps our restriction - speed this up with a
      //  binary search on the low end of the restriction if we're 1-D
      if(N == 1)
	cur_entry = bsearch_map_entries(entries, restriction.lo);
      else
	cur_entry = 0;

      while(cur_entry < entries.size()) {
	const SparsityMapEntry<N,T>& e = entries[cur_entry];
	rect = restriction.intersection(e.bounds);
	if(!rect.empty()) {
	  assert(!e.sparsity.exists());
	  assert(e.bitmap == 0);
	  valid = true;
	  return;
	}
	cur_entry++;
      }
      // if we fall through, there was no intersection
      valid = false;
    }
  }

  // steps to the next subrect, returning true if a next subrect exists
  template <int N, typename T>
  inline bool ZIndexSpaceIterator<N,T>::step(void)
  {
    assert(valid);  // can't step an interator that's already done

    // a dense space is covered in the first step
    if(!s_impl) {
      valid = false;
      return false;
    }

    // TODO: handle iteration within a sparsity entry

    // move onto the next sparsity entry (that overlaps our restriction)
    const std::vector<SparsityMapEntry<N,T> >& entries = s_impl->get_entries();
    for(cur_entry++; cur_entry < entries.size(); cur_entry++) {
      const SparsityMapEntry<N,T>& e = entries[cur_entry];
      rect = restriction.intersection(e.bounds);
      if(rect.empty()) {
	// in 1-D, our entries are sorted, so the first one whose bounds fall
	//   outside our restriction means we're completely done
	if(N == 1) break;
	continue;
      }

      assert(!e.sparsity.exists());
      assert(e.bitmap == 0);
      return true;
    }

    // if we fall through, we're done
    valid = false;
    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LinearizedIndexSpaceIntfc

  inline LinearizedIndexSpaceIntfc::LinearizedIndexSpaceIntfc(int _dim, int _idxtype)
    : dim(_dim), idxtype(_idxtype)
  {}

  inline LinearizedIndexSpaceIntfc::~LinearizedIndexSpaceIntfc(void)
  {}

  // check and conversion routines to get a dimension-aware intermediate
  template <int N, typename T>
  inline bool LinearizedIndexSpaceIntfc::check_dim(void) const
  {
    return (dim == N) && (idxtype == (int)sizeof(T));
  }

  template <int N, typename T>
  inline LinearizedIndexSpace<N,T>& LinearizedIndexSpaceIntfc::as_dim(void)
  {
    assert((dim == N) && (idxtype == (int)sizeof(T)));
    return *static_cast<LinearizedIndexSpace<N,T> *>(this);
  }

  template <int N, typename T>
  inline const LinearizedIndexSpace<N,T>& LinearizedIndexSpaceIntfc::as_dim(void) const
  {
    assert((dim == N) && (idxtype == (int)sizeof(T)));
    return *static_cast<const LinearizedIndexSpace<N,T> *>(this);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LinearizedIndexSpace<N,T>

  template <int N, typename T>
  inline LinearizedIndexSpace<N,T>::LinearizedIndexSpace(const ZIndexSpace<N,T>& _indexspace)
    : LinearizedIndexSpaceIntfc(N, (int)sizeof(T))
    , indexspace(_indexspace)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class AffineLinearizedIndexSpace<N,T>

  template <int N, typename T>
  inline AffineLinearizedIndexSpace<N,T>::AffineLinearizedIndexSpace(const ZIndexSpace<N,T>& _indexspace,
								     bool fortran_order /*= true*/)
    : LinearizedIndexSpace<N,T>(_indexspace)
  {
    const ZRect<N,T>& bounds = this->indexspace.bounds;
    volume = bounds.volume();
    dbg_bounds = bounds;
    if(volume) {
      offset = 0;
      ptrdiff_t s = 1;  // initial stride == 1
      if(fortran_order) {
	for(int i = 0; i < N; i++) {
	  offset += bounds.lo[i] * s;
	  strides[i] = s;
	  s *= bounds.hi[i] - bounds.lo[i] + 1;
	}
      } else {
	for(int i = N-1; i >= 0; i--) {
	  offset += bounds.lo[i] * s;
	  strides[i] = s;
	  s *= bounds.hi[i] - bounds.lo[i] + 1;
	}
      }
      assert(s == (ptrdiff_t)volume);
    } else {
      offset = 0;
      for(int i = 0; i < N; i++) strides[i] = 0;
    }
  }

  template <int N, typename T>
  inline LinearizedIndexSpaceIntfc *AffineLinearizedIndexSpace<N,T>::clone(void) const
  {
    return new AffineLinearizedIndexSpace<N,T>(*this);
  }
    
  template <int N, typename T>
  inline size_t AffineLinearizedIndexSpace<N,T>::size(void) const
  {
    return volume;
  }

  template <int N, typename T>
  inline size_t AffineLinearizedIndexSpace<N,T>::linearize(const ZPoint<N,T>& p) const
  {
    size_t x = 0;
    for(int i = 0; i < N; i++)
      x += p[i] * strides[i];
    assert(x >= offset);
    return x - offset;
  }



}; // namespace Realm

namespace std {
  template<int N, typename T>
  inline bool less<Realm::ZPoint<N,T> >::operator()(const Realm::ZPoint<N,T>& p1,
						    const Realm::ZPoint<N,T>& p2) const
  {
    for(int i = 0; i < N; i++) {
      if(p1[i] < p2[i]) return true;
      if(p1[i] > p2[i]) return false;
    }
    return false;
  }

  template<int N, typename T>
  inline bool less<Realm::ZRect<N,T> >::operator()(const Realm::ZRect<N,T>& r1,
						   const Realm::ZRect<N,T>& r2) const
  {
    if(std::less<Realm::ZPoint<N,T> >()(r1.lo, r2.lo)) return true;
    if(std::less<Realm::ZPoint<N,T> >()(r2.lo, r1.lo)) return false;
    if(std::less<Realm::ZPoint<N,T> >()(r1.hi, r2.hi)) return true;
    if(std::less<Realm::ZPoint<N,T> >()(r2.hi, r1.hi)) return false;
    return false;
  }

  template<int N, typename T>
  inline bool less<Realm::ZIndexSpace<N,T> >::operator()(const Realm::ZIndexSpace<N,T>& is1,
							 const Realm::ZIndexSpace<N,T>& is2) const
  {
    if(std::less<Realm::ZRect<N,T> >()(is1.bounds, is2.bounds)) return true;
    if(std::less<Realm::ZRect<N,T> >()(is2.bounds, is1.bounds)) return false;
    return (is1.sparsity < is2.sparsity);
  }

};
