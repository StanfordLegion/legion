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

#include "instance.h"

#include "serialize.h"

TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::ZPoint<N,T>);
TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::ZRect<N,T>);
TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::ZIndexSpace<N,T>);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ZPoint<N,T>

  template <int N, typename T>
  inline ZPoint<N,T>::ZPoint(void)
  {}

  template <int N, typename T>
  template <typename T2>
  inline ZPoint<N,T>::ZPoint(const ZPoint<N,T2>& copy_from)
  {
    for(int i = 0; i < N; i++)
      (&x)[i] = (&copy_from.x)[i];
  }

  template <int N, typename T>
  template <typename T2>
  inline ZPoint<N,T>& ZPoint<N,T>::operator=(const ZPoint<N,T2>& copy_from)
  {
    for(int i = 0; i < N; i++)
      (&x)[i] = (&copy_from.x)[i];
    return *this;
  }

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

  template <int N, typename T>
  template <typename T2>
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
    ZPoint(void) {}
    ZPoint(T _x) : x(_x) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2>
    ZPoint(const ZPoint<1, T2>& copy_from) : x(copy_from.x) {}
    template <typename T2>
    ZPoint<1,T>& operator=(const ZPoint<1, T2>& copy_from)
    {
      x = copy_from.x;
      return *this;
    }

    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2>
    T dot(const ZPoint<1, T2>& rhs) const
    {
      return (x * rhs.x);
    }

    // special case: for N == 1, we're willing to coerce to T
    operator T(void) const { return x; }
  };

  template <typename T>
  struct ZPoint<2,T> {
    T x, y;
    ZPoint(void) {}
    ZPoint(T _x, T _y) : x(_x), y(_y) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2>
    ZPoint(const ZPoint<2, T2>& copy_from)
      : x(copy_from.x), y(copy_from.y) {}
    template <typename T2>
    ZPoint<2,T>& operator=(const ZPoint<2,T2>& copy_from)
    {
      x = copy_from.x;
      y = copy_from.y;
      return *this;
    }

    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2>
    T dot(const ZPoint<2, T2>& rhs) const
    {
      return (x * rhs.x) + (y * rhs.y);
    }
  };

  template <typename T>
  struct ZPoint<3,T> {
    T x, y, z;
    ZPoint(void) {}
    ZPoint(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2>
    ZPoint(const ZPoint<3, T2>& copy_from)
      : x(copy_from.x), y(copy_from.y), z(copy_from.z) {}
    template <typename T2>
    ZPoint<3,T>& operator=(const ZPoint<3,T2>& copy_from)
    {
      x = copy_from.x;
      y = copy_from.y;
      z = copy_from.z;
      return *this;
    }

    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2>
    T dot(const ZPoint<3, T2>& rhs) const
    {
      return (x * rhs.x) + (y * rhs.y) + (z * rhs.z);
    }
  };

  template <typename T>
  struct ZPoint<4,T> {
    T x, y, z, w;
    ZPoint(void) {}
    ZPoint(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2>
    ZPoint(const ZPoint<4, T2>& copy_from)
      : x(copy_from.x), y(copy_from.y), z(copy_from.z), w(copy_from.w) {}
    template <typename T2>
    ZPoint<4,T>& operator=(const ZPoint<4,T2>& copy_from)
    {
      x = copy_from.x;
      y = copy_from.y;
      z = copy_from.z;
      w = copy_from.w;
      return *this;
    }

    T& operator[](int index) { return (&x)[index]; }
    const T& operator[](int index) const { return (&x)[index]; }

    template <typename T2>
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
  template <int N, typename T, typename T2> 
  inline bool operator==(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) if(lhs[i] != rhs[i]) return false;
    return true;
  }
    
  template <int N, typename T, typename T2>
  inline bool operator!=(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) if(lhs[i] != rhs[i]) return true;
    return false;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T> operator+(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] + rhs[i];
    return out;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T>& operator+=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] += rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T> operator-(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] - rhs[i];
    return out;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T>& operator-=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] -= rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T> operator*(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] * rhs[i];
    return out;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T>& operator*=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] *= rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T> operator/(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] / rhs[i];
    return out;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T>& operator/=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] /= rhs[i];
    return lhs;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T> operator%(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    ZPoint<N,T> out;
    for(int i = 0; i < N; i++) out[i] = lhs[i] % rhs[i];
    return out;
  }

  template <int N, typename T, typename T2>
  inline ZPoint<N,T>& operator%=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    for(int i = 0; i < N; i++) lhs[i] %= rhs[i];
    return lhs;
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
  template <typename T2>
  inline ZRect<N,T>::ZRect(const ZRect<N, T2>& copy_from)
    : lo(copy_from.lo), hi(copy_from.hi)
  {}

  template <int N, typename T>
  template <typename T2>
  inline ZRect<N,T>& ZRect<N,T>::operator=(const ZRect<N, T2>& copy_from)
  {
    lo = copy_from.lo;
    hi = copy_from.hi;
    return *this;
  }

  template <int N, typename T>
  inline bool ZRect<N,T>::empty(void) const
  {
    for(int i = 0; i < N; i++) if(lo[i] > hi[i]) return true;
    return false;
  }

  template <int N, typename T>
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

  template <int N, typename T>
  inline bool ZRect<N,T>::contains(const ZPoint<N,T>& p) const
  {
    for(int i = 0; i < N; i++)
      if((p[i] < lo[i]) || (p[i] > hi[i])) return false;
    return true;
  }

  // true if all points in other are in this rectangle
  template <int N, typename T>
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
  template <int N, typename T>
  inline bool ZRect<N,T>::contains(const ZIndexSpace<N,T>& other) const
  {
    return contains(other.bounds);
  }

  // true if there are any points in the intersection of the two rectangles
  template <int N, typename T>
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

  template <int N, typename T>
  inline ZRect<N,T> ZRect<N,T>::intersection(const ZRect<N,T>& other) const
  {
    ZRect<N,T> out;
    for(int i = 0; i < N; i++) {
      out.lo[i] = std::max(lo[i], other.lo[i]);
      out.hi[i] = std::min(hi[i], other.hi[i]);
    }
    return out;
  };

  template <int N, typename T>
  inline ZRect<N,T> ZRect<N,T>::union_bbox(const ZRect<N,T>& other) const
  {
    ZRect<N,T> out;
    for(int i = 0; i < N; i++) {
      out.lo[i] = std::min(lo[i], other.lo[i]);
      out.hi[i] = std::max(hi[i], other.hi[i]);
    }
    return out;
  };

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const ZRect<N,T>& p)
  {
    os << p.lo << ".." << p.hi;
    return os;
  }

  // rectangles may be displaced by a vector (i.e. point)
  template <int N, typename T, typename T2>
  inline ZRect<N,T> operator+(const ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    return ZRect<N,T>(lhs.lo + rhs, lhs.hi + rhs);
  }

  template <int N, typename T, typename T2>
  inline ZRect<N,T>& operator+=(ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    lhs.lo += rhs;
    lhs.hi += rhs;
    return lhs;
  }

  template <int N, typename T, typename T2>
  inline ZRect<N,T> operator-(const ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs)
  {
    return ZRect<N,T>(lhs.lo - rhs, lhs.hi - rhs);
  }

  template <int N, typename T, typename T2>
  inline ZRect<N,T>& operator-=(ZRect<N,T>& lhs, const ZRect<N,T2>& rhs)
  {
    lhs.lo -= rhs;
    lhs.hi -= rhs;
    return lhs;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZMatrix<M,N,T>

  template <int M, int N, typename T>
  inline ZMatrix<M,N,T>::ZMatrix(void)
  {}

  // copies allow type coercion (assuming the underlying type does)
  template <int M, int N, typename T>
  template <typename T2>
  inline ZMatrix<M,N,T>::ZMatrix(const ZMatrix<M, N, T2>& copy_from)
  {
    for(int i = 0; i < M; i++)
      rows[i] = copy_from[i];
  }
  
  template <int M, int N, typename T>
  template <typename T2>
  inline ZMatrix<M, N, T>& ZMatrix<M,N,T>::operator=(const ZMatrix<M, N, T2>& copy_from)
  {
    for(int i = 0; i < M; i++)
      rows[i] = copy_from[i];
    return *this;
  }

  template <int M, int N, typename T, typename T2>
  inline ZPoint<M, T> operator*(const ZMatrix<M, N, T>& m, const ZPoint<N, T2>& p)
  {
    ZPoint<M,T> out;
    for(int j = 0; j < M; j++)
      out[j] = m.rows[j].dot(p);
    return out;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ZPointInRectIterator<N,T>

  template <int N, typename T>
  inline ZPointInRectIterator<N,T>::ZPointInRectIterator(const ZRect<N,T>& _r,
							 bool _fortran_order /*= true*/)
    : p(_r.lo), valid(!_r.empty()), rect(_r), fortran_order(_fortran_order)
  {}

  template <int N, typename T>
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
  template <int N, typename T>
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
      assert(0);
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
  inline Event ZIndexSpace<N,T>::fill(const std::vector<CopySrcDstField> &dsts,
				      const ProfilingRequestSet &requests,
				      const void *fill_value, size_t fill_value_size,
				      Event wait_on /*= Event::NO_EVENT*/) const
  {
    assert(0);
    return wait_on;
  }

  template <int N, typename T>
  inline Event ZIndexSpace<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
				      const std::vector<CopySrcDstField> &dsts,
				      const ProfilingRequestSet &requests,
				      Event wait_on /*= Event::NO_EVENT*/,
				      ReductionOpID redop_id /*= 0*/,
				      bool red_fold /*= false*/) const
  {
    assert(0);
    return wait_on;
  }

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
      // no restrictions, so we'll take the first entry (assuming it exists)
      if(entries.empty()) {
	valid = false;
	return;
      }
      cur_entry = 0;
      const SparsityMapEntry<N,T>& e = entries[cur_entry];

      assert(!e.sparsity.exists());
      assert(e.bitmap == 0);
      valid = true;
      rect = e.bounds;
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


  ////////////////////////////////////////////////////////////////////////
  //
  // class AffineAccessor<FT,N,T>

  // NOTE: these constructors will die horribly if the conversion is not
  //  allowed - call is_compatible(...) first if you're not sure

  // implicitly tries to cover the entire instance's domain
  template <typename FT, int N, typename T>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst, ptrdiff_t field_offset)
  {
    //const LinearizedIndexSpace<N,T>& lis = inst.get_lis().as_dim<N,T>();
    //const AffineLinearizedIndexSpace<N,T>& alis = dynamic_cast<const AffineLinearizedIndexSpace<N,T>&>(lis);
    const AffineLinearizedIndexSpace<N,T>& alis = dynamic_cast<const AffineLinearizedIndexSpace<N,T>&>(inst.get_lis());

    ptrdiff_t element_stride;
    inst.get_strided_access_parameters(0, alis.volume, field_offset, sizeof(FT), base, element_stride);

    // base offset is currently done in get_strided_access_parameters, since we're piggybacking on the
    //  old-style linearizers for now
    //base -= element_stride * alis.offset;
    for(int i = 0; i < N; i++)
      strides[i] = element_stride * alis.strides[i];
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = alis.dbg_bounds;
#endif
  }

  template <typename FT, int N, typename T> template <typename INST>
  inline AffineAccessor<FT,N,T>::AffineAccessor(const INST &inst, unsigned fid)
  {
    ptrdiff_t field_offset = 0;
    RegionInstance instance = inst.get_instance(fid, field_offset);
    const AffineLinearizedIndexSpace<N,T>& alis = 
      dynamic_cast<const AffineLinearizedIndexSpace<N,T>&>(instance.get_lis());
    ptrdiff_t element_stride;
    instance.get_strided_access_parameters(0, alis.volume, field_offset,
                                           sizeof(FT), base, element_stride);
    // base offset is currently done in get_strided_access_parameters, 
    //   since we're piggybacking on the old-style linearizers for now
    // base -= element_stride * alis.offset;
    for(int i = 0; i < N; i++)
      strides[i] = element_stride * alis.strides[i];
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = instance;
    dbg_bounds = alis.dbg_bounds;
#endif
#ifdef PRIVILEGE_CHECKS
    privileges = inst.get_accessor_privileges();
#endif
#ifdef BOUNDS_CHECKS
    bounds = inst.template get_bounds<N,T>();
#endif
  }

  template <typename FT, int N, typename T>
  inline AffineAccessor<FT,N,T>::~AffineAccessor(void)
  {}
#if 0
    // limits domain to a subrectangle
  template <typename FT, int N, typename T>
    AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst, ptrdiff_t field_offset, const ZRect<N,T>& subrect);

  template <typename FT, int N, typename T>
    AffineAccessor<FT,N,T>::~AffineAccessor(void);

  template <typename FT, int N, typename T>
    static bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst, ptrdiff_t field_offset);
  template <typename FT, int N, typename T>
    static bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst, ptrdiff_t field_offset, const ZRect<N,T>& subrect);
#endif

  template <typename FT, int N, typename T>
  inline FT *AffineAccessor<FT,N,T>::ptr(const ZPoint<N,T>& p) const
  {
#ifdef PRIVILEGE_CHECKS
    assert(privileges & ACCESSOR_PRIV_ALL);
#endif
#ifdef BOUNDS_CHECKS
    assert(bounds.contains(p));
#endif
    intptr_t rawptr = base;
    for(int i = 0; i < N; i++) rawptr += p[i] * strides[i];
    return reinterpret_cast<FT *>(rawptr);
  }

  template <typename FT, int N, typename T>
  inline FT AffineAccessor<FT,N,T>::read(const ZPoint<N,T>& p) const
  {
#ifdef PRIVILEGE_CHECKS
    assert(privileges & ACCESSOR_PRIV_READ);
#endif
#ifdef BOUNDS_CHECKS
    assert(bounds.contains(p));
#endif
    return *(this->ptr(p));
  }

  template <typename FT, int N, typename T>
  inline void AffineAccessor<FT,N,T>::write(const ZPoint<N,T>& p, FT newval) const
  {
#ifdef PRIVILEGE_CHECKS
    assert(privileges & ACCESSOR_PRIV_WRITE);
#endif
#ifdef BOUNDS_CHECKS
    assert(bounds.contains(p));
#endif
    *(ptr(p)) = newval;
  }

  template <typename FT, int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const AffineAccessor<FT,N,T>& a)
  {
    os << "AffineAccessor{ base=" << std::hex << a.base << std::dec << " strides=" << a.strides;
#ifdef REALM_ACCESSOR_DEBUG
    os << " inst=" << a.dbg_inst;
    os << " bounds=" << a.dbg_bounds;
    os << "->[" << std::hex << a.ptr(a.dbg_bounds.lo) << "," << a.ptr(a.dbg_bounds.hi)+1 << std::dec << "]";
#endif
    os << " }";
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance

  template <int N, typename T>
  inline /*static*/ RegionInstance RegionInstance::create_instance(Memory memory,
								   const ZIndexSpace<N,T>& space,
								   const std::vector<size_t> &field_sizes,
								   const ProfilingRequestSet& reqs)
  {
    if(N == 1) {
      assert(space.dense());
      LegionRuntime::Arrays::Rect<1> r;
      r.lo = space.bounds.lo.x;
      r.hi = space.bounds.hi.x;
      Domain d = Domain::from_rect<1>(r);
      return d.create_instance(memory, field_sizes, space.bounds.volume(), reqs);
    } else {
      // TODO: all sorts of serialization fun...
      assert(false);
      return RegionInstance::NO_INST;
    }
  }

  template <int N, typename T>
  inline /*static*/ RegionInstance RegionInstance::create_file_instance(const char *file_name,
									const ZIndexSpace<N,T>& space,
									const std::vector<size_t> &field_sizes,
									legion_lowlevel_file_mode_t file_mode,
									const ProfilingRequestSet& prs)
  {
    assert(0);
    return RegionInstance::NO_INST;
  }

  template <int N, typename T>
  inline /*static*/ RegionInstance RegionInstance::create_hdf5_instance(const char *file_name,
									const ZIndexSpace<N,T>& space,
									const std::vector<size_t> &field_sizes,
									const std::vector<const char*> &field_files,
									bool read_only,
									const ProfilingRequestSet& prs)
  {
    assert(0);
    return RegionInstance::NO_INST;
  }

  template <int N, typename T>
  inline const ZIndexSpace<N,T>& RegionInstance::get_indexspace(void) const
  {
    return get_lis().as_dim<N,T>().indexspace;
  }
		
  template <int N>
  inline const ZIndexSpace<N,int>& RegionInstance::get_indexspace(void) const
  {
    return get_lis().as_dim<N,int>().indexspace;
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
