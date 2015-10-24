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

    // special case: for N == 1, we're willing to coerce to T
    operator T(void) const { return x; }
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

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const ZPoint<N,T>& p)
  {
    os << '<' << p[0];
    for(int i = 1; i < N; i++)
      os << ',' << p[i];
    os << '>';
    return os;
  }

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

  template <int N, typename T>
  bool ZRect<N,T>::contains(const ZPoint<N,T>& p) const
  {
    for(int i = 0; i < N; i++)
      if((p[i] < lo[i]) || (p[i] > hi[i])) return false;
    return true;
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

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const ZRect<N,T>& p)
  {
    os << p.lo << ".." << p.hi;
    return os;
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

  // queries for individual points or rectangles
  template <int N, typename T>
  inline bool ZIndexSpace<N,T>::contains(const ZPoint<N,T>& p) const
  {
    // test on bounding box first
    if(!bounds.contains(p))
      return false;

    if(!dense()) {
      // test against sparsity map too
      assert(0);
    }

    return true;
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

  // actual number of points in index space (may be less than volume of bounding box)
  template <int N, typename T>
  inline size_t ZIndexSpace<N,T>::volume(void) const
  {
    if(dense())
      return bounds.volume();

    assert(0);
    return 0;
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
  Event ZIndexSpace<N,T>::create_subspace_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > >& field_data,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_union(const ZIndexSpace<N,T>& lhs,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_unions(const ZIndexSpace<N,T>& lhs,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_unions(const std::vector<ZIndexSpace<N,T> >& lhss,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_intersection(const ZIndexSpace<N,T>& lhs,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_intersections(const ZIndexSpace<N,T>& lhs,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_intersections(const std::vector<ZIndexSpace<N,T> >& lhss,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_difference(const ZIndexSpace<N,T>& lhs,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_differences(const ZIndexSpace<N,T>& lhs,
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
  /*static*/ Event ZIndexSpace<N,T>::compute_differences(const std::vector<ZIndexSpace<N,T> >& lhss,
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
      os << ",sparse(" << is.sparsity.id << ")";
    }
    return os;
  }

  template <int N, typename T>
  inline RegionInstance ZIndexSpace<N,T>::create_instance(Memory memory,
							  const std::vector<size_t> &field_sizes,
							  size_t block_size,
							  const ProfilingRequestSet& reqs) const
  {
    // for now, create a instance that holds our entire bounding box and can therefore use a simple
    //  linearizer

    return RegionInstance::create_instance(memory,
					   AffineLinearizedIndexSpace<N,T>(*this),
					   field_sizes,
					   reqs);
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
      assert(s == volume);
    } else {
      offset = 0;
      for(int i = 0; i < N; i++) strides[i] = 0;
    }
  }

  template <int N, typename T>
  LinearizedIndexSpaceIntfc *AffineLinearizedIndexSpace<N,T>::clone(void) const
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

    base -= element_stride * alis.offset;
    strides = element_stride * alis.strides;
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
  inline FT *AffineAccessor<FT,N,T>::ptr(const ZPoint<N,T>& p)
  {
    intptr_t rawptr = base;
    for(int i = 0; i < N; i++) rawptr += p[i] * strides[i];
    return reinterpret_cast<FT *>(rawptr);
  }

  template <typename FT, int N, typename T>
  inline FT AffineAccessor<FT,N,T>::read(const ZPoint<N,T>& p)
  {
    return *(this->ptr(p));
  }

  template <typename FT, int N, typename T>
  inline void AffineAccessor<FT,N,T>::write(const ZPoint<N,T>& p, FT newval)
  {
    *(ptr(p)) = newval;
  }



  template <int N, typename T>
  const ZIndexSpace<N,T>& RegionInstance::get_indexspace(void) const
  {
    return get_lis().as_dim<N,T>().indexspace;
  }
		
  template <int N>
  const ZIndexSpace<N,int>& RegionInstance::get_indexspace(void) const
  {
    return get_lis().as_dim<N,int>().indexspace;
  }

}; // namespace Realm

