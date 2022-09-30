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

// index spaces for Realm

// nop, but helps IDEs
#include "realm/indexspace.h"

#include "realm/sparsity.h"
//include "realm/instance.h"
//include "realm/inst_layout.h"

#include "realm/serialize.h"
#include "realm/logging.h"

TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::IndexSpace<N,T>);

namespace Realm {

  REALM_INTERNAL_API_EXTERNAL_LINKAGE
  extern Logger log_dpops;

  ////////////////////////////////////////////////////////////////////////
  //
  // class TranslationTransform<N, T>

  template <int N, typename T>
  inline TranslationTransform<N, T>::TranslationTransform(
      const Point<N, T>& _offset)
      : offset(_offset) {}

  template <int N, typename T>
  template <typename T2>
  inline Point<N, T> TranslationTransform<N, T>::operator[](
      const Point<N, T2>& point) const {
    return point + offset;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class AfineTransform<M, N, T>

  template <int M, int N, typename T>
  inline AffineTransform<M, N, T>::AffineTransform(
      const Matrix<M, N, T>& _transform, const Point<M, T>& _offset)
      : transform(_transform), offset(_offset) {}

  template <int M, int N, typename T>
  template <typename T2>
  inline Point<M, T> AffineTransform<M, N, T>::operator[](
      const Point<N, T2>& point) const {
    return (transform * point) + offset;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class StructuredTransform<N, T, N2, T2>

  template <int N, typename T, int N2, typename T2>
  StructuredTransform<N, T, N2, T2>::StructuredTransform(
      const AffineTransform<N, N2, T2>& _transform)
      : transform_matrix(_transform.transform),
        offset(_transform.offset),
        type(StructuredTransformType::AFFINE) {}

  template <int N, typename T, int N2, typename T2>
  StructuredTransform<N, T, N2, T2>::StructuredTransform(
      const TranslationTransform<N, T2>& _transform)
      : offset(_transform.offset), type(StructuredTransformType::TRANSLATION) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N2; j++) {
        transform_matrix[i][j] = (i == j);
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  inline Point<N, T> StructuredTransform<N, T, N2, T2>::operator[](
      const Point<N2, T>& point) const {
    return (transform_matrix * point) + offset;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Domainransform<N, T, N2, T2>

  template <int N, typename T, int N2, typename T2>
  DomainTransform<N, T, N2, T2>::DomainTransform(
      const StructuredTransform<N, T, N2, T2>& _transform)
      : structured_transform(_transform),
        type(DomainTransformType::STRUCTURED) {}

  template <int N, typename T, int N2, typename T2>
  DomainTransform<N, T, N2, T2>::DomainTransform(
      const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>&
          _field_data)
      : ptr_data(_field_data), type(DomainTransformType::UNSTRUCTURED_PTR) {}

  template <int N, typename T, int N2, typename T2>
  DomainTransform<N, T, N2, T2>::DomainTransform(
      const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Rect<N, T>>>&
          _field_data)
      : range_data(_field_data),
        type(DomainTransformType::UNSTRUCTURED_RANGE) {}

  ////////////////////////////////////////////////////////////////////////
  //
  // class Rect<N,T>

  // copy and fill operations (wrappers for IndexSpace versions)
  template <int N, typename T>
  inline Event Rect<N,T>::fill(const std::vector<CopySrcDstField> &dsts,
				const ProfilingRequestSet &requests,
				const void *fill_value, size_t fill_value_size,
				Event wait_on /*= Event::NO_EVENT*/) const
  {
    return IndexSpace<N,T>(*this).fill(dsts, requests,
					fill_value, fill_value_size,
					wait_on);
  }

  template <int N, typename T>
  inline Event Rect<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				const ProfilingRequestSet &requests,
				Event wait_on /*= Event::NO_EVENT*/) const
  {
    return IndexSpace<N,T>(*this).copy(srcs, dsts,
					requests, wait_on);
  }

  template <int N, typename T>
  inline Event Rect<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				const IndexSpace<N,T> &mask,
				const ProfilingRequestSet &requests,
				Event wait_on /*= Event::NO_EVENT*/) const
  {
    return IndexSpace<N,T>(*this).copy(srcs, dsts, mask,
					requests, wait_on);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct CopySrcDstField

  inline CopySrcDstField::CopySrcDstField(void)
    : inst(RegionInstance::NO_INST)
    , field_id(FieldID(-1))
    , size(0)
    , redop_id(0)
    , red_fold(false)
    , red_exclusive(false)
    , serdez_id(0)
    , subfield_offset(0)
    , indirect_index(-1)
  {
    fill_data.indirect = 0;
  }

  inline CopySrcDstField::CopySrcDstField(const CopySrcDstField& copy_from)
    : inst(copy_from.inst)
    , field_id(copy_from.field_id)
    , size(copy_from.size)
    , redop_id(copy_from.redop_id)
    , red_fold(copy_from.red_fold)
    , red_exclusive(copy_from.red_exclusive)
    , serdez_id(copy_from.serdez_id)
    , subfield_offset(copy_from.subfield_offset)
    , indirect_index(copy_from.indirect_index)
  {
    // we know there's a fill value if the field ID is -1
    if(copy_from.field_id == FieldID(-1)) {
      if(size <= MAX_DIRECT_SIZE) {
	// copy whole buffer to make sure indirect is initialized too
	memcpy(fill_data.direct, copy_from.fill_data.direct, MAX_DIRECT_SIZE);
      } else {
	if(copy_from.fill_data.indirect) {
	  fill_data.indirect = malloc(size);
	  memcpy(fill_data.indirect, copy_from.fill_data.indirect, size);
	} else
	  fill_data.indirect = 0;
      }
    } else
      fill_data.indirect = 0;
  }

  inline CopySrcDstField& CopySrcDstField::operator=(const CopySrcDstField& copy_from)
  {
    if(this == &copy_from) return *this; // self-assignment
    if((field_id != FieldID(-1)) && (size > MAX_DIRECT_SIZE) && fill_data.indirect)
      free(fill_data.indirect);

    inst = copy_from.inst;
    field_id = copy_from.field_id;
    size = copy_from.size;
    redop_id = copy_from.redop_id;
    red_fold = copy_from.red_fold;
    red_exclusive = copy_from.red_exclusive;
    serdez_id = copy_from.serdez_id;
    subfield_offset = copy_from.subfield_offset;
    indirect_index = copy_from.indirect_index;

    // we know there's a fill value if the field ID is -1
    if(copy_from.field_id == FieldID(-1)) {
      if(size <= MAX_DIRECT_SIZE) {
	// copy whole buffer to make sure indirect is initialized too
	memcpy(fill_data.direct, copy_from.fill_data.direct, MAX_DIRECT_SIZE);
      } else {
	if(copy_from.fill_data.indirect) {
	  fill_data.indirect = malloc(size);
	  memcpy(fill_data.indirect, copy_from.fill_data.indirect, size);
	} else
	  fill_data.indirect = 0;
      }
    } else
      fill_data.indirect = 0;

    return *this;
  }

  inline CopySrcDstField::~CopySrcDstField(void)
  {
    if((field_id == FieldID(-1)) && (size > MAX_DIRECT_SIZE)) {
      free(fill_data.indirect);
    }
  }

  inline CopySrcDstField &CopySrcDstField::set_field(RegionInstance _inst,
						     FieldID _field_id,
						     size_t _size,
						     size_t _subfield_offset /*= 0*/)
  {
    inst = _inst;
    field_id = _field_id;
    size = _size;
    subfield_offset = _subfield_offset;
    return *this;
  }

  inline CopySrcDstField &CopySrcDstField::set_indirect(int _indirect_index,
							FieldID _field_id,
							size_t _size,
							size_t _subfield_offset /*= 0*/)
  {
    indirect_index = _indirect_index;
    field_id = _field_id;
    size = _size;
    subfield_offset = _subfield_offset;
    return *this;
  }

  inline CopySrcDstField &CopySrcDstField::set_redop(ReductionOpID _redop_id,
                                                     bool _is_fold,
                                                     bool _is_excl)
  {
    redop_id = _redop_id;
    red_fold = _is_fold;
    red_exclusive = _is_excl;
    return *this;
  }

  inline CopySrcDstField &CopySrcDstField::set_serdez(CustomSerdezID _serdez_id)
  {
    serdez_id = _serdez_id;
    return *this;
  }
  
  inline CopySrcDstField &CopySrcDstField::set_fill(const void *_data, size_t _size)
  {
    field_id = -1;
    size = _size;
    if(size <= MAX_DIRECT_SIZE) {
      memcpy(&fill_data.direct, _data, size);
    } else {
      fill_data.indirect = malloc(size);
      memcpy(fill_data.indirect, _data, size);
    }
    return *this;
  }

  template <typename T>
  inline CopySrcDstField &CopySrcDstField::set_fill(T value)
  {
    return set_fill(&value, sizeof(T));
  }

  template <typename S>
  inline bool serialize(S& s, const CopySrcDstField& v)
  {
    if(!((s << v.inst) &&
	 (s << v.field_id) &&
	 (s << v.size) &&
	 (s << v.redop_id) &&
	 (s << v.red_fold) &&
	 (s << v.serdez_id) &&
	 (s << v.subfield_offset) &&
	 (s << v.indirect_index))) return false;

    // we know there's a fill value if the field ID is -1
    if(v.field_id == FieldID(-1)) {
      if(!s.append_bytes(((v.size <= CopySrcDstField::MAX_DIRECT_SIZE) ?
			    v.fill_data.direct :
			    v.fill_data.indirect),
			 v.size))
	return false;
    }

    return true;
  }

  template <typename S>
  inline bool deserialize(S& s, CopySrcDstField& v)
  {
    if(!((s >> v.inst) &&
	 (s >> v.field_id) &&
	 (s >> v.size) &&
	 (s >> v.redop_id) &&
	 (s >> v.red_fold) &&
	 (s >> v.serdez_id) &&
	 (s >> v.subfield_offset) &&
	 (s >> v.indirect_index))) return false;

    // we know there's a fill value if the field ID is -1
    if(v.field_id == FieldID(-1)) {
      if(v.size <= CopySrcDstField::MAX_DIRECT_SIZE) {
	if(!s.extract_bytes(v.fill_data.direct, v.size))
	  return false;
      } else {
	v.fill_data.indirect = malloc(v.size);
	if(!s.extract_bytes(v.fill_data.indirect, v.size))
	  return false;
      }
    }

    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IndexSpace<N,T>

  template <int N, typename T>
  inline IndexSpace<N,T>::IndexSpace(void)
  {}

  template <int N, typename T>
  inline IndexSpace<N,T>::IndexSpace(const Rect<N,T>& _bounds)
    : bounds(_bounds)
  {
    sparsity.id = 0;
  }

  template <int N, typename T>
  inline IndexSpace<N,T>::IndexSpace(const Rect<N,T>& _bounds, SparsityMap<N,T> _sparsity)
    : bounds(_bounds), sparsity(_sparsity)
  {}

  // construct an index space from a list of points or rects
  template <int N, typename T>
  inline IndexSpace<N,T>::IndexSpace(const std::vector<Point<N,T> >& points,
                                     bool disjoint /*= false*/)
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
	  bounds = bounds.union_bbox(Rect<N,T>(points[i], points[i]));
	sparsity = SparsityMap<N,T>::construct(points, false /*!always_create*/,
                                               disjoint);
      }
    }
    log_dpops.info() << "construct: " << *this;
  }

  template <int N, typename T>
  inline IndexSpace<N,T>::IndexSpace(const std::vector<Rect<N,T> >& rects,
                                     bool disjoint /*= false*/)
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
	sparsity = SparsityMap<N,T>::construct(rects, false /*!always_create*/,
                                               disjoint);
      }
    }
    log_dpops.info() << "construct: " << *this;
  }

  // constructs a guaranteed-empty index space
  template <int N, typename T>
  inline /*static*/ IndexSpace<N,T> IndexSpace<N,T>::make_empty(void)
  {
    return IndexSpace<N,T>(Rect<N,T>::make_empty());
  }

  // reclaim any physical resources associated with this index space
  //  will clear the sparsity map of this index space if it exists
  template <int N, typename T>
  inline void IndexSpace<N,T>::destroy(Event wait_on /*= Event::NO_EVENT*/)
  {}

  // true if we're SURE that there are no points in the space (may be imprecise due to
  //  lazy loading of sparsity data)
  template <int N, typename T>
  inline bool IndexSpace<N,T>::empty(void) const
  {
    return bounds.empty();
  }
    
  // true if there is no sparsity map (i.e. the bounds fully define the domain)
  template <int N, typename T>
  REALM_CUDA_HD
  inline bool IndexSpace<N,T>::dense(void) const
  {
    return !sparsity.exists();
  }

  // kicks off any operation needed to get detailed sparsity information - asking for
  //  approximate data can be a lot quicker for complicated index spaces
  template <int N, typename T>
  inline Event IndexSpace<N,T>::make_valid(bool precise /*= true*/) const
  {
    if(sparsity.exists())
      return sparsity.impl()->make_valid(precise);
    else
      return Event::NO_EVENT;
  }

  template <int N, typename T>
  inline bool IndexSpace<N,T>::is_valid(bool precise /*= true*/) const
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
  IndexSpace<N,T> IndexSpace<N,T>::tighten(bool precise /*= true*/) const
  {
    if(sparsity.exists()) {
      SparsityMapPublicImpl<N,T> *impl = sparsity.impl();

      // if we don't have the data, it's too late - somebody should have waited
      REALM_ASSERT(impl->is_valid(precise),
		   "IndexSpace<N,T>::tighten called without waiting for valid metadata");

      // always use precise info if it's available
      if(impl->is_valid(true /*precise*/)) {
	IndexSpace<N,T> result;
	const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
	// three cases:
	// 1) empty index space
	if(entries.empty()) {
	  result = IndexSpace<N,T>::make_empty();
	} else

	// 2) single dense rectangle
	if((entries.size() == 1) &&
	   !entries[0].sparsity.exists() && (entries[0].bitmap == 0)) {
	  result = IndexSpace<N,T>(bounds.intersection(entries[0].bounds));
	} else

	// 3) anything else - walk rectangles and count/union those that
	//   overlap our bounds - if only 1 or if the volume inside the
        //   tightened bounds equals the bounds themselves, we can drop
        //   the sparsity map
	{
	  size_t overlap_count = 0;
          size_t volume_sum = 0;
	  bool need_sparsity = false;
	  result = IndexSpace<N,T>::make_empty();
	  for(size_t i = 0; i < entries.size(); i++) {
	    Rect<N,T> isect = bounds.intersection(entries[i].bounds);
	    if(!isect.empty()) {
	      overlap_count++;
              volume_sum += isect.volume();
	      result.bounds = result.bounds.union_bbox(isect);
	      if(entries[i].sparsity.exists() || (entries[i].bitmap != 0))
		need_sparsity = true;
	    }
	  }
	  if(need_sparsity ||
             ((overlap_count > 1) && (result.bounds.volume() > volume_sum)))
	    result.sparsity = sparsity;
	}

	log_dpops.info() << "tighten: " << *this << " = " << result;
	return result;
      } else {
	const std::vector<Rect<N,T> >& approx_rects = impl->get_approx_rects();

	// two cases:
	// 1) empty index space
	if(approx_rects.empty()) {
	  Rect<N,T> empty;
	  empty.hi = bounds.lo;
	  for(int i = 0; i < N; i++)
	    empty.lo[i] = empty.hi[i] + 1;
	  return IndexSpace<N,T>(empty);
	}

	// 2) anything else - keep the sparsity map but tighten the bounds,
	//   respecting the previous bounds
	Rect<N,T> bbox = bounds.intersection(approx_rects[0]);
	for(size_t i = 1; i < approx_rects.size(); i++)
	  bbox = bbox.union_bbox(bounds.intersection(approx_rects[i]));
	return IndexSpace<N,T>(bbox, sparsity);
      }
    } else
      return *this;
  }


  // helper function that binary searches a (1-D) sparsity map entry list and returns
  //  the index of the entry that contains the point, or the first one to appear after
  //  that point
  template <int N, typename T>
  static size_t bsearch_map_entries(const std::vector<SparsityMapEntry<N,T> >& entries,
				    const Point<N,T>& p)
  {
    assert(N == 1);
    // search range at any given time is [lo, hi)
    int lo = 0;
    int hi = entries.size();
    while(lo < hi) {
      size_t mid = (lo + hi) >> 1;  // rounding down keeps up from picking hi
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
  inline bool IndexSpace<N,T>::contains(const Point<N,T>& p) const
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
      size_t idx = bsearch_map_entries<N,T>(entries, p);
      if(idx >= entries.size()) return false;

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
  inline bool IndexSpace<N,T>::contains_all(const Rect<N,T>& r) const
  {
    // test on bounding box first
    if(!bounds.contains(r))
      return false;

    if(!dense()) {
      // test against sparsity map too
      size_t total_volume = 0;
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
          Rect<N,T> isect = it->bounds.intersection(r);
          total_volume += isect.volume();
	}
      }

      // did we miss anything?
      if(total_volume < r.volume())
        return false;
    }

    return true;
  }

  template <int N, typename T>
  inline bool IndexSpace<N,T>::contains_any(const Rect<N,T>& r) const
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
  inline bool IndexSpace<N,T>::overlaps(const IndexSpace<N,T>& other) const
  {
    // this covers the both-dense case as well as the same-sparsity-map case
    if(sparsity == other.sparsity)
      return bounds.overlaps(other.bounds);

    // dense vs. sparse in both directions
    if(dense())
      return other.contains_any(bounds);

    if(other.dense())
      return contains_any(other.bounds);

    // both sparse case can be expensive...
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    SparsityMapPublicImpl<N,T> *other_impl = other.sparsity.impl();
    // overlap can only be within intersecion of bounds
    Rect<N,T> isect = bounds.intersection(other.bounds);

    return impl->overlaps(other_impl, isect, false /*!approx*/);
  }

  // actual number of points in index space (may be less than volume of bounding box)
  template <int N, typename T>
  inline size_t IndexSpace<N,T>::volume(void) const
  {
    if(dense())
      return bounds.volume();

    size_t total = 0;
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
    for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	it != entries.end();
	it++) {
      Rect<N,T> isect = bounds.intersection(it->bounds);
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
  inline bool IndexSpace<N,T>::contains_approx(const Point<N,T>& p) const
  {
    // test on bounding box first
    if(!bounds.contains(p))
      return false;

    // if it's a dense rectangle, no further tests
    if(dense())
      return true;

    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<Rect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<Rect<N,T> >::const_iterator it = approx_rects.begin();
	it != approx_rects.end();
	it++)
      if(it->contains(p))
	return true;

    // no entries matched, so the point is definitely not contained in this space
    return false;
  }

  template <int N, typename T>
  inline bool IndexSpace<N,T>::contains_all_approx(const Rect<N,T>& r) const
  {
    // test on bounding box first
    if(!bounds.contains(r))
      return false;

    // if it's a dense rectangle, no further tests
    if(dense())
      return true;

    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<Rect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<Rect<N,T> >::const_iterator it = approx_rects.begin();
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
  inline bool IndexSpace<N,T>::contains_any_approx(const Rect<N,T>& r) const
  {
    // test on bounding box first
    if(!bounds.overlaps(r))
      return false;

    // if it's a dense rectangle, no further tests
    if(dense())
      return true;

    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<Rect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<Rect<N,T> >::const_iterator it = approx_rects.begin();
	it != approx_rects.end();
	it++) {
      if(it->overlaps(r))
	return true;
    }

    // no entries matched, so the point is definitely not contained in this space
    return false;
  }

  template <int N, typename T>
  inline bool IndexSpace<N,T>::overlaps_approx(const IndexSpace<N,T>& other) const
  {
    // this covers the both-dense case as well as the same-sparsity-map case
    if(sparsity == other.sparsity)
      return bounds.overlaps(other.bounds);

    // dense vs. sparse in both directions
    if(dense())
      return other.contains_any_approx(bounds);

    if(other.dense())
      return contains_any_approx(other.bounds);

    // both sparse case can be expensive...
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    SparsityMapPublicImpl<N,T> *other_impl = other.sparsity.impl();
    // overlap can only be within intersecion of bounds
    Rect<N,T> isect = bounds.intersection(other.bounds);

    return impl->overlaps(other_impl, isect, true /*approx*/);
  }

  // approximage number of points in index space (may be less than volume of bounding box, but larger than
  //   actual volume)
  template <int N, typename T>
  inline size_t IndexSpace<N,T>::volume_approx(void) const
  {
    if(dense())
      return bounds.volume();

    size_t total = 0;
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    const std::vector<Rect<N,T> >& approx_rects = impl->get_approx_rects();
    for(typename std::vector<Rect<N,T> >::const_iterator it = approx_rects.begin();
	it != approx_rects.end();
	it++)
      total += it->volume();

    return total;
  }

  // copy and fill operations

  template <int N, typename T>
  inline Event IndexSpace<N,T>::fill(const std::vector<CopySrcDstField> &dsts,
				     const Realm::ProfilingRequestSet &requests,
				     const void *fill_value, size_t fill_value_size,
				     Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<CopySrcDstField> srcs;
    srcs.resize(dsts.size());
    size_t offset = 0;
    for(size_t i = 0; i < dsts.size(); i++) {
      assert((offset + dsts[i].size) <= fill_value_size);
      srcs[i].set_fill(reinterpret_cast<const char *>(fill_value) + offset,
		       dsts[i].size);
      // special case: if a field uses all of the fill value, the next
      //  field (if any) is allowed to use the same value
      if((offset > 0) || (dsts[i].size != fill_value_size))
	offset += dsts[i].size;
    }
    return copy(srcs, dsts,
		std::vector<const typename CopyIndirection<N,T>::Base *>(),
		requests, wait_on);
  }

  template <int N, typename T>
  inline Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
				     const std::vector<CopySrcDstField> &dsts,
				     const ProfilingRequestSet &requests,
				     Event wait_on /*= Event::NO_EVENT*/) const
  {
    return copy(srcs, dsts,
		std::vector<const typename CopyIndirection<N,T>::Base *>(),
		requests, wait_on);
  }

  // integer version of weighted subspace is a wrapper around size_t version
  template <int N, typename T>
  inline Event IndexSpace<N,T>::create_weighted_subspaces(size_t count, size_t granularity,
							  const std::vector<int>& weights,
							  std::vector<IndexSpace<N,T> >& subspaces,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<size_t> weights_size_t(weights.size());

    // clamp negative values to 0
    for(size_t i = 0; i < weights.size(); i++)
      weights_size_t[i] = (weights[i] > 0) ? weights[i] : 0;

    return create_weighted_subspaces(count, granularity, weights_size_t,
				     subspaces, reqs, wait_on);
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <typename FT>
  inline Event IndexSpace<N,T>::create_subspace_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
							  FT color,
							  IndexSpace<N,T>& subspace,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<FT> colors(1, color);
    std::vector<IndexSpace<N,T> > subspaces;
    Event e = create_subspaces_by_field(field_data, colors, subspaces, reqs, wait_on);
    subspace = subspaces[0];
    return e;
  }

  template <int N, typename T>
  template <int N2, typename T2, typename TRANSFORM>
  inline Event IndexSpace<N, T>::create_subspace_by_image(
      const TRANSFORM& transform, const IndexSpace<N2, T2>& source,
      const IndexSpace<N, T>& image, const ProfilingRequestSet& reqs,
      Event wait_on) const {
   return create_subspaces_by_image(transform, {source}, {image}, reqs,
                                    wait_on);
  }

  template <int N, typename T>
  template <int N2, typename T2, typename TRANSFORM>
  inline Event IndexSpace<N, T>::create_subspaces_by_image(
      const TRANSFORM& transform,
      const std::vector<IndexSpace<N2, T2>>& sources,
      std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
      Event wait_on) const {
   // TODO(apryakhin): For now we just support building a general structured
   // transform from an affince transform. This will be extended later
   // to support more transform types.
   assert(typeid(transform) == typeid(AffineTransform<N, N2, T2>) ||
          typeid(transform) == typeid(TranslationTransform<N2, T2>));
   return create_subspaces_by_image(DomainTransform<N, T, N2, T2>(transform),
                                    sources, images, reqs, wait_on);
  }

  template <int N, typename T>
  template <int N2, typename T2>
  inline Event IndexSpace<N, T>::create_subspaces_by_image(
      const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>&
          field_data,
      const std::vector<IndexSpace<N2, T2>>& sources,
      std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
      Event wait_on) const {
   return create_subspaces_by_image(DomainTransform<N, T, N2, T2>(field_data),
                                    sources, images, reqs, wait_on);
  }

  template <int N, typename T>
  template <int N2, typename T2>
  inline Event IndexSpace<N, T>::create_subspaces_by_image(
      const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Rect<N, T>>>&
          field_data,
      const std::vector<IndexSpace<N2, T2>>& sources,
      std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
      Event wait_on) const {
   return create_subspaces_by_image(DomainTransform<N, T, N2, T2>(field_data),
                                    sources, images, reqs, wait_on);
  }

  template <int N, typename T>
  template <int N2, typename T2>
  inline Event IndexSpace<N, T>::create_subspaces_by_image_with_difference(
      const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>&
          field_data,
      const std::vector<IndexSpace<N2, T2>>& sources,
      const std::vector<IndexSpace<N, T>>& diff_rhs,
      std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
      Event wait_on) const {
   return create_subspaces_by_image_with_difference(
       DomainTransform<N, T, N2, T2>(field_data), sources, diff_rhs, images,
       reqs, wait_on);
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event IndexSpace<N,T>::create_subspace_by_image(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Point<N,T> > >& field_data,
							  const IndexSpace<N2,T2>& source,
							  IndexSpace<N,T>& image,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<IndexSpace<N2,T2> > sources(1, source);
    std::vector<IndexSpace<N,T> > images;
    Event e = create_subspaces_by_image(field_data, sources, images, reqs, wait_on);
    image = images[0];
    return e;
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event IndexSpace<N,T>::create_subspace_by_image(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Rect<N,T> > >& field_data,
							  const IndexSpace<N2,T2>& source,
							  IndexSpace<N,T>& image,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<IndexSpace<N2,T2> > sources(1, source);
    std::vector<IndexSpace<N,T> > images;
    Event e = create_subspaces_by_image(field_data, sources, images, reqs, wait_on);
    image = images[0];
    return e;
  }

  template <int N, typename T>
  template <int N2, typename T2, typename TRANSFORM>
  Event IndexSpace<N, T>::create_subspace_by_preimage(
      const TRANSFORM& transform, const IndexSpace<N2, T2>& target,
      IndexSpace<N, T>& preimage, const ProfilingRequestSet& reqs,
      Event wait_on /*= Event::NO_EVENT*/) const {
   return create_subspaces_by_preimage(DomainTransform<N2, T2, N, T>(transform),
                                       {target}, {preimage}, reqs, wait_on);
  }

  template <int N, typename T>
  template <int N2, typename T2, typename TRANSFORM>
  Event IndexSpace<N, T>::create_subspaces_by_preimage(
      const TRANSFORM& transform,
      const std::vector<IndexSpace<N2, T2>>& targets,
      std::vector<IndexSpace<N, T>>& preimages, const ProfilingRequestSet& reqs,
      Event wait_on /*= Event::NO_EVENT*/) const {
   // TODO(apryakhin): For now we just support building a general structured
   // transform from an affince transform. This will be extended later
   // to support more transform types.
   assert(typeid(transform) == typeid(AffineTransform<N, N2, T2>) ||
          typeid(transform) == typeid(TranslationTransform<N2, T2>));
   return create_subspaces_by_preimage(DomainTransform<N2, T2, N, T>(transform),
                                       targets, preimages, reqs, wait_on);
  }

  template <int N, typename T>
  template <int N2, typename T2>
  Event IndexSpace<N, T>::create_subspaces_by_preimage(
      const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Point<N2, T2>>>&
          field_data,
      const std::vector<IndexSpace<N2, T2>>& targets,
      std::vector<IndexSpace<N, T>>& preimages, const ProfilingRequestSet& reqs,
      Event wait_on /*= Event::NO_EVENT*/) const {
   return create_subspaces_by_preimage(
       DomainTransform<N2, T2, N, T>(field_data), targets, preimages, reqs,
       wait_on);
  }

  template <int N, typename T>
  template <int N2, typename T2>
  Event IndexSpace<N, T>::create_subspaces_by_preimage(
      const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Rect<N2, T2>>>&
          field_data,
      const std::vector<IndexSpace<N2, T2>>& targets,
      std::vector<IndexSpace<N, T>>& preimages, const ProfilingRequestSet& reqs,
      Event wait_on /*= Event::NO_EVENT*/) const {
    return create_subspaces_by_preimage(
        DomainTransform<N2, T2, N, T>(field_data), targets, preimages, reqs,
        wait_on);
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event IndexSpace<N,T>::create_subspace_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,Point<N2,T2> > >& field_data,
							     const IndexSpace<N2,T2>& target,
							     IndexSpace<N,T>& preimage,
							     const ProfilingRequestSet &reqs,
							     Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<IndexSpace<N2,T2> > targets(1, target);
    std::vector<IndexSpace<N,T> > preimages;
    Event e = create_subspaces_by_preimage(field_data, targets, preimages, reqs, wait_on);
    preimage = preimages[0];
    return e;
  }

  // simple wrapper for the multiple subspace version
  template <int N, typename T>
  template <int N2, typename T2>
  inline Event IndexSpace<N,T>::create_subspace_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,Rect<N2,T2> > >& field_data,
							     const IndexSpace<N2,T2>& target,
							     IndexSpace<N,T>& preimage,
							     const ProfilingRequestSet &reqs,
							     Event wait_on /*= Event::NO_EVENT*/) const
  {
    std::vector<IndexSpace<N2,T2> > targets(1, target);
    std::vector<IndexSpace<N,T> > preimages;
    Event e = create_subspaces_by_preimage(field_data, targets, preimages, reqs, wait_on);
    preimage = preimages[0];
    return e;
  }

  // simple wrappers for the multiple subspace version
  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_union(const IndexSpace<N,T>& lhs,
							  const IndexSpace<N,T>& rhs,
							  IndexSpace<N,T>& result,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > lhss(1, lhs);
    std::vector<IndexSpace<N,T> > rhss(1, rhs);
    std::vector<IndexSpace<N,T> > results;
    Event e = compute_unions(lhss, rhss, results, reqs, wait_on);
    result = results[0];
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_unions(const IndexSpace<N,T>& lhs,
							   const std::vector<IndexSpace<N,T> >& rhss,
							   std::vector<IndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > lhss(1, lhs);
    Event e = compute_unions(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_unions(const std::vector<IndexSpace<N,T> >& lhss,
							   const IndexSpace<N,T>& rhs,
							   std::vector<IndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > rhss(1, rhs);
    Event e = compute_unions(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  // simple wrappers for the multiple subspace version
  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_intersection(const IndexSpace<N,T>& lhs,
								 const IndexSpace<N,T>& rhs,
								 IndexSpace<N,T>& result,
								 const ProfilingRequestSet &reqs,
								 Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > lhss(1, lhs);
    std::vector<IndexSpace<N,T> > rhss(1, rhs);
    std::vector<IndexSpace<N,T> > results;
    Event e = compute_intersections(lhss, rhss, results, reqs, wait_on);
    result = results[0];
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_intersections(const IndexSpace<N,T>& lhs,
								  const std::vector<IndexSpace<N,T> >& rhss,
								  std::vector<IndexSpace<N,T> >& results,
								  const ProfilingRequestSet &reqs,
								  Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > lhss(1, lhs);
    Event e = compute_intersections(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_intersections(const std::vector<IndexSpace<N,T> >& lhss,
								  const IndexSpace<N,T>& rhs,
								  std::vector<IndexSpace<N,T> >& results,
								  const ProfilingRequestSet &reqs,
								  Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > rhss(1, rhs);
    Event e = compute_intersections(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  // simple wrappers for the multiple subspace version
  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_difference(const IndexSpace<N,T>& lhs,
							       const IndexSpace<N,T>& rhs,
							       IndexSpace<N,T>& result,
							       const ProfilingRequestSet &reqs,
							       Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > lhss(1, lhs);
    std::vector<IndexSpace<N,T> > rhss(1, rhs);
    std::vector<IndexSpace<N,T> > results;
    Event e = compute_differences(lhss, rhss, results, reqs, wait_on);
    result = results[0];
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_differences(const IndexSpace<N,T>& lhs,
								const std::vector<IndexSpace<N,T> >& rhss,
								std::vector<IndexSpace<N,T> >& results,
								const ProfilingRequestSet &reqs,
								Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > lhss(1, lhs);
    Event e = compute_differences(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline /*static*/ Event IndexSpace<N,T>::compute_differences(const std::vector<IndexSpace<N,T> >& lhss,
								const IndexSpace<N,T>& rhs,
								std::vector<IndexSpace<N,T> >& results,
								const ProfilingRequestSet &reqs,
								Event wait_on /*= Event::NO_EVENT*/)
  {
    std::vector<IndexSpace<N,T> > rhss(1, rhs);
    Event e = compute_differences(lhss, rhss, results, reqs, wait_on);
    return e;
  }

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const IndexSpace<N,T>& is)
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
  // class IndexSpaceIterator<N,T>

  template <int N, typename T>
  inline IndexSpaceIterator<N,T>::IndexSpaceIterator(void)
    : valid(false)
  {}

  template <int N, typename T>
  inline IndexSpaceIterator<N,T>::IndexSpaceIterator(const IndexSpace<N,T>& _space)
    : valid(false)
  {
    reset(_space);
  }

  template <int N, typename T>
  inline IndexSpaceIterator<N,T>::IndexSpaceIterator(const IndexSpace<N,T>& _space,
						       const Rect<N,T>& _restriction)
    : valid(false)
  {
    reset(_space, _restriction);
  }

  template <int N, typename T>
  inline void IndexSpaceIterator<N,T>::reset(const IndexSpace<N,T>& _space)
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
  inline void IndexSpaceIterator<N,T>::reset(const IndexSpace<N,T>& _space,
					      const Rect<N,T>& _restriction)
  {
    space = _space;
    restriction = space.bounds.intersection(_restriction);
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
  inline bool IndexSpaceIterator<N,T>::step(void)
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
	if(N == 1)
	  break;
	else
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
    return (dim == N) && (idxtype == int(sizeof(T)));
  }

  template <int N, typename T>
  inline LinearizedIndexSpace<N,T>& LinearizedIndexSpaceIntfc::as_dim(void)
  {
    assert((dim == N) && (idxtype == int(sizeof(T))));
    return *static_cast<LinearizedIndexSpace<N,T> *>(this);
  }

  template <int N, typename T>
  inline const LinearizedIndexSpace<N,T>& LinearizedIndexSpaceIntfc::as_dim(void) const
  {
    assert((dim == N) && (idxtype == int(sizeof(T))));
    return *static_cast<const LinearizedIndexSpace<N,T> *>(this);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LinearizedIndexSpace<N,T>

  template <int N, typename T>
  inline LinearizedIndexSpace<N,T>::LinearizedIndexSpace(const IndexSpace<N,T>& _indexspace)
    : LinearizedIndexSpaceIntfc(N, int(sizeof(T)))
    , indexspace(_indexspace)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class AffineLinearizedIndexSpace<N,T>

  template <int N, typename T>
  inline AffineLinearizedIndexSpace<N,T>::AffineLinearizedIndexSpace(const IndexSpace<N,T>& _indexspace,
								     bool fortran_order /*= true*/)
    : LinearizedIndexSpace<N,T>(_indexspace)
  {
    const Rect<N,T>& bounds = this->indexspace.bounds;
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
      assert(s == ptrdiff_t(volume));
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
  inline size_t AffineLinearizedIndexSpace<N,T>::linearize(const Point<N,T>& p) const
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
  inline bool less<Realm::IndexSpace<N,T> >::operator()(const Realm::IndexSpace<N,T>& is1,
							 const Realm::IndexSpace<N,T>& is2) const
  {
    if(std::less<Realm::Rect<N,T> >()(is1.bounds, is2.bounds)) return true;
    if(std::less<Realm::Rect<N,T> >()(is2.bounds, is1.bounds)) return false;
    return (is1.sparsity < is2.sparsity);
  }

};
