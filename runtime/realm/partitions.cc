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

// index space partitioning for Realm

#include "partitions.h"

#include "profiling.h"

#include "runtime_impl.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ZIndexSpace<N,T>

  template <int N, typename T>
  inline Event ZIndexSpace<N,T>::create_equal_subspaces(size_t count, size_t granularity,
							std::vector<ZIndexSpace<N,T> >& subspaces,
							const ProfilingRequestSet &reqs,
							Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());
    //assert(reqs.empty());

    // dense case is easy(er)
    if(dense()) {
      // always split in x dimension for now
      assert(count >= 1);
      T total_x = std::max(bounds.hi.x - bounds.lo.x + 1, 0);
      subspaces.reserve(count);
      T px = bounds.lo.x;
      for(size_t i = 0; i < count; i++) {
	ZIndexSpace<N,T> ss(*this);
	T nx = bounds.lo.x + (total_x * (i + 1) / count);
	ss.bounds.lo.x = px;
	ss.bounds.hi.x = nx - 1;
	subspaces.push_back(ss);
	px = nx;
      }
      return Event::NO_EVENT;
    }

    // TODO: sparse case
    assert(0);
    return Event::NO_EVENT;
  }

  template <int N, typename T>
  inline Event ZIndexSpace<N,T>::create_weighted_subspaces(size_t count, size_t granularity,
							   const std::vector<int>& weights,
							   std::vector<ZIndexSpace<N,T> >& subspaces,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());
    //assert(reqs.empty());

    // determine the total weight
    size_t total_weight = 0;
    assert(weights.size() == count);
    for(size_t i = 0; i < count; i++)
      total_weight += weights[i];

    // dense case is easy(er)
    if(dense()) {
      // always split in x dimension for now
      assert(count >= 1);
      T total_x = std::max(bounds.hi.x - bounds.lo.x + 1, 0);
      subspaces.reserve(count);
      T px = bounds.lo.x;
      size_t cum_weight = 0;
      for(size_t i = 0; i < count; i++) {
	ZIndexSpace<N,T> ss(*this);
	cum_weight += weights[i];
	T nx = bounds.lo.x + (total_x * cum_weight / total_weight);
	ss.bounds.lo.x = px;
	ss.bounds.hi.x = nx - 1;
	subspaces.push_back(ss);
	px = nx;
      }
      return Event::NO_EVENT;
    }

    // TODO: sparse case
    assert(0);
    return Event::NO_EVENT;
  }

  template <int N, typename T>
  template <typename FT>
  inline Event ZIndexSpace<N,T>::create_subspaces_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
							   const std::vector<FT>& colors,
							   std::vector<ZIndexSpace<N,T> >& subspaces,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // create a new sparsity map for each subspace
    size_t n = colors.size();
    subspaces.resize(n);
    for(size_t i = 0; i < n; i++) {
      subspaces[i].bounds = this->bounds;
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
      impl->update_contributor_count(field_data.size());
      subspaces[i].sparsity = sparsity;      
    }

    for(size_t i = 0; i < field_data.size(); i++) {
      ByFieldMicroOp<N,T,FT> uop(*this,
				 field_data[i].index_space,
				 field_data[i].inst,
				 field_data[i].field_offset);
      for(size_t j = 0; j < colors.size(); j++)
	uop.add_sparsity_output(colors[j], subspaces[j].sparsity);
      //uop.set_value_set(colors);
      uop.execute();
    }

    return Event::NO_EVENT;
  }


  template <int N, typename T>
  template <int N2, typename T2>
  inline Event ZIndexSpace<N,T>::create_subspaces_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& field_data,
							   const std::vector<ZIndexSpace<N2,T2> >& sources,
							   std::vector<ZIndexSpace<N,T> >& images,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // create a new sparsity map for each subspace
    size_t n = sources.size();
    images.resize(n);
    for(size_t i = 0; i < n; i++) {
      images[i].bounds = this->bounds;
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
      impl->update_contributor_count(field_data.size());
      images[i].sparsity = sparsity;
    }

    for(size_t i = 0; i < field_data.size(); i++) {
      ImageMicroOp<N,T,N2,T2> uop(*this,
				  field_data[i].index_space,
				  field_data[i].inst,
				  field_data[i].field_offset);
      for(size_t j = 0; j < sources.size(); j++)
	uop.add_sparsity_output(sources[j], images[j].sparsity);
      uop.execute();
    }

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  template <int N2, typename T2>
  Event ZIndexSpace<N,T>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > >& field_data,
						       const std::vector<ZIndexSpace<N2,T2> >& targets,
						       std::vector<ZIndexSpace<N,T> >& preimages,
						       const ProfilingRequestSet &reqs,
						       Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // create a new sparsity map for each subspace
    size_t n = targets.size();
    preimages.resize(n);
    for(size_t i = 0; i < n; i++) {
      preimages[i].bounds = this->bounds;
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
      impl->update_contributor_count(field_data.size());
      preimages[i].sparsity = sparsity;
    }

    for(size_t i = 0; i < field_data.size(); i++) {
      PreimageMicroOp<N,T,N2,T2> uop(*this,
				     field_data[i].index_space,
				     field_data[i].inst,
				     field_data[i].field_offset);
      for(size_t j = 0; j < targets.size(); j++)
	uop.add_sparsity_output(targets[j], preimages[j].sparsity);
      uop.execute();
    }

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_unions(const std::vector<ZIndexSpace<N,T> >& lhss,
						    const std::vector<ZIndexSpace<N,T> >& rhss,
						    std::vector<ZIndexSpace<N,T> >& results,
						    const ProfilingRequestSet &reqs,
						    Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      results[i].bounds = lhss[li].bounds.union_bbox(rhss[ri].bounds);
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
      impl->update_contributor_count(1);
      results[i].sparsity = sparsity;

      UnionMicroOp<N,T> uop(lhss[li],
			    rhss[ri]);
      uop.add_sparsity_output(results[i].sparsity);
      uop.execute();
    }

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_intersections(const std::vector<ZIndexSpace<N,T> >& lhss,
							   const std::vector<ZIndexSpace<N,T> >& rhss,
							   std::vector<ZIndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      results[i].bounds = lhss[li].bounds.intersection(rhss[ri].bounds);
      // TODO: early out for empty bounds intersection

      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
      impl->update_contributor_count(1);
      results[i].sparsity = sparsity;

      IntersectionMicroOp<N,T> uop(lhss[li],
				   rhss[ri]);
      uop.add_sparsity_output(results[i].sparsity);
      uop.execute();
    }

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_differences(const std::vector<ZIndexSpace<N,T> >& lhss,
							 const std::vector<ZIndexSpace<N,T> >& rhss,
							 std::vector<ZIndexSpace<N,T> >& results,
							 const ProfilingRequestSet &reqs,
							 Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // just give copies of lhss for now
    assert(lhss.size() == rhss.size());
    size_t n = lhss.size();
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      results[i].bounds = lhss[i].bounds;
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
      impl->update_contributor_count(1);
      results[i].sparsity = sparsity;

      DifferenceMicroOp<N,T> uop(lhss[i],
				 rhss[i]);
      uop.add_sparsity_output(results[i].sparsity);
      uop.execute();
    }

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_union(const std::vector<ZIndexSpace<N,T> >& subspaces,
						   ZIndexSpace<N,T>& result,
						   const ProfilingRequestSet &reqs,
						   Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    assert(!subspaces.empty());
    // build a union_bbox for the new bounds
    ZRect<N,T> bounds = subspaces[0].bounds;
    for(size_t i = 1; i < subspaces.size(); i++)
      bounds = bounds.union_bbox(subspaces[i].bounds);

    result.bounds = bounds;
    SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
    impl->update_contributor_count(1);
    result.sparsity = sparsity;

    UnionMicroOp<N,T> uop(subspaces);
    uop.add_sparsity_output(result.sparsity);
    uop.execute();

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_intersection(const std::vector<ZIndexSpace<N,T> >& subspaces,
							  ZIndexSpace<N,T>& result,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    assert(!subspaces.empty());
    // build an intersection of all bounds for the new bounds
    // TODO: optimize empty case
    ZRect<N,T> bounds = subspaces[0].bounds;
    for(size_t i = 1; i < subspaces.size(); i++)
      bounds = bounds.intersection(subspaces[i].bounds);

    result.bounds = bounds;
    SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity);
    impl->update_contributor_count(1);
    result.sparsity = sparsity;

    IntersectionMicroOp<N,T> uop(subspaces);
    uop.add_sparsity_output(result.sparsity);
    uop.execute();

    return Event::NO_EVENT;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CoverageCounter<N,T>

  template <int N, typename T>
  inline CoverageCounter<N,T>::CoverageCounter(void)
    : count(0)
  {}

  template <int N, typename T>
  inline void CoverageCounter<N,T>::add_point(const ZPoint<N,T>& p)
  {
    count++;
  }

  template <int N, typename T>
  inline void CoverageCounter<N,T>::add_rect(const ZRect<N,T>& r)
  {
    count += r.volume();
  }

  template <int N, typename T>
  inline size_t CoverageCounter<N,T>::get_count(void) const
  {
    return count;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DenseRectangleList<N,T>

  template <int N, typename T>
  inline DenseRectangleList<N,T>::DenseRectangleList(void)
  {}

  template <int N, typename T>
  inline void DenseRectangleList<N,T>::add_point(const ZPoint<N,T>& p)
  {
    if(N == 1) {
      int merge_lo = -1;
      int merge_hi = -1;
      for(size_t i = 0; i < rects.size(); i++) {
	if((rects[i].lo.x <= p.x) && (p.x <= rects[i].hi.x))
	  return;
	// keep track of adjacent rectangles, if any
	if(rects[i].lo.x == p.x + 1)
	  merge_hi = i;
	if(rects[i].hi.x == p.x - 1)
	  merge_lo = i;
      }

      // four cases, based on merging
      if(merge_lo == -1) {
	if(merge_hi == -1) {
	  // no merging - add new rectangle
	  rects.push_back(ZRect<N,T>(p,p));
	} else {
	  // merge to rectangle covering range above
	  rects[merge_hi].lo = p;
	}
      } else {
	if(merge_hi == -1) {
	  // merge to rectangle below
	  rects[merge_lo].hi = p;
	} else {
	  // merge this point and the high rectangle into the low, delete high
	  rects[merge_lo].hi = rects[merge_hi].hi;
	  int last = rects.size() - 1;
	  if(merge_hi < last)
	    std::swap(rects[merge_hi], rects[last]);
	  rects.resize(last);
	}
      }
    } else {
      // just treat it as a small rectangle
      add_rect(ZRect<N,T>(p,p));
    }
  }

  template <int N, typename T>
  inline bool can_merge(const ZRect<N,T>& r1, const ZRect<N,T>& r2)
  {
    // N-1 dimensions must match exactly and 1 may be adjacent
    int idx = 0;
    while((idx < N) && (r1.lo[idx] == r2.lo[idx]) && (r1.hi[idx] == r2.hi[idx]))
      idx++;

    // if we get all the way through, the rectangles are equal and can be "merged"
    if(idx >= N) return true;

    // if not, this has to be the dimension that is adjacent
    if((r1.lo[idx] != (r2.hi[idx] + 1)) && (r2.lo[idx] != (r1.hi[idx] + 1)))
      return false;

    // and the rest of the dimensions have to match too
    while(++idx < N)
      if((r1.lo[idx] != r2.lo[idx]) || (r1.hi[idx] != r2.hi[idx]))
	return false;

    return true;
  }

  template <int N, typename T>
  inline void DenseRectangleList<N,T>::add_rect(const ZRect<N,T>& r)
  {
    // scan through rectangles, looking for containment (really good),
    //   mergability (also good), or overlap (bad)
    int merge_with = -1;
    for(size_t i = 0; i < rects.size(); i++) {
      if(rects[i].contains(r)) return;
      assert(!rects[i].overlaps(r));
      if((merge_with == -1) && can_merge(rects[i], r))
	merge_with = i;
    }

    if(merge_with == -1) {
      // no merge candidates, just add the new rectangle
      rects.push_back(r);
      return;
    }

    std::cout << "merge: " << rects[merge_with] << " and " << r << std::endl;
    rects[merge_with] = rects[merge_with].union_bbox(r);

    // this may trigger a cascade merge, so look again
    int last_merged = merge_with;
    while(true) {
      merge_with = -1;
      for(int i = 0; i < (int)rects.size(); i++) {
	if((i != last_merged) && can_merge(rects[i], rects[last_merged])) {
	  merge_with = i;
	  break;
	}
      }
      if(merge_with == -1)
	return;  // all done

      // merge downward in case one of these is the last one
      if(merge_with > last_merged)
	std::swap(merge_with, last_merged);

      std::cout << "merge: " << rects[merge_with] << " and " << rects[last_merged] << std::endl;
      rects[merge_with] = rects[merge_with].union_bbox(rects[last_merged]);

      // can delete last merged
      int last = rects.size() - 1;
      if(last != last_merged)
	std::swap(rects[last_merged], rects[last]);
      rects.resize(last);

      last_merged = merge_with;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMap<N,T>

  // looks up the public subset of the implementation object
  template <int N, typename T>
  SparsityMapPublicImpl<N,T> *SparsityMap<N,T>::impl(void) const
  {
    SparsityMapImplWrapper *wrapper = get_runtime()->get_sparsity_impl(*this);
    return wrapper->get_or_create<N,T>();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImplWrapper

  SparsityMapImplWrapper::SparsityMapImplWrapper(void)
    : me((ID::IDType)-1), owner(-1), dim(0), idxtype(0), map_impl(0)
  {}

  void SparsityMapImplWrapper::init(ID _me, unsigned _init_owner)
  {
    me = _me;
    owner = _init_owner;
  }

  template <int N, typename T>
  /*static*/ SparsityMapImpl<N,T> *SparsityMapImplWrapper::get_or_create(void)
  {
    // set the size if it's zero and check if it's not
    int olddim = __sync_val_compare_and_swap(&dim, 0, N);
    assert((olddim == 0) || (olddim == N));
    int oldtype = __sync_val_compare_and_swap(&idxtype, 0, (int)sizeof(T));
    assert((oldtype == 0) || (oldtype == (int)sizeof(T)));
    // now see if the pointer is valid
    void *impl = map_impl;
    if(impl)
      return static_cast<SparsityMapImpl<N,T> *>(impl);

    // create one and try to swap it in
    SparsityMapImpl<N,T> *new_impl = new SparsityMapImpl<N,T>;
    impl = __sync_val_compare_and_swap(&map_impl, 0, (void *)new_impl);
    if(impl != 0) {
      // we lost the race - free the one we made and return the winner
      delete new_impl;
      return static_cast<SparsityMapImpl<N,T> *>(impl);
    } else {
      // ours is the winner - return it
      return new_impl;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapPublicImpl<N,T>

  template <int N, typename T>
  SparsityMapPublicImpl<N,T>::SparsityMapPublicImpl(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImpl<N,T>

  template <int N, typename T>
  SparsityMapImpl<N,T>::SparsityMapImpl(void)
    : remaining_contributor_count(0)
  {}

  template <int N, typename T>
  inline /*static*/ SparsityMapImpl<N,T> *SparsityMapImpl<N,T>::lookup(SparsityMap<N,T> sparsity)
  {
    SparsityMapImplWrapper *wrapper = get_runtime()->get_sparsity_impl(sparsity);
    return wrapper->get_or_create<N,T>();
  }

  // methods used in the population of a sparsity map

  // when we plan out a partitioning operation, we'll know how many
  //  different uops are going to contribute something (or nothing) to
  //  the sparsity map - once all of those contributions arrive, we can
  //  finalize the sparsity map
  template <int N, typename T>
  void SparsityMapImpl<N,T>::update_contributor_count(int delta /*= 1*/)
  {
    // just increment the count atomically
    __sync_fetch_and_add(&remaining_contributor_count, delta);
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::contribute_nothing(void)
  {
    int left = __sync_sub_and_fetch(&remaining_contributor_count, 1);
    assert(left >= 0);
    if(left == 0)
      finalize();
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::contribute_dense_rect_list(const DenseRectangleList<N,T>& rects)
  {
    {
      AutoHSLLock al(mutex);
      
      // each new rectangle has to be tested against existing ones for containment, overlap,
      //  or mergeability
      // can't use iterators on entry list, since push_back invalidates end()
      size_t orig_count = this->entries.size();

      for(typename std::vector<ZRect<N,T> >::const_iterator it = rects.rects.begin();
	  it != rects.rects.end();
	  it++) {
	const ZRect<N,T>& r = *it;

	// index is declared outside for loop so we can detect early exits
	size_t idx;
	for(idx = 0; idx < orig_count; idx++) {
	  SparsityMapEntry<N,T>& e = this->entries[idx];
	  if(e.bounds.contains(r)) {
	    // existing entry contains us - still three cases though
	    if(e.sparsity.exists()) {
	      assert(0);
	    } else if(e.bitmap != 0) {
	      assert(0);
	    } else {
	      // dense entry containing new one - nothing to do
	      break;
	    }
	  }
	  if(e.bounds.overlaps(r)) {
	    assert(0);
	    break;
	  }
	  // only worth merging against a dense rectangle
	  if(can_merge(e.bounds, r) && !e.sparsity.exists() && (e.bitmap == 0)) {
	    e.bounds = e.bounds.union_bbox(r);
	    break;
	  }
	}
	if(idx == orig_count) {
	  // no matches against existing stuff, so add a new entry
	  idx = this->entries.size();
	  this->entries.resize(idx + 1);
	  this->entries[idx].bounds = r;
	  this->entries[idx].sparsity.id = 0; //SparsityMap<N,T>::NO_SPACE;
	  this->entries[idx].bitmap = 0;
	}
      }
    }

    int left = __sync_sub_and_fetch(&remaining_contributor_count, 1);
    assert(left >= 0);
    if(left == 0)
      finalize();
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::finalize(void)
  {
    std::cout << "finalizing " << this << ", " << this->entries.size() << " entries" << std::endl;
    for(size_t i = 0; i < this->entries.size(); i++)
      std::cout << "  [" << i
		<< "]: bounds=" << this->entries[i].bounds
		<< " sparsity=" << this->entries[i].sparsity
		<< " bitmap=" << this->entries[i].bitmap
		<< std::endl;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningMicroOp<N,T>

  template <int N, typename T>
  PartitioningMicroOp<N,T>::~PartitioningMicroOp(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class ByFieldMicroOp<N,T,FT>

  template <int N, typename T, typename FT>
  ByFieldMicroOp<N,T,FT>::ByFieldMicroOp(ZIndexSpace<N,T> _parent_space,
					 ZIndexSpace<N,T> _inst_space,
					 RegionInstance _inst,
					 size_t _field_offset)
    : parent_space(_parent_space)
    , inst_space(_inst_space)
    , inst(_inst)
    , field_offset(_field_offset)
    , value_range_valid(false)
    , value_set_valid(false)
  {}

  template <int N, typename T, typename FT>
  ByFieldMicroOp<N,T,FT>::~ByFieldMicroOp(void)
  {}

  template <int N, typename T, typename FT>
  void ByFieldMicroOp<N,T,FT>::set_value_range(FT _lo, FT _hi)
  {
    assert(!value_range_valid);
    range_lo = _lo;
    range_hi = _hi;
    value_range_valid = true;
  }

  template <int N, typename T, typename FT>
  void ByFieldMicroOp<N,T,FT>::set_value_set(const std::vector<FT>& _value_set)
  {
    assert(!value_set_valid);
    value_set.insert(_value_set.begin(), _value_set.end());
    value_set_valid = true;
  }

  template <int N, typename T, typename FT>
  void ByFieldMicroOp<N,T,FT>::add_sparsity_output(FT _val, SparsityMap<N,T> _sparsity)
  {
    value_set.insert(_val);
    sparsity_outputs[_val] = _sparsity;
  }

  template <int N, typename T, typename FT>
  template <typename BM>
  void ByFieldMicroOp<N,T,FT>::populate_bitmasks(std::map<FT, BM *>& bitmasks)
  {
    // for now, one access for the whole instance
    AffineAccessor<FT,N,T> a_data(inst, field_offset);

    // double iteration - use the instance's space first, since it's probably smaller
    for(ZIndexSpaceIterator<N,T> it(inst_space); it.valid; it.step()) {
      for(ZIndexSpaceIterator<N,T> it2(parent_space, it.rect); it2.valid; it2.step()) {
	const ZRect<N,T>& r = it2.rect;
	ZPoint<N,T> p = r.lo;
	while(true) {
	  FT val = a_data.read(p);
	  ZPoint<N,T> p2 = p;
	  while(p2.x < r.hi.x) {
	    ZPoint<N,T> p3 = p2;
	    p3.x++;
	    FT val2 = a_data.read(p3);
	    if(val != val2) {
	      // record old strip
	      BM *&bmp = bitmasks[val];
	      if(!bmp) bmp = new BM;
	      bmp->add_rect(ZRect<N,T>(p,p2));
	      //std::cout << val << ": " << p << ".." << p2 << std::endl;
	      val = val2;
	      p = p3;
	    }
	    p2 = p3;
	  }
	  // record whatever strip we have at the end
	  BM *&bmp = bitmasks[val];
	  if(!bmp) bmp = new BM;
	  bmp->add_rect(ZRect<N,T>(p,p2));
	  //std::cout << val << ": " << p << ".." << p2 << std::endl;

	  // are we done?
	  if(p2 == r.hi) break;

	  // now go to the next span, if there is one (can't be in 1-D)
	  assert(N > 1);
	  for(int i = 0; i < (N - 1); i++) {
	    p[i] = r.lo[i];
	    if(p[i + 1] < r.hi[i+1]) {
	      p[i + 1] += 1;
	      break;
	    }
	  }
	}
      }
    }
  }

  template <int N, typename T, typename FT>
  void ByFieldMicroOp<N,T,FT>::execute(void)
  {
    std::map<FT, CoverageCounter<N,T> *> values_present;

    populate_bitmasks(values_present);

    std::cout << values_present.size() << " values present in instance " << inst << std::endl;
    for(typename std::map<FT, CoverageCounter<N,T> *>::const_iterator it = values_present.begin();
	it != values_present.end();
	it++)
      std::cout << "  " << it->first << " = " << it->second->get_count() << std::endl;

    std::map<FT, DenseRectangleList<N,T> *> rect_map;

    populate_bitmasks(rect_map);

    std::cout << values_present.size() << " values present in instance " << inst << std::endl;
    for(typename std::map<FT, DenseRectangleList<N,T> *>::const_iterator it = rect_map.begin();
	it != rect_map.end();
	it++)
      std::cout << "  " << it->first << " = " << it->second->rects.size() << " rectangles" << std::endl;

    // iterate over sparsity outputs and contribute to all (even if we didn't have any
    //  points found for it)
    for(typename std::map<FT, SparsityMap<N,T> >::const_iterator it = sparsity_outputs.begin();
	it != sparsity_outputs.end();
	it++) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(it->second);
      typename std::map<FT, DenseRectangleList<N,T> *>::const_iterator it2 = rect_map.find(it->first);
      if(it2 != rect_map.end()) {
	impl->contribute_dense_rect_list(*(it2->second));
	delete it2->second;
      } else
	impl->contribute_nothing();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ImageMicroOp<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  ImageMicroOp<N,T,N2,T2>::ImageMicroOp(ZIndexSpace<N,T> _parent_space,
					ZIndexSpace<N2,T2> _inst_space,
					RegionInstance _inst,
					size_t _field_offset)
    : parent_space(_parent_space)
    , inst_space(_inst_space)
    , inst(_inst)
    , field_offset(_field_offset)
  {}

  template <int N, typename T, int N2, typename T2>
  ImageMicroOp<N,T,N2,T2>::~ImageMicroOp(void)
  {}

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::add_sparsity_output(ZIndexSpace<N2,T2> _source,
						    SparsityMap<N,T> _sparsity)
  {
    sources.push_back(_source);
    sparsity_outputs.push_back(_sparsity);
  }

  template <int N, typename T, int N2, typename T2>
  template <typename BM>
  void ImageMicroOp<N,T,N2,T2>::populate_bitmasks(std::map<int, BM *>& bitmasks)
  {
    // for now, one access for the whole instance
    AffineAccessor<ZPoint<N,T>,N2,T2> a_data(inst, field_offset);

    // double iteration - use the instance's space first, since it's probably smaller
    for(ZIndexSpaceIterator<N,T> it(inst_space); it.valid; it.step()) {
      for(size_t i = 0; i < sources.size(); i++) {
	for(ZIndexSpaceIterator<N,T> it2(sources[i], it.rect); it2.valid; it2.step()) {
	  BM **bmpp = 0;

	  // iterate over each point in the source and see if it points into the parent space	  
	  for(ZPointInRectIterator<N,T> pir(it2.rect); pir.valid; pir.step()) {
	    ZPoint<N,T> ptr = a_data.read(pir.p);

	    if(parent_space.contains(ptr)) {
	      //std::cout << "image " << i << "(" << sources[i] << ") -> " << pir.p << " -> " << ptr << std::endl;
	      if(!bmpp) bmpp = &bitmasks[i];
	      if(!*bmpp) *bmpp = new BM;
	      (*bmpp)->add_point(ptr);
	    }
	  }
	}
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::execute(void)
  {
    std::map<int, DenseRectangleList<N,T> *> rect_map;

    populate_bitmasks(rect_map);

    std::cout << rect_map.size() << " non-empty images present in instance " << inst << std::endl;
    for(typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it = rect_map.begin();
	it != rect_map.end();
	it++)
      std::cout << "  " << sources[it->first] << " = " << it->second->rects.size() << " rectangles" << std::endl;

    // iterate over sparsity outputs and contribute to all (even if we didn't have any
    //  points found for it)
    for(size_t i = 0; i < sparsity_outputs.size(); i++) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_outputs[i]);
      typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it2 = rect_map.find(i);
      if(it2 != rect_map.end()) {
	impl->contribute_dense_rect_list(*(it2->second));
	delete it2->second;
      } else
	impl->contribute_nothing();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PreimageMicroOp<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  PreimageMicroOp<N,T,N2,T2>::PreimageMicroOp(ZIndexSpace<N,T> _parent_space,
					 ZIndexSpace<N,T> _inst_space,
					 RegionInstance _inst,
					 size_t _field_offset)
    : parent_space(_parent_space)
    , inst_space(_inst_space)
    , inst(_inst)
    , field_offset(_field_offset)
  {}

  template <int N, typename T, int N2, typename T2>
  PreimageMicroOp<N,T,N2,T2>::~PreimageMicroOp(void)
  {}

  template <int N, typename T, int N2, typename T2>
  void PreimageMicroOp<N,T,N2,T2>::add_sparsity_output(ZIndexSpace<N2,T2> _target,
						       SparsityMap<N,T> _sparsity)
  {
    targets.push_back(_target);
    sparsity_outputs.push_back(_sparsity);
  }

  template <int N, typename T, int N2, typename T2>
  template <typename BM>
  void PreimageMicroOp<N,T,N2,T2>::populate_bitmasks(std::map<int, BM *>& bitmasks)
  {
    // for now, one access for the whole instance
    AffineAccessor<ZPoint<N2,T2>,N,T> a_data(inst, field_offset);

    // double iteration - use the instance's space first, since it's probably smaller
    for(ZIndexSpaceIterator<N,T> it(inst_space); it.valid; it.step()) {
      for(ZIndexSpaceIterator<N,T> it2(parent_space, it.rect); it2.valid; it2.step()) {
	// now iterate over each point
	for(ZPointInRectIterator<N,T> pir(it2.rect); pir.valid; pir.step()) {
	  // fetch the pointer and test it against every possible target (ugh)
	  ZPoint<N2,T2> ptr = a_data.read(pir.p);

	  for(size_t i = 0; i < targets.size(); i++)
	    if(targets[i].contains(ptr)) {
	      BM *&bmp = bitmasks[i];
	      if(!bmp) bmp = new BM;
	      bmp->add_point(pir.p);
	    }
	}
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageMicroOp<N,T,N2,T2>::execute(void)
  {
    std::map<int, DenseRectangleList<N,T> *> rect_map;

    populate_bitmasks(rect_map);

    std::cout << rect_map.size() << " non-empty preimages present in instance " << inst << std::endl;
    for(typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it = rect_map.begin();
	it != rect_map.end();
	it++)
      std::cout << "  " << targets[it->first] << " = " << it->second->rects.size() << " rectangles" << std::endl;

    // iterate over sparsity outputs and contribute to all (even if we didn't have any
    //  points found for it)
    for(size_t i = 0; i < sparsity_outputs.size(); i++) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_outputs[i]);
      typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it2 = rect_map.find(i);
      if(it2 != rect_map.end()) {
	impl->contribute_dense_rect_list(*(it2->second));
	delete it2->second;
      } else
	impl->contribute_nothing();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UnionMicroOp<N,T>

  template <int N, typename T>
  UnionMicroOp<N,T>::UnionMicroOp(const std::vector<ZIndexSpace<N,T> >& _inputs)
    : inputs(_inputs)
  {
    sparsity_output.id = 0;
  }

  template <int N, typename T>
  UnionMicroOp<N,T>::UnionMicroOp(ZIndexSpace<N,T> _lhs,
				  ZIndexSpace<N,T> _rhs)
    : inputs(2)
  {
    inputs[0] = _lhs;
    inputs[1] = _rhs;
    sparsity_output.id = 0;
  }

  template <int N, typename T>
  UnionMicroOp<N,T>::~UnionMicroOp(void)
  {}

  template <int N, typename T>
  void UnionMicroOp<N,T>::add_sparsity_output(SparsityMap<N,T> _sparsity)
  {
    sparsity_output = _sparsity;
  }

  template <int N, typename T>
  template <typename BM>
  void UnionMicroOp<N,T>::populate_bitmask(BM& bitmask)
  {
    // iterate over all the inputs, adding dense (sub)rectangles first
    for(typename std::vector<ZIndexSpace<N,T> >::const_iterator it = inputs.begin();
	it != inputs.end();
	it++) {
      if(it->dense()) {
	bitmask.add_rect(it->bounds);
      } else {
	SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(it->sparsity);
	for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it2 = impl->entries.begin();
	    it2 != impl->entries.end();
	    it2++) {
	  ZRect<N,T> isect = it->bounds.intersection(it2->bounds);
	  if(isect.empty())
	    continue;
	  assert(!it2->sparsity.exists());
	  assert(it2->bitmap == 0);
	  bitmask.add_rect(isect);
	}
      }
    }
  }

  template <int N, typename T>
  void UnionMicroOp<N,T>::execute(void)
  {
    std::cout << "calc union: " << inputs[0];
    for(size_t i = 1; i < inputs.size(); i++)
      std::cout << " + " << inputs[i];
    std::cout << std::endl;
    DenseRectangleList<N,T> drl;
    populate_bitmask(drl);
    if(sparsity_output.exists()) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_output);
      impl->contribute_dense_rect_list(drl);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntersectionMicroOp<N,T>

  template <int N, typename T>
  IntersectionMicroOp<N,T>::IntersectionMicroOp(const std::vector<ZIndexSpace<N,T> >& _inputs)
    : inputs(_inputs)
  {
    sparsity_output.id = 0;
  }

  template <int N, typename T>
  IntersectionMicroOp<N,T>::IntersectionMicroOp(ZIndexSpace<N,T> _lhs,
				  ZIndexSpace<N,T> _rhs)
    : inputs(2)
  {
    inputs[0] = _lhs;
    inputs[1] = _rhs;
    sparsity_output.id = 0;
  }

  template <int N, typename T>
  IntersectionMicroOp<N,T>::~IntersectionMicroOp(void)
  {}

  template <int N, typename T>
  void IntersectionMicroOp<N,T>::add_sparsity_output(SparsityMap<N,T> _sparsity)
  {
    sparsity_output = _sparsity;
  }

  template <int N, typename T>
  template <typename BM>
  void IntersectionMicroOp<N,T>::populate_bitmask(BM& bitmask)
  {
    // first build the intersection of all the bounding boxes
    ZRect<N,T> bounds = inputs[0].bounds;
    for(size_t i = 1; i < inputs.size(); i++)
      bounds = bounds.intersection(inputs[i].bounds);
    if(bounds.empty()) {
      // early out
      std::cout << "empty intersection bounds!" << std::endl;
      return;
    }

    // handle 2 input case with simple double-iteration
    if(inputs.size() == 2) {
      // double iteration - use the instance's space first, since it's probably smaller
      for(ZIndexSpaceIterator<N,T> it(inputs[0], bounds); it.valid; it.step())
	for(ZIndexSpaceIterator<N,T> it2(inputs[1], it.rect); it2.valid; it2.step())
	  bitmask.add_rect(it2.rect);
    } else {
      assert(0);
    }
  }

  template <int N, typename T>
  void IntersectionMicroOp<N,T>::execute(void)
  {
    std::cout << "calc intersection: " << inputs[0];
    for(size_t i = 1; i < inputs.size(); i++)
      std::cout << " & " << inputs[i];
    std::cout << std::endl;
    DenseRectangleList<N,T> drl;
    populate_bitmask(drl);
    if(sparsity_output.exists()) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_output);
      impl->contribute_dense_rect_list(drl);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DifferenceMicroOp<N,T>

  template <int N, typename T>
  DifferenceMicroOp<N,T>::DifferenceMicroOp(ZIndexSpace<N,T> _lhs,
					    ZIndexSpace<N,T> _rhs)
    : lhs(_lhs), rhs(_rhs)
  {
    sparsity_output.id = 0;
  }

  template <int N, typename T>
  DifferenceMicroOp<N,T>::~DifferenceMicroOp(void)
  {}

  template <int N, typename T>
  void DifferenceMicroOp<N,T>::add_sparsity_output(SparsityMap<N,T> _sparsity)
  {
    sparsity_output = _sparsity;
  }

  template <int N, typename T>
  void subtract_rects(const ZRect<N,T>& lhs, const ZRect<N,T>& rhs,
		      std::vector<ZRect<N,T> >& pieces)
  {
    // should only be called if we have overlapping rectangles
    assert(!lhs.empty() && !rhs.empty() && lhs.overlaps(rhs));
    ZRect<N,T> r = lhs;
    for(int i = 0; i < N; i++) {
      if(lhs.lo[i] < rhs.lo[i]) {
	// some coverage "below"
	r.lo[i] = lhs.lo[i];
	r.hi[i] = rhs.lo[i] - 1;
	pieces.push_back(r);
      }
      if(lhs.hi[i] > rhs.hi[i]) {
	// some coverage "below"
	r.lo[i] = rhs.hi[i] + 1;
	r.hi[i] = lhs.hi[i];
	pieces.push_back(r);
      }
      // clamp to the rhs range for the next dimension
      r.lo[i] = std::max(lhs.lo[i], rhs.lo[i]);
      r.hi[i] = std::min(lhs.hi[i], rhs.hi[i]);
    }
  }

  template <int N, typename T>
  template <typename BM>
  void DifferenceMicroOp<N,T>::populate_bitmask(BM& bitmask)
  {
    // the basic idea here is to build a list of rectangles from the lhs and clip them
    //  based on the rhs until we're done
    std::deque<ZRect<N,T> > todo;

    if(lhs.dense()) {
      todo.push_back(lhs.bounds);
    } else {
      SparsityMapImpl<N,T> *l_impl = SparsityMapImpl<N,T>::lookup(lhs.sparsity);
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = l_impl->entries.begin();
	  it != l_impl->entries.end();
	  it++) {
	ZRect<N,T> isect = lhs.bounds.intersection(it->bounds);
	if(isect.empty())
	  continue;
	assert(!it->sparsity.exists());
	assert(it->bitmap == 0);
	todo.push_back(isect);
      }
    }

    while(!todo.empty()) {
      ZRect<N,T> r = todo.front();
      todo.pop_front();

      // iterate over all subrects in the rhs - any that contain it eliminate this rect,
      //  overlap chops it into pieces
      bool fully_covered = false;
      for(ZIndexSpaceIterator<N,T> it(rhs); it.valid; it.step()) {
	std::cout << "check " << r << " -= " << it.rect << std::endl;
	if(it.rect.contains(r)) {
	  fully_covered = true;
	  break;
	}

	if(it.rect.overlaps(r)) {
	  // subtraction is nasty - can result in 2N subrectangles
	  std::vector<ZRect<N,T> > pieces;
	  subtract_rects(r, it.rect, pieces);
	  assert(!pieces.empty());

	  // continue on with the first piece, and stick the rest on the todo list
	  typename std::vector<ZRect<N,T> >::iterator it2 = pieces.begin();
	  r = *(it2++);
	  todo.insert(todo.end(), it2, pieces.end());
	}
      }
      if(!fully_covered) {
	std::cout << "difference += " << r << std::endl;
	bitmask.add_rect(r);
      }
    }
  }

  template <int N, typename T>
  void DifferenceMicroOp<N,T>::execute(void)
  {
    std::cout << "calc difference: " << lhs << " - " << rhs << std::endl;
    DenseRectangleList<N,T> drl;
    populate_bitmask(drl);
    if(sparsity_output.exists()) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_output);
      impl->contribute_dense_rect_list(drl);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // template instantiation goo

  namespace {
    template <int N, typename T>
    class InstantiatePartitioningStuff {
    public:
      typedef ZIndexSpace<N,T> IS;
      template <typename FT>
      static void inst_field(void)
      {
	ZIndexSpace<N,T> i;
	std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> > field_data;
	std::vector<FT> colors;
	std::vector<ZIndexSpace<N,T> > subspaces;
	i.create_subspaces_by_field(field_data, colors, subspaces,
				    Realm::ProfilingRequestSet());
      }
      template <int N2, typename T2>
      static void inst_image_and_preimage(void)
      {
	std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > > field_data;
	std::vector<ZIndexSpace<N, T> > list1;
	std::vector<ZIndexSpace<N2, T2> > list2;
	list1[0].create_subspaces_by_preimage(field_data, list2, list1,
					      Realm::ProfilingRequestSet());
	list2[0].create_subspaces_by_image(field_data, list1, list2,
					   Realm::ProfilingRequestSet());
      }
      static void inst_stuff(void)
      {
	inst_field<int>();
	inst_field<bool>();
	inst_image_and_preimage<1,int>();

	ZIndexSpace<N,T> i;
	std::vector<int> weights;
	std::vector<ZIndexSpace<N,T> > list;
	i.create_equal_subspaces(0, 0, list, Realm::ProfilingRequestSet());
	i.create_weighted_subspaces(0, 0, weights, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_unions(list, list, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_intersections(list, list, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_differences(list, list, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_union(list, i, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_intersection(list, i, Realm::ProfilingRequestSet());
      }
    };

    //
  };

  void (*dummy)(void) __attribute__((unused)) = &InstantiatePartitioningStuff<1,int>::inst_stuff;
  //InstantiatePartitioningStuff<1,int> foo __attribute__((unused));
};

