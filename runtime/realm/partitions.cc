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

  Logger log_part("part");
  Logger log_uop_timing("uop_timing");

  namespace {
    // module-level globals

    PartitioningOpQueue *op_queue = 0;

    FragmentAssembler fragment_assembler;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class ZIndexSpace<N,T>

  template <int N, typename T>
  __attribute__ ((noinline))
  Event ZIndexSpace<N,T>::create_equal_subspaces(size_t count, size_t granularity,
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
  __attribute__ ((noinline))
  Event ZIndexSpace<N,T>::create_weighted_subspaces(size_t count, size_t granularity,
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
  Event ZIndexSpace<N,T>::create_subspaces_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
						    const std::vector<FT>& colors,
						    std::vector<ZIndexSpace<N,T> >& subspaces,
						    const ProfilingRequestSet &reqs,
						    Event wait_on /*= Event::NO_EVENT*/) const
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    ByFieldOperation<N,T,FT> *op = new ByFieldOperation<N,T,FT>(*this, field_data, reqs, e);

    size_t n = colors.size();
    subspaces.resize(n);
    for(size_t i = 0; i < n; i++)
      subspaces[i] = op->add_color(colors[i]);

    op->deferred_launch(wait_on);
    return e;
  }


  template <int N, typename T>
  template <int N2, typename T2>
  Event ZIndexSpace<N,T>::create_subspaces_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& field_data,
							   const std::vector<ZIndexSpace<N2,T2> >& sources,
							   std::vector<ZIndexSpace<N,T> >& images,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    ImageOperation<N,T,N2,T2> *op = new ImageOperation<N,T,N2,T2>(*this, field_data, reqs, e);

    size_t n = sources.size();
    images.resize(n);
    for(size_t i = 0; i < n; i++)
      images[i] = op->add_source(sources[i]);

    op->deferred_launch(wait_on);
    return e;
  }

  template <int N, typename T>
  template <int N2, typename T2>
  Event ZIndexSpace<N,T>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > >& field_data,
						       const std::vector<ZIndexSpace<N2,T2> >& targets,
						       std::vector<ZIndexSpace<N,T> >& preimages,
						       const ProfilingRequestSet &reqs,
						       Event wait_on /*= Event::NO_EVENT*/) const
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    PreimageOperation<N,T,N2,T2> *op = new PreimageOperation<N,T,N2,T2>(*this, field_data, reqs, e);

    size_t n = targets.size();
    preimages.resize(n);
    for(size_t i = 0; i < n; i++)
      preimages[i] = op->add_target(targets[i]);

    op->deferred_launch(wait_on);
    return e;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_unions(const std::vector<ZIndexSpace<N,T> >& lhss,
						    const std::vector<ZIndexSpace<N,T> >& rhss,
						    std::vector<ZIndexSpace<N,T> >& results,
						    const ProfilingRequestSet &reqs,
						    Event wait_on /*= Event::NO_EVENT*/)
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    UnionOperation<N,T> *op = new UnionOperation<N,T>(reqs, e);

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      results[i] = op->add_union(lhss[li], rhss[ri]);
    }

    op->deferred_launch(wait_on);
    return e;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_intersections(const std::vector<ZIndexSpace<N,T> >& lhss,
							   const std::vector<ZIndexSpace<N,T> >& rhss,
							   std::vector<ZIndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    IntersectionOperation<N,T> *op = new IntersectionOperation<N,T>(reqs, e);

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      results[i] = op->add_intersection(lhss[li], rhss[ri]);
    }

    op->deferred_launch(wait_on);
    return e;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_differences(const std::vector<ZIndexSpace<N,T> >& lhss,
							 const std::vector<ZIndexSpace<N,T> >& rhss,
							 std::vector<ZIndexSpace<N,T> >& results,
							 const ProfilingRequestSet &reqs,
							 Event wait_on /*= Event::NO_EVENT*/)
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    DifferenceOperation<N,T> *op = new DifferenceOperation<N,T>(reqs, e);

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      results[i] = op->add_difference(lhss[li], rhss[ri]);
    }

    op->deferred_launch(wait_on);
    return e;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_union(const std::vector<ZIndexSpace<N,T> >& subspaces,
						   ZIndexSpace<N,T>& result,
						   const ProfilingRequestSet &reqs,
						   Event wait_on /*= Event::NO_EVENT*/)
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    UnionOperation<N,T> *op = new UnionOperation<N,T>(reqs, e);

    result = op->add_union(subspaces);

    op->deferred_launch(wait_on);
    return e;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_intersection(const std::vector<ZIndexSpace<N,T> >& subspaces,
							  ZIndexSpace<N,T>& result,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    Event e = GenEventImpl::create_genevent()->current_event();
    IntersectionOperation<N,T> *op = new IntersectionOperation<N,T>(reqs, e);

    result = op->add_intersection(subspaces);

    op->deferred_launch(wait_on);
    return e;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FragmentAssembler

  FragmentAssembler::FragmentAssembler(void)
    : next_sequence_id(0)
  {}

  FragmentAssembler::~FragmentAssembler(void)
  {}

  // returns a sequence ID that may not be unique, but hasn't been used in a 
  //   long time
  inline int FragmentAssembler::get_sequence_id(void)
  {
    return __sync_fetch_and_add(&next_sequence_id, 1);
  }

  // adds a fragment to the list, returning true if this is the last one from
  //  a sequence
  inline bool FragmentAssembler::add_fragment(gasnet_node_t sender,
					      int sequence_id,
					      int sequence_count)
  {
    // easy case - a fragment with a sequence_count == 1 is a whole message
    if(sequence_count == 1) return true;

    // rest of this has to be protected by a lock
    {
      AutoHSLLock al(mutex);

      std::map<int, int>& by_sender = fragments[sender];

      std::map<int, int>::iterator it = by_sender.find(sequence_id);
      if(it != by_sender.end()) {
	int new_count = it->second + sequence_count - 1;
	if(new_count == 0) {
	  // this was the last packet - delete the entry from the map and return true
	  by_sender.erase(it);
	  return true;
	} else 
	  it->second = new_count;
      } else {
	// first packet (we've seen) of new sequence
	by_sender[sequence_id] = sequence_count - 1;
      }
    }
    return false;
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
    if(rects.empty()) {
      rects.push_back(ZRect<N,T>(p, p));
      return;
    }

    if(N == 1) {
      // optimize for sorted insertion (i.e. stuff at end)
      {
	ZRect<N,T> &lr = *rects.rbegin();
	if(p.x == (lr.hi.x + 1)) {
	  lr.hi.x = p.x;
	  return;
	}
	if(p.x > (lr.hi.x + 1)) {
	  rects.push_back(ZRect<N,T>(p, p));
	  return;
	}
      }

      std::cout << "{{";
      for(size_t i = 0; i < rects.size(); i++) std::cout << " " << rects[i];
      std::cout << " }} <- " << p << "\n";
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
      std::cout << "{{";
      for(size_t i = 0; i < rects.size(); i++) std::cout << " " << rects[i];
      std::cout << " }}\n";
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
  inline void DenseRectangleList<N,T>::add_rect(const ZRect<N,T>& _r)
  {
    if(rects.empty()) {
      rects.push_back(_r);
      return;
    }

    if(N == 1) {
      // try to optimize for sorted insertion (i.e. stuff at end)
      ZRect<N,T> &lr = *rects.rbegin();
      if(_r.lo.x == (lr.hi.x + 1)) {
	lr.hi.x = _r.hi.x;
	return;
      }
      if(_r.lo.x > (lr.hi.x + 1)) {
	rects.push_back(_r);
	return;
      }
    }

    ZRect<N,T> r = _r;

    // scan through rectangles, looking for containment (really good),
    //   mergability (also good), or overlap (bad)
    int merge_with = -1;
    std::vector<int> absorbed;
    int count = rects.size();
    for(int i = 0; i < count; i++) {
      if(rects[i].contains(r)) return;
      if(rects[i].overlaps(r)) {
        assert(N == 1);  // TODO: splitting for 2+-D
        r = r.union_bbox(rects[i]);
        absorbed.push_back(i);
        continue;
      }
      if((merge_with == -1) && can_merge(rects[i], r))
	merge_with = i;
    }

    if(merge_with == -1) {
      if(absorbed.empty()) {
        // no merge candidates and nothing absorbed, just add the new rectangle
        rects.push_back(r);
      } else {
        // replace the first absorbed rectangle, delete the others (if any)
        rects[absorbed[0]] = r;
        for(size_t i = 1; i < absorbed.size(); i++) {
          if(absorbed[i] < (count - 1))
            std::swap(rects[absorbed[i]], rects[count - 1]);
          count--;
        }
        rects.resize(count);
      }
      return;
    }

#ifdef DEBUG_PARTITIONING
    std::cout << "merge: " << rects[merge_with] << " and " << r << std::endl;
#endif
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

#ifdef DEBUG_PARTITIONING
      std::cout << "merge: " << rects[merge_with] << " and " << rects[last_merged] << std::endl;
#endif
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
  // class HybridRectangleList<N,T>

  template <typename T>
  HybridRectangleList<1,T>::HybridRectangleList(void)
    : is_vector(true)
  {}

  template <typename T>
  void HybridRectangleList<1,T>::add_point(const ZPoint<1,T>& p)
  {
    if(is_vector) {
      DenseRectangleList<1,T>::add_point(p);
      if(this->rects.size() > HIGH_WATER_MARK)
	convert_to_map();
      return;
    }

    // otherwise add to the map
    assert(!as_map.empty());
    typename std::map<T, T>::iterator it = as_map.lower_bound(p.x);
    if(it == as_map.end()) {
      // bigger than everything - see if we can merge with the last guy
      T& last = as_map.rbegin()->second;
      if(last == (p.x - 1))
	last = p.x;
      else
	as_map[p.x] = p.x;
    } 
    else if(it->first == p.x) {
      // we're the beginning of an existing range - nothing to do
    } else if(it == as_map.begin()) {
      // we're before everything - see if we can merge with the first guy
      if(it->first == (p.x + 1)) {
	T last = it->second;
	as_map.erase(it);
	as_map[p.x] = last;
      } else {
	as_map[p.x] = p.x;
      }
    } else {
      typename std::map<T, T>::iterator it2 = it; --it2;
      if(it2->second >= p.x) {
	// range below us includes us - nothing to do
      } else {
	bool merge_above = it->first == (p.x + 1);
	bool merge_below = it2->second == (p.x - 1);

	if(merge_below) {
	  if(merge_above) {
	    it2->second = it->second;
	    as_map.erase(it);
	  } else
	    it2->second = p.x;
	} else {
	  T last;
	  if(merge_above) {
	    last = it->second;
	    as_map.erase(it);
	  } else
	    last = p.x;
	  as_map[p.x] = last;
	}
      }
    }
    // mergers can cause us to drop below LWM
    if(as_map.size() < LOW_WATER_MARK)
      convert_to_vector();
  }

  template <typename T>
  void HybridRectangleList<1,T>::convert_to_map(void)
  {
    if(!is_vector) return;
    assert(as_map.empty());
    for(typename std::vector<ZRect<1,T> >::iterator it = this->rects.begin();
	it != this->rects.end();
	it++)
      as_map[it->lo.x] = it->hi.x;
    this->rects.clear();
    is_vector = false;
  }

  template <typename T>
  void HybridRectangleList<1,T>::convert_to_vector(void)
  {
    if(is_vector) return;
    assert(this->rects.empty());
    for(typename std::map<T, T>::iterator it = as_map.begin();
	it != as_map.end();
	it++) {
      ZRect<1,T> r;
      r.lo.x = it->first;
      r.hi.x = it->second;
      this->rects.push_back(r);
    }
    as_map.clear();
    is_vector = true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMap<N,T>

  // looks up the public subset of the implementation object
  template <int N, typename T>
  SparsityMapPublicImpl<N,T> *SparsityMap<N,T>::impl(void) const
  {
    SparsityMapImplWrapper *wrapper = get_runtime()->get_sparsity_impl(*this);
    return wrapper->get_or_create<N,T>(*this);
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
  /*static*/ SparsityMapImpl<N,T> *SparsityMapImplWrapper::get_or_create(SparsityMap<N,T> me)
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
    SparsityMapImpl<N,T> *new_impl = new SparsityMapImpl<N,T>(me);
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
    : entries_valid(false), approx_valid(false)
  {}

  // call actual implementation - inlining makes this cheaper than a virtual method
  template <int N, typename T>
  Event SparsityMapPublicImpl<N,T>::make_valid(bool precise /*= true*/)
  {
    return static_cast<SparsityMapImpl<N,T> *>(this)->make_valid(precise);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImpl<N,T>

  template <int N, typename T>
  SparsityMapImpl<N,T>::SparsityMapImpl(SparsityMap<N,T> _me)
    : me(_me), remaining_contributor_count(0)
    , precise_requested(false), approx_requested(false)
    , precise_ready_event(Event::NO_EVENT), approx_ready_event(Event::NO_EVENT)
    , sizeof_precise(0)
  {}

  template <int N, typename T>
  inline /*static*/ SparsityMapImpl<N,T> *SparsityMapImpl<N,T>::lookup(SparsityMap<N,T> sparsity)
  {
    SparsityMapImplWrapper *wrapper = get_runtime()->get_sparsity_impl(sparsity);
    return wrapper->get_or_create<N,T>(sparsity);
  }

  // actual implementation - SparsityMapPublicImpl's version just calls this one
  template <int N, typename T>
  Event SparsityMapImpl<N,T>::make_valid(bool precise /*= true*/)
  {
    // early out
    if(precise ? this->entries_valid : this->approx_valid)
      return Event::NO_EVENT;

    // take lock to get/create event cleanly
    bool request_approx = false;
    bool request_precise = false;
    Event e = Event::NO_EVENT;
    {
      AutoHSLLock al(mutex);

      if(precise) {
	if(!this->entries_valid) {
	  // do we need to request the data?
	  if((ID(me).node() != gasnet_mynode()) && !precise_requested) {
	    request_precise = true;
	    precise_requested = true;
	    // also get approx while we're at it
	    request_approx = !(this->approx_valid || approx_requested);
	    approx_requested = true;
	  }
	  // do we have a finish event?
	  if(precise_ready_event.exists()) {
	    e = precise_ready_event;
	  } else {
	    e = GenEventImpl::create_genevent()->current_event();
	    precise_ready_event = e;
	  }
	}
      } else {
	if(!this->approx_valid) {
	  // do we need to request the data?
	  if((ID(me).node() != gasnet_mynode()) && !approx_requested) {
	    request_approx = true;
	    approx_requested = true;
	  }
	  // do we have a finish event?
	  if(approx_ready_event.exists()) {
	    e = approx_ready_event;
	  } else {
	    e = GenEventImpl::create_genevent()->current_event();
	    approx_ready_event = e;
	  }
	}
      }
    }
    
    if(request_approx || request_precise)
      RemoteSparsityRequestMessage::send_request(ID(me).node(), me,
						 request_approx,
						 request_precise);

    return e;
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
    gasnet_node_t owner = ID(me).node();

    if(owner != gasnet_mynode()) {
      // send (the lack of) data to the owner to collect
      int seq_id = fragment_assembler.get_sequence_id();
      RemoteSparsityContribMessage::send_request<N,T>(owner, me, seq_id, 1,
						      0, 0);
      return;
    }

    int left = __sync_sub_and_fetch(&remaining_contributor_count, 1);
    assert(left >= 0);
    if(left == 0)
      finalize();
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::contribute_dense_rect_list(const DenseRectangleList<N,T>& rects)
  {
    gasnet_node_t owner = ID(me).node();

    if(owner != gasnet_mynode()) {
      // send the data to the owner to collect
      int seq_id = fragment_assembler.get_sequence_id();
      const size_t max_to_send = 2048 / sizeof(ZRect<N,T>);
      const ZRect<N,T> *rdata = &rects.rects[0];
      int seq_count = 0;
      size_t remaining = rects.rects.size();
      // send partial messages first
      while(remaining > max_to_send) {
	RemoteSparsityContribMessage::send_request<N,T>(owner, me, seq_id, 0,
							rdata, max_to_send);
	seq_count++;
	remaining -= max_to_send;
	rdata += max_to_send;
      }
      // final message includes the count of all messages (including this one!)
      RemoteSparsityContribMessage::send_request<N,T>(owner, me, 
						      seq_id, seq_count + 1,
						      rdata, remaining);
      return;
    }

    contribute_raw_rects(&rects.rects[0], rects.rects.size(), true);
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::contribute_raw_rects(const ZRect<N,T>* rects,
						  size_t count, bool last)
  {
    if(count > 0) {
      AutoHSLLock al(mutex);

      if(N == 1) {
	// demand that our input data is sorted
	for(size_t i = 1; i < count; i++)
	  assert(rects[i-1].hi.x < (rects[i].lo.x - 1));

	// do a merge of the new data with the old
	std::vector<SparsityMapEntry<N,T> > old_data;
	old_data.swap(this->entries);
	size_t i = 0;
	size_t n = 0;
	typename std::vector<SparsityMapEntry<N,T> >::const_iterator old_it = old_data.begin();
	while((i < count) && (old_it != old_data.end())) {
	  if(rects[i].hi.x < (old_it->bounds.lo.x - 1)) {
	    this->entries.resize(n + 1);
	    this->entries[n].bounds = rects[i];
	    this->entries[n].sparsity.id = 0; // no sparsity map
	    this->entries[n].bitmap = 0;
	    n++;
	    i++;
	    continue;
	  }

	  if(old_it->bounds.hi.x < (rects[i].lo.x - 1)) {
	    this->entries.push_back(*old_it);
	    n++;
	    old_it++;
	    continue;
	  }

	  ZRect<N,T> u = rects[i].union_bbox(old_it->bounds);
	  // step rects, but not old_it - want sanity checks below to be done
	  i++;
	  while(true) {
	    if((i < count) && (rects[i].lo.x <= (u.hi.x + 1))) {
	      u.hi.x = std::max(u.hi.x, rects[i].hi.x);
	      i++;
	      continue;
	    }
	    if((old_it != old_data.end()) && (old_it->bounds.lo.x <= (u.hi.x + 1))) {
	      assert(!old_it->sparsity.exists());
	      assert(old_it->bitmap == 0);
	      u.hi.x = std::max(u.hi.x, old_it->bounds.hi.x);
	      old_it++;
	      continue;
	    }
	    // if neither test passed, the chain is broken
	    break;
	  }
	  this->entries.resize(n + 1);
	  this->entries[n].bounds = u;
	  this->entries[n].sparsity.id = 0; // no sparsity map
	  this->entries[n].bitmap = 0;
	  n++;
	}

	// leftovers...
	while(i < count) {
	  this->entries.resize(n + 1);
	  this->entries[n].bounds = rects[i];
	  this->entries[n].sparsity.id = 0; // no sparsity map
	  this->entries[n].bitmap = 0;
	  n++;
	  i++;
	}

	while(old_it != old_data.end()) {
	  this->entries.push_back(*old_it);
	  old_it++;
	}
      } else {
	// each new rectangle has to be tested against existing ones for containment, overlap,
	//  or mergeability
	// can't use iterators on entry list, since push_back invalidates end()
	size_t orig_count = this->entries.size();

	for(size_t i = 0; i < count; i++) {
	  const ZRect<N,T>& r = rects[i];

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
    }

    if(last) {
      if(ID(me).node() == gasnet_mynode()) {
	// we're the owner, so remaining_contributor_count tracks our expected contributions
	int left = __sync_sub_and_fetch(&remaining_contributor_count, 1);
	assert(left >= 0);
	if(left == 0)
	  finalize();
      } else {
	// this is a remote sparsity map, so sanity check that we requested the data
	assert(precise_requested);
	finalize();
      }
    }
  }

  // adds a microop as a waiter for valid sparsity map data - returns true
  //  if the uop is added to the list (i.e. will be getting a callback at some point),
  //  or false if the sparsity map became valid before this call (i.e. no callback)
  template <int N, typename T>
  bool SparsityMapImpl<N,T>::add_waiter(PartitioningMicroOp *uop, bool precise)
  {
    // early out
    if(precise ? this->entries_valid : this->approx_valid)
      return false;

    // take lock and retest, and register if not ready
    bool registered = false;
    bool request_approx = false;
    bool request_precise = false;
    {
      AutoHSLLock al(mutex);

      if(precise) {
	if(!this->entries_valid) {
	  precise_waiters.push_back(uop);
	  registered = true;
	  // do we need to request the data?
	  if((ID(me).node() != gasnet_mynode()) && !precise_requested) {
	    request_precise = true;
	    precise_requested = true;
	    // also get approx while we're at it
	    request_approx = !(this->approx_valid || approx_requested);
	    approx_requested = true;
	  }
	}
      } else {
	if(!this->approx_valid) {
	  approx_waiters.push_back(uop);
	  registered = true;
	  // do we need to request the data?
	  if((ID(me).node() != gasnet_mynode()) && !approx_requested) {
	    request_approx = true;
	    approx_requested = true;
	  }
	}
      }
    }

    if(request_approx || request_precise)
      RemoteSparsityRequestMessage::send_request(ID(me).node(), me,
						 request_approx,
						 request_precise);

    return registered;
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::remote_data_request(gasnet_node_t requestor, bool send_precise, bool send_approx)
  {
    // first sanity check - we should be the owner of the data
    assert(ID(me).node() == gasnet_mynode());

    // take the long to determine atomically if we can send data or if we need to register as a listener
    bool reply_precise = false;
    bool reply_approx = false;
    {
      AutoHSLLock al(mutex);

      // always add the requestor to the sharer list
      remote_sharers.add(requestor);

      if(send_precise) {
	if(this->entries_valid)
	  reply_precise = true;
	else
	  remote_precise_waiters.add(requestor);
      }

      if(send_approx) {
	if(this->approx_valid)
	  reply_approx = true;
	else
	  remote_approx_waiters.add(requestor);
      }
    }

    if(reply_approx || reply_precise)
      remote_data_reply(requestor, reply_precise, reply_approx);
  }

  
  template <int N, typename T>
  void SparsityMapImpl<N,T>::remote_data_reply(gasnet_node_t requestor, bool reply_precise, bool reply_approx)
  {
    if(reply_approx) {
      // TODO
      assert(this->approx_valid);
    }

    if(reply_precise) {
      log_part.info() << "sending precise data: sparsity=" << me << " target=" << requestor;
      
      int seq_id = fragment_assembler.get_sequence_id();
      int seq_count = 0;

      // scan the entry list, sending bitmaps first and making a list of rects
      std::vector<ZRect<N,T> > rects;
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = this->entries.begin();
	  it != this->entries.end();
	  it++) {
	if(it->bitmap) {
	  // TODO: send bitmap
	  assert(0);
	}
	else if(it->sparsity.exists()) {
	  // TODO: ?
	  assert(0);
	} else {
	  rects.push_back(it->bounds);
	}
      }
	
      const ZRect<N,T> *rdata = &rects[0];
      size_t remaining = rects.size();
      const size_t max_to_send = 2048 / sizeof(ZRect<N,T>);
      // send partial messages first
      while(remaining > max_to_send) {
	RemoteSparsityContribMessage::send_request<N,T>(requestor, me, seq_id, 0,
							rdata, max_to_send);
	seq_count++;
	remaining -= max_to_send;
	rdata += max_to_send;
      }
      // final message includes the count of all messages (including this one!)
      RemoteSparsityContribMessage::send_request<N,T>(requestor, me, 
						      seq_id, seq_count + 1,
						      rdata, remaining);
    }
  }
  
  template <int N, typename T>
  static inline bool non_overlapping_bounds_1d_comp(const SparsityMapEntry<N,T>& lhs,
						    const SparsityMapEntry<N,T>& rhs)
  {
    return lhs.bounds.lo.x < rhs.bounds.lo.x;
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::finalize(void)
  {
//define DEBUG_PARTITIONING
#ifdef DEBUG_PARTITIONING
    std::cout << "finalizing " << this << ", " << this->entries.size() << " entries" << std::endl;
    for(size_t i = 0; i < this->entries.size(); i++)
      std::cout << "  [" << i
		<< "]: bounds=" << this->entries[i].bounds
		<< " sparsity=" << this->entries[i].sparsity
		<< " bitmap=" << this->entries[i].bitmap
		<< std::endl;
#endif

    // first step is to organize the data a little better - for N=1, this means sorting
    //  the entries list
    if(N == 1) {
      std::sort(this->entries.begin(), this->entries.end(), non_overlapping_bounds_1d_comp<N,T>);
      for(size_t i = 1; i < this->entries.size(); i++)
	assert(this->entries[i-1].bounds.hi.x < (this->entries[i].bounds.lo.x - 1));
    }

#ifdef DEBUG_PARTITIONING
    std::cout << "finalizing " << this << ", " << this->entries.size() << " entries" << std::endl;
    for(size_t i = 0; i < this->entries.size(); i++)
      std::cout << "  [" << i
		<< "]: bounds=" << this->entries[i].bounds
		<< " sparsity=" << this->entries[i].sparsity
		<< " bitmap=" << this->entries[i].bitmap
		<< std::endl;
#endif

    NodeSet sendto_precise, sendto_approx;
    Event trigger_precise = Event::NO_EVENT;
    Event trigger_approx = Event::NO_EVENT;
    {
      AutoHSLLock al(mutex);

      assert(!this->entries_valid);
      this->entries_valid = true;
      precise_requested = false;
      if(precise_ready_event.exists()) {
	trigger_precise = precise_ready_event;
	precise_ready_event = Event::NO_EVENT;
      }

      for(std::vector<PartitioningMicroOp *>::const_iterator it = precise_waiters.begin();
	  it != precise_waiters.end();
	  it++)
	(*it)->sparsity_map_ready(this, true);
      precise_waiters.clear();

      sendto_precise = remote_precise_waiters;
      remote_precise_waiters.clear();
    }

    if(!sendto_approx.empty()) {
      for(gasnet_node_t i = 0; (i < gasnet_nodes()) && !sendto_approx.empty(); i++)
	if(sendto_approx.contains(i)) {
	  bool also_precise = sendto_precise.contains(i);
	  if(also_precise)
	    sendto_precise.remove(i);
	  remote_data_reply(i, also_precise, true);
	  sendto_approx.remove(i);
	}
    }

    if(!sendto_precise.empty()) {
      for(gasnet_node_t i = 0; (i < gasnet_nodes()) && !sendto_precise.empty(); i++)
	if(sendto_precise.contains(i)) {
	  remote_data_reply(i, true, false);
	  sendto_precise.remove(i);
	}
    }

    if(trigger_approx.exists())
      get_runtime()->get_genevent_impl(trigger_approx)->trigger(trigger_approx.gen,
								gasnet_mynode());

    if(trigger_precise.exists())
      get_runtime()->get_genevent_impl(trigger_precise)->trigger(trigger_precise.gen,
								 gasnet_mynode());
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class AsyncMicroOp

  AsyncMicroOp::AsyncMicroOp(Operation *_op, PartitioningMicroOp *_uop)
    : Operation::AsyncWorkItem(_op)
    , uop(_uop)
  {}
    
  void AsyncMicroOp::request_cancellation(void)
  {
    // ignored
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningMicroOp

  PartitioningMicroOp::PartitioningMicroOp(void)
    : wait_count(2), requestor(gasnet_mynode()), async_microop(0)
  {}

  PartitioningMicroOp::PartitioningMicroOp(gasnet_node_t _requestor,
					   AsyncMicroOp *_async_microop)
    : wait_count(2), requestor(_requestor), async_microop(_async_microop)
  {}

  PartitioningMicroOp::~PartitioningMicroOp(void)
  {}

  void PartitioningMicroOp::mark_started(void)
  {}

  void PartitioningMicroOp::mark_finished(void)
  {
    if(async_microop) {
      if(requestor == gasnet_mynode())
	async_microop->mark_finished();
      else
	RemoteMicroOpCompleteMessage::send_request(requestor, async_microop);
    }
  }

  template <int N, typename T>
  void PartitioningMicroOp::sparsity_map_ready(SparsityMapImpl<N,T> *sparsity, bool precise)
  {
    int left = __sync_sub_and_fetch(&wait_count, 1);
    if(left == 0)
      op_queue->enqueue_partitioning_microop(this);
  }

  void PartitioningMicroOp::finish_dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // if there were no registrations by caller (or if they're really fast), the count will be 2
    //  and we can execute this microop inline
    int left1 = __sync_sub_and_fetch(&wait_count, 1);
    if(left1 == 1) {
      if(inline_ok) {
	mark_started();
	execute();
	mark_finished();
      } else
	op_queue->enqueue_partitioning_microop(this);
      return;
    }

    // if the count was greater than 1, it probably has to be queued, so create an 
    //  AsyncMicroOp so that the op knows we're not done yet
    if(requestor == gasnet_mynode()) {
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);
    } else {
      // request came from somewhere else - it had better have a async_microop already
      assert(async_microop != 0);
    }

    // now do the last decrement - if it returns 0, we can still do the operation inline
    //  (otherwise it'll get queued when somebody else does the last decrement)
    int left2 = __sync_sub_and_fetch(&wait_count, 1);
    if(left2 == 0) {
      if(inline_ok) {
	mark_started();
	execute();
	mark_finished();
      } else
	op_queue->enqueue_partitioning_microop(this);
    }
  }

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
    TimeStamp ts("ByFieldMicroOp::execute", true, &log_uop_timing);
#ifdef DEBUG_PARTITIONING
    std::map<FT, CoverageCounter<N,T> *> values_present;

    populate_bitmasks(values_present);

    std::cout << values_present.size() << " values present in instance " << inst << std::endl;
    for(typename std::map<FT, CoverageCounter<N,T> *>::const_iterator it = values_present.begin();
	it != values_present.end();
	it++)
      std::cout << "  " << it->first << " = " << it->second->get_count() << std::endl;
#endif

    std::map<FT, DenseRectangleList<N,T> *> rect_map;

    populate_bitmasks(rect_map);

#ifdef DEBUG_PARTITIONING
    std::cout << values_present.size() << " values present in instance " << inst << std::endl;
    for(typename std::map<FT, DenseRectangleList<N,T> *>::const_iterator it = rect_map.begin();
	it != rect_map.end();
	it++)
      std::cout << "  " << it->first << " = " << it->second->rects.size() << " rectangles" << std::endl;
#endif

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

  template <int N, typename T, typename FT>
  void ByFieldMicroOp<N,T,FT>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // a ByFieldMicroOp should always be executed on whichever node the field data lives
    gasnet_node_t exec_node = ID(inst).node();

    if(exec_node != gasnet_mynode()) {
      // we're going to ship it elsewhere, which means we always need an AsyncMicroOp to
      //  track it
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);

      RemoteMicroOpMessage::send_request(exec_node, op, *this);
      delete this;
      return;
    }

    // instance index spaces should always be valid
    assert(inst_space.is_valid(true /*precise*/));

    // need valid data for the parent space too
    if(!parent_space.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N,T>::lookup(parent_space.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
	__sync_fetch_and_add(&wait_count, 1);
    }
    
    finish_dispatch(op, inline_ok);
  }

  template <int N, typename T, typename FT>
  template <typename S>
  bool ByFieldMicroOp<N,T,FT>::serialize_params(S& s) const
  {
    return((s << parent_space) &&
	   (s << inst_space) &&
	   (s << inst) &&
	   (s << field_offset) &&
	   (s << value_set) &&
	   (s << sparsity_outputs));
  }

  template <int N, typename T, typename FT>
  template <typename S>
  ByFieldMicroOp<N,T,FT>::ByFieldMicroOp(gasnet_node_t _requestor,
					 AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> parent_space) &&
	       (s >> inst_space) &&
	       (s >> inst) &&
	       (s >> field_offset) &&
	       (s >> value_set) &&
	       (s >> sparsity_outputs));
    assert(ok);
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
    //std::cout << "a_data = " << a_data << "\n";

    // double iteration - use the instance's space first, since it's probably smaller
    for(ZIndexSpaceIterator<N,T> it(inst_space); it.valid; it.step()) {
      for(size_t i = 0; i < sources.size(); i++) {
	for(ZIndexSpaceIterator<N,T> it2(sources[i], it.rect); it2.valid; it2.step()) {
	  BM **bmpp = 0;

	  // iterate over each point in the source and see if it points into the parent space	  
	  for(ZPointInRectIterator<N,T> pir(it2.rect); pir.valid; pir.step()) {
	    ZPoint<N,T> ptr = a_data.read(pir.p);

	    if(parent_space.contains(ptr)) {
	      std::cout << "image " << i << "(" << sources[i] << ") -> " << pir.p << " -> " << ptr << std::endl;
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
    TimeStamp ts("ImageMicroOp::execute", true, &log_uop_timing);
    //std::map<int, DenseRectangleList<N,T> *> rect_map;
    std::map<int, HybridRectangleList<N,T> *> rect_map;

    populate_bitmasks(rect_map);

#ifdef DEBUG_PARTITIONING
    std::cout << rect_map.size() << " non-empty images present in instance " << inst << std::endl;
    for(typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it = rect_map.begin();
	it != rect_map.end();
	it++)
      std::cout << "  " << sources[it->first] << " = " << it->second->rects.size() << " rectangles" << std::endl;
#endif

    // iterate over sparsity outputs and contribute to all (even if we didn't have any
    //  points found for it)
    for(size_t i = 0; i < sparsity_outputs.size(); i++) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_outputs[i]);
      typename std::map<int, HybridRectangleList<N,T> *>::const_iterator it2 = rect_map.find(i);
      if(it2 != rect_map.end()) {
	it2->second->convert_to_vector();
	impl->contribute_dense_rect_list(*(it2->second));
	delete it2->second;
      } else
	impl->contribute_nothing();
    }
  }

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // an ImageMicroOp should always be executed on whichever node the field data lives
    gasnet_node_t exec_node = ID(inst).node();

    if(exec_node != gasnet_mynode()) {
      // we're going to ship it elsewhere, which means we always need an AsyncMicroOp to
      //  track it
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);

      RemoteMicroOpMessage::send_request(exec_node, op, *this);
      delete this;
      return;
    }

    // instance index spaces should always be valid
    assert(inst_space.is_valid(true /*precise*/));

    // need valid data for each source
    for(size_t i = 0; i < sources.size(); i++) {
      if(!sources[i].dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N,T>::lookup(sources[i].sparsity)->add_waiter(this, true /*precise*/);
	if(registered)
	  __sync_fetch_and_add(&wait_count, 1);
      }
    }

    // need valid data for the parent space too
    if(!parent_space.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N,T>::lookup(parent_space.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
	__sync_fetch_and_add(&wait_count, 1);
    }
    
    finish_dispatch(op, inline_ok);
  }

  template <int N, typename T, int N2, typename T2>
  template <typename S>
  bool ImageMicroOp<N,T,N2,T2>::serialize_params(S& s) const
  {
    return((s << parent_space) &&
	   (s << inst_space) &&
	   (s << inst) &&
	   (s << field_offset) &&
	   (s << sources) &&
	   (s << sparsity_outputs));
  }

  template <int N, typename T, int N2, typename T2>
  template <typename S>
  ImageMicroOp<N,T,N2,T2>::ImageMicroOp(gasnet_node_t _requestor,
					AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> parent_space) &&
	       (s >> inst_space) &&
	       (s >> inst) &&
	       (s >> field_offset) &&
	       (s >> sources) &&
	       (s >> sparsity_outputs));
    assert(ok);
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
    TimeStamp ts("PreimageMicroOp::execute", true, &log_uop_timing);
    std::map<int, DenseRectangleList<N,T> *> rect_map;

    populate_bitmasks(rect_map);

#ifdef DEBUG_PARTITIONING
    std::cout << rect_map.size() << " non-empty preimages present in instance " << inst << std::endl;
    for(typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it = rect_map.begin();
	it != rect_map.end();
	it++)
      std::cout << "  " << targets[it->first] << " = " << it->second->rects.size() << " rectangles" << std::endl;
#endif

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

  template <int N, typename T, int N2, typename T2>
  void PreimageMicroOp<N,T,N2,T2>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // a PreimageMicroOp should always be executed on whichever node the field data lives
    gasnet_node_t exec_node = ID(inst).node();

    if(exec_node != gasnet_mynode()) {
      // we're going to ship it elsewhere, which means we always need an AsyncMicroOp to
      //  track it
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);

      RemoteMicroOpMessage::send_request(exec_node, op, *this);
      delete this;
      return;
    }

    // instance index spaces should always be valid
    assert(inst_space.is_valid(true /*precise*/));

    // need valid data for each target
    for(size_t i = 0; i < targets.size(); i++) {
      if(!targets[i].dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N,T>::lookup(targets[i].sparsity)->add_waiter(this, true /*precise*/);
	if(registered)
	  __sync_fetch_and_add(&wait_count, 1);
      }
    }

    // need valid data for the parent space too
    if(!parent_space.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N,T>::lookup(parent_space.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
	__sync_fetch_and_add(&wait_count, 1);
    }
    
    finish_dispatch(op, inline_ok);
  }

  template <int N, typename T, int N2, typename T2>
  template <typename S>
  bool PreimageMicroOp<N,T,N2,T2>::serialize_params(S& s) const
  {
    return((s << parent_space) &&
	   (s << inst_space) &&
	   (s << inst) &&
	   (s << field_offset) &&
	   (s << targets) &&
	   (s << sparsity_outputs));
  }

  template <int N, typename T, int N2, typename T2>
  template <typename S>
  PreimageMicroOp<N,T,N2,T2>::PreimageMicroOp(gasnet_node_t _requestor,
					      AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> parent_space) &&
	       (s >> inst_space) &&
	       (s >> inst) &&
	       (s >> field_offset) &&
	       (s >> targets) &&
	       (s >> sparsity_outputs));
    assert(ok);
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
  class NWayMerge {
  public:
    NWayMerge(const std::vector<ZIndexSpace<N,T> >& spaces);
    ~NWayMerge(void);

    const ZRect<N,T>& operator[](int idx) const;
    size_t size(void) const;

    // steps an iterator - does not immediately update its position
    bool step(int idx);

    // called after you call step at least once on a given iterator
    void update(int idx);

    void print(void) const;

  protected:
    int n;
    std::vector<ZIndexSpaceIterator<N,T> > its;
    std::vector<int> order;
  };

  template <int N, typename T>
  NWayMerge<N,T>::NWayMerge(const std::vector<ZIndexSpace<N,T> >& spaces)
    : n(0)
  {
    its.resize(spaces.size());
    order.resize(spaces.size());
    for(size_t i = 0; i < spaces.size(); i++) {
      its[i].reset(spaces[i]);
      if(its[i].valid) {
	order[n] = i;
	T lo = its[i].rect.lo.x;
	for(int j = n; j > 0; j--)
	  if(its[order[j-1]].rect.lo.x > lo)
	    std::swap(order[j-1], order[j]);
	  else
	    break;
	n++;
      }
    }
  }

  template <int N, typename T>
  NWayMerge<N,T>::~NWayMerge(void)
  {}

  template <int N, typename T>
  const ZRect<N,T>& NWayMerge<N,T>::operator[](int idx) const
  {
    assert(idx < n);
    return its[order[idx]].rect;
  }

  template <int N, typename T>
  size_t NWayMerge<N,T>::size(void) const
  {
    return n;
  }

  // steps an iterator - does not immediately update its position
  template <int N, typename T>
  bool NWayMerge<N,T>::step(int idx)
  {
    assert(idx < n);
    return its[order[idx]].step();
  }

  // called after you call step at least once on a given iterator
  template <int N, typename T>
  void NWayMerge<N,T>::update(int idx)
  {
    if(its[order[idx]].valid) {
      // can only move upwards
      T lo = its[order[idx]].rect.lo;
      for(int j = idx + 1; j < n; j++)
	if(its[order[j]].rect.lo < lo)
	  std::swap(order[j], order[j-1]);
	else
	  break;
    } else {
      // just delete it
      order.erase(order.begin() + idx);
      n--;
    }
  }

  template <int N, typename T>
  void NWayMerge<N,T>::print(void) const
  {
    std::cout << "[[";
    for(size_t i = 0; i < n; i++) std::cout << " " << i << "=" << order[i] << "=" << its[order[i]].rect;
    std::cout << "]]\n";
  }

  template <int N, typename T>
  template <typename BM>
  void UnionMicroOp<N,T>::populate_bitmask(BM& bitmask)
  {
    // special case: in 1-D, we can count on the iterators being ordered and just do an O(N)
    //  merge-union of the two streams
    if(N == 1) {
      // even more special case where inputs.size() == 2
      if(inputs.size() == 2) {
	ZIndexSpaceIterator<N,T> it_lhs(inputs[0]);
	ZIndexSpaceIterator<N,T> it_rhs(inputs[1]);
       
	while(it_lhs.valid && it_rhs.valid) {
	  // if either side comes completely before the other, emit it and continue
	  if(it_lhs.rect.hi.x < (it_rhs.rect.lo.x - 1)) {
	    bitmask.add_rect(it_lhs.rect);
	    it_lhs.step();
	    continue;
	  }

	  if(it_rhs.rect.hi.x < (it_lhs.rect.lo.x - 1)) {
	    bitmask.add_rect(it_rhs.rect);
	    it_rhs.step();
	    continue;
	  }

	  // new rectangle will be at least the union of these two
	  ZRect<N,T> u = it_lhs.rect.union_bbox(it_rhs.rect);
	  it_lhs.step();
	  it_rhs.step();
	  // try to consume even more
	  while(true) {
	    if(it_lhs.valid && (it_lhs.rect.lo.x <= (u.hi.x + 1))) {
	      u.hi.x = std::max(u.hi.x, it_lhs.rect.hi.x);
	      it_lhs.step();
	      continue;
	    }
	    if(it_rhs.valid && (it_rhs.rect.lo.x <= (u.hi.x + 1))) {
	      u.hi.x = std::max(u.hi.x, it_rhs.rect.hi.x);
	      it_rhs.step();
	      continue;
	    }
	    // if both fail, we're done
	    break;
	  }
	  bitmask.add_rect(u);
	}

	// leftover rects from one side or the other just get added
	while(it_lhs.valid) {
	  bitmask.add_rect(it_lhs.rect);
	  it_lhs.step();
	}
	while(it_rhs.valid) {
	  bitmask.add_rect(it_rhs.rect);
	  it_rhs.step();
	}
      } else {
	// N-way merge
	NWayMerge<N,T> nwm(inputs);
	//nwm.print();
	while(nwm.size() > 1) {
	  //nwm.print();

	  // consume rectangles off the first one until there's overlap with the next guy
	  T lo1 = nwm[1].lo.x;
	  if(nwm[0].hi.x < (lo1 - 1)) {
	    while(nwm[0].hi.x < (lo1 - 1)) {
	      bitmask.add_rect(nwm[0]);
	      if(!nwm.step(0)) break;
	    }
	    nwm.update(0);
	    continue;
	  }

	  // at least a little overlap, so start accumulating a value
	  ZRect<N,T> u = nwm[0];
	  nwm.step(0); nwm.update(0);
	  while((nwm.size() > 0) && (nwm[0].lo.x <= (u.hi.x + 1))) {
	    u.hi.x = std::max(u.hi.x, nwm[0].hi.x);
	    nwm.step(0);
	    nwm.update(0);
	  }
	  bitmask.add_rect(u);
	}

	// any stragglers?
	if(nwm.size() > 0)
	  do {
	    bitmask.add_rect(nwm[0]);
	  } while(nwm.step(0));
#if 0
	std::vector<ZIndexSpaceIterator<N,T> > its(inputs.size());
	std::vector<int> order(inputs.size());
	size_t n = 0;
	for(size_t i = 0; i < inputs.size(); i++) {
	  its[i].reset(inputs[i]);
	  if(its[i].valid) {
	    order[n] = i;
	    T lo = its[i].rect.lo.x;
	    for(size_t j = n; j > 0; j--)
	      if(its[order[j-1]].rect.lo.x > lo)
		std::swap(order[j-1], order[j]);
	      else
		break;
	    n++;
	  }
	}
	std::cout << "[[";
	for(size_t i = 0; i < n; i++) std::cout << " " << i << "=" << order[i] << "=" << its[order[i]].rect;
	std::cout << "]]\n";
	while(n > 1) {
	  std::cout << "[[";
	  for(size_t i = 0; i < n; i++) std::cout << " " << i << "=" << order[i] << "=" << its[order[i]].rect;
	  std::cout << "]]\n";
	  // consume rectangles off the first one until there's overlap with the next guy
	  if(its[order[0]].rect.hi.x < (its[order[1]].rect.lo.x - 1)) {
	    while(its[order[0]].rect.hi.x < (its[order[1]].rect.lo.x - 1)) {
	      bitmask.add_rect(its[order[0]].rect);
	      if(!its[order[0]].step()) break;
	    }
	    if(its[order[0]].valid) {
	      for(size_t j = 0; j < n - 1; j++)
		if(its[order[j]].rect.lo.x > its[order[j+1]].rect.lo.x)
		  std::swap(order[j], order[j+1]);
		else
		  break;
	    } else {
	      order.erase(order.begin());
	      n--;
	    }
	    continue;
	  }

	  // at least some overlap, switch to consuming and appending to next guy
	  ZRect<N,T> 
	  break;
	}

	// whichever one is left can just emit all its remaining rectangles
	while(its[order[0]].valid) {
	  bitmask.add_rect(its[order[0]].rect);
	  its[order[0]].step();
	}
#endif
      }
      return;
    }

    // iterate over all the inputs, adding dense (sub)rectangles first
    for(typename std::vector<ZIndexSpace<N,T> >::const_iterator it = inputs.begin();
	it != inputs.end();
	it++) {
      if(it->dense()) {
	bitmask.add_rect(it->bounds);
      } else {
	SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(it->sparsity);
	const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
	for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it2 = entries.begin();
	    it2 != entries.end();
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
    TimeStamp ts("UnionMicroOp::execute", true, &log_uop_timing);
#ifdef DEBUG_PARTITIONING
    std::cout << "calc union: " << inputs[0];
    for(size_t i = 1; i < inputs.size(); i++)
      std::cout << " + " << inputs[i];
    std::cout << std::endl;
#endif
    DenseRectangleList<N,T> drl;
    populate_bitmask(drl);
    if(sparsity_output.exists()) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_output);
      impl->contribute_dense_rect_list(drl);
    }
  }

  template <int N, typename T>
  void UnionMicroOp<N,T>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // need valid data for each input
    for(typename std::vector<ZIndexSpace<N,T> >::const_iterator it = inputs.begin();
	it != inputs.end();
	it++) {
      if(!it->dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N,T>::lookup(it->sparsity)->add_waiter(this, true /*precise*/);
	if(registered)
	  __sync_fetch_and_add(&wait_count, 1);
      }
    }

    finish_dispatch(op, inline_ok);
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
    // special case: in 1-D, we can count on the iterators being ordered and just do an O(N)
    //  merge-intersection of the two streams
    if(N == 1) {
      // even more special case where inputs.size() == 2
      if(inputs.size() == 2) {
	ZIndexSpaceIterator<N,T> it_lhs(inputs[0]);
	ZIndexSpaceIterator<N,T> it_rhs(inputs[1]);
       
	// can only generate data while both sides have rectangles left
	while(it_lhs.valid && it_rhs.valid) {
	  // skip rectangles if they completely preceed the one on the other side
	  if(it_lhs.rect.hi.x < it_rhs.rect.lo.x) {
	    it_lhs.step();
	    continue;
	  }

	  if(it_rhs.rect.hi.x < it_lhs.rect.lo.x) {
	    it_rhs.step();
	    continue;
	  }

	  // we have at least partial overlap - add the intersection and then consume whichever
	  //  rectangle ended first (or both if equal)
	  bitmask.add_rect(it_lhs.rect.intersection(it_rhs.rect));
	  T diff = it_lhs.rect.hi.x - it_rhs.rect.hi.x;
	  if(diff <= 0)
	    it_lhs.step();
	  if(diff >= 0)
	    it_rhs.step();
	}
      } else {
	assert(0);
      }
      return;
    }

    // general version
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
    TimeStamp ts("IntersectionMicroOp::execute", true, &log_uop_timing);
#ifdef DEBUG_PARTITIONING
    std::cout << "calc intersection: " << inputs[0];
    for(size_t i = 1; i < inputs.size(); i++)
      std::cout << " & " << inputs[i];
    std::cout << std::endl;
#endif
    DenseRectangleList<N,T> drl;
    populate_bitmask(drl);
    if(sparsity_output.exists()) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_output);
      impl->contribute_dense_rect_list(drl);
    }
  }

  template <int N, typename T>
  void IntersectionMicroOp<N,T>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // need valid data for each input
    for(typename std::vector<ZIndexSpace<N,T> >::const_iterator it = inputs.begin();
	it != inputs.end();
	it++) {
      if(!it->dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N,T>::lookup(it->sparsity)->add_waiter(this, true /*precise*/);
	if(registered)
	  __sync_fetch_and_add(&wait_count, 1);
      }
    }

    finish_dispatch(op, inline_ok);
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
    // special case: in 1-D, we can count on the iterators being ordered and just do an O(N)
    //  merge-subtract of the two streams
    if(N == 1) {
      ZIndexSpaceIterator<N,T> it_lhs(lhs);
      ZIndexSpaceIterator<N,T> it_rhs(rhs);

      while(it_lhs.valid) {
	// throw away any rhs rectangles that come before this one
	while(it_rhs.valid && (it_rhs.rect.hi.x < it_lhs.rect.lo.x))
	  it_rhs.step();

	// out of rhs rectangles? just copy over all the rest on the lhs and we're done
	if(!it_rhs.valid) {
	  while(it_lhs.valid) {
	    bitmask.add_rect(it_lhs.rect);
	    it_lhs.step();
	  }
	  break;
	}

	// consume lhs rectangles until we get one that overlaps
	while(it_lhs.rect.hi.x < it_rhs.rect.lo.x) {
	  bitmask.add_rect(it_lhs.rect);
	  if(!it_lhs.step()) break;
	}

	// last case - partial overlap - subtract out rhs rect(s)
	if(it_lhs.valid) {
	  ZPoint<N,T> p = it_lhs.rect.lo;
	  while(it_rhs.valid) {
	    if(p.x < it_rhs.rect.lo.x) {
	      // add a partial rect below the rhs
	      ZPoint<N,T> p2 = it_rhs.rect.lo;
	      p2.x -= 1;
	      bitmask.add_rect(ZRect<N,T>(p, p2));
	    }

	    // if the rhs ends after the lhs, we're done
	    if(it_rhs.rect.hi.x >= it_lhs.rect.hi.x)
	      break;

	    // otherwise consume the rhs and update p
	    p = it_rhs.rect.hi;
	    p.x += 1;
	    if(!it_rhs.step()) {
	      // no rhs left - emit the rest and break out
	      bitmask.add_rect(ZRect<N,T>(p, it_lhs.rect.hi));
	      break;
	    }
	  }
	  it_lhs.step();
	}
      }
      return;
    }

    // the basic idea here is to build a list of rectangles from the lhs and clip them
    //  based on the rhs until we're done
    std::deque<ZRect<N,T> > todo;

    if(lhs.dense()) {
      todo.push_back(lhs.bounds);
    } else {
      SparsityMapImpl<N,T> *l_impl = SparsityMapImpl<N,T>::lookup(lhs.sparsity);
      const std::vector<SparsityMapEntry<N,T> >& entries = l_impl->get_entries();
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	  it != entries.end();
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
#ifdef DEBUG_PARTITIONING
	std::cout << "check " << r << " -= " << it.rect << std::endl;
#endif
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
#ifdef DEBUG_PARTITIONING
	std::cout << "difference += " << r << std::endl;
#endif
	bitmask.add_rect(r);
      }
    }
  }

  template <int N, typename T>
  void DifferenceMicroOp<N,T>::execute(void)
  {
    TimeStamp ts("DifferenceMicroOp::execute", true, &log_uop_timing);
#ifdef DEBUG_PARTITIONING
    std::cout << "calc difference: " << lhs << " - " << rhs << std::endl;
#endif
    DenseRectangleList<N,T> drl;
    populate_bitmask(drl);
    if(sparsity_output.exists()) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_output);
      impl->contribute_dense_rect_list(drl);
    }
  }

  template <int N, typename T>
  void DifferenceMicroOp<N,T>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // need valid data for each source
    if(!lhs.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N,T>::lookup(lhs.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
	__sync_fetch_and_add(&wait_count, 1);
    }

    if(!rhs.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N,T>::lookup(rhs.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
	__sync_fetch_and_add(&wait_count, 1);
    }

    finish_dispatch(op, inline_ok);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMicroOpMessage
  
  template <int N, typename T>
  /*static*/ void RemoteMicroOpMessage::decode_request(RequestArgs args,
						       const void *data, size_t datalen)
  {
    Serialization::FixedBufferDeserializer fbd(data, datalen);

    switch(args.opcode) {
    case PartitioningMicroOp::UOPCODE_BY_FIELD:
      {
	ByFieldMicroOp<N,T,int> *uop = new ByFieldMicroOp<N,T,int>(args.sender,
								 args.async_microop,
								 fbd);
	uop->dispatch(args.operation, false /*not ok to run in this thread*/);
	break;
      }
    case PartitioningMicroOp::UOPCODE_IMAGE:
      {
	ImageMicroOp<N,T,N,T> *uop = new ImageMicroOp<N,T,N,T>(args.sender,
							       args.async_microop,
							       fbd);
	uop->dispatch(args.operation, false /*not ok to run in this thread*/);
	break;
      }
    case PartitioningMicroOp::UOPCODE_PREIMAGE:
      {
	PreimageMicroOp<N,T,N,T> *uop = new PreimageMicroOp<N,T,N,T>(args.sender,
								     args.async_microop,
								     fbd);
	uop->dispatch(args.operation, false /*not ok to run in this thread*/);
	break;
      }
    default:
      assert(0);
    }
  }

  /*static*/ void RemoteMicroOpMessage::handle_request(RequestArgs args,
						       const void *data, size_t datalen)
  {
    log_part.info() << "received remote micro op message: dim=" << args.dim
		    << " idxtype=" << args.idxtype << " opcode=" << args.opcode;

    if((args.dim == 1) && (args.idxtype == (int)sizeof(int))) {
      decode_request<1,int>(args, data, datalen);
      return;
    }
    assert(0);
  }
  
  template <typename T>
  /*static*/ void RemoteMicroOpMessage::send_request(gasnet_node_t target, 
						     PartitioningOperation *operation,
						     const T& microop)
  {
    RequestArgs args;

    args.sender = gasnet_mynode();
    args.dim = T::DIM;
    args.idxtype = sizeof(typename T::IDXTYPE);
    args.opcode = T::OPCODE;
    args.operation = operation;
    args.async_microop = microop.async_microop;

    Serialization::DynamicBufferSerializer dbs(256);
    microop.serialize_params(dbs);
    ByteArray b = dbs.detach_bytearray();

    Message::request(target, args, b.base(), b.size(), PAYLOAD_FREE);
    b.detach();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMicroOpCompleteMessage

  struct RequestArgs {
    AsyncMicroOp *async_microop;
  };
  
  /*static*/ void RemoteMicroOpCompleteMessage::handle_request(RequestArgs args)
  {
    log_part.info() << "received remote micro op complete message: " << args.async_microop;
    args.async_microop->mark_finished();
  }

  /*static*/ void RemoteMicroOpCompleteMessage::send_request(gasnet_node_t target,
							     AsyncMicroOp *async_microop)
  {
    RequestArgs args;
    args.async_microop = async_microop;
    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteSparsityContribMessage

  template <int N, typename T>
  /*static*/ void RemoteSparsityContribMessage::decode_request(RequestArgs args,
							       const void *data, size_t datalen)
  {
    SparsityMap<N,T> sparsity;
    sparsity.id = args.sparsity_id;

    log_part.info() << "received remote contribution: sparsity=" << sparsity << " len=" << datalen;
    size_t count = datalen / sizeof(ZRect<N,T>);
    assert((datalen % sizeof(ZRect<N,T>)) == 0);
    bool last_fragment = fragment_assembler.add_fragment(args.sender,
							 args.sequence_id,
							 args.sequence_count);
    SparsityMapImpl<N,T>::lookup(sparsity)->contribute_raw_rects((const ZRect<N,T> *)data,
								 count,
								 last_fragment);
  }

  /*static*/ void RemoteSparsityContribMessage::handle_request(RequestArgs args,
							       const void *data, size_t datalen)
  {
    if((args.dim == 1) && (args.idxtype == (int)sizeof(int))) {
      decode_request<1,int>(args, data, datalen);
      return;
    }
    assert(0);
  }

  template <int N, typename T>
  /*static*/ void RemoteSparsityContribMessage::send_request(gasnet_node_t target,
							     SparsityMap<N,T> sparsity,
							     int sequence_id,
							     int sequence_count,
							     const ZRect<N,T> *rects,
							     size_t count)
  {
    RequestArgs args;

    args.sender = gasnet_mynode();
    args.dim = N;
    args.idxtype = sizeof(T);
    args.sparsity_id = sparsity.id;
    args.sequence_id = sequence_id;
    args.sequence_count = sequence_count;

    Message::request(target, args, rects, count * sizeof(ZRect<N,T>),
		     PAYLOAD_COPY);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteSparsityRequestMessage

  template <int N, typename T>
  /*static*/ void RemoteSparsityRequestMessage::decode_request(RequestArgs args)
  {
    SparsityMap<N,T> sparsity;
    sparsity.id = args.sparsity_id;

    log_part.info() << "received sparsity request: sparsity=" << sparsity << " precise=" << args.send_precise << " approx=" << args.send_approx;
    SparsityMapImpl<N,T>::lookup(sparsity)->remote_data_request(args.sender, args.send_precise, args.send_approx);
  }

  /*static*/ void RemoteSparsityRequestMessage::handle_request(RequestArgs args)
  {
    if((args.dim == 1) && (args.idxtype == (int)sizeof(int))) {
      decode_request<1,int>(args);
      return;
    }
    assert(0);
  }

  template <int N, typename T>
  /*static*/ void RemoteSparsityRequestMessage::send_request(gasnet_node_t target,
							     SparsityMap<N,T> sparsity,
							     bool send_precise,
							     bool send_approx)
  {
    RequestArgs args;

    args.sender = gasnet_mynode();
    args.dim = N;
    args.idxtype = sizeof(T);
    args.sparsity_id = sparsity.id;
    args.send_precise = send_precise;
    args.send_approx = send_approx;

    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningOperation

  class DeferredPartitioningOp : public EventWaiter {
  public:
    DeferredPartitioningOp(PartitioningOperation *_op) : op(_op) {}

    virtual bool event_triggered(void) { op_queue->enqueue_partitioning_operation(op); return true; }

    virtual void print_info(FILE *f) {}

  protected:
    PartitioningOperation *op;
  };

  PartitioningOperation::PartitioningOperation(const ProfilingRequestSet &reqs,
					       Event _finish_event)
    : Operation(_finish_event, reqs)
  {}

  void PartitioningOperation::deferred_launch(Event wait_for)
  {
    if(1 || wait_for.has_triggered())
      op_queue->enqueue_partitioning_operation(this);
    else
      EventImpl::add_waiter(wait_for, new DeferredPartitioningOp(this));
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class ByFieldOperation<N,T,FT>

  template <int N, typename T, typename FT>
  ByFieldOperation<N,T,FT>::ByFieldOperation(const ZIndexSpace<N,T>& _parent,
					     const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& _field_data,
					     const ProfilingRequestSet &reqs,
					     Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
    , parent(_parent)
    , field_data(_field_data)
  {}

  template <int N, typename T, typename FT>
  ByFieldOperation<N,T,FT>::~ByFieldOperation(void)
  {}

  template <int N, typename T, typename FT>
  ZIndexSpace<N,T> ByFieldOperation<N,T,FT>::add_color(FT color)
  {
    // an empty parent leads to trivially empty subspaces
    if(parent.empty())
      return ZIndexSpace<N,T>(/*empty*/);

    // otherwise it'll be something smaller than the current parent
    ZIndexSpace<N,T> subspace;
    subspace.bounds = parent.bounds;
    SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    subspace.sparsity = sparsity;

    colors.push_back(color);
    subspaces.push_back(sparsity);

    return subspace;
  }

  template <int N, typename T, typename FT>
  void ByFieldOperation<N,T,FT>::execute(void)
  {
    for(size_t i = 0; i < subspaces.size(); i++)
      SparsityMapImpl<N,T>::lookup(subspaces[i])->update_contributor_count(field_data.size());

    for(size_t i = 0; i < field_data.size(); i++) {
      ByFieldMicroOp<N,T,FT> *uop = new ByFieldMicroOp<N,T,FT>(parent,
							       field_data[i].index_space,
							       field_data[i].inst,
							       field_data[i].field_offset);
      for(size_t j = 0; j < colors.size(); j++)
	uop->add_sparsity_output(colors[j], subspaces[j]);
      //uop.set_value_set(colors);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ImageOperation<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  ImageOperation<N,T,N2,T2>::ImageOperation(const ZIndexSpace<N,T>& _parent,
					    const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& _field_data,
					    const ProfilingRequestSet &reqs,
					    Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
    , parent(_parent)
    , field_data(_field_data)
  {}

  template <int N, typename T, int N2, typename T2>
  ImageOperation<N,T,N2,T2>::~ImageOperation(void)
  {}

  template <int N, typename T, int N2, typename T2>
  ZIndexSpace<N,T> ImageOperation<N,T,N2,T2>::add_source(const ZIndexSpace<N2,T2>& source)
  {
    // try to filter out obviously empty sources
    if(parent.empty() || source.empty())
      return ZIndexSpace<N,T>(/*empty*/);

    // otherwise it'll be something smaller than the current parent
    ZIndexSpace<N,T> image;
    image.bounds = parent.bounds;
    SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    image.sparsity = sparsity;

    sources.push_back(source);
    images.push_back(sparsity);

    return image;
  }

  template <int N, typename T, int N2, typename T2>
  void ImageOperation<N,T,N2,T2>::execute(void)
  {
    for(size_t i = 0; i < images.size(); i++)
      SparsityMapImpl<N,T>::lookup(images[i])->update_contributor_count(field_data.size());

    for(size_t i = 0; i < field_data.size(); i++) {
      ImageMicroOp<N,T,N2,T2> *uop = new ImageMicroOp<N,T,N2,T2>(parent,
								 field_data[i].index_space,
								 field_data[i].inst,
								 field_data[i].field_offset);
      for(size_t j = 0; j < sources.size(); j++)
	uop->add_sparsity_output(sources[j], images[j]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PreimageOperation<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  PreimageOperation<N,T,N2,T2>::PreimageOperation(const ZIndexSpace<N,T>& _parent,
						  const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > >& _field_data,
						  const ProfilingRequestSet &reqs,
						  Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
    , parent(_parent)
    , field_data(_field_data)
  {}

  template <int N, typename T, int N2, typename T2>
  PreimageOperation<N,T,N2,T2>::~PreimageOperation(void)
  {}

  template <int N, typename T, int N2, typename T2>
  ZIndexSpace<N,T> PreimageOperation<N,T,N2,T2>::add_target(const ZIndexSpace<N2,T2>& target)
  {
    // try to filter out obviously empty targets
    if(parent.empty() || target.empty())
      return ZIndexSpace<N,T>(/*empty*/);

    // otherwise it'll be something smaller than the current parent
    ZIndexSpace<N,T> preimage;
    preimage.bounds = parent.bounds;
    SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    preimage.sparsity = sparsity;

    targets.push_back(target);
    preimages.push_back(sparsity);

    return preimage;
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageOperation<N,T,N2,T2>::execute(void)
  {
    for(size_t i = 0; i < preimages.size(); i++)
      SparsityMapImpl<N,T>::lookup(preimages[i])->update_contributor_count(field_data.size());

    for(size_t i = 0; i < field_data.size(); i++) {
      PreimageMicroOp<N,T,N2,T2> *uop = new PreimageMicroOp<N,T,N2,T2>(parent,
								       field_data[i].index_space,
								       field_data[i].inst,
								       field_data[i].field_offset);
      for(size_t j = 0; j < targets.size(); j++)
	uop->add_sparsity_output(targets[j], preimages[j]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UnionOperation<N,T>

  template <int N, typename T>
  UnionOperation<N,T>::UnionOperation(const ProfilingRequestSet& reqs,
				      Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
  {}

  template <int N, typename T>
  UnionOperation<N,T>::~UnionOperation(void)
  {}

  template <int N, typename T>
  ZIndexSpace<N,T> UnionOperation<N,T>::add_union(const ZIndexSpace<N,T>& lhs,
						  const ZIndexSpace<N,T>& rhs)
  {
    // simple case - if both lhs and rhs are empty, the union must be empty too
    if(lhs.empty() && rhs.empty())
      return ZIndexSpace<N,T>(/*empty*/);

    // otherwise create a new index space whose bounds can fit both lhs and rhs
    ZIndexSpace<N,T> output;
    output.bounds = lhs.bounds.union_bbox(rhs.bounds);
    SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    output.sparsity = sparsity;

    std::vector<ZIndexSpace<N,T> > ops(2);
    ops[0] = lhs;
    ops[1] = rhs;
    inputs.push_back(ops);
    outputs.push_back(sparsity);

    return output;
  }

  template <int N, typename T>
  ZIndexSpace<N,T> UnionOperation<N,T>::add_union(const std::vector<ZIndexSpace<N,T> >& ops)
  {
    // build a bounding box that can hold all the operands, and pay attention to the
    //  case where they're all empty
    ZIndexSpace<N,T> output;
    bool all_empty = true;
    for(size_t i = 0; i < ops.size(); i++)
      if(!ops[i].empty()) {
	all_empty = false;
	output.bounds = output.bounds.union_bbox(ops[i].bounds);
      }

    if(!all_empty) {
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      output.sparsity = sparsity;

      inputs.push_back(ops);
      outputs.push_back(sparsity);
    }

    return output;
  }

  template <int N, typename T>
  void UnionOperation<N,T>::execute(void)
  {
    for(size_t i = 0; i < outputs.size(); i++) {
      SparsityMapImpl<N,T>::lookup(outputs[i])->update_contributor_count(1);

      UnionMicroOp<N,T> *uop = new UnionMicroOp<N,T>(inputs[i]);
      uop->add_sparsity_output(outputs[i]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntersectionOperation<N,T>

  template <int N, typename T>
  IntersectionOperation<N,T>::IntersectionOperation(const ProfilingRequestSet& reqs,
						    Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
  {}

  template <int N, typename T>
  IntersectionOperation<N,T>::~IntersectionOperation(void)
  {}

  template <int N, typename T>
  ZIndexSpace<N,T> IntersectionOperation<N,T>::add_intersection(const ZIndexSpace<N,T>& lhs,
								const ZIndexSpace<N,T>& rhs)
  {
    ZIndexSpace<N,T> output;
    output.bounds = lhs.bounds.intersection(rhs.bounds);
    
    if(!output.bounds.empty()) {
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      output.sparsity = sparsity;

      std::vector<ZIndexSpace<N,T> > ops(2);
      ops[0] = lhs;
      ops[1] = rhs;
      inputs.push_back(ops);
      outputs.push_back(sparsity);
    }

    return output;
  }

  template <int N, typename T>
  ZIndexSpace<N,T> IntersectionOperation<N,T>::add_intersection(const std::vector<ZIndexSpace<N,T> >& ops)
  {
    // special case for empty operand list
    if(ops.empty())
      return ZIndexSpace<N,T>(/*empty*/);

    // build the intersection of all bounding boxes
    ZIndexSpace<N,T> output;
    output.bounds = ops[0].bounds;
    for(size_t i = 1; i < ops.size(); i++)
      output.bounds = output.bounds.intersection(ops[i].bounds);

    if(!output.bounds.empty()) {
      SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
      SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
      output.sparsity = sparsity;

      inputs.push_back(ops);
      outputs.push_back(sparsity);
    }

    return output;
  }

  template <int N, typename T>
  void IntersectionOperation<N,T>::execute(void)
  {
    for(size_t i = 0; i < outputs.size(); i++) {
      SparsityMapImpl<N,T>::lookup(outputs[i])->update_contributor_count(1);

      IntersectionMicroOp<N,T> *uop = new IntersectionMicroOp<N,T>(inputs[i]);
      uop->add_sparsity_output(outputs[i]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DifferenceOperation<N,T>

  template <int N, typename T>
  DifferenceOperation<N,T>::DifferenceOperation(const ProfilingRequestSet& reqs,
				      Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
  {}

  template <int N, typename T>
  DifferenceOperation<N,T>::~DifferenceOperation(void)
  {}

  template <int N, typename T>
  ZIndexSpace<N,T> DifferenceOperation<N,T>::add_difference(const ZIndexSpace<N,T>& lhs,
							    const ZIndexSpace<N,T>& rhs)
  {
    // simple cases - an empty lhs or a dense rhs that covers lhs both yield an empty
    //  difference
    if(lhs.empty() || (rhs.dense() && rhs.bounds.contains(lhs.bounds)))
      return ZIndexSpace<N,T>(/*empty*/);

    // otherwise the difference is no larger than the lhs
    ZIndexSpace<N,T> output;
    output.bounds = lhs.bounds;
    SparsityMapImplWrapper *wrap = get_runtime()->local_sparsity_map_free_list->alloc_entry();
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    output.sparsity = sparsity;

    lhss.push_back(lhs);
    rhss.push_back(rhs);
    outputs.push_back(sparsity);

    return output;
  }

  template <int N, typename T>
  void DifferenceOperation<N,T>::execute(void)
  {
    for(size_t i = 0; i < outputs.size(); i++) {
      SparsityMapImpl<N,T>::lookup(outputs[i])->update_contributor_count(1);

      DifferenceMicroOp<N,T> *uop = new DifferenceMicroOp<N,T>(lhss[i], rhss[i]);
      uop->add_sparsity_output(outputs[i]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningOpQueue

  static int num_partitioning_workers = 1;
  PartitioningOpQueue::PartitioningOpQueue( CoreReservation *_rsrv)
    : shutdown_flag(false), rsrv(_rsrv), condvar(mutex)
  {}
  
  PartitioningOpQueue::~PartitioningOpQueue(void)
  {
    assert(shutdown_flag);
    delete rsrv;
  }

  /*static*/ void PartitioningOpQueue::configure_from_cmdline(std::vector<std::string>& cmdline)
  {
    CommandLineParser cp;

    cp.add_option_int("-dp:workers", num_partitioning_workers);

    cp.parse_command_line(cmdline);
  }

  /*static*/ void PartitioningOpQueue::start_worker_threads(CoreReservationSet& crs)
  {
    assert(op_queue == 0);
    CoreReservation *rsrv = new CoreReservation("partitioning", crs,
						CoreReservationParameters());
    op_queue = new PartitioningOpQueue(rsrv);
    ThreadLaunchParameters tlp;
    for(int i = 0; i < num_partitioning_workers; i++) {
      Thread *t = Thread::create_kernel_thread<PartitioningOpQueue,
					       &PartitioningOpQueue::worker_thread_loop>(op_queue,
											 tlp,
											 *rsrv);
      op_queue->workers.push_back(t);
    }
  }

  /*static*/ void PartitioningOpQueue::stop_worker_threads(void)
  {
    assert(op_queue != 0);

    op_queue->shutdown_flag = true;
    {
      AutoHSLLock al(op_queue->mutex);
      op_queue->condvar.broadcast();
    }
    for(size_t i = 0; i < op_queue->workers.size(); i++) {
      op_queue->workers[i]->join();
      delete op_queue->workers[i];
    }
    op_queue->workers.clear();

    delete op_queue;
    op_queue = 0;
  }
      
  void PartitioningOpQueue::enqueue_partitioning_operation(PartitioningOperation *op)
  {
    op->mark_ready();

    AutoHSLLock al(mutex);

    queued_ops.put(op, OPERATION_PRIORITY);

    op_queue->condvar.broadcast();
  }

  void PartitioningOpQueue::enqueue_partitioning_microop(PartitioningMicroOp *uop)
  {
    AutoHSLLock al(mutex);

    queued_ops.put(uop, MICROOP_PRIORITY);

    op_queue->condvar.broadcast();
  }

  void PartitioningOpQueue::worker_thread_loop(void)
  {
    log_part.info() << "worker " << Thread::self() << " started for op queue " << this;

    while(!shutdown_flag) {
      void *op = 0;
      int priority;
      while(!op && !shutdown_flag) {
	AutoHSLLock al(mutex);
	op = queued_ops.get(&priority);
	if(!op && !shutdown_flag)
	  condvar.wait();
      }
      if(op) {
	switch(priority) {
	case OPERATION_PRIORITY:
	  {
	    PartitioningOperation *p_op = static_cast<PartitioningOperation *>(op);
	    log_part.info() << "worker " << this << " starting op " << p_op;
	    p_op->mark_started();
	    p_op->execute();
	    log_part.info() << "worker " << this << " finished op " << p_op;
	    p_op->mark_finished();
	    break;
	  }
	case MICROOP_PRIORITY:
	  {
	    PartitioningMicroOp *p_uop = static_cast<PartitioningMicroOp *>(op);
	    log_part.info() << "worker " << this << " starting uop " << p_uop;
	    p_uop->mark_started();
	    p_uop->execute();
	    log_part.info() << "worker " << this << " starting uop " << p_uop;
	    p_uop->mark_finished();
	    break;
	  }
	default: assert(0);
	}
      }
    }

    log_part.info() << "worker " << Thread::self() << " finishing for op queue " << this;
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

