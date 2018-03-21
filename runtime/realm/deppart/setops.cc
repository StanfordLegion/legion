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

// set operations for Realm dependent partitioning

#include "realm/deppart/setops.h"

#include "realm/deppart/deppart_config.h"
#include "realm/deppart/rectlist.h"
#include "realm/deppart/inst_helper.h"
#include "realm/deppart/image.h"
#include "realm/logging.h"

namespace Realm {

  extern Logger log_part;
  extern Logger log_uop_timing;

  // an intersection of two rectangles is always a rectangle, but the same is not
  //  true of unions and differences

  template <int N, typename T>
  static bool union_is_rect(const Rect<N,T>& lhs, const Rect<N,T>& rhs)
  {
    if(N == 1) {
      // 1-D case is simple - no gap allowed
      if((lhs.hi[0] < rhs.lo[0]) && ((lhs.hi[0] + 1) != rhs.lo[0]))
	return false;
      if((rhs.hi[0] < lhs.lo[0]) && ((rhs.hi[0] + 1) != lhs.lo[0]))
	return false;
      return true;
    }
    
    // containment case is easy
    if(lhs.contains(rhs) || rhs.contains(lhs))
      return true;

    // interval in each dimension must match, except one, which must have no gap
    int i = 0;
    while((i < N) && (lhs.lo[i] == rhs.lo[i]) && (lhs.hi[i] == rhs.hi[i]))
      i++;
    assert(i < N); // containment test above should eliminate i==N case

    // check overlap
    if((lhs.hi[i] < rhs.lo[i]) && ((lhs.hi[i] + 1) != rhs.lo[i]))
      return false;
    if((rhs.hi[i] < lhs.lo[i]) && ((rhs.hi[i] + 1) != lhs.lo[i]))
      return false;

    // remaining dimensions must match
    while(++i < N)
      if((lhs.lo[i] != rhs.lo[i]) || (lhs.hi[i] != rhs.hi[i]))
	return false;

    return true;
  }

  template <int N, typename T>
  static bool attempt_simple_diff(const Rect<N,T>& lhs, const Rect<N,T>& rhs,
				  Rect<N,T>& out)
  {
    // rhs containing lhs always works
    if(rhs.contains(lhs)) {
      out = Rect<N,T>::make_empty();
      return true;
    }

    // disjoint rectangles always work too
    if(!rhs.overlaps(lhs)) {
      out = lhs;
      return true;
    }		 

    if(N == 1) {
      if(lhs.lo[0] < rhs.lo[0]) {
	out.lo[0] = lhs.lo[0];
	// lhs hi must not extend past rhs
	if(lhs.hi[0] <= rhs.hi[0]) {
	  out.hi[0] = rhs.lo[0] - 1;
	  return true;
	} else
	  return false;
      }
      if(lhs.hi[0] > rhs.hi[0]) {
	out.hi[0] = lhs.hi[0];
	// lhs lo must not extend past rhs
	if(lhs.lo[0] >= rhs.lo[0]) {
	  out.lo[0] = rhs.hi[0] + 1;
	  return true;
	} else
	  return false;
      }
      // shouldn't get here?
      assert(0);
    }

    // we need N-1 dims to match (or be subsets), and exactly one to stick out
    //  (and only on one side)
    int i = 0;
    out = lhs;
    while((i < N) && (lhs.lo[i] >= rhs.lo[i]) && (lhs.hi[i] <= rhs.hi[i]))
      i++;
    assert(i < N); // containment test above should eliminate i==N case

    // compute difference in dim i
    {
      if(lhs.lo[i] < rhs.lo[i]) {
	out.lo[i] = lhs.lo[i];
	// lhs hi must not extend past rhs
	if(lhs.hi[i] <= rhs.hi[i]) {
	  out.hi[i] = rhs.lo[i] - 1;
	} else
	  return false;
      } else
      if(lhs.hi[i] > rhs.hi[i]) {
	out.hi[i] = lhs.hi[i];
	// lhs lo must not extend past rhs
	if(lhs.lo[i] >= rhs.lo[i]) {
	  out.lo[i] = rhs.hi[i] + 1;
	} else
	  return false;
      } else
      {
	// shouldn't get here?
	assert(0);
      }
    }

    // remaining dimensions must match (or be subsets)
    while(++i < N)
      if((lhs.lo[i] < rhs.lo[i]) || (lhs.hi[i] > rhs.hi[i]))
	return false;

    return true;
  }


  template <int N, typename T>
  __attribute__ ((noinline))
  /*static*/ Event IndexSpace<N,T>::compute_unions(const std::vector<IndexSpace<N,T> >& lhss,
						    const std::vector<IndexSpace<N,T> >& rhss,
						    std::vector<IndexSpace<N,T> >& results,
						    const ProfilingRequestSet &reqs,
						    Event wait_on /*= Event::NO_EVENT*/)
  {
    // output vector should start out empty
    assert(results.empty());

    Event e = wait_on;
    UnionOperation<N,T> *op = 0;

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;

      // handle a bunch of special cases
      const IndexSpace<N,T> &l = lhss[li];
      const IndexSpace<N,T> &r = rhss[ri];

      // 1) empty lhs
      if(l.empty()) {
	results[i] = r;
	continue;
      }

      // 2) empty rhs
      if(rhss[li].empty()) {
	results[i] = l;
	continue;
      }

      // 3) dense lhs containing rhs' bounds -> lhs
      if(l.dense() && l.bounds.contains(r.bounds)) {
	results[i] = l;
	continue;
      }

      // 4) dense rhs containing lhs' bounds -> rhs
      if(r.dense() && r.bounds.contains(l.bounds)) {
	results[i] = r;
	continue;
      }

      // 5) same sparsity map (or none) and simple union for bbox
      if((l.sparsity == r.sparsity) && union_is_rect(l.bounds, r.bounds)) {
	results[i] = IndexSpace<N,T>(l.bounds.union_bbox(r.bounds),
				      l.sparsity);
	continue;
      }

      // general case - create op if needed
      if(!op) {
	e = GenEventImpl::create_genevent()->current_event();
	op = new UnionOperation<N,T>(reqs, e);
      }
      results[i] = op->add_union(l, r);
    }

    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      log_dpops.info() << "union: " << lhss[li] << " " << rhss[ri] << " -> " << results[i] << " (" << e << ")";
    }

    if(op)
      op->deferred_launch(wait_on);
    else
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
    return e;
  }

  template <int N, typename T>
  __attribute__ ((noinline))
  /*static*/ Event IndexSpace<N,T>::compute_intersections(const std::vector<IndexSpace<N,T> >& lhss,
							   const std::vector<IndexSpace<N,T> >& rhss,
							   std::vector<IndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    // output vector should start out empty
    assert(results.empty());

    Event e = wait_on;
    IntersectionOperation<N,T> *op = 0;

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;

      // handle a bunch of special cases
      const IndexSpace<N,T> &l = lhss[li];
      const IndexSpace<N,T> &r = rhss[ri];

      // 1) either side empty or disjoint inputs
      if(l.empty() || r.empty() || !l.bounds.overlaps(r.bounds)) {
	results[i] = IndexSpace<N,T>::make_empty();
	continue;
      }

      // 2) rhs is dense or has same sparsity map
      if(r.dense() || (r.sparsity == l.sparsity)) {
	results[i] = IndexSpace<N,T>(l.bounds.intersection(r.bounds),
				      l.sparsity);
	continue;
      }

      // 3) lhs is dense
      if(l.dense()) {
	results[i] = IndexSpace<N,T>(l.bounds.intersection(r.bounds),
				      r.sparsity);
	continue;
      }

      // general case - create op if needed
      if(!op) {
	e = GenEventImpl::create_genevent()->current_event();
	op = new IntersectionOperation<N,T>(reqs, e);
      }
      results[i] = op->add_intersection(lhss[li], rhss[ri]);
    }

    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      log_dpops.info() << "isect: " << lhss[li] << " " << rhss[ri] << " -> " << results[i] << " (" << e << ")";
    }

    if(op)
      op->deferred_launch(wait_on);
    else
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
    return e;
  }

  template <int N, typename T>
  __attribute__ ((noinline))
  /*static*/ Event IndexSpace<N,T>::compute_differences(const std::vector<IndexSpace<N,T> >& lhss,
							 const std::vector<IndexSpace<N,T> >& rhss,
							 std::vector<IndexSpace<N,T> >& results,
							 const ProfilingRequestSet &reqs,
							 Event wait_on /*= Event::NO_EVENT*/)
  {
    // output vector should start out empty
    assert(results.empty());

    Event e = wait_on;
    DifferenceOperation<N,T> *op = 0;

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

    size_t n = std::max(lhss.size(), rhss.size());
    assert((lhss.size() == rhss.size()) || (lhss.size() == 1) || (rhss.size() == 1));
    results.resize(n);
    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;

      // handle a bunch of special cases
      const IndexSpace<N,T> &l = lhss[li];
      const IndexSpace<N,T> &r = rhss[ri];

      // 1) empty lhs
      if(l.empty()) {
	results[i] = IndexSpace<N,T>::make_empty();
	continue;
      }

      // 2) empty rhs
      if(r.empty()) {
	results[i] = l;
	continue;
      }

      // 3) no overlap between lhs and rhs
      if(!l.bounds.overlaps(r.bounds)) {
	results[i] = l;
	continue;
      }

      // 4) dense rhs containing lhs' bounds -> empty
      if(r.dense() && r.bounds.contains(l.bounds)) {
	results[i] = IndexSpace<N,T>::make_empty();
	continue;
      }

      // 5) same sparsity map (or none) and simple difference
      if(r.dense() || (l.sparsity == r.sparsity)) {
	Rect<N,T> sdiff;
	if(attempt_simple_diff(l.bounds, r.bounds, sdiff)) {
	  results[i] = IndexSpace<N,T>(sdiff, l.sparsity);
	  continue;
	}
      }

      // general case - create op if needed
      if(!op) {
	e = GenEventImpl::create_genevent()->current_event();
	op = new DifferenceOperation<N,T>(reqs, e);
      }
      results[i] = op->add_difference(lhss[li], rhss[ri]);
    }

    for(size_t i = 0; i < n; i++) {
      size_t li = (lhss.size() == 1) ? 0 : i;
      size_t ri = (rhss.size() == 1) ? 0 : i;
      log_dpops.info() << "diff: " << lhss[li] << " " << rhss[ri] << " -> " << results[i] << " (" << e << ")";
    }

    if(op)
      op->deferred_launch(wait_on);
    else
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
    return e;
  }

  template <int N, typename T>
  /*static*/ Event IndexSpace<N,T>::compute_union(const std::vector<IndexSpace<N,T> >& subspaces,
						   IndexSpace<N,T>& result,
						   const ProfilingRequestSet &reqs,
						   Event wait_on /*= Event::NO_EVENT*/)
  {
    Event e = wait_on;

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();
    bool was_inline = true;

    // various special cases
    if(subspaces.empty()) {
      result = IndexSpace<N,T>::make_empty();
    } else {
      result = subspaces[0];

      for(size_t i = 1; i < subspaces.size(); i++) {
	// empty rhs - skip
	if(subspaces[i].empty())
	  continue;

	// lhs dense or subspace match, and containment - skip
	if((result.dense() || (result.sparsity == subspaces[i].sparsity)) &&
	   result.bounds.contains(subspaces[i].bounds))
	  continue;

	// TODO: subspace match ought to be sufficient here - also handle
	//  merge-into-rectangle case?
	// rhs dense and contains lhs - take rhs
	if(subspaces[i].dense() && subspaces[i].bounds.contains(result.bounds)) {
	  result = subspaces[i];
	  continue;
	}

	// general case - do full computation
	e = GenEventImpl::create_genevent()->current_event();
	UnionOperation<N,T> *op = new UnionOperation<N,T>(reqs, e);

	result = op->add_union(subspaces);
	op->deferred_launch(wait_on);
	was_inline = false;
	break;
      }
    }

    {
      LoggerMessage msg = log_dpops.info();
      if(msg.is_active()) {
	msg << "union:";
	for(typename std::vector<IndexSpace<N,T> >::const_iterator it = subspaces.begin();
	    it != subspaces.end();
	    ++it)
	  msg << " " << *it;
	msg << " -> " << result << " (" << e << ")";
      }
    }

    if(was_inline)
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);

    return e;
  }

  template <int N, typename T>
  /*static*/ Event IndexSpace<N,T>::compute_intersection(const std::vector<IndexSpace<N,T> >& subspaces,
							  IndexSpace<N,T>& result,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    Event e = wait_on;

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();
    bool was_inline = true;

    // various special cases
    if(subspaces.empty()) {
      result = IndexSpace<N,T>::make_empty();
    } else {
      result = subspaces[0];

      for(size_t i = 1; i < subspaces.size(); i++) {
	// no point in continuing if our result is empty
	if(result.empty()) {
	  result.sparsity.id = 0;  // forget any sparsity map we had
	  break;
	}

	// empty rhs - result is empty
	if(subspaces[i].empty()) {
	  result = IndexSpace<N,T>::make_empty();
	  break;
	}

	// disjointness of lhs and rhs bounds - result is empty
	if(!result.bounds.overlaps(subspaces[i].bounds)) {
	  result = IndexSpace<N,T>::make_empty();
	  break;
	}

	// rhs dense or has same sparsity map
	if(subspaces[i].dense() || (subspaces[i].sparsity == result.sparsity)) {
	  result.bounds = result.bounds.intersection(subspaces[i].bounds);
	  continue;
	}

	// lhs dense and rhs sparse - intersect and adopt rhs' sparsity map
	if(result.dense()) {
	  result.bounds = result.bounds.intersection(subspaces[i].bounds);
	  result.sparsity = subspaces[i].sparsity;
	  continue;	  
	}

	// general case - do full computation
	e = GenEventImpl::create_genevent()->current_event();
	IntersectionOperation<N,T> *op = new IntersectionOperation<N,T>(reqs, e);

	result = op->add_intersection(subspaces);
	op->deferred_launch(wait_on);
	was_inline = false;
	break;
      }
    }

    {
      LoggerMessage msg = log_dpops.info();
      if(msg.is_active()) {
	msg << "isect:";
	for(typename std::vector<IndexSpace<N,T> >::const_iterator it = subspaces.begin();
	    it != subspaces.end();
	    ++it)
	  msg << " " << *it;
	msg << " -> " << result << " (" << e << ")";
      }
    }

    if(was_inline)
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);

    return e;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UnionMicroOp<N,T>

  template <int N, typename T>
  inline /*static*/ DynamicTemplates::TagType UnionMicroOp<N,T>::type_tag(void)
  {
    return NT_TemplateHelper::encode_tag<N,T>();
  }

  template <int N, typename T>
  UnionMicroOp<N,T>::UnionMicroOp(const std::vector<IndexSpace<N,T> >& _inputs)
    : inputs(_inputs)
  {
    sparsity_output.id = 0;
  }

  template <int N, typename T>
  UnionMicroOp<N,T>::UnionMicroOp(IndexSpace<N,T> _lhs,
				  IndexSpace<N,T> _rhs)
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
    NWayMerge(const std::vector<IndexSpace<N,T> >& spaces);
    ~NWayMerge(void);

    const Rect<N,T>& operator[](int idx) const;
    size_t size(void) const;

    // steps an iterator - does not immediately update its position
    bool step(int idx);

    // called after you call step at least once on a given iterator
    void update(int idx);

    void print(void) const;

  protected:
    int n;
    std::vector<IndexSpaceIterator<N,T> > its;
    std::vector<int> order;
  };

  template <int N, typename T>
  NWayMerge<N,T>::NWayMerge(const std::vector<IndexSpace<N,T> >& spaces)
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
  const Rect<N,T>& NWayMerge<N,T>::operator[](int idx) const
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
  class Fast1DUnion {
  public:
    template <typename BM>
    static bool attempt_union(BM& bitmask, const std::vector<IndexSpace<N,T> >& spaces)
    {
      assert(N != 1); // N==1 covered by case below
      return false;  // general case doesn't work
    }
  };

  template <typename T>
  class Fast1DUnion<1,T> {
  public:
    static const int N = 1;
    template <typename BM>
    static bool attempt_union(BM& bitmask, const std::vector<IndexSpace<N,T> >& inputs)
    {
      // stuff
      // even more special case where inputs.size() == 2
      if(inputs.size() == 2) {
	IndexSpaceIterator<N,T> it_lhs(inputs[0]);
	IndexSpaceIterator<N,T> it_rhs(inputs[1]);
       
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
	  Rect<N,T> u = it_lhs.rect.union_bbox(it_rhs.rect);
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
	  Rect<N,T> u = nwm[0];
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
	std::vector<IndexSpaceIterator<N,T> > its(inputs.size());
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
	  Rect<N,T> 
	  break;
	}

	// whichever one is left can just emit all its remaining rectangles
	while(its[order[0]].valid) {
	  bitmask.add_rect(its[order[0]].rect);
	  its[order[0]].step();
	}
#endif
      }
      return true;
    }
  };

  template <int N, typename T>
  template <typename BM>
  void UnionMicroOp<N,T>::populate_bitmask(BM& bitmask)
  {
    // special case: in 1-D, we can count on the iterators being ordered and just do an O(N)
    //  merge-union of the two streams
    //if(try_fast_1d_union<N,T>(bitmask, inputs))
    if(Fast1DUnion<N,T>::attempt_union(bitmask, inputs))
      return;
#if 0
    if(N == 1) {
      // even more special case where inputs.size() == 2
      if(inputs.size() == 2) {
	IndexSpaceIterator<N,T> it_lhs(inputs[0]);
	IndexSpaceIterator<N,T> it_rhs(inputs[1]);
       
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
	  Rect<N,T> u = it_lhs.rect.union_bbox(it_rhs.rect);
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
	  Rect<N,T> u = nwm[0];
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
	std::vector<IndexSpaceIterator<N,T> > its(inputs.size());
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
	  Rect<N,T> 
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
#endif

    // iterate over all the inputs, adding dense (sub)rectangles first
    for(typename std::vector<IndexSpace<N,T> >::const_iterator it = inputs.begin();
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
	  Rect<N,T> isect = it->bounds.intersection(it2->bounds);
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
      impl->contribute_dense_rect_list(drl.rects);
    }
  }

  template <int N, typename T>
  void UnionMicroOp<N,T>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // execute wherever our sparsity output is
    NodeID exec_node = ID(sparsity_output).sparsity.creator_node;

    if(exec_node != my_node_id) {
      // we're going to ship it elsewhere, which means we always need an AsyncMicroOp to
      //  track it
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);

      RemoteMicroOpMessage::send_request(exec_node, op, *this);
      delete this;
      return;
    }

    // need valid data for each input
    for(typename std::vector<IndexSpace<N,T> >::const_iterator it = inputs.begin();
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

  template <int N, typename T>
  template <typename S>
  bool UnionMicroOp<N,T>::serialize_params(S& s) const
  {
    return((s << inputs) &&
	   (s << sparsity_output));
  }

  template <int N, typename T>
  template <typename S>
  UnionMicroOp<N,T>::UnionMicroOp(NodeID _requestor,
				  AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> inputs) &&
	       (s >> sparsity_output));
    assert(ok);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntersectionMicroOp<N,T>

  template <int N, typename T>
  inline /*static*/ DynamicTemplates::TagType IntersectionMicroOp<N,T>::type_tag(void)
  {
    return NT_TemplateHelper::encode_tag<N,T>();
  }

  template <int N, typename T>
  IntersectionMicroOp<N,T>::IntersectionMicroOp(const std::vector<IndexSpace<N,T> >& _inputs)
    : inputs(_inputs)
  {
    sparsity_output.id = 0;
  }

  template <int N, typename T>
  IntersectionMicroOp<N,T>::IntersectionMicroOp(IndexSpace<N,T> _lhs,
				  IndexSpace<N,T> _rhs)
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
	IndexSpaceIterator<N,T> it_lhs(inputs[0]);
	IndexSpaceIterator<N,T> it_rhs(inputs[1]);
       
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
    Rect<N,T> bounds = inputs[0].bounds;
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
      for(IndexSpaceIterator<N,T> it(inputs[0], bounds); it.valid; it.step())
	for(IndexSpaceIterator<N,T> it2(inputs[1], it.rect); it2.valid; it2.step())
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
      impl->contribute_dense_rect_list(drl.rects);
    }
  }

  template <int N, typename T>
  void IntersectionMicroOp<N,T>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // execute wherever our sparsity output is
    NodeID exec_node = ID(sparsity_output).sparsity.creator_node;

    if(exec_node != my_node_id) {
      // we're going to ship it elsewhere, which means we always need an AsyncMicroOp to
      //  track it
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);

      RemoteMicroOpMessage::send_request(exec_node, op, *this);
      delete this;
      return;
    }

    // need valid data for each input
    for(typename std::vector<IndexSpace<N,T> >::const_iterator it = inputs.begin();
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

  template <int N, typename T>
  template <typename S>
  bool IntersectionMicroOp<N,T>::serialize_params(S& s) const
  {
    return((s << inputs) &&
	   (s << sparsity_output));
  }

  template <int N, typename T>
  template <typename S>
  IntersectionMicroOp<N,T>::IntersectionMicroOp(NodeID _requestor,
						AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> inputs) &&
	       (s >> sparsity_output));
    assert(ok);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DifferenceMicroOp<N,T>

  template <int N, typename T>
  inline /*static*/ DynamicTemplates::TagType DifferenceMicroOp<N,T>::type_tag(void)
  {
    return NT_TemplateHelper::encode_tag<N,T>();
  }

  template <int N, typename T>
  DifferenceMicroOp<N,T>::DifferenceMicroOp(IndexSpace<N,T> _lhs,
					    IndexSpace<N,T> _rhs)
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
  void subtract_rects(const Rect<N,T>& lhs, const Rect<N,T>& rhs,
		      std::vector<Rect<N,T> >& pieces)
  {
    // should only be called if we have overlapping rectangles
    assert(!lhs.empty() && !rhs.empty() && lhs.overlaps(rhs));
    Rect<N,T> r = lhs;
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
      IndexSpaceIterator<N,T> it_lhs(lhs);
      IndexSpaceIterator<N,T> it_rhs(rhs);

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

	// an lhs rectangle that is entirely below the first rhs is taken as is
	if(it_lhs.rect.hi.x < it_rhs.rect.lo.x) {
	  bitmask.add_rect(it_lhs.rect);
	  it_lhs.step();
	  continue;
	}

	// last case - partial overlap - subtract out rhs rect(s)
	if(it_lhs.valid) {
	  Point<N,T> p = it_lhs.rect.lo;
	  while(it_rhs.valid) {
	    if(p.x < it_rhs.rect.lo.x) {
	      // add a partial rect below the rhs
	      Point<N,T> p2 = it_rhs.rect.lo;
	      p2.x -= 1;
	      bitmask.add_rect(Rect<N,T>(p, p2));
	    }

	    // if the rhs ends after the lhs, we're done
	    if(it_rhs.rect.hi.x >= it_lhs.rect.hi.x)
	      break;

	    // otherwise consume the rhs and update p
	    p = it_rhs.rect.hi;
	    p.x += 1;
	    if(!it_rhs.step() || (it_lhs.rect.hi.x < it_rhs.rect.lo.x)) {
	      // no rhs left in this lhs piece - emit the rest and break out
	      bitmask.add_rect(Rect<N,T>(p, it_lhs.rect.hi));
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
    std::deque<Rect<N,T> > todo;

    if(lhs.dense()) {
      todo.push_back(lhs.bounds);
    } else {
      SparsityMapImpl<N,T> *l_impl = SparsityMapImpl<N,T>::lookup(lhs.sparsity);
      const std::vector<SparsityMapEntry<N,T> >& entries = l_impl->get_entries();
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	  it != entries.end();
	  it++) {
	Rect<N,T> isect = lhs.bounds.intersection(it->bounds);
	if(isect.empty())
	  continue;
	assert(!it->sparsity.exists());
	assert(it->bitmap == 0);
	todo.push_back(isect);
      }
    }

    while(!todo.empty()) {
      Rect<N,T> r = todo.front();
      todo.pop_front();

      // iterate over all subrects in the rhs - any that contain it eliminate this rect,
      //  overlap chops it into pieces
      bool fully_covered = false;
      for(IndexSpaceIterator<N,T> it(rhs); it.valid; it.step()) {
#ifdef DEBUG_PARTITIONING
	std::cout << "check " << r << " -= " << it.rect << std::endl;
#endif
	if(it.rect.contains(r)) {
	  fully_covered = true;
	  break;
	}

	if(it.rect.overlaps(r)) {
	  // subtraction is nasty - can result in 2N subrectangles
	  std::vector<Rect<N,T> > pieces;
	  subtract_rects(r, it.rect, pieces);
	  assert(!pieces.empty());

	  // continue on with the first piece, and stick the rest on the todo list
	  typename std::vector<Rect<N,T> >::iterator it2 = pieces.begin();
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
      impl->contribute_dense_rect_list(drl.rects);
    }
  }

  template <int N, typename T>
  void DifferenceMicroOp<N,T>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // execute wherever our sparsity output is
    NodeID exec_node = ID(sparsity_output).sparsity.creator_node;

    if(exec_node != my_node_id) {
      // we're going to ship it elsewhere, which means we always need an AsyncMicroOp to
      //  track it
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);

      RemoteMicroOpMessage::send_request(exec_node, op, *this);
      delete this;
      return;
    }

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

  template <int N, typename T>
  template <typename S>
  bool DifferenceMicroOp<N,T>::serialize_params(S& s) const
  {
    return((s << lhs) &&
	   (s << rhs) &&
	   (s << sparsity_output));
  }

  template <int N, typename T>
  template <typename S>
  DifferenceMicroOp<N,T>::DifferenceMicroOp(NodeID _requestor,
					    AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> lhs) &&
	       (s >> rhs) &&
	       (s >> sparsity_output));
    assert(ok);
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
  IndexSpace<N,T> UnionOperation<N,T>::add_union(const IndexSpace<N,T>& lhs,
						  const IndexSpace<N,T>& rhs)
  {
    // simple cases should all be handled before we get here, so
    // create a new index space whose bounds can fit both lhs and rhs
    IndexSpace<N,T> output;
    output.bounds = lhs.bounds.union_bbox(rhs.bounds);

    // try to assign sparsity ID near one or both of the input sparsity maps (if present)
    // if the target has a sparsity map, use the same node - otherwise
    // get a sparsity ID by round-robin'ing across the nodes that have field data
    int target_node;
    if(lhs.dense()) {
      if(rhs.dense()) {
	target_node = my_node_id;  // operation will be cheap anyway
      } else {
	target_node = ID(rhs.sparsity).sparsity.creator_node;
      }
    } else {
      if(rhs.dense()) {
	target_node = ID(lhs.sparsity).sparsity.creator_node;
      } else {
	int lhs_node = ID(lhs.sparsity).sparsity.creator_node;
	int rhs_node = ID(rhs.sparsity).sparsity.creator_node;
	//if(lhs_node != rhs_node)
	//  std::cout << "UNION PICK " << lhs_node << " or " << rhs_node << "\n";
	// if they're different, and lhs is us, choose rhs to load-balance maybe
	target_node = (lhs_node == my_node_id) ? rhs_node : lhs_node;
      }
    }
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    output.sparsity = sparsity;

    std::vector<IndexSpace<N,T> > ops(2);
    ops[0] = lhs;
    ops[1] = rhs;
    inputs.push_back(ops);
    outputs.push_back(sparsity);

    return output;
  }

  template <int N, typename T>
  IndexSpace<N,T> UnionOperation<N,T>::add_union(const std::vector<IndexSpace<N,T> >& ops)
  {
    // simple cases should be handled before we get here
    assert(ops.size() > 1);

    // build a bounding box that can hold all the operands
    IndexSpace<N,T> output(ops[0].bounds);
    for(size_t i = 1; i < ops.size(); i++)
      output.bounds = output.bounds.union_bbox(ops[i].bounds);

    // try to assign sparsity ID near the input sparsity maps (if present)
    int target_node = my_node_id;
    int node_count = 0;
    for(size_t i = 0; i < ops.size(); i++)
      if(!ops[i].dense()) {
	int node = ID(ops[i].sparsity).sparsity.creator_node;
	if(node_count == 0) {
	  node_count = 1;
	  target_node = node;
	} else if((node_count == 1) && (node != target_node)) {
	  //std::cout << "UNION DIFF " << target_node << " or " << node << "\n";
	  target_node = my_node_id;
	  break;
	}
      }
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    output.sparsity = sparsity;

    inputs.push_back(ops);
    outputs.push_back(sparsity);

    return output;
  }

  template <int N, typename T>
  void UnionOperation<N,T>::execute(void)
  {
    for(size_t i = 0; i < outputs.size(); i++) {
      SparsityMapImpl<N,T>::lookup(outputs[i])->set_contributor_count(1);

      UnionMicroOp<N,T> *uop = new UnionMicroOp<N,T>(inputs[i]);
      uop->add_sparsity_output(outputs[i]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }

  template <int N, typename T>
  void UnionOperation<N,T>::print(std::ostream& os) const
  {
    os << "UnionOperation";
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
  IndexSpace<N,T> IntersectionOperation<N,T>::add_intersection(const IndexSpace<N,T>& lhs,
								const IndexSpace<N,T>& rhs)
  {
    IndexSpace<N,T> output;
    output.bounds = lhs.bounds.intersection(rhs.bounds);
    
    if(output.bounds.empty()) {
      // this optimization should be handled earlier
      assert(0);
      output.sparsity.id = 0;
      return output;
    }

    // try to assign sparsity ID near one or both of the input sparsity maps (if present)
    // if the target has a sparsity map, use the same node - otherwise
    // get a sparsity ID by round-robin'ing across the nodes that have field data
    int target_node;
    if(lhs.dense()) {
      if(rhs.dense()) {
	target_node = my_node_id;  // operation will be cheap anyway
      } else {
	target_node = ID(rhs.sparsity).sparsity.creator_node;
      }
    } else {
      if(rhs.dense()) {
	target_node = ID(lhs.sparsity).sparsity.creator_node;
      } else {
	int lhs_node = ID(lhs.sparsity).sparsity.creator_node;
	int rhs_node = ID(rhs.sparsity).sparsity.creator_node;
	//if(lhs_node != rhs_node)
	//  std::cout << "ISECT PICK " << lhs_node << " or " << rhs_node << "\n";
	// if they're different, and lhs is us, choose rhs to load-balance maybe
	target_node = (lhs_node == my_node_id) ? rhs_node : lhs_node;
      }
    }
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    output.sparsity = sparsity;

    std::vector<IndexSpace<N,T> > ops(2);
    ops[0] = lhs;
    ops[1] = rhs;
    inputs.push_back(ops);
    outputs.push_back(sparsity);

    return output;
  }

  template <int N, typename T>
  IndexSpace<N,T> IntersectionOperation<N,T>::add_intersection(const std::vector<IndexSpace<N,T> >& ops)
  {
    // simple cases should be handled before we get here
    assert(ops.size() > 1);

    // build the intersection of all bounding boxes
    IndexSpace<N,T> output(ops[0].bounds);
    for(size_t i = 1; i < ops.size(); i++)
      output.bounds = output.bounds.intersection(ops[i].bounds);

    // another optimization handled above
    assert(!output.bounds.empty());

    // try to assign sparsity ID near the input sparsity maps (if present)
    int target_node = my_node_id;
    int node_count = 0;
    for(size_t i = 0; i < ops.size(); i++)
      if(!ops[i].dense()) {
	int node = ID(ops[i].sparsity).sparsity.creator_node;
	if(node_count == 0) {
	  node_count = 1;
	  target_node = node;
	} else if((node_count == 1) && (node != target_node)) {
	  //std::cout << "ISECT DIFF " << target_node << " or " << node << "\n";
	  target_node = my_node_id;
	  break;
	}
      }
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    output.sparsity = sparsity;

    inputs.push_back(ops);
    outputs.push_back(sparsity);

    return output;
  }

  template <int N, typename T>
  void IntersectionOperation<N,T>::execute(void)
  {
    for(size_t i = 0; i < outputs.size(); i++) {
      SparsityMapImpl<N,T>::lookup(outputs[i])->set_contributor_count(1);

      IntersectionMicroOp<N,T> *uop = new IntersectionMicroOp<N,T>(inputs[i]);
      uop->add_sparsity_output(outputs[i]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }

  template <int N, typename T>
  void IntersectionOperation<N,T>::print(std::ostream& os) const
  {
    os << "IntersectionOperation";
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
  IndexSpace<N,T> DifferenceOperation<N,T>::add_difference(const IndexSpace<N,T>& lhs,
							    const IndexSpace<N,T>& rhs)
  {
    if(lhs.empty() || (rhs.dense() && rhs.bounds.contains(lhs.bounds))) {
      // optimization should be handled above
      assert(0);
      return IndexSpace<N,T>::make_empty();
    }

    // the difference is no larger than the lhs
    IndexSpace<N,T> output;
    output.bounds = lhs.bounds;

    // try to assign sparsity ID near one or both of the input sparsity maps (if present)
    // if the target has a sparsity map, use the same node - otherwise
    // get a sparsity ID by round-robin'ing across the nodes that have field data
    int target_node;
    if(lhs.dense()) {
      if(rhs.dense()) {
	target_node = my_node_id;  // operation will be cheap anyway
      } else {
	target_node = ID(rhs.sparsity).sparsity.creator_node;
      }
    } else {
      if(rhs.dense()) {
	target_node = ID(lhs.sparsity).sparsity.creator_node;
      } else {
	int lhs_node = ID(lhs.sparsity).sparsity.creator_node;
	int rhs_node = ID(rhs.sparsity).sparsity.creator_node;
	//if(lhs_node != rhs_node)
	//  std::cout << "DIFF PICK " << lhs_node << " or " << rhs_node << "\n";
	// if they're different, and lhs is us, choose rhs to load-balance maybe
	target_node = (lhs_node == my_node_id) ? rhs_node : lhs_node;
      }
    }
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
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
      SparsityMapImpl<N,T>::lookup(outputs[i])->set_contributor_count(1);

      DifferenceMicroOp<N,T> *uop = new DifferenceMicroOp<N,T>(lhss[i], rhss[i]);
      uop->add_sparsity_output(outputs[i]);
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }

  template <int N, typename T>
  void DifferenceOperation<N,T>::print(std::ostream& os) const
  {
    os << "DifferenceOperation";
  }


#define DOIT(N,T) \
  template class UnionMicroOp<N,T>; \
  template class IntersectionMicroOp<N,T>; \
  template class DifferenceMicroOp<N,T>; \
  template class UnionOperation<N,T>; \
  template class IntersectionOperation<N,T>; \
  template class DifferenceOperation<N,T>; \
  template UnionMicroOp<N,T>::UnionMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template IntersectionMicroOp<N,T>::IntersectionMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template DifferenceMicroOp<N,T>::DifferenceMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template Event IndexSpace<N,T>::compute_unions(const std::vector<IndexSpace<N,T> >&, \
						  const std::vector<IndexSpace<N,T> >&, \
						  std::vector<IndexSpace<N,T> >&, \
						  const ProfilingRequestSet &, \
						  Event); \
  template Event IndexSpace<N,T>::compute_intersections(const std::vector<IndexSpace<N,T> >&, \
							 const std::vector<IndexSpace<N,T> >&, \
							 std::vector<IndexSpace<N,T> >&, \
							 const ProfilingRequestSet &, \
							 Event); \
  template Event IndexSpace<N,T>::compute_differences(const std::vector<IndexSpace<N,T> >&, \
						       const std::vector<IndexSpace<N,T> >&, \
						       std::vector<IndexSpace<N,T> >&, \
						       const ProfilingRequestSet &, \
						       Event); \
  template Event IndexSpace<N,T>::compute_union(const std::vector<IndexSpace<N,T> >&, \
						 IndexSpace<N,T>&, \
						 const ProfilingRequestSet &, \
						 Event); \
  template Event IndexSpace<N,T>::compute_intersection(const std::vector<IndexSpace<N,T> >&, \
							IndexSpace<N,T>&, \
							const ProfilingRequestSet &, \
							Event);
  FOREACH_NT(DOIT)
};
