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

// index space partitioning for Realm

#include "realm/deppart/partitions.h"

#include "realm/profiling.h"

#include "realm/runtime_impl.h"
#include "realm/deppart/inst_helper.h"
#include "realm/deppart/rectlist.h"

#include "realm/deppart/image.h"
#include "realm/deppart/preimage.h"
#include "realm/deppart/byfield.h"
#include "realm/deppart/setops.h"

namespace Realm {

  Logger log_part("part");
  Logger log_uop_timing("uop_timing");
  REALM_INTERNAL_API_EXTERNAL_LINKAGE
  Logger log_dpops("dpops");

  PartitioningOpQueue *deppart_op_queue = 0;

  namespace DeppartConfig {

    int cfg_num_partitioning_workers = 0; // use bgwork by default
    bool cfg_disable_intersection_optimization = false;
    int cfg_max_rects_in_approximation = 32;
    bool cfg_worker_threads_sleep = true;
    bool cfg_allow_inline_operations = false;
  };

  // TODO: C++11 has type_traits and std::make_unsigned
  namespace {
    template <typename T> struct MakeUnsigned { typedef T U; };
#define SIGNED_CASE(S) \
    template <> struct MakeUnsigned<S> { typedef unsigned S U; }
    SIGNED_CASE(int);
    SIGNED_CASE(short);
    SIGNED_CASE(long);
    SIGNED_CASE(long long);
#undef SIGNED_CASE
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class IndexSpace<N,T>

  // this is used by create_equal_subspace(s) to decompose an N-D rectangle
  //  into bounds with roughly-even volumes in a sparse index space
  template <int N, typename T>
  static void recursive_decomposition(const Rect<N,T>& bounds,
				      size_t start, size_t count, size_t volume,
				      IndexSpace<N,T> *results,
				      size_t first_result, size_t last_result,
				      const std::vector<SparsityMapEntry<N,T> >& entries)
  {
    // should never be here with empty bounds
    assert(!bounds.empty());

    // if we have more subspaces than elements, the last subspaces will be
    //  empty
    if(count > volume) {
      size_t first_empty = std::max(first_result, start + volume);
      size_t last_empty = std::min(start + count - 1, last_result);
      for(size_t i = first_empty; i <= last_empty; i++)
	results[i - first_result] = IndexSpace<N,T>::make_empty();
      if(volume == 0) {
	// this shouldn't happen in the recursion case, but can happen on the
	//  initial call for an empty parent space
	return;
      }
      count = volume;
    }

    // base case
    if(count == 1) {
      //log_part.info() << "split result " << start << " = " << bounds;
      // save these bounds if they're one we care about
      if((start >= first_result) && (start <= last_result))
	results[start - first_result].bounds = bounds;
      return;
    }

    // look at half-ish splits in each dimension and choose the one that
    //  yields the best division
    Rect<N,T> lo_half[N];
    for(int i = 0; i < N; i++) {
      T split = bounds.lo[i] + ((bounds.hi[i] - bounds.lo[i]) >> 1);
      lo_half[i] = bounds;  lo_half[i].hi[i] = split;
    }
    // compute the volume within each split domain
    size_t lo_volume[N];
    for(int i = 0; i < N; i++)
      lo_volume[i] = 0;
    for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	it != entries.end();
	it++) {
      for(int i = 0; i < N; i++)
	lo_volume[i] += it->bounds.intersection(lo_half[i]).volume();
    }
    // now compute how many subspaces would fall in each half and the 
    //  inefficiency of the split
    size_t lo_count[N], inefficiency[N];
    for(int i = 0; i < N; i++) {
      lo_count[i] = (count * lo_volume[i] + (volume >> 1)) / volume;
      // lo_count can't be 0 or count if the volume split is nontrivial
      if((lo_volume[i] > 0) && (lo_count[i] == 0))
	lo_count[i] = 1;
      if((lo_volume[i] < volume) && (lo_count[i] == count))
	lo_count[i] = count - 1;
      int delta = (count * lo_volume[i]) - (lo_count[i] * volume);
      // the high-order inefficiency comes from subspaces with extra elements
      inefficiency[i] = 0;
      if(delta > 0)
	// low half has too many
	inefficiency[i] = (delta + lo_count[i] - 1) / lo_count[i];
      if(delta < 0)
	// hi half has too many
	inefficiency[i] = (-delta + (count - lo_count[i]) - 1) / (count - lo_count[i]);
      // low-order inefficiency comes from an uneven split in the coutns
      inefficiency[i] = inefficiency[i] * count + std::max(lo_count[i],
							   count - lo_count[i]);
      //log_part.info() << "dim " << i << ": half=" << lo_half[i] << " vol=" << lo_volume[i] << "/" << volume << " count=" << lo_count[i] << "/" << count << " delta=" << delta << " ineff=" << inefficiency[i];
    }
    int best_dim = -1;
    for(int i = 0; i < N; i++) {
      // can't split a dimension that was already minimal size
      if(bounds.lo[i] == bounds.hi[i]) continue;
      if((best_dim < 0) || (inefficiency[i] < inefficiency[best_dim]))
	best_dim = i;
    }
    assert(best_dim >= 0);

    //log_part.info() << "split: " << bounds << " count=" << count << " dim=" << best_dim << " half=" << lo_half[best_dim] << " count=" << lo_count[best_dim] << " ineff=" << inefficiency[best_dim];
    // recursive split
    if(lo_count[best_dim] > 0)
      recursive_decomposition(lo_half[best_dim],
			      start, lo_count[best_dim], lo_volume[best_dim],
			      results, first_result, last_result,
			      entries);
    if(lo_count[best_dim] < count) {
      Rect<N,T> hi_half = bounds;
      hi_half.lo[best_dim] = lo_half[best_dim].hi[best_dim] + 1;
      recursive_decomposition(hi_half,
			      start + lo_count[best_dim],
			      count - lo_count[best_dim],
			      volume - lo_volume[best_dim],
			      results, first_result, last_result,
			      entries);
    }
  }

  template <int N, typename T>
  Event IndexSpace<N,T>::create_equal_subspace(size_t count, size_t granularity,
						unsigned index, IndexSpace<N,T> &subspace,
						const ProfilingRequestSet &reqs,
						Event wait_on /*= Event::NO_EVENT*/) const
  {
    // must always be creating at least one subspace (no "divide by zero")
    assert(count >= 1);

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

    // either an empty input or a count of 1 allow us to return the input
    //  verbatim
    if(empty() || (count == 1)) {
      subspace = *this;
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
      return wait_on;
    }

    // dense case is easy(er)
    if(dense()) {
      // split in the largest dimension available
      // avoiding over/underflow here is tricky - use unsigned math and watch
      //  out for empty subspace case
      // TODO: still not handling maximal-size input properly
      int split_dim = 0;
      typedef typename MakeUnsigned<T>::U U;
      U total = std::max(U(bounds.hi[0]) - U(bounds.lo[0]) + 1, U(0));
      if(N > 1)
	for(int i = 1; i < N; i++) {
	  U extent = std::max(U(bounds.hi[i]) - U(bounds.lo[i]) + 1, U(0));
	  if(extent > total) {
	    total = extent;
	    split_dim = i;
	  }
	}
      // have to divide before multiplying to avoid overflow
      U base_span_size = total / count;
      U base_span_rem = total - (base_span_size * count);
      U rel_span_start = index * base_span_size;
      U rel_span_size = base_span_size;
      if(base_span_rem != 0) {
	U start_adj = index * base_span_rem / count;
	U end_adj = (index + 1) * base_span_rem / count;
	rel_span_start += start_adj;
	rel_span_size += (end_adj - start_adj);
      }
      if(rel_span_size > 0) {
	subspace = *this;
	subspace.bounds.lo[split_dim] = bounds.lo[split_dim] + rel_span_start;
	subspace.bounds.hi[split_dim] = (bounds.lo[split_dim] +
					 rel_span_start + (rel_span_size - 1));
      } else {
	subspace = IndexSpace<N,T>::make_empty();
      }
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
      return wait_on;
    }

    // TODO: sparse case where we have to wait
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    assert(impl->is_valid());
    const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
    // initially every subspace will be a copy of this one, and then
    //  we'll decompose the bounds
    subspace = *this;
    recursive_decomposition(this->bounds, 0, count, this->volume(),
			    &subspace, index, index,
			    entries);
    PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
    return wait_on;
  }

  template <int N, typename T>
  Event IndexSpace<N,T>::create_equal_subspaces(size_t count, size_t granularity,
						 std::vector<IndexSpace<N,T> >& subspaces,
						 const ProfilingRequestSet &reqs,
						 Event wait_on /*= Event::NO_EVENT*/) const
  {
    // output vector should start out empty
    assert(subspaces.empty());
    // must always be creating at least one subspace (no "divide by zero")
    assert(count >= 1);

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

    // partitioning an empty space and/or making a single subspace is easy
    if(empty() || (count == 1)) {
      subspaces.resize(count, *this);
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
      return wait_on;
    }

    // dense case is easy(er)
    if(dense()) {
      subspaces.reserve(count);
      // split in the largest dimension available
      int split_dim = 0;
      T total = std::max(bounds.hi[0] - bounds.lo[0] + 1, T(0));
      if(N > 1)
	for(int i = 1; i < N; i++) {
	  T extent = std::max(bounds.hi[i] - bounds.lo[i] + 1, T(0));
	  if(extent > total) {
	    total = extent;
	    split_dim = i;
	  }
	}
      T px = bounds.lo[split_dim];
      // have to divide before multiplying to avoid overflow
      T base_span_size = total / count;
      T base_span_rem = total - (base_span_size * count);
      T leftover = 0;
      for(size_t i = 0; i < count; i++) {
	IndexSpace<N,T> ss(*this);
	T nx = px + (base_span_size - 1);
	if(base_span_rem != 0) {
	  leftover += base_span_rem;
	  if(leftover >= T(count)) {
	    nx += 1;
	    leftover -= count;
	  }
	}
	ss.bounds.lo[split_dim] = px;
	ss.bounds.hi[split_dim] = nx;
	subspaces.push_back(ss);
	px = nx + 1;
      }
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
      return wait_on;
    }

    // TODO: sparse case where we have to wait
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    assert(impl->is_valid());
    const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
    // initially every subspace will be a copy of this one, and then
    //  we'll decompose the bounds
    subspaces.resize(count, *this);
    recursive_decomposition(this->bounds, 0, count, this->volume(),
			    subspaces.data(), 0, count-1,
			    entries);
    PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
    return wait_on;
  }

  template <int N, typename T>
  Event IndexSpace<N,T>::create_weighted_subspaces(size_t count, size_t granularity,
						   const std::vector<size_t>& weights,
						   std::vector<IndexSpace<N,T> >& subspaces,
						   const ProfilingRequestSet &reqs,
						   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // output vector should start out empty
    assert(subspaces.empty());

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

    // partitioning an empty space and/or making a single subspace is easy
    if(empty() || (count == 1)) {
      subspaces.resize(count, *this);
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
      return wait_on;
    }

    // determine the total weight
    size_t total_weight = 0;
    assert(weights.size() == count);
    for(size_t i = 0; i < count; i++)
      total_weight += weights[i];

    // dense case is easy(er)
    if(dense()) {
      // always split in x dimension for now
      assert(count >= 1);
      // unsafe to subtract and test against zero - compare first
      size_t total_x;
      if(bounds.lo.x <= bounds.hi.x)
        total_x = ((long long)bounds.hi.x) - ((long long)bounds.lo.x) + 1;
      else
        total_x = 0;
      subspaces.reserve(count);
      T px = bounds.lo.x;
      size_t cum_weight = 0;
      for(size_t i = 0; i < count; i++) {
	IndexSpace<N,T> ss(*this);
	cum_weight += weights[i];
        // if the total_weight cleanly divides into the total x, use
        //  that ratio to avoid overflow problems
        T nx;
        if((total_x % total_weight) == 0)
          nx = bounds.lo.x + cum_weight * (total_x / total_weight);
        else
	  nx = bounds.lo.x + (total_x * cum_weight / total_weight);
	// wrap-around here means bad math
	assert(nx >= px);
	ss.bounds.lo.x = px;
	ss.bounds.hi.x = nx - 1;
	subspaces.push_back(ss);
	px = nx;
      }
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
      return wait_on;
    }

    // TODO: sparse case
    assert(0);
    return wait_on;
  }

  template <int N, typename T>
  template <int N2, typename T2>
  Event IndexSpace<N,T>::create_association(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
					                       Point<N2,T2> > >& field_data,
					     const IndexSpace<N2,T2> &range,
					     const ProfilingRequestSet &reqs,
					     Event wait_on /*= Event::NO_EVENT*/) const
  {
    assert(0);
    return wait_on;
  }

  template <int N, typename T>
  bool IndexSpace<N,T>::compute_covering(size_t max_rects, int max_overhead,
					 std::vector<Rect<N,T> >& covering) const
  {
    // handle really simple cases first
    if(empty()) {
      covering.clear();
      return true;
    }

    if(dense()) {
      covering.resize(1);
      covering[0] = bounds;
      return true;
    }

    // anything else requires sparsity data - we'd better have it
    SparsityMapPublicImpl<N,T> *impl = sparsity.impl();
    assert(impl->is_valid(true /*precise*/) &&
	   "IndexSpace<N,T>::compute_covering called without waiting for valid metadata");

    return impl->compute_covering(bounds, max_rects, max_overhead,
				  covering);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class OverlapTester<N,T>

  template <int N, typename T>
  OverlapTester<N,T>::OverlapTester(void)
  {}

  template <int N, typename T>
  OverlapTester<N,T>::~OverlapTester(void)
  {}

  template <int N, typename T>
  void OverlapTester<N,T>::add_index_space(int label, const IndexSpace<N,T>& space,
					   bool use_approx /*= true*/)
  {
    labels.push_back(label);
    spaces.push_back(space);
    approxs.push_back(use_approx);
  }

  template <int N, typename T>
  void OverlapTester<N,T>::construct(void)
  {
    // nothing special yet
  }

  template <int N, typename T>
  void OverlapTester<N,T>::test_overlap(const Rect<N,T> *rects, size_t count, std::set<int>& overlaps)
  {
    for(size_t i = 0; i < labels.size(); i++)
      if(approxs[i]) {
	for(size_t j = 0; j < count; j++)
	  if(spaces[i].overlaps_approx(rects[j])) {
	    overlaps.insert(labels[i]);
	    break;
	  }
      } else {
	for(size_t j = 0; j < count; j++)
	  if(spaces[i].overlaps(rects[j])) {
	    overlaps.insert(labels[i]);
	    break;
	  }
      }
  }

  template <int N, typename T>
  void OverlapTester<N,T>::test_overlap(const IndexSpace<N,T>& space, std::set<int>& overlaps,
					bool approx)
  {
    for(size_t i = 0; i < labels.size(); i++)
      if(approxs[i] && approx) {
	if(space.overlaps_approx(spaces[i]))
	  overlaps.insert(labels[i]);
      } else {
	if(space.overlaps(spaces[i]))
	  overlaps.insert(labels[i]);
      }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class OverlapTester<1,T>

  template <typename T>
  OverlapTester<1,T>::OverlapTester(void)
  {}

  template <typename T>
  OverlapTester<1,T>::~OverlapTester(void)
  {}

  template <typename T>
  class RectListAdapter {
  public:
    RectListAdapter(const std::vector<Rect<1,T> >& _rects)
      : rects(_rects.empty() ? 0 : &_rects[0]), count(_rects.size()) {}
    RectListAdapter(const Rect<1,T> *_rects, size_t _count)
      : rects(_rects), count(_count) {}
    size_t size(void) const { return count; }
    T start(size_t idx) const { return rects[idx].lo.x; }
    T end(size_t idx) const { return rects[idx].hi.x; }
  protected:
    const Rect<1,T> *rects;
    size_t count;
  };

  template <typename T>
  void OverlapTester<1,T>::add_index_space(int label, const IndexSpace<1,T>& space,
					   bool use_approx /*= true*/)
  {
    if(use_approx) {
      if(space.dense())
	interval_tree.add_interval(space.bounds.lo.x, space.bounds.hi.x,label);
      else {
	SparsityMapImpl<1,T> *impl = SparsityMapImpl<1,T>::lookup(space.sparsity);
	interval_tree.add_intervals(RectListAdapter<T>(impl->get_approx_rects()), label);
      }
    } else {
      for(IndexSpaceIterator<1,T> it(space); it.valid; it.step())
	interval_tree.add_interval(it.rect.lo.x, it.rect.hi.x, label);
    }
  }

  template <typename T>
  void OverlapTester<1,T>::construct(void)
  {
    interval_tree.construct_tree();
  }

  template <typename T>
  void OverlapTester<1,T>::test_overlap(const Rect<1,T> *rects, size_t count, std::set<int>& overlaps)
  {
    interval_tree.test_sorted_intervals(RectListAdapter<T>(rects, count), overlaps);
  }

  template <typename T>
  void OverlapTester<1,T>::test_overlap(const IndexSpace<1,T>& space, std::set<int>& overlaps,
					bool approx)
  {
    if(space.dense()) {
      interval_tree.test_interval(space.bounds.lo.x, space.bounds.hi.x, overlaps);
    } else {
      if(approx) {
	SparsityMapImpl<1,T> *impl = SparsityMapImpl<1,T>::lookup(space.sparsity);
	interval_tree.test_sorted_intervals(RectListAdapter<T>(impl->get_approx_rects()), overlaps);
      } else {
	for(IndexSpaceIterator<1,T> it(space); it.valid; it.step())
	  interval_tree.test_interval(it.rect.lo.x, it.rect.hi.x, overlaps);
      }
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class AsyncMicroOp

  AsyncMicroOp::AsyncMicroOp(Operation *_op, PartitioningMicroOp *_uop)
    : Operation::AsyncWorkItem(_op)
    , uop(_uop)
  {}

  AsyncMicroOp::~AsyncMicroOp()
  {
    if(uop)
      delete uop;
  }
    
  void AsyncMicroOp::request_cancellation(void)
  {
    // ignored
  }

  void AsyncMicroOp::print(std::ostream& os) const
  {
    os << "AsyncMicroOp(" << (void *)uop << ")";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningMicroOp

  PartitioningMicroOp::PartitioningMicroOp(void)
    : wait_count(2), requestor(Network::my_node_id), async_microop(0)
  {}

  PartitioningMicroOp::PartitioningMicroOp(NodeID _requestor,
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
      if(requestor == Network::my_node_id) {
	async_microop->mark_finished(true /*successful*/);
	// async micro op will delete us when it's ready
      } else {
	ActiveMessage<RemoteMicroOpCompleteMessage> amsg(requestor);
	amsg->async_microop = async_microop;
	amsg.commit();
	delete this;
      }
    } else {
      delete this;
    }
  }

  template <int N, typename T>
  void PartitioningMicroOp::sparsity_map_ready(SparsityMapImpl<N,T> *sparsity, bool precise)
  {
    int left = wait_count.fetch_sub(1) - 1;
    if(left == 0)
      deppart_op_queue->enqueue_partitioning_microop(this);
  }

  void PartitioningMicroOp::finish_dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // make sure we generate work that other threads can help with
    if(!DeppartConfig::cfg_allow_inline_operations)
      inline_ok = false;
    // if there were no registrations by caller (or if they're really fast), the count will be 2
    //  and we can execute this microop inline (if we're allowed to)
    int left1 = wait_count.fetch_sub(1) - 1;
    if((left1 == 1) && inline_ok) {
      mark_started();
      execute();
      mark_finished();
      return;
    }

    // if the count was greater than 1, it probably has to be queued, so create an 
    //  AsyncMicroOp so that the op knows we're not done yet
    if(requestor == Network::my_node_id) {
      async_microop = new AsyncMicroOp(op, this);
      op->add_async_work_item(async_microop);
    } else {
      // request came from somewhere else - it had better have a async_microop already
      assert(async_microop != 0);
    }

    // now do the last decrement - if it returns 0, we can still do the operation inline
    //  (otherwise it'll get queued when somebody else does the last decrement)
    int left2 = wait_count.fetch_sub(1) - 1;
    if(left2 == 0) {
      if(inline_ok) {
	mark_started();
	execute();
	mark_finished();
      } else
	deppart_op_queue->enqueue_partitioning_microop(this);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ComputeOverlapMicroOp<N,T>

  template <int N, typename T>
  ComputeOverlapMicroOp<N,T>::ComputeOverlapMicroOp(PartitioningOperation *_op)
    : op(_op)
  {}

  template <int N, typename T>
  ComputeOverlapMicroOp<N,T>::~ComputeOverlapMicroOp(void)
  {}

  template <int N, typename T>
  void ComputeOverlapMicroOp<N,T>::add_input_space(const IndexSpace<N,T>& input_space)
  {
    input_spaces.push_back(input_space);
  }

  template <int N, typename T>
  void ComputeOverlapMicroOp<N,T>::add_extra_dependency(const IndexSpace<N,T>& dep_space)
  {
    if(!dep_space.dense()) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(dep_space.sparsity);
      extra_deps.push_back(impl);
    }
  }

  template <int N, typename T>
  void ComputeOverlapMicroOp<N,T>::execute(void)
  {
    OverlapTester<N,T> *overlap_tester;
    {
      TimeStamp ts("ComputeOverlapMicroOp::execute", true, &log_uop_timing);

      overlap_tester = new OverlapTester<N,T>;
      for(size_t i = 0; i < input_spaces.size(); i++)
	overlap_tester->add_index_space(i, input_spaces[i]);
      overlap_tester->construct();
    }

    // don't include this call in our timing - it may kick off a bunch of microops that get inlined
    op->set_overlap_tester(overlap_tester);
  }

  template <int N, typename T>
  void ComputeOverlapMicroOp<N,T>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // need valid data for each input space
    for(size_t i = 0; i < input_spaces.size(); i++) {
      if(!input_spaces[i].dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N,T>::lookup(input_spaces[i].sparsity)->add_waiter(this, 
											     true /*precise*/);
	if(registered)
	  wait_count.fetch_add(1);
      }
    }

    // add any extra dependencies too
    for(size_t i = 0; i < extra_deps.size(); i++) {
      bool registered = extra_deps[i]->add_waiter(this, true /*precise*/);
	if(registered)
	  wait_count.fetch_add(1);
    }

    finish_dispatch(op, inline_ok);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMicroOpCompleteMessage

  /*static*/ void RemoteMicroOpCompleteMessage::handle_message(NodeID sender,
							       const RemoteMicroOpCompleteMessage &msg,
							       const void *data, size_t datalen)
  {
    log_part.info() << "received remote micro op complete message: " << msg.async_microop;
    msg.async_microop->mark_finished(true /*successful*/);
  }

  ActiveMessageHandlerReg<RemoteMicroOpCompleteMessage> RemoteMicroOpCompleteMessage::areg;


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningOperation::DeferredLaunch

  void PartitioningOperation::DeferredLaunch::defer(PartitioningOperation *_op,
						Event wait_on)
  {
    op = _op;
    EventImpl::add_waiter(wait_on, this);
  }

  void PartitioningOperation::DeferredLaunch::event_triggered(bool poisoned,
							      TimeLimit work_until)
  {
    assert(!poisoned); // TODO: POISON_FIXME
    deppart_op_queue->enqueue_partitioning_operation(op);
  }

  void PartitioningOperation::DeferredLaunch::print(std::ostream& os) const
  {
    os << "DeferredPartitioningOp(" << (void *)op << ")";
  }
  
  Event PartitioningOperation::DeferredLaunch::get_finish_event(void) const
  {
    return op->get_finish_event();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningOperation

  PartitioningOperation::PartitioningOperation(const ProfilingRequestSet &reqs,
					       GenEventImpl *_finish_event,
					       EventImpl::gen_t _finish_gen)
    : Operation(_finish_event, _finish_gen, reqs)
  {}

  void PartitioningOperation::launch(Event wait_for)
  {
    get_runtime()->optable.add_local_operation(get_finish_event(), this);

    if(wait_for.has_triggered())
      deppart_op_queue->enqueue_partitioning_operation(this);
    else
      deferred_launch.defer(this, wait_for);
  };

  void PartitioningOperation::set_overlap_tester(void *tester)
  {
    // should only be called for ImageOperation and PreimageOperation, which override this
    assert(0);
  }

  /*static*/ void PartitioningOperation::do_inline_profiling(const ProfilingRequestSet &reqs,
							     long long inline_start_time)
  {
    if(!reqs.empty()) {
      using namespace ProfilingMeasurements;
      ProfilingMeasurementCollection pmc;
      pmc.import_requests(reqs);
      if(pmc.wants_measurement<OperationTimeline>()) {
	OperationTimeline t;

	// if we handled the request inline, we need to generate profiling responses
	long long inline_finish_time = Clock::current_time_in_nanoseconds();

	t.create_time = inline_start_time;
	t.ready_time = inline_start_time;
	t.start_time = inline_start_time;
	t.end_time = inline_finish_time;
	t.complete_time = inline_finish_time;
	pmc.add_measurement(t);
      }
      pmc.send_responses(reqs);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningOpQueue

  PartitioningOpQueue::PartitioningOpQueue( CoreReservation *_rsrv,
					    BackgroundWorkManager *_bgwork)
    : BackgroundWorkItem("deppart op queue")
    , shutdown_flag(false), rsrv(_rsrv), condvar(mutex)
    , work_advertised(false)
  {
    if(_bgwork)
      add_to_manager(_bgwork);
  }
  
  PartitioningOpQueue::~PartitioningOpQueue(void)
  {
    assert(shutdown_flag.load());
    delete rsrv;
  }

  /*static*/ void PartitioningOpQueue::configure_from_cmdline(std::vector<std::string>& cmdline)
  {
    CommandLineParser cp;

    cp.add_option_int("-dp:workers", DeppartConfig::cfg_num_partitioning_workers);
    cp.add_option_bool("-dp:noisectopt", DeppartConfig::cfg_disable_intersection_optimization);
    cp.add_option_int("-dp:sleep", DeppartConfig::cfg_worker_threads_sleep);
    cp.add_option_int("-dp:inline_ok", DeppartConfig::cfg_allow_inline_operations);

    cp.parse_command_line(cmdline);
  }

  /*static*/ void PartitioningOpQueue::start_worker_threads(CoreReservationSet& crs,
				     BackgroundWorkManager *_bgwork)
  {
    assert(deppart_op_queue == 0);
    CoreReservation *rsrv = 0;
    if(DeppartConfig::cfg_num_partitioning_workers > 0)
      rsrv = new CoreReservation("partitioning", crs,
				 CoreReservationParameters());
    deppart_op_queue = new PartitioningOpQueue(rsrv, _bgwork);
    ThreadLaunchParameters tlp;
    for(int i = 0; i < DeppartConfig::cfg_num_partitioning_workers; i++) {
      Thread *t = Thread::create_kernel_thread<PartitioningOpQueue,
					       &PartitioningOpQueue::worker_thread_loop>(deppart_op_queue,
											 tlp,
											 *rsrv);
      deppart_op_queue->workers.push_back(t);
    }
  }

  /*static*/ void PartitioningOpQueue::stop_worker_threads(void)
  {
    assert(deppart_op_queue != 0);

#ifdef DEBUG_REALM
    deppart_op_queue->shutdown_work_item();
#endif

    deppart_op_queue->shutdown_flag.store(true);
    {
      AutoLock<> al(deppart_op_queue->mutex);
      deppart_op_queue->condvar.broadcast();
    }
    for(size_t i = 0; i < deppart_op_queue->workers.size(); i++) {
      deppart_op_queue->workers[i]->join();
      delete deppart_op_queue->workers[i];
    }
    deppart_op_queue->workers.clear();

    delete deppart_op_queue;
    deppart_op_queue = 0;
  }
      
  void PartitioningOpQueue::enqueue_partitioning_operation(PartitioningOperation *op)
  {
    op->mark_ready();

    bool need_advertise;
    {
      AutoLock<> al(mutex);

      need_advertise = !work_advertised;
      work_advertised = true;
      op_list.push_back(op);

      if(!workers.empty())
	deppart_op_queue->condvar.broadcast();
    }

    if(need_advertise)
      make_active();
  }

  void PartitioningOpQueue::enqueue_partitioning_microop(PartitioningMicroOp *uop)
  {
    bool need_advertise;
    {
      AutoLock<> al(mutex);

      need_advertise = !work_advertised;
      work_advertised = true;
      uop_list.push_back(uop);

      if(!workers.empty())
	deppart_op_queue->condvar.broadcast();
    }

    if(need_advertise)
      make_active();
  }

  bool PartitioningOpQueue::do_work(TimeLimit work_until)
  {
    // attempt to take one item off the work queue - readvertise work if
    //  more remains
    PartitioningOperation *op = 0;
    PartitioningMicroOp *uop = 0;
    bool readvertise;
    {
      AutoLock<> al(mutex);

      // prefer micro ops over operations
      if(!uop_list.empty())
	uop = uop_list.pop_front();
      else if(!op_list.empty())
	op = op_list.pop_front();

#ifdef DEBUG_REALM
      assert(work_advertised);
#endif
      work_advertised = !op_list.empty() || !uop_list.empty();
      readvertise = work_advertised;
    }
    if(readvertise) {
      assert(((op != 0) || (uop != 0)) && (manager != 0));
      make_active();
    }

    // now we can work on the op we got in parallel with everybody else
    //  (neither branch will be taken if there are dedicated workers and they
    //  already got to the queued operations)
    if(op != 0) {
      bool ok_to_run = op->mark_started();
      if(ok_to_run) {
	log_part.info() << "worker " << this << " starting op " << op;
	op->execute();
	log_part.info() << "worker " << this << " finished op " << op;
	op->mark_finished(true /*successful*/);
      } else {
	log_part.info() << "worker " << this << " cancelled op " << op;
	op->mark_finished(false /*!successful*/);
      }
    }

    if(uop != 0) {
      log_part.info() << "worker " << this << " starting uop " << uop;
      uop->mark_started();
      uop->execute();
      log_part.info() << "worker " << this << " finished uop " << uop;
      uop->mark_finished();
    }

    // make_active was called above (if needed)
    return false;
  }

  void PartitioningOpQueue::worker_thread_loop(void)
  {
    log_part.info() << "worker " << Thread::self() << " started for op queue " << this;

    while(!shutdown_flag.load()) {
      PartitioningOperation *op = 0;
      PartitioningMicroOp *uop = 0;
      while(!op && !uop && !shutdown_flag.load()) {
	AutoLock<> al(mutex);

	// prefer micro ops over operations
	if(!uop_list.empty())
	  uop = uop_list.pop_front();
	else if(!op_list.empty())
	  op = op_list.pop_front();

	if(!op && !uop && !shutdown_flag.load()) {
          if(DeppartConfig::cfg_worker_threads_sleep) {
	    condvar.wait();
          } else {
            mutex.unlock();
            Thread::yield();
            mutex.lock();
          }
        }
      }

      if(op) {
	bool ok_to_run = op->mark_started();
	if(ok_to_run) {
	  log_part.info() << "worker " << this << " starting op " << op;
	  op->execute();
	  log_part.info() << "worker " << this << " finished op " << op;
	  op->mark_finished(true /*successful*/);
	} else {
	  log_part.info() << "worker " << this << " cancelled op " << op;
	  op->mark_finished(false /*!successful*/);
	}
      }

      if(uop) {
	log_part.info() << "worker " << this << " starting uop " << uop;
	uop->mark_started();
	uop->execute();
	log_part.info() << "worker " << this << " finished uop " << uop;
	uop->mark_finished();
      }
    }

    log_part.info() << "worker " << Thread::self() << " finishing for op queue " << this;
  }

#define DOIT(N,T) \
  template struct IndexSpace<N,T>; \
  template void PartitioningMicroOp::sparsity_map_ready(SparsityMapImpl<N,T>*, bool); \
  template class OverlapTester<N,T>; \
  template class ComputeOverlapMicroOp<N,T>;
  FOREACH_NT(DOIT)

#define DOIT2(N1,T1,N2,T2) \
  template Event IndexSpace<N1,T1>::create_association(std::vector<FieldDataDescriptor<IndexSpace<N1,T1>, Point<N2,T2> > > const&, IndexSpace<N2,T2> const&, ProfilingRequestSet const&, Event) const;
  FOREACH_NTNT(DOIT2)

};

