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
  Logger log_dpops("dpops");

  namespace {
    // module-level globals

    PartitioningOpQueue *op_queue = 0;

  };

  namespace DeppartConfig {

    int cfg_num_partitioning_workers = 1;
    bool cfg_disable_intersection_optimization = false;
    int cfg_max_rects_in_approximation = 32;
    size_t cfg_max_bytes_per_packet = 2048;//32768;
    bool cfg_worker_threads_sleep = false;
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

  template <int N, typename T>
  Event IndexSpace<N,T>::create_equal_subspace(size_t count, size_t granularity,
						unsigned index, IndexSpace<N,T> &subspace,
						const ProfilingRequestSet &reqs,
						Event wait_on /*= Event::NO_EVENT*/) const
  {
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
      // always split in x dimension for now
      assert(count >= 1);
      // avoiding over/underflow here is tricky - use unsigned math and watch
      //  out for empty subspace case
      // TODO: still not handling maximal-size input properly
      typedef typename MakeUnsigned<T>::U U;
      U total_x = std::max(((U)bounds.hi.x -
			    (U)bounds.lo.x + 1),
			   U(0));
      U rel_span_start = (total_x * index / count);
      U rel_span_size = (total_x * (index + 1) / count) - rel_span_start;
      if(rel_span_size > 0) {
	subspace = *this;
	subspace.bounds.lo.x = bounds.lo.x + rel_span_start;
	subspace.bounds.hi.x = bounds.lo.x + rel_span_start + (rel_span_size - 1);
      } else {
	subspace = IndexSpace<N,T>::make_empty();
      }
      PartitioningOperation::do_inline_profiling(reqs, inline_start_time);
      return wait_on;
    }

    // TODO: sparse case
    assert(0);
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

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

    // dense case is easy(er)
    if(dense()) {
      // always split in x dimension for now
      assert(count >= 1);
      T total_x = std::max(bounds.hi.x - bounds.lo.x + 1, T(0));
      subspaces.reserve(count);
      T px = bounds.lo.x;
      for(size_t i = 0; i < count; i++) {
	IndexSpace<N,T> ss(*this);
	T nx = bounds.lo.x + (total_x * (i + 1) / count);
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
  Event IndexSpace<N,T>::create_weighted_subspaces(size_t count, size_t granularity,
						    const std::vector<int>& weights,
						    std::vector<IndexSpace<N,T> >& subspaces,
						    const ProfilingRequestSet &reqs,
						    Event wait_on /*= Event::NO_EVENT*/) const
  {
    // output vector should start out empty
    assert(subspaces.empty());

    // record the start time of the potentially-inline operation if any
    //  profiling has been requested
    long long inline_start_time = reqs.empty() ? 0 : Clock::current_time_in_nanoseconds();

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
      : rects(&_rects[0]), count(_rects.size()) {}
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
    : wait_count(2), requestor(my_node_id), async_microop(0)
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
      if(requestor == my_node_id)
	async_microop->mark_finished(true /*successful*/);
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
    // make sure we generate work that other threads can help with
    if(DeppartConfig::cfg_num_partitioning_workers > 1)
      inline_ok = false;
    // if there were no registrations by caller (or if they're really fast), the count will be 2
    //  and we can execute this microop inline (if we're allowed to)
    int left1 = __sync_sub_and_fetch(&wait_count, 1);
    if((left1 == 1) && inline_ok) {
      mark_started();
      execute();
      mark_finished();
      return;
    }

    // if the count was greater than 1, it probably has to be queued, so create an 
    //  AsyncMicroOp so that the op knows we're not done yet
    if(requestor == my_node_id) {
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
	  __sync_fetch_and_add(&wait_count, 1);
      }
    }

    // add any extra dependencies too
    for(size_t i = 0; i < extra_deps.size(); i++) {
      bool registered = extra_deps[i]->add_waiter(this, true /*precise*/);
	if(registered)
	  __sync_fetch_and_add(&wait_count, 1);
    }

    finish_dispatch(op, inline_ok);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMicroOpMessage

  template <typename NT, typename T, typename FT>
  inline /*static*/ void RemoteMicroOpMessage::ByFieldDecoder::demux(const RequestArgs *args,
								     const void *data,
								     size_t datalen)
  {
    Serialization::FixedBufferDeserializer fbd(data, datalen);
    ByFieldMicroOp<NT::N,T,FT> *uop = new ByFieldMicroOp<NT::N,T,FT>(args->sender,
								     args->async_microop,
								     fbd);
    uop->dispatch(args->operation, false /*not ok to run in this thread*/);
  }

  template <typename NT, typename T, typename N2T, typename T2>
  inline /*static*/ void RemoteMicroOpMessage::ImageDecoder::demux(const RequestArgs *args,
								   const void *data,
								   size_t datalen)
  {
    Serialization::FixedBufferDeserializer fbd(data, datalen);
    ImageMicroOp<NT::N,T,N2T::N,T2> *uop = new ImageMicroOp<NT::N,T,N2T::N,T2>(args->sender,
									       args->async_microop,
									       fbd);
    uop->dispatch(args->operation, false /*not ok to run in this thread*/);
  }

  template <typename NT, typename T, typename N2T, typename T2>
  inline /*static*/ void RemoteMicroOpMessage::PreimageDecoder::demux(const RequestArgs *args,
								      const void *data,
								      size_t datalen)
  {
    Serialization::FixedBufferDeserializer fbd(data, datalen);
    PreimageMicroOp<NT::N,T,N2T::N,T2> *uop = new PreimageMicroOp<NT::N,T,N2T::N,T2>(args->sender,
										     args->async_microop,
										     fbd);
    uop->dispatch(args->operation, false /*not ok to run in this thread*/);
  }

  template <typename NT, typename T>
  inline /*static*/ void RemoteMicroOpMessage::UnionDecoder::demux(const RequestArgs *args,
								   const void *data,
								   size_t datalen)
  {
    Serialization::FixedBufferDeserializer fbd(data, datalen);
    UnionMicroOp<NT::N,T> *uop = new UnionMicroOp<NT::N,T>(args->sender,
							   args->async_microop,
							   fbd);
    uop->dispatch(args->operation, false /*not ok to run in this thread*/);
  }

  template <typename NT, typename T>
  inline /*static*/ void RemoteMicroOpMessage::IntersectionDecoder::demux(const RequestArgs *args,
								   const void *data,
								   size_t datalen)
  {
    Serialization::FixedBufferDeserializer fbd(data, datalen);
    IntersectionMicroOp<NT::N,T> *uop = new IntersectionMicroOp<NT::N,T>(args->sender,
									 args->async_microop,
									 fbd);
    uop->dispatch(args->operation, false /*not ok to run in this thread*/);
  }

  template <typename NT, typename T>
  inline /*static*/ void RemoteMicroOpMessage::DifferenceDecoder::demux(const RequestArgs *args,
								   const void *data,
								   size_t datalen)
  {
    Serialization::FixedBufferDeserializer fbd(data, datalen);
    DifferenceMicroOp<NT::N,T> *uop = new DifferenceMicroOp<NT::N,T>(args->sender,
								     args->async_microop,
								     fbd);
    uop->dispatch(args->operation, false /*not ok to run in this thread*/);
  }

  /*static*/ void RemoteMicroOpMessage::handle_request(RequestArgs args,
						       const void *data, size_t datalen)
  {
    log_part.info() << "received remote micro op message: tag=" 
		    << std::hex << args.type_tag << std::dec
		    << " opcode=" << args.opcode;

    // switch on the opcode first, since they use different numbers of template arguments
    switch(args.opcode) {
    case PartitioningMicroOp::UOPCODE_BY_FIELD:
      {
	NTF_TemplateHelper::demux<ByFieldDecoder>(args.type_tag, &args, data, datalen);
	break;
      }
    case PartitioningMicroOp::UOPCODE_IMAGE:
      {
	NTNT_TemplateHelper::demux<ImageDecoder>(args.type_tag, &args, data, datalen);
	break;
      }
    case PartitioningMicroOp::UOPCODE_PREIMAGE:
      {
	NTNT_TemplateHelper::demux<PreimageDecoder>(args.type_tag, &args, data, datalen);
	break;
      }
    case PartitioningMicroOp::UOPCODE_UNION:
      {
	NT_TemplateHelper::demux<UnionDecoder>(args.type_tag, &args, data, datalen);
	break;
      }
    case PartitioningMicroOp::UOPCODE_INTERSECTION:
      {
	NT_TemplateHelper::demux<IntersectionDecoder>(args.type_tag, &args, data, datalen);
	break;
      }
    case PartitioningMicroOp::UOPCODE_DIFFERENCE:
      {
	NT_TemplateHelper::demux<DifferenceDecoder>(args.type_tag, &args, data, datalen);
	break;
      }
    default:
      assert(0);
    }
  }
  
#if 0
  template <typename T>
  /*static*/ void RemoteMicroOpMessage::send_request(NodeID target, 
						     PartitioningOperation *operation,
						     const T& microop)
  {
    RequestArgs args;

    args.sender = my_node_id;
    args.type_tag = T::type_tag();
    args.opcode = T::OPCODE;
    args.operation = operation;
    args.async_microop = microop.async_microop;

    Serialization::DynamicBufferSerializer dbs(256);
    microop.serialize_params(dbs);
    ByteArray b = dbs.detach_bytearray();

    Message::request(target, args, b.base(), b.size(), PAYLOAD_FREE);
    b.detach();
  }
#endif

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteMicroOpCompleteMessage

  struct RequestArgs {
    AsyncMicroOp *async_microop;
  };
  
  /*static*/ void RemoteMicroOpCompleteMessage::handle_request(RequestArgs args)
  {
    log_part.info() << "received remote micro op complete message: " << args.async_microop;
    args.async_microop->mark_finished(true /*successful*/);
  }

  /*static*/ void RemoteMicroOpCompleteMessage::send_request(NodeID target,
							     AsyncMicroOp *async_microop)
  {
    RequestArgs args;
    args.async_microop = async_microop;
    Message::request(target, args);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningOperation

  class DeferredPartitioningOp : public EventWaiter {
  public:
    DeferredPartitioningOp(PartitioningOperation *_op) : op(_op) {}

    virtual bool event_triggered(Event e, bool poisoned)
    {
      assert(!poisoned); // TODO: POISON_FIXME
      op_queue->enqueue_partitioning_operation(op);
      return true;
    }

    virtual void print(std::ostream& os) const
    {
      os << "DeferredPartitioningOp(" << (void *)op << ")";
    }

    virtual Event get_finish_event(void) const
    {
      return op->get_finish_event();
    }

  protected:
    PartitioningOperation *op;
  };

  PartitioningOperation::PartitioningOperation(const ProfilingRequestSet &reqs,
					       Event _finish_event)
    : Operation(_finish_event, reqs)
  {}

  void PartitioningOperation::deferred_launch(Event wait_for)
  {
    if(wait_for.has_triggered())
      op_queue->enqueue_partitioning_operation(this);
    else
      EventImpl::add_waiter(wait_for, new DeferredPartitioningOp(this));
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

    cp.add_option_int("-dp:workers", DeppartConfig::cfg_num_partitioning_workers);
    cp.add_option_bool("-dp:noisectopt", DeppartConfig::cfg_disable_intersection_optimization);

    cp.parse_command_line(cmdline);
  }

  /*static*/ void PartitioningOpQueue::start_worker_threads(CoreReservationSet& crs)
  {
    assert(op_queue == 0);
    CoreReservation *rsrv = new CoreReservation("partitioning", crs,
						CoreReservationParameters());
    op_queue = new PartitioningOpQueue(rsrv);
    ThreadLaunchParameters tlp;
    for(int i = 0; i < DeppartConfig::cfg_num_partitioning_workers; i++) {
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
	if(!op && !shutdown_flag) {
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
	switch(priority) {
	case OPERATION_PRIORITY:
	  {
	    PartitioningOperation *p_op = static_cast<PartitioningOperation *>(op);
	    log_part.info() << "worker " << this << " starting op " << p_op;
	    p_op->mark_started();
	    p_op->execute();
	    log_part.info() << "worker " << this << " finished op " << p_op;
	    p_op->mark_finished(true /*successful*/);
	    break;
	  }
	case MICROOP_PRIORITY:
	  {
	    PartitioningMicroOp *p_uop = static_cast<PartitioningMicroOp *>(op);
	    log_part.info() << "worker " << this << " starting uop " << p_uop;
	    p_uop->mark_started();
	    p_uop->execute();
	    log_part.info() << "worker " << this << " finished uop " << p_uop;
	    p_uop->mark_finished();
	    break;
	  }
	default: assert(0);
	}
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

