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

// preimage operations for Realm dependent partitioning

#include "realm/deppart/preimage.h"

#include "realm/deppart/deppart_config.h"
#include "realm/deppart/rectlist.h"
#include "realm/deppart/inst_helper.h"
#include "realm/deppart/image.h"
#include "realm/logging.h"

namespace Realm {

  extern Logger log_part;
  extern Logger log_uop_timing;


  template <int N, typename T>
  template <int N2, typename T2>
  Event IndexSpace<N,T>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,Point<N2,T2> > >& field_data,
						       const std::vector<IndexSpace<N2,T2> >& targets,
						       std::vector<IndexSpace<N,T> >& preimages,
						       const ProfilingRequestSet &reqs,
						       Event wait_on /*= Event::NO_EVENT*/) const
  {
    // output vector should start out empty
    assert(preimages.empty());

    Event e = GenEventImpl::create_genevent()->current_event();
    PreimageOperation<N,T,N2,T2> *op = new PreimageOperation<N,T,N2,T2>(*this, field_data, reqs, e);

    size_t n = targets.size();
    preimages.resize(n);
    for(size_t i = 0; i < n; i++) {
      preimages[i] = op->add_target(targets[i]);
      log_dpops.info() << "preimage: " << *this << " tgt=" << targets[i] << " -> " << preimages[i] << " (" << e << ")";
    }

    op->deferred_launch(wait_on);
    return e;
  }

  template <int N, typename T>
  template <int N2, typename T2>
  __attribute__ ((noinline))
  Event IndexSpace<N,T>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,Rect<N2,T2> > >& field_data,
						       const std::vector<IndexSpace<N2,T2> >& targets,
						       std::vector<IndexSpace<N,T> >& preimages,
						       const ProfilingRequestSet &reqs,
						       Event wait_on /*= Event::NO_EVENT*/) const
  {
    // output vector should start out empty
    assert(preimages.empty());

    Event e = GenEventImpl::create_genevent()->current_event();
    PreimageOperation<N,T,N2,T2> *op = new PreimageOperation<N,T,N2,T2>(*this, field_data, reqs, e);

    size_t n = targets.size();
    preimages.resize(n);
    for(size_t i = 0; i < n; i++) {
      preimages[i] = op->add_target(targets[i]);
      log_dpops.info() << "preimage: " << *this << " tgt=" << targets[i] << " -> " << preimages[i] << " (" << e << ")";
    }

    op->deferred_launch(wait_on);
    return e;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PreimageMicroOp<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  inline /*static*/ DynamicTemplates::TagType PreimageMicroOp<N,T,N2,T2>::type_tag(void)
  {
    return NTNT_TemplateHelper::encode_tag<N,T,N2,T2>();
  }

  template <int N, typename T, int N2, typename T2>
  PreimageMicroOp<N,T,N2,T2>::PreimageMicroOp(IndexSpace<N,T> _parent_space,
					 IndexSpace<N,T> _inst_space,
					 RegionInstance _inst,
					 size_t _field_offset,
					 bool _is_ranged)
    : parent_space(_parent_space)
    , inst_space(_inst_space)
    , inst(_inst)
    , field_offset(_field_offset)
    , is_ranged(_is_ranged)
  {}

  template <int N, typename T, int N2, typename T2>
  PreimageMicroOp<N,T,N2,T2>::~PreimageMicroOp(void)
  {}

  template <int N, typename T, int N2, typename T2>
  void PreimageMicroOp<N,T,N2,T2>::add_sparsity_output(IndexSpace<N2,T2> _target,
						       SparsityMap<N,T> _sparsity)
  {
    targets.push_back(_target);
    sparsity_outputs.push_back(_sparsity);
  }

  template <int N, typename T, int N2, typename T2>
  template <typename BM>
  void PreimageMicroOp<N,T,N2,T2>::populate_bitmasks_ptrs(std::map<int, BM *>& bitmasks)
  {
    // for now, one access for the whole instance
    AffineAccessor<Point<N2,T2>,N,T> a_data(inst, field_offset);

    // double iteration - use the instance's space first, since it's probably smaller
    for(IndexSpaceIterator<N,T> it(inst_space); it.valid; it.step()) {
      for(IndexSpaceIterator<N,T> it2(parent_space, it.rect); it2.valid; it2.step()) {
	// now iterate over each point
	for(PointInRectIterator<N,T> pir(it2.rect); pir.valid; pir.step()) {
	  // fetch the pointer and test it against every possible target (ugh)
	  Point<N2,T2> ptr = a_data.read(pir.p);

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
  template <typename BM>
  void PreimageMicroOp<N,T,N2,T2>::populate_bitmasks_ranges(std::map<int, BM *>& bitmasks)
  {
    // for now, one access for the whole instance
    AffineAccessor<Rect<N2,T2>,N,T> a_data(inst, field_offset);

    // double iteration - use the instance's space first, since it's probably smaller
    for(IndexSpaceIterator<N,T> it(inst_space); it.valid; it.step()) {
      for(IndexSpaceIterator<N,T> it2(parent_space, it.rect); it2.valid; it2.step()) {
	// now iterate over each point
	for(PointInRectIterator<N,T> pir(it2.rect); pir.valid; pir.step()) {
	  // fetch the pointer and test it against every possible target (ugh)
	  Rect<N2,T2> rng = a_data.read(pir.p);

	  for(size_t i = 0; i < targets.size(); i++)
	    if(targets[i].contains_any(rng)) {
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

    if(is_ranged)
      populate_bitmasks_ranges(rect_map);
    else
      populate_bitmasks_ptrs(rect_map);

#ifdef DEBUG_PARTITIONING
    std::cout << rect_map.size() << " non-empty preimages present in instance " << inst << std::endl;
    for(typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it = rect_map.begin();
	it != rect_map.end();
	it++)
      std::cout << "  " << targets[it->first] << " = " << it->second->rects.size() << " rectangles" << std::endl;
#endif

    // iterate over sparsity outputs and contribute to all (even if we didn't have any
    //  points found for it)
    int empty_count = 0;
    for(size_t i = 0; i < sparsity_outputs.size(); i++) {
      SparsityMapImpl<N,T> *impl = SparsityMapImpl<N,T>::lookup(sparsity_outputs[i]);
      typename std::map<int, DenseRectangleList<N,T> *>::const_iterator it2 = rect_map.find(i);
      if(it2 != rect_map.end()) {
	impl->contribute_dense_rect_list(it2->second->rects);
	delete it2->second;
      } else {
	impl->contribute_nothing();
	empty_count++;
      }
    }
    if(empty_count > 0)
      log_part.info() << empty_count << " empty preimages (out of " << sparsity_outputs.size() << ")";
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageMicroOp<N,T,N2,T2>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // a PreimageMicroOp should always be executed on whichever node the field data lives
    NodeID exec_node = ID(inst).sparsity.creator_node;

    if(exec_node != my_node_id) {
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
	bool registered = SparsityMapImpl<N2,T2>::lookup(targets[i].sparsity)->add_waiter(this, true /*precise*/);
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
	   (s << is_ranged) &&
	   (s << targets) &&
	   (s << sparsity_outputs));
  }

  template <int N, typename T, int N2, typename T2>
  template <typename S>
  PreimageMicroOp<N,T,N2,T2>::PreimageMicroOp(NodeID _requestor,
					      AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> parent_space) &&
	       (s >> inst_space) &&
	       (s >> inst) &&
	       (s >> field_offset) &&
	       (s >> is_ranged) &&
	       (s >> targets) &&
	       (s >> sparsity_outputs));
    assert(ok);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PreimageOperation<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  PreimageOperation<N,T,N2,T2>::PreimageOperation(const IndexSpace<N,T>& _parent,
						  const std::vector<FieldDataDescriptor<IndexSpace<N,T>,Point<N2,T2> > >& _field_data,
						  const ProfilingRequestSet &reqs,
						  Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
    , parent(_parent)
    , ptr_data(_field_data)
    , overlap_tester(0)
    , dummy_overlap_uop(0)
  {}

  template <int N, typename T, int N2, typename T2>
  PreimageOperation<N,T,N2,T2>::PreimageOperation(const IndexSpace<N,T>& _parent,
						  const std::vector<FieldDataDescriptor<IndexSpace<N,T>,Rect<N2,T2> > >& _field_data,
						  const ProfilingRequestSet &reqs,
						  Event _finish_event)
    : PartitioningOperation(reqs, _finish_event)
    , parent(_parent)
    , range_data(_field_data)
    , overlap_tester(0)
    , dummy_overlap_uop(0)
  {}

  template <int N, typename T, int N2, typename T2>
  PreimageOperation<N,T,N2,T2>::~PreimageOperation(void)
  {}

  template <int N, typename T, int N2, typename T2>
  IndexSpace<N,T> PreimageOperation<N,T,N2,T2>::add_target(const IndexSpace<N2,T2>& target)
  {
    // try to filter out obviously empty targets
    if(parent.empty() || target.empty())
      return IndexSpace<N,T>::make_empty();

    // otherwise it'll be something smaller than the current parent
    IndexSpace<N,T> preimage;
    preimage.bounds = parent.bounds;

    // if the target has a sparsity map, use the same node - otherwise
    // get a sparsity ID by round-robin'ing across the nodes that have field data
    int target_node;
    if(!target.dense())
      target_node = ID(target.sparsity).sparsity.creator_node;
    else
      if(!ptr_data.empty())
	target_node = ID(ptr_data[targets.size() % ptr_data.size()].inst).sparsity.creator_node;
      else
	target_node = ID(range_data[targets.size() % range_data.size()].inst).sparsity.creator_node;
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    preimage.sparsity = sparsity;

    targets.push_back(target);
    preimages.push_back(sparsity);

    return preimage;
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageOperation<N,T,N2,T2>::execute(void)
  {
    if(!DeppartConfig::cfg_disable_intersection_optimization) {
      // build the overlap tester based on the targets, since they're at least known
      ComputeOverlapMicroOp<N2,T2> *uop = new ComputeOverlapMicroOp<N2,T2>(this);

      remaining_sparse_images = ptr_data.size() + range_data.size();
      contrib_counts.resize(preimages.size(), 0);

      // create a dummy async microop that lives until we've received all the sparse images
      dummy_overlap_uop = new AsyncMicroOp(this, 0);
      add_async_work_item(dummy_overlap_uop);

      // add each target, but also generate a bounding box for all of them
      Rect<N2,T2> target_bbox;
      for(size_t i = 0; i < targets.size(); i++) {
	uop->add_input_space(targets[i]);
	if(i == 0)
	  target_bbox = targets[i].bounds;
	else
	  target_bbox = target_bbox.union_bbox(targets[i].bounds);
      }

      for(size_t i = 0; i < ptr_data.size(); i++) {
	// in parallel, we will request the approximate images of each instance's
	//  data (ideally limited to the target_bbox)
	ImageMicroOp<N2,T2,N,T> *img = new ImageMicroOp<N2,T2,N,T>(target_bbox,
								   ptr_data[i].index_space,
								   ptr_data[i].inst,
								   ptr_data[i].field_offset,
								   false /*ptrs*/);
	img->add_approx_output(i, this);
	img->dispatch(this, false /* do not run in this thread */);
      }

      for(size_t i = 0; i < range_data.size(); i++) {
	// in parallel, we will request the approximate images of each instance's
	//  data (ideally limited to the target_bbox)
	ImageMicroOp<N2,T2,N,T> *img = new ImageMicroOp<N2,T2,N,T>(target_bbox,
								   range_data[i].index_space,
								   range_data[i].inst,
								   range_data[i].field_offset,
								   true /*ranges*/);
	img->add_approx_output(i + ptr_data.size(), this);
	img->dispatch(this, false /* do not run in this thread */);
      }

      uop->dispatch(this, true /* ok to run in this thread */);
    } else {
      for(size_t i = 0; i < preimages.size(); i++)
	SparsityMapImpl<N,T>::lookup(preimages[i])->set_contributor_count(ptr_data.size() +
									  range_data.size());

      for(size_t i = 0; i < ptr_data.size(); i++) {
	PreimageMicroOp<N,T,N2,T2> *uop = new PreimageMicroOp<N,T,N2,T2>(parent,
									 ptr_data[i].index_space,
									 ptr_data[i].inst,
									 ptr_data[i].field_offset,
									 false /*ptrs*/);
	for(size_t j = 0; j < targets.size(); j++)
	  uop->add_sparsity_output(targets[j], preimages[j]);
	uop->dispatch(this, true /* ok to run in this thread */);
      }

      for(size_t i = 0; i < range_data.size(); i++) {
	PreimageMicroOp<N,T,N2,T2> *uop = new PreimageMicroOp<N,T,N2,T2>(parent,
									 range_data[i].index_space,
									 range_data[i].inst,
									 range_data[i].field_offset,
									 true /*ranges*/);
	for(size_t j = 0; j < targets.size(); j++)
	  uop->add_sparsity_output(targets[j], preimages[j]);
	uop->dispatch(this, true /* ok to run in this thread */);
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageOperation<N,T,N2,T2>::provide_sparse_image(int index, const Rect<N2,T2> *rects, size_t count)
  {
    // atomically check the overlap tester's readiness and queue us if not
    bool tester_ready = false;
    {
      AutoHSLLock al(mutex);
      if(overlap_tester != 0) {
	tester_ready = true;
      } else {
	std::vector<Rect<N2,T2> >& r = pending_sparse_images[index];
	r.insert(r.end(), rects, rects + count);
      }
    }

    if(tester_ready) {
      // see which of the targets this image overlaps
      std::set<int> overlaps;
      overlap_tester->test_overlap(rects, count, overlaps);
      if((size_t)index < ptr_data.size()) {
	log_part.info() << "image of ptr_data[" << index << "] overlaps " << overlaps.size() << " targets";
	PreimageMicroOp<N,T,N2,T2> *uop = new PreimageMicroOp<N,T,N2,T2>(parent,
									 ptr_data[index].index_space,
									 ptr_data[index].inst,
									 ptr_data[index].field_offset,
									 false /*ptrs*/);
	for(std::set<int>::const_iterator it2 = overlaps.begin();
	    it2 != overlaps.end();
	    it2++) {
	  int j = *it2;
	  __sync_fetch_and_add(&contrib_counts[j], 1);
	  uop->add_sparsity_output(targets[j], preimages[j]);
	}
	uop->dispatch(this, false /* do not run in this thread */);
      } else {
	size_t rel_index = index - ptr_data.size();
	assert(rel_index < range_data.size());
	log_part.info() << "image of range_data[" << rel_index << "] overlaps " << overlaps.size() << " targets";
	PreimageMicroOp<N,T,N2,T2> *uop = new PreimageMicroOp<N,T,N2,T2>(parent,
									 range_data[rel_index].index_space,
									 range_data[rel_index].inst,
									 range_data[rel_index].field_offset,
									 true /*ranges*/);
	for(std::set<int>::const_iterator it2 = overlaps.begin();
	    it2 != overlaps.end();
	    it2++) {
	  int j = *it2;
	  __sync_fetch_and_add(&contrib_counts[j], 1);
	  uop->add_sparsity_output(targets[j], preimages[j]);
	}
	uop->dispatch(this, false /* do not run in this thread */);
      }

      // if these were the last sparse images, we can now set the contributor counts
      int v = __sync_sub_and_fetch(&remaining_sparse_images, 1);
      if(v == 0) {
	for(size_t j = 0; j < preimages.size(); j++) {
	  log_part.info() << contrib_counts[j] << " total contributors to preimage " << j;
	  SparsityMapImpl<N,T>::lookup(preimages[j])->set_contributor_count(contrib_counts[j]);
	}
	dummy_overlap_uop->mark_finished(true /*successful*/);
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageOperation<N,T,N2,T2>::set_overlap_tester(void *tester)
  {
    // atomically set the overlap tester and see if there are any pending entries
    std::map<int, std::vector<Rect<N2,T2> > > pending;
    {
      AutoHSLLock al(mutex);
      assert(overlap_tester == 0);
      overlap_tester = static_cast<OverlapTester<N2,T2> *>(tester);
      pending.swap(pending_sparse_images);
    }

    // now issue work for any sparse images we got before the tester was ready
    if(!pending.empty()) {
      for(typename std::map<int, std::vector<Rect<N2,T2> > >::const_iterator it = pending.begin();
	  it != pending.end();
	  it++) {
	// see which instance this is an image from
	size_t idx = it->first;
	// see which of the targets that image overlaps
	std::set<int> overlaps;
	overlap_tester->test_overlap(&it->second[0], it->second.size(), overlaps);
	if(idx < ptr_data.size()) {
	  log_part.info() << "image of ptr_data[" << idx << "] overlaps " << overlaps.size() << " targets";
	  PreimageMicroOp<N,T,N2,T2> *uop = new PreimageMicroOp<N,T,N2,T2>(parent,
									   ptr_data[idx].index_space,
									   ptr_data[idx].inst,
									   ptr_data[idx].field_offset,
									   false /*ptrs*/);
	  for(std::set<int>::const_iterator it2 = overlaps.begin();
	      it2 != overlaps.end();
	      it2++) {
	    int j = *it2;
	    __sync_fetch_and_add(&contrib_counts[j], 1);
	    uop->add_sparsity_output(targets[j], preimages[j]);
	  }
	  uop->dispatch(this, true /* ok to run in this thread */);
	} else {
	  size_t rel_index = idx - ptr_data.size();
	  assert(rel_index < range_data.size());
	  log_part.info() << "image of range_data[" << rel_index << "] overlaps " << overlaps.size() << " targets";
	  PreimageMicroOp<N,T,N2,T2> *uop = new PreimageMicroOp<N,T,N2,T2>(parent,
									   range_data[rel_index].index_space,
									   range_data[rel_index].inst,
									   range_data[rel_index].field_offset,
									   true /*ranges*/);
	  for(std::set<int>::const_iterator it2 = overlaps.begin();
	      it2 != overlaps.end();
	      it2++) {
	    int j = *it2;
	    __sync_fetch_and_add(&contrib_counts[j], 1);
	    uop->add_sparsity_output(targets[j], preimages[j]);
	  }
	  uop->dispatch(this, true /* ok to run in this thread */);
	}
      }

      // if these were the last sparse images, we can now set the contributor counts
      int v = __sync_sub_and_fetch(&remaining_sparse_images, pending.size());
      if(v == 0) {
	for(size_t j = 0; j < preimages.size(); j++) {
	  log_part.info() << contrib_counts[j] << " total contributors to preimage " << j;
	  SparsityMapImpl<N,T>::lookup(preimages[j])->set_contributor_count(contrib_counts[j]);
	}
	dummy_overlap_uop->mark_finished(true /*successful*/);
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageOperation<N,T,N2,T2>::print(std::ostream& os) const
  {
    os << "PreimageOperation(" << parent << ")";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ApproxImageResponseMessage
  
  template <typename NT, typename T, typename N2T, typename T2>
  inline /*static*/ void ApproxImageResponseMessage::DecodeHelper::demux(const RequestArgs *args,
									 const void *data, size_t datalen)
  {
    PreimageOperation<NT::N,T,N2T::N,T2> *op = reinterpret_cast<PreimageOperation<NT::N,T,N2T::N,T2> *>(args->approx_output_op);
    op->provide_sparse_image(args->approx_output_index,
			     static_cast<const Rect<N2T::N,T2> *>(data),
			     datalen / sizeof(Rect<N2T::N,T2>));
  }

  /*static*/ void ApproxImageResponseMessage::handle_request(RequestArgs args,
							     const void *data, size_t datalen)
  {
    log_part.info() << "received approx image response: tag=" << std::hex << args.type_tag << std::dec
		    << " op=" << args.approx_output_op;

    NTNT_TemplateHelper::demux<DecodeHelper>(args.type_tag, &args, data, datalen);
  }

#if 0  
  template <int N, typename T, int N2, typename T2>
  /*static*/ void ApproxImageResponseMessage::send_request(NodeID target, 
							   intptr_t output_op, int output_index,
							   const Rect<N2,T2> *rects, size_t count)
  {
    RequestArgs args;

    args.type_tag = NTNT_TemplateHelper::encode_tag<N,T,N2,T2>();
    args.approx_output_op = output_op;
    args.approx_output_index = output_index;

    Message::request(target, args, rects, count * sizeof(Rect<N2,T2>), PAYLOAD_COPY);
  }
#endif


#define DOIT(N1,T1,N2,T2) \
  template class PreimageMicroOp<N1,T1,N2,T2>; \
  template class PreimageOperation<N1,T1,N2,T2>; \
  template PreimageMicroOp<N1,T1,N2,T2>::PreimageMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template Event IndexSpace<N1,T1>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N1,T1>,Point<N2,T2> > >&, \
								  const std::vector<IndexSpace<N2,T2> >&, \
								  std::vector<IndexSpace<N1,T1> >&, \
								  const ProfilingRequestSet &, \
								  Event) const; \
  template Event IndexSpace<N1,T1>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N1,T1>,Rect<N2,T2> > >&, \
								  const std::vector<IndexSpace<N2,T2> >&, \
								  std::vector<IndexSpace<N1,T1> >&, \
								  const ProfilingRequestSet &, \
								  Event) const;

  FOREACH_NTNT(DOIT)

}; // namespace Realm
