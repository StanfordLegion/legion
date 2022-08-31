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

// preimage operations for Realm dependent partitioning

#include "realm/deppart/preimage.h"

#include "realm/deppart/deppart_config.h"
#include "realm/deppart/rectlist.h"
#include "realm/deppart/inst_helper.h"
#include "realm/deppart/image.h"
#include "realm/logging.h"
#include <ctime>

namespace Realm {

  extern Logger log_part;
  extern Logger log_uop_timing;

  template <int N, typename T>
  template <int N2, typename T2>
  Event IndexSpace<N, T>::create_subspaces_by_preimage(
      const DomainTransform<N2, T2, N, T> &domain_transform,
      const std::vector<IndexSpace<N2, T2> > &targets,
      std::vector<IndexSpace<N, T> > &preimages,
      const ProfilingRequestSet &reqs,
      Event wait_on /*= Event::NO_EVENT*/) const {
   // output vector should start out empty
   assert(preimages.empty());

   GenEventImpl *finish_event = GenEventImpl::create_genevent();
   Event e = finish_event->current_event();
   PreimageOperation<N, T, N2, T2> *op = new PreimageOperation<N, T, N2, T2>(
       *this, domain_transform, reqs, finish_event, ID(e).event_generation());

   size_t n = targets.size();
   preimages.resize(n);
   for (size_t i = 0; i < n; i++) {
    preimages[i] = op->add_target(targets[i]);
    log_dpops.info() << "preimage: " << *this << " tgt=" << targets[i] << " -> "
                     << preimages[i] << " (" << e << ")";
   }

   op->launch(wait_on);
   return e;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class PreimageMicroOp<N,T,N2,T2>

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
  {
    areg.force_instantiation();
  }

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
	impl->contribute_dense_rect_list(it2->second->rects, true /*disjoint*/);
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
    NodeID exec_node = ID(inst).instance_owner_node();

    if(exec_node != Network::my_node_id) {
      forward_microop<PreimageMicroOp<N,T,N2,T2> >(exec_node, op, this);
      return;
    }

    // Need valid data for the instance space
    if (!inst_space.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N,T>::lookup(inst_space.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
        wait_count.fetch_add(1);
    }

    // need valid data for each target
    for(size_t i = 0; i < targets.size(); i++) {
      if(!targets[i].dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N2,T2>::lookup(targets[i].sparsity)->add_waiter(this, true /*precise*/);
	if(registered)
	  wait_count.fetch_add(1);
      }
    }

    // need valid data for the parent space too
    if(!parent_space.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N,T>::lookup(parent_space.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
	wait_count.fetch_add(1);
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
    (void)ok;
  }

  template <int N, typename T, int N2, typename T2>
  ActiveMessageHandlerReg<RemoteMicroOpMessage<PreimageMicroOp<N,T,N2,T2> > > PreimageMicroOp<N,T,N2,T2>::areg;


  ////////////////////////////////////////////////////////////////////////
  //
  // class PreimageOperation<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  PreimageOperation<N, T, N2, T2>::PreimageOperation(
      const IndexSpace<N, T> &_parent,
      const DomainTransform<N2, T2, N, T> &_domain_transform,
      const ProfilingRequestSet &reqs, GenEventImpl *_finish_event,
      EventImpl::gen_t _finish_gen)
      : PartitioningOperation(reqs, _finish_event, _finish_gen),
        parent(_parent),
        domain_transform(_domain_transform),
        overlap_tester(0),
        dummy_overlap_uop(0) {
   areg.force_instantiation();
  }

  template <int N, typename T, int N2, typename T2>
  PreimageOperation<N,T,N2,T2>::~PreimageOperation(void)
  {
    if(overlap_tester)
      delete overlap_tester;
  }

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
      target_node = ID(target.sparsity).sparsity_creator_node();
    else if (!domain_transform.ptr_data.empty())
     target_node =
         ID(domain_transform
                .ptr_data[targets.size() % domain_transform.ptr_data.size()]
                .inst)
             .instance_owner_node();
    else
     target_node =
         ID(domain_transform
                .range_data[targets.size() % domain_transform.range_data.size()]
                .inst)
             .instance_owner_node();
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    preimage.sparsity = sparsity;

    targets.push_back(target);
    preimages.push_back(sparsity);

    return preimage;
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageOperation<N, T, N2, T2>::execute(void) {
   if (domain_transform.type ==
       DomainTransform<N2, T2, N, T>::DomainTransformType::STRUCTURED) {
    for (size_t i = 0; i < preimages.size(); i++) {
     SparsityMapImpl<N, T>::lookup(preimages[i])->set_contributor_count(1);
    }

    StructuredPreimageMicroOp<N, T, N2, T2> *micro_op =
        new StructuredPreimageMicroOp<N, T, N2, T2>(
            domain_transform.structured_transform, parent);

    for (size_t j = 0; j < targets.size(); j++) {
     micro_op->add_sparsity_output(targets[j], preimages[j]);
    }
    micro_op->dispatch(this, true);
   } else {
    if (!DeppartConfig::cfg_disable_intersection_optimization) {
     // build the overlap tester based on the targets, since they're at least
     // known
     ComputeOverlapMicroOp<N2, T2> *uop =
         new ComputeOverlapMicroOp<N2, T2>(this);

     remaining_sparse_images.store(domain_transform.ptr_data.size() +
                                   domain_transform.range_data.size());
     contrib_counts.resize(preimages.size(), atomic<int>(0));

     // create a dummy async microop that lives until we've received all the
     // sparse images
     dummy_overlap_uop = new AsyncMicroOp(this, 0);
     add_async_work_item(dummy_overlap_uop);

     // add each target, but also generate a bounding box for all of them
     Rect<N2, T2> target_bbox;
     for (size_t i = 0; i < targets.size(); i++) {
      uop->add_input_space(targets[i]);
      if (i == 0)
       target_bbox = targets[i].bounds;
      else
       target_bbox = target_bbox.union_bbox(targets[i].bounds);
     }

     for (size_t i = 0; i < domain_transform.ptr_data.size(); i++) {
      // in parallel, we will request the approximate images of each instance's
      //  data (ideally limited to the target_bbox)
      ImageMicroOp<N2, T2, N, T> *img = new ImageMicroOp<N2, T2, N, T>(
          target_bbox, domain_transform.ptr_data[i].index_space,
          domain_transform.ptr_data[i].inst,
          domain_transform.ptr_data[i].field_offset, false /*ptrs*/);
      img->add_approx_output(i, this);
      img->dispatch(this, false /* do not run in this thread */);
     }

     for (size_t i = 0; i < domain_transform.range_data.size(); i++) {
      // in parallel, we will request the approximate images of each instance's
      //  data (ideally limited to the target_bbox)
      ImageMicroOp<N2, T2, N, T> *img = new ImageMicroOp<N2, T2, N, T>(
          target_bbox, domain_transform.range_data[i].index_space,
          domain_transform.range_data[i].inst,
          domain_transform.range_data[i].field_offset, true /*ranges*/);
      img->add_approx_output(i + domain_transform.ptr_data.size(), this);
      img->dispatch(this, false /* do not run in this thread */);
     }

     uop->dispatch(this, true /* ok to run in this thread */);
    } else {
     for (size_t i = 0; i < preimages.size(); i++)
      SparsityMapImpl<N, T>::lookup(preimages[i])
          ->set_contributor_count(domain_transform.ptr_data.size() +
                                  domain_transform.range_data.size());

     for (size_t i = 0; i < domain_transform.ptr_data.size(); i++) {
      PreimageMicroOp<N, T, N2, T2> *uop = new PreimageMicroOp<N, T, N2, T2>(
          parent, domain_transform.ptr_data[i].index_space,
          domain_transform.ptr_data[i].inst,
          domain_transform.ptr_data[i].field_offset, false /*ptrs*/);
      for (size_t j = 0; j < targets.size(); j++)
       uop->add_sparsity_output(targets[j], preimages[j]);
      uop->dispatch(this, true /* ok to run in this thread */);
     }

     for (size_t i = 0; i < domain_transform.range_data.size(); i++) {
      PreimageMicroOp<N, T, N2, T2> *uop = new PreimageMicroOp<N, T, N2, T2>(
          parent, domain_transform.range_data[i].index_space,
          domain_transform.range_data[i].inst,
          domain_transform.range_data[i].field_offset, true /*ranges*/);
      for (size_t j = 0; j < targets.size(); j++)
       uop->add_sparsity_output(targets[j], preimages[j]);
      uop->dispatch(this, true /* ok to run in this thread */);
     }
    }
   }
  }

  template <int N, typename T, int N2, typename T2>
  void PreimageOperation<N,T,N2,T2>::provide_sparse_image(int index, const Rect<N2,T2> *rects, size_t count)
  {
    // atomically check the overlap tester's readiness and queue us if not
    bool tester_ready = false;
    {
      AutoLock<> al(mutex);
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
      if((size_t)index < domain_transform.ptr_data.size()) {
	log_part.info() << "image of ptr_data[" << index << "] overlaps " << overlaps.size() << " targets";
        PreimageMicroOp<N, T, N2, T2> *uop = new PreimageMicroOp<N, T, N2, T2>(
            parent, domain_transform.ptr_data[index].index_space,
            domain_transform.ptr_data[index].inst,
            domain_transform.ptr_data[index].field_offset, false /*ptrs*/);
        for(std::set<int>::const_iterator it2 = overlaps.begin();
	    it2 != overlaps.end();
	    it2++) {
	  int j = *it2;
	  contrib_counts[j].fetch_add(1);
	  uop->add_sparsity_output(targets[j], preimages[j]);
	}
	uop->dispatch(this, false /* do not run in this thread */);
      } else {
	size_t rel_index = index - domain_transform.ptr_data.size();
	assert(rel_index < domain_transform.range_data.size());
	log_part.info() << "image of range_data[" << rel_index << "] overlaps " << overlaps.size() << " targets";
        PreimageMicroOp<N, T, N2, T2> *uop = new PreimageMicroOp<N, T, N2, T2>(
            parent, domain_transform.range_data[rel_index].index_space,
            domain_transform.range_data[rel_index].inst,
            domain_transform.range_data[rel_index].field_offset,
            true /*ranges*/);
        for(std::set<int>::const_iterator it2 = overlaps.begin();
	    it2 != overlaps.end();
	    it2++) {
	  int j = *it2;
	  contrib_counts[j].fetch_add(1);
	  uop->add_sparsity_output(targets[j], preimages[j]);
	}
	uop->dispatch(this, false /* do not run in this thread */);
      }

      // if these were the last sparse images, we can now set the contributor counts
      int v = remaining_sparse_images.fetch_sub(1) - 1;
      if(v == 0) {
	for(size_t j = 0; j < preimages.size(); j++) {
	  log_part.info() << contrib_counts[j].load() << " total contributors to preimage " << j;
	  SparsityMapImpl<N,T>::lookup(preimages[j])->set_contributor_count(contrib_counts[j].load());
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
      AutoLock<> al(mutex);
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
	if(idx < domain_transform.ptr_data.size()) {
	  log_part.info() << "image of ptr_data[" << idx << "] overlaps " << overlaps.size() << " targets";
          PreimageMicroOp<N, T, N2, T2> *uop =
              new PreimageMicroOp<N, T, N2, T2>(
                  parent, domain_transform.ptr_data[idx].index_space,
                  domain_transform.ptr_data[idx].inst,
                  domain_transform.ptr_data[idx].field_offset, false /*ptrs*/);
          for(std::set<int>::const_iterator it2 = overlaps.begin();
	      it2 != overlaps.end();
	      it2++) {
	    int j = *it2;
	    contrib_counts[j].fetch_add(1);
	    uop->add_sparsity_output(targets[j], preimages[j]);
	  }
	  uop->dispatch(this, true /* ok to run in this thread */);
	} else {
	  size_t rel_index = idx - domain_transform.ptr_data.size();
	  assert(rel_index < domain_transform.range_data.size());
	  log_part.info() << "image of range_data[" << rel_index << "] overlaps " << overlaps.size() << " targets";
          PreimageMicroOp<N, T, N2, T2> *uop =
              new PreimageMicroOp<N, T, N2, T2>(
                  parent, domain_transform.range_data[rel_index].index_space,
                  domain_transform.range_data[rel_index].inst,
                  domain_transform.range_data[rel_index].field_offset,
                  true /*ranges*/);
          for(std::set<int>::const_iterator it2 = overlaps.begin();
	      it2 != overlaps.end();
	      it2++) {
	    int j = *it2;
	    contrib_counts[j].fetch_add(1);
	    uop->add_sparsity_output(targets[j], preimages[j]);
	  }
	  uop->dispatch(this, true /* ok to run in this thread */);
	}
      }

      // if these were the last sparse images, we can now set the contributor counts
      int v = remaining_sparse_images.fetch_sub(pending.size()) - pending.size();
      if(v == 0) {
	for(size_t j = 0; j < preimages.size(); j++) {
	  log_part.info() << contrib_counts[j].load() << " total contributors to preimage " << j;
	  SparsityMapImpl<N,T>::lookup(preimages[j])->set_contributor_count(contrib_counts[j].load());
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

  template <int N, typename T, int N2, typename T2>
  ActiveMessageHandlerReg<ApproxImageResponseMessage<PreimageOperation<N,T,N2,T2> > > PreimageOperation<N,T,N2,T2>::areg;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ApproxImageResponseMessage<T>

  template <typename T>
  /*static*/ void ApproxImageResponseMessage<T>::handle_message(NodeID sender,
								const ApproxImageResponseMessage<T> &msg,
								const void *data, size_t datalen)
  {
    T *op = reinterpret_cast<T *>(msg.approx_output_op);
    op->provide_sparse_image(msg.approx_output_index,
			     static_cast<const Rect<T::DIM2, typename T::IDXTYPE2> *>(data),
			     datalen / sizeof(Rect<T::DIM2, typename T::IDXTYPE2>));
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class StructuredPreimageMicroOp<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  StructuredPreimageMicroOp<N, T, N2, T2>::StructuredPreimageMicroOp(
      const StructuredTransform<N2, T2, N, T> &_transform,
      IndexSpace<N, T> _parent_space)
      : transform(_transform), parent_space(_parent_space) {}

  template <int N, typename T, int N2, typename T2>
  StructuredPreimageMicroOp<N, T, N2, T2>::~StructuredPreimageMicroOp(void) {}

  template <int N, typename T, int N2, typename T2>
  void StructuredPreimageMicroOp<N, T, N2, T2>::add_sparsity_output(
      IndexSpace<N2, T2> _target, SparsityMap<N, T> _sparsity) {
   targets.push_back(_target);
   sparsity_outputs.push_back(_sparsity);
  }

  template <int N, typename T, int N2, typename T2>
  template <typename BM>
  void StructuredPreimageMicroOp<N, T, N2, T2>::populate_bitmasks(
      std::map<int, BM *> &bitmasks) {
   Rect<N2, T2> target_bbox = targets[0].bounds;
   for (size_t i = 1; i < targets.size(); i++) {
    target_bbox = target_bbox.union_bbox(targets[i].bounds);
   }
   for (IndexSpaceIterator<N, T> it2(parent_space); it2.valid; it2.step()) {
    Rect<N2, T2> parent_bbox;
    parent_bbox.lo = transform[it2.rect.lo];
    parent_bbox.hi = transform[it2.rect.hi];

    if (target_bbox.intersection(parent_bbox).empty()) continue;

    for (PointInRectIterator<N, T> pir(it2.rect); pir.valid; pir.step()) {
     Point<N2, T2> target_point = transform[pir.p];
     for (size_t i = 0; i < targets.size(); i++) {
      if (targets[i].contains(target_point)) {
       BM *&bmp = bitmasks[i];
       if (!bmp) bmp = new BM;
       bmp->add_point(pir.p);
      }
     }
    }
   }
  }

  template <int N, typename T, int N2, typename T2>
  void StructuredPreimageMicroOp<N,T,N2,T2>::execute(void)
  {
    TimeStamp ts("PreimageMicroOp::execute", true, &log_uop_timing);
    std::map<int, DenseRectangleList<N,T> *> rect_map;

    populate_bitmasks(rect_map);
#ifdef DEBUG_PARTITIONING
    std::cout << rect_map.size() << " non-empty preimages present in instance "
              << inst << std::endl;
    for (typename std::map<int, DenseRectangleList<N, T> *>::const_iterator it =
             rect_map.begin();
         it != rect_map.end(); it++)
      std::cout << "  " << targets[it->first] << " = "
                << it->second->rects.size() << " rectangles" << std::endl;
#endif
    // iterate over sparsity outputs and contribute to all (even if we
    // didn't have any points found for it)
    int empty_count = 0;
    for (size_t i = 0; i < sparsity_outputs.size(); i++) {
      SparsityMapImpl<N, T> *impl =
          SparsityMapImpl<N, T>::lookup(sparsity_outputs[i]);
      typename std::map<int, DenseRectangleList<N, T> *>::const_iterator it2 =
          rect_map.find(i);
      if (it2 != rect_map.end()) {
        impl->contribute_dense_rect_list(it2->second->rects, true /*disjoint*/);
        delete it2->second;
      } else {
        impl->contribute_nothing();
        empty_count++;
      }
    }

    if (empty_count > 0) {
      log_part.info() << empty_count << " empty preimages (out of "
                      << sparsity_outputs.size() << ")";
    }
  }

  template <int N, typename T, int N2, typename T2>
  void StructuredPreimageMicroOp<N, T, N2, T2>::dispatch(
      PartitioningOperation *op, bool inline_ok) {
    // need valid data for each target
    for (size_t i = 0; i < targets.size(); i++) {
      if (!targets[i].dense()) {
        // it's safe to add the count after the registration only because we
        // initialized the count to 2 instead of 1
        bool registered = SparsityMapImpl<N2, T2>::lookup(targets[i].sparsity)
                              ->add_waiter(this, true /*precise*/);
        if (registered) wait_count.fetch_add(1);
      }
    }

    // need valid data for the parent space too
    if (!parent_space.dense()) {
      // it's safe to add the count after the registration only because we
      // initialized the count to 2 instead of 1
      bool registered = SparsityMapImpl<N, T>::lookup(parent_space.sparsity)
                            ->add_waiter(this, true /*precise*/);
      if (registered) wait_count.fetch_add(1);
    }

    finish_dispatch(op, inline_ok);
  }

  // instantiations of templates handled in preimage_tmpl.cc

}; // namespace Realm
