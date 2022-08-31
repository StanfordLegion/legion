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

// image operations for Realm dependent partitioning

#include "realm/deppart/image.h"

#include "realm/deppart/deppart_config.h"
#include "realm/deppart/rectlist.h"
#include "realm/deppart/inst_helper.h"
#include "realm/deppart/preimage.h"
#include "realm/logging.h"

namespace Realm {

  extern Logger log_part;
  extern Logger log_uop_timing;

  template <int N, typename T>
  template <int N2, typename T2>
  Event IndexSpace<N, T>::create_subspaces_by_image(
      const DomainTransform<N, T, N2, T2> &domain_transform,
      const std::vector<IndexSpace<N2, T2>> &sources,
      std::vector<IndexSpace<N, T>> &images, const ProfilingRequestSet &reqs,
      Event wait_on) const {
   // output vector should start out empty
   assert(images.empty());

   GenEventImpl *finish_event = GenEventImpl::create_genevent();
   Event e = finish_event->current_event();

   ImageOperation<N, T, N2, T2> *op = new ImageOperation<N, T, N2, T2>(
       *this, domain_transform, reqs, finish_event, ID(e).event_generation());

   size_t n = sources.size();
   images.resize(n);
   for (size_t i = 0; i < n; i++) {
    images[i] = op->add_source(sources[i]);
    log_dpops.info() << "image: " << *this << " src=" << sources[i] << " -> "
                     << images[i] << " (" << e << ")";
   }

   op->launch(wait_on);
   return e;
  }

  template <int N, typename T>
  template <int N2, typename T2>
  Event IndexSpace<N,T>::create_subspaces_by_image_with_difference(
      const DomainTransform<N, T, N2, T2> &domain_transform,
							   const std::vector<IndexSpace<N2,T2> >& sources,
							   const std::vector<IndexSpace<N,T> >& diff_rhss,
							   std::vector<IndexSpace<N,T> >& images,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // output vector should start out empty
    assert(images.empty());

    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event e = finish_event->current_event();
    ImageOperation<N,T,N2,T2> *op = new ImageOperation<N,T,N2,T2>(*this, domain_transform, reqs, finish_event, ID(e).event_generation());

    size_t n = sources.size();
    images.resize(n);
    for(size_t i = 0; i < n; i++) {
      images[i] = op->add_source_with_difference(sources[i], diff_rhss[i]);
      log_dpops.info() << "image: " << *this << " src=" << sources[i] << " mask=" << diff_rhss[i] << " -> " << images[i] << " (" << e << ")";
    }

    op->launch(wait_on);
    return e;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ImageMicroOp<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  ImageMicroOp<N,T,N2,T2>::ImageMicroOp(IndexSpace<N,T> _parent_space,
					IndexSpace<N2,T2> _inst_space,
					RegionInstance _inst,
					size_t _field_offset,
					bool _is_ranged)
    : parent_space(_parent_space)
    , inst_space(_inst_space)
    , inst(_inst)
    , field_offset(_field_offset)
    , is_ranged(_is_ranged)
    , approx_output_index(-1)
    , approx_output_op(0)
  {
    areg.force_instantiation();
  }

  template <int N, typename T, int N2, typename T2>
  ImageMicroOp<N,T,N2,T2>::~ImageMicroOp(void)
  {}

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::add_sparsity_output(IndexSpace<N2,T2> _source,
						    SparsityMap<N,T> _sparsity)
  {
    sources.push_back(_source);
    sparsity_outputs.push_back(_sparsity);
  }

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::add_sparsity_output_with_difference(IndexSpace<N2,T2> _source,
                                                    IndexSpace<N,T> _diff_rhs,
						    SparsityMap<N,T> _sparsity)
  {
    sources.push_back(_source);
    diff_rhss.push_back(_diff_rhs);
    sparsity_outputs.push_back(_sparsity);
  }

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::add_approx_output(int index, PartitioningOperation *op)
  {
    assert(approx_output_index == -1);
    approx_output_index = index;
    approx_output_op = reinterpret_cast<intptr_t>(op);
  }

  template <int N, typename T, int N2, typename T2>
  template <typename BM>
  void ImageMicroOp<N,T,N2,T2>::populate_bitmasks_ptrs(std::map<int, BM *>& bitmasks)
  {
    // for now, one access for the whole instance
    AffineAccessor<Point<N,T>,N2,T2> a_data(inst, field_offset);

    // double iteration - use the instance's space first, since it's probably smaller
    for(IndexSpaceIterator<N2,T2> it(inst_space); it.valid; it.step()) {
      for(size_t i = 0; i < sources.size(); i++) {
	for(IndexSpaceIterator<N2,T2> it2(sources[i], it.rect); it2.valid; it2.step()) {
	  BM **bmpp = 0;

	  // iterate over each point in the source and see if it points into the parent space	  
	  for(PointInRectIterator<N2,T2> pir(it2.rect); pir.valid; pir.step()) {
	    Point<N,T> ptr = a_data.read(pir.p);

	    if(parent_space.contains(ptr)) {
              // optional filter
              if(!diff_rhss.empty())
                if(diff_rhss[i].contains(ptr)) {
                  //std::cout << "point " << ptr << " filtered!\n";
                  continue;
                }
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
  template <typename BM>
  void ImageMicroOp<N,T,N2,T2>::populate_bitmasks_ranges(std::map<int, BM *>& bitmasks)
  {
    // for now, one access for the whole instance
    AffineAccessor<Rect<N,T>,N2,T2> a_data(inst, field_offset);

    // double iteration - use the instance's space first, since it's probably smaller
    for(IndexSpaceIterator<N2,T2> it(inst_space); it.valid; it.step()) {
      for(size_t i = 0; i < sources.size(); i++) {
	for(IndexSpaceIterator<N2,T2> it2(sources[i], it.rect); it2.valid; it2.step()) {
	  BM **bmpp = 0;

	  // iterate over each point in the source and see if it points into the parent space	  
	  for(PointInRectIterator<N2,T2> pir(it2.rect); pir.valid; pir.step()) {
	    Rect<N,T> rng = a_data.read(pir.p);

	    // precisely test rng against each subrect of the parent space
	    //  by performing restricted iteration
	    for(IndexSpaceIterator<N,T> it3(parent_space, rng);
		it3.valid;
		it3.step()) {
	      // if any of the points in the range overlap the diff_rhs, we'll
	      //  need to test each point individually
	      bool check_diff = (!diff_rhss.empty() &&
				 diff_rhss[i].contains_any(it3.rect));

	      if(check_diff) {
		for(PointInRectIterator<N,T> pir2(it3.rect); pir2.valid; pir2.step())
		  if(!diff_rhss[i].contains(pir2.p)) {
		    if(!bmpp) bmpp = &bitmasks[i];
		    if(!*bmpp) *bmpp = new BM;
		    (*bmpp)->add_point(pir2.p);
		  }
	      } else {
		// add whole rectangle
		if(!bmpp) bmpp = &bitmasks[i];
		if(!*bmpp) *bmpp = new BM;
		(*bmpp)->add_rect(it3.rect);
	      }
	    }
	  }
	}
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  template <typename BM>
  void ImageMicroOp<N,T,N2,T2>::populate_approx_bitmask_ptrs(BM& bitmask)
  {
    // for now, one access for the whole instance
    AffineAccessor<Point<N,T>,N2,T2> a_data(inst, field_offset);
    //std::cout << "a_data = " << a_data << "\n";

    // simple image operation - project ever 
    for(IndexSpaceIterator<N2,T2> it(inst_space); it.valid; it.step()) {
      // iterate over each point in the source and mark what it touches
      for(PointInRectIterator<N2,T2> pir(it.rect); pir.valid; pir.step()) {
	Point<N,T> ptr = a_data.read(pir.p);

	// make sure to check for containment in the parent space
	if(parent_space.contains(ptr))
	  bitmask.add_point(ptr);
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  template <typename BM>
  void ImageMicroOp<N,T,N2,T2>::populate_approx_bitmask_ranges(BM& bitmask)
  {
    // for now, one access for the whole instance
    AffineAccessor<Rect<N,T>,N2,T2> a_data(inst, field_offset);
    //std::cout << "a_data = " << a_data << "\n";

    // simple image operation - project ever 
    for(IndexSpaceIterator<N2,T2> it(inst_space); it.valid; it.step()) {
      // iterate over each point in the source and mark what it touches
      for(PointInRectIterator<N2,T2> pir(it.rect); pir.valid; pir.step()) {
	Rect<N,T> rng = a_data.read(pir.p);

	// don't test against the diff_rhss filter, but we do need to stay
	//  inside the parent space
	for(IndexSpaceIterator<N,T> it2(parent_space, rng);
	    it2.valid;
	    it2.step())
	  bitmask.add_rect(rng);
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::execute(void)
  {
    TimeStamp ts("ImageMicroOp::execute", true, &log_uop_timing);

    if(!sparsity_outputs.empty()) {
      //std::map<int, DenseRectangleList<N,T> *> rect_map;
      std::map<int, HybridRectangleList<N,T> *> rect_map;

      if(is_ranged)
	populate_bitmasks_ranges(rect_map);
      else
	populate_bitmasks_ptrs(rect_map);

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
	  impl->contribute_dense_rect_list(it2->second->convert_to_vector(),
                                           false /*!disjoint*/);
	  delete it2->second;
	} else
	  impl->contribute_nothing();
      }
    }

    if(approx_output_index != -1) {
      DenseRectangleList<N,T> approx_rects(DeppartConfig::cfg_max_rects_in_approximation);

      if(is_ranged)
	populate_approx_bitmask_ranges(approx_rects);
      else
	populate_approx_bitmask_ptrs(approx_rects);

      if(requestor == Network::my_node_id) {
	PreimageOperation<N2,T2,N,T> *op = reinterpret_cast<PreimageOperation<N2,T2,N,T> *>(approx_output_op);
	op->provide_sparse_image(approx_output_index, &approx_rects.rects[0], approx_rects.rects.size());
      } else {
	size_t payload_size = approx_rects.rects.size() * sizeof(Rect<N,T>);
	ActiveMessage<ApproxImageResponseMessage<PreimageOperation<N2,T2,N,T> > > amsg(requestor, payload_size);
	amsg->approx_output_op = approx_output_op;
	amsg->approx_output_index = approx_output_index;
	amsg.add_payload(&approx_rects.rects[0], payload_size);
	amsg.commit();
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void ImageMicroOp<N,T,N2,T2>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // an ImageMicroOp should always be executed on whichever node the field data lives
    NodeID exec_node = ID(inst).instance_owner_node();

    if(exec_node != Network::my_node_id) {
      forward_microop<ImageMicroOp<N,T,N2,T2> >(exec_node, op, this);
      return;
    }

    // Need valid data for the instance space
    if (!inst_space.dense()) {
      // it's safe to add the count after the registration only because we initialized
      //  the count to 2 instead of 1
      bool registered = SparsityMapImpl<N2,T2>::lookup(inst_space.sparsity)->add_waiter(this, true /*precise*/);
      if(registered)
        wait_count.fetch_add(1);
    }

    // need valid data for each source
    for(size_t i = 0; i < sources.size(); i++) {
      if(!sources[i].dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N2,T2>::lookup(sources[i].sparsity)->add_waiter(this, true /*precise*/);
	if(registered)
	  wait_count.fetch_add(1);
      }
    }

    // need valid data for each diff_rhs (if present)
    for(size_t i = 0; i < diff_rhss.size(); i++) {
      if(!diff_rhss[i].dense()) {
	// it's safe to add the count after the registration only because we initialized
	//  the count to 2 instead of 1
	bool registered = SparsityMapImpl<N,T>::lookup(diff_rhss[i].sparsity)->add_waiter(this, true /*precise*/);
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
  bool ImageMicroOp<N,T,N2,T2>::serialize_params(S& s) const
  {
    return((s << parent_space) &&
	   (s << inst_space) &&
	   (s << inst) &&
	   (s << field_offset) &&
	   (s << is_ranged) &&
	   (s << sources) &&
	   (s << diff_rhss) &&
	   (s << sparsity_outputs) &&
	   (s << approx_output_index) &&
	   (s << approx_output_op));
  }

  template <int N, typename T, int N2, typename T2>
  template <typename S>
  ImageMicroOp<N,T,N2,T2>::ImageMicroOp(NodeID _requestor,
					AsyncMicroOp *_async_microop, S& s)
    : PartitioningMicroOp(_requestor, _async_microop)
  {
    bool ok = ((s >> parent_space) &&
	       (s >> inst_space) &&
	       (s >> inst) &&
	       (s >> field_offset) &&
	       (s >> is_ranged) &&
	       (s >> sources) &&
	       (s >> diff_rhss) &&
	       (s >> sparsity_outputs) &&
	       (s >> approx_output_index) &&
	       (s >> approx_output_op));
    assert(ok);
    (void)ok;
  }

  template <int N, typename T, int N2, typename T2>
  ActiveMessageHandlerReg<RemoteMicroOpMessage<ImageMicroOp<N,T,N2,T2> > > ImageMicroOp<N,T,N2,T2>::areg;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ImageOperation<N,T,N2,T2>

  template <int N, typename T, int N2, typename T2>
  ImageOperation<N, T, N2, T2>::ImageOperation(
      const IndexSpace<N, T> &_parent,
      const DomainTransform<N, T, N2, T2> &_domain_transform,
      const ProfilingRequestSet &reqs, GenEventImpl *_finish_event,
      EventImpl::gen_t _finish_gen)
      : PartitioningOperation(reqs, _finish_event, _finish_gen),
        parent(_parent),
        domain_transform(_domain_transform) {}

  template <int N, typename T, int N2, typename T2>
  ImageOperation<N,T,N2,T2>::~ImageOperation(void)
  {}

  template <int N, typename T, int N2, typename T2>
  IndexSpace<N,T> ImageOperation<N,T,N2,T2>::add_source(const IndexSpace<N2,T2>& source)
  {
    // try to filter out obviously empty sources
    if(parent.empty() || source.empty())
      return IndexSpace<N,T>::make_empty();

    // otherwise it'll be something smaller than the current parent
    IndexSpace<N,T> image;
    image.bounds = parent.bounds;

    // if the source has a sparsity map, use the same node - otherwise
    // get a sparsity ID by round-robin'ing across the nodes that have field data
    int target_node = 0;
    if(!source.dense())
      target_node = ID(source.sparsity).sparsity_creator_node();
    else
      if(!domain_transform.ptr_data.empty())
	target_node = ID(domain_transform.ptr_data[sources.size() % domain_transform.ptr_data.size()].inst).instance_owner_node();
      else
	target_node = ID(domain_transform.range_data[sources.size() % domain_transform.range_data.size()].inst).instance_owner_node();

    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    image.sparsity = sparsity;

    sources.push_back(source);
    images.push_back(sparsity);

    return image;
  }

  template <int N, typename T, int N2, typename T2>
  IndexSpace<N,T> ImageOperation<N,T,N2,T2>::add_source_with_difference(const IndexSpace<N2,T2>& source,
                                                                         const IndexSpace<N,T>& diff_rhs)
  {
    // try to filter out obviously empty sources
    if(parent.empty() || source.empty())
      return IndexSpace<N,T>::make_empty();

    // otherwise it'll be something smaller than the current parent
    IndexSpace<N,T> image;
    image.bounds = parent.bounds;

    // if the source has a sparsity map, use the same node - otherwise
    // get a sparsity ID by round-robin'ing across the nodes that have field data
    int target_node;
    if(!source.dense())
      target_node = ID(source.sparsity).sparsity_creator_node();
    else
      if(!domain_transform.ptr_data.empty())
	target_node = ID(domain_transform.ptr_data[sources.size() % domain_transform.ptr_data.size()].inst).instance_owner_node();
      else
	target_node = ID(domain_transform.range_data[sources.size() % domain_transform.range_data.size()].inst).instance_owner_node();
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    image.sparsity = sparsity;

    sources.push_back(source);
    diff_rhss.push_back(diff_rhs);
    images.push_back(sparsity);

    return image;
  }

  template <int N, typename T, int N2, typename T2>
  void ImageOperation<N, T, N2, T2>::execute(void) {
   if (domain_transform.type ==
       DomainTransform<N, T, N2, T2>::DomainTransformType::STRUCTURED) {
    for (size_t i = 0; i < sources.size(); i++) {
     SparsityMapImpl<N, T>::lookup(images[i])->set_contributor_count(1);
    }

    StructuredImageMicroOp<N, T, N2, T2> *micro_op =
        new StructuredImageMicroOp<N, T, N2, T2>(
            parent, domain_transform.structured_transform);

    for (size_t j = 0; j < sources.size(); j++) {
     micro_op->add_sparsity_output(sources[j], images[j]);
    }

    micro_op->dispatch(this, /*inline_ok=*/true);
   } else {
    if (!DeppartConfig::cfg_disable_intersection_optimization) {
     // build the overlap tester based on the field index spaces - they're more
     // likely to be known and
     //  denser
     ComputeOverlapMicroOp<N2, T2> *uop =
         new ComputeOverlapMicroOp<N2, T2>(this);

     for (size_t i = 0; i < domain_transform.ptr_data.size(); i++)
      uop->add_input_space(domain_transform.ptr_data[i].index_space);

     for (size_t i = 0; i < domain_transform.range_data.size(); i++)
      uop->add_input_space(domain_transform.range_data[i].index_space);

     // we will ask this uop to also prefetch the sources we will intersect test
     // against it
     for (size_t i = 0; i < sources.size(); i++)
      uop->add_extra_dependency(sources[i]);

     uop->dispatch(this, true /* ok to run in this thread */);
    } else {
     // launch full cross-product of image micro ops right away
     for (size_t i = 0; i < sources.size(); i++)
      SparsityMapImpl<N, T>::lookup(images[i])->set_contributor_count(
          domain_transform.ptr_data.size() +
          domain_transform.range_data.size());

     for (size_t i = 0; i < domain_transform.ptr_data.size(); i++) {
      ImageMicroOp<N, T, N2, T2> *uop = new ImageMicroOp<N, T, N2, T2>(
          parent, domain_transform.ptr_data[i].index_space,
          domain_transform.ptr_data[i].inst,
          domain_transform.ptr_data[i].field_offset, false /*ptrs*/);
      for (size_t j = 0; j < sources.size(); j++)
       if (diff_rhss.empty())
        uop->add_sparsity_output(sources[j], images[j]);
       else
        uop->add_sparsity_output_with_difference(sources[j], diff_rhss[j],
                                                 images[j]);

      uop->dispatch(this, true /* ok to run in this thread */);
     }

     for (size_t i = 0; i < domain_transform.range_data.size(); i++) {
      ImageMicroOp<N, T, N2, T2> *uop = new ImageMicroOp<N, T, N2, T2>(
          parent, domain_transform.range_data[i].index_space,
          domain_transform.range_data[i].inst,
          domain_transform.range_data[i].field_offset, true /*ranges*/);
      for (size_t j = 0; j < sources.size(); j++)
       if (diff_rhss.empty())
        uop->add_sparsity_output(sources[j], images[j]);
       else
        uop->add_sparsity_output_with_difference(sources[j], diff_rhss[j],
                                                 images[j]);

      uop->dispatch(this, true /* ok to run in this thread */);
     }
    }
   }
  }

  template <int N, typename T, int N2, typename T2>
  void ImageOperation<N,T,N2,T2>::set_overlap_tester(void *tester)
  {
    OverlapTester<N2,T2> *overlap_tester = static_cast<OverlapTester<N2,T2> *>(tester);

    // we asked the overlap tester to prefetch all the source data we need, so we can use it
    //  right away (and then delete it)
    std::vector<std::set<int> > overlaps_by_field_data(domain_transform.ptr_data.size() +
						       domain_transform.range_data.size());
    for(size_t i = 0; i < sources.size(); i++) {
      std::set<int> overlaps_by_source;

      overlap_tester->test_overlap(sources[i], overlaps_by_source, true /*approx*/);

      log_part.info() << overlaps_by_source.size() << " overlaps for source " << i;

      SparsityMapImpl<N,T>::lookup(images[i])->set_contributor_count(overlaps_by_source.size());

      // now scatter these values into the overlaps_by_field_data
      for(std::set<int>::const_iterator it = overlaps_by_source.begin();
	  it != overlaps_by_source.end();
	  it++)
	overlaps_by_field_data[*it].insert(i);
    }
    delete overlap_tester;

    for(size_t i = 0; i < domain_transform.ptr_data.size(); i++) {
      const std::set<int>& overlaps = overlaps_by_field_data[i];
      size_t n = overlaps.size();
      if(n == 0) continue;

      ImageMicroOp<N,T,N2,T2> *uop = new ImageMicroOp<N,T,N2,T2>(parent,
								 domain_transform.ptr_data[i].index_space,
								 domain_transform.ptr_data[i].inst,
								 domain_transform.ptr_data[i].field_offset,
								 false /*ptrs*/);
      for(std::set<int>::const_iterator it = overlaps.begin();
	  it != overlaps.end();
	  it++) {
	int j = *it;
        if(diff_rhss.empty())
	  uop->add_sparsity_output(sources[j], images[j]);
        else
	  uop->add_sparsity_output_with_difference(sources[j], diff_rhss[j], images[j]);
      }
      uop->dispatch(this, true /* ok to run in this thread */);
    }

    for(size_t i = 0; i < domain_transform.range_data.size(); i++) {
      const std::set<int>& overlaps = overlaps_by_field_data[i + domain_transform.ptr_data.size()];
      size_t n = overlaps.size();
      if(n == 0) continue;

      ImageMicroOp<N,T,N2,T2> *uop = new ImageMicroOp<N,T,N2,T2>(parent,
								 domain_transform.range_data[i].index_space,
								 domain_transform.range_data[i].inst,
								 domain_transform.range_data[i].field_offset,
								 true /*ranges*/);
      for(std::set<int>::const_iterator it = overlaps.begin();
	  it != overlaps.end();
	  it++) {
	int j = *it;
        if(diff_rhss.empty())
	  uop->add_sparsity_output(sources[j], images[j]);
        else
	  uop->add_sparsity_output_with_difference(sources[j], diff_rhss[j], images[j]);
      }
      uop->dispatch(this, true /* ok to run in this thread */);
    }
  }

  template <int N, typename T, int N2, typename T2>
  void ImageOperation<N,T,N2,T2>::print(std::ostream& os) const
  {
    os << "ImageOperation(" << parent << ")";
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class StructuredImageMicroOp<N, T, N2, T2>

  template <int N, typename T, int N2, typename T2>
  StructuredImageMicroOp<N, T, N2, T2>::StructuredImageMicroOp(
      const IndexSpace<N, T> &_parent,
      const StructuredTransform<N, T, N2, T2> &_transform)
      : parent_space(_parent), transform(_transform) {}

  template <int N, typename T, int N2, typename T2>
  StructuredImageMicroOp<N, T, N2, T2>::~StructuredImageMicroOp() {}

  template <int N, typename T, int N2, typename T2>
  void StructuredImageMicroOp<N, T, N2, T2>::dispatch(
      PartitioningOperation *op, bool inline_ok) {
    for (size_t i = 0; i < sources.size(); i++) {
      if (!sources[i].dense()) {
        bool registered = SparsityMapImpl<N2, T2>::lookup(sources[i].sparsity)
                              ->add_waiter(this, true /*precise*/);
        if (registered) wait_count.fetch_add(1);
      }
    }

    if (!parent_space.dense()) {
      bool registered = SparsityMapImpl<N, T>::lookup(parent_space.sparsity)
                            ->add_waiter(this, true /*precise*/);
      if (registered) wait_count.fetch_add(1);
    }
    this->finish_dispatch(op, inline_ok);
  }

  template <int N, typename T, int N2, typename T2>
  void StructuredImageMicroOp<N, T, N2, T2>::add_sparsity_output(
      IndexSpace<N2, T2> _source, SparsityMap<N, T> _sparsity) {
   sources.push_back(_source);
   sparsity_outputs.push_back(_sparsity);
  }

  template <int N, typename T, int N2, typename T2>
  void StructuredImageMicroOp<N, T, N2, T2>::execute(void) {
    TimeStamp ts("StructuredImageMicroOp::execute", true, &log_uop_timing);

    if (!sparsity_outputs.empty()) {
      std::map<int, HybridRectangleList<N, T> *> rect_map;

      populate(rect_map);

#ifdef DEBUG_PARTITIONING
      std::cout << rect_map.size() << " non-empty images present in instance "
                << inst << std::endl;
      for (typename std::map<int, DenseRectangleList<N, T> *>::const_iterator
               it = rect_map.begin();
           it != rect_map.end(); it++)
        std::cout << "  " << sources[it->first] << " = "
                  << it->second->rects.size() << " rectangles" << std::endl;
#endif

      // iterate over sparsity outputs and contribute to all (even if we didn't
      // have any points found for it)
      for (size_t i = 0; i < sparsity_outputs.size(); i++) {
        SparsityMapImpl<N, T> *impl =
            SparsityMapImpl<N, T>::lookup(sparsity_outputs[i]);
        typename std::map<int, HybridRectangleList<N, T> *>::const_iterator
            it2 = rect_map.find(i);
        if (it2 != rect_map.end()) {
          impl->contribute_dense_rect_list(it2->second->convert_to_vector(),
                                           false /*!disjoint*/);
          delete it2->second;
        } else
          impl->contribute_nothing();
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  void StructuredImageMicroOp<N, T, N2, T2>::populate(
      std::map<int, HybridRectangleList<N, T> *> &bitmasks) {
   // TODO(apyakhin@): Just for now we effectively defaulting to affine
   // an transform. Consider adding both more generalized version that
   // works for any type of structured transforms as well as specialized
   // version.
   std::vector<Rect<N, T>> parent_rects;
   if (this->parent_space.dense()) {
    parent_rects.push_back(this->parent_space.bounds);
   } else {
    for (IndexSpaceIterator<N, T> parent_it(this->parent_space);
         parent_it.valid; parent_it.step()) {
     parent_rects.push_back(parent_it.rect);
    }
   }

   assert(!parent_rects.empty());
   Rect<N, T> parent_bbox = parent_rects[0];
   for (size_t i = 1; i < parent_rects.size(); i++) {
    parent_bbox = parent_bbox.union_bbox(parent_rects[i]);
   }

   for (size_t i = 0; i < this->sources.size(); i++) {
    for (IndexSpaceIterator<N2, T2> it2(this->sources[i]); it2.valid;
         it2.step()) {
     for (PointInRectIterator<N2, T2> point(it2.rect); point.valid;
          point.step()) {
      Point<N, T> source_point = transform[point.p];

      if (!parent_bbox.contains(source_point)) continue;

      for (const auto &parent_rect : parent_rects) {
       if (parent_rect.contains(source_point)) {
        HybridRectangleList<N, T> **bmpp = 0;
        if (!bmpp) bmpp = &bitmasks[i];
        if (!*bmpp) *bmpp = new HybridRectangleList<N, T>;
        (*bmpp)->add_point(source_point);
       }
      }
     }
    }
   }
  }

  ////////////////////////////////////////////////////////////////////////

  // instantiations of templates handled in image_tmpl.cc

  };  // namespace Realm
