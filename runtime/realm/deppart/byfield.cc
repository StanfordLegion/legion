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

// byfield (filter) operations for Realm dependent partitioning

#include "realm/deppart/byfield.h"

#include "realm/deppart/deppart_config.h"
#include "realm/deppart/rectlist.h"
#include "realm/deppart/inst_helper.h"
#include "realm/logging.h"

namespace Realm {

  extern Logger log_part;
  extern Logger log_uop_timing;


  template <int N, typename T>
  template <typename FT>
  Event IndexSpace<N,T>::create_subspaces_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
						    const std::vector<FT>& colors,
						    std::vector<IndexSpace<N,T> >& subspaces,
						    const ProfilingRequestSet &reqs,
						    Event wait_on /*= Event::NO_EVENT*/) const
  {
    // output vector should start out empty
    assert(subspaces.empty());

    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event e = finish_event->current_event();
    ByFieldOperation<N,T,FT> *op = new ByFieldOperation<N,T,FT>(*this, field_data, reqs, finish_event, ID(e).event_generation());

    size_t n = colors.size();
    subspaces.resize(n);
    for(size_t i = 0; i < n; i++) {
      subspaces[i] = op->add_color(colors[i]);
      log_dpops.info() << "byfield: " << *this << ", " << colors[i] << " -> " << subspaces[i] << " (" << e << ")";
    }

    op->launch(wait_on);
    return e;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ByFieldMicroOp<N,T,FT>

  template <int N, typename T, typename FT>
  ByFieldMicroOp<N,T,FT>::ByFieldMicroOp(IndexSpace<N,T> _parent_space,
					 IndexSpace<N,T> _inst_space,
					 RegionInstance _inst,
					 size_t _field_offset)
    : parent_space(_parent_space)
    , inst_space(_inst_space)
    , inst(_inst)
    , field_offset(_field_offset)
    , value_range_valid(false)
    , value_set_valid(false)
  {
    areg.force_instantiation();
  }

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
    for(IndexSpaceIterator<N,T> it(inst_space); it.valid; it.step()) {
      for(IndexSpaceIterator<N,T> it2(parent_space, it.rect); it2.valid; it2.step()) {
	const Rect<N,T>& r = it2.rect;
	Point<N,T> p = r.lo;
	while(true) {
	  FT val = a_data.read(p);
	  Point<N,T> p2 = p;
	  while(p2.x < r.hi.x) {
	    Point<N,T> p3 = p2;
	    p3.x++;
	    FT val2 = a_data.read(p3);
	    if(val != val2) {
	      // record old strip
	      BM *&bmp = bitmasks[val];
	      if(!bmp) bmp = new BM;
	      bmp->add_rect(Rect<N,T>(p,p2));
	      //std::cout << val << ": " << p << ".." << p2 << std::endl;
	      val = val2;
	      p = p3;
	    }
	    p2 = p3;
	  }
	  // record whatever strip we have at the end
	  BM *&bmp = bitmasks[val];
	  if(!bmp) bmp = new BM;
	  bmp->add_rect(Rect<N,T>(p,p2));
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
	impl->contribute_dense_rect_list(it2->second->rects, true /*disjoint*/);
	delete it2->second;
      } else
	impl->contribute_nothing();
    }
  }

  template <int N, typename T, typename FT>
  void ByFieldMicroOp<N,T,FT>::dispatch(PartitioningOperation *op, bool inline_ok)
  {
    // a ByFieldMicroOp should always be executed on whichever node the field data lives
    NodeID exec_node = ID(inst).instance_owner_node();

    if(exec_node != Network::my_node_id) {
      forward_microop<ByFieldMicroOp<N,T,FT> >(exec_node, op, this);
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
  ByFieldMicroOp<N,T,FT>::ByFieldMicroOp(NodeID _requestor,
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
    (void)ok;
  }

  template <int N, typename T, typename FT>
  ActiveMessageHandlerReg<RemoteMicroOpMessage<ByFieldMicroOp<N,T,FT> > > ByFieldMicroOp<N,T,FT>::areg;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ByFieldOperation<N,T,FT>

  template <int N, typename T, typename FT>
  ByFieldOperation<N,T,FT>::ByFieldOperation(const IndexSpace<N,T>& _parent,
					     const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& _field_data,
					     const ProfilingRequestSet &reqs,
					     GenEventImpl *_finish_event,
					     EventImpl::gen_t _finish_gen)
    : PartitioningOperation(reqs, _finish_event, _finish_gen)
    , parent(_parent)
    , field_data(_field_data)
  {}

  template <int N, typename T, typename FT>
  ByFieldOperation<N,T,FT>::~ByFieldOperation(void)
  {}

  template <int N, typename T, typename FT>
  IndexSpace<N,T> ByFieldOperation<N,T,FT>::add_color(FT color)
  {
    // an empty parent leads to trivially empty subspaces
    if(parent.empty())
      return IndexSpace<N,T>::make_empty();

    // otherwise it'll be something smaller than the current parent
    IndexSpace<N,T> subspace;
    subspace.bounds = parent.bounds;

    // get a sparsity ID by round-robin'ing across the nodes that have field data
    int target_node = ID(field_data[colors.size() % field_data.size()].inst).instance_owner_node();
    SparsityMap<N,T> sparsity = get_runtime()->get_available_sparsity_impl(target_node)->me.convert<SparsityMap<N,T> >();
    subspace.sparsity = sparsity;

    colors.push_back(color);
    subspaces.push_back(sparsity);

    return subspace;
  }

  template <int N, typename T, typename FT>
  void ByFieldOperation<N,T,FT>::execute(void)
  {
    for(size_t i = 0; i < subspaces.size(); i++)
      SparsityMapImpl<N,T>::lookup(subspaces[i])->set_contributor_count(field_data.size());

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

  template <int N, typename T, typename FT>
  void ByFieldOperation<N,T,FT>::print(std::ostream& os) const
  {
    os << "ByFieldOperation(" << parent << ")";
  }

#define DOIT(N,T,F) \
  template class ByFieldMicroOp<N,T,F>; \
  template class ByFieldOperation<N,T,F>; \
  template ByFieldMicroOp<N,T,F>::ByFieldMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template Event IndexSpace<N,T>::create_subspaces_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,F> >&, \
							     const std::vector<F>&, \
							     std::vector<IndexSpace<N,T> >&, \
							     const ProfilingRequestSet &, \
							     Event) const;
#ifndef REALM_TEMPLATES_ONLY
  FOREACH_NTF(DOIT)
#endif

  // instantiations of point/rect-field templates handled in byfield_tmpl.cc

};
