/* Copyright 2019 Stanford University, NVIDIA Corporation
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

// data transfer (a.k.a. dma) engine for Realm

#include "realm/transfer/transfer.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/mem_impl.h"
#include "realm/inst_layout.h"
#ifdef USE_HDF
#include "realm/hdf5/hdf5_access.h"
#endif

namespace Realm {

  extern Logger log_dma;

  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIterator
  //

  TransferIterator::~TransferIterator(void)
  {}

  Event TransferIterator::request_metadata(void)
  {
    // many iterators have no metadata
    return Event::NO_EVENT;
  }

#ifdef USE_HDF
  size_t TransferIterator::step(size_t max_bytes, AddressInfoHDF5& info,
				bool tentative /*= false*/)
  {
    // should never be called
    return 0;
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIteratorIndexSpace<N,T>
  //

  template <int N, typename T>
  class TransferIteratorIndexSpace : public TransferIterator {
  protected:
    TransferIteratorIndexSpace(void); // used by deserializer
  public:
    TransferIteratorIndexSpace(const IndexSpace<N,T> &_is,
				RegionInstance inst,
			        const int _dim_order[N],
				const std::vector<FieldID>& _fields,
				size_t _extra_elems);

    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);
      
    virtual ~TransferIteratorIndexSpace(void);

    virtual Event request_metadata(void);

    virtual void reset(void);
    virtual bool done(void);
    virtual size_t step(size_t max_bytes, AddressInfo& info,
			unsigned flags,
			bool tentative = false);
#ifdef USE_HDF
    virtual size_t step(size_t max_bytes, AddressInfoHDF5& info,
			bool tentative = false);
#endif
    virtual void confirm_step(void);
    virtual void cancel_step(void);

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndexSpace<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    IndexSpace<N,T> is;
    IndexSpaceIterator<N,T> iter;
    bool iter_init_deferred;
    Point<N,T> cur_point, next_point;
    bool carry;
    RegionInstanceImpl *inst_impl;
    const InstanceLayout<N,T> *inst_layout;
    std::vector<FieldID> fields;
    size_t field_idx;
    size_t extra_elems;
    bool tentative_valid;
    int dim_order[N];
  };

  template <int N, typename T>
  TransferIteratorIndexSpace<N,T>::TransferIteratorIndexSpace(const IndexSpace<N,T>& _is,
								RegionInstance inst,
							        const int _dim_order[N],
								const std::vector<FieldID>& _fields,
								size_t _extra_elems)
    : is(_is), field_idx(0), extra_elems(_extra_elems), tentative_valid(false)
  {
    for(int i = 0; i < N; i++) dim_order[i] = _dim_order[i];

    if(is.is_valid()) {
      iter.reset(is);
      iter_init_deferred = false;
    } else
      iter_init_deferred = true;

    // special case - skip a lot of the init if we know the space is empty
    if(!iter_init_deferred && !iter.valid) {
      inst_impl = 0;
      inst_layout = 0;
    } else {
      cur_point = iter.rect.lo;

      inst_impl = get_runtime()->get_instance_impl(inst);
      inst_layout = checked_cast<const InstanceLayout<N,T> *>(inst.get_layout());
      fields = _fields;
    }
  }

  template <int N, typename T>
  TransferIteratorIndexSpace<N,T>::TransferIteratorIndexSpace(void)
    : iter_init_deferred(false)
    , field_idx(0)
    , tentative_valid(false)
  {}

  template <int N, typename T>
  template <typename S>
  /*static*/ TransferIterator *TransferIteratorIndexSpace<N,T>::deserialize_new(S& deserializer)
  {
    IndexSpace<N,T> is;
    RegionInstance inst;
    std::vector<FieldID> fields;
    size_t extra_elems;
    int dim_order[N];

    if(!((deserializer >> is) &&
	 (deserializer >> inst) &&
	 (deserializer >> fields) &&
	 (deserializer >> extra_elems)))
      return 0;

    for(int i = 0; i < N; i++)
      if(!(deserializer >> dim_order[i]))
	return 0;

    TransferIteratorIndexSpace<N,T> *tiis = new TransferIteratorIndexSpace<N,T>(is,
										inst,
										dim_order,
										fields,
										extra_elems);

    return tiis;
  }

  template <int N, typename T>
  TransferIteratorIndexSpace<N,T>::~TransferIteratorIndexSpace(void)
  {}

  template <int N, typename T>
  Event TransferIteratorIndexSpace<N,T>::request_metadata(void)
  {
    std::set<Event> events;

    if(iter_init_deferred)
      events.insert(is.make_valid());

    if(inst_impl && !inst_impl->metadata.is_valid())
      events.insert(inst_impl->request_metadata());

    return Event::merge_events(events);
  }

  template <int N, typename T>
  void TransferIteratorIndexSpace<N,T>::reset(void)
  {
    field_idx = 0;
    assert(!iter_init_deferred);
    iter.reset(iter.space);
    cur_point = iter.rect.lo;
  }

  template <int N, typename T>
  bool TransferIteratorIndexSpace<N,T>::done(void)
  {
    if(iter_init_deferred) {
      // index space must be valid now (i.e. somebody should have waited)
      assert(is.is_valid());
      iter.reset(is);
      if(iter.valid)
	cur_point = iter.rect.lo;
      else
	fields.clear();
      iter_init_deferred = false;
    }
    return(field_idx == fields.size());
  }

  template <int N, typename T>
  size_t TransferIteratorIndexSpace<N,T>::step(size_t max_bytes, AddressInfo& info,
						unsigned flags,
						bool tentative /*= false*/)
  {
    assert(!done());
    assert(!tentative_valid);

    // shouldn't be here if the iterator isn't valid
    assert(iter.valid);

    // find the layout piece the current point is in
    const InstanceLayoutPiece<N,T> *layout_piece;
    int field_rel_offset;
    size_t field_size;
    size_t total_bytes = 0;
    {
      std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(fields[field_idx]);
      assert(it != inst_layout->fields.end());
      const InstancePieceList<N,T>& piece_list = inst_layout->piece_lists[it->second.list_idx];
      layout_piece = piece_list.find_piece(cur_point);
      assert(layout_piece != 0);
      field_rel_offset = it->second.rel_offset;
      field_size = it->second.size_in_bytes;
      //log_dma.print() << "F " << field_idx << " " << fields[field_idx] << " : " << it->second.list_idx << " " << field_rel_offset << " " << field_size;
    }

    size_t max_elems = max_bytes / field_size;
    // less than one element?  give up immediately
    if(max_elems == 0)
      return 0;

    // the subrectangle we give always starts with the current point
    Rect<N,T> target_subrect;
    target_subrect.lo = cur_point;
    if(layout_piece->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType) {
      const AffineLayoutPiece<N,T> *affine = static_cast<const AffineLayoutPiece<N,T> *>(layout_piece);

      // using the current point, find the biggest subrectangle we want to try
      //  giving out, paying attention to the piece's bounds, where we've stopped,
      //  and piece strides
      int cur_dim = 0;
      int max_dims = (((flags & LINES_OK) == 0)  ? 1 :
		      ((flags & PLANES_OK) == 0) ? 2 :
		                                   3);
      ssize_t act_counts[3], act_strides[3];
      act_counts[0] = field_size;
      act_strides[0] = 1;
      total_bytes = field_size;
      for(int d = 1; d < 3; d++) {
	act_counts[d] = 1;
	act_strides[d] = 0;
      }
      // follow the agreed-upon dimension ordering
      for(int di = 0; di < N; di++) {
	int d = dim_order[di];

	// the stride for a degenerate dimensions does not matter - don't cause
	//   a "break" if it mismatches
	if((cur_dim < max_dims) &&
	   (cur_point[d] < iter.rect.hi[d]) &&
	   ((ssize_t)affine->strides[d] != (act_counts[cur_dim] * act_strides[cur_dim]))) {
	  cur_dim++;
	  if(cur_dim < max_dims)
	    act_strides[cur_dim] = (ssize_t)affine->strides[d];
	}
	if(cur_dim < max_dims) {
	  size_t len = iter.rect.hi[d] - cur_point[d] + 1;
	  size_t piece_limit = affine->bounds.hi[d] - cur_point[d] + 1;
	  bool cropped = false;
	  if(piece_limit < len) {
	    len = piece_limit;
	    cropped = true;
	  }
	  size_t byte_limit = max_bytes / total_bytes;
	  if(byte_limit < len) {
	    len = byte_limit;
	    cropped = true;
	  }
	  target_subrect.hi[d] = cur_point[d] + len - 1;
	  total_bytes *= len;
	  act_counts[cur_dim] *= len;
	  // if we didn't start this dimension at the lo point, we can't
	  //  grow any further
	  if(cropped || (cur_point[d] > iter.rect.lo[d]))
	    cur_dim = max_dims;
	} else
	  target_subrect.hi[d] = cur_point[d];
      }

      info.base_offset = (inst_impl->metadata.inst_offset +
			  affine->offset +
			  affine->strides.dot(cur_point) +
			  field_rel_offset);
      //log_dma.print() << "A " << inst_impl->metadata.inst_offset << " + " << affine->offset << " + (" << affine->strides << " . " << cur_point << ") + " << field_rel_offset << " = " << info.base_offset;
      info.bytes_per_chunk = act_counts[0];
      info.num_lines = act_counts[1];
      info.line_stride = act_strides[1];
      info.num_planes = act_counts[2];
      info.plane_stride = act_strides[2];
    } else {
      assert(0 && "no support for non-affine pieces yet");
    }

    // now set 'next_point' to the next point we want - this is just based on
    //  the iterator rectangle so that iterators using different layouts still
    //  agree
    carry = true;
    for(int di = 0; di < N; di++) {
      int d = dim_order[di];

      if(carry) {
	if(target_subrect.hi[d] == iter.rect.hi[d]) {
	  next_point[d] = iter.rect.lo[d];
	} else {
	  next_point[d] = target_subrect.hi[d] + 1;
	  carry = false;
	}
      } else
	next_point[d] = target_subrect.lo[d];
    }

    //log_dma.print() << "iter " << ((void *)this) << " " << field_idx << " " << iter.rect << " " << cur_point << " " << max_bytes << " : " << target_subrect << " " << next_point << " " << carry;

    if(tentative) {
      tentative_valid = true;
    } else {
      // if the "carry" propagated all the way through, go on to the next field
      //  (defer if tentative)
      if(carry) {
	if(iter.step()) {
	  cur_point = iter.rect.lo;
	} else {
	  field_idx++;
	  iter.reset(iter.space);
	  cur_point = iter.rect.lo;
	}
      } else
	cur_point = next_point;
    }

    return total_bytes;
  }

#ifdef USE_HDF
  template <int N, typename T>
  size_t TransferIteratorIndexSpace<N,T>::step(size_t max_bytes, AddressInfoHDF5& info,
						bool tentative /*= false*/)
  {
    if(iter_init_deferred) {
      // index space must be valid now (i.e. somebody should have waited)
      assert(is.is_valid());
      iter.reset(is);
      if(!iter.valid) fields.clear();
      iter_init_deferred = false;
    }
    assert(!done());
    assert(!tentative_valid);

    // shouldn't be here if the iterator isn't valid
    assert(iter.valid);

    // find the layout piece the current point is in
    const InstanceLayoutPiece<N,T> *layout_piece;
    //int field_rel_offset;
    size_t field_size;
    {
      std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(fields[field_idx]);
      assert(it != inst_layout->fields.end());
      const InstancePieceList<N,T>& piece_list = inst_layout->piece_lists[it->second.list_idx];
      layout_piece = piece_list.find_piece(cur_point);
      assert(layout_piece != 0);
      //field_rel_offset = it->second.rel_offset;
      field_size = it->second.size_in_bytes;
      //log_dma.print() << "F " << field_idx << " " << fields[field_idx] << " : " << it->second.list_idx << " " << field_rel_offset << " " << field_size;
    }

    size_t max_elems = max_bytes / field_size;
    // less than one element?  give up immediately
    if(max_elems == 0)
      return 0;

    // std::cout << "step " << this << " " << r << " " << p << " " << field_idx
    // 	      << " " << max_bytes << ":";

    // HDF5 requires we handle dimensions in order - no permutation allowed
    // using the current point, find the biggest subrectangle we want to try
    //  giving out
    Rect<N,T> target_subrect;
    size_t cur_bytes = 0;
    target_subrect.lo = cur_point;
    if(layout_piece->layout_type == InstanceLayoutPiece<N,T>::HDF5LayoutType) {
      const HDF5LayoutPiece<N,T> *hlp = static_cast<const HDF5LayoutPiece<N,T> *>(layout_piece);

      info.filename = &hlp->filename;
      info.dsetname = &hlp->dsetname;

      bool grow = true;
      cur_bytes = field_size;
      // follow the agreed-upon dimension ordering
      for(int di = 0; di < N; di++) {
	int d = dim_order[di];

	if(grow) {
	  size_t len = iter.rect.hi[d] - cur_point[d] + 1;
	  size_t piece_limit = hlp->bounds.hi[d] - cur_point[d] + 1;
	  if(piece_limit < len) {
	    len = piece_limit;
	    grow = false;
	  }
	  size_t byte_limit = max_bytes / cur_bytes;
	  if(byte_limit < len) {
	    len = byte_limit;
	    grow = false;
	  }
	  target_subrect.hi[d] = cur_point[d] + len - 1;
	  cur_bytes *= len;
	  // if we didn't start this dimension at the lo point, we can't
	  //  grow any further
	  if(cur_point[d] > iter.rect.lo[d])
	    grow = false;
	} else
	  target_subrect.hi[d] = cur_point[d];
      }

      // translate the target_subrect into the dataset's coordinates
      //   using the dimension order specified in the instance's layout
      info.dset_bounds.resize(N);
      info.offset.resize(N);
      info.extent.resize(N);
      for(int di = 0; di < N; di++) {
	int d = dim_order[di];

	info.offset[hlp->dim_order[d]] = (target_subrect.lo[d] - hlp->bounds.lo[d] + hlp->offset[d]);
	info.extent[hlp->dim_order[d]] = (target_subrect.hi[d] - target_subrect.lo[d] + 1);
	info.dset_bounds[hlp->dim_order[d]] = (hlp->offset[d] +
					       (hlp->bounds.hi[d] - hlp->bounds.lo[d]) +
					       1);
      }
    } else {
      assert(0);
    }

    // now set 'next_point' to the next point we want - this is just based on
    //  the iterator rectangle so that iterators using different layouts still
    //  agree
    carry = true;
    for(int di = 0; di < N; di++) {
      int d = dim_order[di];

      if(carry) {
	if(target_subrect.hi[d] == iter.rect.hi[d]) {
	  next_point[d] = iter.rect.lo[d];
	} else {
	  next_point[d] = target_subrect.hi[d] + 1;
	  carry = false;
	}
      } else
	next_point[d] = target_subrect.lo[d];
    }

    if(tentative) {
      tentative_valid = true;
    } else {
      // if the "carry" propagated all the way through, go on to the next field
      //  (defer if tentative)
      if(carry) {
	if(iter.step()) {
	  cur_point = iter.rect.lo;
	} else {
	  field_idx++;
	  iter.reset(iter.space);
	  cur_point = iter.rect.lo;
	}
      } else
	cur_point = next_point;
    }

    return cur_bytes;
  }
#endif
  
  template <int N, typename T>
  void TransferIteratorIndexSpace<N,T>::confirm_step(void)
  {
    assert(tentative_valid);
    if(carry) {
      if(iter.step()) {
	cur_point = iter.rect.lo;
      } else {
	field_idx++;
	iter.reset(iter.space);
	cur_point = iter.rect.lo;
      }
    } else
      cur_point = next_point;
    tentative_valid = false;
  }

  template <int N, typename T>
  void TransferIteratorIndexSpace<N,T>::cancel_step(void)
  {
    assert(tentative_valid);
    tentative_valid = false;
  }

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndexSpace<N,T> > TransferIteratorIndexSpace<N,T>::serdez_subclass;

  template <int N, typename T>
  template <typename S>
  bool TransferIteratorIndexSpace<N,T>::serialize(S& serializer) const
  {
    if(!((serializer << iter.space) &&
	 (serializer << (inst_impl ? inst_impl->me :
			 RegionInstance::NO_INST)) &&
	 (serializer << fields) &&
	 (serializer << extra_elems)))
      return false;

    for(int i = 0; i < N; i++)
      if(!(serializer << dim_order[i]))
	return false;

    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDomain
  //

  TransferDomain::TransferDomain(void)
  {}

  TransferDomain::~TransferDomain(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDomainIndexSpace<N,T>
  //

  template <int N, typename T>
  class TransferDomainIndexSpace : public TransferDomain {
  public:
    TransferDomainIndexSpace(IndexSpace<N,T> _is);

    template <typename S>
    static TransferDomain *deserialize_new(S& deserializer);

    virtual TransferDomain *clone(void) const;

    virtual Event request_metadata(void);

    virtual size_t volume(void) const;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      RegionInstance peer,
					      const std::vector<FieldID>& fields) const;

    virtual void print(std::ostream& os) const;

    static Serialization::PolymorphicSerdezSubclass<TransferDomain, TransferDomainIndexSpace<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

    //protected:
    IndexSpace<N,T> is;
  };

  template <int N, typename T>
  TransferDomainIndexSpace<N,T>::TransferDomainIndexSpace(IndexSpace<N,T> _is)
    : is(_is)
  {}

  template <int N, typename T>
  template <typename S>
  /*static*/ TransferDomain *TransferDomainIndexSpace<N,T>::deserialize_new(S& deserializer)
  {
    IndexSpace<N,T> is;
    if(deserializer >> is)
      return new TransferDomainIndexSpace<N,T>(is);
    else
      return 0;
  }

  template <int N, typename T>
  TransferDomain *TransferDomainIndexSpace<N,T>::clone(void) const
  {
    return new TransferDomainIndexSpace<N,T>(is);
  }

  template <int N, typename T>
  Event TransferDomainIndexSpace<N,T>::request_metadata(void)
  {
    if(!is.is_valid())
      return is.make_valid();

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  size_t TransferDomainIndexSpace<N,T>::volume(void) const
  {
    return is.volume();
  }

  template <int N, typename T>
  TransferIterator *TransferDomainIndexSpace<N,T>::create_iterator(RegionInstance inst,
								    RegionInstance peer,
								    const std::vector<FieldID>& fields) const
  {
    size_t extra_elems = 0;
    int dim_order[N];
    bool have_ordering = false;
    bool force_fortran_order = false;
    std::vector<RegionInstance> insts(1, inst);
    if(peer.exists()) insts.push_back(peer);
    for(std::vector<RegionInstance>::iterator ii = insts.begin();
	ii != insts.end();
	++ii) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(*ii);
      // can't wait for it here - make sure it's valid before calling
      assert(impl->metadata.is_valid());
      const InstanceLayout<N,T> *layout = checked_cast<const InstanceLayout<N,T> *>(impl->metadata.layout);
      for(typename std::vector<InstancePieceList<N,T> >::const_iterator it = layout->piece_lists.begin();
	  it != layout->piece_lists.end();
	  ++it) {
	for(typename std::vector<InstanceLayoutPiece<N,T> *>::const_iterator it2 = it->pieces.begin();
	    it2 != it->pieces.end();
	    ++it2) {
	  if((*it2)->layout_type != InstanceLayoutPiece<N,T>::AffineLayoutType) {
	    force_fortran_order = true;
	    break;
	  }
	  const AffineLayoutPiece<N,T> *affine = checked_cast<const AffineLayoutPiece<N,T> *>(*it2);
	  int piece_preferred_order[N];
	  size_t prev_stride = 0;
	  for(int i = 0; i < N; i++) {
	    size_t best_stride = size_t(-1);
	    for(int j = 0; j < N; j++) {
	      if(affine->strides[j] < prev_stride) continue;
	      if(affine->strides[j] >= best_stride) continue;
	      // make sure each dimension with the same stride appears once
	      if((i > 0) && (affine->strides[j] == prev_stride) &&
		 (j <= piece_preferred_order[i-1])) continue;
	      piece_preferred_order[i] = j;
	      best_stride = affine->strides[j];
	    }
	    assert(best_stride < size_t(-1));
	    prev_stride = best_stride;
	  }
	  // log_dma.print() << "order: " << *affine << " -> "
	  // 		  << piece_preferred_order[0] << ", "
	  // 		  << ((N > 1) ? piece_preferred_order[1] : -1) << ", "
	  // 		  << ((N > 2) ? piece_preferred_order[2] : -1);
	  if(have_ordering) {
	    if(memcmp(dim_order, piece_preferred_order, N * sizeof(int)) != 0) {
	      force_fortran_order = true;
	      break;
	    }
	  } else {
	    memcpy(dim_order, piece_preferred_order, N * sizeof(int));
	    have_ordering = true;
	  }
	}
      }
    }
    if(!have_ordering || force_fortran_order)
      for(int i = 0; i < N; i++) dim_order[i] = i;
    return new TransferIteratorIndexSpace<N,T>(is, inst, dim_order, fields, extra_elems);
  }
  
  template <int N, typename T>
  void TransferDomainIndexSpace<N,T>::print(std::ostream& os) const
  {
    os << is;
  }

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferDomain, TransferDomainIndexSpace<N,T> > TransferDomainIndexSpace<N,T>::serdez_subclass;

  template <int N, typename T>
  template <typename S>
  inline bool TransferDomainIndexSpace<N,T>::serialize(S& serializer) const
  {
    return (serializer << is);
  }

  template <int N, typename T>
  inline /*static*/ TransferDomain *TransferDomain::construct(const IndexSpace<N,T>& is)
  {
    return new TransferDomainIndexSpace<N,T>(is);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferPlan
  //

  TransferPlan::TransferPlan(void)
  {}

  TransferPlan::~TransferPlan(void)
  {}

  class TransferPlanCopy : public TransferPlan {
  public:
    TransferPlanCopy(OASByInst *_oas_by_inst);
    virtual ~TransferPlanCopy(void);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority);

  protected:
    OASByInst *oas_by_inst;
  };

  TransferPlanCopy::TransferPlanCopy(OASByInst *_oas_by_inst)
    : oas_by_inst(_oas_by_inst)
  {}

  TransferPlanCopy::~TransferPlanCopy(void)
  {
    delete oas_by_inst;
  }

  static NodeID select_dma_node(Memory src_mem, Memory dst_mem,
				       ReductionOpID redop_id, bool red_fold)
  {
    NodeID src_node = ID(src_mem).memory_owner_node();
    NodeID dst_node = ID(dst_mem).memory_owner_node();

    bool src_is_rdma = get_runtime()->get_memory_impl(src_mem)->kind == MemoryImpl::MKIND_GLOBAL;
    bool dst_is_rdma = get_runtime()->get_memory_impl(dst_mem)->kind == MemoryImpl::MKIND_GLOBAL;

    if(src_is_rdma) {
      if(dst_is_rdma) {
	// gasnet -> gasnet - blech
	log_dma.warning("WARNING: gasnet->gasnet copy being serialized on local node (%d)", my_node_id);
	return my_node_id;
      } else {
	// gathers by the receiver
	return dst_node;
      }
    } else {
      if(dst_is_rdma) {
	// writing to gasnet is also best done by the sender
	return src_node;
      } else {
	// if neither side is gasnet, favor the sender (which may be the same as the target)
	return src_node;
      }
    }
  }

  Event TransferPlanCopy::execute_plan(const TransferDomain *td,
				       const ProfilingRequestSet& requests,
				       Event wait_on, int priority)
  {
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event ev = finish_event->current_event();

#if 0
    int priority = 0;
    if (get_runtime()->get_memory_impl(src_mem)->kind == MemoryImpl::MKIND_GPUFB)
      priority = 1;
    else if (get_runtime()->get_memory_impl(dst_mem)->kind == MemoryImpl::MKIND_GPUFB)
      priority = 1;
#endif

    // ask which node should perform the copy
    Memory src_mem, dst_mem;
    {
      assert(!oas_by_inst->empty());
      OASByInst::const_iterator it = oas_by_inst->begin();
      src_mem = it->first.first.get_location();
      dst_mem = it->first.second.get_location();
    }
    NodeID dma_node = select_dma_node(src_mem, dst_mem, 0, false);
    log_dma.debug() << "copy: srcmem=" << src_mem << " dstmem=" << dst_mem
		    << " node=" << dma_node;

    CopyRequest *r = new CopyRequest(td, oas_by_inst, 
				     wait_on,
				     finish_event, ID(ev).event_generation(),
				     priority, requests);
    // we've given oas_by_inst to the copyrequest, so forget it
    assert(oas_by_inst != 0);
    oas_by_inst = 0;

    if(dma_node == my_node_id) {
      log_dma.debug("performing copy on local node");

      get_runtime()->optable.add_local_operation(ev, r);
      r->check_readiness(false, dma_queue);
    } else {
      r->forward_request(dma_node);
      get_runtime()->optable.add_remote_operation(ev, dma_node);
      // done with the local copy of the request
      r->remove_reference();
    }

    return ev;
  }

  class TransferPlanReduce : public TransferPlan {
  public:
    TransferPlanReduce(const CopySrcDstField& _src,
		       const CopySrcDstField& _dst,
		       ReductionOpID _redop_id, bool _red_fold);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority);

  protected:
    CopySrcDstField src;
    CopySrcDstField dst;
    ReductionOpID redop_id;
    bool red_fold;
  };

  TransferPlanReduce::TransferPlanReduce(const CopySrcDstField& _src,
					 const CopySrcDstField& _dst,
					 ReductionOpID _redop_id, bool _red_fold)
    : src(_src)
    , dst(_dst)
    , redop_id(_redop_id)
    , red_fold(_red_fold)
  {}

  Event TransferPlanReduce::execute_plan(const TransferDomain *td,
					 const ProfilingRequestSet& requests,
					 Event wait_on, int priority)
  {
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event ev = finish_event->current_event();

    // TODO
    bool inst_lock_needed = false;

    ReduceRequest *r = new ReduceRequest(td,
					 std::vector<CopySrcDstField>(1, src),
					 dst,
					 inst_lock_needed,
					 redop_id, red_fold,
					 wait_on,
					 finish_event,
					 ID(ev).event_generation(),
					 0 /*priority*/, requests);

    NodeID src_node = ID(src.inst).instance_owner_node();
    if(src_node == my_node_id) {
      log_dma.debug("performing reduction on local node");

      get_runtime()->optable.add_local_operation(ev, r);	  
      r->check_readiness(false, dma_queue);
    } else {
      r->forward_request(src_node);
      get_runtime()->optable.add_remote_operation(ev, src_node);
      // done with the local copy of the request
      r->remove_reference();
    }
    return ev;
  }

  class TransferPlanFill : public TransferPlan {
  public:
    TransferPlanFill(const void *_data, size_t _size,
		     RegionInstance _inst, FieldID _field_id);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority);

  protected:
    ByteArray data;
    RegionInstance inst;
    FieldID field_id;
  };

  TransferPlanFill::TransferPlanFill(const void *_data, size_t _size,
				     RegionInstance _inst, FieldID _field_id)
    : data(_data, _size)
    , inst(_inst)
    , field_id(_field_id)
  {}

  Event TransferPlanFill::execute_plan(const TransferDomain *td,
				       const ProfilingRequestSet& requests,
				       Event wait_on, int priority)
  {
    CopySrcDstField f;
    f.inst = inst;
    f.field_id = field_id;
    f.subfield_offset = 0;
    f.size = data.size();

    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event ev = finish_event->current_event();
    FillRequest *r = new FillRequest(td, f, data.base(), data.size(),
				     wait_on,
				     finish_event,
				     ID(ev).event_generation(),
				     priority, requests);

    NodeID tgt_node = ID(inst).instance_owner_node();
    if(tgt_node == my_node_id) {
      get_runtime()->optable.add_local_operation(ev, r);
      r->check_readiness(false, dma_queue);
    } else {
      r->forward_request(tgt_node);
      get_runtime()->optable.add_remote_operation(ev, tgt_node);
      // release local copy of operation
      r->remove_reference();
    }

    return ev;
  }

  template <int N, typename T>
  Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField>& srcs,
			      const std::vector<CopySrcDstField>& dsts,
			      const std::vector<const typename CopyIndirection<N,T>::Base *> &indirects,
			      const Realm::ProfilingRequestSet &requests,
			      Event wait_on) const
  {
    TransferDomain *td = TransferDomain::construct(*this);
    std::vector<TransferPlan *> plans;
    assert(srcs.size() == dsts.size());
    for(size_t i = 0; i < srcs.size(); i++) {
      assert(srcs[i].size == dsts[i].size);
      assert(srcs[i].indirect_index == -1);
      assert(dsts[i].indirect_index == -1);

      // if the source field id is -1 and dst has no redop, we can use old fill
      if(srcs[i].field_id == FieldID(-1)) {
	// no support for reduction fill yet
	assert(dsts[i].redop_id == 0);
	TransferPlan *p = new TransferPlanFill(((srcs[i].size <= srcs[i].MAX_DIRECT_SIZE) ?
						  &(srcs[i].fill_data.direct) :
						  srcs[i].fill_data.indirect),
					       srcs[i].size,
					       dsts[i].inst,
					       dsts[i].field_id);
	plans.push_back(p);
	continue;
      }

      // if the dst has a reduction op, do a reduce
      if(dsts[i].redop_id != 0) {
	TransferPlan *p = new TransferPlanReduce(srcs[i], dsts[i],
						 dsts[i].redop_id,
						 dsts[i].red_fold);
	plans.push_back(p);
	continue;
      }

      // per-field copy otherwise - TODO: re-merge fields in the same
      //  instance (as long as they don't have serdez active)
      OffsetsAndSize oas;
      oas.src_field_id = srcs[i].field_id;
      oas.dst_field_id = dsts[i].field_id;
      oas.src_subfield_offset = srcs[i].subfield_offset;
      oas.dst_subfield_offset = dsts[i].subfield_offset;
      oas.size = srcs[i].size;
      oas.serdez_id = srcs[i].serdez_id;
      OASByInst *oas_by_inst = new OASByInst;
      (*oas_by_inst)[InstPair(srcs[i].inst, dsts[i].inst)].push_back(oas);
      TransferPlanCopy *p = new TransferPlanCopy(oas_by_inst);
      plans.push_back(p);
    }
    // hack to eliminate duplicate profiling responses
    //assert(requests.empty() || (plans.size() == 1));
    ProfilingRequestSet empty_prs;
    const ProfilingRequestSet *prsptr = &requests;
    std::set<Event> finish_events;
    for(std::vector<TransferPlan *>::iterator it = plans.begin();
	it != plans.end();
	++it) {
      //Event e = (*it)->execute_plan(td, requests, wait_on, 0 /*priority*/);
      Event e = (*it)->execute_plan(td, *prsptr, wait_on, 0 /*priority*/);
      prsptr = &empty_prs;
      finish_events.insert(e);
      delete *it;
    }
    delete td;
    return Event::merge_events(finish_events);
  }

#define DOIT(N,T) \
  template Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField>&, \
				       const std::vector<CopySrcDstField>&, \
				       const std::vector<const CopyIndirection<N,T>::Base *>&, \
				       const ProfilingRequestSet&,	\
				       Event) const;			\
  template class TransferIteratorIndexSpace<N,T>; \
  template class TransferDomainIndexSpace<N,T>;
  FOREACH_NT(DOIT)

}; // namespace Realm
