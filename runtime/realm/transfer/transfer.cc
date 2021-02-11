/* Copyright 2021 Stanford University, NVIDIA Corporation
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
#include "realm/transfer/channel.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/mem_impl.h"
#include "realm/inst_layout.h"
#ifdef REALM_USE_HDF5
#include "realm/hdf5/hdf5_access.h"
#endif

#ifdef REALM_ON_WINDOWS
#include <basetsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace Realm {

  extern Logger log_dma;
  Logger log_xplan("xplan");

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

  void TransferIterator::set_indirect_input_port(XferDes *xd, int port_idx,
						 TransferIterator *inner_iter)
  {
    // should not be called
    assert(0);
  }
  
#ifdef REALM_USE_HDF5
  size_t TransferIterator::step_hdf5(size_t max_bytes, AddressInfoHDF5& info,
				     bool tentative /*= false*/)
  {
    // should never be called
    return 0;
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIteratorBase<N,T>
  //

  template <int N, typename T>
  class TransferIteratorBase : public TransferIterator {
  protected:
    TransferIteratorBase(void); // used by deserializer
  public:
    TransferIteratorBase(RegionInstance inst,
			 const int _dim_order[N]);

    virtual Event request_metadata(void);

    virtual void reset(void);
    virtual bool done(void);
    virtual size_t step(size_t max_bytes, AddressInfo& info,
			unsigned flags,
			bool tentative = false);
#ifdef REALM_USE_HDF5
    virtual size_t step_hdf5(size_t max_bytes, AddressInfoHDF5& info,
			     bool tentative = false);
#endif
    virtual void confirm_step(void);
    virtual void cancel_step(void);

    virtual bool get_addresses(AddressList &addrlist);

  protected:
    virtual bool get_next_rect(Rect<N,T>& r, FieldID& fid,
			       size_t& offset, size_t& fsize) = 0;
    
    bool have_rect, is_done;
    Rect<N,T> cur_rect;
    FieldID cur_field_id;
    size_t cur_field_offset, cur_field_size;
    Point<N,T> cur_point, next_point;
    bool carry;
    RegionInstanceImpl *inst_impl;
    const InstanceLayout<N,T> *inst_layout;
    bool tentative_valid;
    int dim_order[N];
  };

  template <int N, typename T>
  TransferIteratorBase<N,T>::TransferIteratorBase(RegionInstance inst,
						  const int _dim_order[N])
    : have_rect(false), is_done(false)
    , inst_layout(0)
    , tentative_valid(false)
  {
    inst_impl = get_runtime()->get_instance_impl(inst);

    if(_dim_order)
      for(int i = 0; i < N; i++) dim_order[i] = _dim_order[i];
    else
      for(int i = 0; i < N; i++) dim_order[i] = i;
  }

  template <int N, typename T>
  TransferIteratorBase<N,T>::TransferIteratorBase(void)
    : have_rect(false), is_done(false)
    , inst_impl(0)
    , inst_layout(0)
    , tentative_valid(false)
  {}

  template <int N, typename T>
  Event TransferIteratorBase<N,T>::request_metadata(void)
  {
    if(inst_impl && !inst_impl->metadata.is_valid())
      return inst_impl->request_metadata();
    else
      return Event::NO_EVENT;
  }

  template <int N, typename T>
  void TransferIteratorBase<N,T>::reset(void)
  {
    have_rect = false;
    is_done = false;
    tentative_valid = false;
  }

  template <int N, typename T>
  bool TransferIteratorBase<N,T>::done(void)
  {
    if(have_rect)
      return false;

    if(is_done)
      return true;

    // if we haven't fetched the layout, now's our last chance
    if(inst_layout == 0) {
      assert(inst_impl->metadata.is_valid());
      inst_layout = checked_cast<const InstanceLayout<N,T> *>(inst_impl->metadata.layout);
    }

    // try to get a new (non-empty) rectangle
    while(true) {
      if(get_next_rect(cur_rect, cur_field_id,
		       cur_field_offset, cur_field_size)) {
	//log_dma.print() << "got: " << cur_rect;
	if(!cur_rect.empty()) {
	  have_rect = true;
	  cur_point = cur_rect.lo;
	  return false;
	}
      } else {
	have_rect = false;
	return is_done;
      }
    }
  }

  template <int N, typename T>
  size_t TransferIteratorBase<N,T>::step(size_t max_bytes, AddressInfo& info,
					 unsigned flags,
					 bool tentative /*= false*/)
  {
    // check to see if we're done - if not, we'll have a valid rectangle
    if(done() || !have_rect)
      return 0;

    assert(!tentative_valid);

    // find the layout piece the current point is in
    const InstanceLayoutPiece<N,T> *layout_piece;
    size_t field_rel_offset;
    size_t total_bytes = 0;
    {
      std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(cur_field_id);
      assert(it != inst_layout->fields.end());
      assert((cur_field_offset + cur_field_size) <= size_t(it->second.size_in_bytes));
      const InstancePieceList<N,T>& piece_list = inst_layout->piece_lists[it->second.list_idx];
      layout_piece = piece_list.find_piece(cur_point);
      assert(layout_piece != 0);
      field_rel_offset = it->second.rel_offset + cur_field_offset;
      //log_dma.print() << "F " << field_idx << " " << fields[field_idx] << " : " << it->second.list_idx << " " << field_rel_offset << " " << field_size;
    }

    size_t max_elems = max_bytes / cur_field_size;
    // less than one element?  give up immediately
    if(max_elems == 0)
      return 0;

    // the subrectangle we give always starts with the current point
    Rect<N,T> target_subrect;
    target_subrect.lo = cur_point;
    if(layout_piece->layout_type == PieceLayoutTypes::AffineLayoutType) {
      const AffineLayoutPiece<N,T> *affine = static_cast<const AffineLayoutPiece<N,T> *>(layout_piece);

      // using the current point, find the biggest subrectangle we want to try
      //  giving out, paying attention to the piece's bounds, where we've stopped,
      //  and piece strides
      int cur_dim = 0;
      int max_dims = (((flags & LINES_OK) == 0)  ? 1 :
		      ((flags & PLANES_OK) == 0) ? 2 :
		                                   3);
      ssize_t act_counts[3], act_strides[3];
      act_counts[0] = cur_field_size;
      act_strides[0] = 1;
      total_bytes = cur_field_size;
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
	   (cur_point[d] < cur_rect.hi[d]) &&
	   ((ssize_t)affine->strides[d] != (act_counts[cur_dim] * act_strides[cur_dim]))) {
	  cur_dim++;
	  if(cur_dim < max_dims)
	    act_strides[cur_dim] = (ssize_t)affine->strides[d];
	}
	if(cur_dim < max_dims) {
	  size_t len = cur_rect.hi[d] - cur_point[d] + 1;
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
	  if(cropped || (cur_point[d] > cur_rect.lo[d]))
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
	if(target_subrect.hi[d] == cur_rect.hi[d]) {
	  next_point[d] = cur_rect.lo[d];
	} else {
	  next_point[d] = target_subrect.hi[d] + 1;
	  carry = false;
	}
      } else
	next_point[d] = target_subrect.lo[d];
    }

    //log_dma.print() << "iter " << ((void *)this) << " " << field_idx << " " << cur_rect << " " << cur_point << " " << max_bytes << " : " << target_subrect << " " << next_point << " " << carry;

    if(tentative) {
      tentative_valid = true;
    } else {
      // if the "carry" propagated all the way through, go on to the next field
      //  (defer if tentative)
      if(carry) {
	have_rect = false;
      } else
	cur_point = next_point;
    }

    return total_bytes;
  }

#ifdef REALM_USE_HDF5
  template <int N, typename T>
  size_t TransferIteratorBase<N,T>::step_hdf5(size_t max_bytes,
					      AddressInfoHDF5& info,
					      bool tentative /*= false*/)
  {
    // check to see if we're done - if not, we'll have a valid rectangle
    if(done() || !have_rect)
      return 0;

    assert(!tentative_valid);

    // find the layout piece the current point is in
    const InstanceLayoutPiece<N,T> *layout_piece;
    //int field_rel_offset;
    {
      std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(cur_field_id);
      assert(it != inst_layout->fields.end());
      assert((cur_field_offset == 0) &&
	     (cur_field_size == size_t(it->second.size_in_bytes)) &&
	     "no support for accessing partial HDF5 fields yet");
      const InstancePieceList<N,T>& piece_list = inst_layout->piece_lists[it->second.list_idx];
      layout_piece = piece_list.find_piece(cur_point);
      assert(layout_piece != 0);
      //field_rel_offset = it->second.rel_offset;
      //log_dma.print() << "F " << field_idx << " " << fields[field_idx] << " : " << it->second.list_idx << " " << field_rel_offset << " " << field_size;
    }

    size_t max_elems = max_bytes / cur_field_size;
    // less than one element?  give up immediately
    if(max_elems == 0)
      return 0;

    // filename comes from the external resource info
    const ExternalHDF5Resource *res = checked_cast<ExternalHDF5Resource *>(inst_impl->metadata.ext_resource);

    // std::cout << "step " << this << " " << r << " " << p << " " << field_idx
    // 	      << " " << max_bytes << ":";

    // HDF5 requires we handle dimensions in order - no permutation allowed
    // using the current point, find the biggest subrectangle we want to try
    //  giving out
    Rect<N,T> target_subrect;
    size_t cur_bytes = 0;
    target_subrect.lo = cur_point;
    if(layout_piece->layout_type == PieceLayoutTypes::HDF5LayoutType) {
      const HDF5LayoutPiece<N,T> *hlp = static_cast<const HDF5LayoutPiece<N,T> *>(layout_piece);

      info.field_id = cur_field_id;
      info.filename = &res->filename;
      info.dsetname = &hlp->dsetname;

      bool grow = true;
      cur_bytes = cur_field_size;
      // follow the agreed-upon dimension ordering
      for(int di = 0; di < N; di++) {
	int d = dim_order[di];

	if(grow) {
	  size_t len = cur_rect.hi[d] - cur_point[d] + 1;
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
	  if(cur_point[d] > cur_rect.lo[d])
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
	if(target_subrect.hi[d] == cur_rect.hi[d]) {
	  next_point[d] = cur_rect.lo[d];
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
	have_rect = false;
      } else
	cur_point = next_point;
    }

    return cur_bytes;
  }
#endif
  
  template <int N, typename T>
  void TransferIteratorBase<N,T>::confirm_step(void)
  {
    assert(tentative_valid);
    if(carry) {
      have_rect = false;
    } else
      cur_point = next_point;
    tentative_valid = false;
  }

  template <int N, typename T>
  void TransferIteratorBase<N,T>::cancel_step(void)
  {
    assert(tentative_valid);
    tentative_valid = false;
  }

  template <int N, typename T>
  bool TransferIteratorBase<N,T>::get_addresses(AddressList &addrlist)
  {
#ifdef DEBUG_REALM
    assert(!tentative_valid);
#endif

    while(!done()) {
      if(!have_rect)
	return false; // no more addresses at the moment, but expect more later

      // we may be able to compact dimensions, but ask for space to write a
      //  an address record of the maximum possible dimension (i.e. N)
      size_t *addr_data = addrlist.begin_nd_entry(N);
      if(!addr_data)
	return true; // out of space for now

      // find the layout piece the current point is in
      const InstanceLayoutPiece<N,T> *layout_piece;
      size_t field_rel_offset;
      {
	std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(cur_field_id);
	assert(it != inst_layout->fields.end());
	assert((cur_field_offset + cur_field_size) <= size_t(it->second.size_in_bytes));
	const InstancePieceList<N,T>& piece_list = inst_layout->piece_lists[it->second.list_idx];
	layout_piece = piece_list.find_piece(cur_point);
	assert(layout_piece != 0);
	field_rel_offset = it->second.rel_offset + cur_field_offset;
      }

      // figure out the largest iteration-consistent subrectangle that fits in
      //  the current piece
      Rect<N,T> target_subrect;
      target_subrect.lo = cur_point;
      target_subrect.hi = cur_point;
      have_rect = false;  // tentatively clear - we'll (re-)set it below if needed
      for(int di = 0; di < N; di++) {
	int d = dim_order[di];

	// our target subrect in this dimension can be trimmed at the front by
	//  having already done a partial step, or trimmed at the end by the layout
	if(cur_rect.hi[d] <= layout_piece->bounds.hi[d]) {
	  if(cur_point[d] == cur_rect.lo[d]) {
	    // simple case - we are at the start in this dimension and the piece
	    //  covers the entire range
	    target_subrect.hi[d] = cur_rect.hi[d];
	    continue;
	  } else {
	    // we started in the middle, so we can finish this dimension, but
	    //  not continue to further dimensions
	    target_subrect.hi[d] = cur_rect.hi[d];
	    if(di < (N - 1)) {
	      // rewind the first di+1 dimensions and any after that that are
	      //  at the end
	      int d2 = 0;
	      while((d2 < N) &&
		    ((d2 <= di) ||
		     (cur_point[dim_order[d2]] == cur_rect.hi[dim_order[d2]]))) {
		cur_point[dim_order[d2]] = cur_rect.lo[dim_order[d2]];
		d2++;
	      }
	      if(d2 < N) {
		// carry didn't propagate all the way, so we have some left for
		//  next time
		cur_point[dim_order[d2]]++;
		have_rect = true;
	      }
	    }
	    break;
	  }
	} else {
	  // stopping short (doesn't matter where we started) - limit this subrect
	  //  based on the piece and start just past it in this dimension
	  //  (rewinding previous dimensions)
	  target_subrect.hi[d] = layout_piece->bounds.hi[d];
	  have_rect = true;
	  for(int d2 = 0; d2 < di; d2++)
	    cur_point[dim_order[d2]] = cur_rect.lo[dim_order[d2]];
	  cur_point[d] = layout_piece->bounds.hi[d] + 1;
	  break;
	}
      }
      //log_dma.print() << "step: cr=" << cur_rect << " bounds=" << layout_piece->bounds << " tgt=" << target_subrect << " next=" << cur_point << " (" << have_rect << ")";
#ifdef DEBUG_REALM
      assert(layout_piece->bounds.contains(target_subrect));
#endif

      if(layout_piece->layout_type == PieceLayoutTypes::AffineLayoutType) {
	const AffineLayoutPiece<N,T> *affine = static_cast<const AffineLayoutPiece<N,T> *>(layout_piece);

	// offset of initial entry is easy to compute
	addr_data[1] = (inst_impl->metadata.inst_offset +
			affine->offset +
			affine->strides.dot(target_subrect.lo) +
			field_rel_offset);

	size_t bytes = cur_field_size;
	int cur_dim = 1;
	int di = 0;
	// compact any dimensions that are contiguous first
	for(; di < N; di++) {
	  // follow the agreed-upon dimension ordering
	  int d = dim_order[di];

	  // skip degenerate dimensions
	  if(target_subrect.lo[d] == target_subrect.hi[d])
	    continue;

	  // if the stride doesn't match the current size, stop
	  if(affine->strides[d] != bytes)
	    break;

	  // it's contiguous - multiply total bytes by extent and continue
	  bytes *= (target_subrect.hi[d] - target_subrect.lo[d] + 1);
	}

	// if any dimensions are left, they need to become count/stride pairs
	size_t total_bytes = bytes;
	while(di < N) {
	  size_t total_count = 1;
	  size_t stride = affine->strides[dim_order[di]];

	  for(; di < N; di++) {
	    int d = dim_order[di];

	    if(target_subrect.lo[d] == target_subrect.hi[d])
	      continue;

	    size_t count = (target_subrect.hi[d] - target_subrect.lo[d] + 1);

	    if(affine->strides[d] != (stride * total_count))
	      break;

	    total_count *= count;
	  }

	  addr_data[cur_dim * 2] = total_count;
	  addr_data[cur_dim * 2 + 1] = stride;
	  total_bytes *= total_count;
	  cur_dim++;
	}

	// now that we know the compacted dimension, we can finish the address
	//  record
	addr_data[0] = (bytes << 4) + cur_dim;
	addrlist.commit_nd_entry(cur_dim, total_bytes);
      } else {
	assert(0 && "no support for non-affine pieces yet");
      }
    }

    return true; // we have no more addresses to produce
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIteratorIndexSpace<N,T>
  //

  template <int N, typename T>
  class TransferIteratorIndexSpace : public TransferIteratorBase<N,T> {
  protected:
    TransferIteratorIndexSpace(void); // used by deserializer
  public:
    TransferIteratorIndexSpace(const IndexSpace<N,T> &_is,
			       RegionInstance inst,
			       const int _dim_order[N],
			       const std::vector<FieldID>& _fields,
			       const std::vector<size_t>& _fld_offsets,
			       const std::vector<size_t>& _fld_sizes,
			       size_t _extra_elems);

    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);
      
    virtual ~TransferIteratorIndexSpace(void);

    virtual Event request_metadata(void);

    virtual void reset(void);

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndexSpace<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    virtual bool get_next_rect(Rect<N,T>& r, FieldID& fid,
			       size_t& offset, size_t& fsize);
    
    IndexSpace<N,T> is;
    IndexSpaceIterator<N,T> iter;
    bool iter_init_deferred;
    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;
    size_t field_idx;
    size_t extra_elems;
  };

  template <int N, typename T>
  TransferIteratorIndexSpace<N,T>::TransferIteratorIndexSpace(const IndexSpace<N,T>& _is,
							      RegionInstance inst,
							      const int _dim_order[N],
							      const std::vector<FieldID>& _fields,
							      const std::vector<size_t>& _fld_offsets,
							      const std::vector<size_t>& _fld_sizes,
							      size_t _extra_elems)
    : TransferIteratorBase<N,T>(inst, _dim_order)
    , is(_is)
    , field_idx(0), extra_elems(_extra_elems)
  {
    if(is.is_valid()) {
      iter.reset(is);
      this->is_done = !iter.valid;
      iter_init_deferred = false;
    } else
      iter_init_deferred = true;

    // special case - skip a lot of the init if we know the space is empty
    if(iter_init_deferred || iter.valid) {
      fields = _fields;
      fld_offsets = _fld_offsets;
      fld_sizes = _fld_sizes;
    }
  }

  template <int N, typename T>
  TransferIteratorIndexSpace<N,T>::TransferIteratorIndexSpace(void)
    : iter_init_deferred(false)
    , field_idx(0)
  {}

  template <int N, typename T>
  template <typename S>
  /*static*/ TransferIterator *TransferIteratorIndexSpace<N,T>::deserialize_new(S& deserializer)
  {
    IndexSpace<N,T> is;
    RegionInstance inst;
    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;
    size_t extra_elems;
    int dim_order[N];

    if(!((deserializer >> is) &&
	 (deserializer >> inst) &&
	 (deserializer >> fields) &&
	 (deserializer >> fld_offsets) &&
	 (deserializer >> fld_sizes) &&
	 (deserializer >> extra_elems)))
      return 0;

    for(int i = 0; i < N; i++)
      if(!(deserializer >> dim_order[i]))
	return 0;

    TransferIteratorIndexSpace<N,T> *tiis = new TransferIteratorIndexSpace<N,T>(is,
										inst,
										dim_order,
										fields,
										fld_offsets,
										fld_sizes,
										extra_elems);

    return tiis;
  }

  template <int N, typename T>
  TransferIteratorIndexSpace<N,T>::~TransferIteratorIndexSpace(void)
  {}

  template <int N, typename T>
  Event TransferIteratorIndexSpace<N,T>::request_metadata(void)
  {
    Event e = TransferIteratorBase<N,T>::request_metadata();;

    if(iter_init_deferred)
      e = Event::merge_events(e, is.make_valid());

    return e;
  }

  template <int N, typename T>
  void TransferIteratorIndexSpace<N,T>::reset(void)
  {
    TransferIteratorBase<N,T>::reset();
    field_idx = 0;
    assert(!iter_init_deferred);
    iter.reset(iter.space);
    this->is_done = !iter.valid;
  }

  template <int N, typename T>
  bool TransferIteratorIndexSpace<N,T>::get_next_rect(Rect<N,T>& r,
						      FieldID& fid,
						      size_t& offset,
						      size_t& fsize)
  {
    if(iter_init_deferred) {
      // index space must be valid now (i.e. somebody should have waited)
      assert(is.is_valid());
      iter.reset(is);
      iter_init_deferred = false;
      if(!iter.valid) {
	this->is_done = true;
	return false;
      }
    }

    if(this->is_done)
      return false;

    r = iter.rect;
    fid = fields[field_idx];
    offset = fld_offsets[field_idx];
    fsize = fld_sizes[field_idx];

    iter.step();
    if(!iter.valid) {
      iter.reset(is);
      field_idx++;
      if(field_idx == fields.size())
	this->is_done = true;
    }
    return true;
  }

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndexSpace<N,T> > TransferIteratorIndexSpace<N,T>::serdez_subclass;

  template <int N, typename T>
  template <typename S>
  bool TransferIteratorIndexSpace<N,T>::serialize(S& serializer) const
  {
    if(!((serializer << iter.space) &&
	 (serializer << (this->inst_impl ? this->inst_impl->me :
			 RegionInstance::NO_INST)) &&
	 (serializer << fields) &&
	 (serializer << fld_offsets) &&
	 (serializer << fld_sizes) &&
	 (serializer << extra_elems)))
      return false;

    for(int i = 0; i < N; i++)
      if(!(serializer << this->dim_order[i]))
	return false;

    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIteratorIndirect<N,T>
  //

  template <int N, typename T>
  class TransferIteratorIndirect : public TransferIteratorBase<N,T> {
  protected:
    TransferIteratorIndirect(void); // used by deserializer
  public:
    TransferIteratorIndirect(Memory _addrs_mem,
			     //const IndexSpace<N,T> &_is,
			     RegionInstance inst,
			     //const int _dim_order[N],
			     const std::vector<FieldID>& _fields,
			     const std::vector<size_t>& _fld_offsets,
			     const std::vector<size_t>& _fld_sizes);

    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);
      
    virtual ~TransferIteratorIndirect(void);

    virtual Event request_metadata(void);

    // specify the xd port used for indirect address flow control, if any
    virtual void set_indirect_input_port(XferDes *xd, int port_idx,
					 TransferIterator *inner_iter);

    virtual void reset(void);

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndirect<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    virtual bool get_next_rect(Rect<N,T>& r, FieldID& fid,
			       size_t& offset, size_t& fsize);
    
    TransferIterator *addrs_in;
    Memory addrs_mem;
    intptr_t addrs_mem_base;
    //IndexSpace<N,T> is;
    bool can_merge;
    static const size_t MAX_POINTS = 64;
    Point<N,T> points[MAX_POINTS];
    size_t point_pos, num_points;
    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;
    XferDes *indirect_xd;
    int indirect_port_idx;
  };

  template <int N, typename T>
  TransferIteratorIndirect<N,T>::TransferIteratorIndirect(void)
    : can_merge(true)
    , point_pos(0), num_points(0)
  {}
  
  template <int N, typename T>
  TransferIteratorIndirect<N,T>::TransferIteratorIndirect(Memory _addrs_mem,
							  //const IndexSpace<N,T> &_is,
							  RegionInstance inst,
							  //const int _dim_order[N],
							  const std::vector<FieldID>& _fields,
							  const std::vector<size_t>& _fld_offsets,
							  const std::vector<size_t>& _fld_sizes)
    : TransferIteratorBase<N,T>(inst, 0)
    , addrs_in(0)
    , addrs_mem(_addrs_mem)
    , addrs_mem_base(0)
    , point_pos(0)
    , num_points(0)
      //, is(_is)
    , fields(_fields)
    , fld_offsets(_fld_offsets)
    , fld_sizes(_fld_sizes)
    , indirect_xd(0)
    , indirect_port_idx(-1)
  {}
    
  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndirect<N,T> > TransferIteratorIndirect<N,T>::serdez_subclass;

  template <int N, typename T>
  template <typename S>
  bool TransferIteratorIndirect<N,T>::serialize(S& serializer) const
  {
    return ((serializer << addrs_mem) &&
	    (serializer << this->inst_impl->me) &&
	    (serializer << fields) &&
	    (serializer << fld_offsets) &&
	    (serializer << fld_sizes));
  }

  template <int N, typename T>
  template <typename S>
  /*static*/ TransferIterator *TransferIteratorIndirect<N,T>::deserialize_new(S& deserializer)
  {
    Memory addrs_mem;
    RegionInstance inst;
    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;

    if(!((deserializer >> addrs_mem) &&
	 (deserializer >> inst) &&
	 (deserializer >> fields) &&
	 (deserializer >> fld_offsets) &&
	 (deserializer >> fld_sizes)))
      return 0;

    return new TransferIteratorIndirect<N,T>(addrs_mem,
					     inst,
					     fields,
					     fld_offsets,
					     fld_sizes);
  }
      
  template <int N, typename T>
  TransferIteratorIndirect<N,T>::~TransferIteratorIndirect(void)
  {
  }

  template <int N, typename T>
  Event TransferIteratorIndirect<N,T>::request_metadata(void)
  {
    Event ev1 = addrs_in->request_metadata();
    //Event ev2 = is.make_valid();
    Event ev3 = TransferIteratorBase<N,T>::request_metadata();
    return Event::merge_events(ev1, /*ev2,*/ ev3);
  }

  template <int N, typename T>
  void TransferIteratorIndirect<N,T>::set_indirect_input_port(XferDes *xd,
							      int port_idx,
							      TransferIterator *inner_iter)
  {
    indirect_xd = xd;
    indirect_port_idx = port_idx;
    addrs_in = inner_iter;

    assert(indirect_xd != 0);
    assert(indirect_port_idx >= 0);
    void *mem_base = indirect_xd->input_ports[indirect_port_idx].mem->get_direct_ptr(0, 0);
    assert(mem_base != 0);
    addrs_mem_base = reinterpret_cast<intptr_t>(mem_base);
  }
  
  template <int N, typename T>
  void TransferIteratorIndirect<N,T>::reset(void)
  {
    TransferIteratorBase<N,T>::reset();
    addrs_in->reset();
  }
  
  template <int N, typename T>
  bool TransferIteratorIndirect<N,T>::get_next_rect(Rect<N,T>& r,
						    FieldID& fid,
						    size_t& offset,
						    size_t& fsize)
  {
    assert(fields.size() == 1);
    fid = fields[0];
    offset = fld_offsets[0];
    fsize = fld_sizes[0];

    bool nonempty = false;
    int merge_dim = -1;
    while(true) {
      // do we need new addresses?
      if(point_pos >= num_points) {
	// get the next address
	if(addrs_in->done()) {
	  this->is_done = true;
	  return nonempty;
	}
	TransferIterator::AddressInfo a_info;
	size_t addr_max_bytes = sizeof(Point<N,T>) * MAX_POINTS;
	if(indirect_xd != 0) {
	  XferDes::XferPort& iip = indirect_xd->input_ports[indirect_port_idx];
	  if(iip.peer_guid != XferDes::XFERDES_NO_GUID) {
	    addr_max_bytes = iip.seq_remote.span_exists(iip.local_bytes_total,
							addr_max_bytes);
	    if(addr_max_bytes == 0) {
	      // end of data?
	      if(iip.remote_bytes_total.load() == iip.local_bytes_total)
		this->is_done = true;
	      return nonempty;
	    }
	  }
	}
	size_t amt = addrs_in->step(addr_max_bytes, a_info, 0,
				    false /*!tentative*/);
	if(amt == 0)
	  return nonempty;
	point_pos = 0;
	num_points = amt / sizeof(Point<N,T>);
	assert(amt == (num_points * sizeof(Point<N,T>)));

	memcpy(points,
	       reinterpret_cast<const void *>(addrs_mem_base +
					      a_info.base_offset),
	       amt);
	//log_dma.print() << "got points: " << points[0] << "(+" << (num_points - 1) << ")";
	if(indirect_xd != 0) {
	  XferDes::XferPort& iip = indirect_xd->input_ports[indirect_port_idx];
	  indirect_xd->update_bytes_read(indirect_port_idx,
					 iip.local_bytes_total, amt);
	  iip.local_bytes_total += amt;
	}
      }

      while(point_pos < num_points) {
	const Point<N,T>& p = points[point_pos];
      
	if(nonempty) {
	  // attempt merge
	  if(merge_dim >= 0) {
	    // we've already chosen which dimension we can merge along
	    for(int i = 0; i < N; i++)
	      if(p[i] != (r.hi[i] + ((i == merge_dim) ? 1 : 0))) {
		// merge fails - return what we have
		return true;
	      }
	    // if we fall through, merging is ok
	    r.hi = p;
	    point_pos++;
	  } else {
	    for(int i = 0; i < N; i++) {
	      if(p[i] == r.hi[i]) continue;
	      if(p[i] == (r.hi[i] + 1)) {
		if(merge_dim == -1) {
		  merge_dim = i;
		  continue;
		} else {
		  merge_dim = -1;
		  break;
		}
	      }
	      // not mergeable
	      merge_dim = -1;
	      break;
	    }
	    if(merge_dim >= 0) {
	      // merge and continue
	      r.hi = p;
	      point_pos++;
	    } else {
	      return true;
	    }
	  }
	} else {
	  r = Rect<N,T>(p, p);
	  point_pos++;
	  nonempty = true;
	}
      }
    }
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIteratorIndirectRange<N,T>
  //

  template <int N, typename T>
  class TransferIteratorIndirectRange : public TransferIteratorBase<N,T> {
  protected:
    TransferIteratorIndirectRange(void); // used by deserializer
  public:
    TransferIteratorIndirectRange(Memory _addrs_mem,
				  //const IndexSpace<N,T> &_is,
				  RegionInstance inst,
				  //const int _dim_order[N],
				  const std::vector<FieldID>& _fields,
				  const std::vector<size_t>& _fld_offsets,
				  const std::vector<size_t>& _fld_sizes);

    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);
      
    virtual ~TransferIteratorIndirectRange(void);

    virtual Event request_metadata(void);

    // specify the xd port used for indirect address flow control, if any
    virtual void set_indirect_input_port(XferDes *xd, int port_idx,
					 TransferIterator *inner_iter);

    virtual void reset(void);

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndirectRange<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    virtual bool get_next_rect(Rect<N,T>& r, FieldID& fid,
			       size_t& offset, size_t& fsize);
    
    TransferIterator *addrs_in;
    Memory addrs_mem;
    intptr_t addrs_mem_base;
    //IndexSpace<N,T> is;
    bool can_merge;
    static const size_t MAX_RECTS = 64;
    Rect<N,T> rects[MAX_RECTS];
    size_t rect_pos, num_rects;
    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;
    XferDes *indirect_xd;
    int indirect_port_idx;
  };

  template <int N, typename T>
  TransferIteratorIndirectRange<N,T>::TransferIteratorIndirectRange(void)
    : can_merge(true)
    , rect_pos(0), num_rects(0)
  {}
  
  template <int N, typename T>
  TransferIteratorIndirectRange<N,T>::TransferIteratorIndirectRange(Memory _addrs_mem,
							  //const IndexSpace<N,T> &_is,
							  RegionInstance inst,
							  //const int _dim_order[N],
								    const std::vector<FieldID>& _fields,
								    const std::vector<size_t>& _fld_offsets,
								    const std::vector<size_t>& _fld_sizes)
    : TransferIteratorBase<N,T>(inst, 0)
    , addrs_in(0)
    , addrs_mem(_addrs_mem)
    , addrs_mem_base(0)
    , can_merge(true)
    , rect_pos(0)
    , num_rects(0)
      //, is(_is)
    , fields(_fields)
    , fld_offsets(_fld_offsets)
    , fld_sizes(_fld_sizes)
    , indirect_xd(0)
    , indirect_port_idx(-1)
  {}
    
  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndirectRange<N,T> > TransferIteratorIndirectRange<N,T>::serdez_subclass;

  template <int N, typename T>
  template <typename S>
  bool TransferIteratorIndirectRange<N,T>::serialize(S& serializer) const
  {
    return ((serializer << addrs_mem) &&
	    (serializer << this->inst_impl->me) &&
	    (serializer << fields) &&
	    (serializer << fld_offsets) &&
	    (serializer << fld_sizes));
  }

  template <int N, typename T>
  template <typename S>
  /*static*/ TransferIterator *TransferIteratorIndirectRange<N,T>::deserialize_new(S& deserializer)
  {
    Memory addrs_mem;
    RegionInstance inst;
    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;

    if(!((deserializer >> addrs_mem) &&
	 (deserializer >> inst) &&
	 (deserializer >> fields) &&
	 (deserializer >> fld_offsets) &&
	 (deserializer >> fld_sizes)))
      return 0;

    return new TransferIteratorIndirectRange<N,T>(addrs_mem,
						  inst,
						  fields,
						  fld_offsets,
						  fld_sizes);
  }
      
  template <int N, typename T>
  TransferIteratorIndirectRange<N,T>::~TransferIteratorIndirectRange(void)
  {
  }

  template <int N, typename T>
  Event TransferIteratorIndirectRange<N,T>::request_metadata(void)
  {
    Event ev1 = addrs_in->request_metadata();
    //Event ev2 = is.make_valid();
    Event ev3 = TransferIteratorBase<N,T>::request_metadata();
    return Event::merge_events(ev1, /*ev2,*/ ev3);
  }

  template <int N, typename T>
  void TransferIteratorIndirectRange<N,T>::set_indirect_input_port(XferDes *xd,
							      int port_idx,
							      TransferIterator *inner_iter)
  {
    indirect_xd = xd;
    indirect_port_idx = port_idx;
    addrs_in = inner_iter;

    assert(indirect_xd != 0);
    assert(indirect_port_idx >= 0);
    void *mem_base = indirect_xd->input_ports[indirect_port_idx].mem->get_direct_ptr(0, 0);
    assert(mem_base != 0);
    addrs_mem_base = reinterpret_cast<intptr_t>(mem_base);
  }
  
  template <int N, typename T>
  void TransferIteratorIndirectRange<N,T>::reset(void)
  {
    TransferIteratorBase<N,T>::reset();
    addrs_in->reset();
  }
  
  template <int N, typename T>
  bool TransferIteratorIndirectRange<N,T>::get_next_rect(Rect<N,T>& r,
							 FieldID& fid,
							 size_t& offset,
							 size_t& fsize)
  {
    assert(fields.size() == 1);
    fid = fields[0];
    offset = fld_offsets[0];
    fsize = fld_sizes[0];

    bool nonempty = false;
    while(true) {
      // do we need new addresses?
      if(rect_pos >= num_rects) {
	// get the next address
	if(addrs_in->done()) {
	  this->is_done = true;
	  return nonempty;
	}
	TransferIterator::AddressInfo a_info;
	size_t addr_max_bytes = sizeof(Rect<N,T>) * MAX_RECTS;
	if(indirect_xd != 0) {
	  XferDes::XferPort& iip = indirect_xd->input_ports[indirect_port_idx];
	  if(iip.peer_guid != XferDes::XFERDES_NO_GUID) {
	    addr_max_bytes = iip.seq_remote.span_exists(iip.local_bytes_total,
							addr_max_bytes);
	    if(addr_max_bytes == 0) {
	      // end of data?
	      if(iip.remote_bytes_total.load() == iip.local_bytes_total)
	      	this->is_done = true;
	      return nonempty;
	    }
	  }
	}
	size_t amt = addrs_in->step(addr_max_bytes, a_info, 0,
				    false /*!tentative*/);
	if(amt == 0)
	  return nonempty;
	rect_pos = 0;
	num_rects = amt / sizeof(Rect<N,T>);
	assert(amt == (num_rects * sizeof(Rect<N,T>)));

	memcpy(rects,
	       reinterpret_cast<const void *>(addrs_mem_base +
					      a_info.base_offset),
	       amt);
	//log_dma.print() << "got rects: " << rects[0] << "(+" << (num_rects - 1) << ")";
	if(indirect_xd != 0) {
	  XferDes::XferPort& iip = indirect_xd->input_ports[indirect_port_idx];
	  indirect_xd->update_bytes_read(indirect_port_idx,
					 iip.local_bytes_total, amt);
	  iip.local_bytes_total += amt;
	}
      }

      // scan through the rectangles for something nonempty
      while(rect_pos < num_rects) {
	if(rects[rect_pos].empty()) {
	  rect_pos++;
	  continue;
	}

	if(nonempty) {
	  // attempt merge
	  int merge_dim = -1;
	  if(N == 1) {
	    // simple 1-D case
	    if(rects[rect_pos].lo.x == (r.hi.x + 1)) {
	      merge_dim = 0;
	    }
	  } else {
	    const Rect<N,T>& r2 = rects[rect_pos];
	    int dims_match = 0;
	    while(dims_match < (N-1))
	      if((r.lo[dims_match] == r2.lo[dims_match]) &&
		 (r.hi[dims_match] == r2.hi[dims_match]))
		dims_match++;
	      else
		break;
	    if((r2.lo[dims_match] == (r.hi[dims_match] + 1))) {
	      merge_dim = dims_match;  // unless checks below fail
	      // rest of dims must be degenerate and match
	      for(int i = dims_match + 1; i < N; i++)
		if((r.lo[i] != r.hi[i]) ||
		   (r2.lo[i] != r.lo[i]) || (r2.hi[i] != r.hi[i])) {
		  merge_dim = -1;
		  break;
		}
	    }
	  }
	  if(merge_dim >= 0) {
	    // merge and continue
	    r.hi[merge_dim] = rects[rect_pos++].hi[merge_dim];
	  } else {
	    // can't merge - return what we've got
	    return true;
	  }
	} else {
	  r = rects[rect_pos++];
	  nonempty = true;
	}
      }
    }
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

    virtual bool empty(void) const;
    virtual size_t volume(void) const;

    virtual void choose_dim_order(std::vector<int>& dim_order,
				  const std::vector<CopySrcDstField>& srcs,
				  const std::vector<CopySrcDstField>& dsts,
				  const std::vector<IndirectionInfo *>& indirects,
				  bool force_fortran_order,
				  size_t max_stride) const;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      const std::vector<int>& dim_order,
					      const std::vector<FieldID>& fields,
					      const std::vector<size_t>& fld_offsets,
					      const std::vector<size_t>& fld_sizes) const;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      RegionInstance peer,
					      const std::vector<FieldID>& fields,
					      const std::vector<size_t>& fld_offsets,
					      const std::vector<size_t>& fld_sizes) const;

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
  bool TransferDomainIndexSpace<N,T>::empty(void) const
  {
    return is.empty();
  }

  template <int N, typename T>
  size_t TransferDomainIndexSpace<N,T>::volume(void) const
  {
    return is.volume();
  }

  // determines the preferred dim order for an affine layout - dims sorted
  //  by stride, ignoring dimensions that are trivial or strides > max_stride
  template <int N, typename T>
  static void preferred_dim_order(std::vector<int>& dim_order,
				  const AffineLayoutPiece<N,T> *affine,
				  const std::vector<bool>& trivial,
				  size_t max_stride)
  {
    size_t prev_stride = 0;
    for(int i = 0; i < N; i++) {
      size_t best_stride = max_stride+1;
      int best_dim = -1;
      for(int j = 0; j < N; j++) {
	// ignore strides we've already seen, those that are too big, and
	//  those that correspond to trivial dimensions
	if(trivial[j] ||
	   (affine->strides[j] <= prev_stride) ||
	   (affine->strides[j] >= best_stride))
	  continue;
	best_dim = j;
	best_stride = affine->strides[j];
      }
      if(best_dim >= 0) {
	dim_order.push_back(best_dim);
	prev_stride = best_stride;
      } else {
	break;
      }
    }
  }

  // combines a new dim order preference with existing decisions, choosing
  //  the new order if it's a superset of the existing order
  // TODO: come up with rules for being willing to throw away the existing
  //  order if they conflict
  static void reconcile_dim_orders(std::vector<int>& existing_order,
				   const std::vector<int>& new_order)
  {
    // a shorter order can't be a (strict) superset
    if(new_order.size() <= existing_order.size()) return;
    // differences in the common length are deal-breakers too
    if(!std::equal(existing_order.begin(), existing_order.end(),
		   new_order.begin())) return;
    existing_order = new_order;
  }

  template <int N, typename T>
  static void preferred_dim_order(std::vector<int>& dim_order,
				  const Rect<N,T>& bounds,
				  RegionInstance inst,
				  FieldID field_id,
				  const std::vector<bool>& trivial,
				  size_t max_stride)
  {
    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(inst);
    // can't wait for it here - make sure it's valid before calling
    assert(impl->metadata.is_valid());
    const InstanceLayout<N,T> *layout = checked_cast<const InstanceLayout<N,T> *>(impl->metadata.layout);
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    assert(it != layout->fields.end());
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    std::vector<int> preferred;
    preferred.reserve(N);
    for(typename std::vector<InstanceLayoutPiece<N,T> *>::const_iterator it2 = ipl.pieces.begin();
	it2 != ipl.pieces.end();
	++it2) {
      // ignore pieces that aren't affine or are outside bounds
      if((*it2)->layout_type != PieceLayoutTypes::AffineLayoutType)
	continue;
      if(!bounds.overlaps((*it2)->bounds))
	continue;
      preferred_dim_order(preferred,
			  checked_cast<AffineLayoutPiece<N,T> *>(*it2),
			  trivial,
			  max_stride);
      reconcile_dim_orders(dim_order, preferred);
      preferred.clear();
    }
  }

  template <int N, typename T>
  void TransferDomainIndexSpace<N,T>::choose_dim_order(std::vector<int>& dim_order,
				  const std::vector<CopySrcDstField>& srcs,
				  const std::vector<CopySrcDstField>& dsts,
				  const std::vector<IndirectionInfo *>& indirects,
				  bool force_fortran_order,
				  size_t max_stride) const
  {
    if(force_fortran_order) {
      dim_order.resize(N);
      for(int i = 0; i < N; i++)
	dim_order[i] = i;
      return;
    }

    // start with an unconstrained order
    dim_order.clear();
    dim_order.reserve(N);

    // dimensions with an extent==1 are trivial and should not factor into
    //  order decision
    std::vector<bool> trivial(N);
    for(int i = 0; i < N; i++)
      trivial[i] = (is.bounds.lo[i] == is.bounds.hi[i]);

    // allocate this vector once and we'll reuse it
    std::vector<int> preferred;
    preferred.reserve(N);

    // consider destinations first
    for(size_t i = 0; i < dsts.size(); i++) {
      if((dsts[i].field_id != FieldID(-1)) && dsts[i].inst.exists()) {
	preferred_dim_order(preferred,
			    is.bounds, dsts[i].inst, dsts[i].field_id,
			    trivial, max_stride);
	reconcile_dim_orders(dim_order, preferred);
	preferred.clear();
	continue;
      }

      // TODO: ask opinion of indirections?
    }

    // then consider sources
    for(size_t i = 0; i < srcs.size(); i++) {
      if((srcs[i].field_id != FieldID(-1)) && srcs[i].inst.exists()) {
	preferred_dim_order(preferred,
			    is.bounds, srcs[i].inst, srcs[i].field_id,
			    trivial, max_stride);
	reconcile_dim_orders(dim_order, preferred);
	preferred.clear();
	continue;
      }

      // TODO: ask opinion of indirections?
    }

    // if we didn't end up choosing all the dimensions, add the rest back in
    //  in arbitrary (ascending, currently) order
    if(dim_order.size() != N) {
      std::vector<bool> present(N, false);
      for(size_t i = 0; i < dim_order.size(); i++)
	present[dim_order[i]] = true;
      for(int i = 0; i < N; i++)
	if(!present[i])
	  dim_order.push_back(i);
#ifdef DEBUG_REALM
      assert(dim_order.size() == N);
#endif
    }
  }

  template <int N, typename T>
  TransferIterator *TransferDomainIndexSpace<N,T>::create_iterator(RegionInstance inst,
								   const std::vector<int>& dim_order,
								   const std::vector<FieldID>& fields,
								   const std::vector<size_t>& fld_offsets,
								   const std::vector<size_t>& fld_sizes) const
  {
    size_t extra_elems = 0;
    assert(dim_order.size() == N);
    return new TransferIteratorIndexSpace<N,T>(is, inst, dim_order.data(),
					       fields, fld_offsets, fld_sizes,
					       extra_elems);
  }

  template <int N, typename T>
  TransferIterator *TransferDomainIndexSpace<N,T>::create_iterator(RegionInstance inst,
								   RegionInstance peer,
								   const std::vector<FieldID>& fields,
								   const std::vector<size_t>& fld_offsets,
								   const std::vector<size_t>& fld_sizes) const
  {
    std::vector<int> dim_order(N, -1);
    bool have_ordering = false;
    bool force_fortran_order = true;//false;
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
	  if((*it2)->layout_type != PieceLayoutTypes::AffineLayoutType) {
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
	    if(memcmp(dim_order.data(), piece_preferred_order, N * sizeof(int)) != 0) {
	      force_fortran_order = true;
	      break;
	    }
	  } else {
	    memcpy(dim_order.data(), piece_preferred_order, N * sizeof(int));
	    have_ordering = true;
	  }
	}
      }
    }
    if(!have_ordering || force_fortran_order)
      for(int i = 0; i < N; i++) dim_order[i] = i;
    return this->create_iterator(inst, dim_order,
				 fields, fld_offsets, fld_sizes);
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
  // class IndirectionInfo
  //

  std::ostream& operator<<(std::ostream& os, const IndirectionInfo& ii)
  {
    ii.print(os);
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressSplitChannel
  //

  static AddressSplitChannel *local_addrsplit_channel = 0;

  AddressSplitChannel::AddressSplitChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<AddressSplitChannel, AddressSplitXferDesBase>(bgwork,
								     XFER_ADDR_SPLIT,
								     "address split")
  {
    assert(!local_addrsplit_channel);
    local_addrsplit_channel = this;
  }

  AddressSplitChannel::~AddressSplitChannel()
  {
    assert(local_addrsplit_channel == this);
    local_addrsplit_channel = 0;
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDesCreateMessage<AddressSplitXferDes<N,T>>
  //

  template <int N, typename T>
  class AddressSplitXferDes;
  
  template <int N, typename T>
  struct AddressSplitXferDesCreateMessage : public XferDesCreateMessageBase {
  public:
    static void handle_message(NodeID sender,
			       const AddressSplitXferDesCreateMessage<N,T> &args,
			       const void *msgdata,
			       size_t msglen)
    {
      std::vector<XferDesPortInfo> inputs_info, outputs_info;
      int priority = 0;
      size_t element_size;
      std::vector<IndexSpace<N,T> > spaces;

      Realm::Serialization::FixedBufferDeserializer fbd(msgdata, msglen);

      bool ok = ((fbd >> inputs_info) &&
		 (fbd >> outputs_info) &&
		 (fbd >> priority) &&
		 (fbd >> element_size) &&
		 (fbd >> spaces));
      assert(ok);
      assert(fbd.bytes_left() == 0);
  
      //assert(!args.inst.exists());
      assert(local_addrsplit_channel);
      XferDes *xd = new AddressSplitXferDes<N,T>(args.dma_op,
						 local_addrsplit_channel,
						 args.launch_node,
						 args.guid,
						 inputs_info,
						 outputs_info,
						 priority,
						 element_size,
						 spaces);

      local_addrsplit_channel->enqueue_ready_xd(xd);
    }
  };
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressSplitXferDesFactory<N,T>
  //

  template <int N, typename T>
  class AddressSplitXferDesFactory : public XferDesFactory {
  public:
    AddressSplitXferDesFactory(size_t _bytes_per_element,
			       const std::vector<IndexSpace<N,T> >& _spaces);

  protected:
    virtual ~AddressSplitXferDesFactory();

  public:
    virtual void release();

    virtual void create_xfer_des(uintptr_t dma_op,
				 NodeID launch_node,
				 NodeID target_node,
				 XferDesID guid,
				 const std::vector<XferDesPortInfo>& inputs_info,
				 const std::vector<XferDesPortInfo>& outputs_info,
				 int priority,
				 XferDesRedopInfo redop_info,
				 const void *fill_data, size_t fill_size);

    static ActiveMessageHandlerReg<AddressSplitXferDesCreateMessage<N,T> > areg;

  protected:
    size_t bytes_per_element;
    std::vector<IndexSpace<N,T> > spaces;
  };

  template <int N, typename T>
  /*static*/ ActiveMessageHandlerReg<AddressSplitXferDesCreateMessage<N,T> > AddressSplitXferDesFactory<N,T>::areg;
  
  template <int N, typename T>
  class AddressSplitXferDes : public AddressSplitXferDesBase {
  protected:
    friend class AddressSplitXferDesFactory<N,T>;
    friend struct AddressSplitXferDesCreateMessage<N,T>;
    
    AddressSplitXferDes(uintptr_t _dma_op, Channel *_channel,
			NodeID _launch_node, XferDesID _guid,
			const std::vector<XferDesPortInfo>& inputs_info,
			const std::vector<XferDesPortInfo>& outputs_info,
			int _priority,
			size_t _element_size,
			const std::vector<IndexSpace<N,T> >& _spaces);

  public:
    ~AddressSplitXferDes();

    virtual Event request_metadata();

    virtual bool progress_xd(AddressSplitChannel *channel, TimeLimit work_until);

  protected:
    int find_point_in_spaces(Point<N,T> p, int guess_idx) const;
    
    std::vector<IndexSpace<N,T> > spaces;
    size_t element_size;
    static const size_t MAX_POINTS = 64;
    size_t point_index, point_count;
    Point<N,T> points[MAX_POINTS];
    int output_space_id;
    unsigned output_count;
  };

  template <int N, typename T>
  AddressSplitXferDesFactory<N,T>::AddressSplitXferDesFactory(size_t _bytes_per_element,
							      const std::vector<IndexSpace<N,T> >& _spaces)
    : bytes_per_element(_bytes_per_element)
    , spaces(_spaces)
  {}

  template <int N, typename T>
  AddressSplitXferDesFactory<N,T>::~AddressSplitXferDesFactory()
  {}
    
  template <int N, typename T>
  void AddressSplitXferDesFactory<N,T>::release()
  {
    delete this;
  }

  template <int N, typename T>
  void AddressSplitXferDesFactory<N,T>::create_xfer_des(uintptr_t dma_op,
							NodeID launch_node,
							NodeID target_node,
							XferDesID guid,
							const std::vector<XferDesPortInfo>& inputs_info,
							const std::vector<XferDesPortInfo>& outputs_info,
							int priority,
							XferDesRedopInfo redop_info,
							const void *fill_data, size_t fill_size)
  {
    assert(redop_info.id == 0);
    assert(fill_size == 0);
    if(target_node == Network::my_node_id) {
      // local creation
      //assert(!inst.exists());
      assert(local_addrsplit_channel);
      XferDes *xd = new AddressSplitXferDes<N,T>(dma_op,
						 local_addrsplit_channel,
						 launch_node, guid,
						 inputs_info, outputs_info,
						 priority,
						 bytes_per_element,
						 spaces);

      local_addrsplit_channel->enqueue_ready_xd(xd);
    } else {
      // remote creation
      Serialization::ByteCountSerializer bcs;
      {
	bool ok = ((bcs << inputs_info) &&
		   (bcs << outputs_info) &&
		   (bcs << priority) &&
		   (bcs << bytes_per_element) &&
		   (bcs << spaces));
	assert(ok);
      }
      size_t req_size = bcs.bytes_used();
      ActiveMessage<AddressSplitXferDesCreateMessage<N,T> > amsg(target_node, req_size);
      //amsg->inst = inst;
      amsg->launch_node = launch_node;
      amsg->guid = guid;
      amsg->dma_op = dma_op;
      {
	bool ok = ((amsg << inputs_info) &&
		   (amsg << outputs_info) &&
		   (amsg << priority) &&
		   (amsg << bytes_per_element) &&
		   (amsg << spaces));
	assert(ok);
      }
      amsg.commit();
    }
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressSplitXferDes<N,T>
  //

  AddressSplitXferDesBase::AddressSplitXferDesBase(uintptr_t _dma_op,
						   Channel *_channel,
						   NodeID _launch_node, XferDesID _guid,
						   const std::vector<XferDesPortInfo>& inputs_info,
						   const std::vector<XferDesPortInfo>& outputs_info,
						   int _priority)
    : XferDes(_dma_op, _channel, _launch_node, _guid,
	      inputs_info, outputs_info,
	      _priority, 0, 0)
  {}

  long AddressSplitXferDesBase::get_requests(Request** requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }
  
  void AddressSplitXferDesBase::notify_request_read_done(Request* req)
  {
    // unused
    assert(0);
  }
  
  void AddressSplitXferDesBase::notify_request_write_done(Request* req)
  {
    // unused
    assert(0);
  }
  
  void AddressSplitXferDesBase::flush()
  {
    // do nothing
  }

  template <int N, typename T>
  AddressSplitXferDes<N,T>::AddressSplitXferDes(uintptr_t _dma_op, Channel *_channel,
						NodeID _launch_node, XferDesID _guid,
						const std::vector<XferDesPortInfo>& inputs_info,
						const std::vector<XferDesPortInfo>& outputs_info,
						int _priority,
						size_t _element_size,
						const std::vector<IndexSpace<N,T> >& _spaces)
    : AddressSplitXferDesBase(_dma_op, _channel, _launch_node, _guid,
			      inputs_info, outputs_info,
			      _priority)
    , spaces(_spaces)
    , element_size(_element_size)
    , point_index(0), point_count(0)
    , output_space_id(-1), output_count(0)
  {}

  template <int N, typename T>
  AddressSplitXferDes<N,T>::~AddressSplitXferDes()
  {}

  template <int N, typename T>
  int AddressSplitXferDes<N,T>::find_point_in_spaces(Point<N,T> p, int guess_idx) const
  {
    // try the guessed (e.g. same as previous hit) space first
    if(guess_idx >= 0)
      if(spaces[guess_idx].contains(p))
	return guess_idx;

    // try all the rest
    for(size_t i = 0; i < spaces.size(); i++)
      if(i != size_t(guess_idx))
	if(spaces[i].contains(p))
	  return i;

    return -1;
  }
    
  template <int N, typename T>
  Event AddressSplitXferDes<N,T>::request_metadata()
  {
    std::vector<Event> events;

    for(size_t i = 0; i < spaces.size(); i++) {
      Event e = spaces[i].make_valid();
      if(e.exists())
	events.push_back(e);
    }

    return Event::merge_events(events);
  }

  template <int N, typename T>
  bool AddressSplitXferDes<N,T>::progress_xd(AddressSplitChannel *channel,
					     TimeLimit work_until)
  {
    assert(!iteration_completed.load());

    ReadSequenceCache rseqcache(this);
    WriteSequenceCache wseqcache(this);

    bool did_work = false;
    while(true) {
      size_t output_bytes = 0;
      bool input_done = false;
      while(true) {
	// step 1: get some points if we are out
	if(point_index >= point_count) {
	  if(input_ports[0].iter->done()) {
	    input_done = true;
	    break;
	  }
	
	  TransferIterator::AddressInfo p_info;
	  size_t max_bytes = MAX_POINTS * sizeof(Point<N,T>);
	  if(input_ports[0].peer_guid != XFERDES_NO_GUID) {
	    max_bytes = input_ports[0].seq_remote.span_exists(input_ports[0].local_bytes_total, max_bytes);
	    if(max_bytes < sizeof(Point<N,T>))
	      break;
	  }
	  size_t bytes = input_ports[0].iter->step(max_bytes, p_info,
						   0, false /*!tentative*/);
	  if(bytes == 0) break;
	  point_count = bytes / sizeof(Point<N,T>);
	  assert(bytes == (point_count * sizeof(Point<N,T>)));
	  const void *srcptr = input_ports[0].mem->get_direct_ptr(p_info.base_offset,
								  bytes);
	  assert(srcptr != 0);
	  memcpy(points, srcptr, bytes);
	  point_index = 0;
	  rseqcache.add_span(0, input_ports[0].local_bytes_total, bytes);
	  input_ports[0].local_bytes_total += bytes;
	  did_work = true;
	}

	// step 2: process the first point we've got on hand
	int new_space_id = find_point_in_spaces(points[point_index],
						output_space_id);

	// can only extend an existing run with another point from the same
	//  space
	if(output_count == 0)
	  output_space_id = new_space_id;
	else
	  if(new_space_id != output_space_id)
	    break;

	// can't let our count overflow a 24-bit value
	if((((output_count + 1) * element_size) >> 24) > 0)
	  break;

	// if it matched a space, we have to emit the point to that space's
	//  output address stream before we can accept the point
	if(output_space_id != -1) {
	  XferPort &op = output_ports[output_space_id];
	  if(op.seq_remote.span_exists(op.local_bytes_total + output_bytes,
				       sizeof(Point<N,T>)) < sizeof(Point<N,T>))
	    break;
	  TransferIterator::AddressInfo o_info;
	  size_t bytes = op.iter->step(sizeof(Point<N,T>), o_info,
				       0, false /*!tentative*/);
	  assert(bytes == sizeof(Point<N,T>));
	  void *dstptr = op.mem->get_direct_ptr(o_info.base_offset,
						sizeof(Point<N,T>));
	  assert(dstptr != 0);
	  memcpy(dstptr, &points[point_index], sizeof(Point<N,T>));
	  output_bytes += sizeof(Point<N,T>);
	}
	output_count++;
	point_index++;
      }
        
      // if we wrote any points out, update their validity now
      if(output_bytes > 0) {
	assert(output_space_id >= 0);
	wseqcache.add_span(output_space_id,
			   output_ports[output_space_id].local_bytes_total,
			   output_bytes);
	output_ports[output_space_id].local_bytes_total += output_bytes;
	did_work = true;
      }

      // now try to write the control information
      if((output_count > 0) || input_done) {
	unsigned cword = (((output_count * element_size) << 8) +
			  (input_done ? 128 : 0) + // bit 7
			  (output_space_id + 1));
	assert(cword != 0);
      
	XferPort &cp = output_ports[spaces.size()];
	if(cp.seq_remote.span_exists(cp.local_bytes_total,
				     sizeof(unsigned)) < sizeof(unsigned))
	  break;  // no room to write control work

	TransferIterator::AddressInfo c_info;
	size_t bytes = cp.iter->step(sizeof(unsigned), c_info,
				     0, false /*!tentative*/);
	assert(bytes == sizeof(unsigned));
	void *dstptr = cp.mem->get_direct_ptr(c_info.base_offset, sizeof(unsigned));
	assert(dstptr != 0);
	memcpy(dstptr, &cword, sizeof(unsigned));

	if(input_done) {
	  iteration_completed.store_release(true);
	  // mark all address streams as done (dummy write update)
	  for(size_t i = 0; i < spaces.size(); i++)
	    wseqcache.add_span(i, output_ports[i].local_bytes_total, 0);
	}
	size_t old_lbt = cp.local_bytes_total;
	cp.local_bytes_total += sizeof(unsigned);
	wseqcache.add_span(spaces.size(), old_lbt, sizeof(unsigned));
	output_space_id = -1;
	output_count = 0;
	did_work = true;
      } else
	break;

      if(iteration_completed.load() || work_until.is_expired())
	break;
    }

    rseqcache.flush();
    wseqcache.flush();

    return did_work;
  }
  
  
  ////////////////////////////////////////////////////////////////////////
  //
  // class IndirectionInfoTyped<N,T,N2,T2>
  //

  template <int N, typename T, int N2, typename T2>
  class IndirectionInfoTyped : public IndirectionInfo {
  public:
    IndirectionInfoTyped(const IndexSpace<N,T>& is,
			 const typename CopyIndirection<N,T>::template Unstructured<N2,T2>& ind);

    virtual Event request_metadata(void);

    virtual void generate_gather_paths(Memory dst_mem,
				       TransferGraph::XDTemplate::IO dst_edge,
				       unsigned indirect_idx,
				       unsigned src_fld_start,
				       unsigned src_fld_count,
				       size_t bytes_per_element,
				       CustomSerdezID serdez_id,
				       std::vector<TransferGraph::XDTemplate>& xd_nodes,
				       std::vector<TransferGraph::IBInfo>& ib_edges,
				       std::vector<TransferDesc::FieldInfo>& src_fields);

    virtual void generate_scatter_paths(Memory src_mem,
					TransferGraph::XDTemplate::IO src_edge,
					unsigned indirect_idx,
					unsigned dst_fld_start,
					unsigned dst_fld_count,
					size_t bytes_per_element,
					CustomSerdezID serdez_id,
					std::vector<TransferGraph::XDTemplate>& xd_nodes,
					std::vector<TransferGraph::IBInfo>& ib_edges,
					std::vector<TransferDesc::FieldInfo>& src_fields);

    virtual RegionInstance get_pointer_instance(void) const;

    virtual TransferIterator *create_address_iterator(RegionInstance peer) const;

    virtual TransferIterator *create_indirect_iterator(Memory addrs_mem,
						       RegionInstance inst,
						       const std::vector<FieldID>& fields,
						       const std::vector<size_t>& fld_offsets,
						       const std::vector<size_t>& fld_sizes) const;

    virtual void print(std::ostream& os) const;

    IndexSpace<N,T> domain;
    bool structured;
    FieldID field_id;
    RegionInstance inst;
    bool is_ranges;
    bool oor_possible;
    bool aliasing_possible;
    size_t subfield_offset;
    std::vector<IndexSpace<N2,T2> > spaces;
    std::vector<RegionInstance> insts;
  };

  template <int N, typename T, int N2, typename T2>
  IndirectionInfoTyped<N,T,N2,T2>::IndirectionInfoTyped(const IndexSpace<N,T>& is,
							const typename CopyIndirection<N,T>::template Unstructured<N2,T2>& ind)
    : domain(is)
    , structured(false)
    , field_id(ind.field_id)
    , inst(ind.inst)
    , is_ranges(ind.is_ranges)
    , oor_possible(ind.oor_possible)
    , aliasing_possible(ind.aliasing_possible)
    , subfield_offset(ind.subfield_offset)
    , spaces(ind.spaces)
    , insts(ind.insts)
  {}

  template <int N, typename T, int N2, typename T2>
  Event IndirectionInfoTyped<N,T,N2,T2>::request_metadata(void)
  {
    std::set<Event> evs;

    {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(inst);
      Event e = impl->request_metadata();
      if(!e.has_triggered()) evs.insert(e);
    }

    for(std::vector<RegionInstance>::const_iterator it = insts.begin();
	it != insts.end();
	++it) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(*it);
      Event e = impl->request_metadata();
      if(!e.has_triggered()) evs.insert(e);
    }

    return Event::merge_events(evs);
  }

  static TransferGraph::XDTemplate::IO add_copy_path(std::vector<TransferGraph::XDTemplate>& xd_nodes,
						     std::vector<TransferGraph::IBInfo>& ib_edges,
						     TransferGraph::XDTemplate::IO start_edge,
						     const MemPathInfo& info)
  {
    size_t hops = info.xd_channels.size();

    if(hops == 0) {
      // no xd's needed at all - return input edge as output
      return start_edge;
    }

    size_t xd_base = xd_nodes.size();
    size_t ib_base = ib_edges.size();
    xd_nodes.resize(xd_base + hops);
    ib_edges.resize(ib_base + hops);
    for(size_t i = 0; i < hops; i++) {
      TransferGraph::XDTemplate& xdn = xd_nodes[xd_base + i];

      xdn.factory = info.xd_channels[i]->get_factory();
      xdn.gather_control_input = -1;
      xdn.scatter_control_input = -1;
      xdn.target_node = info.xd_channels[i]->node;
      xdn.inputs.resize(1);
      xdn.inputs[0] = ((i == 0) ?
		         start_edge :
		         TransferGraph::XDTemplate::mk_edge(ib_base + i - 1));
      //xdn.inputs[0].indirect_inst = RegionInstance::NO_INST;
      xdn.outputs.resize(1);
      xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_base + i);

      TransferGraph::IBInfo& ibe = ib_edges[ib_base + i];
      ibe.memory = info.path[i + 1];
      ibe.size = 65536; // TODO: pick size?
    }

    // last edge we created is the output
    return TransferGraph::XDTemplate::mk_edge(ib_base + hops - 1);
  }

  template <int N, typename T, int N2, typename T2>
  void IndirectionInfoTyped<N,T,N2,T2>::generate_gather_paths(Memory dst_mem,
							      TransferGraph::XDTemplate::IO dst_edge,
							      unsigned indirect_idx,
							      unsigned src_fld_start,
							      unsigned src_fld_count,
							      size_t bytes_per_element,
							      CustomSerdezID serdez_id,
							      std::vector<TransferGraph::XDTemplate>& xd_nodes,
							      std::vector<TransferGraph::IBInfo>& ib_edges,
							      std::vector<TransferDesc::FieldInfo>& src_fields)
  {
    // compute the paths from each src data instance and the dst instance
    std::vector<size_t> path_idx;
    std::vector<MemPathInfo> path_infos;
    path_idx.reserve(spaces.size());
    for(size_t i = 0; i < insts.size(); i++) {
      size_t idx = path_infos.size();
      for(size_t j = 0; j < i; j++)
	if(insts[i].get_location() == insts[j].get_location()) {
	  idx = path_idx[j];
	  break;
	}

      path_idx.push_back(idx);
      if(idx >= path_infos.size()) {
	// new path to compute
	path_infos.resize(idx + 1);
	bool ok = find_shortest_path(insts[i].get_location(),
				     dst_mem,
				     serdez_id,
                                     0 /*redop_id*/,
				     path_infos[idx]);
	assert(ok);
      }
    }

    // add a source field for the address
    unsigned addr_field_start = src_fields.size();
    size_t addr_size = (is_ranges ?
			  sizeof(Rect<N2,T2>) :
			  sizeof(Point<N2,T2>));
    src_fields.push_back(TransferDesc::FieldInfo { field_id, 0, addr_size, 0 });
    TransferGraph::XDTemplate::IO addr_edge =
      TransferGraph::XDTemplate::mk_inst(inst, addr_field_start, 1);

    // special case - a gather from a single source with no out of range
    //  accesses
    if((spaces.size() == 1) && !oor_possible) {
      size_t pathlen = path_infos[0].xd_channels.size();
      // HACK!
      Memory local_ib_mem = ID::make_ib_memory(path_infos[0].xd_channels[0]->node, 0).convert<Memory>();
      // do we have to do anything to get the addresses into a cpu-readable
      //  memory on that node?
      MemPathInfo addr_path;
      bool ok = find_shortest_path(inst.get_location(),
				   local_ib_mem,
				   0 /*no serdez*/,
                                   0 /*redop_id*/,
				   addr_path,
				   true /*skip_final_memcpy*/);
      assert(ok);
      addr_edge = add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path);

      size_t xd_idx = xd_nodes.size();
      size_t ib_idx = ib_edges.size();
      xd_nodes.resize(xd_idx + pathlen);
      ib_edges.resize(ib_idx + pathlen - 1);

      for(size_t i = 0; i < pathlen; i++) {
	TransferGraph::XDTemplate& xdn = xd_nodes[xd_idx + i];

	xdn.target_node = path_infos[0].xd_channels[i]->node;
	//xdn.kind = path_infos[0].xd_kinds[i];
	xdn.factory = path_infos[0].xd_channels[i]->get_factory();
	xdn.gather_control_input = -1;
	xdn.scatter_control_input = -1;
	if(i == 0) {
	  xdn.inputs.resize(2);
	  xdn.inputs[0] = TransferGraph::XDTemplate::mk_indirect(indirect_idx,
								 1,
								 insts[0],
								 src_fld_start,
								 src_fld_count);
	  xdn.inputs[1] = addr_edge;
	} else {
	  xdn.inputs.resize(1);
	  xdn.inputs[0] = TransferGraph::XDTemplate::mk_edge(ib_idx + i - 1);
	}
	if(i == (pathlen - 1)) {
	  xdn.outputs.resize(1);
	  xdn.outputs[0] = dst_edge;
	} else {
	  xdn.outputs.resize(1);
	  xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_idx + i);

	  TransferGraph::IBInfo& ibe = ib_edges[ib_idx + i];
	  ibe.memory = path_infos[0].path[i + 1];
	  ibe.size = 1 << 20; /*TODO*/
	}
      }
    } else {
      // step 1: we need the address decoder, possibly with some hops to get
      //  the data to where a cpu can look at it
      NodeID addr_node = ID(inst).instance_owner_node();
      // HACK!
      Memory addr_ib_mem = ID::make_ib_memory(addr_node, 0).convert<Memory>();
      MemPathInfo addr_path;
      bool ok = find_shortest_path(inst.get_location(),
				   addr_ib_mem,
				   0 /*no serdez*/,
                                   0 /*redop_id*/,
				   addr_path,
				   true /*skip_final_memcpy*/);
      assert(ok);
      addr_edge = add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path);

      std::vector<TransferGraph::XDTemplate::IO> decoded_addr_edges(spaces.size());
      TransferGraph::XDTemplate::IO ctrl_edge;
      {
	// instantiate decoder
	size_t xd_base = xd_nodes.size();
	size_t ib_base = ib_edges.size();
	xd_nodes.resize(xd_base + 1);
	ib_edges.resize(ib_base + spaces.size() + 1);

	TransferGraph::XDTemplate& xdn = xd_nodes[xd_base];
	xdn.target_node = addr_node;
	//xdn.kind = XFER_ADDR_SPLIT;
	assert(!is_ranges && "need range address splitter");
	xdn.factory = new AddressSplitXferDesFactory<N2,T2>(bytes_per_element,
							    spaces);
	xdn.gather_control_input = -1;
	xdn.scatter_control_input = -1;
	xdn.inputs.resize(1);
	xdn.inputs[0] = addr_edge;
	xdn.outputs.resize(spaces.size() + 1);
	for(size_t i = 0; i < spaces.size(); i++) {
	  xdn.outputs[i] = TransferGraph::XDTemplate::mk_edge(ib_base + i);
	  decoded_addr_edges[i] = xdn.outputs[i];
	  ib_edges[ib_base + i].memory = addr_ib_mem;
	  ib_edges[ib_base + i].size = 65536; // TODO
	}
	xdn.outputs[spaces.size()] = TransferGraph::XDTemplate::mk_edge(ib_base + spaces.size());
	ctrl_edge = xdn.outputs[spaces.size()];
	ib_edges[ib_base + spaces.size()].memory = addr_ib_mem;
	ib_edges[ib_base + spaces.size()].size = 65536; // TODO
      }

      // next, see what work we need to get the addresses to where the
      //  data instances live
      for(size_t i = 0; i < spaces.size(); i++) {
	// HACK!
	Memory src_ib_mem = ID::make_ib_memory(ID(insts[i]).instance_owner_node(), 0).convert<Memory>();
	if(src_ib_mem != addr_ib_mem) {
	  MemPathInfo path;
	  bool ok = find_shortest_path(addr_ib_mem, src_ib_mem,
				       0 /*no serdez*/,
                                       0 /*redop_id*/,
				       path);
	  assert(ok);
	  decoded_addr_edges[i] = add_copy_path(xd_nodes, ib_edges,
						decoded_addr_edges[i],
						path);
	}
      }

      // control information has to get to the merge at the end
      // HACK!
      NodeID dst_node = ID(dst_mem).memory_owner_node();
      Memory dst_ib_mem = ID::make_ib_memory(dst_node, 0).convert<Memory>();
      if(dst_ib_mem != addr_ib_mem) {
	MemPathInfo path;
	bool ok = find_shortest_path(addr_ib_mem, dst_ib_mem,
				     0 /*no serdez*/,
                                     0 /*redop_id*/,
				     path);
	assert(ok);
	ctrl_edge = add_copy_path(xd_nodes, ib_edges,
				  ctrl_edge,
				  path);
      }

      // next complication: if all the data paths don't use the same final
      //  step, we need to force them to go through an intermediate
      // also insist that the final step be owned by the destination node
      //  (i.e. the merging should not be done via rdma)
      Channel *last_channel = path_infos[0].xd_channels[path_infos[0].xd_channels.size() - 1];
      bool same_last_channel = true;
      for(size_t i = 0; i < path_infos.size(); i++) {
	if(path_infos[i].xd_channels[path_infos[i].xd_channels.size() - 1] !=
	   last_channel) {
	  same_last_channel = false;
	  break;
	}
      }
      if(!same_last_channel) {
	// figure out what the final kind will be (might not be the same as
	//  any of the current paths)
	MemPathInfo tail_path;
	bool ok = find_shortest_path(dst_ib_mem, dst_mem,
				     serdez_id, 0 /*redop_id*/, tail_path);
	assert(ok && (tail_path.xd_channels.size() == 1));
	last_channel = tail_path.xd_channels[0];
	// and fix any path that doesn't use that channel
	for(size_t i = 0; i < path_infos.size(); i++) {
	  if(path_infos[i].xd_channels[path_infos[i].xd_channels.size() - 1] ==
	     last_channel) continue;
	  //log_new_dma.print() << "fix " << i << " " << path_infos[i].path[0] << " -> " << dst_ib_mem;
	  bool ok = find_shortest_path(path_infos[i].path[0], dst_ib_mem,
				       0 /*no serdez*/, 0 /*redop_id*/, path_infos[i]);
	  assert(ok);
	  // append last step
	  path_infos[i].xd_channels.push_back(last_channel);
	  path_infos[i].path.push_back(dst_mem);
	  //path_infos[i].xd_target_nodes.push_back(ID(dst_mem).memory_owner_node());
	}
      }

      // now any data paths with more than one hop need all but the last hop
      //  added to the graph
      std::vector<TransferGraph::XDTemplate::IO> data_edges(spaces.size());
      for(size_t i = 0; i < spaces.size(); i++) {
	const MemPathInfo& mpi = path_infos[path_idx[i]];
	size_t hops = mpi.xd_channels.size() - 1;
	if(hops > 0) {
	  size_t xd_base = xd_nodes.size();
	  size_t ib_base = ib_edges.size();
	  xd_nodes.resize(xd_base + hops);
	  ib_edges.resize(ib_base + hops);
	  for(size_t j = 0; j < hops; j++) {
	    TransferGraph::XDTemplate& xdn = xd_nodes[xd_base + j];

	    xdn.factory = mpi.xd_channels[j]->get_factory();
	    xdn.gather_control_input = -1;
	    xdn.scatter_control_input = -1;
	    xdn.target_node = mpi.xd_channels[j]->node;
	    if(j == 0) {
	      xdn.inputs.resize(2);
	      xdn.inputs[0] = TransferGraph::XDTemplate::mk_indirect(indirect_idx,
								     1,
								     insts[i],
								     src_fld_start,
								     src_fld_count);
	      xdn.inputs[1] = decoded_addr_edges[i];
	    } else {
	      xdn.inputs.resize(1);
	      xdn.inputs[0] = TransferGraph::XDTemplate::mk_edge(ib_base + j - 1);
	    }
	    xdn.outputs.resize(1);
	    xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_base + j);
	    ib_edges[ib_base + j].memory = mpi.path[j + 1];
	    ib_edges[ib_base + j].size = 65536; // TODO: pick size?
	  }
	  data_edges[i] = TransferGraph::XDTemplate::mk_edge(ib_base + hops - 1);
	}
      }

      // and finally the last xd that merges the streams together
      size_t xd_idx = xd_nodes.size();
      xd_nodes.resize(xd_idx + 1);
      TransferGraph::XDTemplate& xdn = xd_nodes[xd_idx];

      xdn.target_node = ID(dst_mem).memory_owner_node();
      //xdn.kind = last_kind;
      xdn.factory = last_channel->get_factory();
      xdn.gather_control_input = spaces.size();
      xdn.scatter_control_input = -1;
      xdn.outputs.resize(1);
      xdn.outputs[0] = dst_edge;

      xdn.inputs.resize(spaces.size() + 1);

      for(size_t i = 0; i < spaces.size(); i++) {
	// can we read (indirectly) right from the source instance?
	if(path_infos[path_idx[i]].xd_channels.size() == 1) {
	  int ind_port = xdn.inputs.size();
	  xdn.inputs.resize(ind_port + 1);
	  xdn.inputs[i] = TransferGraph::XDTemplate::mk_indirect(indirect_idx,
								 ind_port,
								 insts[i],
								 src_fld_start,
								 src_fld_count);
	  xdn.inputs[ind_port] = decoded_addr_edges[i];
	} else {
	  //assert(data_edges[i] >= 0);
	  xdn.inputs[i] = data_edges[i];
	}
      }

      // control input
      xdn.inputs[spaces.size()] = ctrl_edge;
    }
  }
  
  template <int N, typename T, int N2, typename T2>
  void IndirectionInfoTyped<N,T,N2,T2>::generate_scatter_paths(Memory src_mem,
							       TransferGraph::XDTemplate::IO src_edge,
							       unsigned indirect_idx,
							       unsigned dst_fld_start,
							       unsigned dst_fld_count,
							       size_t bytes_per_element,
							       CustomSerdezID serdez_id,
							       std::vector<TransferGraph::XDTemplate>& xd_nodes,
							       std::vector<TransferGraph::IBInfo>& ib_edges,
							       std::vector<TransferDesc::FieldInfo>& src_fields)
  {
    // compute the paths from the src instance to each dst data instance
    std::vector<size_t> path_idx;
    std::vector<MemPathInfo> path_infos;
    path_idx.reserve(spaces.size());
    for(size_t i = 0; i < insts.size(); i++) {
      size_t idx = path_infos.size();
      for(size_t j = 0; j < i; j++)
	if(insts[i].get_location() == insts[j].get_location()) {
	  idx = path_idx[j];
	  break;
	}

      path_idx.push_back(idx);
      if(idx >= path_infos.size()) {
	// new path to compute
	path_infos.resize(idx + 1);
	bool ok = find_shortest_path(src_mem,
				     insts[i].get_location(),
				     serdez_id,
                                     0 /*redop_id*/,
				     path_infos[idx]);
	assert(ok);
      }
    }

    // add a source field for the address
    unsigned addr_field_start = src_fields.size();
    size_t addr_size = (is_ranges ?
			  sizeof(Rect<N2,T2>) :
			  sizeof(Point<N2,T2>));
    src_fields.push_back(TransferDesc::FieldInfo { field_id, 0, addr_size, 0 });
    TransferGraph::XDTemplate::IO addr_edge =
      TransferGraph::XDTemplate::mk_inst(inst, addr_field_start, 1);

    // special case - a scatter to a single destination with no out of
    //  range accesses
    if((spaces.size() == 1) && !oor_possible) {
      size_t pathlen = path_infos[0].xd_channels.size();
      // HACK!
      Memory local_ib_mem = ID::make_ib_memory(path_infos[0].xd_channels[pathlen - 1]->node, 0).convert<Memory>();
      // do we have to do anything to get the addresses into a cpu-readable
      //  memory on that node?
      MemPathInfo addr_path;
      bool ok = find_shortest_path(inst.get_location(),
				   local_ib_mem,
				   0 /*no serdez*/,
                                   0 /*redop_id*/,
				   addr_path,
				   true /*skip_final_memcpy*/);
      assert(ok);
      addr_edge = add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path);

      size_t xd_idx = xd_nodes.size();
      size_t ib_idx = ib_edges.size();
      xd_nodes.resize(xd_idx + pathlen);
      ib_edges.resize(ib_idx + pathlen - 1);

      for(size_t i = 0; i < pathlen; i++) {
	TransferGraph::XDTemplate& xdn = xd_nodes[xd_idx + i];

	xdn.target_node = path_infos[0].xd_channels[i]->node;
	//xdn.kind = path_infos[0].xd_kinds[i];
	xdn.factory = path_infos[0].xd_channels[i]->get_factory();
	xdn.gather_control_input = -1;
	xdn.scatter_control_input = -1;
	if(i == (pathlen - 1)) {
	  xdn.inputs.resize(2);
	  xdn.inputs[1] = addr_edge;
	} else {
	  xdn.inputs.resize(1);
	}
	xdn.inputs[0] = ((i == 0) ?
			   src_edge :
			   TransferGraph::XDTemplate::mk_edge(ib_idx + i - 1));

	xdn.outputs.resize(1);
	if(i == (pathlen - 1)) {
	  xdn.outputs[0] = TransferGraph::XDTemplate::mk_indirect(indirect_idx,
								  1,
								  insts[0],
								  dst_fld_start,
								  dst_fld_count);
	} else {
	  xdn.outputs.resize(1);
	  xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_idx + i);

	  TransferGraph::IBInfo& ibe = ib_edges[ib_idx + i];
	  ibe.memory = path_infos[0].path[i + 1];
	  ibe.size = 1 << 20; /*TODO*/
	}
      }
    } else {
      // step 1: we need the address decoder, possibly with some hops to get
      //  the data to where a cpu can look at it
      NodeID addr_node = ID(inst).instance_owner_node();
      // HACK!
      Memory addr_ib_mem = ID::make_ib_memory(addr_node, 0).convert<Memory>();
      MemPathInfo addr_path;
      bool ok = find_shortest_path(inst.get_location(),
				   addr_ib_mem,
				   0 /*no serdez*/,
                                   0 /*redop_id*/,
				   addr_path,
				   true /*skip_final_memcpy*/);
      assert(ok);
      addr_edge = add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path);

      std::vector<TransferGraph::XDTemplate::IO> decoded_addr_edges(spaces.size());
      TransferGraph::XDTemplate::IO ctrl_edge;
      {
	// instantiate decoder
	size_t xd_base = xd_nodes.size();
	size_t ib_base = ib_edges.size();
	xd_nodes.resize(xd_base + 1);
	ib_edges.resize(ib_base + spaces.size() + 1);

	TransferGraph::XDTemplate& xdn = xd_nodes[xd_base];
	xdn.target_node = addr_node;
	//xdn.kind = XFER_ADDR_SPLIT;
	assert(!is_ranges && "need range address splitter");
	xdn.factory = new AddressSplitXferDesFactory<N2,T2>(bytes_per_element,
							    spaces);
	xdn.gather_control_input = -1;
	xdn.scatter_control_input = -1;
	xdn.inputs.resize(1);
	xdn.inputs[0] = addr_edge;
	xdn.outputs.resize(spaces.size() + 1);
	for(size_t i = 0; i < spaces.size(); i++) {
	  xdn.outputs[i] = TransferGraph::XDTemplate::mk_edge(ib_base + i);
	  decoded_addr_edges[i] = xdn.outputs[i];
	  ib_edges[ib_base + i].memory = addr_ib_mem;
	  ib_edges[ib_base + i].size = 65536; // TODO
	}
	xdn.outputs[spaces.size()] = TransferGraph::XDTemplate::mk_edge(ib_base + spaces.size());
	ctrl_edge = xdn.outputs[spaces.size()];
	ib_edges[ib_base + spaces.size()].memory = addr_ib_mem;
	ib_edges[ib_base + spaces.size()].size = 65536; // TODO
      }

      // next, see what work we need to get the addresses to where the
      //  last step of each path is running
      for(size_t i = 0; i < spaces.size(); i++) {
	// HACK!
	NodeID dst_node = path_infos[path_idx[i]].xd_channels[path_infos[path_idx[i]].xd_channels.size() - 1]->node;
	Memory dst_ib_mem = ID::make_ib_memory(dst_node, 0).convert<Memory>();
	if(dst_ib_mem != addr_ib_mem) {
	  MemPathInfo path;
	  bool ok = find_shortest_path(addr_ib_mem, dst_ib_mem,
				       0 /*no serdez*/,
                                       0 /*redop_id*/,
				       path);
	  assert(ok);
	  decoded_addr_edges[i] = add_copy_path(xd_nodes, ib_edges,
						decoded_addr_edges[i],
						path);
	}
      }

      // control information has to get to the split at the start
      // HACK!
      Memory src_ib_mem = ID::make_ib_memory(ID(src_mem).memory_owner_node(), 0).convert<Memory>();
      if(src_ib_mem != addr_ib_mem) {
	MemPathInfo path;
	bool ok = find_shortest_path(addr_ib_mem, src_ib_mem,
				     0 /*no serdez*/,
                                     0 /*redop_id*/,
				     path);
	assert(ok);
	ctrl_edge = add_copy_path(xd_nodes, ib_edges,
				  ctrl_edge,
				  path);
      }

      // next complication: if all the data paths don't use the same first
      //  step, we need to force them to go through an intermediate
      Channel *first_channel = path_infos[0].xd_channels[0];
      bool same_first_channel = true;
      for(size_t i = 1; i < path_infos.size(); i++)
	if(path_infos[i].xd_channels[0] != first_channel) {
	  same_first_channel = false;
	  break;
	}
      if(!same_first_channel) {
	// figure out what the first channel will be (might not be the same as
	//  any of the current paths)
	MemPathInfo head_path;
	bool ok = find_shortest_path(src_mem, src_ib_mem,
				     serdez_id,
                                     0 /*redop_id*/,
                                     head_path);
	assert(ok && (head_path.xd_channels.size() == 1));
	first_channel = head_path.xd_channels[0];
	// and fix any path that doesn't use that channel
	for(size_t i = 0; i < path_infos.size(); i++) {
	  if(path_infos[i].xd_channels[0] == first_channel) continue;

	  bool ok = find_shortest_path(src_ib_mem,
				       path_infos[i].path[path_infos[i].path.size() - 1],
				       0 /*no serdez*/,
                                       0 /*redop_id*/,
                                       path_infos[i]);
	  assert(ok);
	  // prepend last step
	  path_infos[i].xd_channels.insert(path_infos[i].xd_channels.begin(),
					   first_channel);
	  path_infos[i].path.insert(path_infos[i].path.begin(), src_mem);
	  //path_infos[i].xd_target_nodes.insert(path_infos[i].xd_target_nodes.begin(),
	  //				       ID(src_mem).memory_owner_node());
	}
      }

      // next comes the xd that reads the source and splits the data into
      //  the various output streams
      std::vector<TransferGraph::XDTemplate::IO> data_edges(spaces.size());
      {
	size_t xd_idx = xd_nodes.size();
	xd_nodes.resize(xd_idx + 1);

	TransferGraph::XDTemplate& xdn = xd_nodes[xd_idx];
	xdn.target_node = ID(src_mem).memory_owner_node();
	//xdn.kind = first_kind;
	xdn.factory = first_channel->get_factory();
	xdn.gather_control_input = -1;
	xdn.scatter_control_input = 1;
	xdn.inputs.resize(2);
	xdn.inputs[0] = src_edge;
	xdn.inputs[1] = ctrl_edge;

	xdn.outputs.resize(spaces.size());

	for(size_t i = 0; i < spaces.size(); i++) {
	  // can we write (indirectly) right into the dest instance?
	  if(path_infos[path_idx[i]].xd_channels.size() == 1) {
	    int ind_port = xdn.inputs.size();
	    xdn.inputs.resize(ind_port + 1);
	    xdn.outputs[i] = TransferGraph::XDTemplate::mk_indirect(indirect_idx,
								    ind_port,
								    insts[i],
								    dst_fld_start,
								    dst_fld_count);
	    xdn.inputs[ind_port] = decoded_addr_edges[i];
	  } else {
	    // need an ib to write to
	    size_t ib_idx = ib_edges.size();
	    ib_edges.resize(ib_idx + 1);
	    data_edges[i] = TransferGraph::XDTemplate::mk_edge(ib_idx);
	    xdn.outputs[i] = data_edges[i];
	    ib_edges[ib_idx].memory = path_infos[path_idx[i]].path[1];
	    ib_edges[ib_idx].size = 65536; // TODO: pick size?
	  }
	}
      }

      // finally, any data paths with more than one hop need the rest of
      //  their path added to the graph
      for(size_t i = 0; i < spaces.size(); i++) {
	const MemPathInfo& mpi = path_infos[path_idx[i]];
	size_t hops = mpi.xd_channels.size() - 1;
	if(hops > 0) {
	  //assert(data_edges[i] >= 0);
	  
	  size_t xd_base = xd_nodes.size();
	  size_t ib_base = ib_edges.size();
	  xd_nodes.resize(xd_base + hops);
	  ib_edges.resize(ib_base + hops - 1);
	  for(size_t j = 0; j < hops; j++) {
	    TransferGraph::XDTemplate& xdn = xd_nodes[xd_base + j];

	    xdn.factory = mpi.xd_channels[j + 1]->get_factory();
	    xdn.gather_control_input = -1;
	    xdn.scatter_control_input = -1;
	    xdn.target_node = mpi.xd_channels[j + 1]->node;
	    if(j < (hops - 1)) {
	      xdn.inputs.resize(1);
	      xdn.inputs[0] = ((j == 0) ?
			         data_edges[i] :
			         TransferGraph::XDTemplate::mk_edge(ib_base + j - 1));
	      xdn.outputs.resize(1);
	      xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_base + j);
	      ib_edges[ib_base + j].memory = mpi.path[j + 2];
	      ib_edges[ib_base + j].size = 65536; // TODO: pick size?
	    } else {
	      // last hop uses the address stream
	      xdn.inputs.resize(2);
	      xdn.inputs[0] = ((j == 0) ?
			         data_edges[i] :
			         TransferGraph::XDTemplate::mk_edge(ib_base + j - 1));
	      xdn.inputs[1] = decoded_addr_edges[i];
	      xdn.outputs.resize(1);
	      xdn.outputs[0] = TransferGraph::XDTemplate::mk_indirect(indirect_idx,
								      1,
								      insts[i],
								      dst_fld_start,
								      dst_fld_count);
	    }
	  }
	}
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  RegionInstance IndirectionInfoTyped<N,T,N2,T2>::get_pointer_instance(void) const
  {
    return inst;
  }
  
  template <int N, typename T, int N2, typename T2>
  TransferIterator *IndirectionInfoTyped<N,T,N2,T2>::create_address_iterator(RegionInstance peer) const
  {
    TransferDomainIndexSpace<N,T> tdis(domain);
    std::vector<FieldID> fields(1, field_id);
    std::vector<size_t> fld_offsets(1, 0);
    std::vector<size_t> fld_sizes(1, (is_ranges ? sizeof(Rect<N2,T2>) :
				                  sizeof(Point<N2,T2>)));
    return tdis.create_iterator(inst, peer, fields, fld_offsets, fld_sizes);
  }

  template <int N, typename T, int N2, typename T2>
  TransferIterator *IndirectionInfoTyped<N,T,N2,T2>::create_indirect_iterator(Memory addrs_mem,
									      RegionInstance inst,
									      const std::vector<FieldID>& fields,
									      const std::vector<size_t>& fld_offsets,
									      const std::vector<size_t>& fld_sizes) const
  {
    if(is_ranges)
      return new TransferIteratorIndirectRange<N2,T2>(addrs_mem,
						      inst,
						      fields,
						      fld_offsets,
						      fld_sizes);
    else
      return new TransferIteratorIndirect<N2,T2>(addrs_mem,
						 inst,
						 fields,
						 fld_offsets,
						 fld_sizes);
  }
  
  template <int N, typename T, int N2, typename T2>
  void IndirectionInfoTyped<N,T,N2,T2>::print(std::ostream& os) const
  {
    if(structured) {
      assert(0);
    } else {
      os << inst << '[' << field_id << '+' << subfield_offset << ']';
      for(size_t i = 0; i < spaces.size(); i++) {
	if(i)
	  os << ", ";
	else
	  os << " -> ";
	os << spaces[i] << ':' << insts[i];
      }
    }
  }

  template <int N, typename T>
  template <int N2, typename T2>
  IndirectionInfo *CopyIndirection<N,T>::Unstructured<N2,T2>::create_info(const IndexSpace<N,T>& is) const
  {
    return new IndirectionInfoTyped<N,T,N2,T2>(is, *this);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDesc
  //

  TransferDesc::~TransferDesc()
  {
    log_xplan.info() << "destroyed: plan=" << (void *)this;

    delete domain;
    for(size_t i = 0; i < indirects.size(); i++)
      delete indirects[i];
    if(fill_data)
      free(fill_data);
  }

  bool TransferDesc::request_analysis(TransferOperation *op)
  {
    // early out without lock
    if(analysis_complete.load_acquire())
      return true;

    AutoLock<> al(mutex);
    if(analysis_complete.load()) {
      return true;
    } else {
      pending_ops.push_back(op);
      return false;
    }
  }

  void TransferDesc::check_analysis_preconditions()
  {
    log_xplan.info() << "created: plan=" << (void *)this << " domain=" << *domain << " srcs=" << srcs.size() << " dsts=" << dsts.size();
    if(log_xplan.want_debug()) {
      for(size_t i = 0; i < srcs.size(); i++)
	log_xplan.debug() << "created: plan=" << (void *)this << " srcs[" << i << "]=" << srcs[i];
      for(size_t i = 0; i < dsts.size(); i++)
	log_xplan.debug() << "created: plan=" << (void *)this << " dsts[" << i << "]=" << dsts[i];
      for(size_t i = 0; i < indirects.size(); i++)
	log_xplan.debug() << "created: plan=" << (void *)this << " indirects[" << i << "]=" << *indirects[i];
    }

    std::vector<Event> preconditions;

    // we need metadata for the domain and every instance
    {
      Event e = domain->request_metadata();
      if(e.exists()) preconditions.push_back(e);
    }

    std::set<RegionInstance> insts_seen;
    for(size_t i = 0; i < srcs.size(); i++)
      if(srcs[i].inst.exists()) {
	if(insts_seen.count(srcs[i].inst) > 0) continue;
	insts_seen.insert(srcs[i].inst);
	RegionInstanceImpl *impl = get_runtime()->get_instance_impl(srcs[i].inst);
	Event e = impl->request_metadata();
	if(e.exists()) preconditions.push_back(e);
      }

    for(size_t i = 0; i < dsts.size(); i++)
      if(dsts[i].inst.exists()) {
	if(insts_seen.count(dsts[i].inst) > 0) continue;
	insts_seen.insert(dsts[i].inst);
	RegionInstanceImpl *impl = get_runtime()->get_instance_impl(dsts[i].inst);
	Event e = impl->request_metadata();
	if(e.exists()) preconditions.push_back(e);
      }

    for(size_t i = 0; i < indirects.size(); i++) {
      Event e = indirects[i]->request_metadata();
      if(e.exists()) preconditions.push_back(e);
    }

    if(preconditions.empty()) {
      perform_analysis();
    } else {
      Event merged = Event::merge_events(preconditions);
      EventImpl::add_waiter(merged, &deferred_analysis);
    }
  }

  static size_t compute_ib_size(size_t combined_field_size,
				size_t domain_size,
				CustomSerdezID serdez_id)
  {
    size_t element_size;
    size_t serdez_pad = 0;
    size_t min_granularity = 1;
    if(serdez_id != 0) {
      const CustomSerdezUntyped *serdez_op = get_runtime()->custom_serdez_table.get(serdez_id, 0);
      assert(serdez_op != 0);
      element_size = serdez_op->max_serialized_size;
      serdez_pad = serdez_op->max_serialized_size;
    } else {
      element_size = combined_field_size;
      min_granularity = combined_field_size;
    }

    size_t ib_size = domain_size * element_size + serdez_pad;
    const size_t IB_MAX_SIZE = 16 << 20; // 16MB
    if(ib_size > IB_MAX_SIZE) {
      // take up to IB_MAX_SIZE, respecting the min granularity
      if(min_granularity > 1) {
	// (really) corner case: if min_granularity exceeds IB_MAX_SIZE, use it
	//  directly and hope it's ok
	if(min_granularity > IB_MAX_SIZE) {
	  ib_size = min_granularity;
	} else {
	  size_t extra = IB_MAX_SIZE % min_granularity;
	  ib_size = IB_MAX_SIZE - extra;
	}
      } else
	ib_size = IB_MAX_SIZE;
    }

    return ib_size;
  }

  void TransferDesc::perform_analysis()
  {
    // initialize profiling data
    prof_usage.source = Memory::NO_MEMORY;
    prof_usage.target = Memory::NO_MEMORY;
    prof_usage.size = 0;

    // quick check - if the domain is empty, there's nothing to actually do
    if(domain->empty()) {
      log_xplan.debug() << "analysis: plan=" << (void *)this << " empty";

      // well, we still have to poke pending ops
      std::vector<TransferOperation *> to_alloc;
      {
	AutoLock<> al(mutex);
	to_alloc.swap(pending_ops);
	analysis_complete.store(true);  // release done by mutex
      }

      for(size_t i = 0; i < to_alloc.size(); i++)
	to_alloc[i]->allocate_ibs();
      return;
    }

    size_t domain_size = domain->volume();

    // first, scan over the sources and figure out how much space we need
    //  for fill data - don't need to know field order yet
    for(size_t i = 0; i < srcs.size(); i++)
      if(srcs[i].field_id == FieldID(-1))
	fill_size += srcs[i].size;

    size_t fill_ofs = 0;
    if(fill_size > 0) {
      fill_data = malloc(fill_size);
      assert(fill_data);
    }

    // for now, pick a global dimension ordering
    // TODO: allow this to vary for independent subgraphs (or dependent ones
    //   with transposes in line)
    domain->choose_dim_order(dim_order,
			     srcs, dsts, indirects,
			     false /*!force_fortran_order*/,
			     65536 /*max_stride*/);

    src_fields.resize(srcs.size());
    dst_fields.resize(dsts.size());

    // TODO: look at layouts and decide if fields should be grouped into
    //  a smaller number of copies
    assert(srcs.size() == dsts.size());
    for(size_t i = 0; i < srcs.size(); i++) {
      assert(srcs[i].redop_id == 0);
      if(dsts[i].redop_id == 0) {
        // sizes of fills or copies should match
        assert(srcs[i].size == dsts[i].size);
      } else {
        // redop must exist and match src/dst sizes
        const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(dsts[i].redop_id, 0);
        if(redop == 0) {
          log_dma.fatal() << "no reduction op registered for ID " << dsts[i].redop_id;
          abort();
        }
        size_t exp_dst_size = dsts[i].red_fold ? redop->sizeof_rhs : redop->sizeof_lhs;
        assert(srcs[i].size == redop->sizeof_rhs);
        assert(dsts[i].size == exp_dst_size);
      }
      assert(srcs[i].serdez_id == dsts[i].serdez_id);

      size_t combined_field_size = srcs[i].size;
      CustomSerdezID serdez_id = srcs[i].serdez_id;

      if(dsts[i].redop_id != 0) {
        // reduction
	assert((srcs[i].indirect_index == -1) && "help: gather reduce!");
	assert((dsts[i].indirect_index == -1) && "help: scatter reduce!");
        assert((srcs[i].serdez_id == 0) &&
               (dsts[i].serdez_id == 0) && "help: serdez reduce!");

	src_fields[i] = FieldInfo { srcs[i].field_id, srcs[i].subfield_offset, srcs[i].size, srcs[i].serdez_id };
	dst_fields[i] = FieldInfo { dsts[i].field_id, dsts[i].subfield_offset, dsts[i].size, dsts[i].serdez_id };

        Memory src_mem = srcs[i].inst.get_location();
        Memory dst_mem = dsts[i].inst.get_location();

        MemPathInfo path_info;
        bool ok = find_shortest_path(src_mem, dst_mem, serdez_id,
                                     dsts[i].redop_id,
                                     path_info);
        if(!ok) {
          log_new_dma.fatal() << "FATAL: no path found from " << src_mem << " to " << dst_mem << " (redop=" << dsts[i].redop_id << ")";
          assert(0);
        }
        size_t pathlen = path_info.xd_channels.size();
        size_t xd_idx = graph.xd_nodes.size();
        size_t ib_idx = graph.ib_edges.size();
        size_t ib_alloc_size = 0;
        graph.xd_nodes.resize(xd_idx + pathlen);
        if(pathlen > 1) {
          graph.ib_edges.resize(ib_idx + pathlen - 1);
          ib_alloc_size = compute_ib_size(combined_field_size,
                                          domain_size,
                                          serdez_id);
        }
        for(size_t j = 0; j < pathlen; j++) {
          TransferGraph::XDTemplate& xdn = graph.xd_nodes[xd_idx++];

          //xdn.kind = path_info.xd_kinds[j];
          xdn.factory = path_info.xd_channels[j]->get_factory();
          xdn.gather_control_input = -1;
          xdn.scatter_control_input = -1;
          xdn.target_node = path_info.xd_channels[j]->node;
          if(j == (pathlen - 1))
            xdn.redop = XferDesRedopInfo(dsts[i].redop_id,
                                         dsts[i].red_fold,
                                         true /*in_place*/);
          xdn.inputs.resize(1);
          xdn.inputs[0] = ((j == 0) ?
                             TransferGraph::XDTemplate::mk_inst(srcs[i].inst,
                                                                i, 1) :
                             TransferGraph::XDTemplate::mk_edge(ib_idx - 1));
          //xdn.inputs[0].indirect_inst = RegionInstance::NO_INST;
          xdn.outputs.resize(1);
          xdn.outputs[0] = ((j == (pathlen - 1)) ?
                              TransferGraph::XDTemplate::mk_inst(dsts[i].inst,
                                                                 i, 1) :
                              TransferGraph::XDTemplate::mk_edge(ib_idx));
          //xdn.outputs[0].indirect_inst = RegionInstance::NO_INST;
          if(j < (pathlen - 1)) {
            TransferGraph::IBInfo& ibe = graph.ib_edges[ib_idx++];
            ibe.memory = path_info.path[j + 1];
            ibe.size = ib_alloc_size;
          }
        }

        prof_usage.source = src_mem;
        prof_usage.target = dst_mem;
        prof_usage.size += domain_size * combined_field_size;
        prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
          srcs[i].inst,
          dsts[i].inst,
          1 /*num_fields*/,
          ProfilingMeasurements::OperationCopyInfo::REDUCE,
          unsigned(pathlen) });
      }
      else if(srcs[i].field_id == FieldID(-1)) {
	// fill
	assert((dsts[i].indirect_index == -1) && "help: scatter fill!");

	src_fields[i] = FieldInfo { FieldID(-1), 0, 0, 0 };
	dst_fields[i] = FieldInfo { dsts[i].field_id, dsts[i].subfield_offset, dsts[i].size, dsts[i].serdez_id };

	Memory dst_mem = dsts[i].inst.get_location();
	MemPathInfo path_info;
	bool ok = find_shortest_path(Memory::NO_MEMORY, dst_mem, serdez_id,
                                     0 /*redop_id*/,
				     path_info);
	if(!ok) {
	  log_new_dma.fatal() << "FATAL: no fill path found for " << dst_mem << " (serdez=" << serdez_id << ")";
	  assert(0);
	}

	size_t pathlen = path_info.xd_channels.size();
	size_t xd_idx = graph.xd_nodes.size();
	//size_t ib_idx = graph.ib_edges.size();

	graph.xd_nodes.resize(xd_idx + pathlen);
	if(pathlen > 1) {
	  log_new_dma.fatal() << "FATAL: multi-hop fill path found for " << dst_mem << " (serdez=" << serdez_id << ")";
	  assert(0);
	}
	// just one node for now
	{
	  TransferGraph::XDTemplate& xdn = graph.xd_nodes[xd_idx++];
	      
	  //xdn.kind = path_info.xd_kinds[j];
	  xdn.factory = path_info.xd_channels[0]->get_factory();
	  xdn.gather_control_input = -1;
	  xdn.scatter_control_input = -1;
	  xdn.target_node = path_info.xd_channels[0]->node;
	  xdn.inputs.resize(1);
	  xdn.inputs[0] = TransferGraph::XDTemplate::mk_fill(fill_ofs,
							     combined_field_size);
	  // FIXME: handle multiple fields
	  memcpy(static_cast<char *>(fill_data)+fill_ofs,
		 ((srcs[i].size <= CopySrcDstField::MAX_DIRECT_SIZE) ?
		    srcs[i].fill_data.direct :
		    srcs[i].fill_data.indirect),
		 srcs[i].size);
	  fill_ofs += srcs[i].size;

	  xdn.outputs.resize(1);
	  xdn.outputs[0] = TransferGraph::XDTemplate::mk_inst(dsts[i].inst,
							      i, 1);
	}

        prof_usage.source = Memory::NO_MEMORY;
        prof_usage.target = dst_mem;
        prof_usage.size += domain_size * combined_field_size;
        prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
              RegionInstance::NO_INST,
              dsts[i].inst,
              1 /*num_fields*/,
              ProfilingMeasurements::OperationCopyInfo::FILL,
              unsigned(pathlen) });
      } else {
	src_fields[i] = FieldInfo { srcs[i].field_id, srcs[i].subfield_offset, srcs[i].size, srcs[i].serdez_id };
	dst_fields[i] = FieldInfo { dsts[i].field_id, dsts[i].subfield_offset, dsts[i].size, dsts[i].serdez_id };
	
	if(srcs[i].indirect_index == -1) {
	  Memory src_mem = srcs[i].inst.get_location();

	  if(dsts[i].indirect_index == -1) {
	    Memory dst_mem = dsts[i].inst.get_location();

	    MemPathInfo path_info;
	    bool ok = find_shortest_path(src_mem, dst_mem, serdez_id,
                                         0 /*redop_id*/,
					 path_info);
	    if(!ok) {
	      log_new_dma.fatal() << "FATAL: no path found from " << src_mem << " to " << dst_mem << " (serdez=" << serdez_id << ")";
	      assert(0);
	    }
	    size_t pathlen = path_info.xd_channels.size();
	    size_t xd_idx = graph.xd_nodes.size();
	    size_t ib_idx = graph.ib_edges.size();
	    size_t ib_alloc_size = 0;
	    graph.xd_nodes.resize(xd_idx + pathlen);
	    if(pathlen > 1) {
	      graph.ib_edges.resize(ib_idx + pathlen - 1);
	      ib_alloc_size = compute_ib_size(combined_field_size,
					      domain_size,
					      serdez_id);
	    }
	    for(size_t j = 0; j < pathlen; j++) {
	      TransferGraph::XDTemplate& xdn = graph.xd_nodes[xd_idx++];
	      
	      //xdn.kind = path_info.xd_kinds[j];
	      xdn.factory = path_info.xd_channels[j]->get_factory();
	      xdn.gather_control_input = -1;
	      xdn.scatter_control_input = -1;
	      xdn.target_node = path_info.xd_channels[j]->node;
	      xdn.inputs.resize(1);
	      xdn.inputs[0] = ((j == 0) ?
			         TransferGraph::XDTemplate::mk_inst(srcs[i].inst,
								    i, 1) :
			         TransferGraph::XDTemplate::mk_edge(ib_idx - 1));
	      //xdn.inputs[0].indirect_inst = RegionInstance::NO_INST;
	      xdn.outputs.resize(1);
	      xdn.outputs[0] = ((j == (pathlen - 1)) ?
				  TransferGraph::XDTemplate::mk_inst(dsts[i].inst,
								     i, 1) :
				  TransferGraph::XDTemplate::mk_edge(ib_idx));
	      //xdn.outputs[0].indirect_inst = RegionInstance::NO_INST;
	      if(j < (pathlen - 1)) {
		TransferGraph::IBInfo& ibe = graph.ib_edges[ib_idx++];
		ibe.memory = path_info.path[j + 1];
		ibe.size = ib_alloc_size;
	      }
	    }

            prof_usage.source = src_mem;
            prof_usage.target = dst_mem;
            prof_usage.size += domain_size * combined_field_size;
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
                 srcs[i].inst,
                 dsts[i].inst,
                 1 /*num_fields*/,
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(pathlen) });
	  } else {
	    // scatter
	    IndirectionInfo *scatter_info = indirects[dsts[i].indirect_index];
	    size_t addrsplit_bytes_per_element = ((serdez_id == 0) ?
						    combined_field_size :
						    1);
            size_t prev_nodes = graph.xd_nodes.size();
	    scatter_info->generate_scatter_paths(src_mem,
						 TransferGraph::XDTemplate::mk_inst(srcs[i].inst, i, 1),
						 dsts[i].indirect_index,
						 i, 1,
						 addrsplit_bytes_per_element,
						 serdez_id,
						 graph.xd_nodes,
						 graph.ib_edges,
						 src_fields);

            prof_usage.source = src_mem;
            prof_usage.target = Memory::NO_MEMORY;
            prof_usage.size += domain_size * combined_field_size;
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
                 srcs[i].inst,
                 RegionInstance::NO_INST,
                 1 /*num_fields*/,
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(graph.xd_nodes.size() - prev_nodes) });
	  }
	} else {
	  size_t addrsplit_bytes_per_element = ((serdez_id == 0) ?
						  combined_field_size :
						  1);
	  if(dsts[i].indirect_index == -1) {
	    Memory dst_mem = dsts[i].inst.get_location();
	    IndirectionInfo *gather_info = indirects[srcs[i].indirect_index];
            size_t prev_nodes = graph.xd_nodes.size();
	    gather_info->generate_gather_paths(dst_mem,
					       TransferGraph::XDTemplate::mk_inst(dsts[i].inst, i, 1),
					       srcs[i].indirect_index,
					       i, 1,
					       addrsplit_bytes_per_element,
					       serdez_id,
					       graph.xd_nodes,
					       graph.ib_edges,
					       src_fields);

            prof_usage.source = Memory::NO_MEMORY;
            prof_usage.target = dst_mem;
            prof_usage.size += domain_size * combined_field_size;
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
                 RegionInstance::NO_INST,
                 dsts[i].inst,
                 1 /*num_fields*/,
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(graph.xd_nodes.size() - prev_nodes) });
	  } else {
	    // scatter+gather
	    // TODO: optimize case of single source and single dest

            size_t prev_nodes = graph.xd_nodes.size();

	    // need an intermediate buffer for the output of the gather
	    //  (and input of the scatter)
	    // TODO: local node isn't necessarily the right choice!
	    NodeID ib_node = Network::my_node_id;
	    Memory ib_mem = ID::make_ib_memory(ib_node, 0).convert<Memory>();

	    int ib_idx = graph.ib_edges.size();
	    graph.ib_edges.resize(ib_idx + 1);
	    graph.ib_edges[ib_idx].memory = ib_mem;
	    graph.ib_edges[ib_idx].size = 1 << 20;  //HACK

	    IndirectionInfo *gather_info = indirects[srcs[i].indirect_index];
	    gather_info->generate_gather_paths(ib_mem,
					       TransferGraph::XDTemplate::mk_edge(ib_idx),
					       srcs[i].indirect_index,
					       i, 1,
					       addrsplit_bytes_per_element,
					       serdez_id,
					       graph.xd_nodes,
					       graph.ib_edges,
					       src_fields);

	    IndirectionInfo *scatter_info = indirects[dsts[i].indirect_index];
	    scatter_info->generate_scatter_paths(ib_mem,
						 TransferGraph::XDTemplate::mk_edge(ib_idx),
						 dsts[i].indirect_index,
						 i, 1,
						 addrsplit_bytes_per_element,
						 serdez_id,
						 graph.xd_nodes,
						 graph.ib_edges,
						 src_fields);

            prof_usage.source = Memory::NO_MEMORY;
            prof_usage.target = Memory::NO_MEMORY;
            prof_usage.size += domain_size * combined_field_size;
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
                 RegionInstance::NO_INST,
                 RegionInstance::NO_INST,
                 1 /*num_fields*/,
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(graph.xd_nodes.size() - prev_nodes) });
	  }
	}
      }
    }

    if(log_xplan.want_debug()) {
      log_xplan.debug() << "analysis: plan=" << (void *)this
			<< " dim_order=" << PrettyVector<int>(dim_order)
			<< " xds=" << graph.xd_nodes.size()
			<< " ibs=" << graph.ib_edges.size();
      for(size_t i = 0; i < graph.xd_nodes.size(); i++)
        if(graph.xd_nodes[i].redop.id != 0)
          log_xplan.debug() << "analysis: plan=" << (void *)this
                            << " xds[" << i << "]: target=" << graph.xd_nodes[i].target_node
                            << " inputs=" << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].inputs)
                            << " outputs=" << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].outputs)
                            << " redop=(" << graph.xd_nodes[i].redop.id << "," << graph.xd_nodes[i].redop.is_fold << "," << graph.xd_nodes[i].redop.in_place << ")";
        else
          log_xplan.debug() << "analysis: plan=" << (void *)this
                            << " xds[" << i << "]: target=" << graph.xd_nodes[i].target_node
                            << " inputs=" << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].inputs)
                            << " outputs=" << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].outputs);
      for(size_t i = 0; i < graph.ib_edges.size(); i++)
	log_xplan.debug() << "analysis: plan=" << (void *)this
			  << " ibs[" << i << "]: memory=" << graph.ib_edges[i].memory
			  << " size=" << graph.ib_edges[i].size;
    }

    // mark that the analysis is complete and see if there are any pending
    //  ops that can start allocating ibs
    std::vector<TransferOperation *> to_alloc;
    {
      AutoLock<> al(mutex);
      to_alloc.swap(pending_ops);
      analysis_complete.store(true);  // release done by mutex
    }

    for(size_t i = 0; i < to_alloc.size(); i++)
      to_alloc[i]->allocate_ibs();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDesc::DeferredAnalysis
  //

  TransferDesc::DeferredAnalysis::DeferredAnalysis(TransferDesc *_desc)
    : desc(_desc)
  {}

  void TransferDesc::DeferredAnalysis::event_triggered(bool poisoned,
						       TimeLimit work_until)
  {
    assert(!poisoned);
    // TODO: respect time limit

    desc->perform_analysis();
  }

  void TransferDesc::DeferredAnalysis::print(std::ostream& os) const
  {
    os << "deferred_analysis(" << (void *)desc << ")";
  }

  Event TransferDesc::DeferredAnalysis::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

			      
  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferOperation
  //

  TransferOperation::TransferOperation(TransferDesc& _desc,
				       Event _precondition,
				       GenEventImpl *_finish_event,
				       EventImpl::gen_t _finish_gen)
    : Operation(_finish_event, _finish_gen, _desc.prs)
    , deferred_start(this)
    , desc(_desc)
    , precondition(_precondition)
    , ib_responses_needed(0)
    , priority(0) // FIXME
  {
    desc.add_reference();
  }

  TransferOperation::~TransferOperation()
  {
    desc.remove_reference();
  }

  void TransferOperation::print(std::ostream& os) const
  {
    os << "transfer_op(" << (void *)this << ")";
  }

  void TransferOperation::start_or_defer(void)
  {
    log_dma.info() << "dma request " << (void *)this
		   << " created - plan=" << (void *)&desc
		   << " before=" << precondition
		   << " after=" << get_finish_event();

    bool poisoned;
    if(!precondition.has_triggered_faultaware(poisoned)) {
      EventImpl::add_waiter(precondition, &deferred_start);
      return;
    }
    assert(!poisoned);

    // see if we need to wait for the transfer description analysis
    if(desc.request_analysis(this)) {
      // it's ready - go ahead and do ib creation
      allocate_ibs();
    } else {
      // do nothing - the TransferDesc will call us when it's ready
    }
  }

  bool TransferOperation::mark_ready(void)
  {
    bool ok_to_run = Operation::mark_ready();
    if(ok_to_run)
      log_dma.info() << "dma request " << (void *)this
		     << " ready - plan=" << (void *)&desc
		     << " before=" << precondition
		     << " after=" << get_finish_event();
    return ok_to_run;
  }

  bool TransferOperation::mark_started(void)
  {
    bool ok_to_run = Operation::mark_started();
    if(ok_to_run)
      log_dma.info() << "dma request " << (void *)this
		     << " started - plan=" << (void *)&desc
		     << " before=" << precondition
		     << " after=" << get_finish_event();
    return ok_to_run;
  }

  void TransferOperation::mark_completed(void)
  {
    log_dma.info() << "dma request " << (void *)this
		   << " completed - plan=" << (void *)&desc
		   << " before=" << precondition
		   << " after=" << get_finish_event();

    Operation::mark_completed();
  }

  void TransferOperation::allocate_ibs()
  {
    // make sure we haven't been cancelled
    bool ok_to_run = mark_ready();
    if(!ok_to_run) {
      mark_finished(false /*!successful*/);
      return;
    }

    const TransferGraph& tg = desc.graph;

    if(!tg.ib_edges.empty()) {
      ib_offsets.resize(tg.ib_edges.size(), -1);

      // increase the count by one to prevent a trigger before we finish
      //  this loop
      ib_responses_needed.store(tg.ib_edges.size() + 1);

      // sort requests by target memory to reduce risk of resource deadlock
      // TODO: actually merge requests so that there's at most one per memory
      typedef std::map<Memory, std::vector<unsigned> > PendingIBRequests;
      PendingIBRequests pending;
      for(size_t i = 0; i < tg.ib_edges.size(); i++)
	pending[tg.ib_edges[i].memory].push_back(i);

      for(PendingIBRequests::const_iterator it = pending.begin();
	  it != pending.end();
	  ++it) {
	Memory tgt_mem = it->first;
	for(std::vector<unsigned>::const_iterator it2 = it->second.begin();
	    it2 != it->second.end();
	    ++it2) {
	  NodeID owner = ID(tgt_mem).memory_owner_node();
	  if(owner == Network::my_node_id) {
	    // local request
	    ib_req_queue.enqueue_request(tgt_mem, Network::my_node_id,
					 reinterpret_cast<uintptr_t>(this),
					 *it2, tg.ib_edges[*it2].size);
	  } else {
	    // send message to remote owner
	    ActiveMessage<RemoteIBAllocRequest> amsg(owner);
	    amsg->memory = tgt_mem;
	    amsg->size = tg.ib_edges[*it2].size;
	    amsg->req_op = reinterpret_cast<uintptr_t>(this);
	    amsg->ib_index = *it2;
	    amsg.commit();
	  }
	}
      }

      // once all requests are made, do the extra decrement and continue if they
      //  are all satisfied
      if(ib_responses_needed.fetch_sub_acqrel(1) > 1)
	return;
    }

    // fall through to creating xds
    create_xds();
  }

  std::ostream& operator<<(std::ostream& os, const TransferGraph::XDTemplate::IO& io)
  {
    switch(io.iotype) {
    case TransferGraph::XDTemplate::IO_INST:
      os << "inst(" << io.inst.inst << "," << io.inst.fld_start << "+" << io.inst.fld_count << ")"; break;
    case TransferGraph::XDTemplate::IO_INDIRECT_INST:
      os << "ind(" << io.indirect.ind_idx << "," << io.indirect.port << "," << io.indirect.inst << "," << io.indirect.fld_start << "+" << io.indirect.fld_count << ")"; break;
    case TransferGraph::XDTemplate::IO_EDGE:
      os << "edge(" << io.edge << ")"; break;
    case TransferGraph::XDTemplate::IO_FILL_DATA:
      os << "fill(" << io.fill.fill_start << "+" << io.fill.fill_size << ")"; break;
    default:
      assert(0);
    }
    return os;
  }

  void TransferOperation::notify_ib_allocation(unsigned ib_index,
					       off_t ib_offset)
  {
    assert(ib_index < ib_offsets.size());
#ifdef DEBUG_REALM
    assert(ib_offsets[ib_index] == -1);
#endif
    ib_offsets[ib_index] = ib_offset;

    // if this was the last response needed, we can continue on to creating xds
    if(ib_responses_needed.fetch_sub_acqrel(1) == 1)
      create_xds();
  }

  void TransferOperation::create_xds()
  {
    // make sure we haven't been cancelled
    bool ok_to_run = this->mark_started();
    if(!ok_to_run) {
      mark_finished(false /*!successful*/);
      return;
    }

    const TransferGraph& tg = desc.graph;

    // we're going to need pre/next xdguids, so precreate all of them
    xd_ids.resize(tg.xd_nodes.size(), XferDes::XFERDES_NO_GUID);
    typedef std::pair<XferDesID,int> IBEdge;
    const IBEdge dfl_edge(XferDes::XFERDES_NO_GUID, 0);
    std::vector<IBEdge> ib_pre_ids(tg.ib_edges.size(), dfl_edge);
    std::vector<IBEdge> ib_next_ids(tg.ib_edges.size(), dfl_edge);

    XferDesQueue *xdq = XferDesQueue::get_singleton();
    for(size_t i = 0; i < tg.xd_nodes.size(); i++) {
      const TransferGraph::XDTemplate& xdn = tg.xd_nodes[i];

      XferDesID new_xdid = xdq->get_guid(xdn.target_node);

      xd_ids[i] = new_xdid;
	
      for(size_t j = 0; j < xdn.inputs.size(); j++)
	if(xdn.inputs[j].iotype == TransferGraph::XDTemplate::IO_EDGE)
	  ib_next_ids[xdn.inputs[j].edge] = std::make_pair(new_xdid, j);
	
      for(size_t j = 0; j < xdn.outputs.size(); j++)
	if(xdn.outputs[j].iotype == TransferGraph::XDTemplate::IO_EDGE)
	  ib_pre_ids[xdn.outputs[j].edge] = std::make_pair(new_xdid, j);
    }

    log_new_dma.info() << "xds created: " << std::hex << PrettyVector<XferDesID>(xd_ids) << std::dec;
    
    // now actually create xfer descriptors for each template node in our DAG
    xd_trackers.resize(tg.xd_nodes.size(), 0);
    for(size_t i = 0; i < tg.xd_nodes.size(); i++) {
      const TransferGraph::XDTemplate& xdn = tg.xd_nodes[i];
	
      NodeID xd_target_node = xdn.target_node;
      XferDesID xd_guid = xd_ids[i];
      XferDesFactory *xd_factory = xdn.factory;

      const void *fill_data = 0;
      size_t fill_size = 0;

      std::vector<XferDesPortInfo> inputs_info(xdn.inputs.size());
      for(size_t j = 0; j < xdn.inputs.size(); j++) {
	XferDesPortInfo& ii = inputs_info[j];

	ii.port_type = ((int(j) == xdn.gather_control_input) ?
			  XferDesPortInfo::GATHER_CONTROL_PORT :
			(int(j) == xdn.scatter_control_input) ?
			  XferDesPortInfo::SCATTER_CONTROL_PORT :
			  XferDesPortInfo::DATA_PORT);

	switch(xdn.inputs[j].iotype) {
	case TransferGraph::XDTemplate::IO_INST:
	  {
	    ii.peer_guid = XferDes::XFERDES_NO_GUID;
	    ii.peer_port_idx = 0;
	    ii.indirect_port_idx = -1;
	    ii.mem = xdn.inputs[j].inst.inst.get_location();
	    ii.inst = xdn.inputs[j].inst.inst;
	    std::vector<FieldID> src_fields(xdn.inputs[j].inst.fld_count);
	    std::vector<size_t> src_offsets(xdn.inputs[j].inst.fld_count);
	    std::vector<size_t> src_sizes(xdn.inputs[j].inst.fld_count);
	    for(size_t k = 0; k < xdn.inputs[j].inst.fld_count; k++) {
	      src_fields[k] = desc.src_fields[xdn.inputs[j].inst.fld_start + k].id;
	      src_offsets[k] = desc.src_fields[xdn.inputs[j].inst.fld_start + k].offset;
	      src_sizes[k] = desc.src_fields[xdn.inputs[j].inst.fld_start + k].size;
	    }
	    ii.iter = desc.domain->create_iterator(xdn.inputs[j].inst.inst,
						   desc.dim_order,
						   src_fields,
						   src_offsets,
						   src_sizes);
	    // use first field's serdez - they all have to be the same
	    ii.serdez_id = desc.src_fields[xdn.inputs[j].inst.fld_start].serdez_id;
	    ii.ib_offset = 0;
	    ii.ib_size = 0;
	    break;
	  }
	case TransferGraph::XDTemplate::IO_INDIRECT_INST:
	  {
	    ii.peer_guid = XferDes::XFERDES_NO_GUID;
	    ii.peer_port_idx = 0;
	    ii.indirect_port_idx = xdn.inputs[j].indirect.port;
	    ii.mem = xdn.inputs[j].indirect.inst.get_location();
	    ii.inst = xdn.inputs[j].indirect.inst;
	    std::vector<FieldID> src_fields(xdn.inputs[j].indirect.fld_count);
	    std::vector<size_t> src_offsets(xdn.inputs[j].indirect.fld_count);
	    std::vector<size_t> src_sizes(xdn.inputs[j].indirect.fld_count);
	    for(size_t k = 0; k < xdn.inputs[j].indirect.fld_count; k++) {
	      src_fields[k] = desc.src_fields[xdn.inputs[j].indirect.fld_start + k].id;
	      src_offsets[k] = desc.src_fields[xdn.inputs[j].indirect.fld_start + k].offset;
	      src_sizes[k] = desc.src_fields[xdn.inputs[j].indirect.fld_start + k].size;
	    }
	    IndirectionInfo *gather_info = desc.indirects[xdn.inputs[j].indirect.ind_idx];
	    ii.iter = gather_info->create_indirect_iterator(ii.mem,
							    xdn.inputs[j].indirect.inst,
							    src_fields,
							    src_offsets,
							    src_sizes);
	    // use first field's serdez - they all have to be the same
	    ii.serdez_id = desc.src_fields[xdn.inputs[j].indirect.fld_start].serdez_id;
	    ii.ib_offset = 0;
	    ii.ib_size = 0;
	    break;
	  }
	case TransferGraph::XDTemplate::IO_EDGE:
	  {
	    ii.peer_guid = ib_pre_ids[xdn.inputs[j].edge].first;
	    ii.peer_port_idx = ib_pre_ids[xdn.inputs[j].edge].second;
	    ii.indirect_port_idx = -1;
	    ii.mem = tg.ib_edges[xdn.inputs[j].edge].memory;
	    ii.inst = RegionInstance::NO_INST;
	    ii.ib_offset = ib_offsets[xdn.inputs[j].edge];
	    ii.ib_size = tg.ib_edges[xdn.inputs[j].edge].size;
	    ii.iter = new WrappingFIFOIterator(ii.ib_offset, ii.ib_size);
	    ii.serdez_id = 0;
	    break;
	  }
	case TransferGraph::XDTemplate::IO_FILL_DATA:
	  {
	    // don't actually want an input in this case
	    assert((j == 0) && (xdn.inputs.size() == 1));
	    inputs_info.clear();
	    fill_data = static_cast<const char *>(desc.fill_data) + xdn.inputs[j].fill.fill_start;
	    fill_size = xdn.inputs[j].fill.fill_size;
	    break;
	  }
	default:
	  assert(0);
	}
#if 0
	if(0) {
	    //mark_started = true;
	  } else if(xdn.inputs[j].edge_id <= XDTemplate::INDIRECT_BASE) {
	    int ind_idx = XDTemplate::INDIRECT_BASE - xdn.inputs[j].edge_id;
	    assert(xdn.inputs[j].indirect_inst.exists());
	    //log_dma.print() << "indirect iter: inst=" << xdn.inputs[j].indirect_inst;
	    ii.peer_guid = XferDes::XFERDES_NO_GUID;
	    ii.peer_port_idx = 0;
	    ii.indirect_port_idx = ind_idx;
	    ii.mem = xdn.inputs[j].indirect_inst.get_location();
	    ii.inst = xdn.inputs[j].indirect_inst;
	    ii.iter = gather_info->create_indirect_iterator(ii.mem,
							    xdn.inputs[j].indirect_inst,
							    src_fields,
							    src_field_offsets,
							    field_sizes);
	    ii.serdez_id = serdez_id;
	    ii.ib_offset = 0;
	    ii.ib_size = 0;
	  } else if(xdn.inputs[j].edge_id == XDTemplate::DST_INST) {
	    // this happens when doing a scatter and DST_INST is actually the
	    //  destination addresses (which we read)
	    ii.peer_guid = XferDes::XFERDES_NO_GUID;
	    ii.peer_port_idx = 0;
	    ii.indirect_port_idx = -1;
	    ii.mem = dst_inst.get_location();
	    ii.inst = dst_inst;
	    ii.iter = dst_iter;
	    ii.serdez_id = 0;
	    ii.ib_offset = 0;
	    ii.ib_size = 0;
	  } else {
	    ii.peer_guid = ib_pre_ids[xdn.inputs[j].edge_id].first;
	    ii.peer_port_idx = ib_pre_ids[xdn.inputs[j].edge_id].second;
	    ii.indirect_port_idx = -1;
	    ii.mem = ib_edges[xdn.inputs[j].edge_id].memory;
	    ii.inst = RegionInstance::NO_INST;
	    ii.ib_offset = ib_edges[xdn.inputs[j].edge_id].offset;
	    ii.ib_size = ib_edges[xdn.inputs[j].edge_id].size;
	    ii.iter = new WrappingFIFOIterator(ii.ib_offset, ii.ib_size);
	    ii.serdez_id = 0;
	}
#endif
      }

      std::vector<XferDesPortInfo> outputs_info(xdn.outputs.size());
      for(size_t j = 0; j < xdn.outputs.size(); j++) {
	XferDesPortInfo& oi = outputs_info[j];

	oi.port_type = XferDesPortInfo::DATA_PORT;

	switch(xdn.outputs[j].iotype) {
	case TransferGraph::XDTemplate::IO_INST:
	  {
	    oi.peer_guid = XferDes::XFERDES_NO_GUID;
	    oi.peer_port_idx = 0;
	    oi.indirect_port_idx = -1;
	    oi.mem = xdn.outputs[j].inst.inst.get_location();
	    oi.inst = xdn.outputs[j].inst.inst;
	    std::vector<FieldID> dst_fields(xdn.outputs[j].inst.fld_count);
	    std::vector<size_t> dst_offsets(xdn.outputs[j].inst.fld_count);
	    std::vector<size_t> dst_sizes(xdn.outputs[j].inst.fld_count);
	    for(size_t k = 0; k < xdn.outputs[j].inst.fld_count; k++) {
	      dst_fields[k] = desc.dst_fields[xdn.outputs[j].inst.fld_start + k].id;
	      dst_offsets[k] = desc.dst_fields[xdn.outputs[j].inst.fld_start + k].offset;
	      dst_sizes[k] = desc.dst_fields[xdn.outputs[j].inst.fld_start + k].size;
	    }
	    oi.iter = desc.domain->create_iterator(xdn.outputs[j].inst.inst,
						   desc.dim_order,
						   dst_fields,
						   dst_offsets,
						   dst_sizes);
	    // use first field's serdez - they all have to be the same
	    oi.serdez_id = desc.dst_fields[xdn.outputs[j].inst.fld_start].serdez_id;
	    oi.ib_offset = 0;
	    oi.ib_size = 0;
	    break;
	  }
	case TransferGraph::XDTemplate::IO_INDIRECT_INST:
	  {
	    oi.peer_guid = XferDes::XFERDES_NO_GUID;
	    oi.peer_port_idx = 0;
	    oi.indirect_port_idx = xdn.outputs[j].indirect.port;
	    oi.mem = xdn.outputs[j].indirect.inst.get_location();
	    oi.inst = xdn.outputs[j].indirect.inst;
	    std::vector<FieldID> dst_fields(xdn.outputs[j].indirect.fld_count);
	    std::vector<size_t> dst_offsets(xdn.outputs[j].indirect.fld_count);
	    std::vector<size_t> dst_sizes(xdn.outputs[j].indirect.fld_count);
	    for(size_t k = 0; k < xdn.outputs[j].indirect.fld_count; k++) {
	      dst_fields[k] = desc.dst_fields[xdn.outputs[j].indirect.fld_start + k].id;
	      dst_offsets[k] = desc.dst_fields[xdn.outputs[j].indirect.fld_start + k].offset;
	      dst_sizes[k] = desc.dst_fields[xdn.outputs[j].indirect.fld_start + k].size;
	    }
	    IndirectionInfo *scatter_info = desc.indirects[xdn.outputs[j].indirect.ind_idx];
	    oi.iter = scatter_info->create_indirect_iterator(oi.mem,
							     xdn.outputs[j].indirect.inst,
							     dst_fields,
							     dst_offsets,
							     dst_sizes);
	    // use first field's serdez - they all have to be the same
	    oi.serdez_id = desc.dst_fields[xdn.outputs[j].indirect.fld_start].serdez_id;
	    oi.ib_offset = 0;
	    oi.ib_size = 0;
	    break;
	  }
	case TransferGraph::XDTemplate::IO_EDGE:
	  {
	    oi.peer_guid = ib_next_ids[xdn.outputs[j].edge].first;
	    oi.peer_port_idx = ib_next_ids[xdn.outputs[j].edge].second;
	    oi.indirect_port_idx = -1;
	    oi.mem = tg.ib_edges[xdn.outputs[j].edge].memory;
	    oi.inst = RegionInstance::NO_INST;
	    oi.ib_offset = ib_offsets[xdn.outputs[j].edge];
	    oi.ib_size = tg.ib_edges[xdn.outputs[j].edge].size;
	    oi.iter = new WrappingFIFOIterator(oi.ib_offset, oi.ib_size);
	    oi.serdez_id = 0;
	    break;
	  }
	default:
	  assert(0);
	}
#if 0
	  if(xdn.outputs[j].edge_id == XDNemplate::DST_INST) {
	    oi.peer_guid = XferDes::XFERDES_NO_GUID;
	    oi.peer_port_idx = 0;
	    oi.indirect_port_idx = -1;
	    oi.mem = dst_inst.get_location();
	    oi.inst = dst_inst;
	    oi.iter = dst_iter;
	    oi.serdez_id = serdez_id;
	    oi.ib_offset = 0;
	    oi.ib_size = 0;  // doesn't matter
	  } else if(xdn.outputs[j].edge_id <= XDNemplate::INDIRECT_BASE) {
	    int ind_idx = XDNemplate::INDIRECT_BASE - xdn.outputs[j].edge_id;
	    assert(xdn.outputs[j].indirect_inst.exists());
	    //log_dma.print() << "indirect iter: inst=" << xdn.outputs[j].indirect_inst;
	    oi.peer_guid = XferDes::XFERDES_NO_GUID;
	    oi.peer_port_idx = 0;
	    oi.indirect_port_idx = ind_idx;
	    oi.mem = xdn.outputs[j].indirect_inst.get_location();
	    oi.inst = xdn.outputs[j].indirect_inst;
	    oi.iter = scatter_info->create_indirect_iterator(oi.mem,
							     xdn.outputs[j].indirect_inst,
							     dst_fields,
							     dst_field_offsets,
							     field_sizes);
	    oi.serdez_id = serdez_id;
	    oi.ib_offset = 0;
	    oi.ib_size = 0;
	  } else {
	    oi.peer_guid = ib_next_ids[xdn.outputs[j].edge_id].first;
	    oi.peer_port_idx = ib_next_ids[xdn.outputs[j].edge_id].second;
	    oi.indirect_port_idx = -1;
	    oi.mem = ib_edges[xdn.outputs[j].edge_id].memory;
	    oi.inst = RegionInstance::NO_INST;
	    oi.ib_offset = ib_edges[xdn.outputs[j].edge_id].offset;
	    oi.ib_size = ib_edges[xdn.outputs[j].edge_id].size;
	    oi.iter = new WrappingFIFOIterator(oi.ib_offset, oi.ib_size);
	    oi.serdez_id = 0;
	  }
#endif
      }

      xd_trackers[i] = new XDLifetimeTracker(this, xd_ids[i]);
      add_async_work_item(xd_trackers[i]);
	
      xd_factory->create_xfer_des(reinterpret_cast<uintptr_t>(this),
				  Network::my_node_id, xd_target_node,
				  xd_guid,
				  inputs_info,
				  outputs_info,
				  priority,
                                  xdn.redop,
				  fill_data, fill_size);
    }

    // record requested profiling information
    {
      using namespace ProfilingMeasurements;

      if(measurements.wants_measurement<OperationCopyInfo>())
        measurements.add_measurement(desc.prof_cpinfo);

      if(measurements.wants_measurement<OperationMemoryUsage>())
        measurements.add_measurement(desc.prof_usage);
    }

    mark_finished(true /*successful*/);
  }

  void TransferOperation::notify_xd_completion(XferDesID xd_id)
  {
    // TODO: get timing info as well?

    // scan out list of ids for a match
    for(size_t i = 0; i < xd_ids.size(); i++) {
      if(xd_id == xd_ids[i]) {
	XDLifetimeTracker *tracker = xd_trackers[i];
	assert(tracker);
	xd_trackers[i] = 0;
	destroy_xfer_des(xd_id);
	// calling this has to be the last thing we do
	tracker->mark_finished(true /*successful*/);
	return;
      }
    }
    assert(0);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferOperation::XDLifetimeTracker
  //

  TransferOperation::XDLifetimeTracker::XDLifetimeTracker(TransferOperation *_op,
							  XferDesID _xd_id)
    : Operation::AsyncWorkItem(_op)
    , xd_id(_xd_id)
  {}

  void TransferOperation::XDLifetimeTracker::request_cancellation(void)
  {
    // ignored
  }

  void TransferOperation::XDLifetimeTracker::print(std::ostream& os) const
  {
    os << "xd(" << std::hex << xd_id << std::dec << ")";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferOperation::DeferredStart
  //

  TransferOperation::DeferredStart::DeferredStart(TransferOperation *_op)
    : op(_op)
  {}

  void TransferOperation::DeferredStart::event_triggered(bool poisoned,
							 TimeLimit work_until)
  {
    assert(!poisoned);
    // TODO: respect time limit

    // see if we need to wait for the transfer description analysis
    if(op->desc.request_analysis(op)) {
      // it's ready - go ahead and do ib creation
      op->allocate_ibs();
    } else {
      // do nothing - the TransferDesc will call us when it's ready
    }
  }

  void TransferOperation::DeferredStart::print(std::ostream& os) const
  {
    os << "deferred_start(";
    op->print(os);
    os << ")";
  }

  Event TransferOperation::DeferredStart::get_finish_event(void) const
  {
    return op->finish_event->make_event(op->finish_gen);
  }

			      
  template <int N, typename T>
  Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField>& srcs,
			      const std::vector<CopySrcDstField>& dsts,
			      const std::vector<const typename CopyIndirection<N,T>::Base *> &indirects,
			      const Realm::ProfilingRequestSet &requests,
			      Event wait_on) const
  {
    // create a (one-use) transfer description
    TransferDesc *tdesc = new TransferDesc(*this,
                                           srcs,
                                           dsts,
                                           indirects,
                                           requests);

    // and now an operation that uses it
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event ev = finish_event->current_event();
    TransferOperation *op = new TransferOperation(*tdesc,
                                                  wait_on,
                                                  finish_event,
                                                  ID(ev).event_generation());
    get_runtime()->optable.add_local_operation(ev, op);
    op->start_or_defer();

    // remove our reference to the description (op holds one)
    tdesc->remove_reference();

    return ev;
  }

#define DOIT(N,T) \
  template Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField>&, \
				       const std::vector<CopySrcDstField>&, \
				       const std::vector<const CopyIndirection<N,T>::Base *>&, \
				       const ProfilingRequestSet&,	\
				       Event) const;			\
  template class TransferIteratorIndexSpace<N,T>; \
  template class TransferIteratorIndirect<N,T>; \
  template class TransferIteratorIndirectRange<N,T>; \
  template class TransferDomainIndexSpace<N,T>; \
  template class AddressSplitXferDesFactory<N,T>;
  FOREACH_NT(DOIT)

#define DOIT2(N,T,N2,T2) \
  template class IndirectionInfoTyped<N,T,N2,T2>; \
  template IndirectionInfo *CopyIndirection<N,T>::Unstructured<N2,T2>::create_info(const IndexSpace<N,T>& is) const;
  FOREACH_NTNT(DOIT2)

}; // namespace Realm
