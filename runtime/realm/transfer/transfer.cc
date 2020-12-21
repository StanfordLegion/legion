/* Copyright 2020 Stanford University, NVIDIA Corporation
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
  size_t TransferIterator::step(size_t max_bytes, AddressInfoHDF5& info,
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
    virtual size_t step(size_t max_bytes, AddressInfoHDF5& info,
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
  size_t TransferIteratorBase<N,T>::step(size_t max_bytes, AddressInfoHDF5& info,
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

      info.field_id = cur_field_id;
      info.filename = &hlp->filename;
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

      if(layout_piece->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType) {
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

    virtual size_t volume(void) const;

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
  size_t TransferDomainIndexSpace<N,T>::volume(void) const
  {
    return is.volume();
  }

  template <int N, typename T>
  TransferIterator *TransferDomainIndexSpace<N,T>::create_iterator(RegionInstance inst,
								   RegionInstance peer,
								   const std::vector<FieldID>& fields,
								   const std::vector<size_t>& fld_offsets,
								   const std::vector<size_t>& fld_sizes) const
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
    return new TransferIteratorIndexSpace<N,T>(is, inst, dim_order,
					       fields, fld_offsets, fld_sizes,
					       extra_elems);
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


  std::ostream& operator<<(std::ostream& os, const IndirectionInfo& ii)
  {
    ii.print(os);
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDesCreateMessage<AddressSplitXferDes<N,T>>
  //

  template <int N, typename T>
  class AddressSplitXferDes;
  
  template <int N, typename T>
  struct XferDesCreateMessage<AddressSplitXferDes<N,T> > : public XferDesCreateMessageBase {
  public:
    static void handle_message(NodeID sender,
			       const XferDesCreateMessage<AddressSplitXferDes<N,T> > &args,
			       const void *msgdata,
			       size_t msglen)
    {
      std::vector<XferDesPortInfo> inputs_info, outputs_info;
      bool mark_started = false;
      uint64_t max_req_size = 0;
      long max_nr = 0;
      int priority = 0;
      size_t element_size;
      std::vector<IndexSpace<N,T> > spaces;

      Realm::Serialization::FixedBufferDeserializer fbd(msgdata, msglen);

      bool ok = ((fbd >> inputs_info) &&
		 (fbd >> outputs_info) &&
		 (fbd >> mark_started) &&
		 (fbd >> max_req_size) &&
		 (fbd >> max_nr) &&
		 (fbd >> priority) &&
		 (fbd >> element_size) &&
		 (fbd >> spaces));
      assert(ok);
      assert(fbd.bytes_left() == 0);
  
      assert(!args.inst.exists());
      XferDes *xd = new AddressSplitXferDes<N,T>(args.dma_request,
						 args.launch_node,
						 args.guid,
						 inputs_info,
						 outputs_info,
						 mark_started,
						 max_req_size,
						 max_nr, priority,
						 args.complete_fence,
						 element_size,
						 spaces);

      xd->channel->enqueue_ready_xd(xd);
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

    virtual void create_xfer_des(DmaRequest *dma_request,
				 NodeID launch_node,
				 NodeID target_node,
				 XferDesID guid,
				 const std::vector<XferDesPortInfo>& inputs_info,
				 const std::vector<XferDesPortInfo>& outputs_info,
				 bool mark_started,
				 uint64_t max_req_size, long max_nr, int priority,
				 XferDesFence *complete_fence,
				 RegionInstance inst = RegionInstance::NO_INST);

    static ActiveMessageHandlerReg<XferDesCreateMessage<AddressSplitXferDes<N,T> > > areg;

  protected:
    size_t bytes_per_element;
    std::vector<IndexSpace<N,T> > spaces;
  };

  template <int N, typename T>
  /*static*/ ActiveMessageHandlerReg<XferDesCreateMessage<AddressSplitXferDes<N,T> > > AddressSplitXferDesFactory<N,T>::areg;
  
  template <int N, typename T>
  class AddressSplitXferDes : public AddressSplitXferDesBase {
  protected:
    friend class AddressSplitXferDesFactory<N,T>;
    friend struct XferDesCreateMessage<AddressSplitXferDes<N,T> >;
    
    AddressSplitXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
			const std::vector<XferDesPortInfo>& inputs_info,
			const std::vector<XferDesPortInfo>& outputs_info,
			bool _mark_start,
			uint64_t _max_req_size, long max_nr, int _priority,
			XferDesFence* _complete_fence,
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
  void AddressSplitXferDesFactory<N,T>::create_xfer_des(DmaRequest *dma_request,
							NodeID launch_node,
							NodeID target_node,
							XferDesID guid,
							const std::vector<XferDesPortInfo>& inputs_info,
							const std::vector<XferDesPortInfo>& outputs_info,
							bool mark_started,
							uint64_t max_req_size, long max_nr, int priority,
							XferDesFence *complete_fence,
							RegionInstance inst /*= RegionInstance::NO_INST*/)
  {
    if(target_node == Network::my_node_id) {
      // local creation
      assert(!inst.exists());
      XferDes *xd = new AddressSplitXferDes<N,T>(dma_request, launch_node, guid,
						 inputs_info, outputs_info,
						 mark_started,
						 max_req_size, max_nr, priority,
						 complete_fence,
						 bytes_per_element,
						 spaces);

      xd->channel->enqueue_ready_xd(xd);
    } else {
      // marking the transfer started has to happen locally
      if(mark_started)
	dma_request->mark_started();
      
      // remote creation
      Serialization::ByteCountSerializer bcs;
      {
	bool ok = ((bcs << inputs_info) &&
		   (bcs << outputs_info) &&
		   (bcs << false /*mark_started*/) &&
		   (bcs << max_req_size) &&
		   (bcs << max_nr) &&
		   (bcs << priority) &&
		   (bcs << bytes_per_element) &&
		   (bcs << spaces));
	assert(ok);
      }
      size_t req_size = bcs.bytes_used();
      ActiveMessage<XferDesCreateMessage<AddressSplitXferDes<N,T> > > amsg(target_node, req_size);
      amsg->inst = inst;
      amsg->complete_fence  = complete_fence;
      amsg->launch_node = launch_node;
      amsg->guid = guid;
      amsg->dma_request = dma_request;
      {
	bool ok = ((amsg << inputs_info) &&
		   (amsg << outputs_info) &&
		   (amsg << false /*mark_started*/) &&
		   (amsg << max_req_size) &&
		   (amsg << max_nr) &&
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

  AddressSplitXferDesBase::AddressSplitXferDesBase(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
						   const std::vector<XferDesPortInfo>& inputs_info,
						   const std::vector<XferDesPortInfo>& outputs_info,
						   bool _mark_start,
						   uint64_t _max_req_size, long max_nr, int _priority,
						   XferDesFence* _complete_fence)
    : XferDes(_dma_request, _launch_node, _guid, inputs_info, outputs_info,
	      _mark_start, _max_req_size, _priority, _complete_fence)
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
  AddressSplitXferDes<N,T>::AddressSplitXferDes(DmaRequest *_dma_request, NodeID _launch_node, XferDesID _guid,
						const std::vector<XferDesPortInfo>& inputs_info,
						const std::vector<XferDesPortInfo>& outputs_info,
						bool _mark_start,
						uint64_t _max_req_size, long max_nr, int _priority,
						XferDesFence* _complete_fence,
						size_t _element_size,
						const std::vector<IndexSpace<N,T> >& _spaces)
    : AddressSplitXferDesBase(_dma_request, _launch_node, _guid,
			      inputs_info, outputs_info,
			      _mark_start, _max_req_size, max_nr, _priority,
			      _complete_fence)
    , spaces(_spaces)
    , element_size(_element_size)
    , point_index(0), point_count(0)
    , output_space_id(-1), output_count(0)
  {
    channel = get_channel_manager()->get_address_split_channel();
  }

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
  // class AddressSplitChannel
  //

  AddressSplitChannel::AddressSplitChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<AddressSplitChannel, AddressSplitXferDesBase>(bgwork,
								     XFER_ADDR_SPLIT,
								     "address split")
  {}

  AddressSplitChannel::~AddressSplitChannel()
  {}

  
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
    virtual Memory generate_gather_paths(Memory dst_mem, int dst_edge_id,
					 size_t bytes_per_element,
					 CustomSerdezID serdez_id,
					 std::vector<CopyRequest::XDTemplate>& xd_nodes,
					 std::vector<IBInfo>& ib_edges);

    virtual Memory generate_scatter_paths(Memory src_mem, int src_edge_id,
					  size_t bytes_per_element,
					  CustomSerdezID serdez_id,
					  std::vector<CopyRequest::XDTemplate>& xd_nodes,
					  std::vector<IBInfo>& ib_edges);

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

  // adds a memory path to the DAG, optionally skipping a final memcpy
  static int add_copy_path(std::vector<CopyRequest::XDTemplate>& xd_nodes,
			   std::vector<IBInfo>& ib_edges,
			   int start_edge,
			   const MemPathInfo& info,
			   bool skip_final_memcpy)
  {
    size_t hops = info.xd_kinds.size();
    if(skip_final_memcpy &&
       (info.xd_kinds[hops - 1] == XFER_MEM_CPY))
      hops -= 1;

    if(hops == 0) {
      // no xd's needed at all - return input edge as output
      return start_edge;
    }

    size_t xd_base = xd_nodes.size();
    size_t ib_base = ib_edges.size();
    xd_nodes.resize(xd_base + hops);
    ib_edges.resize(ib_base + hops);
    for(size_t i = 0; i < hops; i++) {
      xd_nodes[xd_base + i].set_simple(info.xd_target_nodes[i],
				       info.xd_kinds[i],
				       (i == 0) ? start_edge :
				                  (ib_base + i - 1),
				       ib_base + i);
      ib_edges[ib_base + i].set(info.path[i + 1],
				65536); // TODO: pick size?
    }

    // last edge we created is the output
    return (ib_base + hops - 1);
  }

  
  template <int N, typename T, int N2, typename T2>
  Memory IndirectionInfoTyped<N,T,N2,T2>::generate_gather_paths(Memory dst_mem,
								int dst_edge_id,
								size_t bytes_per_element,
								CustomSerdezID serdez_id,
								std::vector<CopyRequest::XDTemplate>& xd_nodes,
								std::vector<IBInfo>& ib_edges)
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
				     path_infos[idx]);
	assert(ok);
      }
    }

    // special case - a gather from a single source with no out of range
    //  accesses
    if((spaces.size() == 1) && !oor_possible) {
      size_t pathlen = path_infos[0].xd_kinds.size();
      // HACK!
      Memory local_ib_mem = ID::make_ib_memory(path_infos[0].xd_target_nodes[0], 0).convert<Memory>();
      // do we have to do anything to get the addresses into a cpu-readable
      //  memory on that node?
      MemPathInfo addr_path;
      bool ok = find_shortest_path(inst.get_location(),
				   local_ib_mem,
				   0 /*no serdez*/,
				   addr_path);
      assert(ok);
      int addr_edge = add_copy_path(xd_nodes, ib_edges,
				    CopyRequest::XDTemplate::SRC_INST,
				    addr_path, true /*skip_memcpy*/);

      size_t xd_idx = xd_nodes.size();
      size_t ib_idx = ib_edges.size();
      xd_nodes.resize(xd_idx + pathlen);
      ib_edges.resize(ib_idx + pathlen - 1);

      for(size_t i = 0; i < pathlen; i++) {
	xd_nodes[xd_idx+i].target_node = path_infos[0].xd_target_nodes[i];
	xd_nodes[xd_idx+i].kind = path_infos[0].xd_kinds[i];
	xd_nodes[xd_idx+i].factory = get_xd_factory_by_kind(path_infos[0].xd_kinds[i]);
	xd_nodes[xd_idx+i].gather_control_input = -1;
	xd_nodes[xd_idx+i].scatter_control_input = -1;
	if(i == 0) {
	  xd_nodes[xd_idx+i].inputs.resize(2);
	  xd_nodes[xd_idx+i].inputs[0].edge_id = CopyRequest::XDTemplate::INDIRECT_BASE - 1;
	  xd_nodes[xd_idx+i].inputs[0].indirect_inst = insts[0];
	  xd_nodes[xd_idx+i].inputs[1].edge_id = addr_edge;
	  xd_nodes[xd_idx+i].inputs[1].indirect_inst = RegionInstance::NO_INST;
	} else {
	  xd_nodes[xd_idx+i].inputs.resize(1);
	  xd_nodes[xd_idx+i].inputs[0].edge_id = ib_idx + (i - 1);
	  xd_nodes[xd_idx+i].inputs[0].indirect_inst = RegionInstance::NO_INST;
	}
	if(i == (pathlen - 1)) {
	  xd_nodes[xd_idx+i].outputs.resize(1);
	  xd_nodes[xd_idx+i].outputs[0].edge_id = dst_edge_id;
	  xd_nodes[xd_idx+i].outputs[0].indirect_inst = RegionInstance::NO_INST;
	} else {
	  xd_nodes[xd_idx+i].outputs.resize(1);
	  xd_nodes[xd_idx+i].outputs[0].edge_id = ib_idx + i;
	  xd_nodes[xd_idx+i].outputs[0].indirect_inst = RegionInstance::NO_INST;

	  ib_edges[ib_idx+i].set(path_infos[0].path[i + 1],
				 1 << 20 /*TODO*/);
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
				   addr_path);
      assert(ok);
      int addr_edge = add_copy_path(xd_nodes, ib_edges,
				    CopyRequest::XDTemplate::SRC_INST,
				    addr_path, true /*skip_memcpy*/);
      std::vector<int> decoded_addr_edges(spaces.size(), -1);
      int ctrl_edge = -1;
      {
	// instantiate decoder
	size_t xd_base = xd_nodes.size();
	size_t ib_base = ib_edges.size();
	xd_nodes.resize(xd_base + 1);
	ib_edges.resize(ib_base + spaces.size() + 1);
	xd_nodes[xd_base].target_node = addr_node;
	xd_nodes[xd_base].kind = XFER_ADDR_SPLIT;
	assert(!is_ranges && "need range address splitter");
	xd_nodes[xd_base].factory = new AddressSplitXferDesFactory<N2,T2>(bytes_per_element,
									  spaces);
	xd_nodes[xd_base].gather_control_input = -1;
	xd_nodes[xd_base].scatter_control_input = -1;
	xd_nodes[xd_base].inputs.resize(1);
	xd_nodes[xd_base].inputs[0].edge_id = addr_edge;
	xd_nodes[xd_base].inputs[0].indirect_inst = RegionInstance::NO_INST;
	xd_nodes[xd_base].outputs.resize(spaces.size() + 1);
	for(size_t i = 0; i < spaces.size(); i++) {
	  xd_nodes[xd_base].outputs[i].edge_id = ib_base + i;
	  xd_nodes[xd_base].outputs[i].indirect_inst = RegionInstance::NO_INST;
	  ib_edges[ib_base + i].set(addr_ib_mem, 65536); // TODO
	  decoded_addr_edges[i] = ib_base + i;
	}
	xd_nodes[xd_base].outputs[spaces.size()].edge_id = ib_base + spaces.size();
	xd_nodes[xd_base].outputs[spaces.size()].indirect_inst = RegionInstance::NO_INST;
	ib_edges[ib_base + spaces.size()].set(addr_ib_mem, 65536); // TODO
	ctrl_edge = ib_base + spaces.size();
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
				       path);
	  assert(ok);
	  decoded_addr_edges[i] = add_copy_path(xd_nodes, ib_edges,
						decoded_addr_edges[i],
						path, false /*!skip_memcpy*/);
	}
      }

      // control information has to get to the merge at the end
      // HACK!
      Memory dst_ib_mem = ID::make_ib_memory(ID(dst_mem).memory_owner_node(), 0).convert<Memory>();
      if(dst_ib_mem != addr_ib_mem) {
	MemPathInfo path;
	bool ok = find_shortest_path(addr_ib_mem, dst_ib_mem,
				     0 /*no serdez*/,
				     path);
	assert(ok);
	ctrl_edge = add_copy_path(xd_nodes, ib_edges,
				  ctrl_edge,
				  path, false /*!skip_memcpy*/);
      }

      // next complication: if all the data paths don't use the same final
      //  step, we need to force them to go through an intermediate
      XferDesKind last_kind = path_infos[0].xd_kinds[path_infos[0].xd_kinds.size() - 1];
      bool same_last_kind = true;
      for(size_t i = 1; i < path_infos.size(); i++)
	if(path_infos[i].xd_kinds[path_infos[i].xd_kinds.size() - 1] !=
	   last_kind) {
	  same_last_kind = false;
	  break;
	}
      if(!same_last_kind) {
	// figure out what the final kind will be (might not be the same as
	//  any of the current paths)
	MemPathInfo tail_path;
	bool ok = find_shortest_path(dst_ib_mem, dst_mem,
				     serdez_id, tail_path);
	assert(ok && (tail_path.xd_kinds.size() == 1));
	last_kind = tail_path.xd_kinds[0];
	// and fix any path that doesn't use that kind
	for(size_t i = 0; i < path_infos.size(); i++) {
	  if(path_infos[i].xd_kinds[path_infos[i].xd_kinds.size() - 1] ==
	     last_kind) continue;
	  log_new_dma.print() << "fix " << i << " " << path_infos[i].path[0] << " -> " << dst_ib_mem;
	  bool ok = find_shortest_path(path_infos[i].path[0], dst_ib_mem,
				       0 /*no serdez*/, path_infos[i]);
	  assert(ok);
	  // append last step
	  path_infos[i].xd_kinds.push_back(last_kind);
	  path_infos[i].path.push_back(dst_mem);
	  path_infos[i].xd_target_nodes.push_back(ID(dst_mem).memory_owner_node());
	}
      }

      // now any data paths with more than one hop need all but the last hop
      //  added to the graph
      std::vector<int> data_edges(spaces.size(), -1);
      for(size_t i = 0; i < spaces.size(); i++) {
	const MemPathInfo& mpi = path_infos[path_idx[i]];
	size_t hops = mpi.xd_kinds.size() - 1;
	if(hops > 0) {
	  size_t xd_base = xd_nodes.size();
	  size_t ib_base = ib_edges.size();
	  xd_nodes.resize(xd_base + hops);
	  ib_edges.resize(ib_base + hops);
	  for(size_t j = 0; j < hops; j++) {
	    xd_nodes[xd_base + j].set_simple(mpi.xd_target_nodes[j],
					     mpi.xd_kinds[j],
					     (j == 0) ? 0 /*fixed below*/ :
					                (ib_base + j - 1),
					     ib_base + j);
	    if(j == 0) {
	      xd_nodes[xd_base + j].inputs.resize(2);
	      xd_nodes[xd_base + j].inputs[0].edge_id = CopyRequest::XDTemplate::INDIRECT_BASE - 1;
	      xd_nodes[xd_base + j].inputs[0].indirect_inst = insts[i];
	      xd_nodes[xd_base + j].inputs[1].edge_id = decoded_addr_edges[i];
	      xd_nodes[xd_base + j].inputs[1].indirect_inst = RegionInstance::NO_INST;
	    }
	    ib_edges[ib_base + j].set(mpi.path[j + 1],
				      65536); // TODO: pick size?
	  }
	  data_edges[i] = ib_base + hops - 1;
	}
      }

      // and finally the last xd that merges the streams together
      size_t xd_idx = xd_nodes.size();
      xd_nodes.resize(xd_idx + 1);

      xd_nodes[xd_idx].target_node = ID(dst_mem).memory_owner_node();
      xd_nodes[xd_idx].kind = last_kind;
      xd_nodes[xd_idx].factory = get_xd_factory_by_kind(last_kind);
      xd_nodes[xd_idx].gather_control_input = spaces.size();
      xd_nodes[xd_idx].scatter_control_input = -1;
      xd_nodes[xd_idx].outputs.resize(1);
      xd_nodes[xd_idx].outputs[0].edge_id = dst_edge_id;
      xd_nodes[xd_idx].outputs[0].indirect_inst = RegionInstance::NO_INST;
      xd_nodes[xd_idx].inputs.resize(spaces.size() + 1);

      for(size_t i = 0; i < spaces.size(); i++) {
	// can we read (indirectly) right from the source instance?
	if(path_infos[path_idx[i]].xd_kinds.size() == 1) {
	  int ind_idx = xd_nodes[xd_idx].inputs.size();
	  xd_nodes[xd_idx].inputs.resize(ind_idx + 1);
	  xd_nodes[xd_idx].inputs[i].edge_id = CopyRequest::XDTemplate::INDIRECT_BASE - ind_idx;
	  xd_nodes[xd_idx].inputs[i].indirect_inst = insts[i];
	  xd_nodes[xd_idx].inputs[ind_idx].edge_id = decoded_addr_edges[i];
	  xd_nodes[xd_idx].inputs[ind_idx].indirect_inst = RegionInstance::NO_INST;
	} else {
	  assert(data_edges[i] >= 0);
	  xd_nodes[xd_idx].inputs[i].edge_id = data_edges[i];
	  xd_nodes[xd_idx].inputs[i].indirect_inst = RegionInstance::NO_INST;
	}
      }

      // control input
      xd_nodes[xd_idx].inputs[spaces.size()].edge_id = ctrl_edge;
      xd_nodes[xd_idx].inputs[spaces.size()].indirect_inst = RegionInstance::NO_INST;      
    }
    return Memory::NO_MEMORY;
  }

  template <int N, typename T, int N2, typename T2>
  Memory IndirectionInfoTyped<N,T,N2,T2>::generate_scatter_paths(Memory src_mem,
								 int src_edge_id,
								 size_t bytes_per_element,
								 CustomSerdezID serdez_id,
								 std::vector<CopyRequest::XDTemplate>& xd_nodes,
								 std::vector<IBInfo>& ib_edges)
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
				     path_infos[idx]);
	assert(ok);
      }
    }

    // special case - a scatter to a single destination with no out of
    //  range accesses
    if((spaces.size() == 1) && !oor_possible) {
      size_t pathlen = path_infos[0].xd_kinds.size();
      // HACK!
      Memory local_ib_mem = ID::make_ib_memory(path_infos[0].xd_target_nodes[pathlen - 1], 0).convert<Memory>();
      // do we have to do anything to get the addresses into a cpu-readable
      //  memory on that node?
      MemPathInfo addr_path;
      bool ok = find_shortest_path(inst.get_location(),
				   local_ib_mem,
				   0 /*no serdez*/,
				   addr_path);
      assert(ok);
      int addr_edge = add_copy_path(xd_nodes, ib_edges,
				    CopyRequest::XDTemplate::DST_INST,
				    addr_path, true /*skip_memcpy*/);

      size_t xd_idx = xd_nodes.size();
      size_t ib_idx = ib_edges.size();
      xd_nodes.resize(xd_idx + pathlen);
      ib_edges.resize(ib_idx + pathlen - 1);

      for(size_t i = 0; i < pathlen; i++) {
	xd_nodes[xd_idx+i].target_node = path_infos[0].xd_target_nodes[i];
	xd_nodes[xd_idx+i].kind = path_infos[0].xd_kinds[i];
	xd_nodes[xd_idx+i].factory = get_xd_factory_by_kind(path_infos[0].xd_kinds[i]);
	xd_nodes[xd_idx+i].gather_control_input = -1;
	xd_nodes[xd_idx+i].scatter_control_input = -1;
	if(i == (pathlen - 1)) {
	  xd_nodes[xd_idx+i].inputs.resize(2);
	  xd_nodes[xd_idx+i].inputs[1].edge_id = addr_edge;
	  xd_nodes[xd_idx+i].inputs[1].indirect_inst = RegionInstance::NO_INST;
	} else {
	  xd_nodes[xd_idx+i].inputs.resize(1);
	}
	xd_nodes[xd_idx+i].inputs[0].edge_id = ((i == 0) ? src_edge_id :
						           (ib_idx + i - 1));
	xd_nodes[xd_idx+i].inputs[0].indirect_inst = RegionInstance::NO_INST;

	if(i == (pathlen - 1)) {
	  xd_nodes[xd_idx+i].outputs.resize(1);
	  xd_nodes[xd_idx+i].outputs[0].edge_id = CopyRequest::XDTemplate::INDIRECT_BASE - 1;
	  xd_nodes[xd_idx+i].outputs[0].indirect_inst = insts[0];
	} else {
	  xd_nodes[xd_idx+i].outputs.resize(1);
	  xd_nodes[xd_idx+i].outputs[0].edge_id = ib_idx + i;
	  xd_nodes[xd_idx+i].outputs[0].indirect_inst = RegionInstance::NO_INST;

	  ib_edges[ib_idx+i].set(path_infos[0].path[i + 1],
				 1 << 20 /*TODO*/);
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
				   addr_path);
      assert(ok);
      int addr_edge = add_copy_path(xd_nodes, ib_edges,
				    CopyRequest::XDTemplate::DST_INST,
				    addr_path, true /*skip_memcpy*/);
      std::vector<int> decoded_addr_edges(spaces.size(), -1);
      int ctrl_edge = -1;
      {
	// instantiate decoder
	size_t xd_base = xd_nodes.size();
	size_t ib_base = ib_edges.size();
	xd_nodes.resize(xd_base + 1);
	ib_edges.resize(ib_base + spaces.size() + 1);
	xd_nodes[xd_base].target_node = addr_node;
	xd_nodes[xd_base].kind = XFER_ADDR_SPLIT;
	assert(!is_ranges && "need range address splitter");
	xd_nodes[xd_base].factory = new AddressSplitXferDesFactory<N2,T2>(bytes_per_element,
									  spaces);
	xd_nodes[xd_base].gather_control_input = -1;
	xd_nodes[xd_base].scatter_control_input = -1;
	xd_nodes[xd_base].inputs.resize(1);
	xd_nodes[xd_base].inputs[0].edge_id = addr_edge;
	xd_nodes[xd_base].inputs[0].indirect_inst = RegionInstance::NO_INST;
	xd_nodes[xd_base].outputs.resize(spaces.size() + 1);
	for(size_t i = 0; i < spaces.size(); i++) {
	  xd_nodes[xd_base].outputs[i].edge_id = ib_base + i;
	  xd_nodes[xd_base].outputs[i].indirect_inst = RegionInstance::NO_INST;
	  ib_edges[ib_base + i].set(addr_ib_mem, 65536); // TODO
	  decoded_addr_edges[i] = ib_base + i;
	}
	xd_nodes[xd_base].outputs[spaces.size()].edge_id = ib_base + spaces.size();
	xd_nodes[xd_base].outputs[spaces.size()].indirect_inst = RegionInstance::NO_INST;
	ib_edges[ib_base + spaces.size()].set(addr_ib_mem, 65536); // TODO
	ctrl_edge = ib_base + spaces.size();
      }

      // next, see what work we need to get the addresses to where the
      //  target data instances live
      for(size_t i = 0; i < spaces.size(); i++) {
	// HACK!
	Memory dst_ib_mem = ID::make_ib_memory(ID(insts[i]).instance_owner_node(), 0).convert<Memory>();
	if(dst_ib_mem != addr_ib_mem) {
	  MemPathInfo path;
	  bool ok = find_shortest_path(addr_ib_mem, dst_ib_mem,
				       0 /*no serdez*/,
				       path);
	  assert(ok);
	  decoded_addr_edges[i] = add_copy_path(xd_nodes, ib_edges,
						decoded_addr_edges[i],
						path, false /*!skip_memcpy*/);
	}
      }

      // control information has to get to the split at the start
      // HACK!
      Memory src_ib_mem = ID::make_ib_memory(ID(src_mem).memory_owner_node(), 0).convert<Memory>();
      if(src_ib_mem != addr_ib_mem) {
	MemPathInfo path;
	bool ok = find_shortest_path(addr_ib_mem, src_ib_mem,
				     0 /*no serdez*/,
				     path);
	assert(ok);
	ctrl_edge = add_copy_path(xd_nodes, ib_edges,
				  ctrl_edge,
				  path, false /*!skip_memcpy*/);
      }

      // next complication: if all the data paths don't use the same first
      //  step, we need to force them to go through an intermediate
      XferDesKind first_kind = path_infos[0].xd_kinds[0];
      bool same_first_kind = true;
      for(size_t i = 1; i < path_infos.size(); i++)
	if(path_infos[i].xd_kinds[0] != first_kind) {
	  same_first_kind = false;
	  break;
	}
      if(!same_first_kind) {
	// figure out what the first kind will be (might not be the same as
	//  any of the current paths)
	MemPathInfo head_path;
	bool ok = find_shortest_path(src_mem, src_ib_mem,
				     serdez_id, head_path);
	assert(ok && (head_path.xd_kinds.size() == 1));
	first_kind = head_path.xd_kinds[0];
	// and fix any path that doesn't use that kind
	for(size_t i = 0; i < path_infos.size(); i++) {
	  if(path_infos[i].xd_kinds[0] == first_kind) continue;

	  bool ok = find_shortest_path(src_ib_mem,
				       path_infos[i].path[path_infos[i].path.size() - 1],
				       0 /*no serdez*/, path_infos[i]);
	  assert(ok);
	  // prepend last step
	  path_infos[i].xd_kinds.insert(path_infos[i].xd_kinds.begin(),
					first_kind);
	  path_infos[i].path.insert(path_infos[i].path.begin(), src_mem);
	  path_infos[i].xd_target_nodes.insert(path_infos[i].xd_target_nodes.begin(),
					       ID(src_mem).memory_owner_node());
	}
      }

      // next comes the xd that reads the source and splits the data into
      //  the various output streams
      std::vector<int> data_edges(spaces.size(), -1);
      {
	size_t xd_idx = xd_nodes.size();
	xd_nodes.resize(xd_idx + 1);

	xd_nodes[xd_idx].target_node = ID(src_mem).memory_owner_node();
	xd_nodes[xd_idx].kind = first_kind;
	xd_nodes[xd_idx].factory = get_xd_factory_by_kind(first_kind);
	xd_nodes[xd_idx].gather_control_input = -1;
	xd_nodes[xd_idx].scatter_control_input = 1;
	xd_nodes[xd_idx].inputs.resize(2);
	xd_nodes[xd_idx].inputs[0].edge_id = src_edge_id;
	xd_nodes[xd_idx].inputs[0].indirect_inst = RegionInstance::NO_INST;
	xd_nodes[xd_idx].inputs[1].edge_id = ctrl_edge;
	xd_nodes[xd_idx].inputs[1].indirect_inst = RegionInstance::NO_INST;
	xd_nodes[xd_idx].outputs.resize(spaces.size());

	for(size_t i = 0; i < spaces.size(); i++) {
	  // can we write (indirectly) right into the dest instance?
	  if(path_infos[path_idx[i]].xd_kinds.size() == 1) {
	    int ind_idx = xd_nodes[xd_idx].inputs.size();
	    xd_nodes[xd_idx].inputs.resize(ind_idx + 1);
	    xd_nodes[xd_idx].outputs[i].edge_id = CopyRequest::XDTemplate::INDIRECT_BASE - ind_idx;
	    xd_nodes[xd_idx].outputs[i].indirect_inst = insts[i];
	    xd_nodes[xd_idx].inputs[ind_idx].edge_id = decoded_addr_edges[i];
	    xd_nodes[xd_idx].inputs[ind_idx].indirect_inst = RegionInstance::NO_INST;
	  } else {
	    // need an ib to write to
	    size_t ib_idx = ib_edges.size();
	    ib_edges.resize(ib_idx + 1);
	    data_edges[i] = ib_idx;
	    xd_nodes[xd_idx].outputs[i].edge_id = data_edges[i];
	    xd_nodes[xd_idx].outputs[i].indirect_inst = RegionInstance::NO_INST;
	    ib_edges[ib_idx].set(path_infos[path_idx[i]].path[1],
				 65536); // TODO: pick size?
	  }
	}

	// control input
	xd_nodes[xd_idx].inputs[1].edge_id = ctrl_edge;
	xd_nodes[xd_idx].inputs[1].indirect_inst = RegionInstance::NO_INST;
      }

      // finally, any data paths with more than one hop need the rest of
      //  their path added to the graph
      for(size_t i = 0; i < spaces.size(); i++) {
	const MemPathInfo& mpi = path_infos[path_idx[i]];
	size_t hops = mpi.xd_kinds.size() - 1;
	if(hops > 0) {
	  assert(data_edges[i] >= 0);
	  
	  size_t xd_base = xd_nodes.size();
	  size_t ib_base = ib_edges.size();
	  xd_nodes.resize(xd_base + hops);
	  ib_edges.resize(ib_base + hops - 1);
	  for(size_t j = 0; j < hops; j++) {
	    xd_nodes[xd_base + j].set_simple(mpi.xd_target_nodes[j + 1],
					     mpi.xd_kinds[j + 1],
					     (j == 0) ? data_edges[i] :
					                (ib_base + j - 1),
					     ib_base + j);
	    if(j < hops - 1) {
	      ib_edges[ib_base + j].set(mpi.path[j + 2],
					65536); // TODO: pick size?
	    } else {
	      // last hop uses the address stream
	      xd_nodes[xd_base + j].inputs.resize(2);
	      xd_nodes[xd_base + j].inputs[1].edge_id = decoded_addr_edges[i];
	      xd_nodes[xd_base + j].inputs[1].indirect_inst = RegionInstance::NO_INST;
	      xd_nodes[xd_base + j].outputs[0].edge_id = CopyRequest::XDTemplate::INDIRECT_BASE - 1;
	      xd_nodes[xd_base + j].outputs[0].indirect_inst = insts[i];
	    }
	  }
	}
      }
    }
    return Memory::NO_MEMORY;
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

  class TransferPlan {
  protected:
    // subclasses constructed in plan_* calls below
    TransferPlan(void) {}

  public:
    virtual ~TransferPlan(void) {}

    static bool plan_copy(std::vector<TransferPlan *>& plans,
			  const std::vector<CopySrcDstField> &srcs,
			  const std::vector<CopySrcDstField> &dsts,
			  ReductionOpID redop_id = 0, bool red_fold = false);

    static bool plan_fill(std::vector<TransferPlan *>& plans,
			  const std::vector<CopySrcDstField> &dsts,
			  const void *fill_value, size_t fill_value_size);

    virtual void execute_plan() = 0;

    virtual Event create_plan(const TransferDomain *td,
                              const ProfilingRequestSet& requests,
                              Event wait_on, int priority) = 0;

  };

  class TransferPlanCopy : public TransferPlan {
  public:
    TransferPlanCopy(OASByInst *_oas_by_inst,
		     IndirectionInfo *_gather_info,
		     IndirectionInfo *_scatter_info);
    virtual ~TransferPlanCopy(void);

    virtual void execute_plan();

    virtual Event create_plan(const TransferDomain *td,
                              const ProfilingRequestSet& requests,
                              Event wait_on, int priority);
  protected:
    OASByInst *oas_by_inst;
    IndirectionInfo *gather_info;
    IndirectionInfo *scatter_info;
    Event ev;
    CopyRequest *r;
  };

  TransferPlanCopy::TransferPlanCopy(OASByInst *_oas_by_inst,
				     IndirectionInfo *_gather_info,
				     IndirectionInfo *_scatter_info)
    : oas_by_inst(_oas_by_inst)
    , gather_info(_gather_info)
    , scatter_info(_scatter_info)
    , ev(Event::NO_EVENT)
    , r(NULL)
  {}

  TransferPlanCopy::~TransferPlanCopy(void)
  {
    delete oas_by_inst;
    delete gather_info;
    delete scatter_info;
  }

  class TransferProfile {
    CopyProfile *r;
  public:
    TransferProfile(): r(NULL) {}

    void add_reduc_entry(const CopySrcDstField &src,
                         const CopySrcDstField &dst,
                         const TransferDomain *td)
    {
      assert(r != NULL);
      r->add_reduc_entry(src,dst,td);
    }

    void add_fill_entry(const CopySrcDstField &dst, const TransferDomain *td)
    {
      assert(r != NULL);
      r->add_fill_entry(dst,td);
    }

    void add_copy_entry(const OASByInst *oas_by_inst,const TransferDomain *td, bool is_src_indirect, bool is_dst_indirect)
    {
      assert(r != NULL);
      r->add_copy_entry(oas_by_inst, td, is_src_indirect, is_dst_indirect);
    }

    Event create_profile(const ProfilingRequestSet &requests)
    {
      GenEventImpl *after_copy = GenEventImpl::create_genevent();
      Event ev = after_copy->current_event();
      EventImpl::gen_t after_gen = ID(ev).event_generation();
      r = new CopyProfile(after_copy,
                          after_gen, 0, requests);
      get_runtime()->optable.add_local_operation(ev, r);
      return ev;
    }

    void execute_plan() { assert(0); return; }
    Event create_plan(const TransferDomain *td,
                      const ProfilingRequestSet& requests,
                      Event wait_on, int priority)
    { assert(0); return Event::NO_EVENT; }
    void execute(Event end_copy, Event start_copy)
    {
      assert(r != NULL);
      r->set_end_copy(end_copy);
      r->set_start_copy(start_copy);
      r->check_readiness();
    }
    ~TransferProfile() {}
  };


#ifdef MIGRATE_COPY_REQUESTS
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
	log_dma.warning("WARNING: gasnet->gasnet copy being serialized on local node (%d)", Network::my_node_id);
	return Network::my_node_id;
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
#endif

    Event TransferPlanCopy::create_plan(const TransferDomain *td,
                                        const ProfilingRequestSet& requests,
                                        Event wait_on, int priority)
  {
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    ev = finish_event->current_event();
    r = new CopyRequest(td, oas_by_inst,
                                     gather_info, scatter_info,
                                     wait_on,
                                     finish_event,
                                     ID(ev).event_generation(),
                                     priority, requests);
    return ev;
  }

  void TransferPlanCopy::execute_plan()
  {
#ifdef MIGRATE_COPY_REQUESTS
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
#else
    NodeID dma_node = Network::my_node_id;
#endif
    // we've given oas_by_inst and indirection info to the copyrequest, so forget it
    assert(oas_by_inst != 0);
    oas_by_inst = 0;
    gather_info = 0;
    scatter_info = 0;
    if(dma_node == Network::my_node_id) {
      log_dma.debug("performing copy on local node");

      get_runtime()->optable.add_local_operation(ev, r);
      r->check_readiness();
    } else {
      r->forward_request(dma_node);
      get_runtime()->optable.add_remote_operation(ev, dma_node);
      // done with the local copy of the request
      r->remove_reference();
    }
  }

  class TransferPlanReduce : public TransferPlan {
  public:
    TransferPlanReduce(const CopySrcDstField& _src,
		       const CopySrcDstField& _dst,
		       ReductionOpID _redop_id, bool _red_fold);

    virtual void execute_plan();

    virtual Event create_plan(const TransferDomain *td,
                              const ProfilingRequestSet& requests,
                              Event wait_on, int priority);

  protected:
    CopySrcDstField src;
    CopySrcDstField dst;
    ReductionOpID redop_id;
    bool red_fold;
    Event ev;
    ReduceRequest *r;
  };

  TransferPlanReduce::TransferPlanReduce(const CopySrcDstField& _src,
					 const CopySrcDstField& _dst,
					 ReductionOpID _redop_id, bool _red_fold)
    : src(_src)
    , dst(_dst)
    , redop_id(_redop_id)
    , red_fold(_red_fold)
    , ev(Event::NO_EVENT)
    , r(NULL)
  {}

  Event TransferPlanReduce::create_plan(const TransferDomain *td,
                                        const ProfilingRequestSet& requests,
                                        Event wait_on, int priority)
  {
    // TODO
    bool inst_lock_needed = false;
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    ev = finish_event->current_event();
    r = new ReduceRequest(td,
                          std::vector<CopySrcDstField>(1, src),
                          dst,
                          inst_lock_needed,
                          redop_id, red_fold,
                          wait_on,
                          finish_event,
                          ID(ev).event_generation(),
                          0 /*priority*/, requests);
    return ev;
  }

  void TransferPlanReduce::execute_plan()
  {
    NodeID src_node = ID(src.inst).instance_owner_node();
    if(src_node == Network::my_node_id) {
      log_dma.debug("performing reduction on local node");

      get_runtime()->optable.add_local_operation(ev, r);
      r->set_dma_queue(dma_queue);
      r->check_readiness();
    } else {
      r->forward_request(src_node);
      get_runtime()->optable.add_remote_operation(ev, src_node);
      // done with the local copy of the request
      r->remove_reference();
    }
  }

  class TransferPlanFill : public TransferPlan {
  public:
    TransferPlanFill(const void *_data, size_t _size,
		     RegionInstance _inst, FieldID _field_id);

    virtual void execute_plan();

    virtual Event create_plan(const TransferDomain *td,
                              const ProfilingRequestSet& requests,
                              Event wait_on, int priority);


  protected:
    ByteArray data;
    RegionInstance inst;
    FieldID field_id;
    Event ev;
    FillRequest *r;
  };

  TransferPlanFill::TransferPlanFill(const void *_data, size_t _size,
				     RegionInstance _inst, FieldID _field_id)
    : data(_data, _size)
    , inst(_inst)
    , field_id(_field_id)
    , ev(Event::NO_EVENT)
    , r(NULL)
  {}

  Event TransferPlanFill::create_plan(const TransferDomain *td,
                                     const ProfilingRequestSet& requests,
                                     Event wait_on, int priority)
  {
    CopySrcDstField f;
    f.inst = inst;
    f.field_id = field_id;
    f.subfield_offset = 0;
    f.size = data.size();

    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    ev = finish_event->current_event();
    r = new FillRequest(td, f, data.base(), data.size(),
                                     wait_on,
                                     finish_event,
                                     ID(ev).event_generation(),
                                     priority, requests);
    return ev;
  }


  void TransferPlanFill::execute_plan()
  {
    NodeID tgt_node = ID(inst).instance_owner_node();
    if(tgt_node == Network::my_node_id) {
      get_runtime()->optable.add_local_operation(ev, r);
      r->set_dma_queue(dma_queue);
      r->check_readiness();
    } else {
      r->forward_request(tgt_node);
      get_runtime()->optable.add_remote_operation(ev, tgt_node);
      // release local copy of operation
      r->remove_reference();
    }
  }

  static void find_mergeable_fields(int idx,
				    const std::vector<CopySrcDstField>& srcs,
				    const std::vector<CopySrcDstField>& dsts,
				    const std::vector<bool>& merged,
				    std::set<int>& fields_to_merge)
  {
    for(size_t i = idx+1; i < srcs.size(); i++) {
      if(merged[i]) continue;
      if(srcs[i].inst != srcs[idx].inst) continue;
      if(srcs[i].serdez_id != srcs[idx].serdez_id) continue;
      if(srcs[i].indirect_index != srcs[idx].indirect_index) continue;
      if(dsts[i].inst != dsts[idx].inst) continue;
      if(dsts[i].serdez_id != dsts[idx].serdez_id) continue;
      if(dsts[i].indirect_index != dsts[idx].indirect_index) continue;
      // only merge fields of the same size to avoid issues with wrapping
      //  around in intermediate buffers
      if(srcs[i].size != srcs[idx].size) continue;
      if(dsts[i].size != dsts[idx].size) continue;
      fields_to_merge.insert(i);
    }
  }


			      
  template <int N, typename T>
  Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField>& srcs,
			      const std::vector<CopySrcDstField>& dsts,
			      const std::vector<const typename CopyIndirection<N,T>::Base *> &indirects,
			      const Realm::ProfilingRequestSet &requests,
			      Event wait_on) const
  {
    TransferProfile tp;
    bool prof_requests=false;
    Event prof_ev = Event::NO_EVENT;
    if (!requests.empty()) {
      prof_ev = tp.create_profile(requests);
      prof_requests = true;
    }

    TransferDomain *td = TransferDomain::construct(*this);
    std::vector<TransferPlan *> plans;
    assert(srcs.size() == dsts.size());
    std::vector<bool> merged(srcs.size(), false);
    for(size_t i = 0; i < srcs.size(); i++) {
      // did we already do this field?
      if(merged[i]) continue;

      assert(srcs[i].size == dsts[i].size);

      IndirectionInfo *src_indirect = 0;
      if(srcs[i].indirect_index != -1)
	src_indirect = indirects[srcs[i].indirect_index]->create_info(*this);

      IndirectionInfo *dst_indirect = 0;
      if(dsts[i].indirect_index != -1)
	dst_indirect = indirects[dsts[i].indirect_index]->create_info(*this);

      // if the source field id is -1 and dst has no redop, we can use old fill
      if(srcs[i].field_id == FieldID(-1)) {
	// no support for reduction fill yet
	assert(dsts[i].redop_id == 0);
	assert(src_indirect == 0);
	assert(dst_indirect == 0);
	TransferPlan *p = new TransferPlanFill(((srcs[i].size <= srcs[i].MAX_DIRECT_SIZE) ?
						  &(srcs[i].fill_data.direct) :
						  srcs[i].fill_data.indirect),
					       srcs[i].size,
					       dsts[i].inst,
					       dsts[i].field_id);
	plans.push_back(p);
        if (prof_requests)
          tp.add_fill_entry(dsts[i], td);
	continue;
      }

      // if the dst has a reduction op, do a reduce
      if(dsts[i].redop_id != 0) {
	assert(src_indirect == 0);
	assert(dst_indirect == 0);
	TransferPlan *p = new TransferPlanReduce(srcs[i], dsts[i],
						 dsts[i].redop_id,
						 dsts[i].red_fold);
	plans.push_back(p);
        if (prof_requests) {
          tp.add_reduc_entry(srcs[i], dsts[i], td);
        }
	continue;
      }

      // per-field copy
      OffsetsAndSize oas;
      oas.src_field_id = srcs[i].field_id;
      oas.dst_field_id = dsts[i].field_id;
      oas.src_subfield_offset = srcs[i].subfield_offset;
      oas.dst_subfield_offset = dsts[i].subfield_offset;
      oas.size = srcs[i].size;
      oas.serdez_id = srcs[i].serdez_id;
      OASByInst *oas_by_inst = new OASByInst;
      (*oas_by_inst)[InstPair(srcs[i].inst, dsts[i].inst)].push_back(oas);
      if(srcs[i].serdez_id == 0) {
	std::set<int> fields_to_merge;
	find_mergeable_fields(i, srcs, dsts, merged, fields_to_merge);
	for(std::set<int>::const_iterator it = fields_to_merge.begin();
	    it != fields_to_merge.end();
	    ++it) {
	  //log_dma.print() << "merging field " << *it;
	  OffsetsAndSize oas;
	  oas.src_field_id = srcs[*it].field_id;
	  oas.dst_field_id = dsts[*it].field_id;
	  oas.src_subfield_offset = srcs[*it].subfield_offset;
	  oas.dst_subfield_offset = dsts[*it].subfield_offset;
	  oas.size = srcs[*it].size;
	  oas.serdez_id = srcs[*it].serdez_id;
	  (*oas_by_inst)[InstPair(srcs[i].inst, dsts[i].inst)].push_back(oas);
	  merged[*it] = true;
	}
      }
      TransferPlanCopy *p = new TransferPlanCopy(oas_by_inst,
					 src_indirect, dst_indirect);
      plans.push_back(p);
      if (prof_requests) {
        bool is_dst_indirect = dst_indirect == 0 ? false: true;
        bool is_src_indirect = src_indirect == 0 ? false: true;
        tp.add_copy_entry(oas_by_inst, td, is_src_indirect,
                          is_dst_indirect);
      }
    }
    // hack to eliminate duplicate profiling responses
    //assert(requests.empty() || (plans.size() == 1));
    ProfilingRequestSet empty_prs;
    const ProfilingRequestSet *prsptr = &empty_prs;
    std::set<Event> finish_events;
    for(std::vector<TransferPlan *>::iterator it = plans.begin();
	it != plans.end();
	++it) {
      Event e = (*it)->create_plan(td, *prsptr, wait_on, 0 /*priority*/);
      finish_events.insert(e);
    }
    // record profiling state before copies ops begin
    if (prof_requests)
      {
        tp.execute(Event::merge_events(finish_events), wait_on);
        finish_events.insert(prof_ev);
      }

    for(std::vector<TransferPlan *>::iterator it = plans.begin();
        it != plans.end();
        ++it) {
      (*it)->execute_plan();
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
