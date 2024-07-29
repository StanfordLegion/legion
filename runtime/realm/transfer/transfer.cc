/* Copyright 2024 Stanford University, NVIDIA Corporation
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
#include "realm/transfer/transfer_utils.h"
#include "realm/transfer/channel.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/ib_memory.h"
#include "realm/inst_layout.h"

#ifdef REALM_ON_WINDOWS
#include <basetsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace Realm {

  extern Logger log_dma;
  extern Logger log_ib_alloc;
  Logger log_xplan("xplan");
  Logger log_xpath("xpath");
  Logger log_xpath_cache("xpath_cache");

  namespace Config {
    // the size of the cache
    size_t path_cache_lru_size = 0;
    size_t ib_size_bytes = 65536;
  };

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

  size_t TransferIterator::get_base_offset(void) const
  {
    // should not be called
    assert(0);
    return 0;
  }

  size_t TransferIterator::get_address_size(void) const
  {
    // should not be called
    assert(0);
    return 0;
  }

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
    virtual size_t step_custom(size_t max_bytes, AddressInfoCustom& info,
                               bool tentative = false);

    virtual void confirm_step(void);
    virtual void cancel_step(void);

    virtual size_t get_base_offset(void) const;

    virtual bool get_addresses(AddressList &addrlist,
                               const InstanceLayoutPieceBase *&nonaffine);

  protected:
    virtual bool get_next_rect(Rect<N, T> &r, FieldID &fid, size_t &offset,
                               size_t &fsize) = 0;

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
  static size_t get_layout_piece(const InstanceLayout<N, T> *inst_layout,
                                 const InstanceLayoutPiece<N, T> *&layout_piece,
                                 size_t field_id, size_t field_size, size_t field_offset,
                                 int piece_idx)
  {
    size_t field_rel_offset;
    {
      std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it =
          inst_layout->fields.find(field_id);
      assert(it != inst_layout->fields.end());
      assert((field_offset + field_size) <= size_t(it->second.size_in_bytes));
      const InstancePieceList<N, T> &piece_list =
          inst_layout->piece_lists[it->second.list_idx];
      assert(piece_idx >= 0);
      assert(piece_list.pieces.size() > static_cast<size_t>(piece_idx));
      layout_piece = piece_list.pieces[piece_idx];
      // TODO(apryakhin@): Consider adding point lookup.
      // layout_piece = piece_list.find_piece(cur_point) and testing
      // for non-affine layouts.
      if(REALM_UNLIKELY(layout_piece == 0)) {
        abort();
      }
      assert(layout_piece->layout_type == PieceLayoutTypes::AffineLayoutType);
      field_rel_offset = it->second.rel_offset + field_offset;
    }
    return field_rel_offset;
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

  template <int N, typename T>
  size_t TransferIteratorBase<N,T>::step_custom(size_t max_bytes,
                                                AddressInfoCustom& info,
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
	     "no support for accessing partial fields with step_custom");
      const InstancePieceList<N,T>& piece_list = inst_layout->piece_lists[it->second.list_idx];
      layout_piece = piece_list.find_piece(cur_point);
      assert(layout_piece != 0);
    }

    // less than one element?  give up immediately
    if(max_bytes < cur_field_size)
      return 0;

    // figure out the subrectangle we'd ideally like to do, staying within
    //  the max bytes limit
    Rect<N,T> target_subrect;
    target_subrect.lo = cur_point;
    bool grow = true;
    size_t cur_bytes = cur_field_size;
    // follow the agreed-upon dimension ordering
    for(int di = 0; di < N; di++) {
      int d = dim_order[di];

      if(grow) {
        size_t len = cur_rect.hi[d] - cur_point[d] + 1;
        size_t piece_limit = layout_piece->bounds.hi[d] - cur_point[d] + 1;
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

    // convert the target rectangle for the dimension-agnostic interface
    // also make the rectangle _relative_ to the bounds of the piece
    int64_t target_lo[N], target_hi[N];
    for(int i = 0; i < N; i++) {
      target_lo[i] = target_subrect.lo[i] - layout_piece->bounds.lo[i];
      target_hi[i] = target_subrect.hi[i] - layout_piece->bounds.lo[i];
    }

    // offer the rectangle - it can be reduced by pruning dimensions
    int ndims = info.set_rect(inst_impl, layout_piece,
                              cur_field_size, cur_field_offset,
                              N, target_lo, target_hi, dim_order);

    // if pruning did occur, update target_subrect and cur_bytes to match
    if(ndims < N) {
      for(int i = ndims; i < N; i++)
        target_subrect.hi[dim_order[i]] = target_subrect.lo[dim_order[i]];
      cur_bytes = cur_field_size;
      for(int i = 0; i < ndims; i++)
        cur_bytes *= (target_subrect.hi[dim_order[i]] - target_subrect.lo[dim_order[i]] + 1);
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
  uintptr_t TransferIteratorBase<N, T>::get_base_offset(void) const
  {
    return inst_impl->metadata.inst_offset;
  }

  template <int N, typename T>
  bool TransferIteratorBase<N,T>::get_addresses(AddressList &addrlist,
                                                const InstanceLayoutPieceBase *&nonaffine)
  {
#ifdef DEBUG_REALM
    assert(!tentative_valid);
#endif

    nonaffine = 0;

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
        log_dma.debug() << "Find piece found for " << cur_point
                        << " in instance " << inst_impl->me
                        << " (list: " << piece_list << ")";
        if (REALM_UNLIKELY(layout_piece == 0)) {
          log_dma.fatal() << "no piece found for " << cur_point;
          abort();
        }
        if(layout_piece->layout_type != PieceLayoutTypes::AffineLayoutType) {
          // can't handle this piece here - let the caller know what the
          //  non-affine piece is and maybe it can handle it
          nonaffine = layout_piece;
          return true;
        }
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
      log_dma.debug() << "step: cur_rect=" << cur_rect
                      << " layout_bounds=" << layout_piece->bounds
                      << " target_subrect=" << target_subrect
                      << " next_point=" << cur_point
                      << " (have_rect=" << have_rect << ")";
#ifdef DEBUG_REALM
      assert(layout_piece->bounds.contains(target_subrect));
#endif

      // TODO: remove now-redundant condition here
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
          log_dma.debug() << "Add addr data dim=" << cur_dim
                          << " total_count=" << total_count
                          << " stride=" << stride;
          total_bytes *= total_count;
	  cur_dim++;
	}

	// now that we know the compacted dimension, we can finish the address
	//  record
	addr_data[0] = (bytes << 4) + cur_dim;
	addrlist.commit_nd_entry(cur_dim, total_bytes);
        log_dma.debug() << "Finalize addr data dim=" << cur_dim
                        << " total_bytes" << total_bytes;
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
  // class WrappingTransferIteratorIndirect<N,T>
  //

  template <int N, typename T>
  class WrappingTransferIteratorIndirect : public TransferIteratorBase<N, T> {
  protected:
    WrappingTransferIteratorIndirect(void); // used by deserializer
  public:
    WrappingTransferIteratorIndirect(RegionInstance inst,
                                     const std::vector<FieldID> &_fields,
                                     const std::vector<size_t> &_fld_offsets,
                                     const std::vector<size_t> &_fld_sizes);

    template <typename S>
    static TransferIterator *deserialize_new(S &deserializer);

    virtual ~WrappingTransferIteratorIndirect(void);

    virtual Event request_metadata(void);

    // specify the xd port used for indirect address flow control, if any
    virtual void set_indirect_input_port(XferDes *xd, int port_idx,
                                         TransferIterator *inner_iter);

    virtual bool get_addresses(AddressList &addrlist,
                               const InstanceLayoutPieceBase *&nonaffine);

    virtual size_t get_base_offset(void) const;

    virtual void reset(void);

    virtual size_t get_address_size(void) const;
    virtual size_t step(size_t max_bytes, TransferIterator::AddressInfo &info,
                        unsigned flags, bool tentative = false);

    static Serialization::PolymorphicSerdezSubclass<
        TransferIterator, WrappingTransferIteratorIndirect<N, T>>
        serdez_subclass;

    template <typename S>
    bool serialize(S &serializer) const;

  protected:
    virtual bool get_next_rect(Rect<N, T> &r, FieldID &fid, size_t &offset,
                               size_t &fsize);

    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;
    XferDes *indirect_xd;
    size_t piece_idx;
    int indirect_port_idx;
    TransferIterator *addrs_in;
    size_t addrs_in_offset = 0;
    size_t point_pos = 0;
    size_t num_points = 0;
    static const size_t MAX_POINTS = 4194304;
  };

  template <int N, typename T>
  WrappingTransferIteratorIndirect<N, T>::WrappingTransferIteratorIndirect(void)
  {}

  template <int N, typename T>
  WrappingTransferIteratorIndirect<N, T>::WrappingTransferIteratorIndirect(
      RegionInstance inst, const std::vector<FieldID> &_fields,
      const std::vector<size_t> &_fld_offsets, const std::vector<size_t> &_fld_sizes)
    : TransferIteratorBase<N, T>(inst, 0)
    , fields(_fields)
    , fld_offsets(_fld_offsets)
    , fld_sizes(_fld_sizes)
    , piece_idx(0)
  {}

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<
      TransferIterator, WrappingTransferIteratorIndirect<N, T>>
      WrappingTransferIteratorIndirect<N, T>::serdez_subclass;

  template <int N, typename T>
  template <typename S>
  bool WrappingTransferIteratorIndirect<N, T>::serialize(S &serializer) const
  {
    return ((serializer << this->inst_impl->me) && (serializer << fields) &&
            (serializer << fld_offsets) && (serializer << fld_sizes));
  }

  template <int N, typename T>
  template <typename S>
  /*static*/ TransferIterator *
  WrappingTransferIteratorIndirect<N, T>::deserialize_new(S &deserializer)
  {
    RegionInstance inst;
    std::vector<FieldID> fields;
    std::vector<size_t> fld_offsets, fld_sizes;

    if(!((deserializer >> inst) && (deserializer >> fields) &&
         (deserializer >> fld_offsets) && (deserializer >> fld_sizes)))
      return 0;

    return new WrappingTransferIteratorIndirect<N, T>(inst, fields, fld_offsets,
                                                      fld_sizes);
  }

  template <int N, typename T>
  WrappingTransferIteratorIndirect<N, T>::~WrappingTransferIteratorIndirect(void)
  {}

  template <int N, typename T>
  Event WrappingTransferIteratorIndirect<N, T>::request_metadata(void)
  {
    return TransferIteratorBase<N, T>::request_metadata();
  }

  template <int N, typename T>
  void WrappingTransferIteratorIndirect<N, T>::set_indirect_input_port(
      XferDes *xd, int port_idx, TransferIterator *inner_iter)
  {
    indirect_xd = xd;
    indirect_port_idx = port_idx;
    addrs_in = inner_iter;
  }

  template <int N, typename T>
  void WrappingTransferIteratorIndirect<N, T>::reset(void)
  {
    TransferIteratorBase<N, T>::reset();
    piece_idx = 0;
  }

  template <int N, typename T>
  size_t WrappingTransferIteratorIndirect<N, T>::get_address_size(void) const
  {
    return sizeof(Point<N, T>);
  }

  template <int N, typename T>
  uintptr_t WrappingTransferIteratorIndirect<N, T>::get_base_offset(void) const
  {
    return addrs_in_offset;
  }

  template <int N, typename T>
  size_t WrappingTransferIteratorIndirect<N, T>::step(size_t max_bytes,
                                                      TransferIterator::AddressInfo &info,
                                                      unsigned flags,
                                                      bool tentative /*= false*/)
  {
    FieldID cur_field_id = fields[0];
    size_t cur_field_offset = fld_offsets[0];
    size_t cur_field_size = fld_sizes[0];

    if(this->inst_layout == 0) {
      assert(this->inst_impl->metadata.is_valid());
      this->inst_layout =
          checked_cast<const InstanceLayout<N, T> *>(this->inst_impl->metadata.layout);
    }

    assert(this->inst_layout);
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it =
        this->inst_layout->fields.find(cur_field_id);
    assert(it != this->inst_layout->fields.end());
    size_t pieces = this->inst_layout->piece_lists[it->second.list_idx].pieces.size();

    if(piece_idx < pieces) {
      const InstanceLayoutPiece<N, T> *layout_piece;
      size_t field_rel_offset =
          get_layout_piece(this->inst_layout, layout_piece, cur_field_id, cur_field_size,
                           cur_field_offset, piece_idx);

      if(layout_piece->layout_type == PieceLayoutTypes::AffineLayoutType) {
        const AffineLayoutPiece<N, T> *affine =
            static_cast<const AffineLayoutPiece<N, T> *>(layout_piece);

        info.base_offset = (this->inst_impl->metadata.inst_offset + affine->offset +
                            affine->strides.dot(affine->bounds.lo) + field_rel_offset);

        size_t cur_dim = 0;
        info.bytes_per_chunk = affine->strides[cur_dim++];
        if(N > cur_dim) {
          info.num_lines = affine->bounds.hi[cur_dim] - affine->bounds.lo[cur_dim] + 1;
          info.line_stride = affine->strides[cur_dim];
        }
        cur_dim++;
        if(N > cur_dim) {
          info.num_planes = affine->bounds.hi[cur_dim] - affine->bounds.lo[cur_dim] + 1;
          info.plane_stride = affine->strides[cur_dim];
        }
        piece_idx++;
      }
    }
    piece_idx %= pieces;
    return 0;
  }

  template <int N, typename T>
  bool WrappingTransferIteratorIndirect<N, T>::get_addresses(
      AddressList &addrlist, const InstanceLayoutPieceBase *&nonaffine)
  {
    nonaffine = 0;

    while(!this->done()) {
      if(!this->have_rect) {
        return false;
      }

      size_t *addr_data = addrlist.begin_nd_entry(1);
      if(!addr_data) {
        return true;
      }

      int cur_dim = 1;
      size_t total_bytes = this->cur_rect.volume() * this->cur_field_size;
      this->have_rect = false;
      addr_data[0] = ((total_bytes) << 4) + cur_dim;
      addrlist.commit_nd_entry(cur_dim, total_bytes);
      log_dma.debug() << "Finalize gather/scatter addr data dim=" << cur_dim
                      << " total_bytes=" << total_bytes;
      break;
    }
    return true;
  }

  template <int N, typename T>
  bool WrappingTransferIteratorIndirect<N, T>::get_next_rect(Rect<N, T> &r, FieldID &fid,
                                                             size_t &offset,
                                                             size_t &fsize)
  {
    assert(fields.size() == 1);
    fid = fields[0];
    offset = fld_offsets[0];
    fsize = fld_sizes[0];

    r.lo = Point<N, T>::ZEROES();
    r.hi = Point<N, T>::ZEROES();

    addrs_in->done();

    XferDes::XferPort &iip = indirect_xd->input_ports[indirect_port_idx];

    bool nonempty = false;
    while(true) {

      if(point_pos * sizeof(Point<N, T>) > iip.local_bytes_total) {
        return nonempty;
      }

      if(point_pos >= num_points) {
        if(addrs_in->done()) {
          this->is_done = true;
          return nonempty;
        }

        TransferIterator::AddressInfo a_info;
        size_t addr_max_bytes = sizeof(Point<N, T>) * MAX_POINTS;
        if(indirect_xd != 0) {
          if(iip.peer_guid != XferDes::XFERDES_NO_GUID) {
            addr_max_bytes =
                iip.seq_remote.span_exists(iip.local_bytes_total, addr_max_bytes);
            size_t rem = addr_max_bytes % sizeof(Point<N, T>);
            if(rem > 0) {
              addr_max_bytes -= rem;
            }
            if(addr_max_bytes == 0) {
              if(iip.remote_bytes_total.load() == iip.local_bytes_total)
                this->is_done = true;
              return nonempty;
            }
          }
        }

        size_t amt = addrs_in->step(addr_max_bytes, a_info, 0, false);
        if(amt == 0) {
          return nonempty;
        }

        // TODO: handle partial point reads
        addrs_in_offset = a_info.base_offset;
        num_points = amt / sizeof(Point<N, T>);
      }
      r.lo[0] = point_pos;
      r.hi[0] = point_pos + num_points - 1;
      point_pos += num_points;
      nonempty = true;
    }
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
  Event TransferIteratorIndirect<N, T>::request_metadata(void)
  {
    return Event::merge_events(
        {addrs_in->request_metadata(), TransferIteratorBase<N, T>::request_metadata()});
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

    addrs_in->done();

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
            // round down to multiple of sizeof(Point<N,T>)
            size_t rem = addr_max_bytes % sizeof(Point<N,T>);
            if(rem > 0)
              addr_max_bytes -= rem;
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

        if(amt == 0) {
          return nonempty;
        }
        memcpy(points,
               reinterpret_cast<const void *>(addrs_mem_base + a_info.base_offset), amt);
        // handle reads of partial points
        while((amt % sizeof(Point<N,T>)) != 0) {
          // get some more - should never be empty
          size_t todo = addrs_in->step(addr_max_bytes - amt, a_info, 0,
                                       false /*!tentative*/);
          assert(todo > 0);

          memcpy(reinterpret_cast<char *>(points) + amt,
                 reinterpret_cast<const void *>(addrs_mem_base +
                                                a_info.base_offset),
                 todo);
          amt += todo;
        }

	point_pos = 0;
	num_points = amt / sizeof(Point<N,T>);
        log_dma.debug() << "indirect-iterator read num_points=" << num_points;
	assert(amt == (num_points * sizeof(Point<N,T>)));

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
                log_dma.debug() << "indirect-iterator merge fails next_rect=" << r;
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
              log_dma.debug() << "indirect-iterator next_rect=" << r;
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
            // round down to multiple of sizeof(Rect<N,T>)
            size_t rem = addr_max_bytes % sizeof(Rect<N,T>);
            if(rem > 0)
              addr_max_bytes -= rem;
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
	memcpy(rects,
	       reinterpret_cast<const void *>(addrs_mem_base +
					      a_info.base_offset),
	       amt);
        // handle reads of partial rects
        while((amt % sizeof(Rect<N,T>)) != 0) {
          // get some more - should never be empty
          size_t todo = addrs_in->step(addr_max_bytes - amt, a_info, 0,
                                       false /*!tentative*/);
          assert(todo > 0);

          memcpy(reinterpret_cast<char *>(rects) + amt,
                 reinterpret_cast<const void *>(addrs_mem_base +
                                                a_info.base_offset),
                 todo);
          amt += todo;
        }

	rect_pos = 0;
	num_rects = amt / sizeof(Rect<N,T>);
	assert(amt == (num_rects * sizeof(Rect<N,T>)));

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
            if(rects[rect_pos].lo[0] == (r.hi[0] + 1)) {
              merge_dim = 0;
            }
          } else {
            const Rect<N, T> &r2 = rects[rect_pos];
            int dims_match = 0;
            while(dims_match < (N - 1))
              if((r.lo[dims_match] == r2.lo[dims_match]) &&
                 (r.hi[dims_match] == r2.hi[dims_match]))
                dims_match++;
              else
                break;
            if((r2.lo[dims_match] == (r.hi[dims_match] + 1))) {
              merge_dim = dims_match; // unless checks below fail
              // rest of dims must be degenerate and match
              for(int i = dims_match + 1; i < N; i++)
                if((r.lo[i] != r.hi[i]) || (r2.lo[i] != r.lo[i]) ||
                   (r2.hi[i] != r.hi[i])) {
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

    virtual void count_fragments(RegionInstance inst,
                                 const std::vector<int>& dim_order,
                                 const std::vector<FieldID>& fields,
                                 const std::vector<size_t>& fld_sizes,
                                 std::vector<size_t>& fragments) const;

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
  static void add_fragments_for_rect(const Rect<N,T>& rect,
                                     size_t field_size,
                                     size_t field_count,
                                     const Point<N,size_t>& strides,
                                     const std::vector<int>& dim_order,
                                     std::vector<size_t>& fragments)
  {
    int collapsed[N+1];
    int breaks = 0;
    collapsed[0] = 1;
    size_t exp_stride = field_size;
    for(int di = 0; di < N; di++) {
      int d = dim_order[di];
      // skip trivial dimensions
      if(rect.lo[d] == rect.hi[d]) continue;

      size_t extent = size_t(rect.hi[d]) - size_t(rect.lo[d]) + 1;

      if(exp_stride == strides[d]) {
        // stride match?  collapse more
        collapsed[breaks] *= extent;
        exp_stride *= extent;
      } else {
        // stride mismatch - break here
        breaks++;
        collapsed[breaks] = extent;
        exp_stride = strides[d] * extent;
      }
    }

    // now work back down from the top dimension and increase fragment
    //  count for each break
    size_t frags = field_count;
    for(int d = N; d >= 0; d--) {
      if(d <= breaks)
        frags *= collapsed[d];
      fragments[d] += frags;
    }
  }

  template <int N, typename T>
  void TransferDomainIndexSpace<N,T>::count_fragments(RegionInstance inst,
                                                      const std::vector<int>& dim_order,
                                                      const std::vector<FieldID>& fields,
                                                      const std::vector<size_t>& fld_sizes,
                                                      std::vector<size_t>& fragments) const
  {
    RegionInstanceImpl *inst_impl = get_runtime()->get_instance_impl(inst);
#ifdef DEBUG_REALM
    assert(inst_impl->metadata.is_valid());
#endif
    const InstanceLayout<N,T> *inst_layout = checked_cast<const InstanceLayout<N,T> *>(inst_impl->metadata.layout);

    fragments.assign(N+2, 0);

    for(size_t i = 0; i < fields.size(); i++) {
      FieldID fid = fields[i];
      size_t field_size = fld_sizes[i];

      const InstancePieceList<N,T> *ipl;
      {
        std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(fid);
        assert(it != inst_layout->fields.end());
        ipl = &inst_layout->piece_lists[it->second.list_idx];
      }

      IndexSpaceIterator<N,T> isi(is);

      // get the piece for the first index
      const InstanceLayoutPiece<N,T> *layout_piece = ipl->find_piece(isi.rect.lo);
      assert(layout_piece != 0);

      if(layout_piece->bounds.contains(is)) {
        // easy case: one piece covers our entire domain and the iteration order
        //  doesn't impact the fragment count
        if(layout_piece->layout_type == PieceLayoutTypes::AffineLayoutType) {
          const AffineLayoutPiece<N,T> *affine = static_cast<const AffineLayoutPiece<N,T> *>(layout_piece);
          do {
            add_fragments_for_rect(isi.rect, field_size, 1 /*field count*/,
                                   affine->strides, dim_order, fragments);
            isi.step();
          } while(isi.valid);
        } else {
          // not affine - add one fragment for each rectangle
          size_t num_rects;
          if(is.dense()) {
            num_rects = 1;
          } else {
            SparsityMapPublicImpl<N,T> *s_impl = is.sparsity.impl();
            num_rects = s_impl->get_entries().size();
          }

          for(int i = 0; i < (N + 2); i++)
            fragments[i] += num_rects;
        }
      } else {
        size_t non_affine_rects = 0;
        do {
          Point<N,T> next_start = isi.rect.lo;
          while(true) {
            // look up new piece if needed
            if(!layout_piece->bounds.contains(next_start)) {
              layout_piece = ipl->find_piece(next_start);
              assert(layout_piece != 0);
            }

            Rect<N,T> subrect;
            bool last = next_subrect(isi.rect, next_start, layout_piece->bounds,
                                     dim_order.data(), subrect, next_start);
            if(layout_piece->layout_type == PieceLayoutTypes::AffineLayoutType) {
              const AffineLayoutPiece<N,T> *affine = static_cast<const AffineLayoutPiece<N,T> *>(layout_piece);
              add_fragments_for_rect(isi.rect, field_size, 1 /*field count*/,
                                     affine->strides, dim_order, fragments);
            } else
              non_affine_rects++;

            if(last) break;
          }

          isi.step();
        } while(isi.valid);

        if(non_affine_rects > 0)
          for(int i = 0; i < (N + 2); i++)
            fragments[i] += non_affine_rects;
      }
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
      size_t element_size = 0;
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
                               const std::vector<IndexSpace<N, T>> &_spaces,
                               Channel *_channel);

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
				 const void *fill_data, size_t fill_size,
                                 size_t fill_total);

    virtual Channel *get_channel() const { return channel; }

    static ActiveMessageHandlerReg<AddressSplitXferDesCreateMessage<N,T> > areg;

  protected:
    size_t bytes_per_element;
    std::vector<IndexSpace<N,T> > spaces;
    Channel *channel;
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
    ControlPort::Encoder ctrl_encoder;
  };

  template <int N, typename T>
  AddressSplitXferDesFactory<N, T>::AddressSplitXferDesFactory(
      size_t _bytes_per_element, const std::vector<IndexSpace<N, T>> &_spaces,
      Channel *_channel)
    : bytes_per_element(_bytes_per_element)
    , spaces(_spaces)
    , channel(_channel)
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
							const void *fill_data,
                                                        size_t fill_size,
                                                        size_t fill_total)
  {
    assert(redop_info.id == 0);
    assert(fill_size == 0);
    if(target_node == Network::my_node_id) {
      // local creation
      //assert(!inst.exists());
      assert(channel);
      XferDes *xd = new AddressSplitXferDes<N, T>(dma_op, channel, launch_node, guid,
                                                  inputs_info, outputs_info, priority,
                                                  bytes_per_element, spaces);

      channel->enqueue_ready_xd(xd);
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
  {
    ctrl_encoder.set_port_count(spaces.size());
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
            // round down to multiple of sizeof(Point<N,T>)
            size_t rem = max_bytes % sizeof(Point<N,T>);
            if(rem > 0)
              max_bytes -= rem;
	    if(max_bytes < sizeof(Point<N,T>)) {
              // check to see if this is the end of the input
              if(input_ports[0].local_bytes_total ==
                 input_ports[0].remote_bytes_total.load_acquire())
                input_done = true;
	      break;
            }
	  }
	  size_t bytes = input_ports[0].iter->step(max_bytes, p_info,
						   0, false /*!tentative*/);
	  if(bytes == 0) break;
	  const void *srcptr = input_ports[0].mem->get_direct_ptr(p_info.base_offset,
								  bytes);
	  assert(srcptr != 0);
	  memcpy(points, srcptr, bytes);
          // handle reads of partial points
          while((bytes % sizeof(Point<N,T>)) != 0) {
            // get some more - should never be empty
            size_t todo = input_ports[0].iter->step(max_bytes - bytes, p_info,
                                                    0, false /*!tentative*/);
            assert(todo > 0);
            const void *srcptr = input_ports[0].mem->get_direct_ptr(p_info.base_offset,
                                                                    todo);
            assert(srcptr != 0);
            memcpy(reinterpret_cast<char *>(points) + bytes, srcptr, todo);
            bytes += todo;
          }

	  point_count = bytes / sizeof(Point<N,T>);
	  assert(bytes == (point_count * sizeof(Point<N,T>)));
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

	// if it matched a space, we have to emit the point to that space's
	//  output address stream before we can accept the point
	if(output_space_id != -1) {
	  XferPort &op = output_ports[output_space_id];
	  if(op.seq_remote.span_exists(op.local_bytes_total + output_bytes,
				       sizeof(Point<N,T>)) < sizeof(Point<N,T>))
	    break;
	  TransferIterator::AddressInfo o_info;
          size_t partial = 0;
          while(partial < sizeof(Point<N,T>)) {
            size_t bytes = op.iter->step(sizeof(Point<N,T>) - partial, o_info,
                                         0, false /*!tentative*/);
            void *dstptr = op.mem->get_direct_ptr(o_info.base_offset, bytes);
            assert(dstptr != 0);
            memcpy(dstptr,
                   reinterpret_cast<const char *>(&points[point_index])+partial,
                   bytes);
            partial += bytes;
          }
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
	output_ports[output_space_id].local_bytes_cons.fetch_add(output_bytes);
	did_work = true;
      }

      // now try to write the control information
      if((output_count > 0) || input_done) {
	XferPort &cp = output_ports[spaces.size()];

        // may take us a few tries to send the control word
        bool ctrl_sent = false;
	size_t old_lbt = cp.local_bytes_total;
        do {
          if(cp.seq_remote.span_exists(cp.local_bytes_total,
                                       sizeof(unsigned)) < sizeof(unsigned))
            break;  // no room to write control work

          TransferIterator::AddressInfo c_info;
          size_t bytes = cp.iter->step(sizeof(unsigned), c_info,
                                       0, false /*!tentative*/);
          assert(bytes == sizeof(unsigned));
          void *dstptr = cp.mem->get_direct_ptr(c_info.base_offset,
                                                sizeof(unsigned));
          assert(dstptr != 0);

          unsigned cword;
          ctrl_sent = ctrl_encoder.encode(cword,
                                          output_count * element_size,
                                          output_space_id,
                                          input_done);
          memcpy(dstptr, &cword, sizeof(unsigned));

          cp.local_bytes_total += sizeof(unsigned);
          cp.local_bytes_cons.fetch_add(sizeof(unsigned));
        } while(!ctrl_sent);

	if(input_done && ctrl_sent) {
          begin_completion();

	  // mark all address streams as done (dummy write update)
	  for(size_t i = 0; i < spaces.size(); i++)
	    wseqcache.add_span(i, output_ports[i].local_bytes_total, 0);
	}

        // push out the partial write even if we're not done
        if(cp.local_bytes_total > old_lbt) {
          wseqcache.add_span(spaces.size(), old_lbt,
                             cp.local_bytes_total - old_lbt);
          did_work = true;
        }

        // but only actually clear the output_count if we sent the whole
        //  control packet
        if(!ctrl_sent)
          break;

	output_space_id = -1;
	output_count = 0;
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
  // transfer path search logic
  //

// #define PATH_CACHE_EARLY_INIT

  // a map to cache the path from src memory to dst memory. 
  //  the key is the pair of src and dst memory id. 
  //  the value is a LRU. 
  //  The LRU is indexed by LRUKey, the value is MemPathInfo
  static std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *> path_cache;

  // a RWLock to control the access to the path_cache
  static RWLock path_cache_rwlock;
  static bool path_cache_inited = false;

  // counters for calculating cache misses and hits
  atomic<unsigned int> nb_cache_miss(0);
  atomic<unsigned int> nb_cache_hit(0);

  // The path cache initialization function, which is called by 
  //   RuntimeImpl::configure_from_command_line
  void init_path_cache(void)
  {
    assert(path_cache_inited == false);
#ifdef PATH_CACHE_EARLY_INIT
    std::vector<Memory> memories;
    Machine machine = Machine::get_machine();
    for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it; ++it) {
      Memory m = *it;
      memories.push_back(m);
    }
    for(std::vector<Memory>::const_iterator src_it = memories.begin(); src_it != memories.end(); ++src_it) {
      for(std::vector<Memory>::const_iterator dst_it = memories.begin(); dst_it != memories.end(); ++dst_it) {
        std::pair<realm_id_t, realm_id_t> key((*src_it).id, (*dst_it).id);
        if (path_cache.find(key) != path_cache.end()) {
          assert(0);
        }
        PathLRU *lru = new PathLRU(Config::path_cache_lru_size);
        path_cache[key] = lru;
      }
    }
#endif
    path_cache_inited = true;
    nb_cache_miss.store_release(0);
    nb_cache_hit.store_release(0);
  }

  // The path cache finalize function, which is called by
  //   RuntimeImpl::wait_for_shutdown
  void finalize_path_cache(void)
  {
    assert(path_cache_inited == true);
    log_xpath_cache.info() << "Cache Miss: " << nb_cache_miss.load_fenced() << " Cache Hit: " << nb_cache_hit.load_fenced();
    std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *>::iterator it;
    for (it = path_cache.begin(); it != path_cache.end(); it++) {
      delete it->second;
    }
    path_cache.clear();
    path_cache_inited = false;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class PathLRU::LRUKey

  PathLRU::LRUKey::LRUKey(const CustomSerdezID serdez_id, const ReductionOpID redop_id, 
                          const size_t total_bytes, 
                          const std::vector<size_t> src_frags, 
                          const std::vector<size_t> dst_frags)
  : timestamp(0), serdez_id(serdez_id), redop_id(redop_id), total_bytes(total_bytes), 
    src_frags(src_frags), dst_frags(dst_frags)
  {
  }

  bool PathLRU::LRUKey::operator==(const LRUKey &rhs) const 
  {
    if ( (serdez_id == rhs.serdez_id) && (redop_id == rhs.redop_id) && 
         (total_bytes == rhs.total_bytes) && 
         (src_frags == rhs.src_frags) && (dst_frags == rhs.dst_frags)) {
      return true;
    } else {
      return false;
    }
  }

  std::ostream& operator<<(std::ostream& out, const PathLRU::LRUKey& lru_key)
  {
    out << "LRUKey:{";
    out << " serdez_id: " << lru_key.serdez_id;
    out << " redop_id: " << lru_key.redop_id;
    out << " size: " << lru_key.total_bytes;
    out << " src_frags:(";
    for (size_t i = 0; i < lru_key.src_frags.size(); i++) {
      out << lru_key.src_frags[i] << ",";
    }
    out << ")";
    out << " dst_frags:(";
    for (size_t i = 0; i < lru_key.dst_frags.size(); i++) {
      out << lru_key.dst_frags[i] << ",";
    }
    out << ") }";
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const MemPathInfo& info)
  {
    out << "MemPathInfo:{ ";
    for (size_t i = 0; i < info.path.size(); i++) {
      out << "Mem:" << info.path[i] << " kind:" << info.path[i].kind() << " ";
    }
    for (size_t i = 0; i < info.xd_channels.size(); i++) {
      out << "Channel:" << info.xd_channels[i]->kind << " ";
    }
    out << "}";
    return out;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class PathLRU

  PathLRU::PathLRU(size_t size)
  : max_size(size), timestamp(0)
  {
  }

  void PathLRU::miss(LRUKey &key, const MemPathInfo &path)
  {
    unsigned long current_timestamp = timestamp.fetch_add_acqrel(1);
    assert(current_timestamp <= ULONG_MAX);
    // get the current timestamp and assign it to the lru key
    key.timestamp.store_release(current_timestamp);
    std::pair<LRUKey,MemPathInfo> item = std::make_pair(key, path);
    if (item_list.size() < max_size) {
      // if the LRU not full, we just insert the item
      item_list.push_back(item);
    } else {
      // if the LRU is full, we need to iterate the LRU to find 
      //   the item that has the earliest timestamp, and replace 
      //   it with the new item
      assert(item_list.size() == max_size);
      size_t earliest_idx = 0;
      unsigned long earliest_timestamp = item_list[earliest_idx].first.timestamp.load(); 
      for (size_t i = 0; i < item_list.size(); i++) {
        unsigned long t = item_list[i].first.timestamp.load();
        if (t < earliest_timestamp) {
        earliest_timestamp = t;
        earliest_idx = i;
        }
      }
      // log_xpath_cache.debug() << "Cache full, remove LRUKey: " << item_list[earliest_idx].first;
      item_list[earliest_idx] = item;
    }
  }

  void PathLRU::hit(PathLRU::PathLRUIterator it)
  {
    unsigned long current_timestamp = timestamp.fetch_add_acqrel(1);
    assert(current_timestamp <= ULONG_MAX);
    // update the timestamp of the lru key with the newest one.   
    //   When 2 threads calls the hit, even though we can not guarantee that
    //   it->first.timestamp gets the latest timestamp, but it is fine
    //   because it->first.timestamp > timestamps of other items in the LRU,
    //   which guarantees the correctness of LRU.
    it->first.timestamp.store_release(current_timestamp);
  }

  PathLRU::PathLRUIterator PathLRU::find(const PathLRU::LRUKey &key)
  {
    PathLRUIterator it;
    for (it = item_list.begin(); it != item_list.end(); it++) {
      if (it->first == key) break;
    }
    return it;
  }

  PathLRU::PathLRUIterator PathLRU::end(void)
  {
    return item_list.end();
  }

  bool find_best_channel_for_memories(
      const Node *nodes_info, ChannelCopyInfo channel_copy_info,
      CustomSerdezID src_serdez_id, CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
      size_t total_bytes, const std::vector<size_t> *src_frags,
      const std::vector<size_t> *dst_frags, uint64_t &best_cost, Channel *&best_channel,
      XferDesKind &best_kind)
  {
    // consider dma channels available on either source or dest node
    NodeID src_node = ID(channel_copy_info.src_mem).memory_owner_node();
    NodeID dst_node = ID(channel_copy_info.dst_mem).memory_owner_node();

    best_cost = 0;
    best_channel = 0;
    best_kind = XFER_NONE;

    {
      const Node &n = nodes_info[src_node];
      for(std::vector<Channel *>::const_iterator it = n.dma_channels.begin();
          it != n.dma_channels.end(); ++it) {
        XferDesKind kind = XFER_NONE;
        uint64_t cost =
            (*it)->supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                 redop_id, total_bytes, src_frags, dst_frags, &kind);
        if((cost > 0) && ((best_cost == 0) || (cost < best_cost))) {
          best_cost = cost;
          best_channel = *it;
          best_kind = kind;
        }
      }
    }

    if(dst_node != src_node) {
      const Node &n = nodes_info[dst_node];
      for(std::vector<Channel *>::const_iterator it = n.dma_channels.begin();
          it != n.dma_channels.end(); ++it) {
        XferDesKind kind = XFER_NONE;
        uint64_t cost =
            (*it)->supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                 redop_id, total_bytes, src_frags, dst_frags, &kind);
        if((cost > 0) && ((best_cost == 0) || (cost < best_cost))) {
          best_cost = cost;
          best_channel = *it;
          best_kind = kind;
        }
      }
    }

    return (best_cost != 0);
  }

  bool find_fastest_path(const Node *nodes_info, PathCache &path_cache,
                         ChannelCopyInfo channel_copy_info, CustomSerdezID serdez_id,
                         ReductionOpID redop_id, size_t total_bytes,
                         const std::vector<size_t> *src_frags,
                         const std::vector<size_t> *dst_frags, MemPathInfo &info,
                         bool skip_final_memcpy)
  {
    Memory src_mem = channel_copy_info.src_mem;
    Memory dst_mem = channel_copy_info.dst_mem;
    NodeID src_node = ID(src_mem).memory_owner_node();
    NodeID dst_node = ID(dst_mem).memory_owner_node();
    std::vector<size_t> empty_vec;

    log_xpath.info() << "FFP: " << src_mem << "->" << dst_mem << " serdez=" << serdez_id
                     << " redop=" << redop_id << " bytes=" << total_bytes << " frags="
                     << PrettyVector<size_t>(*(src_frags ? src_frags : &empty_vec)) << "/"
                     << PrettyVector<size_t>(*(dst_frags ? dst_frags : &empty_vec));

    if (path_cache_inited) {
      std::pair<realm_id_t, realm_id_t> key(src_mem.id, dst_mem.id);
      PathLRU *lru = nullptr;
#ifdef PATH_CACHE_EARLY_INIT
      std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *>::iterator path_cache_it;
      path_cache_it = path_cache.find(key);
      assert(path_cache_it != path_cache.end());
      lru = path_cache_it->second;
#else
      {
        // first check if the key is existed, if not create a PathLRU for the given key
        RWLock::AutoReaderLock arl(path_cache_rwlock);
        if (path_cache.find(key) == path_cache.end()) {
          // drop reader lock and take writer lock
          arl.release();
          RWLock::AutoWriterLock awl(path_cache_rwlock);
          // double check if the key is existed because it could be created by 
          //   another thread before we get the wrlock. 
          if (path_cache.find(key) == path_cache.end()) {
            path_cache[key] = new PathLRU(Config::path_cache_lru_size);
          }
        }
        lru = path_cache.find(key)->second;
      }
#endif
      assert(lru != nullptr);
      // check if we can find the LRU key inside the LRU. If yes, we call the hit
      {
        PathLRU::LRUKey lru_key(serdez_id, redop_id, total_bytes, *src_frags, *dst_frags);
        RWLock::AutoReaderLock arl(lru->rwlock);
        PathLRU::PathLRUIterator lru_it = lru->find(lru_key);
        if (lru_it != lru->end()) {
          info = lru_it->second;
          lru->hit(lru_it);
          nb_cache_hit.fetch_add(1);
          log_xpath_cache.debug() << "src:" << src_mem << ", dst:" << dst_mem << ", " << info << ", " << lru_key << ", Hit";
          arl.release();
          return true;
        }
      }
    }

    // baseline - is a direct path possible?
    uint64_t best_cost = 0;
    {
      Channel *channel;
      XferDesKind kind;
      if(find_best_channel_for_memories(nodes_info, channel_copy_info, serdez_id,
                                        serdez_id, redop_id, total_bytes, src_frags,
                                        dst_frags, best_cost, channel, kind)) {
        log_xpath.info() << "direct: " << src_mem << "(n:" << src_node << ")->" << dst_mem
                         << " (n:" << dst_node << ") cost=" << best_cost
                         << " channel=" << channel->kind;
        info.path.assign(1, src_mem);
        if(!skip_final_memcpy || (kind != XFER_MEM_CPY)) {
          info.path.push_back(dst_mem);
          info.xd_channels.assign(1, channel);
        } else
          info.xd_channels.clear();
      }
    }

    // multi-hop search (have to do this even if a direct path exists)
    // any intermediate memory on the src or dst node is a candidate
    struct PartialPath {
      Memory ib_mem;
      uint64_t cost;
      std::vector<Memory> path;
      std::vector<Channel *> channels;
    };
    std::vector<PartialPath> partials;
    size_t num_src_ibs, total_ibs;
    {
      const Node &n = nodes_info[src_node];
      num_src_ibs = n.ib_memories.size();
      partials.resize(num_src_ibs);
      for(size_t i = 0; i < n.ib_memories.size(); i++) {
        partials[i].ib_mem = n.ib_memories[i]->me;
        partials[i].cost = 0;
      }
    }
    if(dst_node != src_node) {
      const Node &n = nodes_info[dst_node];
      total_ibs = num_src_ibs + n.ib_memories.size();
      partials.resize(total_ibs);
      for(size_t i = 0; i < n.ib_memories.size(); i++) {
        partials[num_src_ibs + i].ib_mem = n.ib_memories[i]->me;
        partials[num_src_ibs + i].cost = 0;
      }
    } else
      total_ibs = num_src_ibs;

    // see which of the ib memories we can get to from the original srcmem
    std::set<size_t> active_ibs;
    for(size_t i = 0; i < total_ibs; i++) {
      uint64_t cost = 0;
      Channel *channel;
      XferDesKind kind;
      ChannelCopyInfo copy_info = channel_copy_info;
      copy_info.dst_mem = partials[i].ib_mem;
      copy_info.is_direct = false;
      if(channel_copy_info.is_scatter) {
        copy_info.ind_mem = Memory::NO_MEMORY;
      }
      if(find_best_channel_for_memories(
             nodes_info, copy_info, serdez_id, 0 /*no dst serdez*/,
             0 /*no redop on not-last hops*/, total_bytes, src_frags, 0 /*no dst_frags*/,
             cost, channel, kind)) {
        NodeID dst_node = ID(partials[i].ib_mem).memory_owner_node();
        log_xpath.info() << "first: " << src_mem << "(n:" << src_node << ")->"
                         << partials[i].ib_mem << "(n:" << dst_node << ") cost=" << cost
                         << " channel=" << channel->kind;
        // ignore anything that's already worse than the direct path
        if((best_cost == 0) || (cost < best_cost)) {
          active_ibs.insert(i);
          partials[i].cost = cost;
          partials[i].path.resize(2);
          partials[i].path[0] = src_mem;
          partials[i].path[1] = partials[i].ib_mem;
          partials[i].channels.assign(1, channel);
        }
      }
    }

    // look for multi-ib-hop paths (as long as they improve on the shorter
    //  ones)
    while(!active_ibs.empty()) {
      size_t src_idx = *(active_ibs.begin());
      active_ibs.erase(active_ibs.begin());
      // an ib on the dst node isn't allowed to go back to the source
      size_t first_dst_idx = ((src_idx < num_src_ibs) ? 0 : num_src_ibs);
      for(size_t dst_idx = first_dst_idx; dst_idx < total_ibs; dst_idx++) {
        // no self-loops either
        if(dst_idx == src_idx) continue;

        uint64_t cost;
        Channel *channel;
        XferDesKind kind;
        ChannelCopyInfo copy_info = channel_copy_info;
        copy_info.src_mem = partials[src_idx].ib_mem;
        copy_info.dst_mem = partials[dst_idx].ib_mem;
        copy_info.ind_mem = Memory::NO_MEMORY;
        copy_info.is_direct = false;
        if(find_best_channel_for_memories(nodes_info, copy_info, 0, 0,
                                          0, // no serdez or redop on interhops
                                          total_bytes, 0, 0, // no fragmentation also
                                          cost, channel, kind)) {

          NodeID src_node = ID(partials[src_idx].ib_mem).memory_owner_node();
          NodeID dst_node = ID(partials[dst_idx].ib_mem).memory_owner_node();
          size_t total_cost = partials[src_idx].cost + cost;
          log_xpath.info() << "inter: src_idx:" << src_idx << " "
                           << partials[src_idx].ib_mem << "(n:" << src_node
                           << ")-> dst_idx:" << dst_idx << " " << partials[dst_idx].ib_mem
                           << "(n:" << dst_node << ")"
                           << " channel=" << channel->kind
                           << " cost=" << partials[src_idx].cost << "+" << cost << " = "
                           << total_cost << " <? " << partials[dst_idx].cost;
          // also prune any path that already exceeds the cost of the direct path
          if(((partials[dst_idx].cost == 0) ||
              (total_cost < partials[dst_idx].cost)) &&
             ((best_cost == 0) || (total_cost < best_cost))) {
            // replace existing path to this dst ibmem
            partials[dst_idx].cost = total_cost;
            partials[dst_idx].path = partials[src_idx].path;
            partials[dst_idx].path.push_back(partials[dst_idx].ib_mem);
            partials[dst_idx].channels = partials[src_idx].channels;
            partials[dst_idx].channels.push_back(channel);
            active_ibs.insert(dst_idx);
          }
        }
      }
    }

    // finally, see which (reachable) ibs can get to the destination mem
    //  (and do it better than any previously known path)
    for(size_t i = 0; i < total_ibs; i++) {
      if(partials[i].cost == 0) continue;

      uint64_t cost;
      Channel *channel;
      XferDesKind kind;
      ChannelCopyInfo copy_info = channel_copy_info;
      copy_info.src_mem = partials[i].ib_mem;
      if(!channel_copy_info.is_scatter) {
        copy_info.ind_mem = Memory::NO_MEMORY;
      }
      copy_info.is_direct = false;
      if(find_best_channel_for_memories(
             nodes_info, copy_info, 0 /*no src serdez*/, serdez_id, redop_id, total_bytes,
             0 /*no src_frags*/, dst_frags, cost, channel, kind)) {
        NodeID src_node = ID(partials[i].ib_mem).memory_owner_node();
        size_t total_cost = partials[i].cost + cost;
        log_xpath.info() << "last: " << partials[i].ib_mem << "(n:" << src_node << ")->"
                         << dst_mem << "(n:" << dst_node << ")"
                         << " channel=" << channel->kind << " cost=" << partials[i].cost
                         << "+" << cost << " = " << total_cost << " <? " << best_cost;
        if((best_cost == 0) || (total_cost < best_cost)) {
          best_cost = total_cost;
          info.path.swap(partials[i].path);
          info.xd_channels.swap(partials[i].channels);

          if(!skip_final_memcpy || (kind != XFER_MEM_CPY)) {
            info.path.push_back(dst_mem);
            info.xd_channels.push_back(channel);
          }
        }
      }
    }

    if (path_cache_inited) {
      std::pair<realm_id_t, realm_id_t> key(src_mem.id, dst_mem.id);
      PathLRU *lru = path_cache.find(key)->second;
      PathLRU::LRUKey lru_key(serdez_id, redop_id, total_bytes, *src_frags, *dst_frags);
      // the LRU key is not in the LRU, now we cache it
      {
        RWLock::AutoWriterLock awl(lru->rwlock);
        // double check if lru key is already put in the LRU by another thread
        PathLRU::PathLRUIterator lru_it = lru->find(lru_key);
        if (lru_it != lru->end()) {
          lru->hit(lru_it);
          log_xpath_cache.debug() << "src:" << src_mem << ", dst:" << dst_mem << ", " << info << ", " << lru_key << ", Miss-Hit";
        } else {
          lru->miss(lru_key, info);
          log_xpath_cache.debug() << "src:" << src_mem << ", dst:" << dst_mem << ", " << info << ", " << lru_key << ", Miss";
        }
        nb_cache_miss.fetch_add(1);
      }
    }

    return (best_cost != 0);
  }

  IndirectionInfoBase::IndirectionInfoBase(bool _structured,
                                           FieldID _field_id,
                                           RegionInstance _inst,
                                           bool _is_ranges,
                                           bool _oor_possible,
                                           bool _aliasing_possible,
                                           size_t _subfield_offset,
                                           const std::vector<RegionInstance> _insts)
    : structured(_structured)
    , field_id(_field_id)
    , inst(_inst)
    , is_ranges(_is_ranges)
    , oor_possible(_oor_possible)
    , aliasing_possible(_aliasing_possible)
    , subfield_offset(_subfield_offset)
    , insts(_insts)
  {}

  static TransferGraph::XDTemplate::IO
  add_copy_path(std::vector<TransferGraph::XDTemplate> &xd_nodes,
                std::vector<TransferGraph::IBInfo> &ib_edges,
                TransferGraph::XDTemplate::IO start_edge, const MemPathInfo &info,
                size_t ib_size = Config::ib_size_bytes)
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

      xdn.channel = info.xd_channels[i];
      xdn.factory = info.xd_channels[i]->get_factory();
      xdn.gather_control_input = -1;
      xdn.scatter_control_input = -1;
      xdn.target_node = info.xd_channels[i]->node;
      xdn.channel = info.xd_channels[i];
      xdn.inputs.resize(1);
      xdn.inputs[0] = ((i == 0) ?
		         start_edge :
		         TransferGraph::XDTemplate::mk_edge(ib_base + i - 1));
      //xdn.inputs[0].indirect_inst = RegionInstance::NO_INST;
      xdn.outputs.resize(1);
      xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_base + i);

      TransferGraph::IBInfo& ibe = ib_edges[ib_base + i];
      ibe.memory = info.path[i + 1];
      ibe.size = ib_size;
    }

    // last edge we created is the output
    return TransferGraph::XDTemplate::mk_edge(ib_base + hops - 1);
  }

  // address splitters need to be able to read addresses from sysmem
  // TODO: query this from channel in order to support heterogeneity?
  static Memory find_sysmem_ib_memory(NodeID node)
  {
    Node& n = get_runtime()->nodes[node];
    for(std::vector<IBMemory *>::const_iterator it = n.ib_memories.begin();
        it != n.ib_memories.end();
        ++it)
      if(((*it)->lowlevel_kind == Memory::SYSTEM_MEM) ||
         ((*it)->lowlevel_kind == Memory::REGDMA_MEM) ||
         ((*it)->lowlevel_kind == Memory::SOCKET_MEM) ||
         ((*it)->lowlevel_kind == Memory::Z_COPY_MEM))
        return (*it)->me;

    log_dma.fatal() << "no sysmem ib memory on node " << node;
    abort();
    return Memory::NO_MEMORY;
  }

  void IndirectionInfoBase::generate_gather_paths(
      const Node *nodes_info, Memory dst_mem, TransferGraph::XDTemplate::IO dst_edge,
      unsigned indirect_idx, unsigned src_fld_start, unsigned src_fld_count,
      size_t bytes_per_element, CustomSerdezID serdez_id,
      std::vector<TransferGraph::XDTemplate> &xd_nodes,
      std::vector<TransferGraph::IBInfo> &ib_edges,
      std::vector<TransferDesc::FieldInfo> &src_fields)
  {
    // TODO: see how much of this we can reuse for the structured case?
    assert(!structured);

    size_t spaces_size = num_spaces();

    // compute the paths from each src data instance and the dst instance
    std::vector<size_t> path_idx;
    std::vector<MemPathInfo> path_infos;
    path_idx.reserve(spaces_size);
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
        std::vector<size_t> src_frags{domain_size()}, dst_frags{1};
        log_xpath.info() << "Find fastest path for gather op spaces:" << spaces_size;
        ChannelCopyInfo copy_info{insts[i].get_location(), dst_mem, inst.get_location(),
                                  spaces_size,
                                  /*is_scatter=*/false};
        populate_copy_info(copy_info);
        bool ok = find_fastest_path(nodes_info, path_cache, copy_info, serdez_id, 0,
                                    domain_size() * bytes_per_element, &src_frags,
                                    &dst_frags, path_infos[idx]);
        if(!ok) {
          // Couldn't find a path with the given indirect memory, so use a path without it
          // and we'll move the indirection buffer somewhere that channel can access it
          copy_info.ind_mem = Memory::NO_MEMORY;
          ok = find_fastest_path(nodes_info, path_cache, copy_info, serdez_id, 0,
                                 domain_size() * bytes_per_element, &src_frags,
                                 &dst_frags, path_infos[idx]);
        }
        assert(ok);
      }
    }

    // add a source field for the address
    unsigned addr_field_start = src_fields.size();
    src_fields.push_back(TransferDesc::FieldInfo{field_id, 0, address_size(), 0});
    TransferGraph::XDTemplate::IO addr_edge =
        TransferGraph::XDTemplate::mk_inst(inst, addr_field_start, 1);

    // special case - a gather from a single source with no out of range
    //  accesses
    if((spaces_size == 1) && !oor_possible) {
      size_t pathlen = path_infos[0].xd_channels.size();
      auto channel = path_infos[0].xd_channels[0];
      log_xpath.info() << "Gather channel kind=" << channel->kind
                       << " node=" << channel->node << " path len=" << pathlen;
      Memory ind_ib_mem = channel->suggest_ib_memories(inst.get_location());
      if(ind_ib_mem != Memory::NO_MEMORY) {
        /*log_xpath.info() << "Copy indirectiom from src_node="
                         << NodeID(ID(inst.get_location()).memory_owner_node())
                         << " to dst_node=" << NodeID(ID(ind_ib_mem).memory_owner_node())
                         << " ind_mem=" << ind_ib_mem
                         << " ind_mem_kind=" << ind_ib_mem.kind();*/
        MemPathInfo addr_path;
        bool ok = find_shortest_path(nodes_info, inst.get_location(), ind_ib_mem,
                                     0 /*no serdez*/, 0 /*redop_id*/, addr_path,
                                     true /*skip_final_memcpy*/);
        assert(ok);
        size_t aligned_ib_size =
            Config::ib_size_bytes +
            (address_size() - (Config::ib_size_bytes % address_size())) % address_size();
        addr_edge =
            add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path, aligned_ib_size);
      }

      size_t xd_idx = xd_nodes.size();
      size_t ib_idx = ib_edges.size();
      xd_nodes.resize(xd_idx + pathlen);
      ib_edges.resize(ib_idx + pathlen - 1);

      for(size_t i = 0; i < pathlen; i++) {
        TransferGraph::XDTemplate &xdn = xd_nodes[xd_idx + i];
        xdn.target_node = path_infos[0].xd_channels[i]->node;
        xdn.channel = path_infos[0].xd_channels[i];

        xdn.factory = path_infos[0].xd_channels[i]->get_factory();
        xdn.gather_control_input = -1;
        xdn.scatter_control_input = -1;
        if(i == 0) {
          xdn.inputs.resize(2);
          xdn.inputs[0] = TransferGraph::XDTemplate::mk_indirect(
              indirect_idx, 1, insts[0], src_fld_start, src_fld_count);
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

          TransferGraph::IBInfo &ibe = ib_edges[ib_idx + i];
          ibe.memory = path_infos[0].path[i + 1];
          ibe.size = Config::ib_size_bytes; // 1 << 20; /*TODO*/
        }
      }
    } else {
      // First complication: if all the data paths don't use the same final
      //  step, we need to force them to go through an intermediate
      // also insist that the final step be owned by the destination node
      //  (i.e. the merging should not be done via rdma)
      NodeID dst_node = ID(dst_mem).memory_owner_node();
      Memory dst_ib_mem = Memory::NO_MEMORY;
      Channel *last_channel =
          path_infos[0].xd_channels[path_infos[0].xd_channels.size() - 1];
      bool same_last_channel = true;
      if(last_channel->node == dst_node) {
        dst_ib_mem = last_channel->suggest_ib_memories(dst_mem);
        for(size_t i = 1; i < path_infos.size(); i++) {
          if(path_infos[i].xd_channels[path_infos[i].xd_channels.size() - 1] !=
             last_channel) {
            same_last_channel = false;
            break;
          }
        }
      } else {
        same_last_channel = false;
        dst_ib_mem = last_channel->suggest_ib_memories_for_node(dst_node);
      }

      if(!same_last_channel) {
        // figure out what the final kind will be (might not be the same as
        //  any of the current paths)
        MemPathInfo tail_path;
        bool ok = find_shortest_path(nodes_info, dst_ib_mem, dst_mem, serdez_id,
                                     0 /*redop_id*/, tail_path);
        assert(ok && (tail_path.xd_channels.size() == 1));
        last_channel = tail_path.xd_channels[0];
        // and fix any path that doesn't use that channel
        for(size_t i = 0; i < path_infos.size(); i++) {
          if(path_infos[i].xd_channels[path_infos[i].xd_channels.size() - 1] ==
             last_channel)
            continue;
          // log_new_dma.print() << "fix " << i << " " << path_infos[i].path[0] << " -> "
          // << dst_ib_mem;
          bool ok = find_shortest_path(nodes_info, path_infos[i].path[0], dst_ib_mem,
                                       0 /*no serdez*/, 0 /*redop_id*/, path_infos[i]);
          assert(ok);
          // append last step
          path_infos[i].xd_channels.push_back(last_channel);
          path_infos[i].path.push_back(dst_mem);
          // path_infos[i].xd_target_nodes.push_back(ID(dst_mem).memory_owner_node());
        }
      }

      // step 1: we need the address decoder, possibly with some hops to get
      //  the data to where a cpu can look at it
      NodeID addr_node = ID(inst).instance_owner_node();
      // HACK!
      MemPathInfo addr_path;

      XferDesFactory *addr_split_factory = create_addrsplit_factory(bytes_per_element);
      Memory addr_ib_mem =
          addr_split_factory->get_channel()->suggest_ib_memories_for_node(addr_node);

      bool ok = find_shortest_path(nodes_info, inst.get_location(), addr_ib_mem,
                                   0 /*no serdez*/, 0 /*redop_id*/, addr_path,
                                   true /*skip_final_memcpy*/);
      assert(ok);
      addr_edge = add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path);

      std::vector<TransferGraph::XDTemplate::IO> decoded_addr_edges(spaces_size);
      TransferGraph::XDTemplate::IO ctrl_edge;
      {
        // instantiate decoder
        size_t xd_base = xd_nodes.size();
        size_t ib_base = ib_edges.size();
        xd_nodes.resize(xd_base + 1);
        ib_edges.resize(ib_base + spaces_size + 1);

        TransferGraph::XDTemplate &xdn = xd_nodes[xd_base];
        xdn.target_node = addr_node;
        assert(!is_ranges && "need range address splitter");
        xdn.factory = addr_split_factory;
        xdn.gather_control_input = -1;
        xdn.scatter_control_input = -1;
        xdn.inputs.resize(1);
        xdn.inputs[0] = addr_edge;
        xdn.outputs.resize(spaces_size + 1);
        for(size_t i = 0; i < spaces_size; i++) {
          xdn.outputs[i] = TransferGraph::XDTemplate::mk_edge(ib_base + i);
          decoded_addr_edges[i] = xdn.outputs[i];
          ib_edges[ib_base + i].memory = addr_ib_mem;
          ib_edges[ib_base + i].size = 65536; // TODO
        }
        xdn.outputs[spaces_size] =
            TransferGraph::XDTemplate::mk_edge(ib_base + spaces_size);
        ctrl_edge = xdn.outputs[spaces_size];
        ib_edges[ib_base + spaces_size].memory = addr_ib_mem;
        ib_edges[ib_base + spaces_size].size = 65536; // TODO
      }

      // next, see what work we need to get the addresses to where the
      //  data instances live
      for(size_t i = 0; i < spaces_size; i++) {
        // HACK!
        Memory src_ib_mem = path_infos[path_idx[i]].xd_channels[0]->suggest_ib_memories(
            insts[i].get_location());
        if(src_ib_mem != addr_ib_mem) {
          MemPathInfo path;
          bool ok = find_shortest_path(nodes_info, addr_ib_mem, src_ib_mem,
                                       0 /*no serdez*/, 0 /*redop_id*/, path);
          assert(ok);
          decoded_addr_edges[i] =
              add_copy_path(xd_nodes, ib_edges, decoded_addr_edges[i], path);
        }
      }

      // control information has to get to the merge at the end
      // HACK!
      if(dst_ib_mem != addr_ib_mem) {
        MemPathInfo path;
        bool ok = find_shortest_path(nodes_info, addr_ib_mem, dst_ib_mem, 0 /*no serdez*/,
                                     0 /*redop_id*/, path);
        assert(ok);
        ctrl_edge = add_copy_path(xd_nodes, ib_edges, ctrl_edge, path);
      }

      // now any data paths with more than one hop need all but the last hop
      //  added to the graph
      std::vector<TransferGraph::XDTemplate::IO> data_edges(spaces_size);
      for(size_t i = 0; i < spaces_size; i++) {
        const MemPathInfo &mpi = path_infos[path_idx[i]];
        size_t hops = mpi.xd_channels.size() - 1;
        if(hops > 0) {
          size_t xd_base = xd_nodes.size();
          size_t ib_base = ib_edges.size();
          xd_nodes.resize(xd_base + hops);
          ib_edges.resize(ib_base + hops);
          for(size_t j = 0; j < hops; j++) {
            TransferGraph::XDTemplate &xdn = xd_nodes[xd_base + j];

            xdn.factory = mpi.xd_channels[j]->get_factory();
            xdn.gather_control_input = -1;
            xdn.scatter_control_input = -1;
            xdn.target_node = mpi.xd_channels[j]->node;
            xdn.channel = mpi.xd_channels[j];
            if(j == 0) {
              xdn.inputs.resize(2);
              xdn.inputs[0] = TransferGraph::XDTemplate::mk_indirect(
                  indirect_idx, 1, insts[i], src_fld_start, src_fld_count);
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
      TransferGraph::XDTemplate &xdn = xd_nodes[xd_idx];
      xdn.target_node = ID(dst_mem).memory_owner_node();
      // xdn.kind = last_kind;
      xdn.factory = last_channel->get_factory();
      xdn.gather_control_input = spaces_size;
      xdn.scatter_control_input = -1;
      xdn.outputs.resize(1);
      xdn.outputs[0] = dst_edge;

      xdn.inputs.resize(spaces_size + 1);

      for(size_t i = 0; i < spaces_size; i++) {
        // can we read (indirectly) right from the source instance?
        if(path_infos[path_idx[i]].xd_channels.size() == 1) {
          int ind_port = xdn.inputs.size();
          xdn.inputs.resize(ind_port + 1);
          xdn.inputs[i] = TransferGraph::XDTemplate::mk_indirect(
              indirect_idx, ind_port, insts[i], src_fld_start, src_fld_count);
          xdn.inputs[ind_port] = decoded_addr_edges[i];
        } else {
          // assert(data_edges[i] >= 0);
          xdn.inputs[i] = data_edges[i];
        }
      }

      // control input
      xdn.inputs[spaces_size] = ctrl_edge;
    }
  }

  void IndirectionInfoBase::generate_scatter_paths(
      Memory src_mem, TransferGraph::XDTemplate::IO src_edge, unsigned indirect_idx,
      unsigned dst_fld_start, unsigned dst_fld_count, size_t bytes_per_element,
      CustomSerdezID serdez_id, std::vector<TransferGraph::XDTemplate> &xd_nodes,
      std::vector<TransferGraph::IBInfo> &ib_edges,
      std::vector<TransferDesc::FieldInfo> &src_fields)
  {
    // TODO: see how much of this we can reuse for the structured case?
    assert(!structured);
    size_t spaces_size = num_spaces();

    // compute the paths from the src instance to each dst data instance
    std::vector<size_t> path_idx;
    std::vector<MemPathInfo> path_infos;
    path_idx.reserve(spaces_size);
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
        // TODO(apryakhin@): Technically this is not a correct number
        // of destination fragements that we get during scatter.
        // domain_size() returns just a maximum number of addresses
        // that we need to scatter and hence this value would be the
        // worst case. The actual number of consecutive rectangles (e.g
        // fragments) to handle can be less. Consider finding a
        // better way to handle this (same for gather op).
        std::vector<size_t> src_frags{1}, dst_frags{domain_size()};
        log_xpath.info() << "Find fastest path for scatter op.";
        ChannelCopyInfo copy_info{src_mem, insts[i].get_location(), inst.get_location(),
                                  spaces_size,
                                  /*is_scatter=*/true};
        populate_copy_info(copy_info);
        bool ok = find_fastest_path(get_runtime()->nodes, path_cache, copy_info,
                                    serdez_id, 0, domain_size() * bytes_per_element,
                                    &src_frags, &dst_frags, path_infos[idx]);
        if(!ok) {
          // Couldn't find a path with the given indirect memory, so use a path without it
          // and we'll move the indirection buffer somewhere that channel can access it
          copy_info.ind_mem = Memory::NO_MEMORY;
          ok = find_fastest_path(get_runtime()->nodes, path_cache, copy_info, serdez_id,
                                 0, domain_size() * bytes_per_element, &src_frags,
                                 &dst_frags, path_infos[idx]);
        }
        assert(ok);
      }
    }

    // add a source field for the address
    unsigned addr_field_start = src_fields.size();
    src_fields.push_back(TransferDesc::FieldInfo{field_id, 0, address_size(), 0});
    TransferGraph::XDTemplate::IO addr_edge =
        TransferGraph::XDTemplate::mk_inst(inst, addr_field_start, 1);

    // special case - a scatter to a single destination with no out of
    //  range accesses
    if((spaces_size == 1) && !oor_possible) {
      size_t pathlen = path_infos[0].xd_channels.size();
      auto channel = path_infos[0].xd_channels[pathlen - 1];
      log_xpath.info() << "Scatter channel kind=" << channel->kind
                       << " node=" << channel->node << " path len=" << pathlen;
      Memory ind_ib_mem = channel->suggest_ib_memories(inst.get_location());

      if(ind_ib_mem != Memory::NO_MEMORY) {
        log_xpath.info() << "Copy indirectiom from src_node="
                         << NodeID(ID(inst.get_location()).memory_owner_node())
                         << " to dst_node=" << NodeID(ID(ind_ib_mem).memory_owner_node())
                         << " ind_mem=" << ind_ib_mem
                         << " ind_mem_kind=" << ind_ib_mem.kind();
        // do we have to do anything to get the addresses into a cpu-readable
        //  memory on that node?
        MemPathInfo addr_path;
        bool ok = find_shortest_path(get_runtime()->nodes, inst.get_location(),
                                     ind_ib_mem, 0 /*no serdez*/, 0 /*redop_id*/,
                                     addr_path, true /*skip_final_memcpy*/);
        assert(ok);
        size_t aligned_ib_size =
            Config::ib_size_bytes +
            (address_size() - (Config::ib_size_bytes % address_size())) % address_size();
        addr_edge =
            add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path, aligned_ib_size);
      }

      size_t xd_idx = xd_nodes.size();
      size_t ib_idx = ib_edges.size();
      xd_nodes.resize(xd_idx + pathlen);
      ib_edges.resize(ib_idx + pathlen - 1);

      for(size_t i = 0; i < pathlen; i++) {
        TransferGraph::XDTemplate &xdn = xd_nodes[xd_idx + i];
        xdn.channel = path_infos[0].xd_channels[i];
        xdn.target_node = path_infos[0].xd_channels[i]->node;

        xdn.factory = path_infos[0].xd_channels[i]->get_factory();
        xdn.gather_control_input = -1;
        xdn.scatter_control_input = -1;
        if(i == (pathlen - 1)) {
          xdn.inputs.resize(2);
          xdn.inputs[1] = addr_edge;
        } else {
          xdn.inputs.resize(1);
        }
        xdn.inputs[0] =
            ((i == 0) ? src_edge : TransferGraph::XDTemplate::mk_edge(ib_idx + i - 1));

        xdn.outputs.resize(1);
        if(i == (pathlen - 1)) {
          xdn.outputs[0] = TransferGraph::XDTemplate::mk_indirect(
              indirect_idx, 1, insts[0], dst_fld_start, dst_fld_count);
        } else {
          xdn.outputs.resize(1);
          xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_idx + i);

          TransferGraph::IBInfo &ibe = ib_edges[ib_idx + i];
          ibe.memory = path_infos[0].path[i + 1];
          ibe.size = Config::ib_size_bytes; // 1 << 20; /*TODO*/
        }
      }
    } else {

      // First complication: if all the data paths don't use the same first
      //  step, we need to force them to go through an intermediate
      Memory src_ib_mem = find_sysmem_ib_memory(ID(src_mem).memory_owner_node());
      Channel *first_channel = path_infos[0].xd_channels[0];
      bool same_first_channel = true;
      for(size_t i = 1; i < path_infos.size(); i++) {
        if(path_infos[i].xd_channels[0] != first_channel) {
          same_first_channel = false;
          break;
        }
      }
      if(!same_first_channel) {
        // figure out what the first channel will be (might not be the same as
        //  any of the current paths)
        MemPathInfo head_path;
        bool ok = find_shortest_path(get_runtime()->nodes, src_mem, src_ib_mem, serdez_id,
                                     0 /*redop_id*/, head_path);
        assert(ok && (head_path.xd_channels.size() == 1));
        first_channel = head_path.xd_channels[0];
        // and fix any path that doesn't use that channel
        for(size_t i = 0; i < path_infos.size(); i++) {
          if(path_infos[i].xd_channels[0] == first_channel)
            continue;

          bool ok = find_shortest_path(get_runtime()->nodes, src_ib_mem,
                                       path_infos[i].path[path_infos[i].path.size() - 1],
                                       0 /*no serdez*/, 0 /*redop_id*/, path_infos[i]);
          assert(ok);
          // prepend last step
          path_infos[i].xd_channels.insert(path_infos[i].xd_channels.begin(),
                                           first_channel);
          path_infos[i].path.insert(path_infos[i].path.begin(), src_mem);
          // path_infos[i].xd_target_nodes.insert(path_infos[i].xd_target_nodes.begin(),
          //				       ID(src_mem).memory_owner_node());
        }
      }

      // step 1: we need the address decoder, possibly with some hops to get
      //  the data to where a cpu can look at it
      NodeID addr_node = ID(inst).instance_owner_node();
      // HACK!
      Memory addr_ib_mem = find_sysmem_ib_memory(addr_node);
      MemPathInfo addr_path;
      bool ok = find_shortest_path(get_runtime()->nodes, inst.get_location(), addr_ib_mem,
                                   0 /*no serdez*/, 0 /*redop_id*/, addr_path,
                                   true /*skip_final_memcpy*/);
      assert(ok);
      addr_edge = add_copy_path(xd_nodes, ib_edges, addr_edge, addr_path);

      std::vector<TransferGraph::XDTemplate::IO> decoded_addr_edges(spaces_size);
      TransferGraph::XDTemplate::IO ctrl_edge;
      {
        // instantiate decoder
        size_t xd_base = xd_nodes.size();
        size_t ib_base = ib_edges.size();
        xd_nodes.resize(xd_base + 1);
        ib_edges.resize(ib_base + spaces_size + 1);

        TransferGraph::XDTemplate &xdn = xd_nodes[xd_base];
        xdn.target_node = addr_node;
        // xdn.kind = XFER_ADDR_SPLIT;
        assert(!is_ranges && "need range address splitter");
        xdn.factory = create_addrsplit_factory(bytes_per_element);
        xdn.gather_control_input = -1;
        xdn.scatter_control_input = -1;
        xdn.inputs.resize(1);
        xdn.inputs[0] = addr_edge;
        xdn.outputs.resize(spaces_size + 1);
        for(size_t i = 0; i < spaces_size; i++) {
          xdn.outputs[i] = TransferGraph::XDTemplate::mk_edge(ib_base + i);
          decoded_addr_edges[i] = xdn.outputs[i];
          ib_edges[ib_base + i].memory = addr_ib_mem;
          ib_edges[ib_base + i].size = 65536; // TODO
        }
        xdn.outputs[spaces_size] =
            TransferGraph::XDTemplate::mk_edge(ib_base + spaces_size);
        ctrl_edge = xdn.outputs[spaces_size];
        ib_edges[ib_base + spaces_size].memory = addr_ib_mem;
        ib_edges[ib_base + spaces_size].size = 65536; // TODO
      }

      // control information has to get to the split at the start
      // HACK!
      if(src_ib_mem != addr_ib_mem) {
        MemPathInfo path;
        bool ok = find_shortest_path(get_runtime()->nodes, addr_ib_mem, src_ib_mem,
                                     0 /*no serdez*/, 0 /*redop_id*/, path);
        assert(ok);
        ctrl_edge = add_copy_path(xd_nodes, ib_edges, ctrl_edge, path);
      }

      // next, see what work we need to get the addresses to where the
      //  last step of each path is running
      for(size_t i = 0; i < spaces_size; i++) {
        // HACK!
        NodeID dst_node = path_infos[path_idx[i]]
                              .xd_channels[path_infos[path_idx[i]].xd_channels.size() - 1]
                              ->node;
        Memory dst_ib_mem = find_sysmem_ib_memory(dst_node);
        if(dst_ib_mem != addr_ib_mem) {
          MemPathInfo path;
          bool ok = find_shortest_path(get_runtime()->nodes, addr_ib_mem, dst_ib_mem,
                                       0 /*no serdez*/, 0 /*redop_id*/, path);
          assert(ok);
          decoded_addr_edges[i] =
              add_copy_path(xd_nodes, ib_edges, decoded_addr_edges[i], path);
        }
      }

      // next comes the xd that reads the source and splits the data into
      //  the various output streams
      std::vector<TransferGraph::XDTemplate::IO> data_edges(spaces_size);
      {
        size_t xd_idx = xd_nodes.size();
        xd_nodes.resize(xd_idx + 1);

        TransferGraph::XDTemplate &xdn = xd_nodes[xd_idx];
        xdn.target_node = ID(src_mem).memory_owner_node();
        // xdn.kind = first_kind;
        xdn.factory = first_channel->get_factory();
        xdn.gather_control_input = -1;
        xdn.scatter_control_input = 1;
        xdn.inputs.resize(2);
        xdn.inputs[0] = src_edge;
        xdn.inputs[1] = ctrl_edge;

        xdn.outputs.resize(spaces_size);

        for(size_t i = 0; i < spaces_size; i++) {
          // can we write (indirectly) right into the dest instance?
          if(path_infos[path_idx[i]].xd_channels.size() == 1) {
            int ind_port = xdn.inputs.size();
            xdn.inputs.resize(ind_port + 1);
            xdn.outputs[i] = TransferGraph::XDTemplate::mk_indirect(
                indirect_idx, ind_port, insts[i], dst_fld_start, dst_fld_count);
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
      for(size_t i = 0; i < spaces_size; i++) {
        const MemPathInfo &mpi = path_infos[path_idx[i]];
        size_t hops = mpi.xd_channels.size() - 1;
        if(hops > 0) {
          // assert(data_edges[i] >= 0);

          size_t xd_base = xd_nodes.size();
          size_t ib_base = ib_edges.size();
          xd_nodes.resize(xd_base + hops);
          ib_edges.resize(ib_base + hops - 1);
          for(size_t j = 0; j < hops; j++) {
            TransferGraph::XDTemplate &xdn = xd_nodes[xd_base + j];

            xdn.factory = mpi.xd_channels[j + 1]->get_factory();
            xdn.gather_control_input = -1;
            xdn.scatter_control_input = -1;
            xdn.target_node = mpi.xd_channels[j + 1]->node;
            xdn.channel = mpi.xd_channels[j + 1];
            if(j < (hops - 1)) {
              xdn.inputs.resize(1);
              xdn.inputs[0] =
                  ((j == 0) ? data_edges[i]
                            : TransferGraph::XDTemplate::mk_edge(ib_base + j - 1));
              xdn.outputs.resize(1);
              xdn.outputs[0] = TransferGraph::XDTemplate::mk_edge(ib_base + j);
              ib_edges[ib_base + j].memory = mpi.path[j + 2];
              ib_edges[ib_base + j].size = 65536; // TODO: pick size?
            } else {
              // last hop uses the address stream
              xdn.inputs.resize(2);
              xdn.inputs[0] =
                  ((j == 0) ? data_edges[i]
                            : TransferGraph::XDTemplate::mk_edge(ib_base + j - 1));
              xdn.inputs[1] = decoded_addr_edges[i];
              xdn.outputs.resize(1);
              xdn.outputs[0] = TransferGraph::XDTemplate::mk_indirect(
                  indirect_idx, 1, insts[i], dst_fld_start, dst_fld_count);
            }
          }
        }
      }
    }
  }

  template <int N, typename T, int N2, typename T2>
  IndirectionInfoTyped<N, T, N2, T2>::IndirectionInfoTyped(
      const IndexSpace<N, T> &is,
      const typename CopyIndirection<N, T>::template Unstructured<N2, T2> &ind,
      Channel *_addr_split_channel)
    : IndirectionInfoBase(false /*!structured*/, ind.field_id, ind.inst, ind.is_ranges,
                          ind.oor_possible, ind.aliasing_possible, ind.subfield_offset,
                          ind.insts)
    , domain(is)
    , spaces(ind.spaces)
    , addr_split_channel(_addr_split_channel)
  {}

  template <int N, typename T, int N2, typename T2>
  Event IndirectionInfoTyped<N,T,N2,T2>::request_metadata(void)
  {
    std::vector<Event> evs;

    {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(inst);
      Event e = impl->request_metadata();
      if(!e.has_triggered()) evs.push_back(e);
    }

    for(std::vector<RegionInstance>::const_iterator it = insts.begin();
	it != insts.end();
	++it) {
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(*it);
      Event e = impl->request_metadata();
      if(!e.has_triggered()) evs.push_back(e);
    }

    return Event::merge_events(evs);
  }

  template <int N, typename T, int N2, typename T2>
  size_t IndirectionInfoTyped<N,T,N2,T2>::num_spaces() const
  {
    return spaces.size();
  }

  template <int N, typename T, int N2, typename T2>
  void IndirectionInfoTyped<N, T, N2, T2>::populate_copy_info(ChannelCopyInfo &info) const
  {
    info.is_ranges = is_ranges;
    info.addr_size = sizeof(T2);
    info.oor_possible = oor_possible;
  }

  template <int N, typename T, int N2, typename T2>
  size_t IndirectionInfoTyped<N, T, N2, T2>::domain_size() const {
    return domain.volume();
  }

  template <int N, typename T, int N2, typename T2>
  size_t IndirectionInfoTyped<N,T,N2,T2>::address_size() const
  {
    return (is_ranges ?
              sizeof(Rect<N2,T2>) :
              sizeof(Point<N2,T2>));
  }

  template <int N, typename T, int N2, typename T2>
  XferDesFactory *IndirectionInfoTyped<N,T,N2,T2>::create_addrsplit_factory(size_t bytes_per_element) const
  {
    return new AddressSplitXferDesFactory<N2, T2>(bytes_per_element, spaces,
                                                  addr_split_channel);
  }

  template <int N, typename T, int N2, typename T2>
  RegionInstance IndirectionInfoTyped<N,T,N2,T2>::get_pointer_instance(void) const
  {
    return inst;
  }

  template <int N, typename T, int N2, typename T2>
  const std::vector<RegionInstance>* IndirectionInfoTyped<N,T,N2,T2>::get_instances(void) const
  {
    return &insts;
  }

  template <int N, typename T, int N2, typename T2>
  FieldID IndirectionInfoTyped<N,T,N2,T2>::get_field(void) const
  {
    return field_id;
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
  TransferIterator *IndirectionInfoTyped<N, T, N2, T2>::create_indirect_iterator(
      Memory addrs_mem, RegionInstance inst, const std::vector<FieldID> &fields,
      const std::vector<size_t> &fld_offsets, const std::vector<size_t> &fld_sizes,
      Channel *channel) const
  {
    if(channel && channel->needs_wrapping_iterator()) {
      return new WrappingTransferIteratorIndirect<N2, T2>(inst, fields, fld_offsets,
                                                          fld_sizes);
    } else {
      if(is_ranges)
        return new TransferIteratorIndirectRange<N2, T2>(addrs_mem, inst, fields,
                                                         fld_offsets, fld_sizes);
      else
        return new TransferIteratorIndirect<N2, T2>(addrs_mem, inst, fields, fld_offsets,
                                                    fld_sizes);
    }
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
    // The next indirection is not allowed to be specified yet.
    assert(next_indirection == nullptr);
    return new IndirectionInfoTyped<N, T, N2, T2>(is, *this, local_addrsplit_channel);
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

    if(!preconditions.empty()) {
      Event merged = Event::merge_events(preconditions);
      if(merged.exists()) {
	deferred_analysis.precondition = merged;
        EventImpl::add_waiter(merged, &deferred_analysis);
        return;
      }
    }

    // no (untriggered) preconditions, so we fall through to immediate analysis
    perform_analysis();
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

  struct IBAllocOrderSorter {
    IBAllocOrderSorter(const std::vector<TransferGraph::IBInfo>& _edges)
      : edges(_edges) {}

    bool operator()(unsigned a, unsigned b) const {
      // first sort by ascending memory ID
      if(edges[a].memory.id < edges[b].memory.id) return true;
      if(edges[a].memory.id > edges[b].memory.id) return false;
      // next by decreasing size
      if(edges[a].size > edges[b].size) return true;
      if(edges[a].size < edges[b].size) return false;
      // finally by index itself for stability
      return (a < b);
    }

    const std::vector<TransferGraph::IBInfo>& edges;
  };

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
        // release before the mutex is released so to_alloc is visible before the
        // analysis_complete flag is set
        analysis_complete.store_release(true);
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
    std::vector<bool> field_done(srcs.size(), false);
    // fields will get reordered to be contiguous per xd subgraph
    size_t fld_start = 0;
    for(size_t i = 0; i < srcs.size(); i++) {
      // did this field already get grouped into a previous path?
      if(field_done[i])
        continue;

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

	src_fields[fld_start] = FieldInfo { srcs[i].field_id,
                                            srcs[i].subfield_offset,
                                            srcs[i].size, srcs[i].serdez_id };
	dst_fields[fld_start] = FieldInfo { dsts[i].field_id,
                                            dsts[i].subfield_offset,
                                            dsts[i].size, dsts[i].serdez_id };

        Memory src_mem = srcs[i].inst.get_location();
        Memory dst_mem = dsts[i].inst.get_location();

        std::vector<size_t> src_frags, dst_frags;
        domain->count_fragments(srcs[i].inst, dim_order,
                                std::vector<FieldID>(1, srcs[i].field_id),
                                std::vector<size_t>(1, srcs[i].size),
                                src_frags);
        domain->count_fragments(dsts[i].inst, dim_order,
                                std::vector<FieldID>(1, dsts[i].field_id),
                                std::vector<size_t>(1, dsts[i].size),
                                dst_frags);

        MemPathInfo path_info;
        bool ok = find_fastest_path(get_runtime()->nodes, path_cache,
                                    ChannelCopyInfo{src_mem, dst_mem}, serdez_id,
                                    dsts[i].redop_id, domain_size * combined_field_size,
                                    &src_frags, &dst_frags, path_info);
        if(!ok) {
          log_new_dma.fatal() << "FATAL: no path found from " << src_mem << " to "
                              << dst_mem << " (redop=" << dsts[i].redop_id << ")";
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
          xdn.channel = path_info.xd_channels[j];
          if(j == (pathlen - 1))
            xdn.redop = XferDesRedopInfo(dsts[i].redop_id,
                                         dsts[i].red_fold,
                                         true /*in_place*/,
                                         dsts[i].red_exclusive);
          xdn.inputs.resize(1);
          xdn.inputs[0] = ((j == 0) ?
                             TransferGraph::XDTemplate::mk_inst(srcs[i].inst,
                                                                fld_start, 1) :
                             TransferGraph::XDTemplate::mk_edge(ib_idx - 1));
          //xdn.inputs[0].indirect_inst = RegionInstance::NO_INST;
          xdn.outputs.resize(1);
          xdn.outputs[0] = ((j == (pathlen - 1)) ?
                              TransferGraph::XDTemplate::mk_inst(dsts[i].inst,
                                                                 fld_start, 1) :
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
	std::vector<RegionInstance> instinfo_src_insts{srcs[i].inst};
	std::vector<RegionInstance> instinfo_dst_insts{dsts[i].inst};
	std::vector<FieldID> instinfo_src_field_ids{srcs[i].field_id};
	std::vector<FieldID> instinfo_dst_field_ids{dsts[i].field_id};
        prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
	  instinfo_src_insts,
	  instinfo_dst_insts,
	  RegionInstance::NO_INST,
	  RegionInstance::NO_INST,
	  instinfo_src_field_ids,
	  instinfo_dst_field_ids,
	  0,
	  0,
          ProfilingMeasurements::OperationCopyInfo::REDUCE,
          unsigned(pathlen) });
        fld_start += 1;
      }
      else if(srcs[i].field_id == FieldID(-1)) {
	// fill
	assert((dsts[i].indirect_index == -1) && "help: scatter fill!");

	src_fields[fld_start] = FieldInfo { FieldID(-1), 0, 0, 0 };
	dst_fields[fld_start] = FieldInfo { dsts[i].field_id,
                                            dsts[i].subfield_offset,
                                            dsts[i].size,
                                            dsts[i].serdez_id };

	Memory dst_mem = dsts[i].inst.get_location();
	MemPathInfo path_info;

        ChannelCopyInfo copy_info(Memory::NO_MEMORY, dst_mem);
        bool ok = find_fastest_path(get_runtime()->nodes, path_cache,
                                    copy_info, serdez_id, 0, domain_size, nullptr,
                                    nullptr, path_info);
        if(!ok) {
          log_new_dma.fatal() << "FATAL: no fill path found for " << dst_mem
                              << " (serdez=" << serdez_id << ")";
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
          xdn.channel = path_info.xd_channels[j];
          xdn.inputs.resize(1);
          xdn.inputs[0] = ((j == 0) ?
                             TransferGraph::XDTemplate::mk_fill(fill_ofs,
                                                                combined_field_size,
                                                                domain_size * combined_field_size) :
                             TransferGraph::XDTemplate::mk_edge(ib_idx - 1));

	  xdn.outputs.resize(1);
          xdn.outputs[0] = ((j == (pathlen - 1)) ?
                              TransferGraph::XDTemplate::mk_inst(dsts[i].inst,
                                                                 fld_start, 1) :
                              TransferGraph::XDTemplate::mk_edge(ib_idx));
          if(j < (pathlen - 1)) {
            TransferGraph::IBInfo& ibe = graph.ib_edges[ib_idx++];
            ibe.memory = path_info.path[j + 1];
            ibe.size = ib_alloc_size;
          }
	}

        // FIXME: handle multiple fields
        memcpy(static_cast<char *>(fill_data)+fill_ofs,
               ((srcs[i].size <= CopySrcDstField::MAX_DIRECT_SIZE) ?
                  srcs[i].fill_data.direct :
                  srcs[i].fill_data.indirect),
               srcs[i].size);
        fill_ofs += srcs[i].size;

        prof_usage.source = Memory::NO_MEMORY;
        prof_usage.target = dst_mem;
        prof_usage.size += domain_size * combined_field_size;
	std::vector<RegionInstance> instinfo_src_insts;
	std::vector<RegionInstance> instinfo_dst_insts{dsts[i].inst};
	std::vector<FieldID> instinfo_src_field_ids;
	std::vector<FieldID> instinfo_dst_field_ids{dsts[i].field_id};
        prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
	      instinfo_src_insts,
	      instinfo_dst_insts,
	      RegionInstance::NO_INST,
	      RegionInstance::NO_INST,
	      instinfo_src_field_ids,
	      instinfo_dst_field_ids,
	      0,
	      0,
              ProfilingMeasurements::OperationCopyInfo::FILL,
              unsigned(pathlen) });
        fld_start += 1;
      } else {
	src_fields[fld_start] = FieldInfo { srcs[i].field_id,
                                            srcs[i].subfield_offset,
                                            srcs[i].size,
                                            srcs[i].serdez_id };
	dst_fields[fld_start] = FieldInfo { dsts[i].field_id,
                                            dsts[i].subfield_offset,
                                            dsts[i].size,
                                            dsts[i].serdez_id };
	
	if(srcs[i].indirect_index == -1) {
	  Memory src_mem = srcs[i].inst.get_location();

	  if(dsts[i].indirect_index == -1) {
	    Memory dst_mem = dsts[i].inst.get_location();

            unsigned num_fields = 1;
            std::vector<FieldID> src_field_ids(1, srcs[i].field_id);
            std::vector<size_t> src_field_sizes(1, srcs[i].size);
            std::vector<FieldID> dst_field_ids(1, dsts[i].field_id);
            std::vector<size_t> dst_field_sizes(1, dsts[i].size);

            // group any other fields that have the same insts/size/redop/serdez
            for(size_t j = i + 1; j < srcs.size(); j++) {
              if(field_done[j]) continue;
              if(srcs[j].inst != srcs[i].inst) continue;
              if(srcs[j].size != srcs[i].size) continue;
              if(srcs[j].redop_id != srcs[i].redop_id) continue;
              if(srcs[j].serdez_id != srcs[i].serdez_id) continue;
              if(dsts[j].inst != dsts[i].inst) continue;
              if(dsts[j].size != dsts[i].size) continue;
              if(dsts[j].redop_id != dsts[i].redop_id) continue;
              if(dsts[j].serdez_id != dsts[i].serdez_id) continue;

              src_field_ids.push_back(srcs[j].field_id);
              src_field_sizes.push_back(srcs[j].size);
              dst_field_ids.push_back(dsts[j].field_id);
              dst_field_sizes.push_back(dsts[j].size);

              src_fields[fld_start + num_fields] = FieldInfo {
                                                       srcs[j].field_id,
                                                       srcs[j].subfield_offset,
                                                       srcs[j].size,
                                                       srcs[j].serdez_id };
              dst_fields[fld_start + num_fields] = FieldInfo {
                                                       dsts[j].field_id,
                                                       dsts[j].subfield_offset,
                                                       dsts[j].size,
                                                       dsts[j].serdez_id };
              num_fields += 1;
              combined_field_size += srcs[j].size;
              field_done[j] = true;
            }

            std::vector<size_t> src_frags, dst_frags;
            domain->count_fragments(srcs[i].inst, dim_order,
                                    src_field_ids, src_field_sizes,
                                    src_frags);
            domain->count_fragments(dsts[i].inst, dim_order,
                                    dst_field_ids, dst_field_sizes,
                                    dst_frags);
            //log_new_dma.print() << "fragments: domain=" << *domain
            //                    << " src_inst=" << srcs[i].inst << " frags=" << PrettyVector<size_t>(src_frags)
            //                    << " dst_inst=" << dsts[i].inst << " frags=" << PrettyVector<size_t>(dst_frags);

            MemPathInfo path_info;
            bool ok = find_fastest_path(get_runtime()->nodes, path_cache,
                                        ChannelCopyInfo{src_mem, dst_mem}, serdez_id,
                                        0 /*redop_id*/, domain_size * combined_field_size,
                                        &src_frags, &dst_frags, path_info);
            if(!ok) {
              log_new_dma.fatal() << "FATAL: no path found from " << src_mem << " to "
                                  << dst_mem << " (serdez=" << serdez_id << ")";
              assert(0);
            }
            size_t pathlen = path_info.xd_channels.size();
            size_t xd_idx = graph.xd_nodes.size();
            size_t ib_idx = graph.ib_edges.size();
            size_t ib_alloc_size = 0;
            graph.xd_nodes.resize(xd_idx + pathlen);
            if(pathlen > 1) {
              graph.ib_edges.resize(ib_idx + pathlen - 1);
              ib_alloc_size =
                  compute_ib_size(combined_field_size, domain_size, serdez_id);
            }
            for(size_t j = 0; j < pathlen; j++) {
              TransferGraph::XDTemplate &xdn = graph.xd_nodes[xd_idx++];

              // xdn.kind = path_info.xd_kinds[j];
              xdn.factory = path_info.xd_channels[j]->get_factory();
              xdn.gather_control_input = -1;
              xdn.scatter_control_input = -1;
              xdn.target_node = path_info.xd_channels[j]->node;
              xdn.channel = path_info.xd_channels[j];
              xdn.inputs.resize(1);
              xdn.inputs[0] = ((j == 0) ? TransferGraph::XDTemplate::mk_inst(
                                              srcs[i].inst, fld_start, num_fields)
                                        : TransferGraph::XDTemplate::mk_edge(ib_idx - 1));
              // xdn.inputs[0].indirect_inst = RegionInstance::NO_INST;
              xdn.outputs.resize(1);
              xdn.outputs[0] =
                  ((j == (pathlen - 1)) ? TransferGraph::XDTemplate::mk_inst(
                                              dsts[i].inst, fld_start, num_fields)
                                        : TransferGraph::XDTemplate::mk_edge(ib_idx));
              // xdn.outputs[0].indirect_inst = RegionInstance::NO_INST;
              if(j < (pathlen - 1)) {
                TransferGraph::IBInfo &ibe = graph.ib_edges[ib_idx++];
                ibe.memory = path_info.path[j + 1];
                ibe.size = ib_alloc_size;
              }
            }

            prof_usage.source = src_mem;
            prof_usage.target = dst_mem;
            prof_usage.size += domain_size * combined_field_size;
	    std::vector<RegionInstance> instinfo_src_insts{srcs[i].inst};
	    std::vector<RegionInstance> instinfo_dst_insts{dsts[i].inst};
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
		 instinfo_src_insts,
		 instinfo_dst_insts,
		 RegionInstance::NO_INST,
		 RegionInstance::NO_INST,
		 src_field_ids,
		 dst_field_ids,
		 0,
		 0,
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(pathlen) });
            fld_start += num_fields;
	  } else {
	    // scatter
	    IndirectionInfo *scatter_info = indirects[dsts[i].indirect_index];
	    size_t addrsplit_bytes_per_element = ((serdez_id == 0) ?
						    combined_field_size :
						    1);
            size_t prev_nodes = graph.xd_nodes.size();
	    scatter_info->generate_scatter_paths(src_mem,
						 TransferGraph::XDTemplate::mk_inst(srcs[i].inst, fld_start, 1),
						 dsts[i].indirect_index,
						 fld_start, 1,
						 addrsplit_bytes_per_element,
						 serdez_id,
						 graph.xd_nodes,
						 graph.ib_edges,
						 src_fields);

            prof_usage.source = src_mem;
            prof_usage.target = Memory::NO_MEMORY;
            prof_usage.size += domain_size * combined_field_size;
	    std::vector<RegionInstance> instinfo_src_insts{srcs[i].inst};
	    std::vector<RegionInstance> instinfo_dst_insts;
	    if (scatter_info->get_instances()) {
	      instinfo_dst_insts = *(scatter_info->get_instances());
	    }
	    std::vector<FieldID> instinfo_src_field_ids{srcs[i].field_id};
	    std::vector<FieldID> instinfo_dst_field_ids{dsts[i].field_id};
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
		 instinfo_src_insts,
		 instinfo_dst_insts,
		 RegionInstance::NO_INST,
		 scatter_info->get_pointer_instance(),
		 instinfo_src_field_ids,
		 instinfo_dst_field_ids,
		 0,
		 scatter_info->get_field(),
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(graph.xd_nodes.size() - prev_nodes) });
            fld_start += 1;
	  }
	} else {
	  size_t addrsplit_bytes_per_element = ((serdez_id == 0) ?
						  combined_field_size :
						  1);
	  if(dsts[i].indirect_index == -1) {
	    Memory dst_mem = dsts[i].inst.get_location();
	    IndirectionInfo *gather_info = indirects[srcs[i].indirect_index];
            size_t prev_nodes = graph.xd_nodes.size();
            gather_info->generate_gather_paths(
                get_runtime()->nodes, dst_mem,
                TransferGraph::XDTemplate::mk_inst(dsts[i].inst, fld_start, 1),
                srcs[i].indirect_index, fld_start, 1, addrsplit_bytes_per_element,
                serdez_id, graph.xd_nodes, graph.ib_edges, src_fields);

            prof_usage.source = Memory::NO_MEMORY;
            prof_usage.target = dst_mem;
            prof_usage.size += domain_size * combined_field_size;
	    std::vector<RegionInstance> instinfo_src_insts;
	    if (gather_info->get_instances()) {
	      instinfo_src_insts = *(gather_info->get_instances());
	    }
	    std::vector<RegionInstance> instinfo_dst_insts{dsts[i].inst};
	    std::vector<FieldID> instinfo_src_field_ids{srcs[i].field_id};
	    std::vector<FieldID> instinfo_dst_field_ids{dsts[i].field_id};
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
		 instinfo_src_insts,
		 instinfo_dst_insts,
		 gather_info->get_pointer_instance(),
		 RegionInstance::NO_INST,
		 instinfo_src_field_ids,
		 instinfo_dst_field_ids,
		 gather_info->get_field(),
		 0,
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(graph.xd_nodes.size() - prev_nodes) });
            fld_start += 1;
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
            gather_info->generate_gather_paths(
                get_runtime()->nodes, ib_mem, TransferGraph::XDTemplate::mk_edge(ib_idx),
                srcs[i].indirect_index, fld_start, 1, addrsplit_bytes_per_element,
                serdez_id, graph.xd_nodes, graph.ib_edges, src_fields);

            IndirectionInfo *scatter_info = indirects[dsts[i].indirect_index];
            scatter_info->generate_scatter_paths(
                ib_mem, TransferGraph::XDTemplate::mk_edge(ib_idx),
                dsts[i].indirect_index, fld_start, 1, addrsplit_bytes_per_element,
                serdez_id, graph.xd_nodes, graph.ib_edges, src_fields);

            prof_usage.source = Memory::NO_MEMORY;
            prof_usage.target = Memory::NO_MEMORY;
            prof_usage.size += domain_size * combined_field_size;
	    std::vector<RegionInstance> instinfo_src_insts;
	    if (gather_info->get_instances()) {
	      instinfo_src_insts = *(gather_info->get_instances());
	    }
	    std::vector<RegionInstance> instinfo_dst_insts;
	    if (scatter_info->get_instances()) {
	      instinfo_dst_insts = *(scatter_info->get_instances());
	    }
	    std::vector<FieldID> instinfo_src_field_ids{srcs[i].field_id};
	    std::vector<FieldID> instinfo_dst_field_ids{dsts[i].field_id};
            prof_cpinfo.inst_info.push_back(ProfilingMeasurements::OperationCopyInfo::InstInfo {
		 instinfo_src_insts,
		 instinfo_dst_insts,
		 gather_info->get_pointer_instance(),
		 scatter_info->get_pointer_instance(),
		 instinfo_src_field_ids,
		 instinfo_dst_field_ids,
		 gather_info->get_field(),
		 scatter_info->get_field(),
                 ProfilingMeasurements::OperationCopyInfo::COPY,
                 unsigned(graph.xd_nodes.size() - prev_nodes) });
            fld_start += 1;
	  }
	}
      }
    }
    // make sure the reordered field list includes them all
    assert(fld_start == srcs.size());

    // once we've enumerated all the ibs we'll need, we need to pick an order in
    //  which to allocate them that will avoid deadlock when multiple transfer
    //  operations are allocating ibs concurrently - sorting by the memory (and
    //  then having the allocation code do all ibs for the same memory for a single
    //  transfer op atomically) does the trick
    if(!graph.ib_edges.empty()) {
      graph.ib_alloc_order.resize(graph.ib_edges.size());
      for(size_t i = 0; i < graph.ib_edges.size(); i++)
        graph.ib_alloc_order[i] = i;

      std::sort(graph.ib_alloc_order.begin(), graph.ib_alloc_order.end(),
                IBAllocOrderSorter(graph.ib_edges));
    }

    if(log_xplan.want_debug()) {
      log_xplan.debug() << "analysis: plan=" << (void *)this
                        << " dim_order=" << PrettyVector<int>(dim_order)
                        << " xds=" << graph.xd_nodes.size()
                        << " ibs=" << graph.ib_edges.size();

      for(size_t i = 0; i < graph.xd_nodes.size(); i++) {
        if(graph.xd_nodes[i].redop.id != 0) {
          log_xplan.debug()
              << "analysis: plan=" << (void *)this << " xds[" << i
              << "]: target=" << graph.xd_nodes[i].target_node << " inputs="
              << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].inputs)
              << " outputs="
              << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].outputs)
              << " channel="
              << ((graph.xd_nodes[i].channel) ? graph.xd_nodes[i].channel->kind : -1)
              << " redop=(" << graph.xd_nodes[i].redop.id << ","
              << graph.xd_nodes[i].redop.is_fold << ","
              << graph.xd_nodes[i].redop.in_place << ")";
        } else {
          log_xplan.debug()
              << "analysis: plan=" << (void *)this << " xds[" << i
              << "]: target=" << graph.xd_nodes[i].target_node << " inputs="
              << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].inputs)
              << " outputs="
              << PrettyVector<TransferGraph::XDTemplate::IO>(graph.xd_nodes[i].outputs)
              << " channel="
              << ((graph.xd_nodes[i].channel) ? graph.xd_nodes[i].channel->kind : -1);
        }
      }

      for(size_t i = 0; i < graph.ib_edges.size(); i++) {
        log_xplan.debug() << "analysis: plan=" << (void *)this << " ibs[" << i
                          << "]: memory=" << graph.ib_edges[i].memory << ":"
                          << graph.ib_edges[i].memory.kind()
                          << " size=" << graph.ib_edges[i].size;
      }

      if(!graph.ib_edges.empty()) {
        log_xplan.debug() << "analysis: plan=" << (void *)this
                          << " ib_alloc=" << PrettyVector<unsigned>(graph.ib_alloc_order);
      }
    }

    // mark that the analysis is complete and see if there are any pending
    //  ops that can start allocating ibs
    std::vector<TransferOperation *> to_alloc;
    {
      AutoLock<> al(mutex);
      to_alloc.swap(pending_ops);
      analysis_successful = true;
      // release before the mutex is released so to_alloc is visible before the
      // analysis_complete flag is set
      analysis_complete.store_release(true);
    }

    for(size_t i = 0; i < to_alloc.size(); i++)
      to_alloc[i]->allocate_ibs();
  }

  void TransferDesc::cancel_analysis(Event failed_precondition)
  {
    // mark that the analysis is failed and see if there are any pending
    //  ops that need to also fail
    std::vector<TransferOperation *> to_alloc;
    {
      AutoLock<> al(mutex);
      to_alloc.swap(pending_ops);
      analysis_successful = false;
      // release before the mutex is released so to_alloc is visible before the
      // analysis_complete flag is set
      analysis_complete.store_release(true);
    }

    for(size_t i = 0; i < to_alloc.size(); i++)
      to_alloc[i]->handle_poisoned_precondition(failed_precondition);
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
    // TODO: respect time limit
    if(poisoned)
      desc->cancel_analysis(precondition);
    else
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
				       EventImpl::gen_t _finish_gen,
				       int _priority)
    : Operation(_finish_event, _finish_gen, _desc.prs)
    , deferred_start(this)
    , desc(_desc)
    , precondition(_precondition)
    , ib_responses_needed(0)
    , priority(_priority)
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
      deferred_start.precondition = precondition;
      EventImpl::add_waiter(precondition, &deferred_start);
      return;
    }
    if(poisoned) {
      handle_poisoned_precondition(precondition);
      return;
    }

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

    // if the transfer analysis was not successful, we can't continue
    if(!desc.analysis_successful) {
      mark_terminated(0, ByteArray());
      return;
    }

    const TransferGraph& tg = desc.graph;

    if(!tg.ib_edges.empty()) {
      ib_offsets.resize(tg.ib_edges.size(), -1);

      // increase the count by one to prevent a trigger before we finish
      //  this loop
      ib_responses_needed.store(tg.ib_edges.size() + 1);

      // respect computed ib allocation order
      // TODO: attempt opportunistic unordered allocation

      // see who owns the first memory we need to allocate from
      NodeID first_owner = ID(tg.ib_edges[tg.ib_alloc_order[0]].memory).memory_owner_node();
      unsigned immed_count = 0;
      if(first_owner == Network::my_node_id) {
        // attempt immediate allocation of local IBs
        std::vector<size_t> sizes;
        std::vector<off_t> offsets;
        while(immed_count < tg.ib_edges.size()) {
          Memory tgt_mem = tg.ib_edges[tg.ib_alloc_order[immed_count]].memory;
          first_owner = ID(tgt_mem).memory_owner_node();
          // if we've gotten to IB requests that are non-local, stop
          if(first_owner != Network::my_node_id)
            break;
          sizes.assign(1, tg.ib_edges[tg.ib_alloc_order[immed_count]].size);
          unsigned same_mem = 1;
          while(((immed_count + same_mem) < tg.ib_edges.size()) &&
                (tgt_mem == tg.ib_edges[tg.ib_alloc_order[immed_count + same_mem]].memory)) {
            sizes.push_back(tg.ib_edges[tg.ib_alloc_order[immed_count + same_mem]].size);
            same_mem += 1;
          }

          offsets.assign(same_mem, -1);
          IBMemory *ib_mem = get_runtime()->get_ib_memory_impl(tgt_mem);
          if(ib_mem->attempt_immediate_allocation(Network::my_node_id,
                                                  reinterpret_cast<uintptr_t>(this),
                                                  same_mem, sizes.data(),
                                                  offsets.data())) {
            log_ib_alloc.debug() << "satisfied: op=" << Network::my_node_id
                                 << "/" << (void *)this
                                 << " index=" << immed_count << "+" << same_mem
                                 << " mem=" << tgt_mem;

            notify_ib_allocations(same_mem, immed_count, offsets.data());
            immed_count += same_mem;
          } else {
            // an immediate allocation failed, so it's time to enqueue
            break;
          }
        }
      }

      unsigned rem_count = tg.ib_edges.size() - immed_count;
      if(rem_count > 0) {
        if(first_owner == Network::my_node_id) {
          // enqueue all remaining requests with the first memory
          PendingIBRequests *reqs = new PendingIBRequests(Network::my_node_id,
                                                          reinterpret_cast<uintptr_t>(this),
                                                          rem_count,
                                                          immed_count, 0);
          for(unsigned i = 0; i < rem_count; i++) {
            reqs->memories.push_back(tg.ib_edges[tg.ib_alloc_order[immed_count + i]].memory);
            reqs->sizes.push_back(tg.ib_edges[tg.ib_alloc_order[immed_count + i]].size);
          }

          IBMemory *ib_mem = get_runtime()->get_ib_memory_impl(reqs->memories[0]);
          ib_mem->enqueue_requests(reqs);
        } else {
          // active message time - special case for rem_count == 1
          if(rem_count == 1) {
            ActiveMessage<RemoteIBAllocRequestSingle> amsg(first_owner);
            amsg->memory = tg.ib_edges[tg.ib_alloc_order[immed_count]].memory;
            amsg->size = tg.ib_edges[tg.ib_alloc_order[immed_count]].size;
            amsg->req_op = reinterpret_cast<uintptr_t>(this);
            amsg->req_index = immed_count;
            amsg->immediate = false;
            amsg.commit();
          } else {
            size_t bytes = (rem_count * (sizeof(Memory) + sizeof(size_t)));
            ActiveMessage<RemoteIBAllocRequestMultiple> amsg(first_owner, bytes);
            amsg->requestor = Network::my_node_id;
            amsg->count = rem_count;
            amsg->first_index = immed_count;
            amsg->curr_index = 0;
            amsg->req_op = reinterpret_cast<uintptr_t>(this);
            amsg->immediate = false;

            for(unsigned i = 0; i < rem_count; i++)
              amsg << tg.ib_edges[tg.ib_alloc_order[immed_count + i]].memory;
            for(unsigned i = 0; i < rem_count; i++)
              amsg << tg.ib_edges[tg.ib_alloc_order[immed_count + i]].size;

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

  std::ostream &operator<<(std::ostream &os, const TransferGraph::XDTemplate::IO &io)
  {
    switch(io.iotype) {
    case TransferGraph::XDTemplate::IO_INST:
      os << "inst(" << io.inst.inst << ":" << io.inst.inst.get_location().kind() << ","
         << io.inst.fld_start << "+" << io.inst.fld_count << ")";
      break;
    case TransferGraph::XDTemplate::IO_INDIRECT_INST:
      os << "ind(" << io.indirect.ind_idx << "," << io.indirect.port << ","
         << io.indirect.inst << ":" << io.indirect.inst.get_location().kind() << ","
         << io.indirect.fld_start << "+" << io.indirect.fld_count << ")";
      break;
    case TransferGraph::XDTemplate::IO_EDGE:
      os << "edge(" << io.edge << ")";
      break;
    case TransferGraph::XDTemplate::IO_FILL_DATA:
      os << "fill(" << io.fill.fill_start << "+" << io.fill.fill_size << ")";
      break;
    default:
      assert(0);
    }
    return os;
  }

  void TransferOperation::notify_ib_allocation(unsigned ib_index,
					       off_t ib_offset)
  {
    log_ib_alloc.info() << "notify: op=" << (void *)this
                        << " index=" << ib_index << "+1"
                        << " ok=" << ((ib_offset >= 0) ? 1 : 0);

    // translate alloc order back to original ib index
    ib_index = desc.graph.ib_alloc_order[ib_index];
    assert(ib_index < ib_offsets.size());
    // TODO: handle failed immediate allocation attempts
    assert(ib_offset >= 0);
#ifdef DEBUG_REALM
    assert(ib_offsets[ib_index] == -1);
#endif
    ib_offsets[ib_index] = ib_offset;

    // if this was the last response needed, we can continue on to creating xds
    if(ib_responses_needed.fetch_sub_acqrel(1) == 1)
      create_xds();
  }

  void TransferOperation::notify_ib_allocations(unsigned count,
                                                unsigned first_index,
                                                const off_t *offsets)
  {
    log_ib_alloc.info() << "notify: op=" << (void *)this
                        << " index=" << first_index << "+" << count
                        << " ok=" << ((offsets != 0) ? 1 : 0);

    assert((first_index + count) <= ib_offsets.size());
    // TODO: handle failed immediate allocation attempts
    assert(offsets);

    for(unsigned i = 0; i < count; i++) {
      // translate alloc order back to original ib index
      unsigned ib_index = desc.graph.ib_alloc_order[first_index + i];
#ifdef DEBUG_REALM
      assert(ib_offsets[ib_index] == -1);
#endif
      ib_offsets[ib_index] = offsets[i];
    }

    // if this was the last response needed, we can continue on to creating xds
    if(ib_responses_needed.fetch_sub_acqrel(count) == count)
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
      size_t fill_total = 0;

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
          ii.iter = desc.domain->create_iterator(xdn.inputs[j].inst.inst, desc.dim_order,
                                                 src_fields, src_offsets, src_sizes);
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
          ii.iter = gather_info->create_indirect_iterator(
              ii.mem, xdn.inputs[j].indirect.inst, src_fields, src_offsets, src_sizes,
              xdn.channel);
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
          fill_data =
              static_cast<const char *>(desc.fill_data) + xdn.inputs[j].fill.fill_start;
          fill_size = xdn.inputs[j].fill.fill_size;
          fill_total = xdn.inputs[j].fill.fill_total;
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
          oi.iter = desc.domain->create_iterator(xdn.outputs[j].inst.inst, desc.dim_order,
                                                 dst_fields, dst_offsets, dst_sizes);
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
            dst_offsets[k] =
                desc.dst_fields[xdn.outputs[j].indirect.fld_start + k].offset;
            dst_sizes[k] = desc.dst_fields[xdn.outputs[j].indirect.fld_start + k].size;
          }
          IndirectionInfo *scatter_info = desc.indirects[xdn.outputs[j].indirect.ind_idx];
          oi.iter = scatter_info->create_indirect_iterator(
              oi.mem, xdn.outputs[j].indirect.inst, dst_fields, dst_offsets, dst_sizes,
              xdn.channel);
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
				  fill_data, fill_size, fill_total);
      xd_factory->release();
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
    // TODO: respect time limit
    if(poisoned) {
      op->handle_poisoned_precondition(precondition);
    } else {
      // see if we need to wait for the transfer description analysis
      if(op->desc.request_analysis(op)) {
	// it's ready - go ahead and do ib creation
	op->allocate_ibs();
      } else {
	// do nothing - the TransferDesc will call us when it's ready
      }
    }
  }

  void TransferOperation::DeferredStart::print(std::ostream& os) const
  {
    os << "deferred_start(";
    op->print(os);
    os << ") finish=" << op->get_finish_event();
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
			      Event wait_on,
			      int priority) const
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
                                                  ID(ev).event_generation(),
                                                  priority);
    op->start_or_defer();

    // remove our reference to the description (op holds one)
    tdesc->remove_reference();

    return ev;
  }

#define DOIT(N, T)                                                                       \
  template Event IndexSpace<N, T>::copy(                                                 \
      const std::vector<CopySrcDstField> &, const std::vector<CopySrcDstField> &,        \
      const std::vector<const CopyIndirection<N, T>::Base *> &,                          \
      const ProfilingRequestSet &, Event, int) const;                                    \
  template class TransferIteratorIndexSpace<N, T>;                                       \
  template class TransferIteratorIndirect<N, T>;                                         \
  template class WrappingTransferIteratorIndirect<N, T>;                                 \
  template class TransferIteratorIndirectRange<N, T>;                                    \
  template class TransferDomainIndexSpace<N, T>;                                         \
  template class AddressSplitXferDesFactory<N, T>;
  FOREACH_NT(DOIT)

#define DOIT2(N,T,N2,T2) \
  template class IndirectionInfoTyped<N,T,N2,T2>; \
  template IndirectionInfo *CopyIndirection<N,T>::Unstructured<N2,T2>::create_info(const IndexSpace<N,T>& is) const;
  FOREACH_NTNT(DOIT2)

}; // namespace Realm
