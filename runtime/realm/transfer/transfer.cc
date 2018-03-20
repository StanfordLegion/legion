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
  // class TransferIteratorIndexSpace
  //

#if 0
  class TransferIteratorIndexSpace : public TransferIterator {
  protected:
    TransferIteratorIndexSpace(void); // used by deserializer
  public:
    TransferIteratorIndexSpace(const IndexSpace _is,
			       RegionInstance inst,
			       const std::vector<unsigned>& _fields,
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
    virtual void confirm_step(void);
    virtual void cancel_step(void);

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndexSpace> serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    IndexSpace is;
    const ElementMask *valid_mask;
    coord_t first_enabled;
    ElementMask::Enumerator *enumerator;
    coord_t rewind_pos;
    RegionInstanceImpl *inst_impl;
    const Mapping<1, 1> *mapping;
    std::vector<unsigned> field_offsets;
    std::vector<size_t> field_sizes;
    size_t field_idx, next_idx;
    size_t extra_elems;
    bool tentative_valid;
  };

  TransferIteratorIndexSpace::TransferIteratorIndexSpace(const IndexSpace _is,
							 RegionInstance inst,
							 const std::vector<unsigned>& _fields,
							 size_t _extra_elems)
    : is(_is), enumerator(0), field_idx(0), extra_elems(_extra_elems)
    , tentative_valid(false)
  {
    IndexSpaceImpl *idx_impl = get_runtime()->get_index_space_impl(is);
    // if the valid mask is available, grab it and pre-calculate the first
    //  enabled element, allowing us to optimize the empty-domain case
    if(idx_impl->request_valid_mask().has_triggered()) {
      valid_mask = idx_impl->valid_mask;
      assert(valid_mask != 0);
      first_enabled = valid_mask->find_enabled();
    } else {
      valid_mask = 0;
      first_enabled = 0; // this prevents the empty-space optimization below
    }
    // special case - an empty index space skips the rest of the init
    if((first_enabled == -1) || _fields.empty()) {
      // leave fields empty so that done() is always true
      inst_impl = 0;
      mapping = 0;
    } else {
      inst_impl = get_runtime()->get_instance_impl(inst);
      mapping = inst_impl->metadata.linearization_OLD.get_mapping<1>();
      for(std::vector<unsigned>::const_iterator it = _fields.begin();
	  it != _fields.end();
	  ++it) {
	// iterate over the fields in the instance to find the one we want
	size_t s = 0;
	assert(!inst_impl->metadata.field_sizes.empty());
	std::vector<size_t>::const_iterator it2 = inst_impl->metadata.field_sizes.begin();
	while(*it != s) {
	  s += *it2;
	  ++it2;
	  assert(it2 != inst_impl->metadata.field_sizes.end());
	}
	field_offsets.push_back(*it);
	field_sizes.push_back(*it2);
      }
    }
  }

  TransferIteratorIndexSpace::TransferIteratorIndexSpace(void)
    : enumerator(0), field_idx(0)
    , tentative_valid(false)
  {}

  template <typename S>
  /*static*/ TransferIterator *TransferIteratorIndexSpace::deserialize_new(S& deserializer)
  {
    IndexSpace is;
    RegionInstance inst;
    std::vector<unsigned> field_offsets;
    std::vector<size_t> field_sizes;
    size_t extra_elems;

    if(!((deserializer >> is) &&
	 (deserializer >> inst) &&
	 (deserializer >> field_offsets) &&
	 (deserializer >> field_sizes) &&
	 (deserializer >> extra_elems)))
      return 0;

    TransferIteratorIndexSpace *tiis = new TransferIteratorIndexSpace;

    tiis->is = is;
    tiis->field_offsets.swap(field_offsets);
    tiis->field_sizes.swap(field_sizes);
    tiis->extra_elems = extra_elems;

    // if the valid mask is available, grab it and pre-calculate the first
    //  enabled element, allowing us to optimize the empty-domain case
    // also, don't even request the valid mask if there are no fields - there's
    //  nothing to copy anyway
    if(!tiis->field_sizes.empty()) {
      IndexSpaceImpl *idx_impl = get_runtime()->get_index_space_impl(is);
      if(idx_impl->request_valid_mask().has_triggered()) {
	tiis->valid_mask = idx_impl->valid_mask;
	assert(tiis->valid_mask != 0);
	tiis->first_enabled = tiis->valid_mask->find_enabled();
	if(tiis->first_enabled == -1) {
	  tiis->field_offsets.clear();
	  tiis->field_sizes.clear();
	}
      } else {
	tiis->valid_mask = 0;
	tiis->first_enabled = 0;
      }

      if(inst.exists()) {
	tiis->inst_impl = get_runtime()->get_instance_impl(inst);
	tiis->mapping = tiis->inst_impl->metadata.linearization_OLD.get_mapping<1>();
	assert(tiis->mapping != 0);
      } else {
	tiis->inst_impl = 0;
	tiis->mapping = 0;
      }
    } else {
      tiis->valid_mask = 0;
      tiis->first_enabled = 0;

      tiis->inst_impl = 0;
      tiis->mapping = 0;
    }

    return tiis;
  }

  TransferIteratorIndexSpace::~TransferIteratorIndexSpace(void)
  {
    delete enumerator;
  }

  Event TransferIteratorIndexSpace::request_metadata(void)
  {
    IndexSpaceImpl *idx_impl = get_runtime()->get_index_space_impl(is);
    return idx_impl->request_valid_mask();
  }

  void TransferIteratorIndexSpace::reset(void)
  {
    field_idx = 0;
    delete enumerator;
    enumerator = 0;
  }

  bool TransferIteratorIndexSpace::done(void)
  {
    return(field_idx == field_offsets.size());
  }

  size_t TransferIteratorIndexSpace::step(size_t max_bytes, AddressInfo& info,
					  unsigned flags,
					  bool tentative /*= false*/)
  {
    // if we don't have the valid mask (and we have any fields that we intend
    //  to iterate over), grab it now (it must be ready, as we aren't allowed
    //  to wait here), and detect empty case
    if(!field_sizes.empty() && !valid_mask) {
      IndexSpaceImpl *idx_impl = get_runtime()->get_index_space_impl(is);
      Event ready = idx_impl->request_valid_mask();
      assert(ready.has_triggered());
      valid_mask = idx_impl->valid_mask;
      assert(valid_mask != 0);
      first_enabled = valid_mask->find_enabled();
      if(first_enabled == -1) {
	// clear out fields so that done() will return true, even after reset
	field_idx = 0;
	field_offsets.clear();
	field_sizes.clear();
	return 0;
      }
    }
    assert(!done());
    assert(!tentative_valid);

    size_t max_elems = max_bytes / field_sizes[field_idx];
    // less than one element?  give up immediately
    if(max_elems == 0)
      return 0;

    // build an enumerator if we don't have one
    if(!enumerator)
      enumerator = valid_mask->enumerate_enabled(first_enabled);

    coord_t span_start;
    size_t span_len;
    bool ok = enumerator->get_next(span_start, span_len);
    assert(ok);

    // if this step is cancelled, we'll back up to span_start (no
    //  need to re-scan the 0's before span_start)
    rewind_pos = span_start;

    bool last_span;
    if(span_len > max_elems) {
      // we're only going to take part of this span this time,
      //   but that means we know we're not the last span
      span_len = max_elems;
      enumerator->set_pos(span_start + span_len);
      last_span = false;
    } else {
      // peek ahead to see if we're done, and maybe merge spans if
      //  extra_elems > 0
      while(1) {
	coord_t peek_start;
	size_t peek_len;
	bool peek_ok = enumerator->peek_next(peek_start, peek_len);
	if(peek_ok) {
	  // to merge, the gap needs to be <= extra_elems and the total size
	  //  needs to be <= max_elems
	  if(((peek_start - (span_start + span_len)) <= extra_elems) &&
	     ((peek_start + peek_len - span_start)) <= max_elems) {
	    // merge and keep going
	    span_len = (peek_start - span_start) + peek_len;
	  } else {
	    // no merge, but this isn't the last span
	    last_span = false;
	    break;
	  }
	} else {
	  // all done
	  last_span = true;
	  break;
	}
      }
    }

    // map the target rectangle, and (for now) assume it works
    Rect<1> target_subrect(span_start,
			   (coord_t)(span_start + span_len - 1));
    Rect<1> act_subrect;
    Rect<1> image = mapping->image_dense_subrect(target_subrect, act_subrect);
    assert(act_subrect == target_subrect);

    coord_t first_block;
    coord_t block_ofs;
    if(inst_impl->metadata.block_size > (size_t)image.hi[0]) {
      // SOA is always a single block
      first_block = 0;
      block_ofs = image.lo[0];
    } else {
      // handle AOS/hybrid cases
      // see which block the start and end are in
      first_block = image.lo[0] / inst_impl->metadata.block_size;
      block_ofs = image.lo[0] - (first_block *
				 inst_impl->metadata.block_size);
      coord_t last_block = image.hi[0] / inst_impl->metadata.block_size;
      if(first_block != last_block) {
	// shorten span to remain contiguous
	span_len = (inst_impl->metadata.block_size -
		    (image.lo[0] % inst_impl->metadata.block_size));
	enumerator->set_pos(image.lo[0] + span_len);
	last_span = false;
      }
    }

    // fill in address info
    info.base_offset = (inst_impl->metadata.alloc_offset +
			(first_block * inst_impl->metadata.block_size * inst_impl->metadata.elmt_size) +
			(field_offsets[field_idx] * inst_impl->metadata.block_size) +
			(block_ofs * field_sizes[field_idx]));
    info.bytes_per_chunk = span_len * field_sizes[field_idx];
    info.num_lines = 1;
    info.line_stride = 0;
    info.num_planes = 1;
    info.plane_stride = 0;

    if(tentative) {
      tentative_valid = true;
      next_idx = field_idx + (last_span ? 1 : 0);
    } else {
      if(last_span) {
	delete enumerator;
	enumerator = 0;
	field_idx++;
      }
    }

    return info.bytes_per_chunk;
  }

  void TransferIteratorIndexSpace::confirm_step(void)
  {
    assert(tentative_valid);
    if(next_idx != field_idx) {
      delete enumerator;
      enumerator = 0;
      field_idx = next_idx;
    }
    tentative_valid = false;
  }

  void TransferIteratorIndexSpace::cancel_step(void)
  {
    assert(tentative_valid);
    assert(enumerator != 0);
    enumerator->set_pos(rewind_pos);
    tentative_valid = false;
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorIndexSpace> TransferIteratorIndexSpace::serdez_subclass;

  template <typename S>
  bool TransferIteratorIndexSpace::serialize(S& serializer) const
  {
    return ((serializer << is) &&
	    (serializer << (inst_impl ? inst_impl->me :
                                        RegionInstance::NO_INST)) &&
	    (serializer << field_offsets) &&
	    (serializer << field_sizes) &&
	    (serializer << extra_elems));
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIteratorRect<DIM>
  //

#if 0
  template <unsigned DIM>
  class TransferIteratorRect : public TransferIterator {
  protected:
    TransferIteratorRect(void); // used by deserializer
  public:
    TransferIteratorRect(const Rect<DIM>& _r,
			 RegionInstance inst,
			 const std::vector<unsigned>& _fields);

    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);
      
    virtual void reset(void);
    virtual bool done(void);
    virtual size_t step(size_t max_bytes, AddressInfo& info,
			unsigned flags,
			bool tentative = false);
    virtual void confirm_step(void);
    virtual void cancel_step(void);

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorRect<DIM> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    Rect<DIM> r;
    Point<DIM> p, next_p;
    RegionInstanceImpl *inst_impl;
    const Mapping<DIM, 1> *mapping;
    std::vector<unsigned> field_offsets;
    std::vector<size_t> field_sizes;
    size_t field_idx, next_idx;
    bool tentative_valid;
  };

  template <unsigned DIM>
  TransferIteratorRect<DIM>::TransferIteratorRect(const Rect<DIM>& _r,
						  RegionInstance _inst,
						  const std::vector<unsigned>& _fields)
    : r(_r), p(_r.lo), field_idx(0), tentative_valid(false)
  {
    // special case - an empty rectangle (or field list) skips most initialization
    if((r.volume() == 0) || _fields.empty()) {
      inst_impl = 0;
      mapping = 0;
      // leave fields empty so that done() is always true
    } else {
      inst_impl = get_runtime()->get_instance_impl(_inst);
      mapping = inst_impl->metadata.linearization_OLD.get_mapping<DIM>();
      for(std::vector<unsigned>::const_iterator it = _fields.begin();
	  it != _fields.end();
	  ++it) {
	// iterate over the fields in the instance to find the one we want
	size_t s = 0;
	assert(!inst_impl->metadata.field_sizes.empty());
	std::vector<size_t>::const_iterator it2 = inst_impl->metadata.field_sizes.begin();
	while(*it != s) {
	  s += *it2;
	  ++it2;
	  assert(it2 != inst_impl->metadata.field_sizes.end());
	}
	field_offsets.push_back(*it);
	field_sizes.push_back(*it2);
      }
    }
  }

  template <unsigned DIM>
  TransferIteratorRect<DIM>::TransferIteratorRect(void)
    : field_idx(0)
    , tentative_valid(false)
  {}

  template <unsigned DIM>
  template <typename S>
  /*static*/ TransferIterator *TransferIteratorRect<DIM>::deserialize_new(S& deserializer)
  {
    Rect<DIM> r;
    RegionInstance inst;
    std::vector<unsigned> field_offsets;
    std::vector<size_t> field_sizes;

    if(!((deserializer >> r) &&
	 (deserializer >> inst) &&
	 (deserializer >> field_offsets) &&
	 (deserializer >> field_sizes)))
      return 0;

    TransferIteratorRect<DIM> *tir = new TransferIteratorRect<DIM>;

    tir->r = r;
    tir->field_offsets.swap(field_offsets);
    tir->field_sizes.swap(field_sizes);

    tir->p = r.lo;

    // if there are no fields, don't even bother with the instance info
    if(tir->field_sizes.empty()) {
      tir->inst_impl = 0;
      tir->mapping = 0;
    } else {
      if(inst.exists()) {
	tir->inst_impl = get_runtime()->get_instance_impl(inst);
	tir->mapping = tir->inst_impl->metadata.linearization_OLD.template get_mapping<DIM>();
	assert(tir->mapping != 0);
      } else {
	// clear out fields too since there's no instance data
	tir->inst_impl = 0;
	tir->mapping = 0;
	tir->field_offsets.clear();
	tir->field_sizes.clear();
      }
    }

    return tir;
  }

  template <unsigned DIM>
  void TransferIteratorRect<DIM>::reset(void)
  {
    field_idx = 0;
    p = r.lo;
    tentative_valid = false;
  }

  template <unsigned DIM>
  bool TransferIteratorRect<DIM>::done(void)
  {
    return(field_idx == field_offsets.size());
  }

  template <unsigned DIM>
  size_t TransferIteratorRect<DIM>::step(size_t max_bytes, AddressInfo& info,
					 unsigned flags,
					 bool tentative /*= false*/)
  {
    assert(!done());
    assert(!tentative_valid);

    size_t max_elems = max_bytes / field_sizes[field_idx];
    // less than one element?  give up immediately
    if(max_elems == 0)
      return 0;

    // std::cout << "step " << this << " " << r << " " << p << " " << field_idx
    // 	      << " " << max_bytes << ":";

    // using the current point, find the biggest subrectangle we want to try
    //  giving out
    Rect<DIM> target_subrect;
    target_subrect.lo = p;
    bool grow = true;
    size_t count = 1;
    for(unsigned d = 0; d < DIM; d++) {
      if(grow) {
	size_t len = r.hi[d] - p[d] + 1;
	if((count * len) <= max_elems) {
	  // full growth in that direction
	  target_subrect.hi.x[d] = r.hi[d];
	  count *= len;
	  // once we get to a dimension we've only partially handled, we can't
	  //  grow in any subsequent ones
	  if(p[d] != r.lo[d])
	    grow = false;
	} else {
	  size_t actlen = max_elems / count;
	  assert(actlen >= 1);
	  // take what we can, and don't try to grow other dimensions
	  target_subrect.hi.x[d] = p[d] + actlen - 1;
	  count *= actlen;
	  grow = false;
	}
      } else
	target_subrect.hi.x[d] = p[d];
    }

    // request a linear subrectangle and then knock out dimensions that are not
    //  contiguous (allowing for 2D/3D copies if requested)
    Rect<DIM> act_subrect;
    Point<1> strides[DIM];
    Point<1> image_lo = mapping->image_linear_subrect(target_subrect,
						      act_subrect,
						      strides);
    int dims_left = (((flags & LINES_OK) == 0)  ? 1 :
		     ((flags & PLANES_OK) == 0) ? 2 :
		                                  3);
    // these are strides in the mapping, which count in units of elements,
    //  not bytes
    coord_t exp_stride = 1;
    for(unsigned d = 0; d < DIM; d++) {
      // if the stride does not what we want, it can't be merged and costs us
      //  one of our copy dimensions to change it
      if((dims_left > 0) && ((coord_t)strides[d] != exp_stride)) {
	exp_stride = strides[d];
	dims_left--;
      }
      // if we're still merging, take the full subrect and update the expected
      //  stride - if not, collapse this dimension of the act_subrect
      if(dims_left > 0) {
	exp_stride *= (act_subrect.hi[d] - act_subrect.lo[d] + 1);
      } else {
	act_subrect.hi.x[d] = act_subrect.lo[d];
      }
    }
    // get the high bounds of our image (which might not be contiguous)
    Point<1> image_hi = mapping->image(act_subrect.hi);
    size_t act_count = act_subrect.volume();
    
    coord_t first_block;
    coord_t block_ofs;
    if(inst_impl->metadata.block_size > (size_t)image_hi[0]) {
      // SOA is always a single block
      first_block = 0;
      block_ofs = image_lo[0];
    } else {
      // handle AOS/hybrid cases
      // see which block the start and end are in
      first_block = image_lo[0] / inst_impl->metadata.block_size;
      block_ofs = image_lo[0] - (first_block *
				 inst_impl->metadata.block_size);
      coord_t last_block = image_hi[0] / inst_impl->metadata.block_size;
      if(first_block != last_block) {
	// shrink rectangle to remain contiguous
	coord_t max_len = (inst_impl->metadata.block_size -
			   (image_lo[0] % inst_impl->metadata.block_size));
	if(max_len < (act_subrect.hi[0] - act_subrect.lo[0] + 1)) {
	  // can't even get the first dimension done - collapse everything else
	  act_subrect.hi = act_subrect.lo;
	  act_subrect.hi.x[0] = act_subrect.lo[0] + max_len - 1;
	  act_count = max_len;
	} else {
	  // got the whole first dimension - keep track of how much we have
	  //  left for subsequent dimensions
	  max_len -= (act_subrect.hi[0] - act_subrect.lo[0] + 1);
	  for(unsigned d = 1; d < DIM; d++) {
	    // don't spend time on degenerate domains
	    if(act_subrect.lo[d] == act_subrect.hi[d]) continue;

	    // does this whole dimension fit?
	    // we already have room for the first subspace in this dimension, so
	    //  see how many more we can add (no +1 here because we want 1 less
	    //  than the total extent)
	    coord_t needed_len = ((coord_t)strides[d] *
				  (act_subrect.hi[d] - act_subrect.lo[d]));
	    if(needed_len <= max_len) {
	      max_len -= needed_len;
	      continue;
	    }

	    // nope, have to shorten it and then collapse higher dimensions
	    coord_t avail_len = max_len / strides[d];
	    act_subrect.hi.x[d] = act_subrect.lo[d] + avail_len;
	    while(++d < DIM)
	      act_subrect.hi.x[d] = act_subrect.lo[d];
	    break;
	  }
	  // update our count
	  act_count = act_subrect.volume();
	}
      }
    }

    // fill in address info
    info.base_offset = (inst_impl->metadata.alloc_offset +
			(first_block * inst_impl->metadata.block_size * inst_impl->metadata.elmt_size) +
			(field_offsets[field_idx] * inst_impl->metadata.block_size) +
			(block_ofs * field_sizes[field_idx]));
    // redo dimension collapsing now that we know exactly what rectangle we've
    //  selected, and use bytes as the unit now
    coord_t act_counts[3], act_strides[3];
    int cur_dim = 0;
    act_counts[0] = field_sizes[field_idx];
    act_strides[0] = 1;
    for(unsigned d = 0; d < DIM; d++) {
      // only worry about non-degenerate dimensions
      if(act_subrect.lo[d] == act_subrect.hi[d])
	continue;
      coord_t exp_stride = (coord_t)strides[d] * field_sizes[field_idx];
      if(exp_stride != (act_strides[cur_dim] * act_counts[cur_dim])) {
	cur_dim++;
	assert(cur_dim < 3);
	act_counts[cur_dim] = 1;
	act_strides[cur_dim] = exp_stride;
      }
      act_counts[cur_dim] *= (act_subrect.hi[d] - act_subrect.lo[d] + 1);
    }
    info.bytes_per_chunk = act_counts[0];
    if(cur_dim >= 1) {
      assert((flags & LINES_OK) != 0);
      info.num_lines = act_counts[1];
      info.line_stride = act_strides[1];
    } else {
      info.num_lines = 1;
      info.line_stride = 0;
    }
    if(cur_dim >= 2) {
      assert((flags & PLANES_OK) != 0);
      info.num_planes = act_counts[2];
      info.plane_stride = act_strides[2];
    } else {
      info.num_planes = 1;
      info.plane_stride = 0;
    }

    size_t total_bytes = act_count * field_sizes[field_idx];
    assert(total_bytes == (info.bytes_per_chunk * info.num_lines * info.num_planes));

    // now set 'next_p' to the next point we want
    bool carry = true;
    for(unsigned d = 0; d < DIM; d++) {
      if(carry) {
	if(act_subrect.hi[d] == r.hi[d]) {
	  next_p.x[d] = r.lo[d];
	} else {
	  next_p.x[d] = act_subrect.hi[d] + 1;
	  carry = false;
	}
      } else
	next_p.x[d] = act_subrect.lo[d];
    }
    // if the "carry" propagated all the way through, go on to the next field
    next_idx = field_idx + (carry ? 1 : 0);

    //std::cout << " " << act_subrect << " " << next_p << " " << next_idx << " " << info.bytes_per_chunk << "\n";

    if(tentative) {
      tentative_valid = true;
    } else {
      p = next_p;
      field_idx = next_idx;
    }

    return total_bytes;
  }

  template <unsigned DIM>
  void TransferIteratorRect<DIM>::confirm_step(void)
  {
    assert(tentative_valid);
    p = next_p;
    field_idx = next_idx;
    tentative_valid = false;
  }

  template <unsigned DIM>
  void TransferIteratorRect<DIM>::cancel_step(void)
  {
    assert(tentative_valid);
    tentative_valid = false;
  }

  template <unsigned DIM>
  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, TransferIteratorRect<DIM> > TransferIteratorRect<DIM>::serdez_subclass;

  template <unsigned DIM>
  template <typename S>
  bool TransferIteratorRect<DIM>::serialize(S& serializer) const
  {
    return ((serializer << r) &&
	    (serializer << (inst_impl ? inst_impl->me :
                                        RegionInstance::NO_INST)) &&
	    (serializer << field_offsets) &&
	    (serializer << field_sizes));
  }

  template class TransferIteratorRect<1>;
  template class TransferIteratorRect<2>;
  template class TransferIteratorRect<3>;
#endif


#ifdef USE_HDF_OLD
  template <unsigned DIM>
  class TransferIteratorHDF5 : public TransferIterator {
  public:
    TransferIteratorHDF5(const Rect<DIM>& _r,
			 RegionInstance inst,
			 const std::vector<unsigned>& _fields);

    // NOTE: TransferIteratorHDF5 does NOT support serialization - you
    //  can't move away from the process that called H5Fopen

    virtual void reset(void);
    virtual bool done(void);
    virtual size_t step(size_t max_bytes, AddressInfo& info,
			unsigned flags,
			bool tentative = false);
    virtual size_t step(size_t max_bytes, AddressInfoHDF5& info,
			bool tentative = false);
    virtual void confirm_step(void);
    virtual void cancel_step(void);

  protected:
    Rect<DIM> r;
    Point<DIM> p, next_p;
    std::vector<hid_t> dset_ids, dtype_ids;
    const HDF5::HDF5Memory::HDFMetadata *hdf_metadata;
    size_t field_idx, next_idx;
    bool tentative_valid;
  };

  template <unsigned DIM>
  TransferIteratorHDF5<DIM>::TransferIteratorHDF5(const Rect<DIM>& _r,
						  RegionInstance inst,
						  const std::vector<unsigned>& _fields)
    : r(_r), p(_r.lo), field_idx(0), tentative_valid(false)
  {
    MemoryImpl *mem_impl = get_runtime()->get_memory_impl(inst);
    assert(mem_impl->kind == MemoryImpl::MKIND_HDF);
    HDF5::HDF5Memory *hdf5mem = (HDF5::HDF5Memory *)mem_impl;
    std::map<RegionInstance, HDF5::HDF5Memory::HDFMetadata *>::const_iterator it = hdf5mem->hdf_metadata.find(inst);
    assert(it != hdf5mem->hdf_metadata.end());
    hdf_metadata = it->second;
    assert(hdf_metadata->ndims == DIM);

    // look up dataset and datatype ids for all fields
    for(std::vector<unsigned>::const_iterator it = _fields.begin();
	it != _fields.end();
	++it) {
      std::map<size_t, hid_t>::const_iterator it2;

      it2 = hdf_metadata->dataset_ids.find(*it);
      assert(it2 != hdf_metadata->dataset_ids.end());
      dset_ids.push_back(it2->second);

      it2 = hdf_metadata->datatype_ids.find(*it);
      assert(it2 != hdf_metadata->datatype_ids.end());
      dtype_ids.push_back(it2->second);
    }
  }

  template <unsigned DIM>
  void TransferIteratorHDF5<DIM>::reset(void)
  {
    p = r.lo;
    field_idx = 0;
  }

  template <unsigned DIM>
  bool TransferIteratorHDF5<DIM>::done(void)
  {
    return (field_idx == dset_ids.size());
  }
  
  template <unsigned DIM>
  size_t TransferIteratorHDF5<DIM>::step(size_t max_bytes, AddressInfo& info,
					 unsigned flags,
					 bool tentative /*= false*/)
  {
    // normal address infos not allowed
    return 0;
  }

  template <unsigned DIM>
  size_t TransferIteratorHDF5<DIM>::step(size_t max_bytes, AddressInfoHDF5& info,
					 bool tentative /*= false*/)
  {
    assert(!done());
    assert(!tentative_valid);

    info.dset_id = dset_ids[field_idx];
    info.dtype_id = dtype_ids[field_idx];

    // convert max_bytes into desired number of elements
    size_t elmt_size = H5Tget_size(info.dtype_id);
    size_t max_elems = max_bytes / elmt_size;
    if(max_elems == 0)
      return 0;

    // std::cout << "step " << this << " " << r << " " << p << " " << field_idx
    // 	      << " " << max_bytes << ":";

    // HDF5 requires we handle dimensions in order - no permutation allowed
    // using the current point, find the biggest subrectangle we want to try
    //  giving out
    Rect<DIM> target_subrect;
    target_subrect.lo = p;
    bool grow = true;
    size_t count = 1;
    for(unsigned d = 0; d < DIM; d++) {
      if(grow) {
	size_t len = r.hi[d] - p[d] + 1;
	if((count * len) <= max_elems) {
	  // full growth in that direction
	  target_subrect.hi.x[d] = r.hi[d];
	  count *= len;
	  // once we get to a dimension we've only partially handled, we can't
	  //  grow in any subsequent ones
	  if(p[d] != r.lo[d])
	    grow = false;
	} else {
	  size_t actlen = max_elems / count;
	  assert(actlen >= 1);
	  // take what we can, and don't try to grow other dimensions
	  target_subrect.hi.x[d] = p[d] + actlen - 1;
	  count *= actlen;
	  grow = false;
	}
      } else
	target_subrect.hi.x[d] = p[d];
    }

    // translate the target_subrect into the dataset's coordinates
    // HDF5 uses C-style (row-major) ordering, so invert array indices
    info.dset_bounds.resize(DIM);
    info.offset.resize(DIM);
    info.extent.resize(DIM);
    for(unsigned d = 0; d < DIM; d++) {
      assert(target_subrect.lo[d] >= hdf_metadata->lo[d]);
      info.offset[DIM - 1 - d] = (target_subrect.lo[d] - hdf_metadata->lo[d]);
      info.extent[DIM - 1 - d] = (target_subrect.hi[d] - target_subrect.lo[d] + 1);
      assert(info.extent[DIM - 1 - d] <= hdf_metadata->dims[d]);
      info.dset_bounds[DIM - 1 - d] = hdf_metadata->dims[d];
    }

    // now set 'next_p' to the next point we want
    bool carry = true;
    for(unsigned d = 0; d < DIM; d++) {
      if(carry) {
	if(target_subrect.hi[d] == r.hi[d]) {
	  next_p.x[d] = r.lo[d];
	} else {
	  next_p.x[d] = target_subrect.hi[d] + 1;
	  carry = false;
	}
      } else
	next_p.x[d] = target_subrect.lo[d];
    }
    // if the "carry" propagated all the way through, go on to the next field
    next_idx = field_idx + (carry ? 1 : 0);

    size_t act_bytes = count * elmt_size;

    //std::cout << " " << target_subrect << " " << next_p << " " << next_idx << " " << act_bytes << "\n";

    if(tentative) {
      tentative_valid = true;
    } else {
      p = next_p;
      field_idx = next_idx;
    }

    return act_bytes;
  }

  template <unsigned DIM>
  void TransferIteratorHDF5<DIM>::confirm_step(void)
  {
    assert(tentative_valid);
    p = next_p;
    field_idx = next_idx;
    tentative_valid = false;
  }

  template <unsigned DIM>
  void TransferIteratorHDF5<DIM>::cancel_step(void)
  {
    assert(tentative_valid);
    tentative_valid = false;
  }
#endif // USE_HDF


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
  };

  template <int N, typename T>
  TransferIteratorIndexSpace<N,T>::TransferIteratorIndexSpace(const IndexSpace<N,T>& _is,
								RegionInstance inst,
								const std::vector<FieldID>& _fields,
								size_t _extra_elems)
    : is(_is), field_idx(0), extra_elems(_extra_elems), tentative_valid(false)
  {
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
      inst_layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
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

    if(!((deserializer >> is) &&
	 (deserializer >> inst) &&
	 (deserializer >> fields) &&
	 (deserializer >> extra_elems)))
      return 0;

    TransferIteratorIndexSpace<N,T> *tiis = new TransferIteratorIndexSpace<N,T>(is,
										inst,
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
      for(int d = 0; d < N; d++) {
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
    for(int d = 0; d < N; d++) {
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
      for(int d = 0; d < N; d++) {
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
	} else
	  target_subrect.hi[d] = cur_point[d];
      }

      // translate the target_subrect into the dataset's coordinates
      // HDF5 uses C-style (row-major) ordering, so invert array indices
      info.dset_bounds.resize(N);
      info.offset.resize(N);
      info.extent.resize(N);
      for(unsigned d = 0; d < N; d++) {
	info.offset[N - 1 - d] = (target_subrect.lo[d] - hlp->bounds.lo[d] + hlp->offset[d]);
	info.extent[N - 1 - d] = (target_subrect.hi[d] - target_subrect.lo[d] + 1);
	info.dset_bounds[N - 1 - d] = (hlp->offset[d] +
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
    for(int d = 0; d < N; d++) {
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
    return ((serializer << iter.space) &&
	    (serializer << (inst_impl ? inst_impl->me :
                                        RegionInstance::NO_INST)) &&
	    (serializer << fields) &&
	    (serializer << extra_elems));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDomain
  //

  TransferDomain::TransferDomain(void)
  {}

  TransferDomain::~TransferDomain(void)
  {}

#if 0
  class TransferDomainIndexSpace : public TransferDomain {
  public:
    TransferDomainIndexSpace(IndexSpace _is);

    template <typename S>
    static TransferDomain *deserialize_new(S& deserializer);

    virtual TransferDomain *clone(void) const;

    virtual Event request_metadata(void);

    virtual size_t volume(void) const;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      RegionInstance peer,
					      const std::vector<FieldID>& fields) const;

    virtual void print(std::ostream& os) const;

    static Serialization::PolymorphicSerdezSubclass<TransferDomain, TransferDomainIndexSpace> serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

    //protected:
    IndexSpace is;
  };

  TransferDomainIndexSpace::TransferDomainIndexSpace(IndexSpace _is)
    : is(_is)
  {}

  template <typename S>
  /*static*/ TransferDomain *TransferDomainIndexSpace::deserialize_new(S& deserializer)
  {
    IndexSpace is;
    if(deserializer >> is)
      return new TransferDomainIndexSpace(is);
    else
      return 0;
  }

  TransferDomain *TransferDomainIndexSpace::clone(void) const
  {
    return new TransferDomainIndexSpace(is);
  }

  Event TransferDomainIndexSpace::request_metadata(void)
  {
    IndexSpaceImpl *is_impl = get_runtime()->get_index_space_impl(is);
    if(!is_impl->locked_data.valid) {
      log_dma.debug("dma request %p - no index space metadata yet", this);

      Event e = is_impl->lock.acquire(1, false, ReservationImpl::ACQUIRE_BLOCKING);
      if(e.has_triggered()) {
	log_dma.debug("request %p - index space metadata invalid - instant trigger", this);
	is_impl->lock.release();
      } else {
	log_dma.debug("request %p - index space metadata invalid - sleeping on lock " IDFMT "", this, is_impl->lock.me.id);
	is_impl->lock.me.release(e);
	return e;
      }
    }

    // we will need more than just the metadata - we also need the valid mask
    {
      Event e = is_impl->request_valid_mask();
      if(!e.has_triggered()) {
	log_dma.debug() << "request " << (void *)this << " - valid mask needed for index space "
			<< is << " - sleeping on event " << e;
	return e;
      }
    }

    return Event::NO_EVENT;
  }

  size_t TransferDomainIndexSpace::volume(void) const
  {
    //return is.get_volume();
    return is.get_valid_mask().pop_count();
  }

  TransferIterator *TransferDomainIndexSpace::create_iterator(RegionInstance inst,
							      RegionInstance peer,
							      const std::vector<FieldID>& fields) const
  {
    size_t extra_elems = 0;
    return new TransferIteratorIndexSpace(is, inst, fields, extra_elems);
  }
  
  void TransferDomainIndexSpace::print(std::ostream& os) const
  {
    os << is;
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferDomain, TransferDomainIndexSpace> TransferDomainIndexSpace::serdez_subclass;

  template <typename S>
  inline bool TransferDomainIndexSpace::serialize(S& serializer) const
  {
    return (serializer << is);
  }
#endif


#if 0
  template <unsigned DIM>
  class TransferDomainRect : public TransferDomain {
  public:
    TransferDomainRect(LegionRuntime::Arrays::Rect<DIM> _r);

    template <typename S>
    static TransferDomain *deserialize_new(S& deserializer);

    virtual TransferDomain *clone(void) const;

    virtual Event request_metadata(void);

    virtual size_t volume(void) const;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      RegionInstance peer,
					      const std::vector<FieldID>& fields) const;

    virtual void print(std::ostream& os) const;

    static Serialization::PolymorphicSerdezSubclass<TransferDomain, TransferDomainRect<DIM> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

    //protected:
    LegionRuntime::Arrays::Rect<DIM> r;
  };

  template <unsigned DIM>
  TransferDomainRect<DIM>::TransferDomainRect(LegionRuntime::Arrays::Rect<DIM> _r)
    : r(_r)
  {}

  template <unsigned DIM>
  template <typename S>
  /*static*/ TransferDomain *TransferDomainRect<DIM>::deserialize_new(S& deserializer)
  {
    Rect<DIM> r;
    if(deserializer >> r)
      return new TransferDomainRect<DIM>(r);
    else
      return 0;
  }

  template <unsigned DIM>
  TransferDomain *TransferDomainRect<DIM>::clone(void) const
  {
    return new TransferDomainRect<DIM>(r);
  }

  template <unsigned DIM>
  Event TransferDomainRect<DIM>::request_metadata(void)
  {
    // nothing to request
    return Event::NO_EVENT;
  }

  template <unsigned DIM>
  size_t TransferDomainRect<DIM>::volume(void) const
  {
    return r.volume();
  }

  template <unsigned DIM>
  TransferIterator *TransferDomainRect<DIM>::create_iterator(RegionInstance inst,
							      RegionInstance peer,
							      const std::vector<FieldID>& fields) const
  {
#ifdef USE_HDF_OLD
    // HDF5 memories need special iterators
    if(inst.get_location().kind() == Memory::HDF_MEM)
      return new TransferIteratorHDF5<DIM>(r, inst, fields);
#endif

    return new TransferIteratorRect<DIM>(r, inst, fields);
  }
  
  template <unsigned DIM>
  void TransferDomainRect<DIM>::print(std::ostream& os) const
  {
    os << r;
  }

  template <unsigned DIM>
  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferDomain, TransferDomainRect<DIM> > TransferDomainRect<DIM>::serdez_subclass;

  template <unsigned DIM>
  template <typename S>
  inline bool TransferDomainRect<DIM>::serialize(S& serializer) const
  {
    return (serializer << r);
  }

  template class TransferDomainRect<1>;
  template class TransferDomainRect<2>;
  template class TransferDomainRect<3>;

  /*static*/ TransferDomain *TransferDomain::construct(Domain d)
  {
    switch(d.get_dim()) {
    case 0: return new TransferDomainIndexSpace(d.get_index_space());
    case 1: return new TransferDomainRect<1>(d.get_rect<1>());
    case 2: return new TransferDomainRect<2>(d.get_rect<2>());
    case 3: return new TransferDomainRect<3>(d.get_rect<3>());
    }
    assert(0);
    return 0;
  }
#endif


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
#ifdef USE_HDF_OLD
    // HDF5 memories need special iterators
    if(inst.get_location().kind() == Memory::HDF_MEM) {
      assert(0);
      return 0;
    }
#endif

    size_t extra_elems = 0;
    return new TransferIteratorIndexSpace<N,T>(is, inst, fields, extra_elems);
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
    NodeID src_node = ID(src_mem).memory.owner_node;
    NodeID dst_node = ID(dst_mem).memory.owner_node;

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
    Event ev = GenEventImpl::create_genevent()->current_event();

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
				     wait_on, ev, priority, requests);
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
#if 0
      RemoteCopyArgs args;
      args.redop_id = 0;
      args.red_fold = false;
      args.before_copy = wait_on;
      args.after_copy = ev;
      args.priority = priority;

      size_t msglen = r->compute_size();
      void *msgdata = malloc(msglen);

      r->serialize(msgdata);

      log_dma.debug("performing copy on remote node (%d), event=" IDFMT, dma_node, args.after_copy.id);
      get_runtime()->optable.add_remote_operation(ev, dma_node);
      RemoteCopyMessage::request(dma_node, args, msgdata, msglen, PAYLOAD_FREE);
#endif
      // done with the local copy of the request
      r->remove_reference();
    }

    return ev;
  }

  class TransferPlanReduce : public TransferPlan {
  public:
    TransferPlanReduce(const std::vector<CopySrcDstField>& _srcs,
		       const CopySrcDstField& _dst,
		       ReductionOpID _redop_id, bool _red_fold);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority);

  protected:
    std::vector<CopySrcDstField> srcs;
    CopySrcDstField dst;
    ReductionOpID redop_id;
    bool red_fold;
  };

  TransferPlanReduce::TransferPlanReduce(const std::vector<CopySrcDstField>& _srcs,
					 const CopySrcDstField& _dst,
					 ReductionOpID _redop_id, bool _red_fold)
    : srcs(_srcs)
    , dst(_dst)
    , redop_id(_redop_id)
    , red_fold(_red_fold)
  {}

  Event TransferPlanReduce::execute_plan(const TransferDomain *td,
					 const ProfilingRequestSet& requests,
					 Event wait_on, int priority)
  {
    Event ev = GenEventImpl::create_genevent()->current_event();

    // TODO
    bool inst_lock_needed = false;

    ReduceRequest *r = new ReduceRequest(td,
					 srcs, dst,
					 inst_lock_needed,
					 redop_id, red_fold,
					 wait_on, ev,
					 0 /*priority*/, requests);

    NodeID src_node = ID(srcs[0].inst).instance.owner_node;
    if(src_node == my_node_id) {
      log_dma.debug("performing reduction on local node");

      get_runtime()->optable.add_local_operation(ev, r);	  
      r->check_readiness(false, dma_queue);
    } else {
      r->forward_request(src_node);
      get_runtime()->optable.add_remote_operation(ev, src_node);
#if 0
      RemoteCopyArgs args;
      args.redop_id = redop_id;
      args.red_fold = red_fold;
      args.before_copy = wait_on;
      args.after_copy = ev;
      args.priority = 0 /*priority*/;

      size_t msglen = r->compute_size();
      void *msgdata = malloc(msglen);
      r->serialize(msgdata);

      log_dma.debug("performing reduction on remote node (%d), event=" IDFMT,
		    src_node, args.after_copy.id);
      get_runtime()->optable.add_remote_operation(ev, src_node);
      RemoteCopyMessage::request(src_node, args, msgdata, msglen, PAYLOAD_FREE);
#endif
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

    Event ev = GenEventImpl::create_genevent()->current_event();
    FillRequest *r = new FillRequest(td, f, data.base(), data.size(),
				     wait_on, ev, priority, requests);

    NodeID tgt_node = ID(inst).instance.owner_node;
    if(tgt_node == my_node_id) {
      get_runtime()->optable.add_local_operation(ev, r);
      r->check_readiness(false, dma_queue);
    } else {
      r->forward_request(tgt_node);
      get_runtime()->optable.add_remote_operation(ev, tgt_node);
#if 0
      RemoteFillArgs args;
      args.inst = inst;
      args.offset = offset;
      args.size = data.size();
      args.before_fill = wait_on;
      args.after_fill = ev;
      //args.priority = 0;

      size_t msglen = r->compute_size();
      void *msgdata = malloc(msglen);

      r->serialize(msgdata);

      get_runtime()->optable.add_remote_operation(ev, tgt_node);

      RemoteFillMessage::request(tgt_node, args, msgdata, msglen, PAYLOAD_FREE);
#endif
      // release local copy of operation
      r->remove_reference();
    }

    return ev;
  }

  /*static*/ bool TransferPlan::plan_copy(std::vector<TransferPlan *>& plans,
					  const std::vector<CopySrcDstField> &srcs,
					  const std::vector<CopySrcDstField> &dsts,
					  ReductionOpID redop_id /*= 0*/,
					  bool red_fold /*= false*/)
  {
    if(redop_id == 0) {
      // not a reduction, so sort fields by src/dst mem pairs
      //log_new_dma.info("Performing copy op");

      OASByMem oas_by_mem;

      std::vector<CopySrcDstField>::const_iterator src_it = srcs.begin();
      std::vector<CopySrcDstField>::const_iterator dst_it = dsts.begin();
      unsigned src_subfield_offset = (src_it != srcs.end()) ? src_it->subfield_offset : 0;
      unsigned dst_subfield_offset = (dst_it != dsts.end()) ? dst_it->subfield_offset : 0;

      while((src_it != srcs.end()) && (dst_it != dsts.end())) {
	InstPair ip(src_it->inst, dst_it->inst);
	MemPair mp(get_runtime()->get_instance_impl(src_it->inst)->memory,
		   get_runtime()->get_instance_impl(dst_it->inst)->memory);

	// printf("I:(%x/%x) M:(%x/%x) sub:(%d/%d) src=(%d/%d) dst=(%d/%d)\n",
	//        ip.first.id, ip.second.id, mp.first.id, mp.second.id,
	//        src_suboffset, dst_suboffset,
	//        src_it->offset, src_it->size, 
	//        dst_it->offset, dst_it->size);

	OffsetsAndSize oas;
	oas.src_field_id = src_it->field_id;
	assert(src_it->field_id != (FieldID)-1);
	oas.dst_field_id = dst_it->field_id;
	assert(dst_it->field_id != (FieldID)-1);
	oas.src_subfield_offset = src_subfield_offset;
	oas.dst_subfield_offset = dst_subfield_offset;
	oas.size = std::min(src_it->size - src_subfield_offset,
			    dst_it->size - dst_subfield_offset);
	oas.serdez_id = src_it->serdez_id;

	// This is a little bit of hack: if serdez_id != 0 we directly create a
	// separate copy plan instead of inserting it into ''oasvec''
	if (oas.serdez_id != 0) {
	  OASByInst* oas_by_inst = new OASByInst;
	  (*oas_by_inst)[ip].push_back(oas);
	  TransferPlanCopy *p = new TransferPlanCopy(oas_by_inst);
	  plans.push_back(p);
	} else {
	  // </SERDEZ_DMA>
	  OASByInst *oas_by_inst;
	  OASByMem::iterator it = oas_by_mem.find(mp);
	  if(it != oas_by_mem.end()) {
	    oas_by_inst = it->second;
	  } else {
	    oas_by_inst = new OASByInst;
	    oas_by_mem[mp] = oas_by_inst;
	  }
	  (*oas_by_inst)[ip].push_back(oas);
	}
	src_subfield_offset += oas.size;
	assert(src_subfield_offset <= src_it->size);
	if(src_subfield_offset == src_it->size) {
	  src_it++;
	  if(src_it != srcs.end())
	    src_subfield_offset = src_it->subfield_offset;
	}
	dst_subfield_offset += oas.size;
	assert(dst_subfield_offset <= dst_it->size);
	if(dst_subfield_offset == dst_it->size) {
	  dst_it++;
	  if(dst_it != dsts.end())
	    dst_subfield_offset = dst_it->subfield_offset;
	}
      }
      // make sure we used up both
      assert(src_it == srcs.end());
      assert(dst_it == dsts.end());

      log_dma.debug() << "copy: " << oas_by_mem.size() << " distinct src/dst mem pairs";

      for(OASByMem::const_iterator it = oas_by_mem.begin(); it != oas_by_mem.end(); it++) {
	OASByInst *oas_by_inst = it->second;
	// TODO: teach new DMA code to handle multiple instances in the same memory
	for(OASByInst::const_iterator it2 = oas_by_inst->begin();
	    it2 != oas_by_inst->end();
	    ++it2) {
	  OASByInst *new_oas_by_inst = new OASByInst;
	  (*new_oas_by_inst)[it2->first] = it2->second;
	  TransferPlanCopy *p = new TransferPlanCopy(new_oas_by_inst);
	  plans.push_back(p);
	}
	// done with original oas_by_inst
	delete oas_by_inst;
      }
    } else {
      // reduction op case

      // sanity checks:
      // 1) all sources in same node
      for(size_t i = 1; i < srcs.size(); i++)
	assert(ID(srcs[i].inst).instance.owner_node == ID(srcs[0].inst).instance.owner_node);
      // 2) single destination field
      assert(dsts.size() == 1);

      TransferPlanReduce *p = new TransferPlanReduce(srcs, dsts[0],
						     redop_id, red_fold);
      plans.push_back(p);
    }

    return true;
  }

  /*static*/ bool TransferPlan::plan_fill(std::vector<TransferPlan *>& plans,
					  const std::vector<CopySrcDstField> &dsts,
					  const void *fill_value,
					  size_t fill_value_size)
  {
    // when 'dsts' contains multiple fields, the 'fill_value' should look
    // like a packed struct with a fill value for each field in order -
    // track the offset and complain if we run out of data
    size_t fill_ofs = 0;
    for(std::vector<CopySrcDstField>::const_iterator it = dsts.begin();
	it != dsts.end();
	++it) {
      if((fill_ofs + it->size) > fill_value_size) {
	log_dma.fatal() << "insufficient data for fill - need at least "
			<< (fill_ofs + it->size) << " bytes, but have only " << fill_value_size;
	assert(0);
      }
      assert(it->subfield_offset == 0);
      TransferPlan *p = new TransferPlanFill(((const char *)fill_value) + fill_ofs,
					     it->size,
					     it->inst,
					     it->field_id);
      plans.push_back(p);

      // special case: if a field uses all of the fill value, the next
      //  field (if any) is allowed to use the same value
      if((fill_ofs > 0) || (it->size != fill_value_size))
	fill_ofs += it->size;
    }

    return true;
  }


  template <int N, typename T>
  Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField>& srcs,
			       const std::vector<CopySrcDstField>& dsts,
			       const Realm::ProfilingRequestSet &requests,
			       Event wait_on,
			       ReductionOpID redop_id, bool red_fold) const
  {
    TransferDomain *td = TransferDomain::construct(*this);
    std::vector<TransferPlan *> plans;
    bool ok = TransferPlan::plan_copy(plans, srcs, dsts, redop_id, red_fold);
    assert(ok);
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

  template <int N, typename T>
  Event IndexSpace<N,T>::fill(const std::vector<CopySrcDstField> &dsts,
			       const Realm::ProfilingRequestSet &requests,
			       const void *fill_value, size_t fill_value_size,
			       Event wait_on /*= Event::NO_EVENT*/) const
  {
    TransferDomain *td = TransferDomain::construct(*this);
    std::vector<TransferPlan *> plans;
    bool ok = TransferPlan::plan_fill(plans, dsts, fill_value, fill_value_size);
    assert(ok);
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
					const ProfilingRequestSet&, \
					Event, \
					ReductionOpID, bool) const; \
  template Event IndexSpace<N,T>::fill(const std::vector<CopySrcDstField>&, \
					const ProfilingRequestSet&, \
					const void *, size_t, \
					Event wait_on) const; \
  template class TransferIteratorIndexSpace<N,T>; \
  template class TransferDomainIndexSpace<N,T>;
  FOREACH_NT(DOIT)

}; // namespace Realm
