/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// inlined and templated methods for Realm DMA subsystem

// nop, but helps IDEs
#include "lowlevel_dma.h"

#include "realm/logging.h"

namespace Realm {

    extern Realm::Logger log_dma;

    ////////////////////////////////////////////////////////////////////////
    //
    // class InstPairCopier

    inline InstPairCopier::InstPairCopier(void)
    {
    }

    inline InstPairCopier::~InstPairCopier(void)
    {
    }

    // default implementation is just to iterate over lines
    inline void InstPairCopier::copy_all_fields(off_t src_index, off_t dst_index, off_t count_per_line,
						off_t src_stride, off_t dst_stride, off_t lines)
    {
      for(int i = 0; i < lines; i++) {
	copy_all_fields(src_index, dst_index, count_per_line);
	src_index += src_stride;
	dst_index += dst_stride;
      }
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class SpanBasedInstPairCopier<T>

    // instead of the accessor, we'll grab the implementation pointers
    //  and do address calculation ourselves
    template <typename T>
    SpanBasedInstPairCopier<T>::SpanBasedInstPairCopier(T *_span_copier, 
							RegionInstance _src_inst, 
							RegionInstance _dst_inst,
							OASVec &_oas_vec)
      : span_copier(_span_copier), 
	src_inst(get_runtime()->get_instance_impl(_src_inst)), 
	dst_inst(get_runtime()->get_instance_impl(_dst_inst)), oas_vec(_oas_vec)
    {
      assert(src_inst->metadata.is_valid());
      assert(dst_inst->metadata.is_valid());

      // Precompute our field offset information
      src_start.resize(oas_vec.size());
      dst_start.resize(oas_vec.size());
      src_size.resize(oas_vec.size());
      dst_size.resize(oas_vec.size());
      partial_field.resize(oas_vec.size());
      for(unsigned idx = 0; idx < oas_vec.size(); idx++) {
	find_field_start(src_inst->metadata.field_sizes, oas_vec[idx].src_offset,
			 oas_vec[idx].size, src_start[idx], src_size[idx]);
	find_field_start(dst_inst->metadata.field_sizes, oas_vec[idx].dst_offset,
			 oas_vec[idx].size, dst_start[idx], dst_size[idx]);

	// mark an OASVec entry as being "partial" if src and/or dst don't fill the whole instance field
	partial_field[idx] = ((src_start[idx] != oas_vec[idx].src_offset) ||
			      (src_size[idx] != oas_vec[idx].size) ||
			      (dst_start[idx] != oas_vec[idx].dst_offset) ||
			      (dst_size[idx] != oas_vec[idx].size));
      }
    }

    template <typename T>
    SpanBasedInstPairCopier<T>::~SpanBasedInstPairCopier(void)
    {
    }

    static inline int min(int a, int b) { return (a < b) ? a : b; }

    template <typename T>
    void SpanBasedInstPairCopier<T>::copy_field(off_t src_index, off_t dst_index,
						off_t elem_count, unsigned offset_index)
    {
      off_t src_offset = oas_vec[offset_index].src_offset;
      off_t dst_offset = oas_vec[offset_index].dst_offset;
      int bytes = oas_vec[offset_index].size;

      assert(src_inst->metadata.is_valid());
      assert(dst_inst->metadata.is_valid());

      off_t src_field_start, dst_field_start;
      int src_field_size, dst_field_size;

      //find_field_start(src_inst->metadata.field_sizes, src_offset, bytes, src_field_start, src_field_size);
      //find_field_start(dst_inst->metadata.field_sizes, dst_offset, bytes, dst_field_start, dst_field_size);
      src_field_start = src_start[offset_index];
      dst_field_start = dst_start[offset_index];
      src_field_size = src_size[offset_index];
      dst_field_size = dst_size[offset_index];

      // if both source and dest fill up an entire field, we might be able to copy whole ranges at the same time
      if((src_field_start == src_offset) && (src_field_size == bytes) &&
	 (dst_field_start == dst_offset) && (dst_field_size == bytes)) {
	// let's see how many we can copy
	int done = 0;
	while(done < elem_count) {
	  off_t src_in_this_block = src_inst->metadata.block_size - ((src_index + done) % src_inst->metadata.block_size);
	  off_t dst_in_this_block = dst_inst->metadata.block_size - ((dst_index + done) % dst_inst->metadata.block_size);
	  off_t todo = min(elem_count - done, min(src_in_this_block, dst_in_this_block));

	  //printf("copying range of %d elements (%d, %d, %d)\n", todo, src_index, dst_index, done);

	  off_t src_start = calc_mem_loc(src_inst->metadata.alloc_offset + (src_offset - src_field_start),
					 src_field_start, src_field_size, src_inst->metadata.elmt_size,
					 src_inst->metadata.block_size, src_index + done);
	  off_t dst_start = calc_mem_loc(dst_inst->metadata.alloc_offset + (dst_offset - dst_field_start),
					 dst_field_start, dst_field_size, dst_inst->metadata.elmt_size,
					 dst_inst->metadata.block_size, dst_index + done);

	  // sanity check that the range we calculated really is contiguous
	  assert(calc_mem_loc(src_inst->metadata.alloc_offset + (src_offset - src_field_start),
			      src_field_start, src_field_size, src_inst->metadata.elmt_size,
			      src_inst->metadata.block_size, src_index + done + todo - 1) == 
		 (src_start + (todo - 1) * bytes));
	  assert(calc_mem_loc(dst_inst->metadata.alloc_offset + (dst_offset - dst_field_start),
			      dst_field_start, dst_field_size, dst_inst->metadata.elmt_size,
			      dst_inst->metadata.block_size, dst_index + done + todo - 1) == 
		 (dst_start + (todo - 1) * bytes));

	  span_copier->copy_span(src_start, dst_start, bytes * todo);

	  done += todo;
	}
      } else {
	// fallback - calculate each address separately
	for(int i = 0; i < elem_count; i++) {
	  off_t src_start = calc_mem_loc(src_inst->metadata.alloc_offset + (src_offset - src_field_start),
					 src_field_start, src_field_size, src_inst->metadata.elmt_size,
					 src_inst->metadata.block_size, src_index + i);
	  off_t dst_start = calc_mem_loc(dst_inst->metadata.alloc_offset + (dst_offset - dst_field_start),
					 dst_field_start, dst_field_size, dst_inst->metadata.elmt_size,
					 dst_inst->metadata.block_size, dst_index + i);

	  span_copier->copy_span(src_start, dst_start, bytes);
	}
      }
    }

    template <typename T>
    void SpanBasedInstPairCopier<T>::copy_all_fields(off_t src_index, off_t dst_index, off_t elem_count)
    {
      // first check - if the span we're copying straddles a block boundary
      //  go back to old way - block size of 1 is ok only if both are
      assert(src_inst->metadata.is_valid());
      assert(dst_inst->metadata.is_valid());

      size_t src_bsize = src_inst->metadata.block_size;
      size_t dst_bsize = dst_inst->metadata.block_size;

      if(((src_bsize == 1) != (dst_bsize == 1)) ||
	 ((src_bsize > 1) && ((src_index / src_bsize) != ((src_index + elem_count - 1) / src_bsize))) ||
	 ((dst_bsize > 1) && ((dst_index / dst_bsize) != ((dst_index + elem_count - 1) / dst_bsize)))) {
	// SJT: would like to include the instance info, but it's tripping over some namespace-related template
	//  ambiguity between Realm loggers and serializers...
	//log_dma.info() << "copy between instances " << src_inst->me << " and " << dst_inst->me << " straddles block boundaries - falling back";
	log_dma.info() << "copy between instances straddles block boundaries - falling back";
	for(unsigned i = 0; i < oas_vec.size(); i++)
	  copy_field(src_index, dst_index, elem_count, i);
	return;
      }

      // start with the first field, grabbing as many at a time as we can

      unsigned field_idx = 0;

      while(field_idx < oas_vec.size()) {
	// get information about the first field
	off_t src_offset = oas_vec[field_idx].src_offset;
	off_t dst_offset = oas_vec[field_idx].dst_offset;
	unsigned bytes = oas_vec[field_idx].size;

	// if src and/or dst aren't a full field, fall back to the old way for this field
	off_t src_field_start = src_start[field_idx];
	int src_field_size = src_size[field_idx];
	off_t dst_field_start = dst_start[field_idx];
	int dst_field_size = dst_size[field_idx];

	if(partial_field[field_idx]) {
	  printf("not a full field - falling back\n");
	  copy_field(src_index, dst_index, elem_count, field_idx);
	  field_idx++;
	  continue;
	}

	// see if we can tack on more fields
	unsigned field_idx2 = field_idx + 1;
	int src_fstride = 0;
	int dst_fstride = 0;
	unsigned total_bytes = bytes;
	unsigned total_lines = 1;
	while(field_idx2 < oas_vec.size()) {
	  // TODO: for now, don't merge fields here because it can result in too-large copies
	  break;

	  // is this a partial field?  if so, stop
	  if(partial_field[field_idx2])
	    break;

	  off_t src_offset2 = oas_vec[field_idx2].src_offset;
	  off_t dst_offset2 = oas_vec[field_idx2].dst_offset;

	  // test depends on AOS (bsize == 1) vs (hybrid)SOA (bsize > 1)
	  if(src_bsize == 1) {
	    // for AOS, we need this field's offset to be the next byte
	    if((src_offset2 != (off_t)(src_offset + total_bytes)) ||
	       (dst_offset2 != (off_t)(dst_offset + total_bytes)))
	      break;

	    // if tests pass, add this field's size to our total and keep going
	    total_bytes += oas_vec[field_idx2].size;
	  } else {
	    // in SOA, we need the field's strides to match, but non-contiguous is ok
	    // first stride will be ok by construction
	    int src_fstride2 = src_offset2 - src_offset;
	    int dst_fstride2 = dst_offset2 - dst_offset;
	    if(src_fstride == 0) src_fstride = src_fstride2;
	    if(dst_fstride == 0) dst_fstride = dst_fstride2;
	    if((src_fstride2 != (int)(field_idx2 - field_idx) * src_fstride) ||
	       (dst_fstride2 != (int)(field_idx2 - field_idx) * dst_fstride))
	      break;

	    // if tests pass, we have another line
	    total_lines++;
	  }

	  field_idx2++;
	}

	// now we can copy something
	off_t src_start = calc_mem_loc(src_inst->metadata.alloc_offset + (src_offset - src_field_start),
				       src_field_start, src_field_size, src_inst->metadata.elmt_size,
				       src_inst->metadata.block_size, src_index);
	off_t dst_start = calc_mem_loc(dst_inst->metadata.alloc_offset + (dst_offset - dst_field_start),
				       dst_field_start, dst_field_size, dst_inst->metadata.elmt_size,
				       dst_inst->metadata.block_size, dst_index);

	// AOS merging doesn't work if we don't end up with the full element
	if((src_bsize == 1) &&
	   ((total_bytes < src_inst->metadata.elmt_size) || (total_bytes < dst_inst->metadata.elmt_size))) {
          // AOS copy doesn't include all fields in source and/or dest, so we have to turn it into a
          //  "2-D" copy in which each element is a "line"
          span_copier->copy_span(src_start, dst_start, total_bytes,
                                 src_fstride, dst_fstride, elem_count);
        } else {
	  span_copier->copy_span(src_start, dst_start, elem_count * total_bytes,
	                        src_fstride * src_bsize,
	                        dst_fstride * dst_bsize,
			                    total_lines);
        }

	// continue with the first field we couldn't take for this pass
	field_idx = field_idx2;
      }
    }

    template <typename T>
    void SpanBasedInstPairCopier<T>::copy_all_fields(off_t src_index, off_t dst_index, off_t count_per_line,
						     off_t src_stride, off_t dst_stride, off_t lines)
    {
      // first check - if the span we're copying straddles a block boundary
      //  go back to old way - block size of 1 is ok only if both are
      assert(src_inst->metadata.is_valid());
      assert(dst_inst->metadata.is_valid());
      
      size_t src_bsize = src_inst->metadata.block_size;
      size_t dst_bsize = dst_inst->metadata.block_size;
      
      off_t src_last = src_index + (count_per_line - 1) + (lines - 1) * src_stride;
      off_t dst_last = dst_index + (count_per_line - 1) + (lines - 1) * dst_stride;

      if(((src_bsize == 1) != (dst_bsize == 1)) ||
	 ((src_bsize > 1) && ((src_index / src_bsize) != (src_last / src_bsize))) ||
	 ((dst_bsize > 1) && ((dst_index / dst_bsize) != (dst_last / dst_bsize)))) {
	// SJT: would like to include the instance info, but it's tripping over some namespace-related template
	//  ambiguity between Realm loggers and serializers...
	//log_dma.info() << "copy between instances " << src_inst->me << " and " << dst_inst->me << " straddles block boundaries - falling back";
	log_dma.info() << "copy between instances straddles block boundaries - falling back";
	for(unsigned i = 0; i < oas_vec.size(); i++)
	  for(int l = 0; l < lines; l++)
	    copy_field(src_index + l * src_stride, 
		       dst_index + l * dst_stride, count_per_line, i);
	return;
      }

      // start with the first field, grabbing as many at a time as we can

      unsigned field_idx = 0;

      while(field_idx < oas_vec.size()) {
	// get information about the first field
	off_t src_offset = oas_vec[field_idx].src_offset;
	off_t dst_offset = oas_vec[field_idx].dst_offset;
	unsigned bytes = oas_vec[field_idx].size;

	// if src and/or dst aren't a full field, fall back to the old way for this field
	off_t src_field_start = src_start[field_idx];
	int src_field_size = src_size[field_idx];
	off_t dst_field_start = dst_start[field_idx];
	int dst_field_size = dst_size[field_idx];

	if(partial_field[field_idx]) {
	  log_dma.info() << "not a full field - falling back";
	  copy_field(src_index, dst_index, count_per_line, field_idx);
	  field_idx++;
	  continue;
	}

	// see if we can tack on more fields
	unsigned field_idx2 = field_idx + 1;
	int src_fstride = 0;
	int dst_fstride = 0;
	unsigned total_bytes = bytes;
	unsigned total_lines = 1;
	while(field_idx2 < oas_vec.size()) {
	  // is this a partial field?  if so, stop
	  if(partial_field[field_idx2])
	    break;

	  unsigned src_offset2 = oas_vec[field_idx2].src_offset;
	  unsigned dst_offset2 = oas_vec[field_idx2].dst_offset;

	  // test depends on AOS (bsize == 1) vs (hybrid)SOA (bsize > 1)
	  if(src_bsize == 1) {
	    // for AOS, we need this field's offset to be the next byte
	    if((src_offset2 != (src_offset + total_bytes)) ||
	       (dst_offset2 != (dst_offset + total_bytes)))
	      break;

	    // if tests pass, add this field's size to our total and keep going
	    total_bytes += oas_vec[field_idx2].size;
	  } else {
	    // in SOA, we need the field's strides to match, but non-contiguous is ok
	    // first stride will be ok by construction
	    int src_fstride2 = src_offset2 - src_offset;
	    int dst_fstride2 = dst_offset2 - dst_offset;
	    if(src_fstride == 0) src_fstride = src_fstride2;
	    if(dst_fstride == 0) dst_fstride = dst_fstride2;
	    if((src_fstride2 != (int)(field_idx2 - field_idx) * src_fstride) ||
	       (dst_fstride2 != (int)(field_idx2 - field_idx) * dst_fstride))
	      break;

	    // if tests pass, we have another line
	    total_lines++;
	  }

	  field_idx2++;
	}

	// now we can copy something
	off_t src_start = calc_mem_loc(src_inst->metadata.alloc_offset + (src_offset - src_field_start),
				       src_field_start, src_field_size, src_inst->metadata.elmt_size,
				       src_inst->metadata.block_size, src_index);
	off_t dst_start = calc_mem_loc(dst_inst->metadata.alloc_offset + (dst_offset - dst_field_start),
				       dst_field_start, dst_field_size, dst_inst->metadata.elmt_size,
				       dst_inst->metadata.block_size, dst_index);

	// AOS merging doesn't work if we don't end up with the full element
	if((src_bsize == 1) && 
	   ((total_bytes < src_inst->metadata.elmt_size) || (total_bytes < dst_inst->metadata.elmt_size)) &&
	   (count_per_line > 1)) {
	  log_dma.error() << "help: AOS tried to merge subset of fields with multiple elements - not contiguous!";
	  assert(0);
	}

	// since we're already 2D, we need line strides to match up
	if(total_lines > 1) {
	  if(0) {
	  } else {
	    // no?  punt on the field merging
	    total_lines = 1;
	    //printf("CCC: eliminating field merging\n");
	    field_idx2 = field_idx + 1;
	  }
	}

	span_copier->copy_span(src_start, dst_start, count_per_line * total_bytes,
			       src_stride * bytes,
			       dst_stride * bytes,
			       lines);

	// continue with the first field we couldn't take for this pass
	field_idx = field_idx2;
      }
    }

    template <typename T>
    void SpanBasedInstPairCopier<T>::flush(void)
    {
    }


};

