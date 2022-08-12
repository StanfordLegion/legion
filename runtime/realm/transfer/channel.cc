/* Copyright 2022 Stanford University
 * Copyright 2022 Los Alamos National Laboratory
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

#include "realm/realm_config.h"

#ifdef REALM_ON_WINDOWS
#define NOMINMAX
#endif

#include "realm/transfer/channel.h"
#include "realm/transfer/channel_disk.h"
#include "realm/transfer/transfer.h"
#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/ib_memory.h"
#include "realm/utils.h"

#include <algorithm>

TYPE_IS_SERIALIZABLE(Realm::XferDesKind);

namespace Realm {

    Logger log_new_dma("new_dma");
    Logger log_request("request");
    Logger log_xd("xd");

  
  // fast memcpy stuff - uses std::copy instead of memcpy to communicate
  //  alignment guarantees to the compiler
  template <typename T>
  static void memcpy_1d_typed(uintptr_t dst_base, uintptr_t src_base,
			      size_t bytes)
  {
    std::copy(reinterpret_cast<const T *>(src_base),
	      reinterpret_cast<const T *>(src_base + bytes),
	      reinterpret_cast<T *>(dst_base));
  }

  template <typename T>
  static void memcpy_2d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
			      uintptr_t src_base, uintptr_t src_lstride,
			      size_t bytes, size_t lines)
  {
    for(size_t i = 0; i < lines; i++) {
      std::copy(reinterpret_cast<const T *>(src_base),
		reinterpret_cast<const T *>(src_base + bytes),
		reinterpret_cast<T *>(dst_base));
      // manual strength reduction
      src_base += src_lstride;
      dst_base += dst_lstride;
    }
  }

  template <typename T>
  static void memcpy_3d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
			      uintptr_t dst_pstride,
			      uintptr_t src_base, uintptr_t src_lstride,
			      uintptr_t src_pstride,
			      size_t bytes, size_t lines, size_t planes)
  {
    // adjust plane stride amounts to account for line strides (so we don't have
    //  to subtract the line strides back out in the loop)
    uintptr_t dst_pstride_adj = dst_pstride - (lines * dst_lstride);
    uintptr_t src_pstride_adj = src_pstride - (lines * src_lstride);

    for(size_t j = 0; j < planes; j++) {
      for(size_t i = 0; i < lines; i++) {
	std::copy(reinterpret_cast<const T *>(src_base),
		  reinterpret_cast<const T *>(src_base + bytes),
		  reinterpret_cast<T *>(dst_base));
	// manual strength reduction
	src_base += src_lstride;
	dst_base += dst_lstride;
      }
      src_base += src_pstride_adj;
      dst_base += dst_pstride_adj;
    }
  }

  template <typename T>
  static void memset_1d_typed(uintptr_t dst_base, size_t bytes, T val)
  {
    std::fill(reinterpret_cast<T *>(dst_base),
	      reinterpret_cast<T *>(dst_base + bytes),
              val);
  }

  template <typename T>
  static void memset_2d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
			      size_t bytes, size_t lines, T val)
  {
    for(size_t i = 0; i < lines; i++) {
      std::fill(reinterpret_cast<T *>(dst_base),
		reinterpret_cast<T *>(dst_base + bytes),
                val);
      // manual strength reduction
      dst_base += dst_lstride;
    }
  }

  template <typename T>
  static void memset_3d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
			      uintptr_t dst_pstride,
			      size_t bytes, size_t lines, size_t planes,
                              T val)
  {
    // adjust plane stride amounts to account for line strides (so we don't have
    //  to subtract the line strides back out in the loop)
    uintptr_t dst_pstride_adj = dst_pstride - (lines * dst_lstride);

    for(size_t j = 0; j < planes; j++) {
      for(size_t i = 0; i < lines; i++) {
	std::fill(reinterpret_cast<T *>(dst_base),
		  reinterpret_cast<T *>(dst_base + bytes),
                  val);
	// manual strength reduction
	dst_base += dst_lstride;
      }
      dst_base += dst_pstride_adj;
    }
  }

  // need types with various powers-of-2 size/alignment - we have up to
  //  uint64_t as builtins, but we need trivially-copyable 16B and 32B things
  struct dummy_16b_t { uint64_t a, b; };
  struct dummy_32b_t { uint64_t a, b, c, d; };
  REALM_ALIGNED_TYPE_CONST(aligned_16b_t, dummy_16b_t, 16);
  REALM_ALIGNED_TYPE_CONST(aligned_32b_t, dummy_32b_t, 32);

  void memcpy_1d(uintptr_t dst_base, uintptr_t src_base,
                 size_t bytes)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (src_base - 1) &
			  (bytes - 1));
//define DEBUG_MEMCPYS
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memcpy_1d: dst=" << dst_base
                   << " src=" << src_base
                   << std::dec << " bytes=" << bytes
                   << " align=" << (alignment & 31);
#endif
    // TODO: consider jump table approach?
    if((alignment & 31) == 31)
      memcpy_1d_typed<aligned_32b_t>(dst_base, src_base, bytes);
    else if((alignment & 15) == 15)
      memcpy_1d_typed<aligned_16b_t>(dst_base, src_base, bytes);
    else if((alignment & 7) == 7)
      memcpy_1d_typed<uint64_t>(dst_base, src_base, bytes);
    else if((alignment & 3) == 3)
      memcpy_1d_typed<uint32_t>(dst_base, src_base, bytes);
    else if((alignment & 1) == 1)
      memcpy_1d_typed<uint16_t>(dst_base, src_base, bytes);
    else
      memcpy_1d_typed<uint8_t>(dst_base, src_base, bytes);
  }

  void memcpy_2d(uintptr_t dst_base, uintptr_t dst_lstride,
                 uintptr_t src_base, uintptr_t src_lstride,
                 size_t bytes, size_t lines)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (dst_lstride - 1) &
			  (src_base - 1) & (src_lstride - 1) &
			  (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memcpy_2d: dst=" << dst_base
                   << "+" << dst_lstride
                   << " src=" << src_base
                   << "+" << src_lstride
                   << std::dec << " bytes=" << bytes
                   << " lines=" << lines
                   << " align=" << (alignment & 31);
#endif
    // TODO: consider jump table approach?
    if((alignment & 31) == 31)
      memcpy_2d_typed<aligned_32b_t>(dst_base, dst_lstride,
				     src_base, src_lstride,
				     bytes, lines);
    else if((alignment & 15) == 15)
      memcpy_2d_typed<aligned_16b_t>(dst_base, dst_lstride,
				     src_base, src_lstride,
				     bytes, lines);
    else if((alignment & 7) == 7)
      memcpy_2d_typed<uint64_t>(dst_base, dst_lstride, src_base, src_lstride,
				bytes, lines);
    else if((alignment & 3) == 3)
      memcpy_2d_typed<uint32_t>(dst_base, dst_lstride, src_base, src_lstride,
				bytes, lines);
    else if((alignment & 1) == 1)
      memcpy_2d_typed<uint16_t>(dst_base, dst_lstride, src_base, src_lstride,
				bytes, lines);
    else
      memcpy_2d_typed<uint8_t>(dst_base, dst_lstride, src_base, src_lstride,
			       bytes, lines);
  }

  void memcpy_3d(uintptr_t dst_base, uintptr_t dst_lstride,
                 uintptr_t dst_pstride,
                 uintptr_t src_base, uintptr_t src_lstride,
                 uintptr_t src_pstride,
                 size_t bytes, size_t lines, size_t planes)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (dst_lstride - 1) &
			  (dst_pstride - 1) &
			  (src_base - 1) & (src_lstride - 1) &
			  (src_pstride - 1) &
			  (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memcpy_3d: dst=" << dst_base
                   << "+" << dst_lstride << "+" << dst_pstride
                   << " src=" << src_base
                   << "+" << src_lstride << "+" << src_pstride
                   << std::dec << " bytes=" << bytes
                   << " lines=" << lines
                   << " planes=" << planes
                   << " align=" << (alignment & 31);
#endif
    // performance optimization for intel (and probably other) cpus: walk
    //  destination addresses as linearly as possible, even if that messes up
    //  the source address pattern (probably because writebacks are more
    //  expensive than cache fills?)
    if(dst_pstride < dst_lstride) {
      // switch lines and planes
      std::swap(dst_pstride, dst_lstride);
      std::swap(src_pstride, src_lstride);
      std::swap(planes, lines);
    }
    // TODO: consider jump table approach?
    if((alignment & 31) == 31)
      memcpy_3d_typed<aligned_32b_t>(dst_base, dst_lstride, dst_pstride,
				     src_base, src_lstride, src_pstride,
				     bytes, lines, planes);
    else if((alignment & 15) == 15)
      memcpy_3d_typed<aligned_16b_t>(dst_base, dst_lstride, dst_pstride,
				     src_base, src_lstride, src_pstride,
				     bytes, lines, planes);
    else if((alignment & 7) == 7)
      memcpy_3d_typed<uint64_t>(dst_base, dst_lstride, dst_pstride,
				src_base, src_lstride, src_pstride,
				bytes, lines, planes);
    else if((alignment & 3) == 3)
      memcpy_3d_typed<uint32_t>(dst_base, dst_lstride, dst_pstride,
				src_base, src_lstride, src_pstride,
				bytes, lines, planes);
    else if((alignment & 1) == 1)
      memcpy_3d_typed<uint16_t>(dst_base, dst_lstride, dst_pstride,
				src_base, src_lstride, src_pstride,
				bytes, lines, planes);
    else
      memcpy_3d_typed<uint8_t>(dst_base, dst_lstride, dst_pstride,
				src_base, src_lstride, src_pstride,
			       bytes, lines, planes);
  }

  void memset_1d(uintptr_t dst_base, size_t bytes,
                 const void *fill_data, size_t fill_size)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memset_1d: dst=" << dst_base
                   << std::dec << " bytes=" << bytes
                   << " align=" << (alignment & 31);
#endif
    // alignment must be at least as good as fill size to use memset
    // TODO: consider jump table approach?
    if((fill_size == 32) && ((alignment & 31) == 31))
      memset_1d_typed<aligned_32b_t>(dst_base, bytes,
                                     *reinterpret_cast<const aligned_32b_t *>(fill_data));
    else if((fill_size == 16) && ((alignment & 15) == 15))
      memset_1d_typed<aligned_16b_t>(dst_base, bytes,
                                     *reinterpret_cast<const aligned_16b_t *>(fill_data));
    else if((fill_size == 8) && ((alignment & 7) == 7))
      memset_1d_typed<uint64_t>(dst_base, bytes,
                                *reinterpret_cast<const uint64_t *>(fill_data));
    else if((fill_size == 4) && ((alignment & 3) == 3))
      memset_1d_typed<uint32_t>(dst_base, bytes,
                                *reinterpret_cast<const uint32_t *>(fill_data));
    else if((fill_size == 2) && ((alignment & 1) == 1))
      memset_1d_typed<uint16_t>(dst_base, bytes,
                                *reinterpret_cast<const uint16_t *>(fill_data));
    else if(fill_size == 1)
      memset_1d_typed<uint8_t>(dst_base, bytes,
                               *reinterpret_cast<const uint8_t *>(fill_data));
    else {
      // fallback based on memcpy
      memcpy_2d(dst_base, fill_size,
                reinterpret_cast<uintptr_t>(fill_data), 0,
                fill_size, bytes / fill_size);
    }
  }

  void memset_2d(uintptr_t dst_base, uintptr_t dst_lstride,
                 size_t bytes, size_t lines,
                 const void *fill_data, size_t fill_size)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (dst_lstride - 1) &
			  (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memset_2d: dst=" << dst_base
                   << "+" << dst_lstride
                   << std::dec << " bytes=" << bytes
                   << " lines=" << lines
                   << " align=" << (alignment & 31);
#endif
    // alignment must be at least as good as fill size to use memset
    // TODO: consider jump table approach?
    if((fill_size == 32) && ((alignment & 31) == 31))
      memset_2d_typed<aligned_32b_t>(dst_base, dst_lstride,
                                     bytes, lines,
                                     *reinterpret_cast<const aligned_32b_t *>(fill_data));
    else if((fill_size == 16) && ((alignment & 15) == 15))
      memset_2d_typed<aligned_16b_t>(dst_base, dst_lstride,
                                     bytes, lines,
                                     *reinterpret_cast<const aligned_16b_t *>(fill_data));
    else if((fill_size == 8) && ((alignment & 7) == 7))
      memset_2d_typed<uint64_t>(dst_base, dst_lstride,
                                bytes, lines,
                                *reinterpret_cast<const uint64_t *>(fill_data));
    else if((fill_size == 4) && ((alignment & 3) == 3))
      memset_2d_typed<uint32_t>(dst_base, dst_lstride,
                                bytes, lines,
                                *reinterpret_cast<const uint32_t *>(fill_data));
    else if((fill_size == 2) && ((alignment & 1) == 1))
      memset_2d_typed<uint16_t>(dst_base, dst_lstride,
                                bytes, lines,
                                *reinterpret_cast<const uint16_t *>(fill_data));
    else if(fill_size == 1)
      memset_2d_typed<uint8_t>(dst_base, dst_lstride,
                               bytes, lines,
                               *reinterpret_cast<const uint8_t *>(fill_data));
    else {
      // fallback based on memcpy
      memcpy_3d(dst_base, fill_size, dst_lstride,
                reinterpret_cast<uintptr_t>(fill_data), 0, 0,
                fill_size, bytes / fill_size, lines);
    }
  }

  void memset_3d(uintptr_t dst_base, uintptr_t dst_lstride,
                 uintptr_t dst_pstride,
                 size_t bytes, size_t lines, size_t planes,
                 const void *fill_data, size_t fill_size)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (dst_lstride - 1) &
			  (dst_pstride - 1) &
			  (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memset_3d: dst=" << dst_base
                   << "+" << dst_lstride << "+" << dst_pstride
                   << std::dec << " bytes=" << bytes
                   << " lines=" << lines
                   << " planes=" << planes
                   << " align=" << (alignment & 31);
#endif
    // alignment must be at least as good as fill size to use memset
    // TODO: consider jump table approach?
    if((fill_size == 32) && ((alignment & 31) == 31))
      memset_3d_typed<aligned_32b_t>(dst_base, dst_lstride, dst_pstride,
                                     bytes, lines, planes,
                                     *reinterpret_cast<const aligned_32b_t *>(fill_data));
    else if((fill_size == 16) && ((alignment & 15) == 15))
      memset_3d_typed<aligned_16b_t>(dst_base, dst_lstride, dst_pstride,
                                     bytes, lines, planes,
                                     *reinterpret_cast<const aligned_16b_t *>(fill_data));
    else if((fill_size == 8) && ((alignment & 7) == 7))
      memset_3d_typed<uint64_t>(dst_base, dst_lstride, dst_pstride,
                                bytes, lines, planes,
                                *reinterpret_cast<const uint64_t *>(fill_data));
    else if((fill_size == 4) && ((alignment & 3) == 3))
      memset_3d_typed<uint32_t>(dst_base, dst_lstride, dst_pstride,
                                bytes, lines, planes,
                                *reinterpret_cast<const uint32_t *>(fill_data));
    else if((fill_size == 2) && ((alignment & 1) == 1))
      memset_3d_typed<uint16_t>(dst_base, dst_lstride, dst_pstride,
                                bytes, lines, planes,
                                *reinterpret_cast<const uint16_t *>(fill_data));
    else if(fill_size == 1)
      memset_3d_typed<uint8_t>(dst_base, dst_lstride, dst_pstride,
                               bytes, lines, planes,
                               *reinterpret_cast<const uint8_t *>(fill_data));
    else {
      // fallback based on memcpy
      for(size_t p = 0; p < planes; p++)
        memcpy_3d(dst_base + (p * dst_pstride), fill_size, dst_lstride,
                  reinterpret_cast<uintptr_t>(fill_data), 0, 0,
                  fill_size, bytes / fill_size, lines);
    }
  }

#if 0
      static inline bool cross_ib(off_t start, size_t nbytes, size_t buf_size)
      {
        return (nbytes > 0) && (start / buf_size < (start + nbytes - 1) / buf_size);
      }
#endif
    ////////////////////////////////////////////////////////////////////////
    //
    // class SequenceAssembler
    //

    SequenceAssembler::SequenceAssembler(void)
      : contig_amount_x2(0)
      , first_noncontig((size_t)-1)
      , mutex(0)
    {}

    SequenceAssembler::SequenceAssembler(const SequenceAssembler& copy_from)
      : contig_amount_x2(copy_from.contig_amount_x2)
      , first_noncontig(copy_from.first_noncontig)
      , mutex(0)
      , spans(copy_from.spans)
    {}

    SequenceAssembler::~SequenceAssembler(void)
    {
      if(mutex.load())
        delete mutex.load();
    }

    Mutex *SequenceAssembler::ensure_mutex()
    {
      Mutex *ptr = mutex.load();
      if(ptr)
        return ptr;
      // allocate one and try to install it
      Mutex *new_mutex = new Mutex;
      if(mutex.compare_exchange(ptr, new_mutex)) {
        // succeeded - return the mutex we made
        return new_mutex;
      } else {
        // failed - destroy the one we made and use the one that got there first
        delete new_mutex;
        return ptr;
      }
    }

    bool SequenceAssembler::empty() const
    {
      return (contig_amount_x2.load() == 0);
    }

    void SequenceAssembler::swap(SequenceAssembler& other)
    {
      // NOT thread-safe - taking mutexes won't help
      std::swap(contig_amount_x2, other.contig_amount_x2);
      std::swap(first_noncontig, other.first_noncontig);
      spans.swap(other.spans);
    }

    void SequenceAssembler::import(SequenceAssembler& other) const
    {
      size_t contig_sample_x2 = contig_amount_x2.load_acquire();
      if(contig_sample_x2 > 1)
        other.add_span(0, contig_sample_x2 >> 1);
      if((contig_sample_x2 & 1) != 0) {
        for(std::map<size_t, size_t>::const_iterator it = spans.begin();
            it != spans.end();
            ++it)
          other.add_span(it->first, it->second);
      }
    }

    // asks if a span exists - return value is number of bytes from the
    //  start that do
    size_t SequenceAssembler::span_exists(size_t start, size_t count)
    {
      // lock-free case 1: start < contig_amount
      size_t contig_sample_x2 = contig_amount_x2.load_acquire();
      if(start < (contig_sample_x2 >> 1)) {
	size_t max_avail = (contig_sample_x2 >> 1) - start;
	if(count < max_avail)
	  return count;
	else
	  return max_avail;
      }

      // lock-free case 2a: no noncontig ranges known
      if((contig_sample_x2 & 1) == 0)
	return 0;

      // lock-free case 2b: contig_amount <= start < first_noncontig
      size_t noncontig_sample = first_noncontig.load();
      if(start < noncontig_sample)
	return 0;

      // general case 3: take the lock and look through spans/etc.
      {
	AutoLock<> al(*ensure_mutex());

	// first, recheck the contig_amount, in case both it and the noncontig
	//  counters were bumped in between looking at the two of them
	size_t contig_sample = contig_amount_x2.load_acquire() >> 1;
	if(start < contig_sample) {
	  size_t max_avail = contig_sample - start;
	  if(count < max_avail)
	    return count;
	  else
	    return max_avail;
	}

	// recheck noncontig as well
	if(start < first_noncontig.load())
	  return 0;

	// otherwise find the first span after us and then back up one to find
	//  the one that might contain our 'start'
	std::map<size_t, size_t>::const_iterator it = spans.upper_bound(start);
	// this should never be the first span
	assert(it != spans.begin());
	--it;
	assert(it->first <= start);
	// does this span overlap us?
	if((it->first + it->second) > start) {
	  size_t max_avail = it->first + it->second - start;
	  while(max_avail < count) {
	    // try to get more - return the current 'max_avail' if we fail
	    if(++it == spans.end())
	      return max_avail; // no more
	    if(it->first > (start + max_avail))
	      return max_avail; // not contiguous
	    max_avail += it->second;
	  }
	  // got at least as much as we wanted
	  return count;
	} else
	  return 0;
      }
    }

    // returns the amount by which the contiguous range has been increased
    //  (i.e. from [pos, pos+retval) )
    size_t SequenceAssembler::add_span(size_t pos, size_t count)
    {
      // nothing to do for empty spans
      if(count == 0)
        return 0;

      // fastest case - try to bump the contig amount without a lock, assuming
      //  there's no noncontig spans
      size_t prev_x2 = pos << 1;
      size_t next_x2 = (pos + count) << 1;
      if(contig_amount_x2.compare_exchange(prev_x2, next_x2)) {
	// success - we bumped by exactly 'count'
	return count;
      }

      // second best case - the CAS failed, but only because there are
      //  noncontig spans...  assuming spans aren't getting too out of order
      //  in the common case, we take the mutex and pick up any other spans we
      //  connect with
      if((prev_x2 >> 1) == pos) {
	size_t span_end = pos + count;
	{
	  AutoLock<> al(*ensure_mutex());

	  size_t new_noncontig = size_t(-1);
	  while(!spans.empty()) {
	    std::map<size_t, size_t>::iterator it = spans.begin();
	    if(it->first == span_end) {
	      span_end += it->second;
	      spans.erase(it);
	    } else {
	      // stop here - this is the new first noncontig
	      new_noncontig = it->first;
	      break;
	    }
	  }

	  // to avoid false negatives in 'span_exists', update contig amount
	  //  before we bump first_noncontig
	  next_x2 = (span_end << 1) + (spans.empty() ? 0 : 1);
	  // this must succeed
	  bool ok = contig_amount_x2.compare_exchange(prev_x2, next_x2);
	  assert(ok);

	  first_noncontig.store(new_noncontig);
	}

	return (span_end - pos);
      }

      // worst case - our span doesn't appear to be contiguous, so we have to
      //  take the mutex and add to the noncontig list (we may end up being
      //  contiguous if we're the first noncontig and things have caught up)
      {
	AutoLock<> al(*ensure_mutex());

	spans[pos] = count;

	if(pos > first_noncontig.load()) {
	  // in this case, we also know that spans wasn't empty and somebody
	  //  else has already set the LSB of contig_amount_x2
	  return 0;
	} else {
	  // we need to re-check contig_amount_x2 and make sure the LSB is
	  //  set - do both with an atomic OR
	  prev_x2 = contig_amount_x2.fetch_or(1);

	  if((prev_x2 >> 1) == pos) {
	    // we've been caught, so gather up spans and do another bump
	    size_t span_end = pos;
	    size_t new_noncontig = size_t(-1);
	    while(!spans.empty()) {
	      std::map<size_t, size_t>::iterator it = spans.begin();
	      if(it->first == span_end) {
		span_end += it->second;
		spans.erase(it);
	      } else {
		// stop here - this is the new first noncontig
		new_noncontig = it->first;
		break;
	      }
	    }
	    assert(span_end > pos);

	    // to avoid false negatives in 'span_exists', update contig amount
	    //  before we bump first_noncontig
	    next_x2 = (span_end << 1) + (spans.empty() ? 0 : 1);
	    // this must succeed (as long as we remember we set the LSB)
	    prev_x2 |= 1;
	    bool ok = contig_amount_x2.compare_exchange(prev_x2, next_x2);
	    assert(ok);

	    first_noncontig.store(new_noncontig);

	    return (span_end - pos);
	  } else {
	    // not caught, so no forward progress to report
	    return 0;
	  }
	}
      }
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressList
  //

  AddressList::AddressList()
    : total_bytes(0)
    , write_pointer(0)
    , read_pointer(0)
  {}

  size_t *AddressList::begin_nd_entry(int max_dim)
  {
    size_t entries_needed = max_dim * 2;

    size_t new_wp = write_pointer + entries_needed;
    if(new_wp > MAX_ENTRIES) {
      // have to wrap around
      if(read_pointer <= entries_needed)
	return 0;

      // fill remaining entries with 0's so reader skips over them
      while(write_pointer < MAX_ENTRIES)
	data[write_pointer++] = 0;

      write_pointer = 0;
    } else {
      // if the write pointer would cross over the read pointer, we have to wait
      if((write_pointer < read_pointer) && (new_wp >= read_pointer))
	return 0;

      // special case: if the write pointer would wrap and read is at 0, that'd
      //  be a collision too
      if((new_wp == MAX_ENTRIES) && (read_pointer == 0))
	return 0;
    }

    // all good - return a pointer to the first available entry
    return (data + write_pointer);
  }

  void AddressList::commit_nd_entry(int act_dim, size_t bytes)
  {
    size_t entries_used = act_dim * 2;

    write_pointer += entries_used;
    if(write_pointer >= MAX_ENTRIES) {
      assert(write_pointer == MAX_ENTRIES);
      write_pointer = 0;
    }

    total_bytes += bytes;
  }

  size_t AddressList::bytes_pending() const
  {
    return total_bytes;
  }

  const size_t *AddressList::read_entry()
  {
    assert(total_bytes > 0);
    if(read_pointer >= MAX_ENTRIES) {
      assert(read_pointer == MAX_ENTRIES);
      read_pointer = 0;
    }
    // skip trailing 0's
    if(data[read_pointer] == 0)
      read_pointer = 0;
    return (data + read_pointer);
  }
	 

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressListCursor
  //

  AddressListCursor::AddressListCursor()
    : addrlist(0)
    , partial(false)
  {
    for(int i = 0; i < MAX_DIM; i++)
      pos[i] = 0;
  }

  void AddressListCursor::set_addrlist(AddressList *_addrlist)
  {
    addrlist = _addrlist;
  }

  int AddressListCursor::get_dim() const
  {
    assert(addrlist);
    // with partial progress, we restrict ourselves to just the rest of that dim
    if(partial) {
      return (partial_dim + 1);
    } else {
      const size_t *entry = addrlist->read_entry();
      int act_dim = (entry[0] & 15);
      return act_dim;
    }
  }

  uintptr_t AddressListCursor::get_offset() const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    uintptr_t ofs = entry[1];
    if(partial) {
      for(int i = partial_dim; i < act_dim; i++)
	if(i == 0) {
	  // dim 0 is counted in bytes
	  ofs += pos[0];
	} else {
	  // rest use the strides from the address list
	  ofs += pos[i] * entry[1 + (2 * i)];
	}
    }
    return ofs;
  }

  uintptr_t AddressListCursor::get_stride(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    assert((dim > 0) && (dim < act_dim));
    return entry[2 * dim + 1];
  }

  size_t AddressListCursor::remaining(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    assert(dim < act_dim);
    size_t r = entry[2 * dim];
    if(dim == 0) r >>= 4;
    if(partial) {
      if(dim > partial_dim) r = 1;
      if(dim == partial_dim) {
	assert(r > pos[dim]);
	r -= pos[dim];
      }
    }
    return r;
  }

  void AddressListCursor::advance(int dim, size_t amount)
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    assert(dim < act_dim);
    size_t r = entry[2 * dim];
    if(dim == 0) r >>= 4;

    size_t bytes = amount;
    if(dim > 0) {
#ifdef DEBUG_REALM
      for(int i = 0; i < dim; i++)
	assert(pos[i] == 0);
#endif
      bytes *= (entry[0] >> 4);
      for(int i = 1; i < dim; i++)
	bytes *= entry[2 * i];
    }
#ifdef DEBUG_REALM
    assert(addrlist->total_bytes >= bytes);
#endif
    addrlist->total_bytes -= bytes;
    
    if(!partial) {
      if((dim == (act_dim - 1)) && (amount == r)) {
	// simple case - we consumed the whole thing
	addrlist->read_pointer += 2 * act_dim;
	return;
      } else {
	// record partial consumption
	partial = true;
	partial_dim = dim;
	pos[partial_dim] = amount;
      }
    } else {
      // update a partial consumption in progress
      assert(dim <= partial_dim);
      partial_dim = dim;
      pos[partial_dim] += amount;
    }

    while(pos[partial_dim] == r) {
      pos[partial_dim++] = 0;
      if(partial_dim == act_dim) {
	// all done
	partial = false;
	addrlist->read_pointer += 2 * act_dim;
	break;
      } else {
	pos[partial_dim]++;  // carry into next dimension
	r = entry[2 * partial_dim]; // no shift because partial_dim > 0
      }
    }
  }

  void AddressListCursor::skip_bytes(size_t bytes)
  {
    while(bytes > 0) {
      int act_dim = get_dim();

      if(act_dim == 0) {
	assert(0);
      } else {
	size_t chunk = remaining(0);
	if(chunk <= bytes) {
	  int dim = 0;
	  size_t count = chunk;
	  while((dim + 1) < act_dim) {
	    dim++;
	    count = bytes / chunk;
	    assert(count > 0);
	    size_t r = remaining(dim + 1);
	    if(count < r) {
	      chunk *= count;
	      break;
	    } else {
	      count = r;
	      chunk *= count;
	    }
	  }
	  advance(dim, count);
	  bytes -= chunk;
	} else {
	  advance(0, bytes);
	  return;
	}
      }
    }
  }

  std::ostream& operator<<(std::ostream& os, const AddressListCursor& alc)
  {
    os << alc.remaining(0);
    for(int i = 1; i < alc.get_dim(); i++)
      os << 'x' << alc.remaining(i);
    os << ',' << alc.get_offset();
    for(int i = 1; i < alc.get_dim(); i++)
      os << '+' << alc.get_stride(i);
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ControlPort::Encoder
  //

  ControlPort::Encoder::Encoder()
    : port_shift(0)
    , state(STATE_INIT)
  {}

  ControlPort::Encoder::~Encoder()
  {
    assert(state == STATE_DONE);
  }

  void ControlPort::Encoder::set_port_count(size_t ports)
  {
    assert(state == STATE_INIT);
    // we add one to valid port indices, so we need to encode values in [0,ports]
    port_shift = 1;
    while((ports >> port_shift) > 0) {
      port_shift++;
      assert(port_shift <= 30); // 1B ports will be bad for other reasons too...
    }
    state = STATE_HAVE_PORT_COUNT;
  }

  // encodes some/all of the { count, port, last } packet into the next
  //  32b - returns true if encoding is complete or false if it should
  //  be called again with the same arguments for another 32b packet
  bool ControlPort::Encoder::encode(unsigned& data,
                                    size_t count, int port, bool last)
  {
    unsigned port_p1 = port + 1;
    assert((port_p1 >> port_shift) == 0);

    switch(state) {
    case STATE_INIT:
      assert(0 && "encoding control word without known port count");

    case STATE_HAVE_PORT_COUNT:
      {
        // special case - if we're sending a single packet with count=0,last=1,
        //  we don't need to send the port shift first
        if((count == 0) && last) {
          data = 0;
          state = STATE_DONE;
          log_xd.print() << "encode: " << count << " " << port << " " << last;
          return true;
        } else {
          data = port_shift;
          state = STATE_IDLE;
          return false;
        }
      }

    case STATE_IDLE:
      {
        // figure out if we need 1, 2, or 3 chunks for this
        unsigned mid = (count >> (30 - port_shift));
        unsigned hi = ((sizeof(size_t) > 4) ? (count >> (60 - port_shift)) : 0);

        if(hi != 0) {
          // will take three words - send HIGH first
          data = (hi << 2) | CTRL_HIGH;
          state = STATE_SENT_HIGH;
          return false;
        } else if(mid != 0) {
          // will take two words - send MID first
          data = (mid << 2) | CTRL_MID;
          state = STATE_SENT_MID;
          return false;
        } else {
          // fits in a single word
          data = ((count << (port_shift + 2)) |
                  (port_p1 << 2) |
                  (last ? CTRL_LO_LAST : CTRL_LO_MORE));
          state = (last ? STATE_DONE : STATE_IDLE);
          //log_xd.print() << "encode: " << count << " " << port << " " << last;
          return true;
        }
      }

    case STATE_SENT_HIGH:
      {
        // since we just sent HIGH, must send MID next
        unsigned mid = (count >> (30 - port_shift));
        data = (mid << 2) | CTRL_MID;
        state = STATE_SENT_MID;
        return false;
      }

    case STATE_SENT_MID:
      {
        // since we just sent MID, send LO to finish
        data = ((count << (port_shift + 2)) |
                (port_p1 << 2) |
                (last ? CTRL_LO_LAST : CTRL_LO_MORE));
        state = (last ? STATE_DONE : STATE_IDLE);
        //log_xd.print() << "encode: " << count << " " << port << " " << last;
        return true;
      }

    case STATE_DONE:
      assert(0 && "sending after last?");
    }

    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ControlPort::Decoder
  //

  ControlPort::Decoder::Decoder()
    : temp_count(0)
    , port_shift(0)
  {}

  ControlPort::Decoder::~Decoder()
  {
    // shouldn't end with a partial count
    assert(temp_count == 0);
  }

  // decodes the next 32b of packed data, returning true if a complete
  //  { count, port, last } has been received
  bool ControlPort::Decoder::decode(unsigned data,
                                    size_t& count, int& port, bool& last)
  {
    if(port_shift == 0) {
      // we haven't received the port shift yet, so it's either this or a 0
      //  meaning there's no data at all
      if(data != 0) {
        port_shift = data;
        return false;
      } else {
        count = 0;
        port = -1;
        last = true;
        //log_xd.print() << "decode: " << count << " " << port << " " << last;
        return true;
      }
    } else {
      // bottom 2 bits tell us the chunk type
      unsigned ctrl = data & 3;

      if(ctrl == CTRL_HIGH) {
        assert(temp_count == 0);  // should not be clobbering an existing count
        temp_count = size_t(data >> 2) << (60 - port_shift);
        assert(temp_count != 0);  // should not have gotten HIGH with 0 value
        return false;
      } else if(ctrl == CTRL_MID) {
        temp_count |= size_t(data >> 2) << (30 - port_shift);
        assert(temp_count != 0);  // must have gotten HIGH or nonzero here
        return false;
      } else {
        // LO means we have a full control packet
        count = temp_count | (data >> (port_shift + 2));
        unsigned port_p1 = (data >> 2) & ((1U << port_shift) - 1);
        port = port_p1 - 1;
        last = (ctrl == CTRL_LO_LAST);
        temp_count = 0;
        //log_xd.print() << "decode: " << count << " " << port << " " << last;
        return true;
      }
    }
  }


      XferDes::XferDes(uintptr_t _dma_op, Channel *_channel,
		       NodeID _launch_node, XferDesID _guid,
		       const std::vector<XferDesPortInfo>& inputs_info,
		       const std::vector<XferDesPortInfo>& outputs_info,
		       int _priority,
                       const void *_fill_data, size_t _fill_size)
        : dma_op(_dma_op),
	  xferDes_queue(XferDesQueue::get_singleton()),
	  launch_node(_launch_node),
	  iteration_completed(false),
          bytes_write_pending(0),
	  transfer_completed(false),
          max_req_size(16 << 20 /*TO REMOVE*/), priority(_priority),
          guid(_guid),
          channel(_channel),
          fill_data(&inline_fill_storage),
          fill_size(_fill_size),
          orig_fill_size(_fill_size),
	  progress_counter(0), reference_count(1)
      {
	input_ports.resize(inputs_info.size());
	int gather_control_port = -1;
	int scatter_control_port = -1;
	for(size_t i = 0; i < inputs_info.size(); i++) {
	  XferPort& p = input_ports[i];
	  const XferDesPortInfo& ii = inputs_info[i];

	  p.mem = get_runtime()->get_memory_impl(ii.mem);
	  p.iter = ii.iter;
	  if(ii.serdez_id != 0) {
	    const CustomSerdezUntyped *op = get_runtime()->custom_serdez_table.get(ii.serdez_id, 0);
	    assert(op != 0);
	    p.serdez_op = op;
	  } else
	    p.serdez_op = 0;
	  p.peer_guid = ii.peer_guid;
	  p.peer_port_idx = ii.peer_port_idx;
	  p.indirect_port_idx = ii.indirect_port_idx;
	  p.is_indirect_port = false;  // we'll set these below as needed
	  p.needs_pbt_update.store(false);  // never needed for inputs
	  p.local_bytes_total = 0;
	  p.local_bytes_cons.store(0);
	  p.remote_bytes_total.store(size_t(-1));
	  p.ib_offset = ii.ib_offset;
	  p.ib_size = ii.ib_size;
	  p.addrcursor.set_addrlist(&p.addrlist);
	  switch(ii.port_type) {
	  case XferDesPortInfo::GATHER_CONTROL_PORT:
	    gather_control_port = i; break;
	  case XferDesPortInfo::SCATTER_CONTROL_PORT:
	    scatter_control_port = i; break;
	  default: break;
	  }
	}
	// connect up indirect input ports in a second pass
	for(size_t i = 0; i < inputs_info.size(); i++) {
	  XferPort& p = input_ports[i];
	  if(p.indirect_port_idx >= 0) {
	    p.iter->set_indirect_input_port(this, p.indirect_port_idx,
					    input_ports[p.indirect_port_idx].iter);
	    input_ports[p.indirect_port_idx].is_indirect_port = true;
	  }
	}
	if(gather_control_port >= 0) {
	  input_control.control_port_idx = gather_control_port;
	  input_control.current_io_port = 0;
	  input_control.remaining_count = 0;
	  input_control.eos_received = false;
	} else {
	  input_control.control_port_idx = -1;
	  input_control.current_io_port = 0;
	  input_control.remaining_count = size_t(-1);
	  input_control.eos_received = false;
	}

	output_ports.resize(outputs_info.size());
	for(size_t i = 0; i < outputs_info.size(); i++) {
	  XferPort& p = output_ports[i];
	  const XferDesPortInfo& oi = outputs_info[i];

	  p.mem = get_runtime()->get_memory_impl(oi.mem);
	  p.iter = oi.iter;
	  if(oi.serdez_id != 0) {
	    const CustomSerdezUntyped *op = get_runtime()->custom_serdez_table.get(oi.serdez_id, 0);
	    assert(op != 0);
	    p.serdez_op = op;
	  } else
	    p.serdez_op = 0;
	  p.peer_guid = oi.peer_guid;
	  p.peer_port_idx = oi.peer_port_idx;
	  p.indirect_port_idx = oi.indirect_port_idx;
	  p.is_indirect_port = false;  // outputs are never indirections
	  if(oi.indirect_port_idx >= 0) {
	    p.iter->set_indirect_input_port(this, oi.indirect_port_idx,
					    inputs_info[oi.indirect_port_idx].iter);
	    input_ports[p.indirect_port_idx].is_indirect_port = true;
	  }
	  // TODO: further refine this to exclude peers that can figure out
	  //  the end of a tranfer some othe way
	  p.needs_pbt_update.store(oi.peer_guid != XFERDES_NO_GUID);
	  p.local_bytes_total = 0;
	  p.local_bytes_cons.store(0);
	  p.remote_bytes_total.store(size_t(-1));
	  p.ib_offset = oi.ib_offset;
	  p.ib_size = oi.ib_size;
	  p.addrcursor.set_addrlist(&p.addrlist);

	  // if we're writing into an IB, the first 'ib_size' byte
	  //  locations can be freely written
	  if(p.ib_size > 0)
	    p.seq_remote.add_span(0, p.ib_size);
	}

	if(scatter_control_port >= 0) {
	  output_control.control_port_idx = scatter_control_port;
	  output_control.current_io_port = 0;
	  output_control.remaining_count = 0;
	  output_control.eos_received = false;
	} else {
	  output_control.control_port_idx = -1;
	  output_control.current_io_port = 0;
	  output_control.remaining_count = size_t(-1);
	  output_control.eos_received = false;
	}

        // allocate a larger buffer if needed for fill data
        if(fill_size > ALIGNED_FILL_STORAGE_SIZE) {
          fill_data = malloc(fill_size);
          assert(fill_data);
        }
        if(fill_size > 0)
          memcpy(fill_data, _fill_data, fill_size);
      }

      XferDes::~XferDes() {
        // clear available_reqs
        while (!available_reqs.empty()) {
          available_reqs.pop();
        }
	for(std::vector<XferPort>::const_iterator it = input_ports.begin();
	    it != input_ports.end();
	    ++it)
	  delete it->iter;
	for(std::vector<XferPort>::const_iterator it = output_ports.begin();
	    it != output_ports.end();
	    ++it)
	  delete it->iter;

        if(fill_data != &inline_fill_storage)
          free(fill_data);
      };

      Event XferDes::request_metadata()
      {
	std::vector<Event> preconditions;
	for(std::vector<XferPort>::iterator it = input_ports.begin();
	    it != input_ports.end();
	    ++it) {
	  Event e = it->iter->request_metadata();
	  if(!e.has_triggered())
	    preconditions.push_back(e);
	}
	for(std::vector<XferPort>::iterator it = output_ports.begin();
	    it != output_ports.end();
	    ++it) {
	  Event e = it->iter->request_metadata();
	  if(!e.has_triggered())
	    preconditions.push_back(e);
	}
	return Event::merge_events(preconditions);
      }
  
      void XferDes::mark_completed() {
	for(std::vector<XferPort>::const_iterator it = input_ports.begin();
	    it != input_ports.end();
	    ++it)
	  if(it->ib_size > 0)
	    free_intermediate_buffer(it->mem->me,
				     it->ib_offset,
				     it->ib_size);

        // notify owning DmaRequest upon completion of this XferDes
        //printf("complete XD = %lu\n", guid);
        if (launch_node == Network::my_node_id) {
	  TransferOperation *op = reinterpret_cast<TransferOperation *>(dma_op);
	  op->notify_xd_completion(guid);
        } else {
	  TransferOperation *op = reinterpret_cast<TransferOperation *>(dma_op);
	  NotifyXferDesCompleteMessage::send_request(launch_node, op, guid);
        }
      }

#if 0
      static inline off_t calc_mem_loc_ib(off_t alloc_offset,
                                          off_t field_start,
                                          int field_size,
                                          size_t elmt_size,
                                          size_t block_size,
                                          size_t buf_size,
                                          size_t domain_size,
                                          off_t index)
      {
        off_t idx2 = domain_size / block_size * block_size;
        off_t offset;
        if (index < idx2) {
          offset = Realm::calc_mem_loc(alloc_offset, field_start, field_size, elmt_size, block_size, index);
        } else {
          offset = (alloc_offset + field_start * domain_size + (elmt_size - field_start) * idx2 + (index - idx2) * field_size);
        }
	// the final step is to wrap the offset around within the buf_size
	//  (i.e. offset %= buf_size), but that is done by the caller after
	//  checking flow control limits
        return offset;
      }
#endif

#define MAX_GEN_REQS 3

      bool support_2d_xfers(XferDesKind kind)
      {
        return (kind == XFER_GPU_TO_FB)
               || (kind == XFER_GPU_FROM_FB)
               || (kind == XFER_GPU_IN_FB)
               || (kind == XFER_GPU_PEER_FB)
               || (kind == XFER_REMOTE_WRITE)
               || (kind == XFER_MEM_CPY);
      }

  size_t XferDes::update_control_info(ReadSequenceCache *rseqcache)
  {
    if(iteration_completed.load_acquire())
      return 0;

    // pull control information if we need it
    if(input_control.remaining_count == 0) {
      if(input_control.control_port_idx >= 0) {
        XferPort& icp = input_ports[input_control.control_port_idx];
        size_t avail = icp.seq_remote.span_exists(icp.local_bytes_total,
                                                  4 * sizeof(unsigned));
        size_t old_lbt = icp.local_bytes_total;

        // may take a few chunks of data to get a control packet
        while(true) {
          if(avail < sizeof(unsigned))
            return 0;  // no data right now

          TransferIterator::AddressInfo c_info;
          size_t amt = icp.iter->step(sizeof(unsigned), c_info, 0,
                                      false /*!tentative*/);
          assert(amt == sizeof(unsigned));
          const void *srcptr = icp.mem->get_direct_ptr(c_info.base_offset, amt);
          assert(srcptr != 0);
          unsigned cword;
          memcpy(&cword, srcptr, sizeof(unsigned));

          icp.local_bytes_total += sizeof(unsigned);
          avail -= sizeof(unsigned);

          if(input_control.decoder.decode(cword,
                                          input_control.remaining_count,
                                          input_control.current_io_port,
                                          input_control.eos_received))
            break;
        }

        // can't get here unless we read something, so ack it
        if(rseqcache != 0)
          rseqcache->add_span(input_control.control_port_idx,
                              old_lbt, icp.local_bytes_total - old_lbt);
        else
          update_bytes_read(input_control.control_port_idx,
                            old_lbt, icp.local_bytes_total - old_lbt);

        log_xd.info() << "input control: xd=" << std::hex << guid << std::dec
                      << " port=" << input_control.current_io_port
                      << " count=" << input_control.remaining_count
                      << " done=" << input_control.eos_received;
      }
      // if count is still zero, we're done
      if(input_control.remaining_count == 0) {
	assert(input_control.eos_received);
	begin_completion();
	return 0;
      }
    }

    if(output_control.remaining_count == 0) {
      if(output_control.control_port_idx >= 0) {
        // this looks wrong, but the port that controls the output is
        //  an input port! vvv
        XferPort& ocp = input_ports[output_control.control_port_idx];
        size_t avail = ocp.seq_remote.span_exists(ocp.local_bytes_total,
                                                  4 * sizeof(unsigned));
        size_t old_lbt = ocp.local_bytes_total;

        // may take a few chunks of data to get a control packet
        while(true) {
          if(avail < sizeof(unsigned))
            return 0;  // no data right now

          TransferIterator::AddressInfo c_info;
          size_t amt = ocp.iter->step(sizeof(unsigned), c_info, 0,
                                      false /*!tentative*/);
          assert(amt == sizeof(unsigned));
          const void *srcptr = ocp.mem->get_direct_ptr(c_info.base_offset, amt);
          assert(srcptr != 0);
          unsigned cword;
          memcpy(&cword, srcptr, sizeof(unsigned));

          ocp.local_bytes_total += sizeof(unsigned);
          avail -= sizeof(unsigned);

          if(output_control.decoder.decode(cword,
                                           output_control.remaining_count,
                                           output_control.current_io_port,
                                           output_control.eos_received))
            break;
        }

        // can't get here unless we read something, so ack it
        if(rseqcache != 0)
          rseqcache->add_span(output_control.control_port_idx,
                              old_lbt, ocp.local_bytes_total - old_lbt);
        else
          update_bytes_read(output_control.control_port_idx,
                            old_lbt, ocp.local_bytes_total - old_lbt);

        log_xd.info() << "output control: xd=" << std::hex << guid << std::dec
                      << " port=" << output_control.current_io_port
                      << " count=" << output_control.remaining_count
                      << " done=" << output_control.eos_received;
      }
      // if count is still zero, we're done
      if(output_control.remaining_count == 0) {
	assert(output_control.eos_received);
	begin_completion();
	return 0;
      }
    }

    return std::min(input_control.remaining_count,
		    output_control.remaining_count);
  }
  
  size_t XferDes::get_addresses(size_t min_xfer_size,
				ReadSequenceCache *rseqcache)
  {
    const InstanceLayoutPieceBase *in_nonaffine;
    const InstanceLayoutPieceBase *out_nonaffine;
    size_t ret = get_addresses(min_xfer_size, rseqcache,
                               in_nonaffine, out_nonaffine);
    assert(!in_nonaffine && !out_nonaffine);
    return ret;
  }

  size_t XferDes::get_addresses(size_t min_xfer_size,
				ReadSequenceCache *rseqcache,
                                const InstanceLayoutPieceBase *&in_nonaffine,
                                const InstanceLayoutPieceBase *&out_nonaffine)
  {
    size_t control_count = update_control_info(rseqcache);
    if(control_count == 0) {
      in_nonaffine = 0;
      out_nonaffine = 0;
      return 0;
    }
    if(control_count < min_xfer_size)
      min_xfer_size = control_count;
    size_t max_bytes = control_count;

    // get addresses for the input, if it exists
    if(input_control.current_io_port >= 0) {
      XferPort *in_port = &input_ports[input_control.current_io_port];

      // do we need more addresses?
      size_t read_bytes_avail = in_port->addrlist.bytes_pending();
      if(read_bytes_avail < min_xfer_size) {
        bool flush = in_port->iter->get_addresses(in_port->addrlist,
                                                  in_nonaffine);
	read_bytes_avail = in_port->addrlist.bytes_pending();
        if(flush) {
          if(read_bytes_avail > 0) {
            // ignore a nonaffine piece as we still have some affine bytes
            in_nonaffine = 0;
          }

	  // adjust min size to flush as requested (unless we're non-affine)
          if(!in_nonaffine)
            min_xfer_size = std::min(min_xfer_size, read_bytes_avail);
	}
      } else
        in_nonaffine = 0;

      // if we're not the first in the chain, respect flow control too
      if(in_port->peer_guid != XFERDES_NO_GUID) {
	read_bytes_avail = in_port->seq_remote.span_exists(in_port->local_bytes_total,
							   read_bytes_avail);
	size_t pbt_limit = (in_port->remote_bytes_total.load_acquire() -
			    in_port->local_bytes_total);
	min_xfer_size = std::min(min_xfer_size, pbt_limit);

        // don't ever expect to be able to read more than half the size of the
        //  incoming intermediate buffer
        if(min_xfer_size > (in_port->ib_size >> 1))
          min_xfer_size = std::max<size_t>(1, (in_port->ib_size >> 1));
      }

      // we'd like to wait until there's `min_xfer_size` bytes available on the
      //  input, but in gather copies with fork-joins in the dataflow, we
      //  can't be guaranteed that's possible, so move whatever we've got,
      //  relying on the upstream producer to be producing it in the largest
      //  chunks it can
      if((read_bytes_avail > 0) && (read_bytes_avail < min_xfer_size))
	min_xfer_size = read_bytes_avail;

      if(!in_nonaffine)
        max_bytes = std::min(max_bytes, read_bytes_avail);
    } else {
      in_nonaffine = 0;
    }

    // get addresses for the output, if it exists
    if(output_control.current_io_port >= 0) {
      XferPort *out_port = &output_ports[output_control.current_io_port];

      // do we need more addresses?
      size_t write_bytes_avail = out_port->addrlist.bytes_pending();
      if(write_bytes_avail < min_xfer_size) {
	bool flush = out_port->iter->get_addresses(out_port->addrlist,
                                                   out_nonaffine);
	write_bytes_avail = out_port->addrlist.bytes_pending();
        if(flush) {
          if(write_bytes_avail > 0) {
            // ignore a nonaffine piece as we still have some affine bytes
            out_nonaffine = 0;
          }

	  // adjust min size to flush as requested (unless we're non-affine)
          if(!out_nonaffine)
            min_xfer_size = std::min(min_xfer_size, write_bytes_avail);
	}
      } else
        out_nonaffine = 0;

      // if we're not the last in the chain, respect flow control too
      if(out_port->peer_guid != XFERDES_NO_GUID) {
	write_bytes_avail = out_port->seq_remote.span_exists(out_port->local_bytes_total,
							     write_bytes_avail);

        // we'd like to wait until there's `min_xfer_size` bytes available on
        //  the output, but if we're landing in an intermediate buffer and need
        //  to wrap around, waiting won't do any good
        if(min_xfer_size > (out_port->ib_size >> 1))
          min_xfer_size = std::max<size_t>(1, (out_port->ib_size >> 1));
      }

      if(!out_nonaffine)
        max_bytes = std::min(max_bytes, write_bytes_avail);
    } else {
      out_nonaffine = 0;
    }

    if(min_xfer_size == 0) {
      // should only happen in the absence of control ports
      assert((input_control.control_port_idx == -1) &&
	     (output_control.control_port_idx == -1));
      begin_completion();
      return 0;
    }

    // if we don't have a big enough chunk, wait for more to show up
    if((max_bytes < min_xfer_size) && !in_nonaffine && !out_nonaffine)
      return 0;

    return max_bytes;
  }

  bool XferDes::record_address_consumption(size_t total_read_bytes,
                                           size_t total_write_bytes)
  {
    bool in_done = false;
    if(input_control.current_io_port >= 0) {
      XferPort *in_port = &input_ports[input_control.current_io_port];

      in_port->local_bytes_total += total_read_bytes;
      in_port->local_bytes_cons.fetch_add(total_read_bytes);

      if(in_port->peer_guid == XFERDES_NO_GUID)
	in_done = ((in_port->addrlist.bytes_pending() == 0) &&
		   in_port->iter->done());
      else
	in_done = (in_port->local_bytes_total ==
		   in_port->remote_bytes_total.load_acquire());
    }

    bool out_done = false;
    if(output_control.current_io_port >= 0) {
      XferPort *out_port = &output_ports[output_control.current_io_port];

      out_port->local_bytes_total += total_write_bytes;
      out_port->local_bytes_cons.fetch_add(total_write_bytes);

      if(out_port->peer_guid == XFERDES_NO_GUID)
	out_done = ((out_port->addrlist.bytes_pending() == 0) &&
		    out_port->iter->done());
    }
	  
    input_control.remaining_count -= total_read_bytes;
    output_control.remaining_count -= total_write_bytes;

    // input or output controls override our notion of done-ness
    if(input_control.control_port_idx >= 0)
      in_done = ((input_control.remaining_count == 0) &&
		 input_control.eos_received);

    if(output_control.control_port_idx >= 0)
      out_done = ((output_control.remaining_count == 0) &&
		  output_control.eos_received);
	  
    if(in_done || out_done) {
      begin_completion();
      return true;
    } else
      return false;
  }

  void XferDes::replicate_fill_data(size_t new_size)
  {
    if(new_size > fill_size) {
#ifdef DEBUG_REALM
      assert((fill_size > 0) && ((new_size % orig_fill_size) == 0));
#endif
      char *new_fill_data;
      if(new_size > ALIGNED_FILL_STORAGE_SIZE) {
        new_fill_data = (char *)malloc(new_size);
        assert(new_fill_data);
        memcpy(new_fill_data, fill_data, fill_size /*old size*/);
      } else {
        // can still fit in the inline storage, so no bootstrap copy needed
        new_fill_data = (char *)&inline_fill_storage;
      }
      do {
        // can't increase by more than 2x per copy
        size_t to_copy = std::min(new_size - fill_size, fill_size);
        memcpy(new_fill_data + fill_size, new_fill_data, to_copy);
        fill_size += to_copy;
      } while(fill_size < new_size);

      // delete old buffer, if it was allocated
      if(fill_data != &inline_fill_storage)
        free(fill_data);

      fill_data = new_fill_data;
    }
  }

    long XferDes::default_get_requests(Request** reqs, long nr,
				       unsigned flags)
      {
        long idx = 0;
	
	while((idx < nr) && request_available()) {
	  // TODO: we really shouldn't even be trying if the iteration
	  //   is already done
	  if(iteration_completed.load()) break;

	  // pull control information if we need it
	  if(input_control.remaining_count == 0) {
	    XferPort& icp = input_ports[input_control.control_port_idx];
	    size_t avail = icp.seq_remote.span_exists(icp.local_bytes_total,
						      4 * sizeof(unsigned));
            size_t old_lbt = icp.local_bytes_total;

            // may take a few chunks of data to get a control packet
            bool got_packet = false;
            do {
              if(avail < sizeof(unsigned))
                break;  // no data right now

              TransferIterator::AddressInfo c_info;
              size_t amt = icp.iter->step(sizeof(unsigned), c_info, 0,
                                          false /*!tentative*/);
              assert(amt == sizeof(unsigned));
              const void *srcptr = icp.mem->get_direct_ptr(c_info.base_offset, amt);
              assert(srcptr != 0);
              unsigned cword;
              memcpy(&cword, srcptr, sizeof(unsigned));

              icp.local_bytes_total += sizeof(unsigned);
              avail -= sizeof(unsigned);

              got_packet = input_control.decoder.decode(cword,
                                                        input_control.remaining_count,
                                                        input_control.current_io_port,
                                                        input_control.eos_received);
            } while(!got_packet);

            // can't make further progress if we didn't get a full packet
            if(!got_packet)
              break;

            update_bytes_read(input_control.control_port_idx,
                              old_lbt, icp.local_bytes_total - old_lbt);

	    log_xd.info() << "input control: xd=" << std::hex << guid << std::dec
			  << " port=" << input_control.current_io_port
			  << " count=" << input_control.remaining_count
			  << " done=" << input_control.eos_received;
	    // if count is still zero, we're done
	    if(input_control.remaining_count == 0) {
	      assert(input_control.eos_received);
	      begin_completion();
	      break;
	    }
	  }
	  if(output_control.remaining_count == 0) {
	    // this looks wrong, but the port that controls the output is
	    //  an input port! vvv
	    XferPort& ocp = input_ports[output_control.control_port_idx];
	    size_t avail = ocp.seq_remote.span_exists(ocp.local_bytes_total,
						      4 * sizeof(unsigned));
            size_t old_lbt = ocp.local_bytes_total;

            // may take a few chunks of data to get a control packet
            bool got_packet = false;
            do {
              if(avail < sizeof(unsigned))
                break;  // no data right now

              TransferIterator::AddressInfo c_info;
              size_t amt = ocp.iter->step(sizeof(unsigned), c_info, 0, false /*!tentative*/);
              assert(amt == sizeof(unsigned));
              const void *srcptr = ocp.mem->get_direct_ptr(c_info.base_offset, amt);
              assert(srcptr != 0);
              unsigned cword;
              memcpy(&cword, srcptr, sizeof(unsigned));

              ocp.local_bytes_total += sizeof(unsigned);
              avail -= sizeof(unsigned);

              got_packet = output_control.decoder.decode(cword,
                                                         output_control.remaining_count,
                                                         output_control.current_io_port,
                                                         output_control.eos_received);
            } while(!got_packet);

            // can't make further progress if we didn't get a full packet
            if(!got_packet)
              break;

	    update_bytes_read(output_control.control_port_idx,
                              old_lbt, ocp.local_bytes_total - old_lbt);

	    log_xd.info() << "output control: xd=" << std::hex << guid << std::dec
			  << " port=" << output_control.current_io_port
			  << " count=" << output_control.remaining_count
			  << " done=" << output_control.eos_received;
	    // if count is still zero, we're done
	    if(output_control.remaining_count == 0) {
	      assert(output_control.eos_received);
	      begin_completion();
	      break;
	    }
	  }

	  XferPort *in_port = ((input_control.current_io_port >= 0) ?
			         &input_ports[input_control.current_io_port] :
			         0);
	  XferPort *out_port = ((output_control.current_io_port >= 0) ?
				  &output_ports[output_control.current_io_port] :
				  0);

	  // special cases for OOR scatter/gather
	  if(!in_port) {
	    if(!out_port) {
	      // no input or output?  just skip the count?
	      assert(0);
	    } else {
	      // no valid input, so no write to the destination -
	      //  just step the output transfer iterator if it's a real target
	      //  but barf if it's an IB
	      assert((out_port->peer_guid == XferDes::XFERDES_NO_GUID) &&
		     !out_port->serdez_op);
	      TransferIterator::AddressInfo dummy;
	      size_t skip_bytes = out_port->iter->step(std::min(input_control.remaining_count,
								output_control.remaining_count),
						       dummy,
						       flags & TransferIterator::DST_FLAGMASK,
						       false /*!tentative*/);
	      log_xd.debug() << "skipping " << skip_bytes << " bytes of output";
	      assert(skip_bytes > 0);
	      input_control.remaining_count -= skip_bytes;
	      output_control.remaining_count -= skip_bytes;
	      // TODO: pull this code out to a common place?
	      if(((input_control.remaining_count == 0) && input_control.eos_received) ||
		 ((output_control.remaining_count == 0) && output_control.eos_received)) {
		log_xd.info() << "iteration completed via control port: xd=" << std::hex << guid << std::dec;
		begin_completion();
		break;
	      }
	      continue;  // try again
	    }
	  } else if(!out_port) {
	    // valid input that we need to throw away
	    assert(!in_port->serdez_op);
	    TransferIterator::AddressInfo dummy;
	    // although we're not reading the IB input data ourselves, we need
	    //  to wait until it's ready before not-reading it to avoid WAW
	    //  races on the producer side
	    size_t skip_bytes = std::min(input_control.remaining_count,
					 output_control.remaining_count);
	    if(in_port->peer_guid != XferDes::XFERDES_NO_GUID) {
	      skip_bytes = in_port->seq_remote.span_exists(in_port->local_bytes_total,
							   skip_bytes);
	      if(skip_bytes == 0) break;
	    }
	    skip_bytes = in_port->iter->step(skip_bytes,
					     dummy,
					     flags & TransferIterator::SRC_FLAGMASK,
					     false /*!tentative*/);
	    log_xd.debug() << "skipping " << skip_bytes << " bytes of input";
	    assert(skip_bytes > 0);
	    update_bytes_read(input_control.current_io_port,
			      in_port->local_bytes_total,
			      skip_bytes);
	    in_port->local_bytes_total += skip_bytes;
	    input_control.remaining_count -= skip_bytes;
	    output_control.remaining_count -= skip_bytes;
	    // TODO: pull this code out to a common place?
	    if(((input_control.remaining_count == 0) && input_control.eos_received) ||
	       ((output_control.remaining_count == 0) && output_control.eos_received)) {
	      log_xd.info() << "iteration completed via control port: xd=" << std::hex << guid << std::dec;
	      begin_completion();
	      break;
	    }
	    continue;  // try again
	  }
	  
	  // there are several variables that can change asynchronously to
	  //  the logic here:
	  //   pre_bytes_total - the max bytes we'll ever see from the input IB
	  //   read_bytes_cons - conservative estimate of bytes we've read
	  //   write_bytes_cons - conservative estimate of bytes we've written
	  //
	  // to avoid all sorts of weird race conditions, sample all three here
	  //  and only use them in the code below (exception: atomic increments
	  //  of rbc or wbc, for which we adjust the snapshot by the same)
	  size_t pbt_snapshot = in_port->remote_bytes_total.load_acquire();
	  size_t rbc_snapshot = in_port->local_bytes_cons.load_acquire();
	  size_t wbc_snapshot = out_port->local_bytes_cons.load_acquire();

	  // normally we detect the end of a transfer after initiating a
	  //  request, but empty iterators and filtered streams can cause us
	  //  to not realize the transfer is done until we are asking for
	  //  the next request (i.e. now)
	  if((in_port->peer_guid == XFERDES_NO_GUID) ?
	       in_port->iter->done() :
	       (in_port->local_bytes_total == pbt_snapshot)) {
	    if(in_port->local_bytes_total == 0)
	      log_request.info() << "empty xferdes: " << guid;
	    // TODO: figure out how to eliminate false positives from these
	    //  checks with indirection and/or multiple remote inputs
#if 0
	    assert((out_port->peer_guid != XFERDES_NO_GUID) ||
		   out_port->iter->done());
#endif

	    begin_completion();
	    break;
	  }
	  
	  TransferIterator::AddressInfo src_info, dst_info;
	  size_t read_bytes, write_bytes, read_seq, write_seq;
	  size_t write_pad_bytes = 0;
	  size_t read_pad_bytes = 0;

	  // handle serialization-only and deserialization-only cases 
	  //  specially, because they have uncertainty in how much data
	  //  they write or read
	  if(in_port->serdez_op && !out_port->serdez_op) {
	    // serialization only - must be into an IB
	    assert(in_port->peer_guid == XFERDES_NO_GUID);
	    assert(out_port->peer_guid != XFERDES_NO_GUID);

	    // when serializing, we don't know how much output space we're
	    //  going to consume, so do not step the dst_iter here
	    // instead, see what we can get from the source and conservatively
	    //  check flow control on the destination and let the stepping
	    //  of dst_iter happen in the actual execution of the request

	    // if we don't have space to write a single worst-case
	    //  element, try again later
	    if(out_port->seq_remote.span_exists(wbc_snapshot,
						in_port->serdez_op->max_serialized_size) <
	       in_port->serdez_op->max_serialized_size)
	      break;

	    size_t max_bytes = max_req_size;

	    size_t src_bytes = in_port->iter->step(max_bytes, src_info,
						   flags & TransferIterator::SRC_FLAGMASK,
						   true /*tentative*/);

	    size_t num_elems = src_bytes / in_port->serdez_op->sizeof_field_type;
	    // no input data?  try again later
	    if(num_elems == 0)
	      break;
	    assert((num_elems * in_port->serdez_op->sizeof_field_type) == src_bytes);
	    size_t max_dst_bytes = num_elems * in_port->serdez_op->max_serialized_size;

	    // if we have an output control, restrict the max number of
	    //  elements
	    if(output_control.control_port_idx >= 0) {
	      if(num_elems > output_control.remaining_count) {
		log_xd.info() << "scatter/serialize clamp: " << num_elems << " -> " << output_control.remaining_count;
		num_elems = output_control.remaining_count;
	      }
	    }

	    size_t clamp_dst_bytes = num_elems * in_port->serdez_op->max_serialized_size;
	    // test for space using our conserative bytes written count
	    size_t dst_bytes_avail = out_port->seq_remote.span_exists(wbc_snapshot,
								      clamp_dst_bytes);

	    if(dst_bytes_avail == max_dst_bytes) {
	      // enough space - confirm the source step
	      in_port->iter->confirm_step();
	    } else {
	      // not enough space - figure out how many elements we can
	      //  actually take and adjust the source step
	      size_t act_elems = dst_bytes_avail / in_port->serdez_op->max_serialized_size;
	      // if there was a remainder in the division, get rid of it
	      dst_bytes_avail = act_elems * in_port->serdez_op->max_serialized_size;
	      size_t new_src_bytes = act_elems * in_port->serdez_op->sizeof_field_type;
	      in_port->iter->cancel_step();
	      src_bytes = in_port->iter->step(new_src_bytes, src_info,
					      flags & TransferIterator::SRC_FLAGMASK,
					 false /*!tentative*/);
	      // this can come up shorter than we expect if the source
	      //  iterator is 2-D or 3-D - if that happens, re-adjust the
	      //  dest bytes again
	      if(src_bytes < new_src_bytes) {
		if(src_bytes == 0) break;

		num_elems = src_bytes / in_port->serdez_op->sizeof_field_type;
		assert((num_elems * in_port->serdez_op->sizeof_field_type) == src_bytes);

		// no need to recheck seq_next_read
		dst_bytes_avail = num_elems * in_port->serdez_op->max_serialized_size;
	      }
	    }

	    // since the dst_iter will be stepped later, the dst_info is a 
	    //  don't care, so copy the source so that lines/planes/etc match
	    //  up
	    dst_info = src_info;

	    read_seq = in_port->local_bytes_total;
	    read_bytes = src_bytes;
	    in_port->local_bytes_total += src_bytes;

	    write_seq = 0; // filled in later
	    write_bytes = dst_bytes_avail;
	    out_port->local_bytes_cons.fetch_add(dst_bytes_avail);
	    wbc_snapshot += dst_bytes_avail;
	  } else
	  if(!in_port->serdez_op && out_port->serdez_op) {
	    // deserialization only - must be from an IB
	    assert(in_port->peer_guid != XFERDES_NO_GUID);
	    assert(out_port->peer_guid == XFERDES_NO_GUID);

	    // when deserializing, we don't know how much input data we need
	    //  for each element, so do not step the src_iter here
	    //  instead, see what the destination wants
	    // if the transfer is still in progress (i.e. pre_bytes_total
	    //  hasn't been set), we have to be conservative about how many
	    //  elements we can get from partial data

	    // input data is done only if we know the limit AND we have all
	    //  the remaining bytes (if any) up to that limit
	    bool input_data_done = ((pbt_snapshot != size_t(-1)) &&
				    ((rbc_snapshot >= pbt_snapshot) ||
				     (in_port->seq_remote.span_exists(rbc_snapshot,
								      pbt_snapshot - rbc_snapshot) ==
				      (pbt_snapshot - rbc_snapshot))));
	    // if we're using an input control and it's not at the end of the
	    //  stream, the above checks may not be precise
	    if((input_control.control_port_idx >= 0) &&
	       !input_control.eos_received)
	      input_data_done = false;

	    // this done-ness overrides many checks based on the conservative
	    //  out_port->serdez_op->max_serialized_size
	    if(!input_data_done) {
	      // if we don't have enough input data for a single worst-case
	      //  element, try again later
	      if((in_port->seq_remote.span_exists(rbc_snapshot,
						  out_port->serdez_op->max_serialized_size) <
		  out_port->serdez_op->max_serialized_size)) {
		break;
	      }
	    }

	    size_t max_bytes = max_req_size;

	    size_t dst_bytes = out_port->iter->step(max_bytes, dst_info,
						    flags & TransferIterator::DST_FLAGMASK,
						    !input_data_done);

	    size_t num_elems = dst_bytes / out_port->serdez_op->sizeof_field_type;
	    if(num_elems == 0) break;
	    assert((num_elems * out_port->serdez_op->sizeof_field_type) == dst_bytes);
	    size_t max_src_bytes = num_elems * out_port->serdez_op->max_serialized_size;
	    // if we have an input control, restrict the max number of
	    //  elements
	    if(input_control.control_port_idx >= 0) {
	      if(num_elems > input_control.remaining_count) {
		log_xd.info() << "gather/deserialize clamp: " << num_elems << " -> " << input_control.remaining_count;
		num_elems = input_control.remaining_count;
	      }
	    }

	    size_t clamp_src_bytes = num_elems * out_port->serdez_op->max_serialized_size;
	    size_t src_bytes_avail;
	    if(input_data_done) {
	      // we're certainty to have all the remaining data, so keep
	      //  the limit at max_src_bytes - we won't actually overshoot
	      //  (unless the serialized data is corrupted)
	      src_bytes_avail = max_src_bytes;
	    } else {
	      // test for space using our conserative bytes read count
	      src_bytes_avail = in_port->seq_remote.span_exists(rbc_snapshot,
								clamp_src_bytes);

	      if(src_bytes_avail == max_src_bytes) {
		// enough space - confirm the dest step
		out_port->iter->confirm_step();
	      } else {
		log_request.info() << "pred limits deserialize: " << max_src_bytes << " -> " << src_bytes_avail;
		// not enough space - figure out how many elements we can
		//  actually read and adjust the dest step
		size_t act_elems = src_bytes_avail / out_port->serdez_op->max_serialized_size;
		// if there was a remainder in the division, get rid of it
		src_bytes_avail = act_elems * out_port->serdez_op->max_serialized_size;
		size_t new_dst_bytes = act_elems * out_port->serdez_op->sizeof_field_type;
		out_port->iter->cancel_step();
		dst_bytes = out_port->iter->step(new_dst_bytes, dst_info,
						 flags & TransferIterator::SRC_FLAGMASK,
						 false /*!tentative*/);
		// this can come up shorter than we expect if the destination
		//  iterator is 2-D or 3-D - if that happens, re-adjust the
		//  source bytes again
		if(dst_bytes < new_dst_bytes) {
		  if(dst_bytes == 0) break;

		  num_elems = dst_bytes / out_port->serdez_op->sizeof_field_type;
		  assert((num_elems * out_port->serdez_op->sizeof_field_type) == dst_bytes);

		  // no need to recheck seq_pre_write
		  src_bytes_avail = num_elems * out_port->serdez_op->max_serialized_size;
		}
	      }
	    }

	    // since the src_iter will be stepped later, the src_info is a 
	    //  don't care, so copy the source so that lines/planes/etc match
	    //  up
	    src_info = dst_info;

	    read_seq = 0; // filled in later
	    read_bytes = src_bytes_avail;
	    in_port->local_bytes_cons.fetch_add(src_bytes_avail);
	    rbc_snapshot += src_bytes_avail;

	    write_seq = out_port->local_bytes_total;
	    write_bytes = dst_bytes;
	    out_port->local_bytes_total += dst_bytes;
	    out_port->local_bytes_cons.store(out_port->local_bytes_total); // completion detection uses this
	  } else {
	    // either no serialization or simultaneous serdez

	    // limit transfer based on the max request size, or the largest
	    //  amount of data allowed by the control port(s)
	    size_t max_bytes = std::min(size_t(max_req_size),
					std::min(input_control.remaining_count,
						 output_control.remaining_count));

	    // if we're not the first in the chain, and we know the total bytes
	    //  written by the predecessor, don't exceed that
	    if(in_port->peer_guid != XFERDES_NO_GUID) {
	      size_t pre_max = pbt_snapshot - in_port->local_bytes_total;
	      if(pre_max == 0) {
		// should not happen with snapshots
		assert(0);
		// due to unsynchronized updates to pre_bytes_total, this path
		//  can happen for an empty transfer reading from an intermediate
		//  buffer - handle it by looping around and letting the check
		//  at the top of the loop notice it the second time around
		if(in_port->local_bytes_total == 0)
		  continue;
		// otherwise, this shouldn't happen - we should detect this case
		//  on the the transfer of those last bytes
		assert(0);
		begin_completion();
		break;
	      }
	      if(pre_max < max_bytes) {
		log_request.info() << "pred limits xfer: " << max_bytes << " -> " << pre_max;
		max_bytes = pre_max;
	      }

	      // further limit by bytes we've actually received
	      max_bytes = in_port->seq_remote.span_exists(in_port->local_bytes_total, max_bytes);
	      if(max_bytes == 0) {
		// TODO: put this XD to sleep until we do have data
		break;
	      }
	    }

	    if(out_port->peer_guid != XFERDES_NO_GUID) {
	      // if we're writing to an intermediate buffer, make sure to not
	      //  overwrite previously written data that has not been read yet
	      max_bytes = out_port->seq_remote.span_exists(out_port->local_bytes_total, max_bytes);
	      if(max_bytes == 0) {
		// TODO: put this XD to sleep until we do have data
		break;
	      }
	    }

	    // tentatively get as much as we can from the source iterator
	    size_t src_bytes = in_port->iter->step(max_bytes, src_info,
						   flags & TransferIterator::SRC_FLAGMASK,
						   true /*tentative*/);
	    if(src_bytes == 0) {
	      // not enough space for even one element
	      // TODO: put this XD to sleep until we do have data
	      break;
	    }

	    // destination step must be tentative for an non-IB source or
	    //  target that might collapse dimensions differently
	    bool dimension_mismatch_possible = (((in_port->peer_guid == XFERDES_NO_GUID) ||
						 (out_port->peer_guid == XFERDES_NO_GUID)) &&
						((flags & TransferIterator::LINES_OK) != 0));

	    size_t dst_bytes = out_port->iter->step(src_bytes, dst_info,
						    flags & TransferIterator::DST_FLAGMASK,
						    dimension_mismatch_possible);
	    if(dst_bytes == 0) {
	      // not enough space for even one element

	      // if this happens when the input is an IB, the output is not,
	      //  and the input doesn't seem to be limited by max_bytes, this
	      //  is (probably?) the case that requires padding on the input
	      //  side
	      if((in_port->peer_guid != XFERDES_NO_GUID) &&
		 (out_port->peer_guid == XFERDES_NO_GUID) &&
		 (src_bytes < max_bytes)) {
		log_xd.info() << "padding input buffer by " << src_bytes << " bytes";
		src_info.bytes_per_chunk = 0;
		src_info.num_lines = 1;
		src_info.num_planes = 1;
		dst_info.bytes_per_chunk = 0;
		dst_info.num_lines = 1;
		dst_info.num_planes = 1;
		read_pad_bytes = src_bytes;
		src_bytes = 0;
		dimension_mismatch_possible = false;
		// src iterator will be confirmed below
		//in_port->iter->confirm_step();
		// dst didn't actually take a step, so we don't need to cancel it
	      } else {
		in_port->iter->cancel_step();
		// TODO: put this XD to sleep until we do have data
		break;
	      }
	    }

	    // does source now need to be shrunk?
	    if(dst_bytes < src_bytes) {
	      // cancel the src step and try to just step by dst_bytes
	      in_port->iter->cancel_step();
	      // this step must still be tentative if a dimension mismatch is
	      //  posisble
	      src_bytes = in_port->iter->step(dst_bytes, src_info,
					      flags & TransferIterator::SRC_FLAGMASK,
					      dimension_mismatch_possible);
	      if(src_bytes == 0) {
		// corner case that should occur only with a destination 
		//  intermediate buffer - no transfer, but pad to boundary
		//  destination wants as long as we're not being limited by
		//  max_bytes
		assert((in_port->peer_guid == XFERDES_NO_GUID) &&
		       (out_port->peer_guid != XFERDES_NO_GUID));
		if(dst_bytes < max_bytes) {
		  log_xd.info() << "padding output buffer by " << dst_bytes << " bytes";
		  src_info.bytes_per_chunk = 0;
		  src_info.num_lines = 1;
		  src_info.num_planes = 1;
		  dst_info.bytes_per_chunk = 0;
		  dst_info.num_lines = 1;
		  dst_info.num_planes = 1;
		  write_pad_bytes = dst_bytes;
		  dst_bytes = 0;
		  dimension_mismatch_possible = false;
		  // src didn't actually take a step, so we don't need to cancel it
		  out_port->iter->confirm_step();
		} else {
		  // retry later
		  // src didn't actually take a step, so we don't need to cancel it
		  out_port->iter->cancel_step();
		  break;
		}
	      }
	      // a mismatch is still possible if the source is 2+D and the
	      //  destination wants to stop mid-span
	      if(src_bytes < dst_bytes) {
		assert(dimension_mismatch_possible);
		out_port->iter->cancel_step();
		dst_bytes = out_port->iter->step(src_bytes, dst_info,
						 flags & TransferIterator::DST_FLAGMASK,
						 true /*tentative*/);
	      }
	      // byte counts now must match
	      assert(src_bytes == dst_bytes);
	    } else {
	      // in the absense of dimension mismatches, it's safe now to confirm
	      //  the source step
	      if(!dimension_mismatch_possible)
		in_port->iter->confirm_step();
	    }

	    // when 2D transfers are allowed, it is possible that the
	    // bytes_per_chunk don't match, and we need to add an extra
	    //  dimension to one side or the other
	    // NOTE: this transformation can cause the dimensionality of the
	    //  transfer to grow.  Allow this to happen and detect it at the
	    //  end.
	    if(!dimension_mismatch_possible) {
	      assert(src_info.bytes_per_chunk == dst_info.bytes_per_chunk);
	      assert(src_info.num_lines == 1);
	      assert(src_info.num_planes == 1);
	      assert(dst_info.num_lines == 1);
	      assert(dst_info.num_planes == 1);
	    } else {
	      // track how much of src and/or dst is "lost" into a 4th
	      //  dimension
	      size_t src_4d_factor = 1;
	      size_t dst_4d_factor = 1;
	      if(src_info.bytes_per_chunk < dst_info.bytes_per_chunk) {
		size_t ratio = dst_info.bytes_per_chunk / src_info.bytes_per_chunk;
		assert((src_info.bytes_per_chunk * ratio) == dst_info.bytes_per_chunk);
		dst_4d_factor *= dst_info.num_planes; // existing planes lost
		dst_info.num_planes = dst_info.num_lines;
		dst_info.plane_stride = dst_info.line_stride;
		dst_info.num_lines = ratio;
		dst_info.line_stride = src_info.bytes_per_chunk;
		dst_info.bytes_per_chunk = src_info.bytes_per_chunk;
	      }
	      if(dst_info.bytes_per_chunk < src_info.bytes_per_chunk) {
		size_t ratio = src_info.bytes_per_chunk / dst_info.bytes_per_chunk;
		assert((dst_info.bytes_per_chunk * ratio) == src_info.bytes_per_chunk);
		src_4d_factor *= src_info.num_planes; // existing planes lost
		src_info.num_planes = src_info.num_lines;
		src_info.plane_stride = src_info.line_stride;
		src_info.num_lines = ratio;
		src_info.line_stride = dst_info.bytes_per_chunk;
		src_info.bytes_per_chunk = dst_info.bytes_per_chunk;
	      }
	  
	      // similarly, if the number of lines doesn't match, we need to promote
	      //  one of the requests from 2D to 3D
	      if(src_info.num_lines < dst_info.num_lines) {
		size_t ratio = dst_info.num_lines / src_info.num_lines;
		assert((src_info.num_lines * ratio) == dst_info.num_lines);
		dst_4d_factor *= dst_info.num_planes; // existing planes lost
		dst_info.num_planes = ratio;
		dst_info.plane_stride = dst_info.line_stride * src_info.num_lines;
		dst_info.num_lines = src_info.num_lines;
	      }
	      if(dst_info.num_lines < src_info.num_lines) {
		size_t ratio = src_info.num_lines / dst_info.num_lines;
		assert((dst_info.num_lines * ratio) == src_info.num_lines);
		src_4d_factor *= src_info.num_planes; // existing planes lost
		src_info.num_planes = ratio;
		src_info.plane_stride = src_info.line_stride * dst_info.num_lines;
		src_info.num_lines = dst_info.num_lines;
	      }

	      // sanity-checks: src/dst should match on lines/planes and we
	      //  shouldn't have multiple planes if we don't have multiple lines
	      assert(src_info.num_lines == dst_info.num_lines);
	      assert((src_info.num_planes * src_4d_factor) == 
		     (dst_info.num_planes * dst_4d_factor));
	      assert((src_info.num_lines > 1) || (src_info.num_planes == 1));
	      assert((dst_info.num_lines > 1) || (dst_info.num_planes == 1));

	      // only do as many planes as both src and dst can manage
	      if(src_info.num_planes > dst_info.num_planes)
		src_info.num_planes = dst_info.num_planes;
	      else
		dst_info.num_planes = src_info.num_planes;

	      // if 3D isn't allowed, set num_planes back to 1
	      if((flags & TransferIterator::PLANES_OK) == 0) {
		src_info.num_planes = 1;
		dst_info.num_planes = 1;
	      }

	      // now figure out how many bytes we're actually able to move and
	      //  if it's less than what we got from the iterators, try again
	      size_t act_bytes = (src_info.bytes_per_chunk *
				  src_info.num_lines *
				  src_info.num_planes);
	      if(act_bytes == src_bytes) {
		// things match up - confirm the steps
		in_port->iter->confirm_step();
		out_port->iter->confirm_step();
	      } else {
		//log_request.info() << "dimension mismatch! " << act_bytes << " < " << src_bytes << " (" << bytes_total << ")";
		TransferIterator::AddressInfo dummy_info;
		in_port->iter->cancel_step();
		src_bytes = in_port->iter->step(act_bytes, dummy_info,
						flags & TransferIterator::SRC_FLAGMASK,
						false /*!tentative*/);
		assert(src_bytes == act_bytes);
		out_port->iter->cancel_step();
		dst_bytes = out_port->iter->step(act_bytes, dummy_info,
						 flags & TransferIterator::DST_FLAGMASK,
						 false /*!tentative*/);
		assert(dst_bytes == act_bytes);
	      }
	    }

	    size_t act_bytes = (src_info.bytes_per_chunk *
				src_info.num_lines *
				src_info.num_planes);
	    read_seq = in_port->local_bytes_total;
	    read_bytes = act_bytes + read_pad_bytes;

	    // update bytes read unless we're using indirection
	    if(in_port->indirect_port_idx < 0) 
	      in_port->local_bytes_total += read_bytes;

	    write_seq = out_port->local_bytes_total;
	    write_bytes = act_bytes + write_pad_bytes;
	    out_port->local_bytes_total += write_bytes;
	    out_port->local_bytes_cons.store(out_port->local_bytes_total); // completion detection uses this
	  }

	  Request* new_req = dequeue_request();
	  new_req->src_port_idx = input_control.current_io_port;
	  new_req->dst_port_idx = output_control.current_io_port;
	  new_req->read_seq_pos = read_seq;
	  new_req->read_seq_count = read_bytes;
	  new_req->write_seq_pos = write_seq;
	  new_req->write_seq_count = write_bytes;
	  new_req->dim = ((src_info.num_planes == 1) ?
			  ((src_info.num_lines == 1) ? Request::DIM_1D :
			                               Request::DIM_2D) :
			                              Request::DIM_3D);
	  new_req->src_off = src_info.base_offset;
	  new_req->dst_off = dst_info.base_offset;
	  new_req->nbytes = src_info.bytes_per_chunk;
	  new_req->nlines = src_info.num_lines;
	  new_req->src_str = src_info.line_stride;
	  new_req->dst_str = dst_info.line_stride;
	  new_req->nplanes = src_info.num_planes;
	  new_req->src_pstr = src_info.plane_stride;
	  new_req->dst_pstr = dst_info.plane_stride;

	  // we can actually hit the end of an intermediate buffer input
	  //  even if our initial pbt_snapshot was (size_t)-1 because
	  //  we use the asynchronously-updated seq_pre_write, so if
	  //  we think we might be done, go ahead and resample here if
	  //  we still have -1
	  if((in_port->peer_guid != XFERDES_NO_GUID) &&
	     (pbt_snapshot == (size_t)-1))
	    pbt_snapshot = in_port->remote_bytes_total.load_acquire();

	  // if we have control ports, they tell us when we're done
	  if((input_control.control_port_idx >= 0) ||
	     (output_control.control_port_idx >= 0)) {
	    // update control port counts, which may also flag a completed iteration
	    size_t input_count = read_bytes - read_pad_bytes;
	    size_t output_count = write_bytes - write_pad_bytes;
	    // if we're serializing or deserializing, we count in elements,
	    //  not bytes
	    if(in_port->serdez_op != 0) {
	      // serializing impacts output size
	      assert((output_count % in_port->serdez_op->max_serialized_size) == 0);
	      output_count /= in_port->serdez_op->max_serialized_size;
	    }
	    if(out_port->serdez_op != 0) {
	      // and deserializing impacts input size
	      assert((input_count % out_port->serdez_op->max_serialized_size) == 0);
	      input_count /= out_port->serdez_op->max_serialized_size;
	    }
	    assert(input_control.remaining_count >= input_count);
	    assert(output_control.remaining_count >= output_count);
	    input_control.remaining_count -= input_count;
	    output_control.remaining_count -= output_count;
	    if(((input_control.remaining_count == 0) && input_control.eos_received) ||
	       ((output_control.remaining_count == 0) && output_control.eos_received)) {
	      log_xd.info() << "iteration completed via control port: xd=" << std::hex << guid << std::dec;
	      begin_completion();

#if 0
	      // non-ib iterators should end at the same time?
	      for(size_t i = 0; i < input_ports.size(); i++)
		assert((input_ports[i].peer_guid != XFERDES_NO_GUID) ||
		       input_ports[i].iter->done());
	      for(size_t i = 0; i < output_ports.size(); i++)
		assert((output_ports[i].peer_guid != XFERDES_NO_GUID) ||
		       output_ports[i].iter->done());
#endif
	    }
	  } else {
	    // otherwise, we go by our iterators
	    if(in_port->iter->done() || out_port->iter->done() ||
	       (in_port->local_bytes_total == pbt_snapshot)) {
	      assert(!iteration_completed.load());
	      begin_completion();
	    
	      // TODO: figure out how to eliminate false positives from these
	      //  checks with indirection and/or multiple remote inputs
#if 0
	      // non-ib iterators should end at the same time
	      assert((in_port->peer_guid != XFERDES_NO_GUID) || in_port->iter->done());
	      assert((out_port->peer_guid != XFERDES_NO_GUID) || out_port->iter->done());
#endif

	      if(!in_port->serdez_op && out_port->serdez_op) {
		// ok to be over, due to the conservative nature of
		//  deserialization reads
		assert((rbc_snapshot >= pbt_snapshot) ||
		       (pbt_snapshot == size_t(-1)));
	      } else {
		// TODO: this check is now too aggressive because the previous
		//  xd doesn't necessarily know when it's emitting its last
		//  data, which means the update of local_bytes_total might
		//  be delayed
#if 0
		assert((in_port->peer_guid == XFERDES_NO_GUID) ||
		       (pbt_snapshot == in_port->local_bytes_total));
#endif
	      }
	    }
	  }

	  switch(new_req->dim) {
	  case Request::DIM_1D:
	    {
	      log_request.info() << "request: guid=" << std::hex << guid << std::dec
				 << " ofs=" << new_req->src_off << "->" << new_req->dst_off
				 << " len=" << new_req->nbytes;
	      break;
	    }
	  case Request::DIM_2D:
	    {
	      log_request.info() << "request: guid=" << std::hex << guid << std::dec
				 << " ofs=" << new_req->src_off << "->" << new_req->dst_off
				 << " len=" << new_req->nbytes
				 << " lines=" << new_req->nlines << "(" << new_req->src_str << "," << new_req->dst_str << ")";
	      break;
	    }
	  case Request::DIM_3D:
	    {
	      log_request.info() << "request: guid=" << std::hex << guid << std::dec
				 << " ofs=" << new_req->src_off << "->" << new_req->dst_off
				 << " len=" << new_req->nbytes
				 << " lines=" << new_req->nlines << "(" << new_req->src_str << "," << new_req->dst_str << ")"
				 << " planes=" << new_req->nplanes << "(" << new_req->src_pstr << "," << new_req->dst_pstr << ")";
	      break;
	    }
	  }
	  reqs[idx++] = new_req;
	}
#if 0
        coord_t src_idx, dst_idx, todo, src_str, dst_str;
        size_t nitems, nlines;
        while (idx + MAX_GEN_REQS <= nr && offset_idx < oas_vec.size()
        && MAX_GEN_REQS <= available_reqs.size()) {
          if (DIM == 0) {
            todo = std::min((coord_t)(max_req_size / oas_vec[offset_idx].size),
                       me->continuous_steps(src_idx, dst_idx));
            nitems = src_str = dst_str = todo;
            nlines = 1;
          }
          else
            todo = std::min((coord_t)(max_req_size / oas_vec[offset_idx].size),
                       li->continuous_steps(src_idx, dst_idx,
                                            src_str, dst_str,
                                            nitems, nlines));
          coord_t src_in_block = src_buf.block_size
                               - src_idx % src_buf.block_size;
          coord_t dst_in_block = dst_buf.block_size
                               - dst_idx % dst_buf.block_size;
          todo = std::min(todo, std::min(src_in_block, dst_in_block));
          if (todo == 0)
            break;
          coord_t src_start, dst_start;
          if (src_buf.is_ib) {
            src_start = calc_mem_loc_ib(0,
                                        oas_vec[offset_idx].src_offset,
                                        oas_vec[offset_idx].size,
                                        src_buf.elmt_size,
                                        src_buf.block_size,
                                        src_buf.buf_size,
                                        domain.get_volume(), src_idx);
            todo = std::min(todo, std::max((coord_t)0,
					   (coord_t)(pre_bytes_write - src_start))
                                    / oas_vec[offset_idx].size);
	    // wrap src_start around within src_buf if needed
	    src_start %= src_buf.buf_size;
          } else {
            src_start = Realm::calc_mem_loc(0,
                                     oas_vec[offset_idx].src_offset,
                                     oas_vec[offset_idx].size,
                                     src_buf.elmt_size,
                                     src_buf.block_size, src_idx);
          }
          if (dst_buf.is_ib) {
            dst_start = calc_mem_loc_ib(0,
                                        oas_vec[offset_idx].dst_offset,
                                        oas_vec[offset_idx].size,
                                        dst_buf.elmt_size,
                                        dst_buf.block_size,
                                        dst_buf.buf_size,
                                        domain.get_volume(), dst_idx);
            todo = std::min(todo, std::max((coord_t)0,
					   (coord_t)(next_bytes_read + dst_buf.buf_size - dst_start))
                                    / oas_vec[offset_idx].size);
	    // wrap dst_start around within dst_buf if needed
	    dst_start %= dst_buf.buf_size;
          } else {
            dst_start = Realm::calc_mem_loc(0,
                                     oas_vec[offset_idx].dst_offset,
                                     oas_vec[offset_idx].size,
                                     dst_buf.elmt_size,
                                     dst_buf.block_size, dst_idx);
          }
          if (todo == 0)
            break;
          bool cross_src_ib = false, cross_dst_ib = false;
          if (src_buf.is_ib)
            cross_src_ib = cross_ib(src_start,
                                    todo * oas_vec[offset_idx].size,
                                    src_buf.buf_size);
          if (dst_buf.is_ib)
            cross_dst_ib = cross_ib(dst_start,
                                    todo * oas_vec[offset_idx].size,
                                    dst_buf.buf_size);
          // We are crossing ib, fallback to 1d case
          // We don't support 2D, fallback to 1d case
          if (cross_src_ib || cross_dst_ib || !support_2d_xfers(kind))
            todo = std::min(todo, (coord_t)nitems);
          if ((size_t)todo <= nitems) {
            // fallback to 1d case
            nitems = (size_t)todo;
            nlines = 1;
          } else {
            nlines = todo / nitems;
            todo = nlines * nitems;
          }
          if (nlines == 1) {
            // 1D case
            size_t nbytes = todo * oas_vec[offset_idx].size;
            while (nbytes > 0) {
              size_t req_size = nbytes;
              Request* new_req = dequeue_request();
              new_req->dim = Request::DIM_1D;
              if (src_buf.is_ib) {
                src_start = src_start % src_buf.buf_size;
                req_size = std::min(req_size, (size_t)(src_buf.buf_size - src_start));
              }
              if (dst_buf.is_ib) {
                dst_start = dst_start % dst_buf.buf_size;
                req_size = std::min(req_size, (size_t)(dst_buf.buf_size - dst_start));
              }
              new_req->src_off = src_start;
              new_req->dst_off = dst_start;
              new_req->nbytes = req_size;
              new_req->nlines = 1;
              log_request.info("[1D] guid(%llx) src_off(%lld) dst_off(%lld)"
                               " nbytes(%zu) offset_idx(%u)",
                               guid, src_start, dst_start, req_size, offset_idx);
              reqs[idx++] = new_req;
              nbytes -= req_size;
              src_start += req_size;
              dst_start += req_size;
            }
          } else {
            // 2D case
            Request* new_req = dequeue_request();
            new_req->dim = Request::DIM_2D;
            new_req->src_off = src_start;
            new_req->dst_off = dst_start;
            new_req->src_str = src_str * oas_vec[offset_idx].size;
            new_req->dst_str = dst_str * oas_vec[offset_idx].size;
            new_req->nbytes = nitems * oas_vec[offset_idx].size;
            new_req->nlines = nlines;
            reqs[idx++] = new_req;
          }
          if (DIM == 0) {
            me->move(todo);
            if (!me->any_left()) {
              me->reset();
              offset_idx ++;
            }
          } else {
            li->move(todo);
            if (!li->any_left()) {
              li->reset();
              offset_idx ++;
            }
          }
        } // while
#endif
        return idx;
      }

    void XferDes::begin_completion()
    {
#ifdef DEBUG_REALM
      // shouldn't be called more than once
      assert(!iteration_completed.load());
#endif
      iteration_completed.store_release(true);

      // give all output channels a chance to indicate completion and determine
      //  the total number of bytes we've written
      size_t total_bytes_written = 0;
      for(size_t i = 0; i < output_ports.size(); i++) {
        total_bytes_written += output_ports[i].local_bytes_cons.load();
        update_bytes_write(i, output_ports[i].local_bytes_total, 0);

        // see if we still need to send the total bytes
        if(output_ports[i].needs_pbt_update.load() &&
           (output_ports[i].local_bytes_total == output_ports[i].local_bytes_cons.load())) {
 #ifdef DEBUG_REALM
          assert(output_ports[i].peer_guid != XFERDES_NO_GUID);
#endif
          // exchange sets the flag to false and tells us previous value
          if(output_ports[i].needs_pbt_update.exchange(false))
            xferDes_queue->update_pre_bytes_total(output_ports[i].peer_guid,
                                                  output_ports[i].peer_port_idx,
                                                  output_ports[i].local_bytes_total);
        }
      }

      // bytes pending is total minus however many writes have already
      //  finished - if that's all of them, we can mark full transfer completion
      int64_t prev = bytes_write_pending.fetch_add(total_bytes_written);
      int64_t pending = prev + total_bytes_written;
      log_xd.info() << "completion: xd=" << std::hex << guid << std::dec
                    << " total_bytes=" << total_bytes_written << " pending=" << pending;
      assert(pending >= 0);
      if(pending == 0)
        transfer_completed.store_release(true);
    }

      void XferDes::update_bytes_read(int port_idx, size_t offset, size_t size)
      {
	XferPort *in_port = &input_ports[port_idx];
	size_t inc_amt = in_port->seq_local.add_span(offset, size);
	log_xd.info() << "bytes_read: " << std::hex << guid << std::dec
		      << "(" << port_idx << ") " << offset << "+" << size << " -> " << inc_amt;
	if(in_port->peer_guid != XFERDES_NO_GUID) {
	  if(inc_amt > 0) {
	    // we're actually telling the previous XD which offsets are ok to
	    //  overwrite, so adjust our offset by our (circular) IB size
            xferDes_queue->update_next_bytes_read(in_port->peer_guid,
						  in_port->peer_port_idx,
						  offset + in_port->ib_size,
						  inc_amt);
	  } else {
	    // TODO: mode to send non-contiguous updates?
	  }
	}
      }

#if 0
      inline void XferDes::simple_update_bytes_read(int64_t offset, uint64_t size)
        //printf("update_read[%lx]: offset = %ld, size = %lu, pre = %lx, next = %lx\n", guid, offset, size, pre_xd_guid, next_xd_guid);
        if (pre_xd_guid != XFERDES_NO_GUID) {
          bool update = false;
          if ((int64_t)(bytes_read % src_buf.buf_size) == offset) {
            bytes_read += size;
            update = true;
          }
          else {
            //printf("[%lx] insert: key = %ld, value = %lu\n", guid, offset, size);
            segments_read[offset] = size;
          }
          std::map<int64_t, uint64_t>::iterator it;
          while (true) {
            it = segments_read.find(bytes_read % src_buf.buf_size);
            if (it == segments_read.end())
              break;
            bytes_read += it->second;
            update = true;
            //printf("[%lx] erase: key = %ld, value = %lu\n", guid, it->first, it->second);
            segments_read.erase(it);
          }
          if (update) {
            xferDes_queue->update_next_bytes_read(pre_xd_guid, bytes_read);
          }
        }
        else {
          bytes_read += size;
        }
      }
#endif

      void XferDes::update_bytes_write(int port_idx, size_t offset, size_t size)
      {
	XferPort *out_port = &output_ports[port_idx];
	size_t inc_amt = out_port->seq_local.add_span(offset, size);
	log_xd.info() << "bytes_write: " << std::hex << guid << std::dec
		      << "(" << port_idx << ") " << offset << "+" << size << " -> " << inc_amt;

	if(out_port->peer_guid != XFERDES_NO_GUID) {
	  // update bytes total if needed (and available)
	  if(out_port->needs_pbt_update.load() &&
	     iteration_completed.load_acquire() &&
             (out_port->local_bytes_total == out_port->local_bytes_cons.load())) {
	    // exchange sets the flag to false and tells us previous value
	    if(out_port->needs_pbt_update.exchange(false))
	      xferDes_queue->update_pre_bytes_total(out_port->peer_guid,
						    out_port->peer_port_idx,
						    out_port->local_bytes_total);
	  }
	  // we can skip an update if this was empty
	  if(inc_amt > 0) {
            xferDes_queue->update_pre_bytes_write(out_port->peer_guid,
						  out_port->peer_port_idx,
						  offset,
						  inc_amt);
	  } else {
	    // TODO: mode to send non-contiguous updates?
	  }
	}

        // subtract bytes written from the pending count - if that causes it to
        //  go to zero, we can mark the transfer completed and update progress
        //  in case the xd is just waiting for that
        // NOTE: as soon as we set `transfer_completed`, the other references
        //  to this xd may be removed, so do this last, and hold a reference of
        //  our own long enough to call update_progress
        if(inc_amt > 0) {
          int64_t prev = bytes_write_pending.fetch_sub(inc_amt);
          if(prev > 0)
            log_xd.info() << "completion: xd=" << std::hex << guid << std::dec
                          << " remaining=" << (prev - inc_amt);
          if(inc_amt == static_cast<size_t>(prev)) {
            add_reference();
            transfer_completed.store_release(true);
            update_progress();
            remove_reference();
          }
        }
      }

#if 0
      inline void XferDes::simple_update_bytes_write(int64_t offset, uint64_t size)
      {
        log_request.info(
            "update_write: guid(%llx) off(%zd) size(%zu) pre(%llx) next(%llx)",
            guid, (ssize_t)offset, (size_t)size, pre_xd_guid, next_xd_guid);

	
        if (next_xd_guid != XFERDES_NO_GUID) {
          bool update = false;
          if ((int64_t)(bytes_write % dst_buf.buf_size) == offset) {
            bytes_write += size;
            update = true;
          } else {
            segments_write[offset] = size;
          }
          std::map<int64_t, uint64_t>::iterator it;
          while (true) {
            it = segments_write.find(bytes_write % dst_buf.buf_size);
            if (it == segments_write.end())
              break;
            bytes_write += it->second;
            update = true;
            segments_write.erase(it);
          }
          if (update) {
            xferDes_queue->update_pre_bytes_write(next_xd_guid, bytes_write);
          }
        }
        else {
          bytes_write += size;
        }
        //printf("[%d] offset(%ld), size(%lu), bytes_writes(%lx): %ld\n", Network::my_node_id, offset, size, guid, bytes_write);
      }
#endif

      void XferDes::update_pre_bytes_write(int port_idx, size_t offset, size_t size)
      {
	XferPort *in_port = &input_ports[port_idx];

	size_t inc_amt = in_port->seq_remote.add_span(offset, size);
	log_xd.info() << "pre_write: " << std::hex << guid << std::dec
		      << "(" << port_idx << ") " << offset << "+" << size << " -> " << inc_amt << " (" << in_port->remote_bytes_total.load() << ")";
	// if we got new data at the current pointer OR if we now know the
	//  total incoming bytes, update progress
	if(inc_amt > 0) update_progress();
      }

      void XferDes::update_pre_bytes_total(int port_idx, size_t pre_bytes_total)
      {
	XferPort *in_port = &input_ports[port_idx];

	// should always be exchanging -1 -> (not -1)
#ifdef DEBUG_REALM
	size_t oldval =
#endif
	  in_port->remote_bytes_total.exchange(pre_bytes_total);
#ifdef DEBUG_REALM
	assert((oldval == size_t(-1)) && (pre_bytes_total != size_t(-1)));
#endif
	log_xd.info() << "pre_total: " << std::hex << guid << std::dec
		      << "(" << port_idx << ") = " << pre_bytes_total;
	// this may unblock an xd that has consumed all input but didn't
	//  realize there was no more
	update_progress();
      }

      void XferDes::update_next_bytes_read(int port_idx, size_t offset, size_t size)
      {
	XferPort *out_port = &output_ports[port_idx];

	size_t inc_amt = out_port->seq_remote.add_span(offset, size);
	log_xd.info() << "next_read: "  << std::hex << guid << std::dec
		      << "(" << port_idx << ") " << offset << "+" << size << " -> " << inc_amt;
	// if we got new room at the current pointer (and we're still
        //  iterating), update progress
	if((inc_amt > 0) && !iteration_completed.load()) update_progress();
      }

      void XferDes::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      void XferDes::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      void XferDes::flush()
      {
      }

      void XferDes::default_notify_request_read_done(Request* req)
      {  
        req->is_read_done = true;
	update_bytes_read(req->src_port_idx,
			  req->read_seq_pos, req->read_seq_count);
#if 0
        if (req->dim == Request::DIM_1D)
          simple_update_bytes_read(req->src_off, req->nbytes);
        else
          simple_update_bytes_read(req->src_off, req->nbytes * req->nlines);
#endif
      }

      void XferDes::default_notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
	// calling update_bytes_write can cause the transfer descriptor to
	//  be destroyed, so enqueue the request first, and cache the values
	//  we need
	int dst_port_idx = req->dst_port_idx;
	size_t write_seq_pos = req->write_seq_pos;
	size_t write_seq_count = req->write_seq_count;
	update_bytes_write(dst_port_idx, write_seq_pos, write_seq_count);
#if 0
        if (req->dim == Request::DIM_1D)
          simple_update_bytes_write(req->dst_off, req->nbytes);
        else
          simple_update_bytes_write(req->dst_off, req->nbytes * req->nlines);
#endif
        enqueue_request(req);
      }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemcpyXferDes
  //

      MemcpyXferDes::MemcpyXferDes(uintptr_t _dma_op, Channel *_channel,
				   NodeID _launch_node, XferDesID _guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   int _priority)
	: XferDes(_dma_op, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _priority, 0, 0)
	, memcpy_req_in_use(false)
      {
	kind = XFER_MEM_CPY;

	// scan input and output ports to see if any use serdez ops
	has_serdez = false;
	for(size_t i = 0; i < inputs_info.size(); i++)
	  if(inputs_info[i].serdez_id != 0)
	    has_serdez = true;
	for(size_t i = 0; i < outputs_info.size(); i++)
	  if(outputs_info[i].serdez_id != 0)
	    has_serdez = true;

	// ignore requested max_nr and always use 1
	memcpy_req.xd = this;
      }

      long MemcpyXferDes::get_requests(Request** requests, long nr)
      {
        MemcpyRequest** reqs = (MemcpyRequest**) requests;
	// allow 2D and 3D copies
	unsigned flags = (TransferIterator::LINES_OK |
			  TransferIterator::PLANES_OK);
        long new_nr = default_get_requests(requests, nr, flags);
        for (long i = 0; i < new_nr; i++)
        {
	  bool src_is_serdez = (input_ports[reqs[i]->src_port_idx].serdez_op != 0);
	  bool dst_is_serdez = (output_ports[reqs[i]->dst_port_idx].serdez_op != 0);
          if(!src_is_serdez && dst_is_serdez) {
            // source offset is determined later - not safe to call get_direct_ptr now
            reqs[i]->src_base = 0;
          } else {
	    reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off,
										       reqs[i]->nbytes);
	    assert(reqs[i]->src_base != 0);
          }
          if(src_is_serdez && !dst_is_serdez) {
            // dest offset is determined later - not safe to call get_direct_ptr now
            reqs[i]->dst_base = 0;
          } else {
	    reqs[i]->dst_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off,
											reqs[i]->nbytes);
	    assert(reqs[i]->dst_base != 0);
          }
        }
        return new_nr;

#ifdef TO_BE_DELETE
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
          off_t src_start, dst_start;
          size_t nbytes;
          if (DIM == 0) {
            simple_get_mask_request(src_start, dst_start, nbytes, me, offset_idx, min(available_reqs.size(), nr - idx));
          } else {
            simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
          }
          if (nbytes == 0)
            break;
          //printf("[MemcpyXferDes] guid = %lx, offset_idx = %lld, oas_vec.size() = %lu, nbytes = %lu\n", guid, offset_idx, oas_vec.size(), nbytes);
          while (nbytes > 0) {
            size_t req_size = nbytes;
            if (src_buf.is_ib) {
              src_start = src_start % src_buf.buf_size;
              req_size = std::min(req_size, (size_t)(src_buf.buf_size - src_start));
            }
            if (dst_buf.is_ib) {
              dst_start = dst_start % dst_buf.buf_size;
              req_size = std::min(req_size, (size_t)(dst_buf.buf_size - dst_start));
            }
            mem_cpy_reqs[idx] = (MemcpyRequest*) available_reqs.front();
            available_reqs.pop();
            //printf("[MemcpyXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, nbytes);
            mem_cpy_reqs[idx]->is_read_done = false;
            mem_cpy_reqs[idx]->is_write_done = false;
            mem_cpy_reqs[idx]->src_buf = (char*)(src_buf_base + src_start);
            mem_cpy_reqs[idx]->dst_buf = (char*)(dst_buf_base + dst_start);
            mem_cpy_reqs[idx]->nbytes = req_size;
            src_start += req_size; // here we don't have to mod src_buf.buf_size since it will be performed in next loop
            dst_start += req_size; //
            nbytes -= req_size;
            idx++;
          }
        }
        return idx;
#endif
      }

      void MemcpyXferDes::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      void MemcpyXferDes::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      void MemcpyXferDes::flush()
      {
      }

      bool MemcpyXferDes::request_available()
      {
	return !memcpy_req_in_use;
      }

      Request* MemcpyXferDes::dequeue_request()
      {
	assert(!memcpy_req_in_use);
	memcpy_req_in_use = true;
	memcpy_req.is_read_done = false;
	memcpy_req.is_write_done = false;
	// memcpy request is handled in-thread, so no need to mess with refcount
        return &memcpy_req;
      }

      void MemcpyXferDes::enqueue_request(Request* req)
      {
	assert(memcpy_req_in_use);
	assert(req == &memcpy_req);
	memcpy_req_in_use = false;
      }

      bool MemcpyXferDes::progress_xd(MemcpyChannel *channel,
				      TimeLimit work_until)
      {
	if(has_serdez) {
	  Request *rq;
	  bool did_work = false;
	  do {
	    long count = get_requests(&rq, 1);
	    if(count > 0) {
	      channel->submit(&rq, count);
	      did_work = true;
	    } else
	      break;
	  } while(!work_until.is_expired());

	  return did_work;
	}

	// fast path - assumes no serdez
	bool did_work = false;
	ReadSequenceCache rseqcache(this, 2 << 20);  // flush after 2MB
	WriteSequenceCache wseqcache(this, 2 << 20);

	while(true) {
	  size_t min_xfer_size = 4096;  // TODO: make controllable
	  size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
	  if(max_bytes == 0)
	    break;

	  XferPort *in_port = 0, *out_port = 0;
	  size_t in_span_start = 0, out_span_start = 0;
	  if(input_control.current_io_port >= 0) {
	    in_port = &input_ports[input_control.current_io_port];
	    in_span_start = in_port->local_bytes_total;
	  }
	  if(output_control.current_io_port >= 0) {
	    out_port = &output_ports[output_control.current_io_port];
	    out_span_start = out_port->local_bytes_total;
	  }

	  size_t total_bytes = 0;
	  if(in_port != 0) {
	    if(out_port != 0) {
	      // input and output both exist - transfer what we can
	      log_xd.info() << "memcpy chunk: min=" << min_xfer_size
			    << " max=" << max_bytes;

	      uintptr_t in_base = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
	      uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

	      while(total_bytes < max_bytes) {
		AddressListCursor& in_alc = in_port->addrcursor;
		AddressListCursor& out_alc = out_port->addrcursor;

		uintptr_t in_offset = in_alc.get_offset();
		uintptr_t out_offset = out_alc.get_offset();

		// the reported dim is reduced for partially consumed address
		//  ranges - whatever we get can be assumed to be regular
		int in_dim = in_alc.get_dim();
		int out_dim = out_alc.get_dim();

		size_t bytes = 0;
		size_t bytes_left = max_bytes - total_bytes;
		// memcpys don't need to be particularly big to achieve
		//  peak efficiency, so trim to something that takes
		//  10's of us to be responsive to the time limit
		bytes_left = std::min(bytes_left, size_t(256 << 10));

		if(in_dim > 0) {
		  if(out_dim > 0) {
		    size_t icount = in_alc.remaining(0);
		    size_t ocount = out_alc.remaining(0);

		    // contig bytes is always the min of the first dimensions
		    size_t contig_bytes = std::min(std::min(icount, ocount),
						   bytes_left);

		    // catch simple 1D case first
		    if((contig_bytes == bytes_left) ||
		       ((contig_bytes == icount) && (in_dim == 1)) ||
		       ((contig_bytes == ocount) && (out_dim == 1))) {
		      bytes = contig_bytes;
		      memcpy_1d(out_base + out_offset,
				in_base + in_offset,
				bytes);
		      in_alc.advance(0, bytes);
		      out_alc.advance(0, bytes);
		    } else {
		      // grow to a 2D copy
		      int id;
		      int iscale;
		      uintptr_t in_lstride;
		      if(contig_bytes < icount) {
			// second input dim comes from splitting first
			id = 0;
			in_lstride = contig_bytes;
			size_t ilines = icount / contig_bytes;
			if((ilines * contig_bytes) != icount)
			  in_dim = 1;  // leftover means we can't go beyond this
			icount = ilines;
			iscale = contig_bytes;
		      } else {
			assert(in_dim > 1);
			id = 1;
			icount = in_alc.remaining(id);
			in_lstride = in_alc.get_stride(id);
			iscale = 1;
		      }

		      int od;
		      int oscale;
		      uintptr_t out_lstride;
		      if(contig_bytes < ocount) {
			// second output dim comes from splitting first
			od = 0;
			out_lstride = contig_bytes;
			size_t olines = ocount / contig_bytes;
			if((olines * contig_bytes) != ocount)
			  out_dim = 1;  // leftover means we can't go beyond this
			ocount = olines;
			oscale = contig_bytes;
		      } else {
			assert(out_dim > 1);
			od = 1;
			ocount = out_alc.remaining(od);
			out_lstride = out_alc.get_stride(od);
			oscale = 1;
		      }

		      size_t lines = std::min(std::min(icount, ocount),
					      bytes_left / contig_bytes);

		      // see if we need to stop at 2D
		      if(((contig_bytes * lines) == bytes_left) ||
			 ((lines == icount) && (id == (in_dim - 1))) ||
			 ((lines == ocount) && (od == (out_dim - 1)))) {
			bytes = contig_bytes * lines;
			memcpy_2d(out_base + out_offset, out_lstride,
				  in_base + in_offset, in_lstride,
				  contig_bytes, lines);
			in_alc.advance(id, lines * iscale);
			out_alc.advance(od, lines * oscale);
		      } else {
			uintptr_t in_pstride;
			if(lines < icount) {
			  // third input dim comes from splitting current
			  in_pstride = in_lstride * lines;
			  size_t iplanes = icount / lines;
			  // check for leftovers here if we go beyond 3D!
			  icount = iplanes;
			  iscale *= lines;
			} else {
			  id++;
			  assert(in_dim > id);
			  icount = in_alc.remaining(id);
			  in_pstride = in_alc.get_stride(id);
			  iscale = 1;
			}

			uintptr_t out_pstride;
			if(lines < ocount) {
			  // third output dim comes from splitting current
			  out_pstride = out_lstride * lines;
			  size_t oplanes = ocount / lines;
			  // check for leftovers here if we go beyond 3D!
			  ocount = oplanes;
			  oscale *= lines;
			} else {
			  od++;
			  assert(out_dim > od);
			  ocount = out_alc.remaining(od);
			  out_pstride = out_alc.get_stride(od);
			  oscale = 1;
			}

			size_t planes = std::min(std::min(icount, ocount),
						 (bytes_left /
						  (contig_bytes * lines)));

			bytes = contig_bytes * lines * planes;
			memcpy_3d(out_base + out_offset, out_lstride, out_pstride,
				  in_base + in_offset, in_lstride, in_pstride,
				  contig_bytes, lines, planes);
			in_alc.advance(id, planes * iscale);
			out_alc.advance(od, planes * oscale);
		      }
		    }
		  } else {
		    // scatter adddress list
		    assert(0);
		  }
		} else {
		  if(out_dim > 0) {
		    // gather address list
		    assert(0);
		  } else {
		    // gather and scatter
		    assert(0);
		  }
		}

#ifdef DEBUG_REALM
		assert(bytes <= bytes_left);
#endif
		total_bytes += bytes;

		// stop if it's been too long, but make sure we do at least the
		//  minimum number of bytes
		if((total_bytes >= min_xfer_size) && work_until.is_expired()) break;
	      }
	    } else {
	      // input but no output, so skip input bytes
	      total_bytes = max_bytes;
	      in_port->addrcursor.skip_bytes(total_bytes);
	    }
	  } else {
	    if(out_port != 0) {
	      // output but no input, so skip output bytes
	      total_bytes = max_bytes;
	      out_port->addrcursor.skip_bytes(total_bytes);
	    } else {
	      // skipping both input and output is possible for simultaneous
	      //  gather+scatter
	      total_bytes = max_bytes;
	    }
	  }

	  // memcpy is always immediate, so handle both skip and copy with the
	  //  same code
	  rseqcache.add_span(input_control.current_io_port,
			     in_span_start, total_bytes);
	  in_span_start += total_bytes;
	  wseqcache.add_span(output_control.current_io_port,
			     out_span_start, total_bytes);
	  out_span_start += total_bytes;

	  bool done = record_address_consumption(total_bytes, total_bytes);

	  did_work = true;

	  if(done || work_until.is_expired())
	    break;
	}

	rseqcache.flush();
	wseqcache.flush();

	return did_work;
      }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemfillXferDes
  //

  MemfillXferDes::MemfillXferDes(uintptr_t _dma_op, Channel *_channel,
				 NodeID _launch_node, XferDesID _guid,
				 const std::vector<XferDesPortInfo>& inputs_info,
				 const std::vector<XferDesPortInfo>& outputs_info,
				 int _priority,
				 const void *_fill_data, size_t _fill_size,
                                 size_t _fill_total)
	: XferDes(_dma_op, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _priority, _fill_data, _fill_size)
      {
	kind = XFER_MEM_FILL;

	// no direct input data for us, but we know how much data to produce
        //  (in case the output is an intermediate buffer)
	assert(input_control.control_port_idx == -1);
	input_control.current_io_port = -1;
        input_control.remaining_count = _fill_total;
        input_control.eos_received = true;
      }

      long MemfillXferDes::get_requests(Request** requests, long nr)
      {
	// unused
	assert(0);
	return 0;
      }

      bool MemfillXferDes::request_available()
      {
	// unused
	assert(0);
	return false;
      }

      Request* MemfillXferDes::dequeue_request()
      {
	// unused
	assert(0);
	return 0;
      }

      void MemfillXferDes::enqueue_request(Request* req)
      {
	// unused
	assert(0);
      }

      bool MemfillXferDes::progress_xd(MemfillChannel *channel,
				      TimeLimit work_until)
      {
	bool did_work = false;
	ReadSequenceCache rseqcache(this, 2 << 20);
	WriteSequenceCache wseqcache(this, 2 << 20);

	while(true) {
	  size_t min_xfer_size = 4096;  // TODO: make controllable
	  size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
	  if(max_bytes == 0)
	    break;

	  XferPort *out_port = 0;
	  size_t out_span_start = 0;
	  if(output_control.current_io_port >= 0) {
	    out_port = &output_ports[output_control.current_io_port];
	    out_span_start = out_port->local_bytes_total;
	  }

	  size_t total_bytes = 0;
	  if(out_port != 0) {
	    // input and output both exist - transfer what we can
	    log_xd.info() << "memfill chunk: min=" << min_xfer_size
			  << " max=" << max_bytes;

	    uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

	    while(total_bytes < max_bytes) {
	      AddressListCursor& out_alc = out_port->addrcursor;

	      uintptr_t out_offset = out_alc.get_offset();

	      // the reported dim is reduced for partially consumed address
	      //  ranges - whatever we get can be assumed to be regular
	      int out_dim = out_alc.get_dim();

	      size_t bytes = 0;
	      size_t bytes_left = max_bytes - total_bytes;
	      // memfills don't need to be particularly big to achieve
	      //  peak efficiency, so trim to something that takes
	      //  10's of us to be responsive to the time limit
              // NOTE: have to be a little careful and make sure the limit
              //  is a multiple of the fill size - we'll make it a power-of-2
              const size_t TARGET_CHUNK_SIZE = 256 << 10; // 256KB
              if(bytes_left > TARGET_CHUNK_SIZE) {
                size_t max_chunk = fill_size;
                while(max_chunk < TARGET_CHUNK_SIZE) max_chunk <<= 1;
                bytes_left = std::min(bytes_left, max_chunk);
              }

	      if(out_dim > 0) {
		size_t ocount = out_alc.remaining(0);

		// contig bytes is always the first dimension
		size_t contig_bytes = std::min(ocount, bytes_left);

		// catch simple 1D case first
		if((contig_bytes == bytes_left) ||
		   ((contig_bytes == ocount) && (out_dim == 1))) {
		  bytes = contig_bytes;
		  memset_1d(out_base + out_offset, contig_bytes,
                            fill_data, fill_size);
		  out_alc.advance(0, bytes);
		} else {
		  // grow to a 2D fill
		  ocount = out_alc.remaining(1);
		  uintptr_t out_lstride = out_alc.get_stride(1);

		  size_t lines = std::min(ocount,
					  bytes_left / contig_bytes);

		  bytes = contig_bytes * lines;
                  memset_2d(out_base + out_offset, out_lstride,
                            contig_bytes, lines,
                            fill_data, fill_size);
		  out_alc.advance(1, lines);
		}
	      } else {
		// scatter adddress list
		assert(0);
	      }

#ifdef DEBUG_REALM
	      assert(bytes <= bytes_left);
#endif
	      total_bytes += bytes;

	      // stop if it's been too long, but make sure we do at least the
	      //  minimum number of bytes
	      if((total_bytes >= min_xfer_size) && work_until.is_expired()) break;
	    }
	  } else {
	    // fill with no output, so just count the bytes
	    total_bytes = max_bytes;
	  }

	  // mem fill is always immediate, so handle both skip and copy with
	  //  the same code
	  wseqcache.add_span(output_control.current_io_port,
			     out_span_start, total_bytes);
	  out_span_start += total_bytes;

	  bool done = record_address_consumption(total_bytes, total_bytes);

	  did_work = true;

	  if(done || work_until.is_expired())
	    break;
	}

	rseqcache.flush();
	wseqcache.flush();

	return did_work;
      }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemfillXferDes
  //

      MemreduceXferDes::MemreduceXferDes(uintptr_t _dma_op, Channel *_channel,
                                         NodeID _launch_node, XferDesID _guid,
                                         const std::vector<XferDesPortInfo>& inputs_info,
                                         const std::vector<XferDesPortInfo>& outputs_info,
                                         int _priority,
                                         XferDesRedopInfo _redop_info)
	: XferDes(_dma_op, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _priority, 0, 0)
        , redop_info(_redop_info)
      {
	kind = XFER_MEM_CPY;
        redop = get_runtime()->reduce_op_table.get(redop_info.id, 0);
        assert(redop);
      }

      long MemreduceXferDes::get_requests(Request** requests, long nr)
      {
	// unused
	assert(0);
	return 0;
      }

      bool MemreduceXferDes::progress_xd(MemreduceChannel *channel,
				      TimeLimit work_until)
      {
	bool did_work = false;
	ReadSequenceCache rseqcache(this, 2 << 20);  // flush after 2MB
	WriteSequenceCache wseqcache(this, 2 << 20);

        const size_t in_elem_size = redop->sizeof_rhs;
        const size_t out_elem_size = (redop_info.is_fold ? redop->sizeof_rhs : redop->sizeof_lhs);
        assert(redop_info.in_place);  // TODO: support for out-of-place reduces

	while(true) {
	  size_t min_xfer_size = 4096;  // TODO: make controllable
	  size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
	  if(max_bytes == 0)
	    break;

	  XferPort *in_port = 0, *out_port = 0;
	  size_t in_span_start = 0, out_span_start = 0;
	  if(input_control.current_io_port >= 0) {
	    in_port = &input_ports[input_control.current_io_port];
	    in_span_start = in_port->local_bytes_total;
	  }
	  if(output_control.current_io_port >= 0) {
	    out_port = &output_ports[output_control.current_io_port];
	    out_span_start = out_port->local_bytes_total;
	  }

          // have to count in terms of elements, which requires redoing some math
          //  if in/out sizes do not match
          size_t max_elems;
          if(in_elem_size == out_elem_size) {
            max_elems = max_bytes / in_elem_size;
          } else {
            max_elems = std::min(input_control.remaining_count / in_elem_size,
                                 output_control.remaining_count / out_elem_size);
            if(in_port != 0) {
              max_elems = std::min(max_elems,
                                   in_port->addrlist.bytes_pending() / in_elem_size);
              if(in_port->peer_guid != XFERDES_NO_GUID) {
                size_t read_bytes_avail = in_port->seq_remote.span_exists(in_port->local_bytes_total,
                                                                          (max_elems * in_elem_size));
                max_elems = std::min(max_elems,
                                     (read_bytes_avail / in_elem_size));
              }
            }
            if(out_port != 0) {
              max_elems = std::min(max_elems,
                                   out_port->addrlist.bytes_pending() / out_elem_size);
              // no support for reducing into an intermediate buffer
              assert(out_port->peer_guid == XFERDES_NO_GUID);
            }
          }

	  size_t total_elems = 0;
	  if(in_port != 0) {
	    if(out_port != 0) {
	      // input and output both exist - transfer what we can
	      log_xd.info() << "memreduce chunk: min=" << min_xfer_size
			    << " max_elems=" << max_elems;

	      uintptr_t in_base = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
	      uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

	      while(total_elems < max_elems) {
		AddressListCursor& in_alc = in_port->addrcursor;
		AddressListCursor& out_alc = out_port->addrcursor;

		uintptr_t in_offset = in_alc.get_offset();
		uintptr_t out_offset = out_alc.get_offset();

		// the reported dim is reduced for partially consumed address
		//  ranges - whatever we get can be assumed to be regular
		int in_dim = in_alc.get_dim();
		int out_dim = out_alc.get_dim();

                // the current reduction op interface can reduce multiple elements
                //  with a fixed address stride, which looks to us like either
                //  1D (stride = elem_size), or 2D with 1 elem/line

                size_t icount = in_alc.remaining(0) / in_elem_size;
                size_t ocount = out_alc.remaining(0) / out_elem_size;
                size_t istride, ostride;
                if((in_dim > 1) && (icount == 1)) {
                  in_dim = 2;
                  icount = in_alc.remaining(1);
                  istride = in_alc.get_stride(1);
                } else {
                  in_dim = 1;
                  istride = in_elem_size;
                }
                if((out_dim > 1) && (ocount == 1)) {
                  out_dim = 2;
                  ocount = out_alc.remaining(1);
                  ostride = out_alc.get_stride(1);
                } else {
                  out_dim = 1;
                  ostride = out_elem_size;
                }

		size_t elems_left = max_elems - total_elems;
                size_t elems = std::min(std::min(icount, ocount), elems_left);
                assert(elems > 0);

                void *out_ptr = reinterpret_cast<void *>(out_base + out_offset);
                const void *in_ptr = reinterpret_cast<const void *>(in_base + in_offset);
                if(redop_info.is_fold) {
                  if(redop_info.is_exclusive)
                    (redop->cpu_fold_excl_fn)(out_ptr, ostride,
                                              in_ptr, istride,
                                              elems, redop->userdata);
                  else
                    (redop->cpu_fold_nonexcl_fn)(out_ptr, ostride,
                                                 in_ptr, istride,
                                                 elems, redop->userdata);
                } else {
                  if (redop_info.is_exclusive)
                    (redop->cpu_apply_excl_fn)(out_ptr, ostride,
                                               in_ptr, istride,
                                               elems, redop->userdata);
                  else
                    (redop->cpu_apply_nonexcl_fn)(out_ptr, ostride,
                                                  in_ptr, istride,
                                                  elems, redop->userdata);
                }

                in_alc.advance(in_dim-1,
                               elems * ((in_dim == 1) ? in_elem_size : 1));
                out_alc.advance(out_dim-1,
                                elems * ((out_dim == 1) ? out_elem_size : 1));

#ifdef DEBUG_REALM
		assert(elems <= elems_left);
#endif
		total_elems += elems;

		// stop if it's been too long, but make sure we do at least the
		//  minimum number of bytes
		if(((total_elems * in_elem_size) >= min_xfer_size) &&
                   work_until.is_expired()) break;
	      }
	    } else {
	      // input but no output, so skip input bytes
              total_elems = max_elems;
	      in_port->addrcursor.skip_bytes(total_elems * in_elem_size);
	    }
	  } else {
	    if(out_port != 0) {
	      // output but no input, so skip output bytes
              total_elems = max_elems;
	      out_port->addrcursor.skip_bytes(total_elems * out_elem_size);
	    } else {
	      // skipping both input and output is possible for simultaneous
	      //  gather+scatter
              total_elems = max_elems;
	    }
	  }

	  // memcpy is always immediate, so handle both skip and copy with the
	  //  same code
	  rseqcache.add_span(input_control.current_io_port,
			     in_span_start, total_elems * in_elem_size);
	  in_span_start += total_elems * in_elem_size;
	  wseqcache.add_span(output_control.current_io_port,
			     out_span_start, total_elems * out_elem_size);
	  out_span_start += total_elems * out_elem_size;

	  bool done = record_address_consumption(total_elems * in_elem_size,
                                                 total_elems * out_elem_size);

	  did_work = true;

	  if(done || work_until.is_expired())
	    break;
	}

	rseqcache.flush();
	wseqcache.flush();

	return did_work;
      }


      GASNetXferDes::GASNetXferDes(uintptr_t _dma_op, Channel *_channel,
				   NodeID _launch_node, XferDesID _guid,
				   const std::vector<XferDesPortInfo>& inputs_info,
				   const std::vector<XferDesPortInfo>& outputs_info,
				   int _priority)
	: XferDes(_dma_op, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _priority, 0, 0)
      {
	if((inputs_info.size() >= 1) &&
	   (input_ports[0].mem->kind == MemoryImpl::MKIND_GLOBAL)) {
	  kind = XFER_GASNET_READ;
	} else if((outputs_info.size() >= 1) &&
		  (output_ports[0].mem->kind == MemoryImpl::MKIND_GLOBAL)) {
	  kind = XFER_GASNET_WRITE;
	} else {
	  assert(0 && "neither source nor dest of GASNetXferDes is gasnet!?");
	}
	const int max_nr = 10; // FIXME
        gasnet_reqs = (GASNetRequest*) calloc(max_nr, sizeof(GASNetRequest));
        for (int i = 0; i < max_nr; i++) {
          gasnet_reqs[i].xd = this;
	  available_reqs.push(&gasnet_reqs[i]);
        }
      }

      long GASNetXferDes::get_requests(Request** requests, long nr)
      {
        GASNetRequest** reqs = (GASNetRequest**) requests;
        long new_nr = default_get_requests(requests, nr);
        switch (kind) {
          case XFER_GASNET_READ:
          {
            for (long i = 0; i < new_nr; i++) {
              reqs[i]->gas_off = /*src_buf.alloc_offset +*/ reqs[i]->src_off;
              //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
	      reqs[i]->mem_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off,
											  reqs[i]->nbytes);
	      assert(reqs[i]->mem_base != 0);
            }
            break;
          }
          case XFER_GASNET_WRITE:
          {
            for (long i = 0; i < new_nr; i++) {
              //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
	      reqs[i]->mem_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off,
											 reqs[i]->nbytes);
	      assert(reqs[i]->mem_base != 0);
              reqs[i]->gas_off = /*dst_buf.alloc_offset +*/ reqs[i]->dst_off;
            }
            break;
          }
          default:
            assert(0);
        }
        return new_nr;
      }

      bool GASNetXferDes::progress_xd(GASNetChannel *channel,
				      TimeLimit work_until)
      {
	Request *rq;
	bool did_work = false;
	do {
	  long count = get_requests(&rq, 1);
	  if(count > 0) {
	    channel->submit(&rq, count);
	    did_work = true;
	  } else
	    break;
	} while(!work_until.is_expired());

	return did_work;
      }

      void GASNetXferDes::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      void GASNetXferDes::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      void GASNetXferDes::flush()
      {
      }

      RemoteWriteXferDes::RemoteWriteXferDes(uintptr_t _dma_op, Channel *_channel,
					     NodeID _launch_node, XferDesID _guid,
					     const std::vector<XferDesPortInfo>& inputs_info,
					     const std::vector<XferDesPortInfo>& outputs_info,
					     int _priority)
	: XferDes(_dma_op, _channel, _launch_node, _guid,
		  inputs_info, outputs_info,
		  _priority, 0, 0)
      {
	kind = XFER_REMOTE_WRITE;
	requests = 0;
#if 0
        requests = (RemoteWriteRequest*) calloc(max_nr, sizeof(RemoteWriteRequest));
        for (int i = 0; i < max_nr; i++) {
          requests[i].xd = this;
          //requests[i].dst_node = dst_node;
          available_reqs.push(&requests[i]);
        }
#endif
      }

      long RemoteWriteXferDes::get_requests(Request** requests, long nr)
      {
	xd_lock.lock();
        RemoteWriteRequest** reqs = (RemoteWriteRequest**) requests;
	// remote writes allow 2D on source, but not destination
	unsigned flags = TransferIterator::SRC_LINES_OK;
        long new_nr = default_get_requests(requests, nr, flags);
        for (long i = 0; i < new_nr; i++)
        {
          //reqs[i]->src_base = (char*)(src_buf_base + reqs[i]->src_off);
	  reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off,
										     reqs[i]->nbytes);
	  assert(reqs[i]->src_base != 0);
	  //RemoteMemory *remote = checked_cast<RemoteMemory *>(output_ports[reqs[i]->dst_port_idx].mem);
	  //reqs[i]->dst_base = static_cast<char *>(remote->get_remote_addr(reqs[i]->dst_off));
	  //assert(reqs[i]->dst_base != 0);
        }
	xd_lock.unlock();
        return new_nr;
      }

      // callbacks for updating read/write spans
      class ReadBytesUpdater {
      public:
	ReadBytesUpdater(XferDes *_xd, int _port_idx,
			 size_t _offset, size_t _size)
	  : xd(_xd), port_idx(_port_idx), offset(_offset), size(_size)
	{}

	void operator()() const
	{
	  xd->update_bytes_read(port_idx, offset, size);
	  xd->remove_reference();
	}

      protected:
	XferDes *xd;
	int port_idx;
	size_t offset, size;
      };

      class WriteBytesUpdater {
      public:
	WriteBytesUpdater(XferDes *_xd, int _port_idx,
			  size_t _offset, size_t _size)
	  : xd(_xd), port_idx(_port_idx), offset(_offset), size(_size)
	{}

	void operator()() const
	{
	  xd->update_bytes_write(port_idx, offset, size);
	}

      protected:
	XferDes *xd;
	int port_idx;
	size_t offset, size;
      };

      bool RemoteWriteXferDes::progress_xd(RemoteWriteChannel *channel,
					   TimeLimit work_until)
      {
#if 0
	Request *rq;
	bool did_work = false;
	do {
	  long count = get_requests(&rq, 1);
	  if(count > 0) {
	    channel->submit(&rq, count);
	    did_work = true;
	  } else
	    break;
	} while(!work_until.is_expired());

	return did_work;
#else
	bool did_work = false;
	// immediate acks for reads happen when we assemble or skip input,
	//  while immediate acks for writes happen only if we skip output
	ReadSequenceCache rseqcache(this);
	WriteSequenceCache wseqcache(this);

	const size_t MAX_ASSEMBLY_SIZE = 4096;
	while(true) {
	  size_t min_xfer_size = 4096;  // TODO: make controllable
	  size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
	  if(max_bytes == 0)
	    break;

	  XferPort *in_port = 0, *out_port = 0;
	  size_t in_span_start = 0, out_span_start = 0;
	  if(input_control.current_io_port >= 0) {
	    in_port = &input_ports[input_control.current_io_port];
	    in_span_start = in_port->local_bytes_total;
	  }
	  if(output_control.current_io_port >= 0) {
	    out_port = &output_ports[output_control.current_io_port];
	    out_span_start = out_port->local_bytes_total;
	  }

	  size_t total_bytes = 0;
	  if(in_port != 0) {
	    if(out_port != 0) {
	      // input and output both exist - transfer what we can
	      log_xd.info() << "remote write chunk: min=" << min_xfer_size
			    << " max=" << max_bytes;

	      while(total_bytes < max_bytes) {
		AddressListCursor& in_alc = in_port->addrcursor;
		AddressListCursor& out_alc = out_port->addrcursor;
		int in_dim = in_alc.get_dim();
		int out_dim = out_alc.get_dim();
		size_t icount = in_alc.remaining(0);
		size_t ocount = out_alc.remaining(0);

		size_t bytes = 0;
		size_t bytes_left = max_bytes - total_bytes;

		// look at the output first, because that controls the message
		//  size
		size_t dst_1d_maxbytes = ((out_dim > 0) ?
					    std::min(bytes_left, ocount) :
					    0);
		size_t dst_2d_maxbytes = (((out_dim > 1) &&
					   (ocount <= (MAX_ASSEMBLY_SIZE / 2))) ?
					    (ocount * std::min(MAX_ASSEMBLY_SIZE / ocount,
							       out_alc.remaining(1))) :
					    0);
		// would have to scan forward through the dst address list to
		//  get the exact number of bytes that we can fit into
		//  MAX_ASSEMBLY_SIZE after considering address info overhead,
		//  but this is a last resort anyway, so just use a probably-
		//  pessimistic estimate;
		size_t dst_sc_maxbytes = std::min(bytes_left,
						  MAX_ASSEMBLY_SIZE / 4);
		// TODO: actually implement 2d and sc
		dst_2d_maxbytes = 0;
		dst_sc_maxbytes = 0;

		// favor 1d >> 2d >> sc
		if((dst_1d_maxbytes >= dst_2d_maxbytes) &&
		   (dst_1d_maxbytes >= dst_sc_maxbytes)) {
		  // 1D target
		  NodeID dst_node = ID(out_port->mem->me).memory_owner_node();
		  RemoteAddress dst_buf;
		  bool ok = out_port->mem->get_remote_addr(out_alc.get_offset(),
							   dst_buf);
		  assert(ok);

		  // now look at the input
		  const void *src_buf = in_port->mem->get_direct_ptr(in_alc.get_offset(), icount);
		  size_t src_1d_maxbytes = 0;
		  if(in_dim > 0) {
		    size_t rec_bytes = ActiveMessage<Write1DMessage>::recommended_max_payload(dst_node,
											      src_buf, icount, 1, 0,
											      dst_buf,
											      true /*w/ congestion*/);
		    src_1d_maxbytes = std::min({ dst_1d_maxbytes,
					         icount,
					         rec_bytes });
		  }

		  size_t src_2d_maxbytes = 0;
                  // TODO: permit if source memory is cpu-accessible?
#ifdef ALLOW_RDMA_SOURCE_2D
		  if(in_dim > 1) {
		    size_t lines = in_alc.remaining(1);
		    size_t rec_bytes = ActiveMessage<Write1DMessage>::recommended_max_payload(dst_node,
											      src_buf, icount,
											      lines,
											      in_alc.get_stride(1),
											      dst_buf,
											      true /*w/ congestion*/);
		    // round the recommendation down to a multiple of the line size
		    rec_bytes -= (rec_bytes % icount);
		    src_2d_maxbytes = std::min({ dst_1d_maxbytes,
			                         icount * lines,
			                         rec_bytes });
		  }
#endif
		  size_t src_ga_maxbytes = 0;
                  // TODO: permit if source memory is cpu-accessible?
#ifdef ALLOW_RDMA_GATHER
		  {
		    // a gather will assemble into a buffer provided by the network
		    size_t rec_bytes = ActiveMessage<Write1DMessage>::recommended_max_payload(dst_node,
											      dst_buf,
											      true /*w/ congestion*/);
		    src_ga_maxbytes = std::min({ dst_1d_maxbytes,
					         bytes_left,
					         rec_bytes });
		  }
#endif

		  // source also favors 1d >> 2d >> gather
		  if((src_1d_maxbytes >= src_2d_maxbytes) &&
		     (src_1d_maxbytes >= src_ga_maxbytes)) {
                    // TODO: if congestion is telling us not to send anything
                    //  at all, it'd be better to sleep until the network
                    //  says it's reasonable to try again - this approach
                    //  will effectively spinwait (but at least guarantees to
                    //  intersperse it with calls to the network progress
                    //  work item)
                    if(src_1d_maxbytes == 0) break;

		    // 1D source
		    bytes = src_1d_maxbytes;
		    //log_xd.info() << "remote write 1d: guid=" << guid
		    //              << " src=" << src_buf << " dst=" << dst_buf
		    //              << " bytes=" << bytes;
		    ActiveMessage<Write1DMessage> amsg(dst_node,
						       src_buf, bytes,
						       dst_buf);
		    amsg->next_xd_guid = out_port->peer_guid;
		    amsg->next_port_idx = out_port->peer_port_idx;
		    amsg->span_start = out_span_start;

		    // reads aren't consumed until local completion, but
		    //  only ask if we have a previous xd that's going to
		    //  care
		    if(in_port->peer_guid != XFERDES_NO_GUID) {
		      // a ReadBytesUpdater holds a reference to the xd
		      add_reference();
		      amsg.add_local_completion(ReadBytesUpdater(this,
								 input_control.current_io_port,
								 in_span_start,
								 bytes));
		    }
		    in_span_start += bytes;
		    // the write isn't complete until it's ack'd by the target
		    amsg.add_remote_completion(WriteBytesUpdater(this,
								 output_control.current_io_port,
								 out_span_start,
								 bytes));
		    out_span_start += bytes;

		    amsg.commit();
		    in_alc.advance(0, bytes);
		    out_alc.advance(0, bytes);
		  }
		  else if(src_2d_maxbytes >= src_ga_maxbytes) {
		    // 2D source
		    size_t bytes_per_line = icount;
		    size_t lines = src_2d_maxbytes / icount;
		    bytes = bytes_per_line * lines;
		    assert(bytes == src_2d_maxbytes);
		    size_t src_stride = in_alc.get_stride(1);
		    //log_xd.info() << "remote write 2d: guid=" << guid
		    //              << " src=" << src_buf << " dst=" << dst_buf
		    //              << " bytes=" << bytes << " lines=" << lines
		    //              << " stride=" << src_stride;
		    ActiveMessage<Write1DMessage> amsg(dst_node,
						       src_buf, bytes_per_line,
						       lines, src_stride,
						       dst_buf);
		    amsg->next_xd_guid = out_port->peer_guid;
		    amsg->next_port_idx = out_port->peer_port_idx;
		    amsg->span_start = out_span_start;

		    // reads aren't consumed until local completion, but
		    //  only ask if we have a previous xd that's going to
		    //  care
		    if(in_port->peer_guid != XFERDES_NO_GUID) {
		      // a ReadBytesUpdater holds a reference to the xd
		      add_reference();
		      amsg.add_local_completion(ReadBytesUpdater(this,
								 input_control.current_io_port,
								 in_span_start,
								 bytes));
		    }
		    in_span_start += bytes;
		    // the write isn't complete until it's ack'd by the target
		    amsg.add_remote_completion(WriteBytesUpdater(this,
								 output_control.current_io_port,
								 out_span_start,
								 bytes));
		    out_span_start += bytes;

		    amsg.commit();
		    in_alc.advance(1, lines);
		    out_alc.advance(0, bytes);
		  } else {
		    // gather: assemble data
		    bytes = src_ga_maxbytes;
		    ActiveMessage<Write1DMessage> amsg(dst_node,
						       bytes,
						       dst_buf);
		    amsg->next_xd_guid = out_port->peer_guid;
		    amsg->next_port_idx = out_port->peer_port_idx;
		    amsg->span_start = out_span_start;

		    size_t todo = bytes;
		    while(true) {
		      if(in_dim > 0) {
			if((icount >= todo/2) || (in_dim == 1)) {
			  size_t chunk = std::min(todo, icount);
			  uintptr_t src = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(in_alc.get_offset(), chunk));
			  uintptr_t dst = reinterpret_cast<uintptr_t>(amsg.payload_ptr(chunk));
			  memcpy_1d(dst, src, chunk);
			  in_alc.advance(0, chunk);
			  todo -= chunk;
			} else {
			  size_t lines = std::min(todo / icount,
						  in_alc.remaining(1));

			  if(((icount * lines) >= todo/2) || (in_dim == 2)) {
			    uintptr_t src = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(in_alc.get_offset(), icount));
			    uintptr_t dst = reinterpret_cast<uintptr_t>(amsg.payload_ptr(icount * lines));
			    memcpy_2d(dst, icount /*lstride*/,
				      src, in_alc.get_stride(1),
				      icount, lines);
			    in_alc.advance(1, lines);
			    todo -= icount * lines;
			  } else {
			    size_t planes = std::min(todo / (icount * lines),
						     in_alc.remaining(2));
			    uintptr_t src = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(in_alc.get_offset(), icount));
			    uintptr_t dst = reinterpret_cast<uintptr_t>(amsg.payload_ptr(icount * lines * planes));
			    memcpy_3d(dst,
				      icount /*lstride*/,
				      (icount * lines) /*pstride*/,
				      src,
				      in_alc.get_stride(1),
				      in_alc.get_stride(2),
				      icount, lines, planes);
			    in_alc.advance(2, planes);
			    todo -= icount * lines * planes;
			  }
			}
		      } else {
			assert(0);
		      }

		      if(todo == 0) break;

		      // read next entry
		      in_dim = in_alc.get_dim();
		      icount = in_alc.remaining(0);
		    }

		    // the write isn't complete until it's ack'd by the target
		    amsg.add_remote_completion(WriteBytesUpdater(this,
								 output_control.current_io_port,
								 out_span_start,
								 bytes));
		    out_span_start += bytes;

		    // assembly complete - send message
		    amsg.commit();

		    // we made a copy of input data, so "read" is complete
		    rseqcache.add_span(input_control.current_io_port,
				       in_span_start, bytes);
		    in_span_start += bytes;

		    out_alc.advance(0, bytes);
		  }
		}
		else if(dst_2d_maxbytes >= dst_sc_maxbytes) {
		  // 2D target
		  assert(0);
		} else {
		  // scatter target
		  assert(0);
		}

#ifdef DEBUG_REALM
		assert((bytes > 0) && (bytes <= bytes_left));
#endif
		total_bytes += bytes;

		// stop if it's been too long, but make sure we do at least the
		//  minimum number of bytes
		if((total_bytes >= min_xfer_size) && work_until.is_expired()) break;
	      }
	    } else {
	      // input but no output, so skip input bytes
	      total_bytes = max_bytes;
	      in_port->addrcursor.skip_bytes(total_bytes);
	      rseqcache.add_span(input_control.current_io_port,
				 in_span_start, total_bytes);
	      in_span_start += total_bytes;
	    }
	  } else {
	    if(out_port != 0) {
	      // output but no input, so skip output bytes
	      total_bytes = max_bytes;
	      out_port->addrcursor.skip_bytes(total_bytes);
	      wseqcache.add_span(output_control.current_io_port,
				 out_span_start, total_bytes);
	      out_span_start += total_bytes;
	    } else {
	      // skipping both input and output is possible for simultaneous
	      //  gather+scatter
	      total_bytes = max_bytes;
	    }
	  }

	  bool done = record_address_consumption(total_bytes, total_bytes);

	  did_work = true;

	  if(done || work_until.is_expired())
	    break;
	}

	rseqcache.flush();
	wseqcache.flush();

	return did_work;
#endif
      }

      void RemoteWriteXferDes::notify_request_read_done(Request* req)
      {
        xd_lock.lock();
        default_notify_request_read_done(req);
        xd_lock.unlock();
      }

      void RemoteWriteXferDes::notify_request_write_done(Request* req)
      {
        xd_lock.lock();
        default_notify_request_write_done(req);
        xd_lock.unlock();
      }

      void RemoteWriteXferDes::flush()
      {
        //xd_lock.lock();
        //xd_lock.unlock();
      }

      // doesn't do pre_bytes_write updates, since the remote write message
      //  takes care of it with lower latency (except for zero-byte
      //  termination updates)
      void RemoteWriteXferDes::update_bytes_write(int port_idx, size_t offset, size_t size)
      {
	XferPort *out_port = &output_ports[port_idx];
	size_t inc_amt = out_port->seq_local.add_span(offset, size);
	log_xd.info() << "bytes_write: " << std::hex << guid << std::dec
		      << "(" << port_idx << ") " << offset << "+" << size << " -> " << inc_amt;

	// pre_bytes_write update was handled in the remote AM handler
	if(out_port->peer_guid != XFERDES_NO_GUID) {
	  // update bytes total if needed (and available)
	  if(out_port->needs_pbt_update.load() &&
	     iteration_completed.load_acquire() &&
             (out_port->local_bytes_total == out_port->local_bytes_cons.load())) {
	    // exchange sets the flag to false and tells us previous value
	    if(out_port->needs_pbt_update.exchange(false))
	      xferDes_queue->update_pre_bytes_total(out_port->peer_guid,
						    out_port->peer_port_idx,
						    out_port->local_bytes_total);
	  }
        }

        // subtract bytes written from the pending count - if that causes it to
        //  go to zero, we can mark the transfer completed and update progress
        //  in case the xd is just waiting for that
        // NOTE: as soon as we set `transfer_completed`, the other references
        //  to this xd may be removed, so do this last, and hold a reference of
        //  our own long enough to call update_progress
        if(inc_amt > 0) {
          int64_t prev = bytes_write_pending.fetch_sub(inc_amt);
          if(prev > 0)
            log_xd.info() << "completion: xd=" << std::hex << guid << std::dec
                          << " remaining=" << (prev - inc_amt);
          if(inc_amt == static_cast<size_t>(prev)) {
            add_reference();
            transfer_completed.store_release(true);
            update_progress();
            remove_reference();
          }
        }
      }

      /*static*/
      void RemoteWriteXferDes::Write1DMessage::handle_message(NodeID sender,
							      const RemoteWriteXferDes::Write1DMessage &args,
							      const void *data,
							      size_t datalen)
      {
        // assert data copy is in right position
        //assert(data == args.dst_buf);

	log_xd.info() << "remote write recieved: next=" << args.next_xd_guid
		      << " start=" << args.span_start
		      << " size=" << datalen;

	// if requested, notify (probably-local) next XD
	if(args.next_xd_guid != XferDes::XFERDES_NO_GUID)
	  XferDesQueue::get_singleton()->update_pre_bytes_write(args.next_xd_guid,
								args.next_port_idx,
								args.span_start,
								datalen);
      }

      /*static*/ bool RemoteWriteXferDes::Write1DMessage::handle_inline(NodeID sender,
									const RemoteWriteXferDes::Write1DMessage &args,
									const void *data,
									size_t datalen,
									TimeLimit work_until)
      {
	handle_message(sender, args, data, datalen);
	return true;
      }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Channel::SupportedPath
  //

  Channel::SupportedPath& Channel::SupportedPath::set_max_dim(int src_and_dst_dim)
  {
    for(SupportedPath *p = this; p; p = p->chain)
      p->max_src_dim = p->max_dst_dim = src_and_dst_dim;
    return *this;
  }

  Channel::SupportedPath& Channel::SupportedPath::set_max_dim(int src_dim,
                                                              int dst_dim)
  {
    for(SupportedPath *p = this; p; p = p->chain) {
      p->max_src_dim = src_dim;
      p->max_dst_dim = dst_dim;
    }
    return *this;
  }

  Channel::SupportedPath& Channel::SupportedPath::allow_redops()
  {
    for(SupportedPath *p = this; p; p = p->chain)
      p->redops_allowed = true;
    return *this;
  }

  Channel::SupportedPath& Channel::SupportedPath::allow_serdez()
  {
    for(SupportedPath *p = this; p; p = p->chain)
      p->serdez_allowed = true;
    return *this;
  }

  void Channel::SupportedPath::populate_memory_bitmask(span<const Memory> mems,
                                                       NodeID node,
                                                       Channel::SupportedPath::MemBitmask& bitmask)
  {
    bitmask.node = node;

    for(size_t i = 0; i < SupportedPath::MemBitmask::BITMASK_SIZE; i++)
      bitmask.mems[i] = bitmask.ib_mems[i] = 0;

    for(size_t i = 0; i < mems.size(); i++)
      if(mems[i].exists() && (NodeID(ID(mems[i]).memory_owner_node()) == node)) {
        if(ID(mems[i]).is_memory())
          bitmask.mems[ID(mems[i]).memory_mem_idx() >> 6] |= (uint64_t(1) << (ID(mems[i]).memory_mem_idx() & 63));
        else if(ID(mems[i]).is_ib_memory())
          bitmask.ib_mems[ID(mems[i]).memory_mem_idx() >> 6] |= (uint64_t(1) << (ID(mems[i]).memory_mem_idx() & 63));
      }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Channel
  //

      std::ostream& operator<<(std::ostream& os, const Channel::SupportedPath& p)
      {
	switch(p.src_type) {
	case Channel::SupportedPath::SPECIFIC_MEMORY:
	  { os << "src=" << p.src_mem; break; }
	case Channel::SupportedPath::LOCAL_KIND:
	  { os << "src=" << p.src_kind << "(lcl)"; break; }
	case Channel::SupportedPath::GLOBAL_KIND:
	  { os << "src=" << p.src_kind << "(gbl)"; break; }
	case Channel::SupportedPath::LOCAL_RDMA:
	  { os << "src=rdma(lcl)"; break; }
	case Channel::SupportedPath::REMOTE_RDMA:
	  { os << "src=rdma(rem)"; break; }
        case Channel::SupportedPath::MEMORY_BITMASK:
          { os << "src=" << p.src_bitmask.node << '/';
            bool first = true;
            for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
              for(int j = 0; j < 64; j++)
                if((p.src_bitmask.mems[i] & (uint64_t(1) << j)) != 0) {
                  if(!first) os << ",";
                  first = false;
                  os << (64*i + j);
                }
            if(first) os << '-';
            os << '/';
            first = true;
            for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
              for(int j = 0; j < 64; j++)
                if((p.src_bitmask.ib_mems[i] & (uint64_t(1) << j)) != 0) {
                  if(!first) os << ",";
                  first = false;
                  os << (64*i + j);
                }
            if(first) os << '-';
            break;
          }
	default:
	  assert(0);
	}
	switch(p.dst_type) {
	case Channel::SupportedPath::SPECIFIC_MEMORY:
	  { os << " dst=" << p.dst_mem; break; }
	case Channel::SupportedPath::LOCAL_KIND:
	  { os << " dst=" << p.dst_kind << "(lcl)"; break; }
	case Channel::SupportedPath::GLOBAL_KIND:
	  { os << " dst=" << p.dst_kind << "(gbl)"; break; }
	case Channel::SupportedPath::LOCAL_RDMA:
	  { os << " dst=rdma(lcl)"; break; }
	case Channel::SupportedPath::REMOTE_RDMA:
	  { os << " dst=rdma(rem)"; break; }
        case Channel::SupportedPath::MEMORY_BITMASK:
          { os << " dst=" << p.dst_bitmask.node << '/';
            bool first = true;
            for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
              for(int j = 0; j < 64; j++)
                if((p.dst_bitmask.mems[i] & (uint64_t(1) << j)) != 0) {
                  if(!first) os << ",";
                  first = false;
                  os << (64*i + j);
                }
            if(first) os << '-';
            os << '/';
            first = true;
            for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
              for(int j = 0; j < 64; j++)
                if((p.dst_bitmask.ib_mems[i] & (uint64_t(1) << j)) != 0) {
                  if(!first) os << ",";
                  first = false;
                  os << (64*i + j);
                }
            if(first) os << '-';
            break;
          }
	default:
	  assert(0);
	}
	os << " bw=" << p.bandwidth << " lat=" << p.latency;
	if(p.serdez_allowed)
	  os << " serdez";
	if(p.redops_allowed)
	  os << " redop";
	return os;
      }
	  
      RemoteChannelInfo *Channel::construct_remote_info() const
      {
        return new SimpleRemoteChannelInfo(node, kind,
                                           reinterpret_cast<uintptr_t>(this),
                                           paths);
      }

      void Channel::print(std::ostream& os) const
      {
	os << "channel{ node=" << node << " kind=" << kind << " paths=[";
	if(!paths.empty()) {
	  for(std::vector<SupportedPath>::const_iterator it = paths.begin();
	      it != paths.end();
	      ++it)
	    os << "\n    " << *it;
	  os << "\n";
	}
	os << "] }";
      }

      const std::vector<Channel::SupportedPath>& Channel::get_paths(void) const
      {
	return paths;
      }
	  
      uint64_t Channel::supports_path(Memory src_mem, Memory dst_mem,
                                      CustomSerdezID src_serdez_id,
                                      CustomSerdezID dst_serdez_id,
                                      ReductionOpID redop_id,
                                      size_t total_bytes,
                                      const std::vector<size_t> *src_frags,
                                      const std::vector<size_t> *dst_frags,
                                      XferDesKind *kind_ret /*= 0*/,
                                      unsigned *bw_ret /*= 0*/,
                                      unsigned *lat_ret /*= 0*/)
      {
	for(std::vector<SupportedPath>::const_iterator it = paths.begin();
	    it != paths.end();
	    ++it) {
	  if(!it->serdez_allowed && ((src_serdez_id != 0) ||
				     (dst_serdez_id != 0)))
	    continue;
	  if(!it->redops_allowed && (redop_id != 0))
	    continue;

	  bool src_ok = false;
	  switch(it->src_type) {
	    case SupportedPath::SPECIFIC_MEMORY: {
	      src_ok = (src_mem == it->src_mem);
	      break;
	    }
	    case SupportedPath::LOCAL_KIND: {
	      src_ok = (src_mem.exists() &&
			(src_mem.kind() == it->src_kind) &&
			(NodeID(ID(src_mem).memory_owner_node()) == node));
	      break;
	    }
	    case SupportedPath::GLOBAL_KIND: {
	      src_ok = (src_mem.exists() &&
			(src_mem.kind() == it->src_kind));
	      break;
	    }
	    case SupportedPath::LOCAL_RDMA: {
	      if(src_mem.exists() &&
		 (NodeID(ID(src_mem).memory_owner_node()) == node)) {
		MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
		// detection of rdma-ness depends on whether memory is
		//  local/remote to us, not the channel
		if(NodeID(ID(src_mem).memory_owner_node()) == Network::my_node_id) {
		  src_ok = (src_impl->get_rdma_info(Network::single_network) != nullptr);
		} else {
		  RemoteAddress dummy;
		  src_ok = src_impl->get_remote_addr(0, dummy);
		}
	      }
	      break;
	    }
	    case SupportedPath::REMOTE_RDMA: {
	      if(src_mem.exists() &&
		 (NodeID(ID(src_mem).memory_owner_node()) != node)) {
		MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
		// detection of rdma-ness depends on whether memory is
		//  local/remote to us, not the channel
		if(NodeID(ID(src_mem).memory_owner_node()) == Network::my_node_id) {
		  src_ok = (src_impl->get_rdma_info(Network::single_network) != nullptr);
		} else {
		  RemoteAddress dummy;
		  src_ok = src_impl->get_remote_addr(0, dummy);
		}
	      }
	      break;
	    }
            case SupportedPath::MEMORY_BITMASK: {
              ID src_id(src_mem);
              if(NodeID(src_id.memory_owner_node()) == it->src_bitmask.node) {
                if(src_id.is_memory())
                  src_ok = ((it->src_bitmask.mems[src_id.memory_mem_idx() >> 6] &
                             (uint64_t(1) << (src_id.memory_mem_idx() & 63))) != 0);
                else if(src_id.is_ib_memory())
                  src_ok = ((it->src_bitmask.ib_mems[src_id.memory_mem_idx() >> 6] &
                             (uint64_t(1) << (src_id.memory_mem_idx() & 63))) != 0);
                else
                  src_ok = false;  // consider asserting on a non-memory ID?
              } else
                src_ok = false;
              break;
            }
	  }
	  if(!src_ok)
	    continue;

	  bool dst_ok = false;
	  switch(it->dst_type) {
	    case SupportedPath::SPECIFIC_MEMORY: {
	      dst_ok = (dst_mem == it->dst_mem);
	      break;
	    }
	    case SupportedPath::LOCAL_KIND: {
	      dst_ok = ((dst_mem.kind() == it->dst_kind) &&
			(NodeID(ID(dst_mem).memory_owner_node()) == node));
	      break;
	    }
	    case SupportedPath::GLOBAL_KIND: {
	      dst_ok = (dst_mem.kind() == it->dst_kind);
	      break;
	    }
	    case SupportedPath::LOCAL_RDMA: {
	      if(NodeID(ID(dst_mem).memory_owner_node()) == node) {
		MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
		// detection of rdma-ness depends on whether memory is
		//  local/remote to us, not the channel
		if(NodeID(ID(dst_mem).memory_owner_node()) == Network::my_node_id) {
		  dst_ok = (dst_impl->get_rdma_info(Network::single_network) != nullptr);
		} else {
		  RemoteAddress dummy;
		  dst_ok = dst_impl->get_remote_addr(0, dummy);
		}
	      }
	      break;
	    }
	    case SupportedPath::REMOTE_RDMA: {
	      if(NodeID(ID(dst_mem).memory_owner_node()) != node) {
		MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
		// detection of rdma-ness depends on whether memory is
		//  local/remote to us, not the channel
		if(NodeID(ID(dst_mem).memory_owner_node()) == Network::my_node_id) {
		  dst_ok = (dst_impl->get_rdma_info(Network::single_network) != nullptr);
		} else {
		  RemoteAddress dummy;
		  dst_ok = dst_impl->get_remote_addr(0, dummy);
		}
	      }
	      break;
	    }
            case SupportedPath::MEMORY_BITMASK: {
              ID dst_id(dst_mem);
              if(NodeID(dst_id.memory_owner_node()) == it->dst_bitmask.node) {
                if(dst_id.is_memory())
                  dst_ok = ((it->dst_bitmask.mems[dst_id.memory_mem_idx() >> 6] &
                             (uint64_t(1) << (dst_id.memory_mem_idx() & 63))) != 0);
                else if(dst_id.is_ib_memory())
                  dst_ok = ((it->dst_bitmask.ib_mems[dst_id.memory_mem_idx() >> 6] &
                             (uint64_t(1) << (dst_id.memory_mem_idx() & 63))) != 0);
                else
                  dst_ok = false;  // consider asserting on a non-memory ID?
              } else
                dst_ok = false;
              break;
            }
	  }
	  if(!dst_ok)
	    continue;

	  // match
	  if(kind_ret) *kind_ret = it->xd_kind;
	  if(bw_ret) *bw_ret = it->bandwidth;
	  if(lat_ret) *lat_ret = it->latency;

          // estimate transfer time
          uint64_t xfer_time = uint64_t(total_bytes) * 1000 / it->bandwidth;
          size_t frags = 1;
          if(src_frags)
            frags = std::max(frags, (*src_frags)[std::min<size_t>(src_frags->size()-1,
                                                                  it->max_src_dim)]);
          if(dst_frags)
            frags = std::max(frags, (*dst_frags)[std::min<size_t>(dst_frags->size()-1,
                                                                  it->max_dst_dim)]);
          xfer_time += uint64_t(frags) * it->frag_overhead;

          // make sure returned value is strictly positive
	  return std::max<uint64_t>(xfer_time, 1);
	}

	return 0;
      }

      // sometimes we need to return a reference to a SupportedPath that won't
      //  actually be added to a channel
      Channel::SupportedPath dummy_supported_path;

      Channel::SupportedPath& Channel::add_path(span<const Memory> src_mems,
                                                span<const Memory> dst_mems,
                                                unsigned bandwidth, unsigned latency,
                                                unsigned frag_overhead,
                                                XferDesKind xd_kind)
      {
        NodeSet src_nodes;
        for(size_t i = 0; i < src_mems.size(); i++)
          if(src_mems[i].exists())
            src_nodes.add(ID(src_mems[i]).memory_owner_node());
          else
            src_nodes.add(Network::max_node_id + 1);  // src fill placeholder

        NodeSet dst_nodes;
        for(size_t i = 0; i < dst_mems.size(); i++)
          if(dst_mems[i].exists())
            dst_nodes.add(ID(dst_mems[i]).memory_owner_node());

        if(src_nodes.empty() || dst_nodes.empty()) {
          // don't actually add a path
          return dummy_supported_path;
        }

        size_t num_new = src_nodes.size() * dst_nodes.size();
	size_t first_idx = paths.size();
	paths.resize(first_idx + num_new);

        SupportedPath *cur_sp = &paths[first_idx];
        NodeSetIterator src_iter = src_nodes.begin();
        NodeSetIterator dst_iter = dst_nodes.begin();

        while(true) {
          if(src_mems.size() == 1) {
            cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
            cur_sp->src_mem = src_mems[0];
          } else if(*src_iter > Network::max_node_id) {
            cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
            cur_sp->src_mem = Memory::NO_MEMORY; // src fill
          } else {
            cur_sp->src_type = SupportedPath::MEMORY_BITMASK;
            cur_sp->populate_memory_bitmask(src_mems, *src_iter, cur_sp->src_bitmask);
          }

          if(dst_mems.size() == 1) {
            cur_sp->dst_type = SupportedPath::SPECIFIC_MEMORY;
            cur_sp->dst_mem = dst_mems[0];
          } else {
            cur_sp->dst_type = SupportedPath::MEMORY_BITMASK;
            cur_sp->populate_memory_bitmask(dst_mems, *dst_iter, cur_sp->dst_bitmask);
          }

          cur_sp->bandwidth = bandwidth;
          cur_sp->latency = latency;
          cur_sp->frag_overhead = frag_overhead;
          cur_sp->max_src_dim = cur_sp->max_dst_dim = 1; // default
          cur_sp->redops_allowed = false; // default
          cur_sp->serdez_allowed = false; // default
          cur_sp->xd_kind = xd_kind;

          // bump iterators, wrapping dst if not done with src
          ++dst_iter;
          if(dst_iter == dst_nodes.end()) {
            ++src_iter;
            if(src_iter == src_nodes.end()) {
              // end of chain and of loop
              cur_sp->chain = 0;
              break;
            }
            dst_iter = dst_nodes.begin();
          }
          // not end of chain, so connect to next before bumping current pointer
          cur_sp->chain = cur_sp + 1;
          ++cur_sp;
        }
#ifdef DEBUG_REALM
        assert(cur_sp == (paths.data() + paths.size() - 1));
#endif
        // return reference to beginning of chain
        return paths[first_idx];
      }

      Channel::SupportedPath& Channel::add_path(span<const Memory> src_mems,
                                                Memory::Kind dst_kind, bool dst_global,
                                                unsigned bandwidth, unsigned latency,
                                                unsigned frag_overhead,
                                                XferDesKind xd_kind)
      {
        NodeSet src_nodes;
        for(size_t i = 0; i < src_mems.size(); i++)
          if(src_mems[i].exists())
            src_nodes.add(ID(src_mems[i]).memory_owner_node());
          else
            src_nodes.add(Network::max_node_id + 1);  // src fill placeholder

        if(src_nodes.empty()) {
          // don't actually add a path
          return dummy_supported_path;
        }

        size_t num_new = src_nodes.size();
	size_t first_idx = paths.size();
	paths.resize(first_idx + num_new);

        SupportedPath *cur_sp = &paths[first_idx];
        NodeSetIterator src_iter = src_nodes.begin();

        while(true) {
          if(src_mems.size() == 1) {
            cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
            cur_sp->src_mem = src_mems[0];
          } else if(*src_iter > Network::max_node_id) {
            cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
            cur_sp->src_mem = Memory::NO_MEMORY; // src fill
          } else {
            cur_sp->src_type = SupportedPath::MEMORY_BITMASK;
            cur_sp->populate_memory_bitmask(src_mems, *src_iter, cur_sp->src_bitmask);
          }

          cur_sp->dst_type = (dst_global ? SupportedPath::GLOBAL_KIND :
                                           SupportedPath::LOCAL_KIND);
          cur_sp->dst_kind = dst_kind;

          cur_sp->bandwidth = bandwidth;
          cur_sp->latency = latency;
          cur_sp->frag_overhead = frag_overhead;
          cur_sp->max_src_dim = cur_sp->max_dst_dim = 1; // default
          cur_sp->redops_allowed = false; // default
          cur_sp->serdez_allowed = false; // default
          cur_sp->xd_kind = xd_kind;

          ++src_iter;
          if(src_iter == src_nodes.end()) {
            // end of chain and of loop
            cur_sp->chain = 0;
            break;
          }

          // not end of chain, so connect to next before bumping current pointer
          cur_sp->chain = cur_sp + 1;
          ++cur_sp;
        }
#ifdef DEBUG_REALM
        assert(cur_sp == (paths.data() + paths.size() - 1));
#endif
        // return reference to beginning of chain
        return paths[first_idx];
      }

      Channel::SupportedPath& Channel::add_path(Memory::Kind src_kind, bool src_global,
                                                span<const Memory> dst_mems,
                                                unsigned bandwidth, unsigned latency,
                                                unsigned frag_overhead,
                                                XferDesKind xd_kind)
      {
        NodeSet dst_nodes;
        for(size_t i = 0; i < dst_mems.size(); i++)
          if(dst_mems[i].exists())
            dst_nodes.add(ID(dst_mems[i]).memory_owner_node());

        if(dst_nodes.empty()) {
          // don't actually add a path
          return dummy_supported_path;
        }

        size_t num_new = dst_nodes.size();
	size_t first_idx = paths.size();
	paths.resize(first_idx + num_new);

        SupportedPath *cur_sp = &paths[first_idx];
        NodeSetIterator dst_iter = dst_nodes.begin();

        while(true) {
          cur_sp->src_type = (src_global ? SupportedPath::GLOBAL_KIND :
                                           SupportedPath::LOCAL_KIND);
          cur_sp->src_kind = src_kind;

          if(dst_mems.size() == 1) {
            cur_sp->dst_type = SupportedPath::SPECIFIC_MEMORY;
            cur_sp->dst_mem = dst_mems[0];
          } else {
            cur_sp->dst_type = SupportedPath::MEMORY_BITMASK;
            cur_sp->populate_memory_bitmask(dst_mems, *dst_iter, cur_sp->dst_bitmask);
          }

          cur_sp->bandwidth = bandwidth;
          cur_sp->latency = latency;
          cur_sp->frag_overhead = frag_overhead;
          cur_sp->max_src_dim = cur_sp->max_dst_dim = 1; // default
          cur_sp->redops_allowed = false; // default
          cur_sp->serdez_allowed = false; // default
          cur_sp->xd_kind = xd_kind;

          ++dst_iter;
          if(dst_iter == dst_nodes.end()) {
            // end of chain and of loop
            cur_sp->chain = 0;
            break;
          }

          // not end of chain, so connect to next before bumping current pointer
          cur_sp->chain = cur_sp + 1;
          ++cur_sp;
        }
#ifdef DEBUG_REALM
        assert(cur_sp == (paths.data() + paths.size() - 1));
#endif
        // return reference to beginning of chain
        return paths[first_idx];
      }

      Channel::SupportedPath& Channel::add_path(Memory::Kind src_kind, bool src_global,
                                                Memory::Kind dst_kind, bool dst_global,
                                                unsigned bandwidth, unsigned latency,
                                                unsigned frag_overhead,
                                                XferDesKind xd_kind)
      {
	size_t idx = paths.size();
	paths.resize(idx + 1);
	SupportedPath &p = paths[idx];
        p.chain = 0;

	p.src_type = (src_global ? SupportedPath::GLOBAL_KIND :
		                   SupportedPath::LOCAL_KIND);
	p.src_kind = src_kind;
	p.dst_type = (dst_global ? SupportedPath::GLOBAL_KIND :
		                   SupportedPath::LOCAL_KIND);
	p.dst_kind = dst_kind;
	p.bandwidth = bandwidth;
	p.latency = latency;
        p.frag_overhead = frag_overhead;
        p.max_src_dim = p.max_dst_dim = 1; // default
	p.redops_allowed = false; // default
	p.serdez_allowed = false; // default
	p.xd_kind = xd_kind;
        return p;
      }

      // TODO: allow rdma path to limit by kind?
      Channel::SupportedPath& Channel::add_path(bool local_loopback,
                                                unsigned bandwidth, unsigned latency,
                                                unsigned frag_overhead,
                                                XferDesKind xd_kind)
      {
	size_t idx = paths.size();
	paths.resize(idx + 1);
	SupportedPath &p = paths[idx];
        p.chain = 0;

	p.src_type = SupportedPath::LOCAL_RDMA;
	p.dst_type = (local_loopback ? SupportedPath::LOCAL_RDMA :
		                       SupportedPath::REMOTE_RDMA);
	p.bandwidth = bandwidth;
	p.latency = latency;
        p.frag_overhead = frag_overhead;
        p.max_src_dim = p.max_dst_dim = 1; // default
	p.redops_allowed = false; // default
	p.serdez_allowed = false; // default
	p.xd_kind = xd_kind;
        return p;
      }

      long Channel::progress_xd(XferDes *xd, long max_nr)
      {
	const long MAX_NR = 8;
	Request *requests[MAX_NR];
	long nr_got = xd->get_requests(requests, std::min(max_nr, MAX_NR));
	if(nr_got == 0) return 0;
	long nr_submitted = submit(requests, nr_got);
	assert(nr_got == nr_submitted);
	return nr_submitted;
      }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SimpleXferDesFactory
  //

  SimpleXferDesFactory::SimpleXferDesFactory(uintptr_t _channel)
    : channel(_channel)
  {}

  void SimpleXferDesFactory::release()
  {
    // do nothing since we are a singleton
  }

  void SimpleXferDesFactory::create_xfer_des(uintptr_t dma_op,
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
    if(target_node == Network::my_node_id) {
      // local creation
      //assert(!inst.exists());
      LocalChannel *c = reinterpret_cast<LocalChannel *>(channel);
      XferDes *xd = c->create_xfer_des(dma_op, launch_node, guid,
				       inputs_info, outputs_info,
				       priority, redop_info,
				       fill_data, fill_size, fill_total);

      c->enqueue_ready_xd(xd);
    } else {
      // remote creation
      Serialization::ByteCountSerializer bcs;
      {
	bool ok = ((bcs << inputs_info) &&
		   (bcs << outputs_info) &&
		   (bcs << priority) &&
		   (bcs << redop_info) &&
                   (bcs << fill_total));
	if(ok && (fill_size > 0))
	  ok = bcs.append_bytes(fill_data, fill_size);
	assert(ok);
      }
      size_t req_size = bcs.bytes_used();
      ActiveMessage<SimpleXferDesCreateMessage> amsg(target_node, req_size);
      //amsg->inst = inst;
      amsg->launch_node = launch_node;
      amsg->guid = guid;
      amsg->dma_op = dma_op;
      amsg->channel = channel;
      {
	bool ok = ((amsg << inputs_info) &&
		   (amsg << outputs_info) &&
		   (amsg << priority) &&
		   (amsg << redop_info) &&
                   (amsg << fill_total));
	if(ok && (fill_size > 0))
	  amsg.add_payload(fill_data, fill_size);
	assert(ok);
      }
      amsg.commit();

      // normally ownership of input and output iterators would be taken
      //  by the local XferDes we create, but here we sent a copy, so delete
      //  the originals
      for(std::vector<XferDesPortInfo>::const_iterator it = inputs_info.begin();
	  it != inputs_info.end();
	  ++it)
	delete it->iter;

      for(std::vector<XferDesPortInfo>::const_iterator it = outputs_info.begin();
	  it != outputs_info.end();
	  ++it)
	delete it->iter;
    }
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class SimpleXferDesCreateMessage
  //

  /*static*/ void SimpleXferDesCreateMessage::handle_message(NodeID sender,
							     const SimpleXferDesCreateMessage &args,
							     const void *msgdata,
							     size_t msglen)
  {
    std::vector<XferDesPortInfo> inputs_info, outputs_info;
    int priority = 0;
    XferDesRedopInfo redop_info;
    size_t fill_total = 0;

    Realm::Serialization::FixedBufferDeserializer fbd(msgdata, msglen);

    bool ok = ((fbd >> inputs_info) &&
	       (fbd >> outputs_info) &&
	       (fbd >> priority) &&
	       (fbd >> redop_info) &&
               (fbd >> fill_total));
    assert(ok);
    const void *fill_data;
    size_t fill_size;
    if(fbd.bytes_left() == 0) {
      fill_data = 0;
      fill_size = 0;
    } else {
      fill_size = fbd.bytes_left();
      fill_data = fbd.peek_bytes(fill_size);
    }
  
    //assert(!args.inst.exists());
    LocalChannel *c = reinterpret_cast<LocalChannel *>(args.channel);
    XferDes *xd = c->create_xfer_des(args.dma_op, args.launch_node,
				     args.guid,
				     inputs_info,
				     outputs_info,
				     priority,
				     redop_info,
				     fill_data, fill_size, fill_total);

    c->enqueue_ready_xd(xd);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class NotifyXferDesCompleteMessage
  //

  /*static*/ void NotifyXferDesCompleteMessage::handle_message(NodeID sender,
							       const NotifyXferDesCompleteMessage &args,
							       const void *data,
							       size_t datalen)
  {
    args.op->notify_xd_completion(args.xd_id);
  }

  /*static*/ void NotifyXferDesCompleteMessage::send_request(NodeID target, TransferOperation *op, XferDesID xd_id)
  {
    ActiveMessage<NotifyXferDesCompleteMessage> amsg(target);
    amsg->op = op;
    amsg->xd_id = xd_id;
    amsg.commit();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalChannel
  //

  LocalChannel::LocalChannel(XferDesKind _kind)
    : Channel(_kind)
    , factory_singleton(reinterpret_cast<uintptr_t>(this))
  {}

  XferDesFactory *LocalChannel::get_factory()
  {
    return &factory_singleton;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SimpleRemoteChannelInfo
  //

  SimpleRemoteChannelInfo::SimpleRemoteChannelInfo()
  {}

  SimpleRemoteChannelInfo::SimpleRemoteChannelInfo(NodeID _owner,
                                                   XferDesKind _kind,
                                                   uintptr_t _remote_ptr,
                                                   const std::vector<Channel::SupportedPath>& _paths)
    : owner(_owner)
    , kind(_kind)
    , remote_ptr(_remote_ptr)
    , paths(_paths)
  {}

  RemoteChannel *SimpleRemoteChannelInfo::create_remote_channel()
  {
    RemoteChannel *rc = new RemoteChannel(remote_ptr);
    rc->node = owner;
    rc->kind = kind;
    rc->paths.swap(paths);
    return rc;
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                      SimpleRemoteChannelInfo> SimpleRemoteChannelInfo::serdez_subclass;


  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteChannel
  //

      RemoteChannel::RemoteChannel(uintptr_t _remote_ptr)
	: Channel(XFER_NONE)
	, factory_singleton(_remote_ptr)
      {}

      void RemoteChannel::shutdown()
      {}

      XferDesFactory *RemoteChannel::get_factory()
      {
	return &factory_singleton;
      }

      long RemoteChannel::submit(Request** requests, long nr)
      {
	assert(0);
	return 0;
      }

      void RemoteChannel::pull()
      {
	assert(0);
      }

      long RemoteChannel::available()
      {
	assert(0);
	return 0;
      }

      uint64_t RemoteChannel::supports_path(Memory src_mem, Memory dst_mem,
                                            CustomSerdezID src_serdez_id,
                                            CustomSerdezID dst_serdez_id,
                                            ReductionOpID redop_id,
                                            size_t total_bytes,
                                            const std::vector<size_t> *src_frags,
                                            const std::vector<size_t> *dst_frags,
                                            XferDesKind *kind_ret /*= 0*/,
                                            unsigned *bw_ret /*= 0*/,
                                            unsigned *lat_ret /*= 0*/)
      {
	// simultaneous serialization/deserialization not
	//  allowed anywhere right now
	if((src_serdez_id != 0) && (dst_serdez_id != 0))
	  return 0;

	// fall through to normal checks
	return Channel::supports_path(src_mem, dst_mem,
				      src_serdez_id, dst_serdez_id, redop_id,
                                      total_bytes, src_frags, dst_frags,
				      kind_ret, bw_ret, lat_ret);
      }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemcpyChannel
  //

      MemcpyChannel::MemcpyChannel(BackgroundWorkManager *bgwork)
	: SingleXDQChannel<MemcpyChannel,MemcpyXferDes>(bgwork,
							XFER_MEM_CPY,
							"memcpy channel")
      {
        //cbs = (MemcpyRequest**) calloc(max_nr, sizeof(MemcpyRequest*));
	unsigned bw = 5000; // HACK - estimate at 5 GB/s
	unsigned latency = 100; // HACK - estimate at 100ns
        unsigned frag_overhead = 100; // HACK - estimate at 100ns

        // all local cpu memories are valid sources and dests
        std::vector<Memory> local_cpu_mems;
        enumerate_local_cpu_memories(local_cpu_mems);

        add_path(local_cpu_mems, local_cpu_mems,
                 bw, latency, frag_overhead, XFER_MEM_CPY)
          .set_max_dim(3)
          .allow_serdez();

	xdq.add_to_manager(bgwork);
      }

      MemcpyChannel::~MemcpyChannel()
      {
        //free(cbs);
      }

      /*static*/ void MemcpyChannel::enumerate_local_cpu_memories(std::vector<Memory>& mems)
      {
        Node& n = get_runtime()->nodes[Network::my_node_id];

        for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
            it != n.memories.end();
            ++it)
          if(((*it)->lowlevel_kind == Memory::SYSTEM_MEM) ||
             ((*it)->lowlevel_kind == Memory::REGDMA_MEM) ||
             ((*it)->lowlevel_kind == Memory::Z_COPY_MEM) ||
             ((*it)->lowlevel_kind == Memory::SOCKET_MEM) ||
             ((*it)->lowlevel_kind == Memory::GPU_MANAGED_MEM))
            mems.push_back((*it)->me);

        for(std::vector<IBMemory *>::const_iterator it = n.ib_memories.begin();
            it != n.ib_memories.end();
            ++it)
          if(((*it)->lowlevel_kind == Memory::SYSTEM_MEM) ||
             ((*it)->lowlevel_kind == Memory::REGDMA_MEM) ||
             ((*it)->lowlevel_kind == Memory::Z_COPY_MEM) ||
             ((*it)->lowlevel_kind == Memory::SOCKET_MEM) ||
             ((*it)->lowlevel_kind == Memory::GPU_MANAGED_MEM))
            mems.push_back((*it)->me);
      }


      uint64_t MemcpyChannel::supports_path(Memory src_mem, Memory dst_mem,
                                            CustomSerdezID src_serdez_id,
                                            CustomSerdezID dst_serdez_id,
                                            ReductionOpID redop_id,
                                            size_t total_bytes,
                                            const std::vector<size_t> *src_frags,
                                            const std::vector<size_t> *dst_frags,
                                            XferDesKind *kind_ret /*= 0*/,
                                            unsigned *bw_ret /*= 0*/,
                                            unsigned *lat_ret /*= 0*/)
      {
	// simultaneous serialization/deserialization not
	//  allowed anywhere right now
	if((src_serdez_id != 0) && (dst_serdez_id != 0))
	  return 0;

	// fall through to normal checks
	return Channel::supports_path(src_mem, dst_mem,
				      src_serdez_id, dst_serdez_id, redop_id,
                                      total_bytes, src_frags, dst_frags,
				      kind_ret, bw_ret, lat_ret);
      }

      XferDes *MemcpyChannel::create_xfer_des(uintptr_t dma_op,
					      NodeID launch_node,
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
	return new MemcpyXferDes(dma_op, this, launch_node, guid,
				 inputs_info, outputs_info,
				 priority);
      }

      long MemcpyChannel::submit(Request** requests, long nr)
      {
        MemcpyRequest** mem_cpy_reqs = (MemcpyRequest**) requests;
        for (long i = 0; i < nr; i++) {
          MemcpyRequest* req = mem_cpy_reqs[i];
	  // handle 1-D, 2-D, and 3-D in a single loop
	  switch(req->dim) {
	  case Request::DIM_1D:
	    assert(req->nplanes == 1);
	    assert(req->nlines == 1);
	    break;
	  case Request::DIM_2D:
	    assert(req->nplanes == 1);
	    break;
	  case Request::DIM_3D:
	    // nothing to check
	    break;
	  default:
	    assert(0);
	  }
	  size_t rewind_src = 0;
	  size_t rewind_dst = 0;
	  XferDes::XferPort *in_port = &req->xd->input_ports[req->src_port_idx];
	  XferDes::XferPort *out_port = &req->xd->output_ports[req->dst_port_idx];
	  const CustomSerdezUntyped *src_serdez_op = in_port->serdez_op;
	  const CustomSerdezUntyped *dst_serdez_op = out_port->serdez_op;
	  if(src_serdez_op && !dst_serdez_op) {
	    // we manage write_bytes_total, write_seq_{pos,count}
	    req->write_seq_pos = out_port->local_bytes_total;
	  }
	  if(!src_serdez_op && dst_serdez_op) {
	    // we manage read_bytes_total, read_seq_{pos,count}
	    req->read_seq_pos = in_port->local_bytes_total;
	  }
	  {
	    char *wrap_buffer = 0;
	    bool wrap_buffer_malloced = false;
	    const size_t ALLOCA_LIMIT = 4096;
	    const char *src_p = (const char *)(req->src_base);
	    char *dst_p = (char *)(req->dst_base);
	    for (size_t j = 0; j < req->nplanes; j++) {
	      const char *src = src_p;
	      char *dst = dst_p;
	      for (size_t i = 0; i < req->nlines; i++) {
		if(src_serdez_op) {
		  if(dst_serdez_op) {
		    // serialization AND deserialization
		    assert(0);
		  } else {
		    // serialization
		    size_t field_size = src_serdez_op->sizeof_field_type;
		    size_t num_elems = req->nbytes / field_size;
		    assert((num_elems * field_size) == req->nbytes);
		    size_t maxser_size = src_serdez_op->max_serialized_size;
		    size_t max_bytes = num_elems * maxser_size;
		    // ask the dst iterator (which should be a
		    //  WrappingFIFOIterator for enough space to write all the
		    //  serialized data in the worst case
		    TransferIterator::AddressInfo dst_info;
		    size_t bytes_avail = out_port->iter->step(max_bytes,
							      dst_info,
							      0,
							      true /*tentative*/);
		    size_t bytes_used;
		    if(bytes_avail == max_bytes) {
		      // got enough space to do it all in one go
		      void *dst = out_port->mem->get_direct_ptr(dst_info.base_offset,
								bytes_avail);
		      assert(dst != 0);
		      bytes_used = src_serdez_op->serialize(src,
							    field_size,
							    num_elems,
							    dst);
		      if(bytes_used == max_bytes) {
			out_port->iter->confirm_step();
		      } else {
			out_port->iter->cancel_step();
			bytes_avail = out_port->iter->step(bytes_used,
							   dst_info,
							   0,
							   false /*!tentative*/);
			assert(bytes_avail == bytes_used);
		      }
		    } else {
		      // we didn't get the worst case amount, but it might be
		      //  enough
		      void *dst = out_port->mem->get_direct_ptr(dst_info.base_offset,
								bytes_avail);
		      assert(dst != 0);
		      size_t elems_done = 0;
		      size_t bytes_left = bytes_avail;
		      bytes_used = 0;
		      while((elems_done < num_elems) &&
			    (bytes_left >= maxser_size)) {
			size_t todo = std::min(num_elems - elems_done,
					       bytes_left / maxser_size);
			size_t amt = src_serdez_op->serialize(((const char *)src) + (elems_done * field_size),
							      field_size,
							      todo,
							      dst);
			assert(amt <= bytes_left);
			elems_done += todo;
			bytes_left -= amt;
			dst = ((char *)dst) + amt;
			bytes_used += amt;
		      }
		      if(elems_done == num_elems) {
			// we ended up getting all we needed without wrapping
			if(bytes_used == bytes_avail) {
			  out_port->iter->confirm_step();
			} else {
			  out_port->iter->cancel_step();
			  bytes_avail = out_port->iter->step(bytes_used,
							     dst_info,
							     0,
							     false /*!tentative*/);
			  assert(bytes_avail == bytes_used);
			}
		      } else {
			// did we get lucky and finish on the wrap boundary?
			if(bytes_left == 0) {
			  out_port->iter->confirm_step();
			} else {
			  // need a temp buffer to deal with wraparound
			  if(!wrap_buffer) {
			    if(maxser_size > ALLOCA_LIMIT) {
			      wrap_buffer_malloced = true;
			      wrap_buffer = (char *)malloc(maxser_size);
			    } else {
			      wrap_buffer = (char *)alloca(maxser_size);
			    }
			  }
			  while((elems_done < num_elems) && (bytes_left > 0)) {
			    // serialize one element into our buffer
			    size_t amt = src_serdez_op->serialize(((const char *)src) + (elems_done * field_size),
								  wrap_buffer);
			    if(amt < bytes_left) {
			      memcpy(dst, wrap_buffer, amt);
			      bytes_left -= amt;
			      dst = ((char *)dst) + amt;
			    } else {
			      memcpy(dst, wrap_buffer, bytes_left);
			      out_port->iter->confirm_step();
			      if(amt > bytes_left) {
				size_t amt2 = out_port->iter->step(amt - bytes_left,
								      dst_info,
								      0,
								      false /*!tentative*/);
				assert(amt2 == (amt - bytes_left));
				void *dst = out_port->mem->get_direct_ptr(dst_info.base_offset,
									     amt2);
				assert(dst != 0);
				memcpy(dst, wrap_buffer+bytes_left, amt2);
			      }
			      bytes_left = 0;
			    }
			    elems_done++;
			    bytes_used += amt;
			  }
			  // if we still finished with bytes left over, give 
			  //  them back to the iterator
			  if(bytes_left > 0) {
			    assert(elems_done == num_elems);
			    out_port->iter->cancel_step();
			    size_t amt = out_port->iter->step(bytes_used,
							      dst_info,
							      0,
							      false /*!tentative*/);
			    assert(amt == bytes_used);
			  }
			}

			// now that we're after the wraparound, any remaining
			//  elements are fairly straightforward
			if(elems_done < num_elems) {
			  size_t max_remain = ((num_elems - elems_done) * maxser_size);
			  size_t amt = out_port->iter->step(max_remain,
							    dst_info,
							    0,
							    true /*tentative*/);
			  assert(amt == max_remain); // no double-wrap
			  void *dst = out_port->mem->get_direct_ptr(dst_info.base_offset,
								    amt);
			  assert(dst != 0);
			  size_t amt2 = src_serdez_op->serialize(((const char *)src) + (elems_done * field_size),
								 field_size,
								 num_elems - elems_done,
								 dst);
			  bytes_used += amt2;
			  if(amt2 == max_remain) {
			    out_port->iter->confirm_step();
			  } else {
			    out_port->iter->cancel_step();
			    size_t amt3 = out_port->iter->step(amt2,
							       dst_info,
							       0,
							       false /*!tentative*/);
			    assert(amt3 == amt2);
			  }
			}
		      }
		    }
		    assert(bytes_used <= max_bytes);
		    if(bytes_used < max_bytes)
		      rewind_dst += (max_bytes - bytes_used);
		    out_port->local_bytes_total += bytes_used;
		  }
		} else {
		  if(dst_serdez_op) {
		    // deserialization
		    size_t field_size = dst_serdez_op->sizeof_field_type;
		    size_t num_elems = req->nbytes / field_size;
		    assert((num_elems * field_size) == req->nbytes);
		    size_t maxser_size = dst_serdez_op->max_serialized_size;
		    size_t max_bytes = num_elems * maxser_size;
		    // ask the srct iterator (which should be a
		    //  WrappingFIFOIterator for enough space to read all the
		    //  serialized data in the worst case
		    TransferIterator::AddressInfo src_info;
		    size_t bytes_avail = in_port->iter->step(max_bytes,
							     src_info,
							     0,
							     true /*tentative*/);
		    size_t bytes_used;
		    if(bytes_avail == max_bytes) {
		      // got enough space to do it all in one go
		      const void *src = in_port->mem->get_direct_ptr(src_info.base_offset,
								     bytes_avail);
		      assert(src != 0);
		      bytes_used = dst_serdez_op->deserialize(dst,
							      field_size,
							      num_elems,
							      src);
		      if(bytes_used == max_bytes) {
			in_port->iter->confirm_step();
		      } else {
			in_port->iter->cancel_step();
			bytes_avail = in_port->iter->step(bytes_used,
							  src_info,
							  0,
							  false /*!tentative*/);
			assert(bytes_avail == bytes_used);
		      }
		    } else {
		      // we didn't get the worst case amount, but it might be
		      //  enough
		      const void *src = in_port->mem->get_direct_ptr(src_info.base_offset,
								     bytes_avail);
		      assert(src != 0);
		      size_t elems_done = 0;
		      size_t bytes_left = bytes_avail;
		      bytes_used = 0;
		      while((elems_done < num_elems) &&
			    (bytes_left >= maxser_size)) {
			size_t todo = std::min(num_elems - elems_done,
					       bytes_left / maxser_size);
			size_t amt = dst_serdez_op->deserialize(((char *)dst) + (elems_done * field_size),
								field_size,
								todo,
								src);
			assert(amt <= bytes_left);
			elems_done += todo;
			bytes_left -= amt;
			src = ((const char *)src) + amt;
			bytes_used += amt;
		      }
		      if(elems_done == num_elems) {
			// we ended up getting all we needed without wrapping
			if(bytes_used == bytes_avail) {
			  in_port->iter->confirm_step();
			} else {
			  in_port->iter->cancel_step();
			  bytes_avail = in_port->iter->step(bytes_used,
							    src_info,
							    0,
							    false /*!tentative*/);
			  assert(bytes_avail == bytes_used);
			}
		      } else {
			// did we get lucky and finish on the wrap boundary?
			if(bytes_left == 0) {
			  in_port->iter->confirm_step();
			} else {
			  // need a temp buffer to deal with wraparound
			  if(!wrap_buffer) {
			    if(maxser_size > ALLOCA_LIMIT) {
			      wrap_buffer_malloced = true;
			      wrap_buffer = (char *)malloc(maxser_size);
			    } else {
			      wrap_buffer = (char *)alloca(maxser_size);
			    }
			  }
			  // keep a snapshot of the iterator in cse we don't wrap after all
			  Serialization::DynamicBufferSerializer dbs(64);
			  dbs << *(in_port->iter);
			  memcpy(wrap_buffer, src, bytes_left);
			  // get pointer to data on other side of wrap
			  in_port->iter->confirm_step();
			  size_t amt = in_port->iter->step(max_bytes - bytes_avail,
							   src_info,
							   0,
							   true /*tentative*/);
			  // it's actually ok for this to appear to come up short - due to
			  //  flow control we know we won't ever actually wrap around
			  //assert(amt == (max_bytes - bytes_avail));
			  const void *src = in_port->mem->get_direct_ptr(src_info.base_offset,
									 amt);
			  assert(src != 0);
			  memcpy(wrap_buffer + bytes_left, src, maxser_size - bytes_left);
			  src = ((const char *)src) + (maxser_size - bytes_left);

			  while((elems_done < num_elems) && (bytes_left > 0)) {
			    // deserialize one element from our buffer
			    amt = dst_serdez_op->deserialize(((char *)dst) + (elems_done * field_size),
							     wrap_buffer);
			    if(amt < bytes_left) {
			      // slide data, get a few more bytes
			      memmove(wrap_buffer,
				      wrap_buffer + amt,
				      maxser_size - amt);
			      memcpy(wrap_buffer + maxser_size, src, amt);
			      bytes_left -= amt;
			      src = ((const char *)src) + amt;
			    } else {
			      // update iterator to say how much wrapped data was actually used
			      in_port->iter->cancel_step();
			      if(amt > bytes_left) {
				size_t amt2 = in_port->iter->step(amt - bytes_left,
								  src_info,
								  0,
								  false /*!tentative*/);
				assert(amt2 == (amt - bytes_left));
			      }
			      bytes_left = 0;
			    }
			    elems_done++;
			    bytes_used += amt;
			  }
			  // if we still finished with bytes left, we have
			  //  to restore the iterator because we
			  //  can't double-cancel
			  if(bytes_left > 0) {
			    assert(elems_done == num_elems);
			    delete in_port->iter;
			    Serialization::FixedBufferDeserializer fbd(dbs.get_buffer(), dbs.bytes_used());
			    in_port->iter = TransferIterator::deserialize_new(fbd);
			    in_port->iter->cancel_step();
			    size_t amt2 = in_port->iter->step(bytes_used,
							      src_info,
							      0,
							      false /*!tentative*/);
			    assert(amt2 == bytes_used);
			  }
			}

			// now that we're after the wraparound, any remaining
			//  elements are fairly straightforward
			if(elems_done < num_elems) {
			  size_t max_remain = ((num_elems - elems_done) * maxser_size);
			  size_t amt = in_port->iter->step(max_remain,
							   src_info,
							   0,
							   true /*tentative*/);
			  assert(amt == max_remain); // no double-wrap
			  const void *src = in_port->mem->get_direct_ptr(src_info.base_offset,
									 amt);
			  assert(src != 0);
			  size_t amt2 = dst_serdez_op->deserialize(((char *)dst) + (elems_done * field_size),
								   field_size,
								   num_elems - elems_done,
								   src);
			  bytes_used += amt2;
			  if(amt2 == max_remain) {
			    in_port->iter->confirm_step();
			  } else {
			    in_port->iter->cancel_step();
			    size_t amt3 = in_port->iter->step(amt2,
							      src_info,
							      0,
							      false /*!tentative*/);
			    assert(amt3 == amt2);
			  }
			}
		      }
		    }
		    assert(bytes_used <= max_bytes);
		    if(bytes_used < max_bytes)
		      rewind_src += (max_bytes - bytes_used);
		    in_port->local_bytes_total += bytes_used;
		  } else {
		    // normal copy
		    memcpy(dst, src, req->nbytes);
		  }
		}
		if(req->dim == Request::DIM_1D) break;
		// serdez cases update src/dst directly
		// NOTE: this looks backwards, but it's not - a src serdez means it's the
		//  destination that moves unpredictably
		if(!dst_serdez_op) src += req->src_str;
		if(!src_serdez_op) dst += req->dst_str;
	      }
	      if((req->dim == Request::DIM_1D) ||
		 (req->dim == Request::DIM_2D)) break;
	      // serdez cases update src/dst directly - copy back to src/dst_p
	      src_p = (dst_serdez_op ? src : src_p + req->src_pstr);
	      dst_p = (src_serdez_op ? dst : dst_p + req->dst_pstr);
	    }
	    // clean up our wrap buffer, if we malloc'd it
	    if(wrap_buffer_malloced)
	      free(wrap_buffer);
	  }
	  if(src_serdez_op && !dst_serdez_op) {
	    // we manage write_bytes_total, write_seq_{pos,count}
	    req->write_seq_count = out_port->local_bytes_total - req->write_seq_pos;
	    if(rewind_dst > 0) {
	      //log_request.print() << "rewind dst: " << rewind_dst;
              // if we've finished iteration, it's too late to rewind the
              //  conservative count, so decrement the number of write bytes
              //  pending (we know we can't drive it to zero) as well
              if(req->xd->iteration_completed.load()) {
                int64_t prev = req->xd->bytes_write_pending.fetch_sub(rewind_dst);
                assert((prev > 0) && (static_cast<size_t>(prev) > rewind_dst));
              }
              out_port->local_bytes_cons.fetch_sub(rewind_dst);
	    }
	  } else
	    assert(rewind_dst == 0);
	  if(!src_serdez_op && dst_serdez_op) {
	    // we manage read_bytes_total, read_seq_{pos,count}
	    req->read_seq_count = in_port->local_bytes_total - req->read_seq_pos;
	    if(rewind_src > 0) {
	      //log_request.print() << "rewind src: " << rewind_src;
	      in_port->local_bytes_cons.fetch_sub(rewind_src);
	    }
	  } else
	      assert(rewind_src == 0);
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
        return nr;
        /*
        pending_lock.lock();
        //if (nr > 0)
          //printf("MemcpyChannel::submit[nr = %ld]\n", nr);
        for (long i = 0; i < nr; i++) {
          pending_queue.push_back(mem_cpy_reqs[i]);
        }
        if (sleep_threads) {
          pthread_cond_broadcast(&pending_cond);
          sleep_threads = false;
        }
        pending_lock.unlock();
        return nr;
        */
        /*
        for (int i = 0; i < nr; i++) {
          push_request(mem_cpy_reqs[i]);
          memcpy(mem_cpy_reqs[i]->dst_buf, mem_cpy_reqs[i]->src_buf, mem_cpy_reqs[i]->nbytes);
          mem_cpy_reqs[i]->xd->notify_request_read_done(mem_cpy_reqs[i]);
          mem_cpy_reqs[i]->xd->notify_request_write_done(mem_cpy_reqs[i]);
        }
        return nr;
        */
      }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemfillChannel
  //

  MemfillChannel::MemfillChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<MemfillChannel,MemfillXferDes>(bgwork,
						      XFER_MEM_FILL,
						      "memfill channel")
  {
    unsigned bw = 10000; // HACK - estimate at 10 GB/s
    unsigned latency = 100; // HACK - estimate at 100ns
    unsigned frag_overhead = 100; // HACK - estimate at 100ns

    // all local cpu memories are valid dests
    std::vector<Memory> local_cpu_mems;
    MemcpyChannel::enumerate_local_cpu_memories(local_cpu_mems);

    add_path(Memory::NO_MEMORY, local_cpu_mems,
             bw, latency, frag_overhead, XFER_MEM_FILL)
      .set_max_dim(3);

    xdq.add_to_manager(bgwork);
  }

  MemfillChannel::~MemfillChannel()
  {}

  XferDes *MemfillChannel::create_xfer_des(uintptr_t dma_op,
					   NodeID launch_node,
					   XferDesID guid,
					   const std::vector<XferDesPortInfo>& inputs_info,
					   const std::vector<XferDesPortInfo>& outputs_info,
					   int priority,
					   XferDesRedopInfo redop_info,
					   const void *fill_data,
                                           size_t fill_size,
                                           size_t fill_total)
  {
    assert(redop_info.id == 0); // TODO: add support
    assert(fill_size > 0);
    return new MemfillXferDes(dma_op, this, launch_node, guid,
			      inputs_info, outputs_info,
			      priority,
			      fill_data, fill_size, fill_total);
  }

  long MemfillChannel::submit(Request** requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemreduceChannel
  //

  MemreduceChannel::MemreduceChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<MemreduceChannel,MemreduceXferDes>(bgwork,
                                                          XFER_MEM_CPY,
                                                          "memreduce channel")
  {
    unsigned bw = 1000; // HACK - estimate at 1 GB/s
    unsigned latency = 100; // HACK - estimate at 100ns
    unsigned frag_overhead = 100; // HACK - estimate at 100ns

    // all local cpu memories are valid sources and dests
    std::vector<Memory> local_cpu_mems;
    MemcpyChannel::enumerate_local_cpu_memories(local_cpu_mems);

    add_path(local_cpu_mems, local_cpu_mems,
             bw, latency, frag_overhead, XFER_MEM_CPY)
      .set_max_dim(3)
      .allow_redops();

    xdq.add_to_manager(bgwork);
  }

  uint64_t MemreduceChannel::supports_path(Memory src_mem, Memory dst_mem,
                                           CustomSerdezID src_serdez_id,
                                           CustomSerdezID dst_serdez_id,
                                           ReductionOpID redop_id,
                                           size_t total_bytes,
                                           const std::vector<size_t> *src_frags,
                                           const std::vector<size_t> *dst_frags,
                                           XferDesKind *kind_ret /*= 0*/,
                                           unsigned *bw_ret /*= 0*/,
                                           unsigned *lat_ret /*= 0*/)
  {
    // if it's not a reduction, we don't want it
    if(redop_id == 0)
      return 0;

    // otherwise consult the normal supports_path logic
    return Channel::supports_path(src_mem, dst_mem,
                                  src_serdez_id, dst_serdez_id, redop_id,
                                  total_bytes, src_frags, dst_frags,
                                  kind_ret, bw_ret, lat_ret);
  }

  XferDes *MemreduceChannel::create_xfer_des(uintptr_t dma_op,
                                             NodeID launch_node,
                                             XferDesID guid,
                                             const std::vector<XferDesPortInfo>& inputs_info,
                                             const std::vector<XferDesPortInfo>& outputs_info,
                                             int priority,
                                             XferDesRedopInfo redop_info,
                                             const void *fill_data,
                                             size_t fill_size,
                                             size_t fill_total)
  {
    assert(redop_info.id != 0); // redop is required
    assert(fill_size == 0);
    return new MemreduceXferDes(dma_op, this, launch_node, guid,
                                inputs_info, outputs_info,
                                priority, redop_info);
  }

  long MemreduceChannel::submit(Request** requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetChannel
  //

      // TODO: deprecate this channel/memory entirely
      GASNetChannel::GASNetChannel(BackgroundWorkManager *bgwork,
				   XferDesKind _kind)
	: SingleXDQChannel<GASNetChannel, GASNetXferDes>(bgwork,
							 _kind,
							 stringbuilder() << "gasnet channel (kind= " << _kind << ")")
      {
	unsigned bw = 1000;  // HACK - estimate at 1 GB/s
	unsigned latency = 5000; // HACK - estimate at 5 us
        unsigned frag_overhead = 1000; // HACK - estimate at 1 us

        // all local cpu memories are valid sources/dests
        std::vector<Memory> local_cpu_mems;
        MemcpyChannel::enumerate_local_cpu_memories(local_cpu_mems);

        if(_kind == XFER_GASNET_READ)
          add_path(Memory::GLOBAL_MEM, true,
                   local_cpu_mems,
                   bw, latency, frag_overhead, XFER_GASNET_READ);
        else
          add_path(local_cpu_mems,
                   Memory::GLOBAL_MEM, true,
                   bw, latency, frag_overhead, XFER_GASNET_WRITE);
      }

      GASNetChannel::~GASNetChannel()
      {
      }

      XferDes *GASNetChannel::create_xfer_des(uintptr_t dma_op,
					      NodeID launch_node,
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
	return new GASNetXferDes(dma_op, this, launch_node, guid,
				 inputs_info, outputs_info,
				 priority);
      }

      long GASNetChannel::submit(Request** requests, long nr)
      {
        for (long i = 0; i < nr; i++) {
          GASNetRequest* req = (GASNetRequest*) requests[i];
	  // no serdez support
	  assert(req->xd->input_ports[req->src_port_idx].serdez_op == 0);
	  assert(req->xd->output_ports[req->dst_port_idx].serdez_op == 0);
          switch (kind) {
            case XFER_GASNET_READ:
            {
	      req->xd->input_ports[req->src_port_idx].mem->get_bytes(req->gas_off,
								     req->mem_base,
								     req->nbytes);
              break;
            }
            case XFER_GASNET_WRITE:
            {
	      req->xd->output_ports[req->dst_port_idx].mem->put_bytes(req->gas_off,
								      req->mem_base,
								      req->nbytes);
              break;
            }
            default:
              assert(0);
          }
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
        return nr;
      }

      RemoteWriteChannel::RemoteWriteChannel(BackgroundWorkManager *bgwork)
	: SingleXDQChannel<RemoteWriteChannel, RemoteWriteXferDes>(bgwork,
								   XFER_REMOTE_WRITE,
								   "remote write channel")
      {
	unsigned bw = 5000;  // HACK - estimate at 5 GB/s
	unsigned latency = 2000;  // HACK - estimate at 2 us
        unsigned frag_overhead = 1000; // HACK - estimate at 1 us
	// any combination of SYSTEM/REGDMA/Z_COPY/SOCKET_MEM
	// for(size_t i = 0; i < num_cpu_mem_kinds; i++)
	//   add_path(cpu_mem_kinds[i], false,
	// 	   Memory::REGDMA_MEM, true,
	// 	   bw, latency, false, false, XFER_REMOTE_WRITE);
	add_path(false /*!local_loopback*/,
		 bw, latency, frag_overhead,
		 XFER_REMOTE_WRITE);
        // TODO: permit 2d sources?
      }

      RemoteWriteChannel::~RemoteWriteChannel() {}

      XferDes *RemoteWriteChannel::create_xfer_des(uintptr_t dma_op,
						   NodeID launch_node,
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
	return new RemoteWriteXferDes(dma_op, this, launch_node, guid,
				      inputs_info, outputs_info,
				      priority);
      }

      long RemoteWriteChannel::submit(Request** requests, long nr)
      {
        for (long i = 0; i < nr; i ++) {
          RemoteWriteRequest* req = (RemoteWriteRequest*) requests[i];
	  XferDes::XferPort *in_port = &req->xd->input_ports[req->src_port_idx];
	  XferDes::XferPort *out_port = &req->xd->output_ports[req->dst_port_idx];
	  // no serdez support
	  assert((in_port->serdez_op == 0) && (out_port->serdez_op == 0));
	  NodeID dst_node = ID(out_port->mem->me).memory_owner_node();
	  size_t write_bytes_total = (size_t)-1;
	  if(out_port->needs_pbt_update.load() &&
	     req->xd->iteration_completed.load_acquire()) {
	    // this can result in sending the pbt twice, but this code path
	    //  is "mostly dead" and should be nuked soon
	    out_port->needs_pbt_update.store(false);
	    write_bytes_total = out_port->local_bytes_total;
	  }
	  RemoteAddress dst_buf;
	  bool ok = out_port->mem->get_remote_addr(req->dst_off, dst_buf);
	  assert(ok);
	  // send a request if there's data or if there's a next XD to update
	  if((req->nbytes > 0) ||
	     (out_port->peer_guid != XferDes::XFERDES_NO_GUID)) {
	    if (req->dim == Request::DIM_1D) {
	      XferDesRemoteWriteMessage::send_request(
                dst_node, dst_buf, req->src_base, req->nbytes, req,
		out_port->peer_guid, out_port->peer_port_idx,
		req->write_seq_pos, req->write_seq_count, 
		write_bytes_total);
	    } else {
	      assert(req->dim == Request::DIM_2D);
	      // dest MUST be continuous
	      assert(req->nlines <= 1 || ((size_t)req->dst_str) == req->nbytes);
	      XferDesRemoteWriteMessage::send_request(
                dst_node, dst_buf, req->src_base, req->nbytes,
                req->src_str, req->nlines, req,
		out_port->peer_guid, out_port->peer_port_idx,
		req->write_seq_pos, req->write_seq_count, 
		write_bytes_total);
	    }
	  }
	  // for an empty transfer, we do the local completion ourselves
	  //   instead of waiting for an ack from the other node
	  if(req->nbytes == 0) {
	    req->xd->notify_request_read_done(req);
	    req->xd->notify_request_write_done(req);
	  }
        /*RemoteWriteRequest* req = (RemoteWriteRequest*) requests[i];
          req->complete_event = GenEventImpl::create_genevent()->current_event();
          Realm::RemoteWriteMessage::RequestArgs args;
          args.mem = req->dst_mem;
          args.offset = req->dst_offset;
          args.event = req->complete_event;
          args.sender = Network::my_node_id;
          args.sequence_id = 0;

          Realm::RemoteWriteMessage::Message::request(ID(args.mem).node(), args,
                                                      req->src_buf, req->nbytes,
                                                      PAYLOAD_KEEPREG,
                                                      req->dst_buf);*/
        }
        return nr;
      }

      /*static*/
      void XferDesRemoteWriteMessage::handle_message(NodeID sender,
						     const XferDesRemoteWriteMessage &args,
						     const void *data,
						     size_t datalen)
      {
        // assert data copy is in right position
        //assert(data == args.dst_buf);

	log_xd.info() << "remote write recieved: next="
		      << std::hex << args.next_xd_guid << std::dec
		      << " start=" << args.span_start
		      << " size=" << args.span_size
		      << " pbt=" << args.pre_bytes_total;

	// if requested, notify (probably-local) next XD
	if(args.next_xd_guid != XferDes::XFERDES_NO_GUID) {
	  XferDesQueue *xdq = XferDesQueue::get_singleton();
	  if(args.pre_bytes_total != size_t(-1))
	    xdq->update_pre_bytes_total(args.next_xd_guid,
					args.next_port_idx,
					args.pre_bytes_total);
	  xdq->update_pre_bytes_write(args.next_xd_guid,
				      args.next_port_idx,
				      args.span_start,
				      args.span_size);
	}

	// don't ack empty requests
	if(datalen > 0)
	  XferDesRemoteWriteAckMessage::send_request(sender, args.req);
      }

      /*static*/
      void XferDesRemoteWriteAckMessage::handle_message(NodeID sender,
							const XferDesRemoteWriteAckMessage &args,
							const void *data,
							size_t datalen)
      {
        RemoteWriteRequest* req = args.req;
        req->xd->notify_request_read_done(req);
        req->xd->notify_request_write_done(req);
      }

      /*static*/ void XferDesDestroyMessage::handle_message(NodeID sender,
							    const XferDesDestroyMessage &args,
							    const void *msgdata,
							    size_t msglen)
      {
	XferDesQueue::get_singleton()->destroy_xferDes(args.guid);
      }

      /*static*/ void UpdateBytesTotalMessage::handle_message(NodeID sender,
							      const UpdateBytesTotalMessage &args,
							      const void *msgdata,
							      size_t msglen)
      {
        XferDesQueue::get_singleton()->update_pre_bytes_total(args.guid,
							      args.port_idx,
							      args.pre_bytes_total);
      }

      /*static*/ void UpdateBytesWriteMessage::handle_message(NodeID sender,
							      const UpdateBytesWriteMessage &args,
							      const void *msgdata,
							      size_t msglen)
      {
        XferDesQueue::get_singleton()->update_pre_bytes_write(args.guid,
							      args.port_idx,
							      args.span_start,
							      args.span_size);
      }

      /*static*/ void UpdateBytesReadMessage::handle_message(NodeID sender,
							    const UpdateBytesReadMessage &args,
							    const void *msgdata,
							    size_t msglen)
      {
        XferDesQueue::get_singleton()->update_next_bytes_read(args.guid,
							      args.port_idx,
							      args.span_start,
							      args.span_size);
      }

      ////////////////////////////////////////////////////////////////////////
      //
      // class XferDesPlaceholder
      //

      XferDesPlaceholder::XferDesPlaceholder()
        : refcount(1)
        , xd(0)
      {
        for(int i = 0; i < INLINE_PORTS; i++)
          inline_bytes_total[i] = ~size_t(0);
      }

      XferDesPlaceholder::~XferDesPlaceholder()
      {}

      void XferDesPlaceholder::add_reference()
      {
        refcount.fetch_add_acqrel(1);
      }

      void XferDesPlaceholder::remove_reference()
      {
        unsigned prev = refcount.fetch_sub_acqrel(1);
        // if this is the last reference to a placeholder that was assigned an
        //  xd (the unassigned case should only happen on an insertion race),
        //  propagate our progress info to the xd
        if((prev == 1) && xd) {
          bool updated = false;
          for(int i = 0; i < INLINE_PORTS; i++) {
            if(inline_bytes_total[i] != ~size_t(0)) {
              xd->update_pre_bytes_total(i, inline_bytes_total[i]);
              updated = true;
            }
            if(!inline_pre_write[i].empty()) {
              inline_pre_write[i].import(xd->input_ports[i].seq_remote);
              updated = true;
            }
          }
          for(std::map<int, size_t>::const_iterator it = extra_bytes_total.begin();
              it != extra_bytes_total.end();
              ++it) {
            xd->update_pre_bytes_total(it->first, it->second);
            updated = true;
          }
          for(std::map<int, SequenceAssembler>::const_iterator it = extra_pre_write.begin();
              it != extra_pre_write.end();
              ++it) {
            it->second.import(xd->input_ports[it->first].seq_remote);
            updated = true;
          }
          if(updated)
            xd->update_progress();
          xd->remove_reference();
        }

        if(prev == 1)
          delete this;
      }

      void XferDesPlaceholder::update_pre_bytes_write(int port_idx,
                                                      size_t span_start,
                                                      size_t span_size)
      {
        if(port_idx < INLINE_PORTS) {
          inline_pre_write[port_idx].add_span(span_start, span_size);
        } else {
          // need a mutex around getting the reference to the SequenceAssembler
          SequenceAssembler *sa;
          {
            AutoLock<> al(extra_mutex);
            sa = &extra_pre_write[port_idx];
          }
          sa->add_span(span_start, span_size);
        }
      }

      void XferDesPlaceholder::update_pre_bytes_total(int port_idx,
                                                      size_t pre_bytes_total)
      {
        if(port_idx < INLINE_PORTS) {
          inline_bytes_total[port_idx] = pre_bytes_total;
        } else {
          AutoLock<> al(extra_mutex);
          extra_bytes_total[port_idx] = pre_bytes_total;
        }
      }

      void XferDesPlaceholder::set_real_xd(XferDes *_xd)
      {
        // remember the xd and add a reference to it - actual updates will
        //  happen once we're destroyed
        xd = _xd;
        xd->add_reference();
      }


      ////////////////////////////////////////////////////////////////////////
      //
      // class XferDesQueue
      //

      /*static*/ XferDesQueue* XferDesQueue::get_singleton()
      {
	// we use a single queue for all xferDes
	static XferDesQueue xferDes_queue;
        return &xferDes_queue;
      }

      void XferDesQueue::update_pre_bytes_write(XferDesID xd_guid, int port_idx,
						size_t span_start, size_t span_size)
      {
        NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == Network::my_node_id) {
          XferDes *xd = 0;
          XferDesPlaceholder *ph = 0;
          {
            AutoLock<> al(guid_lock);
            std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
            if(it != guid_to_xd.end()) {
              if((it->second & 1) == 0) {
                // is a real xd - add a reference before we release the lock
                xd = reinterpret_cast<XferDes *>(it->second);
                xd->add_reference();
              } else {
                // is a placeholder - add a reference before we release lock
                ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
                ph->add_reference();
              }
            }
          }
          // if we got neither, create a new placeholder and try to add it,
          //  coping with the case where we lose to another insertion
          if(!xd && !ph) {
            XferDesPlaceholder *new_ph = new XferDesPlaceholder;
            {
              AutoLock<> al(guid_lock);
              std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
              if(it != guid_to_xd.end()) {
                if((it->second & 1) == 0) {
                  // is a real xd - add a reference before we release the lock
                  xd = reinterpret_cast<XferDes *>(it->second);
                  xd->add_reference();
                } else {
                  // is a placeholder - add a reference before we release lock
                  ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
                  ph->add_reference();
                }
              } else {
                guid_to_xd.insert(std::make_pair(xd_guid,
                                                 reinterpret_cast<uintptr_t>(new_ph) + 1));
                ph = new_ph;
                new_ph->add_reference();  // table keeps the original reference
              }
            }
            // if we didn't install our placeholder, remove the reference so it
            //  goes away
            if(ph != new_ph)
              new_ph->remove_reference();
          }
          // now we can update the xd or the placeholder and then release the
          //  reference we kept
          if(xd) {
            xd->update_pre_bytes_write(port_idx, span_start, span_size);
            xd->remove_reference();
          } else {
            ph->update_pre_bytes_write(port_idx, span_start, span_size);
            ph->remove_reference();
          }
        }
        else {
          // send a active message to remote node
          // this can happen if we have a non-network path (e.g. ipc) to another rank
          UpdateBytesWriteMessage::send_request(execution_node, xd_guid,
						port_idx,
						span_start, span_size);
        }
      }

      void XferDesQueue::update_pre_bytes_total(XferDesID xd_guid, int port_idx,
						size_t pre_bytes_total)
      {
        NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == Network::my_node_id) {
          XferDes *xd = 0;
          XferDesPlaceholder *ph = 0;
          {
            AutoLock<> al(guid_lock);
            std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
            if(it != guid_to_xd.end()) {
              if((it->second & 1) == 0) {
                // is a real xd - add a reference before we release the lock
                xd = reinterpret_cast<XferDes *>(it->second);
                xd->add_reference();
              } else {
                // is a placeholder - add a reference before we release lock
                ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
                ph->add_reference();
              }
            }
          }
          // if we got neither, create a new placeholder and try to add it,
          //  coping with the case where we lose to another insertion
          if(!xd && !ph) {
            XferDesPlaceholder *new_ph = new XferDesPlaceholder;
            {
              AutoLock<> al(guid_lock);
              std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
              if(it != guid_to_xd.end()) {
                if((it->second & 1) == 0) {
                  // is a real xd - add a reference before we release the lock
                  xd = reinterpret_cast<XferDes *>(it->second);
                  xd->add_reference();
                } else {
                  // is a placeholder - add a reference before we release lock
                  ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
                  ph->add_reference();
                }
              } else {
                guid_to_xd.insert(std::make_pair(xd_guid,
                                                 reinterpret_cast<uintptr_t>(new_ph) + 1));
                ph = new_ph;
                new_ph->add_reference();  // table keeps the original reference
              }
            }
            // if we didn't install our placeholder, remove the reference so it
            //  goes away
            if(ph != new_ph)
              new_ph->remove_reference();
          }
          // now we can update the xd or the placeholder and then release the
          //  reference we kept
          if(xd) {
            xd->update_pre_bytes_total(port_idx, pre_bytes_total);
            xd->remove_reference();
          } else {
            ph->update_pre_bytes_total(port_idx, pre_bytes_total);
            ph->remove_reference();
          }
        }
        else {
          // send an active message to remote node
	  ActiveMessage<UpdateBytesTotalMessage> amsg(execution_node);
	  amsg->guid = xd_guid;
	  amsg->port_idx = port_idx;
	  amsg->pre_bytes_total = pre_bytes_total;
	  amsg.commit();
        }
      }

      void XferDesQueue::update_next_bytes_read(XferDesID xd_guid, int port_idx,
						size_t span_start, size_t span_size)
      {
        NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == Network::my_node_id) {
          XferDes *xd = 0;
          {
            AutoLock<> al(guid_lock);
            std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
            if(it != guid_to_xd.end()) {
              if((it->second & 1) == 0) {
                // is a real xd - add a reference before we release the lock
                xd = reinterpret_cast<XferDes *>(it->second);
                xd->add_reference();
              } else {
                // should never be a placeholder!
                assert(0);
              }
            } else {
              // ok if we don't find it - upstream xd's can be destroyed before
              //  the downstream xd has stopped updating it
            }
          }
          if(xd) {
            xd->update_next_bytes_read(port_idx, span_start, span_size);
            xd->remove_reference();
          }
        }
        else {
          // send a active message to remote node
          UpdateBytesReadMessage::send_request(execution_node, xd_guid,
					       port_idx,
					       span_start, span_size);
        }
      }

      void XferDesQueue::destroy_xferDes(XferDesID guid)
      {
	XferDes *xd = 0;
        {
          AutoLock<> al(guid_lock);
          std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(guid);
          if(it != guid_to_xd.end()) {
            if((it->second & 1) == 0) {
              // remember xd but remove from table (stealing table's reference)
              xd = reinterpret_cast<XferDes *>(it->second);
              guid_to_xd.erase(it);
            } else {
              // should never be a placeholder!
              assert(0);
            }
          } else {
            // should always be present!
            assert(0);
          }
        }
        // just remove table's reference (actual destruction may be delayed
        //   if some other thread is still poking it)
	xd->remove_reference();
      }

      bool XferDesQueue::enqueue_xferDes_local(XferDes* xd,
					       bool add_to_queue /*= true*/)
      {
	Event wait_on = xd->request_metadata();
	if(!wait_on.has_triggered()) {
	  log_new_dma.info() << "xd metadata wait: xd=" << xd->guid << " ready=" << wait_on;
	  xd->deferred_enqueue.defer(this, xd, wait_on);
	  return false;
	}

        // insert ourselves in the table, replacing a placeholder if present
        XferDesPlaceholder *ph = 0;
        {
          AutoLock<> al(guid_lock);
          std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd->guid);
          if(it != guid_to_xd.end()) {
            if((it->second & 1) == 0) {
              // should never be a real xd!
              assert(0);
              guid_to_xd.erase(it);
            } else {
              // remember placeholder (stealing table's reference)
              ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
              // put xd in, donating the initial reference to the table
              it->second = reinterpret_cast<uintptr_t>(xd);
            }
          } else {
            guid_to_xd.insert(std::make_pair(xd->guid,
                                             reinterpret_cast<uintptr_t>(xd)));
          }
        }
        if(ph) {
          // tell placeholder about real xd and have it update it once there
          //  are no other concurrent updates
          ph->set_real_xd(xd);
          ph->remove_reference();
        }

	if(!add_to_queue) return true;
	assert(0);

	return true;
      }

      void XferDes::DeferredXDEnqueue::defer(XferDesQueue *_xferDes_queue,
					     XferDes *_xd, Event wait_on)
      {
	xferDes_queue = _xferDes_queue;
	xd = _xd;
	Realm::EventImpl::add_waiter(wait_on, this);
      }

      void XferDes::DeferredXDEnqueue::event_triggered(bool poisoned,
						       TimeLimit work_until)
      {
	// TODO: handle poisoning
	assert(!poisoned);
	log_new_dma.info() << "xd metadata ready: xd=" << xd->guid;
	xd->channel->enqueue_ready_xd(xd);
	//xferDes_queue->enqueue_xferDes_local(xd);
      }

      void XferDes::DeferredXDEnqueue::print(std::ostream& os) const
      {
	os << "deferred xd enqueue: xd=" << xd->guid;
      }

      Event XferDes::DeferredXDEnqueue::get_finish_event(void) const
      {
	// TODO: would be nice to provide dma op's finish event here
	return Event::NO_EVENT;
      }

    void destroy_xfer_des(XferDesID _guid)
    {
      log_new_dma.info("Destroy XferDes: id(" IDFMT ")", _guid);
      NodeID execution_node = _guid >> (XferDesQueue::NODE_BITS + XferDesQueue::INDEX_BITS);
      if (execution_node == Network::my_node_id) {
	XferDesQueue::get_singleton()->destroy_xferDes(_guid);
      }
      else {
        XferDesDestroyMessage::send_request(execution_node, _guid);
      }
    }

ActiveMessageHandlerReg<SimpleXferDesCreateMessage> simple_xfer_des_create_message_handler;
ActiveMessageHandlerReg<NotifyXferDesCompleteMessage> notify_xfer_des_complete_handler;
ActiveMessageHandlerReg<XferDesRemoteWriteMessage> xfer_des_remote_write_handler;
ActiveMessageHandlerReg<XferDesRemoteWriteAckMessage> xfer_des_remote_write_ack_handler;
ActiveMessageHandlerReg<XferDesDestroyMessage> xfer_des_destroy_message_handler;
ActiveMessageHandlerReg<UpdateBytesTotalMessage> update_bytes_total_message_handler;
ActiveMessageHandlerReg<UpdateBytesWriteMessage> update_bytes_write_message_handler;
ActiveMessageHandlerReg<UpdateBytesReadMessage> update_bytes_read_message_handler;
ActiveMessageHandlerReg<RemoteWriteXferDes::Write1DMessage> remote_write_1d_message_handler;

}; // namespace Realm


