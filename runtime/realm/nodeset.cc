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

// dynamic node set implementation for Realm

#include "realm/nodeset.h"

#include <cstring>
#include <cassert>

#ifdef REALM_ON_WINDOWS
#include <intrin.h>
#endif

static unsigned popcount(uint64_t v)
{
#ifdef REALM_ON_WINDOWS
#ifdef _WIN64
  return __popcnt64(v);
#else
  unsigned v_lo = v;
  unsigned v_hi = v >> 32;
  return __popcnt(v_lo) + __popcnt(v_hi);
#endif
#else
  return __builtin_popcountll(v);
#endif
}

static unsigned ctz(uint64_t v)
{
#ifdef REALM_ON_WINDOWS
  unsigned long index;
#ifdef _WIN64
  if(_BitScanForward64(&index, v))
    return index;
#else
  unsigned v_lo = v;
  unsigned v_hi = v >> 32;
  if(_BitScanForward(&index, v_lo))
    return index;
  else if(_BitScanForward(&index, v_hi))
    return index + 32;
#endif
  else
    return 0;
#else
  return __builtin_ctzll(v);
#endif
}

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class NodeSet

  void NodeSet::remove(NodeID id)
  {
#ifdef DEBUG_REALM_NODESET
    reference_set.erase(id);
#endif

    if(count == 0) {
      // nothing to do
      return;
    }

    switch(enc_format) {
    case ENC_VALS:
      {
	for(short i = 0; i < short(count); i++)
	  if(data.values[i] == id) {
	    // compact values if needed
	    if(i < short(count - 1))
	      data.values[i] = data.values[count - 1];
	    count--;
	    return;
	  }
	break;
      }

    case ENC_RANGES:
      {
	for(short i = 0; i < range_count; i++)
	  if((data.ranges[i].lo <= id) && (id <= data.ranges[i].hi)) {
	    // match, but what we do depends on where this is in the range
	    if(data.ranges[i].lo == id) {
	      if(data.ranges[i].hi == id) {
		// singleton range match - delete range
		if(i < (range_count - 1))
		  data.ranges[i] = data.ranges[range_count - 1];
		range_count--;
	      } else {
		// match lo end - trim it
		data.ranges[i].lo = id + 1;
	      }
	    } else {
	      if(data.ranges[i].hi == id) {
		// match hi end - trim it
		data.ranges[i].hi = id - 1;
	      } else {
		// breaks our range in half
		if(range_count < MAX_RANGES) {
		  // create a new range
		  data.ranges[range_count].lo = id + 1;
		  data.ranges[range_count].hi = data.ranges[i].hi;
		  data.ranges[i].hi = id - 1;
		  range_count++;
		} else {
		  convert_to_bitmask();
		  data.bitmask->clear_bit(id);
		}
	      }
	    }
	    count--;
	    return;
	  }

	break;
      }

    case ENC_BITMASK:
      {
	count -= data.bitmask->clear_bit(id);
	if(count == 0)
	  NodeSetBitmask::release_bitmask(data.bitmask, true /*already_empty*/);
	break;
      }
    }
  }

  void NodeSet::remove_range(NodeID lo, NodeID hi)
  {
    if(lo > hi) return; // empty range

#ifdef DEBUG_REALM_NODESET
    for(NodeID id = lo; id <= hi; id++)
      reference_set.erase(id);
#endif

    if(count == 0) {
      // nothing to do
      return;
    }

    switch(enc_format) {
    case ENC_VALS:
      {
	short idx = 0;
	while(idx < short(count))
	  if((lo <= data.values[idx]) && (data.values[idx] <= hi)) {
	    if(idx < short(count - 1))
	      data.values[idx] = data.values[count - 1];
	    count--;
	  } else
	    idx++;
	break;
      }

    case ENC_RANGES:
      {
	short i = 0;
	while(i < range_count) {
	  if((data.ranges[i].hi < lo) || (data.ranges[i].lo > hi)) {
	    // no overlap - go to next one
	    i++;
	    continue;
	  }

	  if(data.ranges[i].lo < lo) {
	    // partial range remaining below
	    if(data.ranges[i].hi > hi) {
	      // partial above too - we need another range
	      if(range_count < MAX_RANGES) {
		data.ranges[range_count].lo = hi + 1;
		data.ranges[range_count].hi = data.ranges[i].hi;
		data.ranges[i].hi = lo - 1;
		range_count++;
		count -= (hi - lo + 1);
		// this was the only range that could overlap [lo,hi], so
		//  we're done
		return;
	      } else {
		// have to switch to a bitmask
		convert_to_bitmask();
		count -= data.bitmask->clear_range(lo, hi);
		assert(count > 0); // shouldn't clear everything
		return;
	      }
	    } else {
	      // no leftover above, just trim range and continue
	      count -= (data.ranges[i].hi - lo + 1);
	      data.ranges[i].hi = lo - 1;
	      i++;
	    }
	  } else {
	    if(data.ranges[i].hi > hi) {
	      // only leftover above, trim bottom of range
	      count -= (hi - data.ranges[i].lo + 1);
	      data.ranges[i].lo = hi + 1;
	      i++;
	    } else {
	      // completely contained - delete range
	      count -= (data.ranges[i].hi - data.ranges[i].lo + 1);
	      if(i < (range_count - 1))
		data.ranges[i] = data.ranges[range_count - 1];
	      range_count--;
	      // do NOT increase 'i' here...
	    }
	  }
	}
	break;
      }

    case ENC_BITMASK:
      {
	count -= data.bitmask->clear_range(lo, hi);
	if(count == 0)
	  NodeSetBitmask::release_bitmask(data.bitmask, true /*already_empty*/);
	break;
      }
    }
  }

  void NodeSet::convert_to_bitmask()
  {
    assert(count > 0);
    NodeSetBitmask *newmask = NodeSetBitmask::acquire_bitmask();

    switch(enc_format) {
    case ENC_VALS:
      {
	for(short i = 0; i < short(count); i++)
	  newmask->set_bit(data.values[i]);
	break;
      }
    case ENC_RANGES:
      {
	for(short i = 0; i < range_count; i++)
	  newmask->set_range(data.ranges[i].lo, data.ranges[i].hi);
	break;
      }
    default:
      assert(0);
    }
    data.bitmask = newmask;
    enc_format = ENC_BITMASK;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class NodeSetBitmask

  NodeSetBitmask::NodeSetBitmask()
  {
#ifdef DEBUG_REALM_NODESET
    for(size_t i = 0; i < (bitset_elements + bitset_twolevel); i++)
      assert(bits[i] == 0);
#endif
  }

  /*static*/ NodeSetBitmask *NodeSetBitmask::acquire_bitmask()
  {
    uintptr_t base;
    {
      AutoLock<> al(free_list_mutex);
      base = free_list_head;
      if(base != 0) {
	bitmask_elem_t *link = reinterpret_cast<bitmask_elem_t *>(base);
	free_list_head = *link;
	*link = 0;
      }
    }
    if(!base) {
      // need to allocate a new chunk
      void *raw_base = calloc(1 + (bitset_elements +
				   bitset_twolevel) * bitsets_per_chunk,
			      sizeof(bitmask_elem_t));
      //printf("CALLOC = %p\n", raw_base);
      assert(raw_base != 0);
      bitmask_elem_t *chunk_base = reinterpret_cast<bitmask_elem_t *>(raw_base);
      // first bitset is one we'll use
      base = reinterpret_cast<uintptr_t>(&chunk_base[1]);
      // rest needs to be hooked into free list
      {
	AutoLock<> al(free_list_mutex);
	// alloc chain link
	chunk_base[0] = alloc_chain_head;
	alloc_chain_head = reinterpret_cast<uintptr_t>(chunk_base);
	// free chain link(s)
	size_t stride = bitset_elements + bitset_twolevel;
	if(bitsets_per_chunk > 1) {
	  chunk_base[1 + stride] = free_list_head;
	  for(size_t i = 2; i < bitsets_per_chunk; i++)
	    chunk_base[1 + i * stride] = reinterpret_cast<uintptr_t>(&chunk_base[1 + (i - 1) * stride]);
	  free_list_head = reinterpret_cast<uintptr_t>(&chunk_base[1 + (bitsets_per_chunk - 1) * stride]);
	}
      }
    }
    //printf("ACQUIRE = %lx\n", base);
    return new(reinterpret_cast<void *>(base)) NodeSetBitmask;
  }

  /*static*/ NodeSetBitmask *NodeSetBitmask::clone_bitmask(const NodeSetBitmask *clone_from)
  {
    NodeSetBitmask *newmask = acquire_bitmask();
    newmask->copy(clone_from);
    return newmask;
  }

  /*static*/ void NodeSetBitmask::release_bitmask(NodeSetBitmask *bitmask,
						  bool already_empty)
  {
    //printf("RELEASE = %p\n", bitmask);
    if(already_empty) {
#ifdef DEBUG_REALM_NODESET
      for(size_t i = 0; i < (bitset_elements + bitset_twolevel); i++)
	assert(bitmask->bits[i] == 0);
#endif
    } else {
      // clear things out so the next reuse starts fresh
      memset(&bitmask->bits[1], 0,
	     (bitset_elements + bitset_twolevel - 1) * sizeof(bitmask_elem_t));
    }
    {
      AutoLock<> al(free_list_mutex);
      bitmask->bits[0] = free_list_head;
      free_list_head = reinterpret_cast<uintptr_t>(&bitmask->bits[0]);
    }
  }

  size_t NodeSetBitmask::set_bit(NodeID id)
  {
#ifdef DEBUG_REALM
    assert((id >= 0) && (id <= max_node_id));
#endif
    size_t elmt_idx = id / BITS_PER_ELEM;
    size_t elmt_ofs = id % BITS_PER_ELEM;
    bitmask_elem_t mask = bitmask_elem_t(1) << elmt_ofs;
    if((bits[elmt_idx] & mask) == 0) {
      if(bitset_twolevel && (bits[elmt_idx] == 0))
	l2_set(elmt_idx);
      bits[elmt_idx] += mask;
      return 1;
    } else
      return 0;
  }

  size_t NodeSetBitmask::clear_bit(NodeID id)
  {
#ifdef DEBUG_REALM
    assert((id >= 0) && (id <= max_node_id));
#endif
    size_t elmt_idx = id / BITS_PER_ELEM;
    size_t elmt_ofs = id % BITS_PER_ELEM;
    bitmask_elem_t mask = bitmask_elem_t(1) << elmt_ofs;
    if((bits[elmt_idx] & mask) != 0) {
      bits[elmt_idx] -= mask;
      if(bitset_twolevel && (bits[elmt_idx] == 0))
	l2_clear(elmt_idx);
      return 1;
    } else
      return 0;
  }

  size_t NodeSetBitmask::set_range(NodeID lo, NodeID hi)
  {
#ifdef DEBUG_REALM
    assert((lo >= 0) && (hi <= max_node_id));
#endif
    if(lo > hi) return 0; // empty range
    size_t lo_idx = lo / BITS_PER_ELEM;
    size_t lo_ofs = lo % BITS_PER_ELEM;
    size_t hi_idx = hi / BITS_PER_ELEM;
    size_t hi_ofs = hi % BITS_PER_ELEM;
    size_t count = 0;
    if(lo_idx == hi_idx) {
      bitmask_elem_t mask = (~bitmask_elem_t(0)) << lo_ofs;
      if(hi_ofs < (BITS_PER_ELEM - 1))
	mask &= (bitmask_elem_t(2) << hi_ofs) - 1;
      mask &= ~bits[lo_idx];
      if(mask != 0) {
	count += popcount(mask);
	if(bitset_twolevel && (bits[lo_idx] == 0))
	  l2_set(lo_idx);
	bits[lo_idx] += mask;
      }
    } else {
      // lo_idx
      {
	bitmask_elem_t mask = (~bitmask_elem_t(0)) << lo_ofs;
	mask &= ~bits[lo_idx];
	if(mask != 0) {
	  count += popcount(mask);
	  if(bitset_twolevel && (bits[lo_idx] == 0))
	    l2_set(lo_idx);
	  bits[lo_idx] += mask;
	}
      }
      // middle entries
      for(size_t i = lo_idx + 1; i < hi_idx; i++) {
	bitmask_elem_t mask = ~bits[i];
	if(mask != 0) {
	  count += popcount(mask);
	  if(bitset_twolevel && (bits[i] == 0))
	    l2_set(i);
	  bits[i] += mask;
	}
      }
      // hi_idx
      {
	bitmask_elem_t mask = (bitmask_elem_t(2) << hi_ofs) - 1;
	mask &= ~bits[hi_idx];
	if(mask != 0) {
	  count += popcount(mask);
	  if(bitset_twolevel && (bits[hi_idx] == 0))
	    l2_set(hi_idx);
	  bits[hi_idx] += mask;
	}
      }
    }
    return count;
  }

  size_t NodeSetBitmask::clear_range(NodeID lo, NodeID hi)
  {
#ifdef DEBUG_REALM
    assert((lo >= 0) && (hi <= max_node_id));
#endif
    if(lo > hi) return 0; // empty range
    size_t lo_idx = lo / BITS_PER_ELEM;
    size_t lo_ofs = lo % BITS_PER_ELEM;
    size_t hi_idx = hi / BITS_PER_ELEM;
    size_t hi_ofs = hi % BITS_PER_ELEM;
    size_t count = 0;
    if(lo_idx == hi_idx) {
      bitmask_elem_t mask = (~bitmask_elem_t(0)) << lo_ofs;
      if(hi_ofs < (BITS_PER_ELEM - 1))
	mask &= (bitmask_elem_t(2) << hi_ofs) - 1;
      mask &= bits[lo_idx];
      if(mask != 0) {
	count += popcount(mask);
	bits[lo_idx] -= mask;
	if(bitset_twolevel && (bits[lo_idx] == 0))
	  l2_clear(lo_idx);
      }
    } else {
      // lo_idx
      {
	bitmask_elem_t mask = (~bitmask_elem_t(0)) << lo_ofs;
	mask &= bits[lo_idx];
	if(mask != 0) {
	  count += popcount(mask);
	  bits[lo_idx] -= mask;
	  if(bitset_twolevel && (bits[lo_idx] == 0))
	    l2_clear(lo_idx);
	}
      }
      // middle entries
      for(size_t i = lo_idx + 1; i < hi_idx; i++) {
	bitmask_elem_t mask = bits[i];
	if(mask != 0) {
	  count += popcount(mask);
	  bits[i] -= mask;
	  if(bitset_twolevel && (bits[i] == 0))
	    l2_clear(i);
	}
      }
      // hi_idx
      {
	bitmask_elem_t mask = (bitmask_elem_t(2) << hi_ofs) - 1;
	mask &= bits[hi_idx];
	if(mask != 0) {
	  count += popcount(mask);
	  bits[hi_idx] -= mask;
	  if(bitset_twolevel && (bits[hi_idx] == 0))
	    l2_clear(hi_idx);
	}
      }
    }
    return count;
  }

  bool NodeSetBitmask::is_set(NodeID id) const
  {
#ifdef DEBUG_REALM
    assert((id >= 0) && (id <= max_node_id));
#endif
    size_t elmt_idx = id / BITS_PER_ELEM;
    size_t elmt_ofs = id % BITS_PER_ELEM;
    bitmask_elem_t mask = bitmask_elem_t(1) << elmt_ofs;
    return ((bits[elmt_idx] & mask) != 0);
  }

  NodeID NodeSetBitmask::first_set() const
  {
    size_t elmt_idx = 0;
    if(bitset_twolevel) {
      int found = l2_find(0);
#ifdef DEBUG_REALM
      assert(found >= 0);
#endif
      elmt_idx = found;
    } else {
      while(bits[elmt_idx] == 0) {
	elmt_idx++;
#ifdef DEBUG_REALM
	assert(elmt_idx < bitset_elements);
#endif
      }
    }
    size_t elmt_ofs = ctz(bits[elmt_idx]);
    return (elmt_idx * BITS_PER_ELEM + elmt_ofs);
  }

  NodeID NodeSetBitmask::next_set(NodeID after) const
  {
#ifdef DEBUG_REALM
    assert((after >= 0) && (after <= max_node_id));
#endif
    if(after == max_node_id)
      return -1;

    size_t elmt_idx = (after + 1) / BITS_PER_ELEM;
    size_t elmt_ofs = (after + 1) % BITS_PER_ELEM;
    bitmask_elem_t v = bits[elmt_idx] >> elmt_ofs;
    if(v != 0) {
      return (after + 1 + ctz(v));
    } else {
      if(bitset_twolevel) {
	int found = l2_find(elmt_idx + 1);
	if(found == -1)
	  return -1;
	elmt_idx = found;
#ifdef DEBUG_REALM
	assert(bits[elmt_idx] != 0);
#endif
	elmt_ofs = ctz(bits[elmt_idx]);
	return (elmt_idx * BITS_PER_ELEM + elmt_ofs);
      } else {
	while(++elmt_idx < bitset_elements)
	  if(bits[elmt_idx] != 0) {
	    elmt_ofs = ctz(bits[elmt_idx]);
	    return (elmt_idx * BITS_PER_ELEM + elmt_ofs);
	  }
	// no more bits...
	return -1;
      }
    }
  }

  void NodeSetBitmask::l2_set(int elmt_idx)
  {
    size_t l2_idx = bitset_elements + (elmt_idx / BITS_PER_ELEM);
    size_t l2_ofs = (elmt_idx % BITS_PER_ELEM);
    bitmask_elem_t mask = bitmask_elem_t(1) << l2_ofs;
    bits[l2_idx] |= mask;
  }

  void NodeSetBitmask::l2_clear(int elmt_idx)
  {
    size_t l2_idx = bitset_elements + (elmt_idx / BITS_PER_ELEM);
    size_t l2_ofs = (elmt_idx % BITS_PER_ELEM);
    bitmask_elem_t mask = bitmask_elem_t(1) << l2_ofs;
    bits[l2_idx] &= ~mask;
  }

  int NodeSetBitmask::l2_find(int first_idx) const
  {
    if(first_idx >= int(bitset_elements))
      return -1;
    size_t l2_idx = bitset_elements + (first_idx / BITS_PER_ELEM);
    size_t l2_ofs = (first_idx % BITS_PER_ELEM);
    bitmask_elem_t v = bits[l2_idx] >> l2_ofs;
    if(v != 0)
      return first_idx + ctz(v);
    size_t l2_max_idx = bitset_elements + bitset_twolevel - 1;
    while(++l2_idx <= l2_max_idx)
      if(bits[l2_idx] != 0) {
	l2_ofs = ctz(bits[l2_idx]);
	return (((l2_idx - bitset_elements) * BITS_PER_ELEM) + l2_ofs);
      }
    return -1;
  }

  void NodeSetBitmask::copy(const NodeSetBitmask *copy_from)
  {
    if(this != copy_from)
      memcpy(&bits[0], &copy_from->bits[0],
	     (bitset_elements + bitset_twolevel) * sizeof(bitmask_elem_t));
  }

  /*static*/ NodeID NodeSetBitmask::max_node_id = -1;
  /*static*/ size_t NodeSetBitmask::bitset_elements = 0;
  /*static*/ size_t NodeSetBitmask::bitsets_per_chunk = 0;
  /*static*/ size_t NodeSetBitmask::bitset_twolevel = 0;
  /*static*/ uintptr_t NodeSetBitmask::alloc_chain_head = 0;
  /*static*/ uintptr_t NodeSetBitmask::free_list_head = 0;
  /*static*/ Mutex NodeSetBitmask::free_list_mutex;

  /*static*/ void NodeSetBitmask::configure_allocator(NodeID _max_node_id,
						      size_t _bitsets_per_chunk,
						      bool _use_twolevel)
  {
    // can't reconfigure with a different node count
    if(max_node_id != -1) {
      assert(max_node_id == _max_node_id);
      return;
    }

    max_node_id = _max_node_id;
    bitset_elements = 1 + (max_node_id / BITS_PER_ELEM);

    if(_use_twolevel)
      bitset_twolevel = (bitset_elements + BITS_PER_ELEM - 1) / BITS_PER_ELEM;
    else
      bitset_twolevel = 0;
    bitsets_per_chunk = _bitsets_per_chunk;
  }

  /*static*/ void NodeSetBitmask::free_allocations()
  {
    while(alloc_chain_head != 0) {
      bitmask_elem_t *ptr = reinterpret_cast<bitmask_elem_t *>(alloc_chain_head);
      alloc_chain_head = ptr[0];
      free(ptr);
    }
  }


}; // namespace Realm
