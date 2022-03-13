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

// Memory implementations for Realm

#ifndef REALM_MEMORY_IMPL_INL
#define REALM_MEMORY_IMPL_INL

// nop, but helpful for IDEs
#include "realm/mem_impl.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryImpl
  //

  template <typename T>
  T *MemoryImpl::find_module_specific()
  {
    ModuleSpecificInfo *info = module_specific;
    while(info) {
      T *downcast = dynamic_cast<T *>(info);
      if(downcast)
        return downcast;
      info = info->next;
    }
    return 0;
  }

  template <typename T>
  const T *MemoryImpl::find_module_specific() const
  {
    const ModuleSpecificInfo *info = module_specific;
    while(info) {
      const T *downcast = dynamic_cast<const T *>(info);
      if(downcast)
        return downcast;
      info = info->next;
    }
    return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class BasicRangeAllocator<RT,TT>
  //

#if 0
  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT,TT>::Range::Range(RT _first, RT _last)
    : first(_first), last(_last)
    , prev(-1), next(-1)
    , prev_free(-1), next_free(-1)
  {}
#endif

  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT,TT>::BasicRangeAllocator(void)
    : first_free_range(SENTINEL)
  {
    ranges.resize(1);
    Range& s = ranges[SENTINEL];
    s.first = RT(-1);
    s.last = 0;
    s.prev = s.next = s.prev_free = s.next_free = SENTINEL;
  }

  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT,TT>::~BasicRangeAllocator(void)
  {}

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT,TT>::swap(BasicRangeAllocator<RT, TT>& swap_with)
  {
    allocated.swap(swap_with.allocated);
#ifdef DEBUG_REALM
    by_first.swap(swap_with.by_first);
#endif
    ranges.swap(swap_with.ranges);
    std::swap(first_free_range, swap_with.first_free_range);
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT,TT>::add_range(RT first, RT last)
  {
    // ignore empty ranges
    if(first == last)
      return;

    int new_idx = alloc_range(first, last);

    Range& newr = ranges[new_idx];
    Range& sentinel = ranges[SENTINEL];

    // simple case - starting range
    if(sentinel.next == SENTINEL) {
      // all block list
      newr.prev = newr.next = SENTINEL;
      sentinel.prev = sentinel.next = new_idx;
      // free block list
      newr.prev_free = newr.next_free = SENTINEL;
      sentinel.prev_free = sentinel.next_free = new_idx;

#ifdef DEBUG_REALM
      by_first[first] = new_idx;
#endif
      return;
    }

    assert(0);
  }

  template <typename RT, typename TT>
  inline unsigned BasicRangeAllocator<RT,TT>::alloc_range(RT first, RT last)
  {
    // find/make a free index in the range list for this range
    int new_idx;
    if(first_free_range != SENTINEL) {
      new_idx = first_free_range;
      first_free_range = ranges[new_idx].next;
    } else {
      new_idx = ranges.size();
      ranges.resize(new_idx + 1);
    }
    ranges[new_idx].first = first;
    ranges[new_idx].last = last;
    return new_idx;
  }
   
  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT,TT>::free_range(unsigned index)
  {
    ranges[index].next = first_free_range;
    first_free_range = index;
  }
  
  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT,TT>::can_allocate(TT tag,
						       RT size, RT alignment)
  {
    // empty allocation requests are trivial
    if(size == 0) {
      return true;
    }

    // walk free ranges and just take the first that fits
    unsigned idx = ranges[SENTINEL].next_free;
    while(idx != SENTINEL) {
      Range *r = &ranges[idx];

      RT ofs = 0;
      if(alignment) {
	RT rem = r->first % alignment;
	if(rem > 0)
	  ofs = alignment - rem;
      }
      // do we have enough space?
      if((r->last - r->first) >= (size + ofs))
	return true;

      // no, go to next one
      idx = r->next_free;
    }

    // allocation failed
    return false;
  }

  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT,TT>::allocate(TT tag, RT size, RT alignment, RT& alloc_first)
  {
    // empty allocation requests are trivial
    if(size == 0) {
      allocated[tag] = SENTINEL;
      return true;
    }

    // walk free ranges and just take the first that fits
    unsigned idx = ranges[SENTINEL].next_free;
    while(idx != SENTINEL) {
      Range *r = &ranges[idx];

      RT ofs = 0;
      if(alignment) {
	RT rem = r->first % alignment;
	if(rem > 0)
	  ofs = alignment - rem;
      }
      // do we have enough space?
      if((r->last - r->first) >= (size + ofs)) {
	// yes, but we may need to chop things up to make the exact range we want
	alloc_first = r->first + ofs;
	RT alloc_last = alloc_first + size;

        // do we need to carve off a new (free) block before us?
        if(alloc_first != r->first) {
	  unsigned new_idx = alloc_range(r->first, alloc_first);
	  Range *new_prev = &ranges[new_idx];
	  r = &ranges[idx];  // alloc may have moved this!
	  
	  r->first = alloc_first;
	  // insert into all-block dllist
	  new_prev->prev = r->prev;
	  new_prev->next = idx;
	  ranges[r->prev].next = new_idx;
	  r->prev = new_idx;
	  // insert into free-block dllist
	  new_prev->prev_free = r->prev_free;
	  new_prev->next_free = idx;
	  ranges[r->prev_free].next_free = new_idx;
	  r->prev_free = new_idx;

#ifdef DEBUG_REALM
	  // fix up by_first entries
	  by_first[r->first] = new_idx;
	  by_first[alloc_first] = idx;
#endif
        }

	// two cases to deal with
	if(alloc_last == r->last) {
	  // case 1 - exact fit
	  //
	  // all we have to do here is remove this range from the free range dlist
	  //  and add to the allocated lookup map
	  ranges[r->prev_free].next_free = r->next_free;
	  ranges[r->next_free].prev_free = r->prev_free;
	} else {
	  // case 2 - leftover at end - put in new range
	  unsigned after_idx = alloc_range(alloc_last, r->last);
	  Range *r_after = &ranges[after_idx];
	  r = &ranges[idx];  // alloc may have moved this!

#ifdef DEBUG_REALM
	  by_first[alloc_last] = after_idx;
#endif
	  r->last = alloc_last;
	  
	  // r_after goes after r in all block list
	  r_after->prev = idx;
	  r_after->next = r->next;
	  r->next = after_idx;
	  ranges[r_after->next].prev = after_idx;

	  // r_after replaces r in the free block list
	  r_after->prev_free = r->prev_free;
	  r_after->next_free = r->next_free;
	  ranges[r_after->next_free].prev_free = after_idx;
	  ranges[r_after->prev_free].next_free = after_idx;
	}

	// tie this off because we use it to detect allocated-ness
	r->prev_free = r->next_free = idx;

	allocated[tag] = idx;
	return true;
      }

      // no, go to next one
      idx = r->next_free;
    }
    // allocation failed
    return false;
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT,TT>::deallocate(TT tag,
						     bool missing_ok /*= false*/)
  {
    typename std::map<TT, unsigned>::iterator it = allocated.find(tag);
    if(it == allocated.end()) {
      assert(missing_ok);
      return;
    }
    unsigned del_idx = it->second;
    allocated.erase(it);

    // if there was no Range associated with this tag, it was an zero-size
    //  allocation, and there's nothing to add to the free list
    if(del_idx == SENTINEL)
      return;

    Range& r = ranges[del_idx];

    unsigned pf_idx = r.prev;
    while((pf_idx != SENTINEL) && (ranges[pf_idx].prev_free == pf_idx)) {
      pf_idx = ranges[pf_idx].prev;
      assert(pf_idx != del_idx);  // wrapping around would be bad
    }
    unsigned nf_idx = r.next;
    while((nf_idx != SENTINEL) && (ranges[nf_idx].next_free == nf_idx)) {
      nf_idx = ranges[nf_idx].next;
      assert(nf_idx != del_idx);
    }

    // do we need to merge?
    bool merge_prev = (pf_idx == r.prev) && (pf_idx != SENTINEL);
    bool merge_next = (nf_idx == r.next) && (nf_idx != SENTINEL);

    // four cases - ordered to match the allocation cases
    if(!merge_next) {
      if(!merge_prev) {
	// case 1 - no merging (exact match)
	// just add ourselves to the free list
	r.prev_free = pf_idx;
	r.next_free = nf_idx;
	ranges[pf_idx].next_free = del_idx;
	ranges[nf_idx].prev_free = del_idx;
      } else {
	// case 2 - merge before
	// merge ourselves into the range before
	Range& r_before = ranges[pf_idx];

	r_before.last = r.last;
	r_before.next = r.next;
	ranges[r.next].prev = pf_idx;
	// r_before was already in free list, so no changes to that

#ifdef DEBUG_REALM
	by_first.erase(r.first);
#endif
	free_range(del_idx);
      }
    } else {
      if(!merge_prev) {
	// case 3 - merge after
	// merge ourselves into the range after
	Range& r_after = ranges[nf_idx];

#ifdef DEBUG_REALM
	by_first[r.first] = nf_idx;
	by_first.erase(r_after.first);
#endif

	r_after.first = r.first;
	r_after.prev = r.prev;
	ranges[r.prev].next = nf_idx;
	// r_after was already in the free list, so no changes to that

	free_range(del_idx);
      } else {
	// case 4 - merge both
	// merge both ourselves and range after into range before
	Range& r_before = ranges[pf_idx];
	Range& r_after = ranges[nf_idx];

	r_before.last = r_after.last;
#ifdef DEBUG_REALM
	by_first.erase(r.first);
	by_first.erase(r_after.first);
#endif

	// adjust both normal list and free list
	r_before.next = r_after.next;
	ranges[r_after.next].prev = pf_idx;

	r_before.next_free = r_after.next_free;
	ranges[r_after.next_free].prev_free = pf_idx;

	free_range(del_idx);
	free_range(nf_idx);
      }
    }
  }
  
  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT,TT>::lookup(TT tag, RT& first, RT& size)
  {
    typename std::map<TT, unsigned>::iterator it = allocated.find(tag);

    if(it != allocated.end()) {
      // if there was no Range associated with this tag, it was an zero-size
      //  allocation
      if(it->second == SENTINEL) {
	first = 0;
	size = 0;
      } else {
	const Range& r = ranges[it->second];
	first = r.first;
	size = r.last - r.first;
      }

      return true;
    } else
      return false;
  }
  
    
}; // namespace Realm

#endif // ifndef REALM_MEM_IMPL_INL

