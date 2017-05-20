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

// Memory implementations for Realm

#ifndef REALM_MEMORY_IMPL_INL
#define REALM_MEMORY_IMPL_INL

// nop, but helpful for IDEs
#include "mem_impl.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class BasicRangeAllocator<RT,TT>
  //

  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT,TT>::Range::Range(RT _first, RT _last)
    : first(_first), last(_last)
    , prev(0), next(0)
    , prev_free(0), next_free(0)
  {}

  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT,TT>::BasicRangeAllocator(void)
    : sentinel((RT)-1,0)
  {
    // sentinel is the start and end of both dllists
    sentinel.prev = sentinel.next = &sentinel;
    sentinel.prev_free = sentinel.next_free = &sentinel;
  }

  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT,TT>::~BasicRangeAllocator(void)
  {
    // delete range objects (allocated or free)
    Range *r = sentinel.next;
    while(r != &sentinel) {
      Range *rnext = r->next;
      delete r;
      r = rnext;
    }
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT,TT>::add_range(RT first, RT last)
  {
    Range *newr = new Range(first, last);

    // simple case - starting range
    if(sentinel.next == &sentinel) {
      // insert after sentinel
      Range *prev = &sentinel;
      Range *prev_free = &sentinel;
      // all block list
      newr->prev = prev; newr->next = prev->next;
      prev->next = newr->next->prev = newr;
      // free block list
      newr->prev_free = prev_free; newr->next_free = prev_free->next_free;
      prev_free->next_free = newr->next_free->prev_free = newr;
      return;
    }

    assert(0);
  }
   
  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT,TT>::allocate(TT tag, RT size, RT alignment, RT& alloc_first)
  {
    // walk free ranges and just take the first that fits
    Range *r = sentinel.next_free;
    while(r != &sentinel) {
      RT ofs = 0;
      if(alignment) {
	RT rem = r->first % alignment;
	if(rem > 0)
	  ofs = alignment - rem;
      }
      // do we have enough space?
      if((r->last - r->first) >= (size + ofs)) {
	// yes, but we may need chop things up to make the exact range we want
	alloc_first = r->first + ofs;
	RT alloc_last = alloc_first + size;

	// four cases to deal with
	if(alloc_last == r->last) {
	  if(alloc_first == r->first) {
	    // case 1 - exact fit
	    //
	    // all we have to do here is remove this range from the free range dlist
	    //  and add to the allocated lookup map
	    r->prev_free->next_free = r->next_free;
	    r->next_free->prev_free = r->prev_free;
	    r->prev_free = r->next_free = 0;

	    allocated[tag] = r;
	    return true;
	  } else {
	    // case 2 - leftover at beginning
	    assert(0);
	  }
	} else {
	  if(alloc_first == r->first) {
	    // case 3 - leftover at end
	    Range *r_after = new Range(alloc_last, r->last);
	    by_first[alloc_last] = r_after;

	    // r_after goes after r in all block list
	    r_after->prev = r; r_after->next = r->next;
	    r->next->prev = r_after; r->next = r_after;

	    // r_after replaces r in the free block list
	    r_after->prev_free = r->prev_free;
	    r_after->next_free = r->next_free;
	    r->prev_free->next_free = r_after;
	    r->next_free->prev_free = r_after;
	    r->prev_free = r->next_free = 0;

	    allocated[tag] = r;
	    return true;
	  } else {
	    // case 4 - leftover on both sides
	    assert(0);
	  }
	}
      }

      // no, go to next one
      r = r->next_free;
    }
    // allocation failed
    return false;
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT,TT>::deallocate(TT tag)
  {
    typename std::map<TT, Range *>::iterator it = allocated.find(tag);
    assert(it != allocated.end());
    Range *r = it->second;
    allocated.erase(it);

    // need to add ourselves back to the free list - find previous and next
    //  free entries (which have non-null prev_free/next_free pointers)
    Range *prev_free = r->prev;
    while(!prev_free->next_free) {
      prev_free = prev_free->prev;
      assert(prev_free != r);  // wrapping around would be bad
    }
    Range *next_free = r->next;
    while(!next_free->prev_free) {
      next_free = next_free->next;
      assert(next_free != r);
    }

    // do we need to merge?
    bool merge_prev = (prev_free == r->prev) && (prev_free != &sentinel);
    bool merge_next = (next_free == r->next) && (next_free != &sentinel);

    // four cases - ordered to match the allocation cases
    if(!merge_next) {
      if(!merge_prev) {
	// case 1 - no merging (exact match)
	r->prev_free = prev_free; r->next_free = next_free;
	prev_free->next_free = next_free->prev_free = r;
      } else {
	// case 2 - merge before
	assert(0);
      }
    } else {
      if(!merge_prev) {
	// case 3 - merge after
	assert(0);
      } else {
	// case 4 - merge both
	assert(0);
      }
    }
  };
  
    
}; // namespace Realm

#endif // ifndef REALM_MEM_IMPL_INL

