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

// The range allocator implementation copied and pasted from Realm.

#ifndef __RANGE_ALLOCATOR_H__
#define __RANGE_ALLOCATOR_H__

namespace Legion {
  namespace Internal {

    // manages a basic free list of ranges (using range type RT) and allocated
    //  ranges, which are tagged (tag type TT)
    // NOT thread-safe - must be protected from outside
    template <typename RT, typename TT>
    class BasicRangeAllocator {
    public:
      struct Range {
        //Range(RT _first, RT _last);

        RT first, last;  // half-open range: [first, last)
        unsigned prev, next;  // double-linked list of all ranges (by index)
        unsigned prev_free, next_free;  // double-linked list of just free ranges
      };

      std::map<TT, unsigned> allocated;  // direct lookup of allocated ranges by tag
#ifdef DEBUG_LEGION
      std::map<RT, unsigned> by_first;   // direct lookup of all ranges by first
      // TODO: sized-based lookup of free ranges
#endif

      static const unsigned SENTINEL = 0;
      // TODO: small (medium?) vector opt
      std::vector<Range> ranges;

      BasicRangeAllocator(void);
      ~BasicRangeAllocator(void);

      void swap(BasicRangeAllocator<RT, TT>& swap_with);

      void add_range(RT first, RT last);
      bool can_allocate(TT tag, RT size, RT alignment);
      bool allocate(TT tag, RT size, RT alignment, RT& first);
      void deallocate(TT tag, bool missing_ok = false);
      bool lookup(TT tag, RT& first, RT& size);

    protected:
      unsigned first_free_range;
      unsigned alloc_range(RT first, RT last);
      void free_range(unsigned index);
    };

  }; // namespace Internal
}; // namespace Legion

#include "legion/range_allocator.inl"

#endif // __RANGE_ALLOCATOR_H__

// EOF
