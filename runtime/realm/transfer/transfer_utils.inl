/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#include "realm/transfer/transfer_utils.h"
#include "realm/transfer/channel.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/ib_memory.h"
#include "realm/inst_layout.h"

namespace Realm {

  template <int N, typename T>
  bool next_subrect(const Rect<N, T> &domain, const Point<N, T> &start,
                    const Rect<N, T> &restriction, const int *dim_order,
                    Rect<N, T> &subrect, Point<N, T> &next_start)
  {
    // special case for when we can do the whole domain in one subrect
    if((start == domain.lo) && restriction.contains(domain)) {
      subrect = domain;
      return true;
    }

#ifdef DEBUG_REALM
    // starting point better be inside the restriction
    assert(restriction.contains(start));
#endif
    subrect.lo = start;

    for(int di = 0; di < N; di++) {
      int d = dim_order[di];

      // can we go to the end of the domain in this dimension?
      if(domain.hi[d] <= restriction.hi[d]) {
        // we can go to the end of the domain in this dimension ...
        subrect.hi[d] = domain.hi[d];
        next_start[d] = domain.lo[d];

        if(start[d] == domain.lo[d]) {
          // ... and since we started at the start, we can continue to next dim
          continue;
        } else {
          // ... but we have to stop since this wasn't a full span
          if(++di < N) {
            d = dim_order[di];
            subrect.hi[d] = start[d];
            next_start[d] = start[d] + 1;

            while(++di < N) {
              d = dim_order[di];
              subrect.hi[d] = start[d];
              next_start[d] = start[d];
            }

            return false; // still more to do
          }
        }
      } else {
        // we didn't get to the end, so we'll have to pick up where we left off
        subrect.hi[d] = restriction.hi[d];
        next_start[d] = restriction.hi[d] + 1;

        while(++di < N) {
          d = dim_order[di];
          subrect.hi[d] = start[d];
          next_start[d] = start[d];
        }

        return false; // still more to do
      }
    }

    // if we got through all dimensions, we're done with this domain
    return true;
  }
} // namespace Realm
