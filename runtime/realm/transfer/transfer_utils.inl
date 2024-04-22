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

#include "realm/transfer/transfer_utils.h"

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

  template <int N, typename T>
  bool compute_target_subrect(const Rect<N, T> &layout_bounds, Rect<N, T> &cur_rect,
                              Point<N, T> &cur_point, Rect<N, T> &target_subrect,
                              const int dim_order[N])
  {
    target_subrect.lo = cur_point;
    target_subrect.hi = cur_point;

    bool have_rect = false; // tentatively clear - we'll (re-)set it below if needed
    for(int di = 0; di < N; di++) {
      int d = dim_order[di];

      // our target subrect in this dimension can be trimmed at the front by
      //  having already done a partial step, or trimmed at the end by the
      //  layout
      if(cur_rect.hi[d] <= layout_bounds.hi[d]) {
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
            while((d2 < N) && ((d2 <= di) || (cur_point[dim_order[d2]] ==
                                              cur_rect.hi[dim_order[d2]]))) {
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
        // stopping short (doesn't matter where we started) - limit this
        // subrect
        //  based on the piece and start just past it in this dimension
        //  (rewinding previous dimensions)
        target_subrect.hi[d] = layout_bounds.hi[d];
        have_rect = true;
        for(int d2 = 0; d2 < di; d2++)
          cur_point[dim_order[d2]] = cur_rect.lo[dim_order[d2]];
        cur_point[d] = layout_bounds.hi[d] + 1;
        break;
      }
    }

    return have_rect;
  }
} // namespace Realm
