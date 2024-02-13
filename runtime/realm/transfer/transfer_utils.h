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

#ifndef REALM_TRANSFER_UTILS_H
#define REALM_TRANSFER_UTILS_H

#include "realm/point.h"

namespace Realm {
  // finds the largest subrectangle of 'domain' that starts with 'start',
  //  lies entirely within 'restriction', and is consistent with an iteration
  //  order (over the original 'domain') of 'dim_order'
  // the subrectangle is returned in 'subrect', the start of the next subrect
  //  is in 'next_start', and the return value indicates whether the 'domain'
  //  has been fully covered
  template <int N, typename T>
  bool next_subrect(const Rect<N, T> &domain, const Point<N, T> &start,
                    const Rect<N, T> &restriction, const int *dim_order,
                    Rect<N, T> &subrect, Point<N, T> &next_start);

  template <int N, typename T>
  bool next_target_subrect(const Rect<N, T> &layout_bounds, Rect<N, T> &cur_rect,
                           Point<N, T> &cur_point, Rect<N, T> &target_subrect,
                           const int dim_order[N]);

} // namespace Realm

#include "realm/transfer/transfer_utils.inl"

#endif
