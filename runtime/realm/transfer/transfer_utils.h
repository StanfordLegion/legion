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
#include "realm/transfer/channel.h"

namespace Realm {
  // Finds the largest subrectangle of 'domain' that starts with 'start',
  //  lies entirely within 'restriction', and is consistent with an iteration
  //  order (over the original 'domain') of 'dim_order'
  // the subrectangle is returned in 'subrect', the start of the next subrect
  //  is in 'next_start', and the return value indicates whether the 'domain'
  //  has been fully covered
  template <int N, typename T>
  bool next_subrect(const Rect<N, T> &domain, const Point<N, T> &start,
                    const Rect<N, T> &restriction, const int *dim_order,
                    Rect<N, T> &subrect, Point<N, T> &next_start);

  // Returns true if successfully found a DMA channel that has a minimum
  // transfer cost from source to destination memories.
  bool find_best_channel_for_memories(
      const Node *nodes_info, ChannelCopyInfo channel_copy_info,
      CustomSerdezID src_serdez_id, CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
      size_t total_bytes, const std::vector<size_t> *src_frags,
      const std::vector<size_t> *dst_frags, uint64_t &best_cost, Channel *&best_channel,
      XferDesKind &best_kind);
} // namespace Realm

#include "realm/transfer/transfer_utils.inl"

#endif
