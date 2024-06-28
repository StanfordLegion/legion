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
#include "realm/runtime_impl.h"

namespace Realm {

  bool find_best_channel_for_memories(
      const Node *nodes_info, ChannelCopyInfo channel_copy_info,
      CustomSerdezID src_serdez_id, CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
      size_t total_bytes, const std::vector<size_t> *src_frags,
      const std::vector<size_t> *dst_frags, uint64_t &best_cost, Channel *&best_channel,
      XferDesKind &best_kind)
  {
    // consider dma channels available on either source or dest node
    NodeID src_node = ID(channel_copy_info.src_mem).memory_owner_node();
    NodeID dst_node = ID(channel_copy_info.dst_mem).memory_owner_node();

    best_cost = 0;
    best_channel = 0;
    best_kind = XFER_NONE;

    {
      const Node &n = nodes_info[src_node];
      for(std::vector<Channel *>::const_iterator it = n.dma_channels.begin();
          it != n.dma_channels.end(); ++it) {
        XferDesKind kind = XFER_NONE;
        uint64_t cost =
            (*it)->supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                 redop_id, total_bytes, src_frags, dst_frags, &kind);
        if((cost > 0) && ((best_cost == 0) || (cost < best_cost))) {
          best_cost = cost;
          best_channel = *it;
          best_kind = kind;
        }
      }
    }

    if(dst_node != src_node) {
      const Node &n = nodes_info[dst_node];
      for(std::vector<Channel *>::const_iterator it = n.dma_channels.begin();
          it != n.dma_channels.end(); ++it) {
        XferDesKind kind = XFER_NONE;
        uint64_t cost =
            (*it)->supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                 redop_id, total_bytes, src_frags, dst_frags, &kind);
        if((cost > 0) && ((best_cost == 0) || (cost < best_cost))) {
          best_cost = cost;
          best_channel = *it;
          best_kind = kind;
        }
      }
    }

    return (best_cost != 0);
  }

} // namespace Realm
