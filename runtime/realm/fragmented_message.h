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

#ifndef FRAGMENTED_MESSAGE_HPP
#define FRAGMENTED_MESSAGE_HPP

#include "realm/realm_config.h"

#include <vector>
#include <cstdint>
#include <cstring>

namespace Realm {

  /**
   * \file fragmented_message.h
   * \brief Helper utility for reconstructing large ActiveMessages that were
   *        split into multiple network packets.
   *
   * Realm (and the underlying network transports such as GASNet-EX or UCX) may
   * have an upper bound on the size of a single *medium* or *long* active
   * message.  When an application attempts to send a payload larger than that
   * limit, the `FragmentedActiveMessage` helper in activemsg.h automatically
   * chops the data into `N` smaller packets.  On the receiver side those
   * packets must be collected and concatenated *before* the logical AM handler
   * can be invoked.
   *
   * The `FragmentedMessage` class performs exactly that role:
   *   – The sender embeds a small `FragmentInfo` structure in every fragment
   *     that conveys a 0-based `chunk_id`, the total number of chunks, and a
   *     unique `msg_id`.
   *   – The receiver creates one `FragmentedMessage` instance (looked up by the
   *     `<sender, msg_id>` pair) and calls `add_chunk()` each time a fragment
   *     arrives.
   *   – When all `total_chunks` pieces have been delivered, `is_complete()`
   *     returns *true* and the caller can safely call `reassemble()` to obtain
   *     a contiguous `std::vector<char>` containing the payload in order.
   *
   * A few design notes:
   *   • Duplicate fragments are ignored, allowing simple resend logic on the
   *     transmit path.
   *   • No per-fragment dynamic allocations are performed – each chunk's
   *     storage is reserved exactly once when its size becomes known.
   */

  class FragmentedMessage {
  public:
    explicit FragmentedMessage(uint32_t total_chunks = 0);

    bool add_chunk(uint32_t chunk_id, const void *data, size_t size);

    bool is_complete() const;
    size_t size() const;
    std::vector<char> reassemble() const;

  private:
    uint32_t total_chunks{0};
    size_t total_size{0};
    std::vector<std::vector<char>> received_chunks;
    std::vector<bool> received_flags;
    uint32_t received_count{0};
  };
} // namespace Realm

#endif // FRAGMENTED_MESSAGE_HPP
