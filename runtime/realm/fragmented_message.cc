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

#include "realm/realm_config.h"
#include "realm/fragmented_message.h"

#include <stdexcept>
#include <assert.h>

namespace Realm {

  FragmentedMessage::FragmentedMessage(uint32_t total_chunks)
    : total_chunks(total_chunks)
    , received_chunks(total_chunks)
    , received_flags(total_chunks, false)
  {}

  bool FragmentedMessage::add_chunk(uint32_t chunk_id, const void *data, size_t size)
  {
    if(chunk_id >= total_chunks || received_flags[chunk_id]) {
      return false;
    }

    received_chunks[chunk_id].resize(size);
    std::memcpy(received_chunks[chunk_id].data(), data, size);
    received_flags[chunk_id] = true;
    received_count++;
    return true;
  }

  bool FragmentedMessage::is_complete() const
  {
    assert(received_count <= total_chunks);
    return received_count == total_chunks;
  }

  size_t FragmentedMessage::size() const
  {
    size_t total_size = 0;
    for(size_t i = 0; i < total_chunks; ++i) {
      if(received_flags[i]) { // Only sum valid chunks
        total_size += received_chunks[i].size();
      }
    }
    return total_size;
  }

  std::vector<char> FragmentedMessage::reassemble() const
  {
    if(!is_complete()) {
      return std::vector<char>();
    }

    std::vector<char> message;
    message.reserve(size());
    for(const std::vector<char> &chunk : received_chunks) {
      message.insert(message.end(), chunk.begin(), chunk.end());
    }

    return message;
  }
} // namespace Realm
