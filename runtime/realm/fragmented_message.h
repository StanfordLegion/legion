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

  class FragmentedMessage {
  public:
    explicit FragmentedMessage(uint32_t total_chunks = 0);

    void add_chunk(uint32_t chunk_id, const void *data, size_t size);

    bool is_complete() const;

    size_t size() const;

    std::vector<char> reassemble() const;

  private:
    uint32_t total_chunks{0};
    std::vector<std::vector<char>> received_chunks;
    std::vector<bool> received_flags;
    uint32_t received_count{0};
  };
} // namespace Realm

#endif // FRAGMENTED_MESSAGE_HPP
