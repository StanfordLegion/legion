/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_BUFFERS_H__
#define __LEGION_BUFFERS_H__

#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Buffer Manager
    /////////////////////////////////////////////////////////////

    /**
     * A class that is helpful in keeping track of owned buffer
     */
    template<typename T, AllocationLifetime L>
    class BufferManager : public NoHeapify {
    public:
      inline BufferManager(void) : size(0) { }
      inline BufferManager(const void* buffer, size_t s)
      {
        save_buffer(buffer, s);
      }
      inline BufferManager(const BufferManager& rhs) : size(rhs.size)
      {
        if (size <= VALUE_SIZE)
          std::memcpy(data.value, rhs.data.value, size);
        else
        {
          // If it's not shared yet make it shared
          if (rhs.data.shared.references == nullptr)
          {
            rhs.data.shared.references = new std::atomic<uint32_t>(2);
            data.shared.references = rhs.data.shared.references;
          }
          else
          {
            data.shared.references = rhs.data.shared.references;
            data.shared.references->fetch_add(1);
          }
          data.shared.buffer = rhs.data.shared.buffer;
        }
      }
      inline BufferManager(BufferManager&& rhs) noexcept : size(rhs.size)
      {
        if (rhs.size <= VALUE_SIZE)
          std::memcpy(data.value, rhs.data.value, size);
        else
          data.shared = rhs.data.shared;
        rhs.size = 0;
      }
      inline ~BufferManager(void) { clear(); }
    public:
      inline BufferManager& operator=(const BufferManager& rhs)
      {
        if (this == &rhs)
          return *this;
        clear();
        size = rhs.size;
        if (size <= VALUE_SIZE)
          std::memcpy(data.value, rhs.data.value, size);
        else
        {
          // If it's not shared yet make it shared
          if (rhs.data.shared.references == nullptr)
          {
            rhs.data.shared.references = new std::atomic<uint32_t>(2);
            data.shared.references = rhs.data.shared.references;
          }
          else
          {
            data.shared.references = rhs.data.shared.references;
            data.shared.references->fetch_add(1);
          }
          data.shared.buffer = rhs.data.shared.buffer;
        }
        return *this;
      }
      inline BufferManager& operator=(BufferManager&& rhs) noexcept
      {
        if (this == &rhs)
          return *this;
        clear();
        size = rhs.size;
        if (rhs.size <= VALUE_SIZE)
          std::memcpy(data.value, rhs.data.value, size);
        else
          data.shared = rhs.data.shared;
        rhs.size = 0;
        return *this;
      }
    public:
      inline void clear(void)
      {
        if (VALUE_SIZE < size)
        {
          // Check to see if we're sharing it
          if (data.shared.references == nullptr)
            legion_free<void, BufferManager<T, L> >(data.shared.buffer, size);
          else
          {
            const uint32_t previous = data.shared.references->fetch_sub(1);
            legion_assert(previous > 0);
            if (previous == 1)
            {
              legion_free<void, BufferManager<T, L> >(data.shared.buffer, size);
              delete data.shared.references;
            }
          }
        }
        size = 0;
      }
      inline void save_buffer(const void* buffer, size_t s)
      {
        clear();
        size = s;
        if (size <= VALUE_SIZE)
          std::memcpy(data.value, buffer, size);
        else
        {
          data.shared.buffer = legion_malloc<void, L, BufferManager<T, L> >(
              size, alignof(std::max_align_t));
          std::memcpy(data.shared.buffer, buffer, size);
          data.shared.references = nullptr;
        }
      }
      inline const void* get_buffer(void) const
      {
        if (size <= VALUE_SIZE)
          return data.value;
        else
          return data.shared.buffer;
      }
      inline size_t get_size(void) const { return size; }
      inline void serialize(Serializer& rez) const
      {
        rez.serialize(size);
        if (size <= VALUE_SIZE)
          rez.serialize(data.value, size);
        else
          rez.serialize(data.shared.buffer, size);
      }
      inline void deserialize(Deserializer& derez)
      {
        clear();
        derez.deserialize(size);
        if (size <= VALUE_SIZE)
          std::memcpy(data.value, derez.get_current_pointer(), size);
        else
        {
          data.shared.buffer = legion_malloc<void, L, BufferManager<T, L> >(
              size, alignof(std::max_align_t));
          std::memcpy(data.shared.buffer, derez.get_current_pointer(), size);
          data.shared.references = nullptr;
        }
        derez.advance_pointer(size);
      }
    private:
      size_t size;
      struct SharedBuffer {
        void* buffer = nullptr;
        mutable std::atomic<uint32_t>* references = nullptr;
      };
      // If the data is less than or equal to the VALUE_SIZE then
      // we pass it by value when the buffer is copied, otherwise
      // we can just share the pointer to the allocation. We pick
      // 64 bytes since it is a common cache line size.
      static constexpr size_t VALUE_SIZE = 64;
      union {
        SharedBuffer shared = SharedBuffer{};
        std::byte value[VALUE_SIZE];
      } data;
    };

    /////////////////////////////////////////////////////////////
    // Semantic Info
    /////////////////////////////////////////////////////////////

    /**
     * \struct SemanticInfo
     * A struct for storing semantic information for various things
     */
    struct SemanticInfo {
    public:
      SemanticInfo(void) : is_mutable(false) { }
      SemanticInfo(const void* buf, size_t s, bool is_mut = true)
        : buffer(buf, s), is_mutable(is_mut)
      { }
      SemanticInfo(RtUserEvent ready) : ready_event(ready), is_mutable(true) { }
    public:
      inline bool is_valid(void) const { return ready_event.has_triggered(); }
    public:
      BufferManager<SemanticInfo, LONG_LIFETIME> buffer;
      RtUserEvent ready_event;
      bool is_mutable;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_BUFFERS_H__
