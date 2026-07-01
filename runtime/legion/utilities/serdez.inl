/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// Included from serdez.h - do not include this directly

// Useful for IDEs
#include "legion/utilities/serdez.h"

namespace Legion {

  //--------------------------------------------------------------------------
  inline Serializer::Serializer(void)
    : total_bytes(STATIC_SIZE), buffer(nullptr), index(0)
  //--------------------------------------------------------------------------
  {
#ifdef LEGION_DEBUG
    context_bytes = 0;
#endif
  }

  //--------------------------------------------------------------------------
  inline Serializer::~Serializer(void)
  //--------------------------------------------------------------------------
  {
    if (buffer != nullptr)
      Internal::legion_free(buffer, total_bytes);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline void Serializer::serialize(const T& element)
  //--------------------------------------------------------------------------
  {
    static_assert(std::is_trivially_copyable<T>::value);
    while ((index + sizeof(T)) > total_bytes) resize();
    if (buffer != nullptr)
      std::memcpy(buffer + index, (const void*)&element, sizeof(T));
    else
      std::memcpy(static_buffer + index, (const void*)&element, sizeof(T));
    index += sizeof(T);
#ifdef LEGION_DEBUG
    context_bytes += sizeof(T);
#endif
  }

  //--------------------------------------------------------------------------
  template<>
  inline void Serializer::serialize<bool>(const bool& element)
  //--------------------------------------------------------------------------
  {
    const uint32_t flag = element ? 1 : 0;
    serialize<uint32_t>(flag);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline void Serializer::serialize(const std::optional<T>& opt)
  //--------------------------------------------------------------------------
  {
    if (opt)
    {
      serialize<bool>(true);
      serialize(*opt);
    }
    else
      serialize<bool>(false);
  }

  //--------------------------------------------------------------------------
  template<typename T, typename A>
  inline void Serializer::serialize(const std::vector<T, A>& vector)
  //--------------------------------------------------------------------------
  {
    serialize<size_t>(vector.size());
    if (!vector.empty())
    {
      if constexpr (!std::is_trivially_copyable_v<T>)
      {
        for (unsigned idx = 0; idx < vector.size(); idx++)
          serialize(vector[idx]);
      }
      else
        serialize(&vector.front(), vector.size() * sizeof(T));
    }
  }

  //--------------------------------------------------------------------------
  template<typename T, typename C, typename A>
  inline void Serializer::serialize(const std::set<T, C, A>& set)
  //--------------------------------------------------------------------------
  {
    serialize<size_t>(set.size());
    for (const T& element : set) serialize(element);
  }

  //--------------------------------------------------------------------------
  template<typename T1, typename T2, typename C, typename A>
  inline void Serializer::serialize(const std::map<T1, T2, C, A>& map)
  //--------------------------------------------------------------------------
  {
    serialize<size_t>(map.size());
    for (const std::pair<const T1, T2>& entry : map)
    {
      serialize(entry.first);
      serialize(entry.second);
    }
  }

  //--------------------------------------------------------------------------
  template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
  inline void Serializer::serialize(
      const Internal::BitMask<T, MAX, SHIFT, MASK>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }

  //--------------------------------------------------------------------------
  template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
  inline void Serializer::serialize(
      const Internal::TLBitMask<T, MAX, SHIFT, MASK>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }

#ifdef __SSE2__
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::SSEBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::SSETLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }
#endif

#ifdef __AVX__
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::AVXBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::AVXTLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }
#endif

#ifdef __ALTIVEC__
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::PPCBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::PPCTLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }
#endif

#ifdef __ARM_NEON
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::NeonBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Serializer::serialize(const Internal::NeonTLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.serialize(*this);
  }
#endif

  //--------------------------------------------------------------------------
  template<
      typename DT, Internal::AllocationLifetime L, unsigned BLOAT, bool BIDIR>
  inline void Serializer::serialize(
      const Internal::CompoundBitMask<DT, L, BLOAT, BIDIR>& m)
  //--------------------------------------------------------------------------
  {
    m.serialize(*this);
  }

  //--------------------------------------------------------------------------
  inline void Serializer::serialize(const Domain& dom)
  //--------------------------------------------------------------------------
  {
    serialize(dom.is_id);
    if (dom.is_id > 0)
      serialize(dom.is_type);
    serialize(dom.dim);
    for (int i = 0; i < 2 * dom.dim; i++) serialize(dom.rect_data[i]);
  }

  //--------------------------------------------------------------------------
  inline void Serializer::serialize(const DomainPoint& dp)
  //--------------------------------------------------------------------------
  {
    serialize(dp.dim);
    if (dp.dim == 0)
      serialize(dp.point_data[0]);
    else
    {
      for (int idx = 0; idx < dp.dim; idx++) serialize(dp.point_data[idx]);
    }
  }

  //--------------------------------------------------------------------------
  inline void Serializer::serialize(const Internal::CopySrcDstField& field)
  //--------------------------------------------------------------------------
  {
    serialize(field.inst);
    serialize(field.field_id);
    serialize(field.redop_id);
    if (field.redop_id > 0)
    {
      serialize<bool>(field.red_fold);
      serialize<bool>(field.red_exclusive);
    }
    serialize(field.serdez_id);
    serialize(field.subfield_offset);
    serialize(field.indirect_index);
    serialize(field.size);
    // we know if there's a fill value if the field ID is -1
    if (field.field_id == (Realm::FieldID)-1)
    {
      if (field.size <= Internal::CopySrcDstField::MAX_DIRECT_SIZE)
        serialize(field.fill_data.direct, field.size);
      else
        serialize(field.fill_data.indirect, field.size);
    }
  }

  //--------------------------------------------------------------------------
  inline void Serializer::serialize(const void* src, size_t bytes)
  //--------------------------------------------------------------------------
  {
    while ((index + bytes) > total_bytes) resize();
    if (buffer != nullptr)
      std::memcpy(buffer + index, src, bytes);
    else
      std::memcpy(static_buffer + index, src, bytes);
    index += bytes;
#ifdef LEGION_DEBUG
    context_bytes += bytes;
#endif
  }

  //--------------------------------------------------------------------------
  inline void Serializer::begin_context(void)
  //--------------------------------------------------------------------------
  {
#ifdef LEGION_DEBUG
    while ((index + sizeof(context_bytes)) > total_bytes) resize();
    if (buffer != nullptr)
      std::memcpy(buffer + index, &context_bytes, sizeof(context_bytes));
    else
      std::memcpy(static_buffer + index, &context_bytes, sizeof(context_bytes));
    index += sizeof(context_bytes);
    context_bytes = 0;
#endif
  }

  //--------------------------------------------------------------------------
  inline void Serializer::end_context(void)
  //--------------------------------------------------------------------------
  {
#ifdef LEGION_DEBUG
    // Save the size into the buffer
    while ((index + sizeof(context_bytes)) > total_bytes) resize();
    if (buffer != nullptr)
      std::memcpy(buffer + index, &context_bytes, sizeof(context_bytes));
    else
      std::memcpy(static_buffer + index, &context_bytes, sizeof(context_bytes));
    index += sizeof(context_bytes);
    context_bytes = 0;
#endif
  }

  //--------------------------------------------------------------------------
  inline void* Serializer::reserve_bytes(size_t bytes)
  //--------------------------------------------------------------------------
  {
    while ((index + bytes) > total_bytes) resize();
    void* result = ((buffer != nullptr) ? buffer : static_buffer) + index;
    index += bytes;
#ifdef LEGION_DEBUG
    context_bytes += bytes;
#endif
    return result;
  }

  //--------------------------------------------------------------------------
  inline void Serializer::reset(void)
  //--------------------------------------------------------------------------
  {
    index = 0;
#ifdef LEGION_DEBUG
    context_bytes = 0;
#endif
  }

  //--------------------------------------------------------------------------
  inline void Serializer::resize(void)
  //--------------------------------------------------------------------------
  {
    // Double the buffer size
    if (total_bytes == STATIC_SIZE)
    {
      uint8_t* next =
          Internal::legion_malloc<uint8_t, Internal::TASK_LOCAL_LIFETIME>(
              2 * total_bytes, alignof(size_t));
      legion_assert(next != nullptr);
      std::memcpy(next, static_buffer, total_bytes);
      buffer = next;
    }
    else
    {
      legion_assert(total_bytes > STATIC_SIZE);
      uint8_t* next =
          Internal::legion_realloc<uint8_t, Internal::TASK_LOCAL_LIFETIME>(
              buffer, total_bytes, 2 * total_bytes);
      legion_assert(next != nullptr);
      buffer = next;
    }
    total_bytes *= 2;
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline void Deserializer::deserialize(T& element)
  //--------------------------------------------------------------------------
  {
    static_assert(std::is_trivially_copyable<T>::value);
    // Check to make sure we don't read past the end
    legion_assert((index + sizeof(T)) <= total_bytes);
    std::memcpy(&element, buffer + index, sizeof(T));
    index += sizeof(T);
#ifdef LEGION_DEBUG
    context_bytes += sizeof(T);
#endif
  }

  //--------------------------------------------------------------------------
  template<>
  inline void Deserializer::deserialize<bool>(bool& element)
  //--------------------------------------------------------------------------
  {
    uint32_t flag;
    deserialize<uint32_t>(flag);
    element = (flag != 0);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline void Deserializer::deserialize(std::optional<T>& opt)
  //--------------------------------------------------------------------------
  {
    bool valid;
    deserialize<bool>(valid);
    if (valid)
    {
      T value;
      deserialize(value);
      opt = value;
    }
  }

  //--------------------------------------------------------------------------
  template<typename T, typename A>
  inline void Deserializer::deserialize(std::vector<T, A>& vector)
  //--------------------------------------------------------------------------
  {
    size_t size;
    deserialize<size_t>(size);
    vector.resize(size);
    if (size > 0)
    {
      if constexpr (!std::is_trivially_copyable_v<T>)
      {
        for (unsigned idx = 0; idx < size; idx++) deserialize(vector[idx]);
      }
      else
        deserialize(&vector.front(), size * sizeof(T));
    }
  }

  //--------------------------------------------------------------------------
  template<typename T, typename C, typename A>
  inline void Deserializer::deserialize(std::set<T, C, A>& set)
  //--------------------------------------------------------------------------
  {
    size_t size;
    deserialize<size_t>(size);
    for (unsigned idx = 0; idx < size; idx++)
    {
      T elem;
      deserialize(elem);
      set.insert(elem);
    }
  }

  //--------------------------------------------------------------------------
  template<typename T1, typename T2, typename C, typename A>
  inline void Deserializer::deserialize(std::map<T1, T2, C, A>& map)
  //--------------------------------------------------------------------------
  {
    size_t size;
    deserialize<size_t>(size);
    for (unsigned idx = 0; idx < size; idx++)
    {
      T1 key;
      deserialize(key);
      deserialize(map[key]);
    }
  }

  //--------------------------------------------------------------------------
  template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
  inline void Deserializer::deserialize(
      Internal::BitMask<T, MAX, SHIFT, MASK>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }

  //--------------------------------------------------------------------------
  template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
  inline void Deserializer::deserialize(
      Internal::TLBitMask<T, MAX, SHIFT, MASK>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }

#ifdef __SSE2__
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::SSEBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::SSETLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }
#endif

#ifdef __AVX__
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::AVXBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::AVXTLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }
#endif

#ifdef __ALTIVEC__
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::PPCBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::PPCTLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }
#endif

#ifdef __ARM_NEON
  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::NeonBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }

  //--------------------------------------------------------------------------
  template<unsigned int MAX>
  inline void Deserializer::deserialize(Internal::NeonTLBitMask<MAX>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }
#endif

  //--------------------------------------------------------------------------
  template<
      typename DT, Internal::AllocationLifetime L, unsigned BLOAT, bool BIDIR>
  inline void Deserializer::deserialize(
      Internal::CompoundBitMask<DT, L, BLOAT, BIDIR>& mask)
  //--------------------------------------------------------------------------
  {
    mask.deserialize(*this);
  }

  //--------------------------------------------------------------------------
  inline void Deserializer::deserialize(Domain& dom)
  //--------------------------------------------------------------------------
  {
    deserialize(dom.is_id);
    if (dom.is_id > 0)
      deserialize(dom.is_type);
    deserialize(dom.dim);
    for (int i = 0; i < 2 * dom.dim; i++) deserialize(dom.rect_data[i]);
  }

  //--------------------------------------------------------------------------
  inline void Deserializer::deserialize(DomainPoint& dp)
  //--------------------------------------------------------------------------
  {
    deserialize(dp.dim);
    if (dp.dim == 0)
      deserialize(dp.point_data[0]);
    else
    {
      for (int idx = 0; idx < dp.dim; idx++) deserialize(dp.point_data[idx]);
    }
  }

  //--------------------------------------------------------------------------
  inline void Deserializer::deserialize(Internal::CopySrcDstField& field)
  //--------------------------------------------------------------------------
  {
    deserialize(field.inst);
    deserialize(field.field_id);
    deserialize(field.redop_id);
    if (field.redop_id > 0)
    {
      deserialize<bool>(field.red_fold);
      deserialize<bool>(field.red_exclusive);
    }
    deserialize(field.serdez_id);
    deserialize(field.subfield_offset);
    deserialize(field.indirect_index);
    if (field.size > Internal::CopySrcDstField::MAX_DIRECT_SIZE)
    {
      free(field.fill_data.indirect);
      field.fill_data.indirect = nullptr;
    }
    deserialize(field.size);
    // we know if there's a fill value if the field ID is -1
    if (field.field_id == (Realm::FieldID)-1)
    {
      if (field.size > Internal::CopySrcDstField::MAX_DIRECT_SIZE)
      {
        field.fill_data.indirect = malloc(field.size);
        deserialize(field.fill_data.indirect, field.size);
      }
      else
        deserialize(field.fill_data.direct, field.size);
    }
  }

  //--------------------------------------------------------------------------
  inline void Deserializer::deserialize(void* dst, size_t bytes)
  //--------------------------------------------------------------------------
  {
    legion_assert((index + bytes) <= total_bytes);
    std::memcpy(dst, buffer + index, bytes);
    index += bytes;
#ifdef LEGION_DEBUG
    context_bytes += bytes;
#endif
  }

  //--------------------------------------------------------------------------
  inline void Deserializer::begin_context(void)
  //--------------------------------------------------------------------------
  {
#ifdef LEGION_DEBUG
    // Save our enclosing context on the stack
    decltype(context_bytes) sent_context = 0;
    std::memcpy(&sent_context, buffer + index, sizeof(sent_context));
    index += sizeof(context_bytes);
    // Check to make sure that they match
    legion_assert(sent_context == context_bytes);
    context_bytes = 0;
#endif
  }

  //--------------------------------------------------------------------------
  inline void Deserializer::end_context(void)
  //--------------------------------------------------------------------------
  {
#ifdef LEGION_DEBUG
    // Read the send context size out of the buffer
    decltype(context_bytes) sent_context = 0;
    std::memcpy(&sent_context, buffer + index, sizeof(sent_context));
    index += sizeof(context_bytes);
    // Check to make sure that they match
    legion_assert(sent_context == context_bytes);
    context_bytes = 0;
#endif
  }

  //--------------------------------------------------------------------------
  inline size_t Deserializer::get_remaining_bytes(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(index <= total_bytes);
    return total_bytes - index;
  }

  //--------------------------------------------------------------------------
  inline const void* Deserializer::get_current_pointer(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(index <= total_bytes);
    return (const void*)(buffer + index);
  }

  //--------------------------------------------------------------------------
  inline void Deserializer::advance_pointer(size_t bytes)
  //--------------------------------------------------------------------------
  {
#ifdef LEGION_DEBUG
    legion_assert((index + bytes) <= total_bytes);
    context_bytes += bytes;
#endif
    index += bytes;
  }

}  // namespace Legion
