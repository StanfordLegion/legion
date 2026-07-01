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

// Included from hasher.h - do not include this directly

// Useful for IDEs
#include "legion/utilities/hasher.h"

template<>
struct std::hash<Legion::Internal::Murmur3Hasher::Hash> {
  std::size_t operator()(
      const Legion::Internal::Murmur3Hasher::Hash& h) const noexcept
  {
    return h.x ^ (h.y << 1);
  }
};

template<>
struct std::equal_to<Legion::Internal::Murmur3Hasher::Hash> {
  constexpr bool operator()(
      const Legion::Internal::Murmur3Hasher::Hash& lhs,
      const Legion::Internal::Murmur3Hasher::Hash& rhs) const
  {
    return lhs.x == rhs.x && lhs.y == rhs.y;
  }
};

template<>
struct std::less<Legion::Internal::Murmur3Hasher::Hash> {
  constexpr bool operator()(
      const Legion::Internal::Murmur3Hasher::Hash& lhs,
      const Legion::Internal::Murmur3Hasher::Hash& rhs) const
  {
    return lhs.x < rhs.x && lhs.y < rhs.y;
  }
};

namespace Legion {
  namespace Internal {

    //-------------------------------------------------------------------------
    inline Murmur3Hasher::Murmur3Hasher(uint64_t seed)
      : h1(seed), h2(seed), len(0), bytes(0)
    //-------------------------------------------------------------------------
    { }

    //-------------------------------------------------------------------------
    template<typename T, bool PRECISE>
    inline void Murmur3Hasher::hash(const T& value)
    //-------------------------------------------------------------------------
    {
      static_assert(std::is_trivially_copyable<T>::value);
      const T* ptr = &value;
      const uint8_t* data = nullptr;
      static_assert(sizeof(ptr) == sizeof(data));
      memcpy(&data, &ptr, sizeof(data));
      for (unsigned idx = 0; idx < sizeof(T); idx++)
      {
        blocks[bytes++] = data[idx];
        if (bytes == 16)
        {
          // body
          uint64_t k1, k2;
          memcpy(&k1, blocks, sizeof(k1));
          memcpy(&k2, blocks + sizeof(k1), sizeof(k2));
          static_assert(sizeof(blocks) == (sizeof(k1) + sizeof(k2)));
          k1 *= c1;
          k1 = rotl64(k1, 31);
          k1 *= c2;
          h1 ^= k1;
          h1 = rotl64(h1, 27);
          h1 += h2;
          h1 = h1 * 5 + 0x52dce729;
          k2 *= c2;
          k2 = rotl64(k2, 33);
          k2 *= c1;
          h2 ^= k2;
          h2 = rotl64(h2, 31);
          h2 += h1;
          h2 = h2 * 5 + 0x38495ab5;
          len += 16;
          bytes = 0;
        }
      }
    }

    //-------------------------------------------------------------------------
    template<>
    inline void Murmur3Hasher::hash<Domain, true>(const Domain& value)
    //-------------------------------------------------------------------------
    {
      for (int i = 0; i < 2 * value.dim; i++) hash(value.rect_data[i]);
      if (!value.dense())
      {
        IndexSpaceHasher functor(value, *this);
        Internal::NT_TemplateHelper::demux<IndexSpaceHasher>(
            value.is_type, &functor);
      }
    }

    //-------------------------------------------------------------------------
    template<>
    inline void Murmur3Hasher::hash<Domain, false>(const Domain& value)
    //-------------------------------------------------------------------------
    {
      for (int i = 0; i < 2 * value.dim; i++) hash(value.rect_data[i]);
    }

    //-------------------------------------------------------------------------
    template<>
    inline void Murmur3Hasher::hash<DomainPoint, true>(const DomainPoint& value)
    //-------------------------------------------------------------------------
    {
      for (int i = 0; i < value.dim; i++) hash(value.point_data[i]);
    }

    //-------------------------------------------------------------------------
    template<>
    inline void Murmur3Hasher::hash<DomainPoint, false>(
        const DomainPoint& value)
    //-------------------------------------------------------------------------
    {
      for (int i = 0; i < value.dim; i++) hash(value.point_data[i]);
    }

    //-------------------------------------------------------------------------
    inline void Murmur3Hasher::hash(const void* value, size_t size)
    //-------------------------------------------------------------------------
    {
      const uint8_t* data = nullptr;
      static_assert(sizeof(data) == sizeof(value));
      memcpy(&data, &value, sizeof(data));
      for (unsigned idx = 0; idx < size; idx++)
      {
        blocks[bytes++] = data[idx];
        if (bytes == 16)
        {
          // body
          uint64_t k1, k2;
          memcpy(&k1, blocks, sizeof(k1));
          memcpy(&k2, blocks + sizeof(k1), sizeof(k2));
          static_assert(sizeof(blocks) == (sizeof(k1) + sizeof(k2)));
          k1 *= c1;
          k1 = rotl64(k1, 31);
          k1 *= c2;
          h1 ^= k1;
          h1 = rotl64(h1, 27);
          h1 += h2;
          h1 = h1 * 5 + 0x52dce729;
          k2 *= c2;
          k2 = rotl64(k2, 33);
          k2 *= c1;
          h2 ^= k2;
          h2 = rotl64(h2, 31);
          h2 += h1;
          h2 = h2 * 5 + 0x38495ab5;
          len += 16;
          bytes = 0;
        }
      }
    }

    //-------------------------------------------------------------------------
    inline void Murmur3Hasher::finalize(uint64_t hash[2])
    //-------------------------------------------------------------------------
    {
      // tail
      uint64_t k1 = 0;
      uint64_t k2 = 0;
      switch (bytes)
      {
        case 15:
          k2 ^= ((uint64_t)blocks[14]) << 48;
        case 14:
          k2 ^= ((uint64_t)blocks[13]) << 40;
        case 13:
          k2 ^= ((uint64_t)blocks[12]) << 32;
        case 12:
          k2 ^= ((uint64_t)blocks[11]) << 24;
        case 11:
          k2 ^= ((uint64_t)blocks[10]) << 16;
        case 10:
          k2 ^= ((uint64_t)blocks[9]) << 8;
        case 9:
          k2 ^= ((uint64_t)blocks[8]) << 0;
          k2 *= c2;
          k2 = rotl64(k2, 33);
          k2 *= c1;
          h2 ^= k2;

        case 8:
          k1 ^= ((uint64_t)blocks[7]) << 56;
        case 7:
          k1 ^= ((uint64_t)blocks[6]) << 48;
        case 6:
          k1 ^= ((uint64_t)blocks[5]) << 40;
        case 5:
          k1 ^= ((uint64_t)blocks[4]) << 32;
        case 4:
          k1 ^= ((uint64_t)blocks[3]) << 24;
        case 3:
          k1 ^= ((uint64_t)blocks[2]) << 16;
        case 2:
          k1 ^= ((uint64_t)blocks[1]) << 8;
        case 1:
          k1 ^= ((uint64_t)blocks[0]) << 0;
          k1 *= c1;
          k1 = rotl64(k1, 31);
          k1 *= c2;
          h1 ^= k1;
      }

      // finalization
      len += bytes;

      h1 ^= len;
      h2 ^= len;

      h1 += h2;
      h2 += h1;

      h1 = fmix64(h1);
      h2 = fmix64(h2);

      h1 += h2;
      h2 += h1;

      hash[0] = h1;
      hash[1] = h2;
    }

    //-------------------------------------------------------------------------
    inline void Murmur3Hasher::finalize(Hash& hash)
    //-------------------------------------------------------------------------
    {
      uint64_t temp[2];
      finalize(temp);
      hash.x = temp[0];
      hash.y = temp[1];
    }

    //-------------------------------------------------------------------------
    inline uint64_t Murmur3Hasher::rotl64(uint64_t x, uint8_t r)
    //-------------------------------------------------------------------------
    {
      return (x << r) | (x >> (64 - r));
    }

    //-------------------------------------------------------------------------
    inline uint64_t Murmur3Hasher::fmix64(uint64_t k)
    //-------------------------------------------------------------------------
    {
      k ^= k >> 33;
      k *= 0xff51afd7ed558ccdULL;
      k ^= k >> 33;
      k *= 0xc4ceb9fe1a85ec53ULL;
      k ^= k >> 33;
      return k;
    }

  }  // namespace Internal
}  // namespace Legion
