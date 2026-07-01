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

#ifndef __LEGION_HASHER_H__
#define __LEGION_HASHER_H__

#include "legion/api/geometry.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Murmur3Hasher
    /////////////////////////////////////////////////////////////

    /**
     * \class Murmur3Hasher
     * This class implements an object-oriented version of the
     * MurmurHash3 hashing algorithm for computing a 128-bit
     * hash value. It is taken from the public domain here:
     * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
     */
    class Murmur3Hasher {
    public:
      Murmur3Hasher(uint64_t seed = 0xCC892563);
      Murmur3Hasher(const Murmur3Hasher&) = delete;
      Murmur3Hasher& operator=(const Murmur3Hasher&) = delete;
    public:
      template<typename T, bool PRECISE = true>
      inline void hash(const T& value);
      inline void hash(const void* value, size_t size);
      inline void finalize(uint64_t hash[2]);
      struct Hash {
        uint64_t x = 0;
        uint64_t y = 0;
        inline bool operator!=(const Hash& rhs) const
        {
          return std::tie(x, y) != std::tie(rhs.x, rhs.y);
        }
        inline bool operator==(const Hash& rhs) const
        {
          return std::tie(x, y) == std::tie(rhs.x, rhs.y);
        }
        inline bool operator<(const Hash& rhs) const
        {
          return std::tie(x, y) < std::tie(rhs.x, rhs.y);
        }
        inline bool operator!(void) const { return (x == 0) && (y == 0); }
      };
      inline void finalize(Hash& hash);
    private:
      inline uint64_t rotl64(uint64_t x, uint8_t r);
      inline uint64_t fmix64(uint64_t k);
    private:
      uint8_t blocks[16];
      uint64_t h1, h2, len;
      uint8_t bytes;
    public:
      static constexpr uint64_t c1 = 0x87c37b91114253d5ULL;
      static constexpr uint64_t c2 = 0x4cf5ad432745937fULL;
    private:
      struct IndexSpaceHasher {
      public:
        IndexSpaceHasher(const Domain& d, Murmur3Hasher& h)
          : domain(d), hasher(h)
        { }
      public:
        template<typename N, typename T>
        static inline void demux(IndexSpaceHasher* functor)
        {
          const DomainT<N::N, T> is = functor->domain;
          for (RectInDomainIterator<N::N, T> itr(is); itr(); itr.step())
          {
            const Rect<N::N, T> rect = *itr;
            for (int d = 0; d < N::N; d++)
            {
              functor->hasher.hash(rect.lo[d]);
              functor->hasher.hash(rect.hi[d]);
            }
          }
        }
      public:
        const Domain& domain;
        Murmur3Hasher& hasher;
      };
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/utilities/hasher.inl"

#endif  // __LEGION_HASHER_H__
