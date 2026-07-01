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

#ifndef __LEGION_BITMASK_H__
#define __LEGION_BITMASK_H__

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <type_traits>

#ifndef __MACH__
// SJT: this comes first because some systems require __STDC_FORMAT_MACROS
//  to be defined before inttypes.h is included anywhere
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#endif

// Apple can go screw itself
#ifndef __MACH__
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#endif
#ifdef __ALTIVEC__
#include <altivec.h>
// Don't let IBM screw us over
#undef bool
#undef vector
#endif
#else  // !__MACH__
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#endif
#ifdef __ARM_NEON
#include "arm_neon.h"
#endif
#ifndef BITMASK_MAX_ALIGNMENT
#define BITMASK_MAX_ALIGNMENT (2 * sizeof(void*))
#endif
// This statically computes an integer log base 2 for a number
// which is guaranteed to be a power of 2. Adapted from
// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
#ifndef STATIC_LOG2
#define STATIC_LOG2(x) (LOG2_LOOKUP(uint32_t(x * 0x077CB531U) >> 27))
#endif
#ifndef LOG2_LOOKUP
#define LOG2_LOOKUP(x) \
  ((x == 0)  ? 0 :     \
   (x == 1)  ? 1 :     \
   (x == 2)  ? 28 :    \
   (x == 3)  ? 2 :     \
   (x == 4)  ? 29 :    \
   (x == 5)  ? 14 :    \
   (x == 6)  ? 24 :    \
   (x == 7)  ? 3 :     \
   (x == 8)  ? 30 :    \
   (x == 9)  ? 22 :    \
   (x == 10) ? 20 :    \
   (x == 11) ? 15 :    \
   (x == 12) ? 25 :    \
   (x == 13) ? 17 :    \
   (x == 14) ? 4 :     \
   (x == 15) ? 8 :     \
   (x == 16) ? 31 :    \
   (x == 17) ? 27 :    \
   (x == 18) ? 13 :    \
   (x == 19) ? 23 :    \
   (x == 20) ? 21 :    \
   (x == 21) ? 19 :    \
   (x == 22) ? 16 :    \
   (x == 23) ? 7 :     \
   (x == 24) ? 26 :    \
   (x == 25) ? 12 :    \
   (x == 26) ? 18 :    \
   (x == 27) ? 6 :     \
   (x == 28) ? 11 :    \
   (x == 29) ? 5 :     \
   (x == 30) ? 10 :    \
               9)
#endif

#include "legion/kernel/allocation.h"

namespace Legion {
  namespace Internal {

    // Internal helper name space for bitmasks
    namespace BitMaskHelp {
#ifdef __SSE2__
      template<bool READ_ONLY, typename T = uint64_t>
      class SSEView {
      public:
        inline SSEView(T* base, unsigned index)
          : ptr(base + ((sizeof(__m128d) / sizeof(T)) * index))
        { }
      public:
        inline operator __m128i(void) const
        {
          __m128i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
      public:
        inline void operator=(const __m128i& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const __m128d& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const SSEView<WHOCARES>& rhs)
        {
          memcpy(ptr, rhs.ptr, sizeof(__m128d));
        }
      public:
        T* const ptr;
      };
      template<typename T>
      class SSEView<true, T> {
      public:
        inline SSEView(const T* base, unsigned index)
          : ptr(base + ((sizeof(__m128d) / sizeof(T)) * index))
        { }
      public:
        inline operator __m128i(void) const
        {
          __m128i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
      public:
        const T* const ptr;
      };
#endif
#ifdef __AVX__
      template<bool READ_ONLY, typename T = uint64_t>
      class AVXView {
      public:
        inline AVXView(T* base, unsigned index)
          : ptr(base + ((sizeof(__m256d) / sizeof(T)) * index))
        { }
      public:
#ifdef __AVX2__
        inline operator __m256i(void) const
        {
          __m256i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
#else
        inline operator __m256d(void) const
        {
          __m256d result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
#endif
      public:
        inline void operator=(const __m256i& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const __m256d& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const AVXView<WHOCARES>& rhs)
        {
          memcpy(ptr, rhs.ptr, sizeof(__m256d));
        }
      public:
        T* const ptr;
      };
      template<typename T>
      class AVXView<true, T> {
      public:
        inline AVXView(const T* base, unsigned index)
          : ptr(base + ((sizeof(__m256d) / sizeof(T)) * index))
        { }
      public:
        inline operator __m256i(void) const
        {
          __m256i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __m256d(void) const
        {
          __m256d result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        const T* const ptr;
      };
#endif
#ifdef __ALTIVEC__
      template<bool READ_ONLY, typename T = uint64_t>
      class PPCView {
      public:
        inline PPCView(T* base, unsigned index)
          : ptr(base + ((sizeof(__vector double) / sizeof(T)) * index))
        { }
      public:
        inline operator __vector unsigned long long(void) const
        {
          __vector unsigned long long result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __vector double(void) const
        {
          __vector double result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        inline void operator=(const __vector unsigned long long& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const __vector double& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const PPCView<WHOCARES>& rhs)
        {
          memcpy(ptr, rhs.ptr, sizeof(__vector double));
        }
      public:
        T* const ptr;
      };
      template<typename T>
      class PPCView<true, T> {
      public:
        inline PPCView(const T* base, unsigned index)
          : ptr(base + ((sizeof(__vector double) / sizeof(T)) * index))
        { }
      public:
        inline operator __vector unsigned long long(void) const
        {
          __vector unsigned long long result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __vector double(void) const
        {
          __vector double result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        const T* const ptr;
      };
#endif
#ifdef __ARM_NEON
      template<bool READ_ONLY, typename T = uint64_t>
      class NeonView {
      public:
        inline NeonView(T* base, unsigned index)
          : ptr(base + ((sizeof(float32x4_t) / sizeof(T)) * index))
        { }
      public:
        inline operator uint32x4_t(void) const
        {
          uint32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator float32x4_t(void) const
        {
          float32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        inline void operator=(const uint32x4_t& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const float32x4_t& value)
        {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const NeonView<WHOCARES>& rhs)
        {
          memcpy(ptr, rhs.ptr, sizeof(float32x4_t));
        }
      public:
        T* const ptr;
      };
      template<typename T>
      class NeonView<true, T> {
      public:
        inline NeonView(const T* base, unsigned index)
          : ptr(base + ((sizeof(float32x4_t) / sizeof(T)) * index))
        { }
      public:
        inline operator uint32x4_t(void) const
        {
          uint32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator float32x4_t(void) const
        {
          float32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        const T* const ptr;
      };
#endif

      // Help with safe type-punning of bit representations
      // This is only because C++ is a stupid fucking language
      // and doesn't even follow the same semantics as C's union
      // As a result we explode our compilation time and generate worse code
      template<int MAX, typename ELEMENT_TYPE = uint64_t>
      struct BitVector {
      public:
#ifdef __SSE2__
        inline SSEView<false, ELEMENT_TYPE> sse_view(unsigned index)
        {
          return SSEView<false, ELEMENT_TYPE>(bit_vector, index);
        }
        inline SSEView<true, ELEMENT_TYPE> sse_view(unsigned index) const
        {
          return SSEView<true, ELEMENT_TYPE>(bit_vector, index);
        }
#endif
#ifdef __AVX__
        inline AVXView<false, ELEMENT_TYPE> avx_view(unsigned index)
        {
          return AVXView<false, ELEMENT_TYPE>(bit_vector, index);
        }
        inline AVXView<true, ELEMENT_TYPE> avx_view(unsigned index) const
        {
          return AVXView<true, ELEMENT_TYPE>(bit_vector, index);
        }
#endif
#ifdef __ALTIVEC__
        inline PPCView<false, ELEMENT_TYPE> ppc_view(unsigned index)
        {
          return PPCView<false, ELEMENT_TYPE>(bit_vector, index);
        }
        inline PPCView<true, ELEMENT_TYPE> ppc_view(unsigned index) const
        {
          return PPCView<true, ELEMENT_TYPE>(bit_vector, index);
        }
#endif
#ifdef __ARM_NEON
        inline NeonView<false, ELEMENT_TYPE> neon_view(unsigned index)
        {
          return NeonView<false, ELEMENT_TYPE>(bit_vector, index);
        }
        inline NeonView<true, ELEMENT_TYPE> neon_view(unsigned index) const
        {
          return NeonView<true, ELEMENT_TYPE>(bit_vector, index);
        }
#endif
      public:
        // Number of bits in the bit vector based element
        static constexpr unsigned ELEMENT_SIZE = 8 * sizeof(ELEMENT_TYPE);
        static_assert((MAX % ELEMENT_SIZE) == 0);
        ELEMENT_TYPE bit_vector[MAX / ELEMENT_SIZE];
        // Shift to get the upper bits for indexing assuming a 64-bit base type
        static constexpr unsigned SHIFT = STATIC_LOG2(ELEMENT_SIZE);
        // Mask to get the lower bits for indexing assuming a 64-bit base type
        static constexpr unsigned MASK = ELEMENT_SIZE - 1;
      };
    };  // namespace BitMaskHelp

    /////////////////////////////////////////////////////////////
    // Bit Mask
    /////////////////////////////////////////////////////////////
    template<
        typename T, unsigned int MAX, unsigned int SHIFT, unsigned int MASK>
    class BitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE = 8 * sizeof(T);
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit BitMask(T init = 0);
      BitMask(const BitMask& rhs);
      ~BitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const BitMask& rhs) const;
      inline bool operator<(const BitMask& rhs) const;
      inline bool operator!=(const BitMask& rhs) const;
    public:
      inline const T& operator[](const unsigned& idx) const;
      inline T& operator[](const unsigned& idx);
      inline BitMask& operator=(const BitMask& rhs);
    public:
      inline BitMask operator~(void) const;
      inline BitMask operator|(const BitMask& rhs) const;
      inline BitMask operator&(const BitMask& rhs) const;
      inline BitMask operator^(const BitMask& rhs) const;
    public:
      inline BitMask& operator|=(const BitMask& rhs);
      inline BitMask& operator&=(const BitMask& rhs);
      inline BitMask& operator^=(const BitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const BitMask& rhs) const;
      // Set difference
      inline BitMask operator-(const BitMask& rhs) const;
      inline BitMask& operator-=(const BitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline BitMask operator<<(unsigned shift) const;
      inline BitMask operator>>(unsigned shift) const;
    public:
      inline BitMask& operator<<=(unsigned shift);
      inline BitMask& operator>>=(unsigned shift);
    public:
      inline T get_hash_key(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(
          const BitMask<unsigned, MAX, SHIFT, MASK>& mask);
      static inline unsigned pop_count(
          const BitMask<unsigned long, MAX, SHIFT, MASK>& mask);
      static inline unsigned pop_count(
          const BitMask<unsigned long long, MAX, SHIFT, MASK>& mask);
    protected:
      T bit_vector[BIT_ELMTS];
    };

    /////////////////////////////////////////////////////////////
    // Two-Level Bit Mask
    /////////////////////////////////////////////////////////////
    /*
     * This class is a two-level bit mask which makes the
     * operations * ! & all faster at the cost of making the
     * other operations slower.  This done by using a summary
     * mask which keeps track of whether any bits are set in
     * the word at a given location in the summary mask.  The
     * summary is a single instance of the summary type ST.
     */
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    class TLBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE = 8 * sizeof(T);
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit TLBitMask(T init = 0);
      TLBitMask(const TLBitMask& rhs);
      ~TLBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const TLBitMask& rhs) const;
      inline bool operator<(const TLBitMask& rhs) const;
      inline bool operator!=(const TLBitMask& rhs) const;
    public:
      inline const T& operator[](const unsigned& idx) const;
      inline T& operator[](const unsigned& idx);
      inline TLBitMask& operator=(const TLBitMask& rhs);
    public:
      inline TLBitMask operator~(void) const;
      inline TLBitMask operator|(const TLBitMask& rhs) const;
      inline TLBitMask operator&(const TLBitMask& rhs) const;
      inline TLBitMask operator^(const TLBitMask& rhs) const;
    public:
      inline TLBitMask& operator|=(const TLBitMask& rhs);
      inline TLBitMask& operator&=(const TLBitMask& rhs);
      inline TLBitMask& operator^=(const TLBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const TLBitMask& rhs) const;
      // Set difference
      inline TLBitMask operator-(const TLBitMask& rhs) const;
      inline TLBitMask& operator-=(const TLBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline TLBitMask operator<<(unsigned shift) const;
      inline TLBitMask operator>>(unsigned shift) const;
    public:
      inline TLBitMask& operator<<=(unsigned shift);
      inline TLBitMask& operator>>=(unsigned shift);
    public:
      inline T get_hash_key(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(
          const TLBitMask<unsigned, MAX, SHIFT, MASK>& mask);
      static inline unsigned pop_count(
          const TLBitMask<unsigned long, MAX, SHIFT, MASK>& mask);
      static inline unsigned pop_count(
          const TLBitMask<unsigned long long, MAX, SHIFT, MASK>& mask);
    protected:
      T bit_vector[BIT_ELMTS];
      T sum_mask;
    };

#ifdef __SSE2__
    /////////////////////////////////////////////////////////////
    // SSE Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) SSEBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned SSE_ELMTS = MAX / 128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit SSEBitMask(uint64_t init = 0);
      SSEBitMask(const SSEBitMask& rhs);
      ~SSEBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const SSEBitMask& rhs) const;
      inline bool operator<(const SSEBitMask& rhs) const;
      inline bool operator!=(const SSEBitMask& rhs) const;
    public:
      inline BitMaskHelp::SSEView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::SSEView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline SSEBitMask& operator=(const SSEBitMask& rhs);
    public:
      inline SSEBitMask operator~(void) const;
      inline SSEBitMask operator|(const SSEBitMask& rhs) const;
      inline SSEBitMask operator&(const SSEBitMask& rhs) const;
      inline SSEBitMask operator^(const SSEBitMask& rhs) const;
    public:
      inline SSEBitMask& operator|=(const SSEBitMask& rhs);
      inline SSEBitMask& operator&=(const SSEBitMask& rhs);
      inline SSEBitMask& operator^=(const SSEBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const SSEBitMask& rhs) const;
      // Set difference
      inline SSEBitMask operator-(const SSEBitMask& rhs) const;
      inline SSEBitMask& operator-=(const SSEBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline SSEBitMask operator<<(unsigned shift) const;
      inline SSEBitMask operator>>(unsigned shift) const;
    public:
      inline SSEBitMask& operator<<=(unsigned shift);
      inline SSEBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const SSEBitMask<MAX>& mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
    };

    /////////////////////////////////////////////////////////////
    // SSE Two-Level Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) SSETLBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned SSE_ELMTS = MAX / 128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit SSETLBitMask(uint64_t init = 0);
      SSETLBitMask(const SSETLBitMask& rhs);
      ~SSETLBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const SSETLBitMask& rhs) const;
      inline bool operator<(const SSETLBitMask& rhs) const;
      inline bool operator!=(const SSETLBitMask& rhs) const;
    public:
      inline BitMaskHelp::SSEView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::SSEView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline SSETLBitMask& operator=(const SSETLBitMask& rhs);
    public:
      inline SSETLBitMask operator~(void) const;
      inline SSETLBitMask operator|(const SSETLBitMask& rhs) const;
      inline SSETLBitMask operator&(const SSETLBitMask& rhs) const;
      inline SSETLBitMask operator^(const SSETLBitMask& rhs) const;
    public:
      inline SSETLBitMask& operator|=(const SSETLBitMask& rhs);
      inline SSETLBitMask& operator&=(const SSETLBitMask& rhs);
      inline SSETLBitMask& operator^=(const SSETLBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const SSETLBitMask& rhs) const;
      // Set difference
      inline SSETLBitMask operator-(const SSETLBitMask& rhs) const;
      inline SSETLBitMask& operator-=(const SSETLBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline SSETLBitMask operator<<(unsigned shift) const;
      inline SSETLBitMask operator>>(unsigned shift) const;
    public:
      inline SSETLBitMask& operator<<=(unsigned shift);
      inline SSETLBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const SSETLBitMask<MAX>& mask);
      static inline uint64_t extract_mask(__m128i value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask;
    };
#endif  // __SSE2__

#ifdef __AVX__
    /////////////////////////////////////////////////////////////
    // AVX Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(32) AVXBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned AVX_ELMTS = MAX / 256;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit AVXBitMask(uint64_t init = 0);
      AVXBitMask(const AVXBitMask& rhs);
      ~AVXBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const AVXBitMask& rhs) const;
      inline bool operator<(const AVXBitMask& rhs) const;
      inline bool operator!=(const AVXBitMask& rhs) const;
    public:
      inline BitMaskHelp::AVXView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::AVXView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline AVXBitMask& operator=(const AVXBitMask& rhs);
    public:
      inline AVXBitMask operator~(void) const;
      inline AVXBitMask operator|(const AVXBitMask& rhs) const;
      inline AVXBitMask operator&(const AVXBitMask& rhs) const;
      inline AVXBitMask operator^(const AVXBitMask& rhs) const;
    public:
      inline AVXBitMask& operator|=(const AVXBitMask& rhs);
      inline AVXBitMask& operator&=(const AVXBitMask& rhs);
      inline AVXBitMask& operator^=(const AVXBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const AVXBitMask& rhs) const;
      // Set difference
      inline AVXBitMask operator-(const AVXBitMask& rhs) const;
      inline AVXBitMask& operator-=(const AVXBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline AVXBitMask operator<<(unsigned shift) const;
      inline AVXBitMask operator>>(unsigned shift) const;
    public:
      inline AVXBitMask& operator<<=(unsigned shift);
      inline AVXBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const AVXBitMask<MAX>& mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
    };

    /////////////////////////////////////////////////////////////
    // AVX Two-Level Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(32) AVXTLBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned AVX_ELMTS = MAX / 256;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit AVXTLBitMask(uint64_t init = 0);
      AVXTLBitMask(const AVXTLBitMask& rhs);
      ~AVXTLBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const AVXTLBitMask& rhs) const;
      inline bool operator<(const AVXTLBitMask& rhs) const;
      inline bool operator!=(const AVXTLBitMask& rhs) const;
    public:
      inline BitMaskHelp::AVXView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::AVXView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline AVXTLBitMask& operator=(const AVXTLBitMask& rhs);
    public:
      inline AVXTLBitMask operator~(void) const;
      inline AVXTLBitMask operator|(const AVXTLBitMask& rhs) const;
      inline AVXTLBitMask operator&(const AVXTLBitMask& rhs) const;
      inline AVXTLBitMask operator^(const AVXTLBitMask& rhs) const;
    public:
      inline AVXTLBitMask& operator|=(const AVXTLBitMask& rhs);
      inline AVXTLBitMask& operator&=(const AVXTLBitMask& rhs);
      inline AVXTLBitMask& operator^=(const AVXTLBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const AVXTLBitMask& rhs) const;
      // Set difference
      inline AVXTLBitMask operator-(const AVXTLBitMask& rhs) const;
      inline AVXTLBitMask& operator-=(const AVXTLBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline AVXTLBitMask operator<<(unsigned shift) const;
      inline AVXTLBitMask operator>>(unsigned shift) const;
    public:
      inline AVXTLBitMask& operator<<=(unsigned shift);
      inline AVXTLBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const AVXTLBitMask<MAX>& mask);
      static inline uint64_t extract_mask(__m256i value);
      static inline uint64_t extract_mask(__m256d value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask;
    };
#endif  // __AVX__

#ifdef __ALTIVEC__
    /////////////////////////////////////////////////////////////
    // PPC Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) PPCBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned PPC_ELMTS = MAX / 128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit PPCBitMask(uint64_t init = 0);
      PPCBitMask(const PPCBitMask& rhs);
      ~PPCBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const PPCBitMask& rhs) const;
      inline bool operator<(const PPCBitMask& rhs) const;
      inline bool operator!=(const PPCBitMask& rhs) const;
    public:
      inline BitMaskHelp::PPCView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::PPCView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline PPCBitMask& operator=(const PPCBitMask& rhs);
    public:
      inline PPCBitMask operator~(void) const;
      inline PPCBitMask operator|(const PPCBitMask& rhs) const;
      inline PPCBitMask operator&(const PPCBitMask& rhs) const;
      inline PPCBitMask operator^(const PPCBitMask& rhs) const;
    public:
      inline PPCBitMask& operator|=(const PPCBitMask& rhs);
      inline PPCBitMask& operator&=(const PPCBitMask& rhs);
      inline PPCBitMask& operator^=(const PPCBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const PPCBitMask& rhs) const;
      // Set difference
      inline PPCBitMask operator-(const PPCBitMask& rhs) const;
      inline PPCBitMask& operator-=(const PPCBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline PPCBitMask operator<<(unsigned shift) const;
      inline PPCBitMask operator>>(unsigned shift) const;
    public:
      inline PPCBitMask& operator<<=(unsigned shift);
      inline PPCBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const PPCBitMask<MAX>& mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
    };

    /////////////////////////////////////////////////////////////
    // PPC Two-Level Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) PPCTLBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned PPC_ELMTS = MAX / 128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit PPCTLBitMask(uint64_t init = 0);
      PPCTLBitMask(const PPCTLBitMask& rhs);
      ~PPCTLBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const PPCTLBitMask& rhs) const;
      inline bool operator<(const PPCTLBitMask& rhs) const;
      inline bool operator!=(const PPCTLBitMask& rhs) const;
    public:
      inline BitMaskHelp::PPCView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::PPCView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline PPCTLBitMask& operator=(const PPCTLBitMask& rhs);
    public:
      inline PPCTLBitMask operator~(void) const;
      inline PPCTLBitMask operator|(const PPCTLBitMask& rhs) const;
      inline PPCTLBitMask operator&(const PPCTLBitMask& rhs) const;
      inline PPCTLBitMask operator^(const PPCTLBitMask& rhs) const;
    public:
      inline PPCTLBitMask& operator|=(const PPCTLBitMask& rhs);
      inline PPCTLBitMask& operator&=(const PPCTLBitMask& rhs);
      inline PPCTLBitMask& operator^=(const PPCTLBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const PPCTLBitMask& rhs) const;
      // Set difference
      inline PPCTLBitMask operator-(const PPCTLBitMask& rhs) const;
      inline PPCTLBitMask& operator-=(const PPCTLBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline PPCTLBitMask operator<<(unsigned shift) const;
      inline PPCTLBitMask operator>>(unsigned shift) const;
    public:
      inline PPCTLBitMask& operator<<=(unsigned shift);
      inline PPCTLBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const PPCTLBitMask<MAX>& mask);
      static inline uint64_t extract_mask(__vector unsigned long long value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask;
    };
#endif  // __ALTIVEC__

#ifdef __ARM_NEON
    /////////////////////////////////////////////////////////////
    // Neon Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) NeonBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned NEON_ELMTS = MAX / 128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit NeonBitMask(uint64_t init = 0);
      NeonBitMask(const NeonBitMask& rhs);
      ~NeonBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const NeonBitMask& rhs) const;
      inline bool operator<(const NeonBitMask& rhs) const;
      inline bool operator!=(const NeonBitMask& rhs) const;
    public:
      inline BitMaskHelp::NeonView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::NeonView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline NeonBitMask& operator=(const NeonBitMask& rhs);
    public:
      inline NeonBitMask operator~(void) const;
      inline NeonBitMask operator|(const NeonBitMask& rhs) const;
      inline NeonBitMask operator&(const NeonBitMask& rhs) const;
      inline NeonBitMask operator^(const NeonBitMask& rhs) const;
    public:
      inline NeonBitMask& operator|=(const NeonBitMask& rhs);
      inline NeonBitMask& operator&=(const NeonBitMask& rhs);
      inline NeonBitMask& operator^=(const NeonBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const NeonBitMask& rhs) const;
      // Set difference
      inline NeonBitMask operator-(const NeonBitMask& rhs) const;
      inline NeonBitMask& operator-=(const NeonBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline NeonBitMask operator<<(unsigned shift) const;
      inline NeonBitMask operator>>(unsigned shift) const;
    public:
      inline NeonBitMask& operator<<=(unsigned shift);
      inline NeonBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const NeonBitMask<MAX>& mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
    };

    /////////////////////////////////////////////////////////////
    // Neon Two-Level Bit Mask
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) NeonTLBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE =
          BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX / ELEMENT_SIZE;
      static constexpr unsigned NEON_ELMTS = MAX / 128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit NeonTLBitMask(uint64_t init = 0);
      NeonTLBitMask(const NeonTLBitMask& rhs);
      ~NeonTLBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const NeonTLBitMask& rhs) const;
      inline bool operator<(const NeonTLBitMask& rhs) const;
      inline bool operator!=(const NeonTLBitMask& rhs) const;
    public:
      inline BitMaskHelp::NeonView<true> operator()(const unsigned& idx) const;
      inline BitMaskHelp::NeonView<false> operator()(const unsigned& idx);
      inline const uint64_t& operator[](const unsigned& idx) const;
      inline uint64_t& operator[](const unsigned& idx);
      inline NeonTLBitMask& operator=(const NeonTLBitMask& rhs);
    public:
      inline NeonTLBitMask operator~(void) const;
      inline NeonTLBitMask operator|(const NeonTLBitMask& rhs) const;
      inline NeonTLBitMask operator&(const NeonTLBitMask& rhs) const;
      inline NeonTLBitMask operator^(const NeonTLBitMask& rhs) const;
    public:
      inline NeonTLBitMask& operator|=(const NeonTLBitMask& rhs);
      inline NeonTLBitMask& operator&=(const NeonTLBitMask& rhs);
      inline NeonTLBitMask& operator^=(const NeonTLBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const NeonTLBitMask& rhs) const;
      // Set difference
      inline NeonTLBitMask operator-(const NeonTLBitMask& rhs) const;
      inline NeonTLBitMask& operator-=(const NeonTLBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline NeonTLBitMask operator<<(unsigned shift) const;
      inline NeonTLBitMask operator>>(unsigned shift) const;
    public:
      inline NeonTLBitMask& operator<<=(unsigned shift);
      inline NeonTLBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      inline const uint64_t* base(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DT>
      inline void deserialize(DT& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const NeonTLBitMask<MAX>& mask);
      static inline uint64_t extract_mask(uint32x4_t value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask;
    };
#endif  // __ARM_NEON

    template<
        typename DT, AllocationLifetime L, unsigned BLOAT = 1,
        bool BIDIR = true>
    class CompoundBitMask : public NoHeapify {
    public:
      static constexpr unsigned ELEMENT_SIZE = DT::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = DT::BIT_ELMTS;
      static constexpr unsigned MAXSIZE = DT::MAXSIZE;
    public:
      explicit CompoundBitMask(uint64_t init = 0);
      CompoundBitMask(const CompoundBitMask& rhs);
      ~CompoundBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_next_set(unsigned start) const;
      inline int find_index(unsigned bit) const;
      inline int get_index(unsigned index) const;
      inline bool empty(void) const;
      inline void clear(void);
    public:
      inline size_t size(void) const { return pop_count(); }
      inline bool contains(unsigned bit) const { return is_set(bit); }
      inline void add(unsigned bit) { set_bit(bit); }
      inline void insert(unsigned bit) { set_bit(bit); }
      inline void remove(unsigned bit) { unset_bit(bit); }
    public:
      inline bool operator==(const CompoundBitMask& rhs) const;
      inline bool operator<(const CompoundBitMask& rhs) const;
      inline bool operator!=(const CompoundBitMask& rhs) const;
    public:
      inline CompoundBitMask& operator=(const CompoundBitMask& rhs);
    public:
      inline CompoundBitMask operator~(void) const;
      inline CompoundBitMask operator|(const CompoundBitMask& rhs) const;
      inline CompoundBitMask operator&(const CompoundBitMask& rhs) const;
      inline CompoundBitMask operator^(const CompoundBitMask& rhs) const;
    public:
      inline CompoundBitMask& operator|=(const CompoundBitMask& rhs);
      inline CompoundBitMask& operator&=(const CompoundBitMask& rhs);
      inline CompoundBitMask& operator^=(const CompoundBitMask& rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const CompoundBitMask& rhs) const;
      // Set difference
      inline CompoundBitMask operator-(const CompoundBitMask& rhs) const;
      inline CompoundBitMask& operator-=(const CompoundBitMask& rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    public:
      inline CompoundBitMask operator<<(unsigned shift) const;
      inline CompoundBitMask operator>>(unsigned shift) const;
    public:
      inline CompoundBitMask& operator<<=(unsigned shift);
      inline CompoundBitMask& operator>>=(unsigned shift);
    public:
      inline uint64_t get_hash_key(void) const;
      template<typename ST>
      inline void serialize(ST& rez) const;
      template<typename DZ>
      inline void deserialize(DZ& derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR& functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(
          const CompoundBitMask<DT, L, BLOAT, BIDIR>& mask);
    protected:
      inline bool is_sparse(void) const;
      inline void sparsify(void);
    protected:
      using IT = typename std::conditional<
          DT::MAXSIZE <= (1 << 8), uint8_t,
          typename std::conditional<
              DT::MAXSIZE <= (1 << 16), uint16_t, uint32_t>::type>::type;
      static constexpr size_t MAX_SPARSE =
          (BLOAT * sizeof(DT*) + sizeof(IT) - 1) / sizeof(IT);
      using SA = std::array<IT, MAX_SPARSE>;
      union {
        // The sparse array is unique and sorted
        SA sparse;
        HeapifyBox<DT, L>* dense;
      } mask;
      unsigned sparse_size;
    };

    // A little bit of logic here to figure out the
    // kind of bit mask to use for FieldMask

// The folowing macros are used in the FieldMask instantiation of BitMask
// If you change one you probably have to change the others too
#define LEGION_FIELD_MASK_FIELD_TYPE uint64_t
#define LEGION_FIELD_MASK_FIELD_SHIFT 6
#define LEGION_FIELD_MASK_FIELD_MASK 0x3F
#define LEGION_FIELD_MASK_FIELD_ALL_ONES 0xFFFFFFFFFFFFFFFF

#if defined(__AVX__)
#if (LEGION_MAX_FIELDS > 256)
    typedef AVXTLBitMask<LEGION_MAX_FIELDS> FieldMask;
#elif (LEGION_MAX_FIELDS > 128)
    typedef AVXBitMask<LEGION_MAX_FIELDS> FieldMask;
#elif (LEGION_MAX_FIELDS > 64)
    typedef SSEBitMask<LEGION_MAX_FIELDS> FieldMask;
#else
    typedef BitMask<
        LEGION_FIELD_MASK_FIELD_TYPE, LEGION_MAX_FIELDS,
        LEGION_FIELD_MASK_FIELD_SHIFT, LEGION_FIELD_MASK_FIELD_MASK>
        FieldMask;
#endif
#elif defined(__SSE2__)
#if (LEGION_MAX_FIELDS > 128)
    typedef SSETLBitMask<LEGION_MAX_FIELDS> FieldMask;
#elif (LEGION_MAX_FIELDS > 64)
    typedef SSEBitMask<LEGION_MAX_FIELDS> FieldMask;
#else
    typedef BitMask<
        LEGION_FIELD_MASK_FIELD_TYPE, LEGION_MAX_FIELDS,
        LEGION_FIELD_MASK_FIELD_SHIFT, LEGION_FIELD_MASK_FIELD_MASK>
        FieldMask;
#endif
#elif defined(__ALTIVEC__)
#if (LEGION_MAX_FIELDS > 128)
    typedef PPCTLBitMask<LEGION_MAX_FIELDS> FieldMask;
#elif (LEGION_MAX_FIELDS > 64)
    typedef PPCBitMask<LEGION_MAX_FIELDS> FieldMask;
#else
    typedef BitMask<
        LEGION_FIELD_MASK_FIELD_TYPE, LEGION_MAX_FIELDS,
        LEGION_FIELD_MASK_FIELD_SHIFT, LEGION_FIELD_MASK_FIELD_MASK>
        FieldMask;
#endif
#elif defined(__ARM_NEON)
#if (LEGION_MAX_FIELDS > 128)
    typedef NeonTLBitMask<LEGION_MAX_FIELDS> FieldMask;
#elif (LEGION_MAX_FIELDS > 64)
    typedef NeonBitMask<LEGION_MAX_FIELDS> FieldMask;
#else
    typedef BitMask<
        LEGION_FIELD_MASK_FIELD_TYPE, LEGION_MAX_FIELDS,
        LEGION_FIELD_MASK_FIELD_SHIFT, LEGION_FIELD_MASK_FIELD_MASK>
        FieldMask;
#endif
#else
#if (LEGION_MAX_FIELDS > 64)
    typedef TLBitMask<
        LEGION_FIELD_MASK_FIELD_TYPE, LEGION_MAX_FIELDS,
        LEGION_FIELD_MASK_FIELD_SHIFT, LEGION_FIELD_MASK_FIELD_MASK>
        FieldMask;
#else
    typedef BitMask<
        LEGION_FIELD_MASK_FIELD_TYPE, LEGION_MAX_FIELDS,
        LEGION_FIELD_MASK_FIELD_SHIFT, LEGION_FIELD_MASK_FIELD_MASK>
        FieldMask;
#endif
#endif
#undef LEGION_FIELD_MASK_FIELD_SHIFT
#undef LEGION_FIELD_MASK_FIELD_MASK

    // Similar logic as field masks for node masks

// The following macros are used in the NodeMask instantiation of BitMask
// If you change one you probably have to change the others too
#define LEGION_NODE_MASK_NODE_TYPE uint64_t
#define LEGION_NODE_MASK_NODE_SHIFT 6
#define LEGION_NODE_MASK_NODE_MASK 0x3F
#define LEGION_NODE_MASK_NODE_ALL_ONES 0xFFFFFFFFFFFFFFFF

#if defined(__AVX__)
#if (LEGION_MAX_NUM_NODES > 256)
    typedef AVXTLBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#elif (LEGION_MAX_NUM_NODES > 128)
    typedef AVXBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#elif (LEGION_MAX_NUM_NODES > 64)
    typedef SSEBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<
        LEGION_NODE_MASK_NODE_TYPE, LEGION_MAX_NUM_NODES,
        LEGION_NODE_MASK_NODE_SHIFT, LEGION_NODE_MASK_NODE_MASK>
        NodeMask;
#endif
#elif defined(__SSE2__)
#if (LEGION_MAX_NUM_NODES > 128)
    typedef SSETLBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#elif (LEGION_MAX_NUM_NODES > 64)
    typedef SSEBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<
        LEGION_NODE_MASK_NODE_TYPE, LEGION_MAX_NUM_NODES,
        LEGION_NODE_MASK_NODE_SHIFT, LEGION_NODE_MASK_NODE_MASK>
        NodeMask;
#endif
#elif defined(__ALTIVEC__)
#if (LEGION_MAX_NUM_NODES > 128)
    typedef PPCTLBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#elif (LEGION_MAX_NUM_NODES > 64)
    typedef PPCBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<
        LEGION_NODE_MASK_NODE_TYPE, LEGION_MAX_NUM_NODES,
        LEGION_NODE_MASK_NODE_SHIFT, LEGION_NODE_MASK_NODE_MASK>
        NodeMask;
#endif
#elif defined(__ARM_NEON)
#if (LEGION_MAX_NUM_NODES > 128)
    typedef NeonTLBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#elif (LEGION_MAX_NUM_NODES > 64)
    typedef NeonBitMask<LEGION_MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<
        LEGION_NODE_MASK_NODE_TYPE, LEGION_MAX_NUM_NODES,
        LEGION_NODE_MASK_NODE_SHIFT, LEGION_NODE_MASK_NODE_MASK>
        NodeMask;
#endif
#else
#if (LEGION_MAX_NUM_NODES > 64)
    typedef TLBitMask<
        LEGION_NODE_MASK_NODE_TYPE, LEGION_MAX_NUM_NODES,
        LEGION_NODE_MASK_NODE_SHIFT, LEGION_NODE_MASK_NODE_MASK>
        NodeMask;
#else
    typedef BitMask<
        LEGION_NODE_MASK_NODE_TYPE, LEGION_MAX_NUM_NODES,
        LEGION_NODE_MASK_NODE_SHIFT, LEGION_NODE_MASK_NODE_MASK>
        NodeMask;
#endif
#endif
    template<AllocationLifetime L>
    using NodeSet = CompoundBitMask<NodeMask, L, 1 /*bloat*/, true /*bidir*/>;

#undef LEGION_NODE_MASK_NODE_SHIFT
#undef LEGION_NODE_MASK_NODE_MASK

// The following macros are used in the ProcessorMask instantiation of BitMask
// If you change one you probably have to change the others too
#define LEGION_PROC_MASK_PROC_TYPE uint64_t
#define LEGION_PROC_MASK_PROC_SHIFT 6
#define LEGION_PROC_MASK_PROC_MASK 0x3F
#define LEGION_PROC_MASK_PROC_ALL_ONES 0xFFFFFFFFFFFFFFFF

#if defined(__AVX__)
#if (LEGION_MAX_NUM_PROCS > 256)
    typedef AVXTLBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#elif (LEGION_MAX_NUM_PROCS > 128)
    typedef AVXBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#elif (LEGION_MAX_NUM_PROCS > 64)
    typedef SSEBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#else
    typedef BitMask<
        LEGION_PROC_MASK_PROC_TYPE, LEGION_MAX_NUM_PROCS,
        LEGION_PROC_MASK_PROC_SHIFT, LEGION_PROC_MASK_PROC_MASK>
        ProcessorMask;
#endif
#elif defined(__SSE2__)
#if (LEGION_MAX_NUM_PROCS > 128)
    typedef SSETLBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#elif (LEGION_MAX_NUM_PROCS > 64)
    typedef SSEBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#else
    typedef BitMask<
        LEGION_PROC_MASK_PROC_TYPE, LEGION_MAX_NUM_PROCS,
        LEGION_PROC_MASK_PROC_SHIFT, LEGION_PROC_MASK_PROC_MASK>
        ProcessorMask;
#endif
#elif defined(__ALTIVEC__)
#if (LEGION_MAX_NUM_PROCS > 128)
    typedef PPCTLBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#elif (LEGION_MAX_NUM_PROCS > 64)
    typedef PPCBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#else
    typedef BitMask<
        LEGION_PROC_MASK_PROC_TYPE, LEGION_MAX_NUM_PROCS,
        LEGION_PROC_MASK_PROC_SHIFT, LEGION_PROC_MASK_PROC_MASK>
        ProcessorMask;
#endif
#elif defined(__ARM_NEON)
#if (LEGION_MAX_NUM_PROCS > 128)
    typedef NeonTLBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#elif (LEGION_MAX_NUM_PROCS > 64)
    typedef NeonBitMask<LEGION_MAX_NUM_PROCS> ProcessorMask;
#else
    typedef BitMask<
        LEGION_PROC_MASK_PROC_TYPE, LEGION_MAX_NUM_PROCS,
        LEGION_PROC_MASK_PROC_SHIFT, LEGION_PROC_MASK_PROC_MASK>
        ProcessorMask;
#endif
#else
#if (LEGION_MAX_NUM_PROCS > 64)
    typedef TLBitMask<
        LEGION_PROC_MASK_PROC_TYPE, LEGION_MAX_NUM_PROCS,
        LEGION_PROC_MASK_PROC_SHIFT, LEGION_PROC_MASK_PROC_MASK>
        ProcessorMask;
#else
    typedef BitMask<
        LEGION_PROC_MASK_PROC_TYPE, LEGION_MAX_NUM_PROCS,
        LEGION_PROC_MASK_PROC_SHIFT, LEGION_PROC_MASK_PROC_MASK>
        ProcessorMask;
#endif
#endif

#undef PROC_SHIFT
#undef PROC_MASK

  }  // namespace Internal
}  // namespace Legion

#include "legion/utilities/bitmask.inl"

#endif  // __LEGION_BITMASK_H__
