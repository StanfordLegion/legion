/* Copyright 2022 Stanford University, NVIDIA Corporation
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


#ifndef __BITMASK_H__
#define __BITMASK_H__

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <set>
#include <vector>
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
#else // !__MACH__
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
#define BITMASK_MAX_ALIGNMENT   (2*sizeof(void *))
#endif
// This statically computes an integer log base 2 for a number
// which is guaranteed to be a power of 2. Adapted from
// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
#ifndef STATIC_LOG2
#define STATIC_LOG2(x)  (LOG2_LOOKUP(uint32_t(x * 0x077CB531U) >> 27))
#endif
#ifndef LOG2_LOOKUP
#define LOG2_LOOKUP(x) ((x==0) ? 0 : (x==1) ? 1 : (x==2) ? 28 : (x==3) ? 2 : \
                   (x==4) ? 29 : (x==5) ? 14 : (x==6) ? 24 : (x==7) ? 3 : \
                   (x==8) ? 30 : (x==9) ? 22 : (x==10) ? 20 : (x==11) ? 15 : \
                   (x==12) ? 25 : (x==13) ? 17 : (x==14) ? 4 : (x==15) ? 8 : \
                   (x==16) ? 31 : (x==17) ? 27 : (x==18) ? 13 : (x==19) ? 23 : \
                   (x==20) ? 21 : (x==21) ? 19 : (x==22) ? 16 : (x==23) ? 7 : \
                   (x==24) ? 26 : (x==25) ? 12 : (x==26) ? 18 : (x==27) ? 6 : \
                   (x==28) ? 11 : (x==29) ? 5 : (x==30) ? 10 : 9)
#endif

  // Internal helper name space for bitmasks
    namespace BitMaskHelp {
      // A class for Bitmask objects to inherit from to have their dynamic
      // memory allocations managed for alignment and tracing
      template<typename T>
      class Heapify {
      public:
        static inline void* operator new(size_t count);
        static inline void* operator new[](size_t count);
      public:
        static inline void* operator new(size_t count, void *ptr);
        static inline void* operator new[](size_t count, void *ptr);
      public:
        static inline void operator delete(void *ptr);
        static inline void operator delete[](void *ptr);
      public:
        static inline void operator delete(void *ptr, void *place);
        static inline void operator delete[](void *ptr, void *place);
      };

#ifdef __SSE2__
      template<bool READ_ONLY, typename T = uint64_t>
      class SSEView {
      public:
        inline SSEView(T *base, unsigned index) 
          : ptr(base + ((sizeof(__m128d)/sizeof(T))*index)) { }
      public:
        inline operator __m128i(void) const {
          __m128i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __m128d(void) const {
          __m128d result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        inline void operator=(const __m128i &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const __m128d &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const SSEView<WHOCARES> &rhs) {
          memcpy(ptr, rhs.ptr, sizeof(__m128d));
        }
      public:
        T *const ptr;
      };
      template<typename T>
      class SSEView<true,T> {
      public:
        inline SSEView(const T *base, unsigned index) 
          : ptr(base + ((sizeof(__m128d)/sizeof(T))*index)) { }
      public:
        inline operator __m128i(void) const {
          __m128i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __m128d(void) const {
          __m128d result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        const T *const ptr;
      };
#endif
#ifdef __AVX__
      template<bool READ_ONLY, typename T = uint64_t>
      class AVXView {
      public:
        inline AVXView(T *base, unsigned index) 
          : ptr(base + ((sizeof(__m256d)/sizeof(T))*index)) { }
      public:
        inline operator __m256i(void) const {
          __m256i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __m256d(void) const {
          __m256d result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        inline void operator=(const __m256i &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const __m256d &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const AVXView<WHOCARES> &rhs) {
          memcpy(ptr, rhs.ptr, sizeof(__m256d));
        }
      public:
        T *const ptr;
      };
      template<typename T>
      class AVXView<true,T> {
      public:
        inline AVXView(const T *base, unsigned index) 
          : ptr(base + ((sizeof(__m256d)/sizeof(T))*index)) { }
      public:
        inline operator __m256i(void) const {
          __m256i result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __m256d(void) const {
          __m256d result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        const T *const ptr;
      };
#endif
#ifdef __ALTIVEC__
      template<bool READ_ONLY, typename T = uint64_t>
      class PPCView {
      public:
        inline PPCView(T *base, unsigned index) 
          : ptr(base + ((sizeof(__vector double)/sizeof(T))*index)) { }
      public:
        inline operator __vector unsigned long long(void) const {
          __vector unsigned long long result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __vector double(void) const {
          __vector double result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        inline void operator=(const __vector unsigned long long &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const __vector double &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const PPCView<WHOCARES> &rhs) {
          memcpy(ptr, rhs.ptr, sizeof(__vector double));
        }
      public:
        T *const ptr;
      };
      template<typename T>
      class PPCView<true,T> {
      public:
        inline PPCView(const T *base, unsigned index) 
          : ptr(base + ((sizeof(__vector double)/sizeof(T))*index)) { }
      public:
        inline operator __vector unsigned long long(void) const {
          __vector unsigned long long result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator __vector double(void) const {
          __vector double result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        const T *const ptr;
      };
#endif
#ifdef __ARM_NEON
      template<bool READ_ONLY, typename T = uint64_t>
      class NeonView {
      public:
        inline NeonView(T *base, unsigned index) 
          : ptr(base + ((sizeof(float32x4_t)/sizeof(T))*index)) { }
      public:
        inline operator uint32x4_t(void) const {
          uint32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator float32x4_t(void) const {
          float32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        inline void operator=(const uint32x4_t &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        inline void operator=(const float32x4_t &value) {
          memcpy(ptr, &value, sizeof(value));
        }
        template<bool WHOCARES>
        inline void operator=(const NeonView<WHOCARES> &rhs) {
          memcpy(ptr, rhs.ptr, sizeof(float32x4_t));
        }
      public:
        T *const ptr;
      };
      template<typename T>
      class NeonView<true,T> {
      public:
        inline NeonView(const T *base, unsigned index) 
          : ptr(base + ((sizeof(float32x4_t)/sizeof(T))*index)) { }
      public:
        inline operator uint32x4_t(void) const {
          uint32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        }
        inline operator float32x4_t(void) const {
          float32x4_t result;
          memcpy(&result, ptr, sizeof(result));
          return result;
        };
      public:
        const T *const ptr;
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
        inline SSEView<false,ELEMENT_TYPE> sse_view(unsigned index)
          { return SSEView<false,ELEMENT_TYPE>(bit_vector, index); }
        inline SSEView<true,ELEMENT_TYPE> sse_view(unsigned index) const
          { return SSEView<true,ELEMENT_TYPE>(bit_vector, index); }
#endif
#ifdef __AVX__
        inline AVXView<false,ELEMENT_TYPE> avx_view(unsigned index)
          { return AVXView<false,ELEMENT_TYPE>(bit_vector, index); }
        inline AVXView<true,ELEMENT_TYPE> avx_view(unsigned index) const
          { return AVXView<true,ELEMENT_TYPE>(bit_vector, index); }
#endif
#ifdef __ALTIVEC__
        inline PPCView<false,ELEMENT_TYPE> ppc_view(unsigned index)
          { return PPCView<false,ELEMENT_TYPE>(bit_vector, index); }
        inline PPCView<true,ELEMENT_TYPE> ppc_view(unsigned index) const
          { return PPCView<true,ELEMENT_TYPE>(bit_vector, index); }
#endif
#ifdef __ARM_NEON
        inline NeonView<false,ELEMENT_TYPE> neon_view(unsigned index)
          { return NeonView<false,ELEMENT_TYPE>(bit_vector, index); }
        inline NeonView<true,ELEMENT_TYPE> neon_view(unsigned index) const
          { return NeonView<true,ELEMENT_TYPE>(bit_vector, index); }
#endif
      public:
        // Number of bits in the bit vector based element
        static constexpr unsigned ELEMENT_SIZE = 8 * sizeof(ELEMENT_TYPE);
        static_assert((MAX % ELEMENT_SIZE) == 0, "Bad bitmask size");
        ELEMENT_TYPE bit_vector[MAX/ELEMENT_SIZE];
        // Shift to get the upper bits for indexing assuming a 64-bit base type
        static constexpr unsigned SHIFT = STATIC_LOG2(ELEMENT_SIZE);
        // Mask to get the lower bits for indexing assuming a 64-bit base type
        static constexpr unsigned MASK = ELEMENT_SIZE - 1;
      };
    };

    /////////////////////////////////////////////////////////////
    // Bit Mask 
    /////////////////////////////////////////////////////////////
    template<typename T, unsigned int MAX,
             unsigned int SHIFT, unsigned int MASK>
    class BitMask : public BitMaskHelp::Heapify<BitMask<T,MAX,SHIFT,MASK> > {
    public:
      static constexpr unsigned ELEMENT_SIZE = 8*sizeof(T);
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit BitMask(T init = 0);
      BitMask(const BitMask &rhs);
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
      inline bool operator==(const BitMask &rhs) const;
      inline bool operator<(const BitMask &rhs) const;
      inline bool operator!=(const BitMask &rhs) const;
    public:
      inline const T& operator[](const unsigned &idx) const;
      inline T& operator[](const unsigned &idx);
      inline BitMask& operator=(const BitMask &rhs);
    public:
      inline BitMask operator~(void) const; 
      inline BitMask operator|(const BitMask &rhs) const;
      inline BitMask operator&(const BitMask &rhs) const;
      inline BitMask operator^(const BitMask &rhs) const;
    public:
      inline BitMask& operator|=(const BitMask &rhs);
      inline BitMask& operator&=(const BitMask &rhs);
      inline BitMask& operator^=(const BitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const BitMask &rhs) const;
      // Set difference
      inline BitMask operator-(const BitMask &rhs) const;
      inline BitMask& operator-=(const BitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(
            const BitMask<unsigned,MAX,SHIFT,MASK> &mask);
      static inline unsigned pop_count(
            const BitMask<unsigned long,MAX,SHIFT,MASK> &mask);
      static inline unsigned pop_count(
            const BitMask<unsigned long long,MAX,SHIFT,MASK> &mask);
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
    class TLBitMask : 
      public BitMaskHelp::Heapify<TLBitMask<T,MAX,SHIFT,MASK> > {
    public:
      static constexpr unsigned ELEMENT_SIZE = 8*sizeof(T);
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit TLBitMask(T init = 0);
      TLBitMask(const TLBitMask &rhs);
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
      inline bool operator==(const TLBitMask &rhs) const;
      inline bool operator<(const TLBitMask &rhs) const;
      inline bool operator!=(const TLBitMask &rhs) const;
    public:
      inline const T& operator[](const unsigned &idx) const;
      inline T& operator[](const unsigned &idx);
      inline TLBitMask& operator=(const TLBitMask &rhs);
    public:
      inline TLBitMask operator~(void) const;
      inline TLBitMask operator|(const TLBitMask &rhs) const;
      inline TLBitMask operator&(const TLBitMask &rhs) const;
      inline TLBitMask operator^(const TLBitMask &rhs) const;
    public:
      inline TLBitMask& operator|=(const TLBitMask &rhs);
      inline TLBitMask& operator&=(const TLBitMask &rhs);
      inline TLBitMask& operator^=(const TLBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const TLBitMask &rhs) const;
      // Set difference
      inline TLBitMask operator-(const TLBitMask &rhs) const;
      inline TLBitMask& operator-=(const TLBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(
            const TLBitMask<unsigned,MAX,SHIFT,MASK> &mask);
      static inline unsigned pop_count(
            const TLBitMask<unsigned long,MAX,SHIFT,MASK> &mask);
      static inline unsigned pop_count(
            const TLBitMask<unsigned long long,MAX,SHIFT,MASK> &mask);
    protected:
      T bit_vector[BIT_ELMTS];
      T sum_mask; 
    };

#ifdef __SSE2__
    /////////////////////////////////////////////////////////////
    // SSE Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) SSEBitMask 
      : public BitMaskHelp::Heapify<SSEBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned SSE_ELMTS = MAX/128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit SSEBitMask(uint64_t init = 0);
      SSEBitMask(const SSEBitMask &rhs);
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
      inline bool operator==(const SSEBitMask &rhs) const;
      inline bool operator<(const SSEBitMask &rhs) const;
      inline bool operator!=(const SSEBitMask &rhs) const;
    public:
      inline BitMaskHelp::SSEView<true>
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::SSEView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline SSEBitMask& operator=(const SSEBitMask &rhs);
    public:
      inline SSEBitMask operator~(void) const;
      inline SSEBitMask operator|(const SSEBitMask &rhs) const;
      inline SSEBitMask operator&(const SSEBitMask &rhs) const;
      inline SSEBitMask operator^(const SSEBitMask &rhs) const;
    public:
      inline SSEBitMask& operator|=(const SSEBitMask &rhs);
      inline SSEBitMask& operator&=(const SSEBitMask &rhs);
      inline SSEBitMask& operator^=(const SSEBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const SSEBitMask &rhs) const;
      // Set difference
      inline SSEBitMask operator-(const SSEBitMask &rhs) const;
      inline SSEBitMask& operator-=(const SSEBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const SSEBitMask<MAX> &mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits; 
    };

    /////////////////////////////////////////////////////////////
    // SSE Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) SSETLBitMask
      : public BitMaskHelp::Heapify<SSETLBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned SSE_ELMTS = MAX/128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit SSETLBitMask(uint64_t init = 0);
      SSETLBitMask(const SSETLBitMask &rhs);
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
      inline bool operator==(const SSETLBitMask &rhs) const;
      inline bool operator<(const SSETLBitMask &rhs) const;
      inline bool operator!=(const SSETLBitMask &rhs) const;
    public:
      inline BitMaskHelp::SSEView<true>
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::SSEView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline SSETLBitMask& operator=(const SSETLBitMask &rhs);
    public:
      inline SSETLBitMask operator~(void) const;
      inline SSETLBitMask operator|(const SSETLBitMask &rhs) const;
      inline SSETLBitMask operator&(const SSETLBitMask &rhs) const;
      inline SSETLBitMask operator^(const SSETLBitMask &rhs) const;
    public:
      inline SSETLBitMask& operator|=(const SSETLBitMask &rhs);
      inline SSETLBitMask& operator&=(const SSETLBitMask &rhs);
      inline SSETLBitMask& operator^=(const SSETLBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const SSETLBitMask &rhs) const;
      // Set difference
      inline SSETLBitMask operator-(const SSETLBitMask &rhs) const;
      inline SSETLBitMask& operator-=(const SSETLBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const SSETLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__m128i value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask; 
    };
#endif // __SSE2__

#ifdef __AVX__
    /////////////////////////////////////////////////////////////
    // AVX Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(32) AVXBitMask 
      : public BitMaskHelp::Heapify<AVXBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned AVX_ELMTS = MAX/256;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit AVXBitMask(uint64_t init = 0);
      AVXBitMask(const AVXBitMask &rhs);
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
      inline bool operator==(const AVXBitMask &rhs) const;
      inline bool operator<(const AVXBitMask &rhs) const;
      inline bool operator!=(const AVXBitMask &rhs) const;
    public:
      inline BitMaskHelp::AVXView<true> 
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::AVXView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline AVXBitMask& operator=(const AVXBitMask &rhs);
    public:
      inline AVXBitMask operator~(void) const;
      inline AVXBitMask operator|(const AVXBitMask &rhs) const;
      inline AVXBitMask operator&(const AVXBitMask &rhs) const;
      inline AVXBitMask operator^(const AVXBitMask &rhs) const;
    public:
      inline AVXBitMask& operator|=(const AVXBitMask &rhs);
      inline AVXBitMask& operator&=(const AVXBitMask &rhs);
      inline AVXBitMask& operator^=(const AVXBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const AVXBitMask &rhs) const;
      // Set difference
      inline AVXBitMask operator-(const AVXBitMask &rhs) const;
      inline AVXBitMask& operator-=(const AVXBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const AVXBitMask<MAX> &mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits; 
    };
    
    /////////////////////////////////////////////////////////////
    // AVX Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(32) AVXTLBitMask
      : public BitMaskHelp::Heapify<AVXTLBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned AVX_ELMTS = MAX/256;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit AVXTLBitMask(uint64_t init = 0);
      AVXTLBitMask(const AVXTLBitMask &rhs);
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
      inline bool operator==(const AVXTLBitMask &rhs) const;
      inline bool operator<(const AVXTLBitMask &rhs) const;
      inline bool operator!=(const AVXTLBitMask &rhs) const;
    public:
      inline BitMaskHelp::AVXView<true> 
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::AVXView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline AVXTLBitMask& operator=(const AVXTLBitMask &rhs);
    public:
      inline AVXTLBitMask operator~(void) const;
      inline AVXTLBitMask operator|(const AVXTLBitMask &rhs) const;
      inline AVXTLBitMask operator&(const AVXTLBitMask &rhs) const;
      inline AVXTLBitMask operator^(const AVXTLBitMask &rhs) const;
    public:
      inline AVXTLBitMask& operator|=(const AVXTLBitMask &rhs);
      inline AVXTLBitMask& operator&=(const AVXTLBitMask &rhs);
      inline AVXTLBitMask& operator^=(const AVXTLBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const AVXTLBitMask &rhs) const;
      // Set difference
      inline AVXTLBitMask operator-(const AVXTLBitMask &rhs) const;
      inline AVXTLBitMask& operator-=(const AVXTLBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const AVXTLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__m256i value);
      static inline uint64_t extract_mask(__m256d value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask; 
    };
#endif // __AVX__

#ifdef __ALTIVEC__
    /////////////////////////////////////////////////////////////
    // PPC Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) PPCBitMask
      : public BitMaskHelp::Heapify<PPCBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned PPC_ELMTS = MAX/128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit PPCBitMask(uint64_t init = 0);
      PPCBitMask(const PPCBitMask &rhs);
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
      inline bool operator==(const PPCBitMask &rhs) const;
      inline bool operator<(const PPCBitMask &rhs) const;
      inline bool operator!=(const PPCBitMask &rhs) const;
    public:
      inline BitMaskHelp::PPCView<true> 
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::PPCView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline PPCBitMask& operator=(const PPCBitMask &rhs);
    public:
      inline PPCBitMask operator~(void) const;
      inline PPCBitMask operator|(const PPCBitMask &rhs) const;
      inline PPCBitMask operator&(const PPCBitMask &rhs) const;
      inline PPCBitMask operator^(const PPCBitMask &rhs) const;
    public:
      inline PPCBitMask& operator|=(const PPCBitMask &rhs);
      inline PPCBitMask& operator&=(const PPCBitMask &rhs);
      inline PPCBitMask& operator^=(const PPCBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const PPCBitMask &rhs) const;
      // Set difference
      inline PPCBitMask operator-(const PPCBitMask &rhs) const;
      inline PPCBitMask& operator-=(const PPCBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const PPCBitMask<MAX> &mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits; 
    };
    
    /////////////////////////////////////////////////////////////
    // PPC Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) PPCTLBitMask
      : public BitMaskHelp::Heapify<PPCTLBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned PPC_ELMTS = MAX/128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit PPCTLBitMask(uint64_t init = 0);
      PPCTLBitMask(const PPCTLBitMask &rhs);
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
      inline bool operator==(const PPCTLBitMask &rhs) const;
      inline bool operator<(const PPCTLBitMask &rhs) const;
      inline bool operator!=(const PPCTLBitMask &rhs) const;
    public:
      inline BitMaskHelp::PPCView<true> 
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::PPCView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline PPCTLBitMask& operator=(const PPCTLBitMask &rhs);
    public:
      inline PPCTLBitMask operator~(void) const;
      inline PPCTLBitMask operator|(const PPCTLBitMask &rhs) const;
      inline PPCTLBitMask operator&(const PPCTLBitMask &rhs) const;
      inline PPCTLBitMask operator^(const PPCTLBitMask &rhs) const;
    public:
      inline PPCTLBitMask& operator|=(const PPCTLBitMask &rhs);
      inline PPCTLBitMask& operator&=(const PPCTLBitMask &rhs);
      inline PPCTLBitMask& operator^=(const PPCTLBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const PPCTLBitMask &rhs) const;
      // Set difference
      inline PPCTLBitMask operator-(const PPCTLBitMask &rhs) const;
      inline PPCTLBitMask& operator-=(const PPCTLBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const PPCTLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__vector unsigned long long value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask; 
    };
#endif // __ALTIVEC__
       
#ifdef __ARM_NEON
    /////////////////////////////////////////////////////////////
    // Neon Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) NeonBitMask
      : public BitMaskHelp::Heapify<NeonBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned NEON_ELMTS = MAX/128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit NeonBitMask(uint64_t init = 0);
      NeonBitMask(const NeonBitMask &rhs);
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
      inline bool operator==(const NeonBitMask &rhs) const;
      inline bool operator<(const NeonBitMask &rhs) const;
      inline bool operator!=(const NeonBitMask &rhs) const;
    public:
      inline BitMaskHelp::NeonView<true> 
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::NeonView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline NeonBitMask& operator=(const NeonBitMask &rhs);
    public:
      inline NeonBitMask operator~(void) const;
      inline NeonBitMask operator|(const NeonBitMask &rhs) const;
      inline NeonBitMask operator&(const NeonBitMask &rhs) const;
      inline NeonBitMask operator^(const NeonBitMask &rhs) const;
    public:
      inline NeonBitMask& operator|=(const NeonBitMask &rhs);
      inline NeonBitMask& operator&=(const NeonBitMask &rhs);
      inline NeonBitMask& operator^=(const NeonBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const NeonBitMask &rhs) const;
      // Set difference
      inline NeonBitMask operator-(const NeonBitMask &rhs) const;
      inline NeonBitMask& operator-=(const NeonBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const NeonBitMask<MAX> &mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits; 
    };
    
    /////////////////////////////////////////////////////////////
    // Neon Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) NeonTLBitMask
      : public BitMaskHelp::Heapify<NeonTLBitMask<MAX> > {
    public:
      static constexpr unsigned ELEMENT_SIZE =
        BitMaskHelp::BitVector<MAX>::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = MAX/ELEMENT_SIZE;
      static constexpr unsigned NEON_ELMTS = MAX/128;
      static constexpr unsigned MAXSIZE = MAX;
    public:
      explicit NeonTLBitMask(uint64_t init = 0);
      NeonTLBitMask(const NeonTLBitMask &rhs);
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
      inline bool operator==(const NeonTLBitMask &rhs) const;
      inline bool operator<(const NeonTLBitMask &rhs) const;
      inline bool operator!=(const NeonTLBitMask &rhs) const;
    public:
      inline BitMaskHelp::NeonView<true> 
        operator()(const unsigned &idx) const;
      inline BitMaskHelp::NeonView<false>
        operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline NeonTLBitMask& operator=(const NeonTLBitMask &rhs);
    public:
      inline NeonTLBitMask operator~(void) const;
      inline NeonTLBitMask operator|(const NeonTLBitMask &rhs) const;
      inline NeonTLBitMask operator&(const NeonTLBitMask &rhs) const;
      inline NeonTLBitMask operator^(const NeonTLBitMask &rhs) const;
    public:
      inline NeonTLBitMask& operator|=(const NeonTLBitMask &rhs);
      inline NeonTLBitMask& operator&=(const NeonTLBitMask &rhs);
      inline NeonTLBitMask& operator^=(const NeonTLBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const NeonTLBitMask &rhs) const;
      // Set difference
      inline NeonTLBitMask operator-(const NeonTLBitMask &rhs) const;
      inline NeonTLBitMask& operator-=(const NeonTLBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DT>
      inline void deserialize(DT &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const NeonTLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(uint32x4_t value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask; 
    };
#endif // __ARM_NEON

    template<typename DT, unsigned BLOAT=1, bool BIDIR=true>
    class CompoundBitMask {
    public:
      static constexpr unsigned ELEMENT_SIZE = DT::ELEMENT_SIZE;
      static constexpr unsigned BIT_ELMTS = DT::BIT_ELMTS; 
      static constexpr unsigned MAXSIZE = DT::MAXSIZE;
    public:
      explicit CompoundBitMask(uint64_t init = 0);
      CompoundBitMask(const CompoundBitMask &rhs);
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
      inline bool operator==(const CompoundBitMask &rhs) const;
      inline bool operator<(const CompoundBitMask &rhs) const;
      inline bool operator!=(const CompoundBitMask &rhs) const;
    public:
      inline CompoundBitMask& operator=(const CompoundBitMask &rhs);
    public:
      inline CompoundBitMask operator~(void) const;
      inline CompoundBitMask operator|(const CompoundBitMask &rhs) const;
      inline CompoundBitMask operator&(const CompoundBitMask &rhs) const;
      inline CompoundBitMask operator^(const CompoundBitMask &rhs) const;
    public:
      inline CompoundBitMask& operator|=(const CompoundBitMask &rhs);
      inline CompoundBitMask& operator&=(const CompoundBitMask &rhs);
      inline CompoundBitMask& operator^=(const CompoundBitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const CompoundBitMask &rhs) const;
      // Set difference
      inline CompoundBitMask operator-(const CompoundBitMask &rhs) const;
      inline CompoundBitMask& operator-=(const CompoundBitMask &rhs);
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
      inline void serialize(ST &rez) const;
      template<typename DZ>
      inline void deserialize(DZ &derez);
      // The functor class must have an 'apply' method that
      // takes one unsigned argument. This method will map
      // the functor over all the entries in the mask.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline unsigned pop_count(void) const;
      static inline unsigned pop_count(const 
                        CompoundBitMask<DT,BLOAT,BIDIR> &mask);
    protected:
      inline bool is_sparse(void) const;
      inline void sparsify(void);
    protected:
      using IT = typename std::conditional<DT::MAXSIZE <= (1 << 8),uint8_t,
                  typename std::conditional<DT::MAXSIZE <= (1 << 16),uint16_t,
                    uint32_t>::type>::type;
      static constexpr size_t MAX_SPARSE =
        (BLOAT * sizeof(DT*) + sizeof(IT) - 1) / sizeof(IT);
      using SA = std::array<IT,MAX_SPARSE>;
      union {
        // The sparse array is unique and sorted
        SA sparse;
        DT *dense;
      } mask;
      unsigned sparse_size; 
    };

  namespace BitMaskHelp {

    //--------------------------------------------------------------------------
    inline char* to_string(const uint64_t *bits, int count)
    //--------------------------------------------------------------------------
    {
      int length = ((count + 3) >> 2) + 1; // includes trailing \0
      char *result = (char*)malloc(length * sizeof(char));
#ifdef DEBUG_LEGION
      assert(result != 0);
#endif
      int index = 0;
      int words = (count + 63) >> 6;
      for (int w = 0; w < words; w++)
      {
        uint64_t word = bits[w];
        for (int n = 0; n < 16; n++)
        {
          int nibble = word & 0xF;
          if (nibble < 10)
            result[index++] = '0' + nibble;
          else
            result[index++] = 'A' + (nibble-10);
          if ((index * 4) >= count)
            break;
          word >>= 4;
        }
      }
#ifdef DEBUG_LEGION
      assert(index == (length-1));
#endif
      result[index] = '\0';
      return result;
    }

    /**
     * A helper class for determining alignment of types
     */
    template<typename T>
    class AlignmentTrait {
    public:
      struct AlignmentFinder {
        char a;
        T b;
      };
      enum { AlignmentOf = sizeof(AlignmentFinder) - sizeof(T) };
    };

    //--------------------------------------------------------------------------
    template<size_t SIZE, size_t ALIGNMENT, bool BYTES>
    inline void* alloc_aligned(size_t cnt)
    //--------------------------------------------------------------------------
    {
      static_assert((SIZE % ALIGNMENT) == 0, "Bad size");
      size_t alloc_size = cnt;
      if (!BYTES)
        alloc_size *= SIZE;
      void *result = NULL;
      if (ALIGNMENT > BITMASK_MAX_ALIGNMENT)
      {
#if defined(DEBUG_LEGION) || defined(DEBUG_REALM)
        assert((alloc_size % ALIGNMENT) == 0);
#endif
#if (defined(DEBUG_LEGION) || defined(DEBUG_REALM)) && !defined(NDEBUG)
        int error = 
#else
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#endif
#endif
          posix_memalign(&result, ALIGNMENT, alloc_size);
#if (defined(DEBUG_LEGION) || defined(DEBUG_REALM)) && !defined(NDEBUG)
        assert(error == 0);
#else
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
      }
      else
        result = malloc(alloc_size);

#if defined(DEBUG_LEGION) || defined(DEBUG_REALM)
      assert(result != NULL);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool BYTES>
    inline void* alloc_aligned(size_t cnt)
    //--------------------------------------------------------------------------
    {
      return alloc_aligned<sizeof(T),
              AlignmentTrait<T>::AlignmentOf,BYTES>(cnt);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* Heapify<T>::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return alloc_aligned<T,true/*bytes*/>(count);  
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* Heapify<T>::operator new[](size_t count)
    //--------------------------------------------------------------------------
    {
      return alloc_aligned<T,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* Heapify<T>::operator new(size_t count, void *ptr)
    //--------------------------------------------------------------------------
    {
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* Heapify<T>::operator new[](size_t count, void *ptr)
    //--------------------------------------------------------------------------
    {
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void Heapify<T>::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void Heapify<T>::operator delete[](void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void Heapify<T>::operator delete(void *ptr, void *place)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void Heapify<T>::operator delete[](void *ptr, void *place)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

  };

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    BitMask<T,MAX,SHIFT,MASK>::BitMask(T init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % (8*sizeof(T))) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    BitMask<T,MAX,SHIFT,MASK>::BitMask(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % (8*sizeof(T))) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    BitMask<T,MAX,SHIFT,MASK>::~BitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK> 
    inline void BitMask<T,MAX,SHIFT,MASK>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> SHIFT;
      bit_vector[idx] |= (1ULL << (bit & MASK));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void BitMask<T,MAX,SHIFT,MASK>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> SHIFT;
      bit_vector[idx] &= ~((1ULL << (bit & MASK)));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void BitMask<T,MAX,SHIFT,MASK>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK> 
    inline bool BitMask<T,MAX,SHIFT,MASK>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> SHIFT;
      return (bit_vector[idx] & (1ULL << (bit & MASK)));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int BitMask<T,MAX,SHIFT,MASK>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx])
        {
          for (unsigned j = 0; j < 8*sizeof(T); j++)
          {
            if (bit_vector[idx] & (1ULL << j))
            {
              return (idx*8*sizeof(T) + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int BitMask<T,MAX,SHIFT,MASK>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> SHIFT;
      unsigned offset = bit & MASK;
      if (bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bit_vector[element] << (ELEMENT_SIZE - offset)); 
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int BitMask<T,MAX,SHIFT,MASK>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int BitMask<T,MAX,SHIFT,MASK>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bit_vector[idx] > 0) // if it has any valid entries, find the next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline void BitMask<T,MAX,SHIFT,MASK>::clear(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = 0;
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline const T& BitMask<T,MAX,SHIFT,MASK>::operator[](
                                                    const unsigned &idx) const
    //-------------------------------------------------------------------------
    {
      return bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline T& BitMask<T,MAX,SHIFT,MASK>::operator[](const unsigned &idx) 
    //-------------------------------------------------------------------------
    {
      return bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool BitMask<T,MAX,SHIFT,MASK>::operator==(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] != rhs[idx]) 
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool BitMask<T,MAX,SHIFT,MASK>::operator<(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] < rhs[idx])
          return true;
        else if (bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool BitMask<T,MAX,SHIFT,MASK>::operator!=(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>& 
                      BitMask<T,MAX,SHIFT,MASK>::operator=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK> 
                              BitMask<T,MAX,SHIFT,MASK>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX,SHIFT,MASK> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = ~bit_vector[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK> 
                BitMask<T,MAX,SHIFT,MASK>::operator|(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX,SHIFT,MASK> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] | rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK> 
                BitMask<T,MAX,SHIFT,MASK>::operator&(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX,SHIFT,MASK> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] & rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK> 
                BitMask<T,MAX,SHIFT,MASK>::operator^(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX,SHIFT,MASK> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] ^ rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>& 
                      BitMask<T,MAX,SHIFT,MASK>::operator|=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] |= rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>& 
                      BitMask<T,MAX,SHIFT,MASK>::operator&=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] &= rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>& 
                      BitMask<T,MAX,SHIFT,MASK>::operator^=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] ^= rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool BitMask<T,MAX,SHIFT,MASK>::operator*(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] & rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK> 
                BitMask<T,MAX,SHIFT,MASK>::operator-(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX,SHIFT,MASK> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] & ~(rhs[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>& 
                      BitMask<T,MAX,SHIFT,MASK>::operator-=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] &= ~(rhs[idx]);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool BitMask<T,MAX,SHIFT,MASK>::empty(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] != 0)
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool BitMask<T,MAX,SHIFT,MASK>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>
                    BitMask<T,MAX,SHIFT,MASK>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      BitMask<T,MAX,SHIFT,MASK> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          T left = bit_vector[idx-range] << local;
          T right = bit_vector[idx-(range+1)] >> ((1 << SHIFT) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[range] = bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>
                    BitMask<T,MAX,SHIFT,MASK>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      BitMask<T,MAX,SHIFT,MASK> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          T right = bit_vector[idx+range] >> local;
          T left = bit_vector[idx+range+1] << ((1 << SHIFT) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>&
                        BitMask<T,MAX,SHIFT,MASK>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bit_vector[idx] = bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          T left = bit_vector[idx-range] << local;
          T right = bit_vector[idx-(range+1)] >> ((1 << SHIFT) - local);
          bit_vector[idx] = left | right;
        }
        // Handle the last case
        bit_vector[range] = bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline BitMask<T,MAX,SHIFT,MASK>&
                        BitMask<T,MAX,SHIFT,MASK>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bit_vector[idx] = bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          T right = bit_vector[idx+range] >> local;
          T left = bit_vector[idx+range+1] << ((1 << SHIFT) - local);
          bit_vector[idx] = left | right;
        }
        // Handle the last case
        bit_vector[BIT_ELMTS-(range+1)] = bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline T BitMask<T,MAX,SHIFT,MASK>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      T result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result |= bit_vector[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK> 
      template<typename ST>
    inline void BitMask<T,MAX,SHIFT,MASK>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bit_vector, (MAX/8));
    } 

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK> 
      template<typename DT>
    inline void BitMask<T,MAX,SHIFT,MASK>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK> 
      template<typename FUNCTOR>
    inline void BitMask<T,MAX,SHIFT,MASK>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline char* BitMask<T,MAX,SHIFT,MASK>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline unsigned BitMask<T,MAX,SHIFT,MASK>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        result += __builtin_popcountll(bit_vector[idx]);
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline unsigned BitMask<T,MAX,SHIFT,MASK>::pop_count(
                                  const BitMask<unsigned,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcount(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline unsigned BitMask<T,MAX,SHIFT,MASK>::pop_count(
                            const BitMask<unsigned long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline unsigned BitMask<T,MAX,SHIFT,MASK>::pop_count(
                        const BitMask<unsigned long long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    TLBitMask<T,MAX,SHIFT,MASK>::TLBitMask(T init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % (8*sizeof(T))) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        bit_vector[idx] = init;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    TLBitMask<T,MAX,SHIFT,MASK>::TLBitMask(const TLBitMask &rhs)
      : sum_mask(rhs.sum_mask)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % (8*sizeof(T))) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    TLBitMask<T,MAX,SHIFT,MASK>::~TLBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> SHIFT;
      const T set_mask = (1ULL << (bit & MASK));
      bit_vector[idx] |= set_mask;
      sum_mask |= set_mask;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> SHIFT;
      const T set_mask = (1ULL << (bit & MASK));
      const T unset_mask = ~set_mask;
      bit_vector[idx] &= unset_mask;
      // Unset the summary mask and then reset if necessary
      sum_mask &= unset_mask;
      for (unsigned i = 0; i < BIT_ELMTS; i++)
        sum_mask |= bit_vector[i];
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::assign_bit(unsigned b, bool v)
    //-------------------------------------------------------------------------
    {
      if (v)
        set_bit(b);
      else
        unset_bit(b);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> SHIFT;
      return (bit_vector[idx] & (1ULL << (bit & MASK)));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int TLBitMask<T,MAX,SHIFT,MASK>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx])
        {
          for (unsigned j = 0; j < 8*sizeof(T); j++)
          {
            if (bit_vector[idx] & (1ULL << j))
            {
              return (idx*8*sizeof(T) + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int TLBitMask<T,MAX,SHIFT,MASK>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> SHIFT;
      unsigned offset = bit & MASK;
      if (bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bit_vector[element] << (ELEMENT_SIZE - offset)); 
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int TLBitMask<T,MAX,SHIFT,MASK>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int TLBitMask<T,MAX,SHIFT,MASK>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bit_vector[idx] > 0) // if it has any valid entries, find the next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::clear(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = 0;
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline const T& TLBitMask<T,MAX,SHIFT,MASK>::operator[](
                                                    const unsigned &idx) const
    //-------------------------------------------------------------------------
    {
      return bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline T& TLBitMask<T,MAX,SHIFT,MASK>::operator[](const unsigned &idx)
    //-------------------------------------------------------------------------
    {
      return bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::operator==(
                                                    const TLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask != rhs.sum_mask)
        return false;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] != rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::operator<(const TLBitMask &rhs)
                                                                        const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subets of the rhs bits 
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] < rhs[idx])
          return true;
        else if (bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::operator!=(
                                                    const TLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>&
                  TLBitMask<T,MAX,SHIFT,MASK>::operator=(const TLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = rhs.sum_mask;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>
                            TLBitMask<T,MAX,SHIFT,MASK>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      TLBitMask<T,MAX,SHIFT,MASK> result; 
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = ~bit_vector[idx];
        result.sum_mask |= result[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>
            TLBitMask<T,MAX,SHIFT,MASK>::operator|(const TLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      TLBitMask<T,MAX,SHIFT,MASK> result; 
      result.sum_mask = sum_mask | rhs.sum_mask;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] | rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>
          TLBitMask<T,MAX,SHIFT,MASK>::operator&(const TLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      TLBitMask<T,MAX,SHIFT,MASK> result; 
      // If they are independent then we are done
      if (sum_mask & rhs.sum_mask)
      {
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        {
          result[idx] = bit_vector[idx] & rhs[idx];
          result.sum_mask |= result[idx];
        }
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>
            TLBitMask<T,MAX,SHIFT,MASK>::operator^(const TLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      TLBitMask<T,MAX,SHIFT,MASK> result; 
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] ^ rhs[idx];
        result.sum_mask |= result[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>&
                 TLBitMask<T,MAX,SHIFT,MASK>::operator|=(const TLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask |= rhs.sum_mask;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] |= rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>&
                 TLBitMask<T,MAX,SHIFT,MASK>::operator&=(const TLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        sum_mask = 0;
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        {
          bit_vector[idx] &= rhs[idx];
          sum_mask |= bit_vector[idx];
        }
      }
      else
      {
        sum_mask = 0;
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
          bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>&
                 TLBitMask<T,MAX,SHIFT,MASK>::operator^=(const TLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] ^= rhs[idx];
        sum_mask |= bit_vector[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::operator*(const TLBitMask &rhs)
                                                                          const
    //-------------------------------------------------------------------------
    {
      // This is the whole reason we have sum mask right here
      if (sum_mask & rhs.sum_mask)
      {
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        {
          if (bit_vector[idx] & rhs[idx])
            return false;
        }
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>
            TLBitMask<T,MAX,SHIFT,MASK>::operator-(const TLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      TLBitMask<T,MAX,SHIFT,MASK> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] & ~(rhs[idx]);
        result.sum_mask |= result[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>&
                 TLBitMask<T,MAX,SHIFT,MASK>::operator-=(const TLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] &= ~(rhs[idx]);
        sum_mask |= bit_vector[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::empty(void) const
    //-------------------------------------------------------------------------
    {
      // Here is another great reason to have sum mask
      return (sum_mask == 0);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>
                 TLBitMask<T,MAX,SHIFT,MASK>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      TLBitMask<T,MAX,SHIFT,MASK> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bit_vector[idx-range]; 
          result.sum_mask |= result[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          T left = bit_vector[idx-range] << local;
          T right = bit_vector[idx-(range+1)] >> ((1 << SHIFT) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[range] = bit_vector[0] << local; 
        result.sum_mask |= result[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>
                 TLBitMask<T,MAX,SHIFT,MASK>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      TLBitMask<T,MAX,SHIFT,MASK> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bit_vector[idx+range];
          result.sum_mask |= result[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          T right = bit_vector[idx+range] >> local;
          T left = bit_vector[idx+range+1] << ((1 << SHIFT) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bit_vector[BIT_ELMTS-1] >> local;
        result.sum_mask |= result[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>&
                      TLBitMask<T,MAX,SHIFT,MASK>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bit_vector[idx] = bit_vector[idx-range]; 
          sum_mask |= bit_vector[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          T left = bit_vector[idx-range] << local;
          T right = bit_vector[idx-(range+1)] >> ((1 << SHIFT) - local);
          bit_vector[idx] = left | right;
          sum_mask |= bit_vector[idx];
        }
        // Handle the last case
        bit_vector[range] = bit_vector[0] << local; 
        sum_mask |= bit_vector[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline TLBitMask<T,MAX,SHIFT,MASK>&
                      TLBitMask<T,MAX,SHIFT,MASK>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> SHIFT;
      unsigned local = shift & MASK;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bit_vector[idx] = bit_vector[idx+range];
          sum_mask |= bit_vector[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          T right = bit_vector[idx+range] >> local;
          T left = bit_vector[idx+range+1] << ((1 << SHIFT) - local);
          bit_vector[idx] = left | right;
          sum_mask |= bit_vector[idx];
        }
        // Handle the last case
        bit_vector[BIT_ELMTS-(range+1)] = bit_vector[BIT_ELMTS-1] >> local;
        sum_mask |= bit_vector[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline T TLBitMask<T,MAX,SHIFT,MASK>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      return sum_mask;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      template<typename ST>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.template serialize(sum_mask);
      rez.serialize(bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      template<typename DT>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.template deserialize(sum_mask);
      derez.deserialize(bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK> 
      template<typename FUNCTOR>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline char* TLBitMask<T,MAX,SHIFT,MASK>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline unsigned TLBitMask<T,MAX,SHIFT,MASK>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline unsigned TLBitMask<T,MAX,SHIFT,MASK>::pop_count(
                               const TLBitMask<unsigned,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      if (!mask.sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcount(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline unsigned TLBitMask<T,MAX,SHIFT,MASK>::pop_count(
                          const TLBitMask<unsigned long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      if (!mask.sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline unsigned TLBitMask<T,MAX,SHIFT,MASK>::pop_count(
                     const TLBitMask<unsigned long long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      if (!mask.sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

#ifdef __SSE2__
    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    SSEBitMask<MAX>::SSEBitMask(uint64_t init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    SSEBitMask<MAX>::SSEBitMask(const SSEBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    SSEBitMask<MAX>::~SSEBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSEBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] |= (1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSEBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] &= ~(1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSEBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSEBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSEBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSEBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSEBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSEBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSEBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_set1_epi32(0);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::SSEView<true>
                     SSEBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.sse_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::SSEView<false>
                           SSEBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.sse_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& SSEBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& SSEBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSEBitMask<MAX>::operator==(const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != rhs[idx]) 
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSEBitMask<MAX>::operator<(const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] < rhs[idx])
          return true;
        else if (bits.bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSEBitMask<MAX>::operator!=(const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX>& SSEBitMask<MAX>::operator=(const SSEBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX> SSEBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      SSEBitMask<MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = ~(bits.bit_vector[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX> SSEBitMask<MAX>::operator|(
                                                   const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSEBitMask<MAX> result;
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        result(idx) = _mm_or_si128(bits.sse_view(idx), rhs(idx));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX> SSEBitMask<MAX>::operator&(
                                                   const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSEBitMask<MAX> result;
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        result(idx) = _mm_and_si128(bits.sse_view(idx), rhs(idx));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX> SSEBitMask<MAX>::operator^(
                                                   const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSEBitMask<MAX> result;
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        result(idx) = _mm_xor_si128(bits.sse_view(idx), rhs(idx));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX>& SSEBitMask<MAX>::operator|=(const SSEBitMask &rhs) 
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_or_si128(bits.sse_view(idx), rhs(idx));
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX>& SSEBitMask<MAX>::operator&=(const SSEBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_and_si128(bits.sse_view(idx), rhs(idx));
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX>& SSEBitMask<MAX>::operator^=(const SSEBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_xor_si128(bits.sse_view(idx), rhs(idx));
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSEBitMask<MAX>::operator*(const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] & rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX> SSEBitMask<MAX>::operator-(
                                                   const SSEBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSEBitMask<MAX> result;
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        result(idx) = _mm_andnot_si128(rhs(idx), bits.sse_view(idx));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX>& SSEBitMask<MAX>::operator-=(const SSEBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_andnot_si128(rhs(idx), bits.sse_view(idx));
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSEBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != 0)
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSEBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX> SSEBitMask<MAX>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      SSEBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX> SSEBitMask<MAX>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      SSEBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX>& SSEBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSEBitMask<MAX>& SSEBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                      bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t SSEBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      uint64_t result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result |= bits.bit_vector[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* SSEBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void SSEBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void SSEBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void SSEBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* SSEBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned SSEBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned SSEBitMask<MAX>::pop_count(
                                                   const SSEBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    SSETLBitMask<MAX>::SSETLBitMask(uint64_t init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    SSETLBitMask<MAX>::SSETLBitMask(const SSETLBitMask &rhs)
      : sum_mask(rhs.sum_mask)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    SSETLBitMask<MAX>::~SSETLBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSETLBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      bits.bit_vector[idx] |= set_mask;
      sum_mask |= set_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSETLBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      const uint64_t unset_mask = ~set_mask;
      bits.bit_vector[idx] &= unset_mask;
      // Unset the summary mask and then reset if necessary
      sum_mask &= unset_mask;
      for (unsigned i = 0; i < BIT_ELMTS; i++)
        sum_mask |= bits.bit_vector[i];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSETLBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSETLBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSETLBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSETLBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSETLBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSETLBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSETLBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_set1_epi32(0); 
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::SSEView<true>
                   SSETLBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.sse_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::SSEView<false>
                         SSETLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.sse_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& SSETLBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& SSETLBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSETLBitMask<MAX>::operator==(const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask != rhs.sum_mask)
        return false;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != rhs[idx]) 
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSETLBitMask<MAX>::operator<(const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] < rhs[idx])
          return true;
        else if (bits.bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSETLBitMask<MAX>::operator!=(const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX>& SSETLBitMask<MAX>::operator=(
                                                       const SSETLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = rhs.sum_mask;
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX> SSETLBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      SSETLBitMask<MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = ~(bits.bit_vector[idx]);
        result.sum_mask |= result[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX> SSETLBitMask<MAX>::operator|(
                                                 const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSETLBitMask<MAX> result;
      result.sum_mask = sum_mask | rhs.sum_mask;
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        result(idx) = _mm_or_si128(bits.sse_view(idx), rhs(idx));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX> SSETLBitMask<MAX>::operator&(
                                                 const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSETLBitMask<MAX> result;
      // If they are independent then we are done
      if (sum_mask & rhs.sum_mask)
      {
        __m128i temp_sum = _mm_set1_epi32(0);
        for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
        {
          result(idx) = _mm_and_si128(bits.sse_view(idx), rhs(idx));
          temp_sum = _mm_or_si128(temp_sum, result(idx));
        }
        result.sum_mask = extract_mask(temp_sum); 
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX> SSETLBitMask<MAX>::operator^(
                                                 const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSETLBitMask<MAX> result;
      __m128i temp_sum = _mm_set1_epi32(0);
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        result(idx) = _mm_xor_si128(bits.sse_view(idx), rhs(idx));
        temp_sum = _mm_or_si128(temp_sum, result(idx));
      }
      result.sum_mask = extract_mask(temp_sum);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX>& SSETLBitMask<MAX>::operator|=(
                                                       const SSETLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask |= rhs.sum_mask;
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_or_si128(bits.sse_view(idx), rhs(idx));
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX>& SSETLBitMask<MAX>::operator&=(
                                                       const SSETLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        __m128i temp_sum = _mm_set1_epi32(0);
        for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
        {
          bits.sse_view(idx) = _mm_and_si128(bits.sse_view(idx), rhs(idx));
          temp_sum = _mm_or_si128(temp_sum, bits.sse_view(idx));
        }
        sum_mask = extract_mask(temp_sum); 
      }
      else
      {
        sum_mask = 0;
        for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
          bits.sse_view(idx) = _mm_set1_epi32(0);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX>& SSETLBitMask<MAX>::operator^=(
                                                       const SSETLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      __m128i temp_sum = _mm_set1_epi32(0);
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_xor_si128(bits.sse_view(idx), rhs(idx));
        temp_sum = _mm_or_si128(temp_sum, bits.sse_view(idx));
      }
      sum_mask = extract_mask(temp_sum);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSETLBitMask<MAX>::operator*(const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        {
          if (bits.bit_vector[idx] & rhs[idx])
            return false;
        }
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX> SSETLBitMask<MAX>::operator-(
                                                 const SSETLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      SSETLBitMask<MAX> result;
      __m128i temp_sum = _mm_set1_epi32(0);
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        result(idx) = _mm_andnot_si128(rhs(idx), bits.sse_view(idx));
        temp_sum = _mm_or_si128(temp_sum, result(idx));
      }
      result.sum_mask = extract_mask(temp_sum);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX>& SSETLBitMask<MAX>::operator-=(
                                                       const SSETLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      __m128i temp_sum = _mm_set1_epi32(0);
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_view(idx) = _mm_andnot_si128(rhs(idx), bits.sse_view(idx));
        temp_sum = _mm_or_si128(temp_sum, bits.sse_view(idx));
      }
      sum_mask = extract_mask(temp_sum);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSETLBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      // A great reason to have a summary mask
      return (sum_mask == 0);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool SSETLBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX> SSETLBitMask<MAX>::operator<<(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      SSETLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
          result.sum_mask |= result[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        result.sum_mask |= result[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX> SSETLBitMask<MAX>::operator>>(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      SSETLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
          result.sum_mask |= result[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        result.sum_mask |= result[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX>& SSETLBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
          sum_mask |= bits.bit_vector[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        sum_mask |= bits.bit_vector[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline SSETLBitMask<MAX>& SSETLBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
          sum_mask |= bits.bit_vector[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                        bits.bit_vector[BIT_ELMTS-1] >> local;
        sum_mask |= bits.bit_vector[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t SSETLBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      return sum_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* SSETLBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void SSETLBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sum_mask);
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void SSETLBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(sum_mask);
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void SSETLBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* SSETLBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned SSETLBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned SSETLBitMask<MAX>::pop_count(
                                                 const SSETLBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline uint64_t SSETLBitMask<MAX>::extract_mask(__m128i value)
    //-------------------------------------------------------------------------
    {
#if !defined(__LP64__) // handle the case for when we don't have 64-bit support
      uint64_t left = _mm_cvtsi128_si32(value);
      left |= uint64_t(_mm_cvtsi128_si32(_mm_shuffle_epi32(value, 1))) << 32;
      uint64_t right = _mm_cvtsi128_si32(_mm_shuffle_epi32(value, 2));
      right |= uint64_t(_mm_cvtsi128_si32(_mm_shuffle_epi32(value, 3))) << 32;
#elif defined(__SSE4_1__) // see if we have sse 4.1
      uint64_t left = _mm_extract_epi64(value, 0);
      uint64_t right = _mm_extract_epi64(value, 1);
#else // Assume we have sse 2
      uint64_t left = _mm_cvtsi128_si64(value);
      uint64_t right = _mm_cvtsi128_si64(_mm_shuffle_epi32(value, 14));
#endif
      return (left | right);
    }
#endif // __SSE2__

#ifdef __AVX__
    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    AVXBitMask<MAX>::AVXBitMask(uint64_t init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 256) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    AVXBitMask<MAX>::AVXBitMask(const AVXBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 256) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    AVXBitMask<MAX>::~AVXBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] |= (1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] &= ~(1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_set1_epi32(0);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::AVXView<true>
                     AVXBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.avx_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::AVXView<false>
                           AVXBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.avx_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& AVXBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& AVXBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXBitMask<MAX>::operator==(const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != rhs[idx]) 
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXBitMask<MAX>::operator<(const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] < rhs[idx])
          return true;
        else if (bits.bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXBitMask<MAX>::operator!=(const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX>& AVXBitMask<MAX>::operator=(const AVXBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX> AVXBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      AVXBitMask<MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = ~(bits.bit_vector[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX> AVXBitMask<MAX>::operator|(
                                                   const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXBitMask<MAX> result;
#ifdef __AVX2__
      // If we have this instruction use it because it has higher throughput
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_or_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_or_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX> AVXBitMask<MAX>::operator&(
                                                   const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXBitMask<MAX> result;
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_and_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_and_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX> AVXBitMask<MAX>::operator^(
                                                   const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXBitMask<MAX> result;
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_xor_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_xor_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX>& AVXBitMask<MAX>::operator|=(const AVXBitMask &rhs) 
    //-------------------------------------------------------------------------
    {
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_or_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_or_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX>& AVXBitMask<MAX>::operator&=(const AVXBitMask &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_and_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_and_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX>& AVXBitMask<MAX>::operator^=(const AVXBitMask &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_xor_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_xor_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXBitMask<MAX>::operator*(const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] & rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX> AVXBitMask<MAX>::operator-(
                                                   const AVXBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXBitMask<MAX> result;
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_andnot_si256(rhs(idx), bits.avx_view(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_andnot_pd(rhs(idx),
                                            bits.avx_view(idx));
      }
#endif
      _mm256_zeroall();
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX>& AVXBitMask<MAX>::operator-=(const AVXBitMask &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_andnot_si256(rhs(idx), bits.avx_view(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_andnot_pd(rhs(idx),
                                                bits.avx_view(idx));
      }
#endif
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != 0)
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX> AVXBitMask<MAX>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      AVXBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX> AVXBitMask<MAX>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      AVXBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX>& AVXBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXBitMask<MAX>& AVXBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                      bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t AVXBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      uint64_t result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result |= bits.bit_vector[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* AVXBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void AVXBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void AVXBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void AVXBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* AVXBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned AVXBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned AVXBitMask<MAX>::pop_count(
                                                   const AVXBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    AVXTLBitMask<MAX>::AVXTLBitMask(uint64_t init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 256) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    AVXTLBitMask<MAX>::AVXTLBitMask(const AVXTLBitMask &rhs)
      : sum_mask(rhs.sum_mask)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 256) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    AVXTLBitMask<MAX>::~AVXTLBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXTLBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      bits.bit_vector[idx] |= set_mask;
      sum_mask |= set_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXTLBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      const uint64_t unset_mask = ~set_mask;
      bits.bit_vector[idx] &= unset_mask;
      // Unset the summary mask and then reset if necessary
      sum_mask &= unset_mask;
      for (unsigned i = 0; i < BIT_ELMTS; i++)
        sum_mask |= bits.bit_vector[i];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXTLBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXTLBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXTLBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE+ j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXTLBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXTLBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXTLBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXTLBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_set1_epi32(0);
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::AVXView<true>
                   AVXTLBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.avx_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::AVXView<false>
                         AVXTLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.avx_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& AVXTLBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& AVXTLBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXTLBitMask<MAX>::operator==(const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask != rhs.sum_mask)
        return false;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != rhs[idx]) 
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXTLBitMask<MAX>::operator<(const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] < rhs[idx])
          return true;
        else if (bits.bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXTLBitMask<MAX>::operator!=(const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX>& AVXTLBitMask<MAX>::operator=(
                                                       const AVXTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = rhs.sum_mask;
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX> AVXTLBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      AVXTLBitMask<MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = ~(bits.bit_vector[idx]);
        result.sum_mask |= result[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX> AVXTLBitMask<MAX>::operator|(
                                                 const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXTLBitMask<MAX> result;
      result.sum_mask = sum_mask | rhs.sum_mask;
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_or_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_or_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX> AVXTLBitMask<MAX>::operator&(
                                                 const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXTLBitMask<MAX> result;
      // If they are independent then we are done
      if (sum_mask & rhs.sum_mask)
      {
#ifdef __AVX2__
        __m256i temp_sum = _mm256_set1_epi32(0);
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
        {
          result(idx) = _mm256_and_si256(bits.avx_view(idx), rhs(idx));
          temp_sum = _mm256_or_si256(temp_sum, result(idx));
        }
#else
        __m256d temp_sum = _mm256_set1_pd(0.0);
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
        {
          result(idx) = _mm256_and_pd(bits.avx_view(idx), rhs(idx));
          temp_sum = _mm256_or_pd(temp_sum, result(idx));
        }
#endif
        result.sum_mask = extract_mask(temp_sum); 
        _mm256_zeroall();
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX> AVXTLBitMask<MAX>::operator^(
                                                 const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXTLBitMask<MAX> result;
#ifdef __AVX2__
      __m256i temp_sum = _mm256_set1_epi32(0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_xor_si256(bits.avx_view(idx), rhs(idx));
        temp_sum = _mm256_or_si256(temp_sum, result(idx));
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_xor_pd(bits.avx_view(idx), rhs(idx));
        temp_sum = _mm256_or_pd(temp_sum, result(idx));
      }
#endif
      result.sum_mask = extract_mask(temp_sum);
      _mm256_zeroall();
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX>& AVXTLBitMask<MAX>::operator|=(
                                                       const AVXTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask |= rhs.sum_mask;
#ifdef __AVX2__
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_or_si256(bits.avx_view(idx), rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_or_pd(bits.avx_view(idx), rhs(idx));
      }
#endif
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX>& AVXTLBitMask<MAX>::operator&=(
                                                       const AVXTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
#ifdef __AVX2__
        __m256i temp_sum = _mm256_set1_epi32(0);
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
        {
          bits.avx_view(idx) = _mm256_and_si256(bits.avx_view(idx), 
                                                  rhs(idx));
          temp_sum = _mm256_or_si256(temp_sum, bits.avx_view(idx));
        }
#else
        __m256d temp_sum = _mm256_set1_pd(0.0);
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
        {
          bits.avx_view(idx) = _mm256_and_pd(bits.avx_view(idx), 
                                               rhs(idx));
          temp_sum = _mm256_or_pd(temp_sum, bits.avx_view(idx));
        }
#endif
        sum_mask = extract_mask(temp_sum); 
      }
      else
      {
        sum_mask = 0;
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
          bits.avx_view(idx) = _mm256_set1_epi32(0);
      }
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX>& AVXTLBitMask<MAX>::operator^=(
                                                       const AVXTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef __AVX2__
      __m256i temp_sum = _mm256_set1_epi32(0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_xor_si256(bits.avx_view(idx), rhs(idx));
        temp_sum = _mm256_or_si256(temp_sum, bits.avx_view(idx));
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_xor_pd(bits.avx_view(idx), rhs(idx));
        temp_sum = _mm256_or_pd(temp_sum, bits.avx_view(idx));
      }
#endif
      sum_mask = extract_mask(temp_sum);
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXTLBitMask<MAX>::operator*(const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        {
          if (bits.bit_vector[idx] & rhs[idx])
            return false;
        }
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX> AVXTLBitMask<MAX>::operator-(
                                                 const AVXTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      AVXTLBitMask<MAX> result;
#ifdef __AVX2__
      __m256i temp_sum = _mm256_set1_epi32(0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_andnot_si256(rhs(idx), bits.avx_view(idx));
        temp_sum = _mm256_or_si256(temp_sum, result(idx));
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result(idx) = _mm256_andnot_pd(rhs(idx), bits.avx_view(idx));
        temp_sum = _mm256_or_pd(temp_sum, result(idx));
      }
#endif
      result.sum_mask = extract_mask(temp_sum);
      _mm256_zeroall();
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX>& AVXTLBitMask<MAX>::operator-=(
                                                       const AVXTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef __AVX2__
      __m256i temp_sum = _mm256_set1_epi32(0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_andnot_si256(rhs(idx), 
                                                   bits.avx_view(idx));
        temp_sum = _mm256_or_si256(temp_sum, bits.avx_view(idx));
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_view(idx) = _mm256_andnot_pd(rhs(idx),
                                                bits.avx_view(idx));
        temp_sum = _mm256_or_pd(temp_sum, bits.avx_view(idx));
      }
#endif
      sum_mask = extract_mask(temp_sum);
      _mm256_zeroall();
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXTLBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      // A great reason to have a summary mask
      return (sum_mask == 0);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool AVXTLBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX> AVXTLBitMask<MAX>::operator<<(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      AVXTLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
          result.sum_mask |= result[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        result.sum_mask |= result[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX> AVXTLBitMask<MAX>::operator>>(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      AVXTLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
          result.sum_mask |= result[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        result.sum_mask |= result[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX>& AVXTLBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
          sum_mask |= bits.bit_vector[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        sum_mask |= bits.bit_vector[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline AVXTLBitMask<MAX>& AVXTLBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
          sum_mask |= bits.bit_vector[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                        bits.bit_vector[BIT_ELMTS-1] >> local;
        sum_mask |= bits.bit_vector[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t AVXTLBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      return sum_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* AVXTLBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void AVXTLBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sum_mask);
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void AVXTLBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(sum_mask);
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void AVXTLBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* AVXTLBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned AVXTLBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned AVXTLBitMask<MAX>::pop_count(
                                                 const AVXTLBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline uint64_t AVXTLBitMask<MAX>::extract_mask(__m256i value)
    //-------------------------------------------------------------------------
    {
      __m128i left, right;
      right = _mm256_extractf128_si256(value, 0);
      left = _mm256_extractf128_si256(value, 1);
#if !defined(__LP64__) // handle 32-bit support
      uint64_t result = _mm_cvtsi128_si32(right);
      result |= uint64_t(_mm_cvtsi128_si32(_mm_shuffle_epi32(right,1))) << 32;
      result |= _mm_cvtsi128_si32(_mm_shuffle_epi32(right,2));
      result |= int64_t(_mm_cvtsi128_si32(_mm_shuffle_epi32(right,3))) << 32;
      result |= _mm_cvtsi128_si32(left);
      result |= uint64_t(_mm_cvtsi128_si32(_mm_shuffle_epi32(left,1))) << 32;
      result |= _mm_cvtsi128_si32(_mm_shuffle_epi32(left,2));
      result |= int64_t(_mm_cvtsi128_si32(_mm_shuffle_epi32(left,3))) << 32;
#elif defined(__SSE4_1__) // case we have sse 4.1
      uint64_t result = _mm_extract_epi64(right, 0);
      result |= _mm_extract_epi64(right, 1);
      result |= _mm_extract_epi64(left, 0);
      result |= _mm_extract_epi64(left, 1);
#else // Assume we have sse 2
      uint64_t result = _mm_cvtsi128_si64(right);
      result |= _mm_cvtsi128_si64(_mm_shuffle_epi32(right, 14));
      result |= _mm_cvtsi128_si64(left);
      result |= _mm_cvtsi128_si64(_mm_shuffle_epi32(left, 14));
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline uint64_t AVXTLBitMask<MAX>::extract_mask(__m256d value)
    //-------------------------------------------------------------------------
    {
      __m256i temp = _mm256_castpd_si256(value);
      return extract_mask(temp);
    }
#endif // __AVX__

#ifdef __ALTIVEC__
    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    PPCBitMask<MAX>::PPCBitMask(uint64_t init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    PPCBitMask<MAX>::PPCBitMask(const PPCBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    PPCBitMask<MAX>::~PPCBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] |= (1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] &= ~(1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      const __vector unsigned long long zero_vec = vec_splats(0ULL);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_view(idx) = zero_vec;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::PPCView<true>
                     PPCBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.ppc_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::PPCView<false>
                           PPCBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.ppc_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& PPCBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& PPCBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCBitMask<MAX>::operator==(const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
	if (bits.bit_vector[idx] != rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCBitMask<MAX>::operator<(const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.ppc_view(idx) < rhs[idx])
          return true;
        else if (bits.bits_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCBitMask<MAX>::operator!=(const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX>& PPCBitMask<MAX>::operator=(const PPCBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX> PPCBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      PPCBitMask<MAX> result;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs = bits.ppc_view(idx);
        result(idx) = ~rhs;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX> PPCBitMask<MAX>::operator|(
                                                   const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCBitMask<MAX> result;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        result(idx) = vec_or(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX> PPCBitMask<MAX>::operator&(
                                                   const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCBitMask<MAX> result;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        result(idx) = vec_and(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX> PPCBitMask<MAX>::operator^(
                                                   const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCBitMask<MAX> result;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        result(idx) = vec_xor(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX>& PPCBitMask<MAX>::operator|=(const PPCBitMask &rhs) 
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        bits.ppc_view(idx) = vec_or(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX>& PPCBitMask<MAX>::operator&=(const PPCBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        bits.ppc_view(idx) = vec_and(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX>& PPCBitMask<MAX>::operator^=(const PPCBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        bits.ppc_view(idx) = vec_xor(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCBitMask<MAX>::operator*(const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] & rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX> PPCBitMask<MAX>::operator-(
                                                   const PPCBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCBitMask<MAX> result;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        result(idx) = vec_and(rhs1, ~rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX>& PPCBitMask<MAX>::operator-=(const PPCBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        bits.ppc_view(idx) = vec_and(rhs1, ~rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != 0)
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX> PPCBitMask<MAX>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      PPCBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX> PPCBitMask<MAX>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      PPCBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX>& PPCBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCBitMask<MAX>& PPCBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                      bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t PPCBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      uint64_t result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result |= bits.bit_vector[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* PPCBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void PPCBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void PPCBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void PPCBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* PPCBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned PPCBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned PPCBitMask<MAX>::pop_count(
                                                   const PPCBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    PPCTLBitMask<MAX>::PPCTLBitMask(uint64_t init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    PPCTLBitMask<MAX>::PPCTLBitMask(const PPCTLBitMask &rhs)
      : sum_mask(rhs.sum_mask)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    PPCTLBitMask<MAX>::~PPCTLBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCTLBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      bits.bit_vector[idx] |= set_mask;
      sum_mask |= set_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCTLBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      const uint64_t unset_mask = ~set_mask;
      bits.bit_vector[idx] &= unset_mask;
      // Unset the summary mask and then reset if necessary
      sum_mask &= unset_mask;
      for (unsigned i = 0; i < BIT_ELMTS; i++)
        sum_mask |= bits.bit_vector[i];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCTLBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCTLBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCTLBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCTLBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCTLBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCTLBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCTLBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      const __vector unsigned long long zero_vec = vec_splats(0ULL);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_view(idx) = zero_vec; 
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::PPCView<true>
                   PPCTLBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.ppc_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::PPCView<false>
                         PPCTLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.ppc_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& PPCTLBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& PPCTLBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCTLBitMask<MAX>::operator==(const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask != rhs.sum_mask)
        return false;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
	if (bits.bit_vector[idx] != rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCTLBitMask<MAX>::operator<(const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] < rhs[idx])
          return true;
        else if (bits.bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCTLBitMask<MAX>::operator!=(const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator=(
                                                       const PPCTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = rhs.sum_mask;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX> PPCTLBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      PPCTLBitMask<MAX> result;
      __vector unsigned long long result_mask = vec_splats(0ULL);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs = bits.ppc_view(idx);
        __vector unsigned long long lhs = ~rhs;
        result(idx) = lhs;
        result_mask = vec_or(result_mask, lhs);
      }
      result.sum_mask = extract_mask(result_mask);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX> PPCTLBitMask<MAX>::operator|(
                                                 const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCTLBitMask<MAX> result;
      result.sum_mask = sum_mask | rhs.sum_mask;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        result(idx) = vec_or(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX> PPCTLBitMask<MAX>::operator&(
                                                 const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCTLBitMask<MAX> result;
      // If they are independent then we are done
      if (sum_mask & rhs.sum_mask)
      {
        __vector unsigned long long temp_sum = vec_splats(0ULL);
        for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
        {
          __vector unsigned long long rhs1 = bits.ppc_view(idx);
          __vector unsigned long long rhs2 = rhs(idx);
          __vector unsigned long long lhs = vec_and(rhs1, rhs2);
          result(idx) = lhs;
          temp_sum = vec_or(temp_sum, lhs);
        }
        result.sum_mask = extract_mask(temp_sum); 
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX> PPCTLBitMask<MAX>::operator^(
                                                 const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCTLBitMask<MAX> result;
      __vector unsigned long long temp_sum = vec_splats(0ULL);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        __vector unsigned long long lhs = vec_xor(rhs1, rhs2);
        result(idx) = lhs;
        temp_sum = vec_or(temp_sum, lhs);
      }
      result.sum_mask = extract_mask(temp_sum);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator|=(
                                                       const PPCTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask |= rhs.sum_mask;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        bits.ppc_view(idx) = vec_or(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator&=(
                                                       const PPCTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        __vector unsigned long long temp_sum = vec_splats(0ULL);
        for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
        {
          __vector unsigned long long rhs1 = bits.ppc_view(idx);
          __vector unsigned long long rhs2 = rhs(idx);
          __vector unsigned long long lhs = vec_and(rhs1, rhs2);
          bits.ppc_view(idx) = lhs;
          temp_sum = vec_or(temp_sum, lhs);
        }
        sum_mask = extract_mask(temp_sum); 
      }
      else
      {
        sum_mask = 0;
	const __vector unsigned long long zero_vec = vec_splats(0ULL);
        for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
          bits.ppc_view(idx) = zero_vec;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator^=(
                                                       const PPCTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      __vector unsigned long long temp_sum = vec_splats(0ULL);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        __vector unsigned long long lhs = vec_xor(rhs1, rhs2);
        bits.ppc_view(idx) = lhs;
        temp_sum = vec_or(temp_sum, lhs);
      }
      sum_mask = extract_mask(temp_sum);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCTLBitMask<MAX>::operator*(const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        {
          if (bits.bit_vector[idx] & rhs[idx])
            return false;
        }
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX> PPCTLBitMask<MAX>::operator-(
                                                 const PPCTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      PPCTLBitMask<MAX> result;
      __vector unsigned long long temp_sum = vec_splats(0ULL);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        __vector unsigned long long lhs = vec_and(rhs1, ~rhs2);
        result(idx) = lhs;
        temp_sum = vec_or(temp_sum, lhs);
      }
      result.sum_mask = extract_mask(temp_sum);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator-=(
                                                       const PPCTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      __vector unsigned long long temp_sum = vec_splats(0ULL);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        __vector unsigned long long rhs1 = bits.ppc_view(idx);
        __vector unsigned long long rhs2 = rhs(idx);
        __vector unsigned long long lhs = vec_and(rhs1, ~rhs2);
        bits.ppc_view(idx) = lhs;
        temp_sum = vec_or(temp_sum, lhs);
      }
      sum_mask = extract_mask(temp_sum);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCTLBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      // A great reason to have a summary mask
      return (sum_mask == 0);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool PPCTLBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX> PPCTLBitMask<MAX>::operator<<(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      PPCTLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
          result.sum_mask |= result[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        result.sum_mask |= result[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX> PPCTLBitMask<MAX>::operator>>(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      PPCTLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
          result.sum_mask |= result[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        result.sum_mask |= result[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
          sum_mask |= bits.bit_vector[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        sum_mask |= bits.bit_vector[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
          sum_mask |= bits.bit_vector[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                        bits.bit_vector[BIT_ELMTS-1] >> local;
        sum_mask |= bits.bit_vector[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t PPCTLBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      return sum_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* PPCTLBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void PPCTLBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sum_mask);
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void PPCTLBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(sum_mask);
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void PPCTLBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* PPCTLBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned PPCTLBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned PPCTLBitMask<MAX>::pop_count(
                                                 const PPCTLBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline uint64_t PPCTLBitMask<MAX>::extract_mask(
                                             __vector unsigned long long value)
    //-------------------------------------------------------------------------
    {
      uint64_t left = value[0];
      uint64_t right = value[1];
      return (left | right);
    }
#endif // __ALTIVEC__

#ifdef __ARM_NEON
    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    NeonBitMask<MAX>::NeonBitMask(uint64_t init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    NeonBitMask<MAX>::NeonBitMask(const NeonBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        bits.neon_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    NeonBitMask<MAX>::~NeonBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] |= (1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      bits.bit_vector[idx] &= ~(1UL << (bit & 0x3F));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      const uint32x4_t zero_vec = vdupq_n_u32(0);
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        bits.neon_view(idx) = zero_vec;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::NeonView<true>
                    NeonBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.neon_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::NeonView<false>
                          NeonBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.neon_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& NeonBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& NeonBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonBitMask<MAX>::operator==(const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
	if (bits.bit_vector[idx] != rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonBitMask<MAX>::operator<(const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.neon_view(idx) < rhs[idx])
          return true;
        else if (bits.bits_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonBitMask<MAX>::operator!=(const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX>& NeonBitMask<MAX>::operator=(
                                                        const NeonBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        bits.neon_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX> NeonBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      NeonBitMask<MAX> result;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs = bits.neon_view(idx);
        result(idx) = vmvnq_u32(rhs);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX> NeonBitMask<MAX>::operator|(
                                                  const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonBitMask<MAX> result;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        result(idx) = vorrq_u32(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX> NeonBitMask<MAX>::operator&(
                                                  const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonBitMask<MAX> result;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        result(idx) = vandq_u32(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX> NeonBitMask<MAX>::operator^(
                                                  const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonBitMask<MAX> result;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        result(idx) = veorq_u32(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX>& NeonBitMask<MAX>::operator|=(
                                                        const NeonBitMask &rhs) 
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        bits.neon_view(idx) = vorrq_u32(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX>& NeonBitMask<MAX>::operator&=(
                                                        const NeonBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        bits.neon_view(idx) = vandq_u32(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX>& NeonBitMask<MAX>::operator^=(
                                                        const NeonBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        bits.neon_view(idx) = veorq_u32(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonBitMask<MAX>::operator*(const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] & rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX> NeonBitMask<MAX>::operator-(
                                                  const NeonBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonBitMask<MAX> result;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        result(idx) = vandq_u32(rhs1, vmvnq_u32(rhs2));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX>& NeonBitMask<MAX>::operator-=(
                                                        const NeonBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        bits.neon_view(idx) = vandq_u32(rhs1, ~rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] != 0)
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX> NeonBitMask<MAX>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      NeonBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX> NeonBitMask<MAX>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      NeonBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX>& NeonBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonBitMask<MAX>& NeonBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                      bits.bit_vector[BIT_ELMTS-1] >> local;
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t NeonBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      uint64_t result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result |= bits.bit_vector[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* NeonBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void NeonBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void NeonBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void NeonBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* NeonBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned NeonBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned NeonBitMask<MAX>::pop_count(
                                                  const NeonBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    NeonTLBitMask<MAX>::NeonTLBitMask(uint64_t init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bits.bit_vector[idx] = init;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    NeonTLBitMask<MAX>::NeonTLBitMask(const NeonTLBitMask &rhs)
      : sum_mask(rhs.sum_mask)
    //-------------------------------------------------------------------------
    {
      static_assert((MAX % 128) == 0, "Bad MAX");
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        bits.neon_view(idx) = rhs(idx);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    NeonTLBitMask<MAX>::~NeonTLBitMask(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonTLBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      bits.bit_vector[idx] |= set_mask;
      sum_mask |= set_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonTLBitMask<MAX>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      const uint64_t set_mask = (1UL << (bit & 0x3F));
      const uint64_t unset_mask = ~set_mask;
      bits.bit_vector[idx] &= unset_mask;
      // Unset the summary mask and then reset if necessary
      sum_mask &= unset_mask;
      for (unsigned i = 0; i < BIT_ELMTS; i++)
        sum_mask |= bits.bit_vector[i];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonTLBitMask<MAX>::assign_bit(unsigned bit, bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonTLBitMask<MAX>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit < MAX);
#endif
      unsigned idx = bit >> 6;
      return (bits.bit_vector[idx] & (1UL << (bit & 0x3F)));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonTLBitMask<MAX>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*ELEMENT_SIZE + j);
            }
          }
        }
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonTLBitMask<MAX>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      unsigned element = bit >> bits.SHIFT; 
      unsigned offset = bit & bits.MASK;
      if (bits.bit_vector[element] & (1ULL << offset))
      {
        int index = 0;
        for (unsigned idx = 0; idx < element; idx++)
          index += __builtin_popcountll(bits.bit_vector[idx]);
        // Handle dumb c++ shift overflow
        if (offset == 0)
          return index;
        // Just count the bits up to but not including the actual
        // bit we're looking for since indexes are zero-base
        index += __builtin_popcountll(
            bits.bit_vector[element] << (ELEMENT_SIZE - offset));
        return index;
      }
      else // It's not set otherwise so we couldn't find an index
        return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonTLBitMask<MAX>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        unsigned local = __builtin_popcountll(bits.bit_vector[idx]);
        if (index < local)
        {
          for (unsigned j = 0; j < ELEMENT_SIZE; j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
            {
              if (index == 0)
                return (offset + j);
              index--;
            }
          }
        }
        index -= local;
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int NeonTLBitMask<MAX>::find_next_set(unsigned start) const
    //-------------------------------------------------------------------------
    {
      int idx = start / ELEMENT_SIZE; // truncate
      int offset = idx * ELEMENT_SIZE; 
      int j = start % ELEMENT_SIZE;
      if (j > 0) // if we are already in the middle of element search it
      {
        for ( ; j < int(ELEMENT_SIZE); j++)
        {
          if (bits.bit_vector[idx] & (1ULL << j))
            return (offset + j);
        }
        idx++;
        offset += ELEMENT_SIZE;
      }
      for ( ; idx < int(BIT_ELMTS); idx++)
      {
        if (bits.bit_vector[idx] > 0) // if it has any valid entries, find next
        {
          for (j = 0; j < int(ELEMENT_SIZE); j++)
          {
            if (bits.bit_vector[idx] & (1ULL << j))
              return (offset + j);
          }
        }
        offset += ELEMENT_SIZE;
      }
      return -1;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void NeonTLBitMask<MAX>::clear(void)
    //-------------------------------------------------------------------------
    {
      const uint32x4_t zero_vec = vdupq_n_u32(0);
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        bits.neon_view(idx) = zero_vec; 
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::NeonView<true>
                  NeonTLBitMask<MAX>::operator()(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.neon_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline BitMaskHelp::NeonView<false>
                        NeonTLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.neon_view(idx);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& NeonTLBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& NeonTLBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonTLBitMask<MAX>::operator==(const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask != rhs.sum_mask)
        return false;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
	if (bits.bit_vector[idx] != rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonTLBitMask<MAX>::operator<(const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx] < rhs[idx])
          return true;
        else if (bits.bit_vector[idx] > rhs[idx])
          return false;
      }
      // Otherwise they are equal so false
      return false;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonTLBitMask<MAX>::operator!=(const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX>& NeonTLBitMask<MAX>::operator=(
                                                      const NeonTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = rhs.sum_mask;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        bits.neon_view(idx) = rhs(idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX> NeonTLBitMask<MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      NeonTLBitMask<MAX> result;
      uint32x4_t result_mask = vdupq_n_u32(0);
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs = bits.neon_view(idx);
        uint32x4_t lhs = vmvnq_u32(rhs);
        result(idx) = lhs;
        result_mask = vorrq_u32(result_mask, lhs);
      }
      result.sum_mask = extract_mask(result_mask);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX> NeonTLBitMask<MAX>::operator|(
                                                const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonTLBitMask<MAX> result;
      result.sum_mask = sum_mask | rhs.sum_mask;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        result(idx) = vorrq_u32(rhs1, rhs2);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX> NeonTLBitMask<MAX>::operator&(
                                                const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonTLBitMask<MAX> result;
      // If they are independent then we are done
      if (sum_mask & rhs.sum_mask)
      {
        uint32x4_t temp_sum = vdupq_n_u32(0);
        for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
        {
          uint32x4_t rhs1 = bits.neon_view(idx);
          uint32x4_t rhs2 = rhs(idx);
          uint32x4_t lhs = vandq_u32(rhs1, rhs2);
          result(idx) = lhs;
          temp_sum = vorrq_u32(temp_sum, lhs);
        }
        result.sum_mask = extract_mask(temp_sum); 
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX> NeonTLBitMask<MAX>::operator^(
                                                const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonTLBitMask<MAX> result;
      uint32x4_t temp_sum = vdupq_n_u32(0);
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        uint32x4_t lhs = veorq_u32(rhs1, rhs2);
        result(idx) = lhs;
        temp_sum = vorrq_u32(temp_sum, lhs);
      }
      result.sum_mask = extract_mask(temp_sum);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX>& NeonTLBitMask<MAX>::operator|=(
                                                      const NeonTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask |= rhs.sum_mask;
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        bits.neon_view(idx) = vorrq_u32(rhs1, rhs2);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX>& NeonTLBitMask<MAX>::operator&=(
                                                      const NeonTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        uint32x4_t temp_sum = vdupq_n_u32(0);
        for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
        {
          uint32x4_t rhs1 = bits.neon_view(idx);
          uint32x4_t rhs2 = rhs(idx);
          uint32x4_t lhs = vandq_u32(rhs1, rhs2);
          bits.neon_view(idx) = lhs;
          temp_sum = vorrq_u32(temp_sum, lhs);
        }
        sum_mask = extract_mask(temp_sum); 
      }
      else
      {
        sum_mask = 0;
        const uint32x4_t zero_vec = vdupq_n_u32(0);
        for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
          bits.neon_view(idx) = zero_vec;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX>& NeonTLBitMask<MAX>::operator^=(
                                                      const NeonTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      uint32x4_t temp_sum = vdupq_n_u32(0);
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        uint32x4_t lhs = veorq_u32(rhs1, rhs2);
        bits.neon_view(idx) = lhs;
        temp_sum = vorrq_u32(temp_sum, lhs);
      }
      sum_mask = extract_mask(temp_sum);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonTLBitMask<MAX>::operator*(const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      if (sum_mask & rhs.sum_mask)
      {
        for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        {
          if (bits.bit_vector[idx] & rhs[idx])
            return false;
        }
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX> NeonTLBitMask<MAX>::operator-(
                                                const NeonTLBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      NeonTLBitMask<MAX> result;
      uint32x4_t temp_sum = vdupq_n_u32(0);
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        uint32x4_t lhs = vandq_u32(rhs1, vmvnq_u32(rhs2));
        result(idx) = lhs;
        temp_sum = vorrq_u32(temp_sum, lhs);
      }
      result.sum_mask = extract_mask(temp_sum);
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX>& NeonTLBitMask<MAX>::operator-=(
                                                      const NeonTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      uint32x4_t temp_sum = vdupq_n_u32(0);
      for (unsigned idx = 0; idx < NEON_ELMTS; idx++)
      {
        uint32x4_t rhs1 = bits.neon_view(idx);
        uint32x4_t rhs2 = rhs(idx);
        uint32x4_t lhs = vandq_u32(rhs1, vmvnq_u32(rhs2));
        bits.neon_view(idx) = lhs;
        temp_sum = vorrq_u32(temp_sum, lhs);
      }
      sum_mask = extract_mask(temp_sum);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonTLBitMask<MAX>::empty(void) const
    //-------------------------------------------------------------------------
    {
      // A great reason to have a summary mask
      return (sum_mask == 0);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline bool NeonTLBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX> NeonTLBitMask<MAX>::operator<<(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      NeonTLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          result[idx] = bits.bit_vector[idx-range]; 
          result.sum_mask |= result[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[range] = bits.bit_vector[0] << local; 
        result.sum_mask |= result[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX> NeonTLBitMask<MAX>::operator>>(
                                                          unsigned shift) const
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      NeonTLBitMask<MAX> result;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          result[idx] = bits.bit_vector[idx+range];
          result.sum_mask |= result[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          result[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          result[idx] = left | right;
          result.sum_mask |= result[idx];
        }
        // Handle the last case
        result[BIT_ELMTS-(range+1)] = bits.bit_vector[BIT_ELMTS-1] >> local;
        result.sum_mask |= result[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          result[idx] = 0;
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX>& NeonTLBitMask<MAX>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      // Find the range
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move the individual words
        for (int idx = (BIT_ELMTS-1); idx >= int(range); idx--)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx-range]; 
          sum_mask |= bits.bit_vector[idx];
        }
        // fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (int idx = (BIT_ELMTS-1); idx > int(range); idx--)
        {
          uint64_t left = bits.bit_vector[idx-range] << local;
          uint64_t right = bits.bit_vector[idx-(range+1)] >> ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[range] = bits.bit_vector[0] << local; 
        sum_mask |= bits.bit_vector[range];
        // Fill in everything else with zeros
        for (unsigned idx = 0; idx < range; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline NeonTLBitMask<MAX>& NeonTLBitMask<MAX>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      unsigned range = shift >> 6;
      unsigned local = shift & 0x3F;
      sum_mask = 0;
      if (!local)
      {
        // Fast case where we just have to move individual words
        for (unsigned idx = 0; idx < (BIT_ELMTS-range); idx++)
        {
          bits.bit_vector[idx] = bits.bit_vector[idx+range];
          sum_mask |= bits.bit_vector[idx];
        }
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS); idx++)
          bits.bit_vector[idx] = 0;
      }
      else
      {
        // Slow case with merging words
        for (unsigned idx = 0; idx < (BIT_ELMTS-(range+1)); idx++)
        {
          uint64_t right = bits.bit_vector[idx+range] >> local;
          uint64_t left = bits.bit_vector[idx+range+1] << ((1 << 6) - local);
          bits.bit_vector[idx] = left | right;
          sum_mask |= bits.bit_vector[idx];
        }
        // Handle the last case
        bits.bit_vector[BIT_ELMTS-(range+1)] = 
                                        bits.bit_vector[BIT_ELMTS-1] >> local;
        sum_mask |= bits.bit_vector[BIT_ELMTS-(range+1)];
        // Fill in everything else with zeros
        for (unsigned idx = (BIT_ELMTS-range); idx < BIT_ELMTS; idx++)
          bits.bit_vector[idx] = 0;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t NeonTLBitMask<MAX>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      return sum_mask;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t* NeonTLBitMask<MAX>::base(void) const
    //-------------------------------------------------------------------------
    {
      return bits.bit_vector;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename ST>
    inline void NeonTLBitMask<MAX>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sum_mask);
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename DT>
    inline void NeonTLBitMask<MAX>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(sum_mask);
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX> template<typename FUNCTOR>
    inline void NeonTLBitMask<MAX>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bits.bit_vector[idx])
        {
          unsigned value = idx * ELEMENT_SIZE;
          for (unsigned i = 0; i < ELEMENT_SIZE; i++, value++)
            if (bits.bit_vector[idx] & (1ULL << i))
              functor.apply(value);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* NeonTLBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline unsigned NeonTLBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(bits.bit_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline unsigned NeonTLBitMask<MAX>::pop_count(
                                                const NeonTLBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      unsigned result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (unsigned idx = 0; idx < MAX; idx++)
      {
        if (mask.is_set(idx))
          result++;
      }
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    /*static*/ inline uint64_t NeonTLBitMask<MAX>::extract_mask(
                                                              uint32x4_t value)
    //-------------------------------------------------------------------------
    {
      uint64_t zero = vgetq_lane_u32(value, 0);
      uint64_t one = vgetq_lane_u32(value, 1);
      uint64_t two = vgetq_lane_u32(value, 2);
      uint64_t three = vgetq_lane_u32(value, 3);
      one <<= 32;
      three <<= 32;
      return (zero | one | two | three);
    }
#endif // __ARM_NEON

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    CompoundBitMask<DT,BLOAT,BIDIR>::CompoundBitMask(uint64_t init)
    //-------------------------------------------------------------------------
    {
      if (init != 0)
      {
        mask.dense = new DT(init);
        sparse_size = MAX_SPARSE+1;
        sparsify();
      }
      else
        sparse_size = 0;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    CompoundBitMask<DT,BLOAT,BIDIR>::CompoundBitMask(
                                    const CompoundBitMask<DT,BLOAT,BIDIR> &rhs)
      : sparse_size(rhs.sparse_size)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
        mask.sparse = rhs.mask.sparse;
      else
        mask.dense = new DT(*(rhs.mask.dense));
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    CompoundBitMask<DT,BLOAT,BIDIR>::~CompoundBitMask(void) 
    //-------------------------------------------------------------------------
    {
      if (!is_sparse())
        delete mask.dense;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void CompoundBitMask<DT,BLOAT, BIDIR>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        // Go through the existing elements and see if it's already here
        if (std::binary_search(mask.sparse.begin(), 
                               mask.sparse.begin() + sparse_size, bit))
          return;
        // Need to add it at this point
        if (sparse_size == MAX_SPARSE)
        {
          DT *newmask = new DT();
          for (unsigned idx = 0; idx < sparse_size; idx++)
            newmask->set_bit(mask.sparse[idx]);
          newmask->set_bit(bit);
          mask.dense = newmask;
          sparse_size++;
        }
        else
        {
          // Insert it at the right place in the array
          for (int idx = sparse_size-1; idx >= 0; idx--)
          {
            if (mask.sparse[idx] < bit)
            {
              mask.sparse[idx+1] = bit;
              sparse_size++;
              return;
            }
            else
              mask.sparse[idx+1] = mask.sparse[idx];
          }
          // If we get here, we still haven't added it
          mask.sparse[0] = bit;
          sparse_size++;
        }
      }
      else
        mask.dense->set_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void CompoundBitMask<DT,BLOAT,BIDIR>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        if (!std::binary_search(mask.sparse.begin(), 
                                mask.sparse.begin() + sparse_size, bit))
          return;
        bool shifting = false;
        for (unsigned idx = 0; idx < sparse_size; idx++)
        {
          if (mask.sparse[idx] != bit)
          {
            if (shifting)
              mask.sparse[idx-1] = mask.sparse[idx];
          }
          else
            shifting = true;
        }
        if (shifting)
          sparse_size--;
      }
      else
      {
        mask.dense->unset_bit(bit);
        sparsify();
      }
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void CompoundBitMask<DT,BLOAT,BIDIR>::assign_bit(unsigned bit, 
                                                            bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
        return std::binary_search(mask.sparse.begin(), 
                                  mask.sparse.begin() + sparse_size, bit);
      else
        return mask.dense->is_set(bit);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline int CompoundBitMask<DT,BLOAT,BIDIR>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
#ifdef DEBUG_LEGION
        assert(sparse_size > 0);
#endif
        return mask.sparse[0];
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!!(*mask.dense));
#endif
        return mask.dense->find_first_set();
      }
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline int CompoundBitMask<DT,BLOAT,BIDIR>::find_next_set(
                                                          unsigned start) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        int index = find_index(start);
#ifdef DEBUG_LEGION
        assert(index >= 0);
#endif
        // Increment to the next index
        if (++index == sparse_size)
          return -1;
        else
          return mask.sparse[index];
      }
      else
        return mask.dense->find_next_set(start);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline int CompoundBitMask<DT,BLOAT,BIDIR>::find_index(unsigned bit) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bit >= 0);
      assert(bit < pop_count());
#endif
      if (is_sparse())
      {
        // Binary search for it
        unsigned first = 0;
        unsigned last = sparse_size - 1;
        unsigned mid = 0;
        while (first <= last)
        {
          mid = (first + last) / 2;
          const unsigned midval = mask.sparse[mid];
          if (bit == midval)
            return mid;
          else if (bit < midval)
            last = mid - 1;
          else if (midval < bit)
            first = mid + 1;
          else
            break;
        }
        return -1;
      }
      else
        return mask.dense->find_index(bit);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline int CompoundBitMask<DT,BLOAT,BIDIR>::get_index(unsigned index) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        if (index < sparse_size)
          return mask.sparse[index];
        else // Don't have any entry at that index
          return -1;
      }
      else
        return mask.dense->get_index(index);
    }
    
    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void CompoundBitMask<DT,BLOAT,BIDIR>::clear(void)
    //-------------------------------------------------------------------------
    {
      if (!is_sparse())
        delete mask.dense;
      sparse_size = 0;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::operator==(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const
    //-------------------------------------------------------------------------
    {
      if (pop_count() != rhs.pop_count())
        return false;
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (mask.sparse[idx] != rhs.mask.sparse[idx])
            return false;
        return true;
      }
      else if (rhs.is_sparse())
      {
        for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
          if (!mask.dense->is_set(rhs.mask.sparse[idx]))
            return false;
        return true;
      }
      else
        return (*mask.dense) == (*rhs.mask.dense);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::operator<(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const
    //-------------------------------------------------------------------------
    {
      size_t lhs_size = pop_count();
      size_t rhs_size = rhs.pop_count();
      if (lhs_size < rhs_size)
        return true;
      if (rhs_size > lhs_size)
        return false;
      // Same size so now we can do lexicographic comparison
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (mask.sparse[idx] < rhs.mask.sparse[idx])
            return true;
          else if (mask.sparse[idx] > rhs.mask.sparse[idx])
            return false;
        // Otherwise they are equal so not strictly less than
        return false;
      }
      else
        return (*mask.dense) < (*rhs.mask.dense);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::operator!=(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR>& 
        CompoundBitMask<DT,BLOAT,BIDIR>::operator=(
                                    const CompoundBitMask<DT,BLOAT,BIDIR> &rhs)
    //-------------------------------------------------------------------------
    {
      const bool was_dense = !is_sparse();
      sparse_size = rhs.sparse_size;
      if (is_sparse())
      {
        if (was_dense)
          delete mask.dense;
        mask.sparse = rhs.mask.sparse;
      }
      else
      {
        if (was_dense)
          (*mask.dense) = (*rhs.mask.dense);
        else
          mask.dense = new DT(*rhs.mask.dense);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR> 
                         CompoundBitMask<DT,BLOAT,BIDIR>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<DT,BLOAT,BIDIR> result;
      if (is_sparse())
      {
        result.mask.dense = new DT(0xFFFFFFFFFFFFFFFFULL);
        result.sparse_size = MAX_SPARSE+1;
        for (unsigned idx = 0; idx < sparse_size; idx++)
          result.unset_bit(mask.sparse[idx]);
      }
      else
      {
        result.mask.dense = new DT(~(*mask.dense));
        result.sparse_size = MAX_SPARSE+1;
        result.sparsify();
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR> 
      CompoundBitMask<DT,BLOAT,BIDIR>::operator|(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<DT,BLOAT,BIDIR> result;
      if (!is_sparse())
      {
        result.mask.dense = new DT(*(mask.dense));
        result.sparse_size = MAX_SPARSE+1;
        if (rhs.is_sparse())
        {
          for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
            result.set_bit(rhs.mask.sparse[idx]);
        }
        else
          (*result.mask.dense) |= (*rhs.mask.dense);
      }
      else if (!rhs.is_sparse())
      {
        result.mask.dense = new DT(*rhs.mask.dense);
        result.sparse_size = MAX_SPARSE+1;
        for (unsigned idx = 0; idx < sparse_size; idx++)
          result.set_bit(mask.sparse[idx]);
      }
      else
      {
        result.sparse_size = sparse_size;
        result.mask.sparse = mask.sparse;
        for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
          result.set_bit(rhs.mask.sparse[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR> 
      CompoundBitMask<DT,BLOAT,BIDIR>::operator&(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<DT,BLOAT,BIDIR> result;
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (rhs.is_set(mask.sparse[idx]))
            result.set_bit(mask.sparse[idx]);
      }
      else if (rhs.is_sparse())
      {
        for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
          if (is_set(rhs.mask.sparse[idx]))
            result.set_bit(rhs.mask.sparse[idx]);
      }
      else
      {
        result.mask.dense = new DT(*mask.dense);
        result.sparse_size = MAX_SPARSE+1;
        (*result.mask.dense) &= (*rhs.mask.dense);
        result.sparsify();
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR> 
      CompoundBitMask<DT,BLOAT,BIDIR>::operator^(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<DT,BLOAT,BIDIR> result;
      if (!is_sparse())
      {
        result.mask.dense = new DT(*mask.dense);
        result.sparse_size = MAX_SPARSE+1;
        if (rhs.is_sparse())
        {
          for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
            if (is_set(rhs.mask.sparse[idx]))
              result.unset_bit(rhs.mask.sparse[idx]);
            else
              result.set_bit(rhs.mask.sparse[idx]);
        }
        else
        {
          (*result.mask.dense) ^= (*rhs.mask.dense);
          result.sparsify();
        }
      }
      else if (!rhs.is_sparse())
      {
        result.mask.dense = new DT(*rhs.mask.dense);
        result.sparse_size = MAX_SPARSE+1;
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (rhs.is_set(mask.sparse[idx]))
            result.unset_bit(mask.sparse[idx]);
          else
            result.set_bit(mask.sparse[idx]);
      }
      else
      {
        result.mask.sparse = mask.sparse;
        result.sparse_size = sparse_size;
        for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
          if (is_set(rhs.mask.sparse[idx]))
            result.unset_bit(rhs.mask.sparse[idx]);
          else
            result.set_bit(rhs.mask.sparse[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR>& 
            CompoundBitMask<DT,BLOAT,BIDIR>::operator|=(
                                    const CompoundBitMask<DT,BLOAT,BIDIR> &rhs)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        if (rhs.is_sparse())
        {
          for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
            set_bit(rhs.mask.sparse[idx]);
        }
        else
        {
          DT *newmask = new DT(*rhs.mask.dense);
          for (unsigned idx = 0; idx < sparse_size; idx++)
            newmask->set_bit(mask.sparse[idx]);
          mask.dense = newmask;
          sparse_size = MAX_SPARSE+1;
        }
      }
      else
      {
        if (rhs.is_sparse())
        {
          for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
            mask.dense->set_bit(rhs.mask.sparse[idx]);
        }
        else
          (*mask.dense) |= (*rhs.mask.dense);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR>& 
            CompoundBitMask<DT,BLOAT,BIDIR>::operator&=(
                                    const CompoundBitMask<DT,BLOAT,BIDIR> &rhs)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        unsigned offset = 0;
        for (unsigned idx = 0; idx < sparse_size; idx++)
        {
          if (rhs.is_set(mask.sparse[idx]))
          {
            if (offset != idx)
              mask.sparse[offset++] = mask.sparse[idx];
            else
              offset++;
          }
        }
        sparse_size = offset;
      }
      else
      {
        if (rhs.is_sparse())
        {
          DT *oldmask = mask.dense;
          sparse_size = 0;
          for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
            if (oldmask->is_set(rhs.mask.sparse[idx]))
              mask.sparse[sparse_size++] = rhs.mask.sparse[idx];
          delete oldmask;
        }
        else
        {
          (*mask.dense) &= (*rhs.mask.dense);
          sparsify();
        }
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR>& 
            CompoundBitMask<DT,BLOAT,BIDIR>::operator^=(
                                    const CompoundBitMask<DT,BLOAT,BIDIR> &rhs)
    //-------------------------------------------------------------------------
    {
      if (rhs.is_sparse())
      {
        for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
          if (is_set(rhs.mask.sparse[idx]))
            unset_bit(rhs.mask.sparse[idx]);
          else
            set_bit(rhs.mask.sparse[idx]);
      }
      else if (is_sparse())
      {
        DT *newmask = new DT(*rhs.mask.dense);
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (newmask->is_set(mask.sparse[idx]))
            newmask->unset_bit(mask.sparse[idx]);
          else
            newmask->set_bit(mask.sparse[idx]);
        mask.dense = newmask;
        sparse_size = MAX_SPARSE+1;
        sparsify();
      }
      else
      {
        (*mask.dense) ^= (*rhs.mask.dense);
        sparsify();
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::operator*(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const 
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (rhs.is_set(mask.sparse[idx]))
            return false;
        return true;
      }
      else if (rhs.is_sparse())
      {
        for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
          if (mask.dense->is_set(rhs.mask.sparse[idx]))
            return false;
        return true;
      }
      else
        return (*mask.dense) * (*rhs.mask.dense);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::empty(void) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
        return (sparse_size == 0);
      if (BIDIR)
        return false;
      else
        return !(*mask.dense);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return empty();
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR> 
      CompoundBitMask<DT,BLOAT,BIDIR>::operator-(
                              const CompoundBitMask<DT,BLOAT,BIDIR> &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<DT,BLOAT,BIDIR> result;
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (!rhs.is_set(mask.sparse[idx]))
            result.set_bit(mask.sparse[idx]);
      }
      else
      {
        result.mask.dense = new DT(*mask.dense);
        result.sparse_size = MAX_SPARSE+1;
        if (rhs.is_sparse())
        {
          for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
            result.unset_bit(rhs.mask.sparse[idx]);
        }
        else
        {
          (*result.mask.dense) -= (*rhs.mask.dense);
          result.sparsify();
        }
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR>&
            CompoundBitMask<DT,BLOAT,BIDIR>::operator-=(
                                    const CompoundBitMask<DT,BLOAT,BIDIR> &rhs)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        unsigned offset = 0;
        for (unsigned idx = 0; idx < sparse_size; idx++)
        {
          if (!rhs.is_set(mask.sparse[idx]))
          {
            if (offset != idx)
              mask.sparse[offset++] = mask.sparse[idx];
            else
              offset++;
          }
        }
        sparse_size = offset;
      }
      else
      {
        if (rhs.is_sparse())
        {
          for (unsigned idx = 0; idx < rhs.sparse_size; idx++)
            mask.dense->unset_bit(rhs.mask.sparse[idx]);
        }
        else
          (*mask.dense) -= (*rhs.mask.dense);
        sparsify();
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR> 
              CompoundBitMask<DT,BLOAT,BIDIR>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<DT,BLOAT,BIDIR> result;
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if ((mask.sparse[idx]+shift) < DT::MAXSIZE)
            result.set_bit(mask.sparse[idx] + shift);
      }
      else
      {
        result.mask.dense = new DT(*mask.dense);
        result.sparse_size = MAX_SPARSE+1;
        (*result.mask.dense) <<= shift;
        result.sparsify();
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR> 
              CompoundBitMask<DT,BLOAT,BIDIR>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<DT,BLOAT,BIDIR> result;
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (shift <= mask.sparse[idx])
            result.set_bit(mask.sparse[idx] - shift);
      }
      else
      {
        result.mask.dense = new DT(*mask.dense);
        result.sparse_size = MAX_SPARSE+1;
        (*result.mask.dense) >>= shift;
        result.sparsify();
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR>& 
                   CompoundBitMask<DT,BLOAT,BIDIR>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        unsigned offset = 0;
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if ((mask.sparse[idx] + shift) < DT::MAXSIZE)
            mask.sparse[offset++] = mask.sparse[idx] + shift;
        sparse_size = offset;
      }
      else
      {
        (*mask.dense) <<= shift;
        sparsify();
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline CompoundBitMask<DT,BLOAT,BIDIR>& 
                   CompoundBitMask<DT,BLOAT,BIDIR>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        unsigned offset = 0;
        for (unsigned idx = 0; idx < sparse_size; idx++)
          if (shift <= mask.sparse[idx])
            mask.sparse[offset++] = mask.sparse[idx] - shift;
        sparse_size = offset;
      }
      else
      {
        (*mask.dense) >>= shift;
        sparsify();
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline uint64_t CompoundBitMask<DT,BLOAT,BIDIR>::get_hash_key(void) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        uint64_t result = 0;
        for (unsigned idx = 0; idx < sparse_size; idx++)
          result |= (1ULL << (mask.sparse[idx] % (8*sizeof(result))));
        return result;
      }
      else
        return mask.dense->get_hash_key();
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR> template<typename ST>
    inline void CompoundBitMask<DT,BLOAT,BIDIR>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sparse_size);
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          rez.serialize(mask.sparse[idx]);
      }
      else
        mask.dense->serialize(rez);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR> template<typename D>
    inline void CompoundBitMask<DT,BLOAT,BIDIR>::deserialize(D &derez)
    //-------------------------------------------------------------------------
    {
      const bool was_dense = !is_sparse();
      derez.deserialize(sparse_size);
      if (is_sparse())
      {
        if (was_dense)
          delete mask.dense;
        for (unsigned idx = 0; idx < sparse_size; idx++)
          derez.deserialize(mask.sparse[idx]);
      }
      else
      {
        if (!was_dense)
          mask.dense = new DT(); 
        mask.dense->deserialize(derez);
      }
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR> template<typename FUNC>
    inline void CompoundBitMask<DT,BLOAT,BIDIR>::map(FUNC &functor) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        for (unsigned idx = 0; idx < sparse_size; idx++)
          functor.apply(mask.sparse[idx]);
      }
      else
        mask.dense->map(functor);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline char* CompoundBitMask<DT,BLOAT,BIDIR>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
      {
        char *result = (char*)malloc(1024*sizeof(char));
        sprintf(result,"Compound Sparse %d:", sparse_size);
        for (unsigned idx = 0; idx < sparse_size; idx++)
        {
          char temp[64];
          sprintf(temp, " %d", mask.sparse[idx]);
          strcat(result,temp);
        }
        return result;
      }
      else
        return mask.dense->to_string();
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline unsigned CompoundBitMask<DT,BLOAT,BIDIR>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (is_sparse())
        return sparse_size;
      else
        return mask.dense->pop_count();
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    /*static*/ inline unsigned CompoundBitMask<DT,BLOAT,BIDIR>::pop_count(
                                   const CompoundBitMask<DT,BLOAT,BIDIR> &mask)
    //-------------------------------------------------------------------------
    {
      return mask.pop_count();
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline bool CompoundBitMask<DT,BLOAT,BIDIR>::is_sparse(void) const
    //-------------------------------------------------------------------------
    {
      return (sparse_size <= MAX_SPARSE);
    }

    //-------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void CompoundBitMask<DT,BLOAT,BIDIR>::sparsify(void)
    //-------------------------------------------------------------------------
    {
      if (!BIDIR)
        return;
      if (MAX_SPARSE < mask.dense->pop_count())
        return;
      DT *oldmask = mask.dense;
      sparse_size = 0;
      for (unsigned idx = 0; idx < DT::BIT_ELMTS; idx++)
      {
        if ((*oldmask)[idx])
        {
          unsigned value = idx * DT::ELEMENT_SIZE;
          for (unsigned i = 0; i < DT::ELEMENT_SIZE; i++, value++)
            if (oldmask->is_set(value))
              mask.sparse[sparse_size++] = value;
        }
      }
      delete oldmask;
    }

#endif // __BITMASK_H__
