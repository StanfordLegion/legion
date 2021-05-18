/* Copyright 2021 Stanford University, NVIDIA Corporation
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

#include <vector>
#include <set>

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
      template<bool READ_ONLY>
      class SSEView {
      public:
        inline SSEView(uint64_t *base, unsigned index) 
          : ptr(base + ((sizeof(__m128d)/sizeof(uint64_t))*index)) { }
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
        uint64_t *const ptr;
      };
      template<>
      class SSEView<true> {
      public:
        inline SSEView(const uint64_t *base, unsigned index) 
          : ptr(base + ((sizeof(__m128d)/sizeof(uint64_t))*index)) { }
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
        const uint64_t *const ptr;
      };
#endif
#ifdef __AVX__
      template<bool READ_ONLY>
      class AVXView {
      public:
        inline AVXView(uint64_t *base, unsigned index) 
          : ptr(base + ((sizeof(__m256d)/sizeof(uint64_t))*index)) { }
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
        uint64_t *const ptr;
      };
      template<>
      class AVXView<true> {
      public:
        inline AVXView(const uint64_t *base, unsigned index) 
          : ptr(base + ((sizeof(__m256d)/sizeof(uint64_t))*index)) { }
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
        const uint64_t *const ptr;
      };
#endif
#ifdef __ALTIVEC__
      template<bool READ_ONLY>
      class PPCView {
      public:
        inline PPCView(uint64_t *base, unsigned index) 
          : ptr(base + ((sizeof(__vector double)/sizeof(uint64_t))*index)) { }
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
        uint64_t *const ptr;
      };
      template<>
      class PPCView<true> {
      public:
        inline PPCView(const uint64_t *base, unsigned index) 
          : ptr(base + ((sizeof(__vector double)/sizeof(uint64_t))*index)) { }
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
        const uint64_t *const ptr;
      };
#endif

      // Help with safe type-punning of bit representations
      // This is only because C++ is a stupid fucking language
      // and doesn't even follow the same semantics as C's union
      // As a result we explode our compilation time and generate worse code
      template<int MAX>
      struct BitVector {
      public:
#ifdef __SSE2__
        inline SSEView<false> sse_view(unsigned index)
          { return SSEView<false>(bit_vector, index); }
        inline SSEView<true> sse_view(unsigned index) const
          { return SSEView<true>(bit_vector, index); }
#endif
#ifdef __AVX__
        inline AVXView<false> avx_view(unsigned index)
          { return AVXView<false>(bit_vector, index); }
        inline AVXView<true> avx_view(unsigned index) const
          { return AVXView<true>(bit_vector, index); }
#endif
#ifdef __ALTIVEC__
        inline PPCView<false> ppc_view(unsigned index)
          { return PPCView<false>(bit_vector, index); }
        inline PPCView<true> ppc_view(unsigned index) const
          { return PPCView<true>(bit_vector, index); }
#endif
      public:
        static_assert((MAX % 64) == 0, "Bad bitmask size");
        uint64_t bit_vector[MAX/64];
      };
    };

    /////////////////////////////////////////////////////////////
    // Bit Mask 
    /////////////////////////////////////////////////////////////
    template<typename T, unsigned int MAX,
             unsigned int SHIFT, unsigned int MASK>
    class BitMask : public BitMaskHelp::Heapify<BitMask<T,MAX,SHIFT,MASK> > {
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
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(
            const BitMask<unsigned,MAX,SHIFT,MASK> &mask);
      static inline int pop_count(
            const BitMask<unsigned long,MAX,SHIFT,MASK> &mask);
      static inline int pop_count(
            const BitMask<unsigned long long,MAX,SHIFT,MASK> &mask);
    protected:
      T bit_vector[MAX/(8*sizeof(T))];
    public:
      static const unsigned ELEMENT_SIZE = 8*sizeof(T);
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
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
      explicit TLBitMask(T init = 0);
      TLBitMask(const TLBitMask &rhs);
      ~TLBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(
            const TLBitMask<unsigned,MAX,SHIFT,MASK> &mask);
      static inline int pop_count(
            const TLBitMask<unsigned long,MAX,SHIFT,MASK> &mask);
      static inline int pop_count(
            const TLBitMask<unsigned long long,MAX,SHIFT,MASK> &mask);
    protected:
      T bit_vector[MAX/(8*sizeof(T))];
      T sum_mask;
    public:
      static const unsigned ELEMENT_SIZE = 8*sizeof(T);
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
    };

#ifdef __SSE2__
    /////////////////////////////////////////////////////////////
    // SSE Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) SSEBitMask 
      : public BitMaskHelp::Heapify<SSEBitMask<MAX> > {
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
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(const SSEBitMask<MAX> &mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
    };

    /////////////////////////////////////////////////////////////
    // SSE Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) SSETLBitMask
      : public BitMaskHelp::Heapify<SSETLBitMask<MAX> > {
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
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(const SSETLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__m128i value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
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
      explicit AVXBitMask(uint64_t init = 0);
      AVXBitMask(const AVXBitMask &rhs);
      ~AVXBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(const AVXBitMask<MAX> &mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
    };
    
    /////////////////////////////////////////////////////////////
    // AVX Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(32) AVXTLBitMask
      : public BitMaskHelp::Heapify<AVXTLBitMask<MAX> > {
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
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(const AVXTLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__m256i value);
      static inline uint64_t extract_mask(__m256d value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
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
      explicit PPCBitMask(uint64_t init = 0);
      PPCBitMask(const PPCBitMask &rhs);
      ~PPCBitMask(void);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(const PPCBitMask<MAX> &mask);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
    };
    
    /////////////////////////////////////////////////////////////
    // PPC Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class alignas(16) PPCTLBitMask
      : public BitMaskHelp::Heapify<PPCTLBitMask<MAX> > {
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
      inline int find_index_set(int index) const;
      inline int find_next_set(int start) const;
      inline void clear(void);
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
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(const PPCTLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__vector unsigned long long value);
    protected:
      BitMaskHelp::BitVector<MAX> bits;
      uint64_t sum_mask;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
    };
#endif // __ALTIVEC__

    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    class CompoundBitMask {
    public:
      static const int CNT_BITS = 8; // default this to 8 for now
      static const uint64_t CNT_MASK = (1UL << CNT_BITS) - 1UL;
    public:
      static const int VAL_BITS = STATIC_LOG2(MAX);
      static const uint64_t VAL_MASK = (1UL << VAL_BITS) - 1UL;
    public:
      static const int MAX_CNT = 
        (WORDS*8*sizeof(uint64_t) - CNT_BITS)/VAL_BITS;
      static const int SPARSE_CNT = MAX_CNT+1;
      static const int DENSE_CNT = SPARSE_CNT+1;
    public:
      static const int WORD_SIZE = 64;
      static const int WORD_BITS = STATIC_LOG2(WORD_SIZE);
      static const uint64_t WORD_MASK = 0x3F;
    public:
      static const bool OVERLAP = ((WORD_SIZE % VAL_BITS) != 0) ||
                          (((WORD_SIZE - CNT_BITS) % VAL_BITS) != 0);
    public:
      typedef std::set<int> SparseSet;
      // Size of an STL Node object in bytes
      // This value is approximated over different STL
      // implementations but in general it should be close
      static const size_t STL_SET_NODE_SIZE = 32;
      static const int SPARSE_MAX = 
        sizeof(BITMASK) / (sizeof(int) + sizeof(STL_SET_NODE_SIZE));
    public:
      explicit CompoundBitMask(uint64_t init = 0);
      CompoundBitMask(const CompoundBitMask &rhs);
      ~CompoundBitMask(void);
    public:
      inline int get_count(void) const;
      inline void set_count(int size);
      inline SparseSet* get_sparse(void) const;
      inline void set_sparse(SparseSet *ptr);
      inline BITMASK* get_dense(void) const;
      inline void set_dense(BITMASK *ptr);
      template<bool CAN_OVERLAP>
      inline int get_value(int idx) const;
      template<bool CAN_OVERLAP>
      inline void set_value(int idx, int value);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
      inline int find_index_set(int index) const;
      inline void clear(void);
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
      template<typename DT>
      inline void deserialize(DT &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      inline int pop_count(void) const;
      static inline int pop_count(const 
                        CompoundBitMask<BITMASK,MAX,WORDS> &mask);
    protected:
      uint64_t bits[WORDS];
    public:
      static const int ELEMENTS = 1;
      static const int ELEMENT_SIZE = MAX;
    };

    /////////////////////////////////////////////////////////////
    // Integer Set 
    /////////////////////////////////////////////////////////////
    template<typename IT/*int type*/, typename DT/*dense type (BitMask)*/,
             bool BIDIR=false/*(bi-directional)*/>
    class IntegerSet {
    public:
      // Size of an STL Node object in bytes
      // This value is approximated over different STL
      // implementations but in general it should be close
      static const size_t STL_SET_NODE_SIZE = 32;
    public:
      // Need to inherit form LegionHeapify for alignment
      struct DenseSet : public BitMaskHelp::Heapify<DenseSet> {
      public:
        DT set;
      };
      struct UnionFunctor {
      public:
        UnionFunctor(IntegerSet &t) : target(t) { }
      public:
        inline void apply(IT value) { target.add(value); }
      private:
        IntegerSet &target;
      };
      struct IntersectFunctor {
      public:
        IntersectFunctor(IntegerSet &t, const IntegerSet &r)
          : target(t), rhs(r) { }
      public:
        inline void apply(IT value) 
          { if (rhs.contains(value)) target.add(value); }
      private:
        IntegerSet &target;
        const IntegerSet &rhs;
      };
      struct DifferenceFunctor {
      public:
        DifferenceFunctor(IntegerSet &t) : target(t) { }
      public:
        inline void apply(IT value) { target.remove(value); }
      private:
        IntegerSet &target;
      };
    public:
      IntegerSet(void);
      IntegerSet(const IntegerSet &rhs);
      ~IntegerSet(void);
    public:
      IntegerSet& operator=(const IntegerSet &rhs);
    public:
      inline bool contains(IT index) const;
      inline void add(IT index);
      inline void remove(IT index);
      inline IT find_first_set(void) const;
      inline IT find_index_set(int index) const;
      // The functor class must have an 'apply' method that
      // take one argument of type IT. This method will map
      // the functor over all the entries in the set.
      template<typename FUNCTOR>
      inline void map(FUNCTOR &functor) const;
    public:
      template<typename ST>
      inline void serialize(ST &rez) const;
      template<typename ZT>
      inline void deserialize(ZT &derez);
    public:
      inline IntegerSet operator|(const IntegerSet &rhs) const;
      inline IntegerSet operator&(const IntegerSet &rhs) const;
      inline IntegerSet operator-(const IntegerSet &rhs) const;
    public:
      inline IntegerSet& operator|=(const IntegerSet &rhs);
      inline IntegerSet& operator&=(const IntegerSet &rhs);
      inline IntegerSet& operator-=(const IntegerSet &rhs);
    public:
      inline bool operator!(void) const;
      inline bool empty(void) const { return !*this; }
      inline size_t size(void) const;
      inline void clear(void);
      inline IntegerSet& swap(IntegerSet &rhs);
    protected:
      bool sparse;
      union {
        typename std::set<IT>* sparse;
        DenseSet*              dense;
      } set_ptr;
    };

  namespace BitMaskHelp {

    /**
     * \struct BitMaskStaticAssert
     * Help with static assertions.
     */
    template<bool> struct StaticAssert;
    template<> struct StaticAssert<true> { };
#define BITMASK_STATIC_ASSERT(condition) \
    do { BitMaskHelp::StaticAssert<(condition)>(); } while (0)

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
      BITMASK_STATIC_ASSERT((SIZE % ALIGNMENT) == 0);
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

#define BIT_ELMTS (MAX/(8*sizeof(T)))
    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    BitMask<T,MAX,SHIFT,MASK>::BitMask(T init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
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
      BITMASK_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
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
    inline int BitMask<T,MAX,SHIFT,MASK>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bit_vector[idx]);
        if (index <= local)
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
    inline int BitMask<T,MAX,SHIFT,MASK>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool BitMask<T,MAX,SHIFT,MASK>::operator!(void) const
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
        T carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    inline char* BitMask<T,MAX,SHIFT,MASK>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int BitMask<T,MAX,SHIFT,MASK>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      int result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcount(bit_vector[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline int BitMask<T,MAX,SHIFT,MASK>::pop_count(
                                  const BitMask<unsigned,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
    /*static*/ inline int BitMask<T,MAX,SHIFT,MASK>::pop_count(
                            const BitMask<unsigned long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
    /*static*/ inline int BitMask<T,MAX,SHIFT,MASK>::pop_count(
                        const BitMask<unsigned long long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
      BITMASK_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        bit_vector[idx] = init;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    TLBitMask<T,MAX,SHIFT,MASK>::TLBitMask(const TLBitMask &rhs)
      : sum_mask(rhs.sum_mask)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
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
    inline int TLBitMask<T,MAX,SHIFT,MASK>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bit_vector[idx]);
        if (index <= local)
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
    inline int TLBitMask<T,MAX,SHIFT,MASK>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool TLBitMask<T,MAX,SHIFT,MASK>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      // Here is another great reason to have sum mask
      return (sum_mask == 0);
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
        T carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline char* TLBitMask<T,MAX,SHIFT,MASK>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline int TLBitMask<T,MAX,SHIFT,MASK>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      int result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcount(bit_vector[idx]);
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
    /*static*/ inline int TLBitMask<T,MAX,SHIFT,MASK>::pop_count(
                               const TLBitMask<unsigned,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      if (!mask.sum_mask)
        return 0;
      int result = 0;
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
    /*static*/ inline int TLBitMask<T,MAX,SHIFT,MASK>::pop_count(
                          const TLBitMask<unsigned long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      if (!mask.sum_mask)
        return 0;
      int result = 0;
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
    /*static*/ inline int TLBitMask<T,MAX,SHIFT,MASK>::pop_count(
                     const TLBitMask<unsigned long long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      if (!mask.sum_mask)
        return 0;
      int result = 0;
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
#undef BIT_ELMTS

#ifdef __SSE2__
#define SSE_ELMTS (MAX/128)
#define BIT_ELMTS (MAX/64)
    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    SSEBitMask<MAX>::SSEBitMask(uint64_t init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
    inline int SSEBitMask<MAX>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bits.bit_vector[idx]);
        if (index <= local)
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
    inline int SSEBitMask<MAX>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool SSEBitMask<MAX>::operator!(void) const
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
        uint64_t carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    template<unsigned int MAX>
    inline char* SSEBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSEBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      int result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(bits.bit_vector[idx]);
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
    /*static*/ inline int SSEBitMask<MAX>::pop_count(
                                                   const SSEBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
    template<unsigned int MAX>
    SSETLBitMask<MAX>::SSETLBitMask(uint64_t init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
    inline int SSETLBitMask<MAX>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bits.bit_vector[idx]);
        if (index <= local)
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
    inline int SSETLBitMask<MAX>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool SSETLBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      // A great reason to have a summary mask
      return (sum_mask == 0);
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
        uint64_t carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    template<unsigned int MAX>
    inline char* SSETLBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int SSETLBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      int result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(bits.bit_vector[idx]);
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
    /*static*/ inline int SSETLBitMask<MAX>::pop_count(
                                                 const SSETLBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
#undef BIT_ELMTS
#undef SSE_ELMTS
#endif // __SSE2__

#ifdef __AVX__
#define AVX_ELMTS (MAX/256)
#define BIT_ELMTS (MAX/64)
    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    AVXBitMask<MAX>::AVXBitMask(uint64_t init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % 256) == 0);
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
      BITMASK_STATIC_ASSERT((MAX % 256) == 0);
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
    inline int AVXBitMask<MAX>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bits.bit_vector[idx]);
        if (index <= local)
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
    inline int AVXBitMask<MAX>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool AVXBitMask<MAX>::operator!(void) const
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
        uint64_t carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    template<unsigned int MAX>
    inline char* AVXBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      int result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(bits.bit_vector[idx]);
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
    /*static*/ inline int AVXBitMask<MAX>::pop_count(
                                                   const AVXBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
    template<unsigned int MAX>
    AVXTLBitMask<MAX>::AVXTLBitMask(uint64_t init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % 256) == 0);
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
      BITMASK_STATIC_ASSERT((MAX % 256) == 0);
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
    inline int AVXTLBitMask<MAX>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bits.bit_vector[idx]);
        if (index <= local)
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
    inline int AVXTLBitMask<MAX>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool AVXTLBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      // A great reason to have a summary mask
      return (sum_mask == 0);
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
        uint64_t carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    template<unsigned int MAX>
    inline char* AVXTLBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int AVXTLBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      int result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(bits.bit_vector[idx]);
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
    /*static*/ inline int AVXTLBitMask<MAX>::pop_count(
                                                 const AVXTLBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
#undef BIT_ELMTS
#undef AVX_ELMTS
#endif // __AVX__

#ifdef __ALTIVEC__
#define PPC_ELMTS (MAX/128)
#define BIT_ELMTS (MAX/64)
    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    PPCBitMask<MAX>::PPCBitMask(uint64_t init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
    inline int PPCBitMask<MAX>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bits.bit_vector[idx]);
        if (index <= local)
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
    inline int PPCBitMask<MAX>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool PPCBitMask<MAX>::operator!(void) const
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
        uint64_t carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    template<unsigned int MAX>
    inline char* PPCBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      int result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(bits.bit_vector[idx]);
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
    /*static*/ inline int PPCBitMask<MAX>::pop_count(
                                                   const PPCBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
    template<unsigned int MAX>
    PPCTLBitMask<MAX>::PPCTLBitMask(uint64_t init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
      BITMASK_STATIC_ASSERT((MAX % 128) == 0);
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
    inline int PPCTLBitMask<MAX>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
      int offset = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        int local = __builtin_popcount(bits.bit_vector[idx]);
        if (index <= local)
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
    inline int PPCTLBitMask<MAX>::find_next_set(int start) const
    //-------------------------------------------------------------------------
    {
      if (start < 0)
        start = 0;
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
    inline bool PPCTLBitMask<MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      // A great reason to have a summary mask
      return (sum_mask == 0);
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
        uint64_t carry_mask = 0;
        for (unsigned idx = 0; idx < local; idx++)
          carry_mask |= (1 << idx);
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
    template<unsigned int MAX>
    inline char* PPCTLBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelp::to_string(bits.bit_vector, MAX);
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline int PPCTLBitMask<MAX>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      if (!sum_mask)
        return 0;
      int result = 0;
#ifndef VALGRIND
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(bits.bit_vector[idx]);
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
    /*static*/ inline int PPCTLBitMask<MAX>::pop_count(
                                                 const PPCTLBitMask<MAX> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
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
    template<unsigned int MAX>
    /*static*/ inline uint64_t PPCTLBitMask<MAX>::extract_mask(
                                             __vector unsigned long long value)
    //-------------------------------------------------------------------------
    {
      uint64_t left = value[0];
      uint64_t right = value[1];
      return (left | right);
    }
#undef BIT_ELMTS
#undef PPC_ELMTS
#endif // __ALTIVEC__

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    CompoundBitMask<BITMASK,MAX,WORDS>::CompoundBitMask(uint64_t init)
    //-------------------------------------------------------------------------
    {
      BITMASK_STATIC_ASSERT(WORDS >= 2);
      BITMASK_STATIC_ASSERT(VAL_BITS <= (8*sizeof(uint64_t)));
      BITMASK_STATIC_ASSERT((1 << VAL_BITS) >= MAX);
      if (init == 0)
      {
        set_count(0);
      }
      else
      {
        set_count(DENSE_CNT);
        set_dense(new BITMASK(init));
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    CompoundBitMask<BITMASK,MAX,WORDS>::CompoundBitMask(
                                                    const CompoundBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      int rhs_count = rhs.get_count();
      if (rhs_count == SPARSE_CNT)
      {
        set_count(SPARSE_CNT);
        set_sparse(new SparseSet(*rhs.get_sparse()));
      }
      else if (rhs_count == DENSE_CNT)
      {
        set_count(DENSE_CNT); 
        set_dense(new BITMASK(*rhs.get_dense()));
      }
      else
      {
        for (unsigned idx = 0; idx < WORDS; idx++)
          bits[idx] = rhs.bits[idx];
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    CompoundBitMask<BITMASK,MAX,WORDS>::~CompoundBitMask(void) 
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == SPARSE_CNT)
        delete get_sparse();
      else if (count == DENSE_CNT)
        delete get_dense();
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline int CompoundBitMask<BITMASK,MAX,WORDS>::get_count(void) const
    //-------------------------------------------------------------------------
    {
      return (CNT_MASK & bits[0]); 
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::set_count(int size) 
    //-------------------------------------------------------------------------
    {
      bits[0] = (size & CNT_MASK) | (bits[0] & ~CNT_MASK);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline typename CompoundBitMask<BITMASK,MAX,WORDS>::SparseSet* 
                    CompoundBitMask<BITMASK,MAX,WORDS>::get_sparse(void) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_count() == SPARSE_CNT);
#endif
      return reinterpret_cast<SparseSet*>(bits[1]);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::set_sparse(SparseSet *ptr)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_count() == SPARSE_CNT);
#endif
      bits[1] = reinterpret_cast<uint64_t>(ptr);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline BITMASK* CompoundBitMask<BITMASK,MAX,WORDS>::get_dense(void) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_count() == DENSE_CNT);
#endif
      return reinterpret_cast<BITMASK*>(bits[1]);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::set_dense(BITMASK *ptr) 
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_count() == DENSE_CNT);
#endif
      bits[1] = reinterpret_cast<uint64_t>(ptr);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
      template<bool CAN_OVERLAP>
    inline int CompoundBitMask<BITMASK,MAX,WORDS>::get_value(int idx) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < MAX_CNT);
#endif
      int start_bit = CNT_BITS + idx*VAL_BITS;
      int start_index = start_bit >> WORD_BITS;
      start_bit &= WORD_MASK;
      if (CAN_OVERLAP)
      {
        // See if we straddle words 
        if ((start_bit + (VAL_BITS-1)) < WORD_SIZE)
        {
#ifdef DEBUG_LEGION
          assert(start_index < WORDS);
#endif
          // Common case, all in one word
          return (bits[start_index] & (VAL_MASK << start_bit)) >> start_bit;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert((start_index+1) < WORDS);
#endif
          // Value split across words
          int first_word_bits = WORD_SIZE - start_bit;
          // Okay to shit off the end here
          int result = 
            int((bits[start_index] & (VAL_MASK << start_bit)) >> start_bit);
          // Okay to shift off the end here
          uint64_t next_mask = VAL_MASK >> first_word_bits;
          result |= int((bits[start_index+1] & next_mask) << first_word_bits);
          return result;
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(start_index < WORDS);
#endif
        return (bits[start_index] & (VAL_MASK << start_bit)) >> start_bit;
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
      template<bool CAN_OVERLAP>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::set_value(int idx, 
                                                              int value) 
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < MAX_CNT);
#endif
      // Cast up the value
      uint64_t up_value = value;
      int start_bit = CNT_BITS + idx*VAL_BITS;
      int start_index = start_bit >> WORD_BITS;
      start_bit &= WORD_MASK;
      uint64_t negative_mask = ~(VAL_MASK << start_bit);
      if (CAN_OVERLAP)
      {
        // See if we straddle words
        if ((start_bit + (VAL_BITS-1)) < WORD_SIZE)
        {
#ifdef DEBUG_LEGION
          assert(start_index < WORDS);
#endif
          // Common case, all in one word
          bits[start_index] = (bits[start_index] & negative_mask) |
                              (up_value << start_bit);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert((start_index+1) < WORDS);
#endif
          // Value split across words
          // Okay to shift off the end here
          bits[start_index] = (bits[start_index] & negative_mask) |
                              (up_value << start_bit);
          int first_word_bits = WORD_SIZE - start_bit;
          // Okay to shift off the end here
          uint64_t next_negative = ~(VAL_MASK >> first_word_bits);
          bits[start_index+1] = (bits[start_index+1] & next_negative) |
                                (up_value >> first_word_bits);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(start_index < WORDS);
#endif
        bits[start_index] = (bits[start_index] & negative_mask) |
                            (up_value << start_bit);
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count < MAX_CNT)
      {
        // Check to make sure it isn't already in our list
        for (int idx = 0; idx < count; idx++)
          if (get_value<OVERLAP>(idx) == bit)
            return;
        // Add it at the next available location
        set_value<OVERLAP>(count, bit); 
        set_count(count+1);
      }
      else if (count == MAX_CNT)
      {
        // We've maxed out, go to sparse or dense 
        if (SPARSE_MAX > MAX_CNT)
        {
          SparseSet *next = new SparseSet();
          next->insert(bit);
          for (int idx = 0; idx < MAX_CNT; idx++)
            next->insert(get_value<OVERLAP>(idx));
          set_count(SPARSE_CNT);
          set_sparse(next);
        }
        else
        {
          BITMASK *next = new BITMASK();
          next->set_bit(bit);
          for (int idx = 0; idx < MAX_CNT; idx++)
            next->set_bit(get_value<OVERLAP>(idx));
          set_count(DENSE_CNT);
          set_dense(next);
        }
      }
      else if (count == SPARSE_CNT)
      {
        // sparse case 
        SparseSet *sparse = get_sparse();
        if (sparse->size() == SPARSE_MAX)
        {
          // upgrade to dense 
          BITMASK *next = new BITMASK();
          for (SparseSet::const_iterator it = sparse->begin();
                it != sparse->end(); it++)
            next->set_bit(*it);
          delete sparse;
          set_count(DENSE_CNT);
          set_dense(next);
        }
        else // otherwise just insert
          sparse->insert(bit);
      }
      else
      {
        // Dense case
        get_dense()->set_bit(bit);
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::unset_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
      int count = get_count(); 
      if (count == 0)
        return;
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        sparse->erase(bit);
        if (sparse->size() == MAX_CNT)
        {
          set_count(MAX_CNT);
          unsigned idx = 0;
          for (SparseSet::const_iterator it = sparse->begin();
                it != sparse->end(); it++, idx++)
            set_value<OVERLAP>(idx, *it);   
          delete sparse;
        }
      }
      else if (count == DENSE_CNT)
      {
        BITMASK *dense = get_dense();
        dense->unset_bit(bit);
        // If dense is empty come back to zero
        if (!(*dense))
        {
          delete dense;
          set_count(0);
        }
      }
      else
      {
        // Iterate through our elements and see if we find it
        int found_idx = -1;
        for (int idx = 0; idx < count; idx++)
        {
          if (get_value<OVERLAP>(idx) == bit)
          {
            found_idx = idx;
            break;
          }
        }
        if (found_idx >= 0)
        {
          // If we found it shift everyone down and decrement the count
          for (int idx = found_idx; idx < (count-1); idx++)
          {
            int next = get_value<OVERLAP>(idx+1);
            set_value<OVERLAP>(idx, next);
          }
          set_count(count-1);
        }
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::assign_bit(unsigned bit,  
                                                               bool val)
    //-------------------------------------------------------------------------
    {
      if (val)
        set_bit(bit);
      else
        unset_bit(bit);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline bool CompoundBitMask<BITMASK,MAX,WORDS>::is_set(unsigned bit) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == 0)
        return false;
      else if (count == DENSE_CNT)
        return get_dense()->is_set(bit);
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        return (sparse->find(bit) != sparse->end());
      }
      else
      {
        // Iterate through our elements and see if we have it
        for (int idx = 0; idx < count; idx++)
        {
          if (get_value<OVERLAP>(idx) == bit)
            return true;
        }
      }
      return false;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline int CompoundBitMask<BITMASK,MAX,WORDS>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == 0)
        return -1;
      if (count == SPARSE_CNT)
        return (*(get_sparse()->begin()));
      if (count == DENSE_CNT)
        return get_dense()->find_first_set();
      return get_value<OVERLAP>(0);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline int CompoundBitMask<BITMASK,MAX,WORDS>::find_index_set(
                                                               int index) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == DENSE_CNT)
        return get_dense()->find_index_set(index);
      if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        if (index >= sparse->size())
          return -1;
        SparseSet::const_iterator it = sparse->begin();
        while (index > 0)
        {
          index--;
          it++;
        }
        return (*it);
      }
      if (index >= count)
        return -1;
      return get_value<OVERLAP>(index);
    }
    
    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::clear(void)
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == SPARSE_CNT)
        delete get_sparse();
      if (count == DENSE_CNT)
        delete get_dense();
      set_count(0);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline bool CompoundBitMask<BITMASK,MAX,WORDS>::operator==(
                                              const CompoundBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      int rhs_count = rhs.get_count();
      if (count != rhs_count)
        return false;
      // If they are dense see if they are equal
      if (count == DENSE_CNT)
        return (*get_dense() == *rhs.get_dense());
      if (count == SPARSE_CNT)
        return (*get_sparse() == *rhs.get_sparse());
      // See if there are all matching bits
      for (int idx = 0; idx < count; idx++)
        if (!rhs.is_set(get_value<OVERLAP>(idx)))
          return false;
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline bool CompoundBitMask<BITMASK,MAX,WORDS>::operator<(
                                              const CompoundBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      int rhs_count = rhs.get_count();
      if (count < rhs_count)
        return true;
      else if (count > rhs_count)
        return false;
      // Otherwise they are equal in size, see if they are actually equal
      if (count == DENSE_CNT)
        return (*get_dense() < *rhs.get_dense());
      if (count == SPARSE_CNT)
        return (*get_sparse() < *rhs.get_sparse());
      // Now we just have indexes, nothing good to do here, we
      // need to sort them so put them in stl sets and compare them
      std::set<int> local_set, rhs_set;
      for (int idx = 0; idx < count; idx++)
        local_set.insert(get_value<OVERLAP>(idx));
      for (int idx = 0; idx < rhs_count; idx++)
        rhs_set.insert(rhs.get_value<OVERLAP>(idx));
      return (local_set < rhs_set);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline bool CompoundBitMask<BITMASK,MAX,WORDS>::operator!=(
                                              const CompoundBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS>& 
      CompoundBitMask<BITMASK,MAX,WORDS>::operator=(const CompoundBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      int rhs_count = rhs.get_count();
      if (count != rhs_count)
      {
        if (count == DENSE_CNT)
        {
          // Free our dense count and copy over bits
          delete get_dense();
          if (rhs_count == SPARSE_CNT)
          {
            set_count(SPARSE_CNT);
            set_sparse(new SparseSet(*rhs.get_sparse()));
          }
          else
          {
            for (int idx = 0; idx < WORDS; idx++)
              bits[idx] = rhs.bits[idx];
          }
        }
        else if (count == SPARSE_CNT)
        {
          // Free our set
          delete get_sparse();
          if (rhs_count == DENSE_CNT)
          {
            // If the rhs is dense copy it over
            set_count(DENSE_CNT);
            set_dense(new BITMASK(*rhs.get_dense()));
          }
          else
          {
            // Otherwise it is just bit, so copy it over
            for (int idx = 0; idx < WORDS; idx++)
              bits[idx] = rhs.bits[idx];
          }
        }
        else
        {
          // We are just bits, see if we need to copy over a set or a mask
          if (rhs_count == SPARSE_CNT)
          {
            set_count(SPARSE_CNT);
            set_sparse(new SparseSet(*rhs.get_sparse()));
          }
          else if (rhs_count == DENSE_CNT)
          {
            set_count(DENSE_CNT);
            set_dense(new BITMASK(*rhs.get_dense()));
          }
          else
          {
            for (int idx = 0; idx < WORDS; idx++)
              bits[idx] = rhs.bits[idx];
          }
        }
      }
      else
      {
        if (count == DENSE_CNT)
        {
          // both dense, just copy over dense masks
          *get_dense() = *rhs.get_dense();
        }
        else if (count == SPARSE_CNT)
        {
          *get_sparse() = *rhs.get_sparse();
        }
        else
        {
          // both not dense, copy over bits
          for (int idx = 0; idx < WORDS; idx++)
            bits[idx] = rhs.bits[idx];
        }
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS> 
                      CompoundBitMask<BITMASK,MAX,WORDS>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<BITMASK,MAX,WORDS> result;
      int count = get_count();
      if (count == DENSE_CNT)
      {
        BITMASK next = ~(*get_dense());
        if (!!next)
        {
          result.set_count(DENSE_CNT);
          result.set_dense(new BITMASK(next));
        }
      }
      else if (count == SPARSE_CNT)
      {
        BITMASK *dense = new BITMASK(0xFFFFFFFFFFFFFFFF);
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
          dense->unset_bit(*it);
        result.set_count(DENSE_CNT);
        result.set_dense(dense);
      }
      else
      {
        BITMASK *dense = new BITMASK(0xFFFFFFFFFFFFFFFF); 
        for (int idx = 0; idx < count; idx++)
          dense->unset_bit(get_value<OVERLAP>(idx));
        result.set_count(DENSE_CNT);
        result.set_dense(dense);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS> 
      CompoundBitMask<BITMASK,MAX,WORDS>::operator|(
                                              const CompoundBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<BITMASK,MAX,WORDS> result;
      int count = get_count();
      int rhs_count = rhs.get_count();
      if (count == DENSE_CNT)
      {
        if (rhs_count == DENSE_CNT)
        {
          BITMASK *next = new BITMASK((*get_dense()) | (*rhs.get_dense()));
          result.set_count(DENSE_CNT);
          result.set_dense(next);
        }
        else if (rhs_count == SPARSE_CNT)
        {
          BITMASK *next = new BITMASK(*get_dense());
          SparseSet *other = rhs.get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
            next->set_bit(*it);
          result.set_count(DENSE_CNT);
          result.set_dense(next);
        }
        else
        {
          BITMASK *next = new BITMASK(*get_dense());
          for (int idx = 0; idx < rhs_count; idx++)
            next->set_bit(rhs.get_value<OVERLAP>(idx));
          result.set_count(DENSE_CNT);
          result.set_dense(next);
        }
      }
      else if (rhs_count == DENSE_CNT)
      {
        BITMASK *next = new BITMASK(*rhs.get_dense());
        if (count == SPARSE_CNT)
        {
          SparseSet *other = get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
            next->set_bit(*it);
        }
        else
        {
          for (int idx = 0; idx < count; idx++)
            next->set_bit(get_value<OVERLAP>(idx));
        }
        result.set_count(DENSE_CNT);
        result.set_dense(next);
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *next = new SparseSet(*get_sparse());
        result.set_count(SPARSE_CNT);
        result.set_sparse(next);
        if (rhs_count == SPARSE_CNT)
        {
          SparseSet *other = rhs.get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
            result.set_bit(*it);
        }
        else
        {
          for (int idx = 0; idx < rhs_count; idx++)
            result.set_bit(rhs.get_value<OVERLAP>(idx));
        }
      }
      else if (rhs_count == SPARSE_CNT)
      {
        SparseSet *next = new SparseSet(*rhs.get_sparse());
        result.set_count(SPARSE_CNT);
        result.set_sparse(next);
        for (int idx = 0; idx < count; idx++)
          result.set_bit(get_value<OVERLAP>(idx));
      }
      else
      {
        for (int idx = 0; idx < count; idx++)
          result.set_bit(get_value<OVERLAP>(idx));
        for (int idx = 0; idx < rhs_count; idx++)
          result.set_bit(rhs.get_value<OVERLAP>(idx));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS> 
      CompoundBitMask<BITMASK,MAX,WORDS>::operator&(
                                              const CompoundBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<BITMASK,MAX,WORDS> result;
      int count = get_count();
      int rhs_count = rhs.get_count();
      if (count < SPARSE_CNT)
      {
        for (int idx = 0; idx < count; idx++) 
        {
          int bit = get_value<OVERLAP>(idx);
          if (rhs.is_set(bit))
            result.set_bit(bit);
        }
      }
      else if (rhs_count < SPARSE_CNT)
      {
        for (int idx = 0; idx < rhs_count; idx++)
        {
          int bit = rhs.get_value<OVERLAP>(idx);
          if (is_set(bit))
            result.set_bit(bit);
        }
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          if (rhs.is_set(*it))
            result.set_bit(*it);
        }
      }
      else if (rhs_count == SPARSE_CNT)
      {
        SparseSet *sparse = rhs.get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          if (is_set(*it))
            result.set_bit(*it);
        }
      }
      else
      {
        // both dense
        BITMASK next((*get_dense()) & (*rhs.get_dense()));
        if (!!next)
        {
          result.set_count(DENSE_CNT);
          result.set_dense(new BITMASK(next));
        }
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS> 
      CompoundBitMask<BITMASK,MAX,WORDS>::operator^(
                                              const CompoundBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<BITMASK,MAX,WORDS> result;
      int count = get_count();
      int rhs_count = rhs.get_count();
      if (count == DENSE_CNT)
      {
        if (rhs_count == DENSE_CNT)
        {
          // both dense
          BITMASK next((*get_dense()) ^ (*rhs.get_dense()));
          if (!!next)
          {
            result.set_count(DENSE_CNT);
            result.set_dense(new BITMASK(next));
          }
        }
        else if (rhs_count == SPARSE_CNT)
        {
          BITMASK next((*get_dense()));
          SparseSet *sparse = rhs.get_sparse();
          for (SparseSet::const_iterator it = sparse->begin();
                it != sparse->end(); it++)
          {
            if (!next.is_set(*it))
              next.set_bit(*it);
            else
              next.unset_bit(*it);
          }
          if (!!next)
          {
            result.set_count(DENSE_CNT);
            result.set_dense(new BITMASK(next));
          }
        }
        else
        {
          BITMASK next((*get_dense()));
          for (int idx = 0; idx < rhs_count; idx++)
          {
            int bit = rhs.get_value<OVERLAP>(idx);
            if (!next.is_set(bit))
              next.set_bit(bit);
            else
              next.unset_bit(bit);
          }
          if (!!next)
          {
            result.set_count(DENSE_CNT);
            result.set_dense(new BITMASK(next));
          }
        }
      }
      else if (rhs_count == DENSE_CNT)
      {
        BITMASK next(*(rhs.get_dense()));
        if (count == SPARSE_CNT)
        {
          SparseSet *sparse = get_sparse();
          for (SparseSet::const_iterator it = sparse->begin();
                it != sparse->end(); it++)
          {
            if (!next.is_set(*it))
              next.set_bit(*it);
            else
              next.unset_bit(*it);
          }
        }
        else
        {
          for (int idx = 0; idx < count; idx++)
          {
            int bit = get_value<OVERLAP>(idx);
            if (!next.is_set(bit))
              next.set_bit(bit);
            else
              next.unset_bit(bit);
          }
        }
        if (!!next)
        {
          result.set_count(DENSE_CNT);
          result.set_dense(new BITMASK(next));
        }
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *next = new SparseSet(*get_sparse());
        if (rhs_count == SPARSE_CNT)
        {
          SparseSet *other = rhs.get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
          {
            SparseSet::iterator finder = next->find(*it);
            if (finder != next->end())
              next->erase(finder);
            else
              next->insert(*it);
          }
        }
        else
        {
          for (int idx = 0; idx < rhs_count; idx++)
          {
            int bit = rhs.get_value<OVERLAP>(idx);
            SparseSet::iterator finder = next->find(bit);
            if (finder != next->end())
              next->erase(finder);
            else
              next->insert(bit);
          }
        }
        if (!next->empty())
        {
          if (next->size() <= MAX_CNT)
          {
            for (SparseSet::const_iterator it = next->begin();
                  it != next->end(); it++)
              result.set_bit(*it);
            delete next;
          }
          else
          {
            result.set_count(SPARSE_CNT);
            result.set_sparse(next);
          }
        }
        else
          delete next;
      }
      else if (rhs_count == SPARSE_CNT)
      {
        SparseSet *next = new SparseSet(*rhs.get_sparse());
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx);
          SparseSet::iterator finder = next->find(bit);
          if (finder != next->end())
            next->erase(finder);
          else
            next->insert(bit);
        }
        if (!next->empty())
        {
          if (next->size() <= MAX_CNT)
          {
            for (SparseSet::const_iterator it = next->begin();
                  it != next->end(); it++)
              result.set_bit(*it);
            delete next;
          }
          else
          {
            result.set_count(SPARSE_CNT);
            result.set_sparse(next);
          }
        }
        else
          delete next;
      }
      else
      {
        if (count < rhs_count)
        {
          for (int idx = 0; idx < WORDS; idx++)
            result.bits[idx] = rhs.bits[idx];
          for (int idx = 0; idx < count; idx++)
          {
            int bit = get_value<OVERLAP>(idx); 
            if (!rhs.is_set(bit))
              result.set_bit(bit);
            else
              result.unset_bit(bit);
          }  
        }
        else
        {
          for (int idx = 0; idx < WORDS; idx++)
            result.bits[idx] = bits[idx];
          for (int idx = 0; idx < rhs_count; idx++)
          {
            int bit = rhs.get_value<OVERLAP>(idx);
            if (!is_set(bit))
              result.set_bit(bit);
            else
              result.unset_bit(bit);
          }
        }
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS>& 
     CompoundBitMask<BITMASK,MAX,WORDS>::operator|=(const CompoundBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      int rhs_count = rhs.get_count();
      if (rhs_count == DENSE_CNT)
      {
        int count = get_count();
        if (count == DENSE_CNT)
        {
          BITMASK *mask = get_dense();
          (*mask) |= (*rhs.get_dense());
        }
        else
        {
          BITMASK *mask = new BITMASK(*rhs.get_dense());
          if (count == SPARSE_CNT)
          {
            SparseSet *sparse = get_sparse();
            for (SparseSet::const_iterator it = sparse->begin();
                  it != sparse->end(); it++)
              mask->set_bit(*it);
            delete sparse;
          }
          else
          {
            for (unsigned idx = 0; idx < count; idx++)
              mask->set_bit(get_value<OVERLAP>(idx));
          }
          set_count(DENSE_CNT);
          set_dense(mask);
        }
      }
      else if (rhs_count == SPARSE_CNT)
      {
        SparseSet *other = rhs.get_sparse();
        for (SparseSet::const_iterator it = other->begin();
              it != other->end(); it++)
          set_bit(*it);
      }
      else
      {
        for (int idx = 0; idx < rhs_count; idx++)
          set_bit(rhs.get_value<OVERLAP>(idx));
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS>& 
     CompoundBitMask<BITMASK,MAX,WORDS>::operator&=(const CompoundBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == DENSE_CNT)
      {
        int rhs_count = rhs.get_count();
        BITMASK *dense = get_dense();
        if (rhs_count == DENSE_CNT)
        {
          (*dense) &= (*rhs.get_dense());  
          if (!(*dense))
          {
            delete dense;
            set_count(0);
          }
        }
        else if (rhs_count == SPARSE_CNT)
        {
          SparseSet *other = rhs.get_sparse();
          set_count(0);
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
          {
            if (dense->is_set(*it))
              set_bit(*it);
          }
          delete dense;
        }
        else
        {
          set_count(0);
          for (unsigned idx = 0; idx < rhs_count; idx++)
          {
            int bit = rhs.get_value<OVERLAP>(idx);
            if (dense->is_set(bit))
              set_bit(bit);
          }
          delete dense;
        }
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        std::vector<int> to_delete;
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          if (!rhs.is_set(*it))
            to_delete.push_back(*it);
        }
        if (!to_delete.empty())
        {
          for (std::vector<int>::const_iterator it = to_delete.begin();
                it != to_delete.end(); it++)
            unset_bit(*it);
        }
      }
      else
      {
        int next_idx = 0;
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx);
          if (rhs.is_set(bit))
          {
            if (next_idx != idx)
              set_value<OVERLAP>(next_idx++, bit);
            else
              next_idx++;
          }
        }
        if (next_idx != count)
          set_count(next_idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS>& 
     CompoundBitMask<BITMASK,MAX,WORDS>::operator^=(const CompoundBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      int rhs_count = rhs.get_count();
      if (count == DENSE_CNT)
      {
        BITMASK *dense = get_dense();
        if (rhs_count == DENSE_CNT)
          (*dense) ^= (*rhs.get_dense());
        else if (rhs_count == SPARSE_CNT)
        {
          SparseSet *other = rhs.get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
          {
            if (dense->is_set(*it))
              dense->unset_bit(*it);
            else
              dense->set_bit(*it);
          }
        }
        else
        {
          for (unsigned idx = 0; idx < rhs_count; idx++)
          {
            int bit = rhs.get_value<OVERLAP>(idx);
            if (dense->is_set(bit))
              dense->unset_bit(bit);
            else
              dense->set_bit(bit);
          }
        }
        if (!(*dense))
        {
          delete dense;
          set_count(0);
        }
      }
      else if (rhs_count == DENSE_CNT)
      {
        BITMASK next = *rhs.get_dense();
        if (count == SPARSE_CNT)
        {
          SparseSet *sparse = get_sparse();
          for (SparseSet::const_iterator it = sparse->begin();
                it != sparse->end(); it++)
          {
            if (next.is_set(*it))
              next.unset_bit(*it);
            else
              next.set_bit(*it);
          }
          delete sparse;
        }
        else
        {
          for (unsigned idx = 0; idx < count; idx++)
          {
            int bit = get_value<OVERLAP>(idx);
            if (next.is_set(bit))
              next.unset_bit(bit);
            else
              next.set_bit(bit);
          }
        }
        if (!!next)
        {
          set_count(DENSE_CNT);
          set_dense(new BITMASK(next));
        }
        else
          set_count(0);
      }
      else if (rhs_count == SPARSE_CNT)
      {
        SparseSet *other = rhs.get_sparse();
        for (SparseSet::const_iterator it = other->begin();
              it != other->end(); it++)
        {
          if (is_set(*it))
            unset_bit(*it);
          else
            set_bit(*it);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < rhs_count; idx++)
        {
          int bit = rhs.get_value<OVERLAP>(idx);
          if (is_set(bit))
            unset_bit(bit);
          else
            set_bit(bit);
        }
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline bool CompoundBitMask<BITMASK,MAX,WORDS>::operator*(
                                              const CompoundBitMask &rhs) const 
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == 0)
        return true;
      int rhs_count = rhs.get_count();
      if (rhs_count == 0)
        return true;
      if (count == DENSE_CNT)
      {
        if (rhs_count == DENSE_CNT)
          return ((*get_dense()) * (*rhs.get_dense()));
        else if (rhs_count == SPARSE_CNT)
        {
          SparseSet *other = rhs.get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
            if (is_set(*it))
              return false;
        }
        else
          for (int idx = 0; idx < rhs_count; idx++)
            if (is_set(rhs.get_value<OVERLAP>(idx)))
              return false;
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
          if (rhs.is_set(*it))
            return false;
      }
      else
      {
        for (int idx = 0; idx < count; idx++)
          if (rhs.is_set(get_value<OVERLAP>(idx)))
            return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline bool CompoundBitMask<BITMASK,MAX,WORDS>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      return (get_count() == 0); 
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS> 
      CompoundBitMask<BITMASK,MAX,WORDS>::operator-(
                                              const CompoundBitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<BITMASK,MAX,WORDS> result;
      int count = get_count();
      if (count == DENSE_CNT)
      {
        BITMASK next = *(get_dense());
        int rhs_count = rhs.get_count();
        if (rhs_count == DENSE_CNT)
        {
          next -= (*rhs.get_dense());
        }
        else if (rhs_count == SPARSE_CNT)
        {
          SparseSet *other = rhs.get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
            next.unset_bit(*it);
        }
        else
        {
          for (int idx = 0; idx < rhs_count; idx++)
            next.unset_bit(rhs.get_value<OVERLAP>(idx));
        }
        if (!!next)
        {
          result.set_count(DENSE_CNT);
          result.set_dense(new BITMASK(next));
        }
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          if (!rhs.is_set(*it))
            result.set_bit(*it);
        }
      }
      else
      {
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx);
          if (!rhs.is_set(bit))
            result.set_bit(bit);
        }
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS>&
     CompoundBitMask<BITMASK,MAX,WORDS>::operator-=(const CompoundBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == DENSE_CNT)
      {
        int rhs_count = rhs.get_count();
        BITMASK *dense = get_dense();
        if (rhs_count == DENSE_CNT)
        {
          (*dense) -= (*rhs.get_dense());
        }
        else if (rhs_count == SPARSE_CNT)
        {
          SparseSet *other = rhs.get_sparse();
          for (SparseSet::const_iterator it = other->begin();
                it != other->end(); it++)
            dense->unset_bit(*it);
        }
        else
        {
          for (int idx = 0; idx < rhs_count; idx++)
            dense->unset_bit(rhs.get_value<OVERLAP>(idx));
        }
        if (!(*dense))
        {
          delete dense;
          set_count(0);
        }
      }
      else if (count == SPARSE_CNT)
      {
        std::vector<int> to_delete;
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          if (rhs.is_set(*it))
            to_delete.push_back(*it);
        }
        if (!to_delete.empty())
        {
          for (std::vector<int>::const_iterator it = to_delete.begin();
                it != to_delete.end(); it++)
            unset_bit(*it);
        }
      }
      else
      {
        int next_idx = 0;
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx);
          if (!rhs.is_set(bit))
          {
            if (next_idx != idx)
              set_value<OVERLAP>(next_idx++, bit);
            else
              next_idx++;
          }
        }
        if (next_idx != count)
          set_count(next_idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS> 
           CompoundBitMask<BITMASK,MAX,WORDS>::operator<<(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<BITMASK,MAX,WORDS> result;
      int count = get_count();
      if (count == DENSE_CNT)
      {
        BITMASK dense = (*get_dense()) << shift;
        if (!!dense)
        {
          result.set_count(DENSE_CNT);
          result.set_dense(new BITMASK(dense));
        }
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          int bit = (*it) + shift;
          if (bit < MAX)
            result.set_bit(bit);
        }
      }
      else
      {
        int next_idx = 0;
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx) + shift;
          if (bit < MAX)
            result.set_value<OVERLAP>(next_idx++, bit);
        }
        if (next_idx > 0)
          result.set_count(next_idx);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS> 
           CompoundBitMask<BITMASK,MAX,WORDS>::operator>>(unsigned shift) const
    //-------------------------------------------------------------------------
    {
      CompoundBitMask<BITMASK,MAX,WORDS> result;
      int count = get_count();
      if (count == DENSE_CNT)
      {
        BITMASK dense = (*get_dense()) >> shift;
        if (!!dense)
        {
          result.set_count(DENSE_CNT);
          result.set_dense(new BITMASK(dense));
        }
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          int bit = (*it) - shift;
          if (bit >= 0)
            result.set_bit(bit);
        }
      }
      else
      {
        int next_idx = 0;
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx) - shift;
          if (bit >= 0)
            result.set_value<OVERLAP>(next_idx++, bit);
        }
        if (next_idx > 0)
          result.set_count(next_idx);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS>& 
                CompoundBitMask<BITMASK,MAX,WORDS>::operator<<=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      int count = get_count(); 
      if (count == DENSE_CNT)
      {
        BITMASK *dense = get_dense();
        (*dense) <<= shift;
        if (!(*dense))
        {
          delete dense;
          set_count(0);
        }
      }
      else if (count == SPARSE_CNT)
      {
        std::vector<int> to_delete;
        SparseSet *sparse = get_sparse();
        set_count(0);
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          int bit = (*it) + shift;
          if (bit < MAX)
            set_bit(bit);
        }
        delete sparse;
      }
      else
      {
        int next_idx = 0; 
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx) + shift;
          if (bit < MAX)
            set_value<OVERLAP>(next_idx++, bit);
        }
        if (next_idx != count)
          set_count(next_idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline CompoundBitMask<BITMASK,MAX,WORDS>& 
                CompoundBitMask<BITMASK,MAX,WORDS>::operator>>=(unsigned shift)
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == DENSE_CNT)
      {
        BITMASK *dense = get_dense();
        (*dense) >>= shift;
        if (!(*dense))
        {
          delete dense;
          set_count(0);
        }
      }
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        set_count(0);
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          int bit = (*it) - shift;
          if (bit >= 0)
            set_bit(bit);
        }
        delete sparse;
      }
      else
      {
        int next_idx = 0;
        for (int idx = 0; idx < count; idx++)
        {
          int bit = get_value<OVERLAP>(idx) - shift;
          if (bit >= 0)
            set_value<OVERLAP>(next_idx++, bit);
        }
        if (next_idx != count)
          set_count(next_idx);
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline uint64_t CompoundBitMask<BITMASK,MAX,WORDS>::get_hash_key(void) 
                                                                          const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == DENSE_CNT)
        return get_dense()->get_hash_key();
      uint64_t result = 0;
      if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
          result |= (1UL << ((*it) % (8*sizeof(result))));
      }
      else
      {
        for (int idx = 0; idx < count; idx++)
          result |= (1UL << (get_value<OVERLAP>(idx) % (8*sizeof(result))));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
      template<typename ST>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      rez.serialize(count);
      if (count == DENSE_CNT)
        get_dense()->serialize(rez); 
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        rez.template serialize<size_t>(sparse->size());
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
          rez.serialize(*it);
      }
      else
      {
        // Just serialize all the words, won't be that hard
        for (unsigned idx = 0; idx < WORDS; idx++)
          rez.serialize(bits[idx]);
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
      template<typename DT>
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::deserialize(DT &derez)
    //-------------------------------------------------------------------------
    {
      int current_count = get_count();
      int next_count;
      derez.deserialize(next_count);
      if (next_count == DENSE_CNT)
      {
        BITMASK *dense;
        if (current_count != DENSE_CNT)
        {
          if (current_count == SPARSE_CNT)
            delete get_sparse();
          set_count(DENSE_CNT);
          dense = new BITMASK();
          set_dense(dense);
        }
        else
          dense = get_dense();
        dense->deserialize(derez);
      }
      else if (next_count == SPARSE_CNT)
      {
        SparseSet *sparse = new SparseSet();
        if (current_count == DENSE_CNT)
          delete get_dense();
        size_t num_elements;
        derez.deserialize(num_elements);
        for (unsigned idx = 0; idx < num_elements; idx++)
        {
          int bit;
          derez.deserialize(bit);
          sparse->insert(bit);
        }
        set_count(SPARSE_CNT);
        set_sparse(sparse);
      }
      else
      {
        if (current_count == DENSE_CNT)
          delete get_dense();
        else if (current_count == SPARSE_CNT)
          delete get_sparse();
        for (unsigned idx = 0; idx < WORDS; idx++)
          derez.deserialize(bits[idx]);
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline char* CompoundBitMask<BITMASK,MAX,WORDS>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == DENSE_CNT)
      {
        return get_dense()->to_string(); 
      }
      else if (count == SPARSE_CNT)
      {
        char *result = (char*)malloc(1024*sizeof(char));
        sprintf(result,"Compound Sparse %d:", count);
        SparseSet *sparse = get_sparse();
        for (SparseSet::const_iterator it = sparse->begin();
              it != sparse->end(); it++)
        {
          char temp[64];
          sprintf(temp, " %d", (*it));
          strcat(result,temp);
        }
        return result;
      }
      else
      {
        char *result = (char*)malloc(1024*sizeof(char));
        sprintf(result,"Compound Base %d:", count);
        for (int idx = 0; idx < count; idx++)
        {
          char temp[64];
          sprintf(temp, " %d", get_value<OVERLAP>(idx));
          strcat(result,temp);
        }
        return result;
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    inline int CompoundBitMask<BITMASK,MAX,WORDS>::pop_count(void) const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      if (count == DENSE_CNT)
        return get_dense()->pop_count();
      else if (count == SPARSE_CNT)
        return get_sparse()->size();
      return count;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned int MAX, unsigned int WORDS>
    /*static*/ inline int CompoundBitMask<BITMASK,MAX,WORDS>::pop_count(
                                                   const CompoundBitMask &mask)
    //-------------------------------------------------------------------------
    {
      int count = mask.get_count();
      if (count == DENSE_CNT)
        count = BITMASK::pop_count(*mask.get_dense());
      else if (count == SPARSE_CNT)
        count = mask.get_sparse()->size();
      return count;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    IntegerSet<IT,DT,BIDIR>::IntegerSet(void)
      : sparse(true)
    //-------------------------------------------------------------------------
    {
      set_ptr.sparse = 0;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    IntegerSet<IT,DT,BIDIR>::IntegerSet(const IntegerSet &rhs)
      : sparse(rhs.sparse)
    //-------------------------------------------------------------------------
    {
      if (rhs.sparse)
      {
	if (rhs.set_ptr.sparse && !rhs.set_ptr.sparse->empty())
	  set_ptr.sparse = new typename std::set<IT>(*rhs.set_ptr.sparse);
	else
	  set_ptr.sparse = 0;
      }
      else
      {
        set_ptr.dense = new DenseSet();
        set_ptr.dense->set = rhs.set_ptr.dense->set;
      }
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    IntegerSet<IT,DT,BIDIR>::~IntegerSet(void)
    //-------------------------------------------------------------------------
    {
      if (sparse)
      {
	// this may be a null pointer, but delete does the right thing
        delete set_ptr.sparse;
      }
      else
      {
#ifdef DEBUG_LEGION
	assert(set_ptr.dense != NULL);
#endif
        delete set_ptr.dense;
      }
    }
    
    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    IntegerSet<IT,DT,BIDIR>& 
                      IntegerSet<IT,DT,BIDIR>::operator=(const IntegerSet &rhs)
    //-------------------------------------------------------------------------
    {
      if (rhs.sparse)
      {
        if (!sparse)
        {
          delete set_ptr.dense;
          set_ptr.sparse = 0;
	  sparse = true;
        }
        else if (set_ptr.sparse)
	  set_ptr.sparse->clear();
	// if rhs has any contents, copy them over, creating set if needed
	if (rhs.set_ptr.sparse && !rhs.set_ptr.sparse->empty())
	{
	  if (!set_ptr.sparse)
	    set_ptr.sparse = new typename std::set<IT>(*rhs.set_ptr.sparse);
	  else
	    *(set_ptr.sparse) = *(rhs.set_ptr.sparse);
	}
      }
      else
      {
        if (sparse)
        {
          delete set_ptr.sparse;
          set_ptr.dense = new DenseSet();
	  sparse = false;
        }
        else
          set_ptr.dense->set.clear();
        set_ptr.dense->set = rhs.set_ptr.dense->set;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline bool IntegerSet<IT,DT,BIDIR>::contains(IT index) const
    //-------------------------------------------------------------------------
    {
      if (sparse)
        return (set_ptr.sparse &&
		(set_ptr.sparse->find(index) != set_ptr.sparse->end()));
      else
        return set_ptr.dense->set.is_set(index);
    }
    
    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline void IntegerSet<IT,DT,BIDIR>::add(IT index)
    //-------------------------------------------------------------------------
    {
      if (sparse)
      {
	// Create set if this is the first addition
	if (!set_ptr.sparse)
	  set_ptr.sparse = new typename std::set<IT>;

        // Add it and see if it is too big
        set_ptr.sparse->insert(index);
        if (sizeof(DT) < (set_ptr.sparse->size() * 
                          (sizeof(IT) + STL_SET_NODE_SIZE)))
        {
          DenseSet *dense_set = new DenseSet();
          for (typename std::set<IT>::const_iterator it = 
                set_ptr.sparse->begin(); it != set_ptr.sparse->end(); it++)
          {
            dense_set->set.set_bit(*it);
          }
          // Delete the sparse set
          delete set_ptr.sparse;
          set_ptr.dense = dense_set;
          sparse = false;
        }
      }
      else
        set_ptr.dense->set.set_bit(index);
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline void IntegerSet<IT,DT,BIDIR>::remove(IT index)
    //-------------------------------------------------------------------------
    {
      if (!sparse)
      {
        set_ptr.dense->set.unset_bit(index); 
        // Only check for flip back if we are bi-directional
        if (BIDIR)
        {
          IT count = DT::pop_count(set_ptr.dense->set);
          if ((count * (sizeof(IT) + STL_SET_NODE_SIZE)) < sizeof(DT))
          {
            typename std::set<IT> *sparse_set = new typename std::set<IT>();
            for (IT idx = 0; idx < DT::ELEMENTS; idx++)
            {
              if (set_ptr.dense->set[idx])
              {
                for (IT i = 0; i < DT::ELEMENT_SIZE; i++)
                {
                  IT value = idx * DT::ELEMENT_SIZE + i;
                  if (set_ptr.dense->set.is_set(value))
                    sparse_set->insert(value);
                }
              }
            }
            // Delete the dense set
            delete set_ptr.dense;
            set_ptr.sparse = sparse_set;
            sparse = true;
          }
        }
      }
      else
	if (set_ptr.sparse)
	  set_ptr.sparse->erase(index);
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IT IntegerSet<IT,DT,BIDIR>::find_first_set(void) const
    //-------------------------------------------------------------------------
    {
      if (sparse)
      {
#ifdef DEBUG_LEGION
        assert(set_ptr.sparse && !set_ptr.sparse->empty());
#endif
        return *(set_ptr.sparse->begin());
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!!(set_ptr.dense->set));
#endif
        return set_ptr.dense->set.find_first_set();
      }
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IT IntegerSet<IT,DT,BIDIR>::find_index_set(int index) const
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index >= 0);
      assert(index < int(size()));
#endif
      if (index == 0)
        return find_first_set();
      if (sparse)
      {
#ifdef DEBUG_LEGION
	assert(set_ptr.sparse);
#endif
        typename std::set<IT>::const_iterator it = set_ptr.sparse->begin();
        while (index > 0)
        {
          it++;
          index--;
        }
        return *it;
      }
      else
        return set_ptr.dense->set.find_index_set(index);
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR> template<typename FUNCTOR>
    inline void IntegerSet<IT,DT,BIDIR>::map(FUNCTOR &functor) const
    //-------------------------------------------------------------------------
    {
      if (sparse)
      {
	if (set_ptr.sparse)
	{
	  for (typename std::set<IT>::const_iterator it = 
                set_ptr.sparse->begin(); it != set_ptr.sparse->end(); it++)
          {
	    functor.apply(*it);
          }
        }
      }
      else
      {
        for (IT idx = 0; idx < DT::ELEMENTS; idx++)
        {
          if (set_ptr.dense->set[idx])
          {
            IT value = idx * DT::ELEMENT_SIZE;
            for (IT i = 0; i < DT::ELEMENT_SIZE; i++, value++)
            {
              if (set_ptr.dense->set.is_set(value))
                functor.apply(value);
            }
          }
        }
      }
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR> template<typename ST>
    inline void IntegerSet<IT,DT,BIDIR>::serialize(ST &rez) const
    //-------------------------------------------------------------------------
    {
      rez.template serialize<bool>(sparse);
      if (sparse)
      {
	if (set_ptr.sparse)
	{
          rez.template serialize<size_t>(set_ptr.sparse->size());
          for (typename std::set<IT>::const_iterator it = 
                set_ptr.sparse->begin(); it != set_ptr.sparse->end(); it++)
          {
            rez.serialize(*it);
          }
        }
	else
          rez.template serialize<size_t>(0);
      }
      else
        rez.serialize(set_ptr.dense->set);
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR> template<typename ZT>
    inline void IntegerSet<IT,DT,BIDIR>::deserialize(ZT &derez)
    //-------------------------------------------------------------------------
    {
      bool is_sparse;
      derez.template deserialize<bool>(is_sparse);
      if (is_sparse)
      {
        // If it doesn't match then replace the old one
        if (!sparse)
        {
          delete set_ptr.dense;
          set_ptr.sparse = 0;
	  sparse = true;
        }
        else if (set_ptr.sparse)
	  set_ptr.sparse->clear();
        size_t num_elements;
        derez.template deserialize<size_t>(num_elements);
	if (num_elements > 0) {
	  if (!set_ptr.sparse)
	    set_ptr.sparse = new typename std::set<IT>;
          for (unsigned idx = 0; idx < num_elements; idx++)
          {
	    IT element;
            derez.deserialize(element);
            set_ptr.sparse->insert(element);
          }
        }
      }
      else
      {
        // If it doesn't match then replace the old one
        if (sparse)
        {
          delete set_ptr.sparse;
          set_ptr.dense = new DenseSet();
	  sparse = false;
        }
        else
          set_ptr.dense->set.clear();
        derez.deserialize(set_ptr.dense->set);
      }
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IntegerSet<IT,DT,BIDIR>
                IntegerSet<IT,DT,BIDIR>::operator|(const IntegerSet &rhs) const
    //-------------------------------------------------------------------------
    {
      // Do the fast case here
      if (!sparse)
      {
        IntegerSet<IT,DT,BIDIR> result(*this);
        if (rhs.sparse)
        {
          UnionFunctor functor(result);
          rhs.map(functor);
        }
        else
          result.set_ptr.dense->set |= rhs.set_ptr.dense->set;
        return result;
      }
      IntegerSet<IT,DT,BIDIR> result(rhs); 
      UnionFunctor functor(result);
      this->map(functor);
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IntegerSet<IT,DT,BIDIR>
                IntegerSet<IT,DT,BIDIR>::operator&(const IntegerSet &rhs) const
    //-------------------------------------------------------------------------
    {
      // Do the fast case here
      if (!sparse && !rhs.sparse)
      {
        IntegerSet<IT,DT,BIDIR> result(*this);
        result.set_ptr.dense->set &= rhs.set_ptr.dense->set;
        return result;
      }
      IntegerSet<IT,DT,BIDIR> result;
      IntersectFunctor functor(result, *this);
      rhs.map(functor);
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IntegerSet<IT,DT,BIDIR>
                IntegerSet<IT,DT,BIDIR>::operator-(const IntegerSet &rhs) const
    //-------------------------------------------------------------------------
    {
      // Do the fast case here
      if (!sparse && !rhs.sparse)
      {
        IntegerSet<IT,DT,BIDIR> result(*this); 
        result.set_ptr.dense->set -= rhs.set_ptr.dense->set;
        return result;
      }
      IntegerSet<IT,DT,BIDIR> result(*this);
      DifferenceFunctor functor(result);
      rhs.map(functor);
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IntegerSet<IT,DT,BIDIR>&
                     IntegerSet<IT,DT,BIDIR>::operator|=(const IntegerSet &rhs)
    //-------------------------------------------------------------------------
    {
      // Do the fast case here
      if (!sparse && !rhs.sparse)
      {
        set_ptr.dense->set |= rhs.set_ptr.dense->set;
        return *this;
      }
      UnionFunctor functor(*this);
      rhs.map(functor);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IntegerSet<IT,DT,BIDIR>&
                     IntegerSet<IT,DT,BIDIR>::operator&=(const IntegerSet &rhs)
    //-------------------------------------------------------------------------
    {
      // Do the fast case
      if (!sparse && !rhs.sparse)
      {
        set_ptr.dense->set &= rhs.set_ptr.dense->set;
        return *this;
      }
      // Can't overwrite ourselves
      IntegerSet<IT,DT,BIDIR> temp;
      IntersectFunctor functor(temp, *this);
      rhs.map(functor);
      (*this) = temp;
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IntegerSet<IT,DT,BIDIR>&
                     IntegerSet<IT,DT,BIDIR>::operator-=(const IntegerSet &rhs)
    //-------------------------------------------------------------------------
    {
      // Do the fast case
      if (!sparse && !rhs.sparse)
      {
        set_ptr.dense->set -= rhs.set_ptr.dense->set;
        return *this;
      }
      DifferenceFunctor functor(*this);
      rhs.map(functor);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline bool IntegerSet<IT,DT,BIDIR>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      if (sparse)
        return (!set_ptr.sparse || set_ptr.sparse->empty());
      else
        return !(set_ptr.dense->set);
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline size_t IntegerSet<IT,DT,BIDIR>::size(void) const
    //-------------------------------------------------------------------------
    {
      if (sparse)
        return (set_ptr.sparse ? set_ptr.sparse->size() : 0);
      else
        return set_ptr.dense->set.pop_count(set_ptr.dense->set);
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline void IntegerSet<IT,DT,BIDIR>::clear(void)
    //-------------------------------------------------------------------------
    {
      // always switch back to set on a clear
      if (!sparse)
      {
	delete set_ptr.dense;
	set_ptr.sparse = 0;
	sparse = true;
      } 
      else if (set_ptr.sparse)
	set_ptr.sparse->clear();
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline IntegerSet<IT,DT,BIDIR>& 
                                 IntegerSet<IT,DT,BIDIR>::swap(IntegerSet &rhs)
    //-------------------------------------------------------------------------
    {
      std::swap(sparse, rhs.sparse);
      std::swap(set_ptr.sparse, rhs.set_ptr.sparse);
      // don't do dense because it's a union and that'd just swap things back
      return *this;
    }

#endif // __BITMASK_H__
