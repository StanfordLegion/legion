/* Copyright 2018 Stanford University, NVIDIA Corporation
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


#ifndef __LEGION_UTILITIES_H__
#define __LEGION_UTILITIES_H__

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "legion/legion_types.h"
#include "legion.h"
#include "legion/legion_allocation.h"

// Apple can go screw itself
#ifndef __MACH__
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#endif
#ifdef __ALTIVEC__
#include <altivec.h>
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

// Useful macros
#define IS_NO_ACCESS(req) (((req).privilege & READ_WRITE) == NO_ACCESS)
#define IS_READ_ONLY(req) (((req).privilege & READ_WRITE) <= READ_ONLY)
#define HAS_READ(req) ((req).privilege & READ_ONLY)
#define HAS_WRITE(req) ((req).privilege & (WRITE_DISCARD | REDUCE))
#define IS_WRITE(req) ((req).privilege & WRITE_DISCARD)
#define IS_WRITE_ONLY(req) (((req).privilege & READ_WRITE) == WRITE_DISCARD)
#define IS_REDUCE(req) (((req).privilege & READ_WRITE) == REDUCE)
#define IS_EXCLUSIVE(req) ((req).prop == EXCLUSIVE)
#define IS_ATOMIC(req) ((req).prop == ATOMIC)
#define IS_SIMULT(req) ((req).prop == SIMULTANEOUS)
#define IS_RELAXED(req) ((req).prop == RELAXED)

namespace Legion {

    /////////////////////////////////////////////////////////////
    // Serializer 
    /////////////////////////////////////////////////////////////
    class Serializer {
    public:
      Serializer(size_t base_bytes = 4096)
        : total_bytes(base_bytes), buffer((char*)malloc(base_bytes)), 
          index(0) 
#ifdef DEBUG_LEGION
          , context_bytes(0)
#endif
      { }
      Serializer(const Serializer &rhs)
      {
        // should never be called
        assert(false);
      }
    public:
      ~Serializer(void)
      {
        free(buffer);
      }
    public:
      inline Serializer& operator=(const Serializer &rhs);
    public:
      template<typename T>
      inline void serialize(const T &element);
      // we need special serializers for bit masks
      template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      inline void serialize(const Internal::BitMask<T,MAX,SHIFT,MASK> &mask);
      template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      inline void serialize(const Internal::TLBitMask<T,MAX,SHIFT,MASK> &mask);
#ifdef __SSE2__
      template<unsigned int MAX>
      inline void serialize(const Internal::SSEBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void serialize(const Internal::SSETLBitMask<MAX> &mask);
#endif
#ifdef __AVX__
      template<unsigned int MAX>
      inline void serialize(const Internal::AVXBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void serialize(const Internal::AVXTLBitMask<MAX> &mask);
#endif
#ifdef __ALTIVEC__
      template<unsigned int MAX>
      inline void serialize(const PPCBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void serialize(const PPCTLBitMask<MAX> &mask);
#endif
      template<typename IT, typename DT, bool BIDIR>
      inline void serialize(
          const Internal::IntegerSet<IT,DT,BIDIR> &index_set);
      inline void serialize(const Domain &domain);
      inline void serialize(const DomainPoint &dp);
      inline void serialize(const void *src, size_t bytes);
    public:
      inline void begin_context(void);
      inline void end_context(void);
    public:
      inline size_t get_index(void) const { return index; }
      inline const void* get_buffer(void) const { return buffer; }
      inline size_t get_buffer_size(void) const { return total_bytes; }
      inline size_t get_used_bytes(void) const { return index; }
      inline void* reserve_bytes(size_t size);
    private:
      inline void resize(void);
    private:
      size_t total_bytes;
      char *buffer;
      size_t index;
#ifdef DEBUG_LEGION
      size_t context_bytes;
#endif
    };

    /////////////////////////////////////////////////////////////
    // Deserializer 
    /////////////////////////////////////////////////////////////
    class Deserializer {
    public:
      Deserializer(const void *buf, size_t buffer_size)
        : total_bytes(buffer_size), buffer((const char*)buf), index(0)
#ifdef DEBUG_LEGION
          , context_bytes(0)
#endif
      { }
      Deserializer(const Deserializer &rhs)
        : total_bytes(0)
      {
        // should never be called
        assert(false);
      }
    public:
      ~Deserializer(void)
      {
#ifdef DEBUG_LEGION
        // should have used the whole buffer
        assert(index == total_bytes); 
#endif
      }
    public:
      inline Deserializer& operator=(const Deserializer &rhs);
    public:
      template<typename T>
      inline void deserialize(T &element);
      // We need specialized deserializers for bit masks
      template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      inline void deserialize(Internal::BitMask<T,MAX,SHIFT,MASK> &mask);
      template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      inline void deserialize(Internal::TLBitMask<T,MAX,SHIFT,MASK> &mask);
#ifdef __SSE2__
      template<unsigned int MAX>
      inline void deserialize(Internal::SSEBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void deserialize(Internal::SSETLBitMask<MAX> &mask);
#endif
#ifdef __AVX__
      template<unsigned int MAX>
      inline void deserialize(Internal::AVXBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void deserialize(Internal::AVXTLBitMask<MAX> &mask);
#endif
#ifdef __ALTIVEC__
      template<unsigned int MAX>
      inline void deserialize(PPCBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void deserialize(PPCTLBitMask<MAX> &mask);
#endif
      template<typename IT, typename DT, bool BIDIR>
      inline void deserialize(Internal::IntegerSet<IT,DT,BIDIR> &index_set);
      inline void deserialize(Domain &domain);
      inline void deserialize(DomainPoint &dp);
      inline void deserialize(void *dst, size_t bytes);
    public:
      inline void begin_context(void);
      inline void end_context(void);
    public:
      inline size_t get_remaining_bytes(void) const;
      inline const void* get_current_pointer(void) const;
      inline void advance_pointer(size_t bytes);
    private:
      const size_t total_bytes;
      const char *buffer;
      size_t index;
#ifdef DEBUG_LEGION
      size_t context_bytes;
#endif
    };

  namespace Internal {
    /**
     * \struct RegionUsage
     * A minimal structure for performing dependence analysis.
     */
    struct RegionUsage {
    public:
      RegionUsage(void)
        : privilege(NO_ACCESS), prop(EXCLUSIVE), redop(0) { }
      RegionUsage(PrivilegeMode p, CoherenceProperty c, ReductionOpID r)
        : privilege(p), prop(c), redop(r) { }
      RegionUsage(const RegionRequirement &req)
        : privilege(req.privilege), prop(req.prop), redop(req.redop) { }
    public:
      inline bool operator==(const RegionUsage &rhs) const
      { return ((privilege == rhs.privilege) && (prop == rhs.prop) 
                && (redop == rhs.redop)); }
      inline bool operator!=(const RegionUsage &rhs) const
      { return !((*this) == rhs); }
    public:
      PrivilegeMode     privilege;
      CoherenceProperty prop;
      ReductionOpID     redop;
    };

    // The following two methods define the dependence analysis
    // for all of Legion.  Modifying them can have enormous
    // consequences on how programs execute.

    //--------------------------------------------------------------------------
    static inline DependenceType check_for_anti_dependence(
            const RegionUsage &u1, const RegionUsage &u2, DependenceType actual)
    //--------------------------------------------------------------------------
    {
      // Check for WAR or WAW with write-only
      if (IS_READ_ONLY(u1))
      {
#ifdef DEBUG_LEGION
        // We know at least req1 or req2 is a writers, so if req1 is not...
        assert(HAS_WRITE(u2)); 
#endif
        return ANTI_DEPENDENCE;
      }
      else
      {
        if (IS_WRITE_ONLY(u2))
        {
          // WAW with a write-only
          return ANTI_DEPENDENCE;
        }
        else
        {
          // This defaults to whatever the actual dependence is
          return actual;
        }
      }
    }

    //--------------------------------------------------------------------------
    static inline DependenceType check_dependence_type(const RegionUsage &u1,
                                                       const RegionUsage &u2)
    //--------------------------------------------------------------------------
    {
      // Two readers are never a dependence
      if (IS_READ_ONLY(u1) && IS_READ_ONLY(u2))
      {
        return NO_DEPENDENCE;
      }
      else if (IS_REDUCE(u1) && IS_REDUCE(u2))
      {
        // If they are the same kind of reduction, no dependence, 
        // otherwise true dependence
        if (u1.redop == u2.redop)
          return NO_DEPENDENCE;
        else
          return TRUE_DEPENDENCE;
      }
      else
      {
        // Everything in here has at least one right
#ifdef DEBUG_LEGION
        assert(HAS_WRITE(u1) || HAS_WRITE(u2));
#endif
        // If anything exclusive 
        if (IS_EXCLUSIVE(u1) || IS_EXCLUSIVE(u2))
        {
          return check_for_anti_dependence(u1,u2,TRUE_DEPENDENCE/*default*/);
        }
        // Anything atomic (at least one is a write)
        else if (IS_ATOMIC(u1) || IS_ATOMIC(u2))
        {
          // If they're both atomics, return an atomic dependence
          if (IS_ATOMIC(u1) && IS_ATOMIC(u2))
          {
            return check_for_anti_dependence(u1,u2,
                                             ATOMIC_DEPENDENCE/*default*/); 
          }
          // If the one that is not an atomic is a read, we're also ok
          // We still need a simultaneous dependence if we don't have an
          // actual dependence
          else if ((!IS_ATOMIC(u1) && IS_READ_ONLY(u1)) ||
                   (!IS_ATOMIC(u2) && IS_READ_ONLY(u2)))
          {
            return SIMULTANEOUS_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(u1,u2,TRUE_DEPENDENCE/*default*/);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(u1) || IS_SIMULT(u2))
        {
          return SIMULTANEOUS_DEPENDENCE;
        }
        else if (IS_RELAXED(u1) && IS_RELAXED(u2))
        {
          // TODO: Make this truly relaxed, right now it is the 
          // same as simultaneous
          return SIMULTANEOUS_DEPENDENCE;
          // This is what it should be: return NO_DEPENDENCE;
          // What needs to be done:
          // - RegionNode::update_valid_instances needs to allow multiple 
          //               outstanding writers
          // - RegionNode needs to detect relaxed case and make copies from all 
          //              relaxed instances to non-relaxed instance
        }
        // We should never make it here
        assert(false);
        return NO_DEPENDENCE;
      }
    } 

    /////////////////////////////////////////////////////////////
    // Semantic Info 
    /////////////////////////////////////////////////////////////

    /**
     * \struct SemanticInfo
     * A struct for storing semantic information for various things
     */
    struct SemanticInfo {
    public:
      SemanticInfo(void)
        : buffer(NULL), size(0) { }  
      SemanticInfo(void *buf, size_t s, bool is_mut = true) 
        : buffer(buf), size(s), is_mutable(is_mut) { }
      SemanticInfo(RtUserEvent ready)
        : buffer(NULL), size(0), ready_event(ready), is_mutable(true) { }
    public:
      inline bool is_valid(void) const { return ready_event.has_triggered(); }
    public:
      void *buffer;
      size_t size;
      RtUserEvent ready_event;
      bool is_mutable;
    }; 

    /////////////////////////////////////////////////////////////
    // Rez Checker 
    /////////////////////////////////////////////////////////////
    /*
     * Helps in making the calls to begin and end context for
     * both Serializer and Deserializer classes.
     */
    class RezCheck {
    public:
      RezCheck(Serializer &r) : rez(r) { rez.begin_context(); }
      RezCheck(RezCheck &rhs) : rez(rhs.rez) { assert(false); }
      ~RezCheck(void) { rez.end_context(); }
    public:
      inline RezCheck& operator=(const RezCheck &rhs)
        { assert(false); return *this; }
    private:
      Serializer &rez;
    };
    // Same thing except for deserializers, yes we could template
    // it, but then we have to type out to explicitly instantiate
    // the template on the constructor call and that is a lot of
    // unnecessary typing.
    class DerezCheck {
    public:
      DerezCheck(Deserializer &r) : derez(r) { derez.begin_context(); }
      DerezCheck(DerezCheck &rhs) : derez(rhs.derez) { assert(false); }
      ~DerezCheck(void) { derez.end_context(); }
    public:
      inline DerezCheck& operator=(const DerezCheck &rhs)
        { assert(false); return *this; }
    private:
      Deserializer &derez;
    };

    /////////////////////////////////////////////////////////////
    // Fraction 
    /////////////////////////////////////////////////////////////
    template<typename T>
    class Fraction {
    public:
      Fraction(void);
      Fraction(T num, T denom);
      Fraction(const Fraction<T> &f);
    public:
      void divide(T factor);
      void add(const Fraction<T> &rhs);
      void subtract(const Fraction<T> &rhs);
      // Return a fraction that can be taken from this fraction 
      // such that it leaves at least 1/ways parts local after (ways-1) portions
      // are taken from this instance
      Fraction<T> get_part(T ways);
    public:
      bool is_whole(void) const;
      bool is_empty(void) const;
    public:
      inline T get_num(void) const { return numerator; }
      inline T get_denom(void) const { return denominator; }
    public:
      Fraction<T>& operator=(const Fraction<T> &rhs);
    private:
      T numerator;
      T denominator;
    };

    class BitMaskHelper {
    public:
      // Allocates memory that becomes owned by the caller
      static char* to_string(const uint64_t *bits, int count);
    };

    /////////////////////////////////////////////////////////////
    // Bit Mask 
    /////////////////////////////////////////////////////////////
    template<typename T, unsigned int MAX,
             unsigned int SHIFT, unsigned int MASK>
    class BitMask : public Internal::LegionHeapify<BitMask<T,MAX,SHIFT,MASK> > {
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
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
      public Internal::LegionHeapify<TLBitMask<T,MAX,SHIFT,MASK> > {
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
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
#if __cplusplus >= 201103L
    class alignas(16) SSEBitMask 
#else
    class SSEBitMask // alignment handled below
#endif
      : public Internal::LegionHeapify<SSEBitMask<MAX> > {
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
      inline const __m128i& operator()(const unsigned &idx) const;
      inline __m128i& operator()(const unsigned &idx);
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      static inline int pop_count(const SSEBitMask<MAX> &mask);
    protected:
      union {
        __m128i sse_vector[MAX/128];
        uint64_t bit_vector[MAX/64];
      } bits;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
#if __cplusplus >= 201103L
    }; // alignment handled above
#else
    } __attribute__((aligned(16)));
#endif

    /////////////////////////////////////////////////////////////
    // SSE Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
#if __cplusplus >= 201103L
    class alignas(16) SSETLBitMask
#else
    class SSETLBitMask 
#endif
      : public Internal::LegionHeapify<SSETLBitMask<MAX> > {
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
      inline const __m128i& operator()(const unsigned &idx) const;
      inline __m128i& operator()(const unsigned &idx);
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      static inline int pop_count(const SSETLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__m128i value);
    protected:
      union {
        __m128i sse_vector[MAX/128];
        uint64_t bit_vector[MAX/64];
      } bits;
      uint64_t sum_mask;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
#if __cplusplus >= 201103L
    };
#else
    } __attribute__((aligned(16)));
#endif
#endif // __SSE2__

#ifdef __AVX__
    /////////////////////////////////////////////////////////////
    // AVX Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
#if __cplusplus >= 201103L
    class alignas(32) AVXBitMask 
#else
    class AVXBitMask // alignment handled below
#endif
      : public Internal::LegionHeapify<AVXBitMask<MAX> > {
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
      inline const __m256i& operator()(const unsigned &idx) const;
      inline __m256i& operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline AVXBitMask& operator=(const AVXBitMask &rhs);
      inline const __m256d& elem(const unsigned &idx) const;
      inline __m256d& elem(const unsigned &idx);
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      static inline int pop_count(const AVXBitMask<MAX> &mask);
    protected:
      union {
        __m256i avx_vector[MAX/256];
        __m256d avx_double[MAX/256];
        uint64_t bit_vector[MAX/64];
      } bits;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
#if __cplusplus >= 201103L
    }; // alignment handled above
#else
    } __attribute__((aligned(32)));
#endif
    
    /////////////////////////////////////////////////////////////
    // AVX Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
#if __cplusplus >= 201103L
    class alignas(32) AVXTLBitMask
#else
    class AVXTLBitMask // alignment handled below
#endif
      : public Internal::LegionHeapify<AVXTLBitMask<MAX> > {
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
      inline const __m256i& operator()(const unsigned &idx) const;
      inline __m256i& operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline AVXTLBitMask& operator=(const AVXTLBitMask &rhs);
      inline const __m256d& elem(const unsigned &idx) const;
      inline __m256d& elem(const unsigned &idx);
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      static inline int pop_count(const AVXTLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__m256i value);
      static inline uint64_t extract_mask(__m256d value);
    protected:
      union {
        __m256i avx_vector[MAX/256];
        __m256d avx_double[MAX/256];
        uint64_t bit_vector[MAX/64];
      } bits;
      uint64_t sum_mask;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
#if __cplusplus >= 201103L
    }; // alignment handled above
#else
    } __attribute__((aligned(32)));
#endif
#endif // __AVX__

#ifdef __ALTIVEC__
    /////////////////////////////////////////////////////////////
    // PPC Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class PPCBitMask // alignment handled below
      : public Internal::LegionHeapify<PPCBitMask<MAX> > {
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
      inline const __vector unsigned long long& 
        operator()(const unsigned &idx) const;
      inline __vector unsigned long long& operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline PPCBitMask& operator=(const PPCBitMask &rhs);
      inline const __vector double& elem(const unsigned &idx) const;
      inline __vector double& elem(const unsigned &idx);
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      static inline int pop_count(const PPCBitMask<MAX> &mask);
    protected:
      union {
        __vector unsigned long long ppc_vector[MAX/128];
        __vector double ppc_double[MAX/128];
        uint64_t bit_vector[MAX/64];
      } bits;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
    } __attribute__((aligned(16)));
    
    /////////////////////////////////////////////////////////////
    // PPC Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class PPCTLBitMask // alignment handled below
      : public Internal::LegionHeapify<PPCTLBitMask<MAX> > {
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
      inline const __vector unsigned long long& 
        operator()(const unsigned &idx) const;
      inline __vector unsigned long long& operator()(const unsigned &idx);
      inline const uint64_t& operator[](const unsigned &idx) const;
      inline uint64_t& operator[](const unsigned &idx);
      inline PPCTLBitMask& operator=(const PPCTLBitMask &rhs);
      inline const __vector double& elem(const unsigned &idx) const;
      inline __vector double& elem(const unsigned &idx);
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      static inline int pop_count(const PPCTLBitMask<MAX> &mask);
      static inline uint64_t extract_mask(__vector unsigned long long value);
    protected:
      union {
        __vector unsigned long long ppc_vector[MAX/128];
        __vector double ppc_double[MAX/128];
        uint64_t bit_vector[MAX/64];
      } bits;
      uint64_t sum_mask;
    public:
      static const unsigned ELEMENT_SIZE = 64;
      static const unsigned ELEMENTS = MAX/ELEMENT_SIZE;
    } __attribute__((aligned(16)));
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
    public:
      // Allocates memory that becomes owned by the caller
      inline char* to_string(void) const;
    public:
      static inline int pop_count(const 
                        CompoundBitMask<BITMASK,MAX,WORDS> &mask);
    protected:
      uint64_t bits[WORDS];
    public:
      static const int ELEMENTS = 1;
      static const int ELEMENT_SIZE = MAX;
    };

    /////////////////////////////////////////////////////////////
    // Bit Permutation 
    /////////////////////////////////////////////////////////////
    /*
     * This is a class used for storing and performing fast 
     * permutations of bit mask objects.  It is based on sections
     * 7.4 and 7.5 of the first edition of Hacker's Delight.
     * The bit mask MAX field must be a power of 2 and the
     * second template parameter passed to this class must be
     * the log2(MAX) from the bit mask's type.
     *
     * The permutation initially begins as the identity partition
     * and is modified by the send_to command.  The send_to
     * command maintains the invariant that a partition is 
     * always represented for correctness.
     *
     *
     */
    template<typename BITMASK, unsigned LOG2MAX>
    class BitPermutation : 
      public Internal::LegionHeapify<BitPermutation<BITMASK,LOG2MAX> > {
    public:
      BitPermutation(void);
      BitPermutation(const BitPermutation &rhs);
    public:
      inline bool is_identity(bool retest = false);
      inline void send_to(unsigned src, unsigned dst);
    public:
      inline BITMASK generate_permutation(const BITMASK &mask);
      inline void permute(BITMASK &mask);
    protected:
      inline BITMASK sheep_and_goats(const BITMASK &x, const BITMASK &m);
      inline BITMASK compress_left(BITMASK x, BITMASK m);
      inline BITMASK compress_right(BITMASK x, BITMASK m);
    protected:
      inline void set_edge(unsigned src, unsigned dst);
      inline unsigned get_src(unsigned dst);
      inline unsigned get_dst(unsigned src);
    protected:
      void compress_representation(void);
      void test_identity(void);
    protected:
      bool dirty;
      bool identity;
      BITMASK p[LOG2MAX];
      BITMASK comp[LOG2MAX];
    };

    /////////////////////////////////////////////////////////////
    // Index Set 
    /////////////////////////////////////////////////////////////
    template<typename IT/*int type*/, typename DT/*dense type (BitMask)*/,
             bool BIDIR/* = false (bi-directional)*/>
    class IntegerSet {
    public:
      // Size of an STL Node object in bytes
      // This value is approximated over different STL
      // implementations but in general it should be close
      static const size_t STL_SET_NODE_SIZE = 32;
    public:
      // Need to inherit form LegionHeapify for alignment
      struct DenseSet : public Internal::LegionHeapify<DenseSet> {
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
      inline void serialize(Serializer &rez) const;
      inline void deserialize(Deserializer &derez);
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

    /////////////////////////////////////////////////////////////
    // Dynamic Table 
    /////////////////////////////////////////////////////////////
    template<typename IT>
    struct DynamicTableNodeBase {
    public:
      DynamicTableNodeBase(int _level, IT _first_index, IT _last_index)
        : level(_level), first_index(_first_index), 
          last_index(_last_index) { }
      virtual ~DynamicTableNodeBase(void) { }
    public:
      const int level;
      const IT first_index, last_index;
      mutable LocalLock lock;
    };

    template<typename ET, size_t _SIZE, typename IT>
    struct DynamicTableNode : public DynamicTableNodeBase<IT> {
    public:
      static const size_t SIZE = _SIZE;
    public:
      DynamicTableNode(int _level, IT _first_index, IT _last_index)
        : DynamicTableNodeBase<IT>(_level, _first_index, _last_index) 
      { 
        for (size_t i = 0; i < SIZE; i++)
          elems[i] = 0;
      }
      DynamicTableNode(const DynamicTableNode &rhs) { assert(false); }
      virtual ~DynamicTableNode(void)
      {
        for (size_t i = 0; i < SIZE; i++)
        {
          if (elems[i] != 0)
            delete elems[i];
        }
      }
    public:
      DynamicTableNode& operator=(const DynamicTableNode &rhs)
        { assert(false); return *this; }
    public:
      ET *elems[SIZE];
    };

    template<typename ET, size_t _SIZE, typename IT>
    struct LeafTableNode : public DynamicTableNodeBase<IT> {
    public:
      static const size_t SIZE = _SIZE;
    public:
      LeafTableNode(int _level, IT _first_index, IT _last_index)
        : DynamicTableNodeBase<IT>(_level, _first_index, _last_index) 
      { 
        for (size_t i = 0; i < SIZE; i++)
          elems[i] = 0;
      }
      LeafTableNode(const LeafTableNode &rhs) { assert(false); }
      virtual ~LeafTableNode(void)
      {
        for (size_t i = 0; i < SIZE; i++)
        {
          if (elems[i] != 0)
          {
            delete elems[i];
          }
        }
      }
    public:
      LeafTableNode& operator=(const LeafTableNode &rhs)
        { assert(false); return *this; }
    public:
      ET *elems[SIZE];
    };

    template<typename ALLOCATOR>
    class DynamicTable {
    public:
      typedef typename ALLOCATOR::IT IT;
      typedef typename ALLOCATOR::ET ET;
      typedef DynamicTableNodeBase<IT> NodeBase;
    public:
      DynamicTable(void);
      DynamicTable(const DynamicTable &rhs);
      ~DynamicTable(void);
    public:
      DynamicTable& operator=(const DynamicTable &rhs);
    public:
      size_t max_entries(void) const;
      bool has_entry(IT index) const;
      ET* lookup_entry(IT index);
      template<typename T>
      ET* lookup_entry(IT index, const T &arg);
      template<typename T1, typename T2>
      ET* lookup_entry(IT index, const T1 &arg1, const T2 &arg2);
    protected:
      NodeBase* new_tree_node(int level, IT first_index, IT last_index);
      NodeBase* lookup_leaf(IT index);
    protected:
      NodeBase *volatile root;
      mutable LocalLock lock; 
    };

    template<typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
    class DynamicTableAllocator {
    public:
      typedef _ET ET;
      static const size_t INNER_BITS = _INNER_BITS;
      static const size_t LEAF_BITS = _LEAF_BITS;

      typedef LocalLock LT;
      typedef int IT;
      typedef DynamicTableNode<DynamicTableNodeBase<IT>,
                               1 << INNER_BITS, IT> INNER_TYPE;
      typedef LeafTableNode<ET, 1 << LEAF_BITS, IT> LEAF_TYPE;

      static LEAF_TYPE* new_leaf_node(IT first_index, IT last_index)
      {
        return new LEAF_TYPE(0/*level*/, first_index, last_index);
      }
    };
  }; // namspace Internal

    //--------------------------------------------------------------------------
    // Give the implementations here so the templates get instantiated
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    inline Serializer& Serializer::operator=(const Serializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Serializer::serialize(const T &element)
    //--------------------------------------------------------------------------
    {
      while ((index + sizeof(T)) > total_bytes)
        resize();
      *((T*)(buffer+index)) = element;
      index += sizeof(T);
#ifdef DEBUG_LEGION
      context_bytes += sizeof(T);
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Serializer::serialize<bool>(const bool &element)
    //--------------------------------------------------------------------------
    {
      while ((index + 4) > total_bytes)
        resize();
      *((bool*)buffer+index) = element;
      index += 4;
#ifdef DEBUG_LEGION
      context_bytes += 4;
#endif
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Serializer::serialize(
                                const Internal::BitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Serializer::serialize(
                              const Internal::TLBitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

#ifdef __SSE2__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::SSEBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::SSETLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

#ifdef __AVX__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::AVXBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::AVXTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

#ifdef __ALTIVEC__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const PPCBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const PPCTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

    //--------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline void Serializer::serialize(
                               const Internal::IntegerSet<IT,DT,BIDIR> &int_set)
    //--------------------------------------------------------------------------
    {
      int_set.serialize(*this);
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const Domain &dom)
    //--------------------------------------------------------------------------
    {
      serialize(dom.dim);
      if (dom.dim == 0)
        serialize(dom.is_id);
      else
      {
        for (int i = 0; i < 2*dom.dim; i++)
          serialize(dom.rect_data[i]);
      }
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      serialize(dp.dim);
      if (dp.dim == 0)
        serialize(dp.point_data[0]);
      else
      {
        for (int idx = 0; idx < dp.dim; idx++)
          serialize(dp.point_data[idx]);
      }
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const void *src, size_t bytes)
    //--------------------------------------------------------------------------
    {
      while ((index + bytes) > total_bytes)
        resize();
      memcpy(buffer+index,src,bytes);
      index += bytes;
#ifdef DEBUG_LEGION
      context_bytes += bytes;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::begin_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      while ((index + sizeof(size_t)) > total_bytes)
        resize();
      *((size_t*)(buffer+index)) = context_bytes;
      index += sizeof(size_t);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::end_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Save the size into the buffer
      while ((index + sizeof(size_t)) > total_bytes)
        resize();
      *((size_t*)(buffer+index)) = context_bytes;
      index += sizeof(size_t);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void* Serializer::reserve_bytes(size_t bytes)
    //--------------------------------------------------------------------------
    {
      while ((index + bytes) > total_bytes)
        resize();
      void *result = buffer+index;
      index += bytes;
#ifdef DEBUG_LEGION
      context_bytes += bytes;
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline void Serializer::resize(void)
    //--------------------------------------------------------------------------
    {
      // Double the buffer size
      total_bytes *= 2;
#ifdef DEBUG_LEGION
      assert(total_bytes != 0); // this would cause deallocation
#endif
      char *next = (char*)realloc(buffer,total_bytes);
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      buffer = next;
    }

    //--------------------------------------------------------------------------
    inline Deserializer& Deserializer::operator=(const Deserializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Deserializer::deserialize(T &element)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check to make sure we don't read past the end
      assert((index+sizeof(T)) <= total_bytes);
#endif
      element = *((const T*)(buffer+index));
      index += sizeof(T);
#ifdef DEBUG_LEGION
      context_bytes += sizeof(T);
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Deserializer::deserialize<bool>(bool &element)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check to make sure we don't read past the end
      assert((index+4) <= total_bytes);
#endif
      element = *((const bool *)(buffer+index));
      index += 4;
#ifdef DEBUG_LEGION
      context_bytes += 4;
#endif
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Deserializer::deserialize(
                                      Internal::BitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Deserializer::deserialize(
                                    Internal::TLBitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

#ifdef __SSE2__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::SSEBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::SSETLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

#ifdef __AVX__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::AVXBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::AVXTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

#ifdef __ALTIVEC__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(PPCBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(PPCTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

    //--------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline void Deserializer::deserialize(
                                     Internal::IntegerSet<IT,DT,BIDIR> &int_set)
    //--------------------------------------------------------------------------
    {
      int_set.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(Domain &dom)
    //--------------------------------------------------------------------------
    {
      deserialize(dom.dim);
      if (dom.dim == 0)
        deserialize(dom.is_id);
      else
      {
        for (int i = 0; i < 2*dom.dim; i++)
          deserialize(dom.rect_data[i]);
      }
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      deserialize(dp.dim);
      if (dp.dim == 0)
        deserialize(dp.point_data[0]);
      else
      {
        for (int idx = 0; idx < dp.dim; idx++)
          deserialize(dp.point_data[idx]);
      }
    }
      
    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(void *dst, size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((index + bytes) <= total_bytes);
#endif
      memcpy(dst,buffer+index,bytes);
      index += bytes;
#ifdef DEBUG_LEGION
      context_bytes += bytes;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::begin_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Save our enclosing context on the stack
#ifndef NDEBUG
      size_t sent_context = *((const size_t*)(buffer+index));
#endif
      index += sizeof(size_t);
      // Check to make sure that they match
      assert(sent_context == context_bytes);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::end_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Read the send context size out of the buffer      
#ifndef NDEBUG
      size_t sent_context = *((const size_t*)(buffer+index));
#endif
      index += sizeof(size_t);
      // Check to make sure that they match
      assert(sent_context == context_bytes);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline size_t Deserializer::get_remaining_bytes(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index <= total_bytes);
#endif
      return total_bytes - index;
    }

    //--------------------------------------------------------------------------
    inline const void* Deserializer::get_current_pointer(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index <= total_bytes);
#endif
      return (const void*)(buffer+index);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::advance_pointer(size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((index+bytes) <= total_bytes);
      context_bytes += bytes;
#endif
      index += bytes;
    }

  namespace Internal {
    // There is an interesting design decision about how to break up the 32 bit
    // address space for fractions.  We'll assume that there will be some
    // balance between the depth and breadth of the task tree so we can split up
    // the fractions efficiently.  We assume that there will be large fan-outs
    // in the task tree as well as potentially large numbers of task calls at
    // each node.  However, we'll assume that the tree is not very deep.
#define MIN_FRACTION_SPLIT    256
    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>::Fraction(void)
      : numerator(256), denominator(256)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>::Fraction(T num, T denom)
      : numerator(num), denominator(denom)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(denom > 0);
#endif
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>::Fraction(const Fraction<T> &f)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(f.denominator > 0);
#endif
      numerator = f.numerator;
      denominator = f.denominator;
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::divide(T factor)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(factor != 0);
      assert(denominator > 0);
#endif
      T new_denom = denominator * factor;
#ifdef DEBUG_LEGION
      assert(new_denom > 0); // check for integer overflow
#endif
      denominator = new_denom;
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::add(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(denominator > 0);
#endif
      if (denominator == rhs.denominator)
      {
        numerator += rhs.numerator;
      }
      else
      {
        // Denominators are different, make them the same
        // Check if one denominator is divisible by another
        if ((denominator % rhs.denominator) == 0)
        {
          // Our denominator is bigger
          T factor = denominator/rhs.denominator; 
          numerator += (rhs.numerator*factor);
        }
        else if ((rhs.denominator % denominator) == 0)
        {
          // Rhs denominator is bigger
          T factor = rhs.denominator/denominator;
          numerator = (numerator*factor) + rhs.numerator;
          denominator *= factor;
#ifdef DEBUG_LEGION
          assert(denominator > 0); // check for integer overflow
#endif
        }
        else
        {
          // One denominator is not divisible by the other, 
          // compute a common denominator
          T lhs_num = numerator * rhs.denominator;
          T rhs_num = rhs.numerator * denominator;
          numerator = lhs_num + rhs_num;
          denominator *= rhs.denominator;
#ifdef DEBUG_LEGION
          assert(denominator > 0); // check for integer overflow
#endif
        }
      }
#ifdef DEBUG_LEGION
      // Should always be less than or equal to 1
      assert(numerator <= denominator); 
#endif
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::subtract(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(denominator > 0);
#endif
      if (denominator == rhs.denominator)
      {
#ifdef DEBUG_LEGION
        assert(numerator >= rhs.numerator); 
#endif
        numerator -= rhs.numerator;
      }
      else
      {
        if ((denominator % rhs.denominator) == 0)
        {
          // Our denominator is bigger
          T factor = denominator/rhs.denominator;
#ifdef DEBUG_LEGION
          assert(numerator >= (rhs.numerator*factor));
#endif
          numerator -= (rhs.numerator*factor);
        }
        else if ((rhs.denominator % denominator) == 0)
        {
          // Rhs denominator is bigger
          T factor = rhs.denominator/denominator;
#ifdef DEBUG_LEGION
          assert((numerator*factor) >= rhs.numerator);
#endif
          numerator = (numerator*factor) - rhs.numerator;
          denominator *= factor;
#ifdef DEBUG_LEGION
          assert(denominator > 0); // check for integer overflow
#endif
        }
        else
        {
          // One denominator is not divisible by the other, 
          // compute a common denominator
          T lhs_num = numerator * rhs.denominator;
          T rhs_num = rhs.numerator * denominator;
#ifdef DEBUG_LEGION
          assert(lhs_num >= rhs_num);
#endif
          numerator = lhs_num - rhs_num;
          denominator *= rhs.denominator; 
#ifdef DEBUG_LEGION
          assert(denominator > 0); // check for integer overflow
#endif
        }
      }
      // Check to see if the numerator has gotten down to one, 
      // if so bump up the fraction split
      if (numerator == 1)
      {
        numerator *= MIN_FRACTION_SPLIT;
        denominator *= MIN_FRACTION_SPLIT;
#ifdef DEBUG_LEGION
        assert(denominator > 0); // check for integer overflow
#endif
      }
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T> Fraction<T>::get_part(T ways)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ways > 0);
      assert(denominator > 0);
      assert(numerator > 0);
#endif
      // Check to see if we have enough parts in the numerator, if not
      // multiply both numerator and denominator by ways
      // and return one over denominator
      if (ways >= numerator)
      {
        // Check to see if the ways is at least as big as 
        // the minimum split factor
        if (ways < MIN_FRACTION_SPLIT)
        {
          ways = MIN_FRACTION_SPLIT;
        }
        numerator *= ways;
        T new_denom = denominator * ways;
#ifdef DEBUG_LEGION
        assert(new_denom > 0); // check for integer overflow
#endif
        denominator = new_denom;
      }
#ifdef DEBUG_LEGION
      assert(numerator >= ways);
#endif
      return Fraction(1,denominator);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    bool Fraction<T>::is_whole(void) const
    //-------------------------------------------------------------------------
    {
      return (numerator == denominator);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    bool Fraction<T>::is_empty(void) const
    //-------------------------------------------------------------------------
    {
      return (numerator == 0);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>& Fraction<T>::operator=(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
      numerator = rhs.numerator;
      denominator = rhs.denominator;
      return *this;
    }
#undef MIN_FRACTION_SPLIT

#define BIT_ELMTS (MAX/(8*sizeof(T)))
    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    BitMask<T,MAX,SHIFT,MASK>::BitMask(T init /*= 0*/)
    //-------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
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
      LEGION_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
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
    inline void BitMask<T,MAX,SHIFT,MASK>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline void BitMask<T,MAX,SHIFT,MASK>::deserialize(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned MAX, unsigned SHIFT, unsigned MASK>
    inline char* BitMask<T,MAX,SHIFT,MASK>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelper::to_string(bit_vector, MAX);
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
      LEGION_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        bit_vector[idx] = init;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    TLBitMask<T,MAX,SHIFT,MASK>::TLBitMask(const TLBitMask &rhs)
      : sum_mask(rhs.sum_mask)
    //-------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT((MAX % (8*sizeof(T))) == 0);
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
    inline void TLBitMask<T,MAX,SHIFT,MASK>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.template serialize(sum_mask);
      rez.serialize(bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void TLBitMask<T,MAX,SHIFT,MASK>::deserialize(Deserializer &derez)
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
      return BitMaskHelper::to_string(bit_vector, MAX);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_vector[idx] = rhs(idx);
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
        bits.sse_vector[idx] = _mm_set1_epi32(0);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __m128i& SSEBitMask<MAX>::operator()(
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.sse_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m128i& SSEBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.sse_vector[idx];
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
        bits.sse_vector[idx] = rhs(idx);
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
        result(idx) = _mm_or_si128(bits.sse_vector[idx], rhs(idx));
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
        result(idx) = _mm_and_si128(bits.sse_vector[idx], rhs(idx));
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
        result(idx) = _mm_xor_si128(bits.sse_vector[idx], rhs(idx));
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
        bits.sse_vector[idx] = _mm_or_si128(bits.sse_vector[idx], rhs(idx));
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
        bits.sse_vector[idx] = _mm_and_si128(bits.sse_vector[idx], rhs(idx));
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
        bits.sse_vector[idx] = _mm_xor_si128(bits.sse_vector[idx], rhs(idx));
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
        result(idx) = _mm_andnot_si128(rhs(idx), bits.sse_vector[idx]);
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
        bits.sse_vector[idx] = _mm_andnot_si128(rhs(idx), bits.sse_vector[idx]);
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
    template<unsigned int MAX>
    inline void SSEBitMask<MAX>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSEBitMask<MAX>::deserialize(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* SSEBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelper::to_string(bits.bit_vector, MAX);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
      for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
      {
        bits.sse_vector[idx] = rhs(idx);
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
        bits.sse_vector[idx] = _mm_set1_epi32(0); 
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __m128i& SSETLBitMask<MAX>::operator()(
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.sse_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m128i& SSETLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.sse_vector[idx];
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
        bits.sse_vector[idx] = rhs(idx);
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
        result(idx) = _mm_or_si128(bits.sse_vector[idx], rhs(idx));
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
          result(idx) = _mm_and_si128(bits.sse_vector[idx], rhs(idx));
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
        result(idx) = _mm_xor_si128(bits.sse_vector[idx], rhs(idx));
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
        bits.sse_vector[idx] = _mm_or_si128(bits.sse_vector[idx], rhs(idx));
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
          bits.sse_vector[idx] = _mm_and_si128(bits.sse_vector[idx], rhs(idx));
          temp_sum = _mm_or_si128(temp_sum, bits.sse_vector[idx]);
        }
        sum_mask = extract_mask(temp_sum); 
      }
      else
      {
        sum_mask = 0;
        for (unsigned idx = 0; idx < SSE_ELMTS; idx++)
          bits.sse_vector[idx] = _mm_set1_epi32(0);
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
        bits.sse_vector[idx] = _mm_xor_si128(bits.sse_vector[idx], rhs(idx));
        temp_sum = _mm_or_si128(temp_sum, bits.sse_vector[idx]);
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
        result(idx) = _mm_andnot_si128(rhs(idx), bits.sse_vector[idx]);
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
        bits.sse_vector[idx] = _mm_andnot_si128(rhs(idx), bits.sse_vector[idx]);
        temp_sum = _mm_or_si128(temp_sum, bits.sse_vector[idx]);
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
    template<unsigned int MAX>
    inline void SSETLBitMask<MAX>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sum_mask);
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void SSETLBitMask<MAX>::deserialize(Deserializer &derez)
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
      return BitMaskHelper::to_string(bits.bit_vector, MAX);
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
      LEGION_STATIC_ASSERT((MAX % 256) == 0);
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
      LEGION_STATIC_ASSERT((MAX % 256) == 0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_vector[idx] = rhs(idx);
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
        bits.avx_vector[idx] = _mm256_set1_epi32(0);
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __m256i& AVXBitMask<MAX>::operator()(
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.avx_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256i& AVXBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.avx_vector[idx];
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
    inline const __m256d& AVXBitMask<MAX>::elem(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.avx_double[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256d& AVXBitMask<MAX>::elem(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.avx_double[idx];
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
        bits.avx_vector[idx] = rhs(idx);
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
        result(idx) = _mm256_or_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result.elem(idx) = _mm256_or_pd(bits.avx_double[idx],
                                        rhs.elem(idx));
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
        result(idx) = _mm256_and_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result.elem(idx) = _mm256_and_pd(bits.avx_double[idx],
                                         rhs.elem(idx));
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
        result(idx) = _mm256_xor_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result.elem(idx) = _mm256_xor_pd(bits.avx_double[idx],
                                         rhs.elem(idx));
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
        bits.avx_vector[idx] = _mm256_or_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_double[idx] = _mm256_or_pd(bits.avx_double[idx],rhs.elem(idx));
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
        bits.avx_vector[idx] = _mm256_and_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_double[idx] = _mm256_and_pd(bits.avx_double[idx],
                                             rhs.elem(idx));
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
        bits.avx_vector[idx] = _mm256_xor_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_double[idx] = _mm256_xor_pd(bits.avx_double[idx], 
                                             rhs.elem(idx));
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
        result(idx) = _mm256_andnot_si256(rhs(idx), bits.avx_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result.elem(idx) = _mm256_andnot_pd(rhs.elem(idx),
                                            bits.avx_double[idx]);
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
        bits.avx_vector[idx] = _mm256_andnot_si256(rhs(idx), 
                                                   bits.avx_vector[idx]);
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_double[idx] = _mm256_andnot_pd(rhs.elem(idx),
                                                bits.avx_double[idx]);
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
    template<unsigned int MAX>
    inline void AVXBitMask<MAX>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXBitMask<MAX>::deserialize(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* AVXBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelper::to_string(bits.bit_vector, MAX);
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
      LEGION_STATIC_ASSERT((MAX % 256) == 0);
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
      LEGION_STATIC_ASSERT((MAX % 256) == 0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_vector[idx] = rhs(idx);
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
        bits.avx_vector[idx] = _mm256_set1_epi32(0);
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __m256i& AVXTLBitMask<MAX>::operator()(
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.avx_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256i& AVXTLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.avx_vector[idx];
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
    inline const __m256d& AVXTLBitMask<MAX>::elem(const unsigned &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.avx_double[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256d& AVXTLBitMask<MAX>::elem(const unsigned &idx)
    //-------------------------------------------------------------------------
    {
      return bits.avx_double[idx];
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
        bits.avx_vector[idx] = rhs(idx);
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
        result(idx) = _mm256_or_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result.elem(idx) = _mm256_or_pd(bits.avx_double[idx], rhs.elem(idx));
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
          result(idx) = _mm256_and_si256(bits.avx_vector[idx], rhs(idx));
          temp_sum = _mm256_or_si256(temp_sum, result(idx));
        }
#else
        __m256d temp_sum = _mm256_set1_pd(0.0);
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
        {
          result.elem(idx) = _mm256_and_pd(bits.avx_double[idx],rhs.elem(idx));
          temp_sum = _mm256_or_pd(temp_sum, result.elem(idx));
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
        result(idx) = _mm256_xor_si256(bits.avx_vector[idx], rhs(idx));
        temp_sum = _mm256_or_si256(temp_sum, result(idx));
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result.elem(idx) = _mm256_xor_pd(bits.avx_double[idx], rhs.elem(idx));
        temp_sum = _mm256_or_pd(temp_sum, result.elem(idx));
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
        bits.avx_vector[idx] = _mm256_or_si256(bits.avx_vector[idx], rhs(idx));
      }
#else
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_double[idx] = _mm256_or_pd(bits.avx_double[idx],rhs.elem(idx));
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
          bits.avx_vector[idx] = _mm256_and_si256(bits.avx_vector[idx], 
                                                  rhs(idx));
          temp_sum = _mm256_or_si256(temp_sum, bits.avx_vector[idx]);
        }
#else
        __m256d temp_sum = _mm256_set1_pd(0.0);
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
        {
          bits.avx_double[idx] = _mm256_and_pd(bits.avx_double[idx], 
                                               rhs.elem(idx));
          temp_sum = _mm256_or_pd(temp_sum, bits.avx_double[idx]);
        }
#endif
        sum_mask = extract_mask(temp_sum); 
      }
      else
      {
        sum_mask = 0;
        for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
          bits.avx_vector[idx] = _mm256_set1_epi32(0);
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
        bits.avx_vector[idx] = _mm256_xor_si256(bits.avx_vector[idx], rhs(idx));
        temp_sum = _mm256_or_si256(temp_sum, bits.avx_vector[idx]);
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_double[idx] = _mm256_xor_pd(bits.avx_double[idx], 
                                             rhs.elem(idx));
        temp_sum = _mm256_or_pd(temp_sum, bits.avx_double[idx]);
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
        result(idx) = _mm256_andnot_si256(rhs(idx), bits.avx_vector[idx]);
        temp_sum = _mm256_or_si256(temp_sum, result(idx));
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        result.elem(idx) = _mm256_andnot_pd(rhs.elem(idx),bits.avx_double[idx]);
        temp_sum = _mm256_or_pd(temp_sum, result.elem(idx));
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
        bits.avx_vector[idx] = _mm256_andnot_si256(rhs(idx), 
                                                   bits.avx_vector[idx]);
        temp_sum = _mm256_or_si256(temp_sum, bits.avx_vector[idx]);
      }
#else
      __m256d temp_sum = _mm256_set1_pd(0.0);
      for (unsigned idx = 0; idx < AVX_ELMTS; idx++)
      {
        bits.avx_double[idx] = _mm256_andnot_pd(rhs.elem(idx),
                                                bits.avx_double[idx]);
        temp_sum = _mm256_or_pd(temp_sum, bits.avx_double[idx]);
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
    template<unsigned int MAX>
    inline void AVXTLBitMask<MAX>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sum_mask);
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXTLBitMask<MAX>::deserialize(Deserializer &derez)
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
      return BitMaskHelper::to_string(bits.bit_vector, MAX);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_vector[idx] = rhs(idx);
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
        bits.ppc_vector[idx] = zero_vec;
      }
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __vector unsigned long long& PPCBitMask<MAX>::operator()(
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.ppc_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __vector unsigned long long& PPCBitMask<MAX>::operator()(
                                                       const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.ppc_vector[idx];
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
        if (bits.ppc_vector[idx] < rhs[idx])
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
        bits.ppc_vector[idx] = rhs(idx);
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
        result(idx) = ~(bits.ppc_vector[idx]);
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
        result(idx) = vec_or(bits.ppc_vector[idx], rhs(idx));
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
        result(idx) = vec_and(bits.ppc_vector[idx], rhs(idx));
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
        result(idx) = vec_xor(bits.ppc_vector[idx], rhs(idx));
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
        bits.ppc_vector[idx] = vec_or(bits.ppc_vector[idx], rhs(idx));
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
        bits.ppc_vector[idx] = vec_and(bits.ppc_vector[idx], rhs(idx));
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
        bits.ppc_vector[idx] = vec_xor(bits.ppc_vector[idx], rhs(idx));
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
        result(idx) = vec_and(~rhs(idx), bits.ppc_vector[idx]);
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
        bits.ppc_vector[idx] = vec_and(~rhs(idx), bits.ppc_vector[idx]);
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
    template<unsigned int MAX>
    inline void PPCBitMask<MAX>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCBitMask<MAX>::deserialize(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline char* PPCBitMask<MAX>::to_string(void) const
    //-------------------------------------------------------------------------
    {
      return BitMaskHelper::to_string(bits.bit_vector, MAX);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
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
      LEGION_STATIC_ASSERT((MAX % 128) == 0);
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_vector[idx] = rhs(idx);
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
        bits.ppc_vector[idx] = zero_vec; 
      }
      sum_mask = 0;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __vector unsigned long long& PPCTLBitMask<MAX>::operator()(
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
      return bits.ppc_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __vector unsigned long long& 
                         PPCTLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
      return bits.ppc_vector[idx];
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
        bits.ppc_vector[idx] = rhs(idx);
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
        result(idx) = ~(bits.ppc_vector[idx]);
        result_mask = vec_or(result_mask, result(idx));
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
        result(idx) = vec_or(bits.ppc_vector[idx], rhs(idx));
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
          result(idx) = vec_and(bits.ppc_vector[idx], rhs(idx));
          temp_sum = vec_or(temp_sum, result(idx));
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
        result(idx) = vec_xor(bits.ppc_vector[idx], rhs(idx));
        temp_sum = vec_or(temp_sum, result(idx));
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
	//bits.ppc_vector[idx] |= rhs(idx);
        bits.ppc_vector[idx] = vec_or(bits.ppc_vector[idx], rhs(idx));
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
          bits.ppc_vector[idx] = vec_and(bits.ppc_vector[idx], rhs(idx));
          temp_sum = vec_or(temp_sum, bits.ppc_vector[idx]);
        }
        sum_mask = extract_mask(temp_sum); 
      }
      else
      {
        sum_mask = 0;
	const __vector unsigned long long zero_vec = vec_splats(0ULL);
        for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
          bits.ppc_vector[idx] = zero_vec;
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline PPCTLBitMask<MAX>& PPCTLBitMask<MAX>::operator^=(
                                                       const PPCTLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      __vector unsigned long long temp_sum = 0;
      for (unsigned idx = 0; idx < PPC_ELMTS; idx++)
      {
        bits.ppc_vector[idx] = vec_xor(bits.ppc_vector[idx], rhs(idx));
        temp_sum = vec_or(temp_sum, bits.ppc_vector[idx]);
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
        result(idx) = vec_and(~rhs(idx), bits.ppc_vector[idx]);
        temp_sum = vec_or(temp_sum, result(idx));
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
        bits.ppc_vector[idx] = vec_and(~rhs(idx), bits.ppc_vector[idx]);
        temp_sum = vec_or(temp_sum, bits.ppc_vector[idx]);
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
    template<unsigned int MAX>
    inline void PPCTLBitMask<MAX>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize(sum_mask);
      rez.serialize(bits.bit_vector, (MAX/8));
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void PPCTLBitMask<MAX>::deserialize(Deserializer &derez)
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
      return BitMaskHelper::to_string(bits.bit_vector, MAX);
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
      LEGION_STATIC_ASSERT(WORDS >= 2);
      LEGION_STATIC_ASSERT(VAL_BITS <= (8*sizeof(uint64_t)));
      LEGION_STATIC_ASSERT((1 << VAL_BITS) >= MAX);
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
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::serialize(Serializer &rez) 
                                                                          const
    //-------------------------------------------------------------------------
    {
      int count = get_count();
      rez.serialize(count);
      if (count == DENSE_CNT)
        get_dense()->serialize(rez); 
      else if (count == SPARSE_CNT)
      {
        SparseSet *sparse = get_sparse();
        rez.serialize<size_t>(sparse->size());
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
    inline void CompoundBitMask<BITMASK,MAX,WORDS>::deserialize(
                                                           Deserializer &derez)
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
    template<typename BITMASK, unsigned LOG2MAX>
    BitPermutation<BITMASK,LOG2MAX>::BitPermutation(void)
      : dirty(false), identity(true)
    //-------------------------------------------------------------------------
    {
      // First zero everything out
      for (unsigned idx = 0; idx < LOG2MAX; idx++)
        p[idx] = BITMASK();
      // Initialize everything to the identity permutation
      for (unsigned idx = 0; idx < (1 << LOG2MAX); idx++)
        set_edge(idx, idx);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    BitPermutation<BITMASK,LOG2MAX>::BitPermutation(const BitPermutation &rhs)
      : dirty(rhs.dirty), identity(rhs.identity)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < LOG2MAX; idx++)
      {
        p[idx] = rhs.p[idx];
        comp[idx] = rhs.comp[idx];
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline bool BitPermutation<BITMASK,LOG2MAX>::is_identity(bool retest)
    //-------------------------------------------------------------------------
    {
      if (identity)
        return true;
      if (retest)
      {
        test_identity();
        return identity;
      }
      return false;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline void BitPermutation<BITMASK,LOG2MAX>::send_to(unsigned src,
                                                         unsigned dst)
    //-------------------------------------------------------------------------
    {
      // If we're already in identity mode and the src
      // and dst are equal then we are done
      if (identity && (src == dst))
        return;
      unsigned old_dst = get_dst(src);
      unsigned old_src = get_src(dst);
      // Check to see if we already had this edge
      if ((old_src == src) && (old_dst == dst))
        return;
      // Otherwise flip the edges and mark that we are no longer
      // in identity mode
      set_edge(src, dst);
      set_edge(old_src, old_dst);
      identity = false;
      dirty = true;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline BITMASK BitPermutation<BITMASK,LOG2MAX>::generate_permutation(
                                                          const BITMASK &mask)
    //-------------------------------------------------------------------------
    {
      if (dirty)
      {
        compress_representation();
        // If we're going to do an expensive operation retest for identity
        test_identity();
      }
      if (identity)
        return mask;
      BITMASK result = mask;
      for (unsigned idx = 0; idx < LOG2MAX; idx++)
        result = sheep_and_goats(result, comp[idx]);
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline void BitPermutation<BITMASK,LOG2MAX>::permute(BITMASK &mask)
    //-------------------------------------------------------------------------
    {
      if (dirty)
      {
        compress_representation();
        // If we're going to do an expensive operation retest for identity
        test_identity();
      }
      if (identity)
        return;
      for (unsigned idx = 0; idx < LOG2MAX; idx++)
        mask = sheep_and_goats(mask, comp[idx]);
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline BITMASK BitPermutation<BITMASK,LOG2MAX>::sheep_and_goats(
                                            const BITMASK &x, const BITMASK &m)
    //-------------------------------------------------------------------------
    {
      return (compress_left(x, m) | compress_right(x, ~m));
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline BITMASK BitPermutation<BITMASK,LOG2MAX>::compress_left(
                                                          BITMASK x, BITMASK m)
    //-------------------------------------------------------------------------
    {
      BITMASK mk, mp, mv, t;

      x = x & m;
      mk = ~m >> 1;

      for (unsigned i = 0; i < LOG2MAX; i++)
      {
        mp = mk ^ (mk >> 1);
        for (unsigned idx = 1; idx < LOG2MAX; idx++)
          mp = mp ^ (mp >> (1 << idx));
        mv = mp & m;
        m = (m ^ mv) | (mv << (1 << i));
        t = x & mv;
        x = (x ^ t) | (t << (1 << i));
        mk = mk & ~mp;
      }
      return x;
    }
    
    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline BITMASK BitPermutation<BITMASK,LOG2MAX>::compress_right(
                                                          BITMASK x, BITMASK m)
    //-------------------------------------------------------------------------
    {
      BITMASK mk, mp, mv, t;

      x = x & m;
      mk = ~m << 1;

      for (unsigned i = 0; i < LOG2MAX; i++)
      {
        mp = mk ^ (mk << 1);
        for (unsigned idx = 1; idx < LOG2MAX; idx++)
          mp = mp ^ (mp << (1 << idx));
        mv = mp & m;
        m = (m ^ mv) | (mv >> (1 << i));
        t = x & mv;
        x = (x ^ t) | (t >> (1 << i));
        mk = mk & ~mp;
      }
      return x;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline void BitPermutation<BITMASK,LOG2MAX>::set_edge(unsigned src,
                                                          unsigned dst)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < LOG2MAX; idx++)
      {
        unsigned bit = (dst & (1 << idx));
        if (bit)
          p[idx].set_bit(src);
        else
          p[idx].unset_bit(src);
      }
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline unsigned BitPermutation<BITMASK,LOG2MAX>::get_src(unsigned dst)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < (1 << LOG2MAX); idx++)
      {
        if (get_dst(idx) == dst)
          return idx;
      }
      // If we ever get here then we no longer had a permutation
      assert(false);
      return 0;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    inline unsigned BitPermutation<BITMASK,LOG2MAX>::get_dst(unsigned src)
    //-------------------------------------------------------------------------
    {
      unsigned dst = 0;
      for (unsigned idx = 0; idx < LOG2MAX; idx++)
      {
        if (p[idx].is_set(src))
          dst |= (1 << idx);
      }
      return dst;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    void BitPermutation<BITMASK,LOG2MAX>::compress_representation(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < LOG2MAX; idx++)
      {
        if (idx == 0)
          comp[0] = p[0];
        else
        {
          comp[idx] = sheep_and_goats(p[idx],comp[0]);
          for (unsigned i = 1; i < idx; i++)
            comp[idx] = sheep_and_goats(comp[idx],comp[i]);
        }
      }
      dirty = false;
    }

    //-------------------------------------------------------------------------
    template<typename BITMASK, unsigned LOG2MAX>
    void BitPermutation<BITMASK,LOG2MAX>::test_identity(void)
    //-------------------------------------------------------------------------
    {
      // If we're still the identity then we're done
      if (identity)
        return;
      for (unsigned idx = 0; idx < (1 << LOG2MAX); idx++)
      {
        unsigned src = 1 << idx;
        unsigned dst = get_dst(src);
        if (src != dst)
        {
          identity = false;
          return;
        }
      }
      identity = true;
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
    template<typename IT, typename DT, bool BIDIR>
    inline void IntegerSet<IT,DT,BIDIR>::serialize(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize<bool>(sparse);
      if (sparse)
      {
	if (set_ptr.sparse)
	{
          rez.serialize<size_t>(set_ptr.sparse->size());
          for (typename std::set<IT>::const_iterator it = 
                set_ptr.sparse->begin(); it != set_ptr.sparse->end(); it++)
          {
            rez.serialize(*it);
          }
        }
	else
          rez.serialize<size_t>(0);
      }
      else
        rez.serialize(set_ptr.dense->set);
    }

    //-------------------------------------------------------------------------
    template<typename IT, typename DT, bool BIDIR>
    inline void IntegerSet<IT,DT,BIDIR>::deserialize(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      bool is_sparse;
      derez.deserialize<bool>(is_sparse);
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
        derez.deserialize<size_t>(num_elements);
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

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::DynamicTable(void)
      : root(0)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::DynamicTable(const DynamicTable &rhs)
    //-------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::~DynamicTable(void)
    //-------------------------------------------------------------------------
    {
      if (root != 0)
      {
        delete root;
        root = NULL;
      }
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    DynamicTable<ALLOCATOR>& 
                    DynamicTable<ALLOCATOR>::operator=(const DynamicTable &rhs)
    //-------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    typename DynamicTable<ALLOCATOR>::NodeBase* 
              DynamicTable<ALLOCATOR>::new_tree_node(int level, IT first_index,
                                                     IT last_index)
    //-------------------------------------------------------------------------
    {
      if (level > 0)
      {
        // we know how to create inner nodes
        typename ALLOCATOR::INNER_TYPE *inner = 
          new typename ALLOCATOR::INNER_TYPE(level, first_index, last_index);
        return inner;
      }
      return ALLOCATOR::new_leaf_node(first_index, last_index);
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    size_t DynamicTable<ALLOCATOR>::max_entries(void) const
    //-------------------------------------------------------------------------
    {
      if (!root)
        return 0;
      size_t elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      for (int i = 0; i < root->level; i++)
        elems_addressable <<= ALLOCATOR::INNER_BITS;
      return elems_addressable;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    bool DynamicTable<ALLOCATOR>::has_entry(IT index) const
    //-------------------------------------------------------------------------
    {
      // first, figure out how many levels the tree must have to find our index
      int level_needed = 0;
      int elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      while(index >= elems_addressable) {
	level_needed++;
	elems_addressable <<= ALLOCATOR::INNER_BITS;
      }

      NodeBase *n = root;
      if (!n || (n->level < level_needed))
        return false;

#ifdef DEBUG_LEGION
      // when we get here, root is high enough
      assert((level_needed <= n->level) &&
	     (index >= n->first_index) &&
	     (index <= n->last_index));
#endif
      // now walk tree, populating the path we need
      while (n->level > 0)
      {
        // intermediate nodes
        typename ALLOCATOR::INNER_TYPE *inner = 
          static_cast<typename ALLOCATOR::INNER_TYPE*>(n);
        IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) *
            ALLOCATOR::INNER_BITS)) & ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
#ifdef DEBUG_LEGION
        assert((i >= 0) && (((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE));
#endif
        NodeBase *child = inner->elems[i];
        if (child == 0)
          return false;
#ifdef DEBUG_LEGION
        assert((child != 0) && 
               (child->level == (n->level -1)) &&
               (index >= child->first_index) &&
               (index <= child->last_index));
#endif
        n = child;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    typename DynamicTable<ALLOCATOR>::ET* 
                                DynamicTable<ALLOCATOR>::lookup_entry(IT index)
    //-------------------------------------------------------------------------
    {
      NodeBase *n = lookup_leaf(index); 
      // Now we've made it to the leaf node
      typename ALLOCATOR::LEAF_TYPE *leaf = 
        static_cast<typename ALLOCATOR::LEAF_TYPE*>(n);
      int offset = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
      ET *result = leaf->elems[offset];
      if (result == 0)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        if (leaf->elems[offset] == 0)
          leaf->elems[offset] = new ET();
        result = leaf->elems[offset];
      }
#ifdef DEBUG_LEGION
      assert(result != 0);
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR> template<typename T>
    typename DynamicTable<ALLOCATOR>::ET*
                  DynamicTable<ALLOCATOR>::lookup_entry(IT index, const T &arg)
    //-------------------------------------------------------------------------
    {
      NodeBase *n = lookup_leaf(index); 
      // Now we've made it to the leaf node
      typename ALLOCATOR::LEAF_TYPE *leaf = 
        static_cast<typename ALLOCATOR::LEAF_TYPE*>(n);
      int offset = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
      ET *result = leaf->elems[offset];
      if (result == 0)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        if (leaf->elems[offset] == 0)
          leaf->elems[offset] = new ET(arg);
        result = leaf->elems[offset];
      }
#ifdef DEBUG_LEGION
      assert(result != 0);
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR> template<typename T1, typename T2>
    typename DynamicTable<ALLOCATOR>::ET*
          DynamicTable<ALLOCATOR>::lookup_entry(IT index, 
                                                const T1 &arg1, const T2 &arg2)
    //-------------------------------------------------------------------------
    {
      NodeBase *n = lookup_leaf(index); 
      // Now we've made it to the leaf node
      typename ALLOCATOR::LEAF_TYPE *leaf = 
        static_cast<typename ALLOCATOR::LEAF_TYPE*>(n);
      int offset = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
      ET *result = leaf->elems[offset];
      if (result == 0)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        if (leaf->elems[offset] == 0)
          leaf->elems[offset] = new ET(arg1, arg2);
        result = leaf->elems[offset];
      }
#ifdef DEBUG_LEGION
      assert(result != 0);
#endif
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    typename DynamicTable<ALLOCATOR>::NodeBase* 
                                 DynamicTable<ALLOCATOR>::lookup_leaf(IT index)
    //-------------------------------------------------------------------------
    {
      // Figure out how many levels need to be in the tree
      int level_needed = 0;  
      int elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      while (index >= elems_addressable)
      {
        level_needed++;
        elems_addressable <<= ALLOCATOR::INNER_BITS;
      }

      // In most cases we won't need to add levels to the tree, but
      // if we do, then do it now
      NodeBase *n = root;
      if (!n || (n->level < level_needed)) 
      {
        AutoLock l(lock); 
        if (root)
        {
          // some of the tree exists - add new layers on top
          while (root->level < level_needed)
          {
            int parent_level = root->level + 1;
            IT parent_first = 0;
            IT parent_last = 
              (((root->last_index + 1) << ALLOCATOR::INNER_BITS) - 1);
            NodeBase *parent = new_tree_node(parent_level, 
                                             parent_first, parent_last);
            typename ALLOCATOR::INNER_TYPE *inner = 
              static_cast<typename ALLOCATOR::INNER_TYPE*>(parent);
            inner->elems[0] = root;
            root = parent;
          }
        }
        else
          root = new_tree_node(level_needed, 0, elems_addressable - 1);
        n = root;
      }
      // root should be high-enough now
#ifdef DEBUG_LEGION
      assert((level_needed <= n->level) &&
             (index >= n->first_index) &&
             (index <= n->last_index));
#endif
      // now walk the path, instantiating the path we need
      while (n->level > 0)
      {
        typename ALLOCATOR::INNER_TYPE *inner = 
          static_cast<typename ALLOCATOR::INNER_TYPE*>(n);

        IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) *
                ALLOCATOR::INNER_BITS)) & 
                ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
#ifdef DEBUG_LEGION
        assert((i >= 0) && (((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE));
#endif
        NodeBase *child = inner->elems[i];
        if (child == 0)
        {
          AutoLock l(inner->lock);
          // Now that the lock is held, check to see if we lost the race
          if (inner->elems[i] == 0)
          {
            int child_level = inner->level - 1;
            int child_shift = 
              (ALLOCATOR::LEAF_BITS + child_level * ALLOCATOR::INNER_BITS);
            IT child_first = inner->first_index + (i << child_shift);
            IT child_last = inner->first_index + ((i + 1) << child_shift) - 1;

            inner->elems[i] = new_tree_node(child_level, 
                                            child_first, child_last);
          }
          child = inner->elems[i];
        }
#ifdef DEBUG_LEGION
        assert((child != 0) &&
               (child->level == (n->level - 1)) &&
               (index >= child->first_index) &&
               (index <= child->last_index));
#endif
        n = child;
      }
#ifdef DEBUG_LEGION
      assert(n->level == 0);
#endif
      return n;
    }

  }; // namespace Internal
}; // namespace Legion 

#endif // __LEGION_UTILITIES_H__
