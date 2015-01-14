/* Copyright 2014 Stanford University
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

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "legion_types.h"
#include "legion.h"
#include "legion_profiling.h"
#include "garbage_collection.h"

// Apple can go screw itself
#ifndef __MACH__
#include <x86intrin.h>
#else
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#endif
#ifdef DYNAMIC_FIELD_MASKS
#include <malloc.h>
#endif

namespace LegionRuntime {
  namespace HighLevel {

// Useful macros
#define IS_NO_ACCESS(req) (((req).privilege & READ_WRITE) == NO_ACCESS)
#define IS_READ_ONLY(req) (((req).privilege & READ_WRITE) <= READ_ONLY)
#define HAS_READ(req) ((req).privilege & READ_ONLY)
#define HAS_WRITE(req) ((req).privilege & (WRITE_DISCARD | REDUCE))
#define IS_WRITE(req) ((req).privilege & WRITE_DISCARD)
#define IS_WRITE_ONLY(req) (((req).privilege & READ_WRITE) == WRITE_DISCARD)
#define IS_REDUCE(req) (((req).privilege & READ_WRITE) == REDUCE)
#define IS_PROMOTED(req) ((req).privilege & PROMOTED)
#define IS_EXCLUSIVE(req) ((req).prop == EXCLUSIVE)
#define IS_ATOMIC(req) ((req).prop == ATOMIC)
#define IS_SIMULT(req) ((req).prop == SIMULTANEOUS)
#define IS_RELAXED(req) ((req).prop == RELAXED)

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
#ifdef DEBUG_HIGH_LEVEL
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
    static inline DependenceType check_for_promotion(const RegionUsage &u1, 
                                                     DependenceType actual)
    //--------------------------------------------------------------------------
    {
      if (IS_PROMOTED(u1))
        return PROMOTED_DEPENDENCE;
      else
        return actual;
    }

    //--------------------------------------------------------------------------
    static inline DependenceType check_dependence_type(const RegionUsage &u1,
                                                       const RegionUsage &u2)
    //--------------------------------------------------------------------------
    {
      // Two readers are never a dependence
      if (IS_READ_ONLY(u1) && IS_READ_ONLY(u2))
      {
        return check_for_promotion(u1, NO_DEPENDENCE);
      }
      else if (IS_REDUCE(u1) && IS_REDUCE(u2))
      {
        // If they are the same kind of reduction, no dependence, 
        // otherwise true dependence
        if (u1.redop == u2.redop)
          return check_for_promotion(u1, NO_DEPENDENCE);
        else
          return TRUE_DEPENDENCE;
      }
      else
      {
        // Everything in here has at least one right
#ifdef DEBUG_HIGH_LEVEL
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
          else if ((!IS_ATOMIC(u1) && IS_READ_ONLY(u1)) ||
                   (!IS_ATOMIC(u2) && IS_READ_ONLY(u2)))
          {
            return check_for_promotion(u1, NO_DEPENDENCE);
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
    // AutoLock 
    /////////////////////////////////////////////////////////////
    // An auto locking class for taking a lock and releasing it when
    // the object goes out of scope
    class AutoLock { 
    public:
      AutoLock(Reservation r, unsigned mode = 0, bool exclusive = true, 
               Event wait_on = Event::NO_EVENT)
        : is_low(true), low_lock(r)
      {
        Event lock_event = r.acquire(mode,exclusive,wait_on);
        if (lock_event.exists())
        {
#ifdef LEGION_PROF
          LegionProf::register_event(0, PROF_BEGIN_WAIT);
#endif
          lock_event.wait(true/*block*/);
#ifdef LEGION_PROF
          LegionProf::register_event(0, PROF_END_WAIT);
#endif
        }
      }
      AutoLock(ImmovableLock l)
        : is_low(false), immov_lock(l)
      {
        l.lock();
      }
    public:
      AutoLock(const AutoLock &rhs)
        : is_low(false)
      {
        // should never be called
        assert(false);
      }
      ~AutoLock(void)
      {
        if (is_low)
        {
          low_lock.release();
        }
        else
        {
          immov_lock.unlock();
        }
      }
    public:
      AutoLock& operator=(const AutoLock &rhs)
      {
        // should never be called
        assert(false);
        return *this;
      }
    private:
      const bool is_low;
      Reservation low_lock;
      ImmovableLock immov_lock;
    };

    /////////////////////////////////////////////////////////////
    // Serializer 
    /////////////////////////////////////////////////////////////
    class Serializer {
    public:
      Serializer(size_t base_bytes = 4096)
        : total_bytes(base_bytes), buffer((char*)malloc(base_bytes)), 
          index(0) 
#ifdef DEBUG_HIGH_LEVEL
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
      inline void serialize(const void *src, size_t bytes);
    public:
      inline void begin_context(void);
      inline void end_context(void);
    public:
      inline off_t get_index(void) const { return index; }
      inline const void* get_buffer(void) const { return buffer; }
      inline size_t get_buffer_size(void) const { return total_bytes; }
      inline size_t get_used_bytes(void) const { return index; }
    private:
      inline void resize(void);
    private:
      size_t total_bytes;
      char *buffer;
      off_t index;
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        // should have used the whole buffer
        assert(index == off_t(total_bytes)); 
#endif
      }
    public:
      inline Deserializer& operator=(const Deserializer &rhs);
    public:
      template<typename T>
      inline void deserialize(T &element);
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
      off_t index;
#ifdef DEBUG_HIGH_LEVEL
      size_t context_bytes;
#endif
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

    /////////////////////////////////////////////////////////////
    // Bit Mask 
    /////////////////////////////////////////////////////////////
    template<typename T, unsigned int MAX,
             unsigned int SHIFT, unsigned int MASK>
    class BitMask {
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
      static inline int pop_count(
            const BitMask<unsigned,MAX,SHIFT,MASK> &mask);
      static inline int pop_count(
            const BitMask<unsigned long,MAX,SHIFT,MASK> &mask);
      static inline int pop_count(
            const BitMask<unsigned long long,MAX,SHIFT,MASK> &mask);
    protected:
      T bit_vector[MAX/(8*sizeof(T))];
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
    class TLBitMask {
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
    };

#ifdef __SSE2__
    /////////////////////////////////////////////////////////////
    // SSE Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class SSEBitMask {
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
    };

    /////////////////////////////////////////////////////////////
    // SSE Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class SSETLBitMask {
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
    };
#endif // __SSE2__

#ifdef __AVX__
    /////////////////////////////////////////////////////////////
    // AVX Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class AVXBitMask {
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
#ifdef DYNAMIC_FIELD_MASKS
        __m256i *avx_vector;
        __m256d *avx_double;
        uint64_t *bit_vector;
#else
        __m256i avx_vector[MAX/256];
        __m256d avx_double[MAX/256];
        uint64_t bit_vector[MAX/64];
#endif
      } bits;
    };
    
    /////////////////////////////////////////////////////////////
    // AVX Two-Level Bit Mask  
    /////////////////////////////////////////////////////////////
    template<unsigned int MAX>
    class AVXTLBitMask {
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
#ifdef DYNAMIC_FIELD_MASKS
        __m256i *avx_vector;
        __m256d *avx_double;
        uint64_t *bit_vector;
#else
        __m256i avx_vector[MAX/256];
        __m256d avx_double[MAX/256];
        uint64_t bit_vector[MAX/64];
#endif
      } bits;
      uint64_t sum_mask;
    };
#endif // __AVX__

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
    class BitPermutation {
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
      inline unsigned get_dst(unsigned src);;
    protected:
      void compress_representation(void);
      void test_identity(void);
    protected:
      bool dirty;
      bool identity;
      BITMASK p[LOG2MAX];
      BITMASK comp[LOG2MAX];
    };

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
#ifdef DEBUG_HIGH_LEVEL
      context_bytes += sizeof(T);
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Serializer::serialize<bool>(const bool &element)
    //--------------------------------------------------------------------------
    {
      while ((size_t)(index + 4) > total_bytes)
        resize();
      *((bool*)buffer+index) = element;
      index += 4;
#ifdef DEBUG_HIGH_LEVEL
      context_bytes += 4;
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Serializer::serialize<FieldMask>(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const void *src, size_t bytes)
    //--------------------------------------------------------------------------
    {
      while ((index + bytes) > total_bytes)
        resize();
      memcpy(buffer+index,src,bytes);
      index += bytes;
#ifdef DEBUG_HIGH_LEVEL
      context_bytes += bytes;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::begin_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      // Save the size into the buffer
      while ((index + sizeof(size_t)) > total_bytes)
        resize();
      *((size_t*)(buffer+index)) = context_bytes;
      index += sizeof(size_t);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::resize(void)
    //--------------------------------------------------------------------------
    {
      // Double the buffer size
      total_bytes *= 2;
#ifdef DEBUG_HIGH_LEVEL
      assert(total_bytes != 0); // this would cause deallocation
#endif
      char *next = (char*)realloc(buffer,total_bytes);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      // Check to make sure we don't read past the end
      assert((index+sizeof(T)) <= total_bytes);
#endif
      element = *((const T*)(buffer+index));
      index += sizeof(T);
#ifdef DEBUG_HIGH_LEVEL
      context_bytes += sizeof(T);
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Deserializer::deserialize<bool>(bool &element)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Check to make sure we don't read past the end
      assert((index+4) <= total_bytes);
#endif
      element = *((const bool *)(buffer+index));
      index += 4;
#ifdef DEBUG_HIGH_LEVEL
      context_bytes += 4;
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Deserializer::deserialize<FieldMask>(FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
      
    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(void *dst, size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((index + bytes) <= total_bytes);
#endif
      memcpy(dst,buffer+index,bytes);
      index += bytes;
#ifdef DEBUG_HIGH_LEVEL
      context_bytes += bytes;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::begin_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Save our enclosing context on the stack
      size_t sent_context = *((const size_t*)(buffer+index));
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
#ifdef DEBUG_HIGH_LEVEL
      // Read the send context size out of the buffer      
      size_t sent_context = *((const size_t*)(buffer+index));
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
#ifdef DEBUG_HIGH_LEVEL
      assert(index <= off_t(total_bytes));
#endif
      return size_t(total_bytes - index);
    }

    //--------------------------------------------------------------------------
    inline const void* Deserializer::get_current_pointer(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index <= off_t(total_bytes));
#endif
      return (const void*)(buffer+index);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::advance_pointer(size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((index+bytes) <= total_bytes);
      context_bytes += bytes;
#endif
      index += bytes;
    }

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
#ifdef DEBUG_HIGH_LEVEL
      assert(denom > 0);
#endif
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>::Fraction(const Fraction<T> &f)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(factor != 0);
      assert(denominator > 0);
#endif
      T new_denom = denominator * factor;
#ifdef DEBUG_HIGH_LEVEL
      assert(new_denom > 0); // check for integer overflow
#endif
      denominator = new_denom;
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::add(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
          assert(denominator > 0); // check for integer overflow
#endif
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      // Should always be less than or equal to 1
      assert(numerator <= denominator); 
#endif
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::subtract(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(denominator > 0);
#endif
      if (denominator == rhs.denominator)
      {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
          assert(numerator >= (rhs.numerator*factor));
#endif
          numerator -= (rhs.numerator*factor);
        }
        else if ((rhs.denominator % denominator) == 0)
        {
          // Rhs denominator is bigger
          T factor = rhs.denominator/denominator;
#ifdef DEBUG_HIGH_LEVEL
          assert((numerator*factor) >= rhs.numerator);
#endif
          numerator = (numerator*factor) - rhs.numerator;
          denominator *= factor;
#ifdef DEBUG_HIGH_LEVEL
          assert(denominator > 0); // check for integer overflow
#endif
        }
        else
        {
          // One denominator is not divisible by the other, 
          // compute a common denominator
          T lhs_num = numerator * rhs.denominator;
          T rhs_num = rhs.numerator * denominator;
#ifdef DEBUG_HIGH_LEVEL
          assert(lhs_num >= rhs_num);
#endif
          numerator = lhs_num - rhs_num;
          denominator *= rhs.denominator; 
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert(denominator > 0); // check for integer overflow
#endif
      }
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T> Fraction<T>::get_part(T ways)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert(new_denom > 0); // check for integer overflow
#endif
        denominator = new_denom;
      }
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
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
      char *result = (char*)malloc((MAX+1)*sizeof(char));
      for (int idx = (BIT_ELMTS-1); idx >= 0; idx--)
      {
        if (idx == (BIT_ELMTS-1))
          sprintf(result,"%16.16lx",bit_vector[idx]);
        else
        {
          char temp[8*sizeof(T)+1];
          sprintf(temp,"%16.16lx",bit_vector[idx]);
          strcat(result,temp);
        }
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcount(mask[idx]);
      }
#else
      for (int idx = 0; idx < MAX; idx++)
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
      for (int idx = 0; idx < MAX; idx++)
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
#else
      for (int idx = 0; idx < MAX; idx++)
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
      char *result = (char*)malloc((MAX+1)*sizeof(char));
      for (int idx = (BIT_ELMTS-1); idx >= 0; idx--)
      {
        if (idx == (BIT_ELMTS-1))
          sprintf(result,"%16.16lx",bit_vector[idx]);
        else
        {
          char temp[8*sizeof(T)+1];
          sprintf(temp,"%16.16lx",bit_vector[idx]);
          strcat(result,temp);
        }
      }
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
      for (int idx = 0; idx < MAX; idx++)
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
      for (int idx = 0; idx < MAX; idx++)
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
      for (int idx = 0; idx < MAX; idx++)
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
          for (unsigned j = 0; j < 64; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*64 + j);
            }
          }
        }
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
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
      char *result = (char*)malloc((MAX+1)*sizeof(char));
      for (int idx = (BIT_ELMTS-1); idx >= 0; idx--)
      {
        if (idx == (BIT_ELMTS-1))
          sprintf(result,"%16.16lx",bits.bit_vector[idx]);
        else
        {
          char temp[65];
          sprintf(temp,"%16.16lx",bits.bit_vector[idx]);
          strcat(result,temp);
        }
      }
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
#else
      for (int idx = 0; idx < MAX; idx++)
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
          for (unsigned j = 0; j < 64; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*64 + j);
            }
          }
        }
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
      char *result = (char*)malloc((MAX+1)*sizeof(char));
      for (int idx = (BIT_ELMTS-1); idx >= 0; idx--)
      {
        if (idx == (BIT_ELMTS-1))
          sprintf(result,"%16.16lx",bits.bit_vector[idx]);
        else
        {
          char temp[65];
          sprintf(temp,"%16.16lx",bits.bit_vector[idx]);
          strcat(result,temp);
        }
      }
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
#else
      for (int idx = 0; idx < MAX; idx++)
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
      uint64_t left, right;
      right = _mm_cvtsi128_si32(value);
      left = _mm_cvtsi128_si32(_mm_shuffle_epi32(value, 1));
      uint64_t result = (left << 32) | right;
      right = _mm_cvtsi128_si32(_mm_shuffle_epi32(value, 2));
      left = _mm_cvtsi128_si32(_mm_shuffle_epi32(value, 3));
      result |= (left << 32) | right;
      return result;
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
#ifdef DYNAMIC_FIELD_MASKS
      bits.bit_vector = (uint64_t*)memalign(32, (MAX/8));
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
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
#ifdef DYNAMIC_FIELD_MASKS
      bits.bit_vector = (uint64_t*)memalign(32, (MAX/8));
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
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
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
      free(bits.bit_vector);
      bits.bit_vector = NULL;
#endif
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
          for (unsigned j = 0; j < 64; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*64 + j);
            }
          }
        }
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
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_vector != NULL);
#endif
#endif
      return bits.avx_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256i& AVXBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_vector != NULL);
#endif
#endif
      return bits.avx_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& AVXBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& AVXBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __m256d& AVXBitMask<MAX>::elem(const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_double != NULL);
#endif
#endif
      return bits.avx_double[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256d& AVXBitMask<MAX>::elem(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_double != NULL);
#endif
#endif
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
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
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL); 
#endif
#endif
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
      char *result = (char*)malloc((MAX+1)*sizeof(char));
      for (int idx = (BIT_ELMTS-1); idx >= 0; idx--)
      {
        if (idx == (BIT_ELMTS-1))
          sprintf(result,"%16.16lx",bits.bit_vector[idx]);
        else
        {
          char temp[65];
          sprintf(temp,"%16.16lx",bits.bit_vector[idx]);
          strcat(result,temp);
        }
      }
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
#else
      for (int idx = 0; idx < MAX; idx++)
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
#ifdef DYNAMIC_FIELD_MASKS
      bits.bit_vector = (uint64_t*)memalign(32, (MAX/8));
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
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
#ifdef DYNAMIC_FIELD_MASKS
      bits.bit_vector = (uint64_t*)memalign(32, (MAX/8));
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
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
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
      free(bits.bit_vector);
      bits.bit_vector = NULL;
#endif
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void AVXTLBitMask<MAX>::set_bit(unsigned bit)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
          for (unsigned j = 0; j < 64; j++)
          {
            if (bits.bit_vector[idx] & (1UL << j))
            {
              return (idx*64 + j);
            }
          }
        }
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
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_vector != NULL);
#endif
#endif
      return bits.avx_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256i& AVXTLBitMask<MAX>::operator()(const unsigned int &idx)
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_vector != NULL);
#endif
#endif
      return bits.avx_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const uint64_t& AVXTLBitMask<MAX>::operator[](
                                                 const unsigned int &idx) const
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
      return bits.bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline uint64_t& AVXTLBitMask<MAX>::operator[](const unsigned int &idx) 
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
      return bits.bit_vector[idx]; 
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline const __m256d& AVXTLBitMask<MAX>::elem(const unsigned &idx) const
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_double != NULL);
#endif
#endif
      return bits.avx_double[idx];
    }

    //-------------------------------------------------------------------------
    template<unsigned int MAX>
    inline __m256d& AVXTLBitMask<MAX>::elem(const unsigned &idx)
    //-------------------------------------------------------------------------
    {
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.avx_double != NULL);
#endif
#endif
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
#if defined(__AVX2__) and not defined(__MACH__)
        __m256i temp_sum = _mm_set1_epi32(0);
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
#if defined(__AVX2__) and not defined(__MACH__)
      __m256i temp_sum = _mm_set1_epi32(0);
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
      __m128i temp_sum = _mm_set1_epi32(0);
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
      __m128i temp_sum = _mm_set1_epi32(0);
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
        for (unsigned idx = (BIT_ELMTS-range); idx < (BIT_ELMTS-1); idx++)
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
#ifdef DYNAMIC_FIELD_MASKS
#ifdef DEBUG_HIGH_LEVEL
      assert(bits.bit_vector != NULL);
#endif
#endif
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
      char *result = (char*)malloc((MAX+1)*sizeof(char));
      for (int idx = (BIT_ELMTS-1); idx >= 0; idx--)
      {
        if (idx == (BIT_ELMTS-1))
          sprintf(result,"%16.16lx",bits.bit_vector[idx]);
        else
        {
          char temp[65];
          sprintf(temp,"%16.16lx",bits.bit_vector[idx]);
          strcat(result,temp);
        }
      }
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
#else
      for (int idx = 0; idx < MAX; idx++)
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
      uint64_t result = _mm_extract_epi64(right, 0);
      result |= _mm_extract_epi64(right, 1);
      result |= _mm_extract_epi64(left, 0);
      result |= _mm_extract_epi64(left, 1);
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
      BITMASK one;
      one.set_bit(0);

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
      BITMASK one;
      one.set_bit(0);

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

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __LEGION_UTILITIES_H__
