/* Copyright 2013 Stanford University
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

#include "legion_types.h"
#include "legion.h"
#include "legion_profiling.h"

namespace LegionRuntime {
  namespace HighLevel {

// Useful macros
#define IS_NO_ACCESS(req) ((req).privilege == NO_ACCESS)
#define IS_READ_ONLY(req) (((req).privilege == NO_ACCESS) \
    || ((req).privilege == READ_ONLY))
#define HAS_READ(req) (((req).privilege == READ_ONLY) \
    || ((req).privilege == READ_WRITE))
#define HAS_WRITE(req) (((req).privilege == READ_WRITE) \
    || ((req).privilege == REDUCE) || ((req).privilege == WRITE_DISCARD))
#define IS_WRITE(req) (((req).privilege == READ_WRITE) \
    || ((req).privilege == WRITE_DISCARD))
#define IS_WRITE_ONLY(req) ((req).privilege == WRITE_DISCARD)
#define IS_REDUCE(req) ((req).privilege == REDUCE)
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
    // consequences to the how a program executes.

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
        {
          return NO_DEPENDENCE;
        }
        else
        {
          return TRUE_DEPENDENCE;
        }
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
            return NO_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(u1,u2,TRUE_DEPENDENCE/*default*/);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(u1) || IS_SIMULT(u2))
        {
          return check_for_anti_dependence(u1,u2,
                                           SIMULTANEOUS_DEPENDENCE/*default*/);
        }
        else if (IS_RELAXED(u1) && IS_RELAXED(u2))
        {
          // TODO: Make this truly relaxed, right now it is the 
          // same as simultaneous
          return check_for_anti_dependence(u1,u2,
                                           SIMULTANEOUS_DEPENDENCE/*default*/);
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
      BitMask(T init = 0);
      BitMask(const BitMask &rhs);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
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
    // Two-Level Bit Permutation 
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
      TLBitMask(T init = 0);
      TLBitMask(const TLBitMask &rhs);
    public:
      inline void set_bit(unsigned bit);
      inline void unset_bit(unsigned bit);
      inline void assign_bit(unsigned bit, bool val);
      inline bool is_set(unsigned bit) const;
      inline int find_first_set(void) const;
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
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
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
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcount(mask[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline int BitMask<T,MAX,SHIFT,MASK>::pop_count(
                            const BitMask<unsigned long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    /*static*/ inline int BitMask<T,MAX,SHIFT,MASK>::pop_count(
                        const BitMask<unsigned long long,MAX,SHIFT,MASK> &mask)
    //-------------------------------------------------------------------------
    {
      int result = 0;
      for (int idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    TLBitMask<T,MAX,SHIFT,MASK>::TLBitMask(T init /*= 0*/)
      : sum_mask(init)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        bit_vector[idx] = init;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    TLBitMask<T,MAX,SHIFT,MASK>::TLBitMask(const TLBitMask &rhs)
    //-------------------------------------------------------------------------
    {
      sum_mask = rhs.sum_mask;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
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
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
        sum_mask |= (bit_vector[idx] & set_mask);
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
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcount(mask[idx]);
      }
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
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountl(mask[idx]);
      }
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
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result += __builtin_popcountll(mask[idx]);
      }
      return result;
    }
#undef BIT_ELMTS

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
