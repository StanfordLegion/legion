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


#ifndef __LEGION_UTILITIES_H__
#define __LEGION_UTILITIES_H__

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "legion.h"
#include "legion/bitmask.h"
#include "legion/legion_allocation.h"

// Useful macros
#define IS_NO_ACCESS(req) \
  (((req).privilege & LEGION_READ_WRITE) == LEGION_NO_ACCESS)
#define IS_READ_ONLY(req) \
  (((req).privilege & LEGION_READ_WRITE) <= LEGION_READ_PRIV)
#define HAS_READ(req) \
  ((req).privilege & LEGION_READ_PRIV)
#define HAS_WRITE(req) \
  ((req).privilege & (LEGION_WRITE_PRIV | LEGION_REDUCE))
#define IS_WRITE(req) \
  ((req).privilege & LEGION_WRITE_PRIV)
#define HAS_WRITE_DISCARD(req) \
  (((req).privilege & LEGION_WRITE_ONLY) == LEGION_WRITE_ONLY)
#define IS_DISCARD(req) \
  (((req).privilege & LEGION_DISCARD_MASK) == LEGION_DISCARD_MASK)
#define PRIV_ONLY(req) \
  ((req).privilege & LEGION_READ_WRITE)
#define IS_REDUCE(req) \
  (((req).privilege & LEGION_READ_WRITE) == LEGION_REDUCE)
#define IS_EXCLUSIVE(req) \
  ((req).prop == LEGION_EXCLUSIVE)
#define IS_ATOMIC(req) \
  ((req).prop == LEGION_ATOMIC)
#define IS_SIMULT(req) \
  ((req).prop == LEGION_SIMULTANEOUS)
#define IS_RELAXED(req) \
  ((req).prop == LEGION_RELAXED)

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
      inline void serialize(const BitMask<T,MAX,SHIFT,MASK> &mask);
      template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      inline void serialize(const TLBitMask<T,MAX,SHIFT,MASK> &mask);
#ifdef __SSE2__
      template<unsigned int MAX>
      inline void serialize(const SSEBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void serialize(const SSETLBitMask<MAX> &mask);
#endif
#ifdef __AVX__
      template<unsigned int MAX>
      inline void serialize(const AVXBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void serialize(const AVXTLBitMask<MAX> &mask);
#endif
#ifdef __ALTIVEC__
      template<unsigned int MAX>
      inline void serialize(const PPCBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void serialize(const PPCTLBitMask<MAX> &mask);
#endif
#ifdef __ARM_NEON
      template<unsigned int MAX>
      inline void serialize(const NeonBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void serialize(const NeonTLBitMask<MAX> &mask);
#endif
      template<typename DT, unsigned BLOAT, bool BIDIR>
      inline void serialize(const CompoundBitMask<DT,BLOAT,BIDIR> &mask);
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
      inline void reset(void);
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
      inline void deserialize(BitMask<T,MAX,SHIFT,MASK> &mask);
      template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
      inline void deserialize(TLBitMask<T,MAX,SHIFT,MASK> &mask);
#ifdef __SSE2__
      template<unsigned int MAX>
      inline void deserialize(SSEBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void deserialize(SSETLBitMask<MAX> &mask);
#endif
#ifdef __AVX__
      template<unsigned int MAX>
      inline void deserialize(AVXBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void deserialize(AVXTLBitMask<MAX> &mask);
#endif
#ifdef __ALTIVEC__
      template<unsigned int MAX>
      inline void deserialize(PPCBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void deserialize(PPCTLBitMask<MAX> &mask);
#endif
#ifdef __ARM_NEON
      template<unsigned int MAX>
      inline void deserialize(NeonBitMask<MAX> &mask);
      template<unsigned int MAX>
      inline void deserialize(NeonTLBitMask<MAX> &mask);
#endif
      template<typename DT, unsigned BLOAT, bool BIDIR>
      inline void deserialize(CompoundBitMask<DT,BLOAT,BIDIR> &mask);
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
        : privilege(LEGION_NO_ACCESS), prop(LEGION_EXCLUSIVE), redop(0) { }
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
        return LEGION_ANTI_DEPENDENCE;
      }
      else
      {
        if (HAS_WRITE_DISCARD(u2))
        {
          // WAW with a write-only
          return LEGION_ANTI_DEPENDENCE;
        }
        else
        {
          // This defaults to whatever the actual dependence is
          return actual;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<bool REDUCTIONS_INTERFERE>
    static inline DependenceType check_dependence_type(const RegionUsage &u1,
                                                       const RegionUsage &u2)
    //--------------------------------------------------------------------------
    {
      // Two readers are never a dependence
      if (IS_READ_ONLY(u1) && IS_READ_ONLY(u2))
      {
        return LEGION_NO_DEPENDENCE;
      }
      else if (!REDUCTIONS_INTERFERE && IS_REDUCE(u1) && IS_REDUCE(u2))
      {
        // If they are the same kind of reduction, no dependence, 
        // otherwise true dependence
        if (u1.redop == u2.redop)
          return LEGION_NO_DEPENDENCE;
        else
          return LEGION_TRUE_DEPENDENCE;
      }
      else
      {
        // Everything in here has at least one write
#ifdef DEBUG_LEGION
        assert(HAS_WRITE(u1) || HAS_WRITE(u2));
#endif
        // If anything exclusive 
        if (IS_EXCLUSIVE(u1) || IS_EXCLUSIVE(u2))
        {
          return check_for_anti_dependence(u1,u2,LEGION_TRUE_DEPENDENCE);
        }
        // Anything atomic (at least one is a write)
        else if (IS_ATOMIC(u1) || IS_ATOMIC(u2))
        {
          // If they're both atomics, return an atomic dependence
          if (IS_ATOMIC(u1) && IS_ATOMIC(u2))
          {
            return check_for_anti_dependence(u1,u2,LEGION_ATOMIC_DEPENDENCE); 
          }
          // If the one that is not an atomic is a read, we're also ok
          // We still need a simultaneous dependence if we don't have an
          // actual dependence
          else if ((!IS_ATOMIC(u1) && IS_READ_ONLY(u1)) ||
                   (!IS_ATOMIC(u2) && IS_READ_ONLY(u2)))
          {
            return LEGION_SIMULTANEOUS_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(u1,u2,LEGION_TRUE_DEPENDENCE);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(u1) || IS_SIMULT(u2))
        {
          return LEGION_SIMULTANEOUS_DEPENDENCE;
        }
        else if (IS_RELAXED(u1) && IS_RELAXED(u2))
        {
          // TODO: Make this truly relaxed, right now it is the 
          // same as simultaneous
          return LEGION_SIMULTANEOUS_DEPENDENCE;
          // This is what it should be: return NO_DEPENDENCE;
          // What needs to be done:
          // - RegionNode::update_valid_instances needs to allow multiple 
          //               outstanding writers
          // - RegionNode needs to detect relaxed case and make copies from all 
          //              relaxed instances to non-relaxed instance
        }
        // We should never make it here
        assert(false);
        return LEGION_NO_DEPENDENCE;
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
          elems[i].store(NULL);
      }
      DynamicTableNode(const DynamicTableNode &rhs) { assert(false); }
      virtual ~DynamicTableNode(void)
      {
        for (size_t i = 0; i < SIZE; i++)
        {
          ET *elem = elems[i].load();
          if (elem != NULL)
            delete elem;
        }
      }
    public:
      DynamicTableNode& operator=(const DynamicTableNode &rhs)
        { assert(false); return *this; }
    public:
      std::atomic<ET*> elems[SIZE];
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
          elems[i].store(NULL);
      }
      LeafTableNode(const LeafTableNode &rhs) { assert(false); }
      virtual ~LeafTableNode(void)
      {
        for (size_t i = 0; i < SIZE; i++)
        {
          ET *elem = elems[i].load();
          if (elem != NULL)
            delete elem;
        }
      }
    public:
      LeafTableNode& operator=(const LeafTableNode &rhs)
        { assert(false); return *this; }
    public:
      std::atomic<ET*> elems[SIZE];
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
      std::atomic<NodeBase*> root;
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
      // Old versions of g++ don't have support for all of c++11
#if !defined(__GNUC__) || (__GNUC__ >= 5)
      static_assert(std::is_trivially_copyable<T>::value, "unserializable");
#endif
      while ((index + sizeof(T)) > total_bytes)
        resize();
      memcpy(buffer+index, (const void*)&element, sizeof(T));
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
      const uint32_t flag = element ? 1 : 0;
      serialize<uint32_t>(flag);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Serializer::serialize(const BitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Serializer::serialize(const TLBitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

#ifdef __SSE2__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const SSEBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const SSETLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

#ifdef __AVX__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const AVXBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const AVXTLBitMask<MAX> &mask)
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

#ifdef __ARM_NEON
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const NeonBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const NeonTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

    //--------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void Serializer::serialize(const CompoundBitMask<DT,BLOAT,BIDIR> &m)
    //--------------------------------------------------------------------------
    {
      m.serialize(*this);
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const Domain &dom)
    //--------------------------------------------------------------------------
    {
      serialize(dom.is_id);
      if (dom.is_id > 0)
        serialize(dom.is_type);
      serialize(dom.dim);
      for (int i = 0; i < 2*dom.dim; i++)
        serialize(dom.rect_data[i]);
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
    inline void Serializer::reset(void)
    //--------------------------------------------------------------------------
    {
      index = 0;
#ifdef DEBUG_LEGION
      context_bytes = 0;
#endif
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
      // Old versions of g++ don't have support for all of c++11
#if !defined(__GNUC__) || (__GNUC__ >= 5)
      static_assert(std::is_trivially_copyable<T>::value, "unserializable");
#endif
#ifdef DEBUG_LEGION
      // Check to make sure we don't read past the end
      assert((index+sizeof(T)) <= total_bytes);
#endif
      memcpy(&element, buffer+index, sizeof(T));
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
      uint32_t flag;
      deserialize<uint32_t>(flag);
      element = (flag != 0);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Deserializer::deserialize(BitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Deserializer::deserialize(TLBitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

#ifdef __SSE2__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(SSEBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(SSETLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

#ifdef __AVX__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(AVXBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(AVXTLBitMask<MAX> &mask)
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

#ifdef __ARM_NEON
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(NeonBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(NeonTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

    //--------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void Deserializer::deserialize(CompoundBitMask<DT,BLOAT,BIDIR> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(Domain &dom)
    //--------------------------------------------------------------------------
    {
      deserialize(dom.is_id);
      if (dom.is_id > 0)
        deserialize(dom.is_type);
      deserialize(dom.dim);
      for (int i = 0; i < 2*dom.dim; i++)
        deserialize(dom.rect_data[i]);
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
    template<typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::DynamicTable(void)
    //-------------------------------------------------------------------------
    {
      root.store(NULL);
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
      NodeBase *r = root.load();
      if (r != NULL)
      {
        delete r;
        root.store(NULL);
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
      if (root.load() == NULL)
        return 0;
      size_t elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      for (int i = 0; i < root.load()->level; i++)
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

      NodeBase *n = root.load();
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
        NodeBase *child = inner->elems[i].load();
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
      ET *result = leaf->elems[offset].load();
      if (result == NULL)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        if (leaf->elems[offset].load() == NULL)
        {
          ET *elem = new ET();
          leaf->elems[offset].store(elem);
        }
        result = leaf->elems[offset].load();
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
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
      ET *result = leaf->elems[offset].load();
      if (result == NULL)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        if (leaf->elems[offset].load() == NULL)
        {
          ET *elem = new ET(arg);
          leaf->elems[offset].store(elem);
        }
        result = leaf->elems[offset].load();
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
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
      ET *result = leaf->elems[offset].load();
      if (result == NULL)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        if (leaf->elems[offset].load() == NULL)
        {
          ET *elem = new ET(arg1, arg2);
          leaf->elems[offset].store(elem);
        }
        result = leaf->elems[offset].load();
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
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
      NodeBase *n = root.load();
      if (!n || (n->level < level_needed)) 
      {
        AutoLock l(lock); 
        n = root.load();
        if (n != NULL)
        {
          // some of the tree exists - add new layers on top
          while (n->level < level_needed)
          {
            int parent_level = n->level + 1;
            IT parent_first = 0;
            IT parent_last = 
              (((n->last_index + 1) << ALLOCATOR::INNER_BITS) - 1);
            NodeBase *parent = new_tree_node(parent_level, 
                                             parent_first, parent_last);
            typename ALLOCATOR::INNER_TYPE *inner = 
              static_cast<typename ALLOCATOR::INNER_TYPE*>(parent);
            inner->elems[0].store(n);
            n = parent;
          }
        }
        else
          n = new_tree_node(level_needed, 0, elems_addressable - 1);
        root.store(n);
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
        NodeBase *child = inner->elems[i].load();
        if (child == NULL)
        {
          AutoLock l(inner->lock);
          // Now that the lock is held, check to see if we lost the race
          if (inner->elems[i].load() == NULL)
          {
            int child_level = inner->level - 1;
            int child_shift = 
              (ALLOCATOR::LEAF_BITS + child_level * ALLOCATOR::INNER_BITS);
            IT child_first = inner->first_index + (i << child_shift);
            IT child_last = inner->first_index + ((i + 1) << child_shift) - 1;

            NodeBase *next = new_tree_node(child_level, 
                                           child_first, child_last);
            inner->elems[i].store(next);
          }
          child = inner->elems[i].load();
        }
#ifdef DEBUG_LEGION
        assert((child != NULL) &&
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

    /**
     * \struct FieldSet
     * A helper template class for the method below for describing
     * sets of members that all contain the same fields
     */
    template<typename T>
    struct FieldSet {
    public:
      FieldSet(void) { }
      FieldSet(const FieldMask &m)
        : set_mask(m) { }
    public:
      FieldMask set_mask;
      std::set<T> elements;
    };

    //--------------------------------------------------------------------------
    template<typename T>
    inline void compute_field_sets(FieldMask universe_mask,
                                   const LegionMap<T,FieldMask> &inputs,
                                   LegionList<FieldSet<T> > &output_sets)
    //--------------------------------------------------------------------------
    {
      // Special cases for empty and size 1 sets
      if (inputs.empty())
      {
        if (!!universe_mask)
          output_sets.push_back(FieldSet<T>(universe_mask));
        return;
      }
      else if (inputs.size() == 1)
      {
        typename LegionMap<T,FieldMask>::const_iterator first = 
          inputs.begin();
        output_sets.push_back(FieldSet<T>(first->second));
        FieldSet<T> &last = output_sets.back();
        last.elements.insert(first->first);
        if (!!universe_mask)
        {
          universe_mask -= first->second;
          if (!!universe_mask)
            output_sets.push_back(FieldSet<T>(universe_mask));
        }
        return;
      }
      for (typename LegionMap<T,FieldMask>::const_iterator pit = 
            inputs.begin(); pit != inputs.end(); pit++)
      {
        bool inserted = false;
        // Also keep track of which fields have updates
        // but don't have any members 
        if (!!universe_mask)
          universe_mask -= pit->second;
        FieldMask remaining = pit->second;
        // Insert this event into the precondition sets 
        for (typename LegionList<FieldSet<T> >::iterator it = 
              output_sets.begin(); it != output_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->set_mask)
          {
            it->elements.insert(pit->first);
            inserted = true;
            break;
          }
          FieldMask overlap = remaining & it->set_mask;
          // Easy case, they are disjoint so keep going
          if (!overlap)
            continue;
          // Moderate case, we are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == remaining)
          {
            // Leave the existing set and make it the difference 
            it->set_mask -= overlap;
            output_sets.push_back(FieldSet<T>(overlap));
            FieldSet<T> &last = output_sets.back();
            last.elements = it->elements;
            last.elements.insert(pit->first);
            inserted = true;
            break;
          }
          // Moderate case, we dominate the existing set
          if (overlap == it->set_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->elements.insert(pit->first);
            remaining -= overlap;
            // Can't consider ourselves added yet
            continue;
          }
          // Hard case, neither dominates, compute three
          // distinct sets of fields, keep left one in
          // place and reduce scope, add new one at the
          // end for overlap, continue iterating for right one
          it->set_mask -= overlap;
          const std::set<T> &temp_elements = it->elements;
          it = output_sets.insert(it, FieldSet<T>(overlap));
          it->elements = temp_elements;
          it->elements.insert(pit->first);
          remaining -= overlap;
          continue;
        }
        if (!inserted)
        {
          output_sets.push_back(FieldSet<T>(remaining));
          FieldSet<T> &last = output_sets.back();
          last.elements.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any elements, but them in their own set.
      // Put it on the front because it is the copy with
      // no elements so it can start right away!
      if (!!universe_mask)
        output_sets.push_front(FieldSet<T>(universe_mask));
    }

    /**
     * \class FieldMaskSet 
     * A template helper class for tracking collections of 
     * objects associated with different sets of fields
     */
    template<typename T, AllocationType A = UNTRACKED_ALLOC>
    class FieldMaskSet : 
      public LegionHeapify<FieldMaskSet<T> > {
    public:
      // forward declaration
      class const_iterator;
      class iterator {
      public:
        // explicitly set iterator traits
        typedef std::input_iterator_tag iterator_category;
        typedef std::pair<T*const,FieldMask> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<T*const,FieldMask> *pointer;
        typedef std::pair<T*const,FieldMask>& reference;

        iterator(FieldMaskSet *_set, 
            std::pair<T*const,FieldMask> *_result)
          : set(_set), result(_result), single(true) { }
        iterator(FieldMaskSet *_set,
            typename LegionMap<T*,FieldMask,A>::iterator _it,
            bool end = false)
          : set(_set), result(end ? NULL : &(*_it)), it(_it), single(false) { }
      public:
        iterator(const iterator &rhs)
          : set(rhs.set), result(rhs.result), 
            it(rhs.it), single(rhs.single) { }
        ~iterator(void) { }
      public:
        inline iterator& operator=(const iterator &rhs)
          { set = rhs.set; result = rhs.result; 
            it = rhs.it; single = rhs.single; return *this; }
      public:
        inline bool operator==(const iterator &rhs) const
          { 
            if (set != rhs.set) 
              return false;
            if (single)
              return (result == rhs.result);
            else
              return (it == rhs.it);
          }
        inline bool operator!=(const iterator &rhs) const
          {
            if (set != rhs.set)
              return true;
            if (single)
              return (result != rhs.result);
            else
              return (it != rhs.it);
          }
      public:
        inline const std::pair<T*const,FieldMask> operator*(void) 
          { return *result; }
        inline const std::pair<T*const,FieldMask>* operator->(void)
          { return result; }
        inline iterator& operator++(/*prefix*/void)
          {
            if (!single)
            {
              ++it;
              if ((*this) != set->end())
                result = &(*it);
              else
                result = NULL;
            }
            else
              result = NULL;
            return *this;
          }
        inline iterator operator++(/*postfix*/int)
          {
            iterator copy(*this);
            if (!single)
            {
              ++it;
              if ((*this) != set->end())
                result = &(*it);
              else
                result = NULL;
            }
            else
              result = NULL;
            return copy;
          }
      public:
        inline operator bool(void) const
          { return (result != NULL); }
      public:
        inline void merge(const FieldMask &mask)
          {
            result->second |= mask;
            if (!single)
              set->valid_fields |= mask;
          }
        inline void filter(const FieldMask &mask)
          {
            result->second -= mask;
            // Don't filter valid fields since its unsound
          }
        inline void clear(void)
          {
            result->second.clear();
          }
      public:
        inline void erase(LegionMap<T*,FieldMask,A> &target)
        {
#ifdef DEBUG_LEGION
          assert(!single);
#endif
          // Erase it from the target
          target.erase(it);
          // Invalidate the iterator
          it = target.end();
          result = NULL;
        }
      private:
        friend class const_iterator;
        FieldMaskSet *set;
        std::pair<T*const,FieldMask> *result;
        typename LegionMap<T*,FieldMask,A>::iterator it;
        bool single;
      };
    public:
      class const_iterator {
      public:
        // explicitly set iterator traits
        typedef std::input_iterator_tag iterator_category;
        typedef std::pair<T*const,FieldMask> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<T*const,FieldMask> *pointer;
        typedef std::pair<T*const,FieldMask>& reference;

        const_iterator(const FieldMaskSet *_set, 
            const std::pair<T*const,FieldMask> *_result)
          : set(_set), result(_result), single(true) { }
        const_iterator(const FieldMaskSet *_set,
            typename LegionMap<T*,FieldMask,A>::const_iterator _it,
            bool end = false)
          : set(_set), result(end ? NULL : &(*_it)), it(_it), single(false) { }
      public:
        const_iterator(const const_iterator &rhs)
          : set(rhs.set), result(rhs.result), it(rhs.it), single(rhs.single) { }
        // We can also make a const_iterator from a normal iterator
        const_iterator(const iterator &rhs)
          : set(rhs.set), result(rhs.result), it(rhs.it), single(rhs.single) { }
        ~const_iterator(void) { }
      public:
        inline const_iterator& operator=(const const_iterator &rhs)
          { set = rhs.set; result = rhs.result; it = rhs.it;
            single = rhs.single; return *this; }
        inline const_iterator& operator=(const iterator &rhs)
          { set = rhs.set; result = rhs.result; it = rhs.it;
            single = rhs.single; return *this; }
      public:
        inline bool operator==(const const_iterator &rhs) const
          { 
            if (set != rhs.set) 
              return false;
            if (single)
              return (result == rhs.result);
            else
              return (it == rhs.it);
          }
        inline bool operator!=(const const_iterator &rhs) const
          {
            if (set != rhs.set)
              return true;
            if (single)
              return (result != rhs.result);
            else
              return (it != rhs.it);
          }
      public:
        inline const std::pair<T*const,FieldMask> operator*(void) 
          { return *result; }
        inline const std::pair<T*const,FieldMask>* operator->(void)
          { return result; }
        inline const_iterator& operator++(/*prefix*/void)
          {
            if (!single)
            {
              ++it;
              if ((*this) != set->end())
                result = &(*it);
              else
                result = NULL;
            }
            else
              result = NULL;
            return *this;
          }
        inline const_iterator operator++(/*postfix*/int)
          {
            const_iterator copy(*this);
            if (!single)
            {
              ++it;
              if ((*this) != set->end())
                result = &(*it);
              else
                result = NULL;
            }
            else
              result = NULL;
            return copy;
          }
      public:
        inline operator bool(void) const
          { return (result != NULL); }
      private:
        const FieldMaskSet *set;
        const std::pair<T*const,FieldMask> *result;
        typename LegionMap<T*,FieldMask,A>::const_iterator it;
        bool single;
      };
    public:
      FieldMaskSet(void)
        : single(true) { entries.single_entry = NULL; }
      inline FieldMaskSet(const FieldMaskSet &rhs);
      ~FieldMaskSet(void) { clear(); }
    public:
      inline FieldMaskSet& operator=(const FieldMaskSet &rhs);
    public:
      inline bool empty(void) const 
        { return single && (entries.single_entry == NULL); }
      inline const FieldMask& get_valid_mask(void) const 
        { return valid_fields; }
      inline const FieldMask& tighten_valid_mask(void);
      inline void relax_valid_mask(const FieldMask &m);
      inline void filter_valid_mask(const FieldMask &m);
      inline void restrict_valid_mask(const FieldMask &m);
    public:
      inline const FieldMask& operator[](T *entry) const;
    public:
      // Return true if we actually added the entry, false if it already existed
      inline bool insert(T *entry, const FieldMask &mask); 
      inline void filter(const FieldMask &filter);
      inline void erase(T *to_erase);
      inline void clear(void);
      inline size_t size(void) const;
    public:
      inline void swap(FieldMaskSet &other);
    public:
      inline iterator begin(void);
      inline iterator find(T *entry);
      inline void erase(iterator &it);
      inline iterator end(void);
    public:
      inline const_iterator begin(void) const;
      inline const_iterator find(T *entry) const;
      inline const_iterator end(void) const;
    public:
      inline void compute_field_sets(FieldMask universe_mask,
                    LegionList<FieldSet<T*> > &output_sets) const;
    protected:
      // Fun with C, keep these two fields first and in this order
      // so that a FieldMaskSet of size 1 looks the same as an entry
      // in the STL Map in the multi-entries case, 
      // provides goodness for the iterator
      union {
        T *single_entry;
        LegionMap<T*,FieldMask,A> *multi_entries;
      } entries;
      // This can be an overapproximation if we have multiple entries
      FieldMask valid_fields;
      bool single;
    };

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline FieldMaskSet<T,A>::FieldMaskSet(const FieldMaskSet<T,A> &rhs)
      : valid_fields(rhs.valid_fields), single(rhs.single)
    //--------------------------------------------------------------------------
    {
      if (single)
        entries.single_entry = rhs.entries.single_entry;
      else
        entries.multi_entries = new LegionMap<T*,FieldMask,A>(
            rhs.entries.multi_entries->begin(),
            rhs.entries.multi_entries->end());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline FieldMaskSet<T,A>& FieldMaskSet<T,A>::operator=(
                                                   const FieldMaskSet<T,A> &rhs)
    //--------------------------------------------------------------------------
    {
      // Check our current state
      if (single != rhs.single)
      {
        // Different data structures
        if (single)
        {
          entries.multi_entries = new LegionMap<T*,FieldMask,A>(
              rhs.entries.multi_entries->begin(),
              rhs.entries.multi_entries->end());
        }
        else
        {
          // Free our map
          delete entries.multi_entries;
          entries.single_entry = rhs.entries.single_entry;
        }
        single = rhs.single;
      }
      else
      {
        // Same data structures so we can just copy things over
        if (single)
          entries.single_entry = rhs.entries.single_entry;
        else
        {
          entries.multi_entries->clear();
          entries.multi_entries->insert(
              rhs.entries.multi_entries->begin(),
              rhs.entries.multi_entries->end());
        }
      }
      valid_fields = rhs.valid_fields;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline const FieldMask& FieldMaskSet<T,A>::tighten_valid_mask(void)
    //--------------------------------------------------------------------------
    {
      // If we're single then there is nothing to do as we're already tight
      if (single)
        return valid_fields;
      valid_fields.clear();
      for (typename LegionMap<T*,FieldMask,A>::const_iterator it = 
            entries.multi_entries->begin(); it !=
            entries.multi_entries->end(); it++)
        valid_fields |= it->second;
      return valid_fields;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::relax_valid_mask(const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      valid_fields |= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::filter_valid_mask(const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      valid_fields -= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::restrict_valid_mask(const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      valid_fields &= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline const FieldMask& FieldMaskSet<T,A>::operator[](T *entry) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(entry == entries.single_entry);
#endif
        return valid_fields;
      }
      else
      {
        typename LegionMap<T*,FieldMask,A>::const_iterator finder =
          entries.multi_entries->find(entry);
#ifdef DEBUG_LEGION
        assert(finder != entries.multi_entries->end());
#endif
        return finder->second;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline bool FieldMaskSet<T,A>::insert(T *entry, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      if (single)
      {
        if (entries.single_entry == NULL)
        {
          entries.single_entry = entry;
          valid_fields = mask;
        }
        else if (entries.single_entry == entry)
        {
          valid_fields |= mask;
          result = false;
        }
        else
        {
          // Go to multi
          LegionMap<T*,FieldMask,A> *multi = new LegionMap<T*,FieldMask,A>();
          (*multi)[entries.single_entry] = valid_fields;
          (*multi)[entry] = mask;
          entries.multi_entries = multi;
          single = false;
          valid_fields |= mask;
        }
      }
      else
      {
 #ifdef DEBUG_LEGION
        assert(entries.multi_entries != NULL);
#endif   
        typename LegionMap<T*,FieldMask,A>::iterator finder = 
          entries.multi_entries->find(entry);
        if (finder == entries.multi_entries->end())
          (*entries.multi_entries)[entry] = mask;
        else
        {
          finder->second |= mask;
          result = false;
        }
        valid_fields |= mask;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::filter(const FieldMask &filter)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (entries.single_entry != NULL)
        {
          valid_fields -= filter;
          if (!valid_fields)
            entries.single_entry = NULL;
        }
      }
      else
      {
        valid_fields -= filter;
        if (!valid_fields)
        {
          // No fields left so just clean everything up
          delete entries.multi_entries;
          entries.multi_entries = NULL;
          single = true;
        }
        else
        {
          // Manually remove entries
          typename std::vector<T*> to_delete;
          for (typename LegionMap<T*,FieldMask,A>::iterator it = 
                entries.multi_entries->begin(); it !=
                entries.multi_entries->end(); it++)
          {
            it->second -= filter;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (typename std::vector<T*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              entries.multi_entries->erase(*it);
            if (entries.multi_entries->empty())
            {
              delete entries.multi_entries;
              entries.multi_entries = NULL;
              single = true;
            }
            else if (entries.multi_entries->size() == 1)
            {
              typename LegionMap<T*,FieldMask,A>::iterator last = 
                entries.multi_entries->begin();     
              T *temp = last->first; 
              valid_fields = last->second;
              delete entries.multi_entries;
              entries.single_entry = temp;
              single = true;
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::erase(T *to_erase)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(entries.single_entry == to_erase);
#endif
        entries.single_entry = NULL;
        valid_fields.clear();
      }
      else
      {
        typename LegionMap<T*,FieldMask,A>::iterator finder = 
          entries.multi_entries->find(to_erase);
#ifdef DEBUG_LEGION
        assert(finder != entries.multi_entries->end());
#endif
        entries.multi_entries->erase(finder);
        if (entries.multi_entries->size() == 1)
        {
          // go back to single
          finder = entries.multi_entries->begin();
          valid_fields = finder->second;
          T *first = finder->first;
          delete entries.multi_entries;
          entries.single_entry = first;
          single = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::clear(void)
    //--------------------------------------------------------------------------
    {
      if (single)
        entries.single_entry = NULL;
      else
      {
#ifdef DEBUG_LEGION
        assert(entries.multi_entries != NULL);
#endif
        delete entries.multi_entries;
        entries.multi_entries = NULL;
        single = true;
      }
      valid_fields.clear();
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline size_t FieldMaskSet<T,A>::size(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (entries.single_entry == NULL)
          return 0;
        else
          return 1;
      }
      else
        return entries.multi_entries->size();
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::swap(FieldMaskSet &other)
    //--------------------------------------------------------------------------
    {
      // Just use single, doesn't matter for swap
      T *temp_entry = other.entries.single_entry;
      other.entries.single_entry = entries.single_entry;
      entries.single_entry = temp_entry;

      bool temp_single = other.single;
      other.single = single;
      single = temp_single;

      FieldMask temp_valid_fields = other.valid_fields;
      other.valid_fields = valid_fields;
      valid_fields = temp_valid_fields;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline typename FieldMaskSet<T,A>::iterator FieldMaskSet<T,A>::begin(void)
    //--------------------------------------------------------------------------
    {
      // Scariness!
      if (single)
      {
        // If we're empty return end
        if (entries.single_entry == NULL)
          return end();
        FieldMaskSet<T> *ptr = this;
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(result) == sizeof(ptr), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return iterator(this, result); 
      }
      else
        return iterator(this, entries.multi_entries->begin());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline typename FieldMaskSet<T,A>::iterator FieldMaskSet<T,A>::find(T *e)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((entries.single_entry == NULL) || (entries.single_entry != e))
          return end();
        FieldMaskSet<T> *ptr = this;
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(result) == sizeof(ptr), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return iterator(this, result);
      }
      else
      {
        typename LegionMap<T*,FieldMask,A>::iterator finder = 
          entries.multi_entries->find(e);
        if (finder == entries.multi_entries->end())
          return end();
        return iterator(this, finder);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::erase(iterator &it)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(it != end());
#endif
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(entries.single_entry == it->first);
#endif
        entries.single_entry = NULL;
        valid_fields.clear();
      }
      else
      {
        it.erase(*(entries.multi_entries));
        if (entries.multi_entries->size() == 1)
        {
          // go back to single
          typename LegionMap<T*,FieldMask,A>::iterator finder = 
            entries.multi_entries->begin();
          valid_fields = finder->second;
          T *first = finder->first;
          delete entries.multi_entries;
          entries.single_entry = first;
          single = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline typename FieldMaskSet<T,A>::iterator FieldMaskSet<T,A>::end(void)
    //--------------------------------------------------------------------------
    {
      if (single)
        return iterator(this, NULL);
      else
        return iterator(this, entries.multi_entries->end(), true/*end*/);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline typename FieldMaskSet<T,A>::const_iterator 
                                            FieldMaskSet<T,A>::begin(void) const
    //--------------------------------------------------------------------------
    {
      // Scariness!
      if (single)
      {
        // If we're empty return end
        if (entries.single_entry == NULL)
          return end();
        FieldMaskSet<T> *ptr = const_cast<FieldMaskSet<T>*>(this);
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(ptr) == sizeof(result), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return const_iterator(this, result); 
      }
      else
        return const_iterator(this, entries.multi_entries->begin());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline typename FieldMaskSet<T,A>::const_iterator 
                                             FieldMaskSet<T,A>::find(T *e) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((entries.single_entry == NULL) || (entries.single_entry != e))
          return end();
        FieldMaskSet<T> *ptr = const_cast<FieldMaskSet<T>*>(this);
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(ptr) == sizeof(result), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return const_iterator(this, result);
      }
      else
      {
        typename LegionMap<T*,FieldMask,A>::const_iterator finder = 
          entries.multi_entries->find(e);
        if (finder == entries.multi_entries->end())
          return end();
        return const_iterator(this, finder);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline typename FieldMaskSet<T,A>::const_iterator 
                                              FieldMaskSet<T,A>::end(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
        return const_iterator(this, NULL);
      else
        return const_iterator(this, entries.multi_entries->end(), true/*end*/);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A>
    inline void FieldMaskSet<T,A>::compute_field_sets(FieldMask universe_mask,
                                   LegionList<FieldSet<T*> > &output_sets) const
    //--------------------------------------------------------------------------
    {
      // Handle special cases for single entry and single fields
      if (empty())
      {
        if (!!universe_mask)
          output_sets.push_back(FieldSet<T*>(universe_mask));
        return;
      }
      else if (single)
      {
        output_sets.push_back(FieldSet<T*>(valid_fields));
        FieldSet<T*> &last = output_sets.back();
        last.elements.insert(entries.single_entry);
        if (!!universe_mask)
        {
          universe_mask -= valid_fields;
          if (!!universe_mask)
            output_sets.push_back(FieldSet<T*>(universe_mask));
        }
        return;
      }
      else if (valid_fields.pop_count() == 1)
      {
        output_sets.push_back(FieldSet<T*>(valid_fields));
        FieldSet<T*> &last = output_sets.back();
        bool has_empty = false;
        for (const_iterator pit = this->begin(); pit != this->end(); pit++)
        {
          if (!!pit->second)
            last.elements.insert(pit->first);
          else
            has_empty = true;
        }
        if (has_empty)
        {
          output_sets.push_back(FieldSet<T*>(FieldMask()));
          last = output_sets.back();
          for (const_iterator pit = this->begin(); pit != this->end(); pit++)
            if (!pit->second)
              last.elements.insert(pit->first);
        }
        if (!!universe_mask)
        {
          universe_mask -= valid_fields;
          if (!!universe_mask)
            output_sets.push_back(FieldSet<T*>(universe_mask));
        }
        return;
      }
      // Otherwise we fall through and do the full thing
      for (const_iterator pit = this->begin(); pit != this->end(); pit++)
      {
        bool inserted = false;
        // Also keep track of which fields have updates
        // but don't have any members 
        if (!!universe_mask)
          universe_mask -= pit->second;
        FieldMask remaining = pit->second;
        // Insert this event into the precondition sets 
        for (typename LegionList<FieldSet<T*> >::iterator it = 
              output_sets.begin(); it != output_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->set_mask)
          {
            it->elements.insert(pit->first);
            inserted = true;
            break;
          }
          FieldMask overlap = remaining & it->set_mask;
          // Easy case, they are disjoint so keep going
          if (!overlap)
            continue;
          // Moderate case, we are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == remaining)
          {
            // Leave the existing set and make it the difference 
            it->set_mask -= overlap;
            output_sets.push_back(FieldSet<T*>(overlap));
            FieldSet<T*> &last = output_sets.back();
            last.elements = it->elements;
            last.elements.insert(pit->first);
            inserted = true;
            break;
          }
          // Moderate case, we dominate the existing set
          if (overlap == it->set_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->elements.insert(pit->first);
            remaining -= overlap;
            // Can't consider ourselves added yet
            continue;
          }
          // Hard case, neither dominates, compute three
          // distinct sets of fields, keep left one in
          // place and reduce scope, add new one at the
          // end for overlap, continue iterating for right one
          it->set_mask -= overlap;
          const std::set<T*> &temp_elements = it->elements;
          it = output_sets.insert(it, FieldSet<T*>(overlap));
          it->elements = temp_elements;
          it->elements.insert(pit->first);
          remaining -= overlap;
          continue;
        }
        if (!inserted)
        {
          output_sets.push_back(FieldSet<T*>(remaining));
          FieldSet<T*> &last = output_sets.back();
          last.elements.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any elements, but them in their own set.
      // Put it on the front because it is the copy with
      // no elements so it can start right away!
      if (!!universe_mask)
        output_sets.push_front(FieldSet<T*>(universe_mask));
    }

  }; // namespace Internal
}; // namespace Legion 

#endif // __LEGION_UTILITIES_H__
