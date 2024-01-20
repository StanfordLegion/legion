/* Copyright 2023 Stanford University, NVIDIA Corporation
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
  (((req).privilege & LEGION_READ_WRITE) == LEGION_READ_PRIV)
#define HAS_READ(req) \
  ((req).privilege & (LEGION_READ_PRIV | LEGION_REDUCE))
#define HAS_WRITE(req) \
  ((req).privilege & (LEGION_WRITE_PRIV | LEGION_REDUCE))
#define IS_WRITE(req) \
  ((req).privilege & LEGION_WRITE_PRIV)
#define IS_WRITE_DISCARD(req) \
  (((req).privilege & LEGION_WRITE_ONLY) == LEGION_WRITE_ONLY)
#define IS_READ_DISCARD(req) \
  (((req).privilege & (LEGION_READ_PRIV | LEGION_DISCARD_OUTPUT_MASK)) \
   == (LEGION_READ_PRIV | LEGION_DISCARD_OUTPUT_MASK))
#define FILTER_DISCARD(req) \
  ((req).privilege & ~(LEGION_DISCARD_INPUT_MASK | LEGION_DISCARD_OUTPUT_MASK))
#define IS_COLLECTIVE(req) \
  (((req).prop & LEGION_COLLECTIVE_MASK) == LEGION_COLLECTIVE_MASK)
#define PRIV_ONLY(req) \
  ((req).privilege & LEGION_READ_WRITE)
#define IS_REDUCE(req) \
  (((req).privilege & LEGION_READ_WRITE) == LEGION_REDUCE)
#define IS_EXCLUSIVE(req) \
  (((req).prop & LEGION_RELAXED) == LEGION_EXCLUSIVE)
#define IS_ATOMIC(req) \
  (((req).prop & LEGION_RELAXED) == LEGION_ATOMIC)
#define IS_SIMULT(req) \
  (((req).prop & LEGION_RELAXED) == LEGION_SIMULTANEOUS)
#define IS_RELAXED(req) \
  (((req).prop & LEGION_RELAXED) == LEGION_RELAXED)

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
      inline void serialize(const Internal::CopySrcDstField &field);
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
      Deserializer(const void *buf, size_t buffer_size
#ifdef DEBUG_LEGION
          , size_t ctx_bytes = 0
#endif
          )
        : total_bytes(buffer_size), buffer((const char*)buf), index(0)
#ifdef DEBUG_LEGION
          , context_bytes(ctx_bytes)
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
      inline void deserialize(Internal::CopySrcDstField &field);
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
    public:
      inline size_t get_context_bytes(void) const { return context_bytes; }
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
        if (IS_WRITE_DISCARD(u2))
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
        {
          // Exclusive and atomic coherence are effectively the same
          // thing in these contexts. Similarly simultaneous/relaxed
          // are also effectively the same thing for reductions.
          // However, mixing one of those "group modes" with the other
          // can result in races, so we don't allow that
          if (u1.prop != u2.prop)
          {
            const bool atomic1 = IS_EXCLUSIVE(u1) || IS_ATOMIC(u1);
            const bool atomic2 = IS_EXCLUSIVE(u2) || IS_ATOMIC(u2);
            if (atomic1 != atomic2)
              return LEGION_TRUE_DEPENDENCE;
          }
          return LEGION_NO_DEPENDENCE;
        }
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

    //--------------------------------------------------------------------------
    static inline bool configure_collective_settings(const int participants,
                                                     const int local_space,
                                                     int &collective_radix,
                                                     int &collective_log_radix,
                                                     int &collective_stages,
                                                     int &participating_spaces,
                                                     int &collective_last_radix)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(participants > 0);
      assert(collective_radix > 1);
#endif
      const int MultiplyDeBruijnBitPosition[32] = 
      {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
          8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
      };
      // First adjust the radix based on the number of nodes if necessary
      if (collective_radix > participants)
      {
        if (participants == 1)
        {
          // Handle the unsual case of a single participant
          collective_radix = 0;
          collective_log_radix = 0;
          collective_stages = 0;
          participating_spaces = 1;
          collective_last_radix = 0;
          return (local_space == 0);
        }
        else
          collective_radix = participants;
      }
      // Adjust the radix to the next smallest power of 2
      uint32_t radix_copy = collective_radix;
      for (int i = 0; i < 5; i++)
        radix_copy |= radix_copy >> (1 << i);
      collective_log_radix = 
        MultiplyDeBruijnBitPosition[(uint32_t)(radix_copy * 0x07C4ACDDU) >> 27];
      if (collective_radix != (1 << collective_log_radix))
        collective_radix = (1 << collective_log_radix);

      // Compute the number of stages
      uint32_t node_copy = participants;
      for (int i = 0; i < 5; i++)
        node_copy |= node_copy >> (1 << i);
      // Now we have it log 2
      int log_nodes = 
        MultiplyDeBruijnBitPosition[(uint32_t)(node_copy * 0x07C4ACDDU) >> 27];

      // Stages round up in case of incomplete stages
      collective_stages = 
        (log_nodes + collective_log_radix - 1) / collective_log_radix;
      int log_remainder = log_nodes % collective_log_radix;
      if (log_remainder > 0)
      {
        // We have an incomplete last stage
        collective_last_radix = 1 << log_remainder;
        // Now we can compute the number of participating stages
        participating_spaces = 
          1 << ((collective_stages - 1) * collective_log_radix +
                 log_remainder);
      }
      else
      {
        collective_last_radix = collective_radix;
        participating_spaces = 1 << (collective_stages * collective_log_radix);
      }
#ifdef DEBUG_LEGION
      assert((participating_spaces % collective_radix) == 0);
#endif
      const bool participant = (local_space < participating_spaces);
      return participant;
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
      class HashVerifier {
      public:
        virtual bool verify_hash(const uint64_t hash[2],
            const char *description, Provenance *provenance, bool every) = 0;
      };
    public:
      Murmur3Hasher(HashVerifier *verifier, bool precise,
                    bool verify_every_call, Provenance *provenance = NULL,
                    uint64_t seed = 0xCC892563);
      Murmur3Hasher(const Murmur3Hasher&) = delete;
      Murmur3Hasher& operator=(const Murmur3Hasher&) = delete;
    public:
      template<typename T>
      inline void hash(const T &value, const char *description);
      inline void hash(const void *values, size_t size,const char *description);
      inline bool verify(const char *description, bool every_call = false);
    protected:
      template<typename T>
      inline void hash(const T &value);
      inline void hash(const void *value, size_t size);
      inline uint64_t rotl64(uint64_t x, uint8_t r);
      inline uint64_t fmix64(uint64_t k);
    public:
      HashVerifier *const verifier;
      Provenance *const provenance;
    protected:
      uint8_t blocks[16];
      uint64_t h1, h2, len;
      uint8_t bytes;
    public:
      const bool precise;
      const bool verify_every_call;
    public:
      static constexpr uint64_t c1 = 0x87c37b91114253d5ULL;
      static constexpr uint64_t c2 = 0x4cf5ad432745937fULL;
    private:
      struct IndexSpaceHasher {
      public:
        IndexSpaceHasher(const Domain &d, Murmur3Hasher &h)
          : domain(d), hasher(h) { }
      public:
        template<typename N, typename T>
        static inline void demux(IndexSpaceHasher *functor)
        {
          const DomainT<N::N,T> is = functor->domain;
          for (RectInDomainIterator<N::N,T> itr(is); itr(); itr.step())
          {
            const Rect<N::N,T> rect = *itr;
            for (int d = 0; d < N::N; d++)
            {
              functor->hasher.hash(rect.lo[d]);
              functor->hasher.hash(rect.hi[d]);
            }
          }
        }
      public:
        const Domain &domain;
        Murmur3Hasher &hasher;
      };
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

    /////////////////////////////////////////////////////////////
    // BasicRangeAllocator
    /////////////////////////////////////////////////////////////
    // manages a basic free list of ranges (using range type RT) and allocated
    //  ranges, which are tagged (tag type TT)
    // NOT thread-safe - must be protected from outside
    template <typename RT, typename TT>
    class BasicRangeAllocator {
    public:
      struct Range {
        //Range(RT _first, RT _last);

        RT first, last;  // half-open range: [first, last)
        unsigned prev, next;  // double-linked list of all ranges (by index)
        unsigned prev_free, next_free; // double-linked list of just free ranges
      };

      std::map<TT, unsigned> allocated;// direct lookup allocated ranges by tag
#ifdef DEBUG_LEGION
      std::map<RT, unsigned> by_first; // direct lookup of all ranges by first
      // TODO: sized-based lookup of free ranges
#endif

      static const unsigned SENTINEL = 0;
      // TODO: small (medium?) vector opt
      std::vector<Range> ranges;

      BasicRangeAllocator(void);
      ~BasicRangeAllocator(void);

      void swap(BasicRangeAllocator<RT, TT>& swap_with);

      void add_range(RT first, RT last);
      bool can_allocate(TT tag, RT size, RT alignment);
      bool allocate(TT tag, RT size, RT alignment, RT& first);
      void deallocate(TT tag, bool missing_ok = false);
      bool lookup(TT tag, RT& first, RT& size);
      size_t get_size(TT tag);
      void dump_all_free_ranges(Realm::Logger logger);

    protected:
      unsigned first_free_range;
      unsigned alloc_range(RT first, RT last);
      void free_range(unsigned index);
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
    inline void Serializer::serialize(const Internal::CopySrcDstField &field)
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
    inline void Deserializer::deserialize(Internal::CopySrcDstField &field)
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
        field.fill_data.indirect = NULL;
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
    inline Murmur3Hasher::Murmur3Hasher(HashVerifier *v, bool pre, bool every, 
                                        Provenance *prov, uint64_t seed)
      : verifier(v), provenance(prov), h1(seed), h2(seed), len(0), bytes(0),
        precise(pre), verify_every_call(every)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<typename T>
    inline void Murmur3Hasher::hash(const T &value, const char *description)
    //-------------------------------------------------------------------------
    {
      hash<T>(value);
      if (verify_every_call)
        verify(description, true/*verify every call*/);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    inline void Murmur3Hasher::hash(const T &value)
    //-------------------------------------------------------------------------
    {
      const T *ptr = &value;
      const uint8_t *data = NULL;
      static_assert(sizeof(ptr) == sizeof(data), "Fuck c++");
      memcpy(&data, &ptr, sizeof(data));
      for (unsigned idx = 0; idx < sizeof(T); idx++)
      {
        blocks[bytes++] = data[idx];
        if (bytes == 16)
        {
          // body
          uint64_t k1, k2;
          memcpy(&k1, blocks, sizeof(k1));
          memcpy(&k2, blocks+sizeof(k1), sizeof(k2));
          static_assert(sizeof(blocks) == (sizeof(k1)+sizeof(k2)), "sanity");
          k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
          h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;
          k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;
          h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
          len += 16;
          bytes = 0;
        }
      }
    }

    //-------------------------------------------------------------------------
    template<>
    inline void Murmur3Hasher::hash<Domain>(const Domain &value,
                                            const char *description)
    //-------------------------------------------------------------------------
    {
      for (int i = 0; i < 2*value.dim; i++)
        hash(value.rect_data[i]);
      if (!value.dense() && precise)
      {
        IndexSpaceHasher functor(value, *this);
        Internal::NT_TemplateHelper::demux<IndexSpaceHasher>(value.is_type,
                                                             &functor);
      }
      if (verify_every_call)
        verify(description, true/*verify every call*/);
    }

    //-------------------------------------------------------------------------
    template<>
    inline void Murmur3Hasher::hash<DomainPoint>(const DomainPoint &value,
                                                 const char *description)
    //-------------------------------------------------------------------------
    {
      for (int i = 0; i < value.dim; i++)
        hash(value.point_data[i]);
      if (verify_every_call)
        verify(description, true/*verify every call*/);
    }

    //-------------------------------------------------------------------------
    inline void Murmur3Hasher::hash(const void *value, size_t size,
                                    const char *description)
    //-------------------------------------------------------------------------
    {
      hash(value, size);
      if (verify_every_call)
        verify(description, true/*verify every call*/);
    }

    //-------------------------------------------------------------------------
    inline void Murmur3Hasher::hash(const void *value, size_t size)
    //-------------------------------------------------------------------------
    {
      const uint8_t *data = NULL;
      static_assert(sizeof(data) == sizeof(value), "Fuck c++");
      memcpy(&data, &value, sizeof(data));
      for (unsigned idx = 0; idx < size; idx++)
      {
        blocks[bytes++] = data[idx];
        if (bytes == 16)
        {
          // body
          uint64_t k1, k2;
          memcpy(&k1, blocks, sizeof(k1));
          memcpy(&k2, blocks+sizeof(k1), sizeof(k2));
          static_assert(sizeof(blocks) == (sizeof(k1)+sizeof(k2)), "sanity");
          k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
          h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;
          k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;
          h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
          len += 16;
          bytes = 0;
        }
      }
    }

    //-------------------------------------------------------------------------
    inline bool Murmur3Hasher::verify(const char *description, bool every_call)
    //-------------------------------------------------------------------------
    {
      // tail
      uint64_t k1 = 0;
      uint64_t k2 = 0;
      switch (bytes)
      {
        case 15: k2 ^= ((uint64_t)blocks[14]) << 48;
        case 14: k2 ^= ((uint64_t)blocks[13]) << 40;
        case 13: k2 ^= ((uint64_t)blocks[12]) << 32;
        case 12: k2 ^= ((uint64_t)blocks[11]) << 24;
        case 11: k2 ^= ((uint64_t)blocks[10]) << 16;
        case 10: k2 ^= ((uint64_t)blocks[ 9]) << 8;
        case  9: k2 ^= ((uint64_t)blocks[ 8]) << 0;
                 k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

        case  8: k1 ^= ((uint64_t)blocks[ 7]) << 56;
        case  7: k1 ^= ((uint64_t)blocks[ 6]) << 48;
        case  6: k1 ^= ((uint64_t)blocks[ 5]) << 40;
        case  5: k1 ^= ((uint64_t)blocks[ 4]) << 32;
        case  4: k1 ^= ((uint64_t)blocks[ 3]) << 24;
        case  3: k1 ^= ((uint64_t)blocks[ 2]) << 16;
        case  2: k1 ^= ((uint64_t)blocks[ 1]) << 8;
        case  1: k1 ^= ((uint64_t)blocks[ 0]) << 0;
                 k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
      }
      
      // finalization
      len += bytes;

      h1 ^= len; h2 ^= len;

      h1 += h2;
      h2 += h1;

      h1 = fmix64(h1);
      h2 = fmix64(h2);

      h1 += h2;
      h2 += h1;

      uint64_t hash[2] = { h1, h2 };
      return verifier->verify_hash(hash, description, provenance, every_call);
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
      NodeBase *r = root.load();
      if (r == NULL)
        return 0;
      size_t elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
      for (int i = 0; i < r->level; i++)
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
        result = leaf->elems[offset].load();
        if (result == NULL)
        {
          result = new ET();
          leaf->elems[offset].store(result);
        }
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
      ET *result = leaf->elems[offset].load();
      if (result == NULL)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        result = leaf->elems[offset].load();
        if (result == NULL)
        {
          result = new ET(arg);
          leaf->elems[offset].store(result);
        }
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
      ET *result = leaf->elems[offset].load();
      if (result == NULL)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        result = leaf->elems[offset].load();
        if (result == NULL)
        {
          result = new ET(arg1, arg2);
          leaf->elems[offset].store(result);
        }
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
      NodeBase *n = root.load();
      if (!n || (n->level < level_needed)) 
      {
        AutoLock l(lock); 
        n = root.load();
        if (n)
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
          child = inner->elems[i].load();
          if (child == NULL)
          {
            int child_level = inner->level - 1;
            int child_shift = 
              (ALLOCATOR::LEAF_BITS + child_level * ALLOCATOR::INNER_BITS);
            IT child_first = inner->first_index + (i << child_shift);
            IT child_last = inner->first_index + ((i + 1) << child_shift) - 1;

            child = new_tree_node(child_level, child_first, child_last);
            inner->elems[i].store(child);
          }
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

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline BasicRangeAllocator<RT,TT>::BasicRangeAllocator(void)
    //-------------------------------------------------------------------------
      : first_free_range(SENTINEL)
    {
      ranges.resize(1);
      Range& s = ranges[SENTINEL];
      s.first = RT(-1);
      s.last = 0;
      s.prev = s.next = s.prev_free = s.next_free = SENTINEL;
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline BasicRangeAllocator<RT,TT>::~BasicRangeAllocator(void)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline void BasicRangeAllocator<RT,TT>::swap(
                                        BasicRangeAllocator<RT, TT>& swap_with)
    //-------------------------------------------------------------------------
    {
      allocated.swap(swap_with.allocated);
#ifdef DEBUG_LEGION
      by_first.swap(swap_with.by_first);
#endif
      ranges.swap(swap_with.ranges);
      std::swap(first_free_range, swap_with.first_free_range);
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline void BasicRangeAllocator<RT,TT>::add_range(RT first, RT last)
    //-------------------------------------------------------------------------
    {
      // ignore empty ranges
      if(first == last)
        return;

      int new_idx = alloc_range(first, last);

      Range& newr = ranges[new_idx];
      Range& sentinel = ranges[SENTINEL];

      // simple case - starting range
      if(sentinel.next == SENTINEL) {
        // all block list
        newr.prev = newr.next = SENTINEL;
        sentinel.prev = sentinel.next = new_idx;
        // free block list
        newr.prev_free = newr.next_free = SENTINEL;
        sentinel.prev_free = sentinel.next_free = new_idx;

#ifdef DEBUG_LEGION
        by_first[first] = new_idx;
#endif
        return;
      }

      assert(0);
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline unsigned BasicRangeAllocator<RT,TT>::alloc_range(RT first, RT last)
    //-------------------------------------------------------------------------
    {
      // find/make a free index in the range list for this range
      int new_idx;
      if(first_free_range != SENTINEL) {
        new_idx = first_free_range;
        first_free_range = ranges[new_idx].next;
      } else {
        new_idx = ranges.size();
        ranges.resize(new_idx + 1);
      }
      ranges[new_idx].first = first;
      ranges[new_idx].last = last;
      return new_idx;
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline void BasicRangeAllocator<RT,TT>::free_range(unsigned index)
    //-------------------------------------------------------------------------
    {
      ranges[index].next = first_free_range;
      first_free_range = index;
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline bool BasicRangeAllocator<RT,TT>::can_allocate(TT tag,
                                                         RT size, RT alignment)
    //-------------------------------------------------------------------------
    {
      // empty allocation requests are trivial
      if(size == 0) {
        return true;
      }

      // walk free ranges and just take the first that fits
      unsigned idx = ranges[SENTINEL].next_free;
      while(idx != SENTINEL) {
        Range *r = &ranges[idx];

        RT ofs = 0;
        if(alignment) {
          RT rem = r->first % alignment;
          if(rem > 0)
            ofs = alignment - rem;
        }
        // do we have enough space?
        if((r->last - r->first) >= (size + ofs))
          return true;

        // no, go to next one
        idx = r->next_free;
      }

      // allocation failed
      return false;
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline bool BasicRangeAllocator<RT,TT>::allocate(TT tag, RT size, 
                                                 RT alignment, RT& alloc_first)
    //-------------------------------------------------------------------------
    {
      // empty allocation requests are trivial
      if(size == 0) {
        allocated[tag] = SENTINEL;
        return true;
      }

      // walk free ranges and just take the first that fits
      unsigned idx = ranges[SENTINEL].next_free;
      while(idx != SENTINEL) {
        Range *r = &ranges[idx];

        RT ofs = 0;
        if(alignment) {
          RT rem = r->first % alignment;
          if(rem > 0)
            ofs = alignment - rem;
        }
        // do we have enough space?
        if((r->last - r->first) >= (size + ofs)) {
          // yes, but we may need to chop things up to make the exact range 
          alloc_first = r->first + ofs;
          RT alloc_last = alloc_first + size;

          // do we need to carve off a new (free) block before us?
          if(alloc_first != r->first) {
            unsigned new_idx = alloc_range(r->first, alloc_first);
            Range *new_prev = &ranges[new_idx];
            r = &ranges[idx];  // alloc may have moved this!

            r->first = alloc_first;
            // insert into all-block dllist
            new_prev->prev = r->prev;
            new_prev->next = idx;
            ranges[r->prev].next = new_idx;
            r->prev = new_idx;
            // insert into free-block dllist
            new_prev->prev_free = r->prev_free;
            new_prev->next_free = idx;
            ranges[r->prev_free].next_free = new_idx;
            r->prev_free = new_idx;

#ifdef DEBUG_LEGION
            // fix up by_first entries
            by_first[r->first] = new_idx;
            by_first[alloc_first] = idx;
#endif
          }

          // two cases to deal with
          if(alloc_last == r->last) {
            // case 1 - exact fit
            //
            // all we have to do here is remove this range from the free range 
            // dlist and add to the allocated lookup map
            ranges[r->prev_free].next_free = r->next_free;
            ranges[r->next_free].prev_free = r->prev_free;
          } else {
            // case 2 - leftover at end - put in new range
            unsigned after_idx = alloc_range(alloc_last, r->last);
            Range *r_after = &ranges[after_idx];
            r = &ranges[idx];  // alloc may have moved this!

#ifdef DEBUG_LEGION
            by_first[alloc_last] = after_idx;
#endif
            r->last = alloc_last;

            // r_after goes after r in all block list
            r_after->prev = idx;
            r_after->next = r->next;
            r->next = after_idx;
            ranges[r_after->next].prev = after_idx;

            // r_after replaces r in the free block list
            r_after->prev_free = r->prev_free;
            r_after->next_free = r->next_free;
            ranges[r_after->next_free].prev_free = after_idx;
            ranges[r_after->prev_free].next_free = after_idx;
          }

          // tie this off because we use it to detect allocated-ness
          r->prev_free = r->next_free = idx;

          allocated[tag] = idx;
          return true;
        }

        // no, go to next one
        idx = r->next_free;
      }
      // allocation failed
      return false;
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline void BasicRangeAllocator<RT,TT>::deallocate(TT tag,
                                                   bool missing_ok /*= false*/)
    //-------------------------------------------------------------------------
    {
      typename std::map<TT, unsigned>::iterator it = allocated.find(tag);
      if(it == allocated.end()) {
        assert(missing_ok);
        return;
      }
      unsigned del_idx = it->second;
      allocated.erase(it);

      // if there was no Range associated with this tag, it was an zero-size
      //  allocation, and there's nothing to add to the free list
      if(del_idx == SENTINEL)
        return;

      Range& r = ranges[del_idx];

      unsigned pf_idx = r.prev;
      while((pf_idx != SENTINEL) && (ranges[pf_idx].prev_free == pf_idx)) {
        pf_idx = ranges[pf_idx].prev;
        assert(pf_idx != del_idx);  // wrapping around would be bad
      }
      unsigned nf_idx = r.next;
      while((nf_idx != SENTINEL) && (ranges[nf_idx].next_free == nf_idx)) {
        nf_idx = ranges[nf_idx].next;
        assert(nf_idx != del_idx);
      }

      // do we need to merge?
      bool merge_prev = (pf_idx == r.prev) && (pf_idx != SENTINEL);
      bool merge_next = (nf_idx == r.next) && (nf_idx != SENTINEL);

      // four cases - ordered to match the allocation cases
      if(!merge_next) {
        if(!merge_prev) {
          // case 1 - no merging (exact match)
          // just add ourselves to the free list
          r.prev_free = pf_idx;
          r.next_free = nf_idx;
          ranges[pf_idx].next_free = del_idx;
          ranges[nf_idx].prev_free = del_idx;
        } else {
          // case 2 - merge before
          // merge ourselves into the range before
          Range& r_before = ranges[pf_idx];

          r_before.last = r.last;
          r_before.next = r.next;
          ranges[r.next].prev = pf_idx;
          // r_before was already in free list, so no changes to that

#ifdef DEBUG_LEGION
          by_first.erase(r.first);
#endif
          free_range(del_idx);
        }
      } else {
        if(!merge_prev) {
          // case 3 - merge after
          // merge ourselves into the range after
          Range& r_after = ranges[nf_idx];

#ifdef DEBUG_LEGION
          by_first[r.first] = nf_idx;
          by_first.erase(r_after.first);
#endif

          r_after.first = r.first;
          r_after.prev = r.prev;
          ranges[r.prev].next = nf_idx;
          // r_after was already in the free list, so no changes to that

          free_range(del_idx);
        } else {
          // case 4 - merge both
          // merge both ourselves and range after into range before
          Range& r_before = ranges[pf_idx];
          Range& r_after = ranges[nf_idx];

          r_before.last = r_after.last;
#ifdef DEBUG_LEGION
          by_first.erase(r.first);
          by_first.erase(r_after.first);
#endif

          // adjust both normal list and free list
          r_before.next = r_after.next;
          ranges[r_after.next].prev = pf_idx;

          r_before.next_free = r_after.next_free;
          ranges[r_after.next_free].prev_free = pf_idx;

          free_range(del_idx);
          free_range(nf_idx);
        }
      }
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline bool BasicRangeAllocator<RT,TT>::lookup(TT tag, RT& first, RT& size)
    //-------------------------------------------------------------------------
    {
      typename std::map<TT, unsigned>::iterator it = allocated.find(tag);

      if(it != allocated.end()) {
        // if there was no Range associated with this tag, it was an zero-size
        //  allocation
        if(it->second == SENTINEL) {
          first = 0;
          size = 0;
        } else {
          const Range& r = ranges[it->second];
          first = r.first;
          size = r.last - r.first;
        }
        return true;
      } else
        return false;
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

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline size_t BasicRangeAllocator<RT,TT>::get_size(TT tag)
    //-------------------------------------------------------------------------
    {
      typename std::map<TT, unsigned>::iterator it = allocated.find(tag);
      if(it == allocated.end()) {
        assert(false);
        return 0;
      }
      unsigned idx = it->second;
      if (idx == SENTINEL) return 0;

      Range& r = ranges[idx];
      return r.last - r.first;
    }

    //-------------------------------------------------------------------------
    template <typename RT, typename TT>
    inline void BasicRangeAllocator<RT,TT>::dump_all_free_ranges(
                                                          Realm::Logger logger)
    //-------------------------------------------------------------------------
    {
      unsigned idx = ranges[SENTINEL].next_free;
      while (idx != SENTINEL) {
        Range &r = ranges[idx];
        logger.debug("range %u: %zd bytes [%zx,%zx)",
                     idx, r.last - r.first, r.first, r.last);
        idx = r.next_free;
      }
    }

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

    // This is a generalization of the above method but takes a list of 
    // anything that has the same members as a FieldSet
    //--------------------------------------------------------------------------
    template<typename T, typename CT>
    inline void compute_field_sets(FieldMask universe_mask,
                                   const LegionMap<T,FieldMask> &inputs,
                                   LegionList<CT> &output_sets)
    //--------------------------------------------------------------------------
    {
      // Special cases for empty and size 1 sets
      if (inputs.empty())
      {
        if (!!universe_mask)
          output_sets.push_back(CT(universe_mask));
        return;
      }
      else if (inputs.size() == 1)
      {
        typename LegionMap<T,FieldMask>::const_iterator first = 
          inputs.begin();
        output_sets.push_back(CT(first->second));
        CT &last = output_sets.back();
        last.elements.insert(first->first);
        if (!!universe_mask)
        {
          universe_mask -= first->second;
          if (!!universe_mask)
            output_sets.push_back(CT(universe_mask));
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
        for (typename LegionList<CT>::iterator it = 
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
            output_sets.push_back(CT(overlap));
            CT &last = output_sets.back();
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
          it = output_sets.insert(it, CT(overlap));
          it->elements = temp_elements;
          it->elements.insert(pit->first);
          remaining -= overlap;
          continue;
        }
        if (!inserted)
        {
          output_sets.push_back(CT(remaining));
          CT &last = output_sets.back();
          last.elements.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any elements, but them in their own set.
      // Put it on the front because it is the copy with
      // no elements so it can start right away!
      if (!!universe_mask)
        output_sets.push_front(CT(universe_mask));
    }

    /**
     * \class FieldMaskSet 
     * A template helper class for tracking collections of 
     * objects associated with different sets of fields
     */
    template<typename T, AllocationType A = UNTRACKED_ALLOC,
             bool DETERMINISTIC = false>
    class FieldMaskSet : 
      public LegionHeapify<FieldMaskSet<T> > {
    private:
      // Call the deterministic pointer less method for
      // any types that have asked for deterministic sets
      template<typename U>
      struct DeterministicComparator {
      public:
        inline bool operator()(const U *one, const U *two) const
          { return one->deterministic_pointer_less(two); }
      };
      using Comparator = typename std::conditional<DETERMINISTIC,
            DeterministicComparator<T>, std::less<const T*> >::type;
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
            typename LegionMap<T*,FieldMask,A,Comparator>::iterator _it,
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
        inline void erase(LegionMap<T*,FieldMask,A,Comparator> &target)
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
        typename LegionMap<T*,FieldMask,A,Comparator>::iterator it;
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
            typename LegionMap<T*,FieldMask,A,Comparator>::const_iterator _it,
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
        typename LegionMap<T*,FieldMask,A,Comparator>::const_iterator it;
        bool single;
      };
    public:
      FieldMaskSet(void)
        : single(true) { entries.single_entry = NULL; }
      inline FieldMaskSet(T *init, const FieldMask &m, bool no_null = true);
      inline FieldMaskSet(const FieldMaskSet<T,A,DETERMINISTIC> &rhs);
      inline FieldMaskSet(FieldMaskSet<T,A,DETERMINISTIC> &&rhs);
      // If copy is set to false then this is a move constructor
      inline FieldMaskSet(FieldMaskSet<T,A,DETERMINISTIC> &rhs, bool copy);
      ~FieldMaskSet(void) { clear(); }
    public:
      inline FieldMaskSet& operator=(const FieldMaskSet<T,A,DETERMINISTIC> &rh);
      inline FieldMaskSet& operator=(FieldMaskSet<T,A,DETERMINISTIC> &&rhs);
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
      inline void filter(const FieldMask &filter, bool tighten = true);
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
      template<typename T2, AllocationType A2, bool D2>
      friend class FieldMaskSet; 

      // Fun with C, keep these two fields first and in this order
      // so that a FieldMaskSet of size 1 looks the same as an entry
      // in the STL Map in the multi-entries case, 
      // provides goodness for the iterator
      union {
        T *single_entry;
        LegionMap<T*,FieldMask,A,Comparator> *multi_entries;
      } entries;
      // This can be an overapproximation if we have multiple entries
      FieldMask valid_fields;
      bool single;
    };

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline FieldMaskSet<T,A,D>::FieldMaskSet(T *init, const FieldMask &mask, 
                                             bool no_null)
      : single(true)
    //--------------------------------------------------------------------------
    {
      if (!no_null || (init != NULL))
      {
        entries.single_entry = init;
        valid_fields = mask;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline FieldMaskSet<T,A,D>::FieldMaskSet(const FieldMaskSet<T,A,D> &rhs)
      : valid_fields(rhs.valid_fields), single(rhs.single)
    //--------------------------------------------------------------------------
    {
      if (single)
        entries.single_entry = rhs.entries.single_entry;
      else
        entries.multi_entries = new LegionMap<T*,FieldMask,A,Comparator>(
            rhs.entries.multi_entries->begin(),
            rhs.entries.multi_entries->end());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline FieldMaskSet<T,A,D>::FieldMaskSet(FieldMaskSet<T,A,D> &&rhs)
      : valid_fields(rhs.valid_fields), single(rhs.single)
    //--------------------------------------------------------------------------
    {
      if (single)
        entries.single_entry = rhs.entries.single_entry;
      else
        entries.multi_entries = rhs.entries.multi_entries;
      rhs.valid_fields.clear();
      rhs.single = true;
      rhs.entries.single_entry = NULL;
    }
    
    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline FieldMaskSet<T,A,D>::FieldMaskSet(FieldMaskSet<T,A,D> &rhs,bool copy)
      : valid_fields(rhs.valid_fields), single(rhs.single)
    //--------------------------------------------------------------------------
    {
      if (copy)
      {
        if (single)
          entries.single_entry = rhs.entries.single_entry;
        else
          entries.multi_entries = new LegionMap<T*,FieldMask,A,Comparator>(
              rhs.entries.multi_entries->begin(),
              rhs.entries.multi_entries->end());
      }
      else
      {
        if (single)
          entries.single_entry = rhs.entries.single_entry;
        else
          entries.multi_entries = rhs.entries.multi_entries;
        rhs.entries.single_entry = NULL;
        rhs.valid_fields.clear();
        rhs.single = true;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline FieldMaskSet<T,A,D>& FieldMaskSet<T,A,D>::operator=(
                                                 const FieldMaskSet<T,A,D> &rhs)
    //--------------------------------------------------------------------------
    {
      // Check our current state
      if (single != rhs.single)
      {
        // Different data structures
        if (single)
        {
          entries.multi_entries = new LegionMap<T*,FieldMask,A,Comparator>(
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
    template<typename T, AllocationType A, bool D>
    inline FieldMaskSet<T,A,D>& FieldMaskSet<T,A,D>::operator=(
                                                      FieldMaskSet<T,A,D> &&rhs)
    //--------------------------------------------------------------------------
    {
      // Check our current state
      if (single != rhs.single)
      {
        // Different data structures
        if (single)
        {
          entries.multi_entries = rhs.entries.multi_entries;
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
        {
          entries.single_entry = rhs.entries.single_entry;
        }
        else
        {
          delete entries.multi_entries;
          entries.multi_entries = rhs.entries.multi_entries;
        }
      }
      valid_fields = rhs.valid_fields;
      rhs.valid_fields.clear();
      rhs.single = true;
      rhs.entries.single_entry = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline const FieldMask& FieldMaskSet<T,A,D>::tighten_valid_mask(void)
    //--------------------------------------------------------------------------
    {
      // If we're single then there is nothing to do as we're already tight
      if (single)
        return valid_fields;
      valid_fields.clear();
      for (typename LegionMap<T*,FieldMask,A,Comparator>::const_iterator it = 
            entries.multi_entries->begin(); it !=
            entries.multi_entries->end(); it++)
        valid_fields |= it->second;
      return valid_fields;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::relax_valid_mask(const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      if (single && (entries.single_entry != NULL))
      {
        if (!(m - valid_fields))
          return;
        // have to avoid the aliasing case
        T *entry = entries.single_entry;
        entries.multi_entries = new LegionMap<T*,FieldMask,A,Comparator>();
        entries.multi_entries->insert(std::make_pair(entry, valid_fields));
        single = false;
      }
      valid_fields |= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::filter_valid_mask(const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      valid_fields -= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::restrict_valid_mask(const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      valid_fields &= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline const FieldMask& FieldMaskSet<T,A,D>::operator[](T *entry) const
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
        typename LegionMap<T*,FieldMask,A,Comparator>::const_iterator finder =
          entries.multi_entries->find(entry);
#ifdef DEBUG_LEGION
        assert(finder != entries.multi_entries->end());
#endif
        return finder->second;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline bool FieldMaskSet<T,A,D>::insert(T *entry, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      if (single)
      {
        if (entries.single_entry == NULL)
        {
          entries.single_entry = entry;
          valid_fields |= mask;
        }
        else if (entries.single_entry == entry)
        {
          valid_fields |= mask;
          result = false;
        }
        else
        {
          // Go to multi
          LegionMap<T*,FieldMask,A,Comparator> *multi =
            new LegionMap<T*,FieldMask,A,Comparator>();
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
        typename LegionMap<T*,FieldMask,A,Comparator>::iterator finder = 
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
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::filter(const FieldMask &filter,
                                            bool tighten)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (entries.single_entry != NULL)
        {
          if (tighten)
            valid_fields -= filter;
          if (!valid_fields)
            entries.single_entry = NULL;
        }
      }
      else
      {
        if (tighten)
          valid_fields -= filter;
        if (!valid_fields || (!tighten && (filter == valid_fields)))
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
          for (typename LegionMap<T*,FieldMask,A,Comparator>::iterator it =
                entries.multi_entries->begin(); it !=
                entries.multi_entries->end(); it++)
          {
            it->second -= filter;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            if (to_delete.size() < entries.multi_entries->size())
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
              else if ((entries.multi_entries->size() == 1) &&
                  (entries.multi_entries->begin()->second == valid_fields))
              {
                typename LegionMap<T*,FieldMask,A,Comparator>::iterator last =
                  entries.multi_entries->begin();     
                T *temp = last->first; 
                delete entries.multi_entries;
                entries.single_entry = temp;
                single = true;
              }
            }
            else
            {
              delete entries.multi_entries;
              entries.multi_entries = NULL;
              single = true;
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::erase(T *to_erase)
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
        typename LegionMap<T*,FieldMask,A,Comparator>::iterator finder = 
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
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::clear(void)
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
    template<typename T, AllocationType A, bool D>
    inline size_t FieldMaskSet<T,A,D>::size(void) const
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
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::swap(FieldMaskSet &other)
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
    template<typename T, AllocationType A, bool D>
    inline typename FieldMaskSet<T,A,D>::iterator 
                                                FieldMaskSet<T,A,D>::begin(void)
    //--------------------------------------------------------------------------
    {
      // Scariness!
      if (single)
      {
        // If we're empty return end
        if (entries.single_entry == NULL)
          return end();
        FieldMaskSet<T,A,D> *ptr = this;
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(result) == sizeof(ptr), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return iterator(this, result); 
      }
      else
        return iterator(this, entries.multi_entries->begin());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline typename FieldMaskSet<T,A,D>::iterator
                                                 FieldMaskSet<T,A,D>::find(T *e)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((entries.single_entry == NULL) || (entries.single_entry != e))
          return end();
        FieldMaskSet<T,A,D> *ptr = this;
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(result) == sizeof(ptr), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return iterator(this, result);
      }
      else
      {
        typename LegionMap<T*,FieldMask,A,Comparator>::iterator finder = 
          entries.multi_entries->find(e);
        if (finder == entries.multi_entries->end())
          return end();
        return iterator(this, finder);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::erase(iterator &it)
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
          typename LegionMap<T*,FieldMask,A,Comparator>::iterator finder =
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
    template<typename T, AllocationType A, bool D>
    inline typename FieldMaskSet<T,A,D>::iterator FieldMaskSet<T,A,D>::end(void)
    //--------------------------------------------------------------------------
    {
      if (single)
        return iterator(this, NULL);
      else
        return iterator(this, entries.multi_entries->end(), true/*end*/);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline typename FieldMaskSet<T,A,D>::const_iterator 
                                          FieldMaskSet<T,A,D>::begin(void) const
    //--------------------------------------------------------------------------
    {
      // Scariness!
      if (single)
      {
        // If we're empty return end
        if (entries.single_entry == NULL)
          return end();
        FieldMaskSet<T,A,D> *ptr = const_cast<FieldMaskSet<T,A,D>*>(this);
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(ptr) == sizeof(result), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return const_iterator(this, result); 
      }
      else
        return const_iterator(this, entries.multi_entries->begin());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline typename FieldMaskSet<T,A,D>::const_iterator 
                                           FieldMaskSet<T,A,D>::find(T *e) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((entries.single_entry == NULL) || (entries.single_entry != e))
          return end();
        FieldMaskSet<T,A,D> *ptr = const_cast<FieldMaskSet<T,A,D>*>(this);
        std::pair<T*const,FieldMask> *result = NULL;
        static_assert(sizeof(ptr) == sizeof(result), "C++ is dumb");
        memcpy(&result, &ptr, sizeof(result));
        return const_iterator(this, result);
      }
      else
      {
        typename LegionMap<T*,FieldMask,A,Comparator>::const_iterator finder =
          entries.multi_entries->find(e);
        if (finder == entries.multi_entries->end())
          return end();
        return const_iterator(this, finder);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline typename FieldMaskSet<T,A,D>::const_iterator 
                                            FieldMaskSet<T,A,D>::end(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
        return const_iterator(this, NULL);
      else
        return const_iterator(this, entries.multi_entries->end(), true/*end*/);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationType A, bool D>
    inline void FieldMaskSet<T,A,D>::compute_field_sets(FieldMask universe_mask,
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

    //--------------------------------------------------------------------------
    template<typename T1, typename T2>
    inline void unique_join_on_field_mask_sets(
                   const FieldMaskSet<T1> &left, const FieldMaskSet<T2> &right,
                   LegionMap<std::pair<T1*,T2*>,FieldMask> &results)
    //--------------------------------------------------------------------------
    {
      if (left.empty() || right.empty())
        return;
      if (left.get_valid_mask() * right.get_valid_mask())
        return;
#ifdef DEBUG_LEGION
      FieldMask unique_test;
#endif
      if (left.size() == 1)
      {
        typename FieldMaskSet<T1>::const_iterator first = left.begin();
        for (typename FieldMaskSet<T2>::const_iterator it =
              right.begin(); it != right.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          const FieldMask overlap = first->second & it->second;
          if (!overlap)
            continue;
          const std::pair<T1*,T2*> key(first->first, it->first);
          results[key] = overlap;
        }
        return;
      }
      if (right.size() == 1)
      {
        typename FieldMaskSet<T2>::const_iterator first = right.begin();
        for (typename FieldMaskSet<T1>::const_iterator it =
              left.begin(); it != left.end(); it++)
        {
          const FieldMask overlap = first->second & it->second;
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          if (!overlap)
            continue;
          const std::pair<T1*,T2*> key(it->first, first->first);
          results[key] = overlap;
        }
        return;
      }
      // Build the lookup table for the one with fewer fields
      // since it is probably more costly to allocate memory
      if (left.get_valid_mask().pop_count() < 
          right.get_valid_mask().pop_count())
      {
        // Build the hash table for left
        std::map<unsigned,T1*> hash_table;
        for (typename FieldMaskSet<T1>::const_iterator it =
              left.begin(); it != left.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            hash_table[fidx] = it->first;
            fidx = it->second.find_next_set(fidx+1);
          }
        }
#ifdef DEBUG_LEGION
        unique_test.clear();
#endif
        for (typename FieldMaskSet<T2>::const_iterator it =
              right.begin(); it != right.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            typename std::map<unsigned,T1*>::const_iterator
              finder = hash_table.find(fidx);
            if (finder != hash_table.end())
            {
              const std::pair<T1*,T2*> key(finder->second,it->first);
              results[key].set_bit(fidx);
            }
            fidx = it->second.find_next_set(fidx+1);
          }
        }
      }
      else
      {
        // Build the hash table for the right
        std::map<unsigned,T2*> hash_table;
        for (typename FieldMaskSet<T2>::const_iterator it =
              right.begin(); it != right.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            hash_table[fidx] = it->first;
            fidx = it->second.find_next_set(fidx+1);
          }
        }
#ifdef DEBUG_LEGION
        unique_test.clear();
#endif
        for (typename FieldMaskSet<T1>::const_iterator it =
              left.begin(); it != left.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            typename std::map<unsigned,T2*>::const_iterator
              finder = hash_table.find(fidx);
            if (finder != hash_table.end())
            {
              const std::pair<T1*,T2*> key(it->first,finder->second);
              results[key].set_bit(fidx);
            }
            fidx = it->second.find_next_set(fidx+1);
          }
        }
      }
    }

  }; // namespace Internal
}; // namespace Legion 

#endif // __LEGION_UTILITIES_H__
