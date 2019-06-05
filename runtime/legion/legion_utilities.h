/* Copyright 2019 Stanford University, NVIDIA Corporation
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
#include "legion/legion_bitmask.h"

// Useful macros
#define IS_NO_ACCESS(req) (((req).privilege & READ_WRITE) == NO_ACCESS)
#define IS_READ_ONLY(req) (((req).privilege & READ_WRITE) <= READ_PRIV)
#define HAS_READ(req) ((req).privilege & READ_PRIV)
#define HAS_WRITE(req) ((req).privilege & (WRITE_PRIV | REDUCE))
#define IS_WRITE(req) ((req).privilege & WRITE_PRIV)
#define HAS_WRITE_DISCARD(req) (((req).privilege & WRITE_ONLY) == WRITE_ONLY)
#define IS_DISCARD(req) (((req).privilege & DISCARD_MASK) == DISCARD_MASK)
#define PRIV_ONLY(req) ((req).privilege & READ_WRITE)
#define IS_REDUCE(req) (((req).privilege & READ_WRITE) == REDUCE)
#define IS_EXCLUSIVE(req) ((req).prop == EXCLUSIVE)
#define IS_ATOMIC(req) ((req).prop == ATOMIC)
#define IS_SIMULT(req) ((req).prop == SIMULTANEOUS)
#define IS_RELAXED(req) ((req).prop == RELAXED)

namespace Legion {

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
        if (HAS_WRITE_DISCARD(u2))
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
