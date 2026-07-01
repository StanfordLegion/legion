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

// Included from dynamic_table.h - do not include this directly

// Useful for IDEs
#include "legion/utilities/dynamic_table.h"

namespace Legion {
  namespace Internal {

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::DynamicTable(void)
    //-------------------------------------------------------------------------
    {
      root.store(nullptr);
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    DynamicTable<ALLOCATOR>::~DynamicTable(void)
    //-------------------------------------------------------------------------
    {
      NodeBase* r = root.load();
      if (r != nullptr)
      {
        delete r;
        root.store(nullptr);
      }
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    typename DynamicTable<ALLOCATOR>::NodeBase*
        DynamicTable<ALLOCATOR>::new_tree_node(
            int level, IT first_index, IT last_index)
    //-------------------------------------------------------------------------
    {
      if (level > 0)
      {
        // we know how to create inner nodes
        typename ALLOCATOR::INNER_TYPE* inner =
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
      NodeBase* r = root.load();
      if (r == nullptr)
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
      while (index >= elems_addressable)
      {
        level_needed++;
        elems_addressable <<= ALLOCATOR::INNER_BITS;
      }

      NodeBase* n = root.load();
      if (!n || (n->level < level_needed))
        return false;

      // when we get here, root is high enough
      legion_assert(
          (level_needed <= n->level) && (index >= n->first_index) &&
          (index <= n->last_index));
      // now walk tree, populating the path we need
      while (n->level > 0)
      {
        // intermediate nodes
        typename ALLOCATOR::INNER_TYPE* inner =
            static_cast<typename ALLOCATOR::INNER_TYPE*>(n);
        IT i =
            ((index >>
              (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
             ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
        legion_assert((i >= 0) && (((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE));
        NodeBase* child = inner->elems[i].load();
        if (child == 0)
          return false;
        legion_assert(
            (child != 0) && (child->level == (n->level - 1)) &&
            (index >= child->first_index) && (index <= child->last_index));
        n = child;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    typename DynamicTable<ALLOCATOR>::ET* DynamicTable<ALLOCATOR>::lookup_entry(
        IT index)
    //-------------------------------------------------------------------------
    {
      NodeBase* n = lookup_leaf(index);
      // Now we've made it to the leaf node
      typename ALLOCATOR::LEAF_TYPE* leaf =
          static_cast<typename ALLOCATOR::LEAF_TYPE*>(n);
      int offset = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
      ET* result = leaf->elems[offset].load();
      if (result == nullptr)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        result = leaf->elems[offset].load();
        if (result == nullptr)
        {
          result = new ET();
          leaf->elems[offset].store(result);
        }
      }
      legion_assert(result != 0);
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    template<typename T>
    typename DynamicTable<ALLOCATOR>::ET* DynamicTable<ALLOCATOR>::lookup_entry(
        IT index, const T& arg)
    //-------------------------------------------------------------------------
    {
      NodeBase* n = lookup_leaf(index);
      // Now we've made it to the leaf node
      typename ALLOCATOR::LEAF_TYPE* leaf =
          static_cast<typename ALLOCATOR::LEAF_TYPE*>(n);
      int offset = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
      ET* result = leaf->elems[offset].load();
      if (result == nullptr)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        result = leaf->elems[offset].load();
        if (result == nullptr)
        {
          result = new ET(arg);
          leaf->elems[offset].store(result);
        }
      }
      legion_assert(result != 0);
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename ALLOCATOR>
    template<typename T1, typename T2>
    typename DynamicTable<ALLOCATOR>::ET* DynamicTable<ALLOCATOR>::lookup_entry(
        IT index, const T1& arg1, const T2& arg2)
    //-------------------------------------------------------------------------
    {
      NodeBase* n = lookup_leaf(index);
      // Now we've made it to the leaf node
      typename ALLOCATOR::LEAF_TYPE* leaf =
          static_cast<typename ALLOCATOR::LEAF_TYPE*>(n);
      int offset = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
      ET* result = leaf->elems[offset].load();
      if (result == nullptr)
      {
        AutoLock l(leaf->lock);
        // Now that we have the lock, check to see if we lost the race
        result = leaf->elems[offset].load();
        if (result == nullptr)
        {
          result = new ET(arg1, arg2);
          leaf->elems[offset].store(result);
        }
      }
      legion_assert(result != 0);
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
      NodeBase* n = root.load();
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
            NodeBase* parent =
                new_tree_node(parent_level, parent_first, parent_last);
            typename ALLOCATOR::INNER_TYPE* inner =
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
      legion_assert(
          (level_needed <= n->level) && (index >= n->first_index) &&
          (index <= n->last_index));
      // now walk the path, instantiating the path we need
      while (n->level > 0)
      {
        typename ALLOCATOR::INNER_TYPE* inner =
            static_cast<typename ALLOCATOR::INNER_TYPE*>(n);

        IT i =
            ((index >>
              (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
             ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
        legion_assert((i >= 0) && (((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE));
        NodeBase* child = inner->elems[i].load();
        if (child == nullptr)
        {
          AutoLock l(inner->lock);
          // Now that the lock is held, check to see if we lost the race
          child = inner->elems[i].load();
          if (child == nullptr)
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
        legion_assert(
            (child != 0) && (child->level == (n->level - 1)) &&
            (index >= child->first_index) && (index <= child->last_index));
        n = child;
      }
      legion_assert(n->level == 0);
      return n;
    }

  }  // namespace Internal
}  // namespace Legion
