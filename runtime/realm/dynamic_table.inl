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

// INCLDUED FROM dynamic_table.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/dynamic_table.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class DynamicTableNodeBase<LT, IT>
  //

  template <typename LT, typename IT>
  DynamicTableNodeBase<LT, IT>::DynamicTableNodeBase(int _level, IT _first_index, IT _last_index)
    : level(_level), first_index(_first_index), last_index(_last_index)
    , next_alloced_node(0)
  {}

  template <typename LT, typename IT>
  DynamicTableNodeBase<LT, IT>::~DynamicTableNodeBase(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class DynamicTableNode<ET, SIZE, LT,  IT>
  //

  template <typename ET, size_t _SIZE, typename LT, typename IT>
  DynamicTableNode<ET, _SIZE, LT, IT>::DynamicTableNode(int _level, IT _first_index, IT _last_index)
    : DynamicTableNodeBase<LT, IT>(_level, _first_index, _last_index)
  {}

  template <typename ET, size_t _SIZE, typename LT, typename IT>
  DynamicTableNode<ET, _SIZE, LT, IT>::~DynamicTableNode(void)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class DynamicTable<ALLOCATOR>
  //

  template <typename ALLOCATOR>
  DynamicTable<ALLOCATOR>::DynamicTable(void)
    : root(0)
    , first_alloced_node(0)
  {}

  template <typename ALLOCATOR>
  DynamicTable<ALLOCATOR>::~DynamicTable(void)
  {
    // instead of a recursive deletion search from the root, we follow the
    //  list of nodes we allocated and delete directly
    NodeBase *to_delete = first_alloced_node;
    while(to_delete) {
      NodeBase *next = to_delete->next_alloced_node;
      delete to_delete;
      to_delete = next;
    }
  }

  template <typename ALLOCATOR>
  typename DynamicTable<ALLOCATOR>::NodeBase *DynamicTable<ALLOCATOR>::new_tree_node(int level, IT first_index, IT last_index, int owner, typename ALLOCATOR::FreeList *free_list /*= 0*/)
  {
    if(level > 0) {
      // an inner node - we can create that ourselves
      typename ALLOCATOR::INNER_TYPE *inner = new typename ALLOCATOR::INNER_TYPE(level, first_index, last_index);
      for(size_t i = 0; i < ALLOCATOR::INNER_TYPE::SIZE; i++)
	inner->elems[i] = 0;
      return inner;
    } else {
      return ALLOCATOR::new_leaf_node(first_index, last_index, owner, free_list);
    }
  }

  template<typename ALLOCATOR>
  size_t DynamicTable<ALLOCATOR>::max_entries(void) const
  {
    if (!root)
      return 0;
    size_t elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
    for (int i = 0; i < root->level; i++)
      elems_addressable <<= ALLOCATOR::INNER_BITS;
    return elems_addressable;
  }

  template<typename ALLOCATOR>
  bool DynamicTable<ALLOCATOR>::has_entry(IT index) const
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

    // when we get here, root is high enough
    assert((level_needed <= n->level) &&
	   (index >= n->first_index) &&
	   (index <= n->last_index));

    // now walk tree, populating the path we need
    while(n->level > 0) {
      // intermediate nodes
      typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(n);

      IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
	      ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
      assert((i >= 0) && (((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE));

      NodeBase *child = inner->elems[i];
      if(child == 0) {
	return false;	
      }
      assert((child != 0) &&
	     (child->level == (n->level - 1)) &&
	     (index >= child->first_index) &&
	     (index <= child->last_index));
      n = child;
    }
    return true;
  }

  template <typename ALLOCATOR>
  typename DynamicTable<ALLOCATOR>::ET *DynamicTable<ALLOCATOR>::lookup_entry(IT index, int owner, typename ALLOCATOR::FreeList *free_list /*= 0*/)
  {
    // first, figure out how many levels the tree must have to find our index
    int level_needed = 0;
    int elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
    while(index >= elems_addressable) {
      level_needed++;
      elems_addressable <<= ALLOCATOR::INNER_BITS;
    }

    // in the common case, we won't need to add levels to the tree - grab the root (no lock)
    // and see if it covers the range that includes our index
    NodeBase *n = root;
    if(!n || (n->level < level_needed)) {
      // root doesn't appear to be high enough - take lock and fix it if it's really
      //  not high enough
      lock.lock();

      if(!root) {
	// simple case - just create a root node at the level we want
	root = new_tree_node(level_needed, 0, elems_addressable - 1, owner, free_list);
	// we're always first to add a node, so no race conditions here
	bool ok = __sync_bool_compare_and_swap(&first_alloced_node,
					       0,
					       root);
	assert(ok);
      } else {
	// some of the tree already exists - add new layers on top
	while(root->level < level_needed) {
	  int parent_level = root->level + 1;
	  IT parent_first = 0;
	  IT parent_last = (((root->last_index + 1) << ALLOCATOR::INNER_BITS) - 1);
	  NodeBase *parent = new_tree_node(parent_level, parent_first, parent_last, owner, free_list);
	  typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(parent);
	  inner->elems[0] = root;
	  root = parent;
	  // this is not synchronized against threads that might be adding
	  //  interior/leaf nodes, so CAS loop is required
	  while(true) {
	    NodeBase *cur_first = first_alloced_node;
	    parent->next_alloced_node = cur_first;
	    if(__sync_bool_compare_and_swap(&first_alloced_node,
					    cur_first,
					    parent)) break;
	  }
	}
      }
      n = root;

      lock.unlock();
    }
    // when we get here, root is high enough
    assert((level_needed <= n->level) &&
	   (index >= n->first_index) &&
	   (index <= n->last_index));

    // now walk tree, populating the path we need
    while(n->level > 0) {
      // intermediate nodes
      typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(n);

      IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
	      ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
      assert((i >= 0) && (((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE));

      NodeBase *child = inner->elems[i];
      if(child == 0) {
	// need to populate subtree

	// take lock on inner node
	inner->lock.lock();

	// now that lock is held, see if we really need to make new node
	child = inner->elems[i];
	if(child == 0) {
	  int child_level = inner->level - 1;
	  int child_shift = (ALLOCATOR::LEAF_BITS + child_level * ALLOCATOR::INNER_BITS);
	  IT child_first = inner->first_index + (i << child_shift);
	  IT child_last = inner->first_index + ((i + 1) << child_shift) - 1;

	  child = new_tree_node(child_level, child_first, child_last, owner, free_list);
	  inner->elems[i] = child;
	  // this is not synchronized against threads that might be adding
	  //  parent or other interior/leaf nodes, so CAS loop is required
	  while(true) {
	    NodeBase *cur_first = first_alloced_node;
	    child->next_alloced_node = cur_first;
	    if(__sync_bool_compare_and_swap(&first_alloced_node,
					    cur_first,
					    child)) break;
	  }
	}

	inner->lock.unlock();
      }
      assert((child != 0) &&
	     (child->level == (n->level - 1)) &&
	     (index >= child->first_index) &&
	     (index <= child->last_index));
      n = child;
    }

    // leaf node - just return pointer to the target element
    typename ALLOCATOR::LEAF_TYPE *leaf = static_cast<typename ALLOCATOR::LEAF_TYPE *>(n);
    int ofs = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
    return &(leaf->elems[ofs]);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DynamicTableFreeList<ALLOCATOR>
  //

  template <typename ALLOCATOR>
  DynamicTableFreeList<ALLOCATOR>::DynamicTableFreeList(DynamicTable<ALLOCATOR>& _table, int _owner)
    : table(_table), owner(_owner), first_free(0), next_alloc(0)
  {}

  template <typename ALLOCATOR>
  typename DynamicTableFreeList<ALLOCATOR>::ET *DynamicTableFreeList<ALLOCATOR>::alloc_entry(void)
  {
    // take the lock first, since we're messing with the free list
    lock.lock();

    // if the free list is empty, we can fill it up by referencing the next entry to be allocated -
    // this uses the existing dynamic-filling code to avoid race conditions
    while(!first_free) {
      IT to_lookup = next_alloc;
      next_alloc += ((IT)1) << ALLOCATOR::LEAF_BITS; // do this before letting go of lock
      lock.unlock();
#ifndef NDEBUG
      typename DynamicTable<ALLOCATOR>::ET *dummy =
#endif
        table.lookup_entry(to_lookup, owner, this);
      assert(dummy != 0);
      // can't actually use dummy because we let go of lock - retake lock and hopefully find non-empty
      //  list next time
      lock.lock();
    }

    typename DynamicTable<ALLOCATOR>::ET *entry = first_free;
    first_free = entry->next_free;
    lock.unlock();

    return entry;
  }

  template <typename ALLOCATOR>
  void DynamicTableFreeList<ALLOCATOR>::free_entry(ET *entry)
  {
    // just stick ourselves on front of free list
    lock.lock();
    entry->next_free = first_free;
    first_free = entry;
    lock.unlock();
  }

  // allocates a range of IDs that can be given to a remote node for remote allocation
  // these entries do not go on the local free list unless they are deleted after being used
  template <typename ALLOCATOR>
  void DynamicTableFreeList<ALLOCATOR>::alloc_range(int requested, IT& first_id, IT& last_id)
  {
    // to avoid interactions with the local allocator, we must always assign a multiple of 2^LEAF_BITS
    //  ids
    if((requested & ((1 << ALLOCATOR::LEAF_BITS) - 1)) != 0)
      requested = ((requested >> ALLOCATOR::LEAF_BITS) + 1) << ALLOCATOR::LEAF_BITS;

    // take lock and bump next_alloc
    lock.lock();
    first_id = next_alloc;
    next_alloc += requested;
    lock.unlock();

    last_id = first_id + (requested - 1);
  }

}; // namespace Realm
