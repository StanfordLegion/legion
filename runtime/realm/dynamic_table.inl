/* Copyright 2024 Stanford University, NVIDIA Corporation
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
  // class DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>
  //

  template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
  std::vector<typename DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::FreeList *> &
  DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::get_registered_freelists(
      Mutex *&lock)
  {
    static std::vector<FreeList *> registered_freelists;
    static Mutex registered_freelist_lock;
    lock = &registered_freelist_lock;
    return registered_freelists;
  }

  template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
  void DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::register_freelist(
      typename DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::FreeList *free_list)
  {
    Mutex *lock = nullptr;
    std::vector<FreeList *> &freelists = get_registered_freelists(lock);
    AutoLock<> al(*lock);
    freelists.push_back(free_list);
  }

  template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
  typename DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::ET *
  DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::steal_freelist_element(
      typename DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::FreeList *requestor)
  {
    // TODO: improve this by adjusting the starting offset to reduce contention
    Mutex *lock = nullptr;
    std::vector<FreeList *> &freelists = get_registered_freelists(lock);
    AutoLock<> al(*lock);
    for(FreeList *free_list : freelists) {
      if(free_list != requestor) {
        ET *elem = free_list->pop_front();
        if(elem != nullptr) {
          return elem;
        }
      }
    }
    return nullptr;
  }

  template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
  typename DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::LEAF_TYPE *
  DynamicTableAllocator<_ET, _INNER_BITS, _LEAF_BITS>::new_leaf_node(
      IT first_index, IT last_index, int owner, ET **free_list_head, ET **free_list_tail)

  {
    LEAF_TYPE *leaf = new LEAF_TYPE(0, first_index, last_index);
    const IT last_ofs = (((IT)1) << LEAF_BITS) - 1;

    for(IT i = 0; i <= last_ofs; i++)
      leaf->elems[i].init(ET::make_id(leaf->elems[0], owner, first_index + i), owner);
    // leaf->elems[i].init(ID(ET::ID_TYPE, owner, first_index +
    // i).convert<typeof(leaf->elems[0].me)>(), owner);

    if(free_list_head != nullptr && free_list_tail != nullptr) {
      // stitch all the new elements into the free list - we can do this
      //  with a single cmpxchg if we link up all of our new entries
      //  first

      // special case - if the first_index == 0, don't actually enqueue
      //  our first element, which would be global index 0
      const IT first_ofs = ((first_index > 0) ? 0 : 1);

      for(IT i = first_ofs; i < last_ofs; i++) {
        leaf->elems[i].next_free = &leaf->elems[i + 1];
      }

      // Push these new elements on the front of the free list we have so far
      leaf->elems[last_ofs].next_free = *free_list_head;
      *free_list_head = &leaf->elems[first_ofs];
      if(*free_list_tail == nullptr) {
        *free_list_tail = &leaf->elems[last_ofs];
      }
    }

    return leaf;
  }

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
    : root_and_level(0)
    , first_alloced_node(0)
  {}

  template <typename ALLOCATOR>
  DynamicTable<ALLOCATOR>::~DynamicTable(void)
  {
    // instead of a recursive deletion search from the root, we follow the
    //  list of nodes we allocated and delete directly
    NodeBase *to_delete = first_alloced_node.load();
    while(to_delete) {
      NodeBase *next = to_delete->next_alloced_node;
      delete to_delete;
      to_delete = next;
    }
  }

  template <typename ALLOCATOR>
  /*static*/ intptr_t DynamicTable<ALLOCATOR>::encode_root_and_level(NodeBase *root,
								     int level)
  {
#ifdef DEBUG_REALM
    assert(((reinterpret_cast<intptr_t>(root) & 7) == 0) &&
	   (level >= 0) && (level <= 7));
#endif
    return (reinterpret_cast<intptr_t>(root) | level);
  }

  template <typename ALLOCATOR>
  /*static*/ typename DynamicTable<ALLOCATOR>::NodeBase *DynamicTable<ALLOCATOR>::extract_root(intptr_t rlval)
  {
    return reinterpret_cast<NodeBase *>(rlval & ~intptr_t(7));
  }

  template <typename ALLOCATOR>
  /*static*/ int DynamicTable<ALLOCATOR>::extract_level(intptr_t rlval)
  {
    return (rlval & 7);
  }

  template <typename ALLOCATOR>
  void DynamicTable<ALLOCATOR>::prepend_alloced_node(NodeBase *new_node)
  {
    NodeBase *old_first = first_alloced_node.load();
    do {
      new_node->next_alloced_node = old_first;
    } while(!first_alloced_node.compare_exchange(old_first, new_node));
  }

  template <typename ALLOCATOR>
  typename DynamicTable<ALLOCATOR>::NodeBase *
  DynamicTable<ALLOCATOR>::new_tree_node(int level, IT first_index, IT last_index,
                                         int owner, ET **free_list_head,
                                         ET **free_list_tail)
  {
    if(level > 0) {
      // an inner node - we can create that ourselves
      typename ALLOCATOR::INNER_TYPE *inner =
          new typename ALLOCATOR::INNER_TYPE(level, first_index, last_index);
      for(size_t i = 0; i < ALLOCATOR::INNER_TYPE::SIZE; i++)
        inner->elems[i].store(0);
      return inner;
    } else {
      return ALLOCATOR::new_leaf_node(first_index, last_index, owner, free_list_head,
                                      free_list_tail);
    }
  }

  template<typename ALLOCATOR>
  size_t DynamicTable<ALLOCATOR>::max_entries(void) const
  {
    intptr_t rlval = root_and_level.load();
    if(rlval == 0)
      return 0;
    size_t elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
    elems_addressable <<= (ALLOCATOR::INNER_BITS *
			   extract_level(rlval));
    return elems_addressable;
  }

  template<typename ALLOCATOR>
  bool DynamicTable<ALLOCATOR>::has_entry(IT index) const
  {
    // first, figure out how many levels the tree must have to find our index
    int level_needed = 0;
    IT elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
    while(index >= elems_addressable) {
      level_needed++;
      // detect overflow
      IT new_elems = (elems_addressable << ALLOCATOR::INNER_BITS);
      if(new_elems < elems_addressable)
	break;
      elems_addressable = new_elems;
    }

    intptr_t rlval = root_and_level.load();
    if(rlval == 0)
      return false;  // empty tree
#ifdef DEBUG_REALM
    assert(extract_root(rlval)->level == extract_level(rlval));
#endif
    int n_level = extract_level(rlval);
    if(n_level < level_needed)
      return false;  // tree too short
    NodeBase *n = extract_root(rlval);

#ifdef DEBUG_REALM
    // when we get here, root is high enough
    assert((level_needed <= n->level) &&
	   (index >= n->first_index) &&
	   (index <= n->last_index));
#endif

    // now walk tree, populating the path we need
    while(n_level > 0) {
#ifdef DEBUG_REALM
      assert(n_level == n->level);
#endif
      // intermediate nodes
      typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(n);

      IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
	      ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
#ifdef DEBUG_REALM
      assert(((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE);
#endif

      NodeBase *child = inner->elems[i].load_acquire();
      if(child == 0) {
	return false;	
      }
#ifdef DEBUG_REALM
      assert((child != 0) &&
	     (child->level == (n_level - 1)) &&
	     (index >= child->first_index) &&
	     (index <= child->last_index));
#endif
      n = child;
      n_level--;
    }
    return true;
  }

  template <typename ALLOCATOR>
  typename DynamicTable<ALLOCATOR>::ET *DynamicTable<ALLOCATOR>::lookup_entry(IT index, int owner, ET **free_list_head /*= 0*/, ET **free_list_tail /*= 0*/)
  {
    // first, figure out how many levels the tree must have to find our index
    int level_needed = 0;
    IT elems_addressable = 1 << ALLOCATOR::LEAF_BITS;
    while(index >= elems_addressable) {
      level_needed++;
      // detect overflow
      IT new_elems = (elems_addressable << ALLOCATOR::INNER_BITS);
      if(new_elems < elems_addressable)
	break;
      elems_addressable = new_elems;
    }

    // in the common case, we won't need to add levels to the tree - grab the root (no lock)
    // and see if it covers the range that includes our index
    intptr_t rlval = root_and_level.load_acquire();
    NodeBase *n = extract_root(rlval);
    int n_level = extract_level(rlval);
#ifdef DEBUG_REALM
    assert(!n || (n_level == n->level));
#endif
    if(!n || (n_level < level_needed)) {
      // root doesn't appear to be high enough - take lock and fix it if it's really
      //  not high enough
      lock.lock();

      // reload the value from memory
      rlval = root_and_level.load();
      n = extract_root(rlval);
      n_level = extract_level(rlval);
      if(!n) {
	// simple case - just create a root node at the level we want
	n = new_tree_node(level_needed, 0, elems_addressable - 1, owner, free_list_head, free_list_tail);
	n_level = level_needed;
	root_and_level.store_release(encode_root_and_level(n, n_level));

	prepend_alloced_node(n);
      } else {
	// some of the tree already exists - add new layers on top
	while(n_level < level_needed) {
	  int parent_level = n_level + 1;
	  IT parent_first = 0;
	  IT parent_last = (((n->last_index + 1) << ALLOCATOR::INNER_BITS) - 1);
	  NodeBase *parent = new_tree_node(parent_level, parent_first, parent_last, owner, free_list_head, free_list_tail);
	  typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(parent);
	  inner->elems[0].store_release(n);
	  n = parent;
	  n_level = parent_level;
	  root_and_level.store_release(encode_root_and_level(n, n_level));

	  prepend_alloced_node(n);
	}
      }

      lock.unlock();
    }
#ifdef DEBUG_REALM
    // when we get here, root is high enough
    assert((level_needed <= n->level) &&
	   (index >= n->first_index) &&
	   (index <= n->last_index));
#endif

    // now walk tree, populating the path we need
    while(n_level > 0) {
#ifdef DEBUG_REALM
      assert(n_level == n->level);
#endif
      // intermediate nodes
      typename ALLOCATOR::INNER_TYPE *inner = static_cast<typename ALLOCATOR::INNER_TYPE *>(n);

      IT i = ((index >> (ALLOCATOR::LEAF_BITS + (n->level - 1) * ALLOCATOR::INNER_BITS)) &
	      ((((IT)1) << ALLOCATOR::INNER_BITS) - 1));
#ifdef DEBUG_REALM
      assert(((size_t)i) < ALLOCATOR::INNER_TYPE::SIZE);
#endif

      NodeBase *child = inner->elems[i].load_acquire();
      if(child == 0) {
	// need to populate subtree

	// take lock on inner node
	inner->lock.lock();

	// now that lock is held, see if we really need to make new node
	child = inner->elems[i].load_acquire();
	if(child == 0) {
	  int child_level = n_level - 1;
	  int child_shift = (ALLOCATOR::LEAF_BITS + child_level * ALLOCATOR::INNER_BITS);
	  IT child_first = inner->first_index + (i << child_shift);
	  IT child_last = inner->first_index + ((i + 1) << child_shift) - 1;

	  child = new_tree_node(child_level, child_first, child_last, owner, free_list_head, free_list_tail);
	  inner->elems[i].store_release(child);

	  prepend_alloced_node(child);
	}

	inner->lock.unlock();
      }
#ifdef DEBUG_REALM
      assert((child != 0) &&
	     (child->level == (n_level - 1)) &&
	     (index >= child->first_index) &&
	     (index <= child->last_index));
#endif
      n = child;
      n_level--;
    }

    // leaf node - just return pointer to the target element
    typename ALLOCATOR::LEAF_TYPE *leaf = static_cast<typename ALLOCATOR::LEAF_TYPE *>(n);
    IT ofs = (index & ((((IT)1) << ALLOCATOR::LEAF_BITS) - 1));
    return &(leaf->elems[ofs]);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DynamicTableFreeList<ALLOCATOR>
  //

  template <typename ALLOCATOR>
  DynamicTableFreeList<ALLOCATOR>::DynamicTableFreeList(DynamicTable<ALLOCATOR>& _table, int _owner, DynamicTableFreeList<ALLOCATOR> *_parent_list)
    : table(_table), parent_list(_parent_list), owner(_owner), first_free(0), next_alloc(0)
  {
    assert((parent_list == nullptr) || (parent_list->parent_list == nullptr));
    ALLOCATOR::register_freelist(this);
  }

  template<typename ALLOCATOR>
  void DynamicTableFreeList<ALLOCATOR>::push_front(DynamicTableFreeList<ALLOCATOR>::ET *entry)
  {
    assert(entry->next_free == nullptr);
    // no need for lock - use compare and swap to push item onto front of
    //  free list (no ABA problem because the popper is mutex'd)
    DynamicTableFreeList<ALLOCATOR>::ET *old_free = first_free.load_acquire();
    do {
      entry->next_free = old_free;
    } while (!first_free.compare_exchange(old_free, entry));
  }
  template<typename ALLOCATOR>
  void DynamicTableFreeList<ALLOCATOR>::push_front(DynamicTableFreeList<ALLOCATOR>::ET *head, DynamicTableFreeList<ALLOCATOR>::ET *tail)
  {
    // no need for lock - use compare and swap to push item onto front of
    //  free list (no ABA problem because the popper is mutex'd)
    ET *old_head = first_free.load_acquire();
    do {
      tail->next_free = old_head;
    } while (!first_free.compare_exchange(old_head, head));
  }

  template<typename ALLOCATOR>
  typename DynamicTableFreeList<ALLOCATOR>::ET *DynamicTableFreeList<ALLOCATOR>::pop_front_underlock(void)
  {
    // we are the only popper, but we need to use cmpxchg's to play nice
    // with pushers that don't take the lock
    DynamicTableFreeList<ALLOCATOR>::ET *old_first = first_free.load_acquire();
    while((old_first != nullptr) &&
          !first_free.compare_exchange(old_first, old_first->next_free))
      ;
    if(old_first != nullptr) {
      old_first->next_free = nullptr;
    }
    return old_first;
  }

  template<typename ALLOCATOR>
  typename DynamicTableFreeList<ALLOCATOR>::ET *DynamicTableFreeList<ALLOCATOR>::pop_front(void)
  {
    AutoLock<> al(lock);
    return pop_front_underlock();
  }

  template <typename ALLOCATOR>
  typename DynamicTableFreeList<ALLOCATOR>::ET *
  DynamicTableFreeList<ALLOCATOR>::alloc_entry(void)
  {
    while(true) {
      IT to_lookup;

      {
        // take the lock first, since we're messing with the free list
        AutoLock<> al(lock);
        ET *elem = pop_front_underlock();
        if(REALM_LIKELY(elem != nullptr)) {
          return elem;
        }

        // The free list is empty, we can fill it up by referencing the next entry to be
        // allocated - this uses the existing dynamic-filling code to avoid race
        // conditions

        if(parent_list != nullptr) {
          // We can reserve a region from the global list and use it to allocate
          // from next.  This ensures any child lists of the parent list will query unique
          // ranges of ids in order to reduce contention on the dynamic table
          ID::IDType end_id;
          parent_list->alloc_range(((IT)1) << ALLOCATOR::LEAF_BITS, next_alloc, end_id);
        }

        // list appears to be empty - drop the lock and do a lookup that has
        //  the likely side-effect of pushing a bunch of new entries onto
        //  the free list
        to_lookup = next_alloc;
        next_alloc += ((IT)1) << ALLOCATOR::LEAF_BITS;
      }

      ET *head = nullptr, *tail = nullptr;
      ET *dummy = table.lookup_entry(to_lookup, owner, &head, &tail);
      // Can't use dummy here since it may have been already allocated and used elsewhere.
      // Only the items pushed in the free list as a symptom of the lookup are freely
      // available.
      assert(dummy != 0);
      (void)dummy;
      // No one is using the returned list, so we can freely pop the head off and push the
      // rest onto this list for later
      if(REALM_LIKELY(head != nullptr)) {
        ET *rest = head->next_free;
        head->next_free = nullptr;
        if(REALM_LIKELY(rest != nullptr)) {
          push_front(rest, tail);
        }
        return head;
      }
      // We failed to retrieve a new element from the dynamic table.  This is usually due
      // to ID exhaustion, so lets try to steal an element from another registered list in
      // the allocator.  This increases contention on these free lists and can reduce
      // parallelism, so only do this as a last resort.
      // TODO: implement a watermark for lower memory foot-print?
      ET *elem = ALLOCATOR::steal_freelist_element(this);
      if(REALM_LIKELY(elem != nullptr)) {
        return elem;
      }
      // Unable to retrieve anything, try again
      lock.lock();
    }
  }

  template <typename ALLOCATOR>
  void DynamicTableFreeList<ALLOCATOR>::free_entry(ET *entry)
  {
    // TODO: potentially implement a mechanism to limit the size of the free list
    push_front(entry);
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
