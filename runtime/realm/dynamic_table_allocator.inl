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

#include "realm/dynamic_table_allocator.h"

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
      leaf->elems[i].init(make_id(leaf->elems[0], owner, first_index + i), owner);
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
} // namespace Realm
