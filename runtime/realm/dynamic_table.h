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

// monotonic dynamic lookup table for Realm

#ifndef REALM_DYNAMIC_TABLE_H
#define REALM_DYNAMIC_TABLE_H

#include "realm/atomics.h"
#include "realm/id.h"
#include "realm/mutex.h"

namespace Realm {

    // we have a base type that's element-type agnostic
    template <typename LT, typename IT>
    struct DynamicTableNodeBase {
    public:
      DynamicTableNodeBase(int _level, IT _first_index, IT _last_index);
      virtual ~DynamicTableNodeBase(void);

      int level;
      IT first_index, last_index;
      LT lock;
      // all nodes in a table are linked in a list for destruction
      DynamicTableNodeBase<LT,IT> *next_alloced_node;
    };

    template <typename ET, size_t _SIZE, typename LT, typename IT>
      struct DynamicTableNode : public DynamicTableNodeBase<LT, IT> {
    public:
      static const size_t SIZE = _SIZE;

      DynamicTableNode(int _level, IT _first_index, IT _last_index);
      virtual ~DynamicTableNode(void);

      ET elems[SIZE];
    };

    template <typename ALLOCATOR> class DynamicTableFreeList;

    template <typename ALLOCATOR>
    class DynamicTable {
    public:
      typedef typename ALLOCATOR::IT IT;
      typedef typename ALLOCATOR::ET ET;
      typedef typename ALLOCATOR::LT LT;
      typedef DynamicTableNodeBase<LT, IT> NodeBase;

      DynamicTable(void);
      ~DynamicTable(void);

      size_t max_entries(void) const;
      bool has_entry(IT index) const;
      ET *lookup_entry(IT index, int owner, ET **free_list_head = 0, ET **free_list_tail = 0);

    protected:
      NodeBase *new_tree_node(int level, IT first_index, IT last_index,
			      int owner, ET **free_list_head, ET **free_list_tail);

      // lock protects _changes_ to 'root', but not access to it
      LT lock;
      // encode level of root directly in value - saves an extra memory load
      //  per level
      atomic<intptr_t> root_and_level;
      static intptr_t encode_root_and_level(NodeBase *root, int level);
      static NodeBase *extract_root(intptr_t rlval);
      static int extract_level(intptr_t rlval);
      
      // all nodes in a table are linked in a list for destruction
      atomic<NodeBase *> first_alloced_node;
      void prepend_alloced_node(NodeBase *new_node);
    };

    template <typename ALLOCATOR>
    class DynamicTableFreeList {
    public:
      typedef typename ALLOCATOR::IT IT;
      typedef typename ALLOCATOR::ET ET;
      typedef typename ALLOCATOR::LT LT;

      DynamicTableFreeList(DynamicTable<ALLOCATOR>& _table, int _owner, DynamicTableFreeList<ALLOCATOR> *_parent_list = nullptr);

      void push_front(ET *entry);
      void push_front(ET *head, ET *tail);
      ET *pop_front_underlock(void);
      ET *pop_front(void);

      ET *alloc_entry(void);
      void free_entry(ET *entry);

      // allocates a range of IDs that can be given to a remote node for remote allocation
      // these entries do not go on the local free list unless they are deleted after being used
      void alloc_range(int requested, IT& first_id, IT& last_id);

      DynamicTable<ALLOCATOR>& table;
      // Free list from which we will coordinate reservation of IDs from
      DynamicTableFreeList<ALLOCATOR> *parent_list;
      int owner;
      LT lock;
      atomic<ET *> first_free;
      IT next_alloc;
    };

    template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
    class DynamicTableAllocator {
    public:
      typedef _ET ET;
      static const size_t INNER_BITS = _INNER_BITS;
      static const size_t LEAF_BITS = _LEAF_BITS;

      typedef Mutex LT;
      typedef ID::IDType IT;
      typedef DynamicTableNode<atomic<DynamicTableNodeBase<LT, IT> *>, 1 << INNER_BITS,
                               LT, IT>
          INNER_TYPE;
      typedef DynamicTableNode<ET, 1 << LEAF_BITS, LT, IT> LEAF_TYPE;
      typedef DynamicTableFreeList<DynamicTableAllocator<ET, _INNER_BITS, _LEAF_BITS>>
          FreeList;

      static std::vector<FreeList *> &get_registered_freelists(Mutex *&lock);

      static void register_freelist(FreeList *free_list);

      static ET *steal_freelist_element(FreeList *requestor = nullptr);

      static LEAF_TYPE *new_leaf_node(IT first_index, IT last_index, int owner,
                                      ET **free_list_head, ET **free_list_tail);
    };

  }; // namespace Realm

#include "realm/dynamic_table.inl"

#endif // ifndef REALM_DYNAMIC_TABLE_H

