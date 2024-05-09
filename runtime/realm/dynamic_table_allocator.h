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

#ifndef REALM_DYNAMIC_TABLE_ALLOCATOR_H
#define REALM_DYNAMIC_TABLE_ALLOCATOR_H

#include "realm/id.h"
#include "realm/mutex.h"

namespace Realm {
  class ProcessorGroupImpl;
  class MemoryImpl;
  class IBMemory;
  class ProcessorImpl;
  class RegionInstanceImpl;
  class NetworkSegment;

  template <typename _ET, size_t _INNER_BITS, size_t _LEAF_BITS>
  class DynamicTableAllocator {
  public:
    typedef _ET ET;
    static const size_t INNER_BITS = _INNER_BITS;
    static const size_t LEAF_BITS = _LEAF_BITS;

    typedef Mutex LT;
    typedef ID::IDType IT;
    typedef DynamicTableNode<atomic<DynamicTableNodeBase<LT, IT> *>, 1 << INNER_BITS, LT,
                             IT>
        INNER_TYPE;
    typedef DynamicTableNode<ET, 1 << LEAF_BITS, LT, IT> LEAF_TYPE;
    typedef DynamicTableFreeList<DynamicTableAllocator<ET, _INNER_BITS, _LEAF_BITS>>
        FreeList;

    template <typename T>
    static ID make_id(const T &dummy, int owner, IT index)
    {
      ID id;
      id.id = index;
      return id;
    }

    // hack for now - these should be factored out
    static ID make_id(const GenEventImpl &dummy, int owner, IT index)
    {
      return ID::make_event(owner, index, 0);
    }
    static ID make_id(const BarrierImpl &dummy, int owner, IT index)
    {
      return ID::make_barrier(owner, index, 0);
    }
    static Reservation make_id(const ReservationImpl &dummy, int owner, IT index)
    {
      return ID::make_reservation(owner, index).convert<Reservation>();
    }
    static Processor make_id(const ProcessorGroupImpl &dummy, int owner, IT index)
    {
      return ID::make_procgroup(owner, 0, index).convert<Processor>();
    }
    static ID make_id(const SparsityMapImplWrapper &dummy, int owner, IT index)
    {
      return ID::make_sparsity(owner, 0, index);
    }
    static CompletionQueue make_id(const CompQueueImpl &dummy, int owner, IT index)
    {
      return ID::make_compqueue(owner, index).convert<CompletionQueue>();
    }
    static ID make_id(const SubgraphImpl &dummy, int owner, IT index)
    {
      return ID::make_subgraph(owner, 0, index);
    }

    static std::vector<FreeList *> &get_registered_freelists(Mutex *&lock);

    static void register_freelist(FreeList *free_list);

    static ET *steal_freelist_element(FreeList *requestor = nullptr);

    static LEAF_TYPE *new_leaf_node(IT first_index, IT last_index, int owner,
                                    ET **free_list_head, ET **free_list_tail);
  };
} // namespace Realm

#include "realm/dynamic_table_allocator.inl"

#endif // REALM_DYNAMIC_TABLE_ALLOCATOR_H
