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

#ifndef __LEGION_DYNAMIC_TABLE_H__
#define __LEGION_DYNAMIC_TABLE_H__

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Dynamic Table
    /////////////////////////////////////////////////////////////
    template<typename IT>
    struct DynamicTableNodeBase {
    public:
      DynamicTableNodeBase(int _level, IT _first_index, IT _last_index)
        : level(_level), first_index(_first_index), last_index(_last_index)
      { }
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
        for (size_t i = 0; i < SIZE; i++) elems[i].store(nullptr);
      }
      DynamicTableNode(const DynamicTableNode& rhs) = delete;
      virtual ~DynamicTableNode(void)
      {
        for (size_t i = 0; i < SIZE; i++)
        {
          ET* elem = elems[i].load();
          if (elem != nullptr)
            delete elem;
        }
      }
    public:
      DynamicTableNode& operator=(const DynamicTableNode& rhs) = delete;
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
        for (size_t i = 0; i < SIZE; i++) elems[i].store(nullptr);
      }
      LeafTableNode(const LeafTableNode& rhs) = delete;
      virtual ~LeafTableNode(void)
      {
        for (size_t i = 0; i < SIZE; i++)
        {
          ET* elem = elems[i].load();
          if (elem != nullptr)
            delete elem;
        }
      }
    public:
      LeafTableNode& operator=(const LeafTableNode& rhs) = delete;
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
      DynamicTable(const DynamicTable& rhs) = delete;
      ~DynamicTable(void);
    public:
      DynamicTable& operator=(const DynamicTable& rhs) = delete;
    public:
      size_t max_entries(void) const;
      bool has_entry(IT index) const;
      ET* lookup_entry(IT index);
      template<typename T>
      ET* lookup_entry(IT index, const T& arg);
      template<typename T1, typename T2>
      ET* lookup_entry(IT index, const T1& arg1, const T2& arg2);
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
      typedef DynamicTableNode<DynamicTableNodeBase<IT>, 1 << INNER_BITS, IT>
          INNER_TYPE;
      typedef LeafTableNode<ET, 1 << LEAF_BITS, IT> LEAF_TYPE;

      static LEAF_TYPE* new_leaf_node(IT first_index, IT last_index)
      {
        return new LEAF_TYPE(0 /*level*/, first_index, last_index);
      }
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/utilities/dynamic_table.inl"

#endif  // __LEGION_DYNAMIC_TABLE_H__
