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

// dynamic node set implementation for Realm

#ifndef REALM_NODESET_H
#define REALM_NODESET_H

#include "realm/realm_config.h"
#include "realm/mutex.h"

#include <stdint.h>
#include <iterator>

#ifdef DEBUG_REALM
//define DEBUG_REALM_NODESET
#endif

#ifdef DEBUG_REALM_NODESET
#include <set>
#endif

namespace Realm {

  // TODO: optimize for fairly-common case of 'short' being sufficient?
  typedef int NodeID;

  class NodeSet;

  // we do not support mutation of a nodeset, so we're a const_iterator
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE NodeSetIterator {
  public:
    // explicitly set iterator traits
    typedef std::input_iterator_tag iterator_category;
    typedef NodeID value_type;
    typedef std::ptrdiff_t difference_type;
    typedef NodeID *pointer;
    typedef NodeID& reference;

    NodeSetIterator();
    NodeSetIterator(const NodeSet& _nodeset);

    bool operator==(const NodeSetIterator& compare_to) const;
    bool operator!=(const NodeSetIterator& compare_to) const;

    NodeID operator*() const;
    const NodeID *operator->() const;

    NodeSetIterator& operator++(/*prefix*/);
    NodeSetIterator operator++(int /*postfix*/);

  protected:
    const NodeSet *nodeset;
    NodeID cur_node;
    short iter_pos; // needed for non-bitmask encodings
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE NodeSetBitmask {
  protected:
    NodeSetBitmask();

  public:
    static void configure_allocator(NodeID _max_node_id,
				    size_t _bitsets_per_chunk,
				    bool _use_twolevel);
    static void free_allocations();

    static NodeSetBitmask *acquire_bitmask();
    static NodeSetBitmask *clone_bitmask(const NodeSetBitmask *clone_from);

    static void release_bitmask(NodeSetBitmask *bitmask, bool already_empty);

    size_t set_bit(NodeID id);
    size_t clear_bit(NodeID id);

    size_t set_range(NodeID lo, NodeID hi);
    size_t clear_range(NodeID lo, NodeID hi);

    bool is_set(NodeID id) const;

    NodeID first_set() const;
    NodeID next_set(NodeID after) const;

    void copy(const NodeSetBitmask *copy_from);

  protected:
    void l2_set(int elmt_idx);
    void l2_clear(int elmt_idx);
    int l2_find(int first_idx) const;

    typedef uint64_t bitmask_elem_t;

    static const size_t BITS_PER_ELEM = 8 * sizeof(bitmask_elem_t);

    bitmask_elem_t bits[1];

    static NodeID max_node_id;
    static size_t bitset_elements, bitsets_per_chunk;
    static size_t bitset_twolevel;
    static uintptr_t alloc_chain_head, free_list_head;
    static Mutex free_list_mutex;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE NodeSet {
  public:
    NodeSet();
    ~NodeSet();

    NodeSet(const NodeSet& copy_from);

    NodeSet& operator=(const NodeSet& copy_from);

    void swap(NodeSet& swap_with);

    bool empty() const;
    size_t size() const;

    void add(NodeID id);
    void remove(NodeID id);

    void add_range(NodeID lo, NodeID hi);
    void remove_range(NodeID lo, NodeID hi);

    void clear();

    bool contains(NodeID id) const;

    typedef NodeSetIterator const_iterator;

    const_iterator begin() const;
    const_iterator end() const;

  protected:
    friend class NodeSetIterator;

    unsigned count;

    enum {
      ENC_EMPTY,
      ENC_VALS, // one or more distinct values
      ENC_RANGES, // one or more non-overlapping ranges
      ENC_BITMASK, // full (externally-allocated) bitmask
    };
    unsigned char enc_format;
    short range_count;
    static const short MAX_VALUES = 4;
    static const short MAX_RANGES = 2;
    union EncodingUnion {
      NodeID values[MAX_VALUES];
      struct { NodeID lo,hi; } ranges[MAX_RANGES];
      NodeSetBitmask *bitmask;
    };
    EncodingUnion data;
#ifdef DEBUG_REALM_NODESET
    std::set<NodeID> reference_set;
#endif

    void convert_to_bitmask();
  };

};

#include "realm/nodeset.inl"

#endif // ifndef REALM_NODESET_H
