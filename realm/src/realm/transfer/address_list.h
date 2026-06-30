/*
 * Copyright 2025 Los Alamos National Laboratory, Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef ADDRESS_LIST
#define ADDRESS_LIST

#include "realm/realm_config.h"
#include "realm/indexspace.h"
#include "realm/id.h"
#include "realm/deppart/sparsity_impl.h"

#include <array>
#include <cstring>
#include <unordered_map>

namespace Realm {

  template <typename FieldID>
  struct FieldBlockBase {
    std::size_t count;
    FieldID fields[1];

    // allocate a FieldBlock via heap.alloc_obj and store n field IDs
    template <typename Heap>
    static FieldBlockBase<FieldID> *create(Heap &heap, const FieldID *src, size_t n,
                                           size_t align = 16)
    {
      const size_t bytes = sizeof(FieldBlockBase<FieldID>) + (n - 1) * sizeof(FieldID);
      void *mem = heap.alloc_obj(bytes, align);
      FieldBlockBase<FieldID> *field_block = new(mem) FieldBlockBase<FieldID>;
      field_block->count = n;
      std::copy_n(src, n, field_block->fields);
      return field_block;
    }
  };

  using FieldBlock = FieldBlockBase<int>;

  // =================================================================================================
  //                                             AddressList
  // =================================================================================================
  class AddressList {
  public:
    AddressList(size_t _max_entries = 1000);

    // ─── entry construction ──────────────────────────────────────────────────────
    [[nodiscard]] bool
    append_entry(int dims, size_t contig_bytes, size_t total_bytes, size_t base_offset,
                 const std::unordered_map<int, std::pair<size_t, size_t>> &count_strides,
                 bool wrap_around = false);

    [[nodiscard]] size_t *begin_entry(int max_dim, bool wrap_around = true);
    void commit_entry(int act_dim, size_t bytes);
    void attach_field_block(const FieldBlock *_field_block);

    [[nodiscard]] size_t bytes_pending() const;
    [[nodiscard]] size_t full_field_bytes();

    // entry packs:
    // the contiguous byte count (contig_bytes) in the upper bitsthe
    // the actual dimension count (act_dim) in the lower 4 bits
    [[nodiscard]] static size_t pack_entry_header(size_t contig_bytes, int dims);

    // ─── layout constants ───────────────────────────────────────────────────────
    static constexpr size_t SLOT_HEADER = 0;
    static constexpr size_t SLOT_BASE = 1;
    static constexpr size_t DIM_SLOTS = 2;
    static constexpr size_t DIM_MASK = 0xF;
    static constexpr size_t CONTIG_SHIFT = 4;

  protected:
    friend class AddressListCursor;
    [[nodiscard]] const size_t *read_entry();

    const FieldBlock *field_block{nullptr};

    size_t total_bytes{0};
    size_t write_pointer{0};
    size_t read_pointer{0};
    size_t max_entries{0};
    std::vector<size_t> data;
  };

  // =================================================================================================
  //                                           AddressListCursor
  // =================================================================================================
  class AddressListCursor {
  public:
    AddressListCursor();

    void set_addrlist(AddressList *_addrlist);

    // ─── layout accessors ──────────────────────────────────────────────────────
    [[nodiscard]] int get_dim() const;
    [[nodiscard]] uintptr_t get_offset() const;
    [[nodiscard]] uintptr_t get_stride(int dim) const;
    [[nodiscard]] size_t remaining(int dim) const;

    // ─── progress───────────────────────────────────────────────────────────────
    void advance(int dim, size_t amount, int f = 1);
    void skip_bytes(size_t bytes);

    // ─── field accessors ──────────────────────────────────────────────────────
    [[nodiscard]] const FieldBlock *field_block() const;
    [[nodiscard]] const FieldID *fields_data() const;
    [[nodiscard]] size_t remaining_fields() const;

    AddressList *addrlist{nullptr};
    bool partial{false}; // inside a dimension

  protected:
    int partial_dim{0}; // dimension index
    size_t partial_fields{0};
    std::array<size_t, REALM_MAX_DIM + 1> pos{};
  };

  std::ostream &operator<<(std::ostream &os, const AddressListCursor &alc);
} // namespace Realm

#endif
