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

#include "realm/realm_config.h"

#ifdef REALM_ON_WINDOWS
#define NOMINMAX
#endif

#include "realm/transfer/address_list.h"
#include "realm/utils.h"

namespace Realm {

  // helpers
  // -----------------------------------------------------------------------------------------
  namespace detail {
    inline size_t contig_bytes(const size_t *e)
    {
      return ((e[AddressList::SLOT_HEADER] >> AddressList::CONTIG_SHIFT));
    }

    inline int actdim(const size_t *e)
    {
      return int(e[AddressList::SLOT_HEADER] & AddressList::DIM_MASK);
    }

    inline size_t count_index(int dim) { return AddressList::DIM_SLOTS * dim; }

    inline size_t stride_index(int dim) { return count_index(dim) + 1; }
  } // namespace detail

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressList
  //

  AddressList::AddressList(size_t _max_entries)
    : max_entries(_max_entries)
  {
    data.reserve(max_entries);
  }

  bool AddressList::append_entry(
      int dims, size_t contig_bytes, size_t total_bytes, size_t base_offset,
      const std::unordered_map<int, std::pair<size_t, size_t>> &count_strides,
      bool wrap_around)
  {
    size_t *entry = begin_entry(dims, wrap_around);

    if(entry == nullptr) {
      return false;
    }

    entry[AddressList::SLOT_BASE] = base_offset;

    for(auto &[dim, count_stride] : count_strides) {
      entry[detail::count_index(dim)] = count_stride.first;
      entry[detail::stride_index(dim)] = count_stride.second;
    }

    entry[AddressList::SLOT_HEADER] = pack_entry_header(contig_bytes, dims);
    commit_entry(dims, total_bytes);
    return true;
  }

  size_t *AddressList::begin_entry(int max_dim, bool wrap_mode)
  {
    size_t entries_needed = detail::count_index(max_dim);

    if(wrap_mode) {
      size_t new_wp = write_pointer + entries_needed;
      if(new_wp > max_entries) {
        if((read_pointer <= entries_needed) || (write_pointer < read_pointer))
          return nullptr;

        // fill remaining entries with 0's so reader skips
        while(write_pointer < max_entries)
          data[write_pointer++] = 0;

        write_pointer = 0;
        new_wp = entries_needed;
      } else {
        if((write_pointer < read_pointer) && (new_wp >= read_pointer))
          return nullptr;
        if((new_wp == max_entries) && (read_pointer == 0))
          return nullptr;
      }

      // ensure capacity upfront for max_entries once
      if(data.size() < max_entries)
        data.resize(max_entries);

      return data.data() + write_pointer;
    } else {
      if(data.size() < write_pointer + entries_needed)
        data.resize(write_pointer + entries_needed);
      return data.data() + write_pointer;
    }
  }

  void AddressList::commit_entry(int act_dim, size_t bytes)
  {
    size_t entries_used = detail::count_index(act_dim);
    // catch callers whose act_dim exceeds the max_dim they reserved with
    // begin_entry — otherwise the overshoot corrupts adjacent slots (and the
    // heap if it crosses data.size()) and only surfaces later as a stale
    // header in read_entry
    assert(write_pointer + entries_used <= data.size());
    write_pointer += entries_used;
    total_bytes += bytes * (field_block ? field_block->count : 1);
  }

  void AddressList::attach_field_block(const FieldBlock *_field_block)
  {
    field_block = _field_block;
  }

  size_t AddressList::bytes_pending() const { return total_bytes; }

  size_t AddressList::full_field_bytes()
  {
    const size_t *entry = read_entry();
    // decode header
    const size_t contig = detail::contig_bytes(entry);
    const int dims = detail::actdim(entry);
    size_t bytes = contig;
    for(int d = 1; d < dims; d++) {
      bytes *= entry[detail::count_index(d)];
    }
    return bytes;
  }

  size_t AddressList::pack_entry_header(size_t contig_bytes, int dims)
  {
    return (contig_bytes << CONTIG_SHIFT) | (dims & DIM_MASK);
  }

  const size_t *AddressList::read_entry()
  {
    // assert(total_bytes > 0);
    if(read_pointer >= max_entries) {
      assert(read_pointer == max_entries);
      read_pointer = 0;
    }

    // skip trailing 0's
    if(data[read_pointer] == 0)
      read_pointer = 0;
    return (data.data() + read_pointer);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressListCursor
  //

  AddressListCursor::AddressListCursor() { pos.fill(0); }

  void AddressListCursor::set_addrlist(AddressList *_addrlist) { addrlist = _addrlist; }

  int AddressListCursor::get_dim() const
  {
    assert(addrlist);
    // with partial progress, we restrict ourselves to just the rest of that dim
    if(partial) {
      return (partial_dim + 1);
    } else {
      return detail::actdim(addrlist->read_entry());
    }
  }

  uintptr_t AddressListCursor::get_offset() const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = detail::actdim(entry);
    uintptr_t ofs = entry[AddressList::SLOT_BASE];
    if(partial) {
      for(int i = partial_dim; i < act_dim; i++)
        if(i == 0) {
          // dim 0 is counted in bytes
          ofs += pos[0];
        } else {
          // rest use the strides from the address list
          ofs += pos[i] * entry[detail::stride_index(i)];
        }
    }
    return ofs;
  }

  uintptr_t AddressListCursor::get_stride(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = detail::actdim(entry);
    assert((dim > 0) && (dim < act_dim));
    return entry[detail::stride_index(dim)];
  }

  size_t AddressListCursor::remaining(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = detail::actdim(entry);
    assert(dim < act_dim);
    size_t r = entry[detail::count_index(dim)];

    if(dim == 0) {
      r >>= AddressList::CONTIG_SHIFT;
    }

    if(partial) {
      if(dim > partial_dim)
        r = 1;
      if(dim == partial_dim) {
        assert(r > pos[dim]);
        r -= pos[dim];
      }
    }
    return r;
  }

  void AddressListCursor::advance(int dim, size_t amount, int f)
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = detail::actdim(entry);
    assert(dim < act_dim);

    // size of this "slice" in dim
    size_t r = entry[detail::count_index(dim)];
    if(dim == 0) {
      r >>= AddressList::CONTIG_SHIFT;
    }

    // compute how many bytes we're really removing
    size_t bytes = amount;
    if(dim > 0) {
#ifdef DEBUG_REALM
      for(int i = 0; i < dim; i++)
        assert(pos[i] == 0);
#endif
      bytes *= detail::contig_bytes(entry);
      for(int i = 1; i < dim; i++)
        bytes *= entry[detail::count_index(i)];
    }

#ifdef DEBUG_REALM
    assert(addrlist->total_bytes >= bytes * f);
#endif
    addrlist->total_bytes -= bytes * f;

    const FieldBlock *fields = field_block();

    // ——— NEW: if this call exactly finishes *one* rect (the last dim)
    if(dim == (act_dim - 1) && amount == r) {
      if(fields && f > 0) {
        // bump fields only on a full-rect consume
        partial_fields += f;
        if(partial_fields >= fields->count) {
          partial_fields = 0;
          addrlist->read_pointer += detail::count_index(act_dim);
        }
      } else {
        // no fields at all: consume entry immediately
        addrlist->read_pointer += detail::count_index(act_dim);
      }
      // reset any in-flight partial state
      partial = false;
      partial_dim = 0;
      pos.fill(0);
      return;
    }

    // ——— otherwise fall back to the existing "partial" logic
    if(!partial) {
      partial = true;
      partial_dim = dim;
      pos[dim] = amount;
    } else {
      assert(dim <= partial_dim);
      partial_dim = dim;
      pos[dim] += amount;
    }

    while(pos[partial_dim] == r) {
      pos[partial_dim++] = 0;

      if(partial_dim == act_dim) {
        // we have finished the rect described by this entry
        partial = false;

        if(fields && (f > 0)) {
          partial_fields += f;
          if(partial_fields >= fields->count) {
            partial_fields = 0;
            addrlist->read_pointer += detail::count_index(act_dim);
          }
        } else {
          addrlist->read_pointer += detail::count_index(act_dim);
        }
        break;
      } else {
        // carry into the next higher dimension
        pos[partial_dim]++; // increment that dimension
        r = entry[detail::count_index(partial_dim)];
      }
    }
  }

  void AddressListCursor::skip_bytes(size_t bytes)
  {
    while(bytes > 0) {
      int act_dim = get_dim();

      if(act_dim == 0) {
        assert(0);
      } else {
        size_t chunk = remaining(0);
        if(chunk <= bytes) {
          int dim = 0;
          size_t count = chunk;
          while((dim + 1) < act_dim) {
            dim++;
            count = bytes / chunk;
            assert(count > 0);
            size_t r = remaining(dim + 1);
            if(count < r) {
              chunk *= count;
              break;
            } else {
              count = r;
              chunk *= count;
            }
          }
          advance(dim, count);
          bytes -= chunk;
        } else {
          advance(0, bytes);
          return;
        }
      }
    }
  }

  const FieldBlock *AddressListCursor::field_block() const
  {
    return addrlist->field_block;
  }

  const FieldID *AddressListCursor::fields_data() const
  {
    return addrlist->field_block->fields + partial_fields;
  }

  size_t AddressListCursor::remaining_fields() const
  {
    if(addrlist->field_block) {
      return addrlist->field_block->count - partial_fields;
    }
    return 1;
  }

  std::ostream &operator<<(std::ostream &os, const AddressListCursor &alc)
  {
    os << alc.remaining(0);
    for(int i = 1; i < alc.get_dim(); i++)
      os << 'x' << alc.remaining(i);
    os << ',' << alc.get_offset();
    for(int i = 1; i < alc.get_dim(); i++)
      os << '+' << alc.get_stride(i);
    return os;
  }
} // namespace Realm
