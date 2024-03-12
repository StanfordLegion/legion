/* Copyright 2024 Stanford University
 * Copyright 2024 Los Alamos National Laboratory
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

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressList
  //

  AddressList::AddressList()
    : total_bytes(0)
    , write_pointer(0)
    , read_pointer(0)
  {}

  size_t *AddressList::begin_nd_entry(int max_dim)
  {
    size_t entries_needed = max_dim * 2;

    size_t new_wp = write_pointer + entries_needed;
    if(new_wp > MAX_ENTRIES) {
      // have to wrap around
      if((read_pointer <= entries_needed) || (write_pointer < read_pointer))
        return 0;

      // fill remaining entries with 0's so reader skips over them
      while(write_pointer < MAX_ENTRIES)
        data[write_pointer++] = 0;

      write_pointer = 0;
    } else {
      // if the write pointer would cross over the read pointer, we have to wait
      if((write_pointer < read_pointer) && (new_wp >= read_pointer))
        return 0;

      // special case: if the write pointer would wrap and read is at 0, that'd
      //  be a collision too
      if((new_wp == MAX_ENTRIES) && (read_pointer == 0))
        return 0;
    }

    // all good - return a pointer to the first available entry
    return (data + write_pointer);
  }

  void AddressList::commit_nd_entry(int act_dim, size_t bytes)
  {
    size_t entries_used = act_dim * 2;

    write_pointer += entries_used;
    if(write_pointer >= MAX_ENTRIES) {
      assert(write_pointer == MAX_ENTRIES);
      write_pointer = 0;
    }

    total_bytes += bytes;
  }

  size_t AddressList::bytes_pending() const { return total_bytes; }

  const size_t *AddressList::read_entry()
  {
    assert(total_bytes > 0);
    if(read_pointer >= MAX_ENTRIES) {
      assert(read_pointer == MAX_ENTRIES);
      read_pointer = 0;
    }
    // skip trailing 0's
    if(data[read_pointer] == 0)
      read_pointer = 0;
    return (data + read_pointer);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressListCursor
  //

  AddressListCursor::AddressListCursor()
    : addrlist(0)
    , partial(false)
  {
    for(int i = 0; i < MAX_DIM; i++)
      pos[i] = 0;
  }

  void AddressListCursor::set_addrlist(AddressList *_addrlist) { addrlist = _addrlist; }

  int AddressListCursor::get_dim() const
  {
    assert(addrlist);
    // with partial progress, we restrict ourselves to just the rest of that dim
    if(partial) {
      return (partial_dim + 1);
    } else {
      const size_t *entry = addrlist->read_entry();
      int act_dim = (entry[0] & 15);
      return act_dim;
    }
  }

  uintptr_t AddressListCursor::get_offset() const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    uintptr_t ofs = entry[1];
    if(partial) {
      for(int i = partial_dim; i < act_dim; i++)
        if(i == 0) {
          // dim 0 is counted in bytes
          ofs += pos[0];
        } else {
          // rest use the strides from the address list
          ofs += pos[i] * entry[1 + (2 * i)];
        }
    }
    return ofs;
  }

  uintptr_t AddressListCursor::get_stride(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    assert((dim > 0) && (dim < act_dim));
    return entry[2 * dim + 1];
  }

  size_t AddressListCursor::remaining(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    assert(dim < act_dim);
    size_t r = entry[2 * dim];
    if(dim == 0)
      r >>= 4;
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

  void AddressListCursor::advance(int dim, size_t amount)
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & 15);
    assert(dim < act_dim);
    size_t r = entry[2 * dim];
    if(dim == 0)
      r >>= 4;

    size_t bytes = amount;
    if(dim > 0) {
#ifdef DEBUG_REALM
      for(int i = 0; i < dim; i++)
        assert(pos[i] == 0);
#endif
      bytes *= (entry[0] >> 4);
      for(int i = 1; i < dim; i++)
        bytes *= entry[2 * i];
    }
#ifdef DEBUG_REALM
    assert(addrlist->total_bytes >= bytes);
#endif
    addrlist->total_bytes -= bytes;

    if(!partial) {
      if((dim == (act_dim - 1)) && (amount == r)) {
        // simple case - we consumed the whole thing
        addrlist->read_pointer += 2 * act_dim;
        return;
      } else {
        // record partial consumption
        partial = true;
        partial_dim = dim;
        pos[partial_dim] = amount;
      }
    } else {
      // update a partial consumption in progress
      assert(dim <= partial_dim);
      partial_dim = dim;
      pos[partial_dim] += amount;
    }

    while(pos[partial_dim] == r) {
      pos[partial_dim++] = 0;
      if(partial_dim == act_dim) {
        // all done
        partial = false;
        addrlist->read_pointer += 2 * act_dim;
        break;
      } else {
        pos[partial_dim]++;         // carry into next dimension
        r = entry[2 * partial_dim]; // no shift because partial_dim > 0
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
