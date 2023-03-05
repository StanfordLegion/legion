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

// replicated (e.g. to gpus) internal heaps for Realm

#ifndef REALM_REPL_HEAP_H
#define REALM_REPL_HEAP_H

#include "realm/realm_config.h"
#include "realm/mutex.h"

#include <set>

namespace Realm {

  class ReplicatedHeap {
  public:
    ReplicatedHeap();
    ~ReplicatedHeap();

    void init(size_t _chunk_size, size_t _max_chunks);
    void cleanup();

    // objects may be allocated and freed
    void *alloc_obj(size_t bytes, size_t alignment);
    void free_obj(void *ptr);

    // writes to an object must be via the address at which is was allocated,
    //  and be followed by a call to commit_writes for the appropriate address
    //  range
    void commit_writes(void *start, size_t bytes);
  
    // heap listeners are told when new chunks are created and when data is
    //  updated (e.g. allowing software-managed coherency of mirrors of the
    //  heap)
    class Listener {
    public:
      virtual ~Listener() {}

      virtual void chunk_created(void *base, size_t bytes) {}
      virtual void chunk_destroyed(void *base, size_t bytes) {}

      virtual void data_updated(void *base, size_t bytes) {}
    };

    void add_listener(Listener *listener);
    void remove_listener(Listener *listener);

  protected:
    struct ObjectHeader {
      uint64_t state;
      static const uint64_t STATE_FREE = 0x0102030405060708ULL;
      static const uint64_t STATE_ALLOCD = 0x0F0E0D0C0B0A0908ULL;
      static const uint64_t STATE_INVALID = 0;
      uint64_t size;
    };

    Mutex mutex;
    uintptr_t base;
    size_t chunk_size, cur_chunks, max_chunks;
    size_t cur_bytes, peak_bytes;
    // fairly low-bandwidth allocator, so use simple best-fit algorithm
    std::map<uint64_t, uint64_t> free_by_start;
    std::multimap<uint64_t, uint64_t> free_by_size;
    std::set<Listener *> listeners;
  };

};

#endif
