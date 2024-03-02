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

// replicated (e.g. to gpus) internal heaps for Realm

#include "realm/repl_heap.h"

#include "realm/logging.h"

namespace Realm {

  Logger log_rheap("replheap");

  ////////////////////////////////////////////////////////////////////////
  //
  // class ReplicatedHeap
  //

  ReplicatedHeap::ReplicatedHeap()
    : base(0)
    , chunk_size(0)
    , cur_chunks(0)
    , max_chunks(0)
    , cur_bytes(0)
    , peak_bytes(0)
  {}

  ReplicatedHeap::~ReplicatedHeap()
  {
    if(base)
      log_rheap.warning() << "replheap distroyed without being cleaned up!";
  }

#ifdef DEBUG_REALM
  static void sanity_check_metadata(size_t cur_bytes, size_t total_bytes,
                                    const std::map<uint64_t, uint64_t>& by_start,
                                    const std::multimap<uint64_t, uint64_t>& by_size)
  {
    size_t free_bytes = 0;
    assert(by_start.size() == by_size.size());
    for(std::multimap<uint64_t, uint64_t>::const_iterator it = by_size.begin();
        it != by_size.end();
        ++it) {
      std::map<uint64_t, uint64_t>::const_iterator it2 = by_start.find(it->second);
      assert(it2 != by_start.end());
      assert(it2->second == it->first);
      free_bytes += it->first;
    }
    assert((cur_bytes + free_bytes) == total_bytes);
  }
#endif

  void ReplicatedHeap::init(size_t _chunk_size, size_t _max_chunks)
  {
    AutoLock<> al(mutex);

    // can only init once
    assert(base == 0);
    chunk_size = _chunk_size;
    cur_chunks = 1;
    max_chunks = _max_chunks;
    cur_bytes = 0;
    peak_bytes = 0;

    // allocate page-ish aligned memory
    static const size_t ALIGNMENT = 4096;
    void *ptr;
#ifdef REALM_ON_WINDOWS
    ptr = _aligned_malloc(chunk_size, ALIGNMENT);
    assert(ptr != 0);
#else
    int ret = posix_memalign(&ptr, ALIGNMENT, chunk_size);
    assert(ret == 0);
#endif
    base = reinterpret_cast<uintptr_t>(ptr);

    ObjectHeader hdr;
    hdr.state = ObjectHeader::STATE_FREE;
    hdr.size = chunk_size;
    memcpy(ptr, &hdr, sizeof(ObjectHeader));

    free_by_start.insert(std::pair<uint64_t, uint64_t>(0, chunk_size));
    free_by_size.insert(std::pair<uint64_t, uint64_t>(chunk_size, 0));

#ifdef DEBUG_REALM
    sanity_check_metadata(cur_bytes, cur_chunks * chunk_size,
                          free_by_start, free_by_size);
#endif

    // listeners may have already been added - if so, notify them
    for(std::set<Listener *>::const_iterator it = listeners.begin();
        it != listeners.end();
        ++it)
      (*it)->chunk_created(reinterpret_cast<void *>(base), chunk_size);
  }

  void ReplicatedHeap::cleanup()
  {
    AutoLock<> al(mutex);

    // notify any listeners before we actually nuke the memory
    for(std::set<Listener *>::const_iterator it = listeners.begin();
        it != listeners.end();
        ++it)
      for(size_t i = 0; i < cur_chunks; i++)
        (*it)->chunk_destroyed(reinterpret_cast<void *>(base +
                                                        (i * chunk_size)),
                               chunk_size);

    void *ptr = reinterpret_cast<void *>(base);
#ifdef REALM_ON_WINDOWS
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    log_rheap.info() << "peak replheap usage: " << peak_bytes << " bytes";
    base = 0;
    cur_chunks = 0;
    max_chunks = 0;
  }

  // objects may be allocated and freed
  void *ReplicatedHeap::alloc_obj(size_t bytes, size_t alignment)
  {
    AutoLock<> al(mutex);

    // round up bytes to a multiple of ObjectHeader size and add 1 more
    bytes = sizeof(ObjectHeader) * (((bytes - 1) / sizeof(ObjectHeader)) + 2);

    // find the smallest free range that fits our allocation (i.e. "best fit")
    std::multimap<uint64_t, uint64_t>::iterator it = free_by_size.lower_bound(bytes);
    if(it != free_by_size.end()) {
      uint64_t pos = it->second;
      
      ObjectHeader hdr;
      memcpy(&hdr, reinterpret_cast<const void *>(base + pos),
	     sizeof(ObjectHeader));
      assert((hdr.state == ObjectHeader::STATE_FREE) &&
             (hdr.size == it->first));

      // this range is no longer free
      free_by_size.erase(it);
#ifdef DEBUG_REALM
      assert(free_by_start.count(pos) == 1);
#endif
      free_by_start.erase(pos);

      // leftover piece?
      if(hdr.size > bytes) {
        ObjectHeader hdr2;
        hdr2.state = ObjectHeader::STATE_FREE;
        hdr2.size = hdr.size - bytes;
        memcpy(reinterpret_cast<void *>(base + pos + bytes), &hdr2,
               sizeof(ObjectHeader));
        hdr.size = bytes; // trim current block

        free_by_start.insert(std::pair<uint64_t, uint64_t>(pos + bytes,
                                                           hdr2.size));
        free_by_size.insert(std::pair<uint64_t, uint64_t>(hdr2.size,
                                                          pos + bytes));
      }

      // mark current block allocated
      hdr.state = ObjectHeader::STATE_ALLOCD;
      memcpy(reinterpret_cast<void *>(base + pos), &hdr,
             sizeof(ObjectHeader));

      cur_bytes += bytes;
      if(cur_bytes > peak_bytes)
        peak_bytes = cur_bytes;

      log_rheap.debug() << "allocated: bytes=" << bytes << " addr="
                        << std::hex << (base + pos + sizeof(ObjectHeader))
                        << std::dec;

#ifdef DEBUG_REALM
      sanity_check_metadata(cur_bytes, cur_chunks * chunk_size,
                            free_by_start, free_by_size);
#endif

      return reinterpret_cast<void *>(base + pos + sizeof(ObjectHeader));
    } else {
      // TODO: dynamically create chunks if permitted

      // out of space - estimate what would have been needed by assuming the
      //  largest available range would have needed to grow to fit
      uint64_t needed = (cur_chunks * chunk_size) + bytes;
      if(!free_by_size.empty())
        needed -= free_by_size.rbegin()->first;

      log_rheap.fatal() << "FATAL: replicated heap exhausted, grow with -ll:replheap - at least " << needed << " bytes required";
      abort();
      return 0;
    }
  }

  void ReplicatedHeap::free_obj(void *ptr)
  {
    AutoLock<> al(mutex);

    // some frees might come in after we've cleaned up the heap - ignore those
    if(base == 0) return;

    uint64_t pos = reinterpret_cast<uintptr_t>(ptr) - base;
    assert((pos >= sizeof(ObjectHeader)) &&
           (pos < (cur_chunks * chunk_size)) &&
           ((pos & (sizeof(ObjectHeader) - 1)) == 0));
    pos -= sizeof(ObjectHeader);
    ObjectHeader hdr;
    memcpy(&hdr, reinterpret_cast<const void *>(base + pos),
           sizeof(ObjectHeader));
    assert(hdr.state == ObjectHeader::STATE_ALLOCD);
    uint64_t bytes = hdr.size;

    log_rheap.debug() << "freed: bytes=" << bytes << " addr="
                      << std::hex << (base + pos + sizeof(ObjectHeader))
                      << std::dec;
    cur_bytes -= bytes;

    // is there a free block past us?
    std::map<uint64_t, uint64_t>::iterator it = free_by_start.find(pos + bytes);
    if(it != free_by_start.end()) {
      ObjectHeader hdr2;
      memcpy(&hdr2, reinterpret_cast<const void *>(base + pos + bytes),
             sizeof(ObjectHeader));
      assert((hdr2.state == ObjectHeader::STATE_FREE) &&
             (hdr2.size == it->second));

      // remove later block from metadata and invalidate header
      hdr2.state = ObjectHeader::STATE_INVALID;
      memcpy(reinterpret_cast<void *>(base + pos + bytes), &hdr2,
             sizeof(ObjectHeader));
      free_by_start.erase(it);
      std::multimap<uint64_t, uint64_t>::iterator it2 = free_by_size.lower_bound(hdr2.size);
      while(true) {
        assert(it2 != free_by_size.end());
        if(it2->second == pos + bytes) {
          free_by_size.erase(it2);
          break;
        }
        ++it2;
      }

      bytes += hdr2.size;
    }

    // is there a free block before us?
    it = free_by_start.lower_bound(pos);
    if(it != free_by_start.begin()) {
      --it;
      if((it->first + it->second) == pos) {
        // yes, merge with that metadata and invalidate our header
        ObjectHeader hdr3;
        memcpy(&hdr3, reinterpret_cast<const void *>(base + it->first),
               sizeof(ObjectHeader));
        assert((hdr3.state == ObjectHeader::STATE_FREE) &&
               (hdr3.size == it->second));
        hdr3.size += bytes;
        memcpy(reinterpret_cast<void *>(base + it->first), &hdr3,
               sizeof(ObjectHeader));

        hdr.state = ObjectHeader::STATE_INVALID;
        memcpy(reinterpret_cast<void *>(base + pos), &hdr,
               sizeof(ObjectHeader));

        std::multimap<uint64_t, uint64_t>::iterator it2 = free_by_size.lower_bound(it->second);
        while(true) {
          assert(it2 != free_by_size.end());
          if(it2->second == it->first) {
            free_by_size.erase(it2);
            break;
          }
          ++it2;
        }
        it->second = hdr3.size;
        free_by_size.insert(std::pair<uint64_t, uint64_t>(it->second,
                                                          it->first));

#ifdef DEBUG_REALM
        sanity_check_metadata(cur_bytes, cur_chunks * chunk_size,
                              free_by_start, free_by_size);
#endif

        return;
      }
    }

    // mark our block free and add it to the free list
    hdr.state = ObjectHeader::STATE_FREE;
    hdr.size = bytes;
    memcpy(reinterpret_cast<void *>(base + pos), &hdr,
           sizeof(ObjectHeader));
    free_by_start.insert(std::pair<uint64_t, uint64_t>(pos, bytes));
    free_by_size.insert(std::pair<uint64_t, uint64_t>(bytes, pos));

#ifdef DEBUG_REALM
    sanity_check_metadata(cur_bytes, cur_chunks * chunk_size,
                          free_by_start, free_by_size);
#endif
  }

  // writes to an object must be via the address at which is was allocated,
  //  and be followed by a call to commit_writes for the appropriate address
  //  range
  void ReplicatedHeap::commit_writes(void *start, size_t bytes)
  {
    AutoLock<> al(mutex);

    for(std::set<Listener *>::const_iterator it = listeners.begin();
        it != listeners.end();
        ++it)
      (*it)->data_updated(start, bytes);
  }
  
  void ReplicatedHeap::add_listener(Listener *listener)
  {
    log_rheap.debug() << "adding listener: " << listener;

    size_t count;
    {
      AutoLock<> al(mutex);

      listeners.insert(listener);
      // remember how many chunks existed at the time the listener was added
      count = cur_chunks;
    }

    // now we can call the 'chunk_created' callback without holding the mutex
    for(size_t i = 0; i < count; i++)
      listener->chunk_created(reinterpret_cast<void *>(base +
						       (i * chunk_size)),
			      chunk_size);
  }


  void ReplicatedHeap::remove_listener(Listener *listener)
  {
    log_rheap.debug() << "removing listener: " << listener;

    size_t count;
    {
      AutoLock<> al(mutex);

      listeners.erase(listener);
      // remember how many chunks existed at the time the listener was removed
      count = cur_chunks;
    }

    // now we can call the 'chunk_destroyed' callback without holding the mutex
    for(size_t i = 0; i < count; i++)
      listener->chunk_destroyed(reinterpret_cast<void *>(base +
							 (i * chunk_size)),
				chunk_size);
  }

};
