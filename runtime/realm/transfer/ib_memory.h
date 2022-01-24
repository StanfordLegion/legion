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

// IB (Intermediate Buffer) Memory implementations for Realm

#ifndef REALM_IB_MEMORY_H
#define REALM_IB_MEMORY_H

#include "realm/realm_config.h"

#include "realm/mem_impl.h"

namespace Realm {

  // a simple memory used for intermediate buffers in dma system
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE IBMemory : public MemoryImpl {
  public:
    IBMemory(Memory _me, size_t _size,
	     MemoryKind _kind, Memory::Kind _lowlevel_kind,
	     void *prealloc_base, NetworkSegment *_segment);

    virtual ~IBMemory();

    // old-style allocation used by IB memories
    virtual off_t alloc_bytes_local(size_t size);
    virtual void free_bytes_local(off_t offset, size_t size);

    virtual void *get_direct_ptr(off_t offset, size_t size);

    // not used by IB memories
    virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
							bool need_alloc_result,
							bool poisoned,
							TimeLimit work_until);

    virtual void release_storage_immediate(RegionInstanceImpl *inst,
					   bool poisoned,
					   TimeLimit work_until);

    virtual void get_bytes(off_t offset, void *dst, size_t size);
    virtual void put_bytes(off_t offset, const void *src, size_t size);

    // attempts to allocate one or more IBs - either all succeed or all fail
    bool attempt_immediate_allocation(NodeID requestor, uintptr_t req_op,
                                      size_t count, const size_t *sizes,
                                      off_t *offsets);

    // enqueues a batch of PendingIBRequests to be satisfied as soon as possible
    void enqueue_requests(PendingIBRequests *reqs);

    void free_multiple(size_t count,
                       const off_t *offsets, const size_t *sizes);

  protected:
    // these must be called with the mutex held
    off_t do_alloc(size_t size);
    void do_free(off_t offset, size_t size);
    PendingIBRequests *satisfy_pending_reqs();
    void forward_satisfied_reqs(PendingIBRequests *reqs);

    Mutex mutex; // protection for resizing vectors
    std::map<off_t, off_t> free_blocks;
    char *base;
    PendingIBRequests *ibreq_head;
    PendingIBRequests **ibreq_tail;
  };

  // helper routine to free IB whether it is local or remote
  void free_intermediate_buffer(Memory mem, off_t offset, size_t size);

  // active messages related to IB allocation/release

  struct RemoteIBAllocRequestSingle {
    Memory memory;
    size_t size;
    uintptr_t req_op;
    unsigned req_index;
    bool immediate;

    static void handle_message(NodeID sender,
                               const RemoteIBAllocRequestSingle &args,
                               const void *data, size_t msglen);
  };

  struct RemoteIBAllocRequestMultiple {
    NodeID requestor;
    unsigned count, first_index, curr_index;
    uintptr_t req_op;
    bool immediate;

    static void handle_message(NodeID sender,
                               const RemoteIBAllocRequestMultiple &args,
                               const void *data, size_t msglen);
  };

  struct RemoteIBAllocResponseSingle {
    uintptr_t req_op;
    unsigned req_index;
    off_t offset;

    static void handle_message(NodeID sender,
                               const RemoteIBAllocResponseSingle &args,
                               const void *data, size_t msglen);
  };


  struct RemoteIBAllocResponseMultiple {
    uintptr_t req_op;
    unsigned count, first_index;

    static void handle_message(NodeID sender,
                               const RemoteIBAllocResponseMultiple &args,
                               const void *data, size_t msglen);
  };

  struct RemoteIBReleaseSingle {
    Memory memory;
    size_t size;
    off_t offset;

    static void handle_message(NodeID sender,
                               const RemoteIBReleaseSingle &args,
                               const void *data, size_t msglen);
  };

  struct RemoteIBReleaseMultiple {
    unsigned count;

    static void handle_message(NodeID sender,
                               const RemoteIBReleaseMultiple &args,
                               const void *data, size_t msglen);
  };

}; // namespace Realm

#endif
