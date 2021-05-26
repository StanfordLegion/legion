/* Copyright 2021 Stanford University, NVIDIA Corporation
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

#ifndef LOWLEVEL_DMA_H
#define LOWLEVEL_DMA_H

#include "realm/network.h"
#include "realm/id.h"
#include "realm/memory.h"
#include "realm/redop.h"
#include "realm/instance.h"
#include "realm/event.h"
#include "realm/runtime_impl.h"
#include "realm/inst_impl.h"
#include "realm/transfer/channel.h"
#include "realm/circ_queue.h"

namespace Realm {

  // keeps a per-memory queue of pending IB allocation requests
  class PendingIBQueue {
  public:
    // enqueues a request (or handles immediately, if possible)
    void enqueue_request(Memory tgt_mem, NodeID req_node, uintptr_t req_op,
			 unsigned ib_index, size_t size);

    // attempts to dequeue pending requests for the specified memory
    void dequeue_request(Memory tgt_mem);

  protected:
    Mutex queue_mutex;
    struct IBAllocRequest {
      NodeID req_node;
      uintptr_t req_op;
      unsigned ib_index;
      size_t size;
    };
    struct PerMemory {
      int deq_count;  // count of pending dequeuers
      CircularQueue<IBAllocRequest, 16> requests;
    };
    std::map<Memory, PerMemory> queues;
  };

  extern PendingIBQueue ib_req_queue;

  struct RemoteIBAllocRequest {
    Memory memory;
    size_t size;
    uintptr_t req_op;
    unsigned ib_index;
    
    static void handle_message(NodeID sender, const RemoteIBAllocRequest &args,
			       const void *data, size_t msglen);
  };

  struct RemoteIBAllocResponse {
    uintptr_t req_op;
    unsigned ib_index;
    off_t offset;

    static void handle_message(NodeID sender, const RemoteIBAllocResponse &args,
			       const void *data, size_t msglen);
  };

    struct RemoteIBFreeRequestAsync {
      Memory memory;
      off_t ib_offset;
      size_t ib_size;

      static void handle_message(NodeID sender, const RemoteIBFreeRequestAsync &args,
				 const void *data, size_t msglen);
    };

    extern void init_dma_handler(void);

    extern void start_dma_system(BackgroundWorkManager *bgwork);

    extern void stop_dma_system(void);

    struct MemPathInfo {
      std::vector<Memory> path;
      std::vector<Channel *> xd_channels;
      //std::vector<XferDesKind> xd_kinds;
      //std::vector<NodeID> xd_target_nodes;
    };
    
    bool find_shortest_path(Memory src_mem, Memory dst_mem,
			    CustomSerdezID serdez_id,
                            ReductionOpID redop_id,
			    MemPathInfo& info,
			    bool skip_final_memcpy = false);

    void free_intermediate_buffer(Memory mem, off_t offset, size_t size);

    class AsyncFileIOContext : public BackgroundWorkItem {
    public:
      AsyncFileIOContext(int _max_depth);
      ~AsyncFileIOContext(void);

      void enqueue_write(int fd, size_t offset, size_t bytes, const void *buffer, Request* req = NULL);
      void enqueue_read(int fd, size_t offset, size_t bytes, void *buffer, Request* req = NULL);
      void enqueue_fence(Operation *req);

      bool empty(void);
      long available(void);

      static AsyncFileIOContext* get_singleton(void);

      virtual bool do_work(TimeLimit work_until);

      class AIOOperation {
      public:
	virtual ~AIOOperation(void) {}
	virtual void launch(void) = 0;
	virtual bool check_completion(void) = 0;
	bool completed;
        void* req;
      };

    protected:
      void make_progress(void);

      int max_depth;
      std::deque<AIOOperation *> launched_operations, pending_operations;
      Mutex mutex;
#ifdef REALM_USE_KERNEL_AIO
      aio_context_t aio_ctx;
#endif
    };

  class WrappingFIFOIterator : public TransferIterator {
  public:
    WrappingFIFOIterator(size_t _base, size_t _size);

    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);
      
    virtual void reset(void);
    virtual bool done(void);

    virtual size_t step(size_t max_bytes, AddressInfo& info,
			unsigned flags,
			bool tentative = false);
    virtual size_t step_custom(size_t max_bytes, AddressInfoCustom& info,
                               bool tentative = false);
    virtual void confirm_step(void);
    virtual void cancel_step(void);

    virtual bool get_addresses(AddressList &addrlist);

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, WrappingFIFOIterator> serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    size_t base, size, offset, prev_offset;
    bool tentative_valid;
  };

};

#endif
