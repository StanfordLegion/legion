/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "realm/activemsg.h"
#include "realm/id.h"
#include "realm/memory.h"
#include "realm/redop.h"
#include "realm/instance.h"
#include "realm/event.h"
#include "realm/runtime_impl.h"
#include "realm/inst_impl.h"

namespace Realm {
  class CoreReservationSet;

    struct RemoteIBAllocRequestAsync {
      struct RequestArgs {
        int node;
        Memory memory;
        void* req;
        int idx;
        ID::IDType src_inst_id, dst_inst_id;
        size_t size;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<REMOTE_IB_ALLOC_REQUEST_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, Memory tgt_mem, void* req,
                               int idx, ID::IDType src_id, ID::IDType dst_id, size_t size);
    };

    struct RemoteIBAllocResponseAsync {
      struct RequestArgs {
        void* req;
        int idx;
        ID::IDType src_inst_id, dst_inst_id;
        size_t size;
        off_t offset;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<REMOTE_IB_ALLOC_RESPONSE_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, void* req, int idx, ID::IDType src_id,
                               ID::IDType dst_id, size_t ib_size, off_t ib_offset);
    };

    struct RemoteIBFreeRequestAsync {
      struct RequestArgs {
        Memory memory;
        off_t ib_offset;
        size_t ib_size;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<REMOTE_IB_FREE_REQUEST_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(NodeID target, Memory tgt_mem,
                               off_t ib_offset, size_t ib_size);
    };

    void find_shortest_path(Memory src_mem, Memory dst_mem,
			    CustomSerdezID serdez_id, std::vector<Memory>& path);

    struct RemoteCopyArgs : public BaseMedium {
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy, after_copy;
      int priority;
    };

    struct RemoteFillArgs : public BaseMedium {
      RegionInstance inst;
      FieldID field_id;
      unsigned size;
      Event before_fill, after_fill;
      //int priority;
    };

    extern void handle_remote_copy(RemoteCopyArgs args, const void *data, size_t msglen);

    extern void handle_remote_fill(RemoteFillArgs args, const void *data, size_t msglen);

    typedef ActiveMessageMediumNoReply<REMOTE_COPY_MSGID,
				       RemoteCopyArgs,
				       handle_remote_copy> RemoteCopyMessage;

    typedef ActiveMessageMediumNoReply<REMOTE_FILL_MSGID,
                                       RemoteFillArgs,
                                       handle_remote_fill> RemoteFillMessage;

    extern void init_dma_handler(void);

    extern void start_dma_worker_threads(int count, Realm::CoreReservationSet& crs);
    extern void stop_dma_worker_threads(void);

    extern void start_dma_system(int count, bool pinned, int max_nr, Realm::CoreReservationSet& crs);

    extern void stop_dma_system(void);

    /*
    extern Event enqueue_dma(IndexSpace idx,
			     RegionInstance src, 
			     RegionInstance target,
			     size_t elmt_size,
			     size_t bytes_to_copy,
			     Event before_copy,
			     Event after_copy = Event::NO_EVENT);
    */

    // helper methods used in other places
    static inline off_t calc_mem_loc(off_t alloc_offset, off_t field_start, int field_size, size_t elmt_size,
				     size_t block_size, off_t index)
    {
      return (alloc_offset +                                      // start address
	      ((index / block_size) * block_size * elmt_size) +   // full blocks
	      (field_start * block_size) +                        // skip other fields
	      ((index % block_size) * field_size));               // some some of our fields within our block
    }

    void find_field_start(const std::vector<size_t>& field_sizes, off_t byte_offset,
			  size_t size, off_t& field_start, int& field_size);
    
    class DmaRequestQueue;
    // for now we use a single queue for all (local) dmas
    extern DmaRequestQueue *dma_queue;
    
    typedef unsigned long long XferDesID;
    class DmaRequest : public Realm::Operation {
    public:
      DmaRequest(int _priority, Event _after_copy);

      DmaRequest(int _priority, Event _after_copy,
                 const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~DmaRequest(void);

    public:
      virtual void print(std::ostream& os) const;

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq) = 0;

      virtual bool handler_safe(void) = 0;

      virtual void perform_dma(void) = 0;

      enum State {
	STATE_INIT,
	STATE_METADATA_FETCH,
	STATE_DST_FETCH,
	STATE_GEN_PATH,
	STATE_ALLOC_IB,
	STATE_BEFORE_EVENT,
	STATE_INST_LOCK,
	STATE_READY,
	STATE_QUEUED,
	STATE_DONE
      };

      State state;
      int priority;
      // <NEWDMA>
      pthread_mutex_t request_lock;
      std::vector<XferDesID> path;
      std::set<XferDesID> complete_xd;

      // Returns true if all xfer des of this DmaRequest
      // have been marked completed
      // This return val is a signal for delete this DmaRequest
      bool notify_xfer_des_completion(XferDesID guid)
      {
        pthread_mutex_lock(&request_lock);
        complete_xd.insert(guid);
        bool all_completed = (complete_xd.size() == path.size());
        pthread_mutex_unlock(&request_lock);
        return all_completed;
      }
      Event tgt_fetch_completion;
      // </NEWDMA>

      class Waiter : public EventWaiter {
      public:
        Waiter(void);
        virtual ~Waiter(void);
      public:
	Reservation current_lock;
	DmaRequestQueue *queue;
	DmaRequest *req;

	void sleep_on_event(Event e, Reservation l = Reservation::NO_RESERVATION);

	virtual bool event_triggered(Event e, bool poisoned);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;
      };
    };

    void free_intermediate_buffer(DmaRequest* req, Memory mem, off_t offset, size_t size);

    struct OffsetsAndSize {
      FieldID src_field_id, dst_field_id;
      off_t src_subfield_offset, dst_subfield_offset;
      int size;
      CustomSerdezID serdez_id;
    };
    typedef std::vector<OffsetsAndSize> OASVec;
    typedef std::pair<Memory, Memory> MemPair;
    typedef std::pair<RegionInstance, RegionInstance> InstPair;
    typedef std::map<InstPair, OASVec> OASByInst;
    typedef std::map<MemPair, OASByInst *> OASByMem;

    class MemPairCopier;

    struct PendingIBInfo {
      Memory memory;
      int idx;
      InstPair ip;
    };

    class ComparePendingIBInfo {
    public:
      bool operator() (const PendingIBInfo& a, const PendingIBInfo& b) {
        if (a.memory.id == b.memory.id) {
          assert(a.idx != b.idx);
          return a.idx < b.idx;
        }
        else
          return a.memory.id < b.memory.id;
      }
    };

    struct IBInfo {
      enum Status {
        INIT,
        SENT,
        COMPLETED
      };
      Memory memory;
      off_t offset;
      size_t size;
      Status status;
      //IBFence* fence;
      Event event;
    };

    typedef std::set<PendingIBInfo, ComparePendingIBInfo> PriorityIBQueue;
    typedef std::vector<IBInfo> IBVec;
    typedef std::map<InstPair, IBVec> IBByInst;

    class TransferDomain;
    class TransferIterator;

    // dma requests come in two flavors:
    // 1) CopyRequests, which are per memory pair, and
    // 2) ReduceRequests, which have to be handled monolithically

    class CopyRequest : public DmaRequest {
    public:
      CopyRequest(const void *data, size_t datalen,
		  Event _before_copy,
		  Event _after_copy,
		  int _priority);

      CopyRequest(const TransferDomain *_domain, //const Domain& _domain,
		  OASByInst *_oas_by_inst,
		  Event _before_copy,
		  Event _after_copy,
		  int _priority,
                  const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~CopyRequest(void);

    public:
      void forward_request(NodeID target_node);

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq);

      void perform_new_dma(Memory src_mem, Memory dst_mem);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      TransferDomain *domain;
      //Domain domain;
      OASByInst *oas_by_inst;

      // <NEW_DMA>
      void alloc_intermediate_buffer(InstPair inst_pair, Memory tgt_mem, int idx);

      void handle_ib_response(int idx, InstPair inst_pair, size_t ib_size, off_t ib_offset);

      PriorityIBQueue priority_ib_queue;
      // operations on ib_by_inst are protected by ib_mutex
      IBByInst ib_by_inst;
      GASNetHSL ib_mutex;
      class IBAllocOp : public Realm::Operation {
      public:
        IBAllocOp(Event _completion) : Operation(_completion, Realm::ProfilingRequestSet()) {};
        ~IBAllocOp() {};
        void print(std::ostream& os) const {os << "IBAllocOp"; };
      };

      std::vector<Memory> mem_path;
      // </NEW_DMA>

      Event before_copy;
      Waiter waiter; // if we need to wait on events
    };

    class ReduceRequest : public DmaRequest {
    public:
      ReduceRequest(const void *data, size_t datalen,
		    ReductionOpID _redop_id,
		    bool _red_fold,
		    Event _before_copy,
		    Event _after_copy,
		    int _priority);

      ReduceRequest(const TransferDomain *_domain, //const Domain& _domain,
		    const std::vector<CopySrcDstField>& _srcs,
		    const CopySrcDstField& _dst,
		    bool _inst_lock_needed,
		    ReductionOpID _redop_id,
		    bool _red_fold,
		    Event _before_copy,
		    Event _after_copy,
		    int _priority,
                    const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~ReduceRequest(void);

    public:
      void forward_request(NodeID target_node);

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      TransferDomain *domain;
      //Domain domain;
      std::vector<CopySrcDstField> srcs;
      CopySrcDstField dst;
      bool inst_lock_needed;
      Event inst_lock_event;
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy;
      Waiter waiter; // if we need to wait on events
    };

    class FillRequest : public DmaRequest {
    public:
      FillRequest(const void *data, size_t msglen,
                  RegionInstance inst,
                  FieldID field_id, unsigned size,
                  Event _before_fill, 
                  Event _after_fill,
                  int priority);
      FillRequest(const TransferDomain *_domain, //const Domain &_domain,
                  const CopySrcDstField &_dst,
                  const void *fill_value, size_t fill_size,
                  Event _before_fill,
                  Event _after_fill,
                  int priority,
                  const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~FillRequest(void);

    public:
      void forward_request(NodeID target_node);

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      template<int DIM>
      void perform_dma_rect(MemoryImpl *mem_impl);

      size_t optimize_fill_buffer(RegionInstanceImpl *impl, int &fill_elmts);

      TransferDomain *domain;
      //Domain domain;
      CopySrcDstField dst;
      void *fill_buffer;
      size_t fill_size;
      Event before_fill;
      Waiter waiter;
    };

    // each DMA "channel" implements one of these to describe (implicitly) which copies it
    //  is capable of performing and then to actually construct a MemPairCopier for copies 
    //  between a given pair of memories
    // NOTE: we no longer use MemPairCopier's, but these are left in as
    //  placeholders for having channels be created more modularly
    class MemPairCopierFactory {
    public:
      MemPairCopierFactory(const std::string& _name);
      virtual ~MemPairCopierFactory(void);

      const std::string& get_name(void) const;

      // TODO: consider responding with a "goodness" metric that would allow choosing between
      //  multiple capable channels - these metrics are the probably the same as "mem-to-mem affinity"
      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold) = 0;

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold) = 0;
#endif

    protected:
      std::string name;
    };

    class Request;

    class AsyncFileIOContext {
    public:
      AsyncFileIOContext(int _max_depth);
      ~AsyncFileIOContext(void);

      void enqueue_write(int fd, size_t offset, size_t bytes, const void *buffer, Request* req = NULL);
      void enqueue_read(int fd, size_t offset, size_t bytes, void *buffer, Request* req = NULL);
      void enqueue_fence(DmaRequest *req);

      bool empty(void);
      long available(void);
      void make_progress(void);

      static AsyncFileIOContext* get_singleton(void);

      class AIOOperation {
      public:
	virtual ~AIOOperation(void) {}
	virtual void launch(void) = 0;
	virtual bool check_completion(void) = 0;
	bool completed;
        void* req;
      };

      int max_depth;
      std::deque<AIOOperation *> launched_operations, pending_operations;
      GASNetHSL mutex;
#ifdef REALM_USE_KERNEL_AIO
      aio_context_t aio_ctx;
#endif
    };
};

#endif
