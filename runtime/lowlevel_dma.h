/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include "lowlevel_impl.h"
#include "activemsg.h"

namespace Realm {
  class CoreReservationSet;
};

namespace LegionRuntime {
  namespace LowLevel {
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

      static void send_request(gasnet_node_t target, Memory tgt_mem, void* req,
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

      static void send_request(gasnet_node_t target, void* req, int idx, ID::IDType src_id,
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

      static void send_request(gasnet_node_t target, Memory tgt_mem,
                               off_t ib_offset, size_t ib_size);
    };

    void find_shortest_path(Memory src_mem, Memory dst_mem, std::vector<Memory>& path);

    struct RemoteCopyArgs : public BaseMedium {
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy, after_copy;
      int priority;
    };

    struct RemoteFillArgs : public BaseMedium {
      RegionInstance inst;
      unsigned offset, size;
      Event before_fill, after_fill;
      //int priority;
    };

    extern void handle_remote_copy(RemoteCopyArgs args, const void *data, size_t msglen);

    extern void handle_remote_fill(RemoteFillArgs args, const void *data, size_t msglen);

    enum DMAActiveMessageIDs {
      REMOTE_COPY_MSGID = 200,
      REMOTE_FILL_MSGID = 201,
    };

    typedef ActiveMessageMediumNoReply<REMOTE_COPY_MSGID,
				       RemoteCopyArgs,
				       handle_remote_copy> RemoteCopyMessage;

    typedef ActiveMessageMediumNoReply<REMOTE_FILL_MSGID,
                                       RemoteFillArgs,
                                       handle_remote_fill> RemoteFillMessage;

    extern void init_dma_handler(void);

    extern void start_dma_worker_threads(int count, Realm::CoreReservationSet& crs);
    extern void stop_dma_worker_threads(void);

    extern void start_dma_system(int count, int max_nr, Realm::CoreReservationSet& crs);

    extern void stop_dma_system(void);
    extern void create_builtin_dma_channels(Realm::RuntimeImpl *r);

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

    struct OffsetsAndSize {
      off_t src_offset, dst_offset;
      int size;
      CustomSerdezID serdez_id;
    };
    typedef std::vector<OffsetsAndSize> OASVec;

    // an interface for objects that are responsible for copying data from one instance to another
    //  these are generally created by MemPairCopier's
    // an InstPairCopier remembers all the addressing details of the source and destination instance,
    //  allowing a generic "iterate over all elements in the index space to be copied" loop to work
    //  for all copy paths
    class InstPairCopier {
    public:
      InstPairCopier(void);
      virtual ~InstPairCopier(void);
    public:
      virtual bool copy_all_fields(Domain d) { return false; }

      virtual void copy_field(off_t src_index, off_t dst_index, off_t elem_count,
                              unsigned offset_index) = 0;

      virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t elem_count) = 0;

      virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t count_per_line,
				   off_t src_stride, off_t dst_stride, off_t lines);

      virtual void flush(void) = 0;
    };

    // many instance pair copiers are "span-based" and the various copies can be implemented using a single
    //  "copy_span" building block - this is provided via templating rather than inheritance to allow for
    //  inlining the calls to copy_span that will occur in for loops
    template <typename T>
    class SpanBasedInstPairCopier : public InstPairCopier {
    public:
      SpanBasedInstPairCopier(T *_span_copier, 
                              RegionInstance _src_inst, 
                              RegionInstance _dst_inst,
                              OASVec &_oas_vec);

      virtual ~SpanBasedInstPairCopier(void);

      virtual void copy_field(off_t src_index, off_t dst_index, off_t elem_count,
                              unsigned offset_index);

      virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t elem_count);

      virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t count_per_line,
				   off_t src_stride, off_t dst_stride, off_t lines);

      virtual void flush(void);

    protected:
      T *span_copier;
      RegionInstanceImpl *src_inst;
      RegionInstanceImpl *dst_inst;
      OASVec &oas_vec;
      std::vector<off_t> src_start;
      std::vector<off_t> dst_start;
      std::vector<int> src_size;
      std::vector<int> dst_size;
      std::vector<bool> partial_field;
    };

    class MemPairCopier {
    public:
      static MemPairCopier* create_copier(Memory src_mem, Memory dst_mem,
					  ReductionOpID redop_id = 0,
					  CustomSerdezID serdez_id = 0,
					  bool fold = false);

      MemPairCopier(void);

      virtual ~MemPairCopier(void);

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec) = 0;

      // default behavior of flush is just to report bytes (maybe)
      virtual void flush(DmaRequest *req);

      void record_bytes(size_t bytes);

      size_t get_total_bytes() { return total_bytes; }

    protected:
      size_t total_reqs, total_bytes;
    };

    // each DMA "channel" implements one of these to describe (implicitly) which copies it
    //  is capable of performing and then to actually construct a MemPairCopier for copies 
    //  between a given pair of memories
    class MemPairCopierFactory {
    public:
      MemPairCopierFactory(const std::string& _name);
      virtual ~MemPairCopierFactory(void);

      const std::string& get_name(void) const;

      // TODO: consider responding with a "goodness" metric that would allow choosing between
      //  multiple capable channels - these metrics are the probably the same as "mem-to-mem affinity"
      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold) = 0;

      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold) = 0;

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
};

// implementation of templated and inline methods
#include "lowlevel_dma.inl"

#endif
