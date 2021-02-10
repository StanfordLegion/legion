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
  class CoreReservationSet;

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

  struct RemoteIBAllocRequestAsync {
      Memory memory;
      void *req;
      void *ibinfo;
      size_t size;
      //int idx;
      //ID::IDType src_inst_id, dst_inst_id;

      static void handle_message(NodeID sender, const RemoteIBAllocRequestAsync &args,
				 const void *data, size_t msglen);
    };

    struct RemoteIBAllocResponseAsync {
      void *req;
      void *ibinfo;
      //int idx;
      //ID::IDType src_inst_id, dst_inst_id;
      off_t offset;
      //size_t size;

      static void handle_message(NodeID sender, const RemoteIBAllocResponseAsync &args,
				 const void *data, size_t msglen);
    };

    struct RemoteIBFreeRequestAsync {
      Memory memory;
      off_t ib_offset;
      size_t ib_size;

      static void handle_message(NodeID sender, const RemoteIBFreeRequestAsync &args,
				 const void *data, size_t msglen);
    };

    struct RemoteCopyMessage {
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy, after_copy;
      int priority;

      static void handle_message(NodeID sender, const RemoteCopyMessage &args,
				 const void *data, size_t msglen);
    };

    struct RemoteFillMessage {
      RegionInstance inst;
      FieldID field_id;
      unsigned size;
      Event before_fill, after_fill;
      //int priority;

      static void handle_message(NodeID sender, const RemoteFillMessage &args,
				 const void *data, size_t msglen);
    };

    extern void init_dma_handler(void);

    extern void start_dma_worker_threads(int count, Realm::CoreReservationSet& crs);
    extern void stop_dma_worker_threads(void);

    extern void start_dma_system(int count, bool pinned, int max_nr,
				 CoreReservationSet& crs,
				 BackgroundWorkManager *bgwork);

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
    
    class DmaRequestQueue;
    // for now we use a single queue for all legacy fills/reduces
    extern DmaRequestQueue *dma_queue;

    // include files are all tangled up, so some XferDes stuff here...  :(
    class XferDesFactory;

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
    class TransferDomain;

    class DmaRequest : public Realm::Operation {
    public:
      DmaRequest(int _priority,
		 GenEventImpl *_after_copy, EventImpl::gen_t _after_gen);

      DmaRequest(int _priority,
		 GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
                 const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~DmaRequest(void);

    public:
      virtual void print(std::ostream& os) const;

      virtual bool check_readiness(void) = 0;

      virtual bool handler_safe(void) = 0;

      virtual void perform_dma(void) = 0;

      enum State {
	STATE_INIT,
	STATE_METADATA_FETCH,
	STATE_DST_FETCH,
	STATE_GEN_PATH,
	STATE_ALLOC_IB,
	STATE_WAIT_IB,
	STATE_BEFORE_EVENT,
	STATE_INST_LOCK,
	STATE_READY,
	STATE_QUEUED,
	STATE_DONE
      };

      State state;
      int priority;
      // <NEWDMA>
      Mutex request_lock;
      std::vector<XferDesID> xd_ids;

      Event tgt_fetch_completion;
      // </NEWDMA>

      class Waiter : public EventWaiter {
      public:
        Waiter(void);
        virtual ~Waiter(void);
      public:
	DmaRequest *req;
	Event wait_on;

	void sleep_on_event(Event e);

	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;
      };
    };

    void free_intermediate_buffer(Memory mem, off_t offset, size_t size);


    class MemPairCopier;

#if 0
    struct PendingIBInfo {
      Memory memory;
      int idx;
      InstPair ip;
    };

    class ComparePendingIBInfo {
    public:
      bool operator() (const PendingIBInfo& a, const PendingIBInfo& b) const {
        if (a.memory.id == b.memory.id) {
          assert(a.idx != b.idx);
          return a.idx < b.idx;
        }
        else
          return a.memory.id < b.memory.id;
      }
    };
#endif

    struct IBInfo {
#if 0
      enum Status {
        INIT,
        SENT,
        COMPLETED
      };
#endif
      Memory memory;
      off_t offset;
      size_t size;
      //Status status;
      //IBFence* fence;
      //Event event;

      void set(Memory _memory, size_t _size);
    };

  //typedef std::set<PendingIBInfo, ComparePendingIBInfo> PriorityIBQueue;
    typedef std::vector<IBInfo> IBVec;
    typedef std::map<InstPair, IBVec> IBByInst;
    typedef std::map<Memory, std::vector<IBInfo *> > PendingIBRequests;

    class TransferDomain;
    class TransferIterator;
    class IndirectionInfo;

    // dma requests come in two flavors:
    // 1) CopyRequests, which are per memory pair, and
    // 2) ReduceRequests, which have to be handled monolithically

    class CopyRequest : public DmaRequest {
    public:
      CopyRequest(const void *data, size_t datalen,
		  Event _before_copy,
		  GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
		  int _priority);

      CopyRequest(const TransferDomain *_domain, //const Domain& _domain,
		  OASByInst *_oas_by_inst,
		  IndirectionInfo *_gather_info,
		  IndirectionInfo *_scatter_info,
		  Event _before_copy,
		  GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
		  int _priority,
                  const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~CopyRequest(void);

      void create_xfer_descriptors(void);

      virtual void mark_completed(void);

    public:
      void forward_request(NodeID target_node);

      virtual bool check_readiness(void);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      TransferDomain *domain;
      //Domain domain;
      OASByInst *oas_by_inst;
      IndirectionInfo *gather_info;
      IndirectionInfo *scatter_info;

      // <NEW_DMA>
      void alloc_intermediate_buffer(InstPair inst_pair, Memory tgt_mem, int idx);

      void handle_ib_response(//int idx, InstPair inst_pair, size_t ib_size,
			      IBInfo *ibinfo, off_t ib_offset);

      //PriorityIBQueue priority_ib_queue;
      //PendingIBRequests pending_ib_requests;
      atomic<size_t> ib_responses_needed;
      // operations on ib_by_inst are protected by ib_mutex
      //IBByInst ib_by_inst;
      //Mutex ib_mutex;
      //std::vector<Memory> mem_path;
      // </NEW_DMA>

      // copies with generalized scatter and gather have a DAG that describes
      //  the overall transfer: nodes are transfer descriptors and edges are
      //  intermediate buffers
      struct XDTemplate {
	NodeID target_node;
	//XferDesKind kind;
	XferDesFactory *factory;
	int gather_control_input;
	int scatter_control_input;
	enum { // special edge numbers
	  SRC_INST = -1,
	  DST_INST = -2,
	  INDIRECT_BASE = -1000,
	};
	struct IO {
	  int edge_id;
	  RegionInstance indirect_inst;
	};
	std::vector<IO> inputs;  // TODO: short vectors
	std::vector<IO> outputs;

	// helper functions for initializing these things
	void set_simple(Channel *channel,
			int in_edge, int out_edge);
      };
      std::vector<XDTemplate> xd_nodes;
      std::vector<IBInfo> ib_edges;

      Event before_copy;
      Waiter waiter; // if we need to wait on events
    };

  class IndirectionInfo {
  public:
    virtual ~IndirectionInfo(void) {}
    virtual Event request_metadata(void) = 0;
    virtual Memory generate_gather_paths(Memory dst_mem, int dst_edge_id,
					 size_t bytes_per_element,
					 CustomSerdezID serdez_id,
					 std::vector<CopyRequest::XDTemplate>& xd_nodes,
					 std::vector<IBInfo>& ib_edges) = 0;

    virtual void generate_gather_paths(Memory dst_mem,
				       TransferGraph::XDTemplate::IO dst_edge,
				       unsigned indirect_idx,
				       unsigned src_fld_start,
				       unsigned src_fld_count,
				       size_t bytes_per_element,
				       CustomSerdezID serdez_id,
				       std::vector<TransferGraph::XDTemplate>& xd_nodes,
				       std::vector<TransferGraph::IBInfo>& ib_edges,
				       std::vector<TransferDesc::FieldInfo>& src_fields) = 0;

    virtual Memory generate_scatter_paths(Memory src_mem, int src_edge_id,
					  size_t bytes_per_element,
					  CustomSerdezID serdez_id,
					  std::vector<CopyRequest::XDTemplate>& xd_nodes,
					  std::vector<IBInfo>& ib_edges) = 0;

    virtual void generate_scatter_paths(Memory src_mem,
					TransferGraph::XDTemplate::IO src_edge,
					unsigned indirect_idx,
					unsigned dst_fld_start,
					unsigned dst_fld_count,
					size_t bytes_per_element,
					CustomSerdezID serdez_id,
					std::vector<TransferGraph::XDTemplate>& xd_nodes,
					std::vector<TransferGraph::IBInfo>& ib_edges,
					std::vector<TransferDesc::FieldInfo>& src_fields) = 0;

    virtual RegionInstance get_pointer_instance(void) const = 0;
      
    virtual TransferIterator *create_address_iterator(RegionInstance peer) const = 0;

    virtual TransferIterator *create_indirect_iterator(Memory addrs_mem,
						       RegionInstance inst,
						       const std::vector<FieldID>& fields,
						       const std::vector<size_t>& fld_offsets,
						       const std::vector<size_t>& fld_sizes) const = 0;

    virtual void print(std::ostream& os) const = 0;
  };

  std::ostream& operator<<(std::ostream& os, const IndirectionInfo& ii);
    class ReduceRequest : public DmaRequest {
    public:
      ReduceRequest(const void *data, size_t datalen,
		    ReductionOpID _redop_id,
		    bool _red_fold,
		    Event _before_copy,
		    GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
		    int _priority);

      ReduceRequest(const TransferDomain *_domain, //const Domain& _domain,
		    const std::vector<CopySrcDstField>& _srcs,
		    const CopySrcDstField& _dst,
		    bool _inst_lock_needed,
		    ReductionOpID _redop_id,
		    bool _red_fold,
		    Event _before_copy,
		    GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
		    int _priority,
                    const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~ReduceRequest(void);

      virtual void mark_completed(void);

    public:
      void forward_request(NodeID target_node);

      void set_dma_queue(DmaRequestQueue *queue);

      virtual bool check_readiness(void);

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
      DmaRequestQueue *dma_queue;
    };

    class FillRequest : public DmaRequest {
    public:
      FillRequest(const void *data, size_t msglen,
                  RegionInstance inst,
                  FieldID field_id, unsigned size,
                  Event _before_fill, 
		  GenEventImpl *_after_fill, EventImpl::gen_t _after_gen,
                  int priority);
      FillRequest(const TransferDomain *_domain, //const Domain &_domain,
                  const CopySrcDstField &_dst,
                  const void *fill_value, size_t fill_size,
                  Event _before_fill,
		  GenEventImpl *_after_fill, EventImpl::gen_t _after_gen,
                  int priority,
                  const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~FillRequest(void);

      virtual void mark_completed(void);

    public:
      void forward_request(NodeID target_node);

      void set_dma_queue(DmaRequestQueue *queue);

      virtual bool check_readiness(void);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      template<int DIM>
      void perform_dma_rect(MemoryImpl *mem_impl);

      TransferDomain *domain;
      //Domain domain;
      CopySrcDstField dst;
      void *fill_buffer;
      size_t fill_size;
      Event before_fill;
      Waiter waiter;
      DmaRequestQueue *dma_queue;
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

  class CopyProfile : public DmaRequest {
    public:
    CopyProfile(GenEventImpl *_after_copy,
                EventImpl::gen_t _after_gen, int _priority,
                const Realm::ProfilingRequestSet & _reqs)
      : DmaRequest(_priority, _after_copy, _after_gen, _reqs),
        before_copy(Event::NO_EVENT), end_copy(Event::NO_EVENT), total_field_size(0),num_requests(0), src_mem(Memory::NO_MEMORY), dst_mem(Memory::NO_MEMORY), is_src_indirect(false), is_dst_indirect(false)
    {
    }

    void add_copy_entry(const OASByInst *_oas_by_inst,
                        const TransferDomain *domain,
                        bool is_src_indirect,
                        bool is_dst_indirect);

    void add_reduc_entry(const CopySrcDstField &src,
                         const CopySrcDstField &dst,
                         const TransferDomain *domain);

    void add_fill_entry(const CopySrcDstField &dst,
                        const TransferDomain *domain);

  protected:
    ProfilingMeasurements::OperationCopyInfo cpinfo;
    Event before_copy, end_copy;
    size_t total_field_size;
    size_t num_requests;
    Memory src_mem, dst_mem;
    bool is_src_indirect, is_dst_indirect;
    Waiter waiter;
    // deletion performed when reference count goes to zero
    virtual ~CopyProfile(void);
    virtual void mark_completed(void) { Operation::mark_completed(); }

  public:
    void set_end_copy(Event e) { end_copy = e;}
    void set_start_copy(Event e) { before_copy = e;}
    virtual bool check_readiness(void);
    virtual void perform_dma(void) { return; }
    virtual bool handler_safe(void) { return(false); }
  };

    class AsyncFileIOContext : public BackgroundWorkItem {
    public:
      AsyncFileIOContext(int _max_depth);
      ~AsyncFileIOContext(void);

      void enqueue_write(int fd, size_t offset, size_t bytes, const void *buffer, Request* req = NULL);
      void enqueue_read(int fd, size_t offset, size_t bytes, void *buffer, Request* req = NULL);
      void enqueue_fence(DmaRequest *req);

      bool empty(void);
      long available(void);

      static AsyncFileIOContext* get_singleton(void);

      virtual void do_work(TimeLimit work_until);

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
