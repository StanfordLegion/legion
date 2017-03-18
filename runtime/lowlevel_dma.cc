/* Copyright 2017 Stanford University, NVIDIA Corporation
 * Copyright 2017 Los Alamos National Laboratory
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
#include "lowlevel_dma.h"
#include "channel.h"
#include "accessor.h"
#include "realm/threads.h"
#include <errno.h>
// included for file memory data transfer
#include <unistd.h>
#ifdef REALM_USE_KERNEL_AIO
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#else
#include <aio.h>
#endif

#include <queue>
#include <algorithm>
#include <iomanip>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

using namespace LegionRuntime::Accessor;

#ifndef __GNUC__
#include "atomics.h"
#endif

#include "realm/timers.h"
#include "realm/serialize.h"

using namespace Realm::Serialization;

namespace LegionRuntime {
  namespace LowLevel {

    typedef Realm::GASNetMemory GASNetMemory;
    typedef Realm::DiskMemory DiskMemory;
    typedef Realm::FileMemory FileMemory;
    typedef Realm::Thread Thread;
    typedef Realm::ThreadLaunchParameters ThreadLaunchParameters;
    typedef Realm::CoreReservation CoreReservation;
    typedef Realm::CoreReservationParameters CoreReservationParameters;

    Logger::Category log_dma("dma");
    Logger::Category log_ib_alloc("ib_alloc");
    extern Logger::Category log_new_dma;
    Logger::Category log_aio("aio");
#ifdef EVENT_GRAPH_TRACE
    extern Logger::Category log_event_graph;
    extern Event find_enclosing_termination_event(void);
#endif

    typedef std::pair<Memory, Memory> MemPair;
    typedef std::pair<RegionInstance, RegionInstance> InstPair;
    typedef std::map<InstPair, OASVec> OASByInst;
    class IBFence : public Realm::Operation::AsyncWorkItem {
    public:
      IBFence(Realm::Operation *op) : Realm::Operation::AsyncWorkItem(op) {}
      virtual void request_cancellation(void) {
        // ignored for now
      }
      virtual void print(std::ostream& os) const { os << "IBFence"; }
    };

    struct IBInfo {
      Memory memory;
      off_t offset;
      size_t size;
      IBFence* fence;
    };
    typedef std::vector<IBInfo> IBVec;
    typedef std::map<InstPair, IBVec> IBByInst;
    typedef std::map<MemPair, OASByInst *> OASByMem;

    class IBAllocRequest {
    public:
      IBAllocRequest(gasnet_node_t _owner, void* _req, int _idx,
                     ID::IDType _src_inst_id, ID::IDType _dst_inst_id,
                     size_t _ib_size)
        : owner(_owner), req(_req), idx(_idx), src_inst_id(_src_inst_id),
          dst_inst_id(_dst_inst_id), ib_size(_ib_size) {ib_offset = -1;}
    public:
      gasnet_node_t owner;
      void* req;
      int idx;
      ID::IDType src_inst_id, dst_inst_id;
      size_t ib_size;
      off_t ib_offset;
    };

    class PendingIBQueue {
    public:
      PendingIBQueue();

      void enqueue_request(Memory tgt_mem, IBAllocRequest* req);

      void dequeue_request(Memory tgt_mem);

    protected:
      GASNetHSL queue_mutex;
      std::map<Memory, std::queue<IBAllocRequest*> *> queues;
    };

    class DmaRequest;

    class DmaRequestQueue {
    public:
      DmaRequestQueue(Realm::CoreReservationSet& crs);

      void enqueue_request(DmaRequest *r);

      DmaRequest *dequeue_request(bool sleep = true);

      void shutdown_queue(void);

      void start_workers(int count);

      void worker_thread_loop(void);

    protected:
      GASNetHSL queue_mutex;
      GASNetCondVar queue_condvar;
      std::map<int, std::list<DmaRequest *> *> queues;
      int queue_sleepers;
      bool shutdown_flag;
      CoreReservation core_rsrv;
      std::vector<Thread *> worker_threads;
    };

  ////////////////////////////////////////////////////////////////////////
  //
  // class DmaRequest
  //

    DmaRequest::DmaRequest(int _priority, Event _after_copy) 
      : Operation(_after_copy, Realm::ProfilingRequestSet()),
	state(STATE_INIT), priority(_priority)
    {
      tgt_fetch_completion = Event::NO_EVENT;
      pthread_mutex_init(&request_lock, NULL);
    }

    DmaRequest::DmaRequest(int _priority, Event _after_copy,
			   const Realm::ProfilingRequestSet &reqs)
      : Realm::Operation(_after_copy, reqs), state(STATE_INIT),
	priority(_priority)
    {
      tgt_fetch_completion = Event::NO_EVENT;
      pthread_mutex_init(&request_lock, NULL);
    }

    DmaRequest::~DmaRequest(void)
    {
      pthread_mutex_destroy(&request_lock);
    }

    void DmaRequest::print(std::ostream& os) const
    {
      os << "DmaRequest";
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DmaRequest::Waiter
  //

    DmaRequest::Waiter::Waiter(void)
    {
    }

    DmaRequest::Waiter::~Waiter(void)
    {
    }

    // dma requests come in two flavors:
    // 1) CopyRequests, which are per memory pair, and
    // 2) ReduceRequests, which have to be handled monolithically

    class CopyRequest : public DmaRequest {
    public:
      CopyRequest(const void *data, size_t datalen,
		  Event _before_copy,
		  Event _after_copy,
		  int _priority);

      CopyRequest(const Domain& _domain,
		  OASByInst *_oas_by_inst,
		  Event _before_copy,
		  Event _after_copy,
		  int _priority,
                  const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~CopyRequest(void);

    public:
      size_t compute_size(void) const;
      void serialize(void *buffer);

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq);

      void perform_dma_mask(MemPairCopier *mpc);

      template <unsigned DIM>
      void perform_dma_rect(MemPairCopier *mpc);

      template <unsigned DIM>
      void perform_new_dma(Memory src_mem, Memory dst_mem);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      Domain domain;
      OASByInst *oas_by_inst;

      // <NEW_DMA>
      void alloc_intermediate_buffer(InstPair inst_pair, Memory tgt_mem, int idx);

      void handle_ib_response(int idx, InstPair inst_pair, size_t ib_size, off_t ib_offset);

      // operations on ib_by_inst are protected by ib_mutex
      IBByInst ib_by_inst;
      GASNetHSL ib_mutex;
      class IBAllocOp : public Realm::Operation {
      public:
        IBAllocOp(Event _completion) : Operation(_completion, Realm::ProfilingRequestSet()) {};
        ~IBAllocOp() {};
        void print(std::ostream& os) const {os << "IBAllocOp"; };
      };

      IBAllocOp* ib_req;
      Event ib_completion;
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

      ReduceRequest(const Domain& _domain,
		    const std::vector<Domain::CopySrcDstField>& _srcs,
		    const Domain::CopySrcDstField& _dst,
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
      size_t compute_size(void);
      void serialize(void *buffer);

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq);

      void perform_dma_mask(MemPairCopier *mpc);

      template <unsigned DIM>
      void perform_dma_rect(MemPairCopier *mpc);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      Domain domain;
      std::vector<Domain::CopySrcDstField> srcs;
      Domain::CopySrcDstField dst;
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
                  unsigned offset, unsigned size,
                  Event _before_fill, 
                  Event _after_fill,
                  int priority);
      FillRequest(const Domain &_domain,
                  const Domain::CopySrcDstField &_dst,
                  const void *fill_value, size_t fill_size,
                  Event _before_fill,
                  Event _after_fill,
                  int priority,
                  const Realm::ProfilingRequestSet &reqs);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~FillRequest(void);

    public:
      size_t compute_size(void);
      void serialize(void *buffer);

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      template<int DIM>
      void perform_dma_rect(MemoryImpl *mem_impl);

      size_t optimize_fill_buffer(RegionInstanceImpl *impl, int &fill_elmts);

      Domain domain;
      Domain::CopySrcDstField dst;
      void *fill_buffer;
      size_t fill_size;
      Event before_fill;
      Waiter waiter;
    };

    static PendingIBQueue *ib_req_queue = 0;

    PendingIBQueue::PendingIBQueue() {}

    void PendingIBQueue::enqueue_request(Memory tgt_mem, IBAllocRequest* req)
    {
      AutoHSLLock al(queue_mutex);
      assert(ID(tgt_mem).memory.owner_node == gasnet_mynode());
      // If we can allocate in target memory, no need to pend the request
      off_t ib_offset = get_runtime()->get_memory_impl(tgt_mem)->alloc_bytes(req->ib_size);
      if (ib_offset >= 0) {
        if (req->owner == gasnet_mynode()) {
          // local ib alloc request
          CopyRequest* cr = (CopyRequest*) req->req;
          RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(req->src_inst_id);
          RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(req->dst_inst_id);
          InstPair inst_pair(src_impl->me, dst_impl->me);
          cr->handle_ib_response(req->idx, inst_pair, req->ib_size, ib_offset);
        } else {
          // remote ib alloc request
          RemoteIBAllocResponseAsync::send_request(req->owner, req->req, req->idx,
              req->src_inst_id, req->dst_inst_id, req->ib_size, ib_offset); 
        }
        // Remember to free IBAllocRequest
        delete req;

        return;
      }
      log_ib_alloc.info() << "IBAllocRequest (" << req->src_inst_id << "," 
        << req->dst_inst_id << "): no enough space in memory" << tgt_mem;
      std::map<Memory, std::queue<IBAllocRequest*> *>::iterator it = queues.find(tgt_mem);
      if (it == queues.end()) {
        std::queue<IBAllocRequest*> *q = new std::queue<IBAllocRequest*>;
        q->push(req);
        queues[tgt_mem] = q;
      } else {
        it->second->push(req);
      }
    }

    void PendingIBQueue::dequeue_request(Memory tgt_mem)
    {
      AutoHSLLock al(queue_mutex);
      assert(ID(tgt_mem).memory.owner_node == gasnet_mynode());
      std::map<Memory, std::queue<IBAllocRequest*> *>::iterator it = queues.find(tgt_mem);
      // no pending ib requests
      if (it == queues.end()) return;
      while (!it->second->empty()) {
        IBAllocRequest* req = it->second->front();
        off_t ib_offset = get_runtime()->get_memory_impl(tgt_mem)->alloc_bytes(req->ib_size);
        if (ib_offset < 0) break;
        //printf("req: src_inst_id(%llx) dst_inst_id(%llx) ib_size(%lu) idx(%d)\n", req->src_inst_id, req->dst_inst_id, req->ib_size, req->idx);
        // deal with the completed ib alloc request
        log_ib_alloc.info() << "IBAllocRequest (" << req->src_inst_id << "," 
          << req->dst_inst_id << "): completed!";
        if (req->owner == gasnet_mynode()) {
          // local ib alloc request
          CopyRequest* cr = (CopyRequest*) req->req;
          RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(req->src_inst_id);
          RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(req->dst_inst_id);
          InstPair inst_pair(src_impl->me, dst_impl->me);
          cr->handle_ib_response(req->idx, inst_pair, req->ib_size, ib_offset);
        } else {
          // remote ib alloc request
          RemoteIBAllocResponseAsync::send_request(req->owner, req->req, req->idx,
              req->src_inst_id, req->dst_inst_id, req->ib_size, ib_offset); 
        }
        it->second->pop();
        // Remember to free IBAllocRequest
        delete req;
      }
      // if queue is empty, delete from list
      if(it->second->empty()) {
	delete it->second;
	queues.erase(it);
      }
    }

    DmaRequestQueue::DmaRequestQueue(Realm::CoreReservationSet& crs)
      : queue_condvar(queue_mutex)
      , core_rsrv("DMA request queue", crs, CoreReservationParameters())
    {
      queue_sleepers = 0;
      shutdown_flag = false;
    }

    void DmaRequestQueue::shutdown_queue(void)
    {
      queue_mutex.lock();

      assert(queues.empty());

      // set the shutdown flag and wake up any sleepers
      shutdown_flag = true;

      queue_condvar.broadcast();
      queue_mutex.unlock();

      // reap all the threads
      for(std::vector<Thread *>::iterator it = worker_threads.begin();
	  it != worker_threads.end();
	  it++) {
	(*it)->join();
	delete (*it);
      }
      worker_threads.clear();
    }

    void DmaRequestQueue::enqueue_request(DmaRequest *r)
    {
      // Record that it is ready - check for cancellation though
      bool ok_to_run = r->mark_ready();
      if(!ok_to_run) {
	r->mark_finished(false /*!successful*/);
	return;
      }

      queue_mutex.lock();

      // there's a queue per priority level
      // priorities are negated so that the highest logical priority comes first
      int p = -r->priority;
      std::map<int, std::list<DmaRequest *> *>::iterator it = queues.find(p);
      if(it == queues.end()) {
	// nothing at this priority level - make a new list
	std::list<DmaRequest *> *l = new std::list<DmaRequest *>;
	l->push_back(r);
	queues[p] = l;
      } else {
	// push ourselves onto the back of the existing queue
	it->second->push_back(r);
      }

      // if anybody was sleeping, wake them up
      if(queue_sleepers > 0) {
	queue_sleepers = 0;
	queue_condvar.broadcast();
      }

      queue_mutex.unlock();
    }

    DmaRequest *DmaRequestQueue::dequeue_request(bool sleep /*= true*/)
    {
      queue_mutex.lock();

      // quick check - are there any requests at all?
      while(queues.empty()) {
	if(!sleep || shutdown_flag) {
	  queue_mutex.unlock();
	  return 0;
	}

	// sleep until there are, or until shutdown
	queue_sleepers++;
	queue_condvar.wait();
      }

      // grab the first request from the highest-priority queue there is
      // priorities are negated so that the highest logical priority comes first
      std::map<int, std::list<DmaRequest *> *>::iterator it = queues.begin();
      assert(!it->second->empty());
      DmaRequest *r = it->second->front();
      it->second->pop_front();
      // if queue is empty, delete from list
      if(it->second->empty()) {
	delete it->second;
	queues.erase(it);
      }

      queue_mutex.unlock();
      
      return r;
    } 

    CopyRequest::CopyRequest(const void *data, size_t datalen,
			     Event _before_copy,
			     Event _after_copy,
			     int _priority)
      : DmaRequest(_priority, _after_copy),
	oas_by_inst(0),
	before_copy(_before_copy)
    {
      const IDType *idata = (const IDType *)data;

      idata = domain.deserialize(idata);

      oas_by_inst = new OASByInst;

      // <NEW_DMA>
      ib_completion = GenEventImpl::create_genevent()->current_event();
      ib_req = new IBAllocOp(ib_completion);
      ib_by_inst.clear();
      // </NEW_DMA>

      size_t num_pairs = *idata++;

      for (unsigned idx = 0; idx < num_pairs; idx++) {
	RegionInstance src_inst = ID((IDType)*idata++).convert<RegionInstance>();
	RegionInstance dst_inst = ID((IDType)*idata++).convert<RegionInstance>();
	InstPair ip(src_inst, dst_inst);

        // If either one of the instances is in GPU memory increase priority
        if (priority == 0)
        {
          MemoryImpl::MemoryKind src_kind = get_runtime()->get_memory_impl(get_runtime()->get_instance_impl(src_inst)->memory)->kind;
          if (src_kind == MemoryImpl::MKIND_GPUFB)
            priority = 1;
          else
          {
            MemoryImpl::MemoryKind dst_kind = get_runtime()->get_memory_impl(get_runtime()->get_instance_impl(dst_inst)->memory)->kind;
            if (dst_kind == MemoryImpl::MKIND_GPUFB)
              priority = 1;
          }
        }

	OASVec& oasvec = (*oas_by_inst)[ip];

	unsigned count = *idata++;
	for(unsigned i = 0; i < count; i++) {
	  OffsetsAndSize oas;
	  oas.src_offset = *idata++;
	  oas.dst_offset = *idata++;
	  oas.size = *idata++;
          oas.serdez_id = *idata++;
	  oasvec.push_back(oas);
	}
      }
      // Unpack any profiling requests 
      // TODO: unbreak once the serialization stuff is repaired
      //const void *result = requests.deserialize(idata);
      //Realm::Operation::reconstruct_measurements();
      // better have consumed exactly the right amount of data
      //assert((((unsigned long)result) - ((unsigned long)data)) == datalen);
      size_t request_size = *reinterpret_cast<const size_t*>(idata);
      idata += sizeof(size_t) / sizeof(IDType);
      FixedBufferDeserializer deserializer(idata, request_size);
      deserializer >> requests;
      Realm::Operation::reconstruct_measurements();

      log_dma.info() << "dma request " << (void *)this << " deserialized - is="
		     << domain << " before=" << before_copy << " after=" << get_finish_event();
      for(OASByInst::const_iterator it = oas_by_inst->begin();
	  it != oas_by_inst->end();
	  it++)
	for(OASVec::const_iterator it2 = it->second.begin();
	    it2 != it->second.end();
	    it2++)
	  log_dma.info() << "dma request " << (void *)this << " field: " <<
	    it->first.first << "[" << it2->src_offset << "]->" <<
	    it->first.second << "[" << it2->dst_offset << "] size=" << it2->size <<
	    " serdez=" << it2->serdez_id;
    }

    CopyRequest::CopyRequest(const Domain& _domain,
			     OASByInst *_oas_by_inst,
			     Event _before_copy,
			     Event _after_copy,
			     int _priority,
                             const Realm::ProfilingRequestSet &reqs)
      : DmaRequest(_priority, _after_copy, reqs),
	domain(_domain), oas_by_inst(_oas_by_inst),
	before_copy(_before_copy)
    {
      // <NEW_DMA>
      ib_completion = GenEventImpl::create_genevent()->current_event();
      ib_req = new IBAllocOp(ib_completion);
      ib_by_inst.clear();
      // </NEW_DMA>
      log_dma.info() << "dma request " << (void *)this << " created - is="
		     << domain << " before=" << before_copy << " after=" << get_finish_event();
      for(OASByInst::const_iterator it = oas_by_inst->begin();
	  it != oas_by_inst->end();
	  it++)
	for(OASVec::const_iterator it2 = it->second.begin();
	    it2 != it->second.end();
	    it2++)
	  log_dma.info() << "dma request " << (void *)this << " field: " <<
	    it->first.first << "[" << it2->src_offset << "]->" <<
	    it->first.second << "[" << it2->dst_offset << "] size=" << it2->size <<
	    " serdez=" << it2->serdez_id;
    }
 
    CopyRequest::~CopyRequest(void)
    {
      //<NEWDMA>
      // destroy all xfer des
      std::vector<XferDesID>::iterator it;
      for (it = path.begin(); it != path.end(); it++) {
        destroy_xfer_des(*it);
      }
      // free intermediate buffers
      {
        AutoHSLLock al(ib_mutex);
        for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
          IBVec& ib_vec = ib_by_inst[it->first];
          for (IBVec::iterator it2 = ib_vec.begin(); it2 != ib_vec.end(); it2++) {
            if(ID(it2->memory).memory.owner_node == gasnet_mynode()) {
              get_runtime()->get_memory_impl(it2->memory)->free_bytes(it2->offset, it2->size);
              ib_req_queue->dequeue_request(it2->memory);
            } else {
              RemoteIBFreeRequestAsync::send_request(ID(it2->memory).memory.owner_node,
                  it2->memory, it2->offset, it2->size);
            }
          }
        }
      }
      delete ib_req;
      //</NEWDMA>
      delete oas_by_inst;
    }

    size_t CopyRequest::compute_size(void) const
    {
      size_t result = domain.compute_size();
      result += sizeof(IDType); // number of requests;
      for(OASByInst::iterator it2 = oas_by_inst->begin(); it2 != oas_by_inst->end(); it2++) {
        OASVec& oasvec = it2->second;
        result += (3 + oasvec.size() * 4) * sizeof(IDType);
      }
      // TODO: unbreak once the serialization stuff is repaired
      //result += requests.compute_size();
      ByteCountSerializer counter;
      counter << requests;
      result += sizeof(size_t) + counter.bytes_used();
      return result;
    }

    void CopyRequest::serialize(void *buffer)
    {
      // domain info goes first
      IDType *msgptr = domain.serialize((IDType *)buffer);

      *msgptr++ = oas_by_inst->size();

      // now OAS vectors
      for(OASByInst::iterator it2 = oas_by_inst->begin(); it2 != oas_by_inst->end(); it2++) {
	RegionInstance src_inst = it2->first.first;
	RegionInstance dst_inst = it2->first.second;
	OASVec& oasvec = it2->second;

	*msgptr++ = src_inst.id;
	*msgptr++ = dst_inst.id;
	*msgptr++ = oasvec.size();
	for(OASVec::iterator it3 = oasvec.begin(); it3 != oasvec.end(); it3++) {
	  *msgptr++ = it3->src_offset;
	  *msgptr++ = it3->dst_offset;
	  *msgptr++ = it3->size;
          *msgptr++ = it3->serdez_id;
	}
      }
      // TODO: unbreak once the serialization stuff is repaired
      //requests.serialize(msgptr); 
      // We sent this message remotely, so we need to clear the profiling
      // so it doesn't get sent accidentally
      ByteCountSerializer counter;
      counter << requests;
      *reinterpret_cast<size_t*>(msgptr) = counter.bytes_used();
      msgptr += sizeof(size_t) / sizeof(IDType);
      FixedBufferSerializer serializer(msgptr, counter.bytes_used());
      serializer << requests;
      clear_profiling();
    }

    void DmaRequest::Waiter::sleep_on_event(Event e, 
					    Reservation l /*= Reservation::NO_RESERVATION*/)
    {
      current_lock = l;
      EventImpl::add_waiter(e, this);
    }

    bool DmaRequest::Waiter::event_triggered(Event e, bool poisoned)
    {
      if(poisoned) {
	Realm::log_poison.info() << "cancelling poisoned dma operation - op=" << req << " after=" << req->get_finish_event();
	req->handle_poisoned_precondition(e);
	return false;
      }

      log_dma.debug("request %p triggered in state %d (lock = " IDFMT ")",
		    req, req->state, current_lock.id);

      if(current_lock.exists()) {
	current_lock.release();
	current_lock = Reservation::NO_RESERVATION;
      }

      // this'll enqueue the DMA if it can, or wait on another event if it 
      //  can't
      req->check_readiness(false, queue);

      // don't delete us!
      return false;
    }

    void DmaRequest::Waiter::print(std::ostream& os) const
    {
      os << "dma request " << (void *)req << ": after " << req->get_finish_event();
    }

    Event DmaRequest::Waiter::get_finish_event(void) const
    {
      return req->get_finish_event();
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class RemoteIBAllocRequestAsync
    //

    /*static*/ void RemoteIBAllocRequestAsync::handle_request(RequestArgs args)
    {
      assert(ID(args.memory).memory.owner_node == gasnet_mynode());
      IBAllocRequest* ib_req
          = new IBAllocRequest(args.node, args.req, args.idx,
                               args.src_inst_id, args.dst_inst_id, args.size);
      ib_req_queue->enqueue_request(args.memory, ib_req);
    }

    /*static*/ void RemoteIBAllocRequestAsync::send_request(gasnet_node_t target, Memory tgt_mem, void* req, int idx, ID::IDType src_inst_id, ID::IDType dst_inst_id, size_t ib_size)
    {
      RequestArgs args;
      args.node = gasnet_mynode();
      args.memory = tgt_mem;
      args.req = req;
      args.idx = idx;
      args.src_inst_id = src_inst_id;
      args.dst_inst_id = dst_inst_id;
      args.size = ib_size;
      Message::request(target, args);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class RemoteIBAllocResponseAsync
    //

    /*static*/ void RemoteIBAllocResponseAsync::handle_request(RequestArgs args)
    {
      CopyRequest* req = (CopyRequest*) args.req;
      RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(args.src_inst_id);
      RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(args.dst_inst_id);
      InstPair inst_pair(src_impl->me, dst_impl->me);
      req->handle_ib_response(args.idx, inst_pair, args.size, args.offset);
    }

    /*static*/ void RemoteIBAllocResponseAsync::send_request(gasnet_node_t target, void* req, int idx, ID::IDType src_inst_id, ID::IDType dst_inst_id, size_t ib_size, off_t ib_offset)
    {
      RequestArgs args;
      args.req = req;
      args.idx = idx;
      args.src_inst_id = src_inst_id;
      args.dst_inst_id = dst_inst_id;
      args.size = ib_size;
      args.offset = ib_offset;
      Message::request(target, args);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class RemoteIBFreeRequestAsync
    //

    /*static*/ void RemoteIBFreeRequestAsync::handle_request(RequestArgs args)
    {
      assert(ID(args.memory).memory.owner_node == gasnet_mynode());
      get_runtime()->get_memory_impl(args.memory)->free_bytes(args.ib_offset, args.ib_size);
      ib_req_queue->dequeue_request(args.memory);
    }

    /*static*/ void RemoteIBFreeRequestAsync::send_request(gasnet_node_t target, Memory tgt_mem, off_t ib_offset, size_t ib_size)
    {
      RequestArgs args;
      args.memory = tgt_mem;
      args.ib_offset = ib_offset;
      args.ib_size = ib_size;
      Message::request(target, args);
    }


#define IB_MAX_SIZE (128 * 1024 * 1024)

    void CopyRequest::alloc_intermediate_buffer(InstPair inst_pair, Memory tgt_mem, int idx)
    {
      assert(oas_by_inst->find(inst_pair) != oas_by_inst->end());
      OASVec& oasvec = (*oas_by_inst)[inst_pair];
      size_t ib_elmnt_size = 0, domain_size = 0;
      for(OASVec::const_iterator it = oasvec.begin(); it != oasvec.end(); it++) {
        ib_elmnt_size += it->size;
      }
      if (domain.get_dim() == 0) {
        IndexSpaceImpl *ispace = get_runtime()->get_index_space_impl(domain.get_index_space());
        assert(get_runtime()->get_instance_impl(inst_pair.second)->metadata.is_valid());
        if (ispace->me == get_runtime()->get_instance_impl(inst_pair.second)->metadata.is) {
          // perform a 1D copy from first_element to last_element
          Realm::StaticAccess<IndexSpaceImpl> data(ispace);
          assert(data->num_elmts > 0);
          Rect<1> new_rect(make_point(data->first_elmt), make_point(data->last_elmt));
          Domain new_domain = Domain::from_rect<1>(new_rect);
          domain_size = new_domain.get_volume();
        } else {
          domain_size = domain.get_volume();
        }
      } else {
        domain_size = domain.get_volume();
      }
      size_t ib_size;
      if (domain_size * ib_elmnt_size < IB_MAX_SIZE)
        ib_size = domain_size * ib_elmnt_size;
      else
        ib_size = IB_MAX_SIZE;
      //printf("alloc_ib: src_inst_id(%llx) dst_inst_id(%llx) idx(%d) size(%lu)\n", inst_pair.first.id, inst_pair.second.id, idx, ib_size);
      if (ID(tgt_mem).memory.owner_node == gasnet_mynode()) {
        // create local intermediate buffer
        IBAllocRequest* ib_req
          = new IBAllocRequest(gasnet_mynode(), this, idx, inst_pair.first.id,
                               inst_pair.second.id, ib_size);
        ib_req_queue->enqueue_request(tgt_mem, ib_req);
      } else {
        // create remote intermediate buffer
        RemoteIBAllocRequestAsync::send_request(ID(tgt_mem).memory.owner_node, tgt_mem, this, idx, inst_pair.first.id, inst_pair.second.id, ib_size);
      }
    }

    void CopyRequest::handle_ib_response(int idx, InstPair inst_pair, size_t ib_size, off_t ib_offset)
    {
      AutoHSLLock al(ib_mutex);
      IBByInst::iterator ib_it = ib_by_inst.find(inst_pair);
      assert(ib_it != ib_by_inst.end());
      IBVec& ibvec = ib_it->second;
      assert((int)ibvec.size() > idx);
      ibvec[idx].size = ib_size;
      ibvec[idx].offset = ib_offset;
      ibvec[idx].fence->mark_finished(true);
    }

    bool CopyRequest::check_readiness(bool just_check, DmaRequestQueue *rq)
    {
      if(state == STATE_INIT)
	state = STATE_METADATA_FETCH;

      // remember which queue we're going to be assigned to if we sleep
      waiter.req = this;
      waiter.queue = rq;

      // make sure our node has all the meta data it needs, but don't take more than one lock
      //  at a time
      if(state == STATE_METADATA_FETCH) {
	// index space first
	if(domain.get_dim() == 0) {
	  IndexSpaceImpl *is_impl = get_runtime()->get_index_space_impl(domain.get_index_space());
	  if(!is_impl->locked_data.valid) {
	    log_dma.debug("dma request %p - no index space metadata yet", this);
	    if(just_check) return false;

	    Event e = is_impl->lock.acquire(1, false, ReservationImpl::ACQUIRE_BLOCKING);
	    if(e.has_triggered()) {
	      log_dma.debug("request %p - index space metadata invalid - instant trigger", this);
	      is_impl->lock.release();
	    } else {
	      log_dma.debug("request %p - index space metadata invalid - sleeping on lock " IDFMT "", this, is_impl->lock.me.id);
	      waiter.sleep_on_event(e, is_impl->lock.me);
	      return false;
	    }
	  }

          // we need more than just the metadata - we also need the valid mask
          {
            Event e = is_impl->request_valid_mask();
            if(!e.has_triggered()) {
              log_dma.debug() << "request " << (void *)this << " - valid mask needed for index space "
			      << domain.get_index_space() << " - sleeping on event " << e;
	      waiter.sleep_on_event(e);
              return false;
            }
          }
	}

	// now go through all instance pairs
	for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	  RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(it->first.first);
	  RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(it->first.second);

	  {
	    Event e = src_impl->request_metadata();
	    if(!e.has_triggered()) {
	      if(just_check) {
		log_dma.debug("dma request %p - no src instance (" IDFMT ") metadata yet", this, src_impl->me.id);
		return false;
	      }
	      log_dma.debug() << "request " << (void *)this << " - src instance metadata invalid - sleeping on event " << e;
	      waiter.sleep_on_event(e);
	      return false;
	    }
	  }

	  {
	    Event e = dst_impl->request_metadata();
	    if(!e.has_triggered()) {
	      if(just_check) {
		log_dma.debug("dma request %p - no dst instance (" IDFMT ") metadata yet", this, dst_impl->me.id);
		return false;
	      }
	      log_dma.debug() << "request " << (void *)this << " - dst instance metadata invalid - sleeping on event " << e;
	      waiter.sleep_on_event(e);
	      return false;
	    }
	  }
	}

	// if we got all the way through, we've got all the metadata we need
	state = STATE_DST_FETCH;
      }

      // make sure the destination has fetched metadata
      if(state == STATE_DST_FETCH) {
        Memory tgt_mem = get_runtime()->get_instance_impl(oas_by_inst->begin()->first.second)->memory;
        gasnet_node_t tgt_node = ID(tgt_mem).memory.owner_node;
        if ((domain.get_dim() == 0) && (tgt_node != gasnet_mynode())) {
          if (tgt_fetch_completion == Event::NO_EVENT) {
            tgt_fetch_completion = GenEventImpl::create_genevent()->current_event();
            Realm::ValidMaskFetchMessage::send_request(tgt_node,
                                                domain.get_index_space(),
                                                tgt_fetch_completion);
          }

          if (!tgt_fetch_completion.has_triggered()) {
            if (just_check) return false;
            waiter.sleep_on_event(tgt_fetch_completion);
            return false;
          }
        }
        state = STATE_GEN_PATH;
      }

      if(state == STATE_GEN_PATH) {
        log_dma.debug("generate paths");
        Memory src_mem = get_runtime()->get_instance_impl(oas_by_inst->begin()->first.first)->memory;
        Memory dst_mem = get_runtime()->get_instance_impl(oas_by_inst->begin()->first.second)->memory;
        find_shortest_path(src_mem, dst_mem, mem_path);
        ib_req->mark_ready();
        ib_req->mark_started();
        // Pass 1: create IBInfo blocks
        for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
          AutoHSLLock al(ib_mutex);
          IBVec& ib_vec = ib_by_inst[it->first];
          assert(ib_vec.size() == 0);
          for (size_t i = 1; i < mem_path.size() - 1; i++) {
            IBInfo ib_info;
            ib_info.memory = mem_path[i];
            ib_info.fence = new IBFence(ib_req);
            ib_req->add_async_work_item(ib_info.fence); 
            ib_vec.push_back(ib_info);
          }
        }
        // Pass 2: send ib allocation requests
        for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
          for (size_t i = 1; i < mem_path.size() - 1; i++) {
            alloc_intermediate_buffer(it->first, mem_path[i], i - 1);
          }
        }
        ib_req->mark_finished(true);
        state = STATE_ALLOC_IB;
      }

      if(state == STATE_ALLOC_IB) {
        log_dma.debug("wait for the ib allocations to complete");
        if (ib_completion.has_triggered()) {
          state = STATE_BEFORE_EVENT;
        } else {
          if (just_check) return false;
          waiter.sleep_on_event(ib_completion);
          return false;
        }
      }

      // make sure our functional precondition has occurred
      if(state == STATE_BEFORE_EVENT) {
	// has the before event triggered?  if not, wait on it
	bool poisoned = false;
	if(before_copy.has_triggered_faultaware(poisoned)) {
	  if(poisoned) {
	    log_dma.debug("request %p - poisoned precondition", this);
	    handle_poisoned_precondition(before_copy);
	    return true;  // not enqueued, but never going to be
	  } else {
	    log_dma.debug("request %p - before event triggered", this);
	    state = STATE_READY;
	  }
	} else {
	  log_dma.debug("request %p - before event not triggered", this);
	  if(just_check) return false;

	  log_dma.debug("request %p - sleeping on before event", this);
	  waiter.sleep_on_event(before_copy);
	  return false;
	}
      }

      if(state == STATE_READY) {
	log_dma.debug("request %p ready", this);
	if(just_check) return true;

	state = STATE_QUEUED;
	// <NEWDMA>
	mark_ready();
	perform_dma();
	return true;
	// </NEWDMA>
	assert(rq != 0);
	log_dma.debug("request %p enqueued", this);

	// once we're enqueued, we may be deleted at any time, so no more
	//  references
	rq->enqueue_request(this);
	return true;
      }

      if(state == STATE_QUEUED)
	return true;

      assert(0);
      return false;
    }

    namespace RangeExecutors {
      class Memcpy {
      public:
	Memcpy(void *_dst_base, const void *_src_base, size_t _elmt_size)
	  : dst_base((char *)_dst_base), src_base((const char *)_src_base),
	    elmt_size(_elmt_size) {}

	template <class T>
	Memcpy(T *_dst_base, const T *_src_base)
	  : dst_base((char *)_dst_base), src_base((const char *)_src_base),
	    elmt_size(sizeof(T)) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	  memcpy(dst_base + byte_offset,
		 src_base + byte_offset,
		 byte_count);
	}

      protected:
	char *dst_base;
	const char *src_base;
	size_t elmt_size;
      };

      class GasnetPut {
      public:
	GasnetPut(MemoryImpl *_tgt_mem, off_t _tgt_offset,
		  const void *_src_ptr, size_t _elmt_size)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_ptr((const char *)_src_ptr), elmt_size(_elmt_size) {}

        virtual ~GasnetPut(void) { }

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
	  tgt_mem->put_bytes(tgt_offset + byte_offset,
			     src_ptr + byte_offset,
			     byte_count);
	}

      protected:
	MemoryImpl *tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
      };

      class GasnetPutBatched {
      public:
	GasnetPutBatched(MemoryImpl *_tgt_mem, off_t _tgt_offset,
			 const void *_src_ptr,
			 size_t _elmt_size)
	  : tgt_mem((GASNetMemory *)_tgt_mem), tgt_offset(_tgt_offset),
	    src_ptr((const char *)_src_ptr), elmt_size(_elmt_size) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
	  offsets.push_back(tgt_offset + byte_offset);
	  srcs.push_back(src_ptr + byte_offset);
	  sizes.push_back(byte_count);
	}

	void finish(void)
	{
	  if(offsets.size() > 0) {
	    DetailedTimer::ScopedPush sp(TIME_SYSTEM);
	    tgt_mem->put_batch(offsets.size(),
			       &offsets[0],
			       &srcs[0],
			       &sizes[0]);
	  }
	}

      protected:
	GASNetMemory *tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
	std::vector<off_t> offsets;
	std::vector<const void *> srcs;
	std::vector<size_t> sizes;
      };

      class GasnetPutReduce : public GasnetPut {
      public:
	GasnetPutReduce(MemoryImpl *_tgt_mem, off_t _tgt_offset,
			const ReductionOpUntyped *_redop, bool _redfold,
			const void *_src_ptr, size_t _elmt_size)
	  : GasnetPut(_tgt_mem, _tgt_offset, _src_ptr, _elmt_size),
	    redop(_redop), redfold(_redfold) {}

        virtual ~GasnetPutReduce(void) { }

	void do_span(int offset, int count)
	{
	  assert(redfold == false);
	  off_t tgt_byte_offset = offset * redop->sizeof_lhs;
	  off_t src_byte_offset = offset * elmt_size;
	  assert(elmt_size == redop->sizeof_rhs);

	  char buffer[1024];
	  assert(redop->sizeof_lhs <= 1024);

	  for(int i = 0; i < count; i++) {
	    tgt_mem->get_bytes(tgt_offset + tgt_byte_offset,
			       buffer,
			       redop->sizeof_lhs);

	    redop->apply(buffer, src_ptr + src_byte_offset, 1, true);
	      
	    tgt_mem->put_bytes(tgt_offset + tgt_byte_offset,
			       buffer,
			       redop->sizeof_lhs);
	  }
	}

      protected:
	const ReductionOpUntyped *redop;
	bool redfold;
      };

      class GasnetPutRedList : public GasnetPut {
      public:
	GasnetPutRedList(MemoryImpl *_tgt_mem, off_t _tgt_offset,
			 ReductionOpID _redopid,
			 const ReductionOpUntyped *_redop,
			 const void *_src_ptr, size_t _elmt_size)
	  : GasnetPut(_tgt_mem, _tgt_offset, _src_ptr, _elmt_size),
	    redopid(_redopid), redop(_redop) {}

        virtual ~GasnetPutRedList(void) { }

	void do_span(int offset, int count)
	{
	  if(count == 0) return;
	  assert(offset == 0); // too lazy to do pointer math on _src_ptr
	  unsigned *ptrs = new unsigned[count];
	  redop->get_list_pointers(ptrs, src_ptr, count);

	  // now figure out how many reductions go to each node
	  unsigned *nodecounts = new unsigned[gasnet_nodes()];
	  for(unsigned i = 0; i < gasnet_nodes(); i++)
	    nodecounts[i] = 0;

	  for(int i = 0; i < count; i++) {
	    off_t elem_offset = tgt_offset + ptrs[i] * redop->sizeof_lhs;
	    int home_node = tgt_mem->get_home_node(elem_offset, redop->sizeof_lhs);
	    assert(home_node >= 0);
	    ptrs[i] = home_node;
	    nodecounts[home_node]++;
	  }

	  size_t max_entries_per_msg = (1 << 20) / redop->sizeof_list_entry;
	  char *entry_buffer = new char[max_entries_per_msg * redop->sizeof_list_entry];

	  for(unsigned i = 0; i < gasnet_nodes(); i++) {
	    unsigned pos = 0;
	    for(int j = 0; j < count; j++) {
	      //printf("S: [%d] = %d\n", j, ptrs[j]);
	      if(ptrs[j] != i) continue;

	      memcpy(entry_buffer + (pos * redop->sizeof_list_entry),
		     ((const char *)src_ptr) + (j * redop->sizeof_list_entry),
		     redop->sizeof_list_entry);
	      pos++;

	      if(pos == max_entries_per_msg) {
		if(i == gasnet_mynode()) {
		  tgt_mem->apply_reduction_list(tgt_offset, redop, pos,
						entry_buffer);
		} else {
		  do_remote_apply_red_list(i, tgt_mem->me, tgt_offset,
					   redopid, 
					   entry_buffer, pos * redop->sizeof_list_entry, 0);
		}
		pos = 0;
	      }
	    }
	    if(pos > 0) {
	      if(i == gasnet_mynode()) {
		tgt_mem->apply_reduction_list(tgt_offset, redop, pos,
					      entry_buffer);
	      } else {
		do_remote_apply_red_list(i, tgt_mem->me, tgt_offset,
					 redopid, 
					 entry_buffer, pos * redop->sizeof_list_entry, 0);
	      }
	    }
	  }

	  delete[] entry_buffer;
	  delete[] ptrs;
	  delete[] nodecounts;
	}

      protected:
	ReductionOpID redopid;
	const ReductionOpUntyped *redop;
      };

      class GasnetGet {
      public:
	GasnetGet(void *_tgt_ptr,
		  MemoryImpl *_src_mem, off_t _src_offset,
		  size_t _elmt_size)
	  : tgt_ptr((char *)_tgt_ptr), src_mem(_src_mem),
	    src_offset(_src_offset), elmt_size(_elmt_size) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
	  DetailedTimer::ScopedPush sp(TIME_SYSTEM);
	  src_mem->get_bytes(src_offset + byte_offset,
			     tgt_ptr + byte_offset,
			     byte_count);
	}

      protected:
	char *tgt_ptr;
	MemoryImpl *src_mem;
	off_t src_offset;
	size_t elmt_size;
      };

      class GasnetGetBatched {
      public:
	GasnetGetBatched(void *_tgt_ptr,
			 MemoryImpl *_src_mem, off_t _src_offset,
			 size_t _elmt_size)
	  : tgt_ptr((char *)_tgt_ptr), src_mem((GASNetMemory *)_src_mem),
	    src_offset(_src_offset), elmt_size(_elmt_size) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
	  offsets.push_back(src_offset + byte_offset);
	  dsts.push_back(tgt_ptr + byte_offset);
	  sizes.push_back(byte_count);
	}

	void finish(void)
	{
	  if(offsets.size() > 0) {
	    DetailedTimer::ScopedPush sp(TIME_SYSTEM);
	    src_mem->get_batch(offsets.size(),
			       &offsets[0],
			       &dsts[0],
			       &sizes[0]);
	  }
	}

      protected:
	char *tgt_ptr;
	GASNetMemory *src_mem;
	off_t src_offset;
	size_t elmt_size;
	std::vector<off_t> offsets;
	std::vector<void *> dsts;
	std::vector<size_t> sizes;
      };

      class GasnetGetAndPut {
      public:
	GasnetGetAndPut(MemoryImpl *_tgt_mem, off_t _tgt_offset,
			MemoryImpl *_src_mem, off_t _src_offset,
			size_t _elmt_size)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_mem(_src_mem), src_offset(_src_offset), elmt_size(_elmt_size) {}

	static const size_t CHUNK_SIZE = 16384;

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;

	  while(byte_count > CHUNK_SIZE) {
	    src_mem->get_bytes(src_offset + byte_offset, chunk, CHUNK_SIZE);
	    tgt_mem->put_bytes(tgt_offset + byte_offset, chunk, CHUNK_SIZE);
	    byte_offset += CHUNK_SIZE;
	    byte_count -= CHUNK_SIZE;
	  }
	  if(byte_count > 0) {
	    src_mem->get_bytes(src_offset + byte_offset, chunk, byte_count);
	    tgt_mem->put_bytes(tgt_offset + byte_offset, chunk, byte_count);
	  }
	}

      protected:
	MemoryImpl *tgt_mem;
	off_t tgt_offset;
	MemoryImpl *src_mem;
	off_t src_offset;
	size_t elmt_size;
	char chunk[CHUNK_SIZE];
      };

#ifdef DEAD_DMA_CODE
      class RemoteWrite {
      public:
	RemoteWrite(Memory _tgt_mem, off_t _tgt_offset,
		    const void *_src_ptr, size_t _elmt_size,
		    Event _event)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_ptr((const char *)_src_ptr), elmt_size(_elmt_size),
	    event(_event), span_count(0) {}

	void do_span(int offset, int count)
	{
	  // if this isn't the first span, push the previous one out before
	  //  we overwrite it
	  if(span_count > 0)
	    really_do_span(false);

	  span_count++;
	  prev_offset = offset;
	  prev_count = count;
	}

	Event finish(void)
	{
	  log_dma.debug("remote write done with %d spans", span_count);
	  // if we got any spans, the last one is still waiting to go out
	  if(span_count > 0) {
	    really_do_span(true);
	    return Event::NO_EVENT; // recipient will trigger the event
	  }

	  return event;
	}

      protected:
	void really_do_span(bool last)
	{
	  off_t byte_offset = prev_offset * elmt_size;
	  size_t byte_count = prev_count * elmt_size;

	  // if we don't have an event for our completion, we need one now
	  if(!event.exists())
	    event = GenEventImpl::create_genevent()->current_event();

	  DetailedTimer::ScopedPush sp(TIME_SYSTEM);
	  do_remote_write(tgt_mem, tgt_offset + byte_offset,
			  src_ptr + byte_offset, byte_count,
			  0, last ? event : Event::NO_EVENT);
	}

	Memory tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
	Event event;
	int span_count;
	int prev_offset, prev_count;
      };
#endif

    }; // namespace RangeExecutors

    // helper function to figure out which field we're in
    void find_field_start(const std::vector<size_t>& field_sizes, off_t byte_offset,
				 size_t size, off_t& field_start, int& field_size)
    {
      off_t start = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++) {
	assert((*it) > 0);
	if(byte_offset < (off_t)(*it)) {
	  assert((byte_offset + size) <= (*it));
	  field_start = start;
	  field_size = (*it);
	  return;
	}
	start += (*it);
	byte_offset -= (*it);
      }
      assert(0);
    }

    class RemoteWriteInstPairCopier : public InstPairCopier {
    public:
      RemoteWriteInstPairCopier(RegionInstance src_inst, RegionInstance dst_inst,
                                OASVec &_oas_vec)
	: src_acc(src_inst.get_accessor()), dst_acc(dst_inst.get_accessor()),
          oas_vec(_oas_vec)
      {}

      virtual ~RemoteWriteInstPairCopier(void) { }

      virtual void copy_field(off_t src_index, off_t dst_index, off_t elem_count,
                              unsigned offset_index)
      {
        off_t src_offset = oas_vec[offset_index].src_offset;
        off_t dst_offset = oas_vec[offset_index].dst_offset;
        unsigned bytes = oas_vec[offset_index].size;
	char buffer[1024];

	for(int i = 0; i < elem_count; i++) {
	  src_acc.read_untyped(ptr_t(src_index + i), buffer, bytes, src_offset);
          if(0 && i == 0) {
            printf("remote write: (%zd:%zd->%zd:%zd) %d bytes:",
                   (ssize_t)(src_index + i), (ssize_t)src_offset,
                   (ssize_t)(dst_index + i), (ssize_t)dst_offset, bytes);
            for(unsigned j = 0; j < bytes; j++)
              printf(" %02x", (unsigned char)(buffer[j]));
            printf("\n");
          }
	  dst_acc.write_untyped(ptr_t(dst_index + i), buffer, bytes, dst_offset);
	}
      }

      virtual void flush(void) {}

    protected:
      RegionAccessor<AccessorType::Generic> src_acc;
      RegionAccessor<AccessorType::Generic> dst_acc;
      OASVec &oas_vec;
    };

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemPairCopier
  //

    MemPairCopier::MemPairCopier(void) 
      : total_reqs(0), total_bytes(0)
    { 
    }

    MemPairCopier::~MemPairCopier(void)
    {
    }

    // default behavior of flush is just to report bytes (maybe)
    void MemPairCopier::flush(DmaRequest *req)
    {
#ifdef EVENT_GRAPH_TRACE
      log_event_graph.debug("Copy Size: (" IDFMT ",%d) %ld",
			    after_copy.id, after_copy.gen, total_bytes);
      report_bytes(after_copy);
#endif
    }

    void MemPairCopier::record_bytes(size_t bytes)
    {
      total_reqs++;
      total_bytes += bytes;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MemPairCopierFactory
  //

    MemPairCopierFactory::MemPairCopierFactory(const std::string& _name)
      : name(_name)
    {
    }

    MemPairCopierFactory::~MemPairCopierFactory(void)
    {
    }

    const std::string& MemPairCopierFactory::get_name(void) const
    {
      return name;
    }


    class BufferedMemPairCopier : public MemPairCopier {
    public:
      BufferedMemPairCopier(Memory _src_mem, Memory _dst_mem, size_t _buffer_size = 32768)
	: buffer_size(_buffer_size)
      {
	src_mem = get_runtime()->get_memory_impl(_src_mem);
	dst_mem = get_runtime()->get_memory_impl(_dst_mem);
	buffer = new char[buffer_size];
      }

      virtual ~BufferedMemPairCopier(void)
      {
	delete[] buffer;
      }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<BufferedMemPairCopier>(this, src_inst, 
                                                          dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("buffered copy of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
	while(bytes > buffer_size) {
	  src_mem->get_bytes(src_offset, buffer, buffer_size);
	  dst_mem->put_bytes(dst_offset, buffer, buffer_size);
	  src_offset += buffer_size;
	  dst_offset += buffer_size;
	  bytes -= buffer_size;
          record_bytes(buffer_size);
	}
	if(bytes > 0) {
	  src_mem->get_bytes(src_offset, buffer, bytes);
	  dst_mem->put_bytes(dst_offset, buffer, bytes);
          record_bytes(bytes);
	}
      }

      // default behavior of 2D copy is to unroll to 1D copies
      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
	while(lines-- > 0) {
	  copy_span(src_offset, dst_offset, bytes);
	  src_offset += src_stride;
	  dst_offset += dst_stride;
	}
      }

    protected:
      size_t buffer_size;
      MemoryImpl *src_mem, *dst_mem;
      char *buffer;
    };
     
    class MemcpyMemPairCopier : public MemPairCopier {
    public:
      MemcpyMemPairCopier(Memory _src_mem, Memory _dst_mem)
      {
	MemoryImpl *src_impl = get_runtime()->get_memory_impl(_src_mem);
	src_base = (const char *)(src_impl->get_direct_ptr(0, src_impl->size));
	assert(src_base);

	MemoryImpl *dst_impl = get_runtime()->get_memory_impl(_dst_mem);
	dst_base = (char *)(dst_impl->get_direct_ptr(0, dst_impl->size));
	assert(dst_base);
      }

      virtual ~MemcpyMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<MemcpyMemPairCopier>(this, src_inst, 
                                                          dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("memcpy of %zd bytes\n", bytes);
	memcpy(dst_base + dst_offset, src_base + src_offset, bytes);
        record_bytes(bytes);
      }

      // default behavior of 2D copy is to unroll to 1D copies
      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
	while(lines-- > 0) {
	  copy_span(src_offset, dst_offset, bytes);
	  src_offset += src_stride;
	  dst_offset += dst_stride;
	}
      }

    protected:
      const char *src_base;
      char *dst_base;
    };

    class LocalReductionMemPairCopier : public MemPairCopier {
    public:
      LocalReductionMemPairCopier(Memory _src_mem, Memory _dst_mem,
				  ReductionOpID redop_id, bool _fold)
      {
	MemoryImpl *src_impl = get_runtime()->get_memory_impl(_src_mem);
	src_base = (const char *)(src_impl->get_direct_ptr(0, src_impl->size));
	assert(src_base);

	MemoryImpl *dst_impl = get_runtime()->get_memory_impl(_dst_mem);
	dst_base = (char *)(dst_impl->get_direct_ptr(0, dst_impl->size));
	assert(dst_base);

	redop = get_runtime()->reduce_op_table[redop_id];
	fold = _fold;
      }

      virtual ~LocalReductionMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<LocalReductionMemPairCopier>(this, src_inst, 
									dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("reduction of %zd bytes\n", bytes);
	assert((bytes % redop->sizeof_rhs) == 0);
	if(fold)
	  redop->fold(dst_base + dst_offset, src_base + src_offset,
		      bytes / redop->sizeof_rhs, false /*non-exclusive*/);
	else
	  redop->apply(dst_base + dst_offset, src_base + src_offset,
		       bytes / redop->sizeof_rhs, false /*non-exclusive*/);
      }

      // default behavior of 2D copy is to unroll to 1D copies
      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
	// two cases here:
	// 1) if bytes == sizeof_rhs, we can use the apply/fold_strided calls
	if(bytes == redop->sizeof_rhs) {
	  if(fold)
	    redop->fold_strided(dst_base + dst_offset, src_base + src_offset,
				src_stride, dst_stride, lines, false /*non-exclusive*/);
	  else
	    redop->apply_strided(dst_base + dst_offset, src_base + src_offset,
				 src_stride, dst_stride, lines, false /*non-exclusive*/);
	  return;
	}

	// 2) with multiple elements per line, have to do a apply/fold call per line
	while(lines-- > 0) {
	  copy_span(src_offset, dst_offset, bytes);
	  src_offset += src_stride;
	  dst_offset += dst_stride;
	}
      }

    protected:
      const char *src_base;
      char *dst_base;
      const ReductionOpUntyped *redop;
      bool fold;
    };

    class BufferedReductionMemPairCopier : public MemPairCopier {
    public:
      BufferedReductionMemPairCopier(Memory _src_mem, Memory _dst_mem,
				     ReductionOpID redop_id, bool _fold,
				     size_t _buffer_size = 1024) // in elements
	: buffer_size(_buffer_size)
      {
	src_mem = get_runtime()->get_memory_impl(_src_mem);
	dst_mem = get_runtime()->get_memory_impl(_dst_mem);
	redop = get_runtime()->reduce_op_table[redop_id];
	fold = _fold;

	src_buffer = new char[buffer_size * redop->sizeof_rhs];
	dst_buffer = new char[buffer_size * (fold ? redop->sizeof_rhs : redop->sizeof_lhs)];
      }

      virtual ~BufferedReductionMemPairCopier(void)
      {
	delete[] src_buffer;
	delete[] dst_buffer;
      }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<BufferedReductionMemPairCopier>(this, src_inst, 
									   dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("buffered copy of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
	size_t dst_size = fold ? redop->sizeof_rhs : redop->sizeof_lhs;

	assert((bytes % redop->sizeof_rhs) == 0);
	size_t elems = bytes / redop->sizeof_rhs;

	while(elems > 0) {
	  // figure out how many elements we can do at a time
	  size_t count = (elems > buffer_size) ? buffer_size : elems;

	  // fetch source and dest data into buffers
	  src_mem->get_bytes(src_offset, src_buffer, count * redop->sizeof_rhs);
	  dst_mem->get_bytes(dst_offset, dst_buffer, count * dst_size);

	  // apply reduction to local buffers
	  if(fold)
	    redop->fold(dst_buffer, src_buffer, count, true /*exclusive*/);
	  else
	    redop->apply(dst_buffer, src_buffer, count, true /*exclusive*/);

	  dst_mem->put_bytes(dst_offset, dst_buffer, count * dst_size);

	  src_offset += count * redop->sizeof_rhs;
	  dst_offset += count * dst_size;
	  elems -= count;
	}
      }

      // default behavior of 2D copy is to unroll to 1D copies
      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
	while(lines-- > 0) {
	  copy_span(src_offset, dst_offset, bytes);
	  src_offset += src_stride;
	  dst_offset += dst_stride;
	}
      }

    protected:
      size_t buffer_size;
      MemoryImpl *src_mem, *dst_mem;
      char *src_buffer;
      char *dst_buffer;
      const ReductionOpUntyped *redop;
      bool fold;
    };
     
#if 0
    // a MemPairCopier that keeps a list of events for component copies and doesn't trigger
    //  the completion event until they're all done
    class DelayedMemPairCopierX : public MemPairCopier {
    public:
      DelayedMemPairCopierX(void) {}

      virtual ~DelayedMemPairCopierX(void) {}
      
      virtual void flush(Event after_copy)
      {
	// create a merged event for all the copies and used that to perform a delayed trigger
	//  of the after_copy event (if it exists)
	if(after_copy.exists()) {
	  if(events.size() > 0) {
	    Event merged = GenEventImpl::merge_events(events);

	    // deferred trigger based on this merged event
	    get_runtime()->get_genevent_impl(after_copy)->trigger(after_copy.gen, gasnet_mynode(), merged);
	  } else {
	    // no actual copies occurred, so manually trigger event ourselves
	    get_runtime()->get_genevent_impl(after_copy)->trigger(after_copy.gen, gasnet_mynode());
	  }
	} else {
	  if(events.size() > 0) {
	    // we don't have an event we can use to signal completion, so wait on the events ourself
	    for(std::set<Event>::const_iterator it = events.begin();
		it != events.end();
		it++)
	      (*it).wait();
	  } else {
	    // nothing happened and we don't need to tell anyone - life is simple
	  }
	}
#ifdef EVENT_GRAPH_TRACE
        report_bytes(after_copy);
#endif
      }

    protected:
      std::set<Event> events;
    };
#endif
     
    static unsigned rdma_sequence_no = 1;

    class RemoteWriteMemPairCopier : public MemPairCopier {
    public:
      RemoteWriteMemPairCopier(Memory _src_mem, Memory _dst_mem)
      {
	src_mem = get_runtime()->get_memory_impl(_src_mem);
	src_base = (const char *)(src_mem->get_direct_ptr(0, src_mem->size));
	assert(src_base);

	dst_mem = get_runtime()->get_memory_impl(_dst_mem);
#ifdef TIME_REMOTE_WRITES
        span_time = 0;
        gather_time = 0;
#endif
	sequence_id = __sync_fetch_and_add(&rdma_sequence_no, 1);
	num_writes = 0;
      }

      virtual ~RemoteWriteMemPairCopier(void)
      {
      }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<RemoteWriteMemPairCopier>(this, src_inst, 
                                                              dst_inst, oas_vec);
      }

      struct PendingGather {
	off_t dst_start;
	off_t dst_size;
	SpanList src_spans;
      };

      // this is no longer limited by LMB size, so try for large blocks when
      //  possible
      static const size_t TARGET_XFER_SIZE = 4 << 20;

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("remote write of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#ifdef TIME_REMOTE_WRITES
        unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif
        record_bytes(bytes);
	if(bytes >= TARGET_XFER_SIZE) {
	  // large enough that we can transfer it by itself
#ifdef DEBUG_REMOTE_WRITES
	  printf("remote write of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#endif
	  num_writes += do_remote_write(dst_mem->me, dst_offset, src_base + src_offset, bytes, 
					sequence_id, false /* no copy */);
	} else {
	  // see if this can be added to an existing gather
	  PendingGather *g;
	  std::map<off_t, PendingGather *>::iterator it = gathers.find(dst_offset);
	  if(it != gathers.end()) {
	    g = it->second;
	    gathers.erase(it); // remove from the existing location - we'll re-add it (probably) with the updated offset
	  } else {
	    // nope, start a new one
	    g = new PendingGather;
	    g->dst_start = dst_offset;
	    g->dst_size = 0;
	  }

	  // sanity checks
	  assert((g->dst_start + g->dst_size) == dst_offset);

	  // now see if this particular span is disjoint or can be tacked on to the end of the last one
	  if(g->src_spans.size() > 0) {
	    SpanListEntry& last_span = g->src_spans.back();
	    if(((char *)(last_span.first) + last_span.second) == (src_base + src_offset)) {
	      // append
	      last_span.second += bytes;
	    } else {
	      g->src_spans.push_back(std::make_pair(src_base + src_offset, bytes));
	    }
	  } else {
	    // first span
	    g->src_spans.push_back(std::make_pair(src_base + src_offset, bytes));
	  }
	  g->dst_size += bytes;

	  // is this span big enough to push now?
	  if((size_t)g->dst_size >= TARGET_XFER_SIZE) {
	    // yes, copy it
	    copy_gather(g);
	    delete g;
	  } else {
	    // no, put it back in the pending list, keyed by the next dst_offset that'll match it
	    gathers.insert(std::make_pair(g->dst_start + g->dst_size, g));
	  }
	}
#ifdef TIME_REMOTE_WRITES
        unsigned long long stop = TimeStamp::get_current_time_in_micros();
        span_time += (stop - start);
#endif
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
	// case 1: although described as 2D, both src and dst coalesce
	if(((size_t)src_stride == bytes) && ((size_t)dst_stride == bytes)) {
	  copy_span(src_offset, dst_offset, bytes * lines);
	  return;
	}

	// case 2: if dst coalesces, we'll let the RDMA code do the gather for us -
	//  this won't merge with any other copies though
	if((size_t)dst_stride == bytes) {
#ifdef NEW2D_DEBUG
	  printf("GATHER copy\n");
#endif
	  num_writes += do_remote_write(dst_mem->me, dst_offset,
					src_base + src_offset, bytes, src_stride, lines,
					sequence_id, false /* no copy */);
          record_bytes(bytes * lines);
	  return;
	}
	
	// default is to unroll the lines here, and let the PendingGather code try to
	//  put things back in bigger pieces
	while(lines-- > 0) {
	  copy_span(src_offset, dst_offset, bytes);
	  src_offset += src_stride;
	  dst_offset += dst_stride;
	}
      }

      void copy_gather(const PendingGather *g)
      {
#ifdef TIME_REMOTE_WRITES
        unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif
        // special case: if there's only one source span, it's not actually
        //   a gather (so we don't need to make a copy)
        if(g->src_spans.size() == 1) {
          const SpanListEntry& span = *(g->src_spans.begin());
#ifdef DEBUG_REMOTE_WRITES
	  printf("remote write of %zd bytes (singleton " IDFMT ":%zd -> " IDFMT ":%zd)\n", span.second, src_mem->me.id, (char *)span.first - src_base, dst_mem->me.id, g->dst_start);
          printf("  datb[%p]: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                 span.first,
                 ((unsigned *)(span.first))[0], ((unsigned *)(span.first))[1],
                 ((unsigned *)(span.first))[2], ((unsigned *)(span.first))[3],
                 ((unsigned *)(span.first))[4], ((unsigned *)(span.first))[5],
                 ((unsigned *)(span.first))[6], ((unsigned *)(span.first))[7]);
#endif
	  // TODO: handle case where single write can include event trigger in message
	  //num_writes += do_remote_write(dst_mem->me, g->dst_start, span.first, span.second, trigger, false /* no copy */);
	  num_writes += do_remote_write(dst_mem->me, g->dst_start, span.first, span.second, 
					sequence_id, false /* no copy */);
          return;
        }

	// general case: do_remote_write knows how to take a span list now, so let it
	//  handle the actual gathering
#ifdef DEBUG_REMOTE_WRITES
	printf("remote write of %zd bytes (gather -> " IDFMT ":%zd), trigger=" IDFMT "/%d\n", g->dst_size, dst_mem->me.id, g->dst_start, trigger.id, trigger.gen);
#endif
	// TODO: handle case where single write can include event trigger in message
	num_writes += do_remote_write(dst_mem->me, g->dst_start, 
				      g->src_spans, g->dst_size,
				      sequence_id, false /* no copy - data won't change til copy event triggers */);
#ifdef TIME_REMOTE_WRITES
        unsigned long long stop = TimeStamp::get_current_time_in_micros();
        gather_time += (stop - start);
#endif
      }

      virtual void flush(DmaRequest *req)
      {
#ifdef TIME_REMOTE_WRITES
        size_t total_gathers = gathers.size();
        size_t total_spans = 0;
        for (std::map<off_t,PendingGather*>::const_iterator it = gathers.begin();
              it != gathers.end(); it++)
        {
          total_spans += it->second->src_spans.size();
        }
        unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif

	// do we have any pending gathers to push out?
        while(!gathers.empty()) {
	  std::map<off_t, PendingGather *>::iterator it = gathers.begin();
	  PendingGather *g = it->second;
	  gathers.erase(it);

	  copy_gather(g);
	  delete g;
	}

        // if we did any remote writes, we need a fence, and the DMA request
	//  needs to know about it
        if(total_reqs > 0) {
          Realm::RemoteWriteFence *fence = new Realm::RemoteWriteFence(req);
          req->add_async_work_item(fence);
          do_remote_fence(dst_mem->me, sequence_id, num_writes, fence);
        }

#ifdef TIME_REMOTE_WRITES
        unsigned long long stop = TimeStamp::get_current_time_in_micros();
        gather_time += (stop - start);
        printf("Remote Write: span time: %lld  gather time %lld "
                "total gathers %ld total spans %d\n", 
                span_time, gather_time, total_gathers, total_spans);
#endif

        MemPairCopier::flush(req);
      }

    protected:
      MemoryImpl *src_mem, *dst_mem;
      const char *src_base;
      std::map<off_t, PendingGather *> gathers;
#ifdef TIME_REMOTE_WRITES
      unsigned long long span_time;
      unsigned long long gather_time;
#endif
      unsigned sequence_id, num_writes;
    };
     
    class RemoteReduceMemPairCopier : public MemPairCopier {
    public:
      RemoteReduceMemPairCopier(Memory _src_mem, Memory _dst_mem,
				ReductionOpID _redop_id, bool _fold)
      {
	src_mem = get_runtime()->get_memory_impl(_src_mem);
	src_base = (const char *)(src_mem->get_direct_ptr(0, src_mem->size));
	assert(src_base);

	dst_mem = get_runtime()->get_memory_impl(_dst_mem);

	redop_id = _redop_id;
	redop = get_runtime()->reduce_op_table[redop_id];
	fold = _fold;

	sequence_id = __sync_fetch_and_add(&rdma_sequence_no, 1);
	num_writes = 0;
      }

      virtual ~RemoteReduceMemPairCopier(void)
      {
      }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<RemoteReduceMemPairCopier>(this, src_inst, 
								      dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("remote write of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#ifdef TIME_REMOTE_WRITES
        unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif

#ifdef DEBUG_REMOTE_WRITES
	printf("remote reduce of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#endif
	num_writes += do_remote_reduce(dst_mem->me, dst_offset, redop_id, fold,
				       src_base + src_offset, bytes / redop->sizeof_rhs,
				       redop->sizeof_rhs, redop->sizeof_lhs,
				       sequence_id, false /* no copy */);

        record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
	// TODO: figure out 2D coalescing for reduction case (sizes may not match)
	//if((src_stride == bytes) && (dst_stride == bytes)) {
	//  copy_span(src_offset, dst_offset, bytes * lines);
	//  return;
	//}
	
	// default is to unroll the lines here
	while(lines-- > 0) {
	  copy_span(src_offset, dst_offset, bytes);
	  src_offset += src_stride;
	  dst_offset += dst_stride;
	}
      }

      virtual void flush(DmaRequest *req)
      {
        // if we did any remote writes, we need a fence, and the DMA request
	//  needs to know about it
        if(total_reqs > 0) {
          Realm::RemoteWriteFence *fence = new Realm::RemoteWriteFence(req);
          req->add_async_work_item(fence);
          do_remote_fence(dst_mem->me, sequence_id, num_writes, fence);
        }

        MemPairCopier::flush(req);
      }

    protected:
      MemoryImpl *src_mem, *dst_mem;
      const char *src_base;
      unsigned sequence_id, num_writes;
      ReductionOpID redop_id;
      const ReductionOpUntyped *redop;
      bool fold;
    };

    class RemoteSerdezMemPairCopier : public MemPairCopier {
    public:
      RemoteSerdezMemPairCopier(Memory _src_mem, Memory _dst_mem,
				CustomSerdezID _serdez_id)
      {
	src_mem = get_runtime()->get_memory_impl(_src_mem);
	src_base = (const char *)(src_mem->get_direct_ptr(0, src_mem->size));
	assert(src_base);

	dst_mem = get_runtime()->get_memory_impl(_dst_mem);

	serdez_id = _serdez_id;
	serdez_op = get_runtime()->custom_serdez_table[serdez_id];

	sequence_id = __sync_fetch_and_add(&rdma_sequence_no, 1);
	num_writes = 0;
      }

      virtual ~RemoteSerdezMemPairCopier(void)
      {
      }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<RemoteSerdezMemPairCopier>(this, src_inst,
								      dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("remote write of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#ifdef TIME_REMOTE_WRITES
        unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif

#ifdef DEBUG_REMOTE_WRITES
	printf("remote serdez of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#endif
	num_writes += do_remote_serdez(dst_mem->me, dst_offset, serdez_id,
				       src_base + src_offset, bytes / serdez_op->sizeof_field_type, sequence_id);

        record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
	// TODO: figure out 2D coalescing for reduction case (sizes may not match)
	//if((src_stride == bytes) && (dst_stride == bytes)) {
	//  copy_span(src_offset, dst_offset, bytes * lines);
	//  return;
	//}

	// default is to unroll the lines here
	while(lines-- > 0) {
	  copy_span(src_offset, dst_offset, bytes);
	  src_offset += src_stride;
	  dst_offset += dst_stride;
	}
      }

      virtual void flush(DmaRequest *req)
      {
        // if we did any remote writes, we need a fence, and the DMA request
	//  needs to know about it
        if(total_reqs > 0) {
          Realm::RemoteWriteFence *fence = new Realm::RemoteWriteFence(req);
          req->add_async_work_item(fence);
          do_remote_fence(dst_mem->me, sequence_id, num_writes, fence);
        }

        MemPairCopier::flush(req);
      }

    protected:
      MemoryImpl *src_mem, *dst_mem;
      const char *src_base;
      unsigned sequence_id, num_writes;
      CustomSerdezID serdez_id;
      const CustomSerdezUntyped *serdez_op;
    };
    

    static AsyncFileIOContext *aio_context = 0;

#ifdef REALM_USE_KERNEL_AIO
    inline int io_setup(unsigned nr, aio_context_t *ctxp)
    {
      return syscall(__NR_io_setup, nr, ctxp);
    }

    inline int io_destroy(aio_context_t ctx)
    {
      return syscall(__NR_io_destroy, ctx);
    }

    inline int io_submit(aio_context_t ctx, long nr, struct iocb **iocbpp)
    {
      return syscall(__NR_io_submit, ctx, nr, iocbpp);
    }

    inline int io_getevents(aio_context_t ctx, long min_nr, long max_nr,
                            struct io_event *events, struct timespec *timeout)
    {
      return syscall(__NR_io_getevents, ctx, min_nr, max_nr, events, timeout);
    }

    class KernelAIOWrite : public AsyncFileIOContext::AIOOperation {
    public:
      KernelAIOWrite(aio_context_t aio_ctx,
                     int fd, size_t offset, size_t bytes,
		     const void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      aio_context_t ctx;
      struct iocb cb;
    };

    KernelAIOWrite::KernelAIOWrite(aio_context_t aio_ctx,
				   int fd, size_t offset, size_t bytes,
				   const void *buffer, Request* request)
    {
      completed = false;
      ctx = aio_ctx;
      memset(&cb, 0, sizeof(cb));
      cb.aio_data = (uint64_t)this;
      cb.aio_fildes = fd;
      cb.aio_lio_opcode = IOCB_CMD_PWRITE;
      cb.aio_buf = (uint64_t)buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void KernelAIOWrite::launch(void)
    {
      struct iocb *cbs[1];
      cbs[0] = &cb;
      log_aio.debug("write issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	io_submit(ctx, 1, cbs);
      assert(ret == 1);
    }

    bool KernelAIOWrite::check_completion(void)
    {
      return completed;
    }

    class KernelAIORead : public AsyncFileIOContext::AIOOperation {
    public:
      KernelAIORead(aio_context_t aio_ctx,
                     int fd, size_t offset, size_t bytes,
		     void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      aio_context_t ctx;
      struct iocb cb;
    };

    KernelAIORead::KernelAIORead(aio_context_t aio_ctx,
				 int fd, size_t offset, size_t bytes,
				 void *buffer, Request* request)
    {
      completed = false;
      ctx = aio_ctx;
      memset(&cb, 0, sizeof(cb));
      cb.aio_data = (uint64_t)this;
      cb.aio_fildes = fd;
      cb.aio_lio_opcode = IOCB_CMD_PREAD;
      cb.aio_buf = (uint64_t)buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void KernelAIORead::launch(void)
    {
      struct iocb *cbs[1];
      cbs[0] = &cb;
      log_aio.debug("read issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	io_submit(ctx, 1, cbs);
      assert(ret == 1);
    }

    bool KernelAIORead::check_completion(void)
    {
      return completed;
    }
#else
    class PosixAIOWrite : public AsyncFileIOContext::AIOOperation {
    public:
      PosixAIOWrite(int fd, size_t offset, size_t bytes,
		    const void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      struct aiocb cb;
    };

    PosixAIOWrite::PosixAIOWrite(int fd, size_t offset, size_t bytes,
				 const void *buffer, Request* request)
    {
      completed = false;
      memset(&cb, 0, sizeof(cb));
      cb.aio_fildes = fd;
      cb.aio_buf = (void *)buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void PosixAIOWrite::launch(void)
    {
      log_aio.debug("write issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	aio_write(&cb);
      assert(ret == 0);
    }

    bool PosixAIOWrite::check_completion(void)
    {
      int ret = aio_error(&cb);
      if(ret == EINPROGRESS) return false;
      log_aio.debug("write returned: op=%p cb=%p ret=%d", this, &cb, ret);
      assert(ret == 0);
      return true;
    }

    class PosixAIORead : public AsyncFileIOContext::AIOOperation {
    public:
      PosixAIORead(int fd, size_t offset, size_t bytes,
		   void *buffer, Request* request = NULL);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      struct aiocb cb;
    };

    PosixAIORead::PosixAIORead(int fd, size_t offset, size_t bytes,
			       void *buffer, Request* request)
    {
      completed = false;
      memset(&cb, 0, sizeof(cb));
      cb.aio_fildes = fd;
      cb.aio_buf = buffer;
      cb.aio_offset = offset;
      cb.aio_nbytes = bytes;
      req = request;
    }

    void PosixAIORead::launch(void)
    {
      log_aio.debug("read issued: op=%p cb=%p", this, &cb);
#ifndef NDEBUG
      int ret =
#endif
	aio_read(&cb);
      assert(ret == 0);
    }

    bool PosixAIORead::check_completion(void)
    {
      int ret = aio_error(&cb);
      if(ret == EINPROGRESS) return false;
      log_aio.debug("read returned: op=%p cb=%p ret=%d", this, &cb, ret);
      assert(ret == 0);
      return true;
    }
#endif

    class AIOFence : public Realm::Operation::AsyncWorkItem {
    public:
      AIOFence(Realm::Operation *_op) : Realm::Operation::AsyncWorkItem(_op) {}
      virtual void request_cancellation(void) {}
      virtual void print(std::ostream& os) const { os << "AIOFence"; }
    };

    class AIOFenceOp : public AsyncFileIOContext::AIOOperation {
    public:
      AIOFenceOp(DmaRequest *_req);
      virtual void launch(void);
      virtual bool check_completion(void);

    public:
      DmaRequest *req;
      AIOFence *f;
    };

    AIOFenceOp::AIOFenceOp(DmaRequest *_req)
    {
      completed = false;
      req = _req;
      f = new AIOFence(req);
      req->add_async_work_item(f);
    }

    void AIOFenceOp::launch(void)
    {
      log_aio.debug("fence launched: op=%p req=%p", this, req);
      completed = true;
    }

    bool AIOFenceOp::check_completion(void)
    {
      assert(completed);
      log_aio.debug("fence completed: op=%p req=%p", this, req);
      f->mark_finished(true /*successful*/);
      return true;
    }

    AsyncFileIOContext::AsyncFileIOContext(int _max_depth)
      : max_depth(_max_depth)
    {
#ifdef REALM_USE_KERNEL_AIO
      aio_ctx = 0;
#ifndef NDEBUG
      int ret =
#endif
	io_setup(max_depth, &aio_ctx);
      assert(ret == 0);
#endif
    }

    AsyncFileIOContext::~AsyncFileIOContext(void)
    {
      assert(pending_operations.empty());
      assert(launched_operations.empty());
#ifdef REALM_USE_KERNEL_AIO
#ifndef NDEBUG
      int ret =
#endif
	io_destroy(aio_ctx);
      assert(ret == 0);
#endif
    }

    void AsyncFileIOContext::enqueue_write(int fd, size_t offset, 
					   size_t bytes, const void *buffer,
                                           Request* req)
    {
#ifdef REALM_USE_KERNEL_AIO
      KernelAIOWrite *op = new KernelAIOWrite(aio_ctx,
					      fd, offset, bytes, buffer, req);
#else
      PosixAIOWrite *op = new PosixAIOWrite(fd, offset, bytes, buffer, req);
#endif
      {
	AutoHSLLock al(mutex);
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
    }

    void AsyncFileIOContext::enqueue_read(int fd, size_t offset, 
					  size_t bytes, void *buffer,
                                          Request* req)
    {
#ifdef REALM_USE_KERNEL_AIO
      KernelAIORead *op = new KernelAIORead(aio_ctx,
					    fd, offset, bytes, buffer, req);
#else
      PosixAIORead *op = new PosixAIORead(fd, offset, bytes, buffer, req);
#endif
      {
	AutoHSLLock al(mutex);
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
    }

    void AsyncFileIOContext::enqueue_fence(DmaRequest *req)
    {
      AIOFenceOp *op = new AIOFenceOp(req);
      {
	AutoHSLLock al(mutex);
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
    }

    bool AsyncFileIOContext::empty(void)
    {
      AutoHSLLock al(mutex);
      return launched_operations.empty();
    }

    long AsyncFileIOContext::available(void)
    {
      AutoHSLLock al(mutex);
      return (max_depth - launched_operations.size());
    }

    void AsyncFileIOContext::make_progress(void)
    {
      AutoHSLLock al(mutex);

      // first, reap as many events as we can - oldest first
#ifdef REALM_USE_KERNEL_AIO
      while(true) {
	struct io_event events[8];
	struct timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 0;  // no delay
	int ret = io_getevents(aio_ctx, 1, 8, events, &ts);
	if(ret == 0) break;
	log_aio.debug("io_getevents returned %d events", ret);
	for(int i = 0; i < ret; i++) {
	  AIOOperation *op = (AIOOperation *)(events[i].data);
	  log_aio.debug("io_getevents: event[%d] = %p", i, op);
	  op->completed = true;
	}
      }
#endif

      // now actually mark events completed in oldest-first order
      while(!launched_operations.empty()) {
	AIOOperation *op = launched_operations.front();
	if(!op->check_completion()) break;
	log_aio.debug("aio op completed: op=%p", op);
        // <NEW_DMA>
        if (op->req != NULL) {
          Request* request = (Request*)(op->req);
          request->xd->notify_request_read_done(request);
          request->xd->notify_request_write_done(request);
        }
        // </NEW_DMA>
	delete op;
	launched_operations.pop_front();
      }

      // finally, if there are any pending ops, and room for them, launch them
      while((launched_operations.size() < (size_t)max_depth) &&
	    !pending_operations.empty()) {
	AIOOperation *op = pending_operations.front();
	pending_operations.pop_front();
	op->launch();
	launched_operations.push_back(op);
      }
    }

    /*static*/
    AsyncFileIOContext* AsyncFileIOContext::get_singleton() {
      return aio_context;
    }

    // MemPairCopier from disk memory to cpu memory
    class DisktoCPUMemPairCopier : public MemPairCopier {
    public:
      DisktoCPUMemPairCopier(int _fd, Memory _dst_mem)
      {
        MemoryImpl *dst_impl = get_runtime()->get_memory_impl(_dst_mem);
        dst_base = (char *)(dst_impl->get_direct_ptr(0, dst_impl->size));
        assert(dst_base);
        fd = _fd;
      }

      virtual ~DisktoCPUMemPairCopier(void)
      {
      }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &osa_vec)
      {
        return new SpanBasedInstPairCopier<DisktoCPUMemPairCopier>(this, src_inst,
                                           dst_inst, osa_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	aio_context->enqueue_read(fd, src_offset, bytes,
				  dst_base + dst_offset);
	record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
                     off_t src_stride, off_t dst_stride, size_t lines)
      {
        while(lines-- > 0) {
          copy_span(src_offset, dst_offset, bytes);
          src_offset += src_stride;
          dst_offset += dst_stride;
        }
      }

      void flush(DmaRequest *req)
      {
	aio_context->enqueue_fence(req);
	MemPairCopier::flush(req);
      }

    protected:
      char *dst_base;
      int fd; // file descriptor
    };

    // MemPairCopier from disk memory to cpu memory
    class DiskfromCPUMemPairCopier : public MemPairCopier {
    public:
      DiskfromCPUMemPairCopier(Memory _src_mem, int _fd)
      { 
	MemoryImpl *src_impl = get_runtime()->get_memory_impl(_src_mem);
        src_base = (char *)(src_impl->get_direct_ptr(0, src_impl->size));
        assert(src_base);
        fd = _fd;
      }

      virtual ~DiskfromCPUMemPairCopier(void)
      {
      }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &osa_vec)
      {
        return new SpanBasedInstPairCopier<DiskfromCPUMemPairCopier>(this, src_inst,
                                           dst_inst, osa_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	aio_context->enqueue_write(fd, dst_offset, bytes,
				   src_base + src_offset);
	record_bytes(bytes);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
                     off_t src_stride, off_t dst_stride, size_t lines)
      {
        while(lines-- > 0) {
          copy_span(src_offset, dst_offset, bytes);
          src_offset += src_stride;
          dst_offset += dst_stride;
        }
      }

      void flush(DmaRequest *req)
      {
	aio_context->enqueue_fence(req);
	MemPairCopier::flush(req);
      }

    protected:
      char *src_base;
      int fd; // file descriptor
    };

    class FilefromCPUMemPairCopier : public MemPairCopier {
    public:
      class FileWriteCopier : public InstPairCopier
      {
      public:
        enum {max_nr = 1};
        FileWriteCopier(int _fd, char* _base, RegionInstance src_inst,
                        RegionInstance dst_inst, OASVec &oas_vec,
			FilefromCPUMemPairCopier *_mpc)
        {
	  fd = _fd;
          inst_copier = new SpanBasedInstPairCopier<FileWriteCopier>(this, src_inst, dst_inst, oas_vec);
          src_base = _base;
	  mpc = _mpc;
        }

        ~FileWriteCopier(void)
        {
          delete inst_copier;
        }

        virtual void copy_field(off_t src_index, off_t dst_index, off_t elem_count,
                                unsigned offset_index)
        {
          inst_copier->copy_field(src_index, dst_index, elem_count, offset_index);
        }

        virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t elem_count)
        {
          inst_copier->copy_all_fields(src_index, dst_index, elem_count);
        }

        virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t count_per_line,
  				     off_t src_stride, off_t dst_stride, off_t lines)
        {
          inst_copier->copy_all_fields(src_index, dst_index, count_per_line, src_stride, dst_stride, lines);
        }

        virtual void flush(void)
        {
          inst_copier->flush();
        }

        void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
        {
	  aio_context->enqueue_write(fd, dst_offset, bytes,
				     src_base + src_offset);
	  mpc->record_bytes(bytes);
        }

        void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
                       off_t src_stride, off_t dst_stride, size_t lines)
        {
          while(lines-- > 0) {
            copy_span(src_offset, dst_offset, bytes);
            src_offset += src_stride;
            dst_offset += dst_stride;
          }
        }

      protected:
	int fd;
        char *src_base;
        InstPairCopier* inst_copier;
	FilefromCPUMemPairCopier *mpc;
      };

      FilefromCPUMemPairCopier(Memory _src_mem, Memory _dst_mem)
      {
        MemoryImpl *src_impl = get_runtime()->get_memory_impl(_src_mem);
        src_base = (char *)(src_impl->get_direct_ptr(0, src_impl->size));
        dst_mem = (FileMemory*) get_runtime()->get_memory_impl(_dst_mem);
      }

      virtual ~FilefromCPUMemPairCopier(void) {}

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
        ID id(dst_inst);
        unsigned index = id.instance.inst_idx;
        int fd = dst_mem->get_file_des(index);
        return new FileWriteCopier(fd, src_base, src_inst, dst_inst, oas_vec, this);
      }

      void flush(DmaRequest *req)
      {
	aio_context->enqueue_fence(req);
	MemPairCopier::flush(req);
      }

    protected:
      char *src_base;
      FileMemory *dst_mem;
    };

    class FiletoCPUMemPairCopier : public MemPairCopier {
    public:
      class FileReadCopier : public InstPairCopier
      {
      public:
        enum {max_nr = 1};
    	FileReadCopier(int _fd, char* _base, RegionInstance src_inst,
                       RegionInstance dst_inst, OASVec &oas_vec, FiletoCPUMemPairCopier *_mpc)
        {
          fd = _fd;
          inst_copier = new SpanBasedInstPairCopier<FileReadCopier>(this, src_inst, dst_inst, oas_vec);
          dst_base = _base;
	  mpc = _mpc;
        }

        ~FileReadCopier(void)
        {
          delete inst_copier;
        }

        virtual void copy_field(off_t src_index, off_t dst_index, off_t elem_count,
                                unsigned offset_index)
        {
          inst_copier->copy_field(src_index, dst_index, elem_count, offset_index);
        }

        virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t elem_count)
        {
          inst_copier->copy_all_fields(src_index, dst_index, elem_count);
        }

        virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t count_per_line,
  				     off_t src_stride, off_t dst_stride, off_t lines)
        {
          inst_copier->copy_all_fields(src_index, dst_index, count_per_line, src_stride, dst_stride, lines);
        }

        virtual void flush(void)
        {
          inst_copier->flush();
        }

        void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
        {
	  aio_context->enqueue_read(fd, src_offset, bytes,
				    dst_base + dst_offset);
	  mpc->record_bytes(bytes);
        }

        void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
                       off_t src_stride, off_t dst_stride, size_t lines)
        {
          while(lines-- > 0) {
            copy_span(src_offset, dst_offset, bytes);
            src_offset += src_stride;
            dst_offset += dst_stride;
          }
        }

      protected:
	int fd;
        char *dst_base;
        InstPairCopier* inst_copier;
	FiletoCPUMemPairCopier *mpc;
      };

      FiletoCPUMemPairCopier(Memory _src_mem, Memory _dst_mem)
      {
        MemoryImpl *dst_impl = get_runtime()->get_memory_impl(_dst_mem);
        dst_base = (char *)(dst_impl->get_direct_ptr(0, dst_impl->size));
        src_mem = (FileMemory*) get_runtime()->get_memory_impl(_src_mem);
      }

      virtual ~FiletoCPUMemPairCopier(void) {}

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
        ID id(src_inst);
        unsigned index = id.instance.inst_idx;
        int fd = src_mem->get_file_des(index);
        return new FileReadCopier(fd, dst_base, src_inst, dst_inst, oas_vec, this);
      }

      void flush(DmaRequest *req)
      {
	aio_context->enqueue_fence(req);
	MemPairCopier::flush(req);
      }

    protected:
      char *dst_base;
      FileMemory *src_mem;
    };
     
    // most of the smarts from MemPairCopier::create_copier are now captured in the various factories

    class MemcpyMemPairCopierFactory : public MemPairCopierFactory {
    public:
      MemcpyMemPairCopierFactory(void)
	: MemPairCopierFactory("memcpy")
      {}

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold)
      {
	// non-reduction copies between anything SYSMEM and/or ZC
	//  (TODO: really should be anything with a direct CPU pointer, but GPUFBMemory
	//  returns non-null for that right now...)
	if(redop_id != 0)
	  return false;

	MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
	MemoryImpl::MemoryKind src_kind = src_impl->kind;
	
	if((src_kind != MemoryImpl::MKIND_SYSMEM) &&
	   (src_kind != MemoryImpl::MKIND_ZEROCOPY))
	  return false;

	MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
	MemoryImpl::MemoryKind dst_kind = dst_impl->kind;
	
	if((dst_kind != MemoryImpl::MKIND_SYSMEM) &&
	   (dst_kind != MemoryImpl::MKIND_ZEROCOPY))
	  return false;

	return true;
      }

      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold)
      {
	return new MemcpyMemPairCopier(src_mem, dst_mem);
      }
    };


    void create_builtin_dma_channels(Realm::RuntimeImpl *r)
    {
      r->add_dma_channel(new MemcpyMemPairCopierFactory);
    }

    MemPairCopier *MemPairCopier::create_copier(Memory src_mem, Memory dst_mem,
						ReductionOpID redop_id /*= 0*/,
						CustomSerdezID serdez_id /*= 0*/,
						bool fold /*= false*/)
    {
      // try to use new DMA channels first
      const std::vector<MemPairCopierFactory *>& channels = get_runtime()->get_dma_channels();
      for(std::vector<MemPairCopierFactory *>::const_iterator it = channels.begin();
	  it != channels.end();
	  it++) {
	if((*it)->can_perform_copy(src_mem, dst_mem, redop_id, fold)) {
	  // impls and kinds are just for logging now
	  MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
	  MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);

	  MemoryImpl::MemoryKind src_kind = src_impl->kind;
	  MemoryImpl::MemoryKind dst_kind = dst_impl->kind;

	  log_dma.info("copier: " IDFMT "(%d) -> " IDFMT "(%d) = %s",
		       src_mem.id, src_kind, dst_mem.id, dst_kind, (*it)->get_name().c_str());
	  return (*it)->create_copier(src_mem, dst_mem, redop_id, fold);
	}
      }

      // old style - various options in here are being turned into assert(0)'s as they are 
      //  replaced by DMA channel-provided copiers

      MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
      MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);

      MemoryImpl::MemoryKind src_kind = src_impl->kind;
      MemoryImpl::MemoryKind dst_kind = dst_impl->kind;

      log_dma.info("copier: " IDFMT "(%d) -> " IDFMT "(%d)", src_mem.id, src_kind, dst_mem.id, dst_kind);
      if(redop_id == 0) {
        if (serdez_id != 0) {
          // handle serdez cases, for now we only support remote serdez case
          if((dst_kind == MemoryImpl::MKIND_REMOTE) ||
             (dst_kind == MemoryImpl::MKIND_RDMA)) {
              assert(src_kind != MemoryImpl::MKIND_REMOTE);
              return new RemoteSerdezMemPairCopier(src_mem, dst_mem, serdez_id);
          }
          log_dma.warning("Unsupported serdez transfer case (" IDFMT " -> " IDFMT ")",
                          src_mem.id, dst_mem.id);
          assert(0);
        }
	// can we perform simple memcpy's?
	if(((src_kind == MemoryImpl::MKIND_SYSMEM) || (src_kind == MemoryImpl::MKIND_ZEROCOPY)) &&
	   ((dst_kind == MemoryImpl::MKIND_SYSMEM) || (dst_kind == MemoryImpl::MKIND_ZEROCOPY))) {
	  assert(0);
	  //return new MemcpyMemPairCopier(src_mem, dst_mem);
	}

        // can we perform transfer between disk and cpu memory
        if (((src_kind == MemoryImpl::MKIND_SYSMEM) || (src_kind == MemoryImpl::MKIND_ZEROCOPY)) &&
            (dst_kind == MemoryImpl::MKIND_DISK)) {
          // printf("Create DiskfromCPUMemPairCopier\n");
          int fd = ((DiskMemory *)dst_impl)->fd;
          return new DiskfromCPUMemPairCopier(src_mem, fd);
        }

        if ((src_kind == MemoryImpl::MKIND_DISK) &&
            ((dst_kind == MemoryImpl::MKIND_SYSMEM) || (dst_kind == MemoryImpl::MKIND_ZEROCOPY))) {
          // printf("Create DisktoCPUMemPairCopier\n");
          int fd = ((DiskMemory *)src_impl)->fd;
          return new DisktoCPUMemPairCopier(fd, dst_mem);
        }

        // can we perform transfer between cpu and file memory
        if (((src_kind == MemoryImpl::MKIND_SYSMEM) || (src_kind == MemoryImpl::MKIND_ZEROCOPY)) &&
            (dst_kind == MemoryImpl::MKIND_FILE)) {
          //printf("Create FilefromCPUMemPairCopier\n");
          return new FilefromCPUMemPairCopier(src_mem, dst_mem);
        }

        if ((src_kind == MemoryImpl::MKIND_FILE) &&
            ((dst_kind == MemoryImpl::MKIND_SYSMEM) || (dst_kind == MemoryImpl::MKIND_ZEROCOPY))) {
          //printf("Create FiletoCPUMemPairCopier\n");
          return new FiletoCPUMemPairCopier(src_mem, dst_mem);
        }

	// GPU FB-related copies should be handled by module-provided dma channels now
	if((src_kind == MemoryImpl::MKIND_GPUFB) || (dst_kind == MemoryImpl::MKIND_GPUFB)) {
	  assert(0);
	}

	// try as many things as we can think of
	if((dst_kind == MemoryImpl::MKIND_REMOTE) ||
	   (dst_kind == MemoryImpl::MKIND_RDMA)) {
	  assert(src_kind != MemoryImpl::MKIND_REMOTE);
	  return new RemoteWriteMemPairCopier(src_mem, dst_mem);
	}

	// fallback
	return new BufferedMemPairCopier(src_mem, dst_mem);
      } else {
	// reduction case
	// can we perform simple memcpy's?
	if(((src_kind == MemoryImpl::MKIND_SYSMEM) || (src_kind == MemoryImpl::MKIND_ZEROCOPY)) &&
	   ((dst_kind == MemoryImpl::MKIND_SYSMEM) || (dst_kind == MemoryImpl::MKIND_ZEROCOPY))) {
	  return new LocalReductionMemPairCopier(src_mem, dst_mem, redop_id, fold);
	}

	// reductions to a remote memory get shipped over there to be applied
	if((dst_kind == MemoryImpl::MKIND_REMOTE) ||
	   (dst_kind == MemoryImpl::MKIND_RDMA)) {
	  assert(src_kind != MemoryImpl::MKIND_REMOTE);
	  return new RemoteReduceMemPairCopier(src_mem, dst_mem, redop_id, fold);
	}

	// fallback is pretty damn slow
	log_dma.warning("using a buffering copier for reductions (" IDFMT " -> " IDFMT ")",
			src_mem.id, dst_mem.id);
	return new BufferedReductionMemPairCopier(src_mem, dst_mem, redop_id, fold);
      }
    }

    template <unsigned DIM>
    static unsigned compress_strides(const Arrays::Rect<DIM> &r,
				     const Arrays::Point<1> in1[DIM], const Arrays::Point<1> in2[DIM],
				     Arrays::Point<DIM>& extents,
				     Arrays::Point<1> out1[DIM], Arrays::Point<1> out2[DIM])
    {
      // sort the dimensions by the first set of strides for maximum gathering goodness
      unsigned stride_order[DIM];
      for(unsigned i = 0; i < DIM; i++) stride_order[i] = i;
      // yay, bubble sort!
      for(unsigned i = 0; i < DIM; i++)
	for(unsigned j = 0; j < i; j++)
	  if(in1[stride_order[j]] > in1[stride_order[j+1]]) {
	    int t = stride_order[j];
	    stride_order[j] = stride_order[j+1];
	    stride_order[j+1] = t;
	  }

      int curdim = -1;
      Arrays::coord_t exp1 = 0, exp2 = 0;

      // now go through dimensions, collapsing each if it matches the expected stride for
      //  both sets (can't collapse for first)
      for(unsigned i = 0; i < DIM; i++) {
	unsigned d = stride_order[i];
	Arrays::coord_t e = r.dim_size(d);
	if(i && (exp1 == in1[d][0]) && (exp2 == in2[d][0])) {
	  // collapse and grow extent
	  extents.x[curdim] *= e;
	  exp1 *= e;
	  exp2 *= e;
	} else {
	  // no match - create a new dimension
	  curdim++;
	  extents.x[curdim] = e;
	  exp1 = in1[d][0] * e;
	  exp2 = in2[d][0] * e;
	  out1[curdim] = in1[d];
	  out2[curdim] = in2[d];
	}
      }

      return curdim+1;
    }

    void CopyRequest::perform_dma_mask(MemPairCopier *mpc)
    {
      IndexSpaceImpl *ispace = get_runtime()->get_index_space_impl(domain.get_index_space());
      assert(ispace->valid_mask_complete);

      // this is the SOA-friendly loop nesting
      for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	RegionInstance src_inst = it->first.first;
	RegionInstance dst_inst = it->first.second;
	OASVec& oasvec = it->second;

	InstPairCopier *ipc = mpc->inst_pair(src_inst, dst_inst, oasvec);

	// does the copier want to iterate the domain itself?
	if(ipc->copy_all_fields(domain))
	  continue;

	// index space instances use 1D linearizations for translation
	Arrays::Mapping<1, 1> *src_linearization = get_runtime()->get_instance_impl(src_inst)->metadata.linearization.get_mapping<1>();
	Arrays::Mapping<1, 1> *dst_linearization = get_runtime()->get_instance_impl(dst_inst)->metadata.linearization.get_mapping<1>();

	// does the destination instance space's index space match what we're copying?  if so,
	//  it's ok to copy extra elements (to decrease sparsity) because they're unused in
	//  the destination
	assert(get_runtime()->get_instance_impl(dst_inst)->metadata.is_valid());
	size_t rlen_target;
	if(ispace->me == get_runtime()->get_instance_impl(dst_inst)->metadata.is) {
	  rlen_target = 32768 / 4; // aim for ~32KB transfers at least
	} else {
	  rlen_target = 1;
	}
	
	ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	Arrays::coord_t rstart; size_t rlen;
	while(e->get_next(rstart, rlen)) {
	  // do we want to copy extra elements to fill in some holes?
	  while(rlen < rlen_target) {
	    // see where the next valid elements are
	    Arrays::coord_t rstart2; size_t rlen2;
	    // if none, stop
	    if(!e->peek_next(rstart2, rlen2)) break;
	    // or if they don't even start until outside the window, stop
	    if(rstart2 > (rstart + (off_t)rlen_target)) break;
	    // ok, include the next valid element(s) and any invalid ones in between
	    //printf("bloating from %d to %d\n", rlen, rstart2 + rlen2 - rstart);
	    rlen = rstart2 + rlen2 - rstart;
	    // and actually take the next range from the enumerator
	    e->get_next(rstart2, rlen2);
	  }

	  int sstart = src_linearization->image((Arrays::coord_t)rstart);
	  int dstart = dst_linearization->image((Arrays::coord_t)rstart);
#ifdef DEBUG_LOW_LEVEL
	  assert(src_linearization->image_is_dense(Arrays::Rect<1>((Arrays::coord_t)rstart, (Arrays::coord_t)(rstart + rlen - 1))));
	  assert(dst_linearization->image_is_dense(Arrays::Rect<1>((Arrays::coord_t)rstart, (Arrays::coord_t)(rstart + rlen - 1))));
#endif
	  //printf("X: %d+%d %d %d\n", rstart, rlen, sstart, dstart);

	  for (unsigned idx = 0; idx < oasvec.size(); idx++)
	    ipc->copy_field(sstart, dstart, rlen, idx);
	}
        delete e;
	delete ipc;
      }
    }

    bool oas_sort_by_dst(OffsetsAndSize a, OffsetsAndSize b) {return a.dst_offset < b.dst_offset; }

    DomainLinearization create_ib_linearization(const Domain& dm)
    {
      //std::vector<Layouts::DimKind> kind_vec;
      std::vector<size_t> size_vec;
      switch (dm.get_dim()) {
      case 1:
      {
        /*
        kind_vec.push_back(Layouts::DIM_X);
        size_vec.push_back(dm.get_rect<1>().dim_size(0));
        Layouts::SplitDimLinearization<1> cl(dm.get_rect<1>().lo,
                                             make_point(0),
                                             kind_vec,
                                             size_vec);
        */
        Arrays::FortranArrayLinearization<1> cl(dm.get_rect<1>(), 0);
        return DomainLinearization::from_mapping<1>(
                   Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
      }
      case 2:
      {
        /*
        kind_vec.push_back(Layouts::DIM_X);
        kind_vec.push_back(Layouts::DIM_Y);
        size_vec.push_back(dm.get_rect<2>().dim_size(0));
        size_vec.push_back(dm.get_rect<2>().dim_size(1));
        Layouts::SplitDimLinearization<2> cl(dm.get_rect<2>().lo,
                                             make_point(0),
                                             kind_vec,
                                             size_vec);
        */
        Arrays::FortranArrayLinearization<2> cl(dm.get_rect<2>(), 0);
        return DomainLinearization::from_mapping<2>(
                   Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
      }
      case 3:
      {
        /*
        kind_vec.push_back(Layouts::DIM_X);
        kind_vec.push_back(Layouts::DIM_Y);
        kind_vec.push_back(Layouts::DIM_Z);
        size_vec.push_back(dm.get_rect<3>().dim_size(0));
        size_vec.push_back(dm.get_rect<3>().dim_size(1));
        size_vec.push_back(dm.get_rect<3>().dim_size(2));
        Layouts::SplitDimLinearization<3> cl(dm.get_rect<3>().lo,
                                             make_point(0),
                                             kind_vec,
                                             size_vec);
        */
        Arrays::FortranArrayLinearization<3> cl(dm.get_rect<3>(), 0);
        return DomainLinearization::from_mapping<3>(
                   Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
      }
      default:
        assert(0);
      }
      assert(0);
      return DomainLinearization();
    }

    Buffer simple_create_intermediate_buffer(const IBInfo& ib_info, const Domain& domain,
                                             OASVec oasvec, OASVec& oasvec_src, OASVec& oasvec_dst,
                                             DomainLinearization linearization)
    {
      oasvec_src.clear();
      oasvec_dst.clear();
      std::sort(oasvec.begin(), oasvec.end(), oas_sort_by_dst);
      off_t ib_elmnt_size = 0;
      for (unsigned i = 0; i < oasvec.size(); i++) {
        OffsetsAndSize oas_src, oas_dst;
        oas_src.src_offset = oasvec[i].src_offset;
        oas_src.dst_offset = ib_elmnt_size;
        oas_src.size = oasvec[i].size;
        oas_dst.src_offset = ib_elmnt_size;
        oas_dst.dst_offset = oasvec[i].dst_offset;
        oas_dst.size = oasvec[i].size;
        ib_elmnt_size += oasvec[i].size;
        oasvec_src.push_back(oas_src);
        oasvec_dst.push_back(oas_dst);
      }
      size_t ib_size; /*size of ib (bytes)*/
      if (domain.get_volume() * ib_elmnt_size < IB_MAX_SIZE)
        ib_size = domain.get_volume() * ib_elmnt_size;
      else
        ib_size = IB_MAX_SIZE;
      // Make sure the size we want here matches our previous allocation
      assert(ib_size == ib_info.size);
      //off_t ib_offset = get_runtime()->get_memory_impl(tgt_mem)->alloc_bytes(ib_size);
      //assert(ib_offset >= 0);
      // Create a new linearization order x->y in Domain
      DomainLinearization dl;
      if (domain.get_dim() == 0)
        dl = linearization;
      else
        dl = create_ib_linearization(domain);
      Buffer ib_buf(ib_info.offset, true, domain.get_volume(), ib_elmnt_size, ib_info.size, dl, ib_info.memory);
      return ib_buf;
    }

    inline bool is_cpu_mem(Memory::Kind kind)
    {
      return (kind == Memory::REGDMA_MEM || kind == Memory::LEVEL3_CACHE || kind == Memory::LEVEL2_CACHE
              || kind == Memory::LEVEL1_CACHE || kind == Memory::SYSTEM_MEM || kind == Memory::SOCKET_MEM
              || kind == Memory::Z_COPY_MEM);
    }

    XferDes::XferKind get_xfer_des(Memory src_mem, Memory dst_mem)
    {
      Memory::Kind src_ll_kind = get_runtime()->get_memory_impl(src_mem)->lowlevel_kind;
      Memory::Kind dst_ll_kind = get_runtime()->get_memory_impl(dst_mem)->lowlevel_kind;
      if(ID(src_mem).memory.owner_node == ID(dst_mem).memory.owner_node) {
        switch(src_ll_kind) {
        case Memory::GLOBAL_MEM:
          if (is_cpu_mem(dst_ll_kind))
            return XferDes::XFER_GASNET_READ;
          else
            return XferDes::XFER_NONE;
        case Memory::REGDMA_MEM:
        case Memory::LEVEL3_CACHE:
        case Memory::LEVEL2_CACHE:
        case Memory::LEVEL1_CACHE:
        case Memory::SYSTEM_MEM:
        case Memory::SOCKET_MEM:
        case Memory::Z_COPY_MEM:
          if (is_cpu_mem(dst_ll_kind))
            return XferDes::XFER_MEM_CPY;
          else if (dst_ll_kind == Memory::GLOBAL_MEM)
            return XferDes::XFER_GASNET_WRITE;
          else if (dst_ll_kind == Memory::GPU_FB_MEM) {
            std::set<Memory> visible_mems;
            Machine::get_machine().get_visible_memories(dst_mem, visible_mems);
            if (visible_mems.count(src_mem) == 0)
              return XferDes::XFER_NONE;
            return XferDes::XFER_GPU_TO_FB;
          }
          else if (dst_ll_kind == Memory::DISK_MEM)
            return XferDes::XFER_DISK_WRITE;
          else if (dst_ll_kind == Memory::HDF_MEM)
            return XferDes::XFER_HDF_WRITE;
          else if (dst_ll_kind == Memory::FILE_MEM)
            return XferDes::XFER_FILE_WRITE;
          assert(0);
          break;
        case Memory::GPU_FB_MEM:
        {
          std::set<Memory> visible_mems;
          Machine::get_machine().get_visible_memories(src_mem, visible_mems);
          if (dst_ll_kind == Memory::GPU_FB_MEM) {
            if (src_mem == dst_mem)
              return XferDes::XFER_GPU_IN_FB;
            else if (visible_mems.count(dst_mem) != 0)
              return XferDes::XFER_GPU_PEER_FB;
            return XferDes::XFER_NONE;
          }
          else if (is_cpu_mem(dst_ll_kind) && visible_mems.count(dst_mem) != 0)
              return XferDes::XFER_GPU_FROM_FB;
          else
            return XferDes::XFER_NONE;
        }
        case Memory::DISK_MEM:
          if (is_cpu_mem(dst_ll_kind))
            return XferDes::XFER_DISK_READ;
          else
            return XferDes::XFER_NONE;
        case Memory::FILE_MEM:
          if (is_cpu_mem(dst_ll_kind))
            return XferDes::XFER_FILE_READ;
          else
            return XferDes::XFER_NONE;
        case Memory::HDF_MEM:
          if (is_cpu_mem(dst_ll_kind))
            return XferDes::XFER_HDF_READ;
          else
            return XferDes::XFER_NONE;
        default:
          assert(0);
        }
      } else {
        if (is_cpu_mem(src_ll_kind) && dst_ll_kind == Memory::REGDMA_MEM)
          return XferDes::XFER_REMOTE_WRITE;
        else
          return XferDes::XFER_NONE;
      }
      return XferDes::XFER_NONE;
    }

    void find_shortest_path(Memory src_mem, Memory dst_mem, std::vector<Memory>& path)
    {
      std::map<Memory, std::vector<Memory> > dist;
      std::set<Memory> all_mem;
      std::queue<Memory> active_nodes;
      all_mem.insert(src_mem);
      all_mem.insert(dst_mem);
      Node* node = &(get_runtime()->nodes[ID(src_mem).memory.owner_node]);
      for (std::vector<MemoryImpl*>::const_iterator it = node->ib_memories.begin();
           it != node->ib_memories.end(); it++) {
        all_mem.insert((*it)->me);
      }
      node = &(get_runtime()->nodes[ID(dst_mem).memory.owner_node]);
      for (std::vector<MemoryImpl*>::const_iterator it = node->ib_memories.begin();
           it != node->ib_memories.end(); it++) {
        all_mem.insert((*it)->me);
      }
      for (std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
        if (get_xfer_des(src_mem, *it) != XferDes::XFER_NONE) {
          dist[*it] = std::vector<Memory>();
          dist[*it].push_back(src_mem);
          dist[*it].push_back(*it);
          active_nodes.push(*it);
        }
      }
      while (!active_nodes.empty()) {
        Memory cur = active_nodes.front();
        active_nodes.pop();
        std::vector<Memory> sub_path = dist[cur];
        for(std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it ++) {
          if (get_xfer_des(cur, *it) != XferDes::XFER_NONE) {
            if (dist.find(*it) == dist.end()) {
              dist[*it] = sub_path;
              dist[*it].push_back(*it);
              active_nodes.push(*it);
            }
          }
        }
      }
      assert(dist.find(dst_mem) != dist.end());
      path = dist[dst_mem];
    }

    template<unsigned DIM>
    void CopyRequest::perform_new_dma(Memory src_mem, Memory dst_mem)
    {
      //mark_started();
      //std::vector<Memory> mem_path;
      //find_shortest_path(src_mem, dst_mem, mem_path);
      for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
        std::vector<XferDesID> sub_path;
        for (unsigned idx = 0; idx < mem_path.size() - 1; idx ++) {
          XferDesID new_xdid = get_xdq_singleton()->get_guid(ID(mem_path[idx]).memory.owner_node);
          sub_path.push_back(new_xdid);
          path.push_back(new_xdid);
        }
        RegionInstance src_inst = it->first.first;
        RegionInstance dst_inst = it->first.second;
        OASVec oasvec = it->second, oasvec_src, oasvec_dst;
        IBByInst::iterator ib_it = ib_by_inst.find(it->first);
        assert(ib_it != ib_by_inst.end());
        IBVec& ibvec = ib_it->second;
        RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(src_inst);
        RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(dst_inst);

        //MemoryImpl::MemoryKind src_kind = get_runtime()->get_memory_impl(src_mem)->kind;
        //MemoryImpl::MemoryKind dst_kind = get_runtime()->get_memory_impl(dst_mem)->kind;

        //Memory::Kind dst_ll_kind = get_runtime()->get_memory_impl(dst_mem)->lowlevel_kind;

        // We don't need to care about deallocation of Buffer class
        // This will be handled by XferDes destruction
        Buffer src_buf(&src_impl->metadata, src_mem);
        Buffer dst_buf(&dst_impl->metadata, dst_mem);
        Buffer pre_buf;
        assert(mem_path.size() - 1 == sub_path.size());
        for (unsigned idx = 0; idx < mem_path.size(); idx ++) {
          log_new_dma.info("mem_path[%d]: node(%llu), memory(%d)", idx, ID(mem_path[idx]).memory.owner_node, mem_path[idx].kind());
          if (idx == 0) {
            pre_buf = src_buf;
          } else {
            XferDesID xd_guid = sub_path[idx - 1];
            XferDesID pre_xd_guid = idx == 1 ? XferDes::XFERDES_NO_GUID : sub_path[idx - 2];
            XferDesID next_xd_guid = idx == sub_path.size() ? XferDes::XFERDES_NO_GUID : sub_path[idx];
            bool mark_started = ((idx == 1) && (it == oas_by_inst->begin())) ? true : false;
            Buffer cur_buf;
            XferDes::XferKind kind = get_xfer_des(mem_path[idx - 1], mem_path[idx]);
            XferOrder::Type order;
            if (mem_path.size() == 2)
              order = XferOrder::ANY_ORDER;
            else
              order = idx == 1 ? XferOrder::DST_FIFO : XferOrder::SRC_FIFO;
            RegionInstance attach_inst;
            if ((kind == XferDes::XFER_HDF_READ)
              ||(kind == XferDes::XFER_FILE_READ))
              attach_inst = src_inst;
            else if ((kind == XferDes::XFER_HDF_WRITE)
              ||(kind == XferDes::XFER_FILE_WRITE))
              attach_inst = dst_inst;
            else
              attach_inst = RegionInstance::NO_INST;

            
            XferDesFence* complete_fence = new XferDesFence(this);
            add_async_work_item(complete_fence);
            if (DIM == 0) {
              // Need to do something special for unstructured data: we perform a 1D copy from first_elemnt to last_elemnt
              // First we should make sure destination instance space's index space match what we're copying
              IndexSpaceImpl *ispace = get_runtime()->get_index_space_impl(domain.get_index_space());
              assert(get_runtime()->get_instance_impl(dst_inst)->metadata.is_valid());
              if (ispace->me == get_runtime()->get_instance_impl(dst_inst)->metadata.is) {
                // perform a 1D copy from first_element to last_element
                Realm::StaticAccess<IndexSpaceImpl> data(ispace);
                assert(data->num_elmts > 0);
                Rect<1> new_rect(make_point(data->first_elmt), make_point(data->last_elmt));
                Domain new_domain = Domain::from_rect<1>(new_rect);
                if (idx != mem_path.size() - 1) {
                  cur_buf = simple_create_intermediate_buffer(ibvec[idx-1],
                              new_domain, oasvec, oasvec_src, oasvec_dst, dst_buf.linearization);
                } else {
                  cur_buf = dst_buf;
                  oasvec_src = oasvec;
                  oasvec.clear();
                }
                create_xfer_des<1>(this, gasnet_mynode(), xd_guid, pre_xd_guid,
                                   next_xd_guid, mark_started, pre_buf, cur_buf,
                                   new_domain, oasvec_src,
                                   16 * 1024 * 1024/*max_req_size*/, 100/*max_nr*/,
                                   priority, order, kind, complete_fence, attach_inst);
              } else {
                if (idx != mem_path.size() - 1) {
                  cur_buf = simple_create_intermediate_buffer(ibvec[idx-1],
                              domain, oasvec, oasvec_src, oasvec_dst, dst_buf.linearization);
                } else {
                  cur_buf = dst_buf;
                  oasvec_src = oasvec;
                  oasvec.clear();
                }
                create_xfer_des<0>(this, gasnet_mynode(), xd_guid, pre_xd_guid,
                                   next_xd_guid, mark_started, pre_buf, cur_buf,
                                   domain, oasvec_src,
                                   16 * 1024 * 1024 /*max_req_size*/, 100/*max_nr*/,
                                   priority, order, kind, complete_fence, attach_inst);
              }
            }
            else {
              if (idx != mem_path.size() - 1) {
                cur_buf = simple_create_intermediate_buffer(ibvec[idx-1],
                            domain, oasvec, oasvec_src, oasvec_dst, dst_buf.linearization);
              } else {
                cur_buf = dst_buf;
                oasvec_src = oasvec;
                oasvec.clear();
              }
              create_xfer_des<DIM>(this, gasnet_mynode(), xd_guid, pre_xd_guid,
                                   next_xd_guid, mark_started, pre_buf, cur_buf,
                                   domain, oasvec_src,
                                   16 * 1024 * 1024/*max_req_size*/, 100/*max_nr*/,
                                   priority, order, kind, complete_fence, attach_inst);
            }
            pre_buf = cur_buf;
            oasvec = oasvec_dst;
          }
        }
      }
      mark_finished(true/*successful*/);
    }

    template <unsigned DIM>
    void CopyRequest::perform_dma_rect(MemPairCopier *mpc)
    {
      Arrays::Rect<DIM> orig_rect = domain.get_rect<DIM>();

      // empty rectangles are easy to copy...
      if(orig_rect.volume() == 0)
	return;

      // this is the SOA-friendly loop nesting
      for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	RegionInstance src_inst = it->first.first;
	RegionInstance dst_inst = it->first.second;
	OASVec& oasvec = it->second;

	InstPairCopier *ipc = mpc->inst_pair(src_inst, dst_inst, oasvec);

	// does the copier want to iterate the domain itself?
	if(ipc->copy_all_fields(domain))
	  continue;

	RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(src_inst);
	RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(dst_inst);
	
	assert(src_impl->metadata.is_valid());
	assert(dst_impl->metadata.is_valid());

	Arrays::Mapping<DIM, 1> *src_linearization = src_impl->metadata.linearization.get_mapping<DIM>();
	Arrays::Mapping<DIM, 1> *dst_linearization = dst_impl->metadata.linearization.get_mapping<DIM>();

	// see what linear subrects we can get - again, iterate over destination first for gathering
	for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lso(orig_rect, *dst_linearization); lso; lso++) {
	  for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lsi(lso.subrect, *src_linearization); lsi; lsi++) {
	    // see if we can compress the strides for a more efficient copy
	    Arrays::Point<1> src_cstrides[DIM], dst_cstrides[DIM];
	    Arrays::Point<DIM> extents;
	    int cdim = compress_strides(lsi.subrect, lso.strides, lsi.strides,
					extents, dst_cstrides, src_cstrides);

#ifdef NEW2D_DEBUG
	    printf("ORIG: (%d,%d,%d)->(%d,%d,%d)\n",
		   orig_rect.lo[0], orig_rect.lo[1], orig_rect.lo[2],
		   orig_rect.hi[0], orig_rect.hi[1], orig_rect.hi[2]);
	    printf("DST:  (%d,%d,%d)->(%d,%d,%d)  %d+(%d,%d,%d)\n",
		   lso.subrect.lo[0], lso.subrect.lo[1], lso.subrect.lo[2],
		   lso.subrect.hi[0], lso.subrect.hi[1], lso.subrect.hi[2],
		   lso.image_lo[0],
		   lso.strides[0][0], lso.strides[1][0], lso.strides[2][0]);
	    printf("SRC:  (%d,%d,%d)->(%d,%d,%d)  %d+(%d,%d,%d)\n",
		   lsi.subrect.lo[0], lsi.subrect.lo[1], lsi.subrect.lo[2],
		   lsi.subrect.hi[0], lsi.subrect.hi[1], lsi.subrect.hi[2],
		   lsi.image_lo[0],
		   lsi.strides[0][0], lsi.strides[1][0], lsi.strides[2][0]);
	    printf("CMP:  %d (%d,%d,%d) +(%d,%d,%d) +(%d,%d,%d)\n",
		   cdim,
		   extents[0], extents[1], extents[2],
		   dst_cstrides[0][0], dst_cstrides[1][0], dst_cstrides[2][0],
		   src_cstrides[0][0], src_cstrides[1][0], src_cstrides[2][0]);
#endif
	    if((cdim == 1) && (dst_cstrides[0][0] == 1) && (src_cstrides[0][0] == 1)) {
	      // all the dimension(s) collapse to a 1-D extent in both source and dest, so one big copy
	      ipc->copy_all_fields(lsi.image_lo[0], lso.image_lo[0], extents[0]);
	      continue;
	    }

	    if((cdim == 2) && (dst_cstrides[0][0] == 1) && (src_cstrides[0][0] == 1)) {
	      // source and/or destination need a 2-D description
	      ipc->copy_all_fields(lsi.image_lo[0], lso.image_lo[0], extents[0],
				   src_cstrides[1][0], dst_cstrides[1][0], extents[1]);
	      continue;
	    }

	    // fall through - just identify dense (sub)subrects and copy them

	    // iterate by output rectangle first - this gives us a chance to gather data when linearizations
	    //  don't match up
	    for(Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > dso(lsi.subrect, *dst_linearization); dso; dso++) {
	      // dense subrect in dst might not be dense in src
	      for(Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > dsi(dso.subrect, *src_linearization); dsi; dsi++) {
		Arrays::Rect<1> irect = dsi.image;
		// rectangle in output must be recalculated
		Arrays::Rect<DIM> subrect_check;
		Arrays::Rect<1> orect = dst_linearization->image_dense_subrect(dsi.subrect, subrect_check);
		assert(dsi.subrect == subrect_check);

		//for(OASVec::iterator it2 = oasvec.begin(); it2 != oasvec.end(); it2++)
		for (unsigned idx = 0; idx < oasvec.size(); idx++)
		  ipc->copy_field(irect.lo, orect.lo, irect.hi[0] - irect.lo[0] + 1, idx);
		//it2->src_offset, it2->dst_offset, it2->size);
	      }
	    }
	  }
	}

        // Dammit Sean stop leaking things!
        delete ipc;
      }
    }

    void CopyRequest::perform_dma(void)
    {
      log_dma.debug("request %p executing", this);

      DetailedTimer::ScopedPush sp(TIME_COPY);

      // create a copier for the memory used by all of these instance pairs
      Memory src_mem = get_runtime()->get_instance_impl(oas_by_inst->begin()->first.first)->memory;
      Memory dst_mem = get_runtime()->get_instance_impl(oas_by_inst->begin()->first.second)->memory;

      // <NEWDMA>
      if(measurements.wants_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>()) {
        const InstPair &pair = oas_by_inst->begin()->first;
        size_t total_field_size = 0;
        for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
          for (size_t i = 0; i < it->second.size(); i++) {
            total_field_size += it->second[i].size;
          }
        }

        Realm::ProfilingMeasurements::OperationMemoryUsage usage;
        usage.source = pair.first.get_location();
        usage.target = pair.second.get_location();
        usage.size = total_field_size * domain.get_volume();
        measurements.add_measurement(usage);
      }

      switch (domain.get_dim()) {
      case 0:
        perform_new_dma<0>(src_mem, dst_mem);
        break;
      case 1:
        perform_new_dma<1>(src_mem, dst_mem);
        break;
      case 2:
        perform_new_dma<2>(src_mem, dst_mem);
        break;
      case 3:
        perform_new_dma<3>(src_mem, dst_mem);
        break;
      default:
        assert(0);
      }

      log_dma.info() << "dma request " << (void *)this << " finished - is="
                     << domain << " before=" << before_copy << " after=" << get_finish_event();
      return;
      // </NEWDMA>

      // MemPairCopier *mpc = MemPairCopier::create_copier(src_mem, dst_mem);

      CustomSerdezID serdez_id = oas_by_inst->begin()->second.begin()->serdez_id;
      // for now we launches an individual copy request for every serdez copy
      assert(serdez_id == 0 || (oas_by_inst->size() == 1 && oas_by_inst->begin()->second.size() == 1));
      MemPairCopier *mpc = MemPairCopier::create_copier(src_mem, dst_mem, 0, serdez_id);
      switch(domain.get_dim()) {
      case 0:
	{
	  // iterate over valid ranges of an index space
	  perform_dma_mask(mpc);
	  break;
	}

	// rectangle cases
      case 1: perform_dma_rect<1>(mpc); break;
      case 2: perform_dma_rect<2>(mpc); break;
      case 3: perform_dma_rect<3>(mpc); break;

      default: assert(0);
      };

      log_dma.info() << "dma request " << (void *)this << " finished - is="
		     << domain << " before=" << before_copy << " after=" << get_finish_event();

      mpc->flush(this);

      if(measurements.wants_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>()) {
        const InstPair &pair = oas_by_inst->begin()->first; 

        Realm::ProfilingMeasurements::OperationMemoryUsage usage;
        usage.source = pair.first.get_location();
        usage.target = pair.second.get_location();
        usage.size = mpc->get_total_bytes();
        measurements.add_measurement(usage);
      }

      // if(after_copy.exists())
      // 	after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());

      delete mpc;

#ifdef EVEN_MORE_DEAD_DMA_CODE
      RegionInstanceImpl *src_impl = src.impl();
      RegionInstanceImpl *tgt_impl = target.impl();

      // we should have already arranged to have access to this data, so
      //  assert if we don't
      StaticAccess<RegionInstanceImpl> src_data(src_impl, true);
      StaticAccess<RegionInstanceImpl> tgt_data(tgt_impl, true);

      // code path for copies to/from reduction-only instances not done yet
      // are we doing a reduction?
      const ReductionOpUntyped *redop = ((src_data->redopid >= 0) ?
					   get_runtime()->reduce_op_table[src_data->redopid] :
					   0);
      bool red_fold = (tgt_data->redopid >= 0);
      // if destination is a reduction, source must be also and must match
      assert(tgt_data->redopid == src_data->redopid);

      // for now, a reduction list instance can only be copied back to a
      //  non-reduction instance
      bool red_list = (src_data->redopid >= 0) && (src_data->red_list_size >= 0);
      if(red_list)
	assert(tgt_data->redopid < 0);

      MemoryImpl *src_mem = src_impl->memory.impl();
      MemoryImpl *tgt_mem = tgt_impl->memory.impl();

      // get valid masks from region to limit copy to correct data
      IndexSpaceImpl *is_impl = is.impl();
      //RegionMetaDataUntyped::Impl *src_reg = src_data->region.impl();
      //RegionMetaDataUntyped::Impl *tgt_reg = tgt_data->region.impl();

      log_dma.debug("copy: " IDFMT "->" IDFMT " (" IDFMT "/%p)",
		    src.id, target.id, is.id, is_impl->valid_mask);

      // if we're missing the valid mask at this point, we've screwed up
      if(!is_impl->valid_mask_complete) {
	assert(is_impl->valid_mask_complete);
      }

      log_dma.debug("performing copy " IDFMT " (%d) -> " IDFMT " (%d) - %zd bytes (%zd)", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size);

      switch(src_mem->kind) {
      case MemoryImpl::MKIND_SYSMEM:
      case MemoryImpl::MKIND_ZEROCOPY:
	{
	  const void *src_ptr = src_mem->get_direct_ptr(src_data->alloc_offset, bytes_to_copy);
	  assert(src_ptr != 0);

	  switch(tgt_mem->kind) {
	  case MemoryImpl::MKIND_SYSMEM:
	  case MemoryImpl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->alloc_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      RangeExecutors::Memcpy rexec(tgt_ptr,
					   src_ptr,
					   elmt_size);
	      ElementMask::forall_ranges(rexec, *is_impl->valid_mask);
	    }
	    break;

	  case MemoryImpl::MKIND_GLOBAL:
	    {
	      if(redop) {
		if(red_list) {
		  RangeExecutors::GasnetPutRedList rexec(tgt_mem, tgt_data->alloc_offset,
							 src_data->redopid, redop,
							 src_ptr, redop->sizeof_list_entry);
		  size_t count;
		  src_mem->get_bytes(src_data->count_offset, &count,
				     sizeof(size_t));
		  if(count > 0) {
		    rexec.do_span(0, count);
		    count = 0;
		    src_mem->put_bytes(src_data->count_offset, &count,
				       sizeof(size_t));
		  }
		} else {
		  RangeExecutors::GasnetPutReduce rexec(tgt_mem, tgt_data->alloc_offset,
							redop, red_fold,
							src_ptr, elmt_size);
		  ElementMask::forall_ranges(rexec, *is_impl->valid_mask);
		}
	      } else {
#define USE_BATCHED_GASNET_XFERS
#ifdef USE_BATCHED_GASNET_XFERS
		RangeExecutors::GasnetPutBatched rexec(tgt_mem, tgt_data->alloc_offset,
						       src_ptr, elmt_size);
#else
		RangeExecutors::GasnetPut rexec(tgt_mem, tgt_data->alloc_offset,
						src_ptr, elmt_size);
#endif
		ElementMask::forall_ranges(rexec, *is_impl->valid_mask);

#ifdef USE_BATCHED_GASNET_XFERS
		rexec.finish();
#endif
	      }
	    }
	    break;

	  case MemoryImpl::MKIND_GPUFB:
	    {
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      assert(!redop);
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb(tgt_data->alloc_offset,
							src_ptr,
							is_impl->valid_mask,
							elmt_size,
							Event::NO_EVENT,
							after_copy);
	      return;
	    }
	    break;

	  case MemoryImpl::MKIND_REMOTE:
	    {
	      // use active messages to push data to other node
	      RangeExecutors::RemoteWrite rexec(tgt_impl->memory,
						tgt_data->alloc_offset,
						src_ptr, elmt_size,
						after_copy);

	      ElementMask::forall_ranges(rexec, *is_impl->valid_mask);

	      // if no copies actually occur, we'll get the event back
	      // from the range executor and we have to trigger it ourselves
	      Event finish_event = rexec.finish();
	      if(finish_event.exists()) {
		log_dma.debug("triggering event " IDFMT "/%d after empty remote copy",
			     finish_event.id, finish_event.gen);
		assert(finish_event == after_copy);
		get_runtime()->get_singleevent_impl(finish_event)->trigger(finish_event.gen, gasnet_mynode());
	      }
	      
	      return;
	    }

	  default:
	    assert(0);
	  }
	}
	break;

      case MemoryImpl::MKIND_GLOBAL:
	{
	  switch(tgt_mem->kind) {
	  case MemoryImpl::MKIND_SYSMEM:
	  case MemoryImpl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->alloc_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
#ifdef USE_BATCHED_GASNET_XFERS
	      RangeExecutors::GasnetGetBatched rexec(tgt_ptr, src_mem, 
						     src_data->alloc_offset, elmt_size);
#else
	      RangeExecutors::GasnetGet rexec(tgt_ptr, src_mem, 
					      src_data->alloc_offset, elmt_size);
#endif
	      ElementMask::forall_ranges(rexec, *is_impl->valid_mask);

#ifdef USE_BATCHED_GASNET_XFERS
	      rexec.finish();
#endif
	    }
	    break;

	  case MemoryImpl::MKIND_GLOBAL:
	    {
	      assert(!redop);
	      RangeExecutors::GasnetGetAndPut rexec(tgt_mem, tgt_data->alloc_offset,
						    src_mem, src_data->alloc_offset,
						    elmt_size);
	      ElementMask::forall_ranges(rexec, *is_impl->valid_mask);
	    }
	    break;

	  case MemoryImpl::MKIND_GPUFB:
	    {
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb_generic(tgt_data->alloc_offset,
								src_mem,
								src_data->alloc_offset,
								is_impl->valid_mask,
								elmt_size,
								Event::NO_EVENT,
								after_copy);
	      return;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      case MemoryImpl::MKIND_GPUFB:
	{
	  switch(tgt_mem->kind) {
	  case MemoryImpl::MKIND_SYSMEM:
	  case MemoryImpl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->alloc_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb(tgt_ptr, src_data->alloc_offset,
							  is_impl->valid_mask,
							  elmt_size,
							  Event::NO_EVENT,
							  after_copy);
	      return;
	    }
	    break;

	  case MemoryImpl::MKIND_GLOBAL:
	    {
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb_generic(tgt_mem,
								  tgt_data->alloc_offset,
								  src_data->alloc_offset,
								  is_impl->valid_mask,
								  elmt_size,
								  Event::NO_EVENT,
								  after_copy);
	      return;
	    }
	    break;

	  case MemoryImpl::MKIND_GPUFB:
	    {
	      // only support copies within the same FB for now
	      assert(src_mem == tgt_mem);
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_within_fb(tgt_data->alloc_offset,
							    src_data->alloc_offset,
							    is_impl->valid_mask,
							    elmt_size,
							    Event::NO_EVENT,
							    after_copy);
	      return;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      default:
	assert(0);
      }

      log_dma.debug("finished copy " IDFMT " (%d) -> " IDFMT " (%d) - %zd bytes (%zd), event=" IDFMT "/%d", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size, after_copy.id, after_copy.gen);
#endif
    } 

    ReduceRequest::ReduceRequest(const void *data, size_t datalen,
				 ReductionOpID _redop_id,
				 bool _red_fold,
				 Event _before_copy,
				 Event _after_copy,
				 int _priority)
      : DmaRequest(_priority, _after_copy),
	inst_lock_event(Event::NO_EVENT),
	redop_id(_redop_id), red_fold(_red_fold),
	before_copy(_before_copy)
    {
      const IDType *idata = (const IDType *)data;

      idata = domain.deserialize(idata);

      priority = 0;

      // get sources
      int n_srcs = *idata++;
      for(int i = 0; i < n_srcs; i++) {
	Domain::CopySrcDstField f;
	f.inst.id = *idata++;
	f.offset = *idata++;
	f.size = *idata++;
	srcs.push_back(f);
      }

      // single dst field
      dst.inst.id = *idata++;
      dst.offset = *idata++;
      dst.size = *idata++;

      inst_lock_needed = *idata++;

      // Unpack any requests that we have
      // TODO: unbreak once the serialization stuff is repaired
      //const void *result = requests.deserialize(idata);
      //Realm::Operation::reconstruct_measurements();
      // better have consumed exactly the right amount of data
      //assert((((unsigned long long)result) - ((unsigned long long)data)) == datalen);
      size_t request_size = *reinterpret_cast<const size_t*>(idata);
      idata += sizeof(size_t) / sizeof(IDType);
      FixedBufferDeserializer deserializer(idata, request_size);
      deserializer >> requests;
      Realm::Operation::reconstruct_measurements();

      log_dma.info("dma request %p deserialized - " IDFMT "[%lld]->" IDFMT "[%lld]:%zu (+%zu) %s %d (" IDFMT ") " IDFMT " " IDFMT,
		   this,
		   srcs[0].inst.id, srcs[0].offset,
		   dst.inst.id, dst.offset, dst.size,
		   srcs.size() - 1,
		   (red_fold ? "fold" : "apply"),
		   redop_id,
		   domain.is_id,
		   before_copy.id,
		   get_finish_event().id);
    }

    ReduceRequest::ReduceRequest(const Domain& _domain,
				 const std::vector<Domain::CopySrcDstField>& _srcs,
				 const Domain::CopySrcDstField& _dst,
				 bool _inst_lock_needed,
				 ReductionOpID _redop_id,
				 bool _red_fold,
				 Event _before_copy,
				 Event _after_copy,
				 int _priority, 
                                 const Realm::ProfilingRequestSet &reqs)
      : DmaRequest(_priority, _after_copy, reqs),
	domain(_domain),
	dst(_dst), 
	inst_lock_needed(_inst_lock_needed), inst_lock_event(Event::NO_EVENT),
	redop_id(_redop_id), red_fold(_red_fold),
	before_copy(_before_copy)
    {
      srcs.insert(srcs.end(), _srcs.begin(), _srcs.end());

      log_dma.info("dma request %p created - " IDFMT "[%lld]->" IDFMT "[%lld]:%zu (+%zu) %s %d (" IDFMT ") " IDFMT " " IDFMT,
		   this,
		   srcs[0].inst.id, srcs[0].offset,
		   dst.inst.id, dst.offset, dst.size,
		   srcs.size() - 1,
		   (red_fold ? "fold" : "apply"),
		   redop_id,
		   domain.is_id,
		   before_copy.id,
		   get_finish_event().id);
    }

    ReduceRequest::~ReduceRequest(void)
    {
    }

    size_t ReduceRequest::compute_size(void)
    {
      size_t result = domain.compute_size();
      result += (4 + 3 * srcs.size()) * sizeof(IDType);
      result += sizeof(IDType); // for inst_lock_needed
      // TODO: unbreak once the serialization stuff is repaired
      //result += requests.compute_size();
      ByteCountSerializer counter;
      counter << requests;
      result += sizeof(size_t) + counter.bytes_used();
      return result;
    }

    void ReduceRequest::serialize(void *buffer)
    {
      // domain info goes first
      IDType *msgptr = domain.serialize((IDType *)buffer);

      // now source fields
      *msgptr++ = srcs.size();
      for(std::vector<Domain::CopySrcDstField>::const_iterator it = srcs.begin();
	  it != srcs.end();
	  it++) {
	*msgptr++ = it->inst.id;
	*msgptr++ = it->offset;
	*msgptr++ = it->size;
      }

      // and the dest field
      *msgptr++ = dst.inst.id;
      *msgptr++ = dst.offset;
      *msgptr++ = dst.size;

      *msgptr++ = inst_lock_needed;

      // TODO: unbreak once the serialization stuff is repaired
      //requests.serialize(msgptr);
      // We sent this request remotely so we need to clear it's profiling
      ByteCountSerializer counter;
      counter << requests;
      *reinterpret_cast<size_t*>(msgptr) = counter.bytes_used();
      msgptr += sizeof(size_t) / sizeof(IDType);
      FixedBufferSerializer serializer(msgptr, counter.bytes_used());
      serializer << requests;
      clear_profiling();
    }

    bool ReduceRequest::check_readiness(bool just_check, DmaRequestQueue *rq)
    {
      if(state == STATE_INIT)
	state = STATE_METADATA_FETCH;

      // remember which queue we're going to be assigned to if we sleep
      waiter.req = this;
      waiter.queue = rq;

      // make sure our node has all the meta data it needs, but don't take more than one lock
      //  at a time
      if(state == STATE_METADATA_FETCH) {
	// index space first
	if(domain.get_dim() == 0) {
	  IndexSpaceImpl *is_impl = get_runtime()->get_index_space_impl(domain.get_index_space());
	  if(!is_impl->locked_data.valid) {
	    log_dma.debug("dma request %p - no index space metadata yet", this);
	    if(just_check) return false;

	    Event e = is_impl->lock.acquire(1, false, ReservationImpl::ACQUIRE_BLOCKING);
	    if(e.has_triggered()) {
	      log_dma.debug("request %p - index space metadata invalid - instant trigger", this);
	      is_impl->lock.release();
	    } else {
	      log_dma.debug("request %p - index space metadata invalid - sleeping on lock " IDFMT "", this, is_impl->lock.me.id);
	      waiter.sleep_on_event(e, is_impl->lock.me);
	      return false;
	    }
	  }

          // we need more than just the metadata - we also need the valid mask
          {
            Event e = is_impl->request_valid_mask();
            if(!e.has_triggered()) {
              log_dma.debug("request %p - valid mask needed for index space " IDFMT " - sleeping on event " IDFMT, this, domain.get_index_space().id, e.id);
	      waiter.sleep_on_event(e);
              return false;
            }
          }
	}

	// now go through all source instance pairs
	for(std::vector<Domain::CopySrcDstField>::iterator it = srcs.begin();
	    it != srcs.end();
	    it++) {
	  RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(it->inst);

	  {
	    Event e = src_impl->request_metadata();
	    if(!e.has_triggered()) {
	      if(just_check) {
		log_dma.debug("dma request %p - no src instance (" IDFMT ") metadata yet", this, src_impl->me.id);
		return false;
	      }
	      log_dma.debug("request %p - src instance metadata invalid - sleeping on event " IDFMT, this, e.id);
	      waiter.sleep_on_event(e);
	      return false;
	    }
	  }
	}

	{
	  RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(dst.inst);

	  {
	    Event e = dst_impl->request_metadata();
	    if(!e.has_triggered()) {
	      if(just_check) {
		log_dma.debug("dma request %p - no dst instance (" IDFMT ") metadata yet", this, dst_impl->me.id);
		return false;
	      }
	      log_dma.debug("request %p - dst instance metadata invalid - sleeping on event " IDFMT, this, e.id);
	      waiter.sleep_on_event(e);
	      return false;
	    }
	  }
	}

	// if we got all the way through, we've got all the metadata we need
	state = STATE_BEFORE_EVENT;
      }

      // make sure our functional precondition has occurred
      if(state == STATE_BEFORE_EVENT) {
	// has the before event triggered?  if not, wait on it
	bool poisoned = false;
	if(before_copy.has_triggered_faultaware(poisoned)) {
	  if(poisoned) {
	    log_dma.debug("request %p - poisoned precondition", this);
	    handle_poisoned_precondition(before_copy);
	    return true;  // not enqueued, but never going to be
	  } else {
	    log_dma.debug("request %p - before event triggered", this);
	    if(inst_lock_needed) {
	      // request an exclusive lock on the instance to protect reductions
	      inst_lock_event = get_runtime()->get_instance_impl(dst.inst)->lock.acquire(0, true /*excl*/, ReservationImpl::ACQUIRE_BLOCKING);
	      state = STATE_INST_LOCK;
	      log_dma.debug("request %p - instance lock acquire event " IDFMT,
			    this, inst_lock_event.id);
	    } else {
	      // go straight to ready
	      state = STATE_READY;
	    }
	  }
	} else {
	  log_dma.debug("request %p - before event not triggered", this);
	  if(just_check) return false;

	  log_dma.debug("request %p - sleeping on before event", this);
	  waiter.sleep_on_event(before_copy);
	  return false;
	}
      }

      if(state == STATE_INST_LOCK) {
	if(inst_lock_event.has_triggered()) {
	  log_dma.debug("request %p - instance lock acquired", this);
	  state = STATE_READY;
	} else {
	  log_dma.debug("request %p - instance lock - sleeping on event " IDFMT, this, inst_lock_event.id);
	  waiter.sleep_on_event(inst_lock_event);
	  return false;
	}
      }

      if(state == STATE_READY) {
	log_dma.debug("request %p ready", this);
	if(just_check) return true;

	state = STATE_QUEUED;
#ifdef REDUCE_IN_NEW_DMA
	// <NEWDMA>
	mark_ready();
	bool ok_to_run = mark_started();
	if (ok_to_run) {
	  perform_dma();
	  mark_finished(true/*successful*/);
	} else {
	  mark_finished(false/*!successful*/);
	}
	return true;
	// </NEWDMA>
#endif
	assert(rq != 0);
	log_dma.debug("request %p enqueued", this);

	// once we're enqueued, we may be deleted at any time, so no more
	//  references
	rq->enqueue_request(this);
	return true;
      }

      if(state == STATE_QUEUED)
	return true;

      assert(0);
      return false;
    }

    template <unsigned DIM>
    void ReduceRequest::perform_dma_rect(MemPairCopier *mpc)
    {
      Arrays::Rect<DIM> orig_rect = domain.get_rect<DIM>();

      // empty rectangles are easy to copy...
      if(orig_rect.volume() == 0)
	return;

      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];

      // single source field for now
      assert(srcs.size() == 1);

      // manufacture an OASVec for the copier
      OASVec oasvec(1);
      oasvec[0].src_offset = srcs[0].offset;
      oasvec[0].dst_offset = dst.offset;
      oasvec[0].size = redop->sizeof_rhs;

      RegionInstance src_inst = srcs[0].inst;
      RegionInstance dst_inst = dst.inst;

      InstPairCopier *ipc = mpc->inst_pair(src_inst, dst_inst, oasvec);

      RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(src_inst);
      RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(dst_inst);

      assert(src_impl->metadata.is_valid());
      assert(dst_impl->metadata.is_valid());

      Arrays::Mapping<DIM, 1> *src_linearization = src_impl->metadata.linearization.get_mapping<DIM>();
      Arrays::Mapping<DIM, 1> *dst_linearization = dst_impl->metadata.linearization.get_mapping<DIM>();

      // see what linear subrects we can get - again, iterate over destination first for gathering
      for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lso(orig_rect, *dst_linearization); lso; lso++) {
	for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lsi(lso.subrect, *src_linearization); lsi; lsi++) {
	  // see if we can compress the strides for a more efficient copy
	  Arrays::Point<1> src_cstrides[DIM], dst_cstrides[DIM];
	  Arrays::Point<DIM> extents;
	  int cdim = compress_strides(lsi.subrect, lso.strides, lsi.strides,
				      extents, dst_cstrides, src_cstrides);

#ifdef NEW2D_DEBUG
	  printf("ORIG: (%d,%d,%d)->(%d,%d,%d)\n",
		 orig_rect.lo[0], orig_rect.lo[1], orig_rect.lo[2],
		 orig_rect.hi[0], orig_rect.hi[1], orig_rect.hi[2]);
	  printf("DST:  (%d,%d,%d)->(%d,%d,%d)  %d+(%d,%d,%d)\n",
		 lso.subrect.lo[0], lso.subrect.lo[1], lso.subrect.lo[2],
		 lso.subrect.hi[0], lso.subrect.hi[1], lso.subrect.hi[2],
		 lso.image_lo[0],
		 lso.strides[0][0], lso.strides[1][0], lso.strides[2][0]);
	  printf("SRC:  (%d,%d,%d)->(%d,%d,%d)  %d+(%d,%d,%d)\n",
		 lsi.subrect.lo[0], lsi.subrect.lo[1], lsi.subrect.lo[2],
		 lsi.subrect.hi[0], lsi.subrect.hi[1], lsi.subrect.hi[2],
		 lsi.image_lo[0],
		 lsi.strides[0][0], lsi.strides[1][0], lsi.strides[2][0]);
	  printf("CMP:  %d (%d,%d,%d) +(%d,%d,%d) +(%d,%d,%d)\n",
		 cdim,
		 extents[0], extents[1], extents[2],
		 dst_cstrides[0][0], dst_cstrides[1][0], dst_cstrides[2][0],
		 src_cstrides[0][0], src_cstrides[1][0], src_cstrides[2][0]);
#endif
	  if((cdim == 1) && (dst_cstrides[0][0] == 1) && (src_cstrides[0][0] == 1)) {
	    // all the dimension(s) collapse to a 1-D extent in both source and dest, so one big copy
	    ipc->copy_all_fields(lsi.image_lo[0], lso.image_lo[0], extents[0]);
	    continue;
	  }

	  if((cdim == 2) && (dst_cstrides[0][0] == 1) && (src_cstrides[0][0] == 1)) {
	    // source and/or destination need a 2-D description
	    ipc->copy_all_fields(lsi.image_lo[0], lso.image_lo[0], extents[0],
				 src_cstrides[1][0], dst_cstrides[1][0], extents[1]);
	    continue;
	  }

	  // fall through - just identify dense (sub)subrects and copy them
	  
	  // iterate by output rectangle first - this gives us a chance to gather data when linearizations
	  //  don't match up
	  for(Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > dso(lsi.subrect, *dst_linearization); dso; dso++) {
	    // dense subrect in dst might not be dense in src
	    for(Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > dsi(dso.subrect, *src_linearization); dsi; dsi++) {
	      Arrays::Rect<1> irect = dsi.image;
	      // rectangle in output must be recalculated
	      Arrays::Rect<DIM> subrect_check;
	      Arrays::Rect<1> orect = dst_linearization->image_dense_subrect(dsi.subrect, subrect_check);
	      assert(dsi.subrect == subrect_check);
	      
	      //for(OASVec::iterator it2 = oasvec.begin(); it2 != oasvec.end(); it2++)
	      for (unsigned idx = 0; idx < oasvec.size(); idx++)
		ipc->copy_field(irect.lo, orect.lo, irect.hi[0] - irect.lo[0] + 1, idx);
	      //it2->src_offset, it2->dst_offset, it2->size);
	    }
	  }
	}
      }
      
      delete ipc;
    }

    void ReduceRequest::perform_dma(void)
    {
      log_dma.debug("request %p executing", this);

      DetailedTimer::ScopedPush sp(TIME_COPY);

      // code assumes a single source field for now
      assert(srcs.size() == 1);

      Memory src_mem = get_runtime()->get_instance_impl(srcs[0].inst)->memory;
      Memory dst_mem = get_runtime()->get_instance_impl(dst.inst)->memory;
      MemoryImpl::MemoryKind src_kind = get_runtime()->get_memory_impl(src_mem)->kind;
      MemoryImpl::MemoryKind dst_kind = get_runtime()->get_memory_impl(dst_mem)->kind;

      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];

      //printf("kinds: " IDFMT "=%d " IDFMT "=%d\n", src_mem.id, src_mem.impl()->kind, dst_mem.id, dst_mem.impl()->kind);

      // we have the same jumble of memory type and layout permutations here - again we'll
      //  solve a few of them point-wise and then try to unify later
      size_t total_bytes = 0;
      if(domain.get_dim() == 0) {
	// index space
	IndexSpaceImpl *ispace = get_runtime()->get_index_space_impl(domain.get_index_space());
	assert(ispace->valid_mask_complete);

	if((src_kind == MemoryImpl::MKIND_SYSMEM) ||
	   (src_kind == MemoryImpl::MKIND_ZEROCOPY) ||
	   (src_kind == MemoryImpl::MKIND_RDMA)) {
	  void *src_base = 0;
	  size_t src_stride = 0;
#ifndef NDEBUG
	  bool src_ok =
#endif
	    get_runtime()->get_instance_impl(srcs[0].inst)->get_strided_parameters(src_base, src_stride,
											       srcs[0].offset);
	  assert(src_ok);

	  switch(dst_kind) {
	  case MemoryImpl::MKIND_SYSMEM:
	  case MemoryImpl::MKIND_ZEROCOPY:
	    {
	      void *dst_base = 0;
	      size_t dst_stride = 0;
#ifndef NDEBUG
	      bool dst_ok =
#endif
		get_runtime()->get_instance_impl(dst.inst)->get_strided_parameters(dst_base, dst_stride,
											       dst.offset);
	      assert(dst_ok);

	      // if source and dest are ok, we can just walk the index space's spans
	      ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	      Arrays::coord_t rstart; size_t rlen;
	      while(e->get_next(rstart, rlen)) {
		if(red_fold)
		  redop->fold_strided(((char *)dst_base) + (rstart * dst_stride),
				      ((const char *)src_base) + (rstart * src_stride),
				      dst_stride, src_stride, rlen,
				      false /*not exclusive*/);
		else
		  redop->apply_strided(((char *)dst_base) + (rstart * dst_stride),
				       ((const char *)src_base) + (rstart * src_stride),
				       dst_stride, src_stride, rlen,
				       false /*not exclusive*/);
	      }

              delete e;
	      break;
	    }

	  case MemoryImpl::MKIND_REMOTE:
	  case MemoryImpl::MKIND_RDMA:
            {
	      // we need to figure out how to calculate offsets in the destination memory
	      RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(dst.inst);

	      assert(dst_impl->metadata.is_valid());

	      off_t dst_field_start=0;
	      int dst_field_size=0;
	      find_field_start(dst_impl->metadata.field_sizes, dst.offset, dst.size, dst_field_start, dst_field_size);
	      assert(dst.size == (size_t)dst_field_size);

	      // index space instances use 1D linearizations for translation
	      Arrays::Mapping<1, 1> *dst_linearization = dst_impl->metadata.linearization.get_mapping<1>();

	      ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	      Arrays::coord_t rstart; size_t rlen;

	      // get an RDMA sequence number so we can have the far side trigger the event once all reductions have been
	      //  applied
	      unsigned sequence_id = __sync_fetch_and_add(&rdma_sequence_no, 1);
	      unsigned rdma_count = 0;

	      // for a reduction from a fold instance, it's always ok to copy unused elements, since they'll have an
	      //  identity value stored for them
	      size_t rlen_target = 32768 / dst_field_size;

	      while(e->get_next(rstart, rlen)) {
		// do we want to copy extra elements to fill in some holes?
		while(rlen < rlen_target) {
		  // see where the next valid elements are
		  Arrays::coord_t rstart2; size_t rlen2;
		  // if none, stop
		  if(!e->peek_next(rstart2, rlen2)) break;
		  // or if they don't even start until outside the window, stop
		  if(rstart2 > (rstart + (Arrays::coord_t)rlen_target)) break;
		  // ok, include the next valid element(s) and any invalid ones in between
		  //printf("bloating from %d to %d\n", rlen, rstart2 + rlen2 - rstart);
		  rlen = rstart2 + rlen2 - rstart;
		  // and actually take the next range from the enumerator
		  e->get_next(rstart2, rlen2);
		}

		// translate the index space point to the dst instance's linearization
		int dstart = dst_linearization->image((Arrays::coord_t)rstart);

		// now do an offset calculation for the destination
		off_t dst_offset;
		off_t dst_stride;
		if(dst_impl->metadata.block_size > 1) {
		  // straddling a block boundary is complicated
		  assert((dstart / dst_impl->metadata.block_size) == ((dstart + rlen - 1) / dst_impl->metadata.block_size));
		  dst_offset = calc_mem_loc(dst_impl->metadata.alloc_offset, dst_field_start, dst_field_size,
					    dst_impl->metadata.elmt_size, dst_impl->metadata.block_size, dstart);
		  dst_stride = dst_field_size;
		} else {
		  // easy case
		  dst_offset = dst_impl->metadata.alloc_offset + (dstart * dst_impl->metadata.elmt_size) + dst_field_start;
		  dst_stride = dst_impl->metadata.elmt_size;
		}

		rdma_count += do_remote_reduce(dst_mem, dst_offset, redop_id, red_fold,
					       ((const char *)src_base) + (rstart * src_stride),
					       rlen, src_stride, dst_stride,
					       sequence_id);
	      }

	      // if we did any actual reductions, send a fence, otherwise trigger here
	      if(rdma_count > 0) {
                Realm::RemoteWriteFence *fence = new Realm::RemoteWriteFence(this);
                this->add_async_work_item(fence);
		do_remote_fence(dst_mem, sequence_id, rdma_count, fence);
	      }
              delete e;
	      break;
            }

	  case MemoryImpl::MKIND_GLOBAL:
	    {
	      // make sure we've requested a lock on the dst instance
	      assert(inst_lock_needed);

	      // we need to figure out how to calculate offsets in the destination memory
	      RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(dst.inst);

	      assert(dst_impl->metadata.is_valid());

	      off_t dst_field_start=0;
	      int dst_field_size=0;
	      find_field_start(dst_impl->metadata.field_sizes, dst.offset, dst.size, dst_field_start, dst_field_size);
	      assert(dst.size == (size_t)dst_field_size);

	      // index space instances use 1D linearizations for translation
	      Arrays::Mapping<1, 1> *dst_linearization = dst_impl->metadata.linearization.get_mapping<1>();

	      // if source and dest are ok, we can just walk the index space's spans
	      ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	      Arrays::coord_t rstart; size_t rlen;
	      while(e->get_next(rstart, rlen)) {
		// translate the index space point to the dst instance's linearization
		int dstart = dst_linearization->image((Arrays::coord_t)rstart);

		// now do an offset calculation for the destination
		off_t dst_offset;
		off_t dst_stride;
		if(dst_impl->metadata.block_size > 1) {
		  // straddling a block boundary is complicated
		  assert((dstart / dst_impl->metadata.block_size) == ((dstart + rlen - 1) / dst_impl->metadata.block_size));
		  dst_offset = calc_mem_loc(dst_impl->metadata.alloc_offset, dst_field_start, dst_field_size,
					    dst_impl->metadata.elmt_size, dst_impl->metadata.block_size, dstart);
		  dst_stride = dst_field_size;
		} else {
		  // easy case
		  dst_offset = dst_impl->metadata.alloc_offset + (dstart * dst_impl->metadata.elmt_size) + dst_field_start;
		  dst_stride = dst_impl->metadata.elmt_size;
		}

		// get a temporary buffer in local memory
		// this may have extra data if stride > field_size, but that's
		//  ok - we'll write back whatever we read
		void *buffer = malloc(dst_stride * rlen);

		get_runtime()->get_memory_impl(dst_mem)->get_bytes(dst_offset, buffer, dst_stride * rlen);

		if(red_fold)
		  redop->fold_strided(buffer,
				      ((const char *)src_base) + (rstart * src_stride),
				      dst_stride, src_stride, rlen,
				      true /*exclusive*/);
		else
		  redop->apply_strided(buffer,
				       ((const char *)src_base) + (rstart * src_stride),
				       dst_stride, src_stride, rlen,
				       true /*exclusive*/);

		get_runtime()->get_memory_impl(dst_mem)->put_bytes(dst_offset, buffer, dst_stride * rlen);

		// release the temp buffer
		free(buffer);
	      }

	      // also release the instance lock
	      dst_impl->lock.release();

              delete e;

	      break;
	    }

	  default:
	    assert(0);
	  }
	}
        // TODO: we don't track the size of reduction on unstructred index spaces
        total_bytes = 0;
      } else {
	MemPairCopier *mpc = MemPairCopier::create_copier(src_mem, dst_mem, redop_id, red_fold);

	switch(domain.get_dim()) {
	case 1: perform_dma_rect<1>(mpc); break;
	case 2: perform_dma_rect<2>(mpc); break;
	case 3: perform_dma_rect<3>(mpc); break;
	default: assert(0);
	}

	mpc->flush(this);
        total_bytes = mpc->get_total_bytes();

	// if an instance lock was taken, release it after copy completes
	if(inst_lock_needed)
	  get_runtime()->get_instance_impl(dst.inst)->lock.me.release(get_finish_event());

	delete mpc;
      }

      log_dma.info("dma request %p finished - " IDFMT "[%lld]->" IDFMT "[%lld]:%zu (+%zu) %s %d (" IDFMT ") " IDFMT " " IDFMT,
		   this,
		   srcs[0].inst.id, srcs[0].offset,
		   dst.inst.id, dst.offset, dst.size,
		   srcs.size() - 1,
		   (red_fold ? "fold" : "apply"),
		   redop_id,
		   domain.is_id,
		   before_copy.id,
		   get_finish_event().id);

      if(measurements.wants_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>()) {
        Realm::ProfilingMeasurements::OperationMemoryUsage usage;  
        // Not precise, but close enough for now
        usage.source = srcs[0].inst.get_location();
        usage.target = dst.inst.get_location();
        usage.size = total_bytes;
        measurements.add_measurement(usage);
      }
    }

    FillRequest::FillRequest(const void *data, size_t datalen,
                             RegionInstance inst,
                             unsigned offset, unsigned size,
                             Event _before_fill, Event _after_fill,
                             int _priority)
      : DmaRequest(_priority, _after_fill), before_fill(_before_fill)
    {
      dst.inst = inst;
      dst.offset = offset;
      dst.size = size;

      const IDType *idata = (const IDType *)data;

      idata = domain.deserialize(idata);

      size_t elmts = *idata++;

      fill_size = dst.size;
      fill_buffer = malloc(fill_size);
      memcpy(fill_buffer, idata, fill_size);

      idata += elmts;

      // TODO: unbreak once the serialization stuff is repaired
      //const void *result = requests.deserialize(idata);
      //Realm::Operation::reconstruct_measurements();

      // better have consumed exactly the right amount of data
      //assert((((unsigned long)result) - ((unsigned long)data)) == datalen);
      size_t request_size = *reinterpret_cast<const size_t*>(idata);
      idata += sizeof(size_t) / sizeof(IDType);
      FixedBufferDeserializer deserializer(idata, request_size);
      deserializer >> requests;
      Realm::Operation::reconstruct_measurements();

      log_dma.info() << "dma request " << (void *)this << " deserialized - is="
		     << domain << " fill dst=" << dst.inst << "[" << dst.offset << "+" << dst.size << "] size="
		     << fill_size << " before=" << _before_fill << " after=" << _after_fill;
    }

    FillRequest::FillRequest(const Domain &d, 
                             const Domain::CopySrcDstField &_dst,
                             const void *_fill_value, size_t _fill_size,
                             Event _before_fill, Event _after_fill, int _priority,
                             const Realm::ProfilingRequestSet &reqs)
      : DmaRequest(_priority, _after_fill, reqs), domain(d), dst(_dst),
        before_fill(_before_fill)
    {
      fill_size = _fill_size;
      fill_buffer = malloc(fill_size);
      memcpy(fill_buffer, _fill_value, fill_size);

      log_dma.info() << "dma request " << (void *)this << " created - is="
		     << d << " fill dst=" << dst.inst << "[" << dst.offset << "+" << dst.size << "] size="
		     << fill_size << " before=" << _before_fill << " after=" << _after_fill;
      {
	Realm::LoggerMessage msg(log_dma.debug());
	if(msg.is_active()) {
	  msg << "fill data =";
	  msg << std::hex;
	  for(size_t i = 0; i < _fill_size; i++)
	    msg << ' ' << std::setfill('0') << std::setw(2) << (unsigned)((unsigned char *)(_fill_value))[i];
	  msg << std::dec;
	}
      }
    }

    FillRequest::~FillRequest(void)
    {
      // clean up our mess
      free(fill_buffer);
    }

    size_t FillRequest::compute_size(void)
    {
      size_t result = domain.compute_size();
      size_t elmts = (fill_size + sizeof(IDType) - 1)/sizeof(IDType);
      result += ((elmts+1) * sizeof(IDType)); // +1 for fill size in bytes
      // TODO: unbreak once the serialization stuff is repaired
      //result += requests.compute_size();
      ByteCountSerializer counter;
      counter << requests;
      result += sizeof(size_t) + counter.bytes_used();
      return result;
    }

    void FillRequest::serialize(void *buffer)
    {
      IDType *msgptr = domain.serialize((IDType *)buffer);
      
      assert(dst.size == fill_size);
      size_t elmts = (fill_size + sizeof(IDType) - 1)/sizeof(IDType);
      *msgptr++ = elmts;
      memcpy(msgptr, fill_buffer, fill_size);
      msgptr += elmts;

      // TODO: unbreak once the serialization stuff is repaired
      //requests.serialize(msgptr);
      // We sent this message remotely, so we need to clear the profiling
      // so it doesn't get sent accidentally
      ByteCountSerializer counter;
      counter << requests;
      *reinterpret_cast<size_t*>(msgptr) = counter.bytes_used();
      msgptr += sizeof(size_t) / sizeof(IDType);
      FixedBufferSerializer serializer(msgptr, counter.bytes_used());
      serializer << requests;
      clear_profiling();
    }

    bool FillRequest::check_readiness(bool just_check, DmaRequestQueue *rq)
    {
      if(state == STATE_INIT)
	state = STATE_METADATA_FETCH;

      // remember which queue we're going to be assigned to if we sleep
      waiter.req = this;
      waiter.queue = rq;

      // make sure our node has all the meta data it needs, but don't take more than one lock
      //  at a time
      if(state == STATE_METADATA_FETCH) {
        // index space first
	if(domain.get_dim() == 0) {
	  IndexSpaceImpl *is_impl = get_runtime()->get_index_space_impl(domain.get_index_space());
	  if(!is_impl->locked_data.valid) {
	    log_dma.debug("dma request %p - no index space metadata yet", this);
	    if(just_check) return false;

	    Event e = is_impl->lock.acquire(1, false, ReservationImpl::ACQUIRE_BLOCKING);
	    if(e.has_triggered()) {
	      log_dma.debug("request %p - index space metadata invalid - instant trigger", this);
	      is_impl->lock.release();
	    } else {
	      log_dma.debug("request %p - index space metadata invalid - sleeping on lock " IDFMT "", this, is_impl->lock.me.id);
	      waiter.sleep_on_event(e, is_impl->lock.me);
	      return false;
	    }
	  }

          // we need more than just the metadata - we also need the valid mask
          {
            Event e = is_impl->request_valid_mask();
            if(!e.has_triggered()) {
              log_dma.debug("request %p - valid mask needed for index space " IDFMT " - sleeping on event " IDFMT, this, domain.get_index_space().id, e.id);
	      waiter.sleep_on_event(e);
              return false;
            }
          }
	}
        // No need to check the instance, we are on its local node 
        state = STATE_BEFORE_EVENT;
      }

      // make sure our functional precondition has occurred
      if(state == STATE_BEFORE_EVENT) {
	// has the before event triggered?  if not, wait on it
        bool poisoned = false;
	if(before_fill.has_triggered_faultaware(poisoned)) {
          if(poisoned) {
            log_dma.debug("request %p - poisoned precondition", this);
            handle_poisoned_precondition(before_fill);
            return true; // not enqueued, but never going to be
          } else {
            log_dma.debug("request %p - before event triggered", this);
            state = STATE_READY;
          }
	} else {
	  log_dma.debug("request %p - before event not triggered", this);
	  if(just_check) return false;

	  log_dma.debug("request %p - sleeping on before event", this);
	  waiter.sleep_on_event(before_fill);
	  return false;
	}
      }

      if(state == STATE_READY) {
	log_dma.debug("request %p ready", this);
	if(just_check) return true;

	state = STATE_QUEUED;
#ifdef FILL_IN_NEW_DMA
	// <NEWDMA>
	mark_ready();
	bool ok_to_run = mark_started();
	if (ok_to_run) {
	  perform_dma();
	  mark_finished(true/*successful*/);
	} else {
	  mark_finished(false/*!successful*/);
	}
	return true;
	// </NEWDMA>
#endif
	assert(rq != 0);
	log_dma.debug("request %p enqueued", this);

	// once we're enqueued, we may be deleted at any time, so no more
	//  references
	rq->enqueue_request(this);
	return true;
      }

      if(state == STATE_QUEUED)
	return true;

      assert(0);
      return false;
    }

    void FillRequest::perform_dma(void)
    {
      // First switch on the memory type
      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(dst.inst.get_location());

      MemoryImpl::MemoryKind mem_kind = mem_impl->kind;
      // TODO: optimize transfers for framebuffer to use a memset kernel
      if ((mem_kind == MemoryImpl::MKIND_SYSMEM) ||
          (mem_kind == MemoryImpl::MKIND_ZEROCOPY) ||
          (mem_kind == MemoryImpl::MKIND_RDMA) ||
          (mem_kind == MemoryImpl::MKIND_GPUFB) ||
          (mem_kind == MemoryImpl::MKIND_ZEROCOPY))
      {
        switch (domain.get_dim()) {
          case 0:
            {
              // Iterate over all the points and get the 
              IndexSpaceImpl *ispace = get_runtime()->get_index_space_impl(domain.get_index_space());
              assert(ispace->valid_mask_complete);
              RegionInstanceImpl *inst_impl = get_runtime()->get_instance_impl(dst.inst);
              off_t field_start=0; int field_size=0;
              find_field_start(inst_impl->metadata.field_sizes, dst.offset,
                               dst.size, field_start, field_size);
              assert(field_size <= int(fill_size));
              int fill_elmts = 1;
              // Optimize our buffer for the target instance
              size_t fill_elmts_size = optimize_fill_buffer(inst_impl, fill_elmts);
              Arrays::Mapping<1, 1> *dst_linearization = 
                inst_impl->metadata.linearization.get_mapping<1>();
              ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
              Arrays::coord_t rstart; size_t elem_count;
              while(e->get_next(rstart, elem_count)) {
                int dst_index = dst_linearization->image((Arrays::coord_t)rstart); 
                size_t done = 0;
                while (done < elem_count) {
                  int dst_in_this_block = inst_impl->metadata.block_size - 
                              ((dst_index + done) % inst_impl->metadata.block_size);
                  int todo = min(elem_count, dst_in_this_block);
                  off_t dst_start = calc_mem_loc(inst_impl->metadata.alloc_offset,
                                                 field_start, field_size, 
                                                 inst_impl->metadata.elmt_size,
                                                 inst_impl->metadata.block_size,
                                                 dst_index + done);
                  // Record how many we've done
                  done += todo;
                  // Now do as many bulk transfers as we can
                  while (todo >= fill_elmts) {
                    mem_impl->put_bytes(dst_start, fill_buffer, fill_elmts_size);
                    dst_start += fill_elmts_size;
                    todo -= fill_elmts;
                  }
                  // Handle any remainder elemts
                  if (todo > 0) {
                    mem_impl->put_bytes(dst_start, fill_buffer, todo*fill_size);
                  }
                }
              }
              delete e;
              break;
            }
          case 1:
            {
              perform_dma_rect<1>(mem_impl);
              break;
            }
          case 2:
            {
              perform_dma_rect<2>(mem_impl); 
              break;
            }
          case 3:
            {
              perform_dma_rect<3>(mem_impl); 
              break;
            }
          default:
            assert(false);
        }
      } else {
        // TODO: Implement GASNet and Disk
        assert(false);
      }

      if(measurements.wants_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>()) {
        Realm::ProfilingMeasurements::OperationMemoryUsage usage;
        usage.source = Memory::NO_MEMORY;
        usage.target = dst.inst.get_location();
        measurements.add_measurement(usage);
      }
    }

    template<int DIM>
    void FillRequest::perform_dma_rect(MemoryImpl *mem_impl)
    {
      typename Arrays::Rect<DIM> rect = domain.get_rect<DIM>();
      // empty rectangles are easy to fill...
      if(rect.volume() == 0)
	return;

      RegionInstanceImpl *inst_impl = get_runtime()->get_instance_impl(dst.inst);
      off_t field_start=0; int field_size=0;
      find_field_start(inst_impl->metadata.field_sizes, dst.offset,
                       dst.size, field_start, field_size);
      assert(field_size <= (int)fill_size);
      typename Arrays::Mapping<DIM, 1> *dst_linearization = 
        inst_impl->metadata.linearization.get_mapping<DIM>();

      int fill_elmts = 1;
      // Optimize our buffer for the target instance
      size_t fill_elmts_size = optimize_fill_buffer(inst_impl, fill_elmts);
      for (typename Arrays::Mapping<DIM, 1>::DenseSubrectIterator dso(rect, 
            *dst_linearization); dso; dso++) {
        int dst_index = dso.image.lo[0];
        int elem_count = dso.subrect.volume();
        int done = 0; 
        while (done < elem_count) {
          int dst_in_this_block = inst_impl->metadata.block_size - 
                      ((dst_index + done) % inst_impl->metadata.block_size);
          int todo = min(elem_count, dst_in_this_block);
          off_t dst_start = calc_mem_loc(inst_impl->metadata.alloc_offset,
                                         field_start, field_size, 
                                         inst_impl->metadata.elmt_size,
                                         inst_impl->metadata.block_size,
                                         dst_index + done);
          // Record how many we've done
          done += todo;
          // Now do as many bulk transfers as we can
          while (todo >= fill_elmts) {
            mem_impl->put_bytes(dst_start, fill_buffer, fill_elmts_size); 
            dst_start += fill_elmts_size;
            todo -= fill_elmts;
          }
          // Handle any remainder elemts
          if (todo > 0) {
            mem_impl->put_bytes(dst_start, fill_buffer, todo*fill_size);
          }
        }
      }
    }

    size_t FillRequest::optimize_fill_buffer(RegionInstanceImpl *inst_impl, int &fill_elmts)
    {
      const size_t max_size = 1024; 
      // Only do this optimization for "small" fields
      // which are less than half a page
      if (fill_size <= max_size)
      {
        // If we have a single-field instance or we have a set 
        // of contiguous elmts then make a bulk buffer to use
        if ((inst_impl->metadata.elmt_size == fill_size) ||
            (inst_impl->metadata.block_size > 1)) 
        {
          fill_elmts = min(inst_impl->metadata.block_size,2*max_size/fill_size);
          size_t fill_elmts_size = fill_elmts * fill_size;
          char *next_buffer = (char*)malloc(fill_elmts_size);
          char *next_ptr = next_buffer;
          for (int idx = 0; idx < fill_elmts; idx++) {
            memcpy(next_ptr, fill_buffer, fill_size);
            next_ptr += fill_size;
          }
          // Free the old buffer and replace it
          free(fill_buffer);
          fill_buffer = next_buffer;
          return fill_elmts_size;
        }
      }
      return fill_size;
    }

#if 0
    class CopyCompletionProfiler : public EventWaiter, public Realm::Operation::AsyncWorkItem {
      public:
        CopyCompletionProfiler(DmaRequest* _req)
	  : Realm::Operation::AsyncWorkItem(_req), req(_req)
        { }

        virtual ~CopyCompletionProfiler(void) { }

        virtual bool event_triggered(Event e)
        {
          mark_finished();
          return false;
        }

        virtual void print_info(FILE *f)
        {
          fprintf(f, "copy completion profiler - " IDFMT "/%d\n",
              req->get_finish_event().id, req->get_finish_event().gen);
        }

        virtual void request_cancellation(void)
        {
	  // ignored for now
	}

      protected:
        DmaRequest* req;
    };
#endif

    // for now we use a single queue for all (local) dmas
    static DmaRequestQueue *dma_queue = 0;
    
    void DmaRequestQueue::worker_thread_loop(void)
    {
      log_dma.info("dma worker thread created");

      while(!shutdown_flag) {
	aio_context->make_progress();
	bool aio_idle = aio_context->empty();

	// get a request, sleeping as necessary
	DmaRequest *r = dequeue_request(aio_idle);

	if(r) {
          bool ok_to_run = r->mark_started();
	  if(ok_to_run) {
	    // this will automatically add any necessary AsyncWorkItem's
	    r->perform_dma();

	    r->mark_finished(true /*successful*/);
	  } else
	    r->mark_finished(false /*!successful*/);
	}
      }

      log_dma.info("dma worker thread terminating");
    }

    void DmaRequestQueue::start_workers(int count)
    {
      ThreadLaunchParameters tlp;

      for(int i = 0; i < count; i++) {
	Thread *t = Thread::create_kernel_thread<DmaRequestQueue,
						 &DmaRequestQueue::worker_thread_loop>(this,
										       tlp,
										       core_rsrv,
										       0 /* default scheduler*/);
	worker_threads.push_back(t);
      }
    }
    
    void start_dma_worker_threads(int count, Realm::CoreReservationSet& crs)
    {
      dma_queue = new DmaRequestQueue(crs);
      dma_queue->start_workers(count);
    }

    void stop_dma_worker_threads(void)
    {
      dma_queue->shutdown_queue();
      delete dma_queue;
      dma_queue = 0;
    }

    void start_dma_system(int count, int max_nr, Realm::CoreReservationSet& crs)
    {
      //log_dma.add_stream(&std::cerr, Logger::Category::LEVEL_DEBUG, false, false);
      aio_context = new AsyncFileIOContext(256);
      start_channel_manager(count, max_nr, crs);
      ib_req_queue = new PendingIBQueue();
    }

    void stop_dma_system(void)
    {
      stop_channel_manager();
      delete ib_req_queue;
      ib_req_queue = 0;
      delete aio_context;
      aio_context = 0;
    }
  };
};

namespace Realm {

  using namespace LegionRuntime::LowLevel;

    Event Domain::fill(const std::vector<CopySrcDstField> &dsts,
                       const void *fill_value, size_t fill_value_size,
                       Event wait_on /*= Event::NO_EVENT*/) const
    {
      Realm::ProfilingRequestSet reqs;
      return Domain::fill(dsts, reqs, fill_value, fill_value_size, wait_on);
    }

    Event Domain::fill(const std::vector<CopySrcDstField> &dsts,
                       const Realm::ProfilingRequestSet &requests,
                       const void *fill_value, size_t fill_value_size,
                       Event wait_on /*= Event::NO_EVENT*/) const
    {
      std::set<Event> finish_events; 
      // when 'dsts' contains multiple fields, the 'fill_value' should look
      // like a packed struct with a fill value for each field in order -
      // track the offset and complain if we run out of data
      size_t fill_ofs = 0;
      for (std::vector<CopySrcDstField>::const_iterator it = dsts.begin();
            it != dsts.end(); it++)
      {
        Event ev = GenEventImpl::create_genevent()->current_event();
	if((fill_ofs + it->size) > fill_value_size) {
	  log_dma.fatal() << "insufficient data for fill - need at least "
			  << (fill_ofs + it->size) << " bytes, but have only " << fill_value_size;
	  assert(0);
	}
        FillRequest *r = new FillRequest(*this, *it,
					 ((const char *)fill_value) + fill_ofs,
					 it->size, wait_on,
                                         ev, 0/*priority*/, requests);
	// special case: if a field uses all of the fill value, the next
	//  field (if any) is allowed to use the same value
	if((fill_ofs > 0) || (it->size != fill_value_size))
	  fill_ofs += it->size;

        Memory mem = it->inst.get_location();
        unsigned node = ID(mem).memory.owner_node;
	if(node > ID::MAX_NODE_ID) {
	  assert(0 && "fills to GASNet memory not supported yet");
	  return Event::NO_EVENT;
	}
        if (node == (unsigned)gasnet_mynode()) {
	  get_runtime()->optable.add_local_operation(ev, r);
          r->check_readiness(false, dma_queue);
        } else {
          RemoteFillArgs args;
          args.inst = it->inst;
          args.offset = it->offset;
          args.size = it->size;
          args.before_fill = wait_on;
          args.after_fill = ev;
          //args.priority = 0;

          size_t msglen = r->compute_size();
          void *msgdata = malloc(msglen);

          r->serialize(msgdata);

	  get_runtime()->optable.add_remote_operation(ev, node);

          RemoteFillMessage::request(node, args, msgdata, msglen, PAYLOAD_FREE);

	  // release local copy of operation
	  r->remove_reference();
        }
        finish_events.insert(ev);
      }
      return GenEventImpl::merge_events(finish_events, false /*!ignore faults*/);
    }
    
    Event Domain::copy(RegionInstance src_inst, RegionInstance dst_inst,
		       size_t elem_size, Event wait_on,
		       ReductionOpID redop_id, bool red_fold) const
    {
      assert(0);
      std::vector<CopySrcDstField> srcs, dsts;
      srcs.push_back(CopySrcDstField(src_inst, 0, elem_size));
      dsts.push_back(CopySrcDstField(dst_inst, 0, elem_size));
      return copy(srcs, dsts, wait_on, redop_id, red_fold);
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    static int select_dma_node(Memory src_mem, Memory dst_mem,
			       ReductionOpID redop_id, bool red_fold)
    {
      int src_node = ID(src_mem).memory.owner_node;
      int dst_node = ID(dst_mem).memory.owner_node;

      bool src_is_rdma = get_runtime()->get_memory_impl(src_mem)->kind == MemoryImpl::MKIND_GLOBAL;
      bool dst_is_rdma = get_runtime()->get_memory_impl(dst_mem)->kind == MemoryImpl::MKIND_GLOBAL;

      if(src_is_rdma) {
	if(dst_is_rdma) {
	  // gasnet -> gasnet - blech
	  log_dma.warning("WARNING: gasnet->gasnet copy being serialized on local node (%d)", gasnet_mynode());
	  return gasnet_mynode();
	} else {
	  // gathers by the receiver
	  return dst_node;
	}
      } else {
	if(dst_is_rdma) {
	  // writing to gasnet is also best done by the sender
	  return src_node;
	} else {
	  // if neither side is gasnet, favor the sender (which may be the same as the target)
	  return src_node;
	}
      }
    }

    void handle_remote_copy(RemoteCopyArgs args, const void *data, size_t msglen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      // is this a copy or a reduction (they deserialize differently)
      if(args.redop_id == 0) {
	// a copy
	CopyRequest *r = new CopyRequest(data, msglen,
					 args.before_copy,
					 args.after_copy,
					 args.priority);
	Realm::get_runtime()->optable.add_local_operation(args.after_copy, r);

	r->check_readiness(false, dma_queue);
      } else {
	// a reduction
	ReduceRequest *r = new ReduceRequest(data, msglen,
					     args.redop_id,
					     args.red_fold,
					     args.before_copy,
					     args.after_copy,
					     args.priority);
	Realm::get_runtime()->optable.add_local_operation(args.after_copy, r);

	r->check_readiness(false, dma_queue);
      }
    }

    void handle_remote_fill(RemoteFillArgs args, const void *data, size_t msglen)
    {
      FillRequest *r = new FillRequest(data, msglen,
                                       args.inst,
                                       args.offset,
                                       args.size,
                                       args.before_fill,
                                       args.after_fill,
                                       0 /* no room for args.priority */);
      Realm::get_runtime()->optable.add_local_operation(args.after_fill, r);

      r->check_readiness(false, dma_queue);
    }

    template <typename T> T min(T a, T b) { return (a < b) ? a : b; }

  };
};

namespace Realm {

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       Event wait_on,
		       ReductionOpID redop_id, bool red_fold) const
    {
      Realm::ProfilingRequestSet reqs;
      return Domain::copy(srcs, dsts, reqs, wait_on, redop_id, red_fold);
    }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
                       const Realm::ProfilingRequestSet &requests,
		       Event wait_on,
		       ReductionOpID redop_id, bool red_fold) const
    {
      if(redop_id == 0) {
	// not a reduction, so sort fields by src/dst mem pairs
        //log_new_dma.info("Performing copy op");

	OASByMem oas_by_mem;

	std::vector<CopySrcDstField>::const_iterator src_it = srcs.begin();
	std::vector<CopySrcDstField>::const_iterator dst_it = dsts.begin();
	unsigned src_suboffset = 0;
	unsigned dst_suboffset = 0;
	std::set<Event> finish_events;

	while((src_it != srcs.end()) && (dst_it != dsts.end())) {
	  InstPair ip(src_it->inst, dst_it->inst);
	  MemPair mp(get_runtime()->get_instance_impl(src_it->inst)->memory,
		     get_runtime()->get_instance_impl(dst_it->inst)->memory);

	  // printf("I:(%x/%x) M:(%x/%x) sub:(%d/%d) src=(%d/%d) dst=(%d/%d)\n",
	  //        ip.first.id, ip.second.id, mp.first.id, mp.second.id,
	  //        src_suboffset, dst_suboffset,
	  //        src_it->offset, src_it->size, 
	  //        dst_it->offset, dst_it->size);

	  OffsetsAndSize oas;
	  oas.src_offset = src_it->offset + src_suboffset;
	  oas.dst_offset = dst_it->offset + dst_suboffset;
	  oas.size = min(src_it->size - src_suboffset, dst_it->size - dst_suboffset);
	  oas.serdez_id = src_it->serdez_id;
	  // <SERDEZ_DMA>
	  // This is a little bit of hack: if serdez_id != 0 we directly create a
	  // CopyRequest instead of inserting it into ''oasvec''
	  if (oas.serdez_id != 0) {
	    OASByInst* oas_by_inst = new OASByInst;
	    (*oas_by_inst)[ip].push_back(oas);
	    Event ev = GenEventImpl::create_genevent()->current_event();
	    int priority = 0; // always have priority zero
	    CopyRequest *r = new CopyRequest(*this, oas_by_inst,
  					     wait_on, ev, priority, requests);
            // ask which node should perform the copy
            int dma_node = select_dma_node(mp.first, mp.second, redop_id, red_fold);
            log_dma.debug("copy: srcmem=" IDFMT " dstmem=" IDFMT " node=%d", mp.first.id, mp.second.id, dma_node);

            if(((unsigned)dma_node) == gasnet_mynode()) {
              log_dma.debug("performing serdez on local node");
	      Realm::get_runtime()->optable.add_local_operation(ev, r);
              r->check_readiness(false, dma_queue);
              finish_events.insert(ev);
            } else {
              RemoteCopyArgs args;
              args.redop_id = 0;
              args.red_fold = false;
              args.before_copy = wait_on;
              args.after_copy = ev;
              args.priority = priority;

              size_t msglen = r->compute_size();
              void *msgdata = malloc(msglen);

              r->serialize(msgdata);

              log_dma.debug("performing serdez on remote node (%d), event=" IDFMT, dma_node, args.after_copy.id);
	      get_runtime()->optable.add_remote_operation(ev, dma_node);
              RemoteCopyMessage::request(dma_node, args, msgdata, msglen, PAYLOAD_FREE);

              finish_events.insert(ev);
              // done with the local copy of the request
	      r->remove_reference();
            }
	  }
	  else {
	  // </SERDEZ_DMA>
	    OASByInst *oas_by_inst;
	    OASByMem::iterator it = oas_by_mem.find(mp);
	    if(it != oas_by_mem.end()) {
	      oas_by_inst = it->second;
	    } else {
	      oas_by_inst = new OASByInst;
	      oas_by_mem[mp] = oas_by_inst;
	    }
	    OASVec& oasvec = (*oas_by_inst)[ip];

	    oasvec.push_back(oas);
          }
	  src_suboffset += oas.size;
	  assert(src_suboffset <= src_it->size);
	  if(src_suboffset == src_it->size) {
	    src_it++;
	    src_suboffset = 0;
	  }
	  dst_suboffset += oas.size;
	  assert(dst_suboffset <= dst_it->size);
	  if(dst_suboffset == dst_it->size) {
	    dst_it++;
	    dst_suboffset = 0;
	  }
	}
	// make sure we used up both
	assert(src_it == srcs.end());
	assert(dst_it == dsts.end());

	log_dma.debug("copy: %zd distinct src/dst mem pairs, is=" IDFMT "", oas_by_mem.size(), is_id);

	for(OASByMem::const_iterator it = oas_by_mem.begin(); it != oas_by_mem.end(); it++) {
	  Memory src_mem = it->first.first;
	  Memory dst_mem = it->first.second;
	  OASByInst *oas_by_inst = it->second;

	  Event ev = GenEventImpl::create_genevent()->current_event();
#ifdef EVENT_GRAPH_TRACE
          Event enclosing = find_enclosing_termination_event();
          log_event_graph.info("Copy Request: (" IDFMT ",%d) (" IDFMT ",%d) "
                                "(" IDFMT ",%d) " IDFMT " " IDFMT "",
                                ev.id, ev.gen, wait_on.id, wait_on.gen,
                                enclosing.id, enclosing.gen,
                                src_mem.id, dst_mem.id);
#endif

	  int priority = 0;
	  if (get_runtime()->get_memory_impl(src_mem)->kind == MemoryImpl::MKIND_GPUFB)
	    priority = 1;
	  else if (get_runtime()->get_memory_impl(dst_mem)->kind == MemoryImpl::MKIND_GPUFB)
	    priority = 1;

	  CopyRequest *r = new CopyRequest(*this, oas_by_inst, 
					   wait_on, ev, priority, requests);

	  // ask which node should perform the copy
	  int dma_node = select_dma_node(src_mem, dst_mem, redop_id, red_fold);
	  log_dma.debug("copy: srcmem=" IDFMT " dstmem=" IDFMT " node=%d", src_mem.id, dst_mem.id, dma_node);
	  
	  if(((unsigned)dma_node) == gasnet_mynode()) {
	    log_dma.debug("performing copy on local node");

	    get_runtime()->optable.add_local_operation(ev, r);
	  
	    r->check_readiness(false, dma_queue);

	    finish_events.insert(ev);
	  } else {
	    RemoteCopyArgs args;
	    args.redop_id = 0;
	    args.red_fold = false;
	    args.before_copy = wait_on;
	    args.after_copy = ev;
	    args.priority = priority;

            size_t msglen = r->compute_size();
            void *msgdata = malloc(msglen);

            r->serialize(msgdata);

	    log_dma.debug("performing copy on remote node (%d), event=" IDFMT, dma_node, args.after_copy.id);
	    get_runtime()->optable.add_remote_operation(ev, dma_node);
	    RemoteCopyMessage::request(dma_node, args, msgdata, msglen, PAYLOAD_FREE);
	  
	    finish_events.insert(ev);

	    // done with the local copy of the request
	    r->remove_reference();
	  }
	}

	// final event is merge of all individual copies' events
	return GenEventImpl::merge_events(finish_events, false /*!ignore faults*/);
      } else {
        log_new_dma.info("Performing reduction op redop_id(%d)", redop_id);
	// we're doing a reduction - the semantics require that all source fields be pulled
	//  together and applied as a "structure" to the reduction op

	// figure out where the source data is
	int src_node = -1;

	for(std::vector<CopySrcDstField>::const_iterator src_it = srcs.begin();
	    src_it != srcs.end();
	    src_it++)
	{
	  int n = ID(src_it->inst).instance.owner_node;
	  if((src_node != -1) && (src_node != n)) {
	    // for now, don't handle case where source data is split across nodes
	    assert(0);
	  }
	  src_node = n;
	}

	assert(dsts.size() == 1);

	// some destinations (e.g. GASNET) need a lock taken to ensure
	//  reductions are applied atomically
	MemoryImpl::MemoryKind dst_kind = get_runtime()->get_memory_impl(get_runtime()->get_instance_impl(dsts[0].inst)->memory)->kind;
	bool inst_lock_needed = (dst_kind == MemoryImpl::MKIND_GLOBAL);

	Event ev = GenEventImpl::create_genevent()->current_event();

	ReduceRequest *r = new ReduceRequest(*this, 
					     srcs, dsts[0],
					     inst_lock_needed,
					     redop_id, red_fold,
					     wait_on, ev,
					     0 /*priority*/, requests);

	if(((unsigned)src_node) == gasnet_mynode()) {
	  log_dma.debug("performing reduction on local node");

	  get_runtime()->optable.add_local_operation(ev, r);
	  
	  r->check_readiness(false, dma_queue);
	} else {
	  RemoteCopyArgs args;
	  args.redop_id = redop_id;
	  args.red_fold = red_fold;
	  args.before_copy = wait_on;
	  args.after_copy = ev;
	  args.priority = 0 /*priority*/;

          size_t msglen = r->compute_size();
          void *msgdata = malloc(msglen);
          r->serialize(msgdata);

	  log_dma.debug("performing reduction on remote node (%d), event=" IDFMT,
		       src_node, args.after_copy.id);
	  get_runtime()->optable.add_remote_operation(ev, src_node);
	  RemoteCopyMessage::request(src_node, args, msgdata, msglen, PAYLOAD_FREE);
	  // done with the local copy of the request
	  r->remove_reference();
	}

	return ev;
      }
    } 

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       const ElementMask& mask,
		       Event wait_on,
		       ReductionOpID redop_id, bool red_fold) const
    {
      assert(redop_id == 0);

      assert(0);
      log_dma.warning("ignoring copy\n");
      return Event::NO_EVENT;
    }

    Event Domain::copy_indirect(const CopySrcDstField& idx,
				const std::vector<CopySrcDstField>& srcs,
				const std::vector<CopySrcDstField>& dsts,
				Event wait_on,
				ReductionOpID redop_id, bool red_fold) const
    {
      assert(redop_id == 0);

      assert(0);
      log_dma.warning("ignoring copy\n");
      return Event::NO_EVENT;
    }

    Event Domain::copy_indirect(const CopySrcDstField& idx,
				const std::vector<CopySrcDstField>& srcs,
				const std::vector<CopySrcDstField>& dsts,
				const ElementMask& mask,
				Event wait_on,
				ReductionOpID redop_id, bool red_fold) const
    {
      assert(redop_id == 0);

      assert(0);
      log_dma.warning("ignoring copy\n");
      return Event::NO_EVENT;
    }

};
