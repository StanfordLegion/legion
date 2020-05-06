/* Copyright 2020 Stanford University, NVIDIA Corporation
 * Copyright 2020 Los Alamos National Laboratory
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
#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"
#include "realm/threads.h"
#include "realm/transfer/transfer.h"
#include "realm/transfer/channel_disk.h"

#include <errno.h>
// included for file memory data transfer
#include <unistd.h>
#ifdef REALM_USE_KERNEL_AIO
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#else
#include <aio.h>
#endif

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
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

#include "realm/timers.h"
#include "realm/serialize.h"

TYPE_IS_SERIALIZABLE(Realm::OffsetsAndSize);

namespace Realm {

    Logger log_dma("dma");
    Logger log_ib_alloc("ib_alloc");
    //extern Logger log_new_dma;
    Logger log_aio("aio");
#ifdef EVENT_GRAPH_TRACE
    extern Logger log_event_graph;
    extern Event find_enclosing_termination_event(void);
#endif

    class IBFence : public Operation::AsyncWorkItem {
    public:
      IBFence(Operation *op) : Operation::AsyncWorkItem(op) {}
      virtual void request_cancellation(void) {
        // ignored for now
      }
      virtual void print(std::ostream& os) const { os << "IBFence"; }
    };

    class IBAllocRequest {
    public:
      IBAllocRequest(NodeID _owner, void *_req, void *_ibinfo,
		     //int _idx,
                     //ID::IDType _src_inst_id, ID::IDType _dst_inst_id,
                     size_t _ib_size)
        : owner(_owner), req(_req),
	  ibinfo(_ibinfo),
	  //idx(_idx), src_inst_id(_src_inst_id),
          //dst_inst_id(_dst_inst_id),
	  ib_size(_ib_size) {ib_offset = -1;}
    public:
      NodeID owner;
      void *req;
      void *ibinfo;
      //int idx;
      //ID::IDType src_inst_id, dst_inst_id;
      size_t ib_size;
      off_t ib_offset;
    };

    class PendingIBQueue {
    public:
      PendingIBQueue();

      void enqueue_request(Memory tgt_mem, IBAllocRequest* req);

      void dequeue_request(Memory tgt_mem);

    protected:
      Mutex queue_mutex;
      std::map<Memory, std::queue<IBAllocRequest*> *> queues;
    };

    class DmaRequest;

    class DmaRequestQueue {
    public:
      DmaRequestQueue(CoreReservationSet& crs);

      void enqueue_request(DmaRequest *r);

      DmaRequest *dequeue_request(bool sleep = true);

      void shutdown_queue(void);

      void start_workers(int count);

      void worker_thread_loop(void);

    protected:
      Mutex queue_mutex;
      CondVar queue_condvar;
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

    DmaRequest::DmaRequest(int _priority,
			   GenEventImpl *_after_copy, EventImpl::gen_t _after_gen)
      : Operation(_after_copy, _after_gen, ProfilingRequestSet()),
	state(STATE_INIT), priority(_priority)
    {
      tgt_fetch_completion = Event::NO_EVENT;
    }

    DmaRequest::DmaRequest(int _priority,
			   GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
			   const ProfilingRequestSet &reqs)
      : Operation(_after_copy, _after_gen, reqs), state(STATE_INIT),
	priority(_priority)
    {
      tgt_fetch_completion = Event::NO_EVENT;
    }

    DmaRequest::~DmaRequest(void)
    {
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

    static PendingIBQueue *ib_req_queue = 0;

    PendingIBQueue::PendingIBQueue() {}

    void PendingIBQueue::enqueue_request(Memory tgt_mem, IBAllocRequest* req)
    {
      AutoLock<> al(queue_mutex);
      assert(NodeID(ID(tgt_mem).memory_owner_node()) == Network::my_node_id);
      // If we can allocate in target memory, no need to pend the request
      off_t ib_offset = get_runtime()->get_memory_impl(tgt_mem)->alloc_bytes_local(req->ib_size);
      if (ib_offset >= 0) {
        if (req->owner == Network::my_node_id) {
          // local ib alloc request
          CopyRequest* cr = (CopyRequest*) req->req;
          //RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(req->src_inst_id);
          //RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(req->dst_inst_id);
          //InstPair inst_pair(src_impl->me, dst_impl->me);
          cr->handle_ib_response(reinterpret_cast<IBInfo *>(req->ibinfo), ib_offset);
	  //req->idx, inst_pair, req->ib_size, ib_offset);
        } else {
          // remote ib alloc request
	  ActiveMessage<RemoteIBAllocResponseAsync> amsg(req->owner);
	  amsg->req = req->req;
	  amsg->ibinfo = req->ibinfo;
	  //amsg->idx = req->idx;
	  //amsg->src_inst_id = req->src_inst_id;
	  //amsg->dst_inst_id = req->dst_inst_id;
	  amsg->offset = ib_offset;
	  amsg->size = req->ib_size;
	  amsg.commit();
        }
        // Remember to free IBAllocRequest
        delete req;

        return;
      }
      log_ib_alloc.info() << "enqueue: req=" << req->req << " ibinfo=" << req->ibinfo << " mem=" << tgt_mem << " size=" << req->ib_size;
      //log_ib_alloc.info("enqueue_request: src_inst(%llx) dst_inst(%llx) "
      //                  "no enough space in memory(%llx)", req->src_inst_id, req->dst_inst_id, tgt_mem.id);
      //log_ib_alloc.info() << " (" << req->src_inst_id << "," 
      //  << req->dst_inst_id << "): no enough space in memory" << tgt_mem;
      std::map<Memory, std::queue<IBAllocRequest*> *>::iterator it = queues.find(tgt_mem);
      if (it == queues.end()) {
        std::queue<IBAllocRequest*> *q = new std::queue<IBAllocRequest*>;
        q->push(req);
        queues[tgt_mem] = q;
        //log_ib_alloc.info("enqueue_request: queue_length(%lu)", q->size());
      } else {
        it->second->push(req);
        //log_ib_alloc.info("enqueue_request: queue_length(%lu)", it->second->size());
      }
    }

    void PendingIBQueue::dequeue_request(Memory tgt_mem)
    {
      AutoLock<> al(queue_mutex);
      assert(NodeID(ID(tgt_mem).memory_owner_node()) == Network::my_node_id);
      std::map<Memory, std::queue<IBAllocRequest*> *>::iterator it = queues.find(tgt_mem);
      // no pending ib requests
      if (it == queues.end()) return;
      while (!it->second->empty()) {
        IBAllocRequest* req = it->second->front();
        off_t ib_offset = get_runtime()->get_memory_impl(tgt_mem)->alloc_bytes_local(req->ib_size);
        if (ib_offset < 0) break;
        //printf("req: src_inst_id(%llx) dst_inst_id(%llx) ib_size(%lu) idx(%d)\n", req->src_inst_id, req->dst_inst_id, req->ib_size, req->idx);
        // deal with the completed ib alloc request
	log_ib_alloc.info() << "complete: req=" << req->req << " ibinfo=" << req->ibinfo << " mem=" << tgt_mem << " size=" << req->ib_size;
        //log_ib_alloc.info() << "IBAllocRequest (" << req->src_inst_id << "," 
        //  << req->dst_inst_id << "): completed!";
        if (req->owner == Network::my_node_id) {
          // local ib alloc request
          CopyRequest* cr = (CopyRequest*) req->req;
          //RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(req->src_inst_id);
          //RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(req->dst_inst_id);
          //InstPair inst_pair(src_impl->me, dst_impl->me);
          cr->handle_ib_response(reinterpret_cast<IBInfo *>(req->ibinfo), ib_offset);
	  //req->idx, inst_pair, req->ib_size, ib_offset);
        } else {
          // remote ib alloc request
	  ActiveMessage<RemoteIBAllocResponseAsync> amsg(req->owner, 8);
	  amsg->req = req->req;
	  amsg->ibinfo = req->ibinfo;
	  //amsg->idx = req->idx;
	  //amsg->src_inst_id = req->src_inst_id;
	  //amsg->dst_inst_id = req->dst_inst_id;
	  amsg->offset = ib_offset;
	  amsg << req->ib_size;
	  amsg.commit();
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

    DmaRequestQueue::DmaRequestQueue(CoreReservationSet& crs)
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
			     GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
			     int _priority)
      : DmaRequest(_priority, _after_copy, _after_gen),
	oas_by_inst(0),
	gather_info(0),
	scatter_info(0),
	ib_responses_needed(0),
	before_copy(_before_copy)
    {
      Serialization::FixedBufferDeserializer deserializer(data, datalen);

      domain = TransferDomain::deserialize_new(deserializer);
      oas_by_inst = new OASByInst;
      bool ok = ((deserializer >> *oas_by_inst) &&
		 (deserializer >> requests));
      assert((domain != 0) && ok && (deserializer.bytes_left() == 0));

      // If either one of the instances is in GPU memory increase priority
      if(priority == 0) {
	for(OASByInst::const_iterator it = oas_by_inst->begin();
	    it != oas_by_inst->end();
	    ++it) {
          Memory::Kind src_kind = it->first.first.get_location().kind();
          if (src_kind == Memory::GPU_FB_MEM) {
            priority = 1;
	    break;
	  }
          Memory::Kind dst_kind = it->first.second.get_location().kind();
          if (dst_kind == Memory::GPU_FB_MEM) {
            priority = 1;
	    break;
	  }
	}
      }

      Operation::reconstruct_measurements();

      log_dma.info() << "dma request " << (void *)this << " deserialized - is="
		     << *domain << " before=" << before_copy << " after=" << get_finish_event();
      for(OASByInst::const_iterator it = oas_by_inst->begin();
	  it != oas_by_inst->end();
	  it++)
	for(OASVec::const_iterator it2 = it->second.begin();
	    it2 != it->second.end();
	    it2++)
	  log_dma.info() << "dma request " << (void *)this << " field: " <<
	    it->first.first << "[" << it2->src_field_id << "+" << it2->src_subfield_offset << "]->" <<
	    it->first.second << "[" << it2->dst_field_id << "+" << it2->dst_subfield_offset << "] size=" << it2->size <<
	    " serdez=" << it2->serdez_id;
    }

    CopyRequest::CopyRequest(const TransferDomain *_domain, //const Domain& _domain,
			     OASByInst *_oas_by_inst,
			     IndirectionInfo *_gather_info,
			     IndirectionInfo *_scatter_info,
			     Event _before_copy,
			     GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
			     int _priority,
                             const ProfilingRequestSet &reqs)
      : DmaRequest(_priority, _after_copy, _after_gen, reqs)
      , domain(_domain->clone())
      , oas_by_inst(_oas_by_inst)
      , gather_info(_gather_info)
      , scatter_info(_scatter_info)
      , ib_responses_needed(0)
      , before_copy(_before_copy)
    {
      log_dma.info() << "dma request " << (void *)this << " created - is="
		     << *domain << " before=" << before_copy << " after=" << get_finish_event();
      if(_gather_info)
	log_dma.info() << "dma request " <<  ((void *)this) << " gather: " << *_gather_info;
      if(_scatter_info)
	log_dma.info() << "dma request " <<  ((void *)this) << " scatter: " << *_scatter_info;
      for(OASByInst::const_iterator it = oas_by_inst->begin();
	  it != oas_by_inst->end();
	  it++)
	for(OASVec::const_iterator it2 = it->second.begin();
	    it2 != it->second.end();
	    it2++)
	  log_dma.info() << "dma request " << (void *)this << " field: " <<
	    it->first.first << "[" << it2->src_field_id << "+" << it2->src_subfield_offset << "]->" <<
	    it->first.second << "[" << it2->dst_field_id << "+" << it2->dst_subfield_offset << "] size=" << it2->size <<
	    " serdez=" << it2->serdez_id;
    }
 
    CopyRequest::~CopyRequest(void)
    {
      // destroy all xfer des
      for(std::vector<XferDesID>::const_iterator it = xd_ids.begin();
	  it != xd_ids.end();
	  ++it)
	destroy_xfer_des(*it);
      xd_ids.clear();

      delete oas_by_inst;
      delete gather_info;
      delete scatter_info;
      delete domain;
    }

    void CopyRequest::forward_request(NodeID target_node)
    {
      ActiveMessage<RemoteCopyMessage> amsg(target_node, 65536);
      amsg->redop_id = 0;
      amsg->red_fold = false;
      amsg->before_copy = before_copy;
      amsg->after_copy = get_finish_event();
      amsg->priority = priority;
      amsg << *domain;
      amsg << *oas_by_inst;
      amsg << requests;

      // TODO: serialize these
      assert(gather_info == 0);
      assert(scatter_info == 0);

      amsg.commit();
      log_dma.debug() << "forwarding copy: target=" << target_node << " finish=" << get_finish_event();
      clear_profiling();
    }

    void DmaRequest::Waiter::sleep_on_event(Event e, 
					    Reservation l /*= Reservation::NO_RESERVATION*/)
    {
      current_lock = l;
      wait_on = e;
      EventImpl::add_waiter(e, this);
    }

    void DmaRequest::Waiter::event_triggered(bool poisoned)
    {
      if(poisoned) {
	log_poison.info() << "cancelling poisoned dma operation - op=" << req << " after=" << req->get_finish_event();
	req->handle_poisoned_precondition(wait_on);
	return;
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

  /*static*/ void RemoteIBAllocRequestAsync::handle_message(NodeID sender,
							    const RemoteIBAllocRequestAsync &args,
							    const void *data, size_t msglen)
    {
      assert(NodeID(ID(args.memory).memory_owner_node()) == Network::my_node_id);
      IBAllocRequest* ib_req
 	  = new IBAllocRequest(sender, args.req, args.ibinfo,
			       //args.idx,
                               //args.src_inst_id, args.dst_inst_id,
			       args.size);
      ib_req_queue->enqueue_request(args.memory, ib_req);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class RemoteIBAllocResponseAsync
    //

    /*static*/ void RemoteIBAllocResponseAsync::handle_message(NodeID sender,
							       const RemoteIBAllocResponseAsync &args,
							       const void *data, size_t msglen)
    {
      CopyRequest* req = (CopyRequest*) args.req;
      //RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(args.src_inst_id);
      //RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(args.dst_inst_id);
      //InstPair inst_pair(src_impl->me, dst_impl->me);
      //size_t size = *((size_t*)data);
      req->handle_ib_response(reinterpret_cast<IBInfo *>(args.ibinfo),
			      //args.idx, inst_pair, size,
			      args.offset);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class RemoteIBFreeRequestAsync
    //

    /*static*/ void RemoteIBFreeRequestAsync::handle_message(NodeID sender,
							     const RemoteIBFreeRequestAsync &args,
							     const void *data, size_t msglen)
    {
      assert(NodeID(ID(args.memory).memory_owner_node()) == Network::my_node_id);
      get_runtime()->get_memory_impl(args.memory)->free_bytes_local(args.ib_offset, args.ib_size);
      ib_req_queue->dequeue_request(args.memory);
    }

#define IB_MAX_SIZE size_t(64 * 1024 * 1024)

    void free_intermediate_buffer(DmaRequest* req, Memory mem, off_t offset, size_t size)
    {
      //CopyRequest* cr = (CopyRequest*) req;
      //AutoLock<> al(cr->ib_mutex);
      if(NodeID(ID(mem).memory_owner_node()) == Network::my_node_id) {
        get_runtime()->get_memory_impl(mem)->free_bytes_local(offset, size);
        ib_req_queue->dequeue_request(mem);
      } else {
	ActiveMessage<RemoteIBFreeRequestAsync> amsg(ID(mem).memory_owner_node());
	amsg->memory = mem;
	amsg->ib_offset = offset;
	amsg->ib_size = size;
	amsg.commit();
      }
    }

#if 0
    void CopyRequest::alloc_intermediate_buffer(InstPair inst_pair, Memory tgt_mem, int idx)
    {
      assert(oas_by_inst->find(inst_pair) != oas_by_inst->end());
      OASVec& oasvec = (*oas_by_inst)[inst_pair];
      size_t ib_elmnt_size = 0, domain_size = 0;
      size_t serdez_pad = 0;
      size_t min_granularity = 1;
      for(OASVec::const_iterator it = oasvec.begin(); it != oasvec.end(); it++) {
	if(it->serdez_id != 0) {
	  const CustomSerdezUntyped *serdez_op = get_runtime()->custom_serdez_table.get(it->serdez_id, 0);
	  assert(serdez_op != 0);
	  // sanity-check max_serialized_size
	  if(serdez_op->max_serialized_size > IB_MAX_SIZE) {
	    log_dma.fatal() << "FATAL: serdez " << it->serdez_id << " reports a maximum serialized size of " << serdez_op->max_serialized_size << " bytes - does not fit into max intermediate buffer size of " << IB_MAX_SIZE << " bytes";
	    abort();
	  }
	  ib_elmnt_size += serdez_op->max_serialized_size;
	  if(serdez_op->max_serialized_size > serdez_pad)
	    serdez_pad = serdez_op->max_serialized_size;
	} else {
	  ib_elmnt_size += it->size;
	  min_granularity = lcm(min_granularity, size_t(it->size));
	}
      }
      domain_size = domain->volume();

      size_t ib_size = domain_size * ib_elmnt_size + serdez_pad;
      if(ib_size > IB_MAX_SIZE) {
	// take up to IB_MAX_SIZE, respecting the min granularity
	if(min_granularity > 1) {
	  // (really) corner case: if min_granulary exceeds IB_MAX_SIZE, use it
	  //  directly and hope it's ok
	  if(min_granularity > IB_MAX_SIZE) {
	    ib_size = min_granularity;
	  } else {
	    size_t extra = IB_MAX_SIZE % min_granularity;
	    ib_size = IB_MAX_SIZE - extra;
	  }
	} else
	  ib_size = IB_MAX_SIZE;
      }
      //log_ib_alloc.info("alloc_ib: src_inst_id(%llx) dst_inst_id(%llx) idx(%d) size(%lu) memory(%llx)", inst_pair.first.id, inst_pair.second.id, idx, ib_size, tgt_mem.id);
      if (NodeID(ID(tgt_mem).memory_owner_node()) == Network::my_node_id) {
        // create local intermediate buffer
        IBAllocRequest* ib_req
          = new IBAllocRequest(Network::my_node_id, this, idx, inst_pair.first.id,
                               inst_pair.second.id, ib_size);
        ib_req_queue->enqueue_request(tgt_mem, ib_req);
      } else {
        // create remote intermediate buffer
	ActiveMessage<RemoteIBAllocRequestAsync> amsg(ID(tgt_mem).memory_owner_node(), 8);
	amsg->memory = tgt_mem;
	amsg->req = this;
	amsg->idx = idx;
	amsg->src_inst_id = inst_pair.first.id;
	amsg->dst_inst_id = inst_pair.second.id;
	amsg << ib_size;
	amsg.commit();
      }
    }
#endif

    void CopyRequest::handle_ib_response(IBInfo *ibinfo,
					 //int idx, InstPair inst_pair, size_t ib_size,
					 off_t ib_offset)
    {
      ibinfo->offset = ib_offset;
      size_t remaining = ib_responses_needed.fetch_sub(1) - 1;
      log_ib_alloc.debug() << "received: req=" << ((void *)this) << " info=" << ((void *)ibinfo) << " offset=" << ib_offset << " remain=" << remaining;
      if(remaining == 0) {
	// all IB responses have been retrieved, so advance the copy request
	log_ib_alloc.info() << "all responses received: req=" << ((void *)this);
	check_readiness(false, dma_queue);
      }
#if 0
      Event event_to_trigger = Event::NO_EVENT;
      {
        AutoLock<> al(ib_mutex);
        IBByInst::iterator ib_it = ib_by_inst.find(inst_pair);
        assert(ib_it != ib_by_inst.end());
        IBVec& ibvec = ib_it->second;
        assert((int)ibvec.size() > idx);
        ibvec[idx].size = ib_size;
        ibvec[idx].offset = ib_offset;
        event_to_trigger = ibvec[idx].event;
      }
      GenEventImpl::trigger(event_to_trigger, false/*!poisoned*/);
      //ibvec[idx].fence->mark_finished(true);
#endif
    }

  static size_t compute_ib_size(const OASVec& oasvec,
				const TransferDomain *domain)
  {
    size_t ib_elmnt_size = 0, domain_size = 0;
    size_t serdez_pad = 0;
    size_t min_granularity = 1;
    for(OASVec::const_iterator it2 = oasvec.begin(); it2 != oasvec.end(); it2++) {
      if(it2->serdez_id != 0) {
	const CustomSerdezUntyped *serdez_op = get_runtime()->custom_serdez_table.get(it2->serdez_id, 0);
	assert(serdez_op != 0);
	ib_elmnt_size += serdez_op->max_serialized_size;
	if(serdez_op->max_serialized_size > serdez_pad)
	  serdez_pad = serdez_op->max_serialized_size;
      } else {
	ib_elmnt_size += it2->size;
	min_granularity = lcm(min_granularity, size_t(it2->size));
      }
    }
    domain_size = domain->volume();

    size_t ib_size = domain_size * ib_elmnt_size + serdez_pad;
    if(ib_size > IB_MAX_SIZE) {
      // take up to IB_MAX_SIZE, respecting the min granularity
      if(min_granularity > 1) {
	// (really) corner case: if min_granulary exceeds IB_MAX_SIZE, use it
	//  directly and hope it's ok
	if(min_granularity > IB_MAX_SIZE) {
	  ib_size = min_granularity;
	} else {
	  size_t extra = IB_MAX_SIZE % min_granularity;
	  ib_size = IB_MAX_SIZE - extra;
	}
      } else
	ib_size = IB_MAX_SIZE;
    }

    return ib_size;
  }

  // helper - should come from channels eventually
  XferDesFactory *get_xd_factory_by_kind(XferDesKind kind)
  {
    switch(kind) {
    case XFER_MEM_CPY:
      return SimpleXferDesFactory<MemcpyXferDes>::get_singleton();
    case XFER_GASNET_READ:
    case XFER_GASNET_WRITE:
      return SimpleXferDesFactory<GASNetXferDes>::get_singleton();
    case XFER_REMOTE_WRITE:
      return SimpleXferDesFactory<RemoteWriteXferDes>::get_singleton();
    case XFER_DISK_READ:
    case XFER_DISK_WRITE:
      return SimpleXferDesFactory<DiskXferDes>::get_singleton();
    case XFER_FILE_READ:
    case XFER_FILE_WRITE:
      return SimpleXferDesFactory<FileXferDes>::get_singleton();
#ifdef REALM_USE_CUDA
    case XFER_GPU_FROM_FB:
    case XFER_GPU_TO_FB:
    case XFER_GPU_IN_FB:
    case XFER_GPU_PEER_FB:
      return SimpleXferDesFactory<GPUXferDes>::get_singleton();
#endif
#ifdef REALM_USE_HDF5
    case XFER_HDF5_READ:
    case XFER_HDF5_WRITE:
      return SimpleXferDesFactory<HDF5XferDes>::get_singleton();
#endif
    default:
      assert(0);
    }
  }

  void IBInfo::set(Memory _memory, size_t _size)
  {
    memory = _memory;
    offset = size_t(-1);
    size = _size;
  }

  void CopyRequest::XDTemplate::set_simple(NodeID _target_node,
					   XferDesKind _kind,
					   int _in_edge, int _out_edge)
  {
    kind = _kind;
    factory = get_xd_factory_by_kind(_kind);
    gather_control_input = -1;
    scatter_control_input = -1;
    target_node = _target_node;
    inputs.resize(1);
    inputs[0].edge_id = _in_edge;
    inputs[0].indirect_inst = RegionInstance::NO_INST;
    outputs.resize(1);
    outputs[0].edge_id = _out_edge;
    outputs[0].indirect_inst = RegionInstance::NO_INST;
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
	Event e = domain->request_metadata();
	if(!e.has_triggered()) {
	  log_dma.debug() << "transfer domain metadata - req=" << (void *)this
			  << " ready=" << e;
	  waiter.sleep_on_event(e);
	  return false;
	}

	// now go through all instance pairs
	for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	  if(it->first.first.exists()) {
	    RegionInstanceImpl *src_impl = get_runtime()->get_instance_impl(it->first.first);
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

	  if(it->first.second.exists()) {
	    RegionInstanceImpl *dst_impl = get_runtime()->get_instance_impl(it->first.second);
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

	// need metadata for gather or scatter indirections
	if(gather_info) {
	  Event e = gather_info->request_metadata();
	  if(!e.has_triggered()) {
	    if(just_check) {
	      log_dma.debug() << "dma request " << ((void *)this) << " - no gather info metadata yet";
	      return false;
	    }
	    log_dma.debug() << "request " << (void *)this << " - gather info metadata invalid - sleeping on event " << e;
	    waiter.sleep_on_event(e);
	    return false;
	  }
	}

	if(scatter_info) {
	  Event e = scatter_info->request_metadata();
	  if(!e.has_triggered()) {
	    if(just_check) {
	      log_dma.debug() << "dma request " << ((void *)this) << " - no scatter info metadata yet";
	      return false;
	    }
	    log_dma.debug() << "request " << (void *)this << " - scatter info metadata invalid - sleeping on event " << e;
	    waiter.sleep_on_event(e);
	    return false;
	  }
	}

	// if we got all the way through, we've got all the metadata we need
	state = STATE_DST_FETCH;
      }

      // make sure the destination has fetched metadata
      if(state == STATE_DST_FETCH) {
	// TODO
	// SJT: actually, is this needed any more?
        //Memory tgt_mem = get_runtime()->get_instance_impl(oas_by_inst->begin()->first.second)->memory;
        //NodeID tgt_node = ID(tgt_mem).memory_owner_node();
	//assert(tgt_node == Network::my_node_id);
        state = STATE_GEN_PATH;
      }

      if(state == STATE_GEN_PATH) {
        log_dma.debug("generate paths");
	// SJT: this code is pretty broken if there are multiple instance pairs
	assert(oas_by_inst->size() == 1);
        // Pass 1: create IBInfo blocks
        for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	  Memory src_mem = (gather_info ? Memory::NO_MEMORY :
			                  it->first.first.get_location());
	  Memory dst_mem = (scatter_info ? Memory::NO_MEMORY :
			                   it->first.second.get_location());
	  CustomSerdezID serdez_id = it->second[0].serdez_id;

	  // in gather or scatter case, we need to know the data bytes per
	  //  address for the splitter
	  size_t bytes_per_element = 0;
	  for(OASVec::const_iterator it2 = it->second.begin();
	      it2 != it->second.end();
	      ++it2) {
	    assert(it2->serdez_id == serdez_id); // all must agree
	    if(it2->serdez_id != 0)
	      // units will be in serialized chunks instead of bytes
	      bytes_per_element += 1;
	    else
	      bytes_per_element += it2->size;
	  }

	  if(scatter_info != 0) {
	    if(gather_info != 0) {
	      // combined gather/scatter
	      // TODO: optimize case of single source and single dest

	      // need an intermediate buffer for the output of the gather
	      //  (and input of the scatter)

	      NodeID ib_node = Network::my_node_id;
	      Memory ib_mem = ID::make_ib_memory(ib_node, 0).convert<Memory>();

	      int ib_idx = ib_edges.size();
	      ib_edges.resize(ib_idx + 1);
	      ib_edges[ib_idx].set(ib_mem, 1 << 20);  //HACK

	      gather_info->generate_gather_paths(ib_mem, ib_idx,
						 bytes_per_element,
						 serdez_id,
						 xd_nodes, ib_edges);

	      scatter_info->generate_scatter_paths(ib_mem, ib_idx,
						   bytes_per_element,
						   serdez_id,
						   xd_nodes, ib_edges);

	      continue;
	    } else {
	      assert(!it->first.second.exists());

	      int src_edge_id = XDTemplate::SRC_INST;

	      dst_mem = scatter_info->generate_scatter_paths(src_mem,
							     src_edge_id,
							     bytes_per_element,
							     serdez_id,
							     xd_nodes, ib_edges);

	      //Memory dst_mem = get_runtime()->get_instance_impl(it->first.second)->memory;
	      //ib_responses_needed += gather_info->generate_gather_paths(dst_mem, serdez_id, pending_ib_requests);
	      assert(!dst_mem.exists());
	      continue;
	    }
	  }
	  if(gather_info != 0) {
	    assert(!it->first.first.exists());

	    int dst_edge_id = XDTemplate::DST_INST;

	    src_mem = gather_info->generate_gather_paths(dst_mem,
							 dst_edge_id,
							 bytes_per_element,
							 serdez_id,
							 xd_nodes, ib_edges);
							 
	    //Memory dst_mem = get_runtime()->get_instance_impl(it->first.second)->memory;
	    //ib_responses_needed += gather_info->generate_gather_paths(dst_mem, serdez_id, pending_ib_requests);
	    assert(!src_mem.exists());
	    continue;
	  }
	  MemPathInfo path_info;
	  bool ok = find_shortest_path(src_mem, dst_mem, serdez_id,
				       path_info);
	  if(!ok) {
	    log_new_dma.fatal() << "FATAL: no path found from " << src_mem << " to " << dst_mem << " (serdez=" << serdez_id << ")";
	    assert(0);
	  }
	  size_t pathlen = path_info.xd_kinds.size();
	  size_t xd_idx = xd_nodes.size();
	  size_t ib_idx = ib_edges.size();
	  size_t ib_alloc_size = 0;
	  xd_nodes.resize(xd_idx + pathlen);
	  if(pathlen > 1) {
	    ib_edges.resize(ib_idx + pathlen - 1);
	    ib_alloc_size = compute_ib_size(it->second, domain);
	  }
	  for(size_t i = 0; i < pathlen; i++) {
	    xd_nodes[xd_idx].kind = path_info.xd_kinds[i];
	    xd_nodes[xd_idx].factory = get_xd_factory_by_kind(path_info.xd_kinds[i]);
	    xd_nodes[xd_idx].gather_control_input = -1;
	    xd_nodes[xd_idx].scatter_control_input = -1;
	    xd_nodes[xd_idx].target_node = path_info.xd_target_nodes[i];
	    xd_nodes[xd_idx].inputs.resize(1);
	    xd_nodes[xd_idx].inputs[0].edge_id = ((i == 0) ?
						    XDTemplate::SRC_INST :
						    (ib_idx - 1)); // created by previous step
	    xd_nodes[xd_idx].inputs[0].indirect_inst = RegionInstance::NO_INST;
	    xd_nodes[xd_idx].outputs.resize(1);
	    xd_nodes[xd_idx].outputs[0].edge_id = ((i == (pathlen - 1)) ?
						     XDTemplate::DST_INST :
						     ib_idx);
	    xd_nodes[xd_idx].outputs[0].indirect_inst = RegionInstance::NO_INST;
	    if(i < (pathlen - 1)) {
	      ib_edges[ib_idx].memory = path_info.path[i + 1];
	      ib_edges[ib_idx].offset = size_t(-1);
	      ib_edges[ib_idx].size = ib_alloc_size;
	      ib_idx++;
	    }
	    xd_idx++;
	  }
        }
        // Pass 2: send ib allocation requests
        //for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
        //  for (size_t i = 1; i < mem_path.size() - 1; i++) {
        //    alloc_intermediate_buffer(it->first, mem_path[i], i - 1);
        //  }
        //}
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
	    state = STATE_ALLOC_IB;
	  }
	} else {
	  log_dma.debug("request %p - before event not triggered", this);
	  if(just_check) return false;

	  log_dma.debug("request %p - sleeping on before event", this);
	  waiter.sleep_on_event(before_copy);
	  return false;
	}
      }

      if(state == STATE_ALLOC_IB) {
	if(ib_edges.empty()) {
	  // skip this step, fall through to READY
	  state = STATE_READY;
	} else {
	  // increase the count by one to prevent a trigger before we finish
	  //  this loop
	  ib_responses_needed.store(ib_edges.size() + 1);

	  // sort requests by target memory to reduce risk of resource deadlock
	  PendingIBRequests pending;
	  for(std::vector<IBInfo>::iterator it = ib_edges.begin();
	      it != ib_edges.end();
	      ++it)
	    pending[it->memory].push_back(&*it);

	  for(PendingIBRequests::const_iterator it = pending.begin();
	      it != pending.end();
	      ++it) {
	    for(std::vector<IBInfo *>::const_iterator it2 = it->second.begin();
		it2 != it->second.end();
		++it2) {
	      Memory tgt_mem = it->first;
	      if(NodeID(ID(tgt_mem).memory_owner_node()) == Network::my_node_id) {
		// create local intermediate buffer
		IBAllocRequest* ib_req = new IBAllocRequest(Network::my_node_id,
							    this, *it2, (*it2)->size);
		ib_req_queue->enqueue_request(tgt_mem, ib_req);
	      } else {
		// create remote intermediate buffer
		ActiveMessage<RemoteIBAllocRequestAsync> amsg(ID(tgt_mem).memory_owner_node());
		amsg->memory = tgt_mem;
		amsg->req = this;
		amsg->ibinfo = *it2;
		amsg->size = (*it2)->size;
		amsg.commit();
	      }
	    }
	    
	  }

	  // change our state BEFORE the decrement, since they complete
	  //  asychronously
	  state = STATE_WAIT_IB;

	  // fall through if this ends up being the last decrement
	  if(ib_responses_needed.fetch_sub(1) > 1)
	    return false;
	}
      }

      if(state == STATE_WAIT_IB) {
	// we should never get here unless the count is already 0
	assert(ib_responses_needed.load() == 0);
	state = STATE_READY;
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


    static atomic<unsigned> rdma_sequence_no(1);

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

    class AIOFence : public Operation::AsyncWorkItem {
    public:
      AIOFence(Operation *_op) : Operation::AsyncWorkItem(_op) {}
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
      : BackgroundWorkItem("async file IO")
      , max_depth(_max_depth)
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
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = launched_operations.empty();
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
      if(was_empty)
	make_active();
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
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = launched_operations.empty();
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
      if(was_empty)
	make_active();
    }

    void AsyncFileIOContext::enqueue_fence(DmaRequest *req)
    {
      AIOFenceOp *op = new AIOFenceOp(req);
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = launched_operations.empty();
	if(launched_operations.size() < (size_t)max_depth) {
	  op->launch();
	  launched_operations.push_back(op);
	} else {
	  pending_operations.push_back(op);
	}
      }
      if(was_empty)
	make_active();
    }

    bool AsyncFileIOContext::empty(void)
    {
      AutoLock<> al(mutex);
      return launched_operations.empty();
    }

    long AsyncFileIOContext::available(void)
    {
      AutoLock<> al(mutex);
      return (max_depth - launched_operations.size());
    }

    void AsyncFileIOContext::make_progress(void)
    {
      AutoLock<> al(mutex);

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

    void AsyncFileIOContext::do_work(TimeLimit work_until)
    {
      // first, reap as many events as we can - oldest first
#ifdef REALM_USE_KERNEL_AIO
      assert(!launched_operations.empty());
      while {
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
	// only try again if we got a full set the time before
	if(ret < 8) break;
      } while(!work_until.is_expired());
#endif

      // have to check for completed ops even if we didn't get any this
      //  time - there may have been some last time that we didn't have
      //  enough time to notify
      {
	AutoLock<> al(mutex);

	// now actually mark events completed in oldest-first order
	while(!work_until.is_expired()) {
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
	  if(launched_operations.empty()) {
	    if(pending_operations.empty()) {
	      // finished work - we can return without requeuing ourselves
	      return;
	    } else {
	      // launch some pending work below
	      break;
	    }
	  }
	}

	// finally, if there are any pending ops, and room for them, and
	//   time left, launch them
	while((launched_operations.size() < (size_t)max_depth) &&
	      !pending_operations.empty() &&
	      !work_until.is_expired()) {
	  AIOOperation *op = pending_operations.front();
	  pending_operations.pop_front();
	  op->launch();
	  launched_operations.push_back(op);
	}
      }

      // if we fall through to here, there's still polling for either old
      //  or newly launched work to do
      make_active();
    }

    /*static*/
    AsyncFileIOContext* AsyncFileIOContext::get_singleton() {
      return aio_context;
    }

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

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold)
      {
	return new MemcpyMemPairCopier(src_mem, dst_mem);
      }
#endif
    };


    inline bool is_cpu_mem(Memory::Kind kind)
    {
      return (kind == Memory::REGDMA_MEM || kind == Memory::LEVEL3_CACHE || kind == Memory::LEVEL2_CACHE
              || kind == Memory::LEVEL1_CACHE || kind == Memory::SYSTEM_MEM || kind == Memory::SOCKET_MEM
              || kind == Memory::Z_COPY_MEM);
    }

    XferDesKind old_get_xfer_des(Memory src_mem, Memory dst_mem,
				   CustomSerdezID src_serdez_id,
				   CustomSerdezID dst_serdez_id)
    {
      Memory::Kind src_ll_kind = get_runtime()->get_memory_impl(src_mem)->lowlevel_kind;
      Memory::Kind dst_ll_kind = get_runtime()->get_memory_impl(dst_mem)->lowlevel_kind;
      if(ID(src_mem).memory_owner_node() == ID(dst_mem).memory_owner_node()) {
        switch(src_ll_kind) {
        case Memory::GLOBAL_MEM:
          if (is_cpu_mem(dst_ll_kind)) {
	    // no serdez support
	    if((src_serdez_id != 0) || (dst_serdez_id != 0))
	      return XFER_NONE;
            return XFER_GASNET_READ;
	  }
          else
            return XFER_NONE;
        case Memory::REGDMA_MEM:
        case Memory::LEVEL3_CACHE:
        case Memory::LEVEL2_CACHE:
        case Memory::LEVEL1_CACHE:
        case Memory::SYSTEM_MEM:
        case Memory::SOCKET_MEM:
        case Memory::Z_COPY_MEM:
          if (is_cpu_mem(dst_ll_kind)) {
	    // can't serdez to yourself yet
	    if((src_serdez_id != 0) && (dst_serdez_id != 0))
	      return XFER_NONE;
            return XFER_MEM_CPY;
	  }
          else if (dst_ll_kind == Memory::GLOBAL_MEM) {
	    // no serdez support
	    if((src_serdez_id != 0) || (dst_serdez_id != 0))
	      return XFER_NONE;
            return XFER_GASNET_WRITE;
	  }
#ifdef REALM_USE_CUDA
          else if (dst_ll_kind == Memory::GPU_FB_MEM) {
	    // no serdez support
	    if((src_serdez_id != 0) || (dst_serdez_id != 0))
	      return XFER_NONE;
	    // find which GPU owns the destination memory and see if it
	    //  has the source memory pinned
	    Cuda::GPUFBMemory *fbm = static_cast<Cuda::GPUFBMemory *>(get_runtime()->get_memory_impl(dst_mem));
	    assert(fbm != 0);
	    if(fbm->gpu->pinned_sysmems.count(src_mem) > 0)
	      return XFER_GPU_TO_FB;
	    else
              return XFER_NONE;
          }
#endif
          else if (dst_ll_kind == Memory::DISK_MEM) {
	    // no serdez support
	    if((src_serdez_id != 0) || (dst_serdez_id != 0))
	      return XFER_NONE;
            return XFER_DISK_WRITE;
	  }
          else if (dst_ll_kind == Memory::HDF_MEM) {
	    // no serdez support
	    if((src_serdez_id != 0) || (dst_serdez_id != 0))
	      return XFER_NONE;
            return XFER_HDF5_WRITE;
	  }
          else if (dst_ll_kind == Memory::FILE_MEM) {
	    // no serdez support
	    if((src_serdez_id != 0) || (dst_serdez_id != 0))
	      return XFER_NONE;
            return XFER_FILE_WRITE;
	  }
          assert(0);
          break;
#ifdef REALM_USE_CUDA
        case Memory::GPU_FB_MEM:
        {
	  // no serdez support
	  if((src_serdez_id != 0) || (dst_serdez_id != 0))
	    return XFER_NONE;
	  // find which GPU owns the source memory
	  Cuda::GPUFBMemory *fbm = static_cast<Cuda::GPUFBMemory *>(get_runtime()->get_memory_impl(src_mem));
	  assert(fbm != 0);
	  const Cuda::GPU *gpu = fbm->gpu;

          if (dst_ll_kind == Memory::GPU_FB_MEM) {
            if (src_mem == dst_mem)
              return XFER_GPU_IN_FB;
            else if (gpu->peer_fbs.count(dst_mem) > 0)
              return XFER_GPU_PEER_FB;
	    else
	      return XFER_NONE;
          }
	  else if (is_cpu_mem(dst_ll_kind)) {
	    if (gpu->pinned_sysmems.count(dst_mem) > 0)
              return XFER_GPU_FROM_FB;
	    else
	      return XFER_NONE;
	  } else
            return XFER_NONE;
        }
#endif
        case Memory::DISK_MEM:
	  // no serdez support
	  if((src_serdez_id != 0) || (dst_serdez_id != 0))
	    return XFER_NONE;
          if (is_cpu_mem(dst_ll_kind))
            return XFER_DISK_READ;
          else
            return XFER_NONE;
        case Memory::FILE_MEM:
	  // no serdez support
	  if((src_serdez_id != 0) || (dst_serdez_id != 0))
	    return XFER_NONE;
          if (is_cpu_mem(dst_ll_kind))
            return XFER_FILE_READ;
          else
            return XFER_NONE;
        case Memory::HDF_MEM:
	  // no serdez support
	  if((src_serdez_id != 0) || (dst_serdez_id != 0))
	    return XFER_NONE;
          if (is_cpu_mem(dst_ll_kind))
            return XFER_HDF5_READ;
          else
            return XFER_NONE;
        default:
          assert(0);
        }
      } else {
        if (is_cpu_mem(src_ll_kind) && dst_ll_kind == Memory::REGDMA_MEM) {
	  // destination serdez ok, source not
	  if(src_serdez_id != 0)
	    return XFER_NONE;
          return XFER_REMOTE_WRITE;
	}
        else
          return XFER_NONE;
      }
      return XFER_NONE;
    }

    XferDesKind get_xfer_des(Memory src_mem, Memory dst_mem,
			     CustomSerdezID src_serdez_id,
			     CustomSerdezID dst_serdez_id,
			     ReductionOpID redop_id)
    {
      XferDesKind kind = XFER_NONE;

      // look at the dma channels available on the source node
      NodeID src_node = ID(src_mem).memory_owner_node();
      NodeID dst_node = ID(dst_mem).memory_owner_node();
      const Node& n = get_runtime()->nodes[src_node];
      for(std::vector<DMAChannel *>::const_iterator it = n.dma_channels.begin();
	  it != n.dma_channels.end();
	  ++it) {
	unsigned bw = 0;
	unsigned latency = 0;
	if((*it)->supports_path(src_mem, dst_mem,
				src_serdez_id, dst_serdez_id,
				redop_id,
				&kind, &bw, &latency)) {
	  break;
	}
      }

      // if that didn't work, try the destination node (if different)
      if((kind == XFER_NONE) && (dst_node != src_node)) {
	const Node& n = get_runtime()->nodes[dst_node];
	for(std::vector<DMAChannel *>::const_iterator it = n.dma_channels.begin();
	    it != n.dma_channels.end();
	    ++it) {
	  unsigned bw = 0;
	  unsigned latency = 0;
	  if((*it)->supports_path(src_mem, dst_mem,
				  src_serdez_id, dst_serdez_id,
				  redop_id,
				  &kind, &bw, &latency)) {
	    break;
	  }
	}
      }

      // check against old version
      // exceptions:
      //  1) old code didn't allow nodes other than 0 to
      //       directly access GLOBAL_MEM
      if((src_node == Network::my_node_id) &&
	 !(((Network::my_node_id != 0) || (dst_node != 0)) &&
	   ((src_mem.kind() == Memory::GLOBAL_MEM) ||
	    (dst_mem.kind() == Memory::GLOBAL_MEM)))) {
	XferDesKind old_kind = old_get_xfer_des(src_mem, dst_mem,
						src_serdez_id, dst_serdez_id);
	if(old_kind != kind) {
	  log_dma.fatal() << "kind mismatch: " << kind << " != " << old_kind << ": src=" << src_mem << " dst=" << dst_mem << " serdez=" << src_serdez_id << "," << dst_serdez_id << " redop=" << redop_id;
	  assert(0);
	}
      }
      return kind;
    }

    bool find_shortest_path(Memory src_mem, Memory dst_mem,
			    CustomSerdezID serdez_id,
			    MemPathInfo& info)
    {
      // fast case - can we go straight from src to dst?
      XferDesKind kind = get_xfer_des(src_mem, dst_mem,
				      serdez_id, serdez_id, 0);
      if(kind != XFER_NONE) {
	info.path.resize(2);
	info.path[0] = src_mem;
	info.path[1] = dst_mem;
	info.xd_kinds.push_back(kind);
      } else {
	std::map<Memory, std::vector<Memory> > dist;
	std::map<Memory, std::vector<XferDesKind> > kinds;
	std::list<Memory> mems_left;
	std::queue<Memory> active_nodes;
	Node* node = &(get_runtime()->nodes[ID(src_mem).memory_owner_node()]);
	for (std::vector<MemoryImpl*>::const_iterator it = node->ib_memories.begin();
           it != node->ib_memories.end(); it++) {
	  mems_left.push_back((*it)->me);
	}
	if(ID(dst_mem).memory_owner_node() != ID(src_mem).memory_owner_node()) {
	  node = &(get_runtime()->nodes[ID(dst_mem).memory_owner_node()]);
	  for (std::vector<MemoryImpl*>::const_iterator it = node->ib_memories.begin();
	       it != node->ib_memories.end(); it++) {
	    mems_left.push_back((*it)->me);
	  }
	}
	for(std::list<Memory>::iterator it = mems_left.begin(); it != mems_left.end(); ) {
	  // we know we're doing at least one hop, so no dst_serdez here
	  kind = get_xfer_des(src_mem, *it, serdez_id, 0, 0);
	  if(kind != XFER_NONE) {
	    std::vector<Memory>& v = dist[*it];
	    v.push_back(src_mem);
	    v.push_back(*it);
	    kinds[*it].push_back(kind);
	    active_nodes.push(*it);
	    it = mems_left.erase(it);
	  } else
	    ++it;
	}
	while(true) {
	  if(active_nodes.empty())
	    return false;

	  Memory cur = active_nodes.front();
	  active_nodes.pop();
	  std::vector<Memory> sub_path = dist[cur];
	  
	  // can we reach the destination from here (handling potential
	  //  deserialization?
	  kind = get_xfer_des(cur, dst_mem, 0, serdez_id, 0);
	  if(kind != XFER_NONE) {
	    info.path = dist[cur];
	    info.path.push_back(dst_mem);
	    info.xd_kinds = kinds[cur];
	    info.xd_kinds.push_back(kind);
	    break;
	  }

	  // no, look for another intermediate hop
	  for(std::list<Memory>::iterator it = mems_left.begin(); it != mems_left.end(); ) {
	    kind = get_xfer_des(cur, *it, 0, 0, 0);
	    if(kind != XFER_NONE) {
	      std::vector<Memory>& v = dist[*it];
	      v = dist[cur];
	      v.push_back(*it);
	      std::vector<XferDesKind>& k = kinds[*it];
	      k = kinds[cur];
	      k.push_back(kind);
	      active_nodes.push(*it);
	      it = mems_left.erase(it);
	    } else
	      ++it;
	  }
	}
      }

      // target node is src mem's node for all but gasnet reads
      info.xd_target_nodes.resize(info.xd_kinds.size());
      for(size_t i = 0; i < info.xd_kinds.size(); i++) {
	bool use_dst_mem = (info.xd_kinds[i] == XFER_GASNET_READ);
	Memory mem = (use_dst_mem ? info.path[i + 1] : info.path[i]);
	info.xd_target_nodes[i] = ID(mem).memory_owner_node();
      }
      return true;
    }


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

    static Serialization::PolymorphicSerdezSubclass<TransferIterator, WrappingFIFOIterator> serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

  protected:
    size_t base, size, offset, prev_offset;
    bool tentative_valid;
  };

  WrappingFIFOIterator::WrappingFIFOIterator(size_t _base, size_t _size)
    : base(_base)
    , size(_size)
    , offset(0)
    , tentative_valid(false)
  {}

  template <typename S>
  /*static*/ TransferIterator *WrappingFIFOIterator::deserialize_new(S& deserializer)
  {
    size_t base, size, offset, prev_offset;
    bool tentative_valid;
    if((deserializer >> base) &&
       (deserializer >> size) &&
       (deserializer >> offset) &&
       (deserializer >> prev_offset) &&
       (deserializer >> tentative_valid)) {
      WrappingFIFOIterator *wfi = new WrappingFIFOIterator(base, size);
      wfi->offset = offset;
      wfi->prev_offset = prev_offset;
      wfi->tentative_valid = tentative_valid;
      return wfi;
    } else
      return 0;
  }   

  void WrappingFIFOIterator::reset(void)
  {
    offset = 0;
  }

  bool WrappingFIFOIterator::done(void)
  {
    // we never know when we're done
    return false;
  }

  size_t WrappingFIFOIterator::step(size_t max_bytes, AddressInfo &info,
				    unsigned flags,
				    bool tentative /*= false*/)
  {
    assert(!tentative_valid);

    if(tentative) {
      prev_offset = offset;
      tentative_valid = true;
    }

    info.base_offset = base + offset;
    info.num_lines = 1;
    info.line_stride = 0;
    info.num_planes = 1;
    info.plane_stride = 0;
    size_t bytes;
    size_t bytes_left = size - offset;
    if(bytes_left <= max_bytes) {
      offset = 0;
      bytes = bytes_left;
    } else {
      offset += max_bytes;
      bytes = max_bytes;
    }
    info.bytes_per_chunk = bytes;
    return bytes;
  }

  void WrappingFIFOIterator::confirm_step(void)
  {
    assert(tentative_valid);
    tentative_valid = false;
  }

  void WrappingFIFOIterator::cancel_step(void)
  {
    assert(tentative_valid);
    offset = prev_offset;
    tentative_valid = false;
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<TransferIterator, WrappingFIFOIterator> WrappingFIFOIterator::serdez_subclass;

  template <typename S>
  bool WrappingFIFOIterator::serialize(S& serializer) const
  {
    return ((serializer << base) &&
	    (serializer << size) &&
	    (serializer << offset) &&
	    (serializer << prev_offset) &&
	    (serializer << tentative_valid));
  }

    void CopyRequest::perform_new_dma(Memory src_mem, Memory dst_mem)
    {
      // SJT: this code is pretty broken if there are multiple instance pairs
      assert(oas_by_inst->size() == 1);
      RegionInstance src_inst = oas_by_inst->begin()->first.first;
      RegionInstance dst_inst = oas_by_inst->begin()->first.second;

      // construct the iterators for the source and dest instances - these
      //  know about each other so they can potentially conspire about
      //  iteration order
      std::vector<FieldID> src_fields, dst_fields;
      std::vector<size_t> src_field_offsets, dst_field_offsets, field_sizes;
      CustomSerdezID serdez_id = 0;
      for(OASVec::const_iterator it2 = oas_by_inst->begin()->second.begin();
	  it2 != oas_by_inst->begin()->second.end();
	  ++it2) {
	src_fields.push_back(it2->src_field_id);
	dst_fields.push_back(it2->dst_field_id);
	src_field_offsets.push_back(it2->src_subfield_offset);
	dst_field_offsets.push_back(it2->dst_subfield_offset);
	field_sizes.push_back(it2->size);
	if(it2->serdez_id != 0) {
	  assert((serdez_id == 0) || (serdez_id == it2->serdez_id));
	  serdez_id = it2->serdez_id;
	}
      }
      TransferIterator *src_iter;
      if(gather_info == 0) {
	src_iter = domain->create_iterator(src_inst, dst_inst,
					   src_fields, src_field_offsets,
					   field_sizes);
      } else {
	// "source" is the addresses, not the indirectly-accessed data
	assert(!src_inst.exists());
	src_inst = gather_info->get_pointer_instance();
	src_iter = gather_info->create_address_iterator(dst_inst);
      }
      TransferIterator *dst_iter;
      if(scatter_info == 0) {
	dst_iter = domain->create_iterator(dst_inst,src_inst,
					   dst_fields, dst_field_offsets,
					   field_sizes);
      } else {
	// "dest" is the addresses (which we read!), not the indirectly-named data
	assert(!dst_inst.exists());
	dst_inst = scatter_info->get_pointer_instance();
	dst_iter = scatter_info->create_address_iterator(src_inst);
      }

      // we're going to need pre/next xdguids, so precreate all of them
      xd_ids.resize(xd_nodes.size(), XferDes::XFERDES_NO_GUID);
      typedef std::pair<XferDesID,int> IBEdge;
      const IBEdge dfl_edge(XferDes::XFERDES_NO_GUID, 0);
      std::vector<IBEdge> ib_pre_ids(ib_edges.size(), dfl_edge);
      std::vector<IBEdge> ib_next_ids(ib_edges.size(), dfl_edge);

      for(size_t i = 0; i < xd_nodes.size(); i++) {
	XferDesID new_xdid = get_xdq_singleton()->get_guid(xd_nodes[i].target_node);

	xd_ids[i] = new_xdid;
	
	for(size_t j = 0; j < xd_nodes[i].inputs.size(); j++)
	  if(xd_nodes[i].inputs[j].edge_id >= 0)
	    ib_next_ids[xd_nodes[i].inputs[j].edge_id] = std::make_pair(new_xdid, j);
	
	for(size_t j = 0; j < xd_nodes[i].outputs.size(); j++)
	  if(xd_nodes[i].outputs[j].edge_id >= 0)
	    ib_pre_ids[xd_nodes[i].outputs[j].edge_id] = std::make_pair(new_xdid, j);
      }

      for(size_t i = 0; i < ib_edges.size(); i++)
	log_new_dma.info() << "ib_edges[" << i << "]: mem=" << ib_edges[i].memory
			   << " offset=" << ib_edges[i].offset << " size=" << ib_edges[i].size
			   << " pre=" << std::hex << ib_pre_ids[i].first << std::dec << '[' << ib_pre_ids[i].second << ']'
			   << " next=" << std::hex << ib_next_ids[i].first << std::dec << '[' << ib_next_ids[i].second << ']';
      
      // now actually create xfer descriptors for each template node in our DAG
      for(size_t i = 0; i < xd_nodes.size(); i++) {
	const XDTemplate& xdt = xd_nodes[i];
	
	NodeID xd_target_node = xdt.target_node;
	XferDesID xd_guid = xd_ids[i];
	XferDesKind kind = xdt.kind;
	XferDesFactory *xd_factory = xdt.factory;

	log_new_dma.info() << "xd_nodes[" << i << "]: id=" << std::hex << xd_guid << std::dec << " kind=" << kind;
	// TODO: put back! << " inputs=" << PrettyVector<int>(xdt.inputs) << " outputs=" << PrettyVector<int>(xdt.outputs);
	
	bool mark_started = false;
	std::vector<XferDesPortInfo> inputs_info(xdt.inputs.size());
	for(size_t j = 0; j < xdt.inputs.size(); j++) {
	  XferDesPortInfo& ii = inputs_info[j];

	  ii.port_type = ((int(j) == xdt.gather_control_input) ?
			    XferDesPortInfo::GATHER_CONTROL_PORT :
			  (int(j) == xdt.scatter_control_input) ?
			    XferDesPortInfo::SCATTER_CONTROL_PORT :
			    XferDesPortInfo::DATA_PORT);

	  if(xdt.inputs[j].edge_id == XDTemplate::SRC_INST) {
	    ii.peer_guid = XferDes::XFERDES_NO_GUID;
	    ii.peer_port_idx = 0;
	    ii.indirect_port_idx = -1;
	    ii.mem = src_inst.get_location();
	    ii.inst = src_inst;
	    ii.iter = src_iter;
	    ii.serdez_id = serdez_id;
	    ii.ib_offset = 0;
	    ii.ib_size = 0;
	    mark_started = true;
	  } else if(xdt.inputs[j].edge_id <= XDTemplate::INDIRECT_BASE) {
	    int ind_idx = XDTemplate::INDIRECT_BASE - xdt.inputs[j].edge_id;
	    assert(xdt.inputs[j].indirect_inst.exists());
	    //log_dma.print() << "indirect iter: inst=" << xdt.inputs[j].indirect_inst;
	    ii.peer_guid = XferDes::XFERDES_NO_GUID;
	    ii.peer_port_idx = 0;
	    ii.indirect_port_idx = ind_idx;
	    ii.mem = xdt.inputs[j].indirect_inst.get_location();
	    ii.inst = xdt.inputs[j].indirect_inst;
	    ii.iter = gather_info->create_indirect_iterator(ii.mem,
							    xdt.inputs[j].indirect_inst,
							    src_fields,
							    src_field_offsets,
							    field_sizes);
	    ii.serdez_id = serdez_id;
	    ii.ib_offset = 0;
	    ii.ib_size = 0;
	  } else if(xdt.inputs[j].edge_id == XDTemplate::DST_INST) {
	    // this happens when doing a scatter and DST_INST is actually the
	    //  destination addresses (which we read)
	    ii.peer_guid = XferDes::XFERDES_NO_GUID;
	    ii.peer_port_idx = 0;
	    ii.indirect_port_idx = -1;
	    ii.mem = dst_inst.get_location();
	    ii.inst = dst_inst;
	    ii.iter = dst_iter;
	    ii.serdez_id = 0;
	    ii.ib_offset = 0;
	    ii.ib_size = 0;
	  } else {
	    ii.peer_guid = ib_pre_ids[xdt.inputs[j].edge_id].first;
	    ii.peer_port_idx = ib_pre_ids[xdt.inputs[j].edge_id].second;
	    ii.indirect_port_idx = -1;
	    ii.mem = ib_edges[xdt.inputs[j].edge_id].memory;
	    ii.inst = RegionInstance::NO_INST;
	    ii.ib_offset = ib_edges[xdt.inputs[j].edge_id].offset;
	    ii.ib_size = ib_edges[xdt.inputs[j].edge_id].size;
	    ii.iter = new WrappingFIFOIterator(ii.ib_offset, ii.ib_size);
	    ii.serdez_id = 0;
	  }
	}

	std::vector<XferDesPortInfo> outputs_info(xdt.outputs.size());
	for(size_t j = 0; j < xdt.outputs.size(); j++) {
	  XferDesPortInfo& oi = outputs_info[j];

	  oi.port_type = XferDesPortInfo::DATA_PORT;

	  if(xdt.outputs[j].edge_id == XDTemplate::DST_INST) {
	    oi.peer_guid = XferDes::XFERDES_NO_GUID;
	    oi.peer_port_idx = 0;
	    oi.indirect_port_idx = -1;
	    oi.mem = dst_inst.get_location();
	    oi.inst = dst_inst;
	    oi.iter = dst_iter;
	    oi.serdez_id = serdez_id;
	    oi.ib_offset = 0;
	    oi.ib_size = 0;  // doesn't matter
	  } else if(xdt.outputs[j].edge_id <= XDTemplate::INDIRECT_BASE) {
	    int ind_idx = XDTemplate::INDIRECT_BASE - xdt.outputs[j].edge_id;
	    assert(xdt.outputs[j].indirect_inst.exists());
	    //log_dma.print() << "indirect iter: inst=" << xdt.outputs[j].indirect_inst;
	    oi.peer_guid = XferDes::XFERDES_NO_GUID;
	    oi.peer_port_idx = 0;
	    oi.indirect_port_idx = ind_idx;
	    oi.mem = xdt.outputs[j].indirect_inst.get_location();
	    oi.inst = xdt.outputs[j].indirect_inst;
	    oi.iter = scatter_info->create_indirect_iterator(oi.mem,
							     xdt.outputs[j].indirect_inst,
							     dst_fields,
							     dst_field_offsets,
							     field_sizes);
	    oi.serdez_id = serdez_id;
	    oi.ib_offset = 0;
	    oi.ib_size = 0;
	  } else {
	    oi.peer_guid = ib_next_ids[xdt.outputs[j].edge_id].first;
	    oi.peer_port_idx = ib_next_ids[xdt.outputs[j].edge_id].second;
	    oi.indirect_port_idx = -1;
	    oi.mem = ib_edges[xdt.outputs[j].edge_id].memory;
	    oi.inst = RegionInstance::NO_INST;
	    oi.ib_offset = ib_edges[xdt.outputs[j].edge_id].offset;
	    oi.ib_size = ib_edges[xdt.outputs[j].edge_id].size;
	    oi.iter = new WrappingFIFOIterator(oi.ib_offset, oi.ib_size);
	    oi.serdez_id = 0;
	  }
	}

	XferDesFence* complete_fence = new XferDesFence(this);
	add_async_work_item(complete_fence);
	
	xd_factory->create_xfer_des(this, Network::my_node_id, xd_target_node,
				    xd_guid,
				    inputs_info,
				    outputs_info,
				    mark_started,
				    //pre_buf, cur_buf, domain, oasvec_src,
				    16 * 1024 * 1024/*max_req_size*/, 100/*max_nr*/,
				    priority, complete_fence);
	xd_factory->release();
      }
    }

    void CopyRequest::perform_dma(void)
    {
      log_dma.debug("request %p executing", this);

      DetailedTimer::ScopedPush sp(TIME_COPY);

      // <NEWDMA>
      if(measurements.wants_measurement<ProfilingMeasurements::OperationMemoryUsage>()) {
        const InstPair &pair = oas_by_inst->begin()->first;
        size_t total_field_size = 0;
        for (OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
          for (size_t i = 0; i < it->second.size(); i++) {
            total_field_size += it->second[i].size;
          }
        }

        ProfilingMeasurements::OperationMemoryUsage usage;
        usage.source = pair.first.get_location();
        usage.target = pair.second.get_location();
        usage.size = total_field_size * domain->volume();
        measurements.add_measurement(usage);
      }

      Memory src_mem = oas_by_inst->begin()->first.first.get_location();
      Memory dst_mem = oas_by_inst->begin()->first.second.get_location();
      perform_new_dma(src_mem, dst_mem);

      // make sure logging precedes the call to mark_finished below
      log_dma.info() << "dma request " << (void *)this << " finished - is="
                     << *domain << " before=" << before_copy << " after=" << get_finish_event();
      mark_finished(true/*successful*/);
      return;
      // </NEWDMA>
    } 

    ReduceRequest::ReduceRequest(const void *data, size_t datalen,
				 ReductionOpID _redop_id,
				 bool _red_fold,
				 Event _before_copy,
				 GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
				 int _priority)
      : DmaRequest(_priority, _after_copy, _after_gen),
	inst_lock_event(Event::NO_EVENT),
	redop_id(_redop_id), red_fold(_red_fold),
	before_copy(_before_copy)
    {
      Serialization::FixedBufferDeserializer deserializer(data, datalen);

      domain = TransferDomain::deserialize_new(deserializer);
      bool ok = ((deserializer >> srcs) &&
		 (deserializer >> dst) &&
		 (deserializer >> inst_lock_needed) &&
		 (deserializer >> requests));
      assert((domain != 0) && ok && (deserializer.bytes_left() == 0));

      Operation::reconstruct_measurements();

      log_dma.info() << "dma request " << (void *)this << " deserialized - is="
		     << *domain
		     << " " << (red_fold ? "fold" : "apply") << " " << redop_id
		     << " before=" << before_copy << " after=" << get_finish_event();
      log_dma.info() << "dma request " << (void *)this << " field: "
		     << srcs[0].inst << "[" << srcs[0].field_id << "+" << srcs[0].subfield_offset << "]"
		     << "+" << (srcs.size()-1) << "->"
		     << dst.inst << "[" << dst.field_id << "+" << dst.subfield_offset << "] size=" << dst.size;
    }

    ReduceRequest::ReduceRequest(const TransferDomain *_domain, //const Domain& _domain,
				 const std::vector<CopySrcDstField>& _srcs,
				 const CopySrcDstField& _dst,
				 bool _inst_lock_needed,
				 ReductionOpID _redop_id,
				 bool _red_fold,
				 Event _before_copy,
				 GenEventImpl *_after_copy, EventImpl::gen_t _after_gen,
				 int _priority, 
                                 const ProfilingRequestSet &reqs)
      : DmaRequest(_priority, _after_copy, _after_gen, reqs),
	domain(_domain->clone()),
	dst(_dst), 
	inst_lock_needed(_inst_lock_needed), inst_lock_event(Event::NO_EVENT),
	redop_id(_redop_id), red_fold(_red_fold),
	before_copy(_before_copy)
    {
      srcs.insert(srcs.end(), _srcs.begin(), _srcs.end());

      log_dma.info() << "dma request " << (void *)this << " created - is="
		     << *domain
		     << " " << (red_fold ? "fold" : "apply") << " " << redop_id
		     << " before=" << before_copy << " after=" << get_finish_event();
      log_dma.info() << "dma request " << (void *)this << " field: "
		     << srcs[0].inst << "[" << srcs[0].field_id << "+" << srcs[0].subfield_offset << "]"
		     << "+" << (srcs.size()-1) << "->"
		     << dst.inst << "[" << dst.field_id << "+" << dst.subfield_offset << "] size=" << dst.size;
    }

    ReduceRequest::~ReduceRequest(void)
    {
      delete domain;
    }

    void ReduceRequest::forward_request(NodeID target_node)
    {
      ActiveMessage<RemoteCopyMessage> amsg(target_node,65536);
      amsg->redop_id = redop_id;
      amsg->red_fold = red_fold;
      amsg->before_copy = before_copy;
      amsg->after_copy = get_finish_event();
      amsg->priority = priority;
      amsg << *domain;
      amsg << srcs;
      amsg << dst;
      amsg << inst_lock_needed;
      amsg << requests;
      amsg.commit();
      log_dma.debug() << "forwarding copy: target=" << target_node << " finish=" << get_finish_event();
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
	Event e = domain->request_metadata();
	if(!e.has_triggered()) {
	  log_dma.debug() << "transfer domain metadata - req=" << (void *)this
			  << " ready=" << e;
	  waiter.sleep_on_event(e);
	  return false;
	}

	// now go through all source instance pairs
	for(std::vector<CopySrcDstField>::iterator it = srcs.begin();
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

    void ReduceRequest::perform_dma(void)
    {
      log_dma.debug("request %p executing", this);

      DetailedTimer::ScopedPush sp(TIME_COPY);

      // code assumes a single source field for now
      assert(srcs.size() == 1);

      MemoryImpl *src_mem = get_runtime()->get_memory_impl(srcs[0].inst);
      MemoryImpl *dst_mem = get_runtime()->get_memory_impl(dst.inst);

      bool dst_is_remote = ((dst_mem->kind == MemoryImpl::MKIND_REMOTE) ||
			    (dst_mem->kind == MemoryImpl::MKIND_RDMA));
      unsigned rdma_sequence_id = (dst_is_remote ?
				     rdma_sequence_no.fetch_add(1) :
				     0);
      unsigned rdma_count = 0;

      std::vector<FieldID> src_field(1, srcs[0].field_id);
      std::vector<FieldID> dst_field(1, dst.field_id);
      std::vector<size_t> src_field_offset(1, srcs[0].subfield_offset);
      std::vector<size_t> dst_field_offset(1, dst.subfield_offset);
      std::vector<size_t> src_field_size(1, srcs[0].size);
      std::vector<size_t> dst_field_size(1, dst.size);
      TransferIterator *src_iter = domain->create_iterator(srcs[0].inst,
							   dst.inst,
							   src_field,
							   src_field_offset,
							   src_field_size);
      TransferIterator *dst_iter = domain->create_iterator(dst.inst,
							   srcs[0].inst,
							   dst_field,
							   dst_field_offset,
							   dst_field_size);

      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      if(redop == 0) {
	log_dma.fatal() << "no reduction op registered for ID " << redop_id;
	abort();
      }
      size_t src_elem_size = red_fold ? redop->sizeof_rhs : redop->sizeof_lhs;

      size_t total_bytes = 0;

      void *src_scratch_buffer = 0;
      void *dst_scratch_buffer = 0;
      size_t src_scratch_size = 0;
      size_t dst_scratch_size = 0;

      while(!src_iter->done()) {
	TransferIterator::AddressInfo src_info, dst_info;

	size_t max_bytes = (size_t)-1;
	size_t src_bytes = src_iter->step(max_bytes, src_info, 0,
					  true /*tentative*/);
	assert(src_bytes >= 0);
	size_t num_elems = src_bytes / src_elem_size;
	size_t exp_dst_bytes = num_elems * redop->sizeof_rhs;
	size_t dst_bytes = dst_iter->step(exp_dst_bytes, dst_info, 0);
	if(dst_bytes == exp_dst_bytes) {
	  // good, confirm the source step
	  src_iter->confirm_step();
	} else {
	  // bad, cancel the source step and try a smaller one
	  src_iter->cancel_step();
	  num_elems = dst_bytes / redop->sizeof_rhs;
	  size_t exp_src_bytes = num_elems * src_elem_size;
	  src_bytes = src_iter->step(exp_src_bytes, src_info, 0);
	  assert(src_bytes == exp_src_bytes);
	}

	total_bytes += dst_bytes;

	// can we directly access the source data?
	const void *src_ptr = src_mem->get_direct_ptr(src_info.base_offset,
						      src_info.bytes_per_chunk);
	if((src_ptr == 0) || (src_mem->kind == MemoryImpl::MKIND_GPUFB)) {
	  // nope, make a local copy via get_bytes
	  if(src_info.bytes_per_chunk > src_scratch_size) {
	    if(src_scratch_size > 0)
	      free(src_scratch_buffer);
	    // allocate 2x in case the next block is a little bigger
	    src_scratch_size = src_info.bytes_per_chunk * 2;
	    src_scratch_buffer = malloc(src_scratch_size);
	  }
	  src_mem->get_bytes(src_info.base_offset,
			     src_scratch_buffer,
			     src_info.bytes_per_chunk);
	  src_ptr = src_scratch_buffer;
	}

	// now look at destination and deal with two fast cases
	  
	// case 1: destination is remote (quickest to check)
	if(dst_is_remote) {
	  // have to tell rdma to make a copy if we're using the temp buffer
	  bool make_copy = (src_ptr == src_scratch_buffer);
	  rdma_count += do_remote_reduce(dst_mem->me,
					 dst_info.base_offset,
					 redop_id, red_fold,
					 src_ptr, num_elems,
					 src_elem_size, redop->sizeof_rhs,
					 rdma_sequence_id,
					 make_copy);
	} else {
	  // case 2: destination is directly accessible
	  void *dst_ptr = dst_mem->get_direct_ptr(dst_info.base_offset,
						  dst_info.bytes_per_chunk);
	  if(dst_ptr && (dst_mem->kind != MemoryImpl::MKIND_GPUFB)) {
	    if(red_fold)
	      redop->fold(dst_ptr, src_ptr, num_elems, false /*!excl*/);
	    else
	      redop->apply(dst_ptr, src_ptr, num_elems, false /*!excl*/);
	  } else {
	    // case 3: fallback - use get_bytes/put_bytes combo

	    // need a buffer for destination data
	    if(dst_info.bytes_per_chunk > dst_scratch_size) {
	      if(dst_scratch_size > 0)
		free(dst_scratch_buffer);
	      // allocate 2x in case the next block is a little bigger
	      dst_scratch_size = dst_info.bytes_per_chunk * 2;
	      dst_scratch_buffer = malloc(dst_scratch_size);
	    }
	    dst_mem->get_bytes(dst_info.base_offset,
			       dst_scratch_buffer,
			       dst_info.bytes_per_chunk);
	    if(red_fold)
	      redop->fold(dst_scratch_buffer, src_ptr, num_elems, true/*excl*/);
	    else
	      redop->apply(dst_scratch_buffer, src_ptr, num_elems, true/*excl*/);
	    dst_mem->put_bytes(dst_info.base_offset,
			       dst_scratch_buffer,
			       dst_info.bytes_per_chunk);
	  }
	}
      }
      assert(dst_iter->done());

      delete src_iter;
      delete dst_iter;

      if(src_scratch_size > 0)
	free(src_scratch_buffer);
      if(dst_scratch_size > 0)
	free(dst_scratch_buffer);

      // if we did any actual reductions, send a fence, otherwise trigger here
      if(rdma_count > 0) {
	RemoteWriteFence *fence = new RemoteWriteFence(this);
	this->add_async_work_item(fence);
	do_remote_fence(dst_mem->me, rdma_sequence_id, rdma_count, fence);
      }

      //printf("kinds: " IDFMT "=%d " IDFMT "=%d\n", src_mem.id, src_mem.impl()->kind, dst_mem.id, dst_mem.impl()->kind);

      log_dma.info() << "dma request " << (void *)this << " finished - is="
		     << *domain
		     << " " << (red_fold ? "fold" : "apply") << " " << redop_id
		     << " before=" << before_copy << " after=" << get_finish_event();

      if(measurements.wants_measurement<ProfilingMeasurements::OperationMemoryUsage>()) {
        ProfilingMeasurements::OperationMemoryUsage usage;  
        // Not precise, but close enough for now
        usage.source = srcs[0].inst.get_location();
        usage.target = dst.inst.get_location();
        usage.size = total_bytes;
        measurements.add_measurement(usage);
      }
    }

    FillRequest::FillRequest(const void *data, size_t datalen,
                             RegionInstance inst,
                             FieldID field_id, unsigned size,
                             Event _before_fill,
			     GenEventImpl *_after_fill, EventImpl::gen_t _after_gen,
                             int _priority)
      : DmaRequest(_priority, _after_fill, _after_gen)
      , before_fill(_before_fill)
    {
      dst.inst = inst;
      dst.field_id = field_id;
      dst.subfield_offset = 0;
      dst.size = size;

      Serialization::FixedBufferDeserializer deserializer(data, datalen);

      domain = TransferDomain::deserialize_new(deserializer);
      ByteArray ba; // TODO
      bool ok = ((deserializer >> ba) &&
		 (deserializer >> requests));
      assert((domain != 0) && ok && (deserializer.bytes_left() == 0));

      fill_size = ba.size();
      fill_buffer = ba.detach();

      assert(fill_size == size);

      Operation::reconstruct_measurements();

      log_dma.info() << "dma request " << (void *)this << " deserialized - is="
		     << *domain << " fill dst=" << dst.inst << "[" << dst.field_id << "+" << dst.subfield_offset << "] size="
		     << fill_size << " before=" << _before_fill << " after=" << get_finish_event();
    }

    FillRequest::FillRequest(const TransferDomain *_domain, //const Domain &d, 
                             const CopySrcDstField &_dst,
                             const void *_fill_value, size_t _fill_size,
                             Event _before_fill,
			     GenEventImpl *_after_fill, EventImpl::gen_t _after_gen,
			     int _priority,
                             const ProfilingRequestSet &reqs)
      : DmaRequest(_priority, _after_fill, _after_gen, reqs)
      , domain(_domain->clone())
      , dst(_dst)
      , before_fill(_before_fill)
    {
      fill_size = _fill_size;
      fill_buffer = malloc(fill_size);
      memcpy(fill_buffer, _fill_value, fill_size);

      assert(dst.size == fill_size);
      log_dma.info() << "dma request " << (void *)this << " created - is="
		     << *domain << " fill dst=" << dst.inst << "[" << dst.field_id << "+" << dst.subfield_offset << "] size="
		     << fill_size << " before=" << _before_fill << " after=" << get_finish_event();
      {
	LoggerMessage msg(log_dma.debug());
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
      delete domain;
    }

    void FillRequest::forward_request(NodeID target_node)
    {
      ByteArray ba(fill_buffer, fill_size); // TODO
      ActiveMessage<RemoteFillMessage> amsg(target_node,65536);
      amsg->inst = dst.inst;
      amsg->field_id = dst.field_id;
      assert(dst.subfield_offset == 0);
      amsg->size = fill_size;
      amsg->before_fill = before_fill;
      amsg->after_fill = get_finish_event();
      amsg << *domain;
      amsg << ba;
      amsg << requests;
      amsg.commit();
      log_dma.debug() << "forwarding fill: target=" << target_node << " finish=" << get_finish_event();
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
	Event e = domain->request_metadata();
	if(!e.has_triggered()) {
	  log_dma.debug() << "transfer domain metadata - req=" << (void *)this
			  << " ready=" << e;
	  waiter.sleep_on_event(e);
	  return false;
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

    static void repeat_fill(void *dst, const void *src, size_t bytes, size_t count)
    {
#define SPECIALIZE_FILL(TYPE, N)                                        \
      {									\
	TYPE *dsttyped = reinterpret_cast<TYPE *>(dst);			\
	const TYPE *srctyped = reinterpret_cast<const TYPE *>(src);	\
	for(size_t i = 0; i < count; i++)				\
	  for(size_t j = 0; j < N; j++)					\
	    *dsttyped++ = srctyped[j];					\
      }

      switch(bytes) {
      case sizeof(uint32_t):
	{
	  SPECIALIZE_FILL(uint32_t, 1);
	  break;
	}
      case sizeof(uint64_t):
	{
	  SPECIALIZE_FILL(uint64_t, 1);
	  break;
	}
      case 2 * sizeof(uint64_t):
	{
	  SPECIALIZE_FILL(uint64_t, 2);
	  break;
	}
      case 4 * sizeof(uint64_t):
	{
	  SPECIALIZE_FILL(uint64_t, 4);
	  break;
	}
      default:
	{
	  for(size_t i = 0; i < count; i++)
	    memcpy(reinterpret_cast<char *>(dst) + (bytes * i), src, bytes);
	  break;
	}
      }

#undef SPECIALIZE_FILL
    }

    void FillRequest::perform_dma(void)
    {
      // if we are doing large chunks of data, we will build a buffer with
      //  multiple copies of the same data to make fewer calls to put_bytes
      void *rep_buffer = 0;
      size_t rep_size = 0;

      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(dst.inst.get_location());

      std::vector<FieldID> dst_field(1, dst.field_id);
      std::vector<size_t> dst_field_offset(1, dst.subfield_offset);
      std::vector<size_t> dst_field_size(1, dst.size);
      TransferIterator *iter = domain->create_iterator(dst.inst,
						       RegionInstance::NO_INST,
						       dst_field,
						       dst_field_offset,
						       dst_field_size);
#ifdef REALM_USE_CUDA
      // fills to GPU FB memory are offloaded to the GPU itself
      if (mem_impl->lowlevel_kind == Memory::GPU_FB_MEM) {
	Cuda::GPU *gpu = static_cast<Cuda::GPUFBMemory *>(mem_impl)->gpu;
	size_t total_bytes = 0;
	while(!iter->done()) {
	  TransferIterator::AddressInfo info;

	  size_t max_bytes = (size_t)-1;
	  unsigned flags = (TransferIterator::LINES_OK |
			    TransferIterator::PLANES_OK);
	  size_t act_bytes = iter->step(max_bytes, info, flags);
	  assert(act_bytes >= 0);
	  total_bytes += act_bytes;

	  if(info.num_planes == 1) {
	    if(info.num_lines == 1) {
	      gpu->fill_within_fb(info.base_offset, info.bytes_per_chunk,
				  fill_buffer, fill_size);
	    } else {
	      gpu->fill_within_fb_2d(info.base_offset, info.line_stride,
				     info.bytes_per_chunk, info.num_lines,
				     fill_buffer, fill_size);
	    }
	  } else {
	    gpu->fill_within_fb_3d(info.base_offset, info.line_stride,
				   info.plane_stride,
				   info.bytes_per_chunk, info.num_lines,
				   info.num_planes,
				   fill_buffer, fill_size);
	  }
	}

	// if we did any asynchronous operations, insert a fence
	if(total_bytes > 0)
	  gpu->fence_within_fb(this);

	// fall through so the clean-up is the same
      }
#endif

#ifdef REALM_USE_HDF5
      // fills of an HDF5 instance are also handled specially
      if (mem_impl->lowlevel_kind == Memory::HDF_MEM) {
	hid_t file_id = -1;
	hid_t dset_id = -1;
	hid_t dtype_id = -1;
	const std::string *prev_filename = 0;
	const std::string *prev_dsetname = 0;
	while(!iter->done()) {
	  TransferIterator::AddressInfoHDF5 info;
	  size_t act_bytes = iter->step(size_t(-1), // max_bytes
					info);
	  assert(act_bytes >= 0);

	  // compare the pointers, not the string contents...
	  if(info.filename != prev_filename) {
	    // close dataset too
	    if(dset_id != -1) {
	      CHECK_HDF5( H5Tclose(dtype_id) );
	      CHECK_HDF5( H5Dclose(dset_id) );
	      prev_dsetname = 0;
	    }
	    if(file_id != -1)
	      CHECK_HDF5( H5Fclose(file_id) );

	    CHECK_HDF5( file_id = H5Fopen(info.filename->c_str(),
					  H5F_ACC_RDWR, H5P_DEFAULT) );
	    prev_filename = info.filename;
	  }

	  if(info.dsetname != prev_dsetname) {
	    if(dset_id != -1) {
	      CHECK_HDF5( H5Tclose(dtype_id) );
	      CHECK_HDF5( H5Dclose(dset_id) );
	    }
	    CHECK_HDF5( dset_id = H5Dopen2(file_id, info.dsetname->c_str(),
					   H5P_DEFAULT) );
	    CHECK_HDF5( dtype_id = H5Dget_type(dset_id) );
	    size_t dtype_size = H5Tget_size(dtype_id);
	    assert(dtype_size == fill_size);
	    prev_dsetname = info.dsetname;
	  }

	  // HDF5 doesn't seem to offer a way to fill a file without building
	  //  an equivalently-sized memory buffer first, and we can't do
          //  point-wise because libhdf5 doesn't buffer, so try to find a 
	  //  reasonably-sized chunk to do fills with - this code is still
	  //  limited to each dimension being all-or-nothing
	  int dims = info.extent.size();
	  size_t max_chunk_elems = (32 << 20) / fill_size; // 32MB
	  size_t chunk_elems = 1;
	  // because HDF5 uses C data ordering, we need to group from the last
	  //  dim backwards
	  int step_dims = dims;
	  while(step_dims > 0) {
	    size_t new_elems = chunk_elems * info.extent[step_dims - 1];
	    if(new_elems > max_chunk_elems) break;
	    chunk_elems = new_elems;
	    step_dims--;
	  }
	  void *chunk_data = fill_buffer;
	  if(chunk_elems > 1) {
	    chunk_data = malloc(chunk_elems * fill_size);
	    assert(chunk_data != 0);
	    repeat_fill(chunk_data, fill_buffer, fill_size, chunk_elems);
	  }
	  std::vector<hsize_t> mem_dims(dims, 1);
	  for(int i = step_dims; i < dims; i++)
	    mem_dims[i] = info.extent[i];
	  hid_t mem_space_id, file_space_id;
	  CHECK_HDF5( mem_space_id = H5Screate_simple(dims, mem_dims.data(),
						      NULL) );
	  CHECK_HDF5( file_space_id = H5Dget_space(dset_id) );

	  std::vector<hsize_t> cur_pos(info.offset);
	  std::vector<hsize_t> cur_size = mem_dims;
	  while(true) {
	    CHECK_HDF5( H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET,
					    cur_pos.data(), 0,
					    cur_size.data(), 0) );
	    CHECK_HDF5( H5Dwrite(dset_id, dtype_id,
				 mem_space_id, file_space_id,
				 H5P_DEFAULT, chunk_data) );
	    // advance to next position
	    int d = 0;
	    while(d < step_dims) {
	      if(++cur_pos[d] < (info.offset[d] + info.extent[d]))
		break;
	      cur_pos[d] = info.offset[d];
	      d++;
	    }
	    if(d >= step_dims) break;
	  }

	  if(chunk_data != fill_buffer)
	    free(chunk_data);

	  CHECK_HDF5( H5Sclose(mem_space_id) );
	  CHECK_HDF5( H5Sclose(file_space_id) );
	}

	// close the last dset and file we touched
	if(dset_id != -1) {
	  CHECK_HDF5( H5Tclose(dtype_id) );
	  CHECK_HDF5( H5Dclose(dset_id) );
	}
	if(file_id != -1)
	  CHECK_HDF5( H5Fclose(file_id) );
      }
#endif

      while(!iter->done()) {
	TransferIterator::AddressInfo info;

	size_t max_bytes = (size_t)-1;
	// code below is 2D/3D-capable
	unsigned flags = (TransferIterator::LINES_OK |
			  TransferIterator::PLANES_OK);
	size_t act_bytes = iter->step(max_bytes, info, flags);
	assert(act_bytes >= 0);

	// decide whether to use the original fill buffer or one that
	//  repeats the data several times
	const void *use_buffer = fill_buffer;
	size_t use_size = fill_size;
	const size_t MAX_REP_SIZE = 32768;
	if((info.bytes_per_chunk > fill_size) &&
	   ((fill_size * 2) <= MAX_REP_SIZE)) {
	  if(!rep_buffer) {
	    size_t rep_elems = MAX_REP_SIZE / fill_size;
	    rep_size = rep_elems * fill_size;
	    rep_buffer = malloc(rep_size);
	    assert(rep_buffer != 0);
	    repeat_fill(rep_buffer, fill_buffer, fill_size, rep_elems);
	  }
	  use_buffer = rep_buffer;
	  use_size = rep_size;
	}

	for(size_t p = 0; p < info.num_planes; p++)
	  for(size_t l = 0; l < info.num_lines; l++) {
	    size_t ofs = 0;
	    while((ofs + use_size) < info.bytes_per_chunk) {
	      mem_impl->put_bytes(info.base_offset + 
				  (p * info.plane_stride) +
				  (l * info.line_stride) +
				  ofs, use_buffer, use_size);
	      ofs += use_size;
	    }
	    size_t bytes_left = info.bytes_per_chunk - ofs;
	    assert((bytes_left > 0) && (bytes_left <= use_size));
	    mem_impl->put_bytes(info.base_offset + 
				(p * info.plane_stride) +
				(l * info.line_stride) +
				ofs, use_buffer, bytes_left);
	  }
      }

      if(rep_buffer)
	free(rep_buffer);
      delete iter;

      if(measurements.wants_measurement<ProfilingMeasurements::OperationMemoryUsage>()) {
        ProfilingMeasurements::OperationMemoryUsage usage;
        usage.source = Memory::NO_MEMORY;
        usage.target = dst.inst.get_location();
        measurements.add_measurement(usage);
      }

      log_dma.info() << "dma request " << (void *)this << " finished - is="
		     << *domain << " fill dst=" << dst.inst << "[" << dst.field_id << "+" << dst.subfield_offset << "] size="
		     << fill_size << " before=" << before_fill << " after=" << get_finish_event();
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
          fill_elmts = std::min(inst_impl->metadata.block_size,2*max_size/fill_size);
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

    // for now we use a single queue for all (local) dmas
    DmaRequestQueue *dma_queue = 0;
    
    void DmaRequestQueue::worker_thread_loop(void)
    {
      log_dma.info("dma worker thread created");

      while(!shutdown_flag) {
	//aio_context->make_progress();
	//bool aio_idle = aio_context->empty();
	bool aio_idle = true;  // not polling here anymore

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
    
    void start_dma_worker_threads(int count, CoreReservationSet& crs)
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

    void start_dma_system(int count, bool pinned, int max_nr,
                          CoreReservationSet& crs,
			  BackgroundWorkManager *bgwork)
    {
      //log_dma.add_stream(&std::cerr, Logger::LEVEL_DEBUG, false, false);
      aio_context = new AsyncFileIOContext(256);
      aio_context->add_to_manager(bgwork);
      start_channel_manager(count, pinned, max_nr, crs, bgwork);
      ib_req_queue = new PendingIBQueue();
    }

    void stop_dma_system(void)
    {
      stop_channel_manager();
      delete ib_req_queue;
      ib_req_queue = 0;
#ifdef DEBUG_REALM
      aio_context->shutdown_work_item();
#endif
      delete aio_context;
      aio_context = 0;
    }

  /*static*/ void RemoteCopyMessage::handle_message(NodeID sender, const RemoteCopyMessage &args,
						    const void *data, size_t msglen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      GenEventImpl *after_copy = get_runtime()->get_genevent_impl(args.after_copy);
      
      // is this a copy or a reduction (they deserialize differently)
      if(args.redop_id == 0) {
	// a copy
	CopyRequest *r = new CopyRequest(data, msglen,
					 args.before_copy,
					 after_copy,
					 ID(args.after_copy).event_generation(),
					 args.priority);
	get_runtime()->optable.add_local_operation(args.after_copy, r);

	r->check_readiness(false, dma_queue);
      } else {
	// a reduction
	ReduceRequest *r = new ReduceRequest(data, msglen,
					     args.redop_id,
					     args.red_fold,
					     args.before_copy,
					     after_copy,
					     ID(args.after_copy).event_generation(),
					     args.priority);
	get_runtime()->optable.add_local_operation(args.after_copy, r);

	r->check_readiness(false, dma_queue);
      }
    }

  /*static*/ void RemoteFillMessage::handle_message(NodeID sender, const RemoteFillMessage &args,
						    const void *data, size_t msglen)
    {
      GenEventImpl *after_fill = get_runtime()->get_genevent_impl(args.after_fill);
      FillRequest *r = new FillRequest(data, msglen,
                                       args.inst,
                                       args.field_id,
                                       args.size,
                                       args.before_fill,
                                       after_fill,
				       ID(args.after_fill).event_generation(),
                                       0 /* no room for args.priority */);
      get_runtime()->optable.add_local_operation(args.after_fill, r);

      r->check_readiness(false, dma_queue);
    }

  ActiveMessageHandlerReg<RemoteFillMessage> remote_fill_message_handler;
  ActiveMessageHandlerReg<RemoteCopyMessage> remote_copy_message_handler;
  ActiveMessageHandlerReg<RemoteIBAllocRequestAsync> remote_ib_alloc_request_async_handler;
  ActiveMessageHandlerReg<RemoteIBAllocResponseAsync> remote_ib_alloc_response_async_handler;
  ActiveMessageHandlerReg<RemoteIBFreeRequestAsync> remote_ib_free_request_async_handler;

};
