/* Copyright 2014 Stanford University
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

#include "lowlevel_dma.h"
#ifdef USE_CUDA
#include "lowlevel_gpu.h"
#endif
#include "accessor.h"

#include <queue>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

using namespace LegionRuntime::Accessor;

#ifdef LEGION_LOGGING
#include "legion_logging.h"

using namespace LegionRuntime::HighLevel::LegionLogging;
#endif

#include "atomics.h"

namespace LegionRuntime {
  namespace LowLevel {

    Logger::Category log_dma("dma");

    typedef std::pair<Memory, Memory> MemPair;
    typedef std::pair<RegionInstance, RegionInstance> InstPair;
    struct OffsetsAndSize {
      unsigned src_offset, dst_offset, size;
    };
    typedef std::vector<OffsetsAndSize> OASVec;
    typedef std::map<InstPair, OASVec> OASByInst;
    typedef std::map<MemPair, OASByInst *> OASByMem;

    class MemPairCopier;

    class DmaRequest;

    class DmaRequestQueue {
    public:
      DmaRequestQueue(void);

      void enqueue_request(DmaRequest *r);

      DmaRequest *dequeue_request(bool sleep = true);

      void shutdown_queue(void);

    protected:
      gasnet_hsl_t queue_mutex;
      gasnett_cond_t queue_condvar;
      std::map<int, std::list<DmaRequest *> *> queues;
      int queue_sleepers;
      bool shutdown_flag;
    };

    class DmaRequest {
    public:
      DmaRequest(int _priority, Event _after_copy) 
	: state(STATE_INIT), priority(_priority), after_copy(_after_copy) {}

      virtual ~DmaRequest(void) {}

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq) = 0;

      virtual bool handler_safe(void) = 0;

      virtual void perform_dma(void) = 0;

      enum State {
	STATE_INIT,
	STATE_METADATA_FETCH,
	STATE_BEFORE_EVENT,
	STATE_INST_LOCK,
	STATE_READY,
	STATE_QUEUED,
	STATE_DONE
      };

      State state;
      int priority;
      Event after_copy;

      class Waiter : public Event::Impl::EventWaiter {
      public:
        Waiter(void) { }
        virtual ~Waiter(void) { }
      public:
	Reservation current_lock;
	DmaRequestQueue *queue;
	DmaRequest *req;

	void sleep_on_event(Event e, Reservation l = Reservation::NO_RESERVATION);

	virtual bool event_triggered(void);
	virtual void print_info(FILE *f);
      };
    };

    // dma requests come in two flavors:
    // 1) CopyRequests, which are per memory pair, and
    // 2) ReduceRequests, which have to be handled monolithically

    class CopyRequest : public DmaRequest {
    public:
      CopyRequest(const Domain& _domain,
		  OASByInst *_oas_by_inst,
		  Event _before_copy,
		  Event _after_copy,
		  int _priority);

      CopyRequest(const void *data, size_t datalen,
		  Event _before_copy,
		  Event _after_copy,
		  int _priority);

      virtual ~CopyRequest(void);

      size_t serialize(void *buffer, size_t maxlen);

      virtual bool check_readiness(bool just_check, DmaRequestQueue *rq);

      void perform_dma_mask(MemPairCopier *mpc);

      template <unsigned DIM>
      void perform_dma_rect(MemPairCopier *mpc);

      virtual void perform_dma(void);

      virtual bool handler_safe(void) { return(false); }

      Domain domain;
      OASByInst *oas_by_inst;
      Event before_copy;
      Waiter waiter; // if we need to wait on events
    };

    class ReduceRequest : public DmaRequest {
    public:
      ReduceRequest(const Domain& _domain,
		    const std::vector<Domain::CopySrcDstField>& _srcs,
		    const Domain::CopySrcDstField& _dst,
		    bool _inst_lock_needed,
		    ReductionOpID _redop_id,
		    bool _red_fold,
		    Event _before_copy,
		    Event _after_copy,
		    int _priority);

      ReduceRequest(const void *data, size_t datalen,
		    ReductionOpID _redop_id,
		    bool _red_fold,
		    Event _before_copy,
		    Event _after_copy,
		    int _priority);

      virtual ~ReduceRequest(void);

      size_t serialize(void *buffer, size_t maxlen);

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

    DmaRequestQueue::DmaRequestQueue(void)
    {
      gasnet_hsl_init(&queue_mutex);
      gasnett_cond_init(&queue_condvar);
      queue_sleepers = 0;
      shutdown_flag = false;
    }

    void DmaRequestQueue::shutdown_queue(void)
    {
      gasnet_hsl_lock(&queue_mutex);

      assert(queues.empty());

      // set the shutdown flag and wake up any sleepers
      shutdown_flag = true;

      gasnett_cond_broadcast(&queue_condvar);

      gasnet_hsl_unlock(&queue_mutex);
    }

    void DmaRequestQueue::enqueue_request(DmaRequest *r)
    {
      gasnet_hsl_lock(&queue_mutex);

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
	gasnett_cond_broadcast(&queue_condvar);
      }

      gasnet_hsl_unlock(&queue_mutex);
    }

    DmaRequest *DmaRequestQueue::dequeue_request(bool sleep /*= true*/)
    {
      gasnet_hsl_lock(&queue_mutex);

      // quick check - are there any requests at all?
      while(queues.empty()) {
	if(!sleep || shutdown_flag) {
	  gasnet_hsl_unlock(&queue_mutex);
	  return 0;
	}

	// sleep until there are, or until shutdown
	queue_sleepers++;
	gasnett_cond_wait(&queue_condvar, &queue_mutex.lock);
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

      gasnet_hsl_unlock(&queue_mutex);
      
      return r;
    }

    CopyRequest::CopyRequest(const Domain& _domain,
			     OASByInst *_oas_by_inst,
			     Event _before_copy,
			     Event _after_copy,
			     int _priority)
      : DmaRequest(_priority, _after_copy),
	domain(_domain), oas_by_inst(_oas_by_inst),
	before_copy(_before_copy)
    {
      log_dma.info("dma request %p created - " IDFMT "[%d]->" IDFMT "[%d]:%d (+%zd) (" IDFMT ") " IDFMT "/%d " IDFMT "/%d",
		   this,
		   oas_by_inst->begin()->first.first.id, 
		   oas_by_inst->begin()->second[0].src_offset,
		   oas_by_inst->begin()->first.second.id, 
		   oas_by_inst->begin()->second[0].dst_offset,
		   oas_by_inst->begin()->second[0].size,
		   oas_by_inst->size() - 1,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);

#ifdef LEGION_LOGGING
      log_timing_event(Processor::NO_PROC, after_copy, COPY_INIT);
#endif
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

      int priority = 0;

      while(((idata - ((const IDType *)data))*sizeof(IDType)) < datalen) {
	RegionInstance src_inst = ID((IDType)*idata++).convert<RegionInstance>();
	RegionInstance dst_inst = ID((IDType)*idata++).convert<RegionInstance>();
	InstPair ip(src_inst, dst_inst);

#ifdef USE_CUDA
        // If either one of the instances is in GPU memory increase priority
        if (priority == 0)
        {
          Memory::Impl::MemoryKind src_kind = src_inst.impl()->memory.impl()->kind;
          if (src_kind == Memory::Impl::MKIND_GPUFB)
            priority = 1;
          else
          {
            Memory::Impl::MemoryKind dst_kind = dst_inst.impl()->memory.impl()->kind;
            if (dst_kind == Memory::Impl::MKIND_GPUFB)
              priority = 1;
          }
        }
#endif

	OASVec& oasvec = (*oas_by_inst)[ip];

	unsigned count = *idata++;
	for(unsigned i = 0; i < count; i++) {
	  OffsetsAndSize oas;
	  oas.src_offset = *idata++;
	  oas.dst_offset = *idata++;
	  oas.size = *idata++;
	  oasvec.push_back(oas);
	}
      }

      // better have consumed exactly the right amount of data
      assert(((idata - ((const IDType *)data))*sizeof(IDType)) == datalen);

      log_dma.info("dma request %p deserialized - " IDFMT "[%d]->" IDFMT "[%d]:%d (+%zd) (" IDFMT ") " IDFMT "/%d " IDFMT "/%d",
		   this,
		   oas_by_inst->begin()->first.first.id, 
		   oas_by_inst->begin()->second[0].src_offset,
		   oas_by_inst->begin()->first.second.id, 
		   oas_by_inst->begin()->second[0].dst_offset,
		   oas_by_inst->begin()->second[0].size,
		   oas_by_inst->size() - 1,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);
    }

    CopyRequest::~CopyRequest(void)
    {
      delete oas_by_inst;
    }

    size_t CopyRequest::serialize(void *buffer, size_t maxlen)
    {
      // domain info goes first
      IDType *msgptr = domain.serialize((IDType *)buffer);

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
	}
      }

      size_t msglen = ((const char *)msgptr) - ((const char *)buffer);
      assert(msglen <= maxlen); // TODO: maybe detect _before_ overrunning?

      return msglen;
    }

    void DmaRequest::Waiter::sleep_on_event(Event e, 
					    Reservation l /*= Reservation::NO_RESERVATION*/)
    {
      current_lock = l;
      e.impl()->add_waiter(e, this);
    }

    bool DmaRequest::Waiter::event_triggered(void)
    {
      log_dma.info("request %p triggered in state %d (lock = " IDFMT ")",
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

    void DmaRequest::Waiter::print_info(FILE *f)
    {
      fprintf(f,"dma request %p: after " IDFMT "/%d\n", 
	      req, req->after_copy.id, req->after_copy.gen);
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
	  IndexSpace::Impl *is_impl = domain.get_index_space().impl();
	  if(!is_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no index space metadata yet", this);
	    if(just_check) return false;

	    Event e = is_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - index space metadata invalid - instant trigger", this);
	      is_impl->lock.release();
	    } else {
	      log_dma.info("request %p - index space metadata invalid - sleeping on lock " IDFMT "", this, is_impl->lock.me.id);
	      waiter.sleep_on_event(e, is_impl->lock.me);
	      return false;
	    }
	  }

          // we need more than just the metadata - we also need the valid mask
          {
            Event e = is_impl->request_valid_mask();
            if(!e.has_triggered()) {
              log_dma.info("request %p - valid mask needed for index space " IDFMT " - sleeping on event " IDFMT "/%d", this, domain.get_index_space().id, e.id, e.gen);
	      waiter.sleep_on_event(e);
              return false;
            }
          }
	}

	// now go through all instance pairs
	for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	  RegionInstance::Impl *src_impl = it->first.first.impl();
	  RegionInstance::Impl *dst_impl = it->first.second.impl();

	  if(!src_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no src instance (" IDFMT ") metadata yet", this, it->first.first.id);
	    if(just_check) return false;

	    Event e = src_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - src instance metadata invalid - instant trigger", this);
	      src_impl->lock.release();
	    } else {
	      log_dma.info("request %p - src instance metadata invalid - sleeping on lock " IDFMT "", this, src_impl->lock.me.id);
	      waiter.sleep_on_event(e, src_impl->lock.me);
	      return false;
	    }
	  }
	  if(!src_impl->linearization.valid()) {
	    //printf("deserializing linearizer\n");
	    src_impl->linearization.deserialize(src_impl->locked_data.linearization_bits);
	  }

	  if(!dst_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no dst instance (" IDFMT ") metadata yet", this, it->first.second.id);
	    if(just_check) return false;

	    Event e = dst_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - dst instance metadata invalid - instant trigger", this);
	      dst_impl->lock.release();
	    } else {
	      log_dma.info("request %p - dst instance metadata invalid - sleeping on lock " IDFMT "", this, dst_impl->lock.me.id);
	      waiter.sleep_on_event(e, dst_impl->lock.me);
	      return false;
	    }
	  }
	  if(!dst_impl->linearization.valid()) {
	    //printf("deserializing linearizer\n");
	    dst_impl->linearization.deserialize(dst_impl->locked_data.linearization_bits);
	  }
	}

	// if we got all the way through, we've got all the metadata we need
	state = STATE_BEFORE_EVENT;
      }

      // make sure our functional precondition has occurred
      if(state == STATE_BEFORE_EVENT) {
	// has the before event triggered?  if not, wait on it
	if(before_copy.has_triggered()) {
	  log_dma.info("request %p - before event triggered", this);
	  state = STATE_READY;
	} else {
	  log_dma.info("request %p - before event not triggered", this);
	  if(just_check) return false;

	  log_dma.info("request %p - sleeping on before event", this);
	  waiter.sleep_on_event(before_copy);
	  return false;
	}
      }

      if(state == STATE_READY) {
	log_dma.info("request %p ready", this);
	if(just_check) return true;

	state = STATE_QUEUED;
	assert(rq != 0);
	log_dma.info("request %p enqueued", this);

#ifdef LEGION_LOGGING
	log_timing_event(Processor::NO_PROC, after_copy, COPY_READY);
#endif

	// once we're enqueued, we may be deleted at any time, so no more
	//  references
	rq->enqueue_request(this);
	return true;
      }

      if(state == STATE_QUEUED)
	return true;

      assert(0);
    }

    // defined in lowlevel.cc
    extern unsigned do_remote_write(Memory mem, off_t offset,
				    const void *data, size_t datalen,
				    unsigned sequence_id,
				    Event event, bool make_copy = false);

    extern unsigned do_remote_write(Memory mem, off_t offset,
				    const void *data, size_t datalen,
				    off_t stride, size_t lines,
				    unsigned sequence_id,
				    Event event, bool make_copy = false);
    
    extern unsigned do_remote_write(Memory mem, off_t offset,
				    const SpanList& spans, size_t datalen,
				    unsigned sequence_id,
				    Event event, bool make_copy = false);

    extern unsigned do_remote_reduce(Memory mem, off_t offset,
				     ReductionOpID redop_id, bool red_fold,
				     const void *data, size_t count,
				     off_t src_stride, off_t dst_stride,
				     unsigned sequence_id,
				     Event event, bool make_copy = false);				     

    extern unsigned do_remote_apply_red_list(int node, Memory mem, off_t offset,
					     ReductionOpID redopid,
					     const void *data, size_t datalen,
					     unsigned sequence_id,
					     Event event);

    extern void do_remote_fence(Memory mem, unsigned sequence_id, unsigned count, Event event);

    extern ReductionOpTable reduce_op_table;


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
	GasnetPut(Memory::Impl *_tgt_mem, off_t _tgt_offset,
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
	Memory::Impl *tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
      };

      class GasnetPutBatched {
      public:
	GasnetPutBatched(Memory::Impl *_tgt_mem, off_t _tgt_offset,
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
	GasnetPutReduce(Memory::Impl *_tgt_mem, off_t _tgt_offset,
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
	GasnetPutRedList(Memory::Impl *_tgt_mem, off_t _tgt_offset,
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
					   entry_buffer, pos * redop->sizeof_list_entry, 0, Event::NO_EVENT);
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
					 entry_buffer, pos * redop->sizeof_list_entry, 0, Event::NO_EVENT);
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
		  Memory::Impl *_src_mem, off_t _src_offset,
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
	Memory::Impl *src_mem;
	off_t src_offset;
	size_t elmt_size;
      };

      class GasnetGetBatched {
      public:
	GasnetGetBatched(void *_tgt_ptr,
			 Memory::Impl *_src_mem, off_t _src_offset,
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
	GasnetGetAndPut(Memory::Impl *_tgt_mem, off_t _tgt_offset,
			Memory::Impl *_src_mem, off_t _src_offset,
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
	Memory::Impl *tgt_mem;
	off_t tgt_offset;
	Memory::Impl *src_mem;
	off_t src_offset;
	size_t elmt_size;
	char chunk[CHUNK_SIZE];
      };

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
	    event = Event::Impl::create_event();

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

    }; // namespace RangeExecutors

    class InstPairCopier {
    public:
      InstPairCopier(void) { }
      virtual ~InstPairCopier(void) { }
    public:
      virtual void copy_field(int src_index, int dst_index, int elem_count,
                              unsigned offset_index) = 0;

      virtual void copy_all_fields(int src_index, int dst_index, int elem_count) = 0;

      virtual void copy_all_fields(int src_index, int dst_index, int count_per_line,
				   int src_stride, int dst_stride, int lines)
      {
	// default implementation is just to iterate over lines
	for(int i = 0; i < lines; i++) {
	  copy_all_fields(src_index, dst_index, count_per_line);
	  src_index += src_stride;
	  dst_index += dst_stride;
	}
      }

      virtual void flush(void) = 0;
    };

    // helper function to figure out which field we're in
    static void find_field_start(const int *field_sizes, off_t byte_offset, size_t size, off_t& field_start, int& field_size)
    {
      off_t start = 0;
      for(unsigned i = 0; i < RegionInstance::Impl::MAX_FIELDS_PER_INST; i++) {
	assert(field_sizes[i] > 0);
	if(byte_offset < field_sizes[i]) {
	  assert((int)(byte_offset + size) <= field_sizes[i]);
	  field_start = start;
	  field_size = field_sizes[i];
	  return;
	}
	start += field_sizes[i];
	byte_offset -= field_sizes[i];
      }
      assert(0);
    }

    static inline off_t calc_mem_loc(off_t alloc_offset, off_t field_start, int field_size, int elmt_size,
				     int block_size, int index)
    {
      return (alloc_offset +                                      // start address
	      ((index / block_size) * block_size * elmt_size) +   // full blocks
	      (field_start * block_size) +                        // skip other fields
	      ((index % block_size) * field_size));               // some some of our fields within our block
    }

    static inline int min(int a, int b) { return (a < b) ? a : b; }

    template <typename T>
    class SpanBasedInstPairCopier : public InstPairCopier {
    public:
      // instead of the accessro, we'll grab the implementation pointers
      //  and do address calculation ourselves
      SpanBasedInstPairCopier(T *_span_copier, 
                              RegionInstance _src_inst, 
                              RegionInstance _dst_inst,
                              OASVec &_oas_vec)
	: span_copier(_span_copier), src_inst(_src_inst.impl()), 
          dst_inst(_dst_inst.impl()), oas_vec(_oas_vec)
      {
	StaticAccess<RegionInstance::Impl> src_idata(src_inst);
	assert(src_idata->valid);

	StaticAccess<RegionInstance::Impl> dst_idata(dst_inst);
	assert(dst_idata->valid);

        // Precompute our field offset information
        src_start.resize(oas_vec.size());
        dst_start.resize(oas_vec.size());
        src_size.resize(oas_vec.size());
        dst_size.resize(oas_vec.size());
	partial_field.resize(oas_vec.size());
        for (unsigned idx = 0; idx < oas_vec.size(); idx++)
        {
          find_field_start(src_idata->field_sizes, oas_vec[idx].src_offset,
                            oas_vec[idx].size, src_start[idx], src_size[idx]);
          find_field_start(dst_idata->field_sizes, oas_vec[idx].dst_offset,
                            oas_vec[idx].size, dst_start[idx], dst_size[idx]);

	  // mark an OASVec entry as being "partial" if src and/or dst don't fill the whole instance field
	  partial_field[idx] = ((src_start[idx] != oas_vec[idx].src_offset) ||
				(src_size[idx] != oas_vec[idx].size) ||
				(dst_start[idx] != oas_vec[idx].dst_offset) ||
				(dst_size[idx] != oas_vec[idx].size));
        }
      }

      virtual ~SpanBasedInstPairCopier(void) { }

      virtual void copy_field(int src_index, int dst_index, int elem_count,
                              unsigned offset_index)
      {
        unsigned src_offset = oas_vec[offset_index].src_offset;
        unsigned dst_offset = oas_vec[offset_index].dst_offset;
        unsigned bytes = oas_vec[offset_index].size;
	StaticAccess<RegionInstance::Impl> src_idata(src_inst);
	StaticAccess<RegionInstance::Impl> dst_idata(dst_inst);

	off_t src_field_start, dst_field_start;
	int src_field_size, dst_field_size;

	//find_field_start(src_idata->field_sizes, src_offset, bytes, src_field_start, src_field_size);
	//find_field_start(dst_idata->field_sizes, dst_offset, bytes, dst_field_start, dst_field_size);
        src_field_start = src_start[offset_index];
        dst_field_start = dst_start[offset_index];
        src_field_size = src_size[offset_index];
        dst_field_size = dst_size[offset_index];

	// if both source and dest fill up an entire field, we might be able to copy whole ranges at the same time
	if((src_field_start == src_offset) && (src_field_size == bytes) &&
	   (dst_field_start == dst_offset) && (dst_field_size == bytes)) {
	  // let's see how many we can copy
	  int done = 0;
	  while(done < elem_count) {
	    int src_in_this_block = src_idata->block_size - ((src_index + done) % src_idata->block_size);
	    int dst_in_this_block = dst_idata->block_size - ((dst_index + done) % dst_idata->block_size);
	    int todo = min(elem_count - done, min(src_in_this_block, dst_in_this_block));

	    //printf("copying range of %d elements (%d, %d, %d)\n", todo, src_index, dst_index, done);

	    off_t src_start = calc_mem_loc(src_idata->alloc_offset + (src_offset - src_field_start),
					   src_field_start, src_field_size, src_idata->elmt_size,
					   src_idata->block_size, src_index + done);
	    off_t dst_start = calc_mem_loc(dst_idata->alloc_offset + (dst_offset - dst_field_start),
					   dst_field_start, dst_field_size, dst_idata->elmt_size,
					   dst_idata->block_size, dst_index + done);

	    // sanity check that the range we calculated really is contiguous
	    assert(calc_mem_loc(src_idata->alloc_offset + (src_offset - src_field_start),
				src_field_start, src_field_size, src_idata->elmt_size,
				src_idata->block_size, src_index + done + todo - 1) == 
		   (src_start + (todo - 1) * bytes));
	    assert(calc_mem_loc(dst_idata->alloc_offset + (dst_offset - dst_field_start),
				dst_field_start, dst_field_size, dst_idata->elmt_size,
				dst_idata->block_size, dst_index + done + todo - 1) == 
		   (dst_start + (todo - 1) * bytes));

#ifdef NEW2D_DEBUG
	    printf("ZZZ: %zd %zd %d\n", src_start, dst_start, bytes * todo);
#endif
	    span_copier->copy_span(src_start, dst_start, bytes * todo);
	    //src_mem->get_bytes(src_start, buffer, bytes * todo);
	    //dst_mem->put_bytes(dst_start, buffer, bytes * todo);

	    done += todo;
	  }
	} else {
	  // fallback - calculate each address separately
	  for(int i = 0; i < elem_count; i++) {
	    off_t src_start = calc_mem_loc(src_idata->alloc_offset + (src_offset - src_field_start),
					   src_field_start, src_field_size, src_idata->elmt_size,
					   src_idata->block_size, src_index + i);
	    off_t dst_start = calc_mem_loc(dst_idata->alloc_offset + (dst_offset - dst_field_start),
					   dst_field_start, dst_field_size, dst_idata->elmt_size,
					   dst_idata->block_size, dst_index + i);

#ifdef NEW2D_DEBUG
	    printf("ZZZ: %zd %zd %d\n", src_start, dst_start, bytes);
#endif
	    span_copier->copy_span(src_start, dst_start, bytes);
	    //src_mem->get_bytes(src_start, buffer, bytes);
	    //dst_mem->put_bytes(dst_start, buffer, bytes);
	  }
	}
      }

      virtual void copy_all_fields(int src_index, int dst_index, int elem_count)
      {
	// first check - if the span we're copying straddles a block boundary
	//  go back to old way - block size of 1 is ok only if both are
	StaticAccess<RegionInstance::Impl> src_idata(src_inst);
	StaticAccess<RegionInstance::Impl> dst_idata(dst_inst);

	size_t src_bsize = src_idata->block_size;
	size_t dst_bsize = dst_idata->block_size;

	if(((src_bsize == 1) != (dst_bsize == 1)) ||
	   ((src_bsize > 1) && ((src_index / src_bsize) != ((src_index + elem_count - 1) / src_bsize))) ||
	   ((dst_bsize > 1) && ((dst_index / dst_bsize) != ((dst_index + elem_count - 1) / dst_bsize)))) {
	  printf("copy straddles block boundaries - falling back\n");
	  for(unsigned i = 0; i < oas_vec.size(); i++)
	    copy_field(src_index, dst_index, elem_count, i);
	  return;
	}

	// start with the first field, grabbing as many at a time as we can

	unsigned field_idx = 0;

	while(field_idx < oas_vec.size()) {
	  // get information about the first field
	  unsigned src_offset = oas_vec[field_idx].src_offset;
	  unsigned dst_offset = oas_vec[field_idx].dst_offset;
	  unsigned bytes = oas_vec[field_idx].size;

	  // if src and/or dst aren't a full field, fall back to the old way for this field
	  int src_field_start = src_start[field_idx];
	  int src_field_size = src_size[field_idx];
	  int dst_field_start = dst_start[field_idx];
	  int dst_field_size = dst_size[field_idx];

	  if(partial_field[field_idx]) {
	    printf("not a full field - falling back\n");
	    copy_field(src_index, dst_index, elem_count, field_idx);
	    field_idx++;
	    continue;
	  }

	  // see if we can tack on more fields
	  unsigned field_idx2 = field_idx + 1;
	  int src_fstride = 0;
	  int dst_fstride = 0;
	  unsigned total_bytes = bytes;
	  unsigned total_lines = 1;
	  while(field_idx2 < oas_vec.size()) {
	    // TODO: for now, don't merge fields here because it can result in too-large copies
	    break;

	    // is this a partial field?  if so, stop
	    if(partial_field[field_idx2])
	      break;

	    unsigned src_offset2 = oas_vec[field_idx2].src_offset;
	    unsigned dst_offset2 = oas_vec[field_idx2].dst_offset;

	    // test depends on AOS (bsize == 1) vs (hybrid)SOA (bsize > 1)
	    if(src_bsize == 1) {
	      // for AOS, we need this field's offset to be the next byte
	      if((src_offset2 != (src_offset + total_bytes)) ||
		 (dst_offset2 != (dst_offset + total_bytes)))
		break;

	      // if tests pass, add this field's size to our total and keep going
	      total_bytes += oas_vec[field_idx2].size;
	    } else {
	      // in SOA, we need the field's strides to match, but non-contiguous is ok
	      // first stride will be ok by construction
	      int src_fstride2 = src_offset2 - src_offset;
	      int dst_fstride2 = dst_offset2 - dst_offset;
	      if(src_fstride == 0) src_fstride = src_fstride2;
	      if(dst_fstride == 0) dst_fstride = dst_fstride2;
	      if((src_fstride2 != (field_idx2 - field_idx) * src_fstride) ||
		 (dst_fstride2 != (field_idx2 - field_idx) * dst_fstride))
		break;

	      // if tests pass, we have another line
	      total_lines++;
	    }

	    field_idx2++;
	  }

	  // now we can copy something
	  off_t src_start = calc_mem_loc(src_idata->alloc_offset + (src_offset - src_field_start),
					 src_field_start, src_field_size, src_idata->elmt_size,
					 src_idata->block_size, src_index);
	  off_t dst_start = calc_mem_loc(dst_idata->alloc_offset + (dst_offset - dst_field_start),
					 dst_field_start, dst_field_size, dst_idata->elmt_size,
					 dst_idata->block_size, dst_index);

	  // AOS merging doesn't work if we don't end up with the full element
	  if((src_bsize == 1) && 
	     ((total_bytes < src_idata->elmt_size) || (total_bytes < dst_idata->elmt_size)) &&
	     (elem_count > 1)) {
	    printf("help: AOS tried to merge subset of fields with multiple elements - not contiguous!\n");
	    assert(0);
	  }

#ifdef NEW2D_DEBUG
	  printf("AAA: %d %d %d, %d-%d -> %zd %zd %d, %zd %zd %d\n",
		 src_index, dst_index, elem_count, field_idx, field_idx2-1,
		 src_start, dst_start, elem_count * total_bytes,
		 src_fstride * src_bsize,
		 dst_fstride * dst_bsize,
		 total_lines);
#endif
	  span_copier->copy_span(src_start, dst_start, elem_count * total_bytes,
				 src_fstride * src_bsize,
				 dst_fstride * dst_bsize,
				 total_lines);

	  // continue with the first field we couldn't take for this pass
	  field_idx = field_idx2;
	}
      }

      virtual void copy_all_fields(int src_index, int dst_index, int count_per_line,
				   int src_stride, int dst_stride, int lines)
      {
	// first check - if the span we're copying straddles a block boundary
	//  go back to old way - block size of 1 is ok only if both are
	StaticAccess<RegionInstance::Impl> src_idata(src_inst);
	StaticAccess<RegionInstance::Impl> dst_idata(dst_inst);

	size_t src_bsize = src_idata->block_size;
	size_t dst_bsize = dst_idata->block_size;

	int src_last = src_index + (count_per_line - 1) + (lines - 1) * src_stride;
	int dst_last = dst_index + (count_per_line - 1) + (lines - 1) * dst_stride;

	if(((src_bsize == 1) != (dst_bsize == 1)) ||
	   ((src_bsize > 1) && ((src_index / src_bsize) != (src_last / src_bsize))) ||
	   ((dst_bsize > 1) && ((dst_index / dst_bsize) != (dst_last / dst_bsize)))) {
	  printf("copy straddles block boundaries - falling back\n");
	  for(unsigned i = 0; i < oas_vec.size(); i++)
	    for(int l = 0; l < lines; l++)
	      copy_field(src_index + l * src_stride, 
			 dst_index + l * dst_stride, count_per_line, i);
	  return;
	}

	// start with the first field, grabbing as many at a time as we can

	unsigned field_idx = 0;

	while(field_idx < oas_vec.size()) {
	  // get information about the first field
	  unsigned src_offset = oas_vec[field_idx].src_offset;
	  unsigned dst_offset = oas_vec[field_idx].dst_offset;
	  unsigned bytes = oas_vec[field_idx].size;

	  // if src and/or dst aren't a full field, fall back to the old way for this field
	  int src_field_start = src_start[field_idx];
	  int src_field_size = src_size[field_idx];
	  int dst_field_start = dst_start[field_idx];
	  int dst_field_size = dst_size[field_idx];

	  if(partial_field[field_idx]) {
	    printf("not a full field - falling back\n");
	    copy_field(src_index, dst_index, count_per_line, field_idx);
	    field_idx++;
	    continue;
	  }

	  // see if we can tack on more fields
	  unsigned field_idx2 = field_idx + 1;
	  int src_fstride = 0;
	  int dst_fstride = 0;
	  unsigned total_bytes = bytes;
	  unsigned total_lines = 1;
	  while(field_idx2 < oas_vec.size()) {
	    // is this a partial field?  if so, stop
	    if(partial_field[field_idx2])
	      break;

	    unsigned src_offset2 = oas_vec[field_idx2].src_offset;
	    unsigned dst_offset2 = oas_vec[field_idx2].dst_offset;

	    // test depends on AOS (bsize == 1) vs (hybrid)SOA (bsize > 1)
	    if(src_bsize == 1) {
	      // for AOS, we need this field's offset to be the next byte
	      if((src_offset2 != (src_offset + total_bytes)) ||
		 (dst_offset2 != (dst_offset + total_bytes)))
		break;

	      // if tests pass, add this field's size to our total and keep going
	      total_bytes += oas_vec[field_idx2].size;
	    } else {
	      // in SOA, we need the field's strides to match, but non-contiguous is ok
	      // first stride will be ok by construction
	      int src_fstride2 = src_offset2 - src_offset;
	      int dst_fstride2 = dst_offset2 - dst_offset;
	      if(src_fstride == 0) src_fstride = src_fstride2;
	      if(dst_fstride == 0) dst_fstride = dst_fstride2;
	      if((src_fstride2 != (field_idx2 - field_idx) * src_fstride) ||
		 (dst_fstride2 != (field_idx2 - field_idx) * dst_fstride))
		break;

	      // if tests pass, we have another line
	      total_lines++;
	    }

	    field_idx2++;
	  }

	  // now we can copy something
	  off_t src_start = calc_mem_loc(src_idata->alloc_offset + (src_offset - src_field_start),
					 src_field_start, src_field_size, src_idata->elmt_size,
					 src_idata->block_size, src_index);
	  off_t dst_start = calc_mem_loc(dst_idata->alloc_offset + (dst_offset - dst_field_start),
					 dst_field_start, dst_field_size, dst_idata->elmt_size,
					 dst_idata->block_size, dst_index);

	  // AOS merging doesn't work if we don't end up with the full element
	  if((src_bsize == 1) && 
	     ((total_bytes < src_idata->elmt_size) || (total_bytes < dst_idata->elmt_size)) &&
	     (count_per_line > 1)) {
	    printf("help: AOS tried to merge subset of fields with multiple elements - not contiguous!\n");
	    assert(0);
	  }

#ifdef NEW2D_DEBUG
	  printf("BBB: %d %d %d, %d %d %d, %d-%d -> %zd %zd %d, %zd %zd %d\n",
		 src_index, dst_index, count_per_line,
		 src_stride, dst_stride, lines,
		 field_idx, field_idx2-1,
		 src_start, dst_start, count_per_line * total_bytes,
		 src_fstride * src_bsize,
		 dst_fstride * dst_bsize,
		 lines * total_lines);
#endif

	  // since we're already 2D, we need line strides to match up
	  if(total_lines > 1) {
	    if(0) {
	    } else {
	      // no?  punt on the field merging
	      total_lines = 1;
	      //printf("CCC: eliminating field merging\n");
	      field_idx2 = field_idx + 1;
	    }
	  }

#ifdef NEW2D_DEBUG
	  printf("DDD: %d %d %d, %d %d %d, %d-%d -> %zd %zd %d, %zd %zd %d\n",
		 src_index, dst_index, count_per_line,
		 src_stride, dst_stride, lines,
		 field_idx, field_idx2-1,
		 src_start, dst_start, count_per_line * total_bytes,
		 src_stride * bytes,
		 dst_stride * bytes,
		 lines);
#endif
	  span_copier->copy_span(src_start, dst_start, count_per_line * total_bytes,
				 src_stride * bytes,
				 dst_stride * bytes,
				 lines);

	  // continue with the first field we couldn't take for this pass
	  field_idx = field_idx2;
	}
      }

      virtual void flush(void) {}

    protected:
      T *span_copier;
      RegionInstance::Impl *src_inst;
      RegionInstance::Impl *dst_inst;
      OASVec &oas_vec;
      std::vector<off_t> src_start;
      std::vector<off_t> dst_start;
      std::vector<int> src_size;
      std::vector<int> dst_size;
      std::vector<bool> partial_field;
    };

    class RemoteWriteInstPairCopier : public InstPairCopier {
    public:
      RemoteWriteInstPairCopier(RegionInstance src_inst, RegionInstance dst_inst,
                                OASVec &_oas_vec)
	: src_acc(src_inst.get_accessor()), dst_acc(dst_inst.get_accessor()),
          oas_vec(_oas_vec)
      {}

      virtual ~RemoteWriteInstPairCopier(void) { }

      virtual void copy_field(int src_index, int dst_index, int elem_count,
                              unsigned offset_index)
      {
        unsigned src_offset = oas_vec[offset_index].src_offset;
        unsigned dst_offset = oas_vec[offset_index].dst_offset;
        unsigned bytes = oas_vec[offset_index].size;
	char buffer[1024];

	for(int i = 0; i < elem_count; i++) {
	  src_acc.read_untyped(ptr_t(src_index + i), buffer, bytes, src_offset);
          if(0 && i == 0) {
            printf("remote write: (%d:%d->%d:%d) %d bytes:",
                   src_index + i, src_offset, dst_index + i, dst_offset, bytes);
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

    class MemPairCopier {
    public:
      static MemPairCopier* create_copier(Memory src_mem, Memory dst_mem,
					  const ReductionOpUntyped *redop = 0,
					  bool fold = false);
      MemPairCopier(void) { }
      virtual ~MemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec) = 0;

      // default behavior of flush is just to trigger the event, if it exists
      virtual void flush(Event after_copy)
      {
	if(after_copy.exists())
	  after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
      }
    };

    class BufferedMemPairCopier : public MemPairCopier {
    public:
      BufferedMemPairCopier(Memory _src_mem, Memory _dst_mem, size_t _buffer_size = 32768)
	: buffer_size(_buffer_size)
      {
	src_mem = _src_mem.impl();
	dst_mem = _dst_mem.impl();
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
	}
	if(bytes > 0) {
	  src_mem->get_bytes(src_offset, buffer, bytes);
	  dst_mem->put_bytes(dst_offset, buffer, bytes);
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
      Memory::Impl *src_mem, *dst_mem;
      char *buffer;
    };
     
    class MemcpyMemPairCopier : public MemPairCopier {
    public:
      MemcpyMemPairCopier(Memory _src_mem, Memory _dst_mem)
      {
	Memory::Impl *src_impl = _src_mem.impl();
	src_base = (const char *)(src_impl->get_direct_ptr(0, src_impl->size));
	assert(src_base);

	Memory::Impl *dst_impl = _dst_mem.impl();
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
				  const ReductionOpUntyped *_redop, bool _fold)
      {
	Memory::Impl *src_impl = _src_mem.impl();
	src_base = (const char *)(src_impl->get_direct_ptr(0, src_impl->size));
	assert(src_base);

	Memory::Impl *dst_impl = _dst_mem.impl();
	dst_base = (char *)(dst_impl->get_direct_ptr(0, dst_impl->size));
	assert(dst_base);

	redop = _redop;
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
				     const ReductionOpUntyped *_redop, bool _fold,
				     size_t _buffer_size = 1024) // in elements
	: buffer_size(_buffer_size)
      {
	src_mem = _src_mem.impl();
	dst_mem = _dst_mem.impl();
	redop = _redop;
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
      Memory::Impl *src_mem, *dst_mem;
      char *src_buffer;
      char *dst_buffer;
      const ReductionOpUntyped *redop;
      bool fold;
    };
     
    // a MemPairCopier that keeps a list of events for component copies and doesn't trigger
    //  the completion event until they're all done
    class DelayedMemPairCopier : public MemPairCopier {
    public:
      DelayedMemPairCopier(void) {}

      virtual ~DelayedMemPairCopier(void) {}
      
      virtual void flush(Event after_copy)
      {
	// create a merged event for all the copies and used that to perform a delayed trigger
	//  of the after_copy event (if it exists)
	if(after_copy.exists()) {
	  if(events.size() > 0) {
	    Event merged = Event::merge_events(events);

	    // deferred trigger based on this merged event
	    after_copy.impl()->trigger(after_copy.gen, gasnet_mynode(), merged);
	  } else {
	    // no actual copies occurred, so manually trigger event ourselves
	    after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
	  }
	} else {
	  if(events.size() > 0) {
	    // we don't have an event we can use to signal completion, so wait on the events ourself
	    for(std::set<Event>::const_iterator it = events.begin();
		it != events.end();
		it++)
	      (*it).wait(true);
	  } else {
	    // nothing happened and we don't need to tell anyone - life is simple
	  }
	}
      }

    protected:
      std::set<Event> events;
    };

#ifdef USE_CUDA     
    class GPUtoFBMemPairCopier : public DelayedMemPairCopier {
    public:
      GPUtoFBMemPairCopier(Memory _src_mem, GPUProcessor *_gpu)
	: gpu(_gpu)
      {
	Memory::Impl *src_impl = _src_mem.impl();
	src_base = (const char *)(src_impl->get_direct_ptr(0, src_impl->size));
	assert(src_base);
      }

      virtual ~GPUtoFBMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<GPUtoFBMemPairCopier>(this, src_inst, 
                                                          dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	Event e = Event::Impl::create_event();
	//printf("gpu write of %zd bytes\n", bytes);
	gpu->copy_to_fb(dst_offset, src_base + src_offset, bytes, Event::NO_EVENT, e);
	events.insert(e);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
        Event e = Event::Impl::create_event();
        gpu->copy_to_fb_2d(dst_offset, src_base + src_offset,
                           dst_stride, src_stride, bytes, lines,
                           Event::NO_EVENT, e);
        events.insert(e);
      }

    protected:
      const char *src_base;
      GPUProcessor *gpu;
    };

    class GPUfromFBMemPairCopier : public DelayedMemPairCopier {
    public:
      GPUfromFBMemPairCopier(GPUProcessor *_gpu, Memory _dst_mem)
	: gpu(_gpu)
      {
	Memory::Impl *dst_impl = _dst_mem.impl();
	dst_base = (char *)(dst_impl->get_direct_ptr(0, dst_impl->size));
	assert(dst_base);
      }

      virtual ~GPUfromFBMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<GPUfromFBMemPairCopier>(this, src_inst, 
                                                            dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	Event e = Event::Impl::create_event();
	//printf("gpu read of %zd bytes\n", bytes);
	gpu->copy_from_fb(dst_base + dst_offset, src_offset, bytes, Event::NO_EVENT, e);
        events.insert(e);
	//e.wait(true);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
        Event e = Event::Impl::create_event();
        gpu->copy_from_fb_2d(dst_base + dst_offset, src_offset,
                             dst_stride, src_stride, bytes, lines,
                             Event::NO_EVENT, e);
        events.insert(e);
      }

    protected:
      char *dst_base;
      GPUProcessor *gpu;
    };
     
    class GPUinFBMemPairCopier : public DelayedMemPairCopier {
    public:
      GPUinFBMemPairCopier(GPUProcessor *_gpu)
	: gpu(_gpu)
      {
      }

      virtual ~GPUinFBMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
	return new SpanBasedInstPairCopier<GPUinFBMemPairCopier>(this, src_inst, 
                                                            dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	Event e = Event::Impl::create_event();
	//printf("gpu write of %zd bytes\n", bytes);
	gpu->copy_within_fb(dst_offset, src_offset, bytes, Event::NO_EVENT, e);
	//e.wait(true);
        events.insert(e);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     off_t src_stride, off_t dst_stride, size_t lines)
      {
        Event e = Event::Impl::create_event();
        gpu->copy_within_fb_2d(dst_offset, src_offset,
                               dst_stride, src_stride, bytes, lines,
                               Event::NO_EVENT, e);
        events.insert(e);
      }

    protected:
      GPUProcessor *gpu;
    };

    class GPUPeerMemPairCopier : public DelayedMemPairCopier {
    public:
      GPUPeerMemPairCopier(GPUProcessor *_src, GPUProcessor *_dst)
        : src(_src), dst(_dst)
      {
      }

      virtual ~GPUPeerMemPairCopier(void) { }

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
                                        OASVec &oas_vec)
      {
        return new SpanBasedInstPairCopier<GPUPeerMemPairCopier>(this, src_inst,
                                                              dst_inst, oas_vec);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
        Event e = Event::Impl::create_event();
        src->copy_to_peer(dst, dst_offset, src_offset, bytes, Event::NO_EVENT, e);
        events.insert(e);
      }

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
                     off_t src_stride, off_t dst_stride, size_t lines)
      {
        Event e = Event::Impl::create_event();
        src->copy_to_peer_2d(dst, dst_offset, src_offset,
                             dst_stride, src_stride, bytes, lines,
                             Event::NO_EVENT, e);
        events.insert(e);
      }
    protected:
      GPUProcessor *src, *dst;
    };
#endif
     
    static unsigned rdma_sequence_no = 1;

    class RemoteWriteMemPairCopier : public MemPairCopier {
    public:
      RemoteWriteMemPairCopier(Memory _src_mem, Memory _dst_mem)
      {
	src_mem = _src_mem.impl();
	src_base = (const char *)(src_mem->get_direct_ptr(0, src_mem->size));
	assert(src_base);

	dst_mem = _dst_mem.impl();
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

	if(bytes >= TARGET_XFER_SIZE) {
	  // large enough that we can transfer it by itself
#ifdef DEBUG_REMOTE_WRITES
	  printf("remote write of %zd bytes (" IDFMT ":%zd -> " IDFMT ":%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#endif
	  num_writes += do_remote_write(dst_mem->me, dst_offset, src_base + src_offset, bytes, 
					sequence_id, Event::NO_EVENT, false /* no copy */);
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
	  if(g->dst_size >= TARGET_XFER_SIZE) {
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
	if((src_stride == bytes) && (dst_stride == bytes)) {
	  copy_span(src_offset, dst_offset, bytes * lines);
	  return;
	}

	// case 2: if dst coalesces, we'll let the RDMA code do the gather for us -
	//  this won't merge with any other copies though
	if(dst_stride == bytes) {
#ifdef NEW2D_DEBUG
	  printf("GATHER copy\n");
#endif
	  num_writes += do_remote_write(dst_mem->me, dst_offset,
					src_base + src_offset, bytes, src_stride, lines,
					sequence_id, Event::NO_EVENT, false /* no copy */);
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

      void copy_gather(const PendingGather *g, Event trigger = Event::NO_EVENT)
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
					sequence_id, Event::NO_EVENT, false /* no copy */);
	  if(trigger.exists())
	    do_remote_fence(dst_mem->me, sequence_id, num_writes, trigger);
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
				      sequence_id, Event::NO_EVENT, false /* no copy - data won't change til copy event triggers */);
	if(trigger.exists())
	  do_remote_fence(dst_mem->me, sequence_id, num_writes, trigger);
#ifdef TIME_REMOTE_WRITES
        unsigned long long stop = TimeStamp::get_current_time_in_micros();
        gather_time += (stop - start);
#endif
      }

      virtual void flush(Event after_copy)
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
	if(gathers.size() > 0) {
	  // push out all but the last one with events triggered
	  while(gathers.size() > 1) {
	    std::map<off_t, PendingGather *>::iterator it = gathers.begin();
	    PendingGather *g = it->second;
	    gathers.erase(it);

	    copy_gather(g);
	    delete g;
	  }

	  // for the last one, give it the 'after_copy' event to trigger on arrival
	  std::map<off_t, PendingGather *>::iterator it = gathers.begin();
	  PendingGather *g = it->second;
	  gathers.erase(it);

#define NO_SEPARATE_FENCE
#ifndef SEPARATE_FENCE
	  copy_gather(g, after_copy);
#else
	  copy_gather(g);
#ifdef DEBUG_REMOTE_WRITES
	  printf("remote write fence: " IDFMT "/%d\n", after_copy.id, after_copy.gen);
#endif
	  do_remote_write(dst_mem->me, 0, 0, 0, after_copy);
#endif
	  delete g;
	} else {
	  // didn't have any pending copies, but if we have an event to trigger, send that
	  //  to the remote side as a zero-size copy to push the data in front of it
	  if(after_copy.exists()) {
            if(num_writes == 0) {
              // an empty dma - no need to send a fence - we can trigger the
              //  completion event here and save a message
              after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
            } else {
#ifdef DEBUG_REMOTE_WRITES
	      printf("remote write fence: " IDFMT "/%d\n", after_copy.id, after_copy.gen);
#endif
	      //do_remote_write(dst_mem->me, 0, 0, 0, after_copy);
	      do_remote_fence(dst_mem->me, sequence_id, num_writes, after_copy);
            }
	  }
	}
#ifdef TIME_REMOTE_WRITES
        unsigned long long stop = TimeStamp::get_current_time_in_micros();
        gather_time += (stop - start);
        printf("Remote Write: span time: %lld  gather time %lld "
                "total gathers %ld total spans %d\n", 
                span_time, gather_time, total_gathers, total_spans);
#endif
      }

    protected:
      Memory::Impl *src_mem, *dst_mem;
      const char *src_base;
      std::map<off_t, PendingGather *> gathers;
#ifdef TIME_REMOTE_WRITES
      unsigned long long span_time;
      unsigned long long gather_time;
#endif
      unsigned sequence_id, num_writes;
    };
     
    MemPairCopier *MemPairCopier::create_copier(Memory src_mem, Memory dst_mem,
						const ReductionOpUntyped *redop /*= 0*/,
						bool fold /*= false*/)
    {
      Memory::Impl *src_impl = src_mem.impl();
      Memory::Impl *dst_impl = dst_mem.impl();

      Memory::Impl::MemoryKind src_kind = src_impl->kind;
      Memory::Impl::MemoryKind dst_kind = dst_impl->kind;

      log_dma.info("copier: " IDFMT "(%d) -> " IDFMT "(%d)", src_mem.id, src_kind, dst_mem.id, dst_kind);

      if(redop == 0) {
	// can we perform simple memcpy's?
	if(((src_kind == Memory::Impl::MKIND_SYSMEM) || (src_kind == Memory::Impl::MKIND_ZEROCOPY)) &&
	   ((dst_kind == Memory::Impl::MKIND_SYSMEM) || (dst_kind == Memory::Impl::MKIND_ZEROCOPY))) {
	  return new MemcpyMemPairCopier(src_mem, dst_mem);
	}

#ifdef USE_CUDA
	// copy to a framebuffer
	if(((src_kind == Memory::Impl::MKIND_SYSMEM) || (src_kind == Memory::Impl::MKIND_ZEROCOPY)) &&
	   (dst_kind == Memory::Impl::MKIND_GPUFB)) {
	  GPUProcessor *dst_gpu = ((GPUFBMemory *)dst_impl)->gpu;
	  return new GPUtoFBMemPairCopier(src_mem, dst_gpu);
	}

	// copy from a framebuffer
	if((src_kind == Memory::Impl::MKIND_GPUFB) &&
	   ((dst_kind == Memory::Impl::MKIND_SYSMEM) || (dst_kind == Memory::Impl::MKIND_ZEROCOPY))) {
	  GPUProcessor *src_gpu = ((GPUFBMemory *)src_impl)->gpu;
	  return new GPUfromFBMemPairCopier(src_gpu, dst_mem);
	}

	// copy within a framebuffer
	if((src_kind == Memory::Impl::MKIND_GPUFB) &&
	   (dst_kind == Memory::Impl::MKIND_GPUFB)) {
	  GPUProcessor *src_gpu = ((GPUFBMemory *)src_impl)->gpu;
	  GPUProcessor *dst_gpu = ((GPUFBMemory *)dst_impl)->gpu;
	  if (src_gpu == dst_gpu)
	    return new GPUinFBMemPairCopier(src_gpu);
	  else if (src_gpu->can_access_peer(dst_gpu))
	    return new GPUPeerMemPairCopier(src_gpu, dst_gpu);
	  else
	    {
	      fprintf(stderr,"TIME FOR SEAN TO IMPLEMENT MULTI-HOP COPIES!\n");
	      assert(false);
	      return NULL;
	    }
	}
#endif

	// try as many things as we can think of
	if((dst_kind == Memory::Impl::MKIND_REMOTE) ||
	   (dst_kind == Memory::Impl::MKIND_RDMA)) {
	  assert(src_kind != Memory::Impl::MKIND_REMOTE);
	  return new RemoteWriteMemPairCopier(src_mem, dst_mem);
	}

	// fallback
	return new BufferedMemPairCopier(src_mem, dst_mem);
      } else {
	// reduction case
	// can we perform simple memcpy's?
	if(((src_kind == Memory::Impl::MKIND_SYSMEM) || (src_kind == Memory::Impl::MKIND_ZEROCOPY)) &&
	   ((dst_kind == Memory::Impl::MKIND_SYSMEM) || (dst_kind == Memory::Impl::MKIND_ZEROCOPY))) {
	  return new LocalReductionMemPairCopier(src_mem, dst_mem, redop, fold);
	}

	// fallback is pretty damn slow
	log_dma.warning("using a buffering copier for reductions (%x -> %x)",
			src_mem.id, dst_mem.id);
	return new BufferedReductionMemPairCopier(src_mem, dst_mem, redop, fold);
      }
    }

    template <unsigned DIM>
    static unsigned compress_strides(const Rect<DIM> &r,
				     const Point<1> in1[DIM], const Point<1> in2[DIM],
				     Point<DIM>& extents,
				     Point<1> out1[DIM], Point<1> out2[DIM])
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
      unsigned exp1, exp2;

      // now go through dimensions, collapsing each if it matches the expected stride for
      //  both sets (can't collapse for first)
      for(unsigned i = 0; i < DIM; i++) {
	unsigned d = stride_order[i];
	unsigned e = r.dim_size(d);
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
      IndexSpace::Impl *ispace = domain.get_index_space().impl();
      assert(ispace->valid_mask_complete);

      // this is the SOA-friendly loop nesting
      for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	RegionInstance src_inst = it->first.first;
	RegionInstance dst_inst = it->first.second;
	OASVec& oasvec = it->second;

	InstPairCopier *ipc = mpc->inst_pair(src_inst, dst_inst, oasvec);

	// index space instances use 1D linearizations for translation
	Arrays::Mapping<1, 1> *src_linearization = src_inst.impl()->linearization.get_mapping<1>();
	Arrays::Mapping<1, 1> *dst_linearization = dst_inst.impl()->linearization.get_mapping<1>();

	// does the destination instance space's index space match what we're copying?  if so,
	//  it's ok to copy extra elements (to decrease sparsity) because they're unused in
	//  the destination
	StaticAccess<RegionInstance::Impl> dst_idata(dst_inst.impl(), true /*must be valid already*/);
	int rlen_target;
	if(ispace->me == dst_idata->is) {
	  rlen_target = 32768 / 4; // aim for ~32KB transfers at least
	} else {
	  rlen_target = 1;
	}
	
	ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	int rstart, rlen;
	while(e->get_next(rstart, rlen)) {
	  // do we want to copy extra elements to fill in some holes?
	  while(rlen < rlen_target) {
	    // see where the next valid elements are
	    int rstart2, rlen2;
	    // if none, stop
	    if(!e->peek_next(rstart2, rlen2)) break;
	    // or if they don't even start until outside the window, stop
	    if(rstart2 > (rstart + rlen_target)) break;
	    // ok, include the next valid element(s) and any invalid ones in between
	    //printf("bloating from %d to %d\n", rlen, rstart2 + rlen2 - rstart);
	    rlen = rstart2 + rlen2 - rstart;
	    // and actually take the next range from the enumerator
	    e->get_next(rstart2, rlen2);
	  }

	  int sstart = src_linearization->image(rstart);
	  int dstart = dst_linearization->image(rstart);
#ifdef DEBUG_LOW_LEVEL
	  assert(src_linearization->image_is_dense(Rect<1>(rstart, rstart + rlen - 1)));
	  assert(dst_linearization->image_is_dense(Rect<1>(rstart, rstart + rlen - 1)));
#endif
	  //printf("X: %d+%d %d %d\n", rstart, rlen, sstart, dstart);

	  for (unsigned idx = 0; idx < oasvec.size(); idx++)
	    ipc->copy_field(sstart, dstart, rlen, idx);
	}
        delete e;
	delete ipc;
      }
    }

    template <unsigned DIM>
    void CopyRequest::perform_dma_rect(MemPairCopier *mpc)
    {
      Arrays::Rect<DIM> orig_rect = domain.get_rect<DIM>();

      // this is the SOA-friendly loop nesting
      for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	RegionInstance src_inst = it->first.first;
	RegionInstance dst_inst = it->first.second;
	OASVec& oasvec = it->second;

	InstPairCopier *ipc = mpc->inst_pair(src_inst, dst_inst, oasvec);

	Arrays::Mapping<DIM, 1> *src_linearization = src_inst.impl()->linearization.get_mapping<DIM>();
	Arrays::Mapping<DIM, 1> *dst_linearization = dst_inst.impl()->linearization.get_mapping<DIM>();

	// see what linear subrects we can get - again, iterate over destination first for gathering
	for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lso(orig_rect, *dst_linearization); lso; lso++) {
	  for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lsi(lso.subrect, *src_linearization); lsi; lsi++) {
	    // see if we can compress the strides for a more efficient copy
	    Point<1> src_cstrides[DIM], dst_cstrides[DIM];
	    Point<DIM> extents;
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
		Rect<1> irect = dsi.image;
		// rectangle in output must be recalculated
		Rect<DIM> subrect_check;
		Rect<1> orect = dst_linearization->image_dense_subrect(dsi.subrect, subrect_check);
		assert(dsi.subrect == subrect_check);

		//for(OASVec::iterator it2 = oasvec.begin(); it2 != oasvec.end(); it2++)
		for (unsigned idx = 0; idx < oasvec.size(); idx++)
		  ipc->copy_field(irect.lo, orect.lo, irect.hi - irect.lo + 1, idx);
		//it2->src_offset, it2->dst_offset, it2->size);
	      }
	    }
	  }
	}

        // Dammit Sean stop leaking things!
        delete ipc;
      }
    }

#ifdef LEGION_LOGGING
    class CopyCompletionLogger : public Event::Impl::EventWaiter {
    public:
      CopyCompletionLogger(Event _event) : event(_event) {}

      virtual ~CopyCompletionLogger(void) { }

      virtual bool event_triggered(void)
      {
	log_timing_event(Processor::NO_PROC, event, COPY_END);
	return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"copy completion logger - " IDFMT "/%d\n", event.id, event.gen);
      }

    protected:
      Event event;
    };
#endif

    void CopyRequest::perform_dma(void)
    {
      log_dma.info("request %p executing", this);

#ifdef LEGION_LOGGING
      log_timing_event(Processor::NO_PROC, after_copy, COPY_BEGIN);

      // the copy might not actually finish in this thread, so set up an event waiter
      //  to log the completion
      after_copy.impl()->add_waiter(after_copy, new CopyCompletionLogger(after_copy));
#endif

      DetailedTimer::ScopedPush sp(TIME_COPY);

      // create a copier for the memory used by all of these instance pairs
      Memory src_mem = oas_by_inst->begin()->first.first.impl()->memory;
      Memory dst_mem = oas_by_inst->begin()->first.second.impl()->memory;

      MemPairCopier *mpc = MemPairCopier::create_copier(src_mem, dst_mem);

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

      log_dma.info("dma request %p finished - " IDFMT "[%d]->" IDFMT "[%d]:%d (+%zd) (" IDFMT ") " IDFMT "/%d " IDFMT "/%d",
		   this,
		   oas_by_inst->begin()->first.first.id, 
		   oas_by_inst->begin()->second[0].src_offset,
		   oas_by_inst->begin()->first.second.id, 
		   oas_by_inst->begin()->second[0].dst_offset,
		   oas_by_inst->begin()->second[0].size,
		   oas_by_inst->size() - 1,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);

      // if(after_copy.exists())
      // 	after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());

      mpc->flush(after_copy);
      delete mpc;

#ifdef EVEN_MORE_DEAD_DMA_CODE
      RegionInstance::Impl *src_impl = src.impl();
      RegionInstance::Impl *tgt_impl = target.impl();

      // we should have already arranged to have access to this data, so
      //  assert if we don't
      StaticAccess<RegionInstance::Impl> src_data(src_impl, true);
      StaticAccess<RegionInstance::Impl> tgt_data(tgt_impl, true);

      // code path for copies to/from reduction-only instances not done yet
      // are we doing a reduction?
      const ReductionOpUntyped *redop = ((src_data->redopid >= 0) ?
					   reduce_op_table[src_data->redopid] :
					   0);
      bool red_fold = (tgt_data->redopid >= 0);
      // if destination is a reduction, source must be also and must match
      assert(tgt_data->redopid == src_data->redopid);

      // for now, a reduction list instance can only be copied back to a
      //  non-reduction instance
      bool red_list = (src_data->redopid >= 0) && (src_data->red_list_size >= 0);
      if(red_list)
	assert(tgt_data->redopid < 0);

      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *tgt_mem = tgt_impl->memory.impl();

      // get valid masks from region to limit copy to correct data
      IndexSpace::Impl *is_impl = is.impl();
      //RegionMetaDataUntyped::Impl *src_reg = src_data->region.impl();
      //RegionMetaDataUntyped::Impl *tgt_reg = tgt_data->region.impl();

      log_dma.info("copy: " IDFMT "->" IDFMT " (" IDFMT "/%p)",
		    src.id, target.id, is.id, is_impl->valid_mask);

      // if we're missing the valid mask at this point, we've screwed up
      if(!is_impl->valid_mask_complete) {
	assert(is_impl->valid_mask_complete);
      }

      log_dma.debug("performing copy " IDFMT " (%d) -> " IDFMT " (%d) - %zd bytes (%zd)", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size);

      switch(src_mem->kind) {
      case Memory::Impl::MKIND_SYSMEM:
      case Memory::Impl::MKIND_ZEROCOPY:
	{
	  const void *src_ptr = src_mem->get_direct_ptr(src_data->alloc_offset, bytes_to_copy);
	  assert(src_ptr != 0);

	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
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

	  case Memory::Impl::MKIND_GLOBAL:
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

	  case Memory::Impl::MKIND_GPUFB:
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

	  case Memory::Impl::MKIND_REMOTE:
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
		log_dma.info("triggering event " IDFMT "/%d after empty remote copy",
			     finish_event.id, finish_event.gen);
		assert(finish_event == after_copy);
		finish_event.impl()->trigger(finish_event.gen, gasnet_mynode());
	      }
	      
	      return;
	    }

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GLOBAL:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
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

	  case Memory::Impl::MKIND_GLOBAL:
	    {
	      assert(!redop);
	      RangeExecutors::GasnetGetAndPut rexec(tgt_mem, tgt_data->alloc_offset,
						    src_mem, src_data->alloc_offset,
						    elmt_size);
	      ElementMask::forall_ranges(rexec, *is_impl->valid_mask);
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
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

      case Memory::Impl::MKIND_GPUFB:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
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

	  case Memory::Impl::MKIND_GLOBAL:
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

	  case Memory::Impl::MKIND_GPUFB:
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
    
    ReduceRequest::ReduceRequest(const Domain& _domain,
				 const std::vector<Domain::CopySrcDstField>& _srcs,
				 const Domain::CopySrcDstField& _dst,
				 bool _inst_lock_needed,
				 ReductionOpID _redop_id,
				 bool _red_fold,
				 Event _before_copy,
				 Event _after_copy,
				 int _priority)
      : DmaRequest(_priority, _after_copy),
	domain(_domain),
	dst(_dst), 
	inst_lock_needed(_inst_lock_needed), inst_lock_event(Event::NO_EVENT),
	redop_id(_redop_id), red_fold(_red_fold),
	before_copy(_before_copy)
    {
      srcs.insert(srcs.end(), _srcs.begin(), _srcs.end());

      log_dma.info("dma request %p created - " IDFMT "[%d]->" IDFMT "[%d]:%d (+%zd) %s %d (" IDFMT ") " IDFMT "/%d " IDFMT "/%d",
		   this,
		   srcs[0].inst.id, srcs[0].offset,
		   dst.inst.id, dst.offset, dst.size,
		   srcs.size() - 1,
		   (red_fold ? "fold" : "apply"),
		   redop_id,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);

#ifdef LEGION_LOGGING
      log_timing_event(Processor::NO_PROC, after_copy, COPY_INIT);
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

      // better have consumed exactly the right amount of data
      assert(((idata - ((const IDType *)data))*sizeof(IDType)) == datalen);

      log_dma.info("dma request %p deserialized - " IDFMT "[%d]->" IDFMT "[%d]:%d (+%zd) %s %d (" IDFMT ") " IDFMT "/%d " IDFMT "/%d",
		   this,
		   srcs[0].inst.id, srcs[0].offset,
		   dst.inst.id, dst.offset, dst.size,
		   srcs.size() - 1,
		   (red_fold ? "fold" : "apply"),
		   redop_id,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);
    }

    ReduceRequest::~ReduceRequest(void)
    {
    }

    size_t ReduceRequest::serialize(void *buffer, size_t maxlen)
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

      size_t msglen = ((const char *)msgptr) - ((const char *)buffer);
      assert(msglen <= maxlen); // TODO: maybe detect _before_ overrunning?

      return msglen;
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
	  IndexSpace::Impl *is_impl = domain.get_index_space().impl();
	  if(!is_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no index space metadata yet", this);
	    if(just_check) return false;

	    Event e = is_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - index space metadata invalid - instant trigger", this);
	      is_impl->lock.release();
	    } else {
	      log_dma.info("request %p - index space metadata invalid - sleeping on lock " IDFMT "", this, is_impl->lock.me.id);
	      waiter.sleep_on_event(e, is_impl->lock.me);
	      return false;
	    }
	  }

          // we need more than just the metadata - we also need the valid mask
          {
            Event e = is_impl->request_valid_mask();
            if(!e.has_triggered()) {
              log_dma.info("request %p - valid mask needed for index space " IDFMT " - sleeping on event " IDFMT "/%d", this, domain.get_index_space().id, e.id, e.gen);
	      waiter.sleep_on_event(e);
              return false;
            }
          }
	}

	// now go through all source instance pairs
	for(std::vector<Domain::CopySrcDstField>::iterator it = srcs.begin();
	    it != srcs.end();
	    it++) {
	  RegionInstance::Impl *src_impl = it->inst.impl();

	  if(!src_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no src instance (" IDFMT ") metadata yet", this, it->inst.id);
	    if(just_check) return false;

	    Event e = src_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - src instance metadata invalid - instant trigger", this);
	      src_impl->lock.release();
	    } else {
	      log_dma.info("request %p - src instance metadata invalid - sleeping on lock " IDFMT "", this, src_impl->lock.me.id);
	      waiter.sleep_on_event(e, src_impl->lock.me);
	      return false;
	    }
	  }
	  if(!src_impl->linearization.valid()) {
	    //printf("deserializing linearizer\n");
	    src_impl->linearization.deserialize(src_impl->locked_data.linearization_bits);
	  }
	}

	{
	  RegionInstance::Impl *dst_impl = dst.inst.impl();

	  if(!dst_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no dst instance (" IDFMT ") metadata yet", this, dst.inst.id);
	    if(just_check) return false;

	    Event e = dst_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - dst instance metadata invalid - instant trigger", this);
	      dst_impl->lock.release();
	    } else {
	      log_dma.info("request %p - dst instance metadata invalid - sleeping on lock " IDFMT "", this, dst_impl->lock.me.id);
	      waiter.sleep_on_event(e, dst_impl->lock.me);
	      return false;
	    }
	  }
	  if(!dst_impl->linearization.valid()) {
	    //printf("deserializing linearizer\n");
	    dst_impl->linearization.deserialize(dst_impl->locked_data.linearization_bits);
	  }
	}

	// if we got all the way through, we've got all the metadata we need
	state = STATE_BEFORE_EVENT;
      }

      // make sure our functional precondition has occurred
      if(state == STATE_BEFORE_EVENT) {
	// has the before event triggered?  if not, wait on it
	if(before_copy.has_triggered()) {
	  log_dma.info("request %p - before event triggered", this);
	  if(inst_lock_needed) {
	    // request an exclusive lock on the instance to protect reductions
	    inst_lock_event = dst.inst.impl()->lock.acquire(0, true /*excl*/);
	    state = STATE_INST_LOCK;
	    log_dma.info("request %p - instance lock acquire event " IDFMT "/%d",
			 this, inst_lock_event.id, inst_lock_event.gen);
	  } else {
	    // go straight to ready
	    state = STATE_READY;
	  }
	} else {
	  log_dma.info("request %p - before event not triggered", this);
	  if(just_check) return false;

	  log_dma.info("request %p - sleeping on before event", this);
	  waiter.sleep_on_event(before_copy);
	  return false;
	}
      }

      if(state == STATE_INST_LOCK) {
	if(inst_lock_event.has_triggered()) {
	  log_dma.info("request %p - instance lock acquired", this);
	  state = STATE_READY;
	} else {
	  log_dma.info("request %p - instance lock - sleeping on event " IDFMT "/%d", this, inst_lock_event.id, inst_lock_event.gen);
	  waiter.sleep_on_event(inst_lock_event);
	  return false;
	}
      }

      if(state == STATE_READY) {
	log_dma.info("request %p ready", this);
	if(just_check) return true;

	state = STATE_QUEUED;
	assert(rq != 0);
	log_dma.info("request %p enqueued", this);

#ifdef LEGION_LOGGING
	log_timing_event(Processor::NO_PROC, after_copy, COPY_READY);
#endif

	// once we're enqueued, we may be deleted at any time, so no more
	//  references
	rq->enqueue_request(this);
	return true;
      }

      if(state == STATE_QUEUED)
	return true;

      assert(0);
    }

    template <unsigned DIM>
    void ReduceRequest::perform_dma_rect(MemPairCopier *mpc)
    {
      Arrays::Rect<DIM> orig_rect = domain.get_rect<DIM>();

      const ReductionOpUntyped *redop = reduce_op_table[redop_id];

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

      Arrays::Mapping<DIM, 1> *src_linearization = src_inst.impl()->linearization.get_mapping<DIM>();
      Arrays::Mapping<DIM, 1> *dst_linearization = dst_inst.impl()->linearization.get_mapping<DIM>();

      // see what linear subrects we can get - again, iterate over destination first for gathering
      for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lso(orig_rect, *dst_linearization); lso; lso++) {
	for(typename Arrays::Mapping<DIM, 1>::LinearSubrectIterator lsi(lso.subrect, *src_linearization); lsi; lsi++) {
	  // see if we can compress the strides for a more efficient copy
	  Point<1> src_cstrides[DIM], dst_cstrides[DIM];
	  Point<DIM> extents;
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
	      Rect<1> irect = dsi.image;
	      // rectangle in output must be recalculated
	      Rect<DIM> subrect_check;
	      Rect<1> orect = dst_linearization->image_dense_subrect(dsi.subrect, subrect_check);
	      assert(dsi.subrect == subrect_check);
	      
	      //for(OASVec::iterator it2 = oasvec.begin(); it2 != oasvec.end(); it2++)
	      for (unsigned idx = 0; idx < oasvec.size(); idx++)
		ipc->copy_field(irect.lo, orect.lo, irect.hi - irect.lo + 1, idx);
	      //it2->src_offset, it2->dst_offset, it2->size);
	    }
	  }
	}
      }
      
      delete ipc;
    }

    void ReduceRequest::perform_dma(void)
    {
      log_dma.info("request %p executing", this);

#ifdef LEGION_LOGGING
      log_timing_event(Processor::NO_PROC, after_copy, COPY_BEGIN);

      // the copy might not actually finish in this thread, so set up an event waiter
      //  to log the completion
      after_copy.impl()->add_waiter(after_copy, new CopyCompletionLogger(after_copy));
#endif

      DetailedTimer::ScopedPush sp(TIME_COPY);

      // code assumes a single source field for now
      assert(srcs.size() == 1);

      Memory src_mem = srcs[0].inst.impl()->memory;
      Memory dst_mem = dst.inst.impl()->memory;
      Memory::Impl::MemoryKind src_kind = src_mem.impl()->kind;
      Memory::Impl::MemoryKind dst_kind = dst_mem.impl()->kind;

      const ReductionOpUntyped *redop = reduce_op_table[redop_id];

      //printf("kinds: " IDFMT "=%d " IDFMT "=%d\n", src_mem.id, src_mem.impl()->kind, dst_mem.id, dst_mem.impl()->kind);

      // we have the same jumble of memory type and layout permutations here - again we'll
      //  solve a few of them point-wise and then try to unify later
      if(domain.get_dim() == 0) {
	// index space
	IndexSpace::Impl *ispace = domain.get_index_space().impl();
	assert(ispace->valid_mask_complete);

	if((src_kind == Memory::Impl::MKIND_SYSMEM) ||
	   (src_kind == Memory::Impl::MKIND_ZEROCOPY) ||
	   (src_kind == Memory::Impl::MKIND_RDMA)) {
	  void *src_base = 0;
	  size_t src_stride = 0;
	  bool src_ok = srcs[0].inst.impl()->get_strided_parameters(src_base, src_stride,
								    srcs[0].offset);
	  assert(src_ok);

	  switch(dst_kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	  case Memory::Impl::MKIND_RDMA:
	    {
	      void *dst_base = 0;
	      size_t dst_stride = 0;
	      bool dst_ok = dst.inst.impl()->get_strided_parameters(dst_base, dst_stride,
								    dst.offset);
	      assert(dst_ok);

	      // if source and dest are ok, we can just walk the index space's spans
	      ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	      int rstart, rlen;
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

	      // all done - we can trigger the event locally in this case
	      after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
              delete e;
	      break;
	    }

	  case Memory::Impl::MKIND_REMOTE:
            {
	      // we need to figure out how to calculate offsets in the destination memory
	      StaticAccess<RegionInstance::Impl> dst_idata(dst.inst.impl());
	      assert(dst_idata->valid);

	      off_t dst_field_start;
	      int dst_field_size;
	      find_field_start(dst_idata->field_sizes, dst.offset, dst.size, dst_field_start, dst_field_size);
	      assert(dst.size == dst_field_size);

	      // index space instances use 1D linearizations for translation
	      Arrays::Mapping<1, 1> *dst_linearization = dst.inst.impl()->linearization.get_mapping<1>();

	      ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	      int rstart, rlen;

	      // get an RDMA sequence number so we can have the far side trigger the event once all reductions have been
	      //  applied
	      unsigned sequence_id = __sync_fetch_and_add(&rdma_sequence_no, 1);
	      unsigned rdma_count = 0;

	      // for a reduction from a fold instance, it's always ok to copy unused elements, since they'll have an
	      //  identity value stored for them
	      int rlen_target = 32768 / dst_field_size;

	      while(e->get_next(rstart, rlen)) {
		// do we want to copy extra elements to fill in some holes?
		while(rlen < rlen_target) {
		  // see where the next valid elements are
		  int rstart2, rlen2;
		  // if none, stop
		  if(!e->peek_next(rstart2, rlen2)) break;
		  // or if they don't even start until outside the window, stop
		  if(rstart2 > (rstart + rlen_target)) break;
		  // ok, include the next valid element(s) and any invalid ones in between
		  //printf("bloating from %d to %d\n", rlen, rstart2 + rlen2 - rstart);
		  rlen = rstart2 + rlen2 - rstart;
		  // and actually take the next range from the enumerator
		  e->get_next(rstart2, rlen2);
		}

		// translate the index space point to the dst instance's linearization
		int dstart = dst_linearization->image(rstart);

		// now do an offset calculation for the destination
		off_t dst_offset;
		off_t dst_stride;
		if(dst_idata->block_size > 1) {
		  // straddling a block boundary is complicated
		  assert((dstart / dst_idata->block_size) == ((dstart + rlen - 1) / dst_idata->block_size));
		  dst_offset = calc_mem_loc(dst_idata->alloc_offset, dst_field_start, dst_field_size,
					    dst_idata->elmt_size, dst_idata->block_size, dstart);
		  dst_stride = dst_field_size;
		} else {
		  // easy case
		  dst_offset = dst_idata->alloc_offset + (dstart * dst_idata->elmt_size) + dst_field_start;
		  dst_stride = dst_idata->elmt_size;
		}

		rdma_count += do_remote_reduce(dst_mem, dst_offset, redop_id, red_fold,
					       ((const char *)src_base) + (rstart * src_stride),
					       rlen, src_stride, dst_stride,
					       sequence_id, Event::NO_EVENT);
	      }

	      // if we did any actual reductions, send a fence, otherwise trigger here
	      if(rdma_count > 0) {
		do_remote_fence(dst_mem, sequence_id, rdma_count, after_copy);
	      } else {
		after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
	      }
              delete e;
	      break;
            }

	  case Memory::Impl::MKIND_GLOBAL:
	    {
	      // make sure we've requested a lock on the dst instance
	      assert(inst_lock_needed);

	      // we need to figure out how to calculate offsets in the destination memory
	      RegionInstance::Impl *dst_impl = dst.inst.impl();
	      StaticAccess<RegionInstance::Impl> dst_idata(dst_impl);
	      assert(dst_idata->valid);

	      off_t dst_field_start;
	      int dst_field_size;
	      find_field_start(dst_idata->field_sizes, dst.offset, dst.size, dst_field_start, dst_field_size);
	      assert(dst.size == dst_field_size);

	      // index space instances use 1D linearizations for translation
	      Arrays::Mapping<1, 1> *dst_linearization = dst_impl->linearization.get_mapping<1>();

	      // if source and dest are ok, we can just walk the index space's spans
	      ElementMask::Enumerator *e = ispace->valid_mask->enumerate_enabled();
	      int rstart, rlen;
	      while(e->get_next(rstart, rlen)) {
		// translate the index space point to the dst instance's linearization
		int dstart = dst_linearization->image(rstart);

		// now do an offset calculation for the destination
		off_t dst_offset;
		off_t dst_stride;
		if(dst_idata->block_size > 1) {
		  // straddling a block boundary is complicated
		  assert((dstart / dst_idata->block_size) == ((dstart + rlen - 1) / dst_idata->block_size));
		  dst_offset = calc_mem_loc(dst_idata->alloc_offset, dst_field_start, dst_field_size,
					    dst_idata->elmt_size, dst_idata->block_size, dstart);
		  dst_stride = dst_field_size;
		} else {
		  // easy case
		  dst_offset = dst_idata->alloc_offset + (dstart * dst_idata->elmt_size) + dst_field_start;
		  dst_stride = dst_idata->elmt_size;
		}

		// get a temporary buffer in local memory
		// this may have extra data if stride > field_size, but that's
		//  ok - we'll write back whatever we read
		void *buffer = malloc(dst_stride * rlen);

		dst_mem.impl()->get_bytes(dst_offset, buffer, dst_stride * rlen);

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

		dst_mem.impl()->put_bytes(dst_offset, buffer, dst_stride * rlen);

		// release the temp buffer
		free(buffer);
	      }

	      // all done - we can trigger the event locally in this case
	      after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());

	      // also release the instance lock
	      dst_impl->lock.release();

              delete e;

	      break;
	    }

	  default:
	    assert(0);
	  }
	}
      } else {
	MemPairCopier *mpc = MemPairCopier::create_copier(src_mem, dst_mem, redop, red_fold);

	switch(domain.get_dim()) {
	case 1: perform_dma_rect<1>(mpc); break;
	case 2: perform_dma_rect<2>(mpc); break;
	case 3: perform_dma_rect<3>(mpc); break;
	default: assert(0);
	}

	mpc->flush(after_copy);

	// if an instance lock was taken, release it after copy completes
	if(inst_lock_needed)
	  dst.inst.impl()->lock.me.release(after_copy);

	delete mpc;
      }

      log_dma.info("dma request %p finished - " IDFMT "[%d]->" IDFMT "[%d]:%d (+%zd) %s %d (" IDFMT ") " IDFMT "/%d " IDFMT "/%d",
		   this,
		   srcs[0].inst.id, srcs[0].offset,
		   dst.inst.id, dst.offset, dst.size,
		   srcs.size() - 1,
		   (red_fold ? "fold" : "apply"),
		   redop_id,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);
    }

    static volatile bool terminate_flag = false;
    static int num_threads = 0;
    static pthread_t *worker_threads = 0;

    // for now we use a single queue for all (local) dmas
    static DmaRequestQueue *dma_queue = 0;
    
    static void *dma_worker_thread_loop(void *arg)
    {
      DmaRequestQueue *rq = (DmaRequestQueue *)arg;

      log_dma.info("dma worker thread created");

      while(!terminate_flag) {
	// get a request, sleeping as necessary
	DmaRequest *r = rq->dequeue_request(true);

	if(r) {
	  r->perform_dma();
	  delete r;
	}
      }

      log_dma.info("dma worker thread terminating");

      return 0;
    }
    
    void start_dma_worker_threads(int count)
    {
      dma_queue = new DmaRequestQueue;
      num_threads = count;

      worker_threads = new pthread_t[count];
      for(int i = 0; i < count; i++) {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	if(proc_assignment)
	  proc_assignment->bind_thread(-1, &attr, "DMA worker");
	CHECK_PTHREAD( pthread_create(&worker_threads[i], 0, 
				      dma_worker_thread_loop, dma_queue) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
#ifdef DEADLOCK_TRACE
        Runtime::get_runtime()->add_thread(&worker_threads[i]);
#endif
      }
    }

    void stop_dma_worker_threads(void)
    {
      terminate_flag = true;
      dma_queue->shutdown_queue();

      if(worker_threads) {
	for(int i = 0; i < num_threads; i++) {
	  void *dummy;
	  CHECK_PTHREAD( pthread_join(worker_threads[i], &dummy) );
	}
	num_threads = 0;
	delete[] worker_threads;
      }

      delete dma_queue;
      dma_queue = 0;
      terminate_flag = false;
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

    static int select_dma_node(Memory src_mem, Memory dst_mem,
			       ReductionOpID redop_id, bool red_fold)
    {
      int src_node = ID(src_mem).node();
      int dst_node = ID(dst_mem).node();

      bool src_is_rdma = src_mem.impl()->kind == Memory::Impl::MKIND_GLOBAL;
      bool dst_is_rdma = dst_mem.impl()->kind == Memory::Impl::MKIND_GLOBAL;

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

	r->check_readiness(false, dma_queue);
      } else {
	// a reduction
	ReduceRequest *r = new ReduceRequest(data, msglen,
					     args.redop_id,
					     args.red_fold,
					     args.before_copy,
					     args.after_copy,
					     args.priority);

	r->check_readiness(false, dma_queue);
      }
    }

    template <typename T> T min(T a, T b) { return (a < b) ? a : b; }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       Event wait_on,
		       ReductionOpID redop_id, bool red_fold) const
    {
      if(redop_id == 0) {
	// not a reduction, so sort fields by src/dst mem pairs

	OASByMem oas_by_mem;

	std::vector<CopySrcDstField>::const_iterator src_it = srcs.begin();
	std::vector<CopySrcDstField>::const_iterator dst_it = dsts.begin();
	unsigned src_suboffset = 0;
	unsigned dst_suboffset = 0;
	while((src_it != srcs.end()) && (dst_it != dsts.end())) {
	  InstPair ip(src_it->inst, dst_it->inst);
	  MemPair mp(src_it->inst.impl()->memory, dst_it->inst.impl()->memory);

	  // printf("I:(%x/%x) M:(%x/%x) sub:(%d/%d) src=(%d/%d) dst=(%d/%d)\n",
	  //        ip.first.id, ip.second.id, mp.first.id, mp.second.id,
	  //        src_suboffset, dst_suboffset,
	  //        src_it->offset, src_it->size, 
	  //        dst_it->offset, dst_it->size);

	  OASByInst *oas_by_inst;
	  OASByMem::iterator it = oas_by_mem.find(mp);
	  if(it != oas_by_mem.end()) {
	    oas_by_inst = it->second;
	  } else {
	    oas_by_inst = new OASByInst;
	    oas_by_mem[mp] = oas_by_inst;
	  }
	  OASVec& oasvec = (*oas_by_inst)[ip];

	  OffsetsAndSize oas;
	  oas.src_offset = src_it->offset + src_suboffset;
	  oas.dst_offset = dst_it->offset + dst_suboffset;
	  oas.size = min(src_it->size - src_suboffset, dst_it->size - dst_suboffset);
	  oasvec.push_back(oas);

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

	// now do a copy for each memory pair
	std::set<Event> finish_events;

	log_dma.info("copy: %zd distinct src/dst mem pairs, is=" IDFMT "", oas_by_mem.size(), is_id);

	for(OASByMem::const_iterator it = oas_by_mem.begin(); it != oas_by_mem.end(); it++) {
	  Memory src_mem = it->first.first;
	  Memory dst_mem = it->first.second;
	  OASByInst *oas_by_inst = it->second;

	  Event ev = Event::Impl::create_event();

	  int priority = 0;
#ifdef USE_CUDA
	  if (src_mem.impl()->kind == Memory::Impl::MKIND_GPUFB)
	    priority = 1;
	  else if (dst_mem.impl()->kind == Memory::Impl::MKIND_GPUFB)
	    priority = 1;
#endif

	  CopyRequest *r = new CopyRequest(*this, oas_by_inst, 
					   wait_on, ev, priority);

	  // ask which node should perform the copy
	  int dma_node = select_dma_node(src_mem, dst_mem, redop_id, red_fold);
	  log_dma.info("copy: srcmem=" IDFMT " dstmem=" IDFMT " node=%d", src_mem.id, dst_mem.id, dma_node);
	  
	  if(dma_node == gasnet_mynode()) {
	    log_dma.info("performing copy on local node");
	  
	    r->check_readiness(false, dma_queue);

	    finish_events.insert(ev);
	  } else {
	    RemoteCopyArgs args;
	    args.redop_id = 0;
	    args.red_fold = false;
	    args.before_copy = wait_on;
	    args.after_copy = ev;
	    args.priority = priority;

	    static const size_t MAX_MESSAGE_SIZE = 16384;
	    int msgdata[MAX_MESSAGE_SIZE / sizeof(int)];

	    size_t msglen = r->serialize(msgdata, MAX_MESSAGE_SIZE);

	    log_dma.info("performing copy on remote node (%d), event=" IDFMT "/%d", dma_node, args.after_copy.id, args.after_copy.gen);
	    RemoteCopyMessage::request(dma_node, args, msgdata, msglen, PAYLOAD_COPY);
	  
	    finish_events.insert(ev);

	    // done with the local copy of the request
	    delete r;
	  }
	}

	// final event is merge of all individual copies' events
	return Event::Impl::merge_events(finish_events);
      } else {
	// we're doing a reduction - the semantics require that all source fields be pulled
	//  together and applied as a "structure" to the reduction op

	// figure out where the source data is
	int src_node = -1;

	for(std::vector<CopySrcDstField>::const_iterator src_it = srcs.begin();
	    src_it != srcs.end();
	    src_it++)
	{
	  int n = ID(src_it->inst).node();
	  if((src_node != -1) && (src_node != n)) {
	    // for now, don't handle case where source data is split across nodes
	    assert(0);
	  }
	  src_node = n;
	}

	assert(dsts.size() == 1);

	// some destinations (e.g. GASNET) need a lock taken to ensure
	//  reductions are applied atomically
	Memory::Impl::MemoryKind dst_kind = dsts[0].inst.impl()->memory.impl()->kind;
	bool inst_lock_needed = (dst_kind == Memory::Impl::MKIND_GLOBAL);

	Event ev = Event::Impl::create_event();

	ReduceRequest *r = new ReduceRequest(*this, 
					     srcs, dsts[0],
					     inst_lock_needed,
					     redop_id, red_fold,
					     wait_on, ev,
					     0 /*priority*/);

	if(src_node == gasnet_mynode()) {
	  log_dma.info("performing reduction on local node");
	  
	  r->check_readiness(false, dma_queue);
	} else {
	  RemoteCopyArgs args;
	  args.redop_id = redop_id;
	  args.red_fold = red_fold;
	  args.before_copy = wait_on;
	  args.after_copy = ev;
	  args.priority = 0 /*priority*/;

	  static const size_t MAX_MESSAGE_SIZE = 16384;
	  IDType msgdata[MAX_MESSAGE_SIZE / sizeof(IDType)];

	  size_t msglen = r->serialize(msgdata, MAX_MESSAGE_SIZE);

	  log_dma.info("performing reduction on remote node (%d), event=" IDFMT "/%d",
		       src_node, args.after_copy.id, args.after_copy.gen);
	  RemoteCopyMessage::request(src_node, args, msgdata, msglen, PAYLOAD_COPY);
	  // done with the local copy of the request
	  delete r;
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
    }

  };
};
