/* Copyright 2013 Stanford University
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
#include "lowlevel_gpu.h"
#include "accessor.h"

#include <queue>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

namespace LegionRuntime {
  namespace LowLevel {

    Logger::Category log_dma("dma");

    class DmaRequest : public Event::Impl::EventWaiter {
    public:
      DmaRequest(const Domain& _domain,
		 const std::vector<Domain::CopySrcDstField>& _srcs,
		 const std::vector<Domain::CopySrcDstField>& _dsts,
		 ReductionOpID _redop_id, bool _red_fold,
		 Event _before_copy,
		 Event _after_copy);

      enum State {
	STATE_INIT,
	STATE_METADATA_FETCH,
	STATE_BEFORE_EVENT,
	STATE_READY,
	STATE_QUEUED,
	STATE_DONE
      };

      virtual void event_triggered(void);
      virtual void print_info(void);

      bool check_readiness(bool just_check);

      template <unsigned DIM>
      void perform_dma_rect(void);

      void perform_dma(void);

      bool handler_safe(void) { return(true); }

      Domain domain;
      std::vector<Domain::CopySrcDstField> srcs, dsts;
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy;
      Event after_copy;
      State state;
      Lock current_lock;
    };

    gasnet_hsl_t queue_mutex;
    gasnett_cond_t queue_condvar;
    std::queue<DmaRequest *> dma_queue;
    
    DmaRequest::DmaRequest(const Domain& _domain,
			   const std::vector<Domain::CopySrcDstField>& _srcs,
			   const std::vector<Domain::CopySrcDstField>& _dsts,
			   ReductionOpID _redop_id, bool _red_fold,
			   Event _before_copy,
			   Event _after_copy)
      : domain(_domain), srcs(_srcs), dsts(_dsts),
	redop_id(_redop_id), red_fold(_red_fold),
	before_copy(_before_copy), after_copy(_after_copy),
	state(STATE_INIT), current_lock(Lock::NO_LOCK)
    {
      log_dma.info("dma request %p created - (%x[%d+%d]+%zd)->(%x[%d+%d]+%zd) (%x) %x/%d %x/%d",
		   this,
		   srcs[0].inst.id, srcs[0].offset, srcs[0].size, srcs.size()-1,
		   dsts[0].inst.id, dsts[0].offset, dsts[0].size, dsts.size()-1,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);
    }

    void DmaRequest::event_triggered(void)
    {
      log_dma.info("request %p triggered in state %d (lock = %x)", this, state, current_lock.id);

      if(current_lock.exists()) {
	current_lock.unlock();
	current_lock = Lock::NO_LOCK;
      }

      // this'll enqueue the DMA if it can, or wait on another event if it 
      //  can't
      check_readiness(false);
    }

    void DmaRequest::print_info(void)
    {
      printf("dma request %p", this);
    }

    bool DmaRequest::check_readiness(bool just_check)
    {
      if(state == STATE_INIT)
	state = STATE_METADATA_FETCH;

      // make sure our node has all the meta data it needs, but don't take more than one lock
      //  at a time
      if(state == STATE_METADATA_FETCH) {
	// index space first
	if(domain.get_dim() == 0) {
	  IndexSpace::Impl *is_impl = domain.get_index_space().impl();
	  if(!is_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no index space metadata yet", this);
	    if(just_check) return false;

	    Event e = is_impl->lock.lock(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - index space metadata invalid - instant trigger", this);
	      is_impl->lock.unlock();
	    } else {
	      current_lock = is_impl->lock.me;
	      log_dma.info("request %p - index space metadata invalid - sleeping on lock %x", this, current_lock.id);
	      e.impl()->add_waiter(e, this);
	      return false;
	    }
	  }
	}

	// now go through all src fields and make sure we have instance metadata
	for(std::vector<Domain::CopySrcDstField>::const_iterator it = srcs.begin();
	    it != srcs.end();
	    it++) {
	  RegionInstance::Impl *impl = it->inst.impl();

	  if(!impl->locked_data.valid) {
	    log_dma.info("dma request %p - no src instance (%x) metadata yet", this, it->inst.id);
	    if(just_check) return false;

	    Event e = impl->lock.lock(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - src instance metadata invalid - instant trigger", this);
	      impl->lock.unlock();
	    } else {
	      current_lock = impl->lock.me;
	      log_dma.info("request %p - src instance metadata invalid - sleeping on lock %x", this, current_lock.id);
	      e.impl()->add_waiter(e, this);
	      return false;
	    }
	  }
	  if(!impl->linearization.valid()) {
	    //printf("deserializing linearizer\n");
	    impl->linearization.deserialize(impl->locked_data.linearization_bits);
	  }
	}

	// same for dst fields
	for(std::vector<Domain::CopySrcDstField>::const_iterator it = dsts.begin();
	    it != dsts.end();
	    it++) {
	  RegionInstance::Impl *impl = it->inst.impl();

	  if(!impl->locked_data.valid) {
	    log_dma.info("dma request %p - no dst instance (%x) metadata yet", this, it->inst.id);
	    if(just_check) return false;

	    Event e = impl->lock.lock(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - dst instance metadata invalid - instant trigger", this);
	      impl->lock.unlock();
	    } else {
	      current_lock = impl->lock.me;
	      log_dma.info("request %p - dst instance metadata invalid - sleeping on lock %x", this, current_lock.id);
	      e.impl()->add_waiter(e, this);
	      return false;
	    }
	  }
	  if(!impl->linearization.valid()) {
	    //printf("deserializing linearizer\n");
	    impl->linearization.deserialize(impl->locked_data.linearization_bits);
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
	  before_copy.impl()->add_waiter(before_copy, this);
	  return false;
	}
      }

      if(state == STATE_READY) {
	log_dma.info("request %p ready", this);
	if(just_check) return true;

	// enqueue ourselves for execution
	gasnet_hsl_lock(&queue_mutex);
	dma_queue.push(this);
	state = STATE_QUEUED;
	gasnett_cond_signal(&queue_condvar);
	gasnet_hsl_unlock(&queue_mutex);
	log_dma.info("request %p enqueued", this);
      }

      if(state == STATE_QUEUED)
	return true;

      assert(0);
    }

    // defined in lowlevel.cc
    extern void do_remote_write(Memory mem, off_t offset,
				const void *data, size_t datalen,
				Event event, bool make_copy = false);

    extern void do_remote_apply_red_list(int node, Memory mem, off_t offset,
					 ReductionOpID redopid,
					 const void *data, size_t datalen);

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
					   entry_buffer, pos * redop->sizeof_list_entry);
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
					 entry_buffer, pos * redop->sizeof_list_entry);
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
			  last ? event : Event::NO_EVENT);
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

    template <unsigned DIM>
    void DmaRequest::perform_dma_rect(void)
    {
      Arrays::Rect<DIM> orig_rect = domain.get_rect<DIM>();

      for(Arrays::GenericPointInRectIterator<DIM> pir(orig_rect); pir; pir++) {
	char buffer[1024];
	size_t rcount = 0;
	for(std::vector<Domain::CopySrcDstField>::const_iterator it = srcs.begin(); it != srcs.end(); it++) {
	  it->inst.get_accessor().read_untyped(DomainPoint::from_point<DIM>(pir.p), &buffer[rcount], it->size, it->offset);
	  rcount += it->size;
	}
	size_t wcount = 0;
	for(std::vector<Domain::CopySrcDstField>::const_iterator it = dsts.begin(); it != dsts.end(); it++) {
	  it->inst.get_accessor().write_untyped(DomainPoint::from_point<DIM>(pir.p), &buffer[wcount], it->size, it->offset);
	  wcount += it->size;
	}
	assert(rcount == wcount);
      }
    }

    void DmaRequest::perform_dma(void)
    {
      log_dma.info("request %p executing", this);

      DetailedTimer::ScopedPush sp(TIME_COPY);

      switch(domain.get_dim()) {
      case 0:
	{
	  // iterate over valid ranges of an index space
	  assert(0);
	  break;
	}

	// rectangle cases
      case 1: perform_dma_rect<1>(); break;
      case 2: perform_dma_rect<2>(); break;
      case 3: perform_dma_rect<3>(); break;

      default: assert(0);
      };

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

      log_dma.info("copy: %x->%x (%x/%p)",
		    src.id, target.id, is.id, is_impl->valid_mask);

      // if we're missing the valid mask at this point, we've screwed up
      if(!is_impl->valid_mask_complete) {
	assert(is_impl->valid_mask_complete);
      }

      log_dma.debug("performing copy %x (%d) -> %x (%d) - %zd bytes (%zd)", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size);

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

	  case Memory::Impl::MKIND_GASNET:
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
		log_dma.info("triggering event %x/%d after empty remote copy",
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

      case Memory::Impl::MKIND_GASNET:
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

	  case Memory::Impl::MKIND_GASNET:
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

	  case Memory::Impl::MKIND_GASNET:
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

      log_dma.debug("finished copy %x (%d) -> %x (%d) - %zd bytes (%zd), event=%x/%d", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size, after_copy.id, after_copy.gen);
#endif
      log_dma.info("dma request %p finished - (%x[%d+%d]+%zd)->(%x[%d+%d]+%zd) (%x) %x/%d %x/%d",
		   this,
		   srcs[0].inst.id, srcs[0].offset, srcs[0].size, srcs.size()-1,
		   dsts[0].inst.id, dsts[0].offset, dsts[0].size, dsts.size()-1,
		   domain.is_id,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);

      if(after_copy.exists())
	after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
    }
    
    bool terminate_flag = false;
    int num_threads = 0;
    pthread_t *worker_threads = 0;
    
    void init_dma_handler(void)
    {
      gasnet_hsl_init(&queue_mutex);
      gasnett_cond_init(&queue_condvar);
    }

    static void *dma_worker_thread_loop(void *dummy)
    {
      log_dma.info("dma worker thread created");

      // we spend most of this loop holding the queue mutex - we let go of it
      //  when we have a real copy to do
      gasnet_hsl_lock(&queue_mutex);

      while(!terminate_flag) {
	// take the queue lock and try to pull an item off the front
	if(dma_queue.size() > 0) {
	  DmaRequest *req = dma_queue.front();
	  dma_queue.pop();

	  gasnet_hsl_unlock(&queue_mutex);
	  
	  req->perform_dma();
	  //delete req;

	  gasnet_hsl_lock(&queue_mutex);
	} else {
	  // sleep until we get a signal, or until everybody is woken up
	  //  via broadcast for termination
	  gasnett_cond_wait(&queue_condvar, &queue_mutex.lock);
	}
      }
      gasnet_hsl_unlock(&queue_mutex);

      log_dma.info("dma worker thread terminating");

      return 0;
    }
    
    void start_dma_worker_threads(int count)
    {
      num_threads = count;

      worker_threads = new pthread_t[count];
      for(int i = 0; i < count; i++)
	CHECK_PTHREAD( pthread_create(&worker_threads[i], 0, 
				      dma_worker_thread_loop, 0) );
    }
    
    Event enqueue_dma(const Domain& domain,
		      const std::vector<Domain::CopySrcDstField>& srcs,
		      const std::vector<Domain::CopySrcDstField>& dsts,
		      ReductionOpID redop_id, bool red_fold,
		      Event before_copy,
		      Event after_copy /*= Event::NO_EVENT*/)
    {
      // special case - if we have everything we need, we can consider doing the
      //   copy immediately
      DetailedTimer::ScopedPush sp(TIME_COPY);

      DmaRequest *r = new DmaRequest(domain, srcs, dsts, redop_id, red_fold,
				     before_copy, after_copy);

      bool ready = r->check_readiness(true);

      log_dma.info("dma: request %p appears to be immediately ready?", r);
      
      // copy is all ready to go and safe to perform in a handler thread
      if(0 && ready && r->handler_safe()) {
	r->perform_dma();
	//delete r;
	return after_copy;
      } else {
	if(!after_copy.exists())
	  r->after_copy = after_copy = Event::Impl::create_event();

	// calling this with 'just_check'==false means it'll automatically
	//  enqueue the dma if ready
	r->check_readiness(false);

	return after_copy;
      }
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

      bool src_is_rdma = src_mem.impl()->kind == Memory::Impl::MKIND_GASNET;
      bool dst_is_rdma = dst_mem.impl()->kind == Memory::Impl::MKIND_GASNET;

      if(src_is_rdma) {
	if(dst_is_rdma) {
	  // gasnet -> gasnet - blech
	  assert(0);
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

      Domain domain;
      std::vector<Domain::CopySrcDstField> srcs(args.num_srcs);
      std::vector<Domain::CopySrcDstField> dsts(args.num_dsts);

      const int *idata = (const int *)data;
      idata = domain.deserialize(idata);

      for(int i = 0; i < args.num_srcs; i++) {
	srcs[i].inst.id = *idata++;
	srcs[i].offset = *idata++;
	srcs[i].size = *idata++;
      }
      for(int i = 0; i < args.num_dsts; i++) {
	dsts[i].inst.id = *idata++;
	dsts[i].offset = *idata++;
	dsts[i].size = *idata++;
      }

      assert(((idata - ((const int *)data))*sizeof(int)) == msglen);

      log_dma.info("received remote copy request: srcs=%d tgts=%d before=%x/%d after=%x/%d",
		   args.num_srcs, args.num_dsts,
		   args.before_copy.id, args.before_copy.gen,
		   args.after_copy.id, args.after_copy.gen);

      enqueue_dma(domain, srcs, dsts, args.redop_id, args.red_fold, args.before_copy, args.after_copy);
#if 0
      if(args.before_copy.has_triggered()) {
	RegionInstance::Impl::copy(args.source, args.target,
					  args.region,
					  args.elmt_size,
					  args.bytes_to_copy,
					  args.after_copy);
      } else {
	args.before_copy.impl()->add_waiter(args.before_copy,
					    new DeferredCopy(args.source,
							     args.target,
							     args.region,
							     args.elmt_size,
							     args.bytes_to_copy,
							     args.after_copy));
      }
#endif
    }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       Event wait_on,
		       ReductionOpID redop_id, bool red_fold) const
    {
      std::set<RegionInstance> src_insts, dst_insts;
      std::set<Memory> src_mems, dst_mems;
      
      for(std::vector<CopySrcDstField>::const_iterator it = srcs.begin(); it != srcs.end(); it++) {
	if(src_insts.insert(it->inst).second) {
	  // this was the first time we've seen this instance, so insert the memory too
	  src_mems.insert(it->inst.impl()->memory);
	}
      }
      for(std::vector<CopySrcDstField>::const_iterator it = dsts.begin(); it != dsts.end(); it++) {
	if(dst_insts.insert(it->inst).second) {
	  // this was the first time we've seen this instance, so insert the memory too
	  dst_mems.insert(it->inst.impl()->memory);
	}
      }

      log_dma.info("copy: %zd src mems, %zd dst mems, is=%x", src_mems.size(), dst_mems.size(), is_id);
      assert(redop_id == 0);

      if((src_mems.size() == 1) && (dst_mems.size() == 1)) {
	Memory src_mem = *(src_mems.begin());
	Memory dst_mem = *(dst_mems.begin());

	// ask which node should perform the copy
	int dma_node = select_dma_node(src_mem, dst_mem, redop_id, red_fold);

	if(dma_node == gasnet_mynode()) {
	  log_dma.info("performing copy on local node");
	
	  return enqueue_dma(*this, srcs, dsts, redop_id, red_fold, wait_on, Event::NO_EVENT);
	} else {
	  // need to send dma to remote node for processing, so we need a completion event
	  RemoteCopyArgs args;
	  args.num_srcs = srcs.size();
	  args.num_dsts = dsts.size();
	  args.redop_id = redop_id;
	  args.red_fold = red_fold;
	  args.before_copy = wait_on;
	  args.after_copy = Event::Impl::create_event();

	  int msgdata[64];

	  int *msgptr = serialize(msgdata);

	  for(std::vector<CopySrcDstField>::const_iterator it = srcs.begin(); it != srcs.end(); it++) {
	    *msgptr++ = (*it).inst.id;
	    *msgptr++ = (*it).offset;
	    *msgptr++ = (*it).size;
	  }
	  for(std::vector<CopySrcDstField>::const_iterator it = dsts.begin(); it != dsts.end(); it++) {
	    *msgptr++ = (*it).inst.id;
	    *msgptr++ = (*it).offset;
	    *msgptr++ = (*it).size;
	  }
	  size_t msglen = ((const char *)msgptr) - ((const char *)msgdata);
	  assert(msglen < 256);
	  log_dma.info("performing copy on remote node (%d), event=%x/%d", dma_node, args.after_copy.id, args.after_copy.gen);
	  RemoteCopyMessage::request(dma_node, args, msgdata, msglen, PAYLOAD_COPY);
	  return args.after_copy;
	}
      } else {
	// multiple source and/or dest memories
	log_dma.error("not handing multiple src/dst memories yet!");
	for(std::vector<CopySrcDstField>::const_iterator it = srcs.begin(); it != srcs.end(); it++)
          log_dma.error("  src: inst=%x offset=%d size=%d",
                        (*it).inst.id, (*it).offset, (*it).size);
	for(std::vector<CopySrcDstField>::const_iterator it = dsts.begin(); it != dsts.end(); it++)
          log_dma.error("  dst: inst=%x offset=%d size=%d",
                        (*it).inst.id, (*it).offset, (*it).size);
	assert(0);
	return Event::NO_EVENT;
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
