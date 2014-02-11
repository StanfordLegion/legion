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

using namespace LegionRuntime::Accessor;

#ifdef LEGION_LOGGING
#include "legion_logging.h"

using namespace LegionRuntime::HighLevel::LegionLogging;
#endif

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

    class DmaRequest : public Event::Impl::EventWaiter {
    public:
      DmaRequest(const Domain& _domain,
		 OASByInst *_oas_by_inst,
		 ReductionOpID _redop_id, bool _red_fold,
		 Event _before_copy,
		 Event _after_copy,
                 int priority);

      ~DmaRequest(void);

      enum State {
	STATE_INIT,
	STATE_METADATA_FETCH,
	STATE_BEFORE_EVENT,
	STATE_READY,
	STATE_QUEUED,
	STATE_DONE
      };

      virtual bool event_triggered(void);
      virtual void print_info(void);

      bool check_readiness(bool just_check);

      template <unsigned DIM>
      void perform_dma_rect(MemPairCopier *mpc);

      void perform_dma(void);

      bool handler_safe(void) { return(true); }

      Domain domain;
      OASByInst *oas_by_inst;
      ReductionOpID redop_id;
      bool red_fold;
      Event before_copy;
      Event after_copy;
      State state;
      Reservation current_lock;
      int priority;
    };

    gasnet_hsl_t queue_mutex;
    gasnett_cond_t queue_condvar;
    std::list<DmaRequest *> dma_queue;
    
    DmaRequest::DmaRequest(const Domain& _domain,
			   OASByInst *_oas_by_inst,
			   ReductionOpID _redop_id, bool _red_fold,
			   Event _before_copy,
			   Event _after_copy,
                           int _priority)
      : domain(_domain), oas_by_inst(_oas_by_inst),
	redop_id(_redop_id), red_fold(_red_fold),
	before_copy(_before_copy), after_copy(_after_copy),
	state(STATE_INIT), 
        current_lock(Reservation::NO_RESERVATION), priority(_priority)
    {
      log_dma.info("dma request %p created - %x[%d]->%x[%d]:%d (+%zd) (%x) %x/%d %x/%d",
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

    DmaRequest::~DmaRequest(void)
    {
      delete oas_by_inst;
    }

    bool DmaRequest::event_triggered(void)
    {
      log_dma.info("request %p triggered in state %d (lock = %x)", this, state, current_lock.id);

      if(current_lock.exists()) {
	current_lock.release();
	current_lock = Reservation::NO_RESERVATION;
      }

      // this'll enqueue the DMA if it can, or wait on another event if it 
      //  can't
      check_readiness(false);

      // don't delete us!
      return false;
    }

    void DmaRequest::print_info(void)
    {
      printf("dma request %p: after %x/%d\n", 
          this, after_copy.id, after_copy.gen);
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

	    Event e = is_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - index space metadata invalid - instant trigger", this);
	      is_impl->lock.release();
	    } else {
	      current_lock = is_impl->lock.me;
	      log_dma.info("request %p - index space metadata invalid - sleeping on lock %x", this, current_lock.id);
	      e.impl()->add_waiter(e, this);
	      return false;
	    }
	  }
	}

	// now go through all instance pairs
	for(OASByInst::iterator it = oas_by_inst->begin(); it != oas_by_inst->end(); it++) {
	  RegionInstance::Impl *src_impl = it->first.first.impl();
	  RegionInstance::Impl *dst_impl = it->first.second.impl();

	  if(!src_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no src instance (%x) metadata yet", this, it->first.first.id);
	    if(just_check) return false;

	    Event e = src_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - src instance metadata invalid - instant trigger", this);
	      src_impl->lock.release();
	    } else {
	      current_lock = src_impl->lock.me;
	      log_dma.info("request %p - src instance metadata invalid - sleeping on lock %x", this, current_lock.id);
	      e.impl()->add_waiter(e, this);
	      return false;
	    }
	  }
	  if(!src_impl->linearization.valid()) {
	    //printf("deserializing linearizer\n");
	    src_impl->linearization.deserialize(src_impl->locked_data.linearization_bits);
	  }

	  if(!dst_impl->locked_data.valid) {
	    log_dma.info("dma request %p - no dst instance (%x) metadata yet", this, it->first.second.id);
	    if(just_check) return false;

	    Event e = dst_impl->lock.acquire(1, false);
	    if(e.has_triggered()) {
	      log_dma.info("request %p - dst instance metadata invalid - instant trigger", this);
	      dst_impl->lock.release();
	    } else {
	      current_lock = dst_impl->lock.me;
	      log_dma.info("request %p - dst instance metadata invalid - sleeping on lock %x", this, current_lock.id);
	      e.impl()->add_waiter(e, this);
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
	  before_copy.impl()->add_waiter(before_copy, this);
	  return false;
	}
      }

      if(state == STATE_READY) {
	log_dma.info("request %p ready", this);
	if(just_check) return true;

	// enqueue ourselves for execution
	gasnet_hsl_lock(&queue_mutex);
        // Find the right place to insert this based on the priority
        // Common case: if the guy on the back has our priority or higher
        // then we can just put us on the back
        if (dma_queue.empty() || (dma_queue.back()->priority >= this->priority))
          dma_queue.push_back(this);
        else
        {
          // Uncommon case: go through the list until we find someone
          // who has a priority lower than ours.  We know they
          // exist since we saw them on the back.
          bool inserted = false;
          for (std::list<DmaRequest*>::iterator it = dma_queue.begin();
                it != dma_queue.end(); it++)
          {
            if ((*it)->priority < this->priority)
            {
              dma_queue.insert(it, this);
              inserted = true;
              break;
            }
          }
          // Technically don't need this, but just to be safe
          if (!inserted)
            dma_queue.push_back(this);
        }
	state = STATE_QUEUED;
	gasnett_cond_signal(&queue_condvar);
	gasnet_hsl_unlock(&queue_mutex);
	log_dma.info("request %p enqueued", this);

#ifdef LEGION_LOGGING
	log_timing_event(Processor::NO_PROC, after_copy, COPY_READY);
#endif
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

    class InstPairCopier {
    public:
      virtual void copy_field(int src_index, int dst_index, int elem_count,
                              unsigned offset_index) = 0;
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
        for (unsigned idx = 0; idx < oas_vec.size(); idx++)
        {
          find_field_start(src_idata->field_sizes, oas_vec[idx].src_offset,
                            oas_vec[idx].size, src_start[idx], src_size[idx]);
          find_field_start(dst_idata->field_sizes, oas_vec[idx].dst_offset,
                            oas_vec[idx].size, dst_start[idx], dst_size[idx]);
        }
      }

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

	    span_copier->copy_span(src_start, dst_start, bytes);
	    //src_mem->get_bytes(src_start, buffer, bytes);
	    //dst_mem->put_bytes(dst_start, buffer, bytes);
	  }
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
    };

    class RemoteWriteInstPairCopier : public InstPairCopier {
    public:
      RemoteWriteInstPairCopier(RegionInstance src_inst, RegionInstance dst_inst,
                                OASVec &_oas_vec)
	: src_acc(src_inst.get_accessor()), dst_acc(dst_inst.get_accessor()),
          oas_vec(_oas_vec)
      {}

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
      static MemPairCopier* create_copier(Memory src_mem, Memory dst_mem);

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

      ~BufferedMemPairCopier(void)
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
	//printf("buffered copy of %zd bytes (%x:%zd -> %x:%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
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

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes,
		     size_t lines, off_t src_stride, off_t dst_stride)
      {
	for(size_t i = 0; i < lines; i++) {
	  memcpy(dst_base + dst_offset, src_base + src_offset, bytes);
	  dst_offset += dst_stride;
	  src_offset += src_stride;
	}
      }

    protected:
      const char *src_base;
      char *dst_base;
    };

    // a MemPairCopier that keeps a list of events for component copies and doesn't trigger
    //  the completion event until they're all done
    class DelayedMemPairCopier : public MemPairCopier {
    public:
      DelayedMemPairCopier(void) {}
      
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
     
    class GPUtoFBMemPairCopier : public DelayedMemPairCopier {
    public:
      GPUtoFBMemPairCopier(Memory _src_mem, GPUProcessor *_gpu)
	: gpu(_gpu)
      {
	Memory::Impl *src_impl = _src_mem.impl();
	src_base = (const char *)(src_impl->get_direct_ptr(0, src_impl->size));
	assert(src_base);
      }

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
		     size_t lines, off_t src_stride, off_t dst_stride)
      {
	for(size_t i = 0; i < lines; i++) {
	  copy_span(dst_offset, src_offset, bytes);
	  dst_offset += dst_stride;
	  src_offset += src_stride;
	}
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
		     size_t lines, off_t src_stride, off_t dst_stride)
      {
	for(size_t i = 0; i < lines; i++) {
	  copy_span(dst_offset, src_offset, bytes);
	  dst_offset += dst_stride;
	  src_offset += src_stride;
	}
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
		     size_t lines, off_t src_stride, off_t dst_stride)
      {
	for(size_t i = 0; i < lines; i++) {
	  copy_span(dst_offset, src_offset, bytes);
	  dst_offset += dst_stride;
	  src_offset += src_stride;
	}
      }

    protected:
      GPUProcessor *gpu;
    };
     
    class RemoteWriteMemPairCopier : public MemPairCopier {
    public:
      RemoteWriteMemPairCopier(Memory _src_mem, Memory _dst_mem)
      {
	src_mem = _src_mem.impl();
	src_base = (const char *)(src_mem->get_direct_ptr(0, src_mem->size));
	assert(src_base);

	dst_mem = _dst_mem.impl();
      }

      ~RemoteWriteMemPairCopier(void)
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
	std::vector<std::pair<const char *, size_t> > src_spans;
      };

      static const size_t TARGET_XFER_SIZE = 1 << 20;

      void copy_span(off_t src_offset, off_t dst_offset, size_t bytes)
      {
	//printf("remote write of %zd bytes (%x:%zd -> %x:%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);

	if(bytes >= TARGET_XFER_SIZE) {
	  // large enough that we can transfer it by itself
#ifdef DEBUG_REMOTE_WRITES
	  printf("remote write of %zd bytes (%x:%zd -> %x:%zd)\n", bytes, src_mem->me.id, src_offset, dst_mem->me.id, dst_offset);
#endif
	  do_remote_write(dst_mem->me, dst_offset, src_base + src_offset, bytes, Event::NO_EVENT, false /* no copy */);
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
	    std::pair<const char *, size_t>& last_span = g->src_spans.back();
	    if((last_span.first + last_span.second) == (src_base + src_offset)) {
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
      }

      void copy_gather(const PendingGather *g, Event trigger = Event::NO_EVENT)
      {
        // special case: if there's only one source span, it's not actually
        //   a gather (so we don't need to make a copy)
        if(g->src_spans.size() == 1) {
          const std::pair<const char *, size_t>& span = *(g->src_spans.begin());
#ifdef DEBUG_REMOTE_WRITES
	  printf("remote write of %zd bytes (singleton %x:%zd -> %x:%zd)\n", span.second, src_mem->me.id, span.first - src_base, dst_mem->me.id, g->dst_start);
          printf("  datb[%p]: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                 span.first,
                 ((unsigned *)(span.first))[0], ((unsigned *)(span.first))[1],
                 ((unsigned *)(span.first))[2], ((unsigned *)(span.first))[3],
                 ((unsigned *)(span.first))[4], ((unsigned *)(span.first))[5],
                 ((unsigned *)(span.first))[6], ((unsigned *)(span.first))[7]);
#endif
	  do_remote_write(dst_mem->me, g->dst_start, span.first, span.second, trigger, false /* no copy */);
          return;
        }

	// build a buffer we can copy in one block to the destination
	char *buffer = new char[g->dst_size];
	char *pos = buffer;

	// copy each of the spans
	for(std::vector<std::pair<const char *, size_t> >::const_iterator it = g->src_spans.begin();
	    it != g->src_spans.end();
	    it++) {
	  memcpy(pos, it->first, it->second);
	  pos += it->second;
	}

#ifdef DEBUG_REMOTE_WRITES
	printf("remote write of %zd bytes (gather -> %x:%zd), trigger=%x/%d\n", g->dst_size, dst_mem->me.id, g->dst_start, trigger.id, trigger.gen);
#endif
	do_remote_write(dst_mem->me, g->dst_start, buffer, g->dst_size, trigger, true /* make copy! */);

	delete[] buffer;
      }

      virtual void flush(Event after_copy)
      {
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
	  printf("remote write fence: %x/%d\n", after_copy.id, after_copy.gen);
#endif
	  do_remote_write(dst_mem->me, 0, 0, 0, after_copy);
#endif
	  delete g;
	} else {
	  // didn't have any pending copies, but if we have an event to trigger, send that
	  //  to the remote side as a zero-size copy to push the data in front of it
	  if(after_copy.exists()) {
#ifdef DEBUG_REMOTE_WRITES
	    printf("remote write fence: %x/%d\n", after_copy.id, after_copy.gen);
#endif
	    do_remote_write(dst_mem->me, 0, 0, 0, after_copy);
	  }
	}
      }

    protected:
      Memory::Impl *src_mem, *dst_mem;
      const char *src_base;
      std::map<off_t, PendingGather *> gathers;
    };
     
    MemPairCopier *MemPairCopier::create_copier(Memory src_mem, Memory dst_mem)
    {
      Memory::Impl *src_impl = src_mem.impl();
      Memory::Impl *dst_impl = dst_mem.impl();

      Memory::Impl::MemoryKind src_kind = src_impl->kind;
      Memory::Impl::MemoryKind dst_kind = dst_impl->kind;

      log_dma.info("copier: %x(%d) -> %x(%d)", src_mem.id, src_kind, dst_mem.id, dst_kind);

      // can we perform simple memcpy's?
      if(((src_kind == Memory::Impl::MKIND_SYSMEM) || (src_kind == Memory::Impl::MKIND_ZEROCOPY)) &&
	 ((dst_kind == Memory::Impl::MKIND_SYSMEM) || (dst_kind == Memory::Impl::MKIND_ZEROCOPY))) {
	return new MemcpyMemPairCopier(src_mem, dst_mem);
      }

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
        assert(src_gpu == dst_gpu);
	return new GPUinFBMemPairCopier(src_gpu);
      }

      // try as many things as we can think of
      if(dst_kind == Memory::Impl::MKIND_REMOTE) {
        assert(src_kind != Memory::Impl::MKIND_REMOTE);
        return new RemoteWriteMemPairCopier(src_mem, dst_mem);
      }

      // fallback
      return new BufferedMemPairCopier(src_mem, dst_mem);
    }

    template <unsigned DIM>
    void DmaRequest::perform_dma_rect(MemPairCopier *mpc)
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

	// iterate by output rectangle first - this gives us a chance to gather data when linearizations
	//  don't match up
	for(Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > dso(orig_rect, *dst_linearization); dso; dso++) {
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
        // Dammit Sean stop leaking things!
        delete ipc;
      }
    }

#ifdef LEGION_LOGGING
    class CopyCompletionLogger : public Event::Impl::EventWaiter {
    public:
      CopyCompletionLogger(Event _event) : event(_event) {}

      virtual bool event_triggered(void)
      {
	log_timing_event(Processor::NO_PROC, event, COPY_END);
	return true;
      }

      virtual void print_info(void)
      {
	printf("copy completion logger - %x/%d\n", event.id, event.gen);
      }

    protected:
      Event event;
    };
#endif

    void DmaRequest::perform_dma(void)
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
	  assert(0);
	  break;
	}

	// rectangle cases
      case 1: perform_dma_rect<1>(mpc); break;
      case 2: perform_dma_rect<2>(mpc); break;
      case 3: perform_dma_rect<3>(mpc); break;

      default: assert(0);
      };

      log_dma.info("dma request %p finished - %x[%d]->%x[%d]:%d (+%zd) (%x) %x/%d %x/%d",
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
	  dma_queue.pop_front();

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
      for(int i = 0; i < count; i++) {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	if(proc_assignment)
	  proc_assignment->bind_thread(-1, &attr, "DMA worker");
	CHECK_PTHREAD( pthread_create(&worker_threads[i], 0, 
				      dma_worker_thread_loop, 0) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
      }
    }
    
    Event enqueue_dma(const Domain& domain,
		      OASByInst *oas_by_inst,
		      ReductionOpID redop_id, bool red_fold,
		      Event before_copy,
		      Event after_copy,
                      int priority)
    {
      // special case - if we have everything we need, we can consider doing the
      //   copy immediately
      DetailedTimer::ScopedPush sp(TIME_COPY);

      DmaRequest *r = new DmaRequest(domain, oas_by_inst, redop_id, red_fold,
				     before_copy, after_copy, priority);

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

#ifdef LEGION_LOGGING
	log_timing_event(Processor::NO_PROC, after_copy, COPY_INIT);
#endif

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

      const int *idata = (const int *)data;

      Domain domain;
      idata = domain.deserialize(idata);

      OASByInst *oas_by_inst = new OASByInst;

      int priority = 0;

      while(((idata - ((const int *)data))*sizeof(int)) < msglen) {
	RegionInstance src_inst = ID((unsigned)*idata++).convert<RegionInstance>();
	RegionInstance dst_inst = ID((unsigned)*idata++).convert<RegionInstance>();
	InstPair ip(src_inst, dst_inst);

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
      assert(((idata - ((const int *)data))*sizeof(int)) == msglen);

      log_dma.info("received remote copy request: instpairs=%zd(%x->%x) before=%x/%d after=%x/%d",
		   oas_by_inst->size(), oas_by_inst->begin()->first.first.id, oas_by_inst->begin()->first.second.id,
		   args.before_copy.id, args.before_copy.gen,
		   args.after_copy.id, args.after_copy.gen);

      enqueue_dma(domain, oas_by_inst, args.redop_id, args.red_fold, args.before_copy, args.after_copy, priority);
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

    template <typename T> T min(T a, T b) { return (a < b) ? a : b; }

    Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		       const std::vector<CopySrcDstField>& dsts,
		       Event wait_on,
		       ReductionOpID redop_id, bool red_fold) const
    {
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

      log_dma.info("copy: %zd distinct src/dst mem pairs, is=%x", oas_by_mem.size(), is_id);
      assert(redop_id == 0);

      for(OASByMem::const_iterator it = oas_by_mem.begin(); it != oas_by_mem.end(); it++) {
	Memory src_mem = it->first.first;
	Memory dst_mem = it->first.second;
	OASByInst *oas_by_inst = it->second;

	// ask which node should perform the copy
	int dma_node = select_dma_node(src_mem, dst_mem, redop_id, red_fold);
	log_dma.info("copy: srcmem=%x dstmem=%x node=%d", src_mem.id, dst_mem.id, dma_node);

	if(dma_node == gasnet_mynode()) {
	  log_dma.info("performing copy on local node");

          // If either one of the instances is in GPU memory increase priority
          int priority = 0;
          if (src_mem.impl()->kind == Memory::Impl::MKIND_GPUFB)
            priority = 1;
          else if (dst_mem.impl()->kind == Memory::Impl::MKIND_GPUFB)
            priority = 1;

	  Event ev = enqueue_dma(*this, oas_by_inst, redop_id, red_fold, wait_on, Event::NO_EVENT, priority);
	  finish_events.insert(ev);
	} else {
	  // need to send dma to remote node for processing, so we need a completion event
	  Event ev = Event::Impl::create_event();

	  RemoteCopyArgs args;
	  args.redop_id = redop_id;
	  args.red_fold = red_fold;
	  args.before_copy = wait_on;
	  args.after_copy = ev;

	  static const size_t MAX_MESSAGE_SIZE = 2048;
	  int msgdata[MAX_MESSAGE_SIZE / sizeof(int)];

	  // domain info goes first
	  int *msgptr = serialize(msgdata);

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

	  // we're done with our copy of oas_by_inst now
	  delete oas_by_inst;

	  size_t msglen = ((const char *)msgptr) - ((const char *)msgdata);
	  assert(msglen <= MAX_MESSAGE_SIZE);
	  log_dma.info("performing copy on remote node (%d), event=%x/%d", dma_node, args.after_copy.id, args.after_copy.gen);
	  RemoteCopyMessage::request(dma_node, args, msgdata, msglen, PAYLOAD_COPY);
	  
	  finish_events.insert(ev);
	}
      }

      // final event is merge of all individual copies' events
      return Event::Impl::merge_events(finish_events);
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
