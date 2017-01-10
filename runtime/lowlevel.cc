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

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//  WARNING: Do not add any new code to this file - its days are numbered.
//   It only consists of a few last functions that have not been moved to new
//   homes yet.
//
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


#include "lowlevel.h"
#include "lowlevel_impl.h"
#include "accessor.h"

#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

using namespace LegionRuntime::Accessor;

#include "lowlevel_dma.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <dirent.h>

#include <signal.h>
#include <unistd.h>
#if defined(DEADLOCK_TRACE)
#include <execinfo.h>
#include <cxxabi.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "realm/timers.h"

// Implementation of Detailed Timer
namespace LegionRuntime {
  namespace LowLevel {

    Logger::Category log_mutex("mutex");
    Logger::Category log_timer("timer");
    Logger::Category log_machine("machine");
#ifdef EVENT_GRAPH_TRACE
    Logger::Category log_event_graph("graph");
#endif

#ifdef EVENT_GRAPH_TRACE
    Event find_enclosing_termination_event(void)
    {
      // TODO: fix this for new threading setup
      HI I AM A SYNTAX ERROR
    }
#endif

    void show_event_waiters(std::ostream& os)
    {
      os << "PRINTING ALL PENDING EVENTS:\n";
      for(unsigned i = 0; i < gasnet_nodes(); i++) {
	Node *n = &get_runtime()->nodes[i];
        // Iterate over all the events and get their implementations
        for (unsigned long j = 0; j < n->events.max_entries(); j++) {
          if (!n->events.has_entry(j))
            continue;
	  GenEventImpl *e = n->events.lookup_entry(j, i/*node*/);
	  AutoHSLLock a2(e->mutex);

	  // print anything with either local or remote waiters
	  if(e->current_local_waiters.empty() &&
	     e->future_local_waiters.empty() &&
	     e->remote_waiters.empty())
	    continue;

	  os << "Event " << e->me <<": gen=" << e->generation
	     << " subscr=" << e->gen_subscribed
	     << " local=" << e->current_local_waiters.size()
	     << "+" << e->future_local_waiters.size()
	     << " remote=" << e->remote_waiters.size() << "\n";
	  for(std::vector<EventWaiter *>::const_iterator it = e->current_local_waiters.begin();
	      it != e->current_local_waiters.end();
	      it++) {
	    os << "  [" << (e->generation+1) << "] L:" << (*it) << " - ";
	    (*it)->print(os);
	    os << "\n";
	  }
	  for(std::map<EventImpl::gen_t, std::vector<EventWaiter *> >::const_iterator it = e->future_local_waiters.begin();
	      it != e->future_local_waiters.end();
	      it++) {
	    for(std::vector<EventWaiter *>::const_iterator it2 = it->second.begin();
		it2 != it->second.end();
		it2++) {
	      os << "  [" << (it->first) << "] L:" << (*it2) << " - ";
	      (*it2)->print(os);
	      os << "\n";
	    }
	  }
	  // for(std::map<Event::gen_t, NodeMask>::const_iterator it = e->remote_waiters.begin();
	  //     it != e->remote_waiters.end();
	  //     it++) {
	  //   fprintf(f, "  [%d] R:", it->first);
	  //   for(int k = 0; k < MAX_NUM_NODES; k++)
	  //     if(it->second.is_set(k))
	  // 	fprintf(f, " %d", k);
	  //   fprintf(f, "\n");
	  // }
	}
        for (unsigned long j = 0; j < n->barriers.max_entries(); j++) {
          if (!n->barriers.has_entry(j))
            continue;
          BarrierImpl *b = n->barriers.lookup_entry(j, i/*node*/); 
          AutoHSLLock a2(b->mutex);
          // skip any barriers with no waiters
          if (b->generations.empty())
            continue;

	  os << "Barrier " << b->me << ": gen=" << b->generation
	     << " subscr=" << b->gen_subscribed << "\n";
          for (std::map<EventImpl::gen_t, BarrierImpl::Generation*>::const_iterator git = 
                b->generations.begin(); git != b->generations.end(); git++)
          {
            const std::vector<EventWaiter*> &waiters = git->second->local_waiters;
            for (std::vector<EventWaiter*>::const_iterator it = 
                  waiters.begin(); it != waiters.end(); it++)
            { 
	      os << "  [" << (git->first) << "] L:" << (*it) << " - ";
	      (*it)->print(os);
	      os << "\n";
            }
          }
        }
      }

      // TODO - pending barriers
#if 0
      // // convert from events to barriers
      // fprintf(f,"PRINTING ALL PENDING EVENTS:\n");
      // for(int i = 0; i < gasnet_nodes(); i++) {
      // 	Node *n = &get_runtime()->nodes[i];
      //   // Iterate over all the events and get their implementations
      //   for (unsigned long j = 0; j < n->events.max_entries(); j++) {
      //     if (!n->events.has_entry(j))
      //       continue;
      // 	  EventImpl *e = n->events.lookup_entry(j, i/*node*/);
      // 	  AutoHSLLock a2(e->mutex);

      // 	  // print anything with either local or remote waiters
      // 	  if(e->local_waiters.empty() && e->remote_waiters.empty())
      // 	    continue;

      //     fprintf(f,"Event " IDFMT ": gen=%d subscr=%d local=%zd remote=%zd\n",
      // 		  e->me.id, e->generation, e->gen_subscribed, 
      // 		  e->local_waiters.size(), e->remote_waiters.size());
      // 	  for(std::map<Event::gen_t, std::vector<EventWaiter *> >::iterator it = e->local_waiters.begin();
      // 	      it != e->local_waiters.end();
      // 	      it++) {
      // 	    for(std::vector<EventWaiter *>::iterator it2 = it->second.begin();
      // 		it2 != it->second.end();
      // 		it2++) {
      // 	      fprintf(f, "  [%d] L:%p ", it->first, *it2);
      // 	      (*it2)->print_info(f);
      // 	    }
      // 	  }
      // 	  for(std::map<Event::gen_t, NodeMask>::const_iterator it = e->remote_waiters.begin();
      // 	      it != e->remote_waiters.end();
      // 	      it++) {
      // 	    fprintf(f, "  [%d] R:", it->first);
      // 	    for(int k = 0; k < MAX_NUM_NODES; k++)
      // 	      if(it->second.is_set(k))
      // 		fprintf(f, " %d", k);
      // 	    fprintf(f, "\n");
      // 	  }
      // 	}
      // }
#endif

      os << "DONE\n";
      os.flush();
    }



#ifdef DETAILED_TIMING
#endif

    void gasnet_barrier(void)
    {
      gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
      gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
    }

    template<typename ITEM>
    /*static*/ void Tracer<ITEM>::dump_trace(const char *filename, bool append)
    {
      // each node dumps its stuff in order (using barriers to keep things
      // separate) - nodes other than zero ALWAYS append
      gasnet_barrier();

      for(int i = 0; i < gasnet_nodes(); i++) {
	if(i == gasnet_mynode()) {
	  int fd = open(filename, (O_WRONLY |
				   O_CREAT |
				   ((append || (i > 0)) ? O_APPEND : O_TRUNC)),
			0666);
	  assert(fd >= 0);

	  TraceBlock *block = get_first_block();
	  size_t total = 0;
	  while(block) {
	    if(block->cur_size > 0) {
	      size_t act_size = block->cur_size;
	      if(act_size > block->max_size) act_size = block->max_size;
              total += act_size;

              size_t bytes_to_write = act_size * (sizeof(double) + sizeof(unsigned) + sizeof(ITEM));
	      void *fitems = malloc(bytes_to_write);
              char *ptr = (char*)fitems;

	      for(size_t i = 0; i < act_size; i++) {
		*((double*)ptr) = block->start_time + (block->items[i].time_units /
                                                        block->time_mult);
                ptr += sizeof(double);
                *((unsigned*)ptr) = gasnet_mynode();
                ptr += sizeof(unsigned);
                memcpy(ptr,&(block->items[i]),sizeof(ITEM));
                ptr += sizeof(ITEM);
	      }

	      ssize_t bytes_written = write(fd, fitems, bytes_to_write);
	      assert(bytes_written == (ssize_t)bytes_to_write);

              free(fitems);
	    }

	    block = block->next;
	  }

	  close(fd);

	  printf("%zd trace items dumped to \"%s\"\n",
		 total, filename);
	}

	gasnet_barrier();
      }
    }

#if 0
    size_t IndexSpaceImpl::instance_size(const ReductionOpUntyped *redop /*= 0*/, off_t list_size /*= -1*/)
    {
      Realm::StaticAccess<IndexSpaceImpl> data(this);
      assert(data->num_elmts > 0);
      size_t elmts = data->last_elmt - data->first_elmt + 1;
      size_t bytes;
      if(redop) {
	if(list_size >= 0)
	  bytes = list_size * redop->sizeof_list_entry;
	else
	  bytes = elmts * redop->sizeof_rhs;
      } else {
	assert(data->elmt_size > 0);
	bytes = elmts * data->elmt_size;
      }
      return bytes;
    }

    off_t IndexSpaceImpl::instance_adjust(const ReductionOpUntyped *redop /*= 0*/)
    {
      Realm::StaticAccess<IndexSpaceImpl> data(this);
      assert(data->num_elmts > 0);
      off_t elmt_adjust = -(off_t)(data->first_elmt);
      off_t adjust;
      if(redop) {
	adjust = elmt_adjust * redop->sizeof_rhs;
      } else {
	assert(data->elmt_size > 0);
	adjust = elmt_adjust * data->elmt_size;
      }
      return adjust;
    }
#endif

    


    ///////////////////////////////////////////////////
    // RegionInstance


#if 0
#endif

  };
};

namespace Realm {

  using namespace LegionRuntime::LowLevel;


};

namespace LegionRuntime {
  namespace LowLevel {


    ///////////////////////////////////////////////////
    // Reservations 

    //    /*static*/ ReservationImpl *ReservationImpl::first_free = 0;
    //    /*static*/ GASNetHSL ReservationImpl::freelist_mutex;


  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;


};

namespace LegionRuntime {
  namespace LowLevel {










    ///////////////////////////////////////////////////
    // Task

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    ///////////////////////////////////////////////////
    // Processor

};

namespace LegionRuntime {
  namespace LowLevel {




    ///////////////////////////////////////////////////
    // Runtime


#ifdef DEADLOCK_TRACE
    void RuntimeImpl::add_thread(const pthread_t *thread)
    {
      unsigned idx = __sync_fetch_and_add(&next_thread,1);
      assert(idx < MAX_NUM_THREADS);
      all_threads[idx] = *thread;
      thread_counts[idx] = 0;
    }
#endif

  };
};

namespace Realm {

    ///////////////////////////////////////////////////
    // RegionMetaData


#if 0
    RegionInstance IndexSpace::create_instance_untyped(Memory memory,
									 ReductionOpID redopid) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redopid];

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

      size_t inst_bytes = impl()->instance_size(redop);
      off_t inst_adjust = impl()->instance_adjust(redop);

      RegionInstance i = m_impl->create_instance(get_index_space(), inst_bytes, 
							inst_adjust, redopid);
      log_meta.info("instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd adjust=%zd redop=%d",
	       this->id, memory.id, i.id, inst_bytes, inst_adjust, redopid);
      return i;
    }

    RegionInstance IndexSpace::create_instance_untyped(Memory memory,
									 ReductionOpID redopid,
									 off_t list_size,
									 RegionInstance parent_inst) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redopid];

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

      size_t inst_bytes = impl()->instance_size(redop, list_size);
      off_t inst_adjust = impl()->instance_adjust(redop);

      RegionInstance i = m_impl->create_instance(*this, inst_bytes, 
							inst_adjust, redopid,
							list_size, parent_inst);
      log_meta.info("instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd adjust=%zd redop=%d list_size=%zd parent_inst=" IDFMT "",
	       this->id, memory.id, i.id, inst_bytes, inst_adjust, redopid,
	       list_size, parent_inst.id);
      return i;
    }
#endif

    ///////////////////////////////////////////////////
    // Element Masks

    ///////////////////////////////////////////////////
    // Region Allocators

};

namespace LegionRuntime {
  namespace LowLevel {


    ///////////////////////////////////////////////////
    // Region Instances

#if 0
    class DeferredCopy : public EventWaiter {
    public:
      DeferredCopy(RegionInstance _src, RegionInstance _target,
		   IndexSpace _region,
		   size_t _elmt_size, size_t _bytes_to_copy, Event _after_copy)
	: src(_src), target(_target), region(_region),
	  elmt_size(_elmt_size), bytes_to_copy(_bytes_to_copy), 
	  after_copy(_after_copy) {}

      virtual void event_triggered(void)
      {
	RegionInstanceImpl::copy(src, target, region, 
					  elmt_size, bytes_to_copy, after_copy);
      }

      virtual void print_info(void)
      {
	printf("deferred copy: src=" IDFMT " tgt=" IDFMT " region=" IDFMT " after=" IDFMT "/%d\n",
	       src.id, target.id, region.id, after_copy.id, after_copy.gen);
      }

    protected:
      RegionInstance src, target;
      IndexSpace region;
      size_t elmt_size, bytes_to_copy;
      Event after_copy;
    };
#endif



#ifdef OLD_RANGE_EXECUTORS
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

      class GasnetPutReduce : public GasnetPut {
      public:
	GasnetPutReduce(MemoryImpl *_tgt_mem, off_t _tgt_offset,
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
	  // if we got any spans, the last one is still waiting to go out
	  if(span_count > 0)
	    really_do_span(true);

	  return event;
	}

      protected:
	void really_do_span(bool last)
	{
	  off_t byte_offset = prev_offset * elmt_size;
	  size_t byte_count = prev_count * elmt_size;

	  // if we don't have an event for our completion, we need one now
	  if(!event.exists())
	    event = GenEventImpl::create_event();

	  RemoteWriteArgs args;
	  args.mem = tgt_mem;
	  args.offset = tgt_offset + byte_offset;
	  if(last)
	    args.event = event;
	  else
	    args.event = Event::NO_EVENT;
	
	  RemoteWriteMessage::request(ID(tgt_mem).node(), args,
				      src_ptr + byte_offset,
				      byte_count,
				      PAYLOAD_KEEP);
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
#endif

#if 0
    /*static*/ Event RegionInstanceImpl::copy(RegionInstance src, 
						RegionInstance target,
						IndexSpace is,
						size_t elmt_size,
						size_t bytes_to_copy,
						Event after_copy /*= Event::NO_EVENT*/)
    {
      return(enqueue_dma(is, src, target, elmt_size, bytes_to_copy,
			 Event::NO_EVENT, after_copy));
    }
#endif

#ifdef OLD_COPIES
    Event RegionInstance::copy_to_untyped(RegionInstance target, 
						 Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstanceImpl *src_impl = impl();
      RegionInstanceImpl *dst_impl = target.impl();

      // figure out which of src or target is the smaller - punt if one
      //  is not a direct ancestor of the other
      IndexSpace src_region = Realm::StaticAccess<RegionInstanceImpl>(src_impl)->region;
      IndexSpace dst_region = Realm::StaticAccess<RegionInstanceImpl>(dst_impl)->region;

      if(src_region == dst_region) {
	log_copy.info("region match: " IDFMT "\n", src_region.id);
	return copy_to_untyped(target, src_region, wait_on);
      } else
	if(src_region.impl()->is_parent_of(dst_region)) {
	  log_copy.info("src is parent of dst: " IDFMT " >= " IDFMT "\n", src_region.id, dst_region.id);
	  return copy_to_untyped(target, dst_region, wait_on);
	} else
	  if(dst_region.impl()->is_parent_of(src_region)) {
	    log_copy.info("dst is parent of src: " IDFMT " >= " IDFMT "\n", dst_region.id, src_region.id);
	    return copy_to_untyped(target, src_region, wait_on);
	  } else {
	    assert(0);
	  }
    }

    Event RegionInstance::copy_to_untyped(RegionInstance target,
						 const ElementMask &mask,
						 Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      assert(0);
    }

    Event RegionInstance::copy_to_untyped(RegionInstance target,
						 IndexSpace region,
						 Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      RegionInstanceImpl *src_impl = impl();
      RegionInstanceImpl *dst_impl = target.impl();

      // the region we're being asked to copy must be a subregion (or same
      //  region) of both the src and dst instance's regions
      IndexSpace src_region = Realm::StaticAccess<RegionInstanceImpl>(src_impl)->region;
      IndexSpace dst_region = Realm::StaticAccess<RegionInstanceImpl>(dst_impl)->region;

      log_copy.info("copy_to_untyped(" IDFMT "(" IDFMT "), " IDFMT "(" IDFMT "), " IDFMT ", " IDFMT "/%d)",
		    id, src_region.id,
		    target.id, dst_region.id, 
		    region.id, wait_on.id, wait_on.gen);

      assert(src_region.impl()->is_parent_of(region));
      assert(dst_region.impl()->is_parent_of(region));

      MemoryImpl *src_mem = src_impl->memory.impl();
      MemoryImpl *dst_mem = dst_impl->memory.impl();

      log_copy.debug("copy instance: " IDFMT " (%d) -> " IDFMT " (%d), wait=" IDFMT "/%d", id, src_mem->kind, target.id, dst_mem->kind, wait_on.id, wait_on.gen);

      size_t bytes_to_copy, elmt_size;
      {
	Realm::StaticAccessf<RegionInstanceImpl> src_data(src_impl);
	bytes_to_copy = src_data->region.impl()->instance_size();
	elmt_size = (src_data->is_reduction ?
		       get_runtime()->reduce_op_table[src_data->redopid]->sizeof_rhs :
		       Realm::StaticAccess<IndexSpaceImpl>(src_data->region.impl())->elmt_size);
      }
      log_copy.debug("COPY " IDFMT " (%d) -> " IDFMT " (%d) - %zd bytes (%zd)", id, src_mem->kind, target.id, dst_mem->kind, bytes_to_copy, elmt_size);

      // check to see if we can access the source memory - if not, we'll send
      //  the request to somebody who can
      if(src_mem->kind == MemoryImpl::MKIND_REMOTE) {
	// plan B: if one side is remote, try delegating to the node
	//  that owns the other side of the copy
	unsigned delegate = ID(src_impl->memory).node();
	assert(delegate != gasnet_mynode());

	log_copy.info("passsing the buck to node %d for " IDFMT "->" IDFMT " copy",
		      delegate, src_mem->me.id, dst_mem->me.id);
	Event after_copy = GenEventImpl::create_event();
	RemoteCopyArgs args;
	args.source = *this;
	args.target = target;
	args.region = region;
	args.elmt_size = elmt_size;
	args.bytes_to_copy = bytes_to_copy;
	args.before_copy = wait_on;
	args.after_copy = after_copy;
	RemoteCopyMessage::request(delegate, args);
	
	return after_copy;
      }

      // another interesting case: if the destination is remote, and the source
      //  is gasnet, then the destination can read the source itself
      if((src_mem->kind == MemoryImpl::MKIND_GLOBAL) &&
	 (dst_mem->kind == MemoryImpl::MKIND_REMOTE)) {
	unsigned delegate = ID(dst_impl->memory).node();
	assert(delegate != gasnet_mynode());

	log_copy.info("passsing the buck to node %d for " IDFMT "->" IDFMT " copy",
		      delegate, src_mem->me.id, dst_mem->me.id);
	Event after_copy = GenEventImpl::create_event();
	RemoteCopyArgs args;
	args.source = *this;
	args.target = target;
	args.region = region;
	args.elmt_size = elmt_size;
	args.bytes_to_copy = bytes_to_copy;
	args.before_copy = wait_on;
	args.after_copy = after_copy;
	RemoteCopyMessage::request(delegate, args);
	
	return after_copy;
      }

      if(!wait_on.has_triggered()) {
	Event after_copy = GenEventImpl::create_event();
	log_copy.debug("copy deferred: " IDFMT " (%d) -> " IDFMT " (%d), wait=" IDFMT "/%d after=" IDFMT "/%d", id, src_mem->kind, target.id, dst_mem->kind, wait_on.id, wait_on.gen, after_copy.id, after_copy.gen);
	wait_on.impl()->add_waiter(wait_on,
				   new DeferredCopy(*this, target,
						    region,
						    elmt_size,
						    bytes_to_copy,
						    after_copy));
	return after_copy;
      }

      // we can do the copy immediately here
      return RegionInstanceImpl::copy(*this, target, region,
					       elmt_size, bytes_to_copy);
    }
#endif

#if 0
#ifdef POINTER_CHECKS
    void RegionAccessor<AccessorGeneric>::verify_access(unsigned ptr) const
    {
      ((RegionInstanceImpl *)internal_data)->verify_access(ptr); 
    }

    void RegionAccessor<AccessorArray>::verify_access(unsigned ptr) const
    {
      ((RegionInstanceImpl *)impl_ptr)->verify_access(ptr);
    }
#endif

    void RegionAccessor<AccessorGeneric>::get_untyped(int index, off_t byte_offset, void *dst, size_t size) const
    {
      ((RegionInstanceImpl *)internal_data)->get_bytes(index, byte_offset, dst, size);
    }

    void RegionAccessor<AccessorGeneric>::put_untyped(int index, off_t byte_offset, const void *src, size_t size) const
    {
      ((RegionInstanceImpl *)internal_data)->put_bytes(index, byte_offset, src, size);
    }

    bool RegionAccessor<AccessorGeneric>::is_reduction_only(void) const
    {
      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);
      return(i_data->is_reduction);
    }

    RegionAccessor<AccessorGeneric> RegionAccessor<AccessorGeneric>::get_field_accessor(off_t offset, size_t size) const
    {
      return RegionAccessor<AccessorGeneric>(internal_data, 
					     field_offset + offset);
    }

    template <>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorGeneric>(void) const
    { return true; }
    
    template <>
    RegionAccessor<AccessorGeneric> RegionAccessor<AccessorGeneric>::convert<AccessorGeneric>(void) const
    { return *this; }

    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorArray>(void) const
    {
      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      MemoryImpl *m_impl = i_impl->memory.impl();

      // make sure it's not a reduction fold-only instance
      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);
      if(i_data->is_reduction) return false;

      // only things in local memory (SYSMEM or ZC) can be converted to
      //   array accessors
      if(m_impl->kind == MemoryImpl::MKIND_SYSMEM) return true;
      if(m_impl->kind == MemoryImpl::MKIND_ZEROCOPY) return true;
      return false;
    }
#endif

#ifdef OLD_ACCESSORS
    template<>
    RegionAccessor<AccessorArray> RegionAccessor<AccessorGeneric>::convert<AccessorArray>(void) const
    {
      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      MemoryImpl *m_impl = i_impl->memory.impl();

      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);

      assert(!i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == MemoryImpl::MKIND_SYSMEM) {
	Realm::LocalCPUMemory *lcm = (Realm::LocalCPUMemory *)m_impl;
	char *inst_base = lcm->base + i_data->access_offset;
	RegionAccessor<AccessorArray> ria(inst_base);
#ifdef POINTER_CHECKS
        ria.set_impl(i_impl);
#endif
	return ria;
      }

      if(m_impl->kind == MemoryImpl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	char *inst_base = zcm->cpu_base + i_data->access_offset;
	RegionAccessor<AccessorArray> ria(inst_base);
#ifdef POINTER_CHECKS
        ria.set_impl(i_impl); 
#endif
	return ria;
      }

      assert(0);
    }
    
    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      MemoryImpl *m_impl = i_impl->memory.impl();

      // make sure it's a reduction fold-only instance
      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);
      if(!i_data->is_reduction) return false;
      if(i_data->red_list_size >= 0) return false;

      // only things in local memory (SYSMEM or ZC) can be converted to
      //   array accessors
      if(m_impl->kind == MemoryImpl::MKIND_SYSMEM) return true;
      if(m_impl->kind == MemoryImpl::MKIND_ZEROCOPY) return true;
      return false;
    }

    template<>
    RegionAccessor<AccessorArrayReductionFold> RegionAccessor<AccessorGeneric>::convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      MemoryImpl *m_impl = i_impl->memory.impl();

      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);

      assert(i_data->is_reduction);
      assert(i_data->red_list_size < 0);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == MemoryImpl::MKIND_SYSMEM) {
	Realm::LocalCPUMemory *lcm = (Realm::LocalCPUMemory *)m_impl;
	char *inst_base = lcm->base + i_data->access_offset;
	RegionAccessor<AccessorArrayReductionFold> ria(inst_base);
	return ria;
      }

      if(m_impl->kind == MemoryImpl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	char *inst_base = zcm->cpu_base + i_data->access_offset;
	RegionAccessor<AccessorArrayReductionFold> ria(inst_base);
	return ria;
      }

      assert(0);
    }

    template<>
    bool RegionAccessor<AccessorGeneric>::can_convert<AccessorReductionList>(void) const
    {
      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      //MemoryImpl *m_impl = i_impl->memory.impl();

      // make sure it's a reduction fold-only instance
      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);
      if(!i_data->is_reduction) return false;
      if(i_data->red_list_size < 0) return false;

      // that's the only requirement
      return true;
    }

    template<>
    RegionAccessor<AccessorReductionList> RegionAccessor<AccessorGeneric>::convert<AccessorReductionList>(void) const
    {
      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      //MemoryImpl *m_impl = i_impl->memory.impl();

      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);

      assert(i_data->is_reduction);
      assert(i_data->red_list_size >= 0);

      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[i_data->redopid];

      RegionAccessor<AccessorReductionList> ria(internal_data,
							       i_data->red_list_size,
							       redop->sizeof_list_entry);
      return ria;
    }

    RegionAccessor<AccessorReductionList>::RegionAccessor(void *_internal_data,
											size_t _num_entries,
											size_t _elem_size)
    {
      internal_data = _internal_data;

      RegionInstanceImpl *i_impl = (RegionInstanceImpl *)internal_data;
      MemoryImpl *m_impl = i_impl->memory.impl();

      Realm::StaticAccess<RegionInstanceImpl> i_data(i_impl);

      cur_size = (size_t *)(m_impl->get_direct_ptr(i_data->count_offset, sizeof(size_t)));

      max_size = _num_entries;

      // and a list of reduction entries (unless size == 0)
      entry_list = m_impl->get_direct_ptr(i_data->alloc_offset,
					  i_data->size);
    }

    void RegionAccessor<AccessorReductionList>::flush(void) const
    {
      assert(0);
    }

    void RegionAccessor<AccessorReductionList>::reduce_slow_case(size_t my_pos, unsigned ptrvalue,
										const void *entry, size_t sizeof_entry) const
    {
      assert(0);
    }
#endif


#ifdef EVENT_TRACING
    static char *event_trace_file = 0;
#endif
#ifdef LOCK_TRACING
    static char *lock_trace_file = 0;
#endif

  };
};

namespace Realm {

};

namespace LegionRuntime {
  namespace LowLevel {


#ifdef DEADLOCK_TRACE
    void deadlock_catch(int signal) {
      assert((signal == SIGTERM) || (signal == SIGINT));
      // First thing we do is dump our state
      {
        void *bt[256];
        int bt_size = backtrace(bt, 256);
        char **bt_syms = backtrace_symbols(bt, bt_size);
        size_t buffer_size = 1;
        for (int i = 0; i < bt_size; i++)
          buffer_size += (strlen(bt_syms[i]) + 1);
        char *buffer = (char*)malloc(buffer_size);
        int offset = 0;
        for (int i = 0; i < bt_size; i++)
          offset += sprintf(buffer+offset,"%s\n",bt_syms[i]);
#ifdef NODE_LOGGING
        char file_name[256];
        sprintf(file_name,"%s/backtrace_%d_thread_%ld.txt",
                          RuntimeImpl::prefix, gasnet_mynode(), pthread_self());
        FILE *fbt = fopen(file_name,"w");
        fprintf(fbt,"BACKTRACE (%d, %lx)\n----------\n%s\n----------\n", 
                gasnet_mynode(), pthread_self(), buffer);
        fflush(fbt);
        fclose(fbt);
#else
        fprintf(stderr,"BACKTRACE (%d, %lx)\n----------\n%s\n----------\n", 
                gasnet_mynode(), pthread_self(), buffer);
        fflush(stderr);
#endif
        free(buffer);
      }
      // Check to see if we are the first ones to catch the signal
      RuntimeImpl *rt = get_runtime();
      unsigned prev_count = __sync_fetch_and_add(&(rt->signaled_threads),1);
      // If we're the first do special stuff
      if (prev_count == 0) {
        unsigned expected = 1;
        // we are special, tell any other threads to handle a signal
        for (unsigned idx = 0; idx < rt->next_thread; idx++)
          if (rt->all_threads[idx] != pthread_self()) {
            pthread_kill(rt->all_threads[idx], SIGTERM);
            expected++;
          }
        // dump our waiters
#ifdef NODE_LOGGING
        char file_name[256];
        sprintf(file_name,"%s/waiters_%d.txt",
                          RuntimeImpl::prefix, gasnet_mynode());
        FILE *fw = fopen(file_name,"w");
        show_event_waiters(fw);
        fflush(fw);
        fclose(fw);
#else
        show_event_waiters(stderr);
#endif
        // the wait for everyone else to be done
        while (__sync_fetch_and_add(&(rt->signaled_threads),0) < expected) {
#ifdef __SSE2__
          _mm_pause();
#else
          usleep(1000);
#endif
        }
        // Now that everyone is done we can exit the process
        exit(1);
      }
    }
#endif

  };
};

namespace Realm {

};

namespace LegionRuntime {
  namespace LowLevel {


  }; // namespace LowLevel

}; // namespace LegionRuntime

// Implementation of accessor methods
namespace LegionRuntime {
  namespace Accessor {
    using namespace LegionRuntime::LowLevel;

    namespace DebugHooks {
      // these are calls that can be implemented by a higher level (e.g. Legion) to
      //  perform privilege/bounds checks on accessor reference and produce more useful
      //  information for debug

      /*extern*/ void (*check_bounds_ptr)(void *region, ptr_t ptr) = 0;
      /*extern*/ void (*check_bounds_dpoint)(void *region, const Realm::DomainPoint &dp) = 0;

      /*extern*/ const char *(*find_privilege_task_name)(void *region) = 0;
    };

    void AccessorType::Generic::Untyped::read_untyped(ptr_t ptr, void *dst, size_t bytes, off_t offset) const
    {
#ifdef __OPTIMIZE__
      issue_performance_warning();
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      if(DebugHooks::check_bounds_ptr)
	(DebugHooks::check_bounds_ptr)(region, ptr);
#endif
#ifdef USE_HDF
      // HDF memory doesn't support 
      assert(impl->memory.kind() != Memory::HDF_MEM);
#endif
      Arrays::Mapping<1, 1> *mapping = impl->metadata.linearization.get_mapping<1>();
      int index = mapping->image(ptr.value);
      impl->get_bytes(index, field_offset + offset, dst, bytes);
    }

    //bool debug_mappings = false;
    void AccessorType::Generic::Untyped::read_untyped(const DomainPoint& dp, void *dst, size_t bytes, off_t offset) const
    {
#ifdef __OPTIMIZE__
      issue_performance_warning();
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      if(DebugHooks::check_bounds_dpoint)
	(DebugHooks::check_bounds_dpoint)(region, dp);
#endif
#ifdef OLD_USE_HDF
      // we can directly access HDF memory by domain point
      if (impl->memory.kind() == Memory::HDF_MEM) {
        HDFMemory* hdf = (HDFMemory*)get_runtime()->get_memory_impl(impl->memory);
        int fid = 0;
        off_t byte_offset = field_offset + offset;
        for(std::vector<size_t>::const_iterator it = impl->metadata.field_sizes.begin();
        it != impl->metadata.field_sizes.end(); it++) {
          if (byte_offset < (off_t)(*it)) {
            break;
          }
          byte_offset -= (*it);
          fid ++;
        }
        ID id(impl->me);
        unsigned index = id.instance.inst_idx;
        assert(dp.dim == hdf->hdf_metadata[index]->ndims);
        hdf->get_bytes(index, dp, fid, dst, bytes);
        return;
      }
#endif
      int index = impl->metadata.linearization.get_image(dp);
      impl->get_bytes(index, field_offset + offset, dst, bytes);
      // if (debug_mappings) {
      // 	printf("READ: " IDFMT " (%d,%d,%d,%d) -> %d /", impl->me.id, dp.dim, dp.point_data[0], dp.point_data[1], dp.point_data[2], index);
      // 	for(size_t i = 0; (i < bytes) && (i < 32); i++)
      // 	  printf(" %02x", ((unsigned char *)dst)[i]);
      // 	printf("\n");
      // }
    }

    void AccessorType::Generic::Untyped::write_untyped(ptr_t ptr, const void *src, size_t bytes, off_t offset) const
    {
#ifdef __OPTIMIZE__
      issue_performance_warning();
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      if(DebugHooks::check_bounds_ptr)
	(DebugHooks::check_bounds_ptr)(region, ptr);
#endif
#ifdef USE_HDF
     // HDF memory doesn't support enumerate type
     assert(impl->memory.kind() != Memory::HDF_MEM);
#endif

      Arrays::Mapping<1, 1> *mapping = impl->metadata.linearization.get_mapping<1>();
      int index = mapping->image(ptr.value);
      impl->put_bytes(index, field_offset + offset, src, bytes);
    }

    void AccessorType::Generic::Untyped::write_untyped(const DomainPoint& dp, const void *src, size_t bytes, off_t offset) const
    {
#ifdef __OPTIMIZE__
      issue_performance_warning();
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      if(DebugHooks::check_bounds_dpoint)
	(DebugHooks::check_bounds_dpoint)(region, dp);
#endif
#ifdef OLD_USE_HDF
      if (impl->memory.kind() == Memory::HDF_MEM) {
        HDFMemory* hdf = (HDFMemory*) get_runtime()->get_memory_impl(impl->memory);
        int fid = 0;
        off_t byte_offset = field_offset + offset;
        for(std::vector<size_t>::const_iterator it = impl->metadata.field_sizes.begin();
        it != impl->metadata.field_sizes.end(); it++) {
          if (byte_offset < (off_t)(*it)) {
            break;
          }
          byte_offset -= (*it);
          fid ++;
        }
        ID id(impl->me);
        unsigned index = id.instance.inst_idx;
        assert(dp.dim == hdf->hdf_metadata[index]->ndims);
        hdf->put_bytes(index, dp, fid, src, bytes);
        return;
      }
#endif

      int index = impl->metadata.linearization.get_image(dp);
      // if (debug_mappings) {
      // 	printf("WRITE: " IDFMT " (%d,%d,%d,%d) -> %d /", impl->me.id, dp.dim, dp.point_data[0], dp.point_data[1], dp.point_data[2], index);
      // 	for(size_t i = 0; (i < bytes) && (i < 32); i++)
      // 	  printf(" %02x", ((const unsigned char *)src)[i]);
      // 	printf("\n");
      // }
      impl->put_bytes(index, field_offset + offset, src, bytes);
    }

    bool AccessorType::Generic::Untyped::get_aos_parameters(void *&base, size_t &stride) const
    {
      // TODO: implement this
      return false;
    }

    bool AccessorType::Generic::Untyped::get_soa_parameters(void *&base, size_t &stride) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      return impl->get_strided_parameters(base, stride, field_offset);
#if 0
      MemoryImpl *mem = impl->memory.impl();
      Realm::StaticAccess<RegionInstanceImpl> idata(impl);

      off_t offset = idata->alloc_offset;
      off_t elmt_stride;
      
      if (idata->block_size == 1) {
        offset += field_offset;
        elmt_stride = idata->elmt_size;
      } else {
        off_t field_start;
        int field_size;
        Realm::find_field_start(idata->field_sizes, field_offset, 1, field_start, field_size);

        offset += (field_start * idata->block_size) + (field_offset - field_start);
	elmt_stride = field_size;
      }

      base = mem->get_direct_ptr(offset, 0);
      if (!base) return false;

      // if the caller wants a particular stride and we differ (and have more
      //  than one element), fail
      if(stride != 0) {
        if((stride != elmt_stride) && (idata->size > idata->elmt_size))
          return false;
      } else {
        stride = elmt_stride;
      }

      // if there's a per-element offset, apply it after we've agreed with the caller on 
      //  what we're pretending the stride is
      const DomainLinearization& dl = impl->linearization;
      if(dl.get_dim() > 0) {
	// make sure this instance uses a 1-D linearization
	assert(dl.get_dim() == 1);

	Arrays::Mapping<1, 1> *mapping = dl.get_mapping<1>();
	Rect<1> preimage = mapping->preimage(0);
	assert(preimage.lo == preimage.hi);
	// double-check that whole range maps densely
	preimage.hi.x[0] += 1; // not perfect, but at least detects non-unit-stride case
	assert(mapping->image_is_dense(preimage));
	int inst_first_elmt = preimage.lo[0];
	//printf("adjusting base by %d * %zd\n", inst_first_elmt, stride);
	base = ((char *)base) - inst_first_elmt * stride;
      }

      return true;
#endif
    }

    bool AccessorType::Generic::Untyped::get_hybrid_soa_parameters(void *&base, size_t &stride, 
                                                                   size_t &block_size, size_t &block_stride) const
    {
      // TODO: implement this
      return false;
    }

    bool AccessorType::Generic::Untyped::get_redfold_parameters(void *&base) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *)internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      if (impl->metadata.redopid == 0) return false;
      if (impl->metadata.red_list_size > 0) return false;

      // ReductionFold accessors currently assume packed instances
      size_t stride = impl->metadata.elmt_size;
      return impl->get_strided_parameters(base, stride, field_offset);
#if 0
      off_t offset = impl->metadata.alloc_offset + field_offset;
      off_t elmt_stride;

      if (impl->metadata.block_size == 1) {
        offset += field_offset;
        elmt_stride = impl->metadata.elmt_size;
      } else {
        off_t field_start;
        int field_size;
        Realm::find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

        offset += (field_start * impl->metadata.block_size) + (field_offset - field_start);
	elmt_stride = field_size;
      }
      base = mem->get_direct_ptr(offset, 0);
      if (!base) return false;
      return true;
#endif
    }

    bool AccessorType::Generic::Untyped::get_redlist_parameters(void *&base, ptr_t *&next_ptr) const
    {
      // TODO: implement this
      return false;
    }
#ifdef POINTER_CHECKS
    void AccessorType::verify_access(void *impl_ptr, unsigned ptr)
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) impl_ptr;
      impl->verify_access(ptr);
    }
#endif

    void *AccessorType::Generic::Untyped::raw_span_ptr(ptr_t ptr, size_t req_count, size_t& act_count, ByteOffset& stride) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      void *base;
      size_t act_stride = 0;
#ifndef NDEBUG
      bool ok =
#endif
	impl->get_strided_parameters(base, act_stride, field_offset);
      assert(ok);

#ifdef BOUNDS_CHECKS
      if(DebugHooks::check_bounds_ptr)
	(DebugHooks::check_bounds_ptr)(region, ptr);
#endif

      Arrays::Mapping<1, 1> *mapping = impl->metadata.linearization.get_mapping<1>();
      int index = mapping->image(ptr.value);

      void *elem_ptr = ((char *)base) + (index * act_stride);
      stride.offset = act_stride;
      act_count = req_count; // TODO: return a larger number if we know how big we are
      return elem_ptr;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Arrays::Rect<DIM>& r, Arrays::Rect<DIM>& subrect, ByteOffset *offsets) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      MemoryImpl *mem = get_runtime()->get_memory_impl(impl->memory);

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      Arrays::Mapping<DIM, 1> *mapping = impl->metadata.linearization.get_mapping<DIM>();

      Arrays::Point<1> strides[DIM];
      Arrays::coord_t index = mapping->image_linear_subrect(r, subrect, strides);

      off_t offset = impl->metadata.alloc_offset;
      off_t elmt_stride;

      if(impl->metadata.block_size <= 1) {
	offset += index * impl->metadata.elmt_size + field_offset;
	elmt_stride = impl->metadata.elmt_size;
      } else {
	off_t field_start;
	int field_size;
	Realm::find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

	size_t block_num = index / impl->metadata.block_size;
	size_t block_ofs = index % impl->metadata.block_size;

	offset += (((impl->metadata.elmt_size * block_num + field_start) * impl->metadata.block_size) + 
		   (field_size * block_ofs) +
		   (field_offset - field_start));
	elmt_stride = field_size;
      }

      char *dst = (char *)(mem->get_direct_ptr(offset, subrect.volume() * elmt_stride));
      if(!dst) return 0;

      for(int i = 0; i < DIM; i++)
	offsets[i].offset = strides[i][0] * elmt_stride;

      return dst;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(ByteOffset *offsets) const
    {
      // caller didn't give us a rectangle, so ask for something really big...
      Arrays::Point<DIM> lo = Arrays::Point<DIM>::ZEROES();
      Arrays::Point<DIM> hi;
      for(unsigned i = 0; i < DIM; i++)
	hi.x[i] = INT_MAX;
      Arrays::Rect<DIM> r(lo, hi);
      Arrays::Rect<DIM> subrect;
      void *ptr = raw_rect_ptr<DIM>(r, subrect, offsets);
      assert(r == subrect);
      return ptr;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Arrays::Rect<DIM>& r, Arrays::Rect<DIM>& subrect, ByteOffset *offsets,
						       const std::vector<off_t> &field_offsets, ByteOffset &field_stride) const
    {
      if(field_offsets.size() < 1)
	return 0;

      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      MemoryImpl *mem = get_runtime()->get_memory_impl(impl->memory);

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      Arrays::Mapping<DIM, 1> *mapping = impl->metadata.linearization.get_mapping<DIM>();

      Arrays::Point<1> strides[DIM];
      int index = mapping->image_linear_subrect(r, subrect, strides);

      off_t offset = impl->metadata.alloc_offset;
      off_t elmt_stride;
      off_t fld_stride;

      if(impl->metadata.block_size == 1) {
	offset += index * impl->metadata.elmt_size + field_offset;
	elmt_stride = impl->metadata.elmt_size;

	if(field_offsets.size() == 1) {
	  fld_stride = 0;
	} else {
	  fld_stride = field_offsets[1] - field_offsets[0];
	  for(size_t i = 2; i < field_offsets.size(); i++)
	    if((field_offsets[i] - field_offsets[i-1]) != fld_stride) {
	      // fields aren't evenly spaced - abort
	      return 0;
	    }
	}
      } else {
	off_t field_start;
	int field_size;
	Realm::find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

	int block_num = index / impl->metadata.block_size;
	int block_ofs = index % impl->metadata.block_size;

	offset += (((impl->metadata.elmt_size * block_num + field_start) * impl->metadata.block_size) + 
		   (field_size * block_ofs) +
		   (field_offset - field_start));
	elmt_stride = field_size;

	if(field_offsets.size() == 1) {
	  fld_stride = 0;
	} else {
	  off_t field_start2;
	  int field_size2;
	  Realm::find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[1], 1, field_start2, field_size2);

	  // field sizes much match or element stride isn't consistent
	  if(field_size2 != field_size)
	    return 0;
	  
	  fld_stride = (((field_start2 - field_start) * impl->metadata.block_size) + 
			(field_offsets[1] - field_start2) - (field_offsets[0] - field_start));

	  for(size_t i = 2; i < field_offsets.size(); i++) {
	    Realm::find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[i], 1, field_start2, field_size2);
	    off_t fld_stride2 = (((field_start2 - field_start) * impl->metadata.block_size) + 
			(field_offsets[i] - field_start2) - (field_offsets[0] - field_start));
	    if(fld_stride2 != fld_stride * i) {
	      // fields aren't evenly spaced - abort
	      return 0;
	    }
	  }
	}
      }

      char *dst = (char *)(mem->get_direct_ptr(offset, subrect.volume() * elmt_stride));
      if(!dst) return 0;

      for(int i = 0; i < DIM; i++)
	offsets[i].offset = strides[i] * elmt_stride;

      field_stride.offset = fld_stride;

      return dst;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_dense_ptr(const Arrays::Rect<DIM>& r, Arrays::Rect<DIM>& subrect, ByteOffset &elem_stride) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      MemoryImpl *mem = get_runtime()->get_memory_impl(impl->memory);

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      Arrays::Mapping<DIM, 1> *mapping = impl->metadata.linearization.get_mapping<DIM>();

      Arrays::Rect<1> ir = mapping->image_dense_subrect(r, subrect);
      int index = ir.lo;

      off_t offset = impl->metadata.alloc_offset;
      off_t elmt_stride;

      if(impl->metadata.block_size == 1) {
	offset += index * impl->metadata.elmt_size + field_offset;
	elmt_stride = impl->metadata.elmt_size;
      } else {
	off_t field_start;
	int field_size;
	Realm::find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

	int block_num = index / impl->metadata.block_size;
	int block_ofs = index % impl->metadata.block_size;

	offset += (((impl->metadata.elmt_size * block_num + field_start) * impl->metadata.block_size) + 
		   (field_size * block_ofs) +
		   (field_offset - field_start));
	elmt_stride = field_size;
      }

      char *dst = (char *)(mem->get_direct_ptr(offset, subrect.volume() * elmt_stride));
      if(!dst) return 0;
      
      elem_stride.offset = elmt_stride;

      return dst;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_dense_ptr(const Arrays::Rect<DIM>& r, Arrays::Rect<DIM>& subrect, ByteOffset &elem_stride,
							const std::vector<off_t> &field_offsets, ByteOffset &field_stride) const
    {
      if(field_offsets.size() < 1)
	return 0;

      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      MemoryImpl *mem = get_runtime()->get_memory_impl(impl->memory);

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      Arrays::Mapping<DIM, 1> *mapping = impl->metadata.linearization.get_mapping<DIM>();

      int index = mapping->image_dense_subrect(r, subrect);

      off_t offset = impl->metadata.alloc_offset;
      off_t elmt_stride;
      off_t fld_stride;

      if(impl->metadata.block_size == 1) {
	offset += index * impl->metadata.elmt_size + field_offset + field_offsets[0];
	elmt_stride = impl->metadata.elmt_size;

	if(field_offsets.size() == 1) {
	  fld_stride = 0;
	} else {
	  fld_stride = field_offsets[1] - field_offsets[0];
	  for(size_t i = 2; i < field_offsets.size(); i++)
	    if((field_offsets[i] - field_offsets[i-1]) != fld_stride) {
	      // fields aren't evenly spaced - abort
	      return 0;
	    }
	}
      } else {
	off_t field_start;
	int field_size;
	Realm::find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[0], 1, field_start, field_size);

	int block_num = index / impl->metadata.block_size;
	int block_ofs = index % impl->metadata.block_size;

	offset += (((impl->metadata.elmt_size * block_num + field_start) * impl->metadata.block_size) + 
		   (field_size * block_ofs) +
		   (field_offset + field_offsets[0] - field_start));
	elmt_stride = field_size;

	if(field_offsets.size() == 1) {
	  fld_stride = 0;
	} else {
	  off_t field_start2;
	  int field_size2;
	  Realm::find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[1], 1, field_start2, field_size2);

	  // field sizes much match or element stride isn't consistent
	  if(field_size2 != field_size)
	    return 0;
	  
	  fld_stride = (((field_start2 - field_start) * impl->metadata.block_size) + 
			(field_offsets[1] - field_start2) - (field_offsets[0] - field_start));

	  for(size_t i = 2; i < field_offsets.size(); i++) {
	    Realm::find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[i], 1, field_start2, field_size2);
	    off_t fld_stride2 = (((field_start2 - field_start) * impl->metadata.block_size) + 
			(field_offsets[i] - field_start2) - (field_offsets[0] - field_start));
	    if(fld_stride2 != fld_stride * i) {
	      // fields aren't evenly spaced - abort
	      return 0;
	    }
	  }
	}
      }

      char *dst = (char *)(mem->get_direct_ptr(offset, subrect.volume() * elmt_stride));
      if(!dst) return 0;
      
      elem_stride.offset = elmt_stride;
      field_stride.offset = fld_stride;

      return dst;
    }

    template void *AccessorType::Generic::Untyped::raw_rect_ptr<1>(ByteOffset *offset) const;
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<2>(ByteOffset *offset) const;
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<3>(ByteOffset *offset) const;
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<1>(const Arrays::Rect<1>& r, Arrays::Rect<1>& subrect, ByteOffset *offset) const;
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<2>(const Arrays::Rect<2>& r, Arrays::Rect<2>& subrect, ByteOffset *offset) const;
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<3>(const Arrays::Rect<3>& r, Arrays::Rect<3>& subrect, ByteOffset *offset) const;
    template void *AccessorType::Generic::Untyped::raw_dense_ptr<1>(const Arrays::Rect<1>& r, Arrays::Rect<1>& subrect, ByteOffset &elem_stride) const;
    template void *AccessorType::Generic::Untyped::raw_dense_ptr<2>(const Arrays::Rect<2>& r, Arrays::Rect<2>& subrect, ByteOffset &elem_stride) const;
    template void *AccessorType::Generic::Untyped::raw_dense_ptr<3>(const Arrays::Rect<3>& r, Arrays::Rect<3>& subrect, ByteOffset &elem_stride) const;

    void AccessorType::Generic::Untyped::report_fault(ptr_t ptr, size_t bytes, off_t offset /*= 0*/) const
    {
      assert(0 && "fault injection not implemented yet");
    }

    void AccessorType::Generic::Untyped::report_fault(const Realm::DomainPoint& dp,
                                                      size_t bytes, off_t offset /*= 0*/) const
    {
      assert(0 && "fault injection not implemented yet");
    }
  };

  namespace Arrays {
    //template<> class Mapping<1,1>;
    template <unsigned IDIM, unsigned ODIM>
    /*static*/ MappingRegistry<IDIM, ODIM>& Mapping<IDIM, ODIM>::registry(void)
    {
      static MappingRegistry<IDIM, ODIM> singleton;
      return singleton;
    }
    template class Mapping<1,1>;
    template class Mapping<2,1>;
    template class Mapping<3,1>;
  };
};
