/* Copyright 2015 Stanford University, NVIDIA Corporation
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


#include "lowlevel.h"
#include "lowlevel_impl.h"
#include "accessor.h"

#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

using namespace LegionRuntime::Accessor;

#ifdef USE_CUDA
#include "lowlevel_gpu.h"
#endif

#include "lowlevel_dma.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <dirent.h>

#include <signal.h>
#include <unistd.h>
#if defined(REALM_BACKTRACE) || defined(LEGION_BACKTRACE) || defined(DEADLOCK_TRACE)
#include <execinfo.h>
#include <cxxabi.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

// like strdup, but works on arbitrary byte arrays
static void *bytedup(const void *data, size_t datalen)
{
  if(datalen == 0) return 0;
  void *dst = malloc(datalen);
  assert(dst != 0);
  memcpy(dst, data, datalen);
  return dst;
}

// Implementation of Detailed Timer
namespace LegionRuntime {
  namespace LowLevel {

    GASNETT_THREADKEY_DEFINE(cur_preemptable_thread);
#ifdef USE_CUDA
    GASNETT_THREADKEY_DECLARE(gpu_thread_ptr);
#endif
    
#ifdef USE_CUDA
    Logger::Category log_gpu("gpu");
#endif
    Logger::Category log_mutex("mutex");
    Logger::Category log_timer("timer");
    Logger::Category log_region("region");
    Logger::Category log_malloc("malloc");
    Logger::Category log_machine("machine");
    Logger::Category log_inst("inst");
#ifdef EVENT_GRAPH_TRACE
    Logger::Category log_event_graph("graph");
#endif
    Logger::Category log_barrier("barrier");
    Logger::Category log_meta("meta");

    enum ActiveMessageIDs {
      FIRST_AVAILABLE = 140,
      NODE_ANNOUNCE_MSGID,
      SPAWN_TASK_MSGID,
      LOCK_REQUEST_MSGID,
      LOCK_RELEASE_MSGID,
      LOCK_GRANT_MSGID,
      EVENT_SUBSCRIBE_MSGID,
      EVENT_TRIGGER_MSGID,
      REMOTE_MALLOC_MSGID,
      REMOTE_MALLOC_RPLID,
      CREATE_ALLOC_MSGID,
      CREATE_ALLOC_RPLID,
      CREATE_INST_MSGID,
      CREATE_INST_RPLID,
      VALID_MASK_REQ_MSGID,
      VALID_MASK_DATA_MSGID,
      ROLL_UP_TIMER_MSGID,
      ROLL_UP_DATA_MSGID,
      CLEAR_TIMER_MSGID,
      DESTROY_INST_MSGID,
      REMOTE_WRITE_MSGID,
      REMOTE_REDUCE_MSGID,
      REMOTE_WRITE_FENCE_MSGID,
      DESTROY_LOCK_MSGID,
      REMOTE_REDLIST_MSGID,
      MACHINE_SHUTDOWN_MSGID,
      BARRIER_ADJUST_MSGID,
      BARRIER_SUBSCRIBE_MSGID,
      BARRIER_TRIGGER_MSGID,
      METADATA_REQUEST_MSGID,
      METADATA_RESPONSE_MSGID, // should really be a reply
      METADATA_INVALIDATE_MSGID,
      METADATA_INVALIDATE_RPLID,
    };

#ifdef EVENT_GRAPH_TRACE
    Event find_enclosing_termination_event(void)
    {
      void *tls_val = gasnett_threadkey_get(cur_preemptable_thread);
      if (tls_val != NULL) {
        PreemptableThread *me = (PreemptableThread*)tls_val;
        return me->find_enclosing();
      }
#ifdef USE_CUDA
      tls_val = gasnett_threadkey_get(gpu_thread_ptr);
      if (tls_val != NULL) {
        GPUProcessor *me = (GPUProcessor*)tls_val;
        return me->find_enclosing();
      }
#endif
      return Event::NO_EVENT;
    }
#endif

    // detailed timer stuff

    struct TimerStackEntry {
      int timer_kind;
      double start_time;
      double accum_child_time;
    };

    struct PerThreadTimerData {
    public:
      PerThreadTimerData(void)
      {
        thread = pthread_self();
      }

      pthread_t thread;
      std::list<TimerStackEntry> timer_stack;
      std::map<int, double> timer_accum;
      GASNetHSL mutex;
    };

    GASNetHSL timer_data_mutex;
    std::vector<PerThreadTimerData *> timer_data;
    pthread_key_t thread_timer_key;
    //__thread PerThreadTimerData *thread_timer_data;
    static void thread_timer_free(void *arg)
    {
      assert(arg != NULL);
      PerThreadTimerData *ptr = (PerThreadTimerData*)arg;
      delete ptr;
    }

    struct ClearTimerRequestArgs {
      int sender;
      int dummy; // needed to get sizeof() >= 8
    };

    void handle_clear_timer_request(ClearTimerRequestArgs args)
    {
      DetailedTimer::clear_timers(false);
    }

    typedef ActiveMessageShortNoReply<CLEAR_TIMER_MSGID,
				      ClearTimerRequestArgs,
				      handle_clear_timer_request> ClearTimerRequestMessage;
    
#ifdef DETAILED_TIMING
    /*static*/ void DetailedTimer::clear_timers(bool all_nodes /*= true*/)
    {
      // take global mutex because we need to walk the list
      {
	log_timer.warning("clearing timers");
	AutoHSLLock l1(timer_data_mutex);
	for(std::vector<PerThreadTimerData *>::iterator it = timer_data.begin();
	    it != timer_data.end();
	    it++) {
	  // take each thread's data's lock too
	  AutoHSLLock l2((*it)->mutex);
	  (*it)->timer_accum.clear();
	}
      }

      // if we've been asked to clear other nodes too, send a message
      if(all_nodes) {
	ClearTimerRequestArgs args;
	args.sender = gasnet_mynode();

	for(int i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    ClearTimerRequestMessage::request(i, args);
      }
    }

    /*static*/ void DetailedTimer::push_timer(int timer_kind)
    {
      PerThreadTimerData *thread_timer_data = 
        (PerThreadTimerData*) pthread_getspecific(thread_timer_key);
      if(!thread_timer_data) {
        //printf("creating timer data for thread %lx\n", pthread_self());
        AutoHSLLock l1(timer_data_mutex);
        thread_timer_data = new PerThreadTimerData;
        CHECK_PTHREAD( pthread_setspecific(thread_timer_key, thread_timer_data) );
        timer_data.push_back(thread_timer_data);
      }

      // no lock needed here - only our thread touches the stack
      TimerStackEntry entry;
      entry.timer_kind = timer_kind;
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      entry.start_time = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec);
      entry.accum_child_time = 0;
      PerThreadTimerData *thread_timer_data = 
        (PerThreadTimerData*) pthread_getspecific(thread_timer_key);
      thread_timer_data->timer_stack.push_back(entry);
    }
        
    /*static*/ void DetailedTimer::pop_timer(void)
    {
      PerThreadTimerData *thread_timer_data = 
        (PerThreadTimerData*) pthread_getspecific(thread_timer_key);
      if(!thread_timer_data) {
        printf("got pop without initialized thread data!?\n");
        exit(1);
      }

      // no conflicts on stack
      PerThreadTimerData *thread_timer_data = 
        (PerThreadTimerData*) pthread_getspecific(thread_timer_key);
      TimerStackEntry old_top = thread_timer_data->timer_stack.back();
      thread_timer_data->timer_stack.pop_back();

      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      double elapsed = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec) - old_top.start_time;

      // all the elapsed time is added to new top as child time
      if(thread_timer_data->timer_stack.size() > 0)
        thread_timer_data->timer_stack.back().accum_child_time += elapsed;

      // only the elapsed minus our own child time goes into the timer accumulator
      elapsed -= old_top.accum_child_time;

      // we do need a lock to touch the accumulator map
      if(old_top.timer_kind > 0) {
        AutoHSLLock l1(thread_timer_data->mutex);

        std::map<int,double>::iterator it = thread_timer_data->timer_accum.find(old_top.timer_kind);
        if(it != thread_timer_data->timer_accum.end())
          it->second += elapsed;
        else
          thread_timer_data->timer_accum.insert(std::make_pair<int,double>(old_top.timer_kind, elapsed));
      }
    }
#endif

    class MultiNodeRollUp {
    public:
      MultiNodeRollUp(std::map<int,double>& _timers);

      void execute(void);

      void handle_data(const void *data, size_t datalen);

    protected:
      GASNetHSL mutex;
      GASNetCondVar condvar;
      std::map<int,double> *timerp;
      volatile int count_left;
    };

    struct RollUpRequestArgs {
      int sender;
      void *rollup_ptr;
    };

    void handle_roll_up_request(RollUpRequestArgs args);

    typedef ActiveMessageShortNoReply<ROLL_UP_TIMER_MSGID,
                                      RollUpRequestArgs,
                                      handle_roll_up_request> RollUpRequestMessage;

    struct RollUpDataArgs : public BaseMedium {
      void *rollup_ptr;
    };

    void handle_roll_up_data(RollUpDataArgs args, const void *data, size_t datalen)
    {
      ((MultiNodeRollUp *)args.rollup_ptr)->handle_data(data, datalen); 
    }

    typedef ActiveMessageMediumNoReply<ROLL_UP_DATA_MSGID,
                                       RollUpDataArgs,
                                       handle_roll_up_data> RollUpDataMessage;

    void handle_roll_up_request(RollUpRequestArgs args)
    {
      std::map<int,double> timers;
      DetailedTimer::roll_up_timers(timers, true);

      double return_data[200];
      int count = 0;
      for(std::map<int,double>::iterator it = timers.begin();
          it != timers.end();
          it++) {
        *(int *)(&return_data[count]) = it->first;
        return_data[count+1] = it->second;
        count += 2;
      }
      RollUpDataArgs return_args;
      return_args.rollup_ptr = args.rollup_ptr;
      RollUpDataMessage::request(args.sender, return_args,
                                 return_data, count*sizeof(double),
				 PAYLOAD_COPY);
    }

    MultiNodeRollUp::MultiNodeRollUp(std::map<int,double>& _timers)
      : condvar(mutex), timerp(&_timers)
    {
      count_left = 0;
    }

    void MultiNodeRollUp::execute(void)
    {
      count_left = gasnet_nodes()-1;

      RollUpRequestArgs args;
      args.sender = gasnet_mynode();
      args.rollup_ptr = this;
      for(unsigned i = 0; i < gasnet_nodes(); i++)
        if(i != gasnet_mynode())
          RollUpRequestMessage::request(i, args);

      // take the lock so that we can safely sleep until all the responses
      //  arrive
      {
	AutoHSLLock al(mutex);

	if(count_left > 0)
	  condvar.wait();
      }
      assert(count_left == 0);
    }

    void MultiNodeRollUp::handle_data(const void *data, size_t datalen)
    {
      // have to take mutex here since we're updating shared data
      AutoHSLLock a(mutex);

      const double *p = (const double *)data;
      int count = datalen / (2 * sizeof(double));
      for(int i = 0; i < count; i++) {
        int kind = *(int *)(&p[2*i]);
        double accum = p[2*i+1];

        std::map<int,double>::iterator it = timerp->find(kind);
        if(it != timerp->end())
          it->second += accum;
        else
          timerp->insert(std::make_pair(kind,accum));
      }

      count_left--;
      if(count_left == 0)
	condvar.signal();
    }

#ifdef DETAILED_TIMING
    /*static*/ void DetailedTimer::roll_up_timers(std::map<int, double>& timers,
                                                  bool local_only)
    {
      // take global mutex because we need to walk the list
      {
	AutoHSLLock l1(timer_data_mutex);
	for(std::vector<PerThreadTimerData *>::iterator it = timer_data.begin();
	    it != timer_data.end();
	    it++) {
	  // take each thread's data's lock too
	  AutoHSLLock l2((*it)->mutex);

	  for(std::map<int,double>::iterator it2 = (*it)->timer_accum.begin();
	      it2 != (*it)->timer_accum.end();
	      it2++) {
	    std::map<int,double>::iterator it3 = timers.find(it2->first);
	    if(it3 != timers.end())
	      it3->second += it2->second;
	    else
	      timers.insert(*it2);
	  }
	}
      }

      // get data from other nodes if requested
      if(!local_only) {
        MultiNodeRollUp mnru(timers);
        mnru.execute();
      }
    }

    /*static*/ void DetailedTimer::report_timers(bool local_only /*= false*/)
    {
      std::map<int,double> timers;
      
      roll_up_timers(timers, local_only);

      printf("DETAILED TIMING SUMMARY:\n");
      for(std::map<int,double>::iterator it = timers.begin();
          it != timers.end();
          it++) {
        printf("%12s - %7.3f s\n", stringify(it->first), it->second);
      }
      printf("END OF DETAILED TIMING SUMMARY\n");
    }
#endif

    static void gasnet_barrier(void)
    {
      gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
      gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
    }
    
    /*static*/ double Clock::zero_time = 0;

    /*static*/ void Clock::synchronize(void)
    {
      // basic idea is that we barrier a couple times across the machine
      // and grab the local absolute time in between two of the barriers -
      // that becomes the zero time for the local machine
      gasnet_barrier();
      gasnet_barrier();
      zero_time = abs_time();
      gasnet_barrier();
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

    MachineImpl *machine_singleton = 0;
    RuntimeImpl *runtime_singleton = 0;
#ifdef NODE_LOGGING
    /*static*/ const char* RuntimeImpl::prefix = ".";
#endif

    //static const unsigned MAX_LOCAL_EVENTS = 300000;
    //static const unsigned MAX_LOCAL_LOCKS = 100000;

    Node::Node(void)
    {
    }

    struct LockRequestArgs {
      gasnet_node_t node;
      Reservation lock;
      unsigned mode;
    };

    void handle_lock_request(LockRequestArgs args);

    typedef ActiveMessageShortNoReply<LOCK_REQUEST_MSGID, 
				      LockRequestArgs, 
				      handle_lock_request> LockRequestMessage;

    struct LockReleaseArgs {
      gasnet_node_t node;
      Reservation lock;
    };
    
    void handle_lock_release(LockReleaseArgs args);

    typedef ActiveMessageShortNoReply<LOCK_RELEASE_MSGID,
				      LockReleaseArgs,
				      handle_lock_release> LockReleaseMessage;

    struct LockGrantArgs : public BaseMedium {
      Reservation lock;
      unsigned mode;
    };

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<LOCK_GRANT_MSGID,
				       LockGrantArgs,
				       handle_lock_grant> LockGrantMessage;

    struct ValidMaskRequestArgs {
      IndexSpace is;
      int sender;
    };

    void handle_valid_mask_request(ValidMaskRequestArgs args);

    typedef ActiveMessageShortNoReply<VALID_MASK_REQ_MSGID,
				      ValidMaskRequestArgs,
				      handle_valid_mask_request> ValidMaskRequestMessage;


    struct ValidMaskDataArgs : public BaseMedium {
      IndexSpace is;
      unsigned block_id;
    };

    void handle_valid_mask_data(ValidMaskDataArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<VALID_MASK_DATA_MSGID,
				       ValidMaskDataArgs,
				       handle_valid_mask_data> ValidMaskDataMessage;

    IndexSpaceImpl::IndexSpaceImpl(void)
    {
      init(IndexSpace::NO_SPACE, -1);
    }

    void IndexSpaceImpl::init(IndexSpace _me, unsigned _init_owner)
    {
      assert(!_me.exists() || (_init_owner == ID(_me).node()));

      me = _me;
      locked_data.valid = false;
      lock.init(ID(me).convert<Reservation>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
      valid_mask = 0;
      valid_mask_complete = false;
      valid_mask_event = Event::NO_EVENT;
      valid_mask_event_impl = 0;
    }

    void IndexSpaceImpl::init(IndexSpace _me, IndexSpace _parent,
				size_t _num_elmts,
				const ElementMask *_initial_valid_mask /*= 0*/, bool _frozen /*= false*/)
    {
      me = _me;
      locked_data.valid = true;
      locked_data.parent = _parent;
      locked_data.frozen = _frozen;
      locked_data.num_elmts = _num_elmts;
      locked_data.valid_mask_owners = (1ULL << gasnet_mynode());
      locked_data.avail_mask_owner = gasnet_mynode();
      valid_mask = (_initial_valid_mask?
		    new ElementMask(*_initial_valid_mask) :
		    new ElementMask(_num_elmts));
      valid_mask_complete = true;
      valid_mask_event = Event::NO_EVENT;
      valid_mask_event_impl = 0;
      if(_frozen) {
	avail_mask = 0;
	locked_data.first_elmt = valid_mask->first_enabled();
	locked_data.last_elmt = valid_mask->last_enabled();
	log_region.info("subregion " IDFMT " (of " IDFMT ") restricted to [%zd,%zd]",
			me.id, _parent.id, locked_data.first_elmt,
			locked_data.last_elmt);
      } else {
	avail_mask = new ElementMask(_num_elmts);
	if(_parent == IndexSpace::NO_SPACE) {
	  avail_mask->enable(0, _num_elmts);
	  locked_data.first_elmt = 0;
	  locked_data.last_elmt = _num_elmts - 1;
	} else {
	  StaticAccess<IndexSpaceImpl> pdata(get_runtime()->get_index_space_impl(_parent));
	  locked_data.first_elmt = pdata->first_elmt;
	  locked_data.last_elmt = pdata->last_elmt;
	}
      }
      lock.init(ID(me).convert<Reservation>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    IndexSpaceImpl::~IndexSpaceImpl(void)
    {
      delete valid_mask;
    }

    bool IndexSpaceImpl::is_parent_of(IndexSpace other)
    {
      while(other != IndexSpace::NO_SPACE) {
	if(other == me) return true;
	IndexSpaceImpl *other_impl = get_runtime()->get_index_space_impl(other);
	other = StaticAccess<IndexSpaceImpl>(other_impl)->parent;
      }
      return false;
    }

#if 0
    size_t IndexSpaceImpl::instance_size(const ReductionOpUntyped *redop /*= 0*/, off_t list_size /*= -1*/)
    {
      StaticAccess<IndexSpaceImpl> data(this);
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
      StaticAccess<IndexSpaceImpl> data(this);
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

    Event IndexSpaceImpl::request_valid_mask(void)
    {
      size_t num_elmts = StaticAccess<IndexSpaceImpl>(this)->num_elmts;
      int valid_mask_owner = -1;
      
      Event e;
      {
	AutoHSLLock a(valid_mask_mutex);
	
	if(valid_mask != 0) {
	  // if the mask exists, we've already requested it, so just provide
	  //  the event that we have
          return valid_mask_event;
	}
	
	valid_mask = new ElementMask(num_elmts);
	valid_mask_owner = ID(me).node(); // a good guess?
	valid_mask_count = (valid_mask->raw_size() + 2047) >> 11;
	valid_mask_complete = false;
	valid_mask_event_impl = GenEventImpl::create_genevent();
        valid_mask_event = valid_mask_event_impl->current_event();
        e = valid_mask_event;
      }
      
      ValidMaskRequestArgs args;
      args.is = me;
      args.sender = gasnet_mynode();
      ValidMaskRequestMessage::request(valid_mask_owner, args);

      return e;
    }

    void handle_valid_mask_request(ValidMaskRequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceImpl *r_impl = get_runtime()->get_index_space_impl(args.is);

      assert(r_impl->valid_mask);
      const char *mask_data = (const char *)(r_impl->valid_mask->get_raw());
      assert(mask_data);

      size_t mask_len = r_impl->valid_mask->raw_size();

      // send data in 2KB blocks
      ValidMaskDataArgs resp_args;
      resp_args.is = args.is;
      resp_args.block_id = 0;
      //printf("sending mask data for region " IDFMT " to %d (%p, %zd)\n",
      //	     args.region.id, args.sender, mask_data, mask_len);
      while(mask_len >= (1 << 11)) {
	ValidMaskDataMessage::request(args.sender, resp_args,
				      mask_data,
				      1 << 11,
				      PAYLOAD_KEEP);
	mask_data += 1 << 11;
	mask_len -= 1 << 11;
	resp_args.block_id++;
      }
      if(mask_len) {
	ValidMaskDataMessage::request(args.sender, resp_args,
				      mask_data, mask_len,
				      PAYLOAD_KEEP);
      }
    }

    void handle_valid_mask_data(ValidMaskDataArgs args, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceImpl *r_impl = get_runtime()->get_index_space_impl(args.is);

      assert(r_impl->valid_mask);
      // removing const on purpose here...
      char *mask_data = (char *)(r_impl->valid_mask->get_raw());
      assert(mask_data);
      assert((args.block_id << 11) < r_impl->valid_mask->raw_size());

      memcpy(mask_data + (args.block_id << 11), data, datalen);

      GenEventImpl *to_trigger = 0;
      {
	AutoHSLLock a(r_impl->valid_mask_mutex);
	//printf("got piece of valid mask data for region " IDFMT " (%d expected)\n",
	//       args.region.id, r_impl->valid_mask_count);
	r_impl->valid_mask_count--;
        if(r_impl->valid_mask_count == 0) {
	  r_impl->valid_mask_complete = true;
	  to_trigger = r_impl->valid_mask_event_impl;
	  r_impl->valid_mask_event_impl = 0;
	}
      }

      if(to_trigger) {
	//printf("triggering " IDFMT "/%d\n",
	//       r_impl->valid_mask_event.id, r_impl->valid_mask_event.gen);
	to_trigger->trigger_current();
      }
    }
    

    class IndexSpaceAllocatorImpl {
    public:
      IndexSpaceAllocatorImpl(IndexSpaceImpl *_is_impl);

      ~IndexSpaceAllocatorImpl(void);

      unsigned alloc_elements(unsigned count = 1);

      void reserve_elements(unsigned ptr, unsigned count = 1);

      void free_elements(unsigned ptr, unsigned count = 1);

      IndexSpaceImpl *is_impl;
    };

    ///////////////////////////////////////////////////
    // Metadata

    Logger::Category log_metadata("metadata");

    MetadataBase::MetadataBase(void)
      : state(STATE_INVALID), valid_event_impl(0)
    {}

    MetadataBase::~MetadataBase(void)
    {}

    void MetadataBase::mark_valid(void)
    {
      // don't actually need lock for this
      assert(remote_copies.empty()); // should not have any valid remote copies if we weren't valid
      state = STATE_VALID;
    }

    void MetadataBase::handle_request(int requestor)
    {
      // just add the requestor to the list of remote nodes with copies
      AutoHSLLock a(mutex);

      assert(is_valid());
      assert(!remote_copies.contains(requestor));
      remote_copies.add(requestor);
    }

    void MetadataBase::handle_response(void)
    {
      // update the state, and
      // if there was an event, we'll trigger it
      GenEventImpl *to_trigger = 0;
      {
	AutoHSLLock a(mutex);

	switch(state) {
	case STATE_REQUESTED:
	  {
	    to_trigger = valid_event_impl;
	    valid_event_impl = 0;
	    state = STATE_VALID;
	    break;
	  }

	default:
	  assert(0);
	}
      }

      if(to_trigger)
	to_trigger->trigger_current();
    }

    struct MetadataRequestMessage {
      struct RequestArgs {
	int node;
	IDType id;
      };

      static void handle_request(RequestArgs args)
      {
	// switch on different types of objects that can have metadata
	switch(ID(args.id).type()) {
	case ID::ID_INSTANCE:
	  {
	    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	    impl->metadata.handle_request(args.node);
	    size_t datalen;
	    void *data = impl->metadata.serialize(datalen);
	    log_metadata.info("metadata for " IDFMT " requested by %d - %zd bytes",
			      args.id, args.node, datalen);
	    send_response(args.node, args.id, data, datalen, PAYLOAD_FREE);
	    break;
	  }

	default:
	  assert(0);
	}
      }

      typedef ActiveMessageShortNoReply<METADATA_REQUEST_MSGID,
					RequestArgs,
					handle_request> RequestMessage;

      static void send_request(gasnet_node_t target, IDType id)
      {
	RequestArgs args;

	args.node = gasnet_mynode();
	args.id = id;
	RequestMessage::request(target, args);
      }

      struct ResponseArgs : public BaseMedium {
	IDType id;
      };
	
      static void handle_response(ResponseArgs args, const void *data, size_t datalen)
      {
	log_metadata.info("metadata for " IDFMT " received - %zd bytes",
			  args.id, datalen);

	// switch on different types of objects that can have metadata
	switch(ID(args.id).type()) {
	case ID::ID_INSTANCE:
	  {
	    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	    impl->metadata.deserialize(data, datalen);
	    impl->metadata.handle_response();
	    break;
	  }

	default:
	  assert(0);
	}
      }

      typedef ActiveMessageMediumNoReply<METADATA_RESPONSE_MSGID,
					 ResponseArgs,
					 handle_response> ResponseMessage;

      static void send_response(gasnet_node_t target, IDType id, 
				const void *data, size_t datalen, int payload_mode)
      {
	ResponseArgs args;

	args.id = id;

	ResponseMessage::request(target, args, data, datalen, payload_mode);
      }
    };

    Event MetadataBase::request_data(int owner, IDType id)
    {
      // early out - valid data need not be re-requested
      if(state == STATE_VALID) 
	return Event::NO_EVENT;

      // sanity-check - should never be requesting data from ourselves
      assert(((unsigned)owner) != gasnet_mynode());

      Event e = Event::NO_EVENT;
      bool issue_request = false;
      {
	AutoHSLLock a(mutex);

	switch(state) {
	case STATE_VALID:
	  {
	    // possible if the data came in between our early out check
	    // above and our taking of the lock - nothing more to do
	    break;
	  }

	case STATE_INVALID: 
	  {
	    // if the current state is invalid, we'll need to issue a request
	    state = STATE_REQUESTED;
	    valid_event_impl = GenEventImpl::create_genevent();
            e = valid_event_impl->current_event();
	    issue_request = true;
	    break;
	  }

	case STATE_REQUESTED:
	  {
	    // request has already been issued, but return the event again
	    assert(valid_event_impl);
            e = valid_event_impl->current_event();
	    break;
	  }

	case STATE_INVALIDATE:
	  assert(0 && "requesting metadata we've been told is invalid!");

	case STATE_CLEANUP:
	  assert(0 && "requesting metadata in CLEANUP state!");
	}
      }

      if(issue_request)
	MetadataRequestMessage::send_request(owner, id);

      return e;
    }

    void MetadataBase::await_data(bool block /*= true*/)
    {
      // early out - valid data means no waiting
      if(state == STATE_VALID) return;

      // take lock to get event - must have already been requested (we don't have enough
      //  information to do that now)
      Event e = Event::NO_EVENT;
      {
	AutoHSLLock a(mutex);

	assert(state != STATE_INVALID);
	if(valid_event_impl)
          e = valid_event_impl->current_event();
      }

      if(!e.has_triggered())
        e.wait(); // FIXME
    }

    class MetadataInvalidateMessage {
    public: 
      struct RequestArgs {
	int owner;
	IDType id;
      };

      static void handle_request(RequestArgs args)
      {
	log_metadata.info("received invalidate request for " IDFMT, args.id);

	//
	// switch on different types of objects that can have metadata
	switch(ID(args.id).type()) {
	case ID::ID_INSTANCE:
	  {
	    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	    impl->metadata.handle_invalidate();
	    break;
	  }

	default:
	  assert(0);
	}

	// ack the request
	respond(args.owner, args.id);
      }

      struct ResponseArgs {
	gasnet_node_t node;
	IDType id;
      };

      static void handle_response(ResponseArgs args)
      {
	log_metadata.info("received invalidate ack for " IDFMT, args.id);

	// switch on different types of objects that can have metadata
	switch(ID(args.id).type()) {
	case ID::ID_INSTANCE:
	  {
	    RegionInstanceImpl *impl = get_runtime()->get_instance_impl(args.id);
	    if(impl->metadata.handle_inval_ack(args.node)) {
	      log_metadata.info("last inval ack received for " IDFMT, args.id);
	    }
	    break;
	  }

	default:
	  assert(0);
	}
      }

      typedef ActiveMessageShortNoReply<METADATA_INVALIDATE_MSGID,
					RequestArgs,
					handle_request> RequestMessage;
      typedef ActiveMessageShortNoReply<METADATA_INVALIDATE_RPLID,
					ResponseArgs,
					handle_response> ResponseMessage;
      
      static void request(gasnet_node_t target, IDType id)
      {
	RequestArgs args;

	args.owner = gasnet_mynode();
	args.id = id;
	RequestMessage::request(target, args);
      }

      static void respond(gasnet_node_t target, IDType id)
      {
	ResponseArgs args;

	args.node = gasnet_mynode();
	args.id = id;
	ResponseMessage::request(target, args);
      }

      struct Functor {
      public:
        IDType id;
      public:
        Functor(IDType i) : id(i) { }
      public:
        inline void apply(gasnet_node_t target) { request(target, id); }
      };
    };

    bool MetadataBase::initiate_cleanup(IDType id)
    {
      NodeSet invals_to_send;
      {
	AutoHSLLock a(mutex);

	assert(state == STATE_VALID);

	if(remote_copies.empty()) {
	  state = STATE_INVALID;
	} else {
	  state = STATE_CLEANUP;
	  invals_to_send = remote_copies;
	}
      }

      // send invalidations outside the locked section
      if(invals_to_send.empty())
	return true;

      MetadataInvalidateMessage::Functor functor(id);
      invals_to_send.map(functor);
      //for(int node = 0; node < MAX_NUM_NODES; node++)
      //  if(invals_to_send.contains(node)) {
      //    MetadataInvalidateMessage::request(node, id);
      //    invals_to_send.remove(node);
      //    if(invals_to_send.empty()) break;
      //  }

      // can't free object until we receive all the acks
      return false;
    }

    void MetadataBase::handle_invalidate(void)
    {
      AutoHSLLock a(mutex);

      switch(state) {
      case STATE_VALID: 
	{
	  // was valid, now invalid (up to app to make sure no races exist)
	  state = STATE_INVALID;
	  break;
	}

      case STATE_REQUESTED:
	{
	  // hopefully rare case where invalidation passes response to initial request
	  state = STATE_INVALIDATE;
	  break;
	}

      default:
	assert(0);
      }
    }

    bool MetadataBase::handle_inval_ack(int sender)
    {
      bool last_copy;
      {
	AutoHSLLock a(mutex);

	assert(remote_copies.contains(sender));
	remote_copies.remove(sender);
	last_copy = remote_copies.empty();
      }

      return last_copy;
    }


    ///////////////////////////////////////////////////
    // RegionInstance

    RegionInstanceImpl::RegionInstanceImpl(RegionInstance _me, IndexSpace _is, Memory _memory, 
					   off_t _offset, size_t _size, ReductionOpID _redopid,
					   const DomainLinearization& _linear, size_t _block_size,
					   size_t _elmt_size, const std::vector<size_t>& _field_sizes,
					   const Realm::ProfilingRequestSet &reqs,
					   off_t _count_offset /*= 0*/, off_t _red_list_size /*= 0*/,
					   RegionInstance _parent_inst /*= NO_INST*/)
      : me(_me), memory(_memory)
    {
      metadata.linearization = _linear;

      metadata.block_size = _block_size;
      metadata.elmt_size = _elmt_size;

      metadata.field_sizes = _field_sizes;

      metadata.is = _is;
      metadata.alloc_offset = _offset;
      //metadata.access_offset = _offset + _adjust;
      metadata.size = _size;
      
      //StaticAccess<IndexSpaceImpl> rdata(_is.impl());
      //locked_data.first_elmt = rdata->first_elmt;
      //locked_data.last_elmt = rdata->last_elmt;

      metadata.redopid = _redopid;
      metadata.count_offset = _count_offset;
      metadata.red_list_size = _red_list_size;
      metadata.parent_inst = _parent_inst;

      metadata.mark_valid();

      lock.init(ID(me).convert<Reservation>(), ID(me).node());
      lock.in_use = true;

      if (!reqs.empty()) {
        requests = reqs;
        measurements.import_requests(requests);
        if (measurements.wants_measurement<
                          Realm::ProfilingMeasurements::InstanceTimeline>()) {
          timeline.instance = me;
          timeline.record_create_time();
        }
      }
    }

    // when we auto-create a remote instance, we don't know region/offset
    RegionInstanceImpl::RegionInstanceImpl(RegionInstance _me, Memory _memory)
      : me(_me), memory(_memory)
    {
      lock.init(ID(me).convert<Reservation>(), ID(me).node());
      lock.in_use = true;
    }

    RegionInstanceImpl::~RegionInstanceImpl(void) {}

    // helper function to figure out which field we're in
    static void find_field_start(const std::vector<size_t>& field_sizes, off_t byte_offset,
				 size_t size, off_t& field_start, int& field_size)
    {
      off_t start = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++) {
	assert((*it) > 0);
	if(byte_offset < (off_t)(*it)) {
	  assert((off_t)(byte_offset + size) <= (off_t)(*it));
	  field_start = start;
	  field_size = (*it);
	  return;
	}
	start += (*it);
	byte_offset -= (*it);
      }
      assert(0);
    }

    bool RegionInstanceImpl::get_strided_parameters(void *&base, size_t &stride,
						      off_t field_offset)
    {
      MemoryImpl *mem = get_runtime()->get_memory_impl(memory);

      // must have valid data by now - block if we have to
      metadata.await_data();

      off_t offset = metadata.alloc_offset;
      size_t elmt_stride;
      
      if (metadata.block_size == 1) {
        offset += field_offset;
        elmt_stride = metadata.elmt_size;
      } else {
        off_t field_start;
        int field_size;
        find_field_start(metadata.field_sizes, field_offset, 1, field_start, field_size);

        offset += (field_start * metadata.block_size) + (field_offset - field_start);
	elmt_stride = field_size;
      }

      base = mem->get_direct_ptr(offset, 0);
      if (!base) return false;

      // if the caller wants a particular stride and we differ (and have more
      //  than one element), fail
      if(stride != 0) {
        if((stride != elmt_stride) && (metadata.size > metadata.elmt_size))
          return false;
      } else {
        stride = elmt_stride;
      }

      // if there's a per-element offset, apply it after we've agreed with the caller on 
      //  what we're pretending the stride is
      const DomainLinearization& dl = metadata.linearization;
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
    }

    void RegionInstanceImpl::finalize_instance(void)
    {
      if (!requests.empty()) {
        if (measurements.wants_measurement<
                          Realm::ProfilingMeasurements::InstanceTimeline>()) {
          timeline.record_delete_time();
          measurements.add_measurement(timeline);
        }
        if (measurements.wants_measurement<
                          Realm::ProfilingMeasurements::InstanceMemoryUsage>()) {
          Realm::ProfilingMeasurements::InstanceMemoryUsage usage;
          usage.instance = me;
          usage.memory = memory;
          // Safe to read from meta-data here because we know we are
          // on the owner node so it has up to date copy
          usage.bytes = metadata.size;
          measurements.add_measurement(usage);
        }
        measurements.send_responses(requests);
        requests.clear();
      }
    }

    void *RegionInstanceImpl::Metadata::serialize(size_t& out_size) const
    {
      // figure out how much space we need
      out_size = (sizeof(IndexSpace) +
		  sizeof(off_t) +
		  sizeof(size_t) +
		  sizeof(ReductionOpID) +
		  sizeof(off_t) +
		  sizeof(off_t) +
		  sizeof(size_t) +
		  sizeof(size_t) +
		  sizeof(size_t) + (field_sizes.size() * sizeof(size_t)) +
		  sizeof(RegionInstance) +
		  (MAX_LINEARIZATION_LEN * sizeof(int)));
      void *data = malloc(out_size);
      char *pos = (char *)data;
#define S(val) do { memcpy(pos, &(val), sizeof(val)); pos += sizeof(val); } while(0)
      S(is);
      S(alloc_offset);
      S(size);
      S(redopid);
      S(count_offset);
      S(red_list_size);
      S(block_size);
      S(elmt_size);
      size_t l = field_sizes.size();
      S(l);
      for(size_t i = 0; i < l; i++) S(field_sizes[i]);
      S(parent_inst);
      linearization.serialize((int *)pos);
#undef S
      return data;
    }

    void RegionInstanceImpl::Metadata::deserialize(const void *in_data, size_t in_size)
    {
      const char *pos = (const char *)in_data;
#define S(val) do { memcpy(&(val), pos, sizeof(val)); pos += sizeof(val); } while(0)
      S(is);
      S(alloc_offset);
      S(size);
      S(redopid);
      S(count_offset);
      S(red_list_size);
      S(block_size);
      S(elmt_size);
      size_t l;
      S(l);
      field_sizes.resize(l);
      for(size_t i = 0; i < l; i++) S(field_sizes[i]);
      S(parent_inst);
      linearization.deserialize((const int *)pos);
#undef S
    }

    class DeferredInstDestroy : public EventWaiter {
    public:
      DeferredInstDestroy(RegionInstanceImpl *i) : impl(i) { }
      virtual ~DeferredInstDestroy(void) { }
    public:
      virtual bool event_triggered(void)
      {
        log_meta.info("instance destroyed: space=" IDFMT " id=" IDFMT "",
                 impl->metadata.is.id, impl->me.id);
        get_runtime()->get_memory_impl(impl->memory)->destroy_instance(impl->me, true); 
        return true;
      }

      virtual void print_info(FILE *f)
      {
        fprintf(f,"deferred instance destruction\n");
      }
    protected:
      RegionInstanceImpl *impl;
    };

  };
};

namespace Realm {

  using namespace LegionRuntime::LowLevel;

    AddressSpace RegionInstance::address_space(void) const
    {
      return ID(id).node();
    }

    IDType RegionInstance::local_id(void) const
    {
      return ID(id).index();
    }

    Memory RegionInstance::get_location(void) const
    {
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(*this);
      return i_impl->memory;
    }

    void RegionInstance::destroy(Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(*this);
      if (!wait_on.has_triggered())
      {
	EventImpl::add_waiter(wait_on, new DeferredInstDestroy(i_impl));
        return;
      }

      log_meta.info("instance destroyed: space=" IDFMT " id=" IDFMT "",
	       i_impl->metadata.is.id, this->id);
      get_runtime()->get_memory_impl(i_impl->memory)->destroy_instance(*this, true);
    }

    /*static*/ const RegionInstance RegionInstance::NO_INST = { 0 };

    // a generic accessor just holds a pointer to the impl and passes all 
    //  requests through
    RegionAccessor<AccessorType::Generic> RegionInstance::get_accessor(void) const
    {
      // request metadata (if needed), but don't block on it yet
      RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(*this);
      Event e = i_impl->metadata.request_data(ID(id).node(), id);
      if(!e.has_triggered())
	log_metadata.info("requested metadata in accessor creation: " IDFMT, id);
	
      return RegionAccessor<AccessorType::Generic>(AccessorType::Generic::Untyped((void *)i_impl));
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    ///////////////////////////////////////////////////
    // Events

    ///*static*/ EventImpl *EventImpl::first_free = 0;
    ///*static*/ gasnet_hsl_t EventImpl::freelist_mutex = GASNET_HSL_INITIALIZER;

    GenEventImpl::GenEventImpl(void)
      : me((IDType)-1), owner(-1)
    {
      generation = 0;
      gen_subscribed = 0;
      next_free = 0;
    }

    void GenEventImpl::init(ID _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      generation = 0;
      gen_subscribed = 0;
      next_free = 0;
    }

    struct EventSubscribeArgs {
      gasnet_node_t node;
      Event event;
      Event::gen_t previous_subscribe_gen;
    };

    void handle_event_subscribe(EventSubscribeArgs args);

    typedef ActiveMessageShortNoReply<EVENT_SUBSCRIBE_MSGID,
				      EventSubscribeArgs,
				      handle_event_subscribe> EventSubscribeMessage;

    struct EventTriggerArgs {
    public:
      gasnet_node_t node;
      Event event;
    public:
      void apply(gasnet_node_t target);
    };

    void handle_event_trigger(EventTriggerArgs args);

    typedef ActiveMessageShortNoReply<EVENT_TRIGGER_MSGID,
				      EventTriggerArgs,
				      handle_event_trigger> EventTriggerMessage;

    void EventTriggerArgs::apply(gasnet_node_t target)
    {
      EventTriggerMessage::request(target, *this);
    }

    static Logger::Category log_event("event");

    // only called for generational events
    void handle_event_subscribe(EventSubscribeArgs args)
    {
      log_event.debug("event subscription: node=%d event=" IDFMT "/%d",
		args.node, args.event.id, args.event.gen);

      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);

#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = args.event.id; 
        item.event_gen = args.event.gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif

      // early-out case: if we can see the generation needed has already
      //  triggered, signal without taking the mutex
      unsigned stale_gen = impl->generation;
      if(stale_gen >= args.event.gen) {
	log_event.debug("event subscription early-out: node=%d event=" IDFMT "/%d (<= %d)",
		  args.node, args.event.id, args.event.gen, stale_gen);
	EventTriggerArgs trigger_args;
	trigger_args.node = gasnet_mynode();
	trigger_args.event = args.event;
	trigger_args.event.gen = stale_gen;
	EventTriggerMessage::request(args.node, trigger_args);
	return;
      }

      {
	AutoHSLLock a(impl->mutex);
        // first trigger any generations which are below our current generation
        if(impl->generation > (args.previous_subscribe_gen)) {
          log_event.debug("event subscription already done: node=%d event=" IDFMT "/%d (<= %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	  EventTriggerArgs trigger_args;
	  trigger_args.node = gasnet_mynode();
	  trigger_args.event = args.event;
	  trigger_args.event.gen = impl->generation;
	  EventTriggerMessage::request(args.node, trigger_args);
        }

	// if the subscriber is asking about a generation that JUST triggered, the above trigger message
	//  is all we needed to do
	if(args.event.gen > impl->generation) {
	  // barrier logic is now separated, so we should never hear about an event generate beyond the one
	  //  that will trigger next
	  assert(args.event.gen <= (impl->generation + 1));

	  impl->remote_waiters.add(args.node);
	  log_event.debug("event subscription recorded: node=%d event=" IDFMT "/%d (> %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	}
      }
    } 

    void handle_event_trigger(EventTriggerArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_event.debug("Remote trigger of event " IDFMT "/%d from node %d!",
		args.event.id, args.event.gen, args.node);
      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);
      impl->trigger(args.event.gen, args.node);
    }


    static Barrier::timestamp_t barrier_adjustment_timestamp;

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    /*static*/ const Event Event::NO_EVENT = { 0, 0 };
    // Take this you POS c++ type system
    /* static */ const UserEvent UserEvent::NO_USER_EVENT = 
      *(static_cast<UserEvent*>(const_cast<Event*>(&Event::NO_EVENT)));
  //EventImpl *Event::impl(void) const
  //{
  //  return get_runtime()->get_event_impl(*this);
  //}

    bool Event::has_triggered(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if(!id) return true; // special case: NO_EVENT has always triggered
      EventImpl *e = get_runtime()->get_event_impl(*this);
      return e->has_triggered(gen);
    }

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return GenEventImpl::merge_events(wait_for);
    }

    /*static*/ Event Event::merge_events(Event ev1, Event ev2,
					 Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
					 Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return GenEventImpl::merge_events(ev1, ev2, ev3, ev4, ev5, ev6);
    }

    /*static*/ UserEvent UserEvent::create_user_event(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Event e = GenEventImpl::create_genevent()->current_event();
      assert(e.id != 0);
      UserEvent u;
      u.id = e.id;
      u.gen = e.gen;
      return u;
    }

    void UserEvent::trigger(Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      GenEventImpl *e = get_runtime()->get_genevent_impl(*this);
#ifdef EVENT_GRAPH_TRACE
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.info("Event Trigger: (" IDFMT ",%d) (" IDFMT 
                           ",%d) (" IDFMT ",%d)",
                            id, gen, wait_on.id, wait_on.gen,
                            enclosing.id, enclosing.gen);
#endif
      e->trigger(gen, gasnet_mynode(), wait_on);
    }

    ///////////////////////////////////////////////////
    // Barrier 

    /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals,
					       ReductionOpID redop_id /*= 0*/,
					       const void *initial_value /*= 0*/,
					       size_t initial_value_size /*= 0*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      BarrierImpl *impl = BarrierImpl::create_barrier(expected_arrivals, redop_id, initial_value, initial_value_size);
      Barrier b = impl->me.convert<Barrier>();
      b.gen = impl->generation + 1;
      b.timestamp = 0;

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Barrier Creation: " IDFMT " %d", b.id, expected_arrivals);
#endif

      return b;
    }

    void Barrier::destroy_barrier(void)
    {
      log_barrier.info("barrier destruction request: " IDFMT "/%d", id, gen);
    }

    Barrier Barrier::advance_barrier(void) const
    {
      Barrier nextgen;
      nextgen.id = id;
      nextgen.gen = gen + 1;
      nextgen.timestamp = 0;

      return nextgen;
    }

    Barrier Barrier::alter_arrival_count(int delta) const
    {
      timestamp_t timestamp = __sync_fetch_and_add(&barrier_adjustment_timestamp, 1);
#ifdef EVENT_GRAPH_TRACE
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.info("Barrier Alter: (" IDFMT ",%d) (" IDFMT
                           ",%d) %d", id, gen, enclosing.id, enclosing.gen, delta);
#endif
      BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
      impl->adjust_arrival(gen, delta, timestamp, Event::NO_EVENT, 0, 0);

      Barrier with_ts;
      with_ts.id = id;
      with_ts.gen = gen;
      with_ts.timestamp = timestamp;

      return with_ts;
    }

    Barrier Barrier::get_previous_phase(void) const
    {
      Barrier result = *this;
      result.gen--;
      return result;
    }

    void Barrier::arrive(unsigned count /*= 1*/, Event wait_on /*= Event::NO_EVENT*/,
			 const void *reduce_value /*= 0*/, size_t reduce_value_size /*= 0*/) const
    {
#ifdef EVENT_GRAPH_TRACE
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.info("Barrier Arrive: (" IDFMT ",%d) (" IDFMT
                           ",%d) (" IDFMT ",%d) %d",
                           id, gen, wait_on.id, wait_on.gen,
                           enclosing.id, enclosing.gen, count);
#endif
      // arrival uses the timestamp stored in this barrier object
      BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
      impl->adjust_arrival(gen, -count, timestamp, wait_on,
			   reduce_value, reduce_value_size);
    }

    bool Barrier::get_result(void *value, size_t value_size) const
    {
      BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
      return impl->get_result(gen, value, value_size);
    }
     
    void Event::wait(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if(!id) return;  // special case: never wait for NO_EVENT
      EventImpl *e = get_runtime()->get_event_impl(*this);

      // early out case too
      if(e->has_triggered(gen)) return;

      // waiting on an event does not count against the low level's time
      DetailedTimer::ScopedPush sp2(TIME_NONE);

      // are we a thread that knows how to do something useful while waiting?
      if(PreemptableThread::preemptable_sleep(*this))
	return;

      // maybe a GPU thread?
#ifdef USE_CUDA
      void *ptr = gasnett_threadkey_get(gpu_thread_ptr);
      if(ptr != 0) {
	//assert(0);
	//printf("oh, good - we're a gpu thread - we'll spin for now\n");
	//printf("waiting for " IDFMT "/%d\n", id, gen);
	while(!e->has_triggered(gen)) {
#ifdef __SSE2__
          _mm_pause();
#else
          usleep(1000);
#endif
        }
	//printf("done\n");
	return;
      }
#endif
      // we're probably screwed here - try waiting and polling gasnet while
      //  we wait
      //printf("waiting on event, polling gasnet to hopefully not die\n");
      while(!e->has_triggered(gen)) {
	// can't poll here - the GPU DMA code sometimes polls from inside an active
	//  message handler (consider turning polling back on once that's fixed)
	//do_some_polling();
#ifdef __SSE2__
	_mm_pause();
#endif
	// no sleep - we don't want an OS-scheduler-latency here
	//usleep(10000);
      }
      return;
      //assert(ptr != 0);
    }

    void Event::external_wait(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if(!id) return;  // special case: never wait for NO_EVENT
      EventImpl *e = get_runtime()->get_event_impl(*this);

      // early out case too
      if(e->has_triggered(gen)) return;

      // waiting on an event does not count against the low level's time
      DetailedTimer::ScopedPush sp2(TIME_NONE);

      e->external_wait(gen);
    }

};

namespace LegionRuntime {
  namespace LowLevel {


    // Perform our merging events in a lock free way
    class EventMerger : public EventWaiter {
    public:
      EventMerger(GenEventImpl *_finish_event)
	: count_needed(1), finish_event(_finish_event)
      {
      }

      virtual ~EventMerger(void)
      {
      }

      void add_event(Event wait_for)
      {
	if(wait_for.has_triggered()) return; // early out
        // Increment the count and then add ourselves
        __sync_fetch_and_add(&count_needed, 1);
	// step 2: enqueue ourselves on the input event
	EventImpl::add_waiter(wait_for, this);
      }

      // arms the merged event once you're done adding input events - just
      //  decrements the count for the implicit 'init done' event
      // return a boolean saying whether it triggered upon arming (which
      //  means the caller should delete this EventMerger)
      bool arm(void)
      {
	bool nuke = event_triggered();
        return nuke;
      }

      virtual bool event_triggered(void)
      {
	// save ID and generation because we can't reference finish_event after the
	// decrement (unless last_trigger ends up being true)
	IDType id = finish_event->me.id();
	Event::gen_t gen = finish_event->generation;

	int count_left = __sync_fetch_and_add(&count_needed, -1);

        // Put the logging first to avoid segfaults
        log_event.info("received trigger merged event " IDFMT "/%d (%d)",
		  id, gen, count_left);

	// count is the value before the decrement, so it was 1, it's now 0
	bool last_trigger = (count_left == 1);

	if(last_trigger) {
	  finish_event->trigger_current();
	}

        // caller can delete us if this was the last trigger
        return last_trigger;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"event merger: " IDFMT "/%d\n", finish_event->me.id(), finish_event->generation+1);
      }

    protected:
      int count_needed;
      GenEventImpl *finish_event;
    };

    // creates an event that won't trigger until all input events have
    /*static*/ Event GenEventImpl::merge_events(const std::set<Event>& wait_for)
    {
      if (wait_for.empty())
        return Event::NO_EVENT;
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      for(std::set<Event>::const_iterator it = wait_for.begin();
	  (it != wait_for.end()) && (wait_count < 2);
	  it++)
	if(!(*it).has_triggered()) {
	  if(!wait_count) first_wait = *it;
	  wait_count++;
	}
      log_event.info("merging events - at least %d not triggered",
		wait_count);

      // Avoid these optimizations if we are doing event graph tracing
#ifndef EVENT_GRAPH_TRACE
      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;
#else
      if (wait_for.size() == 1)
        return *(wait_for.begin());
#endif
      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      EventMerger *m = new EventMerger(finish_event);

      // get the Event for this GenEventImpl before any triggers can occur
      Event e = finish_event->current_event();

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) %ld", 
			   e.id, e.gen, wait_for.size());
#endif

      for(std::set<Event>::const_iterator it = wait_for.begin();
	  it != wait_for.end();
	  it++) {
	log_event.info("merged event " IDFMT "/%d waiting for " IDFMT "/%d",
		  finish_event->me.id(), finish_event->generation, (*it).id, (*it).gen);
	m->add_event(*it);
#ifdef EVENT_GRAPH_TRACE
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ",%d)",
                             finish_event->me.id(), finish_event->generation,
                             it->id, it->gen);
#endif
      }

      // once they're all added - arm the thing (it might go off immediately)
      if(m->arm())
        delete m;

      return e;
    }

    /*static*/ Event GenEventImpl::merge_events(Event ev1, Event ev2,
						Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
						Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      if(!ev6.has_triggered()) { first_wait = ev6; wait_count++; }
      if(!ev5.has_triggered()) { first_wait = ev5; wait_count++; }
      if(!ev4.has_triggered()) { first_wait = ev4; wait_count++; }
      if(!ev3.has_triggered()) { first_wait = ev3; wait_count++; }
      if(!ev2.has_triggered()) { first_wait = ev2; wait_count++; }
      if(!ev1.has_triggered()) { first_wait = ev1; wait_count++; }

      // Avoid these optimizations if we are doing event graph tracing
#ifndef EVENT_GRAPH_TRACE
      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;
#else
      int existential_count = 0;
      if (ev1.exists()) existential_count++;
      if (ev2.exists()) existential_count++;
      if (ev3.exists()) existential_count++;
      if (ev4.exists()) existential_count++;
      if (ev5.exists()) existential_count++;
      if (ev6.exists()) existential_count++;
      if (existential_count == 0)
        return Event::NO_EVENT;
      if (existential_count == 1)
      {
        if (ev1.exists()) return ev1;
        if (ev2.exists()) return ev2;
        if (ev3.exists()) return ev3;
        if (ev4.exists()) return ev4;
        if (ev5.exists()) return ev5;
        if (ev6.exists()) return ev6;
      }
#endif

      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      EventMerger *m = new EventMerger(finish_event);

      // get the Event for this GenEventImpl before any triggers can occur
      Event e = finish_event->current_event();

      m->add_event(ev1);
      m->add_event(ev2);
      m->add_event(ev3);
      m->add_event(ev4);
      m->add_event(ev5);
      m->add_event(ev6);

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) %d",
               finish_event->me.id(), finish_event->generation, existential_count);
      if (ev1.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev1.id, ev1.gen);
      if (ev2.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev2.id, ev2.gen);
      if (ev3.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev3.id, ev3.gen);
      if (ev4.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev4.id, ev4.gen);
      if (ev5.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev5.id, ev5.gen);
      if (ev6.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev6.id, ev6.gen);
#endif

      // once they're all added - arm the thing (it might go off immediately)
      if(m->arm())
        delete m;

      return e;
    }

    /*static*/ GenEventImpl *GenEventImpl::create_genevent(void)
    {
      GenEventImpl *impl = get_runtime()->local_event_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).type() == ID::ID_EVENT);

      log_event.info("event created: event=" IDFMT "/%d", impl->me.id(), impl->generation+1);
#ifdef EVENT_TRACING
      {
	EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
	item.event_id = impl->me.id();
	item.event_gen = impl->me.gen;
	item.action = EventTraceItem::ACT_CREATE;
      }
#endif
      return impl;
    }
    
    void GenEventImpl::check_for_catchup(Event::gen_t implied_trigger_gen)
    {
      // early out before we take a lock
      if(implied_trigger_gen <= generation) return;

      // now take a lock and see if we really need to catch up
      std::vector<EventWaiter *> stale_waiters;
      {
	AutoHSLLock a(mutex);

	if(implied_trigger_gen > generation) {
	  assert(owner != gasnet_mynode());  // cannot be a local event

	  log_event.info("event catchup: " IDFMT "/%d -> %d",
			 me.id(), generation, implied_trigger_gen);
	  generation = implied_trigger_gen;
	  stale_waiters.swap(local_waiters);  // we'll actually notify them below
	}
      }

      if(!stale_waiters.empty()) {
	for(std::vector<EventWaiter *>::iterator it = stale_waiters.begin();
	    it != stale_waiters.end();
	    it++)
	  (*it)->event_triggered();
      }
    }

    bool GenEventImpl::add_waiter(Event::gen_t needed_gen, EventWaiter *waiter)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me->id;
        item.event_gen = needed_gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif
      bool trigger_now = false;

      int subscribe_owner = -1;
      EventSubscribeArgs args;
      // initialization to make not-as-clever compilers happy
      args.node = 0;
      args.event = Event::NO_EVENT;
      args.previous_subscribe_gen = 0;
      {
	AutoHSLLock a(mutex);

	if(needed_gen > generation) {
	  log_event.debug("event not ready: event=" IDFMT "/%d owner=%d gen=%d subscr=%d",
		    me.id(), needed_gen, owner, generation, gen_subscribed);

	  // catchup code for remote events has been moved to get_genevent_impl, so
	  //  we should never be asking for a stale version here
	  assert(needed_gen == (generation + 1));

	  // do we need to subscribe?
	  if((owner != gasnet_mynode()) && (gen_subscribed < needed_gen)) {
	    gen_subscribed = needed_gen;
	    subscribe_owner = owner;
	    args.node = gasnet_mynode();
	    args.event = me.convert<Event>();
	    args.event.gen = needed_gen;
	  }

	  // now we add to the local waiter list
	  local_waiters.push_back(waiter);
	} else {
	  // event we are interested in has already triggered!
	  trigger_now = true; // actually do trigger outside of mutex
	}
      }

      if((subscribe_owner != -1))
	EventSubscribeMessage::request(owner, args);

      if(trigger_now) {
	bool nuke = waiter->event_triggered();
        if(nuke)
          delete waiter;
      }

      return true;  // waiter is always either enqueued or triggered right now
    }

    /*static*/ bool EventImpl::add_waiter(Event needed, EventWaiter *waiter)
    {
      return get_runtime()->get_event_impl(needed)->add_waiter(needed.gen, waiter);
    }

    bool GenEventImpl::has_triggered(Event::gen_t needed_gen)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = needed_gen;
        item.action = EventTraceItem::ACT_QUERY;
      }
#endif
      return (needed_gen <= generation);
    }

    class PthreadCondWaiter : public EventWaiter {
    public:
      PthreadCondWaiter(GASNetCondVar &_cv)
        : cv(_cv)
      {
      }
      virtual ~PthreadCondWaiter(void) 
      {
      }

      virtual bool event_triggered(void)
      {
        // Need to hold the lock to avoid the race
        AutoHSLLock(cv.mutex);
	cv.signal();
        // we're allocated on caller's stack, so deleting would be bad
        return false;
      }
      virtual void print_info(FILE *f) { fprintf(f,"external waiter\n"); }

    public:
      GASNetCondVar &cv;
    };

    void GenEventImpl::external_wait(Event::gen_t gen_needed)
    {
      GASNetCondVar cv(mutex);
      PthreadCondWaiter w(cv);
      {
	AutoHSLLock a(mutex);

	if(gen_needed > generation) {
	  local_waiters.push_back(&w);
    
	  if((owner != gasnet_mynode()) && (gen_needed > gen_subscribed)) {
	    printf("AAAH!  Can't subscribe to another node's event in external_wait()!\n");
	    exit(1);
	  }

	  // now just sleep on the condition variable - hope we wake up
	  cv.wait();
	}
      }
    }

    class DeferredEventTrigger : public EventWaiter {
    public:
      DeferredEventTrigger(GenEventImpl *_after_event)
	: after_event(_after_event)
      {}

      virtual ~DeferredEventTrigger(void) { }

      virtual bool event_triggered(void)
      {
	log_event.info("deferred trigger occuring: " IDFMT "/%d", after_event->me.id(), after_event->generation+1);
	after_event->trigger_current();
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred trigger: after=" IDFMT "/%d\n",
		after_event->me.id(), after_event->generation+1);
      }

    protected:
      GenEventImpl *after_event;
    };

    void GenEventImpl::trigger_current(void)
    {
      // wrapper triggers the next generation on the current node
      trigger(generation + 1, gasnet_mynode());
    }

    void GenEventImpl::trigger(Event::gen_t gen_triggered, int trigger_node, Event wait_on)
    {
      if(!wait_on.has_triggered()) {
	// deferred trigger
	// TODO: forward the deferred trigger to the owning node if it's remote
	log_event.info("deferring event trigger: in=" IDFMT "/%d out=" IDFMT "/%d",
		       wait_on.id, wait_on.gen, me.id(), gen_triggered);
	EventImpl::add_waiter(wait_on, new DeferredEventTrigger(this));
	return;
      }

      log_event.spew("event triggered: event=" IDFMT "/%d by node %d", 
		me.id(), gen_triggered, trigger_node);
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = gen_triggered;
        item.action = EventTraceItem::ACT_TRIGGER;
      }
#endif

      std::vector<EventWaiter *> to_wake;
      {
	AutoHSLLock a(mutex);

        // SJT: there is at least one unavoidable case where we'll receive
	//  duplicate trigger notifications, so if we see a triggering of
	//  an older generation, just ignore it
	if(gen_triggered <= generation) return;

	// in preparation for switching everybody over to trigger_current(), complain
	//  LOUDLY if this wouldn't actually be a triggering of the current generation
	if(gen_triggered != (generation + 1))
	  log_event.error("HELP!  non-current event generation being triggered: " IDFMT "/%d vs %d",
			  me.id(), gen_triggered, generation + 1);

        generation = gen_triggered;

	// grab whole list of local waiters - we'll trigger them once we let go of the lock
	//printf("[%d] LOCAL WAITERS: %zd\n", gasnet_mynode(), local_waiters.size());
	to_wake.swap(local_waiters);

	// notify remote waiters and/or event's actual owner
	if(owner == gasnet_mynode()) {
	  // send notifications to every other node that has subscribed
	  //  (except the one that triggered)
          if (!remote_waiters.empty())
          {
            EventTriggerArgs args;
            args.node = trigger_node;
            args.event = me.convert<Event>();
            args.event.gen = gen_triggered;

            NodeSet send_mask;
            send_mask.swap(remote_waiters);
            send_mask.map(args);
            //for(int node = 0; node < MAX_NUM_NODES; node++)
            //  if (send_mask.contains(node) && (node != trigger_node))
            //    EventTriggerMessage::request(node, args);
          }
	} else {
	  if(((unsigned)trigger_node) == gasnet_mynode()) {
	    // if we're not the owner, we just send to the owner and let him
	    //  do the broadcast (assuming the trigger was local)
	    //assert(remote_waiters == 0);

	    EventTriggerArgs args;
	    args.node = trigger_node;
	    args.event = me.convert<Event>();
	    args.event.gen = gen_triggered;
	    EventTriggerMessage::request(owner, args);
	  }
	}
      }

      // if this is one of our events, put ourselves on the free
      //  list (we don't need our lock for this)
      if(owner == gasnet_mynode()) {
	get_runtime()->local_event_free_list->free_entry(this);
      }

      // now that we've let go of the lock, notify all the waiters who wanted
      //  this event generation (or an older one)
      {
	for(std::vector<EventWaiter *>::iterator it = to_wake.begin();
	    it != to_wake.end();
	    it++) {
	  bool nuke = (*it)->event_triggered();
          if(nuke) {
            //printf("deleting: "); (*it)->print_info(); fflush(stdout);
            delete (*it);
          }
        }
      }
    }

    /*static*/ BarrierImpl *BarrierImpl::create_barrier(unsigned expected_arrivals,
							ReductionOpID redopid,
							const void *initial_value /*= 0*/,
							size_t initial_value_size /*= 0*/)
    {
      BarrierImpl *impl = get_runtime()->local_barrier_free_list->alloc_entry();
      assert(impl);
      assert(impl->me.type() == ID::ID_BARRIER);

      // set the arrival count
      impl->base_arrival_count = expected_arrivals;

      if(redopid == 0) {
	assert(initial_value_size == 0);
	impl->redop_id = 0;
	impl->redop = 0;
	impl->initial_value = 0;
	impl->value_capacity = 0;
	impl->final_values = 0;
      } else {
	impl->redop_id = redopid;  // keep the ID too so we can share it
	impl->redop = get_runtime()->reduce_op_table[redopid];

	assert(initial_value != 0);
	assert(initial_value_size == impl->redop->sizeof_lhs);

	impl->initial_value = (char *)malloc(initial_value_size);
	memcpy(impl->initial_value, initial_value, initial_value_size);

	impl->value_capacity = 0;
	impl->final_values = 0;
      }

      // and let the barrier rearm as many times as necessary without being released
      impl->free_generation = (unsigned)-1;

      log_barrier.info("barrier created: " IDFMT "/%d base_count=%d redop=%d",
		       impl->me.id(), impl->generation, impl->base_arrival_count, redopid);
#ifdef EVENT_TRACING
      {
	EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
	item.event_id = impl->me.id();
	item.event_gen = impl->me.gen;
	item.action = EventTraceItem::ACT_CREATE;
      }
#endif
      return impl;
    }

    BarrierImpl::BarrierImpl(void)
      : me((IDType)-1), owner(-1)
    {
      generation = 0;
      gen_subscribed = 0;
      first_generation = free_generation = 0;
      next_free = 0;
      remote_subscribe_gens.clear();
      remote_trigger_gens.clear();
      base_arrival_count = 0;
      redop = 0;
      initial_value = 0;
      value_capacity = 0;
      final_values = 0;
    }

    void BarrierImpl::init(ID _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      generation = 0;
      gen_subscribed = 0;
      first_generation = free_generation = 0;
      next_free = 0;
      remote_subscribe_gens.clear();
      remote_trigger_gens.clear();
      base_arrival_count = 0;
      redop = 0;
      initial_value = 0;
      value_capacity = 0;
      final_values = 0;
    }

    static const int BARRIER_TIMESTAMP_NODEID_SHIFT = 48;

    struct BarrierAdjustMessage {
      struct RequestArgs : public BaseMedium {
	Barrier barrier;
	int delta;
        Event wait_on;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen)
      {
	log_barrier.info("received barrier arrival: delta=%d in=" IDFMT "/%d out=" IDFMT "/%d (%llx)",
			 args.delta, args.wait_on.id, args.wait_on.gen, args.barrier.id, args.barrier.gen, args.barrier.timestamp);
	BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
	impl->adjust_arrival(args.barrier.gen, args.delta, args.barrier.timestamp, args.wait_on,
			     datalen ? data : 0, datalen);
      }

      typedef ActiveMessageMediumNoReply<BARRIER_ADJUST_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(gasnet_node_t target, Barrier barrier, int delta, Event wait_on,
			       const void *data, size_t datalen)
      {
	RequestArgs args;
	
	args.barrier = barrier;
	args.delta = delta;
        args.wait_on = wait_on;

	Message::request(target, args, data, datalen, PAYLOAD_COPY);
      }
    };

    struct BarrierSubscribeMessage {
      struct RequestArgs {
	gasnet_node_t node;
	IDType barrier_id;
	Event::gen_t subscribe_gen;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<BARRIER_SUBSCRIBE_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(gasnet_node_t target, IDType barrier_id, Event::gen_t subscribe_gen)
      {
	RequestArgs args;

	args.node = gasnet_mynode();
	args.barrier_id = barrier_id;
	args.subscribe_gen = subscribe_gen;

	Message::request(target, args);
      }
    };

    struct BarrierTriggerMessage {
      struct RequestArgs : public BaseMedium {
	gasnet_node_t node;
	IDType barrier_id;
	Event::gen_t trigger_gen;
	Event::gen_t previous_gen;
	Event::gen_t first_generation;
	ReductionOpID redop_id;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<BARRIER_TRIGGER_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(gasnet_node_t target, IDType barrier_id,
			       Event::gen_t trigger_gen, Event::gen_t previous_gen,
			       Event::gen_t first_generation, ReductionOpID redop_id,
			       const void *data, size_t datalen)
      {
	RequestArgs args;

	args.node = gasnet_mynode();
	args.barrier_id = barrier_id;
	args.trigger_gen = trigger_gen;
	args.previous_gen = previous_gen;
	args.first_generation = first_generation;
	args.redop_id = redop_id;

	Message::request(target, args, data, datalen, PAYLOAD_COPY);
      }
    };

    class DeferredBarrierArrival : public EventWaiter {
    public:
      DeferredBarrierArrival(Barrier _barrier, int _delta, const void *_data, size_t _datalen)
	: barrier(_barrier), delta(_delta), data(bytedup(_data, _datalen)), datalen(_datalen)
      {}

      virtual ~DeferredBarrierArrival(void)
      {
	if(data)
	  free(data);
      }

      virtual bool event_triggered(void)
      {
	log_barrier.info("deferred barrier arrival: " IDFMT "/%d (%llx), delta=%d",
			 barrier.id, barrier.gen, barrier.timestamp, delta);
	BarrierImpl *impl = get_runtime()->get_barrier_impl(barrier);
	impl->adjust_arrival(barrier.gen, delta, barrier.timestamp, Event::NO_EVENT, data, datalen);
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred arrival: barrier=" IDFMT "/%d (%llx), delta=%d datalen=%zd\n",
		barrier.id, barrier.gen, barrier.timestamp, delta, datalen);
      }

    protected:
      Barrier barrier;
      int delta;
      void *data;
      size_t datalen;
    };

    class BarrierImpl::Generation {
     public:
      struct PerNodeUpdates {
        Barrier::timestamp_t last_ts;
        std::map<Barrier::timestamp_t, int> pending;
      };

      int unguarded_delta;
      std::vector<EventWaiter *> local_waiters;
      std::map<int, PerNodeUpdates *> pernode;
      

      Generation(void) : unguarded_delta(0) {}
      ~Generation(void)
      {
        for(std::map<int, PerNodeUpdates *>::iterator it = pernode.begin();
            it != pernode.end();
            it++)
          delete (it->second);
      }

      void handle_adjustment(Barrier::timestamp_t ts, int delta)
      {
	if(ts == 0) {
	  // simple case - apply delta directly
	  unguarded_delta += delta;
	  return;
	}

        int node = ts >> BARRIER_TIMESTAMP_NODEID_SHIFT;
        PerNodeUpdates *pn;
        std::map<int, PerNodeUpdates *>::iterator it = pernode.find(node);
        if(it != pernode.end()) {
          pn = it->second;
        } else {
          pn = new PerNodeUpdates;
          pernode[node] = pn;
        }
        if(delta > 0) {
          // TODO: really need two timestamps to properly order increments
          unguarded_delta += delta;
          pn->last_ts = ts;
          std::map<Barrier::timestamp_t, int>::iterator it2 = pn->pending.begin();
          while((it2 != pn->pending.end()) && (it2->first <= pn->last_ts)) {
            log_barrier.info("applying pending delta: %llx/%d", it2->first, it2->second);
            unguarded_delta += it2->second;
            pn->pending.erase(it2);
            it2 = pn->pending.begin();
          }
        } else {
          // if the timestamp is late enough, we can apply this directly
          if(ts <= pn->last_ts) {
            log_barrier.info("adjustment can be applied immediately: %llx/%d (%llx)",
                             ts, delta, pn->last_ts);
            unguarded_delta += delta;
          } else {
            log_barrier.info("adjustment must be deferred: %llx/%d (%llx)",
                             ts, delta, pn->last_ts);
            pn->pending[ts] += delta;
          }
        }
      }
    };

    struct RemoteNotification {
      unsigned node;
      Event::gen_t trigger_gen, previous_gen;
    };

    // used to adjust a barrier's arrival count either up or down
    // if delta > 0, timestamp is current time (on requesting node)
    // if delta < 0, timestamp says which positive adjustment this arrival must wait for
    void BarrierImpl::adjust_arrival(Event::gen_t barrier_gen, int delta, 
				     Barrier::timestamp_t timestamp, Event wait_on,
				     const void *reduce_value, size_t reduce_value_size)
    {
      if(!wait_on.has_triggered()) {
	// deferred arrival
	Barrier b = me.convert<Barrier>();
	b.gen = barrier_gen;
	b.timestamp = timestamp;
#ifndef DEFER_ARRIVALS_LOCALLY
        if(owner != gasnet_mynode()) {
	  // let deferral happen on owner node (saves latency if wait_on event
          //   gets triggered there)
          //printf("sending deferred arrival to %d for " IDFMT "/%d (" IDFMT "/%d)\n",
          //       owner, e.id, e.gen, wait_on.id, wait_on.gen);
	  log_barrier.info("forwarding deferred barrier arrival: delta=%d in=" IDFMT "/%d out=" IDFMT "/%d (%llx)",
			   delta, wait_on.id, wait_on.gen, b.id, b.gen, b.timestamp);
	  BarrierAdjustMessage::send_request(owner, b, delta, wait_on, reduce_value, reduce_value_size);
	  return;
        }
#endif
	log_barrier.info("deferring barrier arrival: delta=%d in=" IDFMT "/%d out=" IDFMT "/%d (%llx)",
			 delta, wait_on.id, wait_on.gen, me.id(), barrier_gen, timestamp);
	EventImpl::add_waiter(wait_on, new DeferredBarrierArrival(b, delta, 
								  reduce_value, reduce_value_size));
	return;
      }

      log_barrier.info("barrier adjustment: event=" IDFMT "/%d delta=%d ts=%llx", 
		       me.id(), barrier_gen, delta, timestamp);

#ifdef DEBUG_BARRIER_REDUCTIONS
      if(reduce_value_size) {
        char buffer[129];
	for(size_t i = 0; (i < reduce_value_size) && (i < 64); i++)
	  sprintf(buffer+2*i, "%02x", ((const unsigned char *)reduce_value)[i]);
	log_barrier.info("barrier reduction: event=" IDFMT "/%d size=%zd data=%s",
	                 me.id(), barrier_gen, reduce_value_size, buffer);
      }
#endif

      if(owner != gasnet_mynode()) {
	// all adjustments handled by owner node
	Barrier b = me.convert<Barrier>();
	b.gen = barrier_gen;
	b.timestamp = timestamp;
	BarrierAdjustMessage::send_request(owner, b, delta, Event::NO_EVENT, reduce_value, reduce_value_size);
	return;
      }

      // can't actually trigger while holding the lock, so remember which generation(s),
      //  if any, to trigger and do it at the end
      Event::gen_t trigger_gen = 0;
      std::vector<EventWaiter *> local_notifications;
      std::vector<RemoteNotification> remote_notifications;
      Event::gen_t oldest_previous = 0;
      void *final_values_copy = 0;
      {
	AutoHSLLock a(mutex);

	// sanity checks - is this a valid barrier?
	assert(generation < free_generation);
	assert(base_arrival_count > 0);

	// update whatever generation we're told to
	{
	  assert(barrier_gen > generation);
	  Generation *g;
	  std::map<Event::gen_t, Generation *>::iterator it = generations.find(barrier_gen);
	  if(it != generations.end()) {
	    g = it->second;
	  } else {
	    g = new Generation;
	    generations[barrier_gen] = g;
	    log_barrier.info("added tracker for barrier " IDFMT ", generation %d",
			     me.id(), barrier_gen);
	  }

	  g->handle_adjustment(timestamp, delta);
	}

	// if the update was to the next generation, it may cause one or more generations
	//  to trigger
	if(barrier_gen == (generation + 1)) {
	  std::map<Event::gen_t, Generation *>::iterator it = generations.begin();
	  while((it != generations.end()) &&
		(it->first == (generation + 1)) &&
		((base_arrival_count + it->second->unguarded_delta) == 0)) {
	    // keep the list of local waiters to wake up once we release the lock
	    local_notifications.insert(local_notifications.end(), 
				       it->second->local_waiters.begin(), it->second->local_waiters.end());
	    trigger_gen = generation = it->first;
	    delete it->second;
	    generations.erase(it);
	    it = generations.begin();
	  }

	  // if any triggers occurred, figure out which remote nodes need notifications
	  //  (i.e. any who have subscribed)
	  if(generation >= barrier_gen) {
	    std::map<unsigned, Event::gen_t>::iterator it = remote_subscribe_gens.begin();
	    while(it != remote_subscribe_gens.end()) {
	      RemoteNotification rn;
	      rn.node = it->first;
	      if(it->second <= generation) {
		// we have fulfilled the entire subscription
		rn.trigger_gen = it->second;
		std::map<unsigned, Event::gen_t>::iterator to_nuke = it++;
		remote_subscribe_gens.erase(to_nuke);
	      } else {
		// subscription remains valid
		rn.trigger_gen = generation;
		it++;
	      }
	      // also figure out what the previous generation this node knew about was
	      {
		std::map<unsigned, Event::gen_t>::iterator it2 = remote_trigger_gens.find(rn.node);
		if(it2 != remote_trigger_gens.end()) {
		  rn.previous_gen = it2->second;
		  it2->second = rn.trigger_gen;
		} else {
		  rn.previous_gen = first_generation;
		  remote_trigger_gens[rn.node] = rn.trigger_gen;
		}
	      }
	      if(remote_notifications.empty() || (rn.previous_gen < oldest_previous))
		oldest_previous = rn.previous_gen;
	      remote_notifications.push_back(rn);
	    }
	  }
	}

	// do we have reduction data to apply?  we can do this even if the actual adjustment is
	//  being held - no need to have lots of reduce values lying around
	if(reduce_value_size > 0) {
	  assert(redop != 0);
	  assert(redop->sizeof_rhs == reduce_value_size);

	  // do we have space for this reduction result yet?
	  int rel_gen = barrier_gen - first_generation;
	  assert(rel_gen > 0);

	  if((size_t)rel_gen > value_capacity) {
	    size_t new_capacity = rel_gen;
	    final_values = (char *)realloc(final_values, new_capacity * redop->sizeof_lhs);
	    while(value_capacity < new_capacity) {
	      memcpy(final_values + (value_capacity * redop->sizeof_lhs), initial_value, redop->sizeof_lhs);
	      value_capacity += 1;
	    }
	  }

	  redop->apply(final_values + ((rel_gen - 1) * redop->sizeof_lhs), reduce_value, 1, true);
	}

	// do this AFTER we actually update the reduction value above :)
	// if any remote notifications are going to occur and we have reduction values, make a copy so
	//  we have something stable after we let go of the lock
	if(trigger_gen && redop) {
	  int rel_gen = oldest_previous + 1 - first_generation;
	  assert(rel_gen > 0);
	  int count = trigger_gen - oldest_previous;
	  final_values_copy = bytedup(final_values + ((rel_gen - 1) * redop->sizeof_lhs),
				      count * redop->sizeof_lhs);
	}
      }

      if(trigger_gen != 0) {
	log_barrier.info("barrier trigger: event=" IDFMT "/%d", 
			 me.id(), trigger_gen);

	// notify local waiters first
	for(std::vector<EventWaiter *>::const_iterator it = local_notifications.begin();
	    it != local_notifications.end();
	    it++) {
	  bool nuke = (*it)->event_triggered();
	  if(nuke)
	    delete (*it);
	}

	// now do remote notifications
	for(std::vector<RemoteNotification>::const_iterator it = remote_notifications.begin();
	    it != remote_notifications.end();
	    it++) {
	  log_barrier.info("sending remote trigger notification: " IDFMT "/%d -> %d, dest=%d",
			   me.id(), (*it).previous_gen, (*it).trigger_gen, (*it).node);
	  void *data = 0;
	  size_t datalen = 0;
	  if(final_values_copy) {
	    data = (char *)final_values_copy + (((*it).previous_gen - oldest_previous) * redop->sizeof_lhs);
	    datalen = ((*it).trigger_gen - (*it).previous_gen) * redop->sizeof_lhs;
	  }
	  BarrierTriggerMessage::send_request((*it).node, me.id(), (*it).trigger_gen, (*it).previous_gen,
					      first_generation, redop_id, data, datalen);
	}
      }

      // free our copy of the final values, if we had one
      if(final_values_copy)
	free(final_values_copy);
    }

    bool BarrierImpl::has_triggered(Event::gen_t needed_gen)
    {
      // no need to take lock to check current generation
      if(needed_gen <= generation) return true;

      // if we're not the owner, subscribe if we haven't already
      if(owner != gasnet_mynode()) {
	Event::gen_t previous_subscription;
	// take lock to avoid duplicate subscriptions
	{
	  AutoHSLLock a(mutex);
	  previous_subscription = gen_subscribed;
	  if(gen_subscribed < needed_gen)
	    gen_subscribed = needed_gen;
	}

	if(previous_subscription < needed_gen) {
	  log_barrier.info("subscribing to barrier " IDFMT "/%d", me.id(), needed_gen);
	  BarrierSubscribeMessage::send_request(owner, me.id(), needed_gen);
	}
      }

      // whether or not we subscribed, the answer for now is "no"
      return false;
    }

    void BarrierImpl::external_wait(Event::gen_t needed_gen)
    {
      assert(0);
    }

    bool BarrierImpl::add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/)
    {
      bool trigger_now = false;
      {
	AutoHSLLock a(mutex);

	if(needed_gen > generation) {
	  Generation *g;
	  std::map<Event::gen_t, Generation *>::iterator it = generations.find(needed_gen);
	  if(it != generations.end()) {
	    g = it->second;
	  } else {
	    g = new Generation;
	    generations[needed_gen] = g;
	    log_barrier.info("added tracker for barrier " IDFMT ", generation %d",
			     me.id(), needed_gen);
	  }
	  g->local_waiters.push_back(waiter);

	  // a call to has_triggered should have already handled the necessary subscription
	  assert((owner == gasnet_mynode()) || (gen_subscribed >= needed_gen));
	} else {
	  // needed generation has already occurred - trigger this waiter once we let go of lock
	  trigger_now = true;
	}
      }

      if(trigger_now) {
	bool nuke = waiter->event_triggered();
	if(nuke)
	  delete waiter;
      }

      return true;
    }

    /*static*/ void BarrierSubscribeMessage::handle_request(BarrierSubscribeMessage::RequestArgs args)
    {
      Barrier b;
      b.id = args.barrier_id;
      b.gen = args.subscribe_gen;
      BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

      // take the lock and add the subscribing node - notice if they need to be notified for
      //  any generations that have already triggered
      Event::gen_t trigger_gen = 0;
      Event::gen_t previous_gen = 0;
      void *final_values_copy = 0;
      size_t final_values_size = 0;
      {
	AutoHSLLock a(impl->mutex);

	// make sure the subscription is for this "lifetime" of the barrier
	assert(args.subscribe_gen > impl->first_generation);

	bool already_subscribed = false;
	{
	  std::map<unsigned, Event::gen_t>::iterator it = impl->remote_subscribe_gens.find(args.node);
	  if(it != impl->remote_subscribe_gens.end()) {
	    // a valid subscription should always be for a generation that hasn't
	    //  triggered yet
	    assert(it->second > impl->generation);
	    if(it->second >= args.subscribe_gen)
	      already_subscribed = true;
	    else
	      it->second = args.subscribe_gen;
	  } else {
	    // new subscription - don't reset remote_trigger_gens because the node may have
	    //  been subscribed in the past
	    // NOTE: remote_subscribe_gens should only hold subscriptions for
	    //  generations that haven't triggered, so if we're subscribing to 
	    //  an old generation, don't add it
	    if(args.subscribe_gen > impl->generation)
	      impl->remote_subscribe_gens[args.node] = args.subscribe_gen;
	  }
	}

	// as long as we're not already subscribed to this generation, check to see if
	//  any trigger notifications are needed
	if(!already_subscribed && (impl->generation > impl->first_generation)) {
	  std::map<unsigned, Event::gen_t>::iterator it = impl->remote_trigger_gens.find(args.node);
	  if((it == impl->remote_trigger_gens.end()) or (it->second < impl->generation)) {
	    previous_gen = ((it == impl->remote_trigger_gens.end()) ?
			      impl->first_generation :
			      it->second);
	    trigger_gen = impl->generation;
	    impl->remote_trigger_gens[args.node] = impl->generation;

	    if(impl->redop) {
	      int rel_gen = previous_gen + 1 - impl->first_generation;
	      assert(rel_gen > 0);
	      final_values_size = (trigger_gen - previous_gen) * impl->redop->sizeof_lhs;
	      final_values_copy = bytedup(impl->final_values + ((rel_gen - 1) * impl->redop->sizeof_lhs),
					  final_values_size);
	    }
	  }
	}
      }

      // send trigger message outside of lock, if needed
      if(trigger_gen > 0) {
	log_barrier.info("sending immediate barrier trigger: " IDFMT "/%d -> %d",
			 args.barrier_id, previous_gen, trigger_gen);
	BarrierTriggerMessage::send_request(args.node, args.barrier_id, trigger_gen, previous_gen,
					    impl->first_generation, impl->redop_id,
					    final_values_copy, final_values_size);
      }

      if(final_values_copy)
	free(final_values_copy);
    }

    /*static*/ void BarrierTriggerMessage::handle_request(BarrierTriggerMessage::RequestArgs args,
							  const void *data, size_t datalen)
    {
      log_barrier.info("received remote barrier trigger: " IDFMT "/%d -> %d",
		       args.barrier_id, args.previous_gen, args.trigger_gen);

      Barrier b;
      b.id = args.barrier_id;
      b.gen = args.trigger_gen;
      BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

      // we'll probably end up with a list of local waiters to notify
      std::vector<EventWaiter *> local_notifications;
      {
	AutoHSLLock a(impl->mutex);

	// it's theoretically possible for multiple trigger messages to arrive out
	//  of order, so check if this message triggers the oldest possible range
	if(args.previous_gen == impl->generation) {
	  // see if we can pick up any of the held triggers too
	  while(!impl->held_triggers.empty()) {
	    std::map<Event::gen_t, Event::gen_t>::iterator it = impl->held_triggers.begin();
	    // if it's not contiguous, we're done
	    if(it->first != args.trigger_gen) break;
	    // it is contiguous, so absorb it into this message and remove the held trigger
	    log_barrier.info("collapsing future trigger: " IDFMT "/%d -> %d -> %d",
			     args.barrier_id, args.previous_gen, args.trigger_gen, it->second);
	    args.trigger_gen = it->second;
	    impl->held_triggers.erase(it);
	  }

	  impl->generation = args.trigger_gen;

	  // now iterate through any generations up to and including the latest triggered
	  //  generation, and accumulate local waiters to notify
	  while(!impl->generations.empty()) {
	    std::map<Event::gen_t, BarrierImpl::Generation *>::iterator it = impl->generations.begin();
	    if(it->first > args.trigger_gen) break;

	    local_notifications.insert(local_notifications.end(),
				       it->second->local_waiters.begin(),
				       it->second->local_waiters.end());
	    delete it->second;
	    impl->generations.erase(it);
	  }
	} else {
	  // hold this trigger until we get messages for the earlier generation(s)
	  log_barrier.info("holding future trigger: " IDFMT "/%d (%d -> %d)",
			   args.barrier_id, impl->generation, 
			   args.previous_gen, args.trigger_gen);
	  impl->held_triggers[args.previous_gen] = args.trigger_gen;
	}

	// is there any data we need to store?
	if(datalen) {
	  assert(args.redop_id != 0);

	  // TODO: deal with invalidation of previous instance of a barrier
	  impl->redop_id = args.redop_id;
	  impl->redop = get_runtime()->reduce_op_table[args.redop_id];
	  impl->first_generation = args.first_generation;

	  int rel_gen = args.trigger_gen - impl->first_generation;
	  assert(rel_gen > 0);
	  if(impl->value_capacity < (size_t)rel_gen) {
	    size_t new_capacity = rel_gen;
	    impl->final_values = (char *)realloc(impl->final_values, new_capacity * impl->redop->sizeof_lhs);
	    // no need to initialize new entries - we'll overwrite them now or when data does show up
	    impl->value_capacity = new_capacity;
	  }
	  assert(datalen == (impl->redop->sizeof_lhs * (args.trigger_gen - args.previous_gen)));
	  memcpy(impl->final_values + ((rel_gen - 1) * impl->redop->sizeof_lhs), data, datalen);
	}
      }

      // with lock released, perform any local notifications
      for(std::vector<EventWaiter *>::const_iterator it = local_notifications.begin();
	  it != local_notifications.end();
	  it++) {
	bool nuke = (*it)->event_triggered();
	if(nuke)
	  delete (*it);
      }
    }

    void show_event_waiters(FILE *f = stdout)
    {
      fprintf(f,"PRINTING ALL PENDING EVENTS:\n");
      for(unsigned i = 0; i < gasnet_nodes(); i++) {
	Node *n = &get_runtime()->nodes[i];
        // Iterate over all the events and get their implementations
        for (unsigned long j = 0; j < n->events.max_entries(); j++) {
          if (!n->events.has_entry(j))
            continue;
	  GenEventImpl *e = n->events.lookup_entry(j, i/*node*/);
	  AutoHSLLock a2(e->mutex);

	  // print anything with either local or remote waiters
	  if(e->local_waiters.empty() && e->remote_waiters.empty())
	    continue;

          fprintf(f,"Event " IDFMT ": gen=%d subscr=%d local=%zd remote=%zd\n",
		  e->me.id(), e->generation, e->gen_subscribed, 
		  e->local_waiters.size(),
                  e->remote_waiters.size());
	  for(std::vector<EventWaiter *>::iterator it = e->local_waiters.begin();
	      it != e->local_waiters.end();
	      it++) {
	      fprintf(f, "  [%d] L:%p ", e->generation + 1, *it);
	      (*it)->print_info(f);
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

          fprintf(f,"Barrier " IDFMT ": gen=%d subscr=%d\n",
                  b->me.id(), b->generation, b->gen_subscribed);
          for (std::map<Event::gen_t, BarrierImpl::Generation*>::const_iterator git = 
                b->generations.begin(); git != b->generations.end(); git++)
          {
            const std::vector<EventWaiter*> &waiters = git->second->local_waiters;
            for (std::vector<EventWaiter*>::const_iterator it = 
                  waiters.begin(); it != waiters.end(); it++)
            {
              fprintf(f, "  [%d] L:%p ", git->first, *it);
              (*it)->print_info(f);
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

      fprintf(f,"DONE\n");
      fflush(f);
    }

    bool BarrierImpl::get_result(Event::gen_t result_gen, void *value, size_t value_size)
    {
      // take the lock so we can safely see how many results (if any) are on hand
      AutoHSLLock al(mutex);

      // generation hasn't triggered yet?
      if(result_gen > generation) return false;

      // if it has triggered, we should have the data
      int rel_gen = result_gen - first_generation;
      assert(rel_gen > 0);
      assert((size_t)rel_gen <= value_capacity);

      assert(redop != 0);
      assert(value_size == redop->sizeof_lhs);
      assert(value != 0);
      memcpy(value, final_values + ((rel_gen - 1) * redop->sizeof_lhs), redop->sizeof_lhs);
      return true;
    }

    ///////////////////////////////////////////////////
    // Reservations 

    /*static*/ ReservationImpl *ReservationImpl::first_free = 0;
    /*static*/ GASNetHSL ReservationImpl::freelist_mutex;

    ReservationImpl::ReservationImpl(void)
    {
      init(Reservation::NO_RESERVATION, -1);
    }

    Logger::Category log_reservation("reservation");

    void ReservationImpl::init(Reservation _me, unsigned _init_owner,
			  size_t _data_size /*= 0*/)
    {
      me = _me;
      owner = _init_owner;
      count = ZERO_COUNT;
      log_reservation.spew("count init " IDFMT "=[%p]=%d", me.id, &count, count);
      mode = 0;
      in_use = false;
      remote_waiter_mask = NodeSet(); 
      remote_sharer_mask = NodeSet();
      requested = false;
      if(_data_size) {
	local_data = malloc(_data_size);
	local_data_size = _data_size;
        own_local = true;
      } else {
        local_data = 0;
	local_data_size = 0;
        own_local = false;
      }
    }

    /*static*/ void /*ReservationImpl::*/handle_lock_request(LockRequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ReservationImpl *impl = get_runtime()->get_lock_impl(args.lock);

      log_reservation.debug("reservation request: reservation=" IDFMT ", node=%d, mode=%d",
	       args.lock.id, args.node, args.mode);

      // can't send messages while holding mutex, so remember args and who
      //  (if anyone) to send to
      int req_forward_target = -1;
      int grant_target = -1;
      LockGrantArgs g_args;
      NodeSet copy_waiters;

      do {
	AutoHSLLock a(impl->mutex);

	// case 1: we don't even own the lock any more - pass the request on
	//  to whoever we think the owner is
	if(impl->owner != gasnet_mynode()) {
	  // can reuse the args we were given
	  log_reservation.debug(              "forwarding reservation request: reservation=" IDFMT ", from=%d, to=%d, mode=%d",
		   args.lock.id, args.node, impl->owner, args.mode);
	  req_forward_target = impl->owner;
	  break;
	}

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(impl->me).node() != gasnet_mynode()) ||
	       impl->in_use);

	// case 2: we're the owner, and nobody is holding the lock, so grant
	//  it to the (original) requestor
	if((impl->count == ReservationImpl::ZERO_COUNT) && 
           (impl->remote_sharer_mask.empty())) {
          assert(impl->remote_waiter_mask.empty());

	  log_reservation.debug(              "granting reservation request: reservation=" IDFMT ", node=%d, mode=%d",
		   args.lock.id, args.node, args.mode);
	  g_args.lock = args.lock;
	  g_args.mode = 0; // always give it exclusively for now
	  grant_target = args.node;
          copy_waiters = impl->remote_waiter_mask;

	  impl->owner = args.node;
	  break;
	}

	// case 3: we're the owner, but we can't grant the lock right now -
	//  just set a bit saying that the node is waiting and get back to
	//  work
	log_reservation.debug(            "deferring reservation request: reservation=" IDFMT ", node=%d, mode=%d (count=%d cmode=%d)",
		 args.lock.id, args.node, args.mode, impl->count, impl->mode);
        impl->remote_waiter_mask.add(args.node);
      } while(0);

      if(req_forward_target != -1)
      {
	LockRequestMessage::request(req_forward_target, args);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = impl->me.id;
          item.owner = req_forward_target;
          item.action = LockTraceItem::ACT_FORWARD_REQUEST;
        }
#endif
      }

      if(grant_target != -1)
      {
        // Make a buffer for storing our waiter mask and the the local data
	size_t waiter_count = copy_waiters.size();
        size_t payload_size = ((waiter_count+1) * sizeof(int)) + impl->local_data_size;
        int *payload = (int*)malloc(payload_size);
	int *pos = payload;
	*pos++ = waiter_count;
	// TODO: switch to iterator
        ReservationImpl::PackFunctor functor(pos);
        copy_waiters.map(functor);
        pos = functor.pos;
	//for(int i = 0; i < MAX_NUM_NODES; i++)
	//  if(copy_waiters.contains(i))
	//    *pos++ = i;
        memcpy(pos, impl->local_data, impl->local_data_size);
	LockGrantMessage::request(grant_target, g_args,
                                  payload, payload_size, PAYLOAD_FREE);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = impl->me.id;
          item.owner = grant_target;
          item.action = LockTraceItem::ACT_REMOTE_GRANT;
        }
#endif
      }
    }

    /*static*/ void /*ReservationImpl::*/handle_lock_release(LockReleaseArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      assert(0);
    }

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_reservation.debug(          "reservation request granted: reservation=" IDFMT " mode=%d", // mask=%lx",
	       args.lock.id, args.mode); //, args.remote_waiter_mask);

      std::deque<GenEventImpl *> to_wake;

      ReservationImpl *impl = get_runtime()->get_lock_impl(args.lock);
      {
	AutoHSLLock a(impl->mutex);

	// make sure we were really waiting for this lock
	assert(impl->owner != gasnet_mynode());
	assert(impl->requested);

	// first, update our copy of the protected data (if any)
	const int *pos = (const int *)data;

	size_t waiter_count = *pos++;
	assert(datalen == (((waiter_count+1) * sizeof(int)) + impl->local_data_size));
	impl->remote_waiter_mask.clear();
	for(size_t i = 0; i < waiter_count; i++)
	  impl->remote_waiter_mask.add(*pos++);

	// is there local data to grab?
	if(impl->local_data_size > 0)
          memcpy(impl->local_data, pos, impl->local_data_size);

	if(args.mode == 0) // take ownership if given exclusive access
	  impl->owner = gasnet_mynode();
	impl->mode = args.mode;
	impl->requested = false;

	bool any_local = impl->select_local_waiters(to_wake);
	assert(any_local);
      }

      for(std::deque<GenEventImpl *>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_reservation.debug("release trigger: reservation=" IDFMT " event=" IDFMT "/%d",
			args.lock.id, (*it)->me.id(), (*it)->generation+1);
	(*it)->trigger_current();
      }
    }

    Event ReservationImpl::acquire(unsigned new_mode, bool exclusive,
				     GenEventImpl *after_lock /* = 0*/)
    {
      Event after_lock_event = after_lock ? after_lock->current_event() : Event::NO_EVENT;

      log_reservation.debug(		      "local reservation request: reservation=" IDFMT " mode=%d excl=%d event=" IDFMT "/%d count=%d impl=%p",
		      me.id, new_mode, exclusive, 
		      after_lock_event.id,
		      after_lock_event.gen, count, this);

      // collapse exclusivity into mode
      if(exclusive) new_mode = MODE_EXCL;

      bool got_lock = false;
      int lock_request_target = -1;
      LockRequestArgs args;
      // initialization to make not-as-clever compilers happy
      args.node = 0;
      args.lock = Reservation::NO_RESERVATION;
      args.mode = 0;

      {
	AutoHSLLock a(mutex); // hold mutex on lock while we check things

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(me).node() != gasnet_mynode()) ||
	       in_use);

	if(owner == gasnet_mynode()) {
#ifdef LOCK_TRACING
          {
            LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
            item.lock_id = me.id;
            item.owner = gasnet_mynode();
            item.action = LockTraceItem::ACT_LOCAL_REQUEST;
          }
#endif
	  // case 1: we own the lock
	  // can we grant it?
	  if((count == ZERO_COUNT) || ((mode == new_mode) && (mode != MODE_EXCL))) {
	    mode = new_mode;
	    count++;
	    log_reservation.spew("count ++(1) [%p]=%d", &count, count);
	    got_lock = true;
#ifdef LOCK_TRACING
            {
              LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
              item.lock_id = me.id;
              item.owner = gasnet_mynode();
              item.action = LockTraceItem::ACT_LOCAL_GRANT;
            }
#endif
	  }
	} else {
	  // somebody else owns it
	
	  // are we sharing?
	  if((count > ZERO_COUNT) && (mode == new_mode)) {
	    // we're allowed to grant additional sharers with the same mode
	    assert(mode != MODE_EXCL);
	    if(mode == new_mode) {
	      count++;
	      log_reservation.spew("count ++(2) [%p]=%d", &count, count);
	      got_lock = true;
	    }
	  }
	
	  // if we didn't get the lock, we'll have to ask for it from the
	  //  other node (even if we're currently sharing with the wrong mode)
	  if(!got_lock && !requested) {
	    log_reservation.debug(                "requesting reservation: reservation=" IDFMT " node=%d mode=%d",
		     me.id, owner, new_mode);
	    args.node = gasnet_mynode();
	    args.lock = me;
	    args.mode = new_mode;
	    lock_request_target = owner;
	    // don't actually send message here because we're holding the
	    //  lock's mutex, which'll be bad if we get a message related to
	    //  this lock inside gasnet calls
	  
	    requested = true;
	  }
	}
  
	log_reservation.debug(            "local reservation result: reservation=" IDFMT " got=%d req=%d count=%d",
		 me.id, got_lock ? 1 : 0, requested ? 1 : 0, count);

	// if we didn't get the lock, put our event on the queue of local
	//  waiters - create an event if we weren't given one to use
	if(!got_lock) {
	  if(!after_lock) {
	    after_lock = GenEventImpl::create_genevent();
	    after_lock_event = after_lock->current_event();
	  }
	  local_waiters[new_mode].push_back(after_lock);
	}
      }

      if(lock_request_target != -1)
      {
	LockRequestMessage::request(lock_request_target, args);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = lock_request_target;
          item.action = LockTraceItem::ACT_REMOTE_REQUEST;
        }
#endif
      }

      // if we got the lock, trigger an event if we were given one
      if(got_lock && after_lock) 
	after_lock->trigger(after_lock_event.gen, gasnet_mynode());

      return after_lock_event;
    }

    // factored-out code to select one or more local waiters on a lock
    //  fills events to trigger into 'to_wake' and returns true if any were
    //  found - NOTE: ASSUMES LOCK IS ALREADY HELD!
    bool ReservationImpl::select_local_waiters(std::deque<GenEventImpl *>& to_wake)
    {
      if(local_waiters.size() == 0)
	return false;

      // favor the local waiters
      log_reservation.debug("reservation going to local waiter: size=%zd first=%d(%zd)",
	       local_waiters.size(), 
	       local_waiters.begin()->first,
	       local_waiters.begin()->second.size());
	
      // further favor exclusive waiters
      if(local_waiters.find(MODE_EXCL) != local_waiters.end()) {
	std::deque<GenEventImpl *>& excl_waiters = local_waiters[MODE_EXCL];
	to_wake.push_back(excl_waiters.front());
	excl_waiters.pop_front();
	  
	// if the set of exclusive waiters is empty, delete it
	if(excl_waiters.size() == 0)
	  local_waiters.erase(MODE_EXCL);
	  
	mode = MODE_EXCL;
	count = ZERO_COUNT + 1;
	log_reservation.spew("count <-1 [%p]=%d", &count, count);
      } else {
	// pull a whole list of waiters that want to share with the same mode
	std::map<unsigned, std::deque<GenEventImpl *> >::iterator it = local_waiters.begin();
	
	mode = it->first;
	count = ZERO_COUNT + it->second.size();
	log_reservation.spew("count <-waiters [%p]=%d", &count, count);
	assert(count > ZERO_COUNT);
	// grab the list of events wanting to share the lock
	to_wake.swap(it->second);
	local_waiters.erase(it);  // actually pull list off map!
	// TODO: can we share with any other nodes?
      }
#ifdef LOCK_TRACING
      {
        LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
        item.lock_id = me.id;
        item.owner = gasnet_mynode();
        item.action = LockTraceItem::ACT_LOCAL_GRANT;
      }
#endif

      return true;
    }

    void ReservationImpl::release(void)
    {
      // make a list of events that we be woken - can't do it while holding the
      //  lock's mutex (because the event we trigger might try to take the lock)
      std::deque<GenEventImpl *> to_wake;

      int release_target = -1;
      LockReleaseArgs r_args;
      // initialization to make not-as-clever compilers happy
      r_args.node = 0;
      r_args.lock = Reservation::NO_RESERVATION;

      int grant_target = -1;
      LockGrantArgs g_args;
      NodeSet copy_waiters;

      do {
	log_reservation.debug(            "release: reservation=" IDFMT " count=%d mode=%d owner=%d", // share=%lx wait=%lx",
			me.id, count, mode, owner); //, remote_sharer_mask, remote_waiter_mask);
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	assert(count > ZERO_COUNT);

	// if this isn't the last holder of the lock, just decrement count
	//  and return
	count--;
	log_reservation.spew("count -- [%p]=%d", &count, count);
	log_reservation.debug(            "post-release: reservation=" IDFMT " count=%d mode=%d", // share=%lx wait=%lx",
		 me.id, count, mode); //, remote_sharer_mask, remote_waiter_mask);
	if(count > ZERO_COUNT) break;

	// case 1: if we were sharing somebody else's lock, tell them we're
	//  done
	if(owner != gasnet_mynode()) {
	  assert(mode != MODE_EXCL);
	  mode = 0;

	  r_args.node = gasnet_mynode();
	  r_args.lock = me;
	  release_target = owner;
	  break;
	}

	// case 2: we own the lock, so we can give it to another waiter
	//  (local or remote)
	bool any_local = select_local_waiters(to_wake);
	assert(!any_local || (to_wake.size() > 0));

	if(!any_local && (!remote_waiter_mask.empty())) {
	  // nobody local wants it, but another node does
	  //HACK int new_owner = remote_waiter_mask.find_first_set();
	  // TODO: use iterator - all we need is *begin()
          int new_owner = 0;  while(!remote_waiter_mask.contains(new_owner)) new_owner++;
          remote_waiter_mask.remove(new_owner);

	  log_reservation.debug(              "reservation going to remote waiter: new=%d", // mask=%lx",
		   new_owner); //, remote_waiter_mask);

	  g_args.lock = me;
	  g_args.mode = 0; // TODO: figure out shared cases
	  grant_target = new_owner;
          copy_waiters = remote_waiter_mask;

	  owner = new_owner;
          remote_waiter_mask = NodeSet();
	}
      } while(0);

      if(release_target != -1)
      {
	log_reservation.debug("releasing reservation " IDFMT " back to owner %d",
			      me.id, release_target);
	LockReleaseMessage::request(release_target, r_args);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = release_target;
          item.action = LockTraceItem::ACT_REMOTE_RELEASE;
        }
#endif
      }

      if(grant_target != -1)
      {
        // Make a buffer for storing our waiter mask and the the local data
	size_t waiter_count = copy_waiters.size();
        size_t payload_size = ((waiter_count+1) * sizeof(int)) + local_data_size;
        int *payload = (int*)malloc(payload_size);
	int *pos = payload;
	*pos++ = waiter_count;
	// TODO: switch to iterator
        PackFunctor functor(pos);
        copy_waiters.map(functor);
        pos = functor.pos;
	//for(int i = 0; i < MAX_NUM_NODES; i++)
	//  if(copy_waiters.contains(i))
	//    *pos++ = i;
        memcpy(pos, local_data, local_data_size);
	LockGrantMessage::request(grant_target, g_args,
                                  payload, payload_size, PAYLOAD_FREE);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = grant_target;
          item.action = LockTraceItem::ACT_REMOTE_GRANT;
        }
#endif
      }

      for(std::deque<GenEventImpl *>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_reservation.debug("release trigger: reservation=" IDFMT " event=" IDFMT "/%d",
			me.id, (*it)->me.id(), (*it)->generation + 1);
	(*it)->trigger_current();
      }
    }

    bool ReservationImpl::is_locked(unsigned check_mode, bool excl_ok)
    {
      // checking the owner can be done atomically, so doesn't need mutex
      if(owner != gasnet_mynode()) return false;

      // conservative check on lock count also doesn't need mutex
      if(count == ZERO_COUNT) return false;

      // a careful check of the lock mode and count does require the mutex
      bool held;
      {
	AutoHSLLock a(mutex);

	held = ((count > ZERO_COUNT) &&
		((mode == check_mode) || ((mode == 0) && excl_ok)));
      }

      return held;
    }

    class DeferredLockRequest : public EventWaiter {
    public:
      DeferredLockRequest(Reservation _lock, unsigned _mode, bool _exclusive,
			  GenEventImpl *_after_lock)
	: lock(_lock), mode(_mode), exclusive(_exclusive), after_lock(_after_lock) {}

      virtual ~DeferredLockRequest(void) { }

      virtual bool event_triggered(void)
      {
	get_runtime()->get_lock_impl(lock)->acquire(mode, exclusive, after_lock);
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred lock: lock=" IDFMT " after=" IDFMT "/%d\n",
		lock.id, after_lock->me.id(), after_lock->generation + 1);
      }

    protected:
      Reservation lock;
      unsigned mode;
      bool exclusive;
      GenEventImpl *after_lock;
    };

    class DeferredUnlockRequest : public EventWaiter {
    public:
      DeferredUnlockRequest(Reservation _lock)
	: lock(_lock) {}

      virtual ~DeferredUnlockRequest(void) { }

      virtual bool event_triggered(void)
      {
	get_runtime()->get_lock_impl(lock)->release();
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred unlock: lock=" IDFMT "\n",
	       lock.id);
      }
    protected:
      Reservation lock;
    };

    void ReservationImpl::release_reservation(void)
    {
      // take the lock's mutex to sanity check it and clear the in_use field
      {
	AutoHSLLock al(mutex);

	// should only get here if the current node holds an exclusive lock
	assert(owner == gasnet_mynode());
	assert(count == 1 + ZERO_COUNT);
	assert(mode == MODE_EXCL);
	assert(local_waiters.size() == 0);
        assert(remote_waiter_mask.empty());
	assert(in_use);
        // Mark that we no longer own our data
        if (own_local)
          free(local_data);
        local_data = NULL;
        local_data_size = 0;
        own_local = false;
      	in_use = false;
	count = ZERO_COUNT;
      }
      log_reservation.info() << "releasing reservation: reservation=" << me;

      get_runtime()->local_reservation_free_list->free_entry(this);
    }

    class DeferredLockDestruction : public EventWaiter {
    public:
      DeferredLockDestruction(Reservation _lock) : lock(_lock) {}

      virtual ~DeferredLockDestruction(void) { }

      virtual bool event_triggered(void)
      {
	get_runtime()->get_lock_impl(lock)->release_reservation();
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred lock destruction: lock=" IDFMT "\n", lock.id);
      }

    protected:
      Reservation lock;
    };

    struct DestroyReservationArgs {
      Reservation actual;
      Reservation dummy;
    };

    void handle_destroy_lock(DestroyReservationArgs args)
    {
      args.actual.destroy_reservation();
    }

    typedef ActiveMessageShortNoReply<DESTROY_LOCK_MSGID, DestroyReservationArgs,
				      handle_destroy_lock> DestroyLockMessage;

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    /*static*/ const Reservation Reservation::NO_RESERVATION = { 0 };

    Event Reservation::acquire(unsigned mode /* = 0 */, bool exclusive /* = true */,
		     Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //printf("LOCK(" IDFMT ", %d, %d, " IDFMT ") -> ", id, mode, exclusive, wait_on.id);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	Event e = get_runtime()->get_lock_impl(*this)->acquire(mode, exclusive);
	//printf("(" IDFMT "/%d)\n", e.id, e.gen);
	return e;
      } else {
	GenEventImpl *after_lock = GenEventImpl::create_genevent();
	Event e = after_lock->current_event();
	EventImpl::add_waiter(wait_on, new DeferredLockRequest(*this, mode, exclusive, after_lock));
	//printf("*(" IDFMT "/%d)\n", after_lock.id, after_lock.gen);
	return e;
      }
    }

    // releases a held lock - release can be deferred until an event triggers
    void Reservation::release(Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	get_runtime()->get_lock_impl(*this)->release();
      } else {
	EventImpl::add_waiter(wait_on, new DeferredUnlockRequest(*this));
      }
    }

    // Create a new lock, destroy an existing lock
    /*static*/ Reservation Reservation::create_reservation(size_t _data_size /*= 0*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //DetailedTimer::ScopedPush sp(18);

      // see if the freelist has an event we can reuse
      ReservationImpl *impl = get_runtime()->local_reservation_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).type() == ID::ID_LOCK);
      if(impl) {
	AutoHSLLock al(impl->mutex);

	assert(impl->owner == gasnet_mynode());
	assert(impl->count == ReservationImpl::ZERO_COUNT);
	assert(impl->mode == ReservationImpl::MODE_EXCL);
	assert(impl->local_waiters.size() == 0);
        assert(impl->remote_waiter_mask.empty());
	assert(!impl->in_use);

	impl->in_use = true;

	log_reservation.info("reservation reused: reservation=" IDFMT "", impl->me.id);
	return impl->me;
      }
      assert(false);
      return Reservation::NO_RESERVATION;
#if 0
      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(get_runtime()->nodes[gasnet_mynode()].mutex);

      std::vector<ReservationImpl>& locks = 
        get_runtime()->nodes[gasnet_mynode()].locks;

#ifdef REUSE_LOCKS
      // try to find an lock we can reuse
      for(std::vector<ReservationImpl>::iterator it = locks.begin();
	  it != locks.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != gasnet_mynode())) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == gasnet_mynode())) {
	  // now we really have the lock
	  (*it).in_use = true;
	  Reservation r = (*it).me;
	  return r;
	}
      }
#endif

      // couldn't reuse an lock - make a new one
      // TODO: take a lock here!?
      unsigned index = locks.size();
      assert((index+1) < MAX_LOCAL_LOCKS);
      locks.resize(index + 1);
      Reservation r = ID(ID::ID_LOCK, gasnet_mynode(), index).convert<Reservation>();
      locks[index].init(r, gasnet_mynode());
      locks[index].in_use = true;
      get_runtime()->nodes[gasnet_mynode()].num_locks = index + 1;
      log_reservation.info() << "created new reservation: reservation=" << r;
      return r;
#endif
    }

    void Reservation::destroy_reservation()
    {
      // a lock has to be destroyed on the node that created it
      if(ID(*this).node() != gasnet_mynode()) {
        DestroyReservationArgs args;
        args.actual = *this;
	DestroyLockMessage::request(ID(*this).node(), args);
	return;
      }

      // to destroy a local lock, we first must lock it (exclusively)
      ReservationImpl *lock_impl = get_runtime()->get_lock_impl(*this);
      Event e = lock_impl->acquire(0, true);
      if(!e.has_triggered()) {
	EventImpl::add_waiter(e, new DeferredLockDestruction(*this));
      } else {
	// got grant immediately - can release reservation now
	lock_impl->release_reservation();
      }
    }

    ///////////////////////////////////////////////////
    // Memory

    AddressSpace Memory::address_space(void) const
    {
      return ID(id).node();
    }

    IDType Memory::local_id(void) const
    {
      return ID(id).index();
    }

    Memory::Kind Memory::kind(void) const
    {
      return get_runtime()->get_memory_impl(*this)->get_kind();
    }

    size_t Memory::capacity(void) const
    {
      return get_runtime()->get_memory_impl(*this)->size;
    }

    /*static*/ const Memory Memory::NO_MEMORY = { 0 };

};

namespace LegionRuntime {
  namespace LowLevel {

    struct RemoteMemAllocReqArgs {
      int sender;
      void *resp_ptr;
      Memory memory;
      size_t size;
    };

    void handle_remote_mem_alloc_req(RemoteMemAllocReqArgs args);

    typedef ActiveMessageShortNoReply<REMOTE_MALLOC_MSGID,
				      RemoteMemAllocReqArgs,
				      handle_remote_mem_alloc_req> RemoteMemAllocRequest;

    struct RemoteMemAllocRespArgs {
      void *resp_ptr;
      off_t offset;
    };

    void handle_remote_mem_alloc_resp(RemoteMemAllocRespArgs args);

    typedef ActiveMessageShortNoReply<REMOTE_MALLOC_RPLID,
				      RemoteMemAllocRespArgs,
				      handle_remote_mem_alloc_resp> RemoteMemAllocResponse;

    void handle_remote_mem_alloc_req(RemoteMemAllocReqArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //printf("[%d] handling remote alloc of size %zd\n", gasnet_mynode(), args.size);
      off_t offset = get_runtime()->get_memory_impl(args.memory)->alloc_bytes(args.size);
      //printf("[%d] remote alloc will return %d\n", gasnet_mynode(), result);

      RemoteMemAllocRespArgs r_args;
      r_args.resp_ptr = args.resp_ptr;
      r_args.offset = offset;
      RemoteMemAllocResponse::request(args.sender, r_args);
    }

    void handle_remote_mem_alloc_resp(RemoteMemAllocRespArgs args)
    {
      HandlerReplyFuture<off_t> *f = static_cast<HandlerReplyFuture<off_t> *>(args.resp_ptr);
      f->set(args.offset);
    }

    // make bad offsets really obvious (+1 PB)
    static const off_t ZERO_SIZE_INSTANCE_OFFSET = 1ULL << 50;

    MemoryImpl::MemoryImpl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment, Memory::Kind _lowlevel_kind)
      : me(_me), size(_size), kind(_kind), alignment(_alignment), lowlevel_kind(_lowlevel_kind)
#ifdef REALM_PROFILE_MEMORY_USAGE
      , usage(0), peak_usage(0), peak_footprint(0)
#endif
    {
    }

    MemoryImpl::~MemoryImpl(void)
    {
#ifdef REALM_PROFILE_MEMORY_USAGE
      printf("Memory " IDFMT " usage: peak=%zd (%.1f MB) footprint=%zd (%.1f MB)\n",
	     me.id, 
	     peak_usage, peak_usage / 1048576.0,
	     peak_footprint, peak_footprint / 1048576.0);
#endif
    }

    off_t MemoryImpl::alloc_bytes_local(size_t size)
    {
      AutoHSLLock al(mutex);

      // for zero-length allocations, return a special "offset"
      if(size == 0) {
	return this->size + ZERO_SIZE_INSTANCE_OFFSET;
      }

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding allocation from %zd to %zd",
			  size, size + (alignment - leftover));
	  size += (alignment - leftover);
	}
      }
      // HACK: pad the size by a bit to see if we have people falling off
      //  the end of their allocations
      size += 0;

      // try to minimize footprint by allocating at the highest address possible
      if(!free_blocks.empty()) {
	std::map<off_t, off_t>::iterator it = free_blocks.end();
	do {
	  --it;  // predecrement since we started at the end

	  if(it->second == (off_t)size) {
	    // perfect match
	    off_t retval = it->first;
	    free_blocks.erase(it);
	    log_malloc.info("alloc full block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, retval);
#ifdef REALM_PROFILE_MEMORY_USAGE
	    usage += size;
	    if(usage > peak_usage) peak_usage = usage;
	    size_t footprint = this->size - retval;
	    if(footprint > peak_footprint) peak_footprint = footprint;
#endif
	    return retval;
	  }
	
	  if(it->second > (off_t)size) {
	    // some left over
	    off_t leftover = it->second - size;
	    off_t retval = it->first + leftover;
	    it->second = leftover;
	    log_malloc.info("alloc partial block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, retval);
#ifdef REALM_PROFILE_MEMORY_USAGE
	    usage += size;
	    if(usage > peak_usage) peak_usage = usage;
	    size_t footprint = this->size - retval;
	    if(footprint > peak_footprint) peak_footprint = footprint;
#endif
	    return retval;
	  }
	} while(it != free_blocks.begin());
      }

      // no blocks large enough - boo hoo
      log_malloc.info("alloc FAILED: mem=" IDFMT " size=%zd", me.id, size);
      return -1;
    }

    void MemoryImpl::free_bytes_local(off_t offset, size_t size)
    {
      log_malloc.info() << "free block: mem=" << me << " size=" << size << " ofs=" << offset;
      AutoHSLLock al(mutex);

      // frees of zero bytes should have the special offset
      if(size == 0) {
	assert((size_t)offset == this->size + ZERO_SIZE_INSTANCE_OFFSET);
	return;
      }

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding free from %zd to %zd",
			  size, size + (alignment - leftover));
	  size += (alignment - leftover);
	}
      }

#ifdef REALM_PROFILE_MEMORY_USAGE
      usage -= size;
      // only made things smaller, so can't impact the peak usage
#endif

      if(free_blocks.size() > 0) {
	// find the first existing block that comes _after_ us
	std::map<off_t, off_t>::iterator after = free_blocks.lower_bound(offset);
	if(after != free_blocks.end()) {
	  // found one - is it the first one?
	  if(after == free_blocks.begin()) {
	    // yes, so no "before"
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }
	    free_blocks[offset] = size;
	  } else {
	    // no, get range that comes before us too
	    std::map<off_t, off_t>::iterator before = after; before--;

	    // if we're adjacent to the after, merge with it
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }

	    // if we're adjacent with the before, grow it instead of adding
	    //  a new range
	    assert((before->first + before->second) <= offset);
	    if((before->first + before->second) == offset) {
	      before->second += size;
	    } else {
	      free_blocks[offset] = size;
	    }
	  }
	} else {
	  // nothing's after us, so just see if we can merge with the range
	  //  that's before us

	  std::map<off_t, off_t>::iterator before = after; before--;

	  // if we're adjacent with the before, grow it instead of adding
	  //  a new range
	  assert((before->first + before->second) <= offset);
	  if((before->first + before->second) == offset) {
	    before->second += size;
	  } else {
	    free_blocks[offset] = size;
	  }
	}
      } else {
	// easy case - nothing was free, so now just our block is
	free_blocks[offset] = size;
      }
    }

    off_t MemoryImpl::alloc_bytes_remote(size_t size)
    {
      // RPC over to owner's node for allocation

      HandlerReplyFuture<off_t> result;

      RemoteMemAllocReqArgs args;
      args.memory = me;
      args.size = size;
      args.sender = gasnet_mynode();
      args.resp_ptr = &result;

      RemoteMemAllocRequest::request(ID(me).node(), args);

      // wait for result to come back
      result.wait();
      return result.get();
    }

    void MemoryImpl::free_bytes_remote(off_t offset, size_t size)
    {
      assert(0);
    }

    Memory::Kind MemoryImpl::get_kind(void) const
    {
      return lowlevel_kind;
    }

    static Logger::Category log_copy("copy");

    class LocalCPUMemory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      LocalCPUMemory(Memory _me, size_t _size,
		     void *prealloc_base = 0, bool _registered = false) 
	: MemoryImpl(_me, _size, MKIND_SYSMEM, ALIGNMENT, 
            (_registered ? Memory::REGDMA_MEM : Memory::SYSTEM_MEM))
      {
	if(prealloc_base) {
	  base = (char *)prealloc_base;
	  prealloced = true;
	  registered = _registered;
	} else {
	  // allocate our own space
	  // enforce alignment on the whole memory range
	  base_orig = new char[_size + ALIGNMENT - 1];
	  size_t ofs = reinterpret_cast<size_t>(base_orig) % ALIGNMENT;
	  if(ofs > 0) {
	    base = base_orig + (ALIGNMENT - ofs);
	  } else {
	    base = base_orig;
	  }
	  prealloced = false;
	  assert(!_registered);
	  registered = false;
	}
	log_copy.debug("CPU memory at %p, size = %zd%s%s", base, _size, 
		       prealloced ? " (prealloced)" : "", registered ? " (registered)" : "");
	free_blocks[0] = _size;
      }

      virtual ~LocalCPUMemory(void)
      {
	if(!prealloced)
	  delete[] base_orig;
      }

#ifdef USE_CUDA
      // For pinning CPU memories for use with asynchronous
      // GPU copies
      void pin_memory(GPUProcessor *proc)
      {
        proc->register_host_memory(base, size);
      }
#endif

      virtual RegionInstance create_instance(IndexSpace r,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
                                             const Realm::ProfilingRequestSet &reqs,
					     RegionInstance parent_inst)
      {
	return create_instance_local(r, linearization_bits, bytes_needed,
				     block_size, element_size, field_sizes, redopid,
				     list_size, reqs, parent_inst);
      }

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy)
      {
	destroy_instance_local(i, local_destroy);
      }

      virtual off_t alloc_bytes(size_t size)
      {
	return alloc_bytes_local(size);
      }

      virtual void free_bytes(off_t offset, size_t size)
      {
	free_bytes_local(offset, size);
      }

      virtual void get_bytes(off_t offset, void *dst, size_t size)
      {
	memcpy(dst, base+offset, size);
      }

      virtual void put_bytes(off_t offset, const void *src, size_t size)
      {
	memcpy(base+offset, src, size);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size)
      {
	return (base + offset);
      }

      virtual int get_home_node(off_t offset, size_t size)
      {
	return gasnet_mynode();
      }

    public: //protected:
      char *base, *base_orig;
      bool prealloced, registered;
    };

    class RemoteMemory : public MemoryImpl {
    public:
      RemoteMemory(Memory _me, size_t _size, Memory::Kind k, void *_regbase)
	: MemoryImpl(_me, _size, _regbase ? MKIND_RDMA : MKIND_REMOTE, 0, k), regbase(_regbase)
      {
      }

      virtual RegionInstance create_instance(IndexSpace r,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
                                             const Realm::ProfilingRequestSet &reqs,
					     RegionInstance parent_inst)
      {
	return create_instance_remote(r, linearization_bits, bytes_needed,
				      block_size, element_size, field_sizes, redopid,
				      list_size, reqs, parent_inst);
      }

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy)
      {
	destroy_instance_remote(i, local_destroy);
      }

      virtual off_t alloc_bytes(size_t size)
      {
	return alloc_bytes_remote(size);
      }

      virtual void free_bytes(off_t offset, size_t size)
      {
	free_bytes_remote(offset, size);
      }

      virtual void get_bytes(off_t offset, void *dst, size_t size)
      {
	// this better be an RDMA-able memory
#ifdef USE_GASNET
	assert(kind == MemoryImpl::MKIND_RDMA);
	void *srcptr = ((char *)regbase) + offset;
	gasnet_get(dst, ID(me).node(), srcptr, size);
#else
	assert(0 && "no remote get_bytes without GASNET");
#endif
      }

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size)
      {
	return 0;
      }

      virtual int get_home_node(off_t offset, size_t size)
      {
	return ID(me).node();
      }

    public:
      void *regbase;
    };

    struct PartialWriteKey {
      unsigned sender;
      unsigned sequence_id;
      bool operator<(const PartialWriteKey& rhs) const
      {
	if(sender < rhs.sender) return true;
	if(sender > rhs.sender) return false;
	return sequence_id < rhs.sequence_id;
      }
    };

    struct PartialWriteEntry {
      Event event;
      int remaining_count;
    };

    typedef std::map<PartialWriteKey, PartialWriteEntry> PartialWriteMap;
    static PartialWriteMap partial_remote_writes;
    static GASNetHSL partial_remote_writes_lock;

    struct RemoteWriteArgs : public BaseMedium {
      Memory mem;
      off_t offset;
      unsigned sender;
      unsigned sequence_id;
      Event event;
    };

    void handle_remote_write(RemoteWriteArgs args,
			     const void *data, size_t datalen)
    {
      MemoryImpl *impl = get_runtime()->get_memory_impl(args.mem);

      log_copy.debug() << "received remote write request: mem=" << args.mem
		       << ", offset=" << args.offset << ", size=" << datalen
		       << ", seq=" << args.sender << '/' << args.sequence_id
		       << ", event=" << args.event;
#ifdef DEBUG_REMOTE_WRITES
      printf("received remote write request: mem=" IDFMT ", offset=%zd, size=%zd, seq=%d/%d, event=" IDFMT "/%d\n",
		     args.mem.id, args.offset, datalen,
	             args.sender, args.sequence_id, args.event.id, args.event.gen);
      printf("  data[%p]: %08x %08x %08x %08x %08x %08x %08x %08x\n",
             data,
             ((unsigned *)(data))[0], ((unsigned *)(data))[1],
             ((unsigned *)(data))[2], ((unsigned *)(data))[3],
             ((unsigned *)(data))[4], ((unsigned *)(data))[5],
             ((unsigned *)(data))[6], ((unsigned *)(data))[7]);
#endif

      switch(impl->kind) {
      case MemoryImpl::MKIND_SYSMEM:
	{
	  LocalCPUMemory *cpumem = (LocalCPUMemory *)impl;
	  if(cpumem->registered) {
	    if(data == (cpumem->base + args.offset)) {
	      // copy is in right spot - yay!
	    } else {
	      printf("%d: received remote write to registered memory in wrong spot: %p != %p+%zd = %p\n",
		     gasnet_mynode(), data, cpumem->base, args.offset, cpumem->base + args.offset);
	      impl->put_bytes(args.offset, data, datalen);
	    }
	  } else {
	    impl->put_bytes(args.offset, data, datalen);
          }
	    
	  if(args.event.exists())
	    get_runtime()->get_genevent_impl(args.event)->trigger(args.event.gen,
									   gasnet_mynode());
	  break;
	}

      case MemoryImpl::MKIND_ZEROCOPY:
#ifdef USE_CUDA
      case MemoryImpl::MKIND_GPUFB:
#endif
	{
	  impl->put_bytes(args.offset, data, datalen);

	  if(args.event.exists())
	    get_runtime()->get_genevent_impl(args.event)->trigger(args.event.gen,
									   gasnet_mynode());
	  break;
	}

      default:
	assert(0);
      }

      // track the sequence ID to know when the full RDMA is done
      if(args.sequence_id > 0) {
        PartialWriteKey key;
        key.sender = args.sender;
        key.sequence_id = args.sequence_id;
	partial_remote_writes_lock.lock();
	PartialWriteMap::iterator it = partial_remote_writes.find(key);
	if(it == partial_remote_writes.end()) {
	  // first reference to this one
	  PartialWriteEntry entry;
	  entry.event = Event::NO_EVENT;
          entry.remaining_count = -1;
	  partial_remote_writes[key] = entry;
#ifdef DEBUG_PWT
	  printf("PWT: %d: new entry for %d/%d: " IDFMT "/%d, %d\n",
		 gasnet_mynode(), key.sender, key.sequence_id,
		 entry.event.id, entry.event.gen, entry.remaining_count);
#endif
	} else {
	  // have an existing entry (either another write or the fence)
	  PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	  printf("PWT: %d: have entry for %d/%d: " IDFMT "/%d, %d -> %d\n",
		 gasnet_mynode(), key.sender, key.sequence_id,
		 entry.event.id, entry.event.gen, 
		 entry.remaining_count, entry.remaining_count - 1);
#endif
	  entry.remaining_count--;
	  if(entry.remaining_count == 0) {
	    // we're the last write, and we've already got the fence, so 
	    //  trigger
	    Event e = entry.event;
	    partial_remote_writes.erase(it);
	    partial_remote_writes_lock.unlock();
	    if(e.exists())
	      get_runtime()->get_genevent_impl(e)->trigger(e.gen, gasnet_mynode());
	    return;
	  }
	}
	partial_remote_writes_lock.unlock();
      }
    }

    typedef ActiveMessageMediumNoReply<REMOTE_WRITE_MSGID,
				       RemoteWriteArgs,
				       handle_remote_write> RemoteWriteMessage;

    struct RemoteReduceArgs : public BaseMedium {
      Memory mem;
      off_t offset;
      int stride;
      ReductionOpID redop_id;
      //bool red_fold;
      unsigned sender;
      unsigned sequence_id;
      Event event;
    };

    void handle_remote_reduce(RemoteReduceArgs args,
			     const void *data, size_t datalen)
    {
      ReductionOpID redop_id;
      bool red_fold;
      if(args.redop_id > 0) {
	redop_id = args.redop_id;
	red_fold = false;
      } else if(args.redop_id < 0) {
	redop_id = -args.redop_id;
	red_fold = true;
      } else {
	assert(args.redop_id != 0);
	return;
      }

      log_copy.debug("received remote reduce request: mem=" IDFMT ", offset=%zd+%d, size=%zd, redop=%d(%s), seq=%d/%d, event=" IDFMT "/%d",
		     args.mem.id, args.offset, args.stride, datalen,
		     redop_id, (red_fold ? "fold" : "apply"),
		     args.sender, args.sequence_id,
		     args.event.id, args.event.gen);

      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];

      size_t count = datalen / redop->sizeof_rhs;

      void *lhs = get_runtime()->get_memory_impl(args.mem)->get_direct_ptr(args.offset, args.stride * count);
      assert(lhs);

      if(red_fold)
	redop->fold_strided(lhs, data,
			    args.stride, redop->sizeof_rhs, count, false /*not exclusive*/);
      else
	redop->apply_strided(lhs, data, 
			     args.stride, redop->sizeof_rhs, count, false /*not exclusive*/);

      // track the sequence ID to know when the full RDMA is done
      if(args.sequence_id > 0) {
        PartialWriteKey key;
        key.sender = args.sender;
        key.sequence_id = args.sequence_id;
	partial_remote_writes_lock.lock();
	PartialWriteMap::iterator it = partial_remote_writes.find(key);
	if(it == partial_remote_writes.end()) {
	  // first reference to this one
	  PartialWriteEntry entry;
	  entry.event = Event::NO_EVENT;
          entry.remaining_count = -1;
	  partial_remote_writes[key] = entry;
#ifdef DEBUG_PWT
	  printf("PWT: %d: new entry for %d/%d: " IDFMT "/%d, %d\n",
		 gasnet_mynode(), key.sender, key.sequence_id,
		 entry.event.id, entry.event.gen, entry.remaining_count);
#endif
	} else {
	  // have an existing entry (either another write or the fence)
	  PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	  printf("PWT: %d: have entry for %d/%d: " IDFMT "/%d, %d -> %d\n",
		 gasnet_mynode(), key.sender, key.sequence_id,
		 entry.event.id, entry.event.gen, 
		 entry.remaining_count, entry.remaining_count - 1);
#endif
	  entry.remaining_count--;
	  if(entry.remaining_count == 0) {
	    // we're the last write, and we've already got the fence, so 
	    //  trigger
	    Event e = entry.event;
	    partial_remote_writes.erase(it);
	    partial_remote_writes_lock.unlock();
	    if(e.exists())
	      get_runtime()->get_genevent_impl(e)->trigger(e.gen, gasnet_mynode());
	    return;
	  }
	}
	partial_remote_writes_lock.unlock();
      }
    }

    typedef ActiveMessageMediumNoReply<REMOTE_REDUCE_MSGID,
				       RemoteReduceArgs,
				       handle_remote_reduce> RemoteReduceMessage;

    struct RemoteWriteFenceArgs {
      Memory mem;
      unsigned sender;
      unsigned sequence_id;
      unsigned num_writes;
      Event event;
    };

    void handle_remote_write_fence(RemoteWriteFenceArgs args)
    {
      log_copy.debug("remote write fence (mem = " IDFMT ", seq = %d/%d, count = %d, event = " IDFMT "/%d)\n",
		     args.mem.id, args.sender, args.sequence_id, args.num_writes, args.event.id, args.event.gen);

      assert(args.sequence_id != 0);
      // track the sequence ID to know when the full RDMA is done
      if(args.sequence_id > 0) {
        PartialWriteKey key;
        key.sender = args.sender;
        key.sequence_id = args.sequence_id;
	partial_remote_writes_lock.lock();
	PartialWriteMap::iterator it = partial_remote_writes.find(key);
	if(it == partial_remote_writes.end()) {
	  // first reference to this one
	  PartialWriteEntry entry;
	  entry.event = args.event;
          entry.remaining_count = args.num_writes;
	  partial_remote_writes[key] = entry;
#ifdef DEBUG_PWT
	  printf("PWT: %d: new entry for %d/%d: " IDFMT "/%d, %d\n",
		 gasnet_mynode(), key.sender, key.sequence_id,
		 entry.event.id, entry.event.gen, entry.remaining_count);
#endif
	} else {
	  // have an existing entry (previous writes)
	  PartialWriteEntry& entry = it->second;
#ifdef DEBUG_PWT
	  printf("PWT: %d: have entry for %d/%d: " IDFMT "/%d -> " IDFMT "/%d, %d -> %d\n",
		 gasnet_mynode(), key.sender, key.sequence_id,
		 entry.event.id, entry.event.gen, 
		 args.event.id, args.event.gen,
		 entry.remaining_count, entry.remaining_count + args.num_writes);
#endif
	  entry.event = args.event;
	  entry.remaining_count += args.num_writes;
	  // a negative remaining count means we got too many writes!
	  assert(entry.remaining_count >= 0);
	  if(entry.remaining_count == 0) {
	    // this fence came after all the writes, so trigger
	    Event e = entry.event;
	    partial_remote_writes.erase(it);
	    partial_remote_writes_lock.unlock();
	    if(e.exists())
	      get_runtime()->get_genevent_impl(e)->trigger(e.gen, gasnet_mynode());
	    return;
	  }
	}
	partial_remote_writes_lock.unlock();
      }
    }

    typedef ActiveMessageShortNoReply<REMOTE_WRITE_FENCE_MSGID,
				      RemoteWriteFenceArgs,
				      handle_remote_write_fence> RemoteWriteFenceMessage;

    unsigned do_remote_write(Memory mem, off_t offset,
			     const void *data, size_t datalen,
			     unsigned sequence_id, Event event, bool make_copy = false)
    {
      log_copy.debug("sending remote write request: mem=" IDFMT ", offset=%zd, size=%zd, event=" IDFMT "/%d",
		     mem.id, offset, datalen,
		     event.id, event.gen);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(mem);
      char *dstptr;
      if(m_impl->kind == MemoryImpl::MKIND_RDMA) {
	dstptr = ((char *)(((RemoteMemory *)m_impl)->regbase)) + offset;
	//printf("remote mem write to rdma'able memory: dstptr = %p\n", dstptr);
      } else
	dstptr = 0;
      assert(datalen > 0);

      // if we don't have a destination pointer, we need to use the LMB, which
      //  may require chopping this request into pieces
      if(!dstptr) {
	size_t max_xfer_size = get_lmb_size(ID(mem).node());

	if(datalen > max_xfer_size) {
	  // can't do this if we've been given a trigger event - no guarantee
	  //  on ordering of these xfers
	  assert(!event.exists());

	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;
	  RemoteWriteArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.event = Event::NO_EVENT;
	  args.sender = gasnet_mynode();
	  args.sequence_id = sequence_id;

	  int count = 1;
	  while(datalen > max_xfer_size) {
	    RemoteWriteMessage::request(ID(mem).node(), args,
					pos, max_xfer_size,
					make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += max_xfer_size;
	    pos += max_xfer_size;
	    datalen -= max_xfer_size;
	    count++;
	  }

	  // last send includes whatever's left
	  RemoteWriteMessage::request(ID(mem).node(), args,
				      pos, datalen, 
				      make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	RemoteWriteArgs args;
	args.mem = mem;
	args.offset = offset;
	args.event = event;
        args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;
	RemoteWriteMessage::request(ID(mem).node(), args,
				    data, datalen,
				    (make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP),
				    dstptr);
	return 1;
      }
    }

    unsigned do_remote_write(Memory mem, off_t offset,
			     const void *data, size_t datalen,
			     off_t stride, size_t lines,
			     unsigned sequence_id, Event event, bool make_copy = false)
    {
      log_copy.debug("sending remote write request: mem=" IDFMT ", offset=%zd, size=%zdx%zd, event=" IDFMT "/%d",
		     mem.id, offset, datalen, lines,
		     event.id, event.gen);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(mem);
      char *dstptr;
      if(m_impl->kind == MemoryImpl::MKIND_RDMA) {
	dstptr = ((char *)(((RemoteMemory *)m_impl)->regbase)) + offset;
	//printf("remote mem write to rdma'able memory: dstptr = %p\n", dstptr);
      } else
	dstptr = 0;

      // if we don't have a destination pointer, we need to use the LMB, which
      //  may require chopping this request into pieces
      if(!dstptr) {
	size_t max_xfer_size = get_lmb_size(ID(mem).node());
	size_t max_lines_per_xfer = max_xfer_size / datalen;
	assert(max_lines_per_xfer > 0);

	if(lines > max_lines_per_xfer) {
	  // can't do this if we've been given a trigger event - no guarantee
	  //  on ordering of these xfers
	  assert(!event.exists());

	  log_copy.info("breaking large send into pieces");
	  const char *pos = (const char *)data;
	  RemoteWriteArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.event = Event::NO_EVENT;
	  args.sender = gasnet_mynode();
	  args.sequence_id = sequence_id;

	  int count = 1;
	  while(datalen > max_xfer_size) {
	    RemoteWriteMessage::request(ID(mem).node(), args,
					pos, datalen,
					stride, max_lines_per_xfer,
					make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += datalen * max_lines_per_xfer;
	    pos += stride * max_lines_per_xfer;
	    lines -= max_lines_per_xfer;
	    count++;
	  }

	  // last send includes whatever's left
	  RemoteWriteMessage::request(ID(mem).node(), args,
				      pos, datalen, stride, lines,
				      make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	RemoteWriteArgs args;
	args.mem = mem;
	args.offset = offset;
	args.event = event;
        args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;

	RemoteWriteMessage::request(ID(mem).node(), args,
				    data, datalen, stride, lines,
				    make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP,
				    dstptr);

	return 1;
      }
    }

    unsigned do_remote_write(Memory mem, off_t offset,
			     const SpanList &spans, size_t datalen,
			     unsigned sequence_id, Event event, bool make_copy = false)
    {
      log_copy.debug("sending remote write request: mem=" IDFMT ", offset=%zd, size=%zd(%zd spans), event=" IDFMT "/%d",
		     mem.id, offset, datalen, spans.size(),
		     event.id, event.gen);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(mem);
      char *dstptr;
      if(m_impl->kind == MemoryImpl::MKIND_RDMA) {
	dstptr = ((char *)(((RemoteMemory *)m_impl)->regbase)) + offset;
	//printf("remote mem write to rdma'able memory: dstptr = %p\n", dstptr);
      } else
	dstptr = 0;

      // if we don't have a destination pointer, we need to use the LMB, which
      //  may require chopping this request into pieces
      if(!dstptr) {
	size_t max_xfer_size = get_lmb_size(ID(mem).node());

	if(datalen > max_xfer_size) {
	  // can't do this if we've been given a trigger event - no guarantee
	  //  on ordering of these xfers
	  assert(!event.exists());

	  log_copy.info("breaking large send into pieces");
	  RemoteWriteArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.event = Event::NO_EVENT;
	  args.sender = gasnet_mynode();
	  args.sequence_id = sequence_id;

	  int count = 0;
	  // this is trickier because we don't actually know how much will fit
	  //  in each transfer
	  SpanList::const_iterator it = spans.begin();
	  while(datalen > 0) {
	    // possible special case - if the first span is too big to fit at
	    //   all, chop it up and send it
	    assert(it != spans.end());
	    if(it->second > max_xfer_size) {
              const char *pos = (const char *)(it->first);
              size_t left = it->second;
              while(left > max_xfer_size) {
		RemoteWriteMessage::request(ID(mem).node(), args,
					    pos, max_xfer_size,
					    make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
		args.offset += max_xfer_size;
		pos += max_xfer_size;
		left -= max_xfer_size;
		count++;
	      }
	      RemoteWriteMessage::request(ID(mem).node(), args,
					  pos, left,
					  make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	      args.offset += left;
	      count++;
	      datalen -= it->second;
	      it++;
	      continue;
	    }

	    // take spans in order until we run out of space or spans
	    SpanList subspans;
	    size_t xfer_size = 0;
	    while(it != spans.end()) {
	      // can we fit the next one?
	      if((xfer_size + it->second) > max_xfer_size) break;

	      subspans.push_back(*it);
	      xfer_size += it->second;
	      it++;
	    }
	    // if we didn't get at least one span, we won't make forward progress
	    assert(!subspans.empty());

	    RemoteWriteMessage::request(ID(mem).node(), args,
					subspans, xfer_size,
					make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += xfer_size;
	    datalen -= xfer_size;
	    count++;
	  }

	  return count;
	}
      }

      // we get here with either a valid destination pointer (so no size limit)
      //  or a write smaller than the LMB
      {
	RemoteWriteArgs args;
	args.mem = mem;
	args.offset = offset;
	args.event = event;
        args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;

	RemoteWriteMessage::request(ID(mem).node(), args,
				    spans, datalen,
				    make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP,
				    dstptr);
	
	return 1;
      }
    }

    unsigned do_remote_reduce(Memory mem, off_t offset,
			      ReductionOpID redop_id, bool red_fold,
			      const void *data, size_t count,
			      off_t src_stride, off_t dst_stride,
			      unsigned sequence_id,
			      Event event, bool make_copy = false)
    {
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[redop_id];
      size_t rhs_size = redop->sizeof_rhs;

      log_copy.debug("sending remote reduction request: mem=" IDFMT ", offset=%zd+%zd, size=%zdx%zd, redop=%d(%s), event=" IDFMT "/%d",
		     mem.id, offset, dst_stride, rhs_size, count,
		     redop_id, (red_fold ? "fold" : "apply"),
		     event.id, event.gen);

      // reductions always have to bounce off an intermediate buffer, so are subject to
      //  LMB limits
      {
	size_t max_xfer_size = get_lmb_size(ID(mem).node());
	size_t max_elmts_per_xfer = max_xfer_size / rhs_size;
	assert(max_elmts_per_xfer > 0);

	if(count > max_elmts_per_xfer) {
	  // can't do this if we've been given a trigger event - no guarantee
	  //  on ordering of these xfers
	  assert(!event.exists());

	  log_copy.info("breaking large reduction into pieces");
	  const char *pos = (const char *)data;
	  RemoteReduceArgs args;
	  args.mem = mem;
	  args.offset = offset;
	  args.stride = dst_stride;
	  assert(((off_t)(args.stride)) == dst_stride); // did it fit?
	  // fold encoded as a negation of the redop_id
	  args.redop_id = red_fold ? -redop_id : redop_id;
	  //args.red_fold = red_fold;
	  args.event = Event::NO_EVENT;
	  args.sender = gasnet_mynode();
	  args.sequence_id = sequence_id;

	  int xfers = 1;
	  while(count > max_elmts_per_xfer) {
	    RemoteReduceMessage::request(ID(mem).node(), args,
					 pos, rhs_size,
					 src_stride, max_elmts_per_xfer,
					 make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	    args.offset += dst_stride * max_elmts_per_xfer;
	    pos += src_stride * max_elmts_per_xfer;
	    count -= max_elmts_per_xfer;
	    xfers++;
	  }

	  // last send includes whatever's left
	  RemoteReduceMessage::request(ID(mem).node(), args,
				       pos, rhs_size, src_stride, count,
				       make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);
	  return xfers;
	}
      }

      // we get here with a write smaller than the LMB
      {
	RemoteReduceArgs args;
	args.mem = mem;
	args.offset = offset;
	args.stride = dst_stride;
	assert(((off_t)(args.stride)) == dst_stride); // did it fit?
	// fold encoded as a negation of the redop_id
	args.redop_id = red_fold ? -redop_id : redop_id;
	//args.red_fold = red_fold;
	args.event = event;
	args.sender = gasnet_mynode();
	args.sequence_id = sequence_id;

	RemoteReduceMessage::request(ID(mem).node(), args,
				     data, rhs_size, src_stride, count,
				     make_copy ? PAYLOAD_COPY : PAYLOAD_KEEP);

	return 1;
      }
    }

    void do_remote_fence(Memory mem, unsigned sequence_id, unsigned num_writes, Event event)
    {
      RemoteWriteFenceArgs args;
      args.mem = mem;
      args.sender = gasnet_mynode();
      args.sequence_id = sequence_id;
      args.num_writes = num_writes;
      args.event = event;

      // technically we could handle a num_writes == 0 case, but since it's
      //  probably indicative of badness elsewhere, barf on it for now
      assert(num_writes > 0);

      RemoteWriteFenceMessage::request(ID(mem).node(), args);
    }

    void RemoteMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // can't read/write a remote memory
#define ALLOW_REMOTE_MEMORY_WRITES
#ifdef ALLOW_REMOTE_MEMORY_WRITES
      do_remote_write(me, offset, src, size, 0, Event::NO_EVENT, true /* make copy! */);
#else
      assert(0);
#endif
    }

    GASNetMemory::GASNetMemory(Memory _me, size_t size_per_node)
      : MemoryImpl(_me, 0 /* we'll calculate it below */, MKIND_GLOBAL,
		     MEMORY_STRIDE, Memory::GLOBAL_MEM)
    {
      num_nodes = gasnet_nodes();
      seginfos = new gasnet_seginfo_t[num_nodes];
      CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );
      
      for(int i = 0; i < num_nodes; i++) {
	assert(seginfos[i].size >= size_per_node);
      }

      size = size_per_node * num_nodes;
      memory_stride = MEMORY_STRIDE;
      
      free_blocks[0] = size;
    }

    GASNetMemory::~GASNetMemory(void)
    {
    }

    RegionInstance GASNetMemory::create_instance(IndexSpace r,
						 const int *linearization_bits,
						 size_t bytes_needed,
						 size_t block_size,
						 size_t element_size,
						 const std::vector<size_t>& field_sizes,
						 ReductionOpID redopid,
						 off_t list_size,
                                                 const Realm::ProfilingRequestSet &reqs,
						 RegionInstance parent_inst)
    {
      if(gasnet_mynode() == 0) {
	return create_instance_local(r, linearization_bits, bytes_needed,
				     block_size, element_size, field_sizes, redopid,
				     list_size, reqs, parent_inst);
      } else {
	return create_instance_remote(r, linearization_bits, bytes_needed,
				      block_size, element_size, field_sizes, redopid,
				      list_size, reqs, parent_inst);
      }
    }

    void GASNetMemory::destroy_instance(RegionInstance i, 
					bool local_destroy)
    {
      if(gasnet_mynode() == 0) {
	destroy_instance_local(i, local_destroy);
      } else {
	destroy_instance_remote(i, local_destroy);
      }
    }

    off_t GASNetMemory::alloc_bytes(size_t size)
    {
      if(gasnet_mynode() == 0) {
	return alloc_bytes_local(size);
      } else {
	return alloc_bytes_remote(size);
      }
    }

    void GASNetMemory::free_bytes(off_t offset, size_t size)
    {
      if(gasnet_mynode() == 0) {
	free_bytes_local(offset, size);
      } else {
	free_bytes_remote(offset, size);
      }
    }

    void GASNetMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      char *dst_c = (char *)dst;
      while(size > 0) {
	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;
	gasnet_get(dst_c, node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, chunk_size);
	offset += chunk_size;
	dst_c += chunk_size;
	size -= chunk_size;
      }
    }

    void GASNetMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      char *src_c = (char *)src; // dropping const on purpose...
      while(size > 0) {
	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;
	gasnet_put(node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, src_c, chunk_size);
	offset += chunk_size;
	src_c += chunk_size;
	size -= chunk_size;
      }
    }

    void GASNetMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
					    size_t count, const void *entry_buffer)
    {
      const char *entry = (const char *)entry_buffer;
      unsigned ptr;

      for(size_t i = 0; i < count; i++)
      {
	redop->get_list_pointers(&ptr, entry, 1);
	//printf("ptr[%d] = %d\n", i, ptr);
	off_t elem_offset = offset + ptr * redop->sizeof_lhs;
	off_t blkid = (elem_offset / memory_stride / num_nodes);
	off_t node = (elem_offset / memory_stride) % num_nodes;
	off_t blkoffset = elem_offset % memory_stride;
	assert(node == gasnet_mynode());
	char *tgt_ptr = ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset;
	redop->apply_list_entry(tgt_ptr, entry, 1, ptr);
	entry += redop->sizeof_list_entry;
      }
    }

    void *GASNetMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0;  // can't give a pointer to the caller - have to use RDMA
    }

    int GASNetMemory::get_home_node(off_t offset, size_t size)
    {
      off_t start_blk = offset / memory_stride;
      off_t end_blk = (offset + size - 1) / memory_stride;
      if(start_blk != end_blk) return -1;

      return start_blk % num_nodes;
    }

    void GASNetMemory::get_batch(size_t batch_size,
				 const off_t *offsets, void * const *dsts, 
				 const size_t *sizes)
    {
#define NO_USE_NBI_ACCESSREGION
#ifdef USE_NBI_ACCESSREGION
      gasnet_begin_nbi_accessregion();
#endif
      DetailedTimer::push_timer(10);
      for(size_t i = 0; i < batch_size; i++) {
	off_t offset = offsets[i];
	char *dst_c = (char *)(dsts[i]);
	size_t size = sizes[i];

	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;

	while(size > 0) {
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;

	  char *src_c = (((char *)seginfos[node].addr) +
			 (blkid * memory_stride) + blkoffset);
	  if(node == gasnet_mynode()) {
	    memcpy(dst_c, src_c, chunk_size);
	  } else {
	    gasnet_get_nbi(dst_c, node, src_c, chunk_size);
	  }

	  dst_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

#ifdef USE_NBI_ACCESSREGION
      DetailedTimer::push_timer(11);
      gasnet_handle_t handle = gasnet_end_nbi_accessregion();
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(12);
      gasnet_wait_syncnb(handle);
      DetailedTimer::pop_timer();
#else
      DetailedTimer::push_timer(13);
      gasnet_wait_syncnbi_gets();
      DetailedTimer::pop_timer();
#endif
    }

    void GASNetMemory::put_batch(size_t batch_size,
				 const off_t *offsets,
				 const void * const *srcs, 
				 const size_t *sizes)
    {
      gasnet_begin_nbi_accessregion();

      DetailedTimer::push_timer(14);
      for(size_t i = 0; i < batch_size; i++) {
	off_t offset = offsets[i];
	const char *src_c = (char *)(srcs[i]);
	size_t size = sizes[i];

	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;

	while(size > 0) {
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;

	  char *dst_c = (((char *)seginfos[node].addr) +
			 (blkid * memory_stride) + blkoffset);
	  if(node == gasnet_mynode()) {
	    memcpy(dst_c, src_c, chunk_size);
	  } else {
	    gasnet_put_nbi(node, dst_c, (void *)src_c, chunk_size);
	  }

	  src_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(15);
      gasnet_handle_t handle = gasnet_end_nbi_accessregion();
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(16);
      gasnet_wait_syncnb(handle);
      DetailedTimer::pop_timer();
    }

    RegionInstance MemoryImpl::create_instance_local(IndexSpace r,
						       const int *linearization_bits,
						       size_t bytes_needed,
						       size_t block_size,
						       size_t element_size,
						       const std::vector<size_t>& field_sizes,
						       ReductionOpID redopid,
						       off_t list_size,
                                                       const Realm::ProfilingRequestSet &reqs,
						       RegionInstance parent_inst)
    {
      off_t inst_offset = alloc_bytes(bytes_needed);
      if(inst_offset < 0) {
        return RegionInstance::NO_INST;
      }

      off_t count_offset = -1;
      if(list_size > 0) {
	count_offset = alloc_bytes(sizeof(size_t));
	if(count_offset < 0) {
	  return RegionInstance::NO_INST;
	}

	size_t zero = 0;
	put_bytes(count_offset, &zero, sizeof(zero));
      }

      // SJT: think about this more to see if there are any race conditions
      //  with an allocator temporarily having the wrong ID
      RegionInstance i = ID(ID::ID_INSTANCE, 
			    ID(me).node(),
			    ID(me).index_h(),
			    0).convert<RegionInstance>();


      //RegionMetaDataImpl *r_impl = get_runtime()->get_metadata_impl(r);
      DomainLinearization linear;
      linear.deserialize(linearization_bits);

      RegionInstanceImpl *i_impl = new RegionInstanceImpl(i, r, me, inst_offset, 
                                                              bytes_needed, redopid,
							      linear, block_size, 
                                                              element_size, field_sizes, reqs,
							      count_offset, list_size, 
                                                              parent_inst);

      // find/make an available index to store this in
      IDType index;
      {
	AutoHSLLock al(mutex);

	size_t size = instances.size();
	for(index = 0; index < size; index++)
	  if(!instances[index]) {
	    i.id |= index;
	    i_impl->me = i;
	    instances[index] = i_impl;
	    break;
	  }
        assert(index < (((IDType)1) << ID::INDEX_L_BITS));

	i.id |= index;
	i_impl->me = i;
	i_impl->lock.me = ID(i.id).convert<Reservation>(); // have to change the lock's ID too!
	if(index >= size) instances.push_back(i_impl);
      }

      log_inst.info("local instance " IDFMT " created in memory " IDFMT " at offset %zd (redop=%d list_size=%zd parent_inst=" IDFMT " block_size=%zd)",
		    i.id, me.id, inst_offset, redopid, list_size,
                    parent_inst.id, block_size);

      return i;
    }

    struct CreateInstanceReqArgs : public BaseMedium {
      Memory m;
      IndexSpace r;
      RegionInstance parent_inst;
      int sender;
      void *resp_ptr;
    };

    struct CreateInstancePayload {
      size_t bytes_needed;
      size_t block_size;
      size_t element_size;
      //off_t adjust;
      off_t list_size;
      ReductionOpID redopid;
      int linearization_bits[RegionInstanceImpl::MAX_LINEARIZATION_LEN];
      size_t num_fields; // as long as it needs to be
      const size_t &field_size(int idx) const { return *((&num_fields)+idx+1); }
      size_t &field_size(int idx) { return *((&num_fields)+idx+1); }
    };

    struct CreateInstanceRespArgs {
      void *resp_ptr;
      RegionInstance i;
      off_t inst_offset;
      off_t count_offset;
    };

    void handle_create_instance_req(CreateInstanceReqArgs args, 
				    const void *msgdata, size_t msglen);

    typedef ActiveMessageMediumNoReply<CREATE_INST_MSGID,
				       CreateInstanceReqArgs,
				       handle_create_instance_req> CreateInstanceRequest;

    void handle_create_instance_resp(CreateInstanceRespArgs args);

    typedef ActiveMessageShortNoReply<CREATE_INST_RPLID,
				      CreateInstanceRespArgs,
				      handle_create_instance_resp> CreateInstanceResponse;
				   
    void handle_create_instance_req(CreateInstanceReqArgs args, const void *msgdata, size_t msglen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      CreateInstanceRespArgs resp;

      resp.resp_ptr = args.resp_ptr;

      const CreateInstancePayload *payload = (const CreateInstancePayload *)msgdata;

      std::vector<size_t> field_sizes(payload->num_fields);
      for(size_t i = 0; i < payload->num_fields; i++)
	field_sizes[i] = payload->field_size(i);

      size_t req_offset = sizeof(CreateInstancePayload) + sizeof(size_t) * payload->num_fields;
      Realm::ProfilingRequestSet requests;
      requests.deserialize(((const char*)msgdata)+req_offset);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(args.m);
      resp.i = m_impl->create_instance(args.r, 
				       payload->linearization_bits,
				       payload->bytes_needed,
				       payload->block_size,
				       payload->element_size,
				       field_sizes,
				       payload->redopid,
				       payload->list_size,
				       requests,
				       args.parent_inst);

      //resp.inst_offset = resp.i.impl()->locked_data.offset; // TODO: Static
      // Its' actually only safe to do this if we got an instance
      if (resp.i.exists()) {
        RegionInstanceImpl *i_impl = get_runtime()->get_instance_impl(resp.i);

        resp.inst_offset = i_impl->metadata.alloc_offset;
        resp.count_offset = i_impl->metadata.count_offset;
      }

      CreateInstanceResponse::request(args.sender, resp);
    }

    void handle_create_instance_resp(CreateInstanceRespArgs args)
    {
      HandlerReplyFuture<CreateInstanceRespArgs> *f = static_cast<HandlerReplyFuture<CreateInstanceRespArgs> *>(args.resp_ptr);

      f->set(args);
    }

    RegionInstance MemoryImpl::create_instance_remote(IndexSpace r,
							const int *linearization_bits,
							size_t bytes_needed,
							size_t block_size,
							size_t element_size,
							const std::vector<size_t>& field_sizes,
							ReductionOpID redopid,
							off_t list_size,
                                                        const Realm::ProfilingRequestSet &reqs,
							RegionInstance parent_inst)
    {
      size_t req_offset = sizeof(CreateInstancePayload) + sizeof(size_t) * field_sizes.size();
      size_t payload_size = req_offset + reqs.compute_size();
      CreateInstancePayload *payload = (CreateInstancePayload *)malloc(payload_size);

      payload->bytes_needed = bytes_needed;
      payload->block_size = block_size;
      payload->element_size = element_size;
      //payload->adjust = ?
      payload->list_size = list_size;
      payload->redopid = redopid;

      for(unsigned i = 0; i < RegionInstanceImpl::MAX_LINEARIZATION_LEN; i++)
	payload->linearization_bits[i] = linearization_bits[i];

      payload->num_fields = field_sizes.size();
      for(unsigned i = 0; i < field_sizes.size(); i++)
	payload->field_size(i) = field_sizes[i];

      reqs.serialize(((char*)payload)+req_offset);

      CreateInstanceReqArgs args;
      args.srcptr = 0; // gcc 4.4.7 wants this!?
      args.m = me;
      args.r = r;
      args.parent_inst = parent_inst;
      log_inst.debug("creating remote instance: node=%d", ID(me).node());

      HandlerReplyFuture<CreateInstanceRespArgs> result;
      args.resp_ptr = &result;
      args.sender = gasnet_mynode();
      CreateInstanceRequest::request(ID(me).node(), args,
				     payload, payload_size, PAYLOAD_FREE);

      result.wait();
      CreateInstanceRespArgs resp = result.get();

      // Only do this if the response succeeds
      if (resp.i.exists()) {
        log_inst.debug("created remote instance: inst=" IDFMT " offset=%zd", resp.i.id, resp.inst_offset);

        DomainLinearization linear;
        linear.deserialize(linearization_bits);

        RegionInstanceImpl *i_impl = new RegionInstanceImpl(resp.i, r, me, resp.inst_offset, bytes_needed, redopid,
                                                                linear, block_size, element_size, field_sizes, reqs,
                                                                resp.count_offset, list_size, parent_inst);

        unsigned index = ID(resp.i).index_l();
        // resize array if needed
        if(index >= instances.size()) {
          AutoHSLLock a(mutex);
          if(index >= instances.size()) {
            log_inst.debug("resizing instance array: mem=" IDFMT " old=%zd new=%d",
                     me.id, instances.size(), index+1);
            for(unsigned i = instances.size(); i <= index; i++)
              instances.push_back(0);
          }
        }
        instances[index] = i_impl;
      }
      return resp.i;
    }

    RegionInstanceImpl *MemoryImpl::get_instance(RegionInstance i)
    {
      ID id(i);

      // have we heard of this one before?  if not, add it
      unsigned index = id.index_l();
      if(index >= instances.size()) { // lock not held - just for early out
	AutoHSLLock a(mutex);
	if(index >= instances.size()) // real check
	  instances.resize(index + 1);
      }

      if(!instances[index]) {
	//instances[index] = new RegionInstanceImpl(id.node());
	assert(0);
      }

      return instances[index];
    }

    void MemoryImpl::destroy_instance_local(RegionInstance i, 
					      bool local_destroy)
    {
      log_inst.info("destroying local instance: mem=" IDFMT " inst=" IDFMT "", me.id, i.id);

      // all we do for now is free the actual data storage
      unsigned index = ID(i).index_l();
      assert(index < instances.size());

      RegionInstanceImpl *iimpl = instances[index];

      free_bytes(iimpl->metadata.alloc_offset, iimpl->metadata.size);

      if(iimpl->metadata.count_offset >= 0)
	free_bytes(iimpl->metadata.count_offset, sizeof(size_t));

      // begin recovery of metadata
      if(iimpl->metadata.initiate_cleanup(i.id)) {
	// no remote copies exist, so we can reclaim instance immediately
	log_metadata.info("no remote copies of metadata for " IDFMT, i.id);
	// TODO
      }
      
      // handle any profiling requests
      iimpl->finalize_instance();
      
      return; // TODO: free up actual instance record?
      ID id(i);

      // TODO: actually free corresponding storage
    }

    struct DestroyInstanceArgs {
      Memory m;
      RegionInstance i;
    };

    void handle_destroy_instance(DestroyInstanceArgs args)
    {
      MemoryImpl *m_impl = get_runtime()->get_memory_impl(args.m);
      m_impl->destroy_instance(args.i, false);
    }

    typedef ActiveMessageShortNoReply<DESTROY_INST_MSGID,
				      DestroyInstanceArgs,
				      handle_destroy_instance> DestroyInstanceMessage;

    void MemoryImpl::destroy_instance_remote(RegionInstance i, 
					       bool local_destroy)
    {
      // if we're the original destroyer of the instance, tell the owner
      if(local_destroy) {
	int owner = ID(me).node();

	DestroyInstanceArgs args;
	args.m = me;
	args.i = i;
	log_inst.debug("destroying remote instance: node=%d inst=" IDFMT "", owner, i.id);

	DestroyInstanceMessage::request(owner, args);
      }

      // and right now, we leave the instance itself untouched
      return;
    }

    ///////////////////////////////////////////////////
    // Task

    Logger::Category log_task("task");
    Logger::Category log_util("util");

    Task::Task(Processor _proc, Processor::TaskFuncID _func_id,
	       const void *_args, size_t _arglen,
	       Event _finish_event, int _priority, int expected_count)
      : Realm::Operation(), proc(_proc), func_id(_func_id), arglen(_arglen),
	finish_event(_finish_event), priority(_priority),
        run_count(0), finish_count(expected_count), capture_proc(false)
    {
      if(arglen) {
	args = malloc(arglen);
	memcpy(args, _args, arglen);
      } else
	args = 0;
    }

    Task::Task(Processor _proc, Processor::TaskFuncID _func_id,
	       const void *_args, size_t _arglen,
               const Realm::ProfilingRequestSet &reqs,
	       Event _finish_event, int _priority, int expected_count)
      : Realm::Operation(reqs), proc(_proc), func_id(_func_id), arglen(_arglen),
	finish_event(_finish_event), priority(_priority),
        run_count(0), finish_count(expected_count)
    {
      if(arglen) {
	args = malloc(arglen);
	memcpy(args, _args, arglen);
      } else
	args = 0;
      capture_proc = measurements.wants_measurement<
                        Realm::ProfilingMeasurements::OperationProcessorUsage>();
    }

    Task::~Task(void)
    {
      free(args);
      if (capture_proc) {
        Realm::ProfilingMeasurements::OperationProcessorUsage usage;
        usage.proc = proc;
        measurements.add_measurement(usage);
      }
    }

  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

    ///////////////////////////////////////////////////
    // Processor

    /*static*/ const Processor Processor::NO_PROC = { 0 }; 

    /*static*/ Processor Processor::get_executing_processor(void) 
    { 
      void *tls_val = gasnett_threadkey_get(cur_preemptable_thread);
      if (tls_val != NULL)
      {
        PreemptableThread *me = (PreemptableThread *)tls_val;
        return me->get_processor();
      }
      // Otherwise this better be a GPU processor 
#ifdef USE_CUDA
      return GPUProcessor::get_processor();
#else
      assert(0);
#endif
    }

    Processor::Kind Processor::kind(void) const
    {
      return get_runtime()->get_processor_impl(*this)->kind;
    }

    /*static*/ Processor Processor::create_group(const std::vector<Processor>& members)
    {
      // are we creating a local group?
      if((members.size() == 0) || (ID(members[0]).node() == gasnet_mynode())) {
	ProcessorGroup *grp = get_runtime()->local_proc_group_free_list->alloc_entry();
	grp->set_group_members(members);
#ifdef EVENT_GRAPH_TRACE
        {
          const int base_size = 1024;
          char base_buffer[base_size];
          char *buffer;
          int buffer_size = (members.size() * 20);
          if (buffer_size >= base_size)
            buffer = (char*)malloc(buffer_size+1);
          else
            buffer = base_buffer;
          buffer[0] = '\0';
          int offset = 0;
          for (std::vector<Processor>::const_iterator it = members.begin();
                it != members.end(); it++)
          {
            int written = snprintf(buffer+offset,buffer_size-offset,
                                   " " IDFMT, it->id);
            assert(written < (buffer_size-offset));
            offset += written;
          }
          log_event_graph.info("Group: " IDFMT " %ld%s",
                                grp->me.id, members.size(), buffer); 
          if (buffer_size >= base_size)
            free(buffer);
        }
#endif
	return grp->me;
      }

      assert(0);
    }

    void Processor::get_group_members(std::vector<Processor>& members)
    {
      // if we're a plain old processor, the only member of our "group" is ourself
      if(ID(*this).type() == ID::ID_PROCESSOR) {
	members.push_back(*this);
	return;
      }

      assert(ID(*this).type() == ID::ID_PROCGROUP);

      ProcessorGroup *grp = get_runtime()->get_procgroup_impl(*this);
      grp->get_group_members(members);
    }

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
			   //std::set<RegionInstance> instances_needed,
			   Event wait_on, int priority) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ProcessorImpl *p = get_runtime()->get_processor_impl(*this);

      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      Event e = finish_event->current_event();
#ifdef EVENT_GRAPH_TRACE
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.info("Task Request: %d " IDFMT 
                            " (" IDFMT ",%d) (" IDFMT ",%d)"
                            " (" IDFMT ",%d) %d %p %ld",
                            func_id, id, e.id, e.gen,
                            wait_on.id, wait_on.gen,
                            enclosing.id, enclosing.gen,
                            priority, args, arglen);
#endif

      p->spawn_task(func_id, args, arglen, //instances_needed, 
		    wait_on, e, priority);
      return e;
    }

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
                           const Realm::ProfilingRequestSet &reqs,
			   Event wait_on, int priority) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ProcessorImpl *p = get_runtime()->get_processor_impl(*this);

      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      Event e = finish_event->current_event();
#ifdef EVENT_GRAPH_TRACE
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.info("Task Request: %d " IDFMT 
                            " (" IDFMT ",%d) (" IDFMT ",%d)"
                            " (" IDFMT ",%d) %d %p %ld",
                            func_id, id, e.id, e.gen,
                            wait_on.id, wait_on.gen,
                            enclosing.id, enclosing.gen,
                            priority, args, arglen);
#endif

      p->spawn_task(func_id, args, arglen, reqs,
		    wait_on, e, priority);
      return e;
    }

    AddressSpace Processor::address_space(void) const
    {
      return ID(id).node();
    }

    IDType Processor::local_id(void) const
    {
      return ID(id).index();
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    ProcessorImpl::ProcessorImpl(Processor _me, Processor::Kind _kind)
      : me(_me), kind(_kind), run_counter(0)
    {
    }

    ProcessorImpl::~ProcessorImpl(void)
    {
    }

    ProcessorGroup::ProcessorGroup(void)
      : ProcessorImpl(Processor::NO_PROC, Processor::PROC_GROUP),
	members_valid(false), members_requested(false), next_free(0)
    {
    }

    ProcessorGroup::~ProcessorGroup(void)
    {
    }

    void ProcessorGroup::init(Processor _me, int _owner)
    {
      assert(ID(_me).node() == (unsigned)_owner);

      me = _me;
      lock.init(ID(me).convert<Reservation>(), ID(me).node());
    }

    void ProcessorGroup::set_group_members(const std::vector<Processor>& member_list)
    {
      // can only be perform on owner node
      assert(ID(me).node() == gasnet_mynode());
      
      // can only be done once
      assert(!members_valid);

      for(std::vector<Processor>::const_iterator it = member_list.begin();
	  it != member_list.end();
	  it++) {
	ProcessorImpl *m_impl = get_runtime()->get_processor_impl(*it);
	members.push_back(m_impl);
      }

      members_requested = true;
      members_valid = true;
    }

    void ProcessorGroup::get_group_members(std::vector<Processor>& member_list)
    {
      assert(members_valid);

      for(std::vector<ProcessorImpl *>::const_iterator it = members.begin();
	  it != members.end();
	  it++)
	member_list.push_back((*it)->me);
    }

    void ProcessorGroup::start_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::shutdown_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::initialize_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::finalize_processor(void)
    {
      assert(0);
    }

    void ProcessorGroup::enqueue_task(Task *task)
    {
      for (std::vector<ProcessorImpl *>::const_iterator it = members.begin();
            it != members.end(); it++)
      {
        (*it)->enqueue_task(task);
      }
    }

    /*virtual*/ void ProcessorGroup::spawn_task(Processor::TaskFuncID func_id,
						const void *args, size_t arglen,
						//std::set<RegionInstance> instances_needed,
						Event start_event, Event finish_event,
						int priority)
    {
      // create a task object and insert it into the queue
      Task *task = new Task(me, func_id, args, arglen, 
                            finish_event, priority, members.size());

      if (start_event.has_triggered())
        enqueue_task(task);
      else
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }

    /*virtual*/ void ProcessorGroup::spawn_task(Processor::TaskFuncID func_id,
						const void *args, size_t arglen,
                                                const Realm::ProfilingRequestSet &reqs,
						Event start_event, Event finish_event,
						int priority)
    {
      // create a task object and insert it into the queue
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            finish_event, priority, members.size());

      if (start_event.has_triggered())
        enqueue_task(task);
      else
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }

    LocalThread::LocalThread(LocalProcessor *p)
      : PreemptableThread(), proc(p), state(RUNNING_STATE),
	thread_cond(thread_mutex),
        initialize(false), finalize(false)
    {
    }

    LocalThread::~LocalThread(void)
    {
    }

    Processor LocalThread::get_processor(void) const
    {
      return proc->me;
    }

    void LocalThread::thread_main(void)
    {
      if (initialize)
        proc->initialize_processor();
      while (true)
      {
        assert(state == RUNNING_STATE);
        bool quit = proc->execute_task(this);
        if (quit) break;
      }
      if (finalize)
        proc->finalize_processor();
    }

    void LocalThread::sleep_on_event(Event wait_for)
    {
#ifdef EVENT_GRAPH_TRACE
      unsigned long long start = TimeStamp::get_current_time_in_micros(); 
#endif
      assert(state == RUNNING_STATE);
      // First mark that we are this thread is now paused
      state = PAUSED_STATE;
      // Then tell the processor to pause the thread
      proc->pause_thread(this);
      // Now register ourselves with the event
      EventImpl::add_waiter(wait_for, this);
      // Take our lock and see if we are still in the paused state
      // It's possible that we've already been woken up so check before
      // going to sleep
      thread_mutex.lock();
      // If we are in the paused state or the resumable state then we actually
      // do need to go to sleep so we can be woken up by the processor later
      if ((state == PAUSED_STATE) || (state == RESUMABLE_STATE))
      {
	thread_cond.wait();
      }
      assert(state == RUNNING_STATE);
      thread_mutex.unlock();
#ifdef EVENT_GRAPH_TRACE
      unsigned long long stop = TimeStamp::get_current_time_in_micros();
      Event enclosing = find_enclosing_termination_event();
      log_event_graph.debug("Task Wait: (" IDFMT ",%d) (" IDFMT ",%d) %lld",
                            enclosing.id, enclosing.gen,
                            wait_for.id, wait_for.gen, (stop - start));
#endif
    }

    bool LocalThread::event_triggered(void)
    {
      thread_mutex.lock();
      assert(state == PAUSED_STATE);
      state = RESUMABLE_STATE;
      thread_mutex.unlock();
      // Now tell the processor that this thread is resumable
      proc->resume_thread(this);
      return false;
    }

    void LocalThread::print_info(FILE *f)
    {
      fprintf(f, "Waiting thread %lx\n", (unsigned long)thread); 
    }

    void LocalThread::awake(void)
    {
      thread_mutex.lock();
      assert((state == SLEEPING_STATE) || (state == SLEEP_STATE));
      // Only need to signal if the thread is actually asleep
      if (state == SLEEP_STATE)
	thread_cond.signal();
      state = RUNNING_STATE;
      thread_mutex.unlock();
    }

    void LocalThread::sleep(void)
    {
      thread_mutex.lock();
      assert((state == SLEEPING_STATE) || (state == RUNNING_STATE));
      // If we haven't been told to stay awake, then go to sleep
      if (state == SLEEPING_STATE) {
        state = SLEEP_STATE;
	thread_cond.wait();
      }
      assert(state == RUNNING_STATE);
      thread_mutex.unlock();
    }

    void LocalThread::prepare_to_sleep(void)
    {
      // Don't need the lock since we are running
      assert(state == RUNNING_STATE);
      state = SLEEPING_STATE;
    }

    void LocalThread::resume(void)
    {
      thread_mutex.lock();
      assert(state == RESUMABLE_STATE);
      state = RUNNING_STATE;
      thread_cond.signal();
      thread_mutex.unlock();
    }

    void LocalThread::shutdown(void)
    {
      // wake up the thread
      thread_mutex.lock();
      state = RUNNING_STATE;
      thread_cond.signal();
      thread_mutex.unlock();
      // Now wait to join with the thread
      void *result;
      pthread_join(thread, &result);
    }

    LocalProcessor::LocalProcessor(Processor _me, Processor::Kind _kind, 
                                   size_t stacksize, const char *name, int _core)
      : ProcessorImpl(_me, _kind), core_id(_core), 
        stack_size(stacksize), processor_name(name),
	condvar(mutex), done_initialization(false),
        shutdown(false), shutdown_trigger(false), running_thread(0)
    {
    }

    LocalProcessor::~LocalProcessor(void)
    {
    }

    void LocalProcessor::start_processor(void)
    {
      assert(running_thread == 0);
      running_thread = create_new_thread();
      running_thread->do_initialize();
      running_thread->start_thread(stack_size, core_id, processor_name);
      mutex.lock();
      if (!done_initialization)
        condvar.wait();
      mutex.unlock();
    }

    void LocalProcessor::shutdown_processor(void)
    {
      // First check to make sure that we received the kill
      // pill. If we didn't then wait for it. This is how
      // we distinguish deadlock from just normal termination
      // from all the processors being idle
      std::vector<LocalThread*> to_shutdown;
      mutex.lock();
      if (!shutdown_trigger)
	condvar.wait();
      assert(shutdown_trigger);
      shutdown = true;
      to_shutdown = available_threads;
      if (running_thread)
        to_shutdown.push_back(running_thread);
      assert(resumable_threads.empty());
      assert(paused_threads.empty());
      mutex.unlock();
      //printf("Processor " IDFMT " needed %ld threads\n", 
      //        proc.id, to_shutdown.size());
      // We can now read this outside the lock since we know
      // that the threads are all asleep and are all about to exit
      assert(!to_shutdown.empty());
      for (unsigned idx = 0; idx < to_shutdown.size(); idx++)
      {
        if (idx == 0)
          to_shutdown[idx]->do_finalize();
        to_shutdown[idx]->shutdown();
        delete to_shutdown[idx];
      }
    }

    void LocalProcessor::initialize_processor(void)
    {
      mutex.lock();
      done_initialization = true;
      condvar.signal();
      mutex.unlock();
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_INIT);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor init task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor init task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor init task: proc=" IDFMT "", me.id);
      }
    }

    void LocalProcessor::finalize_processor(void)
    {
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor shutdown task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor shutdown task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor shutdown task: proc=" IDFMT "", me.id);
      }
    }

    LocalThread* LocalProcessor::create_new_thread(void)
    {
      return new LocalThread(this);
    }

    bool LocalProcessor::execute_task(LocalThread *thread)
    {
      mutex.lock();
      // Sanity check, we should be the running thread if we are in here
      assert(thread == running_thread);
      // First check to see if there are any resumable threads
      // If there are then we will switch onto those
      if (!resumable_threads.empty())
      {
        // Move this thread on to the available threads and wake
        // up one of the resumable threads
        thread->prepare_to_sleep();
        available_threads.push_back(thread);
        // Pull the first thread off the resumable threads
        LocalThread *to_resume = resumable_threads.front();
        resumable_threads.pop_front();
        // Make this the running thread
        running_thread = to_resume;
        // Release the lock
	mutex.unlock();
        // Wake up the resumable thread
        to_resume->resume();
        // Put ourselves to sleep
        thread->sleep();
      }
      else if (task_queue.empty())
      {
        // If there are no tasks to run, then we should go to sleep
        thread->prepare_to_sleep();
        available_threads.push_back(thread);
        running_thread = NULL;
	mutex.unlock();
        thread->sleep();
      }
      else
      {
        // Pull a task off the queue and execute it
        Task *task = task_queue.pop();
        if (task->func_id == 0) {
          // This is the kill pill so we need to handle it special
          finished();
          // Mark that we received the shutdown trigger
          shutdown_trigger = true;
	  condvar.signal();
	  mutex.unlock();
          // Trigger the completion task
          if (__sync_fetch_and_add(&(task->run_count),1) == 0)
            get_runtime()->get_genevent_impl(task->finish_event)->
                          trigger(task->finish_event.gen, gasnet_mynode());
          // Delete the task
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        } else {
	  mutex.unlock();
          // Common case: just run the task
          if (__sync_fetch_and_add(&task->run_count,1) == 0)
            thread->run_task(task, me);
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        }
      }
      // This value is monotonic so once it becomes true, then we should exit
      return shutdown;
    }

    void LocalProcessor::pause_thread(LocalThread *thread)
    {
      LocalThread *to_wake = 0;
      LocalThread *to_start = 0;
      LocalThread *to_resume = 0;
      mutex.lock();
      assert(running_thread == thread);
      // Put this on the list of paused threads
      paused_threads.insert(thread);
      // Now see if we have other work to do
      if (!resumable_threads.empty()) {
        to_resume = resumable_threads.front();
        resumable_threads.pop_front();
        running_thread = to_resume;
      } else if (!task_queue.empty()) {
        // Note we might need to make a new thread here
        if (!available_threads.empty()) {
          to_wake = available_threads.back();
          available_threads.pop_back();
          running_thread = to_wake;
        } else {
          // Make a new thread to run
          to_start = create_new_thread();
          running_thread = to_start;
        }
      } else {
        // Nothing else to do, so mark that no one is running
        running_thread = 0;
      }
      mutex.unlock();
      // Wake up any threads while not holding the lock
      if (to_wake)
        to_wake->awake();
      if (to_start)
        to_start->start_thread(stack_size, core_id, processor_name);
      if (to_resume)
        to_resume->resume();
    }

    void LocalProcessor::resume_thread(LocalThread *thread)
    {
      bool resume_now = false;
      mutex.lock();
      std::set<LocalThread*>::iterator finder = 
        paused_threads.find(thread);
      assert(finder != paused_threads.end());
      paused_threads.erase(finder);
      if (running_thread == NULL) {
        // No one else is running now, so resume the thread
        running_thread = thread;
        resume_now = true;
      } else {
        // Easy case, just add it to the list of resumable threads
        resumable_threads.push_back(thread);
      }
      mutex.unlock();
      if (resume_now)
        thread->resume();
    }

    void LocalProcessor::enqueue_task(Task *task)
    {
      // Mark this task as ready
      task->mark_ready();
      LocalThread *to_wake = 0;
      LocalThread *to_start = 0;
      mutex.lock();
      task_queue.insert(task, task->priority);
      // Figure out if we need to wake someone up
      if (running_thread == NULL) {
        if (!available_threads.empty()) {
          to_wake = available_threads.back();
          available_threads.pop_back();
          running_thread = to_wake;
        } else {
          to_start = create_new_thread(); 
          running_thread = to_start;
        }
      }
      mutex.unlock();
      if (to_wake)
        to_wake->awake();
      if (to_start)
        to_start->start_thread(stack_size, core_id, processor_name);
    }

    void LocalProcessor::spawn_task(Processor::TaskFuncID func_id,
                                    const void *args, size_t arglen,
                                    Event start_event, Event finish_event,
                                    int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, finish_event, 
                            priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    void LocalProcessor::spawn_task(Processor::TaskFuncID func_id,
                                    const void *args, size_t arglen,
                                    const Realm::ProfilingRequestSet &reqs,
                                    Event start_event, Event finish_event,
                                    int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            finish_event, priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    bool DeferredTaskSpawn::event_triggered(void)
    {
      log_task.debug("deferred task now ready: func=%d finish=" IDFMT "/%d",
                 task->func_id, 
                 task->finish_event.id, task->finish_event.gen);
      proc->enqueue_task(task);
      return true;
    }

    void DeferredTaskSpawn::print_info(FILE *f)
    {
      fprintf(f,"deferred task: func=%d proc=" IDFMT " finish=" IDFMT "/%d\n",
             task->func_id, task->proc.id, task->finish_event.id, task->finish_event.gen);
    }

    void PreemptableThread::start_thread(size_t stack_size, int core_id, 
                                         const char *debug_name)
    {
      pthread_attr_t attr;
      CHECK_PTHREAD( pthread_attr_init(&attr) );
      CHECK_PTHREAD( pthread_attr_setstacksize(&attr,stack_size) );
      if(proc_assignment)
	proc_assignment->bind_thread(core_id, &attr, debug_name);
      CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_entry, (void *)this) );
      CHECK_PTHREAD( pthread_attr_destroy(&attr) );
#ifdef DEADLOCK_TRACE
      get_runtime()->add_thread(&thread);
#endif
    }

    void PreemptableThread::run_task(Task *task, Processor actual_proc /*=NO_PROC*/)
    {
      Processor::TaskFuncPtr fptr = get_runtime()->task_table[task->func_id];
#if 0
      char argstr[100];
      argstr[0] = 0;
      for(size_t i = 0; (i < task->arglen) && (i < 40); i++)
        sprintf(argstr+2*i, "%02x", ((unsigned char *)(task->args))[i]);
      if(task->arglen > 40) strcpy(argstr+80, "...");
      log_util(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
               "utility task start: %d (%p) (%s)", task->func_id, fptr, argstr);
#endif
#ifdef EVENT_GRAPH_TRACE
      start_enclosing(task->finish_event);
      unsigned long long start = TimeStamp::get_current_time_in_micros();
#endif
      log_task.info("thread running ready task %p for proc " IDFMT "",
                              task, task->proc.id);
      task->mark_started();
      (*fptr)(task->args, task->arglen, 
              (actual_proc.exists() ? actual_proc : task->proc));
      task->mark_completed();
      // Capture the actual processor if necessary
      if (task->capture_proc && actual_proc.exists())
        task->proc = actual_proc;
      log_task.info("thread finished running task %p for proc " IDFMT "",
                              task, task->proc.id);
#ifdef EVENT_GRAPH_TRACE
      unsigned long long stop = TimeStamp::get_current_time_in_micros();
      finish_enclosing();
      log_event_graph.debug("Task Time: (" IDFMT ",%d) %lld",
                            task->finish_event.id, task->finish_event.gen,
                            (stop - start));
#endif
#if 0
      log_util(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
               "utility task end: %d (%p) (%s)", task->func_id, fptr, argstr);
#endif
      if(task->finish_event.exists())
        get_runtime()->get_genevent_impl(task->finish_event)->
                        trigger(task->finish_event.gen, gasnet_mynode());
    }

    /*static*/ bool PreemptableThread::preemptable_sleep(Event wait_for)
    {
      // check TLS to see if we're really a preemptable thread
      void *tls_val = gasnett_threadkey_get(cur_preemptable_thread);
      if(!tls_val) return false;

      PreemptableThread *me = (PreemptableThread *)tls_val;

      me->sleep_on_event(wait_for);
      return true;
    }
    
    /*static*/ void *PreemptableThread::thread_entry(void *data)
    {
      PreemptableThread *me = (PreemptableThread *)data;

      // set up TLS variable so we can remember who we are way down the call
      //  stack
      gasnett_threadkey_set(cur_preemptable_thread, me);

      // Initialize this value to NULL, it will get filled in the first time it is used
      CHECK_PTHREAD( pthread_setspecific(thread_timer_key, NULL) );

      // and then just call the virtual thread_main
      me->thread_main();

      return 0;
    }
 
    // Employ some fancy struct packing here to fit in 64 bytes
    struct SpawnTaskArgs : public BaseMedium {
      Processor proc;
      IDType start_id;
      IDType finish_id;
      size_t user_arglen;
      int priority;
      Processor::TaskFuncID func_id;
      Event::gen_t start_gen;
      Event::gen_t finish_gen;
    };

    GreenletTask::GreenletTask(Task *t, GreenletProcessor *p,
                               void *s, long *ssize)
      : greenlet(NULL, s, ssize), task(t), proc(p)
    {
    }

    GreenletTask::~GreenletTask(void)
    {
      // Make sure we are dead
      assert(isdead());
      // Remove our reference on our task
      if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
        delete task;
    }

    bool GreenletTask::event_triggered(void)
    {
      // Tell the processor we're awake
      proc->unpause_task(this);
      // Don't delete
      return false;
    }

    void GreenletTask::print_info(FILE *f)
    {
      fprintf(f,"Waiting greenlet %p of processor %s\n",
              this, proc->processor_name);
    }

    void* GreenletTask::run(void *arg)
    {
      GreenletThread *thread = static_cast<GreenletThread*>(arg);
      thread->run_task(task, proc->me);
      proc->complete_greenlet(this);
      return NULL;
    }

    GreenletThread::GreenletThread(GreenletProcessor *p)
      : proc(p)
    {
      current_task = NULL;
    }

    GreenletThread::~GreenletThread(void)
    {
    }

    Processor GreenletThread::get_processor(void) const
    {
      return proc->me;
    }

    void GreenletThread::thread_main(void)
    {
      greenlet::init_greenlet_thread();
      proc->initialize_processor();
      while (true)
      {
        bool quit = proc->execute_task();
        if (quit) break;
      }
      proc->finalize_processor();
    }

    void GreenletThread::sleep_on_event(Event wait_for)
    {
      assert(current_task != NULL);
      // Register ourselves as the waiter
      EventImpl::add_waiter(wait_for, current_task);
      GreenletTask *paused_task = current_task;
      // Tell the processor to pause us
      proc->pause_task(paused_task);
      // When we return the event has triggered
      assert(paused_task == current_task);
    }

    void GreenletThread::start_task(GreenletTask *task)
    {
      current_task = task;
      task->switch_to(this);
    }

    void GreenletThread::resume_task(GreenletTask *task)
    {
      current_task = task;
      task->switch_to(this);
    }

    void GreenletThread::return_to_root(void)
    {
      current_task = NULL;
      greenlet *root = greenlet::root();
      root->switch_to(NULL);
    }

    void GreenletThread::wait_for_shutdown(void)
    {
      void *result;
      pthread_join(thread, &result);
    }

    GreenletProcessor::GreenletProcessor(Processor _me, Processor::Kind _kind,
                                         size_t _stack_size, int init_stack_size,
                                         const char *name, int _core_id)
      : ProcessorImpl(_me, _kind), core_id(_core_id), proc_stack_size(_stack_size), 
        processor_name(name),
	condvar(mutex), done_initialization(false),
	shutdown(false), shutdown_trigger(false), 
        greenlet_thread(0), thread_state(GREENLET_RUNNING)
    {
    }

    GreenletProcessor::~GreenletProcessor(void)
    {
    }

    void GreenletProcessor::start_processor(void)
    {
      assert(greenlet_thread == 0);
      greenlet_thread = new GreenletThread(this);
      greenlet_thread->start_thread(proc_stack_size, core_id, processor_name);
      mutex.lock();
      if (!done_initialization)
        condvar.wait();
      mutex.unlock();
    }

    void GreenletProcessor::shutdown_processor(void)
    {
      mutex.lock();
      if (!shutdown_trigger)
	condvar.wait();
      assert(shutdown_trigger);
      shutdown = true;
      // Signal our thread in case it is asleep
      condvar.signal();
      mutex.unlock();
      greenlet_thread->wait_for_shutdown();
    }

    void GreenletProcessor::initialize_processor(void)
    {
      mutex.lock();
      done_initialization = true;
      condvar.signal();
      mutex.unlock();
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_INIT);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor init task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor init task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor init task: proc=" IDFMT "", me.id);
      }
    }

    void GreenletProcessor::finalize_processor(void)
    {
      Processor::TaskIDTable::iterator it = 
        get_runtime()->task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
      if(it != get_runtime()->task_table.end()) {
        log_task.info("calling processor shutdown task: proc=" IDFMT "", me.id);
        (it->second)(0, 0, me);
        log_task.info("finished processor shutdown task: proc=" IDFMT "", me.id);
      } else {
        log_task.info("no processor shutdown task: proc=" IDFMT "", me.id);
      }
    }

    void GreenletProcessor::enqueue_task(Task *task)
    {
      // Mark this task as ready
      task->mark_ready();
      mutex.lock();
      task_queue.insert(task, task->priority); 
      // Wake someone up if we aren't running
      if (thread_state == GREENLET_IDLE)
      {
        thread_state = GREENLET_RUNNING;
	condvar.signal();
      }
      mutex.unlock();
    }

    void GreenletProcessor::spawn_task(Processor::TaskFuncID func_id,
                                       const void *args, size_t arglen,
                                       Event start_event, Event finish_event,
                                       int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, finish_event, 
                            priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    void GreenletProcessor::spawn_task(Processor::TaskFuncID func_id,
                                       const void *args, size_t arglen,
                                       const Realm::ProfilingRequestSet &reqs,
                                       Event start_event, Event finish_event,
                                       int priority)
    {
      // create task object to hold args, etc.
      Task *task = new Task(me, func_id, args, arglen, reqs,
                            finish_event, priority, 1/*users*/);

      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(start_event.has_triggered()) {
        log_task.info("new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen,
                 finish_event.id, finish_event.gen);
        enqueue_task(task);
      } else {
        log_task.debug("deferring spawn: func=%d event=" IDFMT "/%d",
                 func_id, start_event.id, start_event.gen);
	EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
      }
    }

    bool GreenletProcessor::execute_task(void)
    {
      mutex.lock();
      // We should be running
      assert(thread_state == GREENLET_RUNNING);
      if (!resumable_tasks.empty())
      {
        // If we have tasks that are ready to resume, run them
        GreenletTask *to_resume = resumable_tasks.front();
        resumable_tasks.pop_front();
	mutex.unlock();
        greenlet_thread->resume_task(to_resume);
      }
      else if (task_queue.empty())
      {
        // Nothing to do, so let's go to sleep
        thread_state = GREENLET_IDLE;
	condvar.wait();
        if (!shutdown)
          assert(thread_state == GREENLET_RUNNING);
	mutex.unlock();
      }
      else
      {
        // Pull a task off the queue and execute it
        Task *task = task_queue.pop();
        if (task->func_id == 0) {
          // This is the kill pill so we need to handle it special
          finished();
          // Mark that we received the shutdown trigger
          shutdown_trigger = true;
	  condvar.signal();
	  mutex.unlock();
          // Trigger the completion task
          if (__sync_fetch_and_add(&(task->run_count),1) == 0)
            get_runtime()->get_genevent_impl(task->finish_event)->
                          trigger(task->finish_event.gen, gasnet_mynode());
          // Delete the task
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        } else {
	  mutex.unlock();
          if (__sync_fetch_and_add(&(task->run_count),1) == 0) {
            GreenletStack stack;
            if (!allocate_stack(stack))
              create_stack(stack);
            GreenletTask *green_task = new GreenletTask(task, this,
                                            stack.stack, &stack.stack_size);
            greenlet_thread->start_task(green_task);
          } else {
            // Remove our deletion reference
            if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
              delete task;
          }
        }
      }
      // If we have any complete greenlets, clean them up
      if (!complete_greenlets.empty())
      {
        for (std::vector<GreenletTask*>::const_iterator it = 
              complete_greenlets.begin(); it != complete_greenlets.end(); it++)
        {
          delete (*it);
        }
        complete_greenlets.clear();
      }
      if (shutdown)
        return (task_queue.empty() && resumable_tasks.empty());
      return false;
    }

    void GreenletProcessor::pause_task(GreenletTask *paused_task)
    {
      mutex.lock();
      bool found = false;
      // Go through and see if the task is already ready
      for (std::list<GreenletTask*>::reverse_iterator it = 
            resumable_tasks.rbegin(); it != resumable_tasks.rend(); it++)
      {
        if ((*it) == paused_task)
        {
          found = true;
          // Reverse iterator conversion requires adding 1 first
          resumable_tasks.erase((++it).base());
          break;
        }
      }
      // If we found it we're already ready so just return
      if (found)
      {
	mutex.unlock();
        return;
      }
      // Add it to the list of paused tasks
      paused_tasks.insert(paused_task);
      // Now figure out what we want to do
      if (!resumable_tasks.empty())
      {
        // Pick a task to resume and run it
        GreenletTask *to_resume = resumable_tasks.front();
        resumable_tasks.pop_front();
	mutex.unlock();
        greenlet_thread->resume_task(to_resume);
      }
      else if (!task_queue.empty())
      {
        // Pull a task off the queue and execute it
        Task *task = task_queue.pop();
        if (task->func_id == 0) {
          // This is the kill pill so we need to handle it special
          finished();
          // Mark that we received the shutdown trigger
          shutdown_trigger = true;
	  condvar.signal();
	  mutex.unlock();
          // Trigger the completion task
          if (__sync_fetch_and_add(&(task->run_count),1) == 0)
            get_runtime()->get_genevent_impl(task->finish_event)->
                          trigger(task->finish_event.gen, gasnet_mynode());
          // Delete the task
          if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
            delete task;
        } else {
	  mutex.unlock();
          if (__sync_fetch_and_add(&(task->run_count),1) == 0) {
            GreenletStack stack;
            if (!allocate_stack(stack))
              create_stack(stack);
            GreenletTask *green_task = new GreenletTask(task, this,
                                            stack.stack, &stack.stack_size);
            greenlet_thread->start_task(green_task);
          } else {
            // Remove our deletion reference
            if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
              delete task;
          }
        }
      }
      else
      {
	mutex.unlock();
        // Nothing to do, send us back to the root at which
        // point we'll likely go to sleep
        greenlet_thread->return_to_root(); 
      }
    }

    void GreenletProcessor::unpause_task(GreenletTask *paused_task)
    {
      mutex.lock();
      paused_tasks.erase(paused_task);
      resumable_tasks.push_back(paused_task);
      if (thread_state == GREENLET_IDLE)
      {
        thread_state = GREENLET_RUNNING;
	condvar.signal();
      }
      mutex.unlock();
    }

    bool GreenletProcessor::allocate_stack(GreenletStack &stack)
    {
      // No need to hold the lock since only one thread is here
      if (!greenlet_stacks.empty())
      {
        stack = greenlet_stacks.back();
        greenlet_stacks.pop_back();
        return true; // succeeded
      }
      return false; // failed
    }

    void GreenletProcessor::create_stack(GreenletStack &stack)
    {
      // We need to make a stack
      // Set the suggested stack size
      stack.stack_size = proc_stack_size;
      // Then call the greenlet library
      stack.stack = greenlet::alloc_greenlet_stack(&stack.stack_size);
    }

    void GreenletProcessor::complete_greenlet(GreenletTask *greenlet)
    {
      // No need for the lock here, only one thread 
      complete_greenlets.push_back(greenlet);
      // Tricky optimization here, we can actually release
      // the stack now because we know there is only one thread
      // and we are guaranteed to exit after this call so by the
      // time this thread will try to re-use the stack we are
      // guaranteed to have finished using it.
      greenlet_stacks.push_back(GreenletStack());
      GreenletStack &last = greenlet_stacks.back();
      last.stack = greenlet->release_stack(&last.stack_size);
    }

    // can't be static if it's used in a template...
    void handle_spawn_task_message(SpawnTaskArgs args,
				   const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ProcessorImpl *p = get_runtime()->get_processor_impl(args.proc);
      log_task.debug("remote spawn request: proc_id=" IDFMT " task_id=%d event=" IDFMT "/%d",
	       args.proc.id, args.func_id, args.start_id, args.start_gen);
      Event start_event, finish_event;
      start_event.id = args.start_id;
      start_event.gen = args.start_gen;
      finish_event.id = args.finish_id;
      finish_event.gen = args.finish_gen;
      if (args.user_arglen == datalen) {
        // Only have user data
        p->spawn_task(args.func_id, data, datalen,
                      start_event, finish_event, args.priority);
      } else {
        // Unpack the profiling set
        Realm::ProfilingRequestSet reqs;
        reqs.deserialize(((char*)data)+args.user_arglen);
        p->spawn_task(args.func_id, data, args.user_arglen, reqs,
                      start_event, finish_event, args.priority);
      }
    }

    typedef ActiveMessageMediumNoReply<SPAWN_TASK_MSGID,
				       SpawnTaskArgs,
				       handle_spawn_task_message> SpawnTaskMessage;

    class RemoteProcessor : public ProcessorImpl {
    public:
      RemoteProcessor(Processor _me, Processor::Kind _kind)
	: ProcessorImpl(_me, _kind)
      {
      }

      ~RemoteProcessor(void)
      {
      }

      virtual void start_processor(void)
      {
        assert(0);
      }

      virtual void shutdown_processor(void)
      {
        assert(0);
      }

      virtual void initialize_processor(void)
      {
        assert(0);
      }

      virtual void finalize_processor(void)
      {
        assert(0);
      }

      virtual void enqueue_task(Task *task)
      {
        // should never be called
        assert(0);
      }

      virtual void tasks_available(int priority)
      {
	log_task.warning("remote processor " IDFMT " being told about local tasks ready?",
			 me.id);
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority)
      {
	log_task.debug("spawning remote task: proc=" IDFMT " task=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
		 me.id, func_id, 
		 start_event.id, start_event.gen,
		 finish_event.id, finish_event.gen);
	SpawnTaskArgs msgargs;
	msgargs.proc = me;
	msgargs.func_id = func_id;
        msgargs.start_id = start_event.id;
        msgargs.start_gen = start_event.gen;
        msgargs.finish_id = finish_event.id;
        msgargs.finish_gen = finish_event.gen;
        msgargs.priority = priority;
        msgargs.user_arglen = arglen;
	SpawnTaskMessage::request(ID(me).node(), msgargs, args, arglen,
				  PAYLOAD_COPY);
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const Realm::ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority)
      {
        log_task.debug("spawning remote task: proc=" IDFMT " task=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
		 me.id, func_id, 
		 start_event.id, start_event.gen,
		 finish_event.id, finish_event.gen);
	SpawnTaskArgs msgargs;
	msgargs.proc = me;
	msgargs.func_id = func_id;
        msgargs.start_id = start_event.id;
        msgargs.start_gen = start_event.gen;
        msgargs.finish_id = finish_event.id;
        msgargs.finish_gen = finish_event.gen;
        msgargs.priority = priority;
        msgargs.user_arglen = arglen;
        // Make a copy of the arguments and the profiling requests in
        // the same buffer so that we can copy them over
        size_t msg_buffer_size = arglen + reqs.compute_size();
        void *msg_buffer = malloc(msg_buffer_size);
        memcpy(msg_buffer,args,arglen);
        reqs.serialize(((char*)msg_buffer)+arglen);
        SpawnTaskMessage::request(ID(me).node(), msgargs, msg_buffer,
                                  msg_buffer_size, PAYLOAD_FREE);
      }
    };

    ///////////////////////////////////////////////////
    // Runtime

    EventImpl *RuntimeImpl::get_event_impl(Event e)
    {
      ID id(e);
      switch(id.type()) {
      case ID::ID_EVENT:
	return get_genevent_impl(e);
      case ID::ID_BARRIER:
	return get_barrier_impl(e);
      default:
	assert(0);
      }
    }

    GenEventImpl *RuntimeImpl::get_genevent_impl(Event e)
    {
      ID id(e);
      assert(id.type() == ID::ID_EVENT);

      Node *n = &nodes[id.node()];
      GenEventImpl *impl = n->events.lookup_entry(id.index(), id.node());
      assert(impl->me == id);

      // check to see if this is for a generation more than one ahead of what we
      //  know of - this should only happen for remote events, but if it does it means
      //  there are some generations we don't know about yet, so we can catch up (and
      //  notify any local waiters right away)
      impl->check_for_catchup(e.gen - 1);

      return impl;
    }

    BarrierImpl *RuntimeImpl::get_barrier_impl(Event e)
    {
      ID id(e);
      assert(id.type() == ID::ID_BARRIER);

      Node *n = &nodes[id.node()];
      BarrierImpl *impl = n->barriers.lookup_entry(id.index(), id.node());
      assert(impl->me == id);
      return impl;
    }

    ReservationImpl *RuntimeImpl::get_lock_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_LOCK:
	{
	  Node *n = &nodes[id.node()];
	  ReservationImpl *impl = n->reservations.lookup_entry(id.index(), id.node());
	  assert(impl->me == id.convert<Reservation>());
	  return impl;
#if 0
	  std::vector<ReservationImpl>& locks = nodes[id.node()].locks;

	  unsigned index = id.index();
	  if(index >= n->num_locks) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->locks.size();
	    assert(oldsize < MAX_LOCAL_LOCKS);
	    if(index >= oldsize) { // only it's still too small
              assert((index+1) < MAX_LOCAL_LOCKS);
	      n->locks.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->locks[i].init(ID(ID::ID_LOCK, id.node(), i).convert<Reservation>(),
				 id.node());
	      n->num_locks = index + 1;
	    }
	  }
	  return &(locks[index]);
#endif
	}

      case ID::ID_INDEXSPACE:
	return &(get_index_space_impl(id)->lock);

      case ID::ID_INSTANCE:
	return &(get_instance_impl(id)->lock);

      case ID::ID_PROCGROUP:
	return &(get_procgroup_impl(id)->lock);

      default:
	assert(0);
      }
    }

    template <class T>
    inline T *null_check(T *ptr)
    {
      assert(ptr != 0);
      return ptr;
    }

    MemoryImpl *RuntimeImpl::get_memory_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_MEMORY:
      case ID::ID_ALLOCATOR:
      case ID::ID_INSTANCE:
	if(id.index_h() == ID::ID_GLOBAL_MEM)
	  return global_memory;
	return null_check(nodes[id.node()].memories[id.index_h()]);

      default:
	assert(0);
      }
    }

    ProcessorImpl *RuntimeImpl::get_processor_impl(ID id)
    {
      if(id.type() == ID::ID_PROCGROUP)
	return get_procgroup_impl(id);

      assert(id.type() == ID::ID_PROCESSOR);
      return null_check(nodes[id.node()].processors[id.index()]);
    }

    ProcessorGroup *RuntimeImpl::get_procgroup_impl(ID id)
    {
      assert(id.type() == ID::ID_PROCGROUP);

      Node *n = &nodes[id.node()];
      ProcessorGroup *impl = n->proc_groups.lookup_entry(id.index(), id.node());
      assert(impl->me == id.convert<Processor>());
      return impl;
    }

    IndexSpaceImpl *RuntimeImpl::get_index_space_impl(ID id)
    {
      assert(id.type() == ID::ID_INDEXSPACE);

      Node *n = &nodes[id.node()];
      IndexSpaceImpl *impl = n->index_spaces.lookup_entry(id.index(), id.node());
      assert(impl->me == id.convert<IndexSpace>());
      return impl;
#if 0
      unsigned index = id.index();
      if(index >= n->index_spaces.size()) {
	AutoHSLLock a(n->mutex); // take lock before we actually resize

	if(index >= n->index_spaces.size())
	  n->index_spaces.resize(index + 1);
      }

      if(!n->index_spaces[index]) { // haven't seen this metadata before?
	//printf("UNKNOWN METADATA " IDFMT "\n", id.id());
	AutoHSLLock a(n->mutex); // take lock before we actually allocate
	if(!n->index_spaces[index]) {
	  n->index_spaces[index] = new IndexSpaceImpl(id.convert<IndexSpace>());
	} 
      }

      return n->index_spaces[index];
#endif
    }

    RegionInstanceImpl *RuntimeImpl::get_instance_impl(ID id)
    {
      assert(id.type() == ID::ID_INSTANCE);
      MemoryImpl *mem = get_memory_impl(id);
      
      AutoHSLLock al(mem->mutex);

      if(id.index_l() >= mem->instances.size()) {
	assert(id.node() != gasnet_mynode());

	size_t old_size = mem->instances.size();
	if(id.index_l() >= old_size) {
	  // still need to grow (i.e. didn't lose the race)
	  mem->instances.resize(id.index_l() + 1);

	  // don't have region/offset info - will have to pull that when
	  //  needed
	  for(unsigned i = old_size; i <= id.index_l(); i++) 
	    mem->instances[i] = 0;
	}
      }

      if(!mem->instances[id.index_l()]) {
	if(!mem->instances[id.index_l()]) {
	  //printf("[%d] creating proxy instance: inst=" IDFMT "\n", gasnet_mynode(), id.id());
	  mem->instances[id.index_l()] = new RegionInstanceImpl(id.convert<RegionInstance>(), mem->me);
	}
      }
	  
      return mem->instances[id.index_l()];
    }

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

    /*static*/ const IndexSpace IndexSpace::NO_SPACE = { 0 };
    /*static*/ const Domain Domain::NO_DOMAIN = Domain();

    /*static*/ IndexSpace IndexSpace::create_index_space(size_t num_elmts)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpaceImpl *impl = get_runtime()->local_index_space_free_list->alloc_entry();
      
      impl->init(impl->me, NO_SPACE, num_elmts);
      
      log_meta.info("index space created: id=" IDFMT " num_elmts=%zd",
	       impl->me.id, num_elmts);
      return impl->me;
    }

    /*static*/ IndexSpace IndexSpace::create_index_space(const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpaceImpl *impl = get_runtime()->local_index_space_free_list->alloc_entry();
      
      // TODO: actually decide when to safely consider a subregion frozen
      impl->init(impl->me, NO_SPACE, mask.get_num_elmts(), &mask, true);
      
      log_meta.info("index space created: id=" IDFMT " num_elmts=%d",
	       impl->me.id, mask.get_num_elmts());
      return impl->me;
    }

    /*static*/ IndexSpace IndexSpace::create_index_space(IndexSpace parent, const ElementMask &mask, bool allocable)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpaceImpl *impl = get_runtime()->local_index_space_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).type() == ID::ID_INDEXSPACE);

      StaticAccess<IndexSpaceImpl> p_data(get_runtime()->get_index_space_impl(parent));

      impl->init(impl->me, parent,
		 p_data->num_elmts, 
		 &mask,
		 !allocable);  // TODO: actually decide when to safely consider a subregion frozen
      
      log_meta.info("index space created: id=" IDFMT " parent=" IDFMT " (num_elmts=%zd)",
	       impl->me.id, parent.id, p_data->num_elmts);
      return impl->me;
    }

    IndexSpaceAllocator IndexSpace::create_allocator(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceAllocatorImpl *a_impl = new IndexSpaceAllocatorImpl(get_runtime()->get_index_space_impl(*this));
      return IndexSpaceAllocator(a_impl);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   size_t elem_size,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      std::vector<size_t> field_sizes(1);
      field_sizes[0] = elem_size;

      return create_instance(memory, field_sizes, 1, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   size_t elem_size,
                                           const Realm::ProfilingRequestSet &reqs,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      std::vector<size_t> field_sizes(1);
      field_sizes[0] = elem_size;

      return create_instance(memory, field_sizes, 1, reqs, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
					   ReductionOpID redop_id) const
    {
      Realm::ProfilingRequestSet requests;
      return create_instance(memory, field_sizes, block_size, requests, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
                                           const Realm::ProfilingRequestSet &reqs,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

      size_t elem_size = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++)
	elem_size += *it;

      size_t num_elements;
      int linearization_bits[RegionInstanceImpl::MAX_LINEARIZATION_LEN];
      if(get_dim() > 0) {
	// we have a rectangle - figure out its volume and create based on that
	Arrays::Rect<1> inst_extent;
	switch(get_dim()) {
	case 1:
	  {
	    std::vector<Layouts::DimKind> kind_vec;
	    std::vector<size_t> size_vec;
	    kind_vec.push_back(Layouts::DIM_X);
	    size_vec.push_back(get_rect<1>().dim_size(0));
	    Layouts::SplitDimLinearization<1> cl(kind_vec, size_vec);
	    //Arrays::FortranArrayLinearization<1> cl(get_rect<1>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<1>(Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<1>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 2:
	  {
	    Arrays::FortranArrayLinearization<2> cl(get_rect<2>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<2>(Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<2>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 3:
	  {
	    Arrays::FortranArrayLinearization<3> cl(get_rect<3>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<3>(Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<3>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	default: assert(0);
	}

	num_elements = inst_extent.volume();
	//printf("num_elements = %zd\n", num_elements);
      } else {
	IndexSpaceImpl *r = get_runtime()->get_index_space_impl(get_index_space());

	StaticAccess<IndexSpaceImpl> data(r);
	assert(data->num_elmts > 0);

#ifdef FULL_SIZE_INSTANCES
	num_elements = data->last_elmt + 1;
	// linearization is an identity translation
	Translation<1> inst_offset(0);
	DomainLinearization dl = DomainLinearization::from_mapping<1>(Mapping<1,1>::new_dynamic_mapping(inst_offset));
	dl.serialize(linearization_bits);
#else
	num_elements = data->last_elmt - data->first_elmt + 1;
        // round num_elements up to a multiple of 4 to line things up better with vectors, cache lines, etc.
        if(num_elements & 3) {
          if (block_size == num_elements)
            block_size = (block_size + 3) & ~(size_t)3;
          num_elements = (num_elements + 3) & ~(size_t)3;
        }
	if(block_size > num_elements)
	  block_size = num_elements;

	//printf("CI: %zd %zd %zd\n", data->num_elmts, data->first_elmt, data->last_elmt);

	Translation<1> inst_offset(-(int)(data->first_elmt));
	DomainLinearization dl = DomainLinearization::from_mapping<1>(Mapping<1,1>::new_dynamic_mapping(inst_offset));
	dl.serialize(linearization_bits);
#endif
      }

      // for instances with a single element, there's no real difference between AOS and
      //  SOA - force the block size to indicate "full SOA" as it makes the DMA code
      //  use a faster path
      if(field_sizes.size() == 1)
	block_size = num_elements;

#ifdef FORCE_SOA_INSTANCE_LAYOUT
      // the big hammer
      if(block_size != num_elements) {
        log_inst.info("block size changed from %zd to %zd (SOA)",
                      block_size, num_elements);
        block_size = num_elements;
      }
#endif

      if(block_size > 1) {
	size_t leftover = num_elements % block_size;
	if(leftover > 0)
	  num_elements += (block_size - leftover);
      }

      size_t inst_bytes = elem_size * num_elements;

      RegionInstance i = m_impl->create_instance(get_index_space(), linearization_bits, inst_bytes,
						 block_size, elem_size, field_sizes,
						 redop_id,
						 -1 /*list size*/, reqs,
						 RegionInstance::NO_INST);
      log_meta.info("instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd",
	       this->is_id, memory.id, i.id, inst_bytes);
      return i;
    }

    RegionInstance Domain::create_hdf5_instance(const char *file_name,
                                                const std::vector<size_t> &field_sizes,
                                                const std::vector<const char*> &field_files,
                                                bool read_only) const
    {
#ifndef USE_HDF
      // TODO: Implement this
      assert(false);
      return RegionInstance::NO_INST;
#else
      assert(field_sizes.size() == field_files.size());
      Memory memory = Memory::NO_MEMORY;
      Machine machine = Machine::get_machine();
      std::set<Memory> mem;
      machine.get_all_memories(mem);
      for(std::set<Memory>::iterator it = mem.begin(); it != mem.end(); it++) {
        if (it->kind() == Memory::HDF_MEM) {
          memory = *it;
        }
      }
      assert(memory.kind() == Memory::HDF_MEM);
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      HDFMemory* hdf_mem = (HDFMemory*) get_runtime()->get_memory_impl(memory);
      size_t elem_size = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++)
	elem_size += *it;
      
      size_t num_elements;
      int linearization_bits[RegionInstanceImpl::MAX_LINEARIZATION_LEN];
      assert(get_dim() > 0);
      {
        Arrays::Rect<1> inst_extent;
        switch(get_dim()) {
	case 1:
	  {
	    Arrays::FortranArrayLinearization<1> cl(get_rect<1>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<1>(Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<1>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 2:
	  {
	    Arrays::FortranArrayLinearization<2> cl(get_rect<2>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<2>(Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<2>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 3:
	  {
	    Arrays::FortranArrayLinearization<3> cl(get_rect<3>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<3>(Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<3>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	default: assert(0);
	}

	num_elements = inst_extent.volume();
      }

      Realm::ProfilingRequestSet requests;
      size_t inst_bytes = elem_size * num_elements;
      RegionInstance i = hdf_mem->create_instance(get_index_space(), linearization_bits, inst_bytes,
                                                  get_volume() /*block_size*/, elem_size, field_sizes,
                                                  0 /*redop_id*/, -1/*list_size*/, requests, RegionInstance::NO_INST,
                                                  file_name, field_files, *this, read_only);
      log_meta.info("instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd",
	       this->is_id, memory.id, i.id, inst_bytes);
      return i;
#endif
    }

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

    void IndexSpace::destroy(Event wait_on) const
    {
      assert(wait_on.has_triggered());
      //assert(0);
    }

    void IndexSpaceAllocator::destroy(void) 
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (impl != NULL)
      {
        delete (IndexSpaceAllocatorImpl *)impl;
        // Avoid double frees
        impl = NULL;
      }
    }

    const ElementMask &IndexSpace::get_valid_mask(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceImpl *r_impl = get_runtime()->get_index_space_impl(*this);
#ifdef COHERENT_BUT_BROKEN_WAY
      // for now, just hand out the valid mask for the master allocator
      //  and hope it's accessible to the caller
      SharedAccess<IndexSpaceImpl> data(r_impl);
      assert((data->valid_mask_owners >> gasnet_mynode()) & 1);
#else
      if(!r_impl->valid_mask_complete) {
	Event wait_on = r_impl->request_valid_mask();
	
	log_copy.info("missing valid mask (" IDFMT "/%p) - waiting for " IDFMT "/%d",
		      id, r_impl->valid_mask,
		      wait_on.id, wait_on.gen);

	wait_on.wait();
      }
#endif
      return *(r_impl->valid_mask);
    }

    Event IndexSpace::create_equal_subspaces(size_t count, size_t granularity,
                                             std::vector<IndexSpace>& subspaces,
                                             bool mutable_results,
                                             Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_equal_subspaces(size_t count, size_t granularity,
                                             std::vector<IndexSpace>& subspaces,
                                             const Realm::ProfilingRequestSet &reqs,
                                             bool mutable_results,
                                             Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_weighted_subspaces(size_t count, size_t granularity,
                                                const std::vector<int>& weights,
                                                std::vector<IndexSpace>& subspaces,
                                                bool mutable_results,
                                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_weighted_subspaces(size_t count, size_t granularity,
                                                const std::vector<int>& weights,
                                                std::vector<IndexSpace>& subspaces,
                                                const Realm::ProfilingRequestSet &reqs,
                                                bool mutable_results,
                                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::compute_index_spaces(std::vector<BinaryOpDescriptor>& pairs,
                                           bool mutable_results,
					   Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::compute_index_spaces(std::vector<BinaryOpDescriptor>& pairs,
                                           const Realm::ProfilingRequestSet &reqs,
                                           bool mutable_results,
					   Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::reduce_index_spaces(IndexSpaceOperation op,
                                          const std::vector<IndexSpace>& spaces,
                                          IndexSpace& result,
                                          bool mutable_results,
                                          IndexSpace parent /*= IndexSpace::NO_SPACE*/,
				          Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::reduce_index_spaces(IndexSpaceOperation op,
                                          const std::vector<IndexSpace>& spaces,
                                          const Realm::ProfilingRequestSet &reqs,
                                          IndexSpace& result,
                                          bool mutable_results,
                                          IndexSpace parent /*= IndexSpace::NO_SPACE*/,
				          Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_field(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<DomainPoint, IndexSpace>& subspaces,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_field(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<DomainPoint, IndexSpace>& subspaces,
                                const Realm::ProfilingRequestSet &reqs,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_image(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<IndexSpace, IndexSpace>& subspaces,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_image(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<IndexSpace, IndexSpace>& subspaces,
                                const Realm::ProfilingRequestSet &reqs,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_preimage(
                                 const std::vector<FieldDataDescriptor>& field_data,
                                 std::map<IndexSpace, IndexSpace>& subspaces,
                                 bool mutable_results,
                                 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_preimage(
                                 const std::vector<FieldDataDescriptor>& field_data,
                                 std::map<IndexSpace, IndexSpace>& subspaces,
                                 const Realm::ProfilingRequestSet &reqs,
                                 bool mutable_results,
                                 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    ///////////////////////////////////////////////////
    // Element Masks

    ElementMask::ElementMask(void)
      : first_element(-1), num_elements(-1), memory(Memory::NO_MEMORY), offset(-1),
	raw_data(0), first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
    }

    ElementMask::ElementMask(int _num_elements, int _first_element /*= 0*/)
      : first_element(_first_element), num_elements(_num_elements), memory(Memory::NO_MEMORY), offset(-1), first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);
      //((ElementMaskImpl *)raw_data)->count = num_elements;
      //((ElementMaskImpl *)raw_data)->offset = first_element;
    }

    ElementMask::ElementMask(const ElementMask &copy_from, 
			     int _num_elements /*= -1*/, int _first_element /*= 0*/)
    {
      first_element = copy_from.first_element;
      num_elements = copy_from.num_elements;
      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);

      if(copy_from.raw_data) {
	memcpy(raw_data, copy_from.raw_data, bytes_needed);
      } else {
	get_runtime()->get_memory_impl(copy_from.memory)->get_bytes(copy_from.offset, raw_data, bytes_needed);
      }
    }

    ElementMask::~ElementMask(void)
    {
      if (raw_data) {
        free(raw_data);
        raw_data = 0;
      }
    }

    ElementMask& ElementMask::operator=(const ElementMask &rhs)
    {
      first_element = rhs.first_element;
      num_elements = rhs.num_elements;
      first_enabled_elmt = rhs.first_enabled_elmt;
      last_enabled_elmt = rhs.last_enabled_elmt;
      size_t bytes_needed = rhs.raw_size();
      if (raw_data)
        free(raw_data);
      raw_data = calloc(1, bytes_needed);
      if (rhs.raw_data)
        memcpy(raw_data, rhs.raw_data, bytes_needed);
      else
        get_runtime()->get_memory_impl(rhs.memory)->get_bytes(rhs.offset, raw_data, bytes_needed);
      return *this;
    }

    void ElementMask::init(int _first_element, int _num_elements, Memory _memory, off_t _offset)
    {
      first_element = _first_element;
      num_elements = _num_elements;
      memory = _memory;
      offset = _offset;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = get_runtime()->get_memory_impl(memory)->get_direct_ptr(offset, bytes_needed);
    }

    void ElementMask::enable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("ENABLE %p %d %d %d " IDFMT "\n", raw_data, offset, start, count, impl->bits[0]);
	int pos = start - first_element;
        assert(pos < num_elements);
	for(int i = 0; i < count; i++) {
	  uint64_t *ptr = &(impl->bits[pos >> 6]);
	  *ptr |= (1ULL << (pos & 0x3f));
	  pos++;
	}
	//printf("ENABLED %p %d %d %d " IDFMT "\n", raw_data, offset, start, count, impl->bits[0]);
      } else {
	//printf("ENABLE(2) " IDFMT " %d %d %d\n", memory.id, offset, start, count);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  off_t ofs = offset + ((pos >> 6) << 3);
	  uint64_t val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("ENABLED(2) %d,  " IDFMT "\n", ofs, val);
	  val |= (1ULL << (pos & 0x3f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
      }

      if((first_enabled_elmt < 0) || (start < first_enabled_elmt))
	first_enabled_elmt = start;

      if((last_enabled_elmt < 0) || ((start+count-1) > last_enabled_elmt))
	last_enabled_elmt = start + count - 1;
    }

    void ElementMask::disable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  uint64_t *ptr = &(impl->bits[pos >> 6]);
	  *ptr &= ~(1ULL << (pos & 0x3f));
	  pos++;
	}
      } else {
	//printf("DISABLE(2) " IDFMT " %d %d %d\n", memory.id, offset, start, count);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  off_t ofs = offset + ((pos >> 6) << 3);
	  uint64_t val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("DISABLED(2) %d,  " IDFMT "\n", ofs, val);
	  val &= ~(1ULL << (pos & 0x3f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
      }

      // not really right
      if(start == first_enabled_elmt) {
	//printf("pushing first: %d -> %d\n", first_enabled_elmt, first_enabled_elmt+1);
	first_enabled_elmt++;
      }
    }

    int ElementMask::find_enabled(int count /*= 1 */, int start /*= 0*/) const
    {
      if(start == 0)
	start = first_enabled_elmt;
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("FIND_ENABLED %p %d %d " IDFMT "\n", raw_data, first_element, count, impl->bits[0]);
	for(int pos = start; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    uint64_t bit = ((impl->bits[pos >> 6] >> (pos & 0x3f))) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
	//printf("FIND_ENABLED(2) " IDFMT " %d %d %d\n", memory.id, offset, first_element, count);
	for(int pos = start; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    off_t ofs = offset + ((pos >> 6) << 3);
	    uint64_t val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    uint64_t bit = (val >> (pos & 0x3f)) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      }
      return -1;
    }

    int ElementMask::find_disabled(int count /*= 1 */, int start /*= 0*/) const
    {
      if((start == 0) && (first_enabled_elmt > 0))
	start = first_enabled_elmt;
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	for(int pos = start; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    uint64_t bit = ((impl->bits[pos >> 6] >> (pos & 0x3f))) & 1;
	    if(bit != 0) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	assert(0);
      }
      return -1;
    }

    size_t ElementMask::raw_size(void) const
    {
      return ElementMaskImpl::bytes_needed(offset, num_elements);
    }

    const void *ElementMask::get_raw(void) const
    {
      return raw_data;
    }

    void ElementMask::set_raw(const void *data)
    {
      assert(0);
    }

    bool ElementMask::is_set(int ptr) const
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	
	int pos = ptr;// - first_element;
	uint64_t val = (impl->bits[pos >> 6]);
        uint64_t bit = ((val) >> (pos & 0x3f));
        return ((bit & 1) != 0);
      } else {
        assert(0);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

	int pos = ptr - first_element;
	off_t ofs = offset + ((pos >> 6) << 3);
	uint64_t val;
	m_impl->get_bytes(ofs, &val, sizeof(val));
        uint64_t bit = ((val) >> (pos & 0x3f));
        return ((bit & 1) != 0);
      }
    }

    size_t ElementMask::pop_count(bool enabled) const
    {
      size_t count = 0;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        int max_full = (num_elements >> 6);
        bool remainder = (num_elements % 64) != 0;
        for (int index = 0; index < max_full; index++)
          count += __builtin_popcountll(impl->bits[index]);
        if (remainder)
          count += __builtin_popcountll(impl->bits[max_full]);
        if (!enabled)
          count = num_elements - count;
      } else {
        // TODO: implement this
        assert(0);
      }
      return count;
    }

    bool ElementMask::operator!(void) const
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        const int max_full = ((num_elements+63) >> 6);
        for (int index = 0; index < max_full; index++) {
          if (impl->bits[index])
            return false;
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return true;
    }

    bool ElementMask::operator==(const ElementMask &other) const
    {
      if (num_elements != other.num_elements)
        return false;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
          const int max_full = ((num_elements+63) >> 6);
          for (int index = 0; index < max_full; index++)
          {
            if (impl->bits[index] != other_impl->bits[index])
              return false;
          }
        } else {
          // TODO: Implement this
          assert(false);
        }
      } else {
        // TODO: Implement this
        assert(false);
      }
      return true;
    }

    bool ElementMask::operator!=(const ElementMask &other) const
    {
      return !((*this) == other);
    }

    ElementMask ElementMask::operator|(const ElementMask &other) const
    {
      ElementMask result(num_elements);
      ElementMaskImpl *target = (ElementMaskImpl *)result.raw_data;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
          assert(num_elements == other.num_elements);
          const int max_full = ((num_elements+63) >> 6);
          for (int index = 0; index < max_full; index++) {
            target->bits[index] = impl->bits[index] | other_impl->bits[index]; 
          }
        } else {
          // TODO implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return result;
    }

    ElementMask ElementMask::operator&(const ElementMask &other) const
    {
      ElementMask result(num_elements);
      ElementMaskImpl *target = (ElementMaskImpl *)result.raw_data;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
          assert(num_elements == other.num_elements);
          const int max_full = ((num_elements+63) >> 6);
          for (int index = 0; index < max_full; index++) {
            target->bits[index] = impl->bits[index] & other_impl->bits[index];
          }
        } else {
          // TODO: implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return result;
    }

    ElementMask ElementMask::operator-(const ElementMask &other) const
    {
      ElementMask result(num_elements);
      ElementMaskImpl *target = (ElementMaskImpl *)result.raw_data;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
          assert(num_elements == other.num_elements);
          const int max_full = ((num_elements+63) >> 6);
          for (int index = 0; index < max_full; index++) {
            target->bits[index] = impl->bits[index] & ~(other_impl->bits[index]);
          }
        } else {
          // TODO: implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return result;
    }

    ElementMask& ElementMask::operator|=(const ElementMask &other)
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
          assert(num_elements == other.num_elements);
          const int max_full = ((num_elements+63) >> 6);
          for (int index = 0; index < max_full; index++) {
            impl->bits[index] |= other_impl->bits[index];
          }
        } else {
          // TODO: implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return *this;
    }

    ElementMask& ElementMask::operator&=(const ElementMask &other)
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
          assert(num_elements == other.num_elements);
          const int max_full = ((num_elements+63) >> 6);
          for (int index = 0; index < max_full; index++) {
            impl->bits[index] &= other_impl->bits[index];
          }
        } else {
          // TODO: implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return *this;
    }

    ElementMask& ElementMask::operator-=(const ElementMask &other)
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
          assert(num_elements == other.num_elements);
          const int max_full = ((num_elements+63) >> 6);
          for (int index = 0; index < max_full; index++) {
            impl->bits[index] &= ~(other_impl->bits[index]);
          }
        } else {
          // TODO: implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return *this;
    }

    ElementMask::OverlapResult ElementMask::overlaps_with(const ElementMask& other,
							  off_t max_effort /*= -1*/) const
    {
      if (raw_data != 0) {
        ElementMaskImpl *i1 = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *i2 = (ElementMaskImpl *)(other.raw_data);
          assert(num_elements == other.num_elements);
          for(int i = 0; i < (num_elements + 63) >> 6; i++)
            if((i1->bits[i] & i2->bits[i]) != 0)
              return ElementMask::OVERLAP_YES;
          return ElementMask::OVERLAP_NO;
        } else {
          return ElementMask::OVERLAP_MAYBE;
        }
      } else {
        return ElementMask::OVERLAP_MAYBE;
      }
    }

    ElementMask::Enumerator *ElementMask::enumerate_enabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 1);
    }

    ElementMask::Enumerator *ElementMask::enumerate_disabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 0);
    }

    ElementMask::Enumerator::Enumerator(const ElementMask& _mask, int _start, int _polarity)
      : mask(_mask), pos(_start), polarity(_polarity) {}

    ElementMask::Enumerator::~Enumerator(void) {}

    bool ElementMask::Enumerator::get_next(int &position, int &length)
    {
      if(mask.raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)(mask.raw_data);

	// are we already off the end?
	if(pos >= mask.num_elements)
	  return false;

        // if our current pos is below the first known-set element, skip to there
        if((mask.first_enabled_elmt > 0) && (pos < mask.first_enabled_elmt))
          pos = mask.first_enabled_elmt;

	// fetch first value and see if we have any bits set
	int idx = pos >> 6;
	uint64_t bits = impl->bits[idx];
	if(!polarity) bits = ~bits;

	// for the first one, we may have bits to ignore at the start
	if(pos & 0x3f)
	  bits &= ~((1ULL << (pos & 0x3f)) - 1);

	// skip over words that are all zeros, and try to ignore trailing zeros completely
        int stop_at = mask.num_elements;
        if(mask.last_enabled_elmt >= 0)
          stop_at = mask.last_enabled_elmt+1;
	while(!bits) {
	  idx++;
	  if((idx << 6) >= stop_at) {
	    pos = mask.num_elements; // so we don't scan again
	    return false;
	  }
	  bits = impl->bits[idx];
	  if(!polarity) bits = ~bits;
	}

	// if we get here, we've got at least one good bit
	int extra = __builtin_ctzll(bits);
	assert(extra < 64);
	position = (idx << 6) + extra;
	
	// now we're going to turn it around and scan ones
	if(extra)
	  bits |= ((1ULL << extra) - 1);
	bits = ~bits;

	while(!bits) {
	  idx++;
	  // did our 1's take us right to the end?
	  if((idx << 6) >= mask.num_elements) {
	    pos = mask.num_elements; // so we don't scan again
	    length = mask.num_elements - position;
	    return true;
	  }
	  bits = ~impl->bits[idx]; // note the inversion
	  if(!polarity) bits = ~bits;
	}

	// if we get here, we got to the end of the 1's
	int extra2 = __builtin_ctzll(bits);
	pos = (idx << 6) + extra2;
	if(pos >= mask.num_elements)
	  pos = mask.num_elements;
	length = pos - position;
	return true;
      } else {
	assert(0);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(mask.memory);

	// scan until we find a bit set with the right polarity
	while(pos < mask.num_elements) {
	  off_t ofs = mask.offset + ((pos >> 5) << 2);
	  unsigned val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  int bit = ((val >> (pos & 0x1f))) & 1;
	  if(bit != polarity) {
	    pos++;
	    continue;
	  }

	  // ok, found one bit with the right polarity - now see how many
	  //  we have in a row
	  position = pos++;
	  while(pos < mask.num_elements) {
	    off_t ofs = mask.offset + ((pos >> 5) << 2);
	    unsigned val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    int bit = ((val >> (pos & 0x1f))) & 1;
	    if(bit != polarity) break;
	    pos++;
	  }
	  // we get here either because we found the end of the run or we 
	  //  hit the end of the mask
	  length = pos - position;
	  return true;
	}

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      }
    }

    bool ElementMask::Enumerator::peek_next(int &position, int &length)
    {
      int old_pos = pos;
      bool ret = get_next(position, length);
      pos = old_pos;
      return ret;
    }

    ///////////////////////////////////////////////////
    // Region Allocators

    unsigned IndexSpaceAllocator::alloc(unsigned count /*= 1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return ((IndexSpaceAllocatorImpl *)impl)->alloc_elements(count);
    }

    void IndexSpaceAllocator::reserve(unsigned ptr, unsigned count /*= 1  */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return ((IndexSpaceAllocatorImpl *)impl)->reserve_elements(ptr, count);
    }

    void IndexSpaceAllocator::free(unsigned ptr, unsigned count /*= 1  */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return ((IndexSpaceAllocatorImpl *)impl)->free_elements(ptr, count);
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    IndexSpaceAllocatorImpl::IndexSpaceAllocatorImpl(IndexSpaceImpl *_is_impl)
      : is_impl(_is_impl)
    {
    }

    IndexSpaceAllocatorImpl::~IndexSpaceAllocatorImpl(void)
    {
    }

    unsigned IndexSpaceAllocatorImpl::alloc_elements(unsigned count /*= 1 */)
    {
      SharedAccess<IndexSpaceImpl> is_data(is_impl);
      assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
      int start = is_impl->valid_mask->find_disabled(count);
      assert(start >= 0);

      reserve_elements(start, count);

      return start;
    }

    void IndexSpaceAllocatorImpl::reserve_elements(unsigned ptr, unsigned count /*= 1 */)
    {
      // for now, do updates of valid masks immediately
      IndexSpaceImpl *impl = is_impl;
      while(1) {
	SharedAccess<IndexSpaceImpl> is_data(impl);
	assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
	is_impl->valid_mask->enable(ptr, count);
	IndexSpace is = is_data->parent;
	if(is == IndexSpace::NO_SPACE) break;
	impl = get_runtime()->get_index_space_impl(is);
      }
    }

    void IndexSpaceAllocatorImpl::free_elements(unsigned ptr, unsigned count /*= 1*/)
    {
      // for now, do updates of valid masks immediately
      IndexSpaceImpl *impl = is_impl;
      while(1) {
	SharedAccess<IndexSpaceImpl> is_data(impl);
	assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
	is_impl->valid_mask->disable(ptr, count);
	IndexSpace is = is_data->parent;
	if(is == IndexSpace::NO_SPACE) break;
	impl = get_runtime()->get_index_space_impl(is);
      }
    }

    ///////////////////////////////////////////////////
    // Region Instances

#ifdef POINTER_CHECKS
    void RegionInstanceImpl::verify_access(unsigned ptr)
    {
      StaticAccess<RegionInstanceImpl> data(this);
      const ElementMask &mask = data->is.get_valid_mask();
      if (!mask.is_set(ptr))
      {
        fprintf(stderr,"ERROR: Accessing invalid pointer %d in logical region " IDFMT "\n",ptr,data->is.id);
        assert(false);
      }
    }
#endif

    void RegionInstanceImpl::get_bytes(int index, off_t byte_offset, void *dst, size_t size)
    {
      // must have valid data by now - block if we have to
      metadata.await_data();
      off_t o;
      if(metadata.block_size == 1) {
	// no blocking - don't need to know about field boundaries
	o = metadata.alloc_offset + (index * metadata.elmt_size) + byte_offset;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(metadata.field_sizes, byte_offset, size, field_start, field_size);
        o = calc_mem_loc(metadata.alloc_offset, field_start, field_size,
                         metadata.elmt_size, metadata.block_size, index);

      }
      MemoryImpl *m = get_runtime()->get_memory_impl(memory);
      m->get_bytes(o, dst, size);
    }

    void RegionInstanceImpl::put_bytes(int index, off_t byte_offset, const void *src, size_t size)
    {
      // must have valid data by now - block if we have to
      metadata.await_data();
      off_t o;
      if(metadata.block_size == 1) {
	// no blocking - don't need to know about field boundaries
	o = metadata.alloc_offset + (index * metadata.elmt_size) + byte_offset;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(metadata.field_sizes, byte_offset, size, field_start, field_size);
        o = calc_mem_loc(metadata.alloc_offset, field_start, field_size,
                         metadata.elmt_size, metadata.block_size, index);
      }
      MemoryImpl *m = get_runtime()->get_memory_impl(memory);
      m->put_bytes(o, src, size);
    }

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

    struct RemoteRedListArgs : public BaseMedium {
      Memory mem;
      off_t offset;
      ReductionOpID redopid;
    };

    void handle_remote_redlist(RemoteRedListArgs args,
			       const void *data, size_t datalen)
    {
      MemoryImpl *impl = get_runtime()->get_memory_impl(args.mem);

      log_copy.debug("received remote reduction list request: mem=" IDFMT ", offset=%zd, size=%zd, redopid=%d",
		     args.mem.id, args.offset, datalen, args.redopid);

      switch(impl->kind) {
      case MemoryImpl::MKIND_SYSMEM:
      case MemoryImpl::MKIND_ZEROCOPY:
#ifdef USE_CUDA
      case MemoryImpl::MKIND_GPUFB:
#endif
      default:
	assert(0);

      case MemoryImpl::MKIND_GLOBAL:
	{
	  const ReductionOpUntyped *redop = get_runtime()->reduce_op_table[args.redopid];
	  assert((datalen % redop->sizeof_list_entry) == 0);
	  impl->apply_reduction_list(args.offset,
				     redop,
				     datalen / redop->sizeof_list_entry,
				     data);
	}
      }
    }

    typedef ActiveMessageMediumNoReply<REMOTE_REDLIST_MSGID,
				       RemoteRedListArgs,
				       handle_remote_redlist> RemoteRedListMessage;

    void do_remote_apply_red_list(int node, Memory mem, off_t offset,
				  ReductionOpID redopid,
				  const void *data, size_t datalen)
    {
      RemoteRedListArgs args;
      args.mem = mem;
      args.offset = offset;
      args.redopid = redopid;
      RemoteRedListMessage::request(node, args,
				    data, datalen, PAYLOAD_COPY);
    }

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
      IndexSpace src_region = StaticAccess<RegionInstanceImpl>(src_impl)->region;
      IndexSpace dst_region = StaticAccess<RegionInstanceImpl>(dst_impl)->region;

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
      IndexSpace src_region = StaticAccess<RegionInstanceImpl>(src_impl)->region;
      IndexSpace dst_region = StaticAccess<RegionInstanceImpl>(dst_impl)->region;

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
	StaticAccessf<RegionInstanceImpl> src_data(src_impl);
	bytes_to_copy = src_data->region.impl()->instance_size();
	elmt_size = (src_data->is_reduction ?
		       get_runtime()->reduce_op_table[src_data->redopid]->sizeof_rhs :
		       StaticAccess<IndexSpaceImpl>(src_data->region.impl())->elmt_size);
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
      StaticAccess<RegionInstanceImpl> i_data(i_impl);
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
      StaticAccess<RegionInstanceImpl> i_data(i_impl);
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

      StaticAccess<RegionInstanceImpl> i_data(i_impl);

      assert(!i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == MemoryImpl::MKIND_SYSMEM) {
	LocalCPUMemory *lcm = (LocalCPUMemory *)m_impl;
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
      StaticAccess<RegionInstanceImpl> i_data(i_impl);
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

      StaticAccess<RegionInstanceImpl> i_data(i_impl);

      assert(i_data->is_reduction);
      assert(i_data->red_list_size < 0);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == MemoryImpl::MKIND_SYSMEM) {
	LocalCPUMemory *lcm = (LocalCPUMemory *)m_impl;
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
      StaticAccess<RegionInstanceImpl> i_data(i_impl);
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

      StaticAccess<RegionInstanceImpl> i_data(i_impl);

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

      StaticAccess<RegionInstanceImpl> i_data(i_impl);

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

    ///////////////////////////////////////////////////
    // 
    struct MachineShutdownRequestArgs {
      int initiating_node;
      int dummy; // needed to get sizeof() >= 8
    };

    void handle_machine_shutdown_request(MachineShutdownRequestArgs args)
    {
      log_machine.info("received shutdown request from node %d", args.initiating_node);

      get_runtime()->shutdown(false);
    }

    typedef ActiveMessageShortNoReply<MACHINE_SHUTDOWN_MSGID,
				      MachineShutdownRequestArgs,
				      handle_machine_shutdown_request> MachineShutdownRequestMessage;

    static GASNetHSL announcement_mutex;
    static unsigned announcements_received = 0;

    enum {
      NODE_ANNOUNCE_DONE = 0,
      NODE_ANNOUNCE_PROC, // PROC id kind
      NODE_ANNOUNCE_MEM,  // MEM id size
      NODE_ANNOUNCE_PMA,  // PMA proc_id mem_id bw latency
      NODE_ANNOUNCE_MMA,  // MMA mem1_id mem2_id bw latency
    };

    Logger::Category log_annc("announce");

    struct NodeAnnounceData : public BaseMedium {
      gasnet_node_t node_id;
      unsigned num_procs;
      unsigned num_memories;
    };

    void node_announce_handler(NodeAnnounceData annc_data, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_annc.info("%d: received announce from %d (%d procs, %d memories)\n", gasnet_mynode(), annc_data.node_id, annc_data.num_procs, annc_data.num_memories);
      Node *n = &(get_runtime()->nodes[annc_data.node_id]);
      n->processors.resize(annc_data.num_procs);
      n->memories.resize(annc_data.num_memories);

      // do the parsing of this data inside a mutex because it touches common
      //  data structures
      {
	AutoHSLLock al(announcement_mutex);

	get_machine()->parse_node_announce_data(data, datalen,
						annc_data, true);

	announcements_received++;
      }
    }

    typedef ActiveMessageMediumNoReply<NODE_ANNOUNCE_MSGID,
				       NodeAnnounceData,
				       node_announce_handler> NodeAnnounceMessage;

    static std::vector<ProcessorImpl*> local_cpus;
    static std::vector<ProcessorImpl*> local_util_procs;
    static std::vector<ProcessorImpl*> local_io_procs;
    static size_t stack_size_in_mb;
#ifdef USE_CUDA
    static std::vector<GPUProcessor *> local_gpus;
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

    /*static*/ Machine Machine::get_machine(void) 
    {
      return Machine(get_runtime()->machine);
    }

    size_t Machine::get_address_space_count(void) const
    {
      return gasnet_nodes();
    }

    void Machine::get_all_memories(std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_all_memories(mset);
    }
    
    void Machine::get_all_processors(std::set<Processor>& pset) const
    {
      return ((MachineImpl *)impl)->get_all_processors(pset);
    }

    // Return the set of memories visible from a processor
    void Machine::get_visible_memories(Processor p, std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(p, mset);
    }

    // Return the set of memories visible from a memory
    void Machine::get_visible_memories(Memory m, std::set<Memory>& mset) const
    {
      return ((MachineImpl *)impl)->get_visible_memories(m, mset);
    }

    // Return the set of processors which can all see a given memory
    void Machine::get_shared_processors(Memory m, std::set<Processor>& pset) const
    {
      return ((MachineImpl *)impl)->get_shared_processors(m, pset);
    }

    int Machine::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				       Processor restrict_proc /*= Processor::NO_PROC*/,
				       Memory restrict_memory /*= Memory::NO_MEMORY*/) const
    {
      return ((MachineImpl *)impl)->get_proc_mem_affinity(result, restrict_proc, restrict_memory);
    }

    int Machine::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
				      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
				      Memory restrict_mem2 /*= Memory::NO_MEMORY*/) const
    {
      return ((MachineImpl *)impl)->get_mem_mem_affinity(result, restrict_mem1, restrict_mem2);
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    ProcessorAssignment::ProcessorAssignment(int _num_local_procs)
      : num_local_procs(_num_local_procs)
    {
      valid = false;

#ifdef __MACH__
      //printf("thread affinity not supported on Mac OS X\n");
      return;
#else
      cpu_set_t cset;
      int ret = sched_getaffinity(0, sizeof(cset), &cset);
      if(ret < 0) {
	printf("failed to get affinity info - binding disabled\n");
	return;
      }

      SystemProcMap proc_map;
      {
	DIR *nd = opendir("/sys/devices/system/node");
	if(!nd) {
	  printf("can't open /sys/devices/system/node - binding disabled\n");
	  return;
	}
	for(struct dirent *ne = readdir(nd); ne; ne = readdir(nd)) {
	  if(strncmp(ne->d_name, "node", 4)) continue;  // not a node directory
	  int node_id = atoi(ne->d_name + 4);
	  
	  char per_node_path[1024];
	  sprintf(per_node_path, "/sys/devices/system/node/%s", ne->d_name);
	  DIR *cd = opendir(per_node_path);
	  if(!cd) {
	    printf("can't open %s - skipping\n", per_node_path);
	    continue;
	  }

	  for(struct dirent *ce = readdir(cd); ce; ce = readdir(cd)) {
	    if(strncmp(ce->d_name, "cpu", 3)) continue; // definitely not a cpu
	    char *pos;
	    int cpu_id = strtol(ce->d_name + 3, &pos, 10);
	    if(pos && *pos) continue;  // doesn't match cpu[0-9]+
	    
	    // is this a cpu we're allowed to use?
	    if(!CPU_ISSET(cpu_id, &cset)) {
	      printf("cpu %d not available - skipping\n", cpu_id);
	      continue;
	    }

	    // figure out which physical core it is
	    char core_id_path[1024];
	    sprintf(core_id_path, "/sys/devices/system/node/%s/%s/topology/core_id", ne->d_name, ce->d_name);
	    FILE *f = fopen(core_id_path, "r");
	    if(!f) {
	      printf("can't read %s - skipping\n", core_id_path);
	      continue;
	    }
	    int core_id;
	    int count = fscanf(f, "%d", &core_id);
	    fclose(f);
	    if(count != 1) {
	      printf("can't find core id in %s - skipping\n", core_id_path);
	      continue;
	    }
	    
	    //printf("found: %d %d %d\n", node_id, cpu_id, core_id);
	    proc_map[node_id][core_id].push_back(cpu_id);
	  }
	  closedir(cd);
	}
	closedir(nd);
      }
      
#if 0
      printf("Available cores:\n");
      for(SystemProcMap::const_iterator it1 = proc_map.begin(); it1 != proc_map.end(); it1++) {
	printf("  Node %d:", it1->first);
	for(NodeProcMap::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
	  if(it2->second.size() == 1) {
	    printf(" %d", it2->second[0]);
	  } else {
	    printf(" {");
	    for(size_t i = 0; i < it2->second.size(); i++)
	      printf("%s%d", (i ? " " : ""), it2->second[i]);
	    printf("}");
	  }
	}
	printf("\n");
      }
#endif

      // count how many actual cores we have
      int core_count = 0;
      for(SystemProcMap::const_iterator it1 = proc_map.begin(); it1 != proc_map.end(); it1++)
	core_count += it1->second.size();
      
      if(core_count <= num_local_procs) {
	//printf("not enough cores (%zd) to support %d local processors - skipping binding\n", core_count, num_local_procs);
	return;
      }
      
      // pick cores for each local proc - try to round-robin across nodes
      SystemProcMap::iterator curnode = proc_map.end();
      memcpy(&leftover_procs, &cset, sizeof(cset));  // subtract from cset to get leftovers
      for(int i = 0; i < num_local_procs; i++) {
	// pick the next node with any cores left
	do {
	  if(curnode != proc_map.end())
	    curnode++;
	  if(curnode == proc_map.end())
	    curnode = proc_map.begin();
	} while(curnode->second.size() == 0);
	
	NodeProcMap::iterator curcore = curnode->second.begin();
	assert(curcore != curnode->second.end());
	assert(curcore->second.size() > 0);
	
	// take the first cpu id for this core and add it to the local proc assignments
	local_proc_assignments.push_back(curcore->second[0]);
	
	// and remove ALL cpu ids for this core from the leftover set
	for(std::vector<int>::const_iterator it = curcore->second.begin(); it != curcore->second.end(); it++)
	  CPU_CLR(*it, &leftover_procs);
	
	// and now remove this core from the node's list of available cores
	curnode->second.erase(curcore);
      }

      // we now have a valid set of bindings
      valid = true;

      // set the process' default affinity to just the leftover nodes
      bool override_default_affinity = false;
      if(override_default_affinity) {
	int ret = sched_setaffinity(0, sizeof(leftover_procs), &leftover_procs);
	if(ret < 0) {
	  printf("failed to set default affinity info!\n");
	}
      }
	
#if 0
      {
	printf("Local Proc Assignments:");
	for(std::vector<int>::const_iterator it = local_proc_assignments.begin(); it != local_proc_assignments.end(); it++)
	  printf(" %d", *it);
	printf("\n");
	
	printf("Leftover Processors   :");
	for(int i = 0; i < CPU_SETSIZE; i++)
	  if(CPU_ISSET(i, &leftover_procs))
	    printf(" %d", i);
	printf("\n");
      }
#endif
#endif
    }

    // binds a thread to the right set of cores based (-1 = not a local proc)
    void ProcessorAssignment::bind_thread(int core_id, pthread_attr_t *attr, const char *debug_name /*= 0*/)
    {
      if(!valid) {
	//printf("no processor assignment for %s %d (%p)\n", debug_name ? debug_name : "unknown", core_id, attr);
	return;
      }

#ifndef __MACH__
      if((core_id >= 0) && (core_id < num_local_procs)) {
	int cpu_id = local_proc_assignments[core_id];

	//printf("processor assignment for %s %d (%p) = %d\n", debug_name ? debug_name : "unknown", core_id, attr, cpu_id);

	cpu_set_t cset;
	CPU_ZERO(&cset);
	CPU_SET(cpu_id, &cset);
	if(attr)
	  CHECK_PTHREAD( pthread_attr_setaffinity_np(attr, sizeof(cset), &cset) );
	else
	  CHECK_PTHREAD( pthread_setaffinity_np(pthread_self(), sizeof(cset), &cset) );
      } else {
	//printf("processor assignment for %s %d (%p) = leftovers\n", debug_name ? debug_name : "unknown", core_id, attr);

	if(attr)
	  CHECK_PTHREAD( pthread_attr_setaffinity_np(attr, sizeof(leftover_procs), &leftover_procs) );
	else
	  CHECK_PTHREAD( pthread_setaffinity_np(pthread_self(), sizeof(leftover_procs), &leftover_procs) );
      }
#endif
    }

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

#if defined(REALM_BACKTRACE) || defined(LEGION_BACKTRACE)
    static void realm_backtrace(int signal)
    {
      assert((signal == SIGILL) || (signal == SIGFPE) || 
             (signal == SIGABRT) || (signal == SIGSEGV) ||
             (signal == SIGBUS));
      void *bt[256];
      int bt_size = backtrace(bt, 256);
      char **bt_syms = backtrace_symbols(bt, bt_size);
      size_t buffer_size = 2048; // default buffer size
      char *buffer = (char*)malloc(buffer_size);
      size_t offset = 0;
      size_t funcnamesize = 256;
      char *funcname = (char*)malloc(funcnamesize);
      for (int i = 0; i < bt_size; i++) {
        // Modified from https://panthema.net/2008/0901-stacktrace-demangled/ under WTFPL 2.0
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;
        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = bt_syms[i]; *p; ++p) {
          if (*p == '(')
            begin_name = p;
          else if (*p == '+')
            begin_offset = p;
          else if (*p == ')' && begin_offset) {
            end_offset = p;
            break;
          }
        }
        // If offset is within half of the buffer size, double the buffer
        if (offset >= (buffer_size / 2)) {
          buffer_size *= 2;
          buffer = (char*)realloc(buffer, buffer_size);
        }
        if (begin_name && begin_offset && end_offset &&
            (begin_name < begin_offset)) {
          *begin_name++ = '\0';
          *begin_offset++ = '\0';
          *end_offset = '\0';
          // mangled name is now in [begin_name, begin_offset) and caller
          // offset in [begin_offset, end_offset). now apply __cxa_demangle():
          int status;
          char* demangled_name = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
          if (status == 0) {
            funcname = demangled_name; // use possibly realloc()-ed string
            offset += snprintf(buffer+offset,buffer_size-offset,
                               "  %s : %s+%s\n", bt_syms[i], funcname, begin_offset);
          } else {
            // demangling failed. Output function name as a C function with no arguments.
            offset += snprintf(buffer+offset,buffer_size-offset,
                               "  %s : %s()+%s\n", bt_syms[i], begin_name, begin_offset);
          }
        } else {
          // Who knows just print the whole line
          offset += snprintf(buffer+offset,buffer_size-offset,
                             "%s\n",bt_syms[i]);
        }
      }
      fprintf(stderr,"BACKTRACE (%d, %lx)\n----------\n%s\n----------\n", 
              gasnet_mynode(), pthread_self(), buffer);
      fflush(stderr);
      free(buffer);
      free(funcname);
      // returning would almost certainly cause this signal to be raised again,
      //  so sleep for a second in case other threads also want to chronicle
      //  their own deaths, and then exit
      sleep(1);
      exit(1);
    }
#endif

    static void realm_freeze(int signal)
    {
      assert((signal == SIGINT) || (signal == SIGABRT) ||
             (signal == SIGSEGV) || (signal == SIGFPE) ||
             (signal == SIGBUS));
      int process_id = getpid();
      char hostname[128];
      gethostname(hostname, 127);
      fprintf(stderr,"Legion process received signal %d: %s\n",
                      signal, strsignal(signal));
      fprintf(stderr,"Process %d on node %s is frozen!\n", 
                      process_id, hostname);
      fflush(stderr);
      while (true)
        sleep(1);
    }

    ProcessorAssignment *proc_assignment = 0;

    void MachineImpl::parse_node_announce_data(const void *args, size_t arglen,
						 const NodeAnnounceData& annc_data,
						 bool remote)
    {
      const size_t *cur = (const size_t *)args;
      const size_t *limit = (const size_t *)(((const char *)args)+arglen);

      while(1) {
	assert(cur < limit);
	if(*cur == NODE_ANNOUNCE_DONE) break;
	switch(*cur++) {
	case NODE_ANNOUNCE_PROC:
	  {
	    ID id((IDType)*cur++);
	    Processor p = id.convert<Processor>();
	    assert(id.index() < annc_data.num_procs);
	    Processor::Kind kind = (Processor::Kind)(*cur++);
	    if(remote) {
	      RemoteProcessor *proc = new RemoteProcessor(p, kind);
	      get_runtime()->nodes[ID(p).node()].processors[ID(p).index()] = proc;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_MEM:
	  {
	    ID id((IDType)*cur++);
	    Memory m = id.convert<Memory>();
	    assert(id.index_h() < annc_data.num_memories);
            Memory::Kind kind = (Memory::Kind)(*cur++);
	    unsigned size = *cur++;
	    void *regbase = (void *)(*cur++);
	    if(remote) {
	      RemoteMemory *mem = new RemoteMemory(m, size, kind, regbase);
	      get_runtime()->nodes[ID(m).node()].memories[ID(m).index_h()] = mem;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_PMA:
	  {
	    Machine::ProcessorMemoryAffinity pma;
	    pma.p = ID((IDType)*cur++).convert<Processor>();
	    pma.m = ID((IDType)*cur++).convert<Memory>();
	    pma.bandwidth = *cur++;
	    pma.latency = *cur++;

	    proc_mem_affinities.push_back(pma);
	  }
	  break;

	case NODE_ANNOUNCE_MMA:
	  {
	    Machine::MemoryMemoryAffinity mma;
	    mma.m1 = ID((IDType)*cur++).convert<Memory>();
	    mma.m2 = ID((IDType)*cur++).convert<Memory>();
	    mma.bandwidth = *cur++;
	    mma.latency = *cur++;

	    mem_mem_affinities.push_back(mma);
	  }
	  break;

	default:
	  assert(0);
	}
      }
    }

    void MachineImpl::get_all_memories(std::set<Memory>& mset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	mset.insert((*it).m);
      }
    }

    void MachineImpl::get_all_processors(std::set<Processor>& pset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	pset.insert((*it).p);
      }
    }

    // Return the set of memories visible from a processor
    void MachineImpl::get_visible_memories(Processor p, std::set<Memory>& mset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if((*it).p == p)
	  mset.insert((*it).m);
      }
    }

    // Return the set of memories visible from a memory
    void MachineImpl::get_visible_memories(Memory m, std::set<Memory>& mset) const
    {
      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if((*it).m1 == m)
	  mset.insert((*it).m2);

	if((*it).m2 == m)
	  mset.insert((*it).m1);
      }
    }

    // Return the set of processors which can all see a given memory
    void MachineImpl::get_shared_processors(Memory m, std::set<Processor>& pset) const
    {
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if((*it).m == m)
	  pset.insert((*it).p);
      }
    }

    int MachineImpl::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
					     Processor restrict_proc /*= Processor::NO_PROC*/,
					     Memory restrict_memory /*= Memory::NO_MEMORY*/) const
    {
      int count = 0;

      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if(restrict_proc.exists() && ((*it).p != restrict_proc)) continue;
	if(restrict_memory.exists() && ((*it).m != restrict_memory)) continue;
	result.push_back(*it);
	count++;
      }

      return count;
    }

    int MachineImpl::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
					    Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
					    Memory restrict_mem2 /*= Memory::NO_MEMORY*/) const
    {
      // Handle the case for same memories
      if (restrict_mem1.exists() && (restrict_mem1 == restrict_mem2))
      {
	Machine::MemoryMemoryAffinity affinity;
        affinity.m1 = restrict_mem1;
        affinity.m2 = restrict_mem1;
        affinity.bandwidth = 100;
        affinity.latency = 1;
        result.push_back(affinity);
        return 1;
      }

      int count = 0;

      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if(restrict_mem1.exists() && 
	   ((*it).m1 != restrict_mem1)) continue;
	if(restrict_mem2.exists() && 
	   ((*it).m2 != restrict_mem2)) continue;
	result.push_back(*it);
	count++;
      }

      return count;
    }

  };
};

namespace Realm {

    Runtime::Runtime(void)
      : impl(0)
    {
      // ok to construct extra ones - we will make sure only one calls init() though
    }

    /*static*/ Runtime Runtime::get_runtime(void)
    {
      Runtime r;
      r.impl = LegionRuntime::LowLevel::get_runtime();
      return r;
    }

    bool Runtime::init(int *argc, char ***argv)
    {
      if(runtime_singleton != 0) {
	fprintf(stderr, "ERROR: cannot initialize more than one runtime at a time!\n");
	return false;
      }

      impl = new RuntimeImpl;
      runtime_singleton = ((RuntimeImpl *)impl);
      return ((RuntimeImpl *)impl)->init(argc, argv);
    }
    
    bool Runtime::register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->task_table.count(taskid) > 0)
	return false;

      ((RuntimeImpl *)impl)->task_table[taskid] = taskptr;
      return true;
    }

    bool Runtime::register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->reduce_op_table.count(redop_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->reduce_op_table[redop_id] = redop;
      return true;
    }

    void Runtime::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/,
                      bool background /*= false*/)
    {
      ((RuntimeImpl *)impl)->run(task_id, style, args, arglen, background);
    }

    void Runtime::shutdown(void)
    {
      ((RuntimeImpl *)impl)->shutdown(true); // local request
    }

    void Runtime::wait_for_shutdown(void)
    {
      ((RuntimeImpl *)impl)->wait_for_shutdown();

      // after the shutdown, we nuke the RuntimeImpl
      delete ((RuntimeImpl *)impl);
      impl = 0;
      runtime_singleton = 0;
    }

};

namespace LegionRuntime {
  namespace LowLevel {

    RuntimeImpl::RuntimeImpl(void)
      : machine(0), nodes(0), global_memory(0),
	local_event_free_list(0), local_barrier_free_list(0),
	local_reservation_free_list(0), local_index_space_free_list(0),
	local_proc_group_free_list(0), background_pthread(0)
    {
      assert(machine_singleton == 0);

      machine = new MachineImpl;
      machine_singleton = ((MachineImpl *)machine);
    }

    RuntimeImpl::~RuntimeImpl(void)
    {
      machine_singleton = 0;
      delete machine;
    }

    bool RuntimeImpl::init(int *argc, char ***argv)
    {
      // have to register domain mappings too
      Arrays::Mapping<1,1>::register_mapping<Arrays::CArrayLinearization<1> >();
      Arrays::Mapping<2,1>::register_mapping<Arrays::CArrayLinearization<2> >();
      Arrays::Mapping<3,1>::register_mapping<Arrays::CArrayLinearization<3> >();
      Arrays::Mapping<1,1>::register_mapping<Arrays::FortranArrayLinearization<1> >();
      Arrays::Mapping<2,1>::register_mapping<Arrays::FortranArrayLinearization<2> >();
      Arrays::Mapping<3,1>::register_mapping<Arrays::FortranArrayLinearization<3> >();
      Arrays::Mapping<1,1>::register_mapping<Translation<1> >();

      // we also register split dim linearization
      Arrays::Mapping<1,1>::register_mapping<Layouts::SplitDimLinearization<1> >();
      Arrays::Mapping<2,1>::register_mapping<Layouts::SplitDimLinearization<2> >();
      Arrays::Mapping<3,1>::register_mapping<Layouts::SplitDimLinearization<3> >();

      // Create the key for the thread local data
      CHECK_PTHREAD( pthread_key_create(&thread_timer_key,thread_timer_free) );

      // low-level runtime parameters
#ifdef USE_GASNET
      size_t gasnet_mem_size_in_mb = 256;
#else
      size_t gasnet_mem_size_in_mb = 0;
#endif
      size_t cpu_mem_size_in_mb = 512;
      size_t reg_mem_size_in_mb = 0;
      size_t disk_mem_size_in_mb = 0;
      // Static variable for stack size since we need to 
      // remember it when we launch threads in run 
      stack_size_in_mb = 2;
      unsigned init_stack_count = 1;
      unsigned num_local_cpus = 1;
      unsigned num_util_procs = 1;
      unsigned num_io_procs = 0;
      //unsigned cpu_worker_threads = 1;
      unsigned dma_worker_threads = 1;
      unsigned active_msg_worker_threads = 1;
      unsigned active_msg_handler_threads = 1;
      bool     active_msg_sender_threads = false;
#ifdef USE_CUDA
      size_t zc_mem_size_in_mb = 64;
      size_t fb_mem_size_in_mb = 256;
      unsigned num_local_gpus = 0;
      unsigned num_gpu_streams = 12;
      bool     gpu_worker_thread = true;
      bool pin_sysmem_for_gpu = true;
#endif
#ifdef EVENT_TRACING
      size_t   event_trace_block_size = 1 << 20;
      double   event_trace_exp_arrv_rate = 1e3;
#endif
#ifdef LOCK_TRACING
      size_t   lock_trace_block_size = 1 << 20;
      double   lock_trace_exp_arrv_rate = 1e2;
#endif
      // should local proc threads get dedicated cores?
      bool bind_localproc_threads = true;
      bool use_greenlet_procs = true;
      bool disable_greenlets = false;

      for(int i = 1; i < *argc; i++) {
#define INT_ARG(argname, varname)                       \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = atoi((*argv)[++i]);		\
	    continue;					\
	  }

#define BOOL_ARG(argname, varname)                      \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = true;				\
	    continue;					\
	  }

	INT_ARG("-ll:gsize", gasnet_mem_size_in_mb);
	INT_ARG("-ll:csize", cpu_mem_size_in_mb);
	INT_ARG("-ll:rsize", reg_mem_size_in_mb);
        INT_ARG("-ll:dsize", disk_mem_size_in_mb);
        INT_ARG("-ll:stacksize", stack_size_in_mb);
        INT_ARG("-ll:stacks", init_stack_count);
	INT_ARG("-ll:cpu", num_local_cpus);
	INT_ARG("-ll:util", num_util_procs);
        INT_ARG("-ll:io", num_io_procs);
	//INT_ARG("-ll:workers", cpu_worker_threads);
	INT_ARG("-ll:dma", dma_worker_threads);
	INT_ARG("-ll:amsg", active_msg_worker_threads);
	INT_ARG("-ll:ahandlers", active_msg_handler_threads);
        BOOL_ARG("-ll:senders", active_msg_sender_threads);
	INT_ARG("-ll:bind", bind_localproc_threads);
        BOOL_ARG("-ll:greenlet", use_greenlet_procs);
        BOOL_ARG("-ll:gdb", disable_greenlets);
#ifdef USE_CUDA
	INT_ARG("-ll:fsize", fb_mem_size_in_mb);
	INT_ARG("-ll:zsize", zc_mem_size_in_mb);
	INT_ARG("-ll:gpu", num_local_gpus);
        INT_ARG("-ll:streams", num_gpu_streams);
        BOOL_ARG("-ll:gpuworker", gpu_worker_thread);
        INT_ARG("-ll:pin", pin_sysmem_for_gpu);
#endif

	if(!strcmp((*argv)[i], "-ll:eventtrace")) {
#ifdef EVENT_TRACING
	  event_trace_file = strdup((*argv)[++i]);
#else
	  fprintf(stderr, "WARNING: event tracing requested, but not enabled at compile time!\n");
#endif
	  continue;
	}

        if (!strcmp((*argv)[i], "-ll:locktrace"))
        {
#ifdef LOCK_TRACING
          lock_trace_file = strdup((*argv)[++i]);
#else
          fprintf(stderr, "WARNING: lock tracing requested, but not enabled at compile time!\n");
#endif
          continue;
        }

        if (!strcmp((*argv)[i], "-ll:prefix"))
        {
#ifdef NODE_LOGGING
          RuntimeImpl::prefix = strdup((*argv)[++i]);
#else
          fprintf(stderr,"WARNING: prefix set, but NODE_LOGGING not enabled at compile time!\n");
#endif
          continue;
        }

        // Skip arguments that parsed in activemsg.cc
        if (!strcmp((*argv)[i], "-ll:numlmbs") || !strcmp((*argv)[i],"-ll:lmbsize") ||
            !strcmp((*argv)[i], "-ll:forcelong") || !strcmp((*argv)[i],"-ll:sdpsize"))
        {
          i++;
          continue;
        }

        if (strncmp((*argv)[i], "-ll:", 4) == 0)
        {
	  fprintf(stderr, "ERROR: unrecognized lowlevel option: %s\n", (*argv)[i]);
          assert(0);
	}
      }

      if(bind_localproc_threads) {
	// this has to preceed all spawning of threads, including the ones done by things like gasnet_init()
	proc_assignment = new ProcessorAssignment(num_local_cpus);

	// now move ourselves off the reserved cores
	proc_assignment->bind_thread(-1, 0, "machine thread");
      }

      if (disable_greenlets)
        use_greenlet_procs = false;
      if (use_greenlet_procs)
        greenlet::init_greenlet_library();

      //GASNetNode::my_node = new GASNetNode(argc, argv, this);
      // SJT: WAR for issue on Titan with duplicate cookies on Gemini
      //  communication domains
      char *orig_pmi_gni_cookie = getenv("PMI_GNI_COOKIE");
      if(orig_pmi_gni_cookie) {
        char *new_pmi_gni_cookie = (char *)malloc(256);
        sprintf(new_pmi_gni_cookie, "PMI_GNI_COOKIE=%d", 1+atoi(orig_pmi_gni_cookie));
        //printf("changing PMI cookie to: '%s'\n", new_pmi_gni_cookie);
        putenv(new_pmi_gni_cookie);  // libc now owns the memory
      }
      // SJT: another GASNET workaround - if we don't have GASNET_IB_SPAWNER set, assume it was MPI
      if(!getenv("GASNET_IB_SPAWNER"))
	putenv(strdup("GASNET_IB_SPAWNER=mpi"));
#ifdef DEBUG_REALM_STARTUP
      { // we don't have rank IDs yet, so everybody gets to spew
        char s[80];
        gethostname(s, 79);
        strcat(s, " enter gasnet_init");
        LegionRuntime::TimeStamp ts(s, false);
        fflush(stdout);
      }
#endif
      CHECK_GASNET( gasnet_init(argc, argv) );
#ifdef DEBUG_REALM_STARTUP
      { // once we're convinced there isn't skew here, reduce this to rank 0
        char s[80];
        gethostname(s, 79);
        strcat(s, " exit gasnet_init");
        LegionRuntime::TimeStamp ts(s, false);
        fflush(stdout);
      }
#endif

      // Check that we have enough resources for the number of nodes we are using
      if (gasnet_nodes() > MAX_NUM_NODES)
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but runtime is configured "
                       "for at most %d nodes. Update the 'MAX_NUM_NODES' macro "
                       "in legion_types.h", gasnet_nodes(), MAX_NUM_NODES);
        gasnet_exit(1);
      }
      if (gasnet_nodes() > (1 << ID::NODE_BITS))
      {
#ifdef LEGION_IDS_ARE_64BIT
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes. Update the allocation "
                       "of bits in ID", gasnet_nodes(), (1 << ID::NODE_BITS));
#else
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes.  Update the allocation "
                       "of bits in ID or switch to 64-bit IDs with the "
                       "-DLEGION_IDS_ARE_64BIT compile-time flag",
                       gasnet_nodes(), (1 << ID::NODE_BITS));
#endif
        gasnet_exit(1);
      }

      // initialize barrier timestamp
      barrier_adjustment_timestamp = (((Barrier::timestamp_t)(gasnet_mynode())) << BARRIER_TIMESTAMP_NODEID_SHIFT) + 1;

      Realm::Logger::configure_from_cmdline(*argc, (const char **)*argv);

      gasnet_handlerentry_t handlers[128];
      int hcount = 0;
      hcount += NodeAnnounceMessage::add_handler_entries(&handlers[hcount], "Node Announce AM");
      hcount += SpawnTaskMessage::add_handler_entries(&handlers[hcount], "Spawn Task AM");
      hcount += LockRequestMessage::add_handler_entries(&handlers[hcount], "Lock Request AM");
      hcount += LockReleaseMessage::add_handler_entries(&handlers[hcount], "Lock Release AM");
      hcount += LockGrantMessage::add_handler_entries(&handlers[hcount], "Lock Grant AM");
      hcount += EventSubscribeMessage::add_handler_entries(&handlers[hcount], "Event Subscribe AM");
      hcount += EventTriggerMessage::add_handler_entries(&handlers[hcount], "Event Trigger AM");
      hcount += RemoteMemAllocRequest::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Request AM");
      hcount += RemoteMemAllocResponse::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Response AM");
      hcount += CreateInstanceRequest::add_handler_entries(&handlers[hcount], "Create Instance Request AM");
      hcount += CreateInstanceResponse::add_handler_entries(&handlers[hcount], "Create Instance Response AM");
      hcount += RemoteCopyMessage::add_handler_entries(&handlers[hcount], "Remote Copy AM");
      hcount += RemoteFillMessage::add_handler_entries(&handlers[hcount], "Remote Fill AM");
      hcount += ValidMaskRequestMessage::add_handler_entries(&handlers[hcount], "Valid Mask Request AM");
      hcount += ValidMaskDataMessage::add_handler_entries(&handlers[hcount], "Valid Mask Data AM");
      hcount += RollUpRequestMessage::add_handler_entries(&handlers[hcount], "Roll-up Request AM");
      hcount += RollUpDataMessage::add_handler_entries(&handlers[hcount], "Roll-up Data AM");
      hcount += ClearTimerRequestMessage::add_handler_entries(&handlers[hcount], "Clear Timer Request AM");
      hcount += DestroyInstanceMessage::add_handler_entries(&handlers[hcount], "Destroy Instance AM");
      hcount += RemoteWriteMessage::add_handler_entries(&handlers[hcount], "Remote Write AM");
      hcount += RemoteReduceMessage::add_handler_entries(&handlers[hcount], "Remote Reduce AM");
      hcount += RemoteWriteFenceMessage::add_handler_entries(&handlers[hcount], "Remote Write Fence AM");
      hcount += DestroyLockMessage::add_handler_entries(&handlers[hcount], "Destroy Lock AM");
      hcount += RemoteRedListMessage::add_handler_entries(&handlers[hcount], "Remote Reduction List AM");
      hcount += MachineShutdownRequestMessage::add_handler_entries(&handlers[hcount], "Machine Shutdown AM");
      hcount += BarrierAdjustMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Adjust AM");
      hcount += BarrierSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Subscribe AM");
      hcount += BarrierTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Trigger AM");
      hcount += MetadataRequestMessage::RequestMessage::add_handler_entries(&handlers[hcount], "Metadata Request AM");
      hcount += MetadataRequestMessage::ResponseMessage::add_handler_entries(&handlers[hcount], "Metadata Response AM");
      hcount += MetadataInvalidateMessage::RequestMessage::add_handler_entries(&handlers[hcount], "Metadata Invalidate AM");
      hcount += MetadataInvalidateMessage::ResponseMessage::add_handler_entries(&handlers[hcount], "Metadata Inval Ack AM");
      //hcount += TestMessage::add_handler_entries(&handlers[hcount], "Test AM");
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount], "Test 2 AM");

      init_endpoints(handlers, hcount, 
		     gasnet_mem_size_in_mb, reg_mem_size_in_mb,
		     *argc, (const char **)*argv);

      // Put this here so that it complies with the GASNet specification and
      // doesn't make any calls between gasnet_init and gasnet_attach
      gasnet_set_waitmode(GASNET_WAIT_BLOCK);

      nodes = new Node[gasnet_nodes()];

      // create allocators for local node events/locks/index spaces
      {
	Node& n = nodes[gasnet_mynode()];
	local_event_free_list = new EventTableAllocator::FreeList(n.events, gasnet_mynode());
	local_barrier_free_list = new BarrierTableAllocator::FreeList(n.barriers, gasnet_mynode());
	local_reservation_free_list = new ReservationTableAllocator::FreeList(n.reservations, gasnet_mynode());
	local_index_space_free_list = new IndexSpaceTableAllocator::FreeList(n.index_spaces, gasnet_mynode());
	local_proc_group_free_list = new ProcessorGroupTableAllocator::FreeList(n.proc_groups, gasnet_mynode());
      }

#ifdef DEADLOCK_TRACE
      next_thread = 0;
      signaled_threads = 0;
      signal(SIGTERM, deadlock_catch);
      signal(SIGINT, deadlock_catch);
#endif
#if defined(REALM_BACKTRACE) || defined(LEGION_BACKTRACE)
      signal(SIGSEGV, realm_backtrace);
      signal(SIGABRT, realm_backtrace);
      signal(SIGFPE,  realm_backtrace);
      signal(SIGILL,  realm_backtrace);
      signal(SIGBUS,  realm_backtrace);
#endif
      if ((getenv("LEGION_FREEZE_ON_ERROR") != NULL) ||
          (getenv("REALM_FREEZE_ON_ERROR") != NULL))
      {
        signal(SIGSEGV, realm_freeze);
        signal(SIGABRT, realm_freeze);
        signal(SIGFPE,  realm_freeze);
        signal(SIGILL,  realm_freeze);
        signal(SIGBUS,  realm_freeze);
      }
      
      start_polling_threads(active_msg_worker_threads);

      start_handler_threads(active_msg_handler_threads, stack_size_in_mb << 20);

      //start_dma_worker_threads(dma_worker_threads);

      if (active_msg_sender_threads)
        start_sending_threads();

      Clock::synchronize();
      Realm::InitialTime::get_initial_time();

#ifdef EVENT_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save event info
      Tracer<EventTraceItem>::init_trace(event_trace_block_size,
                                         event_trace_exp_arrv_rate);
#endif
#ifdef LOCK_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save lock info
      Tracer<LockTraceItem>::init_trace(lock_trace_block_size,
                                        lock_trace_exp_arrv_rate);
#endif
	
      //gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
      //CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

      if(gasnet_mem_size_in_mb > 0)
	global_memory = new GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>(), gasnet_mem_size_in_mb << 20);
      else
	global_memory = 0;

      Node *n = &nodes[gasnet_mynode()];

      NodeAnnounceData announce_data;
      const unsigned ADATA_SIZE = 4096;
      size_t adata[ADATA_SIZE];
      unsigned apos = 0;

      announce_data.node_id = gasnet_mynode();
      announce_data.num_procs = num_local_cpus + num_util_procs + num_io_procs;
      announce_data.num_memories = (1 + 
				    (reg_mem_size_in_mb > 0 ? 1 : 0) +
				    (disk_mem_size_in_mb > 0 ? 1 : 0));
#ifdef USE_HDF
      announce_data.num_memories += 1;
#endif
#ifdef USE_CUDA
      announce_data.num_procs += num_local_gpus;
      announce_data.num_memories += 2 * num_local_gpus;
#endif

      // create utility processors (if any)
      if (num_util_procs > 0)
      {
        for(unsigned i = 0; i < num_util_procs; i++) {
          ProcessorImpl *up;
          if (use_greenlet_procs)
            up = new GreenletProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), 
                                    n->processors.size()).convert<Processor>(),
                                    Processor::UTIL_PROC, stack_size_in_mb << 20, 
                                    init_stack_count, "utility worker");
          else
            up = new LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), 
                                    n->processors.size()).convert<Processor>(),
                                    Processor::UTIL_PROC, 
                                    stack_size_in_mb << 20, "utility worker");
          n->processors.push_back(up);
          local_util_procs.push_back(up);
          adata[apos++] = NODE_ANNOUNCE_PROC;
          adata[apos++] = up->me.id;
          adata[apos++] = Processor::UTIL_PROC;
        }
      }
      // create i/o processors (if any)
      if (num_io_procs > 0)
      {
        for (unsigned i = 0; i < num_io_procs; i++) {
          LocalProcessor *io = new LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(),
                                            n->processors.size()).convert<Processor>(),
                                            Processor::IO_PROC,
                                            stack_size_in_mb << 20, "io worker");
          n->processors.push_back(io);
          local_io_procs.push_back(io);
          adata[apos++] = NODE_ANNOUNCE_PROC;
          adata[apos++] = io->me.id;
          adata[apos++] = Processor::IO_PROC;
        }
      }

#ifdef USE_CUDA
      // Initialize the driver API
      CHECK_CU( cuInit(0) );
      // Keep track of the local system memories so we can pin them
      // after we've initialized the GPU
      std::vector<LocalCPUMemory*> local_mems;
      // Figure out which GPUs support peer access (if any)
      // and prioritize them so they are used first
      std::vector<int> peer_gpus;
      std::vector<int> dumb_gpus;
      {
        int num_devices;
        CHECK_CU( cuDeviceGetCount(&num_devices) );
        for (int i = 0; i < num_devices; i++)
        {
          CUdevice device;
          CHECK_CU( cuDeviceGet(&device, i) );
          bool has_peer = false;
          // Go through all the other devices and see
          // if we have peer access to them
          for (int j = 0; j < num_devices; j++)
          {
            if (i == j) continue;
            CUdevice peer;
            CHECK_CU( cuDeviceGet(&peer, j) );
            int can_access;
            CHECK_CU( cuDeviceCanAccessPeer(&can_access, device, peer) );
            if (can_access)
            {
              has_peer = true;
              break;
            }
          }
          if (has_peer)
            peer_gpus.push_back(i);
          else
            dumb_gpus.push_back(i);
        }
      }
#endif
      // create local processors
      for(unsigned i = 0; i < num_local_cpus; i++) {
	Processor p = ID(ID::ID_PROCESSOR, 
			 gasnet_mynode(), 
			 n->processors.size()).convert<Processor>();
        ProcessorImpl *lp;
        if (use_greenlet_procs)
          lp = new GreenletProcessor(p, Processor::LOC_PROC,
                                     stack_size_in_mb << 20, init_stack_count,
                                     "local worker", i);
        else
	  lp = new LocalProcessor(p, Processor::LOC_PROC,
                                  stack_size_in_mb << 20,
                                  "local worker", i);
	n->processors.push_back(lp);
	local_cpus.push_back(lp);
	adata[apos++] = NODE_ANNOUNCE_PROC;
	adata[apos++] = lp->me.id;
	adata[apos++] = Processor::LOC_PROC;
      }

      // create local memory
      LocalCPUMemory *cpumem;
      if(cpu_mem_size_in_mb > 0) {
	cpumem = new LocalCPUMemory(ID(ID::ID_MEMORY, 
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    cpu_mem_size_in_mb << 20);
	n->memories.push_back(cpumem);
#ifdef USE_CUDA
        local_mems.push_back(cpumem);
#endif
	adata[apos++] = NODE_ANNOUNCE_MEM;
	adata[apos++] = cpumem->me.id;
        adata[apos++] = Memory::SYSTEM_MEM;
	adata[apos++] = cpumem->size;
	adata[apos++] = 0; // not registered
      } else
	cpumem = 0;

      LocalCPUMemory *regmem;
      if(reg_mem_size_in_mb > 0) {
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );
	char *regmem_base = ((char *)(seginfos[gasnet_mynode()].addr)) + (gasnet_mem_size_in_mb << 20);
	delete[] seginfos;
	regmem = new LocalCPUMemory(ID(ID::ID_MEMORY,
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    reg_mem_size_in_mb << 20,
				    regmem_base,
				    true);
	n->memories.push_back(regmem);
#ifdef USE_CUDA
        local_mems.push_back(regmem);
#endif
	adata[apos++] = NODE_ANNOUNCE_MEM;
	adata[apos++] = regmem->me.id;
        adata[apos++] = Memory::REGDMA_MEM;
	adata[apos++] = regmem->size;
	adata[apos++] = (size_t)(regmem->base);
      } else
	regmem = 0;

      // create local disk memory
      DiskMemory *diskmem;
      if(disk_mem_size_in_mb > 0) {
        diskmem = new DiskMemory(ID(ID::ID_MEMORY,
                                    gasnet_mynode(),
                                    n->memories.size(), 0).convert<Memory>(),
                                 disk_mem_size_in_mb << 20,
                                 "disk_file.tmp");
        n->memories.push_back(diskmem);
        adata[apos++] = NODE_ANNOUNCE_MEM;
        adata[apos++] = diskmem->me.id;
        adata[apos++] = Memory::DISK_MEM;
        adata[apos++] = diskmem->size;
        adata[apos++] = 0;
      } else
        diskmem = 0;

#ifdef USE_HDF
      // create HDF memory
      HDFMemory *hdfmem;
      hdfmem = new HDFMemory(ID(ID::ID_MEMORY,
                                gasnet_mynode(),
                                n->memories.size(), 0).convert<Memory>());
      n->memories.push_back(hdfmem);
      adata[apos++] = NODE_ANNOUNCE_MEM;
      adata[apos++] = hdfmem->me.id;
      adata[apos++] = Memory::HDF_MEM;
      adata[apos++] = hdfmem->size;
      adata[apos++] = 0;
#endif

      // list affinities between local CPUs / memories
      for(std::vector<ProcessorImpl*>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++) {
	if(cpu_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = 100;  // "large" bandwidth
	  adata[apos++] = 1;    // "small" latency
	}

	if(reg_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = regmem->me.id;
	  adata[apos++] = 80;  // "large" bandwidth
	  adata[apos++] = 5;    // "small" latency
	}

        if(disk_mem_size_in_mb > 0) {
          adata[apos++] = NODE_ANNOUNCE_PMA;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = diskmem->me.id;
          adata[apos++] = 5;  // "low" bandwidth
          adata[apos++] = 100;  // "high" latency
        }

#ifdef USE_HDF
         adata[apos++] = NODE_ANNOUNCE_PMA;
         adata[apos++] = (*it)->me.id;
         adata[apos++] = hdfmem->me.id;
         adata[apos++] = 5; // "low" bandwidth
         adata[apos++] = 100; // "high" latency
#endif

	if(global_memory) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = global_memory->me.id;
	  adata[apos++] = 10;  // "lower" bandwidth
	  adata[apos++] = 50;    // "higher" latency
	}
      }

      for(std::vector<ProcessorImpl*>::iterator it = local_io_procs.begin();
          it != local_io_procs.end();
          it++)
      {
        if(cpu_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = 100;  // "large" bandwidth
	  adata[apos++] = 1;    // "small" latency
	}

	if(reg_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = regmem->me.id;
	  adata[apos++] = 80;  // "large" bandwidth
	  adata[apos++] = 5;    // "small" latency
	}

        if(disk_mem_size_in_mb > 0) {
          adata[apos++] = NODE_ANNOUNCE_PMA;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = diskmem->me.id;
          adata[apos++] = 5;  // "low" bandwidth
          adata[apos++] = 100;  // "high" latency
        }

#ifdef USE_HDF
         adata[apos++] = NODE_ANNOUNCE_PMA;
         adata[apos++] = (*it)->me.id;
         adata[apos++] = hdfmem->me.id;
         adata[apos++] = 5; // "low" bandwidth
         adata[apos++] = 100; // "high" latency
#endif

	if(global_memory) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = global_memory->me.id;
	  adata[apos++] = 10;  // "lower" bandwidth
	  adata[apos++] = 50;    // "higher" latency
	}
      }

      // list affinities between local CPUs / memories
      for(std::vector<ProcessorImpl*>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++) {
	if(cpu_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = 100;  // "large" bandwidth
	  adata[apos++] = 1;    // "small" latency
	}

	if(reg_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = regmem->me.id;
	  adata[apos++] = 80;  // "large" bandwidth
	  adata[apos++] = 5;    // "small" latency
	}

	if(global_memory) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = global_memory->me.id;
	  adata[apos++] = 10;  // "lower" bandwidth
	  adata[apos++] = 50;    // "higher" latency
	}

        if(disk_mem_size_in_mb > 0) {
          adata[apos++] = NODE_ANNOUNCE_PMA;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = diskmem->me.id;
          adata[apos++] = 5;  // "low" bandwidth
          adata[apos++] = 100;  // "high" latency
        }
#ifdef USE_HDF
          adata[apos++] = NODE_ANNOUNCE_PMA;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = hdfmem->me.id;
          adata[apos++] = 5;  // "low" bandwidth
          adata[apos++] = 100;  // "high" latency   
#endif
      }

      if((cpu_mem_size_in_mb > 0) && global_memory) {
	adata[apos++] = NODE_ANNOUNCE_MMA;
	adata[apos++] = cpumem->me.id;
	adata[apos++] = global_memory->me.id;
	adata[apos++] = 30;  // "lower" bandwidth
	adata[apos++] = 25;    // "higher" latency
      }

      if((disk_mem_size_in_mb > 0) && (cpu_mem_size_in_mb > 0)) {
        adata[apos++] = NODE_ANNOUNCE_MMA;
        adata[apos++] = cpumem->me.id;
        adata[apos++] = diskmem->me.id;
        adata[apos++] = 15;    // "low" bandwidth
        adata[apos++] = 50;    // "high" latency
      }

#ifdef USE_CUDA
      if(num_local_gpus > 0) {
        if (num_local_gpus > (peer_gpus.size() + dumb_gpus.size()))
        {
          printf("Requested %d GPUs, but only %ld GPUs exist on node %d\n",
            num_local_gpus, peer_gpus.size()+dumb_gpus.size(), gasnet_mynode());
          assert(false);
        }
        GPUWorker *gpu_worker = 0;
        if (gpu_worker_thread) {
          gpu_worker = GPUWorker::start_gpu_worker_thread(stack_size_in_mb << 20);
        }
	for(unsigned i = 0; i < num_local_gpus; i++) {
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
	  //printf("GPU's ID is " IDFMT "\n", p.id);
 	  GPUProcessor *gp = new GPUProcessor(p, Processor::TOC_PROC, "gpu worker",
                                              (i < peer_gpus.size() ?
                                                peer_gpus[i] : 
                                                dumb_gpus[i-peer_gpus.size()]), 
                                              zc_mem_size_in_mb << 20,
                                              fb_mem_size_in_mb << 20,
                                              stack_size_in_mb << 20,
                                              gpu_worker, num_gpu_streams);
	  n->processors.push_back(gp);
	  local_gpus.push_back(gp);

	  adata[apos++] = NODE_ANNOUNCE_PROC;
	  adata[apos++] = p.id;
	  adata[apos++] = Processor::TOC_PROC;

	  Memory m = ID(ID::ID_MEMORY,
			gasnet_mynode(),
			n->memories.size(), 0).convert<Memory>();
	  GPUFBMemory *fbm = new GPUFBMemory(m, gp);
	  n->memories.push_back(fbm);

	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = m.id;
          adata[apos++] = Memory::GPU_FB_MEM;
	  adata[apos++] = fbm->size;
	  adata[apos++] = 0; // not registered

	  // FB has very good bandwidth and ok latency to GPU
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = p.id;
	  adata[apos++] = m.id;
	  adata[apos++] = 200; // "big" bandwidth
	  adata[apos++] = 5;   // "ok" latency

	  Memory m2 = ID(ID::ID_MEMORY,
			 gasnet_mynode(),
			 n->memories.size(), 0).convert<Memory>();
	  GPUZCMemory *zcm = new GPUZCMemory(m2, gp);
	  n->memories.push_back(zcm);

	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = m2.id;
          adata[apos++] = Memory::Z_COPY_MEM;
	  adata[apos++] = zcm->size;
	  adata[apos++] = 0; // not registered

	  // ZC has medium bandwidth and bad latency to GPU
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = p.id;
	  adata[apos++] = m2.id;
	  adata[apos++] = 20;
	  adata[apos++] = 200;

	  // ZC also accessible to all the local CPUs
	  for(std::vector<ProcessorImpl*>::iterator it = local_cpus.begin();
	      it != local_cpus.end();
	      it++) {
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = m2.id;
	    adata[apos++] = 40;
	    adata[apos++] = 3;
	  }
	}
        // Now pin any CPU memories
        if(pin_sysmem_for_gpu)
          for (unsigned idx = 0; idx < local_mems.size(); idx++)
            local_mems[idx]->pin_memory(local_gpus[0]);

        // Register peer access for any GPUs which support it
        if ((num_local_gpus > 1) && (peer_gpus.size() > 1))
        {
          unsigned peer_count = (num_local_gpus < peer_gpus.size()) ? 
                                  num_local_gpus : peer_gpus.size();
          // Needs to go both ways so register in all directions
          for (unsigned i = 0; i < peer_count; i++)
          {
            CUdevice device;
            CHECK_CU( cuDeviceGet(&device, peer_gpus[i]) );
            for (unsigned j = 0; j < peer_count; j++)
            {
              if (i == j) continue;
              CUdevice peer;
              CHECK_CU( cuDeviceGet(&peer, peer_gpus[j]) );
              int can_access;
              CHECK_CU( cuDeviceCanAccessPeer(&can_access, device, peer) );
              if (can_access)
                local_gpus[i]->enable_peer_access(local_gpus[j]);
            }
          }
        }
      }
#endif

      adata[apos++] = NODE_ANNOUNCE_DONE;
      assert(apos < ADATA_SIZE);

      // parse our own data (but don't create remote proc/mem objects)
      {
	AutoHSLLock al(announcement_mutex);
	machine->parse_node_announce_data(adata, apos*sizeof(adata[0]), 
					  announce_data, false);
      }

#ifdef DEBUG_REALM_STARTUP
      if(gasnet_mynode() == 0) {
        LegionRuntime::TimeStamp ts("sending announcements", false);
        fflush(stdout);
      }
#endif

      // now announce ourselves to everyone else
      for(unsigned i = 0; i < gasnet_nodes(); i++)
	if(i != gasnet_mynode())
	  NodeAnnounceMessage::request(i, announce_data, 
				       adata, apos*sizeof(adata[0]),
				       PAYLOAD_COPY);

      // wait until we hear from everyone else?
      while((int)announcements_received < (int)(gasnet_nodes() - 1))
	do_some_polling();

      log_annc.info("node %d has received all of its announcements\n", gasnet_mynode());

#ifdef DEBUG_REALM_STARTUP
      if(gasnet_mynode() == 0) {
        LegionRuntime::TimeStamp ts("received all announcements", false);
        fflush(stdout);
      }
#endif

      // start dma system at the very ending of initialization
      // since we need list of local gpus to create channels
      start_dma_system(dma_worker_threads, 100
#ifdef USE_CUDA
                       ,local_gpus
#endif
                       );

      return true;
    }

    struct MachineRunArgs {
      RuntimeImpl *r;
      Processor::TaskFuncID task_id;
      Runtime::RunStyle style;
      const void *args;
      size_t arglen;
    };  

    static bool running_as_background_thread = false;

    static void *background_run_thread(void *data)
    {
      MachineRunArgs *args = (MachineRunArgs *)data;
      running_as_background_thread = true;
      args->r->run(args->task_id, args->style, args->args, args->arglen,
		   false /* foreground from this thread's perspective */);
      delete args;
      return 0;
    }

    void RuntimeImpl::run(Processor::TaskFuncID task_id /*= 0*/,
			  Runtime::RunStyle style /*= ONE_TASK_ONLY*/,
			  const void *args /*= 0*/, size_t arglen /*= 0*/,
			  bool background /*= false*/)
    { 
      if(background) {
        log_machine.info("background operation requested\n");
	fflush(stdout);
	MachineRunArgs *margs = new MachineRunArgs;
	margs->r = this;
	margs->task_id = task_id;
	margs->style = style;
	margs->args = args;
	margs->arglen = arglen;
	
        pthread_t *threadp = (pthread_t*)malloc(sizeof(pthread_t));
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	CHECK_PTHREAD( pthread_create(threadp, &attr, &background_run_thread, (void *)margs) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
        background_pthread = threadp;
#ifdef DEADLOCK_TRACE
        this->add_thread(threadp); 
#endif
	return;
      }

      // Initialize the shutdown counter
      const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;
      Atomic<int> running_proc_count(local_procs.size());

      for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
	(*it)->run(&running_proc_count);

      // now that we've got the machine description all set up, we can start
      //  the worker threads for local processors, which'll probably ask the
      //  high-level runtime to set itself up
      for(std::vector<ProcessorImpl*>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++)
	(*it)->start_processor();

      for (std::vector<ProcessorImpl*>::iterator it = local_io_procs.begin();
            it != local_io_procs.end();
            it++)
        (*it)->start_processor();

      for(std::vector<ProcessorImpl*>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++)
	(*it)->start_processor();

#ifdef USE_CUDA
      for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	  it != local_gpus.end();
	  it++)
	(*it)->start_processor();
#endif

      if(task_id != 0 && 
	 ((style != Runtime::ONE_TASK_ONLY) || 
	  (gasnet_mynode() == 0))) {//(gasnet_nodes()-1)))) {
	for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  (*it)->spawn_task(task_id, args, arglen, 
			    Event::NO_EVENT, Event::NO_EVENT, 0/*priority*/);
	  if(style != Runtime::ONE_TASK_PER_PROC) break;
	}
      }

      // wait for idle-ness somehow?
      int timeout = -1;
#ifdef TRACE_RESOURCES
      RuntimeImpl *rt = get_runtime();
#endif
      while(running_proc_count.get() > 0) {
	if(timeout >= 0) {
	  timeout--;
	  if(timeout == 0) {
	    printf("TIMEOUT!\n");
	    exit(1);
	  }
	}
	fflush(stdout);
	sleep(1);
#ifdef TRACE_RESOURCES
        log_machine.info("total events: %d", rt->local_event_free_list->next_alloc);
        log_machine.info("total reservations: %d", rt->local_reservation_free_list->next_alloc);
        log_machine.info("total index spaces: %d", rt->local_index_space_free_list->next_alloc);
        log_machine.info("total proc groups: %d", rt->local_proc_group_free_list->next_alloc);
#endif
      }
      log_machine.info("running proc count is now zero - terminating\n");
#ifdef REPORT_REALM_RESOURCE_USAGE
      {
        RuntimeImpl *rt = LegionRuntime::LowLevel::get_runtime();
        printf("node %d realm resource usage: ev=%d, rsrv=%d, idx=%d, pg=%d\n",
               gasnet_mynode(),
               rt->local_event_free_list->next_alloc,
               rt->local_reservation_free_list->next_alloc,
               rt->local_index_space_free_list->next_alloc,
               rt->local_proc_group_free_list->next_alloc);
      }
#endif
#ifdef EVENT_GRAPH_TRACE
      {
        //FILE *log_file = Logger::get_log_file();
        show_event_waiters(/*log_file*/);
      }
#endif

      // Shutdown all the threads
      for(std::vector<ProcessorImpl*>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++)
	(*it)->shutdown_processor();

      for(std::vector<ProcessorImpl*>::iterator it = local_io_procs.begin();
          it != local_io_procs.end();
          it++)
        (*it)->shutdown_processor();

      for(std::vector<ProcessorImpl*>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++)
	(*it)->shutdown_processor();

#ifdef USE_CUDA
      for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	  it != local_gpus.end();
	  it++)
	(*it)->shutdown_processor(); 
#endif


      // delete local processors and memories
      {
	Node& n = nodes[gasnet_mynode()];

	for(std::vector<MemoryImpl *>::iterator it = n.memories.begin();
	    it != n.memories.end();
	    it++)
	  delete (*it);

	// node 0 also deletes the gasnet memory
	if(gasnet_mynode() == 0)
	  delete global_memory;
      }

      // need to kill other threads too so we can actually terminate process
      // Exit out of the thread
      //stop_dma_worker_threads();
      stop_dma_system();
#ifdef USE_CUDA
      GPUWorker::stop_gpu_worker_thread();
#endif
      stop_activemsg_threads();

      // if we are running as a background thread, just terminate this thread
      // if not, do a full process exit - gasnet may have started some threads we don't have handles for,
      //   and if they're left running, the app will hang
      if(running_as_background_thread) {
	pthread_exit(0);
      } else {
	exit(0);
      }
    }

    void RuntimeImpl::shutdown(bool local_request /*= true*/)
    {
      if(local_request) {
	log_machine.info("shutdown request - notifying other nodes\n");
	MachineShutdownRequestArgs args;
	args.initiating_node = gasnet_mynode();

	for(unsigned i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    MachineShutdownRequestMessage::request(i, args);
      }

      log_machine.info("shutdown request - cleaning up local processors\n");

      const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;
      for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
      {
        Event e = GenEventImpl::create_genevent()->current_event();
	(*it)->spawn_task(0 /* shutdown task id */, 0, 0,
			  Event::NO_EVENT, e, 0/*priority*/);
      }
    }

    void RuntimeImpl::wait_for_shutdown(void)
    {
      bool exit_process = true;
      if (background_pthread != 0)
      {
        pthread_t *background_thread = (pthread_t*)background_pthread;
        void *result;
        pthread_join(*background_thread, &result);
        free(background_thread);
        // Set this to null so we don't wait anymore
        background_pthread = 0;
        exit_process = false;
      }

#ifdef EVENT_TRACING
      if(event_trace_file) {
	printf("writing event trace to %s\n", event_trace_file);
        Tracer<EventTraceItem>::dump_trace(event_trace_file, false);
	free(event_trace_file);
	event_trace_file = 0;
      }
#endif
#ifdef LOCK_TRACING
      if (lock_trace_file)
      {
        printf("writing lock trace to %s\n", lock_trace_file);
        Tracer<LockTraceItem>::dump_trace(lock_trace_file, false);
        free(lock_trace_file);
        lock_trace_file = 0;
      }
#endif

      // this terminates the process, so control never gets back to caller
      // would be nice to fix this...
      if (exit_process)
        gasnet_exit(0);
    }

  }; // namespace LowLevel

}; // namespace LegionRuntime

// Implementation of accessor methods
namespace LegionRuntime {
  namespace Accessor {
    using namespace LegionRuntime::LowLevel;

    void AccessorType::Generic::Untyped::read_untyped(ptr_t ptr, void *dst, size_t bytes, off_t offset) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
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
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, dp);
#endif
#ifdef USE_HDF
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
        unsigned index = id.index_l();
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
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
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
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

#ifdef PRIVILEGE_CHECKS
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, dp);
#endif
#ifdef USE_HDF
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
        unsigned index = id.index_l();
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
      StaticAccess<RegionInstanceImpl> idata(impl);

      off_t offset = idata->alloc_offset;
      off_t elmt_stride;
      
      if (idata->block_size == 1) {
        offset += field_offset;
        elmt_stride = idata->elmt_size;
      } else {
        off_t field_start;
        int field_size;
        find_field_start(idata->field_sizes, field_offset, 1, field_start, field_size);

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
        find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

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

    void *AccessorType::Generic::Untyped::raw_span_ptr(ptr_t ptr, size_t req_count, size_t& act_count, ByteOffset& stride)
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      void *base;
      size_t act_stride = 0;
      bool ok = impl->get_strided_parameters(base, act_stride, field_offset);
      assert(ok);

#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
#endif

      Arrays::Mapping<1, 1> *mapping = impl->metadata.linearization.get_mapping<1>();
      int index = mapping->image(ptr.value);

      void *elem_ptr = ((char *)base) + (index * act_stride);
      stride.offset = act_stride;
      act_count = req_count; // TODO: return a larger number if we know how big we are
      return elem_ptr;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset *offsets)
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      MemoryImpl *mem = get_runtime()->get_memory_impl(impl->memory);

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      Arrays::Mapping<DIM, 1> *mapping = impl->metadata.linearization.get_mapping<DIM>();

      Point<1> strides[DIM];
      int index = mapping->image_linear_subrect(r, subrect, strides);

      off_t offset = impl->metadata.alloc_offset;
      off_t elmt_stride;

      if(impl->metadata.block_size == 1) {
	offset += index * impl->metadata.elmt_size + field_offset;
	elmt_stride = impl->metadata.elmt_size;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

	int block_num = index / impl->metadata.block_size;
	int block_ofs = index % impl->metadata.block_size;

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
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset *offsets,
						       const std::vector<off_t> &field_offsets, ByteOffset &field_stride)
    {
      if(field_offsets.size() < 1)
	return 0;

      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      MemoryImpl *mem = get_runtime()->get_memory_impl(impl->memory);

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      Arrays::Mapping<DIM, 1> *mapping = impl->metadata.linearization.get_mapping<DIM>();

      Point<1> strides[DIM];
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
	find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

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
	  find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[1], 1, field_start2, field_size2);

	  // field sizes much match or element stride isn't consistent
	  if(field_size2 != field_size)
	    return 0;
	  
	  fld_stride = (((field_start2 - field_start) * impl->metadata.block_size) + 
			(field_offsets[1] - field_start2) - (field_offsets[0] - field_start));

	  for(size_t i = 2; i < field_offsets.size(); i++) {
	    find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[i], 1, field_start2, field_size2);
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
    void *AccessorType::Generic::Untyped::raw_dense_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset &elem_stride)
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl *) internal;
      MemoryImpl *mem = get_runtime()->get_memory_impl(impl->memory);

      // must have valid data by now - block if we have to
      impl->metadata.await_data();

      Arrays::Mapping<DIM, 1> *mapping = impl->metadata.linearization.get_mapping<DIM>();

      Rect<1> ir = mapping->image_dense_subrect(r, subrect);
      int index = ir.lo;

      off_t offset = impl->metadata.alloc_offset;
      off_t elmt_stride;

      if(impl->metadata.block_size == 1) {
	offset += index * impl->metadata.elmt_size + field_offset;
	elmt_stride = impl->metadata.elmt_size;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(impl->metadata.field_sizes, field_offset, 1, field_start, field_size);

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
    void *AccessorType::Generic::Untyped::raw_dense_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset &elem_stride,
							const std::vector<off_t> &field_offsets, ByteOffset &field_stride)
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
	find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[0], 1, field_start, field_size);

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
	  find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[1], 1, field_start2, field_size2);

	  // field sizes much match or element stride isn't consistent
	  if(field_size2 != field_size)
	    return 0;
	  
	  fld_stride = (((field_start2 - field_start) * impl->metadata.block_size) + 
			(field_offsets[1] - field_start2) - (field_offsets[0] - field_start));

	  for(size_t i = 2; i < field_offsets.size(); i++) {
	    find_field_start(impl->metadata.field_sizes, field_offset + field_offsets[i], 1, field_start2, field_size2);
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

    template void *AccessorType::Generic::Untyped::raw_rect_ptr<1>(const Rect<1>& r, Rect<1>& subrect, ByteOffset *offset);
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<2>(const Rect<2>& r, Rect<2>& subrect, ByteOffset *offset);
    template void *AccessorType::Generic::Untyped::raw_rect_ptr<3>(const Rect<3>& r, Rect<3>& subrect, ByteOffset *offset);
    template void *AccessorType::Generic::Untyped::raw_dense_ptr<1>(const Rect<1>& r, Rect<1>& subrect, ByteOffset &elem_stride);
    template void *AccessorType::Generic::Untyped::raw_dense_ptr<2>(const Rect<2>& r, Rect<2>& subrect, ByteOffset &elem_stride);
    template void *AccessorType::Generic::Untyped::raw_dense_ptr<3>(const Rect<3>& r, Rect<3>& subrect, ByteOffset &elem_stride);
  };

  namespace Arrays {
    //template<> class Mapping<1,1>;
    template <unsigned IDIM, unsigned ODIM>
    MappingRegistry<IDIM, ODIM> Mapping<IDIM, ODIM>::registry;
  };
};
