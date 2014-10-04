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

GASNETT_THREADKEY_DEFINE(cur_thread);
GASNETT_THREADKEY_DEFINE(gpu_thread);

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <dirent.h>

#ifdef DEADLOCK_TRACE
#include <signal.h>
#include <execinfo.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

// Implementation of Detailed Timer
namespace LegionRuntime {
  namespace LowLevel {
    
    Logger::Category log_gpu("gpu");
    Logger::Category log_mutex("mutex");
    Logger::Category log_timer("timer");
    Logger::Category log_region("region");
    Logger::Category log_malloc("malloc");
    Logger::Category log_machine("machine");
    Logger::Category log_inst("inst");

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
    };

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
        gasnet_hsl_init(&mutex);
      }

      pthread_t thread;
      std::list<TimerStackEntry> timer_stack;
      std::map<int, double> timer_accum;
      gasnet_hsl_t mutex;
    };

    gasnet_hsl_t timer_data_mutex = GASNET_HSL_INITIALIZER;
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
      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;
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
      : timerp(&_timers)
    {
      gasnet_hsl_init(&mutex);
      gasnett_cond_init(&condvar);
      count_left = 0;
    }

    void MultiNodeRollUp::execute(void)
    {
      count_left = gasnet_nodes()-1;

      RollUpRequestArgs args;
      args.sender = gasnet_mynode();
      args.rollup_ptr = this;
      for(int i = 0; i < gasnet_nodes(); i++)
        if(i != gasnet_mynode())
          RollUpRequestMessage::request(i, args);

      // take the lock so that we can safely sleep until all the responses
      //  arrive
      {
	AutoHSLLock al(mutex);

	if(count_left > 0)
	  gasnett_cond_wait(&condvar, &mutex.lock);
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
	gasnett_cond_signal(&condvar);
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

    /*static*/ Runtime *Runtime::runtime = 0;

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

    ReductionOpTable reduce_op_table;

    IndexSpace::Impl::Impl(void)
    {
      init(IndexSpace::NO_SPACE, -1);
    }

    void IndexSpace::Impl::init(IndexSpace _me, unsigned _init_owner)
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
      gasnet_hsl_init(&valid_mask_mutex);
    }

    void IndexSpace::Impl::init(IndexSpace _me, IndexSpace _parent,
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
      gasnet_hsl_init(&valid_mask_mutex);
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
	  StaticAccess<IndexSpace::Impl> pdata(_parent.impl());
	  locked_data.first_elmt = pdata->first_elmt;
	  locked_data.last_elmt = pdata->last_elmt;
	}
      }
      lock.init(ID(me).convert<Reservation>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    IndexSpace::Impl::~Impl(void)
    {
      delete valid_mask;
    }

    bool IndexSpace::Impl::is_parent_of(IndexSpace other)
    {
      while(other != IndexSpace::NO_SPACE) {
	if(other == me) return true;
	IndexSpace::Impl *other_impl = other.impl();
	other = StaticAccess<IndexSpace::Impl>(other_impl)->parent;
      }
      return false;
    }

#if 0
    size_t IndexSpace::Impl::instance_size(const ReductionOpUntyped *redop /*= 0*/, off_t list_size /*= -1*/)
    {
      StaticAccess<IndexSpace::Impl> data(this);
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

    off_t IndexSpace::Impl::instance_adjust(const ReductionOpUntyped *redop /*= 0*/)
    {
      StaticAccess<IndexSpace::Impl> data(this);
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

    Event IndexSpace::Impl::request_valid_mask(void)
    {
      size_t num_elmts = StaticAccess<IndexSpace::Impl>(this)->num_elmts;
      int valid_mask_owner = -1;
      
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
	valid_mask_event = Event::Impl::create_event();
      }
      
      ValidMaskRequestArgs args;
      args.is = me;
      args.sender = gasnet_mynode();
      ValidMaskRequestMessage::request(valid_mask_owner, args);

      return valid_mask_event;
    }

    void handle_valid_mask_request(ValidMaskRequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpace::Impl *r_impl = args.is.impl();

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
      IndexSpace::Impl *r_impl = args.is.impl();

      assert(r_impl->valid_mask);
      // removing const on purpose here...
      char *mask_data = (char *)(r_impl->valid_mask->get_raw());
      assert(mask_data);
      assert((args.block_id << 11) < r_impl->valid_mask->raw_size());

      memcpy(mask_data + (args.block_id << 11), data, datalen);

      bool trigger = false;
      {
	AutoHSLLock a(r_impl->valid_mask_mutex);
	//printf("got piece of valid mask data for region " IDFMT " (%d expected)\n",
	//       args.region.id, r_impl->valid_mask_count);
	r_impl->valid_mask_count--;
        if(r_impl->valid_mask_count == 0) {
	  r_impl->valid_mask_complete = true;
	  trigger = true;
	}
      }

      if(trigger) {
	//printf("triggering " IDFMT "/%d\n",
	//       r_impl->valid_mask_event.id, r_impl->valid_mask_event.gen);
	r_impl->valid_mask_event.impl()->trigger(r_impl->valid_mask_event.gen,
						 gasnet_mynode());
      }
    }
    

    class IndexSpaceAllocator::Impl {
    public:
      Impl(IndexSpace::Impl *_is_impl);

      ~Impl(void);

      unsigned alloc_elements(unsigned count = 1);

      void reserve_elements(unsigned ptr, unsigned count = 1);

      void free_elements(unsigned ptr, unsigned count = 1);

      IndexSpace::Impl *is_impl;
    };

    RegionInstance::Impl::Impl(RegionInstance _me, IndexSpace _is, Memory _memory, off_t _offset, size_t _size, ReductionOpID _redopid,
			       const DomainLinearization& _linear, size_t _block_size, size_t _elmt_size, const std::vector<size_t>& _field_sizes,
			       off_t _count_offset /*= 0*/, off_t _red_list_size /*= 0*/, RegionInstance _parent_inst /*= NO_INST*/)
      : me(_me), memory(_memory)
    {
      linearization = _linear;
      linearization.serialize(locked_data.linearization_bits);

      locked_data.block_size = _block_size;
      locked_data.elmt_size = _elmt_size;

      assert(_field_sizes.size() <= MAX_FIELDS_PER_INST);
      for(unsigned i = 0; i < _field_sizes.size(); i++)
	locked_data.field_sizes[i] = _field_sizes[i];
      if(_field_sizes.size() < MAX_FIELDS_PER_INST)
	locked_data.field_sizes[_field_sizes.size()] = 0;

      locked_data.is = _is;
      locked_data.alloc_offset = _offset;
      //locked_data.access_offset = _offset + _adjust;
      locked_data.size = _size;
      
      //StaticAccess<IndexSpace::Impl> rdata(_is.impl());
      //locked_data.first_elmt = rdata->first_elmt;
      //locked_data.last_elmt = rdata->last_elmt;

      locked_data.redopid = _redopid;
      locked_data.count_offset = _count_offset;
      locked_data.red_list_size = _red_list_size;
      locked_data.parent_inst = _parent_inst;

      locked_data.valid = true;

      lock.init(ID(me).convert<Reservation>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    // when we auto-create a remote instance, we don't know region/offset
    RegionInstance::Impl::Impl(RegionInstance _me, Memory _memory)
      : me(_me), memory(_memory)
    {
      locked_data.valid = false;
      locked_data.is = IndexSpace::NO_SPACE;
      locked_data.alloc_offset = -1;
      //locked_data.access_offset = -1;
      locked_data.size = 0;
      //locked_data.first_elmt = 0;
      //locked_data.last_elmt = 0;
      lock.init(ID(me).convert<Reservation>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    RegionInstance::Impl::~Impl(void) {}

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

    bool RegionInstance::Impl::get_strided_parameters(void *&base, size_t &stride,
						      off_t field_offset)
    {
      Memory::Impl *mem = memory.impl();
      StaticAccess<RegionInstance::Impl> idata(this);

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
      const DomainLinearization& dl = linearization;
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

    AddressSpace RegionInstance::address_space(void) const
    {
      return ID(id).node();
    }

    IDType RegionInstance::local_id(void) const
    {
      return ID(id).index();
    }

    ///////////////////////////////////////////////////
    // Events

    ///*static*/ Event::Impl *Event::Impl::first_free = 0;
    ///*static*/ gasnet_hsl_t Event::Impl::freelist_mutex = GASNET_HSL_INITIALIZER;

    Event::Impl::Impl(void)
    {
      Event bad = { -1, -1 };
      init(bad, -1); 
    }

    void Event::Impl::init(Event _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      generation = 0;
      gen_subscribed = 0;
      free_generation = 0;
      next_free = 0;
      mutex = new gasnet_hsl_t;
      //printf("[%d] MUTEX INIT %p\n", gasnet_mynode(), mutex);
      gasnet_hsl_init(mutex);
      remote_waiters.clear();
      base_arrival_count = current_arrival_count = 0;
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
      gasnet_node_t node;
      Event event;
    };

    void handle_event_trigger(EventTriggerArgs args);

    typedef ActiveMessageShortNoReply<EVENT_TRIGGER_MSGID,
				      EventTriggerArgs,
				      handle_event_trigger> EventTriggerMessage;

    static Logger::Category log_event("event");

    void handle_event_subscribe(EventSubscribeArgs args)
    {
      log_event(LEVEL_DEBUG, "event subscription: node=%d event=" IDFMT "/%d",
		args.node, args.event.id, args.event.gen);

      Event::Impl *impl = args.event.impl();

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
	log_event(LEVEL_DEBUG, "event subscription early-out: node=%d event=" IDFMT "/%d (<= %d)",
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
        // first trigger any generations which are below are current generation
        if(impl->generation > (args.previous_subscribe_gen)) {
          log_event(LEVEL_DEBUG, "event subscription already done: node=%d event=" IDFMT "/%d (<= %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	  EventTriggerArgs trigger_args;
	  trigger_args.node = gasnet_mynode();
	  trigger_args.event = args.event;
	  trigger_args.event.gen = impl->generation;
	  EventTriggerMessage::request(args.node, trigger_args);
        }
        NodeMask waiter_mask;
        waiter_mask.set_bit(args.node);
        // now register any remote waiters
        for (Event::gen_t wait_gen = impl->generation+1;
              wait_gen <= args.event.gen; wait_gen++) {
          // nope - needed generation hasn't happened yet, so add this node to
	  //  the mask
	  log_event(LEVEL_DEBUG, "event subscription recorded: node=%d event=" IDFMT "/%d (> %d)",
		    args.node, args.event.id, wait_gen, impl->generation);
	  //impl->remote_waiters |= (1ULL << args.node);
          std::map<Event::gen_t,NodeMask>::iterator finder =
            impl->remote_waiters.find(wait_gen);
          if (finder == impl->remote_waiters.end()) {
            impl->remote_waiters[wait_gen] = waiter_mask;
          }
          else
            finder->second.set_bit(args.node);
        }
      }
    }

    void show_event_waiters(FILE *f = stdout)
    {
      fprintf(f,"PRINTING ALL PENDING EVENTS:\n");
      for(int i = 0; i < gasnet_nodes(); i++) {
	Node *n = &Runtime::runtime->nodes[i];
        // Iterate over all the events and get their implementations
        for (unsigned long j = 0; j < n->events.max_entries(); j++) {
          if (!n->events.has_entry(j))
            continue;
	  Event::Impl *e = n->events.lookup_entry(j, i/*node*/);
	  AutoHSLLock a2(e->mutex);

	  // print anything with either local or remote waiters
	  if(e->local_waiters.empty() && e->remote_waiters.empty())
	    continue;

          fprintf(f,"Event " IDFMT ": gen=%d subscr=%d local=%zd remote=%zd\n",
		  e->me.id, e->generation, e->gen_subscribed, 
		  e->local_waiters.size(), e->remote_waiters.size());
	  for(std::map<Event::gen_t, std::vector<Event::Impl::EventWaiter *> >::iterator it = e->local_waiters.begin();
	      it != e->local_waiters.end();
	      it++) {
	    for(std::vector<Event::Impl::EventWaiter *>::iterator it2 = it->second.begin();
		it2 != it->second.end();
		it2++) {
	      fprintf(f, "  [%d] L:%p ", it->first, *it2);
	      (*it2)->print_info(f);
	    }
	  }
	  for(std::map<Event::gen_t, NodeMask>::const_iterator it = e->remote_waiters.begin();
	      it != e->remote_waiters.end();
	      it++) {
	    fprintf(f, "  [%d] R:", it->first);
	    for(int k = 0; k < MAX_NUM_NODES; k++)
	      if(it->second.is_set(k))
		fprintf(f, " %d", k);
	    fprintf(f, "\n");
	  }
	}
      }
      fprintf(f,"DONE\n");
      fflush(f);
    }

    void handle_event_trigger(EventTriggerArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_event(LEVEL_DEBUG, "Remote trigger of event " IDFMT "/%d from node %d!",
		args.event.id, args.event.gen, args.node);
      args.event.impl()->trigger(args.event.gen, args.node);
    }

    /*static*/ const Event Event::NO_EVENT = { 0, 0 };
    // Take this you POS C++ type system
    /* static */ const UserEvent UserEvent::NO_USER_EVENT = 
          *(static_cast<UserEvent*>(const_cast<Event*>(&Event::NO_EVENT)));
    Event::Impl *Event::impl(void) const
    {
      return Runtime::runtime->get_event_impl(*this);
    }

    bool Event::has_triggered(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if(!id) return true; // special case: NO_EVENT has always triggered
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);
      return e->has_triggered(gen);
    }


    // Perform our merging events in a lock free way
#define LOCK_FREE_MERGED_EVENTS
    class EventMerger : public Event::Impl::EventWaiter {
    public:
      EventMerger(Event _finish_event)
	: count_needed(1), finish_event(_finish_event)
      {
#ifndef LOCK_FREE_MERGED_EVENTS
	gasnet_hsl_init(&mutex);
#endif
      }

      virtual ~EventMerger(void)
      {
#ifndef LOCK_FREE_MERGED_EVENTS
        gasnet_hsl_destroy(&mutex);
#endif
      }

      void add_event(Event wait_for)
      {
	if(wait_for.has_triggered()) return; // early out
	{
#ifdef LOCK_FREE_MERGED_EVENTS
          __sync_fetch_and_add(&count_needed, 1);
#else
	  // step 1: increment our count first - we can't hold the lock while
	  //   we add a listener to the 'wait_for' event (since it might trigger
	  //   instantly and call our count-decrementing function), and we
	  //   need to make sure all increments happen before corresponding
	  //   decrements
	  AutoHSLLock a(mutex);
	  count_needed++;
#endif
	}
	// step 2: enqueue ourselves on the input event
	wait_for.impl()->add_waiter(wait_for, this);
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
	bool last_trigger = false;
#ifdef LOCK_FREE_MERGED_EVENTS
	unsigned count_left = __sync_fetch_and_add(&count_needed, -1);
	log_event(LEVEL_INFO, "recevied trigger merged event " IDFMT "/%d (%d)",
		  finish_event.id, finish_event.gen, count_left);
	last_trigger = (count_left == 1);
#else
	{
	  AutoHSLLock a(mutex);
	  log_event(LEVEL_INFO, "recevied trigger merged event " IDFMT "/%d (%d)",
		    finish_event.id, finish_event.gen, count_needed);
	  count_needed--;
	  if(count_needed == 0) last_trigger = true;
	}
#endif
	// actually do triggering outside of lock (maybe not necessary, but
	//  feels safer :)
	//	if(last_trigger)
	//	  finish_event.impl()->trigger(finish_event.gen, gasnet_mynode());
	if(last_trigger) {
	  Event::Impl *i;
	  {
	    //TimeStamp ts("foo5", true);
	    i = finish_event.impl();
	  }
	  {
	    //TimeStamp ts("foo6", true);
	    i->trigger(finish_event.gen, gasnet_mynode());
	  }
	}

        // caller can delete us if this was the last trigger
        return last_trigger;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"event merger: " IDFMT "/%d\n", finish_event.id, finish_event.gen);
      }

    protected:
      unsigned count_needed;
      Event finish_event;
#ifndef LOCK_FREE_MERGED_EVENTS
      gasnet_hsl_t mutex;
#endif
    };

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::Impl::merge_events(const std::set<Event>& wait_for)
    {
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
      log_event(LEVEL_INFO, "merging events - at least %d not triggered",
		wait_count);

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;

      // counts of 2+ require building a new event and a merger to trigger it
      Event finish_event = Event::Impl::create_event();
      EventMerger *m = new EventMerger(finish_event);

      for(std::set<Event>::const_iterator it = wait_for.begin();
	  it != wait_for.end();
	  it++) {
	log_event(LEVEL_INFO, "merged event " IDFMT "/%d waiting for " IDFMT "/%d",
		  finish_event.id, finish_event.gen, (*it).id, (*it).gen);
	m->add_event(*it);
      }

      // once they're all added - arm the thing (it might go off immediately)
      if(m->arm())
        delete m;

      return finish_event;
    }

    /*static*/ Event Event::Impl::merge_events(Event ev1, Event ev2,
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

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;

      // counts of 2+ require building a new event and a merger to trigger it
      Event finish_event = Event::Impl::create_event();
      EventMerger *m = new EventMerger(finish_event);

      m->add_event(ev1);
      m->add_event(ev2);
      m->add_event(ev3);
      m->add_event(ev4);
      m->add_event(ev5);
      m->add_event(ev6);

      // once they're all added - arm the thing (it might go off immediately)
      if(m->arm())
        delete m;

      return finish_event;
    }

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Event::Impl::merge_events(wait_for);
    }

    /*static*/ Event Event::merge_events(Event ev1, Event ev2,
					 Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
					 Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Event::Impl::merge_events(ev1, ev2, ev3, ev4, ev5, ev6);
    }

    /*static*/ UserEvent UserEvent::create_user_event(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Event e = Event::Impl::create_event();
      assert(e.id != 0);
      UserEvent u;
      u.id = e.id;
      u.gen = e.gen;
      return u;
    }

    void UserEvent::trigger(Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      impl()->trigger(gen, gasnet_mynode(), wait_on);
      //Runtime::get_runtime()->get_event_impl(*this)->trigger();
    }

    /*static*/ Event Event::Impl::create_event(void)
    {
      Event::Impl *impl = Runtime::runtime->local_event_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).type() == ID::ID_EVENT);
      if(impl) {
	assert(impl->generation == impl->free_generation);
	impl->free_generation++; // normal events are one-shot
	Event ev = impl->me;
	ev.gen = impl->generation + 1;
	//printf("REUSE EVENT %x/%d\n", ev.id, ev.gen);
	log_event.info("event created: event=" IDFMT "/%d", ev.id, ev.gen);
#ifdef EVENT_TRACING
        {
          EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
          item.event_id = ev.id;
          item.event_gen = ev.gen;
          item.action = EventTraceItem::ACT_CREATE;
        }
#endif
	return ev;
      }
      assert(false);
      return Event::NO_EVENT;
#if 0
      
      //DetailedTimer::ScopedPush sp(17);
      // see if the freelist has an event we can reuse
      Event::Impl *impl = 0;
      {
	AutoHSLLock al(&freelist_mutex);
	if(first_free) {
	  impl = first_free;
	  first_free = impl->next_free;
	}
      }
      if(impl) {
	assert(impl->generation == impl->free_generation);
	impl->free_generation++; // normal events are one-shot
	Event ev = impl->me;
	ev.gen = impl->generation + 1;
	//printf("REUSE EVENT " IDFMT "/%d\n", ev.id, ev.gen);
	log_event(LEVEL_SPEW, "event reused: event=" IDFMT "/%d", ev.id, ev.gen);
#ifdef EVENT_TRACING
        {
          EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
          item.event_id = ev.id;
          item.event_gen = ev.gen;
          item.action = EventTraceItem::ACT_CREATE;
        }
#endif
	return ev;
      }

      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

      std::vector<Event::Impl>& events = Runtime::runtime->nodes[gasnet_mynode()].events;

#ifdef SCAN_FOR_FREE_EVENTS
      // try to find an event we can reuse
      for(std::vector<Event::Impl>::iterator it = events.begin();
	  it != events.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != gasnet_mynode())) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == gasnet_mynode())) {
	  // now we really have the event
	  (*it).in_use = true;
	  Event ev = (*it).me;
	  ev.gen = (*it).generation + 1;
	  //printf("REUSE EVENT " IDFMT "/%d\n", ev.id, ev.gen);
	  log_event(LEVEL_SPEW, "event reused: event=" IDFMT "/%d", ev.id, ev.gen);
#ifdef EVENT_TRACING
          {
	    EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
            item.event_id = ev.id;
            item.event_gen = ev.gen;
            item.action = EventTraceItem::ACT_CREATE;
          }
#endif
	  return ev;
	}
      }
#endif

      // couldn't reuse an event - make a new one
      // TODO: take a lock here!?
      unsigned index = events.size();
      assert((index+1) < MAX_LOCAL_EVENTS);
      events.resize(index + 1);
      Event ev = ID(ID::ID_EVENT, gasnet_mynode(), index).convert<Event>();
      events[index].init(ev, gasnet_mynode());
      events[index].free_generation++; // this event will be available after this generation
      Runtime::runtime->nodes[gasnet_mynode()].num_events = index + 1;
      ev.gen = 1; // waiting for first generation of this new event
      //printf("NEW EVENT " IDFMT "/%d\n", ev.id, ev.gen);
      log_event(LEVEL_SPEW, "event created: event=" IDFMT "/%d", ev.id, ev.gen);
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = ev.id;
        item.event_gen = ev.gen;
        item.action = EventTraceItem::ACT_CREATE;
      }
#endif
      return ev;
#endif
    }

    void Event::Impl::add_waiter(Event event, EventWaiter *waiter, bool pre_subscribed /*= false*/)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = event.id;
        item.event_gen = event.gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif
      bool trigger_now = false;

      int subscribe_owner = -1;
      EventSubscribeArgs args;

      {
	AutoHSLLock a(mutex);

	if(event.gen > generation) {
	  log_event(LEVEL_DEBUG, "event not ready: event=" IDFMT "/%d owner=%d gen=%d subscr=%d",
		    event.id, event.gen, owner, generation, gen_subscribed);
	  // we haven't triggered the needed generation yet - add to list of
	  //  waiters, and subscribe if we're not the owner
	  local_waiters[event.gen].push_back(waiter);
	  //printf("LOCAL WAITERS CHECK: %zd\n", local_waiters.size());

	  if((owner != gasnet_mynode()) && (event.gen > gen_subscribed)) {
	    args.node = gasnet_mynode();
	    args.event = event;
            args.previous_subscribe_gen = 
              (gen_subscribed > generation ? gen_subscribed : generation);
	    subscribe_owner = owner;
	    gen_subscribed = event.gen;
	  }
	} else {
	  // event we are interested in has already triggered!
	  trigger_now = true; // actually do trigger outside of mutex
	}
      }

      if((subscribe_owner != -1) && !pre_subscribed)
	EventSubscribeMessage::request(owner, args);

      if(trigger_now) {
	bool nuke = waiter->event_triggered();
        if(nuke)
          delete waiter;
      }
    }

    bool Event::Impl::has_triggered(Event::gen_t needed_gen) volatile
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

    class PthreadCondWaiter : public Event::Impl::EventWaiter {
    public:
      PthreadCondWaiter(Event::Impl *i)
        : impl(i)
      {
        gasnett_cond_init(&cond);
      }
      virtual ~PthreadCondWaiter(void) 
      {
        gasnett_cond_destroy(&cond);
      }

      virtual bool event_triggered(void)
      {
        // Need to hold the lock to avoid the race
        AutoHSLLock(impl->mutex);
        gasnett_cond_signal(&cond);
        // we're allocated on caller's stack, so deleting would be bad
        return false;
      }
      virtual void print_info(FILE *f) { fprintf(f,"external waiter\n"); }

    public:
      gasnett_cond_t cond;
      Event::Impl *impl;
    };

    void Event::Impl::external_wait(Event::gen_t gen_needed)
    {
      PthreadCondWaiter w(this);
      {
	AutoHSLLock a(mutex);

	if(gen_needed > generation) {
	  local_waiters[gen_needed].push_back(&w);
    
	  if((owner != gasnet_mynode()) && (gen_needed > gen_subscribed)) {
	    printf("AAAH!  Can't subscribe to another node's event in external_wait()!\n");
	    exit(1);
	  }

	  // now just sleep on the condition variable - hope we wake up
          gasnett_cond_wait(&w.cond, &mutex->lock);
	}
      }
    }

    class DeferredEventTrigger : public Event::Impl::EventWaiter {
    public:
      DeferredEventTrigger(Event _after_event)
	: after_event(_after_event)
      {}

      virtual ~DeferredEventTrigger(void) { }

      virtual bool event_triggered(void)
      {
	log_event.info("deferred trigger occuring: " IDFMT "/%d", after_event.id, after_event.gen);
	after_event.impl()->trigger(after_event.gen, gasnet_mynode(), Event::NO_EVENT);
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred trigger: after=" IDFMT "/%d\n",
	       after_event.id, after_event.gen);
      }

    protected:
      Event after_event;
    };

    void Event::Impl::trigger(Event::gen_t gen_triggered, int trigger_node, Event wait_on)
    {
      if(!wait_on.has_triggered()) {
	// deferred trigger
	// TODO: forward the deferred trigger to the owning node if it's remote
	Event after_event;
	after_event.id = me.id;
	after_event.gen = gen_triggered;
	log_event.info("deferring event trigger: in=" IDFMT "/%d out=" IDFMT "/%d\n",
		       wait_on.id, wait_on.gen, me.id, gen_triggered);
	wait_on.impl()->add_waiter(wait_on, new DeferredEventTrigger(after_event));
	return;
      }

      log_event(LEVEL_SPEW, "event triggered: event=" IDFMT "/%d by node %d", 
		me.id, gen_triggered, trigger_node);
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = gen_triggered;
        item.action = EventTraceItem::ACT_TRIGGER;
      }
#endif
      //printf("[%d] TRIGGER " IDFMT "/%d\n", gasnet_mynode(), me.id, gen_triggered);
      std::deque<EventWaiter *> to_wake;
      bool release_event = false;
      {
	//TimeStamp ts("foo", true);
	//printf("[%d] TRIGGER MUTEX IN " IDFMT "/%d\n", gasnet_mynode(), me.id, gen_triggered);
	AutoHSLLock a(mutex);
	//printf("[%d] TRIGGER MUTEX HOLD " IDFMT "/%d\n", gasnet_mynode(), me.id, gen_triggered);

	//printf("[%d] TRIGGER GEN: " IDFMT "/%d->%d\n", gasnet_mynode(), me.id, generation, gen_triggered);
	// SJT: there is at least one unavoidable case where we'll receive
	//  duplicate trigger notifications, so if we see a triggering of
	//  an older generation, just ignore it
	if(gen_triggered <= generation) return;
	//assert(gen_triggered > generation);
	generation = gen_triggered;
	// if the generation is caught up to the "free generation", we can release the event
	if(generation == free_generation)
	  release_event = true;

	//printf("[%d] LOCAL WAITERS: %zd\n", gasnet_mynode(), local_waiters.size());
	std::map<Event::gen_t, std::vector<EventWaiter *> >::iterator it = local_waiters.begin();
	while((it != local_waiters.end()) && (it->first <= gen_triggered)) {
	  //printf("[%d] LOCAL WAIT: %d (%zd)\n", gasnet_mynode(), it->first, it->second.size());
	  to_wake.insert(to_wake.end(), it->second.begin(), it->second.end());
	  local_waiters.erase(it);
	  it = local_waiters.begin();
	}

	// notify remote waiters and/or event's actual owner
	if(owner == gasnet_mynode()) {
	  // send notifications to every other node that has subscribed
	  //  (except the one that triggered)
	  EventTriggerArgs args;
	  args.node = trigger_node;
	  args.event = me;
	  args.event.gen = gen_triggered;
          NodeMask send_mask;
          std::map<Event::gen_t,NodeMask>::iterator nit = remote_waiters.begin();
          while((nit != remote_waiters.end()) && (nit->first <= gen_triggered)) {
            send_mask |= nit->second;
            remote_waiters.erase(nit);
            nit = remote_waiters.begin();
          }
	  //for(int node = 0; remote_waiters != 0; node++, remote_waiters >>= 1)
	  //  if((remote_waiters & 1) && (node != trigger_node))
	  //    EventTriggerMessage::request(node, args);
          for(int node = 0; node < MAX_NUM_NODES; node++)
            if (send_mask.is_set(node) && (node != trigger_node))
              EventTriggerMessage::request(node, args);
	} else {
	  if(trigger_node == gasnet_mynode()) {
	    // if we're not the owner, we just send to the owner and let him
	    //  do the broadcast (assuming the trigger was local)
	    //assert(remote_waiters == 0);

	    EventTriggerArgs args;
	    args.node = trigger_node;
	    args.event = me;
	    args.event.gen = gen_triggered;
	    EventTriggerMessage::request(owner, args);
	  }
	}
      }

      if(release_event) {
	//TimeStamp ts("foo2", true);
	// if this is one of our events, put ourselves on the free
	//  list (we don't need our lock for this)
	if(owner == gasnet_mynode()) {
	  //AutoHSLLock al(&freelist_mutex);
	  base_arrival_count = current_arrival_count = 0;
	  Runtime::runtime->local_event_free_list->free_entry(this);
	  //next_free = first_free;
	  //first_free = this;
	}
      }

      {
	//TimeStamp ts("foo3", true);
	// now that we've let go of the lock, notify all the waiters who wanted
	//  this event generation (or an older one)
	for(std::deque<EventWaiter *>::iterator it = to_wake.begin();
	    it != to_wake.end();
	    it++) {
	  bool nuke = (*it)->event_triggered();
          if(nuke) {
            //printf("deleting: "); (*it)->print_info(); fflush(stdout);
            delete (*it);
          }
        }
      }

      {
	//TimeStamp ts("foo4", true);
	{
	  //TimeStamp ts("foo4b", true);
	}
      }
    }

    static Barrier::timestamp_t barrier_adjustment_timestamp;

    static const int BARRIER_TIMESTAMP_NODEID_SHIFT = 48;

    struct BarrierAdjustMessage {
      struct RequestArgs {
	Event event;
	Barrier::timestamp_t timestamp;
	int delta;
        Event wait_on;
      };

      static void handle_request(RequestArgs args)
      {
	args.event.impl()->adjust_arrival(args.event.gen, args.delta, args.timestamp, args.wait_on);
      }

      typedef ActiveMessageShortNoReply<BARRIER_ADJUST_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(gasnet_node_t target, Event event, int delta, Barrier::timestamp_t timestamp, Event wait_on)
      {
	RequestArgs args;
	
	args.event = event;
	args.timestamp = timestamp;
	args.delta = delta;
        args.wait_on = wait_on;

	Message::request(target, args);
      }
    };

    static Logger::Category log_barrier("barrier");

    class DeferredBarrierArrival : public Event::Impl::EventWaiter {
    public:
      DeferredBarrierArrival(Barrier _barrier, int _delta)
	: barrier(_barrier), delta(_delta)
      {}

      virtual ~DeferredBarrierArrival(void) { }

      virtual bool event_triggered(void)
      {
	log_barrier.info("deferred barrier arrival: " IDFMT "/%d (%llx), delta=%d\n",
			 barrier.id, barrier.gen, barrier.timestamp, delta);
	barrier.impl()->adjust_arrival(barrier.gen, delta, barrier.timestamp, Event::NO_EVENT);
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred arrival: barrier=" IDFMT "/%d (%llx), delta=%d\n",
	       barrier.id, barrier.gen, barrier.timestamp, delta);
      }

    protected:
      Barrier barrier;
      int delta;
    };

    class Event::Impl::PendingUpdates {
     public:
      struct PerNodeUpdates {
        Barrier::timestamp_t last_ts;
        std::map<Barrier::timestamp_t, int> pending;
      };

      PendingUpdates(void) : unguarded_delta(0) {}
      ~PendingUpdates(void)
      {
        for(std::map<int, PerNodeUpdates *>::iterator it = pernode.begin();
            it != pernode.end();
            it++)
          delete (it->second);
      }

      int handle_adjustment(Barrier::timestamp_t ts, int delta)
      {
        int delta_out = 0;
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
          delta_out += delta;
          pn->last_ts = ts;
          std::map<Barrier::timestamp_t, int>::iterator it2 = pn->pending.begin();
          while((it2 != pn->pending.end()) && (it2->first <= pn->last_ts)) {
            log_barrier.info("applying pending delta: %llx/%d", it2->first, it2->second);
            delta_out += it2->second;
            pn->pending.erase(it2);
            it2 = pn->pending.begin();
          }
        } else {
          // if the timestamp is late enough, we can apply this directly
          if(ts <= pn->last_ts) {
            log_barrier.info("adjustment can be applied immediately: %llx/%d (%llx)",
                             ts, delta, pn->last_ts);
            delta_out += delta;
          } else {
            log_barrier.info("adjustment must be deferred: %llx/%d (%llx)",
                             ts, delta, pn->last_ts);
            pn->pending[ts] += delta;
          }
        }
        return delta_out;
      }

      int unguarded_delta;

      std::map<int, PerNodeUpdates *> pernode;
    };

    // used to adjust a barrier's arrival count either up or down
    // if delta > 0, timestamp is current time (on requesting node)
    // if delta < 0, timestamp says which positive adjustment this arrival must wait for
    void Event::Impl::adjust_arrival(Event::gen_t barrier_gen, int delta, 
				     Barrier::timestamp_t timestamp, Event wait_on)
    {
      if(!wait_on.has_triggered()) {
	// deferred arrival
#ifndef DEFER_ARRIVALS_LOCALLY
        if(owner != gasnet_mynode()) {
	  // let deferral happen on owner node (saves latency if wait_on event
          //   gets triggered there)
          Event e;
          e.id = me.id;
          e.gen = barrier_gen;
          //printf("sending deferred arrival to %d for " IDFMT "/%d (" IDFMT "/%d)\n",
          //       owner, e.id, e.gen, wait_on.id, wait_on.gen);
	  BarrierAdjustMessage::send_request(owner, e, delta, timestamp, wait_on);
	  return;
        }
#endif
	Barrier b;
	b.id = me.id;
	b.gen = barrier_gen;
	b.timestamp = timestamp;
	log_barrier.info("deferring barrier arrival: delta=%d in=" IDFMT "/%d out=" IDFMT "/%d (%llx)\n",
			 delta, wait_on.id, wait_on.gen, me.id, barrier_gen, timestamp);
	wait_on.impl()->add_waiter(wait_on, new DeferredBarrierArrival(b, delta));
	return;
      }

      log_barrier.info("barrier adjustment: event=" IDFMT "/%d delta=%d ts=%llx", 
		       me.id, barrier_gen, delta, timestamp);

      if(owner != gasnet_mynode()) {
	// all adjustments handled by owner node
        Event e;
        e.id = me.id;
        e.gen = barrier_gen;
	BarrierAdjustMessage::send_request(owner, e, delta, timestamp, Event::NO_EVENT);
	return;
      }

      // can't actually trigger while holding the lock, so remember which generation(s),
      //  if any, to trigger and do it at the end
      gen_t trigger_gen = 0;
      {
	AutoHSLLock a(mutex);

	// sanity checks - is this a valid barrier?
	assert(generation < free_generation);
	assert(base_arrival_count > 0);

	// just handle updates to current generation first
	if(barrier_gen == (generation + 1)) {
          int act_delta = 0;
          if(timestamp == 0) {
            act_delta = delta;
          } else {
            // some ordering is required - check pending updates
            PendingUpdates *p;
            std::map<Event::gen_t, PendingUpdates *>::iterator it = pending_updates.find(barrier_gen);
            if(it != pending_updates.end()) {
              p = it->second;
            } else {
	      p = new PendingUpdates;
              pending_updates[barrier_gen] = p;
            }
            act_delta = p->handle_adjustment(timestamp, delta);
            log_barrier.info("barrier timestamp adjustment: " IDFMT "/%d, %llx/%d -> %d",
                             me.id, barrier_gen, timestamp, delta, act_delta);
          }

	  if(act_delta > 0) {
	    current_arrival_count += act_delta;
	  } else {
	    assert(-act_delta <= current_arrival_count);
	    current_arrival_count += act_delta; // delta is negative
	    if(current_arrival_count == 0) {
	      // mark that we want to trigger this barrier generation
	      trigger_gen = barrier_gen;

	      // and reset the arrival count for the next generation
	      current_arrival_count = base_arrival_count;
	    }
	  }
	} else {
	  // defer updates for future generations until then becomes now
	  assert(0);
	}
      }

      if(trigger_gen != 0) {
	log_barrier.info("barrier trigger: event=" IDFMT "/%d", 
			 me.id, trigger_gen);
	trigger(trigger_gen, gasnet_mynode());
      }
    }

    ///////////////////////////////////////////////////
    // Barrier 

    /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      // start by getting a free event
      Event e = Event::Impl::create_event();

      // now turn it into a barrier
      Event::Impl *impl = e.impl();

      // set the arrival count
      impl->base_arrival_count = expected_arrivals;
      impl->current_arrival_count = expected_arrivals;

      // and let the barrier rearm as many times as necessary without being released
      impl->free_generation = (unsigned)-1;

      log_barrier.info("barrier created: " IDFMT "/%d base_count=%d", e.id, e.gen, impl->base_arrival_count);

      Barrier result;
      result.id = e.id;
      result.gen = e.gen;
      result.timestamp = 0;

      return result;
    }

    void Barrier::destroy_barrier(void)
    {
      // TODO: Implement this
      assert(false);
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
      impl()->adjust_arrival(gen, delta, timestamp);

      Barrier with_ts;
      with_ts.id = id;
      with_ts.gen = gen;
      with_ts.timestamp = timestamp;

      return with_ts;
    }

    Event Barrier::get_previous_phase(void) const
    {
      Event result = *this;
      result.gen--;
      return result;
    }

    void Barrier::arrive(unsigned count /*= 1*/, Event wait_on /*= Event::NO_EVENT*/) const
    {
      // arrival uses the timestamp stored in this barrier object
      impl()->adjust_arrival(gen, -count, timestamp, wait_on);
    }

    ///////////////////////////////////////////////////
    // Reservations 

    /*static*/ const Reservation Reservation::NO_RESERVATION = { 0 };

    Reservation::Impl *Reservation::impl(void) const
    {
      return Runtime::runtime->get_lock_impl(*this);
    }

    /*static*/ Reservation::Impl *Reservation::Impl::first_free = 0;
    /*static*/ gasnet_hsl_t Reservation::Impl::freelist_mutex = GASNET_HSL_INITIALIZER;

    Reservation::Impl::Impl(void)
    {
      init(Reservation::NO_RESERVATION, -1);
    }

    Logger::Category log_reservation("reservation");

    void Reservation::Impl::init(Reservation _me, unsigned _init_owner,
			  size_t _data_size /*= 0*/)
    {
      me = _me;
      owner = _init_owner;
      count = ZERO_COUNT;
      log_reservation.spew("count init " IDFMT "=[%p]=%d", me.id, &count, count);
      mode = 0;
      in_use = false;
      mutex = new gasnet_hsl_t;
      gasnet_hsl_init(mutex);
      remote_waiter_mask = NodeMask(); 
      remote_sharer_mask = NodeMask();
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

    /*static*/ void /*Reservation::Impl::*/handle_lock_request(LockRequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Reservation::Impl *impl = args.lock.impl();

      log_reservation(LEVEL_DEBUG, "reservation request: reservation=" IDFMT ", node=%d, mode=%d",
	       args.lock.id, args.node, args.mode);

      // can't send messages while holding mutex, so remember args and who
      //  (if anyone) to send to
      int req_forward_target = -1;
      int grant_target = -1;
      LockGrantArgs g_args;
      NodeMask copy_waiters;

      do {
	AutoHSLLock a(impl->mutex);

	// case 1: we don't even own the lock any more - pass the request on
	//  to whoever we think the owner is
	if(impl->owner != gasnet_mynode()) {
	  // can reuse the args we were given
	  log_reservation(LEVEL_DEBUG, 
              "forwarding reservation request: reservation=" IDFMT ", from=%d, to=%d, mode=%d",
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
	if((impl->count == Reservation::Impl::ZERO_COUNT) && 
           (!impl->remote_sharer_mask)) {
          assert(!impl->remote_waiter_mask);

	  log_reservation(LEVEL_DEBUG, 
              "granting reservation request: reservation=" IDFMT ", node=%d, mode=%d",
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
	log_reservation(LEVEL_DEBUG, 
            "deferring reservation request: reservation=" IDFMT ", node=%d, mode=%d (count=%d cmode=%d)",
		 args.lock.id, args.node, args.mode, impl->count, impl->mode);
        impl->remote_waiter_mask.set_bit(args.node);
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
        size_t payload_size = sizeof(copy_waiters) + impl->local_data_size;
        NodeMask *payload = (NodeMask*)malloc(payload_size);
        *payload = copy_waiters; 
        memcpy(payload+1,impl->local_data,impl->local_data_size);
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

    /*static*/ void /*Reservation::Impl::*/handle_lock_release(LockReleaseArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      assert(0);
    }

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_reservation(LEVEL_DEBUG, 
          "reservation request granted: reservation=" IDFMT " mode=%d", // mask=%lx",
	       args.lock.id, args.mode); //, args.remote_waiter_mask);

      std::deque<Event> to_wake;

      Reservation::Impl *impl = args.lock.impl();
      {
	AutoHSLLock a(impl->mutex);

	// make sure we were really waiting for this lock
	assert(impl->owner != gasnet_mynode());
	assert(impl->requested);

	// first, update our copy of the protected data (if any)
	assert((impl->local_data_size+sizeof(impl->remote_waiter_mask)) == datalen);
        impl->remote_waiter_mask = *((const NodeMask*)data);
        if (datalen > sizeof(impl->remote_waiter_mask))
          memcpy(impl->local_data, ((const NodeMask*)data)+1, impl->local_data_size);

	if(args.mode == 0) // take ownership if given exclusive access
	  impl->owner = gasnet_mynode();
	impl->mode = args.mode;
	impl->requested = false;

	bool any_local = impl->select_local_waiters(to_wake);
	assert(any_local);
      }

      for(std::deque<Event>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_reservation(LEVEL_DEBUG, "release trigger: reservation=" IDFMT " event=" IDFMT "/%d",
		 args.lock.id, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, gasnet_mynode());
      }
    }

    Event Reservation::Impl::acquire(unsigned new_mode, bool exclusive,
			   Event after_lock /*= Event::NO_EVENT*/)
    {
      log_reservation(LEVEL_DEBUG, 
          "local reservation request: reservation=" IDFMT " mode=%d excl=%d event=" IDFMT "/%d count=%d impl=%p",
	       me.id, new_mode, exclusive, after_lock.id, after_lock.gen, count, this);

      // collapse exclusivity into mode
      if(exclusive) new_mode = MODE_EXCL;

      bool got_lock = false;
      int lock_request_target = -1;
      LockRequestArgs args;

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
	    log_reservation(LEVEL_DEBUG, 
                "requesting reservation: reservation=" IDFMT " node=%d mode=%d",
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
  
	log_reservation(LEVEL_DEBUG, 
            "local reservation result: reservation=" IDFMT " got=%d req=%d count=%d",
		 me.id, got_lock ? 1 : 0, requested ? 1 : 0, count);

	// if we didn't get the lock, put our event on the queue of local
	//  waiters - create an event if we weren't given one to use
	if(!got_lock) {
	  if(!after_lock.exists())
	    after_lock = Event::Impl::create_event();
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
      if(got_lock && after_lock.exists()) 
	after_lock.impl()->trigger(after_lock.gen, gasnet_mynode());

      return after_lock;
    }

    // factored-out code to select one or more local waiters on a lock
    //  fills events to trigger into 'to_wake' and returns true if any were
    //  found - NOTE: ASSUMES LOCK IS ALREADY HELD!
    bool Reservation::Impl::select_local_waiters(std::deque<Event>& to_wake)
    {
      if(local_waiters.size() == 0)
	return false;

      // favor the local waiters
      log_reservation(LEVEL_DEBUG, "reservation going to local waiter: size=%zd first=%d(%zd)",
	       local_waiters.size(), 
	       local_waiters.begin()->first,
	       local_waiters.begin()->second.size());
	
      // further favor exclusive waiters
      if(local_waiters.find(MODE_EXCL) != local_waiters.end()) {
	std::deque<Event>& excl_waiters = local_waiters[MODE_EXCL];
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
	std::map<unsigned, std::deque<Event> >::iterator it = local_waiters.begin();
	
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

    void Reservation::Impl::release(void)
    {
      // make a list of events that we be woken - can't do it while holding the
      //  lock's mutex (because the event we trigger might try to take the lock)
      std::deque<Event> to_wake;

      int release_target = -1;
      LockReleaseArgs r_args;

      int grant_target = -1;
      LockGrantArgs g_args;
      NodeMask copy_waiters;

      do {
	log_reservation(LEVEL_DEBUG, 
            "release: reservation=" IDFMT " count=%d mode=%d owner=%d", // share=%lx wait=%lx",
			me.id, count, mode, owner); //, remote_sharer_mask, remote_waiter_mask);
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	assert(count > ZERO_COUNT);

	// if this isn't the last holder of the lock, just decrement count
	//  and return
	count--;
	log_reservation.spew("count -- [%p]=%d", &count, count);
	log_reservation(LEVEL_DEBUG, 
            "post-release: reservation=" IDFMT " count=%d mode=%d", // share=%lx wait=%lx",
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

	if(!any_local && (!!remote_waiter_mask)) {
	  // nobody local wants it, but another node does
	  int new_owner = remote_waiter_mask.find_first_set();
          remote_waiter_mask.unset_bit(new_owner);

	  log_reservation(LEVEL_DEBUG, 
              "reservation going to remote waiter: new=%d", // mask=%lx",
		   new_owner); //, remote_waiter_mask);

	  g_args.lock = me;
	  g_args.mode = 0; // TODO: figure out shared cases
	  grant_target = new_owner;
          copy_waiters = remote_waiter_mask;

	  owner = new_owner;
          remote_waiter_mask = NodeMask();
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
        size_t payload_size = sizeof(copy_waiters) + local_data_size;
        NodeMask *payload = (NodeMask*)malloc(payload_size);
        *payload = copy_waiters;
        memcpy(payload+1,local_data,local_data_size);
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

      for(std::deque<Event>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_reservation(LEVEL_DEBUG, "release trigger: reservation=" IDFMT " event=" IDFMT "/%d",
		 me.id, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, gasnet_mynode());
      }
    }

    bool Reservation::Impl::is_locked(unsigned check_mode, bool excl_ok)
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

    class DeferredLockRequest : public Event::Impl::EventWaiter {
    public:
      DeferredLockRequest(Reservation _lock, unsigned _mode, bool _exclusive,
			  Event _after_lock)
	: lock(_lock), mode(_mode), exclusive(_exclusive), after_lock(_after_lock) {}

      virtual ~DeferredLockRequest(void) { }

      virtual bool event_triggered(void)
      {
	lock.impl()->acquire(mode, exclusive, after_lock);
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred lock: lock=" IDFMT " after=" IDFMT "/%d\n",
	       lock.id, after_lock.id, after_lock.gen);
      }

    protected:
      Reservation lock;
      unsigned mode;
      bool exclusive;
      Event after_lock;
    };

    Event Reservation::acquire(unsigned mode /* = 0 */, bool exclusive /* = true */,
		     Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //printf("LOCK(" IDFMT ", %d, %d, " IDFMT ") -> ", id, mode, exclusive, wait_on.id);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	Event e = impl()->acquire(mode, exclusive);
	//printf("(" IDFMT "/%d)\n", e.id, e.gen);
	return e;
      } else {
	Event after_lock = Event::Impl::create_event();
	wait_on.impl()->add_waiter(wait_on, new DeferredLockRequest(*this, mode, exclusive, after_lock));
	//printf("*(" IDFMT "/%d)\n", after_lock.id, after_lock.gen);
	return after_lock;
      }
    }

    class DeferredUnlockRequest : public Event::Impl::EventWaiter {
    public:
      DeferredUnlockRequest(Reservation _lock)
	: lock(_lock) {}

      virtual ~DeferredUnlockRequest(void) { }

      virtual bool event_triggered(void)
      {
	lock.impl()->release();
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

    // releases a held lock - release can be deferred until an event triggers
    void Reservation::release(Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	impl()->release();
      } else {
	wait_on.impl()->add_waiter(wait_on, new DeferredUnlockRequest(*this));
      }
    }

    // Create a new lock, destroy an existing lock
    /*static*/ Reservation Reservation::create_reservation(size_t _data_size /*= 0*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //DetailedTimer::ScopedPush sp(18);

      // see if the freelist has an event we can reuse
      Reservation::Impl *impl = Runtime::runtime->local_reservation_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).type() == ID::ID_LOCK);
      if(impl) {
	AutoHSLLock al(impl->mutex);

	assert(impl->owner == gasnet_mynode());
	assert(impl->count == Impl::ZERO_COUNT);
	assert(impl->mode == Impl::MODE_EXCL);
	assert(impl->local_waiters.size() == 0);
        assert(!impl->remote_waiter_mask);
	assert(!impl->in_use);

	impl->in_use = true;

	log_reservation(LEVEL_INFO, "reservation reused: reservation=" IDFMT "", impl->me.id);
	return impl->me;
      }
      assert(false);
      return Reservation::NO_RESERVATION;
#if 0
      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

      std::vector<Reservation::Impl>& locks = 
        Runtime::runtime->nodes[gasnet_mynode()].locks;

#ifdef REUSE_LOCKS
      // try to find an lock we can reuse
      for(std::vector<Reservation::Impl>::iterator it = locks.begin();
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
      Runtime::runtime->nodes[gasnet_mynode()].num_locks = index + 1;
      log_reservation.info("created new reservation: reservation=" IDFMT "", r.id);
      return r;
#endif
    }

    void Reservation::Impl::release_reservation(void)
    {
      // take the lock's mutex to sanity check it and clear the in_use field
      {
	AutoHSLLock al(mutex);

	// should only get here if the current node holds an exclusive lock
	assert(owner == gasnet_mynode());
	assert(count == 1 + ZERO_COUNT);
	assert(mode == MODE_EXCL);
	assert(local_waiters.size() == 0);
        assert(!remote_waiter_mask);
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
      log_reservation.info("releasing reservation: reservation=" IDFMT "", me.id);

      Runtime::runtime->local_reservation_free_list->free_entry(this);
    }

    class DeferredLockDestruction : public Event::Impl::EventWaiter {
    public:
      DeferredLockDestruction(Reservation _lock) : lock(_lock) {}

      virtual ~DeferredLockDestruction(void) { }

      virtual bool event_triggered(void)
      {
	lock.impl()->release_reservation();
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred lock destruction: lock=" IDFMT "\n", lock.id);
      }

    protected:
      Reservation lock;
    };

    void handle_destroy_lock(Reservation lock)
    {
      lock.destroy_reservation();
    }

    typedef ActiveMessageShortNoReply<DESTROY_LOCK_MSGID, Reservation,
				      handle_destroy_lock> DestroyLockMessage;

    void Reservation::destroy_reservation()
    {
      // a lock has to be destroyed on the node that created it
      if(ID(*this).node() != gasnet_mynode()) {
	DestroyLockMessage::request(ID(*this).node(), *this);
	return;
      }

      // to destroy a local lock, we first must lock it (exclusively)
      Event e = acquire(0, true);
      if(e.has_triggered()) {
	impl()->release_reservation();
      } else {
	e.impl()->add_waiter(e, new DeferredLockDestruction(*this));
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

    Memory::Impl *Memory::impl(void) const
    {
      return Runtime::runtime->get_memory_impl(*this);
    }

    /*static*/ const Memory Memory::NO_MEMORY = { 0 };

    struct RemoteMemAllocArgs {
      Memory memory;
      size_t size;
    };

    off_t handle_remote_mem_alloc(RemoteMemAllocArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //printf("[%d] handling remote alloc of size %zd\n", gasnet_mynode(), args.size);
      off_t result = args.memory.impl()->alloc_bytes(args.size);
      //printf("[%d] remote alloc will return %d\n", gasnet_mynode(), result);
      return result;
    }

    typedef ActiveMessageShortReply<REMOTE_MALLOC_MSGID, REMOTE_MALLOC_RPLID,
				    RemoteMemAllocArgs, off_t,
				    handle_remote_mem_alloc> RemoteMemAllocMessage;

    // make bad offsets really obvious (+1 PB)
    static const off_t ZERO_SIZE_INSTANCE_OFFSET = 1ULL << 50;

    off_t Memory::Impl::alloc_bytes_local(size_t size)
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

      for(std::map<off_t, off_t>::iterator it = free_blocks.begin();
	  it != free_blocks.end();
	  it++) {
	if(it->second == (off_t)size) {
	  // perfect match
	  off_t retval = it->first;
	  free_blocks.erase(it);
	  log_malloc.info("alloc full block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, retval);
	  return retval;
	}
	
	if(it->second > (off_t)size) {
	  // some left over
	  off_t leftover = it->second - size;
	  off_t retval = it->first + leftover;
	  it->second = leftover;
	  log_malloc.info("alloc partial block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, retval);
	  return retval;
	}
      }

      // no blocks large enough - boo hoo
      log_malloc.info("alloc FAILED: mem=" IDFMT " size=%zd", me.id, size);
      return -1;
    }

    void Memory::Impl::free_bytes_local(off_t offset, size_t size)
    {
      log_malloc.info("free block: mem=" IDFMT " size=%zd ofs=%zd", me.id, size, offset);
      AutoHSLLock al(mutex);

      // frees of zero bytes should have the special offset
      if(size == 0) {
	assert(offset == this->size + ZERO_SIZE_INSTANCE_OFFSET);
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

    off_t Memory::Impl::alloc_bytes_remote(size_t size)
    {
      // RPC over to owner's node for allocation

      RemoteMemAllocArgs args;
      args.memory = me;
      args.size = size;
      off_t retval = RemoteMemAllocMessage::request(ID(me).node(), args);
      //printf("got: %d\n", retval);
      return retval;
    }

    void Memory::Impl::free_bytes_remote(off_t offset, size_t size)
    {
      assert(0);
    }

    Memory::Kind Memory::Impl::get_kind(void) const
    {
      return lowlevel_kind;
    }

    static Logger::Category log_copy("copy");

    class LocalCPUMemory : public Memory::Impl {
    public:
      static const size_t ALIGNMENT = 256;

      LocalCPUMemory(Memory _me, size_t _size,
		     void *prealloc_base = 0, bool _registered = false) 
	: Memory::Impl(_me, _size, MKIND_SYSMEM, ALIGNMENT, 
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
					     RegionInstance parent_inst)
      {
	return create_instance_local(r, linearization_bits, bytes_needed,
				     block_size, element_size, field_sizes, redopid,
				     list_size, parent_inst);
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

    class RemoteMemory : public Memory::Impl {
    public:
      RemoteMemory(Memory _me, size_t _size, Memory::Kind k, void *_regbase)
	: Memory::Impl(_me, _size, _regbase ? MKIND_RDMA : MKIND_REMOTE, 0, k), regbase(_regbase)
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
					     RegionInstance parent_inst)
      {
	return create_instance_remote(r, linearization_bits, bytes_needed,
				      block_size, element_size, field_sizes, redopid,
				      list_size, parent_inst);
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
	// can't read/write a remote memory
	assert(0);
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
    static gasnet_hsl_t partial_remote_writes_lock = GASNET_HSL_INITIALIZER;

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
      Memory::Impl *impl = args.mem.impl();

      log_copy.debug("received remote write request: mem=" IDFMT ", offset=%zd, size=%zd, seq=%d/%d, event=" IDFMT "/%d",
		     args.mem.id, args.offset, datalen,
		     args.sender, args.sequence_id,
		     args.event.id, args.event.gen);
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
      case Memory::Impl::MKIND_SYSMEM:
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
	    args.event.impl()->trigger(args.event.gen,
				       gasnet_mynode());
	  break;
	}

      case Memory::Impl::MKIND_ZEROCOPY:
#ifdef USE_CUDA
      case Memory::Impl::MKIND_GPUFB:
#endif
	{
	  impl->put_bytes(args.offset, data, datalen);

	  if(args.event.exists())
	    args.event.impl()->trigger(args.event.gen,
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
        gasnet_hsl_lock(&partial_remote_writes_lock);
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
	    gasnet_hsl_unlock(&partial_remote_writes_lock);
	    if(e.exists())
	      e.impl()->trigger(e.gen, gasnet_mynode());
	    return;
	  }
	}
        gasnet_hsl_unlock(&partial_remote_writes_lock);
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
      Memory::Impl *impl = args.mem.impl();

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
      }

      log_copy.debug("received remote reduce request: mem=" IDFMT ", offset=%zd+%d, size=%zd, redop=%d(%s), seq=%d/%d, event=" IDFMT "/%d",
		     args.mem.id, args.offset, args.stride, datalen,
		     redop_id, (red_fold ? "fold" : "apply"),
		     args.sender, args.sequence_id,
		     args.event.id, args.event.gen);

      const ReductionOpUntyped *redop = reduce_op_table[redop_id];

      size_t count = datalen / redop->sizeof_rhs;

      void *lhs = args.mem.impl()->get_direct_ptr(args.offset, args.stride * count);
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
        gasnet_hsl_lock(&partial_remote_writes_lock);
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
	    gasnet_hsl_unlock(&partial_remote_writes_lock);
	    if(e.exists())
	      e.impl()->trigger(e.gen, gasnet_mynode());
	    return;
	  }
	}
        gasnet_hsl_unlock(&partial_remote_writes_lock);
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
        gasnet_hsl_lock(&partial_remote_writes_lock);
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
	    gasnet_hsl_unlock(&partial_remote_writes_lock);
	    if(e.exists())
	      e.impl()->trigger(e.gen, gasnet_mynode());
	    return;
	  }
	}
        gasnet_hsl_unlock(&partial_remote_writes_lock);
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

      Memory::Impl *m_impl = mem.impl();
      char *dstptr;
      if(m_impl->kind == Memory::Impl::MKIND_RDMA) {
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

      Memory::Impl *m_impl = mem.impl();
      char *dstptr;
      if(m_impl->kind == Memory::Impl::MKIND_RDMA) {
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

      Memory::Impl *m_impl = mem.impl();
      char *dstptr;
      if(m_impl->kind == Memory::Impl::MKIND_RDMA) {
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
      const ReductionOpUntyped *redop = reduce_op_table[redop_id];
      size_t rhs_size = redop->sizeof_rhs;

      log_copy.debug("sending remote reduction request: mem=" IDFMT ", offset=%zd+%zd, size=%zdx%zd, redop=%d(%s), event=" IDFMT "/%d",
		     mem.id, offset, dst_stride, rhs_size, count,
		     redop_id, (red_fold ? "fold" : "apply"),
		     event.id, event.gen);

      Memory::Impl *m_impl = mem.impl();

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
      : Memory::Impl(_me, 0 /* we'll calculate it below */, MKIND_GLOBAL,
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
						 RegionInstance parent_inst)
    {
      if(gasnet_mynode() == 0) {
	return create_instance_local(r, linearization_bits, bytes_needed,
				     block_size, element_size, field_sizes, redopid,
				     list_size, parent_inst);
      } else {
	return create_instance_remote(r, linearization_bits, bytes_needed,
				      block_size, element_size, field_sizes, redopid,
				      list_size, parent_inst);
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

    RegionInstance Memory::Impl::create_instance_local(IndexSpace r,
						       const int *linearization_bits,
						       size_t bytes_needed,
						       size_t block_size,
						       size_t element_size,
						       const std::vector<size_t>& field_sizes,
						       ReductionOpID redopid,
						       off_t list_size,
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
      else if (redopid > 0)
      {
        assert(field_sizes.size() == 1);
        // Otherwise if this is a fold reduction instance then
        // we need to initialize the memory with the identity
        const ReductionOpUntyped *redop = reduce_op_table[redopid];
        assert(redop->has_identity);
        assert(element_size == redop->sizeof_rhs);
        void *ptr = get_direct_ptr(inst_offset, bytes_needed); 
        size_t num_elements = bytes_needed/element_size;
        redop->init(ptr, num_elements);
      }

      // SJT: think about this more to see if there are any race conditions
      //  with an allocator temporarily having the wrong ID
      RegionInstance i = ID(ID::ID_INSTANCE, 
				   ID(me).node(),
				   ID(me).index_h(),
				   0).convert<RegionInstance>();


      //RegionMetaDataImpl *r_impl = Runtime::runtime->get_metadata_impl(r);
      DomainLinearization linear;
      linear.deserialize(linearization_bits);

      RegionInstance::Impl *i_impl = new RegionInstance::Impl(i, r, me, inst_offset, bytes_needed, redopid,
							      linear, block_size, element_size, field_sizes,
							      count_offset, list_size, parent_inst);

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

    struct CreateInstanceArgs : public BaseMedium {
      Memory m;
      IndexSpace r;
      RegionInstance parent_inst;
    };

    struct CreateInstancePayload {
      size_t bytes_needed;
      size_t block_size;
      size_t element_size;
      //off_t adjust;
      off_t list_size;
      ReductionOpID redopid;
      int linearization_bits[RegionInstance::Impl::MAX_LINEARIZATION_LEN];
      size_t num_fields; // as long as it needs to be
      const size_t &field_size(int idx) const { return *((&num_fields)+idx+1); }
      size_t &field_size(int idx) { return *((&num_fields)+idx+1); }
    };

    struct CreateInstanceResp : public BaseReply {
      RegionInstance i;
      off_t inst_offset;
      off_t count_offset;
    };

    CreateInstanceResp handle_create_instance(CreateInstanceArgs args, const void *msgdata, size_t msglen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      CreateInstanceResp resp;

      const CreateInstancePayload *payload = (const CreateInstancePayload *)msgdata;
      assert(msglen == (sizeof(CreateInstancePayload) + sizeof(size_t) * payload->num_fields));

      std::vector<size_t> field_sizes(payload->num_fields);
      for(int i = 0; i < payload->num_fields; i++)
	field_sizes[i] = payload->field_size(i);

      resp.i = args.m.impl()->create_instance(args.r, 
					      payload->linearization_bits,
					      payload->bytes_needed,
					      payload->block_size,
					      payload->element_size,
					      field_sizes,
					      payload->redopid,
					      payload->list_size,
					      args.parent_inst);

      //resp.inst_offset = resp.i.impl()->locked_data.offset; // TODO: Static
      StaticAccess<RegionInstance::Impl> i_data(resp.i.impl());
      resp.inst_offset = i_data->alloc_offset;
      resp.count_offset = i_data->count_offset;
      //resp.inst_offset = StaticAccess<RegionInstance::Impl>(resp.i.impl())->alloc_offset;
      return resp;
    }

    typedef ActiveMessageMediumReply<CREATE_INST_MSGID, CREATE_INST_RPLID,
				     CreateInstanceArgs, CreateInstanceResp,
				     handle_create_instance> CreateInstanceMessage;

    RegionInstance Memory::Impl::create_instance_remote(IndexSpace r,
							const int *linearization_bits,
							size_t bytes_needed,
							size_t block_size,
							size_t element_size,
							const std::vector<size_t>& field_sizes,
							ReductionOpID redopid,
							off_t list_size,
							RegionInstance parent_inst)
    {
      size_t payload_size = sizeof(CreateInstancePayload) + sizeof(size_t) * field_sizes.size();
      CreateInstancePayload *payload = (CreateInstancePayload *)alloca(payload_size);

      payload->bytes_needed = bytes_needed;
      payload->block_size = block_size;
      payload->element_size = element_size;
      //payload->adjust = ?
      payload->list_size = list_size;
      payload->redopid = redopid;

      for(unsigned i = 0; i < RegionInstance::Impl::MAX_LINEARIZATION_LEN; i++)
	payload->linearization_bits[i] = linearization_bits[i];

      payload->num_fields = field_sizes.size();
      for(unsigned i = 0; i < field_sizes.size(); i++)
	payload->field_size(i) = field_sizes[i];

      CreateInstanceArgs args;
      args.m = me;
      args.r = r;
      args.parent_inst = parent_inst;
      log_inst(LEVEL_DEBUG, "creating remote instance: node=%d", ID(me).node());
      CreateInstanceResp resp = CreateInstanceMessage::request(ID(me).node(), args,
							       payload, payload_size, PAYLOAD_COPY);
      log_inst(LEVEL_DEBUG, "created remote instance: inst=" IDFMT " offset=%zd", resp.i.id, resp.inst_offset);

      DomainLinearization linear;
      linear.deserialize(linearization_bits);

      RegionInstance::Impl *i_impl = new RegionInstance::Impl(resp.i, r, me, resp.inst_offset, bytes_needed, redopid,
							      linear, block_size, element_size, field_sizes,
							      resp.count_offset, list_size, parent_inst);

      unsigned index = ID(resp.i).index_l();
      // resize array if needed
      if(index >= instances.size()) {
	AutoHSLLock a(mutex);
	if(index >= instances.size()) {
	  log_inst(LEVEL_DEBUG, "resizing instance array: mem=" IDFMT " old=%zd new=%d",
		   me.id, instances.size(), index+1);
	  for(unsigned i = instances.size(); i <= index; i++)
	    instances.push_back(0);
	}
      }
      instances[index] = i_impl;
      return resp.i;
    }

    RegionInstance::Impl *Memory::Impl::get_instance(RegionInstance i)
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

    void Memory::Impl::destroy_instance_local(RegionInstance i, 
					      bool local_destroy)
    {
      log_inst(LEVEL_INFO, "destroying local instance: mem=" IDFMT " inst=" IDFMT "", me.id, i.id);

      // all we do for now is free the actual data storage
      unsigned index = ID(i).index_l();
      assert(index < instances.size());
      RegionInstance::Impl *iimpl = instances[index];
      free_bytes(iimpl->locked_data.alloc_offset, iimpl->locked_data.size);

      if(iimpl->locked_data.count_offset >= 0)
	free_bytes(iimpl->locked_data.count_offset, sizeof(size_t));
      
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
      args.m.impl()->destroy_instance(args.i, false);
    }

    typedef ActiveMessageShortNoReply<DESTROY_INST_MSGID,
				      DestroyInstanceArgs,
				      handle_destroy_instance> DestroyInstanceMessage;

    void Memory::Impl::destroy_instance_remote(RegionInstance i, 
					       bool local_destroy)
    {
      // if we're the original destroyer of the instance, tell the owner
      if(local_destroy) {
	int owner = ID(me).node();

	DestroyInstanceArgs args;
	args.m = me;
	args.i = i;
	log_inst(LEVEL_DEBUG, "destroying remote instance: node=%d inst=" IDFMT "", owner, i.id);

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
      : proc(_proc), func_id(_func_id), arglen(_arglen),
	finish_event(_finish_event), priority(_priority),
        run_count(0), finish_count(expected_count)
    {
      if(arglen) {
	args = malloc(arglen);
	memcpy(args, _args, arglen);
      } else
	args = 0;
    }

    Task::~Task(void)
    {
      free(args);
    }

    class DeferredTaskSpawn : public Event::Impl::EventWaiter {
    public:
      DeferredTaskSpawn(Processor::Impl *_proc, Task *_task) 
        : proc(_proc), task(_task) {}

      virtual ~DeferredTaskSpawn(void)
      {
        // we do _NOT_ own the task - do not free it
      }

      virtual bool event_triggered(void)
      {
        log_task(LEVEL_DEBUG, "deferred task now ready: func=%d finish=" IDFMT "/%d",
                 task->func_id, 
                 task->finish_event.id, task->finish_event.gen);

        proc->enqueue_task(task);

        return true;
      }

      virtual void print_info(FILE *f)
      {
        fprintf(f,"deferred task: func=%d proc=" IDFMT " finish=" IDFMT "/%d\n",
               task->func_id, task->proc.id, task->finish_event.id, task->finish_event.gen);
      }

    protected:
      Processor::Impl *proc;
      Task *task;
    };

    ///////////////////////////////////////////////////
    // Processor

    // global because I'm being lazy...
    Processor::TaskIDTable task_id_table;

    /*static*/ const Processor Processor::NO_PROC = { 0 };

    Processor::Impl *Processor::impl(void) const
    {
      return Runtime::runtime->get_processor_impl(*this);
    }

    Processor::Impl::Impl(Processor _me, Processor::Kind _kind, Processor _util /*= Processor::NO_PROC*/)
      : me(_me), kind(_kind), util(_util), util_proc(0), run_counter(0)
    {
    }

    Processor::Impl::~Impl(void)
    {
    }

    void Processor::Impl::set_utility_processor(UtilityProcessor *_util_proc)
    {
      if(is_idle_task_enabled()) {
	log_util.info("delegating idle task handling for " IDFMT " to " IDFMT "",
		      me.id, util.id);
	disable_idle_task();
	_util_proc->enable_idle_task(this);
      }

      util_proc = _util_proc;
      util = util_proc->me;
    }

    ProcessorGroup::ProcessorGroup(void)
      : Processor::Impl(Processor::NO_PROC, Processor::PROC_GROUP),
	members_valid(false), members_requested(false), next_free(0)
    {
    }

    ProcessorGroup::~ProcessorGroup(void)
    {
    }

    void ProcessorGroup::init(Processor _me, int _owner)
    {
      assert(ID(_me).node() == _owner);

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
	Processor::Impl *m_impl = (*it).impl();
	members.push_back(m_impl);
      }

      members_requested = true;
      members_valid = true;
    }

    void ProcessorGroup::get_group_members(std::vector<Processor>& member_list)
    {
      assert(members_valid);

      for(std::vector<Processor::Impl *>::const_iterator it = members.begin();
	  it != members.end();
	  it++)
	member_list.push_back((*it)->me);
    }

    void ProcessorGroup::tasks_available(int priority)
    {
      // wake up everybody for now
      for(std::vector<Processor::Impl *>::const_iterator it = members.begin();
	  it != members.end();
	  it++)
	(*it)->tasks_available(priority);
    }

    void ProcessorGroup::enqueue_task(Task *task)
    {
      for (std::vector<Processor::Impl *>::const_iterator it = members.begin();
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
        start_event.impl()->add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }

    class LocalProcessor : public Processor::Impl {
    public:
#if 0
      // simple task object keeps a copy of args
      class Task {
      public:
	Task(LocalProcessor *_proc,
	     Processor::TaskFuncID _func_id,
	     const void *_args, size_t _arglen,
	     Event _finish_event, int _priority)
	  : proc(_proc), func_id(_func_id), arglen(_arglen),
	    finish_event(_finish_event), priority(_priority)
	{
	  if(arglen) {
	    args = malloc(arglen);
	    memcpy(args, _args, arglen);
	  } else {
	    args = 0;
	  }
	}

	virtual ~Task(void)
	{
	  if(args) free(args);
	}

	void run(Processor actual_proc = Processor::NO_PROC)
	{
	  Processor::TaskFuncPtr fptr = task_id_table[func_id];
	  char argstr[100];
	  argstr[0] = 0;
	  for(size_t i = 0; (i < arglen) && (i < 40); i++)
	    sprintf(argstr+2*i, "%02x", ((unsigned char *)args)[i]);
	  if(arglen > 40) strcpy(argstr+80, "...");
	  log_task(((func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
		   "task start: %d (%p) (%s)", func_id, fptr, argstr);
	  (*fptr)(args, arglen, (actual_proc.exists() ? actual_proc : proc->me));
	  log_task(((func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
		   "task end: %d (%p) (%s)", func_id, fptr, argstr);
	  if(finish_event.exists())
	    finish_event.impl()->trigger(finish_event.gen, gasnet_mynode());
	}

	LocalProcessor *proc;
	Processor::TaskFuncID func_id;
	void *args;
	size_t arglen;
	Event finish_event;
        int priority;
      };
#endif

      // simple thread object just has a task field that you can set and 
      //  wake up to run
      class Thread : public PreemptableThread {
      public:
	class EventStackEntry : public Event::Impl::EventWaiter {
	public:
	  EventStackEntry(Thread *_thread, Event _event, EventStackEntry *_next)
	    : thread(_thread), event(_event), next(_next), triggered(false) {}

          virtual ~EventStackEntry(void) { }

	  void add_waiter(void)
	  {
	    log_task.info("thread %p registering a listener on event " IDFMT "/%d",
			  thread, event.id, event.gen);
	    if(event.has_triggered())
	      triggered = true;
	    else
	      event.impl()->add_waiter(event, this);
	  }

	  virtual bool event_triggered(void)
	  {
	    log_task.info("thread %p notified for event " IDFMT "/%d",
			  thread, event.id, event.gen);

	    // now take the thread's (really the processor's) lock and see if
	    //  this thread is asleep on this event - if so, make him 
	    //  resumable and possibly restart him (or somebody else)
	    AutoHSLLock al(thread->proc->mutex);

	    if(thread->event_stack == this) {
	      if(thread->state == STATE_PREEMPTABLE) {
		thread->state = STATE_RESUMABLE;
		thread->proc->preemptable_threads.remove(thread);
		thread->proc->resumable_threads.push_back(thread);
		thread->proc->start_some_threads();
	      }

	      if(thread->state == STATE_BLOCKED) {
		thread->state = STATE_RESUMABLE;
		thread->proc->resumable_threads.push_back(thread);
		thread->proc->start_some_threads();
	      }
	    }
            // This has to go last to avoid a race condition
            // on deleting this object.
	    triggered = true;
            // don't have caller delete us
            return false;
	  }

	  virtual void print_info(FILE *f)
	  {
	    fprintf(f,"thread %p waiting on " IDFMT "/%d\n",
		   thread, event.id, event.gen);
	  }

	  Thread *thread;
	  Event event;
	  EventStackEntry *next;
	  bool triggered;
	};

      public:
	enum State { STATE_INIT, STATE_INIT_WAIT, STATE_START, STATE_IDLE, 
		     STATE_RUN, STATE_SUSPEND, STATE_PREEMPTABLE,
		     STATE_BLOCKED, STATE_RESUMABLE };

	LocalProcessor *proc;
	State state;
	EventStackEntry *event_stack;
	gasnett_cond_t condvar;

	Thread(LocalProcessor *_proc) 
	  : proc(_proc), state(STATE_INIT), event_stack(0)
	{
	  gasnett_cond_init(&condvar);
	}

	~Thread(void) {}

	void run_task(Task *task, Processor actual_proc = Processor::NO_PROC)
	{
          Processor::TaskFuncPtr fptr = task_id_table[task->func_id];
          char argstr[100];
          argstr[0] = 0;
          for(size_t i = 0; (i < task->arglen) && (i < 40); i++)
            sprintf(argstr+2*i, "%02x", ((unsigned char *)(task->args))[i]);
          if(task->arglen > 40) strcpy(argstr+80, "...");
          log_task(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
                   "task start: %d (%p) (%s)", task->func_id, fptr, argstr);
          (*fptr)(task->args, task->arglen, (actual_proc.exists() ? actual_proc : task->proc));
          log_task(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
                   "task end: %d (%p) (%s)", task->func_id, fptr, argstr);
          if(task->finish_event.exists())
            task->finish_event.impl()->trigger(task->finish_event.gen, gasnet_mynode());
	}

	virtual void sleep_on_event(Event wait_for, bool block = false)
	{
	  // create an entry to go on our event stack (on our stack)
	  EventStackEntry *entry = new EventStackEntry(this, wait_for, event_stack);
	  event_stack = entry;

	  entry->add_waiter();

	  // now take the processor lock that controls everything and see
	  //  what we can do while we're waiting
	  {
	    AutoHSLLock al(proc->mutex);

	    log_task.info("thread %p (proc " IDFMT ") needs to sleep on event " IDFMT "/%d",
			  this, proc->me.id, wait_for.id, wait_for.gen);
	    
	    assert(state == STATE_RUN);

	    // loop until our event has triggered
	    while(!entry->triggered) {
	      // first step - if some other thread is resumable, give up
	      //   our spot and let him run
	      if(proc->resumable_threads.size() > 0) {
		Thread *resumee = proc->resumable_threads.front();
		proc->resumable_threads.pop_front();
		assert(resumee->state == STATE_RESUMABLE);

		log_task.info("waiting thread %p yielding to thread %p for proc " IDFMT "",
			      this, resumee, proc->me.id);

		resumee->state = STATE_RUN;
		gasnett_cond_signal(&resumee->condvar);

		proc->preemptable_threads.push_back(this);
		state = STATE_PREEMPTABLE;
		gasnett_cond_wait(&condvar, &proc->mutex.lock);
		assert(state == STATE_RUN);
		continue;
	      }

	      // plan B - if there's a ready task, we can run it while
	      //   we're waiting (unless 'block' is set)
	      if(!block) {
                if(!proc->task_queue.empty()) {
		  Task *newtask = proc->task_queue.pop();
		  log_task.info("thread %p (proc " IDFMT ") running task %p instead of sleeping",
			        this, proc->me.id, newtask);

		  al.release();
                  if (__sync_fetch_and_add(&(newtask->run_count),1) == 0)
                    run_task(newtask);
		  al.reacquire();

		  log_task.info("thread %p (proc " IDFMT ") done with inner task %p, back to waiting on " IDFMT "/%d",
			        this, proc->me.id, newtask, wait_for.id, wait_for.gen);

                  if (__sync_add_and_fetch(&(newtask->finish_count),-1) == 0)
                    delete newtask;
		  continue;
	        }
	      }

	      // plan C - run the idle task if it's enabled and nobody else
	      //  is currently running it
	      if(!block && proc->idle_task_enabled && 
		 proc->idle_task && !proc->in_idle_task) {
		log_task.info("thread %p (proc " IDFMT ") running idle task instead of sleeping",
			      this, proc->me.id);

		proc->in_idle_task = true;
		
		al.release();
		run_task(proc->idle_task);
		al.reacquire();

		proc->in_idle_task = false;

		log_task.info("thread %p (proc " IDFMT ") done with idle task, back to waiting on " IDFMT "/%d",
			      this, proc->me.id, wait_for.id, wait_for.gen);

		continue;
	      }
	
	      // plan D - no ready tasks, no resumable tasks, no idle task,
	      //   so go to sleep, putting ourselves on the preemptable queue
	      //   if !block
	      if(block) {
		log_task.info("thread %p (proc " IDFMT ") blocked on event " IDFMT "/%d",
			      this, proc->me.id, wait_for.id, wait_for.gen);
		state = STATE_BLOCKED;
	      } else {
		log_task.info("thread %p (proc " IDFMT ") sleeping (preemptable) on event " IDFMT "/%d",
			      this, proc->me.id, wait_for.id, wait_for.gen);
		state = STATE_PREEMPTABLE;
		proc->preemptable_threads.push_back(this);
	      }
	      proc->active_thread_count--;
	      gasnett_cond_wait(&condvar, &proc->mutex.lock);
	      log_task.info("thread %p (proc " IDFMT ") awake",
			    this, proc->me.id);
	      assert(state == STATE_RUN);
	    }

	    log_task.info("thread %p (proc " IDFMT ") done sleeping on event " IDFMT "/%d",
			  this, proc->me.id, wait_for.id, wait_for.gen);
            // Have to do this while holding the lock
            assert(event_stack == entry);
            event_stack = entry->next;
	  }
	  
	  delete entry;
	}

	virtual void thread_main(void)
	{
          // if (proc->core_id >= 0) {
          //   cpu_set_t cset;
          //   CPU_ZERO(&cset);
          //   CPU_SET(proc->core_id, &cset);
          //   pthread_t current_thread = pthread_self();
          //   CHECK_PTHREAD( pthread_setaffinity_np(current_thread, sizeof(cset), &cset) );
          // }
	  // first thing - take the lock and set our status
	  state = STATE_IDLE;

	  log_task(LEVEL_DEBUG, "worker thread ready: proc=" IDFMT "", proc->me.id);

	  // we spend what looks like our whole life holding the processor
	  //  lock - we explicitly let go when we run tasks and implicitly
	  //  when we wait on our condvar
	  AutoHSLLock al(proc->mutex);

	  // add ourselves to the processor's thread list - if we're the first
	  //  we're responsible for calling the proc init task
	  {
	    // HACK: for now, don't wait - this feels like a race condition,
	    //  but the high level expects it to be this way
	    bool wait_for_init_done = false;

	    bool first = proc->all_threads.size() == 0;
	    proc->all_threads.insert(this);

	    // we count as an active thread until we can get to the idle loop
	    proc->active_thread_count++;

	    if(first) {
	      // let go of the lock while we call the init task
	      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_INIT);
	      if(it != task_id_table.end()) {
		log_task(LEVEL_INFO, "calling processor init task: proc=" IDFMT "", proc->me.id);
                // consider ourselves to be running for the duration of the init task
                state = STATE_RUN;
		gasnet_hsl_unlock(&proc->mutex);
		(it->second)(0, 0, proc->me);
		gasnet_hsl_lock(&proc->mutex);
                // now back to "idle"
                state = STATE_IDLE;
		log_task(LEVEL_INFO, "finished processor init task: proc=" IDFMT "", proc->me.id);
	      } else {
		log_task(LEVEL_INFO, "no processor init task: proc=" IDFMT "", proc->me.id);
	      }

	      // now we can set 'init_done', and signal anybody who is in the
	      //  INIT_WAIT state
	      proc->init_done = true;
              // Enable the idle task
              if(proc->util_proc) {
                log_task.info("idle task enabled for processor " IDFMT " on util proc " IDFMT "",
                              proc->me.id, proc->util.id);

                proc->util_proc->enable_idle_task(proc);
              } else {
                assert(proc->kind != Processor::UTIL_PROC);
                log_task.info("idle task enabled for processor " IDFMT "", proc->me.id);
                proc->idle_task_enabled = true;
              }
	      for(std::set<Thread *>::iterator it = proc->all_threads.begin();
		  it != proc->all_threads.end();
		  it++)
		if((*it)->state == STATE_INIT_WAIT) {
		  (*it)->state = STATE_IDLE;
		  gasnett_cond_signal(&((*it)->condvar));
		}
	    } else {
	      // others just wait until 'init_done' becomes set
	      if(wait_for_init_done && !proc->init_done) {
		log_task(LEVEL_INFO, "waiting for processor init to complete");
		state = STATE_INIT_WAIT;
		gasnett_cond_wait(&condvar, &proc->mutex.lock);
	      }
	    }

	    state = STATE_IDLE;
	    proc->active_thread_count--;
	  }

	  while(!proc->shutdown_requested) {
	    // first priority - try to run a task if one is available and
	    //   we're not at the active thread count limit
	    if(proc->active_thread_count < proc->max_active_threads) {
              Task *newtask = 0;
              if(!proc->task_queue.empty()) {
                newtask = proc->task_queue.pop();
              }
              if(newtask) {
                // Keep holding the lock until we are sure we're
                // going to run this task
                if (__sync_fetch_and_add(&(newtask->run_count),1) == 0) {
                  
                  proc->active_thread_count++;
                  state = STATE_RUN;

                  al.release();
                  log_task.info("thread running ready task %p for proc " IDFMT "",
                                newtask, proc->me.id);
                  run_task(newtask, proc->me);
                  log_task.info("thread finished running task %p for proc " IDFMT "",
                                newtask, proc->me.id);
                  if (__sync_add_and_fetch(&(newtask->finish_count),-1) == 0)
                    delete newtask;
                  al.reacquire();

                  state = STATE_IDLE;
                  proc->active_thread_count--;
                  
                } else if (__sync_add_and_fetch(&(newtask->finish_count),-1) == 0) {
                  delete newtask;
                }
	        continue;
              }
	    }

	    // next idea - try to run the idle task
	    if((proc->active_thread_count < proc->max_active_threads) &&
	       proc->idle_task_enabled && 
	       proc->idle_task && !proc->in_idle_task) {
	      log_task.info("thread %p (proc " IDFMT ") running idle task",
			    this, proc->me.id);

	      proc->in_idle_task = true;
	      proc->active_thread_count++;
	      state = STATE_RUN;
		
	      al.release();
	      run_task(proc->idle_task);
	      al.reacquire();

	      proc->in_idle_task = false;
	      proc->active_thread_count--;
	      state = STATE_IDLE;

	      log_task.info("thread %p (proc " IDFMT ") done with idle task",
			    this, proc->me.id);

	      continue;
	    }

	    // out of ideas, go to sleep
	    log_task.info("thread %p (proc " IDFMT ") has no work - sleeping",
			  this, proc->me.id);
	    proc->avail_threads.push_back(this);

	    gasnett_cond_wait(&condvar, &proc->mutex.lock);

	    log_task.info("thread %p (proc " IDFMT ") awake and looking for work",
			  this, proc->me.id);
	  }

	  // shutdown requested...
	  
	  // take ourselves off the list of threads - if we're the last
	  //  call a shutdown task, if one is registered
	  proc->all_threads.erase(this);
	  bool last = proc->all_threads.size() == 0;
	    
	  if(last) {
	    proc->disable_idle_task();
	    if(proc->util_proc)
	      proc->util_proc->wait_for_shutdown();

	    // let go of the lock while we call the shutdown task
	    Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
	    if(it != task_id_table.end()) {
	      log_task(LEVEL_INFO, "calling processor shutdown task: proc=" IDFMT "", proc->me.id);
              proc->active_thread_count++;
              state = STATE_RUN;

	      al.release();
	      (it->second)(0, 0, proc->me);
	      al.reacquire();

              proc->active_thread_count--;
              state = STATE_IDLE;

	      log_task(LEVEL_INFO, "finished processor shutdown task: proc=" IDFMT "", proc->me.id);
	    }
	    if(proc->shutdown_event.exists())
	      proc->shutdown_event.impl()->trigger(proc->shutdown_event.gen, gasnet_mynode());
            proc->finished();
	  }

	  log_task(LEVEL_DEBUG, "worker thread terminating: proc=" IDFMT "", proc->me.id);
	}

        virtual Processor get_processor(void) const
        {
          return proc->me;
        }
      }; 

      LocalProcessor(Processor _me, int _core_id, 
		     int _total_threads = 1, int _max_active_threads = 1)
	: Processor::Impl(_me, Processor::LOC_PROC), core_id(_core_id),
	  total_threads(_total_threads),
	  active_thread_count(0), max_active_threads(_max_active_threads),
	  init_done(false), shutdown_requested(false), shutdown_event(Event::NO_EVENT), in_idle_task(false),
	  idle_task_enabled(false)
      {
        gasnet_hsl_init(&mutex);

	// if a processor-idle task is in the table, make a Task object for it
	Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_IDLE);
	idle_task = ((it != task_id_table.end()) ?
  		       new Task(me, Processor::TASK_ID_PROCESSOR_IDLE, 0, 0, Event::NO_EVENT, 0, 0) :
		       0);
      }

      ~LocalProcessor(void)
      {
	delete idle_task;
      }

      void start_worker_threads(size_t stack_size)
      {
	// create worker threads - they will enqueue themselves when
	//   they're ready
	for(int i = 0; i < total_threads; i++) {
	  Thread *t = new Thread(this);
	  log_task(LEVEL_DEBUG, "creating worker thread : proc=" IDFMT " thread=%p", me.id, t);
	  t->start_thread(stack_size, core_id, "local worker");
	}
      }

      virtual void enqueue_task(Task *task)
      {
	// modifications to task/thread lists require mutex
	AutoHSLLock a(mutex);

	// special case: if task->func_id is 0, that's a shutdown request
	if(task->func_id == 0) {
	  log_task(LEVEL_INFO, "shutdown request received!");
	  shutdown_requested = true;
	  shutdown_event = task->finish_event;
          // Wake up any available threads that may be sleeping
          while (!avail_threads.empty()) {
            Thread *thread = avail_threads.front();
            avail_threads.pop_front();
            log_task(LEVEL_DEBUG, "waking thread to shutdown: proc=" IDFMT " task=%p thread=%p", me.id, task, thread);
	    gasnett_cond_signal(&thread->condvar);
            //thread->set_task_and_wake(task);
          }
	  return;
	}

        task_queue.insert(task, task->priority); 

	log_task.info("pushing ready task %p onto list for proc " IDFMT " (active=%d, idle=%zd, preempt=%zd)",
		      task, me.id, active_thread_count,
		      avail_threads.size(), preemptable_threads.size());

	if(active_thread_count < max_active_threads) {
	  if(avail_threads.size() > 0) {
	    // take a thread and start him - up to him to run this task or not
	    Thread *t = avail_threads.front();
	    avail_threads.pop_front();
	    assert(t->state == Thread::STATE_IDLE);
	    log_task.info("waking up thread %p to handle new task", t);
	    gasnett_cond_signal(&t->condvar);
	  } else
	    if(preemptable_threads.size() > 0) {
	      Thread *t = preemptable_threads.front();
	      preemptable_threads.pop_front();
	      assert(t->state == Thread::STATE_PREEMPTABLE);
	      t->state = Thread::STATE_RUN;
	      active_thread_count++;
	      log_task.info("preempting thread %p to handle new task", t);
	      gasnett_cond_signal(&t->condvar);
	    } else {
	      log_task.info("no threads avialable to run new task %p", task);
	    }
	}
      }

      virtual void tasks_available(int priority)
      {
	log_task.warning("tasks available: priority = %d", priority);
        AutoHSLLock a(mutex);

	if(active_thread_count < max_active_threads) {
	  if(avail_threads.size() > 0) {
	    // take a thread and start him - up to him to run this task or not
	    Thread *t = avail_threads.front();
	    avail_threads.pop_front();
	    assert(t->state == Thread::STATE_IDLE);
	    log_task.info("waking up thread %p to handle new shared task", t);
	    gasnett_cond_signal(&t->condvar);
	  } else
	    if(preemptable_threads.size() > 0) {
	      Thread *t = preemptable_threads.front();
	      preemptable_threads.pop_front();
	      assert(t->state == Thread::STATE_PREEMPTABLE);
	      t->state = Thread::STATE_RUN;
	      active_thread_count++;
	      log_task.info("preempting thread %p to handle new shared task", t);
	      gasnett_cond_signal(&t->condvar);
	    } else {
	      log_task.info("no threads avialable to run new shared task");
	    }
	}
      }

      // see if there are resumable threads and/or new tasks to run, respecting
      //  the available thread and runnable thread limits
      // ASSUMES LOCK IS HELD BY CALLER
      void start_some_threads(void)
      {
	// favor once-running threads that now want to resume
	while((active_thread_count < max_active_threads) &&
	      (resumable_threads.size() > 0)) {
	  Thread *t = resumable_threads.front();
	  resumable_threads.pop_front();
	  active_thread_count++;
	  log_task(LEVEL_SPEW, "ATC = %d", active_thread_count);
	  assert(t->state == Thread::STATE_RESUMABLE);
	  t->state = Thread::STATE_RUN;
	  gasnett_cond_signal(&t->condvar);
	}
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority)
      {
	// create task object to hold args, etc.
	Task *task = new Task(me, func_id, args, arglen, finish_event, 
                              priority, 1/*users*/);

	// early out - if the event has obviously triggered (or is NO_EVENT)
	//  don't build up continuation
	if(start_event.has_triggered()) {
	  log_task(LEVEL_INFO, "new ready task: func=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
		   func_id, start_event.id, start_event.gen,
		   finish_event.id, finish_event.gen);
          enqueue_task(task);
	} else {
	  log_task(LEVEL_DEBUG, "deferring spawn: func=%d event=" IDFMT "/%d",
		   func_id, start_event.id, start_event.gen);
	  start_event.impl()->add_waiter(start_event, new DeferredTaskSpawn(this, task));
	}
      }

      virtual void enable_idle_task(void)
      {
	if(util_proc) {
	  log_task.info("idle task enabled for processor " IDFMT " on util proc " IDFMT "",
			me.id, util.id);

	  util_proc->enable_idle_task(this);
	} else {
	  assert(kind != Processor::UTIL_PROC);
	  log_task.info("idle task enabled for processor " IDFMT "", me.id);
          // need the lock when modifying data structures
          AutoHSLLock a(mutex);
	  idle_task_enabled = true;
          if(active_thread_count < max_active_threads) {
            if(avail_threads.size() > 0) {
              // take a thread and start him - up to him to run this task or not
              Thread *t = avail_threads.front();
              avail_threads.pop_front();
              assert(t->state == Thread::STATE_IDLE);
              log_task.info("waking up thread %p to run idle task", t);
              t->state = Thread::STATE_RUN;
              active_thread_count++;
              gasnett_cond_signal(&t->condvar);
            } else
              if(preemptable_threads.size() > 0) {
                Thread *t = preemptable_threads.front();
                preemptable_threads.pop_front();
                assert(t->state == Thread::STATE_PREEMPTABLE);
                t->state = Thread::STATE_RUN;
                active_thread_count++;
                log_task.info("preempting thread %p to run idle task", t);
                gasnett_cond_signal(&t->condvar);
              } else {
                log_task.info("no threads avialable to run idle task");
              }
          }
	}
      }

      virtual void disable_idle_task(void)
      {
	if(util_proc) {
	  log_task.info("idle task disabled for processor " IDFMT " on util proc " IDFMT "",
			me.id, util.id);

	  util_proc->disable_idle_task(this);
	} else {
	  log_task.info("idle task disabled for processor " IDFMT "", me.id);
	  idle_task_enabled = false;
	}
      }

      virtual bool is_idle_task_enabled(void)
      {
	return idle_task_enabled;
      }

    protected:
      int core_id;
      JobQueue<Task> task_queue;
      int total_threads, active_thread_count, max_active_threads;
      std::list<Thread *> avail_threads;
      std::list<Thread *> resumable_threads;
      std::list<Thread *> preemptable_threads;
      std::set<Thread *> all_threads;
      gasnet_hsl_t mutex;
      bool init_done, shutdown_requested;
      Event shutdown_event;
      bool in_idle_task;
      Task *idle_task;
      bool idle_task_enabled;
    };

    void PreemptableThread::start_thread(size_t stack_size, int core_id, const char *debug_name)
    {
      pthread_attr_t attr;
      CHECK_PTHREAD( pthread_attr_init(&attr) );
      CHECK_PTHREAD( pthread_attr_setstacksize(&attr,stack_size) );
      if(proc_assignment)
	proc_assignment->bind_thread(core_id, &attr, debug_name);
      CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_entry, (void *)this) );
      CHECK_PTHREAD( pthread_attr_destroy(&attr) );
#ifdef DEADLOCK_TRACE
      Runtime::get_runtime()->add_thread(&thread);
#endif
    }

    GASNETT_THREADKEY_DEFINE(cur_preemptable_thread);

    /*static*/ bool PreemptableThread::preemptable_sleep(Event wait_for,
							 bool block /*= false*/)
    {
      // check TLS to see if we're really a preemptable thread
      void *tls_val = gasnett_threadkey_get(cur_preemptable_thread);
      if(!tls_val) return false;

      PreemptableThread *me = (PreemptableThread *)tls_val;

      me->sleep_on_event(wait_for, block);
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

    class UtilityProcessor::UtilityThread : public PreemptableThread {
    public:
      UtilityThread(UtilityProcessor *_proc)
	: proc(_proc) {}

      virtual ~UtilityThread(void) {}

      void run_task(Task *task, Processor actual_proc = Processor::NO_PROC)
      {
	Processor::TaskFuncPtr fptr = task_id_table[task->func_id];
	char argstr[100];
	argstr[0] = 0;
	for(size_t i = 0; (i < task->arglen) && (i < 40); i++)
	  sprintf(argstr+2*i, "%02x", ((unsigned char *)(task->args))[i]);
	if(task->arglen > 40) strcpy(argstr+80, "...");
	log_util(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
		 "utility task start: %d (%p) (%s)", task->func_id, fptr, argstr);
	(*fptr)(task->args, task->arglen, (actual_proc.exists() ? actual_proc : task->proc));
	log_util(((task->func_id == 3) ? LEVEL_SPEW : LEVEL_INFO), 
		 "utility task end: %d (%p) (%s)", task->func_id, fptr, argstr);
	if(task->finish_event.exists())
	  task->finish_event.impl()->trigger(task->finish_event.gen, gasnet_mynode());
      }

      void sleep_on_event(Event wait_for, bool block = false)
      {
        Event::Impl *impl = Runtime::get_runtime()->get_event_impl(wait_for);
        
        while(!impl->has_triggered(wait_for.gen)) {
          if (!block) {
            gasnet_hsl_lock(&proc->mutex);
            if(!proc->task_queue.empty())
	    {
	      Task *task = proc->task_queue.pop();
	      if(task) {
		gasnet_hsl_unlock(&proc->mutex);
		log_util.info("running task %p (%d) in utility thread", task, task->func_id);
                if (__sync_fetch_and_add(&(task->run_count),1) == 0)
                  run_task(task, proc->me);
		log_util.info("done with task %p (%d) in utility thread", task, task->func_id);
                if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
                  delete task;
		// Continue since we no longer hold the lock
		continue;
	      }
	    }
            gasnet_hsl_unlock(&proc->mutex);
          }
#ifdef __SSE2__
          _mm_pause();
#endif
	  log_util.spew("utility thread polling on event " IDFMT "/%d",
			wait_for.id, wait_for.gen);
	  //usleep(1000);
	}
      }

      virtual Processor get_processor(void) const
      {
        return proc->me;
      }

    protected:
      void thread_main(void)
      {
        // if (proc->core_id >= 0) {
        //   cpu_set_t cset;
        //   CPU_ZERO(&cset);
        //   CPU_SET(proc->core_id, &cset);
        //   pthread_t current_thread = pthread_self();
        //   CHECK_PTHREAD( pthread_setaffinity_np(current_thread, sizeof(cset), &cset) );
        // }
	// need to call init for utility processor too
	Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_INIT);
	if(it != task_id_table.end()) {
	  log_task(LEVEL_INFO, "calling processor init task for utility proc: proc=" IDFMT "", proc->me.id);
	  (it->second)(0, 0, proc->me);
	  log_task(LEVEL_INFO, "finished processor init task for utility proc: proc=" IDFMT "", proc->me.id);
	} else {
	  log_task(LEVEL_INFO, "no processor init task for utility proc: proc=" IDFMT "", proc->me.id);
	}

	// we spend most of our life with the utility processor's lock (or
	//   waiting for it) - we only drop it when we have work to do
	gasnet_hsl_lock(&proc->mutex);
	log_util.info("utility worker thread started, proc=" IDFMT "", proc->me.id);

	while(!proc->shutdown_requested) {
	  // try to run tasks from the runnable queue
	  while(!proc->task_queue.empty()) {

	    Task *task = proc->task_queue.pop();

	    // is it the shutdown task?
	    if(task->func_id == 0) {
	      proc->shutdown_requested = true;
	      // wake up any other threads that are sleeping
	      gasnett_cond_broadcast(&proc->condvar);
	      delete task;
	      continue;
	    }

            // hold the lock until we are sure we're running this task
            if (__sync_fetch_and_add(&(task->run_count),1) == 0) {
              gasnet_hsl_unlock(&proc->mutex);
              log_util.info("running task %p (%d) in utility thread", task, task->func_id);
              run_task(task, proc->me);
              log_util.info("done with task %p (%d) in utility thread", task, task->func_id);
              if (__sync_add_and_fetch(&(task->finish_count),-1) == 0)
                delete task;
              gasnet_hsl_lock(&proc->mutex);
            } else if (__sync_add_and_fetch(&(task->finish_count),-1) == 0) {
              delete task;
            }
	  }

	  // run some/all of the idle tasks for idle processors
	  // the set can change, so grab a copy, then let go of the lock
	  //  while we walk the list - for each item, retake the lock to
	  //  see if it's still on the list and make sure nobody else is
	  //  running it
	  if(proc->idle_procs.size() > 0) {
	    std::set<Processor::Impl *> copy_of_idle_procs = proc->idle_procs;

	    for(std::set<Processor::Impl *>::iterator it = copy_of_idle_procs.begin();
		it != copy_of_idle_procs.end();
		it++) {
	      Processor::Impl *idle_proc = *it;
	      // for each item on the list, run the idle task as long as:
	      //  1) it's still in the idle proc set, and
	      //  2) somebody else isn't already running its idle task
	      bool ok_to_run = (proc->idle_task && 
                                (proc->idle_procs.count(idle_proc) > 0) &&
				(proc->procs_in_idle_task.count(idle_proc) == 0));
	      if(ok_to_run) {
		proc->procs_in_idle_task.insert(idle_proc);
		gasnet_hsl_unlock(&proc->mutex);

		// run the idle task on behalf of the idle proc
		log_util.debug("running idle task for " IDFMT "", idle_proc->me.id);
                if (proc->idle_task)
		run_task(proc->idle_task, idle_proc->me);
		log_util.debug("done with idle task for " IDFMT "", idle_proc->me.id);

		gasnet_hsl_lock(&proc->mutex);
		proc->procs_in_idle_task.erase(idle_proc);
	      }
	    }
	  }
	  
	  // if we really have nothing to do, it's ok to go to sleep
	  if(proc->task_queue.empty() && (proc->idle_procs.size() == 0) &&
	     !proc->shutdown_requested) {
	    log_util.info("utility thread going to sleep (%p, %p)", this, proc);
	    gasnett_cond_wait(&proc->condvar, &proc->mutex.lock);
	    log_util.info("utility thread awake again");
	  }
	}

	log_util.info("utility worker thread shutting down, proc=" IDFMT "", proc->me.id);
	proc->threads.erase(this);
	gasnett_cond_broadcast(&proc->condvar);
	gasnet_hsl_unlock(&proc->mutex);
        // Let go of the lock while holding calling the shutdown task
        it = task_id_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
        if(it != task_id_table.end()) {
          log_task(LEVEL_INFO, "calling processor shutdown task for utility proc: proc=" IDFMT "", proc->me.id);

          (it->second)(0, 0, proc->me);

          log_task(LEVEL_INFO, "finished processor shutdown task for utility proc: proc=" IDFMT "", proc->me.id);
        }
        //if(proc->shutdown_event.exists())
        //  proc->shutdown_event.impl()->trigger(proc->shutdown_event.gen, gasnet_mynode());
        proc->finished();
      }

      UtilityProcessor *proc;
      pthread_t thread;
    };

    UtilityProcessor::UtilityProcessor(Processor _me,
                                       int _core_id /*=-1*/,
				       int _num_worker_threads /*= 1*/)
      : Processor::Impl(_me, Processor::UTIL_PROC, Processor::NO_PROC),
	core_id(_core_id), num_worker_threads(_num_worker_threads), shutdown_requested(false)
    {
      gasnet_hsl_init(&mutex);
      gasnett_cond_init(&condvar);

      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_IDLE);
      idle_task = ((it != task_id_table.end()) ?
		     new Task(this->me, 
			      Processor::TASK_ID_PROCESSOR_IDLE, 
			      0, 0, Event::NO_EVENT, 0, 0) :
		     0);
    }

    UtilityProcessor::~UtilityProcessor(void)
    {
      if(idle_task)
	delete idle_task;
    }

    void UtilityProcessor::start_worker_threads(size_t stack_size)
    {
      for(int i = 0; i < num_worker_threads; i++) {
	UtilityThread *t = new UtilityThread(this);
        log_util.info("utility thread %p created for proc "IDFMT"(%p)", t, this->me.id, this);
	threads.insert(t);
	t->start_thread(stack_size, -1, "utility worker");
      }
    }

    void UtilityProcessor::request_shutdown(void)
    {
      // set the flag first
      shutdown_requested = true;

      // now take the mutex so we can wake up anybody who is still asleep
      AutoHSLLock al(mutex);
      gasnett_cond_broadcast(&condvar);
    }

    /*virtual*/ void UtilityProcessor::spawn_task(Processor::TaskFuncID func_id,
						  const void *args, size_t arglen,
						  //std::set<RegionInstance> instances_needed,
						  Event start_event, Event finish_event,
                                                  int priority)
    {
      Task *task = new Task(this->me, func_id, args, arglen,
			    finish_event, priority, 1/*users*/);

      if (start_event.has_triggered())
        enqueue_task(task);
      else
        start_event.impl()->add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }

    void UtilityProcessor::tasks_available(int priority)
    {
      AutoHSLLock al(mutex);
      gasnett_cond_signal(&condvar);
    }

    void UtilityProcessor::enqueue_task(Task *task)
    {
      AutoHSLLock al(mutex);
      task_queue.insert(task, task->priority);
      gasnett_cond_signal(&condvar);
    }

    void UtilityProcessor::enable_idle_task(Processor::Impl *proc)
    {
      AutoHSLLock al(mutex);

      assert(proc != 0);
      idle_procs.insert(proc);
      gasnett_cond_signal(&condvar);
    }
     
    void UtilityProcessor::disable_idle_task(Processor::Impl *proc)
    {
      AutoHSLLock al(mutex);

      idle_procs.erase(proc);
    }

    void UtilityProcessor::wait_for_shutdown(void)
    {
      AutoHSLLock al(mutex);

      if(threads.size() == 0) return;

      log_util.info("thread waiting for utility proc " IDFMT " threads to shut down",
		    me.id);
      while(threads.size() > 0)
	gasnett_cond_wait(&condvar, &mutex.lock);

      log_util.info("thread resuming - utility proc has shut down");
    }

    struct SpawnTaskArgs : public BaseMedium {
      Processor proc;
      Event start_event;
      Event finish_event;
      Processor::TaskFuncID func_id;
      int priority;
    };

    void Event::wait(bool block) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if(!id) return;  // special case: never wait for NO_EVENT
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);

      // early out case too
      if(e->has_triggered(gen)) return;

      // waiting on an event does not count against the low level's time
      DetailedTimer::ScopedPush sp2(TIME_NONE);

      // are we a thread that knows how to do something useful while waiting?
      if(PreemptableThread::preemptable_sleep(*this, block))
	return;

      // figure out which thread we are - better be a local CPU thread!
      void *ptr = gasnett_threadkey_get(cur_thread);
      if(ptr != 0) {
	assert(0);
	LocalProcessor::Thread *thr = (LocalProcessor::Thread *)ptr;
	thr->sleep_on_event(*this, block);
	return;
      }
      // maybe a GPU thread?
      ptr = gasnett_threadkey_get(gpu_thread);
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
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);

      // early out case too
      if(e->has_triggered(gen)) return;

      // waiting on an event does not count against the low level's time
      DetailedTimer::ScopedPush sp2(TIME_NONE);

      e->external_wait(gen);
    }

    // can't be static if it's used in a template...
    void handle_spawn_task_message(SpawnTaskArgs args,
				   const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Processor::Impl *p = args.proc.impl();
      log_task(LEVEL_DEBUG, "remote spawn request: proc_id=" IDFMT " task_id=%d event=" IDFMT "/%d",
	       args.proc.id, args.func_id, args.start_event.id, args.start_event.gen);
      p->spawn_task(args.func_id, data, datalen,
		    args.start_event, args.finish_event, args.priority);
    }

    typedef ActiveMessageMediumNoReply<SPAWN_TASK_MSGID,
				       SpawnTaskArgs,
				       handle_spawn_task_message> SpawnTaskMessage;

    class RemoteProcessor : public Processor::Impl {
    public:
      RemoteProcessor(Processor _me, Processor::Kind _kind, Processor _util)
	: Processor::Impl(_me, _kind, _util)
      {
      }

      ~RemoteProcessor(void)
      {
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
	log_task(LEVEL_DEBUG, "spawning remote task: proc=" IDFMT " task=%d start=" IDFMT "/%d finish=" IDFMT "/%d",
		 me.id, func_id, 
		 start_event.id, start_event.gen,
		 finish_event.id, finish_event.gen);
	SpawnTaskArgs msgargs;
	msgargs.proc = me;
	msgargs.func_id = func_id;
	msgargs.start_event = start_event;
	msgargs.finish_event = finish_event;
        msgargs.priority = priority;
	SpawnTaskMessage::request(ID(me).node(), msgargs, args, arglen,
				  PAYLOAD_COPY);
      }
    };

    /*static*/ Processor Processor::create_group(const std::vector<Processor>& members)
    {
      // are we creating a local group?
      if((members.size() == 0) || (ID(members[0]).node() == gasnet_mynode())) {
	ProcessorGroup *grp = Runtime::runtime->local_proc_group_free_list->alloc_entry();
	grp->set_group_members(members);
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

      ProcessorGroup *grp = Runtime::get_runtime()->get_procgroup_impl(*this);
      grp->get_group_members(members);
    }

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
			   //std::set<RegionInstance> instances_needed,
			   Event wait_on, int priority) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Processor::Impl *p = impl();
      Event finish_event = Event::Impl::create_event();
      p->spawn_task(func_id, args, arglen, //instances_needed, 
		    wait_on, finish_event, priority);
      return finish_event;
    }

    Processor Processor::get_utility_processor(void) const
    {
      Processor u = impl()->util;
      return(u.exists() ? u : *this);
    }

    void Processor::enable_idle_task(void)
    {
      impl()->enable_idle_task();
    }

    void Processor::disable_idle_task(void)
    {
      impl()->disable_idle_task();
    }

    AddressSpace Processor::address_space(void) const
    {
      return ID(id).node();
    }

    IDType Processor::local_id(void) const
    {
      return ID(id).index();
    }

    ///////////////////////////////////////////////////
    // Runtime

    Event::Impl *Runtime::get_event_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_EVENT:
	{
	  Node *n = &runtime->nodes[id.node()];
	  Event::Impl *impl = n->events.lookup_entry(id.index(), id.node());
	  assert(impl->me == id.convert<Event>());
	  return impl;
#if 0
	  unsigned index = id.index();
	  if(index >= n->num_events) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->events.size();
	    if(index >= oldsize) { // only it's still too small
              assert((index+1) < MAX_LOCAL_EVENTS);
	      n->events.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->events[i].init(ID(ID::ID_EVENT, id.node(), i).convert<Event>(),
				  id.node());
	      n->num_events = index + 1;
	    }
	  }
	  return &(n->events[index]);
#endif
	}

      default:
	assert(0);
      }
    }

    Reservation::Impl *Runtime::get_lock_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_LOCK:
	{
	  Node *n = &runtime->nodes[id.node()];
	  Reservation::Impl *impl = n->reservations.lookup_entry(id.index(), id.node());
	  assert(impl->me == id.convert<Reservation>());
	  return impl;
#if 0
	  std::vector<Reservation::Impl>& locks = nodes[id.node()].locks;

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

    Memory::Impl *Runtime::get_memory_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_MEMORY:
      case ID::ID_ALLOCATOR:
      case ID::ID_INSTANCE:
	if(id.index_h() == ID::ID_GLOBAL_MEM)
	  return runtime->global_memory;
	return null_check(runtime->nodes[id.node()].memories[id.index_h()]);

      default:
	assert(0);
      }
    }

    Processor::Impl *Runtime::get_processor_impl(ID id)
    {
      if(id.type() == ID::ID_PROCGROUP)
	return get_procgroup_impl(id);

      assert(id.type() == ID::ID_PROCESSOR);
      return null_check(runtime->nodes[id.node()].processors[id.index()]);
    }

    ProcessorGroup *Runtime::get_procgroup_impl(ID id)
    {
      assert(id.type() == ID::ID_PROCGROUP);

      Node *n = &runtime->nodes[id.node()];
      ProcessorGroup *impl = n->proc_groups.lookup_entry(id.index(), id.node());
      assert(impl->me == id.convert<Processor>());
      return impl;
    }

    IndexSpace::Impl *Runtime::get_index_space_impl(ID id)
    {
      assert(id.type() == ID::ID_INDEXSPACE);

      Node *n = &runtime->nodes[id.node()];
      IndexSpace::Impl *impl = n->index_spaces.lookup_entry(id.index(), id.node());
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
	  n->index_spaces[index] = new IndexSpace::Impl(id.convert<IndexSpace>());
	} 
      }

      return n->index_spaces[index];
#endif
    }

    RegionInstance::Impl *Runtime::get_instance_impl(ID id)
    {
      assert(id.type() == ID::ID_INSTANCE);
      Memory::Impl *mem = get_memory_impl(id);
      
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
	  mem->instances[id.index_l()] = new RegionInstance::Impl(id.convert<RegionInstance>(), mem->me);
	}
      }
	  
      return mem->instances[id.index_l()];
    }

#ifdef DEADLOCK_TRACE
    void Runtime::add_thread(const pthread_t *thread)
    {
      unsigned idx = __sync_fetch_and_add(&next_thread,1);
      assert(idx < MAX_NUM_THREADS);
      all_threads[idx] = *thread;
      thread_counts[idx] = 0;
    }
#endif

    ///////////////////////////////////////////////////
    // RegionMetaData

    static Logger::Category log_meta("meta");

    /*static*/ const IndexSpace IndexSpace::NO_SPACE = IndexSpace();
    /*static*/ const Domain Domain::NO_DOMAIN = Domain();

    IndexSpace::Impl *IndexSpace::impl(void) const
    {
      return Runtime::runtime->get_index_space_impl(*this);
    }

    /*static*/ IndexSpace IndexSpace::create_index_space(size_t num_elmts)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpace::Impl *impl = Runtime::runtime->local_index_space_free_list->alloc_entry();
      
      impl->init(impl->me, NO_SPACE, num_elmts);
      
      log_meta(LEVEL_INFO, "index space created: id=" IDFMT " num_elmts=%zd",
	       impl->me.id, num_elmts);
      return impl->me;
    }

    /*static*/ IndexSpace IndexSpace::create_index_space(const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpace::Impl *impl = Runtime::runtime->local_index_space_free_list->alloc_entry();
      
      // TODO: actually decide when to safely consider a subregion frozen
      impl->init(impl->me, NO_SPACE, mask.get_num_elmts(), &mask, true);
      
      log_meta(LEVEL_INFO, "index space created: id=" IDFMT " num_elmts=%d",
	       impl->me.id, mask.get_num_elmts());
      return impl->me;
    }

    /*static*/ IndexSpace IndexSpace::create_index_space(IndexSpace parent, const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpace::Impl *impl = Runtime::runtime->local_index_space_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).type() == ID::ID_INDEXSPACE);

      StaticAccess<IndexSpace::Impl> p_data(parent.impl());

      impl->init(impl->me, parent,
		 p_data->num_elmts, 
		 &mask,
		 true);  // TODO: actually decide when to safely consider a subregion frozen
      
      log_meta(LEVEL_INFO, "index space created: id=" IDFMT " parent=" IDFMT " (num_elmts=%zd)",
	       impl->me.id, parent.id, p_data->num_elmts);
      return impl->me;
    }

    IndexSpaceAllocator IndexSpace::create_allocator(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceAllocator::Impl *a_impl = new IndexSpaceAllocator::Impl(impl());
      return IndexSpaceAllocator(a_impl);
#if 0
      // we have to calculate the number of bytes needed in case the request
      //  goes to a remote memory
      StaticAccess<IndexSpace::Impl> r_data(impl());
      assert(!r_data->frozen);
      size_t mask_size = ElementMaskImpl::bytes_needed(0, r_data->num_elmts);

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      IndexSpaceAllocator a = m_impl->create_allocator(*this, 2 * mask_size);
      log_meta(LEVEL_INFO, "allocator created: region=" IDFMT " memory=" IDFMT " id=" IDFMT "",
	       this->id, memory.id, a.id);
      return a;
#endif
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
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      size_t elem_size = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++)
	elem_size += *it;

      size_t num_elements;
      int linearization_bits[RegionInstance::Impl::MAX_LINEARIZATION_LEN];
      if(get_dim() > 0) {
	// we have a rectangle - figure out its volume and create based on that
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
	//printf("num_elements = %zd\n", num_elements);
      } else {
	IndexSpace::Impl *r = get_index_space().impl();

	StaticAccess<IndexSpace::Impl> data(r);
	assert(data->num_elmts > 0);

#ifdef FULL_SIZE_INSTANCES
	num_elements = data->last_elmt + 1;
	// linearization is an identity translation
	Translation<1> inst_offset(0);
	DomainLinearization dl = DomainLinearization::from_mapping<1>(Mapping<1,1>::new_dynamic_mapping(inst_offset));
	dl.serialize(linearization_bits);
#else
	num_elements = data->last_elmt - data->first_elmt + 1;
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
						 -1 /*list size*/,
						 RegionInstance::NO_INST);
      log_meta(LEVEL_INFO, "instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd",
	       this->is_id, memory.id, i.id, inst_bytes);
      return i;
    }

#if 0
    RegionInstance IndexSpace::create_instance_untyped(Memory memory,
									 ReductionOpID redopid) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      const ReductionOpUntyped *redop = reduce_op_table[redopid];

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      size_t inst_bytes = impl()->instance_size(redop);
      off_t inst_adjust = impl()->instance_adjust(redop);

      RegionInstance i = m_impl->create_instance(get_index_space(), inst_bytes, 
							inst_adjust, redopid);
      log_meta(LEVEL_INFO, "instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd adjust=%zd redop=%d",
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

      const ReductionOpUntyped *redop = reduce_op_table[redopid];

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      size_t inst_bytes = impl()->instance_size(redop, list_size);
      off_t inst_adjust = impl()->instance_adjust(redop);

      RegionInstance i = m_impl->create_instance(*this, inst_bytes, 
							inst_adjust, redopid,
							list_size, parent_inst);
      log_meta(LEVEL_INFO, "instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd adjust=%zd redop=%d list_size=%zd parent_inst=" IDFMT "",
	       this->id, memory.id, i.id, inst_bytes, inst_adjust, redopid,
	       list_size, parent_inst.id);
      return i;
    }
#endif

    void IndexSpace::destroy(void) const
    {
      //assert(0);
    }

    void IndexSpaceAllocator::destroy(void) 
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (impl != NULL)
      {
        delete impl;
        // Avoid double frees
        impl = NULL;
      }
    }

    class DeferredInstDestroy : public Event::Impl::EventWaiter {
    public:
      DeferredInstDestroy(RegionInstance::Impl *i) : impl(i) { }
      virtual ~DeferredInstDestroy(void) { }
    public:
      virtual bool event_triggered(void)
      {
        StaticAccess<RegionInstance::Impl> i_data(impl);
        log_meta(LEVEL_INFO, "instance destroyed: space=" IDFMT " id=" IDFMT "",
                 i_data->is.id, impl->me.id);
        impl->memory.impl()->destroy_instance(impl->me, true); 
        return true;
      }

      virtual void print_info(FILE *f)
      {
        fprintf(f,"deferred instance destruction\n");
      }
    protected:
      RegionInstance::Impl *impl;
    };

    void RegionInstance::destroy(Event wait_on /*= Event::NO_EVENT*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstance::Impl *i_impl = impl();
      if (!wait_on.has_triggered())
      {
        wait_on.impl()->add_waiter(wait_on, new DeferredInstDestroy(i_impl));
        return;
      }
      StaticAccess<RegionInstance::Impl> i_data(i_impl);
      log_meta(LEVEL_INFO, "instance destroyed: space=" IDFMT " id=" IDFMT "",
	       i_data->is.id, this->id);
      i_impl->memory.impl()->destroy_instance(*this, true);
    }

    const ElementMask &IndexSpace::get_valid_mask(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpace::Impl *r_impl = impl();
#ifdef COHERENT_BUT_BROKEN_WAY
      // for now, just hand out the valid mask for the master allocator
      //  and hope it's accessible to the caller
      SharedAccess<IndexSpace::Impl> data(r_impl);
      assert((data->valid_mask_owners >> gasnet_mynode()) & 1);
#else
      if(!r_impl->valid_mask_complete) {
	Event wait_on = r_impl->request_valid_mask();
	
	log_copy.info("missing valid mask (" IDFMT "/%p) - waiting for " IDFMT "/%d",
		      id, r_impl->valid_mask,
		      wait_on.id, wait_on.gen);

	wait_on.wait(true /*blocking*/);
      }
#endif
      return *(r_impl->valid_mask);
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
	copy_from.memory.impl()->get_bytes(copy_from.offset, raw_data, bytes_needed);
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
        rhs.memory.impl()->get_bytes(rhs.offset, raw_data, bytes_needed);
      return *this;
    }

    void ElementMask::init(int _first_element, int _num_elements, Memory _memory, off_t _offset)
    {
      first_element = _first_element;
      num_elements = _num_elements;
      memory = _memory;
      offset = _offset;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = memory.impl()->get_direct_ptr(offset, bytes_needed);
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
	Memory::Impl *m_impl = memory.impl();

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
	Memory::Impl *m_impl = memory.impl();

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
	Memory::Impl *m_impl = memory.impl();
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
	Memory::Impl *m_impl = memory.impl();

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

	// fetch first value and see if we have any bits set
	int idx = pos >> 6;
	uint64_t bits = impl->bits[idx];
	if(!polarity) bits = ~bits;

	// for the first one, we may have bits to ignore at the start
	if(pos & 0x3f)
	  bits &= ~((1ULL << (pos & 0x3f)) - 1);

	// skip over words that are all zeros
	while(!bits) {
	  idx++;
	  if((idx << 6) >= mask.num_elements) {
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
	Memory::Impl *m_impl = mask.memory.impl();

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
      return impl->alloc_elements(count);
    }

    void IndexSpaceAllocator::reserve(unsigned ptr, unsigned count /*= 1  */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return impl->reserve_elements(ptr, count);
    }

    void IndexSpaceAllocator::free(unsigned ptr, unsigned count /*= 1  */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return impl->free_elements(ptr, count);
    }

    IndexSpaceAllocator::Impl::Impl(IndexSpace::Impl *_is_impl)
      : is_impl(_is_impl)
    {
    }

    IndexSpaceAllocator::Impl::~Impl(void)
    {
    }

    unsigned IndexSpaceAllocator::Impl::alloc_elements(unsigned count /*= 1 */)
    {
      SharedAccess<IndexSpace::Impl> is_data(is_impl);
      assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
      int start = is_impl->valid_mask->find_disabled(count);
      assert(start >= 0);

      reserve_elements(start, count);

      return start;
    }

    void IndexSpaceAllocator::Impl::reserve_elements(unsigned ptr, unsigned count /*= 1 */)
    {
      // for now, do updates of valid masks immediately
      IndexSpace::Impl *impl = is_impl;
      while(1) {
	SharedAccess<IndexSpace::Impl> is_data(is_impl);
	assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
	is_impl->valid_mask->enable(ptr, count);
	IndexSpace is = is_data->parent;
	if(is == IndexSpace::NO_SPACE) break;
	impl = is.impl();
      }
    }

    void IndexSpaceAllocator::Impl::free_elements(unsigned ptr, unsigned count /*= 1*/)
    {
      // for now, do updates of valid masks immediately
      IndexSpace::Impl *impl = is_impl;
      while(1) {
	SharedAccess<IndexSpace::Impl> is_data(is_impl);
	assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
	is_impl->valid_mask->disable(ptr, count);
	IndexSpace is = is_data->parent;
	if(is == IndexSpace::NO_SPACE) break;
	impl = is.impl();
      }
    }

    ///////////////////////////////////////////////////
    // Region Instances

    RegionInstance::Impl *RegionInstance::impl(void) const
    {
      return Runtime::runtime->get_instance_impl(*this);
    }

#ifdef POINTER_CHECKS
    void RegionInstance::Impl::verify_access(unsigned ptr)
    {
      StaticAccess<RegionInstance::Impl> data(this);
      const ElementMask &mask = data->is.get_valid_mask();
      if (!mask.is_set(ptr))
      {
        fprintf(stderr,"ERROR: Accessing invalid pointer %d in logical region " IDFMT "\n",ptr,data->is.id);
        assert(false);
      }
    }
#endif

    void RegionInstance::Impl::get_bytes(int index, off_t byte_offset, void *dst, size_t size)
    {
      StaticAccess<RegionInstance::Impl> data(this);
      off_t o;
      if(data->block_size == 1) {
	// no blocking - don't need to know about field boundaries
	o = (index * data->elmt_size) + byte_offset;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(data->field_sizes, byte_offset, size, field_start, field_size);

	int block_num = index / data->block_size;
	int block_ofs = index % data->block_size;

	o = (((data->elmt_size * block_num + field_start) * data->block_size) + 
	     (field_size * block_ofs) +
	     (byte_offset - field_start));
      }
      Memory::Impl *m = Runtime::runtime->get_memory_impl(memory);
      m->get_bytes(data->alloc_offset + o, dst, size);
    }

    void RegionInstance::Impl::put_bytes(int index, off_t byte_offset, const void *src, size_t size)
    {
      StaticAccess<RegionInstance::Impl> data(this);
      off_t o;
      if(data->block_size == 1) {
	// no blocking - don't need to know about field boundaries
	o = (index * data->elmt_size) + byte_offset;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(data->field_sizes, byte_offset, size, field_start, field_size);

	int block_num = index / data->block_size;
	int block_ofs = index % data->block_size;

	o = (((data->elmt_size * block_num + field_start) * data->block_size) + 
	     (field_size * block_ofs) +
	     (byte_offset - field_start));
      }
      Memory::Impl *m = Runtime::runtime->get_memory_impl(memory);
      m->put_bytes(data->alloc_offset + o, src, size);
    }

    /*static*/ const RegionInstance RegionInstance::NO_INST = RegionInstance();

    // a generic accessor just holds a pointer to the impl and passes all 
    //  requests through
    RegionAccessor<AccessorType::Generic> RegionInstance::get_accessor(void) const
    {
      return RegionAccessor<AccessorType::Generic>(AccessorType::Generic::Untyped((void *)impl()));
    }

#if 0
    class DeferredCopy : public Event::Impl::EventWaiter {
    public:
      DeferredCopy(RegionInstance _src, RegionInstance _target,
		   IndexSpace _region,
		   size_t _elmt_size, size_t _bytes_to_copy, Event _after_copy)
	: src(_src), target(_target), region(_region),
	  elmt_size(_elmt_size), bytes_to_copy(_bytes_to_copy), 
	  after_copy(_after_copy) {}

      virtual void event_triggered(void)
      {
	RegionInstance::Impl::copy(src, target, region, 
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
      Memory::Impl *impl = args.mem.impl();

      log_copy.debug("received remote reduction list request: mem=" IDFMT ", offset=%zd, size=%zd, redopid=%d",
		     args.mem.id, args.offset, datalen, args.redopid);

      switch(impl->kind) {
      case Memory::Impl::MKIND_SYSMEM:
      case Memory::Impl::MKIND_ZEROCOPY:
#ifdef USE_CUDA
      case Memory::Impl::MKIND_GPUFB:
#endif
      default:
	assert(0);

      case Memory::Impl::MKIND_GLOBAL:
	{
	  const ReductionOpUntyped *redop = reduce_op_table[args.redopid];
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
	    event = Event::Impl::create_event();

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

#if 0
    /*static*/ Event RegionInstance::Impl::copy(RegionInstance src, 
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
      RegionInstance::Impl *src_impl = impl();
      RegionInstance::Impl *dst_impl = target.impl();

      // figure out which of src or target is the smaller - punt if one
      //  is not a direct ancestor of the other
      IndexSpace src_region = StaticAccess<RegionInstance::Impl>(src_impl)->region;
      IndexSpace dst_region = StaticAccess<RegionInstance::Impl>(dst_impl)->region;

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

      RegionInstance::Impl *src_impl = impl();
      RegionInstance::Impl *dst_impl = target.impl();

      // the region we're being asked to copy must be a subregion (or same
      //  region) of both the src and dst instance's regions
      IndexSpace src_region = StaticAccess<RegionInstance::Impl>(src_impl)->region;
      IndexSpace dst_region = StaticAccess<RegionInstance::Impl>(dst_impl)->region;

      log_copy.info("copy_to_untyped(" IDFMT "(" IDFMT "), " IDFMT "(" IDFMT "), " IDFMT ", " IDFMT "/%d)",
		    id, src_region.id,
		    target.id, dst_region.id, 
		    region.id, wait_on.id, wait_on.gen);

      assert(src_region.impl()->is_parent_of(region));
      assert(dst_region.impl()->is_parent_of(region));

      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *dst_mem = dst_impl->memory.impl();

      log_copy.debug("copy instance: " IDFMT " (%d) -> " IDFMT " (%d), wait=" IDFMT "/%d", id, src_mem->kind, target.id, dst_mem->kind, wait_on.id, wait_on.gen);

      size_t bytes_to_copy, elmt_size;
      {
	StaticAccess<RegionInstance::Impl> src_data(src_impl);
	bytes_to_copy = src_data->region.impl()->instance_size();
	elmt_size = (src_data->is_reduction ?
		       reduce_op_table[src_data->redopid]->sizeof_rhs :
		       StaticAccess<IndexSpace::Impl>(src_data->region.impl())->elmt_size);
      }
      log_copy.debug("COPY " IDFMT " (%d) -> " IDFMT " (%d) - %zd bytes (%zd)", id, src_mem->kind, target.id, dst_mem->kind, bytes_to_copy, elmt_size);

      // check to see if we can access the source memory - if not, we'll send
      //  the request to somebody who can
      if(src_mem->kind == Memory::Impl::MKIND_REMOTE) {
	// plan B: if one side is remote, try delegating to the node
	//  that owns the other side of the copy
	unsigned delegate = ID(src_impl->memory).node();
	assert(delegate != gasnet_mynode());

	log_copy.info("passsing the buck to node %d for " IDFMT "->" IDFMT " copy",
		      delegate, src_mem->me.id, dst_mem->me.id);
	Event after_copy = Event::Impl::create_event();
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
      if((src_mem->kind == Memory::Impl::MKIND_GLOBAL) &&
	 (dst_mem->kind == Memory::Impl::MKIND_REMOTE)) {
	unsigned delegate = ID(dst_impl->memory).node();
	assert(delegate != gasnet_mynode());

	log_copy.info("passsing the buck to node %d for " IDFMT "->" IDFMT " copy",
		      delegate, src_mem->me.id, dst_mem->me.id);
	Event after_copy = Event::Impl::create_event();
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
	Event after_copy = Event::Impl::create_event();
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
      return RegionInstance::Impl::copy(*this, target, region,
					       elmt_size, bytes_to_copy);
    }
#endif

#if 0
#ifdef POINTER_CHECKS
    void RegionAccessor<AccessorGeneric>::verify_access(unsigned ptr) const
    {
      ((RegionInstance::Impl *)internal_data)->verify_access(ptr); 
    }

    void RegionAccessor<AccessorArray>::verify_access(unsigned ptr) const
    {
      ((RegionInstance::Impl *)impl_ptr)->verify_access(ptr);
    }
#endif

    void RegionAccessor<AccessorGeneric>::get_untyped(int index, off_t byte_offset, void *dst, size_t size) const
    {
      ((RegionInstance::Impl *)internal_data)->get_bytes(index, byte_offset, dst, size);
    }

    void RegionAccessor<AccessorGeneric>::put_untyped(int index, off_t byte_offset, const void *src, size_t size) const
    {
      ((RegionInstance::Impl *)internal_data)->put_bytes(index, byte_offset, src, size);
    }

    bool RegionAccessor<AccessorGeneric>::is_reduction_only(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      StaticAccess<RegionInstance::Impl> i_data(i_impl);
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
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's not a reduction fold-only instance
      StaticAccess<RegionInstance::Impl> i_data(i_impl);
      if(i_data->is_reduction) return false;

      // only things in local memory (SYSMEM or ZC) can be converted to
      //   array accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }
#endif

#ifdef OLD_ACCESSORS
    template<>
    RegionAccessor<AccessorArray> RegionAccessor<AccessorGeneric>::convert<AccessorArray>(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstance::Impl> i_data(i_impl);

      assert(!i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) {
	LocalCPUMemory *lcm = (LocalCPUMemory *)m_impl;
	char *inst_base = lcm->base + i_data->access_offset;
	RegionAccessor<AccessorArray> ria(inst_base);
#ifdef POINTER_CHECKS
        ria.set_impl(i_impl);
#endif
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
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
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's a reduction fold-only instance
      StaticAccess<RegionInstance::Impl> i_data(i_impl);
      if(!i_data->is_reduction) return false;
      if(i_data->red_list_size >= 0) return false;

      // only things in local memory (SYSMEM or ZC) can be converted to
      //   array accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }

    template<>
    RegionAccessor<AccessorArrayReductionFold> RegionAccessor<AccessorGeneric>::convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstance::Impl> i_data(i_impl);

      assert(i_data->is_reduction);
      assert(i_data->red_list_size < 0);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) {
	LocalCPUMemory *lcm = (LocalCPUMemory *)m_impl;
	char *inst_base = lcm->base + i_data->access_offset;
	RegionAccessor<AccessorArrayReductionFold> ria(inst_base);
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
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
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      //Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's a reduction fold-only instance
      StaticAccess<RegionInstance::Impl> i_data(i_impl);
      if(!i_data->is_reduction) return false;
      if(i_data->red_list_size < 0) return false;

      // that's the only requirement
      return true;
    }

    template<>
    RegionAccessor<AccessorReductionList> RegionAccessor<AccessorGeneric>::convert<AccessorReductionList>(void) const
    {
      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      //Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstance::Impl> i_data(i_impl);

      assert(i_data->is_reduction);
      assert(i_data->red_list_size >= 0);

      const ReductionOpUntyped *redop = reduce_op_table[i_data->redopid];

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

      RegionInstance::Impl *i_impl = (RegionInstance::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstance::Impl> i_data(i_impl);

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

#ifdef DIEDIEDIE
    class ProcessorThread;

    // internal structures for locks, event, etc.
    class Task {
    public:
      typedef void(*FuncPtr)(const void *args, size_t arglen, Processor *proc);

      Task(FuncPtr _func, const void *_args, size_t _arglen,
	   ProcessorThread *_thread)
	: func(_func), arglen(_arglen), thread(_thread)
      {
	if(arglen) {
	  args = malloc(arglen);
	  memcpy(args, _args, arglen);
	} else {
	  args = 0;
	}
      }

      ~Task(void)
      {
	if(args) free(args);
      }

      void execute(Processor *proc)
      {
	(this->func)(args, arglen, proc);
      }

      FuncPtr func;
      void *args;
      size_t arglen;
      ProcessorThread *thread;
    };

    class ThreadImpl {
    public:
      ThreadImpl(void)
      {
	gasnet_hsl_init(&mutex);
	gasnett_cond_init(&condvar);
      }

      void start(int core_id, const char *debug_name) {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	if(proc_assignment)
	  proc_assignment->bind_thread(core_id, &attr, debug_name);
	CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_main, (void *)this) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
#ifdef DEADLOCK_TRACE
        Runtime::get_runtime()->add_thread(&thread);
#endif
      }

    protected:
      pthread_t thread;
      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;

      virtual void run(void) = 0;

      static void *thread_main(void *data)
      {
	ThreadImpl *me = (ThreadImpl *) data;
	me->run();
	return 0;
      }
    };

    class ProcessorThread : public ThreadImpl {
    public:
      ProcessorThread(int _id, int _core_id)
	: id(_id), core_id(_core_id)
      {
	
      }

      void add_task(Task::FuncPtr func, const void *args, size_t arglen)
      {
	gasnet_hsl_lock(&mutex);
	pending_tasks.push_back(new Task(func, args, arglen, this));
	gasnett_cond_signal(&condvar);
	gasnet_hsl_unlock(&mutex);
      }

    protected:
      friend class LocalProcessor;
      Processor *proc;
      std::list<Task *> pending_tasks;
      int id, core_id;

      virtual void run(void)
      {
	// if(core_id >= 0) {
	//   cpu_set_t cset;
	//   CPU_ZERO(&cset);
	//   CPU_SET(core_id, &cset);
	//   CHECK_PTHREAD( pthread_setaffinity_np(thread, sizeof(cset), &cset) );
	// }

	printf("thread %ld running on core %d\n", thread, core_id);

	// main task loop - grab a task and run it, or sleep if no tasks
	while(1) {
	  printf("here\n"); fflush(stdout);
	  gasnet_hsl_lock(&mutex);
	  if(pending_tasks.size() > 0) {
	    Task *to_run = pending_tasks.front();
	    pending_tasks.pop_front();
	    gasnet_hsl_unlock(&mutex);

	    printf("executing task\n");
	    to_run->execute(proc);
	    delete to_run;
	  } else {
	    printf("sleeping...\n"); fflush(stdout);
	    gasnett_cond_wait(&condvar, &mutex.lock);
	    gasnet_hsl_unlock(&mutex);
	  }
	}
      }
    };

    // since we can't sent active messages from an active message handler,
    //   we drop them into a local circular buffer and send them out later
    class AMQueue {
    public:
      struct AMQueueEntry {
	gasnet_node_t dest;
	gasnet_handler_t handler;
	gasnet_handlerarg_t arg0, arg1, arg2, arg3;
      };

      AMQueue(unsigned _size = 1024)
	: wptr(0), rptr(0), size(_size)
      {
	gasnet_hsl_init(&mutex);
	buffer = new AMQueueEntry[_size];
      }

      ~AMQueue(void)
      {
	delete[] buffer;
      }

      void enqueue(gasnet_node_t dest, gasnet_handler_t handler,
		   gasnet_handlerarg_t arg0 = 0,
		   gasnet_handlerarg_t arg1 = 0,
		   gasnet_handlerarg_t arg2 = 0,
		   gasnet_handlerarg_t arg3 = 0)
      {
	gasnet_hsl_lock(&mutex);
	buffer[wptr].dest = dest;
	buffer[wptr].handler = handler;
	buffer[wptr].arg0 = arg0;
	buffer[wptr].arg1 = arg1;
	buffer[wptr].arg2 = arg2;
	buffer[wptr].arg3 = arg3;
	
	// now advance the write pointer - if we run into the read pointer,
	//  the world ends
	wptr = (wptr + 1) % size;
	assert(wptr != rptr);

	gasnet_hsl_unlock(&mutex);
      }

      void flush(void)
      {
	gasnet_hsl_lock(&mutex);

	while(rptr != wptr) {
	  CHECK_GASNET( gasnet_AMRequestShort4(buffer[rptr].dest,
					       buffer[rptr].handler,
					       buffer[rptr].arg0,
					       buffer[rptr].arg1,
					       buffer[rptr].arg2,
					       buffer[rptr].arg3) );
	  rptr = (rptr + 1) % size;
	}

	gasnet_hsl_unlock(&mutex);
      }

    protected:
      gasnet_hsl_t mutex;
      unsigned wptr, rptr, size;
      AMQueueEntry *buffer;
    };	
#endif

    struct MachineShutdownRequestArgs {
      int initiating_node;
    };

    void handle_machine_shutdown_request(MachineShutdownRequestArgs args)
    {
      Machine *m = Machine::get_machine();

      log_machine.info("received shutdown request from node %d", args.initiating_node);

      m->shutdown(false);
    }

    typedef ActiveMessageShortNoReply<MACHINE_SHUTDOWN_MSGID,
				      MachineShutdownRequestArgs,
				      handle_machine_shutdown_request> MachineShutdownRequestMessage;

    static gasnet_hsl_t announcement_mutex = GASNET_HSL_INITIALIZER;
    static int announcements_received = 0;

    enum {
      NODE_ANNOUNCE_DONE = 0,
      NODE_ANNOUNCE_PROC, // PROC id kind
      NODE_ANNOUNCE_MEM,  // MEM id size
      NODE_ANNOUNCE_PMA,  // PMA proc_id mem_id bw latency
      NODE_ANNOUNCE_MMA,  // MMA mem1_id mem2_id bw latency
    };

    Logger::Category log_annc("announce");

    struct Machine::NodeAnnounceData : public BaseMedium {
      gasnet_node_t node_id;
      unsigned num_procs;
      unsigned num_memories;
    };

    void Machine::parse_node_announce_data(const void *args, size_t arglen,
					   const Machine::NodeAnnounceData& annc_data,
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
	    ID util_id((IDType)*cur++);
	    Processor util = util_id.convert<Processor>();
	    if(remote) {
	      RemoteProcessor *proc = new RemoteProcessor(p, kind, util);
	      Runtime::runtime->nodes[ID(p).node()].processors[ID(p).index()] = proc;
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
	      Runtime::runtime->nodes[ID(m).node()].memories[ID(m).index_h()] = mem;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_PMA:
	  {
	    ProcessorMemoryAffinity pma;
	    pma.p = ID((IDType)*cur++).convert<Processor>();
	    pma.m = ID((IDType)*cur++).convert<Memory>();
	    pma.bandwidth = *cur++;
	    pma.latency = *cur++;

	    proc_mem_affinities.push_back(pma);
	  }
	  break;

	case NODE_ANNOUNCE_MMA:
	  {
	    MemoryMemoryAffinity mma;
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

    void node_announce_handler(Machine::NodeAnnounceData annc_data, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_annc.info("%d: received announce from %d (%d procs, %d memories)\n", gasnet_mynode(), annc_data.node_id, annc_data.num_procs, annc_data.num_memories);
      Node *n = &(Runtime::get_runtime()->nodes[annc_data.node_id]);
      n->processors.resize(annc_data.num_procs);
      n->memories.resize(annc_data.num_memories);

      // do the parsing of this data inside a mutex because it touches common
      //  data structures
      gasnet_hsl_lock(&announcement_mutex);

      Machine::get_machine()->parse_node_announce_data(data, datalen,
						       annc_data, true);

      announcements_received++;
      gasnet_hsl_unlock(&announcement_mutex);
    }

    typedef ActiveMessageMediumNoReply<NODE_ANNOUNCE_MSGID,
				       Machine::NodeAnnounceData,
				       node_announce_handler> NodeAnnounceMessage;

    static std::vector<LocalProcessor *> local_cpus;
    static std::vector<UtilityProcessor *> local_util_procs;
    static size_t stack_size_in_mb;
#ifdef USE_CUDA
    static std::vector<GPUProcessor *> local_gpus;
#endif

    static Machine *the_machine = 0;

#ifdef EVENT_TRACING
    static char *event_trace_file = 0;
#endif
#ifdef LOCK_TRACING
    static char *lock_trace_file = 0;
#endif

    /*static*/ Machine *Machine::get_machine(void) { return the_machine; }

    /*static*/ Processor Machine::get_executing_processor(void) 
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

    static std::map<Processor, std::set<Processor> *> proc_groups;

    ProcessorAssignment::ProcessorAssignment(int _num_local_procs)
      : num_local_procs(_num_local_procs)
    {
      valid = false;
      
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
      size_t core_count = 0;
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
    }

    // binds a thread to the right set of cores based (-1 = not a local proc)
    void ProcessorAssignment::bind_thread(int core_id, pthread_attr_t *attr, const char *debug_name /*= 0*/)
    {
      if(!valid) {
	//printf("no processor assignment for %s %d (%p)\n", debug_name ? debug_name : "unknown", core_id, attr);
	return;
      }

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
    }

#ifdef DEADLOCK_TRACE
    void sigterm_catch(int signal) {
      assert((signal == SIGTERM) || (signal == SIGINT));
#ifdef NODE_LOGGING
      static int call_count = 0;
      int count = __sync_fetch_and_add(&call_count, 1);
      if (count == 0) {
        FILE *log_file = Logger::get_log_file();
        show_event_waiters(log_file);
        Logger::finalize();
      }
#endif
      Runtime *rt = Runtime::get_runtime();
      // Send sig aborts to all the threads
      for (unsigned idx = 0; idx < rt->next_thread; idx++)
        pthread_kill(rt->all_threads[idx], SIGABRT);
    }

    void sigabrt_catch(int signal) {
      assert(signal == SIGABRT);
      // Figure out which index we are, then see if this is
      // the first time we should dumb ourselves
      pthread_t self = pthread_self();
      Runtime *rt = Runtime::get_runtime();
      int index = -1;
      for (int i = 0; i < rt->next_thread; i++)
      {
        if (rt->all_threads[i] == self)
        {
          index = i;
          break;
        }
      }
      int count = -1;
      if (index >= 0)
        count = __sync_fetch_and_add(&rt->thread_counts[index],1);
      if (count == 0)
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
        fprintf(stderr,"BACKTRACE on node %d\n----------\n%s\n----------\n", 
                gasnet_mynode(), buffer);
        free(buffer);
      }
    }
#endif

    ProcessorAssignment *proc_assignment = 0;

    Machine::Machine(int *argc, char ***argv,
		     const Processor::TaskIDTable &task_table,
		     const ReductionOpTable &redop_table,
		     bool cps_style /* = false */,
		     Processor::TaskFuncID init_id /* = 0 */)
      : background_pthread(NULL)
    {
      the_machine = this;

      for(ReductionOpTable::const_iterator it = redop_table.begin();
	  it != redop_table.end();
	  it++)
	reduce_op_table[it->first] = it->second;

      for(Processor::TaskIDTable::const_iterator it = task_table.begin();
	  it != task_table.end();
	  it++)
	task_id_table[it->first] = it->second;

      // have to register domain mappings too
      Arrays::Mapping<1,1>::register_mapping<Arrays::CArrayLinearization<1> >();
      Arrays::Mapping<2,1>::register_mapping<Arrays::CArrayLinearization<2> >();
      Arrays::Mapping<3,1>::register_mapping<Arrays::CArrayLinearization<3> >();
      Arrays::Mapping<1,1>::register_mapping<Arrays::FortranArrayLinearization<1> >();
      Arrays::Mapping<2,1>::register_mapping<Arrays::FortranArrayLinearization<2> >();
      Arrays::Mapping<3,1>::register_mapping<Arrays::FortranArrayLinearization<3> >();
      Arrays::Mapping<1,1>::register_mapping<Translation<1> >();

      // low-level runtime parameters
      size_t gasnet_mem_size_in_mb = 256;
      size_t cpu_mem_size_in_mb = 512;
      size_t reg_mem_size_in_mb = 0;
      size_t zc_mem_size_in_mb = 64;
      size_t fb_mem_size_in_mb = 256;
      // Static variable for stack size since we need to 
      // remember it when we launch threads in run 
      stack_size_in_mb = 2;
      unsigned num_local_cpus = 1;
      unsigned num_local_gpus = 0;
      unsigned num_util_procs = 1;
      unsigned cpu_worker_threads = 1;
      unsigned dma_worker_threads = 1;
      unsigned active_msg_worker_threads = 1;
      bool     active_msg_sender_threads = false;
      bool     gpu_dma_thread = true;
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
      bool pin_sysmem_for_gpu = true;

      for(int i = 1; i < *argc; i++) {
#define INT_ARG(argname, varname) do { \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = atoi((*argv)[++i]);		\
	    continue;					\
	  } } while(0)

#define BOOL_ARG(argname, varname) do { \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = true;				\
	    continue;					\
	  } } while(0)

	INT_ARG("-ll:gsize", gasnet_mem_size_in_mb);
	INT_ARG("-ll:csize", cpu_mem_size_in_mb);
	INT_ARG("-ll:rsize", reg_mem_size_in_mb);
	INT_ARG("-ll:fsize", fb_mem_size_in_mb);
	INT_ARG("-ll:zsize", zc_mem_size_in_mb);
        INT_ARG("-ll:stack", stack_size_in_mb);
	INT_ARG("-ll:cpu", num_local_cpus);
	INT_ARG("-ll:gpu", num_local_gpus);
	INT_ARG("-ll:util", num_util_procs);
	INT_ARG("-ll:workers", cpu_worker_threads);
	INT_ARG("-ll:dma", dma_worker_threads);
	INT_ARG("-ll:amsg", active_msg_worker_threads);
        BOOL_ARG("-ll:gpudma", gpu_dma_thread);
        BOOL_ARG("-ll:senders", active_msg_sender_threads);
	INT_ARG("-ll:bind", bind_localproc_threads);
        INT_ARG("-ll:pin", pin_sysmem_for_gpu);

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
      }

      if(bind_localproc_threads) {
	// this has to preceed all spawning of threads, including the ones done by things like gasnet_init()
	proc_assignment = new ProcessorAssignment(num_local_cpus);

	// now move ourselves off the reserved cores
	proc_assignment->bind_thread(-1, 0, "machine thread");
      }

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
      CHECK_GASNET( gasnet_init(argc, argv) );

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

      Logger::init(*argc, (const char **)*argv);

      gasnet_handlerentry_t handlers[128];
      int hcount = 0;
      hcount += NodeAnnounceMessage::add_handler_entries(&handlers[hcount]);
      hcount += SpawnTaskMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockReleaseMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockGrantMessage::add_handler_entries(&handlers[hcount]);
      hcount += EventSubscribeMessage::add_handler_entries(&handlers[hcount]);
      hcount += EventTriggerMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteMemAllocMessage::add_handler_entries(&handlers[hcount]);
      hcount += CreateInstanceMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteCopyMessage::add_handler_entries(&handlers[hcount]);
      hcount += ValidMaskRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += ValidMaskDataMessage::add_handler_entries(&handlers[hcount]);
      hcount += RollUpRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += RollUpDataMessage::add_handler_entries(&handlers[hcount]);
      hcount += ClearTimerRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += DestroyInstanceMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteWriteMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteReduceMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteWriteFenceMessage::add_handler_entries(&handlers[hcount]);
      hcount += DestroyLockMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteRedListMessage::add_handler_entries(&handlers[hcount]);
      hcount += MachineShutdownRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += BarrierAdjustMessage::Message::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount]);

      init_endpoints(handlers, hcount, 
		     gasnet_mem_size_in_mb, reg_mem_size_in_mb,
		     *argc, (const char **)*argv);

      // Put this here so that it complies with the GASNet specification and
      // doesn't make any calls between gasnet_init and gasnet_attach
      gasnet_set_waitmode(GASNET_WAIT_BLOCK);

      Runtime *r = Runtime::runtime = new Runtime;
      r->nodes = new Node[gasnet_nodes()];

      // create allocators for local node events/locks/index spaces
      {
	Node& n = r->nodes[gasnet_mynode()];
	r->local_event_free_list = new EventTableAllocator::FreeList(n.events, gasnet_mynode());
	r->local_reservation_free_list = new ReservationTableAllocator::FreeList(n.reservations, gasnet_mynode());
	r->local_index_space_free_list = new IndexSpaceTableAllocator::FreeList(n.index_spaces, gasnet_mynode());
	r->local_proc_group_free_list = new ProcessorGroupTableAllocator::FreeList(n.proc_groups, gasnet_mynode());
      }

#ifdef DEADLOCK_TRACE
      r->next_thread = 0;
      signal(SIGTERM, sigterm_catch);
      signal(SIGINT, sigterm_catch);
      signal(SIGABRT, sigabrt_catch);
#endif
      
      start_polling_threads(active_msg_worker_threads);

      start_dma_worker_threads(dma_worker_threads);

      if (active_msg_sender_threads)
        start_sending_threads();

      Clock::synchronize();

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
	r->global_memory = new GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>(), gasnet_mem_size_in_mb << 20);
      else
	r->global_memory = 0;

      Node *n = &r->nodes[gasnet_mynode()];

      NodeAnnounceData announce_data;
      const unsigned ADATA_SIZE = 4096;
      size_t adata[ADATA_SIZE];
      unsigned apos = 0;

      announce_data.node_id = gasnet_mynode();
      announce_data.num_procs = num_local_cpus + num_local_gpus + num_util_procs;
      announce_data.num_memories = (1 + 
				    (reg_mem_size_in_mb > 0 ? 1 : 0) + 
				    2 * num_local_gpus);

      // create utility processors (if any)
      explicit_utility_procs = (num_util_procs > 0);
      if (num_util_procs > 0)
      {
        for(unsigned i = 0; i < num_util_procs; i++) {
          UtilityProcessor *up = new UtilityProcessor(ID(ID::ID_PROCESSOR, 
                                                         gasnet_mynode(), 
                                                         n->processors.size()).convert<Processor>(),
                                                        num_local_cpus+i/*core id*/);

          n->processors.push_back(up);
          local_util_procs.push_back(up);
          adata[apos++] = NODE_ANNOUNCE_PROC;
          adata[apos++] = up->me.id;
          adata[apos++] = Processor::UTIL_PROC;
          adata[apos++] = up->util.id;
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

	LocalProcessor *lp = new LocalProcessor(p, 
						i,
						cpu_worker_threads, 
						1); // HLRT not thread-safe yet
	if(num_util_procs > 0) {
#ifdef SPECIALIZED_UTIL_PROCS
          UtilityProcessor *up = local_util_procs[0];
#else
	  UtilityProcessor *up = local_util_procs[i % num_util_procs];
#endif

	  lp->set_utility_processor(up);

	  // add this processor to that util proc's group
	  std::map<Processor, std::set<Processor> *>::iterator it = proc_groups.find(up->me);
	  if(it != proc_groups.end()) {
	    it->second->insert(p);
	    proc_groups[p] = it->second;
	  } else {
	    std::set<Processor> *pgptr = new std::set<Processor>;
	    pgptr->insert(p);
	    proc_groups[p] = pgptr;
	    proc_groups[up->me] = pgptr;
	  }
	} else {
	  // this processor is in its own group
	  std::set<Processor> *pgptr = new std::set<Processor>;
	  pgptr->insert(p);
	  proc_groups[p] = pgptr;
	}
	n->processors.push_back(lp);
	local_cpus.push_back(lp);
	adata[apos++] = NODE_ANNOUNCE_PROC;
	adata[apos++] = lp->me.id;
	adata[apos++] = Processor::LOC_PROC;
	adata[apos++] = lp->util.id;
	//local_procs[i]->start();
	//machine->add_processor(new LocalProcessor(local_procs[i]));
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

      // list affinities between local CPUs / memories
      for(std::vector<UtilityProcessor *>::iterator it = local_util_procs.begin();
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

	if(r->global_memory) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = r->global_memory->me.id;
	  adata[apos++] = 10;  // "lower" bandwidth
	  adata[apos++] = 50;    // "higher" latency
	}
      }

      // list affinities between local CPUs / memories
      for(std::vector<LocalProcessor *>::iterator it = local_cpus.begin();
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

	if(r->global_memory) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = r->global_memory->me.id;
	  adata[apos++] = 10;  // "lower" bandwidth
	  adata[apos++] = 50;    // "higher" latency
	}
      }

      if((cpu_mem_size_in_mb > 0) && r->global_memory) {
	adata[apos++] = NODE_ANNOUNCE_MMA;
	adata[apos++] = cpumem->me.id;
	adata[apos++] = r->global_memory->me.id;
	adata[apos++] = 30;  // "lower" bandwidth
	adata[apos++] = 25;    // "higher" latency
      }

#ifdef USE_CUDA
      if(num_local_gpus > 0) {
        if (num_local_gpus > (peer_gpus.size() + dumb_gpus.size()))
        {
          printf("Requested %d GPUs, but only %ld GPUs exist on node %d\n",
            num_local_gpus, peer_gpus.size()+dumb_gpus.size(), gasnet_mynode());
          assert(false);
        }
	for(unsigned i = 0; i < num_local_gpus; i++) {
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
	  //printf("GPU's ID is " IDFMT "\n", p.id);
 	  GPUProcessor *gp = new GPUProcessor(p, 
                                              (i < peer_gpus.size() ?
                                                peer_gpus[i] : 
                                                dumb_gpus[i-peer_gpus.size()]), 
                                              num_local_gpus,
#ifdef UTIL_PROCS_FOR_GPU
					      (num_util_procs ?
					         local_util_procs[i % num_util_procs]->me :
					         Processor::NO_PROC),
#else
					      Processor::NO_PROC,
#endif
					      zc_mem_size_in_mb << 20,
					      fb_mem_size_in_mb << 20,
                                              stack_size_in_mb << 20,
                                              gpu_dma_thread);
#ifdef UTIL_PROCS_FOR_GPU
	  if(num_util_procs > 0)
          {
#ifdef SPECIALIZED_UTIL_PROCS
            UtilityProcessor *up = local_util_procs[0];
#else
            UtilityProcessor *up = local_util_procs[i % num_util_procs];
#endif
            gp->set_utility_processor(up);
            std::map<Processor, std::set<Processor>*>::iterator finder = proc_groups.find(up->me);
            if (finder != proc_groups.end())
            {
              finder->second->insert(p);
              proc_groups[p] = finder->second;
            }
            else
            {
              std::set<Processor> *pgptr = new std::set<Processor>();
              pgptr->insert(p);
              proc_groups[p] = pgptr;
              proc_groups[up->me] = pgptr;
            }
          }
          else
#endif
          {
            // This is a GPU processor so make it its own utility processor
            std::set<Processor> *pgptr = new std::set<Processor>();
            pgptr->insert(p);
            proc_groups[p] = pgptr;
          }
	  n->processors.push_back(gp);
	  local_gpus.push_back(gp);

	  adata[apos++] = NODE_ANNOUNCE_PROC;
	  adata[apos++] = p.id;
	  adata[apos++] = Processor::TOC_PROC;
	  adata[apos++] = gp->util.id;

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
	  for(std::vector<LocalProcessor *>::iterator it = local_cpus.begin();
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

        if (gpu_dma_thread)
          GPUProcessor::start_gpu_dma_thread(local_gpus);
      }
#else // ifdef USE_CUDA
      if(num_local_gpus > 0) {
          printf("Requested %d GPUs, but CUDA is not enabled!\n",
            num_local_gpus);
          assert(false);
      }
#endif

      adata[apos++] = NODE_ANNOUNCE_DONE;
      assert(apos < ADATA_SIZE);

      // parse our own data (but don't create remote proc/mem objects)
      {
	AutoHSLLock al(announcement_mutex);
	parse_node_announce_data(adata, apos*sizeof(adata[0]), 
				 announce_data, false);
      }

      // now announce ourselves to everyone else
      for(int i = 0; i < gasnet_nodes(); i++)
	if(i != gasnet_mynode())
	  NodeAnnounceMessage::request(i, announce_data, 
				       adata, apos*sizeof(adata[0]),
				       PAYLOAD_COPY);

      // wait until we hear from everyone else?
      while(announcements_received < (gasnet_nodes() - 1))
	do_some_polling();

      log_annc.info("node %d has received all of its announcements\n", gasnet_mynode());

      // build old proc/mem lists from affinity data
      for(std::vector<ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	procs.insert((*it).p);
	memories.insert((*it).m);
	visible_memories_from_procs[(*it).p].insert((*it).m);
	visible_procs_from_memory[(*it).m].insert((*it).p);
      }
      for(std::vector<MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	memories.insert((*it).m1);
	memories.insert((*it).m2);
	visible_memories_from_memory[(*it).m1].insert((*it).m2);
	visible_memories_from_memory[(*it).m2].insert((*it).m1);
      }
    }

    Machine::~Machine(void)
    {
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
      gasnet_exit(0);
    }

    // Return the set of processors "local" to a given other one
    const std::set<Processor>& Machine::get_local_processors(Processor p) const
    {
      assert(proc_groups.find(p) != proc_groups.end());
      return *(proc_groups[p]);
    }

    Processor::Kind Machine::get_processor_kind(Processor p) const
    {
      return p.impl()->kind;
    }

    Memory::Kind Machine::get_memory_kind(Memory m) const
    {
      return m.impl()->get_kind();
    }

    size_t Machine::get_memory_size(const Memory m) const
    {
      return m.impl()->size;
    }

    size_t Machine::get_address_space_count(void) const
    {
      return gasnet_nodes();
    }

    int Machine::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				       Processor restrict_proc /*= Processor::NO_PROC*/,
				       Memory restrict_memory /*= Memory::NO_MEMORY*/)
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

    int Machine::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
				      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
				      Memory restrict_mem2 /*= Memory::NO_MEMORY*/)
    {
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

    struct MachineRunArgs {
      Machine *m;
      Processor::TaskFuncID task_id;
      Machine::RunStyle style;
      const void *args;
      size_t arglen;
    };  

    static bool running_as_background_thread = false;

    static void *background_run_thread(void *data)
    {
      MachineRunArgs *args = (MachineRunArgs *)data;
      printf("HERE\n");
      running_as_background_thread = true;
      args->m->run(args->task_id, args->style, args->args, args->arglen,
		   false /* foreground from this thread's perspective */);
      printf("THERE\n");
      delete args;
      return 0;
    }

    void Machine::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/,
                      bool background /*= false*/)
    {
      // Create the key for the thread local data
      CHECK_PTHREAD( pthread_key_create(&thread_timer_key,thread_timer_free) );

      if(background) {
        log_machine.info("background operation requested\n");
	fflush(stdout);
	MachineRunArgs *margs = new MachineRunArgs;
	margs->m = this;
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
        Runtime *rt = Runtime::get_runtime();
        rt->add_thread(threadp); 
#endif
	return;
      }

      // now that we've got the machine description all set up, we can start
      //  the worker threads for local processors, which'll probably ask the
      //  high-level runtime to set itself up
      for(std::vector<UtilityProcessor *>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++)
	(*it)->start_worker_threads(stack_size_in_mb << 20);

      for(std::vector<LocalProcessor *>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++)
	(*it)->start_worker_threads(stack_size_in_mb << 20);

#ifdef USE_CUDA
      for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	  it != local_gpus.end();
	  it++)
	(*it)->start_worker_thread(); // stack size passed in GPUProcessor constructor
#endif

      const std::vector<Processor::Impl *>& local_procs = Runtime::runtime->nodes[gasnet_mynode()].processors;
      Atomic<int> running_proc_count(local_procs.size());

      for(std::vector<Processor::Impl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
	(*it)->run(&running_proc_count);

      if(task_id != 0 && 
	 ((style != ONE_TASK_ONLY) || 
	  (gasnet_mynode() == 0))) {//(gasnet_nodes()-1)))) {
	for(std::vector<Processor::Impl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  (*it)->spawn_task(task_id, args, arglen, 
			    Event::NO_EVENT, Event::NO_EVENT, 0/*priority*/);
	  if(style != ONE_TASK_PER_PROC) break;
	}
      }

      // wait for idle-ness somehow?
      int timeout = -1;
#ifdef TRACE_RESOURCES
      Runtime *rt = Runtime::get_runtime();
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
#if defined(ORDERED_LOGGING) || defined(NODE_LOGGING)
      Logger::finalize();
#endif
      log_machine.info("running proc count is now zero - terminating\n");
      // need to kill other threads too so we can actually terminate process
      // Exit out of the thread
      stop_dma_worker_threads();
#ifdef USE_CUDA
      GPUProcessor::stop_gpu_dma_threads();
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

    void Machine::shutdown(bool local_request /*= true*/)
    {
      if(local_request) {
	log_machine.info("shutdown request - notifying other nodes\n");
	MachineShutdownRequestArgs args;
	args.initiating_node = gasnet_mynode();

	for(int i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    MachineShutdownRequestMessage::request(i, args);
      }

      log_machine.info("shutdown request - cleaning up local processors\n");

      const std::vector<Processor::Impl *>& local_procs = Runtime::runtime->nodes[gasnet_mynode()].processors;
      for(std::vector<Processor::Impl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
      {
        Event e = Event::Impl::create_event();
	(*it)->spawn_task(0 /* shutdown task id */, 0, 0,
			  Event::NO_EVENT, e, 0/*priority*/);
      }
    }

    void Machine::wait_for_shutdown(void)
    {
      if (background_pthread != NULL)
      {
        pthread_t *background_thread = (pthread_t*)background_pthread;
        void *result;
        pthread_join(*background_thread, &result);
        free(background_thread);
        // Set this to null so we don't wait anymore
        background_pthread = NULL;
      }
    }

  }; // namespace LowLevel
  // Implementation of logger for low level runtime
#ifdef ORDERED_LOGGING 
  /*static*/ void Logger::finalize(void)
  {
    //assert(lockf(get_log_file(), F_LOCK, logging_buffer_size) == 0);
    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;
    while (fcntl(get_log_file(), F_SETLK, &fl) == -1) { 
#ifdef __SSE2__
      _mm_pause();
#else
      usleep(1000);
#endif
    }
    // Flush the buffer
    assert(write(get_log_file(), get_logging_buffer(), *get_logging_location()) >= 0);
    //assert(lockf(get_log_file(), F_ULOCK, logging_buffer_size) == 0);
    fl.l_type = F_UNLCK;
    assert(fcntl(get_log_file(), F_SETLK, &fl) != -1);
    // Close the file
    close(get_log_file());
    // Free the memory
    free(get_logging_buffer());
  }
#endif

#ifdef NODE_LOGGING
  /*static*/ FILE* Logger::get_log_file(void)
  {
    static FILE *log_file = NULL;
    if (log_file == NULL)
    {
      const char *prefix = ".";
      char file_name[1024];
      sprintf(file_name,"%s/node_%d.log", prefix, gasnet_mynode());
      log_file = fopen(file_name,"w");
      assert(log_file != NULL);
    }
    return log_file;
  }

  /*static*/ void Logger::finalize(void)
  {
    FILE *log_file = get_log_file();
    fclose(log_file);
  }
#endif

  /*static*/ void Logger::logvprintf(LogLevel level, int category, const char *fmt, va_list args)
  {
    char buffer[1000];
    sprintf(buffer, "[%d - %lx] {%d}{%s}: ",
            gasnet_mynode(), pthread_self(), level, Logger::get_categories_by_id()[category].c_str());
    int len = strlen(buffer);
    vsnprintf(buffer+len, 999-len, fmt, args);
    strcat(buffer, "\n");
#ifdef ORDERED_LOGGING 
    // Update the length to reflect the newline character
    len = strlen(buffer);
    long long loc = __sync_fetch_and_add(get_logging_location(),len);
    // Check to see if we can actually fit
    // If we can't keep looping until we're allowed to write
    int possible_end = (loc <= logging_buffer_size) ? int(loc) : -1;
    // There is an implicit assumption here that the spinning threads
    // will not be able to count to 2^64 before the file lock is acquired
    // otherwise we'll get a segfault.  This should be safe, but knowing
    // how slow NFS can be, who knows!
    while ((loc+len) >= logging_buffer_size)
    {
      // Wait until all the writers have finished
      if (possible_end == *((volatile int*)get_written_location()))
      {
        // We're the first ones to update and not fit, so do
        // the write back of the buffer, and then mark that it
        // is ready to use
        // Since there are many possible distributed processes attempting
        // to access this file, we need a file lock. Use the fcntl system
        // call on unix to get a file lock.
        {
          struct flock fl;
          fl.l_type = F_WRLCK;
          fl.l_whence = SEEK_SET;
          fl.l_start = 0;
          fl.l_len = 0;
          while (fcntl(get_log_file(), F_SETLK, &fl) == -1) 
          { 
#ifdef __SSE2__
            _mm_pause();
#else
            usleep(1000);
#endif
#if 0
            struct flock owner; 
            owner.l_type = F_WRLCK;
            owner.l_whence = SEEK_SET;
            owner.l_start = 0;
            owner.l_len = 0;
            assert(fcntl(get_log_file(), F_GETLK, &owner) != -1);
            fprintf(stdout,"Process %d holds the blocking lock\n",owner.l_pid);
            fflush(stdout);
#endif
          }
          //assert(lockf(get_log_file(), F_LOCK, possible_end) == 0);
          assert(write(get_log_file(), get_logging_buffer(), possible_end) >= 0);
          // Finally release the file lock
          //assert(lockf(get_log_file(), F_ULOCK, -possible_end) == 0);
          fl.l_type = F_UNLCK;
          assert(fcntl(get_log_file(), F_SETLK, &fl) != -1);
        }
        // Reset the end written location first
        *((volatile int*)get_written_location()) = 0;
        // Then use compare and swap to reset the logging location
        *((volatile long long*)get_logging_location()) = 0;
      }
      // Now get a new location and see if it works
      long long new_loc = __sync_fetch_and_add(get_logging_location(),len);
      // If new_loc is less than the old_loc need to reset the possible end
      // since the buffer was reset
      if (new_loc < loc)
        possible_end = (loc <= logging_buffer_size) ? int(loc) : -1;
      loc = new_loc;
    }
    // Once we're here, we can just do our write into the buffer and then
    // mark that we did our write
    memcpy(get_logging_buffer()+loc,buffer,len);
    __sync_fetch_and_add(get_written_location(),len);
#else
#ifdef NODE_LOGGING
    FILE *log_file = get_log_file();
    fputs(buffer, log_file);
    fflush(log_file);
#else
    fflush(stdout);
    fputs(buffer, stderr);
#endif
#endif
  }
}; // namespace LegionRuntime

// Implementation of accessor methods
namespace LegionRuntime {
  namespace Accessor {
    using namespace LegionRuntime::LowLevel;

    void AccessorType::Generic::Untyped::read_untyped(ptr_t ptr, void *dst, size_t bytes, off_t offset) const
    {
#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
#endif
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      assert(impl->linearization.valid());
      Arrays::Mapping<1, 1> *mapping = impl->linearization.get_mapping<1>();
      int index = mapping->image(ptr.value);
      impl->get_bytes(index, field_offset + offset, dst, bytes);
    }

    //bool debug_mappings = false;
    void AccessorType::Generic::Untyped::read_untyped(const DomainPoint& dp, void *dst, size_t bytes, off_t offset) const
    {
#ifdef PRIVILEGE_CHECKS 
      check_privileges<ACCESSOR_READ>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, dp);
#endif
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      assert(impl->linearization.valid());
      int index = impl->linearization.get_image(dp);
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
#ifdef PRIVILEGE_CHECKS
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, ptr);
#endif
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      assert(impl->linearization.valid());
      Arrays::Mapping<1, 1> *mapping = impl->linearization.get_mapping<1>();
      int index = mapping->image(ptr.value);
      impl->put_bytes(index, field_offset + offset, src, bytes);
    }

    void AccessorType::Generic::Untyped::write_untyped(const DomainPoint& dp, const void *src, size_t bytes, off_t offset) const
    {
#ifdef PRIVILEGE_CHECKS
      check_privileges<ACCESSOR_WRITE>(priv, region);
#endif
#ifdef BOUNDS_CHECKS
      check_bounds(region, dp);
#endif
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      assert(impl->linearization.valid());
      int index = impl->linearization.get_image(dp);
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
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      return impl->get_strided_parameters(base, stride, field_offset);
#if 0
      Memory::Impl *mem = impl->memory.impl();
      StaticAccess<RegionInstance::Impl> idata(impl);

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
      RegionInstance::Impl *impl = (RegionInstance::Impl *)internal;
      Memory::Impl *mem = impl->memory.impl();
      StaticAccess<RegionInstance::Impl> idata(impl);
      if (idata->redopid == 0) return false;
      if (idata->red_list_size > 0) return false;

      // ReductionFold accessors currently assume packed instances
      size_t stride = idata->elmt_size;
      return impl->get_strided_parameters(base, stride, field_offset);
#if 0
      off_t offset = idata->alloc_offset + field_offset;
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
      RegionInstance::Impl *impl = (RegionInstance::Impl *) impl_ptr;
      internal->verify_access(ptr);
    }
#endif

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset *offsets)
    {
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      Memory::Impl *mem = impl->memory.impl();

      StaticAccess<RegionInstance::Impl> idata(impl);
      if(!impl->linearization.valid()) {
	impl->linearization.deserialize(idata->linearization_bits);
      }
      Arrays::Mapping<DIM, 1> *mapping = impl->linearization.get_mapping<DIM>();

      Point<1> strides[DIM];
      int index = mapping->image_linear_subrect(r, subrect, strides);

      off_t offset = idata->alloc_offset;
      off_t elmt_stride;

      if(idata->block_size == 1) {
	offset += index * idata->elmt_size + field_offset;
	elmt_stride = idata->elmt_size;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(idata->field_sizes, field_offset, 1, field_start, field_size);

	int block_num = index / idata->block_size;
	int block_ofs = index % idata->block_size;

	offset += (((idata->elmt_size * block_num + field_start) * idata->block_size) + 
		   (field_size * block_ofs) +
		   (field_offset - field_start));
	elmt_stride = field_size;
      }

      char *dst = (char *)(mem->get_direct_ptr(offset, subrect.volume() * elmt_stride));
      if(!dst) return 0;

      for(int i = 0; i < DIM; i++)
	offsets[i].offset = strides[i] * elmt_stride;

      return dst;
    }

    template <int DIM>
    void *AccessorType::Generic::Untyped::raw_rect_ptr(const Rect<DIM>& r, Rect<DIM>& subrect, ByteOffset *offsets,
						       const std::vector<off_t> &field_offsets, ByteOffset &field_stride)
    {
      if(field_offsets.size() < 1)
	return 0;

      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      Memory::Impl *mem = impl->memory.impl();

      StaticAccess<RegionInstance::Impl> idata(impl);
      if(!impl->linearization.valid()) {
	impl->linearization.deserialize(idata->linearization_bits);
      }
      Arrays::Mapping<DIM, 1> *mapping = impl->linearization.get_mapping<DIM>();

      Point<1> strides[DIM];
      int index = mapping->image_linear_subrect(r, subrect, strides);

      off_t offset = idata->alloc_offset;
      off_t elmt_stride;
      off_t fld_stride;

      if(idata->block_size == 1) {
	offset += index * idata->elmt_size + field_offset;
	elmt_stride = idata->elmt_size;

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
	find_field_start(idata->field_sizes, field_offset, 1, field_start, field_size);

	int block_num = index / idata->block_size;
	int block_ofs = index % idata->block_size;

	offset += (((idata->elmt_size * block_num + field_start) * idata->block_size) + 
		   (field_size * block_ofs) +
		   (field_offset - field_start));
	elmt_stride = field_size;

	if(field_offsets.size() == 1) {
	  fld_stride = 0;
	} else {
	  off_t field_start2;
	  int field_size2;
	  find_field_start(idata->field_sizes, field_offset + field_offsets[1], 1, field_start2, field_size2);

	  // field sizes much match or element stride isn't consistent
	  if(field_size2 != field_size)
	    return 0;
	  
	  fld_stride = (((field_start2 - field_start) * idata->block_size) + 
			(field_offsets[1] - field_start2) - (field_offsets[0] - field_start));

	  for(size_t i = 2; i < field_offsets.size(); i++) {
	    find_field_start(idata->field_sizes, field_offset + field_offsets[i], 1, field_start2, field_size2);
	    off_t fld_stride2 = (((field_start2 - field_start) * idata->block_size) + 
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
      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      Memory::Impl *mem = impl->memory.impl();

      StaticAccess<RegionInstance::Impl> idata(impl);
      if(!impl->linearization.valid()) {
	impl->linearization.deserialize(idata->linearization_bits);
      }
      Arrays::Mapping<DIM, 1> *mapping = impl->linearization.get_mapping<DIM>();

      int index = mapping->image_dense_subrect(r, subrect);

      off_t offset = idata->alloc_offset;
      off_t elmt_stride;

      if(idata->block_size == 1) {
	offset += index * idata->elmt_size + field_offset;
	elmt_stride = idata->elmt_size;
      } else {
	off_t field_start;
	int field_size;
	find_field_start(idata->field_sizes, field_offset, 1, field_start, field_size);

	int block_num = index / idata->block_size;
	int block_ofs = index % idata->block_size;

	offset += (((idata->elmt_size * block_num + field_start) * idata->block_size) + 
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

      RegionInstance::Impl *impl = (RegionInstance::Impl *) internal;
      Memory::Impl *mem = impl->memory.impl();

      StaticAccess<RegionInstance::Impl> idata(impl);
      if(!impl->linearization.valid()) {
	impl->linearization.deserialize(idata->linearization_bits);
      }
      Arrays::Mapping<DIM, 1> *mapping = impl->linearization.get_mapping<DIM>();

      int index = mapping->image_dense_subrect(r, subrect);

      off_t offset = idata->alloc_offset;
      off_t elmt_stride;
      off_t fld_stride;

      if(idata->block_size == 1) {
	offset += index * idata->elmt_size + field_offset + field_offsets[0];
	elmt_stride = idata->elmt_size;

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
	find_field_start(idata->field_sizes, field_offset + field_offsets[0], 1, field_start, field_size);

	int block_num = index / idata->block_size;
	int block_ofs = index % idata->block_size;

	offset += (((idata->elmt_size * block_num + field_start) * idata->block_size) + 
		   (field_size * block_ofs) +
		   (field_offset + field_offsets[0] - field_start));
	elmt_stride = field_size;

	if(field_offsets.size() == 1) {
	  fld_stride = 0;
	} else {
	  off_t field_start2;
	  int field_size2;
	  find_field_start(idata->field_sizes, field_offset + field_offsets[1], 1, field_start2, field_size2);

	  // field sizes much match or element stride isn't consistent
	  if(field_size2 != field_size)
	    return 0;
	  
	  fld_stride = (((field_start2 - field_start) * idata->block_size) + 
			(field_offsets[1] - field_start2) - (field_offsets[0] - field_start));

	  for(size_t i = 2; i < field_offsets.size(); i++) {
	    find_field_start(idata->field_sizes, field_offset + field_offsets[i], 1, field_start2, field_size2);
	    off_t fld_stride2 = (((field_start2 - field_start) * idata->block_size) + 
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
  };

  namespace Arrays {
    //template<> class Mapping<1,1>;
    template <unsigned IDIM, unsigned ODIM>
    MappingRegistry<IDIM, ODIM> Mapping<IDIM, ODIM>::registry;
  };
};
