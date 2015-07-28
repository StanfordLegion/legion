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

#ifdef USE_CUDA
GASNETT_THREADKEY_DECLARE(gpu_thread_ptr);
#endif

pthread_key_t thread_timer_key;

// Implementation of Detailed Timer
namespace LegionRuntime {
  namespace LowLevel {

    
#ifdef USE_CUDA
    Logger::Category log_gpu("gpu");
#endif
    Logger::Category log_mutex("mutex");
    Logger::Category log_timer("timer");
    Logger::Category log_region("region");
    Logger::Category log_machine("machine");
#ifdef EVENT_GRAPH_TRACE
    Logger::Category log_event_graph("graph");
#endif
    Logger::Category log_meta("meta");

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

    RuntimeImpl *runtime_singleton = 0;
#ifdef NODE_LOGGING
    /*static*/ const char* RuntimeImpl::prefix = ".";
#endif

    //static const unsigned MAX_LOCAL_EVENTS = 300000;
    //static const unsigned MAX_LOCAL_LOCKS = 100000;

    Node::Node(void)
    {
    }

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
	  Realm::StaticAccess<IndexSpaceImpl> pdata(get_runtime()->get_index_space_impl(_parent));
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
	other = Realm::StaticAccess<IndexSpaceImpl>(other_impl)->parent;
      }
      return false;
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

    Event IndexSpaceImpl::request_valid_mask(void)
    {
      size_t num_elmts = Realm::StaticAccess<IndexSpaceImpl>(this)->num_elmts;
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
      
      //Realm::StaticAccess<IndexSpaceImpl> rdata(_is.impl());
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
    // Reservations 

    //    /*static*/ ReservationImpl *ReservationImpl::first_free = 0;
    //    /*static*/ GASNetHSL ReservationImpl::freelist_mutex;


  };
};

namespace Realm {

  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;


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

    static Logger::Category log_copy("copy");









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

      Realm::StaticAccess<IndexSpaceImpl> p_data(get_runtime()->get_index_space_impl(parent));

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
	IndexSpaceImpl *r = get_runtime()->get_index_space_impl(get_index_space());

	Realm::StaticAccess<IndexSpaceImpl> data(r);
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
      Realm::ProfilingRequestSet requests;

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

      size_t inst_bytes = elem_size * num_elements;
      RegionInstance i = hdf_mem->create_instance(get_index_space(), linearization_bits, inst_bytes, 
                                                  1/*block_size*/, elem_size, field_sizes,
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
      Realm::SharedAccess<IndexSpaceImpl> data(r_impl);
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
      Realm::SharedAccess<IndexSpaceImpl> is_data(is_impl);
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
	Realm::SharedAccess<IndexSpaceImpl> is_data(impl);
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
	Realm::SharedAccess<IndexSpaceImpl> is_data(impl);
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
      Realm::StaticAccess<RegionInstanceImpl> data(this);
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

    static std::vector<ProcessorImpl*> local_cpus;
    static std::vector<ProcessorImpl*> local_util_procs;
    static std::vector<ProcessorImpl*> local_io_procs;
    static size_t stack_size_in_mb;
#ifdef USE_CUDA
    static std::vector<GPUProcessor *> local_gpus;
    static std::map<GPUProcessor *, GPUFBMemory *> gpu_fbmems;
    static std::map<GPUProcessor *, GPUZCMemory *> gpu_zcmems;
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
      machine = new MachineImpl;
    }

    RuntimeImpl::~RuntimeImpl(void)
    {
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
	Realm::proc_assignment = new Realm::ProcessorAssignment(num_local_cpus);

	// now move ourselves off the reserved cores
	Realm::proc_assignment->bind_thread(-1, 0, "machine thread");
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
      BarrierImpl::barrier_adjustment_timestamp = (((Barrier::timestamp_t)(gasnet_mynode())) << BarrierImpl::BARRIER_TIMESTAMP_NODEID_SHIFT) + 1;

      Realm::Logger::configure_from_cmdline(*argc, (const char **)*argv);

      gasnet_handlerentry_t handlers[128];
      int hcount = 0;
      hcount += Realm::NodeAnnounceMessage::Message::add_handler_entries(&handlers[hcount], "Node Announce AM");
      hcount += Realm::SpawnTaskMessage::Message::add_handler_entries(&handlers[hcount], "Spawn Task AM");
      hcount += Realm::LockRequestMessage::Message::add_handler_entries(&handlers[hcount], "Lock Request AM");
      hcount += Realm::LockReleaseMessage::Message::add_handler_entries(&handlers[hcount], "Lock Release AM");
      hcount += Realm::LockGrantMessage::Message::add_handler_entries(&handlers[hcount], "Lock Grant AM");
      hcount += Realm::EventSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Event Subscribe AM");
      hcount += Realm::EventTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Event Trigger AM");
      hcount += Realm::RemoteMemAllocRequest::Request::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Request AM");
      hcount += Realm::RemoteMemAllocRequest::Response::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Response AM");
      hcount += Realm::CreateInstanceRequest::Request::add_handler_entries(&handlers[hcount], "Create Instance Request AM");
      hcount += Realm::CreateInstanceRequest::Response::add_handler_entries(&handlers[hcount], "Create Instance Response AM");
      hcount += RemoteCopyMessage::add_handler_entries(&handlers[hcount], "Remote Copy AM");
      hcount += RemoteFillMessage::add_handler_entries(&handlers[hcount], "Remote Fill AM");
      hcount += ValidMaskRequestMessage::add_handler_entries(&handlers[hcount], "Valid Mask Request AM");
      hcount += ValidMaskDataMessage::add_handler_entries(&handlers[hcount], "Valid Mask Data AM");
      hcount += RollUpRequestMessage::add_handler_entries(&handlers[hcount], "Roll-up Request AM");
      hcount += RollUpDataMessage::add_handler_entries(&handlers[hcount], "Roll-up Data AM");
      hcount += ClearTimerRequestMessage::add_handler_entries(&handlers[hcount], "Clear Timer Request AM");
      hcount += Realm::DestroyInstanceMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Instance AM");
      hcount += Realm::RemoteWriteMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write AM");
      hcount += Realm::RemoteReduceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Reduce AM");
      hcount += Realm::RemoteWriteFenceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write Fence AM");
      hcount += Realm::DestroyLockMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Lock AM");
      hcount += RemoteRedListMessage::add_handler_entries(&handlers[hcount], "Remote Reduction List AM");
      hcount += MachineShutdownRequestMessage::add_handler_entries(&handlers[hcount], "Machine Shutdown AM");
      hcount += Realm::BarrierAdjustMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Adjust AM");
      hcount += Realm::BarrierSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Subscribe AM");
      hcount += Realm::BarrierTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Trigger AM");
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

      start_dma_worker_threads(dma_worker_threads);

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
	global_memory = new Realm::GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>(), gasnet_mem_size_in_mb << 20);
      else
	global_memory = 0;

      Node *n = &nodes[gasnet_mynode()];

      // create utility processors (if any)
      if (num_util_procs > 0)
      {
        for(unsigned i = 0; i < num_util_procs; i++) {
          ProcessorImpl *up;
          if (use_greenlet_procs)
            up = new Realm::GreenletProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), 
                                    n->processors.size()).convert<Processor>(),
                                    Processor::UTIL_PROC, stack_size_in_mb << 20, 
                                    init_stack_count, "utility worker");
          else
            up = new Realm::LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), 
                                    n->processors.size()).convert<Processor>(),
                                    Processor::UTIL_PROC, 
                                    stack_size_in_mb << 20, "utility worker");
          n->processors.push_back(up);
          local_util_procs.push_back(up);
        }
      }
      // create i/o processors (if any)
      if (num_io_procs > 0)
      {
        for (unsigned i = 0; i < num_io_procs; i++) {
          Realm::LocalProcessor *io = new Realm::LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(),
                                            n->processors.size()).convert<Processor>(),
                                            Processor::IO_PROC,
                                            stack_size_in_mb << 20, "io worker");
          n->processors.push_back(io);
          local_io_procs.push_back(io);
        }
      }

#ifdef USE_CUDA
      // Initialize the driver API
      CHECK_CU( cuInit(0) );
      // Keep track of the local system memories so we can pin them
      // after we've initialized the GPU
      std::vector<Realm::LocalCPUMemory*> local_mems;
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
          lp = new Realm::GreenletProcessor(p, Processor::LOC_PROC,
                                     stack_size_in_mb << 20, init_stack_count,
                                     "local worker", i);
        else
	  lp = new Realm::LocalProcessor(p, Processor::LOC_PROC,
                                  stack_size_in_mb << 20,
                                  "local worker", i);
	n->processors.push_back(lp);
	local_cpus.push_back(lp);
      }

      // create local memory
      Realm::LocalCPUMemory *cpumem;
      if(cpu_mem_size_in_mb > 0) {
	cpumem = new Realm::LocalCPUMemory(ID(ID::ID_MEMORY, 
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    cpu_mem_size_in_mb << 20);
	n->memories.push_back(cpumem);
#ifdef USE_CUDA
        local_mems.push_back(cpumem);
#endif
      } else
	cpumem = 0;

      Realm::LocalCPUMemory *regmem;
      if(reg_mem_size_in_mb > 0) {
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );
	char *regmem_base = ((char *)(seginfos[gasnet_mynode()].addr)) + (gasnet_mem_size_in_mb << 20);
	delete[] seginfos;
	regmem = new Realm::LocalCPUMemory(ID(ID::ID_MEMORY,
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    reg_mem_size_in_mb << 20,
				    regmem_base,
				    true);
	n->memories.push_back(regmem);
#ifdef USE_CUDA
        local_mems.push_back(regmem);
#endif
      } else
	regmem = 0;

      // create local disk memory
      Realm::DiskMemory *diskmem;
      if(disk_mem_size_in_mb > 0) {
        diskmem = new Realm::DiskMemory(ID(ID::ID_MEMORY,
                                    gasnet_mynode(),
                                    n->memories.size(), 0).convert<Memory>(),
                                 disk_mem_size_in_mb << 20,
                                 "disk_file.tmp");
        n->memories.push_back(diskmem);
      } else
        diskmem = 0;

#ifdef USE_HDF
      // create HDF memory
      Realm::HDFMemory *hdfmem;
      hdfmem = new Realm::HDFMemory(ID(ID::ID_MEMORY,
                                gasnet_mynode(),
                                n->memories.size(), 0).convert<Memory>());
      n->memories.push_back(hdfmem);
#endif



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

	  Memory m = ID(ID::ID_MEMORY,
			gasnet_mynode(),
			n->memories.size(), 0).convert<Memory>();
	  GPUFBMemory *fbm = new GPUFBMemory(m, gp);
	  n->memories.push_back(fbm);

	  gpu_fbmems[gp] = fbm;

	  Memory m2 = ID(ID::ID_MEMORY,
			 gasnet_mynode(),
			 n->memories.size(), 0).convert<Memory>();
	  GPUZCMemory *zcm = new GPUZCMemory(m2, gp);
	  n->memories.push_back(zcm);

	  gpu_zcmems[gp] = zcm;
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

      {
	const unsigned ADATA_SIZE = 4096;
	size_t adata[ADATA_SIZE];
	unsigned apos = 0;

	unsigned num_procs = 0;
	unsigned num_memories = 0;

	for(std::vector<ProcessorImpl *>::const_iterator it = local_util_procs.begin();
	    it != local_util_procs.end();
	    it++) {
	  num_procs++;
          adata[apos++] = Realm::NODE_ANNOUNCE_PROC;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = Processor::UTIL_PROC;
	}

	for(std::vector<ProcessorImpl *>::const_iterator it = local_io_procs.begin();
	    it != local_io_procs.end();
	    it++) {
	  num_procs++;
          adata[apos++] = Realm::NODE_ANNOUNCE_PROC;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = Processor::IO_PROC;
	}

	for(std::vector<ProcessorImpl *>::const_iterator it = local_cpus.begin();
	    it != local_cpus.end();
	    it++) {
	  num_procs++;
          adata[apos++] = Realm::NODE_ANNOUNCE_PROC;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = Processor::LOC_PROC;
	}

	// memories
	if(cpumem) {
	  num_memories++;
	  adata[apos++] = Realm::NODE_ANNOUNCE_MEM;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = Memory::SYSTEM_MEM;
	  adata[apos++] = cpumem->size;
	  adata[apos++] = 0; // not registered
	}

	if(regmem) {
	  num_memories++;
	  adata[apos++] = Realm::NODE_ANNOUNCE_MEM;
	  adata[apos++] = regmem->me.id;
	  adata[apos++] = Memory::REGDMA_MEM;
	  adata[apos++] = regmem->size;
	  adata[apos++] = (size_t)(regmem->base);
	}

	if(diskmem) {
	  num_memories++;
	  adata[apos++] = Realm::NODE_ANNOUNCE_MEM;
	  adata[apos++] = diskmem->me.id;
	  adata[apos++] = Memory::DISK_MEM;
	  adata[apos++] = diskmem->size;
	  adata[apos++] = 0;
	}

#ifdef USE_HDF
	if(hdfmem) {
	  num_memories++;
	  adata[apos++] = Realm::NODE_ANNOUNCE_MEM;
	  adata[apos++] = hdfmem->me.id;
	  adata[apos++] = Memory::HDF_MEM;
	  adata[apos++] = hdfmem->size;
	  adata[apos++] = 0;
	}
#endif

	// list affinities between local CPUs / memories
	std::vector<ProcessorImpl *> all_local_procs;
	all_local_procs.insert(all_local_procs.end(),
			       local_util_procs.begin(), local_util_procs.end());
	all_local_procs.insert(all_local_procs.end(),
			       local_io_procs.begin(), local_io_procs.end());
	all_local_procs.insert(all_local_procs.end(),
			       local_cpus.begin(), local_cpus.end());
	for(std::vector<ProcessorImpl*>::iterator it = all_local_procs.begin();
	    it != all_local_procs.end();
	    it++) {
	  if(cpumem) {
	    adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = cpumem->me.id;
	    adata[apos++] = 100;  // "large" bandwidth
	    adata[apos++] = 1;    // "small" latency
	  }

	  if(regmem) {
	    adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = regmem->me.id;
	    adata[apos++] = 80;  // "large" bandwidth
	    adata[apos++] = 5;    // "small" latency
	  }

	  if(diskmem) {
	    adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = diskmem->me.id;
	    adata[apos++] = 5;  // "low" bandwidth
	    adata[apos++] = 100;  // "high" latency
	  }

#ifdef USE_HDF
	  if(hdfmem) {
	    adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = hdfmem->me.id;
	    adata[apos++] = 5; // "low" bandwidth
	    adata[apos++] = 100; // "high" latency
	  } 
#endif

	  if(global_memory) {
  	    adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = global_memory->me.id;
	    adata[apos++] = 10;  // "lower" bandwidth
	    adata[apos++] = 50;    // "higher" latency
	  }
	}

	if(cpumem && global_memory) {
	  adata[apos++] = Realm::NODE_ANNOUNCE_MMA;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = global_memory->me.id;
	  adata[apos++] = 30;  // "lower" bandwidth
	  adata[apos++] = 25;    // "higher" latency
	}

	if(cpumem && diskmem) {
	  adata[apos++] = Realm::NODE_ANNOUNCE_MMA;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = diskmem->me.id;
	  adata[apos++] = 15;    // "low" bandwidth
	  adata[apos++] = 50;    // "high" latency
	}

#ifdef USE_CUDA
	for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	    it != local_gpus.end();
	    it++)
	{
	  num_procs++;
	  adata[apos++] = Realm::NODE_ANNOUNCE_PROC;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = Processor::TOC_PROC;

	  GPUFBMemory *fbm = gpu_fbmems[*it];
	  if(fbm) {
	    num_memories++;

	    adata[apos++] = Realm::NODE_ANNOUNCE_MEM;
	    adata[apos++] = fbm->me.id;
	    adata[apos++] = Memory::GPU_FB_MEM;
	    adata[apos++] = fbm->size;
	    adata[apos++] = 0; // not registered

	    // FB has very good bandwidth and ok latency to GPU
	    adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = fbm->me.id;
	    adata[apos++] = 200; // "big" bandwidth
	    adata[apos++] = 5;   // "ok" latency
	  }

	  GPUZCMemory *zcm = gpu_zcmems[*it];
	  if(zcm) {
	    num_memories++;

	    adata[apos++] = Realm::NODE_ANNOUNCE_MEM;
	    adata[apos++] = zcm->me.id;
	    adata[apos++] = Memory::Z_COPY_MEM;
	    adata[apos++] = zcm->size;
	    adata[apos++] = 0; // not registered

	    // ZC has medium bandwidth and bad latency to GPU
	    adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = zcm->me.id;
	    adata[apos++] = 20;
	    adata[apos++] = 200;

	    // ZC also accessible to all the local CPUs
	    for(std::vector<ProcessorImpl*>::iterator it2 = local_cpus.begin();
		it2 != local_cpus.end();
		it2++) {
	      adata[apos++] = Realm::NODE_ANNOUNCE_PMA;
	      adata[apos++] = (*it2)->me.id;
	      adata[apos++] = zcm->me.id;
	      adata[apos++] = 40;
	      adata[apos++] = 3;
	    }
	  }
	}
#endif

	adata[apos++] = Realm::NODE_ANNOUNCE_DONE;
	assert(apos < ADATA_SIZE);

	// parse our own data (but don't create remote proc/mem objects)
	machine->parse_node_announce_data(gasnet_mynode(),
					  num_procs,
					  num_memories,
					  adata, apos*sizeof(adata[0]), 
					  false);

#ifdef DEBUG_REALM_STARTUP
	if(gasnet_mynode() == 0) {
	  LegionRuntime::TimeStamp ts("sending announcements", false);
	  fflush(stdout);
	}
#endif

	// now announce ourselves to everyone else
	for(unsigned i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    Realm::NodeAnnounceMessage::send_request(i,
						     num_procs,
						     num_memories,
						     adata, apos*sizeof(adata[0]),
						     PAYLOAD_COPY);

	Realm::NodeAnnounceMessage::await_all_announcements();

#ifdef DEBUG_REALM_STARTUP
	if(gasnet_mynode() == 0) {
	  LegionRuntime::TimeStamp ts("received all announcements", false);
	  fflush(stdout);
	}
#endif
      }

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
      Realm::Atomic<int> running_proc_count(local_procs.size());

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
      stop_dma_worker_threads();
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
      Realm::StaticAccess<RegionInstanceImpl> idata(impl);

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
