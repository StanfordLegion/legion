/* Copyright 2020 Stanford University, NVIDIA Corporation
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

// clocks, timers for Realm

#include "realm/timers.h"
#include "realm/activemsg.h"

#include <string.h>
#include <list>

#ifdef DETAILED_TIMING
pthread_key_t thread_timer_key;
#endif

namespace Realm {

  // if set_zero_time() is not called, relative time will equal absolute time
  /*static*/ long long Clock::zero_time = 0;

#ifdef DETAILED_TIMERS
  ////////////////////////////////////////////////////////////////////////
  //
  // class MultiNodeRollUp
  //

  class MultiNodeRollUp {
  public:
    MultiNodeRollUp(std::map<int,double>& _timers);

    void execute(void);

    void handle_data(const void *data, size_t datalen);

  protected:
    Mutex mutex;
    CondVar condvar;
    std::map<int,double> *timerp;
    volatile int count_left;
  };

    MultiNodeRollUp::MultiNodeRollUp(std::map<int,double>& _timers)
      : condvar(mutex), timerp(&_timers)
    {
      count_left = 0;
    }

    void MultiNodeRollUp::execute(void)
    {
      count_left = max_node_id;

      NodeSet nodes;
      for(NodeID i = 0; i <= max_node_id; i++)
        if(i != my_node_id)
	  nodes.add(i);
      ActiveMessage<TimerDataRequestMessage> amsg(nodes);
      amsg->rollup_ptr = (void*)this;
      amsg.commit();

      // take the lock so that we can safely sleep until all the responses
      //  arrive
      {
	AutoLock<> al(mutex);

	if(count_left > 0)
	  condvar.wait();
      }
      assert(count_left == 0);
    }

    void MultiNodeRollUp::handle_data(const void *data, size_t datalen)
    {
      // have to take mutex here since we're updating shared data
      AutoLock<> a(mutex);

      const double *p = (const double *)data;
      int count = datalen / (2 * sizeof(double));
      for(int i = 0; i < count; i++) {
        int kind; // memcpy preferred over type-punning
	memcpy(&kind, &p[2*i], sizeof(int));
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
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class DetailedTimer (and friends)
  //

    struct TimerStackEntry {
      int timer_kind;
      double start_time;
      double accum_child_time;
    };

#ifdef DETAILED_TIMING
    struct PerThreadTimerData {
    public:
      PerThreadTimerData(void)
      {
        thread = pthread_self();
      }

      pthread_t thread;
      std::list<TimerStackEntry> timer_stack;
      std::map<int, double> timer_accum;
      Mutex mutex;
    };

    Mutex timer_data_mutex;
    std::vector<PerThreadTimerData *> timer_data;

    static void thread_timer_free(void *arg)
    {
      assert(arg != NULL);
      PerThreadTimerData *ptr = (PerThreadTimerData*)arg;
      delete ptr;
    }
#endif

  /*static*/ void DetailedTimer::init_timers(void)
  {
#ifdef DETAILED_TIMING
    // Create the key for the thread local data
#ifndef NDEBUG
    int ret =
#endif
      pthread_key_create(&thread_timer_key,thread_timer_free);
    assert(ret == 0);
#endif
  }

#ifdef DETAILED_TIMING
    /*static*/ void DetailedTimer::clear_timers(bool all_nodes /*= true*/)
    {
      // take global mutex because we need to walk the list
      {
	log_timer.warning("clearing timers");
	AutoLock<> l1(timer_data_mutex);
	for(std::vector<PerThreadTimerData *>::iterator it = timer_data.begin();
	    it != timer_data.end();
	    it++) {
	  // take each thread's data's lock too
	  AutoLock<> l2((*it)->mutex);
	  (*it)->timer_accum.clear();
	}
      }

      // if we've been asked to clear other nodes too, send a message
      if(all_nodes) {
	NodeSet nodes;
	for(NodeID i = 0; i < max_node_id; i++)
	  if(i != my_node_id)
	    nodes.add(i);
	ActiveMessage<ClearTimersMessage> amsg(nodes);
	amsg->dummy = 0;
	amsg.commit();
      }
    }

    /*static*/ void DetailedTimer::push_timer(int timer_kind)
    {
      PerThreadTimerData *thread_timer_data = 
        (PerThreadTimerData*) pthread_getspecific(thread_timer_key);
      if(!thread_timer_data) {
        //printf("creating timer data for thread %lx\n", pthread_self());
        AutoLock<> l1(timer_data_mutex);
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
        AutoLock<> l1(thread_timer_data->mutex);

        std::map<int,double>::iterator it = thread_timer_data->timer_accum.find(old_top.timer_kind);
        if(it != thread_timer_data->timer_accum.end())
          it->second += elapsed;
        else
          thread_timer_data->timer_accum.insert(std::make_pair<int,double>(old_top.timer_kind, elapsed));
      }
    }

    /*static*/ void DetailedTimer::roll_up_timers(std::map<int, double>& timers,
                                                  bool local_only)
    {
      // take global mutex because we need to walk the list
      {
	AutoLock<> l1(timer_data_mutex);
	for(std::vector<PerThreadTimerData *>::iterator it = timer_data.begin();
	    it != timer_data.end();
	    it++) {
	  // take each thread's data's lock too
	  AutoLock<> l2((*it)->mutex);

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


  ////////////////////////////////////////////////////////////////////////
  //
  // class ClearTimersMessage
  //

  /*static*/ void ClearTimersMessage::handle_message(NodeID sender,
						     const ClearTimerMessage &args,
						     const void *data,
						     size_t datalen)
  {
    DetailedTimer::clear_timers(false);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class TimerDataRequestMessage
  //

  /*static*/ void TimerDataRequestMessage::handle_message(NodeID sender,
							  const TimerDataRequestMessage &args,
							  const void *data,
							  size_t datalen)
  {
    std::map<int,double> timers;
    DetailedTimer::roll_up_timers(timers, true);

    double return_data[200];
    int count = 0;
    for(std::map<int,double>::iterator it = timers.begin();
	it != timers.end();
	it++) {
      // use memcpy instead of type-punning
      //*(int *)(&return_data[count]) = it->first;
      return_data[count] = 0;
      memcpy(&return_data[count], &(it->first), sizeof(int));
      return_data[count+1] = it->second;
      count += 2;
    }
    assert(count <= 200);

    ActiveMessage<TimerDataResponseMessage> amsg(sender, count*sizeof(double));
    amsg->rollup_ptr = args.rollup_ptr;
    amsg.add_payload(return_data, count*sizeof(double));
    amsg.commit();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class TimerDataResponseMessage
  //

  /*static*/ void TimerDataResponseMessage::handle_message(NodeID sender,
							   const TimerDataResponseMessage &args,
							   const void *data,
							   size_t datalen)

  {
    ((MultiNodeRollUp *)args.rollup_ptr)->handle_data(data, datalen); 
  }

  ActiveMessageHandlerReg<ClearTimersMessage> clear_timers_message_handler;
  ActiveMessageHandlerReg<TimerDataRequestMessage> timers_data_request_message_handler;
  ActiveMessageHandlerReg<TimerDataResponseMessage> timer_data_response_message_handler;

#endif  

}; // namespace Realm
