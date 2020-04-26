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

#include "realm/realm_config.h"
#include "realm/atomics.h"

#include "realm/activemsg.h"
#include "realm/mutex.h"
#include "realm/cmdline.h"
#include "realm/logging.h"

#include <math.h>

namespace Realm {

  Realm::Logger log_amhandler("amhandler");

  namespace Config {
    // if true, the number and min/max/avg/stddev duration of handler per
    //  message type is recorded and printed
    bool profile_activemsg_handlers = false;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // struct ActiveMessageHandlerStats
  //

  ActiveMessageHandlerStats::ActiveMessageHandlerStats(void)
    : count(0), sum(0), sum2(0), minval(0), maxval(0)
  {}

  void ActiveMessageHandlerStats::record(long long t_start, long long t_end)
  {
    // TODO: thread safety?
    long long delta = t_end - t_start;
    if(delta < 0) delta = 0;
    size_t val = (delta > 0) ? delta : 0;
    if(!count || (val < minval)) minval = val;
    if(!count || (val > maxval)) maxval = val;
    count++;
    sum += val;
    sum2 += val * val; // TODO: smarter math to avoid overflow
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessageHandlerTable
  //

  ActiveMessageHandlerTable::ActiveMessageHandlerTable(void)
  {}

  ActiveMessageHandlerTable::~ActiveMessageHandlerTable(void)
  {
    for(std::vector<HandlerEntry>::iterator it = handlers.begin();
	it != handlers.end();
	++it)
      if(it->must_free)
	free(const_cast<char *>(it->name));
  }

  ActiveMessageHandlerTable::HandlerEntry *ActiveMessageHandlerTable::lookup_message_handler(ActiveMessageHandlerTable::MessageID id)
  {
    assert(id < handlers.size());
    return &handlers[id];
  }

  const char *ActiveMessageHandlerTable::lookup_message_name(ActiveMessageHandlerTable::MessageID id)
  {
    assert(id < handlers.size());
    return handlers[id].name;
  }

  void ActiveMessageHandlerTable::record_message_handler_call(MessageID id,
							      long long t_start,
							      long long t_end)
  {
    assert(id < handlers.size());
    handlers[id].stats.record(t_start, t_end);
  }

  void ActiveMessageHandlerTable::report_message_handler_stats()
  {
    if(Config::profile_activemsg_handlers) {
      for(size_t i = 0; i < handlers.size(); i++) {
	const ActiveMessageHandlerStats& stats = handlers[i].stats;
	if(stats.count == 0)
	  continue;

	double avg = double(stats.sum) / double(stats.count);
	double stddev = sqrt((double(stats.sum2) / double(stats.count)) -
			     (avg * avg));
	log_amhandler.print() << "handler " << std::hex << i << std::dec << ": " << handlers[i].name
			      << " count=" << stats.count
			      << " avg=" << avg
			      << " dev=" << stddev
			      << " min=" << stats.minval
			      << " max=" << stats.maxval;
      }
    }
  }

  /*static*/ void ActiveMessageHandlerTable::append_handler_reg(ActiveMessageHandlerRegBase *new_reg)
  {
    new_reg->next_handler = pending_handlers;
    pending_handlers = new_reg;
  }

  static inline bool hash_less(const ActiveMessageHandlerTable::HandlerEntry &a,
			       const ActiveMessageHandlerTable::HandlerEntry &b)
  {
    return (a.hash < b.hash);
  }

  void ActiveMessageHandlerTable::construct_handler_table(void)
  {
    for(ActiveMessageHandlerRegBase *nextreg = pending_handlers;
	nextreg;
	nextreg = nextreg->next_handler) {
      HandlerEntry e;
      e.hash = nextreg->hash;
      e.name = nextreg->name;
      e.must_free = nextreg->must_free;
      e.handler = nextreg->get_handler();
      handlers.push_back(e);
    }

    std::sort(handlers.begin(), handlers.end(), hash_less);

    // handler ids are the same everywhere, so only log on node 0
    if(Network::my_node_id == 0)
      for(size_t i = 0; i < handlers.size(); i++)
	log_amhandler.info() << "handler " << std::hex << i << std::dec << ": " << handlers[i].name;
  }

  /*static*/ ActiveMessageHandlerRegBase *ActiveMessageHandlerTable::pending_handlers = 0;

  /*extern*/ ActiveMessageHandlerTable activemsg_handler_table;


  ////////////////////////////////////////////////////////////////////////
  //
  // class IncomingMessageManager
  //

  namespace ThreadLocal {
    // this flag will be true when we are running a message handler
    REALM_THREAD_LOCAL bool in_message_handler = false;
  };

  IncomingMessageManager::IncomingMessageManager(int _nodes,
						 int _dedicated_threads,
						 Realm::CoreReservationSet& crs)
    : BackgroundWorkItem("activemsg handler")
    , nodes(_nodes), dedicated_threads(_dedicated_threads)
    , sleeper_count(0)
    , bgwork_requested(false)
    , shutdown_flag(0), condvar(mutex)
  {
    heads = new Message *[nodes];
    tails = new Message **[nodes];
    in_handler = new bool[nodes];
    for(int i = 0; i < nodes; i++) {
      heads[i] = 0;
      tails[i] = 0;
      in_handler[i] = false;
    }
    todo_list = new int[nodes + 1];  // an extra entry to distinguish full from empty
    todo_oldest = todo_newest = 0;

    if(dedicated_threads > 0)
      core_rsrv = new Realm::CoreReservation("AM handlers", crs,
					     Realm::CoreReservationParameters());
    else
      core_rsrv = 0;
  }

  IncomingMessageManager::~IncomingMessageManager(void)
  {
    delete core_rsrv;
    delete[] heads;
    delete[] tails;
    delete[] in_handler;
    delete[] todo_list;
  }

  // like strdup, but works on arbitrary byte arrays
  static void *bytedup(const void *data, size_t datalen)
  {
    if(datalen == 0) return 0;
    void *dst = malloc(datalen);
    assert(dst != 0);
    memcpy(dst, data, datalen);
    return dst;
  }

  bool IncomingMessageManager::add_incoming_message(NodeID sender,
						    ActiveMessageHandlerTable::MessageID msgid,
						    const void *hdr, size_t hdr_size,
						    int hdr_mode,
						    const void *payload, size_t payload_size,
						    int payload_mode,
						    CallbackFnptr callback_fnptr,
						    CallbackData callback_data)
  {
#ifdef DEBUG_INCOMING
    printf("adding incoming message from %d\n", sender);
#endif

    // look up which message this is
    ActiveMessageHandlerTable::HandlerEntry *handler = activemsg_handler_table.lookup_message_handler(msgid);

    // TODO: try to handle inline

    // can't handle inline - need to create a Message object for it
    // TODO: recycle these!
    Message *msg = new Message;
    msg->next_msg = 0;
    msg->sender = sender;
    msg->handler = handler;
    msg->hdr = ((hdr_mode == PAYLOAD_COPY) ?
		  bytedup(hdr, hdr_size) :
		  const_cast<void *>(hdr));
    msg->hdr_size = hdr_size;
    msg->hdr_needs_free = (hdr_mode != PAYLOAD_KEEP);
    msg->payload = ((payload_mode == PAYLOAD_COPY) ?
		      bytedup(payload, payload_size) :
		      const_cast<void *>(payload));
    msg->payload_size = payload_size;
    msg->payload_needs_free = (payload_mode != PAYLOAD_KEEP);
    msg->callback_fnptr = callback_fnptr;
    msg->callback_data = callback_data;
	  
    mutex.lock();
    if(heads[sender]) {
      // tack this on to the existing list
      assert(tails[sender]);
      *(tails[sender]) = msg;
      tails[sender] = &(msg->next_msg);
    } else {
      // this starts a list, and the node needs to be added to the todo list
      heads[sender] = msg;
      tails[sender] = &(msg->next_msg);

      // enqueue if this sender isn't currently being handled
      if(!in_handler[sender]) {
	bool was_empty = todo_oldest == todo_newest;

	todo_list[todo_newest] = sender;
	todo_newest++;
	if(todo_newest > nodes)
	  todo_newest = 0;
	assert(todo_newest != todo_oldest);  // should never wrap around
	if(sleeper_count > 0)
	  condvar.broadcast();  // wake up any sleepers

	if(was_empty && !bgwork_requested.load()) {
	  bgwork_requested.store(true);
	  make_active();
	}
      }
    }
    mutex.unlock();

    return false;  // not handled right away
  }

  void IncomingMessageManager::start_handler_threads(size_t stack_size)
  {
    handler_threads.resize(dedicated_threads);

    Realm::ThreadLaunchParameters tlp;
    tlp.set_stack_size(stack_size);

    for(int i = 0; i < dedicated_threads; i++)
      handler_threads[i] = Realm::Thread::create_kernel_thread<IncomingMessageManager, 
							       &IncomingMessageManager::handler_thread_loop>(this,
													     tlp,
													     *core_rsrv);
  }

  void IncomingMessageManager::shutdown(void)
  {
#ifdef DEBUG_REALM
    shutdown_work_item();
#endif

    mutex.lock();
    if(!shutdown_flag) {
      shutdown_flag = true;
      condvar.broadcast();  // wake up any sleepers
    }
    mutex.unlock();

    for(std::vector<Realm::Thread *>::iterator it = handler_threads.begin();
	it != handler_threads.end();
	it++) {
      (*it)->join();
      delete (*it);
    }
    handler_threads.clear();
  }

  int IncomingMessageManager::get_messages(IncomingMessageManager::Message *& head,
					   IncomingMessageManager::Message **& tail,
					   bool wait)
  {
    AutoLock<> al(mutex);

    while(todo_oldest == todo_newest) {
      // todo list is empty
      if(shutdown_flag || !wait)
	return -1;

#ifdef DEBUG_INCOMING
      printf("incoming message list is empty - sleeping\n");
#endif
      sleeper_count += 1;
      condvar.wait();
      sleeper_count -= 1;
    }

    // pop the oldest entry off the todo list
    int sender = todo_list[todo_oldest];
    todo_oldest++;
    if(todo_oldest > nodes)
      todo_oldest = 0;
    head = heads[sender];
    tail = tails[sender];
    heads[sender] = 0;
    tails[sender] = 0;
    in_handler[sender] = true;
#ifdef DEBUG_INCOMING
    printf("handling incoming messages from %d\n", sender);
#endif
    // if there are other senders with messages waiting, we can request more
    //  background workers right away
    if((todo_oldest != todo_newest) && !bgwork_requested.load()) {
      bgwork_requested.store(true);
      make_active();
    }

    return sender;
  }

  void IncomingMessageManager::return_messages(int sender,
					       IncomingMessageManager::Message *head,
					       IncomingMessageManager::Message **tail)
  {
    // TODO: if we're in out-of-order mode, we don't need to take the lock
    //  if we're not actually putting anything back

    AutoLock<> al(mutex);
    in_handler[sender] = false;

    bool enqueue_needed = false;
    if(heads[sender] != 0) {
      // list was non-empty
      if(head != 0) {
	// prepend on list
	*tail = heads[sender];
	heads[sender] = head;
      }
      // in in-order mode, we hadn't enqueued this sender, so do that now
      enqueue_needed = true;
    } else {
      if(head != 0) {
	heads[sender] = head;
	tails[sender] = tail;
	enqueue_needed = true;
      }
    }

    if(enqueue_needed) {
      bool was_empty = todo_oldest == todo_newest;

      todo_list[todo_newest] = sender;
      todo_newest++;
      if(todo_newest > nodes)
	todo_newest = 0;
      assert(todo_newest != todo_oldest);  // should never wrap around
      if(sleeper_count > 0)
	condvar.broadcast();  // wake up any sleepers

      if(was_empty && !bgwork_requested.load()) {
	bgwork_requested.store(true);
	make_active();
      }
    }
  }

  void IncomingMessageManager::do_work(TimeLimit work_until)
  {
    // now that we've been called, our previous request for bgwork has been
    //  granted and we will need another one if/when more work comes
    // it's ok if this races with other threads that are adding/getting messages
    //  because we'll do the request ourselves below in that case
    bgwork_requested.store(false);

    Message *current_msg = 0;
    Message **current_tail = 0;
    int sender = get_messages(current_msg, current_tail, false /*!wait*/);

    // we're here because there was work to do, so an empty list is bad unless
    //  there are also dedicated threads that might have grabbed it
    if(sender == -1) {
      assert(dedicated_threads > 0);
      return;
    }

    ThreadLocal::in_message_handler = true;

    while(current_msg) {
      Message *next_msg = current_msg->next_msg;
#ifdef DETAILED_MESSAGE_TIMING
      int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
      CurrentTime start_time;
#endif
      long long t_start = 0;
      if(Config::profile_activemsg_handlers)
	t_start = Clock::current_time_in_nanoseconds();

      (current_msg->handler->handler)(current_msg->sender,
				      current_msg->hdr,
				      current_msg->payload,
				      current_msg->payload_size);

      long long t_end = 0;
      if(Config::profile_activemsg_handlers)
	t_end = Clock::current_time_in_nanoseconds();

      if(current_msg->callback_fnptr)
	(current_msg->callback_fnptr)(current_msg->sender,
				      current_msg->callback_data);

      if(Config::profile_activemsg_handlers)
	current_msg->handler->stats.record(t_start, t_end);
#ifdef DETAILED_MESSAGE_TIMING
      detailed_message_timing.record(timing_idx,
				     current_msg->get_peer(),
				     current_msg->get_msgid(),
				     -4, // 0xc - flagged as an incoming message,
				     current_msg->get_msgsize(),
				     count++, // how many messages we handle in a batch
				     start_time, CurrentTime());
#endif
      // TODO: recycle!
      delete current_msg;
      current_msg = next_msg;

      // do we need to stop early?
      if(current_msg && work_until.is_expired())
	break;
    }

    ThreadLocal::in_message_handler = false;

    // put back whatever we had left, if anything - this'll requeue us if needed
    return_messages(sender, current_msg, current_tail);
  }

  void IncomingMessageManager::handler_thread_loop(void)
  {
    // this thread is ALWAYS in a handler
    ThreadLocal::in_message_handler = true;

    while (true) {
      Message *current_msg = 0;
      Message **current_tail = 0;
      int sender = get_messages(current_msg, current_tail, true /*wait*/);
      if(sender == -1) {
#ifdef DEBUG_INCOMING
	printf("received empty list - assuming shutdown!\n");
#endif
	break;
      }
#ifdef DETAILED_MESSAGE_TIMING
      int count = 0;
#endif
      while(current_msg) {
	Message *next_msg = current_msg->next_msg;
#ifdef DETAILED_MESSAGE_TIMING
	int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
	CurrentTime start_time;
#endif
	long long t_start = 0;
	if(Config::profile_activemsg_handlers)
	  t_start = Clock::current_time_in_nanoseconds();

	(current_msg->handler->handler)(current_msg->sender,
					current_msg->hdr,
					current_msg->payload,
					current_msg->payload_size);

	long long t_end = 0;
	if(Config::profile_activemsg_handlers)
	  t_end = Clock::current_time_in_nanoseconds();

	if(current_msg->callback_fnptr)
	  (current_msg->callback_fnptr)(current_msg->sender,
					current_msg->callback_data);

	if(Config::profile_activemsg_handlers)
	  current_msg->handler->stats.record(t_start, t_end);
#ifdef DETAILED_MESSAGE_TIMING
	detailed_message_timing.record(timing_idx, 
				       current_msg->get_peer(),
				       current_msg->get_msgid(),
				       -4, // 0xc - flagged as an incoming message,
				       current_msg->get_msgsize(),
				       count++, // how many messages we handle in a batch
				       start_time, CurrentTime());
#endif
	delete current_msg;
	current_msg = next_msg;
      }
      // we always handle all the messages, but still indicate we're done
      return_messages(sender, 0, 0);
    }
  }


}; // namespace Realm
