/* Copyright 2022 Stanford University, NVIDIA Corporation
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

    // the maximum time we're willing to spend on inline message
    //  handlers
    long long max_inline_message_time = 5000 /* nanoseconds*/;
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class CompletionCallbackBase
  //

  CompletionCallbackBase::~CompletionCallbackBase()
  {}

  /*static*/ void CompletionCallbackBase::invoke_all(void *start, size_t bytes)
  {
    size_t ofs = 0;
    while(ofs < bytes) {
      CompletionCallbackBase *cc = static_cast<CompletionCallbackBase *>(start);
      cc->invoke();
      size_t step = cc->size();
      start = static_cast<char *>(start) + step;
      ofs += step;
    }
    assert(ofs == bytes);
  }

  /*static*/ void CompletionCallbackBase::clone_all(void *dst,
						    const void *src,
						    size_t bytes)
  {
    size_t ofs = 0;
    while(ofs < bytes) {
      const CompletionCallbackBase *cc = static_cast<const CompletionCallbackBase *>(src);
      cc->clone_at(dst);
      size_t step = cc->size();
      src = static_cast<const char *>(src) + step;
      dst = static_cast<char *>(dst) + step;
      ofs += step;
    }
    assert(ofs == bytes);
  }

  /*static*/ void CompletionCallbackBase::destroy_all(void *start, size_t bytes)
  {
    size_t ofs = 0;
    while(ofs < bytes) {
      CompletionCallbackBase *cc = static_cast<CompletionCallbackBase *>(start);
      size_t step = cc->size();
      cc->~CompletionCallbackBase();
      start = static_cast<char *>(start) + step;
      ofs += step;
    }
    assert(ofs == bytes);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct ActiveMessageHandlerStats
  //

  ActiveMessageHandlerStats::ActiveMessageHandlerStats(void)
    : count(0), sum(0), sum2(0), minval(~size_t(0)), maxval(0)
  {}

  void ActiveMessageHandlerStats::record(long long t_start, long long t_end)
  {
    long long delta = t_end - t_start;
    size_t val = (delta > 0) ? delta : 0;
    count.fetch_add(1);
    minval.fetch_min(val);
    maxval.fetch_max(val);
    sum.fetch_add(val);
    sum2.fetch_add(val * val); // TODO: smarter math to avoid overflow
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessageHandlerTable
  //

  ActiveMessageHandlerTable::ActiveMessageHandlerTable(void)
  {}

  ActiveMessageHandlerTable::~ActiveMessageHandlerTable(void)
  {}

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
	size_t count = stats.count.load();
	if(count == 0)
	  continue;

	size_t sum = stats.sum.load();
	size_t sum2 = stats.sum2.load();
	size_t minval = stats.minval.load();
	size_t maxval = stats.maxval.load();
	double avg = double(sum) / double(count);
	double stddev = sqrt((double(sum2) / double(count)) - (avg * avg));
	log_amhandler.print() << "handler " << i << ": " << handlers[i].name
			      << " count=" << count
			      << " avg=" << avg
			      << " dev=" << stddev
			      << " min=" << minval
			      << " max=" << maxval;
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
      e.handler = nextreg->get_handler();
      e.handler_notimeout = nextreg->get_handler_notimeout();
      // at least one of the two above must be non-null
      assert((e.handler != 0) || (e.handler_notimeout != 0));
      e.handler_inline = nextreg->get_handler_inline();
      handlers.push_back(e);
    }

    std::sort(handlers.begin(), handlers.end(), hash_less);

    // handler ids are the same everywhere, so only log on node 0
    if(Network::my_node_id == 0)
      for(size_t i = 0; i < handlers.size(); i++)
	log_amhandler.info() << "handler " << i
			     << ": " << handlers[i].name
			     << (handlers[i].handler ? " (timeout)" : "")
			     << (handlers[i].handler_inline ? " (inline)" : "");
  }

  /*static*/ ActiveMessageHandlerRegBase *ActiveMessageHandlerTable::pending_handlers = 0;

  /*extern*/ ActiveMessageHandlerTable activemsg_handler_table;


  ////////////////////////////////////////////////////////////////////////
  //
  // class IncomingMessageManager::MessageBlock
  //

  /*static*/ IncomingMessageManager::MessageBlock *IncomingMessageManager::MessageBlock::new_block(size_t _total_size)
  {
    void *ptr = malloc(_total_size);
    assert(ptr != 0);
    MessageBlock *block = new(ptr) MessageBlock;
    block->total_size = _total_size;
    block->next_free = 0;
    block->reset();
    log_amhandler.info() << "creating message block: " << block;
    return block;
  }

  /*static*/ void IncomingMessageManager::MessageBlock::free_block(MessageBlock *block)
  {
    while(block) {
      log_amhandler.info() << "freeing message block: " << block;
      MessageBlock *next = block->next_free;
      block->~MessageBlock();
      free(block);
      block = next;
    }
  }

  void IncomingMessageManager::MessageBlock::reset()
  {
    size_used = sizeof(MessageBlock);
    size_used = (size_used + 15) & ~size_t(15); // 16B alignment
    use_count.store(1);
  }

  IncomingMessageManager::Message *IncomingMessageManager::MessageBlock::append_message(size_t hdr_bytes_needed,
					size_t payload_bytes_needed)
  {
    size_t msg_ofs = size_used;
    size_t new_used = msg_ofs + sizeof(Message);
    new_used = (new_used + 15) & ~size_t(15); // 16B alignment

    size_t hdr_ofs;
    if(hdr_bytes_needed > 0) {
      hdr_ofs = new_used;
      new_used += hdr_bytes_needed;
      new_used = (new_used + 15) & ~size_t(15); // 16B alignment
    } else
      hdr_ofs = 0;

    size_t payload_ofs;
    if(payload_bytes_needed > 0) {
      payload_ofs = new_used;
      new_used += payload_bytes_needed;
      new_used = (new_used + 15) & ~size_t(15); // 16B alignment
    } else
      payload_ofs = 0;

    // does it fit?
    if(new_used <= total_size) {
      use_count.fetch_add(1);
      size_used = new_used;

      uintptr_t base = reinterpret_cast<uintptr_t>(this);
      Message *msg = reinterpret_cast<Message *>(base + msg_ofs);
      msg->block = this;
      msg->hdr = ((hdr_ofs > 0) ?
		    reinterpret_cast<void *>(base + hdr_ofs) :
		    0);
      msg->payload = ((payload_ofs > 0) ?
		        reinterpret_cast<void *>(base + payload_ofs) :
		        0);
      return msg;
    } else {
      // would it have ever fit?
      assert((new_used - size_used) <= (total_size - sizeof(MessageBlock)));

      // return failure - caller will find a new block
      return 0;
    }
  }

  void IncomingMessageManager::MessageBlock::recycle_message(IncomingMessageManager::Message *msg,
							     IncomingMessageManager *manager)
  {
    // first, free any hdr/payload pointer we were borrowing
    if(msg->hdr_needs_free)
      free(msg->hdr);
    if(msg->payload_needs_free)
      free(msg->payload);

    // now decrement our use_count
    unsigned prev_count = use_count.fetch_sub(1);

    // if it was 1 (now 0), take the manager's lock and add ourselves to
    //  the available list (or delete if there's already enough)
    if(prev_count == 1) {
      bool delete_me = false;
      {
	AutoLock<> al(manager->mutex);
	if(manager->num_available_blocks < manager->cfg_max_available_blocks) {
	  reset();
	  next_free = manager->available_blocks;
	  manager->available_blocks = this;
	  manager->num_available_blocks++;
	} else
	  delete_me = true;
      }

      if(delete_me)
	free_block(this);
    }
  }

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
    , shutdown_flag(0)
    , handlers_active(0)
    , drain_pending(false)
    , drain_min_count(0)
    , total_messages_handled(0)
    , condvar(mutex)
    , drain_condvar(mutex)
    , available_blocks(0)
    , num_available_blocks(0)
    , cfg_max_available_blocks(10)
    , cfg_message_block_size(1048576 - 32) // 1MB - space for heap metadata
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

    current_block = MessageBlock::new_block(cfg_message_block_size);
  }

  IncomingMessageManager::~IncomingMessageManager(void)
  {
    delete core_rsrv;
    delete[] heads;
    delete[] tails;
    delete[] in_handler;
    delete[] todo_list;

    MessageBlock::free_block(current_block);
    if(available_blocks)
      MessageBlock::free_block(available_blocks);
  }

  bool IncomingMessageManager::add_incoming_message(NodeID sender,
						    ActiveMessageHandlerTable::MessageID msgid,
						    const void *hdr, size_t hdr_size,
						    int hdr_mode,
						    const void *payload, size_t payload_size,
						    int payload_mode,
						    CallbackFnptr callback_fnptr,
						    CallbackData callback_data1,
						    CallbackData callback_data2,
						    TimeLimit work_until)
  {
#ifdef DEBUG_INCOMING
    printf("adding incoming message from %d\n", sender);
#endif

    // look up which message this is
    ActiveMessageHandlerTable::HandlerEntry *handler = activemsg_handler_table.lookup_message_handler(msgid);

    // if we have an inline handler and enough time to run it, give it
    //  a go
    if((handler->handler_inline != 0) &&
       (Config::max_inline_message_time > 0) &&
       !work_until.will_expire(Config::max_inline_message_time)) {
      long long t_start = 0;
      if(Config::profile_activemsg_handlers)
	t_start = Clock::current_time_in_nanoseconds();

      if((handler->handler_inline)(sender, hdr, payload, payload_size,
				   TimeLimit::relative(Config::max_inline_message_time))) {
	if(Config::profile_activemsg_handlers) {
	  long long t_end = Clock::current_time_in_nanoseconds();
	  handler->stats.record(t_start, t_end);
	}
	if(payload_mode == PAYLOAD_FREE)
	  free(const_cast<void *>(payload));
        // see if we need to wake up a thread waiting on a drain
        {
          AutoLock<> al(mutex);
          total_messages_handled += 1;
          if(drain_pending &&
             (todo_oldest == todo_newest) && (handlers_active == 0) &&
             (total_messages_handled >= drain_min_count)) {
            drain_pending = false;
            drain_condvar.broadcast();
          }
        }
	return true;
      }
    }

    // can't handle inline - need to create a Message object for it

    mutex.lock();

    Message *msg = 0;
    size_t hdr_bytes_needed = ((hdr_mode == PAYLOAD_COPY) ?
			         hdr_size : 0);
    size_t payload_bytes_needed = ((payload_mode == PAYLOAD_COPY) ?
				     payload_size : 0);
    while(true) {
      // try to stick this message in the current block
      msg = current_block->append_message(hdr_bytes_needed,
					  payload_bytes_needed);
      if(msg != 0) break;

      // do we have a new block we can use?
      if((available_blocks != 0) ||
	 (current_block->use_count.load() == 1)) {
	// first release our hold on the current block
	unsigned prev_count = current_block->use_count.fetch_sub(1);

	if(prev_count == 1) {
	  // in the (highly unlikely) case that all of its messages have
	  //  been deleted, we can just reset it and reuse it
	  current_block->reset();
	  log_amhandler.debug() << "reusing message block: " << current_block;
	} else {
	  assert(available_blocks != 0);
	  current_block = available_blocks;
	  available_blocks = available_blocks->next_free;
	  current_block->next_free = 0;
	  num_available_blocks--;
	  log_amhandler.debug() << "switching to message block: " << current_block;
	}

	// either way, this must now succeed
	msg = current_block->append_message(hdr_bytes_needed,
					    payload_bytes_needed);
	assert(msg != 0);
	break;
      }

      // no available blocks - drop the mutex while we allocate a new one
      mutex.unlock();
      MessageBlock *block = MessageBlock::new_block(cfg_message_block_size);
      mutex.lock();
      // we don't know what changed while we weren't holding the lock,
      //  so just stick this on the available list and restart
      block->next_free = available_blocks;
      available_blocks = block;
      num_available_blocks++;
    }

    // fill in message structure
    // TODO: let go of lock if copying a large payload?
    {
      msg->next_msg = 0;
      msg->sender = sender;
      msg->handler = handler;
      msg->callback_fnptr = callback_fnptr;
      msg->callback_data1 = callback_data1;
      msg->callback_data2 = callback_data2;

      if(hdr_mode == PAYLOAD_COPY)
	memcpy(msg->hdr, hdr, hdr_size);
      else
	msg->hdr = const_cast<void *>(hdr);
      msg->hdr_size = hdr_size;
      msg->hdr_needs_free = (hdr_mode == PAYLOAD_FREE);

      if(payload_size > 0) {
	if(payload_mode == PAYLOAD_COPY)
	  memcpy(msg->payload, payload, payload_size);
	else
	  msg->payload = const_cast<void *>(payload);
      }
      msg->payload_size = payload_size;
      msg->payload_needs_free = (payload_mode == PAYLOAD_FREE);
    }

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

  // stalls caller until all incoming messages have been handled
  void IncomingMessageManager::drain_incoming_messages(size_t min_messages_handled)
  {
    AutoLock<> al(mutex);

    while((todo_oldest != todo_newest) || (handlers_active > 0) ||
          (total_messages_handled < min_messages_handled)) {
      drain_min_count = min_messages_handled;
      drain_pending = true;
      drain_condvar.wait();
    }
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
    handlers_active++;
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

  bool IncomingMessageManager::return_messages(int sender,
                                               size_t num_handled,
					       IncomingMessageManager::Message *head,
					       IncomingMessageManager::Message **tail)
  {
    AutoLock<> al(mutex);
    total_messages_handled += num_handled;
    in_handler[sender] = false;
    handlers_active--;

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

    bool now_active = false;
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
	now_active = true;
      }
    }

    // was somebody waiting for the queue to go (perhaps temporarily) empty?
    if(drain_pending &&
       (todo_oldest == todo_newest) && (handlers_active == 0) &&
       (total_messages_handled >= drain_min_count)) {
      drain_pending = false;
      drain_condvar.broadcast();
    }

    return now_active;
  }

  bool IncomingMessageManager::do_work(TimeLimit work_until)
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
      return false;
    }

    ThreadLocal::in_message_handler = true;

    Message *skipped_messages = 0;
    Message **skipped_tail = &skipped_messages;
    size_t num_handled = 0;

    while(current_msg) {
      Message *next_msg = current_msg->next_msg;
#ifdef DETAILED_MESSAGE_TIMING
      int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
      CurrentTime start_time;
#endif
      long long t_start = 0;
      bool do_profile = Config::profile_activemsg_handlers;

      // do we have a handler that understands time limits?
      if(current_msg->handler->handler != 0) {
	if(do_profile)
	  t_start = Clock::current_time_in_nanoseconds();

	(current_msg->handler->handler)(current_msg->sender,
					current_msg->hdr,
					current_msg->payload,
					current_msg->payload_size,
					work_until);
      } else {
	// estimate how long this handler will take, clamping at a
	//  semi-arbitrary 20us
	long long t_estimate = 20000;
	{
	  size_t num = current_msg->handler->stats.sum.load();
	  size_t den = current_msg->handler->stats.count.load();
	  if(num < (den * t_estimate))
	    t_estimate = num / den;
	}
	if(work_until.will_expire(t_estimate)) {
	  // skip this message instead of handling it now
	  *skipped_tail = current_msg;
	  skipped_tail = &current_msg->next_msg;
	  current_msg = current_msg->next_msg;
	  // skipping things can take time too, so check if we're
	  //  completely out of time
	  if(work_until.is_expired()) break;
	  continue;
	}

	// always profile notimeout handlers
	do_profile = true;
	t_start = Clock::current_time_in_nanoseconds();

	(current_msg->handler->handler_notimeout)(current_msg->sender,
						  current_msg->hdr,
						  current_msg->payload,
						  current_msg->payload_size);
      }

      long long t_end = 0;
      if(do_profile)
	t_end = Clock::current_time_in_nanoseconds();

      if(current_msg->callback_fnptr)
	(current_msg->callback_fnptr)(current_msg->sender,
				      current_msg->callback_data1,
				      current_msg->callback_data2);

      if(do_profile)
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
      // recycle message
      current_msg->block->recycle_message(current_msg, this);

      current_msg = next_msg;
      num_handled += 1;

      // do we need to stop early?
      if(current_msg && work_until.is_expired())
	break;
    }

    ThreadLocal::in_message_handler = false;

    // anything we didn't get to goes on the end of the skipped list
    if(current_msg) {
      *skipped_tail = current_msg;
      skipped_tail = current_tail;
    } else
      *skipped_tail = 0;

    // put back whatever we had left, if anything - request requeue if needed
    return return_messages(sender, num_handled, skipped_messages, skipped_tail);
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
      size_t num_handled = 0;
      while(current_msg) {
	Message *next_msg = current_msg->next_msg;
#ifdef DETAILED_MESSAGE_TIMING
	int timing_idx = detailed_message_timing.get_next_index(); // grab this while we still hold the lock
	CurrentTime start_time;
#endif
	long long t_start = 0;
	if(Config::profile_activemsg_handlers)
	  t_start = Clock::current_time_in_nanoseconds();

	if(current_msg->handler->handler != 0)
	  (current_msg->handler->handler)(current_msg->sender,
					  current_msg->hdr,
					  current_msg->payload,
					  current_msg->payload_size,
					  TimeLimit());
	else
	  (current_msg->handler->handler_notimeout)(current_msg->sender,
						    current_msg->hdr,
						    current_msg->payload,
						    current_msg->payload_size);

	long long t_end = 0;
	if(Config::profile_activemsg_handlers)
	  t_end = Clock::current_time_in_nanoseconds();

	if(current_msg->callback_fnptr)
	  (current_msg->callback_fnptr)(current_msg->sender,
					current_msg->callback_data1,
					current_msg->callback_data2);

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
        // recycle message
        current_msg->block->recycle_message(current_msg, this);

	current_msg = next_msg;
        num_handled += 1;
      }
      // we always handle all the messages, but still indicate we're done
      return_messages(sender, num_handled, 0, 0);
    }
  }


}; // namespace Realm
