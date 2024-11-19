/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "realm/comp_queue_impl.h"

#include "realm/runtime_impl.h"
#include "realm/logging.h"

namespace Realm {

  Logger log_compqueue("compqueue");

  // active messages

  struct CompQueueDestroyMessage {
    CompletionQueue comp_queue;
    Event wait_on;

    static void handle_message(NodeID sender, const CompQueueDestroyMessage &msg,
                               const void *data, size_t datalen);
  };

  struct CompQueueAddEventMessage {
    CompletionQueue comp_queue;
    Event event;
    bool faultaware;

    static void handle_message(NodeID sender, const CompQueueAddEventMessage &msg,
                               const void *data, size_t datalen);
  };

  struct CompQueueRemoteProgressMessage {
    CompletionQueue comp_queue;
    Event progress;

    static void handle_message(NodeID sender, const CompQueueRemoteProgressMessage &msg,
                               const void *data, size_t datalen);
  };

  struct CompQueuePopRequestMessage {
    CompletionQueue comp_queue;
    size_t max_to_pop;
    bool discard_events;
    intptr_t request;

    static void handle_message(NodeID sender, const CompQueuePopRequestMessage &msg,
                               const void *data, size_t datalen);
  };

  struct CompQueuePopResponseMessage {
    size_t count;
    intptr_t request;

    static void handle_message(NodeID sender, const CompQueuePopResponseMessage &msg,
                               const void *data, size_t datalen);
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompletionQueue
  //

  /*static*/ const CompletionQueue CompletionQueue::NO_QUEUE = {
      /* zero-initialization */};

  /*static*/ CompletionQueue CompletionQueue::create_completion_queue(size_t max_size)
  {
    CompQueueImpl *cq = get_runtime()->local_compqueue_free_list->alloc_entry();
    // sanity-check that we haven't exhausted the space of cq IDs
    if(get_runtime()->get_compqueue_impl(cq->me) != cq) {
      log_compqueue.fatal() << "completion queue ID space exhausted!";
      abort();
    }
    if(max_size > 0) {
      cq->set_capacity(max_size, false /*!resizable*/);
    } else {
      cq->set_capacity(1024 /*no obvious way to pick this*/, true /*resizable*/);
    }

    log_compqueue.info() << "created completion queue: cq=" << cq->me
                         << " size=" << max_size;
    return cq->me;
  }

  // destroy a completion queue
  void CompletionQueue::destroy(Event wait_on /*= Event::NO_EVENT*/)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    log_compqueue.info() << "destroying completion queue: cq=" << *this
                         << " wait_on=" << wait_on;

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      if(wait_on.has_triggered()) {
        cq->destroy();
        get_runtime()->local_compqueue_free_list->free_entry(cq);
      } else {
        cq->deferred_destroy.defer(cq, wait_on);
      }
    } else {
      ActiveMessage<CompQueueDestroyMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->wait_on = wait_on;
      amsg.commit();
    }
  }

  /*static*/ void CompQueueDestroyMessage::handle_message(
      NodeID sender, const CompQueueDestroyMessage &msg, const void *data, size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    if(msg.wait_on.has_triggered()) {
      cq->destroy();
      get_runtime()->local_compqueue_free_list->free_entry(cq);
    } else {
      cq->deferred_destroy.defer(cq, msg.wait_on);
    }
  }

  // adds an event to the completion queue (once it triggers)
  // non-faultaware version raises a fatal error if the specified 'event'
  //  is poisoned
  void CompletionQueue::add_event(Event event)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    log_compqueue.info() << "event registered with completion queue: cq=" << *this
                         << " event=" << event;

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      cq->add_event(event, false /*!faultaware*/);
    } else {
      ActiveMessage<CompQueueAddEventMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->event = event;
      amsg->faultaware = false;
      amsg.commit();
    }
  }

  void CompletionQueue::add_event_faultaware(Event event)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    log_compqueue.info() << "event registered with completion queue: cq=" << *this
                         << " event=" << event << " (faultaware)";

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      cq->add_event(event, true /*faultaware*/);
    } else {
      ActiveMessage<CompQueueAddEventMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->event = event;
      amsg->faultaware = true;
      amsg.commit();
    }
  }

  /*static*/ void
  CompQueueAddEventMessage::handle_message(NodeID sender,
                                           const CompQueueAddEventMessage &msg,
                                           const void *data, size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    cq->add_event(msg.event, msg.faultaware);
  }

  // requests up to 'max_events' triggered events to be popped from the
  //  queue and stored in the provided 'events' array (if null, the
  //  identities of the triggered events are discarded)
  // this call returns the actual number of events popped, which may be
  //  zero (this call is nonblocking)
  // when 'add_event_faultaware' is used, any poisoning of the returned
  //  events is not signalled explicitly - the caller is expected to
  //  check via 'has_triggered_faultaware' itself
  size_t CompletionQueue::pop_events(Event *events, size_t max_events)
  {
    NodeID owner = ID(*this).compqueue_owner_node();
    size_t count;

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      count = cq->pop_events(events, max_events);
    } else {
      // bounce data off a temp array since we don't know if 'events' is
      //  accessible to an active message handler thread
      Event *ev_copy = 0;
      if(events) {
        ev_copy = reinterpret_cast<Event *>(malloc(max_events * sizeof(Event)));
        assert(ev_copy != 0);
      }
      CompQueueImpl::RemotePopRequest *req =
          new CompQueueImpl::RemotePopRequest(ev_copy, max_events);

      ActiveMessage<CompQueuePopRequestMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->max_to_pop = max_events;
      amsg->discard_events = (events == 0);
      amsg->request = reinterpret_cast<intptr_t>(req);
      amsg.commit();

      // now wait for a response - no real alternative to blocking here?
      {
        AutoLock<> al(req->mutex);
        while(!req->completed)
          req->condvar.wait();
      }

      count = req->count;

      delete req;

      if(ev_copy) {
        if(count > 0)
          memcpy(events, ev_copy, count * sizeof(Event));
        free(ev_copy);
      }
    }

    if(events != 0)
      log_compqueue.info() << "events popped: cq=" << *this << " max=" << max_events
                           << " act=" << count
                           << " events=" << PrettyVector<Event>(events, count);
    else
      log_compqueue.info() << "events popped: cq=" << *this << " max=" << max_events
                           << " act=" << count << " events=(ignored)";

    return count;
  }

  /*static*/ void
  CompQueuePopRequestMessage::handle_message(NodeID sender,
                                             const CompQueuePopRequestMessage &msg,
                                             const void *data, size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    Event *events = 0;
    size_t max_to_pop = msg.max_to_pop;
    if(!msg.discard_events) {
      // we're going to use temp space on the stack, so limit the number of
      //  events to something sane
      if(max_to_pop > 1024)
        max_to_pop = 1024;
      events = reinterpret_cast<Event *>(alloca(max_to_pop * sizeof(Event)));
      assert(events != 0);
    }

    size_t count = cq->pop_events(events, max_to_pop);

    size_t bytes = (msg.discard_events ? 0 : count * sizeof(Event));
    ActiveMessage<CompQueuePopResponseMessage> amsg(sender, bytes);
    amsg->count = count;
    amsg->request = msg.request;
    if(bytes > 0)
      amsg.add_payload(events, bytes, PAYLOAD_COPY);
    amsg.commit();
  }

  /*static*/ void
  CompQueuePopResponseMessage::handle_message(NodeID sender,
                                              const CompQueuePopResponseMessage &msg,
                                              const void *data, size_t datalen)
  {
    CompQueueImpl::RemotePopRequest *req =
        reinterpret_cast<CompQueueImpl::RemotePopRequest *>(msg.request);

    {
      AutoLock<> al(req->mutex);
      assert(msg.count <= req->capacity);
      if(req->events) {
        // data expected
        assert(datalen == (msg.count * sizeof(Event)));
        if(msg.count > 0)
          memcpy(req->events, data, datalen);
      } else {
        // no data expected
        assert(datalen == 0);
      }
      req->count = msg.count;
      req->completed = true;
      req->condvar.broadcast();
    }
  }

  // get an event that, once triggered, guarantees that (at least) one
  //  call to pop_events made since the non-empty event was requested
  //  will return a non-zero number of triggered events
  // once a call to pop_events has been made (by the caller of
  //  get_nonempty_event or anybody else), the guarantee is lost and
  //  a new non-empty event must be requested
  // note that 'get_nonempty_event().has_triggered()' is unlikely to
  //  ever return 'true' if called from a node other than the one that
  //  created the completion queue (i.e. the query at least has the
  //  round-trip network communication latency to deal with) - if polling
  //  on the completion queue is unavoidable, the loop should poll on
  //  pop_events directly
  Event CompletionQueue::get_nonempty_event(void)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      Event e = cq->get_local_progress_event();
      log_compqueue.info() << "local nonempty event: cq=" << *this << " event=" << e;
      return e;
    } else {
      // we can't reuse progress events safely (because we can't see when pop
      //  calls occur), so make a fresh event each time
      GenEventImpl *ev_impl = GenEventImpl::create_genevent();
      Event e = ev_impl->current_event();

      ActiveMessage<CompQueueRemoteProgressMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->progress = e;
      amsg.commit();

      return e;
    }
  }

  /*static*/ void CompQueueRemoteProgressMessage::handle_message(
      NodeID sender, const CompQueueRemoteProgressMessage &msg, const void *data,
      size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    cq->add_remote_progress_event(msg.progress);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl
  //

  CompQueueImpl::~CompQueueImpl(void)
  {
    AutoLock<> al(mutex);
    // assert(pending_events.load() == 0);
    while(batches) {
      CompQueueWaiterBatch *next_batch = batches->next_batch;
      delete batches;
      batches = next_batch;
    }
    if(completed_events) {
      free(completed_events);
    }
  }

  void CompQueueImpl::init(CompletionQueue _me, int _owner)
  {
    this->me = _me;
    this->owner = _owner;
  }

  void CompQueueImpl::set_capacity(size_t _max_size, bool _resizable)
  {
    AutoLock<> al(mutex);
    if(resizable) {
      assert(cur_events == 0);
    } else {
      assert(wr_ptr.load() == consume_ptr.load());
    }
    wr_ptr.store(0);
    rd_ptr.store(0);
    pending_events.store(0);
    commit_ptr.store(0);
    consume_ptr.store(0);
    cur_events = 0;
    resizable = _resizable;
    // round up to a power of 2 for easy modulo arithmetic
    max_events = 1;

    while(max_events < _max_size) {
      max_events <<= 1;
    }

    void *ptr = malloc(sizeof(Event) * max_events);
    assert(ptr != 0);
    completed_events = reinterpret_cast<Event *>(ptr);
  }

  size_t CompQueueImpl::get_capacity() const { return max_events; }

  size_t CompQueueImpl::get_pending_events() const { return pending_events.load(); }

  void CompQueueImpl::destroy(void)
  {
    AutoLock<> al(mutex);
    // ok to have completed events leftover, but no pending events
    assert(pending_events.load() == 0);
    max_events = 0;

    if(completed_events) {
      free(completed_events);
      completed_events = 0;
    }
  }

  void CompQueueImpl::add_event(Event event, EventImpl *ev_impl, bool faultaware)
  {
    EventImpl::gen_t needed_gen = ID(event).event_generation();

    bool poisoned = false;
    if(ev_impl->has_triggered(needed_gen, poisoned)) {
      if(poisoned && !faultaware) {
        log_compqueue.fatal() << "cannot enqueue poisoned event: cq=" << me
                              << " event=" << event;
        abort();
      } else
        add_completed_event(event, 0 /*no waiter*/, TimeLimit::responsive());
    } else {
      // we need a free waiter - make some if needed
      CompQueueWaiter *waiter = 0;
      {
        AutoLock<> al(mutex);
        pending_events.fetch_add(1);

        // try to pop a waiter from the free list - this needs to use
        //  CAS to accomodate unsynchronized pushes to the list, but the
        //  mutex we hold prevents any other poppers (so no ABA problem)
        waiter = first_free_waiter.load_acquire();
        while(waiter) {
          if(first_free_waiter.compare_exchange(waiter, waiter->next_free))
            break;
        }

        if(waiter) {
          waiter->next_free = 0;
        } else {
          // create a batch of waiters, claim the first one and enqueue the others
          //  in the free list
          batches = new CompQueueWaiterBatch(this, batches);
          waiter = &batches->waiters[0];
          for(size_t i = 1; i < CQWAITER_BATCH_SIZE - 1; i++)
            batches->waiters[i].next_free = &batches->waiters[i + 1];
          // CAS for insertion into the free list
          CompQueueWaiter *old_head = first_free_waiter.load();
          while(true) {
            batches->waiters[CQWAITER_BATCH_SIZE - 1].next_free = old_head;
            if(first_free_waiter.compare_exchange(old_head, &batches->waiters[1]))
              break;
          }
        }
      }
      // with the lock released, add the waiter
      waiter->wait_on = event;
      waiter->faultaware = faultaware;
      ev_impl->add_waiter(needed_gen, waiter);
    }
  }

  void CompQueueImpl::add_event(Event event, bool faultaware)
  {
    // special case: NO_EVENT has no impl...
    if(!event.exists()) {
      add_completed_event(event, 0 /*no waiter*/, TimeLimit::responsive());
      return;
    }

    add_event(event, get_runtime()->get_event_impl(event), faultaware);
  }

  Event CompQueueImpl::get_local_progress_event(void)
  {
    // non-resizable queues can observe a non-empty queue and return NO_EVENT
    //  without taking the lock
    if(!resizable && (rd_ptr.load() < commit_ptr.load())) {
      return Event::NO_EVENT;
    }

    {
      AutoLock<> al(mutex);

      // now that we hold the lock, check emptiness consistent with progress
      //  event information
      if(resizable) {
        if(cur_events > 0) {
          assert(local_progress_event == 0);
          return Event::NO_EVENT;
        }
      } else {
        // before we recheck in the non-resizable case, set the
        //  'has_progress_events' flag - this ensures that any pusher we don't
        //  see when we recheck the commit pointer will see the flag and take the
        //  log to handle the progress event we're about to make
        has_progress_events.store(true);
        // commit load has to be fenced to stay after the store above
        if(rd_ptr.load() < commit_ptr.load_fenced()) {
          return Event::NO_EVENT;
        }
      }

      // we appear to be empty - get or create the progress event
      if(local_progress_event) {
        ID id(local_progress_event->me);
        id.event_generation() = local_progress_event_gen;
        return id.convert<Event>();
      } else {
        // make a new one
        local_progress_event = GenEventImpl::create_genevent();
        Event e = local_progress_event->current_event();
        // TODO: we probably don't really need this field because it's always local
        local_progress_event_gen = ID(e).event_generation();
        return e;
      }
    }
  }

  void CompQueueImpl::add_remote_progress_event(Event event)
  {
    // if queue is non-empty, we'll just immediately trigger the event
    bool immediate_trigger = false;

    // non-resizable queues can check without even taking the lock
    if(!resizable && (rd_ptr.load() < commit_ptr.load())) {
      immediate_trigger = true;
    } else {
      AutoLock<> al(mutex);

      // now that we hold the lock, check emptiness consistent with progress
      //  event information
      if(resizable) {
        if(cur_events > 0) {
          immediate_trigger = true;
        }
      } else {
        // before we recheck in the non-resizable case, set the
        //  'has_progress_events' flag - this ensures that any pusher we don't
        //  see when we recheck the commit pointer will see the flag and take the
        //  log to handle the remote progress event we're about to add
        has_progress_events.store(true);
        // commit load has to be fenced to stay after the store above
        if(rd_ptr.load() < commit_ptr.load_fenced()) {
          immediate_trigger = true;
        }
      }

      if(!immediate_trigger) {
        remote_progress_events.push_back(event);
      }
    }

    // lock is released, so we can trigger now if needed
    if(immediate_trigger) {
      // the event is remote, but we know it's a GenEventImpl, so trigger here
      GenEventImpl::trigger(event, false /*!poisoned*/);
    }
  }

  size_t CompQueueImpl::pop_events(Event *events, size_t max_to_pop)
  {
    if(resizable) {
      AutoLock<> al(mutex);
      if((cur_events > 0) && (max_to_pop > 0)) {
        size_t count = std::min(cur_events, max_to_pop);

        // get current offset and advance pointer
        size_t rd_ofs = rd_ptr.fetch_add(count) & (max_events - 1);
        if(events) {
          // does copy wrap around?
          if((rd_ofs + count) > max_events) {
            size_t before_wrap = max_events - rd_ofs;
            // yes, two memcpy's needed
            memcpy(events, completed_events + rd_ofs, before_wrap * sizeof(Event));
            memcpy(events + before_wrap, completed_events,
                   (count - before_wrap) * sizeof(Event));
          } else {
            // no, single memcpy does the job
            memcpy(events, completed_events + rd_ofs, count * sizeof(Event));
          }
        }

        cur_events -= count;

        return count;
      } else
        return 0;
    } else {
      // lock-free version for nonresizable queues
      size_t count, old_rd_ptr;

      // use CAS to move rd_ptr up to (but not past) commit_ptr
      {
        old_rd_ptr = rd_ptr.load();
        while(true) {
          size_t old_commit = commit_ptr.load_acquire();
          if(old_rd_ptr == old_commit) {
            // queue is empty
            return 0;
          }
          count = std::min((old_commit - old_rd_ptr), max_to_pop);
          size_t new_rd_ptr = old_rd_ptr + count;
          if(rd_ptr.compare_exchange(old_rd_ptr, new_rd_ptr))
            break; // success
        }
      }

      // get current offset and advance pointer
      size_t rd_ofs = old_rd_ptr & (max_events - 1);

      if(events) {
        // does copy wrap around?
        if((rd_ofs + count) > max_events) {
          size_t before_wrap = max_events - rd_ofs;
          // yes, two memcpy's needed
          memcpy(events, completed_events + rd_ofs, before_wrap * sizeof(Event));
          memcpy(events + before_wrap, completed_events,
                 (count - before_wrap) * sizeof(Event));
        } else {
          // no, single memcpy does the job
          memcpy(events, completed_events + rd_ofs, count * sizeof(Event));
        }
      }

      // once we've copied out our events, mark that we've consumed the
      //  entries - this has to happen in the same order as the rd_ptr
      //  bumps though
      while(consume_ptr.load() != old_rd_ptr) {
        REALM_SPIN_YIELD();
      }
      size_t check = consume_ptr.fetch_add_acqrel(count);
      assert(check == old_rd_ptr);

      return count;
    }
  }

  void CompQueueImpl::add_completed_event(Event event, CompQueueWaiter *waiter,
                                          TimeLimit work_until)
  {
    log_compqueue.info() << "event pushed: cq=" << me << " event=" << event;
    GenEventImpl *local_trigger = 0;
    EventImpl::gen_t local_trigger_gen;
    std::vector<Event> remote_triggers;

    if(resizable) {
      AutoLock<> al(mutex);
      // check for overflow
      if(cur_events >= max_events) {
        // should detect it precisely
        assert(cur_events == max_events);
        size_t new_size = max_events * 2;
        Event *new_events = reinterpret_cast<Event *>(malloc(new_size * sizeof(Event)));
        assert(new_events != 0);
        size_t rd_ofs = rd_ptr.load() & (max_events - 1);
        if(rd_ofs > 0) {
          // most cases wrap around
          memcpy(new_events, completed_events + rd_ofs,
                 (cur_events - rd_ofs) * sizeof(Event));
          memcpy(new_events + (cur_events - rd_ofs), completed_events,
                 rd_ofs * sizeof(Event));
        } else
          memcpy(new_events, completed_events, cur_events * sizeof(Event));
        free(completed_events);
        completed_events = new_events;
        rd_ptr.store(0);
        wr_ptr.store(cur_events);
        max_events = new_size;
      }

      cur_events++;
      if(waiter != 0) {
        pending_events.fetch_sub(1);
        // add waiter to free list for reuse
        waiter->next_free = first_free_waiter.load();
        // cannot fail since we hold lock
        bool ok = first_free_waiter.compare_exchange(waiter->next_free, waiter);
        assert(ok);
      }

      size_t wr_ofs = wr_ptr.fetch_add(1) & (max_events - 1);
      completed_events[wr_ofs] = event;

      // grab things-to-trigger so we can do it outside the lock
      local_trigger = local_progress_event;
      local_trigger_gen = local_progress_event_gen;
      local_progress_event = 0;
      remote_triggers.swap(remote_progress_events);
    } else {
      // lock-free version for non-resizable queues

      // bump the write pointer and check for overflow
      size_t old_consume = consume_ptr.load();
      size_t old_wr_ptr = wr_ptr.fetch_add(1);
      if((old_wr_ptr - old_consume) >= max_events) {
        log_compqueue.fatal() << "completion queue overflow: cq=" << me
                              << " size=" << max_events;
        abort();
      }

      size_t wr_ofs = old_wr_ptr & (max_events - 1);
      completed_events[wr_ofs] = event;

      // bump commit pointer, but respecting order
      while(commit_ptr.load() != old_wr_ptr) {
        REALM_SPIN_YIELD();
      }
      size_t check = commit_ptr.fetch_add_acqrel(1);
      assert(check == old_wr_ptr);

      // lock-free insertion of waiter into free list
      if(waiter) {
        pending_events.fetch_sub(1);
        CompQueueWaiter *old_head = first_free_waiter.load();
        while(true) {
          waiter->next_free = old_head;
          if(first_free_waiter.compare_exchange(old_head, waiter))
            break;
        }
      }

      // see if we need to do any triggering - fenced load to keep it after the
      //  update of commit_ptr above
      if(has_progress_events.load_fenced()) {
        // take lock for this
        AutoLock<> al(mutex);
        local_trigger = local_progress_event;
        local_trigger_gen = local_progress_event_gen;
        local_progress_event = 0;
        remote_triggers.swap(remote_progress_events);
        has_progress_events.store(false);
      }
    }

    if(local_trigger) {
      log_compqueue.debug() << "triggering local progress event: cq=" << me
                            << " event=" << local_trigger->current_event();
      local_trigger->trigger(local_trigger_gen, Network::my_node_id, false /*!poisoned*/,
                             work_until);
    }

    if(!remote_triggers.empty())
      for(std::vector<Event>::const_iterator it = remote_triggers.begin();
          it != remote_triggers.end(); ++it) {
        log_compqueue.debug() << "triggering remote progress event: cq=" << me
                              << " event=" << (*it);
        GenEventImpl::trigger(*it, false /*!poisoned*/, work_until);
      }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::DeferredDestroy
  //

  void CompQueueImpl::DeferredDestroy::defer(CompQueueImpl *_cq, Event wait_on)
  {
    cq = _cq;
    EventImpl::add_waiter(wait_on, this);
  }

  void CompQueueImpl::DeferredDestroy::event_triggered(bool poisoned,
                                                       TimeLimit work_until)
  {
    assert(!poisoned);
    cq->destroy();
    get_runtime()->local_compqueue_free_list->free_entry(cq);
  }

  void CompQueueImpl::DeferredDestroy::print(std::ostream &os) const
  {
    os << "deferred completion queue destruction: cq=" << cq->me;
  }

  Event CompQueueImpl::DeferredDestroy::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::RemotePopRequest
  //

  CompQueueImpl::RemotePopRequest::RemotePopRequest(Event *_events, size_t _capacity)
    : condvar(mutex)
    , completed(false)
    , count(0)
    , capacity(_capacity)
    , events(_events)
  {}

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::CompQueueWaiter
  //

  void CompQueueImpl::CompQueueWaiter::event_triggered(bool poisoned,
                                                       TimeLimit work_until)
  {
    if(poisoned && !faultaware) {
      log_compqueue.fatal() << "cannot enqueue poisoned event: cq=" << cq->me
                            << " event=" << wait_on;
      abort();
    } else
      cq->add_completed_event(wait_on, this, work_until);
  }

  void CompQueueImpl::CompQueueWaiter::print(std::ostream &os) const
  {
    os << "completion queue insertion: cq=" << cq->me << " event=" << wait_on;
  }

  Event CompQueueImpl::CompQueueWaiter::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::CompQueueWaiterBatch
  //

  CompQueueImpl::CompQueueWaiterBatch::CompQueueWaiterBatch(CompQueueImpl *cq,
                                                            CompQueueWaiterBatch *_next)
    : next_batch(_next)
  {
    for(size_t i = 0; i < CQWAITER_BATCH_SIZE; i++)
      waiters[i].cq = cq;
  }

  CompQueueImpl::CompQueueWaiterBatch::~CompQueueWaiterBatch(void)
  {
    // avoid recursive delete - destroyer should walk `next_batch` chain
    // delete next_batch;
  }

  ActiveMessageHandlerReg<CompQueueDestroyMessage> compqueue_destroy_message_handler;
  ActiveMessageHandlerReg<CompQueueAddEventMessage> compqueue_addevent_message_handler;
  ActiveMessageHandlerReg<CompQueueRemoteProgressMessage>
      compqueue_remoteprogress_message_handler;
  ActiveMessageHandlerReg<CompQueuePopRequestMessage>
      compqueue_poprequest_message_handler;
  ActiveMessageHandlerReg<CompQueuePopResponseMessage>
      compqueue_popresponse_message_handler;

}; // namespace Realm
