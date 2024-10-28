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

#ifndef REALM_COMP_QUEUE_IMPL_H
#define REALM_COMP_QUEUE_IMPL_H

#include "realm/event.h"
#include "realm/id.h"
#include "realm/nodeset.h"
#include "realm/faults.h"

#include "realm/network.h"
#include <realm/activemsg.h>

#include "realm/lists.h"
#include "realm/threads.h"
#include "realm/logging.h"
#include "realm/redop.h"
#include "realm/bgwork.h"
#include "realm/dynamic_table.h"

#include "realm/event_impl.h"

#include <vector>
#include <map>

namespace Realm {

  class CompQueueImpl {
  public:
    CompQueueImpl(void);
    ~CompQueueImpl(void);

    void init(CompletionQueue _me, int _owner);

    static CompletionQueue make_id(const CompQueueImpl &dummy, int owner,
                                   ID::IDType index)
    {
      return ID::make_compqueue(owner, index).convert<CompletionQueue>();
    }

    void set_capacity(size_t _max_size, bool _resizable);

    void destroy(void);

    void add_event(Event event, bool faultaware);

    Event get_local_progress_event(void);
    void add_remote_progress_event(Event event);

    size_t pop_events(Event *events, size_t max_to_pop);

    CompletionQueue me;
    int owner;
    CompQueueImpl *next_free;

    class DeferredDestroy : public EventWaiter {
    public:
      void defer(CompQueueImpl *_cq, Event wait_on);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;

    protected:
      CompQueueImpl *cq;
    };
    DeferredDestroy deferred_destroy;

    // used to track pending remote pop requests
    class RemotePopRequest {
    public:
      RemotePopRequest(Event *_events, size_t _capacity);

      Mutex mutex;
      Mutex::CondVar condvar;
      bool completed;
      size_t count, capacity;
      Event *events;
    };

  protected:
    class CompQueueWaiter : public EventWaiter {
    public:
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;

      CompQueueImpl *cq;
      Event wait_on;
      bool faultaware;
      CompQueueWaiter *next_free;
    };

    void add_completed_event(Event event, CompQueueWaiter *waiter, TimeLimit work_until);

    static const size_t CQWAITER_BATCH_SIZE = 16;
    class CompQueueWaiterBatch {
    public:
      CompQueueWaiterBatch(CompQueueImpl *cq, CompQueueWaiterBatch *_next);
      ~CompQueueWaiterBatch(void);

      CompQueueWaiter waiters[CQWAITER_BATCH_SIZE];
      CompQueueWaiterBatch *next_batch;
    };

    Mutex mutex; // protects everything below here

    bool resizable;
    size_t max_events;
    atomic<size_t> wr_ptr, rd_ptr, pending_events;
    // used if resizable==false
    atomic<size_t> commit_ptr, consume_ptr;
    // used if resizable==true
    size_t cur_events;

    Event *completed_events;

    atomic<bool> has_progress_events;
    GenEventImpl *local_progress_event;
    EventImpl::gen_t local_progress_event_gen;
    // TODO: small vector
    std::vector<Event> remote_progress_events;
    atomic<CompQueueWaiter *> first_free_waiter;
    CompQueueWaiterBatch *batches;
  };

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

}; // namespace Realm

#endif // ifndef REALM_COMP_QUEUE_IMPL_H
