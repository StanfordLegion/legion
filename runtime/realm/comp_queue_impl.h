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
#include <realm/activemsg.h>
#include "realm/event_impl.h"

#include <vector>

namespace Realm {

  class CompQueueImpl {
  public:
    CompQueueImpl() = default;
    ~CompQueueImpl();

    void init(CompletionQueue _me, int _owner);

    static CompletionQueue make_id(const CompQueueImpl &dummy, int owner,
                                   ID::IDType index)
    {
      return ID::make_compqueue(owner, index).convert<CompletionQueue>();
    }

    void set_capacity(size_t _max_size, bool _resizable);
    size_t get_capacity() const;
    size_t get_pending_events() const;

    void destroy(void);

    void add_event(Event event, bool faultaware);
    void add_event(Event event, EventImpl *ev_impl, bool faultaware);

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

    bool resizable = false;
    size_t max_events = 0;
    atomic<size_t> wr_ptr = atomic<size_t>(0);
    atomic<size_t> rd_ptr = atomic<size_t>(0);
    atomic<size_t> pending_events = atomic<size_t>(0);

    atomic<size_t> commit_ptr = atomic<size_t>(0);
    atomic<size_t> consume_ptr = atomic<size_t>(0);
    size_t cur_events = 0;

    std::unique_ptr<Event[]> completed_events = nullptr;

    atomic<bool> has_progress_events = atomic<bool>(0);
    GenEventImpl *local_progress_event = nullptr;
    EventImpl::gen_t local_progress_event_gen = 0;
    std::vector<Event> remote_progress_events;
    atomic<CompQueueWaiter *> first_free_waiter = atomic<CompQueueWaiter *>(nullptr);
    CompQueueWaiterBatch *batches = nullptr;
  };
}; // namespace Realm

#endif // ifndef REALM_COMP_QUEUE_IMPL_H
