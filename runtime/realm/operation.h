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

#ifndef REALM_OPERATION_H
#define REALM_OPERATION_H

#include "realm/profiling.h"
#include "realm/event_impl.h"
#include "realm/atomics.h"

#include "realm/network.h"

#include <set>
#include <iostream>

namespace Realm {

  class Operation {
  protected:
    // must be subclassed
    Operation(GenEventImpl *_finish_event, EventImpl::gen_t _finish_gen,
	      const ProfilingRequestSet &_requests);

    // can't destroy directly either - done when last reference is removed
    // (subclasses may still override the method - just not call it directly)
    virtual ~Operation(void);

  public:
    void add_reference(void);
    void remove_reference(void);

    // marks operation ready - returns true if it should be enqueued for execution
    //  (i.e. it hasn't been cancelled)
    virtual bool mark_ready(void);

    // marks operation started - returns true if successful, false if a cancellation
    //  request has arrived
    virtual bool mark_started(void);

    virtual void mark_finished(bool successful);
    virtual void mark_terminated(int error_code, const ByteArray& details);

    // returns true if its able to perform the cancellation (or if nothing can be done)
    // returns false if a subclass wants to try some other means to cancel an operation
    virtual bool attempt_cancellation(int error_code, const void *reason_data, size_t reason_size);

    virtual void set_priority(int new_priority);

    // a common reason for cancellation is a poisoned precondition - this helper takes care
    //  of recording the error code and marking the operation as (unsuccessfully) finished
    virtual void handle_poisoned_precondition(Event pre);

    bool cancellation_requested(void) const;

    virtual void print(std::ostream& os) const = 0;

    // abstract class to describe asynchronous work started by an operation
    //  that must finish for the operation to become "complete"
    class AsyncWorkItem {
    public:
      AsyncWorkItem(Operation *_op);
      virtual ~AsyncWorkItem(void);

      void mark_finished(bool successful);

      virtual void request_cancellation(void) = 0;

      virtual void print(std::ostream& os) const = 0;

    protected:
      Operation *op;

      // the next_item field is effectively owned by the Operation class
      friend class Operation;
      friend std::ostream& operator<<(std::ostream& os, Operation *op);

      AsyncWorkItem *next_item;
    };

    // once added, the item belongs to the operation (i.e. will be deleted with
    //  the operation)
    void add_async_work_item(AsyncWorkItem *item);

    // used to record event wait intervals, if desired
    ProfilingMeasurements::OperationEventWaits::WaitInterval *create_wait_interval(Event e);

    // used to measure when device-side work starts for a gpu task
    bool wants_gpu_work_start() const;
    void mark_gpu_work_start();

  protected:
    // called by AsyncWorkItem::mark_finished from an arbitrary thread
    void work_item_finished(AsyncWorkItem *item, bool successful);
    virtual void mark_completed(void);

    void clear_profiling(void);
    void reconstruct_measurements();

    void trigger_finish_event(bool poisoned);

    void send_profiling_data(void);

    GenEventImpl *finish_event;
    EventImpl::gen_t finish_gen;
    atomic<int> refcount;
  public:
    Event get_finish_event(void) const;
  protected:
    typedef ProfilingMeasurements::OperationStatus Status;
    atomic<Status::Result> state;

    // allow operations to lazily update their state
    virtual Status::Result get_state(void);

    ProfilingMeasurements::OperationStatus status;
    bool wants_timeline;
    ProfilingMeasurements::OperationTimeline timeline;
    bool wants_gpu_timeline;
    ProfilingMeasurements::OperationTimelineGPU timeline_gpu; // gpu start/end times
    bool wants_event_waits;
    ProfilingMeasurements::OperationEventWaits waits;
    ProfilingRequestSet requests; 
    ProfilingMeasurementCollection measurements;

    // append-only list (until Operation destruction)
    atomic<AsyncWorkItem *> all_work_items;
    atomic<int> pending_work_items;  // uses atomics so we don't have to take lock to check
    atomic<int> failed_work_items;
    
    friend std::ostream& operator<<(std::ostream& os, Operation *op);
  };

  class OperationTable {
  public:
    OperationTable(void);
    ~OperationTable(void);

    // Operations are 'owned' by the table - the table will free them once it
    //  gets the completion event for it
    void add_local_operation(Event finish_event, Operation *local_op);
    void add_remote_operation(Event finish_event, int remote_note);

    void request_cancellation(Event finish_event, const void *reason_data, size_t reason_size);

    void set_priority(Event finish_event, int new_priority);

    void print_operations(std::ostream& os);
    
    static void register_handlers(void);

    // checks that all operations have finished before shutdown
    void shutdown_check(void);

  protected:
    void event_triggered(Event e);

#if 0
    class TableCleaner : public EventWaiter {
    public:
      TableCleaner(OperationTable *_table);
      virtual bool event_triggered(bool poisoned);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      OperationTable *table;
    };
#endif

    struct TableEntry : public EventWaiter {
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

      OperationTable *table;
      Event finish_event;
      Operation *local_op;
      int remote_node;
      bool pending_cancellation;
      void *reason_data;
      size_t reason_size;
    };
    typedef std::map<Event, TableEntry> Table;

#ifdef REALM_USE_OPERATION_TABLE
    // event table is protected by a mutex
    // try to avoid a serial bottleneck by splitting events over 4 different tables
    static const int NUM_TABLES = 4;
    
    Mutex mutexes[NUM_TABLES];
    Table tables[NUM_TABLES];
    //TableCleaner cleaner;
#endif
  };

  struct CancelOperationMessage {
    Event finish_event;

    static void handle_message(NodeID sender, const CancelOperationMessage &msg,
			       const void *data, size_t datalen);
  };

};

#include "realm/operation.inl"

#endif // REALM_OPERATION_H
