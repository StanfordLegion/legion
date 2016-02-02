/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "activemsg.h"

#include <set>
#include <iostream>

namespace Realm {

  class Operation {
  protected:
    // must be subclassed
    Operation(Event _finish_event, const ProfilingRequestSet &_requests);

    // can't destroy directly either - done when last reference is removed
    // (subclasses may still override the method - just not call it directly)
    virtual ~Operation(void);

  public:
    void add_reference(void);
    void remove_reference(void);

    virtual void mark_ready(void);
    virtual void mark_started(void);
    virtual void mark_finished(void);

    // returns true if its able to perform the cancellation (or if nothing can be done)
    // returns false if a subclass wants to try some other means to cancel an operation
    virtual bool attempt_cancellation(int error_code, const void *reason_data, size_t reason_size);

    virtual void print(std::ostream& os) const = 0;

    // abstract class to describe asynchronous work started by an operation
    //  that must finish for the operation to become "complete"
    class AsyncWorkItem {
    public:
      AsyncWorkItem(Operation *_op);
      virtual ~AsyncWorkItem(void);

      void mark_finished(void);

      virtual void request_cancellation(void) = 0;

      virtual void print(std::ostream& os) const = 0;

    protected:
      Operation *op;
    };

    // must only be called by thread performing operation (i.e. not thread safe)
    // once added, the item belongs to the operation (i.e. will be deleted with
    //  the operation)
    void add_async_work_item(AsyncWorkItem *item);

  protected:
    // called by AsyncWorkItem::mark_finished from an arbitrary thread
    void work_item_finished(AsyncWorkItem *item);

    virtual void mark_completed(void);

    void clear_profiling(void);
    void reconstruct_measurements();

    void trigger_finish_event(bool poisoned);

    void send_profiling_data(void);

    Event finish_event;
    int refcount;
  public:
    Event get_finish_event(void) const { return finish_event; }
  protected:
    ProfilingMeasurements::OperationStatus status;
    ProfilingMeasurements::OperationTimeline timeline;
    ProfilingRequestSet requests; 
    ProfilingMeasurementCollection measurements;

    std::set<AsyncWorkItem *> all_work_items;
    int pending_work_items;  // uses atomics so we don't have to take lock to check
    
    friend std::ostream& operator<<(std::ostream& os, const Operation *op);
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
    
    static int register_handlers(gasnet_handlerentry_t *handlers);

  protected:
    void event_triggered(Event e);

    class TableCleaner : public EventWaiter {
    public:
      TableCleaner(OperationTable *_table);
      virtual bool event_triggered(Event e, bool poisoned);
      virtual void print(std::ostream& os) const;

    protected:
      OperationTable *table;
    };

    struct TableEntry {
      //Event finish_event;
      Operation *local_op;
      int remote_node;
      bool pending_cancellation;
      void *reason_data;
      size_t reason_size;
    };
    typedef std::map<Event, TableEntry> Table;

    // event table is protected by a mutex
    // try to avoid a serial bottleneck by splitting events over 4 different tables
    static const int NUM_TABLES = 4;
    
    GASNetHSL mutexes[NUM_TABLES];
    Table tables[NUM_TABLES];
    TableCleaner cleaner;
  };

};

#include "operation.inl"

#endif // REALM_OPERATION_H
