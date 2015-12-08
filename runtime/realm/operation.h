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

#ifndef REALM_OPERATION_H
#define REALM_OPERATION_H

#include "realm/profiling.h"

#include <set>

namespace Realm {

  class Operation {
  protected:
    // must be subclassed
    Operation(Event _finish_event, const ProfilingRequestSet &_requests);

  protected:
    virtual ~Operation(void);
  public:
    virtual void mark_ready(void);
    virtual void mark_started(void);
    virtual void mark_finished(void);

    // abstract class to describe asynchronous work started by an operation
    //  that must finish for the operation to become "complete"
    class AsyncWorkItem {
    public:
      AsyncWorkItem(Operation *_op);
      virtual ~AsyncWorkItem(void);

      void mark_finished(void);

      virtual void request_cancellation(void) = 0;

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

    void trigger_finish_event(void);

    Event finish_event;
  public:
    Event get_finish_event(void) const { return finish_event; }
  protected:
    ProfilingMeasurements::OperationStatus status;
    ProfilingMeasurements::OperationTimeline timeline;
    ProfilingRequestSet requests; 
    ProfilingMeasurementCollection measurements;

    std::set<AsyncWorkItem *> all_work_items;
    int pending_work_items;  // uses atomics so we don't have to take lock to check
  };

};

#include "operation.inl"

#endif // REALM_OPERATION_H
