/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// INCLDUED FROM operation.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/operation.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Operation
  //

  inline Operation::Operation(Event _finish_event,
                              const ProfilingRequestSet &_requests)
    : finish_event(_finish_event)
    , refcount(1)
    , requests(_requests)
    , pending_work_items(1 /* i.e. the main work item */)
    , failed_work_items(0 /* hopefully it stays that way*/)
  {
    status.result = ProfilingMeasurements::OperationStatus::WAITING;
    status.error_code = 0;
    measurements.import_requests(requests); 
    timeline.record_create_time();
    wants_event_waits = measurements.wants_measurement<ProfilingMeasurements::OperationEventWaits>();
  }

  inline void Operation::add_reference(void)
  {
    __sync_fetch_and_add(&refcount, 1);
  }

  inline void Operation::remove_reference(void)
  {
    int left = __sync_add_and_fetch(&refcount, -1);
    if(left == 0)
      delete this;
  }

  // must only be called by thread performing operation (i.e. not thread safe)
  // once added, the item belongs to the operation (i.e. will be deleted with
  //  the operation)
  inline void Operation::add_async_work_item(AsyncWorkItem *item)
  {
    // NO lock taken
    __sync_fetch_and_add(&pending_work_items, 1);
    all_work_items.insert(item);
  }

  // called by AsyncWorkItem::mark_finished
  inline void Operation::work_item_finished(AsyncWorkItem *item, bool successful)
  {
    // update this count first
    if(!successful)
      __sync_fetch_and_add(&failed_work_items, 1);

    // no per-work-item data to record, so just decrement the count, and if it goes
    //  to zero, we're complete
    int remaining = __sync_sub_and_fetch(&pending_work_items, 1);

    if(remaining == 0)
      mark_completed();
  }

  inline bool Operation::cancellation_requested(void) const
  {
    return status.result == Status::INTERRUPT_REQUESTED;
  }

  // used to record event wait intervals, if desired
  inline ProfilingMeasurements::OperationEventWaits::WaitInterval *Operation::create_wait_interval(Event e)
  {
    if(wants_event_waits) {
      size_t idx = waits.intervals.size();
      waits.intervals.resize(idx + 1);
      ProfilingMeasurements::OperationEventWaits::WaitInterval *interval = &waits.intervals[idx];
      interval->wait_event = e;
      return interval;
    } else 
      return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Operation::AsyncWorkItem
  //
  
  inline Operation::AsyncWorkItem::AsyncWorkItem(Operation *_op)
    : op(_op)
  {
  }

  inline Operation::AsyncWorkItem::~AsyncWorkItem(void)
  {
  }

  inline void Operation::AsyncWorkItem::mark_finished(bool successful)
  {
    op->work_item_finished(this, successful);
  }


}; // namespace Realm
