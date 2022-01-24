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

// INCLUDED FROM operation.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/operation.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Operation
  //

  inline Operation::Operation(GenEventImpl *_finish_event,
			      EventImpl::gen_t _finish_gen,
                              const ProfilingRequestSet &_requests)
    : finish_event(_finish_event)
    , finish_gen(_finish_gen)
    , refcount(1)
    , state(ProfilingMeasurements::OperationStatus::WAITING)
    , requests(_requests)
    , all_work_items(0)
    , pending_work_items(1 /* i.e. the main work item */)
    , failed_work_items(0 /* hopefully it stays that way*/)
  {
    status.error_code = 0;
    measurements.import_requests(requests); 
    wants_timeline = measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>();
    wants_gpu_timeline = measurements.wants_measurement<ProfilingMeasurements::OperationTimelineGPU>();
    wants_event_waits = measurements.wants_measurement<ProfilingMeasurements::OperationEventWaits>();
    if(wants_timeline)
      timeline.record_create_time();
  }

  inline void Operation::add_reference(void)
  {
    refcount.fetch_add_acqrel(1);
  }

  inline void Operation::remove_reference(void)
  {
    int left = refcount.fetch_sub_acqrel(1) - 1;
    if(left == 0)
      delete this;
  }

  // must only be called by thread performing operation (i.e. not thread safe)
  // once added, the item belongs to the operation (i.e. will be deleted with
  //  the operation)
  inline void Operation::add_async_work_item(AsyncWorkItem *item)
  {
    // NO lock taken
    pending_work_items.fetch_add_acqrel(1);
    // use compare-and-swap loop to atomically append this item to the front
    //  of the list
    while(true) {
      AsyncWorkItem *prev_head = all_work_items.load();
      item->next_item = prev_head;
      if(all_work_items.compare_exchange(prev_head, item))
	break;
    }
  }

  // called by AsyncWorkItem::mark_finished
  inline void Operation::work_item_finished(AsyncWorkItem *item, bool successful)
  {
    // update this count first
    if (!successful)
      failed_work_items.fetch_add(1);

    // no per-work-item data to record, so just decrement the count, and if it goes
    //  to zero, we're complete
    int remaining = pending_work_items.fetch_sub_acqrel(1) - 1;

    if(remaining == 0) {
      mark_completed();
   }
  }

  inline bool Operation::cancellation_requested(void) const
  {
    return state.load() == Status::INTERRUPT_REQUESTED;
  }

  inline Event Operation::get_finish_event(void) const
  {
    return finish_event->make_event(finish_gen);
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

  inline bool Operation::wants_gpu_work_start() const
  {
    return wants_gpu_timeline;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Operation::AsyncWorkItem
  //
  
  inline Operation::AsyncWorkItem::AsyncWorkItem(Operation *_op)
    : op(_op)
    , next_item(0)
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
