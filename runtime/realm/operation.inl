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

// INCLDUED FROM operation.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "operation.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Operation
  //

  inline Operation::Operation(void)
    : pending_work_items(1 /* i.e. the main work item */)
  {
    timeline.record_create_time();
  }

  inline Operation::Operation(const ProfilingRequestSet &reqs)
    : requests(reqs)
    , pending_work_items(1 /* i.e. the main work item */)
  {
    measurements.import_requests(reqs); 
    timeline.record_create_time();
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
  inline void Operation::work_item_finished(AsyncWorkItem *item)
  {
    // no per-work-item data to record, so just decrement the count, and if it goes
    //  to zero, we're complete
    int remaining = __sync_sub_and_fetch(&pending_work_items, 1);

    if(remaining == 0)
      mark_completed();
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

  inline void Operation::AsyncWorkItem::mark_finished(void)
  {
    op->work_item_finished(this);
  }


}; // namespace Realm
