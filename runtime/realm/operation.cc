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

#include "operation.h"

namespace Realm {

  Operation::~Operation(void)
  {
    // delete all of the async work items we were given to track
    for(std::set<AsyncWorkItem *>::iterator it = all_work_items.begin();
	it != all_work_items.end();
	it++)
      delete *it;
    all_work_items.clear();
  }

  void Operation::mark_ready(void)
  {
    timeline.record_ready_time();
  }

  void Operation::mark_started(void)
  {
    timeline.record_start_time();
  }

  void Operation::mark_finished(void)
  {
    timeline.record_end_time();

    // do an atomic decrement of the work counter to see if we're also complete
    int remaining = __sync_sub_and_fetch(&pending_work_items, 1);

    if(remaining == 0)
      mark_completed();    
  }

  void Operation::mark_completed(void)
  {
    timeline.record_complete_time();

    // once complete, we can send profiling responses
    if(requests.request_count() > 0) {
      if(measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>())
	measurements.add_measurement(timeline);

      measurements.send_responses(requests);
    }

    // we delete ourselves for now - eventually the OperationTable will do this
    delete this;
  }

  void Operation::clear_profiling(void)
  {
    requests.clear();
    measurements.clear();
  }

  void Operation::reconstruct_measurements()
  {
    measurements.import_requests(requests);
    timeline.record_create_time();
  }

}; // namespace Realm
