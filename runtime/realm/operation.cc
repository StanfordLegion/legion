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

  Operation::Operation(void)
  {
    timeline.record_create_time();
  }

  Operation::Operation(const ProfilingRequestSet &reqs)
    : requests(reqs)
  {
    measurements.import_requests(reqs); 
    timeline.record_create_time();
  }

  Operation::~Operation(void)
  {
    // always send profiling responses, even if no measurements were made
    if(requests.request_count() > 0) {
      if(measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>())
	measurements.add_measurement(timeline);

      measurements.send_responses(requests);
    }
  }

  void Operation::mark_ready(void)
  {
    timeline.record_ready_time();
  }

  void Operation::mark_started(void)
  {
    timeline.record_start_time();
  }

  void Operation::mark_completed(void)
  {
    timeline.record_end_time();
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
