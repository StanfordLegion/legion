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
    : capture_timeline(false)
  {
  }

  Operation::Operation(const ProfilingRequestSet &reqs)
    : requests(reqs)
  {
    measurements.import_requests(reqs); 
    capture_timeline = measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>();
    if(capture_timeline)
      timeline.record_create_time();
  }

  Operation::~Operation(void)
  {
    if(requests.request_count() > 0) {
      // send profiling requests only when the timeline is valid
      if(capture_timeline && timeline.is_valid()) {
        measurements.add_measurement(timeline);
        measurements.send_responses(requests);
      }
      else if(!capture_timeline) {
        measurements.send_responses(requests);
      }
    }
  }

  void Operation::mark_ready(void)
  {
    if(capture_timeline)
      timeline.record_ready_time();
  }

  void Operation::mark_started(void)
  {
    if(capture_timeline)
      timeline.record_start_time();
  }

  void Operation::mark_completed(void)
  {
    if(capture_timeline)
      timeline.record_end_time();
  }

  void Operation::clear_profiling(void)
  {
    capture_timeline = false;
    requests.clear();
    measurements.clear();
  }

  void Operation::reconstruct_measurements()
  {
    measurements.import_requests(requests);
    capture_timeline = measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>();
    if(capture_timeline)
      timeline.record_create_time();
  }

}; // namespace Realm
