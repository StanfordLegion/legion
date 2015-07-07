
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
    capture_timeline = measurements.wants_measurement<
                        ProfilingMeasurements::OperationTimeline>();
    if (capture_timeline)
      timeline.record_create_time();
  }

  Operation::~Operation(void)
  {
    if (requests.request_count() > 0) {
      if (capture_timeline)
        measurements.add_measurement(timeline);
      measurements.send_responses(requests);
    }
  }

  void Operation::mark_ready(void)
  {
    if (capture_timeline)
      timeline.record_ready_time();
  }

  void Operation::mark_started(void)
  {
    if (capture_timeline)
      timeline.record_start_time();
  }

  void Operation::mark_completed(void)
  {
    if (capture_timeline)
      timeline.record_end_time();
  }

}; // namespace Realm
