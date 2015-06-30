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

// profiling infrastructure for Realm tasks, copies, etc.

#ifndef REALM_PROFILING_H
#define REALM_PROFILING_H

#include <vector>
#include <set>
#include <map>

#include "lowlevel.h"
#include "bytearray.h"

namespace Realm {
  // hacking in references to old namespace for now
  typedef LegionRuntime::LowLevel::Processor Processor;
  typedef LegionRuntime::LowLevel::Processor::TaskFuncID TaskFuncID;

  // through the wonders of templates, users should never need to work with 
  //  these IDs directly
  enum ProfilingMeasurementID {
    PMID_STATUS,    // completion status of operation
    PMID_TIMELINE,  // when task was ready, started, completed
  };

  namespace ProfilingMeasurements {
    struct OperationStatus {
      static const ProfilingMeasurementID ID = PMID_STATUS;

      enum Result {
	COMPLETED_SUCCESSFULLY,
	COMPLETED_WITH_ERRORS,
	RUNNING,
	TERMINATED_EARLY,
	WAITING,
	READY,
	CANCELLED,  // cancelled without being started
      };

      Result result;
      int error_code;
      ByteArray error_details;
    };

    struct OperationTimeline {
      static const ProfilingMeasurementID ID = PMID_TIMELINE;
          
      // all times reported in nanoseconds from some arbitrary (but fixed, for a given
      //  execution) reference time
      typedef unsigned long long timestamp_t;
      static const timestamp_t INVALID_TIMESTAMP = 0;

      timestamp_t create_time; // when was operation created?
      timestamp_t ready_time;  // when was operation ready to proceed?
      timestamp_t start_time;  // when did operation start?
      timestamp_t end_time;    // when did operation end?

      inline void record_create_time(void);
      inline void record_ready_time(void);
      inline void record_start_time(void);
      inline void record_end_time(void);
    };
  };

  class ProfilingRequest {
  public:
    ProfilingRequest(Processor _response_proc, TaskFuncID _response_task_id);
    ProfilingRequest(const ProfilingRequest& to_copy);

    ~ProfilingRequest(void);

    ProfilingRequest& add_user_data(const void *payload, size_t payload_size);

    template <typename T>
    ProfilingRequest &add_measurement(void);

  protected:
    friend class ProfilingMeasurementCollection;

    Processor response_proc;
    TaskFuncID response_task_id;
    void *user_data;
    size_t user_data_size;
    std::set<ProfilingMeasurementID> requested_measurements;
  };

  // manages a set of profiling requests attached to a Realm operation
  class ProfilingRequestSet {
  public:
    ProfilingRequestSet(void);
    ProfilingRequestSet(const ProfilingRequestSet& to_copy);

    ~ProfilingRequestSet(void);

    ProfilingRequest& add_request(Processor response_proc, TaskFuncID response_task_id,
				  const void *payload = 0, size_t payload_size = 0);

    size_t request_count(void) const;

  protected:
    friend class ProfilingMeasurementCollection;

    std::vector<ProfilingRequest *> requests;
  };

  class ProfilingMeasurementCollection {
  public:
    ProfilingMeasurementCollection(void);
    ~ProfilingMeasurementCollection(void);

    void import_requests(const ProfilingRequestSet& prs);
    void send_responses(const ProfilingRequestSet& prs) const;

    template <typename T>
    bool wants_measurement(void) const;

    template <typename T>
    void add_measurement(const T& data);

  protected:
    std::set<ProfilingMeasurementID> requested_measurements;

    struct MeasurementData {
      void* base;
      size_t size;
    };

    std::map<ProfilingMeasurementID, MeasurementData> measurements;
  };

  class ProfilingResponse {
  public:
    // responses need to be deserialized from the response task's argument data
    ProfilingResponse(const void *_data, size_t _data_size);
    ~ProfilingResponse(void);

    const void *user_data(void) const;
    size_t user_data_size(void) const;

    // even if a measurement was requested, it may not have been performed - use
    //  this to check
    template <typename T>
    bool has_measurement(void) const;

    // extracts a measurement (if available), returning a dynamically allocated result -
    //  caller should delete it when done
    template <typename T>
    T *get_measurement(void) const;

  protected:
    const char *data;
    size_t data_size;
    int measurement_count;
    size_t user_data_offset;
    const int *ids;

    bool find_id(int id, int& offset, int& size) const;
  };
}; // namespace Realm

#include "profiling.inl"

#endif // ifdef REALM_PROFILING_H
