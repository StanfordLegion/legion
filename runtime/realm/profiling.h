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

// profiling infrastructure for Realm tasks, copies, etc.

#ifndef REALM_PROFILING_H
#define REALM_PROFILING_H

#include <climits>
#include <vector>
#include <set>
#include <map>

#include "bytearray.h"
#include "processor.h"
#include "memory.h"
#include "instance.h"
#include "faults.h"

namespace Realm {

  // through the wonders of templates, users should never need to work with 
  //  these IDs directly
  enum ProfilingMeasurementID {
    PMID_OP_STATUS,    // completion status of operation
    PMID_OP_BACKTRACE,  // backtrace of a failed operation
    PMID_OP_TIMELINE,  // when task was ready, started, completed
    PMID_OP_EVENT_WAITS,  // intervals when operation is waiting on events
    PMID_OP_PROC_USAGE, // processor used by task
    PMID_OP_MEM_USAGE, // memories used by a copy
    PMID_INST_TIMELINE, // timeline for a physical instance
    PMID_INST_MEM_USAGE, // memory and size used by an instance
  };

  namespace ProfilingMeasurements {
    struct OperationStatus {
      static const ProfilingMeasurementID ID = PMID_OP_STATUS;

      enum Result {
	COMPLETED_SUCCESSFULLY,
	COMPLETED_WITH_ERRORS,
	RUNNING,
	INTERRUPT_REQUESTED,  // running, but a cancellation requested
	TERMINATED_EARLY,
	WAITING,
	READY,
	CANCELLED,  // cancelled without being started
      };

      Result result;
      int error_code;
      ByteArray error_details;
    };

    struct OperationBacktrace {
      static const ProfilingMeasurementID ID = PMID_OP_BACKTRACE;

      Backtrace backtrace;
    };

    struct OperationTimeline {
      static const ProfilingMeasurementID ID = PMID_OP_TIMELINE;
          
      // all times reported in nanoseconds from the start of program execution
      // on some node. This is necessary because clients can't know where the
      // measurement times were recorded and therefore have no reference. There
      // may be skews between the start times of different nodes.
      typedef long long timestamp_t;
      static const timestamp_t INVALID_TIMESTAMP = LLONG_MIN;

      OperationTimeline() :
        create_time(INVALID_TIMESTAMP),
        ready_time(INVALID_TIMESTAMP),
        start_time(INVALID_TIMESTAMP),
        end_time(INVALID_TIMESTAMP),
	complete_time(INVALID_TIMESTAMP)
      { }

      timestamp_t create_time;   // when was operation created?
      timestamp_t ready_time;    // when was operation ready to proceed?
      timestamp_t start_time;    // when did operation start?
      timestamp_t end_time;      // when did operation end (on processor)?
      timestamp_t complete_time; // when was all work for operation complete?

      inline void record_create_time(void);
      inline void record_ready_time(void);
      inline void record_start_time(void);
      inline void record_end_time(void);
      inline void record_complete_time(void);
      inline bool is_valid(void);
    };

    // records time intervals in which the operation was waiting on events
    struct OperationEventWaits {
      static const ProfilingMeasurementID ID = PMID_OP_EVENT_WAITS;
          
      typedef long long timestamp_t;
      static const timestamp_t INVALID_TIMESTAMP = LLONG_MIN;

      struct WaitInterval {
	timestamp_t wait_start;   // when did the interval begin?
	timestamp_t wait_ready;   // when did the event trigger?
	timestamp_t wait_end;     // when did the interval actually end
	Event       wait_event;   // which event was waited on

	inline void record_wait_start(void);
	inline void record_wait_ready(void);
	inline void record_wait_end(void);
      };

      std::vector<WaitInterval> intervals;
    };

    // Track processor used for tasks
    struct OperationProcessorUsage {
      static const ProfilingMeasurementID ID = PMID_OP_PROC_USAGE;
      Processor proc;
    };

    // Track memories used for copies
    struct OperationMemoryUsage {
      static const ProfilingMeasurementID ID = PMID_OP_MEM_USAGE;
      Memory source;
      Memory target;
      size_t size;
    };

    // Track the timeline of an instance
    struct InstanceTimeline {
      static const ProfilingMeasurementID ID = PMID_INST_TIMELINE;

      // all times reported in nanoseconds from the start of program execution
      // on some node. This is necessary because clients can't know where the
      // measurement times were recorded and therefore have no reference. There
      // may be skews between the start times of different nodes.
      typedef unsigned long long timestamp_t;
      static const timestamp_t INVALID_TIMESTAMP = 0;

      RegionInstance instance;      
      timestamp_t create_time; // when was instance created?
      timestamp_t delete_time; // when was the instance deleted?

      inline void record_create_time(void);
      inline void record_delete_time(void);
    };

    // Track properties of an instance
    struct InstanceMemoryUsage {
      static const ProfilingMeasurementID ID = PMID_INST_MEM_USAGE;
      RegionInstance instance;
      Memory memory;
      size_t bytes;
    };
  };

  class ProfilingRequest {
  public:
    ProfilingRequest(Processor _response_proc, Processor::TaskFuncID _response_task_id);
    ProfilingRequest(const ProfilingRequest& to_copy);

    ~ProfilingRequest(void);

    ProfilingRequest& operator=(const ProfilingRequest &rhs);

    ProfilingRequest& add_user_data(const void *payload, size_t payload_size);

    template <typename T>
    ProfilingRequest &add_measurement(void);

    template <typename S> static ProfilingRequest *deserialize_new(S &s);

  protected:
    friend class ProfilingMeasurementCollection;

    template <typename S> friend bool serialize(S &s, const ProfilingRequest &pr);

    Processor response_proc;
    Processor::TaskFuncID response_task_id;
    ByteArray user_data;
    std::set<ProfilingMeasurementID> requested_measurements;
  };

  // manages a set of profiling requests attached to a Realm operation
  class ProfilingRequestSet {
  public:
    ProfilingRequestSet(void);
    ProfilingRequestSet(const ProfilingRequestSet& to_copy);

    ~ProfilingRequestSet(void);

    ProfilingRequestSet& operator=(const ProfilingRequestSet &rhs);

    ProfilingRequest& add_request(Processor response_proc, 
				  Processor::TaskFuncID response_task_id,
				  const void *payload = 0, size_t payload_size = 0);

    size_t request_count(void) const;
    bool empty(void) const;

    void clear(void);

  protected:
    friend class ProfilingMeasurementCollection;

    template <typename S> friend bool serialize(S &s, const ProfilingRequestSet &prs);
    template <typename S> friend bool deserialize(S &s, ProfilingRequestSet &prs);

    std::vector<ProfilingRequest *> requests;
  };

  class ProfilingMeasurementCollection {
  public:
    ProfilingMeasurementCollection(void);
    ~ProfilingMeasurementCollection(void);

    void import_requests(const ProfilingRequestSet& prs);
    void send_responses(const ProfilingRequestSet& prs);
    void clear(void);

    template <typename T>
    bool wants_measurement(void) const;

    template <typename T>
    void add_measurement(const T& data, bool send_complete_responses = true);

  protected:
    void send_response(const ProfilingRequest& pr) const;

    // in order to efficiently send responses as soon as we have all the requested measurements, we
    //  need to know which profiling requests are needed by a given measurement and how many more
    //  measurements each request wants
    std::map<ProfilingMeasurementID, std::vector<const ProfilingRequest *> > requested_measurements;
    std::map<const ProfilingRequest *, int> measurements_left;
    bool completed_requests_present;  // set if a request is completed but could not be sent right away

    std::map<ProfilingMeasurementID, ByteArray> measurements;
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
