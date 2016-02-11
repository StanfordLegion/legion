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

// INCLDUED FROM profiling.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "profiling.h"
#include "utilities.h"
#include "serialize.h"

TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurementID);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationTimeline);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationMemoryUsage);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationProcessorUsage);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::InstanceMemoryUsage);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::InstanceTimeline);

#include "timers.h"

namespace Realm {

  namespace ProfilingMeasurements {

    TYPE_IS_SERIALIZABLE(OperationStatus::Result);

    template <typename S>
    bool serdez(S& serdez, const OperationStatus& s)
    {
      return ((serdez & s.result) &&
	      (serdez & s.error_code) &&
	      (serdez & s.error_details));
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationBacktrace
    //

    template <typename S>
    bool serdez(S& serdez, const OperationBacktrace& b)
    {
      return (serdez & b.backtrace);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationTimeLine
    //

    inline void OperationTimeline::record_create_time(void)
    {
      create_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_ready_time(void)
    {
      ready_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_start_time(void)
    {
      start_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_end_time(void)
    {
      end_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_complete_time(void)
    {
      complete_time = Clock::current_time_in_nanoseconds();
    }

    inline bool OperationTimeline::is_valid(void)
    {
      return ((create_time != INVALID_TIMESTAMP) &&
	      (ready_time != INVALID_TIMESTAMP) &&
	      (start_time != INVALID_TIMESTAMP) &&
	      (end_time != INVALID_TIMESTAMP) &&
	      (complete_time != INVALID_TIMESTAMP));
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // struct InstanceTimeLine
    //

    inline void InstanceTimeline::record_create_time(void)
    {
      create_time = Clock::current_time_in_nanoseconds();
    }

    inline void InstanceTimeline::record_delete_time(void)
    {
      delete_time = Clock::current_time_in_nanoseconds();
    }

  }; // namespace ProfilingMeasurements

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequest
  //

  template <typename T>
  ProfilingRequest &ProfilingRequest::add_measurement(void) 
  {
    // SJT: the typecast here is a NOP, but somehow it avoids weird linker errors
    requested_measurements.insert((ProfilingMeasurementID)T::ID);
    return *this;
  }

  template <typename S>
  bool serialize(S &s, const ProfilingRequest &pr)
  {
    return((s << pr.response_proc) &&
	   (s << pr.response_task_id) &&
	   (s << pr.user_data) &&
	   (s << pr.requested_measurements));
  }

  template <typename S>
  /*static*/ ProfilingRequest *ProfilingRequest::deserialize_new(S &s)
  {
    // have to get fields of the reqeuest in order to build it
    Processor p;
    Processor::TaskFuncID fid;
    if(!(s >> p)) return 0;
    if(!(s >> fid)) return 0;
    ProfilingRequest *pr = new ProfilingRequest(p, fid);
    if(!(s >> pr->user_data) ||
       !(s >> pr->requested_measurements)) {
      delete pr;
      return 0;
    }
    return pr;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingMeasurementCollection
  //

  template <typename T>
  bool ProfilingMeasurementCollection::wants_measurement(void) const
  {
    return requested_measurements.count((ProfilingMeasurementID)T::ID);
  } 

  template <typename T>
  void ProfilingMeasurementCollection::add_measurement(const T& data)
  {
    // we'll assume the caller already asked if the measurement was wanted before
    //  giving us something expensive...

    // no duplicates for now
    assert(measurements.count((ProfilingMeasurementID)T::ID) == 0);

    // serialize the data
    Serialization::DynamicBufferSerializer dbs(128);
#ifndef NDEBUG
    bool ok =
#endif
      dbs << data;
    assert(ok);

    // measurement data is stored in a ByteArray
    ByteArray& md = measurements[(ProfilingMeasurementID)T::ID];
    ByteArray b = dbs.detach_bytearray(-1);  // no trimming
    md.swap(b);  // avoids a copy
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequestSet
  //

  template <typename S>
  bool serialize(S &s, const ProfilingRequestSet &prs)
  {
    size_t len = prs.requests.size();
    if(!(s << len)) return false;
    for(size_t i = 0; i < len; i++)
      if(!(s << *prs.requests[i])) return false;
    return true;
  }
  
  template <typename S>
  bool deserialize(S &s, ProfilingRequestSet &prs)
  {
    size_t len;
    if(!(s >> len)) return false;
    prs.clear(); // erase any existing data cleanly
    prs.requests.reserve(len);
    for(size_t i = 0; i < len; i++) {
      ProfilingRequest *pr = ProfilingRequest::deserialize_new(s);
      if(!pr) return false;
      prs.requests.push_back(pr);
    }
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingResponse
  //

  template <typename T>
  bool ProfilingResponse::has_measurement(void) const
  {
    int offset, size; // not actually used
    return find_id((int)(T::ID), offset, size);
  }

  template <typename T>
  T *ProfilingResponse::get_measurement(void) const
  {
    int offset, size;
    if(find_id((int)(T::ID), offset, size)) {
      Serialization::FixedBufferDeserializer fbd(data + offset, size);
      T *m = new T;
#ifndef NDEBUG
      bool ok =
#endif
        fbd >> *m;
      assert(ok);
      return m;
    } else
      return 0;
  }


}; // namespace Realm
