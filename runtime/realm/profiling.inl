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

// INCLDUED FROM profiling.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "profiling.h"
#include "utilities.h"
#include "serialize.h"

TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationTimeline);

namespace Realm {

  namespace ProfilingMeasurements {

    TYPE_IS_SERIALIZABLE(OperationStatus::Result);

    template <typename S>
    bool operator&(S& serdez, const OperationStatus& s)
    {
      return ((serdez & s.result) &&
	      (serdez & s.error_code) &&
	      (serdez & s.error_details));
    }

    inline void OperationTimeline::record_create_time(void)
    {
      create_time = LegionRuntime::TimeStamp::get_current_time_in_nanos();
    }

    inline void OperationTimeline::record_ready_time(void)
    {
      ready_time = LegionRuntime::TimeStamp::get_current_time_in_nanos();
    }

    inline void OperationTimeline::record_start_time(void)
    {
      start_time = LegionRuntime::TimeStamp::get_current_time_in_nanos();
    }

    inline void OperationTimeline::record_end_time(void)
    {
      end_time = LegionRuntime::TimeStamp::get_current_time_in_nanos();
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
    bool ok = dbs << data;
    assert(ok);

    MeasurementData& md = measurements[(ProfilingMeasurementID)T::ID];
    md.size = dbs.bytes_used();
    md.base = dbs.detach_buffer();
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
      bool ok = fbd >> *m;
      assert(ok);
      return m;
    } else
      return 0;
  }

}; // namespace Realm
