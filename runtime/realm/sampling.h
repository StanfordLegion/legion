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

// sampling profiler for Realm

#ifndef REALM_SAMPLING_H
#define REALM_SAMPLING_H

#include <limits.h>
#include <vector>
#include <set>
#include <map>

#include "realm/bytearray.h"
#include "realm/processor.h"
#include "realm/memory.h"
#include "realm/instance.h"
#include "realm/faults.h"
#include "realm/atomics.h"

namespace Realm {

  class SamplingProfiler;
  class GaugeSampler;
  class CoreReservationSet;

  namespace ProfilingGauges {

    // these gauges are created by "user" code and are intended to be nearly free
    //  when profiling is not active so that they don't need to be guarded by
    //  compile-time ifdef's

    // parent class for all gauges
    class Gauge {
    public:
      enum GaugeType {
	GTYPE_UNKNOWN = 0,
	GTYPE_ABSOLUTE = 1,
	GTYPE_ABSOLUTERANGE = 2,
	GTYPE_EVENTCOUNT = 3,
      };

      // if profiler==0, the gauge will be connected to the global default profiler
      Gauge(const std::string& _name);
      ~Gauge(void);

      const std::string name;

    private:
      Gauge(const Gauge& copy_from) {}
      Gauge& operator=(const Gauge& copy_from) { return *this; }

    protected:
      // add_gauge is templated to preserve static type info during delayed construction
      template <typename T>
      static void add_gauge(T *gauge, SamplingProfiler *_profiler);
      void remove_gauge(void);

      GaugeSampler *sampler;

    public:
      // don't actually call this - it's here for linker fun
      static size_t instantiate_templates(void);
    };

    template <typename T>
    class AbsoluteGauge : public Gauge {
    public:
      static const int GAUGE_TYPE = GTYPE_ABSOLUTE;
      typedef T DATA_TYPE;

      // if profiler==0, the gauge will be connected to the global default profiler
      AbsoluteGauge(const std::string& _name, T initval = T(),
		    SamplingProfiler *_profiler = 0);

      AbsoluteGauge<T>& operator=(const AbsoluteGauge<T>& copy_from);

      operator T(void) const;
      AbsoluteGauge<T>& operator=(T to_set);
      AbsoluteGauge<T>& operator+=(T to_add);
      AbsoluteGauge<T>& operator-=(T to_sub);

      struct Sample {
	bool operator==(const Sample& other) const { return value == other.value; }

	T value;
      };
	
    protected:
      friend class Realm::GaugeSampler;

      atomic<T> curval;  // current gauge value
    };

    template <typename T>
    class AbsoluteRangeGauge : public Gauge {
    public:
      static const int GAUGE_TYPE = GTYPE_ABSOLUTERANGE;
      typedef T DATA_TYPE;

      // if profiler==0, the gauge will be connected to the global default profiler
      AbsoluteRangeGauge(const std::string& _name, T initval = T(),
			 SamplingProfiler *_profiler = 0);

      AbsoluteRangeGauge<T>& operator=(const AbsoluteRangeGauge<T>& copy_from);

      operator T(void) const;
      AbsoluteRangeGauge<T>& operator=(T to_set);
      AbsoluteRangeGauge<T>& operator+=(T to_add);
      AbsoluteRangeGauge<T>& operator-=(T to_sub);
	
      struct Sample {
	bool operator==(const Sample& other) const
	{ 
	  return ((value == other.value) && 
		  (minval == other.minval) && 
		  (maxval == other.maxval));
	}

	T value;
	T minval;
	T maxval;
      };

    protected:
      friend class Realm::GaugeSampler;

      atomic<T> curval;  // current gauge value
      atomic<T> minval;  // max value seen since last sample
      atomic<T> maxval;  // min value seen since last sample
    };

    template <typename T = int>
    class EventCounter : public Gauge {
    public:
      static const int GAUGE_TYPE = GTYPE_EVENTCOUNT;
      typedef T DATA_TYPE;

      // if profiler==0, the gauge will be connected to the global default profiler
      EventCounter(const std::string& _name, SamplingProfiler *_profiler = 0);

      operator T(void) const;
      EventCounter<T>& operator+=(T to_add);
	
      struct Sample {
	bool operator==(const Sample& other) const { return count == other.count; }

	T count;
      };
	
    protected:
      friend class Realm::GaugeSampler;

      atomic<T> events;  // events recorded since last sample
    };

  }; // namespace ProfilingGauges

  class SamplingProfiler {
  public:
    // there can be only one "default" sampling profiler - attempts to create a second
    //  will cause a fatal error
    SamplingProfiler(bool is_default);
    ~SamplingProfiler(void);

    void configure_from_cmdline(std::vector<std::string>& cmdline,
				CoreReservationSet& crs);
    void flush_data(void);
    void shutdown(void);

  protected:
    friend class ProfilingGauges::Gauge;

    template <typename T>
    GaugeSampler *add_gauge(T *gauge);

    void *impl;
  };

  namespace SampleFile {

    // no actual code in here - just structure definitions for the file that
    //  gets written by the sampling profiler

    struct PacketHeader {
      enum PacketTypes {
	PACKET_EMPTY = 0,
	PACKET_NEWGAUGE = 1,
	PACKET_SAMPLES = 2,
      };
      unsigned packet_type;
      unsigned packet_size;  // individual packets <= 4GB
      // size does NOT include this 8 byte header
    };

    struct PacketNewGauge {
      int gauge_id;    // assigned by the profiler
      int gauge_type;  // from ProfilingGauges::Gauge::GaugeType
      char gauge_dtype[8];
      char name[48];
    };

    struct PacketSamples {
      int gauge_id;
      int compressed_len;
      int first_sample;
      int last_sample;
      // the type/size of a Sample is determined by gauge_type/dtype above
      // Sample samples[compressed_len]
      // unsigned short run_lengths[compressed_len]
    };

  }; // namespace SampleFile

}; // namespace Realm

#include "realm/sampling.inl"

#endif // ifdef REALM_SAMPLING_H
