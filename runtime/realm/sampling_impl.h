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

// sampling profiler implementation for Realm

#ifndef REALM_SAMPLING_IMPL_H
#define REALM_SAMPLING_IMPL_H

#include "realm/sampling.h"
#include "realm/threads.h"

namespace Realm {

  class SamplingProfilerImpl;

  class GaugeSampleBuffer {
  public:
    GaugeSampleBuffer(int _sampler_id);
    virtual ~GaugeSampleBuffer(void);
    
    virtual void write_data(int fd) = 0;
    
    int sampler_id;
    int compressed_len;
    int first_sample;
    int last_sample;
  };

  template <typename T>
  class GaugeSampleBufferImpl : public GaugeSampleBuffer {
  public:
    GaugeSampleBufferImpl(int _sampler_id, size_t _reserve);
    
    virtual void write_data(int fd);

    std::vector<typename T::Sample> samples;
    std::vector<unsigned short> run_lengths;
  };

  class GaugeSampler {
  public:
    GaugeSampler(int _sampler_id, SamplingProfilerImpl *_profiler);
    virtual ~GaugeSampler(void);

    virtual bool sample_gauge(int sample_index) = 0;
    virtual GaugeSampleBuffer *buffer_swap(size_t new_buffer_size,
					   bool nonempty_only = false) = 0;

  protected:
    friend class ProfilingGauges::Gauge;
    friend class SamplingProfilerImpl;

    template <typename T>
    void perform_sample(ProfilingGauges::AbsoluteGauge<T>& gauge,
			typename ProfilingGauges::AbsoluteGauge<T>::Sample &sample);
    template <typename T>
    void perform_sample(ProfilingGauges::AbsoluteRangeGauge<T>& gauge,
			typename ProfilingGauges::AbsoluteRangeGauge<T>::Sample &sample);
    template <typename T>
    void perform_sample(ProfilingGauges::EventCounter<T>& gauge, 
			typename ProfilingGauges::EventCounter<T>::Sample &sample);

    int sampler_id;
    SamplingProfilerImpl *profiler;
    Mutex mutex;  // prevents deletion of a gauge during sampling
    bool gauge_exists;
    GaugeSampler *next;
  };

  template <typename T>
  class GaugeSamplerImpl : public GaugeSampler {
  public:
    GaugeSamplerImpl(int _sampler_id, SamplingProfilerImpl *_profiler, T *_gauge,
		     SampleFile::PacketNewGauge *info);

    virtual bool sample_gauge(int sample_index);
    virtual GaugeSampleBuffer *buffer_swap(size_t new_buffer_size,
					   bool nonempty_only = false);

  protected:
    T *gauge;
    size_t buffer_size;
    GaugeSampleBufferImpl<T> *buffer;
  };

  class DelayedGaugeAddition {
  public:
    DelayedGaugeAddition(ProfilingGauges::Gauge *_gauge, DelayedGaugeAddition *_next);
    virtual ~DelayedGaugeAddition(void);

    virtual GaugeSampler *create_sampler(int sampler_id,
					 SamplingProfilerImpl *profiler,
					 SampleFile::PacketNewGauge *info) = 0;

    ProfilingGauges::Gauge *gauge;
    DelayedGaugeAddition *next;
  };

  template <typename T>
  class DelayedGaugeAdditionImpl : public DelayedGaugeAddition {
  public:
    DelayedGaugeAdditionImpl(ProfilingGauges::Gauge *_gauge, DelayedGaugeAddition *_next);

    virtual GaugeSampler *create_sampler(int sampler_id,
					 SamplingProfilerImpl *profiler,
					 SampleFile::PacketNewGauge *info);
  };
  
  class SamplingProfilerImpl {
  protected:
    friend class SamplingProfiler;

    SamplingProfilerImpl(bool _is_default);
    ~SamplingProfilerImpl(void);
    
  public:
    void configure_from_cmdline(std::vector<std::string>& cmdline,
				CoreReservationSet& crs);
    
    void flush_data(void);
    void shutdown(void);

    static SamplingProfiler& get_profiler(void);
      
    template <typename T>
    GaugeSampler *add_gauge(T *gauge);

    void remove_gauge(ProfilingGauges::Gauge *gauge, GaugeSampler *sampler);
      
    void sampler_loop(void);

  protected:
    bool parse_profile_pattern(const std::string& s);

    bool is_default;
    Mutex mutex;
    bool is_configured, is_shut_down;
    bool cfg_enabled;
    size_t cfg_sample_interval;
    size_t cfg_buffer_size;
    std::vector<std::string> cfg_patterns;
    atomic<int> next_sampler_id;
    atomic<int> next_sample_index;

    bool pattern_match(const std::string &name) const;
    
    // list of active samplers
    std::vector<SampleFile::PacketNewGauge *> new_sampler_infos;
    GaugeSampler *sampler_head;
    GaugeSampler **sampler_tail;

    // we don't create samplers until we're configured - defer any additions that
    //  come before that
    DelayedGaugeAddition *delayed_additions;

    CoreReservation *core_rsrv;
    Thread *sampling_thread;
    int output_fd;
    bool flush_requested;
    ProfilingGauges::AbsoluteGauge<long long> *sampling_start;
    ProfilingGauges::EventCounter<long long> *sampling_time;
  };

};

#endif
