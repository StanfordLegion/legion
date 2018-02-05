/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "realm/sampling_impl.h"
#include "realm/cmdline.h"
#include "realm/timers.h"

#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace Realm {

  Logger log_realmprof("realmprof");


  ////////////////////////////////////////////////////////////////////////
  //
  // class GaugeSampleBuffer
  //
  
  GaugeSampleBuffer::GaugeSampleBuffer(int _sampler_id)
    : sampler_id(_sampler_id)
    , compressed_len(0)
    , first_sample(-1)
    , last_sample(-1)
  {}

  GaugeSampleBuffer::~GaugeSampleBuffer(void)
  {}
 

  ////////////////////////////////////////////////////////////////////////
  //
  // class GaugeSampleBufferImpl<T>
  //
  
  template <typename T>
  GaugeSampleBufferImpl<T>::GaugeSampleBufferImpl(int _sampler_id, size_t _reserve)
    : GaugeSampleBuffer(_sampler_id)
  {
    samples.resize(_reserve);
    run_lengths.resize(_reserve);
  }

  template <typename T>
  void GaugeSampleBufferImpl<T>::write_data(int fd)
  {
    size_t samples_size = compressed_len * sizeof(typename T::Sample);
    size_t runlengths_size = compressed_len * sizeof(unsigned short);
    SampleFile::PacketHeader hdr;
    hdr.packet_type = SampleFile::PacketHeader::PACKET_SAMPLES;
    hdr.packet_size = sizeof(SampleFile::PacketSamples) + samples_size + runlengths_size;
    ssize_t amt = write(fd, &hdr, sizeof(hdr));
#ifdef NDEBUG
    (void)amt;
#else
    assert(amt == (ssize_t)sizeof(hdr));
#endif
    SampleFile::PacketSamples pkt;
    pkt.gauge_id = sampler_id;
    pkt.compressed_len = compressed_len;
    pkt.first_sample = first_sample;
    pkt.last_sample = last_sample;
    amt = write(fd, &pkt, sizeof(pkt));
#ifdef NDEBUG
    (void)amt;
#else
    assert(amt == (ssize_t)sizeof(pkt));
#endif
    amt = write(fd, &samples[0], samples_size);
#ifdef NDEBUG
    (void)amt;
#else
    assert(amt == (ssize_t)samples_size);
#endif
    amt = write(fd, &run_lengths[0], runlengths_size);
#ifdef NDEBUG
    (void)amt;
#else
    assert(amt == (ssize_t)runlengths_size);
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GaugeSampler
  //

  GaugeSampler::GaugeSampler(int _sampler_id, SamplingProfilerImpl *_profiler)
    : sampler_id(_sampler_id)
    , profiler(_profiler)
    , gauge_exists(true)
    , next(0)
  {}

  GaugeSampler::~GaugeSampler(void)
  {}

  template <typename T>
  void GaugeSampler::perform_sample(ProfilingGauges::AbsoluteGauge<T>& gauge,
				    typename ProfilingGauges::AbsoluteGauge<T>::Sample &sample)
  {
    // simple copy is all we need
    sample.value = gauge.curval;
  }

  template <typename T>
  void GaugeSampler::perform_sample(ProfilingGauges::AbsoluteRangeGauge<T>& gauge,
				    typename ProfilingGauges::AbsoluteRangeGauge<T>::Sample &sample)
  {
    // want a consistent sample of current/min/max, and then reset min/max to current
    sample.value = gauge.curval;
    do {
      sample.minval = gauge.minval;
    } while(!__sync_bool_compare_and_swap(&gauge.minval, sample.minval, sample.value));
    do {
      sample.maxval = gauge.maxval;
    } while(!__sync_bool_compare_and_swap(&gauge.maxval, sample.maxval, sample.value));
  }

  template <typename T>
  void GaugeSampler::perform_sample(ProfilingGauges::EventCounter<T>& gauge,
				    typename ProfilingGauges::EventCounter<T>::Sample &sample)
  {
    // need to atomically read the value and write 0 back
    sample.count = __sync_fetch_and_and(&gauge.events, 0);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GaugeSamplerImpl<T>
  //

  template <typename T>
  GaugeSamplerImpl<T>::GaugeSamplerImpl(int _sampler_id, SamplingProfilerImpl *_profiler,
					T *_gauge,
					SampleFile::PacketNewGauge *info)
    : GaugeSampler(_sampler_id, _profiler)
    , gauge(_gauge)
    , buffer_size(0)
    , buffer(0)
  {
    info->gauge_id = _sampler_id;
    info->gauge_type = T::GAUGE_TYPE;
    strncpy(info->gauge_dtype, typeid(typename T::DATA_TYPE).name(), 7);
    info->gauge_dtype[7] = 0;
    strncpy(info->name, gauge->name.c_str(), 47);
    info->name[47] = 0;
  }

  template <typename T>
  bool GaugeSamplerImpl<T>::sample_gauge(int sample_index)
  {
    assert(buffer != 0);
    size_t i = buffer->compressed_len;
    assert(i < buffer_size);
    if(i == 0)
      buffer->first_sample = sample_index;
    buffer->last_sample = sample_index;

    perform_sample(*gauge, buffer->samples[i]);

    // see if we can merge with the previous sample (if it exists and isn't full)
    if((i > 0) && (buffer->run_lengths[i - 1] < 0xffff) &&
       (buffer->samples[i - 1] == buffer->samples[i])) {
      // yes, increase the previous sample's run length by 1
      buffer->run_lengths[i - 1]++;
      return false;  // never full if we merge a sample
    } else {
      buffer->run_lengths[i] = 1;
      buffer->compressed_len++;
      return ((i + 1) == buffer_size);
    }
  }

  template <typename T>
  GaugeSampleBuffer *GaugeSamplerImpl<T>::buffer_swap(size_t new_buffer_size,
						      bool nonempty_only /*= false*/)
  {
    if(nonempty_only && buffer && (buffer->compressed_len > 0))
      return 0;

    GaugeSampleBuffer *oldbuffer = buffer;
    buffer_size = new_buffer_size;
    if(new_buffer_size) {
      buffer = new GaugeSampleBufferImpl<T>(sampler_id, new_buffer_size);
    } else
      buffer = 0;
    return oldbuffer;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DelayedGaugeAddition
  //
  
  DelayedGaugeAddition::DelayedGaugeAddition(ProfilingGauges::Gauge *_gauge,
					     DelayedGaugeAddition *_next)
    : gauge(_gauge), next(_next)
  {}

  DelayedGaugeAddition::~DelayedGaugeAddition(void)
  {}

    
  ////////////////////////////////////////////////////////////////////////
  //
  // class DelayedGaugeAdditionImpl<T>
  //
  
  template <typename T>
  DelayedGaugeAdditionImpl<T>::DelayedGaugeAdditionImpl(ProfilingGauges::Gauge *_gauge,
							DelayedGaugeAddition *_next)
    : DelayedGaugeAddition(_gauge, _next)
  {}

  template <typename T>
  GaugeSampler * DelayedGaugeAdditionImpl<T>::create_sampler(int sampler_id,
							     SamplingProfilerImpl *profiler,
							     SampleFile::PacketNewGauge *info)
  {
    return new GaugeSamplerImpl<T>(sampler_id, profiler, static_cast<T *>(gauge), info);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SamplingProfiler
  //

  SamplingProfiler::SamplingProfiler(bool is_default)
  {
    impl = new SamplingProfilerImpl(is_default);
  }

  SamplingProfiler::~SamplingProfiler(void)
  {
    delete (SamplingProfilerImpl *)impl;
  }

  void SamplingProfiler::configure_from_cmdline(std::vector<std::string>& cmdline,
						CoreReservationSet& crs)
  {
    ((SamplingProfilerImpl *)impl)->configure_from_cmdline(cmdline, crs);
  }

  void SamplingProfiler::flush_data(void)
  {
    ((SamplingProfilerImpl *)impl)->flush_data();
  }

  void SamplingProfiler::shutdown(void)
  {
    ((SamplingProfilerImpl *)impl)->shutdown();
  }

  template <typename T>
  GaugeSampler *SamplingProfiler::add_gauge(T *gauge)
  {
    return ((SamplingProfilerImpl *)impl)->add_gauge(gauge);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class DefaultSamplerHandler
  //

  class DefaultSamplerHandler {
  protected:
    DefaultSamplerHandler(void);

  public:
    static DefaultSamplerHandler& get_handler(void);

    DelayedGaugeAddition *install_default_sampler(SamplingProfilerImpl *new_default);
    void remove_default_sampler(SamplingProfilerImpl *old_default);

    template <typename T>
    GaugeSampler *add_gauge_to_default_sampler(T *gauge);

  protected:
    GASNetHSL mutex;
    SamplingProfilerImpl *default_sampler;
    DelayedGaugeAddition *delayed_additions;
  };

  DefaultSamplerHandler::DefaultSamplerHandler(void)
    : default_sampler(0), delayed_additions(0)
  {}

  /*static*/ DefaultSamplerHandler& DefaultSamplerHandler::get_handler(void)
  {
    static DefaultSamplerHandler handler;
    return handler;
  }

  DelayedGaugeAddition *DefaultSamplerHandler::install_default_sampler(SamplingProfilerImpl *new_default)
  {
    AutoHSLLock al(mutex);
    assert(default_sampler == 0);
    default_sampler = new_default;
    DelayedGaugeAddition *to_return = delayed_additions;
    delayed_additions = 0;
    return to_return;
  }

  void DefaultSamplerHandler::remove_default_sampler(SamplingProfilerImpl *old_default)
  {
    AutoHSLLock al(mutex);
    assert(default_sampler == old_default);
    default_sampler = 0;
  }

  template <typename T>
  GaugeSampler *DefaultSamplerHandler::add_gauge_to_default_sampler(T *gauge)
  {
    AutoHSLLock al(mutex);
    if(default_sampler) {
      return default_sampler->add_gauge(gauge);
    } else {
      delayed_additions = new DelayedGaugeAdditionImpl<T>(gauge, delayed_additions);
      return 0;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SamplingProfilerImpl
  //

  SamplingProfilerImpl::SamplingProfilerImpl(bool _is_default)
    : is_default(_is_default)
    , is_configured(false)
    , is_shut_down(false)
    , cfg_enabled(true)
    , cfg_sample_interval(10000000) // 10 ms
    , cfg_buffer_size(1 << 20)
    , next_sampler_id(0)
    , next_sample_index(0)
    , sampler_head(0)
    , sampler_tail(0)
    , delayed_additions(0)
    , core_rsrv(0)
    , sampling_thread(0)
    , output_fd(-1)
    , flush_requested(false)
    , sampling_start(0)
    , sampling_time(0)
  {
    if(is_default)
      delayed_additions = DefaultSamplerHandler::get_handler().install_default_sampler(this);
  }

  SamplingProfilerImpl::~SamplingProfilerImpl(void)
  {
    if(is_default)
      DefaultSamplerHandler::get_handler().remove_default_sampler(this);

    delete sampling_start;
    delete sampling_time;
  }

  void SamplingProfilerImpl::flush_data(void)
  {
    // flushes need to happen in the sampler thread to avoid races
    flush_requested = true;
  }

  void SamplingProfilerImpl::shutdown(void)
  {
    // set shutdown flag first - prevents any more changes to the sampler list
    //  and flags the sampler thread to clean up
    is_shut_down = true;

    if(!cfg_enabled) {
      assert(sampling_thread == 0);
      assert(core_rsrv == 0);
      assert(sampler_head == 0);
      assert(output_fd == -1);
      return;
    }

    if(sampling_thread) {
      sampling_thread->join();
      delete sampling_thread;
    }
    delete core_rsrv;

    // take (and hold) the mutex, mark that we're shutting down, and then flush all
    //  buffers and destroy all samplers
    AutoHSLLock al(mutex);

    for(std::vector<SampleFile::PacketNewGauge *>::iterator it = new_sampler_infos.begin();
	it != new_sampler_infos.end();
	++it) {
      SampleFile::PacketHeader hdr;
      hdr.packet_type = SampleFile::PacketHeader::PACKET_NEWGAUGE;
      hdr.packet_size = sizeof(SampleFile::PacketNewGauge);
      ssize_t amt = write(output_fd, &hdr, sizeof(hdr));
#ifdef NDEBUG
      (void)amt;
#else
      assert(amt == (ssize_t)sizeof(hdr));
#endif
      amt = write(output_fd, *it, sizeof(SampleFile::PacketNewGauge));
#ifdef NDEBUG
      (void)amt;
#else
      assert(amt == (ssize_t)sizeof(SampleFile::PacketNewGauge));
#endif
      delete *it;
    }
    new_sampler_infos.clear();

    GaugeSampler *sampler = sampler_head;
    sampler_head = 0;
    sampler_tail = 0;

    while(sampler) {
      GaugeSampleBuffer *buffer = sampler->buffer_swap(0);
      if(buffer) {
	if(buffer->compressed_len > 0) buffer->write_data(output_fd);
	delete buffer;
      }
      GaugeSampler *next = sampler->next;
      delete sampler;
      sampler = next;
    }

    close(output_fd);

    log_realmprof.info() << "realm profiler shut down: samples=" << next_sample_index;
  }

  bool SamplingProfilerImpl::parse_profile_pattern(const std::string& s)
  {
    // split string on commas
    size_t spos = 0;
    while(true) {
      size_t cpos = s.find(',', spos);
      if(cpos == std::string::npos) break;
      cfg_patterns.push_back(std::string(s, spos, cpos - spos - 1));
      spos = cpos + 1;
    }
    cfg_patterns.push_back(std::string(s, spos));
    return true;
  }

  void SamplingProfilerImpl::configure_from_cmdline(std::vector<std::string>& cmdline,
						    CoreReservationSet& crs)
  {
    int nodes_profiled = 0;
    std::string logfile = "realmprof_%.dat";

#ifndef NDEBUG
    bool ok =
#endif
              CommandLineParser()
      .add_option_int("-realm:prof", nodes_profiled)
      .add_option_string("-realm:prof_file", logfile)
      .add_option_int("-realm:prof_buffer_size", cfg_buffer_size)
      .add_option_int("-realm:prof_sample_interval", cfg_sample_interval)
      .add_option_method("-realm:prof_pattern", this, &SamplingProfilerImpl::parse_profile_pattern)
      .parse_command_line(cmdline);

    assert(ok);

    cfg_enabled = (my_node_id < nodes_profiled);

    // mark that we're configured and processed deferred additions
    DelayedGaugeAddition *dga = 0;
    {
      AutoHSLLock al(mutex);
      
      is_configured = true;
      dga = delayed_additions;
      delayed_additions = 0;
    }

    long long now = Clock::current_time_in_nanoseconds();
    sampling_start = new ProfilingGauges::AbsoluteGauge<long long>("realm/sampling start", now);
    sampling_time = new ProfilingGauges::EventCounter<long long>("realm/sampling time");

    while(dga) {
      if(cfg_enabled && pattern_match(dga->gauge->name)) {
	int sampler_id = __sync_fetch_and_add(&next_sampler_id, 1);
	SampleFile::PacketNewGauge *info = new SampleFile::PacketNewGauge;
	GaugeSampler *sampler = dga->create_sampler(sampler_id, this, info);
#ifndef NDEBUG
	GaugeSampleBuffer *buffer =
#endif
	                            sampler->buffer_swap(cfg_buffer_size);
	assert(buffer == 0);
	{
	  AutoHSLLock al(mutex);
	  new_sampler_infos.push_back(info);
	  if(sampler_tail)
	    *sampler_tail = sampler;
	  else
	    sampler_head = sampler;
	  sampler_tail = &(sampler->next);
	}
      }
      DelayedGaugeAddition *olddga = dga;
      dga = dga->next;
      delete olddga;
    }

    if(cfg_enabled) {
      CoreReservationParameters params;
      params.set_num_cores(1);
      core_rsrv = new CoreReservation("gauge sampler", crs, params);
      ThreadLaunchParameters tparams;
      sampling_thread = Thread::create_kernel_thread<SamplingProfilerImpl,
						     &SamplingProfilerImpl::sampler_loop>(this,
											  tparams,
											  *core_rsrv);

      // compute a per-node filename
      size_t pct = logfile.find('%');
      if(pct == std::string::npos) {
	// no node number - only ok when profiling a single node
	if(nodes_profiled > 1) {
	  log_realmprof.fatal() << "cannot write profiling data from multiple nodes to common log file '" << logfile << "'";
	  assert(0);
	}
      } else {
	// replace % with node number
	char filename[256];
	sprintf(filename, "%.*s%d%s",
		(int)pct, logfile.c_str(), my_node_id, logfile.c_str() + pct + 1);
	logfile = filename;
      }

      output_fd = open(logfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
      if(output_fd < 0) {
	log_realmprof.fatal() << "could not create/write '" << logfile << "': " << strerror(errno);
	assert(0);
      }

      log_realmprof.info() << "realm profiler enabled: logfile='" << logfile << "' interval=" << cfg_sample_interval << " ns";
    }
  }

  void SamplingProfilerImpl::sampler_loop(void)
  {
    long long last_sample_time = 0;
    while(!is_shut_down) {
      long long wait_time = (cfg_sample_interval -
			     (Clock::current_time_in_nanoseconds() - last_sample_time));
      if(wait_time >= 1000)
	usleep(wait_time / 1000);
      
      GaugeSampler *head = 0;
      GaugeSampler **tail = 0;
      std::vector<SampleFile::PacketNewGauge *> new_infos;
      // take the mutex long enough to sample a consistent head and tail - after
      //  that we can release the lock while we traverse our part of the list
      {
	AutoHSLLock al(mutex);
	head = sampler_head;
	tail = sampler_tail;
	// grab list of new infos atomically as well
	new_infos.swap(new_sampler_infos);
      }
      
      for(std::vector<SampleFile::PacketNewGauge *>::iterator it = new_infos.begin();
	  it != new_infos.end();
	  ++it) {
	SampleFile::PacketHeader hdr;
	hdr.packet_type = SampleFile::PacketHeader::PACKET_NEWGAUGE;
	hdr.packet_size = sizeof(SampleFile::PacketNewGauge);
	ssize_t amt = write(output_fd, &hdr, sizeof(hdr));
#ifdef NDEBUG
	(void)amt;
#else
	assert(amt == (ssize_t)sizeof(hdr));
#endif
	amt = write(output_fd, *it, sizeof(SampleFile::PacketNewGauge));
#ifdef NDEBUG
	(void)amt;
#else
	assert(amt == (ssize_t)sizeof(SampleFile::PacketNewGauge));
#endif
	delete *it;
      }
      new_infos.clear();

      // an empty list means nothing to do
      if(!head)
	continue;

      // as we walk the list, we'll prune out samplers for gauges that have been
      //  removed, which requires remembering the next pointer from the most recent
      //  non-deleted sampler to fix the list
      GaugeSampler *new_head = head;
      GaugeSampler **prev_next = &head;
      // we also defer dealing with the full and/or deleted samplers until after
      //  walking the list to keep the samples as close together as possible
      std::vector<GaugeSampler *> full_samplers, deleted_samplers;
      // pre-preserve some space for these to limit reallocation in traversal
      full_samplers.reserve(16);
      deleted_samplers.reserve(16);
      long long t_start = Clock::current_time_in_nanoseconds();
      last_sample_time = t_start;
      (*sampling_start) = t_start;
      int current_sample_index = __sync_fetch_and_add(&next_sample_index, 1);
      GaugeSampler *sampler = head;
      while(sampler) {
	// take the sampler's mutex while performing the sample to avoid races with
	//  the gauge itself being deleted
	bool full = false;
	bool deleted = false;
	{
	  AutoHSLLock al(sampler->mutex);
	  if(sampler->gauge_exists) {
	    full = sampler->sample_gauge(current_sample_index);
	  } else {
	    deleted = true;
	  }
	}
	if(full) full_samplers.push_back(sampler);
	if(deleted) {
	  deleted_samplers.push_back(sampler);
	  // fix list links
	  if(sampler == new_head)
	    new_head = sampler->next;
	  *prev_next = sampler->next;
	}
	if(!deleted)
	  prev_next = &(sampler->next);
	// stop if we've gotten to the tail we grabbed in the snapshot
	if(&(sampler->next) == tail)
	  break;
	sampler = sampler->next;  // safe since we're deferring deletion
      }
      long long t_end = Clock::current_time_in_nanoseconds();
      (*sampling_time) += (t_end - t_start);

      // now update head and tail if we changed them
      {
	AutoHSLLock al(mutex);
	sampler_head = new_head;  // always safe to update since we know the list wasn't empty
	// only update the tail if the list hasn't grown
	if(sampler_tail == tail) {
	  sampler_tail = prev_next;
	}
      }

      // deal with samplers that were full
      for(std::vector<GaugeSampler *>::const_iterator it = full_samplers.begin();
	  it != full_samplers.end();
	  it++) {
	GaugeSampleBuffer *buffer = (*it)->buffer_swap(cfg_buffer_size);
	assert((buffer != 0) && (buffer->compressed_len > 0));
	buffer->write_data(output_fd);
	delete buffer;
      }

      // samplers whose gauges were deleted need to flush data before deletion
      for(std::vector<GaugeSampler *>::const_iterator it = deleted_samplers.begin();
	  it != deleted_samplers.end();
	  it++) {
	GaugeSampleBuffer *buffer = (*it)->buffer_swap(0);
	if(buffer) {
	  if(buffer->compressed_len > 0) 
	    buffer->write_data(output_fd);
	  delete buffer;
	}
	delete (*it);
      }

      // last, if a flush was requested, go through and dump the data for any other
      //  samplers that weren't full - we're not sampling in this loop, so it's ok to
      //  do the file I/O and memory allocation in the iteration itself
      // we also know no deletion will occur here, so we don't need to remember the tail
      if(flush_requested) {
	flush_requested = false;  // clear the flag now that we're handling it
	GaugeSampler *sampler = sampler_head;
	std::vector<GaugeSampler *>::const_iterator full_it = full_samplers.begin();
	while(sampler) {
	  if((full_it != full_samplers.end()) && (sampler == *full_it)) {
	    // this one was full and we've already handled it
	    ++full_it;
	    continue;
	  }
	  
	  GaugeSampleBuffer *buffer = sampler->buffer_swap(cfg_buffer_size,
							   true /*non-empty only*/);
	  if(buffer) {
	    assert(buffer->compressed_len > 0);
	    buffer->write_data(output_fd);
	    delete buffer;
	  }

	  sampler = sampler->next;
	}
      }
    }
  }
  
  template <typename T>
  GaugeSampler *SamplingProfilerImpl::add_gauge(T *gauge)
  {
    // if we're not configured yes, always defer
    if(!is_configured) {
      AutoHSLLock al(mutex);
      // race conditions are annoying - check again
      if(!is_configured) {
	delayed_additions = new DelayedGaugeAdditionImpl<T>(gauge, delayed_additions);
	return 0;
      }
    }

    // ignore everything if we're not enabled
    if(!cfg_enabled)
      return 0;

    // if we are configured, check against patterns without holding the lock
    if(!pattern_match(gauge->name))
      return 0;
    
    // it matches a pattern, so create the sampler and add it to the list
    int sampler_id = __sync_fetch_and_add(&next_sampler_id, 1);
    SampleFile::PacketNewGauge *info = new SampleFile::PacketNewGauge;
    GaugeSampler *sampler = new GaugeSamplerImpl<T>(sampler_id, this, gauge, info);
#ifndef NDEBUG
    GaugeSampleBuffer *buffer =
#endif
                                sampler->buffer_swap(cfg_buffer_size);
    assert(buffer == 0);
    {
      AutoHSLLock al(mutex);
      // do shutdown check here to avoid race conditions
      if(is_shut_down) {
	delete info;
	delete sampler;
	return 0;
      }
      new_sampler_infos.push_back(info);
      if(sampler_tail)
	*sampler_tail = sampler;
      else
	sampler_head = sampler;
      sampler_tail = &(sampler->next);
    }
    return sampler;
  }

  void SamplingProfilerImpl::remove_gauge(ProfilingGauges::Gauge *gauge,
					  GaugeSampler *sampler)
  {
    // if the shutdown flag is set, the gauge has already been deleted
    AutoHSLLock al(mutex);
    if(is_shut_down)
      return;

    // all we do here is mark that the gauge has been deleted - the sampler will be
    //  cleaned up later (once all data has been written)
    // mutex prevents the very short conflict of deletion while the gauge is being
    //  sampled
    AutoHSLLock al2(sampler->mutex);
    sampler->gauge_exists = false;
  }

  bool SamplingProfilerImpl::pattern_match(const std::string& s) const
  {
    assert(is_configured);
    // TODO: actual pattern matches
    return true;
  }


  namespace ProfilingGauges {

    ////////////////////////////////////////////////////////////////////////
    //
    // class Gauge
    //

    template <typename T>
    /*static*/ void Gauge::add_gauge(T *gauge, SamplingProfiler *profiler)
    {
      if(profiler)
	gauge->sampler = profiler->add_gauge(gauge);
      else
	gauge->sampler = DefaultSamplerHandler::get_handler().add_gauge_to_default_sampler(gauge);
    }

    void Gauge::remove_gauge(void)
    {
      SamplingProfilerImpl *profiler = sampler->profiler;
      if(profiler)
	profiler->remove_gauge(this, sampler);
    }

    // yet another try at forcibly instantiating the right templates
    template void Gauge::add_gauge<AbsoluteGauge<unsigned long long> >(AbsoluteGauge<unsigned long long>*, SamplingProfiler*);
    template void Gauge::add_gauge<AbsoluteGauge<unsigned long> >(AbsoluteGauge<unsigned long>*, SamplingProfiler*);
    template void Gauge::add_gauge<AbsoluteGauge<unsigned> >(AbsoluteGauge<unsigned>*, SamplingProfiler*);
    template void Gauge::add_gauge<AbsoluteRangeGauge<int> >(AbsoluteRangeGauge<int>*, SamplingProfiler*);

  };

}; // namespace Realm
