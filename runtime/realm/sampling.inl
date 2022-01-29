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

// INCLDUED FROM sampling.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/sampling.h"

namespace Realm {

  namespace ProfilingGauges {

    ////////////////////////////////////////////////////////////////////////
    //
    // class Gauge
    //

    inline Gauge::Gauge(const std::string& _name)
      : name(_name)
      , sampler(0) // initialized by subclasses
    {}

    inline Gauge::~Gauge(void)
    {
      if(sampler)
	remove_gauge();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class AbsoluteGauge
    //

    template <typename T>
    inline AbsoluteGauge<T>::AbsoluteGauge(const std::string& _name, T _initval,
					   SamplingProfiler *_profiler /*= 0*/)
      : Gauge(_name)
      , curval(_initval)
    {
      add_gauge(this, _profiler);  // may (eventually) set sampler as a side-effect
    }

    template <typename T>
    AbsoluteGauge<T>& AbsoluteGauge<T>::operator=(const AbsoluteGauge<T>& copy_from)
    {
      curval.store(copy_from.curval.load());
      return *this;
    }

    template <typename T>
    inline AbsoluteGauge<T>::operator T(void) const
    {
      return curval.load();
    }

    template <typename T>
    inline AbsoluteGauge<T>& AbsoluteGauge<T>::operator=(T to_set)
    {
      curval.store(to_set);
      return *this;
    }

    template <typename T>
    inline AbsoluteGauge<T>& AbsoluteGauge<T>::operator+=(T to_add)
    {
      curval.fetch_add(to_add);
      return *this;
    }

    template <typename T>
    inline AbsoluteGauge<T>& AbsoluteGauge<T>::operator-=(T to_sub)
    {
      curval.fetch_sub(to_sub);
      return *this;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class AbsoluteRangeGauge
    //

    template <typename T>
    inline AbsoluteRangeGauge<T>::AbsoluteRangeGauge(const std::string& _name, T _initval,
						     SamplingProfiler *_profiler /*= 0*/)
      : Gauge(_name)
      , curval(_initval)
      , minval(_initval)
      , maxval(_initval)
    {
      add_gauge(this, _profiler);  // may (eventually) set sampler as a side-effect
    }

    template <typename T>
    AbsoluteRangeGauge<T>& AbsoluteRangeGauge<T>::operator=(const AbsoluteRangeGauge<T>& copy_from)
    {
      T newval = copy_from.curval.load();
      curval.store(newval);
      minval.fetch_min(newval);
      maxval.fetch_max(newval);
      return *this;
    }

    template <typename T>
    inline AbsoluteRangeGauge<T>::operator T(void) const
    {
      return curval.load();
    }

    template <typename T>
    inline AbsoluteRangeGauge<T>& AbsoluteRangeGauge<T>::operator=(T to_set)
    {
      curval.store(to_set);
      minval.fetch_min(to_set);
      maxval.fetch_max(to_set);
      return *this;
    }

    template <typename T>
    inline AbsoluteRangeGauge<T>& AbsoluteRangeGauge<T>::operator+=(T to_add)
    {
      T newval = curval.fetch_add(to_add) + to_add;
      if(to_add < 0)
	minval.fetch_min(newval);
      if(to_add > 0)
	maxval.fetch_max(newval);
      return *this;
    }

    template <typename T>
    inline AbsoluteRangeGauge<T>& AbsoluteRangeGauge<T>::operator-=(T to_sub)
    {
      T newval = curval.fetch_sub(to_sub) + to_sub;
      if(to_sub > 0)
	minval.fetch_min(newval);
      if(to_sub < 0)
	maxval.fetch_max(newval);
      return *this;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class EventCounter
    //

    template <typename T>
    inline EventCounter<T>::EventCounter(const std::string& _name,
					 SamplingProfiler *_profiler /*= 0*/)
      : Gauge(_name)
      , events(0)
    {
      add_gauge(this, _profiler);  // may (eventually) set sampler as a side-effect
    }

    template <typename T>
    inline EventCounter<T>& EventCounter<T>::operator+=(T to_add)
    {
      events.fetch_add(to_add);
      return *this;
    }


  }; // namespace ProfilingGauges

}; // namespace Realm
