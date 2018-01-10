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

// INCLDUED FROM rsrv_impl.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/rsrv_impl.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class StaticAccess<T>
  //

    template <typename T>
    StaticAccess<T>::StaticAccess(T* thing_with_data, bool already_valid /*= false*/)
      : data(&thing_with_data->locked_data)
    {
      // if already_valid, just check that data is already valid
      if(already_valid) {
	assert(data->valid);
      } else {
	if(!data->valid) {
	  // get a valid copy of the static data by taking and then releasing
	  //  a shared lock
	  Event e = thing_with_data->lock.acquire(1, false, ReservationImpl::ACQUIRE_BLOCKING);
	  if(!e.has_triggered()) 
            e.wait();
	  thing_with_data->lock.release();
	  assert(data->valid);
	}
      }
    }

    template <typename T>
    SharedAccess<T>::SharedAccess(T* thing_with_data, bool already_held /*= false*/)
      : data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
    {
      // if already_held, just check that it's held (if in debug mode)
      if(already_held) {
	assert(lock->is_locked(1, true));
      } else {
	Event e = thing_with_data->lock.acquire(1, false, ReservationImpl::ACQUIRE_BLOCKING);
	if(!e.has_triggered())
          e.wait();
      }
    }

    template <typename T>
    ExclusiveAccess<T>::ExclusiveAccess(T* thing_with_data, bool already_held /*= false*/)
      : data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
    {
      // if already_held, just check that it's held (if in debug mode)
      if(already_held) {
	assert(lock->is_locked(0, true));
      } else {
	Event e = thing_with_data->lock.acquire(0, true, ReservationImpl::ACQUIRE_BLOCKING);
	if(!e.has_triggered())
          e.wait();
      }
    }

}; // namespace Realm
