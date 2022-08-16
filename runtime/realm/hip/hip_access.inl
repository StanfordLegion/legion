/* Copyright 2021 Stanford University, NVIDIA Corporation
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

// HIP-specific instance layouts and accessors

// NOP but useful for IDE's
#include "realm/hip/hip_access.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalHipMemoryResource

  template <typename S>
  bool ExternalHipMemoryResource::serialize(S& s) const
  {
    return ((s << hip_device_id) &&
            (s << base) &&
            (s << size_in_bytes) &&
	    (s << read_only));
  }

  template <typename S>
  /*static*/ ExternalInstanceResource *ExternalHipMemoryResource::deserialize_new(S& s)
  {
    int hip_device_id;
    uintptr_t base;
    size_t size_in_bytes;
    bool read_only;
    if((s >> hip_device_id) &&
       (s >> base) &&
       (s >> size_in_bytes) &&
       (s >> read_only))
      return new ExternalHipMemoryResource(hip_device_id,
                                           base, size_in_bytes, read_only);
    else
      return 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalHipPinnedHostResource

  template <typename S>
  bool ExternalHipPinnedHostResource::serialize(S& s) const
  {
    return ((s << base) &&
            (s << size_in_bytes) &&
	    (s << read_only));
  }

  template <typename S>
  /*static*/ ExternalInstanceResource *ExternalHipPinnedHostResource::deserialize_new(S& s)
  {
    uintptr_t base;
    size_t size_in_bytes;
    bool read_only;
    if((s >> base) &&
       (s >> size_in_bytes) &&
       (s >> read_only))
      return new ExternalHipPinnedHostResource(base, size_in_bytes, read_only);
    else
      return 0;
  }


}; // namespace Realm
