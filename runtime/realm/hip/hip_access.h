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

#ifndef REALM_HIP_ACCESS_H
#define REALM_HIP_ACCESS_H

#include "realm/inst_layout.h"

namespace Realm {

  class REALM_PUBLIC_API ExternalHipMemoryResource : public ExternalInstanceResource {
  public:
    ExternalHipMemoryResource(int _cuda_device_id, uintptr_t _base,
                               size_t _size_in_bytes, bool _read_only);
    ExternalHipMemoryResource(int _cuda_device_id, void *_base,
                               size_t _size_in_bytes);
    ExternalHipMemoryResource(int _cuda_device_id, const void *_base,
                               size_t _size_in_bytes);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalHipMemoryResource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalHipMemoryResource> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    int hip_device_id;
    uintptr_t base;
    size_t size_in_bytes;
    bool read_only;
  };

  class REALM_PUBLIC_API ExternalHipPinnedHostResource : public ExternalMemoryResource {
  public:
    ExternalHipPinnedHostResource(uintptr_t _base, size_t _size_in_bytes, bool _read_only);
    ExternalHipPinnedHostResource(void *_base, size_t _size_in_bytes);
    ExternalHipPinnedHostResource(const void *_base, size_t _size_in_bytes);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalHipPinnedHostResource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalHipPinnedHostResource> serdez_subclass;

    virtual void print(std::ostream& os) const;
  };

}; // namespace Realm

#include "realm/hip/hip_access.inl"

#endif // ifndef REALM_HIP_ACCESS_H
