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

#include "realm/hip/hip_access.h"

#include "realm/hip/hip_module.h"
#include "realm/hip/hip_internal.h"

namespace Realm {

  namespace Hip {
    extern Logger log_gpu;
  };
  using Hip::log_gpu;


  // helper routine used by both ExternalHipMemoryResource and
  //  ExternalHipArrayResource(not implemented) - chooses an appropriate memory for
  //  external instances on a given gpu (by device id)
  static Memory select_fbmem_for_external_instance(int hip_device_id)
  {
    Hip::HipModule *mod = get_runtime()->get_module<Hip::HipModule>("hip");
    assert(mod);

    Hip::GPU *gpu = 0;
    for(std::vector<Hip::GPU *>::const_iterator it = mod->gpus.begin();
        it != mod->gpus.end();
        ++it)
      if((*it)->info->index == hip_device_id) {
        gpu = *it;
        break;
      }
    if(!gpu) {
      log_gpu.error() << "no gpu with device_id=" << hip_device_id << " found";
      return Memory::NO_MEMORY;
    }

    // now look through memories that belong to that gpu - prefer a dynamic
    //  FB if exists, but use a normal FB as long is it isn't registered?
    const Node& n = get_runtime()->nodes[Network::my_node_id];
    MemoryImpl *fbmem = 0;
    for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
        it != n.memories.end();
        ++it) {
      const Hip::HipDeviceMemoryInfo *spec = (*it)->find_module_specific<Hip::HipDeviceMemoryInfo>();
      if(!spec) continue;

      if(spec->gpu != gpu) continue;

      // we can return a dynamic fb as soon as we find it
      if((*it)->get_kind() == Memory::GPU_DYNAMIC_MEM)
        return (*it)->me;

      // a normal fb we sit on until the end
      if((*it)->get_kind() == Memory::GPU_FB_MEM)
        fbmem = *it;
    }

    // don't choose an fbmem that's registered with any network - our
    //  pointer won't be in the bound address range, which can cause
    //  slowness or crashes from dynamic registration attempts
    if(fbmem) {
      if(!fbmem->segment || fbmem->segment->networks.empty()) {
        return fbmem->me;
      } else {
        log_gpu.info() << "memory " << fbmem->me << " is unsuitable for external instances because it is registered with one or more networks";
      }
    }

    return Memory::NO_MEMORY;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalHipMemoryResource
  //

  ExternalHipMemoryResource::ExternalHipMemoryResource()
    : hip_device_id(-1)
    , base(0)
    , size_in_bytes(0)
    , read_only(false)
  {}

  ExternalHipMemoryResource::ExternalHipMemoryResource(int _hip_device_id,
                                                       uintptr_t _base,
                                                       size_t _size_in_bytes,
                                                       bool _read_only)
    : hip_device_id(_hip_device_id)
    , base(_base)
    , size_in_bytes(_size_in_bytes)
    , read_only(_read_only)
  {}

  ExternalHipMemoryResource::ExternalHipMemoryResource(int _hip_device_id,
                                                       void *_base,
                                                       size_t _size_in_bytes)
    : hip_device_id(_hip_device_id)
    , base(reinterpret_cast<uintptr_t>(_base))
    , size_in_bytes(_size_in_bytes)
    , read_only(false)
  {}

  ExternalHipMemoryResource::ExternalHipMemoryResource(int _hip_device_id,
                                                       const void *_base,
                                                       size_t _size_in_bytes)
    : hip_device_id(_hip_device_id)
    , base(reinterpret_cast<uintptr_t>(_base))
    , size_in_bytes(_size_in_bytes)
    , read_only(true)
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalHipMemoryResource::suggested_memory() const
  {
    return select_fbmem_for_external_instance(hip_device_id);
  }

  ExternalInstanceResource *ExternalHipMemoryResource::clone(void) const
  {
    return new ExternalHipMemoryResource(hip_device_id,
                                         base, size_in_bytes, read_only);
  }

  void ExternalHipMemoryResource::print(std::ostream& os) const
  {
    os << "hipmem(dev=" << hip_device_id
       << ", base=" << std::hex << base << std::dec
       << ", size=" << size_in_bytes;
    if(read_only)
      os << ", readonly";
    os << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalHipMemoryResource> ExternalHipMemoryResource::serdez_subclass;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalHipPinnedHostResource
  //

  ExternalHipPinnedHostResource::ExternalHipPinnedHostResource()
  {}

  ExternalHipPinnedHostResource::ExternalHipPinnedHostResource(uintptr_t _base,
                                                               size_t _size_in_bytes,
                                                               bool _read_only)
    : ExternalMemoryResource(_base, _size_in_bytes, _read_only)
  {}

  ExternalHipPinnedHostResource::ExternalHipPinnedHostResource(void *_base,
                                                               size_t _size_in_bytes)
    : ExternalMemoryResource(_base, _size_in_bytes)
  {}

  ExternalHipPinnedHostResource::ExternalHipPinnedHostResource(const void *_base,
                                                               size_t _size_in_bytes)
    : ExternalMemoryResource(_base, _size_in_bytes)
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalHipPinnedHostResource::suggested_memory() const
  {
    // do we have a cuda module and does it have a zcmem?
    Hip::HipModule *mod = get_runtime()->get_module<Hip::HipModule>("hip");
    if(mod && mod->zcmem) {
      // can't use the zcmem if it's registered and we aren't in the segment
      if(mod->zcmem->segment && mod->zcmem->segment->is_registered() &&
         !mod->zcmem->segment->in_segment(base, size_in_bytes)) {
        log_gpu.info() << "memory " << mod->zcmem->me << " is unsuitable for external instances because it is registered with one or more networks";
      } else {
        return mod->zcmem->me;
      }
    }

    // fall through to ExternalMemoryResource, which should suggest a normal
    //  sysmem (i.e. one with no affinity to gpu procs)
    return ExternalMemoryResource::suggested_memory();
  }

  ExternalInstanceResource *ExternalHipPinnedHostResource::clone(void) const
  {
    return new ExternalHipPinnedHostResource(base, size_in_bytes, read_only);
  }

  void ExternalHipPinnedHostResource::print(std::ostream& os) const
  {
    os << "hiphost(base=" << std::hex << base << std::dec;
    os << ", size=" << size_in_bytes;
    if(read_only)
      os << ", readonly";
    os << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalHipPinnedHostResource> ExternalHipPinnedHostResource::serdez_subclass;


}; // namespace Realm
