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

// CUDA-specific instance layouts and accessors

#include "realm/cuda/cuda_access.h"

#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_internal.h"

namespace Realm {

  namespace Cuda {
    extern Logger log_gpu;
  };
  using Cuda::log_gpu;


  // helper routine used by both ExternalCudaMemoryResource and
  //  ExternalCudaArrayResource - chooses an appropriate memory for
  //  external instances on a given gpu (by device id)
  static Memory select_fbmem_for_external_instance(int cuda_device_id)
  {
    Cuda::CudaModule *mod = get_runtime()->get_module<Cuda::CudaModule>("cuda");
    assert(mod);

    Cuda::GPU *gpu = 0;
    for(std::vector<Cuda::GPU *>::const_iterator it = mod->gpus.begin();
        it != mod->gpus.end();
        ++it)
      if((*it)->info->index == cuda_device_id) {
        gpu = *it;
        break;
      }
    if(!gpu) {
      log_gpu.error() << "no gpu with device_id=" << cuda_device_id << " found";
      return Memory::NO_MEMORY;
    }

    // now look through memories that belong to that gpu - prefer a dynamic
    //  FB if exists, but use a normal FB as long is it isn't registered?
    const Node& n = get_runtime()->nodes[Network::my_node_id];
    MemoryImpl *fbmem = 0;
    for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
        it != n.memories.end();
        ++it) {
      const Cuda::CudaDeviceMemoryInfo *spec = (*it)->find_module_specific<Cuda::CudaDeviceMemoryInfo>();
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
  // class ExternalCudaMemoryResource
  //

  ExternalCudaMemoryResource::ExternalCudaMemoryResource()
    : cuda_device_id(-1)
    , base(0)
    , size_in_bytes(0)
    , read_only(false)
  {}

  ExternalCudaMemoryResource::ExternalCudaMemoryResource(int _cuda_device_id,
                                                         uintptr_t _base,
                                                         size_t _size_in_bytes,
                                                         bool _read_only)
    : cuda_device_id(_cuda_device_id)
    , base(_base)
    , size_in_bytes(_size_in_bytes)
    , read_only(_read_only)
  {}

  ExternalCudaMemoryResource::ExternalCudaMemoryResource(int _cuda_device_id,
                                                         void *_base,
                                                         size_t _size_in_bytes)
    : cuda_device_id(_cuda_device_id)
    , base(reinterpret_cast<uintptr_t>(_base))
    , size_in_bytes(_size_in_bytes)
    , read_only(false)
  {}

  ExternalCudaMemoryResource::ExternalCudaMemoryResource(int _cuda_device_id,
                                                         const void *_base,
                                                         size_t _size_in_bytes)
    : cuda_device_id(_cuda_device_id)
    , base(reinterpret_cast<uintptr_t>(_base))
    , size_in_bytes(_size_in_bytes)
    , read_only(true)
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalCudaMemoryResource::suggested_memory() const
  {
    return select_fbmem_for_external_instance(cuda_device_id);
  }

  ExternalInstanceResource *ExternalCudaMemoryResource::clone(void) const
  {
    return new ExternalCudaMemoryResource(cuda_device_id,
                                          base, size_in_bytes, read_only);
  }

  void ExternalCudaMemoryResource::print(std::ostream& os) const
  {
    os << "cudamem(dev=" << cuda_device_id
       << ", base=" << std::hex << base << std::dec
       << ", size=" << size_in_bytes;
    if(read_only)
      os << ", readonly";
    os << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalCudaMemoryResource> ExternalCudaMemoryResource::serdez_subclass;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalCudaArrayResource
  //

  ExternalCudaArrayResource::ExternalCudaArrayResource()
    : cuda_device_id(-1)
    , array(0)
  {}

  ExternalCudaArrayResource::ExternalCudaArrayResource(int _cuda_device_id,
                                                       CUarray_st *_array)
    : cuda_device_id(_cuda_device_id)
    , array(reinterpret_cast<uintptr_t>(_array))
  {}

  ExternalCudaArrayResource::ExternalCudaArrayResource(int _cuda_device_id,
                                                       cudaArray *_array)
    : cuda_device_id(_cuda_device_id)
    , array(reinterpret_cast<uintptr_t>(_array))
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalCudaArrayResource::suggested_memory() const
  {
    return select_fbmem_for_external_instance(cuda_device_id);
  }

  ExternalInstanceResource *ExternalCudaArrayResource::clone(void) const
  {
    return new ExternalCudaArrayResource(cuda_device_id,
                                         reinterpret_cast<CUarray_st *>(array));
  }

  void ExternalCudaArrayResource::print(std::ostream& os) const
  {
    os << "cudaarray(dev=" << cuda_device_id
       << ", array=" << std::hex << array << std::dec << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalCudaArrayResource> ExternalCudaArrayResource::serdez_subclass;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalCudaPinnedHostResource
  //

  ExternalCudaPinnedHostResource::ExternalCudaPinnedHostResource()
  {}

  ExternalCudaPinnedHostResource::ExternalCudaPinnedHostResource(uintptr_t _base,
                                                                 size_t _size_in_bytes,
                                                                 bool _read_only)
    : ExternalMemoryResource(_base, _size_in_bytes, _read_only)
  {}

  ExternalCudaPinnedHostResource::ExternalCudaPinnedHostResource(void *_base,
                                                                 size_t _size_in_bytes)
    : ExternalMemoryResource(_base, _size_in_bytes)
  {}

  ExternalCudaPinnedHostResource::ExternalCudaPinnedHostResource(const void *_base,
                                                                 size_t _size_in_bytes)
    : ExternalMemoryResource(_base, _size_in_bytes)
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalCudaPinnedHostResource::suggested_memory() const
  {
    // do we have a cuda module and does it have a zcmem?
    Cuda::CudaModule *mod = get_runtime()->get_module<Cuda::CudaModule>("cuda");
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

  ExternalInstanceResource *ExternalCudaPinnedHostResource::clone(void) const
  {
    return new ExternalCudaPinnedHostResource(base, size_in_bytes, read_only);
  }

  void ExternalCudaPinnedHostResource::print(std::ostream& os) const
  {
    os << "cudahost(base=" << std::hex << base << std::dec;
    os << ", size=" << size_in_bytes;
    if(read_only)
      os << ", readonly";
    os << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalCudaPinnedHostResource> ExternalCudaPinnedHostResource::serdez_subclass;


}; // namespace Realm
