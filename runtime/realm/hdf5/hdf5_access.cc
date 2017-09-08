/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// HDF5-specific instance layouts and accessors
#include "inst_impl.h"
#include "runtime_impl.h"
#include "mem_impl.h"

#include "realm/hdf5/hdf5_access.h"
#include "realm/deppart/inst_helper.h"
#include "realm/machine.h"

namespace Realm {

  template <int N, typename T>
  /*static*/ Event RegionInstance::create_hdf5_instance(RegionInstance& inst,
							const char *file_name,
							const IndexSpace<N,T>& space,
							const std::vector<FieldID> &field_ids,
							const std::vector<size_t> &field_sizes,
							const std::vector<const char*> &field_files,
							bool read_only,
							const ProfilingRequestSet& prs,
							Event wait_on /*= Event::NO_EVENT*/)
  {
    // look up the local HDF5 memory
    Memory memory = Machine::MemoryQuery(Machine::get_machine())
      .local_address_space()
      .only_kind(Memory::HDF_MEM)
      .first();
    assert(memory.exists());
    
    // construct an instance layout for the new instance
    InstanceLayout<N,T> *layout = new InstanceLayout<N,T>;
    layout->bytes_used = 0;
    layout->alignment_reqd = 0;  // no allocation being made
    layout->space = space;
    layout->piece_lists.resize(field_sizes.size());

    for(size_t i = 0; i < field_sizes.size(); i++) {
      FieldID id = field_ids[i];
      InstanceLayoutGeneric::FieldLayout& fl = layout->fields[id];
      fl.list_idx = i;
      fl.rel_offset = 0;
      fl.size_in_bytes = field_sizes[i];

      // create a single piece (for non-empty index spaces)
      if(!space.empty()) {
	HDF5LayoutPiece<N,T> *hlp = new HDF5LayoutPiece<N,T>;
	hlp->bounds = space.bounds;
	hlp->filename = file_name;
	hlp->dsetname = field_files[i];
	for(int j = 0; j < N; j++)
	  hlp->offset[j] = 0;
	layout->piece_lists[i].pieces.push_back(hlp);
      }
    }

    // and now create the instance using this layout
    return create_instance(inst, memory, layout, prs, wait_on);
  }

#define DOIT(N,T) \
  template Event RegionInstance::create_hdf5_instance<N,T>(RegionInstance&, \
							      const char *, \
							      const IndexSpace<N,T>&, \
							      const std::vector<FieldID>&, \
							      const std::vector<size_t>&, \
							      const std::vector<const char *>&, \
							      bool, \
							      const ProfilingRequestSet&, \
							      Event);
  FOREACH_NT(DOIT)
    
  
  template <int N, typename T>
  /*static*/ Event RegionInstance::create_array_instance_SOA(RegionInstance& inst,
							  const ZIndexSpace<N,T>& space,
                const std::vector<FieldID> &field_ids,
							  const std::vector<size_t> &field_sizes,
                const std::vector<void*> &field_pointers,
							  size_t block_size,
							  const ProfilingRequestSet& reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    Memory memory = Machine::MemoryQuery(Machine::get_machine())
      .local_address_space()
      .only_kind(Memory::SYSTEM_MEM)
      .first();
    assert(memory.exists());
      
    // smoosh hybrid block sizes back to SOA for now
    if(block_size > 1)
      block_size = 0;

    InstanceLayout<N,T> *layout = new InstanceLayout<N,T>;
    layout->bytes_used = 0;
    layout->alignment_reqd = 0;  // no allocation being made
    layout->space = space;
    layout->piece_lists.resize(field_sizes.size());
    
    LocalCPUMemory *m_impl = (LocalCPUMemory *)get_runtime()->get_memory_impl(memory);
    unsigned char* base = (unsigned char*)m_impl->base;
    unsigned char* ptr = NULL;
    for(size_t i = 0; i < field_sizes.size(); i++) {
      FieldID id = field_ids[i];
      InstanceLayoutGeneric::FieldLayout& fl = layout->fields[id];
      fl.list_idx = i;
      fl.rel_offset = 0;
      fl.size_in_bytes = field_sizes[i];

      // create a single piece (for non-empty index spaces)
      if(!space.empty()) {
	      AffineLayoutPiece<N,T> *alp = new AffineLayoutPiece<N,T>;
	      alp->bounds = space.bounds;
        ptr = (unsigned char*)field_pointers[i];
	      alp->offset = (size_t)(ptr - base);
	      size_t stride = field_sizes[i];
	      for(int j = 0; j < N; j++) {
	        alp->strides[j] = stride;
	        stride *= (space.bounds.hi[j] - space.bounds.lo[j] + 1);
	      }
	      layout->piece_lists[i].pieces.push_back(alp);
      }
    }
        
    Event e = create_instance(inst, memory, layout, reqs, wait_on);
    RegionInstanceImpl *inst_impl = get_runtime()->get_instance_impl(inst);
    printf("inst offset %lu\n", inst_impl->metadata.inst_offset);
    return e;
  }

#define DOIT_ARRAY_SOA(N,T) \
  template Event RegionInstance::create_array_instance_SOA<N,T>(RegionInstance&, \
							      const ZIndexSpace<N,T>&, \
                    const std::vector<FieldID>&, \
							      const std::vector<size_t>&, \
							      const std::vector<void *>&, \
							      size_t, \
							      const ProfilingRequestSet&, \
							      Event);
  FOREACH_NT(DOIT_ARRAY_SOA)  
    
  template <int N, typename T>
  /*static*/ Event RegionInstance::create_array_instance_AOS(RegionInstance& inst,
							  const ZIndexSpace<N,T>& space,
                const std::vector<FieldID> &field_ids,
							  const std::vector<size_t> &field_sizes,
                const std::vector<void*> &field_pointers,
							  size_t block_size, unsigned char* aos_base_ptr, size_t aos_stride,
							  const ProfilingRequestSet& reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    Memory memory = Machine::MemoryQuery(Machine::get_machine())
      .local_address_space()
      .only_kind(Memory::SYSTEM_MEM)
      .first();
    assert(memory.exists());
      
    // smoosh hybrid block sizes back to SOA for now
    if(block_size > 1)
      block_size = 0;

    InstanceLayout<N,T> *layout = new InstanceLayout<N,T>;
    layout->bytes_used = 0;
    layout->alignment_reqd = 0;  // no allocation being made
    layout->space = space;
    layout->piece_lists.resize(field_sizes.size());
    
    LocalCPUMemory *m_impl = (LocalCPUMemory *)get_runtime()->get_memory_impl(memory);
    unsigned char* base = (unsigned char*)m_impl->base;
    for(size_t i = 0; i < field_sizes.size(); i++) {
      FieldID id = field_ids[i];
      InstanceLayoutGeneric::FieldLayout& fl = layout->fields[id];
      fl.list_idx = i;
      if (i > 0) {
        fl.rel_offset = (size_t)(((unsigned char*)field_pointers[i]) - ((unsigned char*)field_pointers[i-1]));
      } else {
        fl.rel_offset = (size_t)(((unsigned char*)field_pointers[i]) - aos_base_ptr);
      }
      fl.size_in_bytes = field_sizes[i];

      // create a single piece (for non-empty index spaces)
      if(!space.empty()) {
	      AffineLayoutPiece<N,T> *alp = new AffineLayoutPiece<N,T>;
	      alp->bounds = space.bounds;
	      alp->offset = (size_t)(aos_base_ptr - base);
	      for(int j = 0; j < N; j++) {
	        alp->strides[j] = aos_stride;
	       // stride *= (space.bounds.hi[j] - space.bounds.lo[j] + 1);
	      }
	      layout->piece_lists[i].pieces.push_back(alp);
      }
    }
        
    Event e = create_instance(inst, memory, layout, reqs, wait_on);
    RegionInstanceImpl *inst_impl = get_runtime()->get_instance_impl(inst);
    printf("inst offset %lu\n", inst_impl->metadata.inst_offset);
    return e;
  }

#define DOIT_ARRAY_AOS(N,T) \
  template Event RegionInstance::create_array_instance_AOS<N,T>(RegionInstance&, \
							      const ZIndexSpace<N,T>&, \
                    const std::vector<FieldID>&, \
							      const std::vector<size_t>&, \
							      const std::vector<void *>&, \
							      size_t, unsigned char*, size_t, \
							      const ProfilingRequestSet&, \
							      Event);
  FOREACH_NT(DOIT_ARRAY_AOS)  

}; // namespace Realm
