/* Copyright 2017 Stanford University, NVIDIA Corporation
 * Copyright 2017 Los Alamos National Laboratory 
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

// IndexSpace implementation for Realm

#include "idx_impl.h"

#include "logging.h"
#include "inst_impl.h"
#include "mem_impl.h"
#include "runtime_impl.h"

#ifdef USE_HDF
#include "realm/hdf5/hdf5_module.h"
#endif

namespace Realm {

  Logger log_meta("meta");
  Logger log_region("region");
  Logger log_copy("copy");

  ////////////////////////////////////////////////////////////////////////
  //
  // class IndexSpace
  //

    /*static*/ const IndexSpace IndexSpace::NO_SPACE = { 0 };
    /*static*/ const Domain Domain::NO_DOMAIN = Domain();

    /*static*/ IndexSpace IndexSpace::create_index_space(size_t num_elmts)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpaceImpl *impl = get_runtime()->local_index_space_free_list->alloc_entry();
      
      impl->init(impl->me, NO_SPACE, num_elmts);
      
      log_meta.info("index space created: id=" IDFMT " num_elmts=%zd",
	       impl->me.id, num_elmts);
      return impl->me;
    }

    /*static*/ IndexSpace IndexSpace::create_index_space(const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpaceImpl *impl = get_runtime()->local_index_space_free_list->alloc_entry();
      
      // TODO: actually decide when to safely consider a subregion frozen
      impl->init(impl->me, NO_SPACE, mask.get_num_elmts(), &mask, true);
      
      log_meta.info("index space created: id=" IDFMT " num_elmts=%zd",
	       impl->me.id, mask.get_num_elmts());
      return impl->me;
    }

    /*static*/ IndexSpace IndexSpace::create_index_space(IndexSpace parent, const ElementMask &mask, bool allocable)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      IndexSpaceImpl *impl = get_runtime()->local_index_space_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).is_idxspace());

      StaticAccess<IndexSpaceImpl> p_data(get_runtime()->get_index_space_impl(parent));

      impl->init(impl->me, parent,
		 p_data->num_elmts, 
		 &mask,
		 !allocable);  // TODO: actually decide when to safely consider a subregion frozen
      
      log_meta.info("index space created: id=" IDFMT " parent=" IDFMT " (num_elmts=%zd)",
	       impl->me.id, parent.id, p_data->num_elmts);
      return impl->me;
    }

    IndexSpaceAllocator IndexSpace::create_allocator(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceAllocatorImpl *a_impl = new IndexSpaceAllocatorImpl(get_runtime()->get_index_space_impl(*this));
      return IndexSpaceAllocator(a_impl);
    }

    void IndexSpace::destroy(Event wait_on) const
    {
      assert(wait_on.has_triggered());
      //assert(0);
    }

    void IndexSpaceAllocator::destroy(void) 
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (impl != NULL)
      {
        delete (IndexSpaceAllocatorImpl *)impl;
        // Avoid double frees
        impl = NULL;
      }
    }

    const ElementMask &IndexSpace::get_valid_mask(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      IndexSpaceImpl *r_impl = get_runtime()->get_index_space_impl(*this);
#ifdef COHERENT_BUT_BROKEN_WAY
      // for now, just hand out the valid mask for the master allocator
      //  and hope it's accessible to the caller
      SharedAccess<IndexSpaceImpl> data(r_impl);
      assert((data->valid_mask_owners >> gasnet_mynode()) & 1);
#else
      if(!r_impl->valid_mask_complete) {
	Event wait_on = r_impl->request_valid_mask();
	
	log_copy.info() << "missing valid mask (" << *this << ") - waiting for " << wait_on;

	wait_on.wait();
      }
#endif
      return *(r_impl->valid_mask);
    }

    Event IndexSpace::create_equal_subspaces(size_t count, size_t granularity,
                                             std::vector<IndexSpace>& subspaces,
                                             bool mutable_results,
                                             Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_equal_subspaces(size_t count, size_t granularity,
                                             std::vector<IndexSpace>& subspaces,
                                             const ProfilingRequestSet &reqs,
                                             bool mutable_results,
                                             Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_weighted_subspaces(size_t count, size_t granularity,
                                                const std::vector<int>& weights,
                                                std::vector<IndexSpace>& subspaces,
                                                bool mutable_results,
                                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_weighted_subspaces(size_t count, size_t granularity,
                                                const std::vector<int>& weights,
                                                std::vector<IndexSpace>& subspaces,
                                                const ProfilingRequestSet &reqs,
                                                bool mutable_results,
                                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::compute_index_spaces(std::vector<BinaryOpDescriptor>& pairs,
                                           bool mutable_results,
					   Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::compute_index_spaces(std::vector<BinaryOpDescriptor>& pairs,
                                           const ProfilingRequestSet &reqs,
                                           bool mutable_results,
					   Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::reduce_index_spaces(IndexSpaceOperation op,
                                          const std::vector<IndexSpace>& spaces,
                                          IndexSpace& result,
                                          bool mutable_results,
                                          IndexSpace parent /*= IndexSpace::NO_SPACE*/,
				          Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    /*static*/
    Event IndexSpace::reduce_index_spaces(IndexSpaceOperation op,
                                          const std::vector<IndexSpace>& spaces,
                                          const ProfilingRequestSet &reqs,
                                          IndexSpace& result,
                                          bool mutable_results,
                                          IndexSpace parent /*= IndexSpace::NO_SPACE*/,
				          Event wait_on /*= Event::NO_EVENT*/)
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_field(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<DomainPoint, IndexSpace>& subspaces,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_field(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<DomainPoint, IndexSpace>& subspaces,
                                const ProfilingRequestSet &reqs,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_image(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<IndexSpace, IndexSpace>& subspaces,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_image(
                                const std::vector<FieldDataDescriptor>& field_data,
                                std::map<IndexSpace, IndexSpace>& subspaces,
                                const ProfilingRequestSet &reqs,
                                bool mutable_results,
                                Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_preimage(
                                 const std::vector<FieldDataDescriptor>& field_data,
                                 std::map<IndexSpace, IndexSpace>& subspaces,
                                 bool mutable_results,
                                 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

    Event IndexSpace::create_subspaces_by_preimage(
                                 const std::vector<FieldDataDescriptor>& field_data,
                                 std::map<IndexSpace, IndexSpace>& subspaces,
                                 const ProfilingRequestSet &reqs,
                                 bool mutable_results,
                                 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: Implement this
      assert(false);
      return Event::NO_EVENT;
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class Domain
  //

    RegionInstance Domain::create_instance(Memory memory,
					   size_t elem_size,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      std::vector<size_t> field_sizes(1);
      field_sizes[0] = elem_size;

      return create_instance(memory, field_sizes, 1, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   size_t elem_size,
                                           const ProfilingRequestSet &reqs,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      std::vector<size_t> field_sizes(1);
      field_sizes[0] = elem_size;

      return create_instance(memory, field_sizes, 1, reqs, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
					   ReductionOpID redop_id) const
    {
      ProfilingRequestSet requests;
      return create_instance(memory, field_sizes, block_size, requests, redop_id);
    }

    RegionInstance Domain::create_instance(Memory memory,
					   const std::vector<size_t> &field_sizes,
					   size_t block_size,
                                           const ProfilingRequestSet &reqs,
					   ReductionOpID redop_id) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

      size_t elem_size = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++)
	elem_size += *it;

      size_t num_elements;
      int linearization_bits[RegionInstanceImpl::MAX_LINEARIZATION_LEN];
      if(get_dim() > 0) {
	// we have a rectangle - figure out its volume and create based on that
	LegionRuntime::Arrays::Rect<1> inst_extent;
	switch(get_dim()) {
	case 1:
	  {
            /*
	    std::vector<LegionRuntime::Layouts::DimKind> kind_vec;
	    std::vector<size_t> size_vec;
	    kind_vec.push_back(LegionRuntime::Layouts::DIM_X);
	    size_vec.push_back(get_rect<1>().dim_size(0));
	    LegionRuntime::Layouts::SplitDimLinearization<1> cl(get_rect<1>().lo, make_point(0), kind_vec, size_vec);
	    */
            LegionRuntime::Arrays::FortranArrayLinearization<1> cl(get_rect<1>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<1>(LegionRuntime::Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<1>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 2:
	  {
            /*
	    std::vector<LegionRuntime::Layouts::DimKind> kind_vec;
	    std::vector<size_t> size_vec;
	    kind_vec.push_back(LegionRuntime::Layouts::DIM_X);
	    kind_vec.push_back(LegionRuntime::Layouts::DIM_Y);
	    size_vec.push_back(get_rect<2>().dim_size(0));
	    size_vec.push_back(get_rect<2>().dim_size(1));
	    LegionRuntime::Layouts::SplitDimLinearization<2> cl(get_rect<2>().lo, make_point(0), kind_vec, size_vec);
	    */
            LegionRuntime::Arrays::FortranArrayLinearization<2> cl(get_rect<2>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<2>(LegionRuntime::Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<2>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 3:
	  {
            /*
	    std::vector<LegionRuntime::Layouts::DimKind> kind_vec;
	    std::vector<size_t> size_vec;
	    kind_vec.push_back(LegionRuntime::Layouts::DIM_X);
	    kind_vec.push_back(LegionRuntime::Layouts::DIM_Y);
	    kind_vec.push_back(LegionRuntime::Layouts::DIM_Z);
	    size_vec.push_back(get_rect<3>().dim_size(0));
	    size_vec.push_back(get_rect<3>().dim_size(1));
	    size_vec.push_back(get_rect<3>().dim_size(2));
	    LegionRuntime:: Layouts::SplitDimLinearization<3> cl(get_rect<3>().lo, make_point(0), kind_vec, size_vec);
	    */
            LegionRuntime::Arrays::FortranArrayLinearization<3> cl(get_rect<3>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<3>(LegionRuntime::Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<3>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	default: assert(0); return RegionInstance::NO_INST;
	}
	num_elements = inst_extent.volume();
	// always at least one element
	if(num_elements <= 0) num_elements = 1;
	//printf("num_elements = %zd\n", num_elements);
      } else {
	IndexSpaceImpl *r = get_runtime()->get_index_space_impl(get_index_space());

	StaticAccess<IndexSpaceImpl> data(r);

#ifdef FULL_SIZE_INSTANCES
	if(data->num_elmts > 0)
	  num_elements = data->last_elmt + 1;
	else
	  num_elements = 1; // always have at least one element
	// linearization is an identity translation
	LegionRuntime::Arrays::Translation<1> inst_offset(0);
	DomainLinearization dl = DomainLinearization::from_mapping<1>(Mapping<1,1>::new_dynamic_mapping(inst_offset));
	dl.serialize(linearization_bits);
#else
	coord_t offset;
	if(data->num_elmts > 0) {
	  num_elements = data->last_elmt - data->first_elmt + 1;
	  offset = data->first_elmt;
	} else {
	  num_elements = 1; // always have at least one element
	  offset = 0;
	}
        // round num_elements up to a multiple of 4 to line things up better with vectors, cache lines, etc.
        if(num_elements & 3) {
          if (block_size == num_elements)
            block_size = (block_size + 3) & ~(size_t)3;
          num_elements = (num_elements + 3) & ~(size_t)3;
        }
	if(block_size > num_elements)
	  block_size = num_elements;

	//printf("CI: %zd %zd %zd\n", data->num_elmts, data->first_elmt, data->last_elmt);

	LegionRuntime::Arrays::Translation<1> inst_offset(-offset);
	DomainLinearization dl = DomainLinearization::from_mapping<1>(LegionRuntime::Arrays::Mapping<1,1>::new_dynamic_mapping(inst_offset));
	dl.serialize(linearization_bits);
#endif
      }

      // for instances with a single element, there's no real difference between AOS and
      //  SOA - force the block size to indicate "full SOA" as it makes the DMA code
      //  use a faster path
      if(field_sizes.size() == 1)
	block_size = num_elements;

#ifdef FORCE_SOA_INSTANCE_LAYOUT
      // the big hammer
      if(block_size != num_elements) {
        log_inst.info("block size changed from %zd to %zd (SOA)",
                      block_size, num_elements);
        block_size = num_elements;
      }
#endif

      if(block_size > 1) {
	size_t leftover = num_elements % block_size;
	if(leftover > 0)
	  num_elements += (block_size - leftover);
      }

      size_t inst_bytes = elem_size * num_elements;

      RegionInstance i = m_impl->create_instance(get_index_space(), linearization_bits, inst_bytes,
						 block_size, elem_size, field_sizes,
						 redop_id,
						 -1 /*list size*/, reqs,
						 RegionInstance::NO_INST);
      log_meta.info("instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd",
	       this->is_id, memory.id, i.id, inst_bytes);
      return i;
    }

    RegionInstance Domain::create_hdf5_instance(const char *file_name,
                                                const std::vector<size_t> &field_sizes,
                                                const std::vector<const char*> &field_files,
                                                bool read_only) const
    {
#ifndef USE_HDF
      // TODO: Implement this
      assert(false);
      return RegionInstance::NO_INST;
#else
      ProfilingRequestSet requests;
      assert(field_sizes.size() == field_files.size());
      return Realm::HDF5::create_hdf5_instance(*this, requests,
					       file_name, field_sizes, field_files,
					       read_only);

#if 0
      Memory memory = Memory::NO_MEMORY;
      Machine machine = Machine::get_machine();
      std::set<Memory> mem;
      machine.get_all_memories(mem);
      for(std::set<Memory>::iterator it = mem.begin(); it != mem.end(); it++) {
        if (it->kind() == Memory::HDF_MEM) {
          memory = *it;
          HDFMemory* hdf_mem = (HDFMemory*) get_runtime()->get_memory_impl(memory);
          if(hdf_mem->kind == MemoryImpl::MKIND_HDF)
            break; /* this is usable, take it */ 
        }
      }
      assert(memory.kind() == Memory::HDF_MEM);
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      HDFMemory* hdf_mem = (HDFMemory*) get_runtime()->get_memory_impl(memory);
      size_t elem_size = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++)
	elem_size += *it;
      
      size_t num_elements;
      int linearization_bits[RegionInstanceImpl::MAX_LINEARIZATION_LEN];
      assert(get_dim() > 0);
      {
        LegionRuntime::Arrays::Rect<1> inst_extent;
        switch(get_dim()) {
	case 1:
	  {
	    LegionRuntime::Arrays::FortranArrayLinearization<1> cl(get_rect<1>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<1>(LegionRuntime::Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<1>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 2:
	  {
	    LegionRuntime::Arrays::FortranArrayLinearization<2> cl(get_rect<2>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<2>(LegionRuntime::Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<2>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 3:
	  {
	    LegionRuntime::Arrays::FortranArrayLinearization<3> cl(get_rect<3>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<3>(LegionRuntime::Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<3>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	default: assert(0);
	}

	num_elements = inst_extent.volume();
      }

      size_t inst_bytes = elem_size * num_elements;
      RegionInstance i = hdf_mem->create_instance(get_index_space(), linearization_bits, inst_bytes, 
                                                  1/*block_size*/, elem_size, field_sizes,
                                                  0 /*redop_id*/, -1/*list_size*/, requests, RegionInstance::NO_INST,
                                                  file_name, field_files, *this, read_only);
      log_meta.info("instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd",
	       this->is_id, memory.id, i.id, inst_bytes);
      return i;
#endif
#endif
    }

    RegionInstance Domain::create_file_instance(const char *file_name,
                                                const std::vector<size_t> &field_sizes,
                                                legion_lowlevel_file_mode_t file_mode) const
    {
      ProfilingRequestSet requests;

      Memory memory = Memory::NO_MEMORY;
      Machine machine = Machine::get_machine();
      std::set<Memory> mem;
      machine.get_all_memories(mem);
      FileMemory *file_mem = NULL;
      for(std::set<Memory>::iterator it = mem.begin(); it != mem.end(); it++) {
        if (it->kind() == Memory::FILE_MEM) {
          memory = *it;
          file_mem = (FileMemory*) get_runtime()->get_memory_impl(memory);
          if(file_mem->kind == MemoryImpl::MKIND_FILE)
            break; /* this is usable, take it */
        }
      }
      assert(file_mem != NULL);
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ID id(memory);
      size_t block_size;

      size_t elem_size = 0;
      for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	  it != field_sizes.end();
	  it++)
	elem_size += *it;

      size_t num_elements;
      int linearization_bits[RegionInstanceImpl::MAX_LINEARIZATION_LEN];
      if(get_dim() > 0) {
	// we have a rectangle - figure out its volume and create based on that
	LegionRuntime::Arrays::Rect<1> inst_extent;
	switch(get_dim()) {
	case 1:
	  {
	    LegionRuntime::Arrays::FortranArrayLinearization<1> cl(get_rect<1>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<1>(LegionRuntime::Arrays::Mapping<1, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<1>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 2:
	  {
	    LegionRuntime::Arrays::FortranArrayLinearization<2> cl(get_rect<2>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<2>(LegionRuntime::Arrays::Mapping<2, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<2>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	case 3:
	  {
	    LegionRuntime::Arrays::FortranArrayLinearization<3> cl(get_rect<3>(), 0);
	    DomainLinearization dl = DomainLinearization::from_mapping<3>(LegionRuntime::Arrays::Mapping<3, 1>::new_dynamic_mapping(cl));
	    inst_extent = cl.image_convex(get_rect<3>());
	    dl.serialize(linearization_bits);
	    break;
	  }

	default: assert(0); return RegionInstance::NO_INST;
	}

	num_elements = inst_extent.volume();
	block_size = num_elements;
	//printf("num_elements = %zd\n", num_elements);
      } else {
	IndexSpaceImpl *r = get_runtime()->get_index_space_impl(get_index_space());

	StaticAccess<IndexSpaceImpl> data(r);
	assert(data->num_elmts > 0);

#ifdef FULL_SIZE_INSTANCES
	num_elements = data->last_elmt + 1;
	// linearization is an identity translation
	LegionRuntime::Arrays::Translation<1> inst_offset(0);
	DomainLinearization dl = DomainLinearization::from_mapping<1>(Mapping<1,1>::new_dynamic_mapping(inst_offset));
	dl.serialize(linearization_bits);
#else
	num_elements = data->last_elmt - data->first_elmt + 1;
	block_size = num_elements;
        // round num_elements up to a multiple of 4 to line things up better with vectors, cache lines, etc.
        if(num_elements & 3) {
          if (block_size == num_elements)
            block_size = (block_size + 3) & ~(size_t)3;
          num_elements = (num_elements + 3) & ~(size_t)3;
        }
	if(block_size > num_elements)
	  block_size = num_elements;

	//printf("CI: %zd %zd %zd\n", data->num_elmts, data->first_elmt, data->last_elmt);

	LegionRuntime::Arrays::Translation<1> inst_offset(-(coord_t)(data->first_elmt));
	DomainLinearization dl = DomainLinearization::from_mapping<1>(LegionRuntime::Arrays::Mapping<1,1>::new_dynamic_mapping(inst_offset));
	dl.serialize(linearization_bits);
#endif
      }

      // for instances with a single element, there's no real difference between AOS and
      //  SOA - force the block size to indicate "full SOA" as it makes the DMA code
      //  use a faster path
      if(field_sizes.size() == 1)
	block_size = num_elements;

#ifdef FORCE_SOA_INSTANCE_LAYOUT
      // the big hammer
      if(block_size != num_elements) {
        log_inst.info("block size changed from %zd to %zd (SOA)",
                      block_size, num_elements);
        block_size = num_elements;
      }
#endif

      if(block_size > 1) {
	size_t leftover = num_elements % block_size;
	if(leftover > 0)
	  num_elements += (block_size - leftover);
      }

      size_t inst_bytes = elem_size * num_elements;

      RegionInstance i = file_mem->create_instance(get_index_space(), linearization_bits, inst_bytes,
						 block_size, elem_size, field_sizes,
						 0 /*reduction op*/, -1 /*list size*/, requests,
						 RegionInstance::NO_INST, file_name, *this, file_mode);
      log_meta.info("instance created: region=" IDFMT " memory=" IDFMT " id=" IDFMT " bytes=%zd",
	       this->is_id, memory.id, i.id, inst_bytes);
      return i;
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class IndexSpaceAllocator
  //

    coord_t IndexSpaceAllocator::alloc(size_t count /*= 1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return ((IndexSpaceAllocatorImpl *)impl)->alloc_elements(count);
    }

    void IndexSpaceAllocator::reserve(coord_t ptr, size_t count /*= 1  */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return ((IndexSpaceAllocatorImpl *)impl)->reserve_elements(ptr, count);
    }

    void IndexSpaceAllocator::free(coord_t ptr, size_t count /*= 1  */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return ((IndexSpaceAllocatorImpl *)impl)->free_elements(ptr, count);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ElementMask
  //

    ElementMask::ElementMask(void)
      : first_element(-1LL), num_elements((size_t)-1LL), memory(Memory::NO_MEMORY), offset(-1LL),
	raw_data(0), first_enabled_elmt(-1LL), last_enabled_elmt(-1LL)
    {
    }

    ElementMask::ElementMask(size_t _num_elements, coord_t _first_element /*= 0*/)
      : first_element(_first_element), num_elements(_num_elements), memory(Memory::NO_MEMORY), offset(-1LL),
        first_enabled_elmt(-1LL), last_enabled_elmt(-1LL)
    {
      // adjust first/num elements to be multiples of 64
      int low_extra = (first_element & 63);
      first_element -= low_extra;
      num_elements += low_extra;
      int high_extra = ((-(coord_t)num_elements) & 63);
      num_elements += high_extra;

      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = (char *)calloc(1, bytes_needed);
      //((ElementMaskImpl *)raw_data)->count = num_elements;
      //((ElementMaskImpl *)raw_data)->offset = first_element;

      assert((first_element & 63) == 0);
      assert((num_elements & 63) == 0);
    }

    ElementMask::ElementMask(const ElementMask &copy_from, 
			     size_t _num_elements, coord_t _first_element /*= -1*/)
    {
      first_element = (_first_element >= 0) ? _first_element : copy_from.first_element;
      num_elements = _num_elements;

      // adjust first/num elements to be multiples of 64
      int low_extra = (first_element & 63);
      first_element -= low_extra;
      num_elements += low_extra;
      int high_extra = ((-(coord_t)num_elements) & 63);
      num_elements += high_extra;

      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      // if we have bounds, make sure they're trimmed to what we actually cover
      if((first_enabled_elmt >= 0) && (first_enabled_elmt < first_element)) {
	first_enabled_elmt = first_element;
      }
      if((last_enabled_elmt >= 0) && (last_enabled_elmt >= (first_element + (coord_t)num_elements))) {
	last_enabled_elmt = first_element + num_elements - 1;
      }
      // figure out the copy offset - must be an integral number of bytes
      ptrdiff_t copy_byte_offset = (first_element - copy_from.first_element);
      assert((copy_from.first_element + (copy_byte_offset << 3)) == first_element);

      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = (char *)calloc(1, bytes_needed);  // sets initial values to 0

      // how much to copy?
      size_t bytes_avail = (ElementMaskImpl::bytes_needed(copy_from.first_element, 
							  copy_from.num_elements) -
			    copy_byte_offset);
      size_t bytes_to_copy = (bytes_needed <= bytes_avail) ? bytes_needed : bytes_avail;

      if(copy_from.raw_data) {
	if(copy_byte_offset >= 0) {
	  memcpy(raw_data, copy_from.raw_data + copy_byte_offset, bytes_to_copy);
	} else {
	  // we start before the input mask, so offset is applied to our pointer
	  memcpy(raw_data + (-copy_byte_offset), copy_from.raw_data, bytes_to_copy);
	}

	// if there were extra bits, make sure they're cleared out
	if(low_extra > 0)
	  *(uint64_t *)raw_data &= ~((~(uint64_t)0) << low_extra);
	if(high_extra > 0)
	  *(uint64_t *)(raw_data + ((num_elements - 64) >> 3)) &= ~((~(uint64_t)0) << high_extra);
      } else {
	if(copy_byte_offset >= 0 ) {
	  get_runtime()->get_memory_impl(copy_from.memory)->get_bytes(copy_from.offset + copy_byte_offset, raw_data, bytes_to_copy);
	} else {
	  // we start before the input mask, so offset is applied to our pointer
	  get_runtime()->get_memory_impl(copy_from.memory)->get_bytes(copy_from.offset, raw_data + (-copy_byte_offset), bytes_to_copy);
	}
	assert(low_extra == 0);
	assert(high_extra == 0);
      }

      assert((first_element & 63) == 0);
      assert((num_elements & 63) == 0);
    }

    ElementMask::ElementMask(const ElementMask &copy_from, bool trim /*= false*/)
    {
      first_element = copy_from.first_element;
      num_elements = copy_from.num_elements;
      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      ptrdiff_t copy_byte_offset = 0;
      if(trim) {
	// trimming from the end is easy - just reduce num_elements - keep to multiples of 64 though
	if(last_enabled_elmt >= 0) {
	  coord_t high_trim = (first_element + num_elements) - (last_enabled_elmt + 1);
	  high_trim &= ~(int)63;
	  num_elements -= high_trim;
	}

	// trimming from the beginning requires stepping by units of 64 so that we can copy uint64_t's
	if(first_enabled_elmt > first_element) {
	  assert(first_enabled_elmt < (first_element + (coord_t)num_elements));
	  coord_t low_trim = (first_enabled_elmt - first_element);
	  low_trim &= ~(int)63;
	  first_element += low_trim;
	  num_elements -= low_trim;
	  copy_byte_offset = low_trim >> 3;  // truncates
	}
      }
      assert(num_elements >= 0);
	
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = (char *)calloc(1, bytes_needed);

      if(copy_from.raw_data) {
	memcpy(raw_data, copy_from.raw_data + copy_byte_offset, bytes_needed);
      } else {
	get_runtime()->get_memory_impl(copy_from.memory)->get_bytes(copy_from.offset + copy_byte_offset, raw_data, bytes_needed);
      }

      assert((first_element & 63) == 0);
      assert((num_elements & 63) == 0);
    }

    ElementMask::~ElementMask(void)
    {
      if (raw_data) {
        free(raw_data);
        raw_data = 0;
      }
    }

    ElementMask& ElementMask::operator=(const ElementMask &rhs)
    {
      first_element = rhs.first_element;
      num_elements = rhs.num_elements;
      first_enabled_elmt = rhs.first_enabled_elmt;
      last_enabled_elmt = rhs.last_enabled_elmt;
      size_t bytes_needed = rhs.raw_size();
      if (raw_data)
        free(raw_data);
      raw_data = (char *)calloc(1, bytes_needed);
      if (rhs.raw_data)
        memcpy(raw_data, rhs.raw_data, bytes_needed);
      else
        get_runtime()->get_memory_impl(rhs.memory)->get_bytes(rhs.offset, raw_data, bytes_needed);
      return *this;
    }

    void ElementMask::init(coord_t _first_element, size_t _num_elements, Memory _memory, coord_t _offset)
    {
      first_element = _first_element;
      num_elements = _num_elements;
      memory = _memory;
      offset = _offset;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = (char *)(get_runtime()->get_memory_impl(memory)->get_direct_ptr(offset, bytes_needed));
    }

    void ElementMask::enable(coord_t start, size_t count /*= 1*/)
    {
      // adjust starting point to our first_element, and make sure span fits
      start -= first_element;
      assert(start >= 0);
      assert((start + count) <= num_elements);

      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("ENABLE %p %d %d %d " IDFMT "\n", raw_data, offset, start, count, impl->bits[0]);
	coord_t pos = start;
	for(size_t i = 0; i < count; i++) {
	  uint64_t *ptr = &(impl->bits[pos >> 6]);
	  *ptr |= (1ULL << (pos & 0x3f));
	  pos++;
	}
	//printf("ENABLED %p %d %d %d " IDFMT "\n", raw_data, offset, start, count, impl->bits[0]);
      } else {
	//printf("ENABLE(2) " IDFMT " %d %d %d\n", memory.id, offset, start, count);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

	coord_t pos = start;
	for(size_t i = 0; i < count; i++) {
	  coord_t ofs = offset + ((pos >> 6) << 3);
	  uint64_t val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("ENABLED(2) %d,  " IDFMT "\n", ofs, val);
	  val |= (1ULL << (pos & 0x3f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
      }

      start += first_element;
      if((first_enabled_elmt < 0) || (start < first_enabled_elmt))
	first_enabled_elmt = start;

      if((last_enabled_elmt < 0) || ((start+(coord_t)count-1) > last_enabled_elmt))
	last_enabled_elmt = start + count - 1;
    }

    void ElementMask::disable(coord_t start, size_t count /*= 1*/)
    {
      // adjust starting point to our first_element, and make sure span fits
      start -= first_element;
      assert(start >= 0);
      assert((start + count) <= num_elements);

      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	coord_t pos = start;
	for(size_t i = 0; i < count; i++) {
	  uint64_t *ptr = &(impl->bits[pos >> 6]);
	  *ptr &= ~(1ULL << (pos & 0x3f));
	  pos++;
	}
      } else {
	//printf("DISABLE(2) " IDFMT " %d %d %d\n", memory.id, offset, start, count);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

	coord_t pos = start;
	for(size_t i = 0; i < count; i++) {
	  coord_t ofs = offset + ((pos >> 6) << 3);
	  uint64_t val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("DISABLED(2) %d,  " IDFMT "\n", ofs, val);
	  val &= ~(1ULL << (pos & 0x3f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
      }

      start += first_element;
      if(start == first_enabled_elmt) {
	//printf("pushing first: %d -> %d\n", first_enabled_elmt, first_enabled_elmt+1);
	first_enabled_elmt = find_enabled(1, first_enabled_elmt + count);
	// if we didn't find anything we just cleared the last enabled bit too
	if(first_enabled_elmt == -1LL)
	  last_enabled_elmt = -1LL;
      }
    }

    coord_t ElementMask::find_enabled(size_t count /*= 1 */, coord_t start /*= 0*/) const
    {
      if(start == 0)
	start = first_enabled_elmt - first_element;
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("FIND_ENABLED %p %d %d " IDFMT "\n", raw_data, first_element, count, impl->bits[0]);
	for(size_t pos = start; pos <= num_elements - count; pos++) {
	  size_t run = 0;
	  while(1) {
	    uint64_t bit = ((impl->bits[pos >> 6] >> (pos & 0x3f))) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return (pos - run) + first_element;
	  }
	}
      } else {
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
	//printf("FIND_ENABLED(2) " IDFMT " %d %d %d\n", memory.id, offset, first_element, count);
	for(size_t pos = start; pos <= num_elements - count; pos++) {
	  size_t run = 0;
	  while(1) {
	    coord_t ofs = offset + ((pos >> 6) << 3);
	    uint64_t val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    uint64_t bit = (val >> (pos & 0x3f)) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return (pos - run) + first_element;
	  }
	}
      }
      return -1LL;
    }

    coord_t ElementMask::find_disabled(size_t count /*= 1 */, coord_t start /*= 0*/) const
    {
      if((start == 0) && (first_enabled_elmt > 0))
	start = first_enabled_elmt - first_element;
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	for(size_t pos = start; pos <= num_elements - count; pos++) {
	  size_t run = 0;
	  while(1) {
	    uint64_t bit = ((impl->bits[pos >> 6] >> (pos & 0x3f))) & 1;
	    if(bit != 0) break;
	    pos++; run++;
	    if(run >= count) return (pos - run) + first_element;
	  }
	}
      } else {
	assert(0);
      }
      return -1LL;
    }

    size_t ElementMask::raw_size(void) const
    {
      return ElementMaskImpl::bytes_needed(offset, num_elements);
    }

    const void *ElementMask::get_raw(void) const
    {
      return raw_data;
    }

    void ElementMask::set_raw(const void *data)
    {
      assert(0);
    }

    bool ElementMask::is_set(coord_t ptr) const
    {
      // adjust starting point to our first_element, and make sure span fits
      ptr -= first_element;
      if((ptr < 0) || (ptr >= (coord_t)num_elements))
	return false;

      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;

	coord_t pos = ptr;// - first_element;
	uint64_t val = (impl->bits[pos >> 6]);
        uint64_t bit = ((val) >> (pos & 0x3f));
        return ((bit & 1) != 0);
      } else {
        assert(0);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);

	coord_t pos = ptr - first_element;
	coord_t ofs = offset + ((pos >> 6) << 3);
	uint64_t val;
	m_impl->get_bytes(ofs, &val, sizeof(val));
        uint64_t bit = ((val) >> (pos & 0x3f));
        return ((bit & 1) != 0);
      }
    }

    size_t ElementMask::pop_count(bool enabled) const
    {
      size_t count = 0;
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        size_t max_full = (num_elements >> 6);
        bool remainder = (num_elements % 64) != 0;
        for (size_t index = 0; index < max_full; index++)
          count += __builtin_popcountll(impl->bits[index]);
        if (remainder)
          count += __builtin_popcountll(impl->bits[max_full]);
        if (!enabled)
          count = num_elements - count;
      } else {
        // TODO: implement this
        assert(0);
      }
      return count;
    }

    bool ElementMask::operator!(void) const
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        const size_t max_full = ((num_elements+63) >> 6);
        for (size_t index = 0; index < max_full; index++) {
          if (impl->bits[index])
            return false;
        }
      } else {
        // TODO: implement this
        assert(0);
      }
      return true;
    }

    bool ElementMask::operator==(const ElementMask &other) const
    {
      // TODO: allow equality between trimmed and untrimmed ElementMasks
      // first check - do the enabled ranges match?
      if(first_enabled_elmt != other.first_enabled_elmt) return false;
      if(last_enabled_elmt != other.last_enabled_elmt) return false;

      // support bitwise operations between ElementMasks with different sizes/starts,
      //  but only if the bits line up conveniently
      assert((first_element & 63) == (other.first_element & 63));

      // empty/singleton masks are also easy
      if(first_enabled_elmt >= last_enabled_elmt)
	return true;

      if(raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	if(other.raw_data != 0) {
	  ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
	  // we're going to walk over two bit arrays in lockstep
	  const uint64_t *bits = impl->bits + ((first_enabled_elmt - first_element) >> 6);
	  const uint64_t *other_bits = other_impl->bits + ((other.first_enabled_elmt - other.first_element) >> 6);
	  // relative positions are relative to the first bit in the words we selected above
	  coord_t rel_pos = (first_enabled_elmt - first_element) & 63;
	  coord_t count = rel_pos + (last_enabled_elmt - first_enabled_elmt + 1);

	  // handle the first word specially
	  if(rel_pos > 0) {
	    int shft = rel_pos & 63;
	    uint64_t v = *bits++ ^ *other_bits++;
	    v &= (~(uint64_t)0) << shft;
	    // subcase if we don't extend outside first word
	    if(count < 64)
	      v &= ~((~(uint64_t)0) << count);
	    if(v != 0)
	      return false;
	  }

	  // whole words next
	  while(count >= 64) {
	    if(*bits++ != *other_bits++)
	      return false;
	    count -= 64;
	  }

	  // trailing bits
	  if(count > 0) {
	    uint64_t v = *bits ^ *other_bits;
	    v &= ~((~(uint64_t)0) << count);
	    if(v != 0)
	      return false;
	  }

	  return true;
        } else {
          // TODO: Implement this
          assert(false);
        }
      } else {
        // TODO: Implement this
        assert(false);
      }
      return true;
    }

    bool ElementMask::operator!=(const ElementMask &other) const
    {
      return !((*this) == other);
    }

    ElementMask ElementMask::operator|(const ElementMask &other) const
    {
      // result needs to be big enough to hold both ranges
      coord_t new_first = std::min(first_element, other.first_element);
      size_t new_count = (std::max(first_element + num_elements,
				   other.first_element + other.num_elements) -
		                   new_first);

      ElementMask result(new_count, new_first);
      result |= *this;
      result |= other;
      return result;
    }

    ElementMask ElementMask::operator&(const ElementMask &other) const
    {
      // result sized the same as lhs
      ElementMask result(*this);
      result &= other;
      return result;
    }

    ElementMask ElementMask::operator-(const ElementMask &other) const
    {
      // result sized the same as lhs
      ElementMask result(*this);
      result -= other;
      return result;
    }

    void ElementMask::recalc_first_last_enabled(void)
    {
      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;

	size_t count = num_elements >> 6;

	// find first word that isn't 0
	first_enabled_elmt = -1LL;
	for(size_t i = 0; i < count; i++) {
	  uint64_t v = impl->bits[i];
	  if(v != 0) {
	    coord_t ofs = __builtin_ctzl(v);
	    first_enabled_elmt = first_element + (i << 6) + ofs;
	    //printf("FOUNDFIRST: %lx %d %d %d\n", v, i, ofs, first_enabled_elmt);
	    break;
	  }
	}

	// find last word that isn't 0 - no search if the first search failed
	last_enabled_elmt = -1LL;
	if(first_enabled_elmt >= 0)  {
	  for(coord_t i = (coord_t)count - 1; i >= 0; i--) {
	    uint64_t v = impl->bits[i];
	    if(v != 0) {
	      coord_t ofs = __builtin_clzl(v);
	      last_enabled_elmt = first_element + (i << 6) + (63 - ofs);
	      //printf("FOUNDLAST: %lx %d %d %d\n", v, i, ofs, last_enabled_elmt);
	      break;
	    }
	  }
	}
      } else {
        // TODO: implement this
        assert(0);
      }
    }

    ElementMask& ElementMask::operator|=(const ElementMask &other)
    {
      // an empty rhs is trivial
      if(other.first_enabled_elmt == -1LL)
	return *this;

      // support bitwise operations between ElementMasks with different sizes/starts,
      //  but only if the bits line up conveniently
      assert((first_element & 63) == (other.first_element & 63));

      // if the rhs's range is larger than ours, die - we can't fit the result
      coord_t abs_start = ((other.first_enabled_elmt >= 0) ?
                         other.first_enabled_elmt :
                         other.first_element);
      coord_t abs_end = ((other.last_enabled_elmt >= 0) ?
                       (other.last_enabled_elmt + 1) :
                       (other.first_element + other.num_elements));
      assert(abs_start >= first_element);
      assert(abs_end <= (first_element + (coord_t)num_elements));

      // no overlap case is simple
      if(abs_start >= abs_end)
	return *this;

      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
	  // we're going to walk over two bit arrays in lockstep
	  uint64_t *bits = impl->bits + ((abs_start - first_element) >> 6);
	  const uint64_t *other_bits = other_impl->bits + ((abs_start - other.first_element) >> 6);
	  // relative positions are relative to the first bit in the words we selected above
	  coord_t rel_pos = (abs_start - first_element) & 63;
	  coord_t count = rel_pos + (abs_end - abs_start);

	  // handle the first word specially
	  if(rel_pos > 0) {
	    coord_t shft = rel_pos & 63;
	    uint64_t v = *other_bits++;
	    v &= (~(uint64_t)0) << shft;
	    // subcase if we don't extend outside first word
	    if(count < 64)
	      v &= ~((~(uint64_t)0) << count);
	    *bits++ |= v;
	    count -= 64;
	  }

	  // whole words next
	  while(count >= 64) {
	    *bits++ |= (*other_bits++);
	    count -= 64;
	  }

	  // trailing bits
	  if(count > 0) {
	    uint64_t v = *other_bits;
	    v &= ~((~(uint64_t)0) << count);
	    *bits |= v;
	  }
        } else {
          // TODO: implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }

      recalc_first_last_enabled();

      return *this;
    }

    ElementMask& ElementMask::operator&=(const ElementMask &other)
    {
      // support bitwise operations between ElementMasks with different sizes/starts,
      //  but only if the bits line up conveniently
      assert((first_element & 63) == (other.first_element & 63));

      // we need to cover our entire range of bits, either and'ing with the other's mask or
      //  clearing them out
      coord_t abs_start = first_element;
      coord_t abs_end = first_element + num_elements;

      // no overlap case is simple
      if(abs_start >= abs_end)
	return *this;

      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
	  // we're going to walk over two bit arrays in lockstep
	  uint64_t *bits = impl->bits + ((abs_start - first_element) >> 6);

	  // 'other_bits' may be out of range if lhs is bigger than rhs - keep the valid start
	  //  and end too and supply zeroes instead of reading from the pointer when it's out of range
	  const uint64_t *other_bits = other_impl->bits + ((abs_start - other.first_element) >> 6);
	  const uint64_t *other_bits_valid_start = other_impl->bits;
	  const uint64_t *other_bits_valid_end = other_impl->bits + (other.num_elements >> 6);

	  // relative positions are relative to the first bit in the words we selected above
	  coord_t rel_pos = (abs_start - first_element) & 63;
	  coord_t count = rel_pos + (abs_end - abs_start);

	  // handle the first word specially
	  if(rel_pos > 0) {
	    if((other_bits >= other_bits_valid_start) && (other_bits < other_bits_valid_end)) {
	      coord_t shft = rel_pos & 63;
	      uint64_t v = *other_bits;
	      v &= (~(uint64_t)0) << shft;
	      // subcase if we don't extend outside first word
	      if(count < 64)
		v &= ~((~(uint64_t)0) << count);
	      *bits++ &= v;
	    } else
	      *bits++ = 0;
	    other_bits++;
	    count -= 64;
	  }

	  // whole words next
	  while(count >= 64) {
	    if((other_bits >= other_bits_valid_start) && (other_bits < other_bits_valid_end))
	      *bits++ &= *other_bits;
	    else
	      *bits++ = 0;
	    other_bits++;
	    count -= 64;
	  }

	  // trailing bits
	  if(count > 0) {
	    if((other_bits >= other_bits_valid_start) && (other_bits < other_bits_valid_end)) {
	      uint64_t v = *other_bits;
	      v &= ~((~(uint64_t)0) << count);
	      *bits &= v;
	    } else
	      *bits = 0;
	  }
	} else {
          // TODO: implement this
          assert(0);
	}
      } else {
        // TODO: implement this
        assert(0);
      }

      recalc_first_last_enabled();

      return *this;
    }

    ElementMask& ElementMask::operator-=(const ElementMask &other)
    {
      // an empty rhs is trivial
      if(other.first_enabled_elmt == -1LL)
	return *this;

      // support bitwise operations between ElementMasks with different sizes/starts,
      //  but only if the bits line up conveniently
      assert((first_element & 63) == (other.first_element & 63));

      // determine the range of bits we're going to cover - trim to both masks
      coord_t abs_start = std::max(first_element, other.first_element);
      coord_t abs_end = std::min(first_element + num_elements,
			       other.first_element + other.num_elements);
      // no overlap case is simple
      if(abs_start >= abs_end)
	return *this;

      if (raw_data != 0) {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *other_impl = (ElementMaskImpl *)other.raw_data;
	  // we're going to walk over two bit arrays in lockstep
	  uint64_t *bits = impl->bits + ((abs_start - first_element) >> 6);
	  const uint64_t *other_bits = other_impl->bits + ((abs_start - other.first_element) >> 6);
	  // relative positions are relative to the first bit in the words we selected above
	  coord_t rel_pos = (abs_start - first_element) & 63;
	  coord_t count = rel_pos + (abs_end - abs_start);

	  // handle the first word specially
	  if(rel_pos > 0) {
	    coord_t shft = rel_pos & 63;
	    uint64_t v = *other_bits++;
	    v &= (~(uint64_t)0) << shft;
	    // subcase if we don't extend outside first word
	    if(count < 64)
	      v &= ~((~(uint64_t)0) << count);
	    *bits++ &= ~v;
	    count -= 64;
	  }

	  // whole words next
	  while(count >= 64) {
	    *bits++ &= ~(*other_bits++);
	    count -= 64;
	  }

	  // trailing bits
	  if(count > 0) {
	    uint64_t v = *other_bits;
	    v &= ~((~(uint64_t)0) << count);
	    *bits &= ~v;
	  }
        } else {
          // TODO: implement this
          assert(0);
        }
      } else {
        // TODO: implement this
        assert(0);
      }

      recalc_first_last_enabled();

      return *this;
    }

    ElementMask::OverlapResult ElementMask::overlaps_with(const ElementMask& other,
							  coord_t max_effort /*= -1*/) const
    {
      // do the spans clearly not interact?
      coord_t first1 = (first_enabled_elmt >= 0) ? first_enabled_elmt : first_element;
      coord_t last1 = (last_enabled_elmt >= 0) ? last_enabled_elmt : (first_element + num_elements - 1);
      coord_t first2 = (other.first_enabled_elmt >= 0) ? other.first_enabled_elmt : other.first_element;
      coord_t last2 = (other.last_enabled_elmt >= 0) ? other.last_enabled_elmt : (other.first_element + other.num_elements - 1);

      coord_t first = std::max(first1, first2);
      coord_t last = std::min(last1, last2);

      if(first > last)
	return ElementMask::OVERLAP_NO;
	
      if (raw_data != 0) {
        ElementMaskImpl *i1 = (ElementMaskImpl *)raw_data;
        if (other.raw_data != 0) {
          ElementMaskImpl *i2 = (ElementMaskImpl *)(other.raw_data);

	  // if different in the first elements is a multiple of 64, we can do 64 bit compares
	  if(((first_element - other.first_element) & 63) == 0) {
	    const uint64_t *bits1 = i1->bits + ((first - first_element) >> 6);
	    const uint64_t *bits2 = i2->bits + ((first - other.first_element) >> 6);
	    size_t count = (last - (first & ~63ULL) + 64) >> 6;
	    for(size_t i = 0; i < count; i++)
	      if((*bits1++ & *bits2++) != 0)
		return ElementMask::OVERLAP_YES;
	    return ElementMask::OVERLAP_NO;
	  } else {
	    // fall back to byte-wise compare
	    assert(((first_element - other.first_element) & 7) == 0);
	    const unsigned char *bits1 = ((const unsigned char *)(i1->bits)) + ((first - first_element) >> 3);
	    const unsigned char *bits2 = ((const unsigned char *)(i2->bits)) + ((first - other.first_element) >> 3);
	    size_t count = (last - (first & ~7ULL) + 8) >> 3;
	    for(size_t i = 0; i < count; i++)
	      if((*bits1++ & *bits2++) != 0)
		return ElementMask::OVERLAP_YES;
	    return ElementMask::OVERLAP_NO;
	  }
        } else {
          return ElementMask::OVERLAP_MAYBE;
        }
      } else {
        return ElementMask::OVERLAP_MAYBE;
      }
    }

    ElementMask::Enumerator *ElementMask::enumerate_enabled(coord_t start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 1);
    }

    ElementMask::Enumerator *ElementMask::enumerate_disabled(coord_t start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 0);
    }

    ElementMask::Enumerator::Enumerator(const ElementMask& _mask, coord_t _start, int _polarity)
      : mask(_mask), pos(_start), polarity(_polarity) {}

    ElementMask::Enumerator::~Enumerator(void) {}

    bool ElementMask::Enumerator::get_next(coord_t &position, size_t &length)
    {
      if(mask.raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)(mask.raw_data);

	// are we already off the end?
	if(pos >= (mask.first_element + (coord_t)mask.num_elements))
	  return false;

	// can never start before the beginning of the mask
	if(pos < mask.first_element)
	  pos = mask.first_element;

        // if our current pos is below the first known-set element, skip to there
        if((mask.first_enabled_elmt > 0) && (pos < mask.first_enabled_elmt))
          pos = mask.first_enabled_elmt;

	// fetch first value and see if we have any bits set
	coord_t rel_pos = pos - mask.first_element;
	coord_t idx = rel_pos >> 6;
	uint64_t bits = impl->bits[idx];
	if(!polarity) bits = ~bits;

	// for the first one, we may have bits to ignore at the start
	if(rel_pos & 0x3f)
	  bits &= ~((1ULL << (rel_pos & 0x3f)) - 1);

	// skip over words that are all zeros, and try to ignore trailing zeros completely
        coord_t stop_at = mask.num_elements;
        if(mask.last_enabled_elmt >= 0)
          stop_at = mask.last_enabled_elmt - mask.first_element + 1; // relative stop location
	while(!bits) {
	  idx++;
	  if((idx << 6) >= stop_at) {
	    pos = mask.num_elements + mask.first_element; // so we don't scan again
	    return false;
	  }
	  bits = impl->bits[idx];
	  if(!polarity) bits = ~bits;
	}

	// if we get here, we've got at least one good bit
	coord_t extra = __builtin_ctzll(bits);
	assert(extra < 64);
	position = mask.first_element + (idx << 6) + extra;
	
	// now we're going to turn it around and scan ones
	if(extra)
	  bits |= ((1ULL << extra) - 1);
	bits = ~bits;

	while(!bits) {
	  idx++;
	  // did our 1's take us right to the end?
	  if((idx << 6) >= stop_at) {
	    pos = mask.num_elements + mask.first_element; // so we don't scan again
	    length = mask.first_element + stop_at - position;
	    return true;
	  }
	  bits = ~impl->bits[idx]; // note the inversion
	  if(!polarity) bits = ~bits;
	}

	// if we get here, we got to the end of the 1's
	coord_t extra2 = __builtin_ctzll(bits);
	// if we run off the end, that's bad, but don't get confused by a few extra bits in the last word
	assert((idx << 6) <= (coord_t)mask.num_elements);
	rel_pos = std::min((idx << 6) + extra2, (coord_t)mask.num_elements);
	pos = mask.first_element + rel_pos;
	length = pos - position;
	return true;
      } else {
	assert(0);
	MemoryImpl *m_impl = get_runtime()->get_memory_impl(mask.memory);

	// scan until we find a bit set with the right polarity
	while(pos < (coord_t)mask.num_elements) {
	  coord_t ofs = mask.offset + ((pos >> 5) << 2);
	  size_t val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  int bit = ((val >> (pos & 0x1f))) & 1;
	  if(bit != polarity) {
	    pos++;
	    continue;
	  }

	  // ok, found one bit with the right polarity - now see how many
	  //  we have in a row
	  position = pos++;
	  while(pos < (coord_t)mask.num_elements) {
	    coord_t ofs = mask.offset + ((pos >> 5) << 2);
	    size_t val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    int bit = ((val >> (pos & 0x1f))) & 1;
	    if(bit != polarity) break;
	    pos++;
	  }
	  // we get here either because we found the end of the run or we 
	  //  hit the end of the mask
	  length = pos - position;
	  return true;
	}

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      }
    }

    bool ElementMask::Enumerator::peek_next(coord_t &position, size_t &length)
    {
      coord_t old_pos = pos;
      bool ret = get_next(position, length);
      pos = old_pos;
      return ret;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IndexSpaceImpl
  //

    IndexSpaceImpl::IndexSpaceImpl(void)
    {
      init(IndexSpace::NO_SPACE, -1);
    }

    void IndexSpaceImpl::init(IndexSpace _me, unsigned _init_owner)
    {
      assert(!_me.exists() || (_init_owner == ID(_me).idxspace.owner_node));

      me = _me;
      locked_data.valid = false;
      lock.init(ID(me).convert<Reservation>(), ID(me).idxspace.owner_node);
      lock.in_use = true;
      lock.set_local_data(&locked_data);
      valid_mask = 0;
      valid_mask_complete = false;
      valid_mask_event = Event::NO_EVENT;
    }

    void IndexSpaceImpl::init(IndexSpace _me, IndexSpace _parent,
				size_t _num_elmts,
				const ElementMask *_initial_valid_mask /*= 0*/, bool _frozen /*= false*/)
    {
      me = _me;
      locked_data.valid = true;
      locked_data.parent = _parent;
      locked_data.frozen = _frozen;
      locked_data.num_elmts = _num_elmts;
      locked_data.valid_mask_owners = (1ULL << gasnet_mynode());
      locked_data.avail_mask_owner = gasnet_mynode();
      valid_mask = (_initial_valid_mask?
		    new ElementMask(*_initial_valid_mask, _frozen) : // trim if frozen
		    new ElementMask(_num_elmts));
      valid_mask_complete = true;
      valid_mask_event = Event::NO_EVENT;
      if(_frozen) {
	avail_mask = 0;
	locked_data.first_elmt = valid_mask->first_enabled();
	locked_data.last_elmt = valid_mask->last_enabled();
	log_region.info("subregion " IDFMT " (of " IDFMT ") restricted to [%zd,%zd]",
			me.id, _parent.id, locked_data.first_elmt,
			locked_data.last_elmt);
      } else {
	avail_mask = new ElementMask(_num_elmts);
	if(_parent == IndexSpace::NO_SPACE) {
	  avail_mask->enable(0, _num_elmts);
	  locked_data.first_elmt = 0;
	  locked_data.last_elmt = _num_elmts - 1;
	} else {
	  StaticAccess<IndexSpaceImpl> pdata(get_runtime()->get_index_space_impl(_parent));
	  locked_data.first_elmt = pdata->first_elmt;
	  locked_data.last_elmt = pdata->last_elmt;
	}
      }
      lock.init(ID(me).convert<Reservation>(), ID(me).idxspace.owner_node);
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    IndexSpaceImpl::~IndexSpaceImpl(void)
    {
      delete valid_mask;
    }

    bool IndexSpaceImpl::is_parent_of(IndexSpace other)
    {
      while(other != IndexSpace::NO_SPACE) {
	if(other == me) return true;
	IndexSpaceImpl *other_impl = get_runtime()->get_index_space_impl(other);
	other = StaticAccess<IndexSpaceImpl>(other_impl)->parent;
      }
      return false;
    }

    Event IndexSpaceImpl::request_valid_mask(void)
    {
      size_t num_elmts = StaticAccess<IndexSpaceImpl>(this)->num_elmts;
      int valid_mask_owner = -1;
      
      Event e;
      {
	AutoHSLLock a(valid_mask_mutex);
	
	if(valid_mask != 0) {
	  // if the mask exists, we've already requested it, so just provide
	  //  the event that we have
          return valid_mask_event;
	}
	
	valid_mask = new ElementMask(num_elmts);
	valid_mask_owner = ID(me).idxspace.owner_node; // a good guess?
	valid_mask_count = (valid_mask->raw_size() + 2047) >> 11;
	valid_mask_complete = false;
	valid_mask_event = GenEventImpl::create_genevent()->current_event();
        e = valid_mask_event;
      }

      ValidMaskRequestMessage::send_request(valid_mask_owner, me);

      return e;
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class IndexSpaceAllocatorImpl
  //

    IndexSpaceAllocatorImpl::IndexSpaceAllocatorImpl(IndexSpaceImpl *_is_impl)
      : is_impl(_is_impl)
    {
    }

    IndexSpaceAllocatorImpl::~IndexSpaceAllocatorImpl(void)
    {
    }

    coord_t IndexSpaceAllocatorImpl::alloc_elements(size_t count /*= 1 */)
    {
      SharedAccess<IndexSpaceImpl> is_data(is_impl);
      assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
      coord_t start = is_impl->valid_mask->find_disabled(count);
      assert(start >= 0);

      reserve_elements(start, count);

      return start;
    }

    void IndexSpaceAllocatorImpl::reserve_elements(coord_t ptr, size_t count /*= 1 */)
    {
      // for now, do updates of valid masks immediately
      IndexSpaceImpl *impl = is_impl;
      while(1) {
	SharedAccess<IndexSpaceImpl> is_data(impl);
	assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
	is_impl->valid_mask->enable(ptr, count);
	IndexSpace is = is_data->parent;
	if(is == IndexSpace::NO_SPACE) break;
	impl = get_runtime()->get_index_space_impl(is);
      }
    }

    void IndexSpaceAllocatorImpl::free_elements(coord_t ptr, size_t count /*= 1*/)
    {
      // for now, do updates of valid masks immediately
      IndexSpaceImpl *impl = is_impl;
      while(1) {
	SharedAccess<IndexSpaceImpl> is_data(impl);
	assert((is_data->valid_mask_owners >> gasnet_mynode()) & 1);
	is_impl->valid_mask->disable(ptr, count);
	IndexSpace is = is_data->parent;
	if(is == IndexSpace::NO_SPACE) break;
	impl = get_runtime()->get_index_space_impl(is);
      }
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class ValidMaskRequestMessage
  //

  /*static*/ void ValidMaskRequestMessage::handle_request(RequestArgs args)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    IndexSpaceImpl *r_impl = get_runtime()->get_index_space_impl(args.is);

    const ElementMask *mask = r_impl->valid_mask;
    assert(mask);
    const char *mask_data = (const char *)(mask->get_raw());
    assert(mask_data);

    size_t mask_len = r_impl->valid_mask->raw_size();

    // send data in 2KB blocks
    unsigned block_id = 0;
    while(mask_len >= (1 << 11)) {
      ValidMaskDataMessage::send_request(args.sender, args.is, block_id,
					 mask->first_element,
					 mask->num_elements,
					 mask->first_enabled_elmt,
					 mask->last_enabled_elmt,
					 mask_data,
					 1 << 11,
					 PAYLOAD_KEEP);
      mask_data += 1 << 11;
      mask_len -= 1 << 11;
      block_id++;
    }
    if(mask_len) {
      ValidMaskDataMessage::send_request(args.sender, args.is, block_id,
					 mask->first_element,
					 mask->num_elements,
					 mask->first_enabled_elmt,
					 mask->last_enabled_elmt,
					 mask_data,
					 mask_len,
					 PAYLOAD_KEEP);
    }
  }

  /*static*/ void ValidMaskRequestMessage::send_request(gasnet_node_t target,
							IndexSpace is)
  {
    RequestArgs args;

    args.sender = gasnet_mynode();
    args.is = is;
    Message::request(target, args);
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class ValidMaskDataMessage
  //

  /*static*/ void ValidMaskDataMessage::handle_request(RequestArgs args,
						       const void *data,
						       size_t datalen)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    IndexSpaceImpl *r_impl = get_runtime()->get_index_space_impl(args.is);

    Event to_trigger = Event::NO_EVENT;
    {
      AutoHSLLock a(r_impl->valid_mask_mutex);
      log_meta.info() << "received valid mask data for " << args.is << ", " << datalen << " bytes (" << r_impl->valid_mask_count << " blocks expected)";

      ElementMask *mask = r_impl->valid_mask;
      assert(mask);

      // make sure parameters match
      if((mask->first_element != args.first_element) ||
	 (mask->num_elements != args.num_elements) ||
	 (mask->first_enabled_elmt != args.first_enabled_elmt) ||
	 (mask->last_enabled_elmt != args.last_enabled_elmt)) {
	log_meta.info() << "resizing valid mask for " << args.is << " (first=" << args.first_element << " num=" << args.num_elements << ")";

	mask->first_element = args.first_element;
	mask->num_elements = args.num_elements;
	mask->first_enabled_elmt = args.first_enabled_elmt;
	mask->last_enabled_elmt = args.last_enabled_elmt;
	free(mask->raw_data);
	size_t bytes_needed = ElementMaskImpl::bytes_needed(args.first_element, args.num_elements);
	mask->raw_data = (char *)calloc(1, bytes_needed);  // sets initial values to 0
	r_impl->valid_mask_count = (mask->raw_size() + 2047) >> 11;
      }

      assert((args.block_id << 11) < mask->raw_size());

      memcpy(mask->raw_data + (args.block_id << 11), data, datalen);

      //printf("got piece of valid mask data for region " IDFMT " (%d expected)\n",
      //       args.region.id, r_impl->valid_mask_count);
      r_impl->valid_mask_count--;
      if(r_impl->valid_mask_count == 0) {
	r_impl->valid_mask_complete = true;
	to_trigger = r_impl->valid_mask_event;
	r_impl->valid_mask_event = Event::NO_EVENT;
      }
    }

    if(to_trigger.exists()) {
      //printf("triggering " IDFMT "/%d\n",
      //       r_impl->valid_mask_event.id, r_impl->valid_mask_event.gen);
      GenEventImpl::trigger(to_trigger, false /*!poisoned*/);
    }
  }

  /*static*/ void ValidMaskDataMessage::send_request(gasnet_node_t target,
						     IndexSpace is, unsigned block_id,
						     coord_t first_element,
						     size_t num_elements,
						     coord_t first_enabled_elmt,
						     coord_t last_enabled_elmt,
						     const void *data,
						     size_t datalen,
						     int payload_mode)
  {
    RequestArgs args;

    args.is = is;
    args.block_id = block_id;
    args.first_element = first_element;
    args.num_elements = num_elements;
    args.first_enabled_elmt = first_enabled_elmt;
    args.last_enabled_elmt = last_enabled_elmt;
    Message::request(target, args, data, datalen, payload_mode);
  }

  class FetchISMaskWaiter : public EventWaiter {
  public:
    FetchISMaskWaiter(Event complete, IndexSpace is);
    virtual ~FetchISMaskWaiter(void);
    void sleep_on_event(Event e, Reservation l = Reservation::NO_RESERVATION);
    bool fetch_is_mask();
    virtual bool event_triggered(Event e, bool poisoned);
    virtual void print(std::ostream& os) const;
    virtual Event get_finish_event(void) const;
    Event complete_event;
    IndexSpace is;
    Reservation current_lock;
  };

  FetchISMaskWaiter::FetchISMaskWaiter(Event _complete, IndexSpace _is)
  : complete_event(_complete), is(_is), current_lock(Reservation::NO_RESERVATION) {}

  FetchISMaskWaiter::~FetchISMaskWaiter() {}

  void FetchISMaskWaiter::sleep_on_event(Event e, Reservation l)
  {
    current_lock = l;
    EventImpl::add_waiter(e, this);
  }

  bool FetchISMaskWaiter::fetch_is_mask()
  {
    IndexSpaceImpl *is_impl = get_runtime()->get_index_space_impl(is);
    if (!is_impl->locked_data.valid) {
      Event e = is_impl->lock.acquire(1, false, ReservationImpl::ACQUIRE_BLOCKING);
      if (e.has_triggered()) {
        is_impl->lock.release();
      } else {
        sleep_on_event(e, is_impl->lock.me);
        return false;
      }
    }

    Event e = is_impl->request_valid_mask();
    if (!e.has_triggered()) {
      sleep_on_event(e);
      return false;
    }

    GenEventImpl::trigger(complete_event, false/*poisoned*/);
    return true;
  }

  bool FetchISMaskWaiter::event_triggered(Event e, bool poisoned)
  {
    if (poisoned) {
      Realm::log_poison.info() << "poisoned fetch index space mask waiter";
      GenEventImpl::trigger(complete_event, true/*poisoned*/);
      return false;
    }

    if (current_lock.exists()) {
      current_lock.release();
      current_lock = Reservation::NO_RESERVATION;
    }

    bool ret = fetch_is_mask();
    return ret;
  }

  void FetchISMaskWaiter::print(std::ostream& os) const
  {
  }

  Event FetchISMaskWaiter::get_finish_event(void) const
  {
    return complete_event;
  }

  /*static*/ void ValidMaskFetchMessage::handle_request(RequestArgs args)
  {
    FetchISMaskWaiter* waiter = new FetchISMaskWaiter(args.complete, args.is);
    // don't need to worry about delete, which will be performed by EventWaiter gc
    waiter->fetch_is_mask();
  }

  /*static*/ void ValidMaskFetchMessage::send_request(gasnet_node_t target,
                                                      IndexSpace is,
                                                      Event complete)
  {
    RequestArgs args;
    args.is = is;
    args.complete = complete;
    Message::request(target, args);
  }
  
}; // namespace Realm
