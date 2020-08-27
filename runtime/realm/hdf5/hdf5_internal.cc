/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#include "realm/hdf5/hdf5_internal.h"

#include "realm/hdf5/hdf5_access.h"
#include "realm/logging.h"

namespace Realm {

  extern Logger log_hdf5;

  namespace HDF5 {

    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5Memory

    HDF5Memory::HDF5Memory(Memory _me)
      : MemoryImpl(_me, 0 /*HDF doesn't have memory space*/, MKIND_HDF,
		   Memory::HDF_MEM, 0)
    {
    }

    HDF5Memory::~HDF5Memory(void)
    {
      // close all HDF metadata
    }

    void HDF5Memory::get_bytes(off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

#if 0
    void HDF5Memory::get_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, void *dst, size_t size)
    {
      assert(0);
      HDFMetadata *metadata = hdf_metadata[inst_id];
      // use index to compute position in space
      assert(size == H5Tget_size(metadata->datatype_ids[fid]));
      hsize_t offset[3], count[3];
      for (int i = 0; i < metadata->ndims; i++) {
        offset[i] = dp.point_data[i] - metadata->lo[i];
      }
      count[0] = count[1] = count[2] = 1;
      hid_t dataspace_id = H5Dget_space(metadata->dataset_ids[fid]);
      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
      hid_t memspace_id = H5Screate_simple(metadata->ndims, count, NULL);
      H5Dread(metadata->dataset_ids[fid], metadata->datatype_ids[fid], memspace_id, dataspace_id, H5P_DEFAULT, dst);
      H5Sclose(dataspace_id);
      H5Sclose(memspace_id);
    }
#endif

    void HDF5Memory::put_bytes(off_t offset, const void *src, size_t size)
    {
      assert(0);
    }

#if 0
    void HDF5Memory::put_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, const void *src, size_t size)
    {
      assert(0);
      HDFMetadata *metadata = hdf_metadata[inst_id];
      // use index to compute position in space
      assert(size == H5Tget_size(hdf_metadata[inst_id]->datatype_ids[fid]));
      hsize_t offset[3], count[3];
      for (int i = 0; i < metadata->ndims; i++) {
        offset[i] = dp.point_data[i] - metadata->lo[i];
      }
      count[0] = count[1] = count[2] = 1;
      hid_t dataspace_id = H5Dget_space(metadata->dataset_ids[fid]);
      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
      hid_t memspace_id = H5Screate_simple(metadata->ndims, count, NULL);
      H5Dwrite(metadata->dataset_ids[fid], metadata->datatype_ids[fid], memspace_id, dataspace_id, H5P_DEFAULT, src);
      H5Sclose(dataspace_id);
      H5Sclose(memspace_id);
    }
#endif

    void *HDF5Memory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it.
    }

    int HDF5Memory::get_home_node(off_t offset, size_t size)
    {
      return Network::my_node_id;
    }

    MemoryImpl::AllocationResult HDF5Memory::allocate_storage_immediate(RegionInstanceImpl *inst,
							    bool need_alloc_result,
							    bool poisoned,
							    TimeLimit work_until)
    {
      // if the allocation request doesn't include an external HDF5 resource,
      //  we fail it immediately
      ExternalHDF5Resource *res = dynamic_cast<ExternalHDF5Resource *>(inst->metadata.ext_resource);
      if(res == 0) {
	if(inst->metadata.ext_resource)
	  log_inst.warning() << "attempt to register non-hdf5 resource: mem=" << me << " resource=" << *(inst->metadata.ext_resource);
	else
	  log_inst.warning() << "attempt to allocate memory in hdf5 memory: layout=" << *(inst->metadata.layout);
	inst->notify_allocation(ALLOC_INSTANT_FAILURE, 0, work_until);
	return ALLOC_INSTANT_FAILURE;
      }

      // poisoned preconditions cancel the allocation request
      if(poisoned) {
	inst->notify_allocation(ALLOC_CANCELLED, 0, work_until);
	return ALLOC_CANCELLED;
      }

      // TODO: try to open the file once here to make sure it exists?

      AllocationResult result = ALLOC_INSTANT_SUCCESS;
      size_t inst_offset = 0;
      inst->notify_allocation(result, inst_offset, work_until);

      return result;
    }

    void HDF5Memory::release_storage_immediate(RegionInstanceImpl *inst,
					       bool poisoned,
					       TimeLimit work_until)
    {
      // nothing to do for a poisoned release
      if(poisoned)
	return;

      // we didn't save anything on the instance, so just ack and return
      inst->notify_deallocation();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5WriteChannel

    HDF5WriteChannel::HDF5WriteChannel(HDF5Memory *_mem)
      : MemPairCopierFactory("hdf5write")
      , mem(_mem)
    {}

    bool HDF5WriteChannel::can_perform_copy(Memory src_mem, Memory dst_mem,
					    ReductionOpID redop_id, bool fold)
    {
      // only look at things that are in our hdf5 memory
      if(dst_mem != mem->me)
	return false;

      // no support for reduction ops
      assert(redop_id == 0);

      return true;
    }

#ifdef OLD_COPIERS
    MemPairCopier *HDF5WriteChannel::create_copier(Memory src_mem, Memory dst_mem,
						   ReductionOpID redop_id, bool fold)
    {
      MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
      return new HDF5WriteCopier(src_impl, mem);
    }
#endif


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5ReadChannel

    HDF5ReadChannel::HDF5ReadChannel(HDF5Memory *_mem)
      : MemPairCopierFactory("hdf5read")
      , mem(_mem)
    {}

    bool HDF5ReadChannel::can_perform_copy(Memory src_mem, Memory dst_mem,
					    ReductionOpID redop_id, bool fold)
    {
      // only look at things that are in our hdf5 memory
      if(src_mem != mem->me)
	return false;

      // no support for reduction ops
      assert(redop_id == 0);

      return true;
    }

#ifdef OLD_COPIERS
    MemPairCopier *HDF5ReadChannel::create_copier(Memory src_mem, Memory dst_mem,
						   ReductionOpID redop_id, bool fold)
    {
      MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
      return new HDF5ReadCopier(mem, dst_impl);
    }
#endif


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5WriteCopier

#ifdef OLD_COPIERS
    HDF5WriteCopier::HDF5WriteCopier(MemoryImpl *_src_impl, HDF5Memory *_mem)
      : src_impl(_src_impl)
      , mem(_mem)
    {}

    HDF5WriteCopier::~HDF5WriteCopier(void)
    {}

    InstPairCopier *HDF5WriteCopier::inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					       OASVec &oas_vec)
    {
      RegionInstanceImpl *local_impl = get_runtime()->get_instance_impl(src_inst);
      std::map<RegionInstance, HDF5Memory::HDFMetadata *>::const_iterator it = mem->hdf_metadata.find(dst_inst);
      assert(it != mem->hdf_metadata.end());
      HDF5Memory::HDFMetadata *md = it->second;
      return new HDF5InstPairCopier<HDF5WriteCopier>(this, local_impl, md, oas_vec);
    }

    inline void HDF5WriteCopier::transfer_data(hid_t dset_id, hid_t dtype_id,
					       hid_t mspace_id, hid_t dspace_id,
					       void *data_ptr)
    {
      herr_t err = H5Dwrite(dset_id, dtype_id,
			    mspace_id,
			    dspace_id,
			    H5P_DEFAULT,
			    data_ptr);
      assert(err >= 0);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5ReadCopier

    HDF5ReadCopier::HDF5ReadCopier(HDF5Memory *_mem, MemoryImpl *_dst_impl)
      : mem(_mem)
      , dst_impl(_dst_impl)
    {}

    HDF5ReadCopier::~HDF5ReadCopier(void)
    {}

    InstPairCopier *HDF5ReadCopier::inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					      OASVec &oas_vec)
    {
      std::map<RegionInstance, HDF5Memory::HDFMetadata *>::const_iterator it = mem->hdf_metadata.find(src_inst);
      assert(it != mem->hdf_metadata.end());
      HDF5Memory::HDFMetadata *md = it->second;
      RegionInstanceImpl *local_impl = get_runtime()->get_instance_impl(dst_inst);
      return new HDF5InstPairCopier<HDF5ReadCopier>(this, local_impl, md, oas_vec);
    }

    inline void HDF5ReadCopier::transfer_data(hid_t dset_id, hid_t dtype_id,
					      hid_t mspace_id, hid_t dspace_id,
					      void *data_ptr)
    {
      herr_t err = H5Dread(dset_id, dtype_id,
			   mspace_id,
			   dspace_id,
			   H5P_DEFAULT,
			   data_ptr);
      assert(err >= 0);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5InstPairCopier

    template <typename T>
    HDF5InstPairCopier<T>::HDF5InstPairCopier(T *_mpc, RegionInstanceImpl *_local_impl, HDF5Memory::HDFMetadata *_md, OASVec& _oas_vec)
      : mpc(_mpc)
      , local_impl(_local_impl)
      , md(_md)
      , oas_vec(_oas_vec)
    {}

    template <typename T>
    HDF5InstPairCopier<T>::~HDF5InstPairCopier(void)
    {}

    // bleah - too much templating
    template <typename T>
    static void unpack_oas(const T& oas, HDF5WriteCopier *mpc, int& local_ofs, int& hdf_ofs)
    {
      local_ofs = oas.src_offset;
      hdf_ofs = oas.dst_offset;
    }
    template <typename T>
    static void unpack_oas(const T& oas, HDF5ReadCopier *mpc, int& local_ofs, int& hdf_ofs)
    {
      hdf_ofs = oas.src_offset;
      local_ofs = oas.dst_offset;
    }
#endif

#if 0
    template <typename T, unsigned DIM>
    static void copy_rect(LegionRuntime::Arrays::Rect<DIM> r,
			  T *mpc,
			  RegionInstanceImpl *local_impl,
			  HDF5Memory::HDFMetadata *md,
			  OASVec& oas_vec)
    {
      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(local_impl->memory);
      char *mem_base = (char *)(mem_impl->get_direct_ptr(0, local_impl->metadata.size));
      assert(mem_base);

      size_t copy_count = 0;

      log_hdf5.debug() << "copy: " << r;
      log_hdf5.debug() << "top: [" << md->ndims << "] (" << md->lo[0] << "," << md->lo[1] << "," << md->lo[2] << ") + (" << md->dims[0] << "," << md->dims[1] << "," << md->dims[2] << ")";
      assert(md->ndims == (int)DIM);
      for(OASVec::const_iterator it = oas_vec.begin(); it != oas_vec.end(); ++it) {
	log_hdf5.debug() << " oas: " << it->src_offset << "," << it->dst_offset << "," << it->size << "," << it->serdez_id;
	int local_ofs, hdf_ofs;
	unpack_oas(*it, mpc, local_ofs, hdf_ofs);
	assert(md->dataset_ids.count(hdf_ofs) > 0);
	hid_t dset_id = md->dataset_ids[hdf_ofs];
	hid_t dtype_id = md->datatype_ids[hdf_ofs];
	log_hdf5.debug() << " hdf: " << md->file_id << "," << dset_id << "," << dtype_id;

	hid_t dspace_id = H5Dget_space(dset_id);
	assert(dspace_id > 0);
	assert((int)DIM == H5Sget_simple_extent_ndims(dspace_id));
	hsize_t dsdims[DIM];
	H5Sget_simple_extent_dims(dspace_id, dsdims, 0);
	log_hdf5.debug() << " dsd: " << dsdims[0];

	off_t local_start;
	int local_size;
	find_field_start(local_impl->metadata.field_sizes, local_ofs, it->size,
			 local_start, local_size);
	assert(local_size == (int)(it->size));

	LegionRuntime::Arrays::Mapping<DIM, 1> *linearization = local_impl->metadata.linearization.get_mapping<DIM>();
	for(typename LegionRuntime::Arrays::Mapping<DIM, 1>::LinearSubrectIterator lsi(r, *linearization); lsi; lsi++) {
	  log_hdf5.debug() << " sr: " << lsi.subrect << " " << lsi.strides[0];
	  off_t l_offset = calc_mem_loc(local_impl->metadata.alloc_offset + (local_ofs - local_start),
					local_start, local_size,
					local_impl->metadata.elmt_size,
					local_impl->metadata.block_size,
					lsi.image_lo[0]);
	  // assume SOA for now
	  assert(local_impl->metadata.block_size >= (lsi.image_lo[0] + r.volume()));
	  void *data_ptr = mem_base + l_offset;
	  log_hdf5.debug() << " ptr: " << data_ptr;

	  // create a memory space that describes the a contiguous N-D array
	  // that our memory buffer is a subset of - i.e. use our strides as
	  // the extents of the effectively-contiguous supserset
	  // CAREFUL: HDF5 wants the fastest-changing dimension to be last - ours
	  //  is first, so reverse the dimensions on things we give to HDF5 API calls
	  hsize_t ms_dims[DIM];
	  assert(lsi.strides[0][0] == 1);
	  if(DIM > 1) {
	    ms_dims[DIM - 1] = lsi.strides[1][0];
	    for(unsigned i = 1; i < DIM - 1; i++) {
	      // this has to divide evenly
	      assert((lsi.strides[i + 1][0] % lsi.strides[i][0]) == 0);
	      ms_dims[DIM - 1 - i] = lsi.strides[i + 1][0] / lsi.strides[i][0];
	    }
	  }
	  // "last" dim just needs to be at least what we want to copy
	  ms_dims[0] = lsi.subrect.hi[DIM - 1] - lsi.subrect.lo[DIM - 1] + 1;
	  log_hdf5.debug() << " ms_dims: " << ms_dims[0];
	  hid_t ms_id = H5Screate_simple(DIM, ms_dims, 0);
	  assert(ms_id > 0);
	  // now select the sub-block starting at the origin of the right size
	  hsize_t ms_start[DIM], ms_count[DIM];
	  for(unsigned i = 0; i < DIM; i++) {
	    ms_start[i] = 0;
	    ms_count[i] = lsi.subrect.hi[DIM - 1 - i] - lsi.subrect.lo[DIM - 1 - i] + 1;
	  }
	  {
	    herr_t err = H5Sselect_hyperslab(ms_id, H5S_SELECT_SET,
					     ms_start, 0 /* default stride=1 */,
					     ms_count, 0 /* default block=1 */);
	    assert(err >= 0);
	  }

	  // now pick the right data in the dataset - make sure to consider
	  //  the offset of this region relative to the attachment point
	  hid_t ds_id = H5Scopy(dspace_id);
	  assert(ds_id > 0);
	  hsize_t ds_start[DIM], ds_count[DIM];
	  for(unsigned i = 0; i < DIM; i++) {
	    ds_start[i] = lsi.subrect.lo[DIM - 1 - i] - md->lo[DIM - 1 - i];
	    ds_count[i] = ms_count[i];
	    assert(ds_start[i] >= 0);
	    assert((ds_start[i] + ds_count[i]) <= md->dims[DIM - 1 - i]);
	  }
	  {
	    herr_t err = H5Sselect_hyperslab(ds_id, H5S_SELECT_SET,
					     ds_start, 0 /* default stride=1 */,
					     ds_count, 0 /* default block=1 */);
	    assert(err >= 0);
	  }

	  mpc->transfer_data(dset_id, dtype_id, ms_id, ds_id, data_ptr);
	  copy_count += 1;

	  // give back our temp data spaces
	  H5Sclose(ms_id);
	  H5Sclose(ds_id);
	}

	// flush each data set
	//herr_t err = H5Dflush(dset_id);
	//assert(err >= 0);
	H5Sclose(dspace_id);
      }
    }
#endif

#ifdef OLD_COPIERS
    template <typename T>
    bool HDF5InstPairCopier<T>::copy_all_fields(Domain d)
    {
      assert(0);
#ifdef DEAD_CODE
      //log_hdf5.print() << "copy all fields";
      switch(d.get_dim()) {
      case 0:
	{
	  log_hdf5.fatal() << "HDF5 copies not supported for unstructured domains!";
	  assert(false);
	  return false;
	}
      case 1:
	{
	  copy_rect(d.get_rect<1>(), mpc, local_impl, md, oas_vec);
	  return true;
	}
      case 2:
	{
	  copy_rect(d.get_rect<2>(), mpc, local_impl, md, oas_vec);
	  return true;
	}
      case 3:
	{
	  copy_rect(d.get_rect<3>(), mpc, local_impl, md, oas_vec);
	  return true;
	}
      default:
	assert(false);
	return false;
      }
#endif
    }

    template <typename T>
    void HDF5InstPairCopier<T>::copy_field(off_t src_index, off_t dst_index, off_t elem_count,
					   unsigned offset_index)
    {
      assert(0);
    }

    template <typename T>
    void HDF5InstPairCopier<T>::copy_all_fields(off_t src_index, off_t dst_index, off_t elem_count)
    {
      assert(0);
    }
    
    template <typename T>
    void HDF5InstPairCopier<T>::flush(void)
    {
      //log_hdf5.print() << "flush";
    }
#endif


  }; // namespace HDF5

}; // namespace Realm
