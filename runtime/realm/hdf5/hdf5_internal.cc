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


  }; // namespace HDF5

}; // namespace Realm
