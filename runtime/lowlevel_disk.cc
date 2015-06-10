/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "lowlevel_impl.h"
#include "lowlevel.h"
#include <aio.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

namespace LegionRuntime {
  namespace LowLevel {
    DiskMemory::DiskMemory(Memory _me, size_t _size, std::string _file)
      : Memory::Impl(_me, _size, MKIND_DISK, ALIGNMENT, Memory::DISK_MEM), file(_file)
    {
      printf("file = %s\n", _file.c_str());
      // do not overwrite an existing file
      fd = open(_file.c_str(), O_CREAT | O_EXCL | O_RDWR, 00777);
      assert(fd != -1);
      // resize the file to what we want
      int ret = ftruncate(fd, _size);
      assert(ret == 0);
      free_blocks[0] = _size;
    }

    DiskMemory::~DiskMemory(void)
    {
      close(fd);
      // attempt to delete the file
      unlink(file.c_str());
    }

    RegionInstance DiskMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     RegionInstance parent_inst)
    {
      return create_instance_local(is, linearization_bits, bytes_needed,
                     block_size, element_size, field_sizes, redopid,
                     list_size, parent_inst);
    }

    void DiskMemory::destroy_instance(RegionInstance i,
				     bool local_destroy)
    {
      destroy_instance_local(i, local_destroy);
    }

    off_t DiskMemory::alloc_bytes(size_t size)
    {
      return alloc_bytes_local(size);
    }

    void DiskMemory::free_bytes(off_t offset, size_t size)
    {
      free_bytes_local(offset, size);
    }

    void DiskMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      aiocb cb;
      memset(&cb, 0, sizeof(cb));
      cb.aio_nbytes = size;
      cb.aio_fildes = fd;
      cb.aio_offset = offset;
      cb.aio_buf = dst;
      int ret = aio_read(&cb);
      assert(ret == 0);
      while(true) {
	ret = aio_error(&cb);
	if(ret == 0) break; // completed
	assert(ret == EINPROGRESS); // only other choice we expect to see
      }
      ssize_t count = aio_return(&cb);
      assert(count == (ssize_t)size);
    }

    void DiskMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      aiocb cb;
      memset(&cb, 0, sizeof(cb));
      cb.aio_nbytes = size;
      cb.aio_fildes = fd;
      cb.aio_offset = offset;
      cb.aio_buf = (void *)src;
      int ret = aio_write(&cb);
      assert(ret == 0);
      while(true) {
	ret = aio_error(&cb);
	if(ret == 0) break; // completed
	assert(ret == EINPROGRESS); // only other choice we expect to see
      }
      ssize_t count = aio_return(&cb);
      assert(count == (ssize_t)size);
    }

    void DiskMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                        size_t count, const void *entry_buffer)
    {
    }

    void *DiskMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it.
    }

    int DiskMemory::get_home_node(off_t offset, size_t size)
    {
      return gasnet_mynode();
    }

#ifdef USE_HDF
    HDFMemory::HDFMemory(Memory _me)
      : Memory::Impl(_me, 0 /*HDF doesn't have memory space*/, MKIND_HDF, ALIGNMENT, Memory::HDF_MEM)
    {
    }

    HDFMemory::~HDFMemory(void)
    {
      // close all HDF metadata
    }

    RegionInstance HDFMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     RegionInstance parent_inst)
    {
      // we use a new create_instance, which could provide
      // more information for creating HDF metadata
      assert(0);
      return RegionInstance::NO_INST;
    }

    RegionInstance HDFMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     RegionInstance parent_inst,
                     const char* file,
                     const std::vector<const char*>& path_names,
                     Domain domain,
                     bool read_only)

    {
      RegionInstance inst = create_instance_local(is,
                 linearization_bits, bytes_needed,
                 block_size, element_size, field_sizes, redopid,
                 list_size, parent_inst);

      HDFMetadata* new_hdf = new HDFMetadata;
      new_hdf->ndims = domain.get_dim();
      for (int i = 0; i < domain.get_dim(); i++) {
        new_hdf->lo[i] = domain.rect_data[i];
        new_hdf->dims[i] = domain.rect_data[i + domain.get_dim()] - domain.rect_data[i];
      }
      unsigned flags;
      if (read_only)
        flags = H5F_ACC_RDONLY;
      else
        flags = H5F_ACC_RDWR;
      new_hdf->file_id = H5Fopen(file, flags, H5P_DEFAULT);
      for (IDType idx = 0; idx < path_names.size(); idx ++) {
        new_hdf->dataset_ids.push_back(H5Dopen2(new_hdf->file_id, path_names[idx], H5P_DEFAULT));
        new_hdf->datatype_ids.push_back(H5Dget_type(new_hdf->dataset_ids[idx]));
      }

      if (inst.id < hdf_metadata.size())
        hdf_metadata[inst.id] = new_hdf;
      else
        hdf_metadata.push_back(new_hdf);

      return inst;
    }

    void HDFMemory::destroy_instance(RegionInstance i,
				     bool local_destroy)
    {
      HDFMetadata* new_hdf = hdf_metadata[i.id];
      assert(new_hdf->dataset_ids.size() == new_hdf->datatype_ids.size());
      for (size_t idx = 0; idx < new_hdf->dataset_ids.size(); idx++) {
        H5Dclose(new_hdf->dataset_ids[idx]);
        H5Tclose(new_hdf->datatype_ids[idx]);
      }
      H5Fclose(new_hdf->file_id);
      new_hdf->dataset_ids.clear();
      new_hdf->datatype_ids.clear();
      delete new_hdf;
      destroy_instance_local(i, local_destroy);
    }

    off_t HDFMemory::alloc_bytes(size_t size)
    {
      // We don't have to actually allocate bytes
      // for HDF memory. So we do nothing in this
      // function
      return 0;
    }

    void HDFMemory::free_bytes(off_t offset, size_t size)
    {
      // We don't have to free bytes for HDF memory
    }

    void HDFMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

    void HDFMemory::get_bytes(IDType inst_id, const DomainPoint& dp, int fid, void *dst, size_t size)
    {
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

    void HDFMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      assert(0);
    }

    void HDFMemory::put_bytes(IDType inst_id, const DomainPoint& dp, int fid, const void *src, size_t size)
    {
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

    void HDFMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                        size_t count, const void *entry_buffer)
    {
    }

    void *HDFMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it.
    }

    int HDFMemory::get_home_node(off_t offset, size_t size)
    {
      return gasnet_mynode();
    }
#endif
  }
}
