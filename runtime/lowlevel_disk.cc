/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "realm/realm_config.h"
#include "lowlevel_impl.h"
#include "lowlevel.h"
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

namespace Realm {

    DiskMemory::DiskMemory(Memory _me, size_t _size, std::string _file)
      : MemoryImpl(_me, _size, MKIND_DISK, ALIGNMENT, Memory::DISK_MEM), file(_file)
    {
      printf("file = %s\n", _file.c_str());
      // do not overwrite an existing file
      fd = open(_file.c_str(), O_CREAT | O_EXCL | O_RDWR, 00777);
      assert(fd != -1);
      // resize the file to what we want
#ifndef NDEBUG
      int ret =
#endif
	ftruncate(fd, _size);
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
                     const Realm::ProfilingRequestSet &reqs,
                     RegionInstance parent_inst)
    {
      return create_instance_local(is, linearization_bits, bytes_needed,
                     block_size, element_size, field_sizes, redopid,
                     list_size, reqs, parent_inst);
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
      // this is a blocking operation
#ifndef NDEBUG
      ssize_t amt =
#endif
	pread(fd, dst, size, offset);
      assert(amt == (ssize_t)size);
    }

    void DiskMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // this is a blocking operation
#ifndef NDEBUG
      ssize_t amt =
#endif
	pwrite(fd, src, size, offset);
      assert(amt == (ssize_t)size);
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

    FileMemory::FileMemory(Memory _me)
      : MemoryImpl(_me, 0 /*no memory space*/, MKIND_FILE, ALIGNMENT, Memory::FILE_MEM)
    {
      pthread_mutex_init(&vector_lock, NULL);
    }

    FileMemory::~FileMemory(void)
    {
      pthread_mutex_destroy(&vector_lock);
    }

    RegionInstance FileMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     const ProfilingRequestSet &reqs,
                     RegionInstance parent_inst)
    {
      // we use a new create_instance API
      assert(0);
      return RegionInstance::NO_INST;
    }

    RegionInstance FileMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     const ProfilingRequestSet &reqs,
                     RegionInstance parent_inst,
                     const char *file_name,
                     Domain domain,
                     legion_lowlevel_file_mode_t file_mode)
    {
      RegionInstance inst =  create_instance_local(is,
                   linearization_bits, bytes_needed,
                   block_size, element_size, field_sizes, redopid,
                   list_size, reqs, parent_inst);
      int fd;
#ifdef REALM_USE_KERNEL_AIO
      int direct_flag = O_DIRECT;
#else
      int direct_flag = 0;
#endif
      switch (file_mode) {
        case LEGION_FILE_READ_ONLY:
        {
          fd = open(file_name, O_RDONLY | direct_flag, 00777);
          assert(fd != -1);
          break;
        }
        case LEGION_FILE_READ_WRITE:
        {
          fd = open(file_name, O_RDWR | direct_flag, 00777);
          assert(fd != -1);
          break;
        }
        case LEGION_FILE_CREATE:
        {
          fd = open(file_name, O_CREAT | O_RDWR | direct_flag, 00777);
          assert(fd != -1);
          // resize the file to what we want
          size_t field_size = 0;
          for(std::vector<size_t>::const_iterator it = field_sizes.begin(); it != field_sizes.end(); it++) {
            field_size += *it;
          }
#ifndef NDEBUG
          int ret =
#endif
	    ftruncate(fd, field_size * domain.get_volume());
          assert(ret == 0);
          break;
        }
        default:
          assert(0);
      }

      pthread_mutex_lock(&vector_lock);
      ID id(inst);
      unsigned index = id.instance.inst_idx;
      if (index < file_vec.size())
        file_vec[index] = fd;
      else {
        assert(index == file_vec.size());
        file_vec.push_back(fd);
      }
      pthread_mutex_unlock(&vector_lock);
      return inst;
    }

    void FileMemory::destroy_instance(RegionInstance i,
                      bool local_destroy)
    {
      pthread_mutex_lock(&vector_lock);
      ID id(i);
      unsigned index = id.instance.inst_idx;
      assert(index < file_vec.size());
      int fd = file_vec[index];
      pthread_mutex_unlock(&vector_lock);
      close(fd);
      destroy_instance_local(i, local_destroy);
    }

    off_t FileMemory::alloc_bytes(size_t size)
    {
      // Offset for every instance should be 0!!!
      return 0;
    }

    void FileMemory::free_bytes(off_t offset, size_t size)
    {
      // Do nothing in this function.
    }

    void FileMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

    void FileMemory::get_bytes(ID::IDType inst_id, off_t offset, void *dst, size_t size)
    {
      pthread_mutex_lock(&vector_lock);
      int fd = file_vec[inst_id];
      pthread_mutex_unlock(&vector_lock);
#ifndef NDEBUG
      size_t ret =
#endif
	pread(fd, dst, size, offset);
      assert(ret == size);
    }

    void FileMemory::put_bytes(off_t offset, const void *src, size_t)
    {
      assert(0);
    }

    void FileMemory::put_bytes(ID::IDType inst_id, off_t offset, const void *src, size_t size)
    {
      pthread_mutex_lock(&vector_lock);
      int fd = file_vec[inst_id];
      pthread_mutex_unlock(&vector_lock);
#ifndef NDEBUG
      size_t ret =
#endif
	pwrite(fd, src, size, offset);
      assert(ret == size);
    }

    void FileMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                                          size_t count, const void *entry_buffer) {}

    void *FileMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it;
    }

    int FileMemory::get_home_node(off_t offset, size_t size)
    {
      return gasnet_mynode();
    }

    int FileMemory::get_file_des(ID::IDType inst_id)
    {
      pthread_mutex_lock(&vector_lock);
      int fd = file_vec[inst_id];
      pthread_mutex_unlock(&vector_lock);
      return fd;
    }

#ifdef USE_HDF
    HDFMemory::HDFMemory(Memory _me)
      : MemoryImpl(_me, 0 /*HDF doesn't have memory space*/, MKIND_HDF, ALIGNMENT, Memory::HDF_MEM)
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
                     const Realm::ProfilingRequestSet &reqs,
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
                     const Realm::ProfilingRequestSet &reqs,
                     RegionInstance parent_inst,
                     const char* file,
                     const std::vector<const char*>& path_names,
                     Domain domain,
                     bool read_only)

    {
      RegionInstance inst = create_instance_local(is,
                 linearization_bits, bytes_needed,
                 block_size, element_size, field_sizes, redopid,
                 list_size, reqs, parent_inst);

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
      for (ID::IDType idx = 0; idx < path_names.size(); idx ++) {
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

    void HDFMemory::get_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, void *dst, size_t size)
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

    void HDFMemory::put_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, const void *src, size_t size)
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
