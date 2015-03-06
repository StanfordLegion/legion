/* Copyright 2015 Stanford University
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

namespace LegionRuntime {
  namespace LowLevel {
    DiskMemory::DiskMemory(Memory _me, size_t _size, std::string _file)
    : Memory::Impl(_me, _size, MKIND_DISK, ALIGNMENT, Memory::DISK_MEM)
    {
      printf("file = %s\n", _file.c_str());
      fd = open(_file.c_str(), O_CREAT | O_RDWR, 00777);
      assert(fd != -1);
      write(fd, "", _size);
      free_blocks[0] = _size;
    }

    DiskMemory::~DiskMemory(void)
    {
      close(fd);
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
      assert(aio_read(&cb) != -1);
      while (aio_error(&cb) == EINPROGRESS) {}
      assert((size_t)aio_return(&cb) == size);
    }

    void DiskMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      aiocb cb;
      memset(&cb, 0, sizeof(cb));
      cb.aio_nbytes = size;
      cb.aio_fildes = fd;
      cb.aio_offset = offset;
      cb.aio_buf = (void *)src;
      assert(aio_write(&cb) != -1);
      while (aio_error(&cb) == EINPROGRESS) {}
      assert((size_t)aio_return(&cb) == size);
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

  }
}
