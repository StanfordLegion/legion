/* Copyright 2024 Stanford University, NVIDIA Corporation
 * Copyright 2024 Los Alamos National Laboratory
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
#include "realm/runtime_impl.h"
#include "realm/deppart/inst_helper.h"
#include "realm/mem_impl.h"
#include "realm/idx_impl.h"
#include "realm/inst_impl.h"

#include <sys/types.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>

#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#include <unistd.h>
#endif

#ifdef REALM_ON_WINDOWS
#include <windows.h>
#include <io.h>

#define open _open
#define close _close
#define unlink _unlink

static ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
  if(_lseeki64(fd, offset, SEEK_SET) < 0)
    return -1;
  int ret = _read(fd, buf, count);
  return ret;
}

static ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset)
{
  if (_lseeki64(fd, offset, SEEK_SET) < 0)
    return -1;
  int ret = _write(fd, buf, count);
  return ret;
}

static int ftruncate(int fd, off_t size)
{
  return _chsize_s(fd, (__int64)size);
}

static int fsync(int fd)
{
  // TODO: is there a way to limit to just the specified file descriptor?
  _flushall();
  return 0;
}
#endif

namespace Realm {

    extern Logger log_inst;
    Logger log_disk("disk");

    DiskMemory::DiskMemory(RuntimeImpl *_runtime_impl, Memory _me, size_t _size,
                           const std::filesystem::path &_file)
      : LocalManagedMemory(_runtime_impl, _me, _size, MKIND_DISK, ALIGNMENT,
                           Memory::DISK_MEM, 0)
      , file(_file)
    {
      // Allow overwriting of an existing file in case Realm crashed
      // before and we are running again and need to overwrite the
      // file instead of crashing again because we can't open the file
      fd = open((const char *)_file.c_str(), O_CREAT | O_RDWR, 00666);
      assert(fd != -1);
      // resize the file to what we want
      int ret =	ftruncate(fd, _size);
#ifdef NDEBUG
      (void)ret;
#else
      assert(ret == 0);
#endif
    }

    DiskMemory::~DiskMemory(void)
    {
      close(fd);
      // attempt to delete the file
      unlink((const char *)file.c_str());
    }

    void DiskMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // this is a blocking operation
      ssize_t amt = pread(fd, dst, size, offset);
#ifdef NDEBUG
      (void)amt;
#else
      assert(amt == (ssize_t)size);
#endif
    }

    void DiskMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // this is a blocking operation
      ssize_t amt = pwrite(fd, src, size, offset);
#ifdef NDEBUG
      (void)amt;
#else
      assert(amt == (ssize_t)size);
#endif
    }

    void *DiskMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it.
    }

    ExternalInstanceResource *
    DiskMemory::generate_resource_info(RegionInstanceImpl *inst,
                                       const IndexSpaceGeneric *subspace,
                                       span<const FieldID> fields, bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
      }

      return new ExternalFileResource(
          file.string(), read_only ? REALM_FILE_READ_ONLY : REALM_FILE_READ_WRITE,
          inst->metadata.inst_offset + rel_base);
    }

    bool DiskMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                        size_t &inst_offset)
    {
      ExternalFileResource *res =
          dynamic_cast<ExternalFileResource *>(inst->metadata.ext_resource);
      if((res == nullptr) || (res->filename != file))
        return false;
      inst_offset = res->offset;
      return true;
    }

    void DiskMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      // Nothing to do here since we're just an offset of the disk memory
    }

    FileMemory::FileMemory(RuntimeImpl *_runtime_impl, Memory _me)
      : MemoryImpl(_runtime_impl, _me, 0 /*no memory space*/, MKIND_FILE,
                   Memory::FILE_MEM, 0)
    {
    }

    FileMemory::~FileMemory(void)
    {
    }

    void FileMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

    void FileMemory::get_bytes(ID::IDType inst_id, off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

    void FileMemory::put_bytes(off_t offset, const void *src, size_t)
    {
      assert(0);
    }

    void FileMemory::put_bytes(ID::IDType inst_id, off_t offset, const void *src, size_t size)
    {
      assert(0);
    }

    void *FileMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it;
    }

    ExternalInstanceResource *
    FileMemory::generate_resource_info(RegionInstanceImpl *inst,
                                       const IndexSpaceGeneric *subspace,
                                       span<const FieldID> fields, bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
      }

      ExternalFileResource *res =
          dynamic_cast<ExternalFileResource *>(inst->metadata.ext_resource);
      assert(res != nullptr);
      OpenFileInfo *info = inst->metadata.find_mem_specific<OpenFileInfo>();
      assert(info != nullptr);
      return new ExternalFileResource(
          res->filename, read_only ? REALM_FILE_READ_ONLY : REALM_FILE_READ_WRITE,
          info->offset + rel_base);
    }

    // FileMemory supports ExternalFileResource
    bool FileMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                        size_t& inst_offset)
    {
      {
        ExternalFileResource *res = dynamic_cast<ExternalFileResource *>(inst->metadata.ext_resource);
        if(res) {
          // try to open the file
          int fd;
          switch(res->mode) {
          case REALM_FILE_READ_ONLY:
          {
            fd = open(res->filename.c_str(), O_RDONLY);
            break;
          }
          case REALM_FILE_READ_WRITE:
          {
            fd = open(res->filename.c_str(), O_RDWR);
            break;
          }
          case REALM_FILE_CREATE:
          {
            fd = open(res->filename.c_str(), O_CREAT | O_RDWR, 0666);
            if(fd == -1) {
              log_disk.fatal() << "unable to open file '" << res->filename
                               << "': " << strerror(errno);
              abort();
            }
            // resize the file to what we want
            int ret = ftruncate(fd, inst->metadata.layout->bytes_used);
            if(ret == -1) {
              log_disk.fatal() << "failed to truncate file '" << res->filename
                               << "': " << strerror(errno);
              abort();
            }
            break;
          }
          default:
            assert(0);
          }

          if(fd == -1) {
            log_disk.fatal() << "unable to open file '" << res->filename << "': " << strerror(errno);
            abort();
          }

          OpenFileInfo *info = new OpenFileInfo;
          info->fd = fd;
          info->offset = res->offset;

          inst->metadata.add_mem_specific(info);
          return true;
        }
      }

      // not a kind we recognize
      return false;
    }

    void FileMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      OpenFileInfo *info = inst->metadata.find_mem_specific<OpenFileInfo>();
      assert(info != 0);
      int ret = close(info->fd);
      if(ret == -1) {
        log_disk.warning() << "file failed to close cleanly, disk contents may be corrupted";
      }
    }

    MemoryImpl::AllocationResult FileMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
									bool need_alloc_result,
									bool poisoned,
									TimeLimit work_until)
    {
      // we can't actually allocate anything in a FileMemory
      if(inst->metadata.ext_resource)
        log_inst.warning() << "attempt to register non-file resource: mem=" << me << " resource=" << *(inst->metadata.ext_resource);
      else
        log_inst.warning() << "attempt to allocate memory in file memory: layout=" << *(inst->metadata.layout);

      AllocationResult result = ALLOC_INSTANT_FAILURE;
      size_t inst_offset = 0;
      inst->notify_allocation(result, inst_offset, work_until);

      return result;
    }

    // release storage associated with an instance
    void FileMemory::release_storage_immediate(RegionInstanceImpl *inst,
					       bool poisoned,
					       TimeLimit work_until)
    {
      // nothing to do for a poisoned release
      if(poisoned)
	return;

      // for external instances, all we have to do is ack the destruction
      if(inst->metadata.ext_resource != 0) {
        unregister_external_resource(inst);
        inst->notify_deallocation();
	return;
      }

      // shouldn't get here - no allocation
      assert(0);
    }
}

