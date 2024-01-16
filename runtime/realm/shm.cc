/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#include "realm/shm.h"
#include "realm/logging.h"
#include "realm/timers.h"

#ifdef REALM_ON_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include "windows.h"
#else
#include "realm/numa/numasysif.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if !defined(MFD_CLOEXEC)
#define MFD_CLOEXEC 0x0001U
#endif

namespace Realm {

  Logger log_shm("shm");

#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
  // Helper function for mapping and sizing up the shared memory region, and migrating to
  // the correct numa node
  static bool setup_shm(void *&base, Realm::OsHandle hdl, size_t sz, int numa_node)
  {
#if defined(REALM_HAS_POSIX_FALLOCATE64)
    if(posix_fallocate64(hdl, 0, sz) != 0)
#else
    if(ftruncate(hdl, sz) != 0)
#endif
    {
      log_shm.info("failed to resize shared memory region: %s", realm_strerror(errno));
      return false;
    }
    base = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, hdl, 0);
    if(base == nullptr) {
      log_shm.info("Failed to map shared memory: %s", realm_strerror(errno));
      return false;
    }
    if(numa_node >= 0) {
      (void)numasysif_bind_mem(numa_node, base, sz, false);
    }
    return base != nullptr;
  }
#endif

  /*static*/ bool SharedMemoryInfo::create(SharedMemoryInfo &info, size_t size,
                                           const char *name /* = nullptr*/,
                                           int numa_node /* = -1*/)
  {
    info.unlink();
    info.size = size;
    info.owner = true;
    info.base = nullptr;
    info.handle = Realm::INVALID_OS_HANDLE;
    if(name != nullptr) {
      info.name = name;
    }
#if defined(REALM_ON_WINDOWS)
    numa_node = numa_node < 0 ? NUMA_NO_PREFERRED_NODE : numa_node;
    HANDLE hMapFile = CreateFileMappingNuma(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (size >> 32ULL),
        (size & ~(1ULL << 32ULL)), info.name.c_str(), numa_node);
    if(hMapFile == nullptr) {
      log_shm.info("Failed to create shm %s", info.name.c_str());
      return false;
    }
    info.base = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if(info.base == nullptr) {
      log_shm.info("Failed to map shared memory");
      CloseHandle(hMapFile);
      return false;
    }
    info.handle = hMapFile;
    return true;
#elif defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
    OsHandle fd = INVALID_OS_HANDLE;
    if(name != nullptr) {
      // Create named shared memory
      std::string path = '/' + info.name;
      fd = shm_open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IXUSR);
    } else {
      // Create anonymous shared memory
      // Try memfd via syscall (as some libc implementations do not provide wrappers
      // yet...)
#if defined(SYS_memfd_create)
      fd = syscall(SYS_memfd_create, "", MFD_CLOEXEC);
#endif
      if(fd < 0) {
        // There's no real portable way across *nixes to share anonymous memory by fd
        // without a file name.  There are some distro specific extensions (SHM_ANON,
        // shm_mkstemp, etC), but none of these are standard.  Therefore, we'll just try a
        // few names and see if they work.
        static const size_t REALM_SHM_MAX_TRIES = 1;
        for(size_t i = 0; i < REALM_SHM_MAX_TRIES; i++) {
          std::string tmp_name("/realm-shm.");
          tmp_name += std::to_string(Clock::current_time_in_nanoseconds(true));
          if(SharedMemoryInfo::create(info, size, tmp_name.c_str(), numa_node)) {
            return true;
          }
        }
        log_shm.info("Failed to find unique shm name");
        return false;
      }
    }

    if(fd < 0) {
      log_shm.info("Failed to create shm %s: %s", name, realm_strerror(errno));
      return false;
    }

    if(!setup_shm(info.base, fd, size, numa_node)) {
      if(!info.name.empty()) {
        std::string path = '/' + info.name;
        shm_unlink(path.c_str());
      }
      close(fd);
      return false;
    }
    log_shm.spew() << "Created " << info.name;
    info.handle = fd;
    return true;
#endif
  }

  /*static*/ bool SharedMemoryInfo::open(SharedMemoryInfo &info, const std::string &name,
                                         size_t size)
  {
    info.size = size;
    info.name = name;
    info.owner = false;
    info.base = nullptr;
    info.handle = Realm::INVALID_OS_HANDLE;
#if defined(REALM_ON_WINDOWS)
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
    if(hMapFile == nullptr) {
      return false;
    }
    info.base = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if(info.base != nullptr) {
      CloseHandle(hMapFile);
      return false;
    }
    info.handle = hMapFile;
    return true;
#elif defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
    std::string path = '/' + name;
    int fd = shm_open(path.c_str(), O_RDWR, 0);
    if(fd < 0) {
      log_shm.info("Failed to open shm: %s", realm_strerror(errno));
      return false;
    }
    info.base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(info.base == nullptr) {
      log_shm.info("Failed to map shm: %s", realm_strerror(errno));
      close(fd);
      return false;
    }
    info.handle = fd;
    return true;
#else
    return false;
#endif
  }

  /*static*/ bool SharedMemoryInfo::open(SharedMemoryInfo &info, OsHandle handle,
                                         size_t size)
  {
    info.size = size;
    info.name.clear();
    info.owner = false;
    info.base = nullptr;
    info.handle = handle;
#if defined(REALM_ON_WINDOWS)
    info.base = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
#elif defined(REALM_ON_LINUX)
    info.base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, handle, 0);
#endif
    return info.base != nullptr;
  }

  SharedMemoryInfo::SharedMemoryInfo(void)
    : name()
    , base(nullptr)
    , size(0)
    , handle(Realm::INVALID_OS_HANDLE)
    , owner(false)
  {}
  SharedMemoryInfo::SharedMemoryInfo(SharedMemoryInfo&& other)
    : name(std::move(other.name))
    , base(nullptr)
    , size(0)
    , handle(Realm::INVALID_OS_HANDLE)
    , owner(false)
  {
    // Swap the currently empty newly constructed shared memory info with that of the
    // other, leaving other empty and the newly constructed SharedMemoryInfo as the
    // owner of the resource
    std::swap(base, other.base);
    std::swap(size, other.size);
    std::swap(handle, other.handle);
    std::swap(owner, other.owner);
  }

  SharedMemoryInfo& SharedMemoryInfo::operator=(SharedMemoryInfo&& other) {
    if (this != &other) {
      // Release whatever's currently in this shared memory info
      unlink();

      // Copy over the shared memory information
      name = other.name;
      base = other.base;
      size = other.size;
      handle = other.handle;
      owner = other.owner;

      // And clear out the other's shared memory infomation
      other.name.clear();
      other.base = nullptr;
      other.size = 0;
      other.handle = Realm::INVALID_OS_HANDLE;
      other.owner = false;
    }
    return *this;
  }

  SharedMemoryInfo::~SharedMemoryInfo(void)
  {
    if(base == nullptr) {
      return;
    }
    log_shm.spew("Unmapping shm %s: %p:%llu", name.c_str(), base,
                 (unsigned long long)size);
    unlink();
#if defined(REALM_ON_WINDOWS)
    UnmapViewOfFile(base);
#elif defined(REALM_ON_LINUX)
    munmap(base, size);
#endif
    base = nullptr;
  }

  void SharedMemoryInfo::unlink(void)
  {
    if(handle != Realm::INVALID_OS_HANDLE) {
      close_handle(handle);
      handle = Realm::INVALID_OS_HANDLE;
    }
    if(owner && !name.empty()) {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
      log_shm.spew() << "Unlinking shm " << name;
      std::string path = '/' + name;
      shm_unlink(path.c_str());
#endif
      name.clear();
    }
  }

} // namespace Realm
