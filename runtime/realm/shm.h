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

#ifndef REALM_SHM_H
#define REALM_SHM_H
#include "realm/realm_config.h"
#include "realm/utils.h"
#include <string>

namespace Realm {

  /// @brief Holds a reference to a mapped shared memory region and all the information
  /// used to create/open it.  Such shared memory region can be shared with other
  /// processes on the same system using a communication mechanism such as via unix domain
  /// sockets, win32 DuplicateHandle, or if names are used, transferring the names to the
  /// processes and opening them (\sa SharedMemoryInfo::open)
  class SharedMemoryInfo {
  public:
    SharedMemoryInfo(void);

    /// @brief On destruction, the SharedMemoryInfo will unmap the memory associated with
    /// this shared region, if any, and unlink it from the file system if it owns it
    ~SharedMemoryInfo(void);

    // disable copying of SharedMemoryInfo
    SharedMemoryInfo(const SharedMemoryInfo &) = delete;
    SharedMemoryInfo &operator=(const SharedMemoryInfo &) = delete;

    /// @name Accessors
    /// @{
    template <typename T>
    T *get_ptr() const
    {
      return reinterpret_cast<T *>(base);
    }
    OsHandle get_handle() const { return handle; }
    size_t get_size() const { return size; }
    std::string get_name() const { return name; }
    explicit operator bool() const { return base != nullptr; }
    /// @}

    /// @brief Removes the name from the shared memory region such that it can't be
    /// opened by it's name any more and closes the associated handle
    /// @note not applicable to windows
    void unlink(void);

    /// @brief Creates a shared memory region accessible via \p handle
    /// @note It is preferred that the name is null here as this creates an anonymous
    /// backing store that requires processor-to-processor communication.
    /// @param[out] info Shared memory object containing the base pointer and size
    /// @param name Name associated with the shared memory allocation, passed to
    /// \sa SharedMemoryInfo::open in another process.
    /// @param size Size of the shared memory region to allocate
    /// @param numa_node NUMA node to allocate the pages on
    /// @return True on success, false otherwise
    static bool create(SharedMemoryInfo &info, size_t size, const char *name = nullptr,
                       int numa_node = -1);

    /// @brief Opens a previously created shared memory region given it's \p name and \p
    /// size
    /// @param[out] info Shared memory object containing the base pointer and size
    /// @param name Name of the shared memory region (to be retrieved from
    /// \sa SharedMemoryInfo::create in another process)
    /// @param size Size of the shared memory region to allocate
    /// @return True on success, false otherwise
    static bool open(SharedMemoryInfo &info, const std::string &name, size_t size);

    /// @brief Opens a previously created shared memory region given it's associated \p
    /// handle and \p size. \sa Realm::close_handle
    /// @param[out] info
    /// @param handle OS handle of the associated memory object (retrieved from \sa
    /// SharedMemoryInfo::create)
    /// @param size Size of the shared memory region to allocate
    /// @return True on success, false otherwise
    static bool open(SharedMemoryInfo &info, OsHandle handle, size_t size);

  private:
    std::string name; // name of the shared memory object, if applicable
    void *base;       // base address of the mapped memory object
    size_t size;      // Size of the mapped memory object
    OsHandle handle;  // OS-native handle to mapped memory object
    bool owner;       // True if this SharedMemoryInfo was created via
                      // \sa SharedMemoryInfo::create, false otherwise
  };

}; // namespace Realm

#endif // ifndef REALM_SHM_H
