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

// memorys for Realm

#ifndef REALM_MEMORY_H
#define REALM_MEMORY_H

#include "lowlevel_config.h"

#include <stddef.h>
#include <iostream>

namespace Realm {

    typedef ::legion_lowlevel_address_space_t AddressSpace;

    class Memory {
    public:
      typedef ::legion_lowlevel_id_t id_t;
      id_t id;
      bool operator<(const Memory &rhs) const { return id < rhs.id; }
      bool operator==(const Memory &rhs) const { return id == rhs.id; }
      bool operator!=(const Memory &rhs) const { return id != rhs.id; }

      static const Memory NO_MEMORY;

      bool exists(void) const { return id != 0; }

      // Return the address space for this memory
      AddressSpace address_space(void) const;
      // Return the local ID within the address space
      id_t local_id(void) const;

      // Different Memory types
      enum Kind {
        GLOBAL_MEM, // Guaranteed visible to all processors on all nodes (e.g. GASNet memory, universally slow)
        SYSTEM_MEM, // Visible to all processors on a node
        REGDMA_MEM, // Registered memory visible to all processors on a node, can be a target of RDMA
        SOCKET_MEM, // Memory visible to all processors within a node, better performance to processors on same socket 
        Z_COPY_MEM, // Zero-Copy memory visible to all CPUs within a node and one or more GPUs 
        GPU_FB_MEM,   // Framebuffer memory for one GPU and all its SMs
        DISK_MEM,   // Disk memory visible to all processors on a node
        HDF_MEM,    // HDF memory visible to all processors on a node
        FILE_MEM,   // file memory visible to all processors on a node
        LEVEL3_CACHE, // CPU L3 Visible to all processors on the node, better performance to processors on same socket 
        LEVEL2_CACHE, // CPU L2 Visible to all processors on the node, better performance to one processor
        LEVEL1_CACHE, // CPU L1 Visible to all processors on the node, better performance to one processor
      };

      // Return what kind of memory this is
      Kind kind(void) const;
      // Return the maximum capacity of this memory
      size_t capacity(void) const;

      // reports a problem with a memory in general (this is primarily for fault injection)
      void report_memory_fault(int reason,
			       const void *reason_data, size_t reason_size) const;
    };

    inline std::ostream& operator<<(std::ostream& os, Memory m) { return os << std::hex << m.id << std::dec; }
	
}; // namespace Realm

//include "memory.inl"

#endif // ifndef REALM_MEMORY_H

