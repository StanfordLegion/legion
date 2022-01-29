/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include "realm/realm_c.h"

#include <stddef.h>
#include <iostream>

namespace Realm {

    typedef ::realm_address_space_t AddressSpace;

    class REALM_PUBLIC_API Memory {
    public:
      typedef ::realm_id_t id_t;
      id_t id;
      bool operator<(const Memory &rhs) const { return id < rhs.id; }
      bool operator==(const Memory &rhs) const { return id == rhs.id; }
      bool operator!=(const Memory &rhs) const { return id != rhs.id; }

      static const Memory NO_MEMORY;

      bool exists(void) const { return id != 0; }

      // Return the address space for this memory
      AddressSpace address_space(void) const;

      // Different Memory types (defined in realm_c.h)
      // can't just typedef the kind because of C/C++ enum scope rules
      enum Kind {
#define C_ENUMS(name, desc) name,
  REALM_MEMORY_KINDS(C_ENUMS)
#undef C_ENUMS
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

