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

// instances for Realm

#ifndef REALM_INSTANCE_H
#define REALM_INSTANCE_H

#include "lowlevel_config.h"

#include "event.h"
#include "memory.h"

#include "accessor.h"
#include "custom_serdez.h"

#include <vector>

namespace Realm {

    class RegionInstance {
    public:
      typedef ::legion_lowlevel_id_t id_t;
      id_t id;
      bool operator<(const RegionInstance &rhs) const { return id < rhs.id; }
      bool operator==(const RegionInstance &rhs) const { return id == rhs.id; }
      bool operator!=(const RegionInstance &rhs) const { return id != rhs.id; }

      static const RegionInstance NO_INST;

      bool exists(void) const { return id != 0; }

      Memory get_location(void) const;

      void destroy(Event wait_on = Event::NO_EVENT) const;

      struct DestroyedField {
      public:
        DestroyedField(void) 
          : offset(0), size(0), serdez_id(0) { }
        DestroyedField(unsigned o, unsigned s, CustomSerdezID sid)
          : offset(o), size(s), serdez_id(sid) { }
      public:
	unsigned offset, size;
	CustomSerdezID serdez_id;
      };

      // if any fields in the instance need custom destruction, use this version
      void destroy(const std::vector<DestroyedField>& destroyed_fields,
		   Event wait_on = Event::NO_EVENT) const;

      AddressSpace address_space(void) const;

      LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic> get_accessor(void) const;

      void report_instance_fault(int reason,
				 const void *reason_data, size_t reason_size) const;
    };

    inline std::ostream& operator<<(std::ostream& os, RegionInstance r) { return os << std::hex << r.id << std::dec; }
		
}; // namespace Realm

//include "instance.inl"

#endif // ifndef REALM_INSTANCE_H

