/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// we need intptr_t - make it if needed
#if __cplusplus >= 201103L
#include <cstdint>
#else
typedef ptrdiff_t intptr_t;
#endif

namespace Realm {

  template <int N, typename T> struct ZIndexSpace;
  class LinearizedIndexSpaceIntfc;
  class InstanceLayoutGeneric;
  class ProfilingRequestSet;

  class RegionInstance {
  public:
    typedef ::legion_lowlevel_id_t id_t;
    id_t id;
    bool operator<(const RegionInstance &rhs) const;
    bool operator==(const RegionInstance &rhs) const;
    bool operator!=(const RegionInstance &rhs) const;

    static const RegionInstance NO_INST;

    bool exists(void) const;

    Memory get_location(void) const;
    const LinearizedIndexSpaceIntfc& get_lis(void) const;

    static RegionInstance create_instance(Memory memory,
					  InstanceLayoutGeneric *ilg,
					  const ProfilingRequestSet& prs);

    template <int N, typename T>
    static RegionInstance create_instance(Memory memory,
					  const ZIndexSpace<N,T>& space,
					  const std::vector<size_t>& field_sizes,
					  const ProfilingRequestSet& prs);

    template <int N, typename T>
    static RegionInstance create_file_instance(const char *file_name,
                                          const ZIndexSpace<N,T>& space,
                                          const std::vector<size_t> &field_sizes,
                                          legion_lowlevel_file_mode_t file_mode,
                                          const ProfilingRequestSet& prs);
    template <int N, typename T>
    static RegionInstance create_hdf5_instance(const char *file_name,
                                          const ZIndexSpace<N,T>& space,
                                          const std::vector<size_t> &field_sizes,
                                          const std::vector<const char*> &field_files,
                                          bool read_only,
                                          const ProfilingRequestSet& prs);

    void destroy(Event wait_on = Event::NO_EVENT) const;

    AddressSpace address_space(void) const;

    // apparently we can't use default template parameters on methods without C++11, but we
    //  can provide templates of two different arities...
    template <int N, typename T>
    ZIndexSpace<N,T> get_indexspace(void) const;

    template <int N>
    ZIndexSpace<N,int> get_indexspace(void) const;

    LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic> get_accessor(void) const;

    // used for accessor construction
    bool increment_accessor_count(void);
    bool decrement_accessor_count(void);

    struct DestroyedField {
    public:
      DestroyedField(void);
      DestroyedField(unsigned o, unsigned s, CustomSerdezID sid);
    public:
      unsigned offset, size;
      CustomSerdezID serdez_id;
    };

    // if any fields in the instance need custom destruction, use this version
    void destroy(const std::vector<DestroyedField>& destroyed_fields,
		 Event wait_on = Event::NO_EVENT) const;

    bool can_get_strided_access_parameters(size_t start, size_t count,
					   ptrdiff_t field_offset, size_t field_size);
    void get_strided_access_parameters(size_t start, size_t count,
				       ptrdiff_t field_offset, size_t field_size,
                                       intptr_t& base, ptrdiff_t& stride);

    void report_instance_fault(int reason,
			       const void *reason_data, size_t reason_size) const;
  };

  std::ostream& operator<<(std::ostream& os, RegionInstance r);
		
}; // namespace Realm
#endif // ifndef REALM_INSTANCE_H

#ifndef REALM_SKIP_INLINES
#include "instance.inl"
#endif


