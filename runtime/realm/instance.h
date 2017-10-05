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

#include <realm/realm_config.h>

#include "lowlevel_config.h"

#include "event.h"
#include "memory.h"

#include "custom_serdez.h"

#include <vector>

// we need intptr_t - make it if needed
#if __cplusplus >= 201103L
#include <cstdint>
#else
typedef ptrdiff_t intptr_t;
#endif

namespace LegionRuntime {
  namespace Accessor {
    #define REGION_ACCESSOR_ALREADY_PROTOTYPED
    template <typename AT, typename ET = void, typename PT = ET> struct RegionAccessor;
    namespace AccessorType {
      struct Generic;
    };
  };
};

namespace Realm {

  typedef int FieldID;

  template <int N, typename T> struct Rect;
  template <int N, typename T> struct IndexSpace;
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
    //const LinearizedIndexSpaceIntfc& get_lis(void) const;
    const InstanceLayoutGeneric *get_layout(void) const;

    // these methods may be used to access the contents of an instance, but
    //  users are encouraged to use various accessors which make repeated
    //  accesses much more efficient

    void read_untyped(size_t offset, void *data, size_t datalen) const;
    void write_untyped(size_t offset, const void *data, size_t datalen) const;
    void reduce_apply_untyped(size_t offset, ReductionOpID redop_id,
			      const void *data, size_t datalen,
			      bool exclusive = false) const;
    void reduce_fold_untyped(size_t offset, ReductionOpID redop_id,
			     const void *data, size_t datalen,
			     bool exclusive = false) const;
    // returns a null pointer if the instance storage cannot be directly
    //  accessed via load/store instructions
    void *pointer_untyped(size_t offset, size_t datalen) const;

    // typed template helpers of the above
    template <typename T>
    T read(size_t offset) const;
    template <typename T>
    void write(size_t offset, T val) const;
    template <typename T>
    void reduce_apply(size_t offset, ReductionOpID redop_id, T val,
		      bool exclusive = false) const;
    template <typename T>
    void reduce_fold(size_t offset, ReductionOpID redop_id, T val,
		     bool exclusive = false) const;
    template <typename T>
    T *pointer(size_t offset) const;

    Event get_ready_event(void) const;

    // calls to create_instance return immediately with a handle, but also
    //  return an event that must be used as a precondition for any use (or
    //  destruction) of the instance
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 InstanceLayoutGeneric *ilg,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);

    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const IndexSpace<N,T>& space,
				 const std::vector<size_t>& field_sizes,
				 size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);

    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const IndexSpace<N,T>& space,
				 const std::map<FieldID, size_t>& field_sizes,
				 size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);

    // we'd like the methods above to accept a Rect<N,T> in place of the
    //  IndexSpace<N,T>, but that doesn't work unless the method template
    //  parameters are specified explicitly, so provide an overload that
    //  takes a Rect directly
    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const Rect<N,T>& rect,
				 const std::vector<size_t>& field_sizes,
				 size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);

    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const Rect<N,T>& rect,
				 const std::map<FieldID, size_t>& field_sizes,
				 size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);

    template <int N, typename T>
    static Event create_file_instance(RegionInstance& inst,
				      const char *file_name,
				      const IndexSpace<N,T>& space,
				      const std::vector<FieldID> &field_ids,
				      const std::vector<size_t> &field_sizes,
				      legion_lowlevel_file_mode_t file_mode,
				      const ProfilingRequestSet& prs,
				      Event wait_on = Event::NO_EVENT);

#ifdef USE_HDF
    template <int N, typename T>
    static Event create_hdf5_instance(RegionInstance& inst,
				      const char *file_name,
				      const IndexSpace<N,T>& space,
				      const std::vector<FieldID> &field_ids,
				      const std::vector<size_t> &field_sizes,
				      const std::vector<const char*> &field_files,
				      bool read_only,
				      const ProfilingRequestSet& prs,
				      Event wait_on = Event::NO_EVENT);
#endif

    void destroy(Event wait_on = Event::NO_EVENT) const;

    AddressSpace address_space(void) const;

    // apparently we can't use default template parameters on methods without C++11, but we
    //  can provide templates of two different arities...
    template <int N, typename T>
    IndexSpace<N,T> get_indexspace(void) const;

    template <int N>
    IndexSpace<N,int> get_indexspace(void) const;

    LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic> get_accessor(void) const;

    // used for accessor construction
    bool increment_accessor_count(void);
    bool decrement_accessor_count(void);

    struct DestroyedField {
    public:
      DestroyedField(void);
      DestroyedField(FieldID fid, unsigned s, CustomSerdezID sid);
    public:
      FieldID field_id;
      unsigned size;
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


