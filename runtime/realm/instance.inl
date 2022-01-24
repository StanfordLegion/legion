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

// instances for Realm

#ifndef REALM_INSTANCE_INL
#define REALM_INSTANCE_INL

// nop, but helps IDEs
#include "realm/instance.h"

#include "realm/indexspace.h"
#include "realm/inst_layout.h"
#include "realm/serialize.h"
#include "realm/machine.h"

TYPE_IS_SERIALIZABLE(Realm::RegionInstance);
TYPE_IS_SERIALIZABLE(realm_file_mode_t);

namespace Realm {

  extern Logger log_inst;

  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance

  inline bool RegionInstance::operator<(const RegionInstance& rhs) const
  {
    return id < rhs.id;
  }

  inline bool RegionInstance::operator==(const RegionInstance& rhs) const
  {
    return id == rhs.id;
  }

  inline bool RegionInstance::operator!=(const RegionInstance& rhs) const
  {
    return id != rhs.id;
  }

  REALM_CUDA_HD
  inline bool RegionInstance::exists(void) const
  {
    return id != 0;
  }

  inline std::ostream& operator<<(std::ostream& os, RegionInstance r)
  {
    return os << std::hex << r.id << std::dec;
  }

  template <int N, typename T>
  inline IndexSpace<N,T> RegionInstance::get_indexspace(void) const
  {
    const InstanceLayout<N,T> *layout = checked_cast<const InstanceLayout<N,T> *>(this->get_layout());
    return layout->space;
  }
		
  template <int N>
  inline IndexSpace<N,int> RegionInstance::get_indexspace(void) const
  {
    return get_indexspace<N,int>();
  }

  template <typename T>
  inline T RegionInstance::read(size_t offset) const
  {
    T val;
    read_untyped(offset, &val, sizeof(T));
    return val;
  }

  template <typename T>
  inline void RegionInstance::write(size_t offset, T val) const
  {
    write_untyped(offset, &val, sizeof(T));
  }

  template <typename T>
  inline void RegionInstance::reduce_apply(size_t offset, ReductionOpID redop_id,
					   T val,
					   bool exclusive /*= false*/) const
  {
    reduce_apply_untyped(offset, redop_id, &val, sizeof(T), exclusive);
  }

  template <typename T>
  inline void RegionInstance::reduce_fold(size_t offset, ReductionOpID redop_id,
					  T val,
					  bool exclusive /*= false*/) const
  {
    reduce_fold_untyped(offset, redop_id, &val, sizeof(T), exclusive);
  }

  template <typename T>
  inline T *RegionInstance::pointer(size_t offset) const
  {
    return static_cast<T *>(pointer_untyped(offset, sizeof(T)));
  }
		
  template <int N, typename T>
  inline /*static*/ Event RegionInstance::create_instance(RegionInstance& inst,
							  Memory memory,
							  const IndexSpace<N,T>& space,
							  const std::vector<size_t> &field_sizes,
							  size_t block_size,
							  const ProfilingRequestSet& reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    // smoosh hybrid block sizes back to SOA for now
    if(block_size > 1)
      block_size = 0;
    InstanceLayoutConstraints ilc(field_sizes, block_size);
    // We use fortran order here
    int dim_order[N];
    for (int i = 0; i < N; i++)
      dim_order[i] = i;
    InstanceLayoutGeneric *layout = InstanceLayoutGeneric::choose_instance_layout<N,T>(space, ilc, dim_order);
    return create_instance(inst, memory, layout, reqs, wait_on);
  }

  template <int N, typename T>
  inline /*static*/ Event RegionInstance::create_instance(RegionInstance& inst,
							  Memory memory,
							  const IndexSpace<N,T>& space,
							  const std::map<FieldID, size_t> &field_sizes,
							  size_t block_size,
							  const ProfilingRequestSet& reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    // smoosh hybrid block sizes back to SOA for now
    if(block_size > 1)
      block_size = 0;
    InstanceLayoutConstraints ilc(field_sizes, block_size);
    // We use fortran order here
    int dim_order[N];
    for (int i = 0; i < N; i++)
      dim_order[i] = i;
    InstanceLayoutGeneric *layout = InstanceLayoutGeneric::choose_instance_layout<N,T>(space, ilc, dim_order);
    return create_instance(inst, memory, layout, reqs, wait_on);
  }

  // we'd like the methods above to accept a Rect<N,T> in place of the
  //  IndexSpace<N,T>, but that doesn't work unless the method template
  //  parameters are specified explicitly, so provide an overload that
  //  takes a Rect directly
  template <int N, typename T>
  inline /*static*/ Event RegionInstance::create_instance(RegionInstance& inst,
							  Memory memory,
							  const Rect<N,T>& rect,
							  const std::vector<size_t>& field_sizes,
							  size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
							  const ProfilingRequestSet& prs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    return RegionInstance::create_instance<N,T>(inst,
						memory,
						IndexSpace<N,T>(rect),
						field_sizes,
						block_size,
						prs,
						wait_on);
  }

  template <int N, typename T>
  inline /*static*/ Event RegionInstance::create_instance(RegionInstance& inst,
							  Memory memory,
							  const Rect<N,T>& rect,
							  const std::map<FieldID, size_t> &field_sizes,
							  size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
							  const ProfilingRequestSet& prs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    return RegionInstance::create_instance<N,T>(inst,
						memory,
						IndexSpace<N,T>(rect),
						field_sizes,
						block_size,
						prs,
						wait_on);
  }

  /*static*/ inline Event RegionInstance::create_external(RegionInstance& inst,
							  Memory memory, uintptr_t base,
							  InstanceLayoutGeneric *ilg,
							  const ProfilingRequestSet& prs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    // this interface doesn't give us a size or read-only ness, so get the size
    //  from the layout and assume it's read/write
    ExternalMemoryResource res(reinterpret_cast<void *>(base),
			       ilg->bytes_used);
    return create_external_instance(inst, memory, ilg, res, prs, wait_on);
  }

  template <int N, typename T>
  /*static*/ Event RegionInstance::create_file_instance(RegionInstance& inst,
							const char *file_name,
							const IndexSpace<N,T>& space,
							const std::vector<FieldID> &field_ids,
							const std::vector<size_t> &field_sizes,
							realm_file_mode_t file_mode,
							const ProfilingRequestSet& prs,
							Event wait_on /*= Event::NO_EVENT*/)
  {
    // this old interface assumes an SOA layout of fields in memory, starting at
    //  the beginning of the file
    InstanceLayoutConstraints ilc(field_ids, field_sizes, 0 /*SOA*/);
    int dim_order[N];
    for (int i = 0; i < N; i++)
      dim_order[i] = i;
    InstanceLayoutGeneric *ilg;
    ilg = InstanceLayoutGeneric::choose_instance_layout(space, ilc, dim_order);

    ExternalFileResource res(file_name, file_mode);
    return create_external_instance(inst,
				    res.suggested_memory(),
				    ilg, res, prs, wait_on);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance::DestroyedField


  inline RegionInstance::DestroyedField::DestroyedField(void) 
    : field_id(FieldID(-1)), size(0), serdez_id(0)
  { }

  inline RegionInstance::DestroyedField::DestroyedField(FieldID fid, unsigned s, CustomSerdezID sid)
    : field_id(fid), size(s), serdez_id(sid)
  { }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalInstanceResource

  template <typename S>
  inline bool serialize(S& serializer, const ExternalInstanceResource& res)
  {
    return Serialization::PolymorphicSerdezHelper<ExternalInstanceResource>::serialize(serializer, res);
  }

  template <typename S>
  /*static*/ inline ExternalInstanceResource *ExternalInstanceResource::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<ExternalInstanceResource>::deserialize_new(deserializer);
  }

  inline std::ostream& operator<<(std::ostream& os, const ExternalInstanceResource& res)
  {
    res.print(os);
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalMemoryResource

  template <typename S>
  bool ExternalMemoryResource::serialize(S& s) const
  {
    return ((s << base) &&
	    (s << size_in_bytes) &&
	    (s << read_only));
  }

  template <typename S>
  /*static*/ ExternalInstanceResource *ExternalMemoryResource::deserialize_new(S& s)
  {
    ExternalMemoryResource *res = new ExternalMemoryResource;
    if((s >> res->base) &&
       (s >> res->size_in_bytes) &&
       (s >> res->read_only)) {
      return res;
    } else {
      delete res;
      return 0;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalFileResource

  template <typename S>
  bool ExternalFileResource::serialize(S& s) const
  {
    return ((s << filename) &&
	    (s << mode) &&
	    (s << offset));
  }

  template <typename S>
  /*static*/ ExternalInstanceResource *ExternalFileResource::deserialize_new(S& s)
  {
    ExternalFileResource *res = new ExternalFileResource;
    if((s >> res->filename) &&
       (s >> res->mode) &&
       (s >> res->offset)) {
      return res;
    } else {
      delete res;
      return 0;
    }
  }


}; // namespace Realm  

#endif // ifndef REALM_INSTANCE_INL
