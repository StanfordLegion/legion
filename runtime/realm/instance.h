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

#ifndef REALM_INSTANCE_H
#define REALM_INSTANCE_H

#include "realm/realm_config.h"

#include "realm/realm_c.h"

#include "realm/event.h"
#include "realm/memory.h"
#include "realm/processor.h"
#include "realm/point.h"

#include "realm/custom_serdez.h"

#include <vector>

// we need intptr_t - make it if needed
#if REALM_CXX_STANDARD >= 11
#include <stdint.h>
#else
typedef ptrdiff_t intptr_t;
#endif

namespace Realm {

  typedef int FieldID;

  template <int N, typename T> struct Rect;
  template <int N, typename T> struct IndexSpace;
  class IndexSpaceGeneric;
  class LinearizedIndexSpaceIntfc;
  class InstanceLayoutGeneric;
  class ProfilingRequestSet;
  class ExternalInstanceResource;

  namespace PieceLookup {
    struct Instruction;
  };

  class REALM_PUBLIC_API RegionInstance {
  public:
    typedef ::realm_id_t id_t;
    id_t id;
    bool operator<(const RegionInstance &rhs) const;
    bool operator==(const RegionInstance &rhs) const;
    bool operator!=(const RegionInstance &rhs) const;

    static const RegionInstance NO_INST;

    REALM_CUDA_HD
    bool exists(void) const;

    Memory get_location(void) const;
    //const LinearizedIndexSpaceIntfc& get_lis(void) const;
    const InstanceLayoutGeneric *get_layout(void) const;

    // gets a compiled piece lookup program for a given field
    template <int N, typename T>
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    const PieceLookup::Instruction *get_lookup_program(FieldID field_id,
						       unsigned allowed_mask,
						       uintptr_t& field_offset);
    template <int N, typename T>
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    const PieceLookup::Instruction *get_lookup_program(FieldID field_id,
						       const Rect<N,T>& subrect,
						       unsigned allowed_mask,
						       uintptr_t& field_offset);

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

    // calls to create_instance return immediately with a handle, but also
    //  return an event that must be used as a precondition for any use (or
    //  destruction) of the instance
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 InstanceLayoutGeneric *ilg,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);

    // creates an instance that is backed by an external resource - Realm
    //  performs no allocation, but allows access and copies as with normal
    //  instances
    static Event create_external_instance(RegionInstance& inst,
					  Memory memory,
					  InstanceLayoutGeneric *ilg,
					  const ExternalInstanceResource& resource,
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
    REALM_ATTR_DEPRECATED("use RegionInstance::create_external_instance instead",
    static Event create_file_instance(RegionInstance& inst,
				      const char *file_name,
				      const IndexSpace<N,T>& space,
				      const std::vector<FieldID> &field_ids,
				      const std::vector<size_t> &field_sizes,
				      realm_file_mode_t file_mode,
				      const ProfilingRequestSet& prs,
				      Event wait_on = Event::NO_EVENT));

#ifdef REALM_USE_HDF5
    template <int N, typename T>
    struct HDF5FieldInfo {
      FieldID field_id;
      size_t field_size;
      std::string dataset_name;
      Point<N,T> offset;
      int dim_order[N];
    };

    REALM_ATTR_DEPRECATED2("use RegionInstance::create_external_instance instead",
    template <int N, typename T>
    static Event create_hdf5_instance(RegionInstance& inst,
				      const char *file_name,
				      const IndexSpace<N,T>& space,
				      const std::vector<HDF5FieldInfo<N,T> >& field_infos,
				      bool read_only,
				      const ProfilingRequestSet& prs,
				      Event wait_on = Event::NO_EVENT));
#endif
              
    REALM_ATTR_DEPRECATED("use RegionInstance::create_external_instance instead",
    static Event create_external(RegionInstance& inst,
				 Memory memory, uintptr_t base,
				 InstanceLayoutGeneric *ilg,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT));

    void destroy(Event wait_on = Event::NO_EVENT) const;

    AddressSpace address_space(void) const;

    // before you can get an instance's index space or construct an accessor for
    //  a given processor, the necessary metadata for the instance must be
    //  available on to that processor
    // this can require network communication and/or completion of the actual
    //  allocation, so an event is returned and (as always) the application
    //  must decide when/where to handle this precondition
    Event fetch_metadata(Processor target) const;

    // apparently we can't use default template parameters on methods without C++11, but we
    //  can provide templates of two different arities...
    template <int N, typename T>
    IndexSpace<N,T> get_indexspace(void) const;

    template <int N>
    IndexSpace<N,int> get_indexspace(void) const;

    // used for accessor construction
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    bool increment_accessor_count(void);
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    bool decrement_accessor_count(void);

    // it is sometimes useful to re-register an existing instance (in whole or
    //  in part) as an "external" instance (e.g. to provide a different view
    //  on the same bits) - this hopefully gives an ExternalInstanceResource *
    //  (which must be deleted by the caller) that corresponds to this instance
    //  but may return a null pointer for instances that do not support
    //  re-registration
    ExternalInstanceResource *generate_resource_info(bool read_only) const;

    // a version of the above that allows limiting the described memory to just
    //  a subset of the original instance (NOTE: this will accept any
    //  IndexSpace<N,T> at compile-time, but N,T must match the instance layout)
    ExternalInstanceResource *generate_resource_info(const IndexSpaceGeneric& space,
						     span<const FieldID> fields,
						     bool read_only) const;

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

  REALM_PUBLIC_API
  std::ostream& operator<<(std::ostream& os, RegionInstance r);


  class REALM_PUBLIC_API ExternalInstanceResource {
  protected:
    // only subclasses can be constructed
    ExternalInstanceResource();
    
  public:
    virtual ~ExternalInstanceResource();

    // returns the suggested memory in which this resource should be created
    virtual Memory suggested_memory() const = 0;

    virtual ExternalInstanceResource *clone(void) const = 0;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

    // pretty-printing
    friend std::ostream& operator<<(std::ostream& os, const ExternalInstanceResource& res);

  protected:
    virtual void print(std::ostream& os) const = 0;
  };

  template <typename S>
  bool serialize(S& serializer, const ExternalInstanceResource& res);


  class REALM_PUBLIC_API ExternalMemoryResource : public ExternalInstanceResource {
  public:
    ExternalMemoryResource(uintptr_t _base, size_t _size_in_bytes, bool _read_only);
    ExternalMemoryResource(void *_base, size_t _size_in_bytes);
    ExternalMemoryResource(const void *_base, size_t _size_in_bytes);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalMemoryResource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalMemoryResource> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    uintptr_t base;
    size_t size_in_bytes;
    bool read_only;
  };

  class REALM_PUBLIC_API ExternalFileResource : public ExternalInstanceResource {
  public:
    ExternalFileResource(const std::string& _filename, realm_file_mode_t _mode,
			 size_t _offset = 0);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalFileResource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalFileResource> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    std::string filename;
    size_t offset;
    realm_file_mode_t mode;
  };

}; // namespace Realm
#endif // ifndef REALM_INSTANCE_H

#ifndef REALM_SKIP_INLINES
#include "realm/instance.inl"
#endif


