/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "legion/api/data_impl.h"
#include "legion/api/geometry.h"
#include "legion/nodes/field.h"
#include "legion/utilities/hasher.h"
#include "legion/utilities/provenance.h"

namespace Legion {

  // Make sure all the handle types are trivially copyable.
  static_assert(
      std::is_trivially_copyable<IndexSpace>::value,
      "IndexSpace is not trivially copyable");
  static_assert(
      std::is_trivially_copyable<IndexPartition>::value,
      "IndexPartition is not trivially copyable");
  static_assert(
      std::is_trivially_copyable<FieldSpace>::value,
      "FieldSpace is not trivially copyable");
  static_assert(
      std::is_trivially_copyable<LogicalRegion>::value,
      "LogicalRegion is not trivially copyable");
  static_assert(
      std::is_trivially_copyable<LogicalPartition>::value,
      "LogicalPartition is not trivially copyable");
#define DIMFUNC(DIM)                                              \
  static_assert(                                                  \
      std::is_trivially_copyable<IndexSpaceT<DIM> >::value,       \
      "IndexSpaceT is not trivially copyable");                   \
  static_assert(                                                  \
      std::is_trivially_copyable<IndexPartitionT<DIM> >::value,   \
      "IndexPartitionT is not trivially copyable");               \
  static_assert(                                                  \
      std::is_trivially_copyable<LogicalRegionT<DIM> >::value,    \
      "LogicalRegionT is not trivially copyable");                \
  static_assert(                                                  \
      std::is_trivially_copyable<LogicalPartitionT<DIM> >::value, \
      "LogicalPartitionT is not trivially copyable");
  LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  // These types should be densely packed with no padding
  static_assert(std::has_unique_object_representations_v<IndexSpace>);
  static_assert(std::has_unique_object_representations_v<IndexPartition>);
  static_assert(std::has_unique_object_representations_v<FieldSpace>);
  static_assert(std::has_unique_object_representations_v<LogicalRegion>);
  static_assert(std::has_unique_object_representations_v<LogicalPartition>);

  const LogicalRegion LogicalRegion::NO_REGION = LogicalRegion();
  const LogicalPartition LogicalPartition::NO_PART = LogicalPartition();
  const Domain Domain::NO_DOMAIN = Domain();

  /////////////////////////////////////////////////////////////
  // IndexSpace
  /////////////////////////////////////////////////////////////

  /*static*/ const IndexSpace IndexSpace::NO_SPACE = IndexSpace();

  //--------------------------------------------------------------------------
  IndexSpace::IndexSpace(DistributedID _id, IndexTreeID _tid, TypeTag _tag)
    : did(_id), tid(_tid), type_tag(_tag)
  //--------------------------------------------------------------------------
  {
    legion_assert(
        (LEGION_DISTRIBUTED_HELP_DECODE(did) ==
         Internal::INDEX_SPACE_NODE_DC) ||
        (did == 0));
  }

  //--------------------------------------------------------------------------
  IndexSpace::IndexSpace(void) : did(0), tid(0), type_tag(0)
  //--------------------------------------------------------------------------
  {
    // Should be able to be densely packed
    static_assert(
        (sizeof(did) + sizeof(tid) + sizeof(type_tag)) == sizeof(*this));
  }

  //--------------------------------------------------------------------------
  bool IndexSpace::valid(void) const
  //--------------------------------------------------------------------------
  {
    return (
        !exists() ||
        (LEGION_DISTRIBUTED_HELP_DECODE(did) == Internal::INDEX_SPACE_NODE_DC));
  }

  /////////////////////////////////////////////////////////////
  // IndexPartition
  /////////////////////////////////////////////////////////////

  /*static*/ const IndexPartition IndexPartition::NO_PART = IndexPartition();

  //--------------------------------------------------------------------------
  IndexPartition::IndexPartition(
      DistributedID _id, IndexTreeID _tid, TypeTag _tag)
    : did(_id), tid(_tid), type_tag(_tag)
  //--------------------------------------------------------------------------
  {
    legion_assert(
        (LEGION_DISTRIBUTED_HELP_DECODE(did) == Internal::INDEX_PART_NODE_DC) ||
        (did == 0));
  }

  //--------------------------------------------------------------------------
  IndexPartition::IndexPartition(void) : did(0), tid(0), type_tag(0)
  //--------------------------------------------------------------------------
  {
    // Should be able to be densely packed
    static_assert(
        (sizeof(did) + sizeof(tid) + sizeof(type_tag)) == sizeof(*this));
  }

  //--------------------------------------------------------------------------
  bool IndexPartition::valid(void) const
  //--------------------------------------------------------------------------
  {
    return (
        !exists() ||
        (LEGION_DISTRIBUTED_HELP_DECODE(did) == Internal::INDEX_PART_NODE_DC));
  }

  /////////////////////////////////////////////////////////////
  // FieldSpace
  /////////////////////////////////////////////////////////////

  /*static*/ const FieldSpace FieldSpace::NO_SPACE = FieldSpace();

  //--------------------------------------------------------------------------
  FieldSpace::FieldSpace(DistributedID _id) : did(_id)
  //--------------------------------------------------------------------------
  {
    legion_assert(
        (LEGION_DISTRIBUTED_HELP_DECODE(did) == Internal::FIELD_SPACE_DC) ||
        (did == 0));
  }

  //--------------------------------------------------------------------------
  FieldSpace::FieldSpace(void) : did(0)
  //--------------------------------------------------------------------------
  {
    // Make sure field spaces can be densely packed
    static_assert(sizeof(did) == sizeof(*this));
  }

  //--------------------------------------------------------------------------
  bool FieldSpace::valid(void) const
  //--------------------------------------------------------------------------
  {
    return (
        !exists() ||
        (LEGION_DISTRIBUTED_HELP_DECODE(did) == Internal::FIELD_SPACE_DC));
  }

  /////////////////////////////////////////////////////////////
  // Logical Region
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  LogicalRegion::LogicalRegion(
      DistributedID tid, IndexSpace index, FieldSpace field)
    : tree_did(tid), index_space(index), field_space(field)
  //--------------------------------------------------------------------------
  {
    legion_assert(
        (LEGION_DISTRIBUTED_HELP_DECODE(tid) ==
         Internal::REGION_TREE_NODE_DC) ||
        (tid == 0));
  }

  //--------------------------------------------------------------------------
  LogicalRegion::LogicalRegion(void)
    : tree_did(0), index_space(IndexSpace::NO_SPACE),
      field_space(FieldSpace::NO_SPACE)
  //--------------------------------------------------------------------------
  {
    // Make sure this can be densely packed
    static_assert(
        (sizeof(tree_did) + sizeof(index_space) + sizeof(field_space)) ==
        sizeof(*this));
  }

  //--------------------------------------------------------------------------
  std::size_t LogicalRegion::hash(void) const
  //--------------------------------------------------------------------------
  {
    Internal::Murmur3Hasher hasher;
    hasher.hash(tree_did);
    hasher.hash(index_space.hash());
    hasher.hash(field_space.hash());
    uint64_t result[2];
    hasher.finalize(result);
    return result[0] ^ result[1];
  }

  //--------------------------------------------------------------------------
  bool LogicalRegion::valid(void) const
  //--------------------------------------------------------------------------
  {
    return (
        !exists() || ((LEGION_DISTRIBUTED_HELP_DECODE(tree_did) ==
                       Internal::REGION_TREE_NODE_DC) &&
                      index_space.valid() && field_space.valid()));
  }

  /////////////////////////////////////////////////////////////
  // Logical Partition
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  LogicalPartition::LogicalPartition(
      DistributedID tid, IndexPartition pid, FieldSpace field)
    : tree_did(tid), index_partition(pid), field_space(field)
  //--------------------------------------------------------------------------
  {
    legion_assert(
        (LEGION_DISTRIBUTED_HELP_DECODE(tid) ==
         Internal::REGION_TREE_NODE_DC) ||
        (tid == 0));
  }

  //--------------------------------------------------------------------------
  LogicalPartition::LogicalPartition(void)
    : tree_did(0), index_partition(IndexPartition::NO_PART),
      field_space(FieldSpace::NO_SPACE)
  //--------------------------------------------------------------------------
  {
    // Make sure this can be densely packed
    static_assert(
        (sizeof(tree_did) + sizeof(index_partition) + sizeof(field_space)) ==
        sizeof(*this));
  }

  //--------------------------------------------------------------------------
  std::size_t LogicalPartition::hash(void) const
  //--------------------------------------------------------------------------
  {
    Internal::Murmur3Hasher hasher;
    hasher.hash(tree_did);
    hasher.hash(index_partition.hash());
    hasher.hash(field_space.hash());
    uint64_t result[2];
    hasher.finalize(result);
    return result[0] ^ result[1];
  }

  //--------------------------------------------------------------------------
  bool LogicalPartition::valid(void) const
  //--------------------------------------------------------------------------
  {
    return (
        !exists() || ((LEGION_DISTRIBUTED_HELP_DECODE(tree_did) ==
                       Internal::REGION_TREE_NODE_DC) &&
                      index_partition.valid() && field_space.valid()));
  }

  /////////////////////////////////////////////////////////////
  // Field Allocator
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  FieldAllocator::FieldAllocator(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  FieldAllocator::FieldAllocator(const FieldAllocator& rhs) : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  FieldAllocator::FieldAllocator(FieldAllocator&& rhs) noexcept : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    rhs.impl = nullptr;
  }

  //--------------------------------------------------------------------------
  FieldAllocator::~FieldAllocator(void)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
      impl = nullptr;
    }
  }

  //--------------------------------------------------------------------------
  FieldAllocator::FieldAllocator(Internal::FieldAllocatorImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  FieldAllocator& FieldAllocator::operator=(const FieldAllocator& rhs)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  FieldAllocator& FieldAllocator::operator=(FieldAllocator&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    rhs.impl = nullptr;
    return *this;
  }

  //--------------------------------------------------------------------------
  FieldID FieldAllocator::allocate_field(
      size_t field_size, FieldID desired_fieldid, CustomSerdezID serdez_id,
      bool local, const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    return impl->allocate_field(
        field_size, desired_fieldid, serdez_id, local, provenance);
  }

  //--------------------------------------------------------------------------
  FieldID FieldAllocator::allocate_field(
      const Future& field_size, FieldID desired_fieldid,
      CustomSerdezID serdez_id, bool local, const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    return impl->allocate_field(
        field_size, desired_fieldid, serdez_id, local, provenance);
  }

  //--------------------------------------------------------------------------
  void FieldAllocator::free_field(
      FieldID fid, const bool unordered, const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    impl->free_field(fid, unordered, provenance);
  }

  //--------------------------------------------------------------------------
  FieldID FieldAllocator::allocate_local_field(
      size_t field_size, FieldID desired_fieldid, CustomSerdezID serdez_id,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    return impl->allocate_field(
        field_size, desired_fieldid, serdez_id, true /*local*/, provenance);
  }

  //--------------------------------------------------------------------------
  void FieldAllocator::allocate_fields(
      const std::vector<size_t>& field_sizes,
      std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
      bool local, const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    impl->allocate_fields(
        field_sizes, resulting_fields, serdez_id, local, provenance);
  }

  //--------------------------------------------------------------------------
  void FieldAllocator::allocate_fields(
      const std::vector<Future>& field_sizes,
      std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
      bool local, const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    impl->allocate_fields(
        field_sizes, resulting_fields, serdez_id, local, provenance);
  }

  //--------------------------------------------------------------------------
  void FieldAllocator::free_fields(
      const std::set<FieldID>& to_free, const bool unordered, const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    impl->free_fields(to_free, unordered, provenance);
  }

  //--------------------------------------------------------------------------
  void FieldAllocator::allocate_local_fields(
      const std::vector<size_t>& field_sizes,
      std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    Internal::AutoProvenance provenance(prov);
    impl->allocate_fields(
        field_sizes, resulting_fields, serdez_id, true /*local*/, provenance);
  }

  //--------------------------------------------------------------------------
  FieldSpace FieldAllocator::get_field_space(void) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
      return FieldSpace::NO_SPACE;
    else
      return impl->get_field_space();
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Field Allocator Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldAllocatorImpl::FieldAllocatorImpl(
        FieldSpaceNode* n, TaskContext* ctx, RtEvent ready)
      : field_space(n->handle), node(n), context(ctx), ready_event(ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(node != nullptr);
      legion_assert(context != nullptr);
      context->add_base_resource_ref(FIELD_ALLOCATOR_REF);
      node->add_base_resource_ref(FIELD_ALLOCATOR_REF);
    }

    //--------------------------------------------------------------------------
    FieldAllocatorImpl::~FieldAllocatorImpl(void)
    //--------------------------------------------------------------------------
    {
      context->destroy_field_allocator(node);
      if (context->remove_base_resource_ref(FIELD_ALLOCATOR_REF))
        delete context;
      if (node->remove_base_resource_ref(FIELD_ALLOCATOR_REF))
        delete node;
    }

    //--------------------------------------------------------------------------
    FieldID FieldAllocatorImpl::allocate_field(
        size_t field_size, FieldID desired_fieldid, CustomSerdezID serdez_id,
        bool local, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      // Need to wait for this allocator to be ready
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      return context->allocate_field(
          field_space, field_size, desired_fieldid, local, serdez_id, prov);
    }

    //--------------------------------------------------------------------------
    FieldID FieldAllocatorImpl::allocate_field(
        const Future& field_size, FieldID desired_fieldid,
        CustomSerdezID serdez_id, bool local, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      // Need to wait for this allocator to be ready
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      return context->allocate_field(
          field_space, field_size, desired_fieldid, local, serdez_id, prov);
    }

    //--------------------------------------------------------------------------
    void FieldAllocatorImpl::free_field(
        FieldID fid, const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Don't need to wait here since deletion operations catch
      // dependences on the allocator themselves
      context->free_field(this, field_space, fid, unordered, provenance);
    }

    //--------------------------------------------------------------------------
    void FieldAllocatorImpl::allocate_fields(
        const std::vector<size_t>& field_sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        bool local, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Need to wait for this allocator to be ready
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      context->allocate_fields(
          field_space, field_sizes, resulting_fields, local, serdez_id,
          provenance);
    }

    //--------------------------------------------------------------------------
    void FieldAllocatorImpl::allocate_fields(
        const std::vector<Future>& field_sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        bool local, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Need to wait for this allocator to be ready
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      context->allocate_fields(
          field_space, field_sizes, resulting_fields, local, serdez_id,
          provenance);
    }

    //--------------------------------------------------------------------------
    void FieldAllocatorImpl::free_fields(
        const std::set<FieldID>& to_free, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Don't need to wait here since deletion operations catch
      // dependences on the allocator themselves
      context->free_fields(this, field_space, to_free, unordered, provenance);
    }

  }  // namespace Internal
}  // namespace Legion
