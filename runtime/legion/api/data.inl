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

// Included from data.h - do not include this directly

// Useful for IDEs
#include "legion/api/data.h"

namespace Legion {

  //--------------------------------------------------------------------------
  inline bool IndexSpace::operator==(const IndexSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    if (did != rhs.did)
      return false;
    if (tid != rhs.tid)
      return false;
    assert(type_tag == rhs.type_tag);
    return true;
  }

  //--------------------------------------------------------------------------
  inline bool IndexSpace::operator!=(const IndexSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    if ((did == rhs.did) && (tid == rhs.tid))
      return false;
    return true;
  }

  //--------------------------------------------------------------------------
  inline bool IndexSpace::operator<(const IndexSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    if (did < rhs.did)
      return true;
    if (did > rhs.did)
      return false;
    return (tid < rhs.tid);
  }

  //--------------------------------------------------------------------------
  inline bool IndexSpace::operator>(const IndexSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    if (did > rhs.did)
      return true;
    if (did < rhs.did)
      return false;
    return (tid > rhs.tid);
  }

  //--------------------------------------------------------------------------
  inline size_t IndexSpace::hash(void) const
  //--------------------------------------------------------------------------
  {
    // uniquely identifies this index space
    return std::hash<decltype(did)>{}(did);
  }

  //--------------------------------------------------------------------------
  inline DistributedID IndexSpace::get_id(bool filter) const
  //--------------------------------------------------------------------------
  {
    if (filter)
      return LEGION_DISTRIBUTED_ID_FILTER(did);
    else
      return did;
  }

  //--------------------------------------------------------------------------
  inline int IndexSpace::get_dim(void) const
  //--------------------------------------------------------------------------
  {
    if (type_tag == 0)
      return 0;
    return Internal::NT_TemplateHelper::get_dim(type_tag);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T>::IndexSpaceT(DistributedID id, IndexTreeID tid)
    : IndexSpace(
          id, tid, Internal::NT_TemplateHelper::template encode_tag<DIM, T>())
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T>::IndexSpaceT(void)
    : IndexSpace(
          0, 0, Internal::NT_TemplateHelper::template encode_tag<DIM, T>())
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T>::IndexSpaceT(const IndexSpace& rhs) : IndexSpace(rhs)
  //--------------------------------------------------------------------------
  {
    Internal::NT_TemplateHelper::template check_type<DIM, T>(type_tag);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  inline IndexSpaceT<DIM, T>& IndexSpaceT<DIM, T>::operator=(
      const IndexSpace& rhs)
  //--------------------------------------------------------------------------
  {
    did = rhs.get_id(false);
    tid = rhs.get_tree_id();
    type_tag = rhs.get_type_tag();
    Internal::NT_TemplateHelper::template check_type<DIM, T>(type_tag);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline bool IndexPartition::operator==(const IndexPartition& rhs) const
  //--------------------------------------------------------------------------
  {
    if (did != rhs.did)
      return false;
    if (tid != rhs.tid)
      return false;
    assert(type_tag == rhs.type_tag);
    return true;
  }

  //--------------------------------------------------------------------------
  inline bool IndexPartition::operator!=(const IndexPartition& rhs) const
  //--------------------------------------------------------------------------
  {
    if ((did == rhs.did) && (tid == rhs.tid))
      return false;
    return true;
  }

  //--------------------------------------------------------------------------
  inline bool IndexPartition::operator<(const IndexPartition& rhs) const
  //--------------------------------------------------------------------------
  {
    if (did < rhs.did)
      return true;
    if (did > rhs.did)
      return false;
    return (tid < rhs.tid);
  }

  //--------------------------------------------------------------------------
  inline bool IndexPartition::operator>(const IndexPartition& rhs) const
  //--------------------------------------------------------------------------
  {
    if (did > rhs.did)
      return true;
    if (did < rhs.did)
      return false;
    return (tid > rhs.tid);
  }

  //--------------------------------------------------------------------------
  inline size_t IndexPartition::hash(void) const
  //--------------------------------------------------------------------------
  {
    // uniquely identifies this index partition
    return std::hash<decltype(did)>{}(did);
  }

  //--------------------------------------------------------------------------
  inline DistributedID IndexPartition::get_id(bool filter) const
  //--------------------------------------------------------------------------
  {
    if (filter)
      return LEGION_DISTRIBUTED_ID_FILTER(did);
    else
      return did;
  }

  //--------------------------------------------------------------------------
  inline int IndexPartition::get_dim(void) const
  //--------------------------------------------------------------------------
  {
    if (type_tag == 0)
      return 0;
    return Internal::NT_TemplateHelper::get_dim(type_tag);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T>::IndexPartitionT(DistributedID id, IndexTreeID tid)
    : IndexPartition(
          id, tid, Internal::NT_TemplateHelper::template encode_tag<DIM, T>())
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T>::IndexPartitionT(void)
    : IndexPartition(
          0, 0, Internal::NT_TemplateHelper::template encode_tag<DIM, T>())
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T>::IndexPartitionT(const IndexPartition& rhs)
    : IndexPartition(rhs)
  //--------------------------------------------------------------------------
  {
    Internal::NT_TemplateHelper::template check_type<DIM, T>(type_tag);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T>& IndexPartitionT<DIM, T>::operator=(
      const IndexPartition& rhs)
  //--------------------------------------------------------------------------
  {
    did = rhs.get_id(false);
    tid = rhs.get_tree_id();
    type_tag = rhs.get_type_tag();
    Internal::NT_TemplateHelper::template check_type<DIM, T>(type_tag);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline bool FieldSpace::operator==(const FieldSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    return (did == rhs.did);
  }

  //--------------------------------------------------------------------------
  inline bool FieldSpace::operator!=(const FieldSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    return (did != rhs.did);
  }

  //--------------------------------------------------------------------------
  inline bool FieldSpace::operator<(const FieldSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    return (did < rhs.did);
  }

  //--------------------------------------------------------------------------
  inline bool FieldSpace::operator>(const FieldSpace& rhs) const
  //--------------------------------------------------------------------------
  {
    return (did > rhs.did);
  }

  //--------------------------------------------------------------------------
  inline size_t FieldSpace::hash(void) const
  //--------------------------------------------------------------------------
  {
    // uniquely identifies this field space
    return std::hash<decltype(did)>{}(did);
  }

  //--------------------------------------------------------------------------
  inline DistributedID FieldSpace::get_id(bool filter) const
  //--------------------------------------------------------------------------
  {
    if (filter)
      return LEGION_DISTRIBUTED_ID_FILTER(did);
    else
      return did;
  }

  //--------------------------------------------------------------------------
  inline RegionTreeID LogicalRegion::get_tree_id(bool filter) const
  //--------------------------------------------------------------------------
  {
    if (filter)
      return LEGION_DISTRIBUTED_ID_FILTER(tree_did);
    else
      return tree_did;
  }

  //--------------------------------------------------------------------------
  inline bool LogicalRegion::operator==(const LogicalRegion& rhs) const
  //--------------------------------------------------------------------------
  {
    return (
        (tree_did == rhs.tree_did) && (index_space == rhs.index_space) &&
        (field_space == rhs.field_space));
  }

  //--------------------------------------------------------------------------
  inline bool LogicalRegion::operator!=(const LogicalRegion& rhs) const
  //--------------------------------------------------------------------------
  {
    return (!((*this) == rhs));
  }

  //--------------------------------------------------------------------------
  inline bool LogicalRegion::operator<(const LogicalRegion& rhs) const
  //--------------------------------------------------------------------------
  {
    if (tree_did < rhs.tree_did)
      return true;
    else if (tree_did > rhs.tree_did)
      return false;
    else
    {
      if (index_space < rhs.index_space)
        return true;
      else if (index_space != rhs.index_space)  // therefore greater than
        return false;
      else
        return field_space < rhs.field_space;
    }
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T>::LogicalRegionT(
      DistributedID tid, IndexSpace is, FieldSpace fs)
    : LogicalRegion(tid, is, fs)
  //--------------------------------------------------------------------------
  {
    Internal::NT_TemplateHelper::template check_type<DIM, T>(is.get_type_tag());
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T>::LogicalRegionT(void) : LogicalRegion()
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T>::LogicalRegionT(const LogicalRegion& rhs)
    : LogicalRegion(
          rhs.get_tree_id(false), rhs.get_index_space(), rhs.get_field_space())
  //--------------------------------------------------------------------------
  {
    Internal::NT_TemplateHelper::template check_type<DIM, T>(
        rhs.get_type_tag());
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T>& LogicalRegionT<DIM, T>::operator=(
      const LogicalRegion& rhs)
  //--------------------------------------------------------------------------
  {
    tree_did = rhs.get_tree_id(false);
    index_space = rhs.get_index_space();
    field_space = rhs.get_field_space();
    Internal::NT_TemplateHelper::template check_type<DIM, T>(
        rhs.get_type_tag());
    return *this;
  }

  //--------------------------------------------------------------------------
  inline RegionTreeID LogicalPartition::get_tree_id(bool filter) const
  //--------------------------------------------------------------------------
  {
    if (filter)
      return LEGION_DISTRIBUTED_ID_FILTER(tree_did);
    else
      return tree_did;
  }

  //--------------------------------------------------------------------------
  inline bool LogicalPartition::operator==(const LogicalPartition& rhs) const
  //--------------------------------------------------------------------------
  {
    return (
        (tree_did == rhs.tree_did) &&
        (index_partition == rhs.index_partition) &&
        (field_space == rhs.field_space));
  }

  //--------------------------------------------------------------------------
  inline bool LogicalPartition::operator!=(const LogicalPartition& rhs) const
  //--------------------------------------------------------------------------
  {
    return (!((*this) == rhs));
  }

  //--------------------------------------------------------------------------
  inline bool LogicalPartition::operator<(const LogicalPartition& rhs) const
  //--------------------------------------------------------------------------
  {
    if (tree_did < rhs.tree_did)
      return true;
    else if (tree_did > rhs.tree_did)
      return false;
    else
    {
      if (index_partition < rhs.index_partition)
        return true;
      else if (index_partition > rhs.index_partition)
        return false;
      else
        return (field_space < rhs.field_space);
    }
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T>::LogicalPartitionT(
      DistributedID tid, IndexPartition pid, FieldSpace fs)
    : LogicalPartition(tid, pid, fs)
  //--------------------------------------------------------------------------
  {
    Internal::NT_TemplateHelper::template check_type<DIM, T>(
        pid.get_type_tag());
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T>::LogicalPartitionT(void) : LogicalPartition()
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T>::LogicalPartitionT(const LogicalPartition& rhs)
    : LogicalPartition(
          rhs.get_tree_id(false), rhs.get_index_partition(),
          rhs.get_field_space())
  //--------------------------------------------------------------------------
  {
    Internal::NT_TemplateHelper::template check_type<DIM, T>(
        rhs.get_type_tag());
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T>& LogicalPartitionT<DIM, T>::operator=(
      const LogicalPartition& rhs)
  //--------------------------------------------------------------------------
  {
    tree_did = rhs.get_tree_id(false);
    index_partition = rhs.get_index_partition();
    field_space = rhs.get_field_space();
    Internal::NT_TemplateHelper::template check_type<DIM, T>(
        rhs.get_type_tag());
    return *this;
  }

  //--------------------------------------------------------------------------
  inline bool FieldAllocator::operator==(const FieldAllocator& rhs) const
  //--------------------------------------------------------------------------
  {
    return (impl == rhs.impl);
  }

  //--------------------------------------------------------------------------
  inline bool FieldAllocator::operator<(const FieldAllocator& rhs) const
  //--------------------------------------------------------------------------
  {
    return (impl < rhs.impl);
  }

}  // namespace Legion

namespace std {

#define LEGION_DEFINE_HASHABLE(__TYPE_NAME__)                     \
  template<>                                                      \
  struct hash<__TYPE_NAME__> {                                    \
    inline std::size_t operator()(const __TYPE_NAME__& obj) const \
    {                                                             \
      return obj.hash();                                          \
    }                                                             \
  };

  LEGION_DEFINE_HASHABLE(Legion::IndexSpace);
  LEGION_DEFINE_HASHABLE(Legion::IndexPartition);
  LEGION_DEFINE_HASHABLE(Legion::FieldSpace);
  LEGION_DEFINE_HASHABLE(Legion::LogicalRegion);
  LEGION_DEFINE_HASHABLE(Legion::LogicalPartition);

#undef LEGION_DEFINE_HASHABLE

}  // namespace std
