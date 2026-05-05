/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/api/exception.h"
#include "legion/api/requirements.h"
#include "legion/utilities/privileges.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Region Requirement
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(void)
    : region(LogicalRegion::NO_REGION), partition(LogicalPartition::NO_PART),
      privilege(LEGION_NO_ACCESS), prop(LEGION_EXCLUSIVE),
      parent(LogicalRegion::NO_REGION), redop(0), tag(0), flags(LEGION_NO_FLAG),
      handle_type(LEGION_SINGULAR_PROJECTION), projection(0),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, const std::set<FieldID>& priv_fields,
      const std::vector<FieldID>& inst_fields, PrivilegeMode _priv,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : region(_handle), privilege(_priv), prop(_prop), parent(_parent), redop(0),
      tag(_tag), flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_SINGULAR_PROJECTION), projection(0),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    privilege_fields = priv_fields;
    instance_fields = inst_fields;
    // For backwards compatibility with the old encoding
    if (privilege == LEGION_WRITE_PRIV)
      privilege = LEGION_WRITE_DISCARD;
    if (IS_REDUCE(*this))  // Shouldn't use this constructor for reductions
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Use a different RegionRequirement constructor for reductions";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalPartition pid, ProjectionID _proj,
      const std::set<FieldID>& priv_fields,
      const std::vector<FieldID>& inst_fields, PrivilegeMode _priv,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : partition(pid), privilege(_priv), prop(_prop), parent(_parent), redop(0),
      tag(_tag), flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_PARTITION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    privilege_fields = priv_fields;
    instance_fields = inst_fields;
    // For backwards compatibility with the old encoding
    if (privilege == LEGION_WRITE_PRIV)
      privilege = LEGION_WRITE_DISCARD;
    if (IS_REDUCE(*this))  // Shouldn't use this constructor for reductions
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Use a different RegionRequirement constructor for reductions";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, ProjectionID _proj,
      const std::set<FieldID>& priv_fields,
      const std::vector<FieldID>& inst_fields, PrivilegeMode _priv,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : region(_handle), privilege(_priv), prop(_prop), parent(_parent), redop(0),
      tag(_tag), flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_REGION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    privilege_fields = priv_fields;
    instance_fields = inst_fields;
    // For backwards compatibility with the old encoding
    if (privilege == LEGION_WRITE_PRIV)
      privilege = LEGION_WRITE_DISCARD;
    if (IS_REDUCE(*this))  // Shouldn't use this constructor for reductions
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Use a different RegionRequirement constructor for reductions";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, const std::set<FieldID>& priv_fields,
      const std::vector<FieldID>& inst_fields, ReductionOpID op,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : region(_handle), privilege(LEGION_REDUCE), prop(_prop), parent(_parent),
      redop(op), tag(_tag),
      flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_SINGULAR_PROJECTION), projection(0),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    privilege_fields = priv_fields;
    instance_fields = inst_fields;
    if (redop == 0)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Zero is a reserved ReductionOpID and cannot be used";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalPartition pid, ProjectionID _proj,
      const std::set<FieldID>& priv_fields,
      const std::vector<FieldID>& inst_fields, ReductionOpID op,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : partition(pid), privilege(LEGION_REDUCE), prop(_prop), parent(_parent),
      redop(op), tag(_tag),
      flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_PARTITION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    privilege_fields = priv_fields;
    instance_fields = inst_fields;
    if (redop == 0)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Zero is a reserved ReductionOpID and cannot be used";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, ProjectionID _proj,
      const std::set<FieldID>& priv_fields,
      const std::vector<FieldID>& inst_fields, ReductionOpID op,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : region(_handle), privilege(LEGION_REDUCE), prop(_prop), parent(_parent),
      redop(op), tag(_tag),
      flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_REGION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    privilege_fields = priv_fields;
    instance_fields = inst_fields;
    if (redop == 0)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Zero is a reserved ReductionOpID and cannot be used";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, PrivilegeMode _priv, CoherenceProperty _prop,
      LogicalRegion _parent, MappingTagID _tag, bool _verified)
    : region(_handle), privilege(_priv), prop(_prop), parent(_parent), redop(0),
      tag(_tag), flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_SINGULAR_PROJECTION), projection(),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    // For backwards compatibility with the old encoding
    if (privilege == LEGION_WRITE_PRIV)
      privilege = LEGION_WRITE_DISCARD;
    if (IS_REDUCE(*this))  // Shouldn't use this constructor for reductions
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Use a different RegionRequirement constructor for reductions";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalPartition pid, ProjectionID _proj, PrivilegeMode _priv,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : partition(pid), privilege(_priv), prop(_prop), parent(_parent), redop(0),
      tag(_tag), flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_PARTITION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    // For backwards compatibility with the old encoding
    if (privilege == LEGION_WRITE_PRIV)
      privilege = LEGION_WRITE_DISCARD;
    if (IS_REDUCE(*this))  // Shouldn't use this constructor for reductions
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Use a different RegionRequirement constructor for reductions";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, ProjectionID _proj, PrivilegeMode _priv,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : region(_handle), privilege(_priv), prop(_prop), parent(_parent), redop(0),
      tag(_tag), flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_REGION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    // For backwards compatibility with the old encoding
    if (privilege == LEGION_WRITE_PRIV)
      privilege = LEGION_WRITE_DISCARD;
    if (IS_REDUCE(*this))  // Shouldn't use this constructor for reductions
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Use a different RegionRequirement constructor for reductions";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, ReductionOpID op, CoherenceProperty _prop,
      LogicalRegion _parent, MappingTagID _tag, bool _verified)
    : region(_handle), privilege(LEGION_REDUCE), prop(_prop), parent(_parent),
      redop(op), tag(_tag),
      flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_SINGULAR_PROJECTION), projection(0),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    if (redop == 0)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Zero is a reserved ReductionOpID and cannot be used";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalPartition pid, ProjectionID _proj, ReductionOpID op,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : partition(pid), privilege(LEGION_REDUCE), prop(_prop), parent(_parent),
      redop(op), tag(_tag),
      flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_PARTITION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    if (redop == 0)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Zero is a reserved ReductionOpID and cannot be used";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(
      LogicalRegion _handle, ProjectionID _proj, ReductionOpID op,
      CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag,
      bool _verified)
    : region(_handle), privilege(LEGION_REDUCE), prop(_prop), parent(_parent),
      redop(op), tag(_tag),
      flags(_verified ? LEGION_VERIFIED_FLAG : LEGION_NO_FLAG),
      handle_type(LEGION_REGION_PROJECTION), projection(_proj),
      projection_args(nullptr), projection_args_size(0)
  //--------------------------------------------------------------------------
  {
    if (redop == 0)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Zero is a reserved ReductionOpID and cannot be used";
      error.raise();
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(const RegionRequirement& rhs)
    : region(rhs.region), partition(rhs.partition),
      privilege_fields(rhs.privilege_fields),
      instance_fields(rhs.instance_fields), privilege(rhs.privilege),
      prop(rhs.prop), parent(rhs.parent), redop(rhs.redop), tag(rhs.tag),
      flags(rhs.flags), handle_type(rhs.handle_type),
      projection(rhs.projection), projection_args(nullptr),
      projection_args_size(rhs.projection_args_size)
  //--------------------------------------------------------------------------
  {
    if (projection_args_size > 0)
    {
      projection_args = malloc(projection_args_size);
      memcpy(projection_args, rhs.projection_args, projection_args_size);
    }
  }

  //--------------------------------------------------------------------------
  RegionRequirement::RegionRequirement(RegionRequirement&& rhs) noexcept
    : region(rhs.region), partition(rhs.partition), privilege(rhs.privilege),
      prop(rhs.prop), parent(rhs.parent), redop(rhs.redop), tag(rhs.tag),
      flags(rhs.flags), handle_type(rhs.handle_type),
      projection(rhs.projection), projection_args(rhs.projection_args),
      projection_args_size(rhs.projection_args_size)
  //--------------------------------------------------------------------------
  {
    static_assert(std::is_trivially_copyable<decltype(region)>::value);
    static_assert(std::is_trivially_copyable<decltype(partition)>::value);
    static_assert(std::is_trivially_copyable<decltype(privilege)>::value);
    static_assert(std::is_trivially_copyable<decltype(prop)>::value);
    static_assert(std::is_trivially_copyable<decltype(parent)>::value);
    static_assert(std::is_trivially_copyable<decltype(redop)>::value);
    static_assert(std::is_trivially_copyable<decltype(tag)>::value);
    static_assert(std::is_trivially_copyable<decltype(flags)>::value);
    static_assert(std::is_trivially_copyable<decltype(handle_type)>::value);
    static_assert(std::is_trivially_copyable<decltype(projection)>::value);
    static_assert(std::is_trivially_copyable<decltype(projection_args)>::value);
    static_assert(
        std::is_trivially_copyable<decltype(projection_args_size)>::value);
    privilege_fields.swap(rhs.privilege_fields);
    instance_fields.swap(rhs.instance_fields);
    rhs.projection_args = nullptr;
    rhs.projection_args_size = 0;
  }

  //--------------------------------------------------------------------------
  RegionRequirement::~RegionRequirement(void)
  //--------------------------------------------------------------------------
  {
    if (projection_args_size > 0)
      free(projection_args);
  }

  //--------------------------------------------------------------------------
  RegionRequirement& RegionRequirement::operator=(const RegionRequirement& rhs)
  //--------------------------------------------------------------------------
  {
    region = rhs.region;
    partition = rhs.partition;
    privilege_fields = rhs.privilege_fields;
    instance_fields = rhs.instance_fields;
    privilege = rhs.privilege;
    prop = rhs.prop;
    parent = rhs.parent;
    redop = rhs.redop;
    tag = rhs.tag;
    flags = rhs.flags;
    handle_type = rhs.handle_type;
    projection = rhs.projection;
    projection_args_size = rhs.projection_args_size;
    if (projection_args != nullptr)
    {
      free(projection_args);
      projection_args = nullptr;
    }
    if (projection_args_size > 0)
    {
      projection_args = malloc(projection_args_size);
      memcpy(projection_args, rhs.projection_args, projection_args_size);
    }
    return *this;
  }

  //--------------------------------------------------------------------------
  RegionRequirement& RegionRequirement::operator=(
      RegionRequirement&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    region = rhs.region;
    partition = rhs.partition;
    privilege_fields.swap(rhs.privilege_fields);
    instance_fields.swap(rhs.instance_fields);
    privilege = rhs.privilege;
    prop = rhs.prop;
    parent = rhs.parent;
    redop = rhs.redop;
    tag = rhs.tag;
    flags = rhs.flags;
    handle_type = rhs.handle_type;
    projection = rhs.projection;
    if (projection_args_size > 0)
      free(projection_args);
    projection_args_size = rhs.projection_args_size;
    projection_args = rhs.projection_args;
    rhs.projection_args = nullptr;
    rhs.projection_args_size = 0;
    return *this;
  }

  //--------------------------------------------------------------------------
  bool RegionRequirement::operator==(const RegionRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    if ((handle_type == rhs.handle_type) && (privilege == rhs.privilege) &&
        (prop == rhs.prop) && (parent == rhs.parent) && (redop == rhs.redop) &&
        (tag == rhs.tag) && (flags == rhs.flags))
    {
      if (((handle_type == LEGION_SINGULAR_PROJECTION) &&
           (region == rhs.region)) ||
          ((handle_type == LEGION_PARTITION_PROJECTION) &&
           (partition == rhs.partition) && (projection == rhs.projection)) ||
          ((handle_type == LEGION_REGION_PROJECTION) && (region == rhs.region)))
      {
        if ((privilege_fields.size() == rhs.privilege_fields.size()) &&
            (instance_fields.size() == rhs.instance_fields.size()))
        {
          if (projection_args_size == rhs.projection_args_size)
          {
            if ((projection_args_size == 0) ||
                (memcmp(
                     projection_args, rhs.projection_args,
                     projection_args_size) == 0))
            {
              return (
                  (privilege_fields == rhs.privilege_fields) &&
                  (instance_fields == rhs.instance_fields));
            }
          }
        }
      }
    }
    return false;
  }

  //--------------------------------------------------------------------------
  bool RegionRequirement::operator<(const RegionRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    if (handle_type < rhs.handle_type)
      return true;
    else if (handle_type > rhs.handle_type)
      return false;
    else
    {
      if (privilege < rhs.privilege)
        return true;
      else if (privilege > rhs.privilege)
        return false;
      else
      {
        if (prop < rhs.prop)
          return true;
        else if (prop > rhs.prop)
          return false;
        else
        {
          if (parent < rhs.parent)
            return true;
          else if (!(parent == rhs.parent))  // therefore greater than
            return false;
          else
          {
            if (redop < rhs.redop)
              return true;
            else if (redop > rhs.redop)
              return false;
            else
            {
              if (tag < rhs.tag)
                return true;
              else if (tag > rhs.tag)
                return false;
              else
              {
                if (flags < rhs.flags)
                  return true;
                else if (flags > rhs.flags)
                  return false;
                else
                {
                  if (privilege_fields < rhs.privilege_fields)
                    return true;
                  else if (privilege_fields > rhs.privilege_fields)
                    return false;
                  else
                  {
                    if (instance_fields < rhs.instance_fields)
                      return true;
                    else if (instance_fields > rhs.instance_fields)
                      return false;
                    else
                    {
                      if (handle_type == LEGION_SINGULAR_PROJECTION)
                        return (region < rhs.region);
                      else if (projection_args_size < rhs.projection_args_size)
                        return true;
                      else if (projection_args_size > rhs.projection_args_size)
                        return false;
                      else if (
                          (projection_args_size > 0) &&
                          (memcmp(
                               projection_args, rhs.projection_args,
                               projection_args_size) < 0))
                        return true;
                      else if (
                          (projection_args_size > 0) &&
                          (memcmp(
                               projection_args, rhs.projection_args,
                               projection_args_size) > 0))
                        return false;
                      else if (handle_type == LEGION_PARTITION_PROJECTION)
                      {
                        if (partition < rhs.partition)
                          return true;
                        // therefore greater than
                        else if (partition != rhs.partition)
                          return false;
                        else
                          return (projection < rhs.projection);
                      }
                      else
                      {
                        if (region < rhs.region)
                          return true;
                        else if (region != rhs.region)
                          return false;
                        else
                          return (projection < rhs.projection);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  //--------------------------------------------------------------------------
  bool RegionRequirement::has_field_privilege(FieldID fid) const
  //--------------------------------------------------------------------------
  {
    return (privilege_fields.find(fid) != privilege_fields.end());
  }

  //--------------------------------------------------------------------------
  const void* RegionRequirement::get_projection_args(size_t* size) const
  //--------------------------------------------------------------------------
  {
    if (size != nullptr)
      *size = projection_args_size;
    return projection_args;
  }

  //--------------------------------------------------------------------------
  void RegionRequirement::set_projection_args(
      const void* args, size_t size, bool own)
  //--------------------------------------------------------------------------
  {
    if (projection_args_size > 0)
    {
      free(projection_args);
      projection_args = nullptr;
    }
    projection_args_size = size;
    if (projection_args_size > 0)
    {
      if (!own)
      {
        projection_args = malloc(projection_args_size);
        memcpy(projection_args, args, projection_args_size);
      }
      else
        projection_args = const_cast<void*>(args);
    }
  }

  /////////////////////////////////////////////////////////////
  // Output Requirement
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  OutputRequirement::OutputRequirement(bool bounded)
    : RegionRequirement(),
      type_tag(Internal::NT_TemplateHelper::encode_tag<1, coord_t>()),
      field_space(FieldSpace::NO_SPACE), global_indexing(false),
      bounded_requirement(bounded), color_space(IndexSpace::NO_SPACE)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  OutputRequirement::OutputRequirement(const RegionRequirement& req)
    : RegionRequirement(req), type_tag(req.parent.get_type_tag()),
      global_indexing(false), bounded_requirement(true),
      color_space(IndexSpace::NO_SPACE)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  OutputRequirement::OutputRequirement(
      FieldSpace _field_space, const std::set<FieldID>& fields, int dim /*=1*/,
      bool _global_indexing /*=false*/)
    : RegionRequirement(), field_space(_field_space),
      global_indexing(_global_indexing), bounded_requirement(false),
      color_space(IndexSpace::NO_SPACE)
  //--------------------------------------------------------------------------
  {
    switch (dim)
    {
#define DIMFUNC(DIM)                                                      \
  case DIM:                                                               \
    {                                                                     \
      type_tag = Internal::NT_TemplateHelper::encode_tag<DIM, coord_t>(); \
      break;                                                              \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "Unsupported dimension " << dim
              << " for output region requirement. This probably means you need "
              << "to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    for (const FieldID& field : fields) RegionRequirement::add_field(field);
  }

  //--------------------------------------------------------------------------
  OutputRequirement::OutputRequirement(const OutputRequirement& other)
    : RegionRequirement(static_cast<const RegionRequirement&>(other)),
      type_tag(other.type_tag), field_space(other.field_space),
      global_indexing(other.global_indexing),
      bounded_requirement(other.bounded_requirement),
      color_space(other.color_space)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  OutputRequirement::~OutputRequirement(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  OutputRequirement& OutputRequirement::operator=(const OutputRequirement& rhs)
  //--------------------------------------------------------------------------
  {
    static_cast<RegionRequirement&>(*this) =
        static_cast<const RegionRequirement&>(rhs);
    field_space = rhs.field_space;
    global_indexing = rhs.global_indexing;
    bounded_requirement = rhs.bounded_requirement;
    type_tag = rhs.type_tag;
    color_space = rhs.color_space;
    return *this;
  }

  //--------------------------------------------------------------------------
  OutputRequirement& OutputRequirement::operator=(const RegionRequirement& rhs)
  //--------------------------------------------------------------------------
  {
    static_cast<RegionRequirement&>(*this) =
        static_cast<const RegionRequirement&>(rhs);
    field_space = FieldSpace::NO_SPACE;
    global_indexing = false;
    bounded_requirement = true;
    type_tag = rhs.region.get_type_tag();
    color_space = IndexSpace::NO_SPACE;
    return *this;
  }

  //--------------------------------------------------------------------------
  bool OutputRequirement::operator==(const OutputRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    if ((field_space != rhs.field_space) ||
        (global_indexing != rhs.global_indexing) ||
        (bounded_requirement != rhs.bounded_requirement) ||
        (color_space != rhs.color_space))
      return false;
    return static_cast<const RegionRequirement&>(*this) ==
           static_cast<const RegionRequirement&>(rhs);
  }

  //--------------------------------------------------------------------------
  bool OutputRequirement::operator<(const OutputRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    if (field_space < rhs.field_space)
      return true;
    if (field_space > rhs.field_space)
      return false;
    if (global_indexing < rhs.global_indexing)
      return true;
    if (global_indexing > rhs.global_indexing)
      return false;
    if (bounded_requirement < rhs.bounded_requirement)
      return true;
    if (bounded_requirement > rhs.bounded_requirement)
      return false;
    if (color_space < rhs.color_space)
      return true;
    if (color_space > rhs.color_space)
      return false;
    return static_cast<const RegionRequirement&>(*this) <
           static_cast<const RegionRequirement&>(rhs);
  }

  //--------------------------------------------------------------------------
  void OutputRequirement::set_projection(ProjectionID proj, IndexSpace cspace)
  //--------------------------------------------------------------------------
  {
    projection = proj;
    color_space = cspace;
  }

  /////////////////////////////////////////////////////////////
  // Index Space Requirement
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  IndexSpaceRequirement::IndexSpaceRequirement(void)
    : handle(IndexSpace::NO_SPACE), privilege(LEGION_NO_MEMORY),
      parent(IndexSpace::NO_SPACE), verified(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexSpaceRequirement::IndexSpaceRequirement(
      IndexSpace _handle, AllocateMode _priv, IndexSpace _parent,
      bool _verified /*=false*/)
    : handle(_handle), privilege(_priv), parent(_parent), verified(_verified)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  bool IndexSpaceRequirement::operator<(const IndexSpaceRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    if (handle < rhs.handle)
      return true;
    else if (handle != rhs.handle)  // therefore greater than
      return false;
    else
    {
      if (privilege < rhs.privilege)
        return true;
      else if (privilege > rhs.privilege)
        return false;
      else
      {
        if (parent < rhs.parent)
          return true;
        else if (parent != rhs.parent)  // therefore greater than
          return false;
        else
          return verified < rhs.verified;
      }
    }
  }

  //--------------------------------------------------------------------------
  bool IndexSpaceRequirement::operator==(const IndexSpaceRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    return (handle == rhs.handle) && (privilege == rhs.privilege) &&
           (parent == rhs.parent) && (verified == rhs.verified);
  }

  /////////////////////////////////////////////////////////////
  // Field Space Requirement
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  FieldSpaceRequirement::FieldSpaceRequirement(void)
    : handle(FieldSpace::NO_SPACE), privilege(LEGION_NO_MEMORY), verified(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  FieldSpaceRequirement::FieldSpaceRequirement(
      FieldSpace _handle, AllocateMode _priv, bool _verified /*=false*/)
    : handle(_handle), privilege(_priv), verified(_verified)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  bool FieldSpaceRequirement::operator<(const FieldSpaceRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    if (handle < rhs.handle)
      return true;
    else if (!(handle == rhs.handle))  // therefore greater than
      return false;
    else
    {
      if (privilege < rhs.privilege)
        return true;
      else if (privilege > rhs.privilege)
        return false;
      else
        return verified < rhs.verified;
    }
  }

  //--------------------------------------------------------------------------
  bool FieldSpaceRequirement::operator==(const FieldSpaceRequirement& rhs) const
  //--------------------------------------------------------------------------
  {
    return (handle == rhs.handle) && (privilege == rhs.privilege) &&
           (verified == rhs.verified);
  }

}  // namespace Legion
