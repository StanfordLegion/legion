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

#include "legion/operations/copy.h"
#include "legion/analysis/across.h"
#include "legion/contexts/replicate.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/region.h"
#include "legion/tracing/recognizer.h"
#include "legion/tracing/shard.h"
#include "legion/utilities/privileges.h"
#include "legion/utilities/provenance.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Copy
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalCopy::ExternalCopy(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalCopy::pack_external_copy(
        Serializer& rez, AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize<size_t>(src_requirements.size());
      for (const RegionRequirement& requirement : src_requirements)
        pack_region_requirement(requirement, rez);
      rez.serialize<size_t>(dst_requirements.size());
      for (const RegionRequirement& requirement : dst_requirements)
        pack_region_requirement(requirement, rez);
      rez.serialize<size_t>(src_indirect_requirements.size());
      for (const RegionRequirement& requirement : src_indirect_requirements)
        pack_region_requirement(requirement, rez);
      rez.serialize<size_t>(dst_indirect_requirements.size());
      for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        pack_region_requirement(dst_indirect_requirements[idx], rez);
      rez.serialize(grants.size());
      for (const Grant& grant : grants) pack_grant(grant, rez);
      rez.serialize(wait_barriers.size());
      for (const PhaseBarrier& barrier : wait_barriers)
        pack_phase_barrier(barrier, rez);
      rez.serialize(arrive_barriers.size());
      for (const PhaseBarrier& barrier : arrive_barriers)
        pack_phase_barrier(barrier, rez);
      rez.serialize<bool>(is_index_space);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      pack_mappable(*this, rez);
      rez.serialize(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalCopy::unpack_external_copy(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_srcs;
      derez.deserialize(num_srcs);
      src_requirements.resize(num_srcs);
      for (unsigned idx = 0; idx < num_srcs; idx++)
        unpack_region_requirement(src_requirements[idx], derez);
      size_t num_dsts;
      derez.deserialize(num_dsts);
      dst_requirements.resize(num_dsts);
      for (unsigned idx = 0; idx < num_dsts; idx++)
        unpack_region_requirement(dst_requirements[idx], derez);
      size_t num_indirect_srcs;
      derez.deserialize(num_indirect_srcs);
      src_indirect_requirements.resize(num_indirect_srcs);
      for (unsigned idx = 0; idx < num_indirect_srcs; idx++)
        unpack_region_requirement(src_indirect_requirements[idx], derez);
      size_t num_indirect_dsts;
      derez.deserialize(num_indirect_dsts);
      dst_indirect_requirements.resize(num_indirect_dsts);
      for (unsigned idx = 0; idx < num_indirect_dsts; idx++)
        unpack_region_requirement(dst_indirect_requirements[idx], derez);
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (Grant& grant : grants) unpack_grant(grant, derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (PhaseBarrier& barrier : wait_barriers)
        unpack_phase_barrier(barrier, derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (PhaseBarrier& barrier : arrive_barriers)
        unpack_phase_barrier(barrier, derez);
      derez.deserialize<bool>(is_index_space);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      unpack_mappable(*this, derez);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Copy Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyOp::CopyOp(void) : ExternalCopy(), PredicatedOp()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = false;
    }

    //--------------------------------------------------------------------------
    CopyOp::~CopyOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename T>
    void CopyOp::initialize_copy_from_launcher(const T& launcher)
    //--------------------------------------------------------------------------
    {
      src_requirements.resize(launcher.src_requirements.size());
      src_parent_indexes.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        src_requirements[idx] = launcher.src_requirements[idx];
        src_requirements[idx].flags |= LEGION_NO_ACCESS_FLAG;
        if (src_requirements[idx].privilege_fields.size() !=
            src_requirements[idx].instance_fields.size())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Missing instance fields for source region requirement "
                << idx << "of " << *this
                << ". The 'instance_fields' member of source "
                << "region requirements must contain exactly the same set of "
                << "fields as the 'privilege_fields' for copy operations.";
          error.raise();
        }
        if (!IS_READ_ONLY(src_requirements[idx]))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Source region requirement " << idx << " of " << *this
                << "does not have read-only privileges. All "
                << "source region requirements for copy operations must be "
                << "read-only.";
          error.raise();
        }
        if (runtime->safe_model)
          verify_requirement(src_requirements[idx], idx, this->is_index_space);
        src_parent_indexes[idx] =
            parent_ctx->find_parent_region_index(this, src_requirements[idx]);
      }
      dst_requirements.resize(launcher.dst_requirements.size());
      dst_parent_indexes.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        dst_requirements[idx] = launcher.dst_requirements[idx];
        dst_requirements[idx].flags |= LEGION_NO_ACCESS_FLAG;
        if ((dst_requirements[idx].privilege == LEGION_READ_WRITE) &&
            (launcher.src_indirect_requirements.size() <= idx) &&
            (launcher.dst_indirect_requirements.size() <= idx) &&
            (src_requirements[idx].region == dst_requirements[idx].region))
          dst_requirements[idx].privilege |= LEGION_DISCARD_MASK;
        if (dst_requirements[idx].privilege_fields.size() !=
            dst_requirements[idx].instance_fields.size())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Missing instance fields for destination region requirement "
                << idx << "of " << *this
                << ". The 'instance_fields' member of destination region "
                << "requirements must contain exactly the same set of "
                << "fields as the 'privilege_fields' for copy operations.";
          error.raise();
        }
        if (src_requirements[idx].instance_fields.size() !=
            dst_requirements[idx].instance_fields.size())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "The 'instance_fields' member of the source and destination "
                << "region requirements at index" << idx
                << " do no have the same size ("
                << src_requirements[idx].instance_fields.size() << " and "
                << dst_requirements[idx].instance_fields.size()
                << " respectively) for " << *this
                << ". The 'instance_fields' data structure must have the same "
                << "number of fields for the copy operation to know how to zip "
                   "the fields "
                << "together.";
          error.raise();
        }
        if (!HAS_WRITE(dst_requirements[idx]))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Destination region requirement " << idx << " of " << *this
                << "is not writing or reducing. All destination "
                << "region requirements for copy operations must either be "
                << "writing or reducing.";
          error.raise();
        }
        if ((dst_requirements[idx].privilege == LEGION_READ_WRITE) &&
            (launcher.src_indirect_requirements.size() <= idx) &&
            (launcher.dst_indirect_requirements.size() <= idx) &&
            (launcher.src_requirements[idx].handle_type ==
             launcher.dst_requirements[idx].handle_type))
        {
          switch (launcher.src_requirements[idx].handle_type)
          {
            case LEGION_SINGULAR_PROJECTION:
              {
                if (src_requirements[idx].region ==
                    dst_requirements[idx].region)
                  dst_requirements[idx].privilege |= LEGION_DISCARD_MASK;
                break;
              }
            case LEGION_PARTITION_PROJECTION:
              {
                if ((src_requirements[idx].partition ==
                     dst_requirements[idx].partition) &&
                    (src_requirements[idx].projection ==
                     dst_requirements[idx].projection))
                  dst_requirements[idx].privilege |= LEGION_DISCARD_MASK;
                break;
              }
            case LEGION_REGION_PROJECTION:
              {
                if ((src_requirements[idx].region ==
                     dst_requirements[idx].region) &&
                    (src_requirements[idx].projection ==
                     dst_requirements[idx].projection))
                  dst_requirements[idx].privilege |= LEGION_DISCARD_MASK;
                break;
              }
            default:
              std::abort();
          }
        }
        if (runtime->safe_model)
          verify_requirement(
              dst_requirements[idx], src_requirements.size() + idx,
              this->is_index_space);
        dst_parent_indexes[idx] =
            parent_ctx->find_parent_region_index(this, dst_requirements[idx]);
      }
      if (src_requirements.size() != dst_requirements.size())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Number of our source requirements ("
              << src_requirements.size()
              << ") does not match the number of destination requirements ("
              << dst_requirements.size() << ") for " << *this << ".";
        error.raise();
      }
      if (!launcher.src_indirect_requirements.empty())
      {
        const size_t gather_size = launcher.src_indirect_requirements.size();
        if (gather_size != src_requirements.size())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Number of source indirect requirements (" << gather_size
                << ") does not match the number of source requirements ("
                << src_requirements.size() << ") for " << *this << ".";
          error.raise();
        }
        src_indirect_requirements.resize(gather_size);
        gather_parent_indexes.resize(gather_size);
        src_indirect_records.resize(gather_size);
        for (unsigned idx = 0; idx < gather_size; idx++)
        {
          RegionRequirement& req = src_indirect_requirements[idx];
          req = launcher.src_indirect_requirements[idx];
          req.flags |= LEGION_NO_ACCESS_FLAG;
          if (req.privilege_fields.size() != 1)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Source indirect region requirement " << idx << " for "
                  << *this << " has " << req.privilege_fields.size()
                  << " fields, but indirection region requirements must always "
                  << "have exactly one field.";
            error.raise();
          }
          if (!IS_READ_ONLY(req))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error
                << "Source indirect region requirement " << idx << " for "
                << *this << " does not have read-only privileges. All source "
                << "indirect region requirements for copy operations must have "
                << "read-only privileges.";
            error.raise();
          }
          if (runtime->safe_model)
            verify_requirement(
                src_indirect_requirements[idx],
                src_requirements.size() + dst_requirements.size() + idx,
                this->is_index_space);
          gather_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, src_indirect_requirements[idx]);
        }
        if (launcher.src_indirect_is_range.size() != gather_size)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Invalid 'src_indirect_is_range' "
                << "size in launcher. The number of entries "
                << launcher.src_indirect_is_range.size()
                << " does not match the number of 'src_indirect_requirments' "
                << gather_size << " for " << *this << ".";
          error.raise();
        }
        for (unsigned idx = 0; idx < gather_size; idx++)
        {
          if (!launcher.src_indirect_is_range[idx])
            continue;
          // For anything that is a gather by range we either need
          // it also to be a scatter by range or we need a reduction
          // on the destination region requirement so we know how
          // to handle reducing down all the values
          if ((idx < launcher.dst_indirect_is_range.size()) &&
              launcher.dst_indirect_is_range[idx])
            continue;
          if (!IS_REDUCE(dst_requirements[idx]))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error
                << "Invalid privileges for destination region requirement "
                << idx << " for " << *this << ". Destination "
                << "region requirements must use reduction privileges when "
                << "there is a range-based source indirection field and there "
                << "is no corresponding range indirection on the destination.";
            error.raise();
          }
        }
        possible_src_indirect_out_of_range =
            launcher.possible_src_indirect_out_of_range;
      }
      if (!launcher.dst_indirect_requirements.empty())
      {
        const size_t scatter_size = launcher.dst_indirect_requirements.size();
        if (scatter_size != dst_requirements.size())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Number of destination indirect requirements ("
                << scatter_size
                << ") does not match the number of destination requirements ("
                << src_requirements.size() << ") for " << *this << ".";
          error.raise();
        }
        dst_indirect_requirements.resize(scatter_size);
        scatter_parent_indexes.resize(scatter_size);
        dst_indirect_records.resize(scatter_size);
        for (unsigned idx = 0; idx < scatter_size; idx++)
        {
          RegionRequirement& req = dst_indirect_requirements[idx];
          req = launcher.dst_indirect_requirements[idx];
          req.flags |= LEGION_NO_ACCESS_FLAG;
          if (req.privilege_fields.size() != 1)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Destination indirect region requirement " << idx
                  << " for " << *this << " has " << req.privilege_fields.size()
                  << " fields, but indirection region "
                  << "requirements must always have exactly one field.";
            error.raise();
          }
          if (!IS_READ_ONLY(req))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Destination indirect region requirement " << idx
                  << " for " << *this << " does not have "
                  << "read-only privileges. All destination indirect region "
                  << "requirements for copy operations must have read-only "
                     "privileges.";
            error.raise();
          }
          if (runtime->safe_model)
            verify_requirement(
                dst_indirect_requirements[idx],
                src_requirements.size() + dst_requirements.size() +
                    src_indirect_requirements.size() + idx,
                this->is_index_space);
          scatter_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, dst_indirect_requirements[idx]);
        }
        if (launcher.dst_indirect_is_range.size() != scatter_size)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Invalid 'dst_indirect_is_range' "
                << "size in launcher. The number of entries "
                << launcher.src_indirect_is_range.size()
                << " does not match the number of 'dst_indirect_requirments' "
                << scatter_size << " for " << *this << ".";
          error.raise();
        }
        if (!src_indirect_requirements.empty())
        {
          // Full indirections need to have the same index space
          for (unsigned idx = 0; (idx < src_indirect_requirements.size()) &&
                                 (idx < dst_indirect_requirements.size());
               idx++)
          {
            const IndexSpace src_space =
                src_indirect_requirements[idx].region.get_index_space();
            const IndexSpace dst_space =
                dst_indirect_requirements[idx].region.get_index_space();
            if (src_space != dst_space)
            {
              Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              error << "Mismatch between source indirect and destination "
                       "indirect "
                    << "index spaces for requirement " << idx << " for "
                    << *this << ". Currently full-indirection copies must "
                    << "specify the index space for both indirection "
                       "requirements.";
              error.raise();
            }
          }
        }
        possible_dst_indirect_out_of_range =
            launcher.possible_dst_indirect_out_of_range;
        possible_dst_indirect_aliasing =
            launcher.possible_dst_indirect_aliasing;
      }

      grants = launcher.grants;
      // Register ourselves with all the grants
      for (const Grant& grant : grants)
        grant.impl->register_operation(get_completion_event());
      wait_barriers = launcher.wait_barriers;
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        for (const PhaseBarrier& bar : launcher.arrive_barriers)
        {
          arrive_barriers.emplace_back(bar);
          LegionSpy::log_event_dependence(
              bar.phase_barrier, arrive_barriers.back().phase_barrier);
        }
      }
      else
        arrive_barriers = launcher.arrive_barriers;
      wait_barriers = launcher.wait_barriers;
      gather_is_range = launcher.src_indirect_is_range;
      scatter_is_range = launcher.dst_indirect_is_range;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::initialize(
        InnerContext* ctx, const CopyLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_predication(ctx, launcher.predicate, provenance);
      initialize_copy_from_launcher(launcher);
      atomic_locks.resize(
          src_requirements.size() + dst_requirements.size() +
          src_indirect_requirements.size() + dst_indirect_requirements.size());
      index_point = launcher.point;
      index_domain = Domain(index_point, index_point);
      sharding_space = launcher.sharding_space;
      if (runtime->safe_model)
        perform_type_checking();
      const unsigned copy_kind = (src_indirect_requirements.empty() ? 0 : 1) +
                                 (dst_indirect_requirements.empty() ? 0 : 2);
      LegionSpy::log_copy_operation(
          parent_ctx->get_unique_id(), unique_op_id, copy_kind, false, false);
    }

    //--------------------------------------------------------------------------
    void CopyOp::perform_type_checking(void) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        // Check that the source and destination field sizes are the same
        const std::vector<FieldID>& src_fields =
            src_requirements[idx].instance_fields;
        const std::vector<FieldID>& dst_fields =
            dst_requirements[idx].instance_fields;
        const FieldSpace src_space =
            src_requirements[idx].parent.get_field_space();
        const FieldSpace dst_space =
            dst_requirements[idx].parent.get_field_space();
        for (unsigned fidx = 0; fidx < src_fields.size(); fidx++)
        {
          const size_t src_size =
              runtime->get_field_size(src_space, src_fields[fidx]);
          const size_t dst_size =
              runtime->get_field_size(dst_space, dst_fields[fidx]);
          if (src_size != dst_size)
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "Different field sizes are not permitted for "
                     "region-to-region "
                  << "copy operations. Fields " << src_fields[fidx] << " and "
                  << dst_fields[fidx] << " of region requirement " << idx
                  << " have different sizes (" << src_size << " bytes and "
                  << dst_size << " bytes respectively) in " << *this << ".";
            error.raise();
          }
          const CustomSerdezID src_serdez =
              runtime->get_field_serdez(src_space, src_fields[fidx]);
          const CustomSerdezID dst_serdez =
              runtime->get_field_serdez(dst_space, dst_fields[fidx]);
          if (src_serdez != dst_serdez)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Fields with different serdez modes are not permitted for "
                  << "region-to-region copy operations. Fields "
                  << src_fields[fidx] << " and " << dst_fields[fidx]
                  << " of region requirement " << idx
                  << " have different serdez modes (" << src_serdez << " and "
                  << dst_serdez << " respectively) in " << *this << ".";
            error.raise();
          }
        }
        if (idx < src_indirect_requirements.size())
        {
          // Check that the size of the source indirect field is same
          // as the size of the source coordinate type
          const RegionRequirement& src_idx_req = src_indirect_requirements[idx];
          const FieldID fid = *src_idx_req.privilege_fields.begin();
          const size_t idx_size = runtime->get_field_size(
              src_idx_req.parent.get_field_space(), fid);
          const IndexSpace src_space =
              src_requirements[idx].parent.get_index_space();
          const size_t coord_size =
              runtime->get_coordinate_size(src_space, false /*range*/);
          if (idx_size != coord_size)
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "The source indirect field for a copy operation has the "
                  << "incorrect size for the source region coordinate space. "
                  << "Field " << fid
                  << " of source indirect region requirement " << idx << " is "
                  << idx_size << " bytes but the coordinate types of the "
                  << "source space is " << coord_size << " bytes for " << *this
                  << ".";
            error.raise();
          }
          const CustomSerdezID idx_serdez = runtime->get_field_serdez(
              src_idx_req.parent.get_field_space(), fid);
          if (idx_serdez != 0)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Serdez fields are not permitted to be used as "
                  << "indirection fields for copy operations. Field " << fid
                  << "of source indirect region requirement " << idx << " in "
                  << *this << "has serdez function " << idx_serdez << ".";
            error.raise();
          }
        }
        if (idx >= dst_indirect_requirements.size())
        {
          if (idx >= src_indirect_requirements.size())
          {
            // Normal copy
            IndexSpace src_space =
                src_requirements[idx].parent.get_index_space();
            IndexSpace dst_space =
                dst_requirements[idx].parent.get_index_space();
            bool diff_dims = false;
            if (!runtime->check_types(
                    src_space.get_type_tag(), dst_space.get_type_tag(),
                    diff_dims))
            {
              Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
              error << "Copy index space mismatch at index " << idx
                    << " of cross-region " << *this
                    << ". The index spaces of the source and destination "
                       "requirements "
                    << "have incompatible types because they have different "
                    << (diff_dims ? "numbers of dimensions." :
                                    "coordinate_types.");
              error.raise();
            }
          }
          else
          {
            // Gather copy
            IndexSpace src_indirect_space =
                src_indirect_requirements[idx].parent.get_index_space();
            IndexSpace dst_space =
                dst_requirements[idx].parent.get_index_space();
            bool diff_dims = false;
            if (!runtime->check_types(
                    src_indirect_space.get_type_tag(), dst_space.get_type_tag(),
                    diff_dims))
            {
              Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
              error << "Copy index space mismatch at index " << idx
                    << " of cross-region " << *this
                    << ".  The index spaces of the source indirect requirement "
                       "and "
                    << "the destination requirement have incompatible types "
                       "because "
                    << "they have different "
                    << (diff_dims ? "numbers of dimensions." :
                                    "coordinate types.");
              error.raise();
            }
          }
        }
        else
        {
          // Check that the size of the source indirect field is same
          // as the size of the source coordinate type
          const RegionRequirement& dst_idx_req = dst_indirect_requirements[idx];
          const FieldID fid = *dst_idx_req.privilege_fields.begin();
          const size_t idx_size = runtime->get_field_size(
              dst_idx_req.parent.get_field_space(), fid);
          const IndexSpace dst_space =
              dst_requirements[idx].parent.get_index_space();
          const size_t coord_size =
              runtime->get_coordinate_size(dst_space, false /*range*/);
          if (idx_size != coord_size)
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "The destination indirect field for a copy operation has "
                     "the "
                  << "incorrect size for the destination region coordinate "
                     "space. "
                  << "Field " << fid
                  << " of destination indirect region requirement " << idx
                  << " is " << idx_size << " bytes but the coordinate types of "
                  << "the destination space is " << coord_size << " bytes for "
                  << *this << ".";
            error.raise();
          }
          const CustomSerdezID idx_serdez = runtime->get_field_serdez(
              dst_idx_req.parent.get_field_space(), fid);
          if (idx_serdez != 0)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error
                << "Serdez fields are not permitted to be used as indirection "
                << "fields for copy operations. Field " << fid
                << " of destination indirect region requirement " << idx
                << " in " << *this << " has serdez function " << idx_serdez
                << ".";
            error.raise();
          }
          if (idx >= src_indirect_requirements.size())
          {
            // Scatter copy
            IndexSpace src_space =
                src_requirements[idx].parent.get_index_space();
            IndexSpace dst_indirect_space =
                dst_indirect_requirements[idx].parent.get_index_space();
            // Just check compatibility here since it's really hard to
            // prove that we're actually going to write everything
            bool diff_dims = false;
            if (!runtime->check_types(
                    src_space.get_type_tag(), dst_indirect_space.get_type_tag(),
                    diff_dims))
            {
              Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
              error
                  << "Copy index space mismatch at index " << idx
                  << " of cross-region " << *this
                  << ". The index spaces of the source requirement and the "
                  << "destination indirect requirement have incompatible types "
                  << "because they have different "
                  << (diff_dims ? "numbers of dimensions." :
                                  "coordinate types.");
              error.raise();
            }
          }
          else
          {
            // Indirect copy
            IndexSpace src_indirect_space =
                src_indirect_requirements[idx].parent.get_index_space();
            IndexSpace dst_indirect_space =
                dst_indirect_requirements[idx].parent.get_index_space();
            // Just check compatibility here since it's really hard to
            // prove that we're actually going to write everything
            bool diff_dims = false;
            if (!runtime->check_types(
                    src_indirect_space.get_type_tag(),
                    dst_indirect_space.get_type_tag(), diff_dims))
            {
              Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
              error << "Copy index space mismatch at index " << idx
                    << " of cross-region " << *this
                    << ". The index spaces of the source indirect requirement "
                       "and "
                    << "the destination indirect requirement have incompatible "
                    << "types because they have different "
                    << (diff_dims ? "numbers of dimensions." :
                                    "coordinate types.");
              error.raise();
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyOp::req_vector_reduce_to_readwrite(
        std::vector<RegionRequirement>& reqs,
        std::vector<unsigned>& changed_idxs)
    //--------------------------------------------------------------------------
    {
      changed_idxs.clear();
      for (unsigned idx = 0; idx < reqs.size(); idx++)
      {
        if (IS_REDUCE(reqs[idx]))
        {
          reqs[idx].privilege = LEGION_READ_WRITE;
          changed_idxs.emplace_back(idx);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyOp::req_vector_reduce_restore(
        std::vector<RegionRequirement>& reqs,
        const std::vector<unsigned>& changed_idxs)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx : changed_idxs) reqs[idx].privilege = LEGION_REDUCE;
    }

    //--------------------------------------------------------------------------
    void CopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PredicatedOp::activate();
      mapper = nullptr;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      copy_fill_priority = 0;
      possible_src_indirect_out_of_range = false;
      possible_dst_indirect_out_of_range = false;
      possible_dst_indirect_aliasing = false;
    }

    //--------------------------------------------------------------------------
    void CopyOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Clear out our region tree state
      src_requirements.clear();
      dst_requirements.clear();
      src_indirect_requirements.clear();
      dst_indirect_requirements.clear();
      grants.clear();
      wait_barriers.clear();
      src_parent_indexes.clear();
      dst_parent_indexes.clear();
      src_versions.clear();
      dst_versions.clear();
      gather_parent_indexes.clear();
      scatter_parent_indexes.clear();
      gather_is_range.clear();
      scatter_is_range.clear();
      gather_versions.clear();
      scatter_versions.clear();
      src_indirect_records.clear();
      dst_indirect_records.clear();
      atomic_locks.clear();
      arrive_barriers.clear();
      across_sources.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (mapper_data != nullptr)
      {
        free(mapper_data);
        mapper_data = nullptr;
        mapper_data_size = 0;
      }
      PredicatedOp::deactivate(false /*free*/);
      // Return this operation to the runtime
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* CopyOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[COPY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind CopyOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return COPY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t CopyOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return src_requirements.size() + dst_requirements.size() +
             src_indirect_requirements.size() +
             dst_indirect_requirements.size();
    }

    //--------------------------------------------------------------------------
    Mappable* CopyOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void CopyOp::log_copy_requirements(void) const
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        const RegionRequirement& req = src_requirements[idx];
        LegionSpy::log_logical_requirement(
            unique_op_id, idx, true /*region*/, req.region.index_space.get_id(),
            req.region.field_space.get_id(), req.region.get_tree_id(),
            req.privilege, req.prop, req.redop,
            req.parent.index_space.get_id());
        LegionSpy::log_requirement_fields(
            unique_op_id, idx, req.instance_fields);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const RegionRequirement& req = dst_requirements[idx];
        LegionSpy::log_logical_requirement(
            unique_op_id, src_requirements.size() + idx, true /*region*/,
            req.region.index_space.get_id(), req.region.field_space.get_id(),
            req.region.get_tree_id(), req.privilege, req.prop, req.redop,
            req.parent.index_space.get_id());
        LegionSpy::log_requirement_fields(
            unique_op_id, src_requirements.size() + idx, req.instance_fields);
      }
      if (!src_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() + dst_requirements.size();
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          const RegionRequirement& req = src_indirect_requirements[idx];
          legion_assert(req.privilege_fields.size() == 1);
          LegionSpy::log_logical_requirement(
              unique_op_id, offset + idx, true /*region*/,
              req.region.index_space.get_id(), req.region.field_space.get_id(),
              req.region.get_tree_id(), req.privilege, req.prop, req.redop,
              req.parent.index_space.get_id());
          LegionSpy::log_requirement_fields(
              unique_op_id, offset + idx, req.privilege_fields);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() +
                              dst_requirements.size() +
                              src_indirect_requirements.size();
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          const RegionRequirement& req = dst_indirect_requirements[idx];
          legion_assert(req.privilege_fields.size() == 1);
          LegionSpy::log_logical_requirement(
              unique_op_id, offset + idx, true /*region*/,
              req.region.index_space.get_id(), req.region.field_space.get_id(),
              req.region.get_tree_id(), req.privilege, req.prop, req.redop,
              req.parent.index_space.get_id());
          LegionSpy::log_requirement_fields(
              unique_op_id, offset + idx, req.privilege_fields);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Initialize the privilege and mapping paths for all of the
      // region requirements that we have
      log_copy_requirements();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis(false /*permit projection*/);

      // Perform reduce dependence analysis as if it was READ_WRITE
      // so that we can get the version numbers correct
      std::vector<unsigned> changed_idxs;
      req_vector_reduce_to_readwrite(dst_requirements, changed_idxs);
      analyze_region_requirements();
      req_vector_reduce_restore(dst_requirements, changed_idxs);
    }

    //--------------------------------------------------------------------------
    void CopyOp::perform_base_dependence_analysis(bool permit_projection)
    //--------------------------------------------------------------------------
    {
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(
            this, wait_barriers, arrive_barriers);
    }

    //--------------------------------------------------------------------------
    void CopyOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Otherwise we need to do the things to clean up this operation
      // Mark that this operation has completed both
      // execution and mapping indicating that we are done
      // Do it in this order to avoid calling 'execute_trigger'
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Do our versioning analysis and then add it to the ready queue
      std::set<RtEvent> preconditions;
      src_versions.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        if (src_parent_indexes[idx] == TRACED_PARENT_INDEX)
          src_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, src_requirements[idx], idx, false /*skip*/, true /*force*/);
        perform_versioning_analysis(
            idx, src_requirements[idx], src_versions[idx], preconditions);
      }
      unsigned offset = src_requirements.size();
      dst_versions.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        if (dst_parent_indexes[idx] == TRACED_PARENT_INDEX)
          dst_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, dst_requirements[idx], offset + idx, false /*skip*/,
              true /*force*/);
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        perform_versioning_analysis(
            offset + idx, dst_requirements[idx], dst_versions[idx],
            preconditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
      }
      offset += dst_requirements.size();
      if (!src_indirect_requirements.empty())
      {
        gather_versions.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          if (gather_parent_indexes[idx] == TRACED_PARENT_INDEX)
            gather_parent_indexes[idx] = parent_ctx->find_parent_region_index(
                this, src_indirect_requirements[idx], offset + idx,
                false /*skip*/, true /*force*/);
          perform_versioning_analysis(
              offset + idx, src_indirect_requirements[idx],
              gather_versions[idx], preconditions);
        }
        offset += src_indirect_requirements.size();
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_versions.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          if (scatter_parent_indexes[idx] == TRACED_PARENT_INDEX)
            scatter_parent_indexes[idx] = parent_ctx->find_parent_region_index(
                this, dst_indirect_requirements[idx], offset + idx,
                false /*skip*/, true /*force*/);
          perform_versioning_analysis(
              offset + idx, dst_indirect_requirements[idx],
              scatter_versions[idx], preconditions);
        }
      }
      if (!preconditions.empty())
      {
        const RtEvent ready = Runtime::merge_events(preconditions);
        enqueue_ready_operation(ready);
      }
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const TraceInfo trace_info(this);
      Mapper::MapCopyInput input;
      Mapper::MapCopyOutput output;
      input.src_instances.resize(src_requirements.size());
      input.dst_instances.resize(dst_requirements.size());
      input.src_indirect_instances.resize(src_indirect_requirements.size());
      input.dst_indirect_instances.resize(dst_indirect_requirements.size());
      input.src_collectives.resize(src_requirements.size());
      input.dst_collectives.resize(dst_requirements.size());
      input.src_indirect_collectives.resize(src_indirect_requirements.size());
      input.dst_indirect_collectives.resize(dst_indirect_requirements.size());
      output.src_instances.resize(src_requirements.size());
      output.dst_instances.resize(dst_requirements.size());
      output.src_indirect_instances.resize(src_indirect_requirements.size());
      output.dst_indirect_instances.resize(dst_indirect_requirements.size());
      output.src_source_instances.resize(src_requirements.size());
      output.dst_source_instances.resize(dst_requirements.size());
      output.src_indirect_source_instances.resize(
          src_indirect_requirements.size());
      output.dst_indirect_source_instances.resize(
          dst_indirect_requirements.size());
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      if (mapper->request_valid_instances)
      {
        // First go through and do the traversals to find the valid instances
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        {
          InstanceSet valid_instances;
          local::FieldMaskMap<ReplicatedView> collectives;
          physical_premap_region(
              idx, src_requirements[idx], src_versions[idx], valid_instances,
              collectives, map_applied_conditions);
          // Convert these to the valid set of mapping instances
          // No need to filter for copies
          prepare_for_mapping(
              valid_instances, collectives, input.src_instances[idx],
              input.src_collectives[idx]);
        }
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          InstanceSet valid_instances;
          local::FieldMaskMap<ReplicatedView> collectives;
          // Little bit of a hack here, if we are going to do a reduction
          // explicit copy, switch the privileges to read-write when doing
          // the registration since we know we are using normal instances
          const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
          if (is_reduce_req)
            dst_requirements[idx].privilege = LEGION_READ_WRITE;
          physical_premap_region(
              idx + src_requirements.size(), dst_requirements[idx],
              dst_versions[idx], valid_instances, collectives,
              map_applied_conditions);
          // No need to filter for copies
          prepare_for_mapping(
              valid_instances, collectives, input.dst_instances[idx],
              input.dst_collectives[idx]);
          // Switch the privileges back when we are done
          if (is_reduce_req)
            dst_requirements[idx].privilege = LEGION_REDUCE;
        }
        if (!src_indirect_requirements.empty())
        {
          const unsigned offset =
              src_requirements.size() + dst_requirements.size();
          for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          {
            InstanceSet valid_instances;
            local::FieldMaskMap<ReplicatedView> collectives;
            physical_premap_region(
                offset + idx, src_indirect_requirements[idx],
                gather_versions[idx], valid_instances, collectives,
                map_applied_conditions);
            // Convert these to the valid set of mapping instances
            // No need to filter for copies
            prepare_for_mapping(
                valid_instances, collectives, input.src_indirect_instances[idx],
                input.src_indirect_collectives[idx]);
          }
        }
        if (!dst_indirect_requirements.empty())
        {
          const unsigned offset = src_requirements.size() +
                                  dst_requirements.size() +
                                  src_indirect_requirements.size();
          for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          {
            InstanceSet valid_instances;
            local::FieldMaskMap<ReplicatedView> collectives;
            physical_premap_region(
                offset + idx, dst_indirect_requirements[idx],
                scatter_versions[idx], valid_instances, collectives,
                map_applied_conditions);
            // Convert these to the valid set of mapping instances
            // No need to filter for copies
            prepare_for_mapping(
                valid_instances, collectives, input.dst_indirect_instances[idx],
                input.dst_indirect_collectives[idx]);
          }
        }
      }
      // Now we can ask the mapper what to do
      mapper->invoke_map_copy(this, input, output);
      copy_fill_priority = output.copy_fill_priority;
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(
            mapper, output.profiling_requests.requested_measurements,
            profiling_requests, true /*warn*/);
        profiling_priority = output.profiling_priority;
        legion_assert(!profiling_reported.exists());
        profiling_reported = Runtime::create_rt_user_event();
      }
      // Now we can carry out the mapping requested by the mapper
      // and issue the across copies, first set up the sync precondition
      ApEvent init_precondition = compute_sync_precondition(trace_info);
      // Register the source and destination regions
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        InstanceSet src_targets, dst_targets, gather_targets, scatter_targets;
        // Make a user event for when this copy across is done
        // and add it to the set of copy complete events
        const ApUserEvent local_postcondition =
            Runtime::create_ap_user_event(&trace_info);
        std::vector<PhysicalManager*> src_sources;
        record_completion_effect(local_postcondition);
        // Convert the src_targets and dst_targets first so we can do any
        // exchanges for collective points
        // The common case
        int src_virtual = -1;
        // Do the conversion and check for errors
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        src_virtual = perform_conversion<SRC_REQ>(
            idx, src_requirements[idx], output.src_instances[idx],
            output.src_source_instances[idx], src_sources, src_targets,
            is_reduce_req);
        log_mapping_decision(idx, src_requirements[idx], src_targets);
        const size_t dst_idx = src_requirements.size() + idx;
        // Little bit of a hack here, if we are going to do a reduction
        // explicit copy, switch the privileges to read-write when doing
        // the registration since we know we are using normal instances
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        std::vector<PhysicalManager*> dst_sources;
        perform_conversion<DST_REQ>(
            idx, dst_requirements[idx], output.dst_instances[idx],
            output.dst_source_instances[idx], dst_sources, dst_targets);
        log_mapping_decision(dst_idx, dst_requirements[idx], dst_targets);
        // Do any exchanges needed for collective cooperation
        const bool src_indirect = (idx < src_indirect_requirements.size());
        const bool dst_indirect = (idx < dst_indirect_requirements.size());
        const ApUserEvent local_precondition =
            (src_indirect || dst_indirect) ?
                Runtime::create_ap_user_event(&trace_info) :
                ApUserEvent::NO_AP_USER_EVENT;
        ApEvent collective_precondition, collective_postcondition;
        ApEvent src_ready, dst_ready, gather_ready, scatter_ready;
        // Track applied conditions special for copy-across
        std::set<RtEvent> perform_ready_events;
        if (src_indirect)
        {
          // Do the exchange to get it in flight
          RtEvent exchange_done = exchange_indirect_records(
              idx, local_precondition, local_postcondition,
              collective_precondition, collective_postcondition, trace_info,
              src_targets, src_requirements[idx], src_indirect_records[idx],
              true /*source*/);
          if (exchange_done.exists())
            perform_ready_events.insert(exchange_done);
        }
        if (dst_indirect)
        {
          // It's ok to overwrite the collective postcondition because we
          // guarantee that they will be the same for multiple calls
          // to exchange for the same operation
          RtEvent exchange_done = exchange_indirect_records(
              idx, local_precondition, local_postcondition,
              collective_precondition, collective_postcondition, trace_info,
              dst_targets, dst_requirements[idx], dst_indirect_records[idx],
              false /*source*/);
          if (exchange_done.exists())
            perform_ready_events.insert(exchange_done);
        }
        if (src_virtual < 0)
        {
          // Don't track source views of copy across operations here,
          // as they will do later when the realm copies are recorded.
          PhysicalTraceInfo src_info(
              trace_info, idx, false /*update validity*/);
          const bool record_valid =
              (output.untracked_valid_srcs.find(idx) ==
               output.untracked_valid_srcs.end());
          src_ready = physical_perform_updates_and_registration(
              src_requirements[idx], src_versions[idx], idx, init_precondition,
              src_indirect ? collective_postcondition :
                             (ApEvent)local_postcondition,
              src_targets, src_sources, src_info, map_applied_conditions,
              false /*check collective*/, record_valid);
        }
        else
        {
          legion_assert(src_targets.size() == 1);
          legion_assert(src_targets[0].is_virtual_ref());
          src_targets.clear();
          if (!output.src_source_instances[idx].empty())
            physical_convert_sources(
                src_requirements[idx], output.src_source_instances[idx],
                across_sources,
                runtime->safe_mapper ? &acquired_instances : nullptr);
          // This is a bit weird but we don't currently have any mechanism
          // for passing the reservations that we find in these cases through
          // to the CopyAcrossAnalysis and through the CopyFillAggregator so
          // for now we're just going to promote privileges on any source and
          // destination requirements to exclusive which is sound with the
          // logical dependence analysis since we're not changing privileges
          if (IS_ATOMIC(src_requirements[idx]))
            src_requirements[idx].prop = LEGION_EXCLUSIVE;
          if (IS_ATOMIC(dst_requirements[idx]))
            dst_requirements[idx].prop = LEGION_EXCLUSIVE;
        }
        // Don't track target views of copy across operations here,
        // as they will do later when the realm copies are recorded.
        PhysicalTraceInfo dst_info(
            trace_info, dst_idx, false /*update_validity*/);
        dst_ready = physical_perform_updates_and_registration(
            dst_requirements[idx], dst_versions[idx], dst_idx,
            init_precondition,
            dst_indirect ? collective_postcondition :
                           (ApEvent)local_postcondition,
            dst_targets, dst_sources, dst_info,
            (src_virtual >= 0) ? perform_ready_events : map_applied_conditions,
            false /*check collective*/, true /*record valid*/,
            // Only check initialized if we don't have an indirection.
            // If we have an indirection then it is impossible to know
            // if we writing everything or not
            (idx >= src_indirect_requirements.size()) &&
                (idx >= dst_indirect_requirements.size()));
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
        if (idx < src_indirect_requirements.size())
        {
          std::vector<MappingInstance> gather_instances(1);
          if (idx < output.src_indirect_instances.size())
            gather_instances[0] = output.src_indirect_instances[idx];
          else
            gather_instances.clear();
          std::vector<PhysicalManager*> gather_sources;
          perform_conversion<GATHER_REQ>(
              idx, src_indirect_requirements[idx], gather_instances,
              output.src_indirect_source_instances[idx], gather_sources,
              gather_targets);
          // Now do the registration
          const size_t gather_idx =
              src_requirements.size() + dst_requirements.size() + idx;
          PhysicalTraceInfo gather_info(trace_info, gather_idx);
          const bool record_valid =
              (output.untracked_valid_ind_srcs.find(idx) ==
               output.untracked_valid_ind_srcs.end());
          gather_ready = physical_perform_updates_and_registration(
              src_indirect_requirements[idx], gather_versions[idx], gather_idx,
              init_precondition, local_postcondition, gather_targets,
              gather_sources, gather_info, map_applied_conditions,
              false /*check collective*/, record_valid);
          log_mapping_decision(
              gather_idx, src_indirect_requirements[idx], gather_targets);
          if (output.compute_preimages)
          {
            // Check that all the gather instances are in host memories
            // since Realm doesn't currently support GPU preimages
            for (unsigned idx = 0; idx < gather_targets.size(); idx++)
            {
              PhysicalManager* manager =
                  gather_targets[idx].get_physical_manager();
              const Memory::Kind kind = manager->memory_manager->memory.kind();
              if ((kind != Memory::GLOBAL_MEM) &&
                  (kind != Memory::SYSTEM_MEM) &&
                  (kind != Memory::REGDMA_MEM) &&
                  (kind != Memory::SOCKET_MEM) && (kind != Memory::Z_COPY_MEM))
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error << "Invalid mapper output from invocation of "
                      << "'map_copy' on mapper " << *mapper << " for " << *this
                      << ". Mapper requested that Legion perform preimage "
                         "optimization on "
                      << "the source indirection instances but mapped at least "
                         "one of "
                      << "the source indirection instances to a " << kind
                      << " which is not a host-visible memory. Realm only "
                         "supports preimage "
                      << "computations on host-visibile memories (see Legion "
                         "issue #516 for "
                      << "more details). For now, please ensure that all "
                         "indirection instances "
                      << "are in host-visible memory when requesting the "
                      << "preimage optimization for copy operations.";
                error.raise();
              }
            }
          }
        }
        if (idx < dst_indirect_requirements.size())
        {
          std::vector<MappingInstance> scatter_instances(1);
          if (idx < output.dst_indirect_instances.size())
            scatter_instances[0] = output.dst_indirect_instances[idx];
          else
            scatter_instances.clear();
          std::vector<PhysicalManager*> scatter_sources;
          perform_conversion<SCATTER_REQ>(
              idx, dst_indirect_requirements[idx], scatter_instances,
              output.dst_indirect_source_instances[idx], scatter_sources,
              scatter_targets);
          // Now do the registration
          const size_t scatter_idx = src_requirements.size() +
                                     dst_requirements.size() +
                                     src_indirect_requirements.size() + idx;
          const bool record_valid =
              (output.untracked_valid_ind_dsts.find(idx) ==
               output.untracked_valid_ind_dsts.end());
          PhysicalTraceInfo scatter_info(trace_info, scatter_idx);
          scatter_ready = physical_perform_updates_and_registration(
              dst_indirect_requirements[idx], scatter_versions[idx],
              scatter_idx, init_precondition, local_postcondition,
              scatter_targets, scatter_sources, scatter_info,
              map_applied_conditions, false /*check collective*/, record_valid);
          log_mapping_decision(
              scatter_idx, dst_indirect_requirements[idx], scatter_targets);
          if (output.compute_preimages)
          {
            // Check that all the scatter instances are in host memories
            // since Realm doesn't currently support GPU preimages
            for (unsigned idx = 0; idx < scatter_targets.size(); idx++)
            {
              PhysicalManager* manager =
                  scatter_targets[idx].get_physical_manager();
              const Memory::Kind kind = manager->memory_manager->memory.kind();
              if ((kind != Memory::GLOBAL_MEM) &&
                  (kind != Memory::SYSTEM_MEM) &&
                  (kind != Memory::REGDMA_MEM) &&
                  (kind != Memory::SOCKET_MEM) && (kind != Memory::Z_COPY_MEM))
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Invalid mapper output from invocation of "
                    << "'map_copy' on mapper " << *mapper << " for " << *this
                    << ". Mapper requested that Legion perform preimage "
                       "optimization on "
                    << "the destination indirection instances but mapped at "
                       "least one "
                    << "of the destination indirection instances to a " << kind
                    << " memory which is not a host-visible memory. Realm only "
                       "supports "
                    << "preimage computations on host-visibile memories (see "
                       "Legion "
                    << "issue #516 for more details). For now, please ensure "
                       "that all "
                    << "indirection instances are in host-visible memory when "
                    << "requesting the preimage optimization for copy "
                       "operations.";
                error.raise();
              }
            }
          }
        }
        // If we made it here, we passed all our error-checking so
        // now we can issue the copy/reduce across operation
        // If we have local completion events then we need to make
        // sure that all those effects have been applied before we
        // can perform the copy across operation, so defer it if necessary
        PhysicalTraceInfo physical_trace_info(
            idx, trace_info, idx + src_requirements.size());
        RtEvent perform_precondition;
        if (!perform_ready_events.empty())
          perform_precondition = Runtime::merge_events(perform_ready_events);
        if (perform_precondition.exists() &&
            !perform_precondition.has_triggered())
        {
          InstanceSet* deferred_src = new InstanceSet();
          deferred_src->swap(src_targets);
          InstanceSet* deferred_dst = new InstanceSet();
          deferred_dst->swap(dst_targets);
          InstanceSet* deferred_gather = nullptr;
          if (!gather_targets.empty())
          {
            deferred_gather = new InstanceSet();
            deferred_gather->swap(gather_targets);
          }
          InstanceSet* deferred_scatter = nullptr;
          if (!scatter_targets.empty())
          {
            deferred_scatter = new InstanceSet();
            deferred_scatter->swap(scatter_targets);
          }
          RtUserEvent deferred_applied = Runtime::create_rt_user_event();
          DeferredCopyAcross args(
              this, physical_trace_info, idx, init_precondition, src_ready,
              dst_ready, gather_ready, scatter_ready, local_precondition,
              local_postcondition, collective_precondition,
              collective_postcondition, true_guard, deferred_applied,
              deferred_src, deferred_dst, deferred_gather, deferred_scatter,
              output.compute_preimages,
              output.compute_preimages && output.shadow_indirections);
          runtime->issue_runtime_meta_task(
              args, LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
          map_applied_conditions.insert(deferred_applied);
        }
        else
          perform_copy_across(
              idx, init_precondition, src_ready, dst_ready, gather_ready,
              scatter_ready, local_precondition, local_postcondition,
              collective_precondition, collective_postcondition, true_guard,
              src_targets, dst_targets,
              gather_targets.empty() ? nullptr : &gather_targets,
              scatter_targets.empty() ? nullptr : &scatter_targets,
              physical_trace_info, map_applied_conditions,
              output.compute_preimages,
              output.compute_preimages && output.shadow_indirections);
      }
      if (is_recording())
        trace_info.record_complete_replay(map_applied_conditions);
      // Mark that we completed mapping
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(
            mapping_applied, acquired_instances);
      complete_mapping(mapping_applied);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void CopyOp::perform_copy_across(
        const unsigned index, const ApEvent init_precondition,
        const ApEvent src_ready, const ApEvent dst_ready,
        const ApEvent gather_ready, const ApEvent scatter_ready,
        const ApUserEvent local_precondition,
        const ApUserEvent local_postcondition,
        const ApEvent collective_precondition,
        const ApEvent collective_postcondition,
        const PredEvent predication_guard, const InstanceSet& src_targets,
        const InstanceSet& dst_targets, const InstanceSet* gather_targets,
        const InstanceSet* scatter_targets, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& applied_conditions, const bool compute_preimages,
        const bool shadow_indirections)
    //--------------------------------------------------------------------------
    {
      ApEvent copy_post;
      if (scatter_targets == nullptr)
      {
        if (gather_targets == nullptr)
        {
          legion_assert(!local_precondition.exists());
          // Normal copy across
          copy_post = copy_across(
              src_requirements[index], dst_requirements[index],
              src_versions[index], dst_versions[index], src_targets,
              dst_targets, across_sources, index, trace_info.dst_index,
              init_precondition, src_ready, dst_ready, predication_guard,
              atomic_locks[index], trace_info, applied_conditions);
        }
        else
        {
          // Gather copy
          legion_assert(!src_indirect_records[index].empty());
          copy_post = gather_across(
              src_requirements[index], src_indirect_requirements[index],
              dst_requirements[index], src_indirect_records[index], src_targets,
              (*gather_targets), dst_targets, index,
              src_requirements.size() + dst_requirements.size() + index,
              src_requirements.size() + index, gather_is_range[index],
              init_precondition, src_ready, dst_ready, gather_ready,
              predication_guard, collective_precondition,
              collective_postcondition, local_precondition, atomic_locks[index],
              trace_info, applied_conditions,
              possible_src_indirect_out_of_range, compute_preimages,
              shadow_indirections);
        }
      }
      else
      {
        if (gather_targets == nullptr)
        {
          // Scatter copy
          legion_assert(!dst_indirect_records[index].empty());
          copy_post = scatter_across(
              src_requirements[index], dst_indirect_requirements[index],
              dst_requirements[index], src_targets, (*scatter_targets),
              dst_targets, dst_indirect_records[index], index,
              src_requirements.size() + dst_requirements.size() + index,
              src_requirements.size() + index, scatter_is_range[index],
              init_precondition, src_ready, dst_ready, scatter_ready,
              predication_guard, collective_precondition,
              collective_postcondition, local_precondition, atomic_locks[index],
              trace_info, applied_conditions,
              possible_dst_indirect_out_of_range,
              possible_dst_indirect_aliasing, compute_preimages,
              shadow_indirections);
        }
        else
        {
          legion_assert(gather_is_range[index] == scatter_is_range[index]);
          legion_assert(!src_indirect_records[index].empty());
          legion_assert(!dst_indirect_records[index].empty());
          // Full indirection copy
          copy_post = indirect_across(
              src_requirements[index], src_indirect_requirements[index],
              dst_requirements[index], dst_indirect_requirements[index],
              src_targets, dst_targets, src_indirect_records[index],
              (*gather_targets), dst_indirect_records[index],
              (*scatter_targets), index, src_requirements.size() + index,
              src_requirements.size() + dst_requirements.size() + index,
              src_requirements.size() + dst_requirements.size() +
                  src_indirect_requirements.size() + index,
              gather_is_range[index], init_precondition, src_ready, dst_ready,
              gather_ready, scatter_ready, predication_guard,
              collective_precondition, collective_postcondition,
              local_precondition, atomic_locks[index], trace_info,
              applied_conditions, possible_src_indirect_out_of_range,
              possible_dst_indirect_out_of_range,
              possible_dst_indirect_aliasing, compute_preimages,
              shadow_indirections);
        }
      }
      if (is_recording())
      {
        legion_assert((tpl != nullptr) && tpl->is_recording());
        // This can happen in cases when the copy index space is empty
        if (!copy_post.exists())
          copy_post = execution_fence_event;
      }
      Runtime::trigger_event(
          local_postcondition, copy_post, trace_info, applied_conditions);
    }

    //--------------------------------------------------------------------------
    void CopyOp::DeferredCopyAcross::execute(void) const
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> applied_conditions;
      copy->perform_copy_across(
          index, init_precondition, src_ready, dst_ready, gather_ready,
          scatter_ready, local_precondition, local_postcondition,
          collective_precondition, collective_postcondition, guard,
          *src_targets, *dst_targets, gather_targets, scatter_targets,
          *trace_info, applied_conditions, compute_preimages,
          shadow_indirections);
      if (!applied_conditions.empty())
        Runtime::trigger_event(
            applied, Runtime::merge_events(applied_conditions));
      else
        Runtime::trigger_event(applied);
      delete trace_info;
      delete src_targets;
      delete dst_targets;
      if (gather_targets != nullptr)
        delete gather_targets;
      if (scatter_targets != nullptr)
        delete scatter_targets;
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_complete(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      // Chain all the unlock and barrier arrivals off of the
      // complete event
      if (!arrive_barriers.empty())
      {
        for (PhaseBarrier& barrier : arrive_barriers)
        {
          LegionSpy::log_phase_barrier_arrival(
              unique_op_id, barrier.phase_barrier);
          runtime->phase_barrier_arrive(
              barrier.phase_barrier, 1 /*count*/, complete);
        }
      }
      complete_operation(complete);
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_copy_profiling();
      commit_operation(true /*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    bool CopyOp::record_trace_hash(TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher hasher;
      hasher.hash(get_operation_kind());
      for (const RegionRequirement& req : src_requirements)
        hash_requirement(hasher, req);
      for (const RegionRequirement& req : dst_requirements)
        hash_requirement(hasher, req);
      for (const RegionRequirement& req : src_indirect_requirements)
        hash_requirement(hasher, req);
      for (const RegionRequirement& req : dst_indirect_requirements)
        hash_requirement(hasher, req);
      // Not including the fields grants, wait_barriers, arrive_barriers.
      hasher.hash<bool>(is_index_space);
      if (is_index_space)
        hasher.hash(index_domain);
      return recorder.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    void CopyOp::finalize_copy_profiling(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(profiling_reported.exists());
      if (outstanding_profiling_requests.load() == 0)
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::CopyProfilingInfo info;
        info.total_reports = 0;
        info.src_index = 0;
        info.dst_index = 0;
        info.fill_response = false;  // make valgrind happy
        mapper->invoke_copy_report_profiling(this, info);
        Runtime::trigger_event(profiling_reported);
      }
    }

    //--------------------------------------------------------------------------
    void CopyOp::report_interfering_requirements(unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx1 < idx2);
      // The logical dependence analysis can report this because there are
      // interfering fields and regions, check to make sure there are also
      // interfering privileges and index spaces
      const RegionRequirement& req1 = get_requirement(idx1);
      const RegionRequirement& req2 = get_requirement(idx2);
      if (IS_READ_ONLY(req1) && IS_READ_ONLY(req2))
        return;
      if (IS_REDUCE(req1) && IS_REDUCE(req2) && (req1.redop == req2.redop))
        return;
      if (!runtime->are_disjoint(
              req1.region.get_index_space(), req2.region.get_index_space()))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Found aliasing region requirements for " << *this
              << " between " << get_requirement_offset(idx1) << " of "
              << get_requirement_name(idx1) << " region requirement and "
              << get_requirement_offset(idx2) << " of "
              << get_requirement_name(idx2) << ". Aliased region "
              << "requirements can lead to races and are not permitted.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    RtEvent CopyOp::exchange_indirect_records(
        const unsigned index, const ApEvent local_pre, const ApEvent local_post,
        ApEvent& collective_pre, ApEvent& collective_post,
        const TraceInfo& trace_info, const InstanceSet& insts,
        const RegionRequirement& req, std::vector<IndirectRecord>& records,
        const bool sources)
    //--------------------------------------------------------------------------
    {
      collective_pre = local_pre;
      collective_post = local_post;
      records.emplace_back(IndirectRecord(req, insts, 1 /*total points*/));
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    unsigned CopyOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (idx < src_parent_indexes.size())
      {
        legion_assert(src_parent_indexes[idx] != TRACED_PARENT_INDEX);
        return src_parent_indexes[idx];
      }
      idx -= src_parent_indexes.size();
      if (idx < dst_parent_indexes.size())
      {
        legion_assert(dst_parent_indexes[idx] != TRACED_PARENT_INDEX);
        return dst_parent_indexes[idx];
      }
      idx -= dst_parent_indexes.size();
      if (idx < gather_parent_indexes.size())
      {
        legion_assert(gather_parent_indexes[idx] != TRACED_PARENT_INDEX);
        return gather_parent_indexes[idx];
      }
      idx -= gather_parent_indexes.size();
      legion_assert(idx < scatter_parent_indexes.size());
      legion_assert(scatter_parent_indexes[idx] != TRACED_PARENT_INDEX);
      return scatter_parent_indexes[idx];
    }

    //--------------------------------------------------------------------------
    void CopyOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectCopySrcInput input;
      Mapper::SelectCopySrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      input.is_src = false;
      input.is_dst = false;
      input.is_src_indirect = false;
      input.is_dst_indirect = false;
      unsigned mod_index = index;
      if (mod_index < src_requirements.size())
      {
        input.region_req_index = mod_index;
        input.is_src = true;
      }
      else
      {
        mod_index -= src_requirements.size();
        if (mod_index < dst_requirements.size())
        {
          input.region_req_index = mod_index;
          input.is_dst = true;
        }
        else
        {
          mod_index -= dst_requirements.size();
          if (mod_index < src_indirect_requirements.size())
          {
            input.region_req_index = mod_index;
            input.is_src_indirect = true;
          }
          else
          {
            mod_index -= src_indirect_requirements.size();
            legion_assert(mod_index < dst_indirect_requirements.size());
            input.is_dst_indirect = true;
          }
        }
      }
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_copy_sources(this, input, output);
      // Fill in the ranking based on the output
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>* CopyOp::get_acquired_instances_ref(
        void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void CopyOp::update_atomic_locks(
        const unsigned index, Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      legion_assert(index < atomic_locks.size());
      std::map<Reservation, bool>& local_locks = atomic_locks[index];
      std::map<Reservation, bool>::iterator finder = local_locks.find(lock);
      if (finder != local_locks.end())
      {
        if (!finder->second && exclusive)
          finder->second = true;
      }
      else
        local_locks[lock] = exclusive;
    }

    //--------------------------------------------------------------------------
    UniqueID CopyOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t CopyOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void CopyOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int CopyOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* CopyOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& CopyOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    void CopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void CopyOp::complete_replay(ApEvent copy_complete_event)
    //--------------------------------------------------------------------------
    {
      // Handle the case for marking when the copy completes
      record_completion_effect(copy_complete_event);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    const VersionInfo& CopyOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (idx < src_versions.size())
        return src_versions[idx];
      idx -= src_versions.size();
      if (idx < dst_versions.size())
        return dst_versions[idx];
      idx -= dst_versions.size();
      if (idx < gather_versions.size())
        return gather_versions[idx];
      idx -= gather_versions.size();
      legion_assert(idx < scatter_versions.size());
      return scatter_versions[idx];
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& CopyOp::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (idx < src_requirements.size())
        return src_requirements[idx];
      idx -= src_requirements.size();
      if (idx < dst_requirements.size())
        return dst_requirements[idx];
      idx -= dst_requirements.size();
      if (idx < src_indirect_requirements.size())
        return src_indirect_requirements[idx];
      idx -= src_indirect_requirements.size();
      legion_assert(idx < dst_indirect_requirements.size());
      return dst_indirect_requirements[idx];
    }

    //--------------------------------------------------------------------------
    unsigned CopyOp::get_requirement_offset(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (idx < src_requirements.size())
        return idx;
      idx -= src_requirements.size();
      if (idx < dst_requirements.size())
        return idx;
      idx -= dst_requirements.size();
      if (idx < src_indirect_requirements.size())
        return idx;
      idx -= src_indirect_requirements.size();
      legion_assert(idx < dst_indirect_requirements.size());
      return idx;
    }

    //--------------------------------------------------------------------------
    const char* CopyOp::get_requirement_name(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (idx < src_requirements.size())
        return "source";
      idx -= src_requirements.size();
      if (idx < dst_requirements.size())
        return "destination";
      idx -= dst_requirements.size();
      if (idx < src_indirect_requirements.size())
        return "source indirect";
      idx -= src_indirect_requirements.size();
      legion_assert(idx < dst_indirect_requirements.size());
      return "destination indirect";
    }

    //--------------------------------------------------------------------------
    template<CopyOp::ReqType REQ_TYPE>
    /*static*/ const char* CopyOp::get_req_type_name(void)
    //--------------------------------------------------------------------------
    {
      static const char* req_type_names[4] = {
          "source",
          "destination",
          "source indirect",
          "destination indirect",
      };
      return req_type_names[REQ_TYPE];
    }

    //--------------------------------------------------------------------------
    template<CopyOp::ReqType REQ_TYPE>
    int CopyOp::perform_conversion(
        unsigned ridx, const RegionRequirement& req,
        std::vector<MappingInstance>& output,
        std::vector<MappingInstance>& input,
        std::vector<PhysicalManager*>& sources, InstanceSet& targets,
        bool is_reduce)
    //--------------------------------------------------------------------------
    {
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      if (!input.empty())
        physical_convert_sources(
            req, input, sources,
            runtime->safe_mapper ? &acquired_instances : nullptr);
      int composite_idx = physical_convert_mapping(
          req, output, targets, bad_tree, missing_fields, &acquired_instances,
          unacquired, runtime->safe_mapper);
      if (bad_tree > 0)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error
            << "Invalid mapper output from invocation of 'map_copy' on mapper "
            << *mapper << ". Mapper selected an instance from "
            << "region tree " << bad_tree << " to satisfy "
            << get_req_type_name<REQ_TYPE>() << " region requirement " << ridx
            << "for explicit region-to_region " << *this
            << " but the logical region for this requirement is from region "
               "tree "
            << req.region.get_tree_id() << ".";
        error.raise();
      }
      if (!missing_fields.empty())
      {
        for (const FieldID& fid : missing_fields)
        {
          const void* name;
          size_t name_size;
          if (!runtime->retrieve_semantic_information(
                  req.region.get_field_space(), fid, LEGION_NAME_SEMANTIC_TAG,
                  name, name_size, true, false))
            name = "(no name)";
          log_legion.error(
              "Missing instance for field %s (FieldID: %d)",
              static_cast<const char*>(name), fid);
        }
        Error error(LEGION_MAPPER_EXCEPTION);
        error
            << "Invalid mapper output from invocation of 'map_copy' on mapper "
            << *mapper << ". Mapper failed to specify a physical "
            << "instance for " << missing_fields.size() << " fields of the "
            << get_req_type_name<REQ_TYPE>() << " region requirement " << ridx
            << "of explicit region-to-region " << *this
            << ". The missing fields are listed above.";
        error.raise();
      }
      if (!unacquired.empty())
      {
        for (PhysicalManager* manager : unacquired)
        {
          if (acquired_instances.find(manager) == acquired_instances.end())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from 'map_copy' invocation on mapper "
                << *mapper << ". Mapper selected physical instance for "
                << get_req_type_name<REQ_TYPE>() << " region requirement "
                << ridx << " of explicit region-to-region " << *this
                << " which has already "
                << "been collected. If the mapper had properly acquired this "
                   "instance "
                << "as part of the mapper call it would have detected this. "
                   "Please "
                << "update the mapper to abide by proper mapping conventions.";
            error.raise();
          }
        }
        // If we did successfully acquire them, still issue the warning
        Warning warning;
        warning << "Mapper " << *mapper << " failed to acquire instances for "
                << get_req_type_name<REQ_TYPE>() << " region requirement "
                << ridx << " of explicit region-to-region " << *this
                << " in 'map_copy' call. "
                << "You may experience undefined behavior as a consequence.";
        warning.raise();
      }
      if (composite_idx >= 0)
      {
        // Destination is not allowed to have composite instances
        if (REQ_TYPE != SRC_REQ)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_copy' on "
                   "mapper "
                << *mapper << ". Mapper requested the creation of a "
                << "virtual instance for " << get_req_type_name<REQ_TYPE>()
                << " region requiremnt " << ridx
                << ". Only source region requirements are permitted to be "
                   "virtual "
                << "instances for explicit region-to-region " << *this << ".";
          error.raise();
        }
        if (is_reduce)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_copy' on "
                   "mapper "
                << *mapper << ". Mapper requested the creation of a "
                << "virtual instance for the " << get_req_type_name<REQ_TYPE>()
                << " requirement " << ridx
                << " of an explicit region-to-region "
                << "reduction for " << *this
                << ". Only real physical instances "
                << "are permitted to be sources of explicit region-to-region "
                   "reductions.";
          error.raise();
        }
        if (ridx < src_indirect_requirements.size())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error
              << "Invalid mapper output from invocation of 'map_copy' on "
                 "mapper "
              << *mapper << ". Mapper requested the creation of a "
              << "virtual instance for " << get_req_type_name<REQ_TYPE>()
              << " region requiremnt " << ridx
              << ". Only source region requirements without source indirection "
              << "requirements are permitted to be virtual instances for "
                 "explicit "
              << "region-to-region " << *this << ".";
          error.raise();
        }
        if (ridx < dst_indirect_requirements.size())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_copy' on "
                   "mapper "
                << *mapper << ". Mapper requested the creation of a "
                << "virtual instance for " << get_req_type_name<REQ_TYPE>()
                << " region requiremnt " << ridx
                << ". Only source region requirements without destination "
                << "indirection requirements are permitted to be virtual "
                << "instances for explicit region-to-region " << *this << ".";
          error.raise();
        }
      }
      if (!runtime->safe_mapper)
        return composite_idx;
      std::vector<LogicalRegion> regions_to_check(1, req.region);
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef& ref = targets[idx];
        InstanceManager* man = ref.get_manager();
        if (man->is_virtual_manager())
          continue;
        PhysicalManager* manager = man->as_physical_manager();
        if (!manager->meets_regions(regions_to_check))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_copy' on "
                   "mapper "
                << *mapper << ". Mapper specified an instance for "
                << get_req_type_name<REQ_TYPE>()
                << " region requirement at index " << ridx << " of " << *this
                << " that does not meet the logical region requirement.";
          error.raise();
        }
      }
      // Make sure all the destinations are real instances, this has
      // to be true for all kinds of explicit copies including reductions
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        if ((REQ_TYPE == SRC_REQ) && (int(idx) == composite_idx))
          continue;
        if (!targets[idx].get_manager()->is_physical_manager())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_copy' on "
                   "mapper "
                << *mapper << ". Mapper specified an illegal "
                << "specialized instance as the target for "
                << get_req_type_name<REQ_TYPE>() << " region requirement "
                << ridx << " of " << *this << ".";
          error.raise();
        }
      }
      return composite_idx;
    }

    //--------------------------------------------------------------------------
    int CopyOp::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return copy_fill_priority;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest& request = requests.add_request(
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, &response,
          sizeof(response), profiling_priority);
      bool has_finish = false;
      for (const ProfilingMeasurementID& it : profiling_requests)
      {
        const Realm::ProfilingMeasurementID measurement =
            (Realm::ProfilingMeasurementID)it;
        request.add_measurement(measurement);
        if (measurement == Realm::PMID_OP_FINISH_EVENT)
          has_finish = true;
      }
      // Need thetimeline for the operation to know how to profile this
      // profiling response
      if (!has_finish && (runtime->profiler != nullptr))
        request.add_measurement(Realm::PMID_OP_FINISH_EVENT);
      handle_profiling_update(count);
      return copy_fill_priority;
    }

    //--------------------------------------------------------------------------
    bool CopyOp::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapper != nullptr);
      const OpProfilingResponse* op_info =
          static_cast<const OpProfilingResponse*>(response.user_data());
      Realm::ProfilingMeasurements::OperationFinishEvent finish_event;
      if (response.get_measurement(finish_event))
        fevent = LgEvent(finish_event.finish_event);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many reports to expect
      const RtEvent mapped = get_mapped_event();
      if (!mapped.has_triggered())
        mapped.wait();
      // If we get here then we can handle the response now
      Mapping::Mapper::CopyProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.src_index = op_info->src;
      info.dst_index = op_info->dst;
      info.total_reports = outstanding_profiling_requests;
      info.fill_response = op_info->fill;
      mapper->invoke_copy_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void CopyOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      legion_assert(!mapped_event.has_triggered());
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void CopyOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_copy(rez, target);
      rez.serialize(copy_fill_priority);
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent CopyOp::copy_across(
        const RegionRequirement& src_req, const RegionRequirement& dst_req,
        const VersionInfo& src_version_info,
        const VersionInfo& dst_version_info, const InstanceSet& src_targets,
        const InstanceSet& dst_targets,
        const std::vector<PhysicalManager*>& sources, unsigned src_index,
        unsigned dst_index, ApEvent precondition, ApEvent src_ready,
        ApEvent dst_ready, PredEvent guard,
        const std::map<Reservation, bool>& reservations,
        const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(
          src_req.instance_fields.size() == dst_req.instance_fields.size());
      RegionNode* src_node = runtime->get_node(src_req.region);
      RegionNode* dst_node = runtime->get_node(dst_req.region);
      IndexSpaceExpression* copy_expr = runtime->intersect_index_spaces(
          src_node->row_source, dst_node->row_source);
      // Quick out if there is nothing to copy to
      if (copy_expr->is_empty())
        return ApEvent::NO_AP_EVENT;
      // Perform the copies/reductions across
      InnerContext* context = find_physical_context(dst_index);
      op::vector<op::FieldMaskMap<InstanceView>> target_views;
      context->convert_analysis_views(dst_targets, target_views);
      if (!src_targets.empty())
      {
        // If we already have the targets there's no need to
        // iterate over the source equivalence sets as we can just
        // build a standard CopyAcrossUnstructured object
        CopyAcrossUnstructured* across = copy_expr->create_across_unstructured(
            reservations, false /*preimages*/, false /*shadow indirections*/);
        across->add_reference();
        across->src_tree_id = src_req.region.get_tree_id();
        across->dst_tree_id = dst_req.region.get_tree_id();
        // Fill in the source fields
        across->initialize_source_fields(src_req, src_targets, trace_info);
        // Fill in the destination fields
        const bool exclusive_redop =
            IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req);
        across->initialize_destination_fields(
            dst_req, dst_targets, trace_info, exclusive_redop);
        // Get the preconditions for this copy
        std::vector<ApEvent> copy_preconditions;
        if (precondition.exists())
          copy_preconditions.emplace_back(precondition);
        if (src_ready.exists())
          copy_preconditions.emplace_back(src_ready);
        if (dst_ready.exists())
          copy_preconditions.emplace_back(dst_ready);
        if (!copy_preconditions.empty())
          precondition = Runtime::merge_events(&trace_info, copy_preconditions);
        ApEvent result = across->execute(
            this, guard, precondition, ApEvent::NO_AP_EVENT,
            ApEvent::NO_AP_EVENT, trace_info);
        if (trace_info.recording)
        {
          // Record this with the trace
          trace_info.record_issue_across(
              result, precondition, precondition, ApEvent::NO_AP_EVENT,
              ApEvent::NO_AP_EVENT, across);
          local::map<UniqueInst, FieldMask> tracing_srcs, tracing_dsts;
          InnerContext* src_context = find_physical_context(src_index);
          std::vector<IndividualView*> source_views;
          src_context->convert_individual_views(src_targets, source_views);
          for (unsigned idx = 0; idx < src_targets.size(); idx++)
          {
            const InstanceRef& ref = src_targets[idx];
            const UniqueInst unique_inst(source_views[idx]);
            tracing_srcs[unique_inst] = ref.get_valid_fields();
          }
          for (unsigned idx = 0; idx < target_views.size(); idx++)
          {
            legion_assert(target_views[idx].size() == 1);
            op::FieldMaskMap<InstanceView>::const_iterator it =
                target_views[idx].begin();
            legion_assert(it->first->is_individual_view());
            IndividualView* view = it->first->as_individual_view();
            const UniqueInst unique_inst(view);
            tracing_dsts[unique_inst] = it->second;
          }
          trace_info.record_across_insts(
              result, src_index, dst_index, LEGION_READ_PRIV, LEGION_WRITE_PRIV,
              copy_expr, tracing_srcs, tracing_dsts, false /*indirect*/,
              false /*indirect*/, map_applied_events);
        }
        if (across->remove_reference())
          delete across;
        return result;
      }
      // Should never need to do any reservations here
      legion_assert(reservations.empty());
      // Get the field indexes for all the fields
      std::vector<unsigned> src_indexes(src_req.instance_fields.size());
      std::vector<unsigned> dst_indexes(dst_req.instance_fields.size());
      src_node->column_source->get_field_indexes(
          src_req.instance_fields, src_indexes);
      dst_node->column_source->get_field_indexes(
          dst_req.instance_fields, dst_indexes);
      FieldMask src_mask, dst_mask;
      for (unsigned idx = 0; idx < dst_indexes.size(); idx++)
      {
        src_mask.set_bit(src_indexes[idx]);
        dst_mask.set_bit(dst_indexes[idx]);
      }
      // Check to see if we have a perfect across-copy
      bool perfect = true;
      for (unsigned idx = 0; idx < src_indexes.size(); idx++)
      {
        if (src_indexes[idx] == dst_indexes[idx])
          continue;
        perfect = false;
        break;
      }
      if (!perfect)
      {
        // Make sure the source indexes are sorted if we're not perfect
        // which will be necessary for the copy-across helper
        std::vector<std::pair<unsigned, unsigned>> pair_indexes;
        pair_indexes.reserve(src_indexes.size());
        for (unsigned idx = 0; idx < src_indexes.size(); idx++)
          pair_indexes.emplace_back(
              std::make_pair(src_indexes[idx], dst_indexes[idx]));
        std::sort(pair_indexes.begin(), pair_indexes.end());
        for (unsigned idx = 0; idx < pair_indexes.size(); idx++)
        {
          src_indexes[idx] = pair_indexes[idx].first;
          dst_indexes[idx] = pair_indexes[idx].second;
        }
      }
      std::vector<IndividualView*> source_views;
      if (!sources.empty())
      {
        InnerContext* src_context = find_physical_context(src_index);
        src_context->convert_individual_views(sources, source_views);
      }
      CopyAcrossAnalysis* analysis = new CopyAcrossAnalysis(
          this, src_index, dst_index, src_req, dst_req, dst_targets,
          target_views, source_views, precondition, dst_ready, guard,
          dst_req.redop, src_indexes, dst_indexes, trace_info, perfect);
      analysis->add_reference();
      const RtEvent traversal_done = analysis->perform_traversal(
          RtEvent::NO_RT_EVENT, src_version_info, map_applied_events);
      // Start with the source mask here in case we need to filter which
      // is all done on the source fields
      analysis->local_exprs.insert(copy_expr, src_mask);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready =
            analysis->perform_remote(traversal_done, map_applied_events);
      RtEvent updates_ready;
      if (remote_ready.exists() || analysis->has_across_updates())
        updates_ready =
            analysis->perform_updates(remote_ready, map_applied_events);
      const ApEvent result =
          analysis->perform_output(updates_ready, map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent CopyOp::gather_across(
        const RegionRequirement& src_req, const RegionRequirement& idx_req,
        const RegionRequirement& dst_req,
        std::vector<IndirectRecord>& src_records,
        const InstanceSet& src_targets, const InstanceSet& idx_targets,
        const InstanceSet& dst_targets, unsigned src_index, unsigned idx_index,
        unsigned dst_index, const bool gather_is_range,
        const ApEvent init_precondition, const ApEvent src_ready,
        const ApEvent dst_ready, const ApEvent idx_ready,
        const PredEvent pred_guard, const ApEvent collective_pre,
        const ApEvent collective_post, const ApUserEvent local_pre,
        const std::map<Reservation, bool>& reservations,
        const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events,
        const bool possible_src_out_of_range, const bool compute_preimages,
        const bool shadow_indirections)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(
          src_req.instance_fields.size() == dst_req.instance_fields.size());
      legion_assert(idx_req.privilege_fields.size() == 1);
      // Get the field indexes for src/dst fields
      IndexSpaceNode* idx_node =
          runtime->get_node(idx_req.region.get_index_space());
      IndexSpaceNode* dst_node =
          runtime->get_node(dst_req.region.get_index_space());
      IndexSpaceExpression* copy_expr =
          (idx_node == dst_node) ?
              dst_node :
              runtime->intersect_index_spaces(idx_node, dst_node);
      // Trigger the source precondition event when all our sources are ready
      std::vector<ApEvent> local_preconditions;
      if (init_precondition.exists())
        local_preconditions.emplace_back(init_precondition);
      if (src_ready.exists())
        local_preconditions.emplace_back(src_ready);
      ApEvent local_precondition;
      if (!local_preconditions.empty())
        local_precondition =
            Runtime::merge_events(&trace_info, local_preconditions);
      Runtime::trigger_event(
          local_pre, local_precondition, trace_info, map_applied_events);
      // Easy out if we're not moving anything
      if (copy_expr->is_empty())
        return local_pre;
      CopyAcrossUnstructured* across = copy_expr->create_across_unstructured(
          reservations, compute_preimages, shadow_indirections);
      across->add_reference();
      // Initialize the source indirection fields
      const InstanceRef& idx_target = idx_targets[0];
      across->initialize_source_indirections(
          src_records, src_req, idx_req, idx_target, gather_is_range,
          possible_src_out_of_range);
      // Initialize the destination fields
      const bool exclusive_redop = IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req);
      across->initialize_destination_fields(
          dst_req, dst_targets, trace_info, exclusive_redop);
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.emplace_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      if (dst_ready.exists())
        copy_preconditions.emplace_back(dst_ready);
      ApEvent src_indirect_ready = idx_ready;
      if (init_precondition.exists())
      {
        if (src_indirect_ready.exists())
          src_indirect_ready = Runtime::merge_events(
              &trace_info, src_indirect_ready, init_precondition);
        else
          src_indirect_ready = init_precondition;
      }
      ApEvent copy_precondition;
      if (!copy_preconditions.empty())
        copy_precondition =
            Runtime::merge_events(&trace_info, copy_preconditions);
      // Launch the copy
      ApEvent copy_post = across->execute(
          this, pred_guard, copy_precondition, src_indirect_ready,
          ApEvent::NO_AP_EVENT, trace_info);
      if (trace_info.recording)
      {
        // Record this with the trace
        trace_info.record_issue_across(
            copy_post, local_precondition, copy_precondition,
            src_indirect_ready, ApEvent::NO_AP_EVENT, across);
        // If we're tracing record the insts for this copy
        local::map<UniqueInst, FieldMask> src_insts, idx_insts, dst_insts;
        // Get the src_insts
        InnerContext* src_context = find_physical_context(src_index);
        std::vector<IndividualView*> source_views;
        src_context->convert_individual_views(src_targets, source_views);
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
        {
          const InstanceRef& ref = src_targets[idx];
          const UniqueInst unique_inst(source_views[idx]);
          src_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the idx_insts
        {
          InnerContext* idx_context = find_physical_context(idx_index);
          std::vector<IndividualView*> indirect_views;
          idx_context->convert_individual_views(idx_targets, indirect_views);
          const UniqueInst unique_inst(indirect_views.back());
          idx_insts[unique_inst] = idx_target.get_valid_fields();
        }
        // Get the dst_insts
        InnerContext* dst_context = find_physical_context(dst_index);
        std::vector<IndividualView*> target_views;
        dst_context->convert_individual_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < dst_targets.size(); idx++)
        {
          const InstanceRef& ref = dst_targets[idx];
          const UniqueInst unique_inst(target_views[idx]);
          dst_insts[unique_inst] = ref.get_valid_fields();
        }
        IndexSpaceNode* src_node =
            runtime->get_node(src_req.region.get_index_space());
        trace_info.record_indirect_insts(
            copy_post, collective_post, src_node, src_insts, map_applied_events,
            LEGION_READ_PRIV);
        trace_info.record_across_insts(
            copy_post, idx_index, dst_index, LEGION_READ_PRIV,
            LEGION_WRITE_PRIV, copy_expr, idx_insts, dst_insts,
            true /*indirect*/, false /*indirect*/, map_applied_events);
      }
      if (across->remove_reference())
        delete across;
      return copy_post;
    }

    //--------------------------------------------------------------------------
    ApEvent CopyOp::scatter_across(
        const RegionRequirement& src_req, const RegionRequirement& idx_req,
        const RegionRequirement& dst_req, const InstanceSet& src_targets,
        const InstanceSet& idx_targets, const InstanceSet& dst_targets,
        std::vector<IndirectRecord>& dst_records, unsigned src_index,
        unsigned idx_index, unsigned dst_index, const bool scatter_is_range,
        const ApEvent init_precondition, const ApEvent src_ready,
        const ApEvent dst_ready, const ApEvent idx_ready,
        const PredEvent pred_guard, const ApEvent collective_pre,
        const ApEvent collective_post, const ApUserEvent local_pre,
        const std::map<Reservation, bool>& reservations,
        const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events,
        const bool possible_dst_out_of_range, const bool possible_dst_aliasing,
        const bool compute_preimages, const bool shadow_indirections)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(
          src_req.instance_fields.size() == dst_req.instance_fields.size());
      legion_assert(idx_req.privilege_fields.size() == 1);
      legion_assert(idx_targets.size() == 1);
      // Get the field indexes for src/dst fields
      IndexSpaceNode* src_node =
          runtime->get_node(src_req.region.get_index_space());
      IndexSpaceNode* idx_node =
          runtime->get_node(idx_req.region.get_index_space());
      IndexSpaceExpression* copy_expr =
          (idx_node == src_node) ?
              idx_node :
              runtime->intersect_index_spaces(src_node, idx_node);
      // Trigger the source precondition event when all our sources are ready
      std::vector<ApEvent> local_preconditions;
      if (init_precondition.exists())
        local_preconditions.emplace_back(init_precondition);
      if (dst_ready.exists())
        local_preconditions.emplace_back(dst_ready);
      ApEvent local_precondition;
      if (!local_preconditions.empty())
        local_precondition =
            Runtime::merge_events(&trace_info, local_preconditions);
      Runtime::trigger_event(
          local_pre, local_precondition, trace_info, map_applied_events);
      // Easy out if we're not going to move anything
      if (copy_expr->is_empty())
        return local_pre;
      CopyAcrossUnstructured* across = copy_expr->create_across_unstructured(
          reservations, compute_preimages, shadow_indirections);
      across->add_reference();
      // Initialize the sources
      across->initialize_source_fields(src_req, src_targets, trace_info);
      // Initialize the destination indirections
      const InstanceRef idx_target = idx_targets[0];
      // Only exclusive if we're the only point sctatting to our instance
      // and we're not racing with any other operations
      const bool exclusive_redop =
          (dst_records.size() == 1) &&
          (IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req));
      across->initialize_destination_indirections(
          dst_records, dst_req, idx_req, idx_target, scatter_is_range,
          possible_dst_out_of_range, possible_dst_aliasing, exclusive_redop);
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.emplace_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      if (src_ready.exists())
        copy_preconditions.emplace_back(src_ready);
      ApEvent dst_indirect_ready = idx_ready;
      if (init_precondition.exists())
      {
        if (dst_indirect_ready.exists())
          dst_indirect_ready = Runtime::merge_events(
              &trace_info, dst_indirect_ready, init_precondition);
        else
          dst_indirect_ready = init_precondition;
      }
      ApEvent copy_precondition;
      if (!copy_preconditions.empty())
        copy_precondition =
            Runtime::merge_events(&trace_info, copy_preconditions);
      // Launch the copy
      ApEvent copy_post = across->execute(
          this, pred_guard, copy_precondition, ApEvent::NO_AP_EVENT,
          dst_indirect_ready, trace_info);
      if (trace_info.recording)
      {
        // Record this with the trace
        trace_info.record_issue_across(
            copy_post, local_precondition, copy_precondition,
            ApEvent::NO_AP_EVENT, dst_indirect_ready, across);
        // If we're tracing record the insts for this copy
        local::map<UniqueInst, FieldMask> src_insts, idx_insts, dst_insts;
        InnerContext* context = find_physical_context(src_index);
        std::vector<IndividualView*> source_views;
        context->convert_individual_views(src_targets, source_views);
        // Get the src_insts
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
        {
          const InstanceRef& ref = src_targets[idx];
          const UniqueInst unique_inst(source_views[idx]);
          src_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the idx_insts
        {
          std::vector<IndividualView*> indirect_views;
          InnerContext* idx_context = find_physical_context(idx_index);
          idx_context->convert_individual_views(idx_targets, indirect_views);
          const UniqueInst unique_inst(indirect_views.back());
          idx_insts[unique_inst] = idx_target.get_valid_fields();
        }
        // Get the dst_insts
        std::vector<IndividualView*> target_views;
        InnerContext* dst_context = find_physical_context(dst_index);
        dst_context->convert_individual_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < dst_targets.size(); idx++)
        {
          const InstanceRef& ref = dst_targets[idx];
          const UniqueInst unique_inst(target_views[idx]);
          dst_insts[unique_inst] = ref.get_valid_fields();
        }
        trace_info.record_across_insts(
            copy_post, src_index, idx_index, LEGION_READ_PRIV, LEGION_READ_PRIV,
            copy_expr, src_insts, idx_insts, false /*indirect*/,
            true /*indirect*/, map_applied_events);
        IndexSpaceNode* dst_node =
            runtime->get_node(dst_req.region.get_index_space());
        trace_info.record_indirect_insts(
            copy_post, collective_post, dst_node, dst_insts, map_applied_events,
            LEGION_WRITE_PRIV);
      }
      if (across->remove_reference())
        delete across;
      return copy_post;
    }

    //--------------------------------------------------------------------------
    ApEvent CopyOp::indirect_across(
        const RegionRequirement& src_req, const RegionRequirement& src_idx_req,
        const RegionRequirement& dst_req, const RegionRequirement& dst_idx_req,
        const InstanceSet& src_targets, const InstanceSet& dst_targets,
        std::vector<IndirectRecord>& src_records,
        const InstanceSet& src_idx_targets,
        std::vector<IndirectRecord>& dst_records,
        const InstanceSet& dst_idx_targets, unsigned src_index,
        unsigned dst_index, unsigned src_idx_index, unsigned dst_idx_index,
        const bool both_are_range, const ApEvent init_precondition,
        const ApEvent src_ready, const ApEvent dst_ready,
        const ApEvent src_idx_ready, const ApEvent dst_idx_ready,
        const PredEvent pred_guard, const ApEvent collective_pre,
        const ApEvent collective_post, const ApUserEvent local_pre,
        const std::map<Reservation, bool>& reservations,
        const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events,
        const bool possible_src_out_of_range,
        const bool possible_dst_out_of_range, const bool possible_dst_aliasing,
        const bool compute_preimages, const bool shadow_indirections)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(src_idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(dst_idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(
          src_req.instance_fields.size() == dst_req.instance_fields.size());
      legion_assert(src_idx_req.privilege_fields.size() == 1);
      legion_assert(dst_idx_req.privilege_fields.size() == 1);
      legion_assert(src_idx_targets.size() == 1);
      legion_assert(dst_idx_targets.size() == 1);
      // Get the field indexes for src/dst fields
      IndexSpaceNode* src_idx_node =
          runtime->get_node(src_idx_req.region.get_index_space());
      IndexSpaceNode* dst_idx_node =
          runtime->get_node(dst_idx_req.region.get_index_space());
      IndexSpaceExpression* copy_expr =
          (src_idx_node == dst_idx_node) ?
              src_idx_node :
              runtime->intersect_index_spaces(src_idx_node, dst_idx_node);
      // Trigger the precondition event when all our srcs and dsts are ready
      std::vector<ApEvent> local_preconditions;
      if (init_precondition.exists())
        local_preconditions.emplace_back(init_precondition);
      if (src_ready.exists())
        local_preconditions.emplace_back(src_ready);
      if (dst_ready.exists())
        local_preconditions.emplace_back(dst_ready);
      ApEvent local_precondition;
      if (!local_preconditions.empty())
        local_precondition =
            Runtime::merge_events(&trace_info, local_preconditions);
      Runtime::trigger_event(
          local_pre, local_precondition, trace_info, map_applied_events);
      // Quick out if there is nothing we're going to copy
      if (copy_expr->is_empty())
        return local_pre;
      CopyAcrossUnstructured* across = copy_expr->create_across_unstructured(
          reservations, compute_preimages, shadow_indirections);
      across->add_reference();
      // Initialize the source indirection fields
      const InstanceRef& src_idx_target = src_idx_targets[0];
      across->initialize_source_indirections(
          src_records, src_req, src_idx_req, src_idx_target, both_are_range,
          possible_src_out_of_range);
      // Initialize the destination indirections
      const InstanceRef& dst_idx_target = dst_idx_targets[0];
      // Only exclusive if we're the only point sctatting to our instance
      // and we're not racing with any other operations
      const bool exclusive_redop =
          (dst_records.size() == 1) &&
          (IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req));
      across->initialize_destination_indirections(
          dst_records, dst_req, dst_idx_req, dst_idx_target, both_are_range,
          possible_dst_out_of_range, possible_dst_aliasing, exclusive_redop);
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.emplace_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      ApEvent src_indirect_ready = src_idx_ready;
      if (init_precondition.exists())
      {
        if (src_indirect_ready.exists())
          src_indirect_ready = Runtime::merge_events(
              &trace_info, src_indirect_ready, init_precondition);
        else
          src_indirect_ready = init_precondition;
      }
      ApEvent dst_indirect_ready = dst_idx_ready;
      if (init_precondition.exists())
      {
        if (dst_indirect_ready.exists())
          dst_indirect_ready = Runtime::merge_events(
              &trace_info, dst_indirect_ready, init_precondition);
        else
          dst_indirect_ready = init_precondition;
      }
      ApEvent copy_precondition;
      if (!copy_preconditions.empty())
        copy_precondition =
            Runtime::merge_events(&trace_info, copy_preconditions);
      // Launch the copy
      ApEvent copy_post = across->execute(
          this, pred_guard, copy_precondition, src_indirect_ready,
          dst_indirect_ready, trace_info);
      if (trace_info.recording)
      {
        // Record this with the trace
        trace_info.record_issue_across(
            copy_post, local_precondition, copy_precondition,
            src_indirect_ready, dst_indirect_ready, across);
        // If we're tracing record the insts for this copy
        local::map<UniqueInst, FieldMask> src_insts, src_idx_insts, dst_insts,
            dst_idx_insts;
        // Get the src_insts
        std::vector<IndividualView*> source_views;
        InnerContext* src_context = find_physical_context(src_index);
        src_context->convert_individual_views(src_targets, source_views);
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
        {
          const InstanceRef& ref = src_targets[idx];
          const UniqueInst unique_inst(source_views[idx]);
          src_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the src_idx_insts
        {
          InnerContext* src_idx_context = find_physical_context(src_idx_index);
          std::vector<IndividualView*> src_indirect_views;
          src_idx_context->convert_individual_views(
              src_idx_targets, src_indirect_views);
          const UniqueInst unique_inst(src_indirect_views.back());
          src_idx_insts[unique_inst] = src_idx_target.get_valid_fields();
        }
        // Get the dst_insts
        std::vector<IndividualView*> target_views;
        InnerContext* dst_context = find_physical_context(dst_index);
        dst_context->convert_individual_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < dst_targets.size(); idx++)
        {
          const InstanceRef& ref = dst_targets[idx];
          const UniqueInst unique_inst(target_views[idx]);
          dst_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the dst_idx_insts
        {
          InnerContext* dst_idx_context = find_physical_context(dst_idx_index);
          std::vector<IndividualView*> dst_indirect_views;
          dst_idx_context->convert_individual_views(
              dst_idx_targets, dst_indirect_views);
          const UniqueInst unique_inst(dst_indirect_views.back());
          dst_idx_insts[unique_inst] = dst_idx_target.get_valid_fields();
        }
        IndexSpaceNode* src_node =
            runtime->get_node(src_req.region.get_index_space());
        trace_info.record_indirect_insts(
            copy_post, collective_post, src_node, src_insts, map_applied_events,
            LEGION_READ_PRIV);
        IndexSpaceNode* dst_node =
            runtime->get_node(dst_req.region.get_index_space());
        trace_info.record_indirect_insts(
            copy_post, collective_post, dst_node, dst_insts, map_applied_events,
            LEGION_WRITE_PRIV);
        trace_info.record_across_insts(
            copy_post, src_idx_index, dst_idx_index, LEGION_READ_PRIV,
            LEGION_READ_PRIV, copy_expr, src_idx_insts, dst_idx_insts,
            true /*indirect*/, true /*indirect*/, map_applied_events);
      }
      if (across->remove_reference())
        delete across;
      return copy_post;
    }

    /////////////////////////////////////////////////////////////
    // Index Copy Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexCopyOp::IndexCopyOp(void) : PointwiseAnalyzable<CopyOp>()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    IndexCopyOp::~IndexCopyOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexCopyOp::initialize(
        InnerContext* ctx, const IndexCopyLauncher& launcher,
        IndexSpace launch_sp, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_predication(ctx, launcher.predicate, provenance);
      initialize_copy_from_launcher(launcher);
      legion_assert(launch_sp.exists());
      launch_space = runtime->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        index_domain = launch_space->get_tight_domain();
      else
        index_domain = launcher.launch_domain;
      sharding_space = launcher.sharding_space;
      collective_exchanges.resize(std::max(
          src_indirect_requirements.size(), dst_indirect_requirements.size()));
      collective_src_indirect_points = launcher.collective_src_indirect_points;
      collective_dst_indirect_points = launcher.collective_dst_indirect_points;
      const unsigned copy_kind = (src_indirect_requirements.empty() ? 0 : 1) +
                                 (dst_indirect_requirements.empty() ? 0 : 2);
      LegionSpy::log_copy_operation(
          parent_ctx->get_unique_id(), unique_op_id, copy_kind,
          collective_src_indirect_points, collective_dst_indirect_points);
      log_launch_space(launch_space->handle);
      if (runtime->safe_model)
        perform_type_checking();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PointwiseAnalyzable<CopyOp>::activate();
      index_domain = Domain::NO_DOMAIN;
      sharding_space = IndexSpace::NO_SPACE;
      launch_space = nullptr;
      points_completed.store(0);
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // We can deactivate all of our point operations
      for (PointCopyOp* point : points) point->deactivate();
      points.clear();
      collective_exchanges.clear();
      commit_preconditions.clear();
      interfering_requirements.clear();
      pending_pointwise_dependences.clear();
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
      PointwiseAnalyzable<CopyOp>::deactivate(false /*free*/);
      // Return this operation to the runtime
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Initialize the privilege and mapping paths for all of the
      // region requirements that we have
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        RegionRequirement& req = src_requirements[idx];
        // Promote any singular region requirements to projection
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0;
        }
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        RegionRequirement& req = dst_requirements[idx];
        // Promote any singular region requirements to projection
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0;
        }
      }
      if (!src_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          RegionRequirement& req = src_indirect_requirements[idx];
          // Promote any singular region requirements to projection
          if (req.handle_type == LEGION_SINGULAR_PROJECTION)
          {
            req.handle_type = LEGION_REGION_PROJECTION;
            req.projection = 0;
          }
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          RegionRequirement& req = dst_indirect_requirements[idx];
          // Promote any singular region requirements to projection
          if (req.handle_type == LEGION_SINGULAR_PROJECTION)
          {
            req.handle_type = LEGION_REGION_PROJECTION;
            req.projection = 0;
          }
        }
      }
      log_index_copy_requirements();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::log_index_copy_requirements(void)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        const RegionRequirement& req = src_requirements[idx];
        const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                         (req.handle_type == LEGION_REGION_PROJECTION);
        const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                          (req.handle_type == LEGION_PARTITION_PROJECTION);

        LegionSpy::log_logical_requirement(
            unique_op_id, idx, reg,
            reg ? req.region.index_space.get_id() :
                  req.partition.index_partition.get_id(),
            reg ? req.region.field_space.get_id() :
                  req.partition.field_space.get_id(),
            reg ? req.region.get_tree_id() : req.partition.get_tree_id(),
            req.privilege, req.prop, req.redop,
            req.parent.index_space.get_id());
        LegionSpy::log_requirement_fields(
            unique_op_id, idx, req.instance_fields);
        if (proj)
          LegionSpy::log_requirement_projection(
              unique_op_id, idx, req.projection);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const RegionRequirement& req = dst_requirements[idx];
        const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                         (req.handle_type == LEGION_REGION_PROJECTION);
        const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                          (req.handle_type == LEGION_PARTITION_PROJECTION);

        LegionSpy::log_logical_requirement(
            unique_op_id, src_requirements.size() + idx, reg,
            reg ? req.region.index_space.get_id() :
                  req.partition.index_partition.get_id(),
            reg ? req.region.field_space.get_id() :
                  req.partition.field_space.get_id(),
            reg ? req.region.get_tree_id() : req.partition.get_tree_id(),
            req.privilege, req.prop, req.redop,
            req.parent.index_space.get_id());
        LegionSpy::log_requirement_fields(
            unique_op_id, src_requirements.size() + idx, req.instance_fields);
        if (proj)
          LegionSpy::log_requirement_projection(
              unique_op_id, src_requirements.size() + idx, req.projection);
      }
      if (!src_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() + dst_requirements.size();
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          const RegionRequirement& req = src_indirect_requirements[idx];
          legion_assert(req.privilege_fields.size() == 1);
          const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                           (req.handle_type == LEGION_REGION_PROJECTION);
          const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                            (req.handle_type == LEGION_PARTITION_PROJECTION);

          LegionSpy::log_logical_requirement(
              unique_op_id, offset + idx, reg,
              reg ? req.region.index_space.get_id() :
                    req.partition.index_partition.get_id(),
              reg ? req.region.field_space.get_id() :
                    req.partition.field_space.get_id(),
              reg ? req.region.get_tree_id() : req.partition.get_tree_id(),
              req.privilege, req.prop, req.redop,
              req.parent.index_space.get_id());
          LegionSpy::log_requirement_fields(
              unique_op_id, offset + idx, req.privilege_fields);
          if (proj)
            LegionSpy::log_requirement_projection(
                unique_op_id, offset + idx, req.projection);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() +
                              dst_requirements.size() +
                              src_indirect_requirements.size();
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          const RegionRequirement& req = dst_indirect_requirements[idx];
          legion_assert(req.privilege_fields.size() == 1);
          const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                           (req.handle_type == LEGION_REGION_PROJECTION);
          const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                            (req.handle_type == LEGION_PARTITION_PROJECTION);

          LegionSpy::log_logical_requirement(
              unique_op_id, offset + idx, reg,
              reg ? req.region.index_space.get_id() :
                    req.partition.index_partition.get_id(),
              reg ? req.region.field_space.get_id() :
                    req.partition.field_space.get_id(),
              reg ? req.region.get_tree_id() : req.partition.get_tree_id(),
              req.privilege, req.prop, req.redop,
              req.parent.index_space.get_id());
          LegionSpy::log_requirement_fields(
              unique_op_id, offset + idx, req.privilege_fields);
          if (proj)
            LegionSpy::log_requirement_projection(
                unique_op_id, offset + idx, req.projection);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis(true /*permit projection*/);

      // Change reduce to readwrite so that version info is correct
      std::vector<unsigned> changed_idxs;
      req_vector_reduce_to_readwrite(dst_requirements, changed_idxs);
      analyze_region_requirements(launch_space);
      req_vector_reduce_restore(dst_requirements, changed_idxs);
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we're in a trace in which case we need to compute the
      // parent region indexes for all the region requirements again
      if (trace != nullptr)
      {
        unsigned offset = 0;
        for (unsigned idx = 0; idx < src_requirements.size(); idx++)
          src_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, src_requirements[idx], offset++, false /*skip*/,
              true /*force*/);
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
          dst_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, dst_requirements[idx], offset++, false /*skip*/,
              true /*force*/);
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          gather_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, src_indirect_requirements[idx], offset++, false /*skip*/,
              true /*force*/);
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          scatter_parent_indexes[idx] = parent_ctx->find_parent_region_index(
              this, dst_indirect_requirements[idx], offset++, false /*skip*/,
              true /*force*/);
      }
      // Enumerate the points
      enumerate_points();
      // Check for interfering point requirements in safe mode
      if (runtime->safe_model)
        start_check_point_requirements();
      // Launch the points
      std::vector<RtEvent> mapped_preconditions(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        mapped_preconditions[idx] = points[idx]->get_mapped_event();
        points[idx]->launch();
      }
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(mapped_preconditions));
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // This should never be called as this operation doesn't
      // go through the rest of the queue normally
      std::abort();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::handle_point_complete(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (effect.exists())
        record_completion_effect(effect);
      const unsigned received = ++points_completed;
      legion_assert(received <= points.size());
      if (received == points.size())
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
        legion_assert(!commit_request);
        commit_request = true;
        commit_now = (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(
            true /*deactivate*/, Runtime::merge_events(commit_preconditions));
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_replaying());
      LegionSpy::log_replay_operation(unique_op_id);
      // Enumerate the points
      enumerate_points();
      // Then call replay analysis on all of them
      std::vector<RtEvent> mapped_preconditions(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        mapped_preconditions[idx] = points[idx]->get_mapped_event();
        points[idx]->trigger_replay();
      }
      complete_mapping(Runtime::merge_events(mapped_preconditions));
    }

    //--------------------------------------------------------------------------
    size_t IndexCopyOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return get_shard_points()->get_volume();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::enumerate_points(void)
    //--------------------------------------------------------------------------
    {
      // Need to get the launch domain in case it is different than
      // the original index domain due to control replication
      IndexSpaceNode* local_points = get_shard_points();
      Domain launch_domain = local_points->get_tight_domain();
      // Now enumerate the points
      size_t num_points = local_points->get_volume();
      legion_assert(num_points > 0);
      std::vector<PointCopyOp*> temp_points;
      temp_points.reserve(num_points);
      for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
      {
        PointCopyOp* point = runtime->get_operation<PointCopyOp>();
        point->initialize(this, itr.p);
        temp_points.emplace_back(point);
      }
      // Perform the projections
      std::vector<ProjectionPoint*> projection_points(
          temp_points.begin(), temp_points.end());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        if (src_requirements[idx].handle_type == LEGION_SINGULAR_PROJECTION)
          continue;
        ProjectionFunction* function =
            runtime->find_projection_function(src_requirements[idx].projection);
        std::map<unsigned, std::vector<PointwiseDependence>>::const_iterator
            finder = pointwise_dependences.find(idx);
        function->project_points(
            this, idx, src_requirements[idx], index_domain, projection_points,
            (finder == pointwise_dependences.end()) ? nullptr : &finder->second,
            parent_ctx->get_total_shards(), is_replaying());
      }
      unsigned offset = src_requirements.size();
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        if (dst_requirements[idx].handle_type == LEGION_SINGULAR_PROJECTION)
          continue;
        ProjectionFunction* function =
            runtime->find_projection_function(dst_requirements[idx].projection);
        std::map<unsigned, std::vector<PointwiseDependence>>::const_iterator
            finder = pointwise_dependences.find(offset + idx);
        function->project_points(
            this, offset + idx, dst_requirements[idx], index_domain,
            projection_points,
            (finder == pointwise_dependences.end()) ? nullptr : &finder->second,
            parent_ctx->get_total_shards(), is_replaying());
      }
      offset += dst_requirements.size();
      if (!src_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          if (src_indirect_requirements[idx].handle_type ==
              LEGION_SINGULAR_PROJECTION)
            continue;
          ProjectionFunction* function = runtime->find_projection_function(
              src_indirect_requirements[idx].projection);
          std::map<unsigned, std::vector<PointwiseDependence>>::const_iterator
              finder = pointwise_dependences.find(offset + idx);
          function->project_points(
              this, offset + idx, src_indirect_requirements[idx], index_domain,
              projection_points,
              (finder == pointwise_dependences.end()) ? nullptr :
                                                        &finder->second,
              parent_ctx->get_total_shards(), is_replaying());
        }
        offset += src_indirect_requirements.size();
      }
      if (!dst_indirect_requirements.empty())
      {
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          if (dst_indirect_requirements[idx].handle_type ==
              LEGION_SINGULAR_PROJECTION)
            continue;
          ProjectionFunction* function = runtime->find_projection_function(
              dst_indirect_requirements[idx].projection);
          std::map<unsigned, std::vector<PointwiseDependence>>::const_iterator
              finder = pointwise_dependences.find(offset + idx);
          function->project_points(
              this, offset + idx, dst_indirect_requirements[idx], index_domain,
              projection_points,
              (finder == pointwise_dependences.end()) ? nullptr :
                                                        &finder->second,
              parent_ctx->get_total_shards(), is_replaying());
        }
      }
      for (PointCopyOp* point : temp_points) point->log_copy_requirements();
      // Need the lock to avoid racing with the pointwise dependence analysis
      AutoLock o_lock(op_lock);
      legion_assert(points.empty());
      points.swap(temp_points);
      for (const std::pair<const DomainPoint, RtUserEvent>& pit :
           pending_pointwise_dependences)
      {
        PointCopyOp* point = nullptr;
        for (PointCopyOp* it : points)
        {
          if (pit.first != it->index_point)
            continue;
          point = it;
          break;
        }
        legion_assert(point != nullptr);
        Runtime::trigger_event(pit.second, point->get_mapped_event());
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Trigger any pending pointwise dependences since they will not
      // be run, safe to do without the lock because we are protected
      // by the predication_state having been set before this
      if (!pending_pointwise_dependences.empty())
      {
        for (const std::pair<const DomainPoint, RtUserEvent>& it :
             pending_pointwise_dependences)
          Runtime::trigger_event(it.second);
        pending_pointwise_dependences.clear();
      }
      CopyOp::predicate_false();
    }

    //--------------------------------------------------------------------------
    bool IndexCopyOp::is_pointwise_analyzable(void) const
    //--------------------------------------------------------------------------
    {
      // We're not pointwise analyzable if we're doing collective gather/scatter
      if (!src_indirect_requirements.empty() && collective_src_indirect_points)
        return false;
      if (!dst_indirect_requirements.empty() && collective_dst_indirect_points)
        return false;
      return CopyOp::is_pointwise_analyzable();
    }

    //--------------------------------------------------------------------------
    RtEvent IndexCopyOp::find_pointwise_dependence(
        const DomainPoint& point, GenerationID needed_gen, bool intra_space,
        RtUserEvent to_trigger, std::optional<unsigned> output_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(!output_index);
      AutoLock o_lock(op_lock);
      legion_assert(needed_gen <= gen);
      if ((needed_gen < gen) || mapped ||
          (predication_state == PREDICATED_FALSE_STATE))
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      if (points.empty())
      {
        std::map<DomainPoint, RtUserEvent>::const_iterator finder =
            pending_pointwise_dependences.find(point);
        if (finder != pending_pointwise_dependences.end())
        {
          if (to_trigger.exists())
          {
            Runtime::trigger_event(to_trigger, finder->second);
            return to_trigger;
          }
          else
            return finder->second;
        }
        if (!to_trigger.exists())
          to_trigger = Runtime::create_rt_user_event();
        pending_pointwise_dependences.emplace(
            std::make_pair(point, to_trigger));
        return to_trigger;
      }
      for (PointCopyOp* it : points)
      {
        if (point != it->index_point)
          continue;
        if (to_trigger.exists())
        {
          Runtime::trigger_event(to_trigger, it->get_mapped_event());
          return to_trigger;
        }
        else
          return it->get_mapped_event();
      }
      // Should never get here, if we do that means we couldn't find the point
      std::abort();
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::handle_point_commit(RtEvent point_committed)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
        points_committed++;
        if (point_committed.exists())
          commit_preconditions.insert(point_committed);
        commit_now = commit_request && (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(
            true /*deactivate*/, Runtime::merge_events(commit_preconditions));
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::report_interfering_requirements(
        unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx1 < idx2);
      // The logical dependence analysis can report this because there are
      // interfering fields and regions, check to make sure there are alos
      // interfering privileges and index spaces
      const RegionRequirement& req1 = get_requirement(idx1);
      const RegionRequirement& req2 = get_requirement(idx2);
      if (IS_READ_ONLY(req1) && IS_READ_ONLY(req2))
        return;
      if (IS_REDUCE(req1) && IS_REDUCE(req2) && (req1.redop == req2.redop))
        return;
      // Save this if the runtime checks are on for it
      if (runtime->safe_model)
        interfering_requirements.emplace(std::make_pair(idx1, idx2));
    }

    //--------------------------------------------------------------------------
    RtEvent IndexCopyOp::exchange_indirect_records(
        const unsigned index, const ApEvent local_pre, const ApEvent local_post,
        ApEvent& collective_pre, ApEvent& collective_post,
        const TraceInfo& trace_info, const InstanceSet& insts,
        const RegionRequirement& req, std::vector<IndirectRecord>& records,
        const bool sources)
    //--------------------------------------------------------------------------
    {
      if (sources && !collective_src_indirect_points)
        return CopyOp::exchange_indirect_records(
            index, local_pre, local_post, collective_pre, collective_post,
            trace_info, insts, req, records, sources);
      if (!sources && !collective_dst_indirect_points)
        return CopyOp::exchange_indirect_records(
            index, local_pre, local_post, collective_pre, collective_post,
            trace_info, insts, req, records, sources);
      legion_assert(local_pre.exists());
      legion_assert(local_post.exists());
      // Take the lock and record our sets and instances
      AutoLock o_lock(op_lock);
      legion_assert(index < collective_exchanges.size());
      IndirectionExchange& exchange = collective_exchanges[index];
      if (sources)
      {
        if (!exchange.collective_pre.exists())
        {
          exchange.collective_pre = Runtime::create_ap_user_event(&trace_info);
          exchange.collective_post = Runtime::create_ap_user_event(&trace_info);
        }
        collective_pre = exchange.collective_pre;
        collective_post = exchange.collective_post;
        if (!exchange.src_ready.exists())
          exchange.src_ready = Runtime::create_rt_user_event();
        if (exchange.local_preconditions.size() < points.size())
        {
          exchange.local_preconditions.insert(local_pre);
          if (exchange.local_preconditions.size() == points.size())
            Runtime::trigger_event(
                exchange.collective_pre,
                Runtime::merge_events(
                    &trace_info, exchange.local_preconditions),
                trace_info, map_applied_conditions);
        }
        if (exchange.local_postconditions.size() < points.size())
        {
          exchange.local_postconditions.insert(local_post);
          if (exchange.local_postconditions.size() == points.size())
            Runtime::trigger_event(
                exchange.collective_post,
                Runtime::merge_events(
                    &trace_info, exchange.local_postconditions),
                trace_info, map_applied_conditions);
        }
        legion_assert(src_indirect_records[index].size() < points.size());
        src_indirect_records[index].emplace_back(
            IndirectRecord(req, insts, launch_space->get_volume()));
        exchange.src_records.emplace_back(&records);
        if (src_indirect_records[index].size() == points.size())
          return finalize_exchange(index, true /*sources*/);
        return exchange.src_ready;
      }
      else
      {
        if (!exchange.collective_pre.exists())
        {
          exchange.collective_pre = Runtime::create_ap_user_event(&trace_info);
          exchange.collective_post = Runtime::create_ap_user_event(&trace_info);
        }
        collective_pre = exchange.collective_pre;
        collective_post = exchange.collective_post;
        if (!exchange.dst_ready.exists())
          exchange.dst_ready = Runtime::create_rt_user_event();
        if (exchange.local_preconditions.size() < points.size())
        {
          exchange.local_preconditions.insert(local_pre);
          if (exchange.local_preconditions.size() == points.size())
            Runtime::trigger_event(
                exchange.collective_pre,
                Runtime::merge_events(
                    &trace_info, exchange.local_preconditions),
                trace_info, map_applied_conditions);
        }
        if (exchange.local_postconditions.size() < points.size())
        {
          exchange.local_postconditions.insert(local_post);
          if (exchange.local_postconditions.size() == points.size())
            Runtime::trigger_event(
                exchange.collective_post,
                Runtime::merge_events(
                    &trace_info, exchange.local_postconditions),
                trace_info, map_applied_conditions);
        }
        legion_assert(dst_indirect_records[index].size() < points.size());
        dst_indirect_records[index].emplace_back(
            IndirectRecord(req, insts, launch_space->get_volume()));
        exchange.dst_records.emplace_back(&records);
        if (dst_indirect_records[index].size() == points.size())
          return finalize_exchange(index, false /*sources*/);
        return exchange.dst_ready;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent IndexCopyOp::finalize_exchange(
        const unsigned index, const bool source)
    //--------------------------------------------------------------------------
    {
      IndirectionExchange& exchange = collective_exchanges[index];
      if (source)
      {
        const std::vector<IndirectRecord>& records =
            src_indirect_records[index];
        for (unsigned idx = 0; idx < exchange.src_records.size(); idx++)
          *exchange.src_records[idx] = records;
        Runtime::trigger_event(exchange.src_ready);
        return exchange.src_ready;
      }
      else
      {
        const std::vector<IndirectRecord>& records =
            dst_indirect_records[index];
        for (unsigned idx = 0; idx < exchange.dst_records.size(); idx++)
          *exchange.dst_records[idx] = records;
        Runtime::trigger_event(exchange.dst_ready);
        return exchange.dst_ready;
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>&
            point_domains)
    //--------------------------------------------------------------------------
    {
      legion_assert(!interfering_requirements.empty());
      // Iterate our local points and check their first region requirements
      // against all the points in the second region requirements
      for (const std::pair<unsigned, unsigned>& rit : interfering_requirements)
      {
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>::
            const_iterator finder = point_domains.find(rit.first);
        legion_assert(finder != point_domains.end());
        for (PointCopyOp* pit : points)
        {
          const RegionRequirement& req = pit->get_requirement(rit.second);
          IndexSpaceNode* node =
              runtime->get_node(req.region.get_index_space());
          DomainPoint interfering;
          if (node->has_interfering_point(
                  finder->second, interfering,
                  (rit.first == rit.second) ? pit->index_point : DomainPoint()))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Index " << *this
                  << " has interfering region requirments between "
                  << get_requirement_name(rit.first) << " requirement "
                  << get_requirement_offset(rit.first) << " of point "
                  << interfering << " and " << get_requirement_name(rit.second)
                  << " requirement " << get_requirement_offset(rit.second)
                  << " of point " << pit->index_point
                  << ". Interfering region requirements are not permitted for "
                     "index "
                  << "copy operations.";
            error.raise();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexCopyOp::start_check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      // If this is part of an indirection copy then we can't really know
      // what is actually being written so we aren't going to check it
      // This is the same as what Legion Spy does for now, TBD if there
      // is something better that we can do
      if (!src_indirect_requirements.empty() ||
          !dst_indirect_requirements.empty())
        return;
      // Handle any region requirements which can interfere with itself
      // which wouldn't have been picked up the logical analysis
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const RegionRequirement& req = dst_requirements[idx];
        if (!IS_WRITE(req) || IS_COLLECTIVE(req))
          continue;
        // If the projection functions are invertible then we don't have to
        // worry about interference because the runtime knows how to hook
        // up those kinds of dependences
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
        {
          if (req.projection == 0)
          {
            if (req.handle_type == LEGION_PARTITION_PROJECTION)
            {
              IndexPartNode* partition =
                  runtime->get_node(req.partition.get_index_partition());
              if (partition->is_disjoint())
                continue;
            }
            else  // Identity functor is invertible
              continue;
          }
          else
          {
            ProjectionFunction* func =
                runtime->find_projection_function(req.projection);
            if (func->is_invertible)
              continue;
          }
        }
        const unsigned index = src_requirements.size() + idx;
        interfering_requirements.insert(
            std::pair<unsigned, unsigned>(index, index));
      }
      // Nothing to do if there are no interfering requirements
      if (interfering_requirements.empty())
        return;
      // Get all of our local point domains
      std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>
          point_domains;
      for (const std::pair<unsigned, unsigned>& rit : interfering_requirements)
      {
        std::vector<std::pair<DomainPoint, Domain>>& domains =
            point_domains[rit.first];
        // Already found it for this region requirements
        if (!domains.empty())
          continue;
        domains.reserve(points.size());
        for (PointCopyOp* pit : points)
        {
          const RegionRequirement& req = pit->get_requirement(rit.first);
          if (!req.region.exists())
            continue;
          IndexSpaceNode* node =
              runtime->get_node(req.region.get_index_space());
          Domain domain = node->get_tight_domain();
          domains.emplace_back(std::make_pair(pit->index_point, domain));
        }
      }
      finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexCopyOp::find_intra_space_dependence(const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      return find_pointwise_dependence(point, get_generation(), true /*intra*/);
    }

    /////////////////////////////////////////////////////////////
    // Point Copy Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointCopyOp::PointCopyOp(void) : CopyOp()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    PointCopyOp::~PointCopyOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PointCopyOp::initialize(IndexCopyOp* own, const DomainPoint& p)
    //--------------------------------------------------------------------------
    {
      // Initialize the operation
      initialize_operation(own->get_context(), own->get_provenance());
      index_point = p;
      index_domain = own->index_domain;
      sharding_space = own->sharding_space;
      owner = own;
      context_index = own->get_context_index();
      exception_handler = own->get_exception_handler();
      execution_fence_event = own->get_execution_fence_event();
      // From Memoizable
      trace_local_id = owner->get_trace_local_id().context_index;
      tpl = owner->get_template();
      if (tpl != nullptr)
        memo_state = owner->get_memoizable_state();
      // From Copy
      src_requirements = owner->src_requirements;
      dst_requirements = owner->dst_requirements;
      src_indirect_requirements = owner->src_indirect_requirements;
      dst_indirect_requirements = owner->dst_indirect_requirements;
      grants = owner->grants;
      wait_barriers = owner->wait_barriers;
      arrive_barriers = owner->arrive_barriers;
      parent_task = owner->parent_task;
      map_id = owner->map_id;
      tag = owner->tag;
      mapper_data_size = owner->mapper_data_size;
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, owner->mapper_data, mapper_data_size);
      }
      // From CopyOp
      possible_src_indirect_out_of_range =
          owner->possible_src_indirect_out_of_range;
      possible_dst_indirect_out_of_range =
          owner->possible_dst_indirect_out_of_range;
      possible_dst_indirect_aliasing = owner->possible_dst_indirect_aliasing;
      gather_is_range = owner->gather_is_range;
      scatter_is_range = owner->scatter_is_range;
      src_indirect_records.resize(src_indirect_requirements.size());
      dst_indirect_records.resize(dst_indirect_requirements.size());
      atomic_locks.resize(
          src_requirements.size() + dst_requirements.size() +
          src_indirect_requirements.size() + dst_indirect_requirements.size());

      LegionSpy::log_index_point(owner->get_unique_op_id(), unique_op_id, p);
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      CopyOp::activate();
      owner = nullptr;
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      pointwise_mapping_dependences.clear();
      CopyOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      memo_state = MEMO_REPLAY;
      tpl->register_operation(this);
      CopyOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::launch(void)
    //--------------------------------------------------------------------------
    {
      // Perform the version analysis
      std::set<RtEvent> preconditions;
      if (!pointwise_mapping_dependences.empty())
        preconditions.insert(
            pointwise_mapping_dependences.begin(),
            pointwise_mapping_dependences.end());
      src_versions.resize(src_requirements.size());
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
        perform_versioning_analysis(
            idx, src_requirements[idx], src_versions[idx], preconditions);
      dst_versions.resize(dst_requirements.size());
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        perform_versioning_analysis(
            src_requirements.size() + idx, dst_requirements[idx],
            dst_versions[idx], preconditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
      }
      if (!src_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() + dst_requirements.size();
        gather_versions.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          perform_versioning_analysis(
              offset + idx, src_indirect_requirements[idx],
              gather_versions[idx], preconditions);
      }
      if (!dst_indirect_requirements.empty())
      {
        const size_t offset = src_requirements.size() +
                              dst_requirements.size() +
                              src_indirect_requirements.size();
        scatter_versions.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          perform_versioning_analysis(
              offset + idx, dst_indirect_requirements[idx],
              scatter_versions[idx], preconditions);
        if (!src_indirect_requirements.empty())
        {
          // Full indirections need to have the same index space
          for (unsigned idx = 0; (idx < src_indirect_requirements.size()) &&
                                 (idx < dst_indirect_requirements.size());
               idx++)
          {
            const IndexSpace src_space =
                src_indirect_requirements[idx].region.get_index_space();
            const IndexSpace dst_space =
                dst_indirect_requirements[idx].region.get_index_space();
            if (src_space != dst_space)
            {
              Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              error << "Mismatch between source indirect and destination "
                       "indirect "
                    << "index spaces for requirement " << idx << " for "
                    << *this << ".";
              error.raise();
            }
          }
        }
      }
      // Then put ourselves in the queue of operations ready to map
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      owner->handle_point_complete(effects);
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_copy_profiling();
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false /*deactivate*/, profiling_reported);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit(profiling_reported);
    }

    //--------------------------------------------------------------------------
    size_t PointCopyOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_collective_points();
    }

    //--------------------------------------------------------------------------
    bool PointCopyOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      return owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    RtEvent PointCopyOp::exchange_indirect_records(
        const unsigned index, const ApEvent local_pre, const ApEvent local_post,
        ApEvent& collective_pre, ApEvent& collective_post,
        const TraceInfo& trace_info, const InstanceSet& insts,
        const RegionRequirement& req, std::vector<IndirectRecord>& records,
        const bool sources)
    //--------------------------------------------------------------------------
    {
      // Exchange via the owner
      return owner->exchange_indirect_records(
          index, local_pre, local_post, collective_pre, collective_post,
          trace_info, insts, req, records, sources);
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointCopyOp::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::set_projection_result(unsigned idx, LogicalRegion result)
    //--------------------------------------------------------------------------
    {
      if (idx < src_requirements.size())
      {
        legion_assert(
            src_requirements[idx].handle_type != LEGION_SINGULAR_PROJECTION);
        src_requirements[idx].region = result;
        src_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
      else if (idx < (src_requirements.size() + dst_requirements.size()))
      {
        idx -= src_requirements.size();
        legion_assert(
            dst_requirements[idx].handle_type != LEGION_SINGULAR_PROJECTION);
        dst_requirements[idx].region = result;
        dst_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
      else if (
          idx < (src_requirements.size() + dst_requirements.size() +
                 src_indirect_requirements.size()))
      {
        idx -= (src_requirements.size() + dst_requirements.size());
        legion_assert(
            src_indirect_requirements[idx].handle_type !=
            LEGION_SINGULAR_PROJECTION);
        src_indirect_requirements[idx].region = result;
        src_indirect_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
      else
      {
        idx -=
            (src_requirements.size() + dst_requirements.size() +
             src_indirect_requirements.size());
        legion_assert(idx < dst_indirect_requirements.size());
        legion_assert(
            dst_indirect_requirements[idx].handle_type !=
            LEGION_SINGULAR_PROJECTION);
        dst_indirect_requirements[idx].region = result;
        dst_indirect_requirements[idx].handle_type = LEGION_SINGULAR_PROJECTION;
      }
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::record_intra_space_dependences(
        unsigned index, const std::vector<DomainPoint>& dependences)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_requirements.size() <= index);
      index -= src_requirements.size();
      legion_assert(index < dst_requirements.size());
      for (unsigned idx = 0; idx < dependences.size(); idx++)
      {
        if (dependences[idx] == index_point)
        {
          // If we've got a prior dependence then record it
          if (idx > 0)
          {
            const DomainPoint& prev = dependences[idx - 1];
            const RtEvent pre = owner->find_intra_space_dependence(prev);
            if (pre.exists())
              pointwise_mapping_dependences.emplace_back(pre);
            // We know we only need a dependence on the previous point but
            // Legion Spy is stupid, so log everything we have a
            // precondition on even if it is transitively implied
            for (unsigned idx2 = 0; idx2 < idx; idx2++)
              LegionSpy::log_intra_space_dependence(
                  unique_op_id, dependences[idx2]);
          }
          return;
        }
      }
      // We should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointCopyOp::record_pointwise_dependence(
        uint64_t previous_context_index, const DomainPoint& previous_point,
        ShardID shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(previous_context_index < context_index);
      const RtEvent pre = parent_ctx->find_pointwise_dependence(
          previous_context_index, previous_point, shard, false /*intra space*/);
      if (pre.exists())
        pointwise_mapping_dependences.emplace_back(pre);
    }

    //--------------------------------------------------------------------------
    TraceLocalID PointCopyOp::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return TraceLocalID(trace_local_id, index_point);
    }

    /////////////////////////////////////////////////////////////
    // Indirect Record Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndirectRecordExchange::IndirectRecordExchange(
        ReplicateContext* ctx, CollectiveID id)
      : AllGatherCollective(ctx, id)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndirectRecordExchange::~IndirectRecordExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RtEvent IndirectRecordExchange::exchange_records(
        std::vector<std::vector<IndirectRecord>*>& targets,
        std::vector<IndirectRecord>& records)
    //--------------------------------------------------------------------------
    {
      local_targets.swap(targets);
      all_records.swap(records);
      perform_collective_async();
      return perform_collective_wait(false /*block*/);
    }

    //--------------------------------------------------------------------------
    void IndirectRecordExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(all_records.size());
      for (unsigned idx = 0; idx < all_records.size(); idx++)
        all_records[idx].serialize(rez);
    }

    //--------------------------------------------------------------------------
    void IndirectRecordExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      // If we are not a participating stage then we already contributed our
      // data into the output so we clear ourself to avoid double counting
      if (!participating)
      {
        legion_assert(stage == -1);
        all_records.clear();
      }
      const size_t offset = all_records.size();
      size_t num_records;
      derez.deserialize(num_records);
      all_records.resize(offset + num_records);
      for (unsigned idx = 0; idx < num_records; idx++)
        all_records[offset + idx].deserialize(derez);
    }

    //--------------------------------------------------------------------------
    RtEvent IndirectRecordExchange::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < local_targets.size(); idx++)
        *local_targets[idx] = all_records;
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Repl Copy Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplCopyOp::ReplCopyOp(void) : CopyOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplCopyOp::~ReplCopyOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplCopyOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle;
      if (index_domain.get_dim() == 0)
      {
        DomainPoint point(0);
        Domain launch_domain(point, point);
        handle = ctx->find_index_launch_space(launch_domain, get_provenance());
      }
      else
        handle = ctx->find_index_launch_space(index_domain, get_provenance());
      launch_space = runtime->get_node(handle);
      // Initialize our index domain of a single point
      index_domain = Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      CopyOp::activate();
      launch_space = nullptr;
      sharding_functor = std::numeric_limits<ShardingID>::max();
      sharding_function = nullptr;
      sharding_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (sharding_collective != nullptr)
        delete sharding_collective;
      CopyOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Do the mapper call to get the sharding function to use
      if (mapper == nullptr)
        mapper =
            runtime->find_mapper(parent_ctx->get_executing_processor(), map_id);
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      mapper->invoke_copy_select_sharding_functor(this, *input, output);
      if (output.chosen_functor == std::numeric_limits<ShardingID>::max())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper " << mapper->get_mapper_name()
              << " failed to pick a valid sharding functor for copy in task "
              << parent_ctx->get_task_name() << " (UID "
              << parent_ctx->get_unique_id() << ").";
        error.raise();
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function =
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      if (runtime->safe_mapper)
      {
        legion_assert(sharding_collective != nullptr);
        sharding_collective->contribute(this->sharding_functor);
        if (sharding_collective->is_target() &&
            !sharding_collective->validate(this->sharding_functor))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << mapper->get_mapper_name()
                << " chose different sharding functions for copy in task "
                << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id() << ").";
          error.raise();
        }
      }
      // Now we can do the normal prepipeline stage
      CopyOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis(false /*permit projection*/);

      // Perform reudciton dependence analysis as if it was READ_WRITE
      // so that we can get the version numbers correct
      std::vector<unsigned> changed_idxs;
      req_vector_reduce_to_readwrite(dst_requirements, changed_idxs);

      // Make these requirements look like projection requirmeents since we
      // need the logical analysis to look at sharding to determine if any
      // kind of close operations are required
      analyze_region_requirements(
          launch_space, sharding_function, sharding_space);

      req_vector_reduce_restore(dst_requirements, changed_idxs);
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Figure out whether this shard owns this point
      ShardID owner_shard;
      if (sharding_space.exists())
      {
        Domain shard_domain;
        runtime->find_domain(sharding_space, shard_domain);
        owner_shard = sharding_function->find_owner(index_point, shard_domain);
      }
      else
        owner_shard = sharding_function->find_owner(index_point, index_domain);
      // If we're recording then record the owner shard
      if (is_recording())
      {
        legion_assert((tpl != nullptr) && tpl->is_recording());
        tpl->record_owner_shard(trace_local_id, owner_shard);
      }
      LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // We don't own it, so we can pretend like we
        // mapped and executed this copy already
        complete_mapping();
        complete_execution();
      }
      else  // We own it, so do the base call
        CopyOp::trigger_ready();
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(tpl != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      const ShardID owner_shard = tpl->find_owner_shard(trace_local_id);
      LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        complete_mapping();
        complete_execution();
      }
      else  // We own it, so do the base call
        CopyOp::trigger_replay();
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Copy Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexCopyOp::ReplIndexCopyOp(void) : IndexCopyOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplIndexCopyOp::~ReplIndexCopyOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      IndexCopyOp::activate();
      sharding_functor = std::numeric_limits<ShardingID>::max();
      sharding_function = nullptr;
      shard_points = nullptr;
      interfering_check_id = 0;
      interfering_exchange = nullptr;
      sharding_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (sharding_collective != nullptr)
        delete sharding_collective;
      pre_indirection_barriers.clear();
      post_indirection_barriers.clear();
      if (!src_collectives.empty())
      {
        for (IndirectRecordExchange* src : src_collectives) delete src;
        src_collectives.clear();
      }
      if (!dst_collectives.empty())
      {
        for (IndirectRecordExchange* dst : dst_collectives) delete dst;
        dst_collectives.clear();
      }
      unique_intra_space_deps.clear();
      remove_launch_space_reference(shard_points);
      if (interfering_exchange != nullptr)
        delete interfering_exchange;
      IndexCopyOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Do the mapper call to get the sharding function to use
      if (mapper == nullptr)
        mapper =
            runtime->find_mapper(parent_ctx->get_executing_processor(), map_id);
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      mapper->invoke_copy_select_sharding_functor(this, *input, output);
      if (output.chosen_functor == std::numeric_limits<ShardingID>::max())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper " << mapper->get_mapper_name()
              << " failed to pick a valid sharding functor for index copy in "
                 "task "
              << parent_ctx->get_task_name() << " (UID "
              << parent_ctx->get_unique_id() << ").";
        error.raise();
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function =
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      if (runtime->safe_mapper)
      {
        legion_assert(sharding_collective != nullptr);
        sharding_collective->contribute(this->sharding_functor);
        if (sharding_collective->is_target() &&
            !sharding_collective->validate(this->sharding_functor))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << mapper->get_mapper_name()
                << " chose different sharding functions for index copy in task "
                << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id() << ").";
          error.raise();
        }
      }
      // Now we can do the normal prepipeline stage
      IndexCopyOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis(true /*permit projection*/);

      // Perform REDUCE dependence analysis as if it was READ_WRITE
      // so that we can get the version numbers correct
      std::vector<unsigned> changed_idxs;
      req_vector_reduce_to_readwrite(dst_requirements, changed_idxs);

      analyze_region_requirements(
          launch_space, sharding_function, sharding_space);

      req_vector_reduce_restore(dst_requirements, changed_idxs);
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>&
            point_domains)
    //--------------------------------------------------------------------------
    {
      // See if this is the first time through or not
      if (interfering_exchange == nullptr)
      {
        // First time through, make the exchange and kick it off
        legion_assert(interfering_check_id > 0);
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        interfering_exchange = new InterferingPointExchange<ReplIndexCopyOp>(
            repl_ctx, interfering_check_id, this);
        // Record a dependence on this to make sure it is done before we
        // clean up the operation
        commit_preconditions.insert(interfering_exchange->get_done_event());
        interfering_exchange->exchange_domain_points(point_domains);
      }
      else  // Second time through call the base class since we have the
            // results
        IndexCopyOp::finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      legion_assert(
          pre_indirection_barriers.size() == post_indirection_barriers.size());
      // Compute the local index space of points for this shard
      IndexSpace local_space;
      if (sharding_space.exists())
        local_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, sharding_space,
            get_provenance());
      else
        local_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, launch_space->handle,
            get_provenance());
      // If we're recording then record the local_space
      if (is_recording())
      {
        legion_assert((tpl != nullptr) && tpl->is_recording());
        tpl->record_local_space(trace_local_id, local_space);
      }
      // If it's empty we're done, otherwise we go back on the queue
      if (!local_space.exists())
      {
        // Check for interfering point requirements in safe mode
        if (runtime->safe_model)
          start_check_point_requirements();
        // If we have indirections then we still need to participate in those
        std::vector<RtEvent> done_events;
        if (!src_indirect_requirements.empty() &&
            collective_src_indirect_points)
        {
          for (unsigned idx = 0; idx < collective_exchanges.size(); idx++)
          {
            const RtEvent done = finalize_exchange(idx, true /*source*/);
            if (done.exists())
              done_events.emplace_back(done);
          }
        }
        if (!dst_indirect_requirements.empty() &&
            collective_dst_indirect_points)
        {
          for (unsigned idx = 0; idx < collective_exchanges.size(); idx++)
          {
            const RtEvent done = finalize_exchange(idx, false /*source*/);
            if (done.exists())
              done_events.emplace_back(done);
          }
        }
        // Arrive on our indirection barriers if we have them
        if (!pre_indirection_barriers.empty())
        {
          const PhysicalTraceInfo trace_info(this, 0 /*index*/);
          for (unsigned idx = 0; idx < pre_indirection_barriers.size(); idx++)
          {
            runtime->phase_barrier_arrive(pre_indirection_barriers[idx], 1);
            if (trace_info.recording)
            {
              const std::pair<size_t, size_t> key(trace_local_id, idx);
              trace_info.record_collective_barrier(
                  pre_indirection_barriers[idx], ApEvent::NO_AP_EVENT, key);
            }
          }
          for (unsigned idx = 0; idx < post_indirection_barriers.size(); idx++)
          {
            runtime->phase_barrier_arrive(post_indirection_barriers[idx], 1);
            if (trace_info.recording)
            {
              const std::pair<size_t, size_t> key(
                  trace_local_id, pre_indirection_barriers.size() + idx);
              trace_info.record_collective_barrier(
                  post_indirection_barriers[idx], ApEvent::NO_AP_EVENT, key);
            }
          }
        }
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // We have no local points, so we can just trigger
        complete_mapping();
        if (!done_events.empty())
          complete_execution(Runtime::merge_events(done_events));
        else
          complete_execution();
      }
      else  // If we have any valid points do the base call
      {
        shard_points = runtime->get_node(local_space);
        add_launch_space_reference(shard_points);
        IndexCopyOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(tpl != nullptr);
      legion_assert(
          pre_indirection_barriers.size() == post_indirection_barriers.size());
      // No matter what we need to tell the shard template about any
      // collective barriers that it is going to need for its replay
      if (!pre_indirection_barriers.empty())
      {
        ShardedPhysicalTemplate* shard_template =
            legion_safe_cast<ShardedPhysicalTemplate*>(tpl);
        std::pair<size_t, size_t> key(trace_local_id, 0);
        for (unsigned idx = 0; idx < pre_indirection_barriers.size(); idx++)
        {
          shard_template->prepare_collective_barrier_replay(
              key, pre_indirection_barriers[idx]);
          key.second++;
        }
        for (unsigned idx = 0; idx < post_indirection_barriers.size(); idx++)
        {
          shard_template->prepare_collective_barrier_replay(
              key, post_indirection_barriers[idx]);
          key.second++;
        }
      }
      // Elide unused collectives
      for (IndirectRecordExchange* src : src_collectives)
        src->elide_collective();
      for (IndirectRecordExchange* dst : dst_collectives)
        dst->elide_collective();
      const IndexSpace local_space = tpl->find_local_space(trace_local_id);
      // If it's empty we're done, otherwise we do the replay
      if (!local_space.exists())
      {
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
      }
      else
      {
        shard_points = runtime->get_node(local_space);
        add_launch_space_reference(shard_points);
        std::vector<ApBarrier> copy_pre_barriers, copy_post_barriers;
        IndexCopyOp::trigger_replay();
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndexCopyOp::exchange_indirect_records(
        const unsigned index, const ApEvent local_pre, const ApEvent local_post,
        ApEvent& collective_pre, ApEvent& collective_post,
        const TraceInfo& trace_info, const InstanceSet& insts,
        const RegionRequirement& req, std::vector<IndirectRecord>& records,
        const bool sources)
    //--------------------------------------------------------------------------
    {
      if (sources && !collective_src_indirect_points)
        return CopyOp::exchange_indirect_records(
            index, local_pre, local_post, collective_pre, collective_post,
            trace_info, insts, req, records, sources);
      if (!sources && !collective_dst_indirect_points)
        return CopyOp::exchange_indirect_records(
            index, local_pre, local_post, collective_pre, collective_post,
            trace_info, insts, req, records, sources);
      legion_assert(local_pre.exists());
      legion_assert(local_post.exists());
      legion_assert(index < pre_indirection_barriers.size());
      legion_assert(index < post_indirection_barriers.size());
      // Take the lock and record our sets and instances
      AutoLock o_lock(op_lock);
      legion_assert(index < collective_exchanges.size());
      IndirectionExchange& exchange = collective_exchanges[index];
      if (sources)
      {
        collective_pre = pre_indirection_barriers[index];
        collective_post = post_indirection_barriers[index];
        ;
        if (!exchange.src_ready.exists())
          exchange.src_ready = Runtime::create_rt_user_event();
        if (exchange.local_preconditions.size() < points.size())
        {
          exchange.local_preconditions.insert(local_pre);
          if (exchange.local_preconditions.size() == points.size())
          {
            const ApEvent local_precondition = Runtime::merge_events(
                &trace_info, exchange.local_preconditions);
            runtime->phase_barrier_arrive(
                pre_indirection_barriers[index], 1 /*count*/,
                local_precondition);
            if (trace_info.recording)
            {
              std::pair<size_t, size_t> key(trace_local_id, index);
              trace_info.record_collective_barrier(
                  pre_indirection_barriers[index], local_precondition, key);
            }
          }
        }
        if (exchange.local_postconditions.size() < points.size())
        {
          exchange.local_postconditions.insert(local_post);
          if (exchange.local_postconditions.size() == points.size())
          {
            const ApEvent local_postcondition = Runtime::merge_events(
                &trace_info, exchange.local_postconditions);
            runtime->phase_barrier_arrive(
                post_indirection_barriers[index], 1 /*count*/,
                local_postcondition);
            if (trace_info.recording)
            {
              std::pair<size_t, size_t> key(
                  trace_local_id, pre_indirection_barriers.size() + index);
              trace_info.record_collective_barrier(
                  post_indirection_barriers[index], local_postcondition, key);
            }
          }
        }
        legion_assert(index < src_indirect_records.size());
        legion_assert(src_indirect_records[index].size() < points.size());
        src_indirect_records[index].emplace_back(
            IndirectRecord(req, insts, launch_space->get_volume()));
        exchange.src_records.emplace_back(&records);
        if (src_indirect_records[index].size() == points.size())
          return finalize_exchange(index, true /*sources*/);
        return exchange.src_ready;
      }
      else
      {
        collective_pre = pre_indirection_barriers[index];
        collective_post = post_indirection_barriers[index];
        if (!exchange.dst_ready.exists())
          exchange.dst_ready = Runtime::create_rt_user_event();
        if (exchange.local_preconditions.size() < points.size())
        {
          exchange.local_preconditions.insert(local_pre);
          if (exchange.local_preconditions.size() == points.size())
          {
            const ApEvent local_precondition = Runtime::merge_events(
                &trace_info, exchange.local_preconditions);
            runtime->phase_barrier_arrive(
                pre_indirection_barriers[index], 1 /*count*/,
                local_precondition);
            if (trace_info.recording)
            {
              std::pair<size_t, size_t> key(trace_local_id, index);
              trace_info.record_collective_barrier(
                  pre_indirection_barriers[index], local_precondition, key);
            }
          }
        }
        if (exchange.local_postconditions.size() < points.size())
        {
          exchange.local_postconditions.insert(local_post);
          if (exchange.local_postconditions.size() == points.size())
          {
            const ApEvent local_postcondition = Runtime::merge_events(
                &trace_info, exchange.local_postconditions);
            runtime->phase_barrier_arrive(
                post_indirection_barriers[index], 1 /*count*/,
                local_postcondition);
            if (trace_info.recording)
            {
              std::pair<size_t, size_t> key(
                  trace_local_id, pre_indirection_barriers.size() + index);
              trace_info.record_collective_barrier(
                  post_indirection_barriers[index], local_postcondition, key);
            }
          }
        }
        legion_assert(index < dst_indirect_records.size());
        legion_assert(dst_indirect_records[index].size() < points.size());
        dst_indirect_records[index].emplace_back(
            IndirectRecord(req, insts, launch_space->get_volume()));
        exchange.dst_records.emplace_back(&records);
        if (dst_indirect_records[index].size() == points.size())
          return finalize_exchange(index, false /*sources*/);
        return exchange.dst_ready;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndexCopyOp::finalize_exchange(
        const unsigned index, const bool source)
    //--------------------------------------------------------------------------
    {
      IndirectionExchange& exchange = collective_exchanges[index];
      if (source)
      {
        legion_assert(index < src_collectives.size());
        const RtEvent ready = src_collectives[index]->exchange_records(
            exchange.src_records, src_indirect_records[index]);
        if (exchange.src_ready.exists())
        {
          Runtime::trigger_event(exchange.src_ready, ready);
          return exchange.src_ready;
        }
        else
          return ready;
      }
      else
      {
        legion_assert(index < dst_collectives.size());
        const RtEvent ready = dst_collectives[index]->exchange_records(
            exchange.dst_records, dst_indirect_records[index]);
        if (exchange.dst_ready.exists())
        {
          Runtime::trigger_event(exchange.dst_ready, ready);
          return exchange.dst_ready;
        }
        else
          return ready;
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      if (!src_indirect_requirements.empty() && collective_src_indirect_points)
      {
        src_collectives.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          src_collectives[idx] = new IndirectRecordExchange(
              ctx, ctx->get_next_collective_index(COLLECTIVE_LOC_80));
      }
      if (!dst_indirect_requirements.empty() && collective_dst_indirect_points)
      {
        dst_collectives.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          dst_collectives[idx] = new IndirectRecordExchange(
              ctx, ctx->get_next_collective_index(COLLECTIVE_LOC_81));
      }
      if (!src_indirect_requirements.empty() ||
          !dst_indirect_requirements.empty())
      {
        legion_assert(
            src_indirect_requirements.empty() ||
            dst_indirect_requirements.empty() ||
            (src_indirect_requirements.size() ==
             dst_indirect_requirements.size()));
        pre_indirection_barriers.resize(
            (src_indirect_requirements.size() >
             dst_indirect_requirements.size()) ?
                src_indirect_requirements.size() :
                dst_indirect_requirements.size());
        post_indirection_barriers.resize(pre_indirection_barriers.size());
        for (unsigned idx = 0; idx < pre_indirection_barriers.size(); idx++)
        {
          pre_indirection_barriers[idx] = ctx->get_next_indirection_barriers();
          ;
          post_indirection_barriers[idx] = pre_indirection_barriers[idx];
          Runtime::advance_barrier(post_indirection_barriers[idx]);
        }
      }
      if (runtime->safe_model)
        interfering_check_id =
            ctx->get_next_collective_index(COLLECTIVE_LOC_69);
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndexCopyOp::find_intra_space_dependence(
        const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      Domain launch_domain;
      if (sharding_space.exists())
        runtime->find_domain(sharding_space, launch_domain);
      else
        launch_domain = launch_space->get_tight_domain();
      const ShardID point_shard =
          sharding_function->find_owner(point, launch_domain);
      if (point_shard == repl_ctx->owner_shard->shard_id)
        return IndexCopyOp::find_intra_space_dependence(point);
      else
        return repl_ctx->find_pointwise_dependence(
            context_index, point, point_shard, true /*intra space*/);
    }

    //--------------------------------------------------------------------------
    bool ReplIndexCopyOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function != nullptr);
      if (sharding_space.exists())
        return sharding_function->find_shard_participants(
            launch_space, sharding_space, shards);
      else
        return sharding_function->find_shard_participants(
            launch_space, launch_space->handle, shards);
    }

    /////////////////////////////////////////////////////////////
    // Remote Copy Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteCopyOp::RemoteCopyOp(Operation* ptr, AddressSpaceID src)
      : ExternalCopy(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteCopyOp::~RemoteCopyOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteCopyOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteCopyOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteCopyOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteCopyOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteCopyOp::get_provenance_string(
        bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    const char* RemoteCopyOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[COPY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteCopyOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return COPY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking, points);
        return;
      }
      Mapper::SelectCopySrcInput input;
      Mapper::SelectCopySrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      input.is_src = false;
      input.is_dst = false;
      input.is_src_indirect = false;
      input.is_dst_indirect = false;
      unsigned mod_index = index;
      if (mod_index < src_requirements.size())
      {
        input.region_req_index = mod_index;
        input.is_src = true;
      }
      else
      {
        mod_index -= src_requirements.size();
        if (mod_index < dst_requirements.size())
        {
          input.region_req_index = mod_index;
          input.is_dst = true;
        }
        else
        {
          mod_index -= dst_requirements.size();
          if (mod_index < src_indirect_requirements.size())
          {
            input.region_req_index = mod_index;
            input.is_src_indirect = true;
          }
          else
          {
            mod_index -= src_indirect_requirements.size();
            legion_assert(mod_index < dst_indirect_requirements.size());
            input.is_dst_indirect = true;
          }
        }
      }
      if (mapper == nullptr)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_copy_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_copy(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteCopyOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_copy(derez);
      unpack_profiling_requests(derez);
    }

  }  // namespace Internal
}  // namespace Legion
