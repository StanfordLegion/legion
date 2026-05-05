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

#include "legion/contexts/leaf.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/api/physical_region_impl.h"
#include "legion/managers/mapper.h"
#include "legion/tasks/index.h"
#include "legion/tasks/individual.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Leaf Context
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LeafContext::LeafContext(
        SingleTask* owner,
        std::map<std::pair<Memory, bool>, MemoryPool*>&& pools,
        bool inline_task)
      : TaskContext(
            owner, owner->get_depth(), owner->regions, owner->output_regions,
            LEGION_DISTRIBUTED_HELP_ENCODE(
                runtime->get_available_distributed_id(), LEAF_CONTEXT_DC),
            false /*perform registration*/, inline_task),
        memory_pools(pools), inlined_tasks(0)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Leaf Context %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    LeafContext::~LeafContext(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(memory_pools.empty());
    }

    //--------------------------------------------------------------------------
    ContextID LeafContext::get_logical_tree_context(void) const
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    ContextID LeafContext::get_physical_tree_context(void) const
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    void LeafContext::compute_task_tree_coordinates(
        TaskTreeCoordinates& coordinates) const
    //--------------------------------------------------------------------------
    {
      owner_task->compute_task_tree_coordinates(coordinates);
    }

    //--------------------------------------------------------------------------
    void LeafContext::inline_child_task(TaskOp* child)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_inline_task(child->get_unique_id());
      // Find the mapped physical regions associated with each of the
      // child task's region requirements. If they aren't mapped then
      // we need a mapping fence to ensure that all the mappings are
      // done before we attempt to run this task. If they are all mapped
      // though then we can run this right away.
      std::vector<PhysicalRegion> child_regions(child->regions.size());
      for (unsigned childidx = 0; childidx < child_regions.size(); childidx++)
      {
        const RegionRequirement& child_req = child->regions[childidx];
        [[maybe_unused]] bool found = false;
        for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
        {
          if (!physical_regions[our_idx].is_mapped())
            continue;
          const RegionRequirement& our_req = regions[our_idx];
          const RegionTreeID our_tid = our_req.region.get_tree_id();
          const IndexSpace our_space = our_req.region.get_index_space();
          const RegionUsage our_usage(our_req);
          if (!check_region_dependence(
                  our_tid, our_space, our_req, our_usage, child_req,
                  false /*ignore privileges*/))
            continue;
          child_regions[childidx] = physical_regions[our_idx];
          found = true;
          break;
        }
        legion_assert(found);
      }
      // Now select the variant for task based on the regions
      std::deque<InstanceSet> physical_instances(child_regions.size());
      VariantImpl* variant =
          select_inline_variant(child, child_regions, physical_instances);
      child->perform_inlining(variant, physical_instances);
      // Finish the inlining of the child task to execute, note this doesn't
      // wait for the effects of the children to be done, it just blocks to
      // make sure the code for the children are done running on this processor
      wait_for_inlined();
    }

    //--------------------------------------------------------------------------
    VariantImpl* LeafContext::select_inline_variant(
        TaskOp* child, const std::vector<PhysicalRegion>& parent_regions,
        std::deque<InstanceSet>& physical_instances)
    //--------------------------------------------------------------------------
    {
      VariantImpl* variant_impl = TaskContext::select_inline_variant(
          child, parent_regions, physical_instances);
      if (!variant_impl->is_leaf())
      {
        MapperManager* child_mapper =
            runtime->find_mapper(executing_processor, child->map_id);
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of "
                 "'select_task_variant' on mapper "
              << child_mapper->get_mapper_name()
              << ". Mapper selected an invalid variant ID " << variant_impl->vid
              << " for inlining of task " << child->get_task_name() << " (UID "
              << child->get_unique_id() << "). Parent task "
              << owner_task->get_task_name() << " (UID "
              << owner_task->get_unique_id()
              << ") is a leaf task but mapper selected non-leaf variant "
              << variant_impl->vid << " for task " << child->get_task_name()
              << ".";
        error.raise();
      }
      return variant_impl;
    }

    //--------------------------------------------------------------------------
    bool LeafContext::is_leaf_context(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent LeafContext::find_pointwise_dependence(
        uint64_t context_index, const DomainPoint& point, ShardID shard,
        bool intra_space, RtUserEvent to_trigger,
        std::optional<unsigned> output_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(!output_index);
      // The only reason we're here is if we're inlining index tasks
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void LeafContext::return_resources(
        ResourceTracker* target, uint64_t return_index,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void LeafContext::pack_return_resources(
        Serializer& rez, uint64_t return_index)
    //--------------------------------------------------------------------------
    {
      ResourceTracker::pack_empty_resources(rez, return_index);
    }

    //--------------------------------------------------------------------------
    void LeafContext::log_created_requirements(void) const
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void LeafContext::report_leaks_and_duplicates(
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(
        const Domain& bounds, bool take_ownership, TypeTag type_tag,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index space creation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(
        const std::vector<DomainPoint>& points, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index space creation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(
        const std::vector<Domain>& rects, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index space creation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(
        const Future& f, TypeTag tag, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index space creation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_unbound_index_space(
        TypeTag tag, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal unbound index space creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::create_shared_ownership(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index space create shared ownership performed in leaf "
               "task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::union_index_spaces(
        const std::vector<IndexSpace>& spaces, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal union index spaces performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::intersect_index_spaces(
        const std::vector<IndexSpace>& spaces, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal intersect index spaces performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::subtract_index_spaces(
        IndexSpace left, IndexSpace right, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal subtract index spaces performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_space(
        IndexSpace handle, const bool unordered, const bool recurse,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index space destruction performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::create_shared_ownership(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index partition create shared ownership performed in "
               "leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_partition(
        IndexPartition handle, const bool unordered, const bool recurse,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index partition destruction performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_equal_partition(
        IndexSpace parent, IndexSpace color_space, size_t granularity,
        Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal equal partition creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_weights(
        IndexSpace parent, const FutureMap& weights, IndexSpace color_space,
        size_t granularity, Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create partition by weights performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_union(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal union partition creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_intersection(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal intersection partition creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_intersection(
        IndexSpace parent, IndexPartition partition, PartitionKind kind,
        Color color, bool dominates, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal intersection partition creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_difference(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal difference partition creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Color LeafContext::create_cross_product_partitions(
        IndexPartition handle1, IndexPartition handle2,
        std::map<IndexSpace, IndexPartition>& handles, PartitionKind kind,
        Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create cross product partitions performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::create_association(
        LogicalRegion domain, LogicalRegion domain_parent, FieldID domain_fid,
        IndexSpace range, MapperID id, MappingTagID tag,
        const UntypedBuffer& marg, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create association performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_restricted_partition(
        IndexSpace parent, IndexSpace color_space, const void* transform,
        size_t transform_size, const void* extent, size_t extent_size,
        PartitionKind part_kind, Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create restricted partition performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_domain(
        IndexSpace parent, const FutureMap& domains, IndexSpace color_space,
        bool perform_intersections, PartitionKind part_kind, Color color,
        Provenance* provenance, bool skip_check)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create partition by domain performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_field(
        LogicalRegion handle, LogicalRegion parent_priv, FieldID fid,
        IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
        PartitionKind part_kind, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal partition by field performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_image(
        IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal partition by image performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_image_range(
        IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal partition by image range performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_preimage(
        IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal partition by preimage performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_preimage_range(
        IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal partition by preimage range performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_pending_partition(
        IndexSpace parent, IndexSpace color_space, PartitionKind part_kind,
        Color color, Provenance* prov, bool trust)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create pending partition performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, const std::vector<IndexSpace>& handles,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create index space union performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexPartition handle, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create index space union performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, const std::vector<IndexSpace>& handles,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create index space intersection performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexPartition handle, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create index space intersection performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_difference(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexSpace initial,
        const std::vector<IndexSpace>& handles, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal create index space difference performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FieldSpace LeafContext::create_field_space(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field space creation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FieldSpace LeafContext::create_field_space(
        const std::vector<size_t>& sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field space creation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FieldSpace LeafContext::create_field_space(
        const std::vector<Future>& sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field space creation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::create_shared_ownership(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field space create shared ownership performed in leaf "
               "task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_field_space(
        FieldSpace handle, const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field space destruction performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FieldAllocatorImpl* LeafContext::create_field_allocator(
        FieldSpace handle, bool unordered)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field allocator creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_field_allocator(FieldSpaceNode* node)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field allocator destruction performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FieldID LeafContext::allocate_field(
        FieldSpace space, size_t field_size, FieldID fid, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field allocation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_fields(
        FieldSpace space, const std::vector<size_t>& sizes,
        std::vector<FieldID>& resulting_fields, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field allocations performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_field(
        FieldAllocatorImpl* allocator, FieldSpace space, FieldID fid,
        const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field free performed in leaf task " << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_fields(
        FieldAllocatorImpl* allocator, FieldSpace space,
        const std::set<FieldID>& to_free, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal field free performed in leaf task " << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FieldID LeafContext::allocate_field(
        FieldSpace space, const Future& field_size, FieldID fid, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal deferred field allocation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_local_field(
        FieldSpace space, size_t field_size, FieldID fid,
        CustomSerdezID serdez_id, std::set<RtEvent>& done_events,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal local field allocation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_fields(
        FieldSpace space, const std::vector<Future>& sizes,
        std::vector<FieldID>& resuling_fields, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal deferred field allocations performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_local_fields(
        FieldSpace space, const std::vector<size_t>& sizes,
        const std::vector<FieldID>& resuling_fields, CustomSerdezID serdez_id,
        std::set<RtEvent>& done_events, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal local field allocations performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    LogicalRegion LeafContext::create_logical_region(
        IndexSpace index_space, FieldSpace field_space, const bool task_local,
        Provenance* provenance, const bool output_region)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal logical region creation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::create_shared_ownership(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal logical region create shared ownership performed in "
               "leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_logical_region(
        LogicalRegion handle, const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal logical region deletion performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::reset_equivalence_sets(
        LogicalRegion parent, LogicalRegion region,
        const std::set<FieldID>& fields)
    //--------------------------------------------------------------------------
    {
      // No-op
    }

    //--------------------------------------------------------------------------
    void LeafContext::get_local_field_set(
        const FieldSpace handle, const std::set<unsigned>& indexes,
        std::set<FieldID>& to_set) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void LeafContext::get_local_field_set(
        const FieldSpace handle, const std::set<unsigned>& indexes,
        std::vector<FieldID>& to_set) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_physical_region(
        const RegionRequirement& req, bool mapped, MapperID mid,
        MappingTagID tag, ApUserEvent& unmap_event, bool virtual_mapped,
        const InstanceSet& physical_instances)
    //--------------------------------------------------------------------------
    {
      legion_assert(!unmap_event.exists());
      PhysicalRegionImpl* impl = new PhysicalRegionImpl(
          req, RtEvent::NO_RT_EVENT, ApEvent::NO_AP_EVENT,
          ApUserEvent::NO_AP_USER_EVENT, mapped, this, mid, tag,
          true /*leaf region*/, virtual_mapped, false /*collective*/,
          InnerContext::NO_BLOCKING_INDEX, physical_regions.size());
      physical_regions.emplace_back(PhysicalRegion(impl));
      if (mapped)
        impl->set_references(physical_instances, true /*safe*/);
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_task(
        const TaskLauncher& launcher, std::vector<OutputRequirement>* outputs,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.enable_inlining)
      {
        if (launcher.predicate == Predicate::FALSE_PRED)
          return predicate_task_false(launcher, provenance);
        IndividualTask* task = runtime->get_operation<IndividualTask>();
        InnerContext* parent = owner_task->get_context();
        Future result = task->initialize_task(
            parent, launcher, provenance, false /*top level*/,
            false /*must epoch*/, outputs);
        inline_child_task(task);
        return result;
      }
      else
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal execute task call performed in leaf task " << *this
              << ".";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::execute_index_space(
        const IndexTaskLauncher& launcher,
        std::vector<OutputRequirement>* outputs, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (!launcher.must_parallelism && launcher.enable_inlining)
      {
        IndexSpace launch_space = launcher.launch_space;
        if (!launch_space.exists())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal execute index space call performed in leaf task "
                << *this
                << ". All inline leaf task index space launches must specify a "
                   "launch index space.";
          error.raise();
        }
        if (launcher.predicate == Predicate::FALSE_PRED)
          return predicate_index_task_false(launch_space, launcher, provenance);
        IndexTask* task = runtime->get_operation<IndexTask>();
        InnerContext* parent = owner_task->get_context();
        FutureMap result = task->initialize_task(
            parent, launcher, launch_space, provenance, false /*track*/,
            outputs);
        inline_child_task(task);
        return result;
      }
      else
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal execute index space call performed in leaf task "
              << *this << ".";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_index_space(
        const IndexTaskLauncher& launcher, ReductionOpID redop,
        bool deterministic, std::vector<OutputRequirement>* outputs,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (!launcher.must_parallelism && launcher.enable_inlining)
      {
        IndexSpace launch_space = launcher.launch_space;
        if (!launch_space.exists())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal execute index space call performed in leaf task "
                << *this
                << ". All inline leaf task index space launches must specify a "
                   "launch index space.";
          error.raise();
        }
        if (launcher.predicate == Predicate::FALSE_PRED)
          return predicate_index_task_reduce_false(
              launcher, launch_space, redop, provenance);
        IndexTask* task = runtime->get_operation<IndexTask>();
        InnerContext* parent = owner_task->get_context();
        Future result = task->initialize_task(
            parent, launcher, launch_space, provenance, redop, deterministic,
            false /*track*/, outputs);
        inline_child_task(task);
        return result;
      }
      else
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal execute index space call performed in leaf task "
              << *this << ".";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    Future LeafContext::reduce_future_map(
        const FutureMap& future_map, ReductionOpID redop, bool deterministic,
        MapperID mapper_id, MappingTagID tag, Provenance* provenance,
        Future initial_value)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal reduce future map call performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(
        IndexSpace domain, const std::map<DomainPoint, UntypedBuffer>& data,
        Provenance* provenance, bool collective, ShardingID sid, bool implicit,
        bool check_space)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal construct future map call performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(
        const Domain& domain, const std::map<DomainPoint, UntypedBuffer>& data,
        bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal construct future map call performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(
        IndexSpace domain, const std::map<DomainPoint, Future>& futures,
        Provenance* provenance, bool collective, ShardingID sid, bool implicit,
        bool check_space)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal construct future map call performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(
        const Domain& domain, const std::map<DomainPoint, Future>& futures,
        bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal construct future map call performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::transform_future_map(
        const FutureMap& fm, IndexSpace new_domain,
        PointTransformFunctor* functor, bool own_func, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal transform future map call performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion LeafContext::map_region(
        const InlineLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal map_region operation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    ApEvent LeafContext::remap_region(
        const PhysicalRegion& region, Provenance* provenance, bool internal)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal remap operation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal unmap operation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::unmap_all_regions(bool external)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal unmap_all_regions call performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::fill_fields(
        const FillLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal fill operation call performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::fill_fields(
        const IndexFillLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index fill operation call performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::discard_fields(
        const DiscardLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal discard operation call performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_copy(
        const CopyLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal copy operation call performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_copy(
        const IndexCopyLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index copy operation call performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_acquire(
        const AcquireLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal acquire operation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_release(
        const ReleaseLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal release operation performed in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion LeafContext::attach_resource(
        const AttachLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal attach resource operation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    ExternalResources LeafContext::attach_resources(
        const IndexAttachLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal attach resources operation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::detach_resource(
        PhysicalRegion region, const bool flush, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal detach resource operation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::detach_resources(
        ExternalResources resources, const bool flush, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal index detach resource operation performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::progress_unordered_operations(bool end_task)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal progress unordered operations performed in leaf task "
            << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::execute_must_epoch(
        const MustEpochLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal Legion execute must epoch call in leaf task " << *this
            << ".";
      error.raise();
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::issue_timing_measurement(
        const TimingLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // We can permit timing operations in leaf tasks since they should
      // just be done immediately. We can ignore any incoming futures on
      // the launcher since we know they will have already triggered since
      // the only futures made in leaf tasks are for inline execution.
      FutureImpl* future = new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance);
      switch (launcher.measurement)
      {
        case LEGION_MEASURE_SECONDS:
          {
            double value = Realm::Clock::current_time();
            future->set_local(&value, sizeof(value));
            break;
          }
        case LEGION_MEASURE_MICRO_SECONDS:
          {
            long long value = Realm::Clock::current_time_in_microseconds();
            future->set_local(&value, sizeof(value));
            break;
          }
        case LEGION_MEASURE_NANO_SECONDS:
          {
            long long value = Realm::Clock::current_time_in_nanoseconds();
            future->set_local(&value, sizeof(value));
            break;
          }
        default:
          std::abort();
      }
      return Future(future);
    }

    //--------------------------------------------------------------------------
    Future LeafContext::select_tunable_value(
        const TunableLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // We can permit timing operations in leaf tasks since they should
      // just be done immediately. We can ignore any incoming futures on
      // the launcher since we know they will have already triggered since
      // the only futures made in leaf tasks are for inline execution.
      FutureImpl* future = new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance);
      MapperManager* mapper =
          runtime->find_mapper(get_executing_processor(), launcher.mapper);
      Mapper::SelectTunableInput input;
      Mapper::SelectTunableOutput output;
      input.tunable_id = launcher.tunable;
      input.mapping_tag = launcher.tag;
      input.futures = launcher.futures;
      input.args = launcher.arg.get_ptr();
      input.size = launcher.arg.get_size();
      output.value = nullptr;
      output.size = 0;
      output.take_ownership = true;
      mapper->invoke_select_tunable_value(owner_task, input, output);
      LegionSpy::log_tunable_value(
          get_unique_id(), get_tunable_index(), output.value, output.size);
      future->set_local(output.value, output.size, output.take_ownership);
      return Future(future);
    }

    //--------------------------------------------------------------------------
    Future LeafContext::issue_mapping_fence(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal Legion mapping fence call in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::issue_execution_fence(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal Legion execution fence call in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::complete_frame(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal Legion complete frame call in leaf task " << *this
            << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(
        const Future& f, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (f.impl == nullptr)
        return Predicate::FALSE_PRED;
      const bool value =
          *(const bool*)f.impl->get_buffer(runtime->runtime_system_memory);
      if (value)
        return Predicate::TRUE_PRED;
      else
        return Predicate::FALSE_PRED;
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::predicate_not(
        const Predicate& p, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (p == Predicate::TRUE_PRED)
        return Predicate::FALSE_PRED;
      else if (p == Predicate::FALSE_PRED)
        return Predicate::TRUE_PRED;
      else  // should never get here, all predicates should be eagerly evaluated
        std::abort();
      return Predicate::TRUE_PRED;
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(
        const PredicateLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.predicates.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal predicate creation performed on a set of empty "
              << "previous predicates in leaf task " << *this << ".";
        error.raise();
      }
      else if (launcher.predicates.size() == 1)
        return launcher.predicates[0];
      if (launcher.and_op)
      {
        // Check for short circuit cases
        for (const Predicate& it : launcher.predicates)
        {
          if (it == Predicate::FALSE_PRED)
            return Predicate::FALSE_PRED;
          else if (it == Predicate::TRUE_PRED)
            continue;
          else  // should never get here,
            // all predicates should be eagerly evaluated
            std::abort();
        }
        return Predicate::TRUE_PRED;
      }
      else
      {
        // Check for short circuit cases
        for (const Predicate& it : launcher.predicates)
        {
          if (it == Predicate::TRUE_PRED)
            return Predicate::TRUE_PRED;
          else if (it == Predicate::FALSE_PRED)
            continue;
          else  // should never get here,
            // all predicates should be eagerly evaluated
            std::abort();
        }
        return Predicate::FALSE_PRED;
      }
    }

    //--------------------------------------------------------------------------
    Future LeafContext::get_predicate_future(
        const Predicate& p, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (p == Predicate::TRUE_PRED)
      {
        Future result(new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance));
        const bool value = true;
        result.impl->set_local(&value, sizeof(value));
        return result;
      }
      else if (p == Predicate::FALSE_PRED)
      {
        Future result(new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance));
        const bool value = false;
        result.impl->set_local(&value, sizeof(value));
        return result;
      }
      else  // should never get here, all predicates should be eagerly
            // evaluated
        std::abort();
      return Future();
    }

    //--------------------------------------------------------------------------
    void LeafContext::begin_trace(
        TraceID tid, bool logical_only, bool static_trace,
        const std::set<RegionTreeID>* trees, bool deprecated,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal Legion begin trace call in leaf task " << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_trace(
        TraceID tid, bool deprecated, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal Legion end trace call in leaf task " << *this << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::record_blocking_call(
        uint64_t blocking_index, bool invalidate_trace)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void LeafContext::wait_on_future(FutureImpl* future, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      ready.wait();
    }

    //--------------------------------------------------------------------------
    void LeafContext::wait_on_future_map(FutureMapImpl* map, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      ready.wait();
    }

    //--------------------------------------------------------------------------
    InnerContext* LeafContext::find_top_context(InnerContext* previous)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    void LeafContext::initialize_region_tree_contexts(
        const std::vector<RegionRequirement>& clone_requirements,
        const std::vector<ApUserEvent>& unmap_events)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void LeafContext::invalidate_logical_context(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void LeafContext::invalidate_region_tree_contexts(
        const bool is_top_level_task, std::set<RtEvent>& applied,
        const ShardMapping* mapping, ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    FutureInstance* LeafContext::create_task_local_future(
        Memory memory, size_t size, bool silence_warnings,
        const char* warning_string)
    //--------------------------------------------------------------------------
    {
      const std::pair<Memory, bool> key(memory, true /*escaping*/);
      std::map<std::pair<Memory, bool>, MemoryPool*>::const_iterator finder =
          memory_pools.find(key);
      if (finder == memory_pools.end())
      {
        TaskTreeCoordinates coordinates;
        compute_task_tree_coordinates(coordinates);
        MemoryManager* manager = runtime->find_memory_manager(memory);
        // This is not safe to block indefinitely on unbounded pools because
        // the unbounded pool might be for a task that depends on us running
        RtEvent safe_for_unbounded_pools;
        // A tiny bit of backwards compatibility here for system level
        // futures in which case we know this will go to the fast path of
        // just calling malloc without relying on Realm's allocator
        if ((memory == runtime->runtime_system_memory) &&
            (size <= LEGION_MAX_RETURN_SIZE))
          return manager->create_future_instance(
              get_unique_id(), coordinates, size, &safe_for_unbounded_pools);
        // WE'RE ABOUT TO DO SOMETHING DANGEROUS!
        // The user didn't bother to pre-allocate a pool so we're going
        // to try to make an immediate instance that has no event precondition
        // If we can do that then we can still use that instance, but if we're
        // given an instance with a precondition we cannot wait for it under
        // any circumstances without risking a deadlock
        FutureInstance* instance = manager->create_future_instance(
            get_unique_id(), coordinates, size, &safe_for_unbounded_pools);
        if (instance != nullptr)
        {
          if (instance->is_immediate())
          {
            if (!silence_warnings)
            {
              Warning warning;
              warning
                  << "WARNING! Leaf task " << get_task_name() << " (UID "
                  << get_unique_id()
                  << ") attempted to allocate a future instance of " << size
                  << " bytes in " << manager->get_name()
                  << " memory but no space was reserved for dynamic "
                     "allocations during "
                  << "the lifetime of this task. Legion has managed to procure "
                  << "for you an allocation this time but there is no "
                     "guarantee "
                  << "that you will be so lucky the next time. We strongly "
                  << "encourage all users to place tight upper bounds on the "
                  << "required memory for all leaf tasks either statically at "
                  << "the point of task variant registration or dynamically at "
                  << "the point that the task is mapped. Warning string: "
                  << ((warning_string == nullptr) ? "" : warning_string) << ".";
              warning.raise();
            }
            return instance;
          }
          else
            delete instance;  // Not immediately available so we can't use it
        }
        else if (safe_for_unbounded_pools.exists())
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Failed to allocate " << size
                << " bytes for a future needed by leaf task " << get_task_name()
                << " (UID " << get_unique_id() << ") in " << manager->get_name()
                << " memory because there was no space reserved at "
                << "the point of mapping the task for dynamic allocations. If "
                   "you "
                << "designate a task as a leaf task variant then it is your "
                << "responsibility to tell Legion how much memory needs to be "
                << "reserved for satisfying dynamic allocations during the "
                << "execution of the task. Legion did try to allocate an eager "
                << "instance this case but discovered an unbounded pool in the "
                << "memory which prevented us from attempted the eager "
                   "allocation "
                << "(because it cannot be done safely), so you might not "
                   "actually "
                << "be out of memory.";
          error.raise();
        }
      }
      else if (finder->second->is_released())
      {
        MemoryManager* manager = runtime->find_memory_manager(memory);
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Failed to allocate future in leaf task " << get_task_name()
              << " (UID " << get_unique_id() << ") in " << manager->get_name()
              << " memory because the pool associated with this memory was "
                 "already released "
              << "by the task. It is illegal to attempt to perform dynamic "
              << "allocations in a memory pool after you released it.";
        error.raise();
      }
      FutureInstance* instance =
          finder->second->allocate_future(get_unique_id(), size);
      if (instance == nullptr)
      {
        MemoryManager* manager = runtime->find_memory_manager(memory);
        const size_t memory_limit = manager->query_available_memory();
        if (finder->second->get_bounds().scope == LEGION_BOUNDED_POOL)
        {
          const size_t pool_limit = finder->second->query_memory_limit();
          const size_t remaining = finder->second->query_available_memory();
          if (remaining < size)
          {
            Error error(LEGION_RESOURCE_EXCEPTION);
            error << "Failed to allocate " << size
                  << " bytes for future needed by leaf task " << get_task_name()
                  << " (UID " << get_unique_id() << ") in "
                  << manager->get_name()
                  << " memory because there was insufficient space "
                  << "reserved for dynamic allocations. Only " << remaining
                  << " bytes remain of " << pool_limit
                  << " reserved bytes. This means that "
                  << "you set your upper bound for the amount of dynamic "
                     "memory required "
                  << "for this task too low.";
            error.raise();
          }
          else
          {
            Error error(LEGION_RESOURCE_EXCEPTION);
            error
                << "Failed to allocate " << size
                << " bytes for future needed by leaf task " << get_task_name()
                << " (UID " << get_unique_id() << ") in " << manager->get_name()
                << " memory because the pool reserved for dynamic "
                << "memory allocations has become fragmented. There are still "
                << remaining << " bytes remaining in the pool of " << pool_limit
                << " bytes, but they are fragmented such that a hole of "
                << size
                << " bytes cannot be found. We recommend you check the order "
                   "of "
                << "allocations and alignment requirements to try to minimize "
                   "the "
                << "amount of padding between instances. Otherwise you will "
                   "need to "
                << "request a larger pool for dynamic allocations that "
                   "considers the "
                << "necessary padding required between instances to satisfy "
                   "your "
                << "alignment needs.";
            error.raise();
          }
        }
        else if (memory_limit < size)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate " << size
              << " bytes for future needed by leaf task " << get_task_name()
              << " (UID " << get_unique_id() << ") in " << manager->get_name()
              << " memory because there was insufficient space "
              << "reserved for dynamic allocations. This was an unbounded "
                 "memory "
              << "pool which means you're actually out of space in this memory "
              << "because it only has " << memory_limit
              << " remaining free bytes. "
              << "We strongly recommend all users put bounds on their dynamic "
                 "memory "
              << "usage so they can detect if space will be available for task "
              << "execution and if not select an alternative mapping.";
          error.raise();
        }
        else
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate " << size
              << " bytes for future needed by leaf task " << get_task_name()
              << " (UID " << get_unique_id() << ") in " << manager->get_name()
              << " memory because the memory is fragmented. "
              << "This was an unbounded memory pool and there are still "
              << memory_limit
              << " bytes free in the memory but not enough of them "
              << "are contiguous to allocate the future instance. We strongly "
              << "recommend all users put bounds on their dynamic memory usage "
                 "so "
              << "they can detect if space will be available for task "
                 "execution "
              << "and if not select an alternative mapping.";
          error.raise();
        }
      }
      return instance;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance LeafContext::create_task_local_instance(
        Memory memory, const Realm::InstanceLayoutGeneric& layout,
        bool can_fail, bool escaping, RtEvent& use_event)
    //--------------------------------------------------------------------------
    {
      const std::pair<Memory, bool> key(memory, escaping);
      // We support multiple OpenMP threads creating buffers in parallel
      AutoLock l_lock(leaf_lock);
      LgEvent unique_event;
      if (runtime->profiler != nullptr)
        Runtime::rename_event(unique_event);
      const size_t footprint = layout.bytes_used;
      std::map<std::pair<Memory, bool>, MemoryPool*>::const_iterator finder =
          memory_pools.find(key);
      // Handle a special case for zero-byte instances here
      if ((footprint == 0) || (finder == memory_pools.end()))
      {
        MemoryManager* manager = runtime->find_memory_manager(memory);
        // WE'RE ABOUT TO DO SOMETHING DANGEROUS!
        // The user didn't bother to pre-allocate a pool so we're going
        // to try to make an immediate instance that has no event precondition
        // If we can do that then we can still use that instance, but if we're
        // given an instance with a precondition we cannot wait for it under
        // any circumstances without risking a deadlock
        TaskTreeCoordinates coordinates;
        compute_task_tree_coordinates(coordinates);
        // It is NOT safe to block for unbounded pools when doing this
        // because those unbounded pools might be from tasks that are behind
        // us in program order and depend on us to finish running
        RtEvent safe_for_unbounded_pools;
        const PhysicalInstance instance = manager->create_task_local_instance(
            get_unique_id(), coordinates, unique_event, layout, use_event,
            &safe_for_unbounded_pools);
        if (footprint == 0)
        {
          legion_assert(instance.exists());
          legion_assert(!use_event.exists());
          task_local_instances[instance] =
              std::make_pair(unique_event, true /*escaping*/);
          return instance;
        }
        if (instance.exists())
        {
          if (!use_event.exists() || use_event.has_triggered())
          {
            Warning warning;
            warning
                << "WARNING! Leaf task " << get_task_name() << " (UID "
                << get_unique_id() << ") attempted to allocate a "
                << "DeferredBuffer/Value/Reduction of " << footprint
                << " bytes in " << manager->get_name()
                << " memory but no space was reserved for "
                << "dynamic allocations during the lifetime of this task. "
                   "Legion "
                << "has managed to procure for you an allocation this time but "
                << "there is no guarantee that you will be so lucky the next "
                << "time. We strongly encourage all users to place tight "
                << "upper bounds on the required memory for all leaf tasks "
                << "either statically at the point of task variant "
                   "registration "
                << "or dynamically at the point that the task is mapped.";
            warning.raise();
            task_local_instances[instance] =
                std::make_pair(unique_event, true /*escaping*/);
            return instance;
          }
          instance.destroy(use_event);  // Can't use so destroy immediately
          if (can_fail)
            return PhysicalInstance::NO_INST;
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate DeferredBuffer/Value/Reduction of "
              << footprint << " bytes for leaf task " << *this << " in "
              << manager->get_name() << " memory because there was no "
              << "space reserved at the point of mapping the task for dynamic "
              << "allocations. If you designate a task as a leaf task variant "
              << "then it is your responsibility to tell Legion how much "
              << "memory needs to be allocated for satisfying dynamic "
              << "allocations during the execution of the task. Legion did "
              << "try to allocate an eager instance in this case and was able "
              << "to get an allocation but it was not immediately ready which "
              << "prevented it from being used without risking a resource "
              << "deadlock. This suggests that with appropriate upper bounds "
              << "your program will likely be able to fit in memory.";
          error.raise();
        }
        else if (can_fail)
          return instance;
        else if (safe_for_unbounded_pools.exists())
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate DeferredBuffer/Value/Reduction of "
              << footprint << " bytes for leaf task " << get_task_name()
              << " (UID " << get_unique_id() << ") in " << manager->get_name()
              << " memory because there was no space reserved at the point of "
              << "mapping the task for dynamic allocations. If you designate a "
              << "task as a leaf task variant then it is your responsibility "
              << "to tell Legion how much memory needs to be allocated for "
              << "satisfying dynamic allocations during the execution of the "
              << "task. Legion did try to allocate an eager instance in this "
              << "case but discovered an unbounded pool in the memory which "
              << "prevented the runtime from attempting the eager allocation "
              << "(because it cannot be done safely), so you "
              << "might not actually be out of memory.";
          error.raise();
        }
        else
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate DeferredBuffer/Value/Reduction of "
              << footprint << " bytes for leaf task " << *this << " in "
              << manager->get_name() << " memory because there was no "
              << "space reserved at the point of mapping the task for dynamic "
              << "allocations. If you designate a task as a leaf task variant "
              << "then it is your responsibility to tell Legion how much "
              << "memory needs to be allocated for satisfying dynamic "
              << "allocations during the execution of the task. Legion did "
              << "try to allocate an eager instance in this case but was "
              << "unable to get an instance meaning you are very likely out "
              << "of memory.";
          error.raise();
        }
      }
      else if (finder->second->is_released())
      {
        MemoryManager* manager = runtime->find_memory_manager(memory);
        Error error(LEGION_RESOURCE_EXCEPTION);
        error
            << "Failed to allocate DeferredBuffer/Value/Reduction in leaf "
            << "task " << *this << " in " << manager->get_name()
            << " memory because the pool associated "
            << "with this memory was already released by the task. It is "
            << "illegal to attempt to perform dynamic allocations in a memory "
            << "pool after it has been released it.";
        error.raise();
      }
      if (finder->second->max_alignment < layout.alignment_reqd)
      {
        MemoryManager* manager = runtime->find_memory_manager(memory);
        Error error(LEGION_RESOURCE_EXCEPTION);
        error
            << "Failed to allocate DeferredBuffer/Value/Reduction of "
            << footprint << " bytes for leaf task " << *this << " in "
            << manager->get_name() << " memory because the maximum "
            << "alignment required by the instance of " << layout.alignment_reqd
            << " bytes is larger the reserved alignment for the pool of "
            << finder->second->max_alignment << " bytes. You need to ask "
            << "for a larger maximum alignment for the pool if you plan to do "
            << "dynamic allocations that require it.";
        error.raise();
      }
      PhysicalInstance instance = finder->second->allocate_instance(
          get_unique_id(), unique_event, layout, use_event);
      if (!instance.exists())
      {
        if (can_fail)
          return instance;
        MemoryManager* manager = runtime->find_memory_manager(memory);
        const size_t memory_limit = manager->query_available_memory();
        if (finder->second->get_bounds().scope == LEGION_BOUNDED_POOL)
        {
          const size_t pool_limit = finder->second->query_memory_limit();
          const size_t remaining = finder->second->query_available_memory();
          if (remaining < footprint)
          {
            Error error(LEGION_RESOURCE_EXCEPTION);
            error
                << "Failed to allocate DeferredBuffer/Value/Reduction of "
                << footprint << " bytes for leaf task " << *this << " in "
                << manager->get_name() << " memory because there was "
                << "insufficient space reserved for dynamic allocations."
                << " Only " << remaining << " bytes remain of " << pool_limit
                << " reserved bytes. This means that you set your upper bound "
                << "for the amount of dynamic memory required for this "
                << "task too low.";
            error.raise();
          }
          else
          {
            Error error(LEGION_RESOURCE_EXCEPTION);
            error
                << "Failed to allocate DeferredBuffer/Value/Reduction of "
                << footprint << " bytes for leaf task " << *this << " in "
                << manager->get_name() << " memory because the memory is "
                << "fragmented. There are still " << remaining
                << " bytes free in the pool of " << pool_limit
                << "bytes but they are sufficiently fragmented such that a "
                   "hole "
                << footprint << " bytes aligned on a " << layout.alignment_reqd
                << " byte boundary cannot be found. We "
                << "recommend you check the order of allocations and alignment "
                << "requirements to try to minimize the amount of padding "
                   "between "
                << "instances. Otherwise you will need to request a larger "
                   "pool "
                << "for dynamic allocations that considers the necessary "
                   "padding "
                << "required between instances to satisfy your alignment "
                   "needs.";
            error.raise();
          }
        }
        else if (memory_limit < footprint)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate DeferredBuffer/Value/Reduction of "
              << footprint << " bytes for leaf task " << *this << " in "
              << manager->get_name() << " memory because there was "
              << "insufficient space reserved for dynamic allocations. This "
                 "was "
              << "an unbounded memory pool which means you're actually out of "
              << "space in this memory because it only has " << memory_limit
              << " remaining free bytes. We strongly recommend all users put "
              << "bounds on their dynamic memory usage so they can detect if "
              << "space will be available for task execution and if not "
              << "select an alternative mapping.";
          error.raise();
        }
        else
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to allocate DeferredBuffer/Value/Reduction of "
              << footprint << " bytes for leaf task " << *this << " in "
              << manager->get_name() << " memory because the memory is "
              << "fragmented. This was an unbounded memory pool and there are "
              << "still " << memory_limit
              << " bytes free in the memory but not "
              << "enough are contiguous to allocate the instance. We strongly "
              << "recommend all users put bounds on their dynamic memory usage "
              << "so they can detect if space will be available for task "
              << "execution and if not select an alternative mapping.";
          error.raise();
        }
      }
      task_local_instances[instance] = std::make_pair(unique_event, escaping);
      return instance;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_task_local_instance(
        PhysicalInstance instance, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      // We support multiple OpenMP threads creating buffers in parallel
      AutoLock l_lock(leaf_lock);
      std::map<PhysicalInstance, std::pair<LgEvent, bool>>::iterator finder =
          task_local_instances.find(instance);
      if (finder == task_local_instances.end())
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error
            << "Failed to allocate DeferredBuffer/Value/Reduction in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ") in memory "
            << "because the pool associated with this memory was already "
               "released "
            << "by the task. It is illegal to attempt to perform dynamic "
            << "allocations in a memory pool after it has been released.";
        error.raise();
      }
      std::pair<Memory, bool> key(instance.get_location(), true /*escaping*/);
      std::map<std::pair<Memory, bool>, MemoryPool*>::const_iterator
          pool_finder = memory_pools.find(key);
      if (pool_finder == memory_pools.end())
      {
        // Try again for non-escaping
        key.second = false;
        pool_finder = memory_pools.find(key);
        if (pool_finder == memory_pools.end())
        {
          // This case occurs when the user has taken the unsafe path in
          // the instance creation code above and needs to delete this
          // instance that is not associated with any pool
          MemoryManager* manager =
              runtime->find_memory_manager(instance.get_location());
          manager->free_task_local_instance(instance, precondition);
        }
        else
          pool_finder->second->free_instance(
              instance, precondition, finder->second.first);
      }
      else
        pool_finder->second->free_instance(
            instance, precondition, finder->second.first);
      task_local_instances.erase(finder);
    }

    //--------------------------------------------------------------------------
    size_t LeafContext::query_available_memory(Memory memory, bool escaping)
    //--------------------------------------------------------------------------
    {
      const std::pair<Memory, bool> key(memory, escaping);
      std::map<std::pair<Memory, bool>, MemoryPool*>::const_iterator finder =
          memory_pools.find(key);
      if (finder == memory_pools.end())
        return 0;
      else
        return finder->second->query_available_memory();
    }

    //--------------------------------------------------------------------------
    void LeafContext::release_memory_pool(Memory target)
    //--------------------------------------------------------------------------
    {
      const std::pair<Memory, bool> key(target, true /*escaping*/);
      std::map<std::pair<Memory, bool>, MemoryPool*>::const_iterator finder =
          memory_pools.find(key);
      if (finder != memory_pools.end())
        finder->second->release_pool(get_unique_id());
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_task(
        const void* res, size_t res_size, bool owned,
        PhysicalInstance deferred_result_instance,
        FutureFunctor* callback_functor,
        const Realm::ExternalInstanceResource* resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&),
        const void* metadataptr, size_t metadatasize, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // No local regions or fields permitted in leaf tasks
      if (overhead_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff =
            current - overhead_profiler->previous_profiling_time;
        overhead_profiler->application_time += diff;
      }
      if (Processor::get_executing_processor().exists())
      {
        legion_assert(!effects.exists());
        effects = ApEvent(Processor::get_current_finish_event());
        if (owner_task->is_concurrent())
          runtime->end_concurrent_task(executing_processor);
      }
      // No need to unmap the physical regions, they never had events
      // but still need to ask the physical regions to report their
      // region usage if they had any in the case we're profiling
      if (runtime->profiler != nullptr)
      {
        for (const PhysicalRegion& region : physical_regions)
          region.impl->report_usage(true /*escaped*/);
      }
      TaskContext::end_task(
          res, res_size, owned, deferred_result_instance, callback_functor,
          resource, freefunc, metadataptr, metadatasize, effects);
    }

    //--------------------------------------------------------------------------
    void LeafContext::post_end_task(void)
    //--------------------------------------------------------------------------
    {
      // We don't have any children so we can just record them committed
      owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    RtEvent LeafContext::escape_task_local_instance(
        PhysicalInstance instance, RtEvent safe_effects, size_t num_results,
        PhysicalInstance* results, LgEvent* unique_events,
        const Realm::InstanceLayoutGeneric** layouts, bool redistrict_only)
    //--------------------------------------------------------------------------
    {
      legion_assert(num_results > 0);
      legion_assert((layouts != nullptr) || (num_results == 1));
      if (!memory_pools.empty() && !redistrict_only)
      {
        // See if this is an instance that we made
        std::map<PhysicalInstance, std::pair<LgEvent, bool>>::iterator finder =
            task_local_instances.find(instance);
        if (finder != task_local_instances.end() && finder->second.second)
        {
          // Special case where we can reuse the existing instance because
          // we're escaping this into exactly one other instance with the
          // same unique event result
          if ((layouts == nullptr) && (num_results == 1) &&
              !unique_events[0].exists() &&
              (unique_events[0] == finder->second.first))
            unique_events[0] = finder->second.first;
          // See if this is in a memory for which we have a pool
          const std::pair<Memory, bool> key(
              instance.get_location(), true /*escaping*/);
          std::map<std::pair<Memory, bool>, MemoryPool*>::const_iterator
              pool_finder = memory_pools.find(key);
          if ((pool_finder != memory_pools.end()) &&
              pool_finder->second->contains_instance(instance))
          {
            task_local_instances.erase(finder);
            return pool_finder->second->escape_task_local_instance(
                instance, safe_effects, num_results, results, unique_events,
                layouts, get_unique_id());
          }
        }
      }
      // Otherwise we fall through and do the base case at this point
      return TaskContext::escape_task_local_instance(
          instance, safe_effects, num_results, results, unique_events, layouts,
          redistrict_only);
    }

    //--------------------------------------------------------------------------
    void LeafContext::release_task_local_instances(
        ApEvent effects, RtEvent safe_effects)
    //--------------------------------------------------------------------------
    {
      if (task_local_instances.empty() && memory_pools.empty())
        return;
      if (effects.exists() && !safe_effects.exists())
        safe_effects = Runtime::protect_event(effects);
      for (std::pair<const PhysicalInstance, std::pair<LgEvent, bool>>& it :
           task_local_instances)
      {
        // Check to see if we have a memory pool that contains this in which
        // case we shouldn't actually free up the instance like this
        const std::pair<Memory, bool> key(
            it.first.get_location(), it.second.second);
        std::map<std::pair<Memory, bool>, MemoryPool*>::const_iterator finder =
            memory_pools.find(key);
        if ((finder != memory_pools.end()) &&
            finder->second->contains_instance(it.first))
          continue;
        MemoryManager* manager =
            runtime->find_memory_manager(it.first.get_location());
#ifdef LEGION_MALLOC_INSTANCES
        manager->free_legion_instance(safe_effects, it.first);
#else
        manager->free_task_local_instance(it.first, safe_effects);
#endif
      }
      task_local_instances.clear();
      for (const std::map<std::pair<Memory, bool>, MemoryPool*>::value_type&
               it : memory_pools)
      {
        // Can skip this if it is non-escaping since it would already
        // have been finalized if that were the case
        if (it.first.second)
          it.second->finalize_pool(safe_effects);
        delete it.second;
      }
#ifdef LEGION_DEBUG
      memory_pools.clear();
#endif
    }

    //--------------------------------------------------------------------------
    void LeafContext::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      if (!memory_pools.empty())
      {
        for (const std::map<std::pair<Memory, bool>, MemoryPool*>::value_type&
                 it : memory_pools)
          delete it.second;
#ifdef LEGION_DEBUG
        memory_pools.clear();
#endif
      }
      TaskContext::handle_mispredication();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_lock(Lock l)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal destroy lock performed in leaf task " << get_task_name()
            << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Grant LeafContext::acquire_grant(const std::vector<LockRequest>& requests)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal acquire grant performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return Grant();
    }

    //--------------------------------------------------------------------------
    void LeafContext::release_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal release grant performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal destroy phase barrier performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    DynamicCollective LeafContext::create_dynamic_collective(
        unsigned arrivals, ReductionOpID redop, const void* init_value,
        size_t init_size)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal create dynamic collective performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return DynamicCollective();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_dynamic_collective(DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal destroy dynamic collective performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::arrive_dynamic_collective(
        DynamicCollective dc, const void* buffer, size_t size, unsigned count)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal arrive dynamic collective performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void LeafContext::defer_dynamic_collective_arrival(
        DynamicCollective dc, const Future& future, unsigned count)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal defer dynamic collective performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::get_dynamic_collective_result(
        DynamicCollective dc, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal get dynamic collective performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return Future();
    }

    //--------------------------------------------------------------------------
    DynamicCollective LeafContext::advance_dynamic_collective(
        DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal advance dynamic collective performed in leaf task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return DynamicCollective();
    }

    //--------------------------------------------------------------------------
    TaskPriority LeafContext::get_current_priority(void) const
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    void LeafContext::set_current_priority(TaskPriority priority)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

  }  // namespace Internal
}  // namespace Legion
