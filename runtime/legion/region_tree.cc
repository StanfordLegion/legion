/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "legion.h"
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/region_tree.h"
#include "legion/legion_spy.h"
#include "legion/legion_context.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_analysis.h"
#include "legion/legion_trace.h"
#include "legion/legion_replication.h"
#include "legion/index_space_value.h"

// templates in legion/region_tree.inl are instantiated by region_tree_tmpl.cc

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    //--------------------------------------------------------------------------
    IndirectRecord::IndirectRecord(RegionTreeForest *forest,
                                   const RegionRequirement &req,
                                   const InstanceSet &insts)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *is = forest->get_node(req.region.get_index_space());
      domain_ready = is->get_domain(domain, false/*tight*/);
#ifdef LEGION_SPY
      index_space = req.region.get_index_space();
#endif
      FieldSpaceNode *fs = forest->get_node(req.region.get_field_space());
      std::vector<unsigned> field_indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, field_indexes);
      instances.resize(field_indexes.size());
      Runtime *runtime = forest->runtime;
      if ((runtime->profiler != NULL) || runtime->legion_spy_enabled)
        instance_events.resize(field_indexes.size());
      // For each of the fields in the region requirement
      // (importantly in the order they will be copied)
      // find the corresponding instance and store them 
      // in the indirect record
      for (unsigned fidx = 0; fidx < field_indexes.size(); fidx++)
      {
#ifdef DEBUG_LEGION
        bool found = false;
#endif
        for (unsigned idx = 0; idx < insts.size(); idx++)
        {
          const InstanceRef &ref = insts[idx];
          const FieldMask &mask = ref.get_valid_fields();
          if (!mask.is_set(field_indexes[fidx]))
            continue;
          PhysicalManager *manager = ref.get_physical_manager();
          instances[fidx] = manager->get_instance();
          if (!instance_events.empty())
            instance_events[fidx] = manager->get_unique_event();
#ifdef DEBUG_LEGION
          found = true;
#endif
          break;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void IndirectRecord::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(domain);
      rez.serialize(domain_ready);
      rez.serialize<size_t>(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
        rez.serialize(instances[idx]);
      rez.serialize<size_t>(instance_events.size());
      for (unsigned idx = 0; idx < instance_events.size(); idx++)
        rez.serialize(instance_events[idx]);
#ifdef LEGION_SPY
      rez.serialize(index_space);
#endif
    }

    //--------------------------------------------------------------------------
    void IndirectRecord::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(domain);
      derez.deserialize(domain_ready);
      size_t num_instances;
      derez.deserialize(num_instances);
      instances.resize(num_instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
        derez.deserialize(instances[idx]);
      size_t num_events;
      derez.deserialize(num_events);
      instance_events.resize(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
        derez.deserialize(instance_events[idx]);
#ifdef LEGION_SPY
      derez.deserialize(index_space);
#endif
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
    /////////////////////////////////////////////////////////////
    
    /*static*/ const unsigned RegionTreeForest::MAX_EXPRESSION_FANOUT;

    //--------------------------------------------------------------------------
    RegionTreeForest::RegionTreeForest(Runtime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::RegionTreeForest(const RegionTreeForest &rhs)
      : runtime(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::~RegionTreeForest(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionTreeForest& RegionTreeForest::operator=(const RegionTreeForest &rhs)
    //--------------------------------------------------------------------------
    {
      // should never happen
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_index_space(IndexSpace handle,
                                        const Domain *domain, DistributedID did, 
                                        Provenance *provenance,
                                        CollectiveMapping *mapping,
                                        IndexSpaceExprID expr_id,
                                        ApEvent ready /*=ApEvent::NO_AP_EVENT*/,
                                        RtEvent init /*= RtEvent::NO_RT_EVENT*/)
    //--------------------------------------------------------------------------
    {
      return create_node(handle, domain, true/*is domain*/, NULL/*parent*/, 
                         0/*color*/, did, init, provenance, ready, expr_id,
                         mapping, true/*add root reference*/);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_union_space(IndexSpace handle,
                    DistributedID did, Provenance *provenance,
                    const std::vector<IndexSpace> &sources, 
                    RtEvent initialized, CollectiveMapping *collective_mapping,
                    IndexSpaceExprID expr_id)
    //--------------------------------------------------------------------------
    {
      // Construct the set of index space expressions
      std::set<IndexSpaceExpression*> exprs;
      for (std::vector<IndexSpace>::const_iterator it = 
            sources.begin(); it != sources.end(); it++)
      {
        if (!it->exists())
          continue;
        exprs.insert(get_node(*it));
      }
#ifdef DEBUG_LEGION
      assert(!exprs.empty());
#endif
      IndexSpaceExpression *expr = union_index_spaces(exprs);
      return expr->create_node(handle, did, initialized, provenance,
                               collective_mapping, expr_id);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_intersection_space(
                                        IndexSpace handle, DistributedID did,
                                        Provenance *provenance,
                                        const std::vector<IndexSpace> &sources,
                                        RtEvent initialized, 
                                        CollectiveMapping *collective_mapping,
                                        IndexSpaceExprID expr_id) 
    //--------------------------------------------------------------------------
    {
      // Construct the set of index space expressions
      std::set<IndexSpaceExpression*> exprs;
      for (std::vector<IndexSpace>::const_iterator it = 
            sources.begin(); it != sources.end(); it++)
      {
        if (!it->exists())
          continue;
        exprs.insert(get_node(*it));
      }
#ifdef DEBUG_LEGION
      assert(!exprs.empty());
#endif
      IndexSpaceExpression *expr = intersect_index_spaces(exprs);
      return expr->create_node(handle, did, initialized, provenance,
                               collective_mapping, expr_id);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_difference_space(
                                         IndexSpace handle, DistributedID did,
                                         Provenance *provenance,
                                         IndexSpace left, IndexSpace right, 
                                         RtEvent initialized, 
                                         CollectiveMapping *collective_mapping,
                                         IndexSpaceExprID expr_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(left.exists());
#endif
      IndexSpaceNode *lhs = get_node(left);
      if (!right.exists())
        return lhs->create_node(handle, did, initialized, provenance,
                                collective_mapping);
      IndexSpaceNode *rhs = get_node(right);
      IndexSpaceExpression *expr = subtract_index_spaces(lhs, rhs);
      return expr->create_node(handle, did, initialized, provenance,
                               collective_mapping, expr_id);
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::create_pending_partition(InnerContext *ctx,
                                                       IndexPartition pid,
                                                       IndexSpace parent,
                                                       IndexSpace color_space,
                                                   LegionColor &partition_color,
                                                       PartitionKind part_kind,
                                                       DistributedID did,
                                                       Provenance *provenance,
                                                     CollectiveMapping *mapping,
                                                       RtEvent initialized)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *parent_node = get_node(parent);
      IndexSpaceNode *color_node = get_node(color_space);
      if (partition_color == INVALID_COLOR)
        partition_color = parent_node->generate_color();
      // If we are making this partition on a different node than the
      // owner node of the parent index space then we have to tell that
      // owner node about the existence of this partition
      RtEvent parent_notified;
      const AddressSpaceID parent_owner = parent_node->get_owner_space();
      if ((parent_owner != runtime->address_space) &&
          ((mapping == NULL) || !mapping->contains(parent_owner)) &&
          ((mapping == NULL) || 
           (mapping->find_nearest(parent_owner) == runtime->address_space)))
      {
        RtUserEvent notified_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(pid);
          rez.serialize(parent);
          rez.serialize(partition_color);
          rez.serialize(notified_event);
        }
        runtime->send_index_partition_notification(parent_owner, rez);
        parent_notified = notified_event;
      }
      if ((part_kind == LEGION_COMPUTE_KIND) || 
          (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        // Use 1 if we know it's complete, 0 if it's not, 
        // otherwise -1 since we don't know
        const int complete = (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ? 1 :
                         (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND) ? 0 : -1;
        create_node(pid, parent_node, color_node, partition_color,
            complete, did, provenance, initialized, mapping);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_index_partition(parent.id, pid.id, -1/*unknown*/,
              complete, partition_color, runtime->address_space, 
              (provenance == NULL) ? NULL : provenance->human_str());
      }
      else
      {
        const bool disjoint = (part_kind == LEGION_DISJOINT_KIND) || 
                              (part_kind == LEGION_DISJOINT_COMPLETE_KIND) ||
                              (part_kind == LEGION_DISJOINT_INCOMPLETE_KIND);
        // Use 1 if we know it's complete, 0 if it's not, 
        // otherwise -1 since we don't know
        const int complete = ((part_kind == LEGION_DISJOINT_COMPLETE_KIND) ||
                              (part_kind == LEGION_ALIASED_COMPLETE_KIND)) ? 1 :
                             ((part_kind == LEGION_DISJOINT_INCOMPLETE_KIND) ||
                        (part_kind == LEGION_ALIASED_INCOMPLETE_KIND)) ? 0 : -1;
        create_node(pid, parent_node, color_node, partition_color,
            disjoint, complete, did, provenance, initialized, mapping);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_index_partition(parent.id, pid.id, disjoint ? 1 : 0,
              complete, partition_color, runtime->address_space,
              (provenance == NULL) ? NULL : provenance->human_str());
      }
      ctx->register_index_partition_creation(pid);
      return parent_notified;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_pending_cross_product(InnerContext *ctx,
                                                 IndexPartition handle1,
                                                 IndexPartition handle2,
                             std::map<IndexSpace,IndexPartition> &user_handles,
                                                 PartitionKind kind,
                                                 Provenance *provenance,
                                                 LegionColor &part_color,
                                                 std::set<RtEvent> &safe_events,
                                                 ShardID local_shard,
                                              const ShardMapping *shard_mapping,
                                   ValueBroadcast<LegionColor> *color_broadcast)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *base = get_node(handle1);
      IndexPartNode *source = get_node(handle2);
      // If we're supposed to compute this, but we already know that
      // the source is disjoint then we can also conlclude the the 
      // resulting partitions will also be disjoint under intersection
      if (((kind == LEGION_COMPUTE_KIND) || 
           (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
           (kind == LEGION_COMPUTE_INCOMPLETE_KIND)) && 
          source->is_disjoint(true/*from app*/))
      {
        if (kind == LEGION_COMPUTE_KIND)
          kind = LEGION_DISJOINT_KIND;
        else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
          kind = LEGION_DISJOINT_COMPLETE_KIND;
        else
          kind = LEGION_DISJOINT_INCOMPLETE_KIND;
      }
      // If the source dominates the base then we know that all the
      // partitions that we are about to make will be complete
      if (((kind == LEGION_DISJOINT_KIND) || (kind == LEGION_ALIASED_KIND) || 
            (kind == LEGION_COMPUTE_KIND)) && 
          source->is_complete() && source->parent->dominates(base->parent))
      {
        if (kind == LEGION_DISJOINT_KIND)
          kind = LEGION_DISJOINT_COMPLETE_KIND;
        else if (kind == LEGION_ALIASED_KIND)
          kind = LEGION_ALIASED_COMPLETE_KIND;
        else
          kind = LEGION_COMPUTE_COMPLETE_KIND;
      }
      // If we haven't been given a color yet, we need to find
      // one that will be valid for all the child partitions
      // We don't have any way to atomically test and set a color
      // for all the partitions we're about to make so instead
      // we'll do this iteratively until we succeed, which 
      // hopefully will not be too many iterations
      LegionColor lower_bound = 0;
      std::set<LegionColor> existing_colors;
      std::vector<IndexSpaceNode*> children_nodes;
      while (part_color == INVALID_COLOR)
      {
        // If this is the first time through populate the existing colors
        if ((lower_bound == 0) && existing_colors.empty())
        {
          for (ColorSpaceIterator itr(base); itr; itr++)
          {
            IndexSpaceNode *child_node = base->get_child(*itr);
            children_nodes.push_back(child_node);
            std::vector<LegionColor> colors;
            LegionColor bound = child_node->get_colors(colors);
            if (!colors.empty())
              existing_colors.insert(colors.begin(), colors.end());
            if (bound > lower_bound)
              lower_bound = bound;
          }
        }
        // Prune out any colors below the lower bound, we know they are never
        // going to be something that we can use across all the children
        while (!existing_colors.empty())
        {
          std::set<LegionColor>::iterator next = existing_colors.begin();
          if ((*next) <= lower_bound)
          {
            if ((*next) == lower_bound)
              lower_bound++;
            existing_colors.erase(next);
          }
          else
            break;
        }
        // Find the next available color
        part_color = lower_bound++;
#ifdef DEBUG_LEGION
        assert(part_color != INVALID_COLOR);
#endif
        // Now confirm that we can reserve this color in all our subregions
        for (std::vector<IndexSpaceNode*>::const_iterator it =
              children_nodes.begin(); it != children_nodes.end(); it++)
        {
          LegionColor result = (*it)->generate_color(part_color);  
          if (result == part_color)
            continue;
          // If we failed we need to remove all the failed colors
          for (std::vector<IndexSpaceNode*>::const_iterator it2 = 
                children_nodes.begin(); it2 != it; it2++)
            (*it)->release_color(part_color);
          // Record that this is an existing color to skip
          existing_colors.insert(part_color);
          part_color = INVALID_COLOR;
          break;
        }
      }
      if (color_broadcast != NULL)
        color_broadcast->broadcast(part_color);
      // Iterate over all our sub-regions and generate partitions
      if (shard_mapping == NULL)
      {
        for (ColorSpaceIterator itr(base); itr; itr++)
        {
          IndexSpaceNode *child_node = base->get_child(*itr);
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag());
          DistributedID did =
            runtime->get_available_distributed_id();
          const RtEvent safe =
            create_pending_partition(ctx, pid, child_node->handle,
                                     source->color_space->handle,
                                     part_color, kind, did, provenance);
          // If the user requested the handle for this point return it
          user_handles[child_node->handle] = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
      }
      else if (((LegionColor)shard_mapping->size()) <= base->total_children)
      {
        // There are more subregions than shards so we can shard the
        // children over all the shards to make the partitions
        for (ColorSpaceIterator itr(base, local_shard, 
              shard_mapping->size()); itr; itr++)
        {
          IndexSpaceNode *child_node = base->get_child(*itr);
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag());
          DistributedID did =
            runtime->get_available_distributed_id();
          const RtEvent safe =
            create_pending_partition(ctx, pid, child_node->handle,
                                     source->color_space->handle,
                                     part_color, kind, did, provenance);
          // If the user requested the handle for this point return it
          user_handles[child_node->handle] = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
      }
      else
      {
        // There are fewer subregions than shards, so we can actually
        // have multiple shards collaborating to create each partition 
        // Round-robin the shards over the children partitions to compute
        const unsigned color_index = local_shard % base->total_children; 
        LegionColor child_color = color_index;
        if (base->total_children < base->max_linearized_color)
        {
          unsigned index = 0;
          for (ColorSpaceIterator itr(base); itr; itr++, index++)
          {
            if (index != color_index)
              continue;
            child_color = *itr;
            break;
          }
        }
        IndexSpaceNode *child_node = base->get_child(child_color);
        // Figure out how many shards are participating on this child 
        // and what their address spaces are so we can make a collective
        // mapping for the new partition. Also tracke if we're the first
        // shard on this address space.
        bool first_local_shard = true;
        std::vector<AddressSpaceID> child_spaces;
        for (ShardID shard = color_index; 
              shard < shard_mapping->size(); shard += base->total_children)
        {
          const AddressSpaceID space = (*shard_mapping)[shard];
          if (std::binary_search(child_spaces.begin(),child_spaces.end(),space))
          {
            if (shard == local_shard)
              first_local_shard = false;
            continue;
          }
          child_spaces.push_back(space);
          std::sort(child_spaces.begin(), child_spaces.end());
        }
#ifdef DEBUG_LEGION
        assert(!child_spaces.empty());
        ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(ctx);
        assert(repl_ctx != NULL);
#else
        ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(ctx);
#endif
        CrossProductExchange exchange(repl_ctx, COLLECTIVE_LOC_50);
        if (first_local_shard)
        {
          // If we're the first space for this child then make the 
          // distributed ID and index partition name for the new partition
          // and then exchange in the collective
          if (child_spaces.front() == runtime->address_space)
          {
            IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag());
            exchange.exchange_ids(child_color,
                runtime->get_available_distributed_id(), pid);
            user_handles[child_node->handle] = pid;
          }
          else
            exchange.perform_collective_async();
          DistributedID child_did = 0;
          IndexPartition child_pid = IndexPartition::NO_PART;
          exchange.sync_child_ids(child_color, child_did, child_pid);
          const RtEvent safe = create_pending_partition(ctx, child_pid, 
              child_node->handle, source->color_space->handle,
              part_color, kind, child_did, provenance,
              new CollectiveMapping(child_spaces,
                runtime->legion_collective_radix));
          if (safe.exists())
            safe_events.insert(safe);
        }
        else
        {
          // Still need to participate in the collective exchange
          exchange.perform_collective_sync();
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_space(IndexSpace handle,
                                              AddressSpaceID source,
                                              std::set<RtEvent> &applied,
                                              const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      if (node->invalidate_root(source, applied, mapping))
        delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition handle,
                                                   std::set<RtEvent> &applied,
                                               const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        IndexPartNode::get_owner_space(handle, runtime);
      if (mapping != NULL)
      {
        if (mapping->contains(owner_space))
        {
          // If we're the owner space node then we do the removal
          if (owner_space == runtime->address_space)
          {
            IndexPartNode *node = get_node(handle);
            if (node->remove_base_valid_ref(APPLICATION_REF))
              delete node;
          }
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == runtime->address_space)
            runtime->send_index_partition_destruction(handle, 
                                        owner_space, applied);
        }
      }
      else
      {
        if (owner_space == runtime->address_space)
        {
          IndexPartNode *node = get_node(handle);
          if (node->remove_base_valid_ref(APPLICATION_REF))
            delete node;
        }
        else
          runtime->send_index_partition_destruction(handle,owner_space,applied);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_equal_partition(Operation *op,
                                                     IndexPartition pid,
                                                     size_t granularity)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->create_equal_children(op, granularity); 
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_weights(Operation *op,
                                                       IndexPartition pid,
                               const std::map<DomainPoint,FutureImpl*> &weights,
                                                       size_t granularity)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->create_by_weights(op, weights, granularity); 
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_union(Operation *op,
                                                        IndexPartition pid,
                                                        IndexPartition handle1,
                                                        IndexPartition handle2)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node1 = get_node(handle1);
      IndexPartNode *node2 = get_node(handle2);
      return new_part->create_by_union(op, node1, node2);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_intersection(Operation *op,
                                                         IndexPartition pid,
                                                         IndexPartition handle1,
                                                         IndexPartition handle2)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node1 = get_node(handle1);
      IndexPartNode *node2 = get_node(handle2);
      return new_part->create_by_intersection(op, node1, node2); 
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_intersection(Operation *op,
                                                           IndexPartition pid,
                                                           IndexPartition part,
                                                           const bool dominates)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node = get_node(part);
      return new_part->create_by_intersection(op, node, dominates);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_difference(Operation *op,
                                                       IndexPartition pid,
                                                       IndexPartition handle1,
                                                       IndexPartition handle2)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node1 = get_node(handle1);
      IndexPartNode *node2 = get_node(handle2);
      return new_part->create_by_difference(op, node1, node2); 
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_restriction(
                                                        IndexPartition pid,
                                                        const void *transform,
                                                        const void *extent)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->create_by_restriction(transform, extent);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_domain(Operation *op,
                                                    IndexPartition pid,
                              const std::map<DomainPoint,FutureImpl*> &weights,
                                                const Domain &future_map_domain,
                                                    bool perform_intersections)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->parent->create_by_domain(op, new_part, weights,
                            future_map_domain, perform_intersections);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_cross_product_partitions(Operation *op,
                                                         IndexPartition base,
                                                         IndexPartition source,
                                                         LegionColor part_color,
                                                         ShardID local_shard,
                                              const ShardMapping *shard_mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *base_node = get_node(base);
      IndexPartNode *source_node = get_node(source);
      std::set<ApEvent> ready_events;
      if (shard_mapping == NULL)
      {
        for (ColorSpaceIterator itr(base_node); itr; itr++)
        {
          IndexSpaceNode *child_node = base_node->get_child(*itr);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready = 
            child_node->create_by_intersection(op, part_node, source_node);
          ready_events.insert(ready);
        }
      }
      else if (((LegionColor)shard_mapping->size()) <= 
                  base_node->total_children)
      {
        for (ColorSpaceIterator itr(base_node, local_shard,
              shard_mapping->size()); itr; itr++)
        {
          IndexSpaceNode *child_node = base_node->get_child(*itr);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready = 
            child_node->create_by_intersection(op, part_node, source_node);
          ready_events.insert(ready);
        }
      }
      else
      {
        const unsigned color_index = local_shard % base_node->total_children; 
        // See if we're the first local shard on this address space
        bool first_local_shard = true;
        for (ShardID shard = color_index;
              shard < shard_mapping->size(); shard += base_node->total_children)
        {
          const AddressSpaceID space = (*shard_mapping)[shard];
          if (space != runtime->address_space)
            continue;
          first_local_shard = (shard == local_shard);
          break;
        }
        if (first_local_shard)
        {
          LegionColor child_color = color_index;
          if (base_node->total_children < base_node->max_linearized_color)
          {
            unsigned index = 0;
            for (ColorSpaceIterator itr(base_node); itr; itr++, index++)
            {
              if (index != color_index)
                continue;
              child_color = *itr;
              break;
            }
          }
          IndexSpaceNode *child_node = base_node->get_child(child_color);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready = 
            child_node->create_by_intersection(op, part_node, source_node);
          ready_events.insert(ready);
        }
      }
      return Runtime::merge_events(NULL, ready_events);
    } 

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_field(Operation *op, 
                                                        FieldID fid,
                                                        IndexPartition pending,
                             const std::vector<FieldDataDescriptor> &instances,
                                   std::vector<DeppartResult> *results,
                                                        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      return partition->parent->create_by_field(op, fid, partition, instances,
                                                results, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_image(Operation *op,
                                                        FieldID fid,
                                                        IndexPartition pending,
                                                        IndexPartition proj,
                                    std::vector<FieldDataDescriptor> &instances,
                                                        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_image(op, fid, partition, projection,
                                                instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_image_range(Operation *op,
                                                      FieldID fid,
                                                      IndexPartition pending,
                                                      IndexPartition proj,
                                  std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_image_range(op, fid, partition,
                                  projection, instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_preimage(Operation *op,
                                                      FieldID fid,
                                                      IndexPartition pending,
                                                      IndexPartition proj,
                             const std::vector<FieldDataDescriptor> &instances,
                             const std::map<DomainPoint,Domain> *remote_targets,
                                    std::vector<DeppartResult> *results,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_preimage(op, fid, partition,
          projection, instances, remote_targets, results, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_preimage_range(Operation *op,
                                                      FieldID fid,
                                                      IndexPartition pending,
                                                      IndexPartition proj,
                             const std::vector<FieldDataDescriptor> &instances,
                             const std::map<DomainPoint,Domain> *remote_targets,
                                    std::vector<DeppartResult> *results,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_preimage_range(op, fid, partition,
          projection, instances, remote_targets, results, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_association(Operation *op, FieldID fid,
                                                 IndexSpace dom, IndexSpace ran,
                              const std::vector<FieldDataDescriptor> &instances,
                                                 ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *domain = get_node(dom);
      IndexSpaceNode *range = get_node(ran);
      return domain->create_association(op,fid,range,instances,instances_ready);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_coordinate_size(IndexSpace handle, bool range)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->get_coordinate_size(range);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op, 
       IndexSpace target, const std::vector<IndexSpace> &handles, bool is_union)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      // See if we own this child or not
      if (!child_node->is_owner() && ((child_node->collective_mapping == NULL)
         || !child_node->collective_mapping->contains(child_node->local_space)))
        return ApEvent::NO_AP_EVENT;
      return child_node->compute_pending_space(op, handles, is_union);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op, 
                        IndexSpace target, IndexPartition handle, bool is_union)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      // See if we own this child or not
      if (!child_node->is_owner() && ((child_node->collective_mapping == NULL)
         || !child_node->collective_mapping->contains(child_node->local_space)))
        return ApEvent::NO_AP_EVENT;
      return child_node->compute_pending_space(op, handle, is_union);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op,
                                         IndexSpace target, IndexSpace initial,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      // See if we own this child or not
      if (!child_node->is_owner() && ((child_node->collective_mapping == NULL)
         || !child_node->collective_mapping->contains(child_node->local_space)))
        return ApEvent::NO_AP_EVENT;
      return child_node->compute_pending_difference(op, initial, handles);
    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_index_partition(IndexSpace parent,
                                                         Color color)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *parent_node = get_node(parent);
      IndexPartNode *child_node = parent_node->get_child(color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_index_subspace(IndexPartition parent,
                                              const void *realm_color,
                                              TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *parent_node = get_node(parent);
      return parent_node->color_space->contains_point(realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_index_subspace(IndexPartition parent,
                                                    const void *realm_color,
                                                    TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *parent_node = get_node(parent);
      LegionColor child_color = 
        parent_node->color_space->linearize_color(realm_color, type_tag);
      IndexSpaceNode *child_node = parent_node->get_child(child_color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::instantiate_subspace(IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *parent_node = get_node(parent);
      LegionColor child_color = 
        parent_node->color_space->linearize_color(realm_color, type_tag);
      IndexSpaceNode *child_node = parent_node->get_child(child_color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_index_space_domain(IndexSpace handle,
                                               void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      node->get_index_space_domain(realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_index_partition_color_space(
                                                               IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(p);
      return node->color_space->handle;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_index_space_partition_colors(IndexSpace sp,
                                                        std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(sp);
      std::vector<LegionColor> temp_colors;
      node->get_colors(temp_colors); 
      for (std::vector<LegionColor>::const_iterator it = temp_colors.begin();
            it != temp_colors.end(); it++)
        colors.insert(*it);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_index_space_color(IndexSpace handle,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      if (node->parent == NULL)
      {
        // We know the answer here
        if (type_tag != NT_TemplateHelper::encode_tag<1,coord_t>())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
            "Dynamic type mismatch in 'get_index_space_color'")
        Realm::Point<1,coord_t> *color = 
          (Realm::Point<1,coord_t>*)realm_color;
        *color = Realm::Point<1,coord_t>(0);
        return;
      }
      // Otherwise we can get the color for the partition color space
      IndexSpaceNode *color_space = node->parent->color_space;
      color_space->delinearize_color(node->color, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::get_index_partition_color(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      return node->color;
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_parent_index_space(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      return (node->parent != NULL);
    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_parent_index_partition(
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      if (node->parent == NULL)
        REPORT_LEGION_ERROR(ERROR_PARENT_INDEX_PARTITION_REQUESTED,
          "Parent index partition requested for "
                            "index space %x with no parent. Use "
                            "has_parent_index_partition to check "
                            "before requesting a parent.", handle.id)
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    unsigned RegionTreeForest::get_index_space_depth(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->depth;
    }

    //--------------------------------------------------------------------------
    unsigned RegionTreeForest::get_index_partition_depth(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->depth;
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_domain_volume(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      return node->get_volume();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::find_domain(IndexSpace handle, Domain &launch_domain)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      node->get_domain(launch_domain);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::validate_slicing(IndexSpace input_space,
                                    const std::vector<IndexSpace> &slice_spaces,
                                    MultiTask *task, MapperManager *mapper)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(input_space);
      node->validate_slicing(slice_spaces, task, mapper);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::log_launch_space(IndexSpace handle, UniqueID op_id)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      node->log_launch_space(op_id);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_index_partition_disjoint(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(p);
      return node->is_disjoint(true/*app query*/);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_index_partition_complete(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(p);
      return node->is_complete(true/*app query*/);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *parent_node = get_node(parent);
      return parent_node->has_color(color);
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_field_space(FieldSpace handle,
                                                    DistributedID did,
                                                    Provenance *provenance,
                                                    CollectiveMapping *mapping,
                                                    RtEvent initialized)
    //--------------------------------------------------------------------------
    {
      return create_node(handle, did, initialized, provenance, mapping);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace handle,
                                               std::set<RtEvent> &applied, 
                                               const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        FieldSpaceNode::get_owner_space(handle, runtime);
      if (mapping != NULL)
      {
        if (mapping->contains(owner_space))
        {
          // If we're the owner space node then we do the removal
          if (owner_space == runtime->address_space)
          {
            FieldSpaceNode *node = get_node(handle);
            if (node->remove_base_gc_ref(APPLICATION_REF))
              delete node;
          }
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == runtime->address_space)
            runtime->send_field_space_destruction(handle, owner_space, applied);
        }
      }
      else
      {
        if (owner_space == runtime->address_space)
        {
          FieldSpaceNode *node = get_node(handle);
          if (node->remove_base_gc_ref(APPLICATION_REF))
            delete node;
        }
        else
          runtime->send_field_space_destruction(handle, owner_space, applied);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::allocate_field(FieldSpace handle, 
                                             size_t field_size, FieldID fid, 
                                             CustomSerdezID serdez_id,
                                             Provenance *provenance,
                                             bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_field(fid, field_size, serdez_id,
                                           provenance, sharded_non_owner);
      return ready;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::allocate_field(FieldSpace handle,
          ApEvent size_ready, FieldID fid, CustomSerdezID serdez_id,
          Provenance *provenance, RtEvent &precondition, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      precondition = node->allocate_field(fid, size_ready, serdez_id,
                                          provenance, sharded_non_owner);
      return node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_field(FieldSpace handle, FieldID fid,
                                      std::set<RtEvent> &applied,
                                      bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_field(fid, runtime->address_space, applied, sharded_non_owner);
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::allocate_fields(FieldSpace handle, 
                                             const std::vector<size_t> &sizes,
                                             const std::vector<FieldID> &fields,
                                             CustomSerdezID serdez_id,
                                             Provenance *provenance,
                                             bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sizes.size() == fields.size());
#endif
      // We know that none of these field allocations are local
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_fields(sizes, fields, serdez_id,
                                            provenance, sharded_non_owner);
      return ready;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::allocate_fields(FieldSpace handle, 
                                           ApEvent sizes_ready,
                                           const std::vector<FieldID> &fields,
                                           CustomSerdezID serdez_id,
                                           Provenance *provenance,
                                           RtEvent &precondition,
                                           bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      // We know that none of these field allocations are local
      FieldSpaceNode *node = get_node(handle);
      precondition = node->allocate_fields(sizes_ready, fields, serdez_id,
                                           provenance, sharded_non_owner);
      return node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_fields(FieldSpace handle,
                                       const std::vector<FieldID> &to_free,
                                       std::set<RtEvent> &applied,
                                       bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_fields(to_free, runtime->address_space, applied, 
                        sharded_non_owner);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_field_indexes(FieldSpace handle,
                                        const std::vector<FieldID> &to_free,
                                        RtEvent freed_event, 
                                        bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_field_indexes(to_free, freed_event, sharded_non_owner);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::allocate_local_fields(FieldSpace handle,
                                      const std::vector<FieldID> &fields,
                                      const std::vector<size_t> &sizes,
                                      CustomSerdezID serdez_id, 
                                      const std::set<unsigned> &current_indexes,
                                            std::vector<unsigned> &new_indexes,
                                            Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      return node->allocate_local_fields(fields, sizes, serdez_id,
                         current_indexes, new_indexes, provenance);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_local_fields(FieldSpace handle,
                                           const std::vector<FieldID> &to_free,
                                           const std::vector<unsigned> &indexes,
                                           const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_local_fields(to_free, indexes, mapping);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::update_local_fields(FieldSpace handle,
                                  const std::vector<FieldID> &fields,
                                  const std::vector<size_t> &sizes,
                                  const std::vector<CustomSerdezID> &serdez_ids,
                                  const std::vector<unsigned> &indexes,
                                  Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->update_local_fields(fields, sizes, serdez_ids, indexes, provenance);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_local_fields(FieldSpace handle,
                                          const std::vector<FieldID> &to_remove)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->remove_local_fields(to_remove);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_field_size(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      if (!node->has_field(fid))
        REPORT_LEGION_ERROR(ERROR_FIELD_SPACE_HAS_NO_FIELD,
          "FieldSpace %x has no field %d", handle.id, fid)
      return node->get_field_size(fid);
    }

    //--------------------------------------------------------------------------
    CustomSerdezID RegionTreeForest::get_field_serdez(FieldSpace handle,
                                                      FieldID fid)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      if (!node->has_field(fid))
        REPORT_LEGION_ERROR(ERROR_FIELD_SPACE_HAS_NO_FIELD,
          "FieldSpace %x has no field %d", handle.id, fid)
      return node->get_field_serdez(fid);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_field_space_fields(FieldSpace handle,
                                                  std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->get_all_fields(fields);
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::create_logical_region(LogicalRegion handle,
                                                     DistributedID did,
                                                     Provenance *provenance,
                                                     CollectiveMapping *mapping,
                                                     RtEvent initialized)
    //--------------------------------------------------------------------------
    {
      return create_node(handle, NULL/*parent*/, initialized, did,
                         provenance, mapping);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_logical_region(LogicalRegion handle,
                                               std::set<RtEvent> &applied,
                                               const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        RegionNode::get_owner_space(handle.get_tree_id(), runtime);
      if (mapping != NULL)
      {
        if (mapping->contains(owner_space))
        {
          // If we're the owner space node then we do the removal
          if (owner_space == runtime->address_space)
          {
            RegionNode *node = get_node(handle);
            if (node->remove_base_gc_ref(APPLICATION_REF))
              delete node;
          }
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == runtime->address_space)
            runtime->send_logical_region_destruction(handle, 
                                      owner_space, applied);
        }
      }
      else
      {
        if (owner_space == runtime->address_space)
        {
          RegionNode *node = get_node(handle);
          if (node->remove_base_gc_ref(APPLICATION_REF))
            delete node;
        }
        else
          runtime->send_logical_region_destruction(handle, owner_space,applied);
      }
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_logical_partition(
                                    LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalPartition(parent.tree_id, handle, parent.field_space);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_logical_partition_by_color(
                                                  LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      RegionNode *parent_node = get_node(parent);
      IndexPartNode *index_node = parent_node->row_source->get_child(c);
      LogicalPartition result(parent.tree_id, index_node->handle, 
                              parent.field_space);
      return result;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_logical_partition_by_color(LogicalRegion parent,
                                                          Color color)
    //--------------------------------------------------------------------------
    {
      RegionNode *parent_node = get_node(parent);
      return parent_node->has_color(color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_logical_partition_by_tree(
                      IndexPartition handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalPartition(tid, handle, space);
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_logical_subregion(
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalRegion(parent.tree_id, handle, parent.field_space);
    }
    
    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_logical_subregion_by_color(
             LogicalPartition parent, const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      PartitionNode *parent_node = get_node(parent);
      IndexSpaceNode *color_space = parent_node->row_source->color_space;
      LegionColor color = color_space->linearize_color(realm_color, type_tag);
      if (!color_space->contains_point(realm_color, type_tag))
        REPORT_LEGION_ERROR(ERROR_INVALID_INDEX_SPACE_COLOR,
                            "Invalid color space color for child %lld of "
                            "logical partition (%d,%d,%d)", color,
                            parent.index_partition.id, parent.field_space.id,
                            parent.tree_id)
      IndexSpaceNode *index_node = parent_node->row_source->get_child(color);
      LogicalRegion result(parent.tree_id, index_node->handle,
                           parent.field_space);
      return result;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_logical_subregion_by_color(
             LogicalPartition parent, const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      PartitionNode *parent_node = get_node(parent);
      IndexSpaceNode *color_space = parent_node->row_source->color_space;
      return color_space->contains_point(realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_logical_subregion_by_tree(
                          IndexSpace handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalRegion(tid, handle, space);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_logical_region_color(LogicalRegion handle,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle.get_index_space());
      if (node->parent == NULL)
      {
        // We know the answer here
        if (type_tag != NT_TemplateHelper::encode_tag<1,coord_t>())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
            "Dynamic type mismatch in 'get_logical_region_color'")
        Realm::Point<1,coord_t> *color = 
          (Realm::Point<1,coord_t>*)realm_color;
        *color = Realm::Point<1,coord_t>(0);
        return;
      }
      // Otherwise we can get the color for the partition color space
      IndexSpaceNode *color_space = node->parent->color_space;
      color_space->delinearize_color(node->color, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::get_logical_partition_color(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      PartitionNode *node = get_node(handle);
      return node->row_source->color;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_parent_logical_region(
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      PartitionNode *node = get_node(handle);
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      return (node->parent != NULL);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_parent_logical_partition(
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      if (node->parent == NULL)
        REPORT_LEGION_ERROR(ERROR_PARENT_LOGICAL_PARTITION_REQUESTED,
          "Parent logical partition requested for "
                            "logical region (%x,%x,%d) with no parent. "
                            "Use has_parent_logical_partition to check "
                            "before requesting a parent.", 
                            handle.index_space.id,
                            handle.field_space.id,
                            handle.tree_id)
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_domain_volume(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      return node->row_source->get_volume();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_dependence_analysis(
                                        Operation *op, unsigned idx,
                                        const RegionRequirement &req,
                                        const ProjectionInfo &proj_info,
                                        LogicalAnalysis &logical_analysis)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_LOGICAL_ANALYSIS_CALL);
      // If this is a NO_ACCESS, then we'll have no dependences so we're done
      if (IS_NO_ACCESS(req))
        return;

      ProjectionType htype = req.handle_type;
      bool is_ispace_htype = (htype == LEGION_SINGULAR_PROJECTION ||
                              htype == LEGION_REGION_PROJECTION);
#ifdef DEBUG_LEGION
      assert(is_ispace_htype || htype == LEGION_PARTITION_PROJECTION);
#endif
      IndexTreeNode *child_node = is_ispace_htype ?
        get_node(req.region.get_index_space()) :
        (IndexTreeNode *)get_node(req.partition.get_index_partition());

      RegionNode *parent_node = get_node(req.parent);

      RegionTreePath path;
      initialize_path(child_node, parent_node->row_source, path);

      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      LogicalTraceInfo trace_info(op, idx, req, user_mask);
      if (trace_info.skip_analysis)
        return;
      // Then compute the logical user
      ProjectionSummary *shard_proj = NULL;
      if (proj_info.is_projecting())
      {
#ifndef POINT_WISE_LOGICAL_ANALYSIS
        if(proj_info.is_sharding()) {
#endif
          // If we're doing a projection in a control replicated context then
          // we need to compute the shard projection up front since it might
          // involve a collective if we don't hit in the cache and we want
          // that to appear nice and deterministic
          RegionTreeNode *destination = 
            (req.handle_type == LEGION_PARTITION_PROJECTION) ?
            static_cast<RegionTreeNode*>(get_node(req.partition)) :
            static_cast<RegionTreeNode*>(get_node(req.region));
          shard_proj = destination->compute_projection_summary(op, idx, req,
                                                logical_analysis, proj_info);
#ifdef POINT_WISE_LOGICAL_ANALYSIS
          if(!shard_proj->is_disjoint() || !shard_proj->can_perform_name_based_self_analysis()) {
            logical_analysis.bail_point_wise_analysis = true;
          }
#endif
#ifndef POINT_WISE_LOGICAL_ANALYSIS
        }
#endif
      }

      LogicalUser *user = new LogicalUser(op, idx, RegionUsage(req),
          shard_proj, (op->get_must_epoch_op() == NULL) ? UINT_MAX :
          op->get_must_epoch_op()->find_operation_index(
            op, op->get_generation()));
      user->add_reference();
#ifdef DEBUG_LEGION
      InnerContext *context = op->get_context();
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     context->get_logical_tree_context(),
                                     true/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Finally do the traversal, note that we don't need to hold the
      // context lock since the runtime guarantees that all dependence
      // analysis for a single context are performed in order
      {
        FieldMask unopened_mask = user_mask;
        FieldMask refinement_mask;
        // We disallow refinements for operations that are part of
        // a must epoch launch because refinements are too hard to 
        // implement correctly in that case
        // We also don't try to update refinements if we're doing a reset
        // operation since that is an internal kind of operation
        if ((op->get_must_epoch_op() == NULL) &&
            (op->get_operation_kind() != Operation::RESET_OP_KIND))
          refinement_mask = user_mask;
        FieldMaskSet<RefinementOp,UNTRACKED_ALLOC,true> refinements;
        parent_node->register_logical_user(req.parent, *user, path,
             trace_info, proj_info, user_mask, unopened_mask,
             refinement_mask, logical_analysis, refinements, true/*root*/);
      }
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     context->get_logical_tree_context(),
                                     false/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      if (user->remove_reference())
        delete user;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::find_open_complete_partitions(Operation *op,
                                                         unsigned idx,
                                                  const RegionRequirement &req,
                                      std::vector<LogicalPartition> &partitions)
    //--------------------------------------------------------------------------
    {
      TaskContext *context = op->get_context();
      ContextID ctx = context->get_logical_tree_context(); 
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *region_node = get_node(req.region);
      FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
      region_node->find_open_complete_partitions(ctx, user_mask, partitions);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_versioning_analysis(Operation *op,
                     unsigned index, const RegionRequirement &req, 
                     VersionInfo &version_info, std::set<RtEvent> &ready_events,
                     RtEvent *output_region_ready, bool collective_rendezvous)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_VERSIONING_ANALYSIS_CALL);
      if (IS_NO_ACCESS(req))
        return;
      InnerContext *context = op->find_physical_context(index);
      ContextID ctx = context->get_physical_tree_context(); 
#ifdef DEBUG_LEGION
      assert((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
      ((req.handle_type == LEGION_REGION_PROJECTION) && (req.projection == 0)));
#endif
      RegionNode *region_node = get_node(req.region);
      FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
      region_node->perform_versioning_analysis(ctx, context,
          &version_info, user_mask, op, index, op->find_parent_index(index),
          ready_events, output_region_ready, collective_rendezvous);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_current_context(ContextID ctx,
                      const RegionRequirement &req, bool filter_specific_fields)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_INVALIDATE_CONTEXT_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *region_node = get_node(req.region);
      if (filter_specific_fields)
      {
        FieldMask user_mask =
          region_node->column_source->get_field_mask(req.privilege_fields);
        DeletionInvalidator invalidator(ctx, user_mask);
        region_node->visit_node(&invalidator);
      }
      else
      {
        CurrentInvalidator invalidator(ctx);
        region_node->visit_node(&invalidator);
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::match_instance_fields(const RegionRequirement &req1,
                                                 const RegionRequirement &req2,
                                                 const InstanceSet &inst1,
                                                 const InstanceSet &inst2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req1.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(req2.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(req1.region.field_space == req2.region.field_space);
#endif
      // We only need to check the fields shared by both region requirements
      std::set<FieldID> intersection_fields;
      if (req1.privilege_fields.size() <= req2.privilege_fields.size())
      {
        for (std::set<FieldID>::const_iterator it = 
              req1.privilege_fields.begin(); it != 
              req1.privilege_fields.end(); it++)
        {
          if (req2.privilege_fields.find(*it) != req2.privilege_fields.end())
            intersection_fields.insert(*it);
        }
      }
      else
      {
        for (std::set<FieldID>::const_iterator it = 
              req2.privilege_fields.begin(); it != 
              req2.privilege_fields.end(); it++)
        {
          if (req1.privilege_fields.find(*it) != req1.privilege_fields.end())
            intersection_fields.insert(*it);
        }
      }
      FieldSpaceNode *node = get_node(req1.region.field_space);
      FieldMask intersection_mask = node->get_field_mask(intersection_fields);
      for (unsigned idx = 0; idx < inst1.size(); idx++)
      {
        const InstanceRef &ref = inst1[idx];
        if (ref.is_virtual_ref())
          continue;
        FieldMask overlap = intersection_mask & ref.get_valid_fields();
        if (!overlap)
          continue;
        InstanceManager *manager = ref.get_manager();
        for (unsigned idx2 = 0; idx2 < inst2.size(); idx2++)
        {
          const InstanceRef &other = inst2[idx2];
          if (other.is_virtual_ref())
            continue;
          // If they are not the same instance we can keep going
          if (manager != other.get_manager())
            continue;
          // There is only one of these in the set
          if (!(overlap - other.get_valid_fields()))
          {
            // Dominated all the fields, so we can remove them
            intersection_mask -= overlap;
          }
          else // We didn't dominate all the fields, so no good
              return false;
          break;
        }
        // If we've satisfied all the fields then we are done early
        if (!intersection_mask)
          break;
      }
      // If we satisfied all the fields then we are good
      return (!intersection_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::physical_premap_region(Operation *op, unsigned index,
                                                RegionRequirement &req,
                                                const VersionInfo &version_info,
                                                InstanceSet &targets,
                                      FieldMaskSet<ReplicatedView> &collectives,
                                      std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PREMAP_ONLY_CALL);
#ifdef DEBUG_LEGION
      assert((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
              (req.handle_type == LEGION_REGION_PROJECTION));
#endif
      // If we are a NO_ACCESS or there are no fields then we are already done 
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return;
      // Iterate over the equivalence sets and get all the instances that
      // are valid for all the different equivalence classes
      IndexSpaceNode *expr_node = get_node(req.region.get_index_space());
      ValidInstAnalysis analysis(runtime, op, index, expr_node,
                                 IS_REDUCE(req) ? req.redop : 0);
      const RtEvent traversal_done = analysis.perform_traversal(
          RtEvent::NO_RT_EVENT, version_info, map_applied_events);
      RtEvent ready;
      if (traversal_done.exists() || analysis.has_remote_sets())
        ready = analysis.perform_remote(traversal_done, map_applied_events);
      // Wait for all the responses to be ready
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      FieldMaskSet<LogicalView> instances;
      if (analysis.report_instances(instances))
        req.flags |= LEGION_RESTRICTED_FLAG;
      const std::vector<LogicalRegion> to_meet(1, req.region); 
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->first->is_instance_view());
#endif
        if (it->first->is_individual_view())
        {
          IndividualView *view = it->first->as_individual_view();
          PhysicalManager *manager = view->get_manager();
          if (manager->meets_regions(to_meet))
            targets.add_instance(InstanceRef(manager, it->second));
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(it->first->is_replicated_view());
#endif
          ReplicatedView *view = it->first->as_replicated_view();
          if (view->meets_regions(to_meet))
            collectives.insert(view, it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::physical_perform_updates(
                               const RegionRequirement &req,
                               const VersionInfo &version_info,
                               Operation *op, unsigned index,
                               ApEvent precondition, ApEvent term_event,
                               const InstanceSet &targets,
                               const std::vector<PhysicalManager*> &sources,
                               const PhysicalTraceInfo &trace_info,
                               std::set<RtEvent> &map_applied_events,
                               UpdateAnalysis *&analysis,
#ifdef DEBUG_LEGION
                               const char *log_name,
                               UniqueID uid,
#endif
                               const bool collective_rendezvous,
                               const bool record_valid,
                               const bool check_initialized,
                               const bool defer_copies)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_REGISTER_ONLY_CALL);
      // If we are a NO_ACCESS or there are no fields then we are already done 
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return RtEvent::NO_RT_EVENT;
#ifdef DEBUG_LEGION
      InnerContext *context = op->find_physical_context(index);
      ContextID ctx = context->get_physical_tree_context();
      assert((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
              (req.handle_type == LEGION_REGION_PROJECTION));
      assert(!targets.empty());
      assert(!targets.is_virtual_mapping());
#endif
      RegionNode *region_node = get_node(req.region);
      const FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
#ifdef DEBUG_LEGION 
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     region_node, ctx, 
                                     true/*before*/, false/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                     FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Perform the registration
#ifdef DEBUG_LEGION
      assert(analysis == NULL);
      // Should be recording or must be read-only
      assert(record_valid || IS_READ_ONLY(req));
#endif
      analysis = new UpdateAnalysis(runtime, op, index, req, region_node,
                                    trace_info, precondition, term_event,
                                    check_initialized, record_valid);
      analysis->add_reference(); 
      const RtEvent views_ready = analysis->convert_views(req.region,
          targets, &sources, &analysis->usage, collective_rendezvous);
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      // Send out any remote updates
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready = 
          analysis->perform_remote(traversal_done, map_applied_events);
      // Issue any release copies/fills that need to be done
      const RtEvent updates_done = 
        analysis->perform_updates(traversal_done, map_applied_events);
      if (remote_ready.exists())
      {
        if (updates_done.exists())
          return Runtime::merge_events(remote_ready, updates_done);
        else
          return remote_ready;
      }
      else
        return updates_done;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::physical_perform_registration(
                           RtEvent precondition, UpdateAnalysis *analysis, 
                           std::set<RtEvent> &map_applied_events, bool symbolic)
    //--------------------------------------------------------------------------
    {
      // If we are a NO_ACCESS or there are no fields then analysis will be NULL
      if (analysis == NULL)
        return ApEvent::NO_AP_EVENT;
      ApEvent instances_ready;
      const RtEvent registered = analysis->perform_registration(precondition,
          analysis->usage, map_applied_events, analysis->precondition,
          analysis->term_event, instances_ready, symbolic);
      // Perform any output copies (e.g. for restriction) that need to be done
      if (registered.exists() || analysis->has_output_updates())
        analysis->perform_output(registered, map_applied_events);
      // Remove the reference that we added in the updates step
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::physical_perform_updates_and_registration(
                                       const RegionRequirement &req,
                                       const VersionInfo &version_info,
                                       Operation *op, unsigned index,
                                       ApEvent precondition, 
                                       ApEvent term_event,
                                       const InstanceSet &targets,
                                       const std::vector<PhysicalManager*> &src,
                                       const PhysicalTraceInfo &trace_info,
                                       std::set<RtEvent> &map_applied_events,
#ifdef DEBUG_LEGION
                                       const char *log_name,
                                       UniqueID uid,
#endif
                                       const bool collective_rendezvous,
                                       const bool record_valid,
                                       const bool check_initialized)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // These are some basic sanity checks that each field is represented
      // by exactly one instance and that the total number of fields 
      // represented matches the number of privilege fields.
      // There has been at least one case where this invariant was violated
      // for attach operations and there were more fields represented in
      // instances than there were privileges, see the attach_2d example.
      FieldMask check_mask;
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const FieldMask &mask = targets[idx].get_valid_fields();
        assert(check_mask * mask);
        check_mask |= mask;
      }
      assert(check_mask.pop_count() == req.privilege_fields.size());
#endif
      UpdateAnalysis *analysis = NULL;
      const RtEvent registration_precondition = physical_perform_updates(req,
         version_info, op, index, precondition, term_event, targets, src,
         trace_info, map_applied_events, analysis,
#ifdef DEBUG_LEGION
         log_name, uid,
#endif
         collective_rendezvous, record_valid,
         check_initialized, false/*defer copies*/);
      return physical_perform_registration(registration_precondition, 
                                           analysis, map_applied_events);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::acquire_restrictions(
                                         const RegionRequirement &req,
                                         const VersionInfo &version_info,
                                         AcquireOp *op, unsigned index,
                                         ApEvent precondition, 
                                         ApEvent term_event,
                                         InstanceSet &restricted_instances,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &map_applied_events
#ifdef DEBUG_LEGION
                                         , const char *log_name
                                         , UniqueID uid
#endif
                                         )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
      // should be exclusive
      assert(IS_EXCLUSIVE(req));
#endif
      const bool known_targets = !restricted_instances.empty();
      RegionNode *region = get_node(req.region);
      AcquireAnalysis *analysis =
        new AcquireAnalysis(runtime, op, index, region, trace_info);
      analysis->add_reference();
      RtEvent views_ready;
      if (known_targets)
        views_ready = analysis->convert_views(req.region, restricted_instances);
      // Iterate through the equivalence classes and find all the restrictions
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready =
          analysis->perform_remote(traversal_done, map_applied_events);
      if (!known_targets)
      {
        if (remote_ready.exists() && !remote_ready.has_triggered())
          remote_ready.wait();
        FieldMaskSet<LogicalView> instances;
        analysis->report_instances(instances);
        restricted_instances.resize(instances.size());
        analysis->target_instances.resize(instances.size());
        analysis->target_views.resize(instances.size());
        unsigned inst_index = 0;
        // Note that all of these should be individual views.
        // The only way to get collective restricted view is by
        // doing attaches in control replicated contexts and we insist
        // that all acquire operations in control replicated context
        // explicitly provide a PhysicalRegion argument so we should
        // always go through the known_targets path, therefore there
        // should be no collective views here.
        for (FieldMaskSet<LogicalView>::const_iterator it =
              instances.begin(); it != instances.end(); it++, inst_index++)
        {
#ifdef DEBUG_LEGION
          assert(it->first->is_individual_view());
#endif         
          IndividualView *inst_view = it->first->as_individual_view();
          PhysicalManager *manager = inst_view->get_manager();
          restricted_instances[inst_index] = InstanceRef(manager, it->second);
          analysis->target_instances[inst_index] = manager;
          analysis->target_views[inst_index].insert(inst_view, it->second);
        }
      }
      // Now add users for all the instances
      ApEvent instances_ready;
      const RegionUsage usage(req);
      analysis->perform_registration(remote_ready, usage, map_applied_events,
                                     precondition, term_event, instances_ready);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::release_restrictions(
                                   const RegionRequirement &req,
                                   const VersionInfo &version_info,
                                   ReleaseOp *op, unsigned index,
                                   ApEvent precondition,
                                   ApEvent term_event,
                                   InstanceSet &restricted_instances,
                                   const std::vector<PhysicalManager*> &sources,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events
#ifdef DEBUG_LEGION
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(IS_EXCLUSIVE(req));
#endif
      const bool known_targets = !restricted_instances.empty();
      RegionNode *region = get_node(req.region);
      ReleaseAnalysis *analysis = new ReleaseAnalysis(runtime, op, index,
                                        precondition, region, trace_info);
      analysis->add_reference();
      RtEvent views_ready;
      if (known_targets)
        views_ready = analysis->convert_views(req.region,
                          restricted_instances, &sources);
      // Iterate through the equivalence classes and find all the restrictions
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready = 
          analysis->perform_remote(traversal_done, map_applied_events);
      // Issue any release copies/fills that need to be done
      RtEvent updates_done = 
        analysis->perform_updates(traversal_done, map_applied_events);
      // There are two cases here: one where we have the target intances
      // already from the operation and we know where to put the users
      // and the second case where we need to wait for the analysis to
      // tell us the names of the instances which are restricted
      const RegionUsage usage(req);
      std::vector<ApEvent> released_events;
      if (!known_targets)
      {
        if (remote_ready.exists() && !remote_ready.has_triggered())
          remote_ready.wait();
        FieldMaskSet<LogicalView> instances;
        analysis->report_instances(instances);
        analysis->target_instances.resize(instances.size());
        analysis->target_views.resize(instances.size());
        restricted_instances.resize(instances.size());
        unsigned inst_index = 0;
        // Note that all of these should be individual views.
        // The only way to get collective restricted view is by
        // doing attaches in control replicated contexts and we insist
        // that all release operations in control replicated context
        // explicitly provide a PhysicalRegion argument so we should
        // always go through the known_targets path, therefore there
        // should be no collective views here.
        for (FieldMaskSet<LogicalView>::const_iterator it =
              instances.begin(); it != instances.end(); it++, inst_index++)
        {
#ifdef DEBUG_LEGION
          assert(it->first->is_individual_view());
#endif         
          IndividualView *inst_view = it->first->as_individual_view();
          PhysicalManager *manager = inst_view->get_manager();
          restricted_instances[inst_index] = InstanceRef(manager, it->second);
          analysis->target_instances[inst_index] = manager;
          analysis->target_views[inst_index].insert(inst_view, it->second);
        }
      }
      else
      {
        if (remote_ready.exists())
        {
          if (updates_done.exists())
            updates_done = Runtime::merge_events(updates_done, remote_ready);
          else
            updates_done = remote_ready;
        }
      }
      ApEvent instances_ready;
      analysis->perform_registration(updates_done, usage, map_applied_events,
                                     precondition, term_event, instances_ready);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::copy_across(
                                   const RegionRequirement &src_req,
                                   const RegionRequirement &dst_req,
                                   const VersionInfo &src_version_info,
                                   const VersionInfo &dst_version_info,
                                   const InstanceSet &src_targets,
                                   const InstanceSet &dst_targets,
                                   const std::vector<PhysicalManager*> &sources,
                                   CopyOp *op, 
                                   unsigned src_index, unsigned dst_index,
                                   ApEvent precondition, ApEvent src_ready,
                                   ApEvent dst_ready, PredEvent guard, 
                                 const std::map<Reservation,bool> &reservations,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_COPY_ACROSS_CALL);
#ifdef DEBUG_LEGION
      assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(src_req.instance_fields.size() == dst_req.instance_fields.size());
#endif 
      RegionNode *src_node = get_node(src_req.region);
      RegionNode *dst_node = get_node(dst_req.region);
      IndexSpaceExpression *copy_expr = intersect_index_spaces(
          src_node->row_source, dst_node->row_source);
      // Quick out if there is nothing to copy to
      if (copy_expr->is_empty())
        return ApEvent::NO_AP_EVENT;
      // Perform the copies/reductions across
      InnerContext *context = op->find_physical_context(dst_index);
      LegionVector<FieldMaskSet<InstanceView> > target_views;
      context->convert_analysis_views(dst_targets, target_views);
      if (!src_targets.empty())
      {
        // If we already have the targets there's no need to 
        // iterate over the source equivalence sets as we can just
        // build a standard CopyAcrossUnstructured object
        CopyAcrossUnstructured *across = 
         copy_expr->create_across_unstructured(reservations,false/*preimages*/);
        across->add_reference();
#ifdef LEGION_SPY
        across->src_tree_id = src_req.region.get_tree_id();
        across->dst_tree_id = dst_req.region.get_tree_id();
#endif
        // Fill in the source fields 
        across->initialize_source_fields(this, src_req, src_targets,trace_info);
        // Fill in the destination fields 
        const bool exclusive_redop = 
          IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req);
        across->initialize_destination_fields(this, dst_req, dst_targets, 
                                              trace_info, exclusive_redop);
        // Get the preconditions for this copy
        std::vector<ApEvent> copy_preconditions;
        if (precondition.exists())
          copy_preconditions.push_back(precondition);
        if (src_ready.exists())
          copy_preconditions.push_back(src_ready);
        if (dst_ready.exists())
          copy_preconditions.push_back(dst_ready);
        if (!copy_preconditions.empty())
          precondition = Runtime::merge_events(&trace_info, copy_preconditions);
        ApEvent result = across->execute(op, guard, precondition,
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT, trace_info);
        if (trace_info.recording)
        {
          // Record this with the trace
          trace_info.record_issue_across(result, precondition, precondition,
                        ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT, across);
          LegionMap<UniqueInst,FieldMask> tracing_srcs, tracing_dsts;
          InnerContext *src_context = op->find_physical_context(src_index);
          std::vector<IndividualView*> source_views;
          src_context->convert_individual_views(src_targets, source_views);
          for (unsigned idx = 0; idx < src_targets.size(); idx++)
          {
            const InstanceRef &ref = src_targets[idx];
            const UniqueInst unique_inst(source_views[idx]);
            tracing_srcs[unique_inst] = ref.get_valid_fields();
          }
          for (unsigned idx = 0; idx < target_views.size(); idx++)
          {
#ifdef DEBUG_LEGION
            assert(target_views[idx].size() == 1);
#endif
            FieldMaskSet<InstanceView>::const_iterator it =
              target_views[idx].begin();
#ifdef DEBUG_LEGION
            assert(it->first->is_individual_view());
#endif
            IndividualView *view = it->first->as_individual_view();
            const UniqueInst unique_inst(view);
            tracing_dsts[unique_inst] = it->second;
          }
          trace_info.record_across_insts(result, src_index, dst_index,
                                         LEGION_READ_PRIV, LEGION_WRITE_PRIV,
                                         copy_expr, tracing_srcs, tracing_dsts,
                                         false/*indirect*/, false/*indirect*/,
                                         map_applied_events);
        }
        if (across->remove_reference())
          delete across;
        return result;
      }
#ifdef DEBUG_LEGION
      // Should never need to do any reservations here
      assert(reservations.empty());
#endif
      // Get the field indexes for all the fields
      std::vector<unsigned> src_indexes(src_req.instance_fields.size());
      std::vector<unsigned> dst_indexes(dst_req.instance_fields.size()); 
      src_node->column_source->get_field_indexes(src_req.instance_fields, 
                                                 src_indexes);   
      dst_node->column_source->get_field_indexes(dst_req.instance_fields,
                                                 dst_indexes);
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
      std::vector<IndividualView*> source_views;
      if (!sources.empty())
      {
        InnerContext *src_context = op->find_physical_context(src_index);
        src_context->convert_individual_views(sources, source_views);
      }
      CopyAcrossAnalysis *analysis = new CopyAcrossAnalysis(runtime, op, 
          src_index, dst_index, src_req, dst_req, dst_targets, target_views, 
          source_views, precondition, dst_ready, guard, dst_req.redop,
          src_indexes, dst_indexes, trace_info, perfect);
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
    ApEvent RegionTreeForest::gather_across(const RegionRequirement &src_req,
                                            const RegionRequirement &idx_req,
                                            const RegionRequirement &dst_req,
                                      std::vector<IndirectRecord> &src_records,
                                            const InstanceSet &src_targets,
                                            const InstanceSet &idx_targets,
                                            const InstanceSet &dst_targets,
                                            CopyOp *op, unsigned src_index,
                                            unsigned idx_index,
                                            unsigned dst_index,
                                            const bool gather_is_range,
                                            const ApEvent init_precondition, 
                                            const ApEvent src_ready,
                                            const ApEvent dst_ready,
                                            const ApEvent idx_ready,
                                            const PredEvent pred_guard,
                                            const ApEvent collective_pre,
                                            const ApEvent collective_post,
                                            const ApUserEvent local_pre,
                                 const std::map<Reservation,bool> &reservations,
                                            const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events,
                                           const bool possible_src_out_of_range,
                                           const bool compute_preimages)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(src_req.instance_fields.size() == dst_req.instance_fields.size());
      assert(idx_req.privilege_fields.size() == 1);
#endif  
      // Get the field indexes for src/dst fields
      IndexSpaceNode *idx_node = get_node(idx_req.region.get_index_space());
      IndexSpaceNode *dst_node = get_node(dst_req.region.get_index_space());
      IndexSpaceExpression *copy_expr = (idx_node == dst_node) ? dst_node :
        intersect_index_spaces(idx_node, dst_node);
      // Trigger the source precondition event when all our sources are ready
      std::vector<ApEvent> local_preconditions;
      if (init_precondition.exists())
        local_preconditions.push_back(init_precondition);
      if (src_ready.exists())
        local_preconditions.push_back(src_ready);
      ApEvent local_precondition;
      if (!local_preconditions.empty())
        local_precondition = 
          Runtime::merge_events(&trace_info, local_preconditions);
      Runtime::trigger_event(&trace_info, local_pre, local_precondition);
      // Easy out if we're not moving anything
      if (copy_expr->is_empty())
        return local_pre;
      CopyAcrossUnstructured *across = 
        copy_expr->create_across_unstructured(reservations, compute_preimages);
      across->add_reference();
      // Initialize the source indirection fields
      const InstanceRef &idx_target = idx_targets[0];
      across->initialize_source_indirections(this, src_records, src_req,
          idx_req, idx_target, gather_is_range, possible_src_out_of_range);
      // Initialize the destination fields
      const bool exclusive_redop =
          IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req);
      across->initialize_destination_fields(this, dst_req, dst_targets,
                                            trace_info, exclusive_redop);
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.push_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      if (dst_ready.exists())
        copy_preconditions.push_back(dst_ready);
      ApEvent src_indirect_ready = idx_ready;
      if (src_indirect_ready.exists())
        copy_preconditions.push_back(src_indirect_ready);
      if (init_precondition.exists())
      {
        if (src_indirect_ready.exists())
          src_indirect_ready = Runtime::merge_events(&trace_info, 
                          src_indirect_ready, init_precondition);
        else
          src_indirect_ready = init_precondition;
      }
      ApEvent copy_precondition;
      if (!copy_preconditions.empty())
        copy_precondition =
          Runtime::merge_events(&trace_info, copy_preconditions);
      // Launch the copy
      ApEvent copy_post = across->execute(op, pred_guard, copy_precondition,
          src_indirect_ready, ApEvent::NO_AP_EVENT, trace_info);
      if (trace_info.recording)
      {
        // Record this with the trace
        trace_info.record_issue_across(copy_post, local_precondition,
           copy_precondition, src_indirect_ready, ApEvent::NO_AP_EVENT, across);
        // If we're tracing record the insts for this copy
        LegionMap<UniqueInst,FieldMask> src_insts, idx_insts, dst_insts;
        // Get the src_insts
        InnerContext *src_context = op->find_physical_context(src_index);
        std::vector<IndividualView*> source_views;
        src_context->convert_individual_views(src_targets, source_views);
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
        {
          const InstanceRef &ref = src_targets[idx];
          const UniqueInst unique_inst(source_views[idx]);
          src_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the idx_insts
        {
          InnerContext *idx_context = op->find_physical_context(idx_index); 
          std::vector<IndividualView*> indirect_views;
          idx_context->convert_individual_views(idx_targets, indirect_views);
          const UniqueInst unique_inst(indirect_views.back());
          idx_insts[unique_inst] = idx_target.get_valid_fields();
        }
        // Get the dst_insts
        InnerContext *dst_context = op->find_physical_context(dst_index);
        std::vector<IndividualView*> target_views;
        dst_context->convert_individual_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < dst_targets.size(); idx++)
        {
          const InstanceRef &ref = dst_targets[idx];
          const UniqueInst unique_inst(target_views[idx]);
          dst_insts[unique_inst] = ref.get_valid_fields();
        }
        IndexSpaceNode *src_node = get_node(src_req.region.get_index_space());
        trace_info.record_indirect_insts(copy_post, collective_post,
            src_node, src_insts, map_applied_events, LEGION_READ_PRIV);
        trace_info.record_across_insts(copy_post, idx_index, dst_index,
            LEGION_READ_PRIV, LEGION_WRITE_PRIV, copy_expr, idx_insts,
            dst_insts, true/*indirect*/, false/*indirect*/, map_applied_events);
      }
      if (across->remove_reference())
        delete across;
      return copy_post;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::scatter_across(const RegionRequirement &src_req,
                                             const RegionRequirement &idx_req,
                                             const RegionRequirement &dst_req,
                                             const InstanceSet &src_targets,
                                             const InstanceSet &idx_targets,
                                             const InstanceSet &dst_targets,
                                      std::vector<IndirectRecord> &dst_records,
                                             CopyOp *op, unsigned src_index,
                                             unsigned idx_index,
                                             unsigned dst_index,
                                             const bool scatter_is_range,
                                             const ApEvent init_precondition, 
                                             const ApEvent src_ready,
                                             const ApEvent dst_ready,
                                             const ApEvent idx_ready,
                                             const PredEvent pred_guard,
                                             const ApEvent collective_pre,
                                             const ApEvent collective_post,
                                             const ApUserEvent local_pre,
                                 const std::map<Reservation,bool> &reservations,
                                            const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events,
                                           const bool possible_dst_out_of_range,
                                             const bool possible_dst_aliasing,
                                             const bool compute_preimages)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(src_req.instance_fields.size() == dst_req.instance_fields.size());
      assert(idx_req.privilege_fields.size() == 1);
      assert(idx_targets.size() == 1);
#endif  
      // Get the field indexes for src/dst fields
      IndexSpaceNode *src_node = get_node(src_req.region.get_index_space());
      IndexSpaceNode *idx_node = get_node(idx_req.region.get_index_space());
      IndexSpaceExpression *copy_expr = (idx_node == src_node) ? idx_node :
        intersect_index_spaces(src_node, idx_node);
      // Trigger the source precondition event when all our sources are ready
      std::vector<ApEvent> local_preconditions;
      if (init_precondition.exists())
        local_preconditions.push_back(init_precondition);
      if (dst_ready.exists())
        local_preconditions.push_back(dst_ready);
      ApEvent local_precondition;
      if (!local_preconditions.empty())
        local_precondition =
          Runtime::merge_events(&trace_info, local_preconditions);
      Runtime::trigger_event(&trace_info, local_pre, local_precondition);
      // Easy out if we're not going to move anything
      if (copy_expr->is_empty())
        return local_pre;
      CopyAcrossUnstructured *across = 
        copy_expr->create_across_unstructured(reservations, compute_preimages);
      across->add_reference();
      // Initialize the sources
      across->initialize_source_fields(this, src_req, src_targets, trace_info);
      // Initialize the destination indirections
      const InstanceRef idx_target = idx_targets[0];
      // Only exclusive if we're the only point sctatting to our instance
      // and we're not racing with any other operations
      const bool exclusive_redop = (dst_records.size() == 1) && 
        (IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req));
      across->initialize_destination_indirections(this, dst_records,
          dst_req, idx_req, idx_target, scatter_is_range,
          possible_dst_out_of_range, possible_dst_aliasing, exclusive_redop);
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.push_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      if (src_ready.exists())
        copy_preconditions.push_back(src_ready);
      ApEvent dst_indirect_ready = idx_ready;
      if (dst_indirect_ready.exists())
        copy_preconditions.push_back(dst_indirect_ready);
      if (init_precondition.exists())
      {
        if (dst_indirect_ready.exists())
          dst_indirect_ready = Runtime::merge_events(&trace_info, 
                          dst_indirect_ready, init_precondition);
        else
          dst_indirect_ready = init_precondition;
      }
      ApEvent copy_precondition;
      if (!copy_preconditions.empty())
        copy_precondition =
          Runtime::merge_events(&trace_info, copy_preconditions);
      // Launch the copy
      ApEvent copy_post = across->execute(op, pred_guard, copy_precondition,
          ApEvent::NO_AP_EVENT, dst_indirect_ready, trace_info);
      if (trace_info.recording)
      {
        // Record this with the trace
        trace_info.record_issue_across(copy_post, local_precondition,
           copy_precondition, ApEvent::NO_AP_EVENT, dst_indirect_ready, across);
        // If we're tracing record the insts for this copy
        LegionMap<UniqueInst,FieldMask> src_insts, idx_insts, dst_insts;
        InnerContext *context = op->find_physical_context(src_index);
        std::vector<IndividualView*> source_views;
        context->convert_individual_views(src_targets, source_views);
        // Get the src_insts
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
        {
          const InstanceRef &ref = src_targets[idx];
          const UniqueInst unique_inst(source_views[idx]);
          src_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the idx_insts
        {
          std::vector<IndividualView*> indirect_views;
          InnerContext *idx_context = op->find_physical_context(idx_index);
          idx_context->convert_individual_views(idx_targets, indirect_views);
          const UniqueInst unique_inst(indirect_views.back());
          idx_insts[unique_inst] = idx_target.get_valid_fields();
        }
        // Get the dst_insts
        std::vector<IndividualView*> target_views;
        InnerContext *dst_context = op->find_physical_context(dst_index);
        dst_context->convert_individual_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < dst_targets.size(); idx++)
        {
          const InstanceRef &ref = dst_targets[idx];
          const UniqueInst unique_inst(target_views[idx]);
          dst_insts[unique_inst] = ref.get_valid_fields();
        }
        trace_info.record_across_insts(copy_post, src_index, idx_index,
            LEGION_READ_PRIV, LEGION_READ_PRIV, copy_expr, src_insts,
            idx_insts, false/*indirect*/, true/*indirect*/,map_applied_events); 
        IndexSpaceNode *dst_node = get_node(dst_req.region.get_index_space());
        trace_info.record_indirect_insts(copy_post, collective_post,
          dst_node, dst_insts, map_applied_events, LEGION_WRITE_PRIV);
      }
      if (across->remove_reference())
        delete across;
      return copy_post;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::indirect_across(const RegionRequirement &src_req,
                              const RegionRequirement &src_idx_req,
                              const RegionRequirement &dst_req,
                              const RegionRequirement &dst_idx_req,
                              const InstanceSet &src_targets,
                              const InstanceSet &dst_targets,
                              std::vector<IndirectRecord> &src_records,
                              const InstanceSet &src_idx_targets,
                              std::vector<IndirectRecord> &dst_records,
                              const InstanceSet &dst_idx_targets, CopyOp *op,
                              unsigned src_index, unsigned dst_index,
                              unsigned src_idx_index, unsigned dst_idx_index,
                              const bool both_are_range,
                              const ApEvent init_precondition, 
                              const ApEvent src_ready,
                              const ApEvent dst_ready,
                              const ApEvent src_idx_ready,
                              const ApEvent dst_idx_ready,
                              const PredEvent pred_guard,
                              const ApEvent collective_pre,
                              const ApEvent collective_post,
                              const ApUserEvent local_pre,
                              const std::map<Reservation,bool> &reservations,
                              const PhysicalTraceInfo &trace_info,
                              std::set<RtEvent> &map_applied_events,
                              const bool possible_src_out_of_range,
                              const bool possible_dst_out_of_range,
                              const bool possible_dst_aliasing,
                              const bool compute_preimages)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(src_idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(dst_idx_req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(src_req.instance_fields.size() == dst_req.instance_fields.size());
      assert(src_idx_req.privilege_fields.size() == 1);
      assert(dst_idx_req.privilege_fields.size() == 1);
      assert(src_idx_targets.size() == 1);
      assert(dst_idx_targets.size() == 1);
#endif  
      // Get the field indexes for src/dst fields
      IndexSpaceNode *src_idx_node = 
        get_node(src_idx_req.region.get_index_space());
      IndexSpaceNode *dst_idx_node = 
        get_node(dst_idx_req.region.get_index_space());
      IndexSpaceExpression *copy_expr = (src_idx_node == dst_idx_node) ?
         src_idx_node : intersect_index_spaces(src_idx_node, dst_idx_node);
      // Trigger the precondition event when all our srcs and dsts are ready
      std::vector<ApEvent> local_preconditions;
      if (init_precondition.exists())
        local_preconditions.push_back(init_precondition);
      if (src_ready.exists())
        local_preconditions.push_back(src_ready);
      if (dst_ready.exists())
        local_preconditions.push_back(dst_ready);
      ApEvent local_precondition;
      if (!local_preconditions.empty())
        local_precondition = 
          Runtime::merge_events(&trace_info, local_preconditions);
      Runtime::trigger_event(&trace_info, local_pre, local_precondition);
      // Quick out if there is nothing we're going to copy
      if (copy_expr->is_empty())
        return local_pre;
      CopyAcrossUnstructured *across = 
        copy_expr->create_across_unstructured(reservations, compute_preimages);
      across->add_reference();
      // Initialize the source indirection fields
      const InstanceRef &src_idx_target = src_idx_targets[0];
      across->initialize_source_indirections(this, src_records, src_req,
        src_idx_req, src_idx_target, both_are_range, possible_src_out_of_range);
      // Initialize the destination indirections
      const InstanceRef &dst_idx_target = dst_idx_targets[0];
      // Only exclusive if we're the only point sctatting to our instance
      // and we're not racing with any other operations
      const bool exclusive_redop = (dst_records.size() == 1) && 
        (IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req));
      across->initialize_destination_indirections(this, dst_records,
          dst_req, dst_idx_req, dst_idx_target, both_are_range,
          possible_dst_out_of_range, possible_dst_aliasing, exclusive_redop);
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.push_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      ApEvent src_indirect_ready = src_idx_ready;
      if (src_indirect_ready.exists())
        copy_preconditions.push_back(src_indirect_ready);
      if (init_precondition.exists())
      {
        if (src_indirect_ready.exists())
          src_indirect_ready = Runtime::merge_events(&trace_info, 
                          src_indirect_ready, init_precondition);
        else
          src_indirect_ready = init_precondition;
      }
      ApEvent dst_indirect_ready = dst_idx_ready;
      if (dst_indirect_ready.exists())
        copy_preconditions.push_back(dst_indirect_ready);
      if (init_precondition.exists())
      {
        if (dst_indirect_ready.exists())
          dst_indirect_ready = Runtime::merge_events(&trace_info, 
                          dst_indirect_ready, init_precondition);
        else
          dst_indirect_ready = init_precondition;
      }
      ApEvent copy_precondition;
      if (!copy_preconditions.empty())
        copy_precondition =
          Runtime::merge_events(&trace_info, copy_preconditions);
      // Launch the copy
      ApEvent copy_post = across->execute(op, pred_guard, copy_precondition,
          src_indirect_ready, dst_indirect_ready, trace_info);
      if (trace_info.recording)
      {
        // Record this with the trace
        trace_info.record_issue_across(copy_post, local_precondition,
            copy_precondition, src_indirect_ready, dst_indirect_ready, across);
        // If we're tracing record the insts for this copy
        LegionMap<UniqueInst,FieldMask> src_insts, src_idx_insts, 
                                        dst_insts, dst_idx_insts;
        // Get the src_insts
        std::vector<IndividualView*> source_views;
        InnerContext *src_context = op->find_physical_context(src_index);
        src_context->convert_individual_views(src_targets, source_views);
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
        {
          const InstanceRef &ref = src_targets[idx];
          const UniqueInst unique_inst(source_views[idx]);
          src_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the src_idx_insts
        {
          InnerContext *src_idx_context =
            op->find_physical_context(src_idx_index);
          std::vector<IndividualView*> src_indirect_views;
          src_idx_context->convert_individual_views(src_idx_targets,
                                                src_indirect_views);
          const UniqueInst unique_inst(src_indirect_views.back());
          src_idx_insts[unique_inst] = src_idx_target.get_valid_fields();
        }
        // Get the dst_insts
        std::vector<IndividualView*> target_views;
        InnerContext *dst_context = op->find_physical_context(dst_index);
        dst_context->convert_individual_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < dst_targets.size(); idx++)
        {
          const InstanceRef &ref = dst_targets[idx];
          const UniqueInst unique_inst(target_views[idx]);
          dst_insts[unique_inst] = ref.get_valid_fields();
        }
        // Get the dst_idx_insts
        {
          InnerContext *dst_idx_context =
            op->find_physical_context(dst_idx_index);
          std::vector<IndividualView*> dst_indirect_views;
          dst_idx_context->convert_individual_views(dst_idx_targets,
                                                dst_indirect_views);
          const UniqueInst unique_inst(dst_indirect_views.back());
          dst_idx_insts[unique_inst] = dst_idx_target.get_valid_fields();
        }
        IndexSpaceNode *src_node = get_node(src_req.region.get_index_space());
        trace_info.record_indirect_insts(copy_post, collective_post,
            src_node, src_insts, map_applied_events, LEGION_READ_PRIV);
        IndexSpaceNode *dst_node = get_node(dst_req.region.get_index_space());
        trace_info.record_indirect_insts(copy_post, collective_post,
           dst_node, dst_insts, map_applied_events, LEGION_WRITE_PRIV);
        trace_info.record_across_insts(copy_post, src_idx_index, dst_idx_index,
            LEGION_READ_PRIV, LEGION_READ_PRIV, copy_expr, src_idx_insts,
            dst_idx_insts,true/*indirect*/,true/*indirect*/,map_applied_events);
      }
      if (across->remove_reference())
        delete across;
      return copy_post;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::fill_fields(FillOp *op,
                                       const RegionRequirement &req,
                                       const unsigned index,
                                       FillView *fill_view,
                                       const VersionInfo &version_info,
                                       ApEvent precondition,
                                       PredEvent true_guard, 
                                       PredEvent false_guard,
                                       const PhysicalTraceInfo &trace_info,
                                       std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_FILL_FIELDS_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *region_node = get_node(req.region);
      bool first_local = true;
      CollectiveMapping *collective_mapping = NULL;
      op->perform_collective_analysis(collective_mapping, first_local);
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime, op, index, 
          RegionUsage(req), region_node->row_source, fill_view, 
          version_info.get_valid_mask(), trace_info, collective_mapping,
          precondition, true_guard, false_guard, false/*add restriction*/,
          first_local);
      analysis->add_reference();
      const RtEvent traversal_done = analysis->perform_traversal(
          RtEvent::NO_RT_EVENT, version_info, map_applied_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_events);
      if (traversal_done.exists() || analysis->has_output_updates())
        analysis->perform_output(traversal_done, map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::discard_fields(DiscardOp *op, const unsigned index,
                                          const RegionRequirement &req,
                                          const VersionInfo &version_info,
                                          const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif     
      RegionNode *region = get_node(req.region);
      FilterAnalysis *analysis = new FilterAnalysis(runtime, op, index,
                                                    region, trace_info);
      analysis->add_reference();
      // Still need to pretend to convert an empty set of views to get
      // the collective mapping initialized properly
      const InstanceSet empty_instances;
      const RtEvent views_ready = analysis->convert_views(req.region,
          empty_instances, NULL/*sources*/, NULL/*usage*/, false/*rendezvous*/);
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      // Send out any remote updates
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::create_external_instance(
                             AttachOp *attach_op, const RegionRequirement &req,
                             const std::vector<FieldID> &field_set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *attach_node = get_node(req.region);
      return attach_node->column_source->create_external_instance(
          req.privilege_fields, field_set, attach_node, attach_op);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::attach_external(AttachOp *attach_op, 
                               unsigned index,
                               const RegionRequirement &req,
                               const InstanceSet &external_instances,
                               const VersionInfo &version_info,
                               const ApEvent termination_event,
                               const PhysicalTraceInfo &trace_info,
                               std::set<RtEvent> &map_applied_events,
                               const bool restricted)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_ATTACH_EXTERNAL_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      IndexSpaceNode *expr_node = get_node(req.region.get_index_space());
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime, attach_op,
            index, RegionUsage(req), expr_node, trace_info,
            ApEvent::NO_AP_EVENT, restricted);
      analysis->add_reference();
      const RtEvent views_ready =
        analysis->convert_views(req.region, external_instances);
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      // Send out any remote updates
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_events);
      // We can perform the registration in parallel with everything else
      ApEvent instances_ready;
      const RegionUsage usage(req);
      RtEvent registration_done = 
        analysis->perform_registration(views_ready, usage, map_applied_events,
          ApEvent::NO_AP_EVENT, termination_event, instances_ready);
      if (registration_done.exists())
        map_applied_events.insert(registration_done);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::detach_external(const RegionRequirement &req,
                                          DetachOp *detach_op,
                                          unsigned index,
                                          const VersionInfo &version_info,
                                          const InstanceSet &instances,
                                          const ApEvent termination_event,
                                          const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events,
                                          RtEvent filter_precondition,
                                          const bool second_analysis)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_DETACH_EXTERNAL_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
      assert(instances.size() == 1);
#endif 
      RegionNode *region = get_node(req.region);
      FilterAnalysis *analysis = new FilterAnalysis(runtime, detach_op, index,
                                region, trace_info, true/*remove restriction*/);
      analysis->add_reference();
      // If we have a filter precondition, then we know this is not the first
      // potential collective analysis to be used here
      const RtEvent views_ready = analysis->convert_views(req.region, 
          instances, NULL/*sources*/, NULL/*usage*/, false/*rendezvous*/, 
          second_analysis ? 1 : 0);
      // Don't start the analysis until the views are ready and the filter
      // precondition has been met
      const RtEvent traversal_precondition = 
        Runtime::merge_events(views_ready, filter_precondition);
      const RtEvent traversal_done = analysis->perform_traversal(
          traversal_precondition, version_info, map_applied_events);
      // Send out any remote updates
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_events);
      ApEvent instances_ready;
      const RegionUsage usage(req);
      analysis->perform_registration(traversal_precondition, usage,
          map_applied_events, ApEvent::NO_AP_EVENT/*no precondition*/,
          termination_event, instances_ready);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_fields(Operation *op, unsigned index,
                                             const RegionRequirement &req,
                                             const VersionInfo &version_info,
                                            const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events,
                                          CollectiveMapping *collective_mapping,
                                              const bool collective_first_local)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      
      const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      IndexSpaceExpression *local_expr = get_node(req.region.get_index_space());
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime, op, index,
          usage, local_expr, NULL/*view*/, version_info.get_valid_mask(), 
          trace_info, collective_mapping, ApEvent::NO_AP_EVENT,
          PredEvent::NO_PRED_EVENT, PredEvent::NO_PRED_EVENT,
          false/*add restriction*/, collective_first_local);
      analysis->add_reference();
      const RtEvent traversal_done = analysis->perform_traversal(
          RtEvent::NO_RT_EVENT, version_info, map_applied_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::physical_convert_sources(Operation *op,
                                  const RegionRequirement &req,
                                  const std::vector<MappingInstance> &sources,
                                  std::vector<PhysicalManager*> &result,
                                  std::map<PhysicalManager*,unsigned> *acquired)
    //--------------------------------------------------------------------------
    {
      const RegionTreeID req_tid = req.parent.get_tree_id();
      std::vector<PhysicalManager*> unacquired;
      for (std::vector<MappingInstance>::const_iterator it = 
            sources.begin(); it != sources.end(); it++)
      {
        InstanceManager *man = it->impl;
        if (man == NULL)
          continue;
        if (man->is_virtual_manager())
          continue;
        PhysicalManager *manager = man->as_physical_manager();
        // Check to see if the region trees are the same
        if (req_tid != manager->tree_id)
          continue;
        if ((acquired != NULL) && (acquired->find(manager) == acquired->end()))
          unacquired.push_back(manager);
        result.push_back(manager);
      }
      if (!unacquired.empty())
      {
        perform_missing_acquires(*acquired, unacquired);
        unsigned unacquired_index = 0;
        for (std::vector<PhysicalManager*>::iterator it = 
              result.begin(); it != result.end(); /*nothing*/)
        {
          if ((*it) == unacquired[unacquired_index])
          {
            if (acquired->find(unacquired[unacquired_index]) == acquired->end())
              it = result.erase(it);
            else
              it++;
            if ((++unacquired_index) == unacquired.size())
              break;
          }
          else
            it++;
        }
      }
    }

    //--------------------------------------------------------------------------
    int RegionTreeForest::physical_convert_mapping(Operation *op,
                                  const RegionRequirement &req,
                                  std::vector<MappingInstance> &chosen,
                                  InstanceSet &result, RegionTreeID &bad_tree,
                                  std::vector<FieldID> &missing_fields,
                                  std::map<PhysicalManager*,unsigned> *acquired,
                                  std::vector<PhysicalManager*> &unacquired,
                                  const bool do_acquire_checks,
                                  const bool allow_partial_virtual)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_CONVERT_MAPPING_CALL);
      // Can be a part projection if we are closing to a partition node
      FieldSpaceNode *node = get_node(req.parent.get_field_space());
      // Get the field mask for the fields we need
      FieldMask needed_fields = node->get_field_mask(req.privilege_fields);
      const RegionTreeID req_tid = req.parent.get_tree_id();
      // Iterate over each one of the chosen instances
      bool has_virtual = false;
      // If we're doing safe mapping, then sort these in order for determinism
      if (!runtime->unsafe_mapper)
        std::sort(chosen.begin(), chosen.end());
      for (std::vector<MappingInstance>::const_iterator it = chosen.begin();
            it != chosen.end(); it++)
      {
        InstanceManager *man = it->impl;
        if (man == NULL)
          continue;
        if (man->is_virtual_manager())
        {
          has_virtual = true;
          continue;
        }
        PhysicalManager *manager = man->as_physical_manager();
        // Check to see if the region trees are the same
        if (req_tid != manager->tree_id)
        {
          bad_tree = manager->tree_id;
          return -1;
        }
        // See if we should be checking the acquired sets
        if (do_acquire_checks && (acquired->find(manager) == acquired->end()))
          unacquired.push_back(manager);
        // See which fields need to be made valid here
        FieldMask valid_fields = 
          manager->layout->allocated_fields & needed_fields;
        if (!valid_fields)
          continue;
        result.add_instance(InstanceRef(manager, valid_fields));
        // We can remove the update fields from the needed mask since
        // we now have space for them
        needed_fields -= valid_fields;
        // If we've seen all our needed fields then we are done
        if (!needed_fields)
          break;
      }
      // If we don't have needed fields, see if we had a composite instance
      // if we did, put all the fields in there, otherwise we put report
      // them as missing fields figure out what field IDs they are
      if (!!needed_fields)
      {
        if (has_virtual)
        {
          if (!allow_partial_virtual)
          {
            // If we don't allow partial virtual results then clear
            // the results and make all the needed fields the result
            result.clear();
            needed_fields = node->get_field_mask(req.privilege_fields);
          }
          int composite_idx = result.size();
          result.add_instance(
              InstanceRef(runtime->virtual_manager, needed_fields));
          return composite_idx;
        }
        else
        {
          // This can be slow because if we get here we are just 
          // going to be reporting an error so performance no
          // longer matters
          std::set<FieldID> missing;
          node->get_field_set(needed_fields, op->get_context(), missing);
          missing_fields.insert(missing_fields.end(), 
                                missing.begin(), missing.end());
        }
      }
      // We'll only run this code when we're checking for errors
      if (!unacquired.empty())
      {
#ifdef DEBUG_LEGION
        assert(acquired != NULL);
#endif
        perform_missing_acquires(*acquired, unacquired); 
      }
      return -1; // no composite index
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::physical_convert_postmapping(Operation *op,
                                  const RegionRequirement &req,
                                  std::vector<MappingInstance> &chosen,
                                  InstanceSet &result,RegionTreeID &bad_tree,
                                  std::map<PhysicalManager*,unsigned> *acquired,
                                  std::vector<PhysicalManager*> &unacquired,
                                  const bool do_acquire_checks)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *reg_node = get_node(req.region);      
      // Get the field mask for the fields we need
      FieldMask optional_fields = 
                reg_node->column_source->get_field_mask(req.privilege_fields);
      const RegionTreeID reg_tree = req.region.get_tree_id();
      // Iterate over each one of the chosen instances
      bool has_composite = false;
      // If we're doing safe mapping, then sort these in order for determinism
      if (!runtime->unsafe_mapper)
        std::sort(chosen.begin(), chosen.end());
      for (std::vector<MappingInstance>::const_iterator it = chosen.begin();
            it != chosen.end(); it++)
      {
        InstanceManager *man = it->impl;
        if (man == NULL)
          continue;
        if (man->is_virtual_manager())
        {
          has_composite = true;
          continue;
        }
        PhysicalManager *manager = man->as_physical_manager();
        // Check to see if the tree IDs are the same
        if (reg_tree != manager->tree_id)
        {
          bad_tree = manager->tree_id;
          return -1;
        }
        // See if we should be checking the acquired sets
        if (do_acquire_checks && (acquired->find(manager) == acquired->end()))
          unacquired.push_back(manager);
        FieldMask valid_fields = 
          manager->layout->allocated_fields & optional_fields;
        if (!valid_fields)
          continue;
        result.add_instance(InstanceRef(manager, valid_fields));
      }
      if (!unacquired.empty())
      {
#ifdef DEBUG_LEGION
        assert(acquired != NULL);
#endif
        perform_missing_acquires(*acquired, unacquired);
      }
      return has_composite;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_missing_acquires(
                                std::map<PhysicalManager*,unsigned> &acquired,
                                const std::vector<PhysicalManager*> &unacquired)
    //--------------------------------------------------------------------------
    {
      // This code is very similar to what we see in the MapperManager
      for (unsigned idx = 0; idx < unacquired.size(); idx++)
      {
        PhysicalManager *manager = unacquired[idx];
        // Try and do the acquires for any instances that weren't acquired
        if (manager->acquire_instance(MAPPING_ACQUIRE_REF))
          // We already know it wasn't there before
          acquired.insert(std::pair<PhysicalManager*,unsigned>(manager, 1));
      }
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void RegionTreeForest::check_context_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      CurrentInitializer init(ctx);
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      // Need to hold references to prevent deletion race
      for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
            tree_nodes.begin(); it != tree_nodes.end(); it++)
        it->second->visit_node(&init);
    }
#endif

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(IndexSpace sp,
                                                  const void *bounds,
                                                  bool is_domain,
                                                  IndexPartNode *parent,
                                                  LegionColor color,
                                                  DistributedID did,
                                                  RtEvent initialized,
                                                  Provenance *provenance,
                                                  ApEvent is_ready,
                                                  IndexSpaceExprID expr_id,
                                                  CollectiveMapping *mapping,
                                                  const bool add_root_reference,
                                                  unsigned depth,
                                                  const bool tree_valid)
    //--------------------------------------------------------------------------
    { 
      IndexSpaceCreator creator(this, sp, parent, color, did, expr_id,
                  initialized, depth, provenance, mapping, tree_valid);
      NT_TemplateHelper::demux<IndexSpaceCreator>(sp.get_type_tag(), &creator);
      IndexSpaceNode *result = creator.result;  
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator it =
          index_nodes.find(sp);
        if (it != index_nodes.end())
        {
          // Need to remove resource reference if not owner
          delete result;
          result = it->second;
          // If the parent is NULL then we don't need to perform a duplicate set
          if ((bounds == NULL) || (parent == NULL))
            return result;
        }
        else
        {
          index_nodes[sp] = result;
          index_space_requests.erase(sp);
          if (add_root_reference)
            result->add_base_valid_ref(APPLICATION_REF);
          // If we didn't give it a value add a reference to be removed once
          // the index space node has been set
          if (bounds == NULL)
          {
            // Hold the reference on the parent partition to keep both it
            // and the child index space alive if there is a a parent
            if (result->parent != NULL)
              result->parent->add_base_gc_ref(REGION_TREE_REF);
            else
              result->add_base_gc_ref(REGION_TREE_REF);
          }
          else
            result->set_bounds(bounds, is_domain, true/*init*/, is_ready);
          if (parent != NULL)
          {
#ifdef DEBUG_LEGION
            assert(!add_root_reference);
#endif
            // Only do this after we've added all the references
            parent->add_child(result);
          }
          result->register_with_runtime();
          return result;
        }
      }
#ifdef DEBUG_LEGION
      assert(bounds != NULL);
#endif
      if (result->set_bounds(bounds, is_domain, false/*init*/, is_ready))
        assert(false); // should never hit this
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(IndexSpace sp,
                                                  IndexPartNode &parent,
                                                  LegionColor color,
                                                  DistributedID did,
                                                  RtEvent initialized,
                                                  Provenance *provenance,
                                                  IndexSpaceExprID expr_id,
                                                  CollectiveMapping *mapping,
                                                  unsigned depth)
    //--------------------------------------------------------------------------
    { 
      IndexSpaceCreator creator(this, sp, &parent, color, did, expr_id,
                                initialized, depth, provenance, mapping,
                                true/*tree valid*/);
      NT_TemplateHelper::demux<IndexSpaceCreator>(sp.get_type_tag(), &creator);
      IndexSpaceNode *result = creator.result;  
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator it =
          index_nodes.find(sp);
        if (it != index_nodes.end())
        {
          delete result;
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
        // Add a reference for when we set this index space node
        // Hold the reference on the parent partition to keep both it
        // and the child index space alive 
        parent.add_base_gc_ref(REGION_TREE_REF);
        // Only record this with the parent after all the references are added
        parent.add_child(result);
        result->register_with_runtime();
      } 
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, 
                                                 IndexSpaceNode *parent,
                                                 IndexSpaceNode *color_space,
                                                 LegionColor color,
                                                 bool disjoint, int complete,
                                                 DistributedID did,
                                                 Provenance *provenance,
                                                 RtEvent initialized,
                                                 CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(this, p, parent, color_space, color, disjoint,
                             complete, did, initialized, mapping, provenance);
      NT_TemplateHelper::demux<IndexPartCreator>(p.get_type_tag(), &creator);
      IndexPartNode *result = creator.result;
#ifdef DEBUG_LEGION
      assert(parent != NULL);
      assert(result != NULL);
#endif
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition,IndexPartNode*>::const_iterator it =
          index_parts.find(p);
        if (it != index_parts.end())
        {
          delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, 
        if (result->is_owner())
          result->add_base_valid_ref(APPLICATION_REF);
        parent->add_child(result);
        // Add it to the partition of our parent if it exists, otherwise
        // our parent index space is a root so we add the reference there
        if (parent->parent != NULL)
          parent->parent->add_nested_valid_ref(did);
        else
          parent->add_nested_valid_ref(did);
        if (color_space->parent != NULL)
          color_space->parent->add_nested_valid_ref(did);
        else
          color_space->add_nested_valid_ref(did);
        // We know if we're disjoint or not but if we're not complete we might 
        // still be getting notifications to compute the complete
        if (complete < 0)
          result->initialize_disjoint_complete_notifications();
        else if ((implicit_profiler != NULL) && result->is_owner())
          implicit_profiler->register_index_partition(parent->handle.id,
              p.id, disjoint, result->color);
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, 
                                                 IndexSpaceNode *parent,
                                                 IndexSpaceNode *color_space,
                                                 LegionColor color,
                                                 int complete, 
                                                 DistributedID did,
                                                 Provenance *provenance,
                                                 RtEvent initialized,
                                                 CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(this, p, parent, color_space, color, 
                               complete, did, initialized, mapping, provenance);
      NT_TemplateHelper::demux<IndexPartCreator>(p.get_type_tag(), &creator);
      IndexPartNode *result = creator.result;
#ifdef DEBUG_LEGION
      assert(parent != NULL);
      assert(result != NULL);
#endif
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition,IndexPartNode*>::const_iterator it =
          index_parts.find(p);
        if (it != index_parts.end())
        {
          // Need to remove resource reference if not owner
          delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, 
        if (result->is_owner())
          result->add_base_valid_ref(APPLICATION_REF);
        parent->add_child(result);
        // Add it to the partition of our parent if it exists, otherwise
        // our parent index space is a root so we add the reference there
        if (parent->parent != NULL)
          parent->parent->add_nested_valid_ref(did);
        else
          parent->add_nested_valid_ref(did);
        if (color_space->parent != NULL)
          color_space->parent->add_nested_valid_ref(did);
        else
          color_space->add_nested_valid_ref(did);
        // We don't know if we're disjonit or yet not so we need to do
        // the disjoint and complete analysis
        result->initialize_disjoint_complete_notifications();
        result->register_with_runtime();
      }
      return result;
    }
 
    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace space,
                                                  DistributedID did,
                                                  RtEvent initialized,
                                                  Provenance *provenance,
                                                  CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *result = new FieldSpaceNode(space, this, did,
          initialized, mapping, provenance);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Hold the lookup lock while modifying the lookup table
      {
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace,FieldSpaceNode*>::const_iterator it =
          field_nodes.find(space);
        if (it != field_nodes.end())
        {
          delete result;
          return it->second;
        }
        field_nodes[space] = result;
        field_space_requests.erase(space);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, otherwise we're remote so we add a gc 
        // reference that will be removed by the owner when we can be
        // safely collected
        if (result->is_owner())
          result->add_base_gc_ref(APPLICATION_REF);
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace space,
                                                  DistributedID did,
                                                  RtEvent initialized,
                                                  Provenance *provenance,
                                                  CollectiveMapping *mapping,
                                                  Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *result = new FieldSpaceNode(space, this, did,
          initialized, mapping, provenance, derez);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Hold the lookup lock while modifying the lookup table
      {
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace,FieldSpaceNode*>::const_iterator it =
          field_nodes.find(space);
        if (it != field_nodes.end())
        {
          delete result;
          return it->second;
        }
        field_nodes[space] = result;
        field_space_requests.erase(space);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, otherwise we're remote so we add a gc 
        // reference that will be removed by the owner when we can be
        // safely collected
        if (result->is_owner())
          result->add_base_gc_ref(APPLICATION_REF);
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::create_node(LogicalRegion r, 
                                              PartitionNode *parent,
                                              RtEvent initialized,
                                              DistributedID did,
                                              Provenance *provenance,
                                              CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (parent != NULL)
      {
        assert(r.field_space == parent->handle.field_space);
        assert(r.tree_id == parent->handle.tree_id);
      }
#endif
      // Special case for root nodes without dids, we better find them
      if ((parent == NULL) && (did == 0))
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        // Check to see if it already exists
        std::map<LogicalRegion,RegionNode*>::const_iterator finder =
          region_nodes.find(r);
#ifdef DEBUG_LEGION
        assert(finder != region_nodes.end());
#endif
        return finder->second;
      }
      RtEvent row_ready, col_ready;
      IndexSpaceNode *row_src = get_node(r.index_space, &row_ready);
      FieldSpaceNode *col_src = get_node(r.field_space, &col_ready);
      if (row_src == NULL)
      {
        row_ready.wait();
        row_src = get_node(r.index_space);
        row_ready = RtEvent::NO_RT_EVENT;
      }
      if (col_src == NULL)
      {
        col_ready.wait();
        col_src = get_node(r.field_space);
        col_ready = RtEvent::NO_RT_EVENT;
      }
      
      if (row_ready.exists() || col_ready.exists())
        initialized = Runtime::merge_events(initialized, row_ready, col_ready); 
      RegionNode *result = new RegionNode(r, parent, row_src, col_src, this,did,
        initialized, (parent == NULL) ? initialized : parent->tree_initialized,
        mapping, provenance);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Special case here in case multiple clients attempt to
      // make the node at the same time
      {
        // Hold the lookup lock when modifying the lookup table
        AutoLock l_lock(lookup_lock);
        // Check to see if it already exists
        std::map<LogicalRegion,RegionNode*>::const_iterator it =
          region_nodes.find(r);
        if (it != region_nodes.end())
        {
          // It already exists, delete our copy and return
          // the one that has already been made
          delete result;
          return it->second;
        }
        // Now we can add it to the map
        region_nodes[r] = result;
        // If this is a top level region add it to the collection
        // of top level tree IDs
        if (parent == NULL)
        {
#ifdef DEBUG_LEGION
          assert(tree_nodes.find(r.tree_id) == tree_nodes.end());
#endif
          tree_nodes[r.tree_id] = result;
          region_tree_requests.erase(r.tree_id);
          // If we're the root we get a valid reference on the owner
          // node otherwise we get a gc ref from the owner node
          if (result->is_owner())
            result->add_base_gc_ref(APPLICATION_REF);
        }
        result->record_registered();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::create_node(LogicalPartition p,
                                                 RegionNode *parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent != NULL);
      assert(p.field_space == parent->handle.field_space);
      assert(p.tree_id == parent->handle.tree_id);
#endif
      RtEvent row_ready, col_ready;
      IndexPartNode *row_src = get_node(p.index_partition, &row_ready);
      FieldSpaceNode *col_src = get_node(p.field_space, &col_ready);
      if (row_src == NULL)
      {
        row_ready.wait();
        row_src = get_node(p.index_partition);
        row_ready = RtEvent::NO_RT_EVENT;
      }
      if (col_src == NULL)
      {
        col_ready.wait();
        col_src = get_node(p.field_space);
        col_ready = RtEvent::NO_RT_EVENT;
      }
      RtEvent initialized = parent->tree_initialized;
      if (row_ready.exists() || col_ready.exists())
        initialized = Runtime::merge_events(initialized, row_ready, col_ready);
      PartitionNode *result = new PartitionNode(p, parent, row_src, col_src, 
                                  this, initialized, parent->tree_initialized);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Special case here in case multiple clients attempt
      // to make the node at the same time
      {
        // Hole the lookup lock when modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<LogicalPartition,PartitionNode*>::const_iterator it =
          part_nodes.find(p);
        if (it != part_nodes.end())
        {
          // It already exists, delete our copy and
          // return the one that has already been made
          delete result;
          return it->second;
        }
        // Now we can put the node in the map
        part_nodes[p] = result;
        // Add gc ref that will be removed when either the root region node
        // or the index partition node has been destroyed
        result->add_base_gc_ref(REGION_TREE_REF);
        result->record_registered();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::get_node(IndexSpace space, 
                                               RtEvent *defer /*=NULL*/,
                                               const bool can_fail /*=false*/,
                                               const bool first /*=true*/)
    //--------------------------------------------------------------------------
    {
      if (!space.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_FOR_INDEXSPACE,
          "Invalid request for IndexSpace NO_SPACE.")
      RtEvent wait_on;
      IndexSpaceNode *result = NULL;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/); 
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator finder = 
          index_nodes.find(space);
        if (finder != index_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          if ((defer != NULL) && !finder->second->initialized.has_triggered())
          {
            *defer = finder->second->initialized;
            return finder->second;
          }
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != NULL)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpace owner = IndexSpaceNode::get_owner_space(space, runtime);
      if (owner == runtime->address_space)
      {
        // See if it is in the set of pending spaces in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<IndexSpaceID,RtUserEvent>::iterator finder = 
            pending_index_spaces.find(space.get_id());
          if (finder != pending_index_spaces.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          if (defer != NULL)
          {
            *defer = pending_wait;
            return NULL;
          }
          else
          {
            pending_wait.wait();
            return get_node(space, defer, false/*can fail*/, false/*first*/);
          }
        }
        else if (can_fail)
          return NULL;
        else
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for index space %x.", space.id)
      }
      // Retake the lock and get something to wait on
      {
        AutoLock l_lock(lookup_lock);
        // Check to make sure we didn't loose the race
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator finder = 
          index_nodes.find(space);
        if (finder != index_nodes.end())
          return finder->second;
        // Still doesn't exists, see if we sent a request already
        std::map<IndexSpace,RtEvent>::const_iterator wait_finder = 
          index_space_requests.find(space);
        if (wait_finder == index_space_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          index_space_requests[space] = done;
          Serializer rez;
          rez.serialize(space);
          rez.serialize(done);
          rez.serialize(runtime->address_space);
          runtime->send_index_space_request(owner, rez);     
          wait_on = done;
        }
        else
          wait_on = wait_finder->second;
      }
      if (defer == NULL)
      {
        // Wait on the event
        wait_on.wait();
        {
          AutoLock l_lock(lookup_lock);
          std::map<IndexSpace,IndexSpaceNode*>::iterator finder = 
              index_nodes.find(space);
          if (finder != index_nodes.end())
          {
            if (finder->second->initialized.exists())
            {
              if (finder->second->initialized.has_triggered())
              {
                finder->second->initialized = RtEvent::NO_RT_EVENT;
                return finder->second;
              }
              else
                wait_on = finder->second->initialized;
            }
            else
              return finder->second;
          }
          else if (can_fail)
            return NULL;
          else
            wait_on = RtEvent::NO_RT_EVENT;
        }
        if (!wait_on.exists())
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for index space %x."
                          "This is definitely a runtime bug.", space.id)
        wait_on.wait();
        return get_node(space, NULL, can_fail, false/*first*/);
      }
      else
      {
        *defer = wait_on;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::get_node(IndexPartition part,
                                              RtEvent *defer/* = NULL*/,
                                              const bool can_fail /* = false*/,
                                              const bool first/* = true*/,
                                              const bool local_only/* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!part.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_INDEXPARTITION,
          "Invalid request for IndexPartition NO_PART.")
      RtEvent wait_on;
      IndexPartNode *result = NULL;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<IndexPartition,IndexPartNode*>::const_iterator finder =
          index_parts.find(part);
        if (finder != index_parts.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          if ((defer != NULL) && !finder->second->initialized.has_triggered())
          {
            *defer = finder->second->initialized;
            return finder->second;
          }
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != NULL)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpace owner = IndexPartNode::get_owner_space(part, runtime);
      // If we only want to do the test locally then return the result too
      if ((owner == runtime->address_space) || local_only)
      {
        // See if it is in the set of pending partitions in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<IndexPartitionID,RtUserEvent>::iterator finder = 
            pending_partitions.find(part.get_id());
          if (finder != pending_partitions.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          if (defer != NULL)
          {
            *defer = pending_wait;
            return NULL;
          }
          else
          {
            pending_wait.wait();
            return get_node(part, defer, false/*can fail*/, false/*first*/);
          }
        }
        else if (can_fail)
          return NULL;
        else
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for index partition %x.",part.id)
      }
      {
        // Retake the lock in exclusive mode and make
        // sure we didn't loose the race
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition,IndexPartNode*>::const_iterator finder =
          index_parts.find(part);
        if (finder != index_parts.end())
          return finder->second;
        // See if we've already sent the request or not
        std::map<IndexPartition,RtEvent>::const_iterator wait_finder = 
          index_part_requests.find(part);
        if (wait_finder == index_part_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          index_part_requests[part] = done;
          Serializer rez;
          rez.serialize(part);
          rez.serialize(done);
          rez.serialize(runtime->address_space);
          runtime->send_index_partition_request(owner, rez);    
          wait_on = done;
        }
        else
          wait_on = wait_finder->second;
      }
      if (defer == NULL)
      {
        // Wait for the event
        wait_on.wait();
        {
          AutoLock l_lock(lookup_lock);
          std::map<IndexPartition,IndexPartNode*>::iterator finder = 
            index_parts.find(part);
          if (finder != index_parts.end())
          {
            if (finder->second->initialized.exists())
            {
              if (finder->second->initialized.has_triggered())
              {
                finder->second->initialized = RtEvent::NO_RT_EVENT;
                return finder->second;
              }
              else
                wait_on = finder->second->initialized;
            }
            else
              return finder->second;
          }
          else if (can_fail)
            return NULL;
          else
            wait_on = RtEvent::NO_RT_EVENT;
        }
        if (!wait_on.exists())
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for index partition %x. "
                          "This is definitely a runtime bug.", part.id)
        wait_on.wait();
        return get_node(part, NULL, can_fail, false/*first*/);
      }
      else
      {
        *defer = wait_on;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::get_node(FieldSpace space,
                                 RtEvent *defer /*=NULL*/, bool first /*=true*/) 
    //--------------------------------------------------------------------------
    {
      if (!space.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_FIELDSPACE,
          "Invalid request for FieldSpace NO_SPACE.")
      RtEvent wait_on;
      FieldSpaceNode *result = NULL;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<FieldSpace,FieldSpaceNode*>::const_iterator finder = 
          field_nodes.find(space);
        if (finder != field_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          if ((defer != NULL) && !finder->second->initialized.has_triggered())
          {
            *defer = finder->second->initialized;
            return finder->second;
          }
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != NULL)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpaceID owner = FieldSpaceNode::get_owner_space(space, runtime); 
      if (owner == runtime->address_space)
      {
        // See if it is in the set of pending spaces in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<FieldSpaceID,RtUserEvent>::iterator finder = 
            pending_field_spaces.find(space.get_id());
          if (finder != pending_field_spaces.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          if (defer != NULL)
          {
            *defer = pending_wait;
            return NULL;
          }
          else
          {
            pending_wait.wait();
            return get_node(space, defer, false/*first*/);
          }
        }
        else
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for field space %x.", space.id)
      }
      {
        // Retake the lock in exclusive mode and 
        // check to make sure we didn't loose the race
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace,FieldSpaceNode*>::const_iterator finder = 
          field_nodes.find(space);
        if (finder != field_nodes.end())
          return finder->second;
        // Now see if we've already sent a request
        std::map<FieldSpace,RtEvent>::const_iterator wait_finder = 
          field_space_requests.find(space);
        if (wait_finder == field_space_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          field_space_requests[space] = done;
          Serializer rez;
          rez.serialize(space);
          rez.serialize(done);
          rez.serialize(runtime->address_space);
          runtime->send_field_space_request(owner, rez);    
          wait_on = done;
        }
        else
          wait_on = wait_finder->second;
      }
      if (defer == NULL)
      {
        // Wait for the event to be ready
        wait_on.wait();
        {
          AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
          std::map<FieldSpace,FieldSpaceNode*>::const_iterator finder = 
            field_nodes.find(space);
          if (finder != field_nodes.end())
          {
            if (finder->second->initialized.exists())
            {
              if (finder->second->initialized.has_triggered())
              {
                finder->second->initialized = RtEvent::NO_RT_EVENT;
                return finder->second;
              }
              else
                wait_on = finder->second->initialized;
            }
            else
              return finder->second;
          }
          else
            wait_on = RtEvent::NO_RT_EVENT;
        }
        if (!wait_on.exists())
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for field space %x. "
                          "This is definitely a runtime bug.", space.id)
        wait_on.wait();
        return get_node(space, NULL, false/*first*/);
      }
      else
      {
        *defer = wait_on;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::get_node(LogicalRegion handle,
                              bool need_check /* = true*/, bool first /*=true*/)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_LOGICALREGION,
          "Invalid request for LogicalRegion NO_REGION.")
      // Check to see if the node already exists
      RtEvent wait_on;
      RegionNode *result = NULL;
      bool has_top_level_region = false;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<LogicalRegion,RegionNode*>::const_iterator finder = 
          region_nodes.find(handle);
        if (finder != region_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          wait_on = finder->second->initialized;
          result = finder->second;
        }
        // Check to see if we have the top level region
        else if (need_check)
          has_top_level_region = 
            (tree_nodes.find(handle.get_tree_id()) != tree_nodes.end());
        else
          has_top_level_region = true;
      }
      if (result != NULL)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // If we don't have the top-level region, we need to request it before
      // we go crawling up the tree so we know where to stop
      if (!has_top_level_region)
      {
        AddressSpaceID owner = 
          RegionTreeNode::get_owner_space(handle.get_tree_id(), runtime);
        if (owner == runtime->address_space)
        {
          // See if it is in the set of pending spaces in which case we
          // can wait for it to be recorded
          RtEvent pending_wait;
          if (first)
          {
            AutoLock l_lock(lookup_lock);
            std::map<RegionTreeID,RtUserEvent>::iterator finder = 
              pending_region_trees.find(handle.get_tree_id());
            if (finder != pending_region_trees.end())
            {
              if (!finder->second.exists())
                finder->second = Runtime::create_rt_user_event();
              pending_wait = finder->second;
            }
          }
          if (pending_wait.exists())
          {
            pending_wait.wait();
            return get_node(handle, need_check, false/*first*/); 
          }
          else
            REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
              "Unable to find entry for logical region tree %d.",
                             handle.get_tree_id());
        }
        {
          // Retake the lock and make sure we didn't loose the race
          AutoLock l_lock(lookup_lock);
          if (tree_nodes.find(handle.get_tree_id()) == tree_nodes.end())
          {
            // Still don't have it, see if we need to request it
            std::map<RegionTreeID,RtEvent>::const_iterator finder = 
              region_tree_requests.find(handle.get_tree_id());
            if (finder == region_tree_requests.end())
            {
              RtUserEvent done = Runtime::create_rt_user_event();
              region_tree_requests[handle.get_tree_id()] = done;
              Serializer rez;
              rez.serialize(handle.get_tree_id());
              rez.serialize(done);
              rez.serialize(runtime->address_space);
              runtime->send_top_level_region_request(owner, rez);
              wait_on = done;
            }
            else
              wait_on = finder->second;
          }
          else
          {
            // We lost the race and it may be here now
            std::map<LogicalRegion,RegionNode*>::const_iterator it = 
              region_nodes.find(handle);
            if (it != region_nodes.end())
              return it->second;
          }
        }
        // If we did find something to wait on, do that now
        if (wait_on.exists())
        {
          wait_on.wait();
          RegionNode *result = NULL;
          {
            // Retake the lock and see again if the handle we
            // were looking for was the top-level node or not
            AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
            std::map<LogicalRegion,RegionNode*>::const_iterator it = 
              region_nodes.find(handle);
            if (it != region_nodes.end())
            {
              result = it->second;
              wait_on = result->initialized;
            }
          }
          if (result != NULL)
          {
            if (wait_on.exists())
            {
              if (!wait_on.has_triggered())
                wait_on.wait();
              AutoLock l_lock(lookup_lock);
              result->initialized = RtEvent::NO_RT_EVENT;
            }
            return result;
          }
        }
      }
      // Otherwise it hasn't been made yet, so make it
      IndexSpaceNode *index_node = get_node(handle.index_space);
      if (index_node->parent != NULL)
      {
#ifdef DEBUG_LEGION
        assert(index_node->parent != NULL);
#endif
        LogicalPartition parent_handle(handle.tree_id, 
                              index_node->parent->handle, handle.field_space);
        // Note this request can recursively build more nodes, but we
        // are guaranteed that the top level node exists
        PartitionNode *parent = get_node(parent_handle, false/*need check*/);
        // Now make our node and then return it
        result = create_node(handle, parent, RtEvent::NO_RT_EVENT, 0/*did*/);
      }
      else
      {
#ifdef DEBUG_LEGION
        // This better be a root node, if it's not then something requested
        // that we construct a logical reigon node after the parent partition
        // was destroyed which is very bad
        assert(index_node->depth == 0);
#endif
        // Even though this is a root node, we'll discover it's already made
        result = create_node(handle, NULL, RtEvent::NO_RT_EVENT, 0/*did*/);
      }
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        if (!result->initialized.exists())
          return result;
        wait_on = result->initialized;
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
      AutoLock l_lock(lookup_lock);
      result->initialized = RtEvent::NO_RT_EVENT;
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::get_node(LogicalPartition handle,
                                              bool need_check /* = true*/)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_LOGICALPARTITION,
          "Invalid request for LogicalPartition NO_PART.")
      RtEvent wait_on;
      PartitionNode *result = NULL;
      // Check to see if the node already exists
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<LogicalPartition,PartitionNode*>::const_iterator it =
          part_nodes.find(handle);
        if (it != part_nodes.end())
        {
          if (it->second->initialized.exists())
          {
            wait_on = it->second->initialized;
            result = it->second;
          }
          else
            return it->second;
        }
      }
      if (result != NULL)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Otherwise it hasn't been made yet so make it
      IndexPartNode *index_node = get_node(handle.index_partition);
#ifdef DEBUG_LEGION
      assert(index_node->parent != NULL);
#endif
      LogicalRegion parent_handle(handle.tree_id, index_node->parent->handle,
                                  handle.field_space);
      // Note this request can recursively build more nodes, but we
      // are guaranteed that the top level node exists
      RegionNode *parent = get_node(parent_handle, need_check);
      // Now create our node and return it
      result = create_node(handle, parent);
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        if (!result->initialized.exists())
          return result;
        wait_on = result->initialized;
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
      AutoLock l_lock(lookup_lock);
      result->initialized = RtEvent::NO_RT_EVENT;
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::get_tree(RegionTreeID tid, bool can_fail,
                                           bool first/*=true*/)
    //--------------------------------------------------------------------------
    {
      if (tid == 0)
      {
        if (can_fail)
          return NULL;
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_TREE_ID,
          "Invalid request for tree ID 0 which is never a tree ID")
      }
      RtEvent wait_on;
      RegionNode *result = NULL;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<RegionTreeID,RegionNode*>::const_iterator finder = 
          tree_nodes.find(tid);
        if (finder != tree_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != NULL)
      {
        wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpaceID owner = RegionTreeNode::get_owner_space(tid, runtime);
      if (owner == runtime->address_space)
      {
        // See if it is in the set of pending spaces in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<RegionTreeID,RtUserEvent>::iterator finder = 
            pending_region_trees.find(tid);
          if (finder != pending_region_trees.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          pending_wait.wait();
          return get_tree(tid, can_fail, false/*first*/); 
        }
        else if (can_fail)
          return NULL;
        else
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for region tree ID %d", tid)
      }
      {
        // Retake the lock in exclusive mode and check to
        // make sure that we didn't lose the race
        AutoLock l_lock(lookup_lock);
        std::map<RegionTreeID,RegionNode*>::const_iterator finder = 
          tree_nodes.find(tid);
        if (finder != tree_nodes.end())
          return finder->second;
        // Now see if we've already send a request
        std::map<RegionTreeID,RtEvent>::const_iterator req_finder =
          region_tree_requests.find(tid);
        if (req_finder == region_tree_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          region_tree_requests[tid] = done;
          Serializer rez;
          rez.serialize(tid);
          rez.serialize(done);
          rez.serialize(runtime->address_space);
          runtime->send_top_level_region_request(owner, rez);
          wait_on = done;
        }
        else
          wait_on = req_finder->second;
      }
      wait_on.wait();
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<RegionTreeID,RegionNode*>::const_iterator finder = 
          tree_nodes.find(tid);
      if (finder == tree_nodes.end())
      {
        if (can_fail)
          return NULL;
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_TOPLEVEL_TREE,
          "Unable to find top-level tree entry for "
                         "region tree %d.  This is either a runtime "
                         "bug or requires Legion fences if names are "
                         "being returned out of the context in which"
                         "they are being created.", tid)
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::find_or_request_node(IndexSpace space,
                                                   AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/); 
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator finder = 
          index_nodes.find(space);
        if (finder != index_nodes.end())
          return RtEvent::NO_RT_EVENT;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpace owner = IndexSpaceNode::get_owner_space(space, runtime);
      if (owner == runtime->address_space)
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for index space %x.", space.id)
      AutoLock l_lock(lookup_lock);
      // Check to make sure we didn't loose the race
      std::map<IndexSpace,IndexSpaceNode*>::const_iterator finder = 
        index_nodes.find(space);
      if (finder != index_nodes.end())
        return RtEvent::NO_RT_EVENT;
      // Still doesn't exists, see if we sent a request already
      std::map<IndexSpace,RtEvent>::const_iterator wait_finder = 
        index_space_requests.find(space);
      if (wait_finder == index_space_requests.end())
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        index_space_requests[space] = done;
        Serializer rez;
        rez.serialize(space);
        rez.serialize(done);
        rez.serialize(runtime->address_space);
        runtime->send_index_space_request(target, rez);
        return done;
      }
      else
        return wait_finder->second;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      return get_node(space)->has_field(fid);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      std::map<IndexSpace,IndexSpaceNode*>::iterator finder = 
        index_nodes.find(space);
      assert(finder != index_nodes.end());
      index_nodes.erase(finder);
#else
      index_nodes.erase(space);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_node(IndexPartition part)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      assert(index_part_requests.find(part) == index_part_requests.end());
      std::map<IndexPartition,IndexPartNode*>::iterator finder = 
        index_parts.find(part);
      assert(finder != index_parts.end());
      index_parts.erase(finder);
#else
      index_parts.erase(part);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_node(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      std::map<FieldSpace,FieldSpaceNode*>::iterator finder = 
        field_nodes.find(space);
      assert(finder != field_nodes.end());
      field_nodes.erase(finder);
#else
      field_nodes.erase(space);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_node(LogicalRegion handle, bool top)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      if (top)
      {
        std::map<RegionTreeID,RegionNode*>::iterator finder = 
          tree_nodes.find(handle.get_tree_id());
        assert(finder != tree_nodes.end());
        tree_nodes.erase(finder);
      }
#else
      if (top)
        tree_nodes.erase(handle.get_tree_id());
#endif
      std::map<LogicalRegion,RegionNode*>::iterator finder = 
        region_nodes.find(handle);
#ifdef DEBUG_LEGION
      assert(finder != region_nodes.end());
#endif
      region_nodes.erase(finder);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
      std::map<LogicalPartition,PartitionNode*>::iterator finder = 
        part_nodes.find(handle);
#ifdef DEBUG_LEGION
      assert(finder != part_nodes.end());
#endif
      part_nodes.erase(finder);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::record_pending_index_space(IndexSpaceID space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // We should be the owner for this space
      assert((space % runtime->total_address_spaces) == runtime->address_space);
#endif
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      assert(pending_index_spaces.find(space) == pending_index_spaces.end());
#endif
      pending_index_spaces[space] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::record_pending_partition(IndexPartitionID pid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // We should be the owner for this space
      assert((pid % runtime->total_address_spaces) == runtime->address_space);
#endif
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      assert(pending_partitions.find(pid) == pending_partitions.end());
#endif
      pending_partitions[pid] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::record_pending_field_space(FieldSpaceID space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // We should be the owner for this space
      assert((space % runtime->total_address_spaces) == runtime->address_space);
#endif
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      assert(pending_field_spaces.find(space) == pending_field_spaces.end());
#endif
      pending_field_spaces[space] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::record_pending_region_tree(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // We should be the owner for this space
      assert((tid % runtime->total_address_spaces) == runtime->address_space);
#endif
      AutoLock l_lock(lookup_lock);
#ifdef DEBUG_LEGION
      assert(pending_region_trees.find(tid) == pending_region_trees.end());
#endif
      pending_region_trees[tid] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::revoke_pending_index_space(IndexSpaceID space)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpaceID,RtUserEvent>::iterator finder = 
          pending_index_spaces.find(space);
#ifdef DEBUG_LEGION
        assert(finder != pending_index_spaces.end());
#endif
        to_trigger = finder->second;
        pending_index_spaces.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::revoke_pending_partition(IndexPartitionID pid)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartitionID,RtUserEvent>::iterator finder = 
          pending_partitions.find(pid);
#ifdef DEBUG_LEGION
        assert(finder != pending_partitions.end());
#endif
        to_trigger = finder->second;
        pending_partitions.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::revoke_pending_field_space(FieldSpaceID space)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpaceID,RtUserEvent>::iterator finder = 
          pending_field_spaces.find(space);
#ifdef DEBUG_LEGION
        assert(finder != pending_field_spaces.end());
#endif
        to_trigger = finder->second;
        pending_field_spaces.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::revoke_pending_region_tree(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<RegionTreeID,RtUserEvent>::iterator finder = 
          pending_region_trees.find(tid);
#ifdef DEBUG_LEGION
        assert(finder != pending_region_trees.end());
#endif
        to_trigger = finder->second;
        pending_region_trees.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_top_level_index_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return (get_node(handle)->parent == NULL);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_top_level_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return (get_node(handle)->parent == NULL);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_subregion(LogicalRegion child, 
                                        LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      if (child == parent)
        return true;
      if (child.get_tree_id() != parent.get_tree_id())
        return false;
      return has_index_path(parent.get_index_space(), child.get_index_space());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_subregion(LogicalRegion child,
                                        LogicalPartition parent)
    //--------------------------------------------------------------------------
    {
      if (child.get_tree_id() != parent.get_tree_id())
        return false;
      RegionNode *child_node = get_node(child);
      PartitionNode *parent_node = get_node(parent);
      while (child_node->parent != NULL)
      {
        if (child_node->parent == parent_node)
          return true;
        child_node = child_node->parent->parent;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_disjoint(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      return node->is_disjoint(true/*app query*/);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_disjoint(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return is_disjoint(handle.get_index_partition());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_disjoint(IndexSpace one, IndexSpace two)
    //--------------------------------------------------------------------------
    {
      if (one == two)
        return false;
      if (one.get_tree_id() != two.get_tree_id())
        return true;
      // See if they intersect with each other
      IndexSpaceNode *sp_one = get_node(one);
      IndexSpaceNode *sp_two = get_node(two);
      return !sp_one->intersects_with(sp_two);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_disjoint(IndexSpace one, IndexPartition two)
    //--------------------------------------------------------------------------
    {
      if (one.get_tree_id() != two.get_tree_id())
        return true;
      IndexSpaceNode *space_node = get_node(one);
      IndexPartNode *part_node = get_node(two);
      return !space_node->intersects_with(part_node);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_disjoint(IndexPartition one, IndexPartition two)
    //--------------------------------------------------------------------------
    {
      if (one == two)
        return false;
      if (one.get_tree_id() != two.get_tree_id())
        return true;
      IndexPartNode *part_one = get_node(one);
      IndexPartNode *part_two = get_node(two);
      return !part_one->intersects_with(part_two);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_disjoint_tree_only(IndexTreeNode *one,
                            IndexTreeNode *two, IndexTreeNode *&common_ancestor)
    //--------------------------------------------------------------------------
    {
      if (one == two)
      {
        common_ancestor = one;
        return false;
      }
      // Some older code still relies on us being able to prove that two index
      // spaces are non-interfering with each other without using the tree so
      // we still check that even if we can't prove it with just the tree
      IndexSpaceNode *original_one = NULL, *original_two = NULL;
      if (one->is_index_space_node())
        original_one = one->as_index_space_node();
      if (two->is_index_space_node())
        original_two = two->as_index_space_node();
      // Bring them to the same depth
      while (one->depth < two->depth)
        two = two->get_parent();
      while (two->depth < one->depth)
        one = one->get_parent();
#ifdef DEBUG_LEGION
      assert(one->depth == two->depth);
#endif
      // Test again
      if (one == two)
      {
        common_ancestor = one;
        return false;
      }
      // Same depth, not the same node
      IndexTreeNode *parent_one = one->get_parent();
      IndexTreeNode *parent_two = two->get_parent();
      while (parent_one != parent_two)
      {
        one = parent_one;
        parent_one = one->get_parent();
        two = parent_two;
        parent_two = two->get_parent();
      }
#ifdef DEBUG_LEGION
      assert(parent_one == parent_two);
      assert(one != two); // can't be the same child
#endif
      // Now we have the common ancestor, see if the two children are disjoint
      if (parent_one->is_index_space_node())
      {
        if (parent_one->as_index_space_node()->are_disjoint(one->color, 
                                                            two->color))
          return true;
      }
      else
      {
        if (parent_one->as_index_part_node()->are_disjoint(one->color, 
                                                           two->color))
          return true;
      }
      // Test if two index spaces are interfering without using the tree
      if ((original_one != NULL) && (original_two != NULL))
      {
        IndexSpaceExpression *intersection =
          intersect_index_spaces(original_one, original_two);
        if (intersection->is_empty())
          return true;
      }
      common_ancestor = parent_one;
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::check_types(TypeTag t1, TypeTag t2, bool &diff_dims)
    //--------------------------------------------------------------------------
    {
      if (t1 == t2)
        return true;
      const int d1 = NT_TemplateHelper::get_dim(t1); 
      const int d2 = NT_TemplateHelper::get_dim(t2);
      diff_dims = (d1 != d2);
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_dominated(IndexSpace src, IndexSpace dst)
    //--------------------------------------------------------------------------
    {
      // Check to see if dst is dominated by source
#ifdef DEBUG_LEGION
      assert(src.get_type_tag() == dst.get_type_tag());
#endif
      IndexSpaceNode *src_node = get_node(src);
      IndexSpaceNode *dst_node = get_node(dst);
      return src_node->dominates(dst_node);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_dominated_tree_only(IndexSpace test,
                                                  IndexPartition dominator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(test.get_tree_id() == dominator.get_tree_id());
#endif
      IndexSpaceNode *node = get_node(test);
      IndexPartNode *const dom = get_node(dominator);
      while (node->depth > (dom->depth + 1))
      {
#ifdef DEBUG_LEGION
        assert(node->parent != NULL);
#endif
        node = node->parent->parent;
      }
      if (node->parent == dom)
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_dominated_tree_only(IndexPartition test,
                                                  IndexSpace dominator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(test.get_tree_id() == dominator.get_tree_id());
#endif
      IndexPartNode *node = get_node(test);
      IndexSpaceNode *const dom = get_node(dominator);
      while (node->depth > (dom->depth + 1))
      {
#ifdef DEBUG_LEGION
        assert(node->parent != NULL);
#endif
        node = node->parent->parent;
      }
      if (node->parent == dom)
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_dominated_tree_only(IndexPartition test,
                                                  IndexPartition dominator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(test.get_tree_id() == dominator.get_tree_id());
#endif
      IndexPartNode *node = get_node(test);
      IndexPartNode *const dom = get_node(dominator);
      while (node->depth > dom->depth)
      {
#ifdef DEBUG_LEGION
        assert(node->parent != NULL);
#endif
        node = node->parent->parent;
      }
      if (node == dom)
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_index_path(IndexSpace parent, 
                                          IndexSpace child)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(child); 
      if (parent == child) 
        return true; // Early out
      IndexSpaceNode *parent_node = get_node(parent);
      while (parent_node != child_node)
      {
        if (parent_node->depth >= child_node->depth)
          return false;
        if (child_node->parent == NULL)
          return false;
        child_node = child_node->parent->parent;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_partition_path(IndexSpace parent, 
                                              IndexPartition child)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *child_node = get_node(child);
      if (child_node->parent == NULL)
        return false;
      return has_index_path(parent, child_node->parent->handle);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexTreeNode *child,
                                           IndexTreeNode *parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(child->depth >= parent->depth);
#endif
      path.initialize(parent->depth, child->depth);
      while (child != parent)
      {
#ifdef DEBUG_LEGION
        assert(child->depth > 0);
#endif
        path.register_child(child->depth-1,child->color);
        child = child->get_parent();
      }
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    unsigned RegionTreeForest::get_projection_depth(LogicalRegion result,
                                                    LogicalRegion upper)
    //--------------------------------------------------------------------------
    {
      RegionNode *start = get_node(result);
      RegionNode *finish = get_node(upper);
      unsigned depth = 0;
      while (start != finish)
      {
        assert(start->get_depth() > finish->get_depth());
        start = start->parent->parent;
        depth++;
      }
      return depth;
    }

    //--------------------------------------------------------------------------
    unsigned RegionTreeForest::get_projection_depth(LogicalRegion result,
                                                    LogicalPartition upper)
    //--------------------------------------------------------------------------
    {
      RegionNode *start = get_node(result);
      assert(start->parent != NULL);
      PartitionNode *finish = get_node(upper);
      unsigned depth = 0;
      while (start->parent != finish)
      {
        assert(start->parent->get_depth() > finish->get_depth());
        start = start->parent->parent;
        depth++;
        assert(start->parent != NULL);
      }
      return depth;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::dump_logical_state(LogicalRegion region,
                                              ContextID ctx)
    //--------------------------------------------------------------------------
    {
      TreeStateLogger dump_logger; 
      assert(region_nodes.find(region) != region_nodes.end());
      region_nodes[region]->dump_logical_context(ctx, &dump_logger,
                                 FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES));
    }
#endif

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(IndexSpace handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size, 
                                                       bool is_mutable,
                                                       bool local_only)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                          size, is_mutable, local_only);
      if (runtime->legion_spy_enabled && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_index_space_name(handle.id, ptr);
      }
      if ((implicit_profiler != NULL) && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
	implicit_profiler->register_index_space(handle.id, ptr);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(IndexPartition handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable,
                                                       bool local_only)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                          size, is_mutable, local_only);
      if (runtime->legion_spy_enabled && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_index_partition_name(handle.id, ptr);
      }
      if ((implicit_profiler != NULL) && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
	implicit_profiler->register_index_part(handle.id, ptr);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(FieldSpace handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable,
                                                       bool local_only)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                          size, is_mutable, local_only);
      if (runtime->legion_spy_enabled && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_field_space_name(handle.id, ptr);
      }
      if ((implicit_profiler != NULL) && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
	implicit_profiler->register_field_space(handle.id, ptr);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(FieldSpace handle,
                                                       FieldID fid,
                                                       SemanticTag tag,
                                                       AddressSpaceID src,
                                                       const void *buf,
                                                       size_t size,
                                                       bool is_mutable,
                                                       bool local_only)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(fid, tag, src, buf, 
                                          size, is_mutable, local_only);
      if (runtime->legion_spy_enabled && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buf) == sizeof(ptr));
        memcpy(&ptr, &buf, sizeof(ptr));
        LegionSpy::log_field_name(handle.id, fid, ptr);
      }
      if ((implicit_profiler != NULL) && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buf) == sizeof(ptr));
        memcpy(&ptr, &buf, sizeof(ptr));
	implicit_profiler->register_field(handle.id, fid, size, ptr); 
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(LogicalRegion handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable,
                                                       bool local_only)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                          size, is_mutable, local_only);
      if (runtime->legion_spy_enabled && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_logical_region_name(handle.index_space.id,
            handle.field_space.id, handle.tree_id, ptr);
      }
      if ((implicit_profiler != NULL) && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
	implicit_profiler->register_logical_region(handle.index_space.id,
            handle.field_space.id, handle.tree_id, ptr);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(LogicalPartition handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable,
                                                       bool local_only)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                          size, is_mutable, local_only);
      if (runtime->legion_spy_enabled && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr));
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_logical_partition_name(handle.index_partition.id,
            handle.field_space.id, handle.tree_id, ptr);
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::retrieve_semantic_information(IndexSpace handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->retrieve_semantic_information(tag, result, size,
                                                          can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::retrieve_semantic_information(IndexPartition handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->retrieve_semantic_information(tag, result, size,
                                                          can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::retrieve_semantic_information(FieldSpace handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->retrieve_semantic_information(tag, result, size,
                                                         can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::retrieve_semantic_information(FieldSpace handle,
                                                         FieldID fid,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->retrieve_semantic_information(fid, tag, result, 
                                                  size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::retrieve_semantic_information(LogicalRegion handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->retrieve_semantic_information(tag, result, size,
                                                         can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::retrieve_semantic_information(LogicalPartition part,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return get_node(part)->retrieve_semantic_information(tag, result, size,
                                                         can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::union_index_spaces(
                           IndexSpaceExpression *lhs, IndexSpaceExpression *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs->type_tag == rhs->type_tag);
      assert(lhs->is_valid());
      assert(rhs->is_valid());
#endif
      if (lhs == rhs)
        return lhs;
      if (lhs->is_empty())
        return rhs;
      if (rhs->is_empty())
        return lhs;
      IndexSpaceExpression *lhs_canon = lhs->get_canonical_expression(this);
      IndexSpaceExpression *rhs_canon = rhs->get_canonical_expression(this);
      if (lhs_canon == rhs_canon)
        return lhs;
      std::vector<IndexSpaceExpression*> exprs(2);
      if (compare_expressions(lhs_canon, rhs_canon))
      {
        exprs[0] = lhs_canon;
        exprs[1] = rhs_canon;
      }
      else
      {
        exprs[0] = rhs_canon;
        exprs[1] = lhs_canon;
      }
      return union_index_spaces(exprs);
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::union_index_spaces(
                                   const std::set<IndexSpaceExpression*> &exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!exprs.empty());
      for (std::set<IndexSpaceExpression*>::const_iterator it =
            exprs.begin(); it != exprs.end(); it++)
        assert((*it)->is_valid());
#endif
      if (exprs.size() == 1)
        return *(exprs.begin());
      std::vector<IndexSpaceExpression*> expressions;
      expressions.reserve(exprs.size());
      for (std::set<IndexSpaceExpression*>::const_iterator it = 
            exprs.begin(); it != exprs.end(); it++)
      {
        // Remove any empty expressions on the way in
        if (!(*it)->is_empty())
          expressions.push_back((*it)->get_canonical_expression(this));
      }
      if (expressions.empty())
        return *(exprs.begin());
      if (expressions.size() == 1)
      {
        IndexSpaceExpression *result = expressions.back();
        if (exprs.find(result) == exprs.end())
        {
          result->add_base_expression_reference(LIVE_EXPR_REF);
          ImplicitReferenceTracker::record_live_expression(result);
        }
        return result;
      }
      // sort them in order by their IDs
      std::sort(expressions.begin(), expressions.end(), compare_expressions);
      // remove duplicates
      std::vector<IndexSpaceExpression*>::iterator last =
        std::unique(expressions.begin(), expressions.end());
      if (last != expressions.end())
      {
        expressions.erase(last, expressions.end());
#ifdef DEBUG_LEGION
        assert(!expressions.empty());
#endif
        if (expressions.size() == 1)
        {
          IndexSpaceExpression *result = expressions.back();
          if (exprs.find(result) == exprs.end())
          {
            result->add_base_expression_reference(LIVE_EXPR_REF);
            ImplicitReferenceTracker::record_live_expression(result);
          }
          return expressions.back();
        }
      }
      bool first_pass = true;
      // this helps make sure we don't overflow our stack
      while (expressions.size() > MAX_EXPRESSION_FANOUT)
      {
        std::vector<IndexSpaceExpression*> next_expressions;
        while (!expressions.empty())
        {
          if (expressions.size() > 1)
          {
            std::vector<IndexSpaceExpression*> temp_expressions;
            temp_expressions.reserve(MAX_EXPRESSION_FANOUT);
            // Pop up to 32 expressions off the back
            for (unsigned idx = 0; idx < MAX_EXPRESSION_FANOUT; idx++)
            {
              temp_expressions.push_back(expressions.back());
              expressions.pop_back();
              if (expressions.empty())
                break;
            }
            IndexSpaceExpression *expr = union_index_spaces(temp_expressions);
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.push_back(expr);
          }
          else
          {
            IndexSpaceExpression *expr = expressions.back();
            expressions.pop_back();
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.push_back(expr);
          }
        }
        if (!first_pass)
        {
          // Remove the expression references on the previous set
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                expressions.begin(); it != expressions.end(); it++)
            if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
              delete (*it);
        }
        else
          first_pass = false;
        expressions.swap(next_expressions);
        // canonicalize and uniquify them all again
        std::set<IndexSpaceExpression*,CompareExpressions> unique_expressions;
        for (unsigned idx = 0; idx < expressions.size(); idx++)
        {
          IndexSpaceExpression *expr = expressions[idx];
          IndexSpaceExpression *unique = expr->get_canonical_expression(this);
          if (unique_expressions.insert(unique).second)
            unique->add_base_expression_reference(REGION_TREE_REF);
        }
        // Remove the expression references
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              expressions.begin(); it != expressions.end(); it++)
          if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
            delete (*it);
        if (unique_expressions.size() == 1)
        {
          IndexSpaceExpression *result = *(unique_expressions.begin());
          if (exprs.find(result) == exprs.end())
          {
            result->add_base_expression_reference(LIVE_EXPR_REF);
            ImplicitReferenceTracker::record_live_expression(result);
          }
          // Remove the extra expression reference we added
          if (result->remove_base_expression_reference(REGION_TREE_REF))
            assert(false); // should never hit this
          return result; 
        }
        expressions.resize(unique_expressions.size());
        unsigned index = 0;
        for (std::set<IndexSpaceExpression*,CompareExpressions>::const_iterator
              it = unique_expressions.begin(); 
              it != unique_expressions.end(); it++)
          expressions[index++] = *it;
      }
      IndexSpaceExpression *result = union_index_spaces(expressions);
      if (exprs.find(result) == exprs.end())
      {
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
      }
      if (!first_pass)
      {
        // Remove the extra references on the expression vector we added
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              expressions.begin(); it != expressions.end(); it++)
          if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
            delete (*it);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::union_index_spaces(
                          const std::vector<IndexSpaceExpression*> &expressions,
                          OperationCreator *creator /*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expressions.size() >= 2);
      assert(expressions.size() <= MAX_EXPRESSION_FANOUT);
#endif
      IndexSpaceExpression *first = expressions[0];
      const IndexSpaceExprID key = first->expr_id;
      // See if we can find it in read-only mode
      {
        AutoLock l_lock(lookup_is_op_lock,1,false/*exclusive*/);
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
          finder = union_ops.find(key);
        if (finder != union_ops.end())
        {
          IndexSpaceExpression *result = NULL;
          ExpressionTrieNode *next = NULL;
          if (finder->second->find_operation(expressions, result, next) &&
              result->try_add_live_reference())
            return result;
          if (creator == NULL)
          {
            UnionOpCreator union_creator(this, first->type_tag, expressions);
            return next->find_or_create_operation(expressions, union_creator);
          }
          else
            return next->find_or_create_operation(expressions, *creator);
        }
      }
      ExpressionTrieNode *node = NULL;
      if (creator == NULL)
      {
        UnionOpCreator union_creator(this, first->type_tag, expressions);
        // Didn't find it, retake the lock, see if we lost the race
        // and if no make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
          finder = union_ops.find(key);
        if (finder == union_ops.end())
        {
          // Didn't lose the race, so make the node
          node = new ExpressionTrieNode(0/*depth*/, first->expr_id);
          union_ops[key] = node;
        }
        else
          node = finder->second;
#ifdef DEBUG_LEGION
        assert(node != NULL);
#endif
        return node->find_or_create_operation(expressions, union_creator);
      }
      else
      {
        // Didn't find it, retake the lock, see if we lost the race
        // and if no make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
          finder = union_ops.find(key);
        if (finder == union_ops.end())
        {
          // Didn't lose the race, so make the node
          node = new ExpressionTrieNode(0/*depth*/, first->expr_id);
          union_ops[key] = node;
        }
        else
          node = finder->second;
#ifdef DEBUG_LEGION
        assert(node != NULL);
#endif
        return node->find_or_create_operation(expressions, *creator);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::intersect_index_spaces(
                           IndexSpaceExpression *lhs, IndexSpaceExpression *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs->type_tag == rhs->type_tag);
      assert(lhs->is_valid());
      assert(rhs->is_valid());
#endif
      if (lhs == rhs)
        return lhs;
      if (lhs->is_empty())
        return lhs;
      if (rhs->is_empty())
        return rhs;
      IndexSpaceExpression *lhs_canon = lhs->get_canonical_expression(this);
      IndexSpaceExpression *rhs_canon = rhs->get_canonical_expression(this);
      if (lhs_canon == rhs_canon)
        return lhs;
      std::vector<IndexSpaceExpression*> exprs(2);
      if (compare_expressions(lhs_canon, rhs_canon))
      {
        exprs[0] = lhs_canon;
        exprs[1] = rhs_canon;
      }
      else
      {
        exprs[0] = rhs_canon;
        exprs[1] = lhs_canon;
      }
      return intersect_index_spaces(exprs);
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::intersect_index_spaces(
                                   const std::set<IndexSpaceExpression*> &exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!exprs.empty());
      for (std::set<IndexSpaceExpression*>::const_iterator it =
            exprs.begin(); it != exprs.end(); it++)
        assert((*it)->is_valid());
#endif
      if (exprs.size() == 1)
        return *(exprs.begin());
      std::vector<IndexSpaceExpression*> expressions(exprs.begin(),exprs.end());
      // Do a quick pass to see if any of them are empty in which case we
      // know that the result of the whole intersection is empty
      for (unsigned idx = 0; idx < expressions.size(); idx++)
      {
        IndexSpaceExpression *&expr = expressions[idx];
        if (expr->is_empty())
          return expr;
        expr = expr->get_canonical_expression(this);
      }
      // sort them in order by their IDs
      std::sort(expressions.begin(), expressions.end(), compare_expressions);
      // remove duplicates
      std::vector<IndexSpaceExpression*>::iterator last =
        std::unique(expressions.begin(), expressions.end());
      if (last != expressions.end())
      {
        expressions.erase(last, expressions.end());
#ifdef DEBUG_LEGION
        assert(!expressions.empty());
#endif
        if (expressions.size() == 1)
        {
          IndexSpaceExpression *result = expressions.back();
          if (exprs.find(result) == exprs.end())
          {
            result->add_base_expression_reference(LIVE_EXPR_REF);
            ImplicitReferenceTracker::record_live_expression(result);
          }
          return result;
        }
      }
      bool first_pass = true;
      // this helps make sure we don't overflow our stack
      while (expressions.size() > MAX_EXPRESSION_FANOUT)
      {
        std::vector<IndexSpaceExpression*> next_expressions;
        while (!expressions.empty())
        {
          if (expressions.size() > 1)
          {
            std::vector<IndexSpaceExpression*> temp_expressions;
            temp_expressions.reserve(MAX_EXPRESSION_FANOUT);
            // Pop up to 32 expressions off the back
            for (unsigned idx = 0; idx < MAX_EXPRESSION_FANOUT; idx++)
            {
              temp_expressions.push_back(expressions.back());
              expressions.pop_back();
              if (expressions.empty())
                break;
            }
            IndexSpaceExpression *expr =
              intersect_index_spaces(temp_expressions);
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.push_back(expr);
          }
          else
          {
            IndexSpaceExpression *expr = expressions.back();
            expressions.pop_back();
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.push_back(expr);
          }
        }
        if (!first_pass)
        {
          // Remove the expression references on the previous set
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                expressions.begin(); it != expressions.end(); it++)
            if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
              delete (*it);
        }
        else
          first_pass = false;
        expressions.swap(next_expressions);
        // canonicalize and uniquify them all again
        std::set<IndexSpaceExpression*,CompareExpressions> unique_expressions;
        for (unsigned idx = 0; idx < expressions.size(); idx++)
        {
          IndexSpaceExpression *expr = expressions[idx];
          IndexSpaceExpression *unique = expr->get_canonical_expression(this);
          if (unique->is_empty())
          {
            // Add a reference to the unique expression
            if (exprs.find(unique) == exprs.end())
            {
              unique->add_base_expression_reference(LIVE_EXPR_REF);
              ImplicitReferenceTracker::record_live_expression(unique);
            }
            // Remove references on all the things we no longer need
            for (std::set<IndexSpaceExpression*,CompareExpressions>::
                  const_iterator it = unique_expressions.begin(); it !=
                  unique_expressions.end(); it++)
              if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
                delete (*it);
            for (std::vector<IndexSpaceExpression*>::const_iterator it =
                  expressions.begin(); it != expressions.end(); it++)
              if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
                delete (*it);
            return unique;
          }
          if (unique_expressions.insert(unique).second)
            unique->add_base_expression_reference(REGION_TREE_REF);
        }
        // Remove the expression references
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              expressions.begin(); it != expressions.end(); it++)
          if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
            delete (*it);
        if (unique_expressions.size() == 1)
        {
          IndexSpaceExpression *result = *(unique_expressions.begin());
          if (exprs.find(result) == exprs.end())
          {
            result->add_base_expression_reference(LIVE_EXPR_REF);
            ImplicitReferenceTracker::record_live_expression(result);
          }
          // Remove the extra expression reference we added
          if (result->remove_base_expression_reference(REGION_TREE_REF))
            assert(false); // should never hit this
          return result; 
        }
        expressions.resize(unique_expressions.size());
        unsigned index = 0;
        for (std::set<IndexSpaceExpression*,CompareExpressions>::const_iterator
              it = unique_expressions.begin(); 
              it != unique_expressions.end(); it++)
          expressions[index++] = *it;
      }
      IndexSpaceExpression *result = intersect_index_spaces(expressions);
      if (exprs.find(result) == exprs.end())
      {
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
      }
      if (!first_pass)
      {
        // Remove the extra references on the expression vector we added
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              expressions.begin(); it != expressions.end(); it++)
          if ((*it)->remove_base_expression_reference(REGION_TREE_REF))
            delete (*it);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::intersect_index_spaces(
                          const std::vector<IndexSpaceExpression*> &expressions,
                          OperationCreator *creator /*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expressions.size() >= 2);
      assert(expressions.size() <= MAX_EXPRESSION_FANOUT);
#endif
      IndexSpaceExpression *first = expressions[0];
      const IndexSpaceExprID key = first->expr_id;
      // See if we can find it in read-only mode
      {
        AutoLock l_lock(lookup_is_op_lock,1,false/*exclusive*/);
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
          finder = intersection_ops.find(key);
        if (finder != intersection_ops.end())
        {
          IndexSpaceExpression *result = NULL;
          ExpressionTrieNode *next = NULL;
          if (finder->second->find_operation(expressions, result, next) &&
              result->try_add_live_reference())
            return result;
          if (creator == NULL)
          {
            IntersectionOpCreator inter_creator(this, first->type_tag, 
                                                expressions);
            return next->find_or_create_operation(expressions, inter_creator);
          }
          else
            return next->find_or_create_operation(expressions, *creator);
        }
      }
      ExpressionTrieNode *node = NULL;
      if (creator == NULL)
      {
        IntersectionOpCreator inter_creator(this, first->type_tag, expressions);
        // Didn't find it, retake the lock, see if we lost the race
        // and if not make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        // See if we lost the race
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
          finder = intersection_ops.find(key);
        if (finder == intersection_ops.end())
        {
          // Didn't lose the race so make the node
          node = new ExpressionTrieNode(0/*depth*/, first->expr_id);
          intersection_ops[key] = node;
        }
        else
          node = finder->second;
#ifdef DEBUG_LEGION
        assert(node != NULL); 
#endif
        return node->find_or_create_operation(expressions, inter_creator);
      }
      else
      {
        // Didn't find it, retake the lock, see if we lost the race
        // and if not make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        // See if we lost the race
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
          finder = intersection_ops.find(key);
        if (finder == intersection_ops.end())
        {
          // Didn't lose the race so make the node
          node = new ExpressionTrieNode(0/*depth*/, first->expr_id);
          intersection_ops[key] = node;
        }
        else
          node = finder->second;
#ifdef DEBUG_LEGION
        assert(node != NULL); 
#endif
        return node->find_or_create_operation(expressions, *creator);
      }
    }
    
    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::subtract_index_spaces(
                           IndexSpaceExpression *lhs, IndexSpaceExpression *rhs,
                           OperationCreator *creator/*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs->type_tag == rhs->type_tag);
      assert(lhs->is_valid());
      assert(rhs->is_valid());
#endif
      // Handle a few easy cases
      if (creator == NULL)
      {
        if (lhs->is_empty())
          return lhs;
        if (rhs->is_empty())
          return lhs;
      }
      std::vector<IndexSpaceExpression*> expressions(2);
      expressions[0] = lhs->get_canonical_expression(this);
      expressions[1] = rhs->get_canonical_expression(this);
      const IndexSpaceExprID key = expressions[0]->expr_id;
      // See if we can find it in read-only mode
      IndexSpaceExpression *result = NULL;
      {
        AutoLock l_lock(lookup_is_op_lock,1,false/*exclusive*/);
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
          finder = difference_ops.find(key);
        if (finder != difference_ops.end())
        {
          IndexSpaceExpression *expr = NULL;
          ExpressionTrieNode *next = NULL;
          if (finder->second->find_operation(expressions, expr, next) &&
              expr->try_add_live_reference())
            result = expr;
          if (result == NULL)
          {
            if (creator == NULL)
            {
              DifferenceOpCreator diff_creator(this, lhs->type_tag, 
                                    expressions[0], expressions[1]);
              result = next->find_or_create_operation(expressions,diff_creator);
            }
            else
              result = next->find_or_create_operation(expressions, *creator);
          }
        }
      }
      if (result == NULL)
      {
        ExpressionTrieNode *node = NULL;
        if (creator == NULL)
        {
          DifferenceOpCreator diff_creator(this, lhs->type_tag,
                                expressions[0], expressions[1]);
          // Didn't find it, retake the lock, see if we lost the race
          // and if not make the actual trie node
          AutoLock l_lock(lookup_is_op_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
            finder = difference_ops.find(key);
          if (finder == difference_ops.end())
          {
            // Didn't lose the race so make the node
            node = new ExpressionTrieNode(0/*depth*/, expressions[0]->expr_id);
            difference_ops[key] = node;
          }
          else
            node = finder->second;
#ifdef DEBUG_LEGION
          assert(node != NULL);
#endif
          result = node->find_or_create_operation(expressions, diff_creator);
        }
        else
        {
          // Didn't find it, retake the lock, see if we lost the race
          // and if not make the actual trie node
          AutoLock l_lock(lookup_is_op_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator 
            finder = difference_ops.find(key);
          if (finder == difference_ops.end())
          {
            // Didn't lose the race so make the node
            node = new ExpressionTrieNode(0/*depth*/, expressions[0]->expr_id);
            difference_ops[key] = node;
          }
          else
            node = finder->second;
#ifdef DEBUG_LEGION
          assert(node != NULL);
#endif
          result = node->find_or_create_operation(expressions, *creator);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::find_canonical_expression(
                                                     IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
      // we'll hash expressions based on the number of dimensions and points
      // to try to get an early separation for them for testing congruence
      const size_t volume = expr->get_volume();
      if (volume == 0)
        return expr;
      const std::pair<size_t,TypeTag> key(volume, expr->type_tag);
      AutoLock c_lock(congruence_lock);
      return expr->find_congruent_expression(canonical_expressions[key]);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_canonical_expression(
                                      IndexSpaceExpression *expr, size_t volume)
    //--------------------------------------------------------------------------
    {
      // Nothing to do for empty expressions
      if (volume == 0)
        return;
      const std::pair<size_t,TypeTag> key(volume, expr->type_tag);
      AutoLock c_lock(congruence_lock);
      std::vector<IndexSpaceExpression*> &exprs = canonical_expressions[key];
      std::vector<IndexSpaceExpression*>::iterator finder =
        std::lower_bound(exprs.begin(), exprs.end(), expr);
#ifdef DEBUG_LEGION
      assert(finder != exprs.end());
      assert(*finder == expr);
#endif
      exprs.erase(finder);
      if (exprs.empty())
        canonical_expressions.erase(key);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_union_operation(IndexSpaceOperation *op,
                                const std::vector<IndexSpaceExpression*> &exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op->op_kind == IndexSpaceOperation::UNION_OP_KIND);
#endif
      const IndexSpaceExprID key = exprs[0]->expr_id;
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID,ExpressionTrieNode*>::iterator 
        finder = union_ops.find(key);
#ifdef DEBUG_LEGION
      assert(finder != union_ops.end());
#endif
      if (finder->second->remove_operation(exprs))
      {
        delete finder->second;
        union_ops.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_intersection_operation(
       IndexSpaceOperation *op, const std::vector<IndexSpaceExpression*> &exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op->op_kind == IndexSpaceOperation::INTERSECT_OP_KIND);
#endif
      const IndexSpaceExprID key(exprs[0]->expr_id);
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID,ExpressionTrieNode*>::iterator 
        finder = intersection_ops.find(key);
#ifdef DEBUG_LEGION
      assert(finder != intersection_ops.end());
#endif
      if (finder->second->remove_operation(exprs))
      {
        delete finder->second;
        intersection_ops.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_subtraction_operation(IndexSpaceOperation *op,
                           IndexSpaceExpression *lhs, IndexSpaceExpression *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op->op_kind == IndexSpaceOperation::DIFFERENCE_OP_KIND);
#endif
      const IndexSpaceExprID key = lhs->expr_id;
      std::vector<IndexSpaceExpression*> exprs(2);
      exprs[0] = lhs;
      exprs[1] = rhs;
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID,ExpressionTrieNode*>::iterator 
        finder = difference_ops.find(key);
#ifdef DEBUG_LEGION
      assert(finder != difference_ops.end());
#endif
      if (finder->second->remove_operation(exprs))
      {
        delete finder->second;
        difference_ops.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::find_or_request_remote_expression(
                                IndexSpaceExprID remote_expr_id, 
                                IndexSpaceExpression *origin, RtEvent *wait_for)
    //--------------------------------------------------------------------------
    {
      // See if we can find it with the read-only lock first
      {
        AutoLock l_lock(lookup_is_op_lock, 1, false/*exclusive*/);
        std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator 
          finder = remote_expressions.find(remote_expr_id);
        if (finder != remote_expressions.end())
          return finder->second;
      }
      const AddressSpaceID owner = 
          IndexSpaceExpression::get_owner_space(remote_expr_id, runtime);
#ifdef DEBUG_LEGION
      assert(owner != runtime->address_space);
#endif
      // Retake the lock in exclusive mode and see if we lost the race
      RtEvent wait_on;
      RtUserEvent request_event;
      {
        AutoLock l_lock(lookup_is_op_lock);
        std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator 
          finder = remote_expressions.find(remote_expr_id);
        if (finder != remote_expressions.end())
          return finder->second;
        // It doesn't exist yet so see if we need to request it from the owner
        std::map<IndexSpaceExprID,RtEvent>::const_iterator event_finder = 
          pending_remote_expressions.find(remote_expr_id);
        if (event_finder == pending_remote_expressions.end())
        {
          request_event = Runtime::create_rt_user_event();
          wait_on = request_event;
          pending_remote_expressions[remote_expr_id] = wait_on; 
        }
        else
          wait_on = event_finder->second;
      }
      // Send the request for the remote expression
      if (request_event.exists())
      { 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_expr_id);
          rez.serialize(origin);
          rez.serialize(request_event);
        }
        runtime->send_index_space_remote_expression_request(owner, rez);
      }
      if (wait_for == NULL)
      {
        wait_on.wait();
        // When we get the lock again it should be there
        AutoLock l_lock(lookup_is_op_lock, 1, false/*exclusive*/);
        std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator 
          finder = remote_expressions.find(remote_expr_id);
#ifdef DEBUG_LEGION
        assert(finder != remote_expressions.end());
#endif
        return finder->second;
      }
      else
      {
        *wait_for = wait_on;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::find_remote_expression(
                                         const PendingRemoteExpression &pending)
    //--------------------------------------------------------------------------
    {
      if (pending.is_index_space)
      {
        IndexSpaceNode *node = get_node(pending.handle);
        node->add_base_expression_reference(LIVE_EXPR_REF);
        if (!pending.done_ref_counting)
          node->unpack_global_ref();
        ImplicitReferenceTracker::record_live_expression(node);
        return node;
      }
      else
      {
        IndexSpaceExpression *result = NULL;
        {
          AutoLock l_lock(lookup_is_op_lock, 1, false/*exclusive*/);
          std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator 
            finder = remote_expressions.find(pending.remote_expr_id);
#ifdef DEBUG_LEGION
          assert(finder != remote_expressions.end());
#endif
          result = finder->second;
        }
#ifdef DEBUG_LEGION
        IndexSpaceOperation *op = dynamic_cast<IndexSpaceOperation*>(result);
        assert(op != NULL);
#else
        IndexSpaceOperation *op = static_cast<IndexSpaceOperation*>(result);
#endif
        result->add_base_expression_reference(LIVE_EXPR_REF);
        if (!pending.done_ref_counting)
          op->unpack_global_ref();
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unregister_remote_expression(
                                                IndexSpaceExprID remote_expr_id)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID,IndexSpaceExpression*>::iterator 
        finder = remote_expressions.find(remote_expr_id);
      if (finder != remote_expressions.end())
        remote_expressions.erase(finder);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::handle_remote_expression_request(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpaceExprID remote_expr_id;
      derez.deserialize(remote_expr_id);
      IndexSpaceExpression *origin;
      derez.deserialize(origin);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(remote_expr_id);
        origin->pack_expression_value(rez, source);
        rez.serialize(done_event);
      }
      runtime->send_index_space_remote_expression_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::handle_remote_expression_response(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpaceExprID remote_expr_id;
      derez.deserialize(remote_expr_id); 
      IndexSpaceExpression *result = unpack_expression_value(derez, source);
      {
        AutoLock l_lock(lookup_is_op_lock);
#ifdef DEBUG_LEGION
        assert(remote_expressions.find(remote_expr_id) == 
                remote_expressions.end());
        assert(pending_remote_expressions.find(remote_expr_id) !=
                pending_remote_expressions.end());
#endif
        remote_expressions[remote_expr_id] = result;
        pending_remote_expressions.erase(remote_expr_id);
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::unpack_expression_value(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // First see if this is a base case of a known index space
      bool is_index_space;
      derez.deserialize<bool>(is_index_space);
      if (is_index_space)
      {
        IndexSpace handle;
        derez.deserialize(handle);
        return get_node(handle);
      }
      TypeTag type_tag;
      derez.deserialize(type_tag);
      RemoteExpressionCreator creator(this, type_tag, derez);
      NT_TemplateHelper::demux<RemoteExpressionCreator>(type_tag, &creator);
#ifdef DEBUG_LEGION
      assert(creator.operation != NULL);
#endif
      return creator.operation;
    }

    /////////////////////////////////////////////////////////////
    // Copy Across Executor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyAcrossExecutor::DeferCopyAcrossArgs::DeferCopyAcrossArgs(
        CopyAcrossExecutor *e, Operation *o, PredEvent g, ApEvent copy_pre,
        ApEvent src_pre, ApEvent dst_pre, const PhysicalTraceInfo &info,
        bool repl, bool recurrent, unsigned s)
      : LgTaskArgs<DeferCopyAcrossArgs>(o->get_unique_op_id()),
        executor(e), op(o), trace_info(new PhysicalTraceInfo(info)), guard(g),
        copy_precondition(copy_pre), src_indirect_precondition(src_pre),
        dst_indirect_precondition(dst_pre), 
        done_event(Runtime::create_ap_user_event(trace_info)),
        stage(s+1), replay(repl), recurrent_replay(recurrent)
    //--------------------------------------------------------------------------
    {
      executor->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyAcrossExecutor::handle_deferred_copy_across(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCopyAcrossArgs *dargs = (const DeferCopyAcrossArgs*)args;
      Runtime::trigger_event(dargs->trace_info, dargs->done_event,
          dargs->executor->execute(dargs->op, dargs->guard, 
            dargs->copy_precondition, dargs->src_indirect_precondition, 
            dargs->dst_indirect_precondition, *dargs->trace_info,
            dargs->replay, dargs->recurrent_replay, dargs->stage));
      if (dargs->executor->remove_reference())
        delete dargs->executor;
      delete dargs->trace_info;
    }

    /////////////////////////////////////////////////////////////
    // Copy Across Unstructured
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_source_fields(
       RegionTreeForest *forest, const RegionRequirement &req,
       const InstanceSet &insts, const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_fields.empty());
#endif
      FieldSpaceNode *fs = forest->get_node(req.region.get_field_space());
      std::vector<unsigned> indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, indexes);
      src_fields.reserve(indexes.size());
      src_unique_events.reserve(indexes.size());
      for (std::vector<unsigned>::const_iterator it =
            indexes.begin(); it != indexes.end(); it++)
      {
#ifdef DEBUG_LEGION
        bool found = false;
#endif
        for (unsigned idx = 0; idx < insts.size(); idx++)
        {
          const InstanceRef &ref = insts[idx];
          const FieldMask &mask = ref.get_valid_fields();
          if (!mask.is_set(*it))
            continue;
          FieldMask copy_mask;
          copy_mask.set_bit(*it);
          PhysicalManager *manager = ref.get_physical_manager();
          manager->compute_copy_offsets(copy_mask, src_fields);
          src_unique_events.push_back(manager->get_unique_event());
#ifdef DEBUG_LEGION
          found = true;
#endif
          break;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_destination_fields(
                  RegionTreeForest *forest, const RegionRequirement &req,
                  const InstanceSet &insts, const PhysicalTraceInfo &trace_info,
                  const bool exclusive_redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dst_fields.empty());
#endif
      FieldSpaceNode *fs = forest->get_node(req.region.get_field_space());
      std::vector<unsigned> indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, indexes);
      dst_fields.reserve(indexes.size());
      dst_unique_events.reserve(indexes.size());
      for (std::vector<unsigned>::const_iterator it =
            indexes.begin(); it != indexes.end(); it++)
      {
#ifdef DEBUG_LEGION
        bool found = false;
#endif
        for (unsigned idx = 0; idx < insts.size(); idx++)
        {
          const InstanceRef &ref = insts[idx];
          const FieldMask &mask = ref.get_valid_fields();
          if (!mask.is_set(*it))
            continue;
          FieldMask copy_mask;
          copy_mask.set_bit(*it);
          PhysicalManager *manager = ref.get_physical_manager();
          manager->compute_copy_offsets(copy_mask, dst_fields);
          dst_unique_events.push_back(manager->get_unique_event());
#ifdef DEBUG_LEGION
          found = true;
#endif
          break;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      if (req.redop != 0)
      {
        for (unsigned idx = 0; idx < dst_fields.size(); idx++)
          dst_fields[idx].set_redop(req.redop, false/*fold*/, exclusive_redop);
      }
    }

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_source_indirections(
            RegionTreeForest *forest, std::vector<IndirectRecord> &records,
            const RegionRequirement &src_req, const RegionRequirement &idx_req,
            const InstanceRef &indirect_instance,
            const bool are_range, const bool possible_out_of_range)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_fields.empty());
      assert(idx_req.privilege_fields.size() == 1);
#endif
      src_indirections.swap(records);
      src_indirect_field = *(idx_req.privilege_fields.begin());
      PhysicalManager *manager = indirect_instance.get_physical_manager();
      src_indirect_instance = manager->get_instance();
      src_indirect_instance_event = manager->get_unique_event();
      src_indirect_type = src_req.region.get_index_space().get_type_tag();
      both_are_range = are_range;
      possible_src_out_of_range = possible_out_of_range;
      src_fields.resize(src_req.instance_fields.size());
      FieldSpaceNode *fs = forest->get_node(src_req.region.get_field_space());
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
      {
        const FieldID fid = src_req.instance_fields[idx];
        src_fields[idx].set_indirect(0/*dummy indirection for now*/,
                                     fid, fs->get_field_size(fid));
      }
    }

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_destination_indirections(
            RegionTreeForest *forest, std::vector<IndirectRecord> &records,
            const RegionRequirement &dst_req, const RegionRequirement &idx_req,
            const InstanceRef &indirect_instance,
            const bool are_range, const bool possible_out_of_range,
            const bool possible_aliasing, const bool exclusive_redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dst_fields.empty());
      assert(idx_req.privilege_fields.size() == 1);
#endif
      dst_indirections.swap(records);
      dst_indirect_field = *(idx_req.privilege_fields.begin());
      PhysicalManager *manager = indirect_instance.get_physical_manager();
      dst_indirect_instance = manager->get_instance();
      dst_indirect_instance_event = manager->get_unique_event();
      dst_indirect_type = dst_req.region.get_index_space().get_type_tag();
      both_are_range = are_range;
      possible_dst_out_of_range = possible_out_of_range;
      possible_dst_aliasing = possible_aliasing;
      dst_fields.resize(dst_req.instance_fields.size());
      FieldSpaceNode *fs = forest->get_node(dst_req.region.get_field_space());
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
      {
        const FieldID fid = dst_req.instance_fields[idx];
        dst_fields[idx].set_indirect(0/*dummy indirection for now*/,
                                     fid, fs->get_field_size(fid));
        if (dst_req.redop != 0)
          dst_fields[idx].set_redop(dst_req.redop, 
                    false/*fold*/, exclusive_redop);
      }
    }

    //--------------------------------------------------------------------------
    LgEvent CopyAcrossUnstructured::find_instance_name(
                                                PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      if (instance == src_indirect_instance)
        return src_indirect_instance_event;
      if (instance == dst_indirect_instance)
        return dst_indirect_instance_event;
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        if (src_fields[idx].inst == instance)
          return src_unique_events[idx];
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        if (dst_fields[idx].inst == instance)
          return dst_unique_events[idx];
      for (std::vector<IndirectRecord>::const_iterator it =
            src_indirections.begin(); it != src_indirections.end(); it++)
        for (unsigned idx = 0; idx < it->instances.size(); idx++)
          if (it->instances[idx] == instance)
            return it->instance_events[idx];
      for (std::vector<IndirectRecord>::const_iterator it =
            dst_indirections.begin(); it != dst_indirections.end(); it++)
        for (unsigned idx = 0; idx < it->instances.size(); idx++)
          if (it->instances[idx] == instance)
            return it->instance_events[idx];
      // Should always have found it before this
      assert(false);
      return src_indirect_instance_event;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Expression 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(LocalLock &lock)
      : type_tag(0), expr_id(0), expr_lock(lock), canonical(NULL),
        sparsity_map_kd_tree(NULL), volume(0), has_volume(false),
        empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(TypeTag tag, Runtime *rt,
                                               LocalLock &lock)
      : type_tag(tag), expr_id(rt->get_unique_index_space_expr_id()), 
        expr_lock(lock), canonical(NULL), sparsity_map_kd_tree(NULL),
        volume(0), has_volume(false), empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(TypeTag tag, IndexSpaceExprID id,
                                               LocalLock &lock)
      : type_tag(tag), expr_id(id), expr_lock(lock), canonical(NULL),
        sparsity_map_kd_tree(NULL), volume(0), has_volume(false),
        empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::~IndexSpaceExpression(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(derived_operations.empty());
#endif
      if (sparsity_map_kd_tree != NULL)
        delete sparsity_map_kd_tree;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceExpression::handle_tighten_index_space(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const TightenIndexSpaceArgs *targs = (const TightenIndexSpaceArgs*)args;
      targs->proxy_this->tighten_index_space();
      if (targs->proxy_dc->remove_base_resource_ref(META_TASK_REF))
        delete targs->proxy_this;
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID IndexSpaceExpression::get_owner_space(
                                     IndexSpaceExprID expr_id, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (expr_id % runtime->runtime_stride);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceExpression::add_derived_operation(IndexSpaceOperation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(expr_lock);
#ifdef DEBUG_LEGION
      assert(derived_operations.find(op) == derived_operations.end());
#endif
      derived_operations.insert(op);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceExpression::remove_derived_operation(IndexSpaceOperation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(expr_lock);
#ifdef DEBUG_LEGION
      assert(derived_operations.find(op) != derived_operations.end());
#endif
      derived_operations.erase(op);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceExpression::invalidate_derived_operations(DistributedID did,
                                                      RegionTreeForest *context)
    //--------------------------------------------------------------------------
    {
      // Traverse upwards for any derived operations and invalidate them
      std::vector<IndexSpaceOperation*> derived;
      {
        AutoLock e_lock(expr_lock,1,false/*exclusive*/);
        if (!derived_operations.empty())
        {
          derived.reserve(derived_operations.size());
          for (std::set<IndexSpaceOperation*>::const_iterator it = 
               derived_operations.begin(); it != derived_operations.end(); it++)
          {
            (*it)->add_nested_resource_ref(did);
            derived.push_back(*it);
          }
        }
      }
      if (!derived.empty())
      {
        for (std::vector<IndexSpaceOperation*>::const_iterator it = 
              derived.begin(); it != derived.end(); it++)
        {
          // Try to invalidate it and remove the tree reference if we did
          if ((*it)->invalidate_operation() &&
              (*it)->remove_base_gc_ref(REGION_TREE_REF))
            assert(false); // should never delete since we have a resource ref
          // Remove any references that we have on the parents
          if ((*it)->remove_nested_resource_ref(did))
            delete (*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceExpression::test_intersection_nonblocking(
                        IndexSpaceExpression *other, RegionTreeForest *context,
                        ApEvent &precondition, bool second)
    //--------------------------------------------------------------------------
    {
      if (second)
      {
        // We've got two non pending expressions, so we can just test them
        IndexSpaceExpression *overlap = 
          context->intersect_index_spaces(this, other);
        return !overlap->is_empty();
      }
      else
      {
        // First time through, we're not pending so keep going
        return other->test_intersection_nonblocking(this, context, 
                                        precondition, true/*second*/);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* IndexSpaceExpression::get_canonical_expression(
                                                       RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_valid());
#endif
      IndexSpaceExpression *expr = canonical.load();
      if (expr != NULL)
      {
        // If we're our own canonical expression then assume we're
        // still alive and don't need to add a live expression
        if (expr == this)
          return expr;
        // If we're not our own canonical expression, we need to make sure
        // that it is still alive and can be used
        if (expr->try_add_live_reference())
          return expr;
        // Fall through and compute a new canonical expression
      }
      expr = forest->find_canonical_expression(this);
      if (expr == this)
      {
        // If we're our own canonical expression then the forest didn't
        // give us a reference to ourself, but we do need to check to see
        // if we're the first one to write to see if we need to remove any
        // references from a prior expression
        IndexSpaceExpression *prev = canonical.exchange(expr);
        if ((prev != NULL) && (prev != expr))
        {
          const DistributedID did = get_distributed_id();
          if (prev->remove_canonical_reference(did))
            delete prev;
        }
        return expr;
      }
      // If the canonical expression is not ourself, then the region tree
      // forest has given us a live reference back on it so we know it
      // can't be collected, but we need to update the canonical result
      // and add a nested reference if we're the first ones to perform 
      // the update
      IndexSpaceExpression *prev = canonical.exchange(expr); 
      if (prev != expr)
      {
        const DistributedID did = get_distributed_id();
        // We're the first to store this result so remove the reference
        // from the previous one if it existed
        if ((prev != NULL) && prev->remove_canonical_reference(did))
          delete prev;
        // Add a nested resource reference for the new one
        expr->add_canonical_reference(did);
      }
      return expr;
    }

    //--------------------------------------------------------------------------
    /*static*/ IndexSpaceExpression* IndexSpaceExpression::unpack_expression(
                             Deserializer &derez, RegionTreeForest *forest, 
                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Handle the special case where this is a local index space expression 
      bool is_local;
      derez.deserialize(is_local);
      if (is_local)
      {
        IndexSpaceExpression *result;
        derez.deserialize(result);
        if (source != forest->runtime->address_space)
        {
#ifdef DEBUG_LEGION
          IndexSpaceOperation *op = 
            dynamic_cast<IndexSpaceOperation*>(result);
          assert(op != NULL);
#else
          IndexSpaceOperation *op = static_cast<IndexSpaceOperation*>(result);
#endif
          op->add_base_expression_reference(LIVE_EXPR_REF);
          op->unpack_global_ref();
        }
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      bool is_index_space;
      derez.deserialize(is_index_space);
      // If this is an index space it is easy
      if (is_index_space)
      {
        IndexSpace handle;
        derez.deserialize(handle);
        IndexSpaceNode *node = forest->get_node(handle);
        node->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(node);
        // Now we can unpack the global expression reference
        node->unpack_global_ref();
        return node;
      }
      else
      {
        IndexSpaceExprID remote_expr_id;
        derez.deserialize(remote_expr_id);
        IndexSpaceExpression *origin;
        derez.deserialize(origin);
        IndexSpaceExpression *result =
          forest->find_or_request_remote_expression(remote_expr_id, origin);
#ifdef DEBUG_LEGION
        IndexSpaceOperation *op = dynamic_cast<IndexSpaceOperation*>(result);
        assert(op != NULL);
#else
        IndexSpaceOperation *op = static_cast<IndexSpaceOperation*>(result);
#endif
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        // Unpack the global reference that we had
        op->unpack_global_ref();
        return result;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ IndexSpaceExpression* IndexSpaceExpression::unpack_expression(
          Deserializer &derez, RegionTreeForest *forest, AddressSpaceID source,
          PendingRemoteExpression &pending, RtEvent &wait_for)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!pending.done_ref_counting);
#endif
      // Handle the special case where this is a local index space expression 
      bool is_local;
      derez.deserialize(is_local);
      if (is_local)
      {
        IndexSpaceExpression *result;
        derez.deserialize(result);
#ifdef DEBUG_LEGION
        IndexSpaceOperation *op = 
          dynamic_cast<IndexSpaceOperation*>(result);
        assert(op != NULL);
#else
        IndexSpaceOperation *op = static_cast<IndexSpaceOperation*>(result);
#endif
        op->add_base_expression_reference(LIVE_EXPR_REF);
        if (source != forest->runtime->address_space)
          op->unpack_global_ref();
        ImplicitReferenceTracker::record_live_expression(result);
        pending.done_ref_counting = true;
        return result;
      }
      derez.deserialize(pending.is_index_space);
      // If this is an index space it is easy
      if (pending.is_index_space)
      {
        derez.deserialize(pending.handle);
        IndexSpaceNode *node = forest->get_node(pending.handle, &wait_for);
        if (node == NULL)
        {
          pending.source = source;
          return node;
        }
        node->add_base_expression_reference(LIVE_EXPR_REF);
        node->unpack_global_ref();
        pending.done_ref_counting = true;
        ImplicitReferenceTracker::record_live_expression(node);
        return node;
      }
      derez.deserialize(pending.remote_expr_id);
      IndexSpaceExpression *origin;
      derez.deserialize(origin);
      IndexSpaceExpression *result =
        forest->find_or_request_remote_expression(pending.remote_expr_id,
                                                  origin, &wait_for);
      if (result == NULL)
      {
        pending.source = source;
        return result;
      }
#ifdef DEBUG_LEGION
      IndexSpaceOperation *op = dynamic_cast<IndexSpaceOperation*>(result);
      assert(op != NULL);
#else
      IndexSpaceOperation *op = static_cast<IndexSpaceOperation*>(result);
#endif
      result->add_base_expression_reference(LIVE_EXPR_REF);
      op->unpack_global_ref();
      pending.done_ref_counting = true;
      ImplicitReferenceTracker::record_live_expression(result);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceOperation::IndexSpaceOperation(TypeTag tag, OperationKind kind,
                                             RegionTreeForest *ctx)
      : IndexSpaceExpression(tag, ctx->runtime, inter_lock), 
        DistributedCollectable(ctx->runtime,LEGION_DISTRIBUTED_HELP_ENCODE(
          ctx->runtime->get_available_distributed_id(), INDEX_EXPR_NODE_DC)),
        context(ctx), origin_expr(this), op_kind(kind), invalidated(0)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Index Expr %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, expr_id);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceOperation::IndexSpaceOperation(TypeTag tag, RegionTreeForest *ctx,
        IndexSpaceExprID eid, DistributedID did, IndexSpaceOperation *origin)
      : IndexSpaceExpression(tag, eid, inter_lock),
        DistributedCollectable(ctx->runtime, did), context(ctx),
        origin_expr(origin), op_kind(REMOTE_EXPRESSION_KIND), invalidated(0)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Expr %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, expr_id);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceOperation::~IndexSpaceOperation(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        context->unregister_remote_expression(expr_id);
      // Invalidate any derived operations
      invalidate_derived_operations(did, context);
      // Remove this operation from the region tree
      remove_operation();
      IndexSpaceExpression *canon = canonical.load();
      if (canon != NULL)
      {
        if (canon == this)
        {
#ifdef DEBUG_LEGION
          assert(has_volume);
#endif
          context->remove_canonical_expression(this, volume);
        }
        else if (canon->remove_canonical_reference(did))
          delete canon;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::try_add_live_reference(void)
    //--------------------------------------------------------------------------
    {
      if (check_global_and_increment(LIVE_EXPR_REF))
      {
        ImplicitReferenceTracker::record_live_expression(this);
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_base_expression_reference(
                                         ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_nested_expression_reference(
                                           DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_base_expression_reference(
                                         ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_nested_expression_reference(
                                           DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_tree_expression_reference(DistributedID id,
                                                            unsigned count)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(id, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_tree_expression_reference(DistributedID id,
                                                               unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(id, count);
    }

    /////////////////////////////////////////////////////////////
    // Operation Creator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OperationCreator::OperationCreator(RegionTreeForest *f)
      : forest(f), result(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OperationCreator::~OperationCreator(void)
    //--------------------------------------------------------------------------
    {
      // If we still have a result then it's because it wasn't consumed need 
      // we need to remove it's reference that was added by the constructor 
      if ((result != NULL) && result->remove_base_resource_ref(REGION_TREE_REF))
        delete result;
    }

    //--------------------------------------------------------------------------
    void OperationCreator::produce(IndexSpaceOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(result == NULL);
#endif
      result = op;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* OperationCreator::consume(void)
    //--------------------------------------------------------------------------
    {
      if (result == NULL)
        create_operation();
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Add an expression reference here since this is going to be put
      // into the region tree expression trie data structure, the reference
      // will be removed when the expressions is removed from the trie
      result->add_base_gc_ref(REGION_TREE_REF);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Expression Trie Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExpressionTrieNode::ExpressionTrieNode(unsigned d, IndexSpaceExprID id,
                                           IndexSpaceExpression *op)
      : depth(d), expr(id), local_operation(op)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ExpressionTrieNode::ExpressionTrieNode(const ExpressionTrieNode &rhs)
      : depth(rhs.depth), expr(rhs.expr)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ExpressionTrieNode::~ExpressionTrieNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ExpressionTrieNode& ExpressionTrieNode::operator=(
                                                  const ExpressionTrieNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ExpressionTrieNode::find_operation(
                       const std::vector<IndexSpaceExpression*> &expressions,
                       IndexSpaceExpression *&result, ExpressionTrieNode *&last)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(depth < expressions.size());
      assert(expressions[depth]->expr_id == expr); // these should match
#endif
      // Three cases here
      if (expressions.size() == (depth+1))
      {
        // We're the node that should have the operation
        // Check to see if we've made the operation yet
        if (local_operation != NULL)
        {
          result = local_operation;
          return true;
        }
        last = this;
        return false;
      }
      else if (expressions.size() == (depth+2))
      {
        // The next node should have the operation, but we might be
        // storing it until it actually gets made
        // See if we already have it or we have the next trie node
        ExpressionTrieNode *next = NULL;
        const IndexSpaceExprID target_expr = expressions.back()->expr_id;
        {
          AutoLock t_lock(trie_lock,1,false/*exclusive*/);
          std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator
            op_finder = operations.find(target_expr);
          if (op_finder != operations.end())
          {
            result = op_finder->second;
            return true;
          }
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            node_finder = nodes.find(target_expr);
          if (node_finder != nodes.end())
            next = node_finder->second;
        }
        // Didn't find either, retake the lock in exclusive mode and then
        // see if we lost the race, if not make the operation or
        if (next == NULL)
        {
          AutoLock t_lock(trie_lock);
          std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator
            op_finder = operations.find(target_expr);
          if (op_finder != operations.end())
          {
            result = op_finder->second;
            return true;
          }
          // Still don't have the op
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            node_finder = nodes.find(target_expr);
          if (node_finder == nodes.end())
          {
            last = this;
            return false;
          }
          else
            next = node_finder->second;
        }
#ifdef DEBUG_LEGION
        assert(next != NULL);
#endif
        return next->find_operation(expressions, result, last);
      }
      else
      {
        // Intermediate case 
        // See if we have the next node, or if we have to make it
        ExpressionTrieNode *next = NULL;
        const IndexSpaceExprID target_expr = expressions[depth+1]->expr_id;
        {
          AutoLock t_lock(trie_lock,1,false/*exclusive*/);
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            finder = nodes.find(target_expr);
          if (finder != nodes.end())
            next = finder->second;
        }
        // Still don't have it so we have to try and make it
        if (next == NULL)
        {
          AutoLock t_lock(trie_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            finder = nodes.find(target_expr);
          if (finder == nodes.end())
          {
            // We have to make the next node, also check to see if we
            // already made an operation expression for it or not
            std::map<IndexSpaceExprID,IndexSpaceExpression*>::iterator
              op_finder = operations.find(target_expr);
            if (op_finder != operations.end())
            {
              next = new ExpressionTrieNode(depth+1, target_expr, 
                                            op_finder->second);
              operations.erase(op_finder);    
            }
            else
              next = new ExpressionTrieNode(depth+1, target_expr);
            nodes[target_expr] = next;
          }
          else // lost the race
            next = finder->second;
        }
#ifdef DEBUG_LEGION
        assert(next != NULL);
#endif
        return next->find_operation(expressions, result, last);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* ExpressionTrieNode::find_or_create_operation(
                          const std::vector<IndexSpaceExpression*> &expressions,
                          OperationCreator &creator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(depth < expressions.size());
      assert(expressions[depth]->expr_id == expr); // these should match
#endif
      // Three cases here
      if (expressions.size() == (depth+1))
      {
        // We're the node that should have the operation
        // Check to see if we've made the operation yet
        if ((local_operation != NULL) &&
            local_operation->try_add_live_reference())
          return local_operation;
        // Operation doesn't exist yet, retake the lock and try to make it
        AutoLock t_lock(trie_lock);
        if ((local_operation != NULL) &&
            local_operation->try_add_live_reference())
          return local_operation;
        local_operation = creator.consume();
        if (!local_operation->try_add_live_reference())
          assert(false); // should never hit this
        return local_operation;
      }
      else if (expressions.size() == (depth+2))
      {
        // The next node should have the operation, but we might be
        // storing it until it actually gets made
        // See if we already have it or we have the next trie node
        ExpressionTrieNode *next = NULL;
        const IndexSpaceExprID target_expr = expressions.back()->expr_id;
        {
          AutoLock t_lock(trie_lock,1,false/*exclusive*/);
          std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator
            op_finder = operations.find(target_expr);
          if ((op_finder != operations.end()) &&
              op_finder->second->try_add_live_reference())
            return op_finder->second;
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            node_finder = nodes.find(target_expr);
          if (node_finder != nodes.end())
            next = node_finder->second;
        }
        // Didn't find either, retake the lock in exclusive mode and then
        // see if we lost the race, if not make the operation or
        if (next == NULL)
        {
          AutoLock t_lock(trie_lock);
          std::map<IndexSpaceExprID,IndexSpaceExpression*>::const_iterator
            op_finder = operations.find(target_expr);
          if ((op_finder != operations.end()) &&
              op_finder->second->try_add_live_reference())
            return op_finder->second;
          // Still don't have the op
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            node_finder = nodes.find(target_expr);
          if (node_finder == nodes.end())
          {
            // Didn't find the sub-node, so make the operation here
            IndexSpaceExpression *result = creator.consume();
            operations[target_expr] = result;
            if (!result->try_add_live_reference())
              assert(false); // should never hit this
            return result;
          }
          else
            next = node_finder->second;
        }
#ifdef DEBUG_LEGION
        assert(next != NULL);
#endif
        return next->find_or_create_operation(expressions, creator);
      }
      else
      {
        // Intermediate case 
        // See if we have the next node, or if we have to make it
        ExpressionTrieNode *next = NULL;
        const IndexSpaceExprID target_expr = expressions[depth+1]->expr_id;
        {
          AutoLock t_lock(trie_lock,1,false/*exclusive*/);
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            finder = nodes.find(target_expr);
          if (finder != nodes.end())
            next = finder->second;
        }
        // Still don't have it so we have to try and make it
        if (next == NULL)
        {
          AutoLock t_lock(trie_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            finder = nodes.find(target_expr);
          if (finder == nodes.end())
          {
            // We have to make the next node, also check to see if we
            // already made an operation expression for it or not
            std::map<IndexSpaceExprID,IndexSpaceExpression*>::iterator
              op_finder = operations.find(target_expr);
            if (op_finder != operations.end())
            {
              next = new ExpressionTrieNode(depth+1, target_expr, 
                                            op_finder->second);
              operations.erase(op_finder);
            }
            else
              next = new ExpressionTrieNode(depth+1, target_expr);
            nodes[target_expr] = next;
          }
          else // lost the race
            next = finder->second;
        }
#ifdef DEBUG_LEGION
        assert(next != NULL);
#endif
        return next->find_or_create_operation(expressions, creator);
      }
    }

    //--------------------------------------------------------------------------
    bool ExpressionTrieNode::remove_operation(
                          const std::vector<IndexSpaceExpression*> &expressions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(depth < expressions.size());
      assert(expressions[depth]->expr_id == expr); // these should match
#endif
      // No need for locks here, we're protected by the big lock at the top
      // Three cases here
      if (expressions.size() == (depth+1))
      {
        // Simple case, clear our local operation
        local_operation = NULL;
      }
      else if (expressions.size() == (depth+2))
      {
        // See if we should continue traversing or if we have the operation
        const IndexSpaceExprID target_expr = expressions.back()->expr_id;
        std::map<IndexSpaceExprID,IndexSpaceExpression*>::iterator op_finder =
          operations.find(target_expr);
        if (op_finder == operations.end())
        {
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::iterator
            node_finder = nodes.find(target_expr);
#ifdef DEBUG_LEGION
          assert(node_finder != nodes.end());
#endif
          if (node_finder->second->remove_operation(expressions))
          {
            delete node_finder->second;
            nodes.erase(node_finder);
          }
        }
        else
          operations.erase(op_finder);
      }
      else
      {
        const IndexSpaceExprID target_expr = expressions[depth+1]->expr_id;
        std::map<IndexSpaceExprID,ExpressionTrieNode*>::iterator finder =
          nodes.find(target_expr);
#ifdef DEBUG_LEGION
        assert(finder != nodes.end());
#endif
        if (finder->second->remove_operation(expressions))
        {
          delete finder->second;
          nodes.erase(finder);
        }
      }
      if (local_operation != NULL)
        return false;
      if (!operations.empty())
        return false;
      if (!nodes.empty())
        return false;
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Index Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTreeNode::IndexTreeNode(RegionTreeForest *ctx, unsigned d,
        LegionColor c, DistributedID did, RtEvent init,
        CollectiveMapping *mapping, Provenance *prov, bool tree_valid)
      : ValidDistributedCollectable(ctx->runtime, did, false/*register*/,
          mapping, tree_valid), context(ctx), depth(d), color(c),
        provenance(prov), initialized(init)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    IndexTreeNode::~IndexTreeNode(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    } 

    //--------------------------------------------------------------------------
    void IndexTreeNode::attach_semantic_information(SemanticTag tag,
                                                    AddressSpaceID source,
                                                    const void *buffer, 
                                                    size_t size,
                                                    bool is_mutable,
                                                    bool local_only)
    //--------------------------------------------------------------------------
    {
      // Make a copy
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          // First check to see if it is valid
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // It's not mutable so check to make 
              // sure that the bits are the same
              if (size != finder->second.size)
                REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                  "Inconsistent Semantic Tag value "
                              "for tag %ld with different sizes of %zd"
                              " and %zd for index tree node", 
                              tag, size, finder->second.size)
              // Otherwise do a bitwise comparison
              {
                const char *orig = (const char*)finder->second.buffer;
                const char *next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                    REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                    "Inconsistent Semantic Tag value "
                                  "for tag %ld with different values at"
                                  "byte %d for index tree node, %x != %x",
                                  tag, idx, orig[idx], next[idx])
                }
              }
              added = false;
            }
            else
            {
              // Mutable so overwrite the result
              legion_free(SEMANTIC_INFO_ALLOC, finder->second.buffer,
                          finder->second.size);
              finder->second.buffer = local;
              finder->second.size = size;
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer = local;
            finder->second.size = size;
            // Trigger will happen by the caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_info[tag] = SemanticInfo(local, size, is_mutable);
      }
      if (added)
      {
        AddressSpaceID owner_space = get_owner_space();
        // If we are not the owner and the message 
        // didn't come from the owner, then send it 
        if ((owner_space != context->runtime->address_space) &&
            (source != owner_space) && !local_only)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          send_semantic_info(owner_space, tag, buffer, size, is_mutable, done); 
          if (!done.has_triggered())
            done.wait();
        }
      }
      else
        legion_free(SEMANTIC_INFO_ALLOC, local, size);
    }

    //--------------------------------------------------------------------------
    bool IndexTreeNode::retrieve_semantic_information(SemanticTag tag,
                                                      const void *&result,
                                                      size_t &size, 
                                                      bool can_fail,
                                                      bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != context->runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        LegionMap<SemanticTag,SemanticInfo>::const_iterator finder = 
          semantic_info.find(tag); 
        if (finder != semantic_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else // can use the canonical event
              wait_on = finder->second.ready_event; 
          }
          else if (wait_until) // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        // Nothing to wait on so we have to do something
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG,
          "invalid semantic tag %ld for "
                      "index tree node", tag)
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
          send_semantic_request(owner_space, tag, can_fail, wait_until,request);
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      LegionMap<SemanticTag,SemanticInfo>::const_iterator finder = 
        semantic_info.find(tag);
      if (finder == semantic_info.end())
      {
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG,
          "invalid semantic tag %ld for "
                            "index tree node", tag)
      }
      result = finder->second.buffer;
      size = finder->second.size;
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Node 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(RegionTreeForest *ctx, IndexSpace h, 
                                   IndexPartNode *par, LegionColor c,
                                   DistributedID did,
                                   IndexSpaceExprID exp_id, RtEvent init,
                                   unsigned dep, Provenance *prov,
                                   CollectiveMapping *map, bool tree_valid)
      : IndexTreeNode(ctx,
          (dep == UINT_MAX) ? ((par == NULL) ? 0 : par->depth + 1) : dep, c, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_SPACE_NODE_DC),
          init, map, prov, tree_valid),
        IndexSpaceExpression(h.type_tag, exp_id > 0 ? exp_id : 
            runtime->get_unique_index_space_expr_id(), node_lock),
        handle(h), parent(par), next_uncollected_color(0),
        index_space_set(false), index_space_tight(false)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (parent != NULL)
        assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
      if (parent != NULL)
        parent->add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info("GC Index Space %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.id);
#endif
      if (is_owner() && ctx->runtime->legion_spy_enabled)
        LegionSpy::log_index_space_expr(handle.get_id(), this->expr_id);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::~IndexSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      if ((parent != NULL) && parent->remove_nested_resource_ref(did))
        delete parent;
      // Remove ourselves from the context
      if (registered_with_runtime)
        context->remove_node(handle);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here currently
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        context->unregister_remote_expression(expr_id);
      // Invalidate any derived operations
      invalidate_derived_operations(did, context);
      IndexSpaceExpression *canon = canonical.load();
      if (canon != NULL)
      {
        if (canon == this)
        {
#ifdef DEBUG_LEGION
          assert(has_volume);
#endif
          context->remove_canonical_expression(this, volume);
        }
        else if (canon->remove_canonical_reference(did))
          delete canon;
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::is_index_space_node(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexSpaceNode::as_index_space_node(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexSpaceNode::as_index_part_node(void)
    //--------------------------------------------------------------------------
    {
      return NULL;
    }
#endif

    //--------------------------------------------------------------------------
    AddressSpaceID IndexSpaceNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle, context->runtime);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID IndexSpaceNode::get_owner_space(IndexSpace handle,
                                                              Runtime *rt)
    //--------------------------------------------------------------------------
    {
      return (handle.id % rt->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* IndexSpaceNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_semantic_request(AddressSpaceID target,
             SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      context->runtime->send_index_space_semantic_request(target, rez);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_semantic_info(AddressSpaceID target,
                                            SemanticTag tag,
                                            const void *buffer, size_t size,
                                            bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      context->runtime->send_index_space_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::process_semantic_request(SemanticTag tag,
       AddressSpaceID source, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_owner_space() == context->runtime->address_space);
#endif
      RtEvent precondition;
      void *result = NULL;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          context->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_semantic_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      IndexSpaceNode *node = forest->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_semantic_info(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      forest->attach_semantic_information(handle, tag, source, buffer, size, 
                                          is_mutable, false/*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *child = get_child(c, NULL/*defer*/, true/*can fail*/);
      if (child == NULL)
        return false;
      if (child->remove_base_resource_ref(REGION_TREE_REF))
        delete child;
      return true;
    }

    //--------------------------------------------------------------------------
    LegionColor IndexSpaceNode::generate_color(LegionColor suggestion)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        AutoLock n_lock(node_lock);
        // First check that we haven't recorded any children that don't have
        // any generated colors as it is illegal to generate colors if the
        // user has determined that they are specifying all the colors
        if (remote_colors.find(INVALID_COLOR) == remote_colors.end())
        {
          if (!color_map.empty() || !remote_colors.empty() || 
              (next_uncollected_color > 0))
            REPORT_LEGION_ERROR(ERROR_MIXED_PARTITION_COLOR_ALLOCATION_MODES,
                "Illegal request for Legion to generated a color for index "
                "space %d after a child was already registered with an "
                "explicit color. Colors of partitions must either be "
                "completely specified by the user or completely generated "
                "by the runtime. Mixing of allocation modes is not allowed.",
                handle.id)
          // If we made it here then there are no other children registered
          // so we record an empty entry to mark that we're generating colors
          remote_colors[INVALID_COLOR] = IndexPartition::NO_PART;
        }
        // If the user made a suggestion see if it was right
        if (suggestion != INVALID_COLOR)
        {
          // If someone already has it then they can't use it
          if ((next_uncollected_color <= suggestion) &&
              (color_map.find(suggestion) == color_map.end()))
          {
            color_map[suggestion] = NULL;
            return suggestion;
          }
          else
            return INVALID_COLOR;
        }
        if (color_map.empty())
        {
          // save a space for later
          color_map[next_uncollected_color] = NULL;
          return next_uncollected_color;
        }
        std::map<LegionColor,IndexPartNode*>::const_iterator next = 
          color_map.begin();
        if (next->first > next_uncollected_color)
        {
          // save a space for later
          color_map[next_uncollected_color] = NULL;
          return next_uncollected_color;
        }
        std::map<LegionColor,IndexPartNode*>::const_iterator prev = next++;
        while (next != color_map.end())
        {
          if (next->first != (prev->first + 1))
          {
            // save a space for later
            color_map[prev->first+1] = NULL;
            return prev->first+1;
          }
          prev = next++;
        }
        color_map[prev->first+1] = NULL;
        return prev->first+1;
      }
      else
      {
        // Send a message to the owner to pick a color and wait for the result
        std::atomic<LegionColor> result(suggestion); 
        RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(suggestion);
          rez.serialize(&result);
          rez.serialize(ready);
        }
        runtime->send_index_space_generate_color_request(owner_space, rez);
        if (!ready.has_triggered())
          ready.wait();
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::release_color(LegionColor color)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        AutoLock n_lock(node_lock);
        std::map<LegionColor,IndexPartNode*>::iterator finder = 
          color_map.find(color);
#ifdef DEBUG_LEGION
        assert(finder != color_map.end());
        assert(finder->second == NULL);
#endif
        color_map.erase(finder);
      }
      else
      {
        pack_valid_ref();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(color);
        }
        runtime->send_index_space_release_color(owner_space, rez);
      }
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexSpaceNode::get_child(const LegionColor c, 
                                             RtEvent *defer, bool can_fail)
    //--------------------------------------------------------------------------
    {
      // See if we have it locally if not go find it
      IndexPartition remote_handle = IndexPartition::NO_PART;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<LegionColor,IndexPartNode*>::const_iterator finder = 
          color_map.find(c);
        if ((finder != color_map.end()) && (finder->second != NULL))
        {
          if (can_fail)
            finder->second->add_base_resource_ref(REGION_TREE_REF);
          return finder->second;
        }
        std::map<LegionColor,IndexPartition>::const_iterator remote_finder = 
          remote_colors.find(c);
        if (remote_finder != remote_colors.end())
          remote_handle = remote_finder->second;
      }
      // if we make it here, send a request
      AddressSpaceID owner_space = get_owner_space();
      if (owner_space == context->runtime->address_space)
      {
        if (remote_handle.exists())
        {
          IndexPartNode *result = context->get_node(remote_handle, defer);
          if (can_fail && (result != NULL))
            result->add_base_resource_ref(REGION_TREE_REF);
          return result;
        }
        if (can_fail)
          return NULL;
        REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_COLOR,
          "Unable to find entry for color %lld in "
                        "index space %x.", c, handle.id)
      }
      RtUserEvent ready_event = Runtime::create_rt_user_event();

      std::atomic<IndexPartitionID> child_id(0);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(c);
        if (defer == NULL)
          rez.serialize(&child_id);
        else
          rez.serialize<std::atomic<IndexPartitionID>*>(NULL);
        rez.serialize(ready_event);
      }
      context->runtime->send_index_space_child_request(owner_space, rez);
      if (defer == NULL)
      {
        ready_event.wait();
        IndexPartitionID cid = child_id.load(); 
        if (cid == 0)
        {
          if (can_fail)
            return NULL;
          REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_COLOR,
            "Unable to find entry for color %lld in "
                          "index space %x.", c, handle.id)
        }
        IndexPartition child_handle(child_id.load(),
            handle.get_tree_id(), handle.get_type_tag());
        IndexPartNode *result = context->get_node(child_handle);
        if (can_fail)
          result->add_base_resource_ref(REGION_TREE_REF);
        // Always unpack the global ref that got sent back with this
        result->unpack_global_ref();
        return result;
      }
      else
      {
        *defer = ready_event;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_child(IndexPartNode *child) 
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      // Can have a NULL pointer
      assert((color_map.find(child->color) == color_map.end()) ||
             (color_map[child->color] == NULL));
#endif
      if (is_owner() && 
          (remote_colors.find(INVALID_COLOR) != remote_colors.end()) &&
          (color_map.find(child->color) == color_map.end()))
        REPORT_LEGION_ERROR(ERROR_MIXED_PARTITION_COLOR_ALLOCATION_MODES,
              "Illegal request for Legion to generated a color for index "
              "space %d after a child was already registered with an "
              "explicit color. Colors of partitions must either be "
              "completely specified by the user or completely generated "
              "by the runtime. Mixing of allocation modes is not allowed.",
              handle.id)
      color_map[child->color] = child;
      if (!remote_colors.empty())
        remote_colors.erase(child->color);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      std::map<LegionColor,IndexPartNode*>::iterator finder = 
        color_map.find(c);
#ifdef DEBUG_LEGION
      assert(finder != color_map.end());
      assert(finder->second != NULL);
      assert(finder->second != ((IndexPartNode*)REMOVED_CHILD));
#endif
      finder->second = (IndexPartNode*)REMOVED_CHILD;
      while ((finder->first == next_uncollected_color) &&
             (finder->second == ((IndexPartNode*)REMOVED_CHILD)))
      {
        next_uncollected_color++;
        color_map.erase(finder);
        if (color_map.empty())
          break;
        finder = color_map.begin();
      }
    }

    //--------------------------------------------------------------------------
    size_t IndexSpaceNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return color_map.size();
    } 

    //--------------------------------------------------------------------------
    RtEvent IndexSpaceNode::get_ready_event(void)
    //--------------------------------------------------------------------------
    {
      if (index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      AutoLock n_lock(node_lock);
      if (index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      if (!index_space_ready.exists())
        index_space_ready = Runtime::create_rt_user_event();
      return index_space_ready;
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::are_disjoint(LegionColor c1, LegionColor c2)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (c1 == c2)
        return false;
      if (c1 > c2)
        std::swap(c1, c2);
      const std::pair<LegionColor,LegionColor> key(c1,c2);
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        if (disjoint_subsets.find(key) != disjoint_subsets.end())
          return true;
        else if (aliased_subsets.find(key) != aliased_subsets.end())
          return false;
      }
      IndexPartNode *left = get_child(c1);
      IndexPartNode *right = get_child(c2);
      const bool intersects = left->intersects_with(right,
            !context->runtime->disable_independence_tests);
      AutoLock n_lock(node_lock);
      if (intersects)
      {
        aliased_subsets.insert(key);
        return false;
      }
      else
      {
        disjoint_subsets.insert(key);
        return true;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::record_remote_child(IndexPartition pid, 
                                             LegionColor part_color)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert((remote_colors.find(part_color) == remote_colors.end()) ||
              (remote_colors[part_color] == pid));
      // should only happen on the owner node
      assert(get_owner_space() == context->runtime->address_space);
#endif
      if ((remote_colors.find(INVALID_COLOR) != remote_colors.end()) &&
          (color_map.find(part_color) == color_map.end()))
        REPORT_LEGION_ERROR(ERROR_MIXED_PARTITION_COLOR_ALLOCATION_MODES,
              "Illegal request for Legion to generated a color for index "
              "space %d after a child was already registered with an "
              "explicit color. Colors of partitions must either be "
              "completely specified by the user or completely generated "
              "by the runtime. Mixing of allocation modes is not allowed.",
              handle.id)
      remote_colors[part_color] = pid;
    }

    //--------------------------------------------------------------------------
    LegionColor IndexSpaceNode::get_colors(std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we need to request an up to date set of colors
      // since it can change arbitrarily
      AddressSpaceID owner_space = get_owner_space();
      if (owner_space != context->runtime->address_space)
      {
        LegionColor bound = INVALID_COLOR;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(&colors);
          rez.serialize(&bound);
          rez.serialize(ready_event); 
        }
        context->runtime->send_index_space_colors_request(owner_space, rez);
        ready_event.wait();
#ifdef DEBUG_LEGION
        assert(bound != INVALID_COLOR);
#endif
        return bound;
      }
      else
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        for (std::map<LegionColor,IndexPartNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
        {
          // Can be NULL in some cases of parallel partitioning
          if ((it->second != NULL) && (!it->second->initialized.exists() ||
                it->second->initialized.has_triggered()))
            colors.push_back(it->first);
        }
        return next_uncollected_color;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::IndexSpaceSetFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target == runtime->address_space)
        return;
      if (target == source)
        return;
      runtime->send_index_space_set(target, rez);
    } 

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_node(AddressSpaceID target,
                                   bool recurse, bool valid)
    //--------------------------------------------------------------------------
    {
      // Quick out if we've already sent this
      if (has_remote_instance(target))
        return;
      // Send our parent first if necessary
      if (recurse && (parent != NULL))
        parent->send_node(target, true/*recurse*/);
      // Only send it if we're the owner without a collective mapping
      // or the target is not in the collective mapping and we're the
      // closest node in the collective mapping to the target
      if ((is_owner() && (collective_mapping == NULL)) ||
          ((collective_mapping != NULL) && 
           !collective_mapping->contains(target) && 
           collective_mapping->contains(local_space) &&
           (local_space == collective_mapping->find_nearest(target))))
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(is_global());
        assert(is_valid() || !recurse);
#endif
        if (!has_remote_instance(target))
        {
          Serializer rez;
          pack_node(rez, target, recurse, valid);
          context->runtime->send_index_space_response(target, rez);
          update_remote_instances(target);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::pack_node(Serializer &rez, AddressSpaceID target,
                                   bool recurse, bool valid)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(handle);
      rez.serialize(did);
      if (recurse && (parent != NULL))
        rez.serialize(parent->handle);
      else
        rez.serialize(IndexPartition::NO_PART);
      rez.serialize(color);
      rez.serialize(index_space_ready);
      rez.serialize(expr_id);
      rez.serialize(initialized);
      rez.serialize(depth);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      if (index_space_set && ((collective_mapping == NULL) ||
            !collective_mapping->contains(target)))
        pack_index_space(rez, true/*include size*/);
      else
        rez.serialize<size_t>(0);
      if (collective_mapping != NULL)
        collective_mapping->pack(rez);
      else
        rez.serialize<size_t>(0); // total spaces
      rez.serialize<bool>(valid); // whether the tree is valid or not
      rez.serialize<size_t>(semantic_info.size());
      for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second.size);
        rez.serialize(it->second.buffer, it->second.size);
        rez.serialize(it->second.is_mutable);
      } 
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::invalidate_root(AddressSpaceID source,
                   std::set<RtEvent> &applied, const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent == NULL);
#endif
      bool need_broadcast = true;
      if (source == local_space)
      {
        // Entry point
        if (mapping != NULL)
        {
          if ((collective_mapping != NULL) && ((mapping == collective_mapping) 
                || (*mapping == *collective_mapping)))
          {
            need_broadcast = false;
          }
          else if (mapping->contains(owner_space))
          {
            if (local_space != owner_space)
              return false;
          }
          else
          {
            // Find the one closest to the owner space
            const AddressSpaceID nearest = mapping->find_nearest(owner_space);
            if (nearest != local_space)
              return false;
            runtime->send_index_space_destruction(handle, owner_space, applied);
            // If we're part of the broadcast tree then we'll get sent back here
            // later so we don't need to do anything now
            if ((collective_mapping != NULL) && 
                collective_mapping->contains(local_space))
              return false;
          }
        }
        else
        {
          // If we're not the owner space, send the message there
          if (!is_owner())
          {
            runtime->send_index_space_destruction(handle, owner_space, applied);
            return false;
          }
        }
      }
      if (need_broadcast && (collective_mapping != NULL) &&
          collective_mapping->contains(local_space))
      {
#ifdef DEBUG_LEGION
        // Should be from our parent
        assert(is_owner() || (source == 
            collective_mapping->get_parent(owner_space, local_space)));
#endif
        // Keep broadcasting this out to all the children
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        for (std::vector<AddressSpaceID>::const_iterator it =
              children.begin(); it != children.end(); it++)
          runtime->send_index_space_destruction(handle, *it, applied);
      }
      return remove_base_valid_ref(APPLICATION_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_creation(
        RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      DistributedID did;
      derez.deserialize(did);
      IndexPartition parent;
      derez.deserialize(parent);
      LegionColor color;
      derez.deserialize(color);
      ApEvent ready_event;
      derez.deserialize(ready_event);
      IndexSpaceExprID expr_id;
      derez.deserialize(expr_id);
      RtEvent initialized;
      derez.deserialize(initialized);
      unsigned depth;
      derez.deserialize(depth);
      AutoProvenance provenance(Provenance::deserialize(derez));
      size_t index_space_size;
      derez.deserialize(index_space_size);
      const void *index_space_ptr = (index_space_size > 0) ?
        derez.get_current_pointer() : NULL;
      derez.advance_pointer(index_space_size);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping *mapping = NULL;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);
      bool valid;
      derez.deserialize<bool>(valid);

      IndexPartNode *parent_node = NULL;
      if (parent != IndexPartition::NO_PART)
        parent_node = context->get_node(parent);
      IndexSpaceNode *node = context->create_node(handle, index_space_ptr,
          false/*is domain*/, parent_node, color, did, initialized, provenance,
          ready_event,expr_id,mapping,false/*add root reference*/,depth,valid);
#ifdef DEBUG_LEGION
      assert(node != NULL);
#endif
      size_t num_semantic;
      derez.deserialize(num_semantic);
      for (unsigned idx = 0; idx < num_semantic; idx++)
      {
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void *buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(tag, source, buffer, buffer_size, 
                                          is_mutable, false/*local only*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_request(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      AddressSpaceID source;
      derez.deserialize(source);
      IndexSpaceNode *target = forest->get_node(handle, NULL, true/*can fail*/);
      bool valid = false;
      if (target != NULL)
      {
        // If there is a collective mapping, check to see if we're on the
        // right node and if not forward it on to the right node
        if (target->collective_mapping != NULL)
        {
#ifdef DEBUG_LEGION
          assert(!target->collective_mapping->contains(source));
          assert(target->collective_mapping->contains(target->local_space));
#endif
          if (target->is_owner())
          {
            const AddressSpaceID nearest = 
              target->collective_mapping->find_nearest(source);
            // If we're not the nearest then forward it on to the
            // proper node to handle the request
            if (nearest != target->local_space)
            {
              Serializer rez;
              rez.serialize(handle);
              rez.serialize(to_trigger);
              rez.serialize(source);
              forest->runtime->send_index_space_request(nearest, rez);
              return;
            }
          }
#ifdef DEBUG_LEGION
          else
          {
            assert(target->local_space == 
                target->collective_mapping->find_nearest(source));
          }
#endif
        }
        // See if we're going to be sending the whole tree or not
        bool recurse = false;
        if (target->parent == NULL)
        {
          if (target->check_valid_and_increment(REGION_TREE_REF))
          {
            valid = true;
            target->pack_valid_ref();
            target->remove_base_valid_ref(REGION_TREE_REF);
          }
          else
            target->pack_global_ref();
        }
        else
        {
          // If we have a parent then we need to do the valid reference
          // check on the partition since that keeps this tree valid
          if (target->parent->check_valid_and_increment(REGION_TREE_REF))
          {
            valid = true;
            recurse = true;
            target->parent->pack_valid_ref();
            target->parent->remove_base_valid_ref(REGION_TREE_REF);
          }
          else
          {
            // We need the state to remain the same while we are in
            // transit so see if this can still be made valid
            if (target->check_valid_and_increment(REGION_TREE_REF))
            {
              valid = true;
              target->pack_valid_ref();
              target->remove_base_valid_ref(REGION_TREE_REF);
            }
            else
              target->pack_global_ref();
          }
        }
        target->send_node(source, recurse, valid);
        // Now send back the results
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(to_trigger);
          rez.serialize(handle);
          rez.serialize(valid);
          rez.serialize(recurse);
        }
        forest->runtime->send_index_space_return(source, rez);
      }
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_return(
                                 RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode *node = context->get_node(handle);
      bool valid;
      derez.deserialize(valid);
      bool recurse;
      derez.deserialize(recurse);
      if (valid)
      {
        if (recurse)
          node->parent->unpack_valid_ref();
        else
          node->unpack_valid_ref();
      }
      else
        node->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_child_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      LegionColor child_color;
      derez.deserialize(child_color);
      std::atomic<IndexPartitionID> *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      IndexSpaceNode *parent = forest->get_node(handle);
      RtEvent defer;
      IndexPartNode *child = 
        parent->get_child(child_color, &defer, true/*can fail*/);
      if (defer.exists())
      {
        // Build a continuation and run it when the node is 
        // ready, we have to do this in order to avoid blocking
        // the virtual channel for nested index tree requests
        DeferChildArgs args(parent, child_color, target, to_trigger, source);
        forest->runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, defer);
        return;
      }
      if (child != NULL)
      {
        if (child->check_global_and_increment(REGION_TREE_REF))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(child->handle);
            rez.serialize(target);
            rez.serialize(to_trigger);
            child->pack_global_ref();
          }
          forest->runtime->send_index_space_child_response(source, rez);
          if (child->remove_base_gc_ref(REGION_TREE_REF))
            delete child;
        }
        else // can fail and unable to get a global reference
          Runtime::trigger_event(to_trigger);
        if (child->remove_base_resource_ref(REGION_TREE_REF))
          delete child;
      }
      else // Failed so just trigger the result
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::defer_node_child_request(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferChildArgs *dargs = (const DeferChildArgs*)args;
      IndexPartNode *child = 
       dargs->proxy_this->get_child(dargs->child_color, NULL, true/*can fail*/);
      if (child != NULL)
      {
        if (child->check_global_and_increment(REGION_TREE_REF))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(child->handle);
            rez.serialize(dargs->target);
            rez.serialize(dargs->to_trigger);
            child->pack_global_ref();
          }
          Runtime *runtime = dargs->proxy_this->context->runtime;
          runtime->send_index_space_child_response(dargs->source, rez);
          if (child->remove_base_gc_ref(REGION_TREE_REF))
            delete child;
        }
        else // Unable to get a global reference
          Runtime::trigger_event(dargs->to_trigger);
        if (child->remove_base_resource_ref(REGION_TREE_REF))
          delete child;
      }
      else // Failed so just trigger the result
        Runtime::trigger_event(dargs->to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_child_response(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      std::atomic<IndexPartitionID> *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      if (target == NULL)
      {
        // In this case we need to block here to make sure we can
        // unpack the global reference we added on the remote node
        // since there's nothing on the local node that is going to do it
        IndexPartNode *child = forest->get_node(handle);
        child->unpack_global_ref();
        Runtime::trigger_event(to_trigger);
      }
      else
      {
        RtEvent defer;
        forest->get_node(handle, &defer);
        // We'll update references and unpack the remote reference on 
        // the requester here so there's no need to block waiting
        target->store(handle.get_id());
        Runtime::trigger_event(to_trigger, defer);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_colors_request(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      std::vector<LegionColor> *target;
      derez.deserialize(target);
      LegionColor *bound_target;
      derez.deserialize(bound_target);
      RtUserEvent ready;
      derez.deserialize(ready);
      IndexSpaceNode *node = context->get_node(handle);
      std::vector<LegionColor> results;
      LegionColor bound = node->get_colors(results);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize<size_t>(results.size());
        for (std::vector<LegionColor>::const_iterator it = results.begin();
              it != results.end(); it++)
          rez.serialize(*it);
        rez.serialize(bound_target);
        rez.serialize(bound);
        rez.serialize(ready);
      }
      context->runtime->send_index_space_colors_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_colors_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::vector<LegionColor> *target;
      derez.deserialize(target);
      size_t num_colors;
      derez.deserialize(num_colors);
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        LegionColor cp;
        derez.deserialize(cp);
        target->push_back(cp);
      }
      LegionColor *bound_target;
      derez.deserialize(bound_target);
      derez.deserialize(*bound_target);
      RtUserEvent ready;
      derez.deserialize(ready);
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_index_space_set(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source) 
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition parent_handle;
      derez.deserialize(parent_handle);
      if (parent_handle.exists())
      {
        LegionColor color;
        derez.deserialize(color);
        IndexPartNode *parent = forest->get_node(parent_handle);
        IndexSpaceNode *child = parent->get_child(color);
        if (child->unpack_index_space(derez, source))
          delete child;
      }
      else
      {
        IndexSpace handle;
        derez.deserialize(handle);
        IndexSpaceNode *node = forest->get_node(handle);
        if (node->unpack_index_space(derez, source))
          delete node;
      }
    } 

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_generate_color_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      LegionColor suggestion;
      derez.deserialize(suggestion);
      std::atomic<LegionColor> *target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      IndexSpaceNode *node = forest->get_node(handle);
      LegionColor result = node->generate_color(suggestion);
      if (result != suggestion)
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize(result);
          rez.serialize(done_event);
        }
        forest->runtime->send_index_space_generate_color_response(source, rez);
      }
      else // if we matched the suggestion we know the value is right
        Runtime::trigger_event(done_event); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_generate_color_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<LegionColor> *target;
      derez.deserialize(target);
      LegionColor result;
      derez.deserialize(result);
      target->store(result);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_release_color(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      LegionColor color;
      derez.deserialize(color);

      IndexSpaceNode *node = forest->get_node(handle);
      node->release_color(color);
      node->unpack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::pack_expression(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != context->runtime->address_space)
      {
        rez.serialize<bool>(false/*local*/);
        rez.serialize<bool>(true/*index space*/);
        rez.serialize(handle);
        pack_global_ref();
      }
      else
      {
        rez.serialize<bool>(true/*local*/);
        rez.serialize<IndexSpaceExpression*>(this);
        add_base_expression_reference(LIVE_EXPR_REF);
      }
    }
    
    //--------------------------------------------------------------------------
    void IndexSpaceNode::pack_expression_value(Serializer &rez,
                                               AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize<bool>(true/*index space*/);
      rez.serialize(handle);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::try_add_live_reference(void)
    //--------------------------------------------------------------------------
    {
      if (check_global_and_increment(LIVE_EXPR_REF))
      {
        ImplicitReferenceTracker::record_live_expression(this);
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_base_expression_reference(
                                         ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_nested_expression_reference(
                                           DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_base_expression_reference(
                                         ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_nested_expression_reference(
                                           DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::is_below_in_tree(IndexPartNode *partition, 
                                          LegionColor &child) const
    //--------------------------------------------------------------------------
    {
      const IndexSpaceNode *node = this;
      while ((node->parent != NULL) && 
              (node->parent->depth <= partition->depth))
      {
        if (node->parent == partition)
        {
          child = node->color;
          return true;
        }
        node = node->parent->parent;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_tree_expression_reference(DistributedID id,
                                                       unsigned count)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(id, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_tree_expression_reference(DistributedID id,
                                                          unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(id, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::intersects_with(IndexSpaceNode *rhs, bool compute)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      if (rhs == this)
        return true;
      // We're about to do something expensive so if these are both 
      // in the same index space tree then walk up to a common partition
      // (if one exists) and see if it is disjoint
      if ((handle.get_tree_id() == rhs->handle.get_tree_id()) && 
          (parent != rhs->parent))
      {
        IndexSpaceNode *one = this;
        IndexSpaceNode *two = rhs;
        // Get them at the same depth
        while (one->depth > two->depth)
          one = one->parent->parent;
        while (one->depth < two->depth)
          two = two->parent->parent;
        // Handle the case where one dominates the other
        if (one == two)
          return true;
        // Now walk up until their parent is the same
        while (one->parent != two->parent)
        {
          one = one->parent->parent;
          two = two->parent->parent;
        }
        // If they have the same parent and it's not NULL and 
        // it is disjoint then they don't intersect if they are different
        if ((one->parent != NULL) && (one != two) && one->parent->is_disjoint())
          return false;
        // Otherwise fall through and do the expensive test
      }
      if (!compute)
        return true;
      IndexSpaceExpression *intersect = 
        context->intersect_index_spaces(this, rhs);
      return !intersect->is_empty();
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::intersects_with(IndexPartNode *rhs, bool compute)
    //--------------------------------------------------------------------------
    {
      return rhs->intersects_with(this, compute);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::dominates(IndexSpaceNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      if (rhs == this)
        return true;
      // We're about to do something expensive, so use the region tree
      // as an acceleration data structure to try to make our tests
      // more efficient. If these are in the same tree, see if we can
      // walk up the tree from rhs and find ourself
      if (handle.get_tree_id() == rhs->handle.get_tree_id())
      {
        // If we're the root of the tree we also trivially dominate
        if (depth == 0)
          return true;
        if (rhs->depth > depth)
        {
          IndexSpaceNode *temp = rhs;
          while (depth < temp->depth)
            temp = temp->parent->parent;
          // If we find ourself at the same depth then we dominate
          if (temp == this)
            return true;
        }
        // Otherwise we fall through and do the expensive test
      }
      IndexSpaceExpression *diff = 
        context->subtract_index_spaces(rhs, this);
      return diff->is_empty();
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(RegionTreeForest *ctx, IndexPartition p, 
                                 IndexSpaceNode *par, IndexSpaceNode *color_sp,
                                 LegionColor c, bool dis, int comp, 
                                 DistributedID did, RtEvent init,
                                 CollectiveMapping *mapping, Provenance *prov)
      : IndexTreeNode(ctx, par->depth+1, c,
                      LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_PART_NODE_DC),
                      init, mapping, prov, true/*tree valid*/),
        handle(p), parent(par), color_space(color_sp), 
        total_children(color_sp->get_volume()), 
        max_linearized_color(color_sp->get_max_linearized_color()),
        total_children_volume(0), total_intersection_volume(0),
        has_disjoint(true), disjoint(dis),
        has_complete(comp >= 0), complete(comp != 0), first_entry(NULL)
    //--------------------------------------------------------------------------
    { 
      parent->add_nested_resource_ref(did);
      color_space->add_nested_resource_ref(did);
#ifdef DEBUG_LEGION
      assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Partition %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.id); 
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(RegionTreeForest *ctx, IndexPartition p, 
                                 IndexSpaceNode *par, IndexSpaceNode *color_sp,
                                 LegionColor c, int comp, DistributedID did,
                                 RtEvent init, CollectiveMapping *map,
                                 Provenance *prov)
      : IndexTreeNode(ctx, par->depth+1, c,
                      LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_PART_NODE_DC),
                      init, map, prov, true/*tree valid*/),
        handle(p), parent(par), color_space(color_sp), 
        total_children(color_sp->get_volume()),
        max_linearized_color(color_sp->get_max_linearized_color()),
        total_children_volume(0), total_intersection_volume(0),
        has_disjoint(false), disjoint(true),
        has_complete(comp >= 0), complete(comp != 0), first_entry(NULL)
    //--------------------------------------------------------------------------
    {
      parent->add_nested_resource_ref(did);
      color_space->add_nested_resource_ref(did);
#ifdef DEBUG_LEGION
      assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Partition %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.id);
#endif
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::initialize_disjoint_complete_notifications(void)
    //--------------------------------------------------------------------------
    {
      // Figure out how many notifications we're waiting for
      if (is_owner() || ((collective_mapping != NULL) && 
            collective_mapping->contains(local_space)))
      {
        remaining_local_disjoint_complete_notifications = 0;
        // Count how many locat notifications we're going to get
        for (ColorSpaceIterator itr(this, true/*local only*/); itr; itr++)
          remaining_local_disjoint_complete_notifications++;
        // One for the disjointness task that will run
        if (remaining_local_disjoint_complete_notifications > 0)
          remaining_global_disjoint_complete_notifications = 1;
        else
          remaining_global_disjoint_complete_notifications = 0;
        // More notifications from any remote nodes
        if (collective_mapping != NULL)
          remaining_global_disjoint_complete_notifications +=
            collective_mapping->count_children(owner_space, local_space);
        if (remaining_global_disjoint_complete_notifications == 0)
        {
#ifdef DEBUG_LEGION
          assert(!is_owner());
#endif
          const AddressSpaceID target =
            collective_mapping->get_parent(owner_space, local_space);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<int>(1); // up and compressed
            rez.serialize(total_children_volume);
            rez.serialize(total_intersection_volume);
          }
          runtime->send_index_partition_disjoint_update(target,rez,initialized);
        }
      }
      else
        remaining_global_disjoint_complete_notifications = 0;
      // Add a reference to be removed only after both the disjointness 
      // and the completeness is set
      add_base_valid_ref(REGION_TREE_REF);
    }

    //--------------------------------------------------------------------------
    IndexPartNode::~IndexPartNode(void)
    //--------------------------------------------------------------------------
    {
      // The reason we would be here is if we were leaked
      if (!partition_trackers.empty())
      {
        for (std::list<PartitionTracker*>::const_iterator it = 
              partition_trackers.begin(); it != partition_trackers.end(); it++)
          if ((*it)->remove_partition_reference())
            delete (*it);
        partition_trackers.clear();
      }
      // Lastly we can unregister ourselves with the context
      if (registered_with_runtime)
        context->remove_node(handle);
      if (parent->remove_nested_resource_ref(did))
        delete parent;
      if (color_space->remove_nested_resource_ref(did))
        delete color_space; 
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // Remove the valid reference that we hold on the color space
      if (color_space->parent != NULL)
        color_space->parent->remove_nested_valid_ref(did);
      else
        color_space->remove_nested_valid_ref(did);
      // Remove valid ref on partition of parent if it exists, otherwise
      // our parent index space is a root so we remove the reference there
      if (parent->parent != NULL)
      {
        if (parent->parent->remove_nested_valid_ref(did))
          delete parent->parent;
      }
      else
        parent->remove_nested_valid_ref(did);
      // Remove valid references on all owner children and any trackers
      // We should not need a lock at this point since nobody else should
      // be modifying the color map
      for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
            color_map.begin(); it != color_map.end(); it++)
        // Remove the nested valid reference on this index space node
        if (it->second->remove_nested_valid_ref(did))
          assert(false); // still holding resource ref so should never be hit
      if (!partition_trackers.empty())
      {
        for (std::list<PartitionTracker*>::const_iterator it = 
              partition_trackers.begin(); it != partition_trackers.end(); it++)
          if ((*it)->remove_partition_reference())
            delete (*it);
        partition_trackers.clear();
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      parent->remove_child(color);  
      for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
            color_map.begin(); it != color_map.end(); it++)
        if (it->second->remove_nested_gc_ref(did))
          delete it->second;
      color_map.clear();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_index_space_node(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::as_index_space_node(void)
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexPartNode::as_index_part_node(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }
#endif

    //--------------------------------------------------------------------------
    AddressSpaceID IndexPartNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle, context->runtime);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID IndexPartNode::get_owner_space(
                                          IndexPartition part, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (part.id % runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* IndexPartNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::send_semantic_request(AddressSpaceID target, 
             SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      context->runtime->send_index_partition_semantic_request(target, rez);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::send_semantic_info(AddressSpaceID target, 
                                           SemanticTag tag, const void *buffer,
                                           size_t size, bool is_mutable,
                                           RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      context->runtime->send_index_partition_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::process_semantic_request(SemanticTag tag, 
       AddressSpaceID source, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_owner_space() == context->runtime->address_space);
#endif
      RtEvent precondition;
      void *result = NULL;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          context->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_semantic_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      IndexPartNode *node = forest->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_semantic_info(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      forest->attach_semantic_information(handle, tag, source, buffer, size, 
                                          is_mutable, false/*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      return color_space->contains_color(c);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID IndexPartNode::find_color_creator_space(LegionColor color,
                                        CollectiveMapping *&child_mapping) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(child_mapping == NULL);
#endif
      if (collective_mapping == NULL)
      {
        if (is_owner())
          return local_space;
        else
          return owner_space;
      }
      else
      {
        // See whether the children are sharded or replicated
        if (((LegionColor)collective_mapping->size()) <= total_children)
        {
          // Sharded, so figure out which space to send the request to
          const size_t chunk = (max_linearized_color + 
              collective_mapping->size() - 1) / collective_mapping->size();
          const unsigned offset = color / chunk;
#ifdef DEBUG_LEGION
          assert(offset < collective_mapping->size());
#endif
          return (*collective_mapping)[offset];
        }
        else
        {
          // Replicated so find the child collective mapping
          std::vector<AddressSpaceID> child_spaces;
          const unsigned offset = color_space->compute_color_offset(color); 
#ifdef DEBUG_LEGION
          assert(offset < collective_mapping->size());
#endif
          for (unsigned idx = offset; 
                idx < collective_mapping->size(); idx += total_children)
            child_spaces.push_back((*collective_mapping)[idx]);
#ifdef DEBUG_LEGION
          assert(!child_spaces.empty());
#endif
          child_mapping = new CollectiveMapping(child_spaces, 
              context->runtime->legion_collective_radix);
          if (child_mapping->contains(local_space))
            return local_space;
          else
            return child_mapping->find_nearest(local_space);
        }
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::get_child(const LegionColor c,RtEvent *defer)
    //--------------------------------------------------------------------------
    {
      // First check to see if we can find it
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/); 
        std::map<LegionColor,IndexSpaceNode*>::const_iterator finder = 
          color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      if (!color_space->contains_color(c, false/*report error*/))
        REPORT_LEGION_ERROR(ERROR_INVALID_INDEX_SPACE_COLOR,
                            "Invalid color space color for child %lld "
                            "of partition %d", c, handle.get_id())
      // Retake the lock and see if we're going to be the one responsible
      // for trying to make the child on this node
      RtUserEvent ready_event;
      {
        AutoLock n_lock(node_lock);
        // Make sure we didn't lose the race
        std::map<LegionColor,IndexSpaceNode*>::const_iterator finder =
          color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
        // See if we're the first ones to make this child
        std::map<LegionColor,RtUserEvent>::iterator pending_finder =
          pending_child_map.find(c);
        if (pending_finder != pending_child_map.end())
        {
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          ready_event = pending_finder->second;
        }
        else
          pending_child_map[c] = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (!ready_event.exists())
      {
        // See if we need to send a request to get the handle for this
        CollectiveMapping *child_mapping = NULL;
        AddressSpaceID creator_space = 
          find_color_creator_space(c, child_mapping);
        if (creator_space != local_space)
        {
          if (child_mapping != NULL)
            delete child_mapping;
          // Find or get the ready event to wait on
          AutoLock n_lock(node_lock);
          std::map<LegionColor,IndexSpaceNode*>::const_iterator finder =
            color_map.find(c);
          if (finder != color_map.end())
            return finder->second;
          // If we get here then we need to send the request
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(c);
          }
          context->runtime->send_index_partition_child_request(creator_space,
                                                               rez);
          // Make sure we have to event to wait on for when the child is ready
          std::map<LegionColor,RtUserEvent>::iterator pending_finder =
            pending_child_map.find(c);
#ifdef DEBUG_LEGION
          assert(pending_finder != pending_child_map.end());
#endif
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          ready_event = pending_finder->second;
        }
        else if ((child_mapping != NULL) && 
            (local_space != child_mapping->get_origin()))
        {
          // We're not the origin that will make IDs, so retake the lock
          // and see if the child has appeared, if not record ourselves as
          // a pending waiter for it
          delete child_mapping;
          AutoLock n_lock(node_lock);
          std::map<LegionColor,IndexSpaceNode*>::const_iterator finder =
            color_map.find(c);
          if (finder != color_map.end())
            return finder->second;
          std::map<LegionColor,RtUserEvent>::iterator pending_finder =
            pending_child_map.find(c);
#ifdef DEBUG_LEGION
          assert(pending_finder != pending_child_map.end());
#endif
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          ready_event = pending_finder->second;
        }
        else
        {
          // If we get here then we're the ones to actually make the name 
          // of the index subspace and instantiate the node
#ifdef DEBUG_LEGION
          assert(is_owner() || ((collective_mapping != NULL) &&
                collective_mapping->contains(local_space)));
          assert((child_mapping == NULL) || 
              (local_space == child_mapping->get_origin()));
#endif
          IndexSpace is(context->runtime->get_unique_index_space_id(),
                        handle.get_tree_id(), handle.get_type_tag());
          const IndexSpaceExprID expr_id =
            context->runtime->get_unique_index_space_expr_id();
          DistributedID child_did = 
            context->runtime->get_available_distributed_id();
          // Make a new index space node ready when the partition is ready
          IndexSpaceNode *result = context->create_node(is, *this, c, child_did,
                               initialized, provenance, expr_id, child_mapping);
          if ((child_mapping != NULL) && (child_mapping->size() > 1))
          {
            // We know other participants are nodes are going to need
            // these IDs so broadcast them up to the other nodes that
            // are also going to consider child as a local child
            std::vector<AddressSpaceID> children;
            child_mapping->get_children(local_space, local_space, children);
#ifdef DEBUG_LEGION
            assert(!children.empty());
#endif
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(c);
              rez.serialize(is);
              rez.serialize(child_did);
              rez.serialize(expr_id);
              child_mapping->pack(rez);
            }
            for (std::vector<AddressSpaceID>::const_iterator it =
                  children.begin(); it != children.end(); it++)
              context->runtime->send_index_partition_child_replication(*it,rez);
          }
          if (runtime->legion_spy_enabled)
            LegionSpy::log_index_subspace(handle.id, is.id, 
                runtime->address_space, result->get_domain_point_color());
          if (implicit_profiler != NULL)
            implicit_profiler->register_index_subspace(handle.id, is.id,
                result->get_domain_point_color());
          return result;
        }
      }
#ifdef DEBUG_LEGION
      assert(ready_event.exists());
#endif
      if (defer == NULL)
      {
        ready_event.wait();
        return get_child(c);
      }
      else
      {
        *defer = ready_event;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_child_replication(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition parent_handle;
      derez.deserialize(parent_handle);
      LegionColor child_color;
      derez.deserialize(child_color);
      IndexSpace child_handle;
      derez.deserialize(child_handle);
      DistributedID child_did;
      derez.deserialize(child_did);
      IndexSpaceExprID expr_id;
      derez.deserialize(expr_id);
      size_t num_spaces;
      derez.deserialize(num_spaces);
#ifdef DEBUG_LEGION
      assert(num_spaces > 0);
#endif
      CollectiveMapping *mapping = new CollectiveMapping(derez, num_spaces);

      IndexPartNode *parent = forest->get_node(parent_handle);
      forest->create_node(child_handle, *parent, child_color, child_did,
              parent->initialized, parent->provenance, expr_id, mapping);
      std::vector<AddressSpaceID> children;
      mapping->get_children(mapping->get_origin(),parent->local_space,children);
      if (!children.empty())
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(parent_handle);
          rez.serialize(child_color);
          rez.serialize(child_handle);
          rez.serialize(child_did);
          rez.serialize(expr_id);
          mapping->pack(rez);
        }
        for (std::vector<AddressSpaceID>::const_iterator it =
              children.begin(); it != children.end(); it++)
          forest->runtime->send_index_partition_child_replication(*it, rez);
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_child(IndexSpaceNode *child) 
    //--------------------------------------------------------------------------
    {
      // This child should live as long as we are alive
      child->add_nested_gc_ref(did);
      child->add_nested_valid_ref(did);
      RtUserEvent to_trigger;
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(is_valid());
        assert(color_map.find(child->color) == color_map.end());
#endif
        color_map[child->color] = child;
        std::map<LegionColor,RtUserEvent>::iterator finder =
          pending_child_map.find(child->color);
        if (finder != pending_child_map.end())
        {
          if (finder->second.exists())
            to_trigger = finder->second; 
          pending_child_map.erase(finder);
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::set_child(IndexSpaceNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (!has_disjoint || !has_complete)
      {
#ifdef DEBUG_LEGION
        assert(remaining_local_disjoint_complete_notifications > 0);
        assert(remaining_global_disjoint_complete_notifications > 0);
#endif
        if (--remaining_local_disjoint_complete_notifications == 0)
        {
          // Launch the task to perform the local disjointness 
          // and completeness tests
          DisjointnessArgs args(this);
          runtime->issue_runtime_meta_task(args,
              LG_THROUGHPUT_DEFERRED_PRIORITY, initialized);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_tracker(PartitionTracker *tracker)
    //--------------------------------------------------------------------------
    {
      std::vector<PartitionTracker*> to_prune;
      {
        AutoLock n_lock(node_lock);
        // To avoid leaks, see if there are any other trackers we can prune
        for (std::list<PartitionTracker*>::iterator it =
              partition_trackers.begin(); it != 
              partition_trackers.end(); /*nothing*/)
        {
          if ((*it)->can_prune())
          {
            to_prune.push_back(*it);
            it = partition_trackers.erase(it);
          }
          else
            it++;
        }
        partition_trackers.push_back(tracker);
      }
      for (std::vector<PartitionTracker*>::const_iterator it =
            to_prune.begin(); it != to_prune.end(); it++)
        if ((*it)->remove_reference())
          delete (*it);
    }

    //--------------------------------------------------------------------------
    size_t IndexPartNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return color_space->get_volume();
    }

    //--------------------------------------------------------------------------
    IndexPartNode::RemoteDisjointnessFunctor::RemoteDisjointnessFunctor(
                                              Serializer &r, Runtime *rt)
      : rez(r), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::RemoteDisjointnessFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != runtime->address_space)
        runtime->send_index_partition_disjoint_update(target, rez);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::compute_disjointness_and_completeness(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner() || ((collective_mapping != NULL) &&
            collective_mapping->contains(local_space)));
#endif
      if (is_complete(false/*from app*/, true/*false if not ready*/) ||
          is_disjoint(false/*from app*/, true/*false if not ready*/))
      {
        // If we know we're complete, then we can check disjointness
        // simply by summing up the volume of all the children and 
        // seeing if it equals the total volume of the parent. If it
        // does then we must be disjoint since any aliasing would result
        // in a volume that is larger than the volume of the parent.
        //
        // If we know we're disjoint, then we can check completeness
        // simply by summing up the volume of all the children and seeing
        // if it equals the the total volume of the parent. If it does then
        // we must be complete because there is no aliasing of the subspaces.
        //
        // If we have no collective mapping or the number of children is
        // larger than the collective mapping then we can eagerly sum
        // the volumes together, otherwise we need to keep them separte
        // so we can deduplicate the volumes across nodes
        if ((collective_mapping == NULL) || 
            (((LegionColor)collective_mapping->size()) <= total_children))
        {
          // Children are sharded so no need to worry about uniqueness
          uint64_t children_volume = 0;
          for (ColorSpaceIterator itr(this, true/*local only*/); itr; itr++)
          {
            IndexSpaceNode *child = get_child(*itr);
            children_volume += child->get_volume();
          }
          return update_disjoint_complete_result(children_volume);
        }
        else
        {
          // Worry about uniqueness of children in this case
          std::map<LegionColor,uint64_t> children_volumes;
          for (ColorSpaceIterator itr(this, true/*local only*/); itr; itr++)
          {
            IndexSpaceNode *child = get_child(*itr);
            children_volumes[*itr] = child->get_volume();
          }
          return update_disjoint_complete_result(children_volumes);
        }
      }
      else
      {
        // In this case we don't know anything so we're computing both
        // disjointness and completeness at the same time. 
        // To check for disjointness we look for any neighboring 
        // children that alias. If we find any then we know that we
        // are not disjoint. To check for completeness, we count the
        // total volume of all the children and then subtract off the
        // volumes of the intersections from any interfering children
        // with a lower legion color to deduplicate counts. To compute
        // this difference for a given color C we first compute the union
        // of all the interfering children with colors <C and then subtract
        // that off the C to create a differende D, then we sum the 
        // intersection of all the remaining interfering children with D
        // Try drawing yourself n-way venn diagrams to convince yourself 
        // this is correct and will count all overlapping points exactly once.
        if ((collective_mapping == NULL) ||
            (((LegionColor)collective_mapping->size()) <= total_children))
        {
          // Children are sharded so no need to worry about uniqueness
          uint64_t children_volume = 0;
          uint64_t intersection_volume = 0;
          for (ColorSpaceIterator itr(this, true/*local only*/); itr; itr++)
          {
            IndexSpaceNode *child = get_child(*itr);
            size_t child_volume = child->get_volume();
            if (child_volume == 0)
              continue;
            children_volume += child_volume;
            std::vector<LegionColor> interfering;
            if (!find_interfering_children_kd(child, interfering))
            {
              // Not enough entries for a kd-tree so do it locally
              IndexSpaceExpression *difference = NULL;
              std::set<IndexSpaceExpression*> previous;
              for (ColorSpaceIterator itr2(this); itr2; itr2++)
              {
                if ((*itr) == (*itr2))
                {
                  if (previous.empty())
                    difference = child;
                  else
                    difference = 
                      context->subtract_index_spaces(child,
                          context->union_index_spaces(previous));
                }
                else
                {
                  IndexSpaceNode *other = get_child(*itr2);
                  if ((*itr) < (*itr2))
                  {
                    IndexSpaceExpression *intersection = 
                      context->intersect_index_spaces(difference, other);
                    intersection_volume += intersection->get_volume();
                  }
                  else
                  {
                    IndexSpaceExpression *intersection = 
                      context->intersect_index_spaces(child, other);
                    if (!intersection->is_empty())
                      previous.insert(intersection);
                  }
                }
              }
            }
            else
            {
#ifdef DEBUG_LEGION
              assert(!interfering.empty());
              std::sort(interfering.begin(), interfering.end());
              assert(std::binary_search(interfering.begin(),
                    interfering.end(), *itr));
#endif
              if (interfering.size() > 1)
              {
                IndexSpaceExpression *difference = NULL;
                std::set<IndexSpaceExpression*> previous;
                for (std::vector<LegionColor>::const_iterator it =
                      interfering.begin(); it != interfering.end(); it++)
                {
                  if ((*itr) == (*it))
                  {
                    IndexSpaceNode *child = get_child(*it);
                    if (previous.empty())
                      difference = child;
                    else
                      difference =
                        context->subtract_index_spaces(child,
                            context->union_index_spaces(previous));
                  }
                  else
                  {
                    IndexSpaceNode *other = get_child(*it);
                    if ((*itr) < (*it))
                    {
                      IndexSpaceExpression *intersection =
                        context->intersect_index_spaces(difference, other);
                      intersection_volume += intersection->get_volume();
                    }
                    else
                      previous.insert(other);
                  }
                }
              }
            }
          }
          return update_disjoint_complete_result(children_volume,
                                            intersection_volume);
        }
        else
        {
          std::map<LegionColor,uint64_t> children_volumes;
          std::map<
            std::pair<LegionColor,LegionColor>,uint64_t> intersection_volumes;
          // Children are not sharded so we need to worry about aliasing
          // across the nodes for the same children
          for (ColorSpaceIterator itr(this, true/*local only*/); itr; itr++)
          {
            IndexSpaceNode *child = get_child(*itr);
            size_t child_volume = child->get_volume();
            children_volumes[*itr] = child_volume;
            if (child_volume == 0)
              continue;
            std::vector<LegionColor> interfering;
            if (!find_interfering_children_kd(child, interfering))
            {
              // Not enough entries for a kd-tree so do it locally
              IndexSpaceExpression *difference = NULL;
              std::set<IndexSpaceExpression*> previous;
              for (ColorSpaceIterator itr2(this); itr2; itr2++)
              {
                if ((*itr) == (*itr2))
                {
                  if (previous.empty())
                    difference = child;
                  else
                    difference = 
                      context->subtract_index_spaces(child,
                          context->union_index_spaces(previous));
                }
                else
                {
                  IndexSpaceNode *other = get_child(*itr2);
                  if ((*itr) < (*itr2))
                  {
                    IndexSpaceExpression *intersection = 
                      context->intersect_index_spaces(difference, other);
                    if (!intersection->is_empty())
                      intersection_volumes[std::make_pair(*itr,*itr2)] =
                        intersection->get_volume();
                  }
                  else
                  {
                    IndexSpaceExpression *intersection = 
                      context->intersect_index_spaces(child, other);
                    if (!intersection->is_empty())
                      previous.insert(intersection);
                  }
                }
              }
            }
            else
            {
#ifdef DEBUG_LEGION
              assert(!interfering.empty());
              std::sort(interfering.begin(), interfering.end());
              assert(std::binary_search(interfering.begin(),
                    interfering.end(), *itr));
#endif
              if (interfering.size() > 1)
              {
                IndexSpaceExpression *difference = NULL;
                std::set<IndexSpaceExpression*> previous;
                for (std::vector<LegionColor>::const_iterator it =
                      interfering.begin(); it != interfering.end(); it++)
                {
                  if ((*itr) == (*it))
                  {
                    IndexSpaceNode *child = get_child(*it);
                    if (previous.empty())
                      difference = child;
                    else
                      difference =
                        context->subtract_index_spaces(child,
                            context->union_index_spaces(previous));
                  }
                  else
                  {
                    IndexSpaceNode *other = get_child(*it);
                    if ((*itr) < (*it))
                    {
                      IndexSpaceExpression *intersection =
                        context->intersect_index_spaces(difference, other);
                      if (!intersection->is_empty())
                        intersection_volumes[std::make_pair(*itr,*it)] =
                          intersection->get_volume();
                    }
                    else
                      previous.insert(other);
                  }
                }
              }
            }
          }
          return update_disjoint_complete_result(children_volumes,
                                            &intersection_volumes);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::update_disjoint_complete_result(
                         uint64_t children_volume, uint64_t intersection_volume)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      total_children_volume += children_volume;
      total_intersection_volume += intersection_volume;
      // Check to see if we've seen all our arrivals
#ifdef DEBUG_LEGION
      assert(remaining_global_disjoint_complete_notifications > 0);
#endif
      if (--remaining_global_disjoint_complete_notifications == 0)
      {
        if (is_owner())
          return finalize_disjoint_complete();
        else
        {
          // Send the result up the tree
          const AddressSpaceID target =
            collective_mapping->get_parent(owner_space, local_space);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<int>(1); // up and compressed
            rez.serialize(total_children_volume);
            rez.serialize(total_intersection_volume);
          }
          runtime->send_index_partition_disjoint_update(target,rez,initialized);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::update_disjoint_complete_result(
                              std::map<LegionColor,uint64_t> &children_volumes,
                              std::map<std::pair<LegionColor,LegionColor>,
                                       uint64_t> *intersection_volumes)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (!total_children_volumes.empty())
      {
        for (std::map<LegionColor,uint64_t>::const_iterator it =
              children_volumes.begin(); it != children_volumes.end(); it++)
          total_children_volumes.insert(*it);
      }
      else
        total_children_volumes.swap(children_volumes);
      if (intersection_volumes != NULL)
      {
        if (!total_intersection_volumes.empty())
        {
          for (std::map<std::pair<LegionColor,LegionColor>,
                        uint64_t>::const_iterator it =
                intersection_volumes->begin(); it != 
                intersection_volumes->end(); it++)
            total_intersection_volumes.insert(*it);
        }
        else
          total_intersection_volumes.swap(*intersection_volumes);
      }
      // Check to see if we've seen all our arrivals
#ifdef DEBUG_LEGION
      assert(remaining_global_disjoint_complete_notifications > 0);
#endif
      if (--remaining_global_disjoint_complete_notifications == 0)
      {
        if (is_owner())
        {
          // We can now compute the final sums
#ifdef DEBUG_LEGION
          assert(total_children_volume == 0);
          assert(total_intersection_volume == 0);
#endif
          for (std::map<LegionColor,uint64_t>::const_iterator it =
                total_children_volumes.begin(); it !=
                total_children_volumes.end(); it++)
            total_children_volume += it->second;
          total_children_volumes.clear();
          for (std::map<std::pair<LegionColor,LegionColor>,
                          uint64_t>::const_iterator it =
                total_intersection_volumes.begin(); it !=
                total_intersection_volumes.end(); it++)
            total_intersection_volume += it->second;
          total_intersection_volumes.clear();
          return finalize_disjoint_complete();
        }
        else
        {
          // Send the result up the tree
          const AddressSpaceID target =
            collective_mapping->get_parent(owner_space, local_space);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<int>(-1); // up and not compressed
            rez.serialize<size_t>(total_children_volumes.size());
            for (std::map<LegionColor,uint64_t>::const_iterator it =
                  total_children_volumes.begin(); it !=
                  total_children_volumes.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second);
            }
            rez.serialize<size_t>(total_intersection_volumes.size());
            for (std::map<std::pair<LegionColor,LegionColor>,uint64_t>::
                  const_iterator it = total_intersection_volumes.begin();
                  it != total_intersection_volumes.end(); it++)
            {
              rez.serialize(it->first.first);
              rez.serialize(it->first.second);
              rez.serialize(it->second);
            }
          }
          runtime->send_index_partition_disjoint_update(target,rez,initialized);
          total_children_volumes.clear();
          total_intersection_volumes.clear();
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::finalize_disjoint_complete(void)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        const size_t parent_volume = parent->get_volume();
        // We can now tell what our status is
        if (is_complete(false/*from app*/, true/*false if not ready*/))
        {
#ifdef DEBUG_LEGION
          assert(parent_volume <= total_children_volume);
#endif
          disjoint.store((parent_volume == total_children_volume));
        }
        else if (is_disjoint(false/*from app*/, true/*false if not ready*/))
        {
#ifdef DEBUG_LEGION
          assert(total_children_volume <= parent_volume);
#endif
          complete.store((parent_volume == total_children_volume));
        }
        else
        {
#ifdef DEBUG_LEGION
          assert((total_children_volume - total_intersection_volume) <=
                  parent_volume);
#endif
          if (total_intersection_volume == 0)
          {
            disjoint.store(true);
#ifdef DEBUG_LEGION
            assert((total_children_volume <= parent_volume));
#endif
            complete.store((total_children_volume == parent_volume));
          }
          else
          {
            disjoint.store(false);
#ifdef DEBUG_LEGION
            assert(total_intersection_volume < total_children_volume);
#endif
            total_children_volume -= total_intersection_volume;
#ifdef DEBUG_LEGION
            assert(total_children_volume <= parent_volume);
#endif
            complete.store((parent_volume == total_children_volume));
          }
        }
        if (implicit_profiler != NULL)
          implicit_profiler->register_index_partition(parent->handle.id,
              handle.id, disjoint.load(), color);
      }
      has_disjoint.store(true);
      has_complete.store(true);
      if (disjoint_complete_ready.exists())
        Runtime::trigger_event(disjoint_complete_ready);
      if ((collective_mapping != NULL) && 
          collective_mapping->contains(local_space))
      {
        // Broadcast the result out to the children
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, 
                                local_space, children);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<int>(0); // down
          rez.serialize<bool>(disjoint.load());
          rez.serialize<bool>(complete.load());
        }
        for (std::vector<AddressSpaceID>::const_iterator it =
              children.begin(); it != children.end(); it++)
          runtime->send_index_partition_disjoint_update(*it, rez);
      }
      if (has_remote_instances())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<int>(0); // down
          rez.serialize<bool>(disjoint.load());
          rez.serialize<bool>(complete.load());
        }
        RemoteDisjointnessFunctor functor(rez, context->runtime);
        map_over_remote_instances(functor);
      }
      return remove_base_valid_ref(REGION_TREE_REF);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_disjoint(bool app_query, bool false_if_not_ready)
    //--------------------------------------------------------------------------
    {
      if (has_disjoint.load())
        return disjoint.load();
      if (false_if_not_ready)
        return false;
      RtEvent wait_on;
      {
        AutoLock n_lock(node_lock);
        if (has_disjoint.load())
          return disjoint.load();
        if (!disjoint_complete_ready.exists())
          disjoint_complete_ready = Runtime::create_rt_user_event();
        wait_on = disjoint_complete_ready;
      }
      wait_on.wait();
#ifdef DEBUG_LEGION
      assert(has_disjoint.load());
#endif
      return disjoint.load();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::are_disjoint(LegionColor c1, LegionColor c2,
                                     bool force_compute)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return false;
      if (!force_compute && is_disjoint(false/*appy query*/))
        return true;
      if (c1 > c2)
        std::swap(c1, c2);
      const std::pair<LegionColor,LegionColor> key(c1,c2);
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        if (disjoint_subspaces.find(key) != disjoint_subspaces.end())
          return true;
        else if (aliased_subspaces.find(key) != aliased_subspaces.end())
          return false;
      }
      // Perform the test
      IndexSpaceNode *left = get_child(c1);
      IndexSpaceNode *right = get_child(c2);
      const bool intersects = left->intersects_with(right,   
            !context->runtime->disable_independence_tests);
      AutoLock n_lock(node_lock);
      if (intersects)
      {
        aliased_subspaces.insert(key);
        return false;
      }
      else
      {
        disjoint_subspaces.insert(key);
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_complete(bool from_app/*=false*/, 
                                    bool false_if_not_ready/*=false*/)
    //--------------------------------------------------------------------------
    {
      if (has_complete.load())
        return complete.load();
      if (false_if_not_ready)
        return false;
      RtEvent wait_on;
      {
        AutoLock n_lock(node_lock);
        if (has_complete.load())
          return complete.load();
        if (!disjoint_complete_ready.exists())
          disjoint_complete_ready = Runtime::create_rt_user_event();
        wait_on = disjoint_complete_ready;
      }
      wait_on.wait();
#ifdef DEBUG_LEGION
      assert(has_complete.load());
#endif
      return complete.load();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::handle_disjointness_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      int mode;
      derez.deserialize(mode);
      if (mode < 0)
      {
        // up and not compressed
        std::map<LegionColor,uint64_t> children_volumes;
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          LegionColor color;
          derez.deserialize(color);
          derez.deserialize(children_volumes[color]);
        }
        size_t num_intersections;
        derez.deserialize(num_intersections);
        std::map<std::pair<LegionColor,LegionColor>,
                  uint64_t> intersection_volumes;
        for (unsigned idx = 0; idx < num_intersections; idx++)
        {
          std::pair<LegionColor,LegionColor> key;
          derez.deserialize(key.first);
          derez.deserialize(key.second);
          derez.deserialize(intersection_volumes[key]);
        }
        return update_disjoint_complete_result(children_volumes,
                                               &intersection_volumes);
      }
      else if (mode > 0)
      {
        // up and already compressed
        uint64_t children_volume, intersection_volume;
        derez.deserialize(children_volume);
        derez.deserialize(intersection_volume);
        return update_disjoint_complete_result(children_volume,
                                               intersection_volume);
      }
      else
      {
        // sending back down to the children
        bool is_disjoint, is_complete;
        derez.deserialize(is_disjoint);
        derez.deserialize(is_complete);
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(remaining_global_disjoint_complete_notifications == 0);
#endif
        disjoint.store(is_disjoint);
        complete.store(is_complete);
        return finalize_disjoint_complete();
      }
    }

    //--------------------------------------------------------------------------
    LegionColor IndexPartNode::get_colors(std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      color_space->instantiate_colors(colors);
      if (!colors.empty())
        return colors.front();
      else
        return 0;
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_equal_children(Operation *op,
                                                 size_t granularity) 
    //--------------------------------------------------------------------------
    {
      return parent->create_equal_children(op, this, granularity);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_weights(Operation *op, 
           const std::map<DomainPoint,FutureImpl*> &weights, size_t granularity)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_weights(op, this, weights, granularity);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_union(Operation *op, 
                                           IndexPartNode *left, 
                                           IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_union(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_intersection(Operation *op,
                                                  IndexPartNode *left,
                                                  IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_intersection(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_intersection(Operation *op,
                                                  IndexPartNode *original,
                                                  const bool dominates)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_intersection(op, this, original, dominates);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_difference(Operation *op,
                                                IndexPartNode *left,
                                                IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_difference(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_restriction(const void *transform,
                                                 const void *extent)
    //--------------------------------------------------------------------------
    {
      return color_space->create_by_restriction(this, transform, extent,
                     NT_TemplateHelper::get_dim(handle.get_type_tag()));
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_disjointness_computation(
                                     const void *args, RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      const DisjointnessArgs *dargs = (const DisjointnessArgs*)args;
      if (dargs->proxy_this->compute_disjointness_and_completeness())
        delete dargs->proxy_this;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::intersects_with(IndexSpaceNode *rhs, bool compute)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      // A very simple test but an obvious one
      if ((rhs->parent == this) || (parent == rhs))
        return true;
      // We're about to do something expensive so if these are both
      // in the same index space tree then walk up to a common partition
      // if one exists and see if it is disjoint
      if (handle.get_tree_id() == rhs->handle.get_tree_id())
      {
        IndexSpaceNode *one = parent;
        IndexSpaceNode *two = rhs;
        // Get them at the same depth
        while (one->depth > two->depth)
          one = one->parent->parent;
        while (one->depth < two->depth)
          two = two->parent->parent;
        // Handle the case where one dominates the other
        if (one == two)
          return true;
        // Now walk up until their parent is the same
        while (one->parent != two->parent)
        {
          one = one->parent->parent;
          two = two->parent->parent;
        }
        // If they have the same parent and it's not NULL and 
        // it is disjoint then they don't intersect if they are different
        if ((one->parent != NULL) && (one != two) && one->parent->is_disjoint())
          return false;
        // Otherwise fall through and do the expensive test
      }
      if (!compute)
        return true;
      std::vector<LegionColor> interfering;
      if (find_interfering_children_kd(rhs, interfering))
        return !interfering.empty();
      for (ColorSpaceIterator itr(this); itr; itr++)
      {
        IndexSpaceNode *child = get_child(*itr);
        IndexSpaceExpression *intersect = 
          context->intersect_index_spaces(child, rhs);
        if (!intersect->is_empty())
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::intersects_with(IndexPartNode *rhs, bool compute)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      // A very simple but obvious test to do
      if (rhs == this)
        return true;
      // Another special case: if they both have the same parent and at least
      // one of them is complete then they do alias
      if ((parent == rhs->parent) && (is_complete() || rhs->is_complete()))
        return true;
      // We're about to do something expensive so see if we can use
      // the region tree as an acceleration data structure first
      if ((handle.get_tree_id() == rhs->handle.get_tree_id()) && 
          (parent != rhs->parent))
      {
        // Parent's are not the same, go up until we find 
        // parents with a common partition
        IndexSpaceNode *one = parent;
        IndexSpaceNode *two = rhs->parent;
        // Get them at the same depth
        while (one->depth > two->depth)
          one = one->parent->parent;
        while (one->depth < two->depth)
          two = two->parent->parent;
        // Handle the case where one dominates the other
        if (one == two)
          return true;
        // Now walk up until their parent is the same
        while (one->parent != two->parent)
        {
          one = one->parent->parent;
          two = two->parent->parent;
        }
        // If they have the same parent and it's not NULL and
        // it is dijsoint then they don't intersect if they are different
        if ((one->parent != NULL) && (one != two) && one->parent->is_disjoint())
          return false;
        // Otherwise we fall through and do the expensive test
      }
      if (!compute)
        return true;
      if (parent != rhs->parent)
      {
        IndexSpaceExpression *intersect = 
          context->intersect_index_spaces(parent, rhs->parent);
        if (intersect->is_empty())
          return false;
      }
      // TODO::intersect KD-trees?
      return true;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::find_interfering_children(IndexSpaceExpression *expr,
                                               std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // This should only be called on disjoint and complete partitions
      assert(is_disjoint());
      assert(is_complete());
      assert(colors.empty());
#endif 
      // Check to see if we have this in the cache
      {
        AutoLock n_lock(node_lock);
        std::map<IndexSpaceExprID,InterferenceEntry>::iterator finder = 
          interference_cache.find(expr->expr_id);
        if (finder != interference_cache.end())
        {
          if (finder->second.expr_id != first_entry->expr_id)
          {
            InterferenceEntry *entry = &finder->second;
            // Remove it from its place in line
            if (entry->older != NULL)
              entry->older->newer = entry->newer;
            if (entry->newer != NULL)
              entry->newer->older = entry->older;
            // Move it to the front of the line
            entry->newer = NULL;
            entry->older = first_entry;
            first_entry->newer = entry;
            first_entry = entry;
          }
          // Record the result
          colors = finder->second.colors;
          return;
        }
      }
      // Do a quick test to see if this expression is below us in the 
      // index space tree which makes this computation simple
      LegionColor below_color = 0;
      if (!expr->is_below_in_tree(this, below_color))
      {
        // We can only test this here after we've ruled out the symbolic check
        if (expr->is_empty())
          return;
        if (!find_interfering_children_kd(expr, colors))
        {
          for (ColorSpaceIterator itr(this); itr; itr++)
          {
            IndexSpaceNode *child = get_child(*itr);
            IndexSpaceExpression *intersection = 
              context->intersect_index_spaces(expr, child);
            if (!intersection->is_empty())
              colors.push_back(*itr);
          }
        }
      }
      else
        colors.push_back(below_color);
      // Save the result in the cache for the future
      AutoLock n_lock(node_lock);
      // If someone else beat us to it then we are done
      if (interference_cache.find(expr->expr_id) != interference_cache.end())
        return;
      // Insert it at the front
      InterferenceEntry *entry = &interference_cache[expr->expr_id];
      entry->expr_id = expr->expr_id;
      entry->colors = colors;
      if (first_entry != NULL)
        first_entry->newer = entry;
      entry->older = first_entry;
      first_entry = entry;
      if (interference_cache.size() > MAX_INTERFERENCE_CACHE_SIZE)
      {
        // Remove the oldest entry in the cache
        InterferenceEntry *last_entry = first_entry; 
        while (last_entry->older != NULL)
          last_entry = last_entry->older;
        if (last_entry->newer != NULL)
          last_entry->newer->older = NULL;
        interference_cache.erase(last_entry->expr_id);
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::send_node(AddressSpaceID target, bool recurse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recurse);
      assert(parent != NULL);
#endif
      // Quick out if we've already sent this
      if (has_remote_instance(target))
        return;
      parent->send_node(target, true/*recurse*/);
      color_space->send_node(target, true/*recurse*/);
      // Only send it if we're the owner without a collective mapping
      // or the target is not in the collective mapping and we're the
      // closest node in the collective mapping to the target
      if ((is_owner() && (collective_mapping == NULL)) ||
          ((collective_mapping != NULL) && 
           !collective_mapping->contains(target) && 
           collective_mapping->contains(local_space) &&
           (local_space == collective_mapping->find_nearest(target))))
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(is_valid());
#endif
        if (!has_remote_instance(target))
        {
          Serializer rez;
          pack_node(rez, target);
          context->runtime->send_index_partition_response(target, rez);
          update_remote_instances(target);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::pack_node(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have computed the disjointness result
      // If not we'll record that we need to do it and then when it 
      // is computed we'll send out the result to all the remote copies
      RezCheck z(rez);
      rez.serialize(handle);
      rez.serialize(did);
      rez.serialize(parent->handle); 
      rez.serialize(color_space->handle);
      rez.serialize(color);
      rez.serialize<bool>(has_disjoint.load());
      rez.serialize<bool>(disjoint.load());
      if (has_complete)
      {
        if (complete)
          rez.serialize<int>(1); // complete
        else
          rez.serialize<int>(0); // not complete
      }
      else
        rez.serialize<int>(-1); // we don't know yet
      rez.serialize(initialized);
      if (collective_mapping != NULL)
        collective_mapping->pack(rez);
      else
        rez.serialize<size_t>(0); // total spaces
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      rez.serialize<size_t>(semantic_info.size());
      for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second.size);
        rez.serialize(it->second.buffer, it->second.size);
        rez.serialize(it->second.is_mutable);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_creation(
        RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      DistributedID did;
      derez.deserialize(did);
      IndexSpace parent;
      derez.deserialize(parent);
      IndexSpace color_space;
      derez.deserialize(color_space);
      LegionColor color;
      derez.deserialize(color);
      bool has_disjoint, disjoint;
      derez.deserialize(has_disjoint);
      derez.deserialize(disjoint);
      int complete;
      derez.deserialize(complete);
      RtEvent initialized;
      derez.deserialize(initialized);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping *mapping = NULL;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);
      AutoProvenance provenance(Provenance::deserialize(derez));
      IndexSpaceNode *parent_node = context->get_node(parent);
      IndexSpaceNode *color_space_node = context->get_node(color_space);
#ifdef DEBUG_LEGION
      assert(parent_node != NULL);
      assert(color_space_node != NULL);
#endif
      IndexPartNode *node = has_disjoint ?
        context->create_node(handle, parent_node, color_space_node, color, 
               disjoint, complete, did, provenance, initialized, mapping) :
        context->create_node(handle, parent_node, color_space_node, color,
               complete, did, provenance, initialized, mapping);
#ifdef DEBUG_LEGION
      assert(node != NULL);
#endif
      size_t num_semantic;
      derez.deserialize(num_semantic);
      for (unsigned idx = 0; idx < num_semantic; idx++)
      {
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void *buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(tag, source, buffer, buffer_size, 
                                          is_mutable, false/*local only*/);
      }
    } 

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_request(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexPartition handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      AddressSpaceID source;
      derez.deserialize(source);
      IndexPartNode *target = forest->get_node(handle, NULL, true/*can fail*/);
      if (target != NULL)
      {
        // If there is a collective mapping, check to see if we're on the
        // right node and if not forward it on to the right node
        if (target->collective_mapping != NULL)
        {
#ifdef DEBUG_LEGION
          assert(!target->collective_mapping->contains(source));
          assert(target->collective_mapping->contains(target->local_space));
#endif
          if (target->is_owner())
          {
            const AddressSpaceID nearest = 
              target->collective_mapping->find_nearest(source);
            // If we're not the nearest then forward it on to the
            // proper node to handle the request
            if (nearest != target->local_space)
            {
              Serializer rez;
              rez.serialize(handle);
              rez.serialize(to_trigger);
              rez.serialize(source);
              forest->runtime->send_index_partition_request(nearest, rez);
              return;
            }
          }
#ifdef DEBUG_LEGION
          else
          {
            assert(target->local_space == 
                target->collective_mapping->find_nearest(source));
          }
#endif
        }
        target->pack_valid_ref();
        target->send_node(source, true/*recurse*/);
        // Now send back the results
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(to_trigger);
          rez.serialize(handle);
        }
        forest->runtime->send_index_partition_return(source, rez);
      }
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_return(
                                 RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode *node = context->get_node(handle);
      node->unpack_valid_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_child_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      LegionColor child_color;
      derez.deserialize(child_color);
      IndexPartNode *parent = forest->get_node(handle);
      RtEvent defer;
      IndexSpaceNode *child = parent->get_child(child_color, &defer);
      // If we got a deferral event then we need to make a continuation
      // to avoid blocking the virtual channel for nested index tree requests
      if (defer.exists())
      {
        DeferChildArgs args(parent, child_color, source);
        forest->runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, defer);
      }
      else
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(child->handle);
        }
        forest->runtime->send_index_partition_child_response(source, rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::defer_node_child_request(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferChildArgs *dargs = (const DeferChildArgs*)args;
      IndexSpaceNode *child = dargs->proxy_this->get_child(dargs->child_color);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(child->handle);
      }
      Runtime *runtime = dargs->proxy_this->context->runtime;
      runtime->send_index_partition_child_response(dargs->source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/void IndexPartNode::defer_find_local_shard_rects(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferFindShardRects *dargs = (const DeferFindShardRects*)args;
      if (dargs->proxy_this->find_local_shard_rects())
        delete dargs->proxy_this;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_child_response(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      forest->find_or_request_node(handle, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_disjoint_update(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode *node = forest->get_node(handle);
      if (node->handle_disjointness_update(derez))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_notification(RegionTreeForest *forest,
                                                       Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition pid;
      derez.deserialize(pid);
      IndexSpace parent;
      derez.deserialize(parent);
      LegionColor part_color;
      derez.deserialize(part_color);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      IndexSpaceNode *parent_node = forest->get_node(parent);
      parent_node->record_remote_child(pid, part_color);
      // Now we can trigger the done event
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexPartNode::request_shard_rects(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      std::vector<AddressSpaceID> children;
      {
        AutoLock n_lock(node_lock);
        if (shard_rects_ready.exists())
          return shard_rects_ready;
        shard_rects_ready = Runtime::create_rt_user_event();
        // Add a reference to keep this node alive until this all done
        add_base_gc_ref(RUNTIME_REF);
        // Figure out how many downstream requests we have
        collective_mapping->get_children(owner_space, local_space, children);
        // Need to see all our children notifications plus our local rectangles 
        remaining_rect_notifications = children.size() + 1;
        initialize_shard_rects();
      }
      if (!children.empty())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
        }
        for (std::vector<AddressSpaceID>::const_iterator it = 
              children.begin(); it != children.end(); it++)
          context->runtime->send_index_partition_shard_rects_request(*it, rez);
      }
      // Compute our local shard rectangles
      if (find_local_shard_rects())
        assert(false); // should never delete ourselves
      return shard_rects_ready;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::process_shard_rects_response(Deserializer &derez,
                                                     AddressSpace source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      bool up;
      derez.deserialize<bool>(up);
      if (up)
      {
        bool need_local = false;
        std::vector<AddressSpaceID> children;
        AutoLock n_lock(node_lock);
        if (!shard_rects_ready.exists())
        {
          // Not initialized, so do the initialization
          shard_rects_ready = Runtime::create_rt_user_event();
          // Add a reference to keep this node alive until this all done
          add_base_gc_ref(RUNTIME_REF);
          // Figure out how many downstream requests we have
          collective_mapping->get_children(owner_space, local_space, children);
#ifdef DEBUG_LEGION
          assert(!children.empty());
          bool found = false;
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
          {
            if (*it != source)
              continue;
            found = true;
            break;
          }
          assert(found);
#endif
          need_local = true;
          remaining_rect_notifications = children.size() + 1;
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
          }
          for (std::vector<AddressSpaceID>::const_iterator it = 
                children.begin(); it != children.end(); it++)
            if ((*it) != source)
              context->runtime->send_index_partition_shard_rects_request(*it,
                                                                         rez);
          initialize_shard_rects();
        }
#ifdef DEBUG_LEGION
        else
        {
          collective_mapping->get_children(owner_space, local_space, children);
          assert(!children.empty());
          bool found = false;
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
          {
            if (*it != source)
              continue;
            found = true;
            break;
          }
          assert(found);
        }
#endif
        unpack_shard_rects(derez);
        if (perform_shard_rects_notification())
        {
#ifdef DEBUG_LEGION
          assert(!need_local);
#endif
          return true;
        }
        else if (!need_local)
          return false;
      }
      else
      {
        // Going down
        AutoLock n_lock(node_lock);
        unpack_shard_rects(derez);
#ifdef DEBUG_LEGION
        assert(shard_rects_ready.exists());
#endif
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        if (!children.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(false); // going down
            pack_shard_rects(rez, false/*clear*/);
          }
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
            context->runtime->send_index_partition_shard_rects_response(*it, 
                                                                        rez);
        }
        // Only trigger this after we've packed the shard rects since the
        // local node is going to mutate it with its own values after this
        Runtime::trigger_event(shard_rects_ready);
        return remove_base_gc_ref(RUNTIME_REF);
      }
      // If we get here then we need to kick off the local analysis
      return find_local_shard_rects();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::perform_shard_rects_notification(void)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
#ifdef DEBUG_LEGION
      assert(remaining_rect_notifications > 0);
#endif
      if (--remaining_rect_notifications == 0)
      {
        if (is_owner())
        {
#ifdef DEBUG_LEGION
          assert(shard_rects_ready.exists());
#endif
          std::vector<AddressSpaceID> children;
          collective_mapping->get_children(owner_space, local_space, children);
          // We've got all the data now, so we can broadcast it back out
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(false); // sending down the tree now
            pack_shard_rects(rez, false/*clear*/);
          }
          // Only trigger this after we've packed the shard rects since the
          // local node is going to mutate it with its own values after this
          Runtime::trigger_event(shard_rects_ready);
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
            context->runtime->send_index_partition_shard_rects_response(*it,
                                                                        rez);
          return remove_base_gc_ref(RUNTIME_REF);
        }
        else
        {
          // Continue propagating it back up the tree
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(true); // still going up
            pack_shard_rects(rez, true/*clear*/);
          }
          context->runtime->send_index_partition_shard_rects_response(
             collective_mapping->get_parent(owner_space, local_space), rez);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    IndexPartNode::RemoteKDTracker::RemoteKDTracker(Runtime *rt)
      : runtime(rt), done_event(RtUserEvent::NO_RT_USER_EVENT), remaining(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RtEvent IndexPartNode::RemoteKDTracker::find_remote_interfering(
        const std::set<AddressSpaceID> &targets, IndexPartition handle,
        IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remaining.load() == 0);
      assert(!targets.empty());
#endif
      remaining.store(targets.size());
      for (std::set<AddressSpaceID>::const_iterator it =
            targets.begin(); it != targets.end(); it++)
      {
        if ((*it) == runtime->address_space)
        {
#ifdef DEBUG_LEGION
          assert(remaining.load() > 0);
#endif
          if (remaining.fetch_sub(1) == 1)
            return RtEvent::NO_RT_EVENT;
          continue;
        }
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          expr->pack_expression(rez, *it);
          rez.serialize(this);
        }
        runtime->send_index_partition_remote_interference_request(*it, rez);
      }
      RtEvent wait_on;
      {
        AutoLock t_lock(tracker_lock);
        if (remaining.load() == 0)
          return RtEvent::NO_RT_EVENT;
        done_event = Runtime::create_rt_user_event();
        wait_on = done_event;
      }
      return wait_on;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::RemoteKDTracker::get_remote_interfering(
                                                  std::set<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      // No need for the lock since we're done at this point
      if (!remote_colors.empty())
      {
        if (colors.empty())
          colors.swap(remote_colors);
        else
          colors.insert(remote_colors.begin(), remote_colors.end());
      }
    }

    //--------------------------------------------------------------------------
    RtUserEvent 
      IndexPartNode::RemoteKDTracker::process_remote_interfering_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_colors;
      derez.deserialize(num_colors);
      AutoLock t_lock(tracker_lock);
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        LegionColor color;
        derez.deserialize(color);
        remote_colors.insert(color);
      }
#ifdef DEBUG_LEGION
      assert(remaining.load() > 0);
#endif
      if ((remaining.fetch_sub(1) == 1) && done_event.exists())
        return done_event;
      else
        return RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_shard_rects_request(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode *node = forest->get_node(handle);
      node->request_shard_rects();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_shard_rects_response(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode *node = forest->get_node(handle);
      if (node->process_shard_rects_response(derez, source))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_remote_interference_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, forest, source);
      RemoteKDTracker *tracker;
      derez.deserialize(tracker);
      
      IndexPartNode *node = forest->get_node(handle);
      std::vector<LegionColor> local_colors;
      node->find_interfering_children_kd(expr, local_colors,true/*local only*/);
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(tracker);
        rez.serialize<size_t>(local_colors.size());
        for (std::vector<LegionColor>::const_iterator it =
              local_colors.begin(); it != local_colors.end(); it++)
          rez.serialize(*it);
      }
      forest->runtime->send_index_partition_remote_interference_response(source,
                                                                         rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_remote_interference_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteKDTracker *tracker;
      derez.deserialize(tracker);
      const RtUserEvent to_trigger = 
        tracker->process_remote_interfering_response(derez);
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    /////////////////////////////////////////////////////////////
    // Color Space Iterator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ColorSpaceIterator::ColorSpaceIterator(IndexPartNode *partition,
                                           bool local_only)
      : color_space(partition->color_space)
    //--------------------------------------------------------------------------
    {
      simple_step = 
            (partition->total_children == partition->max_linearized_color);
      if (local_only && (partition->collective_mapping != NULL))
      {
#ifdef DEBUG_LEGION
        assert(partition->collective_mapping->contains(
              partition->local_space));
#endif
        const unsigned index = 
          partition->collective_mapping->find_index(partition->local_space);
        const LegionColor total_spaces = partition->collective_mapping->size();
        if (partition->total_children < total_spaces)
        {
          // Just a single color to handle here
          current = 0;
          end = partition->max_linearized_color;
          const unsigned offset = index % partition->total_children;
          for (unsigned idx = 0; idx < offset; idx++)
            step();
#ifdef DEBUG_LEGION
          assert(current < end);
#endif
          end = current+1;
        }
        else
        {
          const LegionColor chunk = 
            compute_chunk(partition->max_linearized_color, total_spaces);
          current = index * chunk;
          end = ((current + chunk) < partition->max_linearized_color) ?
            (current + chunk) : partition->max_linearized_color;
          if (!simple_step && (current < end) &&
              !color_space->contains_color(current))
            step();       
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!local_only || partition->is_owner());
#endif
        current = 0;
        end = partition->max_linearized_color;
      }
    }

    //--------------------------------------------------------------------------
    ColorSpaceIterator::ColorSpaceIterator(IndexPartNode *partition,
                                           ShardID shard, size_t total_shards)
      : color_space(partition->color_space)
    //--------------------------------------------------------------------------
    {
      simple_step = 
        (partition->total_children == partition->max_linearized_color);
      const LegionColor chunk = 
        (partition->max_linearized_color + total_shards - 1) / total_shards;
      current = shard * chunk;
      end = ((current + chunk) < partition->max_linearized_color) ?
        (current + chunk) : partition->max_linearized_color;
      if (!simple_step && (current < end) &&
          !color_space->contains_color(current))
        step();
    }

    //--------------------------------------------------------------------------
    /*static*/ LegionColor ColorSpaceIterator::compute_chunk(
                                     LegionColor max_color, size_t total_spaces)
    //--------------------------------------------------------------------------
    {
      return (max_color + total_spaces - 1) / total_spaces;
    }

    //--------------------------------------------------------------------------
    ColorSpaceIterator::operator bool(void) const
    //--------------------------------------------------------------------------
    {
      return (current < end);
    }

    //--------------------------------------------------------------------------
    LegionColor ColorSpaceIterator::operator*(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current < end);
#endif
      return current;
    }

    //--------------------------------------------------------------------------
    ColorSpaceIterator& ColorSpaceIterator::operator++(int)
    //--------------------------------------------------------------------------
    {
      step();
      return *this;
    }

    //--------------------------------------------------------------------------
    void ColorSpaceIterator::step(void)
    //--------------------------------------------------------------------------
    {
      current++;
      if (!simple_step)
      {
        while ((current < end) && !color_space->contains_color(current))
          current++;
      }
    }

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx,
                   DistributedID did, RtEvent init, CollectiveMapping *map,
                   Provenance *prov)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC), 
          false/*register with runtime*/, map),
        handle(sp), context(ctx), provenance(prov), initialized(init), 
        allocation_state((map != NULL) ? FIELD_ALLOC_COLLECTIVE :
            is_owner() ? FIELD_ALLOC_READ_ONLY : FIELD_ALLOC_INVALID), 
        outstanding_allocators(0), outstanding_invalidations(0)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        unallocated_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
        local_index_infos.resize(runtime->max_local_fields, 
            std::pair<size_t,CustomSerdezID>(0, 0));
        if (collective_mapping != NULL)
        {
          const CollectiveMapping &mapping = *collective_mapping;
          for (unsigned idx = 0; idx < mapping.size(); idx++)
          {
            const AddressSpaceID space = mapping[idx];
            if (space != local_space)
              remote_field_infos.insert(mapping[idx]);
          }
          // We can have control replication inside of just a single node
          if (remote_field_infos.empty())
            allocation_state = FIELD_ALLOC_READ_ONLY;
        }
      }
      else if (allocation_state == FIELD_ALLOC_COLLECTIVE)
        unallocated_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
      if (provenance != NULL)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info("GC Field Space %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.id);
#endif
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx,
         DistributedID did, RtEvent init, CollectiveMapping *map,
         Provenance *prov, Deserializer &derez)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC), 
          false/*register with runtime*/, map),
        handle(sp), context(ctx), provenance(prov), initialized(init), 
        allocation_state(FIELD_ALLOC_INVALID), outstanding_allocators(0),
        outstanding_invalidations(0)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      size_t num_fields;
      derez.deserialize(num_fields);
      if (num_fields > 0)
      {
        allocation_state = FIELD_ALLOC_READ_ONLY;
        for (unsigned idx = 0; idx < num_fields; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          field_infos[fid].deserialize(derez);
        }
      }
      if (provenance != NULL)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info("GC Field Space %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.id);
#endif
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::~FieldSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      // Next we can delete our layouts
      for (std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
            LAYOUT_DESCRIPTION_ALLOC>>::iterator it =
            layouts.begin(); it != layouts.end(); it++)
      {
        LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>
          &descs = it->second;
        for (LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::iterator
              it = descs.begin(); it != descs.end(); it++)
        {
          if ((*it)->remove_reference())
            delete (*it);
        }
      }
      layouts.clear();
      for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      }
      for (LegionMap<std::pair<FieldID,SemanticTag>,SemanticInfo>::iterator
            it = semantic_field_info.begin(); 
            it != semantic_field_info.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      }
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      // Unregister ourselves from the context
      if (registered_with_runtime)
        context->remove_node(handle);
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(void)
      : field_size(0), idx(0), serdez_id(0), provenance(NULL), 
        collective(false), local(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(size_t size, unsigned id, 
                       CustomSerdezID sid, Provenance *prov, bool loc, bool col)
      : field_size(size), idx(id), serdez_id(sid), provenance(prov),
        collective(col), local(loc)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(ApEvent ready, unsigned id,
                       CustomSerdezID sid, Provenance *prov, bool loc, bool col)
      : field_size(0), size_ready(ready), idx(id), serdez_id(sid),
        provenance(prov), collective(col), local(loc)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(const FieldInfo &rhs)
      : field_size(rhs.field_size), size_ready(rhs.size_ready), idx(rhs.idx),
        serdez_id(rhs.serdez_id), provenance(rhs.provenance),
        collective(rhs.collective), local(rhs.local)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(FieldInfo &&rhs)
      : field_size(rhs.field_size), size_ready(rhs.size_ready), idx(rhs.idx),
        serdez_id(rhs.serdez_id), provenance(rhs.provenance),
        collective(rhs.collective), local(rhs.local)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = NULL;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::~FieldInfo(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo& FieldSpaceNode::FieldInfo::operator=(
                                                           const FieldInfo &rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      field_size = rhs.field_size;
      size_ready = rhs.size_ready;
      idx = rhs.idx;
      serdez_id = rhs.serdez_id;
      provenance = rhs.provenance;
      collective = rhs.collective;
      local = rhs.local;
      if (provenance != NULL)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo& FieldSpaceNode::FieldInfo::operator=(
                                                                FieldInfo &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      field_size = rhs.field_size;
      size_ready = rhs.size_ready;
      idx = rhs.idx;
      serdez_id = rhs.serdez_id;
      provenance = rhs.provenance;
      collective = rhs.collective;
      local = rhs.local;
      rhs.provenance = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::FieldInfo::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(field_size);
      rez.serialize(size_ready);
      rez.serialize(idx);
      rez.serialize<bool>(collective);
      rez.serialize<bool>(local);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::FieldInfo::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(field_size);
      derez.deserialize(size_ready);
      derez.deserialize(idx);
      derez.deserialize<bool>(collective);
      derez.deserialize<bool>(local);
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    AddressSpaceID FieldSpaceNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle, context->runtime);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID FieldSpaceNode::get_owner_space(FieldSpace handle,
                                                              Runtime *rt)
    //--------------------------------------------------------------------------
    {
      return (handle.id % rt->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::attach_semantic_information(SemanticTag tag,
                                                     AddressSpaceID source,
                                                     const void *buffer, 
                                                     size_t size, 
                                                     bool is_mutable,
                                                     bool local_only)
    //--------------------------------------------------------------------------
    {
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          // First check to see if it is valid
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // Check to make sure that the bits are the same
              if (size != finder->second.size)
                REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                  "Inconsistent Semantic Tag value "
                              "for tag %ld with different sizes of %zd"
                              " and %zd for index tree node", 
                              tag, size, finder->second.size)
              // Otherwise do a bitwise comparison
              {
                const char *orig = (const char*)finder->second.buffer;
                const char *next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                    REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                    "Inconsistent Semantic Tag value "
                                  "for tag %ld with different values at"
                                  "byte %d for index tree node, %x != %x", 
                                  tag, idx, orig[idx], next[idx])
                }
              }
              added = false;
            }
            else
            {
              // Mutable so we can overwrite 
              legion_free(SEMANTIC_INFO_ALLOC, finder->second.buffer,
                          finder->second.size);
              finder->second.buffer = local;
              finder->second.size = size;
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer = local;
            finder->second.size = size;
            // Trigger will happen by caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_info[tag] = SemanticInfo(local, size, is_mutable);
      }
      if (added)
      {
        AddressSpaceID owner_space = get_owner_space();
        // If we are not the owner and the message 
        // didn't come from the owner, then send it 
        if ((owner_space != context->runtime->address_space) &&
            (source != owner_space) && !local_only)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          send_semantic_info(owner_space, tag, buffer, size, is_mutable, done);
          if (!done.has_triggered())
            done.wait();
        }
      }
      else
        legion_free(SEMANTIC_INFO_ALLOC, local, size);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::attach_semantic_information(FieldID fid,
                                                     SemanticTag tag,
                                                     AddressSpaceID source,
                                                     const void *buffer,
                                                     size_t size, 
                                                     bool is_mutable,
                                                     bool local_only)
    //--------------------------------------------------------------------------
    {
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<std::pair<FieldID,SemanticTag>,SemanticInfo>::iterator finder
          = semantic_field_info.find(std::pair<FieldID,SemanticTag>(fid,tag));
        if (finder != semantic_field_info.end())
        {
          // First check to see if it is valid
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // Check to make sure that the bits are the same
              if (size != finder->second.size)
                REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                              "Inconsistent Semantic Tag value "
                              "for tag %ld with different sizes of %zd"
                              " and %zd for index tree node", 
                              tag, size, finder->second.size)
              // Otherwise do a bitwise comparison
              {
                const char *orig = (const char*)finder->second.buffer;
                const char *next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                    REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                                        "Inconsistent Semantic Tag value "
                                  "for tag %ld with different values at"
                                  "byte %d for index tree node, %x != %x", 
                                  tag, idx, orig[idx], next[idx])
                }
              }
              added = false;
            }
            else
            {
              // Mutable so we can overwrite
              legion_free(SEMANTIC_INFO_ALLOC, finder->second.buffer,
                          finder->second.size);
              finder->second.buffer = local;
              finder->second.size = size;
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer = local;
            finder->second.size = size;
            // Trigger will happen by caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
        {
          semantic_field_info[std::pair<FieldID,SemanticTag>(fid,tag)] = 
            SemanticInfo(local, size, is_mutable);
        }
      }
      if (added)
      {
        AddressSpaceID owner_space = get_owner_space();
        // If we are not the owner and the message 
        // didn't come from the owner, then send it 
        if ((owner_space != context->runtime->address_space) &&
            (source != owner_space) && !local_only)
          send_semantic_field_info(owner_space, fid, tag, 
                                   buffer, size, is_mutable); 
      }
      else
        legion_free(SEMANTIC_INFO_ALLOC, local, size);
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::retrieve_semantic_information(SemanticTag tag,
              const void *&result, size_t &size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != context->runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        LegionMap<SemanticTag,SemanticInfo>::const_iterator finder = 
          semantic_info.find(tag); 
        if (finder != semantic_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else // can use the canonical event
              wait_on = finder->second.ready_event; 
          }
          else if (wait_until) // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        // Nothing to wait on so we have to do something
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                      "invalid semantic tag %ld for "
                      "field space %d", tag, handle.id)
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(tag);
            rez.serialize(can_fail);
            rez.serialize(wait_until);
            rez.serialize(wait_on);
          }
          context->runtime->send_field_space_semantic_request(owner_space, rez);
        }
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock,1,false/*exclusive*/); 
      LegionMap<SemanticTag,SemanticInfo>::const_iterator finder = 
        semantic_info.find(tag);
      if (finder == semantic_info.end())
      {
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                            "invalid semantic tag %ld for "
                            "field space %d", tag, handle.id)
      }
      result = finder->second.buffer;
      size = finder->second.size;
      return true;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::retrieve_semantic_information(FieldID fid,
                             SemanticTag tag, const void *&result, size_t &size,
                             bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != context->runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        LegionMap<std::pair<FieldID,SemanticTag>,
                  SemanticInfo>::const_iterator finder = 
            semantic_field_info.find(std::pair<FieldID,SemanticTag>(fid,tag));
        if (finder != semantic_field_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else // can use the canonical event
              wait_on = finder->second.ready_event; 
          }
          else if (wait_until) // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG,
          "invalid semantic tag %ld for field %d "
                      "of field space %d", tag, fid, handle.id)
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(fid);
            rez.serialize(tag);
            rez.serialize(can_fail);
            rez.serialize(wait_until);
            rez.serialize(wait_on);
          }
          context->runtime->send_field_semantic_request(owner_space, rez);
        }
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock,1,false/*exclusive*/); 
      LegionMap<std::pair<FieldID,SemanticTag>,
        SemanticInfo>::const_iterator finder = 
          semantic_field_info.find(std::pair<FieldID,SemanticTag>(fid,tag));
      if (finder == semantic_field_info.end())
      {
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG,
                            "invalid semantic tag %ld for field %d "
                            "of field space %d", tag, fid, handle.id)
      }
      result = finder->second.buffer;
      size = finder->second.size;
      return true;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_semantic_info(AddressSpaceID target, 
                 SemanticTag tag, const void *result, size_t size, 
                 bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(result, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      context->runtime->send_field_space_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_semantic_field_info(AddressSpaceID target,
                  FieldID fid, SemanticTag tag, const void *result, 
                  size_t size, bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(fid);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(result, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      context->runtime->send_field_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_semantic_request(SemanticTag tag,
       AddressSpaceID source, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_owner_space() == context->runtime->address_space);
#endif
      RtEvent precondition;
      void *result = NULL;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          context->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_semantic_field_request(FieldID fid, 
                               SemanticTag tag, AddressSpaceID source, 
                               bool can_fail, bool wait_until,RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_owner_space() == context->runtime->address_space);
#endif
      RtEvent precondition;
      void *result = NULL;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        std::pair<FieldID,SemanticTag> key(fid,tag);
        LegionMap<std::pair<FieldID,SemanticTag>,SemanticInfo>::iterator
          finder = semantic_field_info.find(key);
        if (finder != semantic_field_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_field_info[key] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticFieldRequestArgs args(this, fid, tag, source);
          context->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_field_info(source, fid, tag, result, size, 
                                 is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_semantic_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      FieldSpaceNode *node = forest->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_field_semantic_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldID fid;
      derez.deserialize(fid);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      FieldSpaceNode *node = forest->get_node(handle);
      node->process_semantic_field_request(fid, tag, source, 
                                           can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_semantic_info(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      forest->attach_semantic_information(handle, tag, source, buffer, size, 
                                          is_mutable, false/*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_field_semantic_info(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldID fid;
      derez.deserialize(fid);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      forest->attach_semantic_information(handle, fid, tag, source, buffer, 
                                    size, is_mutable, false/*local lonly*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::FindTargetsFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      targets.push_back(target);
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::create_allocator(AddressSpaceID source,
          RtUserEvent ready_event, bool sharded_owner_context, bool owner_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (sharded_owner_context)
      {
        // If we were the sharded collective context that made this
        // field space and we're still in collective allocation mode
        // then we are trivially done
        if (allocation_state == FIELD_ALLOC_COLLECTIVE)
        {
#ifdef DEBUG_LEGION
          assert(outstanding_allocators == 0);
#endif
          outstanding_allocators = 1;
          return RtEvent::NO_RT_EVENT;
        }
        // Otherwise if we're not the owner shard then we're also done since
        // the owner shard is the only one doing the allocation
        if (!owner_shard)
          return RtEvent::NO_RT_EVENT;
      }
      if (is_owner())
      {
        switch (allocation_state)
        {
          case FIELD_ALLOC_INVALID:
            {
#ifdef DEBUG_LEGION
              assert(outstanding_allocators == 0);
              assert(remote_field_infos.size() == 1);
#endif
              const AddressSpaceID remote_owner = *(remote_field_infos.begin());
              remote_field_infos.clear();
#ifdef DEBUG_LEGION
              assert(remote_owner != local_space);
              // Should never get the ships in the night case either
              assert(remote_owner != source);
#endif
              if (!ready_event.exists())
                ready_event = Runtime::create_rt_user_event();
              outstanding_invalidations++;
              // Add a reference that will be remove when the flush returns
              add_base_gc_ref(FIELD_ALLOCATOR_REF);
              // Send the invalidation and make ourselves the new 
              // pending exclusive allocator value
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(handle);
                rez.serialize(ready_event);
                rez.serialize<bool>(true); // flush allocation
                rez.serialize<bool>(false); // need merge
              }
              runtime->send_field_space_allocator_invalidation(remote_owner, 
                                                               rez); 
              outstanding_allocators = 1;
              pending_field_allocation = ready_event;
              allocation_state = FIELD_ALLOC_PENDING;
              break;
            }
          case FIELD_ALLOC_COLLECTIVE:
            {
              // This is the case when we're still in collective mode
              // and we need to switch to exclusive mode on just one node
              // because someone else asked for an allocator
              if (outstanding_allocators > 0)
              {
#ifdef DEBUG_LEGION
                assert(!remote_field_infos.empty());
                assert(outstanding_invalidations == 0);
#endif
                std::set<RtEvent> preconditions;
                for (std::set<AddressSpaceID>::const_iterator it = 
                      remote_field_infos.begin(); it != 
                      remote_field_infos.end(); it++)
                {
                  const RtUserEvent done = Runtime::create_rt_user_event();
                  outstanding_invalidations++;
                  Serializer rez;
                  {
                    RezCheck z(rez);
                    rez.serialize(handle);
                    rez.serialize(done);
                    rez.serialize<bool>(true); // flush allocation
                    rez.serialize<bool>(true); // need merge
                  }
                  runtime->send_field_space_allocator_invalidation(*it, rez);
                  preconditions.insert(done);
                }
                remote_field_infos.clear();
                pending_field_allocation = Runtime::merge_events(preconditions);
                allocation_state = FIELD_ALLOC_PENDING;
                break;
              }
              // otherwise we fall through to the identical read-only case
            }
          case FIELD_ALLOC_READ_ONLY:
            {
#ifdef DEBUG_LEGION
              assert(outstanding_allocators == 0);
#endif
              // Send any invalidations to anyone not the source
              bool full_update = true;
              RtEvent invalidations_done;
              if (!remote_field_infos.empty())
              {
#ifdef DEBUG_LEGION
                assert(outstanding_invalidations == 0);
#endif
                std::set<RtEvent> preconditions;
                for (std::set<AddressSpaceID>::const_iterator it = 
                      remote_field_infos.begin(); it != 
                      remote_field_infos.end(); it++)
                {
                  if ((*it) == source)
                  {
                    full_update = false;
                    continue;
                  }
                  const RtUserEvent done = Runtime::create_rt_user_event();
                  outstanding_invalidations++;
                  // Add a reference that will be remove when the flush returns
                  add_base_gc_ref(FIELD_ALLOCATOR_REF);
                  Serializer rez;
                  {
                    RezCheck z(rez);
                    rez.serialize(handle);
                    rez.serialize(done);
                    rez.serialize<bool>(false); // flush allocation
                    rez.serialize<bool>(false); // need merge
                  }
                  runtime->send_field_space_allocator_invalidation(*it, rez);
                  preconditions.insert(done);
                }
                remote_field_infos.clear();
                if (!preconditions.empty())
                  invalidations_done = Runtime::merge_events(preconditions);
              }
              if (source != local_space)
              {
#ifdef DEBUG_LEGION
                assert(ready_event.exists());
#endif
                // Send the response back to the source and mark that 
                // we are now invalid
                Serializer rez;
                {
                  RezCheck z(rez);
                  rez.serialize(handle);
                  rez.serialize(invalidations_done);
                  if (full_update)
                  {
                    rez.serialize(field_infos.size());
                    for (std::map<FieldID,FieldInfo>::iterator it = 
                          field_infos.begin(); it != 
                          field_infos.end(); /*nothing*/)
                    {
                      rez.serialize(it->first);
                      it->second.serialize(rez);
                      if (!it->second.local)
                      {
                        std::map<FieldID,FieldInfo>::iterator to_delete = it++;
                        field_infos.erase(to_delete);
                      }
                      else
                        it++; // skip deleting local fields
                    }
                  }
                  if (full_update || 
                      (allocation_state != FIELD_ALLOC_COLLECTIVE)) 
                  {
                    rez.serialize(unallocated_indexes);
                    rez.serialize<size_t>(available_indexes.size());
                    for (std::list<std::pair<unsigned,RtEvent> >::const_iterator
                          it = available_indexes.begin(); it !=
                          available_indexes.end(); it++)
                    {
                      rez.serialize(it->first);
                      rez.serialize(it->second);
                    }
                  }
                  unallocated_indexes.clear();
                  available_indexes.clear();
                  rez.serialize(ready_event);
                }
                // Add a reference to this node to keep it alive until we 
                // get the corresponding free operation from the remote node
                add_base_gc_ref(FIELD_ALLOCATOR_REF);
                runtime->send_field_space_allocator_response(source, rez); 
                remote_field_infos.insert(source);
                allocation_state = FIELD_ALLOC_INVALID; 
              }
              else
              {
                // We are now the exclusive allocation owner
                if (outstanding_invalidations > 0)
                {
                  pending_field_allocation = invalidations_done;
                  allocation_state = FIELD_ALLOC_PENDING;
                }
                else // we're ready now
                  allocation_state = FIELD_ALLOC_EXCLUSIVE;
                outstanding_allocators = 1;
                if (ready_event.exists())
                  Runtime::trigger_event(ready_event, invalidations_done);
                return invalidations_done;
              }
              break;
            }
          case FIELD_ALLOC_PENDING:
            {
              outstanding_allocators++;
              if (ready_event.exists())
                Runtime::trigger_event(ready_event, pending_field_allocation);
              return pending_field_allocation;
            }
          case FIELD_ALLOC_EXCLUSIVE:
            {
              outstanding_allocators++;
              if (ready_event.exists())
                Runtime::trigger_event(ready_event, pending_field_allocation);
              break;
            } 
          default:
            assert(false);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!ready_event.exists());
        assert(source == local_space);
#endif
        // Order remote allocation requests to prevent ships-in-the-night
        while (pending_field_allocation.exists())
        {
          const RtEvent wait_on = pending_field_allocation;
          if (!wait_on.has_triggered())
          {
            n_lock.release();
            wait_on.wait();
            n_lock.reacquire();
          }
          else
            break;
        }
        // See if we already have allocation privileges
        if (allocation_state != FIELD_ALLOC_EXCLUSIVE)
        {
          ready_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(ready_event);
          }
          runtime->send_field_space_allocator_request(owner_space, rez);
          pending_field_allocation = ready_event;
        }
        else // Have privileges, increment our allocator count
          outstanding_allocators++;
      }
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::destroy_allocator(AddressSpaceID source,
                                   bool sharded_owner_context, bool owner_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert((allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
             (allocation_state == FIELD_ALLOC_COLLECTIVE) ||
             (allocation_state == FIELD_ALLOC_INVALID));
#endif
      if (sharded_owner_context)
      {
        // If we were the sharded collective context that made this
        // field space and we're still in collective allocation mode
        // then we are trivially done
        if (allocation_state == FIELD_ALLOC_COLLECTIVE)
        {
#ifdef DEBUG_LEGION
          assert(outstanding_allocators == 1);
#endif
          outstanding_allocators = 0;
          return RtEvent::NO_RT_EVENT;
        }
        // Otherwise if we're not the owner shard then we're also done since
        // the owner shard is the only one doing the allocation
        if (!owner_shard)
          return RtEvent::NO_RT_EVENT;
      }
      if (allocation_state == FIELD_ALLOC_INVALID)
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        return RtEvent::NO_RT_EVENT;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(outstanding_allocators > 0);
#endif
        if (--outstanding_allocators == 0)
        {
          // Now we go back to read-only mode
          allocation_state = FIELD_ALLOC_READ_ONLY;
          if (!is_owner())
          {
            const RtUserEvent done_event = Runtime::create_rt_user_event();
            // Send the allocation data back to the owner node
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize<bool>(true); // return allocation
              rez.serialize(field_infos.size());
              for (std::map<FieldID,FieldInfo>::const_iterator it = 
                    field_infos.begin(); it != field_infos.end(); it++)
              {
                rez.serialize(it->first);
                it->second.serialize(rez);
              }
              rez.serialize(unallocated_indexes);
              unallocated_indexes.clear();
              rez.serialize<size_t>(available_indexes.size());
              while (!available_indexes.empty())
              {
                std::pair<unsigned,RtEvent> &next = available_indexes.front();
                rez.serialize(next.first);
                rez.serialize(next.second);
                available_indexes.pop_front();
              }
              rez.serialize(done_event);
            }
            runtime->send_field_space_allocator_free(owner_space, rez);
            return done_event;
          }
        }
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::initialize_fields(const std::vector<size_t> &sizes,
                  const std::vector<FieldID> &fids, CustomSerdezID serdez_id,
                  Provenance *prov, bool collective)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fids.empty());
#endif
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        FieldID fid = fids[idx];
        if (field_infos.find(fid) != field_infos.end())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
            "Illegal duplicate field ID %d used by the "
            "application in field space %d", fid, handle.id)
        // Find an index in which to allocate this field  
        RtEvent dummy_event;
        int result = allocate_index(dummy_event, true/*initializing*/);
        if (result < 0)
          REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
            "Exceeded maximum number of allocated fields for "
            "field space %x. Change LEGION_MAX_FIELDS from %d "
            "and related macros at the top of legion_config.h "
            "and recompile.", handle.id, LEGION_MAX_FIELDS)
#ifdef DEBUG_LEGION
        assert(!dummy_event.exists());
#endif
        const unsigned index = result;
        field_infos[fid] = FieldInfo(sizes[idx], index, serdez_id, 
                                     prov, false/*local*/, collective);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::initialize_fields(ApEvent sizes_ready,
                    const std::vector<FieldID> &fids, CustomSerdezID serdez_id,
                    Provenance *prov, bool collective)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        FieldID fid = fids[idx];
        if (field_infos.find(fid) != field_infos.end())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
            "Illegal duplicate field ID %d used by the "
                          "application in field space %d", fid, handle.id)
        // Find an index in which to allocate this field  
        RtEvent dummy_event;
        int result = allocate_index(dummy_event, true/*initializing*/);
        if (result < 0)
          REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
            "Exceeded maximum number of allocated fields for "
                          "field space %x. Change LEGION_MAX_FIELDS from %d "
                          "and related macros at the top of legion_config.h "
                          "and recompile.", handle.id, LEGION_MAX_FIELDS)
#ifdef DEBUG_LEGION
        assert(!dummy_event.exists());
#endif
        const unsigned index = result;
        field_infos[fid] = FieldInfo(sizes_ready, index, serdez_id, 
                                     prov, false/*local*/, collective);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_field(FieldID fid, size_t size, 
                                           CustomSerdezID serdez_id,
                                           Provenance *prov,
                                           bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(ApEvent::NO_AP_EVENT);
          if (prov != NULL)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(1); // only allocating one field
          rez.serialize(fid);
          rez.serialize(size);
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::map<FieldID,FieldInfo>::iterator finder = field_infos.find(fid);
      if (finder != field_infos.end())
      {
        // Handle the case of deduplicating fields that were allocated
        // in a collective mode but are now merged together
        if (finder->second.collective)
          return RtEvent::NO_RT_EVENT;
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
          "Illegal duplicate field ID %d used by the "
                        "application in field space %d", fid, handle.id)
      }
      // Find an index in which to allocate this field  
      RtEvent ready_event;
      int result = allocate_index(ready_event);
      if (result < 0)
        REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
                        "Exceeded maximum number of allocated fields for "
                        "field space %x. Change LEGION_MAX_FIELDS from %d and"
                        " related macros at the top of legion_config.h and "
                        "recompile.", handle.id, LEGION_MAX_FIELDS)
      const unsigned index = result;
      field_infos[fid] = FieldInfo(size, index, serdez_id, prov,
          false/*local*/, (allocation_state == FIELD_ALLOC_COLLECTIVE));
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_field(FieldID fid, ApEvent size_ready, 
                                           CustomSerdezID serdez_id,
                                           Provenance *prov,
                                           bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(size_ready); // size ready
          if (prov != NULL)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(1); // only allocating one field
          rez.serialize(fid);
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::map<FieldID,FieldInfo>::iterator finder = field_infos.find(fid);
      if (finder != field_infos.end())
      {
        // Handle the case of deduplicating fields that were allocated
        // in a collective mode but are now merged together
        if (finder->second.collective)
          return RtEvent::NO_RT_EVENT;
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
          "Illegal duplicate field ID %d used by the "
                        "application in field space %d", fid, handle.id)
      }
      // Find an index in which to allocate this field  
      RtEvent ready_event;
      int result = allocate_index(ready_event);
      if (result < 0)
        REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
                        "Exceeded maximum number of allocated fields for "
                        "field space %x. Change LEGION_MAX_FIELDS from %d and"
                        " related macros at the top of legion_config.h and "
                        "recompile.", handle.id, LEGION_MAX_FIELDS)
      const unsigned index = result;
      field_infos[fid] = FieldInfo(size_ready, index, serdez_id, prov,
          false/*local*/, (allocation_state == FIELD_ALLOC_COLLECTIVE));
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_fields(const std::vector<size_t> &sizes,
                                            const std::vector<FieldID> &fids,
                                            CustomSerdezID serdez_id,
                                            Provenance *prov,
                                            bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fids.empty());
      assert(sizes.size() == fids.size());
#endif
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(ApEvent::NO_AP_EVENT);
          if (prov != NULL)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(fids.size());
          for (unsigned idx = 0; idx < fids.size(); idx++)
          {
            rez.serialize(fids[idx]);
            rez.serialize(sizes[idx]);
          }
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::set<RtEvent> allocated_events;
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        const FieldID fid = fids[idx];
        std::map<FieldID,FieldInfo>::iterator finder = field_infos.find(fid);
        if (field_infos.find(fid) != field_infos.end())
        {
          // Handle the case of deduplicating fields that were allocated
          // in a collective mode but are now merged together
          if (finder->second.collective)
            continue;
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
            "Illegal duplicate field ID %d used by the "
                          "application in field space %d", fid, handle.id)
        }
        // Find an index in which to allocate this field  
        RtEvent ready_event;
        int result = allocate_index(ready_event);
        if (result < 0)
          REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
            "Exceeded maximum number of allocated fields for "
                          "field space %x. Change LEGION_MAX_FIELDS from %d "
                          "and related macros at the top of legion_config.h "
                          "and recompile.", handle.id, LEGION_MAX_FIELDS)
        if (ready_event.exists())
          allocated_events.insert(ready_event);
        const unsigned index = result;
        field_infos[fid] = FieldInfo(sizes[idx], index, serdez_id, prov, 
            false/*local*/, (allocation_state == FIELD_ALLOC_COLLECTIVE));
      }
      if (!allocated_events.empty())
        return Runtime::merge_events(allocated_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_fields(ApEvent sizes_ready,
                                            const std::vector<FieldID> &fids,
                                            CustomSerdezID serdez_id,
                                            Provenance *prov,
                                            bool sharded_non_owner)
    //--------------------------------------------------------------------------
    { 
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(sizes_ready);
          if (prov != NULL)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(fids.size());
          for (unsigned idx = 0; idx < fids.size(); idx++)
            rez.serialize(fids[idx]);
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::set<RtEvent> allocated_events;
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        const FieldID fid = fids[idx];
        std::map<FieldID,FieldInfo>::iterator finder = field_infos.find(fid);
        if (finder != field_infos.end())
        {
          // Handle the case of deduplicating fields that were allocated
          // in a collective mode but are now merged together
          if (finder->second.collective)
            continue;
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
            "Illegal duplicate field ID %d used by the "
                          "application in field space %d", fid, handle.id)
        }
        // Find an index in which to allocate this field  
        RtEvent ready_event;
        int result = allocate_index(ready_event);
        if (result < 0)
          REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
            "Exceeded maximum number of allocated fields for "
                          "field space %x. Change LEGION_MAX_FIELDS from %d "
                          "and related macros at the top of legion_config.h "
                          "and recompile.", handle.id, LEGION_MAX_FIELDS)
        if (ready_event.exists())
          allocated_events.insert(ready_event);
        const unsigned index = result;
        field_infos[fid] = FieldInfo(sizes_ready, index, serdez_id, prov,
            false/*local*/, (allocation_state == FIELD_ALLOC_COLLECTIVE));
      }
      if (!allocated_events.empty())
        return Runtime::merge_events(allocated_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::update_field_size(FieldID fid, size_t field_size,
                        std::set<RtEvent> &update_events, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      std::map<FieldID,FieldInfo>::iterator finder = 
        field_infos.find(fid);
      if (finder != field_infos.end())
      {
#ifdef DEBUG_LEGION
        assert(finder->second.field_size == 0);
        assert(finder->second.size_ready.exists());
#endif
        finder->second.field_size = field_size;
        finder->second.size_ready = ApEvent::NO_AP_EVENT;
      }
      // Now figure out where the updates need to go 
      if (is_owner())
      {
        // If we're not the exclusive allocator then broadcast
        // this out to all the other nodes so that they see updates
        if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
            (allocation_state != FIELD_ALLOC_COLLECTIVE))
        {
          // Send messages to all the read-only field infos
          for (std::set<AddressSpaceID>::const_iterator it = 
                remote_field_infos.begin(); it != 
                remote_field_infos.end(); it++)
          {
            if ((*it) == source)
              continue;
            const RtUserEvent done_event = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(done_event);
              rez.serialize(fid);
              rez.serialize(field_size);
            }
            pack_global_ref();
            context->runtime->send_field_size_update(*it, rez);
            update_events.insert(done_event);
          }
        }
      }
      else
      {
        // If the source is not the owner and we're not in a collective 
        // mode then we have to send the message to the owner
        if ((source != owner_space) && 
            (allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
            (allocation_state != FIELD_ALLOC_COLLECTIVE))
        {
          const RtUserEvent done_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(done_event);
            rez.serialize(fid);
            rez.serialize(field_size);
          }
          pack_global_ref();
          context->runtime->send_field_size_update(owner_space, rez);
          update_events.insert(done_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field(FieldID fid, AddressSpaceID source,
                                    std::set<RtEvent> &applied,
                                    bool sharded_non_owner)   
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock); 
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(1);
          rez.serialize(fid);
          rez.serialize(done_event);
        }
        context->runtime->send_field_free(owner_space, rez);
        applied.insert(done_event);
        return;
      }
      std::map<FieldID,FieldInfo>::iterator finder = field_infos.find(fid);
#ifdef DEBUG_LEGION
      assert(finder != field_infos.end());
#endif
      // Remove it from the field map
      field_infos.erase(finder);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_fields(const std::vector<FieldID> &to_free,
                              AddressSpaceID source, std::set<RtEvent> &applied,
                              bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock); 
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(to_free.size());
          for (unsigned idx = 0; idx < to_free.size(); idx++)
            rez.serialize(to_free[idx]);
          rez.serialize(done_event);
        }
        context->runtime->send_field_free(owner_space, rez);
        applied.insert(done_event);
        return;
      }
      for (std::vector<FieldID>::const_iterator it = to_free.begin();
            it != to_free.end(); it++)
      {
        std::map<FieldID,FieldInfo>::iterator finder = field_infos.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != field_infos.end());
#endif
        // Remove it from the fields map
        field_infos.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field_indexes(const std::vector<FieldID> &to_free,
                                    RtEvent freed_event, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(to_free.size());
          for (unsigned idx = 0; idx < to_free.size(); idx++)
            rez.serialize(to_free[idx]);
          rez.serialize(freed_event);
        }
        context->runtime->send_field_free_indexes(owner_space, rez);
        return;
      }
      for (std::vector<FieldID>::const_iterator it = 
            to_free.begin(); it != to_free.end(); it++)
      {
        std::map<FieldID,FieldInfo>::iterator finder = field_infos.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != field_infos.end());
#endif
        // Skip freeing any local field indexes here
        if (!finder->second.local)
          free_index(finder->second.idx, freed_event);
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::allocate_local_fields(
                                            const std::vector<FieldID> &fids,
                                            const std::vector<size_t> &sizes,
                                            CustomSerdezID serdez_id,
                                            const std::set<unsigned> &indexes,
                                            std::vector<unsigned> &new_indexes,
                                            Provenance *prov)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(fids.size() == sizes.size());
      assert(new_indexes.empty());
#endif
      if (!is_owner())
      {
        // If we're not the owner, send a message to the owner
        // to do the local field allocation
        RtUserEvent allocated_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          if (prov != NULL)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(fids.size());
          for (unsigned idx = 0; idx < fids.size(); idx++)
          {
            rez.serialize(fids[idx]);
            rez.serialize(sizes[idx]);
          }
          rez.serialize<size_t>(indexes.size());
          for (std::set<unsigned>::const_iterator it = indexes.begin();
                it != indexes.end(); it++)
            rez.serialize(*it);
          rez.serialize(&new_indexes);
        }
        context->runtime->send_local_field_alloc_request(owner_space, rez);
        // Wait for the result
        allocated_event.wait();
        if (new_indexes.empty())
          return false;
        // When we wake up then fill in the field information
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(!fids.empty());
        assert(new_indexes.size() == fids.size());
#endif
        for (unsigned idx = 0; idx < fids.size(); idx++)
        {
          FieldID fid = fids[idx];
          field_infos[fid] = 
            FieldInfo(sizes[idx],new_indexes[idx],serdez_id,prov,true/*local*/);
        }
      }
      else
      {
        // We're the owner so do the field allocation
        AutoLock n_lock(node_lock);
        if (!allocate_local_indexes(serdez_id, sizes, indexes, new_indexes))
          return false;
#ifdef DEBUG_LEGION
        assert(!fids.empty());
#endif
        for (unsigned idx = 0; idx < fids.size(); idx++)
        {
          FieldID fid = fids[idx];
          if (field_infos.find(fid) != field_infos.end())
            REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
              "Illegal duplicate field ID %d used by the "
                            "application in field space %d", fid, handle.id)
          field_infos[fid] = 
            FieldInfo(sizes[idx],new_indexes[idx],serdez_id,prov,true/*local*/);
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_local_fields(const std::vector<FieldID> &to_free,
                                           const std::vector<unsigned> &indexes,
                                           const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(to_free.size() == indexes.size());
#endif
      if (mapping != NULL)
      {
        if (mapping->contains(owner_space))
        {
          if (local_space != owner_space)
            return;
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == local_space)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize<size_t>(to_free.size());
              for (unsigned idx = 0; idx < to_free.size(); idx++)
              {
                rez.serialize(to_free[idx]);
                rez.serialize(indexes[idx]);
              }
            }
            context->runtime->send_local_field_free(owner_space, rez);
          }
          return;
        }
      }
      else
      {
        if (!is_owner())
        {
          // Send a message to the owner to do the free of the fields
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<size_t>(to_free.size());
            for (unsigned idx = 0; idx < to_free.size(); idx++)
            {
              rez.serialize(to_free[idx]);
              rez.serialize(indexes[idx]);
            }
          }
          context->runtime->send_local_field_free(owner_space, rez);
          return;
        }
      }
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // Do the local free
      AutoLock n_lock(node_lock); 
      for (unsigned idx = 0; idx < to_free.size(); idx++)
      {
        std::map<FieldID,FieldInfo>::iterator finder = 
          field_infos.find(to_free[idx]);
#ifdef DEBUG_LEGION
        assert(finder != field_infos.end());
#endif
        field_infos.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::update_local_fields(const std::vector<FieldID> &fids,
                                  const std::vector<size_t> &sizes,
                                  const std::vector<CustomSerdezID> &serdez_ids,
                                  const std::vector<unsigned> &indexes,
                                  Provenance *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(fids.size() == sizes.size());
      assert(fids.size() == serdez_ids.size());
      assert(fids.size() == indexes.size());
#endif
      AutoLock n_lock(node_lock);
      for (unsigned idx = 0; idx < fids.size(); idx++)
        field_infos[fids[idx]] = FieldInfo(sizes[idx], indexes[idx], 
            serdez_ids[idx], provenance, true/*local*/);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::remove_local_fields(
                                          const std::vector<FieldID> &to_remove)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      for (unsigned idx = 0; idx < to_remove.size(); idx++)
      {
        std::map<FieldID,FieldInfo>::iterator finder = 
          field_infos.find(to_remove[idx]);
        if (finder != field_infos.end())
          field_infos.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        // Check to see if we have a valid copy of the field infos
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID,FieldInfo>::const_iterator finder = 
            field_infos.find(fid);
          if (finder == field_infos.end())
            return false;
          else
            return true;
        }
      }
      std::map<FieldID,FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      std::map<FieldID,FieldInfo>::const_iterator finder = 
        local_infos.find(fid);
      if (finder == local_infos.end())
        return false;
      else
        return true;
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::get_field_size(FieldID fid)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_for;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID,FieldInfo>::const_iterator finder = 
            field_infos.find(fid);
#ifdef DEBUG_LEGION
          assert(finder != field_infos.end());
#endif
          // See if this field has been allocated or not yet
          if (!finder->second.size_ready.exists())
            return finder->second.field_size;
          wait_for = Runtime::protect_event(finder->second.size_ready);
        }
      }
      if (!wait_for.exists())
      {
        std::map<FieldID,FieldInfo> local_infos;
        const RtEvent ready = 
          request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        std::map<FieldID,FieldInfo>::const_iterator finder = 
          local_infos.find(fid);
#ifdef DEBUG_LEGION
        assert(finder != local_infos.end());
#endif
        // See if this field has been allocated or not yet
        if (!finder->second.size_ready.exists())
          return finder->second.field_size;
        wait_for = Runtime::protect_event(finder->second.size_ready);
      }
      if (!wait_for.has_triggered())
        wait_for.wait();
      return get_field_size(fid);
    }

    //--------------------------------------------------------------------------
    CustomSerdezID FieldSpaceNode::get_field_serdez(FieldID fid)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_for;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID,FieldInfo>::const_iterator finder = 
            field_infos.find(fid);
#ifdef DEBUG_LEGION
          assert(finder != field_infos.end());
#endif
          // See if this field has been allocated or not yet
          if (!finder->second.size_ready.exists())
            return finder->second.serdez_id;
          wait_for = Runtime::protect_event(finder->second.size_ready);
        }
      }
      if (!wait_for.exists())
      {
        std::map<FieldID,FieldInfo> local_infos;
        const RtEvent ready = 
          request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        std::map<FieldID,FieldInfo>::const_iterator finder = 
          local_infos.find(fid);
#ifdef DEBUG_LEGION
        assert(finder != local_infos.end());
#endif
        // See if this field has been allocated or not yet
        if (!finder->second.size_ready.exists())
          return finder->second.serdez_id;
        wait_for = Runtime::protect_event(finder->second.size_ready);
      }
      if (!wait_for.has_triggered())
        wait_for.wait();
      return get_field_serdez(fid);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_all_fields(std::vector<FieldID> &to_set)
    //--------------------------------------------------------------------------
    {
      to_set.clear();
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          to_set.reserve(field_infos.size());
          for (std::map<FieldID,FieldInfo>::const_iterator it = 
                field_infos.begin(); it != field_infos.end(); it++)
            to_set.push_back(it->first);
          return;
        }
      }
      std::map<FieldID,FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      to_set.reserve(local_infos.size());
      for (std::map<FieldID,FieldInfo>::const_iterator it = 
            local_infos.begin(); it != local_infos.end(); it++)
        to_set.push_back(it->first);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(const FieldMask &mask, TaskContext *ctx,
                                       std::set<FieldID> &to_set) const
    //--------------------------------------------------------------------------
    {
      std::set<unsigned> local_indexes;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (std::map<FieldID,FieldInfo>::const_iterator it = 
                field_infos.begin(); it != field_infos.end(); it++)
          {
            if (mask.is_set(it->second.idx))
            {
              if (it->second.local)
                local_indexes.insert(it->second.idx);
              else
                to_set.insert(it->first);
            }
          }
          if (local_indexes.empty())
            return;
        }
      }
      if (local_indexes.empty())
      {
        std::map<FieldID,FieldInfo> local_infos;
        const RtEvent ready = 
          request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (std::map<FieldID,FieldInfo>::const_iterator it = 
              local_infos.begin(); it != local_infos.end(); it++)
        {
          if (mask.is_set(it->second.idx))
          {
            if (it->second.local)
              local_indexes.insert(it->second.idx);
            else
              to_set.insert(it->first);
          }
        }
      }
      if (!local_indexes.empty())
        ctx->get_local_field_set(handle, local_indexes, to_set);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(const FieldMask &mask, TaskContext *ctx,
                                       std::vector<FieldID> &to_set) const
    //--------------------------------------------------------------------------
    {
      std::set<unsigned> local_indexes;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (std::map<FieldID,FieldInfo>::const_iterator it = 
                field_infos.begin(); it != field_infos.end(); it++)
          {
            if (mask.is_set(it->second.idx))
            {
              if (it->second.local)
                local_indexes.insert(it->second.idx);
              else
                to_set.push_back(it->first);
            }
          }
          if (local_indexes.empty())
            return;
        }
      }
      if (local_indexes.empty())
      {
        std::map<FieldID,FieldInfo> local_infos;
        const RtEvent ready = 
          request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (std::map<FieldID,FieldInfo>::const_iterator it = 
              local_infos.begin(); it != local_infos.end(); it++)
        {
          if (mask.is_set(it->second.idx))
          {
            if (it->second.local)
              local_indexes.insert(it->second.idx);
            else
              to_set.push_back(it->first);
          }
        }
      }
      if (!local_indexes.empty())
        ctx->get_local_field_set(handle, local_indexes, to_set);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(const FieldMask &mask, 
                                       const std::set<FieldID> &basis,
                                       std::set<FieldID> &to_set) const
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          // Only iterate over the basis fields here
          for (std::set<FieldID>::const_iterator it = basis.begin();
                it != basis.end(); it++)
          {
            std::map<FieldID,FieldInfo>::const_iterator finder = 
              field_infos.find(*it);
#ifdef DEBUG_LEGION
            assert(finder != field_infos.end());
#endif
            if (mask.is_set(finder->second.idx))
              to_set.insert(finder->first);
          }
          return;
        }
      }
      std::map<FieldID,FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      // Only iterate over the basis fields here
      for (std::set<FieldID>::const_iterator it = basis.begin();
            it != basis.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = 
          local_infos.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != local_infos.end());
#endif
        if (mask.is_set(finder->second.idx))
          to_set.insert(finder->first);
      }
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(
                              const std::set<FieldID> &privilege_fields) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (std::set<FieldID>::const_iterator it = privilege_fields.begin();
                it != privilege_fields.end(); it++)
          {
            std::map<FieldID,FieldInfo>::const_iterator finder = 
              field_infos.find(*it);
#ifdef DEBUG_LEGION
            assert(finder != field_infos.end());
#endif
            result.set_bit(finder->second.idx);
          }
          return result;
        }
      }
      std::map<FieldID,FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      for (std::set<FieldID>::const_iterator it = privilege_fields.begin();
            it != privilege_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = 
          local_infos.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != local_infos.end());
#endif
        result.set_bit(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned FieldSpaceNode::get_field_index(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID,FieldInfo>::const_iterator finder = 
            field_infos.find(fid);
#ifdef DEBUG_LEGION
          assert(finder != field_infos.end());
#endif
          return finder->second.idx;
        }
      }
      std::map<FieldID,FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      std::map<FieldID,FieldInfo>::const_iterator finder = 
          local_infos.find(fid);
#ifdef DEBUG_LEGION
      assert(finder != local_infos.end());
#endif
      return finder->second.idx;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_indexes(const std::vector<FieldID> &needed,
                                          std::vector<unsigned> &indexes) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(needed.size() == indexes.size());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (unsigned idx = 0; idx < needed.size(); idx++)
          {
            std::map<FieldID,FieldInfo>::const_iterator finder = 
              field_infos.find(needed[idx]);
#ifdef DEBUG_LEGION
            assert(finder != field_infos.end());
#endif
            indexes[idx] = finder->second.idx;
          }
          return; 
        }
      }
      std::map<FieldID,FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      for (unsigned idx = 0; idx < needed.size(); idx++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = 
          local_infos.find(needed[idx]);
#ifdef DEBUG_LEGION
        assert(finder != local_infos.end());
#endif
        indexes[idx] = finder->second.idx;
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::compute_field_layout(
                                      const std::vector<FieldID> &create_fields,
                                      std::vector<size_t> &field_sizes,
                                      std::vector<unsigned> &mask_index_map,
                                      std::vector<CustomSerdezID> &serdez,
                                      FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_sizes.size() == create_fields.size());
      assert(mask_index_map.size() == create_fields.size());
      assert(serdez.size() == create_fields.size());
#endif
      bool invalid = false;
      std::set<ApEvent> defer_events;
      std::map<unsigned/*mask index*/,unsigned/*layout index*/> index_map;
      {
        // Need to hold the lock when accessing field infos
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (unsigned idx = 0; idx < create_fields.size(); idx++)
          {
            const FieldID fid = create_fields[idx];
            std::map<FieldID,FieldInfo>::const_iterator finder = 
              field_infos.find(fid);
            // Catch unknown fields here for now
            if (finder == field_infos.end())
              REPORT_LEGION_FATAL(LEGION_FATAL_UNKNOWN_FIELD_ID,
                "unknown field ID %d requested during instance creation", fid)
            if (finder->second.size_ready.exists())
              defer_events.insert(finder->second.size_ready);
            else if (defer_events.empty())
            {
              field_sizes[idx] = finder->second.field_size; 
              index_map[finder->second.idx] = idx;
              serdez[idx] = finder->second.serdez_id;
              mask.set_bit(finder->second.idx);
            }
          }
        }
        else
          invalid = true;
      }
      if (invalid)
      {
        std::map<FieldID,FieldInfo> local_infos;
        const RtEvent ready = 
          request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();       
        for (unsigned idx = 0; idx < create_fields.size(); idx++)
        {
          const FieldID fid = create_fields[idx];
          std::map<FieldID,FieldInfo>::const_iterator finder = 
            local_infos.find(fid);
          // Catch unknown fields here for now
          if (finder == local_infos.end())
            REPORT_LEGION_FATAL(LEGION_FATAL_UNKNOWN_FIELD_ID,
              "unknown field ID %d requested during instance creation", fid)
          if (finder->second.size_ready.exists())
            defer_events.insert(finder->second.size_ready);
          else if (defer_events.empty())
          {
            field_sizes[idx] = finder->second.field_size; 
            index_map[finder->second.idx] = idx;
            serdez[idx] = finder->second.serdez_id;
            mask.set_bit(finder->second.idx);
          }
        }
      }
      if (!defer_events.empty())
      {
        const RtEvent wait_on = Runtime::protect_merge_events(defer_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
        compute_field_layout(create_fields, field_sizes, 
                             mask_index_map, serdez, mask);
        return;
      }
      // Now we can linearize the index map without holding the lock
      unsigned idx = 0;
      for (std::map<unsigned,unsigned>::const_iterator it = 
            index_map.begin(); it != index_map.end(); it++, idx++)
        mask_index_map[idx] = it->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_alloc_request(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      CustomSerdezID serdez_id;
      derez.deserialize(serdez_id);
      ApEvent sizes_ready;
      derez.deserialize(sizes_ready);
      AutoProvenance provenance(Provenance::deserialize(derez));
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fids(num_fields);
      RtEvent ready;
      if (!sizes_ready.exists())
      {
        std::vector<size_t> sizes(num_fields);
        for (unsigned idx = 0; idx < num_fields; idx++)
        {
          derez.deserialize(fids[idx]);
          derez.deserialize(sizes[idx]);
        }
        FieldSpaceNode *node = forest->get_node(handle);
        ready = node->allocate_fields(sizes, fids, serdez_id, provenance);
      }
      else
      {
        for (unsigned idx = 0; idx < num_fields; idx++)
          derez.deserialize(fids[idx]);
        FieldSpaceNode *node = forest->get_node(handle);
        ready = node->allocate_fields(sizes_ready, fids, serdez_id, provenance);
      }
      Runtime::trigger_event(done, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_field_free(RegionTreeForest *forest,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
        derez.deserialize(fields[idx]);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      FieldSpaceNode *node = forest->get_node(handle);
      std::set<RtEvent> applied;
      node->free_fields(fields, source, applied);
      if (done_event.exists())
      {
        if (!applied.empty())
          Runtime::trigger_event(done_event, Runtime::merge_events(applied));
        else
          Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_field_free_indexes(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
        derez.deserialize(fields[idx]);
      RtEvent freed_event;
      derez.deserialize(freed_event);
      FieldSpaceNode *node = forest->get_node(handle);
      node->free_field_indexes(fields, freed_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_layout_invalidation(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      unsigned index;
      derez.deserialize(index);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      FieldSpaceNode *node = forest->get_node(handle);
      std::set<RtEvent> applied;
      node->invalidate_layouts(index, applied, source);
      node->unpack_global_ref();
      if (!applied.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_local_alloc_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      CustomSerdezID serdez_id;
      derez.deserialize(serdez_id);
      AutoProvenance provenance(Provenance::deserialize(derez));
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      std::vector<size_t> sizes(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(fields[idx]);
        derez.deserialize(sizes[idx]);
      }
      size_t num_indexes;
      derez.deserialize(num_indexes);
      std::set<unsigned> current_indexes;
      for (unsigned idx = 0; idx < num_indexes; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        current_indexes.insert(index);
      }
      std::vector<unsigned> *destination;
      derez.deserialize(destination);

      FieldSpaceNode *node = forest->get_node(handle);
      std::vector<unsigned> new_indexes;
      if (node->allocate_local_fields(fields, sizes, serdez_id,
                                      current_indexes, new_indexes, provenance))
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(destination);
          rez.serialize<size_t>(new_indexes.size());
          for (unsigned idx = 0; idx < new_indexes.size(); idx++)
            rez.serialize(new_indexes[idx]);
          rez.serialize(done_event);
        }
        forest->runtime->send_local_field_alloc_response(source, rez);
      }
      else // if we failed we can just trigger the event
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_local_alloc_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::vector<unsigned> *destination;
      derez.deserialize(destination);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      destination->resize(num_indexes);
      for (unsigned idx = 0; idx < num_indexes; idx++)
        derez.deserialize((*destination)[idx]);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_local_free(RegionTreeForest *forest,
                                                      Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      std::vector<unsigned> indexes(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(fields[idx]);
        derez.deserialize(indexes[idx]);
      }

      FieldSpaceNode *node = forest->get_node(handle);
      node->free_local_fields(fields, indexes, NULL/*no collective*/); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_field_size_update(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      FieldID fid;
      derez.deserialize(fid);
      size_t field_size;
      derez.deserialize(field_size);

      FieldSpaceNode *node = forest->get_node(handle);
      std::set<RtEvent> done_events;
      node->update_field_size(fid, field_size, done_events, source);
      node->unpack_global_ref();
      if (!done_events.empty())
        Runtime::trigger_event(done, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_defer_infos_request(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRequestFieldInfoArgs *dargs = 
        (const DeferRequestFieldInfoArgs*)args;
      dargs->proxy_this->request_field_infos_copy(dargs->copy, dargs->source, 
                                                  dargs->to_trigger);
    }

    //--------------------------------------------------------------------------
    InstanceRef FieldSpaceNode::create_external_instance(
                                         const std::set<FieldID> &priv_fields,
                                         const std::vector<FieldID> &field_set,
                                         RegionNode *node, AttachOp *attach_op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node->column_source == this);
#endif
      std::vector<size_t> field_sizes(field_set.size());
      std::vector<unsigned> mask_index_map(field_set.size());
      std::vector<CustomSerdezID> serdez(field_set.size());
      FieldMask external_mask;
      compute_field_layout(field_set, field_sizes, 
                           mask_index_map, serdez, external_mask);
      FieldMask privilege_mask = (priv_fields.size() == field_set.size()) ?
        external_mask : get_field_mask(priv_fields);
      // Now make the instance, this should always succeed
      PhysicalManager *manager = attach_op->create_manager(node, field_set,
          field_sizes, mask_index_map, serdez, external_mask);
      return InstanceRef(manager, privilege_mask); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_external_create_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldSpaceNode *fs = runtime->forest->get_node(handle);
      PhysicalInstance inst;
      derez.deserialize(inst);
      ApEvent ready_event;
      derez.deserialize(ready_event);
      LgEvent unique_event;
      derez.deserialize(unique_event);
      size_t footprint;
      derez.deserialize(footprint);
      LayoutConstraintSet constraints;
      constraints.deserialize(derez);
      FieldMask file_mask;
      derez.deserialize(file_mask);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> field_set(num_fields);
      std::vector<size_t> field_sizes(num_fields);
      std::vector<unsigned> mask_index_map(num_fields);
      std::vector<CustomSerdezID> serdez(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(field_set[idx]);
        derez.deserialize(field_sizes[idx]);
        derez.deserialize(mask_index_map[idx]);
        derez.deserialize(serdez[idx]);
      }
      LogicalRegion region_handle;
      derez.deserialize(region_handle);
      RegionNode *region_node = runtime->forest->get_node(region_handle);
      size_t collective_mapping_size;
      derez.deserialize(collective_mapping_size);
      CollectiveMapping *collective_mapping = NULL;
      if (collective_mapping_size > 0)
      {
        collective_mapping =
          new CollectiveMapping(derez, collective_mapping_size);
        collective_mapping->add_reference();
      }
      std::atomic<DistributedID> *did_ptr;
      derez.deserialize(did_ptr);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      PhysicalManager *manager = fs->create_external_manager(inst, ready_event,
          footprint, constraints, field_set, field_sizes, file_mask,
          mask_index_map, unique_event, region_node, serdez,
          runtime->get_available_distributed_id(), collective_mapping);
      
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(did_ptr);
        rez.serialize(manager->did);
        rez.serialize(done_event);
      }
      runtime->send_external_create_response(source, rez);

      if ((collective_mapping != NULL) &&
          collective_mapping->remove_reference())
        delete collective_mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_external_create_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<DistributedID> *did_ptr;
      derez.deserialize(did_ptr);
      DistributedID did;
      derez.deserialize(did);
      did_ptr->store(did);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* FieldSpaceNode::create_external_manager(
            PhysicalInstance inst, ApEvent ready_event, 
            size_t instance_footprint, LayoutConstraintSet &constraints, 
            const std::vector<FieldID> &field_set,
            const std::vector<size_t> &field_sizes, 
            const FieldMask &external_mask,
            const std::vector<unsigned> &mask_index_map, LgEvent unique_event,
            RegionNode *node, const std::vector<CustomSerdezID> &serdez,
            DistributedID did, CollectiveMapping *collective_mapping)
    //--------------------------------------------------------------------------
    {
      // Pull out the pointer constraint so that we can use it separately
      // and not have it included in the layout constraints
      constraints.pointer_constraint = PointerConstraint();
      const unsigned total_dims = node->row_source->get_num_dims();
      // Get the layout
      LayoutDescription *layout = 
        find_layout_description(external_mask, total_dims, constraints);
      if (layout == NULL)
      {
        LayoutConstraints *layout_constraints = 
          context->runtime->register_layout(handle, 
                                            constraints, true/*internal*/);
        layout = create_layout_description(external_mask, total_dims,
                                           layout_constraints,
                                           mask_index_map, field_set,
                                           field_sizes, serdez);
      }
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      MemoryManager *memory = 
        context->runtime->find_memory_manager(inst.get_location());
      PhysicalManager *result = new PhysicalManager(context, did, 
                                         memory, inst, node->row_source, 
                                         NULL/*piece list*/, 
                                         0/*piece list size*/,
                                         node->column_source,
                                         node->handle.get_tree_id(),
                                         layout, 0/*redop*/, 
                                         true/*register now*/,
                                         instance_footprint, 
                                         ready_event, unique_event,
                              PhysicalManager::EXTERNAL_ATTACHED_INSTANCE_KIND,
                                         NULL/*redop*/,
                                         collective_mapping);
      // Remove the reference that was returned to us from either finding
      // or creating the layout
      if (layout->remove_reference())
        delete layout;
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::find_layout_description(
                                      const FieldMask &mask, unsigned num_dims,
                                      const LayoutConstraintSet &constraints)
    //--------------------------------------------------------------------------
    {
      std::deque<LayoutDescription*> candidates;
      {
        uint64_t hash_key = mask.get_hash_key();
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
          LAYOUT_DESCRIPTION_ALLOC>>::const_iterator finder = 
                                                      layouts.find(hash_key);
        if (finder == layouts.end())
          return NULL;
        // Get the ones with a matching mask
        for (std::list<LayoutDescription*>::const_iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if ((*it)->total_dims != num_dims)
            continue;
          if ((*it)->allocated_fields == mask)
            candidates.push_back(*it);
        }
      }
      if (candidates.empty())
        return NULL;
      // First go through the existing descriptions and see if we find
      // one that matches the existing layout
      for (std::deque<LayoutDescription*>::const_iterator it = 
            candidates.begin(); it != candidates.end(); it++)
      {
        if ((*it)->match_layout(constraints, num_dims))
        {
          (*it)->add_reference();
          return (*it);
        }
      }
      return NULL;
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::find_layout_description(
                          const FieldMask &mask, LayoutConstraints *constraints)
    //--------------------------------------------------------------------------
    {
      // This one better work
      uint64_t hash_key = mask.get_hash_key();
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
        LAYOUT_DESCRIPTION_ALLOC>>::const_iterator finder = 
                                                    layouts.find(hash_key);
#ifdef DEBUG_LEGION
      assert(finder != layouts.end());
#endif
      for (std::list<LayoutDescription*>::const_iterator it = 
            finder->second.begin(); it != finder->second.end(); it++)
      {
        if ((*it)->constraints != constraints)
          continue;
        if ((*it)->allocated_fields != mask)
          continue;
        (*it)->add_reference();
        return (*it);
      }
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::create_layout_description(
                                     const FieldMask &layout_mask,
                                     const unsigned total_dims,
                                     LayoutConstraints *constraints,
                                   const std::vector<unsigned> &mask_index_map,
                                   const std::vector<FieldID> &fids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<CustomSerdezID> &serdez)
    //--------------------------------------------------------------------------
    {
      // Make the new field description and then register it
      LayoutDescription *result = new LayoutDescription(this, layout_mask, 
        total_dims, constraints, mask_index_map, fids, field_sizes, serdez);
      result->add_reference();
      return register_layout_description(result);
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::register_layout_description(
                                                      LayoutDescription *layout)
    //--------------------------------------------------------------------------
    {
      uint64_t hash_key = layout->allocated_fields.get_hash_key();
      AutoLock n_lock(node_lock);
      LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>
        &descs = layouts[hash_key];
      if (!descs.empty())
      {
        for (LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>
              ::const_iterator it = descs.begin(); it != descs.end(); it++)
        {
          if (layout->match_layout(*it, layout->total_dims))
          {
            // Delete the layout we are trying to register
            // and return the matching one
            if (layout->remove_reference())
              delete layout;
            (*it)->add_reference();
            return (*it);
          }
        }
      }
      // Otherwise we successfully registered it
      descs.push_back(layout);
      // Add the reference here for our local data structure
      layout->add_reference();
      return layout;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Only send it if we're the owner without a collective mapping
      // or the target is not in the collective mapping and we're the
      // closest node in the collective mapping to the target
      assert((is_owner() && (collective_mapping == NULL)) ||
            ((collective_mapping != NULL) && 
             !collective_mapping->contains(target) &&
             collective_mapping->contains(local_space) && 
             (local_space == collective_mapping->find_nearest(target))));
#endif
      // See if this is in our creation set, if not, send it and all the fields
      AutoLock n_lock(node_lock);
      if (!has_remote_instance(target))
      {
        // First send the node info and then send all the fields
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(did);
          rez.serialize(initialized);
          if (provenance != NULL)
            provenance->serialize(rez);
          else
            Provenance::serialize_null(rez);
          if (collective_mapping != NULL)
            collective_mapping->pack(rez);
          else
            CollectiveMapping::pack_null(rez);
          // Pack the field infos
          if (allocation_state == FIELD_ALLOC_READ_ONLY)
          {
            size_t num_fields = field_infos.size();
            rez.serialize<size_t>(num_fields);
            for (std::map<FieldID,FieldInfo>::const_iterator it = 
                  field_infos.begin(); it != field_infos.end(); it++)
            {
              rez.serialize(it->first);
              it->second.serialize(rez);
            }
            remote_field_infos.insert(target);
          }
          else
            rez.serialize<size_t>(0);
          rez.serialize<size_t>(semantic_info.size());
          for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
                semantic_info.begin(); it != semantic_info.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second.size);
            rez.serialize(it->second.buffer, it->second.size);
            rez.serialize(it->second.is_mutable);
          }
          rez.serialize<size_t>(semantic_field_info.size());
          for (LegionMap<std::pair<FieldID,SemanticTag>,
                SemanticInfo>::iterator
                it = semantic_field_info.begin(); 
                it != semantic_field_info.end(); it++)
          {
            rez.serialize(it->first.first);
            rez.serialize(it->first.second);
            rez.serialize(it->second.size);
            rez.serialize(it->second.buffer, it->second.size);
            rez.serialize(it->second.is_mutable);
          }
        }
        context->runtime->send_field_space_node(target, rez);
        // Finally add it to the creation set
        update_remote_instances(target);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_node_creation(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      DistributedID did;
      derez.deserialize(did);
      RtEvent initialized;
      derez.deserialize(initialized);
      AutoProvenance provenance(Provenance::deserialize(derez));
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping *mapping = NULL;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);
      FieldSpaceNode *node =
        context->create_node(handle,did,initialized,provenance,mapping,derez);
#ifdef DEBUG_LEGION
      assert(node != NULL);
#endif
      size_t num_semantic;
      derez.deserialize(num_semantic);
      for (unsigned idx = 0; idx < num_semantic; idx++)
      {
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void *buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(tag, source, buffer, buffer_size, 
                                          is_mutable, false/*local only*/);
      }
      size_t num_field_semantic;
      derez.deserialize(num_field_semantic);
      for (unsigned idx = 0; idx < num_field_semantic; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void *buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(fid, tag, source, buffer, buffer_size,
                                          is_mutable, false/*local only*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_node_request(
           RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      AddressSpaceID source;
      derez.deserialize(source);
      FieldSpaceNode *target = forest->get_node(handle);
      // If there is a collective mapping, check to see if we're on the
      // right node and if not forward it on to the right node
      if (target->collective_mapping != NULL)
      {
#ifdef DEBUG_LEGION
        assert(!target->collective_mapping->contains(source));
        assert(target->collective_mapping->contains(target->local_space));
#endif
        if (target->is_owner())
        {
          const AddressSpaceID nearest = 
            target->collective_mapping->find_nearest(source);
          // If we're not the nearest then forward it on to the
          // proper node to handle the request
          if (nearest != target->local_space)
          {
            Serializer rez;
            rez.serialize(handle);
            rez.serialize(to_trigger);
            rez.serialize(source);
            forest->runtime->send_field_space_request(nearest, rez);
            return;
          }
        }
#ifdef DEBUG_LEGION
        else
        {
          assert(target->local_space == 
              target->collective_mapping->find_nearest(source));
        }
#endif
      }
      target->send_node(source);
      Serializer rez;
      rez.serialize(to_trigger);
      forest->runtime->send_field_space_return(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_node_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_allocator_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);

      FieldSpaceNode *node = forest->get_node(handle);
      
#ifdef DEBUG_LEGION
      assert(node->is_owner());
#endif
      node->create_allocator(source, ready_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_allocator_response(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtEvent invalidations_done;
      derez.deserialize(invalidations_done);

      FieldSpaceNode *node = forest->get_node(handle);
      // wait for the invalidations to be done before handling ourselves
      if (invalidations_done.exists() && !invalidations_done.has_triggered())
        invalidations_done.wait();
      node->process_allocator_response(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_allocator_invalidation(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool flush_allocation;
      derez.deserialize(flush_allocation);
      bool merge;
      derez.deserialize(merge);

      FieldSpaceNode *node = forest->get_node(handle);
      node->process_allocator_invalidation(done_event, flush_allocation, merge);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_allocator_flush(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);

      FieldSpaceNode *node = forest->get_node(handle);
      const bool remove_free_reference = node->process_allocator_flush(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
      if (node->remove_base_gc_ref(FIELD_ALLOCATOR_REF,
            (remove_free_reference ? 2 : 1)))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_allocator_free(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);

      FieldSpaceNode *node = forest->get_node(handle);
      node->process_allocator_free(derez, source);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
      // Remove the reference that we added when we originally got the request
      if (node->remove_base_gc_ref(FIELD_ALLOCATOR_REF))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_infos_request(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      std::map<FieldID,FieldInfo> *target;
      derez.deserialize(target);
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
#ifdef DEBUG_LEGION
      assert(to_trigger.exists());
#endif
      FieldSpaceNode *node = forest->get_node(handle);
      node->request_field_infos_copy(target, source, to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_infos_response(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::map<FieldID,FieldInfo> *target; 
      derez.deserialize(target);
      size_t num_infos;
      derez.deserialize(num_infos);
      for (unsigned idx = 0; idx < num_infos; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        (*target)[fid].deserialize(derez);
      } 
      FieldSpace handle;
      derez.deserialize(handle);
      if (handle.exists())
      {
        FieldSpaceNode *node = forest->get_node(handle);
        node->record_read_only_infos(*target);
      }
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
#ifdef DEBUG_LEGION
      assert(to_trigger.exists());
#endif
      Runtime::trigger_event(to_trigger); 
    }

    //--------------------------------------------------------------------------
    char* FieldSpaceNode::to_string(const FieldMask &mask, 
                                    TaskContext *ctx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      std::string result;
      std::set<unsigned> local_indexes;
      bool invalid = false;
      size_t count = 0;  // used to skip leading comma
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
#ifdef DEBUG_LEGION
          assert(is_owner());
#endif
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (std::map<FieldID,FieldInfo>::const_iterator it = 
                field_infos.begin(); it != field_infos.end(); it++)
          {
            if (mask.is_set(it->second.idx))
            {
              if (!it->second.local)
              {
                if(count++) result += ',';
                char temp[32];
                snprintf(temp, 32, "%d", it->first);
                result += temp;
              }
              else
                local_indexes.insert(it->second.idx);
            }
          }
        }
        else
          invalid = true;
      }
      if (invalid)
      {
        std::map<FieldID,FieldInfo> local_infos;
        const RtEvent ready = 
          request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (std::map<FieldID,FieldInfo>::const_iterator it = 
              local_infos.begin(); it != local_infos.end(); it++)
        {
          if (mask.is_set(it->second.idx))
          {
            if (!it->second.local)
            {
              if(count++) result += ',';
              char temp[32];
              snprintf(temp, 32, "%d", it->first);
              result += temp;
            }
            else
              local_indexes.insert(it->second.idx);
          }
        }
      }
      if (!local_indexes.empty())
      {
        std::vector<FieldID> local_fields;
        ctx->get_local_field_set(handle, local_indexes, local_fields);
        for (std::vector<FieldID>::const_iterator it =
              local_fields.begin(); it != local_fields.end(); it++)
        {
	  if(count++) result += ',';
	  char temp[32];
          snprintf(temp, 32, "%d", *it);
          result += temp;
        }
      }
      return strdup(result.c_str());
    }

    //--------------------------------------------------------------------------
    int FieldSpaceNode::allocate_index(RtEvent &ready_event, bool initializing)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((allocation_state == FIELD_ALLOC_EXCLUSIVE) || 
              (allocation_state == FIELD_ALLOC_COLLECTIVE) || initializing);
#endif
      // Check to see if we still have spots
      int result = unallocated_indexes.find_first_set();
      if ((result >= 0) && 
          (result < int(LEGION_MAX_FIELDS - runtime->max_local_fields)))
      {
        // We still have unallocated indexes so use those first
        unallocated_indexes.unset_bit(result);
        return result;
      }
      // If there are no available indexes then we are done
      if (available_indexes.empty())
        return -1;
      std::list<std::pair<unsigned,RtEvent> >::iterator backup = 
        available_indexes.end();
      for (std::list<std::pair<unsigned,RtEvent> >::iterator it = 
            available_indexes.begin(); it != available_indexes.end(); it++)
      {
        if (!it->second.exists() || it->second.has_triggered())
        {
          // Found one without an event precondition so use it
          result = it->first;
          available_indexes.erase(it);
          return result;
        }
        else if (backup == available_indexes.end())
        {
          // If we haven't recorded a back-up then this is the
          // first once we've found so record it
          backup = it;
        }
      }
      // We didn't find one without a precondition, see if we got a backup
      if (backup != available_indexes.end())
      {
        result = backup->first;
        available_indexes.erase(backup);
        return result;
      }
      // Didn't find anything
      return -1;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_index(unsigned index, RtEvent ready_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((allocation_state == FIELD_ALLOC_EXCLUSIVE) || 
              (allocation_state == FIELD_ALLOC_COLLECTIVE));
#endif
      // Perform the invalidations across all nodes too
      std::set<RtEvent> invalidation_events;
      invalidate_layouts(index, invalidation_events, 
          context->runtime->address_space, false/*need lock*/);
      if (!invalidation_events.empty())
      {
        if (ready_event.exists())
          invalidation_events.insert(ready_event);
        ready_event = Runtime::merge_events(invalidation_events);
      }
      // Record this as an available index
      available_indexes.push_back(
          std::pair<unsigned,RtEvent>(index, ready_event)); 
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::invalidate_layouts(unsigned index, 
              std::set<RtEvent> &applied, AddressSpaceID source, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock n_lock(node_lock);
        invalidate_layouts(index, applied, source, false/*need lock*/);
        return;
      }
      // Send messages to any remote nodes to perform the invalidation
      // We're already holding the lock 
      if (has_remote_instances())
      {
        std::deque<AddressSpaceID> targets;
        FindTargetsFunctor functor(targets);
        map_over_remote_instances(functor);
        std::set<RtEvent> remote_ready;
        for (std::deque<AddressSpaceID>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
        {
          if ((*it) == source)
            continue;
          RtUserEvent remote_done = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(index);
            rez.serialize(remote_done);
          }
          pack_global_ref();
          runtime->send_field_space_layout_invalidation(*it, rez);
          applied.insert(remote_done);
        }
      }
      std::vector<LEGION_FIELD_MASK_FIELD_TYPE> to_delete;
      for (std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
                  LAYOUT_DESCRIPTION_ALLOC>>::iterator lit = 
            layouts.begin(); lit != layouts.end(); lit++)
      {
        // If the bit is set, remove the layout descriptions
        if (lit->first & (1ULL << index))
        {
          LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>
            &descs = lit->second;
          bool perform_delete = true;
          for (LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::iterator
                it = descs.begin(); it != descs.end(); /*nothing*/)
          {
            if ((*it)->allocated_fields.is_set(index))
            {
              if ((*it)->remove_reference())
                delete (*it);
              it = descs.erase(it);
            }
            else 
            {
              it++;
              perform_delete = false;
            }
          }
          if (perform_delete)
            to_delete.push_back(lit->first);
        }
      }
      for (std::vector<LEGION_FIELD_MASK_FIELD_TYPE>::const_iterator it = 
            to_delete.begin(); it != to_delete.end(); it++)
        layouts.erase(*it);
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::request_field_infos_copy(
                            std::map<FieldID,FieldInfo> *copy, 
                            AddressSpaceID source, RtUserEvent to_trigger) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(copy != NULL);
#endif
      if (is_owner())
      {
        RtEvent wait_on;
        // May need to iterate this in the case of allocation pending 
        while (true)
        {
          if (wait_on.exists() && !wait_on.has_triggered())
          {
            if (source != local_space)
            {
              // Need to defer this to avoid blocking the virtual channel
#ifdef DEBUG_LEGION
              assert(to_trigger.exists());
#endif
              DeferRequestFieldInfoArgs args(this, copy, source, to_trigger);
              context->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_DEFERRED_PRIORITY, wait_on);
              return to_trigger;
            }
            else
              wait_on.wait();
          }
          AutoLock n_lock(node_lock); 
          if (allocation_state == FIELD_ALLOC_INVALID)
          {
#ifdef DEBUG_LEGION
            // If we're invalid, that means there should be exactly
            // one remote copy which is where the allocation privileges are
            assert(remote_field_infos.size() == 1);
#endif
            // forward this message onto the node with the privileges
            const AddressSpaceID target = *(remote_field_infos.begin());
            if (!to_trigger.exists())
              to_trigger = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(copy);
              rez.serialize(source);
              rez.serialize(to_trigger);
            }
            runtime->send_field_space_infos_request(target, rez);
          }
          else if (allocation_state == FIELD_ALLOC_READ_ONLY)
          {
            // We can send back a response, make them a reader if they
            // are not one already
            if (source != local_space)
            {
#ifdef DEBUG_LEGION
              assert(to_trigger.exists());
#endif
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(copy);
                rez.serialize<size_t>(field_infos.size());
                for (std::map<FieldID,FieldInfo>::const_iterator it = 
                      field_infos.begin(); it != field_infos.end(); it++)
                {
                  rez.serialize(it->first);
                  it->second.serialize(rez);
                }
                std::set<AddressSpaceID>::const_iterator finder = 
                  remote_field_infos.find(source);
                if (finder == remote_field_infos.end())
                {
                  rez.serialize(handle);
                  remote_field_infos.insert(source);
                }
                else
                  rez.serialize(FieldSpace::NO_SPACE);
                rez.serialize(to_trigger);
              }
              runtime->send_field_space_infos_response(source, rez);
            }
            else
            {
              *copy = field_infos;
              if (to_trigger.exists())
                Runtime::trigger_event(to_trigger);
            }
          }
          else if (allocation_state == FIELD_ALLOC_PENDING)
          {
            wait_on = pending_field_allocation;
            continue;
          }
          else
          {
            // If we have allocation privileges we can send the response
            // but we can't make them a read-only copy
            if (source != local_space)
            {
#ifdef DEBUG_LEGION
              assert(to_trigger.exists());
#endif
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(copy);
                rez.serialize<size_t>(field_infos.size());
                for (std::map<FieldID,FieldInfo>::const_iterator it = 
                      field_infos.begin(); it != field_infos.end(); it++)
                {
                  rez.serialize(it->first);
                  it->second.serialize(rez);
                }
                rez.serialize(FieldSpace::NO_SPACE);
                rez.serialize(to_trigger);
              }
              runtime->send_field_space_infos_response(source, rez);
            }
            else
            {
              *copy = field_infos;
              if (to_trigger.exists())
                Runtime::trigger_event(to_trigger);
            }
          }
          // Always break out if we make it here
          break;
        }
      }
      else
      {
        // Not the owner
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        // check to see if we lost the race
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          if (source != local_space)
          {
#ifdef DEBUG_LEGION
            assert(to_trigger.exists());
#endif
            // Send the response back to the source
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(copy);
              rez.serialize<size_t>(field_infos.size());
              for (std::map<FieldID,FieldInfo>::const_iterator it = 
                    field_infos.begin(); it != field_infos.end(); it++)
              {
                rez.serialize(it->first);
                it->second.serialize(rez);
              }
              // We can't give them read-only privileges 
              rez.serialize(FieldSpace::NO_SPACE);
              rez.serialize(to_trigger);
            }
            runtime->send_field_space_infos_response(source, rez);
          }
          else
          {
            *copy = field_infos;
            if (to_trigger.exists())
              Runtime::trigger_event(to_trigger);
          }
        }
        else
        {
          // Did not lose the race, send the request back to the owner
          if (!to_trigger.exists())
            to_trigger = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(copy);
            rez.serialize(source);
            rez.serialize(to_trigger);
          }
          runtime->send_field_space_infos_request(owner_space, rez);
        }
      }
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::record_read_only_infos(
                                       const std::map<FieldID,FieldInfo> &infos)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(allocation_state == FIELD_ALLOC_INVALID);
#endif
      field_infos.insert(infos.begin(), infos.end());
      allocation_state = FIELD_ALLOC_READ_ONLY;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_allocator_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert((allocation_state == FIELD_ALLOC_INVALID) || 
              (allocation_state == FIELD_ALLOC_READ_ONLY) ||
              (allocation_state == FIELD_ALLOC_COLLECTIVE)); 
      assert(outstanding_allocators == 0);
#endif
      if (allocation_state == FIELD_ALLOC_INVALID)
      {
        size_t num_infos;
        derez.deserialize(num_infos);
        for (unsigned idx = 0; idx < num_infos; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          field_infos[fid].deserialize(derez);
        }
      }
      if (allocation_state != FIELD_ALLOC_COLLECTIVE)
      {
#ifdef DEBUG_LEGION
        assert(!unallocated_indexes);
        assert(available_indexes.empty());
#endif
        derez.deserialize(unallocated_indexes);
        size_t num_indexes;
        derez.deserialize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
        {
          std::pair<unsigned,RtEvent> index;
          derez.deserialize(index.first);
          derez.deserialize(index.second);
          available_indexes.push_back(index);
        }
      }
      // Make that we now have this in exclusive mode
      outstanding_allocators = 1;
      allocation_state = FIELD_ALLOC_EXCLUSIVE;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_allocator_invalidation(RtUserEvent done_event,
                                         bool flush_allocation, bool need_merge)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert((allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
             (allocation_state == FIELD_ALLOC_COLLECTIVE) ||
             (allocation_state == FIELD_ALLOC_READ_ONLY));
#endif
      Serializer rez;
      // It's possible to be in the read-only state even with a flush because
      // of ships passing in the night. We get sent an invalidation, but we
      // already released our allocator and sent it back to the owner so we are
      // in the read-only state and the messages pass like ships in the night
      if (flush_allocation && (allocation_state != FIELD_ALLOC_READ_ONLY))
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize<bool>(true); // allocation meta data
        rez.serialize<bool>(need_merge);
        rez.serialize(field_infos.size());
        for (std::map<FieldID,FieldInfo>::iterator it = 
              field_infos.begin(); it != field_infos.end(); /*nothing*/)
        {
          rez.serialize(it->first);
          it->second.serialize(rez);
          if (!it->second.local)
          {
            std::map<FieldID,FieldInfo>::iterator to_delete = it++;
            field_infos.erase(to_delete);
          }
          else
            it++;
        }
        rez.serialize(unallocated_indexes);
        unallocated_indexes.clear();
        rez.serialize(available_indexes.size());
        while (!available_indexes.empty())
        {
          std::pair<unsigned,RtEvent> &front = available_indexes.front();
          rez.serialize(front.first);
          rez.serialize(front.second);
          available_indexes.pop_front();
        }
        rez.serialize(outstanding_allocators);
        outstanding_allocators = 0;
        rez.serialize(done_event);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert((allocation_state == FIELD_ALLOC_READ_ONLY) ||
               (allocation_state == FIELD_ALLOC_COLLECTIVE));
#endif
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize<bool>(false); // allocation meta data
        // Invalidate our field infos
        for (std::map<FieldID,FieldInfo>::iterator it = 
              field_infos.begin(); it != field_infos.end(); /*nothing*/)
        {
          if (!it->second.local)
          {
            std::map<FieldID,FieldInfo>::iterator to_delete = it++;
            field_infos.erase(to_delete);
          }
          else
            it++;
        }
        rez.serialize(done_event);
      }
      runtime->send_field_space_allocator_flush(owner_space, rez); 
      // back to the invalid state
      allocation_state = FIELD_ALLOC_INVALID;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::process_allocator_flush(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      bool allocator_meta_data;
      derez.deserialize(allocator_meta_data);
      AutoLock n_lock(node_lock);
      if (allocator_meta_data)
      {
        bool need_merge;
        derez.deserialize(need_merge);
        if (need_merge)
        {
          size_t num_infos;
          derez.deserialize(num_infos);
          for (unsigned idx = 0; idx < num_infos; idx++)
          {
            FieldID fid;
            derez.deserialize(fid);
            field_infos[fid].deserialize(derez);
          }
          FieldMask unallocated;
          derez.deserialize(unallocated);
          unallocated_indexes |= unallocated;
          size_t num_available;
          derez.deserialize(num_available);
          for (unsigned idx = 0; idx < num_available; idx++)
          {
            std::pair<unsigned,RtEvent> next;
            derez.deserialize(next.first);
            derez.deserialize(next.second);
            bool found = false;
            for (std::list<std::pair<unsigned,RtEvent> >::const_iterator it =
                 available_indexes.begin(); it != available_indexes.end(); it++)
            {
              if (it->first != next.first)
                continue;
              found = true;
              break;
            }
            if (!found)
              available_indexes.push_back(next);
          }
          derez.advance_pointer(sizeof(outstanding_allocators));
        }
        else
        {
          size_t num_infos;
          derez.deserialize(num_infos);
          for (unsigned idx = 0; idx < num_infos; idx++)
          {
            FieldID fid;
            derez.deserialize(fid);
            field_infos[fid].deserialize(derez);
          }
#ifdef DEBUG_LEGION
          assert(!unallocated_indexes);
          assert(available_indexes.empty());
#endif
          derez.deserialize(unallocated_indexes);
          size_t num_available;
          derez.deserialize(num_available);
          for (unsigned idx = 0; idx < num_available; idx++)
          {
            std::pair<unsigned,RtEvent> next;
            derez.deserialize(next.first);
            derez.deserialize(next.second);
            available_indexes.push_back(next);
          }
          unsigned remote_allocators;
          derez.deserialize(remote_allocators);
          outstanding_allocators += remote_allocators;
        }
      }
#ifdef DEBUG_LEGION
      assert(outstanding_invalidations > 0);
      assert((allocation_state == FIELD_ALLOC_PENDING) ||
             (allocation_state == FIELD_ALLOC_INVALID)); 
#endif
      if ((--outstanding_invalidations == 0) &&
          (allocation_state == FIELD_ALLOC_PENDING))
        allocation_state = FIELD_ALLOC_EXCLUSIVE;
      return allocator_meta_data;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_allocator_free(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      bool return_allocation;
      derez.deserialize(return_allocation);
      if (return_allocation)
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert((allocation_state == FIELD_ALLOC_INVALID) ||
               (allocation_state == FIELD_ALLOC_PENDING));
        if (allocation_state == FIELD_ALLOC_INVALID)
        {
          assert(remote_field_infos.size() == 1);
          assert(remote_field_infos.find(source) != remote_field_infos.end());
          assert(outstanding_allocators == 0);
        }
        assert(!unallocated_indexes);
        assert(available_indexes.empty());
#endif
        size_t num_infos;
        derez.deserialize(num_infos);
        for (unsigned idx = 0; idx < num_infos; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          field_infos[fid].deserialize(derez);
        }
        derez.deserialize(unallocated_indexes);
        size_t num_indexes;
        derez.deserialize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
        {
          std::pair<unsigned,RtEvent> next;
          derez.deserialize(next.first);
          derez.deserialize(next.second);
          available_indexes.push_back(next);
        }
        if (allocation_state == FIELD_ALLOC_INVALID)
          allocation_state = FIELD_ALLOC_READ_ONLY;
      }
      else
        destroy_allocator(source);
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::allocate_local_indexes(CustomSerdezID serdez,
                                      const std::vector<size_t> &sizes,
                                      const std::set<unsigned> &current_indexes,
                                            std::vector<unsigned> &new_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      new_indexes.resize(sizes.size());
      // Iterate over the different fields to allocate and try to find
      // an index for them in our list of local fields
      for (unsigned fidx = 0; fidx < sizes.size(); fidx++)
      {
        const size_t field_size = sizes[fidx];
        int chosen_index = -1;
        unsigned global_idx = LEGION_MAX_FIELDS - runtime->max_local_fields;
        for (unsigned local_idx = 0; 
              local_idx < local_index_infos.size(); local_idx++, global_idx++)
        {
          // If it's already been allocated in this context then
          // we can't use it
          if (current_indexes.find(global_idx) != current_indexes.end())
            continue;
          // Check if the current local field index is used
          if (local_index_infos[local_idx].first > 0)
          {
            // Already in use, check to see if the field sizes are the same
            if ((local_index_infos[local_idx].first == field_size) &&
                (local_index_infos[local_idx].second == serdez))
            {
              // Same size so we can use it
              chosen_index = global_idx;
              break;
            }
            // Else different field size means we can't reuse it
          }
          else
          {
            // Not in use, so we can assign the size and make
            // ourselves the first user
            local_index_infos[local_idx] = 
              std::pair<size_t,CustomSerdezID>(field_size, serdez);
            chosen_index = global_idx;
            break;
          }
        }
        // If we didn't pick a valid index then we failed
        if (chosen_index < 0)
          return false;
        // Save the result
        new_indexes[fidx] = chosen_index;
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeNode::RegionTreeNode(RegionTreeForest *ctx, 
       FieldSpaceNode *column_src, RtEvent init, RtEvent tree, Provenance *prov,
       DistributedID id, CollectiveMapping *map)
      : DistributedCollectable(ctx->runtime, 
            LEGION_DISTRIBUTED_HELP_ENCODE((id > 0) ? id :
              ctx->runtime->get_available_distributed_id(),
              REGION_TREE_NODE_DC), false/*register with runtime*/, map),
        context(ctx), column_source(column_src), provenance(prov),
        initialized(init), tree_initialized(tree), registered(false)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::~RegionTreeNode(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      }
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID RegionTreeNode::get_owner_space(RegionTreeID tid,
                                                              Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (tid % runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::attach_semantic_information(SemanticTag tag,
                                                     AddressSpaceID source,
                                                     const void *buffer,
                                                     size_t size,
                                                     bool is_mutable,
                                                     bool local_only)
    //--------------------------------------------------------------------------
    {
      // Make a copy
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // Check to make sure that the bits are the same
              if (size != finder->second.size)
                REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                  "Inconsistent Semantic Tag value "
                              "for tag %ld with different sizes of %zd"
                              " and %zd for region tree node", 
                              tag, size, finder->second.size)
              // Otherwise do a bitwise comparison
              {
                const char *orig = (const char*)finder->second.buffer;
                const char *next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                    REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                      "Inconsistent Semantic Tag value "
                                  "for tag %ld with different values at"
                                  "byte %d for region tree node, %x != %x", 
                                  tag, idx, orig[idx], next[idx])
                }
              }
              added = false;
            }
            else
            {
              // Mutable so we can just overwrite it
              legion_free(SEMANTIC_INFO_ALLOC, finder->second.buffer,
                          finder->second.size);
              finder->second.buffer = local;
              finder->second.size = size;
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer = local;
            finder->second.size = size;
            // Trigger will happen by caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_info[tag] = SemanticInfo(local, size, is_mutable);
      }
      if (added)
      {
        AddressSpaceID owner_space = find_semantic_owner();
        // If we are not the owner and the message 
        // didn't come from the owner, then send it 
        if ((owner_space != context->runtime->address_space) &&
            (source != owner_space) && !local_only)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          send_semantic_info(owner_space, tag, buffer, size, is_mutable, done);
          if (!done.has_triggered())
            done.wait();
        }
      }
      else
        legion_free(SEMANTIC_INFO_ALLOC, local, size);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::retrieve_semantic_information(SemanticTag tag,
                                                       const void *&result,
                                                       size_t &size,
                                                       bool can_fail,
                                                       bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = find_semantic_owner();
      const bool is_remote = (owner_space != context->runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        LegionMap<SemanticTag,SemanticInfo>::const_iterator finder = 
          semantic_info.find(tag); 
        if (finder != semantic_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else // can use the canonical event
              wait_on = finder->second.ready_event; 
          }
          else if (wait_until) // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG,
                      "invalid semantic tag %ld for "
                      "region tree node", tag)
      }
      else
      {
        if (is_remote && request.exists())
          send_semantic_request(owner_space, tag, can_fail, wait_until,request);
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      LegionMap<SemanticTag,SemanticInfo>::const_iterator finder = 
        semantic_info.find(tag);
      if (finder == semantic_info.end())
      {
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG,
                            "invalid semantic tag %ld for "
                            "region tree node", tag)
      }
      result = finder->second.buffer;
      size = finder->second.size;
      return true;
    }

    //--------------------------------------------------------------------------
    ProjectionSummary* RegionTreeNode::compute_projection_summary(
                   Operation *op, unsigned index, const RegionRequirement &req,
                   LogicalAnalysis &analysis, const ProjectionInfo &proj_info)
    //--------------------------------------------------------------------------
    {
      const ContextID ctx = analysis.context->get_logical_tree_context();
      LogicalState &state = get_logical_state(ctx);
      return state.find_or_create_projection_summary(op, index, req,
                                                     analysis, proj_info);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_user(LogicalRegion privilege_root,
                                       LogicalUser &user,
                                       const RegionTreePath &path,
                                       const LogicalTraceInfo &trace_info,
                                       const ProjectionInfo &proj_info,
                                       const FieldMask &user_mask,
                                       FieldMask &unopened_field_mask,
                                       FieldMask &refinement_mask,
                                       LogicalAnalysis &logical_analysis,
                                       FieldMaskSet<RefinementOp,
                                        UNTRACKED_ALLOC,true> &refinements,
                                       const bool root_node)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_REGISTER_LOGICAL_USER_CALL);
      const ContextID ctx = 
        logical_analysis.context->get_logical_tree_context();
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
      const unsigned depth = get_depth();
      const bool arrived = !path.has_child(depth);
      FieldMask open_below;
      RegionTreeNode *next_child = NULL;
      if (!arrived)
        next_child = get_tree_child(path.get_child(depth));
      // Check to see if we need to traverse any interfering children
      // and record dependences on prior operations in that tree
      if (!!unopened_field_mask)
        siphon_interfering_children(state, logical_analysis,
            user_mask, user, privilege_root, next_child, open_below);
      else if (!arrived)
        // Everything is open-only so make a state and merge it in
        add_open_field_state(state, user, user_mask, next_child);
      // Perform our local dependence analysis at this node along the path
      FieldMask dominator_mask = 
             perform_dependence_checks<true/*track dom*/>(privilege_root,
                          user, state.curr_epoch_users, user_mask,
                          open_below, arrived, proj_info,
                          state, logical_analysis);
      FieldMask non_dominated_mask = user_mask - dominator_mask;
      // For the fields that weren't dominated, we have to check
      // those fields against the previous epoch's users
      if (!!non_dominated_mask)
        perform_dependence_checks<false/*track dom*/>(privilege_root,
                          user, state.prev_epoch_users, non_dominated_mask,
                          open_below, arrived, proj_info,
                          state, logical_analysis);
      if (arrived)
      {
        // If we dominated and this is our final destination then we 
        // can filter the operations since we actually do dominate them
        if (!!dominator_mask)
        {
          // Dominator mask is not empty
          // Mask off all the dominated fields from the previous set
          // of epoch users and remove any previous epoch users
          // that were totally dominated
          state.filter_previous_epoch_users(dominator_mask); 
          // Mask off all dominated fields from current epoch users and move
          // them to prev epoch users.  If all fields masked off, then remove
          // them from the list of current epoch users.
          state.filter_current_epoch_users(dominator_mask);
        }
        // If we've arrived add ourselves as a user
        state.register_local_user(user, user_mask);
        // If we still have a refinement mask then we record that we should
        // do a refinement operation from this node before the operation
        if (!!refinement_mask)
        {
          if (proj_info.is_projecting())
          {
            if (user.shard_proj == NULL)
            {
              ProjectionSummary *summary = 
                state.find_or_create_projection_summary(user.op, user.idx,
                              trace_info.req, logical_analysis, proj_info);
              state.update_refinement_projection(ctx, summary,
                                                 user.usage, refinement_mask);
            }
            else
              state.update_refinement_projection(ctx, user.shard_proj,
                                          user.usage, refinement_mask);
          }
          else
            state.update_refinement_arrival(ctx, user.usage, refinement_mask);
          // We can skip performing refinements at the root node
          if (!!refinement_mask && !root_node)
            logical_analysis.record_pending_refinement(privilege_root,
                user.idx, user.op->find_parent_index(user.idx),
                this, refinement_mask, refinements);
        }
      }
      else
      {
        // We haven't arrived so we need to traverse to the next child
        // Get our set of fields which are being opened for
        // the first time at the next level
        if (!!unopened_field_mask)
        {
          if (!!open_below)
            // Update our unopened children mask
            // to reflect any fields which are still open below
            unopened_field_mask &= open_below;
          else
            // Open all the unopened fields
            unopened_field_mask.clear();
        }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        else // if they weren't open here, they shouldn't be open below
          assert(!open_below);
#endif
#endif
        if (!!refinement_mask)
          state.update_refinement_child(ctx, next_child, 
                                        user.usage, refinement_mask);
        next_child->register_logical_user(privilege_root, user, path,
            trace_info, proj_info, user_mask, unopened_field_mask,
            refinement_mask, logical_analysis, refinements, false/*root node*/);
      }
      // If we have any refinement operations then we need to perform their
      // dependence analysis now on the way back up the tree after having 
      // done everything else
      if (!refinements.empty())
      {
        const ProjectionInfo no_projection_info(nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                IndexSpace::NO_SPACE);
        const RegionUsage ref_usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
        for (FieldMaskSet<RefinementOp,UNTRACKED_ALLOC,true>::const_iterator
              it = refinements.begin(); it != refinements.end(); it++)
        {
          const LogicalUser refinement_user(it->first, 0/*index*/, ref_usage);
          // Recording refinement dependences will record dependences on 
          // anything in an interfering sub-tree without changing the
          // state of the region tree states
          state.record_refinement_dependences(ctx, refinement_user, it->second,
              no_projection_info, next_child, privilege_root, logical_analysis);
        }
        // A bit of a hairy case: if the user is not read-write and we have
        // refinements below then we need to promote the state of the child
        // sub-tree up to exclusive so that later operations will know that
        // they need to traverse the sub-tree and find the dependence on the
        // refinement operation
        if ((next_child != NULL) && !IS_WRITE(user.usage))
          state.promote_next_child(next_child, refinements.get_valid_mask());
      }
      // Perform any filtering that we need to do for timeout users
      state.filter_timeout_users(logical_analysis);
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_refinement_dependences(ContextID ctx,
        const LogicalUser &refinement_user, const FieldMask &refinement_mask,
        const ProjectionInfo &no_proj_info, RegionTreeNode *previous_child,
        LogicalRegion privilege_root, LogicalAnalysis &logical_analysis)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
      state.record_refinement_dependences(ctx, refinement_user, 
          refinement_mask, no_proj_info, previous_child,
          privilege_root, logical_analysis);
      state.filter_timeout_users(logical_analysis);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_logical_refinement(ContextID ctx,
                                             const FieldMask &invalidation_mask)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      state.invalidate_refinements(ctx, invalidation_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::add_open_field_state(LogicalState &state,
                                              const LogicalUser &user,
                                              const FieldMask &open_mask,
                                              RegionTreeNode *next_child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
      FieldState new_state(user.usage, open_mask, next_child);
      merge_new_field_state(state, new_state);
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
    }
    
    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_interfering_children(LogicalState &state,
                                                 LogicalAnalysis &analysis,
                                                 const FieldMask &closing_mask,
                                                 const LogicalUser &user,
                                                 LogicalRegion privilege_root,
                                                 RegionTreeNode *next_child,
                                                 FieldMask &open_below)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_SIPHON_LOGICAL_CHILDREN_CALL);
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
      // These are fields for which the next_child is already open but it was
      // in the wrong state so we still need to add a new state for them
      FieldMask next_child_fields;
      // Now we can look at all the children
      for (LegionList<FieldState>::iterator it = 
            state.field_states.begin(); it != 
            state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields() * closing_mask)
        {
          it++;
          continue;
        }
        // Now check the current state
        switch (it->open_state)
        {
          case OPEN_READ_ONLY:
            {
              if (IS_READ_ONLY(user.usage))
              {
                // We're read-only too so there is no need to traverse
                // See if the child that we want is already open
                if (next_child != NULL)
                {
                  OrderedFieldMaskChildren::const_iterator finder =
                    it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Remove the child's open fields from the
                    // list of fields we need to open
                    open_below |= (finder->second & closing_mask);
                  }
                }
                it++;
              }
              else
              {
                // Not-read only so traverse the interfering children and
                // close up anything that is not the next child
                perform_close_operations(user, closing_mask, it->open_children,
                    privilege_root, this, analysis, open_below, next_child,
                    &next_child_fields, true/*filter next*/);
                // See if there are still any valid open fields
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          case OPEN_READ_WRITE:
            {
              // Close up any interfering children that conflict
              perform_close_operations(user, closing_mask, it->open_children,
                  privilege_root, this, analysis, open_below, next_child);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_REDUCE:
            {
              // See if this reduction is a reduction of the same kind
              if (IS_REDUCE(user.usage) && (user.usage.redop == it->redop))
              {
                if (next_child != NULL)
                {
                  OrderedFieldMaskChildren::const_iterator finder = 
                    it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Already open, so add the open fields
                    open_below |= (finder->second & closing_mask);
                  }
                }
                it++;
              }
              else
              {
                // Need to close up the open field since we're going
                // to have to do it anyway
                perform_close_operations(user, closing_mask, it->open_children,
                    privilege_root, this, analysis, open_below, next_child,
                    &next_child_fields, true/*filter next*/);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          default:
            assert(false);
        }
      }
      // If we had any fields that still need to be opened, create
      // a new field state and add it into the set of new states
      if (next_child != NULL)
      {
        FieldMask open_mask = closing_mask;
        if (!!open_below)
          open_mask -= open_below;
        if (!!next_child_fields)
          open_mask |= next_child_fields;
        if (!!open_mask)
        {
          FieldState new_state(user.usage, open_mask, next_child);
          merge_new_field_state(state, new_state);
        }
      }
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_close_operations(const LogicalUser &user,
                                        const FieldMask &closing_mask,
                                        OrderedFieldMaskChildren &children,
                                        LogicalRegion privilege_root,
                                        RegionTreeNode *path_node,
                                        LogicalAnalysis &analysis,
                                        FieldMask &open_below,
                                        RegionTreeNode *next_child,
                                        FieldMask *next_child_fields,
                                        const bool filter_next_child)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_PERFORM_LOGICAL_CLOSES_CALL);
      if (next_child != NULL)
      {
        if (!are_all_children_disjoint())
        {
          // The children are not disjoint so we need to traverse them all
          // see if they are disjoint with the next child
          const LegionColor next_color = next_child->get_color();
          std::vector<RegionTreeNode*> to_delete;
          for (OrderedFieldMaskChildren::iterator it =
                children.begin(); it != children.end(); it++)
          {
            if (next_child == it->first) // we'll handle the next child below
              continue;
            const FieldMask close_mask = closing_mask & it->second;
            if (!close_mask)
              continue;
            if (are_children_disjoint(it->first->get_color(), next_color))
              continue;
            FieldMask still_open;
            it->first->close_logical_node(user, close_mask, privilege_root,
                                          path_node, analysis, still_open);
#ifdef POINT_WISE_LOGICAL_ANALYSIS
            analysis.bail_point_wise_analysis = true;
#endif
            if (!!still_open)
            {
              if (still_open != close_mask)
              {
                it.filter(close_mask - still_open);
                if (!it->second)
                  to_delete.push_back(it->first);
              }
            }
            else
            {
              it.filter(close_mask);
              if (!it->second)
                to_delete.push_back(it->first);
            }
          }
          if (!to_delete.empty())
          {
            for (std::vector<RegionTreeNode*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              children.erase(*it);
              if ((*it)->remove_base_gc_ref(FIELD_STATE_REF))
                delete (*it);
            }
            children.tighten_valid_mask();
          }
        }
        // Now handle the next child
        OrderedFieldMaskChildren::iterator finder =
          children.find(next_child);
        if (finder != children.end())
        {
          FieldMask overlap = closing_mask & finder->second;
          if (!!overlap)
          {
            if (filter_next_child)
            {
#ifdef DEBUG_LEGION
              assert(next_child_fields != NULL);
#endif
              FieldMask child_fields;
              next_child->close_logical_node(user, overlap, privilege_root,
                                             path_node, analysis, child_fields);
#ifdef POINT_WISE_LOGICAL_ANALYSIS
              analysis.bail_point_wise_analysis = true;
#endif
              if (!!child_fields)
              {
                open_below |= child_fields;
                (*next_child_fields) |= child_fields;
              }
              finder.filter(overlap);
              if (!finder->second)
              {
                children.erase(finder);
                if (next_child->remove_base_gc_ref(FIELD_STATE_REF))
                  assert(false); // should never delete the next child
              }
              children.tighten_valid_mask();
            }
            else
              open_below |= overlap;
          }
        }
      }
      else
      {
        // We don't have a next child we're doing to, so we just need to
        // close up all the open children
        std::vector<RegionTreeNode*> to_delete;
        for (OrderedFieldMaskChildren::iterator it =
              children.begin(); it != children.end(); it++)
        {
          const FieldMask close_mask = closing_mask & it->second;
          if (!close_mask)
            continue;
          FieldMask still_open;
          it->first->close_logical_node(user, close_mask, privilege_root,
                                        path_node, analysis, still_open);
#ifdef POINT_WISE_LOGICAL_ANALYSIS
          analysis.bail_point_wise_analysis = true;
#endif
          if (!!still_open)
          {
            open_below |= still_open;
            if (still_open != close_mask)
            {
              it.filter(close_mask - still_open);
              if (!it->second)
                to_delete.push_back(it->first);
            }
          }
          else
          {
            it.filter(close_mask);
            if (!it->second)
              to_delete.push_back(it->first);
          }
        }
        if (!to_delete.empty())
        {
          for (std::vector<RegionTreeNode*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            children.erase(*it);
            if ((*it)->remove_base_gc_ref(FIELD_STATE_REF))
              delete (*it);
          }
          children.tighten_valid_mask();
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_logical_node(const LogicalUser &user,
                                            const FieldMask &closing_mask,
                                            LogicalRegion privilege_root,
                                            RegionTreeNode *path_node,
                                            LogicalAnalysis &logical_analysis,
                                            FieldMask &still_open)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_CLOSE_LOGICAL_NODE_CALL);
      LogicalState &state = 
        get_logical_state(logical_analysis.context->get_logical_tree_context());
      // Perform closing checks on both the current epoch users
      // as well as the previous epoch users
      perform_closing_checks(logical_analysis, state.curr_epoch_users,
          user, closing_mask, privilege_root, path_node, still_open);
      perform_closing_checks(logical_analysis, state.prev_epoch_users,
          user, closing_mask, privilege_root, path_node, still_open);
      if (!state.field_states.empty())
      {
        // Recursively traverse any open children and close them as well
        for (std::list<FieldState>::iterator it = state.field_states.begin();
              it != state.field_states.end(); /*nothing*/)
        {
          FieldMask overlap = it->valid_fields() & closing_mask;
          if (!overlap)
          {
            it++;
            continue;
          }
          perform_close_operations(user, overlap, it->open_children,
              privilege_root, path_node, logical_analysis, still_open);
          // Remove the state if it is now empty
          if (!it->valid_fields())
            it = state.field_states.erase(it);
          else
            it++;
        }
      }
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_state(LogicalState &state,
                                               FieldState &new_state)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!new_state.valid_fields());
#endif
      for (std::list<FieldState>::iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->overlaps(new_state))
        {
          it->merge(new_state, this);
          return;
        }
      }
      // Otherwise just push it on the back
      state.field_states.emplace_back(std::move(new_state));
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::report_uninitialized_usage(Operation *op, unsigned idx,
                                  const FieldMask &uninit, RtUserEvent reported)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_region());
      assert(reported.exists());
#endif
      char *field_string = column_source->to_string(uninit, op->get_context());
      op->report_uninitialized_usage(idx, field_string, reported);
      free(field_string);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::initialize_current_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState &state = get_logical_state(ctx);
      state.check_init();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_current_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState &state = get_logical_state(ctx);
      state.clear(); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_deleted_state(ContextID ctx,
                                                  const FieldMask &deleted_mask)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState &state = get_logical_state(ctx);
      state.clear_deleted_state(ctx, deleted_mask);
    }

    //--------------------------------------------------------------------------
    template<bool TRACK_DOM>
    FieldMask RegionTreeNode::perform_dependence_checks(LogicalRegion root,
                const LogicalUser &user, OrderedFieldMaskUsers &prev_users,
                const FieldMask &check_mask, const FieldMask &open_below,
                const bool arrived, const ProjectionInfo &proj_info,
                LogicalState &state, LogicalAnalysis &logical_analysis)
    //--------------------------------------------------------------------------
    {
      FieldMask dominator_mask = check_mask;
      // It's not actually sound to assume we dominate something
      // if we don't observe any users of those fields.  Therefore
      // also keep track of the fields that we observe.  We'll use this
      // at the end when computing the final dominator mask.
      FieldMask observed_mask; 
      if (!(check_mask * prev_users.get_valid_mask()))
      {
        bool tighten = false;
        std::vector<LogicalUser*> to_delete;
        for (OrderedFieldMaskUsers::iterator it =
              prev_users.begin(); it != prev_users.end(); it++)
        {
          // Don't record dependences on any other users from the same op
          LogicalUser &prev = *(it->first);
          if ((prev.ctx_index == user.ctx_index) && 
              // Note this second condition only happens for must-epoch 
              // operations where multiple tasks are coming through here
              // and we still need to record their mapping dependences
              // so we don't want to go into the scope. If we ever get
              // rid of must-epoch operations we can get rid of the 
              // second part of this condition
              ((prev.op == user.op) || (user.op->get_must_epoch_op() == NULL)))
          {
            if (TRACK_DOM)
              dominator_mask -= it->second;
            continue;
          }
          const FieldMask overlap = check_mask & it->second;
#ifdef POINT_WISE_LOGICAL_ANALYSIS
          bool skip_registering_region_dependence = false;
#endif
          if (!!overlap)
          {
            if (TRACK_DOM)
              observed_mask |= overlap;
            const DependenceType dtype = 
              check_dependence_type<true>(prev.usage, user.usage);
            switch (dtype)
            {
              case LEGION_NO_DEPENDENCE:
                {
                  // No dependence so remove bits from the dominator mask
                  dominator_mask -= it->second;
                  break;
                }
              case LEGION_ANTI_DEPENDENCE:
              case LEGION_ATOMIC_DEPENDENCE:
              case LEGION_SIMULTANEOUS_DEPENDENCE:
              case LEGION_TRUE_DEPENDENCE:
                {
#ifdef POINT_WISE_LOGICAL_ANALYSIS
                  if (prev.shard_proj != NULL && user.shard_proj != NULL &&
                      prev.op->get_operation_kind() == Operation::TASK_OP_KIND &&
                      user.op->get_operation_kind() == Operation::TASK_OP_KIND)
                  {
                    if (static_cast<TaskOp*>(user.op)->get_task_kind() ==
                        TaskOp::INDEX_TASK_KIND &&
                        static_cast<TaskOp*>(prev.op)->get_task_kind() ==
                        TaskOp::INDEX_TASK_KIND)
                    {
                      if(logical_analysis.bail_point_wise_analysis)
                      {
                        printf("bailing!!!!\n");
                      }
                      if (static_cast<IndexTask*>(user.op)->
                          prev_point_wise_user_set(user.idx))
                      {
                        // We bail if we have more than one ancestor for now
                        logical_analysis.bail_point_wise_analysis = true;
                      }
                      if (!logical_analysis.bail_point_wise_analysis) {
                        if(!prev.shard_proj->is_disjoint() || !prev.shard_proj->can_perform_name_based_self_analysis()) {
                          logical_analysis.bail_point_wise_analysis = true;
                        }
                        else if ((user.shard_proj->projection->projection_id !=
                              prev.shard_proj->projection->projection_id) ||
                            !user.shard_proj->projection->is_functional ||
                            (!user.shard_proj->projection->is_invertible &&
                             user.shard_proj->projection->projection_id != 0))
                        {
                          logical_analysis.bail_point_wise_analysis = true;
                        }
                        else
                        {
                          bool parent_dominates = prev.shard_proj->domain->dominates(user.shard_proj->domain);
                          if(parent_dominates)
                          {
                            printf("FOUND POINT-WISE ANCESTOR: %d %lld %lld\n", context->runtime->address_space, prev.uid, user.uid);
                            skip_registering_region_dependence = true;
                            if(!static_cast<IndexTask*>(prev.op)->set_next_point_wise_user(
                                &user, prev.gen, prev.idx))
                            {
                              static_cast<IndexTask*>(user.op)->record_point_wise_dependence_completed_points_prev_task(
                                  prev.shard_proj, prev.ctx_index);
                            }
                            static_cast<IndexTask*>(user.op)->set_prev_point_wise_user(
                                &prev, user.idx, dtype, prev.idx);
                          }
                        }
                      }
                    }
                  }
                  if(!skip_registering_region_dependence)
                  {
#endif
                  // If we can validate a region record which of our
                  // predecessors regions we are validating, otherwise
                  // just register a normal dependence
                  user.op->register_region_dependence(user.idx, prev.op,
                                                      prev.gen, prev.idx,
                                                      dtype, overlap);
#ifdef LEGION_SPY
                  LegionSpy::log_mapping_dependence(
                      user.op->get_context()->get_unique_id(),
                      prev.uid, prev.idx, user.uid, user.idx, dtype);
#endif

                  if (prev.shard_proj != NULL)
                  {
                   // Two operations from the same must epoch shouldn't
                    // be recording close dependences on each other so
                    // we can skip that part
                    if ((prev.ctx_index == user.ctx_index) &&
                        (user.op->get_must_epoch_op() != NULL))
                      break;
                    // If this is a sharding projection operation then check 
                    // to see if we need to record a fence dependence here to
                    // ensure that we get dependences between interfering 
                    // points in different shards correct
                    // There are three sceanrios here:
                    // 1. We haven't arrived in which case we don't have any 
                    //    good way to symbolically prove it is safe to elide
                    //    the fence so just record the close
                    // 2. We've arrived but we're not projection in which case
                    //    we'll interfere with any projections anyway so we need
                    //    a full fence for dependences anyway
                    // 3. We've arrived and are projecting in which case we can
                    //    try to elide things symbolically, if that doesn't work
                    //    we may still need to do an expensive analysis to prove
                    //    it is safe to elide the close which we'll only do it
                    //    we are tracing
                    if (arrived && proj_info.is_projecting())
                    {
                      // If we arrived and are projecting then we can test
                      // these two projection trees for intereference with
                      // each other and see if we can prove that they are
                      // disjoint in which case we don't need a close
#ifdef DEBUG_LEGION
#ifndef POINT_WISE_LOGICAL_ANALYSIS
                      assert(proj_info.is_sharding());
#endif
                      assert(user.shard_proj != NULL);
#endif
                      if (!state.has_interfering_shards(logical_analysis,
                                          prev.shard_proj, user.shard_proj))
                        break;
                    }
                    // We weren't able to prove that the projections were
                    // non-interfering with each other so we need a close
                    // Not able to do the symbolic elision so we need a fence
                    // across the shards to be safe
                    logical_analysis.record_close_dependence(root,
                                  user.idx, this, &prev, overlap);
                    it.filter(overlap);
                    tighten = true;
                    if (!it->second)
                      to_delete.push_back(it->first);
                  }

#ifdef POINT_WISE_LOGICAL_ANALYSIS
                  }
#endif
                  break;
                }
              default:
                assert(false); // should never get here
            }
          }
        }
        if (!to_delete.empty())
        {
          for (std::vector<LogicalUser*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            prev_users.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
        }
        if (tighten)
          prev_users.tighten_valid_mask();
      }
      // The result of this computation is the dominator mask.
      // It's only sound to say that we dominate fields that
      // we actually observed users for so intersect the dominator 
      // mask with the observed mask
      if (TRACK_DOM)
      {
        // For writes, there is a special case here we actually
        // want to record that we are dominating fields which 
        // are not actually open below even if we didn't see
        // any users on the way down
        if (IS_WRITE(user.usage))
        {
          FieldMask unobserved = check_mask - observed_mask;
          if (!!unobserved)
          {
            if (!open_below)
              observed_mask |= unobserved;
            else
              observed_mask |= (unobserved - open_below);
          }
        }
        return (dominator_mask & observed_mask);
      }
      else
        return dominator_mask;
    }

    // Instantiate the template for both templates because c++ is stupid
    template FieldMask 
      RegionTreeNode::perform_dependence_checks<true>(LogicalRegion root,
                const LogicalUser &user, OrderedFieldMaskUsers &prev_users,
                const FieldMask &check_mask, const FieldMask &open_below,
                const bool arrived, const ProjectionInfo &proj_info,
                LogicalState &state, LogicalAnalysis &logical_analysis);
    template FieldMask 
      RegionTreeNode::perform_dependence_checks<false>(LogicalRegion root,
                const LogicalUser &user, OrderedFieldMaskUsers &prev_users,
                const FieldMask &check_mask, const FieldMask &open_below,
                const bool arrived, const ProjectionInfo &proj_info,
                LogicalState &state, LogicalAnalysis &logical_analysis);

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::perform_closing_checks(
        LogicalAnalysis &logical_analysis, OrderedFieldMaskUsers &users,
        const LogicalUser &user, const FieldMask &check_mask, 
        LogicalRegion root, RegionTreeNode *path_node, FieldMask &still_open)
    //--------------------------------------------------------------------------
    {
      // Record dependences on all operations with the same field unless they
      // are different region requirements of the same user
      // There's nothing to do if the mask is disjoint from the users
      if (check_mask * users.get_valid_mask())
        return;
      std::vector<LogicalUser*> to_delete;
      for (OrderedFieldMaskUsers::iterator it = 
            users.begin(); it != users.end(); it++)
      {
        const FieldMask overlap = check_mask & it->second;
        if (!overlap)
          continue;
        const LogicalUser &prev = *it->first;
        // Skip any users from the same operation for different requiremnts
        if (prev.ctx_index == user.ctx_index)
          continue;
        logical_analysis.record_close_dependence(root, user.idx, path_node,
                                                 it->first, overlap);
        it.filter(overlap);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<LogicalUser*>::const_iterator it =
            to_delete.begin(); it != to_delete.end(); it++)
      {
        users.erase(*it);
        if ((*it)->remove_reference())
          delete (*it);
      }
      users.tighten_valid_mask();
      if (!users.empty())
        still_open |= (check_mask & users.get_valid_mask());
    }

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion r, PartitionNode *par,
                           IndexSpaceNode *row_src, FieldSpaceNode *col_src,
                           RegionTreeForest *ctx, DistributedID id,
                           RtEvent init, RtEvent tree,
                           CollectiveMapping *mapping, Provenance *prov)
      : RegionTreeNode(ctx, col_src, init, tree, prov, id, mapping), handle(r),
        parent(par), row_source(row_src)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Region %lld %d %d %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, 
          handle.get_index_space().get_id(),
          handle.get_field_space().get_id(),
          handle.get_tree_id());
#endif
    }

    //--------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------
    { 
      // The reason we would be here is if we were leaked
      if (!partition_trackers.empty())
      {
        for (std::list<PartitionTracker*>::const_iterator it = 
              partition_trackers.begin(); it != partition_trackers.end(); it++)
          if ((*it)->remove_partition_reference())
            delete (*it);
        partition_trackers.clear();
      }
      if (registered)
      {
        if (column_source->remove_nested_resource_ref(did))
          delete column_source;
        // Unregister oursleves with the row source
        if (row_source->remove_nested_resource_ref(did))
          delete row_source;
        const bool top_level = (parent == NULL);
        // Unregister ourselves with the context
        context->remove_node(handle, top_level);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        context->runtime->release_tree_instances(handle.get_tree_id());
        if (row_source->parent == NULL)
          row_source->remove_nested_valid_ref(did);
        else
          row_source->parent->remove_nested_valid_ref(did);
        column_source->remove_nested_gc_ref(did);
      }
      if (!partition_trackers.empty())
      {
#ifdef DEBUG_LEGION
        assert(parent == NULL); // should only happen on the root
#endif
        for (std::list<PartitionTracker*>::const_iterator it = 
              partition_trackers.begin(); it != partition_trackers.end(); it++)
          if ((*it)->remove_partition_reference())
            delete (*it);
        partition_trackers.clear();
      }
      for (unsigned idx = 0; idx < current_versions.max_entries(); idx++)
        if (current_versions.has_entry(idx))
          get_current_version_manager(idx).finalize_manager();
    }

    //--------------------------------------------------------------------------
    void RegionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!registered);
#endif
      if (parent == NULL)
      {
        if (row_source->parent == NULL)
          row_source->add_nested_valid_ref(did);
        else
          row_source->parent->add_nested_valid_ref(did);
        column_source->add_nested_gc_ref(did);
      }
      else
        parent->add_child(this);
      column_source->add_nested_resource_ref(did);
      row_source->add_nested_resource_ref(did);
      registered = true;
      if (parent == NULL)
        register_with_runtime();
    }

    //--------------------------------------------------------------------------
    bool RegionNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Ask the row source since it eagerly instantiates
      return row_source->has_color(c);
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::get_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // check to see if we have it, if not try to make it
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<LegionColor,PartitionNode*>::const_iterator finder = 
          color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      // If we get here we didn't immediately have it so try
      // to make it through the proper channels
      IndexPartNode *index_part = row_source->get_child(c);
#ifdef DEBUG_LEGION
      assert(index_part != NULL);
#endif
      LogicalPartition part_handle(handle.tree_id, index_part->handle,
                                   handle.field_space);
      return context->create_node(part_handle, this);
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_child(PartitionNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(color_map.find(child->row_source->color) == color_map.end());
#endif
      color_map[child->row_source->color] = child;
    }

    //--------------------------------------------------------------------------
    void RegionNode::remove_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      std::map<LegionColor,PartitionNode*>::iterator finder = color_map.find(c);
      assert(finder != color_map.end());
      color_map.erase(finder);
#else
      color_map.erase(c);
#endif
    } 

    //--------------------------------------------------------------------------
    void RegionNode::add_tracker(PartitionTracker *tracker)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent == NULL); // should only happen on the root
#endif
      std::vector<PartitionTracker*> to_prune;
      {
        AutoLock n_lock(node_lock);
        // To avoid leaks, see if there are any other trackers we can prune
        for (std::list<PartitionTracker*>::iterator it =
              partition_trackers.begin(); it != 
              partition_trackers.end(); /*nothing*/)
        {
          if ((*it)->can_prune())
          {
            to_prune.push_back(*it);
            it = partition_trackers.erase(it);
          }
          else
            it++;
        }
        partition_trackers.push_back(tracker);
      }
      for (std::vector<PartitionTracker*>::const_iterator it =
            to_prune.begin(); it != to_prune.end(); it++)
        if ((*it)->remove_reference())
          delete (*it);
    }

    //--------------------------------------------------------------------------
    void RegionNode::initialize_no_refine_fields(ContextID ctx,
                                               const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      state.sanity_check();
#endif
      state.initialize_no_refine_fields(mask);
    }

    //--------------------------------------------------------------------------
    unsigned RegionNode::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->depth;
    }

    //--------------------------------------------------------------------------
    LegionColor RegionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* RegionNode::get_row_source(void) const
    //--------------------------------------------------------------------------
    {
      return row_source;
    }

    //--------------------------------------------------------------------------
    RegionTreeID RegionNode::get_tree_id(void) const
    //--------------------------------------------------------------------------
    {
      return handle.get_tree_id();
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_tree_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_children_disjoint(const LegionColor c1, 
                                           const LegionColor c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_all_children_disjoint(void)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::is_region(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    RegionNode* RegionNode::as_region_node(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<RegionNode*>(this);
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::as_partition_node(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }
#endif

    //--------------------------------------------------------------------------
    bool RegionNode::visit_node(PathTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      return traverser->visit_region(this);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::visit_node(NodeTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal = traverser->visit_region(this);
      if (continue_traversal)
      {
        const bool break_early = traverser->break_early();
        if (traverser->force_instantiation)
        {
          // If we are forcing instantiation, then grab the set of 
          // colors from the row source and use them to instantiate children
          std::vector<LegionColor> children_colors;
          row_source->get_colors(children_colors); 
          for (std::vector<LegionColor>::const_iterator it = 
                children_colors.begin(); it != children_colors.end(); it++)
          {
            bool result = get_child(*it)->visit_node(traverser);
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
              break;
          }
        }
        else
        {
          std::map<LegionColor,PartitionNode*> children;
          // Need to hold the lock when reading from 
          // the color map or the valid map
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            for (std::map<LegionColor,PartitionNode*>::const_iterator it = 
                  color_map.begin(); it != color_map.end(); it++)
            {
              children.insert(*it);
              it->second->add_base_resource_ref(REGION_TREE_REF);
            }
          }
          for (std::map<LegionColor,PartitionNode*>::const_iterator it = 
                children.begin(); it != children.end(); it++)
          {
            const bool result = it->second->visit_node(traverser);
            if (it->second->remove_base_resource_ref(REGION_TREE_REF))
              delete it->second;
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
            {
              it++;
              while (it != children.end())
              {
                if (it->second->remove_base_resource_ref(REGION_TREE_REF))
                  delete it->second;
                it++;
              }
              break;
            }
          }
        }
      }
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::is_complete(void)
    //--------------------------------------------------------------------------
    {
      // For now just assume that regions are never complete
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::intersects_with(RegionTreeNode *other, bool compute)
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_region())
        return row_source->intersects_with(
                  other->as_region_node()->row_source, compute);
      else
        return row_source->intersects_with(
                  other->as_partition_node()->row_source, compute);
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_num_children();
    }

    //--------------------------------------------------------------------------
    void RegionNode::send_node(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      {
        AutoLock n_lock(node_lock); 
        if (!has_remote_instance(target))
        {
          continue_up = true;
          update_remote_instances(target);
        }
      }
      if (continue_up)
      {
        if (parent != NULL)
        {
          // Send the parent node first
          parent->send_node(rez, target);
          AutoLock n_lock(node_lock);
          for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
                semantic_info.begin(); it != semantic_info.end(); it++)
          {
            Serializer rez2;
            {
              RezCheck z(rez2);
              rez2.serialize(handle);
              rez2.serialize(initialized);
              rez2.serialize<size_t>(1);
              rez2.serialize(it->first);
              rez2.serialize(it->second.size);
              rez2.serialize(it->second.buffer, it->second.size);
              rez2.serialize(it->second.is_mutable);
            }
            context->runtime->send_logical_region_semantic_info(target, rez2);
          }
        }
        else
        {
          rez.serialize(handle);
          rez.serialize(did);
          rez.serialize(initialized);
          if (provenance != NULL)
            provenance->serialize(rez);
          else
            Provenance::serialize_null(rez);
          if (collective_mapping != NULL)
            collective_mapping->pack(rez);
          else
            CollectiveMapping::pack_null(rez);
          rez.serialize<size_t>(semantic_info.size());
          for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
                semantic_info.begin(); it != semantic_info.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second.size);
            rez.serialize(it->second.buffer, it->second.size);
            rez.serialize(it->second.is_mutable);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_node_creation(RegionTreeForest *context,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LogicalRegion handle;
      derez.deserialize(handle);
      DistributedID did;
      derez.deserialize(did);
      RtEvent initialized;
      derez.deserialize(initialized);
      AutoProvenance prov(Provenance::deserialize(derez));
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping *mapping = NULL;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);

      RegionNode *node = context->create_node(handle, NULL/*parent*/,
                                      initialized, did, prov, mapping);
#ifdef DEBUG_LEGION
      assert(node != NULL);
#endif
      size_t num_semantic;
      derez.deserialize(num_semantic);
      if (num_semantic > 0)
      {
        NodeSet source_mask;
        source_mask.add(source);
        source_mask.add(context->runtime->address_space);
        for (unsigned idx = 0; idx < num_semantic; idx++)
        {
          SemanticTag tag;
          derez.deserialize(tag);
          size_t buffer_size;
          derez.deserialize(buffer_size);
          const void *buffer = derez.get_current_pointer();
          derez.advance_pointer(buffer_size);
          bool is_mutable;
          derez.deserialize(is_mutable);
          node->attach_semantic_information(tag, source, buffer, buffer_size,
                                            is_mutable, false/*local only*/);
        }
      }
    } 

    //--------------------------------------------------------------------------
    void RegionNode::perform_versioning_analysis(ContextID ctx,
                                                 InnerContext *parent_ctx,
                                                 VersionInfo *version_info,
                                                 const FieldMask &mask,
                                                 Operation *op, unsigned index,
                                                 unsigned parent_req_index,
                                                 std::set<RtEvent> &applied,
                                                 RtEvent *output_region_ready,
                                                 bool collective_rendezvous)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      manager.perform_versioning_analysis(parent_ctx, version_info, this, mask,
          op, index, parent_req_index, applied, output_region_ready,
          collective_rendezvous);
    }
    
    //--------------------------------------------------------------------------
    void RegionNode::find_open_complete_partitions(ContextID ctx,
               const FieldMask &mask, std::vector<LogicalPartition> &partitions)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      std::set<LogicalPartition> unique_partitions;
      for (LegionList<FieldState>::const_iterator sit = 
            state.field_states.begin(); sit != state.field_states.end(); sit++)
      {
        if (sit->valid_fields() * mask)
          continue;
        for (OrderedFieldMaskChildren::const_iterator it = 
              sit->open_children.begin(); it != sit->open_children.end(); it++)
        {
          if (it->second * mask)
            continue;
          PartitionNode *child = it->first->as_partition_node();
          if (child->is_complete())
            unique_partitions.insert(child->handle);
        }
      }
      partitions.insert(partitions.end(), 
                        unique_partitions.begin(), unique_partitions.end());
    }

    //--------------------------------------------------------------------------
    void RegionNode::send_semantic_request(AddressSpaceID target,
             SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      context->runtime->send_logical_region_semantic_request(target, rez);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID RegionNode::find_semantic_owner(void) const
    //--------------------------------------------------------------------------
    {
      // If we're the root, then the owner is the owner of the root
      // Otherwise the owner is the owner of the corresponding index space
      if (parent == NULL)
        return owner_space;
      else
        return row_source->owner_space;
    }

    //--------------------------------------------------------------------------
    void RegionNode::send_semantic_info(AddressSpaceID target,
                                        SemanticTag tag,
                                        const void *buffer, size_t size, 
                                        bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      context->runtime->send_logical_region_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void RegionNode::process_semantic_request(SemanticTag tag,
       AddressSpaceID source, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(find_semantic_owner() == context->runtime->address_space);
#endif
      RtEvent precondition;
      void *result = NULL;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          context->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_semantic_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      RegionNode *node = forest->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_semantic_info(RegionTreeForest *forest,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      forest->attach_semantic_information(handle, tag, source, buffer, size, 
                                          is_mutable, false/*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_top_level_request(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tid;
      derez.deserialize(tid);
      RegionNode *node = forest->get_tree(tid, true/*can fail*/);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      AddressSpaceID source;
      derez.deserialize(source);
      if (node != NULL)
      {
        // If there is a collective mapping, check to see if we're on the
        // right node and if not forward it on to the right node
        if (node->collective_mapping != NULL)
        {
#ifdef DEBUG_LEGION
          assert(!node->collective_mapping->contains(source));
          assert(node->collective_mapping->contains(node->local_space));
#endif
          if (node->is_owner())
          {
            const AddressSpaceID nearest = 
              node->collective_mapping->find_nearest(source);
            // If we're not the nearest then forward it on to the
            // proper node to handle the request
            if (nearest != node->local_space)
            {
              Serializer rez;
              rez.serialize(tid);
              rez.serialize(done_event);
              rez.serialize(source);
              forest->runtime->send_top_level_region_request(nearest, rez);
              return;
            }
          }
#ifdef DEBUG_LEGION
          else
          {
            assert(node->local_space == 
                node->collective_mapping->find_nearest(source));
          }
#endif
        }
        Serializer rez;
        {
          RezCheck z(rez);
          node->send_node(rez, source);
          rez.serialize(done_event);
        }
        forest->runtime->send_top_level_region_return(source, rez);
      }
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_top_level_return(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      handle_node_creation(forest, derez, source);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_logical_context(ContextID ctx, 
                                           TreeStateLogger *logger,
                                           const FieldMask &capture_mask) 
    //--------------------------------------------------------------------------
    {
      logger->log("==========");
      print_context_header(logger);
      logger->down();
      FieldMaskSet<PartitionNode> to_traverse;
      if (logical_states.has_entry(ctx))
      {
        LogicalState &state = get_logical_state(ctx);
        print_logical_state(state, capture_mask, to_traverse, logger);  
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (FieldMaskSet<PartitionNode>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
          it->first->print_logical_context(ctx, logger, it->second);
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_context_header(TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      DomainPoint color = row_source->get_domain_point_color();
      switch (color.get_dim())
      {
        case 1:
          {
            logger->log("Region Node (%x,%d,%d) Color %d at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], get_depth());
            break;
          }
#if LEGION_MAX_DIM >= 2
        case 2:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d) at "
                        "depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], get_depth());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 3
        case 3:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d) at "
                        "depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], get_depth());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 4
        case 4:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d) at "
                        "depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], get_depth());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 5
        case 5:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d) "
                        "at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], get_depth());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 6
        case 6:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d,%d) "
                        "at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], get_depth());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 7
        case 7:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d,%d,%d) "
                        "at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], color[6], get_depth());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 8
        case 8:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d,%d,%d,%d)"
                        " at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], color[6], color[7], get_depth());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 9
        case 9:
          {
            logger->log("Region Node (%x,%d,%d) Color "
                        "(%d,%d,%d,%d,%d,%d,%d,%d) at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], color[6], color[7], color[8], get_depth());
            break;
          }
#endif
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_logical_state(LogicalState &state,
                                         const FieldMask &capture_mask,
                                       FieldMaskSet<PartitionNode> &to_traverse,
                                         TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      // Open Field States 
      {
        logger->log("Open Field States (%ld)", state.field_states.size());
        logger->down();
        for (std::list<FieldState>::const_iterator it = 
              state.field_states.begin(); it != 
              state.field_states.end(); it++)
        {
          it->print_state(logger, capture_mask, this);
          if (it->valid_fields() * capture_mask)
            continue;
          for (OrderedFieldMaskChildren::const_iterator cit = 
                it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
          {
            FieldMask overlap = cit->second & capture_mask;
            if (!overlap)
              continue;
            to_traverse.insert(cit->first->as_partition_node(), overlap);
          }
        }
        logger->up();
      }
    }
    
#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void RegionNode::dump_logical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      DomainPoint color = row_source->get_domain_point_color();
      switch (color.get_dim())
      {
        case 1:
          {
            logger->log("Region Node (%x,%d,%d) Color %d at depth %d (%p)",
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], logger->get_depth(), this);
            break;
          }
#if LEGION_MAX_DIM >= 2
        case 2:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d) at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], logger->get_depth(), this);
            break;
          }
#endif
#if LEGION_MAX_DIM >= 3
        case 3:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d) at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], logger->get_depth(), this);
            break;
          }
#endif
#if LEGION_MAX_DIM >= 4
        case 4:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d) at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], logger->get_depth(),this);
            break;
          }
#endif
#if LEGION_MAX_DIM >= 5
        case 5:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d) "
                        "at depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              logger->get_depth(), this);
            break;
          }
#endif
#if LEGION_MAX_DIM >= 6
        case 6:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d,%d) "
                        "at depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], logger->get_depth(), this);
            break;
          }
#endif
#if LEGION_MAX_DIM >= 7
        case 7:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d,%d,%d) "
                        "at depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], color[6], logger->get_depth(), this);
            break;
          }
#endif
#if LEGION_MAX_DIM >= 8
        case 8:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d,%d,%d,%d,%d,%d)"
                        " at depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], color[6], color[7], logger->get_depth(), this);
            break;
          }
#endif
#if LEGION_MAX_DIM >= 9
        case 9:
          {
            logger->log("Region Node (%x,%d,%d) Color "
                        "(%d,%d,%d,%d,%d,%d,%d,%d) at depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[2], color[2], color[3], color[4], 
              color[5], color[6], color[7], color[8], 
              logger->get_depth(), this);
            break;
          }
#endif
        default:
          assert(false);
      }
      logger->down();
      FieldMaskSet<PartitionNode> to_traverse;
      if (logical_states.has_entry(ctx))
        print_logical_state(get_logical_state(ctx), capture_mask,
                            to_traverse, logger);
      else
        logger->log("No state");
      logger->log("");
      if (!to_traverse.empty())
      {
        for (FieldMaskSet<PartitionNode>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
          it->first->dump_logical_context(ctx, logger, it->second);
      }
      logger->up();
    }
#endif

    /////////////////////////////////////////////////////////////
    // Partition Tracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionTracker::PartitionTracker(PartitionNode *part)
      : Collectable(2/*expecting two reference calls*/), partition(part)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PartitionTracker::PartitionTracker(const PartitionTracker &rhs)
      : Collectable(), partition(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PartitionTracker& PartitionTracker::operator=(const PartitionTracker &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PartitionTracker::can_prune(void)
    //--------------------------------------------------------------------------
    {
      const unsigned remainder = references.load(); 
#ifdef DEBUG_LEGION
      assert((remainder == 1) || (remainder == 2));
#endif
      return (remainder == 1);
    }

    //--------------------------------------------------------------------------
    bool PartitionTracker::remove_partition_reference()
    //--------------------------------------------------------------------------
    {
      // Pull a copy of this on to the stack in case we get deleted
      std::atomic<PartitionNode*> node(partition);
      const bool last = remove_reference();
      // If we weren't the last one that means we remove the reference
      if (!last && node.load()->remove_base_gc_ref(REGION_TREE_REF))
        delete node.load();
      return last;
    }

    /////////////////////////////////////////////////////////////
    // Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(LogicalPartition p, RegionNode *par,
                                 IndexPartNode *row_src, 
                                 FieldSpaceNode *col_src, RegionTreeForest *ctx,
                                 RtEvent init, RtEvent tree)
      : RegionTreeNode(ctx, col_src, init, tree, par->provenance), handle(p), 
        parent(par), row_source(row_src)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Partition %lld %d %d %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, 
          handle.get_index_partition().get_id(), 
          handle.get_field_space().get_id(),
          handle.get_tree_id());
#endif
    }

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(const PartitionNode &rhs)
      : RegionTreeNode(NULL, NULL, RtEvent::NO_RT_EVENT, RtEvent::NO_RT_EVENT),
        handle(LogicalPartition::NO_PART), parent(NULL), row_source(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PartitionNode::~PartitionNode(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<LegionColor,RegionNode*>::const_iterator it =   
            color_map.begin(); it != color_map.end(); it++)
        if (it->second->remove_nested_resource_ref(did))
          delete it->second;
      if (registered)
      {
        if (parent->remove_nested_resource_ref(did))
          delete parent;
        // Unregister ourselves with our row source
        if (row_source->remove_nested_resource_ref(did))
          delete row_source;
        // Then unregister ourselves with the context
        context->remove_node(handle);
      }
    }

    //--------------------------------------------------------------------------
    PartitionNode& PartitionNode::operator=(const PartitionNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      parent->remove_child(row_source->color);
      row_source->remove_nested_gc_ref(did);
      // Remove gc references on all of our child nodes
      // We should not need a lock at this point since nobody else should
      // be modifying these data structures at this point
      // No need to check for deletion since we hold resource references
      for (std::map<LegionColor,RegionNode*>::const_iterator it = 
            color_map.begin(); it != color_map.end(); it++)
        it->second->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!registered);
#endif
      row_source->add_nested_resource_ref(did);
      row_source->add_nested_gc_ref(did);
      parent->add_nested_resource_ref(did);
      parent->add_child(this);
      // Create a partition deletion tracker for this node and add it to 
      // both the index partition node and the root logical region for this
      // tree so that we can have our reference removed once either is deleted
      PartitionTracker *tracker = new PartitionTracker(this);
      row_source->add_tracker(tracker); 
      RegionNode *root = parent;
      while (root->parent != NULL)
        root = root->parent->parent;
      root->add_tracker(tracker);
      registered = true;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Ask the row source because it eagerly instantiates
      return row_source->has_color(c);
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::get_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // check to see if we have it, if not try to make it
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<LegionColor,RegionNode*>::const_iterator finder = 
          color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      // If we get here we didn't immediately have it so try
      // to make it through the proper channels
      IndexSpaceNode *index_node = row_source->get_child(c, NULL);
#ifdef DEBUG_LEGION
      assert(index_node != NULL);
#endif
      LogicalRegion reg_handle(handle.tree_id, index_node->handle,
                               handle.field_space);
      return context->create_node(reg_handle, this, 
                                  RtEvent::NO_RT_EVENT, 0/*did*/);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_child(RegionNode *child)
    //--------------------------------------------------------------------------
    {
      child->add_nested_resource_ref(did);
      child->add_nested_gc_ref(did);
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(is_global());
      assert(color_map.find(child->row_source->color) == color_map.end());
#endif
      color_map[child->row_source->color] = child;
    }

    //--------------------------------------------------------------------------
    unsigned PartitionNode::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->depth;
    }

    //--------------------------------------------------------------------------
    LegionColor PartitionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* PartitionNode::get_row_source(void) const
    //--------------------------------------------------------------------------
    {
      return row_source;
    }

    //--------------------------------------------------------------------------
    RegionTreeID PartitionNode::get_tree_id(void) const
    //--------------------------------------------------------------------------
    {
      return handle.get_tree_id();
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_tree_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_children_disjoint(const LegionColor c1, 
                                              const LegionColor c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_all_children_disjoint(void)
    //--------------------------------------------------------------------------
    {
      return row_source->is_disjoint();
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::is_region(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::as_region_node(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    PartitionNode* PartitionNode::as_partition_node(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<PartitionNode*>(this);
    }
#endif

    //--------------------------------------------------------------------------
    bool PartitionNode::visit_node(PathTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      return traverser->visit_partition(this);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::visit_node(NodeTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal = traverser->visit_partition(this);
      if (continue_traversal)
      {
        const bool break_early = traverser->break_early();
        if (traverser->force_instantiation)
        {
          for (ColorSpaceIterator itr(row_source); itr; itr++)
          {
            bool result = get_child(*itr)->visit_node(traverser);
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
              break;
          }
        }
        else
        {
          std::map<LegionColor,RegionNode*> children;
          // Need to hold the lock when reading from 
          // the color map or the valid map
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            children = color_map;
          }
          for (std::map<LegionColor,RegionNode*>::const_iterator it = 
                children.begin(); it != children.end(); it++)
          {
            bool result = it->second->visit_node(traverser);
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
              break;
          }
        }
      }
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::is_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent != NULL);
#endif
      return row_source->is_complete();
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::intersects_with(RegionTreeNode *other, bool compute)
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_region())
        return row_source->intersects_with(
                    other->as_region_node()->row_source, compute);
      else
        return row_source->intersects_with(
                    other->as_partition_node()->row_source, compute);
    }

    //--------------------------------------------------------------------------
    size_t PartitionNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_num_children();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::send_node(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      {
        AutoLock n_lock(node_lock); 
        if (!has_remote_instance(target))
        {
          continue_up = true;
          update_remote_instances(target);
        }
      }
      if (continue_up)
      {
#ifdef DEBUG_LEGION
        assert(parent != NULL);
#endif
        // Send the parent node first
        parent->send_node(rez, target);
        AutoLock n_lock(node_lock);
        for (LegionMap<SemanticTag,SemanticInfo>::iterator it = 
              semantic_info.begin(); it != semantic_info.end(); it++)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(it->first);
            rez.serialize(it->second.size);
            rez.serialize(it->second.buffer, it->second.size);
            rez.serialize(it->second.is_mutable);
          }
          context->runtime->send_logical_partition_semantic_info(target, rez);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::send_semantic_request(AddressSpaceID target,
             SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      context->runtime->send_logical_partition_semantic_request(target, rez);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID PartitionNode::find_semantic_owner(void) const
    //--------------------------------------------------------------------------
    {
      // The owner is the owner of our row source partition
      return row_source->owner_space;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::send_semantic_info(AddressSpaceID target,
                                           SemanticTag tag,
                                           const void *buffer, size_t size,
                                           bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      context->runtime->send_logical_partition_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::process_semantic_request(SemanticTag tag,
       AddressSpaceID source, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(find_semantic_owner() == context->runtime->address_space);
#endif
      RtEvent precondition;
      void *result = NULL;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        LegionMap<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          context->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PartitionNode::handle_semantic_request(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      PartitionNode *node = forest->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PartitionNode::handle_semantic_info(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      forest->attach_semantic_information(handle, tag, source, buffer, size, 
                                          is_mutable, false/*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_logical_context(ContextID ctx,
                                              TreeStateLogger *logger,
                                              const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("==========");
      print_context_header(logger);
      logger->down();
      FieldMaskSet<RegionNode> to_traverse;
      if (logical_states.has_entry(ctx))
      {
        LogicalState &state = get_logical_state(ctx);
        print_logical_state(state, capture_mask, to_traverse, logger);    
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (FieldMaskSet<RegionNode>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
          it->first->print_logical_context(ctx, logger, it->second);
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_context_header(TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      const char* disjointness =
        row_source->is_disjoint() ? "disjoint" : "aliased";
      logger->log("Partition Node (" IDFMT ",%d,%d) Color %d "
          "%s at depth %d", 
          handle.index_partition.id, handle.field_space.id,handle.tree_id,
          row_source->color, disjointness, get_depth());
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_logical_state(LogicalState &state,
                                        const FieldMask &capture_mask,
                                        FieldMaskSet<RegionNode> &to_traverse,
                                        TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      // Open Field States
      {
        logger->log("Open Field States (%ld)", state.field_states.size()); 
        logger->down();
        for (std::list<FieldState>::const_iterator it = 
              state.field_states.begin(); it != 
              state.field_states.end(); it++)
        {
          it->print_state(logger, capture_mask, this);
          if (it->valid_fields() * capture_mask)
            continue;
          for (OrderedFieldMaskChildren::const_iterator cit =
                it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
          {
            FieldMask overlap = cit->second & capture_mask;
            if (!overlap)
              continue;
            to_traverse.insert(cit->first->as_region_node(), overlap);
          }
        }
        logger->up();
      }
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void PartitionNode::dump_logical_context(ContextID ctx,
                                             TreeStateLogger *logger,
                                             const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      print_context_header(logger);
      logger->down();
      FieldMaskSet<RegionNode> to_traverse;
      if (logical_states.has_entry(ctx))
      {
        LogicalState &state = get_logical_state(ctx);
        print_logical_state(state, capture_mask, to_traverse, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (FieldMaskSet<RegionNode>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
          it->first->dump_logical_context(ctx, logger, it->second);
      }
      logger->up();
    }
#endif 

    /* static */
    IndexSpaceOperation *
    InternalExpressionCreator::create_with_domain(TypeTag tag,
                                                 const Domain &dom)
    {
      InternalExpressionCreator creator(tag, dom, implicit_runtime->forest);
      creator.create_operation();

      IndexSpaceOperation *out = creator.result;
      out->add_base_expression_reference(LIVE_EXPR_REF);
      ImplicitReferenceTracker::record_live_expression(out);

      return out;
    }

  }; // namespace Internal 
}; // namespace Legion

// EOF

