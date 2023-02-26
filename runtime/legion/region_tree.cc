/* Copyright 2023 Stanford University, NVIDIA Corporation
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
      domain = is->get_domain(domain_ready, true/*tight*/);
#ifdef LEGION_SPY
      index_space = req.region.get_index_space();
#endif
      FieldSpaceNode *fs = forest->get_node(req.region.get_field_space());
      std::vector<unsigned> field_indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, field_indexes);
      instances.resize(field_indexes.size());
      Runtime *runtime = forest->runtime;
      if ((runtime->num_profiling_nodes > 0) || runtime->legion_spy_enabled)
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
    RtEvent RegionTreeForest::create_pending_partition(TaskContext *ctx,
                                                       IndexPartition pid,
                                                       IndexSpace parent,
                                                       IndexSpace color_space,
                                                    LegionColor partition_color,
                                                       PartitionKind part_kind,
                                                       DistributedID did,
                                                       Provenance *provenance,
                                                       ApEvent partition_ready,
                                                     ApBarrier partial_pending)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (partial_pending.exists())
        assert(partition_ready == partial_pending);
#endif
      IndexSpaceNode *parent_node = get_node(parent);
      IndexSpaceNode *color_node = get_node(color_space);
      if (partition_color == INVALID_COLOR)
        partition_color = parent_node->generate_color();
      // If we are making this partition on a different node than the
      // owner node of the parent index space then we have to tell that
      // owner node about the existence of this partition
      RtEvent parent_notified;
      const AddressSpaceID parent_owner = parent_node->get_owner_space();
      if (parent_owner != runtime->address_space)
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
        RtUserEvent disjointness_event = Runtime::create_rt_user_event();
        // Use 1 if we know it's complete, 0 if it's not, 
        // otherwise -1 since we don't know
        const int complete = (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ? 1 :
                         (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND) ? 0 : -1;
        IndexPartNode *node = create_node(pid, parent_node, color_node, 
            partition_color, disjointness_event, complete, did, provenance,
            partition_ready, partial_pending, RtEvent::NO_RT_EVENT); 
        // Get a reference for the node to hold until disjointness is computed
        node->add_base_valid_ref(APPLICATION_REF);
        IndexPartNode::DisjointnessArgs args(pid, NULL, true/*owner*/);
        Runtime::trigger_event(disjointness_event,
            runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_DEFERRED_PRIORITY,
              Runtime::protect_event(partition_ready)));
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
        create_node(pid, parent_node, color_node, partition_color, disjoint,
                    complete, did, provenance, partition_ready, partial_pending,
                    RtEvent::NO_RT_EVENT);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_index_partition(parent.id, pid.id, disjoint ? 1 : 0,
              complete, partition_color, runtime->address_space,
              (provenance == NULL) ? NULL : provenance->human_str());
	if (runtime->profiler != NULL)
	  runtime->profiler->record_index_partition(parent.id,pid.id, disjoint,
						    partition_color);
      }
      ctx->register_index_partition_creation(pid);
      return parent_notified;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_pending_cross_product(TaskContext *ctx,
                                                 IndexPartition handle1,
                                                 IndexPartition handle2,
                             std::map<IndexSpace,IndexPartition> &user_handles,
                                                 PartitionKind kind,
                                                 Provenance *provenance,
                                                 LegionColor &part_color,
                                                 ApEvent domain_ready,
                                                 std::set<RtEvent> &safe_events,
                                                 ShardID shard,
                                                 size_t total_shards)
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
            (kind == LEGION_COMPUTE_KIND)) && source->dominates(base)) 
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
          if (base->total_children == base->max_linearized_color)
          {
            for (LegionColor color = 0; color < base->total_children; color++)
            {
              IndexSpaceNode *child_node = base->get_child(color);
              children_nodes.push_back(child_node);
              std::vector<LegionColor> colors;
              LegionColor bound = child_node->get_colors(colors);
              if (!colors.empty())
                existing_colors.insert(colors.begin(), colors.end());
              if (bound > lower_bound)
                lower_bound = bound;
            }
          }
          else
          {
            ColorSpaceIterator *itr =
              base->color_space->create_color_space_iterator();
            while (itr->is_valid())
            {
              const LegionColor color = itr->yield_color();
              IndexSpaceNode *child_node = base->get_child(color);
              children_nodes.push_back(child_node);
              std::vector<LegionColor> colors;
              LegionColor bound = child_node->get_colors(colors);
              if (!colors.empty())
                existing_colors.insert(colors.begin(), colors.end());
              if (bound > lower_bound)
                lower_bound = bound;
            }
            delete itr;
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
      // Iterate over all our sub-regions and generate partitions
      if (!children_nodes.empty())
      {
        for (unsigned idx = shard; 
              idx < children_nodes.size(); idx += total_shards)
        {
          IndexSpaceNode *child_node = children_nodes[idx];
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag()); 
          DistributedID did = 
            runtime->get_available_distributed_id();
          const RtEvent safe =
            create_pending_partition(ctx, pid, child_node->handle, 
                                     source->color_space->handle, 
                                     part_color, kind, did, 
                                     provenance, domain_ready); 
          // If the user requested the handle for this point return it
          user_handles[child_node->handle] = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
      }
      else if (base->total_children == base->max_linearized_color)
      {
        for (LegionColor color = shard; 
              color < base->total_children; color += total_shards)
        {
          IndexSpaceNode *child_node = base->get_child(color);
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag()); 
          DistributedID did = 
            runtime->get_available_distributed_id();
          const RtEvent safe =
            create_pending_partition(ctx, pid, child_node->handle, 
                                     source->color_space->handle, 
                                     part_color, kind, did, 
                                     provenance, domain_ready); 
          // If the user requested the handle for this point return it
          user_handles[child_node->handle] = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          base->color_space->create_color_space_iterator();
        // Skip ahead if necessary for our shard
        for (unsigned idx = 0; idx < shard; idx++)
        {
          itr->yield_color();
          if (!itr->is_valid())
            break;
        }
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNode *child_node = base->get_child(color);
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag()); 
          DistributedID did = 
            runtime->get_available_distributed_id();
          const RtEvent safe = 
            create_pending_partition(ctx, pid, child_node->handle, 
                                     source->color_space->handle, 
                                     part_color, kind, did, 
                                     provenance, domain_ready);
          // If the user requested the handle for this point return it
          user_handles[child_node->handle] = pid;
          // Skip ahead for the next color if necessary
          for (unsigned idx = 0; idx < (total_shards-1); idx++)
          {
            itr->yield_color();
            if (!itr->is_valid())
              break;
          }
          if (safe.exists())
            safe_events.insert(safe);
        }
        delete itr;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::create_pending_partition_shard(
                                              ShardID owner_shard,
                                              ReplicateContext *ctx,
                                              IndexPartition pid,
                                              IndexSpace parent,
                                              IndexSpace color_space,
                                              LegionColor &partition_color,
                                              PartitionKind part_kind,
                                              DistributedID did,
                                              Provenance *provenance,
                                              ValueBroadcast<bool> *part_result,
                                              ApEvent partition_ready,
                                              CollectiveMapping *mapping,
                                              ShardMapping *shard_mapping,
                                              RtEvent creation_ready,
                                              ApBarrier partial_pending)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (partial_pending.exists())
        assert(partition_ready == partial_pending);
#endif
      if (owner_shard == ctx->owner_shard->shard_id)
      {
        // We're the owner so we do most of the work
        IndexSpaceNode *parent_node = get_node(parent);
        IndexSpaceNode *color_node = get_node(color_space);
        if (partition_color == INVALID_COLOR)
          partition_color = parent_node->generate_color();
        // If we are making this partition on a different node than the
        // owner node of the parent index space then we have to tell that
        // owner node about the existence of this partition
        RtEvent parent_notified;
        const AddressSpaceID parent_owner = parent_node->get_owner_space();
        if (parent_owner != runtime->address_space)
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
        RtUserEvent disjointness_event;
        if ((part_kind == LEGION_COMPUTE_KIND) || 
            (part_kind == LEGION_COMPUTE_COMPLETE_KIND) || 
            (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
        {
#ifdef DEBUG_LEGION
          assert(part_result != NULL);
#endif
          disjointness_event = Runtime::create_rt_user_event(); 
        }
#ifdef DEBUG_LEGION
        else
          assert(part_result == NULL);
#endif
        IndexPartNode *part_node;
        if ((part_kind != LEGION_COMPUTE_KIND) && 
            (part_kind != LEGION_COMPUTE_COMPLETE_KIND) && 
            (part_kind != LEGION_COMPUTE_INCOMPLETE_KIND))
        {
          const bool disjoint = (part_kind == LEGION_DISJOINT_KIND) || 
                                (part_kind == LEGION_DISJOINT_COMPLETE_KIND) ||
                                (part_kind == LEGION_DISJOINT_INCOMPLETE_KIND);
          // Use 1 if we know it's complete, 0 if it's not, 
          // otherwise -1 since we don't know
          const int complete = ((part_kind == LEGION_DISJOINT_COMPLETE_KIND) ||
                              (part_kind == LEGION_ALIASED_COMPLETE_KIND)) ? 1 :
                             ((part_kind == LEGION_DISJOINT_INCOMPLETE_KIND) ||
                          (part_kind == LEGION_ALIASED_INCOMPLETE_KIND)) ? 0 :-1;
          part_node = create_node(pid, parent_node, color_node, partition_color,
            disjoint, complete, did, provenance, partition_ready,
            partial_pending, creation_ready, mapping, shard_mapping);
          if (runtime->legion_spy_enabled)
            LegionSpy::log_index_partition(parent.id, pid.id, disjoint ? 1 : 0,
                complete, partition_color, runtime->address_space,
                (provenance == NULL) ? NULL : provenance->human_str());
          if (runtime->profiler != NULL)
	    runtime->profiler->record_index_partition(parent.id,pid.id,disjoint,
                                                      partition_color);
        }
        else
        {
          // Use 1 if we know it's complete, 0 if it's not, 
          // otherwise -1 since we don't know
          const int complete = (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ? 1 :
                         (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND) ? 0 : -1;
          part_node = create_node(pid, parent_node, color_node, partition_color,
                                  disjointness_event, complete, did, provenance,
                                  partition_ready, partial_pending,
                                  creation_ready, mapping, 
                                  shard_mapping);
          if (runtime->legion_spy_enabled)
            LegionSpy::log_index_partition(parent.id, pid.id, -1/*unknown*/,
                complete, partition_color, runtime->address_space,
                (provenance == NULL) ? NULL : provenance->human_str());
        }
        if (disjointness_event.exists())
        {
          // Hold a reference on the node until disjointness is performed
          part_node->add_base_valid_ref(APPLICATION_REF);
          IndexPartNode::DisjointnessArgs args(pid, part_result, true/*owner*/);
          // Don't do the disjointness test until all the partition
          // is ready and has been created on all the nodes
          Runtime::trigger_event(disjointness_event,
              runtime->issue_runtime_meta_task(args,
                LG_THROUGHPUT_DEFERRED_PRIORITY,
                Runtime::merge_events(creation_ready,
                  Runtime::protect_event(partition_ready))));
        }
        ctx->register_index_partition_creation(pid);
        return parent_notified;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(partition_color != INVALID_COLOR);
#endif
        // We're not the owner so we just do basic setup work
        IndexSpaceNode *parent_node = get_node(parent);
        IndexSpaceNode *color_node = get_node(color_space);
        RtUserEvent disjointness_event;
        if ((part_kind == LEGION_COMPUTE_KIND) || 
            (part_kind == LEGION_COMPUTE_COMPLETE_KIND) || 
            (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
        {
#ifdef DEBUG_LEGION
          assert(part_result != NULL);
#endif
          disjointness_event = Runtime::create_rt_user_event(); 
        }
#ifdef DEBUG_LEGION
        else
          assert(part_result == NULL);
#endif
        IndexPartNode *part_node;
        if ((part_kind != LEGION_COMPUTE_KIND) && 
            (part_kind != LEGION_COMPUTE_COMPLETE_KIND) && 
            (part_kind != LEGION_COMPUTE_INCOMPLETE_KIND))
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
          part_node = create_node(pid, parent_node, color_node, partition_color,
            disjoint, complete, did, provenance, partition_ready,
            partial_pending, creation_ready, mapping, shard_mapping);
        }
        else
        {
          // Use 1 if we know it's complete, 0 if it's not, 
          // otherwise -1 since we don't know
          const int complete = (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ? 1 :
                         (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND) ? 0 : -1;
          part_node = create_node(pid, parent_node, color_node, partition_color,
                                  disjointness_event, complete, did, provenance,
                                  partition_ready, partial_pending,
                                  creation_ready, mapping, shard_mapping);
        }
        if (disjointness_event.exists())
        {
          // Hold a reference on the node until disjointness is performed
          part_node->add_base_resource_ref(APPLICATION_REF);
          IndexPartNode::DisjointnessArgs args(pid, part_result,false/*owner*/);
          // We only need to wait for the creation to be ready 
          // if we're not the owner
          Runtime::trigger_event(disjointness_event,
              runtime->issue_runtime_meta_task(args,
                LG_THROUGHPUT_DEFERRED_PRIORITY, creation_ready));
        }
        ctx->register_index_partition_creation(pid);
        // We know the parent is notified or we wouldn't even have
        // been given our pid
        return RtEvent::NO_RT_EVENT;
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
                                                     size_t granularity,
                                                     ShardID shard,
                                                     size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->create_equal_children(op, granularity, 
                                             shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_weights(Operation *op,
                                                       IndexPartition pid,
                                                       const FutureMap &weights,
                                                       size_t granularity,
                                                       ShardID shard,
                                                       size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->create_by_weights(op, weights, granularity, 
                                         shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_union(Operation *op,
                                                        IndexPartition pid,
                                                        IndexPartition handle1,
                                                        IndexPartition handle2,
                                                        ShardID shard,
                                                        size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node1 = get_node(handle1);
      IndexPartNode *node2 = get_node(handle2);
      return new_part->create_by_union(op, node1, node2, shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_intersection(Operation *op,
                                                         IndexPartition pid,
                                                         IndexPartition handle1,
                                                         IndexPartition handle2,
                                                         ShardID shard,
                                                         size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node1 = get_node(handle1);
      IndexPartNode *node2 = get_node(handle2);
      return new_part->create_by_intersection(op, node1, node2, 
                                              shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_intersection(Operation *op,
                                                           IndexPartition pid,
                                                           IndexPartition part,
                                                           const bool dominates,
                                                           ShardID shard,
                                                           size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node = get_node(part);
      return new_part->create_by_intersection(op, node, dominates,
                                              shard, total_shards); 
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_difference(Operation *op,
                                                       IndexPartition pid,
                                                       IndexPartition handle1,
                                                       IndexPartition handle2,
                                                       ShardID shard,
                                                       size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      IndexPartNode *node1 = get_node(handle1);
      IndexPartNode *node2 = get_node(handle2);
      return new_part->create_by_difference(op, node1, node2, 
                                            shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_restriction(
                                                        IndexPartition pid,
                                                        const void *transform,
                                                        const void *extent,
                                                        ShardID shard,
                                                        size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->create_by_restriction(transform, extent, 
                                             shard, total_shards); 
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_domain(Operation *op,
                                                    IndexPartition pid,
                                                    const FutureMap &future_map,
                                                    bool perform_intersections,
                                                    ShardID shard,
                                                    size_t total_shards) 
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->parent->create_by_domain(op, new_part, future_map.impl, 
                                  perform_intersections, shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_cross_product_partitions(Operation *op,
                                                         IndexPartition base,
                                                         IndexPartition source,
                                                         LegionColor part_color,
                                                         ShardID shard,
                                                         size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *base_node = get_node(base);
      IndexPartNode *source_node = get_node(source);
      std::set<ApEvent> ready_events;
      if (base_node->total_children == base_node->max_linearized_color)
      {
        for (LegionColor color = shard; 
              color < base_node->total_children; color+=total_shards)
        {
          IndexSpaceNode *child_node = base_node->get_child(color);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready = 
            child_node->create_by_intersection(op, part_node, source_node);
          ready_events.insert(ready);
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          base_node->color_space->create_color_space_iterator();
        // Skip ahead if necessary for our shard
        for (unsigned idx = 0; idx < shard; idx++)
        {
          itr->yield_color();
          if (!itr->is_valid())
            break;
        }
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNode *child_node = base_node->get_child(color);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready =
            child_node->create_by_intersection(op, part_node, source_node);
          ready_events.insert(ready);
          // Skip ahead for the next color if necessary
          for (unsigned idx = 0; idx < (total_shards-1); idx++)
          {
            itr->yield_color();
            if (!itr->is_valid())
              break;
          }
        }
        delete itr;
      }
      return Runtime::merge_events(NULL, ready_events);
    } 

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_field(Operation *op,
                                                        IndexPartition pending,
                             const std::vector<FieldDataDescriptor> &instances,
                                                        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      return partition->parent->create_by_field(op, partition, 
                                                instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_image(Operation *op,
                                                        IndexPartition pending,
                                                        IndexPartition proj,
                              const std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready,
                                                        ShardID shard,
                                                        size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_image(op, partition, projection,
                          instances, instances_ready, shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_image_range(Operation *op,
                                                      IndexPartition pending,
                                                      IndexPartition proj,
                              const std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready,
                                                      ShardID shard,
                                                      size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_image_range(op, partition, projection,
                               instances, instances_ready, shard, total_shards);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_preimage(Operation *op,
                                                      IndexPartition pending,
                                                      IndexPartition proj,
                              const std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_preimage(op, partition, projection,
                                           instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_preimage_range(Operation *op,
                                                      IndexPartition pending,
                                                      IndexPartition proj,
                              const std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_preimage_range(op, partition, 
                                projection, instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_association(Operation *op,
                                                 IndexSpace dom, IndexSpace ran,
                              const std::vector<FieldDataDescriptor> &instances,
                                                 ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *domain = get_node(dom);
      IndexSpaceNode *range = get_node(ran);
      return domain->create_association(op, range, instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_coordinate_size(IndexSpace handle, bool range)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->get_coordinate_size(range);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op, 
       IndexSpace target, const std::vector<IndexSpace> &handles, bool is_union,
       ShardID shard, size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      // Check to see if we own this child or not
      if ((total_shards > 1) && ((child_node->color % total_shards) != shard))
        return ApEvent::NO_AP_EVENT;
      // Convert the ap event for the space into an ap user event and 
      // trigger it once the operation is complete
      ApUserEvent space_ready = *(static_cast<ApUserEvent*>(
                         const_cast<ApEvent*>(&child_node->index_space_ready)));
      if (space_ready.has_triggered())
        REPORT_LEGION_ERROR(ERROR_INVALID_PENDING_CHILD,
          "Invalid pending child!")
      Runtime::trigger_event(NULL, space_ready, op->get_completion_event());
      return child_node->compute_pending_space(op, handles, is_union);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op, 
                        IndexSpace target, IndexPartition handle, bool is_union,
                        ShardID shard, size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      // Check to see if we own this child or not
      if ((total_shards > 1) && ((child_node->color % total_shards) != shard))
        return ApEvent::NO_AP_EVENT;
      // Convert the ap event for the space into an ap user event and 
      // trigger it once the operation is complete
      ApUserEvent space_ready = *(static_cast<ApUserEvent*>(
                         const_cast<ApEvent*>(&child_node->index_space_ready)));
      if (space_ready.has_triggered())
        REPORT_LEGION_ERROR(ERROR_INVALID_PENDING_CHILD, 
                            "Invalid pending child!")
      Runtime::trigger_event(NULL, space_ready, op->get_completion_event());
      return child_node->compute_pending_space(op, handle, is_union);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op,
                                         IndexSpace target, IndexSpace initial,
                                         const std::vector<IndexSpace> &handles,
                                         ShardID shard, size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      // Check to see if we own this child or not
      if ((total_shards > 1) && ((child_node->color % total_shards) != shard))
        return ApEvent::NO_AP_EVENT;
      // Convert the ap event for the space into an ap user event and 
      // trigger it once the operation is complete
      ApUserEvent space_ready = *(static_cast<ApUserEvent*>(
                         const_cast<ApEvent*>(&child_node->index_space_ready)));
      if (space_ready.has_triggered())
        REPORT_LEGION_ERROR(ERROR_INVALID_PENDING_CHILD,
                            "Invalid pending child!\n")
      Runtime::trigger_event(NULL, space_ready, op->get_completion_event());
      return child_node->compute_pending_difference(op, initial, handles);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::set_pending_space_domain(IndexSpace target,
                                                    Domain domain,
                                                    AddressSpaceID source,
                                                    ShardID shard,
                                                    size_t total_shards)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);

      if ((total_shards > 1) && ((child_node->color % total_shards) != shard))
        return;

      if (child_node->set_domain(domain, source))
        assert(false);
      ApUserEvent space_ready = *(static_cast<ApUserEvent*>(
                         const_cast<ApEvent*>(&child_node->index_space_ready)));
      if (space_ready.has_triggered())
        REPORT_LEGION_ERROR(ERROR_INVALID_PENDING_CHILD,
                            "Invalid pending child!\n")
      Runtime::trigger_event(NULL, space_ready);
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
    void RegionTreeForest::find_launch_space_domain(IndexSpace handle,
                                                    Domain &launch_domain)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      node->get_launch_space_domain(launch_domain);
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
                                                    ShardMapping *shard_mapping,
                                                    RtEvent initialized)
    //--------------------------------------------------------------------------
    {
      return create_node(handle, did, initialized, provenance, 
                         mapping, shard_mapping);
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
      if (!has_node(handle))
        return;
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
      if (!has_node(handle))
        return;
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
        RegionNode::get_owner_space(handle, runtime);
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
                                        const ProjectionInfo &projection_info,
                                        const RegionTreePath &path,
                                        LogicalAnalysis &logical_analysis)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_LOGICAL_ANALYSIS_CALL);
      // If this is a NO_ACCESS, then we'll have no dependences so we're done
      if (IS_NO_ACCESS(req))
        return;
      InnerContext *context = op->get_context();
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      RegionNode *parent_node = get_node(req.parent);
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      // Then compute the logical user
      LogicalUser user(op, idx, RegionUsage(req), user_mask); 
      LogicalTraceInfo trace_info(op, idx, req); 
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), true/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Finally do the traversal, note that we don't need to hold the
      // context lock since the runtime guarantees that all dependence
      // analysis for a single context are performed in order
      {
        FieldMask unopened_mask = user_mask;
        FieldMask already_closed_mask;
        FieldMask written_disjoint_complete;
        FieldMask first_touch_refinement = user_mask;
        FieldMaskSet<RefinementOp> refinements;
        const bool check_for_unversioned = 
          !op->is_parent_nonexclusive_virtual_mapping(idx);
        parent_node->register_logical_user(ctx.get_id(), user, path, trace_info,
                     projection_info, unopened_mask, already_closed_mask, 
                     written_disjoint_complete, first_touch_refinement, 
                     refinements, logical_analysis,
                     true/*track disjoint complete below*/, 
                     check_for_unversioned);
#ifdef DEBUG_LEGION
        // should never flow out here unless we're not checking for versioning
        // we aren't checking when we've got an non-read-write virtual mapping
        // because in that case we are sharing equivalence sets with the
        // parent context and therefore are never permitted to do refinements
        assert(!written_disjoint_complete || !check_for_unversioned);
#endif
      }
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), false/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Once we are done we can clear out the list of recorded dependences
      op->clear_logical_records();
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::perform_deletion_analysis(DeletionOp *op,
                                                 unsigned idx,
                                                 RegionRequirement &req,
                                                 const RegionTreePath &path,
                                                 bool invalidate_tree)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_LOGICAL_ANALYSIS_CALL);
      TaskContext *context = op->get_context();
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      RegionNode *parent_node = get_node(req.parent);
#ifdef DEBUG_LEGION
      // We should always be deleting the root of the region tree
      assert(parent_node->parent == NULL);
#endif
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      // Then compute the logical user
      LogicalUser user(op, idx, RegionUsage(req), user_mask);
      LogicalTraceInfo trace_info(op, idx, req);
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), true/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Filter any unversioned fields first
      const bool result = parent_node->filter_unversioned_fields(ctx.get_id(),
                                                      context, user_mask, req);
      // Do the traversal
      FieldMask already_closed_mask;
      parent_node->register_logical_deletion(ctx.get_id(), user, user_mask,
          path, trace_info, already_closed_mask, invalidate_tree);
      // Once we are done we can clear out the list of recorded dependences
      op->clear_logical_records();
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), false/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      return result; 
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::find_open_complete_partitions(Operation *op,
                                                         unsigned idx,
                                                  const RegionRequirement &req,
                                      std::vector<LogicalPartition> &partitions)
    //--------------------------------------------------------------------------
    {
      TaskContext *context = op->get_context();
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *region_node = get_node(req.region);
      FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
      region_node->find_open_complete_partitions(ctx.get_id(), user_mask, 
                                                 partitions);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_versioning_analysis(Operation *op,
                     unsigned index, const RegionRequirement &req, 
                     VersionInfo &version_info, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_VERSIONING_ANALYSIS_CALL);
      if (IS_NO_ACCESS(req))
        return;
      InnerContext *context = op->get_context();
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
      assert((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
      ((req.handle_type == LEGION_REGION_PROJECTION) && (req.projection == 0)));
#endif
      RegionNode *region_node = get_node(req.region);
      FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
      region_node->perform_versioning_analysis(ctx.get_id(), context, 
                  &version_info, user_mask, op->get_unique_op_id(), 
                  runtime->address_space, ready_events);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_current_context(RegionTreeContext ctx,
                                          bool users_only, RegionNode *top_node)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_INVALIDATE_CONTEXT_CALL);
      CurrentInvalidator invalidator(ctx.get_id(), users_only);
      top_node->visit_node(&invalidator);
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
        if (it->first->is_materialized_view())
        {
          MaterializedView *view = it->first->as_materialized_view();
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
      RegionTreeContext ctx = context->get_context();
      assert(ctx.exists());
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
                                     region_node, ctx.get_id(), 
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
      across->src_indirect_instance_event = 
        idx_target.get_physical_manager()->get_unique_event();
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
      across->dst_indirect_instance_event = 
        idx_target.get_physical_manager()->get_unique_event();
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
      across->src_indirect_instance_event = 
       src_idx_target.get_physical_manager()->get_unique_event();
      // Initialize the destination indirections
      const InstanceRef &dst_idx_target = dst_idx_targets[0];
      // Only exclusive if we're the only point sctatting to our instance
      // and we're not racing with any other operations
      const bool exclusive_redop = (dst_records.size() == 1) && 
        (IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req));
      across->initialize_destination_indirections(this, dst_records,
          dst_req, dst_idx_req, dst_idx_target, both_are_range,
          possible_dst_out_of_range, possible_dst_aliasing, exclusive_redop);
      across->dst_indirect_instance_event = 
        dst_idx_target.get_physical_manager()->get_unique_event();
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
                                field_set, attach_node, attach_op);
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
    void RegionTreeForest::check_context_state(RegionTreeContext ctx)
    //--------------------------------------------------------------------------
    {
      CurrentInitializer init(ctx.get_id());
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
      IndexSpaceCreator creator(this, sp, bounds, is_domain, parent, color,
                                did, is_ready, expr_id, initialized, depth,
                                provenance, mapping, tree_valid);
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
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
        if (parent != NULL)
        {
#ifdef DEBUG_LEGION
          assert(!add_root_reference);
#endif
          parent->add_child(result);
        }
        else if (add_root_reference)
          result->add_base_valid_ref(APPLICATION_REF);
        result->register_with_runtime();
      }
      // If we had a realm index space issue the tighten now since
      // we know that we'll probably need it later
      // We have to do this after we've added our reference in case
      // the tighten gets done and tries to delete the node
      if (bounds != NULL)
        result->tighten_index_space();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(IndexSpace sp,
                                                  const void *realm_is,
                                                  IndexPartNode &parent,
                                                  LegionColor color,
                                                  DistributedID did,
                                                  RtEvent initialized,
                                                  Provenance *provenance,
                                                  ApUserEvent is_ready,
                                                  CollectiveMapping *mapping,
                                                  unsigned depth)
    //--------------------------------------------------------------------------
    { 
      IndexSpaceCreator creator(this, sp, realm_is, false/*is domain*/, &parent,
                                color, did, is_ready, 0/*expr id*/, initialized,
                                depth, provenance, mapping, true/*tree valid*/);
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
          // Free up the event since we didn't use it
          Runtime::trigger_event(NULL, is_ready);
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
        // Always add a valid reference from the parent
        parent.add_child(result);
        result->register_with_runtime();
      } 
      // If we had a realm index space issue the tighten now since
      // we know that we'll probably need it later
      // We have to do this after we've added our reference in case
      // the tighten gets done and tries to delete the node
      if (realm_is != NULL)
        result->tighten_index_space();
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
                                                 ApEvent part_ready,
                                                 ApBarrier pending,
                                                 RtEvent initialized,
                                                 CollectiveMapping *mapping,
                                                 ShardMapping *shard_mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(this, p, parent, color_space, color, disjoint,
                               complete, did, part_ready, pending, initialized,
                               mapping, shard_mapping, provenance);
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
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, 
                                                 IndexSpaceNode *parent,
                                                 IndexSpaceNode *color_space,
                                                 LegionColor color,
                                                 RtEvent disjointness_ready,
                                                 int complete, 
                                                 DistributedID did,
                                                 Provenance *provenance,
                                                 ApEvent part_ready,
                                                 ApBarrier pending,
                                                 RtEvent initialized,
                                                 CollectiveMapping *mapping,
                                                 ShardMapping *shard_mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(this, p, parent, color_space, color, 
                               disjointness_ready, complete, did, part_ready, 
                               pending, initialized, mapping, shard_mapping,
                               provenance);
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
                                                  ShardMapping *shard_mapping)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *result = new FieldSpaceNode(space, this, did,
          initialized, mapping, shard_mapping, provenance);
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
                                                  Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *result = 
        new FieldSpaceNode(space, this, did, initialized, provenance, derez);
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
    RegionNode* RegionTreeForest::get_tree(RegionTreeID tid,bool first/*=true*/)
    //--------------------------------------------------------------------------
    {
      if (tid == 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_TREE_ID,
          "Invalid request for tree ID 0 which is never a tree ID")
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
          return get_tree(tid, false/*first*/); 
        }
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
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_TOPLEVEL_TREE,
          "Unable to find top-level tree entry for "
                         "region tree %d.  This is either a runtime "
                         "bug or requires Legion fences if names are "
                         "being returned out of the context in which"
                         "they are being created.", tid)
      return finder->second;
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::request_node(IndexSpace space)
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
        runtime->send_index_space_request(owner, rez);     
        return done;
      }
      else
        return wait_finder->second;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (index_nodes.find(space) != index_nodes.end());
    }
    
    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexPartition part)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (index_parts.find(part) != index_parts.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (field_nodes.find(space) != field_nodes.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Reflect that we can build these nodes whenever this is true
      return (has_node(handle.index_space) && has_node(handle.field_space) &&
              has_tree(handle.tree_id));
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      // Reflect that we can build these nodes whenever this is true
      return (has_node(handle.index_partition) && has_node(handle.field_space)
              && has_tree(handle.tree_id));
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_tree(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (tree_nodes.find(tid) != tree_nodes.end());
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
      std::vector<LegionColor> path;
      return compute_index_path(parent.get_index_space(),
                                child.get_index_space(), path);
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
    bool RegionTreeForest::compute_index_path(IndexSpace parent, 
                               IndexSpace child, std::vector<LegionColor> &path)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(child); 
      path.push_back(child_node->color);
      if (parent == child) 
        return true; // Early out
      IndexSpaceNode *parent_node = get_node(parent);
      while (parent_node != child_node)
      {
        if (parent_node->depth >= child_node->depth)
        {
          path.clear();
          return false;
        }
        if (child_node->parent == NULL)
        {
          path.clear();
          return false;
        }
        path.push_back(child_node->parent->color);
        path.push_back(child_node->parent->parent->color);
        child_node = child_node->parent->parent;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_partition_path(IndexSpace parent, 
                           IndexPartition child, std::vector<LegionColor> &path)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *child_node = get_node(child);
      path.push_back(child_node->color);
      if (child_node->parent == NULL)
      {
        path.clear();
        return false;
      }
      return compute_index_path(parent, child_node->parent->handle, path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexSpace child, IndexSpace parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexPartition child, 
                                           IndexSpace parent, 
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexSpace child,
                                           IndexPartition parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexPartition child,
                                           IndexPartition parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
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

    //--------------------------------------------------------------------------
    void RegionTreeForest::dump_physical_state(LogicalRegion region,
                                               ContextID ctx)
    //--------------------------------------------------------------------------
    {
      TreeStateLogger dump_logger;
      assert(region_nodes.find(region) != region_nodes.end());
      region_nodes[region]->dump_physical_context(ctx, &dump_logger,
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
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_index_space_name(handle.id, ptr);
      }
      if (runtime->profiler && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
	runtime->profiler->record_index_space(handle.id, ptr);
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
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_index_partition_name(handle.id, ptr);
      }
      if (runtime->profiler && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
	runtime->profiler->record_index_part(handle.id, ptr);
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
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_field_space_name(handle.id, ptr);
      }
      if (runtime->profiler && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
	runtime->profiler->record_field_space(handle.id, ptr);
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
        static_assert(sizeof(buf) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buf, sizeof(ptr));
        LegionSpy::log_field_name(handle.id, fid, ptr);
      }
      if (runtime->profiler && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buf) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buf, sizeof(ptr));
	runtime->profiler->record_field(handle.id, fid, size, ptr); 
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
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
        LegionSpy::log_logical_region_name(handle.index_space.id,
            handle.field_space.id, handle.tree_id, ptr);
      }
      if (runtime->profiler && (LEGION_NAME_SEMANTIC_TAG == tag))
      {
        const char *ptr = NULL;
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
        memcpy(&ptr, &buffer, sizeof(ptr));
	runtime->profiler->record_logical_region(handle.index_space.id,
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
        static_assert(sizeof(buffer) == sizeof(ptr), "Fuck c++");
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
      std::set<IndexSpaceExpression*> &exprs = canonical_expressions[key];
      std::set<IndexSpaceExpression*>::iterator finder = exprs.find(expr);
#ifdef DEBUG_LEGION
      assert(finder != exprs.end());
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
          src_unique_events.push_back(
              ref.get_physical_manager()->get_unique_event());
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
          dst_unique_events.push_back(
              ref.get_physical_manager()->get_unique_event());
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
      src_indirect_instance =
        indirect_instance.get_physical_manager()->get_instance();
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
      dst_indirect_instance =
        indirect_instance.get_physical_manager()->get_instance();
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
    void CopyAcrossUnstructured::log_across_profiling(LgEvent copy_post,
                                                      int preimage) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(runtime->profiler != NULL);
      assert(src_fields.size() == dst_fields.size());
#endif
      if (0 <= preimage)
      {
#ifdef DEBUG_LEGION
        assert(((unsigned)preimage) < nonempty_indexes.size());
        assert(src_indirections.empty() != dst_indirections.empty());
#endif
        runtime->profiler->record_indirect_instances(src_indirect_field,
           dst_indirect_field, src_indirect_instance, dst_indirect_instance,
           src_indirect_instance_event, dst_indirect_instance_event, copy_post);
        const unsigned index = nonempty_indexes[preimage];
        if (src_indirections.empty())
        {
          // Scatter
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            runtime->profiler->record_copy_instances(src_fields[idx].field_id,
                dst_fields[idx].field_id, src_fields[idx].inst,
                dst_indirections[index].instances[idx], src_unique_events[idx],
                dst_indirections[index].instance_events[idx], copy_post);
        }
        else
        {
          // Gather
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            runtime->profiler->record_copy_instances(
                src_fields[idx].field_id, dst_fields[idx].field_id,
                src_indirections[index].instances[idx], dst_fields[idx].inst,
                src_indirections[index].instance_events[idx], 
                dst_unique_events[idx], copy_post);
        }
      }
      else if (src_indirections.empty())
      {
        if (dst_indirections.empty())
        {
          // Normal across
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            runtime->profiler->record_copy_instances(
                src_fields[idx].field_id, dst_fields[idx].field_id,
                src_fields[idx].inst, dst_fields[idx].inst,
                src_unique_events[idx], dst_unique_events[idx], copy_post);
        }
        else
        {
          // Scatter
          runtime->profiler->record_indirect_instances(src_indirect_field,
           dst_indirect_field, src_indirect_instance, dst_indirect_instance,
           src_indirect_instance_event, dst_indirect_instance_event, copy_post);
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            for (std::vector<IndirectRecord>::const_iterator it =
                  dst_indirections.begin(); it != dst_indirections.end(); it++)
              runtime->profiler->record_copy_instances(
                  src_fields[idx].field_id, dst_fields[idx].field_id,
                  src_fields[idx].inst, it->instances[idx],
                  src_unique_events[idx], it->instance_events[idx], copy_post);
        }
      }
      else
      {
        if (dst_indirections.empty())
        {
          // Gather
          runtime->profiler->record_indirect_instances(src_indirect_field,
           dst_indirect_field, src_indirect_instance, dst_indirect_instance,
           src_indirect_instance_event, dst_indirect_instance_event, copy_post);
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            for (std::vector<IndirectRecord>::const_iterator it =
                  src_indirections.begin(); it != src_indirections.end(); it++)
              runtime->profiler->record_copy_instances(
                  src_fields[idx].field_id, dst_fields[idx].field_id,
                  it->instances[idx], dst_fields[idx].inst,
                  it->instance_events[idx], dst_unique_events[idx], copy_post);
        }
        else
        {
          // Full indirection
          runtime->profiler->record_indirect_instances(src_indirect_field,
           dst_indirect_field, src_indirect_instance, dst_indirect_instance,
           src_indirect_instance_event, dst_indirect_instance_event, copy_post);
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            for (std::vector<IndirectRecord>::const_iterator src_it =
                  src_indirections.begin(); src_it !=
                  src_indirections.end(); src_it++)
              for (std::vector<IndirectRecord>::const_iterator dst_it =
                    dst_indirections.begin(); dst_it !=
                    dst_indirections.end(); dst_it++)
                runtime->profiler->record_copy_instances(
                    src_fields[idx].field_id, dst_fields[idx].field_id,
                    src_it->instances[idx], dst_it->instances[idx],
                    src_it->instance_events[idx], 
                    dst_it->instance_events[idx], copy_post);
        }
      }
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
                                   DistributedID did, ApEvent ready,
                                   IndexSpaceExprID exp_id, RtEvent init,
                                   unsigned dep, Provenance *prov,
                                   CollectiveMapping *map, bool tree_valid)
      : IndexTreeNode(ctx,
          (dep == UINT_MAX) ? ((par == NULL) ? 0 : par->depth + 1) : dep, c, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_SPACE_NODE_DC),
          init, map, prov, tree_valid),
        IndexSpaceExpression(h.type_tag, exp_id > 0 ? exp_id : 
            runtime->get_unique_index_space_expr_id(), node_lock),
        handle(h), parent(par), index_space_ready(ready),
        next_uncollected_color(0),
        realm_index_space_set(Runtime::create_rt_user_event()), 
        tight_index_space_set(Runtime::create_rt_user_event()),
        index_space_set(false), tight_index_space(false)
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
      // Clean-up any untriggered events
      if (!realm_index_space_set.has_triggered())
        Runtime::trigger_event(realm_index_space_set);
      if (!tight_index_space_set.has_triggered())
        Runtime::trigger_event(tight_index_space_set);
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
      return (child != NULL);
    }

    //--------------------------------------------------------------------------
    LegionColor IndexSpaceNode::generate_color(LegionColor suggestion)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        AutoLock n_lock(node_lock);
        // If the user made a suggestion see if it was right
        if (suggestion != INVALID_COLOR)
        {
          // If someone already has it then they can't use it
          if ((next_uncollected_color <= suggestion) &&
              (color_map.find(suggestion) == color_map.end()))
          {
            color_map[suggestion] = (IndexPartNode*)PENDING_CHILD;
            return suggestion;
          }
          else
            return INVALID_COLOR;
        }
        if (color_map.empty())
        {
          // save a space for later
          color_map[next_uncollected_color] = (IndexPartNode*)PENDING_CHILD;
          return next_uncollected_color;
        }
        std::map<LegionColor,IndexPartNode*>::const_iterator next = 
          color_map.begin();
        if (next->first > next_uncollected_color)
        {
          // save a space for later
          color_map[next_uncollected_color] = (IndexPartNode*)PENDING_CHILD;
          return next_uncollected_color;
        }
        std::map<LegionColor,IndexPartNode*>::const_iterator prev = next++;
        while (next != color_map.end())
        {
          if (next->first != (prev->first + 1))
          {
            // save a space for later
            color_map[prev->first+1] = (IndexPartNode*)PENDING_CHILD;
            return prev->first+1;
          }
          prev = next++;
        }
        color_map[prev->first+1] = (IndexPartNode*)PENDING_CHILD;
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
        assert(finder->second == (IndexPartNode*)PENDING_CHILD);
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
          return finder->second;
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
          return context->get_node(remote_handle, defer);
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
        return context->get_node(child_handle);
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
             (color_map[child->color] == ((IndexPartNode*)PENDING_CHILD)));
#endif
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
      assert(finder->second != ((IndexPartNode*)PENDING_CHILD));
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
    bool IndexSpaceNode::are_disjoint(LegionColor c1, LegionColor c2)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (c1 == c2)
        return false;
      if (c1 > c2)
      {
        LegionColor t = c1;
        c1 = c2;
        c2 = t;
      }
      // Do the test with read-only mode first
      RtEvent ready;
      bool issue_dynamic_test = false;
      std::pair<LegionColor,LegionColor> key(c1,c2);
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        if (disjoint_subsets.find(key) != disjoint_subsets.end())
          return true;
        else if (aliased_subsets.find(key) != aliased_subsets.end())
          return false;
        else
        {
          std::map<std::pair<LegionColor,LegionColor>,RtEvent>::const_iterator
            finder = pending_tests.find(key);
          if (finder != pending_tests.end())
            ready = finder->second;
          else
          {
            if (!implicit_runtime->disable_independence_tests)
              issue_dynamic_test = true;
            else
            {
              aliased_subsets.insert(key);
              return false;
            }
          }
        }
      }
      if (issue_dynamic_test)
      {
        IndexPartNode *left = get_child(c1);
        const bool left_complete = left->is_complete(false, true);
        IndexPartNode *right = get_child(c2);
        const bool right_complete = right->is_complete(false, true);
        AutoLock n_lock(node_lock);
        // If either one is known to be complete then we know that they
        // must be aliased with each other
        if (left_complete || right_complete)
        {
          aliased_subsets.insert(key);
          return false;
        }
        // Test again to make sure we didn't lose the race
        std::map<std::pair<LegionColor,LegionColor>,RtEvent>::const_iterator
          finder = pending_tests.find(key);
        if (finder == pending_tests.end())
        {
          DynamicIndependenceArgs args(this, left, right);
          // Get the preconditions for domains 
          RtEvent pre = Runtime::protect_event(
              Runtime::merge_events(NULL,
                left->partition_ready, right->partition_ready));
          ready = context->runtime->issue_runtime_meta_task(args,
                                      LG_LATENCY_WORK_PRIORITY, pre);
          pending_tests[key] = ready;
        }
        else
          ready = finder->second;
      }
      // Wait for the ready event and then get the result
      ready.wait();
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      if (disjoint_subsets.find(key) != disjoint_subsets.end())
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::record_disjointness(bool disjoint, 
                                             LegionColor c1, LegionColor c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
      if (c1 > c2)
      {
        LegionColor t = c1;
        c1 = c2;
        c2 = t;
      }
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (disjoint)
        disjoint_subsets.insert(std::pair<LegionColor,LegionColor>(c1,c2));
      else
        aliased_subsets.insert(std::pair<LegionColor,LegionColor>(c1,c2));
      pending_tests.erase(std::pair<LegionColor,LegionColor>(c1,c2));
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::record_remote_child(IndexPartition pid, 
                                             LegionColor part_color)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(remote_colors.find(part_color) == remote_colors.end());
      // should only happen on the owner node
      assert(get_owner_space() == context->runtime->address_space);
#endif
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
      if (target == source)
        return;
      if (target == runtime->address_space)
        return;
      runtime->send_index_space_set(target, rez);
    } 

    //--------------------------------------------------------------------------
    IndexSpaceNode::DynamicIndependenceArgs::DynamicIndependenceArgs(
                        IndexSpaceNode *par, IndexPartNode *l, IndexPartNode *r)
      : LgTaskArgs<DynamicIndependenceArgs>(implicit_provenance),
        parent(par), left(l), right(r)
    //--------------------------------------------------------------------------
    {
      left->add_base_resource_ref(META_TASK_REF);
      right->add_base_resource_ref(META_TASK_REF);
      parent->add_base_resource_ref(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_disjointness_test(const void *args)
    //--------------------------------------------------------------------------
    {
      const DynamicIndependenceArgs *dargs =
        (const DynamicIndependenceArgs*)args;
      const bool disjoint = !dargs->left->intersects_with(dargs->right);
      dargs->parent->record_disjointness(disjoint, dargs->left->color,
                                         dargs->right->color);
      if (dargs->left->remove_base_resource_ref(META_TASK_REF))
        delete dargs->left;
      if (dargs->right->remove_base_resource_ref(META_TASK_REF))
        delete dargs->right;
      if (dargs->parent->remove_base_resource_ref(META_TASK_REF))
        delete dargs->parent;
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
      if (is_owner())
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
      if (index_space_set)
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
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      IndexSpaceNode *target = forest->get_node(handle, NULL, true/*can fail*/);
      bool valid = false;
      if (target != NULL)
      {
        // See if we're going to be sending the whole tree or not
        bool recurse = true;
        if (target->parent == NULL)
        {
          if (target->check_valid_and_increment(REGION_TREE_REF))
          {
            valid = true;
            target->pack_valid_ref();
            target->remove_base_valid_ref(REGION_TREE_REF);
          }
          else
          {
            target->pack_global_ref();
            recurse = false;
          }
        }
        else
        {
          // If we have a parent then we need to do the valid reference
          // check on the partition since that keeps this tree valid
          if (target->parent->check_valid_and_increment(REGION_TREE_REF))
          {
            valid = true;
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
            recurse = false;
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
        }
        forest->runtime->send_index_space_return(source, rez);
      }
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_return(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
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
      if (valid)
      {
        if (node->parent == NULL)
          node->unpack_valid_ref();
        else
          node->parent->unpack_valid_ref();
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
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(child->handle);
          rez.serialize(target);
          rez.serialize(to_trigger);
        }
        forest->runtime->send_index_space_child_response(source, rez);
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
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(child->handle);
          rez.serialize(dargs->target);
          rez.serialize(dargs->to_trigger);
        }
        Runtime *runtime = dargs->proxy_this->context->runtime;
        runtime->send_index_space_child_response(dargs->source, rez);
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
        RtEvent defer; 
        forest->get_node(handle, &defer);
        Runtime::trigger_event(to_trigger, defer);
      }
      else
      {
        target->store(handle.get_id());
        Runtime::trigger_event(to_trigger);
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
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode *node = forest->get_node(handle);
      if (node->unpack_index_space(derez, source))
        delete node;
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
      IndexSpaceExpression *intersect = 
        context->intersect_index_spaces(this, rhs);
      return !intersect->is_empty();
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::intersects_with(IndexPartNode *rhs, bool compute)
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
        IndexSpaceNode *one = this;
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
        // it is disjoint then they don't intersect if they are different
        if ((one->parent != NULL) && (one != two) && one->parent->is_disjoint())
          return false;
        // Otherwise fall through and do the expensive test
      }
      IndexSpaceExpression *intersect = 
        context->intersect_index_spaces(this, rhs->get_union_expression());
      return !intersect->is_empty();
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

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::dominates(IndexPartNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      // A simple but common case
      if (rhs->parent == this)
        return true;
      // We're about to do something expensive so use the region tree
      // as an acceleration data structure to try to make our tests
      // more efficient. If these are in the same tree, see if we can
      // walk up the tree from rhs and find ourself
      if (handle.get_tree_id() == rhs->handle.get_tree_id())
      {
        // If we're the root of the tree we also trivially domainate
        if (depth == 0)
          return true;
        if (rhs->depth > depth)
        {
          IndexSpaceNode *temp = rhs->parent;
          while (depth < temp->depth)
            temp = temp->parent->parent;
          // If we find ourself at the same depth then we dominate
          if (temp == this)
            return true;
        }
        // Otherwise we fall through and do the expensive test
      }
      IndexSpaceExpression *diff = 
        context->subtract_index_spaces(rhs->get_union_expression(), this);
      return diff->is_empty();
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(RegionTreeForest *ctx, IndexPartition p, 
                                 IndexSpaceNode *par, IndexSpaceNode *color_sp,
                                 LegionColor c, bool dis, int comp, 
                                 DistributedID did, ApEvent part_ready, 
                                 ApBarrier partial, RtEvent init,
                                 CollectiveMapping *mapping, ShardMapping *map,
                                 Provenance *prov)
      : IndexTreeNode(ctx, par->depth+1, c,
                      LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_PART_NODE_DC),
                      init, mapping, prov, true/*tree valid*/),
        handle(p), parent(par), color_space(color_sp), 
        total_children(color_sp->get_volume()), 
        max_linearized_color(color_sp->get_max_linearized_color()),
        partition_ready(part_ready), partial_pending(partial),
        shard_mapping(map), disjoint(dis), has_complete(comp >= 0),
        complete(comp != 0), first_entry(NULL)
    //--------------------------------------------------------------------------
    { 
      parent->add_nested_resource_ref(did);
      color_space->add_nested_resource_ref(did);
      if (has_complete && complete)
      {
        parent->add_nested_expression_reference(did);
        union_expr.store(parent);
      }
      else
        union_expr.store(NULL);
      if (shard_mapping != NULL)
        shard_mapping->add_reference();
#ifdef DEBUG_LEGION
      if (partial_pending.exists())
        assert(partial_pending == partition_ready);
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
                                 LegionColor c, RtEvent dis_ready,
                                 int comp, DistributedID did,
                                 ApEvent part_ready, ApBarrier part,
                                 RtEvent init, CollectiveMapping *map,
                                 ShardMapping *shard_map, Provenance *prov)
      : IndexTreeNode(ctx, par->depth+1, c,
                      LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_PART_NODE_DC),
                      init, map, prov, true/*tree valid*/),
        handle(p), parent(par), color_space(color_sp), 
        total_children(color_sp->get_volume()),
        max_linearized_color(color_sp->get_max_linearized_color()),
        partition_ready(part_ready), partial_pending(part),
        shard_mapping(shard_map), disjoint_ready(dis_ready), disjoint(false), 
        has_complete(comp >= 0), complete(comp != 0),
        first_entry(NULL)
    //--------------------------------------------------------------------------
    {
      parent->add_nested_resource_ref(did);
      color_space->add_nested_resource_ref(did);
      if (has_complete && complete)
      {
        parent->add_nested_expression_reference(did);
        union_expr.store(parent);
      }
      else
        union_expr.store(NULL);
      if (shard_mapping != NULL)
        shard_mapping->add_reference();
#ifdef DEBUG_LEGION
      if (partial_pending.exists())
        assert(partial_pending == partition_ready);
      assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Partition %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.id);
#endif
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
      if ((shard_mapping != NULL) && shard_mapping->remove_reference())
        delete shard_mapping;
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
      // Remove the reference on our union expression if we have one
      IndexSpaceExpression *expr = union_expr.load();
      if ((expr != NULL) && expr->remove_nested_expression_reference(did))
        delete expr;
      for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
            color_map.begin(); it != color_map.end(); it++)
        if (it->second->remove_nested_resource_ref(did))
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
      // TODO: Remove this check here, it is not precise
      if (!color_space->contains_color(c, false/*report error*/))
        REPORT_LEGION_ERROR(ERROR_INVALID_INDEX_SPACE_COLOR,
                            "Invalid color space color for child %lld "
                            "of partition %d", c, handle.get_id())
      // Check to see if we're the owner space this child, in the
      // common case the owner space is whichever node made the 
      // partition, but in the control replication case, we use the
      // shard mapping to determine which node owns the given child
      const AddressSpaceID owner_space = 
        (shard_mapping == NULL) ? get_owner_space() :
        (*shard_mapping)[c % shard_mapping->size()];
      const AddressSpaceID local_space = context->runtime->address_space;
      // If we own the index partition, create a new subspace here
      if (owner_space == local_space)
      {
        // First do a check to see someone else is already making
        // the child in which case we should just wait for it to
        // be ready
        RtEvent wait_on;
        {
          AutoLock n_lock(node_lock);
          // Check to make sure we didn't loose the race
          std::map<LegionColor,IndexSpaceNode*>::const_iterator child_finder =
            color_map.find(c);
          if (child_finder != color_map.end())
            return child_finder->second;
          // Didn't loose the race so we can keep going
          std::map<LegionColor,RtUserEvent>::const_iterator finder = 
            pending_child_map.find(c);
          if (finder == pending_child_map.end())
            pending_child_map[c] = Runtime::create_rt_user_event();
          else
            wait_on = finder->second;
        }
        if (wait_on.exists())
        {
          // Someone else is already making it so just wait
          if (defer == NULL)
          {
            wait_on.wait();
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            std::map<LegionColor,IndexSpaceNode*>::const_iterator finder =
              color_map.find(c);
#ifdef DEBUG_LEGION
            // It better be here when we wake up
            assert(finder != color_map.end());
#endif
            return finder->second;
          }
          else
          {
            *defer = wait_on;
            return NULL;
          }
        }
        else
        {
          // We're making this so just do it
          IndexSpace is(context->runtime->get_unique_index_space_id(),
                        handle.get_tree_id(), handle.get_type_tag());
          DistributedID did = 
            context->runtime->get_available_distributed_id();
          IndexSpaceNode *result = NULL;
          if (partial_pending.exists())
          {
            ApUserEvent partial_event = Runtime::create_ap_user_event(NULL);
            result = context->create_node(is, NULL/*realm is*/, *this, c, did,
                                          initialized,provenance,partial_event);
            Runtime::phase_barrier_arrive(partial_pending, 
                                          1/*count*/, partial_event);
          }
          else
            // Make a new index space node ready when the partition is ready
            result = context->create_node(is, NULL/*realm is*/, false, this, c,
                                did, initialized, provenance, partition_ready);
          if (runtime->legion_spy_enabled)
            LegionSpy::log_index_subspace(handle.id, is.id, 
                runtime->address_space, result->get_domain_point_color());
          if (runtime->profiler != NULL)
	    runtime->profiler->record_index_subspace(handle.id, is.id,
                result->get_domain_point_color());
          return result; 
        }
      }
      // Otherwise, request a child node from the owner node
      else
      {
        std::atomic<IndexSpaceID> child_id(0);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(c);
          if (defer == NULL)
            rez.serialize(&child_id);
          else
            rez.serialize<std::atomic<IndexSpaceID>*>(NULL);
          rez.serialize(ready_event);
        }
        context->runtime->send_index_partition_child_request(owner_space, rez);
        if (defer == NULL)
        {
          ready_event.wait();
          IndexSpace child_handle(child_id.load(),
              handle.get_tree_id(), handle.get_type_tag());
#ifdef DEBUG_LEGION
          assert(child_handle.exists());
#endif
          return context->get_node(child_handle);
        }
        else
        {
          *defer = ready_event;
          return NULL;
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_child(IndexSpaceNode *child) 
    //--------------------------------------------------------------------------
    {
      // This child should live as long as we are alive
      child->add_nested_resource_ref(did);
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
        if (finder == pending_child_map.end())
          return;
        to_trigger = finder->second;
        pending_child_map.erase(finder);
      }
      Runtime::trigger_event(to_trigger);
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
                        Serializer &r, Runtime *rt, ShardMapping *shard_mapping)
      : rez(r), runtime(rt)
    //--------------------------------------------------------------------------
    {
      if (shard_mapping != NULL)
      {
        for (unsigned idx = 0; idx < shard_mapping->size(); idx++)
          skip_shard_spaces.insert((*shard_mapping)[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::RemoteDisjointnessFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if ((target != runtime->address_space) && 
          (skip_shard_spaces.empty() ||
           (skip_shard_spaces.find(target) == skip_shard_spaces.end())))
        runtime->send_index_partition_disjoint_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::compute_disjointness(ValueBroadcast<bool> *collective,
                                             bool owner)
    //--------------------------------------------------------------------------
    {
      if (owner)
      {
        // If we're the owner we do the disjointness test
        disjoint = true;
        if (is_complete(false/*from app*/, true/*false if not ready*/))
        {
          // If we're complete we can check this using a linear algorithm
          // by suming up the volumes of all the children
          const size_t parent_volume = parent->get_volume();
          size_t children_volume = 0;
          if (total_children == max_linearized_color)
          {
            for (LegionColor color = 0; color < max_linearized_color; color++)
            {
              IndexSpaceNode *child = get_child(color);
              children_volume += child->get_volume();
              if (children_volume > parent_volume)
                break;
            }
          }
          else
          {
            ColorSpaceIterator *itr =
              color_space->create_color_space_iterator();
            while (itr->is_valid())
            {
              const LegionColor color = itr->yield_color();
              IndexSpaceNode *child = get_child(color);
              children_volume += child->get_volume();
              if (children_volume > parent_volume)
                break;
            }
            delete itr;
          }
#ifdef DEBUG_LEGION
          assert(parent_volume <= children_volume);
#endif
          if (parent_volume < children_volume)
            disjoint = false;
        }
        else if (total_children == max_linearized_color)
        {
          for (LegionColor c1 = 0; disjoint && 
                (c1 < max_linearized_color); c1++)
          {
            for (LegionColor c2 = c1 + 1; disjoint && 
                  (c2 < max_linearized_color); c2++)
            {
              if (!are_disjoint(c1, c2, true/*force compute*/))
              {
                disjoint = false;
                break;
              }
            }
            if (!disjoint)
              break;
          }
        }
        else
        {
          for (LegionColor c1 = 0; disjoint && 
                (c1 < max_linearized_color); c1++)
          {
            if (!color_space->contains_color(c1))
              continue;
            for (LegionColor c2 = c1 + 1; disjoint &&
                  (c2 < max_linearized_color); c2++)
            {
              if (!color_space->contains_color(c2))
                continue;
              if (!are_disjoint(c1, c2, true/*force compute*/))
              {
                disjoint = false;
                break;
              }
            }
            if (!disjoint)
              break;
          }
        }
        // Make sure the write of disjoint propagates before 
        // we do the trigger of the event
        __sync_synchronize();
        {
          AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
          assert(disjoint_ready.exists());
#endif
          // We have to send notifications before any other remote
          // requests can record themselves so we need to do it 
          // while we are holding the lock
          if (has_remote_instances() && 
              ((shard_mapping == NULL) || 
               (count_remote_instances() > shard_mapping->size())))
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize<bool>(disjoint);
            }
            RemoteDisjointnessFunctor functor(rez, 
                  context->runtime, shard_mapping);
            map_over_remote_instances(functor);
          }
        }
        // If we have a disjointness barrier, then signal the result
        if (collective != NULL)
          collective->broadcast(disjoint);
        if (implicit_runtime->profiler != NULL)
          runtime->profiler->record_index_partition(parent->handle.id,handle.id,
                                                    disjoint, color);
      }
      else
      {
        // We're not the owner so we should have a barrier that tells
        // us what the disjointness result is
#ifdef DEBUG_LEGION
        assert(collective != NULL);
#endif
        // No need to wait, we know this was a precondition for launching
        // the task to compute the disjointness
        disjoint = collective->get_value();
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_disjoint(bool app_query)
    //--------------------------------------------------------------------------
    {
      if (disjoint_ready.exists() && !disjoint_ready.has_triggered())
        disjoint_ready.wait();
      return disjoint;
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
      {
        LegionColor t = c1;
        c1 = c2;
        c2 = t;
      }
      bool issue_dynamic_test = false;
      std::pair<LegionColor,LegionColor> key(c1,c2);
      RtEvent ready_event;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        if (disjoint_subspaces.find(key) != disjoint_subspaces.end())
          return true;
        else if (aliased_subspaces.find(key) != aliased_subspaces.end())
          return false;
        else
        {
          std::map<std::pair<LegionColor,LegionColor>,RtEvent>::const_iterator
            finder = pending_tests.find(key);
          if (finder != pending_tests.end())
            ready_event = finder->second;
          else
          {
            if (!implicit_runtime->disable_independence_tests)
              issue_dynamic_test = true;
            else
            {
              aliased_subspaces.insert(key);
              return false;
            }
          }
        }
      }
      if (issue_dynamic_test)
      {
        IndexSpaceNode *left = get_child(c1);
        IndexSpaceNode *right = get_child(c2);
        ApEvent left_pre = left->index_space_ready;
        ApEvent right_pre = right->index_space_ready;
        AutoLock n_lock(node_lock);
        // Test again to see if we lost the race
        std::map<std::pair<LegionColor,LegionColor>,RtEvent>::const_iterator
          finder = pending_tests.find(key);
        if (finder == pending_tests.end())
        {
          DynamicIndependenceArgs args(this, left, right);
          ApEvent pre = Runtime::merge_events(NULL, left_pre, right_pre);
          ready_event = context->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_WORK_PRIORITY, Runtime::protect_event(pre));
          pending_tests[key] = ready_event;
        }
        else
          ready_event = finder->second;
      }
      ready_event.wait();
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      if (disjoint_subspaces.find(key) != disjoint_subspaces.end())
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::record_disjointness(bool result,
                                            LegionColor c1, LegionColor c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
      if (c1 > c2)
      {
        LegionColor t = c1;
        c1 = c2;
        c2 = t;
      }
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (result)
        disjoint_subspaces.insert(std::pair<LegionColor,LegionColor>(c1,c2));
      else
        aliased_subspaces.insert(std::pair<LegionColor,LegionColor>(c1,c2));
      pending_tests.erase(std::pair<LegionColor,LegionColor>(c1,c2));
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_complete(bool from_app/*=false*/, 
                                    bool false_if_not_ready/*=false*/)
    //--------------------------------------------------------------------------
    {
      // If we've cached the value then we are good to go
      {
        AutoLock n_lock(node_lock, 1, false/*exclusive*/);
        if (has_complete)
          return complete;
        if (false_if_not_ready)
          return false;
      }
      bool result = false;
      // Otherwise compute it 
      if (is_disjoint(from_app))
      {
	// if the partition is disjoint, we can determine completeness by
	//  seeing if the total volume of the child domains matches the volume
	//  of the parent domains
	const size_t parent_volume = parent->get_volume();
	size_t child_volume = 0;
        if (total_children == max_linearized_color)
        {
          for (LegionColor c = 0; c < max_linearized_color; c++)
          {
            IndexSpaceNode *child = get_child(c);
            child_volume += child->get_volume();
          }
        }
        else
        {
          for (LegionColor c = 0; c < max_linearized_color; c++)
          {
            if (!color_space->contains_color(c))
              continue;
            IndexSpaceNode *child = get_child(c);
            child_volume += child->get_volume();
          }
        }
	result = (child_volume == parent_volume);
      } 
      else 
      {
	// if not disjoint, we have to do a considerably-more-expensive test
	//  that handles overlap (i.e. double-counting) in the child domains
	result = compute_complete();
      }
      // Save the result for the future
      AutoLock n_lock(node_lock);
      // See if we lost the race
      complete = result;
      has_complete = true;
      return complete;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* IndexPartNode::get_union_expression(
                                                            bool check_complete)
    //--------------------------------------------------------------------------
    {
      IndexSpaceExpression *result = union_expr.load();
      if (result != NULL)
        return result;
      // If we're complete then we can use the parent index space expresion
      if (!check_complete || !is_complete())
      {
        // We can always write the result immediately since we know
        // that the common sub-expression code will give the same
        // result if there is a race
        IndexSpaceExpression *expr = compute_union_expression();
        expr->add_nested_expression_reference(did);
        union_expr.store(expr);
      }
      else // if we're complete the parent is our expression
      {
        parent->add_nested_expression_reference(did);
        union_expr.store(parent);
      }
      return union_expr.load();
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* IndexPartNode::compute_union_expression(void)
    //--------------------------------------------------------------------------
    {
      std::set<IndexSpaceExpression*> child_spaces;
      if (total_children == max_linearized_color)
      {
        for (LegionColor color = 0; color < total_children; color++)
          child_spaces.insert(get_child(color));
      }
      else
      {
        for (LegionColor color = 0; color < max_linearized_color; color++)
        {
          if (!color_space->contains_color(color))
            continue;
          child_spaces.insert(get_child(color));
        }
      }
      return context->union_index_spaces(child_spaces);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::record_remote_disjoint_ready(RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ready.exists());
      assert(!remote_disjoint_ready.exists());
#endif
      remote_disjoint_ready = ready;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::record_remote_disjoint_result(const bool result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_disjoint_ready.exists());
#endif
      disjoint = result;
      __sync_synchronize();
      Runtime::trigger_event(remote_disjoint_ready);
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
                                                 size_t granularity, 
                                                 ShardID shard, 
                                                 size_t total_shards)
    //--------------------------------------------------------------------------
    {
      if (total_shards > 1)
        return parent->create_equal_children(op, this, granularity,
                                             shard, total_shards); 
      else
        return parent->create_equal_children(op, this, granularity);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_weights(Operation *op, 
                                   const FutureMap &weights, size_t granularity,
                                   ShardID shard, size_t total_shards)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_weights(op, this, weights.impl, granularity,
                                       shard, total_shards); 
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_union(Operation *op, 
                                           IndexPartNode *left, 
                                           IndexPartNode *right,
                                           ShardID shard,
                                           size_t total_shards)
    //--------------------------------------------------------------------------
    {
      if (total_shards > 1)
        return parent->create_by_union(op, this, left, right, 
                                       shard, total_shards); 
      else
        return parent->create_by_union(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_intersection(Operation *op,
                                                  IndexPartNode *left,
                                                  IndexPartNode *right,
                                                  ShardID shard,
                                                  size_t total_shards)
    //--------------------------------------------------------------------------
    {
      if (total_shards > 1)
        return parent->create_by_intersection(op, this, left, right,
                                              shard, total_shards);
      else
        return parent->create_by_intersection(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_intersection(Operation *op,
                                                  IndexPartNode *original,
                                                  const bool dominates,
                                                  ShardID shard,
                                                  size_t total_shards)
    //--------------------------------------------------------------------------
    {
      if (total_shards > 1)
        return parent->create_by_intersection(op, this, original,
                                              shard, total_shards, dominates);
      else
        return parent->create_by_intersection(op, this, original, dominates);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_difference(Operation *op,
                                                IndexPartNode *left,
                                                IndexPartNode *right,
                                                ShardID shard,
                                                size_t total_shards)
    //--------------------------------------------------------------------------
    {
      if (total_shards > 1)
        return parent->create_by_difference(op, this, left, right,
                                            shard, total_shards);
      else
        return parent->create_by_difference(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_restriction(const void *transform,
                                                 const void *extent,
                                                 ShardID shard,
                                                 size_t total_shards)
    //--------------------------------------------------------------------------
    {
      return color_space->create_by_restriction(this, transform, extent,
                     NT_TemplateHelper::get_dim(handle.get_type_tag()),
                     shard, total_shards);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_disjointness_computation(
                                     const void *args, RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      const DisjointnessArgs *dargs = (const DisjointnessArgs*)args;
      IndexPartNode *node = forest->get_node(dargs->pid);
      node->compute_disjointness(dargs->disjointness_collective, dargs->owner);
      // We can now delete the collective
      if (dargs->disjointness_collective != NULL)
        delete dargs->disjointness_collective;
      if (dargs->owner)
      {
        // Remove the reference on our node as well
        if (node->remove_base_valid_ref(APPLICATION_REF))
          delete node;
      }
      else
      {
        if (node->remove_base_resource_ref(APPLICATION_REF))
          delete node;
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::compute_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_disjoint());
#endif
      IndexSpaceExpression *diff = context->subtract_index_spaces(parent, 
                            get_union_expression(false/*check complete*/));
      return diff->is_empty();
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
      IndexSpaceExpression *intersect = 
        context->intersect_index_spaces(get_union_expression(), rhs);
      return !intersect->is_empty();
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
      IndexSpaceExpression *intersect = 
        context->intersect_index_spaces(get_union_expression(),
                                        rhs->get_union_expression());
      return !intersect->is_empty();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::dominates(IndexSpaceNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      // A simple but common case
      if (rhs->parent == this)
        return true;
      // We're about to do something expensive, so use the region tree
      // as an acceleration data structure to try to make our tests
      // more efficient
      if ((handle.get_tree_id() == rhs->handle.get_tree_id()) && 
          (rhs->depth > depth))
      {
        IndexPartNode *temp = rhs->parent;
        while (depth < temp->depth)
          temp = temp->parent->parent;
        // If we find ourselves at the same depth then we dominate
        if (temp == this)
          return true;
        // Otherwise we fall through and do the expensive test
      }
      IndexSpaceExpression *diff = 
        context->subtract_index_spaces(rhs, get_union_expression());
      return diff->is_empty();
    }
    
    //--------------------------------------------------------------------------
    bool IndexPartNode::dominates(IndexPartNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      if (rhs == this)
        return true;
      // We're about to do something expensive, so use the region tree
      // as an acceleration data structure and try to make our tests
      // more efficient
      if ((handle.get_tree_id() == rhs->handle.get_tree_id()) &&
          (rhs->depth > depth))
      {
        IndexPartNode *temp = rhs;
        while (depth < temp->depth)
          temp = temp->parent->parent;
        // If we find ourselves at the same depth then we dominate
        if (temp == this)
          return true;
        // Otherwise we fall through and do the expensive test
      }
      IndexSpaceExpression *diff = 
        context->subtract_index_spaces(rhs->get_union_expression(),
                                       get_union_expression());
      return diff->is_empty();
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
          if (total_children == max_linearized_color)
          {
            for (LegionColor color = 0; color < total_children; color++)
            {
              IndexSpaceNode *child = get_child(color);
              IndexSpaceExpression *intersection = 
                context->intersect_index_spaces(expr, child);
              if (!intersection->is_empty())
                colors.push_back(color);
            }
          }
          else
          {
            ColorSpaceIterator *itr = color_space->create_color_space_iterator();
            while (itr->is_valid())
            {
              const LegionColor color = itr->yield_color();
              IndexSpaceNode *child = get_child(color);
              IndexSpaceExpression *intersection = 
                context->intersect_index_spaces(expr, child);
              if (!intersection->is_empty())
                colors.push_back(color);
            }
            delete itr;
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
    IndexPartNode::DynamicIndependenceArgs::DynamicIndependenceArgs(
                       IndexPartNode *par, IndexSpaceNode *l, IndexSpaceNode *r)
      : LgTaskArgs<DynamicIndependenceArgs>(implicit_provenance),
        parent(par), left(l), right(r)
    //--------------------------------------------------------------------------
    {
      left->add_base_resource_ref(META_TASK_REF);
      right->add_base_resource_ref(META_TASK_REF);
      parent->add_base_resource_ref(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/void IndexPartNode::handle_disjointness_test(const void *args)
    //--------------------------------------------------------------------------
    {
      const DynamicIndependenceArgs *dargs =
        (const DynamicIndependenceArgs*)args;
      bool disjoint = !dargs->left->intersects_with(dargs->right);
      dargs->parent->record_disjointness(disjoint, dargs->left->color,
                                         dargs->right->color);
      if (dargs->left->remove_base_resource_ref(META_TASK_REF))
        delete dargs->left;
      if (dargs->right->remove_base_resource_ref(META_TASK_REF))
        delete dargs->right;
      if (dargs->parent->remove_base_resource_ref(META_TASK_REF))
        delete dargs->parent;
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
      if (is_owner())
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
      const bool has_disjoint = 
        (!disjoint_ready.exists() || disjoint_ready.has_triggered());
      const bool disjoint_result = has_disjoint ? is_disjoint() : false;
      RezCheck z(rez);
      rez.serialize(handle);
      rez.serialize(did);
      rez.serialize(parent->handle); 
      rez.serialize(color_space->handle);
      rez.serialize(color);
      rez.serialize<bool>(has_disjoint);
      rez.serialize<bool>(disjoint_result);
      if (has_complete)
      {
        if (complete)
          rez.serialize<int>(1); // complete
        else
          rez.serialize<int>(0); // not complete
      }
      else
        rez.serialize<int>(-1); // we don't know yet
      rez.serialize(partition_ready);
      rez.serialize(partial_pending);
      rez.serialize(initialized);
      if (collective_mapping != NULL)
        collective_mapping->pack(rez);
      else
        rez.serialize<size_t>(0); // total spaces
      if (shard_mapping != NULL)
      {
        rez.serialize(shard_mapping->size());
        for (unsigned idx = 0; idx < shard_mapping->size(); idx++)
          rez.serialize((*shard_mapping)[idx]);
      }
      else
        rez.serialize<size_t>(0);
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
      ApEvent ready_event;
      derez.deserialize(ready_event);
      ApBarrier partial_pending;
      derez.deserialize(partial_pending);
      RtEvent initialized;
      derez.deserialize(initialized);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping *mapping = NULL;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);
      size_t num_shard_mapping;
      derez.deserialize(num_shard_mapping);
      ShardMapping *shard_mapping = NULL;
      if (num_shard_mapping > 0)
      {
        shard_mapping = new ShardMapping();
        shard_mapping->resize(num_shard_mapping);
        for (unsigned idx = 0; idx < num_shard_mapping; idx++)
          derez.deserialize((*shard_mapping)[idx]);
      }
      AutoProvenance provenance(Provenance::deserialize(derez));
      IndexSpaceNode *parent_node = context->get_node(parent);
      IndexSpaceNode *color_space_node = context->get_node(color_space);
#ifdef DEBUG_LEGION
      assert(parent_node != NULL);
      assert(color_space_node != NULL);
#endif
      RtUserEvent dis_ready;
      if (!has_disjoint)
        dis_ready = Runtime::create_rt_user_event();
      IndexPartNode *node = has_disjoint ?
        context->create_node(handle, parent_node, color_space_node, color, 
               disjoint, complete, did, provenance, ready_event, 
               partial_pending, initialized, mapping, shard_mapping) :
        context->create_node(handle, parent_node, color_space_node, color,
               dis_ready, complete, did, provenance, ready_event,
               partial_pending, initialized, mapping, shard_mapping);
      if (!has_disjoint)
        node->record_remote_disjoint_ready(dis_ready);
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
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartition handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      IndexPartNode *target = forest->get_node(handle, NULL, true/*can fail*/);
      if (target != NULL)
      {
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
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
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
      std::atomic<IndexSpaceID> *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      IndexPartNode *parent = forest->get_node(handle);
      RtEvent defer;
      IndexSpaceNode *child = parent->get_child(child_color, &defer);
      // If we got a deferral event then we need to make a continuation
      // to avoid blocking the virtual channel for nested index tree requests
      if (defer.exists())
      {
        DeferChildArgs args(parent, child_color, target, to_trigger, source);
        forest->runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, defer);
      }
      else
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(child->handle);
          rez.serialize(target);
          rez.serialize(to_trigger);
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
        rez.serialize(dargs->target);
        rez.serialize(dargs->to_trigger);
      }
      Runtime *runtime = dargs->proxy_this->context->runtime;
      runtime->send_index_partition_child_response(dargs->source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_child_response(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      std::atomic<IndexSpaceID> *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      if (target == NULL)
      {
        RtEvent defer;
        forest->get_node(handle, &defer);
        Runtime::trigger_event(to_trigger, defer);
      }
      else
      {
        target->store(handle.get_id());
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_disjoint_update(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      bool disjoint_result;
      derez.deserialize(disjoint_result);
      IndexPartNode *node = forest->get_node(handle);
      node->record_remote_disjoint_result(disjoint_result);
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
      assert(shard_mapping != NULL);
#endif
      AutoLock n_lock(node_lock);
      if (shard_rects_ready.exists())
        return shard_rects_ready;
      shard_rects_ready = Runtime::create_rt_user_event();
      // Add a reference to keep this node alive until this all done
      add_base_resource_ref(RUNTIME_REF);
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      // Figure out how many downstream requests we have
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(owner_space, local_space, children);
      remaining_rect_notifications = children.size();
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
      // Fill in all the local rectangle values
      initialize_shard_rects();
      if (children.empty())
      {
        if (!is_owner())
        {
          // Pack up the rectangles and send them upstream to the parent
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(true); // up
            pack_shard_rects(rez, true/*clear*/);
          }
          context->runtime->send_index_partition_shard_rects_response(
              collective_mapping->get_parent(owner_space, local_space), rez);
        }
        else
        {
          Runtime::trigger_event(shard_rects_ready);
          if (remove_base_resource_ref(RUNTIME_REF))
            assert(false); // should never hit this
        }
      }
      return shard_rects_ready;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::process_shard_rects_response(Deserializer &derez,
                                                     AddressSpace source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_mapping != NULL);
#endif
      bool up;
      derez.deserialize<bool>(up);
      AutoLock n_lock(node_lock);
      if (up)
      {
        std::vector<AddressSpaceID> children;
        if (!shard_rects_ready.exists())
        {
          // Not initialized, so do the initialization
          shard_rects_ready = Runtime::create_rt_user_event();
          // Add a reference to keep this node alive until this all done
          add_base_resource_ref(RUNTIME_REF);
#ifdef DEBUG_LEGION
          assert(collective_mapping != NULL);
#endif
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
          remaining_rect_notifications = children.size();
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
            if (children.empty())
              collective_mapping->get_children(owner_space, 
                                               local_space, children);
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
            return remove_base_resource_ref(RUNTIME_REF);
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
      }
      else
      {
        // Going down
        unpack_shard_rects(derez);
#ifdef DEBUG_LEGION
        assert(shard_rects_ready.exists());
        assert(collective_mapping != NULL);
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
        return remove_base_resource_ref(RUNTIME_REF);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    IndexPartNode::RemoteKDTracker::RemoteKDTracker(std::set<LegionColor> &c, 
                                                    Runtime *rt)
      : colors(c), runtime(rt), done_event(RtUserEvent::NO_RT_USER_EVENT),
        remaining(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::RemoteKDTracker::find_remote_interfering(
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
            return;
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
          return;
        done_event = Runtime::create_rt_user_event();
        wait_on = done_event;
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
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
        colors.insert(color);
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
      if (node->process_shard_rects_response(derez, source) &&
          node->remove_base_resource_ref(RUNTIME_REF))
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
    // Field Space Node 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx,
                   DistributedID did, RtEvent init, CollectiveMapping *map,
                   ShardMapping *shard_mapping, Provenance *prov)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC), 
          false/*register with runtime*/, map),
        handle(sp), context(ctx), provenance(prov), initialized(init), 
        allocation_state((shard_mapping != NULL) ? FIELD_ALLOC_COLLECTIVE :
            is_owner() ? FIELD_ALLOC_READ_ONLY : FIELD_ALLOC_INVALID), 
        outstanding_allocators(0), outstanding_invalidations(0)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        unallocated_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
        local_index_infos.resize(runtime->max_local_fields, 
            std::pair<size_t,CustomSerdezID>(0, 0));
        if (shard_mapping != NULL)
        {
          const ShardMapping &mapping = *shard_mapping;
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
         DistributedID did, RtEvent init, Provenance *prov, Deserializer &derez)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC), 
          false/*register with runtime*/),
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
      // Now make the instance, this should always succeed
      PhysicalManager *manager = attach_op->create_manager(node, field_set,
          field_sizes, mask_index_map, serdez, external_mask);
      return InstanceRef(manager, external_mask); 
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
      
      if (collective_mapping != NULL)
      {
        // Since we're the owner address space, record that we have 
        // instances on all other address spaces in the control
        // replicated parent task's collective mapping
        for (unsigned idx = 0; idx < collective_mapping->size(); idx++)
        {
          const AddressSpaceID space = (*collective_mapping)[idx];
          if (space == manager->owner_space)
            continue;
          manager->update_remote_instances(space);
        }
      }

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
          return (*it);
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
            delete layout;
            return (*it);
          }
        }
      }
      // Otherwise we successfully registered it
      descs.push_back(layout);
      layout->add_reference();
      return layout;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
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
      FieldSpaceNode *node =
        context->create_node(handle, did, initialized, provenance, derez);
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
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      FieldSpaceNode *target = forest->get_node(handle);
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
      if (is_owner())
      {
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
    bool RegionTreeNode::retrieve_semantic_information(SemanticTag tag,
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
    void RegionTreeNode::register_logical_user(ContextID ctx, 
                                       const LogicalUser &user,
                                       const RegionTreePath &path,
                                       const LogicalTraceInfo &trace_info,
                                       const ProjectionInfo &proj_info,
                                       FieldMask &unopened_field_mask,
                                       FieldMask &already_closed_mask,
                                       FieldMask &disjoint_complete_below,
                                       FieldMask &first_touch_refinement,
                                       FieldMaskSet<RefinementOp> &refinements,
                                       LogicalAnalysis &logical_analysis,
                                       const bool track_disjoint_complete_below,
                                       const bool check_unversioned)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_REGISTER_LOGICAL_USER_CALL);
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      const unsigned depth = get_depth();
      const bool arrived = !path.has_child(depth);
      FieldMask open_below, unversioned;
      if (check_unversioned)
      {
        unversioned = user.field_mask - state.disjoint_complete_tree;
        // We'll make this look like it has been versioned for now
        // and if we don't get a refinement to it, then we'll issue
        // a close operation at the end to initialize the equivalence
        // set at the root of the tree
        if (!!unversioned)
          state.disjoint_complete_tree |= unversioned;
      }
      RegionTreeNode *next_child = NULL;
      if (!arrived)
        next_child = get_tree_child(path.get_child(depth));
      // Now check to see if we need to do any close operations
      if (!!unopened_field_mask)
      {
        // Close up any children which we may have dependences on below
        const bool captures_closes = true;
        LogicalCloser closer(ctx, user, this, arrived/*validates*/); 
        // Special siphon operation for arrived projecting functions
        if (arrived && proj_info.is_projecting())
        {
          siphon_logical_projection(closer, state, unopened_field_mask,
                      proj_info, captures_closes, open_below);
        }
        else
        {
          const FieldMask *aliased_children = path.get_aliased_children(depth);
          siphon_logical_children(closer, state, unopened_field_mask, 
                                  aliased_children, captures_closes, next_child,
                                  open_below);
        }
        // We always need to create and register close operations
        // regardless of whether we are tracing or not
        // If we're not replaying a trace we need to do work here
        // See if we need to register a close operation
        if (closer.has_close_operations(already_closed_mask))
        {
          // Generate the close operations         
          // Also check to see if we have any refinements for the close operation
          const bool check_for_refinements = arrived && IS_WRITE(user.usage) 
              && !proj_info.is_projecting() && !trace_info.replaying_trace;
          closer.initialize_close_operations(state, user.op, trace_info,
                                             check_for_refinements, !arrived);
          // Perform dependence analysis for all the close operations
          closer.perform_dependence_analysis(user, open_below,
                                             state.curr_epoch_users,
                                             state.prev_epoch_users);
          // Note we don't need to update the version numbers because
          // that happened when we recorded dirty fields below. 
          // However, we do need to mark that there is no longer any
          // dirty data below this node for all the closed fields

          // Update the dirty_below and partial close fields
          // and filter the current and previous epochs
          // Also check to see if this close operation should change
          // the refinement of the equivalence sets when we do this
          // This is the "up" case of equivalence set refinement
          // where we coarsen the refinement to a higher level
          // in the region tree
          closer.update_state(state);
          // Now we can add the close operations to the current epoch
          closer.register_close_operations(state.curr_epoch_users);
        }
        // See if we have any open_only fields to merge
        if (!arrived || proj_info.is_projecting())
        {
          FieldMask open_only = user.field_mask - unopened_field_mask;
          if (!!open_only)
            add_open_field_state(state, arrived, proj_info, user, 
                                 open_only, next_child);
        }
      }
      else if (!arrived || proj_info.is_projecting())
      {
        // Everything is open-only so make a state and merge it in
        add_open_field_state(state, arrived, proj_info, user, 
                             user.field_mask, next_child);
      }
      // We also always do our dependence analysis even if we have
      // already traced because we need to pick up dependences on 
      // any dynamic open, advance, or close operations that we need to do
      // Now that we registered any close operation, do our analysis
      FieldMask dominator_mask = 
             perform_dependence_checks<CURR_LOGICAL_ALLOC,
                       true/*record*/,false/*has skip*/,true/*track dom*/>(
                          user, state.curr_epoch_users, user.field_mask, 
                          open_below, arrived/*validates*/ && 
                                        !proj_info.is_projecting());
      FieldMask non_dominated_mask = user.field_mask - dominator_mask;
      // For the fields that weren't dominated, we have to check
      // those fields against the previous epoch's users
      if (!!non_dominated_mask)
      {
        perform_dependence_checks<PREV_LOGICAL_ALLOC,
                      true/*record*/, false/*has skip*/, false/*track dom*/>(
                        user, state.prev_epoch_users, non_dominated_mask, 
                        open_below, arrived/*validates*/ && 
                                      !proj_info.is_projecting());
      }
      // If we dominated and this is our final destination then we 
      // can filter the operations since we actually do dominate them
      if (arrived && !!dominator_mask)
      {
        // Dominator mask is not empty
        // Mask off all the dominated fields from the previous set
        // of epoch users and remove any previous epoch users
        // that were totally dominated
        filter_prev_epoch_users(state, dominator_mask); 
        // Mask off all dominated fields from current epoch users and move
        // them to prev epoch users.  If all fields masked off, then remove
        // them from the list of current epoch users.
        filter_curr_epoch_users(state, dominator_mask, user.op->is_tracing());
      }
      if (arrived)
      { 
        // If we've arrived add ourselves as a user
        register_local_user(state, user, trace_info);
        if (!proj_info.is_projecting() && (user.usage.redop > 0))
        {
          // Not projecting and doing a reduction of some kind so record it
          record_logical_reduction(state, user.usage.redop, user.field_mask);
        }
        // Do the check for disjoint-complete writes/accesses if tracking
        if (track_disjoint_complete_below && !is_region())
        {
          PartitionNode *part_node = as_partition_node();
          IndexPartNode *index_part = part_node->row_source;
          if (index_part->is_disjoint() && index_part->is_complete())
          {
#ifdef DEBUG_LEGION
            assert(proj_info.is_projecting());
#endif
            // See if we are a depth 0 projection function
            if (proj_info.projection->depth > 0)
            {
              // Not depth 0, so we're projecting farther down the tree
              // See if we're writing since we know that all writes must
              // be disjoint and then we only need to ask the projection
              // function whether it will be a complete access
              // We also need this to be a functional projection functor
              // since we won't be able to evaluate it later otherwise
              if (IS_WRITE(user.usage) && proj_info.projection->is_functional)
              {
                // Check to see if there are any fields for us to refine
                disjoint_complete_below = user.field_mask;
                if (!!state.disjoint_complete_tree)
                  disjoint_complete_below -= state.disjoint_complete_tree;
                if (!!disjoint_complete_below)
                  disjoint_complete_below -= 
                    state.disjoint_complete_accesses.get_valid_mask();
                if (!!disjoint_complete_below)
                {
                  // If we're a first touch refinement then we don't need
                  // to do the check for completeness
                  FieldMask remainder = disjoint_complete_below;
                  const FieldMask first_touch_overlap =
                    disjoint_complete_below & first_touch_refinement;
                  if (!!first_touch_overlap)
                    remainder -= first_touch_overlap;
                  if (!!remainder)
                  {
                    if (proj_info.is_complete_projection(this, user))
                    {
                      // Record that we have a projection from this node
                      RefProjectionSummary *summary = 
                        new RefProjectionSummary(proj_info);
                      summary->add_reference();
                      state.disjoint_complete_projections.insert(summary,
                                                                 remainder);
                      // Relax fields so we don't get any future projections
                      state.disjoint_complete_accesses.relax_valid_mask(
                                                              remainder);
                    }
                    else
                      disjoint_complete_below -= remainder;
                  }
                }
              }
            }
            else if (proj_info.projection_space->get_volume() >=
                      ((index_part->get_num_children() + 1) / 2))
            {
              // If depth 0, see if we've got enough points for at least half
              // the subspaces, if so that is good enough for now
              disjoint_complete_below = user.field_mask;
              if (!!state.disjoint_complete_tree)
                disjoint_complete_below -= state.disjoint_complete_tree;
            }
            else if (!!first_touch_refinement)
            {
              disjoint_complete_below = 
                user.field_mask & first_touch_refinement;
              if (!!state.disjoint_complete_tree)
                disjoint_complete_below -= state.disjoint_complete_tree;
            }
          }
        }
        else if (track_disjoint_complete_below)
        {
          // Do this part regardless of whether we're projecting or not
          disjoint_complete_below = user.field_mask;
          if (!!state.disjoint_complete_tree)
            disjoint_complete_below -= state.disjoint_complete_tree;
          if (!!disjoint_complete_below && 
              proj_info.is_projecting() && (proj_info.projection->depth > 0))
          {
            // Not depth 0, so we're projection farther down the tree
            // Only support completeness for writes here
            // We also need this to be a functional projection functor
            // since we won't be able to evaluate it later otherwise
            if (IS_WRITE(user.usage) && proj_info.projection->is_functional)
            {
              // Check to see if there are any fields for us to refine
              disjoint_complete_below -= 
                state.disjoint_complete_accesses.get_valid_mask();
              if (!!disjoint_complete_below)
              {
                // If we're a first touch refinement then we don't need
                // to do the check for completeness
                FieldMask remainder = disjoint_complete_below;
                const FieldMask first_touch_overlap =
                  disjoint_complete_below & first_touch_refinement;
                if (!!first_touch_overlap)
                  remainder -= first_touch_overlap;
                if (!!remainder)
                {
                  if (proj_info.is_complete_projection(this, user))
                  {
                    // Record that we have a projection from this node
                    RefProjectionSummary *summary =
                      new RefProjectionSummary(proj_info);
                    summary->add_reference();
                    state.disjoint_complete_projections.insert(summary,
                                                               remainder);
                    // Relax fields so we don't get any future projections
                    state.disjoint_complete_accesses.relax_valid_mask(
                        remainder);
                  }
                  else
                    disjoint_complete_below -= remainder;
                }
              }
            }
            else
            {
              // We can keep any fields that are first touch
              if (!!first_touch_refinement)
                disjoint_complete_below &= first_touch_refinement;
              else
                disjoint_complete_below.clear();
            }
          }
        }
      }
      else 
      {
        // Haven't arrived, yet, so keep going down the tree
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
        // The first touch refinement tracks if we're traversing for
        // fields that have never been refined before and therefore
        // we don't need to apply our usual tests in sub-trees for 
        // when we should consider a disjoint complete sub-tree to 
        // have been accessed. We want to see them immediately.
        if (track_disjoint_complete_below && !!first_touch_refinement)
        {
          const FieldMask disjoint_complete_overlap = 
            first_touch_refinement & state.disjoint_complete_tree;
          if (!!disjoint_complete_overlap)
          {
            // Remove any fields which already have refinements
            if (!state.disjoint_complete_children.empty())
            {
              const FieldMask &child_mask =
                state.disjoint_complete_children.get_valid_mask();
#ifdef DEBUG_LEGION
              assert(!(child_mask - state.disjoint_complete_tree));
#endif
              first_touch_refinement -= child_mask;
            }
            if (!state.disjoint_complete_accesses.empty())
              first_touch_refinement -= (disjoint_complete_overlap &
                  state.disjoint_complete_accesses.get_valid_mask());
          }
        }
        FieldMask child_disjoint_complete;
        if (track_disjoint_complete_below && !is_region())
        {
          IndexPartNode *part_node = as_partition_node()->row_source;
          // Only continue tracking through disjoint and complete partitions
          if (part_node->is_disjoint() && part_node->is_complete())
            next_child->register_logical_user(ctx, user, path, trace_info,
                     proj_info, unopened_field_mask, already_closed_mask, 
                     child_disjoint_complete, first_touch_refinement,
                     refinements, logical_analysis,
                     true/*track disjoint complete below*/, 
                     false/*check unversioned*/);
          else
            next_child->register_logical_user(ctx, user, path, trace_info,
                     proj_info, unopened_field_mask, already_closed_mask, 
                     child_disjoint_complete, first_touch_refinement,
                     refinements, logical_analysis,
                     false/*track disjoint complete below*/, 
                     false/*check unversioned*/);
        }
        else
          next_child->register_logical_user(ctx, user, path, trace_info,
             proj_info, unopened_field_mask, already_closed_mask, 
             child_disjoint_complete, first_touch_refinement,
             refinements, logical_analysis,
             track_disjoint_complete_below, false/*check unversioned*/);
        if (!refinements.empty() &&
            (!state.curr_epoch_users.empty() || 
             !state.prev_epoch_users.empty()))
        {
          // perform dependence analysis for any refinements done
          // below for users at this level
          Operation *skip_op = user.op;
          const GenerationID skip_gen = skip_op->get_generation();
          for (FieldMaskSet<RefinementOp>::const_iterator it = 
                refinements.begin(); it != refinements.end(); it++)
          {
            const LogicalUser refinement_user(it->first, 0/*index*/,
                RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), it->second);
            perform_dependence_checks<CURR_LOGICAL_ALLOC,
              false/*record*/, true/*has skip*/, false/*track dominance*/>(
                  refinement_user, state.curr_epoch_users, it->second,
                  it->second, false/*validates*/, skip_op, skip_gen);
            perform_dependence_checks<PREV_LOGICAL_ALLOC,
              false/*record*/, true/*has skip*/, false/*track dominance*/>(
                  refinement_user, state.prev_epoch_users, it->second,
                  it->second, false/*validates*/, skip_op, skip_gen);
          }
        }
        // If we're a region and next_child is the current refinement then prune
        // out any disjoint_complete_accesses and disjoint_complete_child_counts
        // because we saw the current refinement again
        // Do this before changing any refinements below to avoid doing this
        // again even though we just refined
        if (is_region() && !!state.disjoint_complete_tree &&
            !state.disjoint_complete_accesses.empty())
        {
          FieldMaskSet<RegionTreeNode>::const_iterator finder =
            state.disjoint_complete_children.find(next_child);
          if (finder != state.disjoint_complete_children.end())
          {
            const FieldMask reset_mask = user.field_mask & finder->second & 
                          state.disjoint_complete_accesses.get_valid_mask();
            if (!!reset_mask)
            {
#ifdef DEBUG_LEGION
              assert(reset_mask * child_disjoint_complete); 
#endif
              // Prune the counts
              for (LogicalState::FieldSizeMap::iterator it = 
                    state.disjoint_complete_child_counts.begin(); it !=
                    state.disjoint_complete_child_counts.end(); /*nothing*/)
              {
                it->second -= reset_mask;
                if (!it->second)
                {
                  LogicalState::FieldSizeMap::iterator to_delete = it++;
                  state.disjoint_complete_child_counts.erase(to_delete);
                }
                else
                  it++;
              }
              // And the previous accesses
              state.disjoint_complete_accesses.filter(reset_mask);
            }
          }
        }
        // Everything in this block of code is for computing any
        // refinements that need to be performed because we wrote
        // to a disjoint complete sub-tree
        if (!!child_disjoint_complete && !trace_info.replaying_trace)
        {
          if (is_region())
          {
            // Check to see which of these fields should trigger
            // a refinement and which should flow back up the tree
            if (!!state.disjoint_complete_tree)
            {
              // If we're writing, then we're going to refine any fields
              // which are already disjoint complete at this level
              FieldMask refinement_mask = child_disjoint_complete &
                state.disjoint_complete_tree;
              if (!!refinement_mask)
              {
                // these do no need to keep going up the tree
                child_disjoint_complete -= refinement_mask;
                // If we're refined for any overlapping fields with 
                // the disjoint+complete child, then there are two cases:
                // 1. We don't overlap with any existing disjoint+complete
                //    child refinements, in which case we're going to refine
                //    right away to maximize analysis parallelism
                FieldMask previously_refined = refinement_mask & 
                  state.disjoint_complete_children.get_valid_mask();
                if (!!previously_refined)
                {
                  // 2. We overlap with some existing children that are refined
                  //    in which case we need to decide if we want to switch to
                  //    a new child for the refinement. Switching refinements
                  //    is expensive, so we only want to switch once we've seen
                  //    some compelling evidence that it is worth it to do so
                  //    Right now we have two criteria for deciding to switch
                  //    a. If we see LEGION_REFINEMENT_SAME_CHILD consecutive
                  //       accesses to the same child we switch
                  //    b. If we see LEGION_REFINEMENT_DIFF_CHILD consecutive
                  //       accesses to any child but the current refinement
                  //       child then we also switch to this child
                  static_assert(LEGION_REFINEMENT_SAME_CHILD > 0,
                      "LEGION_REFINEMENT_SAME_CHILD must be positive");
                  static_assert(LEGION_REFINEMENT_DIFF_CHILD > 0,
                      "LEGION_REFINEMENT_DIFF_CHILD must be positive");
#ifdef DEBUG_LEGION
                  // The child should not already be considered refined for
                  // any of these fields
                  assert((state.disjoint_complete_children.find(next_child) ==
                          state.disjoint_complete_children.end()) ||
                         (state.disjoint_complete_children[next_child] * 
                          previously_refined));
#endif
                  // First check to see if we have enough consecutive accesses
                  // to the next_child which will allow us to switch now, if
                  // not update the counts for ourselves
                  FieldMaskSet<RegionTreeNode>::iterator finder = 
                    state.disjoint_complete_accesses.find(next_child);
                  if (finder != state.disjoint_complete_accesses.end())
                  {
                    FieldMask consecutive = previously_refined & finder->second;
                    if (!!consecutive)
                    {
                      // we've handled these fields now from previously refined
                      previously_refined -= consecutive;
                      FieldMask perform_mask;
                      for (LogicalState::FieldSizeMap::iterator it = 
                            state.disjoint_complete_child_counts.begin(); it !=
                            state.disjoint_complete_child_counts.end();/*none*/)
                      {
                        const FieldMask overlap = consecutive & it->second;
                        if (!overlap)
                        {
                          it++;
                          continue;
                        }
                        size_t count = it->first / 2;
                        const bool same_child = ((it->first % 2) == 0);
                        if ((same_child &&
                              ((count + 1) == LEGION_REFINEMENT_SAME_CHILD)) ||
                            (!same_child &&
                              ((count + 1) == LEGION_REFINEMENT_DIFF_CHILD)))
                        {
                          // We're doing the refinement now
                          perform_mask |= overlap; 
                          // Don't care about these fields anymore since
                          // we've already decided at this point to refine them
                          consecutive -= overlap;
                        }
                        else
                        {
                          // Not refining yet if this is the same child count
                          // If this is the diff child count we're still going
                          // to see the same child count later so we can't
                          // say we're not doing the refinement yet or not
                          if (same_child)
                            refinement_mask -= overlap;
                          // Increment the count and rescale
                          count = (count + 1) * 2 + (it->first % 2);
                          LogicalState::FieldSizeMap::iterator count_finder =
                            state.disjoint_complete_child_counts.find(count);
                          if (count_finder != 
                              state.disjoint_complete_child_counts.end())
                            count_finder->second |= overlap;
                          else
                            state.disjoint_complete_child_counts.insert(
                                std::make_pair(count, overlap));
                        }
                        it->second -= overlap;
                        if (!it->second)
                        {
                          LogicalState::FieldSizeMap::iterator to_delete = it++;
                          state.disjoint_complete_child_counts.erase(to_delete);
                        }
                        else
                          it++;
                        // diff child count > same child count
                        // therefore once we've seen the same child
                        // count for some fields we know we won't see
                        // any counts for those fields again
                        if (same_child)
                          consecutive -= overlap;
                        if (!consecutive)
                          break;
                      }
                      if (!!perform_mask)
                      {
                        // These are fields for which we are performing
                        // the refinements so clean out all meta-data
                        // Remove ourselves from disjoint_complete_accesses
                        finder.filter(perform_mask);
                        state.disjoint_complete_accesses.filter_valid_mask(
                                                              perform_mask);
                        if (!finder->second)
                          state.disjoint_complete_accesses.erase(finder);
                        for (LogicalState::FieldSizeMap::iterator it =
                             state.disjoint_complete_child_counts.begin(); it !=
                             state.disjoint_complete_child_counts.end(); /*no*/)
                        {
                          it->second -= perform_mask;
                          if (!it->second)
                          {
                            LogicalState::FieldSizeMap::iterator 
                              to_delete = it++;
                            state.disjoint_complete_child_counts.erase(
                                                              to_delete);
                          }
                          else
                            it++;
                        }
                      }
                    }
                  }
                  // We've handled all the fields for which next_child was
                  // already in disjoint_complete_accesses, now see if there
                  // are any remaming fields for which next_child is not
                  // the most recent access and update accordingly
                  if (!!previously_refined)
                  {
                    // First we can remove all disjoint complete accesses
                    // that do not agree with next_child
                    state.disjoint_complete_accesses.filter(previously_refined);
                    // Now we can go through and update the counts
                    FieldMask unrefined, perform_refinement;
                    for (LogicalState::FieldSizeMap::iterator it =
                          state.disjoint_complete_child_counts.begin(); it !=
                          state.disjoint_complete_child_counts.end(); /*none*/)
                    {
                      const FieldMask overlap = previously_refined & it->second;
                      if (!overlap)
                      {
                        it++;
                        continue;
                      }
                      size_t count = it->first / 2;
                      const bool same_child = ((it->first % 2) == 0);
                      if (same_child)
                      {
                        // see if we've already seen a diff child count
                        // if not, then we can increment this and make
                        // this the new consecutive diff child count
                        FieldMask no_diff = overlap - unrefined;
                        // do no update counts if we're refining these fields
                        if (!!perform_refinement)
                          no_diff -= perform_refinement;
                        if (!!no_diff)
                        {
                          // switch to diff child count with odd numbers
                          count = (count + 1) * 2 + 1;
                          LogicalState::FieldSizeMap::iterator count_finder =
                            state.disjoint_complete_child_counts.find(count);
                          if (count_finder !=
                              state.disjoint_complete_child_counts.end())
                            count_finder->second |= no_diff;
                          else
                            state.disjoint_complete_child_counts.insert(
                                std::make_pair(count, no_diff));
                          unrefined |= no_diff;
                        }
                        // we know there will be no more counts for these fields
                        previously_refined -= overlap;
                      }
                      else
                      {
                        // Check to see if we can perform a refinement
                        if ((count + 1) != LEGION_REFINEMENT_DIFF_CHILD)
                        {
                          // No refinement yet
                          // update the count for the future
                          // switch to diff child count with odd numbers
                          count = (count + 1) * 2 + 1; 
                          LogicalState::FieldSizeMap::iterator count_finder =
                            state.disjoint_complete_child_counts.find(count);
                          if (count_finder !=
                              state.disjoint_complete_child_counts.end())
                            count_finder->second |= overlap;
                          else
                            state.disjoint_complete_child_counts.insert(
                                std::make_pair(count, overlap));
                          unrefined |= overlap;
                        }
                        else 
                          // doing the refinement
                          // keep going to prune out the other counts
                          perform_refinement |= overlap;
                      }
                      it->second -= overlap;
                      if (!it->second)
                      {
                        LogicalState::FieldSizeMap::iterator to_delete = it++;
                        state.disjoint_complete_child_counts.erase(to_delete);
                      }
                      else
                        it++; 
                      if (!previously_refined)
                        break;
                    }
                    // Any fields we didn't observe before need to go into
                    // the initial unrefined case
                    if (!!previously_refined)
                      unrefined |= previously_refined;
#ifdef DEBUG_LEGION
                    assert(perform_refinement * unrefined);
#endif
                    if (!!unrefined)
                    {
                      // Not performing refinements for these fields
                      refinement_mask -= unrefined;
                      // update disjoint_complete_accesses and
                      // disjoint_complete_counts with next_child
                      state.disjoint_complete_accesses.insert(
                          next_child, unrefined);
                      const size_t count = 2; // (count=1 * 2)
                      LogicalState::FieldSizeMap::iterator count_finder =
                        state.disjoint_complete_child_counts.find(count);
                      if (count_finder !=
                          state.disjoint_complete_child_counts.end())
                        count_finder->second |= unrefined;
                      else
                        state.disjoint_complete_child_counts.insert(
                            std::make_pair(count, unrefined));
                    }
                  }
                }
              }
              if (!!refinement_mask)
              {
                // This is both the "across" and the "down" case of
                // equivalence set refinement where we make a change
                // in the equivalence set refinement to a new tree
                // or further refine the equivalence sets into smaller
                // equivalence sets going down the region tree.
                child_disjoint_complete -= refinement_mask;
                PartitionNode *part_child = next_child->as_partition_node();
                // Check to see if we have already have a refinement
                if (logical_analysis.deduplicate(part_child, refinement_mask))
                {
                  // Create a refinement operation
                  RefinementOp *refinement_op = 
                    logical_analysis.create_refinement(user, part_child, 
                                              refinement_mask, trace_info);
                  // We can't modify the disjoint complete tree yet because
                  // we need to wait for all the region requirements to be
                  // traversed before that to see if any other ones want
                  // to contribute to this refineemnt. We can do the logical
                  // dependence analysis though because we know that all later
                  // region requirements are going to ignore this refinement
                  // Perform the dependence analysis for all open sub-trees
                  // to make sure we record dependences on any operations
                  // we need to before performing this refinement
                  const LogicalUser refinement_user(refinement_op, 0/*index*/,
                    RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/),
                    refinement_mask);
                  // Start the dependence analysis here, the initial caller of
                  // register_logical_user will be the function that finishes
                  // it for each of the refinement operations
                  perform_tree_dominance_analysis(ctx, refinement_user, 
                        refinement_mask, user.op/*skip op*/, user.gen);
                  // Register the refinement as an operation here
                  register_local_user(state, refinement_user, trace_info); 
#ifdef DEBUG_LEGION
                  assert(refinement_mask * refinements.get_valid_mask()); 
#endif
                  // Record this for going up the tree to catch dependences
                  // on anything above us
                  refinements.insert(refinement_op, refinement_mask);
                }
              }
            }
            if (!!child_disjoint_complete)
            {
#ifdef DEBUG_LEGION
              assert(child_disjoint_complete * state.disjoint_complete_tree);
#endif
              // There's no point in remembering anything here, just keep
              // passing it up the tree where the state is stored once we
              // get back to the disjoint complete tree
              disjoint_complete_below = child_disjoint_complete;
              FieldMaskSet<RegionTreeNode> &access_disjoint_complete_children =
                state.disjoint_complete_accesses;
              // We do need to remember what was the most recent child though
              // to propagate this information up the tree
              FieldMaskSet<RegionTreeNode>::iterator finder = 
                access_disjoint_complete_children.find(next_child);
              if (finder != access_disjoint_complete_children.end())
              {
                child_disjoint_complete -= finder->second;
                if (!!child_disjoint_complete)
                  finder.merge(child_disjoint_complete);
              }
              else
                access_disjoint_complete_children.insert(next_child, 
                                            child_disjoint_complete);
              if (!!child_disjoint_complete)
              {
                // Filter out any other children for the fields we added
                std::vector<RegionTreeNode*> to_delete;
                for (FieldMaskSet<RegionTreeNode>::iterator it =
                      access_disjoint_complete_children.begin(); it !=
                      access_disjoint_complete_children.end(); it++)
                {
                  if (it->first == next_child)
                    continue;
                  it.filter(child_disjoint_complete);
                  if (!it->second)
                    to_delete.push_back(it->first);
                }
                if (!to_delete.empty())
                  for (std::vector<RegionTreeNode*>::const_iterator it =
                        to_delete.begin(); it != to_delete.end(); it++)
                    access_disjoint_complete_children.erase(*it);
              }
            }
          }
          else if (!!child_disjoint_complete)
          {
            // At this point we're a partition that is not refined
            IndexPartNode *part_node = as_partition_node()->row_source; 
#ifdef DEBUG_LEGION
            assert(part_node->is_disjoint() && part_node->is_complete());
#endif
            // Filter out any additional fields for which we are
            // already disjoint and complete but just didn't refine
            // for the children because we were the lowest level partition
            if (!!state.disjoint_complete_tree)
              child_disjoint_complete -= state.disjoint_complete_tree;
            if (!!child_disjoint_complete)
            {
              // Check to see if we've already been counted
              FieldMaskSet<RegionTreeNode>::iterator finder =
                state.disjoint_complete_accesses.find(next_child);
              if (finder != state.disjoint_complete_accesses.end())
              {
                child_disjoint_complete -= finder->second;
                if (!!child_disjoint_complete)
                  finder.merge(child_disjoint_complete);
              }
              else
                state.disjoint_complete_accesses.insert(next_child,
                                          child_disjoint_complete);
            }
            // Check to see if this a first-touch refinement in which
            // case we can safely ignore the test for the number of
            // children and propagate the information up immediately
            if (!!first_touch_refinement)
            {
              const FieldMask first_touch_overlap =
                child_disjoint_complete & first_touch_refinement;
              if (!!first_touch_overlap)
              {
                disjoint_complete_below |= first_touch_overlap;
                // filter out the individual children
                state.disjoint_complete_accesses.filter(first_touch_overlap);
                // but then relax the mask for these fields so that when
                // we go to do the refinement update we know we need to
                // traverse down below for those fields
                state.disjoint_complete_accesses.relax_valid_mask(
                                              first_touch_overlap);
                // No more need to consider these children anymore
                child_disjoint_complete -= first_touch_overlap;
              }
            }
            if (!!child_disjoint_complete)
            {
              // Scan through and update the counts, if we hit
              // the thresholds then we can count this as refined
              for (LogicalState::FieldSizeMap::iterator it =
                    state.disjoint_complete_child_counts.begin(); it !=
                    state.disjoint_complete_child_counts.end(); /*nothing*/)
              {
                const FieldMask overlap = 
                  child_disjoint_complete & it->second;
                if (!overlap)
                {
                  it++;
                  continue;
                }
                // Now see if we're done with this one
                const size_t next_count = it->first + 1;
                it->second -= overlap;
                if (!it->second)
                {
                  // Remove it from the set
                  LogicalState::FieldSizeMap::iterator to_delete = it++;
                  state.disjoint_complete_child_counts.erase(to_delete);
                }
                else
                  it++;
                // Add one for the overlap fields, will not break iterator
                // since it is going to be behind our traversal
                LogicalState::FieldSizeMap::iterator finder =
                  state.disjoint_complete_child_counts.find(next_count);
                if (finder != state.disjoint_complete_child_counts.end())
                  finder->second |= overlap;
                else
                  state.disjoint_complete_child_counts[next_count] = overlap;
                child_disjoint_complete -= overlap;
                if (!child_disjoint_complete)
                  break;
              }
              if (!!child_disjoint_complete)
                state.disjoint_complete_child_counts[1] = 
                                            child_disjoint_complete;
              // Only the first count can be big enough to exceed the
              // threshold so check to see if it is big enough
              LogicalState::FieldSizeMap::iterator first =
                state.disjoint_complete_child_counts.begin();
              // This is a heuristic here!
              // We're looking for the count to grow to be at least as
              // big as half of the children, we could change this later
              static_assert(LEGION_REFINEMENT_PARTITION_PERCENTAGE > 0,
                  "LEGION_REFINEMENT_PARTITION_PCT must be positive");
              static_assert(LEGION_REFINEMENT_PARTITION_PERCENTAGE <= 100,
                  "LEGION_REFINEMENT_PARTITION_PCT must be a percentage");
              // x/y >= w/z is the same as xz >= wy when all positive
              if ((100 * first->first) >= (part_node->get_num_children() * 
                    LEGION_REFINEMENT_PARTITION_PERCENTAGE))
              {
                disjoint_complete_below |= first->second;
                // Go through and filter out any child counts for these fields
                state.disjoint_complete_child_counts.erase(first);
                // filter out the individual children
                state.disjoint_complete_accesses.filter(
                                disjoint_complete_below);
                // but then relax the mask for these fields so that when
                // we go to do the refinement update we know we need to
                // traverse down below for those fields
                state.disjoint_complete_accesses.relax_valid_mask(
                                          disjoint_complete_below);
              }
            }
          }
        }
      }
      // Check to see if we have any unversioned fields we need to initialize
      if (check_unversioned && !!unversioned)
      {
#ifdef DEBUG_LEGION
        assert(!trace_info.replaying_trace);
#endif
        // See if we made refinements for any of these fields, if so
        // they will make the initial batch of equivalence sets
        if (!refinements.empty())
        {
          for (FieldMaskSet<RefinementOp>::const_iterator it = 
                refinements.begin(); it != refinements.end(); it++)
          {
            const FieldMask overlap = unversioned & it->second;
            if (!overlap)
              continue;
            it->first->record_uninitialized(overlap);
          }
          unversioned -= refinements.get_valid_mask();
        }
        if (!!unversioned)
        {
          // If we have unversioned fields and no refinements were
          // made for them, then we make a close op (which doesn't actually
          // close anything) to create the first equivalence set for these
          // fields here at the root of the tree
          InnerContext *context = user.op->get_context();
#ifdef DEBUG_LEGION_COLLECTIVES
          MergeCloseOp *initializer = context->get_merge_close_op(user, this);
#else
          MergeCloseOp *initializer = context->get_merge_close_op();
#endif
#ifdef DEBUG_LEGION
          assert(is_region());
#endif
          RegionNode *region_node = as_region_node();
          RegionRequirement req(region_node->handle, LEGION_WRITE_DISCARD,
              LEGION_EXCLUSIVE, region_node->handle);
          region_node->column_source->get_field_set(unversioned,
              trace_info.req.privilege_fields, req.privilege_fields);
          initializer->initialize(context, req, trace_info, trace_info.req_idx,
                                  unversioned, user.op);
          initializer->record_refinements(unversioned, true/*overwrite*/);
          // These fields are unversioned so there is nothing for 
          // this close operation to depend on
          const GenerationID initializer_gen = initializer->get_generation();
#ifdef LEGION_SPY
          const UniqueID initializer_uid = initializer->get_unique_op_id();
#endif
          initializer->execute_dependence_analysis();
          // Make sure our operation has a dependence on this initializer
          if (!user.op->register_region_dependence(user.idx, initializer,
                initializer_gen, 0/*target index*/, LEGION_TRUE_DEPENDENCE,
                false/*validates*/, unversioned))
          {
            const LogicalUser init_user(initializer, initializer_gen, 0/*idx*/,
                                        RegionUsage(req), unversioned);
            register_local_user(state, init_user, trace_info);
          }
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
            user.op->get_context()->get_unique_id(), initializer_uid, 0/*idx*/,
            user.op->get_unique_op_id(), user.idx, LEGION_TRUE_DEPENDENCE);
#endif
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_local_user(LogicalState &state,
                                             const LogicalUser &user,
                                             const LogicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      // Here is the only difference with tracing.  If we already
      // traced then we don't need to register ourselves as a user
      if (!trace_info.already_traced)
      {
        // Register ourself with as a current user of this region
        // Record a mapping reference on this operation
        user.op->add_mapping_reference(user.gen);
        // Add ourselves to the current epoch
        state.curr_epoch_users.push_back(user);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::add_open_field_state(LogicalState &state, bool arrived,
                                              const ProjectionInfo &proj_info,
                                              const LogicalUser &user,
                                              const FieldMask &open_mask,
                                              RegionTreeNode *next_child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      if (arrived && proj_info.is_projecting())
      {
        FieldState new_state(user.usage, open_mask, proj_info.projection,
           proj_info.projection_space, proj_info.sharding_function, 
           proj_info.sharding_space, this);
        merge_new_field_state(state, new_state);
      }
      else if (next_child != NULL)
      {
        FieldState new_state(user, open_mask, next_child);
        merge_new_field_state(state, new_state);
      }
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_logical_node(LogicalCloser &closer,
                                            const FieldMask &closing_mask,
                                            const bool read_only_close)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_CLOSE_LOGICAL_NODE_CALL);
      LogicalState &state = get_logical_state(closer.ctx);
      // Perform closing checks on both the current epoch users
      // as well as the previous epoch users
      perform_closing_checks<CURR_LOGICAL_ALLOC>(closer, state.curr_epoch_users,
                                                 closing_mask);
      perform_closing_checks<PREV_LOGICAL_ALLOC>(closer, state.prev_epoch_users,
                                                 closing_mask);
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
          if (it->is_projection_state())
          {
            // If this was a projection field state then we need to 
            // advance the epoch version numbers
            // If this is a writing or reducing 
            it->filter(overlap);
          }
          else
          {
            // Recursively perform any close operations
            FieldMask already_open;
            perform_close_operations(closer, overlap, *it,
                                     NULL/*next child*/,
                                     false/*allow next*/,
                                     NULL/*aliased children*/,
                                     false/*upgrade*/,
                                     read_only_close,
                                     false/*overwiting close*/,
                                     false/*record close operations*/,
                                     false/*record closed fields*/,
                                     already_open);
          }
          // Remove the state if it is now empty
          if (!it->valid_fields())
            it = state.field_states.erase(it);
          else
            it++;
        }
      }
      // No children, so nothing to do
      // We can clear out our reduction fields
      if (!!state.reduction_fields)
        state.reduction_fields -= closing_mask;
      // We can also clear any outstanding reduction fields
      if (!(state.reduction_fields * closing_mask))
        clear_logical_reduction_fields(state, closing_mask);
      // Clear out any written disjoint complete children too
      filter_disjoint_complete_accesses(state, closing_mask);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_logical_children(LogicalCloser &closer,
                                              LogicalState &state,
                                              const FieldMask &current_mask,
                                              const FieldMask *aliased_children,
                                              bool record_close_operations,
                                              RegionTreeNode *next_child,
                                              FieldMask &open_below)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_SIPHON_LOGICAL_CHILDREN_CALL);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      LegionDeque<FieldState> new_states;
      // Before looking at any child states, first check to see if we need
      // to do any closes to flush open reductions. This should be a pretty
      // rare operation since we often won't have lots of reductions going
      // on at different levels of the region tree.
      if (!!state.reduction_fields)
      {
        FieldMask reduction_flush_fields = 
          current_mask & state.reduction_fields;
        if (!!reduction_flush_fields)
          flush_logical_reductions(closer, state, reduction_flush_fields,
                         record_close_operations, next_child, new_states);
      }
      // If we are overwriting then we don't really care about what dirty
      // data is in a given sub-tree since it isn't going to matter
      // BE CAREFUL: if this is predicated operation then we can't
      // necessarily assume it will be overwritten, in which case we
      // need a full close operation
      // In the future we might consider breaking this out so that we
      // generate two close operations: a read-only one for the predicate
      // true case, and normal close operation for the predicate false case
      const bool overwriting = HAS_WRITE_DISCARD(closer.user.usage) && 
          (next_child == NULL) && !closer.user.op->is_predicated_op();
      // Now we can look at all the children
      for (LegionList<FieldState>::iterator it = 
            state.field_states.begin(); it != 
            state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields() * current_mask)
        {
          it++;
          continue;
        }
        // Now check the current state
        switch (it->open_state)
        {
          case OPEN_READ_ONLY:
            {
              if (IS_READ_ONLY(closer.user.usage))
              {
                // Everything is read-only
                // See if the child that we want is already open
                if (next_child != NULL)
                {
                  FieldMaskSet<RegionTreeNode>::const_iterator finder =
                    it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Remove the child's open fields from the
                    // list of fields we need to open
                    open_below |= finder->second;
                  }
                }
                // Check to see if we are tracing, if we are then 
                // we need to traverse all the children open in read-only
                // mode and record "no-dependences" on any users we find
                // down there in case we need to inject an internal operation
                // later when we go to replay the trace
                if (closer.tracing &&
                    ((next_child == NULL) || !are_all_children_disjoint()))
                {
                  for (FieldMaskSet<RegionTreeNode>::const_iterator cit =
                        it->open_children.begin(); cit != 
                        it->open_children.end(); cit++)
                  {
                    // Can skip the next node since we're going to traverse
                    // it later anyway
                    if (cit->first == next_child)
                      continue;
                    if (cit->second * current_mask)
                      continue;
                    if ((next_child != NULL) && 
                        are_children_disjoint(cit->first->get_color(),
                                              next_child->get_color()))
                      continue;
                    cit->first->record_close_no_dependences(closer.ctx, 
                                                            closer.user);
                  }
                }
                it++;
              }
              else
              {
                // Not read-only
                // Close up all the open partitions except the one
                // we want to go down, make a new state to be added
                // containing the fields that are still open and mark
                // that we need an upgrade from read-only to some
                // kind of write operation. Note that we don't need to
                // actually perform close operations here because closing
                // read-only children requires no work.
                FieldMask already_open;
                perform_close_operations(closer, current_mask, 
                                         *it, next_child,
                                         false/*allow next*/,
                                         aliased_children,
                                         true/*needs upgrade*/,
                                         true/*read only close*/,
                                         false/*overwriting close*/,
                                         record_close_operations,
                                         false/*record closed fields*/,
                                         already_open);
                if (!!already_open)
                {
                  open_below |= already_open;
                  FieldState new_state(closer.user, already_open, next_child);
                  new_states.emplace_back(std::move(new_state));
                }
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
              // Close up any open partitions that conflict with ours
              perform_close_operations(closer, current_mask, 
                                       *it, next_child,
                                       true/*allow next*/,
                                       aliased_children,
                                       false/*needs upgrade*/,
                                       false/*read only close*/,
                                       overwriting/*overwriting close*/,
                                       record_close_operations,
                                       false/*record closed fields*/,
                                       open_below);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_SINGLE_REDUCE:
            {
              // Check to see if we have a child we want to go down
              if (next_child != NULL)
              {
                // There are four cases here:
                //   1. Same reduction, same child -> everything stays the same
                //   2. Same reduction, different child -> go to MULTI_REDUCE
                //   3. Diff operation, same child -> go to READ_WRITE
                //   4. Diff operation, diff child -> close everything up
                if (IS_REDUCE(closer.user.usage) && 
                    (it->redop == closer.user.usage.redop))
                {
                  // Cases 1 and 2
                  bool needs_recompute = false;
                  std::vector<RegionTreeNode*> to_delete;
                  // If we're tracing we need to record nodep dependences
                  // here in any aliased sub-trees in case we need to make
                  // internal operations later when replaying the trace
                  const bool tracing = closer.tracing;
                  // Go through all the children and see if there is any overlap
                  for (FieldMaskSet<RegionTreeNode>::iterator cit = 
                        it->open_children.begin(); cit !=
                        it->open_children.end(); cit++)
                  {
                    FieldMask already_open = cit->second & current_mask;
                    // If disjoint children, nothing to do
                    if (!already_open || 
                        are_children_disjoint(cit->first->get_color(), 
                                              next_child->get_color()))
                      continue;
                    // Case 2
                    if (cit->first != (next_child))
                    {
                      if (tracing)
                        cit->first->record_close_no_dependences(closer.ctx,
                                                                closer.user);
                      // Different child so we need to create a new
                      // FieldState in MULTI_REDUCE mode with two
                      // children open
                      FieldState new_state(closer.user, already_open,
                                           cit->first);
                      // Add the next child as well
                      new_state.add_child(next_child, already_open); 
                      new_state.open_state = OPEN_MULTI_REDUCE;
#ifdef DEBUG_LEGION
                      assert(!!new_state.valid_fields());
#endif
                      new_states.emplace_back(std::move(new_state));
                      // Update the current child, mark that we need to
                      // recompute the valid fields for the state
                      cit.filter(already_open);
                      if (!cit->second)
                        to_delete.push_back(cit->first);
                      needs_recompute = true;
                    }
                    else // Case 1: same child so they are already open
                      open_below |= already_open;
                    // Otherwise same child so case 1 and everything just
                    // stays in SINGLE_REDUCE_MODE
                  }
                  // See if we need to recompute any properties
                  // of the current state to see if they are still valid
                  if (needs_recompute)
                  {
                    // Remove all the empty children
                    for (std::vector<RegionTreeNode*>::const_iterator cit =
                          to_delete.begin(); cit != to_delete.end(); cit++)
                      it->remove_child(*cit);
                    // Then recompute the valid mask for the current state
                    it->open_children.tighten_valid_mask();
                  }
                }
                else
                {
                  // Cases 3 and 4
                  FieldMask already_open;
                  perform_close_operations(closer, current_mask, 
                                           *it, next_child,
                                           false/*allow next*/,
                                           aliased_children,
                                           true/*needs upgrade*/,
                                           false/*read only close*/,
                                           false/*overwriting close*/,
                                           record_close_operations,
                                           false/*record closed fields*/,
                                           already_open);
                  open_below |= already_open;
                  if (!!already_open)
                  {
                    // Create a new FieldState open in whatever mode is
                    // appropriate based on the usage
                    FieldState new_state(closer.user, already_open, 
                                         next_child);
                    // We always have to go to read-write mode here
                    new_state.open_state = OPEN_READ_WRITE;
                    new_states.emplace_back(std::move(new_state));
                  }
                }
              }
              else if (IS_REDUCE(closer.user.usage) && 
                    (it->redop == closer.user.usage.redop))
              {
                // Same reduction as our children so stay in this mode
                // Check to see if we are tracing, if we are then 
                // we need to traverse all the children open in reduce
                // mode and record "no-dependences" on any users we find
                // down there in case we need to inject an internal operation
                // later when we go to replay the trace
                if (closer.tracing)
                {
                  for (FieldMaskSet<RegionTreeNode>::const_iterator cit =
                        it->open_children.begin(); cit != 
                        it->open_children.end(); cit++)
                  {
                    if (cit->second * current_mask)
                      continue;
                    cit->first->record_close_no_dependences(closer.ctx, 
                                                            closer.user);
                  }
                }
              }
              else
              {
                // Closing everything up, so just do it
                FieldMask already_open;
                perform_close_operations(closer, current_mask, 
                                         *it, next_child,
                                         false/*allow next*/,
                                         NULL/*aliased children*/,
                                         false/*needs upgrade*/,
                                         false/*read only close*/,
                                         overwriting/*overwriting close*/,
                                         record_close_operations,
                                         false/*record closed fields*/,
                                         already_open);
                open_below |= already_open;
              }
              // Now see if the current field state is still valid
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_MULTI_REDUCE:
            {
              // See if this reduction is a reduction of the same kind
              if (IS_REDUCE(closer.user.usage) &&
                  (closer.user.usage.redop == it->redop))
              {
                if (next_child != NULL)
                {
                  FieldMaskSet<RegionTreeNode>::const_iterator finder = 
                    it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Already open, so add the open fields
                    open_below |= (finder->second & current_mask);
                  }
                }
                // Check to see if we are tracing, if we are then 
                // we need to traverse all the children open in reduce
                // mode and record "no-dependences" on any users we find
                // down there in case we need to inject an internal operation
                // later when we go to replay the trace
                if (closer.tracing &&
                    ((next_child == NULL) || !are_all_children_disjoint()))
                {
                  for (FieldMaskSet<RegionTreeNode>::const_iterator cit =
                        it->open_children.begin(); cit != 
                        it->open_children.end(); cit++)
                  {
                    // Can skip the next node since we're going to traverse
                    // it later anyway
                    if (cit->first == next_child)
                      continue;
                    if (cit->second * current_mask)
                      continue;
                    if ((next_child != NULL) && 
                        are_children_disjoint(cit->first->get_color(),
                                              next_child->get_color()))
                      continue;
                    cit->first->record_close_no_dependences(closer.ctx, 
                                                            closer.user);
                  }
                }
                it++;
              }
              else
              {
                // Need to close up the open field since we're going
                // to have to do it anyway
                FieldMask already_open;
                perform_close_operations(closer, current_mask, 
                                         *it, next_child,
                                         false/*allow next*/,
                                         NULL/*aliased children*/,
                                         true/*needs upgrade*/,
                                         false/*read only close*/,
                                         overwriting/*overwriting close*/,
                                         record_close_operations,
                                         false/*record closed fields*/,
                                         already_open);
#ifdef DEBUG_LEGION
                assert(!already_open); // should all be closed now
#endif
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          case OPEN_READ_ONLY_PROJ:
            {
              // If we are reading at this level, we can
              // leave it open otherwise we need a read-only close
              if (!IS_READ_ONLY(closer.user.usage) || (next_child != NULL))
              {
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap);
                }
                it->filter(current_mask);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              else // reading at this level so we can keep it open
                it++;
              break;
            }
          case OPEN_READ_WRITE_PROJ:
            {
              // Have to close up this sub-tree no matter what
              if (record_close_operations)
              {
                const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                assert(!!overlap);
#endif
                closer.record_close_operation(overlap);
              }
              it->filter(current_mask);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_REDUCE_PROJ:
            {
              // If we are reducing at this level we can 
              // leave it open otherwise we need a close
              if (!IS_REDUCE(closer.user.usage) || (next_child != NULL) ||
                  (closer.user.usage.redop != it->redop))
              {
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap);
                }
                it->filter(current_mask);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              else // reducing at this level so we can leave it open
                it++;
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
      // If we had any fields that still need to be opened, create
      // a new field state and add it into the set of new states
      FieldMask open_mask = current_mask - open_below;
      if ((next_child != NULL) && !!open_mask)
      {
        FieldState new_state(closer.user, open_mask, next_child);
        new_states.emplace_back(std::move(new_state));
      }
      merge_new_field_states(state, new_states);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif 
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_logical_projection(LogicalCloser &closer,
                                              LogicalState &state,
                                              const FieldMask &current_mask,
                                              const ProjectionInfo &proj_info,
                                              bool record_close_operations,
                                              FieldMask &open_below)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_SIPHON_LOGICAL_PROJECTION_CALL);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      LegionDeque<FieldState> new_states;
      // First let's see if we need to flush any reductions
      RegionTreeNode *no_next_child = NULL; // never a next child here
      if (!!state.reduction_fields)
      {
        FieldMask reduction_flush_fields = 
          current_mask & state.reduction_fields;
        if (!!reduction_flush_fields)
          flush_logical_reductions(closer, state, reduction_flush_fields,
                       record_close_operations, no_next_child,new_states);
      }
      // Now we can look at all the children
      for (LegionList<FieldState>::iterator it = 
            state.field_states.begin(); it != 
            state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields() * current_mask)
        {
          it++;
          continue;
        }
        // Now we can check the current state
        switch (it->open_state)
        {
          // For now, any children open in a normal mode get closed
          case OPEN_READ_ONLY:
            {
              FieldMask already_open;
              perform_close_operations(closer, current_mask, *it, 
                                       no_next_child, false/*allow next*/,
                                       NULL/*aliased children*/,
                                       false/*needs upgrade*/,
                                       true/*read only close*/,
                                       false/*overwriting close*/,
                                       record_close_operations,
                                       false/*record closed fields*/,
                                       already_open);
              open_below |= already_open;
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_READ_WRITE:
          case OPEN_SINGLE_REDUCE:
          case OPEN_MULTI_REDUCE:
            {
              FieldMask already_open;
              perform_close_operations(closer, current_mask, *it, 
                                       no_next_child, false/*allow next*/,
                                       NULL/*aliased children*/,
                                       false/*needs upgrade*/,
                                       false/*read only close*/,
                                       false/*overwriting close*/,
                                       record_close_operations,
                                       false/*record closed fields*/,
                                       already_open);
              open_below |= already_open;
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_READ_ONLY_PROJ:
            {
              if (IS_READ_ONLY(closer.user.usage))
              {
                // These fields are already open below
                open_below |= (it->valid_fields() & current_mask);
                // Keep track of the sharding projections
                // in case we need to promote to read-write later
                if (proj_info.is_sharding())
                    it->record_projection_summary(proj_info, this); 
                // Keep going
                it++;
              }
              // Reductions always need to go into a new mode to know
              // that we need to flush them so close them up
              else if (!IS_REDUCE(closer.user.usage) && 
                  it->can_elide_close_operation(state, closer.user.op, 
                                    closer.user.idx, proj_info, this))
              {
                if (proj_info.is_sharding())
                  it->record_projection_summary(proj_info, this); 
                // Promote this up to a read-write projection state
                const FieldMask overlap = current_mask & it->valid_fields();
                // Record that some fields are already open
                open_below |= overlap;
                if (overlap != it->valid_fields())
                {
                  // Make the new state to add
                  FieldState new_state(closer.user.usage, overlap, 
                    proj_info.projection, proj_info.projection_space, 
                    proj_info.sharding_function,proj_info.sharding_space, this);
                  // Copy over any projections from before
                  new_state.shard_projections.insert(
                    it->shard_projections.begin(), it->shard_projections.end());
                  new_states.emplace_back(std::move(new_state));
                  // If we are a reduction, we can go straight there
                  it->filter(overlap);
                  if (!it->valid_fields())
                    it = state.field_states.erase(it);
                  else
                    it++;
                }
                else
                {
                  // We overlapped all the fields, so just change the mode
                  it->open_state = OPEN_READ_WRITE_PROJ;
                  // Keep going
                  it++;
                }
              }
              else
              {
                // Only need to record the close here if we're sharding
                if (record_close_operations && proj_info.is_sharding())
                {
                  // We need a close operation here
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap);
                }
                // Otherwise we are going to a different mode 
                // no need to do a close since we're staying in
                // projection mode (except in the sharding function
                // case described above)
                it->filter(current_mask);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          case OPEN_READ_WRITE_PROJ:
            {
              // Can only avoid a close operation if we have the 
              // same projection function with the same or smaller
              // size domain as the original index space launch
              if (!IS_REDUCE(closer.user.usage) &&
                  it->can_elide_close_operation(state, closer.user.op, 
                                    closer.user.idx, proj_info, this))
              {
                // If we're a write we need to update the projection space
                if (proj_info.is_sharding())
                  it->record_projection_summary(proj_info, this); 
                open_below |= (it->valid_fields() & current_mask);
                it++;
              }
              else
              {
                // Only need to record the close here if we're sharding
                if (record_close_operations && proj_info.is_sharding())
                {
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap);
                }
                it->filter(current_mask);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          case OPEN_REDUCE_PROJ:
            {
              // Reduce projections of the same kind can always stay open
              // otherwise we need a close operation
              if (closer.user.usage.redop != it->redop)
              {
                // We need a close operation here if we're sharding
                if (record_close_operations && proj_info.is_sharding())
                {
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap);
                }
                it->filter(current_mask);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              else
              {
                open_below |= (it->valid_fields() & current_mask);
                it++;
              }
              break;
            }
          default:
            assert(false);
        }
      }
      FieldMask open_mask = current_mask - open_below;
      // Note that we always open projection functions even if 
      // the child below is not valid because projection functions
      // are guaranteed to project down below
      if (!!open_mask)
      {
        FieldState new_state(closer.user.usage, open_mask, 
              proj_info.projection, proj_info.projection_space, 
              proj_info.sharding_function, proj_info.sharding_space, this);
        new_states.emplace_back(std::move(new_state));
      }
      merge_new_field_states(state, new_states);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::flush_logical_reductions(LogicalCloser &closer,
                                                  LogicalState &state,
                                              FieldMask &reduction_flush_fields,
                                                  bool record_close_operations,
                                                  RegionTreeNode *next_child,
                                            LegionDeque<FieldState> &new_states)
    //--------------------------------------------------------------------------
    {
      // If we are doing a reduction too, check to see if they are 
      // the same in which case we can skip these fields
      if (closer.user.usage.redop > 0)
      {
        LegionMap<ReductionOpID,FieldMask>::const_iterator finder =
          state.outstanding_reductions.find(closer.user.usage.redop);
        // Don't need to flush fields we are reducing to with the
        // same operation
        if (finder != state.outstanding_reductions.end())
          reduction_flush_fields -= finder->second;
      }
      // See if we still have fields to close
      if (!!reduction_flush_fields)
      {
        FieldMask flushed_fields;
        // We need to flush these fields so issue close operations
        for (std::list<FieldState>::iterator it = 
              state.field_states.begin(); it != 
              state.field_states.end(); /*nothing*/)
        {
          FieldMask overlap = it->valid_fields() & reduction_flush_fields;
          if (!overlap)
          {
            it++;
            continue;
          }
          if (it->is_projection_state())
          {
            closer.record_close_operation(overlap);
            it->filter(overlap);
            flushed_fields |= overlap;
          }
          else
          {
            FieldMask closed_child_fields;
            perform_close_operations(closer, overlap, *it,
                                     next_child, false/*allow_next*/,
                                     NULL/*aliased children*/,
                                     false/*needs upgrade*/,
                                     false/*read only close*/,
                                     false/*overwriting close*/,
                                     record_close_operations,
                                     true/*record closed fields*/,
                                     closed_child_fields);
            // We only really flushed fields that were actually closed
            flushed_fields |= closed_child_fields;
          }
          if (!it->valid_fields())
            it = state.field_states.erase(it);
          else
            it++;
        }
        // Check to see if we have any unflushed fields
        // These are fields which still need a close operation
        // to be performed but only to flush the reductions
        FieldMask unflushed = reduction_flush_fields - flushed_fields;
        if (!!unflushed)
          closer.record_close_operation(unflushed);
        // Then we can mark that these fields no longer have 
        // unflushed reductions
        clear_logical_reduction_fields(state, reduction_flush_fields);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_close_operations(LogicalCloser &closer,
                                            const FieldMask &closing_mask,
                                            FieldState &state,
                                            RegionTreeNode *next_child, 
                                            bool allow_next_child,
                                            const FieldMask *aliased_children,
                                            bool upgrade_next_child,
                                            bool read_only_close,
                                            bool overwriting_close,
                                            bool record_close_operations,
                                            bool record_closed_fields,
                                            FieldMask &output_mask)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_PERFORM_LOGICAL_CLOSES_CALL);
#ifdef DEBUG_LEGION
      // These two things have to be mutually exclusive because
      // they cannot both share the output_mask
      assert(!allow_next_child || !record_closed_fields);
#endif
      // First, if we have a next child and we know all pairs of children
      // are disjoint, then we can skip a lot of this
      bool removed_fields = false;
      const bool all_children_disjoint = are_all_children_disjoint();
      if ((next_child != NULL) && all_children_disjoint)
      {
        // If we have a next child and all the children are disjoint
        // then there are never any close operations, all we have to
        // do is determine if we need to upgrade the child or not
#ifdef DEBUG_LEGION
        assert(aliased_children == NULL);
#endif
        // Check to see if we have any open fields already 
        FieldMaskSet<RegionTreeNode>::iterator finder = 
          state.open_children.find(next_child);
        if (finder != state.open_children.end())
        {
          FieldMask overlap = finder->second & closing_mask;
          if (!!overlap)
          {
            output_mask |= overlap; // already open
            // See if we need to upgrade them
            if (upgrade_next_child)
            {
              finder.filter(overlap);
              removed_fields = true;
              if (!finder->second)
              {
                if (next_child->remove_base_gc_ref(FIELD_STATE_REF))
                  delete next_child;
                state.open_children.erase(finder);
              }
            }
          }
        }
      }
      else if (read_only_close || overwriting_close)
      {
        // Read only closes can close specific children without 
        // any issues, so we can selectively filter what we need to close
        // Overwriting closes are the same as read-only closes, but 
        // they actually do need to close all the children
        std::vector<RegionTreeNode*> to_delete;
        for (FieldMaskSet<RegionTreeNode>::iterator it = 
              state.open_children.begin(); it != 
              state.open_children.end(); it++)
        {
          FieldMask close_mask = it->second & closing_mask;
          if (!close_mask)
            continue;
          // Check for same child, only allow
          if (allow_next_child && (next_child != NULL) && 
              (next_child == it->first))
          {
            if (aliased_children == NULL)
            {
              // No aliased children, so everything is open
              output_mask |= close_mask;
              if (upgrade_next_child)
              {
                it.filter(close_mask);
                removed_fields = true;
                if (!it->second)
                  to_delete.push_back(it->first);
                // The upgraded field state gets added by the caller
              }
              continue;
            }
            else
            {
              // Fields that have aliased children can't remain open
              FieldMask already_open_mask = close_mask - *aliased_children;
              if (!!already_open_mask)
              {
                output_mask |= already_open_mask;
                if (upgrade_next_child)
                {
                  it.filter(already_open_mask);
                  removed_fields = true;
                  if (!it->second)
                    to_delete.push_back(it->first);
                }
                // Remove fields that are already open
                close_mask -= already_open_mask;
                // See if we still have fields to close
                if (!close_mask)
                  continue;
              }
            }
          }
          // Check for child disjointness
          if (!overwriting_close && (next_child != NULL) && 
              (it->first != next_child) && (all_children_disjoint || 
               are_children_disjoint(it->first->get_color(), 
                                     next_child->get_color())))
            continue;
          // Perform the close operation
          it->first->close_logical_node(closer, close_mask, true/*read only*/);
          if (record_close_operations)
            closer.record_close_operation(close_mask);
          // Remove the close fields
          it.filter(close_mask);
          removed_fields = true;
          if (!it->second)
            to_delete.push_back(it->first);
          if (record_closed_fields)
            output_mask |= close_mask;
        }
        // Remove the children that can be deleted
        for (std::vector<RegionTreeNode*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
          state.remove_child(*it);
      }
      else
      {
        // See if we need a full close operation, if we do then we
        // must close up all the children for the field(s) being closed
        // If there is no next_child, then we have to close all
        // children, otherwise figure out which fields need closes
        FieldMask full_close;
        if (next_child != NULL)
        {
          // Figure what if any children need to be closed
          for (FieldMaskSet<RegionTreeNode>::const_iterator it =
                state.open_children.begin(); it !=
                state.open_children.end(); it++)
          {
            FieldMask close_mask = it->second & closing_mask;
            if (!close_mask)
              continue;
            if (next_child == it->first)
            {
              // Same child
              if (allow_next_child)
              {
                // We're allowed to have the same child open, see if 
                // there are any aliasing fields that prevent this
                if (aliased_children != NULL)
                {
                  close_mask &= (*aliased_children);
                  // Any fields with aliased children must be closed
                  if (!!close_mask)
                  {
                    full_close |= close_mask;
                    if (full_close == closing_mask)
                      break;
                  }
                }
              }
              else
              {
                // Must close these fields, despite the same child
                // because allow next is not permitted
                full_close |= close_mask;
                if (full_close == closing_mask)
                  break;
              }
            }
            else
            {
              // Different child from next_child, check for child disjointness
              if (all_children_disjoint || are_children_disjoint(
                    it->first->get_color(), next_child->get_color()))
                continue;
              // Now we definitely have to close it
              full_close |= close_mask;
              if (full_close == closing_mask)
                break;
            }
          } 
        }
        else
        {
          // We need to do a full close, but the closing mask
          // can be an overapproximation, so find all the fields
          // for the actual children that are here
          for (FieldMaskSet<RegionTreeNode>::const_iterator it =
                state.open_children.begin(); it !=
                state.open_children.end(); it++)
          {
            FieldMask close_mask = it->second & closing_mask;
            if (!close_mask)
              continue;
            full_close |= close_mask;
            if (full_close == closing_mask)
              break;
          }
        }
        // See if we have any fields which must be closed
        if (!!full_close)
        {
          // We can record this close operation now
          if (record_close_operations)
            closer.record_close_operation(full_close);
          if (record_closed_fields)
            output_mask |= full_close;
          std::vector<RegionTreeNode*> to_delete;
          // Go through and delete all the children which are to be closed
          for (FieldMaskSet<RegionTreeNode>::iterator it = 
                state.open_children.begin(); it !=
                state.open_children.end(); it++)
          {
            FieldMask child_close = it->second & full_close;
            if (!child_close)
              continue;
            // Perform the close operation
            it->first->close_logical_node(closer, child_close, 
                                          false/*read only close*/);
            // Remove the close fields
            it.filter(child_close);
            removed_fields = true;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          // Remove the children that can be deleted
          for (std::vector<RegionTreeNode*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            state.remove_child(*it);
        }
        // If we allow the next child, we need to see if the next
        // child is already open or not
        if (allow_next_child && (next_child != NULL))
        {
          FieldMaskSet<RegionTreeNode>::iterator finder = 
            state.open_children.find(next_child);
          if (finder != state.open_children.end())
          {
            FieldMask overlap = finder->second & closing_mask;
            if (!!overlap)
            {
              output_mask |= overlap; // already open
              // See if we need to upgrade them
              if (upgrade_next_child)
              {
                finder.filter(overlap);
                removed_fields = true;
                if (!finder->second)
                {
                  if (next_child->remove_base_gc_ref(FIELD_STATE_REF))
                    delete next_child;
                  state.open_children.erase(finder);
                }
              }
            }
          }
        }
      }
      // If we have no children, we can clear our fields
      if (removed_fields || state.open_children.empty())
        state.open_children.tighten_valid_mask();
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
    void RegionTreeNode::merge_new_field_states(LogicalState &state,
                                            LegionDeque<FieldState> &new_states)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < new_states.size(); idx++)
        merge_new_field_state(state, new_states[idx]);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_prev_epoch_users(LogicalState &state,
                                                 const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
      for (LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::iterator 
            it = state.prev_epoch_users.begin(); it != 
            state.prev_epoch_users.end(); /*nothing*/)
      {
        it->field_mask -= field_mask;
        if (!it->field_mask)
        {
          // Remove the mapping reference
          it->op->remove_mapping_reference(it->gen);
          it = state.prev_epoch_users.erase(it); // empty so erase it
        }
        else
          it++; // still has non-dominated fields
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_curr_epoch_users(LogicalState &state,
                                const FieldMask &field_mask, const bool tracing)
    //--------------------------------------------------------------------------
    {
      for (LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::iterator 
            it = state.curr_epoch_users.begin(); it !=
            state.curr_epoch_users.end(); /*nothing*/)
      {
        FieldMask local_dom = it->field_mask & field_mask;
        if (!!local_dom)
        {
          // Move a copy over to the previous epoch users for
          // the fields that were dominated
#ifdef LEGION_SPY
          // Add a mapping reference
          it->op->add_mapping_reference(it->gen);
          // Always do this for Legion Spy 
          state.prev_epoch_users.push_back(*it);
          state.prev_epoch_users.back().field_mask = local_dom;
#else
          // Without Legion Spy we can filter early if the op is done
          // Assuming of course that we are not tracing
          if (it->op->add_mapping_reference(it->gen) || tracing)
          {
            state.prev_epoch_users.push_back(*it);
            state.prev_epoch_users.back().field_mask = local_dom;
          }
          else
          {
            // It's already done so just prune it
            it = state.curr_epoch_users.erase(it);
            continue;
          }
#endif
        }
        else
        {
          it++;
          continue;
        }
        // Update the field mask with the non-dominated fields
        it->field_mask -= local_dom;
        if (!it->field_mask)
        {
          // Remove the mapping reference
          it->op->remove_mapping_reference(it->gen);
          it = state.curr_epoch_users.erase(it); // empty so erase it
        }
        else
          it++; // not empty so keep going
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_disjoint_complete_accesses(
                                     LogicalState &state, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (!state.disjoint_complete_accesses.empty())
      {
        const FieldMask &summary_mask = 
          state.disjoint_complete_accesses.get_valid_mask();
        if (!!summary_mask)
        {
          if (!!(summary_mask - mask))
          {
            std::vector<RegionTreeNode*> to_delete;
            for (FieldMaskSet<RegionTreeNode>::iterator it =
                  state.disjoint_complete_accesses.begin(); it !=
                  state.disjoint_complete_accesses.end(); it++)
            {
              it.filter(mask);
              if (!it->second)
                to_delete.push_back(it->first);
            }
            for (std::vector<RegionTreeNode*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
              state.disjoint_complete_accesses.erase(*it);
            state.disjoint_complete_accesses.tighten_valid_mask();
          }
          else
            state.disjoint_complete_accesses.clear();
        }
      }
      if (!state.disjoint_complete_child_counts.empty())
      {
        for (LogicalState::FieldSizeMap::iterator it =
              state.disjoint_complete_child_counts.begin(); it !=
              state.disjoint_complete_child_counts.end(); /*nothing*/)
        {
          it->second -= mask;
          if (!it->second)
          {
            LegionMap<size_t,FieldMask>::iterator to_delete = it++;
            state.disjoint_complete_child_counts.erase(to_delete);
          }
          else
            it++;
        }
      }
      if (!state.disjoint_complete_projections.empty() &&
          !(state.disjoint_complete_projections.get_valid_mask() * mask))
      {
        std::vector<RefProjectionSummary*> to_delete;
        for (FieldMaskSet<RefProjectionSummary>::iterator it =
              state.disjoint_complete_projections.begin(); it !=
              state.disjoint_complete_projections.end(); it++)
        {
          it.filter(mask);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<RefProjectionSummary*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          state.disjoint_complete_projections.erase(*it);
          if ((*it)->remove_reference())
            delete (*it);
        }
        if (!state.disjoint_complete_projections.empty())
          state.disjoint_complete_projections.tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::report_uninitialized_usage(Operation *op, unsigned idx,
         const RegionUsage usage, const FieldMask &uninit, RtUserEvent reported)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_region());
      assert(reported.exists());
#endif
      LogicalRegion handle = as_region_node()->handle;
      char *field_string = column_source->to_string(uninit, op->get_context());
      op->report_uninitialized_usage(idx, handle, usage, field_string,reported);
      free(field_string);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_logical_reduction(LogicalState &state,
                                                  ReductionOpID redop,
                                                  const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      state.reduction_fields |= user_mask;
      LegionMap<ReductionOpID,FieldMask>::iterator finder = 
        state.outstanding_reductions.find(redop);
      if (finder == state.outstanding_reductions.end())
        state.outstanding_reductions[redop] = user_mask;
      else
        finder->second |= user_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::clear_logical_reduction_fields(LogicalState &state,
                                                  const FieldMask &cleared_mask)
    //--------------------------------------------------------------------------
    {
      state.reduction_fields -= cleared_mask; 
      std::vector<ReductionOpID> to_delete;
      for (LegionMap<ReductionOpID,FieldMask>::iterator it = 
            state.outstanding_reductions.begin(); it !=
            state.outstanding_reductions.end(); it++)
      {
        it->second -= cleared_mask; 
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<ReductionOpID>::const_iterator it = 
            to_delete.begin(); it != to_delete.end(); it++)
      {
        state.outstanding_reductions.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::sanity_check_logical_state(LogicalState &state)
    //--------------------------------------------------------------------------
    {
      // For every child and every field, it should only be open in one mode
      FieldMaskSet<RegionTreeNode> previous_children;
      for (std::list<FieldState>::const_iterator fit = 
            state.field_states.begin(); fit != state.field_states.end(); fit++)
      {
        FieldMask actually_valid;
        for (FieldMaskSet<RegionTreeNode>::const_iterator it =
              fit->open_children.begin(); it != 
              fit->open_children.end(); it++)
        {
          actually_valid |= it->second;
          FieldMaskSet<RegionTreeNode>::iterator finder = 
            previous_children.find(it->first);
          if (finder != previous_children.end())
          {
            assert(!(finder->second & it->second));
            finder.merge(it->second);
          }
          else
            previous_children.insert(it->first, it->second);
        }
        // Actually valid should be greater than or equal
        assert(!(actually_valid - fit->valid_fields()));
      }
      // Also check that for each field it is either only open in one mode
      // or two children in different modes are disjoint
      for (std::list<FieldState>::const_iterator it1 = 
            state.field_states.begin(); it1 != 
            state.field_states.end(); it1++)
      {
        for (std::list<FieldState>::const_iterator it2 = 
              state.field_states.begin(); it2 != 
              state.field_states.end(); it2++)
        {
          // No need to do comparisons if they are the same field state
          if (it1 == it2) 
            continue;
          const FieldState &f1 = *it1;
          const FieldState &f2 = *it2;
          for (FieldMaskSet<RegionTreeNode>::const_iterator cit1 = 
                f1.open_children.begin(); cit1 != 
                f1.open_children.end(); cit1++)
          {
            for (FieldMaskSet<RegionTreeNode>::const_iterator cit2 =
                  f2.open_children.begin(); cit2 != 
                  f2.open_children.end(); cit2++)
            {
              
              // Disjointness check on fields
              if (cit1->second * cit2->second)
                continue;
#ifndef NDEBUG
              LegionColor c1 = cit1->first->get_color();
              LegionColor c2 = cit2->first->get_color();
#endif
              // Some aliasing in the fields, so do the check 
              // for child disjointness
              assert(c1 != c2);
              assert(are_children_disjoint(c1, c2));
            }
          }
        }
      }
      // Disjoint complete fields should always dominate any
      // disjoint complete children we have
      assert(state.disjoint_complete_children.empty() ||
          !(state.disjoint_complete_children.get_valid_mask() - 
            state.disjoint_complete_tree));
      if (is_region())
      {
        // for region nodes, there should only be one 
        // disjoint complete child for each field at a time
        FieldMask disjoint_complete_previous;
        for (FieldMaskSet<RegionTreeNode>::const_iterator it =
              state.disjoint_complete_children.begin(); it !=
              state.disjoint_complete_children.end(); it++)
        {
          assert(disjoint_complete_previous * it->second);
          disjoint_complete_previous |= it->second;
        }
        FieldMask access_previous;
        for (FieldMaskSet<RegionTreeNode>::const_iterator it =
              state.disjoint_complete_accesses.begin(); it !=
              state.disjoint_complete_accesses.end(); it++)
        {
          assert(access_previous * it->second);
          access_previous |= it->second;
        }
        FieldMask odd_counts, even_counts;
        for (LogicalState::FieldSizeMap::const_iterator it =
              state.disjoint_complete_child_counts.begin(); it !=
              state.disjoint_complete_child_counts.end(); it++)
        {
          if ((it->first % 2) == 0)
          {
            assert(even_counts * it->second);
            even_counts |= it->second;
          }
          else
          {
            assert(odd_counts * it->second);
            odd_counts |= it->second;
          }
        }
      }
      else
      {
        FieldMask count_previous;
        for (LogicalState::FieldSizeMap::const_iterator it =
              state.disjoint_complete_child_counts.begin(); it !=
              state.disjoint_complete_child_counts.end(); it++)
        {
          assert(count_previous * it->second);
          count_previous |= it->second;
        }
      }
      if (!state.disjoint_complete_projections.empty())
      {
        FieldMask projection_previous;
        for (FieldMaskSet<RefProjectionSummary>::const_iterator it =
              state.disjoint_complete_projections.begin(); it !=
              state.disjoint_complete_projections.end(); it++)
        {
          assert(projection_previous * it->second);
          projection_previous |= it->second;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_tree_dominance_analysis(ContextID ctx, 
                                      const LogicalUser &user, 
                                      const FieldMask &field_mask,
                                      Operation *skip_op, GenerationID skip_gen)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      if (skip_op != NULL)
      {
        const FieldMask dominator_mask = 
          perform_dependence_checks<CURR_LOGICAL_ALLOC,
            false/*record*/, true/*has skip*/, true/*track dom*/>(
                user, state.curr_epoch_users, field_mask, 
                field_mask, false/*validates*/, skip_op, skip_gen);
        const FieldMask non_dominated_mask = field_mask - dominator_mask;
        if (!!non_dominated_mask)
          perform_dependence_checks<PREV_LOGICAL_ALLOC,
            false/*record*/, true/*has skip*/, false/*track dom*/>(
                user, state.prev_epoch_users, field_mask, 
                field_mask, false/*validates*/, skip_op, skip_gen);
        if (!!dominator_mask)
        {
          filter_prev_epoch_users(state, dominator_mask);
          filter_curr_epoch_users(state, dominator_mask, user.op->is_tracing());
        }
      }
      else
      {
        const FieldMask dominator_mask = 
          perform_dependence_checks<CURR_LOGICAL_ALLOC,
            false/*record*/, false/*has skip*/, true/*track dom*/>(
                user, state.curr_epoch_users, field_mask, 
                field_mask, false/*validates*/, skip_op, skip_gen);
        const FieldMask non_dominated_mask = field_mask - dominator_mask;
        if (!!non_dominated_mask)
          perform_dependence_checks<PREV_LOGICAL_ALLOC,
            false/*record*/, false/*has skip*/, false/*track dom*/>(
                user, state.prev_epoch_users, field_mask, 
                field_mask, false/*validates*/, skip_op, skip_gen);
        if (!!dominator_mask)
        {
          filter_prev_epoch_users(state, dominator_mask);
          filter_curr_epoch_users(state, dominator_mask, user.op->is_tracing());
        }
      }
      // Now figure out which open sub-trees need to be traversed
      FieldMaskSet<RegionTreeNode> to_traverse;
      for (LegionList<FieldState>::const_iterator fit = 
            state.field_states.begin(); fit != state.field_states.end(); fit++)
      {
        if (fit->open_children.get_valid_mask() * field_mask)
          continue;
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
              fit->open_children.begin(); it != fit->open_children.end(); it++)
        {
          const FieldMask overlap = field_mask & it->second;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
      }
      for (FieldMaskSet<RegionTreeNode>::const_iterator it =
            to_traverse.begin(); it != to_traverse.end(); it++)
        it->first->perform_tree_dominance_analysis(ctx, user, it->second,
                                                   skip_op, skip_gen);
    } 

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_disjoint_complete_tree(ContextID ctx,
                                               const FieldMask &invalidate_mask,
                                               const bool invalidate_self)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      assert(!(invalidate_mask - state.disjoint_complete_tree));
#endif
      if (invalidate_self)
        state.disjoint_complete_tree -= invalidate_mask;
      if (state.disjoint_complete_children.empty() ||
         (invalidate_mask * state.disjoint_complete_children.get_valid_mask()))
        return;
      std::vector<RegionTreeNode*> to_delete;
      for (FieldMaskSet<RegionTreeNode>::iterator it = 
            state.disjoint_complete_children.begin(); it !=
            state.disjoint_complete_children.end(); it++)
      {
        const FieldMask overlap = it->second & invalidate_mask;
        if (!overlap)
          continue;
        // Recurse down the tree and remove these fields also
        it->first->invalidate_disjoint_complete_tree(ctx, overlap,true/*self*/);
        it.filter(overlap);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<RegionTreeNode*>::const_iterator it = 
            to_delete.begin(); it != to_delete.end(); it++)
      {
        state.disjoint_complete_children.erase(*it);
        if ((*it)->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
          delete (*it);
      }
      state.disjoint_complete_children.tighten_valid_mask(); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_deletion(ContextID ctx,
                                           const LogicalUser &user,
                                           const FieldMask &check_mask,
                                           const RegionTreePath &path,
                                           const LogicalTraceInfo &trace_info,
                                           FieldMask &already_closed_mask,
                                           bool invalidate_tree)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_REGISTER_LOGICAL_USER_CALL);
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      const unsigned depth = get_depth();
      const bool arrived = !path.has_child(depth);
      if (!arrived)
      {
        FieldMask open_below;
        RegionTreeNode *next_child = get_tree_child(path.get_child(depth));
        if (!!check_mask)
        {
          // Perform any close operations
          LogicalCloser closer(ctx, user, this, false/*validates*/);
          siphon_logical_deletion(closer, state, check_mask, next_child, 
              open_below, ((depth+1) == path.get_max_depth()));
          if (closer.has_close_operations(already_closed_mask))
          {
            // Generate the close operations         
            // We need to record the version numbers for this node as well
            closer.initialize_close_operations(state, user.op, trace_info,
                                 false/*check for refinements*/, !arrived);
            // Perform dependence analysis for all the close operations
            closer.perform_dependence_analysis(user, open_below,
                                               state.curr_epoch_users,
                                               state.prev_epoch_users);
            // Note we don't need to update the version numbers because
            // that happened when we recorded dirty fields below. 
            // However, we do need to mark that there is no longer any
            // dirty data below this node for all the closed fields

            // Update the dirty_below and partial close fields
            // and filter the current and previous epochs
            closer.update_state(state);
            // Now we can add the close operations to the current epoch
            closer.register_close_operations(state.curr_epoch_users);
          }
          // Perform our checks on dependences
          FieldMask dominator_mask = 
                 perform_dependence_checks<CURR_LOGICAL_ALLOC,
                           true/*record*/,false/*has skip*/,true/*track dom*/>(
                              user, state.curr_epoch_users, check_mask, 
                              open_below, false/*validates*/);
          FieldMask non_dominated_mask = check_mask - dominator_mask;
          // For the fields that weren't dominated, we have to check
          // those fields against the previous epoch's users
          if (!!non_dominated_mask)
          {
            perform_dependence_checks<PREV_LOGICAL_ALLOC,
                         true/*record*/, false/*has skip*/, false/*track dom*/>(
                         user, state.prev_epoch_users, non_dominated_mask, 
                         open_below, false/*validates*/);
          }
        }
        // Continue the traversal
        // Only continue checking the fields that are open below
        next_child->register_logical_deletion(ctx, user, open_below, path,
            trace_info, already_closed_mask, invalidate_tree);
      }
      else
      {
        // Register dependences on any users in the sub-tree
        if (!!check_mask)
          perform_tree_dominance_analysis(ctx, user, check_mask);
        // If we're doing a full deletion of this region tree then we just
        // want to delete everything from this level on down as no one
        // else is going to be using it in the future. Ohterwise we register
        // ourselves as a user at this level so that future users of the
        // same field index will record dependences on anything before
        if (invalidate_tree)
        {
          CurrentInvalidator invalidator(ctx, false/*users only*/);
          visit_node(&invalidator);
        }
        else
        {
          // Then register the deletion operation
          // In cases where the field index is recycled this deletion
          // operation will act as mapping dependence so that all the
          // operations for the recycled index will not start until
          // all the operations for the original field at the same 
          // index are done mapping
          register_local_user(state, user, trace_info);
        }
      }
    } 

    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_logical_deletion(LogicalCloser &closer,
                                              LogicalState &state,
                                              const FieldMask &current_mask,
                                              RegionTreeNode *next_child,
                                              FieldMask &open_below,
                                              bool force_close_next)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(next_child != NULL);
      sanity_check_logical_state(state);
#endif
      LegionDeque<FieldState> new_states;
      for (LegionList<FieldState>::iterator it = 
            state.field_states.begin(); it != 
            state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields() * current_mask)
        {
          it++;
          continue;
        }
        // See if our child is open, if it's not then we can keep going
        FieldMaskSet<RegionTreeNode>::const_iterator finder = 
          it->open_children.find(next_child);
        if (it->is_projection_state() || (finder == it->open_children.end()))
        {
          it++;
          continue;
        }
        const FieldMask overlap = finder->second & current_mask;
        if (!overlap)
        {
          it++;
          continue;
        }
        // Now check the current state
        switch (it->open_state)
        {
          case OPEN_READ_ONLY:
            {
              // Record that it is already open below
              open_below |= overlap;
              it++;
              break;
            }
          case OPEN_READ_WRITE:
            {
              // See if we need to force the close, otherwise we 
              // can record that it is already open
              if (force_close_next)
              {
                perform_close_operations(closer, current_mask, *it,
                                         next_child, false/*allow next*/,
                                         NULL/*aliased children*/,
                                         false/*upgrade next child*/,
                                         false/*read only close*/,
                                         false/*overwriting close*/,
                                         true/*record close operations*/,
                                         true/*record closed fields*/,
                                         open_below);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              else
              {
                open_below |= (finder->second & current_mask);
                it++;
              }
              break;
            }
          case OPEN_SINGLE_REDUCE:
            {
              // Go to read-write mode and close if necessary
              if (force_close_next)
              {
                perform_close_operations(closer, current_mask, *it,
                                         next_child, false/*allow next*/,
                                         NULL/*aliased children*/,
                                         false/*upgrade next child*/,
                                         false/*read only close*/,
                                         false/*overwriting close*/,
                                         true/*record close operations*/,
                                         true/*record closed fields*/,
                                         open_below);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              else
              {
                open_below |= overlap;
                it++; 
              }
              break;
            }
          case OPEN_MULTI_REDUCE:
            {
              // Do the close here if our child is open
              perform_close_operations(closer, current_mask, *it,
                                       next_child, false/*allow next*/,
                                       NULL/*aliased children*/,
                                       false/*upgrade next child*/,
                                       false/*read only close*/,
                                       false/*overwriting close*/,
                                       true/*record close operations*/,
                                       true/*record closed fields*/,
                                       open_below);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_READ_ONLY_PROJ:
            {
              // Do a read only close here 
              closer.record_close_operation(overlap);
              it->filter(current_mask);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_READ_WRITE_PROJ:
          case OPEN_REDUCE_PROJ:
            {
              // Do the close here 
              closer.record_close_operation(overlap);
              it->filter(current_mask);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
      merge_new_field_states(state, new_states);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_close_no_dependences(ContextID ctx,
                                                     const LogicalUser &user)
    //--------------------------------------------------------------------------
    {
      const LogicalState &state = get_logical_state(ctx);
      perform_nodep_checks<CURR_LOGICAL_ALLOC>(user, state.curr_epoch_users);
      perform_nodep_checks<PREV_LOGICAL_ALLOC>(user, state.prev_epoch_users);
      for (std::list<FieldState>::const_iterator it = 
            state.field_states.begin(); it != state.field_states.end(); it++)
      {
        if (it->open_children.empty())
          continue;
        if (it->valid_fields() * user.field_mask)
          continue;
        if ((it->open_state == OPEN_READ_ONLY) && IS_READ_ONLY(user.usage))
        {
          for (FieldMaskSet<RegionTreeNode>::const_iterator cit =
                it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
          {
            if (cit->second * user.field_mask)
              continue;
            cit->first->record_close_no_dependences(ctx, user);
          }
        }
        else if ((it->redop == user.usage.redop) &&
                 ((it->open_state == OPEN_SINGLE_REDUCE) ||
                  (it->open_state == OPEN_MULTI_REDUCE)))
        {
          for (FieldMaskSet<RegionTreeNode>::const_iterator cit =
                it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
          {
            if (cit->second * user.field_mask)
              continue;
            cit->first->record_close_no_dependences(ctx, user);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::migrate_logical_state(ContextID src, ContextID dst,
                                               bool merge)
    //--------------------------------------------------------------------------
    {
      LogicalState &src_state = get_logical_state(src);
      LogicalState &dst_state = get_logical_state(dst);
      std::set<RegionTreeNode*> to_traverse;
      if (merge)
      {
        // Use the node lock here for serialization
        AutoLock n_lock(node_lock);
        dst_state.merge(src_state, to_traverse); 
      }
      else
        dst_state.swap(src_state, to_traverse);
      for (std::set<RegionTreeNode*>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
        (*it)->migrate_logical_state(src, dst, merge);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::migrate_version_state(ContextID src, ContextID dst,
                                  std::set<RtEvent> &applied_events, bool merge)
    //--------------------------------------------------------------------------
    {
      VersionManager &src_manager = get_current_version_manager(src);
      VersionManager &dst_manager = get_current_version_manager(dst);
      std::set<RegionTreeNode*> to_traverse;
      LegionMap<AddressSpaceID,SubscriberInvalidations> subscribers;
      if (merge)
      {
        // Use the node lock here for serialization
        AutoLock n_lock(node_lock);
        dst_manager.merge(src_manager, to_traverse, subscribers);
      }
      else
        dst_manager.swap(src_manager, to_traverse, subscribers);
      EqSetTracker::finish_subscriptions(context->runtime, src_manager, 
                                         subscribers, applied_events);
      for (std::set<RegionTreeNode*>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
        (*it)->migrate_version_state(src, dst, applied_events, merge);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::pack_logical_state(ContextID ctx, Serializer &rez,
                                            const bool invalidate)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      std::map<LegionColor,RegionTreeNode*> to_traverse;
      RezCheck z(rez);
      rez.serialize(state.reduction_fields);
      rez.serialize<size_t>(state.outstanding_reductions.size());
      for (LegionMap<ReductionOpID,FieldMask>::const_iterator it = 
            state.outstanding_reductions.begin(); it != 
            state.outstanding_reductions.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(state.field_states.size());
      for (LegionList<FieldState>::const_iterator fit = 
            state.field_states.begin(); fit != 
            state.field_states.end(); fit++)
      {
        rez.serialize(fit->valid_fields());
        rez.serialize(fit->open_state);
        rez.serialize(fit->redop);
        if (fit->open_state >= OPEN_READ_ONLY_PROJ)
        {
          rez.serialize<size_t>(fit->shard_projections.size());
          for (std::set<ProjectionSummary>::const_iterator it = 
                fit->shard_projections.begin(); it != 
                fit->shard_projections.end(); it++)
            it->pack_summary(rez);
        }
#ifdef DEBUG_LEGION
        else
          assert(fit->shard_projections.empty());
#endif
        rez.serialize<size_t>(fit->open_children.size());
        for (FieldMaskSet<RegionTreeNode>::const_iterator it =
              fit->open_children.begin(); it != 
              fit->open_children.end(); it++)
        {
          const LegionColor child_color = it->first->get_color();
          rez.serialize(child_color);
          rez.serialize(it->second);
          to_traverse.insert(std::make_pair(child_color, it->first));
        }
      }
      rez.serialize(state.disjoint_complete_tree);
      rez.serialize<size_t>(state.disjoint_complete_children.size());
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            state.disjoint_complete_children.begin(); it !=
            state.disjoint_complete_children.end(); it++)
      {
        const LegionColor child_color = it->first->get_color();
        rez.serialize(child_color);
        rez.serialize(it->second);
        to_traverse.insert(std::make_pair(child_color, it->first));
      }
      rez.serialize<size_t>(state.disjoint_complete_accesses.size());
      for (FieldMaskSet<RegionTreeNode>::const_iterator it =
            state.disjoint_complete_accesses.begin(); it !=
            state.disjoint_complete_accesses.end(); it++)
      {
        const LegionColor child_color = it->first->get_color();
        rez.serialize(child_color);
        rez.serialize(it->second);
#ifdef DEBUG_LEGION
        // All the children here should be open from the field states
        // which means we should be able to find it in to_traverse
        assert(to_traverse.find(child_color) != to_traverse.end());
#endif
      }
      rez.serialize<size_t>(state.disjoint_complete_child_counts.size());
      for (LogicalState::FieldSizeMap::const_iterator it =
            state.disjoint_complete_child_counts.begin(); it !=
            state.disjoint_complete_child_counts.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(state.disjoint_complete_projections.size());
      for (FieldMaskSet<RefProjectionSummary>::const_iterator it =
            state.disjoint_complete_projections.begin(); it !=
            state.disjoint_complete_projections.end(); it++)
      {
        it->first->pack_summary(rez);
        rez.serialize(it->second);
      }
      // Now recurse down the tree in a deterministic way
      for (std::map<LegionColor,RegionTreeNode*>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
        it->second->pack_logical_state(ctx, rez, invalidate);
      // If we were asked to invalidate then do that now
      if (invalidate)
        state.reset();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::unpack_logical_state(ContextID ctx,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      state.check_init();
#endif
      DerezCheck z(derez);
      derez.deserialize(state.reduction_fields);
      size_t num_reductions;
      derez.deserialize(num_reductions);
      for (unsigned idx = 0; idx < num_reductions; idx++)
      {
        ReductionOpID redop;
        derez.deserialize(redop);
        derez.deserialize(state.outstanding_reductions[redop]);
      }
      size_t num_field_states;
      derez.deserialize(num_field_states);
      state.field_states.resize(num_field_states);
      std::map<LegionColor,RegionTreeNode*> to_traverse;
      for (LegionList<FieldState>::iterator fit = state.field_states.begin();
            fit != state.field_states.end(); fit++)
      {
        FieldMask valid_fields;
        derez.deserialize(valid_fields);
        fit->open_children.relax_valid_mask(valid_fields);
        derez.deserialize(fit->open_state);
        derez.deserialize(fit->redop);
        if (fit->open_state >= OPEN_READ_ONLY_PROJ)
        {
          size_t num_summaries;
          derez.deserialize(num_summaries);
          for (unsigned idx = 0; idx < num_summaries; idx++)
            fit->shard_projections.insert(
             ProjectionSummary::unpack_summary(derez, context));
        }
        size_t num_open_children;
        derez.deserialize(num_open_children);
        for (unsigned idx = 0; idx < num_open_children; idx++)
        {
          LegionColor child_color;
          derez.deserialize(child_color);
          RegionTreeNode *child = get_tree_child(child_color);
          FieldMask mask;
          derez.deserialize(mask);
          fit->add_child(child, mask);
          to_traverse.insert(std::make_pair(child_color, child));
        }
      }
      derez.deserialize(state.disjoint_complete_tree);
      size_t num_disjoint_children;
      derez.deserialize(num_disjoint_children);
      for (unsigned idx = 0; idx < num_disjoint_children; idx++)
      {
        LegionColor child_color;
        derez.deserialize(child_color);
        RegionTreeNode *child = get_tree_child(child_color);
        FieldMask mask;
        derez.deserialize(mask);  
        if (state.disjoint_complete_children.insert(child, mask))
          child->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        to_traverse.insert(std::make_pair(child_color, child));
      }
      size_t num_disjoint_complete_accesses;
      derez.deserialize(num_disjoint_complete_accesses);
      for (unsigned idx = 0; idx < num_disjoint_complete_accesses; idx++)
      {
        LegionColor child_color;
        derez.deserialize(child_color);
        RegionTreeNode *child = get_tree_child(child_color);
        FieldMask mask;
        derez.deserialize(mask);
        state.disjoint_complete_accesses.insert(child, mask);
#ifdef DEBUG_LEGION
        assert(to_traverse.find(child_color) != to_traverse.end());
#endif
      }
      size_t num_disjoint_complete_counts;
      derez.deserialize(num_disjoint_complete_counts);
      const bool inline_counts = state.disjoint_complete_child_counts.empty();
      for (unsigned idx = 0; idx < num_disjoint_complete_counts; idx++)
      {
        size_t count;
        derez.deserialize(count);
        if (!inline_counts)
        {
          LogicalState::FieldSizeMap::iterator finder =
            state.disjoint_complete_child_counts.find(count);
          if (finder != state.disjoint_complete_child_counts.end())
          {
            FieldMask mask;
            derez.deserialize(mask);
            finder->second |= mask;
            continue;
          }
        }
        derez.deserialize(state.disjoint_complete_child_counts[count]);
      }
      size_t num_disjoint_complete_projections;
      derez.deserialize(num_disjoint_complete_projections);
      for (unsigned idx = 0; idx < num_disjoint_complete_projections; idx++)
      {
        RefProjectionSummary *summary = new RefProjectionSummary(
            ProjectionSummary::unpack_summary(derez, context));
        FieldMask mask;
        derez.deserialize(mask);
        summary->add_reference();
        state.disjoint_complete_projections.insert(summary, mask);
      }
      // Traverse and remove remote references after we are done
      for (std::map<LegionColor,RegionTreeNode*>::const_iterator it =
            to_traverse.begin(); it != to_traverse.end(); it++)
        it->second->unpack_logical_state(ctx, derez, source);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::pack_version_state(ContextID ctx, Serializer &rez,
                       const bool invalidate, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);  
      LegionMap<AddressSpaceID,SubscriberInvalidations> subscribers;
      std::map<LegionColor,RegionTreeNode*> to_traverse;
      manager.pack_manager(rez, invalidate, to_traverse, subscribers);
      EqSetTracker::finish_subscriptions(context->runtime, manager, 
                                         subscribers, applied_events);
      for (std::map<LegionColor,RegionTreeNode*>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        it->second->pack_version_state(ctx, rez, invalidate, 
                                       applied_events);
        if (it->second->remove_base_resource_ref(VERSION_MANAGER_REF))
          delete it->second;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::unpack_version_state(ContextID ctx, 
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      std::map<LegionColor,RegionTreeNode*> to_traverse;
      manager.unpack_manager(derez, source, to_traverse);
      for (std::map<LegionColor,RegionTreeNode*>::const_iterator it =
            to_traverse.begin(); it != to_traverse.end(); it++)
        it->second->unpack_version_state(ctx, derez, source);
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
    void RegionTreeNode::invalidate_current_state(ContextID ctx,bool users_only)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState &state = get_logical_state(ctx);
      if (users_only)
        state.clear_logical_users();
      else
        state.reset(); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_deleted_state(ContextID ctx,
                                                  const FieldMask &deleted_mask)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState &state = get_logical_state(ctx);
      state.clear_deleted_state(deleted_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_logical_states(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned ctx = 0; ctx < logical_states.max_entries(); ctx++)
      {
        if (logical_states.has_entry(ctx))
          invalidate_current_state(ctx, false/*users only*/);
      }
    }

    //--------------------------------------------------------------------------
    template<AllocationType ALLOC, bool RECORD, bool HAS_SKIP, bool TRACK_DOM>
    /*static*/ FieldMask RegionTreeNode::perform_dependence_checks(
      const LogicalUser &user, LegionList<LogicalUser, ALLOC> &prev_users,
      const FieldMask &check_mask, const FieldMask &open_below,
      bool validates_regions, Operation *to_skip /*= NULL*/, 
      GenerationID skip_gen /* = 0*/)
    //--------------------------------------------------------------------------
    {
      FieldMask dominator_mask = check_mask;
      // It's not actually sound to assume we dominate something
      // if we don't observe any users of those fields.  Therefore
      // also keep track of the fields that we observe.  We'll use this
      // at the end when computing the final dominator mask.
      FieldMask observed_mask; 
      FieldMask user_check_mask = user.field_mask & check_mask;
      const bool tracing = user.op->is_tracing();
      for (typename LegionList<LogicalUser, ALLOC>::iterator 
            it = prev_users.begin(); it != prev_users.end(); /*nothing*/)
      {
        if (HAS_SKIP && (to_skip == it->op) && (skip_gen == it->gen))
        {
          if (TRACK_DOM)
            dominator_mask -= it->field_mask;
          it++;
          continue;
        }
        FieldMask overlap = user_check_mask & it->field_mask;
        if (!!overlap)
        {
          if (TRACK_DOM)
            observed_mask |= overlap;
          const DependenceType dtype = 
            check_dependence_type<true>(it->usage, user.usage);
          bool validate = validates_regions;
          switch (dtype)
          {
            case LEGION_NO_DEPENDENCE:
              {
                // We need to record any other operations in the same "group"
                // as this one in case we insert internal operations in later
                // replays of this trace
                if (tracing)
                  user.op->register_no_dependence(user.idx, it->op,
                                                  it->gen, it->idx, overlap);
                // No dependence so remove bits from the dominator mask
                dominator_mask -= it->field_mask;
                break;
              }
            case LEGION_ANTI_DEPENDENCE:
            case LEGION_ATOMIC_DEPENDENCE:
            case LEGION_SIMULTANEOUS_DEPENDENCE:
              {
                // Mark that these kinds of dependences are not allowed
                // to validate region inputs
                validate = false;
                // No break so we register dependences just like
                // a true dependence
              }
            case LEGION_TRUE_DEPENDENCE:
              {
#ifdef LEGION_SPY
                LegionSpy::log_mapping_dependence(
                    user.op->get_context()->get_unique_id(),
                    it->uid, it->idx, user.uid, user.idx, dtype);
#endif
                if (RECORD)
                  user.op->record_logical_dependence(*it);
                // Do this after the logging since we might 
                // update the iterator.
                // If we can validate a region record which of our
                // predecessors regions we are validating, otherwise
                // just register a normal dependence
                if (user.op->register_region_dependence(user.idx, it->op, 
                                                        it->gen, it->idx,
                                                        dtype, validate,
                                                        overlap))
                {
#ifndef LEGION_SPY
                  // Now we can prune it from the list and continue
                  it = prev_users.erase(it);
#else
                  it++;
#endif
                }
                else
                {
                  // hasn't commited, reset timeout and continue
                  it->timeout = LogicalUser::TIMEOUT;
                  it++;
                }
                continue;
              }
            default:
              assert(false); // should never get here
          }
        }
        // If we didn't register any kind of dependence, check
        // to see if the timeout has expired.  Note that it is
        // unsound to do this if we are tracing so don't perform
        // the check in that case.
        if (!tracing)
        {
          if (it->timeout <= 0)
          {
            // Timeout has expired.  Check whether the operation
            // has committed. If it has prune it from the list.
            // Otherwise reset its timeout and continue.
            if (it->op->is_operation_committed(it->gen))
            {
#ifndef LEGION_SPY
              it = prev_users.erase(it);
#else
              // Can't prune things early for these cases
              it->timeout = LogicalUser::TIMEOUT;
              it++;
#endif
            }
            else
            {
              // Operation hasn't committed, reset timeout
              it->timeout = LogicalUser::TIMEOUT;
              it++;
            }
          }
          else
          {
            // Timeout hasn't expired, decrement it and continue
            it->timeout--;
            it++;
          }
        }
        else
          it++; // Tracing so no timeouts
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

    // This function is a little out of place to make sure we get the 
    // templates instantiated properly
    //--------------------------------------------------------------------------
    void LogicalCloser::register_dependences(CloseOp *close_op,
                                             const LogicalUser &close_user,
                                             const LogicalUser &current,
                                             const FieldMask &open_below,
                          LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC> &ch_users,
                          LegionList<LogicalUser,LOGICAL_REC_ALLOC > &abv_users,
                          LegionList<LogicalUser,CURR_LOGICAL_ALLOC> &cur_users,
                          LegionList<LogicalUser,PREV_LOGICAL_ALLOC> &pre_users)
    //--------------------------------------------------------------------------
    {
      // Mark that we are starting our dependence analysis
      close_op->begin_dependence_analysis();
      // Do any other work for the dependence analysis
      close_op->trigger_dependence_analysis();
      // First tell the operation to register dependences on any children
      // Register dependences on any interfering children
      // We know that only field non-interference is interesting here
      // because close operations have READ_WRITE EXCLUSIVE
      const FieldMask &close_op_mask = close_user.field_mask;
      // A tricky case here.  We know the current operation is
      // going to register dependences on this close operation,
      // so we can't have the close operation register depencnes
      // on any other users from the same op as the current one
      // we are doing the analysis for (e.g. other region reqs)
      RegionTreeNode::perform_dependence_checks<CLOSE_LOGICAL_ALLOC,
        false/*record*/, true/*has skip*/, false/*track dom*/>(
                                    close_user, ch_users,
                                    close_op_mask, open_below,
                                    false/*validates*/,
                                    current.op, current.gen);
      // Next do checks against any operations above in the tree which
      // the operation already recorded dependences. No need for skip
      // here because we know the operation didn't register any 
      // dependences against itself.
      if (!abv_users.empty())
        RegionTreeNode::perform_dependence_checks<LOGICAL_REC_ALLOC,
            false/*record*/, false/*has skip*/, false/*track dom*/>(
                                       close_user, abv_users,
                                       close_op_mask, open_below,
                                       false/*validates*/,
                                       current.op, current.gen);
      // Finally register any dependences on users at this level
      // See the note above the for the tricky case
      FieldMask dominator_mask; 
      if (!cur_users.empty())
        dominator_mask = 
          RegionTreeNode::perform_dependence_checks<CURR_LOGICAL_ALLOC,
            false/*record*/, true/*has skip*/, true/*track dom*/>(
                                      close_user, cur_users,
                                      close_op_mask, open_below,
                                      false/*validates*/,
                                      current.op, current.gen);
      FieldMask non_dominated_mask = close_op_mask - dominator_mask;
      if (!!non_dominated_mask && !pre_users.empty())
        RegionTreeNode::perform_dependence_checks<PREV_LOGICAL_ALLOC,
          false/*record*/, true/*has skip*/, false/*track dom*/>(
                               close_user, pre_users, 
                               non_dominated_mask, open_below,
                               false/*validates*/,
                               current.op, current.gen);
      // Before we kick off this operation, add a mapping
      // reference to it since we know we are going to put it
      // in the state of the logical region tree
      close_op->add_mapping_reference(close_op->get_generation()); 
      // Mark that we are done, this puts the close op in the pipeline!
      // This is why we cache the GenerationID when making the op
      close_op->end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    template<AllocationType ALLOC>
    /*static*/ void RegionTreeNode::perform_closing_checks(
        LogicalCloser &closer, LegionList<LogicalUser, ALLOC> &users, 
        const FieldMask &check_mask)
    //--------------------------------------------------------------------------
    {
      // Since we are performing a close operation on the region
      // tree data structure, we know that we need to register
      // mapping dependences on all of the operations in the 
      // current epoch since close operations must be serialized
      // with respect to mappings.  Finally we have to upgrade the
      // privilege to read-write to ensure that anyone that comes
      // later also records mapping dependences on the users.
      const FieldMask user_check_mask = closer.user.field_mask & check_mask; 
      for (typename LegionList<LogicalUser, ALLOC>::iterator 
            it = users.begin(); it != users.end(); /*nothing*/)
      {
        FieldMask overlap = user_check_mask & it->field_mask;
        if (!overlap)
        {
          it++;
          continue;
        }
        // Skip any users of the same op, we know they won't be dependences
        if ((it->op == closer.user.op) && (it->gen == closer.user.gen))
        {
          it->field_mask -= overlap; 
          if (!it->field_mask)
          {
            it->op->remove_mapping_reference(it->gen);
            it = users.erase(it);
          }
          else
            it++;
          continue;
        }
        // Record that we closed this user
        // Update the field mask and the privilege
        closer.record_closed_user(*it, overlap);
        // Remove the closed set of fields from this user
        it->field_mask -= overlap;
        // If it's empty, remove it from the list and let
        // the mapping reference go up the tree with it
        // Otherwise add a new mapping reference
        if (!!it->field_mask)
        {
#ifdef LEGION_SPY
          // Always add the reference for Legion Spy
          it->op->add_mapping_reference(it->gen);
          it++;
#else
          // If not Legion Spy we can prune the user if it's done
          // Assuming of course that we are not tracing
          if (!it->op->add_mapping_reference(it->gen) && !closer.tracing)
          {
            closer.pop_closed_user();
            it = users.erase(it);
          }
          else
            it++;
#endif
        }
        else
          it = users.erase(it);
      }
    }

    //--------------------------------------------------------------------------
    template<AllocationType ALLOC>
    /*static*/void RegionTreeNode::perform_nodep_checks(const LogicalUser &user,
                                    const LegionList<LogicalUser, ALLOC> &users)
    //--------------------------------------------------------------------------
    {
      for (typename LegionList<LogicalUser,ALLOC>::const_iterator
            it = users.begin(); it != users.end(); it++)
      {
        if (it->usage != user.usage)
          continue;
        const FieldMask overlap = user.field_mask & it->field_mask;
        if (!overlap)
          continue;
        // Skip any users of the same op, we know they won't be dependences
        if ((it->op == user.op) && (it->gen == user.gen))
          continue;
        user.op->register_no_dependence(user.idx, it->op, 
                                        it->gen, it->idx, overlap);
      }
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
    void RegionNode::initialize_disjoint_complete_tree(ContextID ctx,
                                                       const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
      assert(state.disjoint_complete_children.empty());
#endif
      state.disjoint_complete_tree |= mask;
    }

    //--------------------------------------------------------------------------
    void RegionNode::refine_disjoint_complete_tree(ContextID ctx,
            PartitionNode *child, RefinementOp *refinement_op,
            const FieldMask &refinement_mask, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      assert(child->parent == this);
      assert(!(refinement_mask - state.disjoint_complete_tree));
#endif
      // Invalidate the old disjoint complete tree
      invalidate_disjoint_complete_tree(ctx, refinement_mask, false/*self*/);
#ifdef DEBUG_LEGION
      assert(state.disjoint_complete_children.get_valid_mask() * 
              refinement_mask);
#endif
      // Record the new disjoint complete tree
      if (state.disjoint_complete_children.insert(child, refinement_mask))
        child->add_base_gc_ref(DISJOINT_COMPLETE_REF);
      // Update the refinement tree for the full subtree
      child->update_disjoint_complete_tree(ctx, refinement_op,
                                           refinement_mask, applied_events);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::filter_unversioned_fields(ContextID ctx, 
     TaskContext *context, const FieldMask &filter_mask, RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent == NULL);
#endif
      LogicalState &state = get_logical_state(ctx);
      const FieldMask unversioned = filter_mask - state.disjoint_complete_tree;
      if (!unversioned)
        return false;
      std::vector<FieldID> to_remove;
      column_source->get_field_set(unversioned, context, to_remove);
      for (std::vector<FieldID>::const_iterator it =
            to_remove.begin(); it != to_remove.end(); it++)
      {
        std::set<FieldID>::iterator finder = req.privilege_fields.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != req.privilege_fields.end());
#endif
        req.privilege_fields.erase(finder);
      }
      return true;
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
    IndexSpaceExpression* RegionNode::get_index_space_expression(void) const
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
    AddressSpaceID RegionNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle, context->runtime);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID RegionNode::get_owner_space(LogicalRegion handle,
                                                          Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (handle.tree_id % runtime->runtime_stride);
    }

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
    void RegionNode::pack_global_reference(bool need_root)
    //--------------------------------------------------------------------------
    {
      if (need_root)
      {
        RegionNode *root = this;
        while (root->parent != NULL)
          root = root->parent->parent;
        root->pack_global_ref();
      }
      if (row_source->parent != NULL)
        row_source->parent->pack_valid_ref();
      else
        row_source->pack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void RegionNode::unpack_global_reference(bool need_root)
    //--------------------------------------------------------------------------
    {
      if (need_root)
      {
        RegionNode *root = this;
        while (root->parent != NULL)
          root = root->parent->parent;
        root->unpack_global_ref();
      }
      if (row_source->parent != NULL)
        row_source->parent->unpack_valid_ref();
      else
        row_source->unpack_valid_ref();
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
    bool RegionNode::dominates(RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_region())
        return row_source->dominates(other->as_region_node()->row_source);
      else
        return row_source->dominates(other->as_partition_node()->row_source);
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

      RegionNode *node = 
        context->create_node(handle, NULL/*parent*/, initialized, did, prov);
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
    void RegionNode::update_disjoint_complete_tree(ContextID ctx,
                                              RefinementOp *refinement_op, 
                                              const FieldMask &refinement_mask,
                                              FieldMask &refined_partition,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      assert(refinement_mask * state.disjoint_complete_tree);
#endif
      state.disjoint_complete_tree |= refinement_mask;
      // check to see if we have any disjoint complete projections
      FieldMaskSet<RefProjectionSummary> &disjoint_complete_projections =
        state.disjoint_complete_projections;
      if (!disjoint_complete_projections.empty() &&
          !(disjoint_complete_projections.get_valid_mask() * refinement_mask))
      {
        std::vector<RefProjectionSummary*> to_delete;
        for (FieldMaskSet<RefProjectionSummary>::iterator it =
              disjoint_complete_projections.begin(); it !=
              disjoint_complete_projections.end(); it++)
        {
          const FieldMask overlap = it->second & refinement_mask;
          if (!overlap)
            continue;
          refined_partition |= overlap;
          refinement_op->record_refinement(this, overlap, it->first);
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<RefProjectionSummary*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          disjoint_complete_projections.erase(*it);
          if ((*it)->remove_reference())
            delete (*it);
        }
        disjoint_complete_projections.tighten_valid_mask();
      }
      FieldMaskSet<RegionTreeNode> &access_disjoint_complete_children = 
        state.disjoint_complete_accesses;
      if (access_disjoint_complete_children.empty() ||
         (refinement_mask * access_disjoint_complete_children.get_valid_mask()))
        return;
      std::vector<RegionTreeNode*> to_delete;
      for (FieldMaskSet<RegionTreeNode>::iterator it = 
            access_disjoint_complete_children.begin(); it !=
            access_disjoint_complete_children.end(); it++)
      {
        const FieldMask overlap = refinement_mask & it->second;
        if (!overlap)
          continue;
        // Traverse it and add it to the refined partition set
        it->first->as_partition_node()->update_disjoint_complete_tree(ctx,
                                  refinement_op, overlap, applied_events);
        refined_partition |= overlap;
        // Add it to the current set
        if (state.disjoint_complete_children.insert(it->first, overlap))
          it->first->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        // Remove it from this set
        it.filter(overlap);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      if (!to_delete.empty())
      {
        if (to_delete.size() != access_disjoint_complete_children.size())
        {
          for (std::vector<RegionTreeNode*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            access_disjoint_complete_children.erase(*it);
        }
        else
          access_disjoint_complete_children.clear();
      }
      access_disjoint_complete_children.tighten_valid_mask(); 
    }

    //--------------------------------------------------------------------------
    void RegionNode::initialize_versioning_analysis(ContextID ctx,
                                              EquivalenceSet *set,
                                              const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      manager.initialize_versioning_analysis(set, mask);
    }

    //--------------------------------------------------------------------------
    void RegionNode::initialize_nonexclusive_virtual_analysis(ContextID ctx,
                                    const FieldMask &mask,
                                    const FieldMaskSet<EquivalenceSet> &eq_sets)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      manager.initialize_nonexclusive_virtual_analysis(mask, eq_sets);
    }

    //--------------------------------------------------------------------------
    void RegionNode::perform_versioning_analysis(ContextID ctx,
                                                 InnerContext *parent_ctx,
                                                 VersionInfo *version_info,
                                                 const FieldMask &mask,
                                                 const UniqueID opid,
                                                 const AddressSpaceID source,
                                                 std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      manager.perform_versioning_analysis(parent_ctx, version_info, this, 
            row_source, true/*expr covers*/, mask, opid, source, applied);
    }
    
    //--------------------------------------------------------------------------
    void RegionNode::compute_equivalence_sets(ContextID ctx, 
                                              InnerContext *context,
                                              EqSetTracker *target,
                                              const AddressSpaceID target_space,
                                              IndexSpaceExpression *expr,
                                              const FieldMask &mask,
                                              const UniqueID opid,
                                              const AddressSpaceID source,
                                              std::set<RtEvent> &ready_events,
                                              const bool downward_only,
                                              const bool expr_covers)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      FieldMask parent_traversal;
      FieldMaskSet<PartitionNode> children_traversal;
      std::set<RtEvent> deferral_events;
      manager.compute_equivalence_sets(ctx, expr, target, target_space, mask, 
                context, opid, source, ready_events, children_traversal, 
                parent_traversal, deferral_events, downward_only);
      if (!deferral_events.empty())
      {
#ifdef DEBUG_LEGION
        assert(downward_only);
#endif
        const RtEvent deferral_event = Runtime::merge_events(deferral_events);
        if (deferral_event.exists() && !deferral_event.has_triggered())
        {
          // Defer this computation until the version manager data is ready
          DeferComputeEquivalenceSetArgs args(this, ctx, context, target, 
                    target_space, expr, mask, opid, source, expr_covers);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_DEFERRED_PRIORITY, deferral_event);
          ready_events.insert(args.ready);
        }
        else
          compute_equivalence_sets(ctx, context, target, target_space, expr,
            mask, opid, source, ready_events,true/*downward only*/,expr_covers);
      }
      if (!!parent_traversal)
      {
#ifdef DEBUG_LEGION
        assert(parent != NULL);
        assert(!downward_only);
#endif
        parent->compute_equivalence_sets(ctx, context, target, target_space,
                                         expr, parent_traversal, opid, source, 
                                         ready_events, false/*downward only*/,
                                         false/*expr covers*/);
      }
      if (!children_traversal.empty())
      {
        for (FieldMaskSet<PartitionNode>::const_iterator it = 
              children_traversal.begin(); it != children_traversal.end(); it++)
          it->first->compute_equivalence_sets(ctx, context, target,target_space,
                                   expr, it->second, opid, source, ready_events,
                                   true/*downward only*/, expr_covers);
      }
    }

    //--------------------------------------------------------------------------
    RegionNode::DeferComputeEquivalenceSetArgs::DeferComputeEquivalenceSetArgs(
          RegionNode *proxy, ContextID x, InnerContext *c, EqSetTracker *t, 
          const AddressSpaceID ts, IndexSpaceExpression *e, const FieldMask &m, 
          const UniqueID id, const AddressSpaceID s, const bool covers)
      : LgTaskArgs<DeferComputeEquivalenceSetArgs>(implicit_provenance),
        proxy_this(proxy), ctx(x), context(c), target(t), target_space(ts),
        expr(e), mask(new FieldMask(m)), opid(id), source(s),
        ready(Runtime::create_rt_user_event()), expr_covers(covers)
    //--------------------------------------------------------------------------
    {
      expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_deferred_compute_equivalence_sets(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferComputeEquivalenceSetArgs *dargs = 
        (const DeferComputeEquivalenceSetArgs*)args;
      std::set<RtEvent> ready_events;
      dargs->proxy_this->compute_equivalence_sets(dargs->ctx, dargs->context,
          dargs->target, dargs->target_space, dargs->expr, *(dargs->mask), 
          dargs->opid, dargs->source, ready_events, true/*downward only*/,
          dargs->expr_covers);
      if (!ready_events.empty())
        Runtime::trigger_event(dargs->ready, 
            Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(dargs->ready);
      delete dargs->mask;
      if (dargs->expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->expr;
    }

    //--------------------------------------------------------------------------
    void RegionNode::invalidate_refinement(ContextID ctx, const FieldMask &mask, 
                                   bool self, InnerContext &source_context,
                                   std::set<RtEvent> &applied_events, 
                                   std::vector<EquivalenceSet*> &to_release,
                                   bool nonexclusive_virtual_mapping_root)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      FieldMaskSet<RegionTreeNode> to_traverse;
      LegionMap<AddressSpaceID,SubscriberInvalidations> subscribers;
      manager.invalidate_refinement(source_context, mask, self, to_traverse,
                subscribers, to_release, nonexclusive_virtual_mapping_root);
      EqSetTracker::finish_subscriptions(context->runtime, manager, 
                                         subscribers, applied_events);
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(!it->first->is_region());
#endif
        it->first->as_partition_node()->invalidate_refinement(ctx, it->second, 
                                  applied_events, to_release, source_context);
        if (it->first->remove_base_gc_ref(VERSION_MANAGER_REF))
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::record_refinement(ContextID ctx, EquivalenceSet *set,
                                       const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      FieldMask parent_mask;
      manager.record_refinement(set, mask, parent_mask);
      if (!!parent_mask && (parent != NULL))
        parent->propagate_refinement(ctx, this, parent_mask);
    }

    //--------------------------------------------------------------------------
    void RegionNode::propagate_refinement(ContextID ctx, PartitionNode *child,
                                          const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      FieldMask parent_mask;
      manager.propagate_refinement(child, mask, parent_mask);
      if (!!parent_mask && (parent != NULL))
        parent->propagate_refinement(ctx, this, parent_mask);
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
        if ((sit->valid_fields() * mask) || (sit->is_projection_state()))
          continue;
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
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
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tid;
      derez.deserialize(tid);
      RegionNode *node = forest->get_tree(tid);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Serializer rez;
      {
        RezCheck z(rez);
        node->send_node(rez, source);
        rez.serialize(done_event);
      }
      forest->runtime->send_top_level_region_return(source, rez);
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
    void RegionNode::print_physical_context(ContextID ctx, 
                                            TreeStateLogger *logger,
                                            const FieldMask &capture_mask,
                                       std::deque<RegionTreeNode*> &to_traverse)
    //--------------------------------------------------------------------------
    {
      logger->log("==========");
      print_context_header(logger);
      logger->down();
      if (current_versions.has_entry(ctx))
      {
        VersionManager &manager= get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (to_traverse.size() > 0)
      {
        RegionTreeNode *node = to_traverse.front();
        to_traverse.pop_front();
        node->print_physical_context(ctx, logger, capture_mask, to_traverse);
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
        {
          char *mask_string = state.reduction_fields.to_string();
          logger->log("Reduction Mask %s", mask_string);
          free(mask_string);
        }
        // Outstanding Reductions
        {
          logger->log("Outstanding Reductions (%ld)",
              state.outstanding_reductions.size());
          logger->down();
          for (LegionMap<ReductionOpID,FieldMask>::iterator it =
                state.outstanding_reductions.begin(); it !=
                state.outstanding_reductions.end(); it++)
          {
            char *mask_string = it->second.to_string();
            logger->log("Op ID %d Mask %s\n", it->first, mask_string);
            free(mask_string);
          }
          logger->up();
        }
        logger->log("Open Field States (%ld)", state.field_states.size());
        logger->down();
        for (std::list<FieldState>::const_iterator it = 
              state.field_states.begin(); it != 
              state.field_states.end(); it++)
        {
          it->print_state(logger, capture_mask, this);
          if (it->valid_fields() * capture_mask)
            continue;
          for (FieldMaskSet<RegionTreeNode>::const_iterator cit = 
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

    //--------------------------------------------------------------------------
    void RegionNode::dump_physical_context(ContextID ctx,
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
      if (logical_states.has_entry(ctx))
      {
        VersionManager &manager = get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, logger);
      }
      else
        logger->log("No state");
      logger->log("");
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
      IndexSpaceNode *index_node = row_source->get_child(c);
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
    IndexSpaceExpression* PartitionNode::get_index_space_expression(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_union_expression();
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
    AddressSpaceID PartitionNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle, context->runtime);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID PartitionNode::get_owner_space(
                                     LogicalPartition handle, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (handle.tree_id % runtime->total_address_spaces);
    }

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
          if (row_source->total_children == row_source->max_linearized_color)
          {
            for (LegionColor c = 0; c < row_source->total_children; c++)
            {
              bool result = get_child(c)->visit_node(traverser);
              continue_traversal = continue_traversal && result;
              if (!result && break_early)
                break;
            }
          }
          else
          {
            ColorSpaceIterator *itr = 
              row_source->color_space->create_color_space_iterator();
            while (itr->is_valid())
            {
              const LegionColor c = itr->yield_color();
              bool result = get_child(c)->visit_node(traverser);
              continue_traversal = continue_traversal && result;
              if (!result && break_early)
                break;
            }
            delete itr;
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
    void PartitionNode::pack_global_reference(bool need_root)
    //--------------------------------------------------------------------------
    {
      if (need_root)
      {
        RegionNode *root = parent;
        while (root->parent != NULL)
          root = root->parent->parent;
        root->pack_global_ref();
      }
      row_source->pack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::unpack_global_reference(bool need_root)
    //--------------------------------------------------------------------------
    {
      if (need_root)
      {
        RegionNode *root = parent;
        while (root->parent != NULL)
          root = root->parent->parent;
        root->unpack_global_ref();
      }
      row_source->unpack_valid_ref();
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
    bool PartitionNode::dominates(RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_region())
        return row_source->dominates(other->as_region_node()->row_source);
      else
        return row_source->dominates(other->as_partition_node()->row_source);
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
    void PartitionNode::update_disjoint_complete_tree(ContextID ctx,
                                              RefinementOp *refinement_op,
                                              const FieldMask &refinement_mask,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
#ifdef DEBUG_LEGION
      assert(refinement_mask * state.disjoint_complete_tree);
      assert(row_source->is_disjoint(false/*from app*/));
      assert(row_source->is_complete(false/*from app*/));
#endif
      state.disjoint_complete_tree |= refinement_mask;
      // The fields in the disjoint_complete_accesses valid mask 
      // represent the ones that we had child refinement notifications
      // for and therefore we need to traverse all those children
      // to see which refinements they have
      const FieldMask child_overlap = refinement_mask &
        state.disjoint_complete_accesses.get_valid_mask();
      if (!child_overlap)
      {
        // Record that this is a leaf-partition of this refinement
        refinement_op->record_refinement(this, refinement_mask);
        return;
      }
      const FieldMask non_child_overlap = refinement_mask - child_overlap;
      if (!!non_child_overlap)
        refinement_op->record_refinement(this, non_child_overlap);
      FieldMask all_unrefined_children = child_overlap;
      // Check to see if we have any disjoint complete projections that
      // we can filter out first since they are easy to handle
      if (!state.disjoint_complete_projections.empty() && !(refinement_mask *
          state.disjoint_complete_projections.get_valid_mask()))
      {
        std::vector<RefProjectionSummary*> to_delete;
        for (FieldMaskSet<RefProjectionSummary>::iterator it =
              state.disjoint_complete_projections.begin(); it !=
              state.disjoint_complete_projections.end(); it++)
        {
          const FieldMask overlap = it->second & child_overlap;
          if (!overlap)
            continue;
          refinement_op->record_refinement(this, overlap, it->first);
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
          all_unrefined_children -= overlap;
          if (!all_unrefined_children)
            break;
        }
        for (std::vector<RefProjectionSummary*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          state.disjoint_complete_projections.erase(*it);
          if ((*it)->remove_reference())
            delete (*it);
        }
        state.disjoint_complete_projections.tighten_valid_mask();
        // Quick out if this caught all our child overlap
        if (!all_unrefined_children)
        {
          state.disjoint_complete_accesses.filter_valid_mask(child_overlap);
          return;
        }
      }
      // We need to check to see if all the children below here are
      // refined the same way or whether they are "ragged" where 
      // some children are refined and others are not so we need
      // to tell the refinement operation to refine those individual
      // regions which have no partition refinements of their own
      FieldMaskSet<RegionTreeNode> unrefined_children;
      // Have to iterate over all the chlidren here
      if (row_source->total_children == row_source->max_linearized_color)
      {
        for (LegionColor color = 0; color < row_source->total_children; color++)
        {
          RegionNode *child = get_child(color);
          // Traverse it here and perform our checks that all the 
          // children are refined the same way
          FieldMask refined_child;
          child->as_region_node()->update_disjoint_complete_tree(ctx,
              refinement_op, child_overlap, refined_child, applied_events);
          if (!!refined_child)
          {
            const FieldMask unrefined = child_overlap - refined_child;
            if (!!unrefined)
            {
              unrefined_children.insert(child, unrefined);
              if (!!all_unrefined_children)
                all_unrefined_children &= unrefined;
            }
            else if (!!all_unrefined_children)
              all_unrefined_children.clear();
          }
          else
            unrefined_children.insert(child, child_overlap);
          // Add it to the current set
          if (state.disjoint_complete_children.insert(child, child_overlap))
            child->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          row_source->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          RegionNode *child = get_child(itr->yield_color());
          // Traverse it here and perform our checks that all the 
          // children are refined the same way
          FieldMask refined_child;
          child->as_region_node()->update_disjoint_complete_tree(ctx,
              refinement_op, child_overlap, refined_child, applied_events);
          if (!!refined_child)
          {
            const FieldMask unrefined = child_overlap - refined_child;
            if (!!unrefined)
            {
              unrefined_children.insert(child, unrefined);
              if (!!all_unrefined_children)
                all_unrefined_children &= unrefined;
            }
            else if (!!all_unrefined_children)
              all_unrefined_children.clear();
          }
          else
            unrefined_children.insert(child, child_overlap);
          // Add it to the current set
          if (state.disjoint_complete_children.insert(child, child_overlap))
            child->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        }
        delete itr;
      } 
      // Record that the refinement operation should perform refinements for
      // this partition if all the children were unrefined for some fields
      if (!!all_unrefined_children)
      {
        refinement_op->record_refinement(this, all_unrefined_children);
        if (all_unrefined_children != unrefined_children.get_valid_mask())
        {
          // Filter out just the fields that were unrefined for all
          // the children, leaving only the partials remaning
          std::vector<RegionTreeNode*> to_delete;
          for (FieldMaskSet<RegionTreeNode>::iterator it =
               unrefined_children.begin(); it != unrefined_children.end(); it++)
          {
            it.filter(all_unrefined_children);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<RegionTreeNode*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
              unrefined_children.erase(*it);
          }
          unrefined_children.tighten_valid_mask();
        }
        else
          unrefined_children.clear();
      }
      if (!unrefined_children.empty())
        refinement_op->record_refinements(unrefined_children);
      // Clean up the disjoint complete accesses data structure, remove
      // any children present and filter the valid mask appropriately
      if (!state.disjoint_complete_accesses.empty())
      {
        std::vector<RegionTreeNode*> to_delete;
        for (FieldMaskSet<RegionTreeNode>::iterator it =
              state.disjoint_complete_accesses.begin(); it !=
              state.disjoint_complete_accesses.end(); it++)
        {
          it.filter(child_overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<RegionTreeNode*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
          state.disjoint_complete_accesses.erase(*it);
      }
      state.disjoint_complete_accesses.filter_valid_mask(child_overlap);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::compute_equivalence_sets(ContextID ctx,
                                              InnerContext *context,
                                              EqSetTracker *target,
                                              const AddressSpaceID target_space,
                                              IndexSpaceExpression *expr,
                                              const FieldMask &mask,
                                              const UniqueID opid,
                                              const AddressSpaceID source,
                                              std::set<RtEvent> &ready_events,
                                              const bool downward_only,
                                              const bool expr_covers)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      if (downward_only)
      {
        std::vector<LegionColor> interfering_children;
        if (expr_covers)
        {
          // Expr covers so all the children interfere
          if (row_source->total_children == row_source->max_linearized_color)
          {
            interfering_children.resize(row_source->total_children);
            for (LegionColor color = 0; 
                  color < row_source->total_children; color++)
              interfering_children[color] = color;
          }
          else
          {
            ColorSpaceIterator *iterator =
              row_source->color_space->create_color_space_iterator();
            while (iterator->is_valid())
              interfering_children.push_back(iterator->yield_color());
            delete iterator;
          }
        }
        else
          row_source->find_interfering_children(expr, interfering_children);
        for (std::vector<LegionColor>::const_iterator it = 
              interfering_children.begin(); it != 
              interfering_children.end(); it++)
        {
          RegionNode *child = get_child(*it);
          // Still need to filter empty children when traversing here
          if (expr_covers && child->row_source->is_empty())
            continue;
          child->compute_equivalence_sets(ctx, context, target, target_space,
                                expr, mask, opid, source, ready_events,
                                true/*downward only*/, expr_covers);
        }
      }
      else
      {
        FieldMask parent_traversal, children_traversal;
        manager.compute_equivalence_sets(mask, parent_traversal, 
                                         children_traversal);
#ifdef DEBUG_LEGION
        assert((parent_traversal | children_traversal) == mask);
#endif
        if (!!parent_traversal)
          parent->compute_equivalence_sets(ctx, context, target, target_space,
                                           expr, parent_traversal, opid, source, 
                                           ready_events, false/*downward only*/,
                                           false/*expr covers*/);
        if (!!children_traversal)
        {
          std::vector<LegionColor> interfering_children;
          if (expr_covers)
          {
            // Expr covers so all the children interfere
            if (row_source->total_children == row_source->max_linearized_color)
            {
              interfering_children.resize(row_source->total_children);
              for (LegionColor color = 0;
                    color < row_source->total_children; color++)
                interfering_children[color] = color;
            }
            else
            {
              ColorSpaceIterator *iterator =
                row_source->color_space->create_color_space_iterator();
              while (iterator->is_valid())
                interfering_children.push_back(iterator->yield_color());
              delete iterator;
            }
          }
          else
            row_source->find_interfering_children(expr, interfering_children);
          for (std::vector<LegionColor>::const_iterator it = 
                interfering_children.begin(); it != 
                interfering_children.end(); it++)
          {
            RegionNode *child = get_child(*it);
            // Still need to filter empty children when traversing here
            if (expr_covers && child->row_source->is_empty())
              continue;
            child->compute_equivalence_sets(ctx, context, target, 
                target_space, expr, children_traversal, opid, source,
                ready_events, true/*downward only*/, expr_covers);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::invalidate_refinement(ContextID ctx, 
                                       const FieldMask &mask, 
                                       std::set<RtEvent> &applied_events,
                                       std::vector<EquivalenceSet*> &to_release,
                                       InnerContext &source_context)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      FieldMaskSet<RegionTreeNode> to_traverse;
      LegionMap<AddressSpaceID,SubscriberInvalidations> subscribers;
      manager.invalidate_refinement(source_context, mask, true/*delete self*/,
                                    to_traverse, subscribers, to_release);
#ifdef DEBUG_LEGION
      assert(subscribers.empty());
#endif
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->first->is_region());
#endif
        it->first->as_region_node()->invalidate_refinement(ctx, it->second, 
                  true/*self*/, source_context, applied_events, to_release);
        if (it->first->remove_base_gc_ref(VERSION_MANAGER_REF))
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::propagate_refinement(ContextID ctx, RegionNode *child, 
                                             const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      FieldMask parent_mask;
      manager.propagate_refinement(child, mask, parent_mask);
      if (!!parent_mask)
        parent->propagate_refinement(ctx, this, parent_mask);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::propagate_refinement(ContextID ctx,
                const std::vector<RegionNode*> &children, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      FieldMask parent_mask;
      manager.propagate_refinement(children, mask, parent_mask);
      if (!!parent_mask)
        parent->propagate_refinement(ctx, this, parent_mask);
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
    void PartitionNode::print_physical_context(ContextID ctx,
                                               TreeStateLogger *logger,
                                               const FieldMask &capture_mask,
                                       std::deque<RegionTreeNode*> &to_traverse)
    //--------------------------------------------------------------------------
    {
      logger->log("==========");
      print_context_header(logger);
      logger->down();
      if (current_versions.has_entry(ctx))
      {
        VersionManager &manager = get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (to_traverse.size() > 0)
      {
        RegionTreeNode *node = to_traverse.front();
        to_traverse.pop_front();
        node->print_physical_context(ctx, logger, capture_mask, to_traverse);
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
      // Dirty Fields
      {
        char *mask_string = state.reduction_fields.to_string();
        logger->log("Reduction Mask %s", mask_string);
        free(mask_string);
      }
      // Outstanding Reductions
      {
        logger->log("Outstanding Reductions (%ld)",
            state.outstanding_reductions.size());
        logger->down();
        for (LegionMap<ReductionOpID,FieldMask>::iterator it =
              state.outstanding_reductions.begin(); it !=
              state.outstanding_reductions.end(); it++)
        {
          char *mask_string = it->second.to_string();
          logger->log("Op ID %d Mask %s\n", it->first, mask_string);
          free(mask_string);
        }
        logger->up();
      }
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
          for (FieldMaskSet<RegionTreeNode>::const_iterator cit =
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

    //--------------------------------------------------------------------------
    void PartitionNode::dump_physical_context(ContextID ctx,
                                              TreeStateLogger *logger,
                                              const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      print_context_header(logger);
      logger->down();
      if (logical_states.has_entry(ctx))
      {
        VersionManager &manager = get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      logger->up();
    }
#endif 

  }; // namespace Internal 
}; // namespace Legion

// EOF

