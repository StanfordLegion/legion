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

// templates in legion/region_tree.inl are instantiated by region_tree_tmpl.cc

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    //--------------------------------------------------------------------------
    IndirectRecord::IndirectRecord(RegionTreeForest *forest,
                                   const RegionRequirement &req,
                                   const InstanceSet &insts,
                                   const DomainPoint &key)
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
#ifdef LEGION_SPY
      instance_events.resize(field_indexes.size());
#endif
      // For each of the fields in the region requirement
      // (importantly in the order they will be copied)
      // find the corresponding instance and store them 
      // in the indirect record
      for (unsigned fidx = 0; fidx < field_indexes.size(); fidx++)
      {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        bool found = false;
#endif
#endif
        for (unsigned idx = 0; idx < insts.size(); idx++)
        {
          const InstanceRef &ref = insts[idx];
          const FieldMask &mask = ref.get_valid_fields();
          if (!mask.is_set(field_indexes[fidx]))
            continue;
          PhysicalManager *manager = ref.get_physical_manager();
          instances[fidx] = manager->get_instance(key);
#ifdef LEGION_SPY
          instance_events[fidx] = manager->get_unique_event();
#endif
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
                                        ApEvent ready /*=ApEvent::NO_AP_EVENT*/,
                                        std::set<RtEvent> *applied /*=NULL*/)
    //--------------------------------------------------------------------------
    {
      return create_node(handle, domain, true/*domain*/, NULL/*parent*/,
                         0/*color*/, did, RtEvent::NO_RT_EVENT,
                         provenance, ready, 0/*expr id*/, applied);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_union_space(IndexSpace handle,
                    DistributedID did, const char *prov,
                    const std::vector<IndexSpace> &sources, 
                    RtEvent initialized, std::set<RtEvent> *applied /*=NULL*/)
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
      Provenance *provenance = NULL;
      if (prov != NULL)
        provenance = new Provenance(prov);
      IndexSpaceExpression *expr = union_index_spaces(exprs);
      return expr->create_node(handle, did, initialized, provenance, applied);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_intersection_space(
                                      IndexSpace handle, DistributedID did,
                                      const char *prov,
                                      const std::vector<IndexSpace> &sources, 
                                      RtEvent init, std::set<RtEvent> *applied)
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
      Provenance *provenance = NULL;
      if (prov != NULL)
        provenance = new Provenance(prov);
      IndexSpaceExpression *expr = intersect_index_spaces(exprs);
      return expr->create_node(handle, did, init, provenance, applied);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_difference_space(
                                       IndexSpace handle, DistributedID did,
                                       const char *prov,
                                       IndexSpace left, IndexSpace right, 
                                       RtEvent init, std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(left.exists());
#endif
      IndexSpaceNode *lhs = get_node(left);
      Provenance *provenance = NULL;
      if (prov != NULL)
        provenance = new Provenance(prov);
      if (!right.exists())
        return lhs->create_node(handle, did, init, provenance, applied);
      IndexSpaceNode *rhs = get_node(right);
      IndexSpaceExpression *expr = subtract_index_spaces(lhs, rhs);
      return expr->create_node(handle, did, init, provenance, applied);
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
                                                    ApUserEvent partial_pending,
                                                    std::set<RtEvent> *applied)
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
      IndexPartNode *node = NULL;
      RtUserEvent disjointness_event;
      if ((part_kind == LEGION_COMPUTE_KIND) || 
          (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        // Use 1 if we know it's complete, 0 if it's not, 
        // otherwise -1 since we don't know
        const int complete = (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ? 1 :
                         (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND) ? 0 : -1;
        disjointness_event = Runtime::create_rt_user_event();
        node = create_node(pid, parent_node, color_node, partition_color,
                disjointness_event, complete, did, provenance, partition_ready,
                partial_pending, RtEvent::NO_RT_EVENT, applied);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_index_partition(parent.id, pid.id, -1/*unknown*/,
              complete, partition_color, runtime->address_space, 
              (provenance == NULL) ? NULL : provenance->provenance.c_str());
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
        node = create_node(pid, parent_node, color_node, partition_color, 
                    disjoint, complete, did, provenance, partition_ready,
                    partial_pending, RtEvent::NO_RT_EVENT, applied);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_index_partition(parent.id, pid.id, disjoint ? 1 : 0,
              complete, partition_color, runtime->address_space,
              (provenance == NULL) ? NULL : provenance->provenance.c_str());
	if (runtime->profiler != NULL)
	  runtime->profiler->record_index_partition(parent.id,pid.id, disjoint,
						    partition_color);
      }
      // If we need to compute the disjointness, only do that
      // after the partition is actually ready
      if ((part_kind == LEGION_COMPUTE_KIND) || 
          (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
#ifdef DEBUG_LEGION
        assert(disjointness_event.exists());
#endif
        // Record a reference on this node to prevent it from being
        // collected until the disjointness test is done
        node->add_base_resource_ref(APPLICATION_REF);
        // Launch a task to compute the disjointness
        DisjointnessArgs args(pid, disjointness_event);
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                                     Runtime::protect_event(partition_ready));
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
                                                 std::set<RtEvent> &safe_events)
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
      std::set<LegionColor> existing_colors;
      std::vector<IndexSpaceNode*> children_nodes;
      while (part_color == INVALID_COLOR)
      {
        // If this is the first time through populate the existing colors
        if (existing_colors.empty())
        {
          if (base->total_children == base->max_linearized_color)
          {
            for (LegionColor color = 0; color < base->total_children; color++)
            {
              IndexSpaceNode *child_node = base->get_child(color);
              children_nodes.push_back(child_node);
              std::vector<LegionColor> colors;
              child_node->get_colors(colors);
              if (!colors.empty())
                existing_colors.insert(colors.begin(), colors.end());
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
              child_node->get_colors(colors);
              if (!colors.empty())
                existing_colors.insert(colors.begin(), colors.end());
            }
            delete itr;
          }
        }
        // Find the next available color
        if (!existing_colors.empty())
        {
          std::set<LegionColor>::const_iterator next = existing_colors.begin();
          if ((*next) == 0)
          {
            std::set<LegionColor>::const_iterator prev = next++;
            while (next != existing_colors.end())
            {
              if ((*next) != ((*prev) + 1))
              {
                part_color = (*prev) + 1;
                break;
              }
              prev = next++;
            }
            if (part_color == INVALID_COLOR)
              part_color = (*prev) + 1;
          }
          else
            part_color = 0; 
        }
        else
          part_color = 0;
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
        for (std::vector<IndexSpaceNode*>::const_iterator it = 
              children_nodes.begin(); it != children_nodes.end(); it++)
        {
          IndexSpaceNode *child_node = *it;
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag()); 
          DistributedID did = 
            runtime->get_available_distributed_id();
          const RtEvent safe =
            create_pending_partition(ctx, pid, child_node->handle, 
                                     source->color_space->handle, 
                                     part_color, kind, did,
                                     provenance, domain_ready,
                                     ApUserEvent::NO_AP_USER_EVENT, 
                                     &safe_events);
          // If the user requested the handle for this point return it
          std::map<IndexSpace,IndexPartition>::iterator finder = 
            user_handles.find(child_node->handle);
          if (finder != user_handles.end())
            finder->second = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
      }
      else if (base->total_children == base->max_linearized_color)
      {
        for (LegionColor color = 0; color < base->total_children; color++)
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
                                     provenance, domain_ready, 
                                     ApUserEvent::NO_AP_USER_EVENT, 
                                     &safe_events);
          // If the user requested the handle for this point return it
          std::map<IndexSpace,IndexPartition>::iterator finder = 
            user_handles.find(child_node->handle);
          if (finder != user_handles.end())
            finder->second = pid;
          if (safe.exists())
            safe_events.insert(safe);
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
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag()); 
          DistributedID did = 
            runtime->get_available_distributed_id();
          const RtEvent safe = 
            create_pending_partition(ctx, pid, child_node->handle, 
                                     source->color_space->handle, 
                                     part_color, kind, did,
                                     provenance, domain_ready, 
                                     ApUserEvent::NO_AP_USER_EVENT,
                                     &safe_events);
          // If the user requested the handle for this point return it
          std::map<IndexSpace,IndexPartition>::iterator finder = 
            user_handles.find(child_node->handle);
          if (finder != user_handles.end())
            finder->second = pid;
          if (safe.exists())
            safe_events.insert(safe);
        }
        delete itr;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::compute_partition_disjointness(IndexPartition handle,
                                                        RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      node->compute_disjointness(ready_event);
      // Remove the reference that we added when launching this task
      if (node->remove_base_resource_ref(APPLICATION_REF))
        delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_space(IndexSpace handle,
                              AddressSpaceID source, std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      WrapperReferenceMutator mutator(applied);
      if (node->is_owner())
      {
        node->invalidate_root(source, applied);  
        if (node->remove_base_valid_ref(APPLICATION_REF, &mutator))
          delete node;
      }
      else
      {
        runtime->send_index_space_destruction(handle,node->owner_space,applied);
        if (node->remove_base_valid_ref(REMOTE_DID_REF, &mutator))
          delete node;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition handle,
                                                   std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        IndexPartNode::get_owner_space(handle, runtime);
      if (owner_space == runtime->address_space)
      {
        IndexPartNode *node = get_node(handle);
        WrapperReferenceMutator mutator(applied);
        if (node->remove_base_valid_ref(APPLICATION_REF, &mutator))
          delete node;
      }
      else
        runtime->send_index_partition_destruction(handle, owner_space, applied);
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
                                                       const FutureMap &weights,
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
                                                    const FutureMap &future_map,
                                                    bool perform_intersections) 
    //--------------------------------------------------------------------------
    {
      IndexPartNode *new_part = get_node(pid);
      return new_part->parent->create_by_domain(op, new_part, future_map.impl, 
                                              perform_intersections);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_cross_product_partitions(Operation *op,
                                                         IndexPartition base,
                                                         IndexPartition source,
                                                         LegionColor part_color)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *base_node = get_node(base);
      IndexPartNode *source_node = get_node(source);
      std::set<ApEvent> ready_events;
      if (base_node->total_children == base_node->max_linearized_color)
      {
        for (LegionColor color = 0; color < base_node->total_children; color++)
        {
          IndexSpaceNode *child_node = base_node->get_child(color);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready = child_node->create_by_intersection(op, part_node, 
                                                             source_node); 
          ready_events.insert(ready);
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          base_node->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNode *child_node = base_node->get_child(color);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready = child_node->create_by_intersection(op, part_node, 
                                                             source_node); 
          ready_events.insert(ready);
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
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_image(op, partition, projection,
                                        instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::create_partition_by_image_range(Operation *op,
                                                      IndexPartition pending,
                                                      IndexPartition proj,
                              const std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *partition = get_node(pending);
      IndexPartNode *projection = get_node(proj);
      return partition->parent->create_by_image_range(op, partition, projection,
                                              instances, instances_ready);
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
    IndexSpace RegionTreeForest::find_pending_space(IndexPartition parent,
                                                    const void *realm_color,
                                                    TypeTag type_tag,
                                                    ApUserEvent &domain_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *parent_node = get_node(parent);
      LegionColor child_color = 
        parent_node->color_space->linearize_color(realm_color, type_tag);
      IndexSpaceNode *child_node = parent_node->get_child(child_color);
      if (!parent_node->get_pending_child(child_color, domain_ready))
        REPORT_LEGION_ERROR(ERROR_INVALID_PENDING_CHILD,
          "Invalid pending child!")
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op, 
       IndexSpace target, const std::vector<IndexSpace> &handles, bool is_union)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      return child_node->compute_pending_space(op, handles, is_union);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op, 
                        IndexSpace target, IndexPartition handle, bool is_union)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
      return child_node->compute_pending_space(op, handle, is_union);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::compute_pending_space(Operation *op,
                                         IndexSpace target, IndexSpace initial,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *child_node = get_node(target);
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
                DistributedID did, const char *prov, std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
      Provenance *provenance = NULL;
      if (prov != NULL)
        provenance = new Provenance(prov);
      return create_node(handle, did, RtEvent::NO_RT_EVENT, provenance,applied);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace handle,
                                               std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        FieldSpaceNode::get_owner_space(handle, runtime);
      if (owner_space == runtime->address_space)
      {
        FieldSpaceNode *node = get_node(handle);
        WrapperReferenceMutator mutator(applied);
        if (node->remove_base_valid_ref(APPLICATION_REF, &mutator))
          delete node;
      }
      else
        runtime->send_field_space_destruction(handle, owner_space, applied);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::allocate_field(FieldSpace handle, size_t field_size,
                        FieldID fid, CustomSerdezID serdez_id, const char *prov)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_field(fid, field_size, serdez_id, prov);
      if (ready.exists())
        ready.wait();
      return false;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::allocate_field(FieldSpace handle,
    ApEvent size_ready, FieldID fid, CustomSerdezID serdez_id, const char *prov)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_field(fid, size_ready, serdez_id, prov);
      if (ready.exists())
        ready.wait();
      return node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_field(FieldSpace handle, FieldID fid,
                                      std::set<RtEvent> &applied) 
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_field(fid, runtime->address_space, applied);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::allocate_fields(FieldSpace handle, 
                                     const std::vector<size_t> &sizes,
                                     const std::vector<FieldID> &fields,
                                     CustomSerdezID serdez_id, const char *prov)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sizes.size() == fields.size());
#endif
      // We know that none of these field allocations are local
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_fields(sizes, fields, serdez_id, prov);
      // Wait for this to exist
      if (ready.exists())
        ready.wait();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::allocate_fields(FieldSpace handle, 
                                     ApEvent sizes_ready,
                                     const std::vector<FieldID> &fields,
                                     CustomSerdezID serdez_id, const char *prov)
    //--------------------------------------------------------------------------
    {
      // We know that none of these field allocations are local
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_fields(sizes_ready, fields,serdez_id,prov);
      // Wait for this to exist
      if (ready.exists())
        ready.wait();
      return node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_fields(FieldSpace handle,
                                       const std::vector<FieldID> &to_free,
                                       std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_fields(to_free, runtime->address_space, applied);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_field_indexes(FieldSpace handle,
                                        const std::vector<FieldID> &to_free,
                                        RtEvent freed_event)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_field_indexes(to_free, freed_event);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::allocate_local_fields(FieldSpace handle,
                                      const std::vector<FieldID> &fields,
                                      const std::vector<size_t> &sizes,
                                      CustomSerdezID serdez_id, 
                                      const std::set<unsigned> &current_indexes,
                                            std::vector<unsigned> &new_indexes,
                                            const char *provenance)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      return node->allocate_local_fields(fields, sizes, serdez_id,
                         current_indexes, new_indexes, provenance);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_local_fields(FieldSpace handle,
                                           const std::vector<FieldID> &to_free,
                                           const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_local_fields(to_free, indexes);
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
    void RegionTreeForest::create_logical_region(LogicalRegion handle,
                                      DistributedID did, const char *provenance,
                                      std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance); 
      create_node(handle, NULL/*parent*/,RtEvent::NO_RT_EVENT,did,prov,applied);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_logical_region(LogicalRegion handle,
                                                  std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        RegionNode::get_owner_space(handle, runtime);
      if (owner_space == runtime->address_space)
      {
        RegionNode *node = get_node(handle);
        WrapperReferenceMutator mutator(applied);
        if (node->remove_base_valid_ref(APPLICATION_REF, &mutator))
          delete node;
      }
      else
        runtime->send_logical_region_destruction(handle, owner_space, applied);
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
                                        RegionRequirement &req,
                                        const ProjectionInfo &projection_info,
                                        RegionTreePath &path,
                                        std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_LOGICAL_ANALYSIS_CALL);
      // If this is a NO_ACCESS, then we'll have no dependences so we're done
      if (IS_NO_ACCESS(req))
        return;
      TaskContext *context = op->find_logical_context(idx);
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      RegionNode *parent_node = get_node(req.parent);
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      // Then compute the logical user
      LogicalUser user(op, idx, RegionUsage(req), user_mask); 
      LogicalTraceInfo trace_info(op->already_traced(), 
                                  op->get_trace(), idx, req); 
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
        parent_node->register_logical_user(ctx.get_id(), user, path, trace_info,
           projection_info, unopened_mask, already_closed_mask, applied_events);
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
    void RegionTreeForest::perform_deletion_analysis(DeletionOp *op, 
                                                     unsigned idx,
                                                     RegionRequirement &req,
                                                     RegionTreePath &path,
                                                     std::set<RtEvent> &applied,
                                                     bool invalidate_tree)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_LOGICAL_ANALYSIS_CALL);
      TaskContext *context = op->find_logical_context(idx);
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      RegionNode *parent_node = get_node(req.parent);
      
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      // Then compute the logical user
      LogicalUser user(op, idx, RegionUsage(req), user_mask);
      LogicalTraceInfo trace_info(op->already_traced(),op->get_trace(),idx,req);
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), true/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Do the traversal
      FieldMask already_closed_mask;
      parent_node->register_logical_deletion(ctx.get_id(), user, user_mask,
          path, trace_info, already_closed_mask, applied, invalidate_tree);
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
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::find_open_complete_partitions(Operation *op,
                                                         unsigned idx,
                                                  const RegionRequirement &req,
                                      std::vector<LogicalPartition> &partitions)
    //--------------------------------------------------------------------------
    {
      TaskContext *context = op->find_logical_context(idx);
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
    void RegionTreeForest::send_back_logical_state(RegionTreeContext ctx,
                      UniqueID context_uid, const RegionRequirement &req, 
                      AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *top_node = get_node(req.region);
      top_node->send_back_logical_state(ctx.get_id(), context_uid, target);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_versioning_analysis(Operation *op,
                     unsigned idx, const RegionRequirement &req,
                     VersionInfo &version_info, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_VERSIONING_ANALYSIS_CALL);
      if (IS_NO_ACCESS(req))
        return;
      InnerContext *context = op->find_physical_context(idx, req);
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
      assert((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
      ((req.handle_type == LEGION_REGION_PROJECTION) && (req.projection == 0)));
#endif
      RegionNode *region_node = get_node(req.region);
      FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
      const RtEvent ready = 
        region_node->perform_versioning_analysis(ctx.get_id(), context, 
                                     &version_info, req.parent, user_mask, op);
      if (ready.exists())
        ready_events.insert(ready);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_versions(RegionTreeContext ctx, 
                                               LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Check to see if this has already been deleted
      RegionNode *node = find_local_node(handle);
      if (node == NULL)
        return;
      VersioningInvalidator invalidator(ctx);
      node->visit_node(&invalidator);
      if (node->remove_base_resource_ref(REGION_TREE_REF))
        delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_all_versions(RegionTreeContext ctx)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeID,RegionNode*> trees;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        trees = tree_nodes;
        for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
              trees.begin(); it != trees.end(); it++)
          it->second->add_base_resource_ref(REGION_TREE_REF);
      }
      VersioningInvalidator invalidator(ctx); 
      for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
            trees.begin(); it != trees.end(); it++)
      {
        it->second->visit_node(&invalidator);
        if (it->second->remove_base_resource_ref(REGION_TREE_REF))
          delete it->second;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_current_context(RegionTreeContext ctx,
                  const RegionRequirement &req, const bool restricted,
                  const InstanceSet &sources, ApEvent term_event, 
                  InnerContext *context,unsigned init_index,
                  std::map<PhysicalManager*,InstanceView*> &top_views,
                  std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_INITIALIZE_CONTEXT_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode *top_node = get_node(req.region);
      RegionUsage usage(req);
      FieldMask user_mask = 
        top_node->column_source->get_field_mask(req.privilege_fields);
      // Do the normal versioning analysis since this will deal with
      // any aliasing of physical instances in the region requirements
      VersionInfo init_version_info;
      // Perform the version analysis and make it ready
      const RtEvent eq_ready = 
        top_node->perform_versioning_analysis(ctx.get_id(), context,
               &init_version_info, req.region, user_mask, context->owner_task);
      // Now get the top-views for all the physical instances
      std::vector<InstanceView*> corresponding(sources.size());
      const AddressSpaceID local_space = context->runtime->address_space;
      const UniqueID context_uid = context->get_unique_id();
      IndexSpaceExpression *reg_expr = top_node->get_index_space_expression();
      // Build our set of corresponding views
      if (IS_REDUCE(req))
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const InstanceRef &src_ref = sources[idx];
          PhysicalManager *manager = src_ref.get_physical_manager();
          const FieldMask &view_mask = src_ref.get_valid_fields();
#ifdef DEBUG_LEGION
          assert(!(view_mask - user_mask)); // should be dominated
#endif
          // Check to see if the view exists yet or not
          std::map<PhysicalManager*,InstanceView*>::const_iterator 
            finder = top_views.find(manager);
          if (finder == top_views.end())
          {
            ReductionView *new_view = 
              context->create_instance_top_view(manager,
                  local_space)->as_reduction_view();
            top_views[manager] = new_view;
            corresponding[idx] = new_view;
            // Record the initial user for the instance
            new_view->add_initial_user(term_event, usage, view_mask,
                                       reg_expr, context_uid, init_index);
          }
          else
          {
            corresponding[idx] = finder->second;
            // Record the initial user for the instance
            finder->second->add_initial_user(term_event, usage, view_mask,
                                             reg_expr, context_uid, init_index);
          }
        }
      }
      else
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const InstanceRef &src_ref = sources[idx];
          PhysicalManager *manager = src_ref.get_physical_manager();
          const FieldMask &view_mask = src_ref.get_valid_fields();
#ifdef DEBUG_LEGION
          assert(!(view_mask - user_mask)); // should be dominated
#endif
          // Check to see if the view exists yet or not
          std::map<PhysicalManager*,InstanceView*>::const_iterator 
            finder = top_views.find(manager);
          if (finder == top_views.end())
          {
            MaterializedView *new_view = 
             context->create_instance_top_view(manager, 
                 local_space)->as_materialized_view();
            top_views[manager] = new_view;
            corresponding[idx] = new_view;
            // Record the initial user for the instance
            new_view->add_initial_user(term_event, usage, view_mask,
                                       reg_expr, context_uid, init_index);
          }
          else
          {
            corresponding[idx] = finder->second;
            // Record the initial user for the instance
            finder->second->add_initial_user(term_event, usage, view_mask,
                                             reg_expr, context_uid, init_index);
          }
        }
      }
      if (eq_ready.exists() && !eq_ready.has_triggered())
        eq_ready.wait();
      // Iterate over the equivalence classes and initialize them
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        init_version_info.get_equivalence_sets();
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        it->first->initialize_set(usage, it->second, restricted,
                                  sources, corresponding, applied_events);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_current_context(RegionTreeContext ctx,
                                          bool users_only, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_INVALIDATE_CONTEXT_CALL);
      // Handle the case where we already deleted the region tree
      RegionNode *top_node = find_local_node(handle);
      if (top_node == NULL)
        return;
      CurrentInvalidator invalidator(ctx.get_id(), users_only);
      top_node->visit_node(&invalidator);
      if (top_node->remove_base_resource_ref(REGION_TREE_REF))
        delete top_node;
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
                                                  VersionInfo &version_info,
                                                  InstanceSet &targets,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PREMAP_ONLY_CALL);
      // If we are a NO_ACCESS or there are no fields then we are already done 
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return;
      // Iterate over the equivalence sets and get all the instances that
      // are valid for all the different equivalence classes
      const FieldMaskSet<EquivalenceSet> &eq_sets =
        version_info.get_equivalence_sets();
      ValidInstAnalysis analysis(runtime, op, index, version_info, 
                                 IS_REDUCE(req) ? req.redop : 0);
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis.traverse(it->first, it->second, deferral_events,
                          map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent ready;
      if (traversal_done.exists() || analysis.has_remote_sets())
        ready = analysis.perform_remote(traversal_done, map_applied_events);
      // Wait for all the responses to be ready
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      FieldMaskSet<InstanceView> instances;
      if (analysis.report_instances(instances))
        req.flags |= LEGION_RESTRICTED_FLAG;
      const std::vector<LogicalRegion> to_meet(1, req.region); 
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        PhysicalManager *manager = it->first->get_manager();
        if (manager->meets_regions(to_meet))
          targets.add_instance(InstanceRef(manager, it->second));
      }
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::physical_perform_updates(
                               const RegionRequirement &req,
                               VersionInfo &version_info,
                               Operation *op, unsigned index,
                               ApEvent precondition, ApEvent term_event,
                               const InstanceSet &targets,
                               const PhysicalTraceInfo &trace_info,
                               std::set<RtEvent> &map_applied_events,
                               UpdateAnalysis *&analysis,
#ifdef DEBUG_LEGION
                               const char *log_name,
                               UniqueID uid,
#endif
                               const bool record_valid,
                               const bool check_initialized,
                               const bool defer_copies)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_REGISTER_ONLY_CALL);
      // If we are a NO_ACCESS or there are no fields then we are already done 
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return RtEvent::NO_RT_EVENT;
      InnerContext *context = op->find_physical_context(index, req);
#ifdef DEBUG_LEGION
      RegionTreeContext ctx = context->get_context();
      assert(ctx.exists());
      assert((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
              (req.handle_type == LEGION_REGION_PROJECTION));
      assert(!targets.empty());
      assert(!targets.is_virtual_mapping());
#endif
      RegionNode *region_node = get_node(req.region);
      FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
#ifdef DEBUG_LEGION 
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     region_node, ctx.get_id(), 
                                     true/*before*/, false/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                     FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Perform the registration
      std::vector<InstanceView*> target_views;
      context->convert_target_views(targets, target_views);
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();
#ifdef DEBUG_LEGION
      assert(analysis == NULL);
      // Should be recording or must be read-only
      assert(record_valid || IS_READ_ONLY(req));
#endif
      analysis = new UpdateAnalysis(runtime, op, index, version_info, req, 
                                    region_node, targets, target_views, 
                                    trace_info, precondition, term_event, 
                                    check_initialized, record_valid);
      analysis->add_reference();
      // Iterate over all the equivalence classes and perform the analysis
      // Only need to check for uninitialized data for things not discarding
      // and things that are not simultaneous (simultaneous can appear 
      // uninitialized since it might be reading, but then use internal
      // synchronization to wait for something running concurrently to write)
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis->traverse(it->first, it->second, deferral_events,
                           map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready = 
          analysis->perform_remote(traversal_done, map_applied_events);
      // Then perform the updates
      const RtEvent updates_ready = 
        analysis->perform_updates(traversal_done, map_applied_events);
      return Runtime::merge_events(remote_ready, updates_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::physical_perform_registration(
                                         UpdateAnalysis *analysis,
                                         InstanceSet &targets,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      // If we are a NO_ACCESS or there are no fields then analysis will be NULL
      if (analysis == NULL)
        return ApEvent::NO_AP_EVENT;
      // We can skip this if the term event is a 
      // no-event (happens with post-mapping and copies)
      if (analysis->term_event.exists())
      {
        const RtEvent collect_event = trace_info.get_collect_event();
        // Perform the registration
        IndexSpaceNode *local_expr = analysis->node->row_source;
        const UniqueID op_id = analysis->op->get_unique_op_id();
        const AddressSpaceID local_space = runtime->address_space;
        if (analysis->user_registered.exists())
        {
          std::set<RtEvent> user_applied;
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            const FieldMask &inst_mask = targets[idx].get_valid_fields();
            ApEvent ready = analysis->target_views[idx]->register_user(
                analysis->usage, inst_mask, local_expr, op_id, analysis->index, 
                analysis->term_event, collect_event,
                user_applied, trace_info, local_space);
            // Record the event as the precondition for the task
            targets[idx].set_ready_event(ready);
            if (trace_info.recording)
              trace_info.record_op_view(
                  analysis->usage, inst_mask, analysis->target_views[idx]);
          }
          if (!user_applied.empty())
          {
            Runtime::trigger_event(analysis->user_registered, 
                Runtime::merge_events(user_applied));
            map_applied_events.insert(analysis->user_registered);
          }
          else
            Runtime::trigger_event(analysis->user_registered);
        }
        else
        {
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            const FieldMask &inst_mask = targets[idx].get_valid_fields();
            ApEvent ready = analysis->target_views[idx]->register_user(
                analysis->usage, inst_mask, local_expr, op_id, analysis->index,
                analysis->term_event, collect_event, map_applied_events, 
                trace_info, local_space);
            // Record the event as the precondition for the task
            targets[idx].set_ready_event(ready);
            if (trace_info.recording)
              trace_info.record_op_view(
                analysis->usage, inst_mask, analysis->target_views[idx]);
          }
        }
      }
      else if (analysis->user_registered.exists())
        Runtime::trigger_event(analysis->user_registered);
      // Find any atomic locks we need to take for these instances
      if (IS_ATOMIC(analysis->usage))
      {
        const bool exclusive = HAS_WRITE(analysis->usage);
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          const FieldMask &inst_mask = targets[idx].get_valid_fields();
          analysis->target_views[idx]->find_atomic_reservations(inst_mask, 
                                analysis->op, analysis->index, exclusive);
        }
      }
      // Perform any output copies (e.g. for restriction) that need to be done
      ApEvent result;
      if (analysis->has_output_updates())
        result = 
          analysis->perform_output(RtEvent::NO_RT_EVENT, map_applied_events);
      // Remove the reference that we added in the updates step
      if (analysis->remove_reference())
        delete analysis;
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::physical_perform_updates_and_registration(
                                       const RegionRequirement &req,
                                       VersionInfo &version_info,
                                       Operation *op, unsigned index,
                                       ApEvent precondition, 
                                       ApEvent term_event,
                                       InstanceSet &targets,
                                       const PhysicalTraceInfo &trace_info,
                                       std::set<RtEvent> &map_applied_events,
#ifdef DEBUG_LEGION
                                       const char *log_name,
                                       UniqueID uid,
#endif
                                       const bool record_valid,
                                       const bool check_initialized)
    //--------------------------------------------------------------------------
    {
      UpdateAnalysis *analysis = NULL;
      const RtEvent registration_precondition = physical_perform_updates(req,
         version_info, op, index, precondition, term_event, targets, trace_info,
         map_applied_events, analysis,
#ifdef DEBUG_LEGION
         log_name, uid,
#endif
         record_valid, check_initialized, false/*defer copies*/);
      if (registration_precondition.exists() && 
          !registration_precondition.has_triggered())
        registration_precondition.wait();
      return physical_perform_registration(analysis, targets, trace_info, 
                                           map_applied_events);
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::defer_physical_perform_registration(RtEvent pre,
                         UpdateAnalysis *analysis, InstanceSet &targets,
                         std::set<RtEvent> &map_applied_events,
                         ApEvent &result, const PhysicalTraceInfo &info)
    //--------------------------------------------------------------------------
    {
      RtUserEvent map_applied_done = Runtime::create_rt_user_event();
      map_applied_events.insert(map_applied_done);
      DeferPhysicalRegistrationArgs args(analysis->op->get_unique_op_id(),
                             analysis, targets, map_applied_done, result, info);
      return runtime->issue_runtime_meta_task(args, 
                    LG_LATENCY_WORK_PRIORITY, pre);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::handle_defer_registration(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPhysicalRegistrationArgs *dargs = 
        (const DeferPhysicalRegistrationArgs*)args;
      std::set<RtEvent> applied_events;
      dargs->result = physical_perform_registration(dargs->analysis, 
                        dargs->targets, *dargs, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(dargs->map_applied_done,
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(dargs->map_applied_done);
      if (dargs->analysis->remove_reference())
        delete dargs->analysis;
      dargs->remove_recorder_reference();
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::acquire_restrictions(
                                         const RegionRequirement &req,
                                         VersionInfo &version_info,
                                         AcquireOp *op, unsigned index,
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
      // Iterate through the equivalence classes and find all the restrictions
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();
      AcquireAnalysis analysis(runtime, op, index, version_info);
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis.traverse(it->first, it->second, deferral_events,
                          map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis.has_remote_sets())
        remote_ready = 
          analysis.perform_remote(traversal_done, map_applied_events);
      if (remote_ready.exists() && !remote_ready.has_triggered())
        remote_ready.wait();
      FieldMaskSet<InstanceView> instances;
      analysis.report_instances(instances);
      // Fill in the restricted instances and record users
      std::set<ApEvent> acquired_events;
      restricted_instances.resize(instances.size());
      unsigned inst_index = 0;
      const RegionUsage usage(req);
      const UniqueID op_id = op->get_unique_op_id();
      IndexSpaceNode *local_expr = get_node(req.region.get_index_space());
      const RtEvent collect_event = trace_info.get_collect_event();
      // Now add users for all the instances
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            instances.begin(); it != instances.end(); it++, inst_index++)
      {
        restricted_instances[inst_index++] = 
          InstanceRef(it->first->get_manager(), it->second);
        ApEvent ready = it->first->register_user(usage, it->second,
            local_expr, op_id, index, term_event, collect_event,
            map_applied_events, trace_info, runtime->address_space);
        if (ready.exists())
          acquired_events.insert(ready);
      }
      if (!acquired_events.empty())
        return Runtime::merge_events(&trace_info, acquired_events);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::release_restrictions(
                                         const RegionRequirement &req,
                                         VersionInfo &version_info,
                                         ReleaseOp *op, unsigned index,
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
      assert(IS_EXCLUSIVE(req));
#endif
      // Iterate through the equivalence classes and find all the restrictions
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();
      std::set<RtEvent> deferral_events;
      ReleaseAnalysis analysis(runtime, op, index, precondition, 
                               version_info, trace_info);
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis.traverse(it->first, it->second, deferral_events,
                          map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis.has_remote_sets())
        remote_ready = 
          analysis.perform_remote(traversal_done, map_applied_events);
      // Issue any release copies/fills that need to be done
      const RtEvent updates_done = 
        analysis.perform_updates(traversal_done, map_applied_events);
      // Wait for any remote releases to come back to us before we 
      // attempt to get the set of valid instances
      if (remote_ready.exists() && !remote_ready.has_triggered())
        remote_ready.wait();
      FieldMaskSet<InstanceView> instances;
      analysis.report_instances(instances);
      // Now we can register our users
      std::set<ApEvent> released_events;
      restricted_instances.resize(instances.size());
      unsigned inst_index = 0;
      const RegionUsage usage(req);
      const UniqueID op_id = op->get_unique_op_id();
      IndexSpaceNode *local_expr = get_node(req.region.get_index_space());
      // Make sure we're done applying our updates before we do our registration
      if (updates_done.exists() && !updates_done.has_triggered())
        updates_done.wait();
      const RtEvent collect_event = trace_info.get_collect_event();
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            instances.begin(); it != instances.end(); it++, inst_index++)
      {
        restricted_instances[inst_index++] = 
          InstanceRef(it->first->get_manager(), it->second);
        ApEvent ready = it->first->register_user(usage, it->second,
            local_expr, op_id, index, term_event, collect_event,
            map_applied_events, trace_info, runtime->address_space);
        if (ready.exists())
          released_events.insert(ready);
      }
      if (!released_events.empty())
        return Runtime::merge_events(&trace_info, released_events);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::copy_across(
                                        const RegionRequirement &src_req,
                                        const RegionRequirement &dst_req,
                                        VersionInfo &src_version_info,
                                        VersionInfo &dst_version_info,
                                        const InstanceSet &src_targets,
                                        const InstanceSet &dst_targets,
                                        CopyOp *op, 
                                        unsigned src_index, unsigned dst_index,
                                        ApEvent precondition, PredEvent guard, 
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
      WrapperReferenceMutator mutator(map_applied_events);
      IndexSpaceExpression *copy_expr = intersect_index_spaces(
          src_node->row_source, dst_node->row_source, &mutator);
      // Quick out if there is nothing to copy to
      if (copy_expr->is_empty())
        return ApEvent::NO_AP_EVENT;
      // Perform the copies/reductions across
      InnerContext *context = op->find_physical_context(dst_index, dst_req);
      std::vector<InstanceView*> target_views;
      context->convert_target_views(dst_targets, target_views);
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
        InnerContext *src_context = 
          op->find_physical_context(src_index, src_req);
        std::vector<InstanceView*> source_views;
        src_context->convert_target_views(src_targets, source_views);
        across->initialize_source_fields(this, src_req,
                              src_targets, source_views, trace_info);
        // Fill in the destination fields 
        const bool exclusive_redop = 
          IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req);
        across->initialize_destination_fields(this, dst_req, dst_targets, 
                              target_views, trace_info, exclusive_redop);
        // Get the preconditions for this copy
        std::vector<ApEvent> copy_preconditions;
        if (precondition.exists())
          copy_preconditions.push_back(precondition);
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
        {
          const ApEvent ready = src_targets[idx].get_ready_event();
          if (ready.exists())
            copy_preconditions.push_back(ready);
        }
        for (unsigned idx = 0; idx < dst_targets.size(); idx++)
        {
          const ApEvent ready = dst_targets[idx].get_ready_event();
          if (ready.exists())
            copy_preconditions.push_back(ready);
        }
        if (!copy_preconditions.empty())
          precondition = Runtime::merge_events(&trace_info, copy_preconditions);
        ApEvent result = across->execute(op, guard, precondition,
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT, trace_info);
        if (trace_info.recording)
        {
          // Record this with the trace
          trace_info.record_issue_across(result, precondition, precondition,
                        ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT, across);
          FieldMaskSet<InstanceView> tracing_srcs, tracing_dsts;
          for (unsigned idx = 0; idx < src_targets.size(); idx++)
            tracing_srcs.insert(source_views[idx],
                src_targets[idx].get_valid_fields());
          for (unsigned idx = 0; idx < dst_targets.size(); idx++)
            tracing_dsts.insert(target_views[idx],
                dst_targets[idx].get_valid_fields());
          trace_info.record_copy_views(result, copy_expr,
                                       tracing_srcs, tracing_dsts,
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
      const FieldMaskSet<EquivalenceSet> &src_eq_sets = 
        src_version_info.get_equivalence_sets();
      // Check to see if we have a perfect across-copy
      bool perfect = true;
      for (unsigned idx = 0; idx < src_indexes.size(); idx++)
      {
        if (src_indexes[idx] == dst_indexes[idx])
          continue;
        perfect = false;
        break;
      }
      CopyAcrossAnalysis *analysis = new CopyAcrossAnalysis(runtime, op, 
          src_index, dst_index, src_version_info, src_req, dst_req, 
          dst_targets, target_views, precondition, guard, dst_req.redop,
          src_indexes, dst_indexes, trace_info, perfect);
      analysis->add_reference();
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            src_eq_sets.begin(); it != src_eq_sets.end(); it++)
      {
        // Check that the index spaces intersect
        IndexSpaceExpression *overlap = 
          intersect_index_spaces(it->first->set_expr, copy_expr);
        if (overlap->is_empty())
          continue;
        // No alt-set tracking here because some equivalence sets
        // may need to be traversed multiple times
        it->first->issue_across_copies(*analysis, it->second, overlap,
                                       deferral_events, map_applied_events);
      }
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
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
      for (unsigned idx = 0; idx < src_targets.size(); idx++)
      {
        const ApEvent ready = src_targets[idx].get_ready_event();
        if (ready.exists())
          local_preconditions.push_back(ready);
      }
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
      across->initialize_source_indirections(this, src_records,
          src_req, idx_req, idx_target, op->index_point, gather_is_range, 
          possible_src_out_of_range);
#ifdef LEGION_SPY
      across->src_indirect_instance_event = 
        idx_target.get_physical_manager()->get_unique_event();
#endif
      
      // Initialize the destination fields
      InnerContext *context = op->find_physical_context(dst_index, dst_req);
      std::vector<InstanceView*> target_views;
      context->convert_target_views(dst_targets, target_views);
      const bool exclusive_redop =
          IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req);
      across->initialize_destination_fields(this, dst_req, dst_targets,
          target_views, trace_info, exclusive_redop);
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.push_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      for (unsigned idx = 0; idx < dst_targets.size(); idx++)
      {
        const ApEvent ready = dst_targets[idx].get_ready_event();
        if (ready.exists())
          copy_preconditions.push_back(ready);
      }
      ApEvent src_indirect_ready = idx_target.get_ready_event();
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
        // If we're tracing record the views for this copy
        FieldMaskSet<InstanceView> src_views, idx_views, dst_views;
        // Get the src_views
        InnerContext *src_context =
          op->find_physical_context(src_index, src_req);
        std::vector<InstanceView*> source_views;
        src_context->convert_target_views(src_targets, source_views);
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
          src_views.insert(source_views[idx],
              src_targets[idx].get_valid_fields());
        // Get the idx views
        InnerContext *idx_context =
          op->find_physical_context(idx_index, idx_req); 
        std::vector<InstanceView*> indirect_views;
        idx_context->convert_target_views(idx_targets, indirect_views);
        idx_views.insert(indirect_views.back(), idx_target.get_valid_fields());
        // Get the dst views
        for (unsigned idx = 0; idx < target_views.size(); idx++)
          dst_views.insert(target_views[idx],
              dst_targets[idx].get_valid_fields());
        IndexSpaceNode *src_node = get_node(src_req.region.get_index_space());
        trace_info.record_indirect_views(copy_post, collective_post, src_index,
            src_node, src_views, map_applied_events, LEGION_READ_PRIV);
        trace_info.record_copy_views(copy_post, idx_index, dst_index,
            LEGION_READ_PRIV, LEGION_WRITE_PRIV, copy_expr, idx_views,
            dst_views, true/*indirect*/, false/*indirect*/, map_applied_events);
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
      for (unsigned idx = 0; idx < dst_targets.size(); idx++)
      {
        const ApEvent ready = dst_targets[idx].get_ready_event();
        if (ready.exists())
          local_preconditions.push_back(ready);
      }
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
      InnerContext *context = op->find_physical_context(src_index, src_req);
      std::vector<InstanceView*> source_views;
      context->convert_target_views(src_targets, source_views);
      across->initialize_source_fields(this, src_req, src_targets,
                                       source_views, trace_info);
      // Initialize the destination indirections
      const InstanceRef idx_target = idx_targets[0];
      // Only exclusive if we're the only point sctatting to our instance
      // and we're not racing with any other operations
      const bool exclusive_redop = (dst_records.size() == 1) && 
        (IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req));
      across->initialize_destination_indirections(this, dst_records,
          dst_req, idx_req, idx_target, op->index_point, scatter_is_range,
          possible_dst_out_of_range, possible_dst_aliasing, exclusive_redop);
#ifdef LEGION_SPY
      across->dst_indirect_instance_event = 
        idx_target.get_physical_manager()->get_unique_event();
#endif 
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.push_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      for (unsigned idx = 0; idx < src_targets.size(); idx++)
      {
        const ApEvent ready = src_targets[idx].get_ready_event();
        if (ready.exists())
          copy_preconditions.push_back(ready);
      }
      ApEvent dst_indirect_ready = idx_target.get_ready_event();
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
        // If we're tracing record the views for this copy
        FieldMaskSet<InstanceView> src_views, dst_views, idx_views;
        // src_views
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
          src_views.insert(source_views[idx],
              src_targets[idx].get_valid_fields());
        // idx_views
        std::vector<InstanceView*> indirect_views;
        InnerContext *idx_context =
          op->find_physical_context(idx_index, idx_req);
        idx_context->convert_target_views(idx_targets, indirect_views);
        idx_views.insert(indirect_views.back(), idx_target.get_valid_fields());
        // dst_views
        std::vector<InstanceView*> target_views;
        InnerContext *dst_context =
          op->find_physical_context(dst_index, dst_req);
        dst_context->convert_target_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < target_views.size(); idx++)
          dst_views.insert(target_views[idx],
              dst_targets[idx].get_valid_fields());
        trace_info.record_copy_views(copy_post, src_index, idx_index,
            LEGION_READ_PRIV, LEGION_READ_PRIV, copy_expr, src_views,
            idx_views, false/*indirect*/, true/*indirect*/,map_applied_events); 
        IndexSpaceNode *dst_node = get_node(dst_req.region.get_index_space());
        trace_info.record_indirect_views(copy_post, collective_post, dst_index,
          dst_node, dst_views, map_applied_events, LEGION_WRITE_PRIV);
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
      for (unsigned idx = 0; idx < src_targets.size(); idx++)
      {
        const ApEvent ready = src_targets[idx].get_ready_event();
        if (ready.exists())
          local_preconditions.push_back(ready);
      }
      for (unsigned idx = 0; idx < dst_targets.size(); idx++)
      {
        const ApEvent ready = dst_targets[idx].get_ready_event();
        if (ready.exists())
          local_preconditions.push_back(ready);
      }
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
      across->initialize_source_indirections(this, src_records,
          src_req, src_idx_req, src_idx_target, op->index_point, 
          both_are_range, possible_src_out_of_range);
#ifdef LEGION_SPY
      across->src_indirect_instance_event = 
        src_idx_target.get_physical_manager()->get_unique_event();
#endif 
      // Initialize the destination indirections
      const InstanceRef &dst_idx_target = dst_idx_targets[0];
      // Only exclusive if we're the only point sctatting to our instance
      // and we're not racing with any other operations
      const bool exclusive_redop = (dst_records.size() == 1) && 
        (IS_EXCLUSIVE(dst_req) || IS_ATOMIC(dst_req));
      across->initialize_destination_indirections(this, dst_records,
          dst_req, dst_idx_req, dst_idx_target, op->index_point, both_are_range,
          possible_dst_out_of_range, possible_dst_aliasing, exclusive_redop);
#ifdef LEGION_SPY
      across->dst_indirect_instance_event = 
        dst_idx_target.get_physical_manager()->get_unique_event();
#endif 
      // Compute the copy preconditions
      std::vector<ApEvent> copy_preconditions;
      if (collective_pre.exists())
        copy_preconditions.push_back(collective_pre);
      else
        copy_preconditions.swap(local_preconditions);
      ApEvent src_indirect_ready = src_idx_target.get_ready_event();
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
      ApEvent dst_indirect_ready = dst_idx_target.get_ready_event();
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
        // If we're tracing record the views for this copy
        FieldMaskSet<InstanceView> src_views, src_idx_views;
        FieldMaskSet<InstanceView> dst_idx_views, dst_views;
        std::vector<InstanceView*> source_views;
        InnerContext *src_context =
          op->find_physical_context(src_index, src_req);
        src_context->convert_target_views(src_targets, source_views);
        for (unsigned idx = 0; idx < src_targets.size(); idx++)
          src_views.insert(source_views[idx],
              src_targets[idx].get_valid_fields());
        std::vector<InstanceView*> src_indirect_views, dst_indirect_views;
        InnerContext *src_idx_context =
          op->find_physical_context(src_idx_index, src_idx_req);
        src_idx_context->convert_target_views(src_idx_targets,
                                              src_indirect_views);
        src_idx_views.insert(src_indirect_views.back(),
            src_idx_target.get_valid_fields());
        InnerContext *dst_idx_context =
          op->find_physical_context(dst_idx_index, dst_idx_req);
        dst_idx_context->convert_target_views(dst_idx_targets,
                                              dst_indirect_views);
        dst_idx_views.insert(dst_indirect_views.back(),
            dst_idx_target.get_valid_fields());
        std::vector<InstanceView*> target_views;
        InnerContext *dst_context =
          op->find_physical_context(dst_index, dst_req);
        dst_context->convert_target_views(dst_targets, target_views);
        for (unsigned idx = 0; idx < target_views.size(); idx++)
          dst_views.insert(target_views[idx],
              dst_targets[idx].get_valid_fields());

        IndexSpaceNode *src_node = get_node(src_req.region.get_index_space());
        trace_info.record_indirect_views(copy_post, collective_post, src_index,
            src_node, src_views, map_applied_events, LEGION_READ_PRIV);
        IndexSpaceNode *dst_node = get_node(dst_req.region.get_index_space());
        trace_info.record_indirect_views(copy_post, collective_post, dst_index,
           dst_node, dst_views, map_applied_events, LEGION_WRITE_PRIV);
        trace_info.record_copy_views(copy_post, src_idx_index, dst_idx_index,
            LEGION_READ_PRIV, LEGION_READ_PRIV, copy_expr, src_idx_views,
            dst_idx_views,true/*indirect*/,true/*indirect*/,map_applied_events);
      }
      if (across->remove_reference())
        delete across;
      return copy_post;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::fill_fields(FillOp *op,
                                          const RegionRequirement &req,
                                          const unsigned index,
                                          FillView *fill_view,
                                          VersionInfo &version_info,
                                          ApEvent precondition,
                                          PredEvent true_guard, 
                                          const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_FILL_FIELDS_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      if (trace_info.recording)
      {
        RegionNode *region_node = get_node(req.region);
        FieldSpaceNode *fs_node = region_node->column_source;
        trace_info.record_post_fill_view(fill_view,
            fs_node->get_field_mask(req.privilege_fields));
      }
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();     
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime, op, index, 
          RegionUsage(req), version_info, fill_view, trace_info, precondition, 
          RtEvent::NO_RT_EVENT/*reg guard*/, true_guard, true/*track effects*/);
      analysis->add_reference();
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis->traverse(it->first, it->second, deferral_events,
                           map_applied_events, true/*original mask*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready = 
          analysis->perform_remote(traversal_done, map_applied_events);
      RtEvent output_ready;
      if (traversal_done.exists() || analysis->has_output_updates())
        output_ready = 
          analysis->perform_updates(traversal_done, map_applied_events);
      const ApEvent result = analysis->perform_output(
         Runtime::merge_events(remote_ready, output_ready), map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
      return result;
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
                                    std::vector<InstanceView*> &local_views,
                                    std::set<LogicalView*> &registration_views,
                                    const ApEvent termination_event,
                                    VersionInfo &version_info,
                                    const PhysicalTraceInfo &trace_info,
                                    std::set<RtEvent> &map_applied_events,
                                    const bool restricted)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_ATTACH_EXTERNAL_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      const RegionUsage usage(req);
      RegionNode *region_node = get_node(req.region);
      FieldSpaceNode *fs_node = region_node->column_source;
      const FieldMask ext_mask = fs_node->get_field_mask(req.privilege_fields);
      // Perform the registration first since we might need it in case
      // that we have some remote equivalence sets
      std::set<RtEvent> registration_applied;
      const UniqueID op_id = attach_op->get_unique_op_id();
      const RtEvent collect_event = trace_info.get_collect_event();
      std::vector<ApEvent> ready_events;
      for (std::vector<InstanceView*>::const_iterator it =
            local_views.begin(); it != local_views.end(); it++)
      {
        const ApEvent ready = (*it)->register_user(usage, ext_mask,
                    region_node->row_source, op_id, index, termination_event,
                    collect_event, registration_applied, trace_info, 
                    runtime->address_space);
        if (ready.exists())
          ready_events.push_back(ready);
      }
      RtEvent guard_event;
      if (!registration_applied.empty())
      {
        guard_event = Runtime::merge_events(registration_applied);
        if (guard_event.exists())
          map_applied_events.insert(guard_event);
      }
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime, attach_op,
          index, RegionUsage(req), version_info, registration_views, trace_info,
          ApEvent::NO_AP_EVENT,  guard_event, PredEvent::NO_PRED_EVENT, 
          false/*track effects*/, restricted);
      analysis->add_reference();
      std::set<RtEvent> deferral_events;
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis->traverse(it->first, it->second, deferral_events,
                           map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
      if (!ready_events.empty())
        return Runtime::merge_events(&trace_info, ready_events);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::detach_external(const RegionRequirement &req,
                                          DetachOp *detach_op,
                                          unsigned index,
                                          VersionInfo &version_info,
                                          InstanceView *local_view,
                                          const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events,
                                          LogicalView *registration_view)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_DETACH_EXTERNAL_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif 
      RegionNode *region_node = get_node(req.region);
      FieldSpaceNode *fs_node = region_node->column_source;
      const FieldMask ext_mask = fs_node->get_field_mask(req.privilege_fields);
      const UniqueID op_id = detach_op->get_unique_op_id();
      const ApEvent term_event = detach_op->get_completion_event();
      const RtEvent collect_event = trace_info.get_collect_event();
      const RegionUsage usage(req);
      const ApEvent done = local_view->register_user(usage, ext_mask, 
                                                     region_node->row_source,
                                                     op_id, index, term_event,
                                                     collect_event, 
                                                     map_applied_events, 
                                                     trace_info,
                                                     runtime->address_space);
      FilterAnalysis *analysis = new FilterAnalysis(runtime, detach_op, index,
        version_info, local_view,registration_view,true/*remove restriction*/);
      analysis->add_reference();
      std::set<RtEvent> deferral_events;
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis->traverse(it->first, it->second, deferral_events,
                           map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())     
        analysis->perform_remote(traversal_done, map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
      return done;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_fields(Operation *op, unsigned index,
                                             VersionInfo &version_info,
                                            const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events)  
    //--------------------------------------------------------------------------
    {
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();
      const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime, op, index,
          usage, version_info, NULL/*view*/, trace_info, ApEvent::NO_AP_EVENT);
      analysis->add_reference();
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis->traverse(it->first, it->second, deferral_events,
                           map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_events);
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::find_invalid_instances(Operation *op, unsigned index,
                                                  VersionInfo &version_info,
                                  const FieldMaskSet<InstanceView> &valid_views,
                                      FieldMaskSet<InstanceView> &invalid_views,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(invalid_views.empty());
#endif
      const FieldMaskSet<EquivalenceSet> &eq_sets =
        version_info.get_equivalence_sets();
      InvalidInstAnalysis analysis(runtime, op, index,version_info,valid_views);
      std::set<RtEvent> deferral_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            eq_sets.begin(); it != eq_sets.end(); it++)
        analysis.traverse(it->first, it->second, deferral_events,
                          map_applied_events, true/*original set*/);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent ready;
      if (traversal_done.exists() || analysis.has_remote_sets())
        ready = analysis.perform_remote(traversal_done, map_applied_events);
      // Wait for all the responses to be ready
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      analysis.report_instances(invalid_views);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::update_valid_instances(Operation *op, unsigned index,
                                                  VersionInfo &version_info,
                                  const FieldMaskSet<InstanceView> &valid_views,
                                          const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      const FieldMaskSet<EquivalenceSet> &eq_sets = 
        version_info.get_equivalence_sets();
      const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      // Sort the valid views into field mask sets
      LegionList<FieldSet<InstanceView*> > view_sets;
      valid_views.compute_field_sets(FieldMask(), view_sets);
      for (LegionList<FieldSet<InstanceView*> >::const_iterator vit =
            view_sets.begin(); vit != view_sets.end(); vit++)
      {
        // Stupid container problem
        const std::set<LogicalView*> *log_views = NULL; 
        const std::set<InstanceView*> *inst_views = &vit->elements;
        static_assert(sizeof(log_views) == sizeof(inst_views), "Fuck c++");
        memcpy(&log_views, &inst_views, sizeof(log_views));
        OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime, op, index,
           usage, version_info, *log_views, trace_info, ApEvent::NO_AP_EVENT);
        analysis->add_reference();
        std::set<RtEvent> deferral_events;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              eq_sets.begin(); it != eq_sets.end(); it++)
        {
          const FieldMask overlap = it->second & vit->set_mask;
          if (!overlap)
            continue;
          // We don't bother tracking updates here since they could race
          // with each other to update the VersionInfo data structure
          it->first->overwrite_set(*analysis, overlap, deferral_events,
                                   map_applied_events);
        }
        const RtEvent traversal_done = deferral_events.empty() ?
          RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
        if (traversal_done.exists() || analysis->has_remote_sets())
          analysis->perform_remote(traversal_done, map_applied_events);
        if (analysis->remove_reference())
          delete analysis;  
      }
    }

    //--------------------------------------------------------------------------
    int RegionTreeForest::physical_convert_mapping(Operation *op,
                                  const RegionRequirement &req,
                                  const std::vector<MappingInstance> &chosen,
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
        perform_missing_acquires(op, *acquired, unacquired); 
      }
      return -1; // no composite index
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::physical_convert_postmapping(Operation *op,
                                  const RegionRequirement &req,
                                  const std::vector<MappingInstance> &chosen,
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
        perform_missing_acquires(op, *acquired, unacquired);
      }
      return has_composite;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::log_mapping_decision(const UniqueID uid, 
                                                TaskContext *context,
                                                const unsigned index,
                                                const RegionRequirement &req,
                                                const InstanceSet &targets,
                                                bool postmapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(runtime->legion_spy_enabled); 
#endif
      FieldSpaceNode *node = (req.handle_type != LEGION_PARTITION_PROJECTION) ?
        get_node(req.region.get_field_space()) : 
        get_node(req.partition.get_field_space());
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef &inst = targets[idx];
        const FieldMask &valid_mask = inst.get_valid_fields();
        InstanceManager *manager = inst.get_manager();
        std::vector<FieldID> valid_fields;
        node->get_field_set(valid_mask, context, valid_fields);
        for (std::vector<FieldID>::const_iterator it = valid_fields.begin();
              it != valid_fields.end(); it++)
        {
          if (postmapping)
            LegionSpy::log_post_mapping_decision(uid, index, *it,
                                                 manager->get_unique_event());
          else
            LegionSpy::log_mapping_decision(uid, index, *it,
                                            manager->get_unique_event());
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_missing_acquires(Operation *op,
                                std::map<PhysicalManager*,unsigned> &acquired,
                                const std::vector<PhysicalManager*> &unacquired)
    //--------------------------------------------------------------------------
    {
      // Try and do the acquires for any instances that weren't acquired
      for (std::vector<PhysicalManager*>::const_iterator it = 
            unacquired.begin(); it != unacquired.end(); it++)
        if ((*it)->acquire_instance(MAPPING_ACQUIRE_REF, op))
          acquired.insert(std::pair<PhysicalManager*,unsigned>(*it, 1));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::check_context_state(RegionTreeContext ctx)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeID,RegionNode*> trees;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        // Need to hold references to prevent deletion race
        for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
              tree_nodes.begin(); it != tree_nodes.end(); it++)
          it->second->add_base_resource_ref(REGION_TREE_REF);
        trees = tree_nodes;
      }
      CurrentInitializer init(ctx.get_id());
      for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
            trees.begin(); it != trees.end(); it++)
      {
        it->second->visit_node(&init);
        if (it->second->remove_base_resource_ref(REGION_TREE_REF))
          delete it->second;
      }
    }

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
                                                  std::set<RtEvent> *applied,
                                                  bool add_root_reference,
                                                  unsigned depth)
    //--------------------------------------------------------------------------
    { 
      RtUserEvent local_initialized;
      std::set<RtEvent> local_applied;
      if (applied == NULL)
      {
        applied = &local_applied;
        local_initialized = Runtime::create_rt_user_event();
        if (initialized.exists())
          local_applied.insert(initialized);
        initialized = local_initialized;
      }
      IndexSpaceCreator creator(this, sp, bounds, is_domain, parent, color,
                    did, is_ready, expr_id, initialized, depth, provenance);
      NT_TemplateHelper::demux<IndexSpaceCreator>(sp.get_type_tag(), &creator);
      IndexSpaceNode *result = creator.result;  
#ifdef DEBUG_LEGION
      assert(result != NULL);
      assert(applied != NULL);
#endif
      WrapperReferenceMutator mutator(*applied);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator it =
          index_nodes.find(sp);
        if (it != index_nodes.end())
        {
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
        // If we are remote we always have a GC ref from the owner
        // If we are the root then the valid ref comes from the application
        // Otherwise the valid ref comes from parent partition
        if (!result->is_owner())
          // Always add a base gc ref for all index spaces
          result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
        result->register_with_runtime(&mutator);
        if (parent != NULL)
        {
#ifdef DEBUG_LEGION
          assert(!add_root_reference);
#endif
          // Always add a valid reference from the parent
          result->add_nested_valid_ref(parent->did, &mutator);
          // Check to see if the parent is still tree valid
          if (!parent->add_child(result))
          {
            result->invalidate_tree();
            // If the parent is tree invalid then remove the reference
            result->remove_nested_valid_ref(parent->did, &mutator);
          }
        }
        else 
        {
          if (result->is_owner())
          {
#ifdef DEBUG_LEGION
            assert(!add_root_reference);
#endif
            result->add_base_valid_ref(APPLICATION_REF, &mutator);
          }
          else if (add_root_reference)
            result->add_base_valid_ref(REMOTE_DID_REF, &mutator);
        }
      }
      if (local_initialized.exists())
      {
        if (!local_applied.empty())
          Runtime::trigger_event(local_initialized,
              Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(local_initialized);
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
                                                  IndexPartNode *parent,
                                                  LegionColor color,
                                                  DistributedID did,
                                                  RtEvent initialized,
                                                  Provenance *provenance,
                                                  ApUserEvent is_ready,
                                                  std::set<RtEvent> *applied,
                                                  unsigned depth)
    //--------------------------------------------------------------------------
    { 
      RtUserEvent local_initialized;
      std::set<RtEvent> local_applied;
      if (applied == NULL)
      {
        applied = &local_applied;
        local_initialized = Runtime::create_rt_user_event();
        if (initialized.exists())
          local_applied.insert(initialized);
        initialized = local_initialized;
      }
      IndexSpaceCreator creator(this, sp, realm_is, false/*is domain*/, parent,
            color, did, is_ready, 0/*expr id*/, initialized, depth, provenance);
      NT_TemplateHelper::demux<IndexSpaceCreator>(sp.get_type_tag(), &creator);
      IndexSpaceNode *result = creator.result;  
#ifdef DEBUG_LEGION
      assert(result != NULL);
      assert(applied != NULL);
#endif
      WrapperReferenceMutator mutator(*applied);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator it =
          index_nodes.find(sp);
        if (it != index_nodes.end())
        {
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          // Free up the event since we didn't use it
          Runtime::trigger_event(NULL, is_ready);
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
        // If we are remote we always have a GC ref from the owner
        // If we are the root then the valid ref comes from the application
        // Otherwise the valid ref comes from parent partition
        if (!result->is_owner())
          result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
        result->register_with_runtime(&mutator);
        if (parent != NULL)
        {
          // Always add a valid reference from the parent
          result->add_nested_valid_ref(parent->did, &mutator);
          // Check to see if the parent is still tree valid
          if (!parent->add_child(result))
          {
            result->invalidate_tree();
            // If the parent is tree invalid then remove the reference
            result->remove_nested_valid_ref(parent->did, &mutator);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(result->is_owner());
#endif
          result->add_base_valid_ref(APPLICATION_REF, &mutator);
        }
      } 
      if (local_initialized.exists())
      {
        if (!local_applied.empty())
          Runtime::trigger_event(local_initialized,
              Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(local_initialized);
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
                                                 ApUserEvent pending,
                                                 RtEvent initialized,
                                                 std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
      RtUserEvent local_initialized;
      std::set<RtEvent> local_applied;
      if (applied == NULL)
      {
        applied = &local_applied;
        local_initialized = Runtime::create_rt_user_event();
        if (initialized.exists())
          local_applied.insert(initialized);
        initialized = local_initialized;
      }
      IndexPartCreator creator(this, p, parent, color_space, color, disjoint, 
                 complete, did, part_ready, pending, initialized, provenance);
      NT_TemplateHelper::demux<IndexPartCreator>(p.get_type_tag(), &creator);
      IndexPartNode *result = creator.result;
#ifdef DEBUG_LEGION
      assert(parent != NULL);
      assert(result != NULL);
      assert(applied != NULL);
#endif
      WrapperReferenceMutator mutator(*applied);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition,IndexPartNode*>::const_iterator it =
          index_parts.find(p);
        if (it != index_parts.end())
        {
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, otherwise we're remote so we add a gc 
        // reference that will be removed by the owner when we can be
        // safely collected
        if (result->is_owner())
          result->add_base_valid_ref(APPLICATION_REF, &mutator);
        else
          result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
        result->register_with_runtime(&mutator);
        parent->add_child(result);
      }
      if (local_initialized.exists())
      {
        if (!local_applied.empty())
          Runtime::trigger_event(local_initialized,
              Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(local_initialized);
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
                                                 ApUserEvent pending,
                                                 RtEvent initialized,
                                                 std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
      RtUserEvent local_initialized;
      std::set<RtEvent> local_applied;
      if (applied == NULL)
      {
        applied = &local_applied;
        local_initialized = Runtime::create_rt_user_event();
        if (initialized.exists())
          local_applied.insert(initialized);
        initialized = local_initialized;
      }
      IndexPartCreator creator(this, p, parent, color_space, color, 
          disjointness_ready, complete, did, part_ready, pending, 
          initialized, provenance);
      NT_TemplateHelper::demux<IndexPartCreator>(p.get_type_tag(), &creator);
      IndexPartNode *result = creator.result;
#ifdef DEBUG_LEGION
      assert(parent != NULL);
      assert(result != NULL);
      assert(applied != NULL);
#endif
      WrapperReferenceMutator mutator(*applied);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition,IndexPartNode*>::const_iterator it =
          index_parts.find(p);
        if (it != index_parts.end())
        {
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, otherwise we're remote so we add a gc 
        // reference that will be removed by the owner when we can be
        // safely collected
        if (result->is_owner())
          result->add_base_valid_ref(APPLICATION_REF, &mutator);
        else
          result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
        result->register_with_runtime(&mutator);
        parent->add_child(result);
      }
      if (local_initialized.exists())
      {
        if (!local_applied.empty())
          Runtime::trigger_event(local_initialized,
              Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(local_initialized);
      }
      return result;
    }
 
    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace space,
                                         DistributedID did, RtEvent initialized,
                                         Provenance *provenance,
                                         std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
      RtUserEvent local_initialized;
      std::set<RtEvent> local_applied;
      if (applied == NULL)
      {
        applied = &local_applied;
        local_initialized = Runtime::create_rt_user_event();
        if (initialized.exists())
          local_applied.insert(initialized);
        initialized = local_initialized;
      }
      FieldSpaceNode *result = 
        new FieldSpaceNode(space, this, did, initialized, provenance);
#ifdef DEBUG_LEGION
      assert(result != NULL);
      assert(applied != NULL);
#endif
      WrapperReferenceMutator mutator(*applied);
      // Hold the lookup lock while modifying the lookup table
      {
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace,FieldSpaceNode*>::const_iterator it =
          field_nodes.find(space);
        if (it != field_nodes.end())
        {
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
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
          result->add_base_valid_ref(APPLICATION_REF, &mutator);
        result->register_with_runtime(&mutator);
      }
      if (local_initialized.exists())
      {
        if (!local_applied.empty())
          Runtime::trigger_event(local_initialized,
              Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(local_initialized);
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
      std::set<RtEvent> local_applied;
      RtUserEvent local_initialized = Runtime::create_rt_user_event();
      if (initialized.exists())
        local_applied.insert(initialized);
      initialized = local_initialized;
      FieldSpaceNode *result = 
        new FieldSpaceNode(space, this, did, initialized, provenance, derez);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      WrapperReferenceMutator mutator(local_applied);
      // Hold the lookup lock while modifying the lookup table
      {
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace,FieldSpaceNode*>::const_iterator it =
          field_nodes.find(space);
        if (it != field_nodes.end())
        {
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
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
          result->add_base_valid_ref(APPLICATION_REF, &mutator);
        result->register_with_runtime(&mutator);
      }
      if (!local_applied.empty())
        Runtime::trigger_event(local_initialized,
            Runtime::merge_events(local_applied));
      else
        Runtime::trigger_event(local_initialized);
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::create_node(LogicalRegion r, 
                                  PartitionNode *parent, RtEvent initialized,
                                  DistributedID did, Provenance *provenance,
                                  std::set<RtEvent> *applied)
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
      
      RtUserEvent local_initialized;
      std::set<RtEvent> local_applied;
      if (applied == NULL)
      {
        applied = &local_applied;
        local_initialized = Runtime::create_rt_user_event();
        if (initialized.exists())
          local_applied.insert(initialized);
        if (row_ready.exists())
          local_applied.insert(row_ready);
        if (col_ready.exists())
          local_applied.insert(col_ready);
        initialized = local_initialized;
      }
      else if (row_ready.exists() || col_ready.exists())
        initialized = Runtime::merge_events(initialized, row_ready, col_ready); 
      if ((provenance == NULL) && (parent != NULL))
        provenance = parent->provenance;
      RegionNode *result = new RegionNode(r, parent, row_src, col_src, this,
          did, initialized, (parent == NULL) ? initialized :
          parent->tree_initialized, provenance);
#ifdef DEBUG_LEGION
      assert(result != NULL);
      assert(applied != NULL);
#endif
      WrapperReferenceMutator mutator(*applied);
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
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // It already exists, delete our copy and return
          // the one that has already been made
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
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
            result->add_base_valid_ref(APPLICATION_REF, &mutator);
          else
            result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
          // Root nodes get registered with the runtime since we
          // know that they all have the same distributed ID
          // No mutator so no notifications are sent
          result->register_with_runtime(NULL/*no mutator*/);
        }
        else // not a root so we get a gc ref from our parent
          result->add_nested_gc_ref(parent->did, &mutator);
        result->record_registered();
      }
      if (local_initialized.exists())
      {
        if (!local_applied.empty())
          Runtime::trigger_event(local_initialized,
              Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(local_initialized);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::create_node(LogicalPartition p,
                                                 RegionNode *parent,
                                                 std::set<RtEvent> *applied)
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
      RtUserEvent local_initialized;
      std::set<RtEvent> local_applied;
      if (applied == NULL)
      {
        applied = &local_applied;
        local_initialized = Runtime::create_rt_user_event();
        if (initialized.exists())
          local_applied.insert(initialized);
        if (row_ready.exists())
          local_applied.insert(row_ready);
        if (col_ready.exists())
          local_applied.insert(col_ready);
        initialized = local_initialized;
      }
      else if (row_ready.exists() || col_ready.exists())
        initialized = Runtime::merge_events(initialized, row_ready, col_ready);
      PartitionNode *result = new PartitionNode(p, parent, row_src, col_src, 
                                  this, initialized, parent->tree_initialized);
#ifdef DEBUG_LEGION
      assert(result != NULL);
      assert(applied != NULL);
#endif
      WrapperReferenceMutator mutator(*applied);
      // Special case here in case multiple clients attempt
      // to make the node at the same time
      {
        // Hole the lookup lock when modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<LogicalPartition,PartitionNode*>::const_iterator it =
          part_nodes.find(p);
        if (it != part_nodes.end())
        {
          // Free up our user event since we don't need it
          if (local_initialized.exists())
            Runtime::trigger_event(local_initialized);
          // It already exists, delete our copy and
          // return the one that has already been made
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        // Now we can put the node in the map
        part_nodes[p] = result;
        // Add gc ref that will be removed when either the root region node
        // or the index partition node has been destroyed
        result->add_base_gc_ref(REGION_TREE_REF, &mutator);
        result->record_registered();
      }
      if (local_initialized.exists())
      {
        if (!local_applied.empty())
          Runtime::trigger_event(local_initialized,
              Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(local_initialized);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::get_node(IndexSpace space,
                                  RtEvent *defer /*=NULL*/, const bool can_fail)
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
        if (can_fail)
          return NULL;
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
        return get_node(space, NULL, can_fail);
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
                                              const bool can_fail/* = false*/,
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
        if (can_fail)
          return NULL;
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
        return get_node(part, NULL, can_fail);
      }
      else
      {
        *defer = wait_on;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::get_node(FieldSpace space,
                                               RtEvent *defer /*=NULL*/) 
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
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for field space %x.", space.id)
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
        return get_node(space, NULL);
      }
      else
      {
        *defer = wait_on;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::get_node(LogicalRegion handle,
                                           bool need_check /* = true*/)
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
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for logical region tree %d.",
                           handle.get_tree_id());
          assert(false);
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
    RegionNode* RegionTreeForest::get_tree(RegionTreeID tid)
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
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for region tree ID %d", tid)
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
    RegionNode* RegionTreeForest::find_local_node(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<LogicalRegion,RegionNode*>::const_iterator finder = 
        region_nodes.find(handle);
      if (finder == region_nodes.end())
        return NULL;
      finder->second->add_base_resource_ref(REGION_TREE_REF);
      return finder->second;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::find_local_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<LogicalPartition,PartitionNode*>::const_iterator finder =
          part_nodes.find(handle);
      if (finder == part_nodes.end())
        return NULL;
      finder->second->add_base_resource_ref(REGION_TREE_REF);
      return finder->second;
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
                           IndexSpaceExpression *lhs, IndexSpaceExpression *rhs,
                           ReferenceMutator *mutator)
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
      IndexSpaceExpression *result = union_index_spaces(exprs);
      // Add the live reference 
      if (mutator == NULL)
      {
        LocalReferenceMutator local_mutator;
        result->add_base_expression_reference(LIVE_EXPR_REF, &local_mutator);
      }
      else
        result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
      // Save it in the implicit live expression references
      if (implicit_reference_tracker == NULL)
        implicit_reference_tracker = new ImplicitReferenceTracker;
      implicit_reference_tracker->record_live_expression(result);
      // Remove the gc reference that comes back from finding it in the tree
      if (result->remove_live_reference(REGION_TREE_REF))
        assert(false); // should never hit this
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::union_index_spaces(
                                   const std::set<IndexSpaceExpression*> &exprs,
                                   ReferenceMutator *mutator)
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
      LocalReferenceMutator local_mutator;
      if (expressions.size() == 1)
      {
        IndexSpaceExpression *result = expressions.back();
        if (exprs.find(result) == exprs.end())
        {
          if (mutator == NULL)
            result->add_base_expression_reference(LIVE_EXPR_REF,&local_mutator);
          else
            result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
          if (implicit_reference_tracker == NULL)
            implicit_reference_tracker = new ImplicitReferenceTracker;
          implicit_reference_tracker->record_live_expression(result);
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
            if (mutator == NULL)
              result->add_base_expression_reference(LIVE_EXPR_REF,
                                                    &local_mutator);
            else
              result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
            if (implicit_reference_tracker == NULL)
              implicit_reference_tracker = new ImplicitReferenceTracker;
            implicit_reference_tracker->record_live_expression(result);
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
            if (mutator == NULL)
              expr->add_base_expression_reference(REGION_TREE_REF,
                                                  &local_mutator);
            else
              expr->add_base_expression_reference(REGION_TREE_REF, mutator);
            // Remove the gc ref that comes back from the union call
            if (expr->remove_live_reference(REGION_TREE_REF))
              assert(false); // should never hit this
            next_expressions.push_back(expr);
          }
          else
          {
            IndexSpaceExpression *expr = expressions.back();
            expressions.pop_back();
            if (mutator == NULL)
              expr->add_base_expression_reference(REGION_TREE_REF, 
                                                  &local_mutator);
            else
              expr->add_base_expression_reference(REGION_TREE_REF, mutator);
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
          {
            if (mutator == NULL)
              unique->add_base_expression_reference(REGION_TREE_REF, 
                                                    &local_mutator);
            else
              unique->add_base_expression_reference(REGION_TREE_REF, mutator);
          }
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
            if (mutator == NULL)
              result->add_base_expression_reference(LIVE_EXPR_REF,
                                                    &local_mutator);
            else
              result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
            if (implicit_reference_tracker == NULL)
              implicit_reference_tracker = new ImplicitReferenceTracker;
            implicit_reference_tracker->record_live_expression(result);
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
        if (mutator == NULL)
          result->add_base_expression_reference(LIVE_EXPR_REF,&local_mutator);
        else
          result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(result);
      }
      // Remove the reference added by the trie traversal
      if (result->remove_live_reference(REGION_TREE_REF))
        assert(false); // should never hit this deletion
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
              result->try_add_live_reference(REGION_TREE_REF))
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
                           IndexSpaceExpression *lhs, IndexSpaceExpression *rhs,
                           ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs->type_tag == rhs->type_tag);
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
      IndexSpaceExpression *result = intersect_index_spaces(exprs);
      // Add the live reference 
      if (mutator == NULL)
      {
        LocalReferenceMutator local_mutator;
        result->add_base_expression_reference(LIVE_EXPR_REF, &local_mutator);
      }
      else
        result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
      // Save it in the implicit live expression references
      if (implicit_reference_tracker == NULL)
        implicit_reference_tracker = new ImplicitReferenceTracker;
      implicit_reference_tracker->record_live_expression(result);
      // Remove the gc reference that comes back with the trie traversal
      if (result->remove_live_reference(REGION_TREE_REF))
        assert(false); // should never hit this
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* RegionTreeForest::intersect_index_spaces(
                                   const std::set<IndexSpaceExpression*> &exprs,
                                   ReferenceMutator *mutator)
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
      LocalReferenceMutator local_mutator;
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
            if (mutator == NULL)
              result->add_base_expression_reference(LIVE_EXPR_REF,
                                                    &local_mutator);
            else
              result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
            if (implicit_reference_tracker == NULL)
              implicit_reference_tracker = new ImplicitReferenceTracker;
            implicit_reference_tracker->record_live_expression(result);
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
            if (mutator == NULL)
              expr->add_base_expression_reference(REGION_TREE_REF,
                                                  &local_mutator);
            else
              expr->add_base_expression_reference(REGION_TREE_REF, mutator);
            // Remove the gc ref that comes back from the union call
            if (expr->remove_live_reference(REGION_TREE_REF))
              assert(false); // should never hit this
            next_expressions.push_back(expr);
          }
          else
          {
            IndexSpaceExpression *expr = expressions.back();
            expressions.pop_back();
            if (mutator == NULL)
              expr->add_base_expression_reference(REGION_TREE_REF,
                                                  &local_mutator);
            else
              expr->add_base_expression_reference(REGION_TREE_REF, mutator);
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
              if (mutator == NULL)
                unique->add_base_expression_reference(LIVE_EXPR_REF,
                                                      &local_mutator);
              else
                unique->add_base_expression_reference(LIVE_EXPR_REF, mutator);
              if (implicit_reference_tracker == NULL)
                implicit_reference_tracker = new ImplicitReferenceTracker;
              implicit_reference_tracker->record_live_expression(unique);
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
          {
            if (mutator == NULL)
              unique->add_base_expression_reference(REGION_TREE_REF, 
                                                    &local_mutator);
            else
              unique->add_base_expression_reference(REGION_TREE_REF, mutator);
          }
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
            if (mutator == NULL)
              result->add_base_expression_reference(LIVE_EXPR_REF,
                                                    &local_mutator);
            else
              result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
            if (implicit_reference_tracker == NULL)
              implicit_reference_tracker = new ImplicitReferenceTracker;
            implicit_reference_tracker->record_live_expression(result);
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
        if (mutator == NULL)
          result->add_base_expression_reference(LIVE_EXPR_REF,&local_mutator);
        else
          result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(result);
      }
      // Remove the reference added by the trie traversal
      if (result->remove_live_reference(REGION_TREE_REF))
        assert(false); // should never hit this deletion
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
              result->try_add_live_reference(REGION_TREE_REF))
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
                           OperationCreator *creator/*=NULL*/,
                           ReferenceMutator *mutator/*=NULL*/)
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
              expr->try_add_live_reference(REGION_TREE_REF))
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
      if (mutator == NULL)
      {
        LocalReferenceMutator local_mutator;
        result->add_base_expression_reference(LIVE_EXPR_REF, &local_mutator);
      }
      else
        result->add_base_expression_reference(LIVE_EXPR_REF, mutator);
      if (implicit_reference_tracker == NULL)
        implicit_reference_tracker = new ImplicitReferenceTracker;
      implicit_reference_tracker->record_live_expression(result);
      // Remove the gc reference that comes back from finding it in the tree
      if (result->remove_live_reference(REGION_TREE_REF))
        assert(false); // should never hit this
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
#ifdef DEBUG_LEGION
      assert(volume > 0);
#endif
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
        LocalReferenceMutator mutator;
        node->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
        if (!pending.done_ref_counting)
        {
          const RtEvent added = mutator.get_done_event();
          // Special case here: if the source was the owner and we didn't
          // just send a message to add our reference then we can buffer up
          // our reference to be removed when we are no longer valid
          // Be very careful here! You can only do this if the expression
          // was sent from the source or you risk reference count cycles!
          if ((pending.source == node->owner_space) &&
              (!added.exists() || added.has_triggered()))
            node->record_remote_owner_valid_reference();
          else
            node->send_remote_valid_decrement(pending.source,NULL/*mut*/,added);
        }
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(node);
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
        LocalReferenceMutator mutator;
        result->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
        if (!pending.done_ref_counting)
        {
          const RtEvent added = mutator.get_done_event();
          // Special case here: if the source was the owner and we didn't
          // just send a message to add our reference then we can buffer up
          // our reference to be removed when we are no longer valid
          // Be very careful here! You can only do this if the expression
          // was sent from the source or you risk reference count cycles!
          if ((pending.source == op->owner_space) &&
              (!added.exists() || added.has_triggered()))
            result->record_remote_owner_valid_reference();
          else
            op->send_remote_valid_decrement(pending.source, NULL/*mut*/, added);
        }
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(result);
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
        IndexSpaceNode *result = get_node(handle);
        // Remove the reference that we have from when this was packed
        result->send_remote_gc_decrement(source);
        return result;
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
        bool recurrent, unsigned s)
      : LgTaskArgs<DeferCopyAcrossArgs>(o->get_unique_op_id()),
        executor(e), op(o), trace_info(new PhysicalTraceInfo(info)), guard(g),
        copy_precondition(copy_pre), src_indirect_precondition(src_pre),
        dst_indirect_precondition(dst_pre), 
        done_event(Runtime::create_ap_user_event(trace_info)),
        stage(s+1), recurrent_replay(recurrent)
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
            dargs->recurrent_replay, dargs->stage));
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
              const InstanceSet &insts, const std::vector<InstanceView*> &views,
              const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_fields.empty());
#endif
      FieldSpaceNode *fs = forest->get_node(req.region.get_field_space());
      std::vector<unsigned> indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, indexes);
      src_fields.reserve(indexes.size());
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
          views[idx]->copy_from(copy_mask, src_fields);
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
              const InstanceSet &insts, const std::vector<InstanceView*> &views,
              const PhysicalTraceInfo &trace_info, const bool exclusive_redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dst_fields.empty());
#endif
      FieldSpaceNode *fs = forest->get_node(req.region.get_field_space());
      std::vector<unsigned> indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, indexes);
      dst_fields.reserve(indexes.size());
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
          views[idx]->copy_to(copy_mask, dst_fields);
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
            const InstanceRef &indirect_instance, const DomainPoint &point,
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
        indirect_instance.get_physical_manager()->get_instance(point);
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
            const InstanceRef &indirect_instance, const DomainPoint &point,
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
        indirect_instance.get_physical_manager()->get_instance(point);
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

    /////////////////////////////////////////////////////////////
    // Index Space Expression 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(LocalLock &lock)
      : type_tag(0), expr_id(0), expr_lock(lock), canonical(NULL),
        remote_owner_valid_references(0),
        volume(0), has_volume(false), empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(TypeTag tag, Runtime *rt,
                                               LocalLock &lock)
      : type_tag(tag), expr_id(rt->get_unique_index_space_expr_id()), 
        expr_lock(lock), canonical(NULL), remote_owner_valid_references(0),
        volume(0), has_volume(false), empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(TypeTag tag, IndexSpaceExprID id,
                                               LocalLock &lock)
      : type_tag(tag), expr_id(id), expr_lock(lock), canonical(NULL),
        remote_owner_valid_references(0),
        volume(0), has_volume(false), empty(false), has_empty(false)
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
            (*it)->add_tree_expression_reference(did);
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
            assert(false); // should never delete since we have an expr ref
          // Remove any references that we have on the parents
          if ((*it)->remove_tree_expression_reference(did))
            delete (*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* IndexSpaceExpression::get_canonical_expression(
                                                       RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      IndexSpaceExpression *expr = canonical.load();
      if (expr != NULL)
        return expr;
      expr = forest->find_canonical_expression(this);
      if (expr == this)
      {
        // If we're our own canonical expression then the forest didn't
        // give us a reference to ourself, so just write it, everyone will
        // write the same value so there's no risk here
        canonical.store(expr);
        return expr;
      }
      // If the canonical expression is not ourself, then the region tree
      // forest has given us a reference back on it, see if we're the first
      // ones to write it, if not we can remove the reference now
      const DistributedID did = get_distributed_id();
      IndexSpaceExpression *expected = NULL;
      if (!canonical.compare_exchange_strong(expected, expr))
        expr->remove_canonical_reference(did);
      else // We're the first so add our resource reference
        expr->add_tree_expression_reference(did);
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
          // Make this valid and then send the removal of the 
          // remote did expression
          LocalReferenceMutator mutator;
          op->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
          // Always need to send this reference removal back immediately
          // in order to avoid reference counting deadlock
          op->send_remote_valid_decrement(source, NULL/*mutator*/,
              mutator.get_done_event());
        }
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(result);
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
        LocalReferenceMutator mutator;
        node->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
        const RtEvent added = mutator.get_done_event();
        // Special case here: if the source was the owner and we didn't
        // just send a message to add our reference then we can buffer up
        // our reference to be removed when we are no longer valid
        // Be very careful here! You can only do this if the expression
        // was sent from the source or you risk reference count cycles!
        if ((source == node->owner_space) &&
            (!added.exists() || added.has_triggered()))
          node->record_remote_owner_valid_reference();
        else
          node->send_remote_valid_decrement(source, NULL/*mutator*/, added);
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(node);
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
        LocalReferenceMutator mutator;
        result->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
        const RtEvent added = mutator.get_done_event();
        // Special case here: if the source was the owner and we didn't
        // just send a message to add our reference then we can buffer up
        // our reference to be removed when we are no longer valid
        // Be very careful here! You can only do this if the expression
        // was sent from the source or you risk reference count cycles!
        if ((source == op->owner_space) &&
            (!added.exists() || added.has_triggered()))
          result->record_remote_owner_valid_reference();
        else
          op->send_remote_valid_decrement(source, NULL/*mutator*/, added);
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(result);
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
        // Make this valid and then send the removal of the 
        // remote did expression
        LocalReferenceMutator mutator;
        op->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
        if (source != forest->runtime->address_space)
        {
          // Always need to send this reference removal back immediately
          // in order to avoid reference counting deadlock
          op->send_remote_valid_decrement(source, NULL/*mutator*/,
              mutator.get_done_event());
        }
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(result);
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
        LocalReferenceMutator mutator;
        node->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
        const RtEvent added = mutator.get_done_event();
        // Special case here: if the source was the owner and we didn't
        // just send a message to add our reference then we can buffer up
        // our reference to be removed when we are no longer valid
        // Be very careful here! You can only do this if the expression
        // was sent from the source or you risk reference count cycles!
        if ((source == node->owner_space) &&
            (!added.exists() || added.has_triggered()))
          node->record_remote_owner_valid_reference();
        else
          node->send_remote_valid_decrement(source, NULL/*mutator*/, added);
        pending.done_ref_counting = true;
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->record_live_expression(node);
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
      LocalReferenceMutator mutator;
      result->add_base_expression_reference(LIVE_EXPR_REF, &mutator);
      const RtEvent added = mutator.get_done_event();
      // Special case here: if the source was the owner and we didn't
      // just send a message to add our reference then we can buffer up
      // our reference to be removed when we are no longer valid
      // Be very careful here! You can only do this if the expression
      // was sent from the source or you risk reference count cycles!
      if ((source == op->owner_space) &&
          (!added.exists() || added.has_triggered()))
        result->record_remote_owner_valid_reference();
      else
        op->send_remote_valid_decrement(source, NULL/*mutator*/, added);
      pending.done_ref_counting = true;
      if (implicit_reference_tracker == NULL)
        implicit_reference_tracker = new ImplicitReferenceTracker;
      implicit_reference_tracker->record_live_expression(result);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceOperation::IndexSpaceOperation(TypeTag tag, OperationKind kind,
                                             RegionTreeForest *ctx)
      : IndexSpaceExpression(tag, ctx->runtime, inter_lock), 
        DistributedCollectable(ctx->runtime, LEGION_DISTRIBUTED_HELP_ENCODE( 
          ctx->runtime->get_available_distributed_id(), INDEX_EXPR_NODE_DC),
          ctx->runtime->address_space),
        context(ctx), origin_expr(this), op_kind(kind), invalidated(0)
#ifdef DEBUG_LEGION
        , tree_active(true)
#endif
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Index Expr %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, expr_id);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceOperation::IndexSpaceOperation(TypeTag tag, RegionTreeForest *ctx,
        IndexSpaceExprID eid, DistributedID did, AddressSpaceID owner,
        IndexSpaceOperation *origin)
      : IndexSpaceExpression(tag, eid, inter_lock),
        DistributedCollectable(ctx->runtime, did, owner), 
        context(ctx), origin_expr(origin),
        op_kind(REMOTE_EXPRESSION_KIND), invalidated(0)
#ifdef DEBUG_LEGION
        , tree_active(true)
#endif
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      // Keep a gc reference to ensure that this remain active until it 
      // is made invalid on the owner node
      add_base_gc_ref(REMOTE_DID_REF);
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
    void IndexSpaceOperation::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tree_active); // should only happen once
#endif
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner send a valid reference to the owner
      if (!is_owner())
        send_remote_valid_increment(owner_space, mutator);
      IndexSpaceExpression *canon = canonical.load();
      if ((canon != NULL) && (canon != this) &&
          !canon->try_add_canonical_reference(did))
      {
        // We were unsuccessful at adding our canonical reference so 
        // remove the resource reference to the canonical object and
        // and mark that we no longer have a canonical expression
        if (canon->remove_tree_expression_reference(did))
          delete canon;
        canonical.store(NULL);
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_valid_decrement(owner_space, mutator,
           RtEvent::NO_RT_EVENT, remote_owner_valid_references.exchange(0) + 1);
      // If we have a canonical reference that is not ourselves then 
      // we need to remove the nested reference that we are holding on it too
      IndexSpaceExpression *canon = canonical.load();
      if ((canon != NULL) && (canon != this) &&
          canon->remove_canonical_reference(did))
        // should never actually delete it since we have a resource ref too
        assert(false); 
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::InactiveFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      op->send_remote_gc_decrement(target, mutator);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should only go through one cycle of active to not active
      assert(tree_active);
      tree_active = false;
#endif
      if (is_owner())
      {
        // Send the removal of the gc references on any remote nodes 
        if (has_remote_instances())
        {
          InactiveFunctor functor(this, mutator); 
          map_over_remote_instances(functor);
        }
      }
      else
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
        else if (canon->remove_tree_expression_reference(did))
          delete canon;
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::try_add_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return check_active_and_increment(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return remove_nested_gc_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::try_add_live_reference(ReferenceSource source)
    //--------------------------------------------------------------------------
    {
      return check_active_and_increment(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_live_reference(ReferenceSource source)
    //--------------------------------------------------------------------------
    {
      return remove_base_gc_ref(source);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_base_expression_reference(
              ReferenceSource source, ReferenceMutator *mutator, unsigned count)
    //--------------------------------------------------------------------------
    {
      if (mutator == NULL)
      {
        LocalReferenceMutator local_mutator;
        add_base_valid_ref(source, &local_mutator, count);
      }
      else
        add_base_valid_ref(source, mutator, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_nested_expression_reference(
        DistributedID source, std::set<RtEvent> &applied_events, unsigned count)
    //--------------------------------------------------------------------------
    {
      WrapperReferenceMutator mutator(applied_events);
      add_nested_expression_reference(source, &mutator, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_nested_expression_reference(
                DistributedID source, ReferenceMutator *mutator, unsigned count)
    //--------------------------------------------------------------------------
    {
      if (mutator == NULL)
      {
        LocalReferenceMutator local_mutator;
        add_nested_valid_ref(source, &local_mutator, count);
      }
      else
        add_nested_valid_ref(source, mutator, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_base_expression_reference(
                                         ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_base_valid_ref(source, NULL/*mutator*/, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_nested_expression_reference(
                                           DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_valid_ref(source, NULL/*mutator*/, count);
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
            local_operation->try_add_live_reference(REGION_TREE_REF))
          return local_operation;
        // Operation doesn't exist yet, retake the lock and try to make it
        AutoLock t_lock(trie_lock);
        if ((local_operation != NULL) &&
            local_operation->try_add_live_reference(REGION_TREE_REF))
          return local_operation;
        local_operation = creator.consume();
        if (!local_operation->try_add_live_reference(REGION_TREE_REF))
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
              op_finder->second->try_add_live_reference(REGION_TREE_REF))
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
              op_finder->second->try_add_live_reference(REGION_TREE_REF))
            return op_finder->second;
          // Still don't have the op
          std::map<IndexSpaceExprID,ExpressionTrieNode*>::const_iterator
            node_finder = nodes.find(target_expr);
          if (node_finder == nodes.end())
          {
            // Didn't find the sub-node, so make the operation here
            IndexSpaceExpression *result = creator.consume();
            operations[target_expr] = result;
            if (!result->try_add_live_reference(REGION_TREE_REF))
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
        LegionColor c, DistributedID did, AddressSpaceID owner, 
        RtEvent init, Provenance *prov)
      : DistributedCollectable(ctx->runtime, did, owner, false/*register*/),
        context(ctx), depth(d), color(c), provenance(prov), initialized(init)
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
      // make sure all our gc updates are on the wire before sending unregisters
      // we do this in a hacky way by setting the reentrant_event, see the
      // comment on the reentrant_event member of DistributedCollectable to
      // see why we do it this way
      if (!send_effects.empty())
      {
        std::vector<RtEvent> effects;
        for (std::map<AddressSpaceID,RtEvent>::const_iterator it =
              send_effects.begin(); it != send_effects.end(); it++)
          if (!it->second.has_triggered())
            effects.push_back(it->second);
        reentrant_event = Runtime::merge_events(effects);
      }
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
                                   unsigned dep, Provenance *prov)
      : IndexTreeNode(ctx,
          (dep == UINT_MAX) ? ((par == NULL) ? 0 : par->depth + 1) : dep, c, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_SPACE_NODE_DC),
          get_owner_space(h, ctx->runtime), init, prov),
        IndexSpaceExpression(h.type_tag, exp_id > 0 ? exp_id : 
            runtime->get_unique_index_space_expr_id(), node_lock),
        handle(h), parent(par), index_space_ready(ready), 
        send_references((parent != NULL) ? 1 : 0),
        realm_index_space_set(Runtime::create_rt_user_event()), 
        tight_index_space_set(Runtime::create_rt_user_event()),
        tight_index_space(false), tree_valid(true),
#ifdef DEBUG_LEGION
        tree_active(true),
#endif
        root_valid(parent == NULL)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (parent != NULL)
        assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Space %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.id);
#endif
      // We keep a resource reference on the parent until the parent
      // removes it saying that we no longer need to traverse it as
      // part of sending this index space to remote nodes
      if (send_references > 0)
        parent->add_nested_resource_ref(did);
      if (is_owner() && ctx->runtime->legion_spy_enabled)
        LegionSpy::log_index_space_expr(handle.get_id(), this->expr_id);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(const IndexSpaceNode &rhs)
      : IndexTreeNode(NULL, 0, 0, 0, 0, RtEvent::NO_RT_EVENT, NULL),
        IndexSpaceExpression(node_lock), handle(IndexSpace::NO_SPACE), 
        parent(NULL), index_space_ready(ApEvent::NO_AP_EVENT)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::~IndexSpaceNode(void)
    //--------------------------------------------------------------------------
    {
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
    IndexSpaceNode& IndexSpaceNode::operator=(const IndexSpaceNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tree_active);
#endif
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we add a valid reference to the owner
      if (!is_owner())
        send_remote_valid_increment(owner_space, mutator);
      IndexSpaceExpression *canon = canonical.load();
      if ((canon != NULL) && (canon != this) &&
          !canon->try_add_canonical_reference(did))
      {
        // We were unsuccessful at adding our canonical reference so 
        // remove the resource reference to the canonical object and
        // and mark that we no longer have a canonical expression
        if (canon->remove_tree_expression_reference(did))
          delete canon;
        canonical.store(NULL);
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        if (parent == NULL)
        {
          // If we're a root index space node and this is the first time
          // we have become invalid then the tree is no longer valid
          AutoLock n_lock(node_lock);
          tree_valid = false;
        }
      }
      else
        send_remote_valid_decrement(owner_space, mutator,
           RtEvent::NO_RT_EVENT, remote_owner_valid_references.exchange(0) + 1);
      // If we have a canonical reference that is not ourselves then 
      // we need to remove the nested reference that we are holding on it too
      IndexSpaceExpression *canon = canonical.load();
      if ((canon != NULL) && (canon != this) &&
          canon->remove_canonical_reference(did))
        // should never actually delete it since we have a resource ref too
        assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::InactiveFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      std::map<AddressSpaceID,RtEvent>::iterator finder =
        send_effects.find(target);
      if (finder != send_effects.end())
        finder->second =
            node->send_remote_gc_decrement(target, mutator, finder->second);
      else
        send_effects[target] = node->send_remote_gc_decrement(target, mutator);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should only become active one time
      assert(tree_active);
      tree_active = false;
#endif
      if (is_owner())
      {
        // Send the removal of the gc references on any remote nodes 
        if (has_remote_instances())
        {
          InactiveFunctor functor(this, mutator, send_effects); 
          map_over_remote_instances(functor);
        }
      }
      else
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
        else if (canon->remove_tree_expression_reference(did))
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
      return (handle.id % rt->runtime_stride);
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
          if (color_map.find(suggestion) == color_map.end())
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
          color_map[0] = NULL;
          return 0;
        }
        std::map<LegionColor,IndexPartNode*>::const_iterator next = 
          color_map.begin();
        if (next->first > 0)
        {
          // save a space for later
          color_map[0] = NULL;
          return 0;
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
      std::atomic<IndexPartitionID> target(0);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(c);
        if (defer == NULL)
          rez.serialize(&target);
        else
          rez.serialize<std::atomic<IndexPartitionID>*>(NULL);
        rez.serialize(ready_event);
      }
      context->runtime->send_index_space_child_request(owner_space, rez);
      if (defer == NULL)
      {
        ready_event.wait();
        IndexPartitionID child_id = target.load();
        if (child_id == 0)
        {
          if (can_fail)
            return NULL;
          REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_COLOR,
            "Unable to find entry for color %lld in "
                          "index space %x.", c, handle.id)
        }
        IndexPartition child_handle(child_id, handle.tid, handle.type_tag);
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
             (color_map[child->color] == NULL));
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
#endif
      color_map.erase(finder);
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
    void IndexSpaceNode::get_colors(std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we need to request an up to date set of colors
      // since it can change arbitrarily
      AddressSpaceID owner_space = get_owner_space();
      if (owner_space != context->runtime->address_space)
      {
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(&colors);
          rez.serialize(ready_event); 
        }
        context->runtime->send_index_space_colors_request(owner_space, rez);
        ready_event.wait();
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
    bool IndexSpaceNode::send_node(AddressSpaceID target, RtEvent done,
                                   RtEvent &send_precondition,
                                   std::set<IndexTreeNode*> &visited,
                                   std::vector<SendNodeRecord> &nodes_to_send,
                                   const bool above /* = false */)
    //--------------------------------------------------------------------------
    {
      // Do a quick check to see if we already visited this node
      if (!visited.insert(this).second)
        return true;
      // If we're not the owner continue up the tree to see if anyone is
      // and check whether they are still valid
      if (!is_owner())
      {
        if (parent != NULL)
        {
          // Try to get a reference on the parent
          {
            AutoLock n_lock(node_lock);
            if (send_references == 0)
              return false;
            send_references++;
          }
          const bool result = parent->send_node(target, done,
              send_precondition, visited, nodes_to_send, true/*above*/);
          // Remove the reference
          bool remove_reference = false;
          {
            AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
            assert(send_references > 0);
#endif
            remove_reference = (--send_references == 0);
          }
          if (remove_reference && parent->remove_nested_resource_ref(did))
            delete parent;
          return result;
        }
        else
          return true;
      }
      // At this point we are the owner
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(tree_active);
#endif
      bool pack_space = false;
      bool still_valid = false;
      bool has_reference = false;
      bool add_root_reference = false;
      // Do our check to see if we're still valid
      {
        AutoLock n_lock(node_lock);
        // Check to see if it is already in the remote set, if so we're done
        if (has_remote_instance(target))
        {
          if (tree_valid)
          {
            // Still need to record the effects so this is not collected early
            std::map<AddressSpaceID,RtEvent>::iterator finder =
              send_effects.find(target);
#ifdef DEBUG_LEGION
            assert(finder != send_effects.end());
#endif
            send_precondition = finder->second;
            if (!send_precondition.has_triggered())
              finder->second = Runtime::merge_events(send_precondition, done);
            else
              finder->second = done;
            return true;
          }
          else
            return false;
        }
        // First see if we are still valid
        if (tree_valid && ((parent == NULL) || (send_references > 0)))
        {
          still_valid = true; 
          // Grab a reference on the parent to keep it from being deleted
          if (parent != NULL)
          {
            send_references++;
            has_reference = true;
          }
          else if (root_valid)
            add_root_reference = true;
        }
        else if (above)
        {
          std::map<AddressSpaceID,RtEvent>::iterator finder =
            send_effects.find(target);
#ifdef DEBUG_LEGION
          assert(finder != send_effects.end());
#endif
          send_precondition = finder->second;
          return false;
        }
        // Record this as an effect for when the node is no longer valid
#ifdef DEBUG_LEGION
        assert(send_effects.find(target) == send_effects.end());
#endif
        send_effects[target] = done;
        update_remote_instances(target);
        // Have to record this atomically with recording as a remote instance
        pack_space = realm_index_space_set.has_triggered();
      }
      // If we have a parent check to see if it is the owner
      // If it is then we can continue traversing up
      if (still_valid && (parent != NULL) &&
          !parent->send_node(target, done, send_precondition,
            visited, nodes_to_send, true/*above*/))
      {
        if (above)
        {
          // If this is above then we don't care about it if it
          // is not still valid
          bool remove_reference = false;
          LocalReferenceMutator mutator;
          if (has_reference)
          {
            AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
            assert(send_references > 0);
#endif
            remove_reference = (--send_references == 0);
            // Remove this from the remote instances since we did
            // not actually end up sending it
            filter_remote_instances(target);
          }
          if (remove_reference && parent->remove_nested_resource_ref(did))
            delete parent;
          return false;
        }
        else
          still_valid = false;
      }
      // Record that we're going to send this node
      nodes_to_send.emplace_back(SendNodeRecord(this, still_valid,
            add_root_reference, pack_space, has_reference));
      return still_valid;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::pack_node(Serializer &rez, AddressSpaceID target,
                                   const SendNodeRecord &record)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(record.node == this);
#endif
      bool remove_reference = false;
      LocalReferenceMutator mutator;
      {
        AutoLock n_lock(node_lock); 
        {
          // Do this before the rez check
          rez.serialize<bool>(true); // is an index space node
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(did);
          if (record.still_valid && (parent != NULL))
            rez.serialize(parent->handle);
          else
            rez.serialize(IndexPartition::NO_PART);
          rez.serialize(color);
          rez.serialize(index_space_ready);
          rez.serialize(expr_id);
          rez.serialize(initialized);
          rez.serialize(depth);
          rez.serialize<bool>(record.add_root_reference);
          if (provenance != NULL)
            provenance->serialize(rez);
          else
            Provenance::serialize_null(rez);
          if (record.pack_space)
            pack_index_space(rez, true/*include size*/);
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
        }
        if (record.has_reference)
        {
#ifdef DEBUG_LEGION
          assert(send_references > 0);
#endif
          remove_reference = (--send_references == 0);
        }
      }
      if (remove_reference && parent->remove_nested_resource_ref(did))
        delete parent;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::invalidate_tree(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent != NULL);
#endif
      bool remove_reference;
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(tree_valid);
#endif
        tree_valid = false;
        remove_reference = (--send_references == 0);
      }
      if (remove_reference && parent->remove_nested_resource_ref(did))
        delete parent;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::InvalidateRootFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target == source)
        return;
      std::map<AddressSpaceID,RtEvent>::const_iterator finder = 
        effects.find(target);
#ifdef DEBUG_LEGION
      assert(finder != effects.end());
#endif
      node->send_remote_valid_decrement(target, &mutator, finder->second);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::invalidate_root(AddressSpaceID source,
                                         std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(root_valid);
#endif
      root_valid = false;
      if (has_remote_instances())
      {
        WrapperReferenceMutator mutator(applied);
        InvalidateRootFunctor functor(source, this, mutator, 
                                      runtime, send_effects);
        map_over_remote_instances(functor);
      }
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
      bool add_root_reference;
      derez.deserialize(add_root_reference);
      Provenance *provenance = Provenance::deserialize(derez);
      size_t index_space_size;
      derez.deserialize(index_space_size);
      const void *index_space_ptr = 
        (index_space_size > 0) ? derez.get_current_pointer() : NULL;

      IndexPartNode *parent_node = NULL;
      if (parent != IndexPartition::NO_PART)
        parent_node = context->get_node(parent, NULL/*defer*/,
                        true/*can fail*/, true/*local only*/);
      IndexSpaceNode *node = context->create_node(handle, index_space_ptr,
          false/*is domain*/, parent_node, color, did, initialized, provenance,
          ready_event, expr_id, NULL/*applied*/, add_root_reference, depth);
#ifdef DEBUG_LEGION
      assert(node != NULL);
#endif
      // Advance the pointer if necessary
      if (index_space_size > 0)
        derez.advance_pointer(index_space_size);
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
      if (target != NULL)
      {
        RtEvent send_precondition;
        std::set<IndexTreeNode*> visited;
        std::vector<SendNodeRecord> nodes_to_send;
        target->send_node(source, to_trigger,
                          send_precondition, visited, nodes_to_send);
        // Now send back the results
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(nodes_to_send.size());
          for (std::vector<SendNodeRecord>::const_iterator it =
                nodes_to_send.begin(); it != nodes_to_send.end(); it++)
            it->node->pack_node(rez, source, *it);
          rez.serialize(to_trigger);
        }
        forest->runtime->send_index_space_return(source, rez,send_precondition);
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
      size_t node_count;
      derez.deserialize(node_count);
      for (unsigned idx = 0; idx < node_count; idx++)
      {
        bool is_index_space;
        derez.deserialize<bool>(is_index_space);
        if (is_index_space)
          IndexSpaceNode::handle_node_creation(context, derez, source);
        else
          IndexPartNode::handle_node_creation(context, derez, source);
      }
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
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
      RtUserEvent ready;
      derez.deserialize(ready);
      IndexSpaceNode *node = context->get_node(handle);
      std::vector<LegionColor> results;
      node->get_colors(results);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize<size_t>(results.size());
        for (std::vector<LegionColor>::const_iterator it = results.begin();
              it != results.end(); it++)
          rez.serialize(*it);
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
        add_base_expression_reference(REMOTE_DID_REF);
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
      // Make sure this doesn't get collected until we're unpacked
      // This could be a performance bug since it will block if we
      // have to send a reference to a remote node, but that should
      // never actually happen
      LocalReferenceMutator mutator;
      add_base_gc_ref(REMOTE_DID_REF, &mutator);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::try_add_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return check_active_and_increment(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return remove_nested_gc_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::try_add_live_reference(ReferenceSource source)
    //--------------------------------------------------------------------------
    {
      return check_active_and_increment(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_live_reference(ReferenceSource source)
    //--------------------------------------------------------------------------
    {
      return remove_base_gc_ref(source);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_base_expression_reference(
              ReferenceSource source, ReferenceMutator *mutator, unsigned count)
    //--------------------------------------------------------------------------
    {
      if (mutator == NULL)
      {
        LocalReferenceMutator local_mutator;
        add_base_valid_ref(source, &local_mutator, count);
      }
      else
        add_base_valid_ref(source, mutator, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_nested_expression_reference(
        DistributedID source, std::set<RtEvent> &applied_events, unsigned count)
    //--------------------------------------------------------------------------
    {
      WrapperReferenceMutator mutator(applied_events);
      add_nested_expression_reference(source, &mutator, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_nested_expression_reference(
                DistributedID source, ReferenceMutator *mutator, unsigned count)
    //--------------------------------------------------------------------------
    {
      if (mutator == NULL)
      {
        LocalReferenceMutator local_mutator;
        add_nested_valid_ref(source, &local_mutator, count);
      }
      else
        add_nested_valid_ref(source, mutator, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_base_expression_reference(
                                         ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_base_valid_ref(source, NULL/*mutator*/, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_nested_expression_reference(
                                           DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_valid_ref(source, NULL/*mutator*/, count);
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
                                 ApUserEvent partial, RtEvent init,
                                 Provenance *prov)
      : IndexTreeNode(ctx, par->depth+1, c,
                      LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_PART_NODE_DC),
                      get_owner_space(p, ctx->runtime), init, prov), 
        handle(p), parent(par), color_space(color_sp), 
        total_children(color_sp->get_volume()), 
        max_linearized_color(color_sp->get_max_linearized_color()),
        partition_ready(part_ready), partial_pending(partial), disjoint(dis), 
        has_complete(comp >= 0), complete(comp != 0), tree_valid(true), 
        send_count(0)
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
                                 int comp, DistributedID did,ApEvent part_ready,
                                 ApUserEvent part,RtEvent init,Provenance *prov)
      : IndexTreeNode(ctx, par->depth+1, c,
                      LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_PART_NODE_DC),
                      get_owner_space(p, ctx->runtime), init, prov), handle(p), 
        parent(par), color_space(color_sp), 
        total_children(color_sp->get_volume()),
        max_linearized_color(color_sp->get_max_linearized_color()),
        partition_ready(part_ready), partial_pending(part), 
        disjoint_ready(dis_ready), disjoint(false), 
        has_complete(comp >= 0), complete(comp != 0), tree_valid(is_owner()), 
        send_count(0)
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
    IndexPartNode::IndexPartNode(const IndexPartNode &rhs)
      : IndexTreeNode(NULL,0,0,0,0,RtEvent::NO_RT_EVENT,NULL), 
        handle(IndexPartition::NO_PART), parent(NULL), color_space(NULL),
        total_children(0), max_linearized_color(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
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
          if ((*it)->remove_partition_reference(NULL))
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
      for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
            color_map.begin(); it != color_map.end(); it++)
        if (it->second->remove_nested_resource_ref(did))
          delete it->second;
    }

    //--------------------------------------------------------------------------
    IndexPartNode& IndexPartNode::operator=(const IndexPartNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::InvalidFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      std::map<AddressSpaceID,RtEvent>::iterator finder =
        send_effects.find(target);
      if (finder != send_effects.end())
        finder->second =
            node->send_remote_gc_decrement(target, mutator, finder->second);
      else
        send_effects[target] = node->send_remote_gc_decrement(target, mutator);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we add a valid reference to the owner
      if (is_owner())
      {
#ifdef DEBUG_LEGION
        // We should only become valid once on the owner node
        assert(tree_valid);
#endif
        // Check to see if the color space has a parent partition, if so we
        // add the valid reference there since valid references on index spaces
        // that are not the root mean it is a valid expression
        if (color_space->parent != NULL)
          color_space->parent->add_nested_valid_ref(did, mutator);
        else
          color_space->add_nested_valid_ref(did, mutator);
        // Add it to the partition of our parent if it exists, otherwise
        // our parent index space is a root so we add the reference there
        if (parent->parent != NULL)
          parent->parent->add_nested_valid_ref(did, mutator);
        else
          parent->add_nested_valid_ref(did, mutator);
      }
      else
        send_remote_valid_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        // Remove gc references from our remote nodes
        if (has_remote_instances())
        {
          // Make sure invalidation are not handled before send effects
          InvalidFunctor functor(this, mutator, send_effects);
          map_over_remote_instances(functor);
        }
        // Remove the valid reference that we hold on the color space
        if (color_space->parent != NULL)
          color_space->parent->remove_nested_valid_ref(did, mutator);
        else
          color_space->remove_nested_valid_ref(did, mutator);
        // Remove valid ref on partition of parent if it exists, otherwise
        // our parent index space is a root so we remove the reference there
        if (parent->parent != NULL)
        {
          if (parent->parent->remove_nested_valid_ref(did, mutator))
            delete parent->parent;
        }
        else
          parent->remove_nested_valid_ref(did, mutator);
      }
      else // Remove the valid reference that we have on the owner
        send_remote_valid_decrement(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Finally remove valid references on all owner children and any trackers
      // We should not need a lock at this point since nobody else should
      // be modifying these data structures at this point
      // We still hold resource references to the node so we don't need to
      // worry about the child nodes being deleted
      parent->remove_child(color);
      std::vector<IndexSpaceNode*> to_invalidate;
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(tree_valid);
#endif
        tree_valid = false;
        to_invalidate.reserve(color_map.size());
        for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
              color_map.begin(); it != color_map.end(); it++)
          to_invalidate.push_back(it->second);
      }
      for (std::vector<IndexSpaceNode*>::const_iterator it =
            to_invalidate.begin(); it != to_invalidate.end(); it++)
      {
        (*it)->invalidate_tree();
        // Remove the nested valid reference on this index space node
        if ((*it)->remove_nested_valid_ref(did, mutator))
          assert(false); // still holding resource ref so should never be hit 
      }
      if (!partition_trackers.empty())
      {
        for (std::list<PartitionTracker*>::const_iterator it = 
              partition_trackers.begin(); it != partition_trackers.end(); it++)
          if ((*it)->remove_partition_reference(mutator))
            delete (*it);
        partition_trackers.clear();
      }
      // Remove the reference on our union expression if we have one
      IndexSpaceExpression *expr = union_expr.load();
      if ((expr != NULL) && expr->remove_nested_expression_reference(did))
        delete expr;
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
      return (part.id % runtime->runtime_stride);
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
      AddressSpaceID owner_space = get_owner_space();
      AddressSpaceID local_space = context->runtime->address_space;
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
            result = context->create_node(is, NULL/*realm is*/, this, c, did, 
                                          initialized,provenance,partial_event);
            add_pending_child(c, partial_event);
            // Now check to see if we need to trigger our partition ready event
            std::set<ApEvent> child_ready_events;
            {
              AutoLock n_lock(node_lock,1,false/*exclusvie*/);
              if (color_map.size() == size_t(total_children))
              {
                for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
                      color_map.begin(); it != color_map.end(); it++)
                  child_ready_events.insert(it->second->index_space_ready);
              }
            }
            if (!child_ready_events.empty())
              Runtime::trigger_event(NULL, partial_pending,
                  Runtime::merge_events(NULL, child_ready_events));
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
          IndexSpaceID cid = child_id.load();
#ifdef DEBUG_LEGION
          assert(cid > 0);
#endif
          IndexSpace child_handle(cid, 
              handle.get_tree_id(), handle.get_type_tag());
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
    bool IndexPartNode::add_child(IndexSpaceNode *child) 
    //--------------------------------------------------------------------------
    {
      // This child should live as long as we are alive
      child->add_nested_resource_ref(did);
      RtUserEvent to_trigger;
      bool result;
      {
        AutoLock n_lock(node_lock);
        result = tree_valid;
#ifdef DEBUG_LEGION
        assert(color_map.find(child->color) == color_map.end());
#endif
        color_map[child->color] = child;
        std::map<LegionColor,RtUserEvent>::iterator finder = 
          pending_child_map.find(child->color);
        if (finder == pending_child_map.end())
          return result;
        to_trigger = finder->second;
        pending_child_map.erase(finder);
      }
      Runtime::trigger_event(to_trigger);
      return result;
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
    void IndexPartNode::RemoteDisjointnessFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != runtime->address_space)
        runtime->send_index_partition_disjoint_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::compute_disjointness(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(disjoint_ready.exists() && !disjoint_ready.has_triggered());
      assert(ready_event == disjoint_ready);
#endif
      // Now do the pairwise disjointness tests
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
          ColorSpaceIterator *itr = color_space->create_color_space_iterator();
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
        assert(disjoint_ready == ready_event);
#endif
        // We have to send notifications before any other remote
        // requests can record themselves so we need to do it 
        // while we are holding the lock
        if (has_remote_instances())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(disjoint);
          }
          RemoteDisjointnessFunctor functor(rez, context->runtime);
          map_over_remote_instances(functor);
        }
      }
      // Once we get here, we know the disjointness result so we can
      // trigger the event saying when the disjointness value is ready
      Runtime::trigger_event(ready_event);
      if (runtime->profiler != NULL)
	runtime->profiler->record_index_partition(parent->handle.id, handle.id,
            disjoint, color);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_disjoint(bool app_query)
    //--------------------------------------------------------------------------
    {
      if (!disjoint_ready.has_triggered())
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
        // We can always write the result immediately since we know
        // that the common sub-expression code will give the same
        // result if there is a race
        IndexSpaceExpression *expr = context->union_index_spaces(child_spaces);
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
    void IndexPartNode::get_colors(std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      color_space->instantiate_colors(colors);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_pending_child(const LegionColor child_color,
                                          ApUserEvent domain_ready)
    //--------------------------------------------------------------------------
    {
      bool launch_remove = false;
      {
        AutoLock n_lock(node_lock);
        // Duplicate insertions can happen legally so avoid them
        if (pending_children.find(child_color) == pending_children.end())
        {
          pending_children[child_color] = domain_ready;
          launch_remove = true;
        }
      }
      if (launch_remove)
      {
        PendingChildArgs args(this, child_color);
        // Don't remove the pending child until the handle is ready
        context->runtime->issue_runtime_meta_task(args,
          LG_LATENCY_WORK_PRIORITY, Runtime::protect_event(domain_ready));
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::get_pending_child(const LegionColor child_color,
                                          ApUserEvent &domain_ready)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock, 1, false/*exclusive*/);
      std::map<LegionColor,ApUserEvent>::const_iterator finder = 
        pending_children.find(child_color);
      if (finder != pending_children.end())
      {
        domain_ready = finder->second;
        return true;
      }
      return false;
    }
    
    //--------------------------------------------------------------------------
    void IndexPartNode::remove_pending_child(const LegionColor child_color)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      pending_children.erase(child_color);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_pending_child_task(const void *args)
    //--------------------------------------------------------------------------
    {
      const PendingChildArgs *pargs = (const PendingChildArgs*)args;  
      pargs->parent->remove_pending_child(pargs->pending_child);
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
                                   const FutureMap &weights, size_t granularity)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_weights(op, this, weights.impl, granularity); 
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
    bool IndexPartNode::send_node(AddressSpaceID target, RtEvent done,
                                  RtEvent &send_precondition,
                                  std::set<IndexTreeNode*> &visited,
                                  std::vector<SendNodeRecord> &nodes_to_send,
                                  const bool above /* = false */)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent != NULL);
#endif
      // Do a quick check to see if we already visited this
      if (!visited.insert(this).second)
        return true;
      if (!is_owner())
      {
#ifdef DEBUG_LEGION
        assert(above);
#endif
        // See if anything above or the color space needs to be sent
        if (!parent->send_node(target, done, send_precondition,
                               visited, nodes_to_send, true/*above*/))
          return false;
        RtEvent temp_precondition;
        color_space->send_node(target, done, temp_precondition,
                               visited, nodes_to_send, false/*above*/);
        if (temp_precondition.exists())
        {
          if (send_precondition.exists())
            send_precondition =
              Runtime::merge_events(send_precondition, temp_precondition);
          else
            send_precondition = temp_precondition;
        }
        return true;
      }
      // At this point we are the owner
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // Do our check to see if we're still valid
      {
        AutoLock n_lock(node_lock);
        if (!tree_valid)
          return false;
        if (has_remote_instance(target))
        {
          // Still need to record the effects so this is not collected early
          std::map<AddressSpaceID,RtEvent>::iterator finder =
            send_effects.find(target);
#ifdef DEBUG_LEGION
          assert(finder != send_effects.end());
#endif
          send_precondition = finder->second;
          if (!send_precondition.has_triggered())
            finder->second = Runtime::merge_events(send_precondition, done);
          else
            finder->second = done;
          return true;
        }
        // Record this as an effect for when the node is no longer valid
#ifdef DEBUG_LEGION
        assert(send_effects.find(target) == send_effects.end());
#endif
        send_effects[target] = done; 
        send_count++;
        update_remote_instances(target);
      }
      if (!parent->send_node(target, done, send_precondition,
                             visited, nodes_to_send, true/*above*/))
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(send_count > 0);
#endif
        if (((--send_count) == 0) && send_done.exists())
        {
          Runtime::trigger_event(send_done);
          send_done = RtUserEvent::NO_RT_USER_EVENT;
        }
        return false;
      }
      RtEvent temp_precondition;
      color_space->send_node(target, done, temp_precondition,
                             visited, nodes_to_send, false/*above*/);
      if (temp_precondition.exists())
      {
        if (send_precondition.exists())
          send_precondition =
            Runtime::merge_events(send_precondition, temp_precondition);
        else
          send_precondition = temp_precondition;
      }
      // Record that we will be sending this node
      nodes_to_send.emplace_back(SendNodeRecord(this));
      return true;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::pack_node(Serializer &rez, AddressSpaceID target,
                                  const SendNodeRecord &record)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // We're guaranteed to send at this point so we can remove the guard
#ifdef DEBUG_LEGION
      assert(send_count > 0);
#endif
      if (((--send_count) == 0) && send_done.exists())
      {
        Runtime::trigger_event(send_done);
        send_done = RtUserEvent::NO_RT_USER_EVENT;
      }
      // Check to see if we have computed the disjointness result
      // If not we'll record that we need to do it and then when it 
      // is computed we'll send out the result to all the remote copies
      const bool has_disjoint = 
        (!disjoint_ready.exists() || disjoint_ready.has_triggered());
      const bool disjoint_result = has_disjoint ? is_disjoint() : false;
      {
        // Do this before the rez check
        rez.serialize<bool>(false); // is not an index space node
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
        rez.serialize<size_t>(pending_children.size());
        for (std::map<LegionColor,ApUserEvent>::const_iterator it = 
              pending_children.begin(); it != pending_children.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
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
      ApUserEvent partial_pending;
      derez.deserialize(partial_pending);
      RtEvent initialized;
      derez.deserialize(initialized);
      Provenance *provenance = Provenance::deserialize(derez);
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
            partial_pending, initialized) :
        context->create_node(handle, parent_node, color_space_node, color,
            dis_ready, complete, did, provenance, ready_event, 
            partial_pending, initialized);
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
      size_t num_pending;
      derez.deserialize(num_pending);
      for (unsigned idx = 0; idx < num_pending; idx++)
      {
        LegionColor child_color;
        derez.deserialize(child_color);
        ApUserEvent child_ready;
        derez.deserialize(child_ready);
        node->add_pending_child(child_color, child_ready);
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
      RtEvent send_precondition;
      std::set<IndexTreeNode*> visited;
      std::vector<SendNodeRecord> nodes_to_send;
      if ((target != NULL) &&
          target->send_node(source, to_trigger, send_precondition,
                            visited, nodes_to_send))
      {
        // Now send back the results
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(nodes_to_send.size());
          for (std::vector<SendNodeRecord>::const_iterator it =
                nodes_to_send.begin(); it != nodes_to_send.end(); it++)
            it->node->pack_node(rez, source, *it);
          rez.serialize(to_trigger);
        }
        forest->runtime->send_index_partition_return(source, rez,
                                                     send_precondition);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(nodes_to_send.empty());
#endif
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_return(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t node_count;
      derez.deserialize(node_count);
      for (unsigned idx = 0; idx < node_count; idx++)
      {
        bool is_index_space;
        derez.deserialize<bool>(is_index_space);
        if (is_index_space)
          IndexSpaceNode::handle_node_creation(context, derez, source);
        else
          IndexPartNode::handle_node_creation(context, derez, source);
      }
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
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

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx,
                              DistributedID did, RtEvent init, Provenance *prov)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC), 
          get_owner_space(sp, ctx->runtime), false/*register with runtime*/),
        handle(sp), context(ctx), provenance(prov), initialized(init), 
        allocation_state(FIELD_ALLOC_READ_ONLY), outstanding_allocators(0),
        outstanding_invalidations(0)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      unallocated_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
      local_index_infos.resize(runtime->max_local_fields, 
          std::pair<size_t,CustomSerdezID>(0, 0));
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
          get_owner_space(sp, ctx->runtime), false/*register with runtime*/),
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
    FieldSpaceNode::FieldSpaceNode(const FieldSpaceNode &rhs)
      : DistributedCollectable(NULL, 0, 0), handle(FieldSpace::NO_SPACE), 
        context(NULL), provenance(NULL), initialized(rhs.initialized),
        node_lock(rhs.node_lock)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
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
    FieldSpaceNode& FieldSpaceNode::operator=(const FieldSpaceNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(void)
      : field_size(0), idx(0), serdez_id(0), provenance(NULL), local(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(size_t size, unsigned id, 
                                 CustomSerdezID sid, Provenance *prov, bool loc)
      : field_size(size), idx(id), serdez_id(sid), provenance(prov), local(loc)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(ApEvent ready, unsigned id,
                                 CustomSerdezID sid, Provenance *prov, bool loc)
      : field_size(0), size_ready(ready), idx(id), serdez_id(sid),
        provenance(prov), local(loc)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(const FieldInfo &rhs)
      : field_size(rhs.field_size), size_ready(rhs.size_ready), idx(rhs.idx),
        serdez_id(rhs.serdez_id), provenance(rhs.provenance), local(rhs.local)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(FieldInfo &&rhs)
      : field_size(rhs.field_size), size_ready(rhs.size_ready), idx(rhs.idx),
        serdez_id(rhs.serdez_id), provenance(rhs.provenance), local(rhs.local)
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
      derez.deserialize<bool>(local);
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we add a valid reference to the owner
      if (!is_owner())
        send_remote_valid_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_valid_decrement(owner_space, mutator);
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
      return (handle.id % rt->runtime_stride);
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
                                             RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
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
              add_base_resource_ref(FIELD_ALLOCATOR_REF);
              // Send the invalidation and make ourselves the new 
              // pending exclusive allocator value
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(handle);
                rez.serialize(ready_event);
              }
              runtime->send_field_space_allocator_invalidation(remote_owner, 
                                                               rez); 
              outstanding_allocators = 1;
              pending_field_allocation = ready_event;
              allocation_state = FIELD_ALLOC_PENDING;
              break;
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
                  add_base_resource_ref(FIELD_ALLOCATOR_REF);
                  Serializer rez;
                  {
                    RezCheck z(rez);
                    rez.serialize(handle);
                    rez.serialize(done);
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
                  rez.serialize(unallocated_indexes);
                  unallocated_indexes.clear();
                  rez.serialize<size_t>(available_indexes.size());
                  for (std::list<std::pair<unsigned,RtEvent> >::const_iterator
                        it = available_indexes.begin(); it !=
                        available_indexes.end(); it++)
                  {
                    rez.serialize(it->first);
                    rez.serialize(it->second);
                  }
                  available_indexes.clear();
                  rez.serialize(ready_event);
                }
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
          case FIELD_ALLOC_COLLECTIVE:
            {
              // TODO: implement this for control replication
              assert(false);
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
    RtEvent FieldSpaceNode::destroy_allocator(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert((allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
             (allocation_state == FIELD_ALLOC_COLLECTIVE) ||
             (allocation_state == FIELD_ALLOC_INVALID));
#endif
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
                     const char *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fids.empty());
#endif
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);
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
        field_infos[fid] = FieldInfo(sizes[idx], index, serdez_id, prov);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::initialize_fields(ApEvent sizes_ready,
                     const std::vector<FieldID> &fids, CustomSerdezID serdez_id,
                     Provenance *provenance)
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
        field_infos[fid] = FieldInfo(sizes_ready, index, serdez_id, provenance);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_field(FieldID fid, size_t size, 
                               CustomSerdezID serdez_id, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
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
          if (provenance != NULL)
          {
            const size_t length = strlen(provenance);
            rez.serialize<size_t>(length);
            rez.serialize(provenance, length);
          }
          else
            rez.serialize<size_t>(0);
          rez.serialize<size_t>(1); // only allocating one field
          rez.serialize(fid);
          rez.serialize(size);
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      if (field_infos.find(fid) != field_infos.end())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
          "Illegal duplicate field ID %d used by the "
                        "application in field space %d", fid, handle.id)
      // Find an index in which to allocate this field  
      RtEvent ready_event;
      int result = allocate_index(ready_event);
      if (result < 0)
        REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
                        "Exceeded maximum number of allocated fields for "
                        "field space %x. Change LEGION_MAX_FIELDS from %d and"
                        " related macros at the top of legion_config.h and "
                        "recompile.", handle.id, LEGION_MAX_FIELDS)
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);
      const unsigned index = result;
      field_infos[fid] = FieldInfo(size, index, serdez_id, prov);
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_field(FieldID fid, ApEvent size_ready, 
                               CustomSerdezID serdez_id, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
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
          if (provenance != NULL)
          {
            const size_t length = strlen(provenance);
            rez.serialize<size_t>(length);
            rez.serialize(provenance, length);
          }
          else
            rez.serialize<size_t>(0);
          rez.serialize<size_t>(1); // only allocating one field
          rez.serialize(fid);
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      if (field_infos.find(fid) != field_infos.end())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
          "Illegal duplicate field ID %d used by the "
                        "application in field space %d", fid, handle.id)
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
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);
      field_infos[fid] = FieldInfo(size_ready, index, serdez_id, prov);
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_fields(const std::vector<size_t> &sizes,
                                            const std::vector<FieldID> &fids,
                                            CustomSerdezID serdez_id,
                                            const char *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fids.empty());
      assert(sizes.size() == fids.size());
#endif
      AutoLock n_lock(node_lock);
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
          if (provenance != NULL)
          {
            const size_t length = strlen(provenance);
            rez.serialize<size_t>(length);
            rez.serialize(provenance, length);
          }
          else
            rez.serialize<size_t>(0);
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
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);
      // We're the owner so do the field allocation
      std::set<RtEvent> allocated_events;
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        FieldID fid = fids[idx];
        if (field_infos.find(fid) != field_infos.end())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
            "Illegal duplicate field ID %d used by the "
                          "application in field space %d", fid, handle.id)
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
        field_infos[fid] = FieldInfo(sizes[idx], index, serdez_id, prov);
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
                                            const char *provenance)
    //--------------------------------------------------------------------------
    { 
      AutoLock n_lock(node_lock);
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
          if (provenance != NULL)
          {
            const size_t length = strlen(provenance);
            rez.serialize<size_t>(length);
            rez.serialize(provenance, length);
          }
          else
            rez.serialize<size_t>(0);
          rez.serialize<size_t>(fids.size());
          for (unsigned idx = 0; idx < fids.size(); idx++)
            rez.serialize(fids[idx]);
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);
      // We're the owner so do the field allocation
      std::set<RtEvent> allocated_events;
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        FieldID fid = fids[idx];
        if (field_infos.find(fid) != field_infos.end())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
            "Illegal duplicate field ID %d used by the "
                          "application in field space %d", fid, handle.id)
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
        field_infos[fid] = FieldInfo(sizes_ready, index, serdez_id, prov);
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
          context->runtime->send_field_size_update(owner_space, rez);
          update_events.insert(done_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field(FieldID fid, AddressSpaceID source,
                                    std::set<RtEvent> &applied)   
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock); 
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
                              AddressSpaceID source, std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock); 
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
                                            RtEvent freed_event)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
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
                                            const char *provenance)
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
          if (provenance != NULL)
          {
            const size_t length = strlen(provenance);
            rez.serialize<size_t>(length);
            rez.serialize(provenance, length);
          }
          else
            rez.serialize<size_t>(0);
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
        Provenance *prov = NULL;
        if (provenance != NULL)
          prov = new Provenance(provenance);
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
        Provenance *prov = NULL;
        if (provenance != NULL)
          prov = new Provenance(provenance);
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
                                           const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(to_free.size() == indexes.size());
#endif
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
      }
      else
      {
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
      size_t length;
      derez.deserialize(length);
      const char *provenance = NULL;
      if (length > 0)
      {
        provenance = (const char*)derez.get_current_pointer();
        derez.advance_pointer(length);
      }
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
      size_t length;
      derez.deserialize(length);
      const char *provenance = NULL;
      if (length > 0)
      {
        provenance = (const char*)derez.get_current_pointer();
        derez.advance_pointer(length);
      }
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
      node->free_local_fields(fields, indexes);
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
      std::vector<size_t> field_sizes(field_set.size());
      std::vector<unsigned> mask_index_map(field_set.size());
      std::vector<CustomSerdezID> serdez(field_set.size());
      FieldMask external_mask;
      compute_field_layout(field_set, field_sizes, 
                           mask_index_map, serdez, external_mask);
      // Now make the instance, this should always succeed
      ApEvent ready_event;
      size_t instance_footprint;
      LayoutConstraintSet constraints;
      PhysicalInstance inst = 
        attach_op->create_instance(node->row_source, field_set, field_sizes, 
                                   constraints, ready_event,instance_footprint);
      // Check to see if this instance is local or whether we need
      // to send this request to a remote node to make
      if (inst.address_space() != context->runtime->address_space)
      {
        Serializer rez;
        std::atomic<DistributedID> remote_did(0);
        const RtUserEvent wait_for = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(inst);
          rez.serialize(ready_event);
          rez.serialize(instance_footprint);
          constraints.serialize(rez);
          rez.serialize(external_mask);
          rez.serialize<size_t>(field_set.size());
          for (unsigned idx = 0; idx < field_set.size(); idx++)
          {
            rez.serialize(field_set[idx]);
            rez.serialize(field_sizes[idx]);
            rez.serialize(mask_index_map[idx]);
            rez.serialize(serdez[idx]);
          }
          rez.serialize(node->handle);
          rez.serialize(&remote_did);
          rez.serialize(wait_for);
        }
        runtime->send_external_create_request(inst.address_space(), rez);
        // Wait for the response to come back
        wait_for.wait();
        // Now we can request the physical manager
        RtEvent wait_on;
        PhysicalManager *result = 
         context->runtime->find_or_request_instance_manager(remote_did,wait_on);
        if (wait_on.exists())
          wait_on.wait();
        return InstanceRef(result, external_mask);
      }
      else // Local so we can just do this call here
        return InstanceRef(create_external_manager(inst, ready_event, 
                            instance_footprint, constraints, field_set, 
                            field_sizes,  external_mask, mask_index_map, 
                            node, serdez), external_mask);
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
      std::atomic<DistributedID> *did_ptr;
      derez.deserialize(did_ptr);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      PhysicalManager *manager = fs->create_external_manager(inst, ready_event,
          footprint, constraints, field_set, field_sizes, file_mask,
          mask_index_map, region_node, serdez);
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(did_ptr);
        rez.serialize(manager->did);
        rez.serialize(done_event);
      }
      runtime->send_external_create_response(source, rez);
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
            const std::vector<unsigned> &mask_index_map,
            RegionNode *node, const std::vector<CustomSerdezID> &serdez)
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
      DistributedID did = context->runtime->get_available_distributed_id();
      MemoryManager *memory = 
        context->runtime->find_memory_manager(inst.get_location());
      IndividualManager *result = new IndividualManager(context, did, 
                                         context->runtime->address_space,
                                         memory, inst, node->row_source, 
                                         NULL/*piece list*/, 
                                         0/*piece list size*/,
                                         node->column_source,
                                         node->handle.get_tree_id(),
                                         layout, 0/*redop*/, 
                                         true/*register now*/,
                                         instance_footprint, ready_event,
                                         true/*external instance*/);
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
      Provenance *provenance = Provenance::deserialize(derez);
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
      // Add a reference to this node to keep it alive until we get the
      // corresponding free operation from the remote node
#ifdef DEBUG_LEGION
      assert(node->is_owner());
#endif
      node->add_base_resource_ref(FIELD_ALLOCATOR_REF);
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

      FieldSpaceNode *node = forest->get_node(handle);
      node->process_allocator_invalidation(done_event);
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
      if (node->remove_base_resource_ref(FIELD_ALLOCATOR_REF,
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
      if (node->remove_base_resource_ref(FIELD_ALLOCATOR_REF))
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
              (allocation_state == FIELD_ALLOC_READ_ONLY));
      assert(!unallocated_indexes);
      assert(available_indexes.empty());
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
      // Make that we now have this in exclusive mode
      outstanding_allocators = 1;
      allocation_state = FIELD_ALLOC_EXCLUSIVE;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_allocator_invalidation(RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert((allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
             (allocation_state == FIELD_ALLOC_READ_ONLY));
#endif
      Serializer rez;
      if (allocation_state == FIELD_ALLOC_EXCLUSIVE)
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize<bool>(true); // allocation meta data
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
                         FieldSpaceNode *column_src, RtEvent init, RtEvent tree,
                         Provenance *prov, DistributedID id)
      : DistributedCollectable(ctx->runtime, 
            LEGION_DISTRIBUTED_HELP_ENCODE((id > 0) ? id :
              ctx->runtime->get_available_distributed_id(),
              REGION_TREE_NODE_DC), (id > 0) ? ctx->runtime->determine_owner(id)
            : ctx->runtime->address_space, false/*register with runtime*/),
        context(ctx), column_source(column_src), provenance(prov),
        initialized(init), tree_initialized(tree), registered(false)
#ifdef DEBUG_LEGION
        , currently_active(true)
#endif
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
    void RegionTreeNode::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_active);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID RegionTreeNode::get_owner_space(RegionTreeID tid,
                                                              Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (tid % runtime->runtime_stride);
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
                                             RegionTreePath &path,
                                             const LogicalTraceInfo &trace_info,
                                             const ProjectionInfo &proj_info,
                                             FieldMask &unopened_field_mask,
                                             FieldMask &already_closed_mask,
                                             std::set<RtEvent> &applied_events)
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
      FieldMask open_below;
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
                      proj_info, captures_closes, open_below, applied_events);
        }
        else
        {
          const FieldMask *aliased_children = path.get_aliased_children(depth);
          siphon_logical_children(closer, state, unopened_field_mask, 
                                  aliased_children, captures_closes, next_child,
                                  open_below, applied_events);
        }
        // We always need to create and register close operations
        // regardless of whether we are tracing or not
        // If we're not replaying a trace we need to do work here
        // See if we need to register a close operation
        if (closer.has_close_operations(already_closed_mask))
        {
          // Generate the close operations         
          closer.initialize_close_operations(state, user.op, trace_info);
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
        // See if we have any open_only fields to merge
        if (!arrived || proj_info.is_projecting())
        {
          FieldMask open_only = user.field_mask - unopened_field_mask;
          if (!!open_only)
            add_open_field_state(state, arrived, proj_info, user, 
                                 open_only, next_child, applied_events);
        }
      }
      else if (!arrived || proj_info.is_projecting())
      {
        // Everything is open-only so make a state and merge it in
        add_open_field_state(state, arrived, proj_info, user, 
                             user.field_mask, next_child, applied_events);
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
        // Handle any projections that we might have
        // If we're projecting, record any split fields and projection epochs
        if (proj_info.is_projecting())
        {
          // If we're writing, then record our projection info in 
          // the current projection epoch else do the same for non-writes
          state.update_projection_epochs(user.field_mask, proj_info);
        }
        else if (user.usage.redop > 0)
        {
          // Not projecting and doing a reduction of some kind so record it
          record_logical_reduction(state, user.usage.redop, user.field_mask);
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
        else // if they weren't open here, they shouldn't be open below
          assert(!open_below);
#endif
        next_child->register_logical_user(ctx, user, path, trace_info,
           proj_info, unopened_field_mask, already_closed_mask, applied_events);
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
                                              RegionTreeNode *next_child,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      if (arrived && proj_info.is_projecting())
      {
        FieldState new_state(user.usage, open_mask, proj_info.projection,
           proj_info.projection_space, are_all_children_disjoint());
        merge_new_field_state(state, new_state);
      }
      else if (next_child != NULL)
      {
        FieldState new_state(user, open_mask, next_child, applied_events);
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
            state.advance_projection_epochs(overlap);
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
                                              FieldMask &open_below,
                                              std::set<RtEvent> &applied_events)
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
                const bool needs_upgrade = HAS_WRITE(closer.user.usage);
                FieldMask already_open;
                perform_close_operations(closer, current_mask, 
                                         *it, next_child,
                                         true/*allow next*/,
                                         aliased_children,
                                         needs_upgrade,
                                         true/*read only close*/,
                                         false/*overwriting close*/,
                                         record_close_operations,
                                         false/*record closed fields*/,
                                         already_open);
                open_below |= already_open;
                if (needs_upgrade && !!already_open)
                {
                  FieldState new_state(closer.user, already_open, 
                                       next_child, applied_events);
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
                                           cit->first, applied_events);
                      // Add the next child as well
                      new_state.add_child(next_child, already_open, 
                                          applied_events);
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
                                           true/*allow next*/,
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
                                         next_child, applied_events);
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
                                         false/*allow next child*/,
                                         NULL/*aliased children*/,
                                         false/*needs upgrade*/,
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
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
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
          case OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW:
            {
              // Have to close up this sub-tree no matter what
              if (record_close_operations)
              {
                const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                assert(!!overlap);
#endif
                closer.record_close_operation(overlap);
                // Advance the projection epochs
                state.advance_projection_epochs(overlap);
              }
              it->filter(current_mask);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_REDUCE_PROJ:
          case OPEN_REDUCE_PROJ_DIRTY:
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
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
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
        FieldState new_state(closer.user, open_mask, next_child,applied_events);
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
                                              FieldMask &open_below,
                                              std::set<RtEvent> &applied_events)
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
                // Keep going
                it++;
              }
              else
              {
                // Otherwise we are going to a different mode 
                // no need to do a close since we're staying in
                // projection mode, but we do need to advance
                // the projection epochs.
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
              if ((it->projection == proj_info.projection) &&
                  it->projection_domain_dominates(proj_info.projection_space)) 
              {
                // If we're a reduction we have to go into a dirty 
                // reduction mode since we know we're already open below
                if (IS_REDUCE(closer.user.usage)) 
                {
                  // Go to the dirty reduction mode
                  const FieldMask overlap = it->valid_fields() & current_mask;
                  // Record that some fields are already open
                  open_below |= overlap;
                  // Make the new state to add
                  FieldState new_state(closer.user.usage, overlap, 
                           proj_info.projection, proj_info.projection_space, 
                           are_all_children_disjoint(), true/*dirty reduce*/);
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
                  // Update the domain  
                  it->projection_space = proj_info.projection_space;
                  open_below |= (it->valid_fields() & current_mask);
                  it++;
                }
              }
              else
              {
                // Now we need the close operation
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
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
          case OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW:
            {
#ifdef DEBUG_LEGION
              // Can only be here if all children are disjoint
              assert(are_all_children_disjoint());
#endif
              if (IS_REDUCE(closer.user.usage) &&
                  it->projection_domain_dominates(proj_info.projection_space))
              {
                const FieldMask overlap = it->valid_fields() & current_mask;
                // Record that some fields are already open
                open_below |= overlap;
                // Make the new state to add
                FieldState new_state(closer.user.usage, overlap, 
                         proj_info.projection, proj_info.projection_space, 
                         are_all_children_disjoint(), true/*dirty reduce*/);
                new_states.emplace_back(std::move(new_state));
                // If we are a reduction, we can go straight there
                it->filter(overlap);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              else if (IS_READ_ONLY(closer.user.usage))
              {
                // Read-only projections of any depth allow
                // us to stay in disjoint shallow mode because
                // they are not going to mutate the state at
                // all and we can catch dependences on any
                // index spaces without needing a close operation
                it++;
              }
              else if ((proj_info.projection->depth == 0) && 
                        !IS_REDUCE(closer.user.usage))
              {
                // If we are also disjoint shallow we can stay in this mode
                // Exception: reductions that are larger than the current
                // domain are bad cause we can't do advances properly for
                // our projection epoch
                open_below |= (it->valid_fields() & current_mask);
                it++;
              }
              else
              {
                // Otherwise we need a close operation
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap);
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
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
          case OPEN_REDUCE_PROJ_DIRTY:
            {
              // Reduce projections of the same kind can always stay open
              // otherwise we need a close operation
              if (closer.user.usage.redop != it->redop)
              {
                // We need a close operation here
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields();
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
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
              are_all_children_disjoint());
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
            state.advance_projection_epochs(overlap);
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
        assert(allow_next_child);
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
                if (next_child->remove_base_valid_ref(FIELD_STATE_REF))
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
                  if (next_child->remove_base_valid_ref(FIELD_STATE_REF))
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
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_dependences(ContextID ctx, 
                      Operation *op, const FieldMask &field_mask, bool dominate)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      for (LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::iterator 
            it = state.curr_epoch_users.begin(); it != 
            state.curr_epoch_users.end(); /*nothing*/)
      {
        if (!(it->field_mask * field_mask))
        {
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(),
              it->uid, it->idx, op->get_unique_op_id(),
              0/*idx*/, TRUE_DEPENDENCE);
#endif
          // Do this after the logging since we 
          // are going to update the iterator
          if (op->register_dependence(it->op, it->gen))
          {
#ifndef LEGION_SPY
            // Prune it from the list
            it = state.curr_epoch_users.erase(it);
#else
            it++;
#endif
          }
          else if (dominate)
            it = state.curr_epoch_users.erase(it);
          else
            it++;
        }
        else if (dominate)
          it = state.curr_epoch_users.erase(it);
        else
          it++;
      }
      for (LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::iterator 
            it = state.prev_epoch_users.begin(); it != 
            state.prev_epoch_users.end(); /*nothing*/)
      {
        if (!(it->field_mask * field_mask))
        {
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(),
              it->uid, it->idx, op->get_unique_op_id(), 0/*idx*/, 
              TRUE_DEPENDENCE);
#endif
          // Do this after the logging since we are going
          // to update the iterator
          if (op->register_dependence(it->op, it->gen))
          {
#ifndef LEGION_SPY
            // Prune it from the list
            it = state.prev_epoch_users.erase(it);
#else
            it++;
#endif
          }
          else if (dominate)
            it = state.prev_epoch_users.erase(it);
          else
            it++;
        }
        else if (dominate)
          it = state.prev_epoch_users.erase(it);
        else
          it++;
      }
    } 

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_deletion(ContextID ctx,
                                             const LogicalUser &user,
                                             const FieldMask &check_mask,
                                             RegionTreePath &path,
                                             const LogicalTraceInfo &trace_info,
                                             FieldMask &already_closed_mask,
                                             std::set<RtEvent> &applied_events,
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
              open_below, ((depth+1) == path.get_max_depth()), applied_events);
          if (closer.has_close_operations(already_closed_mask))
          {
            // Generate the close operations         
            // We need to record the version numbers for this node as well
            closer.initialize_close_operations(state, user.op, trace_info);
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
            trace_info, already_closed_mask, applied_events, invalidate_tree);
      }
      else
      {
        // Register dependences on any users in the sub-tree
        if (!!check_mask)
        {
          // Don't dominate so later deletions with overlapping region
          // requirements can catch the same dependences
          LogicalRegistrar registrar(ctx, user.op, 
                                     check_mask, false/*dominate*/);
          visit_node(&registrar);
        }
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
                                              bool force_close_next,
                                              std::set<RtEvent> &applied_events)
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
              // Advance the projection epochs
              state.advance_projection_epochs(overlap);
              it->filter(current_mask);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_READ_WRITE_PROJ:
          case OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW:
          case OPEN_REDUCE_PROJ:
            {
              // Do the close here 
              closer.record_close_operation(overlap);
              // Advance the projection epochs
              state.advance_projection_epochs(overlap);
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
    void RegionTreeNode::send_back_logical_state(ContextID ctx, 
                                                 UniqueID context_uid,
                                                 AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      std::set<RegionTreeNode*> to_traverse;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(context_uid);
        if (is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(as_partition_node()->handle);
        }
        rez.serialize(state.reduction_fields);
        rez.serialize<size_t>(state.outstanding_reductions.size());
        for (LegionMap<ReductionOpID,FieldMask>::const_iterator it = 
              state.outstanding_reductions.begin(); it != 
              state.outstanding_reductions.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        rez.serialize<size_t>(state.projection_epochs.size());
        for (std::list<ProjectionEpoch*>::const_iterator pit = 
              state.projection_epochs.begin(); pit != 
              state.projection_epochs.end(); pit++)
        {
          rez.serialize((*pit)->epoch_id);
          rez.serialize((*pit)->valid_fields);
          rez.serialize<size_t>((*pit)->write_projections.size());
          for (std::map<ProjectionFunction*,std::set<IndexSpaceNode*> >::
                const_iterator fit = (*pit)->write_projections.begin(); 
                fit != (*pit)->write_projections.end(); fit++)
          {
            rez.serialize(fit->first->projection_id);
            rez.serialize<size_t>(fit->second.size());
            for (std::set<IndexSpaceNode*>::const_iterator it = 
                  fit->second.begin(); it != fit->second.end(); it++)
              rez.serialize((*it)->handle);
          }
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
#ifdef DEBUG_LEGION
            assert(fit->projection != NULL);
#endif
            rez.serialize(fit->projection->projection_id);
            rez.serialize(fit->projection_space->handle);
          }
#ifdef DEBUG_LEGION
          else
            assert(fit->projection == NULL);
#endif
          rez.serialize<size_t>(fit->open_children.size());
          for (FieldMaskSet<RegionTreeNode>::const_iterator it =
                fit->open_children.begin(); it != 
                fit->open_children.end(); it++)
          {
            // Add a remote valid reference on these nodes to keep
            // them live until we can add on remotely. No need for 
            // a mutator since we know that they are already valid
            it->first->add_base_valid_ref(REMOTE_DID_REF); 
            rez.serialize(it->first->get_color());
            rez.serialize(it->second);
            to_traverse.insert(it->first);
          }
        }
      }
      context->runtime->send_back_logical_state(target, rez);
      // Now recurse down the tree
      for (std::set<RegionTreeNode*>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        (*it)->send_back_logical_state(ctx, context_uid, target);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::process_logical_state_return(ContextID ctx,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      derez.deserialize(state.reduction_fields);
      size_t num_reductions;
      derez.deserialize(num_reductions);
      for (unsigned idx = 0; idx < num_reductions; idx++)
      {
        ReductionOpID redop;
        derez.deserialize(redop);
        derez.deserialize(state.outstanding_reductions[redop]);
      }
      size_t num_projection_epochs;
      derez.deserialize(num_projection_epochs);
      for (unsigned idx1 = 0; idx1 < num_projection_epochs; idx1++)
      {
        ProjectionEpochID epoch_id;
        derez.deserialize(epoch_id);
        FieldMask valid_fields;
        derez.deserialize(valid_fields);
        ProjectionEpoch *epoch = new ProjectionEpoch(epoch_id, valid_fields);
        size_t num_projections;
        derez.deserialize(num_projections);
        for (unsigned idx2 = 0; idx2 < num_projections; idx2++)
        {
          ProjectionID proj_id;
          derez.deserialize(proj_id);
          ProjectionFunction *function = 
            context->runtime->find_projection_function(proj_id);
          std::set<IndexSpaceNode*> &spaces = 
            epoch->write_projections[function];
          size_t num_doms;
          derez.deserialize(num_doms);
          for (unsigned idx3 = 0; idx3 < num_doms; idx3++)
          {
            IndexSpace handle;
            derez.deserialize(handle);
            spaces.insert(context->get_node(handle));
          }
        }
      }
      size_t num_field_states;
      derez.deserialize(num_field_states);
      state.field_states.resize(num_field_states);
      for (LegionList<FieldState>::iterator fit = 
            state.field_states.begin(); fit != state.field_states.end(); fit++)
      {
        FieldMask valid_fields;
        derez.deserialize(valid_fields);
        fit->open_children.relax_valid_mask(valid_fields);
        derez.deserialize(fit->open_state);
        derez.deserialize(fit->redop);
        if (fit->open_state >= OPEN_READ_ONLY_PROJ)
        {
          ProjectionID proj_id;
          derez.deserialize(proj_id);
          fit->projection = context->runtime->find_projection_function(proj_id);
          IndexSpace handle;
          derez.deserialize(handle);
          fit->projection_space = context->get_node(handle);
        }
        size_t num_open_children;
        derez.deserialize(num_open_children);
        std::set<RtEvent> applied_events;
        for (unsigned idx = 0; idx < num_open_children; idx++)
        {
          LegionColor color;
          derez.deserialize(color);
          RegionTreeNode *child = get_tree_child(color);
          FieldMask mask;
          derez.deserialize(mask);
          fit->add_child(child, mask, applied_events);
        }
        // Remove the valid references once all our current referenes are added
        RtEvent applied;
        if (!applied_events.empty())
          applied = Runtime::merge_events(applied_events);
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
              fit->open_children.begin(); it != fit->open_children.end(); it++) 
          it->first->send_remote_valid_decrement(source, NULL, applied);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::handle_logical_state_return(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      InnerContext *context = runtime->find_context(context_uid);
      InnerContext *outermost = context->find_outermost_local_context(); 
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *node = NULL;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        node = runtime->forest->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        node = runtime->forest->get_node(handle);
      }
      node->process_logical_state_return(outermost->get_context().get_id(),
                                         derez, source);
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
    bool RegionTreeNode::invalidate_version_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (!current_versions.has_entry(ctx))
        return false;
      VersionManager &manager = get_current_version_manager(ctx);
      manager.reset();
      return true;
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
    void RegionTreeNode::invalidate_version_managers(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned ctx = 0; ctx < current_versions.max_entries(); ctx++)
      {
        if (current_versions.has_entry(ctx))
          invalidate_version_state(ctx);
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
                           RtEvent init, RtEvent tree, Provenance *prov)
      : RegionTreeNode(ctx, col_src, init, tree, prov, id), handle(r),
        parent(par), row_source(row_src)
#ifdef DEBUG_LEGION
        , currently_valid(true)
#endif
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
    RegionNode::RegionNode(const RegionNode &rhs)
      : RegionTreeNode(NULL, NULL, RtEvent::NO_RT_EVENT, RtEvent::NO_RT_EVENT),
        handle(LogicalRegion::NO_REGION), parent(NULL), row_source(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
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
          if ((*it)->remove_partition_reference(NULL))
            delete (*it);
        partition_trackers.clear();
      }
      if (registered)
      {
        // Only need to unregister ourselves with the column if we're the top
        if (parent == NULL)
        {
          if (column_source->remove_nested_resource_ref(did))
            delete column_source;
        }
        // Unregister oursleves with the row source
        if (row_source->remove_nested_resource_ref(did))
          delete row_source;
        const bool top_level = (parent == NULL);
        // Unregister ourselves with the context
        context->remove_node(handle, top_level);
      }
    }

    //--------------------------------------------------------------------------
    RegionNode& RegionNode::operator=(const RegionNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void RegionNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        if (is_owner())
        {
#ifdef DEBUG_LEGION
          assert(currently_valid);
#endif
          // Add valid references on our index space and our field space
          column_source->add_nested_valid_ref(did, mutator);
          row_source->add_nested_valid_ref(did, mutator);
        }
        else
          send_remote_valid_increment(owner_space, mutator);     
      }
      else
      {
        column_source->add_nested_valid_ref(did, mutator);
        row_source->parent->add_nested_valid_ref(did, mutator);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::InvalidFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      node->send_remote_gc_decrement(target, mutator);
    }

    //--------------------------------------------------------------------------
    void RegionNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        if (is_owner())
        {
#ifdef DEBUG_LEGION
          assert(currently_valid);
          currently_valid = false;
#endif
          // Remove our valid references, no need to check for deletion
          // since we know that we are holding resource references too
          column_source->remove_nested_valid_ref(did, mutator);
          row_source->remove_nested_valid_ref(did, mutator);
          // Send deletion messages to each of our remote instances
          if (has_remote_instances())
          {
            InvalidFunctor functor(this, mutator);
            map_over_remote_instances(functor);
          }
        }
        else
          send_remote_valid_decrement(owner_space, mutator);
      }
      else
      {
        column_source->remove_nested_valid_ref(did, mutator);
        row_source->parent->remove_nested_valid_ref(did, mutator);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_active);
      currently_active = false;
#endif
      invalidate_version_managers();
      if (parent == NULL)
        context->runtime->release_tree_instances(handle.get_tree_id());
      if (!partition_trackers.empty())
      {
#ifdef DEBUG_LEGION
        assert(parent == NULL); // should only happen on the root
#endif
        for (std::list<PartitionTracker*>::const_iterator it = 
              partition_trackers.begin(); it != partition_trackers.end(); it++)
          if ((*it)->remove_partition_reference(mutator))
            delete (*it);
        partition_trackers.clear();
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!registered);
#endif
      if (parent != NULL)
        parent->add_child(this);
      else
        column_source->add_nested_resource_ref(did);
      row_source->add_nested_resource_ref(did);
      registered = true;
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
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(initialized);
              rez.serialize<size_t>(1);
              rez.serialize(it->first);
              rez.serialize(it->second.size);
              rez.serialize(it->second.buffer, it->second.size);
              rez.serialize(it->second.is_mutable);
            }
            context->runtime->send_logical_region_semantic_info(target, rez);
          }
        }
        else
        {
          // We've made it to the top, pack this node
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
      Provenance *prov = Provenance::deserialize(derez);

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
    RtEvent RegionNode::perform_versioning_analysis(ContextID ctx,
                                                    InnerContext *parent_ctx,
                                                    VersionInfo *version_info,
                                                    LogicalRegion upper_bound,
                                                    const FieldMask &mask,
                                                    Operation *op)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      // Make sure we traverse up the tree if necessary
      // This not strictly necessary for correctness, but is done
      // for performance in order to try to help out the analysis 
      // in the shattering code for equivalence sets that tries to
      // recognize disjoint partitions. If we ever switch to using
      // explicit shattering operations then we can remove this code
      if ((handle != upper_bound) && (!manager.has_versions(mask)))
      {
#ifdef DEBUG_LEGION
        assert(parent != NULL);
#endif
        FieldMask up_mask = mask - manager.get_version_mask();
        if (!!up_mask)
        {
          const RtEvent ready = 
            parent->parent->perform_versioning_analysis(ctx, parent_ctx,
                                NULL/*no version info*/, upper_bound, mask, op);
          if (ready.exists() && !ready.has_triggered())
            ready.wait();
        }
      }
      return manager.perform_versioning_analysis(parent_ctx, version_info, 
                                                 this, mask, op);
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
    bool PartitionTracker::remove_partition_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Pull a copy of this on to the stack in case we get deleted
      std::atomic<PartitionNode*> node(partition);
      const bool last = remove_reference();
      // If we weren't the last one that means we remove the reference
      if (!last && node.load()->remove_base_gc_ref(REGION_TREE_REF, mutator))
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
    void PartitionNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      column_source->add_nested_valid_ref(did, mutator);
      row_source->add_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // No need to check for deletion since we hold resource references 
      column_source->remove_nested_valid_ref(did, mutator);
      row_source->remove_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_active);
      currently_active = false;
#endif
      parent->remove_child(row_source->color);
      invalidate_version_managers();
      // Remove gc references on all of our child nodes
      // We should not need a lock at this point since nobody else should
      // be modifying these data structures at this point
      // No need to check for deletion since we hold resource references
      for (std::map<LegionColor,RegionNode*>::const_iterator it = 
            color_map.begin(); it != color_map.end(); it++)
        it->second->remove_nested_gc_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!registered);
#endif
      row_source->add_nested_resource_ref(did);
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
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
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
      return (handle.tree_id % runtime->runtime_stride);
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

