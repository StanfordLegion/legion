/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "legion/region_tree.inl"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
    /////////////////////////////////////////////////////////////

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
    void RegionTreeForest::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      // Remove valid references from any resources that haven't been deleted
      // First delete any region tree nodes that we own
      std::vector<RegionNode*> regions_to_delete;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
              tree_nodes.begin(); it != tree_nodes.end(); it++)
        {
          // Only care about it if it is one that we own
          if (RegionTreeNode::get_owner_space(it->first, runtime) != 
              runtime->address_space)
            continue;
          if (it->second->destroyed)
            continue;
          it->second->add_base_resource_ref(REGION_TREE_REF);
          regions_to_delete.push_back(it->second);
        }
      }
      for (std::vector<RegionNode*>::const_iterator it = 
            regions_to_delete.begin(); it != regions_to_delete.end(); it++)
      {
        if (!(*it)->destroyed)
          (*it)->destroy_node(runtime->address_space, true/*root*/);
        if ((*it)->remove_base_resource_ref(REGION_TREE_REF))
          delete (*it);
      }
      // Then do field space nodes
      std::vector<FieldSpaceNode*> fields_to_delete;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        for (std::map<FieldSpace,FieldSpaceNode*>::const_iterator it = 
              field_nodes.begin(); it != field_nodes.end(); it++)
        {
          // Only care about nodes that we own
          if (!it->second->is_owner())
            continue;
          // If we can actively delete it then it isn't valid
          if (it->second->destroyed)
            continue;
          it->second->add_base_resource_ref(REGION_TREE_REF);
          fields_to_delete.push_back(it->second);
        }
      }
      for (std::vector<FieldSpaceNode*>::const_iterator it =
            fields_to_delete.begin(); it != fields_to_delete.end(); it++)
      {
        if (!(*it)->destroyed)
          (*it)->destroy_node(runtime->address_space);
        if ((*it)->remove_base_resource_ref(REGION_TREE_REF))
          delete (*it);
      }
      // Then do index space nodes
      std::vector<IndexSpaceNode*> indexes_to_delete;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        for (std::map<IndexSpace,IndexSpaceNode*>::const_iterator it = 
              index_nodes.begin(); it != index_nodes.end(); it++)
        {
          // Only care about nodes at the top of the tree
          if (it->second->parent != NULL)
            continue;
          // Only care about nodes that we own
          if (!it->second->is_owner())
            continue;
          // If we can actively delete it then it isn't valid
          if (it->second->destroyed)
            continue;
          it->second->add_base_resource_ref(REGION_TREE_REF);
          indexes_to_delete.push_back(it->second);
        }
      }
      for (std::vector<IndexSpaceNode*>::const_iterator it = 
            indexes_to_delete.begin(); it != indexes_to_delete.end(); it++)
      {
        if (!(*it)->destroyed)
          (*it)->destroy_node(runtime->address_space);
        if ((*it)->remove_base_resource_ref(REGION_TREE_REF))
          delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_space(IndexSpace handle,
                                        const void *realm_is, DistributedID did)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = 
        create_node(handle, realm_is, NULL/*parent*/, 0/*color*/, did);
      if (runtime->legion_spy_enabled)
        node->log_index_space_points();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_union_space(IndexSpace handle, TaskOp *op,
                      const std::vector<IndexSpace> &handles, DistributedID did)
    //--------------------------------------------------------------------------
    {
      ApUserEvent to_trigger = Runtime::create_ap_user_event();
      IndexSpaceNode *node = 
        create_node(handle, NULL, NULL/*parent*/, 0/*color*/, did, to_trigger);
      node->initialize_union_space(to_trigger, op, handles);
      if (runtime->legion_spy_enabled)
      {
        if (!node->index_space_ready.has_triggered())
          node->index_space_ready.wait();
        node->log_index_space_points();
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_intersection_space(IndexSpace handle,
          TaskOp *op, const std::vector<IndexSpace> &handles, DistributedID did)
    //--------------------------------------------------------------------------
    {
      ApUserEvent to_trigger = Runtime::create_ap_user_event();
      IndexSpaceNode *node = 
        create_node(handle, NULL, NULL/*parent*/, 0/*color*/, did, to_trigger);
      node->initialize_intersection_space(to_trigger, op, handles);
      if (runtime->legion_spy_enabled)
      {
        if (!node->index_space_ready.has_triggered())
          node->index_space_ready.wait();
        node->log_index_space_points();
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_difference_space(IndexSpace handle,
               TaskOp *op, IndexSpace left, IndexSpace right, DistributedID did)
    //--------------------------------------------------------------------------
    {
      ApUserEvent to_trigger = Runtime::create_ap_user_event();
      IndexSpaceNode *node = 
        create_node(handle, NULL, NULL/*parent*/, 0/*color*/, did, to_trigger);
      node->initialize_difference_space(to_trigger, op, left, right);
      if (runtime->legion_spy_enabled)
      {
        if (!node->index_space_ready.has_triggered())
          node->index_space_ready.wait();
        node->log_index_space_points();
      }
    }

    //--------------------------------------------------------------------------
    RtEvent RegionTreeForest::create_pending_partition(IndexPartition pid,
                                                       IndexSpace parent,
                                                       IndexSpace color_space,
                                                    LegionColor partition_color,
                                                       PartitionKind part_kind,
                                                       DistributedID did,
                                                       ApEvent partition_ready,
                                                    ApUserEvent partial_pending)
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
      RtUserEvent disjointness_event;
      if ((part_kind == COMPUTE_KIND) || (part_kind == COMPUTE_COMPLETE_KIND) ||
          (part_kind == COMPUTE_INCOMPLETE_KIND))
      {
        disjointness_event = Runtime::create_rt_user_event();
        create_node(pid, parent_node, color_node, partition_color,
                    disjointness_event, did, partition_ready, partial_pending);
      }
      else
      {
        const bool disjoint = (part_kind == DISJOINT_KIND) || 
                              (part_kind == DISJOINT_COMPLETE_KIND) || 
                              (part_kind == DISJOINT_INCOMPLETE_KIND);
        create_node(pid, parent_node, color_node, partition_color,
                    disjoint, did, partition_ready, partial_pending);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_index_partition(parent.id, pid.id, disjoint,
                                         partition_color);
      }
      // If we need to compute the disjointness, only do that
      // after the partition is actually ready
      if ((part_kind == COMPUTE_KIND) || (part_kind == COMPUTE_COMPLETE_KIND) ||
          (part_kind == COMPUTE_INCOMPLETE_KIND))
      {
#ifdef DEBUG_LEGION
        assert(disjointness_event.exists());
#endif
        // Launch a task to compute the disjointness
        DisjointnessArgs args;
        args.handle = pid;
        args.ready = disjointness_event;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                                     Runtime::protect_event(partition_ready));
      }
      return parent_notified;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_pending_cross_product(IndexPartition handle1,
                                                        IndexPartition handle2,
                             std::map<IndexSpace,IndexPartition> &user_handles,
                                                        PartitionKind kind,
                                                        LegionColor &part_color,
                                                        ApEvent domain_ready)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *base = get_node(handle1);
      IndexPartNode *source = get_node(handle2);
      // If we're supposed to compute this, but we already know that
      // the source is disjoint then we can also conlclude the the 
      // resulting partitions will also be disjoint under intersection
      if (((kind == COMPUTE_KIND) || (kind == COMPUTE_COMPLETE_KIND) ||
           (kind == COMPUTE_INCOMPLETE_KIND)) && 
          source->is_disjoint(true/*from app*/))
      {
        if (kind == COMPUTE_KIND)
          kind = DISJOINT_KIND;
        else if (kind == COMPUTE_COMPLETE_KIND)
          kind = DISJOINT_COMPLETE_KIND;
        else
          kind = DISJOINT_INCOMPLETE_KIND;
      }
      // If we haven't been given a color yet, we need to find
      // one that will be valid for all the child partitions
      if (part_color == INVALID_COLOR)
      {
        std::set<LegionColor> existing_colors;
        if (base->total_children == base->max_linearized_color)
        {
          for (LegionColor color = 0; color < base->total_children; color++)
          {
            IndexSpaceNode *child_node = base->get_child(color);
            std::vector<LegionColor> colors;
            child_node->get_colors(colors);
            if (!colors.empty())
              existing_colors.insert(colors.begin(), colors.end());
          }
        }
        else
        {
          for (LegionColor color = 0; 
                color < base->max_linearized_color; color++)
          {
            if (!base->color_space->contains_color(color))
              continue;
            IndexSpaceNode *child_node = base->get_child(color);
            std::vector<LegionColor> colors;
            child_node->get_colors(colors);
            if (!colors.empty())
              existing_colors.insert(colors.begin(), colors.end());
          }
        }
        // Now we have the existing colors, so just do something dumb
        // like add our current runtime ID to the last color
        const unsigned offset = (runtime->address_space == 0) ?
                      runtime->runtime_stride : runtime->address_space;
        if (!existing_colors.empty())
        {
          LegionColor last = *(existing_colors.rbegin());
          part_color = last + offset;
        }
        else
          part_color = offset;
      }
      // Iterate over all our sub-regions and generate partitions
      if (base->total_children == base->max_linearized_color)
      {
        for (LegionColor color = 0; color < base->total_children; color++)
        {
          IndexSpaceNode *child_node = base->get_child(color);
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag()); 
          DistributedID did = 
            runtime->get_available_distributed_id();
          create_pending_partition(pid, child_node->handle, 
                                   source->color_space->handle, 
                                   part_color, kind, did, domain_ready);
          // If the user requested the handle for this point return it
          std::map<IndexSpace,IndexPartition>::iterator finder = 
            user_handles.find(child_node->handle);
          if (finder != user_handles.end())
            finder->second = pid;
        }
      }
      else
      {
        for (LegionColor color = 0; color < base->max_linearized_color; color++)
        {
          if (!base->color_space->contains_color(color))
            continue;
          IndexSpaceNode *child_node = base->get_child(color);
          IndexPartition pid(runtime->get_unique_index_partition_id(),
                             handle1.get_tree_id(), handle1.get_type_tag()); 
          DistributedID did = 
            runtime->get_available_distributed_id();
          create_pending_partition(pid, child_node->handle, 
                                   source->color_space->handle,
                                   part_color, kind, did, domain_ready);
          // If the user requested the handle for this point return it
          std::map<IndexSpace,IndexPartition>::iterator finder = 
            user_handles.find(child_node->handle);
          if (finder != user_handles.end())
            finder->second = pid;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::compute_partition_disjointness(IndexPartition handle,
                                                        RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      node->compute_disjointness(ready_event);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_space(IndexSpace handle,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      if (node->destroy_node(source))
        delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition handle,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      if (node->destroy_node(source))
        delete node;
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
        for (LegionColor color = 0; 
              color < base_node->max_linearized_color; color++)
        {
          if (!base_node->color_space->contains_color(color))
            continue;
          IndexSpaceNode *child_node = base_node->get_child(color);
          IndexPartNode *part_node = child_node->get_child(part_color);
          ApEvent ready = child_node->create_by_intersection(op, part_node, 
                                                             source_node); 
          ready_events.insert(ready);
        }
      }
      return Runtime::merge_events(ready_events);
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
    bool RegionTreeForest::check_partition_by_field_size(IndexPartition pid,
                FieldSpace fs, FieldID fid, bool is_range, bool use_color_space)
    //--------------------------------------------------------------------------
    {
      const size_t field_size = get_node(fs)->get_field_size(fid);
      IndexPartNode *partition = get_node(pid);
      if (use_color_space)
      {
#ifdef DEBUG_LEGION
        assert(!is_range);
#endif
        return partition->color_space->check_field_size(field_size, 
                                                        false/*range*/);
      }
      else
        return partition->parent->check_field_size(field_size, is_range);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::check_association_field_size(IndexSpace is,
                                                     FieldSpace fs, FieldID fid)
    //--------------------------------------------------------------------------
    {
      const size_t field_size = get_node(fs)->get_field_size(fid);
      return get_node(is)->check_field_size(field_size, false/*is range*/);
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
    void RegionTreeForest::create_field_space(FieldSpace handle,
                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      create_node(handle, did);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace handle,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      if (node->destroy_node(source))
        delete node;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::allocate_field(FieldSpace handle, size_t field_size,
                                          FieldID fid, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_field(fid, field_size, serdez_id);
      if (ready.exists())
        ready.wait();
      return false;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_field(FieldSpace handle, FieldID fid) 
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_field(fid, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::allocate_fields(FieldSpace handle, 
                                           const std::vector<size_t> &sizes,
                                           const std::vector<FieldID> &fields,
                                           CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sizes.size() == fields.size());
#endif
      // We know that none of these field allocations are local
      FieldSpaceNode *node = get_node(handle);
      RtEvent ready = node->allocate_fields(sizes, fields, serdez_id);
      // Wait for this to exist
      if (ready.exists())
        ready.wait();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_fields(FieldSpace handle,
                                       const std::vector<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_fields(to_free, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::allocate_local_fields(FieldSpace handle,
                                      const std::vector<FieldID> &fields,
                                      const std::vector<size_t> &sizes,
                                      CustomSerdezID serdez_id, 
                                      const std::set<unsigned> &current_indexes,
                                            std::vector<unsigned> &new_indexes)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      return node->allocate_local_fields(fields, sizes, serdez_id,
                                         current_indexes, new_indexes);
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
                                  const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->update_local_fields(fields, sizes, serdez_ids, indexes);
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
    void RegionTreeForest::get_all_regions(FieldSpace handle,
                                           std::set<LogicalRegion> &regions)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->get_all_regions(regions);
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
    void RegionTreeForest::get_field_space_fields(FieldSpace handle,
                                                  std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->get_all_fields(fields);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_logical_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      create_node(handle, NULL/*parent*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_logical_region(LogicalRegion handle,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        RegionNode::get_owner_space(handle, runtime);
      if (owner_space != runtime->address_space)
      {
        // If we're not the owner then we only have to do something if
        // we actually have a local copy of the node
        RegionNode *node = find_local_node(handle);
        if (node == NULL)
          return;
        // We're still holding a reference so don't bother checking
        // the return value
        node->destroy_node(source, true/*root*/);
        if (node->remove_base_resource_ref(REGION_TREE_REF))
          delete node;
      }
      else
      {
        // We're the owner so we know the reference still exists
        RegionNode *node = get_node(handle);
        if (node->destroy_node(source, true/*root*/))
          delete node;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_logical_partition(LogicalPartition handle,
                                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = 
        PartitionNode::get_owner_space(handle, runtime);
      if (owner_space != runtime->address_space)
      {
        // If we're not the owner then we only have to do something if
        // we actually have a local copy of the node
        PartitionNode *node = find_local_node(handle);
        if (node == NULL)
          return;
        // We're still holding a reference so don't bother checking
        // the return value
        node->destroy_node(source, true/*root*/);
        if (node->remove_base_resource_ref(REGION_TREE_REF))
          delete node;
      }
      else
      {
        // We're the owner so we know the reference still exists
        PartitionNode *node = get_node(handle);
        if (node->destroy_node(source, true/*root*/))
          delete node;
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
      LegionColor color = color_space->linearize_color(realm_color, type_tag);
      return parent_node->has_color(color);
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
                                                RestrictInfo &restrict_info,
                                                VersionInfo &version_info,
                                                ProjectionInfo &projection_info,
                                                RegionTreePath &path)
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
      version_info.set_upper_bound_node(parent_node); 
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      // Then compute the logical user
      LogicalUser user(op, idx, RegionUsage(req), user_mask); 
      TraceInfo trace_info(op->already_traced(), op->get_trace(), idx, req); 
      // Update the version info to set the maximum depth
      if (projection_info.is_projecting())
        version_info.resize(path.get_max_depth(), req.handle_type, 
                            projection_info.projection);
      else
        version_info.resize(path.get_max_depth());
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
        FieldMask unopened = user_mask;
        LegionMap<AdvanceOp*,LogicalUser>::aligned advances;
        parent_node->register_logical_user(ctx.get_id(), user, path, 
                                           trace_info, version_info,
                                           projection_info, unopened, advances);
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
      // Do our check for restricted coherence
      TaskContext *parent_ctx = op->get_context();
      if (parent_ctx->has_restrictions())
        parent_ctx->perform_restricted_analysis(req, restrict_info);
      // If we have a restriction, then record it on the region requirement
      if (restrict_info.has_restrictions())
        req.flags |= RESTRICTED_FLAG;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_deletion_analysis(DeletionOp *op, 
                                                    unsigned idx,
                                                    RegionRequirement &req,
                                                    RestrictInfo &restrict_info,
                                                    RegionTreePath &path)
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
      TraceInfo trace_info(op->already_traced(), op->get_trace(), idx, req);
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), true/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                       FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // We still need a version info to track in case we end up needing
      // to do any close operations
      VersionInfo version_info;
      version_info.set_upper_bound_node(parent_node);
      version_info.resize(path.get_max_depth());
      // Do the traversal
      parent_node->register_logical_deletion(ctx.get_id(), user, user_mask, 
                                             path, restrict_info, 
                                             version_info, trace_info);
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
      assert(req.handle_type == SINGULAR);
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
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *top_node = get_node(req.region);
      top_node->send_back_logical_state(ctx.get_id(), context_uid, target);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_versioning_analysis(Operation *op, 
                        unsigned idx, const RegionRequirement &req,
                        const RegionTreePath &path, VersionInfo &version_info,
                        std::set<RtEvent> &ready_events, bool partial_traversal,
                        bool disjoint_close, FieldMask *filter_mask,
                        RegionTreeNode *parent_node, UniqueID logical_ctx_uid,
          const LegionMap<ProjectionEpochID,FieldMask>::aligned *advance_epochs,
                        bool skip_parent_check)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_VERSIONING_ANALYSIS_CALL);
      if (IS_NO_ACCESS(req))
        return;
      InnerContext *context = op->find_physical_context(idx);
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      RegionUsage usage(req); 
      if (parent_node == NULL)
        parent_node = get_node(req.parent);
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      // If we have a filter mask, then apply it
      if (filter_mask != NULL)
        user_mask &= *filter_mask;
      // If this is not the top of the tree, see if we have to get
      // any version infromation from our parent context (e.g. because
      // it virtually mapped above where we have privileges)
      const unsigned depth = parent_node->get_depth();
      if (!skip_parent_check && (depth > 0))
      {
        TaskContext *parent_ctx = op->get_context();
        unsigned parent_req_index = op->find_parent_index(idx);
        parent_ctx->find_parent_version_info(parent_req_index, depth, 
                                             user_mask, context, 
                                             version_info, ready_events);
      }
      // Keep track of any unversioned fields
      FieldMask unversioned_mask = user_mask;
      // Do the traversal to compute the version numbers
      parent_node->compute_version_numbers(ctx.get_id(), path, usage,
                                           user_mask, unversioned_mask,
                                           op, idx, context, version_info, 
                                           ready_events, partial_traversal,
                                           disjoint_close, logical_ctx_uid,
                                           advance_epochs);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::advance_version_numbers(Operation *op, unsigned idx,
                                                bool update_parent_state,
                                                bool parent_is_upper_bound,
                                                UniqueID logical_ctx_uid,
                                                bool dedup_opens, 
                                                bool dedup_advances, 
                                                ProjectionEpochID open_epoch, 
                                                ProjectionEpochID advance_epoch,
                                                RegionTreeNode *parent, 
                                                const RegionTreePath &path,
                                                const FieldMask &advance_mask, 
                  const LegionMap<unsigned,FieldMask>::aligned &dirty_previous,
                                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_ADVANCE_VERSION_NUMBERS_CALL);
      InnerContext *context = op->find_physical_context(idx);
      RegionTreeContext ctx = context->get_context(); 
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      parent->advance_version_numbers(ctx.get_id(), path, advance_mask,
                                      context, update_parent_state,
                                      parent_is_upper_bound, logical_ctx_uid,
                                      dedup_opens, dedup_advances, open_epoch, 
                                      advance_epoch, dirty_previous, 
                                      ready_events);
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
                  const RegionRequirement &req, const InstanceSet &sources,
                  ApEvent term_event, InnerContext *context,unsigned init_index,
                  std::map<PhysicalManager*,InstanceView*> &top_views,
                  std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_INITIALIZE_CONTEXT_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *top_node = get_node(req.region);
      RegionUsage usage(req);
      FieldMask user_mask = 
        top_node->column_source->get_field_mask(req.privilege_fields);
      // If we're writing, update the logical state with the dirty fields
      // Even mark reads as dirty since we'll assume they are initialized
      if (!IS_REDUCE(req))
        top_node->initialize_logical_state(ctx.get_id(), user_mask);
      // Now we update the physical state
      std::vector<LogicalView*> corresponding(sources.size());
      const AddressSpaceID local_space = context->runtime->address_space;
      // Build our set of corresponding views
      if (IS_REDUCE(req))
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const InstanceRef &src_ref = sources[idx];
          PhysicalManager *manager = src_ref.get_manager();
#ifdef DEBUG_LEGION
          assert(manager->is_reduction_manager());
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
          }
          else
            corresponding[idx] = finder->second;
        }
      }
      else
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const InstanceRef &src_ref = sources[idx];
          PhysicalManager *manager = src_ref.get_manager();
#ifdef DEBUG_LEGION
          assert(manager->is_instance_manager());
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
            // See if we need to get the appropriate subview
            if (top_node != manager->region_node)
              corresponding[idx] = 
                top_node->convert_reference_region(manager, context);
            else
              corresponding[idx] = new_view;
          }
          else
          {
            // See if we have to pull down the right subview
            if (top_node != manager->region_node)
              corresponding[idx] = 
                top_node->convert_reference_region(manager, context);
            else // they are the same so we can just use the view as is
              corresponding[idx] = finder->second;
          }
        }
      }
      // Now we can register all these instances
      top_node->seed_state(ctx.get_id(), term_event, usage, user_mask, 
                 sources, context, init_index, corresponding, applied_events); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_virtual_context(RegionTreeContext ctx,
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_INITIALIZE_CONTEXT_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
      assert(!IS_REDUCE(req));
#endif
      RegionNode *top_node = get_node(req.region);
      RegionUsage usage(req);
      FieldMask user_mask = 
        top_node->column_source->get_field_mask(req.privilege_fields);
      // Mark that it has these fields dirty so they are included
      // in all the analysis
      top_node->initialize_logical_state(ctx.get_id(), user_mask);
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
      assert(req1.handle_type == SINGULAR);
      assert(req2.handle_type == SINGULAR);
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
        PhysicalManager *manager = ref.get_manager();
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
    Restriction* RegionTreeForest::create_coherence_restriction(
                     const RegionRequirement &req, const InstanceSet &instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *node = get_node(req.region);  
      Restriction *result = new Restriction(node);    
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const InstanceRef &ref = instances[idx];
        PhysicalManager *manager = ref.get_manager();
        result->add_restricted_instance(manager, ref.get_valid_fields());
      }
      return result;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::add_acquisition(
                                    const std::list<Restriction*> &restrictions,
                                    AcquireOp *op, const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *node = get_node(req.region);
      FieldMask acquired_fields = 
        node->column_source->get_field_mask(req.privilege_fields);
      const RegionTreeID tid = req.region.get_tree_id();
      for (std::list<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        // If the tree IDs are different, we don't even bother
        if ((*it)->tree_id != tid)
          continue;
        (*it)->add_acquisition(op, node, acquired_fields);
        if (!acquired_fields)
          return true;
      }
      // Return true if don't have any more fields to acquired
      return !acquired_fields;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::remove_acquisition(
                                    const std::list<Restriction*> &restrictions,
                                    ReleaseOp *op, const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *node = get_node(req.region);
      FieldMask released_fields = 
        node->column_source->get_field_mask(req.privilege_fields);
      const RegionTreeID tid = req.region.get_tree_id();
      for (std::list<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        // If the tree IDs are different, we don't even bother
        if ((*it)->tree_id != tid)
          continue;
        (*it)->remove_acquisition(op, node, released_fields);
        if (!released_fields)
          return true;
      }
      // Return true if we don't have any more fields to release
      return !released_fields;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::add_restriction(
                            std::list<Restriction*> &restrictions, AttachOp *op,
                            InstanceManager *inst, const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *node = get_node(req.region);
      FieldMask restricted_fields = 
        node->column_source->get_field_mask(req.privilege_fields);
      const RegionTreeID tid = req.region.get_tree_id();
      for (std::list<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        // If the tree IDs are different, we don't even bother 
        if ((*it)->tree_id != tid)
          continue;
        (*it)->add_restriction(op, node, inst, restricted_fields);
        if (!restricted_fields)
          return;
      }
      // If we still had any restricted fields we can make a new restriction
      if (!!restricted_fields)
      {
        Restriction *restriction = new Restriction(node);
        restriction->add_restricted_instance(inst, restricted_fields);
        restrictions.push_back(restriction);
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::remove_restriction(
                                     std::list<Restriction*> &restrictions, 
                                     DetachOp *op, const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *node = get_node(req.region);
      FieldMask released_fields = 
        node->column_source->get_field_mask(req.privilege_fields);
      const RegionTreeID tid = req.region.get_tree_id();
      for (std::list<Restriction*>::iterator it = restrictions.begin();
            it != restrictions.end(); /*nothing*/)
      {
        // If the tree IDs are different, we don't even bother
        if ((*it)->tree_id != tid)
        {
          it++;
          continue;
        }
        // Try ot match it first
        if ((*it)->matches(op, node, released_fields))
        {
          it = restrictions.erase(it);
          if (!released_fields)
            return true;
          continue;
        }
        // Otherwise try to remove it internally
        (*it)->remove_restriction(op, node, released_fields);
        if (!released_fields)
          return true;
        it++;
      }
      return !released_fields;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_restricted_analysis(
                      const std::list<Restriction*> &restrictions,
                      const RegionRequirement &req, RestrictInfo &restrict_info)
    //--------------------------------------------------------------------------
    {
      // Don't compute this until we need to
      RegionTreeNode *node = NULL;
      FieldMask possibly_restricted;
      const RegionTreeID tid = req.parent.get_tree_id();
      for (std::list<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        // Skip the same tree ID
        if ((*it)->tree_id != tid)
          continue;
        // If we get here we have to do the analysis, if we haven't
        // computed our node and fields yet, do that now
        if (node == NULL)
        {
          if (req.handle_type == PART_PROJECTION)
          {
            PartitionNode *part_node = get_node(req.partition);
            possibly_restricted = 
              part_node->column_source->get_field_mask(req.privilege_fields);
            node = part_node;
          }
          else
          {
            RegionNode *reg_node = get_node(req.region);
            possibly_restricted = 
              reg_node->column_source->get_field_mask(req.privilege_fields);
            node = reg_node;
          }
        }
        // Now we can do the analysis
        (*it)->find_restrictions(node, possibly_restricted, restrict_info);
        // See if we are done early
        if (!possibly_restricted)
          break;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::physical_premap_only(Operation *op, unsigned index,
                                                const RegionRequirement &req,
                                                VersionInfo &version_info,
                                                InstanceSet &valid_instances)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PREMAP_ONLY_CALL);
      // If we are a NO_ACCESS or there are no fields then we are already done 
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return;
      TaskContext *context = op->find_physical_context(index);
      RegionTreeContext ctx = context->get_context();
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      RegionNode *region_node = (req.handle_type == PART_PROJECTION) ? 
        get_node(req.partition)->parent : get_node(req.region);
      FieldMask user_mask = 
        region_node->column_source->get_field_mask(req.privilege_fields);
      region_node->premap_region(ctx.get_id(), req, user_mask, 
                                 version_info, valid_instances);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::physical_register_only(const RegionRequirement &req,
                                               VersionInfo &version_info,
                                               RestrictInfo &restrict_info,
                                               Operation *op, unsigned index,
                                               ApEvent term_event,
                                               bool defer_add_users,
                                               bool need_read_only_reservations,
                                               std::set<RtEvent> &map_applied,
                                               InstanceSet &targets,
                                               const ProjectionInfo *proj_info
#ifdef DEBUG_LEGION
                                               , const char *log_name
                                               , UniqueID uid
#endif
                                               )
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_REGISTER_ONLY_CALL);
      // If we are a NO_ACCESS or there are no fields then we are already done 
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return;
      InnerContext *context = op->find_physical_context(index);
      RegionTreeContext ctx = context->get_context();
#ifdef DEBUG_LEGION
      assert(ctx.exists());
      assert((req.handle_type == SINGULAR) || 
              (req.handle_type == REG_PROJECTION));
      assert(!targets.empty());
      assert(!targets.is_virtual_mapping());
#endif
      RegionNode *child_node = get_node(req.region);
      FieldMask user_mask = 
        child_node->column_source->get_field_mask(req.privilege_fields);
      const UniqueID logical_ctx_uid = op->get_context()->get_context_uid();
      
      RegionUsage usage(req);
#ifdef DEBUG_LEGION 
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     child_node, ctx.get_id(), 
                                     true/*before*/, false/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                     FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Perform the registration
      // See if we need to acquire locks for read-only privileges
      if (need_read_only_reservations && IS_READ_ONLY(req))
      {
        // This is the price of allowing read-only requirements from
        // different operations to be mapped in parallel, we have to 
        // serialize access to the requirements mapping to the same
        // instance otherwise we can get weird effects with partial
        // applications of reductions in composite instances, etc.
        //
        // Note that by using a set we deduplicate and put the
        // locks in the order to be taken to avoid deadlocks
        std::set<Reservation> read_only_locks;
        targets.find_read_only_reservations(read_only_locks);
        RtEvent precondition;
        for (std::set<Reservation>::const_iterator it = 
              read_only_locks.begin(); it != read_only_locks.end(); it++)
        {
          RtEvent next = Runtime::acquire_rt_reservation(*it,
                            true/*exclusive*/, precondition);
          precondition = next;
        }
        // Have to wait for all the locks to be acquired
        if (precondition.exists())
          precondition.wait();
        // Now do the mapping with our own applied event set
        std::set<RtEvent> local_applied;
        // Construct the traversal info
        TraversalInfo info(ctx.get_id(), op, index, req, version_info, 
                           user_mask, local_applied);
        child_node->register_region(info, logical_ctx_uid, context, 
                                    restrict_info, term_event, usage, 
                                    defer_add_users, targets, proj_info);
        
        if (!local_applied.empty())
        {
          // Release the read only locks once our map applied 
          // conditions have been met
          RtEvent done_event = Runtime::merge_events(local_applied);
          for (std::set<Reservation>::const_iterator it = 
                read_only_locks.begin(); it != read_only_locks.end(); it++)
            it->release(done_event);
          // Propagate the condition back to the map applied
          map_applied.insert(done_event);
        }
        else
        {
          // Release the locks with no preconditions
          for (std::set<Reservation>::const_iterator it = 
                read_only_locks.begin(); it != read_only_locks.end(); it++)
            it->release();
        }
      }
      else // The common case
      {
        // Construct the traversal info
        TraversalInfo info(ctx.get_id(), op, index, req, version_info, 
                           user_mask, map_applied);
        child_node->register_region(info, logical_ctx_uid, context, 
                                    restrict_info, term_event, usage, 
                                    defer_add_users, targets, proj_info);
      }
#ifdef DEBUG_LEGION 
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     child_node, ctx.get_id(), 
                                     false/*before*/, false/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                   FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::physical_register_users(
                                  Operation *op, ApEvent term_event,
                                  const std::vector<RegionRequirement> &regions,
                                  const std::vector<bool> &to_skip,
                                  std::vector<VersionInfo> &version_infos,
                                  std::vector<RestrictInfo> &restrict_infos,
                                  std::deque<InstanceSet> &target_sets,
                                  std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_REGISTER_USERS_CALL);
#ifdef DEBUG_LEGION
      assert(regions.size() == to_skip.size());
      assert(regions.size() == version_infos.size());
      assert(regions.size() == target_sets.size());
#endif
      std::vector<std::vector<InstanceView*> > views(regions.size());
      // Do the precondition pass
      std::vector<InnerContext*> enclosing_contexts(regions.size());
      for (unsigned idx1 = 0; idx1 < regions.size(); idx1++)
      {
        if (to_skip[idx1])
          continue;
        if (IS_NO_ACCESS(regions[idx1]) || 
            regions[idx1].privilege_fields.empty())
          continue;
        const RegionRequirement &req = regions[idx1];
        VersionInfo &info = version_infos[idx1];
        InstanceSet &targets = target_sets[idx1];
#ifdef DEBUG_LEGION
        assert(req.handle_type == SINGULAR);
        assert(!targets.empty());
#endif
        InnerContext *context = op->find_physical_context(idx1);
        // Save this for later in case we need it
        enclosing_contexts[idx1] = context;
        RegionNode *region_node = get_node(req.region);
        std::vector<InstanceView*> &target_views = views[idx1];
        target_views.resize(targets.size());
        region_node->convert_target_views(targets, context, target_views);
        RegionUsage usage(req);
        for (unsigned idx2 = 0; idx2 < targets.size(); idx2++)
        {
          InstanceRef &ref = targets[idx2];
#ifdef DEBUG_LEGION
          assert(!ref.is_virtual_ref());
#endif
          ApEvent ready = target_views[idx2]->find_user_precondition(usage, 
                            term_event, ref.get_valid_fields(), op, idx1, 
                            &info, map_applied_events);
          ref.set_ready_event(ready);
        }
      }
      // Then do the registration pass
      for (unsigned idx1 = 0; idx1 < regions.size(); idx1++)
      {
        if (to_skip[idx1])
          continue;
        const RegionRequirement &req = regions[idx1];
        VersionInfo &info = version_infos[idx1];
        InstanceSet &targets = target_sets[idx1];
        std::vector<InstanceView*> &target_views = views[idx1];
        RegionUsage usage(req);
        // Information for dealing with restrictions
        FieldMask restricted_fields;
        LegionMap<LogicalView*,FieldMask>::aligned copy_out_views;
        LegionMap<ReductionView*,FieldMask>::aligned reduce_out_views;
        if (restrict_infos[idx1].has_restrictions())
          restrict_infos[idx1].populate_restrict_fields(restricted_fields);
        const bool restricted_out = !!restricted_fields && !IS_READ_ONLY(req);
        for (unsigned idx2 = 0; idx2 < targets.size(); idx2++)
        {
          InstanceRef &ref = targets[idx2];
          target_views[idx2]->add_user(usage, term_event, 
                                       ref.get_valid_fields(), 
                                       op, idx1, runtime->address_space,
                                       &info, map_applied_events);
          if (restricted_out)
          {
            FieldMask restricted = ref.get_valid_fields() & restricted_fields;
            if (!!restricted)
            {
              if (target_views[idx2]->is_reduction_view())
              {
                ReductionView *reduction_view = 
                  target_views[idx2]->as_reduction_view();
                reduce_out_views[reduction_view] = restricted;
              }
              else
                copy_out_views[target_views[idx2]] = restricted;
            }
          }
        }
        // Handle any restricted copy out operations we need to do
        if (!copy_out_views.empty())
        {
          RegionNode *node = get_node(req.region);
          RegionTreeContext ctx = enclosing_contexts[idx1]->get_context();
#ifdef DEBUG_LEGION
          assert(ctx.exists());
#endif
          TraversalInfo traversal_info(ctx.get_id(), op, idx1, req, info, 
                                  restricted_fields, map_applied_events);
          const InstanceSet &restricted_instances = 
            restrict_infos[idx1].get_instances();
          std::vector<MaterializedView*> restricted_views(
                                             restricted_instances.size());
          node->convert_target_views(restricted_instances,
              enclosing_contexts[idx1], restricted_views);
          node->issue_restricted_copies(traversal_info, restrict_infos[idx1],
              restricted_instances, restricted_views, copy_out_views);
        }
        if (!reduce_out_views.empty())
        {
          RegionNode *node = get_node(req.region);
          RegionTreeContext ctx = enclosing_contexts[idx1]->get_context();
#ifdef DEBUG_LEGION
          assert(ctx.exists());
#endif
          TraversalInfo traversal_info(ctx.get_id(), op, idx1, req, info, 
                                  restricted_fields, map_applied_events);
          const InstanceSet &restricted_instances = 
            restrict_infos[idx1].get_instances();
          std::vector<InstanceView*> restricted_views(
                                             restricted_instances.size());
          node->convert_target_views(restricted_instances,
              enclosing_contexts[idx1], restricted_views);
          node->issue_restricted_reductions(traversal_info, 
              restrict_infos[idx1], restricted_instances, 
              restricted_views, reduce_out_views);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::physical_perform_close(const RegionRequirement &req,
                VersionInfo &version_info, Operation *op, unsigned index, 
                ClosedNode *closed_tree, RegionTreeNode *close_node,
                const FieldMask &closing_mask, std::set<RtEvent> &map_applied, 
                const RestrictInfo &restrict_info, const InstanceSet &targets,
                const ProjectionInfo *projection_info
#ifdef DEBUG_LEGION
                , const char *log_name
                , UniqueID uid
#endif
                )
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_PERFORM_CLOSE_CALL);
      InnerContext *context = op->find_physical_context(index);
      RegionTreeContext ctx = context->get_context();
#ifdef DEBUG_LEGION
      assert(ctx.exists());
#endif
      TraversalInfo info(ctx.get_id(), op, index, req, 
                         version_info, closing_mask, map_applied);
      const UniqueID logical_ctx_uid = op->get_context()->get_context_uid();
      // Always build the composite instance, and then optionally issue
      // update copies to the target instances from the composite instance
      CompositeView *result = close_node->create_composite_instance(info.ctx, 
                          closing_mask, version_info, logical_ctx_uid, 
                          context, closed_tree, map_applied, projection_info);
      if (targets.empty())
        return;
      PhysicalState *physical_state = NULL;
      LegionMap<LogicalView*,FieldMask>::aligned valid_instances;
      bool need_valid = true;
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef &target_ref = targets[idx];
        if (!target_ref.has_ref() || target_ref.is_virtual_ref())
          continue;
        // Issue the update copies to this instance
        if (need_valid)
        {
          valid_instances[result] = closing_mask;
          physical_state = close_node->get_physical_state(version_info);
          need_valid = false;
        }
#ifdef DEBUG_LEGION
        assert(target_ref.has_ref());
#endif
        const FieldMask &target_mask = target_ref.get_valid_fields();
        InstanceView *target_view = close_node->convert_reference(target_ref, 
                                                                  context); 
#ifdef DEBUG_LEGION
        assert(target_view->is_materialized_view());
#endif
        close_node->issue_update_copies(info, 
            target_view->as_materialized_view(), target_mask, 
            valid_instances, restrict_info);
        // Update the valid views for the node
#ifdef DEBUG_LEGION
        assert(physical_state != NULL);
#endif
        close_node->update_valid_views(physical_state, target_mask,
                                       false/*dirty*/, target_view);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::physical_disjoint_close(InterCloseOp *op, 
                       unsigned index, RegionTreeNode *close_node, 
                       const FieldMask &closing_mask, VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
      InnerContext *context = op->find_physical_context(index);
#ifdef DEBUG_LEGION
      assert(!close_node->is_region());
#endif
      close_node->as_partition_node()->perform_disjoint_close(op, index, 
                                    context, closing_mask, version_info);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::physical_close_context(RegionTreeContext ctx,
                                                 const RegionRequirement &req,
                                                 VersionInfo &version_info,
                                                 Operation *op,unsigned index,
                                                 std::set<RtEvent> &map_applied,
                                                 InstanceSet &targets
#ifdef DEBUG_LEGION
                                                 , const char *log_name
                                                 , UniqueID uid
#endif
                                                 )
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_CLOSE_CONTEXT_CALL);
#ifdef DEBUG_LEGION
      assert(!targets.empty());
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *top_node = get_node(req.region);
      FieldMask user_mask = 
        top_node->column_source->get_field_mask(req.privilege_fields);
      RegionUsage usage(req);
      TraversalInfo info(ctx.get_id(), op, index, req, version_info, 
                         user_mask, map_applied);
      const UniqueID logical_ctx_uid = op->get_context()->get_context_uid();
      InnerContext *context = op->find_physical_context(index);
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     top_node, ctx.get_id(), 
                                     true/*before*/, false/*premap*/, 
                                     true/*closing*/, false/*logical*/,
                 FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      top_node->close_state(info, usage, logical_ctx_uid, context, targets);
#ifdef DEBUG_LEGION
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     top_node, ctx.get_id(), 
                                     false/*before*/, false/*premap*/, 
                                     true/*closing*/, false/*logical*/,
                 FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES), user_mask);
#endif
      // Look through all the targets and get their ready events
      // and then merge them together to indicate that the context is closed
      std::set<ApEvent> closed_events;
      for (unsigned idx = 0; idx < targets.size(); idx++)
        closed_events.insert(targets[idx].get_ready_event());
      if (closed_events.size() == 1)
        return *(closed_events.begin());
      return Runtime::merge_events(closed_events);
    }


    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::copy_across(
                                        const RegionRequirement &src_req,
                                        const RegionRequirement &dst_req,
                                              InstanceSet &src_targets, 
                                        const InstanceSet &dst_targets,
                                        VersionInfo &src_version_info,
                                        VersionInfo &dst_version_info, 
                                        ApEvent term_event, Operation *op, 
                                        unsigned src_index, unsigned dst_index,
                                        ApEvent precondition, PredEvent guard, 
                                        std::set<RtEvent> &map_applied)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_COPY_ACROSS_CALL);
#ifdef DEBUG_LEGION
      assert(src_req.handle_type == SINGULAR);
      assert(dst_req.handle_type == SINGULAR);
      assert(src_req.instance_fields.size() == dst_req.instance_fields.size());
#endif
      std::set<ApEvent> result_events;
      std::vector<unsigned> src_indexes(src_req.instance_fields.size());
      std::vector<unsigned> dst_indexes(dst_req.instance_fields.size());
      // Get the field indexes for all the fields
      RegionNode *src_node = get_node(src_req.region);
      RegionNode *dst_node = get_node(dst_req.region);
      src_node->column_source->get_field_indexes(src_req.instance_fields, 
                                                 src_indexes);   
      dst_node->column_source->get_field_indexes(dst_req.instance_fields,
                                                 dst_indexes);
      // Convert the destination references to views
      std::vector<MaterializedView*> dst_views(dst_targets.size());
      InnerContext *dst_context = op->find_physical_context(dst_index);
      for (unsigned idx = 0; idx < dst_targets.size(); idx++)
      {
        const InstanceRef &dst_ref = dst_targets[idx];
#ifdef DEBUG_LEGION
        assert(!dst_ref.is_virtual_ref());
#endif
        InstanceView *view = 
              dst_node->convert_reference(dst_ref, dst_context);
#ifdef DEBUG_LEGION
        assert(view->is_materialized_view());
#endif
        dst_views[idx] = view->as_materialized_view();
      }

      // If we need to, grab the valid views for the source
      // and issue copies to any of the deferred views
      if (src_targets.empty())
      {
        InnerContext *src_context = op->find_physical_context(src_index);
        RegionTreeContext ctx = src_context->get_context();
#ifdef DEBUG_LEGION
        assert(ctx.exists());
#endif
        FieldMask src_mask = src_node->column_source->get_field_mask(
                                            src_req.privilege_fields);
        LegionMap<LogicalView*,FieldMask>::aligned src_views;
        PhysicalState *state = src_node->get_physical_state(src_version_info);
        src_node->find_valid_instance_views(ctx.get_id(), state, src_mask,
            src_mask, src_version_info, false/*needs space*/, src_views);
        // Sort them into deferred and materialized views
        std::vector<DeferredView*> deferred_views;
        std::vector<MaterializedView*> materialized_views;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              src_views.begin(); it != src_views.end(); it++)
        {
          if (!it->first->is_materialized_view())
          {
#ifdef DEBUG_LEGION
            assert(it->first->is_deferred_view());
#endif
            deferred_views.push_back(it->first->as_deferred_view()); 
          }
          else
            materialized_views.push_back(it->first->as_materialized_view());
        }
        if (!deferred_views.empty())
        {
          // For each destination, find which composite views satisfy its fields
          for (unsigned idx1 = 0; idx1 < dst_targets.size(); idx1++)
          {
            const InstanceRef &dst_ref = dst_targets[idx1];
#ifdef DEBUG_LEGION
            assert(!dst_ref.is_virtual_ref());
#endif
            const FieldMask &dst_mask = dst_ref.get_valid_fields(); 
            std::map<DeferredView*,std::vector<unsigned> > src_deferred_indexes;
            std::map<DeferredView*,std::vector<unsigned> > dst_deferred_indexes;
            for (unsigned idx2 = 0; idx2 < dst_indexes.size(); idx2++)
            {
              const unsigned dst_field_index = dst_indexes[idx2];
              // If it's not set for the destination, then keep going 
              if (!dst_mask.is_set(dst_field_index))
                continue;
              const unsigned src_field_index = src_indexes[idx2];
              for (std::vector<DeferredView*>::const_iterator it = 
                    deferred_views.begin(); it != deferred_views.end(); it++)
              {
                FieldMask &src_mask = src_views[*it];
                if (!src_mask)
                  continue; 
                if (src_mask.is_set(src_field_index))
                {
                  src_deferred_indexes[*it].push_back(src_field_index);
                  dst_deferred_indexes[*it].push_back(dst_field_index);
                  src_mask.unset_bit(src_field_index);
                  if (!src_mask)
                    break;
                }
              }
            }
            if (src_deferred_indexes.empty())
              continue;
            // Now we've got the information necessary to do our copies
            MaterializedView *dst_view = dst_views[idx1];
            ApEvent precondition = dst_ref.get_ready_event();
            for (std::map<DeferredView*,std::vector<unsigned> >::const_iterator
                  src_it = src_deferred_indexes.begin(); 
                  src_it != src_deferred_indexes.end(); src_it++)
            {
              std::map<DeferredView*,std::vector<unsigned> >::const_iterator
                dst_it = dst_deferred_indexes.find(src_it->first);
#ifdef DEBUG_LEGION
              assert(dst_it != dst_deferred_indexes.end());
              assert(src_it->second.size() == dst_it->second.size());
#endif
              FieldMask copy_mask;
              for (std::vector<unsigned>::const_iterator it = 
                    src_it->second.begin(); it != src_it->second.end(); it++)
                copy_mask.set_bit(*it);
              TraversalInfo info(ctx.get_id(), op, src_index, src_req,
                                 dst_version_info, copy_mask, map_applied);
              src_it->first->issue_deferred_copies_across(info, dst_view,
                            src_it->second, dst_it->second, precondition, 
                            guard, result_events);
              src_mask -= copy_mask;
            }
            // If we've handled all the sources, we are done
            if (!src_mask)
              break;
          }
        }
        // Save any materialized views for the remaining fields
        // in the src_targets and then issue copies like normal
        if (!!src_mask && !materialized_views.empty())
        {
          RegionUsage src_usage(src_req);
          for (std::vector<MaterializedView*>::const_iterator it = 
                materialized_views.begin(); it != 
                materialized_views.end(); it++)
          {
            FieldMask valid_mask = src_mask & src_views[*it]; 
            if (!valid_mask)
              continue;
            // Need to add ourselves as a user of the instance
            // and get the event precondition for our use
            ApEvent ready = (*it)->add_user_fused(src_usage, term_event,
                                                  valid_mask, op, src_index,
                                                  &src_version_info,
                                                  runtime->address_space,
                                                  map_applied);
            src_targets.add_instance(
                InstanceRef((*it)->get_manager(), valid_mask, ready));
            src_mask -= valid_mask;
            if (!src_mask)
              break;
          }
        }
      }
      // If we have actual concrete instances to copy from now we can
      // issue those copies to the destination
      if (!src_targets.empty())
      {
        for (unsigned idx1 = 0; idx1 < dst_targets.size(); idx1++)
        {
          const InstanceRef &dst_ref = dst_targets[idx1];
#ifdef DEBUG_LEGION
          assert(!dst_ref.is_virtual_ref());
          assert(dst_ref.get_manager()->is_instance_manager());
#endif
          InstanceManager *dst_manager = 
            dst_ref.get_manager()->as_instance_manager();
          const FieldMask &dst_mask = dst_ref.get_valid_fields();
          std::map<unsigned,
                   std::vector<CopySrcDstField> > src_fields,dst_fields;
          for (unsigned idx2 = 0; idx2 < dst_indexes.size(); idx2++)
          {
            const unsigned dst_field_index = dst_indexes[idx2];
            // If it's not set for the destination, then keep going 
            if (!dst_mask.is_set(dst_field_index))
              continue;
            const unsigned src_field_index = src_indexes[idx2];
            for (unsigned idx3 = 0; idx3 < src_targets.size(); idx3++)
            {
              const InstanceRef &src_ref = src_targets[idx3];
#ifdef DEBUG_LEGION
              assert(!src_ref.is_virtual_ref());
              assert(src_ref.get_manager()->is_instance_manager());
#endif
              const FieldMask &src_mask = src_ref.get_valid_fields();   
              if (!src_mask.is_set(src_field_index))
                continue;
              InstanceManager *src_manager = 
                src_ref.get_manager()->as_instance_manager();
              src_manager->compute_copy_offsets(
                  src_req.instance_fields[idx2], src_fields[idx3]);
              dst_manager->compute_copy_offsets(
                  dst_req.instance_fields[idx2], dst_fields[idx3]);
            }
          }
          if (src_fields.empty())
            continue;
          ApEvent dst_precondition = dst_ref.get_ready_event(); 
          for (std::map<unsigned,std::vector<CopySrcDstField> >::
                const_iterator src_it = src_fields.begin(); 
                src_it != src_fields.end(); src_it++)
          {
            std::map<unsigned,std::vector<CopySrcDstField> >::
              const_iterator dst_it = dst_fields.find(src_it->first);
#ifdef DEBUG_LEGION
            assert(dst_it != dst_fields.end());
            assert(src_it->second.size() == dst_it->second.size());
#endif
            ApEvent src_precondition = 
              src_targets[src_it->first].get_ready_event();
            ApEvent copy_pre = Runtime::merge_events(src_precondition,
                                                     dst_precondition,
                                                     precondition);
            ApEvent copy_post = dst_node->issue_copy(op, src_it->second,
                                       dst_it->second, copy_pre, guard);
            if (copy_post.exists())
              result_events.insert(copy_post);
          }
        }
      }
      // Return the merge of all the result events
      return Runtime::merge_events(result_events);
    }
    
    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::reduce_across(const RegionRequirement &src_req,
                                            const RegionRequirement &dst_req,
                                            const InstanceSet &src_targets,
                                            const InstanceSet &dst_targets,
                                            Operation *op, ApEvent precondition,
                                            PredEvent predicate_guard)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_REDUCE_ACROSS_CALL);
#ifdef DEBUG_LEGION
      assert(src_req.handle_type == SINGULAR);
      assert(dst_req.handle_type == SINGULAR);
      assert(src_req.instance_fields.size() == dst_req.instance_fields.size());
      assert(dst_req.privilege == REDUCE);
#endif
      std::set<ApEvent> result_events;
      std::set<ApEvent> fold_copy_preconditions, list_copy_preconditions; 
      fold_copy_preconditions.insert(precondition);
      list_copy_preconditions.insert(precondition);
      std::vector<unsigned> src_indexes(src_req.instance_fields.size());
      std::vector<unsigned> dst_indexes(dst_req.instance_fields.size());
      // Get the field indexes for all the fields
      RegionNode *src_node = get_node(src_req.region);
      RegionNode *dst_node = get_node(dst_req.region);
      src_node->column_source->get_field_indexes(src_req.instance_fields, 
                                                 src_indexes);   
      dst_node->column_source->get_field_indexes(dst_req.instance_fields,
                                                 dst_indexes);
      // Next figure out which src_targets satisfy each field
      std::vector<unsigned> src_target_indexes(src_indexes.size());
      for (unsigned idx1 = 0; idx1 < src_targets.size(); idx1++)
      {
        const FieldMask valid_mask = src_targets[idx1].get_valid_fields();
        for (unsigned idx2 = 0; idx2 < src_indexes.size(); idx2++)
        {
          if (valid_mask.is_set(src_indexes[idx2]))
            src_target_indexes[idx2] = idx1;
        }
      }
      // Let's handle all the actual instances first
      std::vector<CopySrcDstField> src_fields_fold;
      std::vector<CopySrcDstField> dst_fields_fold;
      std::vector<CopySrcDstField> src_fields_list;
      std::vector<CopySrcDstField> dst_fields_list;
      for (unsigned idx1 = 0; idx1 < dst_targets.size(); idx1++)
      {
        const InstanceRef &dst_ref = dst_targets[idx1];
#ifdef DEBUG_LEGION
        assert(!dst_ref.is_virtual_ref());
#endif
        PhysicalManager *dst_manager = dst_ref.get_manager();
        FieldMask dst_valid = dst_ref.get_valid_fields();
        const bool fold = dst_manager->is_reduction_manager();
        // Iterate over all the fields and find the ones for this target
        for (unsigned idx2 = 0; idx2 < src_indexes.size(); idx2++)
        {
          if (dst_valid.is_set(dst_indexes[idx2]))
          {
            // Find the index of the source
            unsigned src_index = src_target_indexes[idx2];
            // Otherwise, this is a normal copy, fill in offsets
            const InstanceRef &src_ref = src_targets[src_index];
#ifdef DEBUG_LEGION
            assert(!src_ref.is_virtual_ref());
#endif
            PhysicalManager *src_manager = src_ref.get_manager();
            if (src_manager->is_reduction_manager())
            {
              FieldMask src_mask;
              src_mask.set_bit(src_indexes[idx2]);
              src_manager->as_reduction_manager()->find_field_offsets(
                  src_mask, fold ? src_fields_fold : src_fields_list);
            }
            else
              src_manager->as_instance_manager()->compute_copy_offsets(
                  src_req.instance_fields[idx2], 
                  fold ? src_fields_fold : src_fields_list);
            if (fold)
            {
              FieldMask dst_mask;
              dst_mask.set_bit(dst_indexes[idx2]);
              dst_manager->as_reduction_manager()->find_field_offsets(
                  dst_mask, dst_fields_fold);
              fold_copy_preconditions.insert(src_ref.get_ready_event());
            }
            else
            {
              dst_manager->as_instance_manager()->compute_copy_offsets(
                  dst_req.instance_fields[idx2], dst_fields_list);
              list_copy_preconditions.insert(src_ref.get_ready_event());
            }
            // Unset the bit and see if we are done with this instance
            dst_valid.unset_bit(dst_indexes[idx2]);
            if (!dst_valid)
              break;
          }
        }
        // Save the copy precondition for the dst
        if (fold)
          fold_copy_preconditions.insert(dst_ref.get_ready_event());
        else
          list_copy_preconditions.insert(dst_ref.get_ready_event());
      }
      // See if we have any fold copies
      if (!dst_fields_fold.empty())
      {
        ApEvent copy_pre = Runtime::merge_events(fold_copy_preconditions);
        ApEvent copy_post = dst_node->issue_copy(op, 
                            src_fields_fold, dst_fields_fold, copy_pre, 
                            predicate_guard, NULL/*intersect*/, 
                            dst_req.redop, true/*fold*/);
        if (copy_post.exists())
          result_events.insert(copy_post);
      }
      // See if we have any reduction copies
      if (!dst_fields_list.empty())
      {
        ApEvent copy_pre = Runtime::merge_events(list_copy_preconditions);
        ApEvent copy_post = dst_node->issue_copy(op, 
                            src_fields_list, dst_fields_list, copy_pre, 
                            predicate_guard, NULL/*intersect*/, 
                            dst_req.redop, false/*fold*/);
        if (copy_post.exists())
          result_events.insert(copy_post);
      }
      return Runtime::merge_events(result_events);
    }

    //--------------------------------------------------------------------------
    int RegionTreeForest::physical_convert_mapping(Operation *op,
                                  const RegionRequirement &req,
                                  const std::vector<MappingInstance> &chosen,
                                  InstanceSet &result, RegionTreeID &bad_tree,
                                  std::vector<FieldID> &missing_fields,
                                  std::map<PhysicalManager*,
                                       std::pair<unsigned,bool> > *acquired,
                                  std::vector<PhysicalManager*> &unacquired,
                                  const bool do_acquire_checks)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_CONVERT_MAPPING_CALL);
      // Can be a part projection if we are closing to a partition node
      FieldSpaceNode *node = get_node(req.parent.get_field_space());
      // Get the field mask for the fields we need
      FieldMask needed_fields = node->get_field_mask(req.privilege_fields);
      const RegionTreeID local_tree = req.parent.get_tree_id();
      // Iterate over each one of the chosen instances
      bool has_composite = false;
      for (std::vector<MappingInstance>::const_iterator it = chosen.begin();
            it != chosen.end(); it++)
      {
        PhysicalManager *manager = it->impl;
        if (manager == NULL)
          continue;
        if (manager->is_virtual_manager())
        {
          has_composite = true;
          continue;
        }
        // Check to see if the tree IDs are the same
        if (local_tree != manager->region_node->handle.get_tree_id())
        {
          bad_tree = manager->region_node->handle.get_tree_id();
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
        if (has_composite)
        {
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
          node->get_field_set(needed_fields, missing);
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
                                     std::map<PhysicalManager*,
                                          std::pair<unsigned,bool> > *acquired,
                                     std::vector<PhysicalManager*> &unacquired,
                                     const bool do_acquire_checks)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *reg_node = get_node(req.region);      
      // Get the field mask for the fields we need
      FieldMask optional_fields = 
                reg_node->column_source->get_field_mask(req.privilege_fields);
      const RegionTreeID local_tree = reg_node->handle.get_tree_id();
      // Iterate over each one of the chosen instances
      bool has_composite = false;
      for (std::vector<MappingInstance>::const_iterator it = chosen.begin();
            it != chosen.end(); it++)
      {
        PhysicalManager *manager = it->impl;
        if (manager == NULL)
          continue;
        if (manager->is_virtual_manager())
        {
          has_composite = true;
          continue;
        }
        // Check to see if the tree IDs are the same
        if (local_tree != manager->region_node->handle.get_tree_id())
        {
          bad_tree = manager->region_node->handle.get_tree_id();
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
    void RegionTreeForest::log_mapping_decision(UniqueID uid, unsigned index,
                                                const RegionRequirement &req,
                                                const InstanceSet &targets,
                                                bool postmapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(runtime->legion_spy_enabled); 
#endif
      FieldSpaceNode *node = (req.handle_type != PART_PROJECTION) ? 
        get_node(req.region.get_field_space()) : 
        get_node(req.partition.get_field_space());
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef &inst = targets[idx];
        const FieldMask &valid_mask = inst.get_valid_fields();
        PhysicalManager *manager = inst.get_manager();
        std::vector<FieldID> valid_fields;
        node->get_field_ids(valid_mask, valid_fields);
        for (std::vector<FieldID>::const_iterator it = valid_fields.begin();
              it != valid_fields.end(); it++)
        {
          if (postmapping)
            LegionSpy::log_post_mapping_decision(uid, index, *it,
                                                 manager->get_use_event());
          else
            LegionSpy::log_mapping_decision(uid, index, *it,
                                            manager->get_use_event());
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_missing_acquires(Operation *op,
                std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired,
                                const std::vector<PhysicalManager*> &unacquired)
    //--------------------------------------------------------------------------
    {
      // This code is very similar to what we see in the memory managers
      std::map<MemoryManager*,MapperManager::AcquireStatus> remote_acquires;
      // Try and do the acquires for any instances that weren't acquired
      for (std::vector<PhysicalManager*>::const_iterator it = 
            unacquired.begin(); it != unacquired.end(); it++)
      {
        if ((*it)->acquire_instance(MAPPING_ACQUIRE_REF, op))
        {
          acquired.insert(std::pair<PhysicalManager*,
             std::pair<unsigned,bool> >(*it,std::pair<unsigned,bool>(1,false)));
          continue;
        }
        // If we failed on the ownr node, it will never work
        // otherwise, we want to try to do a remote acquire
        else if ((*it)->is_owner())
          continue;
        remote_acquires[(*it)->memory_manager].instances.insert(*it);
      }
      if (!remote_acquires.empty())
      {
        std::set<RtEvent> done_events;
        for (std::map<MemoryManager*,MapperManager::AcquireStatus>::iterator
              it = remote_acquires.begin(); it != remote_acquires.end(); it++)
        {
          RtEvent wait_on = it->first->acquire_instances(it->second.instances,
                                                         it->second.results);
          if (wait_on.exists())
            done_events.insert(wait_on);
        }
        if (!done_events.empty())
        {
          RtEvent ready = Runtime::merge_events(done_events);
          ready.wait();
        }
        // Now figure out which ones we successfully acquired and which 
        // ones failed to be acquired
        for (std::map<MemoryManager*,MapperManager::AcquireStatus>::iterator
              req_it = remote_acquires.begin(); 
              req_it != remote_acquires.end(); req_it++)
        {
          unsigned idx = 0;
          for (std::set<PhysicalManager*>::const_iterator it = 
                req_it->second.instances.begin(); it !=
                req_it->second.instances.end(); it++, idx++)
          {
            if (req_it->second.results[idx])
            {
              acquired.insert(std::pair<PhysicalManager*,std::pair<unsigned,
                  bool> >(*it,std::pair<unsigned,bool>(1,false)));
              // make the reference a local reference and 
              // remove our remote did reference
              (*it)->add_base_valid_ref(MAPPING_ACQUIRE_REF, op);
              (*it)->send_remote_valid_update(req_it->first->owner_space,
                                          NULL, 1, false/*add*/);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_colocated(
                            const std::vector<InstanceSet*> &instances,
                            FieldSpace handle, const std::set<FieldID> &fields,
                            unsigned &bad1, unsigned &bad2)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      const FieldMask coloc_mask = node->get_field_mask(fields);
      std::map<PhysicalManager*,FieldMask> colocate_instances;
      // Figure out the first set
      InstanceSet &first_set = *(instances[0]);
      for (unsigned idx = 0; idx < first_set.size(); idx++)
      {
        FieldMask overlap = coloc_mask & first_set[idx].get_valid_fields();
        if (!overlap)
          continue;
        PhysicalManager *manager = first_set[idx].get_manager();
        // Not allowed to have virtual views here
        if (manager->is_virtual_manager())
        {
          bad1 = 0;
          bad2 = 0;
          return false;
        }
        colocate_instances[manager] = overlap;
      }
      // Now we've got the first set, check all the rest
      for (unsigned idx1 = 0; idx1 < instances.size(); idx1++)
      {
        InstanceSet &next_set = *(instances[idx1]);
        for (unsigned idx2 = 0; idx2 < next_set.size(); idx2++)
        {
          FieldMask overlap = coloc_mask & next_set[idx2].get_valid_fields();
          if (!overlap)
            continue;
          PhysicalManager *manager = next_set[idx2].get_manager();
          if (manager->is_virtual_manager())
          {
            bad1 = idx2;
            bad2 = idx2;
            return false;
          }
          std::map<PhysicalManager*,FieldMask>::const_iterator finder = 
            colocate_instances.find(manager);
          if ((finder == colocate_instances.end()) ||
              (!!(overlap - finder->second)))
          {
            bad1 = 0;
            bad2 = idx2;
            return false;
          }
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::fill_fields(Operation *op,
                                          const RegionRequirement &req,
                                          const unsigned index,
                                          const void *value, size_t value_size,
                                          VersionInfo &version_info,
                                          RestrictInfo &restrict_info,
                                          InstanceSet &instances, 
                                          ApEvent precondition,
                                          std::set<RtEvent> &map_applied_events,
                                          PredEvent true_guard, 
                                          PredEvent false_guard)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_FILL_FIELDS_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      InnerContext *context = op->find_physical_context(index);
      RegionTreeContext ctx = context->get_context();
      RegionNode *fill_node = get_node(req.region);
      FieldMask fill_mask = 
        fill_node->column_source->get_field_mask(req.privilege_fields);
      const UniqueID logical_ctx_uid = op->get_context()->get_context_uid();
      // Check to see if we have any restricted fields that
      // we need to fill eagerly
      if (restrict_info.has_restrictions())
      {
#ifdef DEBUG_LEGION
        assert(!instances.empty());
#endif
        // If we have restrictions, we have to eagerly fill these fields 
        FieldMask eager_fields;
        restrict_info.populate_restrict_fields(eager_fields);
        ApEvent done_event = fill_node->eager_fill_fields(ctx.get_id(), 
         op, index, logical_ctx_uid, context, eager_fields, value, value_size,
         version_info, instances, precondition, true_guard, map_applied_events);
        // Remove these fields from the fill set
        fill_mask -= eager_fields;
        // If we still have fields to fill, do that now
        if (!!fill_mask)
          fill_node->fill_fields(ctx.get_id(), fill_mask, value, value_size,
                                 logical_ctx_uid, context, version_info, 
                                 map_applied_events, true_guard, false_guard
#ifdef LEGION_SPY
                                 , op->get_unique_op_id()
#endif
                                 );
        // We know the sync precondition is chained off at least
        // one eager fill so we can return the done event
        return done_event;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(instances.empty());
#endif
        // Fill in these fields on this node
        fill_node->fill_fields(ctx.get_id(), fill_mask, value, value_size,
                               logical_ctx_uid, context, version_info, 
                               map_applied_events, true_guard, false_guard
#ifdef LEGION_SPY
                               , op->get_unique_op_id()
#endif
                               );
        // We didn't actually use the precondition so just return it
        return precondition;
      }
    }

    //--------------------------------------------------------------------------
    InstanceManager* RegionTreeForest::create_external_instance(
                             AttachOp *attach_op, const RegionRequirement &req,
                             const std::vector<FieldID> &field_set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *attach_node = get_node(req.region);
      return attach_node->column_source->create_external_instance(
                                field_set, attach_node, attach_op);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::attach_external(AttachOp *attach_op, 
                                                 unsigned index,
                                                 const RegionRequirement &req,
                                                 InstanceManager *ext_instance,
                                                 VersionInfo &version_info,
                                                 std::set<RtEvent> &map_applied)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_ATTACH_EXTERNAL_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      InnerContext *context = attach_op->find_physical_context(index);
      RegionTreeContext ctx = context->get_context();
      RegionNode *attach_node = get_node(req.region);
      UniqueID logical_ctx_uid = attach_op->get_context()->get_context_uid();
      // Perform the attachment
      return attach_node->attach_external(ctx.get_id(), context,logical_ctx_uid,
                                        ext_instance->layout->allocated_fields, 
                                        req, ext_instance, 
                                        version_info, map_applied);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionTreeForest::detach_external(const RegionRequirement &req,
                                          DetachOp *detach_op,
                                          unsigned index,
                                          VersionInfo &version_info,
                                          const InstanceRef &ref,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REGION_TREE_PHYSICAL_DETACH_EXTERNAL_CALL);
#ifdef DEBUG_LEGION
      assert(req.handle_type == SINGULAR);
#endif
      InnerContext *context = detach_op->find_physical_context(index);
      RegionTreeContext ctx = context->get_context();
      RegionNode *detach_node = get_node(req.region);
      UniqueID logical_ctx_uid = detach_op->get_context()->get_context_uid();
      // Perform the detachment
      return detach_node->detach_external(ctx.get_id(), context,logical_ctx_uid,
                                          version_info, ref,map_applied_events);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::check_context_state(RegionTreeContext ctx)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeID,RegionNode*> trees;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        trees = tree_nodes;
      }
      CurrentInitializer init(ctx.get_id());
      for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
            trees.begin(); it != trees.end(); it++)
      {
        it->second->visit_node(&init);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(IndexSpace sp,
                                                  const void *realm_is,
                                                  IndexPartNode *parent,
                                                  LegionColor color,
                                                  DistributedID did,
                                                  ApEvent is_ready)
    //--------------------------------------------------------------------------
    { 
      IndexSpaceCreator creator(this, sp, realm_is, parent, 
                                color, did, is_ready);
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
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
      }
      LocalReferenceMutator mutator;
      // If we're the owner add a valid reference that will be removed
      // when we are deleted, otherwise we're remote so we add a gc 
      // reference that will be removed by the owner when we can be
      // safely collected
      if (result->is_owner())
        result->add_base_valid_ref(APPLICATION_REF, &mutator);
      else
        result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
      result->register_with_runtime(&mutator);
      if (parent != NULL)
        parent->add_child(result);
      
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(IndexSpace sp,
                                                  const void *realm_is,
                                                  IndexPartNode *parent,
                                                  LegionColor color,
                                                  DistributedID did,
                                                  ApUserEvent is_ready)
    //--------------------------------------------------------------------------
    { 
      IndexSpaceCreator creator(this, sp, realm_is, parent, 
                                color, did, is_ready);
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
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          // Free up the event since we didn't use it
          Runtime::trigger_event(is_ready);
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
      }
      LocalReferenceMutator mutator;
      // If we're the owner add a valid reference that will be removed
      // when we are deleted, otherwise we're remote so we add a gc 
      // reference that will be removed by the owner when we can be
      // safely collected
      if (result->is_owner())
        result->add_base_valid_ref(APPLICATION_REF, &mutator);
      else
        result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
      result->register_with_runtime(&mutator);
      if (parent != NULL)
        parent->add_child(result);
      
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, 
                                                 IndexSpaceNode *parent,
                                                 IndexSpaceNode *color_space,
                                                 LegionColor color,
                                                 bool disjoint, 
                                                 DistributedID did,
                                                 ApEvent part_ready,
                                                 ApUserEvent pending)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(this, p, parent, color_space, 
                               color, disjoint, did, part_ready, pending);
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
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
      }
      LocalReferenceMutator mutator;
      // If we're the owner add a valid reference that will be removed
      // when we are deleted, otherwise we're remote so we add a gc 
      // reference that will be removed by the owner when we can be
      // safely collected
      if (result->is_owner())
      {
        result->add_base_valid_ref(APPLICATION_REF, &mutator);
        // Also add references to our color space
        color_space->add_nested_valid_ref(did, &mutator);
        color_space->add_nested_resource_ref(did);
      }
      else
      {
        result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
        color_space->add_nested_resource_ref(did);
      }
      result->register_with_runtime(&mutator);
      if (parent != NULL)
        parent->add_child(result);
      
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, 
                                                 IndexSpaceNode *parent,
                                                 IndexSpaceNode *color_space,
                                                 LegionColor color,
                                                 RtEvent disjointness_ready,
                                                 DistributedID did,
                                                 ApEvent part_ready,
                                                 ApUserEvent pending)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(this, p, parent, color_space, color, 
                               disjointness_ready, did, part_ready, pending);
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
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
      }
      LocalReferenceMutator mutator;
      // If we're the owner add a valid reference that will be removed
      // when we are deleted, otherwise we're remote so we add a gc 
      // reference that will be removed by the owner when we can be
      // safely collected
      if (result->is_owner())
      {
        result->add_base_valid_ref(APPLICATION_REF, &mutator);
        // Also add references to our color space
        color_space->add_nested_valid_ref(did, &mutator);
        color_space->add_nested_resource_ref(did);
      }
      else
      {
        result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
        color_space->add_nested_resource_ref(did);
      }
      result->register_with_runtime(&mutator);
      if (parent != NULL)
        parent->add_child(result);
      
      return result;
    }
 
    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace space,
                                                  DistributedID did)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *result = new FieldSpaceNode(space, this, did);
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
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        field_nodes[space] = result;
        field_space_requests.erase(space);
      }
      LocalReferenceMutator mutator;
      // If we're the owner add a valid reference that will be removed
      // when we are deleted, otherwise we're remote so we add a gc 
      // reference that will be removed by the owner when we can be
      // safely collected
      if (result->is_owner())
        result->add_base_valid_ref(APPLICATION_REF, &mutator);
      else
        result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
      result->register_with_runtime(&mutator);
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace space,
                                                  DistributedID did,
                                                  Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *result = new FieldSpaceNode(space, this, did, derez);
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
          // Need to remove resource reference if not owner
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        field_nodes[space] = result;
        field_space_requests.erase(space);
      }
      LocalReferenceMutator mutator;
      // If we're the owner add a valid reference that will be removed
      // when we are deleted, otherwise we're remote so we add a gc 
      // reference that will be removed by the owner when we can be
      // safely collected
      if (result->is_owner())
        result->add_base_valid_ref(APPLICATION_REF, &mutator);
      else
        result->add_base_gc_ref(REMOTE_DID_REF, &mutator);
      result->register_with_runtime(&mutator);
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::create_node(LogicalRegion r, 
                                              PartitionNode *parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (parent != NULL)
      {
        assert(r.field_space == parent->handle.field_space);
        assert(r.tree_id == parent->handle.tree_id);
      }
#endif
      IndexSpaceNode *row_src = get_node(r.index_space);
      FieldSpaceNode *col_src = get_node(r.field_space);
      RegionNode *result = new RegionNode(r, parent, row_src, 
                                          col_src, this);
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
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        // Now we can add it to the map
        region_nodes[r] = result;
        // Add a resource reference to it that we'll remove later
        result->add_base_resource_ref(REGION_TREE_REF);
        // If this is a top level region add it to the collection
        // of top level tree IDs
        if (parent == NULL)
        {
#ifdef DEBUG_LEGION
          assert(tree_nodes.find(r.tree_id) == tree_nodes.end());
#endif
          tree_nodes[r.tree_id] = result;
          region_tree_requests.erase(r.tree_id);
        }
      }
      result->record_registered();

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
      assert(p.tree_id = parent->handle.tree_id);
#endif
      IndexPartNode *row_src = get_node(p.index_partition);
      FieldSpaceNode *col_src = get_node(p.field_space);
      PartitionNode *result = new PartitionNode(p, parent, row_src, 
                                                col_src, this);
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
          if (result->is_owner() || 
              result->remove_base_resource_ref(REMOTE_DID_REF))
            delete result;
          return it->second;
        }
        // Now we can put the node in the map
        part_nodes[p] = result;
        // Record a resource reference on it that we'll remove later
        result->add_base_resource_ref(REGION_TREE_REF);
      }
      result->record_registered();
      
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::get_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      if (!space.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_FOR_INDEXSPACE,
          "Invalid request for IndexSpace NO_SPACE.")
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/); 
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator finder = 
          index_nodes.find(space);
        if (finder != index_nodes.end())
          return finder->second;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpace owner = IndexSpaceNode::get_owner_space(space, runtime);
      if (owner == runtime->address_space)
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for index space %x.", space.id)
      // Retake the lock and get something to wait on
      RtEvent wait_on;
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
      // Wait on the event
      wait_on.wait();
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<IndexSpace,IndexSpaceNode*>::const_iterator finder = 
          index_nodes.find(space);
      if (finder == index_nodes.end())
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for index space %x."
                        "This is definitely a runtime bug.", space.id)
      return finder->second;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::get_node(IndexPartition part,
                                              RtEvent *defer/* = NULL*/)
    //--------------------------------------------------------------------------
    {
      if (!part.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_INDEXPARTITION,
          "Invalid request for IndexPartition NO_PART.")
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<IndexPartition,IndexPartNode*>::const_iterator finder =
          index_parts.find(part);
        if (finder != index_parts.end())
          return finder->second;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpace owner = IndexPartNode::get_owner_space(part, runtime);
      if (owner == runtime->address_space)
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for index partition %x.",part.id)
      RtEvent wait_on;
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
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<IndexPartition,IndexPartNode*>::const_iterator finder = 
          index_parts.find(part);
        if (finder == index_parts.end())
          REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
            "Unable to find entry for index partition %x. "
                          "This is definitely a runtime bug.", part.id)
        return finder->second;
      }
      else
      {
        *defer = wait_on;
        return NULL;
      }
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::get_node(FieldSpace space) 
    //--------------------------------------------------------------------------
    {
      if (!space.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_FIELDSPACE,
          "Invalid request for FieldSpace NO_SPACE.")
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<FieldSpace,FieldSpaceNode*>::const_iterator finder = 
          field_nodes.find(space);
        if (finder != field_nodes.end())
          return finder->second;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpaceID owner = FieldSpaceNode::get_owner_space(space, runtime); 
      if (owner == runtime->address_space)
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for field space %x.", space.id)
      RtEvent wait_on;
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
      // Wait for the event to be ready
      wait_on.wait();
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<FieldSpace,FieldSpaceNode*>::const_iterator finder = 
        field_nodes.find(space);
      if (finder == field_nodes.end())
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for field space %x. "
                        "This is definitely a runtime bug.", space.id)
      return finder->second;
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
      bool has_top_level_region;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<LogicalRegion,RegionNode*>::const_iterator it = 
          region_nodes.find(handle);
        if (it != region_nodes.end())
          return it->second;
        // Check to see if we have the top level region
        if (need_check)
          has_top_level_region = 
            (tree_nodes.find(handle.get_tree_id()) != tree_nodes.end());
        else
          has_top_level_region = true;
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
        RtEvent wait_on;
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
          // Retake the lock and see again if the handle we
          // were looking for was the top-level node or not
          AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
          std::map<LogicalRegion,RegionNode*>::const_iterator it = 
            region_nodes.find(handle);
          if (it != region_nodes.end())
            return it->second;
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
        return create_node(handle, parent);
      }
      return create_node(handle, NULL);
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::get_node(LogicalPartition handle,
                                              bool need_check /* = true*/)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_LOGICALPARTITION,
          "Invalid request for LogicalPartition NO_PART.")
      // Check to see if the node already exists
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<LogicalPartition,PartitionNode*>::const_iterator it =
          part_nodes.find(handle);
        if (it != part_nodes.end())
          return it->second;
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
      return create_node(handle, parent);
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::get_tree(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      if (tid == 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_REQUEST_TREE_ID,
          "Invalid request for tree ID 0 which is never a tree ID")
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<RegionTreeID,RegionNode*>::const_iterator finder = 
          tree_nodes.find(tid);
        if (finder != tree_nodes.end())
          return finder->second;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpaceID owner = RegionTreeNode::get_owner_space(tid, runtime);
      if (owner == runtime->address_space)
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_ENTRY,
          "Unable to find entry for region tree ID %d", tid)
      RtEvent wait_on;
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
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        if (index_nodes.find(space) != index_nodes.end())
          return true;
      }
      return (get_node(space) != NULL);
    }
    
    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexPartition part)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        if (index_parts.find(part) != index_parts.end())
          return true;
      }
      return (get_node(part) != NULL);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        if (field_nodes.find(space) != field_nodes.end())
          return true;
      }
      return (get_node(space) != NULL);
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
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        if (tree_nodes.find(tid) != tree_nodes.end())
          return true;
      }
      return (get_tree(tid) != NULL);
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
      RegionNode *node = NULL;
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
        node = finder->second;
        region_nodes.erase(finder);
      }
      if (node->remove_base_resource_ref(REGION_TREE_REF))
        delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::remove_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      PartitionNode *node = NULL;
      {
        AutoLock l_lock(lookup_lock);
        std::map<LogicalPartition,PartitionNode*>::iterator finder = 
          part_nodes.find(handle);
#ifdef DEBUG_LEGION
        assert(finder != part_nodes.end());
#endif
        node = finder->second;
        part_nodes.erase(finder);
      }
      if (node->remove_base_resource_ref(REGION_TREE_REF))
        delete node;
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
    bool RegionTreeForest::are_compatible(IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      return (left.get_type_tag() == right.get_type_tag());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_dominated(IndexSpace src, IndexSpace dst)
    //--------------------------------------------------------------------------
    {
      // Check to see if dst is dominated by source
#ifdef DEBUG_LEGION
      assert(are_compatible(src, dst));
#endif
      IndexSpaceNode *src_node = get_node(src);
      IndexSpaceNode *dst_node = get_node(dst);
      return src_node->dominates(dst_node);
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
                                                       bool is_mutable)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                                    size, is_mutable);
      if (runtime->legion_spy_enabled && (NAME_SEMANTIC_TAG == tag))
        LegionSpy::log_index_space_name(handle.id,
            reinterpret_cast<const char*>(buffer));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(IndexPartition handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                                    size, is_mutable);
      if (runtime->legion_spy_enabled && (NAME_SEMANTIC_TAG == tag))
        LegionSpy::log_index_partition_name(handle.id,
            reinterpret_cast<const char*>(buffer));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(FieldSpace handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                                    size, is_mutable);
      if (runtime->legion_spy_enabled && (NAME_SEMANTIC_TAG == tag))
        LegionSpy::log_field_space_name(handle.id,
            reinterpret_cast<const char*>(buffer));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(FieldSpace handle,
                                                       FieldID fid,
                                                       SemanticTag tag,
                                                       AddressSpaceID src,
                                                       const void *buf,
                                                       size_t size,
                                                       bool is_mutable)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(fid, tag, src, buf, 
                                                    size, is_mutable);
      if (runtime->legion_spy_enabled && (NAME_SEMANTIC_TAG == tag))
        LegionSpy::log_field_name(handle.id, fid,
            reinterpret_cast<const char*>(buf));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(LogicalRegion handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                                    size, is_mutable);
      if (runtime->legion_spy_enabled && (NAME_SEMANTIC_TAG == tag))
        LegionSpy::log_logical_region_name(handle.index_space.id,
            handle.field_space.id, handle.tree_id,
            reinterpret_cast<const char*>(buffer));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::attach_semantic_information(LogicalPartition handle,
                                                       SemanticTag tag,
                                                       AddressSpaceID source,
                                                       const void *buffer,
                                                       size_t size,
                                                       bool is_mutable)
    //--------------------------------------------------------------------------
    {
      get_node(handle)->attach_semantic_information(tag, source, buffer, 
                                                    size, is_mutable);
      if (runtime->legion_spy_enabled && (NAME_SEMANTIC_TAG == tag))
        LegionSpy::log_logical_partition_name(handle.index_partition.id,
            handle.field_space.id, handle.tree_id,
            reinterpret_cast<const char*>(buffer));
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

    /////////////////////////////////////////////////////////////
    // Index Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTreeNode::IndexTreeNode(RegionTreeForest *ctx, unsigned d,
                         LegionColor c, DistributedID did, AddressSpaceID owner)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_TREE_NODE_DC), 
          owner, false/*register*/),
        context(ctx), depth(d), color(c), destroyed(false),
        node_lock(this->gc_lock)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ctx != NULL);
#endif
    }

    //--------------------------------------------------------------------------
    IndexTreeNode::~IndexTreeNode(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTreeNode::attach_semantic_information(SemanticTag tag,
                                                    AddressSpaceID source,
                                                    const void *buffer, 
                                                    size_t size,bool is_mutable)
    //--------------------------------------------------------------------------
    {
      // Make a copy
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
            (source != owner_space))
        {
          send_semantic_info(owner_space, tag, buffer, size, is_mutable); 
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::const_iterator finder = 
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
      LegionMap<SemanticTag,SemanticInfo>::aligned::const_iterator finder = 
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
                                   DistributedID did, ApEvent ready)
      : IndexTreeNode(ctx, (par == NULL) ? 0 : par->depth + 1, c,
                      did, get_owner_space(h, ctx->runtime)),
        handle(h), parent(par), index_space_ready(ready), 
        realm_index_space_set(Runtime::create_rt_user_event()), 
        tight_index_space_set(Runtime::create_rt_user_event()),
        tight_index_space(false)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->add_nested_resource_ref(did);
#ifdef DEBUG_LEGION
      if (parent != NULL)
        assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Space %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, handle.id);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(const IndexSpaceNode &rhs)
      : IndexTreeNode(NULL, 0, 0, 0, 0), handle(IndexSpace::NO_SPACE), 
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
      {
        if (parent != NULL)
          parent->remove_child(color);
        context->remove_node(handle);
      }
      if ((parent != NULL) && parent->remove_nested_resource_ref(did))
        delete parent;
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
    void IndexSpaceNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we add a valid reference to the owner
      if (!is_owner())
        send_remote_valid_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::DestructionFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      node->send_remote_gc_update(target, mutator, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexPartNode*> to_delete;
      if (is_owner())
      {
        // Need read-only lock while accessing these structures, 
        // we know they won't change while accessing them
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        // Remove gc references from our remote nodes
        if (!!remote_instances)
        {
          DestructionFunctor functor(this, mutator);
          remote_instances.map(functor);
        }
        for (std::map<LegionColor,IndexPartNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
          if (it->second->destroy_node(runtime->address_space))
            to_delete.push_back(it->second);
      }
      else // Remove the valid reference that we have on the owner
        send_remote_valid_update(owner_space, mutator, 1/*count*/,false/*add*/);
      for (std::vector<IndexPartNode*>::const_iterator it = 
            to_delete.begin(); it != to_delete.end(); it++)
        delete (*it);
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
          SemanticRequestArgs args;
          args.proxy_this = this;
          args.tag = tag;
          args.source = source;
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
      forest->attach_semantic_information(handle, tag, source, 
                                          buffer, size, is_mutable);
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
        if (finder != color_map.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second != NULL);
#endif
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
          return context->get_node(remote_handle, defer);
        if (can_fail)
          return NULL;
        REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_COLOR,
          "Unable to find entry for color %lld in "
                        "index space %x.", c, handle.id)
      }
      RtUserEvent ready_event = Runtime::create_rt_user_event();
      IndexPartition child_handle = IndexPartition::NO_PART;
      IndexPartition *volatile handle_ptr = &child_handle;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(c);
        rez.serialize(handle_ptr);
        rez.serialize(ready_event);
      }
      context->runtime->send_index_space_child_request(owner_space, rez);
      if (defer == NULL)
      {
        ready_event.wait();
        // Stupid volatile-ness
        IndexPartition handle_copy = *handle_ptr;
        if (!handle_copy.exists())
        {
          if (can_fail)
            return NULL;
          REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_COLOR,
            "Unable to find entry for color %lld in "
                          "index space %x.", c, handle.id)
        }
        return context->get_node(handle_copy);
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
#ifdef DEBUG_LEGION
      std::map<LegionColor,IndexPartNode*>::iterator finder = color_map.find(c);
      assert(finder != color_map.end());
      color_map.erase(finder);
#else
      color_map.erase(c);
#endif
    }

    //--------------------------------------------------------------------------
    size_t IndexSpaceNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return color_map.size();
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::are_disjoint(const LegionColor c1, 
                                      const LegionColor c2)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (c1 == c2)
        return false;
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
            if (implicit_runtime->dynamic_independence_tests)
              issue_dynamic_test = true;
            else
            {
              aliased_subsets.insert(key);
              aliased_subsets.insert(std::pair<LegionColor,LegionColor>(c2,c1));
              return false;
            }
          }
        }
      }
      if (issue_dynamic_test)
      {
        IndexPartNode *left = get_child(c1);
        IndexPartNode *right = get_child(c2);
        AutoLock n_lock(node_lock);
        // Test again to make sure we didn't lose the race
        std::map<std::pair<LegionColor,LegionColor>,RtEvent>::const_iterator
          finder = pending_tests.find(key);
        if (finder == pending_tests.end())
        {
          DynamicIndependenceArgs args;
          args.parent = this;
          args.left = left;
          args.right = right;
          // Get the preconditions for domains 
          RtEvent pre = Runtime::protect_event(
              Runtime::merge_events(left->partition_ready,
                                    right->partition_ready));
          ready = context->runtime->issue_runtime_meta_task(args,
                                      LG_LATENCY_WORK_PRIORITY, pre);
          pending_tests[key] = ready;
          pending_tests[std::pair<LegionColor,LegionColor>(c2,c1)] = ready;
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
                                     const LegionColor c1, const LegionColor c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (disjoint)
      {
        disjoint_subsets.insert(std::pair<LegionColor,LegionColor>(c1,c2));
        disjoint_subsets.insert(std::pair<LegionColor,LegionColor>(c2,c1));
      }
      else
      {
        aliased_subsets.insert(std::pair<LegionColor,LegionColor>(c1,c2));
        aliased_subsets.insert(std::pair<LegionColor,LegionColor>(c2,c1));
      }
      pending_tests.erase(std::pair<LegionColor,LegionColor>(c1,c2));
      pending_tests.erase(std::pair<LegionColor,LegionColor>(c2,c1));
    }

    //--------------------------------------------------------------------------
    LegionColor IndexSpaceNode::generate_color(void)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      LegionColor result;
      if (!color_map.empty())
      {
        unsigned stride = context->runtime->get_color_modulus();
        std::map<LegionColor,IndexPartNode*>::const_reverse_iterator rlast = 
                                                        color_map.rbegin();
        // We know all colors for index spaces are 0-D
        LegionColor new_color =
          (rlast->first + (stride - 1)) / stride * stride +
          context->runtime->get_start_color();
        if (new_color == rlast->first) 
          new_color += stride;
        result = new_color;
      }
      else
        result = context->runtime->get_start_color();
#ifdef DEBUG_LEGION
      assert(color_map.find(result) == color_map.end());
#endif
      // We have to put ourselves in the map to be sound for other parallel
      // allocations of colors which may come later
      color_map[result] = NULL; /* just put in a NULL pointer for now */
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::record_remote_child(IndexPartition pid, 
                                             LegionColor part_color)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(color_map.find(part_color) == color_map.end());
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
          if (it->second != NULL)
            colors.push_back(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(logical_nodes.find(inst) == logical_nodes.end());
#endif
      logical_nodes.insert(inst);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::has_instance(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle.get_tree_id() == tid)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      std::set<RegionNode*>::iterator finder = logical_nodes.find(inst);
      assert(finder != logical_nodes.end());
      logical_nodes.erase(finder);
#else
      logical_nodes.erase(inst);
#endif
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
    /*static*/ void IndexSpaceNode::handle_disjointness_test(
              IndexSpaceNode *parent, IndexPartNode *left, IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
      const bool disjoint = !left->intersects_with(right);
      parent->record_disjointness(disjoint, left->color, right->color);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_tighten_index_space(const void *args)
    //--------------------------------------------------------------------------
    {
      const TightenIndexSpaceArgs *targs = (const TightenIndexSpaceArgs*)args;
      targs->proxy_this->tighten_index_space();
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_node(AddressSpaceID target, bool up)
    //--------------------------------------------------------------------------
    {
      // Go up first so we know those nodes will be there
      if (up && (parent != NULL))
        parent->send_node(target, true/*up*/);
      {
        AutoLock n_lock(node_lock);
        if (!remote_instances.contains(target))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(did);
            if (parent != NULL)
              rez.serialize(parent->handle);
            else
              rez.serialize(IndexPartition::NO_PART);
            rez.serialize(color);
            rez.serialize(index_space_ready);
            if (realm_index_space_set.has_triggered())
              pack_index_space(rez, true/*include size*/);
            else
              rez.serialize<size_t>(0);
            rez.serialize<size_t>(semantic_info.size());
            for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
                  semantic_info.begin(); it != semantic_info.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second.size);
              rez.serialize(it->second.buffer, it->second.size);
              rez.serialize(it->second.is_mutable);
            }
          }
          context->runtime->send_index_space_node(target, rez); 
          remote_instances.add(target);
        }
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
      size_t index_space_size;
      derez.deserialize(index_space_size);
      const void *index_space_ptr = 
        (index_space_size > 0) ? derez.get_current_pointer() : NULL;

      IndexPartNode *parent_node = NULL;
      if (parent != IndexPartition::NO_PART)
      {
        parent_node = context->get_node(parent);
#ifdef DEBUG_LEGION
        assert(parent_node != NULL);
#endif
      }
      IndexSpaceNode *node = context->create_node(handle, index_space_ptr,
                                     parent_node, color, did, ready_event);
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
        node->attach_semantic_information(tag, source, 
                                          buffer, buffer_size, is_mutable);
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
      IndexSpaceNode *target = forest->get_node(handle);
      target->send_node(source, true/*up*/);
      // Then send back the flush
      Serializer rez;
      rez.serialize(to_trigger);
      forest->runtime->send_index_space_return(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
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
      IndexPartNode *target;
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
        DeferChildArgs args;
        args.proxy_this = parent;
        args.child_color = child_color;
        args.target = target;
        args.to_trigger = to_trigger;
        args.source = source;
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
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartition *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      (*target) = handle;
      Runtime::trigger_event(to_trigger);
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
      node->unpack_index_space(derez, source);
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(RegionTreeForest *ctx, IndexPartition p, 
                                 IndexSpaceNode *par, IndexSpaceNode *color_sp,
                                 LegionColor c, bool dis, DistributedID did,
                                 ApEvent part_ready, ApUserEvent partial)
      : IndexTreeNode(ctx, par->depth+1, c, did, 
                      get_owner_space(p, ctx->runtime)), 
        handle(p), parent(par), color_space(color_sp), 
        total_children(color_sp->get_volume()), 
        max_linearized_color(color_sp->get_max_linearized_color()),
        partition_ready(part_ready), partial_pending(partial),
        disjoint(dis), has_complete(false)
    //--------------------------------------------------------------------------
    { 
      parent->add_nested_resource_ref(did);
#ifdef DEBUG_LEGION
      if (partial_pending.exists())
        assert(partial_pending == partition_ready);
      assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Partition %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, handle.id); 
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(RegionTreeForest *ctx, IndexPartition p, 
                                 IndexSpaceNode *par, IndexSpaceNode *color_sp,
                                 LegionColor c, RtEvent dis_ready, 
                                 DistributedID did,
                                 ApEvent part_ready, ApUserEvent part)
      : IndexTreeNode(ctx, par->depth+1, c, did, 
                      get_owner_space(p, ctx->runtime)), handle(p), parent(par),
        color_space(color_sp), total_children(color_sp->get_volume()),
        max_linearized_color(color_sp->get_max_linearized_color()),
        partition_ready(part_ready), partial_pending(part), 
        disjoint_ready(dis_ready), disjoint(false), has_complete(false)
    //--------------------------------------------------------------------------
    {
      parent->add_nested_resource_ref(did);
#ifdef DEBUG_LEGION
      if (partial_pending.exists())
        assert(partial_pending == partition_ready);
      assert(handle.get_type_tag() == parent->handle.get_type_tag());
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Index Partition %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, handle.id);
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(const IndexPartNode &rhs)
      : IndexTreeNode(NULL,0,0,0,0), handle(IndexPartition::NO_PART), 
        parent(NULL),color_space(NULL),total_children(0),max_linearized_color(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexPartNode::~IndexPartNode(void)
    //--------------------------------------------------------------------------
    {
      // Lastly we can unregister ourselves with the context
      if (registered_with_runtime)
      {
        parent->remove_child(color);
        context->remove_node(handle);
        if (color_space->remove_nested_resource_ref(did))
          delete color_space;
      }
      if (parent->remove_nested_resource_ref(did))
        delete parent;
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
    void IndexPartNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we add a valid reference to the owner
      if (!is_owner())
        send_remote_valid_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::DestructionFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      node->send_remote_gc_update(target, mutator, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpaceNode*> to_delete;
      if (is_owner())
      {
        // Remove the valid reference that we hold on the color space
        // No need to check for deletion since we have a resource ref too
        color_space->remove_nested_valid_ref(did, mutator);
        // Need read-only lock while accessing these structures, 
        // we know they won't change while accessing them
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        // Remove gc references from our remote nodes
        if (!!remote_instances)
        {
          DestructionFunctor functor(this, mutator);
          remote_instances.map(functor);
        }
        for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
          if (it->second->destroy_node(runtime->address_space))
            to_delete.push_back(it->second);
      }
      else // Remove the valid reference that we have on the owner
        send_remote_valid_update(owner_space, mutator, 1/*count*/,false/*add*/);
      for (std::vector<IndexSpaceNode*>::const_iterator it = 
            to_delete.begin(); it != to_delete.end(); it++)
        delete (*it);
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
          SemanticRequestArgs args;
          args.proxy_this = this;
          args.tag = tag;
          args.source = source;
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
      forest->attach_semantic_information(handle, tag, source, 
                                          buffer, size, is_mutable);
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
            ApUserEvent partial_event = Runtime::create_ap_user_event();
            result = context->create_node(is, NULL/*realm is*/,
                                          this, c, did, partial_event);
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
              Runtime::trigger_event(partial_pending,
                  Runtime::merge_events(child_ready_events));
          }
          else
            // Make a new index space node ready when the partition is ready
            result = context->create_node(is, NULL/*realm is*/, 
                                          this, c, did, partition_ready);
          if (runtime->legion_spy_enabled)
            LegionSpy::log_index_subspace(handle.id, is.id, 
                          result->get_domain_point_color());
          return result; 
        }
      }
      // Otherwise, request a child node from the owner node
      else
      {
        IndexSpace child_handle = IndexSpace::NO_SPACE;
        IndexSpace *volatile handle_ptr = &child_handle;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(c);
          rez.serialize(handle_ptr);
          rez.serialize(ready_event);
        }
        context->runtime->send_index_partition_child_request(owner_space, rez);
        if (defer == NULL)
        {
          ready_event.wait();
          IndexSpace copy_handle = *handle_ptr;
#ifdef DEBUG_LEGION
          assert(copy_handle.exists());
#endif
          return context->get_node(copy_handle);
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
      RtUserEvent to_trigger;
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
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
    void IndexPartNode::remove_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      std::map<LegionColor,IndexSpaceNode*>::iterator finder = 
        color_map.find(c);
      assert(finder != color_map.end());
      color_map.erase(finder);
#else
      color_map.erase(c);
#endif
    }

    //--------------------------------------------------------------------------
    size_t IndexPartNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return color_space->get_volume();
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
      if (total_children == max_linearized_color)
      {
        for (LegionColor c1 = 0; disjoint && 
              (c1 < max_linearized_color); c1++)
        {
          for (LegionColor c2 = c1 + 1; disjoint && 
                (c2 < max_linearized_color); c2++)
          {
            if (!are_disjoint(c1, c2, true/*force compute*/))
              disjoint = false;
          }
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
              disjoint = false;
          }
        }
      }
      // Once we get here, we know the disjointness result so we can
      // trigger the event saying when the disjointness value is ready
      Runtime::trigger_event(ready_event);
      // Record the result for Legion Spy
      if (runtime->legion_spy_enabled)
          LegionSpy::log_index_partition(parent->handle.id, handle.id, 
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
    bool IndexPartNode::are_disjoint(const LegionColor c1, const LegionColor c2,
                                     bool force_compute)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return false;
      if (!force_compute && is_disjoint(false/*appy query*/))
        return true;
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
            if (implicit_runtime->dynamic_independence_tests)
              issue_dynamic_test = true;
            else
            {
              aliased_subspaces.insert(key);
              aliased_subspaces.insert(
                  std::pair<LegionColor,LegionColor>(c2,c1));
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
          DynamicIndependenceArgs args;
          args.parent = this;
          args.left = left;
          args.right = right;
          ApEvent pre = Runtime::merge_events(left_pre, right_pre);
          ready_event = context->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_WORK_PRIORITY, Runtime::protect_event(pre));
          pending_tests[key] = ready_event;
          pending_tests[std::pair<LegionColor,LegionColor>(c2,c1)] =ready_event;
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
                                     const LegionColor c1, const LegionColor c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (result)
      {
        disjoint_subspaces.insert(std::pair<LegionColor,LegionColor>(c1,c2));
        disjoint_subspaces.insert(std::pair<LegionColor,LegionColor>(c2,c1));
      }
      else
      {
        aliased_subspaces.insert(std::pair<LegionColor,LegionColor>(c1,c2));
        aliased_subspaces.insert(std::pair<LegionColor,LegionColor>(c2,c1));
      }
      pending_tests.erase(std::pair<LegionColor,LegionColor>(c1,c2));
      pending_tests.erase(std::pair<LegionColor,LegionColor>(c2,c1));
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_complete(bool from_app/*=false*/)
    //--------------------------------------------------------------------------
    {
      // If we've cached the value then we are good to go
      {
        AutoLock n_lock(node_lock, 1, false/*exclusive*/);
        if (has_complete)
          return complete;
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
      complete = result;
      has_complete = true;
      return complete;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::get_colors(std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      color_space->instantiate_colors(colors);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_instance(PartitionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(logical_nodes.find(inst) == logical_nodes.end());
#endif
      logical_nodes.insert(inst);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::has_instance(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<PartitionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle.get_tree_id() == tid)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::remove_instance(PartitionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      std::set<PartitionNode*>::iterator finder = logical_nodes.find(inst);
      assert(finder != logical_nodes.end());
      logical_nodes.erase(finder);
#else
      logical_nodes.erase(inst);
#endif
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
        PendingChildArgs args;
        args.parent = this;
        args.pending_child = child_color;
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
    /*static*/void IndexPartNode::handle_disjointness_test(
             IndexPartNode *parent, IndexSpaceNode *left, IndexSpaceNode *right)
    //--------------------------------------------------------------------------
    {
      bool disjoint = !left->intersects_with(right);
      parent->record_disjointness(disjoint, left->color, right->color);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::send_node(AddressSpaceID target, bool up)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent != NULL);
#endif
      if (up)
        parent->send_node(target, true/*up*/);
      // Always send the color space ahead of this 
      color_space->send_node(target, false/*up*/);
      std::map<LegionColor,IndexSpaceNode*> valid_copy;
      {
        // Make sure we know if this is disjoint or not yet
        bool disjoint_result = is_disjoint();
        AutoLock n_lock(node_lock);
        if (!remote_instances.contains(target))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(did);
            rez.serialize(parent->handle); 
            rez.serialize(color_space->handle);
            rez.serialize(color);
            rez.serialize<bool>(disjoint_result);
            rez.serialize(partition_ready);
            rez.serialize(partial_pending);
            rez.serialize<size_t>(semantic_info.size());
            for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
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
          context->runtime->send_index_partition_node(target, rez);
          remote_instances.add(target);
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
      bool disjoint;
      derez.deserialize(disjoint);
      ApEvent ready_event;
      derez.deserialize(ready_event);
      ApUserEvent partial_pending;
      derez.deserialize(partial_pending);
      IndexSpaceNode *parent_node = context->get_node(parent);
      IndexSpaceNode *color_space_node = context->get_node(color_space);
#ifdef DEBUG_LEGION
      assert(parent_node != NULL);
      assert(color_space_node != NULL);
#endif
      IndexPartNode *node = context->create_node(handle, parent_node, 
        color_space_node, color, disjoint, did, ready_event, partial_pending);
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
        node->attach_semantic_information(tag, source,
                                          buffer, buffer_size, is_mutable);
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
      IndexPartNode *target = forest->get_node(handle);
      target->send_node(source, true/*up*/);
      Serializer rez;
      rez.serialize(to_trigger);
      forest->runtime->send_index_partition_return(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
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
      IndexSpace *target;
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
        DeferChildArgs args;
        args.proxy_this = parent;
        args.child_color = child_color;
        args.target = target;
        args.to_trigger = to_trigger;
        args.source = source;
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
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpace *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      (*target) = handle;
      Runtime::trigger_event(to_trigger);
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
                                   DistributedID did)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC), 
          get_owner_space(sp, ctx->runtime), false/*register with runtime*/),
        handle(sp), context(ctx), node_lock(this->gc_lock), destroyed(false)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        this->available_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
        local_field_infos.resize(runtime->max_local_fields);
      } 
#ifdef LEGION_GC
      log_garbage.info("GC Field Space %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, handle.id);
#endif
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx,
                                   DistributedID did, Deserializer &derez)
      : DistributedCollectable(ctx->runtime, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC), 
          get_owner_space(sp, ctx->runtime), false/*register with runtime*/),
        handle(sp), context(ctx), node_lock(this->gc_lock), destroyed(false)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        this->available_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
        local_field_infos.resize(runtime->max_local_fields);
      }
      size_t num_fields;
      derez.deserialize(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        derez.deserialize(fields[fid]);
      }
      size_t num_insts;
      derez.deserialize(num_insts);
      for (unsigned idx = 0; idx < num_insts; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        logical_trees.insert(handle);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Field Space %lld %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, handle.id);
#endif
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(const FieldSpaceNode &rhs)
      : DistributedCollectable(NULL, 0, 0), 
        handle(FieldSpace::NO_SPACE), context(NULL), node_lock(rhs.node_lock)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::~FieldSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      // First we can delete any instances that we have
      if (!local_trees.empty())
      {
        // Must make a copy since the data structure can change
        std::set<RegionNode*> to_delete = local_trees;
        for (std::set<RegionNode*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
          delete (*it);
#ifdef DEBUG_LEGION
        assert(local_trees.empty());
#endif
      }
      // Next we can delete our layouts
      for (std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
            LAYOUT_DESCRIPTION_ALLOC>::tracked>::iterator it =
            layouts.begin(); it != layouts.end(); it++)
      {
        LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::tracked
          &descs = it->second;
        for (LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::
              tracked::iterator it = descs.begin(); it != descs.end(); it++)
        {
          if ((*it)->remove_reference())
            delete (*it);
        }
      }
      layouts.clear();
      for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      }
      for (LegionMap<std::pair<FieldID,SemanticTag>,
            SemanticInfo>::aligned::iterator it = semantic_field_info.begin(); 
            it != semantic_field_info.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      }
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
    void FieldSpaceNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we add a valid reference to the owner
      if (!is_owner())
        send_remote_valid_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::DestructionFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      node->send_remote_gc_update(target, mutator, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        // Need read-only lock while accessing these structures, 
        // we know they won't change while accessing them
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        // Remove gc references from our remote nodes
        if (!!remote_instances)
        {
          DestructionFunctor functor(this, mutator);
          remote_instances.map(functor);
        }
      }
      else // Remove the valid reference that we have on the owner
        send_remote_valid_update(owner_space, mutator, 1/*count*/,false/*add*/);
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
                                                     bool is_mutable)
    //--------------------------------------------------------------------------
    {
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
            (source != owner_space))
          send_semantic_info(owner_space, tag, buffer, size, is_mutable);
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
                                                     bool is_mutable)
    //--------------------------------------------------------------------------
    {
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<std::pair<FieldID,SemanticTag>,
            SemanticInfo>::aligned::iterator finder =
          semantic_field_info.find(std::pair<FieldID,SemanticTag>(fid,tag));
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
            (source != owner_space))
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::const_iterator finder = 
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
      LegionMap<SemanticTag,SemanticInfo>::aligned::const_iterator finder = 
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
          SemanticInfo>::aligned::const_iterator finder = 
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
        SemanticInfo>::aligned::const_iterator finder = 
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
          SemanticRequestArgs args;
          args.proxy_this = this;
          args.tag = tag;
          args.source = source;
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
        LegionMap<std::pair<FieldID,SemanticTag>,SemanticInfo>::aligned::
          iterator finder = semantic_field_info.find(key);
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
          SemanticFieldRequestArgs args;
          args.proxy_this = this;
          args.fid = fid;
          args.tag = tag;
          args.source = source;
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
      forest->attach_semantic_information(handle, tag, source, 
                                          buffer, size, is_mutable);
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
      forest->attach_semantic_information(handle, fid, tag, 
                                          source, buffer, size, is_mutable);
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
    RtEvent FieldSpaceNode::allocate_field(FieldID fid, size_t size, 
                                           CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, send the request to the owner 
      if (!is_owner())
      {
        RtUserEvent allocated_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize<size_t>(1); // only allocating one field
          rez.serialize(fid);
          rez.serialize(size);
        }
        context->runtime->send_field_alloc_request(owner_space, rez);
        return allocated_event;
      }
      std::deque<AddressSpaceID> targets;
      unsigned index = 0;
      {
        // We're the owner so do the field allocation
        AutoLock n_lock(node_lock);
        if (fields.find(fid) != fields.end())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
            "Illegal duplicate field ID %d used by the "
                          "application in field space %d", fid, handle.id)
        // Find an index in which to allocate this field  
        int result = allocate_index();
        if (result < 0)
          REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
            "Exceeded maximum number of allocated fields for "
                          "field space %x. Change MAX_FIELDS from %d and "
                          "related macros at the top of legion_config.h and "
                          "recompile.", handle.id, MAX_FIELDS)
        index = result;
        fields[fid] = FieldInfo(size, index, serdez_id);
        if (!!remote_instances)
        {
          FindTargetsFunctor functor(targets);
          remote_instances.map(functor);   
        }
      }
      if (!targets.empty())
      {
        std::set<RtEvent> allocated_events;
        for (std::deque<AddressSpaceID>::const_iterator it = targets.begin();
              it != targets.end(); it++)
        {
          RtUserEvent done_event = Runtime::create_rt_user_event();
          allocated_events.insert(done_event);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(done_event);
            rez.serialize(serdez_id);
            rez.serialize<size_t>(1);
            rez.serialize(fid);
            rez.serialize(size);
            rez.serialize(index);
          }
          context->runtime->send_field_alloc_notification(*it, rez);
        }
        return Runtime::merge_events(allocated_events);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_fields(const std::vector<size_t> &sizes,
                                            const std::vector<FieldID> &fids,
                                            CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sizes.size() == fids.size());
#endif
      // If we're not the owner, send the request to the owner 
      if (!is_owner())
      {
        RtUserEvent allocated_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
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
      std::deque<AddressSpaceID> targets;
      std::vector<unsigned> indexes(fids.size());
      {
        // We're the owner so do the field allocation
        AutoLock n_lock(node_lock);
        for (unsigned idx = 0; idx < fids.size(); idx++)
        {
          FieldID fid = fids[idx];
          if (fields.find(fid) != fields.end())
            REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
              "Illegal duplicate field ID %d used by the "
                            "application in field space %d", fid, handle.id)
          // Find an index in which to allocate this field  
          int result = allocate_index();
          if (result < 0)
            REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS,
              "Exceeded maximum number of allocated fields for "
                            "field space %x. Change MAX_FIELDS from %d and "
                            "related macros at the top of legion_config.h and "
                            "recompile.", handle.id, MAX_FIELDS)
          unsigned index = result;
          fields[fid] = FieldInfo(sizes[idx], index, serdez_id);
          indexes[idx] = index;
        }
        if (!!remote_instances)
        {
          FindTargetsFunctor functor(targets);
          remote_instances.map(functor);   
        }
      }
      if (!targets.empty())
      {
        std::set<RtEvent> allocated_events;
        for (std::deque<AddressSpaceID>::const_iterator it = targets.begin();
              it != targets.end(); it++)
        {
          RtUserEvent done_event = Runtime::create_rt_user_event();
          allocated_events.insert(done_event);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(done_event);
            rez.serialize(serdez_id);
            rez.serialize<size_t>(fids.size());
            for (unsigned idx = 0; idx < fids.size(); idx++)
            {
              rez.serialize(fids[idx]);
              rez.serialize(sizes[idx]);
              rez.serialize(indexes[idx]);
            }
          }
          context->runtime->send_field_alloc_notification(*it, rez);
        }
        return Runtime::merge_events(allocated_events);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field(FieldID fid, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_owner() && (source != owner_space))
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(1);
          rez.serialize(fid);
        }
        context->runtime->send_field_free(owner_space, rez);
        return;
      }
      std::deque<AddressSpaceID> targets;
      {
        // We can actually do this with the read-only lock since we're
        // not actually going to change the allocation of the fields
        // data structure
        AutoLock n_lock(node_lock); 
        std::map<FieldID,FieldInfo>::iterator finder = fields.find(fid);
        finder->second.destroyed = true;
        if (is_owner())
          free_index(finder->second.idx);
        if (is_owner() && !!remote_instances)
        {
          FindTargetsFunctor functor(targets);
          remote_instances.map(functor);
        }
      }
      if (!targets.empty())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(1);
          rez.serialize(fid);
        }
        for (std::deque<AddressSpaceID>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
        {
          context->runtime->send_field_free(*it, rez);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_fields(const std::vector<FieldID> &to_free,
                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_owner() && (source != owner_space))
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(to_free.size());
          for (unsigned idx = 0; idx < to_free.size(); idx++)
            rez.serialize(to_free[idx]);
        }
        context->runtime->send_field_free(owner_space, rez);
        return;
      }
      std::deque<AddressSpaceID> targets;
      {
        // We can actually do this with the read-only lock since we're
        // not actually going to change the allocation of the fields
        // data structure
        AutoLock n_lock(node_lock); 
        for (std::vector<FieldID>::const_iterator it = to_free.begin();
              it != to_free.end(); it++)
        {
          std::map<FieldID,FieldInfo>::iterator finder = fields.find(*it);
          finder->second.destroyed = true;  
          if (is_owner())
            free_index(finder->second.idx);
        }
        if (is_owner() && !!remote_instances)
        {
          FindTargetsFunctor functor(targets);
          remote_instances.map(functor);
        }
      }
      if (!targets.empty())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(to_free.size());
          for (unsigned idx = 0; idx < to_free.size(); idx++)
            rez.serialize(to_free[idx]);
        }
        for (std::deque<AddressSpaceID>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
        {
          context->runtime->send_field_free(*it, rez);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::allocate_local_fields(
                                            const std::vector<FieldID> &fids,
                                            const std::vector<size_t> &sizes,
                                            CustomSerdezID serdez_id,
                                            const std::set<unsigned> &indexes,
                                            std::vector<unsigned> &new_indexes)
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
        assert(new_indexes.size() == fids.size());
#endif
        for (unsigned idx = 0; idx < fids.size(); idx++)
        {
          FieldID fid = fids[idx];
          fields[fid] = FieldInfo(sizes[idx], new_indexes[idx], serdez_id);
        }
      }
      else
      {
        // We're the owner so do the field allocation
        AutoLock n_lock(node_lock);
        if (!allocate_local_indexes(sizes, indexes, new_indexes))
          return false;
        for (unsigned idx = 0; idx < fids.size(); idx++)
        {
          FieldID fid = fids[idx];
          if (fields.find(fid) != fields.end())
            REPORT_LEGION_ERROR(ERROR_ILLEGAL_DUPLICATE_FIELD_ID,
              "Illegal duplicate field ID %d used by the "
                            "application in field space %d", fid, handle.id)
          fields[fid] = FieldInfo(sizes[idx], new_indexes[idx], serdez_id);
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
            fields.find(to_free[idx]);
#ifdef DEBUG_LEGION
          assert(finder != fields.end());
#endif
          finder->second.destroyed = true;  
        }
        // Now free the indexes
        free_local_indexes(indexes);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::update_local_fields(const std::vector<FieldID> &fids,
                                  const std::vector<size_t> &sizes,
                                  const std::vector<CustomSerdezID> &serdez_ids,
                                  const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(fids.size() == sizes.size());
      assert(fids.size() == serdez_ids.size());
      assert(fids.size() == indexes.size());
#endif
      AutoLock n_lock(node_lock);
      for (unsigned idx = 0; idx < fids.size(); idx++)
        fields[fids[idx]] = FieldInfo(sizes[idx], indexes[idx],serdez_ids[idx]);
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
          fields.find(to_remove[idx]);
#ifdef DEBUG_LEGION
        assert(finder != fields.end());
#endif
        finder->second.destroyed = true;  
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_alloc_notification(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      CustomSerdezID serdez_id;
      derez.deserialize(serdez_id);
      size_t num_fields;
      derez.deserialize(num_fields);
      AutoLock n_lock(node_lock);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        FieldInfo &info = fields[fid];
        derez.deserialize(info.field_size);
        derez.deserialize(info.idx);
        info.serdez_id = serdez_id;
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
      if (finder == fields.end())
        return false;
      // Make sure we haven't destroyed this field
      return (!finder->second.destroyed);
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::get_field_size(FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_LEGION
      assert(finder != fields.end());
#endif
      return finder->second.field_size;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_all_fields(std::vector<FieldID> &to_set)
    //--------------------------------------------------------------------------
    {
      to_set.clear();
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      to_set.reserve(fields.size());
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (!it->second.destroyed)
          to_set.push_back(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_all_regions(std::set<LogicalRegion> &regions)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      regions = logical_trees;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(const FieldMask &mask,
                                       std::set<FieldID> &to_set)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (it->second.destroyed)
          continue;
        if (mask.is_set(it->second.idx))
          to_set.insert(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(const FieldMask &mask,
                                       std::vector<FieldID> &to_set)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (it->second.destroyed)
          continue;
        if (mask.is_set(it->second.idx))
          to_set.push_back(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(const FieldMask &mask, 
                                       const std::set<FieldID> &basis,
                                       std::set<FieldID> &to_set)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      // Only iterate over the basis fields here
      for (std::set<FieldID>::const_iterator it = basis.begin();
            it != basis.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != fields.end());
#endif
        if (finder->second.destroyed)
          continue;
        if (mask.is_set(finder->second.idx))
          to_set.insert(finder->first);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on = add_instance(inst->handle, 
                                     context->runtime->address_space);
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(local_trees.find(inst) == local_trees.end());
#endif
        local_trees.insert(inst);
      }
      if (wait_on.exists())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::add_instance(LogicalRegion inst,
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      std::deque<AddressSpaceID> targets;
      {
        AutoLock n_lock(node_lock);
        // Check to see if we already have it, if we do then we are done
        if (logical_trees.find(inst) != logical_trees.end())
          return RtEvent::NO_RT_EVENT;
        logical_trees.insert(inst);
        if (is_owner())
        {
          if (!!remote_instances)
          {
            // Send messages to everyone except the source
            FindTargetsFunctor functor(targets);
            remote_instances.map(functor);
          }
        }
        else if (source != owner_space)
          targets.push_back(owner_space); // send the message to our owner
      }
      if (!targets.empty())
      {
        std::set<RtEvent> ready_events;
        for (std::deque<AddressSpaceID>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
        {
          if ((*it) == source)
            continue;
          RtUserEvent done = Runtime::create_rt_user_event();
          ready_events.insert(done);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(inst);
            rez.serialize(done);
          }
          context->runtime->send_field_space_top_alloc(*it, rez);
        }
        if (ready_events.empty())
          return RtEvent::NO_RT_EVENT;
        return Runtime::merge_events(ready_events);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_instance(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<LogicalRegion>::const_iterator it = 
            logical_trees.begin(); it != logical_trees.end(); it++)
      {
        if (it->get_tree_id() == tid)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::remove_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      std::set<RegionNode*>::iterator finder = local_trees.find(inst);
      assert(finder != local_trees.end());
      local_trees.erase(finder);
#else
      local_trees.erase(inst);
#endif
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (destroyed)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_DESTROY_FIELD_SPACE,
            "Duplicate deletion of Field Space %d", handle.id)
      destroyed = true;
      // If we're the owner, we can just remove the application valid
      // reference, otherwise if we're remote we do that
      if (!is_owner())
      {
        runtime->send_field_space_destruction(handle, owner_space);
        return false;
      }
      else
        return remove_base_valid_ref(APPLICATION_REF, NULL/*no mutator*/);
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(
                              const std::set<FieldID> &privilege_fields) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      FieldMask result;
      for (std::set<FieldID>::const_iterator it = privilege_fields.begin();
            it != privilege_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != fields.end());
#endif
        result.set_bit(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned FieldSpaceNode::get_field_index(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_LEGION
      assert(finder != fields.end());
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
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < needed.size(); idx++)
      {
      std::map<FieldID,FieldInfo>::const_iterator finder = 
        fields.find(needed[idx]);
#ifdef DEBUG_LEGION
        assert(finder != fields.end());
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
      std::map<unsigned/*mask index*/,unsigned/*layout index*/> index_map;
      {
        // Need to hold the lock when accessing field infos
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        for (unsigned idx = 0; idx < create_fields.size(); idx++)
        {
          FieldID fid = create_fields[idx];
          std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
	  // Catch unknown fields here for now
	  if (finder == fields.end())
            REPORT_LEGION_FATAL(LEGION_FATAL_UNKNOWN_FIELD_ID,
	      "unknown field ID %d requested during instance creation", fid)
          field_sizes[idx] = finder->second.field_size; 
          index_map[finder->second.idx] = idx;
          serdez[idx] = finder->second.serdez_id;
          mask.set_bit(finder->second.idx);
        }
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
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fids(num_fields);
      std::vector<size_t> sizes(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(fids[idx]);
        derez.deserialize(sizes[idx]);
      }
      FieldSpaceNode *node = forest->get_node(handle);
      RtEvent ready = node->allocate_fields(sizes, fids, serdez_id);
      Runtime::trigger_event(done, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_alloc_notification(
                                  RegionTreeForest *forest, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      FieldSpaceNode *node = forest->get_node(handle);
      node->process_alloc_notification(derez);
      Runtime::trigger_event(done); // indicate that we have been notified
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_top_alloc(RegionTreeForest *forest,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      LogicalRegion top_handle;
      derez.deserialize(top_handle);
      RtUserEvent done;
      derez.deserialize(done);
      FieldSpaceNode *node = forest->get_node(handle);
      RtEvent ready = node->add_instance(top_handle, source);
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
      FieldSpaceNode *node = forest->get_node(handle);
      node->free_fields(fields, source);
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
                                      current_indexes, new_indexes))
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
    InstanceManager* FieldSpaceNode::create_external_instance(
                                         const std::vector<FieldID> &field_set,
                                         RegionNode *node, AttachOp *attach_op)
    //--------------------------------------------------------------------------
    {
      std::vector<size_t> field_sizes(field_set.size());
      std::vector<unsigned> mask_index_map(field_set.size());
      std::vector<CustomSerdezID> serdez(field_set.size());
      FieldMask file_mask;
      compute_field_layout(field_set, field_sizes, 
                           mask_index_map, serdez, file_mask);
      // Now make the instance, this should always succeed
      ApEvent ready_event;
      LayoutConstraintSet constraints;
      PhysicalInstance inst = 
        attach_op->create_instance(node->row_source, field_set, field_sizes, 
                                   constraints, ready_event);
      // Pull out the pointer constraint so that we can use it separately
      // and not have it included in the layout constraints
      PointerConstraint pointer_constraint = constraints.pointer_constraint;
      constraints.pointer_constraint = PointerConstraint();
      const unsigned total_dims = node->row_source->get_num_dims();
      // Get the layout
      LayoutDescription *layout = 
        find_layout_description(file_mask, total_dims, constraints);
      if (layout == NULL)
      {
        LayoutConstraints *layout_constraints = 
          context->runtime->register_layout(handle, constraints);
        layout = create_layout_description(file_mask, total_dims,
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
      InstanceManager *result = new InstanceManager(context, did, 
                                         context->runtime->address_space,
                                         memory, inst, node->row_source, 
                                         false/*own*/, node, layout, 
                                         pointer_constraint,
                                         true/*register now*/, ready_event,
                                         true/*external instance*/,
                                         Reservation::create_reservation());
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
          LAYOUT_DESCRIPTION_ALLOC>::tracked>::const_iterator finder = 
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
        LAYOUT_DESCRIPTION_ALLOC>::tracked>::const_iterator finder = 
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
      LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::tracked
        &descs = layouts[hash_key];
      if (!descs.empty())
      {
        for (LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::tracked
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
      // See if this is in our creation set, if not, send it and all the fields
      AutoLock n_lock(node_lock);
      if (!remote_instances.contains(target))
      {
        // First send the node info and then send all the fields
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(did);
          // Pack the field infos
          size_t num_fields = fields.size();
          rez.serialize<size_t>(num_fields);
          for (std::map<FieldID,FieldInfo>::const_iterator it = 
                fields.begin(); it != fields.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          size_t num_instances = logical_trees.size();
          rez.serialize<size_t>(num_instances);
          for (std::set<LogicalRegion>::const_iterator it = 
                logical_trees.begin(); it != logical_trees.end(); it++)
            rez.serialize(*it);
          rez.serialize<size_t>(semantic_info.size());
          for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
                semantic_info.begin(); it != semantic_info.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second.size);
            rez.serialize(it->second.buffer, it->second.size);
            rez.serialize(it->second.is_mutable);
          }
          rez.serialize<size_t>(semantic_field_info.size());
          for (LegionMap<std::pair<FieldID,SemanticTag>,
                SemanticInfo>::aligned::iterator
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
        remote_instances.add(target);
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
      FieldSpaceNode *node = context->create_node(handle, did, derez);
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
        node->attach_semantic_information(tag, source, 
                                          buffer, buffer_size, is_mutable);
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
        node->attach_semantic_information(fid, tag, source,
                                          buffer, buffer_size, is_mutable);
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
    char* FieldSpaceNode::to_string(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      char *result = (char*)malloc(MAX_FIELDS*4); 
      bool first = true;
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (mask.is_set(it->second.idx))
        {
          if (first)
          {
            sprintf(result,"%d",it->first);
            first = false;
          }
          else
          {
            char temp[8];
            sprintf(temp,",%d",it->first);
            strcat(result, temp);
          }
        }
      }
#ifdef DEBUG_LEGION
      assert(!first); // we should have written something
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_ids(const FieldMask &mask,
                                       std::vector<FieldID> &field_ids) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (mask.is_set(it->second.idx))
        {
          field_ids.push_back(it->first);
        }
      }
#ifdef DEBUG_LEGION
      assert(!field_ids.empty()); // we should have found something
#endif
    }

    //--------------------------------------------------------------------------
    int FieldSpaceNode::allocate_index(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      int result = available_indexes.find_first_set();
      if (result >= 0)
      {
        // If we have slots for local fields then we can't use those
        if (result >= int(MAX_FIELDS - runtime->max_local_fields))
          return -1;
        available_indexes.unset_bit(result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_index(unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(!available_indexes.is_set(index));
#endif
      // Assume we are already holding the node lock
      available_indexes.set_bit(index);
      // We also need to invalidate all our layout descriptions
      // that contain this field
      std::vector<LEGION_FIELD_MASK_FIELD_TYPE> to_delete;
      for (std::map<LEGION_FIELD_MASK_FIELD_TYPE,LegionList<LayoutDescription*,
                  LAYOUT_DESCRIPTION_ALLOC>::tracked>::iterator lit = 
            layouts.begin(); lit != layouts.end(); lit++)
      {
        // If the bit is set, remove the layout descriptions
        if (lit->first & (1ULL << index))
        {
          LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::tracked
            &descs = lit->second;
          bool perform_delete = true;
          for (LegionList<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::
                tracked::iterator it = descs.begin(); 
                it != descs.end(); /*nothing*/)
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
      {
        layouts.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::allocate_local_indexes(
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
        unsigned global_idx = MAX_FIELDS - runtime->max_local_fields;
        for (unsigned local_idx = 0; 
              local_idx < local_field_infos.size(); local_idx++, global_idx++)
        {
          // If it's already been allocated in this context then
          // we can't use it
          if (current_indexes.find(global_idx) != current_indexes.end())
            continue;
          LocalFieldInfo &info = local_field_infos[local_idx];
          // Check if the current local field index is used
          if (info.count > 0)
          {
            // Already in use, check to see if the field sizes are the same
            if (info.size == field_size)
            {
              // Same size so we can use it
              info.count++;
              chosen_index = global_idx;
              break;
            }
            // Else different field size means we can't reuse it
          }
          else
          {
            // Not in use, so we can assign the size and make
            // ourselves the first user
            info.size = field_size;
            info.count = 1;
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

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_local_indexes(
                                           const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        // Translate back to a local field index
#ifdef DEBUG_LEGION
        assert(indexes[idx] >= (MAX_FIELDS - runtime->max_local_fields));
#endif
        const unsigned local_index = 
          indexes[idx] - (MAX_FIELDS - runtime->max_local_fields);
#ifdef DEBUG_LEGION
        assert(local_index < local_field_infos.size());
#endif
        LocalFieldInfo &info = local_field_infos[local_index];
#ifdef DEBUG_LEGION
        assert(info.count > 0);
#endif
        info.count--; // decrement the count, if it is zero we can clear it
        if (info.count == 0)
          info.size = 0;
      }
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeNode::RegionTreeNode(RegionTreeForest *ctx, 
                                   FieldSpaceNode *column_src)
      : DistributedCollectable(ctx->runtime, 
            LEGION_DISTRIBUTED_HELP_ENCODE(
              ctx->runtime->get_available_distributed_id(),
              REGION_TREE_NODE_DC),
            ctx->runtime->address_space, false/*register with runtime*/),
        context(ctx), column_source(column_src), 
        registered(false), destroyed(false), 
#ifdef DEBUG_LEGION
        currently_active(true),
#endif
        node_lock(this->gc_lock)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::~RegionTreeNode(void)
    //--------------------------------------------------------------------------
    {
      remote_instances.clear();
      for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
            semantic_info.begin(); it != semantic_info.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer, it->second.size);
      }
      // Notify any contexts that were tracking us that we are now deleted
      if (!tracking_contexts.empty())
      {
        for (std::set<InnerContext*>::const_iterator it = 
              tracking_contexts.begin(); it != tracking_contexts.end(); it++)
          (*it)->notify_region_tree_node_deletion(this);
      }
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
                                                     bool is_mutable)
    //--------------------------------------------------------------------------
    {
      // Make a copy
      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      {
        AutoLock n_lock(node_lock); 
        // See if it already exists
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
            (source != owner_space))
        {
          send_semantic_info(owner_space, tag, buffer, size, is_mutable); 
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::const_iterator finder = 
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
      LegionMap<SemanticTag,SemanticInfo>::aligned::const_iterator finder = 
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
    void RegionTreeNode::initialize_logical_state(ContextID ctx,
                                                  const FieldMask &init_dirty)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      state.dirty_fields |= init_dirty;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_user(ContextID ctx, 
                                               const LogicalUser &user,
                                               RegionTreePath &path,
                                               const TraceInfo &trace_info,
                                               VersionInfo &version_info,
                                               ProjectionInfo &proj_info,
                                               FieldMask &unopened_field_mask,
                           LegionMap<AdvanceOp*,LogicalUser>::aligned &advances)
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
      LegionColor next_child = INVALID_COLOR;
      if (!arrived)
        next_child = path.get_child(depth);
      // Now check to see if we need to do any close operations
      if (!!unopened_field_mask)
      {
        // Close up any children which we may have dependences on below
        const bool captures_closes = true;
        LogicalCloser closer(ctx, user, this, 
                             arrived/*validates*/, captures_closes);
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
              aliased_children, captures_closes, next_child, open_below);
        }
        // We always need to create and register close operations
        // regardless of whether we are tracing or not
        // If we're not replaying a trace we need to do work here
        // See if we need to register a close operation
        if (closer.has_close_operations())
        {
          // Generate the close operations         
          closer.initialize_close_operations(state, user.op, 
                                             version_info, trace_info);
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
            add_open_field_state(state, arrived, proj_info, 
                                 user, open_only, next_child);
        }
      }
      else if (!arrived || proj_info.is_projecting())
      {
        // Everything is open-only so make a state and merge it in
        add_open_field_state(state, arrived, proj_info, 
                             user, user.field_mask, next_child);
      }
      // Perform dependence analysis for any of our advance operations
      // since we know we will depend on them so anything they depend
      // on then we will also automatically depend on us as well
      if (!advances.empty())
      {
        for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator it =
              advances.begin(); it != advances.end(); it++)
          perform_advance_analysis(ctx, state, it->first, it->second, 
                                   user, next_child, trace_info.already_traced);
      }
      // See if we need to perform any advance operations
      // because we're going to be writing below this level
      if (HAS_WRITE(user.usage) && (!arrived || 
            (proj_info.is_projecting() && 
             ((proj_info.projection_type == PART_PROJECTION) || 
              (proj_info.projection->depth > 0)))))
      {
        FieldMask advance_mask = user.field_mask - state.dirty_below; 
        if (!!advance_mask)
        {
          // These fields are now dirty below
          if (!arrived || proj_info.is_projecting())
            state.dirty_below |= advance_mask;
          // Now see which ones are handled from advances from above
          for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator it =
                advances.begin(); it != advances.end(); it++)
          {
            advance_mask -= it->second.field_mask;
            if (!advance_mask)
              break;
          }
          // If we still have fields, then we have to make a new advance
          if (!!advance_mask)
            create_logical_advance(ctx, state, advance_mask, user, trace_info, 
                advances, version_info.is_upper_bound_node(this), next_child);
        }
      }
      else if (arrived && IS_REDUCE(user.usage))
      {
        // Special case
        // If we are the first reduction to dirty fields at this level
        // we have to advance as well because we need our version 
        // state objects to register as dirty with their parent, there
        // might be multiple reductions that need this so we make an
        // advance operation to handle it for all users. We can skip
        // this if some other user has already written or reduced
        FieldMask advance_mask = 
          user.field_mask - (state.dirty_fields | state.reduction_fields);
        if (!!advance_mask)
        {
          // See which ones we've handled from above
          for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator it =
                advances.begin(); it != advances.end(); it++)
          {
            advance_mask -= it->second.field_mask;
            if (!advance_mask)
              break;
          }
          // If we still have fields, then we have to make a new advance
          if (!!advance_mask)
            create_logical_advance(ctx, state, advance_mask, user, trace_info,
                advances, version_info.is_upper_bound_node(this), next_child);
        }
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
        filter_curr_epoch_users(state, dominator_mask);
      }
      if (arrived)
      { 
        // If we've arrived add ourselves as a user
        register_local_user(state, user, trace_info);
        // Finish dependence analysis for any advance operations
        // Do this before updating the dirty and reduction fields at this level
        if (!advances.empty())
        {
          // Three cases
          if (proj_info.is_projecting() && ((proj_info.projection->depth > 0)
                || (proj_info.projection_type == PART_PROJECTION)))
          {
            // We go to this node
            for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator it =
                  advances.begin(); it != advances.end(); it++)
            {
              it->first->set_child_node(this);
              it->first->end_dependence_analysis();
            }
          }
          else if (IS_REDUCE(user.usage) && (!proj_info.is_projecting() ||
                // Special case for depth 0 region projections
                ((proj_info.projection_type == REG_PROJECTION) && 
                 (proj_info.projection->depth == 0))))
          {
            // Figure out which advances need to come to this level
            FieldMask advance_here =
              user.field_mask - (state.dirty_fields | state.reduction_fields);
            RegionTreeNode *parent = get_parent();
            if (parent == NULL)
            {
              // If parent is NULL, then we just advance ourselves
              for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator 
                    it = advances.begin(); it != advances.end(); it++)
              {
                it->first->set_child_node(this);
                it->first->end_dependence_analysis();
              }
            }
            else if (!!advance_here)
            {
              for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator
                    it = advances.begin(); it != advances.end(); it++)
              {
                FieldMask split_mask = it->second.field_mask & advance_here;
                if (!!split_mask)
                {
                  // Everything to the child, record split fields if necessary
                  it->first->set_child_node(this);
                  if (split_mask != it->second.field_mask)
                    it->first->set_split_child_mask(split_mask);
                }
                else // everything to the parent
                  it->first->set_child_node(parent);
                it->first->end_dependence_analysis();
              }
            }
            else
            {
              // Everything to the parent
              for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator 
                    it = advances.begin(); it != advances.end(); it++)
              {
                it->first->set_child_node(parent);
                it->first->end_dependence_analysis();
              }
            }
          }
          else
          {
            // Common case, we go to our parent
            RegionTreeNode *parent = get_parent();
#ifdef DEBUG_LEGION
            assert(parent != NULL);
#endif
            for (LegionMap<AdvanceOp*,LogicalUser>::aligned::const_iterator it =
                  advances.begin(); it != advances.end(); it++)
            {
              it->first->set_child_node(parent); 
              it->first->end_dependence_analysis();
            }
          }
        }
        // Handle any projections that we might have
        // If we're projecting, record any split fields and projection epochs
        if (proj_info.is_projecting())
        {
          FieldMask split_fields = state.dirty_below & user.field_mask;
          if (!!split_fields)
          {
            version_info.record_split_fields(this, split_fields);
            // Also record split versions for any levels below
            // down to the level above our projection function
            if (proj_info.projection->depth > 1)
            {
              for (int idx = 0; 
                    idx < (proj_info.projection->depth - 1); idx++)
                version_info.record_split_fields(this, split_fields, idx);
            }
          }
          // Record our projection epochs and split masks
          state.capture_projection_epochs(user.field_mask, proj_info); 
          // If we're writing, then record our projection info in 
          // the current projection epoch
          if (IS_WRITE(user.usage))
            state.update_projection_epochs(user.field_mask, proj_info);
        }
        else if (user.usage.redop > 0)
        {
          // Not projecting and doing a reduction of some kind so record it
          record_logical_reduction(state, user.usage.redop, user.field_mask);
        }
        else if (IS_WRITE(user.usage))
        {
          // If we're not projecting and writing, indicating that
          // we have written this logical region
          state.dirty_fields |= user.field_mask;
        }
      }
      else 
      {
        // Haven't arrived, yet, so keep going down the tree
        RegionTreeNode *child = get_tree_child(next_child);
        // If we have any split fields we have to record them
        FieldMask split_fields = state.dirty_below & user.field_mask;
        if (!!split_fields)
          version_info.record_split_fields(this, split_fields);
        // Get our set of fields which are being opened for
        // the first time at the next level
        if (!!unopened_field_mask)
        {
          if (!!open_below)
          {
            FieldMask open_next = unopened_field_mask - open_below;
            if (!!open_next)
              child->create_logical_open(ctx, open_next, user, 
                                         path, trace_info);
            // Update our unopened children mask
            // to reflect any fields which are still open below
            unopened_field_mask &= open_below;
          }
          else
          {
            // Open all the unopened fields
            child->create_logical_open(ctx, unopened_field_mask, 
                                       user, path, trace_info);
            unopened_field_mask.clear();
          }
        }
#ifdef DEBUG_LEGION
        else // if they weren't open here, they shouldn't be open below
          assert(!open_below);
#endif
        child->register_logical_user(ctx, user, path, trace_info, version_info,
                                     proj_info, unopened_field_mask, advances);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::create_logical_open(ContextID ctx,
                                             const FieldMask &open_mask,
                                             const LogicalUser &creator,
                                             const RegionTreePath &path,
                                             const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      OpenOp *open = context->runtime->get_available_open_op(); 
      open->initialize(open_mask, this, path, trace_info,
                       creator.op, creator.idx);
      // Add it to the list of current users
      LogicalUser open_user(open, 0/*idx*/,
            RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), open_mask);
      open->add_mapping_reference(open_user.gen);
      // Register dependences against anything else that the creator 
      // operation has_dependences on 
      open->begin_dependence_analysis();
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned &above_users =
        creator.op->get_logical_records();
      perform_dependence_checks<LOGICAL_REC_ALLOC,
            false/*record*/, false/*has skip*/, false/*track dom*/>(
                open_user, above_users, open_mask, 
                open_mask/*doesn't matter*/, false/*validates*/);
      open->end_dependence_analysis();
      // Now add this to the current list
      state.curr_epoch_users.push_back(open_user);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::create_logical_advance(ContextID ctx, 
                                                LogicalState &state,
                                                const FieldMask &advance_mask,
                                                const LogicalUser &creator,
                                                const TraceInfo &trace_info,
                           LegionMap<AdvanceOp*,LogicalUser>::aligned &advances,
                                                bool parent_is_upper_bound,
                                                const LegionColor next_child) 
    //--------------------------------------------------------------------------
    {
      // These are the fields for which we have to issue an advance op
      AdvanceOp *advance = 
        context->runtime->get_available_advance_op(); 
      advance->initialize(this, advance_mask, trace_info, creator.op, 
                          creator.idx, parent_is_upper_bound);
      LogicalUser advance_user(advance, 0/*idx*/, 
          RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), advance_mask);
      // Add a mapping dependence so we can add it later
      advance->add_mapping_reference(advance_user.gen);
      // Start our dependence analysis, we'll finish it when we
      // reach our destination
      advance->begin_dependence_analysis();
      // Perform analysis against everything the creating operation had
      // dependences on as well
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned &above_users = 
        creator.op->get_logical_records();
      if (!above_users.empty())
        perform_dependence_checks<LOGICAL_REC_ALLOC,
          false/*record*/, false/*has skip*/, false/*track dom*/>(
              advance_user, above_users, advance_mask,
              advance_mask/*doesn't matter*/, false/*validates*/);
      // Also perform dependence analysis against any other advance
      // operations that occurred above in the tree, these would only 
      // apply to advances generated by aliased but non-interfering
      // region requirements as we will only issue one advance per
      // field for a given region requirement
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned &above_advances =
        creator.op->get_logical_advances();
      if (!above_advances.empty())
        perform_dependence_checks<LOGICAL_REC_ALLOC,
          false/*record*/, false/*has skip*/, false/*track dom*/>(
          advance_user, above_advances, advance_mask,
          advance_mask/*doesn't matter*/, false/*validates*/);
      // Now perform our local dependence analysis for this advance
      perform_advance_analysis(ctx, state, advance, advance_user, creator, 
                               next_child, trace_info.already_traced);
      // Update our list of advance operations
      advances[advance] = advance_user;
      // Add it to the list of current epoch users even if we're tracing 
      state.curr_epoch_users.push_back(advance_user);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_local_user(LogicalState &state,
                                             const LogicalUser &user,
                                             const TraceInfo &trace_info)
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
                                              ProjectionInfo &proj_info,
                                              const LogicalUser &user,
                                              const FieldMask &open_mask,
                                              const LegionColor next_child)
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
      else if (next_child != INVALID_COLOR)
      {
        FieldState new_state(user, open_mask, next_child);
        merge_new_field_state(state, new_state);
      }
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::traverse_advance_analysis(ContextID ctx, 
                                                   AdvanceOp *advance_op,
                                                const LogicalUser &advance_user,
                                                const LogicalUser &create_user)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      const LegionColor empty_next_child = INVALID_COLOR;
      perform_advance_analysis(ctx, state, advance_op, advance_user, 
          create_user, empty_next_child, false/*traced*/, false/*root*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_advance_analysis(ContextID ctx, 
                                                  LogicalState &state,
                                                  AdvanceOp *advance_op,
                                                const LogicalUser &advance_user,
                                                const LogicalUser &create_user,
                                                const LegionColor next_child,
                                                const bool already_traced,
                                                const bool advance_root)
    //--------------------------------------------------------------------------
    {
      // See if we have any previous dirty data to record at this depth
      // Only have to do this at the root of the advance
      if (advance_root)
      {
        FieldMask previous_dirty = advance_user.field_mask & 
                                  (state.dirty_fields | state.reduction_fields);
        if (!!previous_dirty)
          advance_op->record_dirty_previous(get_depth(), previous_dirty);
      }
      // Advances are going to bump the version number so we have to get
      // mapping dependences on any operations which might need to read
      // the current version number including those that are in disjoint
      // trees from what we might be traversing. We can skip the next child
      // if there is one because we're going to traverse that subtree later 
      // No need to traverse those sub-trees if we are tracing though because
      // we know there is nothing
      if (!already_traced)
      {
        const bool has_next_child = (next_child != INVALID_COLOR);
        for (LegionList<FieldState>::aligned::const_iterator fit = 
             state.field_states.begin(); fit != state.field_states.end(); fit++)
        {
          const FieldMask overlap = advance_user.field_mask & fit->valid_fields;
          if (!overlap)
            continue;
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it = 
               fit->open_children.begin(); it != fit->open_children.end(); it++)
          {
            if (has_next_child && (it->first == next_child))
              continue;
            const FieldMask child_overlap = it->second & overlap;
            if (!child_overlap)
              continue;
            RegionTreeNode *child = get_tree_child(it->first);
            child->traverse_advance_analysis(ctx, advance_op, 
                                             advance_user, create_user);
          }
        }
      }
      // We know we are dominating from above, so we don't care about
      // open below fields for advance analysis
      FieldMask empty_below;
      FieldMask dominator_mask = 
            perform_dependence_checks<CURR_LOGICAL_ALLOC,
                    false/*record*/, true/*has skip*/, true/*track dom*/>(
              advance_user, state.curr_epoch_users, advance_user.field_mask,
              empty_below, false/*validates*/, create_user.op, create_user.gen);
      FieldMask non_dominated_mask = advance_user.field_mask - dominator_mask;
      if (!!non_dominated_mask)
      {
        perform_dependence_checks<PREV_LOGICAL_ALLOC,
                    false/*record*/, true/*has skip*/, false/*track dom*/>(
              advance_user, state.prev_epoch_users, non_dominated_mask,
              empty_below, false/*validates*/, create_user.op, create_user.gen);
      }
      // Can't filter here because we need our creator to record
      // the same dependences in case it has to generate closes or opens later
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_logical_node(LogicalCloser &closer,
                                            const FieldMask &closing_mask,
                                            bool read_only_close)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_CLOSE_LOGICAL_NODE_CALL);
      LogicalState &state = get_logical_state(closer.ctx);
      // Perform closing checks on both the current epoch users
      // as well as the previous epoch users
      perform_closing_checks<CURR_LOGICAL_ALLOC>(closer, read_only_close,
                                     state.curr_epoch_users, closing_mask);
      perform_closing_checks<PREV_LOGICAL_ALLOC>(closer, read_only_close,
                                     state.prev_epoch_users, closing_mask);
      // If this is not a read-only close, then capture the 
      // close information
      ClosedNode *closed_node = NULL;
      if (!read_only_close)
      {
        // Always make the node so that we can know it was open
        closed_node = closer.find_closed_node(this);
        // We only fields that we've actually fully written
        if (!!state.dirty_fields)
          closed_node->record_closed_fields(closing_mask & state.dirty_fields);
      }
      // Recursively traverse any open children and close them as well
      for (std::list<FieldState>::iterator it = state.field_states.begin();
            it != state.field_states.end(); /*nothing*/)
      {
        FieldMask overlap = it->valid_fields & closing_mask;
        if (!overlap)
        {
          it++;
          continue;
        }
        if (it->is_projection_state())
        {
          // If this was a projection field state then we need to 
          // advance the epoch version numbers
          if (!read_only_close && (it->open_state != OPEN_READ_ONLY_PROJ))
          {
#ifdef DEBUG_LEGION
            assert(closed_node != NULL);
#endif
            state.capture_close_epochs(overlap, closed_node);
          }
          // If this is a writing or reducing 
          state.advance_projection_epochs(overlap);
          it->valid_fields -= overlap;
        }
        else
        {
          // Recursively perform any close operations
          FieldMask already_open;
          perform_close_operations(closer, overlap, *it,
                                   INVALID_COLOR/*next child*/,
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
        if (!it->valid_fields)
          it = state.field_states.erase(it);
        else
          it++;
      }
      // We can clear out our dirty fields 
      if (!!state.dirty_fields)
        state.dirty_fields -= closing_mask;
      // We can clear out our reduction fields
      if (!!state.reduction_fields)
        state.reduction_fields -= closing_mask;
      // We can also mark that there is no longer any dirty data below
      state.dirty_below -= closing_mask;
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
                                              const LegionColor next_child,
                                              FieldMask &open_below)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_SIPHON_LOGICAL_CHILDREN_CALL);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      LegionDeque<FieldState>::aligned new_states;
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
        (next_child == INVALID_COLOR) && !closer.user.op->is_predicated_op();
      // Now we can look at all the children
      for (LegionList<FieldState>::aligned::iterator it = 
            state.field_states.begin(); it != 
            state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields * current_mask)
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
                if (next_child != INVALID_COLOR)
                {
                  LegionMap<LegionColor,FieldMask>::aligned::const_iterator 
                    finder = it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Remove the child's open fields from the
                    // list of fields we need to open
                    open_below |= finder->second;
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
                  new_states.push_back(
                      FieldState(closer.user, already_open, next_child));
                // See if there are still any valid open fields
                if (!it->valid_fields)
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
              if (!it->valid_fields)
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_SINGLE_REDUCE:
            {
              // Check to see if we have a child we want to go down
              if (next_child != INVALID_COLOR)
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
                  std::vector<LegionColor> to_delete;
                  // Go through all the children and see if there is any overlap
                  for (LegionMap<LegionColor,FieldMask>::aligned::iterator 
                        cit = it->open_children.begin(); cit !=
                        it->open_children.end(); cit++)
                  {
                    FieldMask already_open = cit->second & current_mask;
                    // If disjoint children, nothing to do
                    if (!already_open || 
                        are_children_disjoint(cit->first, next_child))
                      continue;
                    // Case 2
                    if (cit->first != (next_child))
                    {
                      // Different child so we need to create a new
                      // FieldState in MULTI_REDUCE mode with two
                      // children open
                      FieldState new_state(closer.user,already_open,cit->first);
                      // Add the next child as well
                      new_state.open_children[next_child] = 
                        already_open;
                      new_state.open_state = OPEN_MULTI_REDUCE;
#ifdef DEBUG_LEGION
                      assert(!!new_state.valid_fields);
#endif
                      new_states.push_back(new_state);
                      // Update the current child, mark that we need to
                      // recompute the valid fields for the state
                      cit->second -= already_open;
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
                    for (std::vector<LegionColor>::const_iterator cit = 
                          to_delete.begin(); cit != to_delete.end(); cit++)
                    {
                      LegionMap<LegionColor,FieldMask>::aligned::iterator 
                        finder = it->open_children.find(*cit);
#ifdef DEBUG_LEGION
                      assert(finder != it->open_children.end());
                      assert(!finder->second);
#endif
                      it->open_children.erase(finder);
                    }
                    // Then recompute the valid mask for the current state
                    FieldMask new_valid_mask;
                    for (LegionMap<LegionColor,FieldMask>::aligned::
                          const_iterator cit = it->open_children.begin(); 
                          cit != it->open_children.end(); cit++)
                    {
#ifdef DEBUG_LEGION
                      assert(!!cit->second);
#endif
                      new_valid_mask |= cit->second;
                    }
                    // Update the valid mask on the field state, we'll
                    // check to see fi we need to delete it at the end
                    it->valid_fields = new_valid_mask;
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
                    FieldState new_state(closer.user, already_open, next_child);
                    // We always have to go to read-write mode here
                    new_state.open_state = OPEN_READ_WRITE;
                    new_states.push_back(new_state);
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
              if (!it->valid_fields)
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
                if (next_child != INVALID_COLOR)
                {
                  LegionMap<LegionColor,FieldMask>::aligned::const_iterator
                    finder = it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Already open, so add the open fields
                    open_below |= (finder->second & current_mask);
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
                if (!it->valid_fields)
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
              if (!IS_READ_ONLY(closer.user.usage) ||
                  (next_child != INVALID_COLOR))
              {
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields;
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_read_only_close(overlap, true/*projection*/);
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
                }
                it->valid_fields -= current_mask;
                if (!it->valid_fields)
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
                const FieldMask overlap = current_mask & it->valid_fields;
#ifdef DEBUG_LEGION
                assert(!!overlap);
#endif
                if (overwriting)
                  closer.record_overwriting_close(overlap, true/*projection*/);
                else
                {
                  closer.record_close_operation(overlap, true/*projection*/);
                  state.capture_close_epochs(overlap,
                                             closer.find_closed_node(this));
                }
                // Advance the projection epochs
                state.advance_projection_epochs(overlap);
              }
              it->valid_fields -= current_mask;
              if (!it->valid_fields)
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
              if (!IS_REDUCE(closer.user.usage) || 
                  (next_child != INVALID_COLOR) ||
                  (closer.user.usage.redop != it->redop))
              {
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields;
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  if (overwriting)
                    closer.record_overwriting_close(overlap,true/*projection*/);
                  else
                  {
                    closer.record_close_operation(overlap, true/*projection*/);
                    state.capture_close_epochs(overlap,
                                               closer.find_closed_node(this));
                  }
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
                }
                it->valid_fields -= current_mask;
                if (!it->valid_fields)
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
      if ((next_child != INVALID_COLOR) && !!open_mask)
        new_states.push_back(FieldState(closer.user, open_mask, next_child));
      merge_new_field_states(state, new_states);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif 
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_logical_projection(LogicalCloser &closer,
                                                LogicalState &state,
                                                const FieldMask &current_mask,
                                                ProjectionInfo &proj_info,
                                                bool record_close_operations,
                                                FieldMask &open_below)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_SIPHON_LOGICAL_PROJECTION_CALL);
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
      LegionDeque<FieldState>::aligned new_states;
      // First let's see if we need to flush any reductions
      LegionColor no_next_child = INVALID_COLOR; // never a next child here
      if (!!state.reduction_fields)
      {
        FieldMask reduction_flush_fields = 
          current_mask & state.reduction_fields;
        if (!!reduction_flush_fields)
          flush_logical_reductions(closer, state, reduction_flush_fields,
                       record_close_operations, no_next_child,new_states);
      }
      // Figure out now if we can do disjoint closes at this level of
      // the region tree. This applies to when we close directly to a
      // disjoint partition in the region tree (only happens with
      // projection functions). The rule for projection analysis is that 
      // dirty data can only live at the leaves of the open tree. Therefore
      // we must either being going into a disjoint shallow mode or any
      // mode which is read only that permits us to go to disjoint shallow
      const bool disjoint_close = !is_region() && are_all_children_disjoint() &&
        (IS_READ_ONLY(closer.user.usage) || (proj_info.projection->depth == 0));
      // Now we can look at all the children
      for (LegionList<FieldState>::aligned::iterator it = 
            state.field_states.begin(); it != 
            state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields * current_mask)
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
              if (!it->valid_fields)
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
              if (!it->valid_fields)
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
                open_below |= (it->valid_fields & current_mask);
                // Keep going
                it++;
              }
              else
              {
                // Otherwise we are going to a different mode 
                // no need to do a close since we're staying in
                // projection mode, but we do need to advance
                // the projection epochs.
                it->valid_fields -= current_mask;
                if (!it->valid_fields)
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
                  const FieldMask overlap = it->valid_fields & current_mask;
                  // Record that some fields are already open
                  open_below |= overlap;
                  // Make the new state to add
                  new_states.push_back(FieldState(closer.user.usage, overlap, 
                           proj_info.projection, proj_info.projection_space, 
                           are_all_children_disjoint(), true/*dirty reduce*/));
                  // If we are a reduction, we can go straight there
                  it->valid_fields -= overlap;
                  if (!it->valid_fields)
                    it = state.field_states.erase(it);
                  else
                    it++;
                }
                else
                {
                  // Update the domain  
                  it->projection_space = proj_info.projection_space;
                  open_below |= (it->valid_fields & current_mask);
                  it++;
                }
              }
              else
              {
                // Now we need the close operation
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields;
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap, true/*projection*/,
                                                disjoint_close);
                  state.capture_close_epochs(overlap,
                                             closer.find_closed_node(this));
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
                  // If we are doing a disjoint close, update the open
                  // states with the appropriate new state
                  if (disjoint_close)
                  {
                    // Record the fields are already open
                    open_below |= overlap;
                    // Make a new state with the default projection
                    // function to indicate that we are open in 
                    // disjoint shallow mode with read-write
                    RegionUsage close_usage(READ_WRITE, EXCLUSIVE, 0);
                    IndexSpaceNode *color_space = 
                      as_partition_node()->row_source->color_space;
                    new_states.push_back(FieldState(close_usage, overlap,
                        context->runtime->find_projection_function(0),
                        color_space, true/*disjoint*/));
                  }
                }
                it->valid_fields -= current_mask;
                if (!it->valid_fields)
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
                const FieldMask overlap = it->valid_fields & current_mask;
                // Record that some fields are already open
                open_below |= overlap;
                // Make the new state to add
                new_states.push_back(FieldState(closer.user.usage, overlap, 
                         proj_info.projection, proj_info.projection_space, 
                         are_all_children_disjoint(), true/*dirty reduce*/));
                // If we are a reduction, we can go straight there
                it->valid_fields -= overlap;
                if (!it->valid_fields)
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
                open_below |= (it->valid_fields & current_mask);
                it++;
              }
              else
              {
                // Otherwise we need a close operation
                if (record_close_operations)
                {
                  const FieldMask overlap = current_mask & it->valid_fields;
#ifdef DEBUG_LEGION
                  assert(!!overlap);
                  //assert(!disjoint_close);
#endif
                  closer.record_close_operation(overlap, true/*projection*/);
                  state.capture_close_epochs(overlap,
                                             closer.find_closed_node(this));
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
                }
                it->valid_fields -= current_mask;
                if (!it->valid_fields)
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
                  const FieldMask overlap = current_mask & it->valid_fields;
#ifdef DEBUG_LEGION
                  assert(!!overlap);
#endif
                  closer.record_close_operation(overlap, true/*projection*/,
                                                disjoint_close);
                  state.capture_close_epochs(overlap,
                                             closer.find_closed_node(this));
                  // Advance the projection epochs
                  state.advance_projection_epochs(overlap);
                  // If we're doing a disjoint close update the open
                  // states accordingly 
                  if (disjoint_close)
                  {
                    // Record the fields are already open
                    open_below |= overlap;
                    // Make a new state with the default projection
                    // function to indicate that we are open in 
                    // disjoint shallow mode with read-write
                    RegionUsage close_usage(READ_WRITE, EXCLUSIVE, 0);
                    IndexSpaceNode *color_space = 
                      as_partition_node()->row_source->color_space;
                    new_states.push_back(FieldState(close_usage, overlap,
                        context->runtime->find_projection_function(0),
                        color_space, true/*disjoint*/));
                  }
                }
                it->valid_fields -= current_mask;
                if (!it->valid_fields)
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              else
              {
                // If this is a dirty reduction, then record it
                if (it->open_state == OPEN_REDUCE_PROJ_DIRTY)
                  proj_info.set_dirty_reduction(); 
                open_below |= (it->valid_fields & current_mask);
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
        new_states.push_back(FieldState(closer.user.usage, open_mask, 
              proj_info.projection, proj_info.projection_space, 
              are_all_children_disjoint()));
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
                                                  const LegionColor next_child,
                                   LegionDeque<FieldState>::aligned &new_states)
    //--------------------------------------------------------------------------
    {
      // If we are doing a reduction too, check to see if they are 
      // the same in which case we can skip these fields
      if (closer.user.usage.redop > 0)
      {
        LegionMap<ReductionOpID,FieldMask>::aligned::const_iterator finder =
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
          FieldMask overlap = it->valid_fields & reduction_flush_fields;
          if (!overlap)
          {
            it++;
            continue;
          }
          if (it->is_projection_state())
          {
            closer.record_close_operation(overlap, true/*projection*/);
            ClosedNode *closed_node = closer.find_closed_node(this); 
            if (!!state.dirty_fields)
              closed_node->record_closed_fields(overlap & state.dirty_fields);
            state.capture_close_epochs(overlap, closed_node);
            state.advance_projection_epochs(overlap);
            it->valid_fields -= overlap;
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
          if (!it->valid_fields)
            it = state.field_states.erase(it);
          else
            it++;
        }
        // Check to see if we have any unflushed fields
        // These are fields which still need a close operation
        // to be performed but only to flush the reductions
        FieldMask unflushed = reduction_flush_fields - flushed_fields;
        if (!!unflushed)
          closer.record_flush_only_close(unflushed);
        // Then we can mark that these fields no longer have 
        // unflushed reductions
        clear_logical_reduction_fields(state, reduction_flush_fields);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_close_operations(LogicalCloser &closer,
                                            const FieldMask &closing_mask,
                                            FieldState &state,
                                            const LegionColor next_child, 
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
      if ((next_child != INVALID_COLOR) && all_children_disjoint)
      {
        // If we have a next child and all the children are disjoint
        // then there are never any close operations, all we have to
        // do is determine if we need to upgrade the child or not
#ifdef DEBUG_LEGION
        assert(allow_next_child);
        assert(aliased_children == NULL);
#endif
        // Check to see if we have any open fields already 
        LegionMap<LegionColor,FieldMask>::aligned::iterator finder = 
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
              finder->second -= overlap;
              removed_fields = true;
              if (!finder->second)
                state.open_children.erase(finder);
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
        std::vector<LegionColor> to_delete;
        for (LegionMap<LegionColor,FieldMask>::aligned::iterator it = 
              state.open_children.begin(); it != 
              state.open_children.end(); it++)
        {
          FieldMask close_mask = it->second & closing_mask;
          if (!close_mask)
            continue;
          // Check for same child, only allow
          if (allow_next_child && (next_child != INVALID_COLOR) && 
              (next_child == it->first))
          {
            if (aliased_children == NULL)
            {
              // No aliased children, so everything is open
              output_mask |= close_mask;
              if (upgrade_next_child)
              {
                it->second -= close_mask;
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
                  it->second -= already_open_mask;
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
          if (!overwriting_close && (next_child != INVALID_COLOR) && 
              (it->first != next_child) && (all_children_disjoint || 
               are_children_disjoint(it->first, next_child)))
            continue;
          // Perform the close operation
          RegionTreeNode *child_node = get_tree_child(it->first);
          child_node->close_logical_node(closer, close_mask, 
                                         true/*read only*/);
          if (record_close_operations)
          {
            if (overwriting_close)
              closer.record_overwriting_close(close_mask, false/*projection*/);
            else
              closer.record_read_only_close(close_mask, false/*projection*/);
          }
          // Remove the close fields
          it->second -= close_mask;
          removed_fields = true;
          if (!it->second)
            to_delete.push_back(it->first);
          if (record_closed_fields)
            output_mask |= close_mask;
        }
        // Remove the children that can be deleted
        for (std::vector<LegionColor>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
          state.open_children.erase(*it);
      }
      else
      {
        // See if we need a full close operation, if we do then we
        // must close up all the children for the field(s) being closed
        // If there is no next_child, then we have to close all
        // children, otherwise figure out which fields need closes
        FieldMask full_close;
        if (next_child != INVALID_COLOR)
        {
          // Figure what if any children need to be closed
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
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
              if (all_children_disjoint || 
                  are_children_disjoint(it->first, next_child))
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
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
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
            closer.record_close_operation(full_close, false/*projection*/);
          if (record_closed_fields)
            output_mask |= full_close;
          std::vector<LegionColor> to_delete;
          // Go through and delete all the children which are to be closed
          for (LegionMap<LegionColor,FieldMask>::aligned::iterator it = 
                state.open_children.begin(); it !=
                state.open_children.end(); it++)
          {
            FieldMask child_close = it->second & full_close;
            if (!child_close)
              continue;
            // Perform the close operation
            RegionTreeNode *child_node = get_tree_child(it->first);
            child_node->close_logical_node(closer, child_close, 
                                           false/*read only close*/);
            if (state.open_state == OPEN_SINGLE_REDUCE ||
                state.open_state == OPEN_MULTI_REDUCE)
            {
              ClosedNode *closed_tree = closer.find_closed_node(child_node);
              closed_tree->record_reduced_fields(child_close);
            }
            // Remove the close fields
            it->second -= child_close;
            removed_fields = true;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          // Remove the children that can be deleted
          for (std::vector<LegionColor>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            state.open_children.erase(*it);
        }
        // If we allow the next child, we need to see if the next
        // child is already open or not
        if (allow_next_child && (next_child != INVALID_COLOR))
        {
          LegionMap<LegionColor,FieldMask>::aligned::iterator finder = 
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
                finder->second -= overlap;
                removed_fields = true;
                if (!finder->second)
                  state.open_children.erase(finder);
              }
            }
          }
        }
      }
      // If we have no children, we can clear our fields
      if (state.open_children.empty())
        state.valid_fields.clear();
      // See if it is time to rebuild the valid mask 
      else if (removed_fields)
      {
        if (state.rebuild_timeout == 0)
        {
          // Rebuild the valid fields mask
          FieldMask new_valid_mask;
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it = 
                state.open_children.begin(); it != 
                state.open_children.end(); it++)
          {
            new_valid_mask |= it->second;
          }
          state.valid_fields = new_valid_mask;    
          // Reset the timeout to the order of the number of open children
          state.rebuild_timeout = state.open_children.size();
        }
        else
          state.rebuild_timeout--;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_state(LogicalState &state,
                                               const FieldState &new_state)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!new_state.valid_fields);
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
      state.field_states.push_back(new_state);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_states(LogicalState &state,
                             const LegionDeque<FieldState>::aligned &new_states)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < new_states.size(); idx++)
      {
        const FieldState &next = new_states[idx];
        merge_new_field_state(state, next);
      }
#ifdef DEBUG_LEGION
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_prev_epoch_users(LogicalState &state,
                                                 const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
      for (LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned::iterator 
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
                                                 const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
      for (LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned::iterator 
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
          if (it->op->add_mapping_reference(it->gen))
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
                                                    const FieldMask &uninit)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_region());
#endif
      LogicalRegion handle = as_region_node()->handle;
      char *field_string = column_source->to_string(uninit);
      REPORT_LEGION_WARNING(LEGION_WARNING_REGION_REQUIREMENT_OPERATION_USING,
        "WARNING: Region requirement %d of operation %s "
                      "(UID %lld) is using uninitialized data for field(s) %s "
                      "of logical region (%d,%d,%d)", idx, 
                      op->get_logging_name(), op->get_unique_op_id(),
                      field_string, handle.get_index_space().get_id(),
                      handle.get_field_space().get_id(), 
                      handle.get_tree_id());
      free(field_string);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_logical_reduction(LogicalState &state,
                                                  ReductionOpID redop,
                                                  const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      state.reduction_fields |= user_mask;
      LegionMap<ReductionOpID,FieldMask>::aligned::iterator finder = 
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
      for (LegionMap<ReductionOpID,FieldMask>::aligned::iterator it = 
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
      LegionMap<LegionColor,FieldMask>::aligned previous_children;
      for (std::list<FieldState>::const_iterator fit = 
            state.field_states.begin(); fit != state.field_states.end(); fit++)
      {
        FieldMask actually_valid;
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              fit->open_children.begin(); it != 
              fit->open_children.end(); it++)
        {
          actually_valid |= it->second;
          if (previous_children.find(it->first) == previous_children.end())
          {
            previous_children[it->first] = it->second;
          }
          else
          {
            FieldMask &previous = previous_children[it->first];
            assert(!(previous & it->second));
            previous |= it->second;
          }
        }
        // Actually valid should be greater than or equal
        assert(!(actually_valid - fit->valid_fields));
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
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator 
                cit1 = f1.open_children.begin(); cit1 != 
                f1.open_children.end(); cit1++)
          {
            for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator 
                  cit2 = f2.open_children.begin(); 
                  cit2 != f2.open_children.end(); cit2++)
            {
              
              // Disjointness check on fields
              if (cit1->second * cit2->second)
                continue;
#ifndef NDEBUG
              LegionColor c1 = cit1->first;
              LegionColor c2 = cit2->first;
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
      for (LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned::iterator 
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
      for (LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned::iterator 
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
                                                   RestrictInfo &restrict_info,
                                                   VersionInfo &version_info,
                                                   const TraceInfo &trace_info)
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
        LegionColor next_child = path.get_child(depth);
        if (!!check_mask)
        {
          // Perform any close operations
          LogicalCloser closer(ctx, user, this,
                               false/*validates*/, true/*capture*/);
          siphon_logical_deletion(closer, state, check_mask, next_child, 
                              open_below, ((depth+1) == path.get_max_depth()));
          if (closer.has_close_operations())
          {
            // Generate the close operations         
            // We need to record the version numbers for this node as well
            closer.initialize_close_operations(state, user.op, 
                                               version_info, trace_info);
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
        // If we have any split fields we have to record them
        FieldMask split_fields = state.dirty_below & user.field_mask;
        if (!!split_fields)
          version_info.record_split_fields(this, split_fields);
        // Continue the traversal
        RegionTreeNode *child = get_tree_child(next_child);
        // Only continue checking the fields that are open below
        child->register_logical_deletion(ctx, user, open_below, path,
                                         restrict_info,version_info,trace_info);
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
        // We used to clear out the state below here but we've stopped doing
        // that in order to support multiple deletions. We just leave the
        // logical state in place until the end of the context so that multiple
        // deletions of resources will work properly. We might change this if
        // we ever come up with a way to avoid duplicate deletions.
        // DeletionInvalidator invalidator(ctx, user.field_mask);
        // visit_node(&invalidator);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_logical_deletion(LogicalCloser &closer,
                                                 LogicalState &state,
                                                 const FieldMask &current_mask,
                                                 const LegionColor next_child,
                                                 FieldMask &open_below,
                                                 bool force_close_next)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(next_child != INVALID_COLOR);
      sanity_check_logical_state(state);
#endif
      LegionDeque<FieldState>::aligned new_states;
      for (LegionList<FieldState>::aligned::iterator it = 
            state.field_states.begin(); it != 
            state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields * current_mask)
        {
          it++;
          continue;
        }
        // See if our child is open, if it's not then we can keep going
        LegionMap<LegionColor,FieldMask>::aligned::const_iterator finder = 
          it->open_children.find(next_child);
        if (it->is_projection_state() ||
            (finder == it->open_children.end()))
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
                if (!it->valid_fields)
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
                if (!it->valid_fields)
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
              if (!it->valid_fields)
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_READ_ONLY_PROJ:
            {
              // Do a read only close here 
              closer.record_read_only_close(overlap, true/*projection*/);
              // Advance the projection epochs
              state.advance_projection_epochs(overlap);
              it->valid_fields -= current_mask;
              if (!it->valid_fields)
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
              closer.record_close_operation(overlap, true/*projection*/);
              state.capture_close_epochs(overlap,
                                         closer.find_closed_node(this));
              // Advance the projection epochs
              state.advance_projection_epochs(overlap);
              it->valid_fields -= current_mask;
              if (!it->valid_fields)
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
        rez.serialize(state.dirty_below);
        rez.serialize(state.dirty_fields);
        rez.serialize(state.reduction_fields);
        rez.serialize<size_t>(state.outstanding_reductions.size());
        for (LegionMap<ReductionOpID,FieldMask>::aligned::const_iterator it = 
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
          rez.serialize<size_t>((*pit)->projections.size());
          for (std::map<ProjectionFunction*,std::set<IndexSpaceNode*> >::
                const_iterator fit = (*pit)->projections.begin(); 
                fit != (*pit)->projections.end(); fit++)
          {
            rez.serialize(fit->first->projection_id);
            rez.serialize<size_t>(fit->second.size());
            for (std::set<IndexSpaceNode*>::const_iterator it = 
                  fit->second.begin(); it != fit->second.end(); it++)
              rez.serialize((*it)->handle);
          }
        }
        rez.serialize<size_t>(state.field_states.size());
        for (LegionList<FieldState>::aligned::const_iterator fit = 
              state.field_states.begin(); fit != 
              state.field_states.end(); fit++)
        {
          rez.serialize(fit->valid_fields);
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
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
                fit->open_children.begin(); it != 
                fit->open_children.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
            to_traverse.insert(get_tree_child(it->first));
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
                                                      Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      derez.deserialize(state.dirty_below);
      derez.deserialize(state.dirty_fields);
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
          std::set<IndexSpaceNode*> &spaces = epoch->projections[function];
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
      for (LegionList<FieldState>::aligned::iterator fit = 
            state.field_states.begin(); fit != state.field_states.end(); fit++)
      {
        derez.deserialize(fit->valid_fields);
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
        for (unsigned idx = 0; idx < num_open_children; idx++)
        {
          LegionColor color;
          derez.deserialize(color);
          derez.deserialize(fit->open_children[color]);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::handle_logical_state_return(
                                          Runtime *runtime, Deserializer &derez)
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
                                         derez);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::compute_version_numbers(ContextID ctx,
                                               const RegionTreePath &path,
                                               const RegionUsage &usage,
                                               const FieldMask &version_mask,
                                               FieldMask &unversioned_mask,
                                               Operation *op, unsigned idx,
                                               InnerContext *parent_ctx,
                                               VersionInfo &version_info,
                                               std::set<RtEvent> &ready_events,
                                               bool partial_traversal,
                                               bool disjoint_close,
                                               UniqueID logical_context_uid,
          const LegionMap<ProjectionEpochID,FieldMask>::aligned *advance_epochs)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      const unsigned depth = get_depth();
      const bool arrived = !path.has_child(depth);
      if (arrived)
      {
        if (partial_traversal)
        {
#ifdef DEBUG_LEGION
          assert(advance_epochs == NULL);
#endif
          // If we are doing a partial traversal, then we just do the
          // normal path only traversal and don't recurse
          const FieldMask &split_mask = version_info.get_split_mask(depth);
          manager.record_path_only_versions(version_mask, split_mask,
                               unversioned_mask, parent_ctx, op, idx, 
                               usage, version_info, ready_events);
        }
        else if (disjoint_close)
        {
          manager.record_disjoint_close_versions(version_mask, 
                    parent_ctx, op, idx, version_info, ready_events);
        }
        else
        {
          // Record the current versions to be used here 
          manager.record_current_versions(version_mask, unversioned_mask, 
                    parent_ctx, op, idx, usage, version_info, ready_events);
          // If we had any unversioned fields all the way down then we
          // have to report that uninitialized data is being used
          // Skip this if we have simultaneous coherence as it might
          // be unclear who is reading or writing first
          if (!!unversioned_mask && !IS_SIMULT(usage))
            report_uninitialized_usage(op, idx, unversioned_mask);
        }
      }
      else
      {
        // See if we need to compute the split mask first, this happens
        // only for projection epochs
        if (advance_epochs != NULL)
          manager.compute_advance_split_mask(version_info, logical_context_uid,
                      parent_ctx, version_mask, ready_events, *advance_epochs);
        // Record the path only versions
        const FieldMask &split_mask = version_info.get_split_mask(depth);
        manager.record_path_only_versions(version_mask, split_mask,
                             unversioned_mask, parent_ctx, op, idx, 
                             usage, version_info, ready_events);
        // Continue the traversal down the tree
        RegionTreeNode *child = get_tree_child(path.get_child(depth));
        child->compute_version_numbers(ctx, path, usage, version_mask,
                          unversioned_mask, op, idx, parent_ctx, 
                          version_info, ready_events, partial_traversal,
                          disjoint_close, logical_context_uid, advance_epochs);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::advance_version_numbers(ContextID ctx,
                                               const RegionTreePath &path,
                                               const FieldMask &advance_mask,
                                               InnerContext *parent_ctx, 
                                               bool update_parent_state,
                                               bool skip_update_parent,
                                               UniqueID logical_context_uid,
                                               bool dedup_opens, 
                                               bool dedup_advances, 
                                               ProjectionEpochID open_epoch,
                                               ProjectionEpochID advance_epoch,
                  const LegionMap<unsigned,FieldMask>::aligned &dirty_previous,
                                               std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      VersionManager &manager = get_current_version_manager(ctx);
      // Perform the advance
      const bool local_update_parent = update_parent_state && 
                                       !skip_update_parent;
      const unsigned depth = get_depth();
      const bool arrived = !path.has_child(depth);
      // As long as we are advancing in the middle of the tree
      // then we have no initial state
      bool still_need_advance = true;
      if (!dirty_previous.empty())
      {
        LegionMap<unsigned,FieldMask>::aligned::const_iterator finder = 
          dirty_previous.find(depth);
        if (finder != dirty_previous.end())
        {
          const FieldMask dirty_prev = finder->second & advance_mask;
          if (!!dirty_prev)
          {
            manager.advance_versions(advance_mask, logical_context_uid,
                          parent_ctx, local_update_parent,
                          ready_events, dedup_opens, open_epoch, 
                          dedup_advances, advance_epoch, &dirty_prev);
            still_need_advance = false;
          }
        }
      }
      if (still_need_advance)
        manager.advance_versions(advance_mask, logical_context_uid,
              parent_ctx, local_update_parent, ready_events, 
              dedup_opens, open_epoch, dedup_advances, advance_epoch);
      // Continue the traversal, 
      // wonder if we get tail call recursion optimization
      if (!arrived)
      {
        RegionTreeNode *child = get_tree_child(path.get_child(depth));
        child->advance_version_numbers(ctx, path, advance_mask, 
                                       parent_ctx, update_parent_state, 
                                       false/*update*/, logical_context_uid,
                                       dedup_opens, dedup_advances, 
                                       open_epoch, advance_epoch, 
                                       dirty_previous, ready_events);
      }
    }

    //--------------------------------------------------------------------------
    CompositeView* RegionTreeNode::create_composite_instance(ContextID ctx_id,
                                                const FieldMask &closing_mask,
                                                VersionInfo &version_info, 
                                                UniqueID logical_context_uid,
                                                InnerContext *owner_ctx,
                                                ClosedNode *closed_tree,
                                                std::set<RtEvent> &ready_events,
                                                const ProjectionInfo *proj_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(version_info.is_upper_bound_set());
#endif
      // First we need to record the rest of our closed versions
      VersionManager &manager = get_current_version_manager(ctx_id);
      // A close always advances the version numbers even if we 
      // already did an advance for dirty below. Closes of trees
      // below then effectively are double bumps of the version
      // numbers while reduction flushes are just a single 
      // increment. The double bump for closing sub-trees
      // is necessary to maintain the monotonicity of the 
      // version state objects.
      const bool update_parent_state = !version_info.is_upper_bound_node(this);
      if (proj_info != NULL)
      {
        // This is the uncommon case that occurs when we are doing a 
        // disjoint partition close and we're actually closing each 
        // of the children individually. We we do this we need to advance
        // with an open of the new projection epoch
        const LegionMap<ProjectionEpochID,FieldMask>::aligned &epochs = 
          proj_info->get_projection_epochs();
        for (LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator 
              it = epochs.begin(); it != epochs.end(); it++)
        {
          FieldMask overlap = it->second & closing_mask;
          if (!overlap)
            continue;
          manager.advance_versions(overlap, logical_context_uid, owner_ctx, 
                                   true/*update parent*/,
                                   ready_events, true/*dedup opens*/,
                                   it->first, false/*dedup advances*/, 
                                   0/*epoch*/, NULL/*dirty previous*/, 
                                   proj_info);
        }
      }
      else // The common case
        manager.advance_versions(closing_mask, logical_context_uid, owner_ctx,
                                 update_parent_state, ready_events);
      // Now we can record our advance versions
      manager.record_advance_versions(closing_mask, owner_ctx, 
                                      version_info, ready_events);
      // Fix the closed tree
      closed_tree->fix_closed_tree();
      // Prepare to make the new view
      DistributedID did = context->runtime->get_available_distributed_id();
      // Copy the version info that we need
      DeferredVersionInfo *view_info = new DeferredVersionInfo();
      version_info.copy_to(*view_info);
      const AddressSpace local_space = context->runtime->address_space;
      // Make the view
      CompositeView *result = new CompositeView(context, did, 
                           local_space, this, view_info, 
                           closed_tree, owner_ctx, true/*register now*/);
      // Capture the state of the top of the composite view
      PhysicalState *state = get_physical_state(version_info);
      LegionMap<LogicalView*,FieldMask>::aligned valid_above;
      // See if we have any valid views above we need to capture
      if (!version_info.is_upper_bound_node(this))
      {
        RegionTreeNode *parent = get_parent();
        if (parent != NULL)
        {
          FieldMask up_mask = closing_mask - state->dirty_mask;
          if (!!up_mask)
          {
            PhysicalState *parent_state = 
              parent->get_physical_state(version_info);
            LegionMap<LogicalView*,FieldMask>::aligned local_valid;
            parent->find_valid_instance_views(ctx_id, parent_state, up_mask, 
                  up_mask, version_info, false/*needs space*/, local_valid);
            // Get the subview for this level
            for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator 
                  it = local_valid.begin(); it != local_valid.end(); it++)
            {
              LogicalView *local_view = it->first->get_subview(get_color());
              valid_above[local_view] = it->second;
            }
          }
        }
      }
      WrapperReferenceMutator mutator(ready_events);
      state->capture_composite_root(result, closing_mask, &mutator,valid_above);
      result->finalize_capture(true/*prune*/);
      // Clear out any reductions
      invalidate_reduction_views(state, closing_mask);
      // Update the valid views to reflect the new valid instance
      update_valid_views(state, closing_mask, true/*dirty*/, result);  
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::find_valid_instance_views(ContextID ctx,
                                                   PhysicalState *state,
                                                   const FieldMask &valid_mask,
                                                   const FieldMask &space_mask,
                                                   VersionInfo &version_info,
                                                   bool needs_space,
                        LegionMap<LogicalView*,FieldMask>::aligned &valid_views)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_FIND_VALID_INSTANCE_VIEWS_CALL);
#ifdef DEBUG_LEGION
      assert(state->node == this);
#endif
      if (!version_info.is_upper_bound_node(this))
      {
        RegionTreeNode *parent = get_parent();
        if (parent != NULL)
        {
          FieldMask up_mask = valid_mask - state->dirty_mask;
          if (!!up_mask || needs_space)
          {
            PhysicalState *parent_state = 
              parent->get_physical_state(version_info);
            LegionMap<LogicalView*,FieldMask>::aligned local_valid;
            parent->find_valid_instance_views(ctx, parent_state, up_mask, 
                      space_mask, version_info, needs_space, local_valid);
            // Get the subview for this level
            for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator 
                  it = local_valid.begin(); it != local_valid.end(); it++)
            {
              LogicalView *local_view = it->first->get_subview(get_color());
              valid_views[local_view] = it->second;
            }
          }
        }
      }
      // Now figure out which of our valid views we can add
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            state->valid_views.begin(); it != state->valid_views.end(); it++)
      {
        // If we need the physical instances to be at least as big as the
        // needed fields, check that first
        if (needs_space)
        {
          if (it->first->is_deferred_view())
            continue;
#ifdef DEBUG_LEGION
          assert(it->first->as_instance_view()->is_materialized_view());
#endif
          MaterializedView *current = 
            it->first->as_instance_view()->as_materialized_view();
          if (!!(space_mask - current->get_physical_mask()))
            continue;
          // Also check for empty instances
          if (current->manager->instance_domain->get_volume() == 0)
            continue;
        }
        // If we're looking for instances with space, we want the instances
        // even if they have no valid fields, otherwise if we're not looking
        // for instances with enough space, we can exit out early if they
        // don't have any valid fields
        FieldMask overlap = valid_mask & it->second;
        if (!needs_space && !overlap)
          continue;
        // Check to see if we need to merge the field views.
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          valid_views.find(it->first);
        if (finder == valid_views.end())
          valid_views[it->first] = overlap;
        else
          finder->second |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::find_valid_reduction_views(ContextID ctx,
                                                    PhysicalState *state,
                                                    ReductionOpID redop,
                                                    const FieldMask &valid_mask,
                                                    VersionInfo &version_info,
                                          std::set<ReductionView*> &valid_views)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_FIND_VALID_REDUCTION_VIEWS_CALL);
#ifdef DEBUG_LEGION
      assert(state->node == this);
#endif
      if (!version_info.is_upper_bound_node(this))
      {
        RegionTreeNode *parent = get_parent();
        if (parent != NULL)
        {
          // See if we can continue going up the tree
          FieldMask up_mask = valid_mask - state->dirty_mask;
          if (!!up_mask)
          {
            // Acquire the parent state in non-exclusive mode
            PhysicalState *parent_state = 
              parent->get_physical_state(version_info);
            parent->find_valid_reduction_views(ctx, parent_state, redop, 
                                    up_mask, version_info, valid_views);
          }
        }
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            state->reduction_views.begin(); it != 
            state->reduction_views.end(); it++)
      {
        if (it->first->get_redop() != redop)
          continue;
        FieldMask uncovered = valid_mask - it->second;
        if (!uncovered && (valid_views.find(it->first) == valid_views.end()))
        {
          valid_views.insert(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::pull_valid_instance_views(ContextID ctx,
                                                   PhysicalState *state,
                                                   const FieldMask &mask,
                                                   bool needs_space,
                                                   VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
#endif
      LegionMap<LogicalView*,FieldMask>::aligned new_valid_views;
      find_valid_instance_views(ctx, state, mask, mask, version_info,
                                needs_space, new_valid_views);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            new_valid_views.begin(); it != new_valid_views.end(); it++)
      {
        update_valid_views(state, it->second, false/*dirty*/, it->first);
      }
    }
    
    //--------------------------------------------------------------------------
    void RegionTreeNode::find_copy_across_instances(const TraversalInfo &info,
                                                    MaterializedView *target,
                 LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
                LegionMap<DeferredView*,FieldMask>::aligned &deferred_instances)
    //--------------------------------------------------------------------------
    {
      LegionMap<LogicalView*,FieldMask>::aligned valid_views;
      PhysicalState *state = get_physical_state(info.version_info); 
      find_valid_instance_views(info.ctx, state, info.traversal_mask,
                                info.traversal_mask, info.version_info,
                                false/*needs space*/, valid_views);
      // Now tease them apart into src and composite views and sort
      // them based on the target memory
      FieldMask copy_mask = info.traversal_mask;
      sort_copy_instances(info, target, copy_mask, valid_views,
                          src_instances, deferred_instances);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::issue_update_copies(const TraversalInfo &info,
                                             MaterializedView *dst,
                                             FieldMask copy_mask,
             const LegionMap<LogicalView*,FieldMask>::aligned &valid_instances,
                                             const RestrictInfo &restrict_info,
                                             bool restrict_out /*=false*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_ISSUE_UPDATE_COPIES_CALL);
#ifdef DEBUG_LEGION
      assert(!!copy_mask);
      assert(dst->logical_node == this);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        if (!it->first->is_deferred_view())
          assert(it->first->logical_node == this);
      }
      // We better have space for all the fields in the request
      assert(!(copy_mask - dst->get_space_mask()));
#endif
      // Quick check to see if we are done early
      {
        LegionMap<LogicalView*,FieldMask>::aligned::const_iterator finder = 
          valid_instances.find(dst);
        if (finder != valid_instances.end())
        {
          copy_mask -= finder->second;
          if (!copy_mask)
            return;
        }
      }
      // Another check here to also see if we are already valid
      // There can be races with read-only region requirements from
      // different operations updating the same physical instance,
      // we know that our mapping process is atomic with respect to
      // those other region requirements so check to see if the version
      // numbers on the physical instance match the ones we need, if
      // they do then we know that it has already been updated and
      // we don't have to issue any copies ourself
      // (Except when we have restrictions, and we need the copies)
      if (IS_READ_ONLY(info.req) && !restrict_info.has_restrictions())
      {
        FieldMask already_valid = copy_mask;
        dst->filter_invalid_fields(already_valid, info.version_info);
        if (!!already_valid)
        {
          copy_mask -= already_valid;
          if (!copy_mask)
            return;
        }
      }
      // To facilitate optimized copies in the low-level runtime, 
      // we gather all the information needed to issue gather copies 
      // from multiple instances into the data structures below, we then 
      // issue the copy when we're done and update the destination instance.
      LegionMap<MaterializedView*,FieldMask>::aligned src_instances;
      LegionMap<DeferredView*,FieldMask>::aligned deferred_instances;
      // This call updates copy_mask
      sort_copy_instances(info, dst, copy_mask, valid_instances, 
                          src_instances, deferred_instances);

      // Now we can issue the copy operation to the low-level runtime
      if (!src_instances.empty())
      {
        // Get all the preconditions for each of the different instances
        LegionMap<ApEvent,FieldMask>::aligned preconditions;
        FieldMask update_mask; 
        const AddressSpaceID local_space = context->runtime->address_space;
        for (LegionMap<MaterializedView*,FieldMask>::aligned::const_iterator 
              it = src_instances.begin(); it != src_instances.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(!!it->second);
#endif
          it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                             true/*single copy*/, restrict_out,
                                             it->second, &info.version_info,
                                             info.op->get_unique_op_id(),
                                             info.index, local_space, 
                                             preconditions,
                                             info.map_applied_events);
          update_mask |= it->second;
        }
        // Now do the destination
        dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                     true/*single copy*/, restrict_out,
                                     update_mask, &info.version_info,
                                     info.op->get_unique_op_id(),
                                     info.index, local_space, preconditions,
                                     info.map_applied_events);
        // If we're restricted, get our restrict precondition
        if (restrict_info.has_restrictions())
        {
          FieldMask restrict_mask;
          restrict_info.populate_restrict_fields(restrict_mask);
          restrict_mask &= update_mask;
          if (!!restrict_mask)
          {
            ApEvent restrict_pre = info.op->get_restrict_precondition();
            preconditions[restrict_pre] = restrict_mask;
          }
        }
        LegionMap<ApEvent,FieldMask>::aligned postconditions;
        issue_grouped_copies(info, dst, restrict_out, PredEvent::NO_PRED_EVENT,
                                     preconditions, update_mask, src_instances, 
                                     &info.version_info, postconditions);
        // Tell the destination about all of the copies that were done
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              postconditions.begin(); it != postconditions.end(); it++)
        {
          dst->add_copy_user(0/*redop*/, it->first, &info.version_info, 
                             info.op->get_unique_op_id(), info.index,
                             it->second, false/*reading*/, restrict_out,
                             local_space, info.map_applied_events);
        }
        if (restrict_out && restrict_info.has_restrictions())
        {
          FieldMask restrict_mask;
          restrict_info.populate_restrict_fields(restrict_mask);
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
                postconditions.begin(); it != postconditions.end(); it++)
          {
            if (it->second * restrict_mask)
              continue;
            info.op->record_restrict_postcondition(it->first);
          }
        }
      }
      // If we still have fields that need to be updated and there
      // are composite instances then we need to issue updates copies
      // for those fields from the composite instances
      if (!deferred_instances.empty())
      {
        for (LegionMap<DeferredView*,FieldMask>::aligned::const_iterator it =
              deferred_instances.begin(); it != deferred_instances.end(); it++)
        {
          it->first->issue_deferred_copies(info, dst, it->second, 
                                           restrict_info, restrict_out);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::sort_copy_instances(const TraversalInfo &info,
                                             MaterializedView *dst,
                                             FieldMask &copy_mask,
               const LegionMap<LogicalView*,FieldMask>::aligned &copy_instances,
                 LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
                LegionMap<DeferredView*,FieldMask>::aligned &deferred_instances)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_SORT_COPY_INSTANCES_CALL);
      // No need to call the mapper if there is only one valid instance
      if (copy_instances.size() == 1)
      {
        const std::pair<LogicalView*,FieldMask> &src_info = 
          *(copy_instances.begin());
        FieldMask op_mask = copy_mask & src_info.second;
        if (!!op_mask)
        {
          LogicalView *src = src_info.first;
          // No need to do anything if src and destination are the same
          // Also check for the same instance which can occur in the case
          // of deferred instances
          if (src != dst)
          {
            if (src->is_deferred_view())
            {
              DeferredView *current = src->as_deferred_view();
              deferred_instances[current] = op_mask;
            }
            else
            {
#ifdef DEBUG_LEGION
              assert(src->is_instance_view());
              assert(src->as_instance_view()->is_materialized_view());
#endif
              MaterializedView *current = 
                src->as_instance_view()->as_materialized_view();
              // If they are the same instance (which can happen with
              // composite views) then we don't need to do it
              if (current->manager->get_instance() != 
                  dst->manager->get_instance())
                src_instances[current] = op_mask;
            }
          }
        }
      }
      else if (!copy_instances.empty())
      {
        bool copy_ready = false;
        // Ask the mapper to put everything in order
        // Make the source instance set
        InstanceSet src_refs;
        std::vector<MaterializedView*> src_views;
        src_views.reserve(copy_instances.size());
        LegionMap<DeferredView*,FieldMask>::aligned available_deferred;
        // Keep track of the observed fields to see if we have 
        // multiple valid copies of the data, if we do we'll have
        // to invoke the mapper, otherwise, we know we can just do
        // them in any order.
        FieldMask observed_inst_fields;
        bool duplicate_inst_fields = false;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              copy_instances.begin(); it != copy_instances.end(); it++)
        {
          if (it->first->is_deferred_view())
          {
            DeferredView *current = it->first->as_deferred_view();
            LegionMap<DeferredView*,FieldMask>::aligned::iterator finder = 
              available_deferred.find(current);
            if (finder == available_deferred.end())
              available_deferred[current] = it->second;
            else
              finder->second |= it->second;
          }
          else
          {
#ifdef DEBUG_LEGION
            assert(it->first->is_instance_view());
            assert(it->first->as_instance_view()->is_materialized_view());
#endif
            MaterializedView *src_view = 
              it->first->as_instance_view()->as_materialized_view();
            src_refs.add_instance(
                InstanceRef(src_view->get_manager(), it->second));
            src_views.push_back(src_view);
            // If we have duplicate inst fields, record it,
            // otherwise we can just record our observed fields
            if (!duplicate_inst_fields && !(observed_inst_fields * it->second))
              duplicate_inst_fields = true;
            else
              observed_inst_fields |= it->second;
          }
        }

        if (!src_refs.empty())
        {
          std::vector<unsigned> ranking;
          // See if we have any duplicate fields in which case we
          // will need to ask the mapper to pick an order for us
          if (duplicate_inst_fields)
          {
            // Now we have to invoke the mapper to pick the order
            ranking.reserve(src_refs.size());
            InstanceRef target(dst->get_manager(), copy_mask);
            // Ask the mapper to pick the ranking
            info.op->select_sources(target, src_refs, ranking);
            // Go through all the valid instances and issue copies
            // starting first in the order that the mapper requested
            // and then through the remaining instances until all
            // the copy fields are made valid
            for (unsigned i = 0; i < ranking.size(); i++)
            {
              unsigned idx = ranking[i]; 
#ifdef DEBUG_LEGION
              assert(idx < src_views.size());
#endif
              MaterializedView *current_view = src_views[idx];
              // Mark this one NULL so we won't do it again
              src_views[idx] = NULL;
              LegionMap<LogicalView*,FieldMask>::aligned::const_iterator
                mask_finder = copy_instances.find(current_view);
#ifdef DEBUG_LEGION
              assert(mask_finder != copy_instances.end());
#endif
              FieldMask op_mask = copy_mask & mask_finder->second;
              // Check to see if there are any valid fields in the copy mask
              if (!!op_mask)
              {
                // No need to do anything if they are the same instance
                if ((dst != current_view) || (dst->manager->get_instance() != 
                                      current_view->manager->get_instance()))
                {
                  LegionMap<MaterializedView*,FieldMask>::aligned::iterator 
                    finder = src_instances.find(current_view);
                  if (finder == src_instances.end())
                    src_instances[current_view] = op_mask;
                  else
                    finder->second |= op_mask;
                }
                // Update the copy mask
                copy_mask -= op_mask;
                if (!copy_mask)
                {
                  copy_ready = true;
                  break;
                }
              }
            }
          }
          // If we still have missing fields and we didn't cover all
          // the instances, then do the rest of them
          if (!copy_ready && (ranking.size() < src_refs.size()))
          {
            for (unsigned idx = 0; idx < src_views.size(); idx++)
            {
              MaterializedView *current_view = src_views[idx];
              // See if we already considered this instance
              if (current_view == NULL)
                continue;
              LegionMap<LogicalView*,FieldMask>::aligned::const_iterator
                mask_finder = copy_instances.find(current_view);
#ifdef DEBUG_LEGION
              assert(mask_finder != copy_instances.end());
#endif
              FieldMask op_mask = copy_mask & mask_finder->second;
              // Check to see if there are any valid fields in the copy mask
              if (!!op_mask)
              {
                // No need to do anything if they are the same instance
                if ((dst != current_view) || (dst->manager->get_instance() !=
                                      current_view->manager->get_instance()))
                {
                  LegionMap<MaterializedView*,FieldMask>::aligned::iterator 
                    finder = src_instances.find(current_view);
                  if (finder == src_instances.end())
                    src_instances[current_view] = op_mask;
                  else
                    finder->second |= op_mask;
                }
                // Update the copy mask
                copy_mask -= op_mask;
                if (!copy_mask)
                {
                  copy_ready = true;
                  break;
                }
              }
            }
          }
        }
        
        // Lastly, if we are still not done, see if we have
        // any deferred instances to issue copies from
        if (!copy_ready)
        {
          for (LegionMap<DeferredView*,FieldMask>::aligned::const_iterator cit =
                available_deferred.begin(); cit != 
                available_deferred.end(); cit++)
          {
            FieldMask op_mask = copy_mask & cit->second;
            if (!!op_mask)
            {
              // No need to look for duplicates, we know this is
              // the first time this data structure can be touched
              deferred_instances[cit->first] = op_mask;
              copy_mask -= op_mask;
              if (!copy_mask)
                break;
            }
          }
        }
      }
      // Otherwise all the fields have no current data so they are
      // by defintiion up to date
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::issue_grouped_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              bool restrict_out,
                                              PredEvent predicate_guard,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                       const FieldMask &update_mask,
           const LegionMap<MaterializedView*,FieldMask>::aligned &src_instances,
                                             VersionTracker *src_versions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                             CopyAcrossHelper *helper/*= NULL*/,
                                             RegionTreeNode *intersect/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime,REGION_NODE_ISSUE_GROUPED_COPIES_CALL);
      // Now let's build maximal sets of fields which have
      // identical event preconditions. Use a list so our
      // iterators remain valid under insertion and push back
      LegionList<EventSet>::aligned precondition_sets;
      compute_event_sets(update_mask, preconditions, precondition_sets);
      // Now that we have our precondition sets, it's time
      // to issue the distinct copies to the low-level runtime
      // Issue a copy for each of the different precondition sets
      const AddressSpaceID local_space = context->runtime->address_space;
      for (LegionList<EventSet>::aligned::iterator pit = 
            precondition_sets.begin(); pit != 
            precondition_sets.end(); pit++)
      {
        EventSet &pre_set = *pit;
        // Build the src and dst fields vectors
        std::vector<CopySrcDstField> src_fields;
        std::vector<CopySrcDstField> dst_fields;
        LegionMap<MaterializedView*,FieldMask>::aligned update_views;
        for (LegionMap<MaterializedView*,FieldMask>::aligned::const_iterator 
              it = src_instances.begin(); it != src_instances.end(); it++)
        {
          FieldMask op_mask = pre_set.set_mask & it->second;
          if (!!op_mask)
          {
            it->first->copy_from(op_mask, src_fields);
            dst->copy_to(op_mask, dst_fields, helper);
            update_views[it->first] = op_mask;
          }
        }
#ifdef DEBUG_LEGION
        assert(!src_fields.empty());
        assert(!dst_fields.empty());
        assert(src_fields.size() == dst_fields.size());
#endif
        // Now that we've got our offsets ready, we
        // can now issue the copy to the low-level runtime
        ApEvent copy_pre = Runtime::merge_events(pre_set.preconditions);
        ApEvent copy_post = issue_copy(info.op, src_fields, dst_fields, 
                                       copy_pre, predicate_guard, intersect);
        // Save the copy post in the post conditions
        if (copy_post.exists())
        {
          // Register copy post with the source views
          // Note it is up to the caller to make sure the event
          // gets registered with the destination
          for (LegionMap<MaterializedView*,FieldMask>::aligned::const_iterator 
                it = update_views.begin(); it != update_views.end(); it++)
          {
            it->first->add_copy_user(0/*redop*/, copy_post, src_versions,
                                     info.op->get_unique_op_id(), info.index,
                                     it->second, true/*reading*/, restrict_out,
                                     local_space, info.map_applied_events);
          }
          postconditions[copy_post] = pre_set.set_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::compute_event_sets(FieldMask update_mask, 
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                    LegionList<EventSet>::aligned &precondition_sets)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator pit = 
            preconditions.begin(); pit != preconditions.end(); pit++)
      {
        bool inserted = false;
        // Also keep track of which fields have updates
        // but don't have any preconditions
        update_mask -= pit->second;
        FieldMask remaining = pit->second;
        // Insert this event into the precondition sets 
        for (LegionList<EventSet>::aligned::iterator it = 
              precondition_sets.begin(); it != precondition_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->set_mask)
          {
            it->preconditions.insert(pit->first);
            inserted = true;
            break;
          }
          FieldMask overlap = remaining & it->set_mask;
          // Easy case, they are disjoint so keep going
          if (!overlap)
            continue;
          // Moderate case, we are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == remaining)
          {
            // Leave the existing set and make it the difference 
            it->set_mask -= overlap;
            precondition_sets.push_back(EventSet(overlap));
            EventSet &last = precondition_sets.back();
            last.preconditions = it->preconditions;
            last.preconditions.insert(pit->first);
            inserted = true;
            break;
          }
          // Moderate case, we dominate the existing set
          if (overlap == it->set_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->preconditions.insert(pit->first);
            remaining -= overlap;
            // Can't consider ourselves added yet
            continue;
          }
          // Hard case, neither dominates, compute three
          // distinct sets of fields, keep left one in
          // place and reduce scope, add new one at the
          // end for overlap, continue iterating for right one
          it->set_mask -= overlap;
          const std::set<ApEvent> &temp_preconditions = it->preconditions;
          it = precondition_sets.insert(it, EventSet(overlap));
          it->preconditions = temp_preconditions;
          it->preconditions.insert(pit->first);
          remaining -= overlap;
          continue;
        }
        if (!inserted)
        {
          precondition_sets.push_back(EventSet(remaining));
          EventSet &last = precondition_sets.back();
          last.preconditions.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any preconditions, but them in their own set.
      // Put it on the front because it is the copy with
      // no preconditions so it can start right away!
      if (!!update_mask)
        precondition_sets.push_front(EventSet(update_mask));
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::issue_update_reductions(LogicalView *target,
                                                 const FieldMask &mask,
                                                 VersionInfo &version_info,
           const LegionMap<ReductionView*,FieldMask>::aligned &valid_reductions,
                                                 Operation *op, unsigned index,
                                                 std::set<RtEvent> &map_applied,
                                                 bool restrict_out/*= false*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REGION_NODE_ISSUE_UPDATE_REDUCTIONS_CALL);
      // We should never get a deferred view in here, this function
      // should only be called when the targets are guaranteed to all
      // be actual physical instances
#ifdef DEBUG_LEGION
      assert(target->is_instance_view());
#endif
      InstanceView *inst_target = target->as_instance_view();
      // Go through all of our reduction instances and issue reductions
      // to the target instances
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        // Doesn't need to reduce to itself
        if (inst_target == (it->first))
          continue;
        FieldMask copy_mask = mask & it->second;
        if (!!copy_mask)
        {
#ifdef DEBUG_LEGION
          // all fields in the reduction instances should be used
          assert(!(it->second - copy_mask));
#endif
          // Then we have a reduction to perform
          it->first->perform_reduction(inst_target, copy_mask, &version_info, 
                                       op, index, map_applied, 
                                       PredEvent::NO_PRED_EVENT, restrict_out);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_instance_views(PhysicalState *state,
                                                  const FieldMask &invalid_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
#endif
      std::vector<LogicalView*> to_delete;
      for (LegionMap<LogicalView*,FieldMask>::aligned::iterator it = 
            state->valid_views.begin(); it != state->valid_views.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<LogicalView*>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        state->valid_views.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_valid_views(PhysicalState *state, 
                                            LogicalView *to_filter)
    //--------------------------------------------------------------------------
    {
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        state->valid_views.find(to_filter);
      if (finder != state->valid_views.end())
        state->valid_views.erase(finder);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_reduction_views(PhysicalState *state,
                                                  const FieldMask &invalid_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
#endif
      std::vector<ReductionView*> to_delete;
      for (LegionMap<ReductionView*,FieldMask>::aligned::iterator it = 
            state->reduction_views.begin(); it != 
            state->reduction_views.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<ReductionView*>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        state->reduction_views.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::issue_restricted_copies(const TraversalInfo &info,
      const RestrictInfo &restrict_info,const InstanceSet &restricted_instances,
      const std::vector<MaterializedView*> &restricted_views,
      const LegionMap<LogicalView*,FieldMask>::aligned &copy_out_views)
    //--------------------------------------------------------------------------
    {
      FieldMask src_fields;
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            copy_out_views.begin(); it != copy_out_views.end(); it++)
        src_fields |= it->second;
      // For each one of our restricted views, issue an update copy
      for (unsigned idx = 0; idx < restricted_views.size(); idx++)
      {
        FieldMask overlap = src_fields & 
                              restricted_instances[idx].get_valid_fields();
        if (!overlap)
          continue;
        src_fields -= overlap;
        // If it's already the same there is nothing to do
        LegionMap<LogicalView*,FieldMask>::aligned::const_iterator finder =
          copy_out_views.find(restricted_views[idx]);
        if (finder != copy_out_views.end())
        {
          overlap -= finder->second;
          if (!overlap)
          {
            if (!src_fields)
              break;
            continue;
          }
        }
        // Issue the update copy
        issue_update_copies(info, restricted_views[idx], 
                            overlap, copy_out_views, 
                            restrict_info, true/*restrict out*/);
        if (!src_fields)
          break;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::issue_restricted_reductions(const TraversalInfo &info,
      const RestrictInfo &restrict_info,const InstanceSet &restricted_instances,
      const std::vector<InstanceView*> &restricted_views,
      const LegionMap<ReductionView*,FieldMask>::aligned &reduce_out_views)
    //--------------------------------------------------------------------------
    {
      FieldMask reduction_fields;
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduce_out_views.begin(); it != reduce_out_views.end(); it++)
        reduction_fields |= it->second;
      // Issue updates for any of reductions to the targets
      for (unsigned idx = 0; idx < restricted_views.size(); idx++)
      {
        FieldMask overlap = reduction_fields &
                              restricted_instances[idx].get_valid_fields();
        if (!overlap)
          continue;
        issue_update_reductions(restricted_views[idx], overlap, 
                                info.version_info, reduce_out_views,
                                info.op, info.index, 
                                info.map_applied_events, true/*restrict out*/);
        reduction_fields -= overlap;
        if (!reduction_fields)
          break;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_valid_views(PhysicalState *state, 
                                            const FieldMask &valid_mask,
                                            bool dirty, LogicalView *new_view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
      if (!new_view->is_deferred_view())
      {
        assert(new_view->is_instance_view());
        assert(new_view->as_instance_view()->is_materialized_view());
        assert(!(valid_mask - new_view->as_instance_view()->
                as_materialized_view()->manager->layout->allocated_fields));
        assert(new_view->logical_node == this);
      }
#endif
      if (dirty)
      {
        invalidate_instance_views(state, valid_mask); 
        state->dirty_mask |= valid_mask;
      }
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        state->valid_views.find(new_view);
      if (finder == state->valid_views.end())
      {
        // New valid view, update everything accordingly
        state->valid_views[new_view] = valid_mask;
      }
      else
      {
        // It already existed update the valid mask
        finder->second |= valid_mask;
      }
    } 

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_valid_views(PhysicalState *state, 
                                            const FieldMask &dirty_mask,
                                const std::vector<LogicalView*> &new_views,
                                const InstanceSet &corresponding_references)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
      assert(new_views.size() == corresponding_references.size());
#endif
      if (!!dirty_mask)
      {
        invalidate_instance_views(state, dirty_mask); 
        state->dirty_mask |= dirty_mask;
      }
      for (unsigned idx = 0; idx < new_views.size(); idx++)
      {
        LogicalView *new_view = new_views[idx];
        const FieldMask &valid_mask = 
          corresponding_references[idx].get_valid_fields();
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          state->valid_views.find(new_view);
        // If it is a new view, we can just add the fields
        if (finder == state->valid_views.end())
          state->valid_views[new_view] = valid_mask;
        else // it already exists, so update the mask
          finder->second |= valid_mask;
#ifdef DEBUG_LEGION
        if (!new_view->is_deferred_view())
        {
          assert(new_view->as_instance_view()->is_materialized_view());
          MaterializedView *mat_view = 
            new_view->as_instance_view()->as_materialized_view();
          finder = state->valid_views.find(new_view);
          assert(!(finder->second - 
                          mat_view->manager->layout->allocated_fields));
          assert(new_view->logical_node == this);
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_valid_views(PhysicalState *state,
                                            const FieldMask &dirty_mask,
                               const std::vector<InstanceView*> &new_views,
                               const InstanceSet &corresponding_references)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
      assert(new_views.size() == corresponding_references.size());
#endif
      if (!!dirty_mask)
      {
        invalidate_instance_views(state, dirty_mask); 
        state->dirty_mask |= dirty_mask;
      }
      for (unsigned idx = 0; idx < new_views.size(); idx++)
      {
        InstanceView *new_view = new_views[idx];
        const FieldMask &valid_mask = 
          corresponding_references[idx].get_valid_fields();
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          state->valid_views.find(new_view);
        // If it is a new view, we can just add the fields
        if (finder == state->valid_views.end())
          state->valid_views[new_view] = valid_mask;
        else // it already exists, so update the mask
          finder->second |= valid_mask;
#ifdef DEBUG_LEGION
        assert(new_view->is_materialized_view());
        MaterializedView *mat_view = 
          new_view->as_instance_view()->as_materialized_view();
        finder = state->valid_views.find(new_view);
        assert(!(finder->second - 
                        mat_view->manager->layout->allocated_fields));
        assert(new_view->logical_node == this);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_valid_views(PhysicalState *state, 
                                            const FieldMask &dirty_mask,
                                const std::vector<MaterializedView*> &new_views,
                                const InstanceSet &corresponding_references)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
      assert(new_views.size() == corresponding_references.size());
#endif
      if (!!dirty_mask)
      {
        invalidate_instance_views(state, dirty_mask); 
        state->dirty_mask |= dirty_mask;
      }
      for (unsigned idx = 0; idx < new_views.size(); idx++)
      {
        MaterializedView *new_view = new_views[idx];
        const FieldMask &valid_mask = 
          corresponding_references[idx].get_valid_fields();
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          state->valid_views.find(new_view);
        // If it is a new view, we can just add the fields
        if (finder == state->valid_views.end())
          state->valid_views[new_view] = valid_mask;
        else // it already exists, so update the mask
          finder->second |= valid_mask;
#ifdef DEBUG_LEGION
        finder = state->valid_views.find(new_view);
        assert(!(finder->second - new_view->manager->layout->allocated_fields));
        assert(new_view->logical_node == this);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_reduction_views(PhysicalState *state,
                                                const FieldMask &valid_mask,
                                                ReductionView *new_view) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state->node == this);
#endif
      LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
        state->reduction_views.find(new_view);
      if (finder == state->reduction_views.end())
      {
        state->reduction_views[new_view] = valid_mask;
      }
      else
      {
        finder->second |= valid_mask;
      }
      state->reduction_mask |= valid_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::find_complete_fields(const FieldMask &scope_fields,
                     const LegionMap<LegionColor,FieldMask>::aligned &children,
                     FieldMask &complete_fields)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        FieldMask test_fields = it->second - complete_fields;
        // No fields to test so we are done
        if (!test_fields)
          continue;
        RegionTreeNode *child = get_tree_child(it->first);
        if (child->is_complete())
        {
          complete_fields |= test_fields;
          // If we proved that they are all complete, we are done
          if (scope_fields == complete_fields)
            break;
        }
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionTreeNode::convert_manager(PhysicalManager *manager,
                                                  InnerContext *context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Small sanity check to make sure they are in the same tree
      assert(get_tree_id() == manager->region_node->get_tree_id());
      assert(manager->region_node->context == this->context);
#endif
      // Check to see if we've already registered it
      InstanceView *result = find_instance_view(manager, context);
      if (result != NULL)
        return result;
      // We have to make it if it doesn't exist yet
      // If we're at the root, get the view we need for this context
      if (manager->region_node == this)
        result = context->create_instance_top_view(manager, 
                                    context->runtime->address_space);
      // If we didn't find it immediately, switch over to the explicit
      // versions that don't have so many virtual function calls
      else if (is_region())
        result = as_region_node()->convert_reference_region(manager, context);
      else
        result = 
          as_partition_node()->convert_reference_partition(manager, context); 
      return result;
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionTreeNode::convert_reference(const InstanceRef &ref,
                                                    InnerContext *context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!ref.is_virtual_ref());
#endif
      PhysicalManager *manager = ref.get_manager();
      return convert_manager(manager, context);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::convert_target_views(const InstanceSet &targets,
                InnerContext *context, std::vector<InstanceView*> &target_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(targets.size() == target_views.size());
#endif
      if (targets.size() == 1)
      {
        // If we just have one there is a fast path 
        target_views[0] = convert_reference(targets[0], context);
      }
      else
      {
        // Deduplicate the instances and then go up for any that we still need
        std::map<PhysicalManager*,unsigned/*idx*/> deduplicated;
        std::vector<PhysicalManager*> to_convert;
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          PhysicalManager *manager = targets[idx].get_manager();
          target_views[idx] = find_instance_view(manager, context);
          // Quick check to see if we are done early
          if (target_views[idx] != NULL)
            continue;
          // If we didn't find it we will have to go looking for it 
          // See if we've got this one already or not
          if (deduplicated.find(manager) == deduplicated.end())
          {
            unsigned next = deduplicated.size();
            deduplicated[manager] = next;
            to_convert.push_back(manager);
          }
        }
        if (to_convert.size() == 1)
        {
          // Easy case, there is only one deduplicated instance
          InstanceView *result = is_region() ? 
            as_region_node()->convert_reference_region(to_convert[0], context) :
            as_partition_node()->convert_reference_partition(to_convert[0], 
                                                             context);
          for (unsigned idx = 0; idx < target_views.size(); idx++)
          {
            if (target_views[idx] != NULL) // skip if we already did it
              continue;
            target_views[idx] = result;
          }
        }
        else
        {
          // Go up the tree for all the unhandled ones
          std::vector<bool> up_mask(to_convert.size(), true);
          std::vector<InstanceView*> results(to_convert.size(), NULL);
          if (is_region())
            as_region_node()->convert_references_region(to_convert, 
                                        up_mask, context, results);
          else
            as_partition_node()->convert_references_partition(to_convert,
                                        up_mask, context, results);
          for (unsigned idx = 0; idx < target_views.size(); idx++)
          {
            if (target_views[idx] != NULL) // skip it if we already did it
              continue;
            PhysicalManager *manager = targets[idx].get_manager();
#ifdef DEBUG_LEGION
            assert(deduplicated.find(manager) != deduplicated.end());
#endif
            target_views[idx] = results[deduplicated[manager]];
#ifdef DEBUG_LEGION
            assert(target_views[idx] != NULL);
#endif
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::convert_target_views(const InstanceSet &targets,
            InnerContext *context, std::vector<MaterializedView*> &target_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(targets.size() == target_views.size());
#endif
      std::vector<InstanceView*> instance_views(target_views.size(), NULL);
      // Call the instance view version
      convert_target_views(targets, context, instance_views);
      // Then do our conversion
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        InstanceView *inst_view = instance_views[idx];
#ifdef DEBUG_LEGION
        assert(inst_view->is_materialized_view());
#endif
        target_views[idx] = inst_view->as_materialized_view();
      }
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
    bool RegionTreeNode::register_instance_view(PhysicalManager *manager,
                                       UniqueID context_uid, InstanceView *view)
    //--------------------------------------------------------------------------
    {
      std::pair<PhysicalManager*,UniqueID> key(manager, context_uid);
      AutoLock n_lock(node_lock);
      LegionMap<std::pair<PhysicalManager*,UniqueID>,InstanceView*>::
        tracked::const_iterator finder = instance_views.find(key);
      if (finder != instance_views.end())
      {
        // Duplicate registrations are alright
        if (finder->second == view)
          return true;
        return false;
      }
      instance_views[key] = view;
      return true;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::unregister_instance_view(PhysicalManager *manager,
                                                  UniqueID context_uid)
    //--------------------------------------------------------------------------
    {
      std::pair<PhysicalManager*,UniqueID> key(manager, context_uid);
      AutoLock n_lock(node_lock);
      LegionMap<std::pair<PhysicalManager*,UniqueID>,InstanceView*>::
        tracked::iterator finder = instance_views.find(key);
      if (finder != instance_views.end())
        instance_views.erase(finder);
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionTreeNode::find_instance_view(PhysicalManager *manager,
                                                     InnerContext *context) 
    //--------------------------------------------------------------------------
    {
      std::pair<PhysicalManager*,UniqueID> key(manager, 
                                                  context->get_context_uid());
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      LegionMap<std::pair<PhysicalManager*,UniqueID>,InstanceView*>::
        tracked::const_iterator finder = instance_views.find(key);
      if (finder != instance_views.end())
        return finder->second;
      return NULL;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::register_physical_manager(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (physical_managers.find(manager->did) != physical_managers.end())
        return false;
      physical_managers[manager->did] = manager;
      return true;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::unregister_physical_manager(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      physical_managers.erase(manager->did);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* RegionTreeNode::find_manager(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      LegionMap<DistributedID,PhysicalManager*,
                PHYSICAL_MANAGER_ALLOC>::tracked::const_iterator finder = 
                  physical_managers.find(did);
#ifdef DEBUG_LEGION
      assert(finder != physical_managers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_tracking_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(tracking_contexts.find(context) == tracking_contexts.end());
#endif
      tracking_contexts.insert(context);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::unregister_tracking_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(tracking_contexts.find(context) != tracking_contexts.end());
#endif
      tracking_contexts.erase(context);
    }

    //--------------------------------------------------------------------------
    template<AllocationType ALLOC, bool RECORD, bool HAS_SKIP, bool TRACK_DOM>
    /*static*/ FieldMask RegionTreeNode::perform_dependence_checks(
      const LogicalUser &user, 
      typename LegionList<LogicalUser, ALLOC>::track_aligned &prev_users,
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
      for (typename LegionList<LogicalUser, ALLOC>::track_aligned::iterator 
            it = prev_users.begin(); it != prev_users.end(); /*nothing*/)
      {
        if (HAS_SKIP && (to_skip == it->op) && (skip_gen == it->gen))
        {
          it++;
          continue;
        }
        FieldMask overlap = user_check_mask & it->field_mask;
        if (!!overlap)
        {
          if (TRACK_DOM)
            observed_mask |= overlap;
          DependenceType dtype = check_dependence_type(it->usage, user.usage);
          bool validate = validates_regions;
          switch (dtype)
          {
            case NO_DEPENDENCE:
              {
                // No dependence so remove bits from the dominator mask
                dominator_mask -= it->field_mask;
                break;
              }
            case ANTI_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                // Mark that these kinds of dependences are not allowed
                // to validate region inputs
                validate = false;
                // No break so we register dependences just like
                // a true dependence
              }
            case TRUE_DEPENDENCE:
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
                  continue;
                }
                else
                {
                  // hasn't commited, reset timeout and continue
                  it->timeout = LogicalUser::TIMEOUT;
                  it++;
                  continue;
                }
                break;
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
           LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned &ch_users,
           LegionList<LogicalUser,LOGICAL_REC_ALLOC >::track_aligned &abv_users,
           LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cur_users,
           LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pre_users)
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
        LogicalCloser &closer, bool read_only_close,
        typename LegionList<LogicalUser, ALLOC>::track_aligned &users, 
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
      for (typename LegionList<LogicalUser, ALLOC>::track_aligned::iterator 
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

        if (closer.capture_users)
        {
          // Record that we closed this user
          // Update the field mask and the privilege
          closer.record_closed_user(*it, overlap, read_only_close); 
          // Remove the closed set of fields from this user
          it->field_mask -= overlap;
          // If it's empty, remove it from the list and let
          // the mapping reference go up the tree with it
          // Otherwise add a new mapping reference
          if (!it->field_mask)
            it = users.erase(it);
          else
          {
#ifdef LEGION_SPY
            // Always add the reference for Legion Spy
            it->op->add_mapping_reference(it->gen);
            it++;
#else
            // If not Legion Spy we can prune the user if it's done
            if (!it->op->add_mapping_reference(it->gen))
            {
              closer.pop_closed_user(read_only_close);
              it = users.erase(it);
            }
            else
              it++;
#endif
          }
        }
        else
        {
          // If we're not capturing the users, then we actually
          // have to do the dependence analysis with respect to 
          // the closing user
          DependenceType dtype = check_dependence_type(it->usage, 
                                                     closer.user.usage);
#ifdef LEGION_SPY
          if (dtype != NO_DEPENDENCE)
            LegionSpy::log_mapping_dependence(
                closer.user.op->get_context()->get_unique_id(),
                it->uid, it->idx, closer.user.uid, closer.user.idx, dtype);
#endif
          // Register the dependence 
          if (closer.user.op->register_region_dependence(closer.user.idx, 
                                                         it->op, it->gen, 
                                                         it->idx, dtype,
                                                         closer.validates,
                                                         overlap))
          {
#ifndef LEGION_SPY
            it = users.erase(it);
            continue;
#endif
          }
          else
          {
            // it hasn't committed, reset timeout
            it->timeout = LogicalUser::TIMEOUT;
          }
          // Remove the closed set of fields from this user
          it->field_mask -= overlap;
          // Otherwise, if we can remote it, then remove it's
          // mapping reference from the logical tree.
          if (!it->field_mask)
          {
            it->op->remove_mapping_reference(it->gen);
            it = users.erase(it);
          }
          else
            it++;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion r, PartitionNode *par,
                           IndexSpaceNode *row_src, FieldSpaceNode *col_src,
                           RegionTreeForest *ctx)
      : RegionTreeNode(ctx, col_src), handle(r), 
        parent(par), row_source(row_src)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Region %lld %d %d %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, 
          handle.get_index_space().get_id(),
          handle.get_field_space().get_id(),
          handle.get_tree_id());
#endif
    }

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(const RegionNode &rhs)
      : RegionTreeNode(NULL, NULL), handle(LogicalRegion::NO_REGION), 
        parent(NULL), row_source(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------
    { 
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
    void RegionNode::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_active);
      currently_active = false;
#endif
      if (registered)
      {
        const bool top_level = (parent == NULL);
        // Only need to unregister ourselves with the column if we're the top
        // otherwise we need to unregister ourselves with our parent
        if (top_level)
        {
          column_source->remove_instance(this);
          if (column_source->remove_nested_resource_ref(did))
            delete column_source;
        }
        else
        {
          parent->remove_child(row_source->color);
          if (parent->remove_nested_gc_ref(did))
            delete parent;
        }
        // Unregister oursleves with the row source
        row_source->remove_instance(this);
        if (row_source->remove_nested_resource_ref(did))
          delete row_source;
        // Unregister ourselves with the context
        context->remove_node(handle, top_level);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!registered);
#endif
      LocalReferenceMutator mutator;
      if (parent == NULL)
      {
        column_source->add_nested_valid_ref(did, &mutator);
        column_source->add_nested_resource_ref(did);
        column_source->add_instance(this);
      }
      else
      {
        parent->add_nested_gc_ref(did, &mutator);
        parent->add_child(this);
      }
      row_source->add_nested_valid_ref(did, &mutator);
      row_source->add_nested_resource_ref(did);
      row_source->add_instance(this);
      // Add a reference that will be moved when we're destroyed
      add_base_gc_ref(APPLICATION_REF);
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
    void RegionNode::DestructionFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != source)
        runtime->send_logical_region_destruction(handle, target);
    }

    //--------------------------------------------------------------------------
    void RegionNode::find_remote_instances(NodeSet &target_instances)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        target_instances = remote_instances;
      }
      else
        parent->parent->find_remote_instances(target_instances);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::destroy_node(AddressSpaceID source, bool root)
    //--------------------------------------------------------------------------
    {
      if (destroyed)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_REGION_DESTRUCTION,
            "Duplicate deletion of Logical Region (%d,%d,%d)",
            handle.index_space.id, handle.field_space.id, handle.tree_id)
      // If we're the root of the deletion then we might need to send
      // messages to other nodes
      if (root)
      {
        // If we are not on the row source owner node we need to send a message
        // there, otherwise if we are on the owner, we need to send destruction
        // messages out to every node that might have a remote instance
        const AddressSpaceID owner_space = get_owner_space();
        if (owner_space == context->runtime->address_space)
        {
          // Obtain the remote instances from the top of the tree 
          NodeSet target_instances;
          find_remote_instances(target_instances);
          // Send deletions to all but the source
          DestructionFunctor functor(handle, context->runtime, source);
          target_instances.map(functor);
        }
        else if (source != owner_space)
          context->runtime->send_logical_region_destruction(handle,owner_space);
      }
      // Need to recurse down the tree and do the same thing for our children
      // We need a copy in case we end up deleting anything
      std::map<LegionColor,PartitionNode*> children;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        children = color_map;
      }
      for (std::map<LegionColor,PartitionNode*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->second->destroy_node(source, false/*root*/))
          delete it->second;
      }
      // Invalidate our version managers
      invalidate_version_managers();
      // If we're the root then release our tree instances
      if (parent == NULL)
      {
        context->runtime->release_tree_instances(handle.get_tree_id());
        column_source->remove_nested_valid_ref(did);
      }
      row_source->remove_nested_valid_ref(did);
      // Mark that it is destroyed
      destroyed = true;
      // Remove our reference
      return remove_base_gc_ref(APPLICATION_REF);
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
    ApEvent RegionNode::issue_copy(Operation *op,
                        const std::vector<CopySrcDstField> &src_fields,
                        const std::vector<CopySrcDstField> &dst_fields,
                        ApEvent precondition, PredEvent predicate_guard,
                        RegionTreeNode *intersect/*=NULL*/,
                        ReductionOpID redop /*=0*/,bool reduction_fold/*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      // Have to convert back to Realm structures because C++ is dumb  
      std::vector<Realm::CopySrcDstField> realm_src_fields(src_fields.size());
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        realm_src_fields[idx] = src_fields[idx];
      std::vector<Realm::CopySrcDstField> realm_dst_fields(dst_fields.size());
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        realm_dst_fields[idx] = dst_fields[idx];
      ApEvent result = row_source->issue_copy(op, realm_src_fields, 
          realm_dst_fields, precondition, predicate_guard, 
          (intersect == NULL) ? NULL : intersect->get_row_source(),
          redop, reduction_fold);
      ApEvent copy_pre = precondition;
      if ((op != NULL) && op->has_execution_fence_event())
        copy_pre = Runtime::merge_events(copy_pre,
                                         op->get_execution_fence_event());
      LegionSpy::log_copy_events(op->get_unique_op_id(), handle, copy_pre,
                                 result);
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        LegionSpy::log_copy_field(result, src_fields[idx].field_id,
                                  src_fields[idx].inst_event,
                                  dst_fields[idx].field_id,
                                  dst_fields[idx].inst_event, redop);
      if (intersect != NULL)
      {
        if (intersect->is_region())
        {
          RegionNode *node = intersect->as_region_node();
          LegionSpy::log_copy_intersect(result, 1, node->handle.index_space.id,
              node->handle.field_space.id, node->handle.tree_id);
        }
        else
        {
          PartitionNode *node = intersect->as_partition_node();
          LegionSpy::log_copy_intersect(result, 0,
              node->handle.index_partition.id,
              node->handle.field_space.id, node->handle.tree_id);
        }
      }
      return result;
#else
      return row_source->issue_copy(op, src_fields, dst_fields,
          precondition, predicate_guard, 
          (intersect == NULL) ? NULL : intersect->get_row_source(),
          redop, reduction_fold);
#endif
    }

    //--------------------------------------------------------------------------
    ApEvent RegionNode::issue_fill(Operation *op,
                        const std::vector<CopySrcDstField> &dst_fields,
                        const void *fill_value, size_t fill_size,
                        ApEvent precondition, PredEvent predicate_guard,
#ifdef LEGION_SPY
                        UniqueID fill_uid,
#endif
                        RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
      
#ifdef LEGION_SPY
      // Have to convert back to Realm data structures because C++ is dumb
      std::vector<Realm::CopySrcDstField> realm_dst_fields(dst_fields.size());
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        realm_dst_fields[idx] = dst_fields[idx];
      ApEvent result = row_source->issue_fill(op, realm_dst_fields,
          fill_value, fill_size, precondition, predicate_guard,
          (intersect == NULL) ? NULL : intersect->get_row_source());
      ApEvent fill_pre = precondition;
      if ((op != NULL) && op->has_execution_fence_event())
        fill_pre = Runtime::merge_events(fill_pre,
                                         op->get_execution_fence_event());
      LegionSpy::log_fill_events(op->get_unique_op_id(), handle, fill_pre,
                                 result, fill_uid);
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        LegionSpy::log_fill_field(result, dst_fields[idx].field_id,
                                  dst_fields[idx].inst_event);
      if (intersect != NULL)
      {
        if (intersect->is_region())
        {
          RegionNode *node = intersect->as_region_node();
          LegionSpy::log_fill_intersect(result, 1, node->handle.index_space.id,
              node->handle.field_space.id, node->handle.tree_id);
        }
        else
        {
          PartitionNode *node = intersect->as_partition_node();
          LegionSpy::log_fill_intersect(result, 0, 
              node->handle.index_partition.id,
              node->handle.field_space.id, node->handle.tree_id);
        }
      }
      return result;
#else
      return row_source->issue_fill(op, dst_fields,
          fill_value, fill_size, precondition, predicate_guard,
          (intersect == NULL) ? NULL : intersect->get_row_source());
#endif
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
          if (traverser->visit_only_valid())
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            children = color_map;
          }
          else
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            children = color_map;
          }
          for (std::map<LegionColor,PartitionNode*>::const_iterator it = 
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
    void RegionNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      bool send_deletion = false;
      {
        AutoLock n_lock(node_lock); 
        if (!remote_instances.contains(target))
        {
          continue_up = true;
          remote_instances.add(target);
        }
        if (destroyed)
          send_deletion = true;
      }
      if (continue_up)
      {
        if (parent != NULL)
        {
          // Send the parent node first
          parent->send_node(target);
          AutoLock n_lock(node_lock);
          for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
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
            context->runtime->send_logical_region_semantic_info(target, rez);
          }
        }
        else
        {
          // We've made it to the top, send this node
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<size_t>(semantic_info.size());
            for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
                  semantic_info.begin(); it != semantic_info.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second.size);
              rez.serialize(it->second.buffer, it->second.size);
              rez.serialize(it->second.is_mutable);
            }
          }
          context->runtime->send_logical_region_node(target, rez);
        }
      }
      if (send_deletion)
        context->runtime->send_logical_region_destruction(handle, target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_node_creation(RegionTreeForest *context,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);

      RegionNode *node = context->create_node(handle, NULL/*parent*/);
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
          node->attach_semantic_information(tag, source, 
                                            buffer, buffer_size, is_mutable);
        }
      }
    } 

    //--------------------------------------------------------------------------
    void RegionNode::find_open_complete_partitions(ContextID ctx,
               const FieldMask &mask, std::vector<LogicalPartition> &partitions)
    //--------------------------------------------------------------------------
    {
      LogicalState &state = get_logical_state(ctx);
      std::set<LogicalPartition> unique_partitions;
      for (LegionList<FieldState>::aligned::const_iterator sit = 
            state.field_states.begin(); sit != state.field_states.end(); sit++)
      {
        if ((sit->valid_fields * mask) || (sit->is_projection_state()))
          continue;
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it = 
              sit->open_children.begin(); it != sit->open_children.end(); it++)
        {
          if (it->second * mask)
            continue;
          PartitionNode *child = get_child(it->first);
          if (child->is_complete())
            unique_partitions.insert(child->handle);
        }
      }
      partitions.insert(partitions.end(), 
                        unique_partitions.begin(), unique_partitions.end());
    }

    //--------------------------------------------------------------------------
    void RegionNode::premap_region(ContextID ctx,
                                   const RegionRequirement &req,
                                   const FieldMask &valid_mask,
                                   VersionInfo &version_info, 
                                   InstanceSet &targets)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_PREMAP_REGION_CALL);
      PhysicalState *state = get_physical_state(version_info);
      if (!IS_REDUCE(req))
      {
        pull_valid_instance_views(ctx, state, 
            valid_mask, true/*needs space*/, version_info);
#ifdef DEBUG_LEGION
        // Useful for checking premapping of actual regions
        std::vector<LogicalRegion> to_check(1, req.handle_type != 
            PART_PROJECTION ? req.region : LogicalRegion::NO_REGION);
#endif
        // Record the instances and the fields for which they are valid 
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              state->valid_views.begin(); it != state->valid_views.end();it++)
        {
          // Skip any deferred views, they don't actually count here
          if (it->first->is_deferred_view())
            continue;
#ifdef DEBUG_LEGION
          assert(it->first->as_instance_view()->is_materialized_view());
#endif
          MaterializedView *cur_view = 
            it->first->as_instance_view()->as_materialized_view();
          // Check to see if it has space for any fields, if not we can skip it
          FieldMask containing_fields = 
           cur_view->manager->layout->allocated_fields & valid_mask;
          if (!containing_fields)
            continue;
          // Now see if it has any valid fields already
          FieldMask valid_fields = it->second & valid_mask;
#ifdef DEBUG_LEGION
          if (req.handle_type != PART_PROJECTION)
            assert(cur_view->manager->meets_regions(to_check));
#endif
          // Save the reference
          targets.add_instance(InstanceRef(cur_view->manager, valid_fields));
        }
      }
      else
      {
        // See if there are any reduction instances that match locally
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
              state->reduction_views.begin(); it != 
              state->reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & valid_mask;
          if (!overlap)
            continue;
          targets.add_instance(InstanceRef(it->first->manager, overlap));
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::register_region(const TraversalInfo &info,
                                     UniqueID logical_ctx_uid,
                                     InnerContext *context,
                                     RestrictInfo &restrict_info,
                                   ApEvent term_event, const RegionUsage &usage,
                                     bool defer_add_users, InstanceSet &targets,
                                     const ProjectionInfo *proj_info)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_REGISTER_REGION_CALL);
      // If we're actually mapping this region requirement and its a write
      // then we have to advance the version numbers because we didn't
      // do it before when we were unsure of whether we were going to 
      // virtual map the requirement or not. Note we are guaranteed
      // by the logical analysis that this is safe to do without any
      // kind of synchronization.
      VersionManager &manager = get_current_version_manager(info.ctx);
      if (IS_WRITE(usage))
      {
        // We update the parent state if we are not the top-level node
        const bool update_parent_state = 
          !info.version_info.is_upper_bound_node(this);
        manager.advance_versions(info.traversal_mask, logical_ctx_uid, context,
            update_parent_state, info.map_applied_events,
            false/*dedup opens*/, 0/*epoch*/, false/*dedup advances*/, 
            0/*epoch*/, NULL/*dirty previous*/, proj_info);
      }
      // Sine we're mapping we need to record the advance versions
      // where we'll put the results when we're done
      manager.record_advance_versions(info.traversal_mask, context,
                                    info.version_info, info.map_applied_events);
      PhysicalState *state = get_physical_state(info.version_info);
      const AddressSpaceID local_space = context->runtime->address_space;
#ifdef DEBUG_LEGION
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef &ref = targets[idx];
        const FieldMask &valid_fields = ref.get_valid_fields();
        assert(unsigned(FieldMask::pop_count(valid_fields)) <= 
                              info.req.privilege_fields.size());
      }
#endif
      if (IS_REDUCE(info.req))
      {
        // Reduction only case
        // Get any restricted fields
        FieldMask restricted_fields;
        InstanceSet restricted_instances;
        std::vector<InstanceView*> restricted_views;
        if (restrict_info.has_restrictions())
        {
          restrict_info.populate_restrict_fields(restricted_fields);
          restricted_instances = restrict_info.get_instances();
          restricted_views.resize(restricted_instances.size());
          convert_target_views(restricted_instances, context,
                               restricted_views);
        }
        std::vector<ReductionView*> new_views;
        if (!defer_add_users && (targets.size() > 1))
          new_views.resize(targets.size());
        LegionMap<ReductionView*,FieldMask>::aligned reduce_out_views;
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          InstanceRef &ref = targets[idx];
#ifdef DEBUG_LEGION
          assert(!ref.is_virtual_ref());
#endif
          LogicalView *view = convert_reference(ref, context);
#ifdef DEBUG_LEGION
          assert(view->is_instance_view());
          assert(view->as_instance_view()->is_reduction_view());
#endif
          ReductionView *new_view = 
            view->as_instance_view()->as_reduction_view();
          const FieldMask &user_mask = ref.get_valid_fields(); 
          // Only add reductions which are not restricted
          if (!!restricted_fields)
          {
            FieldMask unrestricted = user_mask - restricted_fields;
            if (!!unrestricted)
              update_reduction_views(state, unrestricted, new_view);
          }
          else // the normal path here
            update_reduction_views(state, user_mask, new_view);
          // Skip adding the user if requested
          if (defer_add_users)
            continue;
          if (targets.size() > 1)
          {
            // Only find the preconditions now 
            ApEvent ready = new_view->find_user_precondition(usage, term_event,
                                   user_mask, info.op, info.index, 
                                   &info.version_info, info.map_applied_events);
            ref.set_ready_event(ready);
            new_views[idx] = new_view;
          }
          else
          {
            // Do the fused find preconditions and add user
            ApEvent ready = new_view->add_user_fused(usage,term_event,user_mask,
                                         info.op, info.index,&info.version_info,
                                         local_space, info.map_applied_events);
            ref.set_ready_event(ready);
          }
          if (!defer_add_users && !!restricted_fields)
          {
            FieldMask restricted = restricted_fields & user_mask;
              if (!!restricted)
                reduce_out_views[new_view] = restricted;
          }
        }
        if (!defer_add_users && (targets.size() > 1))
        {
          // Second pass of the two pass approach, add our users
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            InstanceRef &ref = targets[idx]; 
            new_views[idx]->add_user(usage, term_event, ref.get_valid_fields(),
                                   info.op, info.index, local_space,
                                   &info.version_info, info.map_applied_events);
          }
        }
        if (!reduce_out_views.empty())
          issue_restricted_reductions(info, restrict_info, restricted_instances,
                                      restricted_views, reduce_out_views);
        // If we have any restricted instances, we can now update the state
        // to reflect that they are going to be the valid instances
        if (!!restricted_fields)
        {
          for (unsigned idx = 0; idx < restricted_views.size(); idx++)
          {
            const FieldMask valid_mask = 
               restricted_instances[idx].get_valid_fields() & restricted_fields;
            if (restricted_views[idx]->is_reduction_view())
            {
              ReductionView *restricted_view = 
                restricted_views[idx]->as_reduction_view();
              update_reduction_views(state, valid_mask, restricted_view);
            }
            else
              update_valid_views(state, valid_mask, true/*dirty*/, 
                                 restricted_views[idx]);
          }
        }
      }
      else
      {
        // Normal instances
        // Get any restricted fields
        FieldMask restricted_fields;
        InstanceSet restricted_instances;
        std::vector<MaterializedView*> restricted_views;
        if (restrict_info.has_restrictions())
        {
          restrict_info.populate_restrict_fields(restricted_fields);
          // We only need to get these things if we are not read-only
          if (!IS_READ_ONLY(info.req))
          {
            restricted_instances = restrict_info.get_instances();
            restricted_views.resize(restricted_instances.size());
            convert_target_views(restricted_instances, context,
                                 restricted_views);
          }
        }
        std::vector<InstanceView*> new_views(targets.size());
        convert_target_views(targets, context, new_views);
        if (!HAS_WRITE_DISCARD(info.req))
        {
          // Any case but write-only
          // All close operations have already been done, so all
          // we need to do is bring our instances up to date for
          // any of their missing fields, and then register our views

          // Finding the valid views is expensive so hold off doing
          // it at first, but as soon as we need it for any instance
          // then do it for all of them and remember the result.
          LegionMap<LogicalView*,FieldMask>::aligned valid_views;
          bool has_valid_views = false;
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            const InstanceRef &ref = targets[idx];
#ifdef DEBUG_LEGION
            assert(!ref.is_virtual_ref());
            assert(new_views[idx]->is_instance_view());
            assert(new_views[idx]->as_instance_view()->is_materialized_view());
#endif
            const FieldMask &valid_fields = ref.get_valid_fields();
            MaterializedView *view = 
              new_views[idx]->as_instance_view()->as_materialized_view();
            // See if this instance is valid already 
            LegionMap<LogicalView*,FieldMask>::aligned::const_iterator
              finder = state->valid_views.find(view);
            if (finder != state->valid_views.end())
            {
              // See which fields if any we actually need to update
              FieldMask needed_fields = valid_fields - finder->second;
              if (!!needed_fields)
              {
                if (!has_valid_views)
                {
                  find_valid_instance_views(info.ctx, state, 
                      info.traversal_mask, info.traversal_mask, 
                      info.version_info, false/*needs space*/, valid_views);
                  has_valid_views = true;
                }
                issue_update_copies(info, view, needed_fields, 
                                    valid_views, restrict_info);
              }
            }
            else
            {
              // Not valid for any fields, so bring it up to date for all fields
              if (!has_valid_views)
              {
                find_valid_instance_views(info.ctx, state,
                    info.traversal_mask, info.traversal_mask,
                    info.version_info, false/*needs space*/, valid_views);
                has_valid_views = true;
              }
              issue_update_copies(info, view, valid_fields, 
                                  valid_views, restrict_info);
            }
            // Finally add this to the set of valid views for the state
            if (!!restricted_fields)
            {
              // We can only update instances which aren't restricted
              FieldMask unrestricted = valid_fields - restricted_fields;
              if (!!unrestricted)
                update_valid_views(state, unrestricted,
                                   !IS_READ_ONLY(info.req)/*dirty*/, view);
            }
            else // The normal path
              update_valid_views(state, valid_fields, 
                                 !IS_READ_ONLY(info.req)/*dirty*/, view); 
          }
        }
        else
        {
          // Write-only case
          // Remove any overlapping reducitons
          if (!(info.traversal_mask * state->reduction_mask))
            invalidate_reduction_views(state, info.traversal_mask);
          // This is write-only so update the valid views on the
          // state with the new instance views while invalidating
          // any old instances with views that interfere with the
          // dirty mask
          if (!!restricted_fields)
          {
            // If we have restricted fields, we have to update them
            // one by one to make sure that we only add instances
            // which aren't restricted
            for (unsigned idx = 0; idx < targets.size(); idx++)
            {
              const InstanceRef &ref = targets[idx];
              FieldMask unrestricted = 
                ref.get_valid_fields() - restricted_fields;
              if (!!unrestricted)
                update_valid_views(state, unrestricted,
                                   true/*dirty*/, new_views[idx]); 
            }
          }
          else // The normal path
            update_valid_views(state, info.traversal_mask, 
                               new_views, targets);
        }
        // Finally we have to update our instance references
        // to get the ready events
        if (!defer_add_users)
        {
          // If there is exactly one instance, then we can do
          // the fused analysis and register user, otherwise we
          // have to make two passes in order to avoid event cycles 
          LegionMap<LogicalView*,FieldMask>::aligned copy_out_views;
          if (targets.size() == 1)
          {
            InstanceRef &ref = targets[0];
#ifdef DEBUG_LEGION
            assert(!ref.is_virtual_ref());
            assert(new_views[0]->is_instance_view());
#endif
            ApEvent ready = new_views[0]->as_instance_view()->add_user_fused(
                usage, term_event, ref.get_valid_fields(), info.op, info.index,
                &info.version_info, local_space, info.map_applied_events);
            ref.set_ready_event(ready);
            if (!!restricted_fields && !IS_READ_ONLY(info.req))
            {
              FieldMask restricted = ref.get_valid_fields() & restricted_fields;
              if (!!restricted)
                copy_out_views[new_views[0]] = restricted;
            }
          }
          else
          {
            // Two pass approach to avoid event cycles 
            for (unsigned idx = 0; idx < targets.size(); idx++)
            {
              InstanceRef &ref = targets[idx];
#ifdef DEBUG_LEGION
              assert(!ref.is_virtual_ref());
              assert(new_views[idx]->is_instance_view());
#endif
              ApEvent ready = 
                new_views[idx]->as_instance_view()->find_user_precondition(
                    usage, term_event, ref.get_valid_fields(), info.op, 
                    info.index, &info.version_info, info.map_applied_events);
              ref.set_ready_event(ready);
            }
            const bool restricted_out = 
              !!restricted_fields && !IS_READ_ONLY(info.req);
            for (unsigned idx = 0; idx < targets.size(); idx++)
            {
              InstanceRef &ref = targets[idx];
              new_views[idx]->as_instance_view()->add_user(usage, term_event,
                       ref.get_valid_fields(), info.op, info.index, 
                       local_space, &info.version_info,info.map_applied_events);
              if (restricted_out)
              {
                FieldMask restricted = 
                  ref.get_valid_fields() & restricted_fields;
                if (!!restricted)
                  copy_out_views[new_views[idx]] = restricted;
              }
            }
          }
          // If we have to do any copy out operations, then we do that now
          if (!copy_out_views.empty())
            issue_restricted_copies(info, restrict_info, restricted_instances, 
                                    restricted_views, copy_out_views);
        }
        // If we have any restricted instances, we can now update the state
        // to reflect that they are going to be the valid instances
        if (!!restricted_fields && !IS_READ_ONLY(info.req))
          update_valid_views(state, restricted_fields, 
                             restricted_views, restricted_instances);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::seed_state(ContextID ctx, ApEvent term_event,
                                const RegionUsage &usage,
                                const FieldMask &user_mask,
                                const InstanceSet &targets,
                                InnerContext *context, unsigned init_index,
                                const std::vector<LogicalView*> &corresponding,
                                std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      get_current_version_manager(ctx).initialize_state(term_event, usage, 
        user_mask, targets, context, init_index, corresponding, applied_events);
    } 

    //--------------------------------------------------------------------------
    void RegionNode::close_state(const TraversalInfo &info, RegionUsage &usage, 
                                 UniqueID logical_ctx_uid,InnerContext *context,
                                 InstanceSet &targets) 
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REGION_NODE_CLOSE_STATE_CALL);
#ifdef DEBUG_LEGION
      assert(!IS_REDUCE(info.req));
#endif
      // All we should actually have to do here is just register
      // our region because any actualy close operations that would
      // need to be done would have been issued as part of the 
      // logical analysis or will be done as part of the registration.
      RestrictInfo empty_restrict_info;
      register_region(info, logical_ctx_uid, context, empty_restrict_info, 
                      ApEvent::NO_AP_EVENT, usage, false/*defer add users*/, 
                      targets, NULL/*no advance close projection info*/);
    }

    //--------------------------------------------------------------------------
    void RegionNode::fill_fields(ContextID ctx, const FieldMask &fill_mask,
                                 const void *value, size_t value_size,
                                 UniqueID logical_ctx_uid,
                                 InnerContext *inner_context,
                                 VersionInfo &version_info,
                                 std::set<RtEvent> &map_applied_events,
                                 PredEvent true_guard, PredEvent false_guard
#ifdef LEGION_SPY
                                 , UniqueID fill_op_uid
#endif
                                 )
    //--------------------------------------------------------------------------
    {
      // A fill is a kind of a write, so we have to do the advance
      // of the version information same as if we had mapped this region
      VersionManager &manager = get_current_version_manager(ctx);
      const bool update_parent_state = 
        !version_info.is_upper_bound_node(this);
      const AddressSpaceID local_space = context->runtime->address_space;
      manager.advance_versions(fill_mask, logical_ctx_uid, inner_context,
                               update_parent_state, map_applied_events);
      // Now record just the advanced versions
      manager.record_advance_versions(fill_mask, inner_context, 
                                      version_info, map_applied_events);
      // Make the fill instance
      DistributedID did = context->runtime->get_available_distributed_id();
      FillView::FillViewValue *fill_value = 
        new FillView::FillViewValue(value, value_size);
      FillView *fill_view = 
        new FillView(context, did, local_space,
                     this, fill_value, true/*register now*/
#ifdef LEGION_SPY
                     , fill_op_uid
#endif
                     );
      // Now update the physical state
      PhysicalState *state = get_physical_state(version_info);
      if (true_guard.exists())
      {
        // Special path for handling speculative execution of predicated fills
#ifdef DEBUG_LEGION
        assert(false_guard.exists());
#endif
        // Build a phi view and register that instead
        DistributedID did = 
          context->runtime->get_available_distributed_id();
        // Copy the version info that we need
        DeferredVersionInfo *view_info = new DeferredVersionInfo();
        version_info.copy_to(*view_info); 
        PhiView *phi_view = new PhiView(context, did, local_space,
                                        view_info, this, true_guard, 
                                        false_guard, inner_context,
                                        true/*register now*/);
        // Record the true and false views
        phi_view->record_true_view(fill_view, fill_mask);
        LegionMap<LogicalView*,FieldMask>::aligned current_views;
        find_valid_instance_views(ctx, state, fill_mask, fill_mask,
                                  version_info, false/*needs space*/, 
                                  current_views);
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              current_views.begin(); it != current_views.end(); it++)
          phi_view->record_false_view(it->first, it->second);
        // Update the state with the phi view
        update_valid_views(state, fill_mask, true/*dirty*/, phi_view);
      }
      else
      {
        // This is the non-predicated path
        // Invalidate any reduction views
        if (!(fill_mask * state->reduction_mask))
          invalidate_reduction_views(state, fill_mask);
        update_valid_views(state, fill_mask, true/*dirty*/, fill_view);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent RegionNode::eager_fill_fields(ContextID ctx, Operation *op,
                                          const unsigned index, 
                                          UniqueID logical_ctx_uid,
                                          InnerContext *context,
                                          const FieldMask &fill_mask,
                                          const void *value, size_t value_size,
                                          VersionInfo &version_info, 
                                          InstanceSet &instances,
                                          ApEvent sync_precondition,
                                          PredEvent true_guard,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      // A fill is a kind of a write, so we have to do the advance
      // of the version information same as if we had mapped this region
      VersionManager &manager = get_current_version_manager(ctx);
      const bool update_parent_state = 
        !version_info.is_upper_bound_node(this);
      const AddressSpaceID local_space = context->runtime->address_space;
      manager.advance_versions(fill_mask, logical_ctx_uid, context,
                               update_parent_state, map_applied_events);
      // Now record just the advanced versions
      manager.record_advance_versions(fill_mask, context, 
                                      version_info, map_applied_events);
      // Effectively a fill is a special kind of copy so we can analyze
      // it the same way to figure out how to issue the fill
      std::set<ApEvent> post_events;
      std::vector<InstanceView*> target_views(instances.size(), NULL);
      convert_target_views(instances, context, target_views);
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        InstanceView *target = target_views[idx];
        LegionMap<ApEvent,FieldMask>::aligned preconditions;
        target->find_copy_preconditions(0/*redop*/, false/*reading*/, 
                          true/*single copy*/, true/*restrict out*/, fill_mask,
                          &version_info, op->get_unique_op_id(), index,
                          local_space, preconditions, map_applied_events);
        if (sync_precondition.exists())
          preconditions[sync_precondition] = fill_mask;
        // Sort the preconditions into event sets
        LegionList<EventSet>::aligned event_sets;
        compute_event_sets(fill_mask, preconditions, event_sets);
        // Iterate over the event sets and issue the fill operations on 
        // the different fields
        for (LegionList<EventSet>::aligned::iterator pit = 
              event_sets.begin(); pit != event_sets.end(); pit++)
        {
          // If we have a predicate guard we add that to the set now
          ApEvent precondition = Runtime::merge_events(pit->preconditions);
          std::vector<CopySrcDstField> dst_fields;
          target->copy_to(pit->set_mask, dst_fields);
          ApEvent fill_event = issue_fill(op, dst_fields, value, 
                                          value_size, precondition, true_guard
#ifdef LEGION_SPY
                                          , op->get_unique_op_id() 
#endif
                                          );
          if (fill_event.exists())
            post_events.insert(fill_event);
        }
        // Add user to make record when everyone is done writing
        target->add_copy_user(0/*redop*/, op->get_completion_event(), 
                              &version_info, op->get_unique_op_id(), 
                              index, fill_mask, false/*reading*/,
                              true/*restrict out*/, local_space, 
                              map_applied_events);
      }
      // Finally do the update to the physical state like a normal fill
      PhysicalState *state = get_physical_state(version_info);
      // Invalidate any reduction views
      if (!(fill_mask * state->reduction_mask))
        invalidate_reduction_views(state, fill_mask);
      update_valid_views(state, fill_mask, target_views, instances);
      // Return the merge of all the post events
      return Runtime::merge_events(post_events);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionNode::attach_external(ContextID ctx, 
                                          InnerContext *parent_ctx, 
                                          const UniqueID logical_ctx_uid,
                                          const FieldMask &attach_mask,
                                          const RegionRequirement &req,
                                          InstanceManager *instance_manager,
                                          VersionInfo &version_info,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      // We're effectively writing to this region by doing the attach
      // so we need to bump the version numbers
      VersionManager &manager = get_current_version_manager(ctx);
      const bool update_parent_state = !version_info.is_upper_bound_node(this);
      manager.advance_versions(attach_mask, logical_ctx_uid, parent_ctx,
                               update_parent_state, map_applied_events);
      // Wrap it in a view
      MaterializedView *view = 
        parent_ctx->create_instance_top_view(instance_manager, 
            context->runtime->address_space)->as_materialized_view();
#ifdef DEBUG_LEGION
      assert(view != NULL);
#endif
      // Update the physical state with the new instance
      PhysicalState *state = get_physical_state(version_info);
      // We need to invalidate all other instances for these fields since
      // we are now making this the only valid copy of the data
      update_valid_views(state, attach_mask, true/*dirty*/, view);
      // Return the resulting instance
      return InstanceRef(instance_manager,attach_mask,ApUserEvent::NO_AP_EVENT);
    }

    //--------------------------------------------------------------------------
    ApEvent RegionNode::detach_external(ContextID ctx, InnerContext *context, 
                                        const UniqueID logical_ctx_uid,  
                                        VersionInfo &version_info, 
                                        const InstanceRef &ref,
                                        std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      InstanceView *view = convert_reference(ref, context);
#ifdef DEBUG_LEGION
      assert(view->is_materialized_view());
#endif
      MaterializedView *detach_view = view->as_materialized_view();
      const FieldMask &detach_mask = 
        detach_view->manager->layout->allocated_fields;
      // A detach is also effectively a bumb of the version numbers because
      // we are marking that the data in the logical region is no longer valid
      VersionManager &manager = get_current_version_manager(ctx);
      const bool update_parent_state = !version_info.is_upper_bound_node(this);
      manager.advance_versions(detach_mask, logical_ctx_uid, context,
                               update_parent_state, map_applied_events);
      // First remove this view from the set of valid views
      PhysicalState *state = get_physical_state(version_info);
      filter_valid_views(state, detach_view);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionNode::find_context_view(PhysicalManager *manager,
                                                InnerContext *context)
    //--------------------------------------------------------------------------
    {
      return convert_reference_region(manager, context);
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionNode::convert_reference_region(PhysicalManager *manager,
                                                       InnerContext *context)
    //--------------------------------------------------------------------------
    {
      InstanceView *result = find_instance_view(manager, context);
      if (result != NULL)
        return result;
      if (manager->region_node == this)
        result = context->create_instance_top_view(manager,
                                              context->runtime->address_space);
      else
      {
#ifdef DEBUG_LEGION
        assert(parent != NULL);
#endif
        InstanceView *parent_view = 
          parent->convert_reference_partition(manager, context);
        result = parent_view->get_instance_subview(row_source->color);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::convert_references_region(
                              const std::vector<PhysicalManager*> &managers,
                              std::vector<bool> &up_mask, InnerContext *context,
                              std::vector<InstanceView*> &results)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(managers.size() == up_mask.size());
#endif
      // See which ones are still good here
      std::vector<unsigned> up_indexes;
      for (unsigned idx = 0; idx < managers.size(); idx++)
      {
        if (!up_mask[idx]) // skip any that are no longer valid
          continue;
        // Check to see if we already have it
        PhysicalManager *manager = managers[idx];
        results[idx] = find_instance_view(manager, context);
        // If we've got it we are done with it
        if (results[idx] != NULL)
        {
          // Mark that it is done
          up_mask[idx] = false;
          continue;
        }
        if (manager->region_node == this)
        {
          results[idx] = 
            context->create_instance_top_view(manager, 
                      context->runtime->address_space);
          // Mark that it is done
          up_mask[idx] = false;
        }
        else
          up_indexes.push_back(idx);
      }
      // See if we have to keep going up
      if (!up_indexes.empty())
      {
#ifdef DEBUG_LEGION
        assert(parent != NULL);
#endif
        parent->convert_references_partition(managers, up_mask, 
                                             context, results);
        for (unsigned idx = 0; idx < up_indexes.size(); idx++)
        {
          unsigned index = up_indexes[idx];
          InstanceView *parent_view = results[index];
          results[index] = parent_view->get_instance_subview(row_source->color);
        }
      }
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
          SemanticRequestArgs args;
          args.proxy_this = this;
          args.tag = tag;
          args.source = source;
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
      forest->attach_semantic_information(handle, tag, source, 
                                          buffer, size, is_mutable);
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
      node->send_node(source);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Serializer rez;
      rez.serialize(done_event);
      forest->runtime->send_top_level_region_return(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_top_level_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
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
      DomainPoint color = row_source->get_domain_point_color();
      switch (color.get_dim())
      {
        case 1:
          {
            logger->log("Region Node (%x,%d,%d) Color %d at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], logger->get_depth());
            break;
          }
        case 2:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d) at "
                        "depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], logger->get_depth());
            break;
          }
        case 3:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d) at "
                        "depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], color[2], logger->get_depth());
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
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
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_logical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_physical_context(ContextID ctx, 
                                            TreeStateLogger *logger,
                                            const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      DomainPoint color = row_source->get_domain_point_color();
      switch (color.get_dim())
      {
        case 1:
          {
            logger->log("Region Node (%x,%d,%d) Color %d at depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], logger->get_depth());
            break;
          }
        case 2:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d) at "
                        "depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], logger->get_depth());
            break;
          }
        case 3:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d) at "
                        "depth %d", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], color[2], logger->get_depth());
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
      if (logical_states.has_entry(ctx))
      {
        VersionManager &manager= get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, to_traverse, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_logical_state(LogicalState &state,
                                         const FieldMask &capture_mask,
                         LegionMap<LegionColor,FieldMask>::aligned &to_traverse,
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
          it->print_state(logger, capture_mask);
          if (it->valid_fields * capture_mask)
            continue;
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator 
                cit = it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
          {
            FieldMask overlap = cit->second & capture_mask;
            if (!overlap)
              continue;
            if (to_traverse.find(cit->first) == to_traverse.end())
              to_traverse[cit->first] = overlap;
            else
              to_traverse[cit->first] |= overlap;
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
            logger->log("Region Node (%x,%d,%d) Color %d at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], logger->get_depth(), this);
            break;
          }
        case 2:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d) at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], logger->get_depth(), this);
            break;
          }
        case 3:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d) at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], color[2], logger->get_depth(), this);
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
      if (logical_states.has_entry(ctx))
        print_logical_state(get_logical_state(ctx), capture_mask,
                            to_traverse, logger);
      else
        logger->log("No state");
      logger->log("");
      if (!to_traverse.empty())
      {
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =  
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_logical_context(ctx, logger, it->second);
        }
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
            logger->log("Region Node (%x,%d,%d) Color %d at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], logger->get_depth(), this);
            break;
          }
        case 2:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d) at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], logger->get_depth(), this);
            break;
          }
        case 3:
          {
            logger->log("Region Node (%x,%d,%d) Color (%d,%d,%d) at "
                        "depth %d (%p)", 
              handle.index_space.id, handle.field_space.id,handle.tree_id,
              color[0], color[1], color[2], logger->get_depth(), this);
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
      if (logical_states.has_entry(ctx))
      {
        VersionManager &manager = get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, to_traverse, logger);
      }
      else
        logger->log("No state");
      logger->log("");
      if (!to_traverse.empty())
      {
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }
#endif

    /////////////////////////////////////////////////////////////
    // Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(LogicalPartition p, RegionNode *par,
                                 IndexPartNode *row_src, 
                                 FieldSpaceNode *col_src,
                                 RegionTreeForest *ctx)
      : RegionTreeNode(ctx, col_src), handle(p), 
        parent(par), row_source(row_src)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Partition %lld %d %d %d %d",
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, 
          handle.get_index_partition().get_id(), 
          handle.get_field_space().get_id(),
          handle.get_tree_id());
#endif
    }

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(const PartitionNode &rhs)
      : RegionTreeNode(NULL, NULL), handle(LogicalPartition::NO_PART),
        parent(NULL), row_source(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PartitionNode::~PartitionNode(void)
    //--------------------------------------------------------------------------
    {
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
    void PartitionNode::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_active);
      currently_active = false;
#endif
      if (registered)
      {
        // Unregister ourselves with our parent
        parent->remove_child(row_source->color);
        if (parent->remove_nested_gc_ref(did))
          delete parent;
        // Unregister ourselves with our row source
        row_source->remove_instance(this);
        if (row_source->remove_nested_resource_ref(did))
          delete row_source;
        // Then unregister ourselves with the context
        context->remove_node(handle);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!registered);
#endif
      LocalReferenceMutator mutator;
      row_source->add_nested_valid_ref(did, &mutator);
      row_source->add_nested_resource_ref(did);
      row_source->add_instance(this);
      parent->add_nested_gc_ref(did, &mutator);
      parent->add_child(this);
      // Add a reference that will be moved when we're destroyed
      add_base_gc_ref(APPLICATION_REF);
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
      return context->create_node(reg_handle, this);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_child(RegionNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(color_map.find(child->row_source->color) == color_map.end());
#endif
      color_map[child->row_source->color] = child;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::remove_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      std::map<LegionColor,RegionNode*>::iterator finder = color_map.find(c);
      assert(finder != color_map.end());
      color_map.erase(finder);
#else
      color_map.erase(c);
#endif
    }

    //--------------------------------------------------------------------------
    void PartitionNode::DestructionFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != source)
        runtime->send_logical_partition_destruction(handle, target);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::destroy_node(AddressSpaceID source, bool root)
    //--------------------------------------------------------------------------
    {
      if (destroyed)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_DESTRUCTION,
            "Duplicate deletion of Logical Partition (%d,%d,%d)",
            handle.index_partition.id, handle.field_space.id, handle.tree_id)
      // If we're the root of the deletion then we might need to send
      // messages to other nodes
      if (root)
      {
        // If we are not on the row source owner node we need to send a message
        // there, otherwise if we are on the owner, we need to send destruction
        // messages out to every node that might have a remote instance
        const AddressSpaceID owner_space = get_owner_space();
        if (owner_space == context->runtime->address_space)
        {
          // Obtain the remote instances from the top of the tree 
          NodeSet target_instances;
          parent->find_remote_instances(target_instances);
          // Send deletions to all but the source
          DestructionFunctor functor(handle, context->runtime, source);
          target_instances.map(functor);
        }
        else if (source != owner_space)
          context->runtime->send_logical_partition_destruction(handle, 
                                                               owner_space);
      }
      // Need to recurse down the tree and do the same thing for our children
      // We need a copy in case we end up deleting anything
      std::map<LegionColor,RegionNode*> children;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        children = color_map;
      }
      for (std::map<LegionColor,RegionNode*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->second->destroy_node(source, false/*root*/))
          delete it->second;
      }
      // Invalidate our version managers
      invalidate_version_managers();
      // Remove the valid reference that we hold on our row source
      row_source->remove_nested_valid_ref(did);
      // Make that it is destroyed
      destroyed = true;
      // Remove our reference
      return remove_base_gc_ref(APPLICATION_REF);
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
    ApEvent PartitionNode::issue_copy(Operation *op,
                        const std::vector<CopySrcDstField> &src_fields,
                        const std::vector<CopySrcDstField> &dst_fields,
                        ApEvent precondition, PredEvent predicate_guard,
                        RegionTreeNode *intersect/*=NULL*/,
                        ReductionOpID redop /*=0*/,bool reduction_fold/*=true*/)
    //--------------------------------------------------------------------------
    {
      return parent->issue_copy(op, src_fields, dst_fields, precondition,
                      predicate_guard, intersect, redop, reduction_fold);
    }

    //--------------------------------------------------------------------------
    ApEvent PartitionNode::issue_fill(Operation *op,
                        const std::vector<CopySrcDstField> &dst_fields,
                        const void *fill_value, size_t fill_size,
                        ApEvent precondition, PredEvent predicate_guard,
#ifdef LEGION_SPY
                        UniqueID fill_uid,
#endif
                        RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
      return parent->issue_fill(op, dst_fields, fill_value, fill_size, 
                                precondition, predicate_guard, 
#ifdef LEGION_SPY
                                fill_uid,
#endif
                                intersect);
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
            for (LegionColor c = 0; c < row_source->max_linearized_color; c++)
            {
              if (!row_source->color_space->contains_color(c))
                continue;
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
    void PartitionNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      bool send_deletion = false;
      {
        AutoLock n_lock(node_lock); 
        if (!remote_instances.contains(target))
        {
          continue_up = true;
          remote_instances.add(target);
        }
        if (destroyed)
          send_deletion = true;
      }
      if (continue_up)
      {
#ifdef DEBUG_LEGION
        assert(parent != NULL);
#endif
        // Send the parent node first
        parent->send_node(target);
        AutoLock n_lock(node_lock);
        for (LegionMap<SemanticTag,SemanticInfo>::aligned::iterator it = 
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
      if (send_deletion)
        context->runtime->send_logical_partition_destruction(handle, target);
    }

    //--------------------------------------------------------------------------
    InstanceView* PartitionNode::find_context_view(PhysicalManager *manager,
                                                   InnerContext *context)
    //--------------------------------------------------------------------------
    {
      return convert_reference_partition(manager, context);
    }

    //--------------------------------------------------------------------------
    InstanceView* PartitionNode::convert_reference_partition(
                                PhysicalManager *manager, InnerContext *context)
    //--------------------------------------------------------------------------
    {
      InstanceView *result = find_instance_view(manager, context);
      if (result != NULL)
        return result;
      // No need to bother with the check here to see if we've arrived
#ifdef DEBUG_LEGION
      assert(parent != NULL);
#endif
      InstanceView *parent_view = 
        parent->convert_reference_region(manager, context);
      result = parent_view->get_instance_subview(row_source->color);
      return result;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::convert_references_partition(
                              const std::vector<PhysicalManager*> &managers,
                              std::vector<bool> &up_mask, InnerContext *context,
                              std::vector<InstanceView*> &results)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent != NULL);
      assert(managers.size() == up_mask.size());
#endif
      // See which ones are still good here
      std::vector<unsigned> up_indexes;
      for (unsigned idx = 0; idx < managers.size(); idx++)
      {
        if (!up_mask[idx])
          continue;
        PhysicalManager *manager = managers[idx];
        results[idx] = find_instance_view(manager, context);
        if (results[idx] != NULL)
        {
          up_mask[idx] = false;
          continue;
        }
        up_indexes.push_back(idx);
      }
      if (!up_indexes.empty())
      {
        parent->convert_references_region(managers, up_mask, context, results);
        for (unsigned idx = 0; idx < up_indexes.size(); idx++)
        {
          unsigned index = up_indexes[idx];
          InstanceView *parent_view = results[index];
          results[index] = parent_view->get_instance_subview(row_source->color);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::perform_disjoint_close(InterCloseOp *op, unsigned index,
                           InnerContext *context, const FieldMask &closing_mask,
                           VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(row_source->is_disjoint());
#endif
      PhysicalState *state = get_physical_state(version_info);
      state->perform_disjoint_close(op, index, context, closing_mask);
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
        LegionMap<SemanticTag,SemanticInfo>::aligned::iterator finder = 
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
          SemanticRequestArgs args;
          args.proxy_this = this;
          args.tag = tag;
          args.source = source;
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
      forest->attach_semantic_information(handle, tag, source, 
                                          buffer, size, is_mutable);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_logical_context(ContextID ctx,
                                              TreeStateLogger *logger,
                                              const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (" IDFMT ",%d,%d) Color %d "
                  "disjoint %d at depth %d", 
        handle.index_partition.id, handle.field_space.id,handle.tree_id,
        row_source->color, row_source->is_disjoint(), logger->get_depth());
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
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
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,RegionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_logical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_physical_context(ContextID ctx,
                                               TreeStateLogger *logger,
                                               const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (" IDFMT ",%d,%d) Color %d "
                  "disjoint %d at depth %d", 
        handle.index_partition.id, handle.field_space.id,handle.tree_id,
        row_source->color, row_source->is_disjoint(), logger->get_depth());
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
      if (logical_states.has_entry(ctx))
      {
        VersionManager &manager = get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, to_traverse, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,RegionNode*>::const_iterator 
            finder = color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_logical_state(LogicalState &state,
                                        const FieldMask &capture_mask,
                   LegionMap<LegionColor,FieldMask>::aligned &to_traverse,
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
          it->print_state(logger, capture_mask);
          if (it->valid_fields * capture_mask)
            continue;
          for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator 
                cit = it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
          {
            FieldMask overlap = cit->second & capture_mask;
            if (!overlap)
              continue;
            if (to_traverse.find(cit->first) == to_traverse.end())
              to_traverse[cit->first] = overlap;
            else
              to_traverse[cit->first] |= overlap;
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
      logger->log("Partition Node (" IDFMT ",%d,%d) Color %d "
                  "disjoint %d at depth %d (%p)", 
        handle.index_partition.id, handle.field_space.id,handle.tree_id,
        row_source->color, row_source->is_disjoint(), logger->get_depth(),this);
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
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
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,RegionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_logical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::dump_physical_context(ContextID ctx,
                                              TreeStateLogger *logger,
                                              const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (" IDFMT ",%d,%d) Color %d "
                  "disjoint %d at depth %d (%p)", 
        handle.index_partition.id, handle.field_space.id,handle.tree_id,
        row_source->color, row_source->is_disjoint(), logger->get_depth(),this);
      logger->down();
      LegionMap<LegionColor,FieldMask>::aligned to_traverse;
      if (logical_states.has_entry(ctx))
      {
        VersionManager &manager = get_current_version_manager(ctx);
        manager.print_physical_state(this, capture_mask, to_traverse, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it =
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<LegionColor,RegionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }
#endif 

  }; // namespace Internal 
}; // namespace Legion

// EOF

