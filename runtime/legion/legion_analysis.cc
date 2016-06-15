/* Copyright 2016 Stanford University, NVIDIA Corporation
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
#include "runtime.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "region_tree.h"
#include "legion_spy.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"
#include "legion_analysis.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Users and Info 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(void)
      : GenericUser(), op(NULL), idx(0), gen(0), timeout(TIMEOUT)
#ifdef LEGION_SPY
        , uid(0)
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(Operation *o, unsigned id, const RegionUsage &u,
                             const FieldMask &m)
      : GenericUser(u, m), op(o), idx(id), 
        gen(o->get_generation()), timeout(TIMEOUT)
#ifdef LEGION_SPY
        , uid(o->get_unique_op_id())
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(void)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const RegionUsage &u, const ColorPoint &c,
                               UniqueID id, unsigned idx)
      : usage(u), child(c), op_id(id), index(idx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const PhysicalUser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalUser::~PhysicalUser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalUser& PhysicalUser::operator=(const PhysicalUser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PhysicalUser::pack_user(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(child);
      rez.serialize(usage.privilege);
      rez.serialize(usage.prop);
      rez.serialize(usage.redop);
      rez.serialize(op_id);
      rez.serialize(index);
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalUser* PhysicalUser::unpack_user(Deserializer &derez,
                                                       bool add_reference)
    //--------------------------------------------------------------------------
    {
      PhysicalUser *result = legion_new<PhysicalUser>();
      derez.deserialize(result->child);
      derez.deserialize(result->usage.privilege);
      derez.deserialize(result->usage.prop);
      derez.deserialize(result->usage.redop);
      derez.deserialize(result->op_id);
      derez.deserialize(result->index);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      if (add_reference)
        result->add_reference();
      return result;
    }

    //--------------------------------------------------------------------------
    TraversalInfo::TraversalInfo(ContextID c, Operation *o, unsigned idx,
                                 const RegionRequirement &r, VersionInfo &info, 
                                 const FieldMask &k, std::set<RtEvent> &e)
      : ctx(c), op(o), index(idx), req(r), version_info(info),
        traversal_mask(k), context_uid(o->get_parent()->get_context_uid()),
        map_applied_events(e)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // FieldVersions 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldVersions::FieldVersions(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldVersions::FieldVersions(const FieldVersions &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FieldVersions::~FieldVersions(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldVersions& FieldVersions::operator=(const FieldVersions &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FieldVersions::add_field_version(VersionID vid, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<VersionID,FieldMask>::aligned::iterator finder = 
        field_versions.find(vid);
      if (finder == field_versions.end())
        field_versions[vid] = mask;
      else
        finder->second |= mask;
    }

    /////////////////////////////////////////////////////////////
    // VersionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionInfo::NodeInfo::NodeInfo(const NodeInfo &rhs)
      : physical_state((rhs.physical_state == NULL) ? NULL : 
                        rhs.physical_state->clone(!rhs.needs_capture(),true)),
        field_versions(rhs.field_versions), advance_mask(rhs.advance_mask),
        bit_mask(rhs.bit_mask) 
    //--------------------------------------------------------------------------
    {
      if (field_versions != NULL)
        field_versions->add_reference();
    }

    //--------------------------------------------------------------------------
    VersionInfo::NodeInfo::~NodeInfo(void)
    //--------------------------------------------------------------------------
    {
      if ((field_versions != NULL) && (field_versions->remove_reference()))
        delete field_versions;
      if (physical_state != NULL)
        delete physical_state;
    }

    //--------------------------------------------------------------------------
    VersionInfo::NodeInfo& VersionInfo::NodeInfo::operator=(const NodeInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(physical_state == NULL);
#endif
      if (rhs.physical_state != NULL)
      {
        if (physical_state != NULL)
          delete physical_state;
        physical_state = rhs.physical_state->clone(!rhs.needs_capture(), true);
      }
      if ((field_versions != NULL) && (field_versions->remove_reference()))
        delete field_versions;
      field_versions = rhs.field_versions;
      if (field_versions != NULL)
        field_versions->add_reference();
      advance_mask = rhs.advance_mask;
      bit_mask = rhs.bit_mask;
      return *this;
    }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(void)
      : upper_bound_node(NULL),
        packed(false), packed_buffer(NULL), packed_size(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(const VersionInfo &rhs)
      : node_infos(rhs.node_infos), upper_bound_node(rhs.upper_bound_node), 
        packed(false), packed_buffer(NULL), packed_size(0)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!rhs.packed); // This shouldn't be called when packed
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        assert(it->second.physical_state == NULL);
      }
#endif
    }

    //--------------------------------------------------------------------------
    VersionInfo::~VersionInfo(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        assert(it->second.physical_state == NULL);
      }
#endif
      // If we still have a buffer, then free it now
      if (packed_buffer != NULL)
        free(packed_buffer);
    }

    //--------------------------------------------------------------------------
    VersionInfo& VersionInfo::operator=(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        assert(it->second.physical_state == NULL);
      }
      // shouldn't be called when packed
      assert(!packed);
#endif
      node_infos = rhs.node_infos;
      upper_bound_node = rhs.upper_bound_node;
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::set_upper_bound_node(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(upper_bound_node == NULL);
      assert(!packed);
#endif
      upper_bound_node = node;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::merge(const VersionInfo &rhs, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
      assert(!rhs.packed);
#endif
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator vit = 
            rhs.node_infos.begin(); vit != 
            rhs.node_infos.end(); vit++)
      {
        const LegionMap<VersionID,FieldMask>::aligned &rhs_versions = 
          vit->second.field_versions->get_field_versions();
        FieldVersions *&entry = node_infos[vit->first].field_versions;
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              rhs_versions.begin(); it != rhs_versions.end(); it++)
        {
          FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          if (entry == NULL)
          {
            entry = new FieldVersions();
            entry->add_reference();
          }
          entry->add_field_version(it->first, it->second);
        }
        if (entry != NULL)
        {
          NodeInfo &next = node_infos[vit->first];
          next.advance_mask |= vit->second.advance_mask;
          next.bit_mask |= vit->second.bit_mask;
          if (vit->second.physical_state != NULL)
            next.physical_state = vit->second.physical_state->clone(mask,
                                          false/*clone state*/, true/*adv*/);
        }
      }
      if (upper_bound_node == NULL)
        upper_bound_node = rhs.upper_bound_node;
      else if (rhs.upper_bound_node != NULL)
      {
        unsigned current_depth = upper_bound_node->get_depth();
        unsigned rhs_depth = rhs.upper_bound_node->get_depth();
        if (current_depth < rhs_depth)
          upper_bound_node = rhs.upper_bound_node;
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::apply_mapping(ContextID ctx, AddressSpaceID target,
                                    std::set<RtEvent> &applied_conditions,
				    bool copy_previous/*=false*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->second.physical_state == NULL)
          continue;
        if (it->second.close_node())
        {
#ifdef DEBUG_LEGION
          // There should be no leave-open nodes here
          assert(!it->second.leave_open());
#endif
          // Only need to apply state if we are the top node
          if (it->second.close_top())
            it->second.physical_state->filter_and_apply(it->second.advance_mask,
                target, false/*filter masks*/, false/*filter views*/,
                true/*filter children*/, NULL/*closed children*/, 
                applied_conditions);
          // we skip other kinds of close nodes
        }
        // Apply path only differently
        else if (it->second.path_only())
	{
	  // HACK: no support for predicated tasks that opened subtrees - wait
	  //  for the new mapping API
	  if (copy_previous)
	    assert(0);
          it->second.physical_state->apply_path_only_state(
                    it->second.advance_mask, target, applied_conditions);
        } 
        else 
        {
	  if (copy_previous)
	  {
	    // we probably didn't premap, so go fetch state
	    assert(!it->second.close_top());
	    it->second.physical_state->capture_state(false/*!path_only*/,
						     it->second.split_node());
	  }
          // Otherwise we can apply with no filter
          it->second.physical_state->apply_state(it->second.advance_mask,
                                             target, applied_conditions);
	}
        // Don't delete it because we need to hold onto the 
        // version manager references in case this operation
        // fails to complete
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::apply_close(ContextID ctx, AddressSpaceID target,
              const LegionMap<ColorPoint,FieldMask>::aligned &closed_children,
                                          std::set<RtEvent> &applied_conditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->second.physical_state == NULL)
          continue;
        // We can skip any nodes we never touched
        if (it->second.needs_capture())
          continue;
        if (it->second.path_only())
        {
          it->second.physical_state->apply_path_only_state(
                      it->second.advance_mask, target, applied_conditions);
          continue;
        }
#ifdef DEBUG_LEGION
        assert(it->second.close_node());
#endif
        if (it->second.close_top())
        {
          // If it is the top node we do the full filter and apply
          it->second.physical_state->filter_and_apply(it->second.advance_mask, 
              target, false/*filter masks*/, false/*filter views*/,
              true/*filter children*/, closed_children.empty() ? NULL : 
                &closed_children, applied_conditions);
        }
        else if (it->second.leave_open())
        {
          // Otherwise if this is a node in the close op that has
          // leave open fields, then we have to apply the state
          it->second.physical_state->filter_and_apply(it->second.advance_mask,
              target, true/*filter masks*/, true/*filter views*/,
              true/*filter children*/, NULL, applied_conditions);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::reset(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it =
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->second.physical_state != NULL)
        {
          // First reset the physical state
          it->second.physical_state->reset();
          it->second.set_needs_capture();
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::release(void)
    //--------------------------------------------------------------------------
    {
      // Might be called when packed, in which case this is a no-op
      // Now it is safe to go through and delete all the physical states
      // which will free up all the references on all the version managers
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->second.physical_state != NULL)
        {
          legion_delete(it->second.physical_state);
          it->second.physical_state = NULL;
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clear(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        assert(it->second.physical_state == NULL);
      }
      assert(!packed);
#endif
      node_infos.clear();
      upper_bound_node = NULL;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::recapture_state(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        it->second.set_needs_capture();
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::sanity_check(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      assert(!packed);
      LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator finder = 
        node_infos.find(node);
      if (finder == node_infos.end())
        return;
      FieldMask previous_fields;
      assert(finder->second.field_versions != NULL);
      const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
        finder->second.field_versions->get_field_versions();
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            field_versions.begin(); it != field_versions.end(); it++)
      {
        assert(previous_fields * it->second);
        previous_fields |= it->second;
      }
    }

    //--------------------------------------------------------------------------
    PhysicalState* VersionInfo::find_physical_state(RegionTreeNode *node,
                                                    bool capture)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator finder = 
        node_infos.find(node);
#ifdef DEBUG_LEGION
      assert(finder != node_infos.end());
#endif
      // Check to see if we need a reset
      if ((finder->second.physical_state != NULL) && capture &&
           finder->second.needs_capture())
      { 
        // Recapture the state if we had to be reset
        finder->second.physical_state->capture_state(finder->second.path_only(),
                                                   finder->second.split_node());
        finder->second.unset_needs_capture();
      }
      return finder->second.physical_state;
    }

    //--------------------------------------------------------------------------
    FieldVersions* VersionInfo::get_versions(RegionTreeNode *node) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator finder = 
        node_infos.find(node);
#ifdef DEBUG_LEGION
      assert(finder != node_infos.end());
#endif
      // It's alright for this to return NULL
      return finder->second.field_versions;
    }

    //--------------------------------------------------------------------------
    const FieldMask& VersionInfo::get_advance_mask(RegionTreeNode *node,
                                                   bool &is_split) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator finder = 
        node_infos.find(node);
#ifdef DEBUG_LEGION
      assert(finder != node_infos.end());
#endif
      is_split = finder->second.split_node();
      return finder->second.advance_mask;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_version_info(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      if (packed_buffer != NULL)
      {
        // If we are already packed this is easy
        rez.serialize(packed_size);
        rez.serialize(packed_buffer, packed_size);
      }
      else
      {
        // Otherwise, make our own local serializer so we
        // can record how many bytes we need
        Serializer local_rez;
        pack_buffer(local_rez);
        size_t total_size = local_rez.get_used_bytes();
        rez.serialize(total_size);
        rez.serialize(local_rez.get_buffer(), total_size);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_version_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
      assert(packed_buffer == NULL);
#endif
      derez.deserialize(packed_size);
      packed_buffer = malloc(packed_size);
      derez.deserialize(packed_buffer, packed_size);
      packed = true;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_version_numbers(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      size_t total_regions = 0;
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->first->is_region())
          total_regions++;
      }
      rez.serialize(total_regions);
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->first->is_region())
        {
          rez.serialize(it->first->as_region_node()->handle);
          pack_node_version_numbers(rez, it->second, it->first);
        }
      }
      size_t total_partitions = node_infos.size() - total_regions;
      rez.serialize(total_partitions);
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (!it->first->is_region())
        {
          rez.serialize(it->first->as_partition_node()->handle);
          pack_node_version_numbers(rez, it->second, it->first);
        }
      }
      if (node_infos.size() > 0)
      {
#ifdef DEBUG_LEGION
        assert(upper_bound_node != NULL);
#endif
        if (upper_bound_node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(upper_bound_node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(upper_bound_node->as_partition_node()->handle);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_version_numbers(Deserializer &derez,
                                             RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
      assert(packed_buffer == NULL);
#endif
      // Unpack the node infos
      size_t num_regions;
      derez.deserialize(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionTreeNode *node = forest->get_node(handle);
        unpack_node_version_numbers(node, derez);
      }
      size_t num_partitions;
      derez.deserialize(num_partitions);
      for (unsigned idx = 0; idx < num_partitions; idx++)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        RegionTreeNode *node = forest->get_node(handle);
        unpack_node_version_numbers(node, derez);
      }
      if ((num_regions > 0) || (num_partitions > 0))
      {
        // Unpack the upper bound node
        bool is_region;
        derez.deserialize(is_region);
        if (is_region)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          upper_bound_node = forest->get_node(handle);
        }
        else
        {
          LogicalPartition handle;
          derez.deserialize(handle);
          upper_bound_node = forest->get_node(handle);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::make_local(std::set<RtEvent> &preconditions, 
                                 Operation *owner_op, RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      if (packed)
        unpack_buffer(owner_op, forest);
      // Iterate over all version state infos and build physical states
      // without actually capturing any data
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        NodeInfo &info = it->second;
        if (info.physical_state == NULL)
        {
          info.physical_state = it->first->get_physical_state(*this,
                                                       false/*capture*/);
          info.set_needs_capture();
        }
        // Now get the preconditions for the state
        info.physical_state->make_local(preconditions, info.needs_final(),
                                        info.close_top());
      }
    } 

    //--------------------------------------------------------------------------
    void VersionInfo::clone_version_info(RegionTreeForest *context,
                 LogicalRegion handle, const VersionInfo &rhs, bool check_below)
    //--------------------------------------------------------------------------
    {
      // Copy over all the version infos from the logical region up to
      // the upper bound node
      RegionTreeNode *current = context->get_node(handle);
#ifdef DEBUG_LEGION
      assert(current != NULL);
#endif
      LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator 
        current_finder = node_infos.find(current);
      if (check_below && (current_finder == node_infos.end()))
      {
        // See if we have any children we need to clone over
        std::set<ColorPoint> child_colors;
        current->get_row_source()->get_colors(child_colors);
        std::deque<RegionTreeNode*> children; 
        for (std::set<ColorPoint>::const_iterator it = child_colors.begin();
              it != child_colors.end(); it++)
        {
          children.push_back(current->get_tree_child(*it));
        }
        child_colors.clear();
        while (!children.empty())
        {
          // Pop off the next child
          RegionTreeNode *child_node = children.front();
          children.pop_front();
          // See if rhs has an entry for this child
          LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator finder =
            rhs.node_infos.find(child_node);
          if (finder != rhs.node_infos.end())
          {
            // Copy it over, and then find all it's children
            node_infos.insert(*finder);
            child_node->get_row_source()->get_colors(child_colors);
            for (std::set<ColorPoint>::const_iterator it = child_colors.begin();
                  it != child_colors.end(); it++)
            {
              children.push_back(child_node->get_tree_child(*it));
            }
            child_colors.clear();   
          }
          // Otherwise we can keep going
        }
      }
      while (current_finder == node_infos.end())
      {
        LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator finder =
          rhs.node_infos.find(current);
#ifdef DEBUG_LEGION
        assert(finder != rhs.node_infos.end());
#endif
        node_infos.insert(*finder);
        if (current == rhs.upper_bound_node)
        {
          upper_bound_node = current;
          break;
        }
        current = current->get_parent();
#ifdef DEBUG_LEGION
        assert(current != NULL);
#endif
        current_finder = node_infos.find(current);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clone_from(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
      assert(!rhs.packed);
#endif
      upper_bound_node = rhs.upper_bound_node;
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator nit = 
            rhs.node_infos.begin(); nit != rhs.node_infos.end(); nit++)
      {
        const NodeInfo &current = nit->second;
#ifdef DEBUG_LEGION
        assert(current.field_versions != NULL);
#endif
        NodeInfo &next = node_infos[nit->first];
#ifdef DEBUG_LEGION
        assert(next.physical_state == NULL);
#endif
#ifdef DEBUG_LEGION
        assert(next.physical_state == NULL); 
#endif
        // Capture the physical state versions, but not the actual state
        if (current.physical_state != NULL)
          next.physical_state = current.physical_state->clone(
              false/*capture state*/, false/*need advance*/);
        next.advance_mask = current.advance_mask;
        next.field_versions = current.field_versions;
        next.field_versions->add_reference();
        next.bit_mask = current.bit_mask & NodeInfo::BASE_FIELDS_MASK;
        // Needs capture is already set
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clone_from(const VersionInfo &rhs,CompositeCloser &closer)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
      assert(!rhs.packed);
#endif
      upper_bound_node = rhs.upper_bound_node;
      // Capture all region tree nodes that have not already been
      // captured by the closer
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator nit = 
            rhs.node_infos.begin(); nit != rhs.node_infos.end(); nit++)
      {
        const NodeInfo &current = nit->second;
#ifdef DEBUG_LEGION
        assert(current.field_versions != NULL);
#endif
        FieldMask clone_mask;         
        const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
          current.field_versions->get_field_versions();
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              field_versions.begin(); it != field_versions.end(); it++) 
        {
          clone_mask |= it->second;
        }
        // Filter this node from the closer
        closer.filter_capture_mask(nit->first, clone_mask);
        if (!clone_mask)
          continue;
        NodeInfo &next = node_infos[nit->first];
#ifdef DEBUG_LEGION
        assert(next.physical_state == NULL); 
#endif
        if (current.physical_state != NULL)
          next.physical_state = current.physical_state->clone(
              clone_mask, false/*capture state*/, false/*need advance*/);
        next.advance_mask = current.advance_mask & clone_mask;
        next.field_versions = current.field_versions;
        next.field_versions->add_reference();
        next.bit_mask = current.bit_mask & NodeInfo::BASE_FIELDS_MASK;
        // Needs capture is already set
      }
#ifdef DEBUG_LEGION
      assert(node_infos.find(upper_bound_node) != node_infos.end());
#endif
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_buffer(Serializer &rez) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!packed);
#endif
      size_t total_regions = 0;
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->first->is_region())
          total_regions++;
      }
      rez.serialize(total_regions);
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->first->is_region())
        {
          rez.serialize(it->first->as_region_node()->handle);
          pack_node_info(rez, it->second, it->first);
        }
      }
      size_t total_partitions = node_infos.size() - total_regions;
      rez.serialize(total_partitions);
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (!it->first->is_region())
        {
          rez.serialize(it->first->as_partition_node()->handle);
          pack_node_info(rez, it->second, it->first);
        }
      }
      if (node_infos.size() > 0)
      {
#ifdef DEBUG_LEGION
        assert(upper_bound_node != NULL);
#endif
        if (upper_bound_node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(upper_bound_node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(upper_bound_node->as_partition_node()->handle);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_buffer(Operation *owner_op,
                                    RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(packed);
      assert(packed_buffer != NULL);
#endif
      Deserializer derez(packed_buffer, packed_size);
      // Unpack the node infos
      size_t num_regions;
      derez.deserialize(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionTreeNode *node = forest->get_node(handle);
        unpack_node_info(owner_op, node, derez);
      }
      size_t num_partitions;
      derez.deserialize(num_partitions);
      for (unsigned idx = 0; idx < num_partitions; idx++)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        RegionTreeNode *node = forest->get_node(handle);
        unpack_node_info(owner_op, node, derez);
      }
      if ((num_regions > 0) || (num_partitions > 0))
      {
        // Unpack the upper bound node
        bool is_region;
        derez.deserialize(is_region);
        if (is_region)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          upper_bound_node = forest->get_node(handle);
        }
        else
        {
          LogicalPartition handle;
          derez.deserialize(handle);
          upper_bound_node = forest->get_node(handle);
        }
      }
      // Keep the buffer around for now in case we need
      // to pack it again later (e.g. for composite views)
      packed = false;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_node_info(Serializer &rez, NodeInfo &info,
                                     RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      rez.serialize(info.bit_mask);
      if (info.physical_state == NULL)
        info.physical_state = node->get_physical_state(*this, false/*capture*/);
      PhysicalState *state = info.physical_state;
#ifdef DEBUG_LEGION
      if (!info.advance_mask)
        assert(state->advance_states.empty());
#endif
      const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
        info.field_versions->get_field_versions();
      rez.serialize<size_t>(field_versions.size()); 
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it1 = 
            field_versions.begin(); it1 != field_versions.end(); it1++)
      {
        rez.serialize(it1->first);
        // Special case for version number 0
        if (it1->first == 0)
        {
          rez.serialize(it1->second);
          continue;
        }
#ifdef DEBUG_LEGION
        assert(state->version_states.find(it1->first) != 
                state->version_states.end());
#endif
        const VersionStateInfo &state_info = state->version_states[it1->first];
        rez.serialize<size_t>(state_info.states.size());
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              state_info.states.begin(); it != state_info.states.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->first->owner_space);
          rez.serialize(it->second);
        }
      }
      size_t total_advance_states = 0;
      if (!state->advance_states.empty())
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              it1 = state->advance_states.begin(); it1 != 
              state->advance_states.end(); it1++)
        {
          const VersionStateInfo &state_info = it1->second;
          total_advance_states += state_info.states.size();
        }
        rez.serialize(total_advance_states);
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              it1 = state->advance_states.begin(); it1 != 
              state->advance_states.end(); it1++)
        {
          const VersionStateInfo &state_info = it1->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                state_info.states.begin(); it != state_info.states.end(); it++)
          {
            rez.serialize(it1->first);
            rez.serialize(it->first->did);
            rez.serialize(it->first->owner_space);
            rez.serialize(it->second);
          }
        }
      }
      else
        rez.serialize(total_advance_states);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_node_info(Operation *owner_op,RegionTreeNode *node,
                                       Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      NodeInfo &info = node_infos[node];
      info.physical_state = legion_new<PhysicalState>(node);
      // Don't need premap
      derez.deserialize(info.bit_mask);
      // Mark that we definitely need to recapture this node info
      info.set_needs_capture();
      // Unpack the version states
      size_t num_versions;
      derez.deserialize(num_versions);
      if (num_versions > 0)
      {
#ifdef DEBUG_LEGION
        assert(info.field_versions == NULL);
#endif
        info.field_versions = new FieldVersions();
        info.field_versions->add_reference();
      }
      const AddressSpaceID local_space = node->context->runtime->address_space;
      for (unsigned idx1 = 0; idx1 < num_versions; idx1++)
      {
        VersionID vid;
        derez.deserialize(vid);
        // Special case for zero
        if (vid == 0)
        {
          FieldMask zero_mask;
          derez.deserialize(zero_mask);
          info.field_versions->add_field_version(0, zero_mask);
          continue;
        }
        FieldMask version_mask;
        size_t num_states;
        derez.deserialize(num_states);
        for (unsigned idx2 = 0; idx2 < num_states; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          AddressSpaceID owner;
          derez.deserialize(owner);
          FieldMask mask;
          derez.deserialize(mask);
          // Transform the field mask
          if (owner != local_space)
          {
            VersionState *state = node->find_remote_version_state(vid, did, 
                                                          owner, owner_op);
            info.physical_state->add_version_state(state, mask);
          }
          else
          {
            DistributedCollectable *dc = 
              node->context->runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
            VersionState *state = dynamic_cast<VersionState*>(dc);
            assert(state != NULL);
#else
            VersionState *state = static_cast<VersionState*>(dc);
#endif
            info.physical_state->add_version_state(state, mask);
          }
          version_mask |= mask;
        }
        info.field_versions->add_field_version(vid, version_mask);
      }
      // Unpack the advance states
      size_t num_states;
      derez.deserialize(num_states);
      for (unsigned idx = 0; idx < num_states; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        DistributedID did;
        derez.deserialize(did);
        AddressSpaceID owner;
        derez.deserialize(owner);
        FieldMask mask;
        derez.deserialize(mask);
        // Transform the field mask
        VersionState *state = node->find_remote_version_state(vid, did, 
                                                      owner, owner_op);
        // No point in adding this to the version state infos
        // since we already know we just use that to build the PhysicalState
        info.physical_state->add_advance_state(state, mask);
        // Update the advance mask as we go
        info.advance_mask |= mask;
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_node_version_numbers(Serializer &rez, NodeInfo &info,
                                                RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
        info.field_versions->get_field_versions();
      rez.serialize<size_t>(field_versions.size()); 
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            field_versions.begin(); it != field_versions.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_node_version_numbers(RegionTreeNode *node,
                                                  Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      NodeInfo &info = node_infos[node];
      size_t num_versions;
      derez.deserialize(num_versions);
      if (num_versions > 0)
      {
#ifdef DEBUG_LEGION
        assert(info.field_versions == NULL);
#endif
        info.field_versions = new FieldVersions();
        info.field_versions->add_reference();
        LegionMap<VersionID,FieldMask>::aligned &versions = 
          info.field_versions->get_mutable_field_versions();
        for (unsigned idx = 0; idx < num_versions; idx++)
        {
          VersionID vid;
          derez.deserialize(vid);
          derez.deserialize(versions[vid]);
        }
      }
    }
    
    /////////////////////////////////////////////////////////////
    // RestrictInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RestrictInfo::RestrictInfo(void)
      : perform_check(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RestrictInfo::RestrictInfo(const RestrictInfo &rhs)
      : perform_check(rhs.perform_check), restrictions(rhs.restrictions)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RestrictInfo::~RestrictInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RestrictInfo& RestrictInfo::operator=(const RestrictInfo &rhs)
    //--------------------------------------------------------------------------
    {
      // Only need to copy over perform_check and restrictions
      perform_check = rhs.perform_check;
      restrictions = rhs.restrictions;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool RestrictInfo::has_restrictions(LogicalRegion handle, RegionNode *node,
                                        const std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      LegionMap<LogicalRegion,FieldMask>::aligned::const_iterator finder = 
        restrictions.find(handle);
      if (finder != restrictions.end())
      {
        FieldMask mask = node->column_source->get_field_mask(fields);
        return (!(mask * finder->second));
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::pack_info(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(restrictions.size());
      for (LegionMap<LogicalRegion,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::unpack_info(Deserializer &derez, AddressSpaceID source,
                                   RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      size_t num_restrictions;
      derez.deserialize(num_restrictions);
      FieldSpaceNode *field_node = NULL;
      for (unsigned idx = 0; idx < num_restrictions; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        FieldMask &mask = restrictions[handle];
        derez.deserialize(mask);
        if (field_node == NULL)
          field_node = forest->get_node(handle)->column_source;
      }
    }

    /////////////////////////////////////////////////////////////
    // PathTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PathTraverser::PathTraverser(RegionTreePath &p)
      : path(p)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PathTraverser::PathTraverser(const PathTraverser &rhs)
      : path(rhs.path)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PathTraverser::~PathTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PathTraverser& PathTraverser::operator=(const PathTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PathTraverser::traverse(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      // Continue visiting nodes and then finding their children
      // until we have traversed the entire path.
      while (true)
      {
#ifdef DEBUG_LEGION
        assert(node != NULL);
#endif
        depth = node->get_depth();
        has_child = path.has_child(depth);
        if (has_child)
          next_child = path.get_child(depth);
        bool continue_traversal = node->visit_node(this);
        if (!continue_traversal)
          return false;
        if (!has_child)
          break;
        node = node->get_tree_child(next_child);
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // LogicalPathRegistrar
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::LogicalPathRegistrar(ContextID c, Operation *o,
                                       const FieldMask &m, RegionTreePath &p)
      : PathTraverser(p), ctx(c), field_mask(m), op(o)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::LogicalPathRegistrar(const LogicalPathRegistrar&rhs)
      : PathTraverser(rhs.path), ctx(0), field_mask(FieldMask()), op(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::~LogicalPathRegistrar(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar& LogicalPathRegistrar::operator=(
                                                const LogicalPathRegistrar &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool LogicalPathRegistrar::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask,false/*dominate*/);
      if (!has_child)
      {
        // If we're at the bottom, fan out and do all the children
        LogicalRegistrar registrar(ctx, op, field_mask, false);
        return node->visit_node(&registrar);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalPathRegistrar::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask,false/*dominate*/);
      if (!has_child)
      {
        // If we're at the bottom, fan out and do all the children
        LogicalRegistrar registrar(ctx, op, field_mask, false);
        return node->visit_node(&registrar);
      }
      return true;
    }


    /////////////////////////////////////////////////////////////
    // LogicalRegistrar
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    LogicalRegistrar::LogicalRegistrar(ContextID c, Operation *o,
                                       const FieldMask &m, bool dom)
      : ctx(c), field_mask(m), op(o), dominate(dom)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegistrar::LogicalRegistrar(const LogicalRegistrar &rhs)
      : ctx(0), field_mask(FieldMask()), op(NULL), dominate(rhs.dominate)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalRegistrar::~LogicalRegistrar(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegistrar& LogicalRegistrar::operator=(const LogicalRegistrar &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool LogicalRegistrar::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool LogicalRegistrar::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask, dominate);
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalRegistrar::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask, dominate);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // CurrentInitializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CurrentInitializer::CurrentInitializer(ContextID c)
      : ctx(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInitializer::CurrentInitializer(const CurrentInitializer &rhs)
      : ctx(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CurrentInitializer::~CurrentInitializer(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInitializer& CurrentInitializer::operator=(
                                                  const CurrentInitializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_current_state(ctx); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_current_state(ctx);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // CurrentInvalidator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CurrentInvalidator::CurrentInvalidator(ContextID c, bool logical_only)
      : ctx(c), logical_users_only(logical_only)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator::CurrentInvalidator(const CurrentInvalidator &rhs)
      : ctx(0), logical_users_only(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator::~CurrentInvalidator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator& CurrentInvalidator::operator=(
                                                  const CurrentInvalidator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx, logical_users_only); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx, logical_users_only);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // DeletionInvalidator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeletionInvalidator::DeletionInvalidator(ContextID c, const FieldMask &dm)
      : ctx(c), deletion_mask(dm)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeletionInvalidator::DeletionInvalidator(const DeletionInvalidator &rhs)
      : ctx(0), deletion_mask(rhs.deletion_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DeletionInvalidator::~DeletionInvalidator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeletionInvalidator& DeletionInvalidator::operator=(
                                                 const DeletionInvalidator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_deleted_state(ctx, deletion_mask); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_deleted_state(ctx, deletion_mask);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // RestrictionMutator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RestrictionMutator::RestrictionMutator(ContextID c, const FieldMask &mask,
                                           bool add)
      : ctx(c), restrict_mask(mask), add_restrict(add)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool RestrictionMutator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool RestrictionMutator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      if (add_restrict)
        node->add_restriction(ctx, restrict_mask);
      else
        node->release_restriction(ctx, restrict_mask);
      return true;
    }

    //--------------------------------------------------------------------------
    bool RestrictionMutator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      if (add_restrict)
        node->add_restriction(ctx, restrict_mask);
      else
        node->release_restriction(ctx, restrict_mask);
      return true;
    } 

    /////////////////////////////////////////////////////////////
    // ReductionCloser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionCloser::ReductionCloser(ContextID c, ReductionView *t,
                                     const FieldMask &m, VersionInfo &info, 
                                     Operation *o, unsigned idx,
                                     std::set<RtEvent> &e)
      : ctx(c), target(t), close_mask(m), version_info(info), op(o), 
        index(idx), map_applied_events(e)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReductionCloser::ReductionCloser(const ReductionCloser &rhs)
      : ctx(0), target(NULL), close_mask(FieldMask()), 
        version_info(rhs.version_info), op(NULL), index(0),
        map_applied_events(rhs.map_applied_events)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReductionCloser::~ReductionCloser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReductionCloser& ReductionCloser::operator=(const ReductionCloser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReductionCloser::issue_close_reductions(RegionTreeNode *node,
                                                 PhysicalState *state)
    //--------------------------------------------------------------------------
    {
      LegionMap<ReductionView*,FieldMask>::aligned valid_reductions;
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            state->reduction_views.begin(); it != 
            state->reduction_views.end(); it++)
      {
        // Skip our target
        if (it->first == target)
          continue;
        // If we already issued it then we don't need to do it again
        // because reduction views only work from a single level
        if (issued_reductions.find(it->first) != issued_reductions.end())
          continue;
        FieldMask overlap = it->second & close_mask;
        if (!!overlap)
          valid_reductions[it->first] = overlap;
        issued_reductions.insert(it->first);
      }
      if (!valid_reductions.empty())
        node->issue_update_reductions(target, close_mask, version_info,
                          valid_reductions, op, index, map_applied_events);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTraverser::PhysicalTraverser(RegionTreePath &p, 
                                         TraversalInfo *in, InstanceSet *t)
      : PathTraverser(p), info(in), targets(t)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalTraverser::PhysicalTraverser(const PhysicalTraverser &rhs)
      : PathTraverser(rhs.path), info(NULL), targets(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalTraverser::~PhysicalTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalTraverser& PhysicalTraverser::operator=(const PhysicalTraverser &rs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTraverser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      return traverse_node(node);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTraverser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      return traverse_node(node);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTraverser::traverse_node(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      PhysicalState *state = node->get_physical_state(info->version_info); 
      // If we are traversing an intermediary node, we just have to 
      // update the open children
      if (has_child)
      {
        state->children.valid_fields |= info->traversal_mask;
        LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
                            state->children.open_children.find(next_child);
        if (finder == state->children.open_children.end())
          state->children.open_children[next_child] = info->traversal_mask;
        else
          finder->second |= info->traversal_mask;
      }
      else if (!IS_REDUCE(info->req))
      {
        // We're at the child node, see if we need to find the valid 
        // instances or not, if not then we are done
        if (targets != NULL)
        {
          node->pull_valid_instance_views(info->ctx, state, 
              info->traversal_mask, true/*needs space*/, info->version_info);
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
             cur_view->manager->layout->allocated_fields & info->traversal_mask;
            if (!containing_fields)
              continue;
            // Now see if it has any valid fields already
            FieldMask valid_fields = it->second & info->traversal_mask;
            // Save the reference
            targets->add_instance(InstanceRef(cur_view->manager, valid_fields));
          }
        }
      }
      else
      {
        // See if there are any reduction instances that match locally
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
              state->reduction_views.begin(); it != 
              state->reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & info->traversal_mask;
          if (!overlap)
            continue;
          targets->add_instance(InstanceRef(it->first->manager, overlap));
        }
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // CurrentState 
    /////////////////////////////////////////////////////////////
    
    // C++ is dumb
    const VersionID CurrentState::init_version;

    //--------------------------------------------------------------------------
    CurrentState::CurrentState(RegionTreeNode *node, ContextID ctx)
      : owner(node), state_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      // This first time we create the state, we need to pull down
      // any restricted instances from our parent state
      RegionTreeNode *parent = node->get_parent();
      if (parent != NULL)
        parent->set_restricted_fields(ctx, restricted_fields);
    }

    //--------------------------------------------------------------------------
    CurrentState::CurrentState(const CurrentState &rhs)
      : owner(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CurrentState::~CurrentState(void)
    //--------------------------------------------------------------------------
    {
      state_lock.destroy_reservation();
      state_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    CurrentState& CurrentState::operator=(const CurrentState&rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* CurrentState::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<CurrentState,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void* CurrentState::operator new[](size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<CurrentState,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void CurrentState::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void CurrentState::operator delete[](void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void CurrentState::check_init(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_states.empty());
      assert(curr_epoch_users.empty());
      assert(prev_epoch_users.empty());
      assert(current_version_infos.empty());
      assert(previous_version_infos.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void CurrentState::clear_logical_users(void)
    //--------------------------------------------------------------------------
    {
      if (!curr_epoch_users.empty())
      {
        for (LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned::
              const_iterator it = curr_epoch_users.begin(); it != 
              curr_epoch_users.end(); it++)
        {
          it->op->remove_mapping_reference(it->gen); 
        }
        curr_epoch_users.clear();
      }
      if (!prev_epoch_users.empty())
      {
        for (LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned::
              const_iterator it = prev_epoch_users.begin(); it != 
              prev_epoch_users.end(); it++)
        {
          it->op->remove_mapping_reference(it->gen); 
        }
        prev_epoch_users.clear();
      }
    }

    //--------------------------------------------------------------------------
    void CurrentState::reset(void)
    //--------------------------------------------------------------------------
    {
      field_states.clear();
      clear_logical_users(); 
      restricted_fields.clear();
      dirty_below.clear();
      partially_closed.clear();
      if (!current_version_infos.empty())
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = current_version_infos.begin(); vit != 
              current_version_infos.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
                info.states.begin(); it != info.states.end(); it++)
          {
            if (it->first->remove_base_valid_ref(CURRENT_STATE_REF))
              legion_delete(it->first);
          }
        }
        current_version_infos.clear();
      }
      if (!previous_version_infos.empty())
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = previous_version_infos.begin(); vit != 
              previous_version_infos.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
                info.states.begin(); it != info.states.end(); it++)
          {
            if (it->first->remove_base_valid_ref(CURRENT_STATE_REF))
              legion_delete(it->first);
          }
        }
        previous_version_infos.clear();
      }
      outstanding_reduction_fields.clear();
      outstanding_reductions.clear();
    } 

    //--------------------------------------------------------------------------
    void CurrentState::clear_deleted_state(const FieldMask &deleted_mask)
    //--------------------------------------------------------------------------
    {
      for (LegionList<FieldState>::aligned::iterator it = field_states.begin();
            it != field_states.end(); /*nothing*/)
      {
        it->valid_fields -= deleted_mask;
        if (!it->valid_fields)
        {
          it = field_states.erase(it);
          continue;
        }
        std::vector<ColorPoint> to_delete;
        for (LegionMap<ColorPoint,FieldMask>::aligned::iterator child_it = 
              it->open_children.begin(); child_it != 
              it->open_children.end(); child_it++)
        {
          child_it->second -= deleted_mask;
          if (!child_it->second)
            to_delete.push_back(child_it->first);
        }
        if (!to_delete.empty())
        {
          for (std::vector<ColorPoint>::const_iterator cit = to_delete.begin();
                cit != to_delete.end(); cit++)
            it->open_children.erase(*cit);
        }
        if (!it->open_children.empty())
          it++;
        else
          it = field_states.erase(it);
      }
      // Don't invalidate users so later deletions can see dependences too
      if (!current_version_infos.empty())
      {
        std::vector<VersionID> versions_to_delete;
        for (LegionMap<VersionID,VersionStateInfo>::aligned::iterator 
              vit = current_version_infos.begin(); vit != 
              current_version_infos.end(); vit++)
        {
          VersionStateInfo &info = vit->second;
          info.valid_fields -= deleted_mask;
          std::vector<VersionState*> states_to_delete;
          for (LegionMap<VersionState*,FieldMask>::aligned::iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            it->second -= deleted_mask;
            if (!it->second)
              states_to_delete.push_back(it->first);
          }
          if (!states_to_delete.empty())
          {
            for (std::vector<VersionState*>::iterator it = 
                  states_to_delete.begin(); it != states_to_delete.end(); it++)
            {
              info.states.erase(*it);
              if ((*it)->remove_base_valid_ref(CURRENT_STATE_REF))
                legion_delete(*it);
            }
          }
          if (info.states.empty())
            versions_to_delete.push_back(vit->first);
        }
        if (!versions_to_delete.empty())
        {
          for (std::vector<VersionID>::const_iterator it = 
                versions_to_delete.begin(); it != 
                versions_to_delete.end(); it++)
          {
            current_version_infos.erase(*it);
          }
        }
      }
      if (!previous_version_infos.empty())
      {
        std::vector<VersionID> versions_to_delete;
        for (LegionMap<VersionID,VersionStateInfo>::aligned::iterator 
              vit = previous_version_infos.begin(); vit != 
              previous_version_infos.end(); vit++)
        {
          VersionStateInfo &info = vit->second;
          info.valid_fields -= deleted_mask;
          std::vector<VersionState*> states_to_delete;
          for (LegionMap<VersionState*,FieldMask>::aligned::iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            it->second -= deleted_mask;
            if (!it->second)
              states_to_delete.push_back(it->first);
          }
          if (!states_to_delete.empty())
          {
            for (std::vector<VersionState*>::iterator it = 
                  states_to_delete.begin(); it != states_to_delete.end(); it++)
            {
              info.states.erase(*it);
              if ((*it)->remove_base_valid_ref(CURRENT_STATE_REF))
                legion_delete(*it);
            }
          }
          if (info.states.empty())
            versions_to_delete.push_back(vit->first);
        }
        if (!versions_to_delete.empty())
        {
          for (std::vector<VersionID>::const_iterator it = 
                versions_to_delete.begin(); it != 
                versions_to_delete.end(); it++)
          {
            previous_version_infos.erase(*it);
          }
        }
      }
      outstanding_reduction_fields -= deleted_mask;
      if (!outstanding_reductions.empty())
      {
        std::vector<ReductionOpID> to_delete;
        for (LegionMap<ReductionOpID,FieldMask>::aligned::iterator it = 
              outstanding_reductions.begin(); it != 
              outstanding_reductions.end(); it++)
        {
          it->second -= deleted_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<ReductionOpID>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          outstanding_reductions.erase(*it);
        }
      }
      dirty_below -= deleted_mask;
      partially_closed -= deleted_mask;
      restricted_fields -= deleted_mask;
    }

    //--------------------------------------------------------------------------
    void CurrentState::sanity_check(void)
    //--------------------------------------------------------------------------
    {
      // The partially closed fields should be a subset of the
      // fields that are dirty below
      assert(!(partially_closed - dirty_below));
      // This code is a sanity check that each field appears for at most
      // one version number
      FieldMask current_version_fields;
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator vit =
            current_version_infos.begin(); vit != 
            current_version_infos.end(); vit++)
      {
        const VersionStateInfo &info = vit->second;
        assert(!!info.valid_fields);
        // Make sure each field appears once in each version state info
        FieldMask local_version_fields;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              info.states.begin(); it != info.states.end(); it++)
        {
          assert(!!it->second); // better not be empty
          assert(local_version_fields * it->second); // better not overlap
          local_version_fields |= it->second;
        }
        assert(info.valid_fields == local_version_fields); // beter match
        // Should not overlap with other fields in the current version
        assert(current_version_fields * info.valid_fields);
        current_version_fields |= info.valid_fields;
      }
      FieldMask previous_version_fields;
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator vit =
            previous_version_infos.begin(); vit != 
            previous_version_infos.end(); vit++)
      {
        const VersionStateInfo &info = vit->second;
        assert(!!info.valid_fields);
        // Make sure each field appears once in each version state info
        FieldMask local_version_fields;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              info.states.begin(); it != info.states.end(); it++)
        {
          assert(!!it->second); // better not be empty
          assert(local_version_fields * it->second); // better not overlap
          local_version_fields |= it->second;
        }
        assert(info.valid_fields == local_version_fields); // beter match
        // Should not overlap with other fields in the current version
        assert(previous_version_fields * info.valid_fields);
        previous_version_fields |= info.valid_fields;
      }
    }

    //--------------------------------------------------------------------------
    void CurrentState::initialize_state(ApEvent term_event,
                                        const RegionUsage &usage,
                                        const FieldMask &user_mask,
                                        const InstanceSet &targets,
                                        SingleTask *context,unsigned init_index,
                                 const std::vector<LogicalView*> &corresponding)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_version_infos.empty() || 
              (current_version_infos.size() == 1));
      assert(previous_version_infos.empty());
#endif
      // No need to hold the lock when initializing
      if (current_version_infos.empty())
      {
        VersionState *init_state = 
          owner->create_new_version_state(init_version);
        init_state->add_base_valid_ref(CURRENT_STATE_REF, context);
        init_state->initialize(term_event, usage, user_mask, 
                               targets, context, init_index, corresponding);
        current_version_infos[init_version].valid_fields = user_mask;
        current_version_infos[init_version].states[init_state] = user_mask;
      }
      else
      {
        LegionMap<VersionID,VersionStateInfo>::aligned::iterator finder = 
          current_version_infos.find(init_version);
#ifdef DEBUG_LEGION
        assert(finder != current_version_infos.end());
#endif
        finder->second.valid_fields |= user_mask;
#ifdef DEBUG_LEGION
        assert(finder->second.states.size() == 1);
#endif
        LegionMap<VersionState*,FieldMask>::aligned::iterator it = 
          finder->second.states.begin();
        it->first->initialize(term_event, usage, user_mask,
                              targets, context, init_index, corresponding);
        it->second |= user_mask;
      }
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void CurrentState::record_version_numbers(const FieldMask &mask,
                                              const LogicalUser &user,
                                              VersionInfo &version_info,
                                              bool capture_previous, 
                                              bool path_only, bool needs_final,
                                              bool close_top, bool report,
                                              bool close_node, bool leave_open,
                                              bool split_node)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(owner->context->runtime, 
                        CURRENT_STATE_RECORD_VERSION_NUMBERS_CALL);
#ifdef DEBUG_LEGION
      sanity_check();
      version_info.sanity_check(owner);
      assert(!close_top || !leave_open);
      // only close nodes can be split
      assert(!split_node || close_node);
      // only close nodes can be leave open
      assert(!leave_open || close_node);
      // close top must be a close node
      assert(!close_top || close_node);
#endif
      // Capture the version information for this logical region  
      VersionInfo::NodeInfo &node_info = 
        version_info.find_tree_node_info(owner);
      if (path_only)
        node_info.set_path_only();
      if (needs_final)
        node_info.set_needs_final();
      if (close_top)
        node_info.set_close_top();
      if (close_node)
        node_info.set_close_node();
      if (leave_open)
        node_info.set_leave_open();
      // Path only nodes that capture previous are split
      if (split_node || (path_only && capture_previous))
        node_info.set_split_node();
      if (capture_previous)
        node_info.advance_mask |= mask;
      if (node_info.physical_state == NULL)
        node_info.physical_state = legion_new<PhysicalState>(owner);
      PhysicalState *state = node_info.physical_state;
      FieldMask unversioned = mask;
      if (capture_previous)
      {
        // Iterate over the previous versions
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = previous_version_infos.begin(); vit != 
              previous_version_infos.end(); vit++)
        {
          FieldMask overlap = vit->second.valid_fields & mask;
          if (!overlap)
            continue;
          unversioned -= overlap;
          if (node_info.field_versions == NULL)
          {
            node_info.field_versions = new FieldVersions();
            node_info.field_versions->add_reference();
          }
          node_info.field_versions->add_field_version(vit->first, overlap);
          // Now record all the previous and advance version states
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = vit->second.states.begin(); it != 
                vit->second.states.end(); it++)
          {
            FieldMask state_overlap = it->second & overlap;
            if (!state_overlap)
              continue;
            state->add_version_state(it->first, state_overlap);
          }
          LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator
            finder = current_version_infos.find(vit->first+1);
#ifdef DEBUG_LEGION
          assert(finder != current_version_infos.end());
          assert(!(overlap - finder->second.valid_fields));
#endif
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator
                it = finder->second.states.begin(); it != 
                finder->second.states.end(); it++)
          {
            FieldMask state_overlap = it->second & overlap;
            if (!state_overlap)
              continue;
            state->add_advance_state(it->first, state_overlap);
          }
        }
        // If we had any unversioned fields, it is because we don't track
        // version number 0 explicity, so go find version number 1 in the
        // set of current version numbers, it should dominate, if not there
        // is something wrong with our implementation
        if (!!unversioned)
        {
          if (node_info.field_versions == NULL)
          {
            node_info.field_versions = new FieldVersions();
            node_info.field_versions->add_reference();
          }
          // The field version we record should be field zero
          node_info.field_versions->add_field_version(0,unversioned);
          VersionStateInfo &info = current_version_infos[init_version];
          // See if we have any fields to test
          if (!(info.valid_fields * unversioned))
          {
            for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                  it = info.states.begin(); it != info.states.end(); it++)
            {
              FieldMask state_overlap = it->second & unversioned;
              if (!state_overlap)
                continue;
              state->add_advance_state(it->first, state_overlap);
            }
            unversioned -= info.valid_fields;
          }
          // If we still have unversioned states, we need to make a new state
          if (!!unversioned)
          {
            VersionState *init_state = 
              owner->create_new_version_state(init_version);
            init_state->add_base_valid_ref(CURRENT_STATE_REF);
            info.states[init_state] = unversioned;
            info.valid_fields |= unversioned;
            state->add_advance_state(init_state, unversioned);
          }
        }
      }
      else
      {
        // Iterate over the current versions
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator
              vit = current_version_infos.begin(); vit !=
              current_version_infos.end(); vit++)
        {
          FieldMask overlap = vit->second.valid_fields & mask;
          if (!overlap)
            continue;
          unversioned -= overlap;
          if (node_info.field_versions == NULL)
          {
            node_info.field_versions = new FieldVersions();
            node_info.field_versions->add_reference();
          }
          node_info.field_versions->add_field_version(vit->first, overlap);
          // Only need to capture the current version infos
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator
                it = vit->second.states.begin(); it !=
                vit->second.states.end(); it++)
          {
            FieldMask state_overlap = it->second & overlap;
            if (!state_overlap)
              continue;
            state->add_version_state(it->first, state_overlap);
          }
        }
        if (!!unversioned)
        {
          // For now, we won't report unversioned warnings for 
          // simultaneous or relaxed coherence as the runtime can't
          // actually infer if there are other users that might
          // be writing to the same region.
          if (report && !IS_SIMULT(user.usage) && !IS_RELAXED(user.usage))
            owner->report_uninitialized_usage(user, unversioned);
          // Make version number 1 here for us to use
          if (node_info.field_versions == NULL)
          {
            node_info.field_versions = new FieldVersions();
            node_info.field_versions->add_reference();
          }
          node_info.field_versions->add_field_version(0,unversioned);
          VersionStateInfo &info = current_version_infos[init_version];
          VersionState *init_state = 
            owner->create_new_version_state(init_version);
          init_state->add_base_valid_ref(CURRENT_STATE_REF);
          info.states[init_state] = unversioned;
          info.valid_fields |= unversioned;
          state->add_version_state(init_state, unversioned);
        }
      }
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void CurrentState::advance_version_numbers(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(owner->context->runtime, 
                        CURRENT_STATE_ADVANCE_VERSION_NUMBERS_CALL);
#ifdef DEBUG_LEGION
      sanity_check();
#endif
      // First filter out fields in the previous
      std::vector<VersionID> to_delete_previous;
      FieldMask previous_filter = mask;
      for (LegionMap<VersionID,VersionStateInfo>::aligned::iterator vit =
            previous_version_infos.begin(); vit != 
            previous_version_infos.end(); vit++)
      {
        FieldMask overlap = vit->second.valid_fields & previous_filter;
        if (!overlap)
          continue;
        VersionStateInfo &info = vit->second;
        info.valid_fields -= overlap;
        // See if everyone is going away or just some of them
        if (!info.valid_fields)
        {
          // The whole version number is going away, remove all
          // the valid references on the version state objects
          to_delete_previous.push_back(vit->first);
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            if (it->first->remove_base_valid_ref(CURRENT_STATE_REF))
              legion_delete(it->first);
          }
        }
        else
        {
          // Only some of the state are being filtered
          std::vector<VersionState*> to_delete;
          for (LegionMap<VersionState*,FieldMask>::aligned::iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->second -= overlap;
            if (!it->second)
            {
              to_delete.push_back(it->first);
              if (it->first->remove_base_valid_ref(CURRENT_STATE_REF))
                legion_delete(it->first);
            }
          }
          if (!to_delete.empty())
          {
            for (std::vector<VersionState*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              info.states.erase(*it);
            }
          }
        }
        previous_filter -= overlap;
        if (!previous_filter)
          break;
      }
      if (!to_delete_previous.empty())
      {
        for (std::vector<VersionID>::const_iterator it = 
              to_delete_previous.begin(); it != to_delete_previous.end(); it++)
        {
          previous_version_infos.erase(*it);
        }
        to_delete_previous.clear();
      }
      // Now filter fields from current back to previous
      FieldMask current_filter = mask;
      // Keep a set of version states that have to be added
      LegionMap<VersionState*,FieldMask>::aligned to_add;
      std::set<VersionID> to_delete_current;
      // Do this in reverse order so we can add new states earlier
      for (LegionMap<VersionID,VersionStateInfo>::aligned::reverse_iterator 
            vit = current_version_infos.rbegin(); vit != 
            current_version_infos.rend(); vit++)
      {
        FieldMask overlap = vit->second.valid_fields & current_filter;
        if (!overlap)
          continue;
        VersionStateInfo &info = vit->second;
        info.valid_fields -= overlap;
        if (!info.valid_fields)
        {
          to_delete_current.insert(vit->first);
          // Send back the whole version state info to previous
          // See if we need to merge it or can just copy it
          LegionMap<VersionID,VersionStateInfo>::aligned::iterator prev_finder =
            previous_version_infos.find(vit->first);
          if (prev_finder == previous_version_infos.end())
          {
            // Can just send it back with no merge
            VersionStateInfo &prev_info = previous_version_infos[vit->first];
            prev_info = info;
            prev_info.valid_fields = overlap;
          }
          else
          {
            VersionStateInfo &prev_info = prev_finder->second;
            // Filter back the version states
            for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                  it = info.states.begin(); it != info.states.end(); it++)
            {
              // See if we find it 
              LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
                prev_info.states.find(it->first);
              if (finder != prev_info.states.end())
              {
                finder->second |= it->second;
                // Remove duplicate reference
                it->first->remove_base_valid_ref(CURRENT_STATE_REF);
              }
              else // didn't find it, just insert it
                prev_info.states.insert(*it);
            }
            prev_info.valid_fields |= overlap;
          }
          // always make sure we clear the states in case we actually
          // make a new version state info later
          info.states.clear();
        }
        else
        {
          // Filter back the only some of the version states
          // See if there is some one to filter back to
          std::vector<VersionState*> to_delete;
          LegionMap<VersionID,VersionStateInfo>::aligned::iterator prev_finder =
            previous_version_infos.find(vit->first);
          if (prev_finder == previous_version_infos.end())
          {
            // Make a new version state info
            VersionStateInfo &prev_info = previous_version_infos[vit->first];
            for (LegionMap<VersionState*,FieldMask>::aligned::iterator
                  it = info.states.begin(); it != info.states.end(); it++)
            {
              FieldMask state_overlap = it->second & overlap;
              if (!state_overlap)
                continue;
              prev_info.states[it->first] = state_overlap;
              it->second -= state_overlap;
              if (!it->second) // Whole state flows back including reference
                to_delete.push_back(it->first);
              else // Partial flow back so add a reference
                it->first->add_base_valid_ref(CURRENT_STATE_REF);
            }
            prev_info.valid_fields = overlap;
          }
          else
          {
            VersionStateInfo &prev_info = prev_finder->second;
            for (LegionMap<VersionState*,FieldMask>::aligned::iterator
                  it = info.states.begin(); it != info.states.end(); it++)
            {
              FieldMask state_overlap = it->second & overlap;
              if (!state_overlap)
                continue;
              it->second -= state_overlap;
              if (!it->second)
              {
                to_delete.push_back(it->first);
                // Whole state flows back
                LegionMap<VersionState*,FieldMask>::aligned::iterator 
                  finder = prev_info.states.find(it->first);
                if (finder != prev_info.states.end())
                {
                  // Already exists, merge it back and remove duplicate ref
                  finder->second |= state_overlap;
                  it->first->remove_base_valid_ref(CURRENT_STATE_REF);
                }
                else // just send it back including reference
                  prev_info.states[it->first] = state_overlap;
              }
              else
              {
                // Partial state flows back  
                LegionMap<VersionState*,FieldMask>::aligned::iterator 
                  finder = prev_info.states.find(it->first);
                if (finder == prev_info.states.end())
                {
                  prev_info.states[it->first] = state_overlap;
                  // Add an extra reference
                  it->first->add_base_valid_ref(CURRENT_STATE_REF);
                }
                else
                  finder->second |= state_overlap;
              }
            }
            prev_info.valid_fields |= overlap;
          }
          if (!to_delete.empty())
          {
            for (std::vector<VersionState*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              info.states.erase(*it);
            }
          }
        }
        // Make our new version state object and add if we can
        VersionID next_version = vit->first+1;
        // Remove this version number from the delete set
        to_delete_current.erase(next_version);
        VersionState *new_state = 
          owner->create_new_version_state(next_version);
        // Add our reference now
        new_state->add_base_valid_ref(CURRENT_STATE_REF);
        // Kind of dangerous to be getting another iterator to this
        // data structure that we're iterating, but since neither
        // is mutating, we won't invalidate any iterators
        LegionMap<VersionID,VersionStateInfo>::aligned::iterator 
          next_finder = current_version_infos.find(next_version);
        if (next_finder != current_version_infos.end())
        {
          // We know it doesn't exist yet
#ifdef DEBUG_LEGION
          // Just to be completely safe
          assert(next_finder->second.states.find(new_state) ==
                 next_finder->second.states.end());
#endif
          next_finder->second.states[new_state] = overlap; 
          next_finder->second.valid_fields |= overlap;
        }
        else
          to_add[new_state] = overlap;
        current_filter -= overlap;
        if (!current_filter)
          break;
      }
      // See if we have any fields for which there was no prior
      // version number, if there was then these are fields which
      // are being initialized and should be added as version 1
      if (!!current_filter)
      {
        VersionState *new_state = 
          owner->create_new_version_state(init_version);
        new_state->add_base_valid_ref(CURRENT_STATE_REF);
        VersionStateInfo &info = current_version_infos[init_version];
        info.states[new_state] = current_filter;
        info.valid_fields |= current_filter;
      }
      // Remove any old version state infos
      if (!to_delete_current.empty())
      {
        for (std::set<VersionID>::const_iterator it = 
              to_delete_current.begin(); it != to_delete_current.end(); it++)
        {
          current_version_infos.erase(*it);
        }
      }
      // Finally add in our new states
      if (!to_add.empty())
      {
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
              it = to_add.begin(); it != to_add.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(current_version_infos.find(it->first->version_number) ==
                 current_version_infos.end());
#endif
          VersionStateInfo &info = 
            current_version_infos[it->first->version_number];
          info.states[it->first] = it->second;
          info.valid_fields = it->second;
        }
      }
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    } 

    //--------------------------------------------------------------------------
    void CurrentState::print_physical_state(RegionTreeNode *node,
                                            const FieldMask &capture_mask,
                          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                            TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      PhysicalState temp_state(node);
      logger->log("Versions:");
      logger->down();
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator vit = 
            current_version_infos.begin(); vit != 
            current_version_infos.end(); vit++)
      {
        if (capture_mask * vit->second.valid_fields)
          continue;
        FieldMask version_fields;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              vit->second.states.begin(); it != vit->second.states.end(); it++)
        {
          FieldMask overlap = capture_mask & it->second;
          if (!overlap)
            continue;
          version_fields |= overlap;
          it->first->update_physical_state(&temp_state, overlap);
        }
        assert(!!version_fields);
        char *version_buffer = version_fields.to_string();
        logger->log("%lld: %s", vit->first, version_buffer);
        free(version_buffer);
      }
      logger->up();
      temp_state.print_physical_state(capture_mask, to_traverse, logger);
    }

    /////////////////////////////////////////////////////////////
    // FieldState 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldState::FieldState(void)
      : open_state(NOT_OPEN), redop(0), rebuild_timeout(1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const GenericUser &user, const FieldMask &m, 
                           const ColorPoint &c)
      : ChildState(m), redop(0), rebuild_timeout(1)
    //--------------------------------------------------------------------------
    {
      if (IS_READ_ONLY(user.usage))
        open_state = OPEN_READ_ONLY;
      else if (IS_WRITE(user.usage))
        open_state = OPEN_READ_WRITE;
      else if (IS_REDUCE(user.usage))
      {
        open_state = OPEN_SINGLE_REDUCE;
        redop = user.usage.redop;
      }
      open_children[c] = m;
    }

    //--------------------------------------------------------------------------
    bool FieldState::overlaps(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      if (redop != rhs.redop)
        return false;
      if (redop == 0)
        return (open_state == rhs.open_state);
      else
      {
#ifdef DEBUG_LEGION
        assert((open_state == OPEN_SINGLE_REDUCE) ||
               (open_state == OPEN_MULTI_REDUCE));
        assert((rhs.open_state == OPEN_SINGLE_REDUCE) ||
               (rhs.open_state == OPEN_MULTI_REDUCE));
#endif
        // Only support merging reduction fields with exactly the
        // same mask which should be single fields for reductions
        return (valid_fields == rhs.valid_fields);
      }
    }

    //--------------------------------------------------------------------------
    void FieldState::merge(const FieldState &rhs, RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      valid_fields |= rhs.valid_fields;
      for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
            rhs.open_children.begin(); it != rhs.open_children.end(); it++)
      {
        LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
                                      open_children.find(it->first);
        if (finder == open_children.end())
          open_children[it->first] = it->second;
        else
          finder->second |= it->second;
      }
#ifdef DEBUG_LEGION
      assert(redop == rhs.redop);
#endif
      if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(!open_children.empty());
#endif
        // For the reductions, handle the case where we need to merge
        // reduction modes, if they are all disjoint, we don't need
        // to distinguish between single and multi reduce
        if (node->are_all_children_disjoint())
        {
          open_state = OPEN_READ_WRITE;
          redop = 0;
        }
        else
        {
          if (open_children.size() == 1)
            open_state = OPEN_SINGLE_REDUCE;
          else
            open_state = OPEN_MULTI_REDUCE;
        }
      }
    }

    //--------------------------------------------------------------------------
    void FieldState::print_state(TreeStateLogger *logger,
                                 const FieldMask &capture_mask) const
    //--------------------------------------------------------------------------
    {
      switch (open_state)
      {
        case NOT_OPEN:
          {
            logger->log("Field State: NOT OPEN (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_WRITE:
          {
            logger->log("Field State: OPEN READ WRITE (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_ONLY:
          {
            logger->log("Field State: OPEN READ-ONLY (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_SINGLE_REDUCE:
          {
            logger->log("Field State: OPEN SINGLE REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
            break;
          }
        case OPEN_MULTI_REDUCE:
          {
            logger->log("Field State: OPEN MULTI REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        char *mask_buffer = overlap.to_string();
        switch (it->first.get_dim())
        {
          case 0:
            {
              logger->log("Color %d   Mask %s", 
                          it->first.get_index(), mask_buffer);
              break;
            }
          case 1:
            {
              logger->log("Color %d   Mask %s", 
                          it->first[0], mask_buffer);
              break;
            }
          case 2:
            {
              logger->log("Color (%d,%d)   Mask %s", 
                          it->first[0], it->first[1],
                          mask_buffer);
              break;
            }
          case 3:
            {
              logger->log("Color %d   Mask %s", 
                          it->first[0], it->first[1],
                          it->first[2], mask_buffer);
              break;
            }
          default:
            assert(false); // implemenent more dimensions
        }
        free(mask_buffer);
      }
      logger->up();
    }

    /////////////////////////////////////////////////////////////
    // Closers 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(ContextID c, const LogicalUser &u, 
                                 bool val, bool capture)
      : ctx(c), user(u), validates(val), capture_users(capture)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(const LogicalCloser &rhs)
      : user(rhs.user), validates(rhs.validates),
        capture_users(rhs.capture_users)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalCloser::~LogicalCloser(void)
    //--------------------------------------------------------------------------
    {
      // Clear out our version infos
      closed_version_info.release();
      closed_version_info.clear();
    }

    //--------------------------------------------------------------------------
    LogicalCloser& LogicalCloser::operator=(const LogicalCloser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_closed_child(const ColorPoint &child, 
                                            const FieldMask &mask,
                                            bool leave_open, bool read_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // should never be both, but can be neither
      assert(!(leave_open && read_only));
#endif
      // IMPORTANT: Always do this even if we don't have any closed users
      // They could have been pruned out because they finished executing, but
      // we still need to do the close operation.
      if (read_only)
      {
        LegionMap<ColorPoint,ClosingInfo>::aligned::iterator finder = 
                                              read_only_children.find(child);
        if (finder != read_only_children.end())
        {
          finder->second.child_fields |= mask;
          finder->second.child_users.insert(finder->second.child_users.end(),
                                      closed_users.begin(), closed_users.end());
        }
        else
          read_only_children[child] = ClosingInfo(mask, closed_users);
      }
      else
      {
        // Only actual closes get to count to the closed mask
        closed_mask |= mask;
        LegionMap<ColorPoint,ClosingInfo>::aligned::iterator finder = 
                                              closed_children.find(child);
        if (finder != closed_children.end())
        {
          finder->second.child_fields |= mask;
          finder->second.child_users.insert(finder->second.child_users.end(),
                                      closed_users.begin(), closed_users.end());
        }
        else
          closed_children[child] = ClosingInfo(mask, closed_users);
      }
      // Always clean out our list of closed users
      closed_users.clear();
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_partial_fields(const FieldMask &skipped_fields)
    //--------------------------------------------------------------------------
    {
      partial_mask |= skipped_fields;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_flush_only_fields(const FieldMask &flush_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!flush_only_fields);
#endif
      flush_only_fields = flush_only;
      closed_mask |= flush_only;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::initialize_close_operations(RegionTreeNode *target, 
                                                   Operation *creator,
                                                   const VersionInfo &ver_info,
                                                   const RestrictInfo &res_info,
                                                   const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      // First sort the close operations into sets of fields which all
      // close the same sets of children
      if (!closed_children.empty())
      {
        LegionList<ClosingSet>::aligned closes;
        compute_close_sets(closed_children, closes);
        create_normal_close_operations(target, creator, closed_version_info, 
                                       ver_info, res_info, trace_info, closes);
      }
      if (!read_only_children.empty())
      {
        LegionList<ClosingSet>::aligned read_only;
        compute_close_sets(read_only_children, read_only);
        create_read_only_close_operations(target, creator, 
                                          trace_info, read_only);
      }
      // Finally if we have any fields which are flush only
      // make a close operation for them and add it to force close
      if (!!flush_only_fields)
      {
        LegionMap<ColorPoint,FieldMask>::aligned empty_children;
        InterCloseOp *flush_op = target->create_close_op(creator,
                                                         flush_only_fields,
                                                         empty_children,
                                                         closed_version_info,
                                                         ver_info, res_info,
                                                         trace_info);
        normal_closes[flush_op] = LogicalUser(flush_op, 0/*idx*/,
                              RegionUsage(flush_op->get_region_requirement()),
                              flush_only_fields);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::add_next_child(const ColorPoint &next_child)
    //--------------------------------------------------------------------------
    {
      // Only need to add children to leave open closes
      for (LegionMap<TraceCloseOp*,LogicalUser>::aligned::const_iterator it = 
            normal_closes.begin(); it != normal_closes.end(); it++)
      {
        it->first->add_next_child(next_child);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalCloser::compute_close_sets(
                    const LegionMap<ColorPoint,ClosingInfo>::aligned &children,
                    LegionList<ClosingSet>::aligned &close_sets)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<ColorPoint,ClosingInfo>::aligned::const_iterator 
             cit = children.begin(); cit != children.end(); cit++)
      {
        bool inserted = false;
        FieldMask remaining = cit->second.child_fields;
        FieldMask remaining_leave_open = cit->second.leave_open_mask;
        for (LegionList<ClosingSet>::aligned::iterator it = 
              close_sets.begin(); it != close_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->closing_mask)
          {
            // Add the child
            it->add_child(cit->first, remaining_leave_open);
            inserted = true;
            break;
          }
          FieldMask overlap = remaining & it->closing_mask;
          // If they are disjoint keep going
          if (!overlap)
            continue;
          // We are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == remaining)
          {
            // Leave the existing set and make it the difference
            it->closing_mask -= overlap;
            close_sets.push_back(ClosingSet(overlap));
            ClosingSet &last = close_sets.back();
            last.children = it->children;
            // Insert the new child
            last.add_child(cit->first, remaining_leave_open);
            inserted = true;
            break;
          }
          // We dominate the existing set
          if (overlap == it->closing_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->add_child(cit->first, remaining_leave_open & overlap);
            remaining -= overlap;
            remaining_leave_open -= overlap;
            continue;
          }
          // Hard case, neither dominates, compute
          // three distinct sets of fields, keep left
          // one in place and reduce scope, add a new
          // one at the end for overlap, continue
          // iterating for the right one
          it->closing_mask -= overlap;
          const LegionMap<ColorPoint,FieldMask>::aligned &temp_children = 
                                                            it->children;
          it = close_sets.insert(it, ClosingSet(overlap));
          it->children = temp_children;
          it->add_child(cit->first, remaining_leave_open & overlap);
          remaining -= overlap;
          remaining_leave_open -= overlap;
          continue;
        }
        // If we didn't add it yet, add it now
        if (!inserted)
        {
          close_sets.push_back(ClosingSet(remaining));
          ClosingSet &last = close_sets.back();
          last.add_child(cit->first, remaining_leave_open);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::create_normal_close_operations(RegionTreeNode *target,
                              Operation *creator, const VersionInfo &local_info,
                              const VersionInfo &version_info,
                              const RestrictInfo &restrict_info, 
                              const TraceInfo &trace_info,
                              LegionList<ClosingSet>::aligned &close_sets)
    //--------------------------------------------------------------------------
    {
      for (LegionList<ClosingSet>::aligned::iterator it = 
            close_sets.begin(); it != close_sets.end(); it++)
      {
        // Filter the leave open fields before passing them
        it->filter_children();
        InterCloseOp *close_op = target->create_close_op(creator, 
                                                       it->closing_mask,
                                                       it->children,
                                                       local_info,
                                                       version_info,
                                                       restrict_info, 
                                                       trace_info);
        normal_closes[close_op] = LogicalUser(close_op, 0/*idx*/,
                      RegionUsage(close_op->get_region_requirement()),
                      it->closing_mask);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::create_read_only_close_operations(
                              RegionTreeNode *target, Operation *creator,
                              const TraceInfo &trace_info,
                              const LegionList<ClosingSet>::aligned &close_sets)
    //--------------------------------------------------------------------------
    {
      for (LegionList<ClosingSet>::aligned::const_iterator it = 
            close_sets.begin(); it != close_sets.end(); it++)
      {
        // No need to filter here since we won't ever use those fields
        ReadCloseOp *close_op = target->create_read_only_close_op(creator,
                                                              it->closing_mask,
                                                              it->children,
                                                              trace_info);
        read_only_closes[close_op] = LogicalUser(close_op, 0/*idx*/,
            RegionUsage(close_op->get_region_requirement()), it->closing_mask);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::perform_dependence_analysis(const LogicalUser &current,
                                                    const FieldMask &open_below,
              LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
              LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers)
    //--------------------------------------------------------------------------
    {
      // We also need to do dependence analysis against all the other operations
      // that this operation recorded dependences on above in the tree so we
      // don't run too early.
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned &above_users = 
                                              current.op->get_logical_records();
      if (!normal_closes.empty())
        register_dependences(current, open_below, normal_closes, 
                             closed_children, above_users, cusers, pusers);
      if (!read_only_closes.empty())
        register_dependences(current, open_below, read_only_closes,
                             read_only_children, above_users, cusers, pusers);
    }

    // If you are looking for LogicalCloser::register_dependences it can 
    // be found in region_tree.cc to make sure that templates are instantiated

    //--------------------------------------------------------------------------
    void LogicalCloser::update_state(CurrentState &state)
    //--------------------------------------------------------------------------
    {
      // If we only have read-only closes then we are done
      if (!closed_mask)
        return;
      RegionTreeNode *node = state.owner;
      // Our partial mask is initially an over approximation of
      // the partially closed fields, so intersect it with the
      // fields that were actually closed
      if (!!partial_mask)
        partial_mask &= closed_mask;
      // See if we have any fields that were partially closed
      if (!!partial_mask)
      {
        // Record the partially closed fields
        state.partially_closed |= partial_mask;
        // See if we did have any fully closed fields
        FieldMask fully_closed = closed_mask - partial_mask;
        if (!!fully_closed)
        {
          state.dirty_below -= fully_closed;
          if (!!state.partially_closed)
            state.partially_closed -= fully_closed;
          // We can only filter fields fully closed
          node->filter_prev_epoch_users(state, fully_closed); 
          node->filter_curr_epoch_users(state, fully_closed);
        }
      }
      else
      {
        // All the fields were fully closed, so we can clear dirty below
        state.dirty_below -= closed_mask;
        if (!!state.partially_closed)
          state.partially_closed -= closed_mask;
        // Filter all the fields since they were all fully closed
        node->filter_prev_epoch_users(state, closed_mask);
        node->filter_curr_epoch_users(state, closed_mask);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::register_close_operations(
               LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users)
    //--------------------------------------------------------------------------
    {
      // Add our close operations onto the list
      // Note we already added our mapping references when we made them
      if (!normal_closes.empty())
      {
        for (LegionMap<TraceCloseOp*,LogicalUser>::aligned::const_iterator it =
              normal_closes.begin(); it != normal_closes.end(); it++)
        {
          users.push_back(it->second);
        }
      }
      if (!read_only_closes.empty())
      {
        for (LegionMap<TraceCloseOp*,LogicalUser>::aligned::const_iterator it =
              read_only_closes.begin(); it != read_only_closes.end(); it++)
        {
          users.push_back(it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_version_numbers(RegionTreeNode *node,
                                               CurrentState &state,
                                               const FieldMask &local_mask,
                                               bool leave_open)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime, 
                        LOGICAL_CLOSER_RECORD_VERSION_NUMBERS_CALL);
#ifdef DEBUG_LEGION
      assert(!!local_mask);
#endif
      // If we have dirty below then we do need previous
      if (!!state.dirty_below)
      {
        FieldMask split_fields = local_mask & state.dirty_below;
        if (!!split_fields)
        {
          state.record_version_numbers(split_fields, user, closed_version_info,
                                       true/*previous*/, false/*path only*/,
                                       true/*final*/, false/*close top*/,
                                       false/*report*/, true/*close*/, 
                                       leave_open, true/*split node*/);
          FieldMask non_split = local_mask - split_fields;
          if (!!non_split)
            state.record_version_numbers(non_split, user, closed_version_info,
                                         false/*previous*/, false/*path only*/,
                                         false/*final*/, false/*close top*/,
                                         false/*report*/, true/*close node*/,
                                         leave_open, false/*split node*/);
          return;
        }
        // otherwise we fall through 
      }
      // Don't need the previous because writes were already done in the
      // sub-tree we are closing so the version number for the target
      // region has already been advanced.
      state.record_version_numbers(local_mask, user, closed_version_info, 
                                   false/*previous*/, false/*path only*/, 
                                   false/*final*/, false/*close top*/, 
                                   false/*report*/, true/*close node*/,
                                   leave_open, false/*split node*/);
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_top_version_numbers(RegionTreeNode *node,
                                                   CurrentState &state)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime, 
                        LOGICAL_CLOSER_RECORD_TOP_VERSION_NUMBERS_CALL);
      // If we have any flush only fields, see if we need to bump their
      // version numbers before generating our close operations
      if (!!flush_only_fields)
      {
        FieldMask update_mask = flush_only_fields - state.dirty_below;
        if (!!update_mask)
          state.advance_version_numbers(update_mask);
      }
      // See if there are any partially closed fields
      FieldMask partial_close = closed_mask & state.partially_closed;
      if (!partial_close)
      {
        // Common case: there are no partially closed fields
        state.record_version_numbers(closed_mask, user, closed_version_info,
                                     true/*previous*/, false/*path only*/, 
                                     true/*final*/, true/*close top*/, 
                                     false/*report*/, true/*close node*/,
                                     false/*leave open*/, true/*split*/);
      }
      else
      {
        // Record the partially closed fields from this version
        state.record_version_numbers(partial_close, user, closed_version_info,
                                     false/*previous*/, false/*path only*/, 
                                     true/*final*/, true/*close top*/, 
                                     false/*report*/, true/*close node*/,
                                     false/*leave open*/, false/*split*/);
        FieldMask non_partial = closed_mask - partial_close;
        if (!!non_partial)
          state.record_version_numbers(non_partial, user, closed_version_info,
                                       true/*previous*/, false/*path only*/, 
                                       true/*final*/, true/*close top*/, 
                                       false/*report*/, true/*close node*/,
                                       false/*leave open*/, true/*split*/);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::merge_version_info(VersionInfo &target,
                                           const FieldMask &merge_mask)
    //--------------------------------------------------------------------------
    {
      target.merge(closed_version_info, merge_mask);
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const TraversalInfo &in, LogicalRegion h)
      : info(in), handle(h)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalCloser &rhs)
      : info(rhs.info), handle(rhs.handle), 
        leave_open_mask(rhs.leave_open_mask),
        upper_targets(rhs.lower_targets),
        close_targets(rhs.close_targets)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::~PhysicalCloser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalCloser& PhysicalCloser::operator=(const PhysicalCloser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::initialize_targets(RegionTreeNode *origin,
                                  PhysicalState *state,
                                  const std::vector<MaterializedView*> &targets,
                                  const FieldMask &closing_mask,
                                  const InstanceSet &target_set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(upper_targets.empty());
#endif
      // Figure out which instances need updates before we can close to them
      LegionMap<LogicalView*,FieldMask>::aligned valid_views;
      origin->find_valid_instance_views(info.ctx, state, closing_mask,
                                        closing_mask, info.version_info,
                                        false/*needs space*/, valid_views);
      // Now figure out which fields need updating for each instance
      for (std::vector<MaterializedView*>::const_iterator it = 
            targets.begin(); it != targets.end(); it++)
      {
        // The set of fields we must update
        FieldMask space_mask = (*it)->get_space_mask() & closing_mask;
        // If we don't have any incomplete fields, keep going
        if (!space_mask)
          continue;
        LegionMap<LogicalView*,FieldMask>::aligned::const_iterator finder = 
          valid_views.find(*it);
        if (finder != valid_views.end())
        {
          // We can skip fields for which we are already valid
          FieldMask invalid_mask = space_mask - finder->second;
          if (!!invalid_mask)
            origin->issue_update_copies(info, *it, invalid_mask, valid_views);
        }
        else // update all the incomplete fields we have
          origin->issue_update_copies(info, *it, space_mask, valid_views);
      }
      // Then we can record the targets
      upper_targets = targets;
      close_targets = target_set;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::close_tree_node(RegionTreeNode *node,
                                         const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      // Lower the targets
      lower_targets.resize(upper_targets.size());
      for (unsigned idx = 0; idx < upper_targets.size(); idx++)
        lower_targets[idx] = 
          upper_targets[idx]->get_materialized_subview(node->get_color());

      // Close the node
      node->close_physical_node(*this, closing_mask);

      // Clear out the lowered targets
      lower_targets.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::issue_dirty_updates(RegionTreeNode *node,
                                             const FieldMask &dirty_fields,
              const LegionMap<LogicalView*,FieldMask>::aligned &valid_instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lower_targets.size() == close_targets.size());
#endif
      // Iterate through all our instances and issue updates where necessary
      for (unsigned idx = 0; idx < lower_targets.size(); idx++)
      {
        // Figure out which of the dirty fields we have to issue udpates for
        FieldMask needed_fields = 
          close_targets[idx].get_valid_fields() & dirty_fields;
        // If we don't have any dirty fields, keep going
        if (!needed_fields)
          continue;
        MaterializedView *target = lower_targets[idx];
        // See if any of these fields are already valid
        LegionMap<LogicalView*,FieldMask>::aligned::const_iterator finder = 
          valid_instances.find(target);
        if (finder != valid_instances.end())
        {
          needed_fields -= finder->second;
          // If we're already valid, we're good to go
          if (!needed_fields)
            continue;
        }
        // Now we need to issue update copies for the valid fields
        node->issue_update_copies(info, target, needed_fields, valid_instances);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::issue_reduction_updates(RegionTreeNode *node,
                                                 const FieldMask &reduc_fields,
           const LegionMap<ReductionView*,FieldMask>::aligned &valid_reductions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lower_targets.size() == close_targets.size());
#endif
      for (unsigned idx = 0; idx < lower_targets.size(); idx++)
      {
        FieldMask needed_fields = 
          close_targets[idx].get_valid_fields() & reduc_fields;
        if (!needed_fields)
          continue;
        node->issue_update_reductions(lower_targets[idx], needed_fields,
                           info.version_info, valid_reductions, info.op, 
                           info.index, info.map_applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::update_dirty_mask(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      dirty_mask |= mask;
    }

    //--------------------------------------------------------------------------
    const FieldMask& PhysicalCloser::get_dirty_mask(void) const
    //--------------------------------------------------------------------------
    {
      return dirty_mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::update_node_views(RegionTreeNode *node,
                                           PhysicalState *state)
    //--------------------------------------------------------------------------
    {
      // Note that permit leave open means that we don't update
      // the dirty bits when we update the state
      node->update_valid_views(state, dirty_mask, upper_targets, close_targets);
    } 

    //--------------------------------------------------------------------------
    CompositeCloser::CompositeCloser(ContextID c, 
                                     VersionInfo &info, SingleTask *target)
      : ctx(c), version_info(info), target_ctx(target)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeCloser::CompositeCloser(const CompositeCloser &rhs)
      : ctx(0), version_info(rhs.version_info), target_ctx(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeCloser::~CompositeCloser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeCloser& CompositeCloser::operator=(const CompositeCloser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    CompositeNode* CompositeCloser::get_composite_node(RegionTreeNode *node,
                                                       bool root /*= false*/)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeNode*,CompositeNode*>::const_iterator finder = 
        constructed_nodes.find(node);
      if (finder != constructed_nodes.end())
        return finder->second;
      if (!root)
      {
        // Recurse up the tree until we find the parent
        CompositeNode *parent = get_composite_node(node->get_parent(), false);
        CompositeNode *result = legion_new<CompositeNode>(node, parent);
        constructed_nodes[node] = result;
        return result;
      }
      // Root case
      CompositeNode *result = legion_new<CompositeNode>(node, 
                                                        (CompositeNode*)NULL);
      constructed_nodes[node] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    CompositeView*CompositeCloser::create_valid_view(PhysicalState *state,
                                                    CompositeNode *root,
                                                    const FieldMask &valid_mask)
    //--------------------------------------------------------------------------
    {
      // Finalize the root before we use it
      FieldMask finalize_mask;
      root->finalize(finalize_mask);
      RegionTreeNode *node = root->logical_node;
      DistributedID did = 
                    node->context->runtime->get_available_distributed_id(false);
      CompositeVersionInfo *composite_info = new CompositeVersionInfo();
      //composite_info->get_version_info().clone_from(version_info);
      CompositeView *composite_view = legion_new<CompositeView>(node->context, 
                                   did, node->context->runtime->address_space,
                                   node, node->context->runtime->address_space, 
                                   root, composite_info, 
                                   RtUserEvent::NO_RT_USER_EVENT, 
                                   true/*register now*/);
      // Now update the state of the node
      // Note that if we are permitted to leave the subregions
      // open then we don't make the view dirty
      node->update_valid_views(state, valid_mask,
                               true /*dirty*/, composite_view);
      return composite_view;
    }

    //--------------------------------------------------------------------------
    void CompositeCloser::capture_physical_state(CompositeNode *target,
                                                 RegionTreeNode *node,
                                                 PhysicalState *state,
                                                 const FieldMask &close_mask,
                                                 const FieldMask &dirty_mask,
                                                 const FieldMask &reduc_mask)
    //--------------------------------------------------------------------------
    {
      // Do the capture and then update capture mask
      target->capture_physical_state(*this, state,
                                     close_mask, dirty_mask, reduc_mask);
      // Record that we've captured the fields for this node
      // Important that we only do this after the capture
      update_capture_mask(node, close_mask);
    }

    //--------------------------------------------------------------------------
    void CompositeCloser::update_capture_mask(RegionTreeNode *node,
                                              const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<RegionTreeNode*,FieldMask>::aligned::iterator finder = 
                                                  capture_fields.find(node);
      if (finder == capture_fields.end())
        capture_fields[node] = capture_mask;
      else
        finder->second |= capture_mask;
    }

    //--------------------------------------------------------------------------
    bool CompositeCloser::filter_capture_mask(RegionTreeNode *node,
                                              FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<RegionTreeNode*,FieldMask>::aligned::const_iterator finder = 
        capture_fields.find(node);
      if (finder != capture_fields.end() && !(capture_mask * finder->second))
      {
        capture_mask -= finder->second;
        return true;
      }
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Physical State 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(RegionTreeNode *n)
      : node(n)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(const PhysicalState &rhs)
      : node(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalState::~PhysicalState(void)
    //--------------------------------------------------------------------------
    {
      // Remove references to our version states and delete them if necessary
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator vit =
            version_states.begin(); vit != version_states.end(); vit++)
      {
        const VersionStateInfo &info = vit->second;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              info.states.begin(); it != info.states.end(); it++)
        {
          if (it->first->remove_base_valid_ref(PHYSICAL_STATE_REF)) 
            legion_delete(it->first);
        }
      }
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator vit =
            advance_states.begin(); vit != advance_states.end(); vit++)
      {
        const VersionStateInfo &info = vit->second;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              info.states.begin(); it != info.states.end(); it++)
        {
          if (it->first->remove_base_valid_ref(PHYSICAL_STATE_REF))
            legion_delete(it->first);
        }
      }
      version_states.clear();
      advance_states.clear();
    }

    //--------------------------------------------------------------------------
    PhysicalState& PhysicalState::operator=(const PhysicalState &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }
    
    //--------------------------------------------------------------------------
    void* PhysicalState::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<PhysicalState,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void* PhysicalState::operator new[](size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<PhysicalState,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void PhysicalState::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void PhysicalState::operator delete[](void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void PhysicalState::add_version_state(VersionState *state, 
                                          const FieldMask &state_mask)
    //--------------------------------------------------------------------------
    {
      VersionStateInfo &info = version_states[state->version_number];
      LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
        info.states.find(state);
      if (finder == info.states.end())
      {
        state->add_base_valid_ref(PHYSICAL_STATE_REF);
        info.states[state] = state_mask;
      }
      else
        finder->second |= state_mask;
      info.valid_fields |= state_mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalState::add_advance_state(VersionState *state, 
                                          const FieldMask &state_mask)
    //--------------------------------------------------------------------------
    {
      VersionStateInfo &info = advance_states[state->version_number];
      LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
        info.states.find(state);
      if (finder == info.states.end())
      {
        state->add_base_valid_ref(PHYSICAL_STATE_REF);
        info.states[state] = state_mask;
      }
      else
        finder->second |= state_mask;
      info.valid_fields |= state_mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalState::capture_state(bool path_only, bool split_node)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        PHYSICAL_STATE_CAPTURE_STATE_CALL);
      // Path only first since path only can also be a split
      if (path_only)
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = version_states.begin(); vit != version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            it->first->update_path_only_state(this, it->second);
          }
        }
      }
      else if (split_node)
      {
        // Capture everything but the open children below from the
        // normal version states, but get the open children from the
        // advance states since that's where the sub-operations have
        // been writing to.
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = version_states.begin(); vit != version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            it->first->update_split_previous_state(this, it->second);
          }
        }
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator
              vit = advance_states.begin(); vit != advance_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            it->first->update_split_advance_state(this, it->second);
          }
        }
      }
      else
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = version_states.begin(); vit != version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            it->first->update_physical_state(this, it->second);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::apply_path_only_state(const FieldMask &adv_mask,
             AddressSpaceID target, std::set<RtEvent> &applied_conditions) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        PHYSICAL_STATE_APPLY_PATH_ONLY_CALL);
      if (!advance_states.empty())
      {
        FieldMask non_advance_mask = adv_mask;
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = advance_states.begin(); vit != 
              advance_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->merge_path_only_state(this, it->second, 
                                             target, applied_conditions);
          }
          non_advance_mask -= info.valid_fields;
        }
        if (!!non_advance_mask)
        {
          // If we had non-advanced fields, issue updates to those states
          for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator
                vit = version_states.begin(); vit != 
                version_states.end(); vit++)
          {
            const VersionStateInfo &info = vit->second;
            if (info.valid_fields * non_advance_mask)
              continue;
            for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                  it = info.states.begin(); it != info.states.end(); it++)
            {
              FieldMask overlap = it->second & non_advance_mask;
              if (!overlap)
                continue;
              it->first->merge_path_only_state(this, overlap,
                                               target, applied_conditions);
            }
          }
        }
      }
      else
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = version_states.begin(); vit != 
              version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->merge_path_only_state(this, it->second, 
                                             target, applied_conditions); 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::apply_state(const FieldMask &advance_mask,
                   AddressSpaceID target, std::set<RtEvent> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        PHYSICAL_STATE_APPLY_STATE_CALL);
      if (!advance_states.empty())
      {
#ifdef DEBUG_LEGION
        assert(!!advance_mask);
#endif
        FieldMask non_advance_mask = advance_mask;
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = advance_states.begin(); vit != 
              advance_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->merge_physical_state(this, it->second, 
                                            target, applied_conditions);
          }
          non_advance_mask -= info.valid_fields;
        }
        if (!!non_advance_mask)
        {
          for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
                vit = version_states.begin(); vit != 
                version_states.end(); vit++)
          {
            const VersionStateInfo &info = vit->second;
            if (info.valid_fields * non_advance_mask)
              continue;
            for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                  it = info.states.begin(); it != info.states.end(); it++)
            {
              FieldMask overlap = it->second & non_advance_mask;
              if (!overlap)
                continue;
              it->first->merge_physical_state(this, overlap,
                                              target, applied_conditions); 
            }
          } 
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!advance_mask);
#endif
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = version_states.begin(); vit != 
              version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->merge_physical_state(this, it->second,
                                            target, applied_conditions); 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::filter_and_apply(const FieldMask &advance_mask, 
                               AddressSpaceID target, bool filter_masks, 
                               bool filter_views, bool filter_children, 
               const LegionMap<ColorPoint,FieldMask>::aligned *closed_children,
                               std::set<RtEvent> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        PHYSICAL_STATE_FILTER_AND_APPLY_STATE_CALL);
      if (!advance_states.empty())
      {
#ifdef DEBUG_LEGION
        assert(!!advance_mask);
#endif
        FieldMask non_advance_mask = advance_mask;
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = advance_states.begin(); vit != 
              advance_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->filter_and_merge_physical_state(this, it->second, 
                target, filter_masks, filter_views, filter_children,
                closed_children, applied_conditions);
          }
          non_advance_mask -= info.valid_fields;
        }
        if (!!non_advance_mask)
        {
          for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
                vit = version_states.begin(); vit != 
                version_states.end(); vit++)
          {
            const VersionStateInfo &info = vit->second;
            if (info.valid_fields * non_advance_mask)
              continue;
            for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                  it = info.states.begin(); it != info.states.end(); it++)
            {
              FieldMask overlap = it->second & non_advance_mask;
              if (!overlap)
                continue;
              // Never need to filter previous states
              it->first->merge_physical_state(this, overlap,
                                              target, applied_conditions); 
            }
          } 
        }   
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!advance_mask);
#endif
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = version_states.begin(); vit != 
              version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            // Never need to filter previous states
            it->first->merge_physical_state(this, it->second,
                                            target, applied_conditions); 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::reset(void)
    //--------------------------------------------------------------------------
    {
      dirty_mask.clear();
      reduction_mask.clear();
      children.valid_fields.clear();
      children.open_children.clear();
      valid_views.clear();
      reduction_views.clear();
      // Don't clear version states or advance states, we need those
    }

    //--------------------------------------------------------------------------
    void PhysicalState::filter_open_children(const FieldMask &filter_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<ColorPoint> to_delete; 
      for (LegionMap<ColorPoint,FieldMask>::aligned::iterator 
            it = children.open_children.begin(); 
            it != children.open_children.end(); it++)
      {
        it->second -= filter_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      if (!to_delete.empty())
      {
        if (to_delete.size() != children.open_children.size())
        {
          for (std::vector<ColorPoint>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            children.open_children.erase(*it);  
          }
        }
        else
          children.open_children.clear();
      }
      children.valid_fields -= filter_mask;
    }

    //--------------------------------------------------------------------------
    PhysicalState* PhysicalState::clone(bool capture_state, bool need_adv) const
    //--------------------------------------------------------------------------
    {
      PhysicalState *result = legion_new<PhysicalState>(node);
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator it1 =
            version_states.begin(); it1 != version_states.end(); it1++)
      {
        const VersionStateInfo &info = it1->second;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              info.states.begin(); it != info.states.end(); it++)
        {
          result->add_version_state(it->first, it->second);
        }
      }
      if (need_adv && !advance_states.empty())
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              it1 = advance_states.begin(); it1 != advance_states.end(); it1++)
        {
          const VersionStateInfo &info = it1->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            result->add_advance_state(it->first, it->second);
          }
        }
      }
      if (capture_state)
      {
        // No need to copy over the close mask
        result->dirty_mask = dirty_mask;
        result->reduction_mask = reduction_mask;
        result->children = children;
        result->valid_views = valid_views;
        result->reduction_views = reduction_views;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    PhysicalState* PhysicalState::clone(const FieldMask &clone_mask, 
                                        bool capture_state, bool need_adv) const
    //--------------------------------------------------------------------------
    {
      PhysicalState *result = legion_new<PhysicalState>(node);
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator it1 =
            version_states.begin(); it1 != version_states.end(); it1++)
      {
        const VersionStateInfo &info = it1->second;
        if (it1->second.valid_fields * clone_mask)
          continue;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              info.states.begin(); it != info.states.end(); it++)
        {
          FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          result->add_version_state(it->first, it->second);
        }
      }
      if (need_adv && !advance_states.empty())
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              it1 = advance_states.begin(); it1 != advance_states.end(); it1++)
        {
          const VersionStateInfo &info = it1->second;
          if (it1->second.valid_fields * clone_mask)
            continue;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            FieldMask overlap = it->second & clone_mask;
            if (!overlap)
              continue;
            result->add_advance_state(it->first, it->second);
          }
        }
      }
      if (capture_state)
      {
        // No need to copy over the close mask
        result->dirty_mask = dirty_mask & clone_mask;
        result->reduction_mask = reduction_mask & clone_mask;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          result->valid_views[it->first] = overlap;
        }
        if (!!result->reduction_mask)
        {
          for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
                reduction_views.begin(); it != reduction_views.end(); it++)
          {
            FieldMask overlap = it->second & clone_mask;
            if (!overlap)
              continue;
            result->reduction_views[it->first] = overlap;
          }
        }
        if (!(children.valid_fields * clone_mask))
        {
          for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
                children.open_children.begin(); it !=
                children.open_children.end(); it++)
          {
            FieldMask overlap = it->second & clone_mask;
            if (!overlap)
              continue;
            result->children.open_children[it->first] = overlap;
            result->children.valid_fields |= overlap;
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PhysicalState::make_local(std::set<RtEvent> &preconditions, 
                                   bool needs_final, bool needs_advance)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        PHYSICAL_STATE_MAKE_LOCAL_CALL);
      if (needs_final)
      {
        // If we are either advancing or closing, then we need the final
        // version states for all the field versions
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
             vit = version_states.begin(); vit != version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->request_final_version_state(it->second, preconditions);
          }
        }
      }
      else
      {
        // Otherwise, we just need one instance of the initial version state
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
             vit = version_states.begin(); vit != version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->request_initial_version_state(it->second, preconditions);
          }
        }
      }
      if (needs_advance && !advance_states.empty())
      {
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
             vit = advance_states.begin(); vit != advance_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->request_initial_version_state(it->second, preconditions);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::print_physical_state(const FieldMask &capture_mask,
                          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                             TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      // Dirty Mask
      {
        FieldMask overlap = dirty_mask & capture_mask;
        char *dirty_buffer = overlap.to_string();
        logger->log("Dirty Mask: %s",dirty_buffer);
        free(dirty_buffer);
      }
      // Valid Views
      {
        unsigned num_valid = 0;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        logger->log("Valid Instances (%d)", num_valid);
        logger->down();
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          if (it->first->is_deferred_view())
            continue;
#ifdef DEBUG_LEGION
          assert(it->first->as_instance_view()->is_materialized_view());
#endif
          MaterializedView *current = 
            it->first->as_instance_view()->as_materialized_view();
          char *valid_mask = overlap.to_string();
          logger->log("Instance " IDFMT "   Memory " IDFMT "   Mask %s",
                      current->manager->get_instance().id, 
                      current->manager->get_memory().id, valid_mask);
          free(valid_mask);
        }
        logger->up();
      }
      // Valid Reduction Views
      {
        unsigned num_valid = 0;
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
              reduction_views.begin(); it != 
              reduction_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        logger->log("Valid Reduction Instances (%d)", num_valid);
        logger->down();
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
              reduction_views.begin(); it != 
              reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          char *valid_mask = overlap.to_string();
          logger->log("Reduction Instance " IDFMT "   Memory " IDFMT 
                      "  Mask %s",
                      it->first->manager->get_instance().id, 
                      it->first->manager->get_memory().id, valid_mask);
          free(valid_mask);
        }
        logger->up();
      }
      // Open Children
      {
        logger->log("Open Children (%ld)", 
            children.open_children.size());
        logger->down();
        for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator 
              it = children.open_children.begin(); it !=
              children.open_children.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          char *mask_buffer = overlap.to_string();
          switch (it->first.get_dim())
          {
            case 0:
              {
                logger->log("Color %d   Mask %s", 
                            it->first.get_index(), mask_buffer);
                break;
              }
            case 1:
              {

                logger->log("Color %d   Mask %s", 
                            it->first[0], mask_buffer);
                break;
              }
            case 2:
              {
                logger->log("Color (%d,%d)   Mask %s", 
                            it->first[0],
                            it->first[1], mask_buffer);
                break;
              }
            case 3:
              {
                logger->log("Color (%d,%d,%d)   Mask %s", 
                            it->first[0], it->first[1],
                            it->first[2], mask_buffer);
                break;
              }
            default:
              assert(false);
          }
          free(mask_buffer);
          // Mark that we should traverse this child
          to_traverse[it->first] = overlap;
        }
        logger->up();
      }
    }

    /////////////////////////////////////////////////////////////
    // Version State 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionState::VersionState(VersionID vid, Runtime *rt, DistributedID id,
                               AddressSpaceID own_sp, AddressSpaceID local_sp, 
                               RegionTreeNode *node, bool register_now)
      : DistributedCollectable(rt, id, own_sp, local_sp, 
          RtUserEvent::NO_RT_USER_EVENT, register_now), version_number(vid), 
        logical_node(node), state_lock(Reservation::create_reservation())
#ifdef DEBUG_LEGION
        , currently_active(true), currently_valid(true)
#endif
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, add a valid and resource reference
      if (!is_owner())
      {
        add_base_valid_ref(REMOTE_DID_REF);
        add_base_resource_ref(REMOTE_DID_REF);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Version State %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    VersionState::VersionState(const VersionState &rhs)
      : DistributedCollectable(rhs), version_number(rhs.version_number),
        logical_node(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VersionState::~VersionState(void)
    //--------------------------------------------------------------------------
    {
      state_lock.destroy_reservation();
      state_lock = Reservation::NO_RESERVATION;
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(!currently_valid);
#endif 
      // If we are the owner, then remote resource 
      // references on our remote instances 
      if (is_owner())
      {
        // If we're the owner, remove our valid references on remote nodes
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> 
          functor(this, NULL); 
        map_over_remote_instances(functor);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    VersionState& VersionState::operator=(const VersionState &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* VersionState::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<VersionState,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void* VersionState::operator new[](size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<VersionState,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void VersionState::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void VersionState::operator delete[](void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void VersionState::initialize(ApEvent term_event, const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  const InstanceSet &targets,
                                  SingleTask *context, unsigned init_index,
                                 const std::vector<LogicalView*> &corresponding)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(currently_valid);
#endif
      const UniqueID init_op_id = context->get_unique_id();
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        LogicalView *new_view = corresponding[idx];
        const FieldMask &view_mask = targets[idx].get_valid_fields();
        new_view->add_nested_valid_ref(did, context);
        if (new_view->is_instance_view())
        {
          InstanceView *inst_view = new_view->as_instance_view();
          if (inst_view->is_reduction_view())
          {
            ReductionView *view = inst_view->as_reduction_view();
            LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
              track_aligned::iterator finder = reduction_views.find(view); 
            if (finder == reduction_views.end())
              reduction_views[view] = view_mask;
            else
              finder->second |= view_mask;
            reduction_mask |= view_mask;
            inst_view->add_initial_user(term_event, usage, view_mask,
                                        init_op_id, init_index);
          }
          else
          {
            LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
              track_aligned::iterator finder = valid_views.find(new_view);
            if (finder == valid_views.end())
              valid_views[new_view] = view_mask;
            else
              finder->second |= view_mask;
            if (HAS_WRITE(usage))
              dirty_mask |= view_mask;
            inst_view->add_initial_user(term_event, usage, view_mask,
                                        init_op_id, init_index);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(!term_event.exists());
#endif
          LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
              track_aligned::iterator finder = valid_views.find(new_view);
          if (finder == valid_views.end())
            valid_views[new_view] = view_mask;
          else
            finder->second |= view_mask;
          if (HAS_WRITE(usage))
            dirty_mask |= view_mask;
          // Don't add a user since this is a deferred view and
          // we can't access it anyway
        }
      }
      // Update our field information, we know we are the owner
      initial_nodes[local_space] |= user_mask;
      initial_fields |= user_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::update_split_previous_state(PhysicalState *state,
                                             const FieldMask &update_mask) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_UPDATE_SPLIT_PREVIOUS_CALL);
      // We're reading so we only the need the lock in read-only mode
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      if (!!dirty_mask)
        state->dirty_mask |= (dirty_mask & update_mask);
      FieldMask reduction_update = reduction_mask & update_mask;
      if (!!reduction_update)
        state->reduction_mask |= reduction_update;
      for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
            track_aligned::const_iterator it = valid_views.begin();
            it != valid_views.end(); it++)
      {
        FieldMask overlap = it->second & update_mask;
        if (!overlap && !it->first->has_space(update_mask))
          continue;
        LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::track_aligned::
          iterator finder = state->valid_views.find(it->first);
        if (finder == state->valid_views.end())
          state->valid_views[it->first] = overlap;
        else
          finder->second |= overlap;
      }
      if (!!reduction_update)
      {
        for (LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
              track_aligned::const_iterator it = reduction_views.begin();
              it != reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & update_mask; 
          if (!overlap)
            continue;
          LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
            track_aligned::iterator finder = 
              state->reduction_views.find(it->first);
          if (finder == state->reduction_views.end())
            state->reduction_views[it->first] = overlap;
          else
            finder->second |= overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::update_split_advance_state(PhysicalState *state,
                                             const FieldMask &update_mask) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_UPDATE_SPLIT_ADVANCE_CALL);
      // We're reading so we only the need the lock in read-only mode
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      if (!(update_mask * children.valid_fields))
      {
        for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
              children.open_children.begin(); it != 
              children.open_children.end(); it++)
        {
          FieldMask overlap = update_mask & it->second;
          if (!overlap)
            continue;
          LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
            state->children.open_children.find(it->first);
          if (finder == state->children.open_children.end())
            state->children.open_children[it->first] = overlap;
          else
            finder->second |= overlap;
          state->children.valid_fields |= overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::update_path_only_state(PhysicalState *state,
                                             const FieldMask &update_mask) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_UPDATE_PATH_ONLY_CALL);
      // We're reading so we only the need the lock in read-only mode
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      // If we are premapping, we only need to update the dirty bits
      // and the valid instance views 
      if (!!dirty_mask)
        state->dirty_mask |= (dirty_mask & update_mask);
      for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
            track_aligned::const_iterator it = valid_views.begin();
            it != valid_views.end(); it++)
      {
        FieldMask overlap = it->second & update_mask;
        if (!overlap && !it->first->has_space(update_mask))
          continue;
        LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::track_aligned::
          iterator finder = state->valid_views.find(it->first);
        if (finder == state->valid_views.end())
          state->valid_views[it->first] = overlap;
        else
          finder->second |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::update_physical_state(PhysicalState *state,
                                             const FieldMask &update_mask) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_UPDATE_PATH_ONLY_CALL);
      // We're reading so we only the need the lock in read-only mode
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      if (!!dirty_mask)
        state->dirty_mask |= (dirty_mask & update_mask);
      FieldMask reduction_update = reduction_mask & update_mask;
      if (!!reduction_update)
        state->reduction_mask |= reduction_update;
      FieldMask child_overlap = children.valid_fields & update_mask;
      if (!!child_overlap)
      {
        state->children.valid_fields |= child_overlap;
        for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it =
              children.open_children.begin(); it != 
              children.open_children.end(); it++)
        {
          FieldMask overlap = it->second & update_mask;
          if (!overlap)
            continue;
          LegionMap<ColorPoint,FieldMask>::aligned::iterator finder =
            state->children.open_children.find(it->first);
          if (finder == state->children.open_children.end())
            state->children.open_children[it->first] = overlap;
          else
            finder->second |= overlap;
        }
      }
      for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
            track_aligned::const_iterator it = valid_views.begin();
            it != valid_views.end(); it++)
      {
        FieldMask overlap = it->second & update_mask;
        if (!overlap && !it->first->has_space(update_mask))
          continue;
        LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::track_aligned::
          iterator finder = state->valid_views.find(it->first);
        if (finder == state->valid_views.end())
          state->valid_views[it->first] = overlap;
        else
          finder->second |= overlap;
      }
      if (!!reduction_update)
      {
        for (LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
              track_aligned::const_iterator it = reduction_views.begin();
              it != reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & update_mask; 
          if (!overlap)
            continue;
          LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
            track_aligned::iterator finder = 
              state->reduction_views.find(it->first);
          if (finder == state->reduction_views.end())
            state->reduction_views[it->first] = overlap;
          else
            finder->second |= overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::merge_path_only_state(const PhysicalState *state,
                                             const FieldMask &merge_mask,
                                             AddressSpaceID target,
                                          std::set<RtEvent> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_MERGE_PATH_ONLY_CALL);
      // We're writing so we need the lock in exclusive mode
      AutoLock s_lock(state_lock);
      // For premapping, all we need to merge is the open children
      FieldMask child_overlap = state->children.valid_fields & merge_mask;
      if (!!child_overlap)
      {
        children.valid_fields |= child_overlap;
        for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
              state->children.open_children.begin(); it !=
              state->children.open_children.end(); it++)
        {
          FieldMask overlap = it->second & merge_mask;
          if (!overlap)
            continue;
          LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
            children.open_children.find(it->first);
          if (finder == children.open_children.end())
            children.open_children[it->first] = overlap;
          else
            finder->second |= overlap;
        }
      }
      if (is_owner())
      {
        path_only_nodes[local_space] |= merge_mask;
      }
      else
      {
        FieldMask new_path_only = merge_mask - path_only_fields;
        if (!!new_path_only)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(new_path_only);
            if (target != owner_space)
            {
              RtUserEvent registered_event = Runtime::create_rt_user_event();
              rez.serialize(registered_event);
              applied_conditions.insert(registered_event);
            }
            else
              rez.serialize(RtUserEvent::NO_RT_USER_EVENT);
          }
          runtime->send_version_state_path_only(owner_space, rez);
        }
      }
      path_only_fields |= merge_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::merge_physical_state(const PhysicalState *state,
                                            const FieldMask &merge_mask,
                                            AddressSpaceID target,
                                          std::set<RtEvent> &applied_conditions,
                                            bool need_lock /* = true*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_MERGE_PHYSICAL_STATE_CALL);
      WrapperReferenceMutator mutator(applied_conditions);
      if (need_lock)
      {
        // We're writing so we need the lock in exclusive mode
        RtEvent acquire_event = 
          Runtime::acquire_rt_reservation(state_lock, true/*exclusive*/);
        acquire_event.wait();
      }
      if (!!state->dirty_mask)
        dirty_mask |= (state->dirty_mask & merge_mask);
      FieldMask reduction_merge = state->reduction_mask & merge_mask;
      if (!!reduction_merge)
        reduction_mask |= reduction_merge;
      FieldMask child_overlap = state->children.valid_fields & merge_mask;
      if (!!child_overlap)
      {
        children.valid_fields |= child_overlap;
        for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
              state->children.open_children.begin(); it !=
              state->children.open_children.end(); it++)
        {
          FieldMask overlap = it->second & merge_mask;
          if (!overlap)
            continue;
          LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
            children.open_children.find(it->first);
          if (finder == children.open_children.end())
            children.open_children[it->first] = overlap;
          else
            finder->second |= overlap;
        }
      }
      for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
            track_aligned::const_iterator it = state->valid_views.begin();
            it != state->valid_views.end(); it++)
      {
        FieldMask overlap = it->second & merge_mask;
        if (!overlap)
          continue;
        LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
          track_aligned::iterator finder = valid_views.find(it->first);
        if (finder == valid_views.end())
        {
#ifdef DEBUG_LEGION
          assert(currently_valid);
#endif
          it->first->add_nested_valid_ref(did, &mutator);
          valid_views[it->first] = overlap;
        }
        else
          finder->second |= overlap;
      }
      if (!!reduction_merge)
      {
        for (LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
              track_aligned::const_iterator it = state->reduction_views.begin();
              it != state->reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & merge_mask;
          if (!overlap)
            continue;
          LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
            track_aligned::iterator finder = reduction_views.find(it->first);
          if (finder == reduction_views.end())
          {
#ifdef DEBUG_LEGION
            assert(currently_valid);
#endif
            it->first->add_nested_valid_ref(did, &mutator);
            reduction_views[it->first] = overlap;
          }
          else
            finder->second |= overlap;
        }
      }
      // Update our field information
      if (is_owner())
      {
        initial_nodes[local_space] |= merge_mask;
      }
      else
      {
        // We're remote see if we need to send any notifications
        FieldMask new_initial_fields = merge_mask - initial_fields;
        if (!!new_initial_fields)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(new_initial_fields);
            if (owner_space != target)
            {
              RtUserEvent registered_event = Runtime::create_rt_user_event();
              rez.serialize(registered_event);
              applied_conditions.insert(registered_event);
            }
            else
              rez.serialize(RtUserEvent::NO_RT_USER_EVENT);
          }
          runtime->send_version_state_initialization(owner_space, rez);
        }
      }
      initial_fields |= merge_mask;
      if (need_lock)
        state_lock.release();
    }

    //--------------------------------------------------------------------------
    void VersionState::filter_and_merge_physical_state(
                        const PhysicalState *state, const FieldMask &merge_mask,
                        AddressSpaceID target, bool filter_masks, 
                        bool filter_views, bool filter_children,
                const LegionMap<ColorPoint,FieldMask>::aligned *closed_children,
                        std::set<RtEvent> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_FILTER_AND_MERGE_PHYSICAL_STATE_CALL);
      WrapperReferenceMutator mutator(applied_conditions);
      // We're writing so we need the lock in exclusive mode
      AutoLock s_lock(state_lock);
#ifdef DEBUG_LEGION
      // If we are the top we should not be in final mode
      // Use filter_children as a proxy for the top of a close node
      // We'll see if this continues to be true in the future
      if (filter_children)
        assert(merge_mask * final_fields);
#endif
      // Do the filtering first
      if (filter_masks)
      {
        dirty_mask -= merge_mask;
        reduction_mask -= merge_mask;
      }
      if (filter_views)
      {
        // When filtering these views make sure that they aren't also
        // still in the state before removing references which is necessary
        // for the correctness of the garbage collection scheme
        if (!valid_views.empty())
        {
          std::vector<LogicalView*> to_delete;
          for (LegionMap<LogicalView*,FieldMask>::aligned::iterator it =
                valid_views.begin(); it != valid_views.end(); it++)
          {
            it->second -= merge_mask;
            if (!it->second && (state->valid_views.find(it->first) == 
                                state->valid_views.end()))
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<LogicalView*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              valid_views.erase(*it);
              if ((*it)->remove_nested_valid_ref(did, &mutator))
                legion_delete(*it);
            }
          }
        }
        if (!reduction_views.empty())
        {
          std::vector<ReductionView*> to_delete;
          for (LegionMap<ReductionView*,FieldMask>::aligned::iterator it = 
                reduction_views.begin(); it != reduction_views.end(); it++)
          {
            it->second -= merge_mask;
            if (!it->second && (state->reduction_views.find(it->first) ==
                                state->reduction_views.end()))
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<ReductionView*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              reduction_views.erase(*it);
              if ((*it)->remove_nested_valid_ref(did, &mutator))
                legion_delete(*it);
            }
          }
        }
      }
      if (filter_children && !children.open_children.empty())
      {
        // If we have specific children to filter then we
        // will only remove those children
        if (closed_children != NULL)
        {
          bool changed = false;
          for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
                closed_children->begin(); it != closed_children->end(); it++)
          {
            LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
              children.open_children.find(it->first);
            if (finder == children.open_children.end())
              continue;
            changed = true;
            finder->second -= merge_mask;
            if (!finder->second)
              children.open_children.erase(finder);
          }
          // See if we need to rebuild the open children mask
          if (changed)
          {
            if (!children.open_children.empty())
            {
              FieldMask new_open_child_mask;
              for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it =
                    children.open_children.begin(); it != 
                    children.open_children.end(); it++)
              {
                new_open_child_mask |= it->second;
              }
              children.valid_fields = new_open_child_mask;
            }
            else
              children.valid_fields.clear();
          }
        }
        else
        {
          // Otherwise filter all children for our merge mask
          children.valid_fields -= merge_mask;
          if (!!children.valid_fields)
          {
            std::vector<ColorPoint> to_delete;
            for (LegionMap<ColorPoint,FieldMask>::aligned::iterator it = 
                  children.open_children.begin(); it !=
                  children.open_children.end(); it++)
            {
              it->second -= merge_mask;
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<ColorPoint>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
                children.open_children.erase(*it);
            }
          }
          else
            children.open_children.clear();
        }
      }
      // Now we can do the merge
      merge_physical_state(state, merge_mask, 
                           target, applied_conditions, false/*need lock*/);
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Do nothing 
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Do nothing 
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(currently_valid); // should be monotonic
#endif
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_nested_valid_ref(did, mutator);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        it->first->add_nested_valid_ref(did, mutator);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      if (is_owner())
      {
        assert(currently_valid);
        currently_valid = false;
      }
#endif
      // When we are no longer valid, remove all valid references to version
      // state objects on remote nodes. 
      // No need to hold the lock since no one else should be accessing us
      if (is_owner() && !remote_instances.empty())
      {
        // If we're the owner, remove our valid references on remote nodes
        UpdateReferenceFunctor<VALID_REF_KIND,false/*add*/> functor(this, NULL);
        map_over_remote_instances(functor);
      }
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_nested_valid_ref(did, mutator))
          LogicalView::delete_logical_view(it->first);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        if (it->first->remove_nested_valid_ref(did, mutator))
          legion_delete(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::request_initial_version_state(
                const FieldMask &request_mask, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_REQUEST_INITIAL_CALL);
      RtUserEvent ready_event;
      FieldMask remaining_mask = request_mask;
      LegionDeque<RequestInfo>::aligned targets;
      {
        AutoLock s_lock(state_lock);
        // Check to see which fields we already have initial events for
        if (!(remaining_mask * initial_fields))
        {
          if (!initial_events.empty())
          {
            for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it =
                  initial_events.begin(); it != initial_events.end(); it++)
            {
              if (remaining_mask * it->second)
                continue;
              preconditions.insert(it->first);
            }
          }
          remaining_mask -= initial_fields;
          if (!remaining_mask)
            return;
        }
        // Also check to see which fields we already have final events for
        if (!(remaining_mask * final_fields))
        {
          if (!final_events.empty())
          {
            for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it =
                  final_events.begin(); it != final_events.end(); it++)
            {
              if (remaining_mask * it->second)
                continue;
              preconditions.insert(it->first);
            }
          }
          remaining_mask -= final_fields;
          if (!remaining_mask)
            return;
        }
        // If we are the owner and there are no existing versions, we can
        // immediately make ourselves a local_version
        if (is_owner())
        {
          if (initial_nodes.empty())
          {
            initial_fields |= remaining_mask;
            initial_nodes[local_space] = remaining_mask;
          }
          else
          {
            select_initial_targets(local_space, remaining_mask, 
                                   targets, preconditions);
            // At this point we can now upgrade our initial fields
            // and our node information
            initial_fields |= request_mask;
            initial_nodes[local_space] |= request_mask;
          }
        }
        else
        {
          // Make a user event and record it as a precondition
          ready_event = Runtime::create_rt_user_event();
          initial_events[ready_event] = remaining_mask;
          initial_fields |= remaining_mask;
          preconditions.insert(ready_event);
        }
      }
      if (!is_owner())
      {
#ifdef DEBUG_LEGION
        assert(!!remaining_mask);
#endif
        // If we make it here, then send a request to the owner
        // for the remaining fields
        send_version_state_request(owner_space, local_space, ready_event, 
                                   remaining_mask, INITIAL_VERSION_REQUEST);
      }
      else if (!targets.empty())
      {
        // otherwise we're the owner, send out requests to all 
        for (LegionDeque<RequestInfo>::aligned::const_iterator 
              it = targets.begin(); it != targets.end(); it++)
        {
          send_version_state_request(it->target, local_space, it->to_trigger,
                                     it->request_mask, it->kind);
                                
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::request_final_version_state(const FieldMask &req_mask,
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_REQUEST_FINAL_CALL);
      RtUserEvent ready_event;
      FieldMask remaining_mask = req_mask;
      LegionDeque<RequestInfo>::aligned targets;
      {
        AutoLock s_lock(state_lock);
        if (!(remaining_mask * final_fields))
        {
          if (!final_events.empty())
          {
            for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it =
                  final_events.begin(); it != final_events.end(); it++)
            {
              if (remaining_mask * it->second)
                continue;
              preconditions.insert(it->first);
            }
          }
          remaining_mask -= final_fields;
          if (!remaining_mask)
            return;
        }
        if (is_owner())
        {
          select_final_targets(local_space, remaining_mask,
                               targets, preconditions);
          // We can now update our final fields and local information
          final_fields |= req_mask;
          final_nodes[local_space] |= req_mask;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(!!remaining_mask);
#endif
          // Make a user event and record it as a precondition   
          ready_event = Runtime::create_rt_user_event();
          final_fields |= remaining_mask;
          final_events[ready_event] = remaining_mask;
          preconditions.insert(ready_event);
        }
      }
      if (!is_owner())
      {
        send_version_state_request(owner_space, local_space, ready_event,
                                   remaining_mask, FINAL_VERSION_REQUEST); 
      }
      else if (!targets.empty())
      {
        for (LegionDeque<RequestInfo>::aligned::const_iterator 
              it = targets.begin(); it != targets.end(); it++)
        {
          send_version_state_request(it->target, local_space, it->to_trigger,
                                     it->request_mask, it->kind);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::select_initial_targets(AddressSpaceID request_space,
                                              FieldMask &needed_mask,
                                     LegionDeque<RequestInfo>::aligned &targets,
                                              std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the lock
#ifdef DEBUG_LEGION
      assert(is_owner()); // should only be called on the owner node
#endif
      // Iterate over the initial nodes and issue requests to the first
      // node we happen upon who has the valid data
      for (LegionMap<AddressSpaceID,FieldMask>::aligned::const_iterator it = 
            initial_nodes.begin(); it != initial_nodes.end(); it++)
      {
        // Skip the requesting space
        if (request_space == it->first)
          continue;
        FieldMask overlap = needed_mask & it->second;
        if (!overlap)
          continue;
        targets.push_back(RequestInfo());
        RequestInfo &info = targets.back();
        info.target = it->first;
        info.to_trigger = Runtime::create_rt_user_event();
        info.request_mask = overlap;
        info.kind = INITIAL_VERSION_REQUEST;
        // Add the event to the set of preconditions
        preconditions.insert(info.to_trigger);
        // If we are the requester, then update our initial events
        if (request_space == local_space)
          initial_events[info.to_trigger] = overlap;
        needed_mask -= overlap;
        if (!needed_mask)
          return;
      }
      // If we still have needed fields, check to see if we have
      // any path only nodes that we need to request
      for (LegionMap<AddressSpaceID,FieldMask>::aligned::const_iterator it =
            path_only_nodes.begin(); it != path_only_nodes.end(); it++)
      {
        // Skip the requesting space
        if (request_space == it->first)
          continue;
        FieldMask overlap = needed_mask & it->second;
        if (!overlap)
          continue;
        targets.push_back(RequestInfo());
        RequestInfo &info = targets.back();
        info.target = it->first;
        info.to_trigger = Runtime::create_rt_user_event();
        info.request_mask = overlap;
        info.kind = PATH_ONLY_VERSION_REQUEST;
        // Add the event to the set of preconditions
        preconditions.insert(info.to_trigger);
        // If we are the requester, then update our initial events
        if (request_space == local_space)
          initial_events[info.to_trigger] = overlap;
        needed_mask -= overlap;
        if (!needed_mask)
          break;
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::select_final_targets(AddressSpaceID request_space,
                                            FieldMask &needed_mask,
                                    LegionDeque<RequestInfo>::aligned &targets,
                                            std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the lock
#ifdef DEBUG_LEGION
      assert(is_owner()); // should only be called on the owner node
#endif
      // First check to see if there are any final versions we can copy from
      for (LegionMap<AddressSpaceID,FieldMask>::aligned::const_iterator it = 
            final_nodes.begin(); it != final_nodes.end(); it++)
      {
        // Skip the requesting state
        if (request_space == it->first)
          continue;
        FieldMask overlap = needed_mask & it->second;
        if (!overlap)
          continue;
        targets.push_back(RequestInfo());
        RequestInfo &info = targets.back();
        info.target = it->first;
        info.to_trigger = Runtime::create_rt_user_event();
        info.request_mask = overlap;
        info.kind = FINAL_VERSION_REQUEST;
        // Add the event to the set of preconditions
        preconditions.insert(info.to_trigger);
        // If we are the requester, then update our final events
        if (request_space == local_space)
          final_events[info.to_trigger] = overlap;
        needed_mask -= overlap;
        if (!needed_mask)
          return;
      }
      // Now if we still have needed fields, we need to create a final
      // version from all of the earlier versions across all the nodes
      FieldMask requested_mask;
      std::set<RtEvent> merge_preconditions;
      for (LegionMap<AddressSpaceID,FieldMask>::aligned::const_iterator it = 
            initial_nodes.begin(); it != initial_nodes.end(); it++)
      {
        if (request_space == it->first)
          continue;
        FieldMask overlap = needed_mask & it->second;
        if (!overlap)
          continue;
        targets.push_back(RequestInfo());
        RequestInfo &info = targets.back();
        info.target = it->first;
        info.to_trigger = Runtime::create_rt_user_event();
        info.request_mask = overlap;
        info.kind = INITIAL_VERSION_REQUEST;
        merge_preconditions.insert(info.to_trigger); 
        requested_mask |= overlap;
      }
      needed_mask -= requested_mask;
      // Also check for any path only nodes for fields with no initial states
      if (!!needed_mask)
      {
        for (LegionMap<AddressSpaceID,FieldMask>::aligned::const_iterator it =
              path_only_nodes.begin(); it != path_only_nodes.end(); it++)
        {
          if (request_space == it->first)
            continue;
          FieldMask overlap = needed_mask & it->second;
          if (!overlap)
            continue;
          targets.push_back(RequestInfo());
          RequestInfo &info = targets.back();
          info.target = it->first;
          info.to_trigger = Runtime::create_rt_user_event();
          info.request_mask = overlap;
          info.kind = PATH_ONLY_VERSION_REQUEST;
          merge_preconditions.insert(info.to_trigger); 
          requested_mask |= overlap;
        }
        needed_mask -= requested_mask;
      }
      if (!!requested_mask)
      {
        RtEvent precondition = Runtime::merge_events(merge_preconditions);
        if (precondition.exists())
        {
          preconditions.insert(precondition);
          if (request_space == local_space)
            final_events[precondition] = requested_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state(AddressSpaceID target,
                                          VersionRequestKind request_kind,
                                          const FieldMask &request_mask,
                                          RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_SEND_STATE_CALL);
      Serializer rez;
      if (request_kind == PATH_ONLY_VERSION_REQUEST)
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(to_trigger);
        rez.serialize(request_kind);
        // Hold the lock in read-only mode while iterating these structures
        AutoLock s_lock(state_lock,1,false/*exclusive*/);
        if (!(children.valid_fields * request_mask))
        {
          Serializer child_rez;
          size_t count = 0;
          for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it = 
                children.open_children.begin(); it != 
                children.open_children.end(); it++)
          {
            FieldMask overlap = it->second & request_mask;
            if (!overlap)
              continue;
            child_rez.serialize(it->first);
            child_rez.serialize(overlap);
            count++;
          }
          rez.serialize<size_t>(count);
          rez.serialize(child_rez.get_buffer(), child_rez.get_used_bytes());
        }
        else
          rez.serialize<size_t>(0);
      }
      else // All other request go through the normal path
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(to_trigger);
        rez.serialize(request_kind);
        // Hold the lock in read-only mode while iterating these structures
        AutoLock s_lock(state_lock,1,false/*exclusive*/);
        // See if we should send all the fields or just do a partial send
        if (!((initial_fields | final_fields) - request_mask))
        {
          // Send everything
          rez.serialize(dirty_mask);
          rez.serialize(reduction_mask);
          rez.serialize<size_t>(children.open_children.size()); 
          for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it =
                children.open_children.begin(); it != 
                children.open_children.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(valid_views.size());
          for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
                track_aligned::const_iterator it = valid_views.begin(); it !=
                valid_views.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(reduction_views.size());
          for (LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
                track_aligned::const_iterator it = reduction_views.begin(); 
                it != reduction_views.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
        }
        else
        {
          // Partial send
          rez.serialize(dirty_mask & request_mask);
          rez.serialize(reduction_mask & request_mask);
          if (!children.open_children.empty())
          {
            Serializer child_rez;
            size_t count = 0;
            for (LegionMap<ColorPoint,FieldMask>::aligned::const_iterator it =
                  children.open_children.begin(); it !=
                  children.open_children.end(); it++)
            {
              FieldMask overlap = it->second & request_mask;
              if (!overlap)
                continue;
              child_rez.serialize(it->first);
              child_rez.serialize(overlap);
              count++;
            }
            rez.serialize(count);
            rez.serialize(child_rez.get_buffer(), child_rez.get_used_bytes());
          }
          else
            rez.serialize<size_t>(0);
          if (!valid_views.empty())
          {
            Serializer valid_rez;
            size_t count = 0;
            for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
                  track_aligned::const_iterator it = valid_views.begin(); it !=
                  valid_views.end(); it++)
            {
              FieldMask overlap = it->second & request_mask;
              if (!overlap)
                continue;
              valid_rez.serialize(it->first->did);
              valid_rez.serialize(overlap);
              count++;
            }
            rez.serialize(count);
            rez.serialize(valid_rez.get_buffer(), valid_rez.get_used_bytes());
          }
          else
            rez.serialize<size_t>(0);
          if (!reduction_views.empty())
          {
            Serializer reduc_rez;
            size_t count = 0;
            for (LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
                  track_aligned::const_iterator it = reduction_views.begin();
                  it != reduction_views.end(); it++)
            {
              FieldMask overlap = it->second & request_mask;
              if (!overlap)
                continue;
              reduc_rez.serialize(it->first->did);
              reduc_rez.serialize(overlap);
              count++;
            }
            rez.serialize(count);
            rez.serialize(reduc_rez.get_buffer(), reduc_rez.get_used_bytes());
          }
          else
            rez.serialize<size_t>(0);
        }
      }
      runtime->send_version_state_response(target, rez);
      // The owner will get updated automatically so only update remotely
      if (!is_owner())
        update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state_request(AddressSpaceID target,
                                    AddressSpaceID source, 
                                    RtUserEvent to_trigger,
                                    const FieldMask &request_mask, 
                                    VersionRequestKind request_kind) 
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(source);
        rez.serialize(to_trigger);
        rez.serialize(request_kind);
        rez.serialize(request_mask);
      }
      runtime->send_version_state_request(target, rez);
    }

    //--------------------------------------------------------------------------
    void VersionState::launch_send_version_state(AddressSpaceID target,
                                                 RtUserEvent to_trigger, 
                                                 VersionRequestKind req_kind,
                                                 const FieldMask &request_mask, 
                                                 RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      SendVersionStateArgs args;
      args.hlr_id = HLR_SEND_VERSION_STATE_TASK_ID;
      args.proxy_this = this;
      args.target = target;
      args.request_kind = req_kind;
      args.request_mask = legion_new<FieldMask>(request_mask);
      args.to_trigger = to_trigger;
      runtime->issue_runtime_meta_task(&args, sizeof(args),
                                       HLR_SEND_VERSION_STATE_TASK_ID, 
                                       HLR_LATENCY_PRIORITY,
                                       NULL/*op*/, precondition);
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_path_only(AddressSpaceID source,
                                                      FieldMask &path_only_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      AutoLock s_lock(state_lock);
      path_only_nodes[source] |= path_only_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_initialization(
                                 AddressSpaceID source, FieldMask &initial_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      AutoLock s_lock(state_lock);
      initial_nodes[source] |= initial_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_request(AddressSpaceID source,
                    RtUserEvent to_trigger, VersionRequestKind request_kind, 
                                                    FieldMask &request_mask)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_HANDLE_REQUEST_CALL);
      if (!is_owner())
      {
        // If we are not the owner, we should definitely be able to handle this
        std::set<RtEvent> launch_preconditions;
#ifdef DEBUG_LEGION
        FieldMask remaining_mask = request_mask;
#endif
        if (request_kind == FINAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock,1,false/*exclusive*/);
          for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                final_events.begin(); it != final_events.end(); it++)
          {
            if (it->second * request_mask)
              continue;
            launch_preconditions.insert(it->first);
          }
#ifdef DEBUG_LEGION
          remaining_mask -= final_fields;
#endif
        }
        else if (request_kind == INITIAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock,1,false/*exclusive*/);
          for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                initial_events.begin(); it != initial_events.end(); it++)
          {
            if (it->second * request_mask)
              continue;
            launch_preconditions.insert(it->first);
          }
#ifdef DEBUG_LEGION
          remaining_mask -= initial_fields;
#endif
        }
#ifdef DEBUG_LEGION
        else
        {
          assert(request_kind == PATH_ONLY_VERSION_REQUEST);
          // There are no preconditions for path only, but we
          // will take the lock to update the remaining mask
          // for debugging purposes
          AutoLock s_lock(state_lock,1,false/*exclusive*/);
          remaining_mask -= path_only_fields;
        }
        assert(!remaining_mask); // request mask should now be empty
#endif
        if (!launch_preconditions.empty())
        {
          RtEvent pre = Runtime::merge_events(launch_preconditions);
          launch_send_version_state(source, to_trigger, request_kind, 
                                    request_mask, pre);
        }
        else
          launch_send_version_state(source, to_trigger, 
                                    request_kind, request_mask);
      }
      else
      {
        // We're the owner, figure out what to do
        FieldMask remaining_fields = request_mask;
        FieldMask local_fields;
        int path_only_local_index = -1;
        int initial_local_index = -1;
        std::set<RtEvent> local_preconditions, done_conditions;
        LegionDeque<RequestInfo>::aligned targets;
        if (request_kind == FINAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock);
          // See if we can handle any of the fields locally
          local_fields = remaining_fields & final_fields;
          if (!!local_fields)
          {
            for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                  final_events.begin(); it != final_events.end(); it++)
            {
              if (it->second * local_fields)
                continue;
              local_preconditions.insert(it->first);
            }
            remaining_fields -= local_fields;
          }
          // See if there are any remote nodes that can handle it
          if (!!remaining_fields && !final_nodes.empty())
          {
            select_final_targets(source, remaining_fields,
                                 targets, done_conditions);
          }
          // Once we are here, we can update our remote state information
          final_nodes[source] |= request_mask;
        }
        else if (request_kind == INITIAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock);
          // See if we can handle any of the fields locally
          local_fields = remaining_fields & initial_fields;
          if (!!local_fields)
          {
            for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it =
                  initial_events.begin(); it != initial_events.end(); it++)
            {
              if (it->second * local_fields)
                continue;
              local_preconditions.insert(it->first);
            }
            remaining_fields -= local_fields;
          }
          // Now see if there are any remote nodes that can handle it
          if (!!remaining_fields && !initial_nodes.empty())
          {
            select_initial_targets(source, remaining_fields,
                                   targets, done_conditions);
          }
          // Once we are here we can update our remote state information
          initial_nodes[source] |= request_mask;
        }
        else // should never get a request for path only on the owner node
          assert(false);  
        // First, issue all our remote requests, pull out any that
        // were actually intended for us
        if (!targets.empty())
        {
          int idx = 0;
          for (LegionDeque<RequestInfo>::aligned::const_iterator it = 
                targets.begin(); it != targets.end(); it++, idx++)
          {
            if (it->target == local_space)
            {
              // Handle cases where we were supposed to send something
              if (request_kind == INITIAL_VERSION_REQUEST)
              {
#ifdef DEBUG_LEGION
                assert(it->kind == PATH_ONLY_VERSION_REQUEST);
#endif
                path_only_local_index = idx;
              }
              else
              {
#ifdef DEBUG_LEGION
                assert((request_kind == FINAL_VERSION_REQUEST) &&
                       (it->kind == INITIAL_VERSION_REQUEST));
#endif
                initial_local_index = idx;
              }
            }
            else
              send_version_state_request(it->target, source, it->to_trigger,
                                         it->request_mask, it->kind);
          }
        }
        // Now see if we have any local fields to send
        if (!!local_fields)
        {
          RtUserEvent local_trigger = Runtime::create_rt_user_event();
          if (!local_preconditions.empty())
          {
            RtEvent pre = Runtime::merge_events(local_preconditions);
            launch_send_version_state(source, local_trigger, request_kind,
                                      local_fields, pre); 
          }
          else
            launch_send_version_state(source, local_trigger,
                                      request_kind, local_fields); 
          done_conditions.insert(local_trigger);
        }
        // We might also have some additional local fields to send
        if (path_only_local_index >= 0)
        {
          RequestInfo &info = targets[path_only_local_index];
          launch_send_version_state(source, info.to_trigger, info.kind,
                                    info.request_mask);
          done_conditions.insert(info.to_trigger);
        }
        if (initial_local_index >= 0)
        {
          RequestInfo &info = targets[initial_local_index];
          local_preconditions.clear();
          // Retake the lock to read the local data structure
          {
            AutoLock s_lock(state_lock,1,false/*exclusive*/);
            for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                  initial_events.begin(); it != initial_events.end(); it++)
            {
              if (it->second * info.request_mask)
                continue;
              local_preconditions.insert(it->first);
            }
          }
          if (!local_preconditions.empty())
          {
            RtEvent pre = Runtime::merge_events(local_preconditions);
            launch_send_version_state(source, info.to_trigger, info.kind, 
                                      info.request_mask, pre);
          }
          else
            launch_send_version_state(source, info.to_trigger, info.kind,
                                      info.request_mask);
          done_conditions.insert(info.to_trigger);
        }
        // Now if we have any done conditions we trigger the proper 
        // precondition event, otherwise we can do it immediately
        if (!done_conditions.empty())
          Runtime::trigger_event(to_trigger,
                                 Runtime::merge_events(done_conditions));
        else
          Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_response(AddressSpaceID source,
                                                     RtUserEvent to_trigger, 
                                            VersionRequestKind request_kind, 
                                                     Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        VERSION_STATE_HANDLE_RESPONSE_CALL);
      // Special case for path only response
      if (request_kind == PATH_ONLY_VERSION_REQUEST)
      {
        {
          AutoLock s_lock(state_lock);
          size_t num_children;
          derez.deserialize(num_children);
          for (unsigned idx = 0; idx < num_children; idx++)
          {
            ColorPoint child;
            derez.deserialize(child);
            LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
              children.open_children.find(child);
            if (finder != children.open_children.end())
            {
              FieldMask child_update;
              derez.deserialize(child_update);
              finder->second |= child_update;
              children.valid_fields |= child_update;
            }
            else
            {
              FieldMask &mask = children.open_children[child];
              derez.deserialize(mask);
              children.valid_fields |= mask;
            }
          }
        }
        Runtime::trigger_event(to_trigger);
        return;
      }
      std::set<RtEvent> preconditions;
      std::map<LogicalView*,RtEvent> pending_views;
      WrapperReferenceMutator mutator(preconditions);
      {
        // Hold the lock when touching the data structures because we might
        // be getting multiple updates from different locations
        AutoLock s_lock(state_lock);
        // Check to see what state we are generating, if it is the initial
        // one then we can just do our unpack in-place, otherwise we have
        // to unpack separately and then do a merge.
        const bool in_place = !initial_fields && !final_fields;
        // Unpack the dirty and reduction fields
        if (in_place)
        {
          derez.deserialize(dirty_mask);
          derez.deserialize(reduction_mask);
        }
        else
        {
          FieldMask dirty_update;
          derez.deserialize(dirty_update);
          dirty_mask |= dirty_update;

          FieldMask reduction_update;
          derez.deserialize(reduction_update);
          reduction_mask |= reduction_update;
        }
        // Unpack the open children
        if (in_place && !path_only_fields)
        {
          size_t num_children;
          derez.deserialize(num_children);
          for (unsigned idx = 0; idx < num_children; idx++)
          {
            ColorPoint child;
            derez.deserialize(child);
            FieldMask &mask = children.open_children[child];
            derez.deserialize(mask);
            children.valid_fields |= mask;
          }
        }
        else
        {
          size_t num_children;
          derez.deserialize(num_children);
          for (unsigned idx = 0; idx < num_children; idx++)
          {
            ColorPoint child;
            derez.deserialize(child);
            LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
              children.open_children.find(child);
            if (finder != children.open_children.end())
            {
              FieldMask child_update;
              derez.deserialize(child_update);
              finder->second |= child_update;
              children.valid_fields |= child_update;
            }
            else
            {
              FieldMask &mask = children.open_children[child];
              derez.deserialize(mask);
              children.valid_fields |= mask;
            }
          }
        }
        // Finally do the views
        if (!initial_fields && !final_fields)
        {
          size_t num_valid_views;
          derez.deserialize(num_valid_views);
          for (unsigned idx = 0; idx < num_valid_views; idx++)
          {
            DistributedID view_did;
            derez.deserialize(view_did);
            RtEvent ready;
            LogicalView *view = 
              runtime->find_or_request_logical_view(view_did, ready);
            FieldMask &mask = valid_views[view];
            derez.deserialize(mask);
            if (ready.exists())
            {
              pending_views[view] = ready;
              continue;
            }
            view->add_nested_valid_ref(did, &mutator);
          }
          size_t num_reduction_views;
          derez.deserialize(num_reduction_views);
          for (unsigned idx = 0; idx < num_reduction_views; idx++)
          {
            DistributedID view_did;
            derez.deserialize(view_did);
            RtEvent ready;
            LogicalView *view =
              runtime->find_or_request_logical_view(view_did, ready);
            ReductionView *red_view = static_cast<ReductionView*>(view); 
            FieldMask &mask = reduction_views[red_view];
            derez.deserialize(mask);
            if (ready.exists())
            {
              pending_views[view] = ready;
              continue;
            }
#ifdef DEBUG_LEGION
            assert(view->is_instance_view());
            assert(view->as_instance_view()->is_reduction_view());
#endif
            view->add_nested_valid_ref(did, &mutator);
          }
        }
        else
        {
          size_t num_valid_views;
          derez.deserialize(num_valid_views);
          for (unsigned idx = 0; idx < num_valid_views; idx++)
          {
            DistributedID view_did;
            derez.deserialize(view_did);
            RtEvent ready;
            LogicalView *view =
              runtime->find_or_request_logical_view(view_did, ready);
            LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
              valid_views.find(view);
            if (finder != valid_views.end())
            {
              FieldMask update_mask;
              derez.deserialize(update_mask);
              finder->second |= update_mask;
              if (ready.exists())
              {
                pending_views[view] = ready;
                continue;
              }
            }
            else
            {
              FieldMask &mask = valid_views[view];
              derez.deserialize(mask);
              if (ready.exists())
              {
                pending_views[view] = ready;
                continue;
              }
              view->add_nested_valid_ref(did, &mutator);
            }
          }
          size_t num_reduction_views;
          derez.deserialize(num_reduction_views);
          for (unsigned idx = 0; idx < num_reduction_views; idx++)
          {
            DistributedID view_did;
            derez.deserialize(view_did);
            RtEvent ready;
            LogicalView *view =
              runtime->find_or_request_logical_view(view_did, ready);
            ReductionView *red_view = static_cast<ReductionView*>(view);
            LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
              reduction_views.find(red_view);
            if (finder != reduction_views.end())
            {
              FieldMask update_mask;
              derez.deserialize(update_mask);
              finder->second |= update_mask;
              if (ready.exists())
              {
                pending_views[view] = ready;
                continue;
              }
            }
            else
            {
              FieldMask &mask = reduction_views[red_view];
              derez.deserialize(mask);
              if (ready.exists())
              {
                pending_views[view] = ready;
                continue;
              }
              view->add_nested_valid_ref(did, &mutator);
            }
          }
        }
      }
      if (!pending_views.empty())
      {
        UpdateViewReferences args;
        args.hlr_id = HLR_UPDATE_VIEW_REFERENCES_TASK_ID;
        args.did = this->did;
        for (std::map<LogicalView*,RtEvent>::const_iterator it = 
              pending_views.begin(); it != pending_views.end(); it++)
        {
          if (it->second.has_triggered())
          {
            it->first->add_nested_valid_ref(did, &mutator);
            continue;
          }
          args.view = it->first;
          preconditions.insert(
              runtime->issue_runtime_meta_task(&args, sizeof(args),
                HLR_UPDATE_VIEW_REFERENCES_TASK_ID, HLR_LATENCY_PRIORITY,
                NULL, it->second));
        }
      }
      if (!preconditions.empty())
        Runtime::trigger_event(to_trigger,
                               Runtime::merge_events(preconditions));
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_view_references(const void *args)
    //--------------------------------------------------------------------------
    {
      const UpdateViewReferences *view_args = (const UpdateViewReferences*)args;
      LocalReferenceMutator mutator;
      view_args->view->add_nested_valid_ref(view_args->did, &mutator);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_path_only(
                       Runtime *rt, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask path_only_mask;
      derez.deserialize(path_only_mask);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_path_only(source, path_only_mask);
      RtUserEvent registered_event;
      derez.deserialize(registered_event);
      if (registered_event.exists())
        Runtime::trigger_event(registered_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_initialization(
                       Runtime *rt, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask initial_mask;
      derez.deserialize(initial_mask);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_initialization(source, initial_mask);
      RtUserEvent registered_event;
      derez.deserialize(registered_event);
      if (registered_event.exists())
        Runtime::trigger_event(registered_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_request(Runtime *rt,
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      VersionRequestKind request_kind;
      derez.deserialize(request_kind);
      FieldMask request_mask;
      derez.deserialize(request_mask);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_request(source, to_trigger, 
                                       request_kind, request_mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_response(Runtime *rt,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      VersionRequestKind req_kind;
      derez.deserialize(req_kind);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_response(source, to_trigger, req_kind, derez);
    }

    //--------------------------------------------------------------------------
    void VersionState::remove_version_state_ref(ReferenceSource ref_kind,
                                                RtEvent done_event)
    //--------------------------------------------------------------------------
    {
      RemoveVersionStateRefArgs args;
      args.hlr_id = HLR_REMOVE_VERSION_STATE_REF_TASK_ID;
      args.proxy_this = this;
      args.ref_kind = ref_kind;
      runtime->issue_runtime_meta_task(&args, sizeof(args),
          HLR_REMOVE_VERSION_STATE_REF_TASK_ID, HLR_LATENCY_PRIORITY,
          NULL, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_remove_version_state_ref(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const RemoveVersionStateRefArgs *ref_args = 
        (const RemoveVersionStateRefArgs*)args;
      if (ref_args->proxy_this->remove_base_valid_ref(ref_args->ref_kind))
        legion_delete(ref_args->proxy_this);     
    }

    /////////////////////////////////////////////////////////////
    // RegionTreePath 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreePath::RegionTreePath(void) 
      : min_depth(0), max_depth(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::initialize(unsigned min, unsigned max)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min <= max);
#endif
      min_depth = min;
      max_depth = max;
      path.resize(max_depth+1);
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::register_child(unsigned depth, 
                                        const ColorPoint &color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      path[depth] = color;
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::clear(void)
    //--------------------------------------------------------------------------
    {
      path.clear();
      min_depth = 0;
      max_depth = 0;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    bool RegionTreePath::has_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      return path[depth].is_valid();
    }

    //--------------------------------------------------------------------------
    const ColorPoint& RegionTreePath::get_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      assert(has_child(depth));
      return path[depth];
    }
#endif

    /////////////////////////////////////////////////////////////
    // FatTreePath 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FatTreePath::FatTreePath(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FatTreePath::FatTreePath(const FatTreePath &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FatTreePath::~FatTreePath(void)
    //--------------------------------------------------------------------------
    {
      // Delete all our children
      for (std::map<ColorPoint,FatTreePath*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        delete it->second;
      }
      children.clear();
    }

    //--------------------------------------------------------------------------
    FatTreePath& FatTreePath::operator=(const FatTreePath &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FatTreePath::add_child(const ColorPoint &child_color, 
                                FatTreePath *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(children.find(child_color) == children.end());
#endif
      children[child_color] = child;
    }

    //--------------------------------------------------------------------------
    bool FatTreePath::add_child(const ColorPoint &child_color,
                                FatTreePath *child, IndexTreeNode *tree_node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(children.find(child_color) == children.end());
#endif
      bool overlap = false;
      if (!children.empty())
      {
        if (tree_node->is_index_space_node())
        {
          IndexSpaceNode *node = tree_node->as_index_space_node();
          for (std::map<ColorPoint,FatTreePath*>::const_iterator it = 
                children.begin(); it != children.end(); it++)
          {
            if ((it->first == child_color) || 
                !node->are_disjoint(it->first, child_color))
            {
              overlap = true;
              break;
            }
          }
        }
        else
        {
          IndexPartNode *node = tree_node->as_index_part_node();
          for (std::map<ColorPoint,FatTreePath*>::const_iterator it = 
                children.begin(); it != children.end(); it++)
          {
            if ((it->first == child_color) || 
                !node->are_disjoint(it->first, child_color))
            {
              overlap = true;
              break;
            }
          }
        }
      }
      children[child_color] = child;
      return overlap;
    }

    /////////////////////////////////////////////////////////////
    // InstanceRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(bool comp)
      : ready_event(ApEvent::NO_AP_EVENT), composite(comp), local(true)
    //--------------------------------------------------------------------------
    {
      if (composite)
        ptr.view = NULL;
      else
        ptr.manager = NULL;
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(const InstanceRef &rhs)
      : valid_fields(rhs.valid_fields), ready_event(rhs.ready_event),
        composite(rhs.composite), local(rhs.local)
    //--------------------------------------------------------------------------
    {
      if (composite)
      {
        ptr.view = rhs.ptr.view;
        if (ptr.view != NULL)
          ptr.view->add_base_valid_ref(COMPOSITE_HANDLE_REF);
      }
      else
        ptr.manager = rhs.ptr.manager;
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(PhysicalManager *man, const FieldMask &m,ApEvent r)
      : valid_fields(m), ready_event(r), composite(false), local(true)
    //--------------------------------------------------------------------------
    {
      ptr.manager = man;
    }

    //--------------------------------------------------------------------------
    InstanceRef::~InstanceRef(void)
    //--------------------------------------------------------------------------
    {
      if (composite && (ptr.view != NULL) && 
          ptr.view->remove_base_valid_ref(COMPOSITE_HANDLE_REF))
        legion_delete(ptr.view);
    }

    //--------------------------------------------------------------------------
    InstanceRef& InstanceRef::operator=(const InstanceRef &rhs)
    //--------------------------------------------------------------------------
    {
      if (composite && (ptr.view != NULL) && 
          ptr.view->remove_base_valid_ref(COMPOSITE_HANDLE_REF))
        legion_delete(ptr.view);
      valid_fields = rhs.valid_fields;
      ready_event = rhs.ready_event;
      composite = rhs.composite;
      local = rhs.local;
      if (composite)
      {
        ptr.view = rhs.ptr.view;
        if (ptr.view != NULL)
          ptr.view->add_base_valid_ref(COMPOSITE_HANDLE_REF);
      }
      else
        ptr.manager = rhs.ptr.manager;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator==(const InstanceRef &rhs) const
    //--------------------------------------------------------------------------
    {
      if (composite != rhs.composite)
        return false;
      if (valid_fields != rhs.valid_fields)
        return false;
      if (ready_event != rhs.ready_event)
        return false;
      if (composite)
      {
        if (ptr.manager != rhs.ptr.manager)
          return false;
      }
      else
      {
        if (ptr.view != rhs.ptr.view)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator!=(const InstanceRef &rhs) const
    //--------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::set_composite_view(CompositeView *view,
                                         ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (composite && (ptr.view != NULL) && 
          ptr.view->remove_base_valid_ref(COMPOSITE_HANDLE_REF, mutator))
        legion_delete(ptr.view);
      ptr.view = view;
      if (ptr.view != NULL)
      {
        ptr.view->add_base_valid_ref(COMPOSITE_HANDLE_REF, mutator);
        composite = true;
      }
    }

    //--------------------------------------------------------------------------
    CompositeView* InstanceRef::get_composite_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(composite);
      assert(ptr.view != NULL);
#endif     
      return ptr.view;
    }

    //--------------------------------------------------------------------------
    MappingInstance InstanceRef::get_mapping_instance(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!composite);
#endif
      return MappingInstance(ptr.manager);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::add_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (composite)
      {
#ifdef DEBUG_LEGION
        assert(ptr.view != NULL);
#endif
        ptr.view->add_base_valid_ref(source);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(ptr.manager != NULL);
#endif
        ptr.manager->add_base_valid_ref(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (composite)
      {
#ifdef DEBUG_LEGION
        assert(ptr.view != NULL);
#endif
        if (ptr.view->remove_base_valid_ref(source))
          legion_delete(ptr.view);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(ptr.manager != NULL);
#endif
        if (ptr.manager->remove_base_valid_ref(source))
        {
          if (ptr.manager->is_reduction_manager())
          {
            ReductionManager *reduc_manager = ptr.manager->as_reduction_manager();
            if (reduc_manager->is_list_manager())
              legion_delete(reduc_manager->as_list_manager());
            else
              legion_delete(reduc_manager->as_fold_manager());
          }
          else
            legion_delete(ptr.manager->as_instance_manager());
        }
      }
    }

    //--------------------------------------------------------------------------
    Memory InstanceRef::get_memory(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!composite);
      assert(ptr.manager != NULL);
#endif
      return ptr.manager->get_memory();
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::is_field_set(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!composite);
      assert(ptr.manager != NULL);
#endif
      FieldSpaceNode *field_node = ptr.manager->region_node->column_source; 
      unsigned index = field_node->get_field_index(fid);
      return valid_fields.is_set(index);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic> 
        InstanceRef::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!composite);
      assert(ptr.manager != NULL);
#endif
      return ptr.manager->get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        InstanceRef::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!composite);
      assert(ptr.manager != NULL);
#endif
      return ptr.manager->get_field_accessor(fid);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::pack_reference(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(valid_fields);
      rez.serialize(ready_event);
      rez.serialize(composite);
      if (composite)
      {
        if (ptr.view != NULL)
          rez.serialize(ptr.view->did);
        else
          rez.serialize<DistributedID>(0);
      }
      else
      {
        if (ptr.manager != NULL)
          rez.serialize(ptr.manager->did);
        else
          rez.serialize<DistributedID>(0);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceRef::unpack_reference(Runtime *runtime, TaskOp *task, 
                                       Deserializer &derez, RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(valid_fields);
      derez.deserialize(ready_event);
      derez.deserialize(composite);
      DistributedID did;
      derez.deserialize(did);
      if (did == 0)
        return;
      if (composite)
      {
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists() && !ready.has_triggered())
        {
          // Otherwise we need to defer adding the handle reference until 
          // the view is actually ready
          // Have to static cast this to avoid touching it
          ptr.view = static_cast<CompositeView*>(view);
          DeferCompositeHandleArgs args;
          args.hlr_id = HLR_DEFER_COMPOSITE_HANDLE_TASK_ID;
          args.view = ptr.view;
          ready = runtime->issue_runtime_meta_task(&args, sizeof(args),
                        HLR_DEFER_COMPOSITE_HANDLE_TASK_ID,HLR_LATENCY_PRIORITY,
                        task, ready);
        }
        else
        {
          std::set<RtEvent> ready_events;
          WrapperReferenceMutator mutator(ready_events);
          // No need to wait, we are done now
          ptr.view = view->as_composite_view();
          ptr.view->add_base_valid_ref(COMPOSITE_HANDLE_REF, &mutator);
          if (!ready_events.empty())
            ready = Runtime::merge_events(ready_events);
        }
      }
      else
        ptr.manager = runtime->find_or_request_physical_manager(did, ready);
      local = false;
    } 

    //--------------------------------------------------------------------------
    /*static*/ void InstanceRef::handle_deferred_composite_handle(const void *a)
    //--------------------------------------------------------------------------
    {
      const DeferCompositeHandleArgs *args = (const DeferCompositeHandleArgs*)a;
      LocalReferenceMutator mutator;
      args->view->add_base_valid_ref(COMPOSITE_HANDLE_REF, &mutator);
    }

    /////////////////////////////////////////////////////////////
    // InstanceSet 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    InstanceSet::CollectableRef& InstanceSet::CollectableRef::operator=(
                                         const InstanceSet::CollectableRef &rhs)
    //--------------------------------------------------------------------------
    {
      // Do not copy references
      if (composite && (ptr.view != NULL) && 
          ptr.view->remove_base_valid_ref(COMPOSITE_HANDLE_REF))
        legion_delete(ptr.view);
      valid_fields = rhs.valid_fields;
      ready_event = rhs.ready_event;
      composite = rhs.composite;
      local = rhs.local;
      if (composite)
      {
        ptr.view = rhs.ptr.view;
        if (ptr.view != NULL)
          ptr.view->add_base_valid_ref(COMPOSITE_HANDLE_REF);
      }
      else
        ptr.manager = rhs.ptr.manager;
      return *this;
    }

    //--------------------------------------------------------------------------
    InstanceSet::InstanceSet(size_t init_size /*=0*/)
      : single((init_size <= 1)), shared(false)
    //--------------------------------------------------------------------------
    {
      if (init_size == 0)
        refs.single = NULL;
      else if (init_size == 1)
      {
        refs.single = legion_new<CollectableRef>();
        refs.single->add_reference();
      }
      else
      {
        refs.multi = new InternalSet(init_size);
        refs.multi->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet::InstanceSet(const InstanceSet &rhs)
      : single(rhs.single)
    //--------------------------------------------------------------------------
    {
      // Mark that the other one is sharing too
      if (single)
      {
        refs.single = rhs.refs.single;
        if (refs.single == NULL)
        {
          shared = false;
          return;
        }
        shared = true;
        rhs.shared = true;
        refs.single->add_reference();
      }
      else
      {
        refs.multi = rhs.refs.multi;
        shared = true;
        rhs.shared = true;
        refs.multi->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet::~InstanceSet(void)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((refs.single != NULL) && refs.single->remove_reference())
          legion_delete(refs.single);
      }
      else
      {
        if (refs.multi->remove_reference())
          delete refs.multi;
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet& InstanceSet::operator=(const InstanceSet &rhs)
    //--------------------------------------------------------------------------
    {
      // See if we need to delete our current one
      if (single)
      {
        if ((refs.single != NULL) && refs.single->remove_reference())
          legion_delete(refs.single);
      }
      else
      {
        if (refs.multi->remove_reference())
          delete refs.multi;
      }
      // Now copy over the other one
      single = rhs.single; 
      if (single)
      {
        refs.single = rhs.refs.single;
        if (refs.single != NULL)
        {
          shared = true;
          rhs.shared = true;
          refs.single->add_reference();
        }
        else
          shared = false;
      }
      else
      {
        refs.multi = rhs.refs.multi;
        shared = true;
        rhs.shared = true;
        refs.multi->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::make_copy(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shared);
#endif
      if (single)
      {
        if (refs.single != NULL)
        {
          CollectableRef *next = 
            legion_new<CollectableRef,InstanceRef>(*refs.single);
          next->add_reference();
          if (refs.single->remove_reference())
            legion_delete(refs.single);
          refs.single = next;
        }
      }
      else
      {
        InternalSet *next = new InternalSet(*refs.multi);
        next->add_reference();
        if (refs.multi->remove_reference())
          delete refs.multi;
        refs.multi = next;
      }
      shared = false;
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::operator==(const InstanceSet &rhs) const
    //--------------------------------------------------------------------------
    {
      if (single != rhs.single)
        return false;
      if (single)
      {
        if (refs.single == rhs.refs.single)
          return true;
        if (((refs.single == NULL) && (rhs.refs.single != NULL)) ||
            ((refs.single != NULL) && (rhs.refs.single == NULL)))
          return false;
        return ((*refs.single) == (*rhs.refs.single));
      }
      else
      {
        if (refs.multi->vector.size() != rhs.refs.multi->vector.size())
          return false;
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          if (refs.multi->vector[idx] != rhs.refs.multi->vector[idx])
            return false;
        }
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::operator!=(const InstanceSet &rhs) const
    //--------------------------------------------------------------------------
    {
      return !((*this) == rhs);
    }

    //--------------------------------------------------------------------------
    InstanceRef& InstanceSet::operator[](unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (shared)
        make_copy();
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(idx == 0);
        assert(refs.single != NULL);
#endif
        return *(refs.single);
      }
#ifdef DEBUG_LEGION
      assert(idx < refs.multi->vector.size());
#endif
      return refs.multi->vector[idx];
    }

    //--------------------------------------------------------------------------
    const InstanceRef& InstanceSet::operator[](unsigned idx) const
    //--------------------------------------------------------------------------
    {
      // No need to make a copy if shared here since this is read-only
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(idx == 0);
        assert(refs.single != NULL);
#endif
        return *(refs.single);
      }
#ifdef DEBUG_LEGION
      assert(idx < refs.multi->vector.size());
#endif
      return refs.multi->vector[idx];
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::empty(void) const
    //--------------------------------------------------------------------------
    {
      if (single && (refs.single == NULL))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    size_t InstanceSet::size(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single == NULL)
          return 0;
        return 1;
      }
      if (refs.multi == NULL)
        return 0;
      return refs.multi->vector.size();
    }

    //--------------------------------------------------------------------------
    void InstanceSet::resize(size_t new_size)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (new_size == 0)
        {
          if ((refs.single != NULL) && refs.single->remove_reference())
            legion_delete(refs.single);
          refs.single = NULL;
          shared = false;
        }
        else if (new_size > 1)
        {
          // Switch to multi
          InternalSet *next = new InternalSet(new_size);
          if (refs.single != NULL)
          {
            next->vector[0] = *(refs.single);
            if (refs.single->remove_reference())
              legion_delete(refs.single);
          }
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        // Otherwise new size is 1 so we don't need to do anything
      }
      else
      {
        if (new_size == 0)
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          refs.single = NULL;
          single = true;
          shared = false;
        }
        else if (new_size == 1)
        {
          CollectableRef *next = 
            legion_new<CollectableRef,InstanceRef>(refs.multi->vector[0]);
          if (refs.multi->remove_reference())
            legion_delete(refs.multi);
          next->add_reference();
          refs.single = next;
          single = true;
          shared = false;
        }
        else
        {
          size_t current_size = refs.multi->vector.size();
          if (current_size != new_size)
          {
            if (shared)
            {
              // Make a copy
              InternalSet *next = new InternalSet(new_size);
              // Copy over the elements
              for (unsigned idx = 0; idx < 
                   ((current_size < new_size) ? current_size : new_size); idx++)
                next->vector[idx] = refs.multi->vector[idx];
              if (refs.multi->remove_reference())
                delete refs.multi;
              next->add_reference();
              refs.multi = next;
              shared = false;
            }
            else
            {
              // Resize our existing vector
              refs.multi->vector.resize(new_size);
            }
          }
          // Size is the same so there is no need to do anything
        }
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::clear(void)
    //--------------------------------------------------------------------------
    {
      // No need to copy since we are removing our references and not mutating
      if (single)
      {
        if ((refs.single != NULL) && refs.single->remove_reference())
          legion_delete(refs.single);
        refs.single = NULL;
      }
      else
      {
        if (shared)
        {
          // Small optimization here, if we're told to delete it, we know
          // that means we were the last user so we can re-use it
          if (refs.multi->remove_reference())
          {
            // Put a reference back on it since we're reusing it
            refs.multi->add_reference();
            refs.multi->vector.clear();
          }
          else
          {
            // Go back to single
            refs.multi = NULL;
            single = true;
          }
        }
        else
          refs.multi->vector.clear();
      }
      shared = false;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_instance(const InstanceRef &ref)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        // No need to check for shared, we're going to make new things anyway
        if (refs.single != NULL)
        {
          // Make the new multi version
          InternalSet *next = new InternalSet(2);
          next->vector[0] = *(refs.single);
          next->vector[1] = ref;
          if (refs.single->remove_reference())
            legion_delete(refs.single);
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        else
        {
          refs.single = legion_new<CollectableRef,InstanceRef>(ref);
          refs.single->add_reference();
        }
      }
      else
      {
        if (shared)
          make_copy();
        refs.multi->vector.push_back(ref);
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::has_composite_ref(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single == NULL)
          return false;
        return refs.single->is_composite_ref();
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          if (refs.multi->vector[idx].is_composite_ref())
            return true;
        }
        return false;
      }
    }

    //--------------------------------------------------------------------------
    const InstanceRef& InstanceSet::get_composite_ref(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(refs.single != NULL);
        assert(refs.single->is_composite_ref());
#endif
        return (*refs.single);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          if (refs.multi->vector[idx].is_composite_ref())
            return refs.multi->vector[idx];
        }
        assert(false);
        return refs.multi->vector[0];
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::pack_references(Serializer &rez,
                                      AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single == NULL)
        {
          rez.serialize<size_t>(0);
          return;
        }
        rez.serialize<size_t>(1);
        refs.single->pack_reference(rez, target);
      }
      else
      {
        rez.serialize<size_t>(refs.multi->vector.size());
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].pack_reference(rez, target);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::unpack_references(Runtime *runtime, TaskOp *task,
                           Deserializer &derez, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_refs;
      derez.deserialize(num_refs);
      if (num_refs == 0)
      {
        // No matter what, we can just clear out any references we have
        if (single)
        {
          if ((refs.single != NULL) && refs.single->remove_reference())
            legion_delete(refs.single);
          refs.single = NULL;
        }
        else
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          single = true;
        }
      }
      else if (num_refs == 1)
      {
        // If we're in multi, go back to single
        if (!single)
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          refs.multi = NULL;
          single = true;
        }
        // Now we can unpack our reference, see if we need to make one
        if (refs.single == NULL)
        {
          refs.single = legion_new<CollectableRef>();
          refs.single->add_reference();
        }
        RtEvent ready;
        refs.single->unpack_reference(runtime, task, derez, ready);
        if (ready.exists())
          ready_events.insert(ready);
      }
      else
      {
        // If we're in single, go to multi
        // otherwise resize our multi for the appropriate number of references
        if (single)
        {
          if ((refs.single != NULL) && refs.single->remove_reference())
            legion_delete(refs.single);
          refs.multi = new InternalSet(num_refs);
          refs.multi->add_reference();
          single = false;
        }
        else
          refs.multi->vector.resize(num_refs);
        // Now do the unpacking
        for (unsigned idx = 0; idx < num_refs; idx++)
        {
          RtEvent ready;
          refs.multi->vector[idx].unpack_reference(runtime, task, derez, ready);
          if (ready.exists())
            ready_events.insert(ready);
        }
      }
      // We are always not shared when we are done
      shared = false;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_valid_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->add_valid_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].add_valid_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::remove_valid_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->remove_valid_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].remove_valid_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::update_wait_on_events(std::set<ApEvent> &wait_on) const 
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
        {
          ApEvent ready = refs.single->get_ready_event();
          if (ready.exists())
            wait_on.insert(ready);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          ApEvent ready = refs.multi->vector[idx].get_ready_event();
          if (ready.exists())
            wait_on.insert(ready);
        }
      }
    }
    
    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic> InstanceSet::
                                           get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(refs.single != NULL);
#endif
        return refs.single->get_field_accessor(fid);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          const InstanceRef &ref = refs.multi->vector[idx];
          if (ref.is_field_set(fid))
            return ref.get_field_accessor(fid);
        }
        assert(false);
        return refs.multi->vector[0].get_field_accessor(fid);
      }
    }

  }; // namespace Internal 
}; // namespace Legion

