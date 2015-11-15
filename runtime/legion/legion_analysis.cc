/* Copyright 2015 Stanford University, NVIDIA Corporation
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
#include "legion_logging.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"
#include "legion_analysis.h"

namespace LegionRuntime {
  namespace HighLevel {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Users and Info 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(void)
      : GenericUser(), op(NULL), idx(0), gen(0), timeout(TIMEOUT)
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
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
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
        , uid(o->get_unique_op_id())
#endif
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const RegionUsage &u, const ColorPoint &c,
                               FieldVersions *v)
      : usage(u), child(c), versions(v)
    //--------------------------------------------------------------------------
    {
      // Can be NULL in some cases
      if (versions != NULL)
        versions->add_reference();
#ifdef DEBUG_HIGH_LEVEL
      if (usage.redop > 0) // Use this property in pack and unpack
        assert(versions == NULL);
#endif
    }

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const PhysicalUser &rhs)
      : versions(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalUser::~PhysicalUser(void)
    //--------------------------------------------------------------------------
    {
      if ((versions != NULL) && versions->remove_reference())
        delete versions;
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
    bool PhysicalUser::same_versions(const FieldMask &test_mask,
                                     const FieldVersions *other) const
    //--------------------------------------------------------------------------
    {
      if ((other == NULL) || (versions == NULL))
        return false;
      const LegionMap<VersionID,FieldMask>::aligned &local_versions = 
        versions->get_field_versions();
      const LegionMap<VersionID,FieldMask>::aligned &other_versions = 
        other->get_field_versions();
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator vit = 
            local_versions.begin(); vit != local_versions.end(); vit++)
      {
        FieldMask overlap = vit->second & test_mask;
        if (!overlap)
          continue;
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              other_versions.begin(); it != other_versions.end(); it++)
        {
          FieldMask overlap2 = overlap & it->second;
          if (!overlap2)
            continue;
          if (vit->first != it->first)
            return false;
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalUser::pack_user(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(child);
      rez.serialize(usage.privilege);
      rez.serialize(usage.prop);
      if (versions != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(usage.redop == 0);
#endif
        const LegionMap<VersionID,FieldMask>::aligned &field_versions =
          versions->get_field_versions();
        int count = field_versions.size();
        count = -count; // negate for disambiguation
        rez.serialize(count);
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              field_versions.begin(); it != field_versions.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      else
      {
        int redop = usage.redop;
        rez.serialize(redop);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalUser* PhysicalUser::unpack_user(Deserializer &derez,
                                                       FieldSpaceNode *node,
                                                       AddressSpaceID source,
                                                       bool add_reference)
    //--------------------------------------------------------------------------
    {
      ColorPoint child;
      derez.deserialize(child);
      RegionUsage usage;
      derez.deserialize(usage.privilege);
      derez.deserialize(usage.prop);
      int redop;
      derez.deserialize(redop);
      PhysicalUser *result = NULL;
      if (redop <= 0)
      {
        usage.redop = 0;
        FieldVersions *versions = NULL;
        if (redop < 0)
        {
          int count = -redop;
          versions = new FieldVersions();
          for (int idx = 0; idx < count; idx++)
          {
            VersionID vid;
            derez.deserialize(vid);
            FieldMask version_mask;
            derez.deserialize(version_mask);
            node->transform_field_mask(version_mask, source);
            versions->add_field_version(vid, version_mask);
          }
        }
        result = legion_new<PhysicalUser>(usage, child, versions);
      }
      else
      {
        usage.redop = redop;
        result = legion_new<PhysicalUser>(usage, child);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      if (add_reference)
        result->add_reference();
      return result;
    }

    //--------------------------------------------------------------------------
    MappableInfo::MappableInfo(ContextID c, Operation *o, Processor p,
                               RegionRequirement &r, VersionInfo &info,
                               const FieldMask &k)
      : ctx(c), op(o), local_proc(p), req(r), 
        version_info(info), traversal_mask(k)
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(upper_bound_node == NULL);
      assert(!packed);
#endif
      upper_bound_node = node;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::merge(const VersionInfo &rhs, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
          if (vit->second.path_only())
            next.set_path_only();
          if (vit->second.needs_final())
            next.set_needs_final();
          if (vit->second.close_top())
            next.set_close_top();
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
                                    std::set<Event> &applied_conditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!packed);
#endif
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        if (it->second.physical_state == NULL)
          continue;
        // Apply path only differently
        if (it->second.path_only())
          it->second.physical_state->apply_path_only_state(
                    it->second.advance_mask, target, applied_conditions);
        else
          it->second.physical_state->apply_state(it->second.advance_mask,
                                             target, applied_conditions);
        // Don't delete it because we need to hold onto the 
        // version manager references in case this operation
        // fails to complete
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::apply_close(ContextID ctx, bool permit_leave_open,
                     AddressSpaceID target, std::set<Event> &applied_conditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!packed);
#endif
      if (permit_leave_open)
      {
        for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
              node_infos.begin(); it != node_infos.end(); it++)
        {
          if (it->second.physical_state == NULL)
            continue;
          if (it->second.path_only())
            it->second.physical_state->apply_path_only_state(
                        it->second.advance_mask, target, applied_conditions);
          else
            it->second.physical_state->filter_and_apply(it->second.close_top(),
                        false/*filter children*/, target, applied_conditions);
        }
      }
      else
      {
        for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
              node_infos.begin(); it != node_infos.end(); it++)
        {
          if (it->second.physical_state == NULL)
            continue;
          if (it->second.path_only())
          {
            it->second.physical_state->apply_path_only_state(
                        it->second.advance_mask, target, applied_conditions);
          }
          // We can also skip anything that isn't the top node
          if (!it->second.close_top())
            continue;
          it->second.physical_state->filter_and_apply(true/*top*/,
                         true/*filter children*/, target, applied_conditions);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::reset(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(!packed);
#endif
      LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator finder = 
        node_infos.find(node);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != node_infos.end());
#endif
      // Check to see if we need a reset
      if ((finder->second.physical_state != NULL) && capture &&
           finder->second.needs_capture())
      { 
        // Recapture the state if we had to be reset
        finder->second.physical_state->capture_state(finder->second.path_only(),
                                                    finder->second.close_top());
        finder->second.unset_needs_capture();
      }
      return finder->second.physical_state;
    }

    //--------------------------------------------------------------------------
    FieldVersions* VersionInfo::get_versions(RegionTreeNode *node) const
    //--------------------------------------------------------------------------
    {
      LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator finder = 
        node_infos.find(node);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != node_infos.end());
#endif
      // It's alright for this to return NULL
      return finder->second.field_versions;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_version_info(Serializer &rez, 
                                        AddressSpaceID local, ContextID ctx)
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
        pack_buffer(local_rez, local, ctx);
        size_t total_size = local_rez.get_used_bytes();
        rez.serialize(total_size);
        rez.serialize(local_rez.get_buffer(), total_size);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_version_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!packed);
      assert(packed_buffer == NULL);
#endif
      derez.deserialize(packed_size);
      packed_buffer = malloc(packed_size);
      derez.deserialize(packed_buffer, packed_size);
      packed = true;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::make_local(std::set<Event> &preconditions, 
                                 RegionTreeForest *forest, ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (packed)
        unpack_buffer(forest, ctx);
      // Iterate over all version state infos and build physical states
      // without actually capturing any data
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::iterator it = 
            node_infos.begin(); it != node_infos.end(); it++)
      {
        NodeInfo &info = it->second;
        if (info.physical_state == NULL)
        {
          info.physical_state = it->first->get_physical_state(ctx, *this,
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != rhs.node_infos.end());
#endif
        node_infos.insert(*finder);
        if (current == rhs.upper_bound_node)
        {
          upper_bound_node = current;
          break;
        }
        current = current->get_parent();
#ifdef DEBUG_HIGH_LEVEL
        assert(current != NULL);
#endif
        current_finder = node_infos.find(current);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clone_from(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!packed);
      assert(!rhs.packed);
#endif
      upper_bound_node = rhs.upper_bound_node;
      for (LegionMap<RegionTreeNode*,NodeInfo>::aligned::const_iterator nit = 
            rhs.node_infos.begin(); nit != rhs.node_infos.end(); nit++)
      {
        const NodeInfo &current = nit->second;
#ifdef DEBUG_HIGH_LEVEL
        assert(current.field_versions != NULL);
#endif
        NodeInfo &next = node_infos[nit->first];
#ifdef DEBUG_HIGH_LEVEL
        assert(next.physical_state == NULL);
#endif
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(node_infos.find(upper_bound_node) != node_infos.end());
#endif
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_buffer(Serializer &rez, 
                                  AddressSpaceID local_space, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!packed);
#endif
      rez.serialize(local_space);
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
          pack_node_info(rez, it->second, it->first, ctx);
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
          pack_node_info(rez, it->second, it->first, ctx);
        }
      }
      if (node_infos.size() > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
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
    void VersionInfo::unpack_buffer(RegionTreeForest *forest, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(packed);
      assert(packed_buffer != NULL);
#endif
      Deserializer derez(packed_buffer, packed_size);
      // Unpack the source
      AddressSpaceID source;
      derez.deserialize(source);
      // Unpack the node infos
      size_t num_regions;
      derez.deserialize(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionTreeNode *node = forest->get_node(handle);
        unpack_node_info(node, ctx, derez, source);
      }
      size_t num_partitions;
      derez.deserialize(num_partitions);
      for (unsigned idx = 0; idx < num_partitions; idx++)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        RegionTreeNode *node = forest->get_node(handle);
        unpack_node_info(node, ctx, derez, source);
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
                                     RegionTreeNode *node, ContextID ctx)
    //--------------------------------------------------------------------------
    {
      rez.serialize(info.bit_mask);
      if (info.physical_state == NULL)
        info.physical_state = node->get_physical_state(ctx, *this,
                                                       false/*capture*/);
      PhysicalState *state = info.physical_state;
#ifdef DEBUG_HIGH_LEVEL
      if (!info.advance_mask)
        assert(state->advance_states.empty());
#endif
      size_t total_version_states = 0;
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator it1 = 
            state->version_states.begin(); it1 != 
            state->version_states.end(); it1++)
      {
        const VersionStateInfo &state_info = it1->second;
        total_version_states += state_info.states.size();
      }
      rez.serialize(total_version_states);
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator it1 = 
            state->version_states.begin(); it1 != 
            state->version_states.end(); it1++)
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
    void VersionInfo::unpack_node_info(RegionTreeNode *node, ContextID ctx,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      NodeInfo &info = node_infos[node];
      CurrentState *manager = node->get_current_state_ptr(ctx);
#ifdef DEBUG_HIGH_LEVEL
      info.physical_state = legion_new<PhysicalState>(manager, node);
#else
      info.physical_state = legion_new<PhysicalState>(manager);
#endif
      // Don't need premap
      derez.deserialize(info.bit_mask);
      // Mark that we definitely need to recapture this node info
      info.set_needs_capture();
      // Unpack the version states
      size_t num_states;
      derez.deserialize(num_states);
      if (num_states > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(info.field_versions == NULL);
#endif
        info.field_versions = new FieldVersions();
        info.field_versions->add_reference();
      }
      FieldSpaceNode *field_node = node->column_source;
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
        field_node->transform_field_mask(mask, source);
        VersionState *state = node->find_remote_version_state(ctx, vid, 
                                                              did, owner);
        info.physical_state->add_version_state(state, mask);
        // Also add this to the version numbers
        // Only need to do this for the non-advance states
        info.field_versions->add_field_version(vid, mask);
      }
      // Unpack the advance states
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
        field_node->transform_field_mask(mask, source);
        VersionState *state = node->find_remote_version_state(ctx, vid, 
                                                              did, owner);
        // No point in adding this to the version state infos
        // since we already know we just use that to build the PhysicalState
        info.physical_state->add_advance_state(state, mask);
        // Update the advance mask as we go
        info.advance_mask |= mask;
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
        field_node->transform_field_mask(mask, source);
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
#ifdef DEBUG_HIGH_LEVEL
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
                                     Processor local, Operation *o)
      : ctx(c), target(t), close_mask(m), version_info(info), 
        local_proc(local), op(o)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReductionCloser::ReductionCloser(const ReductionCloser &rhs)
      : ctx(0), target(NULL), close_mask(FieldMask()), 
        version_info(rhs.version_info), local_proc(Processor::NO_PROC), op(NULL)
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
                                      local_proc, valid_reductions, op);
    }

    /////////////////////////////////////////////////////////////
    // PremapTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PremapTraverser::PremapTraverser(RegionTreePath &p, const MappableInfo &i)
      : PathTraverser(p), info(i)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PremapTraverser::PremapTraverser(const PremapTraverser &rhs)
      : PathTraverser(rhs.path), info(rhs.info)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PremapTraverser::~PremapTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PremapTraverser& PremapTraverser::operator=(const PremapTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PremapTraverser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      return premap_node(node, node->handle);
    }

    //--------------------------------------------------------------------------
    bool PremapTraverser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      return premap_node(node, node->parent->handle);
    }

    //--------------------------------------------------------------------------
    bool PremapTraverser::premap_node(RegionTreeNode *node, 
                                                   LogicalRegion closing_handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, PERFORM_PREMAP_CLOSE_CALL);
#endif
      PhysicalState *state = node->get_physical_state(info.ctx, 
                                                      info.version_info);
      // Update our physical state to indicate which child
      // we are opening and in which fields
      if (has_child)
      {
        state->children.valid_fields |= info.traversal_mask;
        LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
                            state->children.open_children.find(next_child);
        if (finder == state->children.open_children.end())
          state->children.open_children[next_child] = info.traversal_mask;
        else
          finder->second |= info.traversal_mask;
      }
      // Finally check to see if we arrived at our destination node
      // in which case we should pull down the valid instance views
      // to our node
      else if (!IS_REDUCE(info.req))
      {
        // If we're not doing a reduction, pull down all the valid views
        // and then record the valid physical instances unless we're
        // doing a reductions in which case it doesn't matter
        node->pull_valid_instance_views(info.ctx, state, info.traversal_mask, 
                                        true/*need space*/, info.version_info);
        // Find the memories for all the instances and report
        // which memories have full instances and which ones
        // only have partial instances
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              state->valid_views.begin(); it != state->valid_views.end(); it++)
        {
          if (it->first->is_deferred_view())
            continue;
#ifdef DEBUG_HIGH_LEVEL
          assert(it->first->as_instance_view()->is_materialized_view());
#endif
          MaterializedView *cur_view = 
            it->first->as_instance_view()->as_materialized_view();
          Memory mem = cur_view->get_location();
          std::map<Memory,bool>::iterator finder = 
            info.req.current_instances.find(mem);
          if ((finder == info.req.current_instances.end()) 
              || !finder->second)
          {
            bool full_instance = !(info.traversal_mask - it->second);
            if (finder == info.req.current_instances.end())
              info.req.current_instances[mem] = full_instance;
            else
              finder->second = full_instance;
          }
        }
        // Also set the maximum blocking factor for this region
        Domain node_domain = node->get_domain_blocking();
        if (node_domain.get_dim() == 0)
        {
          const LowLevel::ElementMask &mask = 
            node_domain.get_index_space().get_valid_mask();
          info.req.max_blocking_factor = mask.get_num_elmts();
        }
        else
          info.req.max_blocking_factor = node_domain.get_volume();
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // MappingTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingTraverser::MappingTraverser(RegionTreePath &p, 
                                       const MappableInfo &i,
                                       const RegionUsage &u, const FieldMask &m,
                                       Processor proc, unsigned idx, bool res)
      : PathTraverser(p), info(i), usage(u), user_mask(m), 
        target_proc(proc), index(idx), restricted(res)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MappingTraverser::MappingTraverser(const MappingTraverser &rhs)
      : PathTraverser(rhs.path), info(rhs.info), usage(RegionUsage()),
        user_mask(FieldMask()), target_proc(rhs.target_proc), 
        index(rhs.index), restricted(rhs.restricted)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MappingTraverser::~MappingTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappingTraverser& MappingTraverser::operator=(const MappingTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      if (!has_child)
      {
        // Now we're ready to map this instance
        // Separate paths for reductions and non-reductions
        if (!IS_REDUCE(info.req))
        {
          // See if we can get or make a physical instance
          // that we can use
          if (restricted)
            return map_restricted_physical(node);
          else
            return map_physical_region(node);
        }
        else
        {
          // See if we can make or use an existing reduction instance
          if (restricted)
            return map_restricted_reduction(node);
          else
            return map_reduction_region(node);
        }
      }
      else
      {
        // Still not there yet, traverse the node
        traverse_node(node);
        return true;
      } 
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      // Since we know we're mapping we know we won't ever stop
      // on a partition node
#ifdef DEBUG_HIGH_LEVEL
      assert(has_child);
#endif
      traverse_node(node);
      return true;
    }

    //--------------------------------------------------------------------------
    const MappingRef& MappingTraverser::get_instance_ref(void) const
    //--------------------------------------------------------------------------
    {
      return result;
    }

    //--------------------------------------------------------------------------
    void MappingTraverser::traverse_node(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, MAPPING_TRAVERSE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(has_child);
#endif
      PhysicalState *state = node->get_physical_state(info.ctx,
                                                      info.version_info);
      state->children.valid_fields |= info.traversal_mask;
      LegionMap<ColorPoint,FieldMask>::aligned::iterator finder = 
                              state->children.open_children.find(next_child);
      if (finder == state->children.open_children.end())
        state->children.open_children[next_child] = info.traversal_mask;
      else
        finder->second |= info.traversal_mask;
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::map_physical_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, MAP_PHYSICAL_REGION_CALL);
#endif
      std::vector<Memory> &chosen_order = info.req.target_ranking;
      const std::set<FieldID> &additional_fields = info.req.additional_fields;
      // Clamp the selected blocking factor
      const size_t blocking_factor = 
        (info.req.blocking_factor <= info.req.max_blocking_factor) ? 
        info.req.blocking_factor : info.req.max_blocking_factor;
      // Filter out any memories that are not visible from 
      // the target processor if there is a processor that 
      // we're targeting (e.g. never do this for premaps)
      // We can also skip this if region requirement has a NO_ACCESS_FLAG
      if (!chosen_order.empty() && target_proc.exists() &&
          !(info.req.flags & NO_ACCESS_FLAG))
      {
        Machine machine = Machine::get_machine();
        std::set<Memory> visible_memories;
	machine.get_visible_memories(target_proc, visible_memories);
        if (visible_memories.empty() && 
            (target_proc.kind() == Processor::PROC_GROUP))
        {
          log_run.warning("Comment on github that you've encountered "
                          "issue #35");
        }
        else
        {
          std::vector<Memory> filtered_memories;
          filtered_memories.reserve(chosen_order.size());
          for (std::vector<Memory>::const_iterator it = chosen_order.begin();
                it != chosen_order.end(); it++)
          {
            if (visible_memories.find(*it) == visible_memories.end())
            {
              log_region.warning("WARNING: Mapper specified memory " IDFMT 
                                       " which is not visible from processor "
                                       "" IDFMT " when mapping region %d of "
                                       "mappable (ID %lld)!  Removing memory "
                                       "from the chosen ordering!", it->id, 
                                       target_proc.id, index, 
                           info.op->get_mappable()->get_unique_mappable_id());
              continue;
            }
            // Otherwise we can add it to the list of filtered memories
            filtered_memories.push_back(*it);
          }
          chosen_order = filtered_memories;
        }
      }
      // Get the set of currently valid instances
      LegionMap<LogicalView*,FieldMask>::aligned valid_instances;
      // Check to see if the mapper requested any additional fields in this
      // instance.  If it did, then re-run the computation to get the list
      // of valid instances with the right set of fields
      std::set<FieldID> new_fields = info.req.privilege_fields;
      PhysicalState *state = node->get_physical_state(info.ctx,
                                                      info.version_info);
      if (!additional_fields.empty())
      {
        new_fields.insert(additional_fields.begin(),
                             additional_fields.end());
        FieldMask additional_mask = 
          node->column_source->get_field_mask(new_fields);
        node->find_valid_instance_views(info.ctx, state, additional_mask,
                                        additional_mask, info.version_info,
                                        true/*space*/, valid_instances);
      }
      else
      {
        node->find_valid_instance_views(info.ctx, state, user_mask,
                                        user_mask, info.version_info,
                                        true/*space*/, valid_instances);
      }
      // Compute the set of valid memories and filter out instance which
      // do not have the proper blocking factor in the process
      std::map<Memory,bool> valid_memories;
      {
        std::vector<LogicalView*> to_erase;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          // Remove any deferred instances
          if (it->first->is_deferred_view())
          {
            to_erase.push_back(it->first);
            continue;
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(it->first->as_instance_view()->is_materialized_view());
#endif
          MaterializedView *current_view = 
            it->first->as_instance_view()->as_materialized_view();
          // For right now allow blocking factors that are greater
          // than or equal to the requested blocking factor
          size_t bf = current_view->get_blocking_factor();
          if (bf >= blocking_factor)
          {
            Memory m = current_view->get_location();
            if (valid_memories.find(m) == valid_memories.end())
              valid_memories[m] = !(user_mask - it->second);
            else if (!valid_memories[m])
              valid_memories[m] = !(user_mask - it->second);
            // Otherwise we already have an instance in this memory that
            // dominates all the fields in which case we don't care
          }
          else
          {
            to_erase.push_back(it->first);
          }
        }
        for (std::vector<LogicalView*>::const_iterator it = to_erase.begin();
              it != to_erase.end(); it++)
          valid_instances.erase(*it);  
        to_erase.clear();
      }

      MaterializedView *chosen_inst = NULL;
      FieldMask needed_fields; 
      // Go through each of the memories provided by the mapper
      for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
            mit != chosen_order.end(); mit++)
      {
        // See if it has any valid instances
        if (valid_memories.find(*mit) != valid_memories.end())
        {
          // Already have a valid instance with at least a 
          // few valid fields, figure out if it has all or 
          // some of the fields valid
          if (valid_memories[*mit])
          {
            // We've got an instance with all the valid fields, go find it
            for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator
                  it = valid_instances.begin(); 
                  it != valid_instances.end(); it++)
            {
              // At this point we know everything is a materialized view
#ifdef DEBUG_HIGH_LEVEL
              assert(it->first->is_instance_view());
              assert(it->first->as_instance_view()->is_materialized_view());
#endif
              MaterializedView *cur_view = 
                it->first->as_instance_view()->as_materialized_view();
              if (cur_view->get_location() != (*mit))
                continue;
              if (!(user_mask - it->second))
              {
                // Check to see if have any WAR dependences
                // in which case we'll skip it for a something better
                if (info.req.enable_WAR_optimization && HAS_WRITE(info.req) 
                    && cur_view->has_war_dependence(usage, user_mask))
                  continue;
                // No WAR problems, so it it is good
                chosen_inst = cur_view;
                // No need to set needed fields since everything is valid
                break;
              }
            }
            // If we found a good instance break, otherwise go onto
            // the partial instances
            if (chosen_inst != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!needed_fields);
#endif
              break;
            }
          }
          // Do this if we couldn't find a better choice
          // Note we can't do this in the read-only case because we might 
          // end up issuing multiple copies to the same location.
          // On second thought this might be ok since they are both 
          // reading and anybody else who mutates this instance will
          // see both copies because of mapping dependences.
          // if (!IS_READ_ONLY(usage))
          {
            // These are instances which have space for all the required fields
            // but only a subset of those fields contain valid data.
            // Find the valid instance with the most valid fields to use.
            int covered_fields = -1;
            for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator 
                  it = valid_instances.begin(); 
                  it != valid_instances.end(); it++)
            {
              // At this point we know everything is a materialized view
#ifdef DEBUG_HIGH_LEVEL
              assert(it->first->is_instance_view());
              assert(it->first->as_instance_view()->is_materialized_view());
#endif
              MaterializedView *cur_view = 
                it->first->as_instance_view()->as_materialized_view();
              if (cur_view->get_location() != (*mit))
                continue;
              int cf = FieldMask::pop_count(it->second);
              if (cf > covered_fields)
              {
                // Check to see if we have any WAR dependences 
                // which might disqualify us
                if (info.req.enable_WAR_optimization && HAS_WRITE(info.req) 
                    && cur_view->has_war_dependence(usage, user_mask))
                  continue;
                covered_fields = cf;
                chosen_inst = cur_view; 
                needed_fields = user_mask - it->second; 
              }
            }
            // If we got a good one break out, otherwise we'll try 
            // to make a new instance
            if (chosen_inst != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!!needed_fields);
#endif
              break;
            }
          }
        }
        // If it didn't find a valid instance, try to make one
        chosen_inst = node->create_instance(*mit, new_fields, 
                                            blocking_factor,
                                          info.op->get_mappable()->get_depth(),
                                            info.op);
        if (chosen_inst != NULL)
        {
          // We successfully made an instance
          needed_fields = user_mask;
          // Make sure to tell our physical state
          state->record_created_instance(chosen_inst);
          break;
        }
      }
      // Save our chosen instance if it exists in the mapping
      // reference and then return if we have an instance
      if (chosen_inst != NULL)
        result = MappingRef(chosen_inst, needed_fields);
      return (chosen_inst != NULL);
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::map_reduction_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, MAP_REDUCTION_REGION_CALL);
#endif
      std::vector<Memory> &chosen_order = info.req.target_ranking;
      // Filter out any memories that are not visible from 
      // the target processor if there is a processor that 
      // we're targeting (e.g. never do this for premaps)
      if (!chosen_order.empty() && target_proc.exists())
      {
        Machine machine = Machine::get_machine();
        std::set<Memory> visible_memories;
	machine.get_visible_memories(target_proc, visible_memories);
        std::vector<Memory> filtered_memories;
        filtered_memories.reserve(chosen_order.size());
        for (std::vector<Memory>::const_iterator it = chosen_order.begin();
              it != chosen_order.end(); it++)
        {
          if (visible_memories.find(*it) != visible_memories.end())
            filtered_memories.push_back(*it);
          else
          {
            log_region.warning("WARNING: Mapper specified memory " IDFMT
                                     " which is not visible from processor "
                                     IDFMT " when mapping region %d of mappable"
                                     " (ID %lld)!  Removing memory from the "
                                     "chosen ordering!", it->id, 
                                     target_proc.id, index, 
                             info.op->get_mappable()->get_unique_mappable_id());
          }
        }
        chosen_order = filtered_memories;
      }

      std::set<ReductionView*> valid_views;
      PhysicalState *state = node->get_physical_state(info.ctx,
                                                      info.version_info);
      node->find_valid_reduction_views(info.ctx, state, usage.redop, 
                              user_mask, info.version_info, valid_views);

      // Compute the set of valid memories
      std::set<Memory> valid_memories;
      for (std::set<ReductionView*>::const_iterator it = valid_views.begin();
            it != valid_views.end(); it++)
      {
        valid_memories.insert((*it)->get_location());
      }

      ReductionView *chosen_inst = NULL;
      // Go through each of the valid memories and see if we can either find
      // a reduction instance or we can make one
      for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
            mit != chosen_order.end(); mit++)
      {
        if (valid_memories.find(*mit) != valid_memories.end())
        {
          // We've got a valid instance, let's go find it
          for (std::set<ReductionView*>::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            if ((*it)->get_location() == *mit)
            {
              chosen_inst = *it;
              break;
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(chosen_inst != NULL);
#endif
          // We've found the instance that we want
          break;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(info.req.privilege_fields.size() == 1);
#endif
          FieldID fid = *(info.req.privilege_fields.begin());
          // Try making a reduction instance in this memory
          chosen_inst = node->create_reduction(*mit, fid, 
                                               info.req.reduction_list,
                                               info.req.redop,
                                               info.op);
          if (chosen_inst != NULL)
          {
            state->record_created_instance(chosen_inst);
            break;
          }
        }
      }
      if (chosen_inst != NULL)
        result = MappingRef(chosen_inst,FieldMask());
      return (chosen_inst != NULL);
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::map_restricted_physical(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      // Grab the set of valid instances, we should find exactly one
      // that matches all the fields, if not that is very bad
      LegionMap<LogicalView*,FieldMask>::aligned valid_instances;
      PhysicalState *state = node->get_physical_state(info.ctx,
                                                      info.version_info);
      node->find_valid_instance_views(info.ctx, state, user_mask,
                                      user_mask, info.version_info,
                                      false/*space*/, valid_instances);
      InstanceView *chosen_inst = NULL;
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        // Skip any deferred views
        if (it->first->is_deferred_view())
          continue;
#ifdef DEBUG_HIGH_LEVEL
        assert(it->first->is_instance_view());
#endif
        InstanceView *inst_view = it->first->as_instance_view();
        FieldMask uncovered = user_mask - it->second;
        // If all the fields were valid, record it
        if (!uncovered)
        {
          if (chosen_inst != NULL)
          {
            for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it2=
                  valid_instances.begin(); it2 != valid_instances.end(); it2++)
            {
              LogicalView *view = it2->first;
              FieldMask mask = it2->second;
              printf("%p, %p\n", view, &mask);
            }
            log_run.error("Multiple valid instances for restricted cohernece! "
                          "This is almost certainly a runtime bug. Please "
                          "create a minimal test case and report it.");
            assert(false);
          }
          else
            chosen_inst = inst_view;
        }
      }
      if (chosen_inst == NULL)
      {
        log_run.error("No single instance is valid for restricted coherence! "
                      "Need support for multiple instances. This is currently "
                      "a pending feature. Please report your use case.");
        assert(false);
      }
      // We know we don't need any fields to be brought up to date
      result = MappingRef(chosen_inst, FieldMask());
      return (chosen_inst != NULL);
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::map_restricted_reduction(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this later
      assert(false);
      return false;
    }

    /////////////////////////////////////////////////////////////
    // CurrentState 
    /////////////////////////////////////////////////////////////
    
    // C++ is dumb
    const VersionID CurrentState::init_version;

    //--------------------------------------------------------------------------
    CurrentState::CurrentState(RegionTreeNode *node, ContextID ctx)
      : owner(node), state_lock(Reservation::create_reservation()),
        has_persistent(false)
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
#ifdef DEBUG_HIGH_LEVEL
      assert(field_states.empty());
      assert(curr_epoch_users.empty());
      assert(prev_epoch_users.empty());
      assert(current_version_infos.empty());
      assert(previous_version_infos.empty());
      assert(!has_persistent);
      assert(persistent_views.empty());
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
      if (!persistent_views.empty())
      {
        for (std::set<MaterializedView*>::const_iterator it =
              persistent_views.begin(); it != persistent_views.end(); it++)
        {
          if ((*it)->remove_base_valid_ref(PERSISTENCE_REF))
            legion_delete(*it);
        }
        persistent_views.clear();
      }
      has_persistent = false;
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
    void CurrentState::initialize_state(LogicalView *view, Event term_event,
                                        const RegionUsage &usage,
                                        const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_version_infos.empty() || 
              (current_version_infos.size() == 1));
      assert(previous_version_infos.empty());
#endif
      // No need to hold the lock when initializing
      if (current_version_infos.empty())
      {
        VersionState *init_state = create_new_version_state(init_version);
        init_state->add_base_valid_ref(VERSION_MANAGER_REF);
        init_state->initialize(view, term_event, usage, user_mask);
        current_version_infos[init_version].valid_fields = user_mask;
        current_version_infos[init_version].states[init_state] = user_mask;
      }
      else
      {
        LegionMap<VersionID,VersionStateInfo>::aligned::iterator finder = 
          current_version_infos.find(init_version);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != current_version_infos.end());
#endif
        finder->second.valid_fields |= user_mask;
#ifdef DEBUG_HIGH_LEVEL
        assert(finder->second.states.size() == 1);
#endif
        LegionMap<VersionState*,FieldMask>::aligned::iterator it = 
          finder->second.states.begin();
        it->first->initialize(view, term_event, usage, user_mask);
        it->second |= user_mask;
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void CurrentState::record_version_numbers(const FieldMask &mask,
                                              const LogicalUser &user,
                                              VersionInfo &version_info,
                                              bool capture_previous, 
                                              bool path_only, bool needs_final,
                                              bool close_top, bool report)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
      version_info.sanity_check(owner);
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
      if (capture_previous)
        node_info.advance_mask |= mask;
      if (node_info.physical_state == NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        node_info.physical_state = legion_new<PhysicalState>(this, owner);
#else
        node_info.physical_state = legion_new<PhysicalState>(this);
#endif 
      }
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
#ifdef DEBUG_HIGH_LEVEL
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
          node_info.field_versions->add_field_version(init_version,unversioned);
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
            VersionState *init_state = create_new_version_state(init_version);
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
          node_info.field_versions->add_field_version(init_version,unversioned);
          VersionStateInfo &info = current_version_infos[init_version];
          VersionState *init_state = create_new_version_state(init_version);
          init_state->add_base_valid_ref(CURRENT_STATE_REF);
          info.states[init_state] = unversioned;
          info.valid_fields |= unversioned;
          state->add_version_state(init_state, unversioned);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void CurrentState::advance_version_numbers(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
        VersionState *new_state = create_new_version_state(next_version);
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
#ifdef DEBUG_HIGH_LEVEL
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
        VersionState *new_state = create_new_version_state(init_version);
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
#ifdef DEBUG_HIGH_LEVEL
          assert(current_version_infos.find(it->first->version_number) ==
                 current_version_infos.end());
#endif
          VersionStateInfo &info = 
            current_version_infos[it->first->version_number];
          info.states[it->first] = it->second;
          info.valid_fields = it->second;
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
    } 

    //--------------------------------------------------------------------------
    VersionState* CurrentState::create_new_version_state(VersionID vid)
    //--------------------------------------------------------------------------
    {
      DistributedID new_did = 
        owner->context->runtime->get_available_distributed_id(false);
      AddressSpace local_space = owner->context->runtime->address_space;
      return legion_new<VersionState>(vid, owner->context->runtime, 
                        new_did, local_space, local_space, this);
    }

    //--------------------------------------------------------------------------
    VersionState* CurrentState::create_remote_version_state(VersionID vid,
                                  DistributedID did, AddressSpaceID owner_space)
    //--------------------------------------------------------------------------
    {
      AddressSpace local_space = owner->context->runtime->address_space;
#ifdef DEBUG_HIGH_LEVEL
      assert(owner_space != local_space);
#endif
      return legion_new<VersionState>(vid, owner->context->runtime, 
                              did, owner_space, local_space, this);
    }

    //--------------------------------------------------------------------------
    VersionState* CurrentState::find_remote_version_state(VersionID vid,
                                  DistributedID did, AddressSpaceID owner_space)
    //--------------------------------------------------------------------------
    {
      // Use the lock on the version manager to ensure that we don't
      // replicated version states on a node
      VersionState *result = NULL;
      Internal *runtime = owner->context->runtime;
      {
        AutoLock s_lock(state_lock);
        if (runtime->has_distributed_collectable(did))
        {
          DistributedCollectable *dc = 
            runtime->find_distributed_collectable(did);
#ifdef DEBUG_HIGH_LEVEL
          result = dynamic_cast<VersionState*>(dc);
          assert(result != NULL);
#else
          result = static_cast<VersionState*>(dc);
#endif
        }
        else // Otherwise make it
          result = create_remote_version_state(vid, did, owner_space);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void CurrentState::add_persistent_view(MaterializedView *view)
    //--------------------------------------------------------------------------
    {
      view->add_base_valid_ref(PERSISTENCE_REF);
      // First see if we need to make the lock
      bool remove_extra = false;
      {
        AutoLock s_lock(state_lock);
        if (persistent_views.find(view) == persistent_views.end())
          persistent_views.insert(view);
        else
          remove_extra = true;
        has_persistent = true;
      }
      if (remove_extra)
        view->remove_base_valid_ref(PERSISTENCE_REF);
    }

    //--------------------------------------------------------------------------
    void CurrentState::remove_persistent_view(MaterializedView *view)
    //--------------------------------------------------------------------------
    {
      bool remove_reference = false;
      {
        AutoLock s_lock(state_lock);
        std::set<MaterializedView*>::iterator finder = 
          persistent_views.find(view);
        if (finder != persistent_views.end())
        {
          persistent_views.erase(finder);
          remove_reference = true;
        }
      }
      if (remove_reference && view->remove_base_valid_ref(PERSISTENCE_REF))
        legion_delete(view);
    }

    //--------------------------------------------------------------------------
    void CurrentState::capture_persistent_views(
                        LegionMap<LogicalView*,FieldMask>::aligned &valid_views,
                                                  const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask empty_mask;
      // If we are here then we know the lock exists
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      for (std::set<MaterializedView*>::const_iterator it = 
            persistent_views.begin(); it != persistent_views.end(); it++)
      {
        if ((*it)->has_space(capture_mask))
        {
          LegionMap<LogicalView*,FieldMask>::aligned::const_iterator finder = 
            valid_views.find(*it);
          // Only need to add it if it is not already there
          if (finder == valid_views.end())
            valid_views[*it] = empty_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CurrentState::print_physical_state(RegionTreeNode *node,
                                            const FieldMask &capture_mask,
                          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                            TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      PhysicalState temp_state(this, node);
#else
      PhysicalState temp_state(this);
#endif
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == rhs.redop);
#endif
      if (redop > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
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
    CopyTracker::CopyTracker(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Event CopyTracker::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      return Event::merge_events(copy_events);
    }

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
      leave_open_versions.release();
      leave_open_versions.clear();
      force_close_versions.release();
      force_close_versions.clear();
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
#ifdef DEBUG_HIGH_LEVEL
      // should never be both, but can be neither
      assert(!(leave_open && read_only));
#endif
      closed_mask |= mask;
      // IMPORTANT: Always do this even if we don't have any closed users
      // They could have been pruned out because they finished executing, but
      // we still need to do the close operation.
      if (leave_open)
      {
        leave_open_mask |= mask;
        LegionMap<ColorPoint,ClosingInfo>::aligned::iterator finder = 
                                              leave_open_children.find(child);
        if (finder != leave_open_children.end())
        {
          finder->second.child_fields |= mask;
          finder->second.child_users.insert(finder->second.child_users.end(),
                                      closed_users.begin(), closed_users.end());
        }
        else
          leave_open_children[child] = ClosingInfo(mask, closed_users);
      }
      else if (read_only)
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
        LegionMap<ColorPoint,ClosingInfo>::aligned::iterator finder = 
                                              force_close_children.find(child);
        if (finder != force_close_children.end())
        {
          finder->second.child_fields |= mask;
          finder->second.child_users.insert(finder->second.child_users.end(),
                                      closed_users.begin(), closed_users.end());
        }
        else
          force_close_children[child] = ClosingInfo(mask, closed_users);
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
#ifdef DEBUG_HIGH_LEVEL
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
      if (!leave_open_children.empty())
      {
        LegionList<ClosingSet>::aligned leave_open;
        compute_close_sets(leave_open_children, leave_open);
        create_close_operations(target, creator, leave_open_versions, ver_info,
                         res_info, trace_info, true/*leave open*/, leave_open);
      }
      if (!read_only_children.empty())
      {
        LegionList<ClosingSet>::aligned read_only;
        compute_close_sets(read_only_children, read_only);
        create_read_only_close_operations(target, creator, 
                                          trace_info, read_only);
      }
      if (!force_close_children.empty())
      {
        LegionList<ClosingSet>::aligned force_close;
        compute_close_sets(force_close_children, force_close);
        create_close_operations(target, creator, force_close_versions, ver_info,
                        res_info, trace_info, false/*leave open*/, force_close);
      }
      // Finally if we have any fields which are flush only
      // make a close operation for them and add it to force close
      if (!!flush_only_fields)
      {
        std::set<ColorPoint> empty_children;
        InterCloseOp *flush_op = target->create_close_op(creator,
                                                         flush_only_fields,
                                                         false/*leave open*/,
                                                         empty_children,
                                                         force_close_versions,
                                                         ver_info, res_info,
                                                         trace_info);
        force_close_closes[flush_op] = LogicalUser(flush_op, 0/*idx*/,
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
            leave_open_closes.begin(); it != leave_open_closes.end(); it++)
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
        for (LegionList<ClosingSet>::aligned::iterator it = 
              close_sets.begin(); it != close_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->closing_mask)
          {
            it->children.insert(cit->first);
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
            last.children.insert(cit->first);
            inserted = true;
            break;
          }
          // We dominate the existing set
          if (overlap == it->closing_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->children.insert(cit->first);
            remaining -= overlap;
            continue;
          }
          // Hard case, neither dominates, compute
          // three distinct sets of fields, keep left
          // one in place and reduce scope, add a new
          // one at the end for overlap, continue
          // iterating for the right one
          it->closing_mask -= overlap;
          const std::set<ColorPoint> &temp_children = it->children;
          it = close_sets.insert(it, ClosingSet(overlap));
          it->children = temp_children;
          it->children.insert(cit->first);
          remaining -= overlap;
          continue;
        }
        // If we didn't add it yet, add it now
        if (!inserted)
        {
          close_sets.push_back(ClosingSet(remaining));
          ClosingSet &last = close_sets.back();
          last.children.insert(cit->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::create_close_operations(RegionTreeNode *target,
                              Operation *creator, const VersionInfo &local_info,
                              const VersionInfo &version_info,
                              const RestrictInfo &restrict_info, 
                              const TraceInfo &trace_info, bool leave_open,
                              const LegionList<ClosingSet>::aligned &close_sets)
    //--------------------------------------------------------------------------
    {
      for (LegionList<ClosingSet>::aligned::const_iterator it = 
            close_sets.begin(); it != close_sets.end(); it++)
      {
        InterCloseOp *close_op = target->create_close_op(creator, 
                                                       it->closing_mask,
                                                       leave_open, it->children,
                                                       local_info,
                                                       version_info,
                                                       restrict_info, 
                                                       trace_info);
        if (leave_open)
          leave_open_closes[close_op] = LogicalUser(close_op, 0/*idx*/,
                        RegionUsage(close_op->get_region_requirement()),
                        it->closing_mask);
        else
          force_close_closes[close_op] = LogicalUser(close_op, 0/*idx*/,
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
      if (!leave_open_closes.empty())
        register_dependences(current, open_below, leave_open_closes, 
                             leave_open_children, above_users, cusers, pusers);
      if (!read_only_closes.empty())
        register_dependences(current, open_below, read_only_closes,
                             read_only_children, above_users, cusers, pusers);
      if (!force_close_closes.empty())
        register_dependences(current, open_below, force_close_closes, 
                             force_close_children, above_users, cusers, pusers);
    }

    // If you are looking for LogicalCloser::register_dependences it can 
    // be found in region_tree.cc to make sure that templates are instantiated

    //--------------------------------------------------------------------------
    void LogicalCloser::update_state(CurrentState &state)
    //--------------------------------------------------------------------------
    {
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
        }
      }
      else
      {
        // All the fields were fully closed, so we can clear dirty below
        state.dirty_below -= closed_mask;
        if (!!state.partially_closed)
          state.partially_closed -= closed_mask;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::register_close_operations(
               LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users)
    //--------------------------------------------------------------------------
    {
      // Add our close operations onto the list
      // Note we already added our mapping references when we made them
      if (!leave_open_closes.empty())
      {
        for (LegionMap<TraceCloseOp*,LogicalUser>::aligned::const_iterator it =
              leave_open_closes.begin(); it != leave_open_closes.end(); it++)
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
      if (!force_close_closes.empty())
      {
        for (LegionMap<TraceCloseOp*,LogicalUser>::aligned::const_iterator it =
              force_close_closes.begin(); it != force_close_closes.end(); it++)
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
#ifdef DEBUG_HIGH_LEVEL
      assert(!!local_mask);
#endif
      // Don't need the previous because writes were already done in the
      // sub-tree we are closing so the version number for the target
      // region has already been advanced.
      if (leave_open)
        state.record_version_numbers(local_mask, user, leave_open_versions, 
                                     false/*previous*/, 
                                     false/*path only*/, true/*final*/,
                                     false/*close top*/, false/*report*/);
      else
        state.record_version_numbers(local_mask, user, force_close_versions, 
                                     false/*previous*/,
                                     false/*path only*/, true/*final*/,
                                     false/*close top*/, false/*report*/);
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_top_version_numbers(RegionTreeNode *node,
                                                   CurrentState &state)
    //--------------------------------------------------------------------------
    {
      // If we have any flush only fields, see if we need to bump their
      // version numbers before generating our close operations
      if (!!flush_only_fields)
      {
        FieldMask update_mask = flush_only_fields - state.dirty_below;
        if (!!update_mask)
          state.advance_version_numbers(update_mask);
      }
      // We don't need to advance the version numbers because we know
      // that was already done when we 
      if (!!leave_open_mask)
      {
        // See if we have any partial closes for these fields, if there
        // are partially closed fields we don't need to capture from 
        // the previous version number because we've already done that
        if (!state.partially_closed || (closed_mask * state.partially_closed))
        {
          // Common case, there are no partially closed fields
          state.record_version_numbers(leave_open_mask, user,
                                       leave_open_versions,
                                       true/*previous*/, 
                                       false/*path only*/, true/*final*/,
                                       true/*close top*/, false/*report*/);
          FieldMask force_close_mask = closed_mask - leave_open_mask;
          if (!!force_close_mask)
            state.record_version_numbers(force_close_mask, user,
                                         force_close_versions, true/*previous*/,
                                         false/*path only*/, true/*final*/, 
                                         true/*close top*/, false/*report*/);
        }
        else
        {
          // Handle partially closed fields
          FieldMask partial_open = leave_open_mask & state.partially_closed;
          if (!!partial_open)
          {
            state.record_version_numbers(partial_open, user,
                                         leave_open_versions,
                                         false/*previous*/,
                                         false/*path only*/, true/*final*/,
                                         true/*close top*/, false/*report*/);
            FieldMask non_partial_open = leave_open_mask - partial_open;
            if (!!non_partial_open)
              state.record_version_numbers(non_partial_open, user,
                                           leave_open_versions,
                                           true/*previous*/, 
                                           false/*path only*/, true/*final*/,
                                           true/*close top*/, false/*report*/);
          }
          else
          {
            // No partial fields for leave open
            state.record_version_numbers(leave_open_mask, user,
                                         leave_open_versions,
                                         true/*previous*/, 
                                         false/*path only*/, true/*final*/,
                                         true/*close top*/, false/*report*/);
          }
          FieldMask force_close_mask = closed_mask - leave_open_mask;
          if (!!force_close_mask)
          {
            FieldMask partial_close = force_close_mask & state.partially_closed;
            if (!!partial_close)
            {
              state.record_version_numbers(partial_close, user,
                                        force_close_versions, false/*previous*/,
                                        false/*path only*/,true/*final*/,
                                        true/*close top*/, false/*report*/);
              FieldMask non_partial_close = force_close_mask - partial_close;
              if (!!non_partial_close)
                state.record_version_numbers(non_partial_close, user,
                                        force_close_versions, true/*previous*/,
                                        false/*path only*/, true/*final*/,
                                        true/*close top*/, false/*report*/);
            }
            else
            {
              state.record_version_numbers(force_close_mask, user,
                                         force_close_versions, true/*previous*/,
                                         false/*path only*/, true/*final*/, 
                                         true/*close top*/, false/*report*/);
            }
          }
        }
      }
      else
      {
        // Normal case is relatively simple
        // See if there are any partially closed fields
        FieldMask partial_close = closed_mask & state.partially_closed;
        if (!partial_close)
        {
          // Common case: there are no partially closed fields
          state.record_version_numbers(closed_mask, user, force_close_versions,
                                       true/*previous*/, false/*path only*/, 
                                       true/*final*/, true/*close top*/,
                                       false/*report*/);
        }
        else
        {
          // Record the partially closed fields from this version
          state.record_version_numbers(partial_close, user, 
                                       force_close_versions, false/*previous*/,
                                       false/*path only*/, true/*final*/,
                                       true/*close top*/, false/*report*/);
          FieldMask non_partial = closed_mask - partial_close;
          if (!!non_partial)
            state.record_version_numbers(non_partial, user,
                                         force_close_versions, true/*previous*/,
                                         false/*path only*/, true/*final*/,
                                         true/*close top*/, false/*report*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::merge_version_info(VersionInfo &target,
                                           const FieldMask &merge_mask)
    //--------------------------------------------------------------------------
    {
      target.merge(leave_open_versions, merge_mask);
      target.merge(force_close_versions, merge_mask);
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const MappableInfo &in, 
                                   bool open, LogicalRegion h)
      : info(in), handle(h), permit_leave_open(open), targets_selected(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalCloser &rhs)
      : info(rhs.info), handle(rhs.handle), 
        permit_leave_open(rhs.permit_leave_open),
        upper_targets(rhs.get_lower_targets())
    //--------------------------------------------------------------------------
    {
      targets_selected = !upper_targets.empty(); 
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
    bool PhysicalCloser::needs_targets(void) const
    //--------------------------------------------------------------------------
    {
      return !targets_selected;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::add_target(MaterializedView *target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(target != NULL);
#endif
      upper_targets.push_back(target);
      targets_selected = true;
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
    const std::vector<MaterializedView*>& PhysicalCloser::
                                                  get_upper_targets(void) const
    //--------------------------------------------------------------------------
    {
      return upper_targets;
    }

    //--------------------------------------------------------------------------
    const std::vector<MaterializedView*>& PhysicalCloser::
                                                  get_lower_targets(void) const
    //--------------------------------------------------------------------------
    {
      return lower_targets;
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
      node->update_valid_views(state, info.traversal_mask,
                               dirty_mask, upper_targets);
    } 

    //--------------------------------------------------------------------------
    CompositeCloser::CompositeCloser(ContextID c, VersionInfo &info, bool open)
      : ctx(c), permit_leave_open(open), version_info(info)
    //--------------------------------------------------------------------------
    {
      composite_version_info = new CompositeVersionInfo();
    }

    //--------------------------------------------------------------------------
    CompositeCloser::CompositeCloser(const CompositeCloser &rhs)
      : ctx(0), permit_leave_open(false), version_info(rhs.version_info)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeCloser::~CompositeCloser(void)
    //--------------------------------------------------------------------------
    {
      // Only delete the version info if there are no constructed nodes
      if (constructed_nodes.empty())
        delete composite_version_info;
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
                                                       CompositeNode *parent)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeNode*,CompositeNode*>::const_iterator finder = 
        constructed_nodes.find(node);
      if (finder != constructed_nodes.end())
        return finder->second;
      CompositeNode *result = legion_new<CompositeNode>(node, parent,
                                                        composite_version_info);
      constructed_nodes[node] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    CompositeRef CompositeCloser::create_valid_view(PhysicalState *state,
                                                    CompositeNode *root,
                                                   const FieldMask &closed_mask,
                                                    bool register_view)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *node = root->logical_node;
      DistributedID did = 
                    node->context->runtime->get_available_distributed_id(false);
      CompositeView *composite_view = legion_new<CompositeView>(node->context, 
                                   did, node->context->runtime->address_space,
                                   node, node->context->runtime->address_space, 
                                   closed_mask, true/*register now*/);
      // Set the root value
      composite_view->add_root(root, closed_mask, true/*top*/);
      // Fill in the version infos
      VersionInfo &target_version_info = 
                                composite_version_info->get_version_info();
      target_version_info.clone_from(version_info);
      if (!reduction_views.empty())
      {
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
              reduction_views.begin(); it != reduction_views.end(); it++)
        {
          composite_view->update_reduction_views(it->first, it->second);
        }
      }
      // Now update the state of the node
      // Note that if we are permitted to leave the subregions
      // open then we don't make the view dirty
      if (register_view)
      {
        node->update_valid_views(state, closed_mask,
                                 true/*dirty*/, composite_view);
        // return an empty composite ref since it won't be used
        return CompositeRef(); 
      }
      return CompositeRef(composite_view);
    }

    //--------------------------------------------------------------------------
    void CompositeCloser::capture_physical_state(CompositeNode *target,
                                                 RegionTreeNode *node,
                                                 PhysicalState *state,
                                                 const FieldMask &capture_mask,
                                                 FieldMask &dirty_mask)
    //--------------------------------------------------------------------------
    {
      // Do the capture and then update capture mask
      target->capture_physical_state(node, state, capture_mask, *this,
                                     dirty_mask, state->dirty_mask,
                                     state->valid_views);
      // Capture any reduction views
      FieldMask reduction_capture = capture_mask & state->reduction_mask;
      if (!!reduction_capture)
      {
        dirty_mask |= reduction_capture;
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
              state->reduction_views.begin(); it != 
              state->reduction_views.end(); it++)
        {
          FieldMask overlap = capture_mask & it->second;
          if (!overlap)
            continue;
          LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
            reduction_views.find(it->first);
          if (finder == reduction_views.end())
            reduction_views[it->first] = overlap;
          else
            finder->second |= overlap;
        }
      }
      // Record that we've captured the fields for this node
      update_capture_mask(node, capture_mask);
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
    PhysicalState::PhysicalState(CurrentState *m)
      : manager(m)
#ifdef DEBUG_HIGH_LEVEL
        , node(NULL)
#endif
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(false); // shouldn't be calling this constructor in debug mode
#endif
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(CurrentState *m, RegionTreeNode *n)
      : manager(m), node(n)
    //--------------------------------------------------------------------------
    {
    }
#endif

    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(const PhysicalState &rhs)
      : manager(NULL)
#ifdef DEBUG_HIGH_LEVEL
        , node(NULL)
#endif
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalState::~PhysicalState(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(created_instances.empty());
#endif
      // Remove references to our version states and delete them if necessary
      for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator vit =
            version_states.begin(); vit != version_states.end(); vit++)
      {
        const VersionStateInfo &info = vit->second;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              info.states.begin(); it != info.states.end(); it++)
        {
          if (it->first->remove_base_gc_ref(PHYSICAL_STATE_REF)) 
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
          if (it->first->remove_base_gc_ref(PHYSICAL_STATE_REF))
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
        state->add_base_gc_ref(PHYSICAL_STATE_REF);
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
        state->add_base_gc_ref(PHYSICAL_STATE_REF);
        info.states[state] = state_mask;
      }
      else
        finder->second |= state_mask;
      info.valid_fields |= state_mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalState::capture_state(bool path_only, bool close_top)
    //--------------------------------------------------------------------------
    {
      if (close_top)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!path_only);
#endif
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
            it->first->update_close_top_state(this, it->second);
          }
        }
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator
              vit = advance_states.begin(); vit != advance_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                info.states.begin(); it != info.states.end(); it++)
          {
            it->first->update_open_children_state(this, it->second);
          }
        }
      }
      else if (path_only)
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
               AddressSpaceID target, std::set<Event> &applied_conditions) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(created_instances.empty());
#endif
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
                     AddressSpaceID target, std::set<Event> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      if (!advance_states.empty())
      {
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
      // If we have any created instances, we can now remove our
      // valid references on them because we've applied all our updates
      if (!created_instances.empty())
      {
        for (std::deque<InstanceView*>::const_iterator it = 
              created_instances.begin(); it != created_instances.end(); it++)
        {
          if ((*it)->remove_base_valid_ref(INITIAL_CREATION_REF))
            LogicalView::delete_logical_view(*it);
        }
        created_instances.clear();
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::filter_and_apply(bool top, bool filter_children,
                     AddressSpaceID target, std::set<Event> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      if (top)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!advance_states.empty());
#endif
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = advance_states.begin(); vit != 
              advance_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->filter_and_merge_physical_state(this, it->second, 
                         true, filter_children, target, applied_conditions);
          }
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(advance_states.empty());
#endif
        for (LegionMap<VersionID,VersionStateInfo>::aligned::const_iterator 
              vit = version_states.begin(); vit != 
              version_states.end(); vit++)
        {
          const VersionStateInfo &info = vit->second;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = info.states.begin(); it != info.states.end(); it++)
          {
            it->first->filter_and_merge_physical_state(this, it->second, 
                         false, filter_children, target, applied_conditions);
          }
        }
      }
      // If we have any created instances, we can now remove our
      // valid references on them because we've applied all our updates
      if (!created_instances.empty())
      {
        for (std::deque<InstanceView*>::const_iterator it = 
              created_instances.begin(); it != created_instances.end(); it++)
        {
          if ((*it)->remove_base_valid_ref(INITIAL_CREATION_REF))
            LogicalView::delete_logical_view(*it);
        }
        created_instances.clear();
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
    void PhysicalState::record_created_instance(InstanceView *view)
    //--------------------------------------------------------------------------
    {
      view->add_base_valid_ref(INITIAL_CREATION_REF); 
      created_instances.push_back(view);
    }

    //--------------------------------------------------------------------------
    PhysicalState* PhysicalState::clone(bool capture_state, bool need_adv) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      PhysicalState *result = legion_new<PhysicalState>(manager, node);
#else
      PhysicalState *result = legion_new<PhysicalState>(manager);
#endif
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
#ifdef DEBUG_HIGH_LEVEL
      PhysicalState *result = legion_new<PhysicalState>(manager, node);
#else
      PhysicalState *result = legion_new<PhysicalState>(manager);
#endif
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
    void PhysicalState::make_local(std::set<Event> &preconditions, 
                                   bool needs_final, bool needs_advance)
    //--------------------------------------------------------------------------
    {
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
#ifdef DEBUG_HIGH_LEVEL
          assert(it->first->as_instance_view()->is_materialized_view());
#endif
          MaterializedView *current = 
            it->first->as_instance_view()->as_materialized_view();
          char *valid_mask = overlap.to_string();
          logger->log("Instance " IDFMT "   Memory " IDFMT "   Mask %s",
                      current->manager->get_instance().id, 
                      current->manager->memory.id, valid_mask);
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
                      it->first->manager->memory.id, valid_mask);
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
    VersionState::VersionState(VersionID vid, Internal *rt, DistributedID id,
                               AddressSpaceID own_sp, AddressSpaceID local_sp, 
                               CurrentState *man)
      : DistributedCollectable(rt, id, own_sp, local_sp), version_number(vid), 
        manager(man), state_lock(Reservation::create_reservation())
#ifdef DEBUG_HIGH_LEVEL
        , currently_active(true), currently_valid(true)
#endif
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, add a valid and resource reference
      if (!is_owner())
      {
        add_base_valid_ref(REMOTE_DID_REF);
        add_base_gc_ref(REMOTE_DID_REF);
        add_base_resource_ref(REMOTE_DID_REF);
      }
    }

    //--------------------------------------------------------------------------
    VersionState::VersionState(const VersionState &rhs)
      : DistributedCollectable(rhs), version_number(rhs.version_number),
        manager(NULL)
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
#ifdef DEBUG_HIGH_LEVEL
      assert(!currently_valid);
#endif 
      // If we are the owner, then remote resource 
      // references on our remote instances 
      if (is_owner())
      {
        // If we're the owner, remove our valid references on remote nodes
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> functor(this); 
        map_over_remote_instances(functor);
      }
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
    void VersionState::initialize(LogicalView *new_view, Event term_event,
                                  const RegionUsage &usage,
                                  const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner());
#endif
      new_view->add_nested_gc_ref(did);
      new_view->add_nested_valid_ref(did);
      if (new_view->is_instance_view())
      {
        InstanceView *inst_view = new_view->as_instance_view();
        if (inst_view->is_reduction_view())
        {
          ReductionView *view = inst_view->as_reduction_view();
          LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
            track_aligned::iterator finder = reduction_views.find(view); 
          if (finder == reduction_views.end())
            reduction_views[view] = user_mask;
          else
            finder->second |= user_mask;
          reduction_mask |= user_mask;
          inst_view->add_initial_user(term_event, usage, user_mask);
        }
        else
        {
          LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
            track_aligned::iterator finder = valid_views.find(new_view);
          if (finder == valid_views.end())
            valid_views[new_view] = user_mask;
          else
            finder->second |= user_mask;
          if (HAS_WRITE(usage))
            dirty_mask |= user_mask;
          inst_view->add_initial_user(term_event, usage, user_mask);
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!term_event.exists());
#endif
        LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
            track_aligned::iterator finder = valid_views.find(new_view);
        if (finder == valid_views.end())
          valid_views[new_view] = user_mask;
        else
          finder->second |= user_mask;
        if (HAS_WRITE(usage))
          dirty_mask |= user_mask;
        // Don't add a user since this is a deferred view and
        // we can't access it anyway
      }
      // Update our field information, we know we are the owner
      initial_nodes[local_space] |= user_mask;
      initial_fields |= user_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::update_close_top_state(PhysicalState *state,
                                             const FieldMask &update_mask) const
    //--------------------------------------------------------------------------
    {
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
    void VersionState::update_open_children_state(PhysicalState *state,
                                             const FieldMask &update_mask) const
    //--------------------------------------------------------------------------
    {
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
                                            std::set<Event> &applied_conditions)
    //--------------------------------------------------------------------------
    {
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
              UserEvent registered_event = UserEvent::create_user_event();
              rez.serialize(registered_event);
              applied_conditions.insert(registered_event);
            }
            else
              rez.serialize(UserEvent::NO_USER_EVENT);
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
                                            std::set<Event> &applied_conditions,
                                            bool need_lock /* = true*/)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        // We're writing so we need the lock in exclusive mode
        Event acquire_event = state_lock.acquire();
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
          it->first->add_nested_gc_ref(did);
          it->first->add_nested_valid_ref(did);
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
            it->first->add_nested_gc_ref(did);
            it->first->add_nested_valid_ref(did);
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
              UserEvent registered_event = UserEvent::create_user_event();
              rez.serialize(registered_event);
              applied_conditions.insert(registered_event);
            }
            else
              rez.serialize(UserEvent::NO_USER_EVENT);
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
                        bool top, bool filter_children, AddressSpaceID target,
                        std::set<Event> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      // We're writing so we need the lock in exclusive mode
      AutoLock s_lock(state_lock);
#ifdef DEBUG_HIGH_LEVEL
      // If we are the top we should not be in final mode
      if (top)
        assert(merge_mask * final_fields);
#endif
      // Do the filtering first
      if (!top)
      {
        dirty_mask -= merge_mask;
        reduction_mask -= merge_mask;
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
              (*it)->remove_nested_valid_ref(did);
              if ((*it)->remove_nested_gc_ref(did))
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
              (*it)->remove_nested_valid_ref(did);
              if ((*it)->remove_nested_gc_ref(did))
                legion_delete(*it);
            }
          }
        }
      }
      if (filter_children)
      {
        children.valid_fields -= merge_mask;  
        if (!children.valid_fields)
          children.open_children.clear();
        else
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
            {
              children.open_children.erase(*it);
            }
          }
        }
      }
      // Now we can do the merge
      merge_physical_state(state, merge_mask, 
                           target, applied_conditions, false/*need lock*/);
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_active(void)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(currently_active); // should be monotonic
#endif
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_nested_gc_ref(did);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        it->first->add_nested_gc_ref(did);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_inactive(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing we only care about valid references
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(currently_active);
      currently_active = false;
#endif
      // When we are no longer valid, remove all valid references to version
      // state objects on remote nodes. 
      // No need to hold the lock since no one else should be accessing us
      if (is_owner() && !remote_instances.empty())
      {
        UpdateReferenceFunctor<GC_REF_KIND,false/*add*/> functor(this);
        map_over_remote_instances(functor);
      }
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->remove_nested_gc_ref(did);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        it->first->remove_nested_gc_ref(did);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(currently_valid); // should be monotonic
#endif
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_nested_valid_ref(did);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        it->first->add_nested_valid_ref(did);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(currently_valid);
      currently_valid = false;
#endif
      // When we are no longer valid, remove all valid references to version
      // state objects on remote nodes. 
      // No need to hold the lock since no one else should be accessing us
      if (is_owner() && !remote_instances.empty())
      {
        // If we're the owner, remove our valid references on remote nodes
        UpdateReferenceFunctor<VALID_REF_KIND,false/*add*/> functor(this); 
        map_over_remote_instances(functor);
      }
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->remove_nested_valid_ref(did);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        it->first->remove_nested_valid_ref(did);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::request_initial_version_state(
                  const FieldMask &request_mask, std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      UserEvent ready_event = UserEvent::NO_USER_EVENT;
      FieldMask remaining_mask = request_mask;
      LegionDeque<RequestInfo>::aligned targets;
      {
        AutoLock s_lock(state_lock);
        // Check to see which fields we already have initial events for
        if (!(remaining_mask * initial_fields))
        {
          if (!initial_events.empty())
          {
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
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
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
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
          ready_event = UserEvent::create_user_event();
          initial_events[ready_event] = remaining_mask;
          initial_fields |= remaining_mask;
          preconditions.insert(ready_event);
        }
      }
      if (!is_owner())
      {
#ifdef DEBUG_HIGH_LEVEL
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
                                                 std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      UserEvent ready_event;
      FieldMask remaining_mask = req_mask;
      LegionDeque<RequestInfo>::aligned targets;
      {
        AutoLock s_lock(state_lock);
        if (!(remaining_mask * final_fields))
        {
          if (!final_events.empty())
          {
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
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
#ifdef DEBUG_HIGH_LEVEL
          assert(!!remaining_mask);
#endif
          // Make a user event and record it as a precondition   
          ready_event = UserEvent::create_user_event();
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
                                              std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the lock
#ifdef DEBUG_HIGH_LEVEL
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
        info.to_trigger = UserEvent::create_user_event();
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
        info.to_trigger = UserEvent::create_user_event();
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
                                            std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the lock
#ifdef DEBUG_HIGH_LEVEL
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
        info.to_trigger = UserEvent::create_user_event();
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
      std::set<Event> merge_preconditions;
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
        info.to_trigger = UserEvent::create_user_event();
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
          info.to_trigger = UserEvent::create_user_event();
          info.request_mask = overlap;
          info.kind = PATH_ONLY_VERSION_REQUEST;
          merge_preconditions.insert(info.to_trigger); 
          requested_mask |= overlap;
        }
        needed_mask -= requested_mask;
      }
      if (!!requested_mask)
      {
        Event precondition = Event::merge_events(merge_preconditions);
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
                                          UserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
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
            DistributedID view_did = it->first->send_view(target, it->second);
            rez.serialize(view_did);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(reduction_views.size());
          for (LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
                track_aligned::const_iterator it = reduction_views.begin(); 
                it != reduction_views.end(); it++)
          {
            DistributedID reduc_did = it->first->send_view(target, it->second);
            rez.serialize(reduc_did);
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
              DistributedID view_did = it->first->send_view(target, overlap);
              valid_rez.serialize(view_did);
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
              DistributedID reduc_did = it->first->send_view(target, overlap);
              reduc_rez.serialize(reduc_did);
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
                                    AddressSpaceID source, UserEvent to_trigger, 
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
                                                 UserEvent to_trigger, 
                                                 VersionRequestKind req_kind,
                                                 const FieldMask &request_mask, 
                                                 Event precondition)
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
                                       NULL/*op*/, precondition);
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_path_only(AddressSpaceID source,
                                                      FieldMask &path_only_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner());
#endif
      // First transform the initial mask
      manager->owner->column_source->transform_field_mask(path_only_mask, 
                                                          source);
      AutoLock s_lock(state_lock);
      path_only_nodes[source] |= path_only_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_initialization(
                                 AddressSpaceID source, FieldMask &initial_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner());
#endif
      // First transform the initial mask
      manager->owner->column_source->transform_field_mask(initial_mask, source);
      AutoLock s_lock(state_lock);
      initial_nodes[source] |= initial_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_request(AddressSpaceID source,
                    UserEvent to_trigger, VersionRequestKind request_kind, 
                                                    FieldMask &request_mask)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // First things first, transform the field mask
        manager->owner->column_source->transform_field_mask(request_mask, 
                                                            owner_space);
        // If we are not the owner, we should definitely be able to handle this
        std::set<Event> launch_preconditions;
#ifdef DEBUG_HIGH_LEVEL
        FieldMask remaining_mask = request_mask;
#endif
        if (request_kind == FINAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock,1,false/*exclusive*/);
          for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
                final_events.begin(); it != final_events.end(); it++)
          {
            if (it->second * request_mask)
              continue;
            launch_preconditions.insert(it->first);
          }
#ifdef DEBUG_HIGH_LEVEL
          remaining_mask -= final_fields;
#endif
        }
        else if (request_kind == INITIAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock,1,false/*exclusive*/);
          for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
                initial_events.begin(); it != initial_events.end(); it++)
          {
            if (it->second * request_mask)
              continue;
            launch_preconditions.insert(it->first);
          }
#ifdef DEBUG_HIGH_LEVEL
          remaining_mask -= initial_fields;
#endif
        }
#ifdef DEBUG_HIGH_LEVEL
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
          Event pre = Event::merge_events(launch_preconditions);
          launch_send_version_state(source, to_trigger, request_kind, 
                                    request_mask, pre);
        }
        else
          launch_send_version_state(source, to_trigger, 
                                    request_kind, request_mask);
      }
      else
      {
        // First things first, transform the field mask
        manager->owner->column_source->transform_field_mask(request_mask, 
                                                            source);
        // We're the owner, figure out what to do
        FieldMask remaining_fields = request_mask;
        FieldMask local_fields;
        int path_only_local_index = -1;
        int initial_local_index = -1;
        std::set<Event> local_preconditions, done_conditions;
        LegionDeque<RequestInfo>::aligned targets;
        if (request_kind == FINAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock);
          // See if we can handle any of the fields locally
          local_fields = remaining_fields & final_fields;
          if (!!local_fields)
          {
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
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
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
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
#ifdef DEBUG_HIGH_LEVEL
                assert(it->kind == PATH_ONLY_VERSION_REQUEST);
#endif
                path_only_local_index = idx;
              }
              else
              {
#ifdef DEBUG_HIGH_LEVEL
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
          UserEvent local_trigger = UserEvent::create_user_event();
          if (!local_preconditions.empty())
          {
            Event pre = Event::merge_events(local_preconditions);
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
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
                  initial_events.begin(); it != initial_events.end(); it++)
            {
              if (it->second * info.request_mask)
                continue;
              local_preconditions.insert(it->first);
            }
          }
          if (!local_preconditions.empty())
          {
            Event pre = Event::merge_events(local_preconditions);
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
          to_trigger.trigger(Event::merge_events(done_conditions));
        else
          to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_response(AddressSpaceID source,
     UserEvent to_trigger, VersionRequestKind request_kind, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *owner_node = manager->owner;
      FieldSpaceNode *field_node = owner_node->column_source;
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
              field_node->transform_field_mask(child_update, source);
              finder->second |= child_update;
              children.valid_fields |= child_update;
            }
            else
            {
              FieldMask &mask = children.open_children[child];
              derez.deserialize(mask);
              field_node->transform_field_mask(mask, source);
              children.valid_fields |= mask;
            }
          }
        }
        to_trigger.trigger();
        return;
      }
      // Keep track of any composite veiws we need to check 
      // for having recursive version states at here
      std::vector<CompositeView*> composite_views;
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
          field_node->transform_field_mask(dirty_mask, source); 
          derez.deserialize(reduction_mask);
          field_node->transform_field_mask(reduction_mask, source);
        }
        else
        {
          FieldMask dirty_update;
          derez.deserialize(dirty_update);
          field_node->transform_field_mask(dirty_update, source);
          dirty_mask |= dirty_update;

          FieldMask reduction_update;
          derez.deserialize(reduction_update);
          field_node->transform_field_mask(reduction_update, source);
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
            field_node->transform_field_mask(mask, source);
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
              field_node->transform_field_mask(child_update, source);
              finder->second |= child_update;
              children.valid_fields |= child_update;
            }
            else
            {
              FieldMask &mask = children.open_children[child];
              derez.deserialize(mask);
              field_node->transform_field_mask(mask, source);
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
            DistributedID did;
            derez.deserialize(did);
            LogicalView *view = owner_node->find_view(did);
            // Check for composite view
            if (view->is_deferred_view())
            {
              DeferredView *def_view = view->as_deferred_view();
              if (def_view->is_composite_view())
                composite_views.push_back(def_view->as_composite_view());
            }
            FieldMask &mask = valid_views[view];
            derez.deserialize(mask);
            field_node->transform_field_mask(mask, source);
            view->add_nested_gc_ref(did);
            view->add_nested_valid_ref(did);
          }
          size_t num_reduction_views;
          derez.deserialize(num_reduction_views);
          for (unsigned idx = 0; idx < num_reduction_views; idx++)
          {
            DistributedID did;
            derez.deserialize(did);
            LogicalView *view = owner_node->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
            assert(view->is_instance_view());
            assert(view->as_instance_view()->is_reduction_view());
#endif
            ReductionView *red_view = 
              view->as_instance_view()->as_reduction_view();
            FieldMask &mask = reduction_views[red_view];
            derez.deserialize(mask);
            field_node->transform_field_mask(mask, source);
            view->add_nested_gc_ref(did);
            view->add_nested_valid_ref(did);
          }
        }
        else
        {
          size_t num_valid_views;
          derez.deserialize(num_valid_views);
          for (unsigned idx = 0; idx < num_valid_views; idx++)
          {
            DistributedID did;
            derez.deserialize(did);
            LogicalView *view = owner_node->find_view(did);
            // Check for composite view
            if (view->is_deferred_view())
            {
              DeferredView *def_view = view->as_deferred_view();
              if (def_view->is_composite_view())
                composite_views.push_back(def_view->as_composite_view());
            }
            LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
              valid_views.find(view);
            if (finder != valid_views.end())
            {
              FieldMask update_mask;
              derez.deserialize(update_mask);
              field_node->transform_field_mask(update_mask, source);
              finder->second |= update_mask;
            }
            else
            {
              FieldMask &mask = valid_views[view];
              derez.deserialize(mask);
              field_node->transform_field_mask(mask, source);
              view->add_nested_gc_ref(did);
              view->add_nested_valid_ref(did);
            }
          }
          size_t num_reduction_views;
          derez.deserialize(num_reduction_views);
          for (unsigned idx = 0; idx < num_reduction_views; idx++)
          {
            DistributedID did;
            derez.deserialize(did);
            LogicalView *view = owner_node->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
            assert(view->is_instance_view());
            assert(view->as_instance_view()->is_reduction_view());
#endif
            ReductionView *red_view = 
              view->as_instance_view()->as_reduction_view();
            LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
              reduction_views.find(red_view);
            if (finder != reduction_views.end())
            {
              FieldMask update_mask;
              derez.deserialize(update_mask);
              field_node->transform_field_mask(update_mask, source);
              finder->second |= update_mask;
            }
            else
            {
              FieldMask &mask = reduction_views[red_view];
              derez.deserialize(mask);
              field_node->transform_field_mask(mask, source);
              view->add_nested_gc_ref(did);
              view->add_nested_valid_ref(did);
            }
          }
        }
      }
      // If we have composite views, then we need to make sure
      // that their version states are local as well
      if (!composite_views.empty())
      {
        std::set<Event> preconditions;
        for (std::vector<CompositeView*>::const_iterator it = 
              composite_views.begin(); it != composite_views.end(); it++)
        {
          (*it)->make_local(preconditions); 
        }
        if (!preconditions.empty())
          to_trigger.trigger(Event::merge_events(preconditions));
        else
          to_trigger.trigger();
      }
      else
      {
        // Finally trigger the event saying we have the data
        to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_path_only(
                       Internal *rt, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask path_only_mask;
      derez.deserialize(path_only_mask);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_HIGH_LEVEL
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_path_only(source, path_only_mask);
      UserEvent registered_event;
      derez.deserialize(registered_event);
      if (registered_event.exists())
        registered_event.trigger();
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_initialization(
                       Internal *rt, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask initial_mask;
      derez.deserialize(initial_mask);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_HIGH_LEVEL
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_initialization(source, initial_mask);
      UserEvent registered_event;
      derez.deserialize(registered_event);
      if (registered_event.exists())
        registered_event.trigger();
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_request(Internal *rt,
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      UserEvent to_trigger;
      derez.deserialize(to_trigger);
      VersionRequestKind request_kind;
      derez.deserialize(request_kind);
      FieldMask request_mask;
      derez.deserialize(request_mask);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_HIGH_LEVEL
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_request(source, to_trigger, 
                                       request_kind, request_mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_response(Internal *rt,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      UserEvent to_trigger;
      derez.deserialize(to_trigger);
      VersionRequestKind req_kind;
      derez.deserialize(req_kind);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_HIGH_LEVEL
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_response(source, to_trigger, req_kind, derez);
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
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

    //--------------------------------------------------------------------------
    bool RegionTreePath::has_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      return path[depth].is_valid();
    }

    //--------------------------------------------------------------------------
    const ColorPoint& RegionTreePath::get_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      assert(has_child(depth));
#endif
      return path[depth];
    }

    //--------------------------------------------------------------------------
    unsigned RegionTreePath::get_path_length(void) const
    //--------------------------------------------------------------------------
    {
      return ((max_depth-min_depth)+1); 
    }

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
#ifdef DEBUG_HIGH_LEVEL
      assert(children.find(child_color) == children.end());
#endif
      children[child_color] = child;
    }

    //--------------------------------------------------------------------------
    bool FatTreePath::add_child(const ColorPoint &child_color,
                                FatTreePath *child, IndexTreeNode *tree_node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
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
    // MappingRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingRef::MappingRef(void)
      : view(NULL), needed_fields(FieldMask())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappingRef::MappingRef(LogicalView *v, const FieldMask &needed)
      : view(v), needed_fields(needed)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappingRef::MappingRef(const MappingRef &rhs)
      : view(rhs.view), needed_fields(rhs.needed_fields)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappingRef::~MappingRef(void)
    //--------------------------------------------------------------------------
    {
      view = NULL;
    }

    //--------------------------------------------------------------------------
    MappingRef& MappingRef::operator=(const MappingRef &rhs)
    //--------------------------------------------------------------------------
    {
      view = rhs.view;
      needed_fields = rhs.needed_fields;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // InstanceRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(void)
      : ready_event(Event::NO_EVENT), view(NULL), manager(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(Event ready, InstanceView *v)
      : ready_event(ready), view(v), 
        manager((v == NULL) ? NULL : v->get_manager()) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(Event ready, InstanceView *v,
                             const std::vector<Reservation> &ls)
      : ready_event(ready), view(v), 
        manager((v == NULL) ? NULL : v->get_manager()), needed_locks(ls)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MaterializedView* InstanceRef::get_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
      assert(view->is_materialized_view());
#endif
      return view->as_materialized_view();
    }

    //--------------------------------------------------------------------------
    ReductionView* InstanceRef::get_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
      assert(view->is_reduction_view());
#endif
      return view->as_reduction_view();
    }

    //--------------------------------------------------------------------------
    void InstanceRef::update_atomic_locks(
                 std::map<Reservation,bool> &atomic_locks, bool exclusive)
    //--------------------------------------------------------------------------
    {
      for (std::vector<Reservation>::const_iterator it = needed_locks.begin();
            it != needed_locks.end(); it++)
      {
        std::map<Reservation,bool>::iterator finder = 
          atomic_locks.find(*it);
        if (finder == atomic_locks.end())
          atomic_locks[*it] = exclusive;
        else
          finder->second = finder->second || exclusive;
      }
      // Once someone has asked for our locks we can let them go
      needed_locks.clear();
    }

    //--------------------------------------------------------------------------
    Memory InstanceRef::get_memory(void) const
    //--------------------------------------------------------------------------
    {
      return manager->memory;
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
      InstanceRef::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      return manager->get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      InstanceRef::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return manager->get_field_accessor(fid);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::pack_reference(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (manager != NULL)
      {
        DistributedID did = manager->send_manager(target);
        rez.serialize(did);
        rez.serialize(ready_event);
        rez.serialize<size_t>(needed_locks.size());
        for (std::vector<Reservation>::const_iterator it = 
              needed_locks.begin(); it != needed_locks.end(); it++)
          rez.serialize(*it);
      }
      else
        rez.serialize<DistributedID>(0);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::unpack_reference(Internal *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      if (did == 0)
        return;
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_HIGH_LEVEL
      manager = dynamic_cast<PhysicalManager*>(dc);
      assert(manager != NULL);
#else
      manager = static_cast<PhysicalManager*>(dc);
#endif
      derez.deserialize(ready_event);
      size_t num_locks;
      derez.deserialize(num_locks);
      needed_locks.resize(num_locks);
      for (unsigned idx = 0; idx < num_locks; idx++)
        derez.deserialize(needed_locks[idx]); 
    } 

    /////////////////////////////////////////////////////////////
    // CompositeRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeRef::CompositeRef(void)
      : view(NULL), local(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeRef::CompositeRef(CompositeView *v)
      : view(v), local(true)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
        view->add_base_valid_ref(COMPOSITE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    CompositeRef::CompositeRef(const CompositeRef &rhs)
      : view(rhs.view), local(rhs.local)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
        view->add_base_valid_ref(COMPOSITE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    CompositeRef::~CompositeRef(void)
    //--------------------------------------------------------------------------
    {
      if ((view != NULL) && view->remove_base_valid_ref(COMPOSITE_HANDLE_REF))
        legion_delete(view);
    }

    //--------------------------------------------------------------------------
    CompositeRef& CompositeRef::operator=(const CompositeRef &rhs)
    //--------------------------------------------------------------------------
    {
      if ((view != NULL) && view->remove_base_valid_ref(COMPOSITE_HANDLE_REF))
        legion_delete(view);
      view = rhs.view;
      local = rhs.local;
      if (view != NULL)
        view->add_base_valid_ref(COMPOSITE_HANDLE_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CompositeRef::pack_reference(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
      {
        DistributedID did = view->send_view_base(target);
        rez.serialize(did);
      }
      else
        rez.serialize<DistributedID>(0);
    }

    //--------------------------------------------------------------------------
    void CompositeRef::unpack_reference(Internal *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view == NULL);
#endif
      DistributedID did;
      derez.deserialize(did);
      if (did == 0)
        return;
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_HIGH_LEVEL
      view = dynamic_cast<CompositeView*>(dc);
      assert(view != NULL);
#else
      view = static_cast<CompositeView*>(dc);
#endif
      local = false;
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

