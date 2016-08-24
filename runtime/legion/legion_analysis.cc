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
                               UniqueID id, unsigned idx, RegionNode *n)
      : usage(u), child(c), op_id(id), index(idx), node(n)
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
      rez.serialize(node->handle);
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalUser* PhysicalUser::unpack_user(Deserializer &derez,
                                                       bool add_reference,
                                                       RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      PhysicalUser *result = legion_new<PhysicalUser>();
      derez.deserialize(result->child);
      derez.deserialize(result->usage.privilege);
      derez.deserialize(result->usage.prop);
      derez.deserialize(result->usage.redop);
      derez.deserialize(result->op_id);
      derez.deserialize(result->index);
      LogicalRegion handle;
      derez.deserialize(handle);
      result->node = forest->get_node(handle);
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
    // VersioningSet
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    VersioningSet<REF_KIND>::VersioningSet(void)
      : single(true)
    //--------------------------------------------------------------------------
    {
      versions.single_version = NULL;
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    VersioningSet<REF_KIND>::VersioningSet(const VersioningSet &rhs)
      : single(true) 
    //--------------------------------------------------------------------------
    {
      // must be empty
#ifdef DEBUG_LEGION
      assert(rhs.single);
      assert(rhs.versions.single_version == NULL);
#endif
      versions.single_version = NULL;
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    VersioningSet<REF_KIND>::~VersioningSet(void)
    //--------------------------------------------------------------------------
    {
      clear(); 
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    VersioningSet<REF_KIND>& VersioningSet<REF_KIND>::operator=(
                                                       const VersioningSet &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void* VersioningSet<REF_KIND>::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<VersioningSet<REF_KIND>,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void VersioningSet<REF_KIND>::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    const FieldMask& VersioningSet<REF_KIND>::operator[](
                                                      VersionState *state) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(state == versions.single_version);
#endif
        return valid_fields;
      }
      else
      {
        LegionMap<VersionState*,FieldMask>::aligned::const_iterator finder =
          versions.multi_versions->find(state);
#ifdef DEBUG_LEGION
        assert(finder != versions.multi_versions->end());
#endif
        return finder->second;
      }
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void VersioningSet<REF_KIND>::insert(VersionState *state, 
                               const FieldMask &mask, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      if (single)
      {
        if (versions.single_version == NULL)
        {
          versions.single_version = state;
          valid_fields = mask;
          if (REF_KIND != LAST_SOURCE_REF)
          {
#ifdef DEBUG_LEGION
            //assert(mutator != NULL);
#endif
            state->add_base_valid_ref(REF_KIND, mutator);
          }
        }
        else if (versions.single_version == state)
        {
          valid_fields |= mask;
        }
        else
        {
          // Go to multi
          LegionMap<VersionState*,FieldMask>::aligned *multi = 
            new LegionMap<VersionState*,FieldMask>::aligned();
          (*multi)[versions.single_version] = valid_fields;
          (*multi)[state] = mask;
          versions.multi_versions = multi;
          single = false;
          valid_fields |= mask;
          if (REF_KIND != LAST_SOURCE_REF)
          {
#ifdef DEBUG_LEGION
            //assert(mutator != NULL);
#endif
            state->add_base_valid_ref(REF_KIND, mutator);
          }
        }
      }
      else
      {
 #ifdef DEBUG_LEGION
        assert(versions.multi_versions != NULL);
#endif   
        LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
          versions.multi_versions->find(state);
        if (finder == versions.multi_versions->end())
        {
          (*versions.multi_versions)[state] = mask;
          if (REF_KIND != LAST_SOURCE_REF)
          {
#ifdef DEBUG_LEGION
            //assert(mutator != NULL);
#endif
            state->add_base_valid_ref(REF_KIND, mutator);
          }
        }
        else
          finder->second |= mask;
        valid_fields |= mask;
      }
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    RtEvent VersioningSet<REF_KIND>::insert(VersionState *state,
                                            const FieldMask &mask, 
                                            Runtime *runtime, RtEvent pre)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      if (single)
      {
        if (versions.single_version == NULL)
        {
          versions.single_version = state;
          valid_fields = mask;
          if (REF_KIND != LAST_SOURCE_REF)
          {
            VersioningSetRefArgs args;
            args.hlr_id = HLR_ADD_VERSIONING_SET_REF_TASK_ID;
            args.state = state;
            args.kind = REF_KIND;
            return runtime->issue_runtime_meta_task(&args, sizeof(args),
                HLR_ADD_VERSIONING_SET_REF_TASK_ID, HLR_LATENCY_PRIORITY,
                NULL, pre);
          }
        }
        else if (versions.single_version == state)
        {
          valid_fields |= mask;
        }
        else
        {
          // Go to multi
          LegionMap<VersionState*,FieldMask>::aligned *multi = 
            new LegionMap<VersionState*,FieldMask>::aligned();
          (*multi)[versions.single_version] = valid_fields;
          (*multi)[state] = mask;
          versions.multi_versions = multi;
          single = false;
          valid_fields |= mask;
          if (REF_KIND != LAST_SOURCE_REF)
          {
            VersioningSetRefArgs args;
            args.hlr_id = HLR_ADD_VERSIONING_SET_REF_TASK_ID;
            args.state = state;
            args.kind = REF_KIND;
            return runtime->issue_runtime_meta_task(&args, sizeof(args),
                HLR_ADD_VERSIONING_SET_REF_TASK_ID, HLR_LATENCY_PRIORITY,
                NULL, pre);
          }
        }
      }
      else
      {
 #ifdef DEBUG_LEGION
        assert(versions.multi_versions != NULL);
#endif   
        LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
          versions.multi_versions->find(state);
        if (finder == versions.multi_versions->end())
        {
          (*versions.multi_versions)[state] = mask;
          if (REF_KIND != LAST_SOURCE_REF)
          {
            VersioningSetRefArgs args;
            args.hlr_id = HLR_ADD_VERSIONING_SET_REF_TASK_ID;
            args.state = state;
            args.kind = REF_KIND;
            return runtime->issue_runtime_meta_task(&args, sizeof(args),
                HLR_ADD_VERSIONING_SET_REF_TASK_ID, HLR_LATENCY_PRIORITY,
                NULL, pre);
          }
        }
        else
          finder->second |= mask;
        valid_fields |= mask;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void VersioningSet<REF_KIND>::erase(VersionState *to_erase) 
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(versions.single_version == to_erase);
#endif
        versions.single_version = NULL;
        valid_fields.clear();
      }
      else
      {
        LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
          versions.multi_versions->find(to_erase);
#ifdef DEBUG_LEGION
        assert(finder != versions.multi_versions->end());
#endif
        valid_fields -= finder->second;
        versions.multi_versions->erase(finder);
        if (versions.multi_versions->size() == 1)
        {
          // go back to single
          finder = versions.multi_versions->begin();
          valid_fields = finder->second;
          VersionState *first = finder->first;
          delete versions.multi_versions;
          versions.single_version = first;
          single = true;
        }
      }
      if ((REF_KIND != LAST_SOURCE_REF) && 
          to_erase->remove_base_valid_ref(REF_KIND))
        legion_delete(to_erase);
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void VersioningSet<REF_KIND>::clear(void)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((REF_KIND != LAST_SOURCE_REF) && (versions.single_version != NULL))
        {
          if (versions.single_version->remove_base_valid_ref(REF_KIND))
            legion_delete(versions.single_version);
        }
        versions.single_version = NULL;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(versions.multi_versions != NULL);
#endif
        if (REF_KIND != LAST_SOURCE_REF)
        {
          for (LegionMap<VersionState*,FieldMask>::aligned::iterator it = 
                versions.multi_versions->begin(); it != 
                versions.multi_versions->end(); it++)
          {
            if (it->first->remove_base_valid_ref(REF_KIND))
              legion_delete(it->first);
          }
        }
        delete versions.multi_versions;
        versions.multi_versions = NULL;
        single = true;
      }
      valid_fields.clear();
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    size_t VersioningSet<REF_KIND>::size(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (versions.single_version == NULL)
          return 0;
        else
          return 1;
      }
      else
        return versions.multi_versions->size(); 
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    std::pair<VersionState*,FieldMask>* 
                      VersioningSet<REF_KIND>::next(VersionState *current) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(current == versions.single_version);
#endif
        return NULL; 
      }
      else
      {
        LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
          versions.multi_versions->find(current);
#ifdef DEBUG_LEGION
        assert(finder != versions.multi_versions->end());
#endif
        finder++;
        if (finder == versions.multi_versions->end())
          return NULL;
        else
          return reinterpret_cast<
                      std::pair<VersionState*,FieldMask>*>(&(*finder));
      }
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void VersioningSet<REF_KIND>::move(VersioningSet &other)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(other.empty());
#endif
      if (single)
      {
        other.versions.single_version = versions.single_version;
        other.single = true;
        versions.single_version = NULL;
      }
      else
      {
        other.versions.multi_versions = versions.multi_versions;
        other.single = false;
        versions.multi_versions = NULL;
        single = true;
      }
      other.valid_fields = valid_fields;
      valid_fields.clear();
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    typename VersioningSet<REF_KIND>::iterator 
                                     VersioningSet<REF_KIND>::begin(void) const
    //--------------------------------------------------------------------------
    {
      // Scariness!
      if (single)
      {
        // If we're empty return end
        if (versions.single_version == NULL)
          return end();
        return iterator(this, 
            reinterpret_cast<std::pair<VersionState*,FieldMask>*>(
              const_cast<VersioningSet<REF_KIND>*>(this)), true/*single*/);
      }
      else
        return iterator(this,
            reinterpret_cast<std::pair<VersionState*,FieldMask>*>(
              &(*(versions.multi_versions->begin()))), false); 
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND> template<ReferenceSource ARG_KIND>
    void VersioningSet<REF_KIND>::reduce(const FieldMask &merge_mask, 
                                         VersioningSet<ARG_KIND> &new_states,
                                         ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If you are looking for the magical reduce function that allows
      // us to know which are the most recent version state objects, well
      // you can congratulate yourself because you've found it
#ifdef DEBUG_LEGION
      sanity_check();
      new_states.sanity_check();
#endif
      std::vector<VersionState*> to_erase_new;
      for (typename VersioningSet<ARG_KIND>::iterator nit = new_states.begin();
            nit != new_states.end(); nit++)
      {
        LegionMap<VersionState*,FieldMask>::aligned to_add; 
        std::vector<VersionState*> to_erase_local;
        FieldMask overlap = merge_mask & nit->second;
        // This VersionState doesn't apply locally if there are no fields
        if (!overlap)
          continue;
        // We can remove these fields from the new states becasue
        // we know that we are going to handle it
        nit->second -= overlap;
        if (!nit->second)
          to_erase_new.push_back(nit->first);
        // Iterate over our states states and see which ones interfere
        for (typename VersioningSet<REF_KIND>::iterator it = begin(); 
              it != end(); it++)
        {
          FieldMask local_overlap = nit->second & overlap;
          if (!local_overlap)
            continue;
          // Overlapping fields to two different version states, compare
          // the version numbers to see which one we should keep
          if (it->first->version_number < nit->first->version_number)
          {
            // Take the next one, throw away this one
            to_add[nit->first] |= local_overlap;
            it->second -= local_overlap;
            if (!it->second)
              to_erase_local.push_back(it->first);
          }
#ifdef DEBUG_LEGION
          else if (it->first->version_number == nit->first->version_number)
            // better be the same object with overlapping fields 
            // and the same version number
            assert(it->first == nit->first);
          // Otherwise we kepp the old one and throw away the new one
#endif  
          overlap -= local_overlap;
          if (!overlap)
            break;
        }
        // If we still have fields for this version state, then
        // we just have to insert it locally
        if (!!overlap)
          insert(nit->first, overlap, mutator);
        if (!to_erase_local.empty())
        {
          for (std::vector<VersionState*>::const_iterator it = 
                to_erase_local.begin(); it != to_erase_local.end(); it++)
            erase(*it);
        }
        if (!to_add.empty())
        {
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator
                it = to_add.begin(); it != to_add.end(); it++)
            insert(it->first, it->second, mutator);
        }
      }
      if (!to_erase_new.empty())
      {
        for (std::vector<VersionState*>::const_iterator it = 
              to_erase_new.begin(); it != to_erase_new.end(); it++)
          new_states.erase(*it);
      }
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void VersioningSet<REF_KIND>::sanity_check(void) const
    //--------------------------------------------------------------------------
    {
      // Each field should exist exactly once
      if (!single)
      {
        assert(versions.multi_versions != NULL);
        FieldMask previous_mask;
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              versions.multi_versions->begin(); it != 
              versions.multi_versions->end(); it++)
        {
          assert(previous_mask * it->second);
          previous_mask |= it->second;
        }
      }
    }
#endif

    /////////////////////////////////////////////////////////////
    // VersionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(void)
      : upper_bound_node(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(const VersionInfo &rhs)
      : upper_bound_node(rhs.upper_bound_node), 
        field_versions(rhs.field_versions), split_masks(rhs.split_masks)
    //--------------------------------------------------------------------------
    {
      physical_states.resize(rhs.physical_states.size(), NULL); 
      for (unsigned idx = 0; idx < physical_states.size(); idx++)
      {
        if (rhs.physical_states[idx] == NULL)
          continue;
        physical_states[idx] = rhs.physical_states[idx]->clone();
      }
    }

    //--------------------------------------------------------------------------
    VersionInfo::~VersionInfo(void)
    //--------------------------------------------------------------------------
    {
      clear();
    }

    //--------------------------------------------------------------------------
    VersionInfo& VersionInfo::operator=(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_versions.empty());
      assert(physical_states.empty());
      assert(split_masks.empty());
#endif
      upper_bound_node = rhs.upper_bound_node;
      field_versions = rhs.field_versions;
      split_masks = rhs.split_masks;
      physical_states.resize(rhs.physical_states.size(), NULL); 
      for (unsigned idx = 0; idx < physical_states.size(); idx++)
      {
        if (rhs.physical_states[idx] == NULL)
          continue;
        physical_states[idx] = rhs.physical_states[idx]->clone();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::record_split_fields(RegionTreeNode *node,
                            const FieldMask &split_mask, unsigned offset/*= 0*/)
    //--------------------------------------------------------------------------
    {
      const unsigned depth = node->get_depth() + offset;
#ifdef DEBUG_LEGION
      assert(depth < split_masks.size());
#endif
      split_masks[depth] = split_mask;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::add_current_version(VersionState *state, 
                                    const FieldMask &state_mask, bool path_only) 
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *node = state->logical_node;
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < physical_states.size());
#endif
      if (physical_states[depth] == NULL)
        physical_states[depth] = legion_new<PhysicalState>(node, path_only);
      physical_states[depth]->add_version_state(state, state_mask);
      // Now record the version information
#ifdef DEBUG_LEGION
      assert(depth < field_versions.size());
#endif
      FieldVersions &local_versions = field_versions[depth];
      FieldVersions::iterator finder = 
        local_versions.find(state->version_number);
      if (finder == local_versions.end())
        local_versions[state->version_number] = state_mask;
      else
        finder->second |= state_mask;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::add_advance_version(VersionState *state, 
                                    const FieldMask &state_mask, bool path_only) 
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *node = state->logical_node;
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < physical_states.size());
#endif
      if (physical_states[depth] == NULL)
        physical_states[depth] = legion_new<PhysicalState>(node, path_only);
      physical_states[depth]->add_advance_state(state, state_mask);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::set_upper_bound_node(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(upper_bound_node == NULL);
#endif
      upper_bound_node = node;
    }

    //--------------------------------------------------------------------------
    bool VersionInfo::has_physical_states(void) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < physical_states.size(); idx++)
      {
        if (physical_states[idx] != NULL)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::apply_mapping(AddressSpaceID target,
             std::set<RtEvent> &applied_conditions, bool copy_through/*=false*/)
    //--------------------------------------------------------------------------
    {
      // We only ever need to apply state at the leaves
#ifdef DEBUG_LEGION
      assert(!physical_states.empty());
#endif
      unsigned last_idx = physical_states.size() - 1;
      if (copy_through)
      {
        // Deal with mis-speculated state that we still have to propagate
        PhysicalState *state = physical_states[last_idx];
        // If we have advance states and we haven't capture, then
        // we need to propagate information
        if (state->has_advance_states() && !state->is_captured())
          state->capture_state();
      }
      physical_states[last_idx]->apply_state(target, applied_conditions);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::resize(size_t max_depth)
    //--------------------------------------------------------------------------
    {
      // Make this max_depth+1
      max_depth += 1;
      field_versions.resize(max_depth);
      physical_states.resize(max_depth,NULL);
      split_masks.resize(max_depth);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      upper_bound_node = NULL;
      field_versions.clear();
      for (std::vector<PhysicalState*>::const_iterator it = 
            physical_states.begin(); it != physical_states.end(); it++)
      {
        if ((*it) != NULL)
          legion_delete(*it);
      }
      physical_states.clear();
      split_masks.clear();
    }

    //--------------------------------------------------------------------------
    void VersionInfo::sanity_check(unsigned depth)
    //--------------------------------------------------------------------------
    {
      if (depth >= field_versions.size())
        return;
      const FieldVersions &versions = field_versions[depth];
      FieldMask previous_fields;
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            versions.begin(); it != versions.end(); it++)
      {
        assert(previous_fields * it->second);
        previous_fields |= it->second;
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clone_logical(const VersionInfo &rhs, 
                                 const FieldMask &mask, RegionTreeNode *to_node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(upper_bound_node == NULL);
      assert(physical_states.empty());
      assert(field_versions.empty());
      assert(split_masks.empty());
#endif
      const unsigned max_depth = to_node->get_depth() + 1;
#ifdef DEBUG_LEGION
      assert(max_depth <= rhs.split_masks.size());
#endif
      // Only need to copy over the upper bound and split masks that
      // are computed as part of the logical analysis
      upper_bound_node = rhs.upper_bound_node;
      split_masks.resize(max_depth);
      for (unsigned idx = 0; idx < max_depth; idx++)
        split_masks[idx] = rhs.split_masks[idx] & mask;
      // Only need to resize the other things
      field_versions.resize(max_depth);
      physical_states.resize(max_depth, NULL);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::copy_to(VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
      rhs.upper_bound_node = upper_bound_node;
      // No need to copy over the physical states
      rhs.field_versions = field_versions;
      rhs.split_masks = split_masks;
    }

    //--------------------------------------------------------------------------
    PhysicalState* VersionInfo::find_physical_state(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < physical_states.size());
#endif
      PhysicalState *result = physical_states[depth];
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      if (!result->is_captured())
        result->capture_state();
      return result;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::get_field_versions(RegionTreeNode *node,
                                         const FieldMask &needed_fields,
                                         FieldVersions &result_versions)
    //--------------------------------------------------------------------------
    {
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < field_versions.size());
#endif
      result_versions = field_versions[depth];
    }

    //--------------------------------------------------------------------------
    void VersionInfo::get_advance_versions(RegionTreeNode *node,
                                           const FieldMask &needed_fields,
                                           FieldVersions &result_versions)
    //--------------------------------------------------------------------------
    {
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < split_masks.size());
      assert(depth < field_versions.size());
#endif
      FieldMask split_overlap = split_masks[depth] & needed_fields;
      if (!!split_overlap)
      {
        // Intermediate node with split fields, we only need the
        // split fields for the advance part
        const FieldVersions &local_versions = field_versions[depth];
        for (FieldVersions::const_iterator it = local_versions.begin();
              it != local_versions.end(); it++)
        {
          FieldMask overlap = it->second & split_overlap;
          if (!overlap)
            continue;
          result_versions[it->first+1] = overlap;
          split_overlap -= overlap;
          if (!split_overlap)
            break;
        }
      }
      else if (depth == (field_versions.size() - 1))
      {
        // Bottom node with no split fields so therefore we need all
        // the fields advanced
        const FieldVersions &local_versions = field_versions[depth];
        for (FieldVersions::const_iterator it = local_versions.begin();
              it != local_versions.end(); it++)
        {
          FieldMask overlap = needed_fields & it->second;
          if (!overlap)
            continue;
          result_versions[it->first+1] = overlap;
        }
      }
      // Otherwise it was an intermediate level with no split mask
      // so there are by definition no advance fields
    }

    //--------------------------------------------------------------------------
    const FieldMask& VersionInfo::get_split_mask(unsigned depth) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(depth < split_masks.size());
#endif
      return split_masks[depth];
    }

    //--------------------------------------------------------------------------
    void VersionInfo::get_split_mask(RegionTreeNode *node,
                                     const FieldMask &needed_fields,
                                     FieldMask &result)
    //--------------------------------------------------------------------------
    {
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < split_masks.size());
#endif
      result = split_masks[depth];
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_version_info(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      pack_version_numbers(rez);
      rez.serialize<size_t>(physical_states.size());
#ifdef DEBUG_LEGION
      bool first = true;
#endif
      bool is_region = upper_bound_node->is_region();
      for (std::vector<PhysicalState*>::const_iterator it = 
            physical_states.begin(); it != physical_states.end(); it++)
      {
        if ((*it) == NULL)
        {
#ifdef DEBUG_LEGION
          assert(first); // should only see NULL above
#endif
          continue;
        }
#ifdef DEBUG_LEGION
        if (first)
          assert((*it)->node == upper_bound_node);
        first = false;
#endif
        rez.serialize<bool>((*it)->path_only);
        if (is_region)
          rez.serialize((*it)->node->as_region_node()->handle);
        else
          rez.serialize((*it)->node->as_partition_node()->handle);
        (*it)->pack_physical_state(rez);
        // Reverse polarity
        is_region = !is_region;
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_version_info(Deserializer &derez, 
                              Runtime *runtime, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      unpack_version_numbers(derez, runtime->forest);
      size_t num_states;
      derez.deserialize(num_states);
      physical_states.resize(num_states, NULL);
      unsigned start_depth = upper_bound_node->get_depth();
      bool is_region = upper_bound_node->is_region();
      for (unsigned idx = start_depth; idx < num_states; idx++)
      {
        bool is_path_only;
        derez.deserialize(is_path_only);
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
        PhysicalState *next = legion_new<PhysicalState>(node, is_path_only);
        next->unpack_physical_state(derez, runtime, ready_events);
        physical_states[idx] = next;
        // Reverse the polarity
        is_region = !is_region;
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_version_numbers(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(split_masks.size() == field_versions.size());
#endif
      pack_upper_bound_node(rez);
      // Then pack the split masks, nothing else needs to be sent
      rez.serialize<size_t>(split_masks.size());
      for (unsigned idx = 0; idx < split_masks.size(); idx++)
        rez.serialize(split_masks[idx]);
      for (unsigned idx = 0; idx < field_versions.size(); idx++)
      {
        const LegionMap<VersionID,FieldMask>::aligned &fields = 
          field_versions[idx];
        rez.serialize<size_t>(fields.size());
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              fields.begin(); it != fields.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_version_numbers(Deserializer &derez,
                                             RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      unpack_upper_bound_node(derez, forest);
      size_t depth;
      derez.deserialize(depth);
      split_masks.resize(depth);
      for (unsigned idx = 0; idx < depth; idx++)
        derez.deserialize(split_masks[idx]);
      field_versions.resize(depth);
      for (unsigned idx = 0; idx < depth; idx++)
      {
        size_t num_versions;
        derez.deserialize(num_versions);
        if (num_versions == 0)
          continue;
        LegionMap<VersionID,FieldMask>::aligned &fields = 
          field_versions[idx];
        for (unsigned idx2 = 0; idx2 < num_versions; idx2++)
        {
          VersionID vid;
          derez.deserialize(vid);
          derez.deserialize(fields[vid]);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_upper_bound_node(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Pack the upper bound node (if there is one)
      if (upper_bound_node != NULL)
      {
        if (upper_bound_node->is_region())
        {
          rez.serialize<bool>(true/*is region*/);
          rez.serialize(upper_bound_node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false/*is region*/);
          rez.serialize(upper_bound_node->as_partition_node()->handle);
        }
      }
      else
      {
        rez.serialize<bool>(true/*is region*/);
        rez.serialize(LogicalRegion::NO_REGION);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_upper_bound_node(Deserializer &derez,
                                              RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(upper_bound_node == NULL);
#endif
      bool is_region;
      derez.deserialize(is_region);
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        if (handle.exists())
          upper_bound_node = forest->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        upper_bound_node = forest->get_node(handle);
      }
    }

    /////////////////////////////////////////////////////////////
    // RestrictInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RestrictInfo::RestrictInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RestrictInfo::RestrictInfo(const RestrictInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(restrictions.empty());
#endif
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it = 
            rhs.restrictions.begin(); it != rhs.restrictions.end(); it++)
      {
        it->first->add_base_gc_ref(RESTRICTED_REF);
        restrictions.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    RestrictInfo::~RestrictInfo(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        if (it->first->remove_base_gc_ref(RESTRICTED_REF))
          legion_delete(it->first);
      }
      restrictions.clear();
    }

    //--------------------------------------------------------------------------
    RestrictInfo& RestrictInfo::operator=(const RestrictInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(restrictions.empty());
#endif
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it = 
            rhs.restrictions.begin(); it != rhs.restrictions.end(); it++)
      {
        it->first->add_base_gc_ref(RESTRICTED_REF);
        restrictions.insert(*it);
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::record_restriction(InstanceManager *inst, 
                                          const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<InstanceManager*,FieldMask>::aligned::iterator finder = 
        restrictions.find(inst);
      if (finder == restrictions.end())
      {
        inst->add_base_gc_ref(RESTRICTED_REF);
        restrictions[inst] = mask;
      }
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::populate_restrict_fields(FieldMask &to_fill) const
    //--------------------------------------------------------------------------
    {
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
        to_fill |= it->second;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        if (it->first->remove_base_gc_ref(RESTRICTED_REF))
          legion_delete(it->first);
      }
      restrictions.clear();
      restricted_instances.clear();
    }

    //--------------------------------------------------------------------------
    const InstanceSet& RestrictInfo::get_instances(void)
    //--------------------------------------------------------------------------
    {
      if (restricted_instances.size() == restrictions.size())
        return restricted_instances;
      restricted_instances.resize(restrictions.size());
      unsigned idx = 0;
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++, idx++)
        restricted_instances[idx] = InstanceRef(it->first, it->second);
      return restricted_instances;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::pack_info(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(restrictions.size());
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::unpack_info(Deserializer &derez, Runtime *runtime,
                                   std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_restrictions;
      derez.deserialize(num_restrictions);
      for (unsigned idx = 0; idx < num_restrictions; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        InstanceManager *manager = static_cast<InstanceManager*>( 
          runtime->find_or_request_physical_manager(did, ready));
        derez.deserialize(restrictions[manager]);
        if (ready.exists() && !ready.has_triggered())
        {
          DeferRestrictedManagerArgs args;
          args.hlr_id = HLR_DEFER_RESTRICTED_MANAGER_TASK_ID;
          args.manager = manager;
          ready = runtime->issue_runtime_meta_task(&args, sizeof(args),
              HLR_DEFER_RESTRICTED_MANAGER_TASK_ID, HLR_LATENCY_PRIORITY,
              NULL, ready);
          ready_events.insert(ready);
        }
        else
        {
          WrapperReferenceMutator mutator(ready_events);
          manager->add_base_gc_ref(RESTRICTED_REF, &mutator);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RestrictInfo::handle_deferred_reference(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRestrictedManagerArgs *margs = 
        (const DeferRestrictedManagerArgs*)args;
      LocalReferenceMutator mutator;
      margs->manager->add_base_gc_ref(RESTRICTED_REF, &mutator);
    }

    /////////////////////////////////////////////////////////////
    // Restriction 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Restriction::Restriction(RegionNode *n)
      : tree_id(n->handle.get_tree_id()), local_node(n)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Restriction::Restriction(const Restriction &rhs)
      : tree_id(rhs.tree_id), local_node(rhs.local_node)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Restriction::~Restriction(void)
    //--------------------------------------------------------------------------
    {
      // Delete our acquisitions
      for (std::set<Acquisition*>::const_iterator it = acquisitions.begin();
            it != acquisitions.end(); it++)
        delete (*it);
      acquisitions.clear();
      // Remove references on any of our instances
      for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator it =
            instances.begin(); it != instances.end(); it++)
      {
        if (it->first->remove_base_gc_ref(RESTRICTED_REF))
          legion_delete(it->first);
      }
      instances.clear();
    }

    //--------------------------------------------------------------------------
    Restriction& Restriction::operator=(const Restriction &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* Restriction::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<Restriction,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void Restriction::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void Restriction::add_restricted_instance(InstanceManager *inst,
                                              const FieldMask &inst_fields)
    //--------------------------------------------------------------------------
    {
      // Always update the restricted fields
      restricted_fields |= inst_fields;
      LegionMap<InstanceManager*,FieldMask>::aligned::iterator finder = 
        instances.find(inst);
      if (finder == instances.end())
      {
        inst->add_base_gc_ref(RESTRICTED_REF);
        instances[inst] = inst_fields;
      }
      else
        finder->second |= inst_fields; 
    }

    //--------------------------------------------------------------------------
    void Restriction::find_restrictions(RegionTreeNode *node, 
              FieldMask &possibly_restricted, RestrictInfo &restrict_info) const
    //--------------------------------------------------------------------------
    {
      if (!local_node->intersects_with(node))    
        return;
      // See if we have any acquires that make this alright
      for (std::set<Acquisition*>::const_iterator it = acquisitions.begin();
            it != acquisitions.end(); it++)
      {
        (*it)->find_restrictions(node, possibly_restricted, restrict_info);
        if (!possibly_restricted)
          return;
      }
      // If we make it here then we are restricted
      FieldMask restricted = possibly_restricted & restricted_fields;
      if (!!restricted)
      {
        // Record the restrictions
        for (LegionMap<InstanceManager*,FieldMask>::aligned::const_iterator
              it = instances.begin(); it != instances.end(); it++)
        {
          FieldMask overlap = it->second & restricted;
          if (!overlap)
            continue;
          restrict_info.record_restriction(it->first, overlap);
        }
        // Remove the restricted fields
        possibly_restricted -= restricted;
      }
    }

    //--------------------------------------------------------------------------
    bool Restriction::matches(DetachOp *op, RegionNode *node,
                              FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      // Not the same node, then we aren't going to match
      if (local_node != node)
        return false;
      FieldMask overlap = remaining_fields & restricted_fields;
      if (!overlap)
        return false;
      // If we have any acquired fields here, we can't match
      for (std::set<Acquisition*>::const_iterator it = acquisitions.begin();
            it != acquisitions.end(); it++)
      {
        (*it)->remove_acquired_fields(overlap);
        if (!overlap)
          return false;
      }
      // These are the fields that we match for
      remaining_fields -= overlap;
      restricted_fields -= overlap;
      // We've been removed, deletion will clean up the references
      if (!restricted_fields)
        return true;
      // Filter out the overlapped instances
      std::vector<InstanceManager*> to_delete;
      for (LegionMap<InstanceManager*,FieldMask>::aligned::iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        it->second -= overlap;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      if (!to_delete.empty())
      {
        for (std::vector<InstanceManager*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          instances.erase(*it);
          if ((*it)->remove_base_gc_ref(RESTRICTED_REF))
            legion_delete(*it);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void Restriction::remove_restricted_fields(FieldMask &remaining) const
    //--------------------------------------------------------------------------
    {
      remaining -= restricted_fields;
    }

    //--------------------------------------------------------------------------
    void Restriction::add_acquisition(AcquireOp *op, RegionNode *node,
                                      FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      FieldMask overlap = restricted_fields & remaining_fields;
      if (!overlap)
        return;
      // If we don't dominate then we can't help
      if (!local_node->dominates(node))
      {
        if (local_node->intersects_with(node))
        {
          log_run.error("Illegal partial acquire operation (ID %lld) "
                        "performed in task %s (ID %lld)", op->get_unique_id(),
                        op->get_parent()->get_task_name(),
                        op->get_parent()->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_ILLEGAL_PARTIAL_ACQUISITION);
        }
        return;
      }
      // At this point we know we'll be handling the fields one 
      // way or another so remove them for the original set
      remaining_fields -= overlap;
      // Try adding it to any of the acquires
      for (std::set<Acquisition*>::const_iterator it = acquisitions.begin();
            it != acquisitions.end(); it++)
      {
        (*it)->add_acquisition(op, node, overlap);
        if (!overlap)
          return;
      }
      // If we still have any remaining fields, we can add them here
      acquisitions.insert(new Acquisition(node, overlap));
    }
    
    //--------------------------------------------------------------------------
    void Restriction::remove_acquisition(ReleaseOp *op, RegionNode *node,
                                         FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      if (restricted_fields * remaining_fields)
        return;
      if (!local_node->intersects_with(node))
        return;
      std::vector<Acquisition*> to_delete;
      for (std::set<Acquisition*>::const_iterator it = acquisitions.begin();
            it != acquisitions.end(); it++)
      {
        if ((*it)->matches(op, node, remaining_fields))
          to_delete.push_back(*it);
        else if (!!remaining_fields)
          (*it)->remove_acquisition(op, node, remaining_fields);
        if (!remaining_fields)
          return;
      }
      if (!to_delete.empty())
      {
        for (std::vector<Acquisition*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          acquisitions.erase(*it);
          delete (*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Restriction::add_restriction(AttachOp *op, RegionNode *node,
                             InstanceManager *inst, FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      if (restricted_fields * remaining_fields)
        return;
      if (!local_node->intersects_with(node))
        return;
      // Try adding it to any of our acquires
      for (std::set<Acquisition*>::const_iterator it = acquisitions.begin();
            it != acquisitions.end(); it++)
      {
        (*it)->add_restriction(op, node, inst, remaining_fields);
        if (!remaining_fields)
          return;
      }
      // It's bad if we get here
      log_run.error("Illegal interfering restriction performed by attach "
                    "operation (ID %lld) in task %s (ID %lld)",
                    op->get_unique_op_id(), op->get_parent()->get_task_name(),
                    op->get_parent()->get_unique_op_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_ILLEGAL_INTERFERING_RESTRICTIONS);
    }
    
    //--------------------------------------------------------------------------
    void Restriction::remove_restriction(DetachOp *op, RegionNode *node,
                                         FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      if (restricted_fields * remaining_fields)
        return;
      if (!local_node->dominates(node))
        return;
      for (std::set<Acquisition*>::const_iterator it = acquisitions.begin();
            it != acquisitions.end(); it++)
      {
        (*it)->remove_restriction(op, node, remaining_fields);
        if (!remaining_fields)
          return;
      }
    }

    /////////////////////////////////////////////////////////////
    // Acquisition 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Acquisition::Acquisition(RegionNode *node, const FieldMask &acquired)
      : local_node(node), acquired_fields(acquired)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Acquisition::Acquisition(const Acquisition &rhs)
      : local_node(rhs.local_node)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Acquisition::~Acquisition(void)
    //--------------------------------------------------------------------------
    {
      for (std::set<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
        delete (*it);
      restrictions.clear();
    }

    //--------------------------------------------------------------------------
    Acquisition& Acquisition::operator=(const Acquisition &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* Acquisition::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<Acquisition,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void Acquisition::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void Acquisition::find_restrictions(RegionTreeNode *node,
                                        FieldMask &possibly_restricted,
                                        RestrictInfo &restrict_info) const
    //--------------------------------------------------------------------------
    {
      if (acquired_fields * possibly_restricted)
        return;
      if (!local_node->intersects_with(node))
        return;
      // Check to see if it is restricted below
      for (std::set<Restriction*>::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        (*it)->find_restrictions(node, possibly_restricted, restrict_info);
        if (!possibly_restricted)
          return;
      }
      FieldMask overlap = acquired_fields & possibly_restricted;
      // If we dominate and they weren't restricted below, we know
      // that they are acquired
      if (!!overlap && local_node->dominates(node))
        possibly_restricted -= overlap;
    }

    //--------------------------------------------------------------------------
    bool Acquisition::matches(ReleaseOp *op, RegionNode *node,
                              FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      if (local_node != node)
        return false;
      FieldMask overlap = remaining_fields & acquired_fields;
      if (!overlap)
        return false;
      // If we have any restricted fields below, then we can't match
      for (std::set<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        (*it)->remove_restricted_fields(overlap);
        if (!overlap)
          return false;
      }
      // These are the fields that we match for
      remaining_fields -= overlap;
      acquired_fields -= overlap;
      if (!acquired_fields)
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void Acquisition::remove_acquired_fields(FieldMask &remaining_fields) const
    //--------------------------------------------------------------------------
    {
      remaining_fields -= acquired_fields;
    }

    //--------------------------------------------------------------------------
    void Acquisition::add_acquisition(AcquireOp *op, RegionNode *node,
                                      FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      if (acquired_fields * remaining_fields)
        return;
      if (!local_node->intersects_with(node))
        return;
      for (std::set<Restriction*>::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        (*it)->add_acquisition(op, node, remaining_fields);
        if (!remaining_fields)
          return;
      }
      // It's bad if we get here
      log_run.error("Illegal interfering acquire operation performed by "
                    "acquire operation (ID %lld) in task %s (ID %lld)",
                    op->get_unique_op_id(), op->get_parent()->get_task_name(),
                    op->get_parent()->get_unique_op_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_ILLEGAL_INTERFERING_ACQUISITIONS);
    }

    //--------------------------------------------------------------------------
    void Acquisition::remove_acquisition(ReleaseOp *op, RegionNode *node,
                                         FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      if (acquired_fields * remaining_fields)
        return;
      if (!local_node->dominates(node))
        return;
      for (std::set<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        (*it)->remove_acquisition(op, node, remaining_fields);
        if (!remaining_fields)
          return;
      }
    }

    //--------------------------------------------------------------------------
    void Acquisition::add_restriction(AttachOp *op, RegionNode *node,
                          InstanceManager *manager, FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      FieldMask overlap = remaining_fields & acquired_fields;
      if (!overlap)
        return;
      if (!local_node->dominates(node))
      {
        if (local_node->intersects_with(node))
        {
          log_run.error("Illegal partial restriction operation performed by "
                        "attach operation (ID %lld) in task %s (ID %lld)",
                        op->get_unique_op_id(), 
                        op->get_parent()->get_task_name(),
                        op->get_parent()->get_unique_op_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_ILLEGAL_PARTIAL_RESTRICTION);
        }
        return;
      }
      // At this point we know we'll be able to do the restriction
      remaining_fields -= overlap;
      for (std::set<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        (*it)->add_restriction(op, node, manager, overlap);
        if (!overlap)
          return;
      }
      Restriction *restriction = new Restriction(node);
      restriction->add_restricted_instance(manager, overlap);
      restrictions.insert(restriction); 
    }

    //--------------------------------------------------------------------------
    void Acquisition::remove_restriction(DetachOp *op, RegionNode *node,
                                         FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      if (acquired_fields * remaining_fields)
        return;
      if (!local_node->intersects_with(node))
        return;
      std::vector<Restriction*> to_delete;
      for (std::set<Restriction*>::const_iterator it = restrictions.begin();
            it != restrictions.end(); it++)
      {
        if ((*it)->matches(op, node, remaining_fields))
          to_delete.push_back(*it);
        else if (!!remaining_fields)
          (*it)->remove_restriction(op, node, remaining_fields);
        if (!remaining_fields)
          return;
      }
      if (!to_delete.empty())
      {
        for (std::vector<Restriction*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          restrictions.erase(*it);
          delete (*it);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // ProjectionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionInfo::ProjectionInfo(Runtime *runtime, 
                      const RegionRequirement &req, const Domain &launch_domain)
      : projection((req.handle_type != SINGULAR) ? 
          runtime->find_projection_function(req.projection) : NULL),
        projection_domain((req.handle_type != SINGULAR) ?
            launch_domain : Domain::NO_DOMAIN)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::record_projection_epoch(ProjectionEpochID epoch,
                                                 const FieldMask &epoch_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(projection_epochs.find(epoch) == projection_epochs.end());
#endif
      projection_epochs[epoch] = epoch_mask;
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::pack_info(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(projection_epochs.size());
      for (LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator 
            it = projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::unpack_info(Deserializer &derez, Runtime *runtime,
                      const RegionRequirement &req, const Domain &launch_domain)
    //--------------------------------------------------------------------------
    {
      if (req.handle_type != SINGULAR)
      {
        projection = runtime->find_projection_function(req.projection);
        projection_domain = launch_domain; 
      }
      size_t num_epochs;
      derez.deserialize(num_epochs);
      for (unsigned idx = 0; idx < num_epochs; idx++)
      {
        ProjectionEpochID epoch_id;
        derez.deserialize(epoch_id);
        derez.deserialize(projection_epochs[epoch_id]);
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
    CurrentInvalidator::CurrentInvalidator(ContextID c)
      : ctx(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator::CurrentInvalidator(const CurrentInvalidator &rhs)
      : ctx(0)
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
      node->invalidate_current_state(ctx); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx);
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
    bool ReductionCloser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      issue_close_reductions(node);
      return true;
    }

    //--------------------------------------------------------------------------
    bool ReductionCloser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      issue_close_reductions(node);
      return true;
    }

    //--------------------------------------------------------------------------
    void ReductionCloser::issue_close_reductions(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      PhysicalState *state = node->get_physical_state(version_info);
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
    // Projection Epoch
    /////////////////////////////////////////////////////////////

    // C++ is really dumb
    const ProjectionEpochID ProjectionEpoch::first_epoch;

    //--------------------------------------------------------------------------
    ProjectionEpoch::ProjectionEpoch(ProjectionEpochID id, const FieldMask &m)
      : epoch_id(id), valid_fields(m)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionEpoch::ProjectionEpoch(const ProjectionEpoch &rhs)
      : epoch_id(rhs.epoch_id), valid_fields(rhs.valid_fields)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ProjectionEpoch::~ProjectionEpoch(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionEpoch& ProjectionEpoch::operator=(const ProjectionEpoch &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* ProjectionEpoch::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<ProjectionEpoch,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void ProjectionEpoch::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void ProjectionEpoch::insert(ProjectionFunction *function, const Domain &d)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!valid_fields);
#endif
      projections[function].insert(d);
    }

    /////////////////////////////////////////////////////////////
    // CurrentState 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    CurrentState::CurrentState(RegionTreeNode *node, ContextID ctx)
      : owner(node)
    //--------------------------------------------------------------------------
    {
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
      assert(projection_epochs.empty());
      assert(!dirty_fields);
      assert(!dirty_below);
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
      dirty_fields.clear();
      dirty_below.clear();
      outstanding_reduction_fields.clear();
      outstanding_reductions.clear();
      for (std::map<ProjectionEpochID,ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
        delete it->second;
      projection_epochs.clear();
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
      dirty_fields -= deleted_mask;
    }

    //--------------------------------------------------------------------------
    void CurrentState::advance_projection_epochs(const FieldMask &advance_mask)
    //--------------------------------------------------------------------------
    {
      std::map<ProjectionEpochID,ProjectionEpoch*> to_add; 
      std::set<ProjectionEpochID> to_delete;
      for (LegionMap<ProjectionEpochID,ProjectionEpoch*>::aligned::
            reverse_iterator it = projection_epochs.rbegin(); 
            it != projection_epochs.rend(); it++) 
      {
        FieldMask overlap = it->second->valid_fields & advance_mask;
        if (!overlap)
          continue;
        it->second->valid_fields -= overlap;
        if (!it->second->valid_fields)
          to_delete.insert(it->first);
        // See if we can add it to an existing one
        LegionMap<ProjectionEpochID,ProjectionEpoch*>::aligned::iterator 
          finder = projection_epochs.find(it->first+1);
        if (finder != projection_epochs.end())
        {
          // If we were going to erase it, then don't erase it anymore
          if (!finder->second->valid_fields)
          {
#ifdef DEBUG_LEGION
            assert(to_delete.find(finder->first) != to_delete.end());
#endif
            to_delete.erase(finder->first);
          }
          finder->second->valid_fields |= overlap;
        }
        else
          to_add[it->first+1] = new ProjectionEpoch(it->first+1, overlap);
      }
      if (!to_delete.empty())
      {
        for (std::set<ProjectionEpochID>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          std::map<ProjectionEpochID,ProjectionEpoch*>::iterator finder =
            projection_epochs.find(*it);
#ifdef DEBUG_LEGION
          assert(finder != projection_epochs.end());
          // Field should be empty
          assert(!finder->second->valid_fields);
#endif
          delete finder->second;
          projection_epochs.erase(finder);
        }
      }
      if (!to_add.empty())
      {
        for (std::map<ProjectionEpochID,ProjectionEpoch*>::const_iterator it = 
              to_add.begin(); it != to_add.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(projection_epochs.find(it->first) == projection_epochs.end());
#endif
          projection_epochs.insert(*it); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void CurrentState::capture_projection_epochs(FieldMask capture_mask,
                                                 ProjectionInfo &info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!capture_mask);
#endif
      for (std::map<ProjectionEpochID,ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        FieldMask overlap = it->second->valid_fields & capture_mask;
        if (!overlap)
          continue;
        info.record_projection_epoch(it->first, overlap);
        capture_mask -= overlap;
        if (!capture_mask)
          return;
      }
      // If it didn't already exist, start a new projection epoch
      ProjectionEpoch *new_epoch = 
        new ProjectionEpoch(ProjectionEpoch::first_epoch, capture_mask);
      new_epoch->insert(info.projection, info.projection_domain);
      projection_epochs[ProjectionEpoch::first_epoch] = new_epoch;
      // Record it
      info.record_projection_epoch(ProjectionEpoch::first_epoch, capture_mask);
    }

    //--------------------------------------------------------------------------
    void CurrentState::capture_close_epochs(FieldMask capture_mask,
                                            ClosedNode *closed_node) const
    //--------------------------------------------------------------------------
    {
      for (std::map<ProjectionEpochID,ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        FieldMask overlap = it->second->valid_fields & capture_mask;
        if (!overlap)
          continue;
        closed_node->record_projections(it->second, overlap);
        capture_mask -= overlap;
        if (!capture_mask)
          return;
      }
    }

    //--------------------------------------------------------------------------
    void CurrentState::update_projection_epochs(FieldMask update_mask,
                                                const ProjectionInfo &info)
    //--------------------------------------------------------------------------
    {
      for (std::map<ProjectionEpochID,ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        FieldMask overlap = it->second->valid_fields & update_mask;
        if (!overlap)
          continue;
        it->second->insert(info.projection, info.projection_domain);
        update_mask -= overlap;
        if (!update_mask)
          return;
      }
#ifdef DEBUG_LEGION
      assert(!!update_mask);
#endif
      // If we get here will still have an update mask so make an epoch
      ProjectionEpoch *new_epoch = 
        new ProjectionEpoch(ProjectionEpoch::first_epoch, update_mask);
      new_epoch->insert(info.projection, info.projection_domain);
      projection_epochs[ProjectionEpoch::first_epoch] = new_epoch;
    }

    /////////////////////////////////////////////////////////////
    // FieldState 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldState::FieldState(void)
      : open_state(NOT_OPEN), redop(0), projection(NULL), 
        projection_domain(Domain::NO_DOMAIN), rebuild_timeout(1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const GenericUser &user, const FieldMask &m, 
                           const ColorPoint &c)
      : ChildState(m), redop(0), projection(NULL), 
        projection_domain(Domain::NO_DOMAIN), rebuild_timeout(1)
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
    FieldState::FieldState(const GenericUser &user, const FieldMask &m,
                ProjectionFunction *proj, const Domain &proj_dom, bool disjoint)
      : ChildState(m), redop(0), projection(proj), 
        projection_domain(proj_dom), rebuild_timeout(1)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(projection != NULL);
#endif
      if (IS_READ_ONLY(user.usage))
        open_state = OPEN_READ_ONLY_PROJ;
      else if (IS_REDUCE(user.usage))
      {
        open_state = OPEN_REDUCE_PROJ;
        redop = user.usage.redop;
      }
      else if (disjoint && (projection->depth == 1))
        open_state = OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW;
      else
        open_state = OPEN_READ_WRITE_PROJ;
    }

    //--------------------------------------------------------------------------
    bool FieldState::overlaps(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      if (redop != rhs.redop)
        return false;
      if (projection != rhs.projection)
        return false;
      // Only do this test if they are both projections
      if ((projection != NULL) && (projection_domain != rhs.projection_domain))
        return false;
      if (redop == 0)
        return (open_state == rhs.open_state);
      else
      {
#ifdef DEBUG_LEGION
        assert((open_state == OPEN_SINGLE_REDUCE) ||
               (open_state == OPEN_MULTI_REDUCE) ||
               (open_state == OPEN_REDUCE_PROJ));
        assert((rhs.open_state == OPEN_SINGLE_REDUCE) ||
               (rhs.open_state == OPEN_MULTI_REDUCE) ||
               (rhs.open_state == OPEN_REDUCE_PROJ));
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
      assert(projection == rhs.projection);
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
    bool FieldState::projection_domain_dominates(const Domain &next_dom) const
    //--------------------------------------------------------------------------
    {
      if (projection_domain == next_dom)
        return true;
#ifdef DEBUG_LEGION
      assert(projection_domain.get_dim() == next_dom.get_dim());
      assert(projection_domain.get_dim() > 0);
#endif
      switch (projection_domain.get_dim())
      {
        case 1:
          {
            LegionRuntime::Arrays::Rect<1> local_rect =  
              projection_domain.get_rect<1>();
            LegionRuntime::Arrays::Rect<1> next_rect = 
              next_dom.get_rect<1>();
            return local_rect.dominates(next_rect);
          }
        case 2:
          {
            LegionRuntime::Arrays::Rect<2> local_rect =  
              projection_domain.get_rect<2>();
            LegionRuntime::Arrays::Rect<2> next_rect = 
              next_dom.get_rect<2>();
            return local_rect.dominates(next_rect);
          }
        case 3:
          {
            LegionRuntime::Arrays::Rect<3> local_rect =  
              projection_domain.get_rect<3>();
            LegionRuntime::Arrays::Rect<3> next_rect = 
              next_dom.get_rect<3>();
            return local_rect.dominates(next_rect);
          }
        default:
          assert(false);
      }
      return false;
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
        case OPEN_READ_ONLY_PROJ:
          {
            logger->log("Field State: OPEN READ-ONLY PROJECTION %d",
                        projection);
            break;
          }
        case OPEN_READ_WRITE_PROJ:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION %d",
                        projection);
            break;
          }
        case OPEN_REDUCE_PROJ:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION %d Mode %d",
                        projection, redop);
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
    // Closed Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ClosedNode::ClosedNode(RegionTreeNode *n)
      : node(n)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ClosedNode::ClosedNode(const ClosedNode &rhs)
      : node(rhs.node)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ClosedNode::~ClosedNode(void)
    //--------------------------------------------------------------------------
    {
      // Recursively delete the rest of the tree
      for (std::map<RegionTreeNode*,ClosedNode*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
        delete it->second;
      children.clear();
    }

    //--------------------------------------------------------------------------
    ClosedNode& ClosedNode::operator=(const ClosedNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* ClosedNode::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<ClosedNode,true/*bytes*/>(count); 
    }

    //--------------------------------------------------------------------------
    void ClosedNode::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void ClosedNode::add_child_node(ClosedNode *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(children.find(child->node) == children.end());
#endif
      children[child->node] = child; 
    }

    //--------------------------------------------------------------------------
    void ClosedNode::record_closed_fields(const FieldMask &fields)
    //--------------------------------------------------------------------------
    {
      covered_fields |= fields;
    }

    //--------------------------------------------------------------------------
    void ClosedNode::record_projections(const ProjectionEpoch *epoch,
                                        const FieldMask &fields)
    //--------------------------------------------------------------------------
    {
      for (std::map<ProjectionFunction*,std::set<Domain> >::const_iterator pit =
            epoch->projections.begin(); pit != epoch->projections.end(); pit++)
      {
        std::map<ProjectionFunction*,LegionMap<Domain,FieldMask>::aligned>::
          iterator finder = projections.find(pit->first);
        if (finder != projections.end())
        {
          for (std::set<Domain>::const_iterator it = pit->second.begin();
                it != pit->second.end(); it++)
          {
            LegionMap<Domain,FieldMask>::aligned::iterator finder2 = 
              finder->second.find(*it);
            if (finder2 == finder->second.end())
              finder->second[*it] = fields;
            else
              finder2->second |= fields;
          }
        }
        else
        {
          // Didn't exist before so we can just insert 
          LegionMap<Domain,FieldMask>::aligned &doms = projections[pit->first];
          for (std::set<Domain>::const_iterator it = pit->second.begin();
                it != pit->second.end(); it++)
            doms[*it] = fields;
        }
      }
    }

    //--------------------------------------------------------------------------
    void ClosedNode::fix_closed_tree(void)
    //--------------------------------------------------------------------------
    {
      // If we are complete and have all our children, that we can also
      // infer covering at this node
      if (!children.empty())
      {
        const bool local_complete = node->is_complete() && 
          (children.size() == node->get_num_children());
        bool first_child = true;
        FieldMask child_covered;
        // Do all our sub-trees first
        for (std::map<RegionTreeNode*,ClosedNode*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          // Recurse down the tree
          it->second->fix_closed_tree();
          // Update our valid mask
          valid_fields |= it->second->get_valid_fields();
          // If the child is complete we can also update covered
          if (it->second->node->is_complete())
            covered_fields |= it->second->get_covered_fields();
          if (local_complete)
          {
            if (first_child)
            {
              child_covered = it->second->get_covered_fields();
              first_child = false;
            }
            else
              child_covered &= it->second->get_covered_fields();
          }
        }
        if (local_complete && !!child_covered)
          covered_fields |= child_covered;
      }
      // All our covered fields are always valid
      valid_fields |= covered_fields;
      // Finally update our valid fields based on any projections
      if (!projections.empty())
      {
        for (std::map<ProjectionFunction*,
                  LegionMap<Domain,FieldMask>::aligned>::const_iterator pit = 
              projections.begin(); pit != projections.end(); pit++) 
        {
          for (LegionMap<Domain,FieldMask>::aligned::const_iterator it = 
                pit->second.begin(); it != pit->second.end(); it++)
            valid_fields |= it->second;
        }
      }
    }

    //--------------------------------------------------------------------------
    void ClosedNode::filter_dominated_fields(const ClosedNode *old_tree, 
                                            FieldMask &non_dominated_mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node == old_tree->node); // should always be the same
#endif
      // We can remove any fields we are covered by here
      if (!!covered_fields)
      {
        non_dominated_mask -= covered_fields;
        if (!non_dominated_mask)
          return;
      }
      // If we have any projections, we can also try to filter by that
      if (!projections.empty())
      {
        old_tree->filter_dominated_projection_fields(non_dominated_mask, 
                                                     projections); 
        if (!non_dominated_mask)
          return;
      }
      // Otherwise try to see if the children zip well
      old_tree->filter_dominated_children(non_dominated_mask, children);
    }

    //--------------------------------------------------------------------------
    void ClosedNode::filter_dominated_projection_fields(
        FieldMask &non_dominated_mask,
        const std::map<ProjectionFunction*,
                  LegionMap<Domain,FieldMask>::aligned> &new_projections) const
    //--------------------------------------------------------------------------
    {
      // In order to remove a dominated field, for each of our projection
      // operations, we need to find one in the new set that dominates it
      FieldMask dominated_mask = non_dominated_mask;
      for (std::map<ProjectionFunction*,
                    LegionMap<Domain,FieldMask>::aligned>::const_iterator pit =
            projections.begin(); pit != projections.end(); pit++)
      {
        // Set this iterator to the begining to start
        // Use it later to find domains with the same projection function
        std::map<ProjectionFunction*,
                 LegionMap<Domain,FieldMask>::aligned>::const_iterator
                   finder = new_projections.begin();
        for (LegionMap<Domain,FieldMask>::aligned::const_iterator dit = 
              pit->second.begin(); dit != pit->second.end(); dit++)
        {
          FieldMask overlap = dit->second & dominated_mask;
          if (!overlap)
            continue;
          // If it's still at the beginning try to find it
          if (finder == new_projections.begin())
            finder = new_projections.find(pit->first);
          // If we found it then we can try to find overlapping domains
          if (finder != new_projections.end())
          {
            for (LegionMap<Domain,FieldMask>::aligned::const_iterator it = 
                  finder->second.begin(); it != finder->second.end(); it++)
            {
              FieldMask dom_overlap = overlap & it->second;
              if (!dom_overlap)
                continue; 
              // Dimensions don't have to match, if they don't we don't care
              if (it->first.get_dim() != dit->first.get_dim())
                continue;
              // See if the domain dominates
              switch (it->first.get_dim())
              {
                case 1:
                  {
                    Rect<1> prev_rect = dit->first.get_rect<1>();
                    Rect<1> next_rect = it->first.get_rect<1>();
                    if (next_rect.dominates(prev_rect))
                      overlap -= dom_overlap;
                    break;
                  }
                case 2:
                  {
                    Rect<2> prev_rect = dit->first.get_rect<2>();
                    Rect<2> next_rect = it->first.get_rect<2>();
                    if (next_rect.dominates(prev_rect))
                      overlap -= dom_overlap;
                    break;
                  }
                case 3:
                  {
                    Rect<3> prev_rect = dit->first.get_rect<3>();
                    Rect<3> next_rect = it->first.get_rect<3>();
                    if (next_rect.dominates(prev_rect))
                      overlap -= dom_overlap;
                    break;
                  }
                default:
                  assert(false); // bad dimension size
              }
              if (!overlap)
                break;
            }
          }
          // Any fields still in overlap are not dominated
          if (!!overlap)
          {
            dominated_mask -= overlap;
            if (!dominated_mask)
              break;
          }
        }
        // Didn't find any dominated fields so we are done
        if (!dominated_mask)
          break;
      }
      if (!!dominated_mask)
        non_dominated_mask -= dominated_mask;
    }

    //--------------------------------------------------------------------------
    void ClosedNode::filter_dominated_children(FieldMask &non_dominated_mask,
               const std::map<RegionTreeNode*,ClosedNode*> &new_children) const
    //--------------------------------------------------------------------------
    {
      // In order to remove a field, it has to be dominated in all our children
      FieldMask dominated_fields = non_dominated_mask;
      for (std::map<RegionTreeNode*,ClosedNode*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        FieldMask overlap = it->second->get_valid_fields() & non_dominated_mask;
        if (!overlap)
          continue;
        std::map<RegionTreeNode*,ClosedNode*>::const_iterator finder = 
          new_children.find(it->first);
        // If we can't find it, then we are immediately done
        if (finder == new_children.end())
          return;
        FieldMask child_non_dominated = overlap;
        finder->second->filter_dominated_fields(it->second,child_non_dominated);
        dominated_fields &= (overlap - child_non_dominated);
        if (!dominated_fields)
          return;
      }
      non_dominated_mask -= dominated_fields;
    }

    //--------------------------------------------------------------------------
    void ClosedNode::pack_closed_node(Serializer &rez) const 
    //--------------------------------------------------------------------------
    {
      if (node->is_region())
        rez.serialize(node->as_region_node()->handle);
      else
        rez.serialize(node->as_partition_node()->handle);
      rez.serialize(valid_fields);
      rez.serialize(covered_fields);
      rez.serialize<size_t>(projections.size());
      for (std::map<ProjectionFunction*,
                    LegionMap<Domain,FieldMask>::aligned>::const_iterator pit =
            projections.begin(); pit != projections.end(); pit++)
      {
        rez.serialize(pit->first->projection_id);
        rez.serialize<size_t>(pit->second.size());
        for (LegionMap<Domain,FieldMask>::aligned::const_iterator it = 
              pit->second.begin(); it != pit->second.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      rez.serialize<size_t>(children.size());
      for (std::map<RegionTreeNode*,ClosedNode*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
        it->second->pack_closed_node(rez);
    }

    //--------------------------------------------------------------------------
    void ClosedNode::perform_unpack(Deserializer &derez, 
                                    Runtime *runtime, bool is_region)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(valid_fields);
      derez.deserialize(covered_fields);
      size_t num_projections;
      derez.deserialize(num_projections);
      for (unsigned idx = 0; idx < num_projections; idx++)
      {
        ProjectionID pid;
        derez.deserialize(pid);
        ProjectionFunction *function = runtime->find_projection_function(pid);
        LegionMap<Domain,FieldMask>::aligned &doms = projections[function];
        size_t num_doms;
        derez.deserialize(num_doms);
        for (unsigned idx2 = 0; idx2 < num_doms; idx2++)
        {
          Domain dom;
          derez.deserialize(dom);
          derez.deserialize(doms[dom]);
        }
      }
      size_t num_children;
      derez.deserialize(num_children);
      for (unsigned idx = 0; idx < num_children; idx++)
      {
        ClosedNode *child = unpack_closed_node(derez, runtime, !is_region); 
        children[child->node] = child;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ ClosedNode* ClosedNode::unpack_closed_node(Deserializer &derez,
                                               Runtime *runtime, bool is_region)
    //--------------------------------------------------------------------------
    {
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
      ClosedNode *result = legion_new<ClosedNode>(node);
      result->perform_unpack(derez, runtime, is_region);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Logical Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(ContextID c, const LogicalUser &u, 
                                 RegionTreeNode *r, bool val, bool capture)
      : ctx(c), user(u), root_node(r), validates(val), capture_users(capture),
        normal_close_op(NULL),read_only_close_op(NULL),flush_only_close_op(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(const LogicalCloser &rhs)
      : user(rhs.user), root_node(rhs.root_node), validates(rhs.validates),
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
    void LogicalCloser::record_close_operation(const FieldMask &mask, 
                               bool leave_open, bool read_only, bool flush_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
      // at most one of these should be true
      assert(!((leave_open && read_only) || (leave_open && flush_only) || 
                (read_only && flush_only)));
#endif
      if (read_only)
        read_only_close_mask |= mask;
      else if (flush_only)
        flush_only_close_mask |= mask;
      else
      {
        normal_close_mask |= mask;
        if (leave_open)
          leave_open_mask |= mask;
      }
    }

    //--------------------------------------------------------------------------
    ClosedNode* LogicalCloser::find_closed_node(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeNode*,ClosedNode*>::const_iterator finder = 
        closed_nodes.find(node);
      if (finder != closed_nodes.end())
        return finder->second;
      // Otherwise we need to make it
      ClosedNode *result = new ClosedNode(node);
      closed_nodes[node] = result;
      // Make it up the tree if necessary
      if (node != root_node)
      {
        ClosedNode *parent = find_closed_node(node->get_parent());
        parent->add_child_node(result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_closed_user(const LogicalUser &user,
                                          const FieldMask &mask, bool read_only)
    //--------------------------------------------------------------------------
    {
      if (read_only)
      {
        read_only_closed_users.push_back(user);
        LogicalUser &closed_user = read_only_closed_users.back();
        closed_user.field_mask = mask;
      }
      else
      {
        normal_closed_users.push_back(user);
        LogicalUser &closed_user = normal_closed_users.back();
        closed_user.field_mask = mask;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::initialize_close_operations(RegionTreeNode *target, 
                                                   Operation *creator,
                                                   const VersionInfo &ver_info,
                                                   const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // These two sets of fields better be disjoint
      assert(normal_close_mask * flush_only_close_mask);
      assert(!(leave_open_mask - normal_close_mask));
#endif
      // Construct a reigon requirement for this operation
      // All privileges are based on the parent logical region
      RegionRequirement req;
      if (root_node->is_region())
        req = RegionRequirement(root_node->as_region_node()->handle,
                                READ_WRITE, EXCLUSIVE, trace_info.req.parent);
      else
        req = RegionRequirement(root_node->as_partition_node()->handle, 0,
                                READ_WRITE, EXCLUSIVE, trace_info.req.parent);
      if (!!normal_close_mask)
      {
        normal_close_op = creator->runtime->get_available_inter_close_op(false);
        // Compute the set of fields that we need
        root_node->column_source->get_field_set(normal_close_mask,
                                               trace_info.req.privilege_fields,
                                               req.privilege_fields);
        std::map<RegionTreeNode*,ClosedNode*>::const_iterator finder = 
          closed_nodes.find(root_node);
#ifdef DEBUG_LEGION
        assert(finder != closed_nodes.end()); // better have a closed tree
#endif
        // We can update the closed fields at the root
        finder->second->record_closed_fields(normal_close_mask);
        // Now initialize the operation
        normal_close_op->initialize(creator->get_parent(), req, finder->second,
                                    trace_info.trace, trace_info.req_idx, 
                                    ver_info, normal_close_mask, 
                                    leave_open_mask,creator);
        // We can clear this now
        closed_nodes.clear();
      }
      if (!!read_only_close_mask)
      {
        read_only_close_op = 
          creator->runtime->get_available_read_close_op(false);
        req.privilege_fields.clear();
        root_node->column_source->get_field_set(read_only_close_mask,
                                               trace_info.req.privilege_fields,
                                               req.privilege_fields);
        read_only_close_op->initialize(creator->get_parent(), req, 
                                       trace_info.trace, trace_info.req_idx, 
                                       read_only_close_mask, creator);
      }
      // Finally if we have any fields which are flush only
      // make a close operation for them and add it to force close
      if (!!flush_only_close_mask)
      {
        flush_only_close_op =
          creator->runtime->get_available_inter_close_op(false);
        req.privilege_fields.clear();
        // Compute the set of fields that we need
        root_node->column_source->get_field_set(normal_close_mask,
                                               trace_info.req.privilege_fields,
                                               req.privilege_fields);
        // Make a closed tree of just the root node
        // There are no dirty fields here since we just flushing reductions
        ClosedNode *closed_tree = new ClosedNode(root_node);
        FieldMask empty_leave_open;
        flush_only_close_op->initialize(creator->get_parent(), req, closed_tree,
            trace_info.trace, trace_info.req_idx, ver_info, 
            flush_only_close_mask, empty_leave_open, creator);
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
      if (normal_close_op != NULL)
      {
        LogicalUser normal_close_user(normal_close_op, 0/*idx*/, 
            RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), normal_close_mask);
        register_dependences(normal_close_op, normal_close_user, current, 
            open_below, normal_closed_users, above_users, cusers, pusers);
      }
      if (read_only_close_op != NULL)
      {
        LogicalUser read_only_close_user(read_only_close_op, 0/*idx*/, 
          RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), read_only_close_mask);
        register_dependences(read_only_close_op, read_only_close_user, current,
            open_below, read_only_closed_users, above_users, cusers, pusers);
      }
      if (flush_only_close_op != NULL)
      {
        LegionList<LogicalUser,CLOSE_LOGICAL_ALLOC>::track_aligned no_users;
        LogicalUser flush_close_user(flush_only_close_op, 0/*idx*/, 
         RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), flush_only_close_mask);
        register_dependences(flush_only_close_op, flush_close_user,
            current, open_below, no_users, above_users, cusers, pusers);
      }
      // Now we can remove our references on our local users
      for (LegionList<LogicalUser>::aligned::const_iterator it = 
            normal_closed_users.begin(); it != normal_closed_users.end(); it++)
      {
        it->op->remove_mapping_reference(it->gen);
      }
      for (LegionList<LogicalUser>::aligned::const_iterator it =
            read_only_closed_users.begin(); it != 
            read_only_closed_users.end(); it++)
      {
        it->op->remove_mapping_reference(it->gen);
      }
    }

    // If you are looking for LogicalCloser::register_dependences it can 
    // be found in region_tree.cc to make sure that templates are instantiated

    //--------------------------------------------------------------------------
    void LogicalCloser::update_state(CurrentState &state)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state.owner == root_node);
#endif
      // If we only have read-only closes then we are done
      FieldMask closed_mask = normal_close_mask | flush_only_close_mask;
      if (!closed_mask)
        return;
      state.dirty_below -= closed_mask;
      root_node->filter_prev_epoch_users(state, closed_mask);
      root_node->filter_curr_epoch_users(state, closed_mask);
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::register_close_operations(
               LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users)
    //--------------------------------------------------------------------------
    {
      // No need to add mapping references, we did that in 
      // LogicalCloser::register_dependences
      if (normal_close_op != NULL)
      {
        LogicalUser close_user(normal_close_op, 0/*idx*/, 
            RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), normal_close_mask);
        users.push_back(close_user);
      }
      if (read_only_close_op != NULL)
      {
        LogicalUser close_user(read_only_close_op, 0/*idx*/, 
          RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), read_only_close_mask);
        users.push_back(close_user);
      }
      if (flush_only_close_op != NULL)
      {
        LogicalUser close_user(flush_only_close_op, 0/*idx*/, 
         RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), flush_only_close_mask);
        users.push_back(close_user);
      }
    }

    /////////////////////////////////////////////////////////////
    // Physical State 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(RegionTreeNode *n, bool path)
      : node(n), path_only(path), captured(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(const PhysicalState &rhs)
      : node(NULL), path_only(false)
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
    void PhysicalState::pack_physical_state(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(version_states.size());
      for (PhysicalVersions::iterator it = version_states.begin();
            it != version_states.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(advance_states.size());
      for (PhysicalVersions::iterator it = advance_states.begin();
            it != advance_states.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::unpack_physical_state(Deserializer &derez,
                                              Runtime *runtime,
                                              std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      WrapperReferenceMutator mutator(ready_events);
      size_t num_versions;
      derez.deserialize(num_versions);
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        FieldMask mask;
        derez.deserialize(mask);
        RtEvent ready;
        VersionState *state = runtime->find_or_request_version_state(did,ready);
        if (ready.exists())
        {
          RtEvent done = version_states.insert(state, mask, runtime, ready);
          ready_events.insert(done);
        }
        else
          version_states.insert(state, mask, &mutator);
      }
      size_t num_advance;
      derez.deserialize(num_advance);
      for (unsigned idx = 0; idx < num_advance; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        FieldMask mask;
        derez.deserialize(mask);
        RtEvent ready;
        VersionState *state = runtime->find_or_request_version_state(did,ready);
        if (ready.exists())
        {
          RtEvent done = advance_states.insert(state, mask, runtime, ready);
          ready_events.insert(done);
        }
        else
          advance_states.insert(state, mask, &mutator);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::add_version_state(VersionState *state, 
                                          const FieldMask &state_mask)
    //--------------------------------------------------------------------------
    {
      version_states.insert(state, state_mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalState::add_advance_state(VersionState *state, 
                                          const FieldMask &state_mask)
    //--------------------------------------------------------------------------
    {
      advance_states.insert(state, state_mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalState::capture_state(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        PHYSICAL_STATE_CAPTURE_STATE_CALL);
#ifdef DEBUG_LEGION
      assert(!captured);
#endif
      // Path only first since path only can also be a split
      if (path_only)
      {
        for (PhysicalVersions::iterator it = version_states.begin();
              it != version_states.end(); it++)
        {
          it->first->update_path_only_state(this, it->second);
        }
      }
      else
      {
        for (PhysicalVersions::iterator it = version_states.begin();
              it != version_states.end(); it++)
        {
          it->first->update_physical_state(this, it->second);
        }
      }
      captured = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalState::apply_state(AddressSpaceID target, 
                                    std::set<RtEvent> &applied_conditions) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime,
                        PHYSICAL_STATE_APPLY_STATE_CALL);
#ifdef DEBUG_LEGION
      assert(captured);
#endif
      // No advance versions then we are done
      if (advance_states.empty())
        return;
      for (PhysicalVersions::iterator it = advance_states.begin();
            it != advance_states.end(); it++)
      {
        it->first->merge_physical_state(this, it->second, target,
                                        applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::capture_composite_root(CompositeView *composite_view,
                                               const FieldMask &close_mask) 
    //--------------------------------------------------------------------------
    {
      // We need to capture the normal instance data from the version
      // states and the open children from the advance states. We 
      // already know that everything is valid
      for (PhysicalVersions::iterator it = version_states.begin();
            it != version_states.end(); it++)
      {
        FieldMask overlap = it->second & close_mask;
        if (!overlap)
          continue;
        it->first->capture_root_instances(composite_view, overlap);
      }
      for (PhysicalVersions::iterator it = advance_states.begin();
            it != advance_states.end(); it++)
      {
        FieldMask overlap = it->second & close_mask;
        if (!overlap)
          continue;
        it->first->capture_root_children(composite_view, overlap);
      }
    }

    //--------------------------------------------------------------------------
    PhysicalState* PhysicalState::clone(void) const
    //--------------------------------------------------------------------------
    {
      PhysicalState *result = legion_new<PhysicalState>(node, path_only);
      if (!version_states.empty())
      {
        for (PhysicalVersions::iterator it = version_states.begin();
              it != version_states.end(); it++)
          result->add_version_state(it->first, it->second);
      }
      if (!advance_states.empty())
      {
        for (PhysicalVersions::iterator it = advance_states.begin();
              it != advance_states.end(); it++)
          result->add_advance_state(it->first, it->second);
      }
      if (is_captured())
      {
        result->dirty_mask = dirty_mask;
        result->reduction_mask = reduction_mask;
        result->valid_views = valid_views;
        result->reduction_views = reduction_views;
      }
      return result;
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
    } 

    /////////////////////////////////////////////////////////////
    // Version Manager 
    /////////////////////////////////////////////////////////////

    // C++ is dumb
    const VersionID VersionManager::init_version;

    //--------------------------------------------------------------------------
    VersionManager::VersionManager(RegionTreeNode *n, ContextID c)
      : ctx(c), node(n), depth(n->get_depth()), runtime(n->context->runtime),
        current_context(NULL), is_owner(false)
    //--------------------------------------------------------------------------
    {
      manager_lock = Reservation::create_reservation();
    }

    //--------------------------------------------------------------------------
    VersionManager::VersionManager(const VersionManager &rhs)
      : ctx(rhs.ctx), node(rhs.node), depth(rhs.depth), runtime(rhs.runtime)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VersionManager::~VersionManager(void)
    //--------------------------------------------------------------------------
    {
      manager_lock.destroy_reservation();
      manager_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    VersionManager& VersionManager::operator=(const VersionManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* VersionManager::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<VersionManager,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void* VersionManager::operator new[](size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<VersionManager,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void VersionManager::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void VersionManager::operator delete[](void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void VersionManager::reset(void)
    //--------------------------------------------------------------------------
    {
      is_owner = false;
      current_context = NULL;
      remote_valid_fields.clear();
      remote_valid.clear();
      previous_advancers.clear();
      if (!current_version_infos.empty())
        current_version_infos.clear();
      if (!previous_version_infos.empty())
        previous_version_infos.clear();
    }

    //--------------------------------------------------------------------------
    void VersionManager::initialize_state(ApEvent term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          const InstanceSet &targets,
                                          SingleTask *context,
                                          unsigned init_index,
                                 const std::vector<LogicalView*> &corresponding)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_version_infos.empty() || 
              (current_version_infos.size() == 1));
      assert(previous_version_infos.empty());
#endif
      // See if we have been assigned
      if (context != current_context)
      {
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = context;
      }
      // No need to hold the lock when initializing
      if (current_version_infos.empty())
      {
        std::set<RtEvent> dummy_events;
        VersionState *init_state = create_new_version_state(init_version);
#ifdef DEBUG_LEGION
        assert(dummy_events.empty());
#endif
        init_state->initialize(term_event, usage, user_mask, 
                               targets, context, init_index, corresponding);
        current_version_infos[init_version].insert(init_state, user_mask);
      }
      else
      {
        LegionMap<VersionID,ManagerVersions>::aligned::iterator finder = 
          current_version_infos.find(init_version);
#ifdef DEBUG_LEGION
        assert(finder != current_version_infos.end());
        assert(finder->second.size() == 1); // should only be one
#endif
        ManagerVersions::iterator it = finder->second.begin();
        VersionState *init_state = it->first;
#ifdef DEBUG_LEGION
        assert(node == init_state->logical_node);
#endif
        init_state->initialize(term_event, usage, user_mask, targets, 
                               context, init_index, corresponding);
        it->second |= user_mask;
      }
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_versions(const FieldMask &version_mask, 
                                    FieldMask &unversioned_mask,
                                    SingleTask *context, 
                                    Operation *op, unsigned index,
                                    const RegionUsage &usage,
                                    VersionInfo &version_info,
                                    std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // See if we have been assigned
      if (context != current_context)
      {
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = context;
      }
      // See if we are the owner
      if (!is_owner)
      {
        FieldMask request_mask = version_mask - remote_valid_fields;
        if (!!request_mask)
        {
          RtEvent wait_on = send_remote_version_request(request_mask,
                                                        ready_events);
          wait_on.wait();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(version_mask - remote_valid_fields));
#endif
        }
      }
      // Now we can record our versions
      FieldMask unversioned = version_mask;
      if (IS_WRITE(usage))
      {
        // At first we only need the lock in read-only mode
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
        sanity_check();
#endif
        // Always need to capture both previous and next versions
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator 
              vit = previous_version_infos.begin(); vit !=
              previous_version_infos.end(); vit++)
        {
          FieldMask local_overlap = vit->second.get_valid_mask() & version_mask;
          if (!local_overlap)
            continue;
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_current_version(it->first, overlap,
                                             false/*path only*/);
            it->first->request_final_version_state(overlap, ready_events);
          }
        }
        // We're about to produce the first version of this instance
        // so there's no need to request it
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
              vit = current_version_infos.begin(); vit != 
              current_version_infos.end(); vit++)
        {
          FieldMask local_overlap = version_mask & vit->second.get_valid_mask();
          if (!local_overlap)
            continue;
          unversioned -= local_overlap;
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_advance_version(it->first, overlap,
                                             false/*path only*/);
          }
          if (!unversioned)
            break;
        }
#ifdef DEBUG_LEGION
        assert(!unversioned); // everyone should be versioned here
#endif
      }
      else
      {
        // At first we only need the lock in read-only mode
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
        sanity_check();
#endif
        // We only need the current versions, but we reocrd them
        // as both the previous and the advance
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
              vit = current_version_infos.begin(); vit != 
              current_version_infos.end(); vit++)
        {
          FieldMask local_overlap = vit->second.get_valid_mask() & version_mask;
          if (!local_overlap)
            continue;
          unversioned -= local_overlap;
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_current_version(it->first, overlap,
                                             false/*path only*/);
            version_info.add_advance_version(it->first, overlap,
                                             false/*path only*/);
            it->first->request_initial_version_state(overlap, ready_events);
          }
          if (!unversioned)
            break;
        }
      }
      // If we have unversioned fields we need to make new 
      // version state object(s) for those fields
      if (!!unversioned)
      {
#ifdef DEBUG_LEGION
        assert(is_owner); // should only get here on the owner
#endif
        // Retake the lock in exclusive mode and see if we lost any races
        AutoLock m_lock(manager_lock);
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
              vit = current_version_infos.begin(); vit != 
              current_version_infos.end(); vit++)
        {
          // Only need to check against unversioned this time
          FieldMask local_overlap = vit->second.get_valid_mask() & unversioned;
          if (!local_overlap)
            continue;
          unversioned -= local_overlap;
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_current_version(it->first, overlap,
                                             false/*path only*/);
            version_info.add_advance_version(it->first, overlap,
                                             false/*path only*/);
            it->first->request_initial_version_state(overlap, ready_events);
          }
          if (!unversioned)
            break;
        }
        if (!!unversioned)
        {
          VersionState *new_state = create_new_version_state(init_version);
          version_info.add_current_version(new_state, unversioned,
                                           false/*path only*/);
          version_info.add_advance_version(new_state, unversioned,
                                           false/*path only*/);
          // No need to query for initial state since we know there is none
          WrapperReferenceMutator mutator(ready_events);
          current_version_infos[init_version].insert(new_state, 
                                                     unversioned, &mutator);
          // Keep any unversioned fields
          unversioned_mask &= unversioned;
        }
        else if (!!unversioned_mask)
          unversioned_mask.clear();
      }
      else if (!!unversioned_mask)
        unversioned_mask.clear();
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_path_only_versions(
                                                const FieldMask &version_mask,
                                                const FieldMask &split_mask,
                                                FieldMask &unversioned_mask,
                                                SingleTask *context,
                                                Operation *op, unsigned index,
                                                const RegionUsage &usage,
                                                VersionInfo &version_info,
                                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // See if we have been assigned
      if (context != current_context)
      {
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = context;
      }
      // See if we are the owner
      if (!is_owner)
      {
        FieldMask request_mask = version_mask - remote_valid_fields;
        if (!!request_mask)
        {
          RtEvent wait_on = send_remote_version_request(request_mask,
                                                        ready_events); 
          wait_on.wait();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(version_mask - remote_valid_fields));
#endif
        }
      }
      FieldMask unversioned = version_mask;
      // We aren't mutating our data structures, so we just need 
      // the manager lock in read only mode
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      sanity_check();
#endif
      // Do different things depending on whether we are 
      // not read-only, have split fields, or no split fields
      if (!IS_READ_ONLY(usage))
      {
        // We are modifying below in the sub-tree so all the fields 
        // should be split
#ifdef DEBUG_LEGION
        assert(version_mask == split_mask);
#endif
        // All fields from previous are current and all fields from 
        // current are advance, unless we are doing reductions so
        // we don't need to get anything from previous
        if (!IS_REDUCE(usage))
        {
          for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
                vit = previous_version_infos.begin(); vit !=
                previous_version_infos.end(); vit++)
          {
            FieldMask local_overlap = 
              vit->second.get_valid_mask() & version_mask;
            if (!local_overlap)
              continue;
            unversioned -= local_overlap;
            for (ManagerVersions::iterator it = vit->second.begin();
                  it != vit->second.end(); it++)
            {
              FieldMask overlap = it->second & local_overlap;
              if (!overlap)
                continue;
              version_info.add_current_version(it->first, overlap,
                                               true/*path only*/);
              it->first->request_final_version_state(overlap, ready_events);
            }
            if (!unversioned)
              break;
          }
          // We don't care about versioning information for writes because
          // they can modify the next version
          if (!!unversioned_mask)
            unversioned_mask.clear();
        }
        else if (!!unversioned_mask)
        {
          // If we are a reduction and we have previously unversioned
          // fields then we need to do a little check to make sure
          // that versions at least exist
          for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
                vit = previous_version_infos.begin(); vit != 
                previous_version_infos.end(); vit++)
          {
            // Don't need to capture the actual states, just need to 
            // remove all the fields for which we have a prior version
            unversioned_mask -= vit->second.get_valid_mask();
            if (!unversioned_mask)
              break;
          }
        }
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
              vit = current_version_infos.begin(); vit !=
              current_version_infos.end(); vit++)
        {
          FieldMask local_overlap = vit->second.get_valid_mask() & version_mask;
          if (!local_overlap)
            continue;
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_advance_version(it->first, overlap,
                                             true/*path only*/);
            // No need to request anything since we're contributing only
          }
        }
      }
      else if (!!split_mask)
      {
        // We are read-only with split fields that we have to deal with
        // Split fields we need the final version of previous, while
        // non-split fields we need final version of current, no need
        // for advance fields because we are read-only
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
              vit = previous_version_infos.begin(); vit !=
              previous_version_infos.end(); vit++)
        {
          FieldMask local_overlap = vit->second.get_valid_mask() & split_mask;
          if (!local_overlap)
            continue;
          unversioned -= local_overlap;
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_current_version(it->first, overlap,
                                             true/*path only*/);
            it->first->request_final_version_state(overlap, ready_events);
          }
          if (!unversioned)
            break;
        }
        FieldMask non_split = version_mask - split_mask;
        if (!!non_split)
        {
          for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
                vit = current_version_infos.begin(); vit !=
                current_version_infos.end(); vit++)
          {
            FieldMask local_overlap = vit->second.get_valid_mask() & non_split;
            if (!local_overlap)
              continue;
            for (ManagerVersions::iterator it = vit->second.begin();
                  it != vit->second.end(); it++)
            {
              FieldMask overlap = it->second & local_overlap;
              if (!overlap)
                continue;
              version_info.add_current_version(it->first, overlap,
                                               true/*path only*/);
              it->first->request_initial_version_state(overlap, ready_events);
            }
          } 
        }
        // Update the unversioned mask if necessary
        if (!!unversioned_mask)
        {
          if (!!unversioned)
            unversioned_mask &= unversioned;
          else
            unversioned_mask.clear();
        }
      }
      else
      {
        // We are read-only with no split fields so everything is easy
        // We do have to request the initial version of the states
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
              vit = current_version_infos.begin(); vit !=
              current_version_infos.end(); vit++)
        {
          FieldMask local_overlap = vit->second.get_valid_mask() & version_mask;
          if (!local_overlap)
            continue;
          unversioned -= local_overlap;
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_current_version(it->first, overlap, 
                                             true/*path only*/);
            it->first->request_initial_version_state(overlap, ready_events);
          }
          if (!unversioned)
            break;
        }
        // Update the unversioned mask if necessary
        if (!!unversioned_mask)
        {
          if (!!unversioned)
            unversioned_mask &= unversioned;
          else
            unversioned_mask.clear();
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_close_only_versions(
                                                const FieldMask &version_mask,
                                                SingleTask *context,
                                                Operation *op, unsigned index,
                                                const RegionUsage &usage,
                                                VersionInfo &version_info,
                                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // See if we have been assigned
      if (context != current_context)
      {
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = context;
      }
      // See if we are the owner
      if (!is_owner)
      {
        FieldMask request_mask = version_mask - remote_valid_fields;
        if (!!request_mask)
        {
          RtEvent wait_on = send_remote_version_request(request_mask,
                                                        ready_events); 
          wait_on.wait();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(version_mask - remote_valid_fields));
#endif
        }
      }     
      // We aren't mutating our data structures, so we just need 
      // the manager lock in read only mode
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      sanity_check();
#endif
      // We need the both previous and advance version numbers
      // We need the final versions of all the previous version so we
      // can see all the instances and reductions. We need the child
      // versions of the advance state so we can see all the open
      // children for when we make the composite instance
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator 
            vit = previous_version_infos.begin(); vit !=
            previous_version_infos.end(); vit++)
      {
        FieldMask local_overlap = vit->second.get_valid_mask() & version_mask;
        if (!local_overlap)
          continue;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          FieldMask overlap = it->second & local_overlap;
          if (!overlap)
            continue;
          version_info.add_current_version(it->first, overlap,
                                           false/*path only*/);
          it->first->request_final_version_state(overlap, ready_events);
        }
      }
      // If we've arrived, then we need to request all the children of
      // the advance state
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator
            vit = current_version_infos.begin(); vit != 
            current_version_infos.end(); vit++)
      {
        FieldMask local_overlap = version_mask & vit->second.get_valid_mask();
        if (!local_overlap)
          continue;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          FieldMask overlap = it->second & local_overlap;
          if (!overlap)
            continue;
          version_info.add_advance_version(it->first, overlap,
                                           false/*path only*/);
          it->first->request_children_version_state(overlap, ready_events);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::advance_versions(FieldMask mask, SingleTask *context, 
                                          bool update_parent_state,
                                          AddressSpaceID source_space,
                                          std::set<RtEvent> &applied_events,
                                          bool dedup_opens,
                                          ProjectionEpochID open_epoch,
                                          bool dedup_advances, 
                                          ProjectionEpochID advance_epoch)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime, 
                        CURRENT_STATE_ADVANCE_VERSION_NUMBERS_CALL);
      // See if we have been assigned
      if (context != current_context)
      {
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = context;
      }
      // If we are deduplicating advances, do that now
      // to see if we can avoid any communication
      if (dedup_opens || dedup_advances)
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        if (dedup_opens)
        {
          LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator
            finder = previous_opens.find(open_epoch);
          if (finder != previous_opens.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
          }
        }
        if (dedup_advances)
        {
          LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator 
            finder = previous_advancers.find(advance_epoch);
          if (finder != previous_advancers.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
          }
        }
      }
      // Check to see if we are the owner
      if (!is_owner)
      {
        // First send back the message to the owner to do the advance there
        RtEvent advanced = send_remote_advance(mask, update_parent_state,
                                               dedup_opens, open_epoch, 
                                               dedup_advances, advance_epoch);
        applied_events.insert(advanced); 
        // Now filter out any of our current version states that are
        // no longer valid for the given fields
        invalidate_version_infos(mask);
        return;
      }
      WrapperReferenceMutator mutator(applied_events);
      // If we have to update our parent version info, then we
      // need to keep track of all the new VersionState objects that we make
      VersioningSet<> new_states;
      // We need the lock in exclusive mode because we are going
      // to be mutating our data structures
      {
        AutoLock m_lock(manager_lock);
        // Recheck for any advance or open fields in case we lost the race
        if (dedup_opens)
        {
          LegionMap<ProjectionEpochID,FieldMask>::aligned::iterator
            finder = previous_opens.find(open_epoch);
          if (finder != previous_opens.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
            finder->second |= mask;
          }
          else
            previous_opens[open_epoch] = mask;
        }
        if (dedup_advances)
        {
          LegionMap<ProjectionEpochID,FieldMask>::aligned::iterator 
            finder = previous_advancers.find(advance_epoch);
          if (finder != previous_advancers.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
            finder->second |= mask;
          }
          else
            previous_advancers[advance_epoch] = mask;
        }
        // Filter out any previous opens if necessary
        if (!previous_opens.empty())
        {
          std::vector<ProjectionEpochID> to_delete;
          for (LegionMap<ProjectionEpochID,FieldMask>::aligned::iterator it =
                previous_opens.begin(); it != previous_opens.end(); it++)
          {
            if ((dedup_opens && (it->first == open_epoch)) ||
                (dedup_advances && (it->first == advance_epoch)))
              continue;
            it->second -= mask;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<ProjectionEpochID>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              previous_opens.erase(*it);
          }
        }
        // Filter out any previous advancers if necessary  
        if (!previous_advancers.empty())
        {
          std::vector<ProjectionEpochID> to_delete;
          for (LegionMap<ProjectionEpochID,FieldMask>::aligned::iterator it = 
                previous_advancers.begin(); it != 
                previous_advancers.end(); it++)
          {
            if ((dedup_opens && (it->first == open_epoch)) ||
                (dedup_advances && (it->first == advance_epoch)))
              continue;
            it->second -= mask;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<ProjectionEpochID>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              previous_advancers.erase(*it);
          }
        }
        // Now send any invalidations to get them in flight
        if (!remote_valid.empty() && !(remote_valid_fields * mask))
        {
          std::vector<AddressSpaceID> to_delete;
          for (LegionMap<AddressSpaceID,FieldMask>::aligned::iterator it = 
                remote_valid.begin(); it != remote_valid.end(); it++)
          {
            // If this is the source, then we can skip it
            // because it already knows to do its own invalidate
            if (it->first == source_space)
              continue;
            FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            RtEvent done = send_remote_invalidate(it->first, overlap);
            applied_events.insert(done); 
            it->second -= mask;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<AddressSpaceID>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              remote_valid.erase(*it);
          }
        }
        // Otherwise we are the owner node so we can do the update
#ifdef DEBUG_LEGION
        sanity_check();
#endif
        // First filter out fields in the previous
        std::vector<VersionID> to_delete_previous;
        FieldMask previous_filter = mask;
        for (LegionMap<VersionID,ManagerVersions>::aligned::iterator vit =
              previous_version_infos.begin(); vit != 
              previous_version_infos.end(); vit++)
        {
          FieldMask overlap = vit->second.get_valid_mask() & previous_filter;
          if (!overlap)
            continue;
          ManagerVersions &info = vit->second;
          // See if everyone is going away or just some of them
          if (overlap == info.get_valid_mask())
          {
            // The whole version number is going away, remove all
            // the valid references on the version state objects
            to_delete_previous.push_back(vit->first);
          }
          else
          {
            // Only some of the state are being filtered
            std::vector<VersionState*> to_delete;
            for (ManagerVersions::iterator it = info.begin();
                  it != info.end(); it++)
            {
              it->second -= overlap;
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<VersionState*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
                info.erase(*it);
            }
          }
          previous_filter -= overlap;
          if (!previous_filter)
            break;
        }
        if (!to_delete_previous.empty())
        {
          for (std::vector<VersionID>::const_iterator it = 
                to_delete_previous.begin(); it != 
                to_delete_previous.end(); it++)
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
        for (LegionMap<VersionID,ManagerVersions>::aligned::reverse_iterator 
              vit = current_version_infos.rbegin(); vit != 
              current_version_infos.rend(); vit++)
        {
          FieldMask version_overlap = 
            vit->second.get_valid_mask() & current_filter;
          if (!version_overlap)
            continue;
          ManagerVersions &info = vit->second;
          if (version_overlap == info.get_valid_mask())
          {
            // Send back the whole version state info to previous
            to_delete_current.insert(vit->first);
            // See if we need to merge it or can just copy it
            LegionMap<VersionID,ManagerVersions>::aligned::iterator 
              prev_finder = previous_version_infos.find(vit->first);
            if (prev_finder == previous_version_infos.end())
            {
              // Can just send it back with no merge
              info.move(previous_version_infos[vit->first]);
            }
            else
            {
              // prev_inf already existed
              // Filter back the version states
              for (ManagerVersions::iterator it = info.begin();
                    it != info.end(); it++)
                prev_finder->second.insert(it->first, it->second, &mutator);
            }
          }
          else
          {
            // Filter back only some of the version states
            std::vector<VersionState*> to_delete;
            ManagerVersions &prev_info = previous_version_infos[vit->first];
            for (ManagerVersions::iterator it = info.begin();
                  it != info.end(); it++)
            {
              FieldMask overlap = it->second & version_overlap;
              if (!overlap)
                continue;
              prev_info.insert(it->first, overlap, &mutator);
              it->second -= overlap;
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<VersionState*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
                info.erase(*it);
            }
          }
          // Make our new version state object and add if we can
          VersionID next_version = vit->first+1;
          // Remove this version number from the delete set if it exists
          to_delete_current.erase(next_version);
          VersionState *new_state = create_new_version_state(next_version);
          if (update_parent_state)
            new_states.insert(new_state, version_overlap, &mutator);
          // Kind of dangerous to be getting another iterator to this
          // data structure that we're iterating, but since neither
          // is mutating, we won't invalidate any iterators
          LegionMap<VersionID,ManagerVersions>::aligned::iterator 
            next_finder = current_version_infos.find(next_version);
          if (next_finder != current_version_infos.end())
            next_finder->second.insert(new_state, version_overlap, &mutator);
          else
            to_add[new_state] = version_overlap;
          current_filter -= version_overlap;
          if (!current_filter)
            break;
        }
        // See if we have any fields for which there was no prior
        // version number, if there was then these are fields which
        // are being initialized and should be added as version 1
        if (!!current_filter)
        {
          VersionState *new_state = create_new_version_state(init_version);
          if (update_parent_state)
            new_states.insert(new_state, current_filter, &mutator);
          current_version_infos[init_version].insert(new_state, 
                                                     current_filter, &mutator);
        }
        // Remove any old version state infos
        if (!to_delete_current.empty())
        {
          for (std::set<VersionID>::const_iterator it = 
                to_delete_current.begin(); it != to_delete_current.end(); it++)
            current_version_infos.erase(*it);
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
            current_version_infos[it->first->version_number].insert(
                it->first, it->second, &mutator);
          }
        }
#ifdef DEBUG_LEGION
        sanity_check();
#endif
      } // release the lock
      // If we recorded any new states then we have to tell our parent manager
      if (!new_states.empty())
      {
#ifdef DEBUG_LEGION
        assert(update_parent_state);
#endif
        RegionTreeNode *parent = node->get_parent();      
#ifdef DEBUG_LEGION
        assert(parent != NULL);
#endif
        VersionManager &parent_manager = 
          parent->get_current_version_manager(ctx);
        const ColorPoint &color = node->get_color();
        // This destroys new states, but whatever we're done anyway
        parent_manager.update_child_versions(context, color, 
                                             new_states, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::update_child_versions(SingleTask *context,
                                              const ColorPoint &child_color,
                                              VersioningSet<> &new_states,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // See if we have been assigned
      if (context != current_context)
      {
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = context;
      }
      // If we are not the owner, see if we need to issue any requests
      if (!is_owner)
      {
        FieldMask request_mask = 
          new_states.get_valid_mask() - remote_valid_fields;
        if (!!request_mask)
        {
          RtEvent wait_on = send_remote_version_request(
              new_states.get_valid_mask(), applied_events);
          wait_on.wait();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(new_states.get_valid_mask() - remote_valid_fields));
#endif
        }
      }
      WrapperReferenceMutator mutator(applied_events);
      // Only need to hold the lock in read-only mode since we're not
      // going to be updating any of these local data structures
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit = 
            current_version_infos.begin(); vit != 
            current_version_infos.end(); vit++)
      {
        FieldMask version_overlap = 
          vit->second.get_valid_mask() & new_states.get_valid_mask();
        if (!version_overlap)
          continue;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          FieldMask overlap = it->second & version_overlap;
          if (!overlap)
            continue;
          it->first->reduce_open_children(child_color, overlap, new_states, 
                                          &mutator, true/*need lock*/);
        }
        // If we've handled all the new states then we are done
        if (new_states.empty())
          break;
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::invalidate_version_infos(const FieldMask &invalid_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner); // should only be called on remote managers
#endif
      AutoLock m_lock(manager_lock);
      // This invalidates our local fields
      remote_valid_fields -= invalid_mask;
      filter_version_info<VERSION_MANAGER_REF>(invalid_mask, 
                                               current_version_infos);
      filter_version_info<VERSION_MANAGER_REF>(invalid_mask, 
                                               previous_version_infos);
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_SRC>
    /*static*/ void VersionManager::filter_version_info(const FieldMask &mask,
      typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned &to_filter)
    //--------------------------------------------------------------------------
    {
      std::vector<VersionID> to_delete;
      for (typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned::
            iterator vit = to_filter.begin(); vit != to_filter.end(); vit++)
      {
        FieldMask overlap = vit->second.get_valid_mask() & mask;
        if (!overlap)
          continue;
        if (overlap == vit->second.get_valid_mask())
          to_delete.push_back(vit->first);
        else
        {
          std::vector<VersionState*> to_remove;
          for (typename VersioningSet<REF_SRC>::iterator it = 
                vit->second.begin(); it != vit->second.end(); it++)
          {
            it->second -= overlap;
            if (!it->second)
              to_remove.push_back(it->first);
          }
          if (!to_remove.empty())
          {
            for (std::vector<VersionState*>::const_iterator it = 
                  to_remove.begin(); it != to_remove.end(); it++)
              vit->second.erase(*it);
          }
        }
      }
      if (!to_delete.empty())
      {
        for (std::vector<VersionID>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
          to_filter.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::print_physical_state(RegionTreeNode *arg_node,
                                const FieldMask &capture_mask,
                          LegionMap<ColorPoint,FieldMask>::aligned &to_traverse,
                                TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node == arg_node);
#endif
      PhysicalState temp_state(node, false/*dummy path only*/);
      logger->log("Versions:");
      logger->down();
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit =
            current_version_infos.begin(); vit != 
            current_version_infos.end(); vit++)
      {
        if (capture_mask * vit->second.get_valid_mask())
          continue;
        FieldMask version_fields;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          FieldMask overlap = capture_mask & it->second;
          if (!overlap)
            continue;
          version_fields |= overlap;
          VersionState *vs = dynamic_cast<VersionState*>(it->first);
          assert(vs != NULL);
          vs->update_physical_state(&temp_state, overlap);
        }
        assert(!!version_fields);
        char *version_buffer = version_fields.to_string();
        logger->log("%lld: %s", vit->first, version_buffer);
        free(version_buffer);
      }
      logger->up();
      temp_state.print_physical_state(capture_mask, to_traverse, logger);
    }

    //--------------------------------------------------------------------------
    VersionState* VersionManager::create_new_version_state(VersionID vid)
    //--------------------------------------------------------------------------
    {
      DistributedID did = runtime->get_available_distributed_id(false);
      return legion_new<VersionState>(vid, runtime, did, 
          runtime->address_space, runtime->address_space,
          node, true/*register now*/);
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::send_remote_advance(const FieldMask &advance_mask,
                                                bool update_parent_state,
                                                bool dedup_opens,
                                                ProjectionEpochID open_epoch,
                                                bool dedup_advances,
                                                ProjectionEpochID advance_epoch)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner);
#endif
      RtUserEvent remote_advance = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_advance);
        rez.serialize(current_context->get_context_uid());
        if (node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(node->as_partition_node()->handle);
        }
        rez.serialize(advance_mask);
        rez.serialize(update_parent_state);
        rez.serialize(dedup_opens);
        if (dedup_opens)
          rez.serialize(open_epoch);
        rez.serialize(dedup_advances);
        if (dedup_advances)
          rez.serialize(advance_epoch);
      }
      runtime->send_version_manager_advance(owner_space, rez);
      return remote_advance;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_remote_advance(Deserializer &derez,
                                  Runtime *runtime, AddressSpaceID source_space)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *node;
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
      FieldMask advance_mask;
      derez.deserialize(advance_mask);
      bool update_parent;
      derez.deserialize(update_parent);
      bool dedup_opens;
      derez.deserialize(dedup_opens);
      ProjectionEpochID open_epoch = 0;
      if (dedup_opens)
        derez.deserialize(dedup_opens);
      bool dedup_advances;
      derez.deserialize(dedup_advances);
      ProjectionEpochID advance_epoch = 0;
      if (dedup_advances)
        derez.deserialize(advance_epoch);

      SingleTask *context = runtime->find_context(context_uid);
      ContextID ctx = context->get_context_id();
      VersionManager &manager = node->get_current_version_manager(ctx);
      std::set<RtEvent> done_preconditions;
      manager.advance_versions(advance_mask, context, source_space, 
                               update_parent, done_preconditions, dedup_opens,
                               open_epoch, dedup_advances, advance_epoch);
      if (!done_preconditions.empty())
        Runtime::trigger_event(done_event, 
            Runtime::merge_events(done_preconditions));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::send_remote_invalidate(AddressSpaceID target,
                                               const FieldMask &invalidate_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner); // should only be called from the owner
#endif
      RtUserEvent remote_invalidate = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_invalidate);
        rez.serialize(current_context->get_context_uid());
        if (node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(node->as_partition_node()->handle);
        }
        rez.serialize(invalidate_mask);
      }
      runtime->send_version_manager_invalidate(target, rez);
      return remote_invalidate;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_remote_invalidate(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *node;
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
      FieldMask invalidate_mask;
      derez.deserialize(invalidate_mask);

      SingleTask *context = runtime->find_context(context_uid);
      ContextID ctx = context->get_context_id();
      VersionManager &manager = node->get_current_version_manager(ctx);
      manager.invalidate_version_infos(invalidate_mask);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::send_remote_version_request(FieldMask request_mask,
                                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner); // better be a remote copy
#endif
      RtUserEvent local_request;
      std::set<RtEvent> preconditions;
      {
        // First check to see what if any outstanding requests we have
        AutoLock m_lock(manager_lock);
        for (LegionMap<RtUserEvent,FieldMask>::aligned::const_iterator it =
              outstanding_requests.begin(); it != 
              outstanding_requests.end(); it++)
        {
          if (it->second * request_mask)
            continue;
          preconditions.insert(it->first);
          request_mask -= it->second;
          if (!request_mask)
            break;
        }
        if (!!request_mask)
        {
          local_request = Runtime::create_rt_user_event();
          outstanding_requests[local_request] = request_mask;
        }
      }
      if (!!request_mask)
      {
        // Send the local request
        Serializer rez;
        {
          RezCheck z(rez);
          // Send a pointer to this object for the response
          rez.serialize(this);
          // Need this for when we do the unpack
          rez.serialize(&ready_events);
          rez.serialize(local_request);
          rez.serialize(current_context->get_context_uid());
          if (node->is_region())
          {
            rez.serialize<bool>(true);
            rez.serialize(node->as_region_node()->handle);
          }
          else
          {
            rez.serialize<bool>(false);
            rez.serialize(node->as_partition_node()->handle);
          }
          rez.serialize(request_mask);
        }
        runtime->send_version_manager_request(owner_space, rez);
      }
      if (!preconditions.empty())
      {
        // Add the local request if there is one
        if (local_request.exists())
          preconditions.insert(local_request);
        return Runtime::merge_events(preconditions);
      }
#ifdef DEBUG_LEGION
      assert(local_request.exists()); // better have an event
#endif
      return local_request;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_request(Deserializer &derez,
                                  Runtime *runtime, AddressSpaceID source_space)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      VersionManager *remote_manager;
      derez.deserialize(remote_manager);
      std::set<RtEvent> *applied_events;
      derez.deserialize(applied_events);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *node;
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
      FieldMask request_mask;
      derez.deserialize(request_mask);

      SingleTask *context = runtime->find_context(context_uid);
      ContextID ctx = context->get_context_id();
      VersionManager &manager = node->get_current_version_manager(ctx);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_manager);
        rez.serialize(applied_events);
        rez.serialize(done_event);
        rez.serialize(request_mask);
        manager.pack_response(rez, source_space, request_mask);
      }
      runtime->send_version_manager_response(source_space, rez);
    }

    //--------------------------------------------------------------------------
    void VersionManager::pack_response(Serializer &rez, AddressSpaceID target,
                                       const FieldMask &request_mask)
    //--------------------------------------------------------------------------
    {
      // Do most of this in ready only mode
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        LegionMap<VersionState*,FieldMask>::aligned send_infos;
        // Do the current infos first
        find_send_infos<VERSION_MANAGER_REF>(current_version_infos, 
                                             request_mask, send_infos);
        pack_send_infos(rez, send_infos);
        send_infos.clear();
        // Then do the previous infos
        find_send_infos<VERSION_MANAGER_REF>(previous_version_infos, 
                                             request_mask, send_infos);
        pack_send_infos(rez, send_infos);
      }
      // Need exclusive access at the end to update the remote information
      AutoLock m_lock(manager_lock);
      remote_valid_fields |= request_mask;
      remote_valid[target] |= request_mask;
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_SRC>
    /*static*/ void VersionManager::find_send_infos(
          typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned& 
            version_infos, const FieldMask &request_mask, 
          LegionMap<VersionState*,FieldMask>::aligned& send_infos)
    //--------------------------------------------------------------------------
    {
      for (typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned::
            const_iterator vit = version_infos.begin(); 
            vit != version_infos.end(); vit++)
      {
        FieldMask overlap = vit->second.get_valid_mask() & request_mask;
        if (!overlap)
          continue;
        for (typename VersioningSet<REF_SRC>::iterator it = 
              vit->second.begin(); it != vit->second.end(); it++)
        {
          FieldMask state_overlap = it->second & overlap;
          if (!state_overlap)
            continue;
          send_infos[it->first] = state_overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::pack_send_infos(Serializer &rez, const
        LegionMap<VersionState*,FieldMask>::aligned& send_infos)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(send_infos.size());
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
            send_infos.begin(); it != send_infos.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::unpack_response(Deserializer &derez, RtUserEvent done,
                                         const FieldMask &update_mask,
                                         std::set<RtEvent> *applied_events)
    //--------------------------------------------------------------------------
    {
      // Unpack all our version states
      LegionMap<VersionState*,FieldMask>::aligned current_update;
      LegionMap<VersionState*,FieldMask>::aligned previous_update;
      std::set<RtEvent> preconditions;
      unpack_send_infos(derez, current_update, runtime, preconditions);
      unpack_send_infos(derez, previous_update, runtime, preconditions);
      WrapperReferenceMutator mutator(*applied_events);
      // If we have any preconditions here we must wait
      if (!preconditions.empty())
      {
        RtEvent wait_on = Runtime::merge_events(preconditions);
        wait_on.wait();
      }
      // Take our lock and apply our updates
      {
        AutoLock m_lock(manager_lock);
        merge_send_infos<VERSION_MANAGER_REF>(current_version_infos, 
                                              current_update, &mutator);
        merge_send_infos<VERSION_MANAGER_REF>(previous_version_infos, 
                                              previous_update, &mutator);
        // Update the remote valid fields
        remote_valid_fields |= update_mask;
        // Remove our outstanding request
#ifdef DEBUG_LEGION
        assert(outstanding_requests.find(done) != outstanding_requests.end());
#endif
        outstanding_requests.erase(done);
      }
      // Now we can trigger our done event
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::unpack_send_infos(Deserializer &derez,
                            LegionMap<VersionState*,FieldMask>::aligned &infos,
                            Runtime *runtime, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      size_t num_states;
      derez.deserialize(num_states);
      for (unsigned idx = 0; idx < num_states; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        RtEvent ready;
        VersionState *state = runtime->find_or_request_version_state(did,ready);
        if (ready.exists())
          preconditions.insert(ready);
        infos[state] = valid_mask;
      }
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_SRC>
    /*static*/ void VersionManager::merge_send_infos(
        typename LegionMap<VersionID,VersioningSet<REF_SRC> >::aligned&
          target_infos,
        const LegionMap<VersionState*,FieldMask>::aligned &source_infos,
        ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (typename LegionMap<VersionState*,FieldMask>::aligned::const_iterator
            it = source_infos.begin(); it != source_infos.end(); it++)
        target_infos[it->first->version_number].insert(it->first, it->second);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      VersionManager *local_manager;
      derez.deserialize(local_manager);
      std::set<RtEvent> *applied_events;
      derez.deserialize(applied_events);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      FieldMask update_mask;
      derez.deserialize(update_mask);
      local_manager->unpack_response(derez, done_event, 
                                     update_mask, applied_events);
    }

    //--------------------------------------------------------------------------
    void VersionManager::sanity_check(void)
    //--------------------------------------------------------------------------
    {
      // This code is a sanity check that each field appears for at most
      // one version number
      FieldMask current_version_fields;
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit =
            current_version_infos.begin(); vit != 
            current_version_infos.end(); vit++)
      {
        const ManagerVersions &info = vit->second;
        assert(!!info.get_valid_mask());
        // Make sure each field appears once in each version state info
        FieldMask local_version_fields;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          assert(!!it->second); // better not be empty
          assert(local_version_fields * it->second); // better not overlap
          local_version_fields |= it->second;
        }
        assert(info.get_valid_mask() == local_version_fields); // beter match
        // Should not overlap with other fields in the current version
        assert(current_version_fields * info.get_valid_mask());
        current_version_fields |= info.get_valid_mask();
      }
      FieldMask previous_version_fields;
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit =
            previous_version_infos.begin(); vit != 
            previous_version_infos.end(); vit++)
      {
        const ManagerVersions &info = vit->second;
        assert(!!info.get_valid_mask());
        // Make sure each field appears once in each version state info
        FieldMask local_version_fields;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          assert(!!it->second); // better not be empty
          assert(local_version_fields * it->second); // better not overlap
          local_version_fields |= it->second;
        }
        assert(info.get_valid_mask() == local_version_fields); // beter match
        // Should not overlap with other fields in the current version
        assert(previous_version_fields * info.get_valid_mask());
        previous_version_fields |= info.get_valid_mask();
      }
    }

    /////////////////////////////////////////////////////////////
    // Version State 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionState::VersionState(VersionID vid, Runtime *rt, DistributedID id,
                               AddressSpaceID own_sp, AddressSpaceID local_sp, 
                               RegionTreeNode *node, bool register_now)
      : DistributedCollectable(rt, id, own_sp, local_sp, register_now), 
        version_number(vid), logical_node(node), 
        state_lock(Reservation::create_reservation())
#ifdef DEBUG_LEGION
        , currently_active(true), currently_valid(true)
#endif
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, add a valid reference
      if (!is_owner())
        add_base_valid_ref(REMOTE_DID_REF);
#ifdef LEGION_GC
      log_garbage.info("GC Version State %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    VersionState::VersionState(const VersionState &rhs)
      : DistributedCollectable(rhs.runtime, rhs.did, rhs.local_space,
                               rhs.owner_space, false/*register now*/),
        version_number(0), logical_node(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VersionState::~VersionState(void)
    //--------------------------------------------------------------------------
    {
      if (is_owner() && registered_with_runtime)
        unregister_with_runtime(DEFAULT_VIRTUAL_CHANNEL);
      state_lock.destroy_reservation();
      state_lock = Reservation::NO_RESERVATION;
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(!currently_valid);
#endif 
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
      valid_fields |= user_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::update_path_only_state(PhysicalState *state,
                                             const FieldMask &update_mask) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
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
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_UPDATE_PATH_ONLY_CALL);
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
    void VersionState::merge_physical_state(const PhysicalState *state,
                                            const FieldMask &merge_mask,
                                            AddressSpaceID target,
                                          std::set<RtEvent> &applied_conditions,
                                            bool need_lock /* = true*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
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
      valid_fields |= merge_mask;
      if (need_lock)
        state_lock.release();
    }

    //--------------------------------------------------------------------------
    void VersionState::reduce_open_children(const ColorPoint &child_color,
                                            const FieldMask &update_mask,
                                            VersioningSet<> &new_states,
                                            ReferenceMutator *mutator,
                                            bool need_lock/*=true*/)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        // Hold the lock in exclusive mode since we might change things
        AutoLock s_lock(state_lock);
        reduce_open_children(child_color, update_mask, new_states, 
                             mutator, false/*need lock*/);
        return;
      }
      LegionMap<ColorPoint,StateVersions>::aligned::iterator finder =
        open_children.find(child_color);
      // We only have to do insertions if the entry didn't exist before
      // or its fields are disjoint with the update mask
      if ((finder == open_children.end()) || 
          (finder->second.get_valid_mask() * update_mask))
      {
        // Otherwise we can just insert these
        // but we only insert the ones for our fields
        StateVersions &local_states = (finder == open_children.end()) ? 
          open_children[child_color] : finder->second;
        for (StateVersions::iterator it = local_states.begin();
              it != local_states.end(); it++)
        {
          FieldMask overlap = it->second & update_mask;
          if (!overlap)
            continue;
          local_states.insert(it->first, overlap, mutator);
        }
      }
      else
        finder->second.reduce(update_mask, new_states, mutator);
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
      // Views have their valid references added when they are inserted 
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      // Should be monotonic
      assert(currently_valid);
      currently_valid = false;
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
      // We can clear out our open children since we don't need them anymore
      // which will also remove the valid references
      open_children.clear();
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
    template<VersionState::VersionRequestKind KIND>
    void VersionState::RequestFunctor<KIND>::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Skip the requestor
      if (target == requestor)
        return;
      RtUserEvent ready_event = Runtime::create_rt_user_event();
      proxy_this->send_version_state_update_request(target, requestor, 
          ready_event, mask, KIND);
      preconditions.insert(ready_event);
    }

    //--------------------------------------------------------------------------
    void VersionState::request_children_version_state(const FieldMask &req_mask,
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_REQUEST_CHILDREN_CALL);
      if (is_owner())
      {
        // We are the owner node so see if we need to do any reequests
        // to remote nodes to get our valid data
        if (has_remote_instances())
        {
          // Figure out which fields we need to request
          FieldMask remaining_mask = req_mask;
          AutoLock s_lock(state_lock);
          for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                child_events.begin(); it != child_events.end(); it++)
          {
            FieldMask overlap = it->second & remaining_mask;
            if (!overlap)
              continue;
            preconditions.insert(it->first);
            remaining_mask -= overlap;
            if (!remaining_mask)
              return; 
          }
          // If we still have remaining fields, we have to send requests to
          // all the other nodes asking for their data
          if (!!remaining_mask)
          {
            std::set<RtEvent> local_preconditions;
            RequestFunctor<CHILD_VERSION_REQUEST> functor(this, local_space, 
                                        remaining_mask, local_preconditions);
            map_over_remote_instances(functor);
            RtEvent ready_event = Runtime::merge_events(local_preconditions);
            child_events[ready_event] = remaining_mask;
            preconditions.insert(ready_event);
          }
        }
        // otherwise we're the only copy so there is nothing to do
      }
      else
      {
        FieldMask remaining_mask = req_mask;
        // We are not the owner so figure out which fields we still need to
        // send a request for
        AutoLock s_lock(state_lock);
        for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
              child_events.begin(); it != child_events.end(); it++)
        {
          FieldMask overlap = it->second & remaining_mask;
          if (!overlap)
            continue;
          preconditions.insert(it->first);
          remaining_mask -= overlap;
          if (!remaining_mask)
            return;
        }
        if (!!remaining_mask)
        {
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          send_version_state_update_request(owner_space, local_space, 
              ready_event, remaining_mask, CHILD_VERSION_REQUEST);
          // Save the event indicating when the fields will be ready
          final_events[ready_event] = remaining_mask;
          preconditions.insert(ready_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::request_initial_version_state(
                const FieldMask &request_mask, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_REQUEST_INITIAL_CALL);
      FieldMask needed_fields;
      {
        AutoLock s_lock(state_lock,1,false/*exclusive*/);
        // See if we have initial data
        needed_fields = request_mask - valid_fields;
      }
      if (!needed_fields)
        return;
      if (is_owner())
      {
        // We're the owner, if we have remote copies then send a 
        // request to them for the needed fields
        if (has_remote_instances())
        {
          AutoLock s_lock(state_lock);
          for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                initial_events.begin(); it != initial_events.end(); it++)
          {
            FieldMask overlap = it->second & needed_fields;
            if (!overlap)
              continue;
            preconditions.insert(it->first);
            needed_fields -= overlap;
            if (!needed_fields)
              return; 
          }
          // If we still have remaining fields, we have to send requests to
          // all the other nodes asking for their data
          if (!!needed_fields)
          {
            std::set<RtEvent> local_preconditions;
            RequestFunctor<INITIAL_VERSION_REQUEST> functor(this, local_space,
                                          needed_fields, local_preconditions);
            map_over_remote_instances(functor);
            RtEvent ready_event = Runtime::merge_events(local_preconditions);
            child_events[ready_event] = needed_fields;
            preconditions.insert(ready_event);
          }
        }
        // Otherwise no one has anything so we are done
      }
      else
      {
        AutoLock s_lock(state_lock);
        // Figure out which requests we haven't sent yet
        for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
              initial_events.begin(); it != initial_events.end(); it++)
        {
          FieldMask overlap = needed_fields & it->second;
          if (!overlap)
            continue;
          preconditions.insert(it->first);
          needed_fields -= overlap;
          if (!needed_fields)
            return;
        }
        // If we still have remaining fields, make a new event and 
        // send a request to the intial owner
        if (!!needed_fields)
        {
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          send_version_state_update_request(owner_space, local_space, 
              ready_event, needed_fields, INITIAL_VERSION_REQUEST);
          // Save the event indicating when the fields will be ready
          initial_events[ready_event] = needed_fields;
          preconditions.insert(ready_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::request_final_version_state(const FieldMask &req_mask,
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_REQUEST_FINAL_CALL);
      if (is_owner())
      {
        // We are the owner node so see if we need to do any reequests
        // to remote nodes to get our valid data
        if (has_remote_instances())
        {
          // Figure out which fields we need to request
          FieldMask remaining_mask = req_mask;
          AutoLock s_lock(state_lock);
          for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                final_events.begin(); it != final_events.end(); it++)
          {
            FieldMask overlap = it->second & remaining_mask;
            if (!overlap)
              continue;
            preconditions.insert(it->first);
            remaining_mask -= overlap;
            if (!remaining_mask)
              return; 
          }
          // If we still have remaining fields, we have to send requests to
          // all the other nodes asking for their data
          if (!!remaining_mask)
          {
            std::set<RtEvent> local_preconditions;
            RequestFunctor<FINAL_VERSION_REQUEST> functor(this, local_space, 
                                        remaining_mask, local_preconditions);
            map_over_remote_instances(functor);
            RtEvent ready_event = Runtime::merge_events(local_preconditions);
            final_events[ready_event] = remaining_mask;
            preconditions.insert(ready_event);
          }
        }
        // otherwise we're the only copy so there is nothing to do
      }
      else
      {
        FieldMask remaining_mask = req_mask;
        // We are not the owner so figure out which fields we still need to
        // send a request for
        AutoLock s_lock(state_lock); 
        for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
              final_events.begin(); it != final_events.end(); it++)
        {
          FieldMask overlap = it->second & remaining_mask;
          if (!overlap)
            continue;
          preconditions.insert(it->first);
          remaining_mask -= overlap;
          if (!remaining_mask)
            return;
        }
        if (!!remaining_mask)
        {
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          send_version_state_update_request(owner_space, local_space, 
              ready_event, remaining_mask, FINAL_VERSION_REQUEST);
          // Save the event indicating when the fields will be ready
          final_events[ready_event] = remaining_mask;
          preconditions.insert(ready_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state_update(AddressSpaceID target,
                                                const FieldMask &request_mask,
                                                VersionRequestKind request_kind,
                                                RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_SEND_STATE_CALL);
      Serializer rez;
      RezCheck z(rez);
      rez.serialize(did);
      rez.serialize(to_trigger);
      rez.serialize(request_mask);
      rez.serialize(request_kind);
      // Hold the lock in read-only mode while iterating these structures
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      // See if we should send all the fields or just do a partial send
      if (!(valid_fields - request_mask))
      {
        // Send everything
        if (request_kind != CHILD_VERSION_REQUEST)
        {
          rez.serialize(dirty_mask);
          rez.serialize(reduction_mask);
        }
        // Only send this if request is for child or final
        if (request_kind != INITIAL_VERSION_REQUEST)
        {
          rez.serialize<size_t>(open_children.size());
          for (LegionMap<ColorPoint,StateVersions>::aligned::const_iterator cit =
                open_children.begin(); cit != open_children.end(); cit++)
          {
            rez.serialize(cit->first);
            rez.serialize<size_t>(cit->second.size());
            for (StateVersions::iterator it = cit->second.begin();
                  it != cit->second.end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
          }
        }
        // Only send this if we are not doing child
        if (request_kind != CHILD_VERSION_REQUEST)
        {
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
      }
      else
      {
        // Partial send
        if (request_kind != CHILD_VERSION_REQUEST)
        {
          rez.serialize(dirty_mask & request_mask);
          rez.serialize(reduction_mask & request_mask);
        }
        if (request_kind != INITIAL_VERSION_REQUEST)
        {
          if (!open_children.empty())
          {
            Serializer child_rez;
            size_t count = 0;
            for (LegionMap<ColorPoint,StateVersions>::aligned::const_iterator 
                 cit = open_children.begin(); cit != open_children.end(); cit++)
            {
              FieldMask overlap = cit->second.get_valid_mask() & request_mask;
              if (!overlap)
                continue;
              child_rez.serialize(cit->first);
              Serializer state_rez;
              size_t state_count = 0;
              for (StateVersions::iterator it = cit->second.begin();
                    it != cit->second.end(); it++)
              {
                FieldMask state_overlap = it->second & overlap;
                if (!state_overlap)
                  continue;
                state_rez.serialize(it->first->did);
                state_rez.serialize(state_overlap);
                state_count++;
              }
              child_rez.serialize(state_count);
              child_rez.serialize(state_rez.get_buffer(), 
                                  state_rez.get_used_bytes());
              count++;
            }
            rez.serialize(count);
            rez.serialize(child_rez.get_buffer(), child_rez.get_used_bytes());
          }
          else
            rez.serialize<size_t>(0);
        }
        if (request_kind != CHILD_VERSION_REQUEST)
        {
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
      runtime->send_version_state_update_response(target, rez);
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state_update_request(AddressSpaceID target,
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
      runtime->send_version_state_update_request(target, rez);
    }

    //--------------------------------------------------------------------------
    void VersionState::launch_send_version_state_update(AddressSpaceID target,
                                                RtUserEvent to_trigger, 
                                                const FieldMask &request_mask, 
                                                VersionRequestKind request_kind,
                                                RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      SendVersionStateArgs args;
      args.hlr_id = HLR_SEND_VERSION_STATE_UPDATE_TASK_ID;
      args.proxy_this = this;
      args.target = target;
      args.request_mask = legion_new<FieldMask>(request_mask);
      args.request_kind = request_kind;
      args.to_trigger = to_trigger;
      runtime->issue_runtime_meta_task(&args, sizeof(args),
                                       HLR_SEND_VERSION_STATE_UPDATE_TASK_ID,
                                       HLR_LATENCY_PRIORITY,
                                       NULL/*op*/, precondition);
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(version_number);
        if (logical_node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(logical_node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(logical_node->as_partition_node()->handle);
        }
      }
      runtime->send_version_state_response(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::handle_version_state_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      VersionState *state = dynamic_cast<VersionState*>(dc);
      assert(state != NULL);
#else
      VersionState *state = static_cast<VersionState*>(dc);
#endif
      state->send_version_state(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::handle_version_state_response(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      VersionID version_number;
      derez.deserialize(version_number);
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *node;
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
      void *location;
      VersionState *state = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        state = legion_new_in_place<VersionState>(location, version_number,
                                                  runtime, did, source, 
                                                  runtime->address_space,
                                                  node, false/*register now*/);
      else
        state = legion_new<VersionState>(version_number, runtime, did,
                                         source, runtime->address_space,
                                         node, false/*register now*/);
      // Once construction is complete then we do the registration
      state->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_update_request(
                    AddressSpaceID source, RtUserEvent to_trigger, 
                    VersionRequestKind request_kind, FieldMask &request_mask)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_HANDLE_REQUEST_CALL);
      // If we are the not the owner, the same thing happens no matter what
      if (!is_owner())
      {
        // If we are not the owner, all we have to do is replay with our state
        AutoLock s_lock(state_lock,1,false/*exclusive*/);
        if (request_kind == CHILD_VERSION_REQUEST)
        {
          // See if we have any children we need to send
          bool has_children = false;
          for (LegionMap<ColorPoint,StateVersions>::aligned::const_iterator it =
                open_children.begin(); it != open_children.end(); it++)
          {
            if (it->second.get_valid_mask() * request_mask)
              continue;
            has_children = true;
            break;
          }
          if (has_children)
            launch_send_version_state_update(source, to_trigger, 
                                             request_mask, request_kind);
          else // no children so we can trigger now
            Runtime::trigger_event(to_trigger);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert((request_kind == INITIAL_VERSION_REQUEST) || 
                 (request_kind == FINAL_VERSION_REQUEST));
#endif
          // Check to see if we have valid fields, if not we
          // can trigger the event immediately
          FieldMask overlap = valid_fields & request_mask;
          if (!overlap)
            Runtime::trigger_event(to_trigger);
          else
            launch_send_version_state_update(source, to_trigger, 
                                             overlap, request_kind);
        }
      }
      else
      {
        // We're the owner, see if we're doing an initial requst or not 
        if (request_kind == INITIAL_VERSION_REQUEST)
        {
          // See if we have any remote instances
          if (has_remote_instances())
          {
            // Send notifications to any remote instances
            FieldMask needed_fields;
            RtEvent local_event;
            {
              // See if we have to send any messages
              AutoLock s_lock(state_lock,1,false/*exclusive*/);
              FieldMask overlap = request_mask & valid_fields;
              if (!!overlap)
              {
                needed_fields = request_mask - overlap;
                if (!!needed_fields)
                {
                  // We're going to have send messages so we need a local event
                  RtUserEvent local = Runtime::create_rt_user_event();
                  launch_send_version_state_update(source, local,
                                                   overlap, request_kind);
                  local_event = local;
                }
                else // no messages, so do the normal thing
                {
                  launch_send_version_state_update(source, to_trigger,
                                                   overlap, request_kind);
                  return; // we're done here
                }
              }
              else
                needed_fields = request_mask; // need all the fields
            }
#ifdef DEBUG_LEGION
            assert(!!needed_fields);
#endif
            std::set<RtEvent> local_preconditions;
            RequestFunctor<INITIAL_VERSION_REQUEST> functor(this, source,
                                      needed_fields, local_preconditions);
            map_over_remote_instances(functor);
            if (!local_preconditions.empty())
            {
              if (local_event.exists())
                local_preconditions.insert(local_event);
              Runtime::trigger_event(to_trigger,
                  Runtime::merge_events(local_preconditions));
            }
            else
            {
              // Sent no messages
              if (local_event.exists())
                Runtime::trigger_event(to_trigger, local_event);
              else
                Runtime::trigger_event(to_trigger);
            }
          }
          else
          {
            // No remote instances so handle ourselves locally only
            AutoLock s_lock(state_lock,1,false/*exclusive*/);
            FieldMask overlap = request_mask & valid_fields;
            if (!overlap)
              Runtime::trigger_event(to_trigger);
            else
              launch_send_version_state_update(source, to_trigger,
                                               overlap, request_kind);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert((request_kind == CHILD_VERSION_REQUEST) || 
                 (request_kind == FINAL_VERSION_REQUEST));
#endif
          // We're the owner handling a child or final version request
          // See if we have any remote instances to handle ourselves
          if (has_remote_instances())
          {
            std::set<RtEvent> local_preconditions;
            if (request_kind == CHILD_VERSION_REQUEST)
            {
              RequestFunctor<CHILD_VERSION_REQUEST> functor(this, source, 
                                      request_mask, local_preconditions);
              map_over_remote_instances(functor);
            }
            else
            {
              RequestFunctor<FINAL_VERSION_REQUEST> functor(this, source, 
                                      request_mask, local_preconditions);
              map_over_remote_instances(functor);
            }
            if (!local_preconditions.empty())
            {
              AutoLock s_lock(state_lock,1,false/*exclusive*/);
              FieldMask overlap = valid_fields & request_mask;
              if (!overlap)
              {
                RtUserEvent local_event = Runtime::create_rt_user_event();
                launch_send_version_state_update(source, local_event, 
                                                 overlap, request_kind);
                local_preconditions.insert(local_event);
              }
              Runtime::trigger_event(to_trigger,
                  Runtime::merge_events(local_preconditions));
            }
            else
            {
              // Didn't send any messages so do the normal path
              AutoLock s_lock(state_lock,1,false/*exclusive*/);
              FieldMask overlap = valid_fields & request_mask;
              if (!overlap)
                Runtime::trigger_event(to_trigger);
              else
                launch_send_version_state_update(source, to_trigger, 
                                                 overlap, request_kind);
            }
          }
          else // We just have to send our local state
          {
            AutoLock s_lock(state_lock,1,false/*exclusive*/);
            FieldMask overlap = valid_fields & request_mask;
            if (!overlap)
              Runtime::trigger_event(to_trigger);
            else
              launch_send_version_state_update(source, to_trigger, 
                                               overlap, request_kind);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_update_response(
                                                RtUserEvent to_trigger, 
                                                Deserializer &derez,
                                                const FieldMask &update,
                                                VersionRequestKind request_kind)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_HANDLE_RESPONSE_CALL);
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
        const bool in_place = !valid_fields;
        if (request_kind != CHILD_VERSION_REQUEST)
        {
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
        }
        if (request_kind != INITIAL_VERSION_REQUEST)
        {
          // Unpack the open children
          if (in_place)
          {
            size_t num_children;
            derez.deserialize(num_children);
            for (unsigned idx = 0; idx < num_children; idx++)
            {
              ColorPoint child;
              derez.deserialize(child);
              StateVersions &versions = open_children[child]; 
              size_t num_states;
              derez.deserialize(num_states);
              if (num_states == 0)
                continue;
              VersioningSet<> *deferred_children = NULL;
              std::set<RtEvent> deferred_children_events;
              for (unsigned idx2 = 0; idx2 < num_states; idx2++)
              {
                DistributedID did;
                derez.deserialize(did);
                RtEvent state_ready;
                VersionState *state = 
                  runtime->find_or_request_version_state(did, state_ready);
                FieldMask state_mask;
                derez.deserialize(state_mask);
                if (state_ready.exists())
                {
                  // We can't actually put this in the data 
                  // structure yet, because the object isn't ready
                  if (deferred_children == NULL)
                    deferred_children = new VersioningSet<>();
                  deferred_children->insert(state, state_mask);
                  deferred_children_events.insert(state_ready);
                }
                else
                  versions.insert(state, state_mask, &mutator); 
              }
              if (deferred_children != NULL)
              {
                // Launch a task to do the merge later
                UpdateStateReduceArgs args;
                args.hlr_id = HLR_UPDATE_VERSION_STATE_REDUCE_TASK_ID;
                args.proxy_this = this;
                args.child_color = child;
                // Takes ownership for deallocation
                args.children = deferred_children;
                RtEvent precondition = 
                  Runtime::merge_events(deferred_children_events);
                RtEvent done = runtime->issue_runtime_meta_task(&args, 
                    sizeof(args), HLR_UPDATE_VERSION_STATE_REDUCE_TASK_ID,
                    HLR_LATENCY_PRIORITY, NULL, precondition);
                preconditions.insert(done);
              }
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
              size_t num_states;
              derez.deserialize(num_states);
              if (num_states == 0)
                continue;
              VersioningSet<> *reduce_children = new VersioningSet<>();
              std::set<RtEvent> reduce_preconditions;
              FieldMask update_mask;
              for (unsigned idx2 = 0; idx2 < num_states; idx2++)
              {
                DistributedID did;
                derez.deserialize(did);
                RtEvent state_ready;
                VersionState *state = 
                  runtime->find_or_request_version_state(did, state_ready);
                if (state_ready.exists())
                  reduce_preconditions.insert(state_ready);
                FieldMask state_mask;
                derez.deserialize(state_mask);
                reduce_children->insert(state, state_mask);
                // As long as we aren't going to defer things, keep
                // updating the update mask
                if (reduce_preconditions.empty())
                  update_mask |= state_mask;
              }
              if (!reduce_preconditions.empty())
              {
                // Launch a task to do the merge later
                UpdateStateReduceArgs args;
                args.hlr_id = HLR_UPDATE_VERSION_STATE_REDUCE_TASK_ID;
                args.proxy_this = this;
                args.child_color = child;
                // Takes ownership for deallocation
                args.children = reduce_children;
                RtEvent precondition = 
                  Runtime::merge_events(reduce_preconditions);
                RtEvent done = runtime->issue_runtime_meta_task(&args,
                    sizeof(args), HLR_UPDATE_VERSION_STATE_REDUCE_TASK_ID,
                    HLR_LATENCY_PRIORITY, NULL, precondition);
                preconditions.insert(done);
              }
              else
              {
                // We can do the merge now
                reduce_open_children(child, update_mask, *reduce_children,
                                     &mutator, false/*need lock*/);
                delete reduce_children;
              }
            }
          }
        }
        if (request_kind != CHILD_VERSION_REQUEST)
        {
          // Finally do the views
          if (!valid_fields)
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
        // Finally update our valid mask from this transaction
        valid_fields |= update;
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
    /*static*/ void VersionState::process_version_state_reduction(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const UpdateStateReduceArgs *reduce_args = 
        (const UpdateStateReduceArgs*)args;
      // Compute the update mask
      FieldMask update_mask;
      for (VersioningSet<>::iterator it = reduce_args->children->begin();
            it != reduce_args->children->end(); it++)
        update_mask |= it->second;
      LocalReferenceMutator mutator;
      reduce_args->proxy_this->reduce_open_children(reduce_args->child_color,
          update_mask, *reduce_args->children, &mutator, true/*need lock*/);
      delete reduce_args->children;
      // Implicitly wait for events in mutator destructor
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

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_view_references(const void *args)
    //--------------------------------------------------------------------------
    {
      const UpdateViewReferences *view_args = (const UpdateViewReferences*)args;
      LocalReferenceMutator mutator;
      view_args->view->add_nested_valid_ref(view_args->did, &mutator);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_update_request(
                                               Runtime *rt, Deserializer &derez)
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
      vs->handle_version_state_update_request(source, to_trigger, 
                                              request_kind, request_mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_update_response(
                                               Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      FieldMask update_mask;
      derez.deserialize(update_mask);
      VersionRequestKind request_kind;
      derez.deserialize(request_kind);
      DistributedCollectable *target = rt->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_update_response(to_trigger, derez, 
                                               update_mask, request_kind);
    } 

    //--------------------------------------------------------------------------
    void VersionState::capture_root_instances(CompositeView *target, 
                                            const FieldMask &capture_mask) const
    //--------------------------------------------------------------------------
    {
      // Only need this in read only mode since we're just reading
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      FieldMask dirty_overlap = dirty_mask & capture_mask;
      if (!!dirty_overlap)
      {
        target->record_dirty_fields(dirty_overlap);
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = it->second & dirty_overlap;
          if (!overlap)
            continue;
          target->record_valid_view(it->first, overlap);
        }
      }
      FieldMask reduction_overlap = reduction_mask & capture_mask;
      if (!!reduction_overlap)
      {
        target->record_reduction_fields(reduction_overlap);
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
              reduction_views.begin(); it != reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & reduction_overlap;
          if (!overlap)
            continue;
          target->record_reduction_view(it->first, overlap);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::capture_root_children(CompositeView *target,
                                            const FieldMask &capture_mask) const
    //--------------------------------------------------------------------------
    {
      // Only need this in read only mode since we're just reading
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      for (LegionMap<ColorPoint,StateVersions>::aligned::const_iterator cit =
            open_children.begin(); cit != open_children.end(); cit++)
      {
        if (cit->second.get_valid_mask() * capture_mask)
          continue;
        for (StateVersions::iterator it = cit->second.begin();
              it != cit->second.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          target->record_child_version_state(cit->first, it->first, overlap);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::capture(CompositeNode *target,
           const FieldMask &capture_mask, SingleTask *translation_context) const
    //--------------------------------------------------------------------------
    {
      // Only need this in read only mode since we're just reading
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      FieldMask dirty_overlap = dirty_mask & capture_mask;
      if (!!dirty_overlap)
      {
        target->record_dirty_fields(dirty_overlap);
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = it->second & dirty_overlap;
          if (!overlap)
            continue;
          target->record_valid_view(it->first, overlap, translation_context);
        }
      }
      FieldMask reduction_overlap = reduction_mask & capture_mask;
      if (!!reduction_overlap)
      {
        target->record_reduction_fields(reduction_overlap);
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
              reduction_views.begin(); it != reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & reduction_overlap;
          if (!overlap)
            continue;
          target->record_reduction_view(it->first, overlap,translation_context);
        }
      }
      for (LegionMap<ColorPoint,StateVersions>::aligned::const_iterator cit =
            open_children.begin(); cit != open_children.end(); cit++)
      {
        if (cit->second.get_valid_mask() * capture_mask)
          continue;
        for (StateVersions::iterator it = cit->second.begin();
              it != cit->second.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          target->record_child_version_state(cit->first, it->first, overlap);
        }
      }
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
        else if (refs.single == NULL)
        {
          // New size is 1 but we were empty before
          CollectableRef *next = legion_new<CollectableRef>();
          next->add_reference();
          refs.single = next;
          single = true;
          shared = false;
        }
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

    /////////////////////////////////////////////////////////////
    // VersioningInvalidator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool VersioningInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_version_managers();
      return true;
    }

    //--------------------------------------------------------------------------
    bool VersioningInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_version_managers();
      return true;
    }

  }; // namespace Internal 
}; // namespace Legion

