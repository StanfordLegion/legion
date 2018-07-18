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
#include "legion/legion_trace.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_analysis.h"
#include "legion/legion_context.h"

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
    LogicalUser::LogicalUser(Operation *o, GenerationID g, unsigned id, 
                             const RegionUsage &u, const FieldMask &m)
      : GenericUser(u, m), op(o), idx(id), gen(g), timeout(TIMEOUT)
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
    PhysicalUser::PhysicalUser(const RegionUsage &u, const LegionColor c,
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
      PhysicalUser *result = new PhysicalUser();
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
        traversal_mask(k), context_uid(o->get_context()->get_context_uid()),
        map_applied_events(e), logical_ctx(-1U)
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
    bool VersioningSet<REF_KIND>::insert(VersionState *state, 
                               const FieldMask &mask, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      bool result = true;
      if (single)
      {
        if (versions.single_version == NULL)
        {
          versions.single_version = state;
          valid_fields = mask;
          if (REF_KIND != LAST_SOURCE_REF)
            state->add_base_valid_ref(REF_KIND, mutator);
        }
        else if (versions.single_version == state)
        {
          valid_fields |= mask;
          result = false;
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
            state->add_base_valid_ref(REF_KIND, mutator);
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
            state->add_base_valid_ref(REF_KIND, mutator);
        }
        else
        {
          finder->second |= mask;
          result = false;
        }
        valid_fields |= mask;
      }
      return result;
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
            if (pre.exists() && !pre.has_triggered())
            {
              VersioningSetRefArgs args;
              args.state = state;
              args.kind = REF_KIND;
              return runtime->issue_runtime_meta_task(args, 
                      LG_LATENCY_WORK_PRIORITY, pre);
            }
            else
            {
              LocalReferenceMutator mutator;
              state->add_base_valid_ref(REF_KIND, &mutator);
              return mutator.get_done_event();
            }
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
            if (pre.exists() && !pre.has_triggered())
            {
              VersioningSetRefArgs args;
              args.state = state;
              args.kind = REF_KIND;
              return runtime->issue_runtime_meta_task(args, 
                      LG_LATENCY_WORK_PRIORITY, pre);
            }
            else
            {
              LocalReferenceMutator mutator;
              state->add_base_valid_ref(REF_KIND, &mutator);
              return mutator.get_done_event();
            }
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
            if (pre.exists() && !pre.has_triggered())
            {
              VersioningSetRefArgs args;
              args.state = state;
              args.kind = REF_KIND;
              return runtime->issue_runtime_meta_task(args, 
                      LG_LATENCY_WORK_PRIORITY, pre);
            }
            else
            {
              LocalReferenceMutator mutator;
              state->add_base_valid_ref(REF_KIND, &mutator);
              return mutator.get_done_event();
            }
          }
        }
        else
          finder->second |= mask;
        valid_fields |= mask;
      }
      if (pre.exists())
        return pre;
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
        delete to_erase; 
    }

    //--------------------------------------------------------------------------
    template<ReferenceSource REF_KIND>
    void VersioningSet<REF_KIND>::clear(void)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((REF_KIND != LAST_SOURCE_REF) && (versions.single_version != NULL) 
            && versions.single_version->remove_base_valid_ref(REF_KIND))
          delete versions.single_version;
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
              delete it->first;
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
              const_cast<VersioningSet<REF_KIND>*>(this)), 
                                                    true/*single*/);
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
      for (typename VersioningSet<ARG_KIND>::iterator nit = 
            new_states.begin(); nit != new_states.end(); nit++)
      {
        LegionMap<VersionState*,FieldMask>::aligned to_add; 
        std::vector<VersionState*> to_erase_local;
        FieldMask overlap = merge_mask & nit->second;
        // This VersionState doesn't apply locally if there are no fields
        if (!overlap)
          continue;
        // We can remove these fields from the new states because
        // we know that we are going to handle it
        nit->second -= overlap;
        if (!nit->second)
          to_erase_new.push_back(nit->first);
        // Iterate over our states and see which ones interfere
        for (typename VersioningSet<REF_KIND>::iterator it = begin();
              it != end(); it++)
        {
          FieldMask local_overlap = it->second & overlap;
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
#endif  
          // Otherwise we keep the old one and throw away the new one
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
      split_masks[depth] |= split_mask;
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
        physical_states[depth] = new PhysicalState(node, path_only);
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
        physical_states[depth] = new PhysicalState(node, path_only);
      physical_states[depth]->add_advance_state(state, state_mask);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::set_upper_bound_node(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
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
    void VersionInfo::apply_mapping(std::set<RtEvent> &applied_conditions, 
                                    bool copy_through/*=false*/)
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
      physical_states[last_idx]->apply_state(applied_conditions);
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
    void VersionInfo::resize(size_t max_depth, HandleType req_handle,
                             ProjectionFunction *function)
    //--------------------------------------------------------------------------
    {
      // Path depth is twice the function depth because it counts region levels
      max_depth += (2*function->depth);
      // If it is a partition projection function we add one more
      // to get to the next region
      if (req_handle == PART_PROJECTION)
        max_depth += 1;
      resize(max_depth);
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
          delete *it;
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
    void VersionInfo::clone_to_depth(unsigned depth, const FieldMask &mask,
                               InnerContext *context, VersionInfo &target_info,
                               std::set<RtEvent> &ready_events) const
    //--------------------------------------------------------------------------
    {
      // If the upper bound nodes are the same, we are done
      const unsigned upper_depth = upper_bound_node->get_depth();
#ifdef DEBUG_LEGION
      assert(upper_depth <= depth);
#endif
      if (upper_depth == depth)
        return;
      // Update the upper bound node
      target_info.set_upper_bound_node(upper_bound_node);
      // Copy data into the target info
      for (unsigned idx = upper_depth; idx < depth; idx++)
      {
        const PhysicalState *state = physical_states[idx];
#ifdef DEBUG_LEGION
        assert(state != NULL);
#endif
        const FieldMask split_overlap = split_masks[idx] & mask;
        if (!!split_overlap)
          target_info.record_split_fields(state->node, split_overlap);
        // Also copy over the needed version states
        state->clone_to(mask, split_overlap, context, target_info,ready_events);
      }
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
      // We can make a physical state if it is below our upper bound node
      if ((result == NULL) && 
          (upper_bound_node->get_depth() <= node->get_depth()))
      {
        result = 
          new PhysicalState(node, (depth < (physical_states.size()-1)));
        result->capture_state();
        physical_states[depth] = result;
        return result;
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
      assert(result->node == node);
#endif
      if (!result->is_captured())
        result->capture_state();
      return result;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::get_field_versions(RegionTreeNode *node, bool split_prev,
                                         const FieldMask &needed_fields,
                                         FieldVersions &result_versions)
    //--------------------------------------------------------------------------
    {
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < field_versions.size());
      assert(depth < split_masks.size());
#endif
      const FieldMask &split_mask = split_masks[depth];
      const FieldVersions &local_versions = field_versions[depth];
      if (!split_prev || !split_mask)
      {
        // If we don't care about the split previous mask then we can
        // just copy over what we need
        for (FieldVersions::const_iterator it = local_versions.begin();
              it != local_versions.end(); it++)
        {
          const FieldMask overlap = needed_fields & it->second;
          if (!overlap)
            continue;
          result_versions[it->first] = overlap;
        }
      }
      else
      {
        // We need to save any fields that are needed, and we want
        // the previous version number for any split fields
        for (FieldVersions::const_iterator it = local_versions.begin();
              it != local_versions.end(); it++)
        {
          FieldMask overlap = needed_fields & it->second;
          if (!overlap)
            continue;
          FieldMask split_overlap = overlap & split_mask;
          if (!split_overlap)
          {
            result_versions[it->first] = overlap;
            continue;
          }
#ifdef DEBUG_LEGION
          assert(it->first > 0);
#endif
          result_versions[it->first - 1] = split_overlap;
          overlap -= split_overlap;
          if (!!overlap)
            result_versions[it->first] = overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::get_advance_versions(RegionTreeNode *node, bool base,
                                           const FieldMask &needed_fields,
                                           FieldVersions &result_versions)
    //--------------------------------------------------------------------------
    {
      const unsigned depth = node->get_depth();
#ifdef DEBUG_LEGION
      assert(depth < split_masks.size());
      assert(depth < field_versions.size());
#endif
      const FieldVersions &local_versions = field_versions[depth];
      if (base)
      {
        // Should be no split masks for base updates
#ifdef DEBUG_LEGION
        assert(!split_masks[depth]);
#endif
        // Bottom node with no split fields so therefore we need all
        // the fields advanced
        for (FieldVersions::const_iterator it = local_versions.begin();
              it != local_versions.end(); it++)
        {
          FieldMask overlap = needed_fields & it->second;
          if (!overlap)
            continue;
          result_versions[it->first+1] = overlap;
        }
      }
      else
      {
        // Above versions have already been advanced as reflected
        // by split fields
        for (FieldVersions::const_iterator it = local_versions.begin();
              it != local_versions.end(); it++)
        {
          FieldMask overlap = needed_fields & it->second;
          if (!overlap)
            continue;
          result_versions[it->first] = overlap;
        }
      }
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
    void VersionInfo::pack_version_info(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      pack_version_numbers(rez);
      if (upper_bound_node != NULL)
      {
        rez.serialize<size_t>(physical_states.size());
        bool is_region = upper_bound_node->is_region();
        for (std::vector<PhysicalState*>::const_iterator it = 
              physical_states.begin(); it != physical_states.end(); it++)
        {
          if ((*it) == NULL)
          {
            rez.serialize<bool>(true); // empty
            continue;
          }
          rez.serialize<bool>(false); // not empty
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
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_version_info(Deserializer &derez, 
                              Runtime *runtime, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      unpack_version_numbers(derez, runtime->forest);
      if (upper_bound_node != NULL)
      {
        size_t num_states;
        derez.deserialize(num_states);
        physical_states.resize(num_states, NULL);
        bool is_region = upper_bound_node->is_region();
        for (unsigned idx = 0; idx < num_states; idx++)
        {
          bool empty;
          derez.deserialize(empty);
          if (empty)
            continue;
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
          PhysicalState *next = new PhysicalState(node, is_path_only);
          next->unpack_physical_state(derez, runtime, ready_events);
          physical_states[idx] = next;
          // Reverse the polarity
          is_region = !is_region;
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_version_numbers(Serializer &rez) const
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
    void VersionInfo::pack_upper_bound_node(Serializer &rez) const
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
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it = 
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
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        if (it->first->remove_base_gc_ref(RESTRICTED_REF))
          delete it->first;
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
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it = 
            rhs.restrictions.begin(); it != rhs.restrictions.end(); it++)
      {
        it->first->add_base_gc_ref(RESTRICTED_REF);
        restrictions.insert(*it);
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::record_restriction(PhysicalManager *inst, 
                                          const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<PhysicalManager*,FieldMask>::aligned::iterator finder = 
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
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
        to_fill |= it->second;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++)
      {
        if (it->first->remove_base_gc_ref(RESTRICTED_REF))
          delete it->first;
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
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it = 
            restrictions.begin(); it != restrictions.end(); it++, idx++)
        restricted_instances[idx] = InstanceRef(it->first, it->second);
      return restricted_instances;
    }

    //--------------------------------------------------------------------------
    void RestrictInfo::pack_info(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(restrictions.size());
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it = 
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
        PhysicalManager *manager =  
          runtime->find_or_request_physical_manager(did, ready);
        derez.deserialize(restrictions[manager]);
        if (ready.exists() && !ready.has_triggered())
        {
          DeferRestrictedManagerArgs args;
          args.manager = manager;
          ready = runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, ready);
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
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator it =
            instances.begin(); it != instances.end(); it++)
      {
        if (it->first->remove_base_gc_ref(RESTRICTED_REF))
          delete it->first;
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
    void Restriction::add_restricted_instance(PhysicalManager *inst,
                                              const FieldMask &inst_fields)
    //--------------------------------------------------------------------------
    {
      // Always update the restricted fields
      restricted_fields |= inst_fields;
      LegionMap<PhysicalManager*,FieldMask>::aligned::iterator finder = 
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
        for (LegionMap<PhysicalManager*,FieldMask>::aligned::const_iterator
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
      std::vector<PhysicalManager*> to_delete;
      for (LegionMap<PhysicalManager*,FieldMask>::aligned::iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        it->second -= overlap;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      if (!to_delete.empty())
      {
        for (std::vector<PhysicalManager*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          instances.erase(*it);
          if ((*it)->remove_base_gc_ref(RESTRICTED_REF))
            delete *it;
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
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTIAL_ACQUIRE, 
                        "Illegal partial acquire operation (ID %lld) "
                        "performed in task %s (ID %lld)", op->get_unique_id(),
                        op->get_context()->get_task_name(),
                        op->get_context()->get_unique_id())
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
          break;
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
                             PhysicalManager *inst, FieldMask &remaining_fields)
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
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INTERFERING_RESTRICTON, 
                    "Illegal interfering restriction performed by attach "
                    "operation (ID %lld) in task %s (ID %lld)",
                    op->get_unique_op_id(), op->get_context()->get_task_name(),
                    op->get_context()->get_unique_id())
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
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INTERFERING_ACQUIRE, 
                    "Illegal interfering acquire operation performed by "
                    "acquire operation (ID %lld) in task %s (ID %lld)",
                    op->get_unique_op_id(), op->get_context()->get_task_name(),
                    op->get_context()->get_unique_id())
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
                          PhysicalManager *manager, FieldMask &remaining_fields)
    //--------------------------------------------------------------------------
    {
      FieldMask overlap = remaining_fields & acquired_fields;
      if (!overlap)
        return;
      if (!local_node->dominates(node))
      {
        if (local_node->intersects_with(node))
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTIAL_RESTRICTION, 
                        "Illegal partial restriction operation performed by "
                        "attach operation (ID %lld) in task %s (ID %lld)",
                        op->get_unique_op_id(), 
                        op->get_context()->get_task_name(),
                        op->get_context()->get_unique_id())
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
          break;
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
    // LogicalTraceInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalTraceInfo::LogicalTraceInfo(bool already_tr, LegionTrace *tr, 
                                       unsigned idx, const RegionRequirement &r)
      : already_traced(already_tr), trace(tr), req_idx(idx), req(r)
    //--------------------------------------------------------------------------
    {
      // If we have a trace but it doesn't handle the region tree then
      // we should mark that this is not part of a trace
      if ((trace != NULL) && 
          !trace->handles_region_tree(req.parent.get_tree_id()))
      {
        already_traced = false;
        trace = NULL;
      }
    }

    /////////////////////////////////////////////////////////////
    // ProjectionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionInfo::ProjectionInfo(Runtime *runtime, 
                      const RegionRequirement &req, IndexSpace launch_space)
      : projection((req.handle_type != SINGULAR) ? 
          runtime->find_projection_function(req.projection) : NULL),
        projection_type(req.handle_type),
        projection_space((req.handle_type != SINGULAR) ?
            runtime->forest->get_node(launch_space) : NULL),
        dirty_reduction(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::record_projection_epoch(ProjectionEpochID epoch,
                                                 const FieldMask &epoch_mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<ProjectionEpochID,FieldMask>::aligned::iterator finder = 
        projection_epochs.find(epoch);
      if (finder == projection_epochs.end())
        projection_epochs[epoch] = epoch_mask;
      else
        finder->second |= epoch_mask;
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      projection = NULL;
      projection_type = SINGULAR;
      projection_space = NULL;
      projection_epochs.clear();
      dirty_reduction = false;
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
      rez.serialize<bool>(dirty_reduction);
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::unpack_info(Deserializer &derez, Runtime *runtime,
                      const RegionRequirement &req, IndexSpaceNode *launch_node)
    //--------------------------------------------------------------------------
    {
      projection_type = req.handle_type;
      if (req.handle_type != SINGULAR)
      {
        projection = runtime->find_projection_function(req.projection);
        projection_space = launch_node; 
      }
      size_t num_epochs;
      derez.deserialize(num_epochs);
      for (unsigned idx = 0; idx < num_epochs; idx++)
      {
        ProjectionEpochID epoch_id;
        derez.deserialize(epoch_id);
        derez.deserialize(projection_epochs[epoch_id]);
      }
      derez.deserialize<bool>(dirty_reduction);
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::pack_epochs(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(projection_epochs.size());
      for (LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator it =
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionInfo::unpack_epochs(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(projection_epochs.empty());
#endif
      size_t num_epochs;
      derez.deserialize(num_epochs);
      for (unsigned idx = 0; idx < num_epochs; idx++)
      {
        ProjectionEpochID epoch;
        derez.deserialize(epoch);
        derez.deserialize(projection_epochs[epoch]);
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
    CurrentInvalidator::CurrentInvalidator(ContextID c, bool only)
      : ctx(c), users_only(only)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator::CurrentInvalidator(const CurrentInvalidator &rhs)
      : ctx(0), users_only(false)
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
      node->invalidate_current_state(ctx, users_only); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx, users_only);
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
    void ProjectionEpoch::insert(ProjectionFunction *function, 
                                 IndexSpaceNode* node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!valid_fields);
#endif
      projections[function].insert(node);
    }

    /////////////////////////////////////////////////////////////
    // LogicalState 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    LogicalState::LogicalState(RegionTreeNode *node, ContextID ctx)
      : owner(node)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalState::LogicalState(const LogicalState &rhs)
      : owner(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalState::~LogicalState(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalState& LogicalState::operator=(const LogicalState&rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LogicalState::check_init(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_states.empty());
      assert(curr_epoch_users.empty());
      assert(prev_epoch_users.empty());
      assert(projection_epochs.empty());
      assert(!dirty_fields);
      assert(!dirty_below);
      assert(!reduction_fields);
#endif
    }

    //--------------------------------------------------------------------------
    void LogicalState::clear_logical_users(void)
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
    void LogicalState::reset(void)
    //--------------------------------------------------------------------------
    {
      field_states.clear();
      clear_logical_users(); 
      dirty_below.clear();
      dirty_fields.clear();
      reduction_fields.clear();
      outstanding_reductions.clear();
      for (std::list<ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
        delete *it;
      projection_epochs.clear();
    } 

    //--------------------------------------------------------------------------
    void LogicalState::clear_deleted_state(const FieldMask &deleted_mask)
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
        std::vector<LegionColor> to_delete;
        for (LegionMap<LegionColor,FieldMask>::aligned::iterator child_it = 
              it->open_children.begin(); child_it != 
              it->open_children.end(); child_it++)
        {
          child_it->second -= deleted_mask;
          if (!child_it->second)
            to_delete.push_back(child_it->first);
        }
        if (!to_delete.empty())
        {
          for (std::vector<LegionColor>::const_iterator cit = to_delete.begin();
                cit != to_delete.end(); cit++)
            it->open_children.erase(*cit);
        }
        if (!it->open_children.empty())
          it++;
        else
          it = field_states.erase(it);
      }
      reduction_fields -= deleted_mask;
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
    void LogicalState::advance_projection_epochs(const FieldMask &advance_mask)
    //--------------------------------------------------------------------------
    {
      // See if we can get some coalescing going on here
      std::map<ProjectionEpochID,ProjectionEpoch*> to_add; 
      for (std::list<ProjectionEpoch*>::iterator it = 
            projection_epochs.begin(); it != 
            projection_epochs.end(); /*nothing*/)
      {
        FieldMask overlap = (*it)->valid_fields & advance_mask;
        if (!overlap)
        {
          it++;
          continue;
        }
        const ProjectionEpochID next_epoch_id = (*it)->epoch_id + 1;
        std::map<ProjectionEpochID,ProjectionEpoch*>::iterator finder = 
          to_add.find(next_epoch_id);
        if (finder == to_add.end())
        {
          ProjectionEpoch *next_epoch = 
            new ProjectionEpoch((*it)->epoch_id+1, overlap);
          to_add[next_epoch_id] = next_epoch;
        }
        else
          finder->second->valid_fields |= overlap;
        // Filter the fields from our old one
        (*it)->valid_fields -= overlap;
        if (!((*it)->valid_fields))
        {
          delete (*it);
          it = projection_epochs.erase(it);
        }
        else
          it++;
      }
      if (!to_add.empty())
      {
        for (std::map<ProjectionEpochID,ProjectionEpoch*>::const_iterator it = 
              to_add.begin(); it != to_add.end(); it++)
          projection_epochs.push_back(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::capture_projection_epochs(FieldMask capture_mask,
                                                 ProjectionInfo &info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!capture_mask);
#endif
      for (std::list<ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        FieldMask overlap = (*it)->valid_fields & capture_mask;
        if (!overlap)
          continue;
        info.record_projection_epoch((*it)->epoch_id, overlap);
        capture_mask -= overlap;
        if (!capture_mask)
          return;
      }
      // If it didn't already exist, start a new projection epoch
      ProjectionEpoch *new_epoch = 
        new ProjectionEpoch(ProjectionEpoch::first_epoch, capture_mask);
      new_epoch->insert(info.projection, info.projection_space);
      projection_epochs.push_back(new_epoch);
      // Record it
      info.record_projection_epoch(ProjectionEpoch::first_epoch, capture_mask);
    }

    //--------------------------------------------------------------------------
    void LogicalState::capture_close_epochs(FieldMask capture_mask,
                                            ClosedNode *closed_node) const
    //--------------------------------------------------------------------------
    {
      for (std::list<ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        FieldMask overlap = (*it)->valid_fields & capture_mask;
        if (!overlap)
          continue;
        closed_node->record_projections(*it, overlap);
        capture_mask -= overlap;
        if (!capture_mask)
          return;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::update_projection_epochs(FieldMask update_mask,
                                                const ProjectionInfo &info)
    //--------------------------------------------------------------------------
    {
      for (std::list<ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        FieldMask overlap = (*it)->valid_fields & update_mask;
        if (!overlap)
          continue;
        (*it)->insert(info.projection, info.projection_space);
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
      new_epoch->insert(info.projection, info.projection_space);
      projection_epochs.push_back(new_epoch);
    }

    /////////////////////////////////////////////////////////////
    // FieldState 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldState::FieldState(void)
      : open_state(NOT_OPEN), redop(0), projection(NULL), 
        projection_space(NULL), rebuild_timeout(1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const GenericUser &user, const FieldMask &m, 
                           const LegionColor c)
      : ChildState(m), redop(0), projection(NULL), 
        projection_space(NULL), rebuild_timeout(1)
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
    FieldState::FieldState(const RegionUsage &usage, const FieldMask &m,
                           ProjectionFunction *proj, IndexSpaceNode *proj_space,
                           bool disjoint, bool dirty_reduction)
      : ChildState(m), redop(0), projection(proj), 
        projection_space(proj_space), rebuild_timeout(1)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(projection != NULL);
#endif
      if (IS_READ_ONLY(usage))
        open_state = OPEN_READ_ONLY_PROJ;
      else if (IS_REDUCE(usage))
      {
        if (dirty_reduction)
          open_state = OPEN_REDUCE_PROJ_DIRTY;
        else
          open_state = OPEN_REDUCE_PROJ;
        redop = usage.redop;
      }
      else if (disjoint && (projection->depth == 0))
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
      if ((projection != NULL) && (projection_space != rhs.projection_space))
        return false;
      if (redop == 0)
        return (open_state == rhs.open_state);
      else
      {
#ifdef DEBUG_LEGION
        assert((open_state == OPEN_SINGLE_REDUCE) ||
               (open_state == OPEN_MULTI_REDUCE) ||
               (open_state == OPEN_REDUCE_PROJ) ||
               (open_state == OPEN_REDUCE_PROJ_DIRTY));
        assert((rhs.open_state == OPEN_SINGLE_REDUCE) ||
               (rhs.open_state == OPEN_MULTI_REDUCE) ||
               (rhs.open_state == OPEN_REDUCE_PROJ) ||
               (rhs.open_state == OPEN_REDUCE_PROJ_DIRTY));
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
      for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it = 
            rhs.open_children.begin(); it != rhs.open_children.end(); it++)
      {
        LegionMap<LegionColor,FieldMask>::aligned::iterator finder = 
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
    bool FieldState::projection_domain_dominates(
                                               IndexSpaceNode *next_space) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(projection_space != NULL);
#endif
      if (projection_space == next_space)
        return true;
      // If the domains do not have the same type, the answer must be no
      if (projection_space->handle.get_type_tag() != 
          next_space->handle.get_type_tag())
        return false;
      return projection_space->dominates(next_space);
    }

    //--------------------------------------------------------------------------
    void FieldState::print_state(TreeStateLogger *logger,
                                 const FieldMask &capture_mask,
                                 RegionNode *node) const
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
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION %d",
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION (Disjoint Shallow) %d",
                        projection->projection_id);
            break;
          }
        case OPEN_REDUCE_PROJ:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION %d Mode %d",
                        projection->projection_id, redop);
            break;
          }
        case OPEN_REDUCE_PROJ_DIRTY:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION (Dirty) %d Mode %d",
                        projection->projection_id, redop);
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        char *mask_buffer = overlap.to_string();
        logger->log("Color %d   Mask %s", it->first, mask_buffer);
        free(mask_buffer);
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void FieldState::print_state(TreeStateLogger *logger,
                                 const FieldMask &capture_mask,
                                 PartitionNode *node) const
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
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION %d",
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION (Disjoint Shallow) %d",
                        projection->projection_id);
            break;
          }
        case OPEN_REDUCE_PROJ:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION %d Mode %d",
                        projection->projection_id, redop);
            break;
          }
        case OPEN_REDUCE_PROJ_DIRTY:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION (Dirty) %d Mode %d",
                        projection->projection_id, redop);
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      for (LegionMap<LegionColor,FieldMask>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        DomainPoint color =
          node->row_source->color_space->delinearize_color_to_point(it->first);
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        char *mask_buffer = overlap.to_string();
        switch (color.get_dim())
        {
          case 1:
            {
              logger->log("Color %d   Mask %s", 
                          color[0], mask_buffer);
              break;
            }
          case 2:
            {
              logger->log("Color (%d,%d)   Mask %s", 
                          color[0], color[1], mask_buffer);
              break;
            }
          case 3:
            {
              logger->log("Color (%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], mask_buffer);
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
    ClosedNode* ClosedNode::clone_disjoint_projection(
                  RegionTreeNode *child_node, const FieldMask &close_mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(children.empty()); // should never have any children here
#endif
      ClosedNode *result = new ClosedNode(child_node);
      for (std::map<ProjectionFunction*,
            LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator pit =
            projections.begin(); pit != projections.end(); pit++)
      {
        for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator it = 
              pit->second.begin(); it != pit->second.end(); it++)
        {
          FieldMask overlap = it->second & close_mask;
          if (!overlap)
            continue;
          result->record_projection(pit->first, it->first, overlap);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ClosedNode::record_projection(ProjectionFunction *function,
                                   IndexSpaceNode *space, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      projections[function][space] |= mask;
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
    void ClosedNode::record_reduced_fields(const FieldMask &fields)
    //--------------------------------------------------------------------------
    {
      reduced_fields |= fields;
    }

    //--------------------------------------------------------------------------
    void ClosedNode::record_projections(const ProjectionEpoch *epoch,
                                        const FieldMask &fields)
    //--------------------------------------------------------------------------
    {
      for (std::map<ProjectionFunction*,std::set<IndexSpaceNode*> >::
            const_iterator pit = epoch->projections.begin(); 
            pit != epoch->projections.end(); pit++)
      {
        std::map<ProjectionFunction*,LegionMap<IndexSpaceNode*,FieldMask>::
          aligned>::iterator finder = projections.find(pit->first);
        if (finder != projections.end())
        {
          for (std::set<IndexSpaceNode*>::const_iterator it = 
                pit->second.begin(); it != pit->second.end(); it++)
          {
            LegionMap<IndexSpaceNode*,FieldMask>::aligned::iterator finder2 = 
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
          LegionMap<IndexSpaceNode*,FieldMask>::aligned &spaces = 
            projections[pit->first];
          for (std::set<IndexSpaceNode*>::const_iterator it = 
                pit->second.begin(); it != pit->second.end(); it++)
            spaces[*it] = fields;
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
                  LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator
              pit = projections.begin(); pit != projections.end(); pit++) 
        {
          for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator 
                it = pit->second.begin(); it != pit->second.end(); it++)
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
      // Otherwise try to see if the children zip well, this only
      // works if we actually have children that can dominate other children
      if (!children.empty())
        old_tree->filter_dominated_children(non_dominated_mask, children);
    }

    //--------------------------------------------------------------------------
    void ClosedNode::filter_dominated_projection_fields(
        FieldMask &non_dominated_mask,
        const std::map<ProjectionFunction*,
          LegionMap<IndexSpaceNode*,FieldMask>::aligned> &new_projections) const
    //--------------------------------------------------------------------------
    {
      // In order to remove a dominated field, for each of our projection
      // operations, we need to find one in the new set that dominates it
      FieldMask dominated_mask = non_dominated_mask;
      for (std::map<ProjectionFunction*,
              LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator 
            pit = projections.begin(); pit != projections.end(); pit++)
      {
        // Set this iterator to the begining to start
        // Use it later to find domains with the same projection function
        std::map<ProjectionFunction*,
                 LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator
                   finder = new_projections.begin();
        for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator dit =
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
            for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator 
                  it = finder->second.begin(); it != finder->second.end(); it++)
            {
              FieldMask dom_overlap = overlap & it->second;
              if (!dom_overlap)
                continue; 
              // Types don't have to match, if they don't we don't care
              if (it->first->handle.get_type_tag() !=
                  dit->first->handle.get_type_tag())
                continue;
              // See if the domain dominates
              if (it->first->dominates(dit->first))
              {
                overlap -= dom_overlap;
                if (!overlap)
                  break;
              }
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
      // If the child is created for a complete partition with an identity
      // projection over the entire color space, we are dominated for its fields
      // TODO: Any bijective projections can use this optimization
      ProjectionFunction *identity =
        node->context->runtime->find_projection_function(0);
      for (std::map<RegionTreeNode*,ClosedNode*>::const_iterator it =
            new_children.begin(); it != new_children.end(); it++)
      {
        if (it->first->is_region() || it->second->projections.empty()) 
          continue;
        PartitionNode *node = it->first->as_partition_node();
        // The disjointness check here is to prevent nested composite instances
        // from being pruned when the new composite instance consists of
        // reduction instances
        if (!node->is_complete()) 
          continue;
        IndexSpaceNode *color_space = node->row_source->color_space;
        std::map<ProjectionFunction*,
                 LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator 
            finder = it->second->projections.find(identity);
        if (finder == it->second->projections.end()) 
          continue;

        FieldMask new_child_dominated;
        for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator dit =
              finder->second.begin(); dit != finder->second.end(); dit++)
        {
          const FieldMask overlap = non_dominated_mask & dit->second;
          if (!overlap)
            continue;
          // Make sure they are the same dimension then see if we
          // dominate the color space in which case we know we cover
          // the entire partition
          if ((color_space->get_num_dims() == dit->first->get_num_dims()) &&
              dit->first->dominates(color_space))
            new_child_dominated |= overlap;
        }
        // See if we have any dominated fields
        if (!new_child_dominated)
          continue;
        // If there are any reduction fields they can't be dominated
        if (!!it->second->reduced_fields)
        {
          new_child_dominated -= it->second->reduced_fields;
          if (!new_child_dominated)
            continue;
        }
        // Remove the fields dominated by this new child
        non_dominated_mask -= new_child_dominated;
        if (!!non_dominated_mask) 
          return;
      }

      // In order to remove a field, it has to be dominated in all our children
      FieldMask dominated_fields = non_dominated_mask;
      // If we have projections instead of explicitly closed children then we 
      // aren't going to directly compare them right now
      // TODO: make this analysis more precise
      if (!projections.empty())
      {
        for (std::map<ProjectionFunction*,
                LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator
              pit = projections.begin(); pit != projections.end(); pit++)
        {
          for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator 
                it = pit->second.begin(); it != pit->second.end(); it++)
          {
            const FieldMask overlap = it->second & dominated_fields;
            if (!overlap)
              continue;
            dominated_fields -= overlap;
            if (!dominated_fields)
              return;
          }
        }
      }
      FieldMask not_dominated_by_all;
      for (std::map<RegionTreeNode*,ClosedNode*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        FieldMask overlap = it->second->get_valid_fields() & non_dominated_mask;
        if (!overlap)
          continue;
        std::map<RegionTreeNode*,ClosedNode*>::const_iterator finder = 
          new_children.find(it->first);
        // If we can't find it, then we are not dominated for those fields
        if (finder == new_children.end())
        {
          dominated_fields -= overlap;
          if (!dominated_fields)
            return;
          continue;
        }
        FieldMask child_non_dominated = overlap;
        finder->second->filter_dominated_fields(it->second,child_non_dominated);
        not_dominated_by_all |= child_non_dominated;
      }
      non_dominated_mask -= dominated_fields - not_dominated_by_all;
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
              LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator 
            pit = projections.begin(); pit != projections.end(); pit++)
      {
        rez.serialize(pit->first->projection_id);
        rez.serialize<size_t>(pit->second.size());
        for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator it =
              pit->second.begin(); it != pit->second.end(); it++)
        {
          rez.serialize(it->first->handle);
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
        LegionMap<IndexSpaceNode*,FieldMask>::aligned &spaces = 
          projections[function];
        size_t num_doms;
        derez.deserialize(num_doms);
        for (unsigned idx2 = 0; idx2 < num_doms; idx2++)
        {
          IndexSpace handle;
          derez.deserialize(handle);
          IndexSpaceNode *node = runtime->forest->get_node(handle);
          derez.deserialize(spaces[node]);
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
      ClosedNode *result = new ClosedNode(node);
      result->perform_unpack(derez, runtime, is_region);
      return result;
    }

    void ClosedNode::record_closed_tree(const FieldMask &fields,
                                        ContextID logical_ctx,
                 LegionMap<std::pair<RegionTreeNode*,ContextID>,
                           FieldMask>::aligned &nodes,
                 std::map<std::pair<RegionTreeNode*,ContextID>,
                          LegionMap<IndexSpaceNode*,FieldMask>::aligned> &projs)
    //--------------------------------------------------------------------------
    {
      std::pair<RegionTreeNode*,ContextID> key(node, logical_ctx);
      if (children.size() == 0 && projections.size() == 0)
      {
        LegionMap<std::pair<RegionTreeNode*,ContextID>,
                  FieldMask>::aligned::iterator finder = nodes.find(key);
        if (finder == nodes.end())
            nodes[key] = fields;
        else
          finder->second |= fields;
        return;
      }
      std::map<std::pair<RegionTreeNode*,ContextID>,
               LegionMap<IndexSpaceNode*,FieldMask>::aligned>::iterator finder =
                 projs.find(key);
      for (std::map<ProjectionFunction*,
           LegionMap<IndexSpaceNode*,FieldMask>::aligned>::const_iterator pit =
           projections.begin(); pit != projections.end(); pit++)
      {
        for (LegionMap<IndexSpaceNode*,FieldMask>::aligned::const_iterator it =
              pit->second.begin(); it != pit->second.end(); it++)
        {
          FieldMask overlap = it->second & fields;
          if (!overlap) continue;
          if (finder == projs.end())
          {
            projs[key] = LegionMap<IndexSpaceNode*,FieldMask>::aligned();
            finder = projs.find(key);
          }
          LegionMap<IndexSpaceNode*,FieldMask>::aligned::iterator finder2 =
            finder->second.find(it->first);
          if (finder2 == finder->second.end())
            finder->second[it->first] = overlap;
          else
            finder2->second |= overlap;
        }
      }
      for (std::map<RegionTreeNode*,ClosedNode*>::const_iterator it =
            children.begin(); it != children.end(); it++)
        it->second->record_closed_tree(fields, logical_ctx, nodes, projs);
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
                                           bool projection, bool disjoint_close)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      normal_close_mask |= mask;
      if (projection && !disjoint_close)
        closed_projections |= mask;
      if (disjoint_close)
      {
#ifdef DEBUG_LEGION
        assert(projection); // should only happen with projections
#endif
        disjoint_close_mask |= mask;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_overwriting_close(const FieldMask &mask,
                                                 bool projection)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      // Overwriting closes do all the same stuff as read-only closes
      // only they also get to clear the dirty-below bits in the state
      overwriting_close_mask |= mask;
      record_read_only_close(mask, projection);
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_read_only_close(const FieldMask &mask,
                                               bool projection)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      read_only_close_mask |= mask;
      if (projection)
        closed_projections |= mask;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_flush_only_close(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      flush_only_close_mask |= mask;
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

#ifndef LEGION_SPY
    //--------------------------------------------------------------------------
    void LogicalCloser::pop_closed_user(bool read_only)
    //--------------------------------------------------------------------------
    {
      if (read_only)
        read_only_closed_users.pop_back();
      else
        normal_closed_users.pop_back();
    }
#endif

    //--------------------------------------------------------------------------
    void LogicalCloser::initialize_close_operations(LogicalState &state, 
                                             Operation *creator,
                                             const VersionInfo &ver_info,
                                             const LogicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // These sets of fields better be disjoint
      assert(normal_close_mask * flush_only_close_mask);
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
        normal_close_op = creator->runtime->get_available_inter_close_op();
        normal_close_gen = normal_close_op->get_generation();
        // Compute the set of fields that we need
        root_node->column_source->get_field_set(normal_close_mask,
                                               trace_info.req.privilege_fields,
                                               req.privilege_fields);
        std::map<RegionTreeNode*,ClosedNode*>::const_iterator finder = 
          closed_nodes.find(root_node);
#ifdef DEBUG_LEGION
        assert(finder != closed_nodes.end()); // better have a closed tree
#endif
        // Now initialize the operation
        normal_close_op->initialize(creator->get_context(), req, finder->second,
                                    trace_info, trace_info.req_idx, 
                                    ver_info, normal_close_mask, creator);
        // We can clear this now
        closed_nodes.clear();
        // See if we are doing a disjoint close for any of these fields
        if (!!disjoint_close_mask)
        {
          // Record the disjoint close, advance the projection epochs
          // and then record the new projection epoch that the close
          // will be a part of
          ProjectionInfo &proj_info = 
            normal_close_op->initialize_disjoint_close(disjoint_close_mask,
               root_node->as_partition_node()->row_source->color_space->handle);
          // Advance these epochs and then mark that we no longer need
          // to advance them during update_state
          state.advance_projection_epochs(disjoint_close_mask);
          closed_projections -= disjoint_close_mask;
          state.capture_projection_epochs(disjoint_close_mask, proj_info);
          // No need to record ourselves in the epoch since we know
          // that we are disjoint shallow
        }
      }
      if (!!read_only_close_mask)
      {
        read_only_close_op = 
          creator->runtime->get_available_read_close_op();
        read_only_close_gen = read_only_close_op->get_generation();
        req.privilege_fields.clear();
        root_node->column_source->get_field_set(read_only_close_mask,
                                               trace_info.req.privilege_fields,
                                               req.privilege_fields);
        read_only_close_op->initialize(creator->get_context(), req, 
                                       trace_info, trace_info.req_idx, 
                                       read_only_close_mask, creator);
      }
      // Finally if we have any fields which are flush only
      // make a close operation for them and add it to force close
      if (!!flush_only_close_mask)
      {
        flush_only_close_op =
          creator->runtime->get_available_inter_close_op();
        flush_only_close_gen = flush_only_close_op->get_generation();
        req.privilege_fields.clear();
        // Compute the set of fields that we need
        root_node->column_source->get_field_set(flush_only_close_mask,
                                               trace_info.req.privilege_fields,
                                               req.privilege_fields);
        // Make a closed tree of just the root node
        // There are no dirty fields here since we just flushing reductions
        ClosedNode *closed_tree = new ClosedNode(root_node);
        flush_only_close_op->initialize(creator->get_context(), req, 
            closed_tree, trace_info, trace_info.req_idx, 
            ver_info, flush_only_close_mask, creator);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::perform_dependence_analysis(const LogicalUser &current,
                                                    const FieldMask &open_below,
              LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &cusers,
              LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::track_aligned &pusers)
    //--------------------------------------------------------------------------
    {
      // A slightly strange case that can occur is if we close two different
      // children of the same node for the same field in different modes then
      // we can get both a normal close operation and a read-only close 
      // operation for the same field, in which case they need to share users
      // This should be a very rare case
      const FieldMask close_overlap = normal_close_mask & read_only_close_mask;
      if (!!close_overlap)
      {
        LegionVector<LogicalUser>::aligned add_normal_closed_users;
        for (std::list<LogicalUser>::const_iterator it = 
              read_only_closed_users.begin(); it != 
              read_only_closed_users.end(); it++)
        {
          const FieldMask overlap = it->field_mask & close_overlap;
          if (!overlap)
            continue;
#ifdef LEGION_SPY
          it->op->add_mapping_reference(it->gen);
#else
          if (!it->op->add_mapping_reference(it->gen))
            continue;
#endif
          add_normal_closed_users.push_back(*it);
          add_normal_closed_users.back().field_mask = overlap;
        }
        LegionVector<LogicalUser>::aligned add_read_only_closed_users;
        for (std::list<LogicalUser>::const_iterator it = 
              normal_closed_users.begin(); it != 
              normal_closed_users.end(); it++)
        {
          const FieldMask overlap = it->field_mask & close_overlap;
          if (!overlap)
            continue;
#ifdef LEGION_SPY
          it->op->add_mapping_reference(it->gen);
#else
          if (!it->op->add_mapping_reference(it->gen))
            continue;
#endif
          add_read_only_closed_users.push_back(*it);
          add_read_only_closed_users.back().field_mask = overlap;
        }
        normal_closed_users.insert(normal_closed_users.end(),
          add_normal_closed_users.begin(), add_normal_closed_users.end());
        read_only_closed_users.insert(read_only_closed_users.end(),
          add_read_only_closed_users.begin(), add_read_only_closed_users.end());
      }
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
    void LogicalCloser::update_state(LogicalState &state)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state.owner == root_node);
#endif
      // Advance any closed projection epochs, note that disjoint
      // closes already did this when we made the disjoint close op
      if (!!closed_projections)
        state.advance_projection_epochs(closed_projections);
      // If we had any overwriting close operations remove dirty below bits
      if (!!overwriting_close_mask)
        state.dirty_below -= overwriting_close_mask;
      // If we only have read-only closes then we are done
      FieldMask closed_mask = normal_close_mask | flush_only_close_mask;
      if (!closed_mask)
        return;
      root_node->filter_prev_epoch_users(state, closed_mask);
      root_node->filter_curr_epoch_users(state, closed_mask);
      // Any dirty data which is not a disjoint close resides at this level
      // otherwise it is still below
      if (!!disjoint_close_mask)
      {
        closed_mask -= disjoint_close_mask;
        if (!closed_mask)
          return;
      }
      // the dirty data now resides at this level
      state.dirty_fields |= closed_mask;
      state.dirty_below -= closed_mask;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::register_close_operations(
               LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::track_aligned &users)
    //--------------------------------------------------------------------------
    {
      // No need to add mapping references, we did that in 
      // Note we also use the cached generation IDs since the close
      // operations have already been kicked off and might be done
      // LogicalCloser::register_dependences
      if (normal_close_op != NULL)
      {
        LogicalUser close_user(normal_close_op, normal_close_gen, 0/*idx*/, 
            RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), normal_close_mask);
        users.push_back(close_user);
      }
      if (read_only_close_op != NULL)
      {
        LogicalUser close_user(read_only_close_op, read_only_close_gen,0/*idx*/,
          RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), read_only_close_mask);
        users.push_back(close_user);
      }
      if (flush_only_close_op != NULL)
      {
        LogicalUser close_user(flush_only_close_op, 
                               flush_only_close_gen, 0/*idx*/,
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
    void PhysicalState::apply_state(std::set<RtEvent> &applied_conditions) const
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
        it->first->merge_physical_state(this, it->second, applied_conditions);
    }

    //--------------------------------------------------------------------------
    void PhysicalState::capture_composite_root(CompositeView *composite_view,
                  const FieldMask &close_mask, ReferenceMutator *mutator,
                  const LegionMap<LogicalView*,FieldMask>::aligned &valid_above)
    //--------------------------------------------------------------------------
    {
      // Capture all the information for the root from the version_states
      for (PhysicalVersions::iterator it = version_states.begin();
            it != version_states.end(); it++)
      {
        FieldMask overlap = it->second & close_mask;
        if (!overlap)
          continue;
        it->first->capture_root(composite_view, overlap, mutator);
      }
      // Finally record any valid above views
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_above.begin(); it != valid_above.end(); it++)
      {
        composite_view->record_valid_view(it->first, it->second);
        // We also have to record these as dirty fields to make 
        // sure that we issue copies from them if necessary
        composite_view->record_dirty_fields(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::perform_disjoint_close(InterCloseOp *op, unsigned index,
                           InnerContext *context, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      // Iterate over the current states
      for (PhysicalVersions::iterator it = version_states.begin();
            it != version_states.end(); it++)
      {
        FieldMask overlap = it->second & closing_mask;
        if (!overlap)
          continue;
        it->first->perform_disjoint_close(op, index, context, overlap);
      }
    }

    //--------------------------------------------------------------------------
    PhysicalState* PhysicalState::clone(void) const
    //--------------------------------------------------------------------------
    {
      PhysicalState *result = new PhysicalState(node, path_only);
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
    void PhysicalState::clone_to(const FieldMask &version_mask, 
                                 const FieldMask &split_mask,
                                 InnerContext *context,
                                 VersionInfo &target_info,
                                 std::set<RtEvent> &ready_events) const
    //--------------------------------------------------------------------------
    {
      // Should only be calling this on path only nodes
#ifdef DEBUG_LEGION
      assert(path_only);
#endif
      if (!version_states.empty())
      {
        // Three different cases here depending on whether we have a split mask
        if (!split_mask)
        {
          // No split mask: just need initial versions of everything
          for (PhysicalVersions::iterator it = version_states.begin();
                it != version_states.end(); it++)
          {
            const FieldMask overlap = it->second & version_mask;
            if (!overlap)
              continue;
            target_info.add_current_version(it->first, overlap, path_only);
            it->first->request_initial_version_state(context, overlap,
                                                     ready_events);
          }
        }
        else if (split_mask == version_mask)
        {
          // All fields are split, need final versions of everything
          for (PhysicalVersions::iterator it = version_states.begin();
                it != version_states.end(); it++)
          {
            const FieldMask overlap = it->second & version_mask;
            if (!overlap)
              continue;
            target_info.add_current_version(it->first, overlap, path_only);
            it->first->request_final_version_state(context, overlap,
                                                   ready_events);
          }
        }
        else
        {
          // Mixed, need to figure out which ones we need initial/final versions
          for (PhysicalVersions::iterator it = version_states.begin();
                it != version_states.end(); it++)
          {
            FieldMask overlap = it->second & version_mask;
            if (!overlap)
              continue;
            target_info.add_current_version(it->first, overlap, path_only);
            const FieldMask split_overlap = overlap & split_mask; 
            if (!!split_overlap)
            {
              it->first->request_final_version_state(context, split_overlap,
                                                     ready_events);
              const FieldMask non_split_overlap = version_mask - split_overlap;
              if (!!non_split_overlap)
                it->first->request_initial_version_state(context, 
                                  non_split_overlap, ready_events);
            }
            else
              it->first->request_initial_version_state(context, overlap,
                                                       ready_events);
          }
        }
      }
      // For advance states we don't need to request any versions
      if (!advance_states.empty())
      {
        for (PhysicalVersions::iterator it = advance_states.begin();
              it != advance_states.end(); it++)
        {
          FieldMask overlap = it->second & version_mask;
          if (!overlap)
            continue;
          target_info.add_advance_version(it->first, overlap, path_only);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalState::print_physical_state(const FieldMask &capture_mask,
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
      // Reduction Mask
      {
        FieldMask overlap = reduction_mask & capture_mask;
        char *reduction_buffer = overlap.to_string();
        logger->log("Reduction Mask: %s",reduction_buffer);
        free(reduction_buffer);
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
        if (num_valid > 0)
        {
          logger->log("Valid Instances (%d)", num_valid);
          logger->down();
          for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            FieldMask overlap = it->second & capture_mask;
            if (!overlap)
              continue;
            if (it->first->is_deferred_view())
            {
              if (it->first->is_composite_view())
              {
                CompositeView *composite_view = it->first->as_composite_view();
                if (composite_view != NULL)
                {
                  logger->log("=== Composite Instance ===");
                  logger->down();
                  // We go only two levels down into the nested composite views
                  composite_view->print_view_state(capture_mask, logger, 0, 2);
                  logger->up();
                  logger->log("==========================");
                }
              }
              continue;
            }
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
        if (num_valid > 0)
        {
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
    void VersionManager::reset(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      is_owner = false;
      current_context = NULL;
      remote_valid_fields.clear();
      pending_remote_advance_summary.clear();
      pending_remote_advances.clear();
      remote_valid.clear();
      previous_opens.clear();
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
                                          InnerContext *context,
                                          unsigned init_index,
                                 const std::vector<LogicalView*> &corresponding,
                                          std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // See if we have been assigned
      if (context != current_context)
      {
#ifdef DEBUG_LEGION
        assert(current_version_infos.empty() || 
                (current_version_infos.size() == 1));
        assert(previous_version_infos.empty());
#endif
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = context;
      }
      // Make a new version state and initialize it, then insert it
      VersionState *init_state = create_new_version_state(init_version);
      init_state->initialize(term_event, usage, user_mask, targets, context, 
                             init_index, corresponding, applied_events);
      // We do need the lock because sometimes these are virtual
      // mapping results comping back
      AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
      sanity_check();
#endif
      LegionMap<VersionID,ManagerVersions>::aligned::iterator finder = 
          current_version_infos.find(init_version);
      if (finder == current_version_infos.end())
        current_version_infos[init_version].insert(init_state, user_mask);
      else
        finder->second.insert(init_state, user_mask);
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_current_versions(const FieldMask &version_mask, 
                                                FieldMask &unversioned_mask,
                                                InnerContext *context, 
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
      // We'll only track unversioned in the reading cases
      FieldMask unversioned;
      // Now we can record our versions
      if (IS_WRITE(usage))
      {
        // Uncomment this if we want READ_WRITE on unversioned
        // data to result in a warning, same with the code below
#ifdef UNVERSIONED_READ_WRITE_WARNING
        if (HAS_READ(usage))
          unversioned = version_mask;
#endif
        // At first we only need the lock in read-only mode
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        // See if we are the owner
        if (!is_owner)
        {
          const FieldMask request_mask = version_mask - remote_valid_fields;
          // Also handle the case where we have stale data from advances
          if (!!request_mask ||
              !(version_mask * pending_remote_advance_summary))
          {
            // Release the lock before sending the message
            m_lock.release();
            // Always pass in the full mask as the call will recompute
            // the request_mask in case we lose a race
            RtEvent wait_on = send_remote_version_request(version_mask,
                                                          ready_events);
            wait_on.wait();
            // Only retake the reservation, when we are ready
            m_lock.reacquire();
#ifdef DEBUG_LEGION
            // When we wake up everything should be good
            assert(!(version_mask - remote_valid_fields));
#endif
          }
        }
#ifdef DEBUG_LEGION
        sanity_check();
#endif
        // Just capture the current versions for now if we end
        // up mapping a physical instance then we'll advance 
        // later and record those versions as necessary
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator 
              vit = current_version_infos.begin(); vit !=
              current_version_infos.end(); vit++)
        {
          FieldMask local_overlap = vit->second.get_valid_mask() & version_mask;
          if (!local_overlap)
            continue;
#ifdef UNVERSIONED_READ_WRITE_WARNING
          unversioned -= local_overlap;
#endif
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            version_info.add_current_version(it->first, overlap,
                                             false/*path only*/);
            it->first->request_final_version_state(context, overlap, 
                                                   ready_events);
          }
        }
#ifdef UNVERSIONED_READ_WRITE_WARNING
        // Have to return since we don't want to make empty versions
        return;
#endif
      }
      else
      {
        // Only track unversioned in non-writing cases for now
        unversioned = version_mask;
        // At first we only need the lock in read-only mode
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        // See if we are the owner
        if (!is_owner)
        {
          const FieldMask request_mask = version_mask - remote_valid_fields;
          // Also handle the case where we have stale data from advances
          if (!!request_mask ||
              !(version_mask * pending_remote_advance_summary))
          {
            // Release the lock before sending the message
            m_lock.release();
            // Always pass in the full mask as the call will recompute
            // the request_mask in case we lose a race
            RtEvent wait_on = send_remote_version_request(version_mask,
                                                          ready_events);
            // Only retake the reservation, when we are ready
            wait_on.wait();
            m_lock.reacquire();
#ifdef DEBUG_LEGION
            // When we wake up everything should be good
            assert(!(version_mask - remote_valid_fields));
#endif
          }
        }
#ifdef DEBUG_LEGION
        sanity_check();
#endif
        // We only need the current versions, but we record them
        // as both the previous and the advance
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
            unversioned -= overlap;
            version_info.add_current_version(it->first, overlap,
                                             false/*path only*/);
            version_info.add_advance_version(it->first, overlap,
                                             false/*path only*/);
            it->first->request_initial_version_state(context, overlap, 
                                                     ready_events);
          }
          if (!unversioned)
            break;
        }
      }
      // If we have unversioned fields we need to make new 
      // version state object(s) for those fields
      if (!!unversioned)
      {
        // Only keep fields that are still unversioned
        unversioned_mask &= unversioned;
        // This is a nasty case: if we're unversioned it means we
        // have no protection from the mapping dependences to know
        // that we are the only one creating any new VersionState
        // objects, so we need a point of serialization (the owner)
        // to handle the creation of any unversioned fields, if we're 
        // on a remote node then we need to send a message to the owner
        LegionMap<VersionState*,FieldMask>::aligned unversioned_states;
        if (!is_owner)
        {
          RtUserEvent wait_on = Runtime::create_rt_user_event();
          // Send a message to the owner to do the update
          Serializer rez;
          {
            RezCheck z(rez);
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
            rez.serialize(unversioned);
            rez.serialize(&unversioned_states);
            rez.serialize(wait_on);
          }
          runtime->send_version_manager_unversioned_request(owner_space, rez);
          wait_on.wait();
        }
        else
        {
          WrapperReferenceMutator mutator(ready_events);
          find_or_create_unversioned_states(unversioned, 
                                            unversioned_states, &mutator);
        }
        // Now we can record the results
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
              unversioned_states.begin(); it != unversioned_states.end(); it++)
        {
          version_info.add_current_version(it->first, it->second,
                                           false/*path only*/);
          version_info.add_advance_version(it->first, it->second, 
                                           false/*path only*/);
          it->first->request_initial_version_state(context, it->second, 
                                                   ready_events);
        }
      }
      else if (!!unversioned_mask)
        unversioned_mask.clear();
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_advance_versions(const FieldMask &version_mask,
                                                InnerContext *context,
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
      AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
      // See if we are the owner
      if (!is_owner)
      {
        const FieldMask request_mask = version_mask - remote_valid_fields;
        // Also handle the case where we have stale data from advances
        if (!!request_mask ||
            !(version_mask * pending_remote_advance_summary))
        {
          // Release the lock before sending the message
          m_lock.release();
          // Always pass in the full mask as the call will recompute
          // the request_mask in case we lose a race
          RtEvent wait_on = send_remote_version_request(version_mask,
                                                        ready_events);
          // Retake the lock only once we're ready to
          wait_on.wait();
          m_lock.reacquire();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(version_mask - remote_valid_fields));
#endif
        }
      }
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
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::compute_advance_split_mask(VersionInfo &version_info,
                                                   UniqueID logical_context_uid,
                                                   InnerContext *context,
                                                const FieldMask &version_mask,
                                                std::set<RtEvent> &ready_events,
          const LegionMap<ProjectionEpochID,FieldMask>::aligned &advance_epochs)
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
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
      // See if we are the owner
      if (!is_owner)
      {
        const FieldMask request_mask = version_mask - remote_valid_fields;
        // Also handle the case where we have stale data from advances
        if (!!request_mask ||
            !(version_mask * pending_remote_advance_summary))
        {
          // Release the lock before sending the message
          m_lock.release();
          // Always pass in the full mask as the call will recompute
          // the request_mask in case we lose a race
          RtEvent wait_on = send_remote_version_request(version_mask,
                                                        ready_events); 
          // Retake the lock only once we're ready to
          wait_on.wait();
          m_lock.reacquire();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(version_mask - remote_valid_fields));
#endif
        }
      }
      // See if we've done any previous advances for this projection epoch
      for (LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator it =
            advance_epochs.begin(); it != advance_epochs.end(); it++)
      {
        LegionMap<ProjectionEpoch,FieldMask>::aligned::const_iterator 
          finder = previous_advancers.find(
              ProjectionEpoch(logical_context_uid, it->first));
        if (finder == previous_advancers.end())
          continue;
        FieldMask overlap = finder->second & it->second;
        if (!overlap)
          continue;
        version_info.record_split_fields(node, overlap);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_path_only_versions(
                                                const FieldMask &version_mask,
                                                const FieldMask &split_mask,
                                                FieldMask &unversioned_mask,
                                                InnerContext *context,
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
      // We aren't mutating our data structures, so we just need 
      // the manager lock in read only mode
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
      // See if we are the owner
      if (!is_owner)
      {
        const FieldMask request_mask = version_mask - remote_valid_fields;
        // Also handle the case where we have stale data from advances
        if (!!request_mask ||
            !(version_mask * pending_remote_advance_summary))
        {
          // Release the lock before sending the message
          m_lock.release();
          // Always pass in the full mask as the call will recompute
          // the request_mask in case we lose a race
          RtEvent wait_on = send_remote_version_request(version_mask,
                                                        ready_events); 
          // Retake the lock only once we're ready to
          wait_on.wait();
          m_lock.reacquire();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(version_mask - remote_valid_fields));
#endif
        }
      }
      FieldMask unversioned = version_mask; 
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
            for (ManagerVersions::iterator it = vit->second.begin();
                  it != vit->second.end(); it++)
            {
              FieldMask overlap = it->second & local_overlap;
              if (!overlap)
                continue;
              unversioned -= overlap;
              version_info.add_current_version(it->first, overlap,
                                               true/*path only*/);
              it->first->request_final_version_state(context, overlap, 
                                                     ready_events);
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
            if (vit->second.size() > 1)
            {
              // Need this to be precise and valid mask might overapproximate
              for (ManagerVersions::iterator it = vit->second.begin();
                    it != vit->second.end(); it++)
                unversioned_mask -= it->second;
            }
            else
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
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            unversioned -= overlap;
            version_info.add_current_version(it->first, overlap,
                                             true/*path only*/);
            it->first->request_final_version_state(context, overlap, 
                                                   ready_events);
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
              it->first->request_initial_version_state(context, overlap,
                                                       ready_events);
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
          for (ManagerVersions::iterator it = vit->second.begin();
                it != vit->second.end(); it++)
          {
            FieldMask overlap = it->second & local_overlap;
            if (!overlap)
              continue;
            unversioned -= overlap;
            version_info.add_current_version(it->first, overlap, 
                                             true/*path only*/);
            it->first->request_initial_version_state(context, overlap, 
                                                     ready_events);
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
    void VersionManager::record_disjoint_close_versions(
                                                const FieldMask &version_mask,
                                                InnerContext *context,
                                                Operation *op, unsigned index,
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
      // We aren't mutating our data structures, so we just need 
      // the manager lock in read only mode
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
      // See if we are the owner
      if (!is_owner)
      {
        const FieldMask request_mask = version_mask - remote_valid_fields;
        // Also handle the case where we have stale data from advances
        if (!!request_mask ||
            !(version_mask * pending_remote_advance_summary))
        {
          // Release the lock before sending the message
          m_lock.release();
          // Always pass in the full mask as the call will recompute
          // the request_mask in case we lose a race
          RtEvent wait_on = send_remote_version_request(version_mask,
                                                        ready_events); 
          // Retake the lock only once we're ready to
          wait_on.wait();
          m_lock.reacquire();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(version_mask - remote_valid_fields));
#endif
        }
      }
#ifdef DEBUG_LEGION
      sanity_check();
#endif
      // We need all the current versions at this level with 
      // their open children information up to date because
      // it is the current version state objects that are 
      // tracking which children are dirty below 
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit =
            current_version_infos.begin(); vit != 
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
          version_info.add_current_version(it->first, overlap,
                                           true/*path only*/); 
          it->first->request_children_version_state(context, overlap,
                                                    ready_events);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::advance_versions(FieldMask mask, 
                                          UniqueID logical_context_uid,
                                          InnerContext *physical_context, 
                                          bool update_parent_state,
                                          std::set<RtEvent> &applied_events,
                                          bool dedup_opens,
                                          ProjectionEpochID open_epoch,
                                          bool dedup_advances, 
                                          ProjectionEpochID advance_epoch,
                                          const FieldMask *dirty_previous,
                                          const ProjectionInfo *proj_info)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(node->context->runtime, 
                        CURRENT_STATE_ADVANCE_VERSION_NUMBERS_CALL);
      // See if we have been assigned
      if (physical_context != current_context)
      {
        const AddressSpaceID local_space = 
          node->context->runtime->address_space;
        owner_space = physical_context->get_version_owner(node, local_space);
        is_owner = (owner_space == local_space);
        current_context = physical_context;
      }
      // Check to see if we are the owner
      if (!is_owner)
      {
        // First send back the message to the owner to do the advance there
        // This also guarantees that we are serialized with respect to 
        // all previous advances since the virtual channel is in order
        // which means we have implicit dependences on all previous 
        // advances issued from this node
        RtEvent advanced = send_remote_advance(mask, update_parent_state,
                                               logical_context_uid,
                                               dedup_opens, open_epoch, 
                                               dedup_advances, advance_epoch,
                                               dirty_previous, proj_info);
        // Now retake the lock and see if we need to save this
        // in the list of pending remote advances
        // We can do this afterwards as we know the advance will always
        // come before any valid requests for a given region requirement
        {
          AutoLock m_lock(manager_lock);
          const FieldMask valid_overlap = mask & remote_valid_fields;
          if (!valid_overlap)
          {
            // if we have no overlap then there is no need to store
            // this i the set of pending remote_advances
            applied_events.insert(advanced);
            return;
          }
          // Otherwise save it in the pending remote advance summary
          // and keep going so we can launch off a task to reclaim it
          pending_remote_advances[advanced] = valid_overlap; 
          pending_remote_advance_summary |= valid_overlap;
        }
        // Launch off a meta-task to reclaim the advanced field
        PendingAdvanceArgs args;
        args.proxy_this = this;
        args.to_reclaim = advanced;
        RtEvent done = 
          runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                                           advanced);
        // Add this event to the set of applied preconditions
        // in order to avoid cleanup races
        applied_events.insert(done);
        return;
      }
      // If we are deduplicating advances, do that now
      // to see if we can avoid any communication
      if (dedup_opens || dedup_advances)
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        if (dedup_opens)
        {
          LegionMap<ProjectionEpoch,FieldMask>::aligned::const_iterator
            finder = previous_opens.find(
                ProjectionEpoch(logical_context_uid, open_epoch));
          if (finder != previous_opens.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
          }
        }
        if (dedup_advances)
        {
          LegionMap<ProjectionEpoch,FieldMask>::aligned::const_iterator 
            finder = previous_advancers.find(
                ProjectionEpoch(logical_context_uid, advance_epoch));
          if (finder != previous_advancers.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
          }
        }
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
          // Filter out any previous opens if necessary
          const ProjectionEpoch key(logical_context_uid, open_epoch);
          if (!previous_opens.empty())
          {
            for (LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator it =
                  previous_opens.begin(); it != 
                  previous_opens.end(); /*nothing*/)
            {
              if (it->first == key) // skip our own
              {
                it++;
                continue;
              }
              it->second -= mask;
              if (!it->second)
              {
                LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator 
                  to_delete = it;
                it++;
                previous_opens.erase(to_delete);
              }
              else
                it++;
            }
          }
          LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator
            finder = previous_opens.find(key);
          if (finder != previous_opens.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
            finder->second |= mask;
          }
          else
            previous_opens[key] = mask;
        }
        if (dedup_advances)
        {
          // Filter out any previous advancers if necessary  
          const ProjectionEpoch key(logical_context_uid, advance_epoch);
          if (!previous_advancers.empty())
          {
            for (LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator it =
                  previous_advancers.begin(); it != 
                  previous_advancers.end(); /*nothing*/)
            {
              if (it->first == key) // skip our own
              {
                it++;
                continue;
              }
              it->second -= mask;
              if (!it->second)
              {
                LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator 
                  to_delete = it;
                it++;
                previous_advancers.erase(to_delete);
              }
              else
                it++;
            }
          }
          LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator 
            finder = previous_advancers.find(key);
          if (finder != previous_advancers.end())
          {
            mask -= finder->second;
            if (!mask)
              return;
            finder->second |= mask;
          }
          else
            previous_advancers[key] = mask;
        }
        // Record any epoch updates
        if (proj_info != NULL)
        {
          const LegionMap<ProjectionEpochID,FieldMask>::aligned 
            &advance_epochs = proj_info->get_projection_epochs();
          for (LegionMap<ProjectionEpochID,FieldMask>::aligned::const_iterator
               pit = advance_epochs.begin(); pit != advance_epochs.end(); pit++)
          {
            const ProjectionEpoch key(logical_context_uid, pit->first);
            FieldMask update_mask = pit->second;
            // Filter out any previous advancers with overlapping fields
            std::vector<ProjectionEpoch> to_delete;
            for (LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator it = 
                  previous_advancers.begin(); it !=
                  previous_advancers.end(); it++)
            {
              if (it->first == key)
              {
                update_mask -= it->second;
                if (!update_mask)
                  break;
                continue;
              }
              it->second -= pit->second;
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<ProjectionEpoch>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
                previous_advancers.erase(*it);
            }
            if (!!update_mask)
            {
              LegionMap<ProjectionEpoch,FieldMask>::aligned::iterator finder =
                previous_advancers.find(key);
              if (finder == previous_advancers.end())
                previous_advancers[key] = update_mask;
              else
                finder->second |= update_mask;
            }
          }
        }
        // Now send any invalidations to get them in flight
        if (!remote_valid.empty() && !(remote_valid_fields * mask))
        {
          std::vector<AddressSpaceID> to_delete;
          for (LegionMap<AddressSpaceID,FieldMask>::aligned::iterator it = 
                remote_valid.begin(); it != remote_valid.end(); it++)
          {
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
          remote_valid_fields -= mask;
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
          // Might be an overapproximation, so we might need to
          // update the overlap because we need it precise below
          const bool need_overlap_update = (info.size() > 1);
          // See if everyone is going away or just some of them
          if (overlap == info.get_valid_mask())
          {
            // The whole version number is going away, remove all
            // the valid references on the version state objects
            if (need_overlap_update)
            {
              overlap.clear();
              for (ManagerVersions::iterator it = info.begin();
                    it != info.end(); it++)
                overlap |= it->second;
            }
            to_delete_previous.push_back(vit->first);
          }
          else
          {
            // Only some of the state are being filtered
            std::vector<VersionState*> to_delete;
            if (need_overlap_update)
              overlap.clear();
            for (ManagerVersions::iterator it = info.begin();
                  it != info.end(); it++)
            {
              FieldMask state_overlap = it->second & previous_filter;
              if (!state_overlap)
                continue;
              if (need_overlap_update)
                overlap |= state_overlap;
              it->second -= state_overlap;
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<VersionState*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
                info.erase(*it);
              if (info.empty())
                to_delete_previous.push_back(vit->first);
            }
          }
          if (need_overlap_update && !overlap)
            continue;
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
          // If we have more than one element then the
          // valid mask might be an over approximation and
          // we're going to need the precise overlap set below
          const bool need_version_overlap_update = (info.size() > 1);
          if (version_overlap == info.get_valid_mask())
          {
            // Send back the whole version state info to previous
            to_delete_current.insert(vit->first);
            // See if we need to merge it or can just copy it
            LegionMap<VersionID,ManagerVersions>::aligned::iterator 
              prev_finder = previous_version_infos.find(vit->first);
            if (prev_finder == previous_version_infos.end())
            {
              if (need_version_overlap_update)
              {
                version_overlap.clear();
                for (ManagerVersions::iterator it = info.begin();
                      it != info.end(); it++)
                  version_overlap |= it->second; 
              }
              // Can just send it back with no merge
              info.move(previous_version_infos[vit->first]);
            }
            else
            {
              // prev_inf already existed
              // Filter back the version states
              if (need_version_overlap_update)
                version_overlap.clear();
              for (ManagerVersions::iterator it = info.begin();
                    it != info.end(); it++)
              {
                if (need_version_overlap_update)
                  version_overlap |= it->second;
                prev_finder->second.insert(it->first, it->second, &mutator);
              }
              info.clear(); // clear it in case we get unerased
            }
          }
          else
          {
            if (need_version_overlap_update)
              version_overlap.clear();
            // Filter back only some of the version states
            std::vector<VersionState*> to_delete;
            ManagerVersions &prev_info = previous_version_infos[vit->first];
            for (ManagerVersions::iterator it = info.begin();
                  it != info.end(); it++)
            {
              FieldMask overlap = it->second & current_filter;
              if (!overlap)
                continue;
              if (need_version_overlap_update)
                version_overlap |= overlap; 
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
            if (info.empty())
              to_delete_current.insert(vit->first);
          }
          // Make our new version state object and add if we can
          VersionID next_version = vit->first+1;
          // Remove this version number from the delete set if it exists
          to_delete_current.erase(next_version);
          VersionState *new_state = create_new_version_state(next_version);
          if (update_parent_state || (dirty_previous != NULL))
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
        // Remove any old version state infos
        if (!to_delete_current.empty())
        {
          for (std::set<VersionID>::const_iterator it = 
                to_delete_current.begin(); it != to_delete_current.end(); it++)
            current_version_infos.erase(*it);
        }
        // See if we have any fields for which there was no prior
        // version number, if there was then these are fields which
        // are being initialized and should be added as version 1
        if (!!current_filter)
        {
          VersionState *new_state = create_new_version_state(init_version);
          if (update_parent_state || (dirty_previous != NULL))
            new_states.insert(new_state, current_filter, &mutator);
          current_version_infos[init_version].insert(new_state, 
                                                     current_filter, &mutator);
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
      // If we have no new states then we are done
      if (new_states.empty())
        return;
      // See if we have to capture any dirty data from the previous versions
      // If we don't have to update the parent state then we are at the
      // top of the tree, so there is no need to worry about closes from above
      if (dirty_previous != NULL)
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit =
              previous_version_infos.begin(); vit != 
              previous_version_infos.end(); vit++)
        {
          // Disjoint so it doesn't matter
          if (vit->second.get_valid_mask() * (*dirty_previous))
            continue;
          for (ManagerVersions::iterator mit = vit->second.begin();
                mit != vit->second.end(); mit++)
          {
            FieldMask state_overlap = mit->second & *dirty_previous;
            if (!state_overlap)
              continue;
            // Get the final version of this version state
            std::set<RtEvent> preconditions;
            mit->first->request_final_version_state(physical_context, 
                                                  state_overlap, preconditions);
            if (!preconditions.empty())
            {
              RtEvent precondition = Runtime::merge_events(preconditions);
              for (VersioningSet<>::iterator it = new_states.begin();
                    it != new_states.end(); it++)
              {
                FieldMask overlap = it->second & state_overlap;
                if (!overlap)
                  continue;
                DirtyUpdateArgs args;
                args.previous = mit->first;
                args.target = it->first;
                args.capture_mask = new FieldMask(overlap);
                RtEvent done = 
                  runtime->issue_runtime_meta_task(args, 
                      LG_LATENCY_WORK_PRIORITY, precondition);
                applied_events.insert(done);
                state_overlap -= overlap;
                if (!state_overlap)
                  break;
              }
            }
            else
            {
              // We can do the captures now 
              for (VersioningSet<>::iterator it = new_states.begin();
                    it != new_states.end(); it++)
              {
                FieldMask overlap = it->second & state_overlap;
                if (!overlap)
                  continue;
                mit->first->capture_dirty_instances(overlap, it->first); 
                state_overlap -= overlap;
                if (!state_overlap)
                  break;
              }
            }
          }
        }
      }
      // If we recorded any new states then we have to tell our parent manager
      if (update_parent_state)
      {
        RegionTreeNode *parent = node->get_parent();      
#ifdef DEBUG_LEGION
        assert(parent != NULL);
#endif
        VersionManager &parent_manager = 
          parent->get_current_version_manager(ctx);
        const LegionColor color = node->get_color();
        // This destroys new states, but whatever we're done anyway
        parent_manager.update_child_versions(physical_context, color, 
                                             new_states, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::reclaim_pending_advance(RtEvent done_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner);
#endif
      AutoLock m_lock(manager_lock);
      LegionMap<RtEvent,FieldMask>::aligned::iterator finder = 
        pending_remote_advances.find(done_event);
      // Might already have been pruned by an invalidate
      if (finder == pending_remote_advances.end())
        return;
      // Otherwise remote it and rebuild the summary mask
      pending_remote_advances.erase(finder);
      FieldMask new_summary;
      for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
            pending_remote_advances.begin(); it != 
            pending_remote_advances.end(); it++)
        new_summary |= it->second;
      pending_remote_advance_summary = new_summary;
    }

    //--------------------------------------------------------------------------
    void VersionManager::find_or_create_unversioned_states(
                FieldMask unversioned,
                LegionMap<VersionState*,FieldMask>::aligned &unversioned_states,
                ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner);
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
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          FieldMask overlap = it->second & local_overlap;
          if (!overlap)
            continue;
          unversioned_states[it->first] = overlap;
          unversioned -= overlap;
          if (!unversioned)
            return;
        }
      }
      // If we get here then we actually need to make a new state
      VersionState *new_state = create_new_version_state(init_version);
      current_version_infos[init_version].insert(new_state, 
                                                 unversioned, mutator);
      unversioned_states[new_state] = unversioned;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_unversioned_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
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
      FieldMask unversioned;
      derez.deserialize(unversioned);
      void *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      InnerContext *context = runtime->find_context(context_uid);
      ContextID ctx = context->get_context_id();
      VersionManager &manager = node->get_current_version_manager(ctx);
      
      std::set<RtEvent> done_preconditions;
      WrapperReferenceMutator mutator(done_preconditions);
      LegionMap<VersionState*,FieldMask>::aligned unversioned_states;

      manager.find_or_create_unversioned_states(unversioned,
                                                unversioned_states, &mutator);
      // Now send the results back to the source
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(target);
        rez.serialize<size_t>(unversioned_states.size());
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              unversioned_states.begin(); it != unversioned_states.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
        if (!done_preconditions.empty())
          rez.serialize(Runtime::merge_events(done_preconditions));
        else
          rez.serialize(RtEvent::NO_RT_EVENT);
        rez.serialize(done);
      }
      runtime->send_version_manager_unversioned_response(source, rez);
    }
    
    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_unversioned_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LegionMap<VersionState*,FieldMask>::aligned *unversioned_states;
      derez.deserialize(unversioned_states);
      size_t num_states;
      derez.deserialize(num_states);
      std::set<RtEvent> preconditions;
      for (unsigned idx = 0; idx < num_states; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        VersionState *state = runtime->find_or_request_version_state(did,ready);
        if (ready.exists())
          preconditions.insert(ready);
        derez.deserialize((*unversioned_states)[state]);
      }
      RtEvent precondition;
      derez.deserialize(precondition);
      if (precondition.exists())
        preconditions.insert(precondition);
      RtUserEvent done;
      derez.deserialize(done);
      if (!preconditions.empty())
        Runtime::trigger_event(done, Runtime::merge_events(preconditions));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::process_capture_dirty(const void *args)
    //--------------------------------------------------------------------------
    {
      const DirtyUpdateArgs *dargs = (const DirtyUpdateArgs*)args;
      dargs->previous->capture_dirty_instances(*(dargs->capture_mask),
                                               dargs->target);
      delete dargs->capture_mask;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::process_pending_advance(const void *args)
    //--------------------------------------------------------------------------
    {
      const PendingAdvanceArgs *pargs = (const PendingAdvanceArgs*)args;
      pargs->proxy_this->reclaim_pending_advance(pargs->to_reclaim);
    }

    //--------------------------------------------------------------------------
    void VersionManager::update_child_versions(InnerContext *context,
                                              const LegionColor child_color,
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
      // Only need to hold the lock in read-only mode since we're not
      // going to be updating any of these local data structures
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
      // If we are not the owner, see if we need to issue any requests
      if (!is_owner)
      {
        const FieldMask request_mask = 
          new_states.get_valid_mask() - remote_valid_fields;
        // Also handle the case where we have stale data from advances
        if (!!request_mask ||
            !(new_states.get_valid_mask() * pending_remote_advance_summary))
        {
          // Release the lock before sending the message
          m_lock.release();
          // Always pass in the full mask as the call will recompute
          // the request_mask in case we lose a race
          RtEvent wait_on = send_remote_version_request(
              new_states.get_valid_mask(), applied_events);
          // Retake the lock only once we're ready to
          wait_on.wait();
          m_lock.reacquire();
#ifdef DEBUG_LEGION
          // When we wake up everything should be good
          assert(!(new_states.get_valid_mask() - remote_valid_fields));
#endif
        }
      }
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
                                          applied_events, true/*need lock*/,
                                          true/*local update*/);
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
#ifdef DEBUG_LEGION
      sanity_check();
#endif
      // This invalidates our local fields
      remote_valid_fields -= invalid_mask;
      // Remove any pending remote advances that we have since
      // we no longer care about them now that we are no longer valid
      if (!(pending_remote_advance_summary * invalid_mask))
      {
        std::vector<RtEvent> to_delete;
        for (LegionMap<RtEvent,FieldMask>::aligned::iterator it = 
              pending_remote_advances.begin(); it != 
              pending_remote_advances.end(); it++)
        {
          it->second -= invalid_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        if (!to_delete.empty())
        {
          for (std::vector<RtEvent>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            pending_remote_advances.erase(*it);
        }
        pending_remote_advance_summary -= invalid_mask;
      }
      filter_version_info(invalid_mask, current_version_infos);
      filter_version_info(invalid_mask, previous_version_infos);
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::filter_version_info(const FieldMask &mask,
                       LegionMap<VersionID,ManagerVersions>::aligned &to_filter)
    //--------------------------------------------------------------------------
    {
      std::vector<VersionID> to_delete;
      for (LegionMap<VersionID,ManagerVersions>::aligned::iterator vit = 
            to_filter.begin(); vit != to_filter.end(); vit++)
      {
        FieldMask overlap = vit->second.get_valid_mask() & mask;
        if (!overlap)
          continue;
        if (overlap == vit->second.get_valid_mask())
          to_delete.push_back(vit->first);
        else
        {
          std::vector<VersionState*> to_remove;
          for (ManagerVersions::iterator it = vit->second.begin(); 
                it != vit->second.end(); it++)
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
            if (vit->second.empty())
              to_delete.push_back(vit->first);
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
      temp_state.print_physical_state(capture_mask, logger);
    }

    //--------------------------------------------------------------------------
    void VersionManager::update_physical_state(PhysicalState *state)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit =
           current_version_infos.begin(); vit !=
           current_version_infos.end(); vit++)
      {
        for (ManagerVersions::iterator it = vit->second.begin();
             it != vit->second.end(); it++)
        {
          VersionState *vs = dynamic_cast<VersionState*>(it->first);
          vs->update_physical_state(state, it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    VersionState* VersionManager::create_new_version_state(VersionID vid)
    //--------------------------------------------------------------------------
    {
      DistributedID did = runtime->get_available_distributed_id();
      return new VersionState(vid, runtime, did, 
          runtime->address_space, node, true/*register now*/);
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::send_remote_advance(const FieldMask &advance_mask,
                                                bool update_parent_state,
                                                UniqueID logical_context_uid,
                                                bool dedup_opens,
                                                ProjectionEpochID open_epoch,
                                                bool dedup_advances,
                                                ProjectionEpochID advance_epoch,
                                                const FieldMask *dirty_previous,
                                                const ProjectionInfo *proj_info)
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
        rez.serialize(logical_context_uid);
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
        rez.serialize<bool>(update_parent_state);
        rez.serialize<bool>(dedup_opens);
        if (dedup_opens)
          rez.serialize(open_epoch);
        rez.serialize<bool>(dedup_advances);
        if (dedup_advances)
          rez.serialize(advance_epoch);
        if (dirty_previous != NULL)
        {
          rez.serialize<bool>(true);
          rez.serialize(*dirty_previous);
        }
        else
          rez.serialize<bool>(false);
        if (proj_info != NULL)
        {
          rez.serialize<bool>(true);
          // Only need to pack the projection epochs
          proj_info->pack_epochs(rez);
        }
        else
          rez.serialize<bool>(false);
      }
      runtime->send_version_manager_advance(owner_space, rez);
      return remote_advance;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_remote_advance(Deserializer &derez,
                                                          Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      UniqueID logical_context_uid;
      derez.deserialize(logical_context_uid);
      UniqueID physical_context_uid;
      derez.deserialize(physical_context_uid);
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
        derez.deserialize(open_epoch);
      bool dedup_advances;
      derez.deserialize(dedup_advances);
      ProjectionEpochID advance_epoch = 0;
      if (dedup_advances)
        derez.deserialize(advance_epoch);
      bool has_dirty_previous;
      derez.deserialize(has_dirty_previous);
      FieldMask dirty_previous;
      if (has_dirty_previous)
        derez.deserialize(dirty_previous);
      bool has_proj_info;
      derez.deserialize(has_proj_info);
      ProjectionInfo proj_info;
      if (has_proj_info)
        proj_info.unpack_epochs(derez);

      InnerContext *context = runtime->find_context(physical_context_uid);
      ContextID ctx = context->get_context_id();
      VersionManager &manager = node->get_current_version_manager(ctx);
      std::set<RtEvent> done_preconditions;
      manager.advance_versions(advance_mask, logical_context_uid, context, 
                               update_parent, done_preconditions, 
                               dedup_opens, open_epoch, 
                               dedup_advances, advance_epoch,
                               has_dirty_previous ? &dirty_previous : NULL,
                               has_proj_info ? &proj_info : NULL);
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

      InnerContext *context = runtime->find_context(context_uid);
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
        // First recheck to see if we lost any races with responses
        // already coming back to us since we computed our request mask
        if (!!pending_remote_advance_summary)
        {
          const FieldMask overlap = 
            request_mask & pending_remote_advance_summary;
          // Always remove the fields that are now valid
          request_mask -= remote_valid_fields;
          // If we had any overlap fields we have to add them back in
          if (!!overlap)
            request_mask |= overlap;
        }
        else // No remote advances so remove any fields now valid
          request_mask -= remote_valid_fields;
        // If we lost races with responses then we might already be done
        if (!request_mask)
          return RtEvent::NO_RT_EVENT;
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

      InnerContext *context = runtime->find_context(context_uid);
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
      // Do most of this in read only mode
#ifdef DEBUG_LEGION
      FieldMask send_mask = request_mask;
#endif
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
        sanity_check();
#else
        FieldMask send_mask = request_mask;
#endif
        // We only need to send it if we know that it is not valid anymore
        LegionMap<AddressSpaceID,FieldMask>::aligned::const_iterator
          finder = remote_valid.find(target);
        // Remove any fields that we've already sent a response for
        if (finder != remote_valid.end())
          send_mask -= finder->second;
        LegionMap<VersionState*,FieldMask>::aligned send_infos;
        // Do the current infos first
        if (!!send_mask)
          find_send_infos(current_version_infos, send_mask, send_infos);
        pack_send_infos(rez, send_infos);
        if (!!send_mask)
        {
          send_infos.clear();
          // Then do the previous infos
          find_send_infos(previous_version_infos, request_mask, send_infos);
        }
        pack_send_infos(rez, send_infos);
      }
      // Need exclusive access at the end to update the remote information
      AutoLock m_lock(manager_lock);
      remote_valid_fields |= request_mask;
#ifdef DEBUG_LEGION
      // Sanity check that no invalidations were sent while
      // we weren't holding the lock
      if (remote_valid.find(target) != remote_valid.end())
        assert(send_mask == (request_mask - remote_valid[target]));
#endif
      remote_valid[target] |= request_mask;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::find_send_infos(
                   LegionMap<VersionID,ManagerVersions>::aligned &version_infos,
                        const FieldMask &request_mask, 
                        LegionMap<VersionState*,FieldMask>::aligned& send_infos)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit = 
            version_infos.begin(); vit != version_infos.end(); vit++)
      {
        FieldMask overlap = vit->second.get_valid_mask() & request_mask;
        if (!overlap)
          continue;
        for (ManagerVersions::iterator it = vit->second.begin(); 
              it != vit->second.end(); it++)
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
        // Always add a remote valid reference in case we get invalidated
        // by an advance before this message is received on the remote node
#ifdef DEBUG_LEGION
        assert(it->first->is_owner());
#endif
        // This reference will be removed by the remote side after
        // it has registered the new version state
        it->first->add_base_valid_ref(REMOTE_DID_REF);
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
      // If we have any preconditions here we must wait
      if (!preconditions.empty())
      {
        RtEvent wait_on = Runtime::merge_events(preconditions);
        wait_on.wait();
      }
#ifdef DEBUG_LEGION
      assert(applied_events != NULL);
#endif
      WrapperReferenceMutator mutator(*applied_events);
      // Take our lock and apply our updates
      {
        AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
        sanity_check();
#endif
        merge_send_infos(current_version_infos, current_update, &mutator);
        merge_send_infos(previous_version_infos, previous_update, &mutator);
        // Update the remote valid fields
        remote_valid_fields |= update_mask;
        // Remove our outstanding request
#ifdef DEBUG_LEGION
        assert(outstanding_requests.find(done) != outstanding_requests.end());
        sanity_check();
#endif
        outstanding_requests.erase(done);
      }
      // Remove the extra valid references that we added for the movement
      // of these version state objects, but no need to track the effects
      // these can be fire and forget
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
            current_update.begin(); it != current_update.end(); it++)
        it->first->send_remote_valid_update(it->first->owner_space,
                                            NULL, 1/*count*/, false/*add*/);
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
            previous_update.begin(); it != previous_update.end(); it++)
        it->first->send_remote_valid_update(it->first->owner_space,
                                            NULL, 1/*count*/, false/*add*/);
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
    /*static*/ void VersionManager::merge_send_infos(
                LegionMap<VersionID,ManagerVersions>::aligned &target_infos,
                const LegionMap<VersionState*,FieldMask>::aligned &source_infos,
                      ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator
            it = source_infos.begin(); it != source_infos.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(!it->first->is_owner());
#endif
        target_infos[it->first->version_number].insert(it->first, 
                                            it->second, mutator);
      }
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
        assert(!info.empty());
        // Make sure each field appears once in each version state info
        FieldMask local_version_fields;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          assert(!!it->second); // better not be empty
          assert(local_version_fields * it->second); // better not overlap
          local_version_fields |= it->second;
        }
        // They can overapproximate if there is more than one
        if (info.size() > 1)
          assert(!(local_version_fields - info.get_valid_mask()));
        else
          assert(info.get_valid_mask() == local_version_fields); // beter match
        // Should not overlap with other fields in the current version
        assert(current_version_fields * local_version_fields);
        current_version_fields |= local_version_fields;
      }
      FieldMask previous_version_fields;
      for (LegionMap<VersionID,ManagerVersions>::aligned::const_iterator vit =
            previous_version_infos.begin(); vit != 
            previous_version_infos.end(); vit++)
      {
        const ManagerVersions &info = vit->second;
        assert(!info.empty());
        // Make sure each field appears once in each version state info
        FieldMask local_version_fields;
        for (ManagerVersions::iterator it = vit->second.begin();
              it != vit->second.end(); it++)
        {
          assert(!!it->second); // better not be empty
          assert(local_version_fields * it->second); // better not overlap
          local_version_fields |= it->second;
        }
        // They can overapproxminate if there is more than one
        if (info.size() > 1)
          assert(!(local_version_fields - info.get_valid_mask()));
        else
          assert(info.get_valid_mask() == local_version_fields); // beter match
        // Should not overlap with other fields in the current version
        assert(previous_version_fields * local_version_fields);
        previous_version_fields |= local_version_fields;
      }
    }

    /////////////////////////////////////////////////////////////
    // Version State 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionState::VersionState(VersionID vid, Runtime *rt, DistributedID id,
                               AddressSpaceID own_sp, 
                               RegionTreeNode *node, bool register_now)
      : DistributedCollectable(rt, 
          LEGION_DISTRIBUTED_HELP_ENCODE(id, VERSION_STATE_DC), 
          own_sp, register_now),
        version_number(vid), logical_node(node)
#ifdef DEBUG_LEGION
        , currently_active(true), currently_valid(true)
#endif
    //--------------------------------------------------------------------------
    {
      // If we're not the owner then add a remove gc ref that will
      // be removed by the owner once no copy of this version state
      // is valid anywhere in the system
      if (!is_owner())
        add_base_gc_ref(REMOTE_DID_REF);
#ifdef LEGION_GC
      log_garbage.info("GC Version State %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    VersionState::VersionState(const VersionState &rhs)
      : DistributedCollectable(rhs.runtime, rhs.did,
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
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(!currently_valid);
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
    void VersionState::initialize(ApEvent term_event, const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  const InstanceSet &targets,
                                  InnerContext *context, unsigned init_index,
                                 const std::vector<LogicalView*> &corresponding,
                                  std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(currently_valid);
#endif
      const UniqueID init_op_id = context->get_unique_id();
      WrapperReferenceMutator mutator(applied_events);
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        LogicalView *new_view = corresponding[idx];
#ifdef DEBUG_LEGION
        if (new_view->is_materialized_view())
          assert(!new_view->as_materialized_view()->
              manager->is_virtual_instance());
#endif
        const FieldMask &view_mask = targets[idx].get_valid_fields();
        if (new_view->is_instance_view())
        {
          InstanceView *inst_view = new_view->as_instance_view();
          if (inst_view->is_reduction_view())
          {
            ReductionView *view = inst_view->as_reduction_view();
            LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
              track_aligned::iterator finder = reduction_views.find(view); 
            if (finder == reduction_views.end())
            {
              new_view->add_nested_valid_ref(did, &mutator);
              reduction_views[view] = view_mask;
            }
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
            {
              new_view->add_nested_valid_ref(did, &mutator);
              valid_views[new_view] = view_mask;
            }
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
          {
            new_view->add_nested_valid_ref(did, &mutator);
            valid_views[new_view] = view_mask;
          }
          else
            finder->second |= view_mask;
          if (HAS_WRITE(usage))
            dirty_mask |= view_mask;
          // Don't add a user since this is a deferred view and
          // we can't access it anyway
        }
      }
      update_fields |= user_mask;
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
        if (!overlap)
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
    void VersionState::perform_disjoint_close(InterCloseOp *op, unsigned index,
                     InnerContext *context, const FieldMask &closing_mask) const
    //--------------------------------------------------------------------------
    {
      // Need the lock in read only mode to access these data structures
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
      for (LegionMap<LegionColor,StateVersions>::aligned::const_iterator cit =
            open_children.begin(); cit != open_children.end(); cit++)
      {
        FieldMask child_overlap = closing_mask & cit->second.get_valid_mask();
        if (!child_overlap)
          continue;
        RegionTreeNode *child_node = logical_node->get_tree_child(cit->first);
        InterCloseOp::DisjointCloseInfo *info = 
          op->find_disjoint_close_child(index, child_node);
        // If we don't care about this child because of control
        // replication then we can keep going
        if (info == NULL)
          continue;
        // Find the version state infos that we need
        for (StateVersions::iterator it = cit->second.begin();
              it != cit->second.end(); it++)
        {
          FieldMask overlap = it->second & child_overlap;
          if (!overlap)
            continue;
          info->close_mask |= overlap;
          // Request the final version of the state
          it->first->request_final_version_state(context, overlap,
                                                 info->ready_events);
          info->version_info.add_current_version(it->first, overlap,
                                                 false/*path only*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::merge_physical_state(const PhysicalState *state,
                                            const FieldMask &merge_mask,
                                          std::set<RtEvent> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_MERGE_PHYSICAL_STATE_CALL);
#ifdef DEBUG_LEGION
      assert(currently_valid);
#endif
      WrapperReferenceMutator mutator(applied_conditions);
      AutoLock s_lock(state_lock);
      if (!!state->dirty_mask)
        dirty_mask |= (state->dirty_mask & merge_mask);
      FieldMask reduction_merge = state->reduction_mask & merge_mask;
      if (!!reduction_merge)
        reduction_mask |= reduction_merge;
      for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
            track_aligned::const_iterator it = state->valid_views.begin();
            it != state->valid_views.end(); it++)
      {
#ifdef DEBUG_LEGION
        if (it->first->is_materialized_view())
          assert(!it->first->as_materialized_view()->
              manager->is_virtual_instance());
#endif
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
      // Tell our owner node that we have valid data
      if (!is_owner() && !update_fields)
        send_valid_notification(applied_conditions);
      update_fields |= merge_mask;
    }

    //--------------------------------------------------------------------------
    void VersionState::reduce_open_children(const LegionColor child_color,
                                            const FieldMask &update_mask,
                                            VersioningSet<> &new_states,
                                            std::set<RtEvent> &applied_events,
                                            bool need_lock, bool local_update)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        // Hold the lock in exclusive mode since we might change things
        AutoLock s_lock(state_lock);
        reduce_open_children(child_color, update_mask, new_states, 
                             applied_events, false/*need lock*/, local_update);
        return;
      }
      WrapperReferenceMutator mutator(applied_events);
      LegionMap<LegionColor,StateVersions>::aligned::iterator finder =
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
        for (VersioningSet<>::iterator it = new_states.begin();
              it != new_states.end(); it++)
        {
          FieldMask overlap = it->second & update_mask;
          if (!overlap)
            continue;
          local_states.insert(it->first, overlap, &mutator);
        }
      }
      else
        finder->second.reduce(update_mask, new_states, &mutator);
      if (local_update)
      {
        // Tell our owner node that we have valid data
        if (!is_owner() && !update_fields)
          send_valid_notification(applied_events);
        // Update the valid fields
        update_fields |= update_mask;
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::send_valid_notification(
                                        std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      RtUserEvent done_event = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(done_event);
      }
      runtime->send_version_state_valid_notification(owner_space, rez);
      applied_events.insert(done_event);
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_valid_notification(
                                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock);
      remote_valid_instances.add(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_version_state_valid_notification(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done;
      derez.deserialize(done);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      VersionState *vs = dynamic_cast<VersionState*>(target);
      assert(vs != NULL);
#else
      VersionState *vs = static_cast<VersionState*>(target);
#endif
      vs->handle_version_state_valid_notification(source);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // This is a little weird, but we track validity in active
      // for VersionStates so that we can use the valid references
      // to track if any copy of a VersionState is valid anywhere
      // The owner then holds gc references to all remote version
      // state and can remove them when no copy of the version 
      // state is valid anywhere else
#ifdef DEBUG_LEGION
      // This should be monotonic on all instances of the version state
      assert(currently_valid);
#endif
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_valid);
      currently_valid = false;
#endif
      // If we're the owner remove the gc references that are held by 
      // each remote copy of the version state object, see the constructor
      // of the VersionState to see where this was added
      if (is_owner() && has_remote_instances())
      {
        UpdateReferenceFunctor<GC_REF_KIND,false/*add*/> functor(this, mutator);
        map_over_remote_instances(functor);
      }
      // We can clear out our open children since we don't need them anymore
      // which will also remove the valid references
      open_children.clear();
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_nested_valid_ref(did, mutator))
          delete it->first;
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        if (it->first->remove_nested_valid_ref(did, mutator))
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, then we have to tell the owner we're valid
      if (!is_owner())
        send_remote_valid_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void VersionState::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // When we are no longer valid we have to send a reference back
      // to our owner to indicate that we are no longer valid
      if (!is_owner())
        send_remote_valid_update(owner_space, mutator, 1/*count*/,false/*add*/);
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
      proxy_this->send_version_state_update_request(target, context, 
          requestor, ready_event, mask, KIND);
      preconditions.insert(ready_event);
    }

    //--------------------------------------------------------------------------
    void VersionState::request_children_version_state(InnerContext *context,
                    const FieldMask &req_mask, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, VERSION_STATE_REQUEST_CHILDREN_CALL);
#ifdef DEBUG_LEGION
      assert(context != NULL);
#endif
      if (is_owner())
      {
        // We are the owner node so see if we need to do any reequests
        // to remote nodes to get our valid data
        AutoLock s_lock(state_lock);
        if (!remote_valid_instances.empty())
        {
          // We always have to request these from scratch for
          // disjoint closes in case we do several of them
          // If we still have remaining fields, we have to send requests to
          // all the other nodes asking for their data
          std::set<RtEvent> local_preconditions;
          RequestFunctor<CHILD_VERSION_REQUEST> functor(this, context, 
              local_space, req_mask, local_preconditions);
          remote_valid_instances.map(functor);
          RtEvent ready_event = Runtime::merge_events(local_preconditions);
          preconditions.insert(ready_event);
        }
        // otherwise we're the only copy so there is nothing to do
      }
      else
      {
        // We are not the owner so figure out which fields we still need to
        // send a request for
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        send_version_state_update_request(owner_space, context, local_space,
            ready_event, req_mask, CHILD_VERSION_REQUEST);
        // Save the event indicating when the fields will be ready
        preconditions.insert(ready_event);
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::request_initial_version_state(InnerContext *context,
                const FieldMask &request_mask, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, VERSION_STATE_REQUEST_INITIAL_CALL);
#ifdef DEBUG_LEGION
      assert(context != NULL);
#endif
      FieldMask needed_fields = request_mask;
      if (is_owner())
      {
        // We're the owner, if we have remote copies then send a 
        // request to them for the needed fields
        AutoLock s_lock(state_lock);
        if (!remote_valid_instances.empty())
        {
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
            RequestFunctor<INITIAL_VERSION_REQUEST> functor(this, context,
                local_space, needed_fields, local_preconditions);
            remote_valid_instances.map(functor);
            RtEvent ready_event = Runtime::merge_events(local_preconditions);
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
          send_version_state_update_request(owner_space, context, local_space,
              ready_event, needed_fields, INITIAL_VERSION_REQUEST);
          // Save the event indicating when the fields will be ready
          initial_events[ready_event] = needed_fields;
          preconditions.insert(ready_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::request_final_version_state(InnerContext *context,
                    const FieldMask &req_mask, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, VERSION_STATE_REQUEST_FINAL_CALL);
#ifdef DEBUG_LEGION
      assert(context != NULL);
#endif
      if (is_owner())
      {
        // We are the owner node so see if we need to do any reequests
        // to remote nodes to get our valid data
        AutoLock s_lock(state_lock);
        if (!remote_valid_instances.empty())
        {
          // Figure out which fields we need to request
          FieldMask remaining_mask = req_mask;
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
            RequestFunctor<FINAL_VERSION_REQUEST> functor(this, context,
                local_space, remaining_mask, local_preconditions);
            remote_valid_instances.map(functor);
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
          send_version_state_update_request(owner_space, context, local_space,
              ready_event, remaining_mask, FINAL_VERSION_REQUEST);
          // Save the event indicating when the fields will be ready
          final_events[ready_event] = remaining_mask;
          preconditions.insert(ready_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state_update(AddressSpaceID target,
                                                InnerContext *context,
                                                const FieldMask &request_mask,
                                                VersionRequestKind request_kind,
                                                RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_SEND_STATE_CALL);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(context);
        rez.serialize(to_trigger);
        rez.serialize(request_mask);
        rez.serialize(request_kind);
        // Hold the lock in read-only mode while iterating these structures
        AutoLock s_lock(state_lock,1,false/*exclusive*/);
        // See if we should send all the fields or just do a partial send
        if (!(update_fields - request_mask))
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
            for (LegionMap<LegionColor,StateVersions>::aligned::const_iterator 
                 cit = open_children.begin(); cit != open_children.end(); cit++)
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
            // Sort the valid views into materialized and deferred
            std::vector<MaterializedView*> materialized;
            std::vector<LogicalView*> deferred;
            for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
                  track_aligned::const_iterator it = valid_views.begin(); it !=
                  valid_views.end(); it++)
            {
              if (it->first->is_materialized_view())
              {
                materialized.push_back(it->first->as_materialized_view());
              }
              else
              {
#ifdef DEBUG_LEGION
                assert(it->first->is_deferred_view());
#endif
                deferred.push_back(it->first);
              }
            }
            rez.serialize<size_t>(materialized.size());
            for (std::vector<MaterializedView*>::const_iterator it = 
                  materialized.begin(); it != materialized.end(); it++)
            {
              // Serialize the DID of the manager and not the view
              // it will be converted on the far side to get the
              // proper subview
              rez.serialize((*it)->manager->did);
              rez.serialize(valid_views[*it]);
            }
            rez.serialize<size_t>(deferred.size());
            for (std::vector<LogicalView*>::const_iterator it = 
                  deferred.begin(); it != deferred.end(); it++)
            {
              rez.serialize((*it)->did);
              rez.serialize(valid_views[*it]);
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
              for (LegionMap<LegionColor,StateVersions>::aligned::const_iterator
                    cit = open_children.begin(); 
                    cit != open_children.end(); cit++)
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
              // Sort into materialized and deferred
              LegionMap<MaterializedView*,FieldMask>::aligned materialized;
              Serializer valid_rez;
              size_t count = 0;
              for (LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
                    track_aligned::const_iterator it = valid_views.begin(); 
                    it != valid_views.end(); it++)
              {
                FieldMask overlap = it->second & request_mask;
                if (!overlap)
                  continue;
                if (it->first->is_materialized_view())
                {
                  materialized[it->first->as_materialized_view()] = overlap;
                  continue;
                }
#ifdef DEBUG_LEGION
                assert(it->first->is_deferred_view());
#endif
                valid_rez.serialize(it->first->did);
                valid_rez.serialize(overlap);
                count++;
              }
              rez.serialize<size_t>(materialized.size());
              for (LegionMap<MaterializedView*,FieldMask>::aligned::
                    const_iterator it = materialized.begin(); it !=
                    materialized.end(); it++)
              {
                // Serialize the manager DID, it will get converted
                // to the proper subview on the far side
                rez.serialize(it->first->manager->did);
                rez.serialize(it->second);
              }
              rez.serialize(count);
              rez.serialize(valid_rez.get_buffer(), valid_rez.get_used_bytes());
            }
            else
            {
              rez.serialize<size_t>(0); // instance views
              rez.serialize<size_t>(0); // valid views
            }
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
      }
      runtime->send_version_state_update_response(target, rez);
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state_update_request(AddressSpaceID target,
                                                InnerContext *context, 
                                                AddressSpaceID source,
                                                RtUserEvent to_trigger,
                                                const FieldMask &request_mask, 
                                                VersionRequestKind request_kind) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!request_mask);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(source);
        rez.serialize(context);
        rez.serialize(to_trigger);
        rez.serialize(request_kind);
        rez.serialize(request_mask);
      }
      runtime->send_version_state_update_request(target, rez);
    }

    //--------------------------------------------------------------------------
    void VersionState::launch_send_version_state_update(AddressSpaceID target,
                                                InnerContext *context,
                                                RtUserEvent to_trigger, 
                                                const FieldMask &request_mask, 
                                                VersionRequestKind request_kind,
                                                RtEvent precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!request_mask);
#endif
      SendVersionStateArgs args;
      args.proxy_this = this;
      args.target = target;
      args.context = context;
      args.request_mask = new FieldMask(request_mask);
      args.request_kind = request_kind;
      args.to_trigger = to_trigger;
      // There is imprecision in our tracking of which nodes have valid
      // meta-data for different fields (i.e. we don't track it at all
      // currently), therefore we may get requests for updates that we
      runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                       precondition);
    }

    //--------------------------------------------------------------------------
    void VersionState::send_version_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(currently_valid); // Must be currently valid
      // We should have had a request for this already
      assert(!has_remote_instance(target));
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
        state = new(location) VersionState(version_number,
                                           runtime, did, source, 
                                           node, false/*register now*/);
      else
        state = new VersionState(version_number, runtime, did,
                                 source, node, false/*register now*/);
      // Once construction is complete then we do the registration
      state->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_update_request(
          AddressSpaceID source, InnerContext *context, RtUserEvent to_trigger, 
          VersionRequestKind request_kind, FieldMask &request_mask)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        VERSION_STATE_HANDLE_REQUEST_CALL);
#ifdef DEBUG_LEGION
      assert(currently_valid);
#endif
      // If we are the not the owner, the same thing happens no matter what
      if (!is_owner())
      {
        // If we are not the owner, all we have to do is replay with our state
        AutoLock s_lock(state_lock,1,false/*exclusive*/);
        // Check to see if we have valid fields, if not we
        // can trigger the event immediately
        FieldMask overlap = update_fields & request_mask;
        if (!overlap)
          Runtime::trigger_event(to_trigger);
        else if (request_kind == CHILD_VERSION_REQUEST)
        {
          // See if we have any children we need to send
          bool has_children = false;
          for (LegionMap<LegionColor,StateVersions>::aligned::const_iterator 
                it = open_children.begin(); it != open_children.end(); it++)
          {
            if (it->second.get_valid_mask() * overlap)
              continue;
            has_children = true;
            break;
          }
          if (has_children)
            launch_send_version_state_update(source, context, to_trigger, 
                                             overlap, request_kind);
          else // no children so we can trigger now
            Runtime::trigger_event(to_trigger);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert((request_kind == INITIAL_VERSION_REQUEST) || 
                 (request_kind == FINAL_VERSION_REQUEST));
#endif
          launch_send_version_state_update(source, context, to_trigger, 
                                           overlap, request_kind);
        }
      }
      else
      {
        // We're the owner, see if we're doing an initial requst or not 
        if (request_kind == INITIAL_VERSION_REQUEST)
        {
          AutoLock s_lock(state_lock,1,false/*exclusive*/);
          // See if we have any remote instances
          if (!remote_valid_instances.empty())
          {
            // Send notifications to any remote instances
            FieldMask needed_fields;
            RtEvent local_event;
            {
              // See if we have to send any messages
              FieldMask overlap = request_mask & update_fields;
              if (!!overlap)
              {
                needed_fields = request_mask - overlap;
                if (!!needed_fields)
                {
                  // We're going to have send messages so we need a local event
                  RtUserEvent local = Runtime::create_rt_user_event();
                  launch_send_version_state_update(source, context, local,
                                                   overlap, request_kind);
                  local_event = local;
                }
                else // no messages, so do the normal thing
                {
                  launch_send_version_state_update(source, context, to_trigger,
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
            RequestFunctor<INITIAL_VERSION_REQUEST> functor(this, context,
                source, needed_fields, local_preconditions);
            remote_valid_instances.map(functor);
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
            FieldMask overlap = request_mask & update_fields;
            if (!overlap)
              Runtime::trigger_event(to_trigger);
            else
              launch_send_version_state_update(source, context, to_trigger,
                                               overlap, request_kind);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert((request_kind == CHILD_VERSION_REQUEST) || 
                 (request_kind == FINAL_VERSION_REQUEST));
#endif
          AutoLock s_lock(state_lock,1,false/*exclusive*/);
          // We're the owner handling a child or final version request
          // See if we have any remote instances to handle ourselves
          if (!remote_valid_instances.empty())
          {
            std::set<RtEvent> local_preconditions;
            if (request_kind == CHILD_VERSION_REQUEST)
            {
              RequestFunctor<CHILD_VERSION_REQUEST> functor(this, context,
                  source, request_mask, local_preconditions);
              remote_valid_instances.map(functor);
            }
            else
            {
              RequestFunctor<FINAL_VERSION_REQUEST> functor(this, context,
                  source, request_mask, local_preconditions);
              remote_valid_instances.map(functor);
            }
            if (!local_preconditions.empty())
            {
              FieldMask overlap = update_fields & request_mask;
              if (!!overlap)
              {
                RtUserEvent local_event = Runtime::create_rt_user_event();
                launch_send_version_state_update(source, context, local_event, 
                                                 overlap, request_kind);
                local_preconditions.insert(local_event);
              }
              Runtime::trigger_event(to_trigger,
                  Runtime::merge_events(local_preconditions));
            }
            else
            {
              // Didn't send any messages so do the normal path
              FieldMask overlap = update_fields & request_mask;
              if (!overlap)
                Runtime::trigger_event(to_trigger);
              else
                launch_send_version_state_update(source, context, to_trigger,
                                                 overlap, request_kind);
            }
          }
          else // We just have to send our local state
          {
            FieldMask overlap = update_fields & request_mask;
            if (!overlap)
              Runtime::trigger_event(to_trigger);
            else
              launch_send_version_state_update(source, context, to_trigger,
                                               overlap, request_kind);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::handle_version_state_update_response(
                                                InnerContext *context,
                                                RtUserEvent to_trigger, 
                                                Deserializer &derez,
                                                const FieldMask &update,
                                                VersionRequestKind request_kind)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, VERSION_STATE_HANDLE_RESPONSE_CALL);
#ifdef DEBUG_LEGION
      assert(currently_valid);
#endif
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
        const bool in_place = !dirty_mask && !reduction_mask &&
                              open_children.empty() && valid_views.empty() && 
                              reduction_views.empty();
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
              LegionColor child;
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
                RtEvent precondition = 
                  Runtime::merge_events(deferred_children_events);
                if (!precondition.has_triggered())
                {
                  // Launch a task to do the merge later
                  UpdateStateReduceArgs args;
                  args.proxy_this = this;
                  args.child_color = child;
                  // Takes ownership for deallocation
                  args.children = deferred_children;
                  // Need resource priority since we asked for the lock
                  RtEvent done = runtime->issue_runtime_meta_task(args, 
                          LG_LATENCY_WORK_PRIORITY, precondition);
                  preconditions.insert(done);
                }
                else // We can run it now
                {
                  FieldMask update_mask;
                  for (VersioningSet<>::iterator it = 
                        deferred_children->begin(); it != 
                        deferred_children->end(); it++)
                    update_mask |= it->second;
                  reduce_open_children(child, update_mask, *deferred_children,
                      preconditions, false/*need lock*/, false/*local update*/);
                  delete deferred_children;
                }
              }
            }
          }
          else
          {
            size_t num_children;
            derez.deserialize(num_children);
            for (unsigned idx = 0; idx < num_children; idx++)
            {
              LegionColor child;
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
                RtEvent precondition = 
                  Runtime::merge_events(reduce_preconditions);
                if (!precondition.has_triggered())
                {
                  // Launch a task to do the merge later
                  UpdateStateReduceArgs args;
                  args.proxy_this = this;
                  args.child_color = child;
                  // Takes ownership for deallocation
                  args.children = reduce_children;
                  // Need resource priority since we asked for the lock
                  RtEvent done = runtime->issue_runtime_meta_task(args,
                          LG_LATENCY_WORK_PRIORITY, precondition);
                  preconditions.insert(done);
                }
                else // We can run it now
                {
                  reduce_open_children(child, update_mask, *reduce_children,
                                       preconditions, false/*need lock*/,
                                       false/*local udpate*/);
                  delete reduce_children;
                }
              }
              else
              {
                // We can do the merge now
                reduce_open_children(child, update_mask, *reduce_children,
                                     preconditions, false/*need lock*/,
                                     false/*local update*/);
                delete reduce_children;
              }
            }
          }
        }
        if (request_kind != CHILD_VERSION_REQUEST)
        {
          // Finally do the views
          // Materialized Views have to convert to the proper
          // subview, so there is no point in doing anything
          // special with them since we'll need to convert
          // them anyway
          size_t num_instance_views;
          derez.deserialize(num_instance_views);
          for (unsigned idx = 0; idx < num_instance_views; idx++)
          {
            DistributedID manager_did;
            derez.deserialize(manager_did);
            RtEvent ready;
            PhysicalManager *manager = 
              runtime->find_or_request_physical_manager(manager_did, ready);
            LegionMap<PhysicalManager*,
              std::pair<RtEvent,FieldMask> >::aligned::iterator finder = 
                pending_instances.find(manager);
            if (finder == pending_instances.end())
            {
              ConvertViewArgs args;
              args.proxy_this = this;
              args.manager = manager;
              args.context = context;
              std::pair<RtEvent,FieldMask> &entry = pending_instances[manager];
              entry.first = runtime->issue_runtime_meta_task(args,
                                       LG_LATENCY_WORK_PRIORITY, ready);
              derez.deserialize(entry.second);
              preconditions.insert(entry.first);
            }
            else
            {
              // Already pending, so we can just add our fields 
              FieldMask update_mask;
              derez.deserialize(update_mask);
              finder->second.second |= update_mask;
              preconditions.insert(finder->second.first);
            }
          }
          if (in_place)
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
#ifdef DEBUG_LEGION
              assert(valid_views.find(view) == valid_views.end());
#endif
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
#ifdef DEBUG_LEGION
              assert(reduction_views.find(red_view) == reduction_views.end());
#endif
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
                  preconditions.insert(ready);
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
                  preconditions.insert(ready);
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
      }
      if (!pending_views.empty())
      {
        UpdateViewReferences args;
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
              runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                                               it->second));
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
      std::set<RtEvent> done_events;
      // Lock was acquired by the caller
      reduce_args->proxy_this->reduce_open_children(reduce_args->child_color,
          update_mask, *reduce_args->children, done_events, 
          true/*need lock*/, false/*local update*/);
      delete reduce_args->children;
      // Wait for all the effects to be applied
      if (!done_events.empty())
      {
        RtEvent done = Runtime::merge_events(done_events);
        done.wait();
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::remove_version_state_ref(ReferenceSource ref_kind,
                                                RtEvent done_event)
    //--------------------------------------------------------------------------
    {
      RemoveVersionStateRefArgs args;
      args.proxy_this = this;
      args.ref_kind = ref_kind;
      runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                                       done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_remove_version_state_ref(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const RemoveVersionStateRefArgs *ref_args = 
        (const RemoveVersionStateRefArgs*)args;
      if (ref_args->proxy_this->remove_base_valid_ref(ref_args->ref_kind))
        delete ref_args->proxy_this;     
    }

    //--------------------------------------------------------------------------
    void VersionState::convert_view(PhysicalManager *manager,
                               InnerContext *context, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!manager->is_virtual_manager());
#endif
      InstanceView *view = logical_node->convert_manager(manager, context);
      AutoLock s_lock(state_lock);
      LegionMap<PhysicalManager*,
                std::pair<RtEvent,FieldMask> >::aligned::iterator finder = 
                  pending_instances.find(manager);
#ifdef DEBUG_LEGION
      assert(finder != pending_instances.end());
#endif
      // See if it is already in our list of valid views
      LegionMap<LogicalView*,FieldMask>::aligned::iterator view_finder = 
        valid_views.find(view);
      if (view_finder == valid_views.end())
      {
        view->add_nested_valid_ref(did, mutator);
        valid_views[view] = finder->second.second;
      }
      else
        view_finder->second |= finder->second.second;
      // Remove the entry 
      pending_instances.erase(finder);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionState::process_convert_view(const void *args)
    //--------------------------------------------------------------------------
    {
      const ConvertViewArgs *view_args = (const ConvertViewArgs*)args;
      LocalReferenceMutator mutator;
      view_args->proxy_this->convert_view(view_args->manager,
                                          view_args->context, &mutator);
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
      InnerContext *context;
      derez.deserialize(context);
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
      vs->handle_version_state_update_request(source, context, to_trigger, 
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
      InnerContext *context;
      derez.deserialize(context);
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
      vs->handle_version_state_update_response(context, to_trigger, derez, 
                                               update_mask, request_kind);
    } 

    //--------------------------------------------------------------------------
    void VersionState::capture_root(CompositeView *target, 
                 const FieldMask &capture_mask, ReferenceMutator *mutator) const
    //--------------------------------------------------------------------------
    { 
      // We'll only capture nested composite views if we have no choice
      // If we have any other kind of instance for those fields, then there
      // is no reason to capture a nested composite instance
      FieldMask non_composite_capture;
      LegionMap<CompositeView*,FieldMask>::aligned composite_views;
      {
        // Only need this in read only mode since we're just reading
        AutoLock s_lock(state_lock,1,false/*exclusive*/);
        for (LegionMap<LegionColor,StateVersions>::aligned::const_iterator cit =
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
            target->record_child_version_state(cit->first, it->first, 
                                               overlap, mutator);
          }
        }
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
            if (!it->first->is_composite_view())
            {
              target->record_valid_view(it->first, overlap);
              non_composite_capture |= overlap;
            }
            else
              composite_views[it->first->as_composite_view()] = overlap;
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
      if (!composite_views.empty())
      {
        if (!!non_composite_capture)
        {
          for (LegionMap<CompositeView*,FieldMask>::aligned::iterator it =
                composite_views.begin(); it != composite_views.end(); it++)
          {
            it->second -= non_composite_capture;
            if (!!it->second)
              target->record_valid_view(it->first, it->second);
          }
        }
        else
        {
          for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it =
                composite_views.begin(); it != composite_views.end(); it++)
            target->record_valid_view(it->first, it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::capture(CompositeNode *target,
                               const FieldMask &capture_mask,
                               ReferenceMutator *mutator) const
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
      for (LegionMap<LegionColor,StateVersions>::aligned::const_iterator cit =
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
          target->record_child_version_state(cit->first, it->first, 
                                             overlap, mutator);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionState::capture_dirty_instances(const FieldMask &capture_mask,
                                               VersionState *target) const
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(state_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      // This assertion doesn't hold true under virtual mappings where
      // an advance operation might think there is dirty field data but
      // it was never actually initialized in the outermost task
      //assert(!(capture_mask - (dirty_mask | reduction_mask)));
      assert((this->version_number + 1) == target->version_number);
#endif
      FieldMask dirty_overlap = dirty_mask & capture_mask;
      if (!!dirty_overlap)
      {
        target->dirty_mask |= dirty_overlap;
        target->update_fields |= dirty_overlap;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = it->second & dirty_overlap;
          if (!overlap)
            continue;
          LegionMap<LogicalView*,FieldMask,VALID_VIEW_ALLOC>::
           track_aligned::iterator finder = target->valid_views.find(it->first);
          if (finder == target->valid_views.end())
          {
#ifdef DEBUG_LEGION
            assert(target->currently_valid);
#endif
            // No need for a mutator here, it is already valid
            it->first->add_nested_valid_ref(target->did);
            target->valid_views[it->first] = overlap;
          }
          else
            finder->second |= overlap;
        }
      }
      FieldMask reduction_overlap = reduction_mask & capture_mask;
      if (!!reduction_overlap)
      {
        target->reduction_mask |= reduction_overlap;
        target->update_fields |= reduction_overlap;
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
              reduction_views.begin(); it != reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & reduction_overlap;
          if (!overlap)
            continue;
          LegionMap<ReductionView*,FieldMask,VALID_REDUCTION_ALLOC>::
                track_aligned::iterator finder = 
                      target->reduction_views.find(it->first);
          if (finder == target->reduction_views.end())
          {
#ifdef DEBUG_LEGION
            assert(target->currently_valid);
#endif
            // No need for a mutator here, it is already valid
            it->first->add_nested_valid_ref(target->did);
            target->reduction_views[it->first] = overlap;
          }
          else
            finder->second |= overlap;
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
      path.resize(max_depth+1, INVALID_COLOR);
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::register_child(unsigned depth, 
                                        const LegionColor color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      path[depth] = color;
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::record_aliased_children(unsigned depth,
                                                 const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      LegionMap<unsigned,FieldMask>::aligned::iterator finder = 
        interfering_children.find(depth);
      if (finder == interfering_children.end())
        interfering_children[depth] = mask;
      else
        finder->second |= mask;
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
      return (path[depth] != INVALID_COLOR);
    }

    //--------------------------------------------------------------------------
    LegionColor RegionTreePath::get_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      assert(has_child(depth));
      return path[depth];
    }
#endif

    //--------------------------------------------------------------------------
    const FieldMask* RegionTreePath::get_aliased_children(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      if (interfering_children.empty())
        return NULL;
      LegionMap<unsigned,FieldMask>::aligned::const_iterator finder = 
        interfering_children.find(depth);
      if (finder == interfering_children.end())
        return NULL;
      return &(finder->second);
    }

    /////////////////////////////////////////////////////////////
    // InstanceRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(bool comp)
      : ready_event(ApEvent::NO_AP_EVENT), manager(NULL), local(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(const InstanceRef &rhs)
      : valid_fields(rhs.valid_fields), ready_event(rhs.ready_event),
        manager(rhs.manager), local(rhs.local)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(PhysicalManager *man, const FieldMask &m,ApEvent r)
      : valid_fields(m), ready_event(r), manager(man), local(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::~InstanceRef(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef& InstanceRef::operator=(const InstanceRef &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields = rhs.valid_fields;
      ready_event = rhs.ready_event;
      local = rhs.local;
      manager = rhs.manager;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator==(const InstanceRef &rhs) const
    //--------------------------------------------------------------------------
    {
      if (valid_fields != rhs.valid_fields)
        return false;
      if (ready_event != rhs.ready_event)
        return false;
      if (manager != rhs.manager)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator!=(const InstanceRef &rhs) const
    //--------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //--------------------------------------------------------------------------
    MappingInstance InstanceRef::get_mapping_instance(void) const
    //--------------------------------------------------------------------------
    {
      return MappingInstance(manager);
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::is_virtual_ref(void) const
    //--------------------------------------------------------------------------
    {
      if (manager == NULL)
        return true;
      return manager->is_virtual_manager(); 
    }

    //--------------------------------------------------------------------------
    void InstanceRef::add_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      manager->add_base_valid_ref(source);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      if (manager->remove_base_valid_ref(source))
        delete manager;
    }

    //--------------------------------------------------------------------------
    Memory InstanceRef::get_memory(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      return manager->get_memory();
    }

    //--------------------------------------------------------------------------
    Reservation InstanceRef::get_read_only_reservation(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
      assert(manager->is_instance_manager());
#endif
      return 
        manager->as_instance_manager()->get_read_only_mapping_reservation();
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::is_field_set(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      FieldSpaceNode *field_node = manager->region_node->column_source; 
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
      assert(manager != NULL);
#endif
      return manager->get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        InstanceRef::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      return manager->get_field_accessor(fid);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::pack_reference(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(valid_fields);
      rez.serialize(ready_event);
      if (manager != NULL)
        rez.serialize(manager->did);
      else
        rez.serialize<DistributedID>(0);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::unpack_reference(Runtime *runtime,
                                       Deserializer &derez, RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(valid_fields);
      derez.deserialize(ready_event);
      DistributedID did;
      derez.deserialize(did);
      if (did == 0)
        return;
      manager = runtime->find_or_request_physical_manager(did, ready);
      local = false;
    } 

    /////////////////////////////////////////////////////////////
    // InstanceSet 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    InstanceSet::CollectableRef& InstanceSet::CollectableRef::operator=(
                                         const InstanceSet::CollectableRef &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields = rhs.valid_fields;
      ready_event = rhs.ready_event;
      local = rhs.local;
      manager = rhs.manager;
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
        refs.single = new CollectableRef();
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
          delete (refs.single);
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
          delete (refs.single);
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
            new CollectableRef(*refs.single);
          next->add_reference();
          if (refs.single->remove_reference())
            delete (refs.single);
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
      else if (!single && refs.multi->empty())
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
            delete (refs.single);
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
              delete (refs.single);
          }
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        else if (refs.single == NULL)
        {
          // New size is 1 but we were empty before
          CollectableRef *next = new CollectableRef();
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
            new CollectableRef(refs.multi->vector[0]);
          if (refs.multi->remove_reference())
            delete (refs.multi);
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
          delete (refs.single);
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
            delete (refs.single);
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        else
        {
          refs.single = new CollectableRef(ref);
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
    bool InstanceSet::is_virtual_mapping(void) const
    //--------------------------------------------------------------------------
    {
      if (empty())
        return true;
      if (size() > 1)
        return false;
      return refs.single->is_virtual_ref();
    }

    //--------------------------------------------------------------------------
    void InstanceSet::pack_references(Serializer &rez) const
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
        refs.single->pack_reference(rez);
      }
      else
      {
        rez.serialize<size_t>(refs.multi->vector.size());
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].pack_reference(rez);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::unpack_references(Runtime *runtime, Deserializer &derez, 
                                        std::set<RtEvent> &ready_events)
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
            delete (refs.single);
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
          refs.single = new CollectableRef();
          refs.single->add_reference();
        }
        RtEvent ready;
        refs.single->unpack_reference(runtime, derez, ready);
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
            delete (refs.single);
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
          refs.multi->vector[idx].unpack_reference(runtime, derez, ready);
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
    void InstanceSet::find_read_only_reservations(
                                             std::set<Reservation> &locks) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          locks.insert(refs.single->get_read_only_reservation());
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          locks.insert(refs.multi->vector[idx].get_read_only_reservation());
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
    VersioningInvalidator::VersioningInvalidator(void)
      : ctx(0), invalidate_all(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersioningInvalidator::VersioningInvalidator(RegionTreeContext c)
      : ctx(c.get_id()), invalidate_all(!c.exists())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool VersioningInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      if (invalidate_all)
        node->invalidate_version_managers();
      else
        node->invalidate_version_state(ctx);
      return true;
    }

    //--------------------------------------------------------------------------
    bool VersioningInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      if (invalidate_all)
        node->invalidate_version_managers();
      else
        node->invalidate_version_state(ctx);
      return true;
    }

  }; // namespace Internal 
}; // namespace Legion

