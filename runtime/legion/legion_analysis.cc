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
    PhysicalUser::PhysicalUser(IndexSpaceExpression *e)
      : expr(e)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      expr->add_expression_reference();
    }
    
    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const RegionUsage &u, const LegionColor c,
                               UniqueID id, unsigned x, IndexSpaceExpression *e)
      : usage(u), child(c), op_id(id), index(x), expr(e)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      expr->add_expression_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const PhysicalUser &rhs) 
      : expr(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalUser::~PhysicalUser(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      if (expr->remove_expression_reference())
        delete expr;
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
    void PhysicalUser::pack_user(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      expr->pack_expression(rez, target);
      rez.serialize(child);
      rez.serialize(usage.privilege);
      rez.serialize(usage.prop);
      rez.serialize(usage.redop);
      rez.serialize(op_id);
      rez.serialize(index);
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalUser* PhysicalUser::unpack_user(Deserializer &derez,
        bool add_reference, RegionTreeForest *forest, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceExpression *expr = 
          IndexSpaceExpression::unpack_expression(derez, forest, source);
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      PhysicalUser *result = new PhysicalUser(expr);
      derez.deserialize(result->child);
      derez.deserialize(result->usage.privilege);
      derez.deserialize(result->usage.prop);
      derez.deserialize(result->usage.redop);
      derez.deserialize(result->op_id);
      derez.deserialize(result->index);
      if (add_reference)
        result->add_reference();
      return result;
    } 

    /////////////////////////////////////////////////////////////
    // VersionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VersionInfo::~VersionInfo(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!mapped_event.exists());
      assert(equivalence_sets.empty());
#endif
    }

    //--------------------------------------------------------------------------
    VersionInfo& VersionInfo::operator=(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::initialize_mapping(RtEvent mapped)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!mapped_event.exists());
#endif
      mapped_event = mapped;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::record_equivalence_set(EquivalenceSet *set,bool need_lock)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION 
      assert(mapped_event.exists());
#endif
      std::pair<std::set<EquivalenceSet*>::iterator,bool> result = 
        equivalence_sets.insert(set);
      // If we added this element then need to add a reference to it
      if (result.second)
      {
        set->add_base_resource_ref(VERSION_INFO_REF);
        set->add_mapping_guard(mapped_event, need_lock);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::make_ready(const RegionRequirement &req, 
                                 const FieldMask &ready_mask,
                                 std::set<RtEvent> &ready_events,
                                 std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapped_event.exists());
#endif
      // We only need an exclusive mode for this operation if we're 
      // writing otherwise, we know we can do things with a shared copy
      for (std::set<EquivalenceSet*>::const_iterator it = 
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        (*it)->request_valid_copy(ready_mask, RegionUsage(req), 
                                  ready_events, applied_events);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::finalize_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapped_event.exists());
#endif
      if (!equivalence_sets.empty())
      {
        for (std::set<EquivalenceSet*>::const_iterator it = 
              equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        {
          (*it)->remove_mapping_guard(mapped_event);
          if ((*it)->remove_base_resource_ref(VERSION_INFO_REF))
            delete (*it);
        }
        equivalence_sets.clear();
      }
      mapped_event = RtEvent::NO_RT_EVENT;
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
    // PhysicalTraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(Operation *o, bool initialize)
    //--------------------------------------------------------------------------
      : op(o), tpl((op == NULL) ? NULL : 
          (op->get_memoizable() == NULL) ? NULL :
            op->get_memoizable()->get_template()),
        recording((tpl == NULL) ? false : tpl->is_recording())
    {
      if (recording && initialize)
        tpl->record_get_term_event(op->get_memoizable());
    }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(Operation *o, Memoizable *memo)
      : op(o), tpl(memo->get_template()),
        recording((tpl == NULL) ? false : tpl->is_recording())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_merge_events(ApEvent &result,
                                                ApEvent e1, ApEvent e2) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_merge_events(result, e1, e2, op);      
    }
    
    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_merge_events(ApEvent &result,
                                       ApEvent e1, ApEvent e2, ApEvent e3) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_merge_events(result, e1, e2, e3, op);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_merge_events(ApEvent &result,
                                          const std::set<ApEvent> &events) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_merge_events(result, events, op);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_op_sync_event(ApEvent &result) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_set_op_sync_event(result, op);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_issue_copy(ApEvent &result, RegionNode *node,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                 ApEvent precondition,
                                 PredEvent predicate_guard,
                                 IndexTreeNode *intersect,
                                 IndexSpaceExpression *mask,
                                 ReductionOpID redop,
                                 bool reduction_fold) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_issue_copy(op, result, node, src_fields, dst_fields,
        precondition, predicate_guard, intersect, mask, redop, reduction_fold);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_issue_fill(ApEvent &result, RegionNode *node,
                                     const std::vector<CopySrcDstField> &fields,
                                     const void *fill_buffer, size_t fill_size,
                                     ApEvent precondition,
                                     PredEvent predicate_guard,
#ifdef LEGION_SPY
                                     UniqueID fill_uid,
#endif
                                     IndexTreeNode *intersect,
                                     IndexSpaceExpression *mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_issue_fill(op, result, node, fields, fill_buffer, fill_size,
          precondition, predicate_guard,
#ifdef LEGION_SPY
          fill_uid,
#endif
          intersect, mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_empty_copy(DeferredView *view,
                                              const FieldMask &copy_mask,
                                              MaterializedView *dst) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_empty_copy(view, copy_mask, dst);
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
            runtime->forest->get_node(launch_space) : NULL)
    //--------------------------------------------------------------------------
    {
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
      write_projections[function].insert(node);
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
    void LogicalState::update_projection_epochs(FieldMask capture_mask,
                                                const ProjectionInfo &info)
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
        capture_mask -= overlap;
        if (!capture_mask)
          return;
      }
      // If it didn't already exist, start a new projection epoch
      ProjectionEpoch *new_epoch = 
        new ProjectionEpoch(ProjectionEpoch::first_epoch, capture_mask);
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
    // Logical Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(ContextID c, const LogicalUser &u, 
                                 RegionTreeNode *r, bool val)
      : ctx(c), user(u), root_node(r), validates(val), close_op(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(const LogicalCloser &rhs)
      : user(rhs.user), root_node(rhs.root_node), validates(rhs.validates)
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
    void LogicalCloser::record_close_operation(const FieldMask &mask) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      close_mask |= mask;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_closed_user(const LogicalUser &user,
                                           const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      closed_users.push_back(user);
      LogicalUser &closed_user = closed_users.back();
      closed_user.field_mask = mask;
    }

#ifndef LEGION_SPY
    //--------------------------------------------------------------------------
    void LogicalCloser::pop_closed_user(void)
    //--------------------------------------------------------------------------
    {
      closed_users.pop_back();
    }
#endif

    //--------------------------------------------------------------------------
    void LogicalCloser::initialize_close_operations(LogicalState &state, 
                                             Operation *creator,
                                             const LogicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // These sets of fields better be disjoint
      assert(!!close_mask);
      assert(close_op == NULL);
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
      close_op = creator->runtime->get_available_merge_close_op();
      merge_close_gen = close_op->get_generation();
      req.privilege_fields.clear();
      root_node->column_source->get_field_set(close_mask,
                                             trace_info.req.privilege_fields,
                                             req.privilege_fields);
      close_op->initialize(creator->get_context(), req, trace_info, 
                           trace_info.req_idx, close_mask, creator);
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
      const LogicalUser merge_close_user(close_op, 0/*idx*/, 
          RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), close_mask);
      register_dependences(close_op, merge_close_user, current, 
          open_below, closed_users, above_users, cusers, pusers);
      // Now we can remove our references on our local users
      for (LegionList<LogicalUser>::aligned::const_iterator it = 
            closed_users.begin(); it != closed_users.end(); it++)
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
      root_node->filter_prev_epoch_users(state, close_mask);
      root_node->filter_curr_epoch_users(state, close_mask);
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
      const LogicalUser close_user(close_op, merge_close_gen,0/*idx*/,
        RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), close_mask);
      users.push_back(close_user);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(Runtime *rt, DistributedID did,
                 AddressSpaceID owner, IndexSpaceExpression *expr, bool reg_now)
      : DistributedCollectable(rt, did, owner, reg_now), set_expr(expr),
        eq_lock(gc_lock), eq_state(is_owner() ? MAPPING_STATE : INVALID_STATE),
        unrefined_remainder(NULL)
    //--------------------------------------------------------------------------
    {
      set_expr->add_expression_reference();
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(const EquivalenceSet &rhs)
      : DistributedCollectable(rhs), set_expr(NULL), eq_lock(gc_lock)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::~EquivalenceSet(void)
    //--------------------------------------------------------------------------
    {
      if (set_expr->remove_expression_reference())
        delete set_expr;
      if (!subsets.empty())
      {
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
          if ((*it)->remove_nested_resource_ref(did))
            delete (*it);
        subsets.clear();
      }
    }

    //--------------------------------------------------------------------------
    EquivalenceSet& EquivalenceSet::operator=(const EquivalenceSet &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::RefinementThunk::RefinementThunk(IndexSpaceExpression *exp,
                                     EquivalenceSet *own, AddressSpaceID source)
      : owner(own), expr(exp), refinement(NULL)
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = owner->runtime;
      // If we're not the owner send a request to make the refinement set
      if (source != runtime->address_space)
      {
        refinement_ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(this);
          expr->pack_expression(rez, source);
        }
        runtime->send_equivalence_set_create_remote_request(source, rez);
      }
      else
      {
        // We're the owner so we can make the refinement set now
        DistributedID did = runtime->get_available_distributed_id();
        refinement = new EquivalenceSet(runtime, did, source, 
                                        expr, true/*register*/);
      }
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* EquivalenceSet::RefinementThunk::perform_refinement(void)
    //--------------------------------------------------------------------------
    {
      if (refinement_ready.exists() && !refinement_ready.has_triggered())
        refinement_ready.wait();
#ifdef DEBUG_LEGION
      assert(refinement != NULL);
#endif
      refinement->clone_from(owner);
      return refinement;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* EquivalenceSet::RefinementThunk::get_refinement(void)
    //--------------------------------------------------------------------------
    {
      if (refinement_ready.exists() && !refinement_ready.has_triggered())
        refinement_ready.wait();
#ifdef DEBUG_LEGION
      assert(refinement != NULL);
#endif
      return refinement;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::RefinementThunk::record_refinement(
                                          EquivalenceSet *result, RtEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(refinement == NULL);
      assert(refinement_ready.exists());
      assert(!refinement_ready.has_triggered());
#endif
      refinement = result;
      Runtime::trigger_event(refinement_ready, ready);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::LocalRefinement::LocalRefinement(IndexSpaceExpression *expr, 
                                                     EquivalenceSet *owner) 
      : RefinementThunk(expr, owner, owner->runtime->address_space)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::RemoteComplete::RemoteComplete(IndexSpaceExpression *expr, 
                                   EquivalenceSet *owner, AddressSpaceID source)
      : RefinementThunk(expr, owner, source), target(source)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::RemoteComplete::~RemoteComplete(void)
    //--------------------------------------------------------------------------
    {
      // We already hold the lock from the caller
      owner->process_subset_request(target, false/*need lock*/);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::add_mapping_guard(RtEvent mapped_event, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock eq(eq_lock);
        add_mapping_guard(mapped_event, false/*need lock*/);
        return;
      }
#ifdef DEBUG_LEGION
      assert((eq_state == MAPPING_STATE) || 
              (eq_state == PENDING_REFINEMENT_STATE) ||
              (eq_state == PENDING_INVALID_STATE));
      assert(subsets.empty()); // should not have any subsets at this point
      assert(unrefined_remainder == NULL); // nor a remainder
#endif
      std::map<RtEvent,unsigned>::iterator finder = 
        mapping_guards.find(mapped_event);
      if (finder == mapping_guards.end())
        mapping_guards[mapped_event] = 1;
      else
        finder->second++;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::remove_mapping_guard(RtEvent mapped_event)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      std::map<RtEvent,unsigned>::iterator finder = 
        mapping_guards.find(mapped_event);
#ifdef DEBUG_LEGION
      assert((eq_state == MAPPING_STATE) || 
              (eq_state == PENDING_REFINEMENT_STATE) ||
              (eq_state == PENDING_INVALID_STATE));
      assert(finder != mapping_guards.end());
      assert(finder->second > 0);
#endif
      finder->second--;
      if (finder->second == 0)
      {
        mapping_guards.erase(finder);
        if (mapping_guards.empty() && (eq_state != MAPPING_STATE))
        {
          if (is_owner())
          {
#ifdef DEBUG_LEGION
            assert(eq_state == PENDING_REFINEMENT_STATE);
#endif
            eq_state = REFINEMENT_STATE;
            // Kick off the refinement task now that we're transitioning
            launch_refinement_task();
          }
          else
          {
#ifdef DEBUG_LEGION
            assert(eq_state == PENDING_INVALID_STATE);
#endif
            eq_state = INVALID_STATE;
          }
#ifdef DEBUG_LEGION
          assert(transition_event.exists());
#endif
          // Check to see if we have any valid meta-data that needs to be
          // sent back to the owner node
          if (!valid_instances.empty() || !reduction_instances.empty())
          {
            // TODO: Send back any valid data to the owner node

          }
          else
          {
#ifdef DEBUG_LEGION
            assert(version_numbers.empty());
#endif
            // We don't have any valid data so we can just invalidate
            // ourself by triggering the transition event
            Runtime::trigger_event(transition_event);  
          }
          transition_event = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent EquivalenceSet::ray_trace_equivalence_sets(VersionManager *target,
                                                     IndexSpaceExpression *expr,
                                                     AddressSpaceID source) 
    //--------------------------------------------------------------------------
    {
      // If this is not the owner node then send the request there
      if (!is_owner())
      {
        RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(target);
          expr->pack_expression(rez, owner_space);
          rez.serialize(source);
          rez.serialize(done_event);
        }
        runtime->send_equivalence_set_ray_trace_request(owner_space, rez);
        return done_event;
      }
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      // At this point we're on the owner
      std::map<EquivalenceSet*,IndexSpaceExpression*> to_traverse;
      std::map<RefinementThunk*,IndexSpaceExpression*> refinements_to_traverse;
      RtEvent refinement_done;
      {
        RegionTreeForest *forest = runtime->forest;
        AutoLock eq(eq_lock);
        // Ray tracing can happen in parallel with mapping since it's a 
        // read-only process with respect to the subsets, but it can't
        // happen in parallel with any refinements
        while ((eq_state == REFINEMENT_STATE) || 
                (eq_state == PENDING_MAPPING_STATE))
        {
          if (eq_state == REFINEMENT_STATE)
          {
#ifdef DEBUG_LEGION
            assert(!transition_event.exists());
#endif
            transition_event = Runtime::create_rt_user_event();
            eq_state = PENDING_MAPPING_STATE;
          }
#ifdef DEBUG_LEGION
          assert(transition_event.exists());
#endif
          RtEvent wait_on = transition_event;
          eq.release();
          wait_on.wait();
          eq.reacquire();
        }
        // Two cases here, one where refinement has already begun and
        // one where it is just starting
        if (!subsets.empty() || !pending_refinements.empty())
        {
          // Iterate through all the subsets and find any overlapping ones
          // that we need to traverse in order to do our subsets
          bool is_empty = false;
          for (std::vector<EquivalenceSet*>::const_iterator it = 
                subsets.begin(); it != subsets.end(); it++)
          {
            IndexSpaceExpression *overlap = 
              forest->intersect_index_spaces((*it)->set_expr, expr);
            if (overlap->is_empty())
              continue;
            to_traverse[*it] = overlap;
            expr = forest->subtract_index_spaces(expr, overlap);
            if ((expr == NULL) || expr->is_empty())
            {
              is_empty = true;
              break;
            }
          }
          for (std::vector<RefinementThunk*>::const_iterator it = 
                pending_refinements.begin(); !is_empty && (it != 
                pending_refinements.end()); it++)
          {
            IndexSpaceExpression *overlap = 
              forest->intersect_index_spaces((*it)->expr, expr);
            if (overlap->is_empty())
              continue;
            (*it)->add_reference();
            refinements_to_traverse[*it] = overlap;
            // If this is a pending refinement then we'll need to
            // wait for it before traversing farther
            if (!refinement_done.exists())
            {
#ifdef DEBUG_LEGION
              assert(eq_state == PENDING_REFINEMENT_STATE);
              assert(transition_event.exists());
#endif
              refinement_done = transition_event;
            }
            expr = forest->subtract_index_spaces(expr, overlap);
            if ((expr == NULL) || expr->is_empty())
              is_empty = true;
          }
          if (!is_empty)
          {
#ifdef DEBUG_LEGION
            assert(unrefined_remainder != NULL);
#endif
            LocalRefinement *refinement = new LocalRefinement(expr, this);
            refinement->add_reference();
            refinements_to_traverse[refinement] = expr;
            add_pending_refinement(refinement);
            // If this is a pending refinement then we'll need to
            // wait for it before traversing farther
            if (!refinement_done.exists())
            {
#ifdef DEBUG_LEGION
              assert(eq_state == PENDING_REFINEMENT_STATE);
              assert(transition_event.exists());
#endif
              refinement_done = transition_event;
            }
            unrefined_remainder = 
              forest->subtract_index_spaces(unrefined_remainder, expr);
            if ((unrefined_remainder != NULL) && 
                  unrefined_remainder->is_empty())
              unrefined_remainder = NULL;
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(unrefined_remainder == NULL);
#endif
          // See if the set expressions are the same or whether we 
          // need to make a refinement
          IndexSpaceExpression *diff = 
            forest->subtract_index_spaces(set_expr, expr);
          if ((diff != NULL) && !diff->is_empty())
          {
            // Time to refine this since we only need a subset of it
            LocalRefinement *refinement = new LocalRefinement(expr, this);
            refinement->add_reference();
            refinements_to_traverse[refinement] = expr;
            add_pending_refinement(refinement);
#ifdef DEBUG_LEGION
            assert(eq_state == PENDING_REFINEMENT_STATE);
            assert(transition_event.exists());
#endif
            refinement_done = transition_event;
            // Update the unrefined remainder
            unrefined_remainder = diff;
          }
          else
          {
            // Just record this as one of the results
            if (source != runtime->address_space)
            {
              // Not local so we need to send a message
              RtUserEvent recorded_event = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(target);
                rez.serialize(recorded_event);
              }
              runtime->send_equivalence_set_ray_trace_response(source, rez);
              return recorded_event;
            }
            else
            {
              // Local so we can update this directly
              target->record_equivalence_set(this);
              return RtEvent::NO_RT_EVENT;
            }
          }
        }
      }
      // If we have a refinement to do then we need to wait for that
      // to be done before we continue our traversal
      if (refinement_done.exists() && !refinement_done.has_triggered())
        refinement_done.wait();
      // Get the actual equivalence sets for any refinements we needed to
      // wait for because they weren't ready earlier
      if (!refinements_to_traverse.empty())
      {
        for (std::map<RefinementThunk*,IndexSpaceExpression*>::const_iterator
              it = refinements_to_traverse.begin(); 
              it != refinements_to_traverse.end(); it++)
        {
          EquivalenceSet *result = it->first->get_refinement();
#ifdef DEBUG_LEGION
          assert(to_traverse.find(result) == to_traverse.end());
#endif
          to_traverse[result] = it->second;
          if (it->first->remove_reference())
            delete it->first;
        }
      }
      // Finally traverse any subsets that we have to do
      if (!to_traverse.empty())
      {
        std::set<RtEvent> done_events;
        for (std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator 
              it = to_traverse.begin(); it != to_traverse.end(); it++)
        {
          RtEvent done = it->first->ray_trace_equivalence_sets(target, 
                                                  it->second, source);
          if (done.exists())
            done_events.insert(done);
        }
        if (!done_events.empty())
          return Runtime::merge_events(done_events);
        // Otherwise fall through and return a no-event
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::perform_versioning_analysis(VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
      std::vector<EquivalenceSet*> to_recurse;
      {
        AutoLock eq(eq_lock);
        // We need to iterate until we get a valid lease while holding the lock
        if (is_owner())
        {
          // If we have a unrefined remainder then we need that now
          if (unrefined_remainder != NULL)
          {
            LocalRefinement *refinement = 
              new LocalRefinement(unrefined_remainder, this);
            // We can clear the unrefined remainder now
            unrefined_remainder = NULL;
            add_pending_refinement(refinement);
#ifdef DEBUG_LEGION
            // Should not be in the mapping state after this
            // so we know the transition event applies to us
            assert(eq_state != MAPPING_STATE);
#endif
            if (eq_state == PENDING_REFINEMENT_STATE)
            {
              // Have to wait to make sure that the refinement is at
              // least in flight before going on to the next step
              // since we need this refinement to do the mapping
              RtEvent wait_on = transition_event;
#ifdef DEBUG_LEGION
              assert(wait_on.exists()); // should have an event here
#endif
              eq.release();
              wait_on.wait();
              eq.reacquire();
            }
          }
          while ((eq_state != MAPPING_STATE) && 
                  (eq_state != PENDING_REFINEMENT_STATE))
          {
            if (eq_state == REFINEMENT_STATE)
            {
#ifdef DEBUG_LEGION
              assert(!transition_event.exists());
#endif
              transition_event = Runtime::create_rt_user_event(); 
              eq_state = PENDING_MAPPING_STATE; 
            }
#ifdef DEBUG_LEGION
            assert(transition_event.exists());
#endif
            RtEvent wait_on = transition_event;
            eq.release();
            wait_on.wait();
            eq.reacquire();
          }
        }
        else
        {
          // We're not the owner so see if we need to request a mapping state
          while ((eq_state != MAPPING_STATE) && 
                  (eq_state != PENDING_INVALID_STATE))
          {
            bool send_request = false;
            if (eq_state == INVALID_STATE)
            {
#ifdef DEBUG_LEGION
              assert(!transition_event.exists());
#endif
              transition_event = Runtime::create_rt_user_event(); 
              eq_state = PENDING_MAPPING_STATE;
              // Send the request for the update
              send_request = true;
            }
#ifdef DEBUG_LEGION
            assert(transition_event.exists());
#endif
            RtEvent wait_on = transition_event;
            eq.release();
            if (send_request)
            {
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
              }
              runtime->send_equivalence_set_subset_request(owner_space, rez);
            }
            wait_on.wait();
            eq.reacquire();
          }
        }
        // If we have subsets then we're going to need to recurse and
        // traverse those as well since we need to get to the leaves
        if (!subsets.empty())
          // Our set of subsets are now what we need
          to_recurse = subsets;
        else // Otherwise we can record ourselves
          version_info.record_equivalence_set(this, false/*need lock*/);
      }
      // If we have subsets then continue the traversal
      if (!to_recurse.empty())
      {
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              to_recurse.begin(); it != to_recurse.end(); it++)
          (*it)->perform_versioning_analysis(version_info);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::perform_refinements(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RefinementThunk*> to_perform;
      do 
      {
        std::vector<EquivalenceSet*> to_add;
        for (std::vector<RefinementThunk*>::const_iterator it = 
              to_perform.begin(); it != to_perform.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert((*it)->owner == this);
#endif
          EquivalenceSet *result = (*it)->perform_refinement();
          // Add our resource reference too
          result->add_nested_resource_ref(did);
          to_add.push_back(result);
        }
        AutoLock eq(eq_lock);
#ifdef DEBUG_LEGION
        assert(is_owner());
        assert((eq_state == REFINEMENT_STATE) || 
                (eq_state == PENDING_MAPPING_STATE));
#endif
        if (!to_add.empty())
          subsets.insert(subsets.end(), to_add.begin(), to_add.end());
        // Add these refinements to our subsets and delete the thunks
        if (!to_perform.empty())
        {
          for (std::vector<RefinementThunk*>::const_iterator it = 
                to_perform.begin(); it != to_perform.end(); it++)
            if ((*it)->remove_reference())
              delete (*it);
          to_perform.clear();
        }
        if (pending_refinements.empty())
        {
          // TODO: If we have too many subsets we need to make intermediates

          // This is the end of the refinement task so we need to update
          // the state and send out any notifications to anyone that the
          // refinements are done
          if (!remote_subsets.empty())
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize<size_t>(subsets.size());
              for (std::vector<EquivalenceSet*>::const_iterator it = 
                    subsets.begin(); it != subsets.end(); it++)
                rez.serialize((*it)->did);
            }
            for (std::set<AddressSpaceID>::const_iterator it = 
                  remote_subsets.begin(); it != remote_subsets.end(); it++)
              runtime->send_equivalence_set_subset_response(*it, rez);
          }
          if (eq_state == PENDING_MAPPING_STATE)
          {
#ifdef DEBUG_LEGION
            assert(transition_event.exists());
#endif
            Runtime::trigger_event(transition_event);
            transition_event = RtUserEvent::NO_RT_USER_EVENT;
          }
#ifdef DEBUG_LEGION
          assert(!transition_event.exists());
#endif
          // No matter what we end up back in the mapping state
          eq_state = MAPPING_STATE;
        }
        else // there are more refinements to do so we go around again
          to_perform.swap(pending_refinements);
      } while (!to_perform.empty());
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::send_equivalence_set(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      // We should have had a request for this already
      assert(!has_remote_instance(target));
#endif
      update_remote_instances(target);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        set_expr->pack_expression(rez, target);
      }
      runtime->send_equivalence_set_response(target, rez);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::add_pending_refinement(RefinementThunk *thunk)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      thunk->add_reference();
      pending_refinements.push_back(thunk);
      if (eq_state == MAPPING_STATE)
      {
        // Check to see if we can transition right now
        if (!mapping_guards.empty())
        {
          eq_state = PENDING_REFINEMENT_STATE;
#ifdef DEBUG_LEGION
          assert(!transition_event.exists());
#endif
          transition_event = Runtime::create_rt_user_event();
        }
        else
        {
          // Transition straight to refinement
          eq_state = REFINEMENT_STATE;
          launch_refinement_task();
        }
      }
      // Otherwise it is in a state where a refinement task is already 
      // running or will be launched eventually to handle this
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::launch_refinement_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(eq_state == REFINEMENT_STATE);
      assert(!pending_refinements.empty());
#endif
      // Send invalidations to all the remote nodes. This will invalidate
      // their mapping privileges and also send back any remote meta data
      // to the local node 
      std::set<RtEvent> refinement_preconditions;
      for (std::set<AddressSpaceID>::const_iterator it = 
            remote_subsets.begin(); it != remote_subsets.end(); it++)
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          RtUserEvent remote_done = Runtime::create_rt_user_event();
          rez.serialize(remote_done);
          refinement_preconditions.insert(remote_done);
        }
        runtime->send_equivalence_set_subset_invalidation(*it, rez);
      }
      // There are no more remote subsets
      remote_subsets.clear();
      // We now hold all the fields in exclusive mode
      exclusive_fields |= shared_fields;
      shared_fields.clear();
      exclusive_copies.clear();
      shared_copies.clear();
      RefinementTaskArgs args(this);      
      if (!refinement_preconditions.empty())
      {
        RtEvent wait_for = Runtime::merge_events(refinement_preconditions);
        runtime->issue_runtime_meta_task(args, 
            LG_THROUGHPUT_DEFERRED_PRIORITY, wait_for);
      }
      else
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_request(AddressSpaceID source,
                                                bool needs_lock/*=true*/)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        AutoLock eq(eq_lock);
        process_subset_request(source, false/*needs lock*/);
        return;
      }
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(remote_subsets.find(source) == remote_subsets.end());
#endif
      // First check to see if we need to complete this refinement
      if (unrefined_remainder != NULL)
      {
        RemoteComplete *refinement = 
          new RemoteComplete(unrefined_remainder, this, source);
        // We can clear the unrefined remainder now
        unrefined_remainder = NULL;
        add_pending_refinement(refinement);
        // We can just return since the refinement will send the
        // response after it's been done
        return;
      }
      // Add ourselves as a remote location for the subsets
      remote_subsets.insert(source);
      // We can only send the 
      if ((eq_state == MAPPING_STATE) || (eq_state == PENDING_REFINEMENT_STATE))
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<size_t>(subsets.size());
          for (std::vector<EquivalenceSet*>::const_iterator it = 
                subsets.begin(); it != subsets.end(); it++)
            rez.serialize((*it)->did);
        }
        runtime->send_equivalence_set_subset_response(source, rez);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(subsets.empty());
      assert(eq_state == PENDING_MAPPING_STATE);
      assert(transition_event.exists());
      assert(!transition_event.has_triggered());
#endif
      size_t num_subsets;
      derez.deserialize(num_subsets);
      std::set<RtEvent> wait_for;
      if (num_subsets > 0)
      {
        subsets.resize(num_subsets);
        for (unsigned idx = 0; idx < num_subsets; idx++)
        {
          DistributedID subdid;
          derez.deserialize(subdid);
          RtEvent ready;
          subsets[idx] = 
            runtime->find_or_request_equivalence_set(subdid, ready);
          if (ready.exists())
            wait_for.insert(ready);
        }
      }
      if (!wait_for.empty())
      {
        // This has to block in case there is an invalidation that comes
        // after it in the same virtual channel and we need to maintain
        // the ordering of those two operations.
        RtEvent wait_on = Runtime::merge_events(wait_for);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
          eq.release();
          wait_on.wait();
          eq.reacquire();
        }
      }
      // Add our references
      for (std::vector<EquivalenceSet*>::const_iterator it = 
            subsets.begin(); it != subsets.end(); it++)
        (*it)->add_nested_resource_ref(did);
      // Update the state
      eq_state = MAPPING_STATE;
      // Trigger the transition state to wake up any waiters
      Runtime::trigger_event(transition_event);
      transition_event = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_invalidation(RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(eq_state == MAPPING_STATE);
      assert(!transition_event.exists());
#endif
      // Check to see if we have any mapping guards in place
      if (mapping_guards.empty())
      {
        // Remove our references and clear the subsets
        if (!subsets.empty())
        {
          for (std::vector<EquivalenceSet*>::const_iterator it = 
                subsets.begin(); it != subsets.end(); it++)
            if ((*it)->remove_nested_resource_ref(did))
              delete (*it);
          subsets.clear();
        }
        // Pack up any state that needs to be sent back to the owner node
        // so that it can do any refinements
        if (!valid_instances.empty() || !reduction_instances.empty())
        {
          // TODO: send back our state to the owner node 
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(version_numbers.empty());
#endif
          // Nothing to send back so just trigger our event
          Runtime::trigger_event(to_trigger);
        }
        // Update the state to reflect that we are now invalid
        eq_state = INVALID_STATE;
      }
      else
      {
        // Update the state and save the event to trigger
        eq_state = PENDING_INVALID_STATE;
        transition_event = to_trigger;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_refinement(const void *args)
    //--------------------------------------------------------------------------
    {
      const RefinementTaskArgs *rargs = (const RefinementTaskArgs*)args;
      rargs->target->perform_refinements();
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_equivalence_set_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      EquivalenceSet *set = dynamic_cast<EquivalenceSet*>(dc);
      assert(set != NULL);
#else
      EquivalenceSet *set = static_cast<EquivalenceSet*>(dc);
#endif
      set->send_equivalence_set(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_equivalence_set_response(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      void *location;
      EquivalenceSet *set = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        set = new(location) EquivalenceSet(runtime, did, source, 
                                           expr, false/*register now*/);
      else
        set = new EquivalenceSet(runtime, did, source, 
                                 expr, false/*register now*/);
      // Once construction is complete then we do the registration
      set->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      EquivalenceSet *set = dynamic_cast<EquivalenceSet*>(dc);
      assert(set != NULL);
#else
      EquivalenceSet *set = static_cast<EquivalenceSet*>(dc);
#endif
      set->process_subset_request(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      EquivalenceSet *set = dynamic_cast<EquivalenceSet*>(dc);
      assert(set != NULL);
#else
      EquivalenceSet *set = static_cast<EquivalenceSet*>(dc);
#endif
      set->process_subset_response(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_invalidation(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      EquivalenceSet *set = dynamic_cast<EquivalenceSet*>(dc);
      assert(set != NULL);
#else
      EquivalenceSet *set = static_cast<EquivalenceSet*>(dc);
#endif
      set->process_subset_invalidation(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      VersionManager *target;
      derez.deserialize(target);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      AddressSpaceID origin;
      derez.deserialize(origin);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      EquivalenceSet *set = dynamic_cast<EquivalenceSet*>(dc);
      assert(set != NULL);
#else
      EquivalenceSet *set = static_cast<EquivalenceSet*>(dc);
#endif
      RtEvent done = set->ray_trace_equivalence_sets(target, expr, origin);
      Runtime::trigger_event(done_event, done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      VersionManager *target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      target->record_equivalence_set(set);
      Runtime::trigger_event(done_event, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_create_remote_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RefinementThunk *thunk;
      derez.deserialize(thunk);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);

      DistributedID did = runtime->get_available_distributed_id();
      EquivalenceSet *result = new EquivalenceSet(runtime, did, 
                                runtime->address_space, expr, true/*register*/);
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(thunk);
        rez.serialize(result->did);
      }
      runtime->send_equivalence_set_create_remote_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_create_remote_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RefinementThunk *thunk;
      derez.deserialize(thunk);
      DistributedID did;
      derez.deserialize(did);

      RtEvent ready;
      EquivalenceSet *result = 
        runtime->find_or_request_equivalence_set(did, ready);
      thunk->record_refinement(result, ready);
    }

    /////////////////////////////////////////////////////////////
    // Version Manager 
    /////////////////////////////////////////////////////////////

    // C++ is dumb
    const VersionID VersionManager::init_version;

    //--------------------------------------------------------------------------
    VersionManager::VersionManager(RegionTreeNode *n, ContextID c)
      : ctx(c), node(n), runtime(n->context->runtime),
        has_equivalence_sets(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionManager::VersionManager(const VersionManager &rhs)
      : ctx(rhs.ctx), node(rhs.node), runtime(rhs.runtime)
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
      if (!equivalence_sets.empty())
      {
        for (std::set<EquivalenceSet*>::const_iterator it = 
              equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        {
          if ((*it)->remove_base_resource_ref(VERSION_MANAGER_REF))
            delete (*it);
        }
        equivalence_sets.clear();
      }
#ifdef DEBUG_LEGION
      assert(!equivalence_sets_ready.exists() || 
              equivalence_sets_ready.has_triggered());
#endif
      equivalence_sets_ready = RtUserEvent::NO_RT_USER_EVENT;
      has_equivalence_sets = false;
    }

#if 0
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
#endif

    //--------------------------------------------------------------------------
    void VersionManager::perform_versioning_analysis(InnerContext *context,
                                                     VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
      // If we don't have equivalence classes for this region yet we 
      // either need to compute them or request them from the owner
      RtEvent wait_on;
      bool compute_sets = false;
      if (!has_equivalence_sets)
      {
        // Retake the lock and see if we lost the race
        AutoLock m_lock(manager_lock);
        if (!has_equivalence_sets)
        {
          if (!equivalence_sets_ready.exists()) 
          {
            equivalence_sets_ready = Runtime::create_rt_user_event();
            compute_sets = true;
          }
          wait_on = equivalence_sets_ready;
        }
      }
      if (compute_sets)
      {
        IndexSpaceExpression *expr = node->get_index_space_expression();
        RtEvent ready = context->compute_equivalence_sets(this, 
            node->get_tree_id(), expr, runtime->address_space);
        Runtime::trigger_event(equivalence_sets_ready, ready);
      }
      // Wait if necessary for the results
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      // Possibly duplicate writes, but that is alright
      if (!has_equivalence_sets)
        has_equivalence_sets = true;
      // Now that we have the equivalence classes we can have them add
      // themselves in case they have been refined and we need to traverse
      for (std::set<EquivalenceSet*>::const_iterator it = 
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        (*it)->perform_versioning_analysis(version_info); 
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_equivalence_set(EquivalenceSet *set)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      equivalence_sets.insert(set);
    }

#if 0
    //--------------------------------------------------------------------------
    void VersionManager::print_physical_state(RegionTreeNode *node,
                                              const FieldMask &capture_mask,
                                              TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      logger->log("Equivalence Sets:");
      logger->down();
      // TODO: log equivalence sets
      logger->up();
    }
#endif

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

