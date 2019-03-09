/* Copyright 2019 Stanford University, NVIDIA Corporation
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
    PhysicalUser::PhysicalUser(const RegionUsage &u, IndexSpaceExpression *e,
                               UniqueID id, unsigned x, bool cpy, bool cov)
      : usage(u), expr(e), op_id(id), index(x), copy_user(cpy), covers(cov)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      expr->add_expression_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const PhysicalUser &rhs) 
      : usage(rhs.usage), expr(rhs.expr), op_id(rhs.op_id), index(rhs.index),
        copy_user(rhs.copy_user), covers(rhs.covers)
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
    void PhysicalUser::pack_user(Serializer &rez, 
                                 const AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(usage);
      expr->pack_expression(rez, target);
      rez.serialize(op_id);
      rez.serialize(index);
      rez.serialize<bool>(copy_user);
      rez.serialize<bool>(covers);
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalUser* PhysicalUser::unpack_user(Deserializer &derez,
                          RegionTreeForest *forest, const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RegionUsage usage;
      derez.deserialize(usage);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, forest, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      bool copy_user, covers;
      derez.deserialize<bool>(copy_user);
      derez.deserialize<bool>(covers);
      return new PhysicalUser(usage, expr, op_id, index, copy_user, covers);
    }

    /////////////////////////////////////////////////////////////
    // VersionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(void)
      : owner(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(const VersionInfo &rhs)
      : owner(NULL)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs.owner == NULL);
      assert(equivalence_sets.empty());
      assert(rhs.equivalence_sets.empty());
#endif
    }

    //--------------------------------------------------------------------------
    VersionInfo::~VersionInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo& VersionInfo::operator=(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs.owner == NULL);
      assert(equivalence_sets.empty());
      assert(rhs.equivalence_sets.empty());
#endif
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::record_equivalence_set(VersionManager *own,
                                             const unsigned ver_num,
                                             EquivalenceSet *set,
                                             const FieldMask &set_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((owner == NULL) || (owner == own));
#endif
      // Save the owner in case we need to update this later
      if (owner == NULL)
      {
        owner = own;
        version_number = ver_num;
      }
      version_number = ver_num;
      equivalence_sets.insert(set, set_mask);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::update_equivalence_sets(
                                  const FieldMaskSet<EquivalenceSet> &to_add,
                                  const FieldMaskSet<EquivalenceSet> &to_remove)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION 
      assert(owner != NULL);
#endif
      // Tell the owner about the updated set
      owner->update_equivalence_sets(version_number, to_add, to_remove);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      owner = NULL;
      equivalence_sets.clear();
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
    PhysicalTraceInfo::PhysicalTraceInfo(const PhysicalTraceInfo &rhs)
      : op(rhs.op), tpl(rhs.tpl), recording(rhs.recording)
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
    void PhysicalTraceInfo::record_issue_copy(ApEvent &result,
                                 IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
#ifdef LEGION_SPY
                                 FieldSpace handle,
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition,
                                 ReductionOpID redop,
                                 bool reduction_fold) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_issue_copy(op, result, expr, src_fields, dst_fields,
#ifdef LEGION_SPY
                             handle, src_tree_id, dst_tree_id,
#endif
                             precondition, redop, reduction_fold);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_issue_fill(ApEvent &result,
                                     IndexSpaceExpression *expr,
                                     const std::vector<CopySrcDstField> &fields,
                                     const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                     UniqueID fill_uid,
                                     FieldSpace handle,
                                     RegionTreeID tree_id,
#endif
                                     ApEvent precondition) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_issue_fill(op, result, expr, fields, fill_value, fill_size,
#ifdef LEGION_SPY
                             fill_uid, handle, tree_id,
#endif
                             precondition);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_issue_indirect(ApEvent &result,
                                 IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                 const std::vector<void*> &indirections,
                                 ApEvent precondition) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(tpl != NULL);
      assert(tpl->is_recording());
#endif
      tpl->record_issue_indirect(op, result, expr, src_fields, dst_fields,
                                 indirections, precondition);
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
#if LEGION_MAX_DIM >= 2
          case 2:
            {
              logger->log("Color (%d,%d)   Mask %s", 
                          color[0], color[1], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 3
          case 3:
            {
              logger->log("Color (%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 4
          case 4:
            {
              logger->log("Color (%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], 
                          color[3], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 5
          case 5:
            {
              logger->log("Color (%d,%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2],
                          color[3], color[4], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 6
          case 6:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 7
          case 7:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], 
                          color[6], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 8
          case 8:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d,%d,%d)   Mask %s",
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], 
                          color[6], color[7], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 9
          case 9:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d,%d,%d,%d)   Mask %s",
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], 
                          color[6], color[7], color[8], mask_buffer);
              break;
            }
#endif
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
    // KDNode
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>::KDNode(IndexSpaceExpression *expr, Runtime *rt,
                        int ref_dim, int last)
      : runtime(rt), bounds(get_bounds(expr)), refinement_dim(ref_dim),
        last_changed_dim(last)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ref_dim < DIM);
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>::KDNode(const Rect<DIM> &rect, Runtime *rt, 
                        int ref_dim, int last_dim)
      : runtime(rt), bounds(rect), refinement_dim(ref_dim), 
        last_changed_dim(last_dim)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ref_dim < DIM);
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>::KDNode(const KDNode<DIM> &rhs)
      : runtime(rhs.runtime), bounds(rhs.bounds), 
        refinement_dim(rhs.refinement_dim), 
        last_changed_dim(rhs.last_changed_dim)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>::~KDNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>& KDNode<DIM>::operator=(const KDNode<DIM> &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ Rect<DIM> KDNode<DIM>::get_bounds(IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
      ApEvent wait_on;
      const Domain d = expr->get_domain(wait_on, true/*tight*/);
      if (wait_on.exists())
        wait_on.wait();
      return d.bounds<DIM,coord_t>();
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    bool KDNode<DIM>::refine(std::vector<EquivalenceSet*> &subsets,
                             const FieldMask &refinement_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(subsets.size() > LEGION_MAX_BVH_FANOUT);
#endif
      std::vector<Rect<DIM> > subset_bounds(subsets.size());
      for (unsigned idx = 0; idx < subsets.size(); idx++)
        subset_bounds[idx] = get_bounds(subsets[idx]->set_expr);
      // Compute a splitting plane 
      coord_t split = 0;
      {
        // Sort the start and end of each equivalence set bounding rectangle
        // along the splitting dimension
        std::set<KDLine> lines;
        for (unsigned idx = 0; idx < subsets.size(); idx++)
        {
          lines.insert(KDLine(subset_bounds[idx].lo[refinement_dim],idx,true));
          lines.insert(KDLine(subset_bounds[idx].hi[refinement_dim],idx,false));
        }
        // Construct two lists by scanning from left-to-right and
        // from right-to-left of the number of rectangles that would
        // be inlcuded on the left or right side by each splitting plane
        std::map<coord_t,unsigned> left_inclusive, right_inclusive;
        unsigned count = 0;
        for (typename std::set<KDLine>::const_iterator it = lines.begin();
              it != lines.end(); it++)
        {
          // Only increment for new rectangles
          if (it->start)
            count++;
          // Always record the count for all splits
          left_inclusive[it->value] = count;
        }
        count = 0;
        for (typename std::set<KDLine>::const_reverse_iterator it = 
              lines.rbegin(); it != lines.rend(); it++)
        {
          // End of rectangles are the beginning in this direction
          if (!it->start)
            count++;
          // Always record the count for all splits
          right_inclusive[it->value] = count;
        }
#ifdef DEBUG_LEGION
        assert(left_inclusive.size() == right_inclusive.size());
#endif
        // We want to take the mini-max of the two numbers in order
        // to try to balance the splitting plane across the two sets
        unsigned split_max = subsets.size();
        for (std::map<coord_t,unsigned>::const_iterator it = 
              left_inclusive.begin(); it != left_inclusive.end(); it++)
        {
          const unsigned left = it->second;
          const unsigned right = right_inclusive[it->first];
          const unsigned max = (left > right) ? left : right;
          if (max < split_max)
          {
            split_max = max;
            split = it->first;
          }
        }
      }
      // Sort the subsets into left and right
      Rect<DIM> left_bounds, right_bounds;
      left_bounds = bounds;
      right_bounds = bounds;
      left_bounds.hi[refinement_dim] = split;
      right_bounds.lo[refinement_dim] = split+1;
      std::vector<EquivalenceSet*> left_set, right_set;
      for (unsigned idx = 0; idx < subsets.size(); idx++)
      {
        const Rect<DIM> &sub_bounds = subset_bounds[idx];
        if (left_bounds.overlaps(sub_bounds))
          left_set.push_back(subsets[idx]);
        if (right_bounds.overlaps(sub_bounds))
          right_set.push_back(subsets[idx]);
      }
      // Check for the non-convex case where we can't refine anymore
      if ((refinement_dim == last_changed_dim) && 
          ((left_set.size() == subsets.size()) ||
           (right_set.size() == subsets.size())))
        return false;
      // Recurse down the tree
      const int next_dim = (refinement_dim + 1) % DIM;
      bool left_changed = false;
      if (left_set.size() > LEGION_MAX_BVH_FANOUT)
      {
        // If all the subsets span our splitting plane then we need
        // to either start tracking the last changed dimension or 
        // continue propagating the current one
        const int left_last_dim = (left_set.size() == subsets.size()) ? 
          ((last_changed_dim != -1) ? last_changed_dim : refinement_dim) : -1;
        KDNode<DIM> left(left_bounds, runtime, next_dim, left_last_dim);
        left_changed = left.refine(left_set, refinement_mask);
      }
      bool right_changed = false;
      if (right_set.size() > LEGION_MAX_BVH_FANOUT)
      {
        // If all the subsets span our splitting plane then we need
        // to either start tracking the last changed dimension or 
        // continue propagating the current one
        const int right_last_dim = (right_set.size() == subsets.size()) ? 
          ((last_changed_dim != -1) ? last_changed_dim : refinement_dim) : -1;
        KDNode<DIM> right(right_bounds, runtime, next_dim, right_last_dim);
        right_changed = right.refine(right_set, refinement_mask);
      }
      // If the sum of the left and right equivalence sets 
      // are too big then build intermediate nodes for each one
      if (((left_set.size() + right_set.size()) > LEGION_MAX_BVH_FANOUT) &&
          (left_set.size() < subsets.size()) && 
          (right_set.size() < subsets.size()))
      {
        // Make a new equivalence class and record all the subsets
        const AddressSpaceID local_space = runtime->address_space;
        std::set<IndexSpaceExpression*> left_exprs, right_exprs;
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              left_set.begin(); it != left_set.end(); it++)
          left_exprs.insert((*it)->set_expr);
        IndexSpaceExpression *left_union_expr = 
          runtime->forest->union_index_spaces(left_exprs);
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              right_set.begin(); it != right_set.end(); it++)
          right_exprs.insert((*it)->set_expr);
        IndexSpaceExpression *right_union_expr = 
          runtime->forest->union_index_spaces(right_exprs);
        EquivalenceSet *left_temp = new EquivalenceSet(runtime,
            runtime->get_available_distributed_id(), local_space,
            local_space, left_union_expr, NULL/*index space*/,
            true/*register now*/);
        EquivalenceSet *right_temp = new EquivalenceSet(runtime,
            runtime->get_available_distributed_id(), local_space,
            local_space, right_union_expr, NULL/*index space*/,
            true/*register now*/);
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              left_set.begin(); it != left_set.end(); it++)
          left_temp->record_subset(*it, refinement_mask);
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              right_set.begin(); it != right_set.end(); it++)
          right_temp->record_subset(*it, refinement_mask);
        subsets.clear();
        subsets.push_back(left_temp);
        subsets.push_back(right_temp);
        return true;
      }
      else if (left_changed || right_changed)
      {
        // If either right or left changed, then we need to recombine
        // and deduplicate the equivalence sets before we can return
        std::set<EquivalenceSet*> children;
        children.insert(left_set.begin(), left_set.end());
        children.insert(right_set.begin(), right_set.end());
        subsets.clear();
        subsets.insert(subsets.end(), children.begin(), children.end());
        return true;
      }
      else // No changes were made
        return false;
    }

    /////////////////////////////////////////////////////////////
    // Copy Fill Aggregator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(RegionTreeForest *f, 
                                           Operation *o, unsigned idx, 
                                           RtEvent g, bool t, PredEvent p)
      : WrapperReferenceMutator(effects), forest(f), 
        local_space(f->runtime->address_space), op(o), src_index(idx), 
        dst_index(idx), guard_precondition(g), 
#ifndef NON_AGGRESSIVE_AGGREGATORS
        guard_postcondition(Runtime::create_rt_user_event()),
#endif
        effects_applied(Runtime::create_rt_user_event()),
        predicate_guard(p), track_events(t)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(RegionTreeForest *f, 
                                Operation *o, unsigned src_idx, unsigned dst_idx,
                                RtEvent g, bool t, PredEvent p)
      : WrapperReferenceMutator(effects), forest(f), 
        local_space(f->runtime->address_space), op(o), src_index(src_idx), 
        dst_index(dst_idx), guard_precondition(g),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        guard_postcondition(Runtime::create_rt_user_event()),
#endif
        effects_applied(Runtime::create_rt_user_event()),
        predicate_guard(p), track_events(t)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(const CopyFillAggregator &rhs)
      : WrapperReferenceMutator(effects), forest(rhs.forest), 
        local_space(rhs.local_space), op(rhs.op),
        src_index(rhs.src_index), dst_index(rhs.dst_index), 
        guard_precondition(rhs.guard_precondition),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        guard_postcondition(rhs.guard_postcondition),
#endif
        effects_applied(rhs.effects_applied),
        predicate_guard(rhs.predicate_guard), track_events(rhs.track_events)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::~CopyFillAggregator(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
#ifndef NON_AGGRESSIVE_AGGREGATORS
      assert(guard_postcondition.has_triggered());
#endif
      assert(effects_applied.has_triggered());
      assert(guarded_sets.empty());
#endif
      // Remove references from any views that we have
      for (std::set<LogicalView*>::const_iterator it = 
            all_views.begin(); it != all_views.end(); it++)
        if ((*it)->remove_base_valid_ref(AGGREGATORE_REF))
          delete (*it);
      // Delete all our copy updates
      for (LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned::
            const_iterator mit = sources.begin(); mit != sources.end(); mit++)
      {
        for (FieldMaskSet<Update>::const_iterator it = 
              mit->second.begin(); it != mit->second.end(); it++)
          delete it->first;
      }
      for (std::vector<LegionMap<InstanceView*,
                FieldMaskSet<Update> >::aligned>::const_iterator rit = 
            reductions.begin(); rit != reductions.end(); rit++)
      {
        for (LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned::
              const_iterator mit = rit->begin(); mit != rit->end(); mit++)
        {
          for (FieldMaskSet<Update>::const_iterator it = 
                mit->second.begin(); it != mit->second.end(); it++)
            delete it->first;
        }
      }
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator& CopyFillAggregator::operator=(
                                                  const CopyFillAggregator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::CopyUpdate::record_source_expressions(
                                            InstanceFieldExprs &src_exprs) const
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<IndexSpaceExpression> &exprs = src_exprs[source];  
      FieldMaskSet<IndexSpaceExpression>::iterator finder = 
        exprs.find(expr);
      if (finder == exprs.end())
        exprs.insert(expr, src_mask);
      else
        finder.merge(src_mask);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::CopyUpdate::compute_source_preconditions(
                     RegionTreeForest *forest,
#ifdef DEBUG_LEGION
                     const bool copy_across,
#endif
                     const std::map<InstanceView*,EventFieldExprs> &src_pre,
                     LegionMap<ApEvent,FieldMask>::aligned &preconditions) const
    //--------------------------------------------------------------------------
    {
      std::map<InstanceView*,EventFieldExprs>::const_iterator finder = 
        src_pre.find(source);
      if (finder == src_pre.end())
        return;
      for (EventFieldExprs::const_iterator eit = 
            finder->second.begin(); eit != finder->second.end(); eit++)
      {
        FieldMask set_overlap = src_mask & eit->second.get_valid_mask();
        if (!set_overlap)
          continue;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
              eit->second.begin(); it != eit->second.end(); it++)
        {
          const FieldMask overlap = set_overlap & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            forest->intersect_index_spaces(expr, it->first);
          if (expr_overlap->is_empty())
            continue;
#ifdef DEBUG_LEGION
          // Since this is an equivalence set update there should be no users 
          // that are using just a part of it, should be all or nothing, with
          // the exception of copy across operations in which case it doesn't
          // matter because we don't need precise preconditions there
          if (copy_across)
            assert(expr_overlap->get_volume() == expr->get_volume());
#endif
          // Overlap in both so record it
          LegionMap<ApEvent,FieldMask>::aligned::iterator
            event_finder = preconditions.find(eit->first);
          if (event_finder == preconditions.end())
            preconditions[eit->first] = overlap;
          else
            event_finder->second |= overlap;
          set_overlap -= overlap;
          if (!set_overlap)
            break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::CopyUpdate::sort_updates(
                    std::map<InstanceView*, std::vector<CopyUpdate*> > &copies,
                    std::vector<FillUpdate*> &fills)
    //--------------------------------------------------------------------------
    {
      copies[source].push_back(this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::FillUpdate::record_source_expressions(
                                            InstanceFieldExprs &src_exprs) const
    //--------------------------------------------------------------------------
    {
      // Do nothing, we have no source expressions
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::FillUpdate::compute_source_preconditions(
                     RegionTreeForest *forest,
#ifdef DEBUG_LEGION
                     const bool copy_across,
#endif
                     const std::map<InstanceView*,EventFieldExprs> &src_pre,
                     LegionMap<ApEvent,FieldMask>::aligned &preconditions) const
    //--------------------------------------------------------------------------
    {
      // Do nothing, we have no source preconditions to worry about
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::FillUpdate::sort_updates(
                    std::map<InstanceView*, std::vector<CopyUpdate*> > &copies,
                    std::vector<FillUpdate*> &fills)
    //--------------------------------------------------------------------------
    {
      fills.push_back(this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_updates(InstanceView *dst_view, 
                                    const FieldMaskSet<LogicalView> &src_views,
                                    const FieldMask &src_mask,
                                    IndexSpaceExpression *expr,
                                    ReductionOpID redop /*=0*/,
                                    CopyAcrossHelper *helper /*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!src_mask);
      assert(!src_views.empty());
      assert(!expr->is_empty());
#endif
      update_fields |= src_mask;
      FieldMaskSet<Update> &updates = sources[dst_view];
      record_view(dst_view);
      if (src_views.size() == 1)
      {
        const LogicalView *view = src_views.begin()->first;
        const FieldMask record_mask = 
          src_views.get_valid_mask() & src_mask;
        if (!!record_mask)
        {
          if (view->is_instance_view())
          {
            InstanceView *inst = view->as_instance_view();
            record_view(inst);
            CopyUpdate *update = 
              new CopyUpdate(inst, record_mask, expr, redop, helper);
            if (helper == NULL)
              updates.insert(update, record_mask);
            else
              updates.insert(update, helper->convert_src_to_dst(record_mask));
          }
          else
          {
            DeferredView *def = view->as_deferred_view();
            def->flatten(*this, dst_view, record_mask, expr, helper);
          }
        }
      }
      else
      {
        // We have multiple views, so let's sort them
        LegionList<FieldSet<LogicalView*> >::aligned view_sets;
        src_views.compute_field_sets(src_mask, view_sets);
        for (LegionList<FieldSet<LogicalView*> >::aligned::const_iterator
              vit = view_sets.begin(); vit != view_sets.end(); vit++)
        {
          if (vit->elements.empty())
            continue;
          if (vit->elements.size() == 1)
          {
            // Easy case, just one view so do it  
            const LogicalView *view = *(vit->elements.begin());
            const FieldMask &record_mask = vit->set_mask;
            if (view->is_instance_view())
            {
              InstanceView *inst = view->as_instance_view();
              record_view(inst);
              CopyUpdate *update = 
                new CopyUpdate(inst, record_mask, expr, redop, helper);
              if (helper == NULL)
                updates.insert(update, record_mask);
              else
                updates.insert(update, helper->convert_src_to_dst(record_mask));
            }
            else
            {
              DeferredView *def = view->as_deferred_view();
              def->flatten(*this, dst_view, record_mask, expr, helper);
            }
          }
          else
          {
            // Sort the views, prefer fills, then instances, then deferred
            FillView *fill = NULL;
            DeferredView *deferred = NULL;
            std::vector<InstanceView*> instances;
            for (std::set<LogicalView*>::const_iterator it = 
                  vit->elements.begin(); it != vit->elements.end(); it++)
            {
              if (!(*it)->is_instance_view())
              {
                DeferredView *def = (*it)->as_deferred_view();
                if (!def->is_fill_view())
                {
                  if (deferred == NULL)
                    deferred = def;
                }
                else
                {
                  fill = def->as_fill_view();
                  // Break out since we found what we're looking for
                  break;
                }
              }
              else
                instances.push_back((*it)->as_instance_view());
            }
            if (fill != NULL)
              record_fill(dst_view, fill, vit->set_mask, expr, helper);
            else if (!instances.empty())
            {
              if (instances.size() == 1)
              {
                // Easy, just one instance to use
                InstanceView *inst = instances.back();
                record_view(inst);
                CopyUpdate *update = 
                  new CopyUpdate(inst, vit->set_mask, expr, redop, helper);
                if (helper == NULL)
                  updates.insert(update, vit->set_mask);
                else
                  updates.insert(update, 
                      helper->convert_src_to_dst(vit->set_mask));
              }
              else
              {
                // Hard, multiple potential sources,
                // ask the mapper which one to use
                // First though check to see if we've already asked it
                bool found = false;
                const std::set<InstanceView*> instances_set(instances.begin(),
                                                            instances.end());
                std::map<InstanceView*,LegionVector<SourceQuery>::aligned>::
                  const_iterator finder = mapper_queries.find(dst_view);
                if (finder != mapper_queries.end())
                {
                  for (LegionVector<SourceQuery>::aligned::const_iterator qit = 
                        finder->second.begin(); qit != 
                        finder->second.end(); qit++)
                  {
                    if ((qit->query_mask == vit->set_mask) &&
                        (qit->sources == instances_set))
                    {
                      found = true;
                      record_view(qit->result);
                      CopyUpdate *update = new CopyUpdate(qit->result, 
                                    qit->query_mask, expr, redop, helper);
                      if (helper == NULL)
                        updates.insert(update, qit->query_mask);
                      else
                        updates.insert(update, 
                            helper->convert_src_to_dst(qit->query_mask));
                      break;
                    }
                  }
                }
                if (!found)
                {
                  // If we didn't find the query result we need to do
                  // it for ourself, start by constructing the inputs
                  InstanceRef dst(dst_view->get_manager(),
                      helper == NULL ? vit->set_mask : 
                        helper->convert_src_to_dst(vit->set_mask));
                  InstanceSet sources(instances.size());
                  unsigned src_idx = 0;
                  for (std::vector<InstanceView*>::const_iterator it = 
                        instances.begin(); it != instances.end(); it++)
                    sources[src_idx++] = InstanceRef((*it)->get_manager(),
                                                     vit->set_mask);
                  std::vector<unsigned> ranking;
                  op->select_sources(dst, sources, ranking);
                  // We know that which ever one was chosen first is
                  // the one that satisfies all our fields since all
                  // these instances are valid for all fields
                  InstanceView *result = ranking.empty() ? 
                    instances.front() : instances[ranking[0]];
                  // Record the update
                  record_view(result);
                  CopyUpdate *update = new CopyUpdate(result, vit->set_mask,
                                                      expr, redop, helper);
                  if (helper == NULL)
                    updates.insert(update, vit->set_mask);
                  else
                    updates.insert(update, 
                        helper->convert_src_to_dst(vit->set_mask));
                  // Save the result for the future
                  mapper_queries[dst_view].push_back(
                      SourceQuery(instances_set, vit->set_mask, result));
                }
              }
            }
            else
            {
#ifdef DEBUG_LEGION
              assert(deferred != NULL);
#endif
              deferred->flatten(*this, dst_view, vit->set_mask, expr, helper);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_fill(InstanceView *dst_view,
                                         FillView *src_view,
                                         const FieldMask &fill_mask,
                                         IndexSpaceExpression *expr,
                                         CopyAcrossHelper *helper /*=NULL*/)
    //--------------------------------------------------------------------------
    {
      // No need to record the destination as we already did that the first
      // time through on our way to finding this fill view
#ifdef DEBUG_LEGION
      assert(all_views.find(dst_view) != all_views.end());
      assert(!!fill_mask);
      assert(!expr->is_empty());
#endif
      update_fields |= fill_mask;
      record_view(src_view);
      FillUpdate *update = new FillUpdate(src_view, fill_mask, expr, helper); 
      if (helper == NULL)
        sources[dst_view].insert(update, fill_mask);
      else
        sources[dst_view].insert(update, helper->convert_src_to_dst(fill_mask));
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_reductions(InstanceView *dst_view,
                                   const std::vector<ReductionView*> &src_views,
                                   const unsigned src_fidx,
                                   const unsigned dst_fidx,
                                   IndexSpaceExpression *expr,
                                   CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!src_views.empty());
      assert(!expr->is_empty());
#endif 
      update_fields.set_bit(src_fidx);
      record_view(dst_view);
      for (std::vector<ReductionView*>::const_iterator it = 
            src_views.begin(); it != src_views.end(); it++)
        record_view(*it);
      const std::pair<InstanceView*,unsigned> dst_key(dst_view, dst_fidx);
      std::vector<ReductionOpID> &redop_epochs = reduction_epochs[dst_key];
      FieldMask src_mask, dst_mask;
      src_mask.set_bit(src_fidx);
      dst_mask.set_bit(dst_fidx);
      // Always start scanning from the first redop index
      unsigned redop_index = 0;
      for (std::vector<ReductionView*>::const_iterator it = 
            src_views.begin(); it != src_views.end(); it++)
      {
        const ReductionOpID redop = (*it)->get_redop();
        CopyUpdate *update = 
          new CopyUpdate(*it, src_mask, expr, redop, across_helper);
        // Scan along looking for a reduction op epoch that matches
        while ((redop_index < redop_epochs.size()) &&
                (redop_epochs[redop_index] != redop))
          redop_index++;
        if (redop_index == redop_epochs.size())
        {
#ifdef DEBUG_LEGION
          assert(redop_index <= reductions.size());
#endif
          // Start a new redop epoch if necessary
          redop_epochs.push_back(redop);
          if (reductions.size() == redop_index)
            reductions.resize(redop_index + 1);
        }
        reductions[redop_index][dst_view].insert(update, dst_mask);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_preconditions(InstanceView *view, 
                                   bool reading, EventFieldExprs &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!preconditions.empty());
#endif
      AutoLock p_lock(pre_lock);
      EventFieldExprs &pre = reading ? src_pre[view] : dst_pre[view]; 
      for (EventFieldExprs::iterator eit = preconditions.begin();
            eit != preconditions.end(); eit++)
      {
        EventFieldExprs::iterator event_finder = pre.find(eit->first);
        if (event_finder != pre.end())
        {
          // Need to do the merge manually 
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            FieldMaskSet<IndexSpaceExpression>::iterator finder = 
              event_finder->second.find(it->first);
            if (finder == event_finder->second.end())
              event_finder->second.insert(it->first, it->second);
            else
              finder.merge(it->second);
          }
        }
        else // We can just swap this over
          pre[eit->first].swap(eit->second);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_precondition(InstanceView *view,
                                                 bool reading, ApEvent event,
                                                 const FieldMask &mask,
                                                 IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(pre_lock);
      FieldMaskSet<IndexSpaceExpression> &event_pre = 
        reading ? src_pre[view][event] : dst_pre[view][event];
      FieldMaskSet<IndexSpaceExpression>::iterator finder = 
        event_pre.find(expr);
      if (finder == event_pre.end())
        event_pre.insert(expr, mask);
      else
        finder.merge(mask);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_updates(
                                           const PhysicalTraceInfo &trace_info,
                                           ApEvent precondition,
                                           const bool has_src_preconditions,
                                           const bool has_dst_preconditions,
                                           const bool need_deferral)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!sources.empty() || !reductions.empty());
#endif
      if (need_deferral || 
          (guard_precondition.exists() && !guard_precondition.has_triggered()))
      {
        CopyFillAggregation args(this, trace_info, precondition, 
          has_src_preconditions, has_dst_preconditions, op->get_unique_op_id());
        op->runtime->issue_runtime_meta_task(args, 
                           LG_THROUGHPUT_DEFERRED_PRIORITY, guard_precondition);
        return;
      }
#ifdef DEBUG_LEGION
      assert(!guard_precondition.exists() || 
              guard_precondition.has_triggered());
#endif
      // Perform updates from any sources first
      if (!sources.empty())
        perform_updates(sources, trace_info, precondition,
                        has_src_preconditions, has_dst_preconditions);
      // Then apply any reductions that we might have
      if (!reductions.empty())
      {
        for (std::vector<LegionMap<InstanceView*,
                   FieldMaskSet<Update> >::aligned>::const_iterator it =
              reductions.begin(); it != reductions.end(); it++)
          perform_updates(*it, trace_info, precondition,
                          has_src_preconditions, has_dst_preconditions);
      }
#ifndef NON_AGGRESSIVE_AGGREGATORS
      Runtime::trigger_event(guard_postcondition);
#endif
      // We can also trigger our guard event once the effects are applied
      if (!effects.empty())
        Runtime::trigger_event(effects_applied,
            Runtime::merge_events(effects));
      else
        Runtime::trigger_event(effects_applied);
    }
    
    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_guard_set(EquivalenceSet *set)
    //--------------------------------------------------------------------------
    {
      guarded_sets.insert(set);
    }

    //--------------------------------------------------------------------------
    bool CopyFillAggregator::release_guards(std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      if (!effects_applied.has_triggered())
      {
        // Meta-task will take responsibility for deletion
        CopyFillDeletion args(this, op->get_unique_op_id(), 
                              !guarded_sets.empty());
        const RtEvent done = op->runtime->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY, effects_applied);
        applied.insert(done);
        return false;
      }
      else if (!guarded_sets.empty())
        release_guarded_sets();
      return true;
    }

    //--------------------------------------------------------------------------
    ApEvent CopyFillAggregator::summarize(const PhysicalTraceInfo &info) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(track_events);
#endif
      if (!events.empty())
        return Runtime::merge_events(&info, events);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_view(LogicalView *new_view)
    //--------------------------------------------------------------------------
    {
      std::pair<std::set<LogicalView*>::iterator,bool> result = 
        all_views.insert(new_view);
      if (result.second)
        new_view->add_base_valid_ref(AGGREGATORE_REF, this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::perform_updates(
         const LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned &updates,
         const PhysicalTraceInfo &trace_info, ApEvent precondition,
         const bool has_src_preconditions, const bool has_dst_preconditions)
    //--------------------------------------------------------------------------
    {
      if (!has_src_preconditions || !has_dst_preconditions)
      {
        // First compute the access expressions for all the copies
        InstanceFieldExprs dst_exprs, src_exprs;
        for (LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned::
              const_iterator uit = updates.begin(); uit != updates.end(); uit++)
        {
          FieldMaskSet<IndexSpaceExpression> &dst_expr = dst_exprs[uit->first];
          for (FieldMaskSet<Update>::const_iterator it = 
                uit->second.begin(); it != uit->second.end(); it++)
          {
            // Update the destinations first
            if (!has_dst_preconditions)
            {
#ifdef DEBUG_LEGION
              // We should not have an across helper in this case
              assert(it->first->across_helper == NULL);
#endif
              FieldMaskSet<IndexSpaceExpression>::iterator finder = 
                dst_expr.find(it->first->expr);
              if (finder == dst_expr.end())
                dst_expr.insert(it->first->expr, it->second);
              else
                finder.merge(it->second);
            }
            // Now record the source expressions
            if (!has_src_preconditions)
              it->first->record_source_expressions(src_exprs);
          }
        }
        // Next compute the event preconditions for these accesses
        std::set<RtEvent> preconditions_ready; 
        const UniqueID op_id = op->get_unique_op_id();
        if (!has_dst_preconditions)
        {
          dst_pre.clear();
          for (InstanceFieldExprs::const_iterator dit = 
                dst_exprs.begin(); dit != dst_exprs.end(); dit++)
          {
            if (dit->second.size() == 1)
            {
              // No need to do any kind of sorts here
              IndexSpaceExpression *copy_expr = dit->second.begin()->first;
              const FieldMask &copy_mask = dit->second.get_valid_mask();
              RtEvent pre_ready = dit->first->find_copy_preconditions(
                              false/*reading*/, copy_mask, copy_expr, op_id, 
                              dst_index, *this, trace_info, local_space);
              if (pre_ready.exists())
                preconditions_ready.insert(pre_ready);
            }
            else
            {
              // Sort into field sets and merge expressions
              LegionList<FieldSet<IndexSpaceExpression*> >::aligned 
                sorted_exprs;
              dit->second.compute_field_sets(FieldMask(), sorted_exprs);
              for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
                    const_iterator it = sorted_exprs.begin(); 
                    it != sorted_exprs.end(); it++)
              {
                const FieldMask &copy_mask = it->set_mask; 
                IndexSpaceExpression *copy_expr = (it->elements.size() == 1) ?
                  *(it->elements.begin()) : 
                  forest->union_index_spaces(it->elements);
                RtEvent pre_ready = dit->first->find_copy_preconditions(
                                false/*reading*/, copy_mask, copy_expr, op_id,
                                dst_index, *this, trace_info, local_space);
                if (pre_ready.exists())
                  preconditions_ready.insert(pre_ready);
              }
            }
          }
        }
        if (!has_src_preconditions)
        {
          src_pre.clear();
          for (InstanceFieldExprs::const_iterator sit = 
                src_exprs.begin(); sit != src_exprs.end(); sit++)
          {
            if (sit->second.size() == 1)
            {
              // No need to do any kind of sorts here
              IndexSpaceExpression *copy_expr = sit->second.begin()->first;
              const FieldMask &copy_mask = sit->second.get_valid_mask();
              RtEvent pre_ready = sit->first->find_copy_preconditions(
                            true/*reading*/, copy_mask, copy_expr, op_id, 
                            src_index, *this, trace_info, local_space);
              if (pre_ready.exists())
                preconditions_ready.insert(pre_ready);
            }
            else
            {
              // Sort into field sets and merge expressions
              LegionList<FieldSet<IndexSpaceExpression*> >::aligned 
                sorted_exprs;
              sit->second.compute_field_sets(FieldMask(), sorted_exprs);
              for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
                    const_iterator it = sorted_exprs.begin(); 
                    it != sorted_exprs.end(); it++)
              {
                const FieldMask &copy_mask = it->set_mask; 
                IndexSpaceExpression *copy_expr = (it->elements.size() == 1) ?
                  *(it->elements.begin()) : 
                  forest->union_index_spaces(it->elements);
                RtEvent pre_ready = sit->first->find_copy_preconditions(
                              true/*reading*/, copy_mask, copy_expr, op_id,
                              src_index, *this, trace_info, local_space);
                if (pre_ready.exists())
                  preconditions_ready.insert(pre_ready);
              }
            }
          }
        }
        // If necessary wait until all we have all the preconditions
        if (!preconditions_ready.empty())
        {
          RtEvent wait_on = Runtime::merge_events(preconditions_ready);
          if (wait_on.exists())
            wait_on.wait();
        }
      }
      // Iterate over the destinations and compute updates that have the
      // same preconditions on different fields
      std::map<std::set<ApEvent>,ApEvent> merge_cache;
      for (LegionMap<InstanceView*,FieldMaskSet<Update> >::aligned::
            const_iterator uit = updates.begin(); uit != updates.end(); uit++)
      {
        EventFieldUpdates update_groups;
        const EventFieldExprs &dst_preconditions = dst_pre[uit->first];
        for (FieldMaskSet<Update>::const_iterator it = 
              uit->second.begin(); it != uit->second.end(); it++)
        {
          // Compute the preconditions for this update
          // This is a little tricky for across copies because we need
          // to make sure that all the fields are in same field space
          // which will be the source field space, so we need to convert
          // some field masks back to that space if necessary
          LegionMap<ApEvent,FieldMask>::aligned preconditions;
          // Compute the destination preconditions first
          if (!dst_preconditions.empty())
          {
            for (EventFieldExprs::const_iterator pit = 
                  dst_preconditions.begin(); pit != 
                  dst_preconditions.end(); pit++)
            {
              FieldMask set_overlap = it->second & pit->second.get_valid_mask();
              if (!set_overlap)
                continue;
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator eit =
                    pit->second.begin(); eit != pit->second.end(); eit++)
              {
                const FieldMask overlap = set_overlap & eit->second;
                if (!overlap)
                  continue;
                IndexSpaceExpression *expr_overlap = 
                  forest->intersect_index_spaces(eit->first, it->first->expr);
                if (expr_overlap->is_empty())
                  continue;
#ifdef DEBUG_LEGION
                // Since this is an equivalence set update there should 
                // be no users that are using just a part of it, should 
                // be all or nothing, unless this is a copy across in 
                // which case it doesn't matter
                if (src_index != dst_index)
                  assert(expr_overlap->get_volume() == 
                          it->first->expr->get_volume());
#endif
                // Overlap on both so add it to the set
                LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
                  preconditions.find(pit->first);
                // Make sure to convert back to the source field space
                // in the case of across copies if necessary
                if (finder == preconditions.end())
                {
                  if (it->first->across_helper == NULL)
                    preconditions[pit->first] = overlap;
                  else
                    preconditions[pit->first] = 
                      it->first->across_helper->convert_dst_to_src(overlap);
                }
                else
                {
                  if (it->first->across_helper == NULL)
                    finder->second |= overlap;
                  else
                    finder->second |= 
                      it->first->across_helper->convert_dst_to_src(overlap);
                }
                set_overlap -= overlap;
                // If we found preconditions on all our fields then we're done
                if (!set_overlap)
                  break;
              }
            }
          }
          // The compute the source preconditions for this update
          it->first->compute_source_preconditions(forest,
#ifdef DEBUG_LEGION
                                                  (src_index != dst_index),
#endif
                                                  src_pre, preconditions);
          if (preconditions.empty())
            // NO precondition so enter it with a no event
            update_groups[ApEvent::NO_AP_EVENT].insert(it->first, 
                                                       it->first->src_mask);
          else if (preconditions.size() == 1)
          {
            LegionMap<ApEvent,FieldMask>::aligned::const_iterator
              first = preconditions.begin();
            update_groups[first->first].insert(it->first, first->second);
            const FieldMask remainder = it->first->src_mask - first->second;
            if (!!remainder)
              update_groups[ApEvent::NO_AP_EVENT].insert(it->first, remainder);
          }
          else
          {
            // Group event preconditions by fields
            LegionList<FieldSet<ApEvent> >::aligned grouped_events;
            compute_field_sets<ApEvent>(it->first->src_mask,
                                        preconditions, grouped_events);
            for (LegionList<FieldSet<ApEvent> >::aligned::const_iterator ait =
                  grouped_events.begin(); ait != grouped_events.end(); ait++) 
            {
              ApEvent key;
              if (ait->elements.size() > 1)
              {
                // See if the set is in the cache or we need to compute it 
                std::map<std::set<ApEvent>,ApEvent>::const_iterator finder =
                  merge_cache.find(ait->elements);
                if (finder == merge_cache.end())
                {
                  key = Runtime::merge_events(&trace_info, ait->elements);
                  merge_cache[ait->elements] = key;
                }
                else
                  key = finder->second;
              }
              else if (ait->elements.size() == 1)
                key = *(ait->elements.begin());
              FieldMaskSet<Update> &group = update_groups[key]; 
              FieldMaskSet<Update>::iterator finder = group.find(it->first);
              if (finder != group.end())
                finder.merge(ait->set_mask);
              else
                group.insert(it->first, ait->set_mask);
            }
          }
        }
        // Now iterate over events and group by fields
        for (EventFieldUpdates::const_iterator eit = 
              update_groups.begin(); eit != update_groups.end(); eit++)
        {
          // Merge in the over-arching precondition if necessary
          const ApEvent group_precondition = precondition.exists() ? 
            Runtime::merge_events(&trace_info, precondition, eit->first) :
            eit->first;
          const FieldMaskSet<Update> &group = eit->second;
#ifdef DEBUG_LEGION
          assert(!group.empty());
#endif
          if (group.size() == 1)
          {
            // Only one update so no need to try to group or merge 
            std::vector<FillUpdate*> fills;
            std::map<InstanceView* /*src*/,std::vector<CopyUpdate*> > copies;
            Update *update = group.begin()->first;
            update->sort_updates(copies, fills);
            const FieldMask &update_mask = group.get_valid_mask();
            if (!fills.empty())
              issue_fills(uit->first, fills, group_precondition, 
                          update_mask, trace_info, has_dst_preconditions);
            if (!copies.empty())
              issue_copies(uit->first, copies, group_precondition, 
                           update_mask, trace_info, has_dst_preconditions);
          }
          else
          {
            // Group by fields
            LegionList<FieldSet<Update*> >::aligned field_groups;
            group.compute_field_sets(FieldMask(), field_groups);
            for (LegionList<FieldSet<Update*> >::aligned::const_iterator fit =
                  field_groups.begin(); fit != field_groups.end(); fit++)
            {
              std::vector<FillUpdate*> fills;
              std::map<InstanceView* /*src*/,
                       std::vector<CopyUpdate*> > copies;
              for (std::set<Update*>::const_iterator it = 
                    fit->elements.begin(); it != fit->elements.end(); it++)
                (*it)->sort_updates(copies, fills);
              if (!fills.empty())
                issue_fills(uit->first, fills, group_precondition,
                            fit->set_mask, trace_info, has_dst_preconditions);
              if (!copies.empty())
                issue_copies(uit->first, copies, group_precondition, 
                             fit->set_mask, trace_info, has_dst_preconditions);
            }
          }
        }
      } // iterate over dst instances
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_fills(InstanceView *target,
                                         const std::vector<FillUpdate*> &fills,
                                         ApEvent precondition, 
                                         const FieldMask &fill_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         const bool has_dst_preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fills.empty());
      assert(!!fill_mask); 
#endif
      std::vector<CopySrcDstField> dst_fields;
      target->copy_to(fill_mask, dst_fields, fills[0]->across_helper);
      const UniqueID op_id = op->get_unique_op_id();
#ifdef LEGION_SPY
      PhysicalManager *manager = target->get_manager();
      FieldSpaceNode *field_space_node = manager->field_space_node;
#endif
      if (fills.size() == 1)
      {
        FillUpdate *update = fills[0];
#ifdef DEBUG_LEGION
        // Should cover all the fields
        assert(!(fill_mask - update->src_mask));
#endif
        IndexSpaceExpression *fill_expr = update->expr;
        FillView *fill_view = update->source;
        const ApEvent result = fill_expr->issue_fill(trace_info, dst_fields,
                                                     fill_view->value->value,
                                                   fill_view->value->value_size,
#ifdef LEGION_SPY
                                                     fill_view->fill_op_uid,
                                                     field_space_node->handle,
                                                     manager->tree_id,
#endif
                                                     precondition,
                                                     predicate_guard);
        // Record the fill result in the destination 
        if (result.exists())
        {
          if (update->across_helper != NULL)
          {
            const FieldMask dst_mask = 
                update->across_helper->convert_src_to_dst(fill_mask);
            target->add_copy_user(false/*reading*/,
                                  result, dst_mask, fill_expr, op_id, 
                                  dst_index, effects, trace_info, local_space);
            // Record this for the next iteration if necessary
            if (has_dst_preconditions)
              record_precondition(target, false/*reading*/, result, 
                                  dst_mask, fill_expr);
          }
          else
          {
            target->add_copy_user(false/*reading*/,
                                  result, fill_mask, fill_expr, op_id, 
                                  dst_index, effects, trace_info, local_space);
            // Record this for the next iteration if necessary
            if (has_dst_preconditions)
              record_precondition(target, false/*reading*/, result,
                                  fill_mask, fill_expr);
          }
          if (track_events)
            events.insert(result);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        // These should all have had the same across helper
        for (unsigned idx = 1; idx < fills.size(); idx++)
          assert(fills[idx]->across_helper == fills[0]->across_helper);
#endif
#endif
        std::map<FillView*,std::set<IndexSpaceExpression*> > exprs;
        for (std::vector<FillUpdate*>::const_iterator it = 
              fills.begin(); it != fills.end(); it++)
        {
#ifdef DEBUG_LEGION
          // Should cover all the fields
          assert(!(fill_mask - (*it)->src_mask));
          // Should also have the same across helper as the first one
          assert(fills[0]->across_helper == (*it)->across_helper);
#endif
          exprs[(*it)->source].insert((*it)->expr);
        }
        const FieldMask dst_mask = 
          (fills[0]->across_helper == NULL) ? fill_mask : 
           fills[0]->across_helper->convert_src_to_dst(fill_mask);
        for (std::map<FillView*,std::set<IndexSpaceExpression*> >::
              const_iterator it = exprs.begin(); it != exprs.end(); it++)
        {
          IndexSpaceExpression *fill_expr = (it->second.size() == 1) ?
            *(it->second.begin()) : forest->union_index_spaces(it->second);
          const ApEvent result = fill_expr->issue_fill(trace_info, dst_fields,
                                                       it->first->value->value,
                                                  it->first->value->value_size,
#ifdef LEGION_SPY
                                                       it->first->fill_op_uid,
                                                       field_space_node->handle,
                                                       manager->tree_id,
#endif
                                                       precondition,
                                                       predicate_guard);
          if (result.exists())
          {
            target->add_copy_user(false/*reading*/,
                                  result, dst_mask, fill_expr, op_id, 
                                  dst_index, effects, trace_info, local_space);
            if (track_events)
              events.insert(result);
            // Record this for the next iteration if necessary
            if (has_dst_preconditions)
              record_precondition(target, false/*reading*/, result,
                                  dst_mask, fill_expr);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_copies(InstanceView *target, 
                              const std::map<InstanceView*,
                                             std::vector<CopyUpdate*> > &copies,
                              ApEvent precondition, const FieldMask &copy_mask,
                              const PhysicalTraceInfo &trace_info,
                              const bool has_dst_preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!copies.empty());
      assert(!!copy_mask);
#endif
      const UniqueID op_id = op->get_unique_op_id();
#ifdef LEGION_SPY
      PhysicalManager *manager = target->get_manager();
      FieldSpaceNode *field_space_node = manager->field_space_node;
#endif
      for (std::map<InstanceView*,std::vector<CopyUpdate*> >::const_iterator
            cit = copies.begin(); cit != copies.end(); cit++)
      {
#ifdef DEBUG_LEGION
        assert(!cit->second.empty());
#endif
        std::vector<CopySrcDstField> dst_fields, src_fields;
        target->copy_to(copy_mask, dst_fields, cit->second[0]->across_helper);
        if (cit->second.size() == 1)
        {
          // Easy case of a single update copy
          CopyUpdate *update = cit->second[0];
#ifdef DEBUG_LEGION
          // Should cover all the fields
          assert(!(copy_mask - update->src_mask));
#endif
          InstanceView *source = update->source;
          source->copy_from(copy_mask, src_fields);
          IndexSpaceExpression *copy_expr = update->expr;
          const ApEvent result = copy_expr->issue_copy(trace_info, 
                                    dst_fields, src_fields, 
#ifdef LEGION_SPY
                                    field_space_node->handle,
                                    source->get_manager()->tree_id,
                                    manager->tree_id,
#endif
                                    precondition, predicate_guard,
                                    update->redop, false/*fold*/);
          if (result.exists())
          {
            source->add_copy_user(true/*reading*/,
                                  result, copy_mask, copy_expr, op_id, 
                                  src_index, effects, trace_info, local_space);
            if (update->across_helper != NULL)
            {
              const FieldMask dst_mask = 
                update->across_helper->convert_src_to_dst(copy_mask);
              target->add_copy_user(false/*reading*/,
                                    result, dst_mask, copy_expr, op_id, 
                                    dst_index, effects, trace_info,local_space);
              // Record this for the next iteration if necessary
              if (has_dst_preconditions)
                record_precondition(target, false/*reading*/, result,
                                    dst_mask, copy_expr);
            }
            else
            {
              target->add_copy_user(false/*reading*/,
                                    result, copy_mask, copy_expr, op_id, 
                                    dst_index, effects, trace_info,local_space);
              // Record this for the next iteration if necessary
              if (has_dst_preconditions)
                record_precondition(target, false/*reading*/, result,
                                    copy_mask, copy_expr);
            }
            if (track_events)
              events.insert(result);
          }
        }
        else
        {
          // Have to group by source instances in order to merge together
          // different index space expressions for the same copy
          std::map<InstanceView*,std::set<IndexSpaceExpression*> > src_exprs;
          const ReductionOpID redop = cit->second[0]->redop;
          for (std::vector<CopyUpdate*>::const_iterator it = 
                cit->second.begin(); it != cit->second.end(); it++)
          {
#ifdef DEBUG_LEGION
            // Should cover all the fields
            assert(!(copy_mask - (*it)->src_mask));
            // Should have the same redop
            assert(redop == (*it)->redop);
            // Should also have the same across helper as the first one
            assert(cit->second[0]->across_helper == (*it)->across_helper);
#endif
            src_exprs[(*it)->source].insert((*it)->expr);
          }
          const FieldMask dst_mask = 
            (cit->second[0]->across_helper == NULL) ? copy_mask : 
             cit->second[0]->across_helper->convert_src_to_dst(copy_mask);
          for (std::map<InstanceView*,std::set<IndexSpaceExpression*> >::
                const_iterator it = src_exprs.begin(); 
                it != src_exprs.end(); it++)
          {
            IndexSpaceExpression *copy_expr = (it->second.size() == 1) ? 
              *(it->second.begin()) : forest->union_index_spaces(it->second);
            src_fields.clear();
            it->first->copy_from(copy_mask, src_fields);
            const ApEvent result = copy_expr->issue_copy(trace_info, 
                                    dst_fields, src_fields, 
#ifdef LEGION_SPY
                                    field_space_node->handle,
                                    it->first->get_manager()->tree_id,
                                    manager->tree_id,
#endif
                                    precondition, predicate_guard,
                                    redop, false/*fold*/);
            if (result.exists())
            {
              it->first->add_copy_user(true/*reading*/,
                                    result, copy_mask, copy_expr, op_id, 
                                    src_index, effects, trace_info,local_space);
              target->add_copy_user(false/*reading*/,
                                    result, dst_mask, copy_expr, op_id, 
                                    dst_index, effects, trace_info,local_space);
              if (track_events)
                events.insert(result);
              // Record this for the next iteration if necessary
              if (has_dst_preconditions)
                record_precondition(target, false/*reading*/, result,
                                    dst_mask, copy_expr);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::release_guarded_sets(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!guarded_sets.empty());
#endif
      for (std::set<EquivalenceSet*>::const_iterator it = 
            guarded_sets.begin(); it != guarded_sets.end(); it++)
        (*it)->remove_update_guard(this);
      guarded_sets.clear();
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyFillAggregator::handle_aggregation(const void *args)
    //--------------------------------------------------------------------------
    {
      const CopyFillAggregation *cfargs = (const CopyFillAggregation*)args;
      cfargs->aggregator->issue_updates(cfargs->info, cfargs->pre,
                cfargs->has_src, cfargs->has_dst, false/*needs deferral*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyFillAggregator::handle_deletion(const void *args)
    //--------------------------------------------------------------------------
    {
      const CopyFillDeletion *dargs = (const CopyFillDeletion*)args;
      if (dargs->remove_guards)
        dargs->aggregator->release_guarded_sets();
      delete dargs->aggregator;
    }

    /////////////////////////////////////////////////////////////
    // Remote Eq Tracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteEqTracker::RemoteEqTracker(Runtime *rt)
      : previous(rt->address_space), original_source(previous), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool RemoteEqTracker::request_remote_instances(
                                              FieldMaskSet<LogicalView> &insts,
                                              std::set<RtEvent> &ready_events,
                                              RemoteEqTracker *target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      if (this == target)
      {
        remote_lock = new LocalLock();
        sync_events = new std::set<RtEvent>();
        remote_insts = new FieldMaskSet<LogicalView>();
        restricted = false;
      }
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(target);
          rez.serialize(ready);
        }
        runtime->send_equivalence_set_remote_request_instances(rit->first, rez);
        if (this == target)
        {
          AutoLock r_lock(*remote_lock);
          sync_events->insert(ready);
        }
        else
          ready_events.insert(ready);
      }
      if (this == target)
      {
        sync_remote_instances(insts);
        return restricted;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool RemoteEqTracker::request_remote_reductions(
                                            FieldMaskSet<ReductionView> &insts,
                                            const ReductionOpID redop,
                                            std::set<RtEvent> &ready_events,
                                            RemoteEqTracker *target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      if (this == target)
      {
        remote_lock = new LocalLock();
        sync_events = new std::set<RtEvent>();
        remote_insts = new FieldMaskSet<LogicalView>();
        restricted = false;
      }
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(redop);
          rez.serialize(target);
          rez.serialize(ready);
        }
        runtime->send_equivalence_set_remote_request_reductions(rit->first,rez);
        if (this == target)
        {
          AutoLock r_lock(*remote_lock);
          sync_events->insert(ready);
        }
        else
          ready_events.insert(ready);
      }
      if (this == target)
      {
        sync_remote_instances(insts);
        return restricted;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::perform_remote_updates(
                                  Operation *op, unsigned index,
                                  LogicalRegion handle,
                                  const RegionUsage &usage,
                                  const InstanceSet &targets,
                                  const std::vector<InstanceView*> &views,
                                  ApEvent precondition, ApEvent term_event,
                                  const RtEvent user_registered,
                                  std::set<RtEvent> &guard_events,
                                  std::set<RtEvent> &map_applied_events,
                                  std::set<ApEvent> &effects_events,
                                  const bool track_effects,
                                  const bool check_initialized) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
      assert(!targets.empty());
      assert(targets.size() == views.size());
#endif
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const RtUserEvent updated = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        const ApUserEvent effects = track_effects ? 
          Runtime::create_ap_user_event() : ApUserEvent::NO_AP_USER_EVENT;
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, rit->first);
          rez.serialize(index);
          rez.serialize(handle);
          rez.serialize(usage);
          rez.serialize<size_t>(targets.size());
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            const InstanceRef &ref = targets[idx];
            rez.serialize(ref.get_manager()->did);
            rez.serialize(views[idx]->did);
            rez.serialize(ref.get_valid_fields());
          }
          rez.serialize(precondition);
          rez.serialize(term_event);
          rez.serialize(updated);
          rez.serialize(user_registered);
          rez.serialize(applied);
          rez.serialize(effects);
          rez.serialize<bool>(check_initialized);
        }
        runtime->send_equivalence_set_remote_updates(rit->first, rez);
        guard_events.insert(updated);
        map_applied_events.insert(applied);
        if (track_effects)
          effects_events.insert(effects);
      }
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::perform_remote_acquires(Operation *op,
                                       std::set<RtEvent> &instances_returned,
                                       std::set<RtEvent> &map_applied_events,
                                       RemoteEqTracker *inst_target,
                                       const AddressSpaceID inst_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      if (this == inst_target)
      {
        remote_lock = new LocalLock();
        sync_events = new std::set<RtEvent>();
        remote_insts = new FieldMaskSet<LogicalView>();
      }
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const RtUserEvent returned = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, rit->first);
          rez.serialize(returned);
          rez.serialize(applied);
          rez.serialize(inst_target);
          rez.serialize(inst_owner);
        }
        runtime->send_equivalence_set_remote_acquires(rit->first, rez);
        map_applied_events.insert(applied);
        if (this == inst_target)
        {
          AutoLock r_lock(*remote_lock);
          sync_events->insert(returned);
        }
        else
          instances_returned.insert(returned);
      }
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::perform_remote_releases(Operation *op,
                                       const ApEvent precondition,
                                       std::set<RtEvent> &instances_returned,
                                       std::set<RtEvent> &map_applied_events,
                                       std::set<RtEvent> &guard_events,
                                       RemoteEqTracker *inst_target,
                                       const AddressSpaceID inst_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      if (this == inst_target)
      {
        remote_lock = new LocalLock();
        sync_events = new std::set<RtEvent>();
        remote_insts = new FieldMaskSet<LogicalView>();
      }
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const RtUserEvent returned = Runtime::create_rt_user_event();
        const RtUserEvent updated = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, rit->first);
          rez.serialize(precondition);
          rez.serialize(returned);
          rez.serialize(updated);
          rez.serialize(applied);
          rez.serialize(inst_target);
          rez.serialize(inst_owner);
        }
        runtime->send_equivalence_set_remote_releases(rit->first, rez);
        map_applied_events.insert(applied);
        guard_events.insert(updated);
        if (this == inst_target)
        {
          AutoLock r_lock(*remote_lock);
          sync_events->insert(returned);
        }
        else
          instances_returned.insert(returned);
      }
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::perform_remote_copies_across(Operation *op, 
                            const unsigned src_index,
                            const unsigned dst_index,
                            const RegionUsage &src_usage,
                            const RegionUsage &dst_usage,
                            const FieldMask &dst_mask,
                            const InstanceSet &dst_instances,
                            const std::vector<InstanceView*> &dst_views,
                            const LogicalRegion src_handle,
                            const LogicalRegion dst_handle,
                            const PredEvent pred_guard,
                            const ApEvent precondition,
                            const ReductionOpID redop,
                            const bool perfect,
                            const std::vector<unsigned> &src_indexes,
                            const std::vector<unsigned> &dst_indexes,
                                  std::set<RtEvent> &map_applied_events,
                                  std::set<ApEvent> &copy_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
      assert(dst_instances.size() == dst_views.size());
      assert(src_indexes.size() == dst_indexes.size());
#endif
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const ApUserEvent copy = Runtime::create_ap_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, rit->first);
          rez.serialize(src_index);
          rez.serialize(dst_index);
          rez.serialize(src_usage);
          rez.serialize(dst_usage);
          rez.serialize(dst_mask);
          rez.serialize<size_t>(dst_instances.size());
          for (unsigned idx = 0; idx < dst_instances.size(); idx++)
          {
            dst_instances[idx].pack_reference(rez);
            rez.serialize(dst_views[idx]->did); 
          }
          rez.serialize(src_handle);
          rez.serialize(dst_handle);
          rez.serialize(pred_guard);
          rez.serialize(precondition);
          rez.serialize(redop);
          rez.serialize<bool>(perfect);
          if (!perfect)
          {
            rez.serialize<size_t>(src_indexes.size());
            for (unsigned idx = 0; idx < src_indexes.size(); idx++)
            {
              rez.serialize(src_indexes[idx]);
              rez.serialize(dst_indexes[idx]);
            }
          }
          rez.serialize(applied);
          rez.serialize(copy);
        }
        runtime->send_equivalence_set_remote_copies_across(rit->first, rez);
        map_applied_events.insert(applied);
        copy_events.insert(copy);
      }
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::perform_remote_overwrite(
                                  Operation *op, const unsigned index,
                                  InstanceView *local_view,
                                  LogicalView *registration_view,
                                  const PredEvent pred_guard,
                                  const ApEvent precondition,
                                  const RtEvent guard_event,
                                  const bool add_restriction,
                                  const bool track_effects,
                                  std::set<RtEvent> &map_applied_events,
                                  std::set<ApEvent> &effects_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      WrapperReferenceMutator mutator(map_applied_events);
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const RtUserEvent applied = Runtime::create_rt_user_event();
        const ApUserEvent effects = track_effects ? 
          Runtime::create_ap_user_event() : ApUserEvent::NO_AP_USER_EVENT;
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, rit->first);
          rez.serialize(index);
          if (local_view != NULL)
          {
            local_view->add_base_valid_ref(REMOTE_DID_REF, &mutator);
            rez.serialize(local_view->did);
          }
          else
            rez.serialize<DistributedID>(0);
          registration_view->add_base_valid_ref(REMOTE_DID_REF, &mutator);
          rez.serialize(registration_view->did);
          rez.serialize(pred_guard);
          rez.serialize(precondition);
          rez.serialize(guard_event);
          rez.serialize<bool>(add_restriction);
          rez.serialize(applied);
          rez.serialize(effects);
        }
        runtime->send_equivalence_set_remote_overwrites(rit->first, rez);
        map_applied_events.insert(applied);
        if (track_effects)
          effects_events.insert(effects);
      }
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::perform_remote_filter(Operation *op, 
                                         InstanceView *inst_view,
                                         LogicalView *registration_view,
                                         const bool remove_restriction,
                                         std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      WrapperReferenceMutator mutator(map_applied_events);
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, rit->first);
          if (inst_view != NULL)
          {
            inst_view->add_base_valid_ref(REMOTE_DID_REF, &mutator);
            rez.serialize(inst_view->did);
          }
          else
            rez.serialize<DistributedID>(0);
          if (registration_view != NULL)
          {
            registration_view->add_base_valid_ref(REMOTE_DID_REF, &mutator);
            rez.serialize(registration_view->did);
          }
          else
            rez.serialize<DistributedID>(0);
          rez.serialize(remove_restriction);
          rez.serialize(applied);
        }
        runtime->send_equivalence_set_remote_filters(rit->first, rez);
        map_applied_events.insert(applied);
      }
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::sync_remote_instances(FieldMaskSet<LogicalView> &set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_lock != NULL);
      assert(sync_events != NULL);
      assert(remote_insts != NULL);
#endif
      RtEvent wait_on;
      {
        AutoLock r_lock(*remote_lock);
#ifdef DEBUG_LEGION
        assert(!sync_events->empty());
#endif
        wait_on = Runtime::merge_events(*sync_events);
        sync_events->clear();
      }
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      // Do a second round to make sure all the views are valid
      // Don't need the lock this time as we know all the updates
      // have arrived now
      if (!sync_events->empty())
      {
        wait_on = Runtime::merge_events(*sync_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            remote_insts->begin(); it != remote_insts->end(); it++)
        set.insert(it->first, it->second);
      delete remote_lock;
      delete sync_events;
      delete remote_insts;
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::sync_remote_instances(FieldMaskSet<InstanceView> &set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_lock != NULL);
      assert(sync_events != NULL);
      assert(remote_insts != NULL);
#endif
      RtEvent wait_on;
      {
        AutoLock r_lock(*remote_lock);
#ifdef DEBUG_LEGION
        assert(!sync_events->empty());
#endif
        wait_on = Runtime::merge_events(*sync_events);
        sync_events->clear();
      }
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      // Do a second round to make sure all the views are valid
      // Don't need the lock this time as we know all the updates
      // have arrived now
      if (!sync_events->empty())
      {
        wait_on = Runtime::merge_events(*sync_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            remote_insts->begin(); it != remote_insts->end(); it++)
      {
        InstanceView *view = it->first->as_instance_view();
        set.insert(view, it->second);
      }
      delete remote_lock;
      delete sync_events;
      delete remote_insts;
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::sync_remote_instances(FieldMaskSet<ReductionView> &st)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_lock != NULL);
      assert(sync_events != NULL);
      assert(remote_insts != NULL);
#endif
      RtEvent wait_on;
      {
        AutoLock r_lock(*remote_lock);
#ifdef DEBUG_LEGION
        assert(!sync_events->empty());
#endif
        wait_on = Runtime::merge_events(*sync_events);
        sync_events->clear();
      }
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      // Do a second round to make sure all the views are valid
      // Don't need the lock this time as we know all the updates
      // have arrived now
      if (!sync_events->empty())
      {
        wait_on = Runtime::merge_events(*sync_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            remote_insts->begin(); it != remote_insts->end(); it++)
      {
        ReductionView *view = it->first->as_reduction_view();
        st.insert(view, it->second);
      }
      delete remote_lock;
      delete sync_events;
      delete remote_insts;
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::process_remote_instances(Deserializer &derez,
                                                   Runtime *runtime)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_lock != NULL);
      assert(sync_events != NULL);
      assert(remote_insts != NULL);
#endif
      size_t num_views;
      derez.deserialize(num_views);
      AutoLock r_lock(*remote_lock);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;  
        LogicalView *view = 
          runtime->find_or_request_logical_view(view_did, ready);
        if (ready.exists())
          sync_events->insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        remote_insts->insert(view, mask);
      }
      bool remote_restrict;
      derez.deserialize(remote_restrict);
      if (remote_restrict)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::process_local_instances(
            const FieldMaskSet<LogicalView> &views, const bool local_restricted)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_LEGION
      assert(remote_lock != NULL);
      assert(sync_events != NULL);
      assert(remote_insts != NULL);
#endif     
      AutoLock r_lock(*remote_lock);
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            views.begin(); it != views.end(); it++)
        remote_insts->insert(it->first, it->second);
      if (local_restricted)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::process_local_instances(
          const FieldMaskSet<ReductionView> &views, const bool local_restricted)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_lock != NULL);
      assert(sync_events != NULL);
      assert(remote_insts != NULL);
#endif
      AutoLock r_lock(*remote_lock);
      for (FieldMaskSet<ReductionView>::const_iterator it = 
            views.begin(); it != views.end(); it++)
        remote_insts->insert(it->first, it->second);
      if (local_restricted)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void RemoteEqTracker::filter_remote_expressions(
                                      FieldMaskSet<IndexSpaceExpression> &exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      FieldMaskSet<IndexSpaceExpression> remote_exprs; 
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::aligned::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              rit->second.begin(); it != rit->second.end(); it++)
          remote_exprs.insert(it->first->set_expr, it->second);
      FieldMaskSet<IndexSpaceExpression> to_add;
      std::vector<IndexSpaceExpression*> to_remove;
      if (remote_exprs.size() > 1)
      {
        LegionList<FieldSet<IndexSpaceExpression*> >::aligned field_sets;
        remote_exprs.compute_field_sets(FieldMask(), field_sets);
        for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
              const_iterator fit = field_sets.begin(); 
              fit != field_sets.end(); fit++)
        {
          IndexSpaceExpression *remote_expr = (fit->elements.size() == 1) ?
            *(fit->elements.begin()) : 
            runtime->forest->union_index_spaces(fit->elements);
          for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
                exprs.begin(); it != exprs.end(); it++)
          {
            const FieldMask overlap = it->second & fit->set_mask;
            if (!overlap)
              continue;
            IndexSpaceExpression *diff = 
              runtime->forest->subtract_index_spaces(it->first, remote_expr);
            if (!diff->is_empty())
              to_add.insert(diff, overlap);
            it.filter(overlap);
            if (!it->second)
              to_remove.push_back(it->first);
          }
        }
      }
      else
      {
        FieldMaskSet<IndexSpaceExpression>::const_iterator first = 
          remote_exprs.begin();
        
        for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
              exprs.begin(); it != exprs.end(); it++)
        {
          const FieldMask overlap = it->second & first->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *diff = 
            runtime->forest->subtract_index_spaces(it->first, first->first);
          if (!diff->is_empty())
            to_add.insert(diff, overlap);
          it.filter(overlap);
          if (!it->second)
            to_remove.push_back(it->first);
        }
      }
      if (!to_remove.empty())
      {
        for (std::vector<IndexSpaceExpression*>::const_iterator it = 
              to_remove.begin(); it != to_remove.end(); it++)
          exprs.erase(*it);
      }
      if (!to_add.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
              to_add.begin(); it != to_add.end(); it++)
          exprs.insert(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_request_instances(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      RemoteEqTracker *target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);

      bool restricted = false;
      FieldMaskSet<LogicalView> valid_insts;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        ready_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_INSTANCE_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_MESSAGE_PRIORITY, wait_on);
          return;
#else
          wait_on.wait();
#endif
        }
      }
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        if (eq_sets[idx]->find_valid_instances(remote_tracker, 
                                               valid_insts, eq_masks[idx]))
          restricted = true;
      if (remote_tracker.has_remote_sets())
        remote_tracker.request_remote_instances(valid_insts,
                                                ready_events, target);
      if (!valid_insts.empty())
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target);
            rez.serialize(response_event);
            rez.serialize<size_t>(valid_insts.size());
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  valid_insts.begin(); it != valid_insts.end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
            rez.serialize<bool>(restricted);
          }
          runtime->send_equivalence_set_remote_instances(original_source, rez);
          ready_events.insert(response_event);
        }
        else
          target->process_local_instances(valid_insts, restricted);
      }
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_request_reductions(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      ReductionOpID redop;
      derez.deserialize(redop);
      RemoteEqTracker *target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);

      bool restricted = false;
      FieldMaskSet<ReductionView> reduction_insts;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        ready_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_REDUCTION_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_MESSAGE_PRIORITY, wait_on);
          return;
#else
          wait_on.wait();
#endif
        }
      }
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        if (eq_sets[idx]->find_reduction_instances(remote_tracker, 
                            reduction_insts, redop, eq_masks[idx]))
          restricted = true;
      if (remote_tracker.has_remote_sets())
        remote_tracker.request_remote_reductions(reduction_insts, redop, 
                                                 ready_events, target);
      if (!reduction_insts.empty())
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target);
            rez.serialize(response_event);
            rez.serialize<size_t>(reduction_insts.size());
            for (FieldMaskSet<ReductionView>::const_iterator it = 
                  reduction_insts.begin(); it != reduction_insts.end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
            rez.serialize<bool>(restricted);
          }
          runtime->send_equivalence_set_remote_instances(original_source, rez);
          ready_events.insert(response_event);
        }
        else
          target->process_local_instances(reduction_insts, restricted);
      }
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_updates(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      FieldMask user_mask;
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
        user_mask |= eq_masks[idx];
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionUsage usage;
      derez.deserialize(usage);
      size_t num_targets;
      derez.deserialize(num_targets);
      InstanceSet targets(num_targets);
      std::vector<InstanceView*> target_views(num_targets, NULL);
      for (unsigned idx = 0; idx < num_targets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        PhysicalManager *manager = 
          runtime->find_or_request_physical_manager(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(did);
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        target_views[idx] = static_cast<InstanceView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask valid_fields;
        derez.deserialize(valid_fields);
        targets[idx] = InstanceRef(manager, valid_fields);
      }
      ApEvent precondition;
      derez.deserialize(precondition);
      ApEvent term_event;
      derez.deserialize(term_event);
      RtUserEvent updated;
      derez.deserialize(updated);
      RtEvent user_registered;
      derez.deserialize(user_registered);
      RtUserEvent applied;
      derez.deserialize(applied);
      ApUserEvent effects_done;
      derez.deserialize(effects_done);
      const bool track_effects = effects_done.exists();
      bool check_initialized;
      derez.deserialize(check_initialized);

      FieldMaskSet<EquivalenceSet> dummy_alt_sets;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      std::map<RtEvent,CopyFillAggregator*> input_aggregators;
      CopyFillAggregator *output_aggregator = NULL;
      std::set<RtEvent> guard_events;
      std::set<RtEvent> map_applied_events;
      std::set<ApEvent> effects_events;
      PhysicalTraceInfo trace_info(op);
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_UPDATE_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_MESSAGE_PRIORITY, wait_on);
          // Don't leak the operation
          delete op;
          return;
#else
          wait_on.wait();
#endif
        }
      }
      if (!IS_DISCARD(usage) && !IS_SIMULT(usage) && check_initialized)
      {
        FieldMask initialized(user_mask);
        for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        {
          eq_sets[idx]->update_set(remote_tracker, dummy_alt_sets, NULL,
              original_source, op, index, usage, eq_masks[idx], targets, 
              target_views, input_aggregators, output_aggregator, 
              map_applied_events, guard_events, &initialized);
        }
        if (user_mask != initialized)
        {
          RegionNode *region_node = runtime->forest->get_node(handle);
          const FieldMask uninitialized = user_mask - initialized;
          region_node->report_uninitialized_usage(op,index,usage,uninitialized);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        {
          eq_sets[idx]->update_set(remote_tracker, dummy_alt_sets, NULL,
              original_source, op, index, usage, eq_masks[idx], targets, 
              target_views, input_aggregators, output_aggregator, 
              map_applied_events, guard_events);
        }
      }
      // If we have remote messages to send do that now
      if (remote_tracker.has_remote_sets())
      {
        remote_tracker.perform_remote_updates(op, index, handle, usage, 
               targets, target_views, precondition, term_event, 
               user_registered, guard_events, map_applied_events, 
               effects_events, track_effects, check_initialized);
      }
      // If we have any input aggregators, perform those copies now too
      // so that we know they'll be done before we do our registration
      if (!input_aggregators.empty())
      {
        for (std::map<RtEvent,CopyFillAggregator*>::const_iterator it = 
              input_aggregators.begin(); it != input_aggregators.end(); it++)
        {
          it->second->issue_updates(trace_info, precondition);
          // We can only use the guard_postcondition if we're on
          // the same node as the original, otherwise we need to use
          // the full effects_applied
#ifdef NON_AGGRESSIVE_AGGREGATORS
          guard_events.insert(it->second->effects_applied);
#else
          if (original_source == runtime->address_space)
            guard_events.insert(it->second->guard_postcondition);
          else
            guard_events.insert(it->second->effects_applied);
#endif
          if (it->second->release_guards(map_applied_events))
            delete it->second;
        }
      }
      // We can trigger our updated event done when all the guards are done 
      if (!guard_events.empty())
        Runtime::trigger_event(updated,
            Runtime::merge_events(guard_events));
      else
        Runtime::trigger_event(updated);
      // If we have outputs we need for the user to be registered
      // before we can apply the output copies
      ApEvent result;
      if (output_aggregator != NULL)
      {
        // Wait until the user registration is done before issuing
        // any output copies so we get the effects right
        if (effects_done.exists() ||
            (user_registered.exists() && !user_registered.has_triggered()))
        {
          RtUserEvent deferred_applied = Runtime::create_rt_user_event();
          ApUserEvent deferred_result;
          if (effects_done.exists())
            deferred_result = Runtime::create_ap_user_event();
          DeferRemoteOutputArgs args(output_aggregator, op, 
              op->get_unique_op_id(), term_event, 
              deferred_applied, deferred_result);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, user_registered);
          map_applied_events.insert(deferred_applied);
          result = deferred_result;
        }
        else
        {
          // No need to summarize the results since we would have
          // deferred this if we needed to track the effects because
          // it could potentially block
          output_aggregator->issue_updates(trace_info, term_event);
          if (output_aggregator->release_guards(map_applied_events))
            delete output_aggregator;
        }
      }
      // Do the rest of the triggers
      if (!map_applied_events.empty())
        Runtime::trigger_event(applied,
            Runtime::merge_events(map_applied_events));
      else
        Runtime::trigger_event(applied);
      if (effects_done.exists())
        Runtime::trigger_event(effects_done, result);
      // We can clean up our remote operation once we know there are no
      // more outstanding copy operations still in flight
      if (!updated.has_triggered() || !applied.has_triggered())
        op->defer_deletion(Runtime::merge_events(updated, applied));
      else
        delete op;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_acquires(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      RtUserEvent returned;
      derez.deserialize(returned);
      RtUserEvent applied;
      derez.deserialize(applied);
      RemoteEqTracker *inst_target;
      derez.deserialize(inst_target);
      AddressSpaceID inst_owner;
      derez.deserialize(inst_owner);

      FieldMaskSet<InstanceView> instances;
      FieldMaskSet<EquivalenceSet> dummy_alt_sets;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      std::set<RtEvent> instances_returned;
      std::set<RtEvent> map_applied_events;
      PhysicalTraceInfo trace_info(op);
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_ACQUIRE_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_MESSAGE_PRIORITY, wait_on);
          // Don't leak the operation
          delete op;
          return;
#else
          wait_on.wait();
#endif
        }
      }
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        eq_sets[idx]->acquire_restrictions(remote_tracker, dummy_alt_sets, NULL,
                                           original_source, op, eq_masks[idx], 
                                           instances, map_applied_events);
      if (remote_tracker.has_remote_sets())
        remote_tracker.perform_remote_acquires(op, instances_returned, 
                                map_applied_events, inst_target, inst_owner);
      // If we have response to send then we do that now
      if ((inst_target != NULL) && !instances.empty())
      {
        const RtUserEvent response_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(inst_target);
          rez.serialize(response_event);
          rez.serialize<size_t>(instances.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                instances.begin(); it != instances.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize<bool>(false); // dummy restricted value
        }
        runtime->send_equivalence_set_remote_instances(inst_owner, rez);
        instances_returned.insert(response_event);
      }
      if (!instances_returned.empty())
        Runtime::trigger_event(returned,
            Runtime::merge_events(instances_returned));
      else
        Runtime::trigger_event(returned);
      // Now we can trigger our applied event
      if (!map_applied_events.empty())
        Runtime::trigger_event(applied, 
            Runtime::merge_events(map_applied_events));
      else
        Runtime::trigger_event(applied);
      // Clean up the remote operation we allocated
      delete op;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_releases(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      ApEvent precondition;
      derez.deserialize(precondition);
      RtUserEvent returned;
      derez.deserialize(returned);
      RtUserEvent updated;
      derez.deserialize(updated);
      RtUserEvent applied;
      derez.deserialize(applied);
      RemoteEqTracker *inst_target;
      derez.deserialize(inst_target);
      AddressSpaceID inst_owner;
      derez.deserialize(inst_owner);

      FieldMaskSet<InstanceView> instances;
      FieldMaskSet<EquivalenceSet> dummy_alt_sets;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      std::set<RtEvent> instances_returned;
      std::set<RtEvent> guard_events;
      std::set<RtEvent> map_applied_events;
      PhysicalTraceInfo trace_info(op);
      CopyFillAggregator *release_aggregator = NULL;
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_RELEASE_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_MESSAGE_PRIORITY, wait_on);
          // Don't leak the operation
          delete op;
          return;
#else
          wait_on.wait();
#endif
        }
      }
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        eq_sets[idx]->release_restrictions(remote_tracker, dummy_alt_sets, NULL,
                                           original_source, op, eq_masks[idx], 
                                           release_aggregator, instances,
                                           map_applied_events);
      if (remote_tracker.has_remote_sets())
      {
        remote_tracker.perform_remote_releases(op, precondition,
            instances_returned, map_applied_events, 
            guard_events, inst_target, inst_owner);
      }
      // Issue any release copies/fills that need to be done
      if (release_aggregator != NULL)
      {
        release_aggregator->issue_updates(trace_info, precondition);
#ifdef NON_AGGRESSIVE_AGGREGATORS
        guard_events.insert(release_aggregator->effects_applied);
#else
        if (original_source == runtime->address_space)
          guard_events.insert(release_aggregator->guard_postcondition);
        else
          guard_events.insert(release_aggregator->effects_applied);
#endif
        if (release_aggregator->release_guards(map_applied_events))
          delete release_aggregator;
      }
      // If we have response to send then we do that now
      if ((inst_target != NULL) && !instances.empty())
      {
        const RtUserEvent response_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(inst_target);
          rez.serialize(response_event);
          rez.serialize<size_t>(instances.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                instances.begin(); it != instances.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
        }
        runtime->send_equivalence_set_remote_instances(inst_owner, rez);
        instances_returned.insert(response_event);
      }
      // Now trigger all our local events
      if (!guard_events.empty())
        Runtime::trigger_event(updated, Runtime::merge_events(guard_events));
      else
        Runtime::trigger_event(updated);
      if (!instances_returned.empty())
        Runtime::trigger_event(returned, 
            Runtime::merge_events(instances_returned));
      else
        Runtime::trigger_event(returned);
      if (!map_applied_events.empty())
        Runtime::trigger_event(applied,
            Runtime::merge_events(map_applied_events));
      else
        Runtime::trigger_event(applied);
      // Clean up the remote operation we allocated once updated triggers
      if (!updated.has_triggered())
        op->defer_deletion(updated);
      else
        delete op;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_copies_across(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      FieldMask src_mask;
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
        src_mask |= eq_masks[idx];
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned src_index, dst_index;
      derez.deserialize(src_index);
      derez.deserialize(dst_index);
      RegionUsage src_usage, dst_usage;
      derez.deserialize(src_usage);
      derez.deserialize(dst_usage);
      FieldMask dst_mask;
      derez.deserialize(dst_mask);
      size_t num_dsts;
      derez.deserialize(num_dsts);
      InstanceSet dst_instances(num_dsts);
      std::vector<InstanceView*> dst_views(num_dsts, NULL);
      for (unsigned idx = 0; idx < num_dsts; idx++)
      {
        RtEvent ready;
        dst_instances[idx].unpack_reference(runtime, derez, ready);
        if (ready.exists())
          ready_events.insert(ready);
        DistributedID did;
        derez.deserialize(did);
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        dst_views[idx] = static_cast<InstanceView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
      }
      LogicalRegion src_handle, dst_handle;
      derez.deserialize(src_handle);
      derez.deserialize(dst_handle);
      PredEvent pred_guard;
      derez.deserialize(pred_guard);
      ApEvent precondition;
      derez.deserialize(precondition);
      ReductionOpID redop;
      derez.deserialize(redop);
      bool perfect;
      derez.deserialize(perfect);
      std::vector<unsigned> src_indexes, dst_indexes;
      if (!perfect)
      {
        size_t num_indexes;
        derez.deserialize(num_indexes);
        src_indexes.resize(num_indexes);
        dst_indexes.resize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
        {
          derez.deserialize(src_indexes[idx]);
          derez.deserialize(dst_indexes[idx]);
        }
      }
      RtUserEvent applied;
      derez.deserialize(applied);
      ApUserEvent copy;
      derez.deserialize(copy);

      FieldMaskSet<EquivalenceSet> dummy_alt_sets;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      CopyFillAggregator *across_aggregator = NULL;
      FieldMask initialized = src_mask;
      std::vector<CopyAcrossHelper*> across_helpers;
      RegionNode *dst_node = runtime->forest->get_node(dst_handle);
      IndexSpaceExpression *dst_expr = dst_node->get_index_space_expression();
      FieldMaskSet<IndexSpaceExpression> local_exprs;
      std::set<RtEvent> map_applied_events;
      std::set<ApEvent> copy_events;
      PhysicalTraceInfo trace_info(op);
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_COPIES_ACROSS_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_MESSAGE_PRIORITY, wait_on);
          // Don't leak the operation
          delete op;
          return;
#else
          wait_on.wait();
#endif
        }
      }
      if (perfect)
      {
        for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        {
          EquivalenceSet *set = eq_sets[idx];
          // Check that the index spaces intersect
          IndexSpaceExpression *overlap = 
            runtime->forest->intersect_index_spaces(set->set_expr, dst_expr);
          if (overlap->is_empty())
            continue;
          set->issue_across_copies(remote_tracker, dummy_alt_sets, NULL,
              original_source, op, src_index, dst_index,dst_usage,eq_masks[idx],
              dst_instances, dst_views, overlap, across_aggregator, pred_guard, 
              redop, initialized, map_applied_events);
          local_exprs.insert(overlap, eq_masks[idx]);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < dst_instances.size(); idx++)
        {
          across_helpers.push_back(
              new CopyAcrossHelper(src_mask, src_indexes, dst_indexes));
          InstanceManager *manager = 
            dst_instances[idx].get_manager()->as_instance_manager();
          manager->initialize_across_helper(across_helpers.back(), 
                                dst_mask, src_indexes, dst_indexes);
        }
        for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        {
          EquivalenceSet *set = eq_sets[idx];
          // Check that the index spaces intersect
          IndexSpaceExpression *overlap = 
            runtime->forest->intersect_index_spaces(set->set_expr, dst_expr);
          if (overlap->is_empty())
            continue;
          set->issue_across_copies(remote_tracker, dummy_alt_sets, NULL,
                                   original_source, op, src_index, dst_index, 
                                   dst_usage, eq_masks[idx], dst_instances, 
                                   dst_views, overlap, across_aggregator, 
                                   pred_guard, redop, initialized, 
                                   map_applied_events, &src_indexes, 
                                   &dst_indexes, &across_helpers);
          local_exprs.insert(overlap, eq_masks[idx]);
        } 
      }
      if (initialized != src_mask)
      {
        RegionNode *src_node = runtime->forest->get_node(src_handle);
        const FieldMask uninitialized = src_mask - initialized;
        src_node->report_uninitialized_usage(op, src_index,
                                             src_usage, uninitialized);
      }
      if (remote_tracker.has_remote_sets())
      {
        std::set<IndexSpaceExpression*> remote_exprs;
        remote_tracker.perform_remote_copies_across(op, src_index, dst_index,
            src_usage, dst_usage, dst_mask, dst_instances, dst_views,
            src_handle, dst_handle, pred_guard, precondition, redop, perfect, 
            src_indexes, dst_indexes, map_applied_events, copy_events);
        remote_tracker.filter_remote_expressions(local_exprs);
      }
      if (across_aggregator != NULL)
      {
#ifdef DEBUG_LEGION
        assert(!local_exprs.empty());
#endif
        // Record the event field preconditions for each view
        // Use the destination expr since we know we we're only actually
        // issuing copies for that particular expression
        if (local_exprs.size() > 1)
        {
          LegionList<FieldSet<IndexSpaceExpression*> >::aligned field_sets;
          local_exprs.compute_field_sets(FieldMask(), field_sets);
          for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
                const_iterator it = field_sets.begin(); 
                it != field_sets.end(); it++)
          {
            IndexSpaceExpression *expr = (it->elements.size() == 1) ? 
              *(it->elements.begin()) :
              runtime->forest->union_index_spaces(it->elements);
            if (expr->is_empty())
              continue;
            for (unsigned idx = 0; idx < dst_instances.size(); idx++)
            {
              const InstanceRef &ref = dst_instances[idx];
              const ApEvent event = ref.get_ready_event();
              if (!event.exists())
                continue;
              const FieldMask &mask = ref.get_valid_fields();
              // Convert these to destination fields if necessary
              const FieldMask overlap = mask & (perfect ? it->set_mask :
                  across_helpers[idx]->convert_src_to_dst(it->set_mask));
              if (!overlap)
                continue;
              InstanceView *view = dst_views[idx];
              across_aggregator->record_precondition(view, false/*reading*/,
                                                     event, overlap, expr);
            }
          }
        }
        else
        {
          FieldMaskSet<IndexSpaceExpression>::const_iterator first = 
            local_exprs.begin();
          if (!first->first->is_empty())
          {
            for (unsigned idx = 0; idx < dst_instances.size(); idx++)
            {
              const InstanceRef &ref = dst_instances[idx];
              const ApEvent event = ref.get_ready_event();
              if (!event.exists())
                continue;
              const FieldMask &mask = ref.get_valid_fields();
              // Convert these to destination fields if necessary
              const FieldMask overlap = mask & (perfect ? first->second : 
                  across_helpers[idx]->convert_src_to_dst(first->second));
              if (!overlap)
                continue;
              InstanceView *view = dst_views[idx];
              across_aggregator->record_precondition(view, false/*reading*/,
                                               event, overlap, first->first);
            }
          }
        }
        across_aggregator->issue_updates(trace_info, precondition,
            false/*has src preconditions*/, true/*has dst preconditions*/);
        // Need to wait before we can get the summary
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!across_aggregator->effects_applied.has_triggered())
          across_aggregator->effects_applied.wait();
#else
        if (!across_aggregator->guard_postcondition.has_triggered())
          across_aggregator->guard_postcondition.wait();
#endif
        const ApEvent result = across_aggregator->summarize(trace_info);
        if (result.exists())
          copy_events.insert(result);
        if (across_aggregator->release_guards(map_applied_events))
          delete across_aggregator;
      }
      if (!across_helpers.empty())
      {
        for (unsigned idx = 0; idx < across_helpers.size(); idx++)
          delete across_helpers[idx];
      }
      if (!copy_events.empty())
        Runtime::trigger_event(copy,
            Runtime::merge_events(&trace_info, copy_events));
      else
        Runtime::trigger_event(copy);
      // Now we can trigger our applied event
      if (!map_applied_events.empty())
        Runtime::trigger_event(applied, 
            Runtime::merge_events(map_applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_overwrites(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      DistributedID view_did;
      derez.deserialize(view_did);
      RtEvent view_ready;
      InstanceView *local_view = NULL;
      if (view_did > 0)
      {
        local_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
        if (view_ready.exists())
          ready_events.insert(view_ready);
      }
      derez.deserialize(view_did);
      LogicalView *registration_view = 
        runtime->find_or_request_logical_view(view_did, view_ready);
      if (view_ready.exists())
        ready_events.insert(view_ready);
      PredEvent pred_guard;
      derez.deserialize(pred_guard);
      ApEvent precondition;
      derez.deserialize(precondition);
      RtEvent guard_event;
      derez.deserialize(guard_event);
      bool add_restriction;
      derez.deserialize(add_restriction);
      RtUserEvent applied;
      derez.deserialize(applied);
      ApUserEvent effects;
      derez.deserialize(effects);

      CopyFillAggregator *output_aggregator = NULL;
      FieldMaskSet<EquivalenceSet> dummy_alt_sets;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      std::set<RtEvent> map_applied_events;
      std::set<ApEvent> effects_events;
      PhysicalTraceInfo trace_info(op);
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_OVERWRITE_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_MESSAGE_PRIORITY, wait_on);
          // Don't leak the operation
          delete op;
          return;
#else
          wait_on.wait(); 
#endif
        }
      }
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
      {
        EquivalenceSet *set = eq_sets[idx];
        set->overwrite_set(remote_tracker, dummy_alt_sets, NULL,
            original_source, op, index, registration_view, eq_masks[idx],
            output_aggregator, map_applied_events, pred_guard, add_restriction);
      }
      if (remote_tracker.has_remote_sets())
      {
        remote_tracker.perform_remote_overwrite(op, index, local_view, 
                registration_view, pred_guard, precondition, 
                guard_event, add_restriction, effects.exists(), 
                map_applied_events, effects_events);
      }
      if (output_aggregator != NULL)
      {
        // If we have a guard event we need to wait for it before we
        // can issue the copy updates
        if (effects.exists() ||
            (guard_event.exists() && !guard_event.has_triggered()))
        {
          RtUserEvent deferred_applied = Runtime::create_rt_user_event();
          ApUserEvent deferred_result;
          if (effects.exists())
          {
            deferred_result = Runtime::create_ap_user_event();
            effects_events.insert(deferred_result);
          }
          DeferRemoteOutputArgs args(output_aggregator, op, 
              op->get_unique_op_id(), precondition, 
              deferred_applied, deferred_result);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, guard_event);
          map_applied_events.insert(deferred_applied);
        }
        else
        {
          // No need to summarize the results since we would have
          // deferred this if we needed to track effects because
          // it could potentially block
          output_aggregator->issue_updates(trace_info, precondition);
          if (output_aggregator->release_guards(map_applied_events))
            delete output_aggregator;
        }
      }
      if (effects.exists())
      {
        if (!effects_events.empty())
          Runtime::trigger_event(effects,
              Runtime::merge_events(&trace_info, effects_events));
        else
          Runtime::trigger_event(effects);
      }
      // Now we can trigger our applied event
      if (!map_applied_events.empty())
        Runtime::trigger_event(applied, 
            Runtime::merge_events(map_applied_events));
      else
        Runtime::trigger_event(applied);
      if (local_view != NULL)
        local_view->send_remote_valid_decrement(previous, applied);
      registration_view->send_remote_valid_decrement(previous, applied);
      if (!applied.has_triggered())
        op->defer_deletion(applied);
      else
        delete op;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_filters(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEFER_REMOTE_EQ
      // Grab these now in case we need to defer them for later
      const void *buffer = derez.get_current_pointer();
      const size_t buffer_size = derez.get_remaining_bytes();
#endif
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask>::aligned eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      DistributedID view_did;
      derez.deserialize(view_did);
      InstanceView *inst_view = NULL;
      if (view_did != 0)
      {
        RtEvent view_ready;
        inst_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
        if (view_ready.exists())
          ready_events.insert(view_ready);
      }
      derez.deserialize(view_did);
      LogicalView *registration_view = NULL;
      if (view_did != 0)
      {
        RtEvent view_ready;
        registration_view = 
          runtime->find_or_request_logical_view(view_did, view_ready);
        if (view_ready.exists())
          ready_events.insert(view_ready);
      }
      bool remove_restriction;
      derez.deserialize(remove_restriction);
      RtUserEvent applied;
      derez.deserialize(applied);

      std::set<RtEvent> map_applied_events;
      FieldMaskSet<EquivalenceSet> dummy_alt_sets;
      RemoteEqTracker remote_tracker(previous, original_source, runtime);
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
#ifdef DEFER_REMOTE_EQ
          DeferRemoteArgs<LG_DEFER_REMOTE_FILTER_TASK_ID> 
            args(buffer, buffer_size, previous);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_MESSAGE_PRIORITY, wait_on);
          // Don't leak the operation
          delete op;
          return;
#else
          wait_on.wait(); 
#endif
        }
      }
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        eq_sets[idx]->filter_set(remote_tracker, dummy_alt_sets, NULL,
            original_source, op, inst_view, eq_masks[idx], map_applied_events,
            registration_view, remove_restriction);
      if (remote_tracker.has_remote_sets())
        remote_tracker.perform_remote_filter(op, inst_view, registration_view,
                                      remove_restriction, map_applied_events);
      // Now we can trigger our applied event
      if (!map_applied_events.empty())
        Runtime::trigger_event(applied, 
            Runtime::merge_events(map_applied_events));
      else
        Runtime::trigger_event(applied);
      if (inst_view != NULL)
        inst_view->send_remote_valid_decrement(previous, applied);
      if (registration_view != NULL)
        registration_view->send_remote_valid_decrement(previous, applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_remote_instances(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteEqTracker *target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      target->process_remote_instances(derez, runtime);
      Runtime::trigger_event(done_event); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_deferred_remote(LgTaskID tid,
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      switch (tid)
      {
        case LG_DEFER_REMOTE_INSTANCE_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_INSTANCE_TASK_ID> *dargs = 
              (const DeferRemoteArgs<LG_DEFER_REMOTE_INSTANCE_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_request_instances(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        case LG_DEFER_REMOTE_REDUCTION_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_REDUCTION_TASK_ID> *dargs = 
              (const DeferRemoteArgs<LG_DEFER_REMOTE_REDUCTION_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_request_reductions(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        case LG_DEFER_REMOTE_UPDATE_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_UPDATE_TASK_ID> *dargs = 
              (const DeferRemoteArgs<LG_DEFER_REMOTE_UPDATE_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_updates(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        case LG_DEFER_REMOTE_ACQUIRE_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_ACQUIRE_TASK_ID> *dargs = 
              (const DeferRemoteArgs<LG_DEFER_REMOTE_ACQUIRE_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_acquires(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        case LG_DEFER_REMOTE_RELEASE_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_RELEASE_TASK_ID> *dargs = 
              (const DeferRemoteArgs<LG_DEFER_REMOTE_RELEASE_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_releases(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        case LG_DEFER_REMOTE_COPIES_ACROSS_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_COPIES_ACROSS_TASK_ID> 
              *dargs = 
            (const DeferRemoteArgs<LG_DEFER_REMOTE_COPIES_ACROSS_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_copies_across(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        case LG_DEFER_REMOTE_OVERWRITE_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_OVERWRITE_TASK_ID> *dargs = 
              (const DeferRemoteArgs<LG_DEFER_REMOTE_OVERWRITE_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_overwrites(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        case LG_DEFER_REMOTE_FILTER_TASK_ID:
          {
            const DeferRemoteArgs<LG_DEFER_REMOTE_FILTER_TASK_ID> *dargs =
              (const DeferRemoteArgs<LG_DEFER_REMOTE_FILTER_TASK_ID>*)args;
            Deserializer derez(dargs->buffer, dargs->size);
            handle_remote_filters(derez, runtime, dargs->previous);
            free(dargs->buffer);
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteEqTracker::handle_deferred_output(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRemoteOutputArgs *dargs = (const DeferRemoteOutputArgs*)args;
      const PhysicalTraceInfo trace_info(dargs->op);
      std::set<RtEvent> map_applied;
      if (dargs->summary.exists())
      {
        dargs->aggregator->issue_updates(trace_info, dargs->precondition);
        // Need to wait before we can get the summary
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!dargs->aggregator->effects_applied.has_triggered())
          dargs->aggregator->effects_applied.wait();
#else
        if (!dargs->aggregator->guard_postcondition.has_triggered())
          dargs->aggregator->guard_postcondition.wait();
#endif
        const ApEvent summary = dargs->aggregator->summarize(trace_info);
        Runtime::trigger_event(dargs->summary, summary);
        if (dargs->aggregator->release_guards(map_applied))
          delete dargs->aggregator;
      }
      else
      {
        dargs->aggregator->issue_updates(trace_info, dargs->precondition);
        if (dargs->aggregator->release_guards(map_applied))
          delete dargs->aggregator;
      }
      if (!map_applied.empty())
        Runtime::trigger_event(dargs->applied,
            Runtime::merge_events(map_applied));
      else
        Runtime::trigger_event(dargs->applied);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set
    /////////////////////////////////////////////////////////////

    // C++ is dumb
    const VersionID EquivalenceSet::init_version;

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(Runtime *rt, DistributedID did,
                                   AddressSpaceID owner, AddressSpace logical,
                                   IndexSpaceExpression *expr,
                                   IndexSpaceNode *node,
                                   bool reg_now)
      : DistributedCollectable(rt, did, owner, reg_now), set_expr(expr),
        index_space_node(node), logical_owner_space(logical),
        eq_state(is_logical_owner() ? MAPPING_STATE : 
            // If we're not the logical owner but we are the owner
            // then we have a valid remote lease of the subsets
            is_owner() ? VALID_STATE : INVALID_STATE), 
        subset_exprs(NULL), sample_count(0)
    //--------------------------------------------------------------------------
    {
      set_expr->add_expression_reference();
      if (index_space_node != NULL)
      {
#ifdef DEBUG_LEGION
        // These two index space expressions should be equivalent
        // Although they don't have to be the same
        // These assertions are pretty expensive so we'll comment them
        // out for now, but put them back in if you think this invariant
        // is being invalidated
        //assert(runtime->forest->subtract_index_spaces(index_space_node,
        //                                              set_expr)->is_empty());
        //assert(runtime->forest->subtract_index_spaces(set_expr,
        //                                      index_space_node)->is_empty());
#endif
        index_space_node->add_nested_resource_ref(did);
      }
      if (is_logical_owner() && !is_owner())
        remote_subsets.insert(owner_space);
#ifdef LEGION_GC
      log_garbage.info("GC Equivalence Set %lld %d", did, local_space);
#endif
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(const EquivalenceSet &rhs)
      : DistributedCollectable(rhs), set_expr(NULL), index_space_node(NULL), 
        logical_owner_space(rhs.logical_owner_space)
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
      if ((index_space_node != NULL) && 
          index_space_node->remove_nested_resource_ref(did))
        delete index_space_node;
      if (!subsets.empty())
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
          if (it->first->remove_nested_resource_ref(did))
            delete it->first;
        subsets.clear();
      }
      if (!valid_instances.empty())
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
          if (it->first->remove_nested_valid_ref(did))
            delete it->first;
      }
      if (!reduction_instances.empty())
      {
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = reduction_instances.begin(); 
              rit != reduction_instances.end(); rit++)
        {
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
            if ((*it)->remove_nested_valid_ref(did))
              delete (*it);
        }
      }
      if (!restricted_instances.empty())
      {
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              restricted_instances.begin(); it != 
              restricted_instances.end(); it++)
          if (it->first->remove_nested_valid_ref(did))
            delete it->first;
      }
      if (!disjoint_partition_refinements.empty())
      {
        for (FieldMaskSet<DisjointPartitionRefinement>::const_iterator it =
              disjoint_partition_refinements.begin(); it !=
              disjoint_partition_refinements.end(); it++)
          delete it->first;
      }
      if (subset_exprs != NULL)
        delete subset_exprs;
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
    void EquivalenceSet::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    } 

    //--------------------------------------------------------------------------
    AddressSpaceID EquivalenceSet::clone_from(const EquivalenceSet *parent,
                                              const FieldMask &clone_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should be cloning from the parent on it's owner space
      assert(parent->logical_owner_space == this->local_space);
#endif
      // Take our lock in exclusive mode since we're going to be updating
      // our data structures
      AutoLock eq(eq_lock);
      // Check to see if we're the logical owner, if not then tell 
      // the refinement task where it should send the data
      if (!is_logical_owner())
        return logical_owner_space;
      // We are the logical owner so clone the meta data
      // No need for a mutator here since all the views already
      // have valid references being held by the parent equivalence set
      if (!parent->valid_instances.empty() && 
          !(clone_mask * parent->valid_instances.get_valid_mask()))
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              parent->valid_instances.begin(); it !=
              parent->valid_instances.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          if (this->valid_instances.insert(it->first, overlap))
            it->first->add_nested_valid_ref(did);
        }
      }
      if (!!parent->reduction_fields)
      {
        const FieldMask reduc_overlap = parent->reduction_fields & clone_mask;
        if (!!reduc_overlap)
        {
          this->reduction_fields |= reduc_overlap;
          int fidx = reduc_overlap.find_first_set();
          while (fidx >= 0)
          {
            std::vector<ReductionView*> &reduc_insts = 
              this->reduction_instances[fidx];
#ifdef DEBUG_LEGION
            assert(reduc_insts.empty());
#endif
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = parent->reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != parent->reduction_instances.end());
#endif
            reduc_insts = finder->second;
            for (unsigned idx = 0; idx < reduc_insts.size(); idx++)
              reduc_insts[idx]->add_nested_valid_ref(did);
            fidx = reduc_overlap.find_next_set(fidx+1);
          }
        }
      }
      if (!parent->restricted_instances.empty() &&
          !(clone_mask * parent->restricted_instances.get_valid_mask()))
      {
        this->restricted_fields |= (clone_mask & parent->restricted_fields);
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              parent->restricted_instances.begin(); it !=
              parent->restricted_instances.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          if (this->restricted_instances.insert(it->first, overlap))
            it->first->add_nested_valid_ref(did);
        }
      }
      // Return our space since we stored the data here
      return local_space;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::remove_update_guard(CopyFillAggregator *aggregator)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      FieldMaskSet<CopyFillAggregator>::iterator finder = 
        update_guards.find(aggregator);
#ifdef DEBUG_LEGION
      assert(finder != update_guards.end());
#endif
      const bool should_tighten = !!finder->second;
      update_guards.erase(finder);
      if (should_tighten)
        update_guards.tighten_valid_mask();
      if ((eq_state == REFINING_STATE) && 
          transition_event.exists() && (update_guards.empty() ||
            (update_guards.get_valid_mask() * 
             pending_refinements.get_valid_mask())))
      {
        Runtime::trigger_event(transition_event);
        transition_event = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::check_for_unrefined_remainder(AutoLock &eq,
                                                       const FieldMask &mask,
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
        return;
      bool first_pass = true;
      do 
      {
        // If this isn't the first pass then we need to wait
        if (!first_pass)
        {
#ifdef DEBUG_LEGION
          assert(refinement_event.exists());
#endif
          const RtEvent wait_on = refinement_event;
          eq.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          eq.reacquire();
          // When we wake up we have to do all the checks again
          // in case there were fields that weren't refined before
          // but are (partially or being) refined now
        }
        else
          first_pass = false;
        // Check for any disjoint pieces
        if (!disjoint_partition_refinements.empty())
        {
          FieldMask disjoint_overlap = 
            disjoint_partition_refinements.get_valid_mask() & mask;
          if (!!disjoint_overlap)
          {
            std::vector<DisjointPartitionRefinement*> to_delete;
            for (FieldMaskSet<DisjointPartitionRefinement>::iterator it = 
                  disjoint_partition_refinements.begin(); it !=
                  disjoint_partition_refinements.end(); it++)
            {
              const FieldMask overlap = it->second & disjoint_overlap;
              if (!overlap)
                continue;
              finalize_disjoint_refinement(it->first, overlap);
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              disjoint_overlap -= overlap;
              if (!disjoint_overlap)
                break;
            }
            if (!to_delete.empty())
            {
              for (std::vector<DisjointPartitionRefinement*>::const_iterator 
                    it = to_delete.begin(); it != to_delete.end(); it++)
              {
                disjoint_partition_refinements.erase(*it);
                delete (*it);
              }
              disjoint_partition_refinements.tighten_valid_mask();
            }
          }
        }
        // Check for unrefined remainder pieces too
        if (!unrefined_remainders.empty())
        {
          FieldMask unrefined_overlap = 
            unrefined_remainders.get_valid_mask() & mask;
          if (!!unrefined_overlap)
          {
            std::vector<IndexSpaceExpression*> to_delete;
            for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
                  unrefined_remainders.begin(); it != 
                  unrefined_remainders.end(); it++)
            {
              const FieldMask overlap = it->second & unrefined_overlap;
              if (!overlap)
                continue;
              add_pending_refinement(it->first, overlap, NULL/*node*/, source); 
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              unrefined_overlap -= overlap;
              if (!unrefined_overlap)
                break;
            }
            if (!to_delete.empty())
            {
              for (std::vector<IndexSpaceExpression*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
                unrefined_remainders.erase(*it);
              unrefined_remainders.tighten_valid_mask();
            }
          }
        }
      } 
      // See if we need to wait for any refinements to finish
      while (!(mask * pending_refinements.get_valid_mask()) ||
              !(mask * refining_fields));
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::ray_trace_equivalence_sets(VersionManager *target,
                                                    IndexSpaceExpression *expr,
                                                    FieldMask ray_mask,
                                                    IndexSpace handle,
                                                    AddressSpaceID source,
                                                    RtUserEvent trace_done,
                                                    RtUserEvent deferral_event)
    //--------------------------------------------------------------------------
    {
      // Handle a special case to avoid over-decomposing with larger index
      // spaces unnecessarily. If it does need to be refined later then 
      // we'll do that as part of an acquire operation. This is only
      // a performance optimization for getting better BVH shapes and
      // does not impact the correctness of the code
      if (handle.exists() && (index_space_node != NULL) &&
          (index_space_node->handle == handle))
      {
#ifdef DEBUG_LEGION
        assert(!deferral_event.exists());
#endif
        // Just record this as one of the results
        if (source != runtime->address_space)
        {
          // Not local so we need to send a message
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(ray_mask);
            rez.serialize(target);
            rez.serialize(trace_done);
          }
          runtime->send_equivalence_set_ray_trace_response(source, rez);
          return;
        }
        else // Local so we can update this directly
        {
          target->record_equivalence_set(this, ray_mask);
          Runtime::trigger_event(trace_done);
          return;
        }
      }
      RegionTreeForest *forest = runtime->forest;
#ifdef DEBUG_LEGION
      assert(expr != NULL);
      // An expensive sanity check if you want to turn it on
      //assert(forest->subtract_index_spaces(expr, set_expr)->is_empty());
#endif
      RtEvent refinement_done;
      FieldMaskSet<EquivalenceSet> to_traverse, pending_to_traverse;
      std::map<EquivalenceSet*,IndexSpaceExpression*> to_traverse_exprs;
      {
        // Try to get the lock, if we don't get it build a continuation 
        AutoTryLock eq(eq_lock);
        if (!eq.has_lock())
        {
          // We didn't get the lock so build a continuation
          // We need a name for our completion event that we can use for
          // the atomic compare and swap below
          if (!deferral_event.exists())
          {
            // If we haven't already been deferred then we need to 
            // add ourselves to the back of the list of deferrals
            deferral_event = Runtime::create_rt_user_event();
            volatile Realm::Event::id_t *ptr = 
              (volatile Realm::Event::id_t*)&next_deferral_precondition.id;
            RtEvent continuation_pre;
            do {
              continuation_pre.id = *ptr;
            } while (!__sync_bool_compare_and_swap(ptr,
                      continuation_pre.id, deferral_event.id));
            DeferRayTraceArgs args(this, target, expr, handle, source, 
                                   trace_done, deferral_event, ray_mask);
            runtime->issue_runtime_meta_task(args, 
                            LG_THROUGHPUT_DEFERRED_PRIORITY, continuation_pre);
          }
          else
          {
            // We've already been deferred and our precondition has already
            // triggered so just launch ourselves again whenever the lock
            // should be ready to try again
            DeferRayTraceArgs args(this, target, expr, handle, source, 
                                   trace_done, deferral_event, ray_mask);
            runtime->issue_runtime_meta_task(args,
                              LG_THROUGHPUT_DEFERRED_PRIORITY, eq.try_next());
          }
          return;
        }
        else if (!is_logical_owner())
        {
          // If we're not the owner node then send the request there
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(target);
            expr->pack_expression(rez, logical_owner_space);
            rez.serialize(ray_mask);
            rez.serialize(handle);
            rez.serialize(source);
            rez.serialize(trace_done);
          }
          runtime->send_equivalence_set_ray_trace_request(logical_owner_space,
                                                          rez);
          // Trigger our deferral event if we had one
          if (deferral_event.exists())
            Runtime::trigger_event(deferral_event);
          return;
        }
        else if ((eq_state == REFINING_STATE) && 
                  !(ray_mask * refining_fields))
        {
          if (!transition_event.exists())
            transition_event = Runtime::create_rt_user_event();
          // If we're refining then we also need to defer this until 
          // the refinements that interfere with us are done
          DeferRayTraceArgs args(this, target, expr, handle, source, 
                                 trace_done, deferral_event, ray_mask);
          runtime->issue_runtime_meta_task(args,
                            LG_THROUGHPUT_DEFERRED_PRIORITY, transition_event);
          return;
        }
        // First check to see which fields are in a disjoint refinement
        // and whether we can continue doing the disjoint refinement
        if (!disjoint_partition_refinements.empty())
        {
#ifdef DEBUG_LEGION
          assert(index_space_node != NULL);
#endif
          FieldMask disjoint_overlap = ray_mask & 
            disjoint_partition_refinements.get_valid_mask();
          if (!!disjoint_overlap)
          {
            std::vector<DisjointPartitionRefinement*> to_delete;
            // Iterate over the disjoint partition refinements and see 
            // which ones we overlap with
            for (FieldMaskSet<DisjointPartitionRefinement>::iterator it =
                  disjoint_partition_refinements.begin(); it !=
                  disjoint_partition_refinements.end(); it++)
            {
              FieldMask overlap = it->second & disjoint_overlap;
              if (!overlap)
                continue;
              // Remove this from the disjoint overlap now in case
              // we end up removing overlap fields later
              disjoint_overlap -= overlap;
              // This is the special case where we are refining 
              // a disjoint partition and all the refinements so far
              // have been specific instances of a subregion of the
              // disjoint partition, check to see if that is still true
              if (handle.exists())
              {
                IndexSpaceNode *node = runtime->forest->get_node(handle);
                if (node->parent == it->first->partition)
                {
                  // Record that we're handling all these ray fields
                  // before we go about filtering the fields out of overlap
                  ray_mask -= overlap;
                  // Another sub-region of the disjoint partition
                  // See if we already made the refinement or not
                  std::map<IndexSpaceNode*,EquivalenceSet*>::const_iterator
                    finder = it->first->children.find(node);
                  if (finder == it->first->children.end())
                  {
                    EquivalenceSet *child = 
                      add_pending_refinement(expr, overlap, node, source);
                    pending_to_traverse.insert(child, overlap);
                    to_traverse_exprs[child] = expr;
                    // If this is a pending refinement then we'll need to
                    // wait for it before traversing farther
                    if (!refinement_done.exists())
                    {
#ifdef DEBUG_LEGION
                      assert(refinement_event.exists());
#endif
                      refinement_done = refinement_event;
                    }
                    // Record this child for the future
                    it->first->children[node] = child; 
                    // Remove these fields from the overlap indicating
                    // that we handled them
                    overlap.clear();
                  }
                  else
                  {
                    // Figure out which fields have already been refined
                    // and which ones are still pending, issue refinements
                    // for any fields that haven't been refined yet
                    FieldMaskSet<EquivalenceSet>::iterator eq_finder = 
                      subsets.find(finder->second);
                    if (eq_finder != subsets.end())
                    {
                      const FieldMask eq_valid = overlap & eq_finder->second;
                      if (!!eq_valid)
                      {
                        to_traverse.insert(finder->second, eq_valid);
                        to_traverse_exprs[finder->second] = expr;
                        overlap -= eq_valid;
                      }
                    }
                    // If we couldn't find it in the already valid set, check
                    // also in the pending refineemnts
                    if (!!overlap)
                    {
                      eq_finder = pending_refinements.find(finder->second);
                      if (eq_finder != pending_refinements.end())
                      {
                        const FieldMask eq_pending = 
                          overlap & eq_finder->second;
                        if (!!eq_pending)
                        {
                          pending_to_traverse.insert(finder->second,eq_pending);
                          to_traverse_exprs[finder->second] = expr;
                          overlap -= eq_pending;
                          // If this is a pending refinement then we'll need to
                          // wait for it before traversing farther
                          if (!refinement_done.exists())
                          {
#ifdef DEBUG_LEGION
                            assert(refinement_event.exists());
#endif
                            refinement_done = refinement_event;
                          }
                        }
                      }
                    }
                    // If we still have valid fields then we need a refinement
                    // for them to bring them up to date
                    if (!!overlap)
                    {
                      EquivalenceSet *child = 
                        add_pending_refinement(expr, overlap, node, source);
#ifdef DEBUG_LEGION
                      // These should be the same equivalence set
                      assert(child == finder->second);
#endif
                      pending_to_traverse.insert(child, overlap);
                      to_traverse_exprs[child] = expr;
                      // If this is a pending refinement then we'll need to
                      // wait for it before traversing farther
                      if (!refinement_done.exists())
                      {
#ifdef DEBUG_LEGION
                        assert(refinement_event.exists());
#endif
                        refinement_done = refinement_event;
                      }
                      // Record that we refined these fields
                      overlap.clear();
                    }
                  }
                }
              }
              // If we get here and we still haven't done a disjoint
              // refinement then we can no longer allow it to continue
              if (!!overlap)
              {
                finalize_disjoint_refinement(it->first, overlap);
                it.filter(overlap);
                if (!it->second)
                  to_delete.push_back(it->first);
              }
              // If we handled our disjoint overlap fields then we're done
              if (!disjoint_overlap)
                break;
            }
            if (!to_delete.empty())
            {
              for (std::vector<DisjointPartitionRefinement*>::const_iterator 
                    it = to_delete.begin(); it != to_delete.end(); it++)
              {
                disjoint_partition_refinements.erase(*it);
                delete (*it);
              }
              disjoint_partition_refinements.tighten_valid_mask();
            }
          }
        }
        // Next handle any fields which are refined or pending refined
        if (!!ray_mask)
        {
          FieldMaskSet<IndexSpaceExpression> intersections;
          if (!pending_refinements.empty() && 
              !(ray_mask * pending_refinements.get_valid_mask()))
          {
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  pending_refinements.begin(); it != 
                  pending_refinements.end(); it++)
            {
              const FieldMask overlap = it->second & ray_mask;
              if (!overlap)
                continue;
              // Next check for expression overlap
              IndexSpaceExpression *expr_overlap = 
                forest->intersect_index_spaces(expr, it->first->set_expr);
              if (expr_overlap->is_empty())
                continue;
              pending_to_traverse.insert(it->first, overlap);
              to_traverse_exprs[it->first] = expr_overlap;
              intersections.insert(expr_overlap, overlap);
              // If this is a pending refinement then we'll need to
              // wait for it before traversing farther
              if (!refinement_done.exists())
              {
#ifdef DEBUG_LEGION
                assert(refinement_event.exists());
#endif
                refinement_done = refinement_event;
              }
            }
          }
          if (!subsets.empty() && 
              !(ray_mask * subsets.get_valid_mask()))
          {
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  subsets.begin(); it != subsets.end(); it++)
            {
              const FieldMask overlap = it->second & ray_mask;
              if (!overlap)
                continue;
              // Next check for expression overlap
              IndexSpaceExpression *expr_overlap = 
                forest->intersect_index_spaces(expr, it->first->set_expr);
              if (expr_overlap->is_empty())
                continue;
              to_traverse.insert(it->first, overlap);
              to_traverse_exprs[it->first] = expr_overlap;
              intersections.insert(expr_overlap, overlap);
            }
          }
          // For all our intersections, compute the remainders after the
          // overlap and if they exist then perform refinements for them
          if (!intersections.empty())
          {
            if (intersections.size() > 1)
            {
              // Sort these into field mask sets
              LegionList<FieldSet<IndexSpaceExpression*> >::aligned field_sets;
              intersections.compute_field_sets(FieldMask(), field_sets);
              for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
                    iterator it = field_sets.begin(); 
                    it != field_sets.end(); it++)
              {
                IndexSpaceExpression *diff = forest->subtract_index_spaces(expr,
                    forest->union_index_spaces(it->elements));
                if (!diff->is_empty())
                {
                  EquivalenceSet *child = 
                    add_pending_refinement(diff, it->set_mask, NULL, source);
                  pending_to_traverse.insert(child, it->set_mask);
                  to_traverse_exprs[child] = diff;
                  // If this is a pending refinement then we'll need to
                  // wait for it before traversing farther
                  if (!refinement_done.exists())
                  {
#ifdef DEBUG_LEGION
                    assert(refinement_event.exists());
#endif
                    refinement_done = refinement_event;
                  }
                  // We need to subtract this off any unrefined remainders
                  // or add the difference of it with the original set
                  // to the set of unrefined remainders
                  filter_unrefined_remainders(it->set_mask, diff);
                  if (!!it->set_mask)
                  {
                    IndexSpaceExpression *remainder = 
                      forest->subtract_index_spaces(set_expr, diff);
                    if (!remainder->is_empty())
                    {
#ifdef DEBUG_LEGION
                      assert(disjoint_partition_refinements.get_valid_mask() * 
                              it->set_mask);
                      assert(unrefined_remainders.get_valid_mask() * 
                              it->set_mask);
#endif
                      unrefined_remainders.insert(remainder, it->set_mask);
                    }
                  }
                }
              }
            }
            else
            {
              // Easy case with just one intersection
              FieldMaskSet<IndexSpaceExpression>::const_iterator
                first = intersections.begin();
              IndexSpaceExpression *diff = 
                forest->subtract_index_spaces(expr, first->first);
              if (!diff->is_empty())
              {
                EquivalenceSet *child = 
                  add_pending_refinement(diff, first->second, NULL, source);
                pending_to_traverse.insert(child, first->second);
                to_traverse_exprs[child] = diff;
                // If this is a pending refinement then we'll need to
                // wait for it before traversing farther
                if (!refinement_done.exists())
                {
#ifdef DEBUG_LEGION
                  assert(refinement_event.exists());
#endif
                  refinement_done = refinement_event;
                }
                // Subtract from any unrefined remainders
                FieldMask to_filter = first->second;
                filter_unrefined_remainders(to_filter, diff);
                if (!!to_filter)
                {
                  IndexSpaceExpression *remainder = 
                    forest->subtract_index_spaces(set_expr, diff);
                  if (!remainder->is_empty())
                  {
#ifdef DEBUG_LEGION
                    assert(disjoint_partition_refinements.get_valid_mask() * 
                            to_filter);
                    assert(unrefined_remainders.get_valid_mask() * 
                            to_filter);
#endif
                    unrefined_remainders.insert(remainder, to_filter);
                  }
                }
              }
            }
            // These fields are all remove from the ray mask
            // since they have now been handled
            ray_mask -= intersections.get_valid_mask();
          }
        }
        // If we still have fields left, see if we need a refinement
        if (!!ray_mask && (set_expr->expr_id != expr->expr_id) &&
            (expr->get_volume() < set_expr->get_volume()))
        {
          IndexSpaceExpression *diff =
            forest->subtract_index_spaces(set_expr, expr);
#ifdef DEBUG_LEGION
          assert(!diff->is_empty());
#endif
          // We're doing a refinement for the first time, see if 
          // we can make this a disjoint partition refeinement
          if ((index_space_node != NULL) && handle.exists())
          {
            FieldMask disjoint_mask = ray_mask;
            // We can't start a new disjoint mask for anything that
            // has already been partially refined
            if (!unrefined_remainders.empty())
              disjoint_mask -= unrefined_remainders.get_valid_mask();
            if (!!disjoint_mask)
            {
              IndexSpaceNode *node = runtime->forest->get_node(handle);
              // We can start a disjoint complete partition if there
              // is exactly one partition between the parent index
              // space for the equivalence class and the child index
              // space for the subset and the partition is disjoint
              if ((node->parent != NULL) && 
                  (node->parent->parent == index_space_node) &&
                  node->parent->is_disjoint())
              {
                DisjointPartitionRefinement *dis = 
                  new DisjointPartitionRefinement(node->parent);
                EquivalenceSet *child = 
                  add_pending_refinement(expr, disjoint_mask, node, source);
                pending_to_traverse.insert(child, disjoint_mask);
                to_traverse_exprs[child] = expr;
                // If this is a pending refinement then we'll need to
                // wait for it before traversing farther
                if (!refinement_done.exists())
                {
#ifdef DEBUG_LEGION
                  assert(refinement_event.exists());
#endif
                  refinement_done = refinement_event;
                }
                // Save this for the future
                dis->children[node] = child; 
#ifdef DEBUG_LEGION
                assert(disjoint_mask * unrefined_remainders.get_valid_mask());
#endif
                disjoint_partition_refinements.insert(dis, disjoint_mask);
                ray_mask -= disjoint_mask;
              }
            }
          }
          // If we didn't make a disjoint partition refeinement
          // then we need to do the normal kind of refinement
          if (!!ray_mask)
          {
            // Time to refine this since we only need a subset of it
            EquivalenceSet *child = 
              add_pending_refinement(expr, ray_mask, NULL, source); 
            pending_to_traverse.insert(child, ray_mask);
            to_traverse_exprs[child] = expr;
            // If this is a pending refinement then we'll need to
            // wait for it before traversing farther
            if (!refinement_done.exists())
            {
#ifdef DEBUG_LEGION
              assert(refinement_event.exists());
#endif
              refinement_done = refinement_event;
            }
            // Subtract from any unrefined remainders
            filter_unrefined_remainders(ray_mask, expr);
            if (!!ray_mask)
            {
#ifdef DEBUG_LEGION
              assert(disjoint_partition_refinements.get_valid_mask() * 
                      ray_mask);
              assert(unrefined_remainders.get_valid_mask() * 
                      ray_mask);
#endif
              unrefined_remainders.insert(diff, ray_mask);
              ray_mask.clear();
            }
          }
        }
        // Otherwise we can fall through because this means the
        // expressions are equivalent
      }
      // We've done our traversal, so if we had a deferral even we can 
      // trigger it now to signal to the next user that they can start
      if (deferral_event.exists())
        Runtime::trigger_event(deferral_event);
      std::set<RtEvent> done_events;
      // Any fields which are still valid should be recorded
      if (!!ray_mask)
      {
        // Not local so we need to send a message
        if (source != runtime->address_space)
        {
          // If there's nothing to do after this we can use
          // the trace_done event directly
          if (to_traverse.empty() && pending_to_traverse.empty())
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(ray_mask);
              rez.serialize(target);
              rez.serialize(trace_done);
            }
            runtime->send_equivalence_set_ray_trace_response(source, rez);
            return;
          }
          else
          {
            RtUserEvent done = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(ray_mask);
              rez.serialize(target);
              rez.serialize(done);
            }
            runtime->send_equivalence_set_ray_trace_response(source, rez);
            done_events.insert(done);
          }
        }
        else
          target->record_equivalence_set(this, ray_mask);
      }
      // Traverse anything we can now before we have to wait
      if (!to_traverse.empty())
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator
            finder = to_traverse_exprs.find(it->first);
#ifdef DEBUG_LEGION
          assert(finder != to_traverse_exprs.end());
#endif
          const IndexSpace subset_handle = 
            (handle.exists() && 
              (finder->second->get_volume() == expr->get_volume())) ? handle :
                IndexSpace::NO_SPACE;
          it->first->ray_trace_equivalence_sets(target, finder->second, 
              it->second, subset_handle, source, done);
          done_events.insert(done);
        }
        // Clear these since we are done doing them
        to_traverse.clear();
      }
      // Get the actual equivalence sets for any refinements we needed to
      // wait for because they weren't ready earlier
      if (!pending_to_traverse.empty())
      {
        // If we have a refinement to do then we need to wait for that
        // to be done before we continue our traversal
        if (refinement_done.exists() && !refinement_done.has_triggered())
        {
          // Defer this until the refinements are done
          FieldMaskSet<EquivalenceSet> *copy_traverse = 
            new FieldMaskSet<EquivalenceSet>();
          copy_traverse->swap(pending_to_traverse);
          std::map<EquivalenceSet*,IndexSpaceExpression*> *copy_exprs = 
            new std::map<EquivalenceSet*,IndexSpaceExpression*>();
          copy_exprs->swap(to_traverse_exprs);
          const RtUserEvent done = Runtime::create_rt_user_event();
          DeferRayTraceFinishArgs args(target, source, copy_traverse,
              copy_exprs, expr->get_volume(), handle, done);
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, refinement_done);
          done_events.insert(done);
        }
        else
        {
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                pending_to_traverse.begin(); it != 
                pending_to_traverse.end(); it++)
          {
            RtUserEvent done = Runtime::create_rt_user_event();
            std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator
              finder = to_traverse_exprs.find(it->first);
#ifdef DEBUG_LEGION
            assert(finder != to_traverse_exprs.end());
#endif
            const IndexSpace subset_handle = 
              (handle.exists() && 
                (finder->second->get_volume() == expr->get_volume())) ? handle :
                  IndexSpace::NO_SPACE;
            it->first->ray_trace_equivalence_sets(target, finder->second, 
                it->second, subset_handle, source, done);
            done_events.insert(done);
          }
        }
      }
      if (!done_events.empty())
        Runtime::trigger_event(trace_done, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(trace_done);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::pack_state(Serializer &rez,
                                    const FieldMask &pack_mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // Pack the valid instances
      rez.serialize<size_t>(valid_instances.size());
      if (!valid_instances.empty())
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          const FieldMask overlap = it->second & pack_mask;
          if (!!overlap)
          {
            rez.serialize(it->first->did);
            rez.serialize(overlap);
          }
          else
            rez.serialize<DistributedID>(0);
        }
      }
      // Pack the reduction instances
      if (!!reduction_fields)
      {
        const FieldMask reduc_mask = reduction_fields & pack_mask;
        if (!!reduc_mask)
        {
          rez.serialize<size_t>(reduc_mask.pop_count());
          int fidx = reduc_mask.find_first_set();
          while (fidx >= 0)
          {
            rez.serialize<unsigned>(fidx);
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != reduction_instances.end());
#endif
            rez.serialize<size_t>(finder->second.size());
            for (std::vector<ReductionView*>::const_iterator it = 
                  finder->second.begin(); it != finder->second.end(); it++)
              rez.serialize((*it)->did);
            fidx = reduc_mask.find_next_set(fidx+1);
          }
        }
        else
          rez.serialize<size_t>(0);
      }
      else
        rez.serialize<size_t>(0);
      // Pack the restricted instances
      if (!!restricted_fields)
      {
        const FieldMask restr_mask = restricted_fields & pack_mask;
        if (!!restr_mask)
        {
          rez.serialize<size_t>(restricted_instances.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                restricted_instances.begin(); it != 
                restricted_instances.end(); it++)
          {
            const FieldMask overlap = pack_mask & it->second;
            if (!!overlap)
            {
              rez.serialize(it->first->did);
              rez.serialize(overlap);
            }
            else
              rez.serialize<DistributedID>(0);
          }
        }
        else
          rez.serialize<size_t>(0);
      }
      else
        rez.serialize<size_t>(0);
      // Pack the version numbers
      rez.serialize<size_t>(version_numbers.size());
      if (!version_numbers.empty())
      {
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              version_numbers.begin(); it != version_numbers.end(); it++)
        {
          const FieldMask overlap = pack_mask & it->second;
          if (!!overlap)
          {
            rez.serialize(it->first);
            rez.serialize(overlap);
          }
          else
            rez.serialize<VersionID>(0);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::unpack_state(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RtUserEvent done_event;
      derez.deserialize(done_event);
      // Do a quick test to see if we're still the owner, if not
      // then we can just forward this on immediately
      {
        AutoLock eq(eq_lock,1,false/*exlcusive*/);
        if (!is_logical_owner())
        {
          Serializer rez;
          // No RezCheck because of forwarding
          rez.serialize(did);
          rez.serialize(done_event);
          // Just move the bytes over to the serializer and return
          const size_t bytes = derez.get_remaining_bytes();
          rez.serialize(derez.get_current_pointer(), bytes);
          runtime->send_equivalence_set_remote_refinement(
                                  logical_owner_space, rez);
          // Keep the deserializer happy
          derez.advance_pointer(bytes);
          return;
        }
      }
      // Keep track of ready events
      std::set<RtEvent> ready_events;
      // Unpack into local data structures which we'll update later
      FieldMaskSet<LogicalView> new_valid;
      size_t num_valid_insts;
      derez.deserialize(num_valid_insts);
      for (unsigned idx = 0; idx < num_valid_insts; idx++)
      {
        DistributedID valid_did;
        derez.deserialize(valid_did);
        if (valid_did == 0)
          continue;
        RtEvent ready;
        LogicalView *view = 
          runtime->find_or_request_logical_view(valid_did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        new_valid.insert(view, mask);
      }
      size_t num_reduc_fields;
      derez.deserialize(num_reduc_fields);
      std::map<unsigned,std::vector<ReductionView*> > new_reductions;
      for (unsigned idx1 = 0; idx1 < num_reduc_fields; idx1++)
      {
        unsigned fidx;
        derez.deserialize(fidx);
        std::vector<ReductionView*> &new_views = new_reductions[fidx];
        size_t num_reduc_insts;
        derez.deserialize(num_reduc_insts);
        new_views.resize(num_reduc_insts);
        for (unsigned idx2 = 0; idx2 < num_reduc_insts; idx2++)
        {
          DistributedID reduc_did;
          derez.deserialize(reduc_did);
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(reduc_did, ready);
          new_views[idx2] = static_cast<ReductionView*>(view);
          if (ready.exists())
            ready_events.insert(ready);
        }
      }
      size_t num_restrict_insts;
      derez.deserialize(num_restrict_insts);
      FieldMaskSet<InstanceView> new_restrictions;
      if (num_restrict_insts > 0)
      {
        for (unsigned idx = 0; idx < num_restrict_insts; idx++)
        {
          DistributedID valid_did;
          derez.deserialize(valid_did);
          if (valid_did == 0)
            continue;
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(valid_did, ready);
          if (ready.exists())
            ready_events.insert(ready);
          InstanceView *inst_view = static_cast<InstanceView*>(view);
          FieldMask mask;
          derez.deserialize(mask);
          new_restrictions.insert(inst_view, mask);
        }
      }
      size_t num_versions;
      derez.deserialize(num_versions);
      LegionMap<VersionID,FieldMask>::aligned new_versions;
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        if (vid == 0)
          continue;
        derez.deserialize(new_versions[vid]);
      }
      // If we have events to wait for then we need to defer this
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
          // Defer the merge or forward until the views are ready
          FieldMaskSet<LogicalView> *view_copy = 
            new FieldMaskSet<LogicalView>();
          view_copy->swap(new_valid);
          std::map<unsigned,std::vector<ReductionView*> > *reduc_copy = 
            new std::map<unsigned,std::vector<ReductionView*> >();
          reduc_copy->swap(new_reductions);
          FieldMaskSet<InstanceView> *restrict_copy = 
            new FieldMaskSet<InstanceView>();
          restrict_copy->swap(new_restrictions);
          LegionMap<VersionID,FieldMask>::aligned *version_copy = 
            new LegionMap<VersionID,FieldMask>::aligned();
          version_copy->swap(new_versions);
          DeferMergeOrForwardArgs args(this, view_copy, reduc_copy,
              restrict_copy, version_copy, done_event);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, wait_on);
          return;
        }
        // Otherwise fall through to do the merge or forward now
      }
      // Either merge or forward the update
      merge_or_forward(done_event, new_valid, new_reductions,
                       new_restrictions, new_versions);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::merge_or_forward(const RtUserEvent done_event,
          const FieldMaskSet<LogicalView> &new_views,
          const std::map<unsigned,std::vector<ReductionView*> > &new_reductions,
          const FieldMaskSet<InstanceView> &new_restrictions,
          const LegionMap<VersionID,FieldMask>::aligned &new_versions)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      if (is_logical_owner())
      {
        // We're the owner so we can do the merge
        LocalReferenceMutator mutator;
        for (FieldMaskSet<LogicalView>::const_iterator it =
              new_views.begin(); it != new_views.end(); it++)
          if (valid_instances.insert(it->first, it->second))
            it->first->add_nested_valid_ref(did, &mutator);
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = new_reductions.begin(); rit != new_reductions.end(); rit++)
        {
          reduction_fields.set_bit(rit->first);
          std::vector<ReductionView*> &reduc_insts = 
            reduction_instances[rit->first];
#ifdef DEBUG_LEGION
          assert(reduc_insts.empty());
#endif
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            reduc_insts.push_back(*it);
            (*it)->add_nested_valid_ref(did, &mutator);
          }
        }
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              new_restrictions.begin(); it != new_restrictions.end(); it++)
        {
          restricted_fields |= it->second;
          if (restricted_instances.insert(it->first, it->second))
            it->first->add_nested_valid_ref(did, &mutator);
        }
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it =
              new_versions.begin(); it != new_versions.end(); it++)
        {
          LegionMap<VersionID,FieldMask>::aligned::iterator finder = 
            version_numbers.find(it->first);
          if (finder == version_numbers.end())
            version_numbers.insert(*it);
          else
            finder->second |= it->second;
        }
        Runtime::trigger_event(done_event, mutator.get_done_event());
      }
      else
      {
        // We're not the owner so we need to forward this on
        Serializer rez;
        // No RezCheck in case of forwarding
        rez.serialize(did);
        rez.serialize(done_event);
        rez.serialize(new_views.size());
        for (FieldMaskSet<LogicalView>::const_iterator it =
              new_views.begin(); it != new_views.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
        rez.serialize<size_t>(new_reductions.size());
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = new_reductions.begin(); rit != new_reductions.end(); rit++)
        {
          rez.serialize(rit->first);
          rez.serialize(rit->second.size());
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
            rez.serialize((*it)->did);
        }
        rez.serialize(new_restrictions.size());
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              new_restrictions.begin(); it != new_restrictions.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
        rez.serialize(new_versions.size());
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it =
              new_versions.begin(); it != new_versions.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        runtime->send_equivalence_set_remote_refinement(
                                logical_owner_space, rez);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::pack_migration(Serializer &rez, RtEvent done_migration)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(pending_refinements.empty());
#endif
      std::map<LogicalView*,unsigned> *late_references = NULL;
      // Pack the valid instances
      rez.serialize<size_t>(valid_instances.size());
      if (!valid_instances.empty())
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          if (late_references == NULL)
            late_references = new std::map<LogicalView*,unsigned>();
          (*late_references)[it->first] = 1;
        }
        valid_instances.clear();
      }
      // Pack the reduction instances
      rez.serialize<size_t>(reduction_instances.size());
      if (!reduction_instances.empty())
      {
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = reduction_instances.begin(); 
              rit != reduction_instances.end(); rit++)
        {
          rez.serialize(rit->first);
          rez.serialize<size_t>(rit->second.size());
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize((*it)->did);
            if (late_references == NULL)
              late_references = new std::map<LogicalView*,unsigned>();
            (*late_references)[*it] = 1;
          }
        }
        reduction_instances.clear();
        reduction_fields.clear();
      }
      // Pack the restricted instances
      rez.serialize<size_t>(restricted_instances.size());  
      if (!restricted_instances.empty())
      {
        rez.serialize(restricted_fields);
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              restricted_instances.begin(); it != 
              restricted_instances.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          if (late_references == NULL)
            late_references = new std::map<LogicalView*,unsigned>();
          std::map<LogicalView*,unsigned>::iterator finder = 
            late_references->find(it->first);
          if (finder == late_references->end())
            (*late_references)[it->first] = 1;
          else
            finder->second += 1;
        }
        restricted_instances.clear();
        restricted_fields.clear();
      }
      // Pack the version numbers
      rez.serialize<size_t>(version_numbers.size());
      if (!version_numbers.empty())
      {
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              version_numbers.begin(); it != version_numbers.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        version_numbers.clear();
      }
      // Pack events to wait on to make sure all remote guards are done
      // before we become the new logical owner
      rez.serialize<size_t>(update_guards.size());
      if (!update_guards.empty())
      {
        for (FieldMaskSet<CopyFillAggregator>::const_iterator it =
              update_guards.begin(); it != update_guards.end(); it++)
          rez.serialize(it->first->effects_applied);
      }
      // Pack subsets
      rez.serialize<size_t>(subsets.size());
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            subsets.begin(); it != subsets.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      // No need to clear subsets since we can still maintain a copy of it
      // Pack remote subsets
      rez.serialize<size_t>(remote_subsets.size());
      if (!remote_subsets.empty())
      {
        for (std::set<AddressSpaceID>::const_iterator it = 
              remote_subsets.begin(); it != remote_subsets.end(); it++)
          rez.serialize(*it);
        remote_subsets.clear();
      }
      // Pack unrefined remainders
      rez.serialize<size_t>(unrefined_remainders.size());
      if (!unrefined_remainders.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
              unrefined_remainders.begin(); it != 
              unrefined_remainders.end(); it++)
        {
          it->first->pack_expression(rez, logical_owner_space);
          rez.serialize(it->second);
        }
        unrefined_remainders.clear();
      }
      // Pack disjoint partition refinements
      rez.serialize<size_t>(disjoint_partition_refinements.size());
      if (!disjoint_partition_refinements.empty())
      {
        for (FieldMaskSet<DisjointPartitionRefinement>::const_iterator it =
              disjoint_partition_refinements.begin(); it !=
              disjoint_partition_refinements.end(); it++)
        {
          rez.serialize(it->first->partition->handle);
          rez.serialize<size_t>(it->first->children.size());
          for (std::map<IndexSpaceNode*,EquivalenceSet*>::const_iterator cit =
                it->first->children.begin(); cit != 
                it->first->children.end(); cit++)
          {
            rez.serialize(cit->first->handle);
            rez.serialize(cit->second->did);
          }
          rez.serialize(it->second);
          delete it->first;
        }
        disjoint_partition_refinements.clear();
      }
      if (late_references != NULL)
      {
        // Launch a task to remove the references once the migration is done
        RemoteRefTaskArgs args(this->did, RtUserEvent::NO_RT_USER_EVENT,
                               false/*add*/, late_references);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY, 
                                         done_migration); 
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::unpack_migration(Deserializer &derez, 
                               ReferenceMutator &mutator, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(valid_instances.empty());
      assert(reduction_instances.empty());
      assert(restricted_instances.empty());
      assert(version_numbers.empty());
      assert(update_guards.empty());
      assert(pending_refinements.empty());
      assert(remote_subsets.empty());
      assert(unrefined_remainders.empty());
      assert(disjoint_partition_refinements.empty());
#endif
      // We can unpack these data structures because we know that no one
      // else touching them since we're not the owner yet
      std::set<RtEvent> deferred_reference_events;
      std::map<LogicalView*,unsigned> deferred_references;
      size_t num_valid_insts;
      derez.deserialize(num_valid_insts);
      for (unsigned idx = 0; idx < num_valid_insts; idx++)
      {
        DistributedID valid_did;
        derez.deserialize(valid_did);
        RtEvent ready;
        LogicalView *view = 
          runtime->find_or_request_logical_view(valid_did, ready);
        FieldMask mask;
        derez.deserialize(mask);
        valid_instances.insert(view, mask);
        if (ready.exists() && !ready.has_triggered())
        {
          deferred_reference_events.insert(ready);
          deferred_references[view] = 1;
        }
        else
          view->add_nested_valid_ref(did, &mutator);
      }
      size_t num_reduc_fields;
      derez.deserialize(num_reduc_fields);
      for (unsigned idx1 = 0; idx1 < num_reduc_fields; idx1++)
      {
        unsigned fidx;
        derez.deserialize(fidx);
        reduction_fields.set_bit(fidx);
        size_t num_reduc_insts;
        derez.deserialize(num_reduc_insts);
        std::vector<ReductionView*> &reduc_views = reduction_instances[fidx];
        for (unsigned idx2 = 0; idx2 < num_reduc_insts; idx2++)
        {
          DistributedID reduc_did;
          derez.deserialize(reduc_did);
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(reduc_did, ready);
          ReductionView *reduc_view = static_cast<ReductionView*>(view);
          reduc_views.push_back(reduc_view);
          if (ready.exists() && !ready.has_triggered())
          {
            deferred_reference_events.insert(ready);
            std::map<LogicalView*,unsigned>::iterator finder = 
              deferred_references.find(view);
            if (finder == deferred_references.end())
              deferred_references[view] = 1;
            else
              finder->second++;
          }
          else
            view->add_nested_valid_ref(did, &mutator);
        }
      }
      size_t num_restrict_insts;
      derez.deserialize(num_restrict_insts);
      if (num_restrict_insts > 0)
      {
        derez.deserialize(restricted_fields);
        for (unsigned idx = 0; idx < num_restrict_insts; idx++)
        {
          DistributedID valid_did;
          derez.deserialize(valid_did);
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(valid_did, ready);
          InstanceView *inst_view = static_cast<InstanceView*>(view);
          FieldMask mask;
          derez.deserialize(mask);
          restricted_instances.insert(inst_view, mask);
          if (ready.exists() && !ready.has_triggered())
          {
            deferred_reference_events.insert(ready);
            std::map<LogicalView*,unsigned>::iterator finder = 
              deferred_references.find(view);
            if (finder == deferred_references.end())
              deferred_references[view] = 1;
            else
              finder->second++;
          }
          else
            view->add_nested_valid_ref(did, &mutator);
        }
      }
      size_t num_versions;
      derez.deserialize(num_versions);
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        derez.deserialize(version_numbers[vid]);
      }
      size_t num_guard_events;
      derez.deserialize(num_guard_events);
      std::set<RtEvent> guard_preconditions;
      for (unsigned idx = 0; idx < num_guard_events; idx++)
      {
        RtEvent guard_event;
        derez.deserialize(guard_event);
        guard_preconditions.insert(guard_event);
      }
      size_t num_subsets;
      derez.deserialize(num_subsets);
      std::vector<EquivalenceSet*> need_ref;
      for (unsigned idx = 0; idx < num_subsets; idx++)
      {
        DistributedID subset_did;
        derez.deserialize(subset_did);
        RtEvent ready;
        EquivalenceSet *subset = 
          runtime->find_or_request_equivalence_set(subset_did, ready);
        if (ready.exists())
          deferred_reference_events.insert(ready);
        FieldMask subset_mask;
        derez.deserialize(subset_mask);
        if (subsets.insert(subset, subset_mask))
          need_ref.push_back(subset);
      }
      size_t num_remote_subsets;
      derez.deserialize(num_remote_subsets);
      for (unsigned idx = 0; idx < num_remote_subsets; idx++)
      {
        AddressSpaceID remote;
        derez.deserialize(remote);
        remote_subsets.insert(remote);
      }
      size_t num_unrefined_remainders;
      derez.deserialize(num_unrefined_remainders);
      for (unsigned idx = 0; idx < num_unrefined_remainders; idx++)
      {
        IndexSpaceExpression *expr = 
          IndexSpaceExpression::unpack_expression(derez,runtime->forest,source);
        FieldMask mask;
        derez.deserialize(mask);
        unrefined_remainders.insert(expr, mask);
      }
      size_t num_disjoint_refinements;
      derez.deserialize(num_disjoint_refinements);
      for (unsigned idx1 = 0; idx1 < num_disjoint_refinements; idx1++)
      {
        IndexPartition handle;
        derez.deserialize(handle);
        IndexPartNode *part = runtime->forest->get_node(handle);
        DisjointPartitionRefinement *dis = new DisjointPartitionRefinement(part);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx2 = 0; idx2 < num_children; idx2++)
        {
          IndexSpace child;
          derez.deserialize(child);
          IndexSpaceNode *node = runtime->forest->get_node(child);
          DistributedID child_did;
          derez.deserialize(child_did);
          RtEvent ready;
          dis->children[node] = 
            runtime->find_or_request_equivalence_set(child_did, ready);
          // No need to worry about the ready event because we know this
          // equivalence set is in our subsets data structure so we'll
          // wait for it through there instead
        }
        FieldMask mask;
        derez.deserialize(mask);
        disjoint_partition_refinements.insert(dis, mask);
      }
      // Make all the events we'll need to wait on
      RtEvent ready_for_references, guards_done;
      if (!deferred_reference_events.empty())
        ready_for_references = Runtime::merge_events(deferred_reference_events);
      if (!guard_preconditions.empty())
        guards_done = Runtime::merge_events(guard_preconditions);
      if (ready_for_references.exists() && 
          !ready_for_references.has_triggered())
        ready_for_references.wait();
      // Add our references
      for (std::map<LogicalView*,unsigned>::const_iterator it = 
            deferred_references.begin(); it != 
            deferred_references.end(); it++)
        it->first->add_nested_valid_ref(did, &mutator, it->second);
      for (std::vector<EquivalenceSet*>::const_iterator it = 
            need_ref.begin(); it != need_ref.end(); it++)
        (*it)->add_nested_resource_ref(did);
      // Wait for all the guards to be done before we mark that
      // we are now the new owner
      if (guards_done.exists() && !guards_done.has_triggered())
      {
        // Defer the call to make this the owner until the event triggers
        DeferMakeOwnerArgs args(this);
        RtEvent done = runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, guards_done);
        mutator.record_reference_mutation_effect(done);
      }
      else
        make_owner();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::make_owner(void)
    //--------------------------------------------------------------------------
    {
      // Now we can mark that we are the logical owner
      AutoLock eq(eq_lock);
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      logical_owner_space = local_space;
      // If we were waiting for a valid copy of the subsets we now have it
      if (eq_state == PENDING_VALID_STATE)
      {
#ifdef DEBUG_LEGION
        assert(transition_event.exists()); 
#endif
        // We can trigger this transition event now that we have a valid
        // copy of the subsets (we are the logical owner)
        Runtime::trigger_event(transition_event);
        transition_event = RtUserEvent::NO_RT_USER_EVENT;
      }
      eq_state = MAPPING_STATE;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_owner(const AddressSpaceID new_logical_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // If we are the owner then we know this update is stale so ignore it
      if (!is_logical_owner())
        logical_owner_space = new_logical_owner;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::initialize_set(const RegionUsage &usage,
                                        const FieldMask &user_mask,
                                        const bool restricted,
                                        const InstanceSet &sources,
                                const std::vector<InstanceView*> &corresponding,
                                        std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(sources.size() == corresponding.size());
#endif
      WrapperReferenceMutator mutator(applied_events);
      AutoLock eq(eq_lock);
      if (IS_REDUCE(usage))
      {
#ifdef DEBUG_LEGION
        // Reduction-only should always be restricted for now
        // Could change if we started issuing reduction close
        // operations at the end of a context
        assert(restricted);
#endif
        // Since these are restricted, we'll make these the actual
        // target logical instances and record them as restricted
        // instead of recording them as reduction instances
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const FieldMask &view_mask = sources[idx].get_valid_fields();
          InstanceView *view = corresponding[idx];
          FieldMaskSet<LogicalView>::iterator finder = 
            valid_instances.find(view);
          if (finder == valid_instances.end())
          {
            valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did, &mutator);
          }
          else
            finder.merge(view_mask);
          // Always restrict reduction-only users since we know the data
          // is going to need to be flushed anyway
          FieldMaskSet<InstanceView>::iterator restricted_finder = 
            restricted_instances.find(view);
          if (restricted_finder == restricted_instances.end())
          {
            restricted_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did, &mutator);
          }
          else
            restricted_finder.merge(view_mask);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const FieldMask &view_mask = sources[idx].get_valid_fields();
          InstanceView *view = corresponding[idx];
#ifdef DEBUG_LEGION
          assert(!view->is_reduction_view());
#endif
          FieldMaskSet<LogicalView>::iterator finder = 
            valid_instances.find(view);
          if (finder == valid_instances.end())
          {
            valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did, &mutator);
          }
          else
            finder.merge(view_mask);
          // If this is restricted then record it
          if (restricted)
          {
            FieldMaskSet<InstanceView>::iterator restricted_finder = 
              restricted_instances.find(view);
            if (restricted_finder == restricted_instances.end())
            {
              restricted_instances.insert(view, view_mask);
              view->add_nested_valid_ref(did, &mutator);
            }
            else
              restricted_finder.merge(view_mask);
          }
        }
      }
      // Update any restricted fields 
      if (restricted)
        restricted_fields |= user_mask;
      // Set the version numbers too
      version_numbers[init_version] |= user_mask;
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::find_valid_instances(RemoteEqTracker &remote_tracker,
                                              FieldMaskSet<LogicalView> &insts,
                                              const FieldMask &user_mask) const
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock,1,false/*exclusive*/);
      if (!is_logical_owner())
      {
        remote_tracker.record_remote(const_cast<EquivalenceSet*>(this), 
                                     user_mask, logical_owner_space);
        return false;
      }
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        if (!it->first->is_instance_view())
          continue;
        const FieldMask overlap = it->second & user_mask;
        if (!overlap)
          continue;
        FieldMaskSet<LogicalView>::iterator finder = insts.find(it->first);
        if (finder == insts.end())
          insts.insert(it->first, it->second);
        else
          finder.merge(it->second);
      }
      return has_restrictions(user_mask);
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::find_reduction_instances(
                       RemoteEqTracker &remote_tracker,
                       FieldMaskSet<ReductionView> &insts, ReductionOpID redop,
                       const FieldMask &user_mask) const
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock,1,false/*exclusive*/);
      if (!is_logical_owner())
      {
        remote_tracker.record_remote(const_cast<EquivalenceSet*>(this), 
                                     user_mask, logical_owner_space);
        return false;
      }
      // Iterate over all the fields
      int fidx = user_mask.find_first_set();
      while (fidx >= 0)
      {
        std::map<unsigned,std::vector<ReductionView*> >::const_iterator 
          current = reduction_instances.find(fidx);
        if (current != reduction_instances.end())
        {
          FieldMask local_mask;
          local_mask.set_bit(fidx);
          for (std::vector<ReductionView*>::const_reverse_iterator it = 
                current->second.rbegin(); it != current->second.rend(); it++)
          {
            ReductionManager *manager = 
              (*it)->get_manager()->as_reduction_manager();
            if (manager->redop != redop)
              break;
            FieldMaskSet<ReductionView>::iterator finder = insts.find(*it);
            if (finder == insts.end())
              insts.insert(*it, local_mask);
            else
              finder.merge(local_mask);
          }
        }
        fidx = user_mask.find_next_set(fidx+1);
      }
      return has_restrictions(user_mask);
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::update_set(RemoteEqTracker &remote_tracker,
                                FieldMaskSet<EquivalenceSet> &alt_sets,
                                FieldMask *remove_mask, // can be NULL
                                const AddressSpaceID source,
                                Operation *op, const unsigned index,
                                const RegionUsage &usage, FieldMask user_mask,
                                const InstanceSet &target_instances,
                                const std::vector<InstanceView*> &target_views,
                                std::map<RtEvent,
                                       CopyFillAggregator*> &input_aggregators,
                                CopyFillAggregator *&output_aggregator,
                                std::set<RtEvent> &applied_events,
                                std::set<RtEvent> &guard_events,
                                FieldMask *initialized/*=NULL*/,
                                const bool original_set/*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target_instances.size() == target_views.size());
#endif
      WrapperReferenceMutator mutator(applied_events);
      AutoLock eq(eq_lock);
      if (!original_set)
      {
        FieldMaskSet<EquivalenceSet>::iterator finder = 
          alt_sets.find(this);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          user_mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!user_mask)
            return false;
          finder.merge(user_mask);
        }
        else
          alt_sets.insert(this, user_mask);
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events); 
        if (subsets.empty())
        {
          remote_tracker.record_remote(this, user_mask, logical_owner_space);
          return false;
        }
        else
        {
          const FieldMask non_subset = user_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            remote_tracker.record_remote(this, non_subset, logical_owner_space);
            user_mask -= non_subset;
            if (!user_mask)
              return false;
          }
        }
        // Otherwise we fall through and record our subsets
      }
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(user_mask))
      {
        check_for_unrefined_remainder(eq, user_mask, 
                                      remote_tracker.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & user_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!original_set)
        {
          FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(this);
          if (finder != alt_sets.end())
          {
            finder.filter(to_traverse.get_valid_mask());
            if (!finder->second)
              alt_sets.erase(finder);
          }
        }
        // Update the user mask and the remove_mask if there is one
        user_mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->update_set(remote_tracker, alt_sets, NULL, source,
              op, index, usage, it->second, target_instances, target_views,
              input_aggregators, output_aggregator, applied_events,
              guard_events, initialized, false/*original set*/);
        eq.reacquire();
        // Return if our user mask is empty
        if (!user_mask)
        {
          if (remove_mask != NULL)
            return !!(*remove_mask);
          else
            return false;
        }
      }
      // Check for any uninitialized data
      // Don't report uninitialized warnings for empty equivalence classes
      if ((initialized != NULL) && !set_expr->is_empty())
        *initialized -= (user_mask - valid_instances.get_valid_mask());
      if (output_aggregator != NULL)
        output_aggregator->clear_update_fields();
      if (IS_REDUCE(usage))
      {
        // Reduction-only
        // Record the reduction instances
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          ReductionView *red_view = target_views[idx]->as_reduction_view();
#ifdef DEBUG_LEGION
          assert(red_view->get_redop() == usage.redop);
#endif
          const FieldMask &update_fields = 
            target_instances[idx].get_valid_fields(); 
          int fidx = update_fields.find_first_set();
          while (fidx >= 0)
          {
            std::vector<ReductionView*> &field_views = 
              reduction_instances[fidx];
            red_view->add_nested_valid_ref(did, &mutator); 
            field_views.push_back(red_view);
            fidx = update_fields.find_next_set(fidx+1);
          }
        }
        // Flush any restricted fields
        if (!!restricted_fields)
        {
          const FieldMask reduce_mask = user_mask & restricted_fields;
          if (!!reduce_mask)
            apply_reductions(reduce_mask, output_aggregator,
                RtEvent::NO_RT_EVENT, op, index, true/*track events*/); 
          // No need to record that we applied the reductions, we'll
          // discover that when we collapse the single/multi-reduce state
          reduction_fields |= (user_mask - restricted_fields);
        }
        else
          reduction_fields |= user_mask;
      }
      else if (IS_WRITE(usage) && IS_DISCARD(usage))
      {
        // Write-only
        // Filter any reductions that we no longer need
        const FieldMask reduce_filter = reduction_fields & user_mask;
        if (!!reduce_filter)
          filter_reduction_instances(reduce_filter);
        // Filter any normal instances that will be overwritten
        const FieldMask non_restricted = user_mask - restricted_fields; 
        if (!!non_restricted)
        {
          filter_valid_instances(non_restricted);
          // Record any non-restricted instances
          record_instances(non_restricted, target_instances, 
                           target_views, mutator);
        }
        // Issue copy-out copies for any restricted fields
        if (!!restricted_fields)
        {
          const FieldMask restricted_mask = user_mask & restricted_fields;
          if (!!restricted_mask)
            copy_out(restricted_mask, target_instances,
                     target_views, op, index, output_aggregator);
        }
        // Advance our version numbers
        advance_version_numbers(user_mask);
      }
      else if (IS_READ_ONLY(usage) && !update_guards.empty() && 
                !(user_mask * update_guards.get_valid_mask()))
      {
        // If we're doing read-only mode, get the set of events that
        // we need to wait for before we can do our registration, this 
        // ensures that we serialize read-only operations correctly
        // In order to avoid deadlock we have to make different copy fill
        // aggregators for each of the different fields of prior updates
        FieldMask remainder_mask = user_mask;
        LegionVector<std::pair<CopyFillAggregator*,FieldMask> >::aligned to_add;
        for (FieldMaskSet<CopyFillAggregator>::iterator it = 
              update_guards.begin(); it != update_guards.end(); it++)
        {
          const FieldMask guard_mask = remainder_mask & it->second;
          if (!guard_mask)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          const RtEvent guard_event = it->first->effects_applied;
#else
          const RtEvent guard_event = 
            (remote_tracker.original_source == local_space) ?
            it->first->guard_postcondition :
            it->first->effects_applied;
#endif
          guard_events.insert(guard_event);
          CopyFillAggregator *input_aggregator = NULL;
          // See if we have an input aggregator that we can use now
          std::map<RtEvent,CopyFillAggregator*>::const_iterator finder = 
            input_aggregators.find(guard_event);
          if (finder != input_aggregators.end())
          {
            input_aggregator = finder->second;
            if (input_aggregator != NULL)
              input_aggregator->clear_update_fields();
          }
          // Use this to see if any new updates are recorded
          update_set_internal(input_aggregator, guard_event, op, index,
                              usage, guard_mask, target_instances, 
                              target_views, applied_events);
          // If we did any updates record ourselves as the new guard here
          if ((input_aggregator != NULL) && 
              ((finder == input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
#ifndef NON_AGGRESSIVE_AGGREGATORS
            // We also have to chain effects in this case 
            input_aggregator->record_reference_mutation_effect(
                                it->first->effects_applied);
#endif
            if (finder == input_aggregators.end())
              input_aggregators[guard_event] = input_aggregator;
            // Record this as a guard for later operations
            to_add.resize(to_add.size() + 1);
            std::pair<CopyFillAggregator*,FieldMask> &back = to_add.back();
            const FieldMask &update_mask = 
              input_aggregator->get_update_fields();
            back.first = input_aggregator;
            back.second = update_mask;
            input_aggregator->record_guard_set(this);
            // Remove the current guard since it doesn't matter anymore
            it.filter(update_mask);
          }
          remainder_mask -= guard_mask;
          if (!remainder_mask)
            break;
        }
        if (!to_add.empty())
        {
          for (LegionVector<std::pair<CopyFillAggregator*,FieldMask> >::
                aligned::const_iterator it = to_add.begin(); 
                it != to_add.end(); it++)
          {
#ifdef DEBUG_LEGION
            assert(it->second * refining_fields);
#endif
            update_guards.insert(it->first, it->second);
          }
        }
        // If we have unguarded fields we can easily do thos
        if (!!remainder_mask)
        {
          CopyFillAggregator *input_aggregator = NULL;
          // See if we have an input aggregator that we can use now
          std::map<RtEvent,CopyFillAggregator*>::const_iterator finder = 
            input_aggregators.find(RtEvent::NO_RT_EVENT);
          if (finder != input_aggregators.end())
          {
            input_aggregator = finder->second;
            if (input_aggregator != NULL)
              input_aggregator->clear_update_fields();
          }
          update_set_internal(input_aggregator, RtEvent::NO_RT_EVENT, op, index,
                              usage, remainder_mask, target_instances, 
                              target_views, applied_events);
          // If we made the input aggregator then store it
          if ((input_aggregator != NULL) && 
              ((finder == input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
            input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
#ifdef DEBUG_LEGION
            assert(input_aggregator->get_update_fields() * refining_fields);
#endif
            // Record this as a guard for later operations
            update_guards.insert(input_aggregator, 
                input_aggregator->get_update_fields());
            input_aggregator->record_guard_set(this);
          }
        }
      }
      else
      {
        // Read-write or read-only case
        // Read-only case if there are no guards
        CopyFillAggregator *input_aggregator = NULL;
        // See if we have an input aggregator that we can use now
        std::map<RtEvent,CopyFillAggregator*>::const_iterator finder = 
          input_aggregators.find(RtEvent::NO_RT_EVENT);
        if (finder != input_aggregators.end())
        {
          input_aggregator = finder->second;
          if (input_aggregator != NULL)
            input_aggregator->clear_update_fields();
        }
        update_set_internal(input_aggregator, RtEvent::NO_RT_EVENT, op, index,
            usage, user_mask, target_instances, target_views, applied_events);
        if (IS_WRITE(usage))
        {
          advance_version_numbers(user_mask);
          // Issue copy-out copies for any restricted fields if we wrote stuff
          const FieldMask restricted_mask = restricted_fields & user_mask;
          if (!!restricted_mask)
            copy_out(restricted_mask, target_instances,
                     target_views, op, index, output_aggregator);
        }
        // If we made the input aggregator then store it
        if ((input_aggregator != NULL) && 
            ((finder == input_aggregators.end()) ||
             input_aggregator->has_update_fields()))
        {
          input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
#ifdef DEBUG_LEGION
          assert(input_aggregator->get_update_fields() * refining_fields);
#endif
          // Record this as a guard for later operations
          update_guards.insert(input_aggregator, 
              input_aggregator->get_update_fields());
          input_aggregator->record_guard_set(this);
        }
      }
      if ((output_aggregator != NULL) && 
          output_aggregator->has_update_fields())
      {
#ifdef DEBUG_LEGION
        assert(output_aggregator->get_update_fields() * refining_fields);
#endif
        update_guards.insert(output_aggregator, 
            output_aggregator->get_update_fields());
        output_aggregator->record_guard_set(this);
      }
      check_for_migration(remote_tracker, source, applied_events);
      if (remove_mask != NULL)
        return !!(*remove_mask);
      return false;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_set_internal(
                                 CopyFillAggregator *&input_aggregator,
                                 const RtEvent guard_event,
                                 Operation *op, const unsigned index,
                                 const RegionUsage &usage,
                                 const FieldMask &user_mask,
                                 const InstanceSet &target_instances,
                                 const std::vector<InstanceView*> &target_views,
                                 std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // Read-write or read-only
      // Check for any copies from normal instances first
      issue_update_copies_and_fills(input_aggregator, guard_event, op, index,
          false/*track*/, user_mask, target_instances, target_views, set_expr);
      // Get the set of fields to filter, any for which we're about
      // to apply pending reductions or overwite, except those that
      // are restricted
      const FieldMask reduce_mask = reduction_fields & user_mask;
      const FieldMask restricted_mask = restricted_fields & user_mask;
      const bool is_write = IS_WRITE(usage);
      FieldMask filter_mask = is_write ? user_mask : reduce_mask;
      if (!!restricted_mask)
        filter_mask -= restricted_mask;
      if (!!filter_mask)
        filter_valid_instances(filter_mask);
      WrapperReferenceMutator mutator(applied_events);
      // Save the instances if they are not restricted
      // Otherwise if they are restricted then the restricted instances
      // are already listed as the valid views so there's nothing more
      // for us to have to do
      if (!!restricted_mask)
      {
        const FieldMask non_restricted = user_mask - restricted_fields;
        if (!!non_restricted)
          record_instances(non_restricted, target_instances, 
                           target_views, mutator); 
      }
      else
        record_instances(user_mask, target_instances,
                         target_views, mutator);
      // Next check for any reductions that need to be applied
      if (!!reduce_mask)
        apply_reductions(reduce_mask, input_aggregator, guard_event,
                         op, index, false/*track events*/); 
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::check_for_migration(RemoteEqTracker &remote_tracker,
                                             const AddressSpaceID eq_source,
                                             std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_EQUIVALENCE_SET_MIGRATION
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(user_samples.size() == user_counts.size());
#endif
      // Record our user in the set of previous users
      bool found = false;
      for (unsigned idx = 0; idx < user_samples.size(); idx++)
      {
        if (user_samples[idx] != eq_source)
          continue;
        found = true;
        user_counts[idx]++;
        break;
      }
      if (!found)
      {
        user_samples.push_back(eq_source);
        user_counts.push_back(1);
      }
      // Increase the sample count and if we haven't done enough
      // for a test then we can return and keep going
      if (++sample_count < SAMPLES_PER_MIGRATION_TEST)
      {
        // Check to see if the request bounced off a stale owner 
        // and we should send the update message
        if ((eq_source != remote_tracker.previous) &&
            (eq_source != local_space))
        {
          RtUserEvent notification_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(logical_owner_space);
            rez.serialize(notification_event);
          }
          runtime->send_equivalence_set_owner_update(eq_source, rez);
          applied_events.insert(notification_event);
        }
        return;
      }
      // Issue a warning and don't migrate if we hit this case
      if (user_samples.size() == SAMPLES_PER_MIGRATION_TEST)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_LARGE_EQUIVALENCE_SET_NODE_USAGE,
            "Internal runtime performance warning: equivalence set %lld has "
            "%zd different users which is the same as the sampling rate of "
            "%d. Please report this application use case to the Legion "
            "developers mailing list.", did, user_samples.size(),
            SAMPLES_PER_MIGRATION_TEST)
        // Reset the data structures for the next run
        user_samples.clear();
        user_counts.clear();
        sample_count = 0;
        return;
      }
      // Don't do any migrations if we have any pending refinements
      if (!pending_refinements.empty() || !!refining_fields)
      {
        // Reset the data structures for the next run
        user_samples.clear();
        user_counts.clear();
        sample_count = 0;
        return;
      }
#ifdef DEBUG_LEGION
      assert(!user_samples.empty());
#endif
      // Figure out which node(s) has/have the most uses 
      // Make sure that the current owner node is sticky
      // if it is tied for the most uses
      unsigned max = user_counts[0]; 
      AddressSpaceID max_user = user_samples[0];
      for (unsigned idx = 1; idx < user_samples.size(); idx++)
      {
        if (user_counts[idx] < max)
          continue;
        if ((user_counts[idx] == max) && 
            (user_samples[idx] != logical_owner_space))
          continue;
        max = user_counts[idx];
        max_user = user_samples[idx];
      }
      // Reset the data structures for the next run
      user_samples.clear();
      user_counts.clear();
      sample_count = 0;
      // Then decide if we need to do the migration
      // If the max_user is the logical_owner_space that is easy
      // since we don't need to move anything
      // We also check for the ping-pong case of two nodes that are
      // about evenly balance, make sure one has a 2/3 majority count
      // before migrating
      if ((max_user == logical_owner_space) ||
          // Everything below here is the test for ping pong cases
          ((user_samples.size() == 2) && 
           ((user_samples[0] == logical_owner_space) ||
            (user_samples[1] == logical_owner_space)) &&
           (max < (2 * SAMPLES_PER_MIGRATION_TEST / 3))))
      {
        // No need to do the migration in this case
        // Check to see if the request bounced off a stale owner 
        // and we should send the update message
        if ((eq_source != remote_tracker.previous) &&
            (eq_source != local_space))
        {
          RtUserEvent notification_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(logical_owner_space);
            rez.serialize(notification_event);
          }
          runtime->send_equivalence_set_owner_update(eq_source, rez);
          applied_events.insert(notification_event);
        }
        return;
      }
      // At this point we've decided to do the migration
      logical_owner_space = max_user;
      // Add ourselves and remove the new owner from remote subsets
      remote_subsets.insert(local_space);
      remote_subsets.erase(logical_owner_space);
      // We can switch our eq_state to being remote valid
      eq_state = VALID_STATE;
      RtUserEvent done_migration = Runtime::create_rt_user_event();
      // Do the migration
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(done_migration);
        pack_migration(rez, done_migration);
      }
      runtime->send_equivalence_set_migration(logical_owner_space, rez);
      applied_events.insert(done_migration);
#endif
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::acquire_restrictions(RemoteEqTracker &remote_tracker,
                                FieldMaskSet<EquivalenceSet> &alt_sets,
                                FieldMask *remove_mask,
                                const AddressSpaceID source,
                                Operation *op, FieldMask acquire_mask,
                                FieldMaskSet<InstanceView> &instances,
                                std::set<RtEvent> &applied_events,
                                const bool original_set /*=true*/)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      if (!original_set)
      {
        FieldMaskSet<EquivalenceSet>::iterator finder = 
          alt_sets.find(this);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          acquire_mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!acquire_mask)
            return false;
          finder.merge(acquire_mask);
        }
        else
          alt_sets.insert(this, acquire_mask);
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          remote_tracker.record_remote(this, acquire_mask, logical_owner_space);
          return false;
        }
        else
        {
          const FieldMask non_subset = acquire_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            remote_tracker.record_remote(this, non_subset, logical_owner_space);
            acquire_mask -= non_subset;
            if (!acquire_mask)
              return false;
          }
        }
      }
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(acquire_mask))
      {
        check_for_unrefined_remainder(eq, acquire_mask,
                                      remote_tracker.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & acquire_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!original_set)
        {
          FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(this);
          if (finder != alt_sets.end())
          {
            finder.filter(to_traverse.get_valid_mask());
            if (!finder->second)
              alt_sets.erase(finder);
          }
        }
        // Update the acquire mask and the remove_mask if there is one
        acquire_mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->acquire_restrictions(remote_tracker, alt_sets, NULL, 
              source, op, it->second, instances,
              applied_events, false/*original set*/);
        eq.reacquire();
        // Return if our acquire user mask is empty
        if (!acquire_mask)
        {
          if (remove_mask != NULL)
            return !!(*remove_mask);
          else
            return false;
        }
      }
      acquire_mask &= restricted_fields;
      if (!acquire_mask)
        return false;
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            restricted_instances.begin(); it != restricted_instances.end();it++)
      {
        const FieldMask overlap = acquire_mask & it->second;
        if (!overlap)
          continue;
        InstanceView *view = it->first->as_instance_view();
        FieldMaskSet<InstanceView>::iterator finder = 
          instances.find(view);
        if (finder != instances.end())
          finder.merge(overlap);
        else
          instances.insert(view, overlap);
      }
      restricted_fields -= acquire_mask;
      check_for_migration(remote_tracker, source, applied_events);
      if (remove_mask != NULL)
        return !!(*remove_mask);
      return false;
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::release_restrictions(RemoteEqTracker &remote_tracker,
                                FieldMaskSet<EquivalenceSet> &alt_sets,
                                FieldMask *remove_mask,
                                const AddressSpaceID source, 
                                Operation *op, FieldMask release_mask,
                                CopyFillAggregator *&release_aggregator,
                                FieldMaskSet<InstanceView> &instances,
                                std::set<RtEvent> &ready_events,
                                const bool original_set /*=true*/)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      if (!original_set)
      {
        FieldMaskSet<EquivalenceSet>::iterator finder = 
          alt_sets.find(this);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          release_mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!release_mask)
            return false;
          finder.merge(release_mask);
        }
        else
          alt_sets.insert(this, release_mask);
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(ready_events);
        if (subsets.empty())
        {
          remote_tracker.record_remote(this, release_mask, logical_owner_space);
          return false;
        }
        else
        {
          const FieldMask non_subset = release_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            remote_tracker.record_remote(this, non_subset, logical_owner_space);
            release_mask -= non_subset;
            if (!release_mask)
              return false;
          }
        }
      }
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(release_mask))
      {
        check_for_unrefined_remainder(eq, release_mask,
                                      remote_tracker.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & release_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!original_set)
        {
          FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(this);
          if (finder != alt_sets.end())
          {
            finder.filter(to_traverse.get_valid_mask());
            if (!finder->second)
              alt_sets.erase(finder);
          }
        }
        // Update the release mask and the remove_mask if there is one
        release_mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->release_restrictions(remote_tracker, alt_sets, NULL,
              source, op, it->second, release_aggregator, instances, 
              ready_events, false/*original set*/);
        eq.reacquire();
        // Return if ourt release mask is empty
        if (!release_mask)
        {
          if (remove_mask != NULL)
            return !!(*remove_mask);
          else
            return false;
        }
      }
      // Find our local restricted instances and views and record them
      InstanceSet local_instances;
      std::vector<InstanceView*> local_views;
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            restricted_instances.begin(); it != restricted_instances.end();it++)
      {
        const FieldMask overlap = it->second & release_mask;
        if (!overlap)
          continue;
        InstanceView *view = it->first->as_instance_view();
        local_instances.add_instance(InstanceRef(view->get_manager(), overlap));
        local_views.push_back(view);
        FieldMaskSet<InstanceView>::iterator finder = instances.find(view);
        if (finder != instances.end())
          finder.merge(overlap);
        else
          instances.insert(view, overlap);
      }
      if (release_aggregator != NULL)
        release_aggregator->clear_update_fields();
      // Issue the updates
      issue_update_copies_and_fills(release_aggregator, RtEvent::NO_RT_EVENT,
                                    op, 0/*index*/, false/*track*/,release_mask,
                                    local_instances, local_views, set_expr);
      // Filter the valid views
      filter_valid_instances(release_mask);
      // Update with just the restricted instances
      WrapperReferenceMutator mutator(ready_events);
      record_instances(release_mask, local_instances, local_views, mutator);
      // See if we have any reductions to apply as well
      const FieldMask reduce_mask = release_mask & reduction_fields;
      if (!!reduce_mask)
        apply_reductions(reduce_mask, release_aggregator, 
                         RtEvent::NO_RT_EVENT, op, 0/*index*/, false/*track*/);
      // Add the fields back to the restricted ones
      restricted_fields |= release_mask;
      if ((release_aggregator != NULL) && 
          release_aggregator->has_update_fields())
      {
#ifdef DEBUG_LEGION
        assert(release_aggregator->get_update_fields() * refining_fields);
#endif
        update_guards.insert(release_aggregator, 
            release_aggregator->get_update_fields());
        release_aggregator->record_guard_set(this);
      }
      check_for_migration(remote_tracker, source, ready_events);
      if (remove_mask != NULL)
        return !!(*remove_mask);
      return false;
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::issue_across_copies(RemoteEqTracker &remote_tracker,
                FieldMaskSet<EquivalenceSet> &alt_sets, FieldMask *remove_mask,
                const AddressSpaceID source, Operation *op,
                const unsigned src_index, const unsigned dst_index,
                const RegionUsage &usage, FieldMask src_mask, 
                const InstanceSet &target_instances,
                const std::vector<InstanceView*> &target_views,
                IndexSpaceExpression *overlap, CopyFillAggregator *&aggregator,
                PredEvent pred_guard, ReductionOpID redop,
                FieldMask &initialized_fields,std::set<RtEvent> &applied_events,
                const std::vector<unsigned> *src_indexes,
                const std::vector<unsigned> *dst_indexes,
                const std::vector<CopyAcrossHelper*> *across_helpers,
                const bool original_set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target_instances.size() == target_views.size());
#endif
      AutoLock eq(eq_lock,1,false/*exclusive*/);
      if (!original_set)
      {
        FieldMaskSet<EquivalenceSet>::iterator finder = 
          alt_sets.find(this);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          src_mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!src_mask)
            return false;
          finder.merge(src_mask);
        }
        else
          alt_sets.insert(this, src_mask);
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          remote_tracker.record_remote(this, src_mask, logical_owner_space);
          return false;
        }
        else
        {
          const FieldMask non_subset = src_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            remote_tracker.record_remote(this, non_subset, logical_owner_space);
            src_mask -= non_subset;
            if (!src_mask)
              return false;
          }
        }
      }
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(src_mask))
      {
        check_for_unrefined_remainder(eq, src_mask,
                                      remote_tracker.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & src_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!original_set)
        {
          FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(this);
          if (finder != alt_sets.end())
          {
            finder.filter(to_traverse.get_valid_mask());
            if (!finder->second)
              alt_sets.erase(finder);
          }
        }
        // Update the release mask and the remove_mask if there is one
        src_mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
        {
          IndexSpaceExpression *subset_overlap = runtime->forest->
            intersect_index_spaces(it->first->set_expr, overlap);
          if (subset_overlap->is_empty())
            continue;
          it->first->issue_across_copies(remote_tracker, alt_sets, NULL,
              source, op, src_index, dst_index, usage, it->second, 
              target_instances, target_views, subset_overlap, aggregator, 
              pred_guard, redop, initialized_fields, applied_events, 
              src_indexes, dst_indexes, across_helpers, false/*original set*/);
        }
        eq.reacquire();
        // Return if ourt source mask is empty
        if (!src_mask)
        {
          if (remove_mask != NULL)
            return !!(*remove_mask);
          else
            return false;
        }
      }
      // Check for any uninitialized fields
      initialized_fields -= (src_mask - valid_instances.get_valid_mask());
      if (aggregator != NULL)
        aggregator->clear_update_fields();
      if (pred_guard.exists())
        assert(false);
      if (across_helpers != NULL)
      {
        // The general case where fields don't align regardless of
        // whether we are doing a reduction across or not
#ifdef DEBUG_LEGION
        assert(src_indexes != NULL);
        assert(dst_indexes != NULL);
        assert(src_indexes->size() == dst_indexes->size());
        assert(across_helpers->size() == target_instances.size());
#endif
        // We need to figure out how to issue these copies ourself since
        // we need to map from one field to another
        // First construct a map from dst indexes to src indexes 
        std::map<unsigned,unsigned> dst_to_src;
        for (unsigned idx = 0; idx < src_indexes->size(); idx++)
          dst_to_src[(*dst_indexes)[idx]] = (*src_indexes)[idx];
        // Iterate over the target instances
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          const FieldMask dst_mask = target_instances[idx].get_valid_fields();
          // Compute a src_mask based on the dst mask
          FieldMask src_mask;
          int fidx = dst_mask.find_first_set();
          while (fidx >= 0)
          {
            std::map<unsigned,unsigned>::const_iterator finder = 
              dst_to_src.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != dst_to_src.end());
#endif
            src_mask.set_bit(finder->second);
            fidx = dst_mask.find_next_set(fidx+1);
          }
          // Now find all the source instances for this destination
          FieldMaskSet<LogicalView> src_views;
          for (FieldMaskSet<LogicalView>::const_iterator it =
                valid_instances.begin(); it != valid_instances.end(); it++)
          {
            const FieldMask field_overlap = it->second & src_mask;
            if (!field_overlap)
              continue;
            src_views.insert(it->first, field_overlap);
          }
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op,
                          src_index, dst_index, RtEvent::NO_RT_EVENT, 
                          true/*track*/, pred_guard);
          aggregator->record_updates(target_views[idx], src_views,
              src_mask, overlap, redop, (*across_helpers)[idx]);
        }
        // Now check for any reductions that need to be applied
        FieldMask reduce_mask = reduction_fields & src_mask;
        if (!!reduce_mask)
        {
#ifdef DEBUG_LEGION
          assert(redop == 0); // can't have reductions of reductions
#endif
          std::map<unsigned,unsigned> src_to_dst;
          for (unsigned idx = 0; idx < src_indexes->size(); idx++)
            src_to_dst[(*src_indexes)[idx]] = (*dst_indexes)[idx];
          int src_fidx = reduce_mask.find_first_set();
          while (src_fidx >= 0)
          {
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = reduction_instances.find(src_fidx);
#ifdef DEBUG_LEGION
            assert(finder != reduction_instances.end());
            assert(src_to_dst.find(src_fidx) != src_to_dst.end());
#endif
            const unsigned dst_fidx = src_to_dst[src_fidx];
            // Find the target targets and record them
            for (unsigned idx = 0; idx < target_views.size(); idx++)
            {
              const FieldMask target_mask = 
                target_instances[idx].get_valid_fields();
              if (!target_mask.is_set(dst_fidx))
                continue;
              if (aggregator == NULL)
                aggregator = new CopyFillAggregator(runtime->forest, op,
                              src_index, dst_index, RtEvent::NO_RT_EVENT, 
                              true/*track*/, pred_guard);
              aggregator->record_reductions(target_views[idx], finder->second,
                         src_fidx, dst_fidx, overlap, (*across_helpers)[idx]);
            }
            src_fidx = reduce_mask.find_next_set(src_fidx+1);
          }
        }
      }
      else if (redop == 0)
      {
        // Fields align and we're not doing a reduction so we can just 
        // do a normal update copy analysis to figure out what to do
        issue_update_copies_and_fills(aggregator, RtEvent::NO_RT_EVENT,
                                      op, src_index, true/*track effects*/,
                                      src_mask, target_instances,
                                      target_views, overlap, 
                                      true/*skip check*/, dst_index);
        // We also need to check for any reductions that need to be applied
        const FieldMask reduce_mask = reduction_fields & src_mask;
        if (!!reduce_mask)
        {
          int fidx = reduce_mask.find_first_set();
          while (fidx >= 0)
          {
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != reduction_instances.end());
#endif
            // Find the target targets and record them
            for (unsigned idx = 0; idx < target_views.size(); idx++)
            {
              const FieldMask target_mask = 
                target_instances[idx].get_valid_fields();
              if (!target_mask.is_set(fidx))
                continue;
              if (aggregator == NULL)
                aggregator = new CopyFillAggregator(runtime->forest, op,
                              src_index, dst_index, RtEvent::NO_RT_EVENT, 
                              true/*track*/, pred_guard);
              aggregator->record_reductions(target_views[idx], 
                            finder->second, fidx, fidx, overlap);
            }
            fidx = reduce_mask.find_next_set(fidx+1);
          }
        }
      }
      else
      {
        // Fields align but we're doing a reduction across
        // Find the valid views that we need for issuing the updates  
        FieldMaskSet<LogicalView> src_views;
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          const FieldMask overlap = it->second & src_mask;
          if (!overlap)
            continue;
          src_views.insert(it->first, overlap);
        }
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          const FieldMask &mask = target_instances[idx].get_valid_fields(); 
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op,
                          src_index, dst_index, RtEvent::NO_RT_EVENT, 
                          true/*track*/, pred_guard);
          aggregator->record_updates(target_views[idx], src_views, mask,
                                     overlap, redop, NULL/*across*/);
        }
        // There shouldn't be any reduction instances to worry about here
#ifdef DEBUG_LEGION
        assert(reduction_fields * src_mask);
#endif
      }
      if ((aggregator != NULL) &&
          aggregator->has_update_fields())
      {
#ifdef DEBUG_LEGION
        assert(aggregator->get_update_fields() * refining_fields);
#endif
        update_guards.insert(aggregator, aggregator->get_update_fields());
        aggregator->record_guard_set(this);
      }
      check_for_migration(remote_tracker, source, applied_events);
      if (remove_mask != NULL)
        return !!(*remove_mask);
      return false;
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::overwrite_set(RemoteEqTracker &remote_tracker,
                                       FieldMaskSet<EquivalenceSet> &alt_sets, 
                                       FieldMask *remove_mask,
                                       const AddressSpaceID source,
                                       Operation *op, const unsigned index,
                                       LogicalView *view, FieldMask mask,
                                       CopyFillAggregator *&output_aggregator,
                                       std::set<RtEvent> &ready_events,
                                       PredEvent pred_guard,
                                       const bool add_restriction,
                                       const bool original_set)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      if (!original_set)
      {
        FieldMaskSet<EquivalenceSet>::iterator finder = 
          alt_sets.find(this);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!mask)
            return false;
          finder.merge(mask);
        }
        else
          alt_sets.insert(this, mask);
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(ready_events);
        if (subsets.empty())
        {
          remote_tracker.record_remote(this, mask, logical_owner_space);
          return false;
        }
        else
        {
          const FieldMask non_subset = mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            remote_tracker.record_remote(this, non_subset, logical_owner_space);
            mask -= non_subset;
            if (!mask)
              return false;
          }
        }
      }
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(mask))
      {
        check_for_unrefined_remainder(eq, mask, remote_tracker.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!original_set)
        {
          FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(this);
          if (finder != alt_sets.end())
          {
            finder.filter(to_traverse.get_valid_mask());
            if (!finder->second)
              alt_sets.erase(finder);
          }
        }
        // Update the mask and the remove_mask if there is one
        mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->overwrite_set(remote_tracker, alt_sets, NULL, source,
              op, index, view, it->second, output_aggregator, ready_events, 
              pred_guard, add_restriction, false/*original set*/);
        eq.reacquire();
        // Return if ourt mask is empty
        if (!mask)
        {
          if (remove_mask != NULL)
            return !!(*remove_mask);
          else
            return false;
        }
      }
      if (output_aggregator != NULL)
        output_aggregator->clear_update_fields();
      // Two different cases here depending on whether we have a precidate 
      if (pred_guard.exists())
      {
#ifdef DEBUG_LEGION
        assert(!add_restriction); // shouldn't be doing this in this case
#endif
        // We have a predicate so collapse everything to all the valid
        // instances and then do predicate fills to all those instances
        assert(false);
      }
      else
      {
        if (add_restriction || !restricted_fields || (restricted_fields * mask))
        {
          // Easy case, just filter everything and add the new view
          const FieldMask reduce_filter = mask & reduction_fields;
          if (!!reduce_filter)
            filter_reduction_instances(reduce_filter);
          filter_valid_instances(mask);
          FieldMaskSet<LogicalView>::iterator finder = 
            valid_instances.find(view);
          if (finder == valid_instances.end())
          {
            WrapperReferenceMutator mutator(ready_events);
            view->add_nested_valid_ref(did, &mutator);
            valid_instances.insert(view, mask);
          }
          else
            finder.merge(mask);
        }
        else
        {
          // We overlap with some restricted fields so we can't filter
          // or update any restricted fields
          const FieldMask update_mask = mask - restricted_fields;
          if (!!update_mask)
          {
            const FieldMask reduce_filter = update_mask & reduction_fields;
            if (!!reduce_filter)
              filter_reduction_instances(reduce_filter);
            filter_valid_instances(update_mask);
            FieldMaskSet<LogicalView>::iterator finder = 
              valid_instances.find(view);
            if (finder == valid_instances.end())
            {
              WrapperReferenceMutator mutator(ready_events);
              view->add_nested_valid_ref(did, &mutator);
              valid_instances.insert(view, update_mask);
            }
            else
              finder.merge(update_mask);
          }
        }
        // Advance the version numbers
        advance_version_numbers(mask);
        if (add_restriction)
        {
#ifdef DEBUG_LEGION
          assert(view->is_instance_view());
#endif
          InstanceView *inst_view = view->as_instance_view();
          FieldMaskSet<InstanceView>::iterator restricted_finder = 
            restricted_instances.find(inst_view);
          if (restricted_finder == restricted_instances.end())
          {
            WrapperReferenceMutator mutator(ready_events);
            view->add_nested_valid_ref(did, &mutator);
            restricted_instances.insert(inst_view, mask);
          }
          else
            restricted_finder.merge(mask);
          restricted_fields |= mask; 
        }
        else if (!!restricted_fields)
        {
          // Check to see if we have any restricted outputs to write
          const FieldMask restricted_overlap = mask & restricted_fields;
          if (!!restricted_overlap)
            copy_out(restricted_overlap, view, op, index, output_aggregator);
        }
      }
      if ((output_aggregator != NULL) &&
          output_aggregator->has_update_fields())
      {
#ifdef DEBUG_LEGION
        assert(output_aggregator->get_update_fields() * refining_fields);
#endif
        update_guards.insert(output_aggregator, 
            output_aggregator->get_update_fields());
        output_aggregator->record_guard_set(this);
      }
      check_for_migration(remote_tracker, source, ready_events);
      if (remove_mask != NULL)
        return !!(*remove_mask);
      return false;
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::filter_set(RemoteEqTracker &remote_tracker,
                                    FieldMaskSet<EquivalenceSet> &alt_sets,
                                    FieldMask *remove_mask,
                                    const AddressSpaceID source,
                                    Operation *op, InstanceView *inst_view, 
                                    FieldMask mask,
                                    std::set<RtEvent> &applied_events,
                                    LogicalView *registration_view/*= NULL*/,
                                    const bool remove_restriction/*= false*/,
                                    const bool original_set/*= true*/)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      if (!original_set)
      {
        FieldMaskSet<EquivalenceSet>::iterator finder = 
          alt_sets.find(this);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!mask)
            return false;
          finder.merge(mask);
        }
        else
          alt_sets.insert(this, mask);
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          remote_tracker.record_remote(this, mask, logical_owner_space);
          return false;
        }
        else
        {
          const FieldMask non_subset = mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            remote_tracker.record_remote(this, non_subset, logical_owner_space);
            mask -= non_subset;
            if (!mask)
              return false;
          }
        }
      }
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(mask))
      {
        check_for_unrefined_remainder(eq, mask, remote_tracker.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!original_set)
        {
          FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(this);
          if (finder != alt_sets.end())
          {
            finder.filter(to_traverse.get_valid_mask());
            if (!finder->second)
              alt_sets.erase(finder);
          }
        }
        // Update the mask and the remove_mask if there is one
        mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->filter_set(remote_tracker, alt_sets, NULL, source,
              op, inst_view, it->second, applied_events, registration_view,
              remove_restriction, false/*original set*/);
        eq.reacquire();
        // Return if ourt mask is empty
        if (!mask)
        {
          if (remove_mask != NULL)
            return !!(*remove_mask);
          else
            return false;
        }
      }
      FieldMaskSet<LogicalView>::iterator finder = 
        valid_instances.find(inst_view);
      if (finder != valid_instances.end())
      {
        finder.filter(mask);
        if (!finder->second)
        {
          if (inst_view->remove_nested_valid_ref(did))
            delete inst_view;
          valid_instances.erase(finder);
        }
      }
      if ((registration_view != NULL) && (registration_view != inst_view))
      {
        finder = valid_instances.find(registration_view);
        if (finder != valid_instances.end())
        {
          finder.filter(mask);
          if (!finder->second)
          {
            if (registration_view->remove_nested_valid_ref(did))
              delete registration_view;
            valid_instances.erase(finder);
          }
        }
      }
      if (remove_restriction)
      {
        restricted_fields -= mask;
#ifdef DEBUG_LEGION
        assert(inst_view != NULL);
#endif
        FieldMaskSet<InstanceView>::iterator restricted_finder = 
          restricted_instances.find(inst_view);
        if (restricted_finder != restricted_instances.end())
        {
          restricted_finder.filter(mask);
          if (!restricted_finder->second)
          {
            if (inst_view->remove_nested_valid_ref(did))
              delete inst_view;
            restricted_instances.erase(restricted_finder);
          }
        }
      }
      check_for_migration(remote_tracker, source, applied_events);
      if (remove_mask != NULL)
        return !!(*remove_mask);
      return false;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::request_remote_subsets(std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
      assert(eq_state == INVALID_STATE);
      assert(!transition_event.exists());
#endif
      // It's not actually ok to block here or we risk a hang so if we're
      // not already valid and haven't requested a valid copy yet then
      // go ahead and do that and record the event as an applied event 
      // to ensure we get the update for the next user
      transition_event = Runtime::create_rt_user_event();
      eq_state = PENDING_VALID_STATE;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(local_space);
      }
      runtime->send_equivalence_set_subset_request(logical_owner_space, rez);
      applied.insert(transition_event);
    }
    
    //--------------------------------------------------------------------------
    void EquivalenceSet::record_instances(const FieldMask &record_mask,
                                 const InstanceSet &target_instances, 
                                 const std::vector<InstanceView*> &target_views,
                                          ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < target_views.size(); idx++)
      {
        const FieldMask valid_mask = 
          target_instances[idx].get_valid_fields() & record_mask;
        if (!valid_mask)
          continue;
        InstanceView *target = target_views[idx];
        // Add it to the set
        FieldMaskSet<LogicalView>::iterator finder = 
          valid_instances.find(target);
        if (finder == valid_instances.end())
        {
          target->add_nested_valid_ref(did, &mutator);
          valid_instances.insert(target, valid_mask);
        }
        else
          finder.merge(valid_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::issue_update_copies_and_fills(
                                 CopyFillAggregator *&aggregator,
                                 const RtEvent guard_event,
                                 Operation *op, const unsigned index,
                                 const bool track_events,
                                 FieldMask update_mask,
                                 const InstanceSet &target_instances,
                                 const std::vector<InstanceView*> &target_views,
                                 IndexSpaceExpression *update_expr,
                                 const bool skip_check,
                                 const int dst_index /*= -1*/) const
    //--------------------------------------------------------------------------
    {
      if (update_expr->is_empty())
        return;
      if (!skip_check)
      {
        // Scan through and figure out which fields are already valid
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          FieldMaskSet<LogicalView>::const_iterator finder = 
            valid_instances.find(target_views[idx]);
          if (finder == valid_instances.end())
            continue;
          const FieldMask &needed_mask = 
            target_instances[idx].get_valid_fields();
          const FieldMask already_valid = needed_mask & finder->second;
          if (!already_valid)
            continue;
          update_mask -= already_valid;
          // If we're already valid for all the fields then we're done
          if (!update_mask)
            return;
        }
      }
#ifdef DEBUG_LEGION
      assert(!!update_mask);
#endif
      // Find the valid views that we need for issuing the updates  
      FieldMaskSet<LogicalView> valid_views;
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        const FieldMask overlap = it->second & update_mask;
        if (!overlap)
          continue;
        valid_views.insert(it->first, overlap);
      }
      // Can happen with uninitialized data, we handle this case
      // before calling this method
      if (valid_views.empty())
        return;
      if (target_instances.size() == 1)
      {
        if (aggregator == NULL)
          aggregator = (dst_index >= 0) ?
            new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                   guard_event, track_events) :
            new CopyFillAggregator(runtime->forest, op, index,
                                   guard_event, track_events); 
        aggregator->record_updates(target_views[0], valid_views, 
                                   update_mask, update_expr);
      }
      else if (valid_views.size() == 1)
      {
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          const FieldMask dst_mask = update_mask &
            target_instances[idx].get_valid_fields();
          if (!dst_mask)
            continue;
          if (aggregator == NULL)
            aggregator = (dst_index >= 0) ?
              new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                     guard_event, track_events) :
              new CopyFillAggregator(runtime->forest, op, index,
                                     guard_event, track_events);
          aggregator->record_updates(target_views[idx], valid_views,
                                     dst_mask, update_expr);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          const FieldMask dst_mask = update_mask & 
            target_instances[idx].get_valid_fields();
          // Can happen in cases with uninitialized data
          if (!dst_mask)
            continue;
          FieldMaskSet<LogicalView> src_views;
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            const FieldMask overlap = dst_mask & it->second;
            if (!overlap)
              continue;
            src_views.insert(it->first, overlap);
          }
          if (aggregator == NULL)
            aggregator = (dst_index >= 0) ?
              new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                     guard_event, track_events) :
              new CopyFillAggregator(runtime->forest, op, index,
                                     guard_event, track_events);
          aggregator->record_updates(target_views[idx], src_views,
                                     dst_mask, update_expr);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_valid_instances(const FieldMask &filter_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!filter_mask);
#endif
      std::vector<LogicalView*> to_erase;
      for (FieldMaskSet<LogicalView>::iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        const FieldMask overlap = it->second & filter_mask;
        if (!overlap)
          continue;
        it.filter(overlap);
        if (!it->second)
          to_erase.push_back(it->first);
      }
      if (!to_erase.empty())
      {
        for (std::vector<LogicalView*>::const_iterator it = 
              to_erase.begin(); it != to_erase.end(); it++)
        {
          valid_instances.erase(*it);
          if ((*it)->remove_nested_valid_ref(did))
            delete (*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_reduction_instances(const FieldMask &to_filter)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!to_filter);
#endif
      int fidx = to_filter.find_first_set();
      while (fidx >= 0)
      {
        std::map<unsigned,std::vector<ReductionView*> >::iterator
          finder = reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
        assert(finder != reduction_instances.end());
#endif
        for (std::vector<ReductionView*>::const_iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
          if ((*it)->remove_nested_valid_ref(did))
            delete (*it);
        reduction_instances.erase(finder);
        fidx = to_filter.find_next_set(fidx+1);
      }
      reduction_fields -= to_filter;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_reductions(const FieldMask &reduce_mask,
                                          CopyFillAggregator *&aggregator,
                                          const RtEvent guard_event,
                                          Operation *op, const unsigned index,
                                          const bool trace_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!reduce_mask);
#endif
      if (set_expr->is_empty())
        return;
      int fidx = reduce_mask.find_first_set();
      while (fidx >= 0)
      {
        std::map<unsigned,std::vector<ReductionView*> >::iterator finder = 
          reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
        assert(finder != reduction_instances.end());
#endif
        // Find the target targets and record them
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          if (!it->second.is_set(fidx))
            continue;
          // Shouldn't have any deferred views here
          InstanceView *dst_view = it->first->as_instance_view();
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                                guard_event, trace_events);
          aggregator->record_reductions(dst_view, finder->second, fidx, 
                                        fidx, set_expr);
        }
        // Remove the reduction views from those available
        for (std::vector<ReductionView*>::const_iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
          if ((*it)->remove_nested_valid_ref(did))
            delete (*it);
        reduction_instances.erase(finder);
        fidx = reduce_mask.find_next_set(fidx+1);
      }
      // Record that we advanced the version number in this case
      advance_version_numbers(reduce_mask);
      // These reductions have been applied so we are done
      reduction_fields -= reduce_mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::copy_out(const FieldMask &restricted_mask,
                                  const InstanceSet &src_instances,
                                  const std::vector<InstanceView*> &src_views,
                                  Operation *op, const unsigned index,
                                  CopyFillAggregator *&aggregator) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!restricted_mask);
#endif
      if (set_expr->is_empty())
        return;
      if (valid_instances.size() == 1)
      {
        // Only 1 destination
        FieldMaskSet<LogicalView>::const_iterator first = 
          valid_instances.begin();
#ifdef DEBUG_LEGION
        assert(!(restricted_mask - first->second));
#endif
        InstanceView *dst_view = first->first->as_instance_view();
        FieldMaskSet<LogicalView> srcs;
        for (unsigned idx = 0; idx < src_views.size(); idx++)
        {
          if (first->first == src_views[idx])
            continue;
          const FieldMask overlap = 
            src_instances[idx].get_valid_fields() & restricted_mask;
          if (!overlap)
            continue;
          srcs.insert(src_views[idx], overlap);
        }
        if (!srcs.empty())
        {
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                      RtEvent::NO_RT_EVENT, true/*track*/);
          aggregator->record_updates(dst_view, srcs, restricted_mask, set_expr);
        }
      }
      else if (src_instances.size() == 1)
      {
        // Only 1 source
#ifdef DEBUG_LEGION
        assert(!(restricted_mask - src_instances[0].get_valid_fields()));
#endif
        FieldMaskSet<LogicalView> srcs;
        srcs.insert(src_views[0], restricted_mask);
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          if (it->first == src_views[0])
            continue;
          const FieldMask overlap = it->second & restricted_mask;
          if (!overlap)
            continue;
          InstanceView *dst_view = it->first->as_instance_view();
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                      RtEvent::NO_RT_EVENT, true/*track*/);
          aggregator->record_updates(dst_view, srcs, overlap, set_expr);
        }
      }
      else
      {
        // General case for cross-products
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          const FieldMask dst_overlap = it->second & restricted_mask;
          if (!dst_overlap)
            continue;
          InstanceView *dst_view = it->first->as_instance_view();
          FieldMaskSet<LogicalView> srcs;
          for (unsigned idx = 0; idx < src_views.size(); idx++)
          {
            if (dst_view == src_views[idx])
              continue;
            const FieldMask src_overlap = 
              src_instances[idx].get_valid_fields() & dst_overlap;
            if (!src_overlap)
              continue;
            srcs.insert(src_views[idx], src_overlap);
          }
          if (!srcs.empty())
          {
            if (aggregator == NULL)
              aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                        RtEvent::NO_RT_EVENT, true/*track*/);
            aggregator->record_updates(dst_view, srcs, dst_overlap, set_expr);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::copy_out(const FieldMask &restricted_mask, 
                                   LogicalView *src_view, 
                                   Operation *op, const unsigned index,
                                   CopyFillAggregator *&aggregator) const
    //--------------------------------------------------------------------------
    {
      if (set_expr->is_empty())
        return;
      FieldMaskSet<LogicalView> srcs;
      srcs.insert(src_view, restricted_mask);
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        if (it->first == src_view)
          continue;
        const FieldMask overlap = it->second & restricted_mask;
        if (!overlap)
          continue;
        InstanceView *dst_view = it->first->as_instance_view();
        if (aggregator == NULL)
          aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                    RtEvent::NO_RT_EVENT, true/*track*/);
        aggregator->record_updates(dst_view, srcs, overlap, set_expr);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::advance_version_numbers(FieldMask advance_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<VersionID> to_remove; 
      for (LegionMap<VersionID,FieldMask>::aligned::iterator it = 
            version_numbers.begin(); it != version_numbers.end(); it++)
      {
        const FieldMask overlap = it->second & advance_mask;
        if (!overlap)
          continue;
        LegionMap<VersionID,FieldMask>::aligned::iterator finder = 
          version_numbers.find(it->first + 1);
        if (finder == version_numbers.end())
          version_numbers[it->first + 1] = overlap;
        else
          finder->second |= overlap;
        it->second -= overlap;
        if (!it->second)
          to_remove.push_back(it->first);
        advance_mask -= overlap;
        if (!advance_mask)
          break;
      }
      if (!to_remove.empty())
      {
        for (std::vector<VersionID>::const_iterator it = 
              to_remove.begin(); it != to_remove.end(); it++)
          version_numbers.erase(*it);
      }
      if (!!advance_mask)
        version_numbers[init_version] = advance_mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::perform_refinements(void)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      FieldMaskSet<EquivalenceSet> to_perform;
      do 
      {
        std::set<RtEvent> refinements_done;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_perform.begin(); it != to_perform.end(); it++)
        {
          // Need the lock in read-only mode when doing the clone
          AutoLock eq(eq_lock,1,false/*exclusive*/);
          AddressSpaceID alt_space = it->first->clone_from(this, it->second);
          // If the user asked us to send it to a different node then do that
          if (alt_space != local_space)
          {
            RtUserEvent done_event = Runtime::create_rt_user_event();
            Serializer rez;
            // No RezCheck here because we might need to forward it
            rez.serialize(it->first->did);
            rez.serialize(done_event);
            pack_state(rez, it->second);
            runtime->send_equivalence_set_remote_refinement(
                it->first->logical_owner_space, rez);
            refinements_done.insert(done_event);
          }
        }
        if (!refinements_done.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(refinements_done);
          wait_on.wait();
        }
        AutoLock eq(eq_lock);
#ifdef DEBUG_LEGION
        assert(is_logical_owner());
        assert(refinement_event.exists());
        assert(eq_state == REFINING_STATE);
#endif
        // Add any new refinements to our set and record any
        // potentially complete fields
        FieldMask complete_mask;
        if (!to_perform.empty())
        {
#ifdef DEBUG_LEGION
          // These masks should be identical
          assert(refining_fields == to_perform.get_valid_mask());
          // There should be no more guards that overlap with 
          // the fields that are being refined
          assert(update_guards.get_valid_mask() * refining_fields);
#endif
          complete_mask = refining_fields;
          refining_fields.clear();
          // References were added to these sets when they were added
          // to the pending refinement queue, if they are already here
          // then we can remove the duplicate reference, no need to 
          // check for deletion since we know we hold another reference
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                to_perform.begin(); it != to_perform.end(); it++)
            if (!subsets.insert(it->first, it->second))
              it->first->remove_nested_resource_ref(did);
          to_perform.clear();
          // See if there was anyone waiting for us to be done
          if (transition_event.exists())
          {
            Runtime::trigger_event(transition_event);
            transition_event = RtUserEvent::NO_RT_USER_EVENT;
          }
        }
#ifdef DEBUG_LEGION
        assert(!refining_fields);
        assert(!transition_event.exists());
#endif
        // Fields which are still being refined are not complete
        while (!!complete_mask)
        {
          if (!pending_refinements.empty())
          {
            complete_mask -= pending_refinements.get_valid_mask();
            if (!complete_mask)
              break;
          }
          if (!unrefined_remainders.empty())
          {
            complete_mask -= unrefined_remainders.get_valid_mask();
            if (!complete_mask)
              break;
          }
          if (!disjoint_partition_refinements.empty())
            complete_mask -= disjoint_partition_refinements.get_valid_mask();
          // Only need one iteration of this loop
          break;
        }
        if (!!complete_mask)
        {
          FieldMaskSet<EquivalenceSet> complete_subsets;
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                subsets.begin(); it != subsets.end(); it++)
          {
            const FieldMask overlap = complete_mask & it->second;
            if (!overlap)
              continue;
            complete_subsets.insert(it->first, overlap);
          }
          if (complete_subsets.size() > LEGION_MAX_BVH_FANOUT)
          {
            // Sort these info field mask sets
            LegionList<FieldSet<EquivalenceSet*> >::aligned field_sets;
            complete_subsets.compute_field_sets(FieldMask(), field_sets);
            for (LegionList<FieldSet<EquivalenceSet*> >::aligned::const_iterator
                  fit = field_sets.begin(); fit != field_sets.end(); fit++)
            {
              if (fit->elements.size() <= LEGION_MAX_BVH_FANOUT)
                continue;
              KDTree *tree = NULL;
              switch (set_expr->get_num_dims())
              {
#define KDDIM(DIM) \
                case DIM: \
                  { \
                    tree = new KDNode<DIM>(set_expr, runtime, 0/*dim*/); \
                    break; \
                  }
                LEGION_FOREACH_N(KDDIM)
#undef KDDIM
                default:
                  assert(false);
              }
              // Refine the tree to make the new subsets
              std::vector<EquivalenceSet*> new_subsets(
                  fit->elements.begin(), fit->elements.end());
              if (tree->refine(new_subsets, fit->set_mask))
              {
                // Remove old references
                for (std::set<EquivalenceSet*>::const_iterator it = 
                      fit->elements.begin(); it != fit->elements.end(); it++)
                {
                  FieldMaskSet<EquivalenceSet>::iterator finder = 
                    subsets.find(*it);
#ifdef DEBUG_LEGION
                  assert(finder != subsets.end());
#endif
                  finder.filter(fit->set_mask);
                  if (!finder->second)
                  {
                    if (finder->first->remove_nested_resource_ref(did))
                      delete finder->first;
                    subsets.erase(finder);
                  }
                  // Also update the complete subsets
                  finder = complete_subsets.find(*it);
#ifdef DEBUG_LEGION
                  assert(finder != complete_subsets.end());
#endif
                  finder.filter(fit->set_mask);
                  if (!finder->second)
                    complete_subsets.erase(finder);
                }
                // Add new references
                for (std::vector<EquivalenceSet*>::const_iterator it =
                      new_subsets.begin(); it != new_subsets.end(); it++)
                {
                  if (subsets.insert(*it, fit->set_mask))
                    (*it)->add_nested_resource_ref(did);
                  // Also add it to the complete subsets
                  complete_subsets.insert(*it, fit->set_mask);
                }
              }
              // Clean up the tree
              delete tree;
            }
          }
          // If we're done refining then send updates to any
          // remote sets informing them of the complete set of subsets
          std::set<RtEvent> remote_subsets_informed;
          if (!remote_subsets.empty())
          {
            for (std::set<AddressSpaceID>::const_iterator it = 
                  remote_subsets.begin(); it != remote_subsets.end(); it++)
            {
              const RtUserEvent informed = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(informed);
                rez.serialize<size_t>(complete_subsets.size());
                for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                      complete_subsets.begin(); it != 
                      complete_subsets.end(); it++)
                {
                  rez.serialize(it->first->did);
                  rez.serialize(it->second);
                }
              }
              runtime->send_equivalence_set_subset_update(*it, rez);
              remote_subsets_informed.insert(informed);
            }
          }
          // Clean out these entries from our data structures
          if (!valid_instances.empty())
          {
            std::vector<LogicalView*> to_delete;
            for (FieldMaskSet<LogicalView>::iterator it = 
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              it.filter(complete_mask);
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<LogicalView*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
              {
                valid_instances.erase(*it);
                if ((*it)->remove_nested_valid_ref(did))
                  delete (*it); 
              }
              valid_instances.tighten_valid_mask();
            }
          }
          if (!reduction_instances.empty() && 
              !(reduction_fields * complete_mask))
          {
            for (std::map<unsigned,std::vector<ReductionView*> >::
                  iterator rit = reduction_instances.begin();
                  rit != reduction_instances.end(); /*nothing*/)
            {
              if (complete_mask.is_set(rit->first))
              {
                for (std::vector<ReductionView*>::const_iterator it = 
                    rit->second.begin(); it != rit->second.end(); it++)
                if ((*it)->remove_nested_valid_ref(did))
                  delete (*it);
                std::map<unsigned,std::vector<ReductionView*> >::iterator
                  to_delete = rit++;
                reduction_instances.erase(to_delete);
              }
              else
                rit++;
            }
            reduction_fields -= complete_mask;
          }
          if (!restricted_instances.empty() && 
              !(restricted_fields * complete_mask))
          {
            std::vector<InstanceView*> to_delete;
            for (FieldMaskSet<InstanceView>::iterator it = 
                  restricted_instances.begin(); it != 
                  restricted_instances.end(); it++)
            {
              it.filter(complete_mask);
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<InstanceView*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
              {
                restricted_instances.erase(*it);
                if ((*it)->remove_nested_valid_ref(did))
                  delete (*it); 
              }
              restricted_instances.tighten_valid_mask();
            }
            restricted_fields -= complete_mask;
          }
          for (LegionMap<VersionID,FieldMask>::aligned::iterator it =
               version_numbers.begin(); it != version_numbers.end();/*nothing*/)
          {
            it->second -= complete_mask;
            if (!it->second)
            {
              LegionMap<VersionID,FieldMask>::aligned::iterator
                to_delete = it++;
              version_numbers.erase(to_delete);
            }
            else
              it++;
          }
          // Wait for everyone to be informed before we record
          // that we are done refining
          if (!remote_subsets_informed.empty())
          {
            const RtEvent wait_on = 
              Runtime::merge_events(remote_subsets_informed);
            if (wait_on.exists() && !wait_on.has_triggered())
            {
              eq.release();
              wait_on.wait();
              eq.reacquire();
            }
          }
        } 
        // See if we have more refinements to do
        if (pending_refinements.empty())
        {
          // Go back to the mapping state and trigger our done event
          eq_state = MAPPING_STATE;
          to_trigger = refinement_event;
          refinement_event = RtUserEvent::NO_RT_USER_EVENT;
        }
        else // there are more refinements to do so we go around again
        {
          // Wait for any update_guards to finish for our pending refinements
          while (!update_guards.empty() && 
                  !(pending_refinements.get_valid_mask() *
                    update_guards.get_valid_mask()))
          {
            // If there are any mapping guards then defer ourselves
            // until a later time when there aren't any mapping guards
#ifdef DEBUG_LEGION
            assert(!transition_event.exists());
#endif
            transition_event = Runtime::create_rt_user_event();
            const RtEvent wait_on = transition_event;
            eq.release();
            wait_on.wait();
            eq.reacquire();
          }
#ifdef DEBUG_LEGION
          assert(!refining_fields); // should be empty prior to this
#endif
          refining_fields = pending_refinements.get_valid_mask();
          to_perform.swap(pending_refinements);
        }
      } while (!to_perform.empty());
#ifdef DEBUG_LEGION
      assert(to_trigger.exists());
#endif
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::record_subset(EquivalenceSet *set,
                                       const FieldMask &set_mask)
    //--------------------------------------------------------------------------
    {
      // This method is only called when adding extra levels to the 
      // equivalence set BVH data structure in order to reduce large
      // fanout. We don't need the lock and we shouldn't have any
      // remote copies of this equivalence set
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(!has_remote_instances());
#endif
      if (subsets.insert(set, set_mask))
        set->add_nested_resource_ref(did);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::finalize_disjoint_refinement(
               DisjointPartitionRefinement *dis, const FieldMask &finalize_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(unrefined_remainders.get_valid_mask() * finalize_mask);
#endif
      // We're not going to be able to finish up this disjoint
      // partition refinement so restore this to the state
      // for normal traversal
      // Figure out if we finished refining or whether there
      // is still an unrefined remainder
      IndexPartNode *partition = dis->partition;
      if (dis->children.size() < size_t(partition->total_children))
      {
        std::set<LegionColor> current_colors;
        for (std::map<IndexSpaceNode*,EquivalenceSet*>::const_iterator it =
              dis->children.begin(); it != dis->children.end(); it++)
          current_colors.insert(it->first->color);
        // No matter what finish making all the children since making
        // disjoint partitions is a good thing
        if (partition->total_children == partition->max_linearized_color)
        {
          for (LegionColor color = 0; 
                color < partition->total_children; color++)
          {
            if (current_colors.find(color) != current_colors.end())
              continue;
            IndexSpaceNode *child = partition->get_child(color);
            if (child->is_empty())
              continue;
            EquivalenceSet *child_set = 
              add_pending_refinement(child, finalize_mask, 
                                     child, runtime->address_space);
            dis->children[child] = child_set;
          }
        }
        else
        {
          for (LegionColor color = 0; 
                color < partition->max_linearized_color; color++)
          {
            if (current_colors.find(color) != current_colors.end())
              continue;
            if (!partition->color_space->contains_color(color))
              continue;
            IndexSpaceNode *child = partition->get_child(color);
            if (child->is_empty())
              continue;
            EquivalenceSet *child_set = 
              add_pending_refinement(child, finalize_mask, 
                                     child, runtime->address_space);
            dis->children[child] = child_set;
          }
        }
      }
      if (!partition->is_complete())
      {
        // We had all the children, but the partition is not 
        // complete so we actually need to do the subtraction
        IndexSpaceExpression *diff_expr = 
          runtime->forest->subtract_index_spaces(set_expr, 
              partition->get_union_expression());
        if ((diff_expr != NULL) && !diff_expr->is_empty())
        {
#ifdef DEBUG_LEGION
          assert(unrefined_remainders.get_valid_mask() * finalize_mask);
#endif
          unrefined_remainders.insert(diff_expr, finalize_mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_unrefined_remainders(FieldMask &to_filter,
                                                     IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!to_filter);
      assert(!expr->is_empty());
#endif
      if (unrefined_remainders.empty())
        return;
      if (to_filter * unrefined_remainders.get_valid_mask())
        return;
      FieldMaskSet<IndexSpaceExpression> to_add;
      std::vector<IndexSpaceExpression*> to_delete;
      for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
            unrefined_remainders.begin(); it !=
            unrefined_remainders.end(); it++)
      {
        const FieldMask overlap = to_filter & it->second;
        if (!overlap)
          continue;
        IndexSpaceExpression *remainder = 
          runtime->forest->subtract_index_spaces(it->first, expr);
        if (!remainder->is_empty())
          to_add.insert(remainder, overlap);
        it.filter(overlap);
        if (!it->second)
          to_delete.push_back(it->first);
        to_filter -= overlap;
        if (!to_filter)
          break;
      }
      if (!to_delete.empty())
      {
        for (std::vector<IndexSpaceExpression*>::const_iterator 
              it = to_delete.begin(); it != to_delete.end(); it++)
          unrefined_remainders.erase(*it);
      }
      if (!to_add.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator
              it = to_add.begin(); it != to_add.end(); it++)
          unrefined_remainders.insert(it->first, it->second);
      }
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
        rez.serialize(logical_owner_space);
        set_expr->pack_expression(rez, target);
        if (index_space_node != NULL)
          rez.serialize(index_space_node->handle);
        else
          rez.serialize(IndexSpace::NO_SPACE);
      }
      runtime->send_equivalence_set_response(target, rez);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* EquivalenceSet::add_pending_refinement(
                              IndexSpaceExpression *expr, const FieldMask &mask,
                              IndexSpaceNode *node, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // See if we already have a subset with this expression
      EquivalenceSet *subset = NULL;
      if ((subset_exprs == NULL) && !subsets.empty())
      {
        // Fill in the data structure if it hasn't already been done
        // e.g. due to migration
        subset_exprs = new std::map<IndexSpaceExpression*,EquivalenceSet*>();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
          (*subset_exprs)[it->first->set_expr] = it->first;
      }
      if (subset_exprs != NULL)
      {
        std::map<IndexSpaceExpression*,EquivalenceSet*>::const_iterator
          finder = subset_exprs->find(expr);
        if (finder != subset_exprs->end())
          subset = finder->second;
      }
      if (subset == NULL)
      {
        // Make a new subset
        subset = new EquivalenceSet(runtime, 
            runtime->get_available_distributed_id(),
            local_space, source, expr, node, true/*register*/);
        if (subset_exprs == NULL)
          subset_exprs = new std::map<IndexSpaceExpression*,EquivalenceSet*>();
        // Save it in the set
        (*subset_exprs)[expr] = subset;
        if (pending_refinements.insert(subset, mask))
          subset->add_nested_resource_ref(did);
      }
      else
      {
        // We already have a subset, see which fields it's already
        // been refined for (maybe none if it is still pending)
        FieldMaskSet<EquivalenceSet>::const_iterator finder = 
          subsets.find(subset);
        if (finder != subsets.end())
        {
          const FieldMask diff_mask = mask - finder->second;
          if (!!diff_mask)
          {
            if (pending_refinements.insert(subset, diff_mask))
              subset->add_nested_resource_ref(did);
          }
          else // It's already refined for all of them, so just return
            return subset;
        }
        else
        {
          // Do the normal insert if we couldn't find it
          if (pending_refinements.insert(subset, mask))
            subset->add_nested_resource_ref(did);
        }
      }
      // Launch the refinement task if there isn't one already running
      if (eq_state == MAPPING_STATE)
      {
#ifdef DEBUG_LEGION
        assert(!transition_event.exists());
        assert(!refinement_event.exists());
        assert(!refining_fields); // should be empty
#endif
        refinement_event = Runtime::create_rt_user_event();
        eq_state = REFINING_STATE;
        // Launch the refinement task to be performed
        RefinementTaskArgs args(this);
        // If we have outstanding guard events then make a transition event
        // for them to trigger when the last one has been removed such
        // that the refinement task will not start before it's ready
        if (!update_guards.empty() && !(pending_refinements.get_valid_mask() 
                                            * update_guards.get_valid_mask()))
        {
#ifdef DEBUG_LEGION
          assert(!transition_event.exists());
#endif
          transition_event = Runtime::create_rt_user_event();
        }
        runtime->issue_runtime_meta_task(args, 
            LG_THROUGHPUT_DEFERRED_PRIORITY, transition_event);
      }
      return subset;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_request(AddressSpaceID source,
                                                RtUserEvent deferral_event)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        // We didn't get the lock so build a continuation
        // We need a name for our completion event that we can use for
        // the atomic compare and swap below
        if (!deferral_event.exists())
        {
          // If we haven't already been deferred then we need to 
          // add ourselves to the back of the list of deferrals
          deferral_event = Runtime::create_rt_user_event();
          volatile Realm::Event::id_t *ptr = 
            (volatile Realm::Event::id_t*)&next_deferral_precondition.id;
          RtEvent continuation_pre;
          do {
            continuation_pre.id = *ptr;
          } while (!__sync_bool_compare_and_swap(ptr,
                    continuation_pre.id, deferral_event.id));
          DeferSubsetRequestArgs args(this, source, deferral_event);
          runtime->issue_runtime_meta_task(args, 
                          LG_LATENCY_DEFERRED_PRIORITY, continuation_pre);
        }
        else
        {
          // We've already been deferred and our precondition has already
          // triggered so just launch ourselves again whenever the lock
          // should be ready to try again
          DeferSubsetRequestArgs args(this, source, deferral_event);
          runtime->issue_runtime_meta_task(args,
                   LG_LATENCY_DEFERRED_PRIORITY, eq.try_next());
        }
        return;
      }
      if (!is_logical_owner())
      {
        // If we're not the owner anymore then forward on the request
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(source);
        }
        runtime->send_equivalence_set_subset_request(logical_owner_space, rez);
        if (deferral_event.exists())
          Runtime::trigger_event(deferral_event);
        return;
      }
      // If we arrived back at ourself after we were made the owner
      // then there is nothing for us to do
      if (source == local_space)
      {
        if (deferral_event.exists())
          Runtime::trigger_event(deferral_event);
        return;
      }
      // If we're in the process of doing a refinement, wait for
      // that to be done before we do anything else
      if (eq_state == REFINING_STATE)
      {
#ifdef DEBUG_LEGION
        assert(refinement_event.exists());
#endif
        DeferSubsetRequestArgs args(this, source, deferral_event);       
        runtime->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY, refinement_event);
        return;
      }
#ifdef DEBUG_LEGION
      assert(remote_subsets.find(source) == remote_subsets.end());
#endif
      // Record the remote subsets
      remote_subsets.insert(source);
      // Remote copies of the subsets either have to be empty or a 
      // full copy of the subsets with no partial refinements
      if (!subsets.empty())
      {
        FieldMask complete_mask = subsets.get_valid_mask();
        // Any fields for which we have partial refinements cannot be sent yet
        if (!pending_refinements.empty())
          complete_mask -= pending_refinements.get_valid_mask();
        if (!!refining_fields)
          complete_mask -= refining_fields;
        if (!unrefined_remainders.empty())
          complete_mask -= unrefined_remainders.get_valid_mask();
        if (!!disjoint_partition_refinements.empty())
          complete_mask -= disjoint_partition_refinements.get_valid_mask();
        if (!!complete_mask)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<size_t>(subsets.size());
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  subsets.begin(); it != subsets.end(); it++)
            {
              const FieldMask overlap = it->second & complete_mask;
              if (!!overlap)
              {
                rez.serialize(it->first->did);
                rez.serialize(overlap);
              }
              else
                rez.serialize<DistributedID>(0);
            }
          }
          runtime->send_equivalence_set_subset_response(source, rez);
          if (deferral_event.exists())
            Runtime::trigger_event(deferral_event);
          return;
        }
      }
      // If we make it here then we just send a message with an 
      // empty set of subsets to allow forward progress to be made
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize<size_t>(0);
      }
      runtime->send_equivalence_set_subset_response(source, rez);
      if (deferral_event.exists())
        Runtime::trigger_event(deferral_event);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_subsets;
      derez.deserialize(num_subsets);
      FieldMaskSet<EquivalenceSet> new_subsets;
      if (num_subsets > 0)
      {
        std::set<RtEvent> ready_events;
        for (unsigned idx = 0; idx < num_subsets; idx++)
        {
          DistributedID subdid;
          derez.deserialize(subdid);
          if (subdid == 0)
            continue;
          RtEvent ready;
          EquivalenceSet *set =
            runtime->find_or_request_equivalence_set(subdid, ready);
          if (ready.exists())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          new_subsets.insert(set, mask);
        }
        if (!ready_events.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(ready_events);
          if (wait_on.exists())
            wait_on.wait();
        }
      }
      AutoLock eq(eq_lock);
      if (is_logical_owner())
      {
        // If we've since been made the logical owner then there
        // should be nothing else for us to do
#ifdef DEBUG_LEGION
        assert(new_subsets.empty());
#endif
        return;
      }
      else if (eq_state == PENDING_VALID_STATE)
      {
#ifdef DEBUG_LEGION
        assert(subsets.empty());
        assert(transition_event.exists());
        assert(!transition_event.has_triggered());
#endif
        if (!new_subsets.empty()) 
        {
          subsets.swap(new_subsets);
          // Add the references
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                subsets.begin(); it != subsets.end(); it++)
            it->first->add_nested_resource_ref(did);
        }
        // Update the state
        eq_state = VALID_STATE;
        // Trigger the transition state to wake up any waiters
        Runtime::trigger_event(transition_event);
        transition_event = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_subsets;
      derez.deserialize(num_subsets);
      if (num_subsets == 0)
        return;
      std::vector<EquivalenceSet*> new_subsets(num_subsets); 
      LegionVector<FieldMask>::aligned new_masks(num_subsets);
      std::set<RtEvent> wait_for;
      for (unsigned idx = 0; idx < num_subsets; idx++)
      {
        DistributedID subdid;
        derez.deserialize(subdid);
        RtEvent ready;
        new_subsets[idx] = 
          runtime->find_or_request_equivalence_set(subdid, ready);
        if (ready.exists())
          wait_for.insert(ready);
        derez.deserialize(new_masks[idx]);
      }
      if (!wait_for.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(wait_for);
        wait_on.wait();
      }
      AutoLock eq(eq_lock);
      if (is_logical_owner())
        // If we've become the logical owner there is nothing to do
        return;
#ifdef DEBUG_LEGION
      assert(eq_state == VALID_STATE);
      assert(!transition_event.exists());
#endif
      for (unsigned idx = 0; idx < num_subsets; idx++)
        if (subsets.insert(new_subsets[idx], new_masks[idx]))
          new_subsets[idx]->add_nested_resource_ref(did);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_refinement(const void *args)
    //--------------------------------------------------------------------------
    {
      const RefinementTaskArgs *rargs = (const RefinementTaskArgs*)args;
      rargs->target->perform_refinements();
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_remote_references(const void *args)
    //--------------------------------------------------------------------------
    {
      const RemoteRefTaskArgs *rargs = (const RemoteRefTaskArgs*)args;
      if (rargs->done_event.exists())
      {
        LocalReferenceMutator mutator; 
        if (rargs->add_references)
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->add_nested_valid_ref(rargs->did, &mutator, it->second);
        }
        else
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->remove_nested_valid_ref(rargs->did, &mutator,it->second);
        }
        const RtEvent done_pre = mutator.get_done_event();
        Runtime::trigger_event(rargs->done_event, done_pre);
      }
      else
      {
        if (rargs->add_references)
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->add_nested_valid_ref(rargs->did, NULL, it->second);
        }
        else
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->remove_nested_valid_ref(rargs->did, NULL, it->second);
        }
      }
      delete rargs->refs;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace(const void *args,
                                                     Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferRayTraceArgs *dargs = (const DeferRayTraceArgs*)args;
      // See if we need to load the expression or not
      if (dargs->expr == NULL)
      {
        IndexSpaceExpression *expr = (dargs->is_expr_space) ?
          runtime->forest->get_node(dargs->expr_handle) :
          runtime->forest->find_remote_expression(dargs->expr_id);
        dargs->set->ray_trace_equivalence_sets(dargs->target, expr,
                            *(dargs->ray_mask), dargs->handle, dargs->origin,
                            dargs->done, dargs->deferral);
      }
      else
        dargs->set->ray_trace_equivalence_sets(dargs->target, dargs->expr,
                            *(dargs->ray_mask), dargs->handle, dargs->origin,
                            dargs->done, dargs->deferral);
      // Clean up our ray mask
      delete dargs->ray_mask;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_finish(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRayTraceFinishArgs *dargs = 
        (const DeferRayTraceFinishArgs*)args;
      std::set<RtEvent> done_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            dargs->to_traverse->begin(); it != dargs->to_traverse->end(); it++)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator
          finder = dargs->exprs->find(it->first);
#ifdef DEBUG_LEGION
        assert(finder != dargs->exprs->end());
#endif
        const IndexSpace subset_handle = 
          (dargs->handle.exists() && 
            (finder->second->get_volume() == dargs->volume)) ? dargs->handle :
              IndexSpace::NO_SPACE;
        it->first->ray_trace_equivalence_sets(dargs->target, finder->second, 
            it->second, subset_handle, dargs->source, done);
        done_events.insert(done);
      }
      if (!done_events.empty())
        Runtime::trigger_event(dargs->done, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(dargs->done);
      delete dargs->to_traverse;
      delete dargs->exprs;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_request(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferSubsetRequestArgs *dargs = (const DeferSubsetRequestArgs*)args;
      dargs->set->process_subset_request(dargs->source, dargs->deferral);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_make_owner(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMakeOwnerArgs *dargs = (const DeferMakeOwnerArgs*)args;
      dargs->set->make_owner();
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_merge_or_forward(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMergeOrForwardArgs *dargs = 
        (const DeferMergeOrForwardArgs*)args;
      dargs->set->merge_or_forward(dargs->done, *(dargs->views), 
          *(dargs->reductions), *(dargs->restricted), *(dargs->versions));
      delete dargs->views;
      delete dargs->reductions;
      delete dargs->restricted;
      delete dargs->versions;
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
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      IndexSpace handle;
      derez.deserialize(handle);

      void *location;
      EquivalenceSet *set = NULL;
      // We only actually need the index space node on the owner and the
      // logical owner otherwise we can skip it
      IndexSpaceNode *node = NULL;
      if (handle.exists() && (logical_owner == runtime->address_space))
        node = runtime->forest->get_node(handle);
      if (runtime->find_pending_collectable_location(did, location))
        set = new(location) EquivalenceSet(runtime, did, source, logical_owner,
                                           expr, node, false/*register now*/);
      else
        set = new EquivalenceSet(runtime, did, source, logical_owner,
                                 expr, node, false/*register now*/);
      // Once construction is complete then we do the registration
      set->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_request(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      AddressSpaceID source;
      derez.deserialize(source);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->process_subset_request(source, RtUserEvent::NO_RT_USER_EVENT);
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
    /*static*/ void EquivalenceSet::handle_subset_update(
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
      set->process_subset_update(derez);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);

      VersionManager *target;
      derez.deserialize(target);
      bool is_expr_space;
      IndexSpace expr_handle;
      IndexSpaceExprID expr_id;
      RtEvent expr_ready;
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source,
                              is_expr_space, expr_handle, expr_id, expr_ready);
      FieldMask ray_mask;
      derez.deserialize(ray_mask);
      IndexSpace handle;
      derez.deserialize(handle);
      AddressSpaceID origin;
      derez.deserialize(origin);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      if (ready.exists() || expr_ready.exists())
      {
        const RtEvent defer = Runtime::merge_events(ready, expr_ready);
        if (defer.exists() && !defer.has_triggered())
        {
          // We need to defer this until things are ready
          DeferRayTraceArgs args(set, target, expr, 
                                 handle, origin, done_event,
                                 RtUserEvent::NO_RT_USER_EVENT,
                                 ray_mask, is_expr_space, 
                                 expr_handle, expr_id);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_DEFERRED_PRIORITY, defer); 
          return;
        }
        if (expr_ready.exists())
          expr = (is_expr_space) ? runtime->forest->get_node(expr_handle) :
            runtime->forest->find_remote_expression(expr_id);
        // Fall through and actually do the operation now
      }
      set->ray_trace_equivalence_sets(target, expr, ray_mask, handle, 
                                      origin, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask eq_mask;
      derez.deserialize(eq_mask);
      VersionManager *target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      if (ready.exists() && !ready.has_triggered())
      {
        target->record_pending_equivalence_set(set, eq_mask);
        Runtime::trigger_event(done_event, ready);
      }
      else
      {
        target->record_equivalence_set(set, eq_mask);
        Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_migration(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      RtUserEvent done;
      derez.deserialize(done);

      LocalReferenceMutator mutator; 
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_migration(derez, mutator, source);
      Runtime::trigger_event(done, mutator.get_done_event());
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_owner_update(Deserializer &derez,
                                                        Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      AddressSpaceID new_owner;
      derez.deserialize(new_owner);
      RtUserEvent done;
      derez.deserialize(done);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->update_owner(new_owner);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_remote_refinement(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_state(derez);
    }

    /////////////////////////////////////////////////////////////
    // Version Manager 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    VersionManager::VersionManager(RegionTreeNode *n, ContextID c)
      : ctx(c), node(n), runtime(n->context->runtime), version_number(0)
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
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        {
          if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
            delete it->first;
        }
        equivalence_sets.clear();
      }
#ifdef DEBUG_LEGION
      assert(waiting_infos.empty());
      assert(equivalence_sets_ready.empty());
#endif
      version_number = 0;
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::perform_versioning_analysis(InnerContext *context,
                             VersionInfo *version_info, RegionNode *region_node,
                             const FieldMask &version_mask, Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node == region_node);
#endif
      // If we don't have equivalence classes for this region yet we 
      // either need to compute them or request them from the owner
      FieldMask remaining_mask(version_mask);
      bool has_waiter = false;
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        // Check to see if any computations of equivalence sets are in progress
        // If so we'll skip out early and go down the slow path which should
        // be a fairly rare thing to do
        if (!equivalence_sets_ready.empty())
        {
          for (LegionMap<RtUserEvent,FieldMask>::aligned::const_iterator it =
                equivalence_sets_ready.begin(); it != 
                equivalence_sets_ready.end(); it++)
          {
            if (remaining_mask * it->second)
              continue;
            // Skip out earlier if we have at least one thing to wait
            // for since we're going to have to go down the slow path
            has_waiter = true;
            break;
          }
        }
        // If we have a waiter, then don't bother doing this
        if (!has_waiter)
        {
          // Get any fields that are already ready
          if (version_info != NULL)
          {
            if (!(version_mask * equivalence_sets.get_valid_mask()))
            {
              for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                   equivalence_sets.begin(); it != equivalence_sets.end(); it++)
              {
                const FieldMask overlap = it->second & version_mask;
                if (!overlap)
                  continue;
                version_info->record_equivalence_set(this, version_number,
                                                     it->first, overlap);
              }
            }
          }
          remaining_mask -= equivalence_sets.get_valid_mask();
          // If we got all our fields then we are done
          if (!remaining_mask)
            return RtEvent::NO_RT_EVENT;
        }
      }
      // Retake the lock in exclusive mode and make sure we don't lose the race
      RtUserEvent compute_event;
      std::set<RtEvent> wait_on;
      {
        FieldMask waiting_mask;
        AutoLock m_lock(manager_lock);
        if (!equivalence_sets_ready.empty())
        {
          for (LegionMap<RtUserEvent,FieldMask>::aligned::const_iterator it =
                equivalence_sets_ready.begin(); it != 
                equivalence_sets_ready.end(); it++)
          {
            const FieldMask overlap = remaining_mask & it->second;
            if (!overlap)
              continue;
            wait_on.insert(it->first);
            waiting_mask |= overlap;
            remaining_mask -= overlap;
            if (!remaining_mask)
              break;
          }
        }
        // Get any fields that are already ready
        if (!(remaining_mask * equivalence_sets.get_valid_mask()))
        {
          if (version_info != NULL)
          {
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  equivalence_sets.begin(); it != equivalence_sets.end(); it++)
            {
              const FieldMask overlap = it->second & remaining_mask;
              if (!overlap)
                continue;
              version_info->record_equivalence_set(this, version_number,
                                                   it->first, overlap);
            }
          }
          remaining_mask -= equivalence_sets.get_valid_mask();
          if (!remaining_mask) // We're done if we got all our fields
            return RtEvent::NO_RT_EVENT;
        }
        // If we have waiting fields then update them now
        if (!!waiting_mask)
          remaining_mask |= waiting_mask;
#ifdef DEBUG_LEGION
        assert(!!remaining_mask);
#endif
        // Record that our version info is waiting for these fields
        if (version_info != NULL)
          waiting_infos.insert(version_info, remaining_mask);
        if (!!remaining_mask)
        {
          compute_event = Runtime::create_rt_user_event();
          equivalence_sets_ready[compute_event] = remaining_mask; 
          wait_on.insert(compute_event);
        }
      }
      if (compute_event.exists())
      {
        IndexSpaceExpression *expr = region_node->row_source; 
        IndexSpace handle = region_node->row_source->handle;
        RtEvent ready = context->compute_equivalence_sets(this, 
                      region_node->get_tree_id(), handle, expr, 
                      remaining_mask, runtime->address_space);
        if (ready.exists() && !ready.has_triggered())
        {
          // Launch task to finalize the sets once they are ready
          LgFinalizeEqSetsArgs args(this, compute_event, 
                                    op->get_unique_op_id());
          runtime->issue_runtime_meta_task(args, 
                             LG_LATENCY_DEFERRED_PRIORITY, ready);
        }
        else
          finalize_equivalence_sets(compute_event);
      }
      return Runtime::merge_events(wait_on); 
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_equivalence_set(EquivalenceSet *set,
                                                const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      if (equivalence_sets.insert(set, mask))
        set->add_base_resource_ref(VERSION_MANAGER_REF);
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_pending_equivalence_set(EquivalenceSet *set,
                                                        const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      pending_equivalence_sets.insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void VersionManager::finalize_equivalence_sets(RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      LegionMap<RtUserEvent,FieldMask>::aligned::iterator finder =
        equivalence_sets_ready.find(done_event);
#ifdef DEBUG_LEGION
      assert(finder != equivalence_sets_ready.end());
#endif
      // If there are any pending equivalence sets, move them into 
      // the actual equivalence sets
      if (!pending_equivalence_sets.empty() && 
          !(finder->second * pending_equivalence_sets.get_valid_mask()))
      {
        std::vector<EquivalenceSet*> to_delete;
        for (FieldMaskSet<EquivalenceSet>::iterator it = 
              pending_equivalence_sets.begin(); it !=
              pending_equivalence_sets.end(); it++)
        {
          // Once it's valid for any field then it's valid for all of them
          if (it->second * finder->second)
            continue;
          if (equivalence_sets.insert(it->first, it->second))
            it->first->add_base_resource_ref(VERSION_MANAGER_REF);
          to_delete.push_back(it->first);
        }
        if (!to_delete.empty())
        {
          if (to_delete.size() < pending_equivalence_sets.size())
          {
            for (std::vector<EquivalenceSet*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
              pending_equivalence_sets.erase(*it);
            pending_equivalence_sets.tighten_valid_mask();
          }
          else
            pending_equivalence_sets.clear();
        }
      }
      if (!waiting_infos.empty() &&
          !(waiting_infos.get_valid_mask() * finder->second))
      {
        std::vector<VersionInfo*> to_delete;
        for (FieldMaskSet<VersionInfo>::iterator vit = 
              waiting_infos.begin(); vit != waiting_infos.end(); vit++)
        {
          const FieldMask info_overlap = vit->second & finder->second;
          if (!info_overlap)
            continue;
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                equivalence_sets.begin(); it != equivalence_sets.end(); it++)
          {
            const FieldMask overlap = info_overlap & it->second;
            if (!overlap)
              continue;
            vit->first->record_equivalence_set(this, version_number,
                                               it->first, overlap);
          }
          vit.filter(info_overlap);
          if (!vit->second)
            to_delete.push_back(vit->first);
        }
        if (!to_delete.empty())
        {
          for (std::vector<VersionInfo*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            waiting_infos.erase(*it);
        }
      }
      Runtime::trigger_event(done_event);
      equivalence_sets_ready.erase(finder);
    }

    //--------------------------------------------------------------------------
    void VersionManager::update_equivalence_sets(const unsigned previous_number,
                                  const FieldMaskSet<EquivalenceSet> &to_add,
                                  const FieldMaskSet<EquivalenceSet> &to_delete)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
      assert(previous_number <= version_number);
#endif
      if (previous_number < version_number)
        return;
      // Increment the version number so that we don't get stale updates
      version_number++;
      // Remove any sets from the old set that aren't in the new one
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            to_delete.begin(); it != to_delete.end(); it++)
      {
        FieldMaskSet<EquivalenceSet>::iterator finder = 
          equivalence_sets.find(it->first);
        // Might already have been removed
        if (finder == equivalence_sets.end())
          continue;
        finder.filter(it->second);
        if (!finder->second)
        {
          equivalence_sets.erase(finder);
          if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
            delete it->first;
        }
      }
      // Add in all the alt_sets and add references where necessary
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            to_add.begin(); it != to_add.end(); it++)
        if (equivalence_sets.insert(it->first, it->second))
          it->first->add_base_resource_ref(VERSION_MANAGER_REF);
    }

    //--------------------------------------------------------------------------
    void VersionManager::print_physical_state(RegionTreeNode *node,
                                              const FieldMask &capture_mask,
                                              TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      logger->log("Equivalence Sets:");
      logger->down();
      // TODO: log equivalence sets
      assert(false);
      logger->up();
    } 

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_finalize_eq_sets(const void *args)
    //--------------------------------------------------------------------------
    {
      const LgFinalizeEqSetsArgs *fargs = (const LgFinalizeEqSetsArgs*)args;
      fargs->manager->finalize_equivalence_sets(fargs->compute);
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
    bool InstanceRef::is_field_set(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      unsigned index = manager->field_space_node->get_field_index(fid);
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
    void InstanceSet::swap(InstanceSet &rhs)
    //--------------------------------------------------------------------------
    {
      // Swap references
      {
        InternalSet *other = rhs.refs.multi;
        rhs.refs.multi = refs.multi;
        refs.multi = other;
      }
      // Swap single
      {
        bool other = rhs.single;
        rhs.single = single;
        single = other;
      }
      // Swap shared
      {
        bool other = rhs.shared;
        rhs.shared = shared;
        shared = other;
      }
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
      // There is no version information on partitions
      return true;
    }

  }; // namespace Internal 
}; // namespace Legion

