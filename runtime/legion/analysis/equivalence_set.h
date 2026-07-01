/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_EQUIVALENCE_SET_H__
#define __LEGION_EQUIVALENCE_SET_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/utilities/instance_set.h"
#include "legion/views/collective.h"

namespace Legion {
  namespace Internal {

    /**
     * \class CollectiveRefinementTree
     * This class provides a base class for performing analyses inside of
     * an equivalence set that might have to deal with aliased names for
     * physical instances due to collective views. It does this by
     * constructing a tree of set of instances that are all performing
     * the same analysis. As new views are traversed, the analysis can
     * fracture into subsets of instances that all have the same state.
     */
    template<typename T>
    class CollectiveRefinementTree {
    public:
      CollectiveRefinementTree(CollectiveView* view);
      CollectiveRefinementTree(std::vector<DistributedID>&& inst_dids);
      ~CollectiveRefinementTree(void);
    public:
      const FieldMask& get_refinement_mask(void) const
      {
        return refinements.get_valid_mask();
      }
      const std::vector<DistributedID>& get_instances(void) const;
      // Traverse the tree and perform refinements as necessary
      template<typename... Args>
      void traverse(InstanceView* view, const FieldMask& mask, Args&&... args);
      // Visit just the leaves of the analysis
      template<typename... Args>
      void visit_leaves(const FieldMask& mask, Args&&... args)
      {
        for (typename local::FieldMaskMap<T>::const_iterator it =
                 refinements.begin();
             it != refinements.end(); it++)
        {
          const FieldMask& overlap = mask & it->second;
          if (!overlap)
            continue;
          it->first->visit_leaves(overlap, args...);
        }
        const FieldMask local_mask = mask - refinements.get_valid_mask();
        if (!!local_mask)
          static_cast<T*>(this)->visit_leaf(local_mask, args...);
      }
      InstanceView* get_instance_view(
          InnerContext* context, RegionTreeID tid) const;
    public:
      // Helper methods for traversing the total and partial valid views
      // These should only be called on the root with collective != nullptr
      void traverse_total(
          const FieldMask& mask, IndexSpaceExpression* set_expr,
          const FieldMapView<LogicalView>& total_valid_views);
      void traverse_partial(
          const FieldMask& mask,
          const MapView<
              LogicalView*, shrt::FieldMaskMap<IndexSpaceExpression> >&
              partial_valid_views);
    protected:
      CollectiveView* const collective;
      const std::vector<DistributedID> inst_dids;
    private:
      local::FieldMaskMap<T> refinements;
    };

    /**
     * \class MakeCollectiveValid
     * This class helps in the aliasing analysis when determining how
     * to issue copies to update collective views. It builds an instance
     * refinement tree to find the valid expressions for each group of
     * instances and then can use that to compute the updates
     */
    class MakeCollectiveValid
      : public CollectiveRefinementTree<MakeCollectiveValid>,
        public Heapify<MakeCollectiveValid, OPERATION_LIFETIME> {
    public:
      MakeCollectiveValid(
          CollectiveView* view,
          const FieldMapView<IndexSpaceExpression>& needed_exprs);
      MakeCollectiveValid(
          std::vector<DistributedID>&& insts,
          const FieldMapView<IndexSpaceExpression>& needed_exprs,
          const FieldMapView<IndexSpaceExpression>& valid_expr,
          const FieldMask& mask, InstanceView* inst_view);
    public:
      MakeCollectiveValid* clone(
          InstanceView* view, const FieldMask& mask,
          std::vector<DistributedID>&& insts) const;
      void analyze(
          InstanceView* view, const FieldMask& mask,
          IndexSpaceExpression* expr);
      void analyze(
          InstanceView* view, const FieldMask& mask,
          const FieldMapView<IndexSpaceExpression>& exprs);
      void visit_leaf(
          const FieldMask& mask, InnerContext* context, RegionTreeID tid,
          local::map<InstanceView*, local::FieldMaskMap<IndexSpaceExpression> >&
              updates);
    public:
      InstanceView* const view;
    protected:
      // Expression fields that we need to make valid for all instances
      const FieldMapView<IndexSpaceExpression>& needed_exprs;
      // Expression fields that are valid for these instances
      local::FieldMaskMap<IndexSpaceExpression> valid_exprs;
    };

    /**
     * \class CollectiveAntiAlias
     * This class helps with the alias analysis when performing a check
     * to see which collective instances are not considered valid
     * It is also used by TraceViewSet::dominates to test for dominance
     * of collective views which is the same as testing that the
     * collective view
     */
    class CollectiveAntiAlias
      : public CollectiveRefinementTree<CollectiveAntiAlias>,
        public Heapify<CollectiveAntiAlias, OPERATION_LIFETIME> {
    public:
      CollectiveAntiAlias(CollectiveView* view);
      CollectiveAntiAlias(
          std::vector<DistributedID>&& insts,
          const FieldMapView<IndexSpaceExpression>& valid_expr,
          const FieldMask& mask);
    public:
      CollectiveAntiAlias* clone(
          InstanceView* view, const FieldMask& mask,
          std::vector<DistributedID>&& insts) const;
      void analyze(
          InstanceView* view, const FieldMask& mask,
          IndexSpaceExpression* expr);
      void analyze(
          InstanceView* view, const FieldMask& mask,
          const FieldMapView<IndexSpaceExpression>& exprs);
      void visit_leaf(
          const FieldMask& mask, FieldMask& allvalid_mask,
          IndexSpaceExpression* expr,
          const FieldMapView<IndexSpaceExpression>& partial_valid_exprs);
      // This version used by TraceViewSet::dominates to find
      // non-dominating overlaps
      void visit_leaf(
          const FieldMask& mask, FieldMask& dominated_mask,
          InnerContext* context, RegionTreeID tree_id, CollectiveView* view,
          local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression> >&
              non_dominated,
          IndexSpaceExpression* expr);
      // This version used by TraceViewSet::antialias_collective_view
      // to get the names of the new views to use for instances
      void visit_leaf(
          const FieldMask& mask, FieldMask& allvalid_mask,
          TraceViewSet& view_set, local::FieldMaskMap<InstanceView>& alt_views);
    protected:
      // Expression fields that are valid for these instances
      local::FieldMaskMap<IndexSpaceExpression> valid_exprs;
    };

    /**
     * \class InitializeCollectiveReduction
     * This class helps with aliasing analysis for recording collective
     * reduction views and figuring out where they are already valid
     */
    class InitializeCollectiveReduction
      : public CollectiveRefinementTree<InitializeCollectiveReduction>,
        public Heapify<InitializeCollectiveReduction, OPERATION_LIFETIME> {
    public:
      InitializeCollectiveReduction(
          AllreduceView* view, IndexSpaceExpression* expr);
      InitializeCollectiveReduction(
          std::vector<DistributedID>&& insts, IndexSpaceExpression* expr,
          InstanceView* view,
          const local::FieldMaskMap<IndexSpaceExpression>& remainders,
          const FieldMask& covered);
    public:
      InitializeCollectiveReduction* clone(
          InstanceView* view, const FieldMask& mask,
          std::vector<DistributedID>&& insts) const;
      void analyze(
          InstanceView* view, const FieldMask& mask,
          IndexSpaceExpression* expr);
      void analyze(
          InstanceView* view, const FieldMask& mask,
          const FieldMapView<IndexSpaceExpression>& exprs);
      // Check for ABA problem
      void visit_leaf(
          const FieldMask& mask, IndexSpaceExpression* expr, bool& failure);
      // Report out any fill operations that we need to perform
      void visit_leaf(
          const FieldMask& mask, InnerContext* context,
          UpdateAnalysis& analysis, CopyFillAggregator*& fill_aggregator,
          FillView* fill_view, RegionTreeID tid, EquivalenceSet* eq_set,
          DistributedID eq_did,
          std::map<
              unsigned,
              std::list<std::pair<InstanceView*, IndexSpaceExpression*> > >&
              reduction_instances);
    public:
      IndexSpaceExpression* const needed_expr;
      InstanceView* const view;
    protected:
      // Expressions for which we still need fill values
      local::FieldMaskMap<IndexSpaceExpression> remainder_exprs;
      FieldMask found_covered;
    };

    /**
     * \class EqSetTracker
     * This is an abstract class that provides an interface for
     * recording the equivalence sets that result from ray tracing
     * an equivalence set tree for a given index space expression.
     */
    class EqSetTracker {
    public:
      struct LgFinalizeEqSetsArgs : public LgTaskArgs<LgFinalizeEqSetsArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_FINALIZE_EQ_SETS_TASK_ID;
      public:
        LgFinalizeEqSetsArgs(void) = default;
        LgFinalizeEqSetsArgs(
            EqSetTracker* t, RtUserEvent c, InnerContext* enclosing,
            InnerContext* outermost, unsigned parent_req_index,
            IndexSpaceExpression* expr);
        void execute(void) const;
      public:
        EqSetTracker* tracker;
        RtUserEvent compute;
        InnerContext* enclosing;
        InnerContext* outermost;
        IndexSpaceExpression* expr;
        unsigned parent_req_index;
      };
    public:
      EqSetTracker(LocalLock& lock);
      virtual ~EqSetTracker(void);
    public:
      void record_equivalence_sets(
          InnerContext* context, const FieldMask& mask,
          op::FieldMaskMap<EquivalenceSet>& eq_sets,
          op::FieldMaskMap<EqKDTree>& to_create,
          op::map<EqKDTree*, Domain>& creation_rects,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
          op::FieldMaskMap<EqKDTree>& subscriptions, unsigned new_references,
          AddressSpaceID source, unsigned expected_responses,
          std::vector<RtEvent>& ready_events,
          const CollectiveMapping& target_mapping,
          const std::vector<EqSetTracker*>& targets,
          const AddressSpaceID creation_target_space);
      void record_output_subscriptions(
          AddressSpaceID source,
          local::FieldMaskMap<EqKDTree>& new_subscriptions);
    public:
      virtual void add_subscription_reference(unsigned count = 1) = 0;
      virtual bool remove_subscription_reference(unsigned count = 1) = 0;
    public:
      virtual RegionTreeID get_region_tree_id(void) const = 0;
      virtual IndexSpaceExpression* get_tracker_expression(void) const = 0;
      virtual ReferenceSource get_reference_source_kind(void) const = 0;
    public:
      unsigned invalidate_equivalence_sets(
          const FieldMask& mask, EqKDTree* tree, AddressSpaceID source,
          std::vector<RtEvent>& invalidated_events);
      void cancel_subscriptions(
          AddressSpaceID target, const FieldMapView<EqKDTree>& to_cancel,
          std::vector<RtEvent>* cancelled_events = nullptr);
      static void invalidate_subscriptions(
          EqKDTree* source,
          const MapView<AddressSpaceID, local::FieldMaskMap<EqSetTracker> >&
              subscribers,
          std::vector<RtEvent>& applied_events);
      void record_pending_equivalence_set(
          EquivalenceSet* set, const FieldMask& mask);
    protected:
      void record_subscriptions(
          AddressSpaceID source, const FieldMapView<EqKDTree>& new_subs);
      void record_creation_sets(
          op::FieldMaskMap<EqKDTree>& to_create,
          op::map<EqKDTree*, Domain>& creation_rects, AddressSpaceID source,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs);
      void extract_creation_sets(
          const FieldMask& mask,
          op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree> >& create_now,
          op::map<Domain, FieldMask>& create_now_rectangles,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs);
      void create_new_equivalence_sets(
          InnerContext* context, std::vector<RtEvent>& ready_events,
          op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree> >& create_now,
          op::map<Domain, FieldMask>& create_now_rectangles,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
          const CollectiveMapping& target_mapping,
          const std::vector<EqSetTracker*>& targets);
      struct SourceState : public FieldSet<Domain> {
      public:
        SourceState(void) : source_expr(nullptr), source_volume(0) { }
        SourceState(const FieldMask& m)
          : FieldSet(m), source_expr(nullptr), source_volume(0)
        { }
        ~SourceState(void);
      public:
        IndexSpaceExpression* get_expression(void) const;
        void set_expression(IndexSpaceExpression* expr);
      public:
        IndexSpaceExpression* source_expr;
        size_t source_volume;
      };
      bool check_for_congruent_source_equivalence_sets(
          FieldSet<Domain>& dest, std::vector<RtEvent>& ready_events,
          shrt::FieldMaskMap<EquivalenceSet>& created_sets,
          local::FieldMaskMap<EquivalenceSet>& unique_sources,
          op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree> >& create_now,
          local::map<EquivalenceSet*, local::list<SourceState> >&
              creation_sources,
          const CollectiveMapping& target_mapping,
          const std::vector<EqSetTracker*>& targets);
      EquivalenceSet* find_congruent_existing_equivalence_set(
          IndexSpaceExpression* expr, const FieldMask& mask,
          shrt::FieldMaskMap<EquivalenceSet>& created_sets,
          InnerContext* context);
      void extract_remote_notifications(
          const FieldMask& mask, AddressSpaceID local_space,
          op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree> >& create_now,
          op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree> >& to_notify);
      RtEvent initialize_new_equivalence_set(
          EquivalenceSet* set, const FieldMask& mask, bool filter_invalidations,
          local::map<EquivalenceSet*, local::list<SourceState> >&
              creation_sources);
      void finalize_equivalence_sets(
          RtUserEvent compute_event, InnerContext* enclosing,
          InnerContext* outermost, unsigned parent_req_index,
          IndexSpaceExpression* expr, UniqueID opid);
      void record_equivalence_sets(
          VersionInfo* version_info, const FieldMask& mask) const;
      void find_cancellations(
          const FieldMask& mask,
          local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree> >&
              to_cancel);
    protected:
      LocalLock& tracker_lock;
      // Member varialbes that are pointers are transient and only used in
      // building up the state for this equivalence set tracker, the non-pointer
      // member variables are the persistent ones that will likely live for
      // a long period of time to store data
      lng::FieldMaskMap<EquivalenceSet> equivalence_sets;
      // These are the EqKDTree objects that we are currently subscribed to
      // for different fields in each address space, this data mirrors the
      // same data structure in EqKDNode
      lng::map<AddressSpaceID, lng::FieldMaskMap<EqKDTree> >
          current_subscriptions;
      // Equivalence sets that are about to become part of the canonical
      // equivalence sets once the compute_equivalence_sets process completes
      shrt::FieldMaskMap<EquivalenceSet>* pending_equivalence_sets;
      // The created equivalence sets are similar to the pending equivalence
      // sets but they have been newly created and already have a
      // VERSION_MANAGER_REF on this local node
      shrt::FieldMaskMap<EquivalenceSet>* created_equivalence_sets;
      // User events marking when our current equivalence sets are ready
      shrt::map<RtUserEvent, FieldMask>* equivalence_sets_ready;
      // Version infos that need to be updated once equivalence sets are ready
      shrt::FieldMaskMap<VersionInfo>* waiting_infos;
      // Track whether there were any intermediate invalidations that occurred
      // while we were in the process of computing equivalence sets
      FieldMask pending_invalidations;
      // These all help with the creation of equivalence sets for which we
      // are the first request to access them
      op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree> >* creation_requests;
      op::map<Domain, FieldMask>* creation_rectangles;
      op::map<EquivalenceSet*, op::map<Domain, FieldMask> >* creation_sources;
      shrt::map<unsigned, FieldMask>* remaining_responses;
    };

    /**
     * \class EquivalenceSet
     * The equivalence set class tracks the physical state of a
     * set of points in a logical region for all the fields. There
     * is an owner node for the equivlance set that uses a ESI
     * protocol in order to manage local and remote copies of
     * the equivalence set for each of the different fields.
     * It's also possible for the equivalence set to be refined
     * into sub equivalence sets which then subsum it's responsibility.
     */
    class EquivalenceSet : public DistributedCollectable,
                           public Heapify<EquivalenceSet, CONTEXT_LIFETIME> {
    public:
      struct ReplicatedOwnerState
        : public Heapify<ReplicatedOwnerState, CONTEXT_LIFETIME> {
      public:
        ReplicatedOwnerState(bool valid);
      public:
        inline bool is_valid(void) const { return !ready.exists(); }
      public:
        std::vector<AddressSpaceID> children;
        RtUserEvent ready;
      };
    public:
      struct DeferMakeOwnerArgs : public LgTaskArgs<DeferMakeOwnerArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_MAKE_OWNER_TASK_ID;
      public:
        DeferMakeOwnerArgs(void) = default;
        DeferMakeOwnerArgs(EquivalenceSet* s, LamportClock clock)
          : LgTaskArgs<DeferMakeOwnerArgs>(false, true), set(s),
            lamport_clock(clock)
        { }
        void execute(void) const;
      public:
        EquivalenceSet* set;
        LamportClock lamport_clock;
      };
      struct DeferApplyStateArgs : public LgTaskArgs<DeferApplyStateArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_APPLY_STATE_TASK_ID;
        typedef shrt::map<
            IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView> >
            ExprLogicalViews;
        typedef std::map<
            unsigned,
            std::list<std::pair<InstanceView*, IndexSpaceExpression*> > >
            ExprReductionViews;
        typedef shrt::map<
            IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >
            ExprInstanceViews;
      public:
        DeferApplyStateArgs(void) = default;
        DeferApplyStateArgs(
            EquivalenceSet* set, bool forward,
            std::vector<RtEvent>& applied_events,
            ExprLogicalViews& valid_updates,
            shrt::FieldMaskMap<IndexSpaceExpression>& init_updates,
            shrt::FieldMaskMap<IndexSpaceExpression>& invalid_updates,
            ExprReductionViews& reduction_updates,
            ExprInstanceViews& restricted_updates,
            ExprInstanceViews& released_updates,
            shrt::FieldMaskMap<CopyFillGuard>& read_only_updates,
            shrt::FieldMaskMap<CopyFillGuard>& reduction_fill_updates,
            TraceViewSet* precondition_updates,
            TraceViewSet* anticondition_updates,
            TraceViewSet* postcondition_updates,
            shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates,
            shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
            shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks);
        void execute(void) const;
        void release_references(void) const;
      public:
        EquivalenceSet* set;
        ExprLogicalViews* valid_updates;
        shrt::FieldMaskMap<IndexSpaceExpression>* initialized_updates;
        shrt::FieldMaskMap<IndexSpaceExpression>* invalidated_updates;
        ExprReductionViews* reduction_updates;
        ExprInstanceViews* restricted_updates;
        ExprInstanceViews* released_updates;
        shrt::FieldMaskMap<CopyFillGuard>* read_only_updates;
        shrt::FieldMaskMap<CopyFillGuard>* reduction_fill_updates;
        TraceViewSet* precondition_updates;
        TraceViewSet* anticondition_updates;
        TraceViewSet* postcondition_updates;
        shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates;
        shrt::map<LogicalView*, LamportClock>* view_lamport_clocks;
        shrt::map<PhysicalManager*, LamportClock>* inst_lamport_clocks;
        std::set<IndexSpaceExpression*>* expr_references;
        RtUserEvent done_event;
        bool forward_to_owner;
      };
    public:
      EquivalenceSet(
          DistributedID did, AddressSpaceID logical_owner,
          IndexSpaceExpression* expr, RegionTreeID tid, InnerContext* context,
          bool register_now, CollectiveMapping* mapping = nullptr,
          bool replicate_logical_owner = false);
      EquivalenceSet(const EquivalenceSet& rhs) = delete;
      virtual ~EquivalenceSet(void);
    public:
      EquivalenceSet& operator=(const EquivalenceSet& rhs) = delete;
      // Must be called while holding the lock
      inline bool is_logical_owner(void) const
      {
        return (local_space == logical_owner_space);
      }
    public:
      // From distributed collectable
      virtual void notify_invalid(void) { std::abort(); }
      virtual void notify_local(void) override;
    public:
      // Analysis methods
      void initialize_set(
          const RegionUsage& usage, const FieldMask& user_mask,
          const bool restricted, const InstanceSet& sources,
          const std::vector<IndividualView*>& corresponding);
      void analyze(
          PhysicalAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, FieldMask traversal_mask,
          std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
          const bool already_deferred);
      void find_valid_instances(
          ValidInstAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& user_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void find_invalid_instances(
          InvalidInstAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& user_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void find_antivalid_instances(
          AntivalidInstAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& user_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void update_set(
          UpdateAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& user_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void acquire_restrictions(
          AcquireAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& acquire_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void release_restrictions(
          ReleaseAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& release_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void issue_across_copies(
          CopyAcrossAnalysis& analysis, const FieldMask& src_mask,
          IndexSpaceExpression* expr, const bool expr_covers,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void overwrite_set(
          OverwriteAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& overwrite_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
      void filter_set(
          FilterAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& filter_mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false);
    public:
      void initialize_collective_references(unsigned local_valid_refs);
      void remove_read_only_guard(CopyFillGuard* guard);
      void remove_reduction_fill_guard(CopyFillGuard* guard);
      void clone_from(
          EquivalenceSet* src, const FieldMask& clone_mask,
          IndexSpaceExpression* clone_expr, const bool record_invalidate,
          std::vector<RtEvent>& applied_events);
      bool filter_partial_invalidations(
          const FieldMask& mask, RtUserEvent& filtered);
      void make_owner(
          LamportClock lamport_clock,
          RtEvent precondition = RtEvent::NO_RT_EVENT);
    public:
      // View that was read by a task during a trace
      void update_tracing_read_only_view(
          InstanceView* view, IndexSpaceExpression* expr,
          const FieldMask& view_mask);
      // View that was only written to by a task during a trace
      void update_tracing_write_discard_view(
          LogicalView* view, IndexSpaceExpression* expr,
          const FieldMask& view_mask);
      // View that was read and written to by a task during a trace
      void update_tracing_read_write_view(
          InstanceView* view, IndexSpaceExpression* expr,
          const FieldMask& view_mask);
      // View that was reduced by a task during a trace
      void update_tracing_reduced_view(
          InstanceView* view, IndexSpaceExpression* expr,
          const FieldMask& view_mask);
      // View that was filled during a trace
      void update_tracing_fill_views(
          FillView* src_view, InstanceView* dst_view,
          IndexSpaceExpression* expr, const FieldMask& fill_mask, bool across);
      // Views that were copied between during a trace
      void update_tracing_copy_views(
          LogicalView* src_view, InstanceView* dst_view,
          IndexSpaceExpression* expr, const FieldMask& copy_mask, bool across);
      // Views that were reduced between during a trace
      void update_tracing_reduction_views(
          InstanceView* src_view, InstanceView* dst_view,
          IndexSpaceExpression* expr, const FieldMask& copy_mask, bool across);
      // Invalidate restricted views that shouldn't be postconditions
      void invalidate_tracing_restricted_views(
          const FieldMapView<InstanceView>& restricted_views,
          IndexSpaceExpression* expr, FieldMask& restricted_mask);
      RtEvent capture_trace_conditions(
          PhysicalTemplate* target, AddressSpaceID target_space,
          unsigned parent_req_index, std::atomic<unsigned>* result,
          RtUserEvent ready_event = RtUserEvent::NO_RT_USER_EVENT);
      AddressSpaceID select_collective_trace_capture_space(void);
    protected:
      void defer_analysis(
          AutoTryLock& eq, PhysicalAnalysis& analysis, const FieldMask& mask,
          std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
          const bool already_deferred);
      inline RtEvent chain_deferral_events(RtUserEvent deferral_event)
      {
        RtEvent continuation_pre;
        continuation_pre.id =
            next_deferral_precondition.exchange(deferral_event.id);
        return continuation_pre;
      }
      bool is_remote_analysis(
          PhysicalAnalysis& analysis, FieldMask& traversal_mask,
          std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
          const bool expr_covers);
    protected:
      template<typename T>
      void check_for_uninitialized_data(
          T& analysis, IndexSpaceExpression* expr, const bool expr_cover,
          FieldMask uninit, std::set<RtEvent>& applied_events) const;
      void update_initialized_data(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& user_mask);
      template<typename T>
      void record_instances(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& record_mask, const FieldMapView<T>& new_views);
      template<typename T>
      void record_unrestricted_instances(
          IndexSpaceExpression* expr, const bool expr_covers,
          FieldMask record_mask, const FieldMapView<T>& new_views);
      bool record_partial_valid_instance(
          LogicalView* instance, IndexSpaceExpression* expr,
          FieldMask valid_mask, bool check_total_valid = true);
      void filter_valid_instance(
          InstanceView* to_filter, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& filter_mask);
      void filter_valid_instances(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& filter_mask,
          std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove =
              nullptr,
          std::map<LogicalView*, unsigned>* view_refs_to_remove = nullptr);
      void filter_unrestricted_instances(
          IndexSpaceExpression* expr, const bool expr_covers,
          FieldMask filter_mask);
      void filter_reduction_instances(
          IndexSpaceExpression* expr, const bool covers, const FieldMask& mask,
          std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove =
              nullptr,
          std::map<LogicalView*, unsigned>* view_refs_to_remove = nullptr);
      void update_set_internal(
          CopyFillAggregator*& input_aggregator, CopyFillGuard* previous_guard,
          PhysicalAnalysis* analysis, const RegionUsage& usage,
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& user_mask,
          const std::vector<PhysicalManager*>& targets,
          const VectorView<op::FieldMaskMap<InstanceView> >& target_views,
          const std::vector<IndividualView*>& source_views,
          const PhysicalTraceInfo& trace_info, const bool record_valid,
          const bool record_release = false);
      void make_instances_valid(
          CopyFillAggregator*& aggregator, CopyFillGuard* previous_guard,
          PhysicalAnalysis* analysis, const bool track_events,
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& update_mask,
          const std::vector<PhysicalManager*>& targets,
          const VectorView<op::FieldMaskMap<InstanceView> >& target_views,
          const std::vector<IndividualView*>& source_views,
          const PhysicalTraceInfo& trace_info, const bool skip_check = false,
          const ReductionOpID redop = 0,
          CopyAcrossHelper* across_helper = nullptr);
      void issue_update_copies_and_fills(
          InstanceView* target, PhysicalManager* target_manager,
          const std::vector<IndividualView*>& source_views,
          CopyFillAggregator*& aggregator, CopyFillGuard* previous_guard,
          PhysicalAnalysis* analysis, const bool track_events,
          IndexSpaceExpression* expr, const bool expr_covers,
          FieldMask update_mask, const PhysicalTraceInfo& trace_info,
          const ReductionOpID redop, CopyAcrossHelper* across_helper);
      void record_reductions(
          UpdateAnalysis& analysis, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& user_mask);
      void apply_reductions(
          const std::vector<PhysicalManager*>& targets,
          const VectorView<op::FieldMaskMap<InstanceView> >& target_views,
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& reduction_mask, CopyFillAggregator*& aggregator,
          CopyFillGuard* previous_guard, PhysicalAnalysis* analysis,
          const bool track_events, const PhysicalTraceInfo& trace_info,
          local::FieldMaskMap<IndexSpaceExpression>* applied_exprs,
          CopyAcrossHelper* across_helper = nullptr);
      void apply_restricted_reductions(
          const FieldMapView<InstanceView>& reduction_targets,
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& reduction_mask, CopyFillAggregator*& aggregator,
          CopyFillGuard* previous_guard, PhysicalAnalysis* analysis,
          const bool track_events, const PhysicalTraceInfo& trace_info,
          local::FieldMaskMap<IndexSpaceExpression>* applied_exprs);
      void apply_reduction(
          InstanceView* target, PhysicalManager* target_manager,
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& reduction_mask, CopyFillAggregator*& aggregator,
          CopyFillGuard* previous_guard, PhysicalAnalysis* analysis,
          const bool track_events, const PhysicalTraceInfo& trace_info,
          local::FieldMaskMap<IndexSpaceExpression>* applied_exprs,
          CopyAcrossHelper* across_helper);
      void copy_out(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& restricted_mask,
          const std::vector<PhysicalManager*>& target_instances,
          const VectorView<op::FieldMaskMap<InstanceView> >& target_views,
          PhysicalAnalysis* analysis, const PhysicalTraceInfo& trace_info,
          CopyFillAggregator*& aggregator);
      template<typename T>
      void copy_out(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& restricted_mask, const FieldMapView<T>& src_views,
          PhysicalAnalysis* analysis, const PhysicalTraceInfo& trace_info,
          CopyFillAggregator*& aggregator);
      void predicate_fill_all(
          FillView* fill, const FieldMask& fill_mask,
          IndexSpaceExpression* expr, const bool expr_covers,
          const PredEvent true_guard, const PredEvent false_guard,
          PhysicalAnalysis* analysis, const PhysicalTraceInfo& trace_info,
          CopyFillAggregator*& aggregator);
      void record_restriction(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& restrict_mask, InstanceView* restricted_view);
      void update_reductions(
          const unsigned fidx,
          std::list<std::pair<InstanceView*, IndexSpaceExpression*> >& updates);
      void update_released(
          IndexSpaceExpression* expr, const bool expr_covers,
          shrt::FieldMaskMap<InstanceView>& updates);
      void filter_initialized_data(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& filter_mask,
          std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove =
              nullptr);
      void filter_restricted_instances(
          IndexSpaceExpression* expr, const bool covers, const FieldMask& mask,
          std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove =
              nullptr,
          std::map<LogicalView*, unsigned>* view_refs_to_remove = nullptr);
      void filter_released_instances(
          IndexSpaceExpression* expr, const bool covers, const FieldMask& mask,
          std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove =
              nullptr,
          std::map<LogicalView*, unsigned>* view_refs_to_remove = nullptr);
      bool find_fully_valid_fields(
          InstanceView* inst, FieldMask& inst_mask, IndexSpaceExpression* expr,
          const bool expr_covers) const;
      bool find_partial_valid_fields(
          InstanceView* inst, FieldMask& inst_mask, IndexSpaceExpression* expr,
          const bool expr_covers,
          local::FieldMaskMap<IndexSpaceExpression>& partial_valid_exprs) const;
    public:
      void send_equivalence_set(AddressSpaceID target);
      void check_for_migration(
          PhysicalAnalysis& analysis, std::set<RtEvent>& applied_events);
      void update_owner(const AddressSpaceID new_logical_owner);
      bool replicate_logical_owner_space(
          AddressSpaceID source, const CollectiveMapping* mapping,
          bool need_lock);
      void process_replication_response(AddressSpaceID owner);
    public:
      void pack_state(
          Serializer& rez, const AddressSpaceID target,
          DistributedID target_did, IndexSpaceExpression* target_expr,
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& mask, const bool pack_guards,
          const bool pack_invalidates);
      void unpack_state_and_apply(
          Deserializer& derez, const AddressSpaceID source,
          std::vector<RtEvent>& ready_events, const bool forward_to_owner);
      void invalidate_state(
          IndexSpaceExpression* expr, const bool expr_covers,
          const FieldMask& mask, bool record_invalidation);
      void clone_to_local(
          EquivalenceSet* dst, FieldMask mask, IndexSpaceExpression* clone_expr,
          std::vector<RtEvent>& applied_events, const bool record_invalidate,
          const bool need_dst_lock = true);
      void clone_to_remote(
          DistributedID target, AddressSpaceID target_space,
          IndexSpaceExpression* target_expr, IndexSpaceExpression* overlap,
          FieldMask mask, std::vector<RtEvent>& applied_events,
          const bool record_invalidate);
      void find_overlap_updates(
          IndexSpaceExpression* overlap, const bool overlap_covers,
          const FieldMask& mask, const bool find_invalidates,
          shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView> >&
              valid_updates,
          shrt::FieldMaskMap<IndexSpaceExpression>& initialized_updates,
          shrt::FieldMaskMap<IndexSpaceExpression>& invalidated_updates,
          std::map<
              unsigned,
              std::list<std::pair<InstanceView*, IndexSpaceExpression*> > >&
              reduction_updates,
          shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >&
              restricted_updates,
          shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >&
              released_updates,
          shrt::FieldMaskMap<CopyFillGuard>* read_only_guard_updates,
          shrt::FieldMaskMap<CopyFillGuard>* reduction_fill_guard_updates,
          TraceViewSet*& precondition_updates,
          TraceViewSet*& anticondition_updates,
          TraceViewSet*& postcondition_updates,
          shrt::FieldMaskMap<IndexSpaceExpression>*& dirty_updates,
          DistributedID target, IndexSpaceExpression* target_expr) const;
      void apply_state(
          shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView> >&
              valid_updates,
          shrt::FieldMaskMap<IndexSpaceExpression>& initialized_updates,
          shrt::FieldMaskMap<IndexSpaceExpression>& invalidated_updates,
          std::map<
              unsigned,
              std::list<std::pair<InstanceView*, IndexSpaceExpression*> > >&
              reduction_updates,
          shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >&
              restricted_updates,
          shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >&
              released_updates,
          TraceViewSet* precondition_updates,
          TraceViewSet* anticondition_updates,
          TraceViewSet* postcondition_updates,
          shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates,
          shrt::FieldMaskMap<CopyFillGuard>* read_only_guard_updates,
          shrt::FieldMaskMap<CopyFillGuard>* reduction_fill_guard_updates,
          std::vector<RtEvent>& applied_events,
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks,
          const bool needs_lock, const bool forward_to_owner,
          const bool unpack_tracing_references);
      static void pack_updates(
          Serializer& rez, const AddressSpaceID target,
          const MapView<
              IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView> >&
              valid_updates,
          const FieldMapView<IndexSpaceExpression>& initialized_updates,
          const FieldMapView<IndexSpaceExpression>& invalidated_updates,
          const std::map<
              unsigned,
              std::list<std::pair<InstanceView*, IndexSpaceExpression*> > >&
              reduction_updates,
          const MapView<
              IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >&
              restricted_updates,
          const MapView<
              IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >&
              released_updates,
          const shrt::FieldMaskMap<CopyFillGuard>* read_only_updates,
          const shrt::FieldMaskMap<CopyFillGuard>* reduction_fill_updates,
          const TraceViewSet* precondition_updates,
          const TraceViewSet* anticondition_updates,
          const TraceViewSet* postcondition_updates,
          const shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates,
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks,
          const bool pack_tracing_references);
    public:
      // Note this context refers to the context from which the views are
      // created in. Normally this is the same as the context in which the
      // equivalence set is made, but it can be from an ancestor task
      // higher up in the task tree in the presencce of virtual mappings.
      // It's crucial to correctness that all views stored in an equivalence
      // set come from the same context.
      InnerContext* const context;
      IndexSpaceExpression* const set_expr;
      const RegionTreeID tree_id;
    protected:
      mutable LocalLock eq_lock;
      // This is the physical state of the equivalence set
      shrt::FieldMaskMap<LogicalView> total_valid_instances;
      typedef shrt::map<LogicalView*, shrt::FieldMaskMap<IndexSpaceExpression> >
          ViewExprMaskSets;
      ViewExprMaskSets partial_valid_instances;
      FieldMask partial_valid_fields;
      // Expressions and fields that have valid data
      lng::FieldMaskMap<IndexSpaceExpression> initialized_data;
      // Expressions for fields that have been invalidated and no longer
      // contain valid meta-data, even though the set_expr encompasses
      // them. This occurs when we have partial invalidations of an equivalence
      // set and therefore we need to record this information
      lng::FieldMaskMap<IndexSpaceExpression> partial_invalidations;
      // Reductions always need to be applied in order so keep them in order
      std::map<
          unsigned /*fidx*/,
          std::list<std::pair<InstanceView*, IndexSpaceExpression*> > >
          reduction_instances;
      FieldMask reduction_fields;
      // The list of expressions with the single instance for each
      // field that represents the restriction of that expression
      typedef shrt::map<
          IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView> >
          ExprViewMaskSets;
      ExprViewMaskSets restricted_instances;
      // Summary of any field that has a restriction
      FieldMask restricted_fields;
      // List of instances that were restricted, but have been acquired
      ExprViewMaskSets released_instances;
      // The names of any collective views we have resident in the
      // total or partial valid instances. This allows us to do faster
      // checks for aliasing when we just have a bunch of individual views
      // and no collective views (the common case). Once you start adding
      // in collective views then some of the analyses can get more
      // expensive but that is the trade-off with using collective views.
      lng::FieldMaskMap<CollectiveView> collective_instances;
    protected:
      // Tracing state for this equivalence set
      TraceViewSet* tracing_preconditions;
      TraceViewSet* tracing_anticonditions;
      TraceViewSet* tracing_postconditions;
      shrt::FieldMaskMap<IndexSpaceExpression>* tracing_dirty_fields;
    protected:
      // This tracks the most recent copy-fill aggregator for each field in
      // read-only cases so that reads the depend on each other are ordered
      shrt::FieldMaskMap<CopyFillGuard> read_only_guards;
      // This tracks the most recent fill-aggregator for each field in reduction
      // cases so that reductions that depend on the same fill are ordered
      shrt::FieldMaskMap<CopyFillGuard> reduction_fill_guards;
      // An event to order to deferral tasks
      std::atomic<Realm::Event::id_t> next_deferral_precondition;
    protected:
      // This node is the node which contains the valid state data
      AddressSpaceID logical_owner_space;
      // To support control-replicated mapping of common regions in index space
      // launches or collective instance mappings, we need to have a way to
      // force all analyses to see the same value for the logical owner space
      // for the equivalence set such that we can then have a single analysis
      // traverse the equivalence set without requiring communication between
      // the analyses to determine which one does the traversal. The
      // replicated_owner_state defines a spanning tree of the existing
      // copies of the equivalence sets that all share knowledge of the
      // logical owner space.
      ReplicatedOwnerState* replicated_owner_state;
    protected:
      // Uses these for determining when we should do migration
      // There is an implicit assumption here that equivalence sets
      // are only used be a small number of nodes that is less than
      // the samples per migration count, if it ever exceeds this
      // then we'll issue a warning
      static constexpr unsigned SAMPLES_PER_MIGRATION_TEST = 64;
      // How many total epochs we want to remember
      static constexpr unsigned MIGRATION_EPOCHS = 2;
      std::vector<std::pair<AddressSpaceID, unsigned> >
          user_samples[MIGRATION_EPOCHS];
      unsigned migration_index;
      unsigned sample_count;
    public:
      // These magic numbers help to protect Legion from making lots of really
      // tiny equivalence sets. The first one sets a general guideline for how
      // small equivalence sets can be (no smaller than 16K points). The second
      // number allows for an exception to that rule in the cases where the
      // ratio between the volumes of a parent region and child region is no
      // more than 64 to allow some limited refinement of equivalence sets
      // on small degrees of parallelism cases.
      static constexpr size_t MINIMUM_SIZE = 16384;
      static constexpr size_t MINIMUM_RATIO = 64;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_EQUIVALENCE_SET_H__
