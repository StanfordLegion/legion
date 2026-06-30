/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_INDIVIDUAL_VIEW_H__
#define __LEGION_INDIVIDUAL_VIEW_H__

#include "legion/views/instance.h"
#include "legion/nodes/expression.h"
#include "legion/tracing/recording.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    template<typename T>
    class ViewComparator {
    public:
      using is_transparent = std::true_type;
      inline bool operator()(const T* lhs, const T* rhs) const
      {
        return (lhs->node->color < rhs->node->color);
      }
      inline bool operator()(const T* lhs, const LegionColor& rhs) const
      {
        return (lhs->node->color < rhs);
      }
      inline bool operator()(const LegionColor& lhs, const T* rhs) const
      {
        return (lhs < rhs->node->color);
      }
    };

    class NodeView : public Collectable {
    public:
      NodeView(IndexTreeNode* node, IndividualView* view);
      virtual ~NodeView(void);
    public:
      virtual bool is_empty(void) const = 0;
      virtual void invalidate_users(void) = 0;
      virtual void find_last_users(
          const RegionUsage& usage, IndexSpaceExpression* expr,
          const bool expr_dominates, const FieldMask& mask,
          std::set<ApEvent>& last_events) const = 0;
      virtual bool find_user_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* user_expr,
          const bool expr_dominates, const FieldMask& user_mask,
          ApEvent term_event, UniqueID op_id, unsigned index,
          std::set<ApEvent>& preconditions, const bool trace_recording) = 0;
      virtual bool find_copy_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* copy_expr,
          const bool expr_dominates, const FieldMask& copy_mask, UniqueID op_id,
          unsigned index, std::set<ApEvent>& preconditions,
          const bool trace_recording) = 0;
      virtual void insert_child(
          NodeView* child, const FieldMask& child_mask) = 0;
      virtual void insert_user(
          PhysicalUser* user, const FieldMask& user_mask,
          local::vector<LegionColor>& path, AutoLock& parent_lock) = 0;
    public:
      IndexTreeNode* const tree_node;
      IndividualView* const view;
    };

    /**
     * \class SpaceView
     * Users for the an individual view on a particular index space node
     */
    class SpaceView : public NodeView {
    public:
      SpaceView(IndexSpaceNode* node, IndividualView* view);
      SpaceView(const SpaceView&) = delete;
      SpaceView(SpaceView&&) = delete;
      ~SpaceView(void);
    public:
      SpaceView& operator=(const SpaceView&) = delete;
      SpaceView& operator=(SpaceView&&) = delete;
    public:
      bool dominated_by(IndexSpaceExpression* expr) const;
      virtual bool is_empty(void) const override;
      virtual void invalidate_users(void) override;
      virtual void find_last_users(
          const RegionUsage& usage, IndexSpaceExpression* expr,
          const bool expr_dominates, const FieldMask& mask,
          std::set<ApEvent>& last_events) const override;
      virtual bool find_user_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* user_expr,
          const bool expr_dominates, const FieldMask& user_mask,
          ApEvent term_event, UniqueID op_id, unsigned index,
          std::set<ApEvent>& preconditions,
          const bool trace_recording) override;
      virtual bool find_copy_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* copy_expr,
          const bool expr_dominates, const FieldMask& copy_mask, UniqueID op_id,
          unsigned index, std::set<ApEvent>& preconditions,
          const bool trace_recording) override;
      virtual void insert_child(
          NodeView* child, const FieldMask& child_mask) override;
      virtual void insert_user(
          PhysicalUser* user, const FieldMask& user_mask,
          local::vector<LegionColor>& path, AutoLock& parent_lock) override;
    protected:
      void find_subviews_to_traverse(
          const FieldMask& mask,
          local::FieldMaskMap<PartitionView>& to_traverse) const;
      void find_current_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* user_expr, ApEvent term_event,
          const UniqueID op_id, const unsigned index, const bool user_covers,
          std::set<ApEvent>& preconditions,
          local::set<PhysicalUser*>& dead_users,
          local::FieldMaskMap<PhysicalUser>& filter_users, FieldMask& observed,
          FieldMask& non_dominated, const bool trace_recording,
          const bool copy_user);
      void find_previous_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* user_expr, ApEvent term_event,
          const UniqueID op_id, const unsigned index, const bool user_covers,
          std::set<ApEvent>& preconditions,
          local::set<PhysicalUser*>& dead_users, const bool trace_recording,
          const bool copy_user);
      void find_previous_filter_users(
          const FieldMask& dominated_mask,
          local::FieldMaskMap<PhysicalUser>& filter_users);
      // Overloads for find_last_users
      void find_current_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* expr, const bool expr_covers,
          std::set<ApEvent>& last_events, FieldMask& observed,
          FieldMask& non_dominated) const;
      void find_previous_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* expr, const bool expr_covers,
          std::set<ApEvent>& last_events) const;
      inline bool has_local_precondition(
          PhysicalUser* prev_user, const RegionUsage& next_user,
          IndexSpaceExpression* user_expr, const UniqueID op_id,
          const unsigned index, const bool user_covers, const bool copy_user,
          bool* dominates = nullptr) const;
    protected:
      void filter_dead_users(const local::set<PhysicalUser*>& dead_users);
      void filter_current_users(const FieldMapView<PhysicalUser>& to_filter);
      void filter_previous_users(const FieldMapView<PhysicalUser>& to_filter);
      static void verify_current_to_filter(
          const FieldMask& dominated,
          local::FieldMaskMap<PhysicalUser>& current_to_filter);
    public:
      IndexSpaceNode* const node;
    protected:
      mutable LocalLock view_lock;
      // There are three operations that are done on materialized views
      // 1. iterate over all the users for use analysis
      // 2. garbage collection to remove old users for an event
      // 3. send updates for a certain set of fields
      // The first and last both iterate over the current and previous
      // user sets, while the second one needs to find specific events.
      // Therefore we store the current and previous sets as maps to
      // users indexed by events. Iterating over the maps are no worse
      // than iterating over lists (for arbitrary insertion and deletion)
      // and will provide fast indexing for removing items. We used to
      // store users in current and previous epochs similar to logical
      // analysis, but have since switched over to storing readers and
      // writers that are not filtered as part of analysis. This let's
      // us perform more analysis in parallel since we'll only need to
      // hold locks in read-only mode prevent user fragmentation. It also
      // deals better with the common case which are higher views in
      // the view tree that less frequently filter their sub-users.
      shrt::FieldMaskMap<PhysicalUser> current_epoch_users;
      shrt::FieldMaskMap<PhysicalUser> previous_epoch_users;
      lng::FieldMaskMap<PartitionView, ViewComparator<PartitionView> > subviews;
    };

    /**
     * \class PartitionView
     * Tracking which child nodes have users below in the tree
     */
    class PartitionView : public NodeView {
    public:
      PartitionView(IndexPartNode* node, IndividualView* view);
      PartitionView(const PartitionView&) = delete;
      PartitionView(PartitionView&&) = delete;
      virtual ~PartitionView(void);
    public:
      PartitionView& operator=(const PartitionView&) = delete;
      PartitionView& operator=(PartitionView&&) = delete;
    public:
      virtual bool is_empty(void) const override;
      virtual void invalidate_users(void) override;
      virtual void find_last_users(
          const RegionUsage& usage, IndexSpaceExpression* expr,
          const bool expr_dominates, const FieldMask& mask,
          std::set<ApEvent>& last_events) const override;
      virtual bool find_user_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* user_expr,
          const bool expr_dominates, const FieldMask& user_mask,
          ApEvent term_event, UniqueID op_id, unsigned index,
          std::set<ApEvent>& preconditions,
          const bool trace_recording) override;
      virtual bool find_copy_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* copy_expr,
          const bool expr_dominates, const FieldMask& copy_mask, UniqueID op_id,
          unsigned index, std::set<ApEvent>& preconditions,
          const bool trace_recording) override;
      virtual void insert_child(
          NodeView* child, const FieldMask& child_mask) override;
      virtual void insert_user(
          PhysicalUser* user, const FieldMask& user_mask,
          local::vector<LegionColor>& path, AutoLock& parent_lock) override;
    protected:
      bool find_subviews_to_traverse(
          IndexSpaceExpression* expr, bool expr_dominates,
          const FieldMask& mask,
          local::FieldMaskMap<SpaceView>& to_traverse) const;
    public:
      IndexPartNode* const node;
    protected:
      mutable LocalLock view_lock;
      // Keep these sorted by their color so we can easily look them up
      lng::FieldMaskMap<SpaceView, ViewComparator<SpaceView> > subviews;
    };

    /**
     * \class IndividualView
     * This class provides an abstract base class for any kind of view
     * that only represents an individual physical instance.
     */
    class IndividualView : public InstanceView {
    public:
      IndividualView(
          DistributedID did, PhysicalManager* man, AddressSpaceID logical_owner,
          bool register_now, CollectiveMapping* mapping);
      virtual ~IndividualView(void);
    public:
      inline bool is_logical_owner(void) const
      {
        return (local_space == logical_owner);
      }
      inline PhysicalManager* get_manager(void) const { return manager; }
    public:
      virtual AddressSpaceID get_analysis_space(
          PhysicalManager* inst) const override;
      virtual bool aliases(InstanceView* other) const override;
    public:
      // Reference counting state change functions
      virtual void notify_local(void) override;
      virtual void notify_valid(void) override;
      virtual bool notify_invalid(void) override;
    public:
      virtual void pack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
          override;
      virtual void unpack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
          override;
    public:
      virtual ApEvent fill_from(
          FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const FieldMask& fill_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool fill_restricted, const bool need_valid_return) override;
      virtual ApEvent copy_from(
          InstanceView* src_view, ApEvent precondition,
          PredEvent predicate_guard, ReductionOpID redop,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const FieldMask& copy_mask,
          PhysicalManager* src_point, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool copy_restricted, const bool need_valid_return) override;
      virtual ApEvent register_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, ApEvent term_event, PhysicalManager* target,
          CollectiveMapping* collective_mapping,
          size_t local_collective_arrivals,
          std::vector<RtEvent>& registered_events,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, const AddressSpaceID source,
          const bool symbolic = false) override;
    public:
      void add_initial_user(
          ApEvent term_event, const RegionUsage& usage,
          const FieldMask& user_mask, IndexSpaceNode* expr,
          const UniqueID op_id, const unsigned index);
      ApEvent find_copy_preconditions(
          bool reading, ReductionOpID redop, const FieldMask& copy_mask,
          IndexSpaceExpression* copy_expr, UniqueID op_id, unsigned index,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info);
      void add_copy_user(
          bool reading, ReductionOpID redop, ApEvent done_event,
          const FieldMask& copy_mask, IndexSpaceExpression* copy_expr,
          IndexSpace upper_bound, UniqueID op_id, unsigned index,
          std::set<RtEvent>& applied_events, const bool trace_recording,
          const AddressSpaceID source);
      void find_last_users(
          PhysicalManager* target, std::set<ApEvent>& events,
          const RegionUsage& usage, const FieldMask& mask,
          IndexSpaceExpression* user_expr, std::vector<RtEvent>& applied) const;
    public:
      void pack_fields(
          Serializer& rez, const std::vector<CopySrcDstField>& fields) const;
      void find_atomic_reservations(
          const FieldMask& mask, Operation* op, const unsigned index,
          bool exclusive);
      void find_field_reservations(
          const FieldMask& mask, std::vector<Reservation>& results);
      RtEvent find_field_reservations(
          const FieldMask& mask, std::vector<Reservation>* results,
          AddressSpaceID source,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
      void update_field_reservations(
          const FieldMask& mask, const std::vector<Reservation>& rsrvs);
    public:
      void register_collective_analysis(
          const CollectiveView* source, CollectiveAnalysis* analysis);
      CollectiveAnalysis* find_collective_analysis(
          size_t context_index, unsigned region_index, IndexSpace match_space);
      void unregister_collective_analysis(
          const CollectiveView* source, size_t context_index,
          unsigned region_index, IndexSpace match_space);
    protected:
      void add_internal_task_user(
          const RegionUsage& usage, IndexSpaceNode* user_expr,
          const FieldMask& user_mask, ApEvent term_event, UniqueID op_id,
          const unsigned index);
      void add_internal_node_user(
          PhysicalUser* user, const FieldMask& user_mask,
          IndexSpaceNode* user_expr);
      ApEvent register_collective_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, ApEvent term_event, PhysicalManager* target,
          CollectiveMapping* analysis_mapping, size_t local_collective_arrivals,
          std::vector<RtEvent>& registered_events,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, const bool symbolic);
    public:
      void process_collective_user_registration(
          const size_t op_ctx_index, const unsigned index,
          const IndexSpace match_space, const AddressSpaceID origin,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* analysis_mapping, ApEvent remote_term_event,
          ApUserEvent remote_ready_event, RtUserEvent remote_registered,
          std::set<RtEvent>& applied_events);
    public:
      PhysicalManager* const manager;
      // This is the owner space for the purpose of logical analysis
      // If you ever make this non-const then be sure to update the
      // code in register_collective_user
      const AddressSpaceID logical_owner;
    protected:
      // The idea behind the roots data structure is to track the minimum
      // number of index space tree nodes that need to be traversed for
      // doing dependence analysis. Each user is recorded in a node associated
      // with a particular index space. When multiple nodes in the tree share
      // a parent we merge them together into a subtree with a root of the
      // parent. We track all the subtrees that have users in them. When all
      // the users in a subtree are dominated or finish executing then we can
      // prune the subtree. The idea here is to only have memory usage
      // propportional to the number of logical regions actually used on the
      // instance. Traversing the sub-trees will leverage any acceleration data
      // structures that exist for partitions. The alternative is to have a
      // general acceleration data structure that handles all possible
      // expressions. This would be subject to over-decomposition (e.g. when
      // partitioning by rows and then partitioning by columns, causing O(N^2)
      // leaves instead of O(2N)). We've learned our lession from older
      // implementations of equivalence sets on that front. The trade-off is
      // that we cannot actively prune domniated users as efficiently and might
      // need to do more analysis and have larger fan-in event mergers as a
      // result. The results of the dependence analysis though will still be
      // sound and precise.
      lng::FieldMaskMap<NodeView> roots;
      std::map<unsigned, Reservation> view_reservations;
    protected:
      // This is an infrequently used data structure for handling collective
      // register user calls on individual managers that occurs with certain
      // operation in control replicated contexts
      struct UserRendezvous {
        UserRendezvous(void)
          : remaining_local_arrivals(0), remaining_remote_arrivals(0),
            trace_info(nullptr), analysis_mapping(nullptr), mask(nullptr),
            expr(nullptr), op_id(0), symbolic(false), local_initialized(false)
        { }
        // event for when local instances can be used
        ApUserEvent ready_event;
        // remote ready events to trigger
        std::map<ApUserEvent, PhysicalTraceInfo*> remote_ready_events;
        // all the local term events
        std::vector<ApEvent> term_events;
        // event that marks when all registrations are done
        RtUserEvent registered;
        // event for when any local effects are applied
        RtUserEvent applied;
        // Counts of remaining notficiations before registration
        unsigned remaining_local_arrivals;
        unsigned remaining_remote_arrivals;
        // PhysicalTraceInfo that made the ready_event and should trigger it
        PhysicalTraceInfo* trace_info;
        CollectiveMapping* analysis_mapping;
        // Arguments for performing the local registration
        RegionUsage usage;
        HeapifyBox<FieldMask, OPERATION_LIFETIME>* mask;
        IndexSpaceNode* expr;
        UniqueID op_id;
        bool symbolic;
        bool local_initialized;
      };
      std::map<RendezvousKey, UserRendezvous> rendezvous_users;
    protected:
      // This is actually quite important!
      // Normally each collective analysis is associated with a specific
      // collective view. However the copies done by that analysis might
      // only be occurring on collective views that are a subset of the
      // collective view for the analysis. Therefore we register the analyses
      // with the individual views so that they can be found by any copies
      struct RegisteredAnalysis {
      public:
        CollectiveAnalysis* analysis;
        RtUserEvent ready;
        // We need to deduplicate across views that are performing
        // registrations on this instance. With multiple fields we
        // can get multiple different views using the same instance
        // and each doing their own registration
        std::set<DistributedID> views;
      };
      std::map<RendezvousKey, RegisteredAnalysis> collective_analyses;
    };

    /**
     * \struct PhysicalUser
     * A class for representing physical users of a logical
     * region including necessary information to
     * register execution dependences on the user.
     */
    struct PhysicalUser : public Collectable {
    public:
      PhysicalUser(
          const RegionUsage& u, IndexSpaceExpression* expr, ApEvent term,
          UniqueID op_id, unsigned index, bool copy, bool covers);
      PhysicalUser(const PhysicalUser& rhs) = delete;
      ~PhysicalUser(void);
    public:
      PhysicalUser& operator=(const PhysicalUser& rhs) = delete;
    public:
      const RegionUsage usage;
      IndexSpaceExpression* const expr;
      const ApEvent term_event;
      const UniqueID op_id;
      const unsigned index;  // region requirement index
      const bool copy_user;  // is this from a copy or an operation
      const bool covers;     // whether the expr covers the ExprView its in
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/views/individual.inl"

#endif  // __LEGION_INDIVIDUAL_VIEW_H__
