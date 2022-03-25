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

#ifndef __LEGION_VIEWS_H__
#define __LEGION_VIEWS_H__

#include "legion/legion_types.h"
#include "legion/legion_analysis.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

// It's unclear what is causing view replication to be buggy, but my guess
// is that it has something to do with the "out-of-order" updates applied
// to remote views that gets us into trouble. It seems like view replication
// is also in general slower than not replicating, so we're turning it off
// for now as it is better to be both correct and faster. We're leaving
// the implementation though here in case a program arises in the future
// where view replication could lead to a performance win.
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Note that this MAYBE was fixed by commit ddc4b70b86 but it has not
// been tested. You can try this, but please verify that it is working
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef ENABLE_VIEW_REPLICATION
#warning "ENABLE_VIEW_REPLICATION is buggy, see issue #653, please be careful!"
#endif

namespace Legion {
  namespace Internal {

    /**
     * \class LogicalView 
     * This class is the abstract base class for representing
     * the logical view onto one or more physical instances
     * in memory.  Logical views are reference counted
     * and will delete themselves once they no longer have
     * any valid handles.
     */
    class LogicalView : public DistributedCollectable {
    public:
      LogicalView(RegionTreeForest *ctx, DistributedID did,
                  AddressSpaceID owner_proc, bool register_now);
      virtual ~LogicalView(void);
    public:
      inline bool is_instance_view(void) const;
      inline bool is_deferred_view(void) const;
      inline bool is_materialized_view(void) const;
      inline bool is_reduction_view(void) const;
      inline bool is_fill_view(void) const;
      inline bool is_phi_view(void) const;
    public:
      inline InstanceView* as_instance_view(void) const;
      inline DeferredView* as_deferred_view(void) const;
      inline MaterializedView* as_materialized_view(void) const;
      inline ReductionView* as_reduction_view(void) const;
      inline FillView* as_fill_view(void) const;
      inline PhiView *as_phi_view(void) const;
    public:
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_space(const FieldMask &space_mask) const = 0;
    public:
      virtual void notify_active(ReferenceMutator *mutator) = 0;
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      static void handle_view_request(Deserializer &derez, Runtime *runtime,
                                      AddressSpaceID source);
    public:
      static inline DistributedID encode_materialized_did(DistributedID did);
      static inline DistributedID encode_reduction_did(DistributedID did);
      static inline DistributedID encode_fill_did(DistributedID did);
      static inline DistributedID encode_phi_did(DistributedID did);
      static inline bool is_materialized_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_fill_did(DistributedID did);
      static inline bool is_phi_did(DistributedID did);
    public:
      RegionTreeForest *const context;
    protected:
      mutable LocalLock view_lock;
    };

    /**
     * \class InstanceView 
     * The InstanceView class is used for managing the meta-data
     * for one or more physical instances which represent the
     * up-to-date version from a logical region's perspective.
     * The InstaceView class has two sub-classes: materialized
     * views which represent a single physical instance a reduction
     * view which is a specialized instance for storing reductions
     */
    class InstanceView : public LogicalView {
    public:
      typedef LegionMap<ApEvent,
                FieldMaskSet<IndexSpaceExpression> > EventFieldExprs; 
      typedef LegionMap<ApEvent,FieldMaskSet<PhysicalUser> > EventFieldUsers;
      typedef FieldMaskSet<PhysicalUser> EventUsers;
    public:
      struct DeferFindCopyPreconditionArgs : 
        public LgTaskArgs<DeferFindCopyPreconditionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_FIND_COPY_PRE_TASK_ID;
      public:
        DeferFindCopyPreconditionArgs(LogicalView *v, bool read, bool trace,
            ReductionOpID r, const FieldMask &m, IndexSpaceExpression *x, 
            UniqueID uid, unsigned idx, AddressSpaceID s, CopyFillAggregator *a,
            RtUserEvent d)
          : LgTaskArgs<DeferFindCopyPreconditionArgs>(uid),
            view(v), reading(read), trace_recording(trace), 
            redop(r), copy_mask(new FieldMask(m)), copy_expr(x),
            op_id(uid), index(idx), source(s), aggregator(a), done_event(d) { }
      public:
        LogicalView *const view;
        const bool reading;
        const bool trace_recording;
        ReductionOpID redop;
        FieldMask *const copy_mask;
        IndexSpaceExpression *const copy_expr;
        const UniqueID op_id;
        const unsigned index;
        const AddressSpaceID source;
        CopyFillAggregator *const aggregator;
        const RtUserEvent done_event;
      };
    public:
      InstanceView(RegionTreeForest *ctx,DistributedID did,PhysicalManager *man,
                   AddressSpaceID owner_proc, AddressSpaceID logical_owner, 
                   UniqueID owner_context, bool register_now); 
      virtual ~InstanceView(void);
    public:
      inline bool is_logical_owner(void) const
        { return (local_space == logical_owner); }
    public:
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_space(const FieldMask &space_mask) const = 0;
    public: 
      // Entry point functions for doing physical dependence analysis
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceExpression *expr,
                                    const UniqueID op_id,
                                    const unsigned index) = 0;
      // Always want users to be full index space expressions
      virtual ApEvent register_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const unsigned index,
                                    ApEvent term_event,
                                    RtEvent collect_event,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info,
                                    const AddressSpaceID source) = 0;
      virtual RtEvent find_copy_preconditions(bool reading,
                                    ReductionOpID redop,              
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    CopyFillAggregator &aggregator,
                                    const bool trace_recording,
                                    const AddressSpaceID source) = 0;
      virtual void find_copy_preconditions_remote(bool reading,
                                    ReductionOpID redop,
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    EventFieldExprs &preconditions,
                                    const bool trace_recording,
                                    const AddressSpaceID source) = 0;
      virtual void add_copy_user(bool reading, ReductionOpID redop,
                                 ApEvent done_event, RtEvent collect_event,
                                 const FieldMask &copy_mask,
                                 IndexSpaceExpression *copy_expr,
                                 UniqueID op_id, unsigned index,
                                 std::set<RtEvent> &applied_events,
                                 const bool trace_recording,
                                 const AddressSpaceID source) = 0;
#ifdef ENABLE_VIEW_REPLICATION
    public:
      virtual void process_replication_request(AddressSpaceID source,
                                 const FieldMask &request_mask,
                                 RtUserEvent done_event);
      virtual void process_replication_response(RtUserEvent done_event,
                                 Deserializer &derez);
      virtual void process_replication_removal(AddressSpaceID source,
                                 const FieldMask &removal_mask);
#endif
    public:
      // Reference counting state change functions
      virtual void notify_active(ReferenceMutator *mutator) = 0;
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
    public:
      // Getting field information for performing copies
      // We used to use these calls for all copy calls, but
      // now they are primarily used by region-to-region copy
      // calls as we use the extra layer of indirection below
      // to issue calls for collective cases
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL) = 0;
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &src_fields) = 0;
    public:
      void find_atomic_reservations(const FieldMask &mask, Operation *op, 
                                    const unsigned index, bool exclusive);
      void find_field_reservations(const FieldMask &mask,
                                   std::vector<Reservation> &results);
      static void handle_send_atomic_reservation_request(Runtime *runtime,
                                  Deserializer &derez, AddressSpaceID source);
      void update_field_reservations(const FieldMask &mask,
                                  const std::vector<Reservation> &reservations);
      static void handle_send_atomic_reservation_response(Runtime *runtime,
                                                          Deserializer &derez);
    public:
      static void handle_view_register_user(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_find_copy_pre_request(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_find_copy_pre_request(const void *args, 
                        Runtime *runtime);
      static void handle_view_find_copy_pre_response(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_add_copy_user(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
#ifdef ENABLE_VIEW_REPLICATION
      static void handle_view_replication_request(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_replication_response(Deserializer &derez,
                        Runtime *runtime);
      static void handle_view_replication_removal(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
#endif
    public:
      PhysicalManager *const manager;
      // The ID of the context that made this view
      // instance made for a virtual mapping
      const UniqueID owner_context;
      // This is the owner space for the purpose of logical analysis
      const AddressSpaceID logical_owner;
    private:
      // Keep track of the locks used for managing atomic coherence
      // on individual fields of this materialized view. Only the
      // top-level view for an instance needs to track this.
      std::map<unsigned,Reservation> atomic_reservations;
    };

    /**
     * \class CollectableView
     * An interface class for handling garbage collection of users
     */
    class CollectableView {
    public:
      virtual ~CollectableView(void) { }
    public:
      virtual void add_collectable_reference(ReferenceMutator *mutator) = 0;
      virtual bool remove_collectable_reference(ReferenceMutator *mutator) = 0;
      virtual void collect_users(const std::set<ApEvent> &to_collect) = 0;
    public:
      void defer_collect_user(PhysicalManager *manager, ApEvent term_event,
                              RtEvent collect,ReferenceMutator *mutator = NULL);
      static void handle_deferred_collect(CollectableView *view,
                                          const std::set<ApEvent> &to_collect);
    };

    /**
     * \class ExprView
     * A ExprView is a node in a tree of ExprViews for capturing users of a
     * physical instance. At each node it tracks the users of a specific
     * index space expression for the physical instance. It also knows about
     * the subviews which are any expressions that are dominated by the 
     * current node and which may overlap but no subview can dominate another.
     * Finding the interfering users then just requires traversing the top
     * node and any overlapping sub nodes and then doing this recursively.
     */
    class ExprView : public LegionHeapify<ExprView>, 
                     public CollectableView, public Collectable {
    public:
      typedef LegionMap<ApEvent,
                FieldMaskSet<IndexSpaceExpression> > EventFieldExprs; 
      typedef LegionMap<ApEvent,FieldMaskSet<PhysicalUser> > EventFieldUsers;
      typedef FieldMaskSet<PhysicalUser> EventUsers;
    public:
      ExprView(RegionTreeForest *ctx, PhysicalManager *manager,
               InstanceView *view, IndexSpaceExpression *expr); 
      ExprView(const ExprView &rhs);
      virtual ~ExprView(void);
    public:
      ExprView& operator=(const ExprView &rhs);
    public:
      virtual void add_collectable_reference(ReferenceMutator *mutator);
      virtual bool remove_collectable_reference(ReferenceMutator *mutator);
      virtual void collect_users(const std::set<ApEvent> &to_collect);
    public:
      void find_user_preconditions(const RegionUsage &usage,
                                   IndexSpaceExpression *user_expr,
                                   const bool user_dominates,
                                   const FieldMask &user_mask,
                                   ApEvent term_event,
                                   UniqueID op_id, unsigned index,
                                   std::set<ApEvent> &preconditions,
                                   const bool trace_recording);
      void find_copy_preconditions(const RegionUsage &usage,
                                   IndexSpaceExpression *copy_expr,
                                   const bool copy_dominates,
                                   const FieldMask &copy_mask,
                                   UniqueID op_id, unsigned index,
                                   EventFieldExprs &preconditions,
                                   const bool trace_recording);
      // Check to see if there is any view with the same shape already
      // in the ExprView tree, if so return it
      ExprView* find_congruent_view(IndexSpaceExpression *expr) const;
      // Add a new subview with fields into the tree
      void insert_subview(ExprView *subview, FieldMask &subview_mask);
      void find_tightest_subviews(IndexSpaceExpression *expr,
                                  FieldMask &expr_mask,
                                  LegionMap<std::pair<size_t,
                                    ExprView*>,FieldMask> &bounding_views);
      void add_partial_user(const RegionUsage &usage,
                            UniqueID op_id, unsigned index,
                            FieldMask user_mask,
                            const ApEvent term_event,
                            const RtEvent collect_event,
                            IndexSpaceExpression *user_expr,
                            const size_t user_volume,
                            const bool trace_recording);
      void add_current_user(PhysicalUser *user, const ApEvent term_event,
                            RtEvent collect_event, const FieldMask &user_mask,
                            const bool trace_recording);
      // TODO: Optimize this so that we prune out intermediate nodes in 
      // the tree that are empty and re-balance the tree. The hard part of
      // this is that it will require stopping any precondition searches
      // which currently can still happen at the same time
      void clean_views(FieldMask &valid_mask,FieldMaskSet<ExprView> &clean_set);
    public:
      void pack_replication(Serializer &rez, 
                            std::map<PhysicalUser*,unsigned> &indexes,
                            const FieldMask &pack_mask,
                            const AddressSpaceID target) const;
      void unpack_replication(Deserializer &derez, ExprView *root,
                              const AddressSpaceID source,
                              std::map<IndexSpaceExprID,ExprView*> &expr_cache,
                              std::vector<PhysicalUser*> &users);
      void deactivate_replication(const FieldMask &deactivate_mask);
    protected:
      void find_current_preconditions(const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      IndexSpaceExpression *user_expr,
                                      ApEvent term_event,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers,
                                      std::set<ApEvent> &preconditions,
                                      std::set<ApEvent> &dead_events,
                                      EventFieldUsers &filter_users,
                                      FieldMask &observed, 
                                      FieldMask &non_dominated,
                                      const bool trace_recording);
      void find_previous_preconditions(const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      IndexSpaceExpression *user_expr,
                                      ApEvent term_event,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers,
                                      std::set<ApEvent> &preconditions,
                                      std::set<ApEvent> &dead_events,
                                      const bool trace_recording);
      void find_previous_filter_users(const FieldMask &dominated_mask,
                                      EventFieldUsers &filter_users);
      // More overload versions for even more precise information including
      // the index space expressions for individual events and fields
      void find_current_preconditions(const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers,
                                      EventFieldExprs &preconditions,
                                      std::set<ApEvent> &dead_events,
                                      EventFieldUsers &filter_events,
                                      FieldMask &observed, 
                                      FieldMask &non_dominated,
                                      const bool trace_recording);
      void find_previous_preconditions(const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers,
                                      EventFieldExprs &preconditions,
                                      std::set<ApEvent> &dead_events,
                                      const bool trace_recording);
      template<bool COPY_USER>
      inline bool has_local_precondition(PhysicalUser *prev_user,
                                      const RegionUsage &next_user,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers,
                                      bool &dominates);
      template<bool COPY_USER>
      inline bool has_local_precondition(PhysicalUser *prev_user,
                                      const RegionUsage &next_user,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers);
    protected:
      void filter_local_users(ApEvent term_event);
      void filter_current_users(const EventFieldUsers &to_filter);
      void filter_previous_users(const EventFieldUsers &to_filter);
      bool refine_users(void);
      static void verify_current_to_filter(const FieldMask &dominated,
                                  EventFieldUsers &current_to_filter);
    public:
      RegionTreeForest *const context;
      PhysicalManager *const manager;
      InstanceView *const inst_view;
      IndexSpaceExpression *const view_expr;
      const size_t view_volume;
#if defined(DEBUG_LEGION_GC) || defined(LEGION_GC)
      const DistributedID view_did;
#endif
      // This is publicly mutable and protected by expr_lock from
      // the owner inst_view
      FieldMask invalid_fields;
    protected:
      mutable LocalLock view_lock;
    protected:
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
      EventFieldUsers current_epoch_users;
      EventFieldUsers previous_epoch_users;
    protected:
      // Subviews for fields that have users in subexpressions
      FieldMaskSet<ExprView> subviews;
    };

    /**
     * \interface RemotePendingUser 
     * This is an interface for capturing users that are deferred
     * on remote views until they become valid replicated views.
     */
    class RemotePendingUser {
    public:
      virtual ~RemotePendingUser(void) { }
    public:
      virtual bool apply(MaterializedView *view, const FieldMask &mask) = 0;
    };
  
    class PendingTaskUser : public RemotePendingUser,
                            public LegionHeapify<PendingTaskUser> {
    public:
      PendingTaskUser(const RegionUsage &usage, const FieldMask &user_mask,
                      IndexSpaceNode *user_expr, const UniqueID op_id,
                      const unsigned index, const ApEvent term_event,
                      const RtEvent collect_event);
      virtual ~PendingTaskUser(void);
    public:
      virtual bool apply(MaterializedView *view, const FieldMask &mask);
    public:
      const RegionUsage usage;
      FieldMask user_mask;
      IndexSpaceNode *const user_expr;
      const UniqueID op_id;
      const unsigned index;
      const ApEvent term_event;
      const RtEvent collect_event;
    };

    class PendingCopyUser : public RemotePendingUser, 
                            public LegionHeapify<PendingCopyUser> {
    public:
      PendingCopyUser(const bool reading, const FieldMask &copy_mask,
                      IndexSpaceExpression *copy_expr, const UniqueID op_id,
                      const unsigned index, const ApEvent term_event,
                      const RtEvent collect_event);
      virtual ~PendingCopyUser(void);
    public:
      virtual bool apply(MaterializedView *view, const FieldMask &mask);
    public:
      const bool reading;
      FieldMask copy_mask;
      IndexSpaceExpression *const copy_expr;
      const UniqueID op_id;
      const unsigned index;
      const ApEvent term_event;
      const RtEvent collect_event;
    };

    /**
     * \class MaterializedView 
     * The MaterializedView class is used for representing a given
     * logical view onto a single physical instance.
     */
    class MaterializedView : public InstanceView, 
                             public LegionHeapify<MaterializedView> {
    public:
      static const AllocationType alloc_type = MATERIALIZED_VIEW_ALLOC;
    public:
      // Number of users to be added between cache invalidations
      static const unsigned user_cache_timeout = 1024;
    public:
      typedef LegionMap<VersionID,FieldMaskSet<IndexSpaceExpression>,
                        PHYSICAL_VERSION_ALLOC> VersionFieldExprs;  
    public:
      struct DeferMaterializedViewArgs : 
        public LgTaskArgs<DeferMaterializedViewArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MATERIALIZED_VIEW_TASK_ID;
      public:
        DeferMaterializedViewArgs(DistributedID d, PhysicalManager *m,
            AddressSpaceID own, AddressSpaceID log, UniqueID ctx)
          : LgTaskArgs<DeferMaterializedViewArgs>(implicit_provenance),
            did(d), manager(m), owner_space(own), 
            logical_owner(log), context_uid(ctx) { }
      public:
        const DistributedID did;
        PhysicalManager *const manager;
        const AddressSpaceID owner_space;
        const AddressSpaceID logical_owner;
        const UniqueID context_uid;
      };
    public:
      MaterializedView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, 
                       AddressSpaceID logical_owner, PhysicalManager *manager,
                       UniqueID owner_context, bool register_now);
      MaterializedView(const MaterializedView &rhs);
      virtual ~MaterializedView(void);
    public:
      MaterializedView& operator=(const MaterializedView &rhs);
    public:
      inline const FieldMask& get_space_mask(void) const 
        { return manager->layout->allocated_fields; }
    public:
      const FieldMask& get_physical_mask(void) const;
    public:
      virtual bool has_space(const FieldMask &space_mask) const;
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &src_fields);
    public:
      virtual bool has_manager(void) const { return true; }
      virtual PhysicalManager* get_manager(void) const { return manager; }
    public:
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceExpression *expr,
                                    const UniqueID op_id,
                                    const unsigned index);
      virtual ApEvent register_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const unsigned index,
                                    ApEvent term_event,
                                    RtEvent collect_event,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info,
                                    const AddressSpaceID source);
      virtual RtEvent find_copy_preconditions(bool reading,
                                    ReductionOpID redop,
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    CopyFillAggregator &aggregator,
                                    const bool trace_recording,
                                    const AddressSpaceID source);
      virtual void find_copy_preconditions_remote(bool reading,
                                    ReductionOpID redop,
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    EventFieldExprs &preconditions,
                                    const bool trace_recording,
                                    const AddressSpaceID source);
      virtual void add_copy_user(bool reading, ReductionOpID redop,
                                 ApEvent term_event, RtEvent collect_event,
                                 const FieldMask &copy_mask,
                                 IndexSpaceExpression *copy_expr,
                                 UniqueID op_id, unsigned index,
                                 std::set<RtEvent> &applied_events,
                                 const bool trace_recording,
                                 const AddressSpaceID source); 
#ifdef ENABLE_VIEW_REPLICATION
    public:
      virtual void process_replication_request(AddressSpaceID source,
                                 const FieldMask &request_mask,
                                 RtUserEvent done_event);
      virtual void process_replication_response(RtUserEvent done_event,
                                 Deserializer &derez);
      virtual void process_replication_removal(AddressSpaceID source,
                                 const FieldMask &removal_mask);
#endif
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target); 
    protected:
      friend class PendingTaskUser;
      friend class PendingCopyUser;
      void add_internal_task_user(const RegionUsage &usage,
                                  IndexSpaceExpression *user_expr,
                                  const FieldMask &user_mask,
                                  ApEvent term_event, 
                                  RtEvent collect_event, UniqueID op_id,
                                  const unsigned index,
                                  const bool trace_recording);
      void add_internal_copy_user(const RegionUsage &usage,
                                  IndexSpaceExpression *user_expr,
                                  const FieldMask &user_mask,
                                  ApEvent term_event, 
                                  RtEvent collect_event, UniqueID op_id,
                                  const unsigned index,
                                  const bool trace_recording);
      template<bool NEED_EXPR_LOCK>
      void clean_cache(void);
#ifdef ENABLE_VIEW_REPLICATION
      // Must be called while holding the replication lock
      void update_remote_replication_state(std::set<RtEvent> &applied_events);
#endif 
    public:
      static void handle_send_materialized_view(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
      static void handle_defer_materialized_view(const void *args, Runtime *rt);
      static void create_remote_view(Runtime *runtime, DistributedID did, 
                                     PhysicalManager *manager,
                                     AddressSpaceID owner_space, 
                                     AddressSpaceID logical_owner, 
                                     UniqueID context_uid);
    protected: 
      // Use a ExprView DAG to track the current users of this instance
      ExprView *current_users; 
      // Lock for serializing creation of ExprView objects
      mutable LocalLock expr_lock;
      // Mapping from user expressions to ExprViews to attach to
      std::map<IndexSpaceExprID,ExprView*> expr_cache;
      // A timeout counter for the cache so we don't permanently keep growing
      // in the case where the sets of expressions we use change over time
      unsigned expr_cache_uses;
      // Helping with making sure that there are no outstanding users being
      // added for when we go to invalidate the cache and clean the views
      std::atomic<unsigned> outstanding_additions;
      RtUserEvent clean_waiting; 
#ifdef ENABLE_VIEW_REPLICATION
    protected:
      // Lock for protecting the following replication data structures
      mutable LocalLock replicated_lock;
      // Track which fields we have replicated clones of our current users
      // On the owner node this tracks which fields have remote copies
      // On remote nodes this tracks which fields we have replicated
      FieldMask replicated_fields;
      // On the owner node we also need to keep track of our set of 
      // which nodes have replicated copies for which field
      union {
        LegionMap<AddressSpaceID,FieldMask> *replicated_copies;
        LegionMap<RtUserEvent,FieldMask> *replicated_requests;
      } repl_ptr;
      // For remote copies we track which fields have seen requests
      // in the past epoch of user adds so that we can reduce our 
      // set of replicated fields if we're not actually being
      // used for copy queries
      FieldMask remote_copy_pre_fields;
      unsigned remote_added_users; 
      // Users that we need to apply once we receive an update from the owner
      std::list<RemotePendingUser*> *remote_pending_users;
#endif
    protected:
      // Keep track of the current version numbers for each field
      // This will allow us to detect when physical instances are no
      // longer valid from a particular view when doing rollbacks for
      // resilience or mis-speculation.
      //VersionFieldExprs current_versions;
    };

    /**
     * \class ReductionView
     * The ReductionView class is used for providing a view
     * onto reduction physical instances from any logical perspective.
     */
    class ReductionView : public InstanceView, public CollectableView, 
                          public LegionHeapify<ReductionView> {
    public:
      static const AllocationType alloc_type = REDUCTION_VIEW_ALLOC;
    public:
      struct DeferReductionViewArgs : 
        public LgTaskArgs<DeferReductionViewArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_REDUCTION_VIEW_TASK_ID;
      public:
        DeferReductionViewArgs(DistributedID d, PhysicalManager *m,
            AddressSpaceID own, AddressSpaceID log, UniqueID ctx)
          : LgTaskArgs<DeferReductionViewArgs>(implicit_provenance),
            did(d), manager(m), owner_space(own), 
            logical_owner(log), context_uid(ctx) { }
      public:
        const DistributedID did;
        PhysicalManager *const manager;
        const AddressSpaceID owner_space;
        const AddressSpaceID logical_owner;
        const UniqueID context_uid;
      };
    public:
      ReductionView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc,
                    AddressSpaceID logical_owner, PhysicalManager *manager,
                    UniqueID owner_context, bool register_now);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      virtual bool has_manager(void) const { return true; } 
      virtual PhysicalManager* get_manager(void) const;
      virtual bool has_space(const FieldMask &space_mask) const
        { return false; }
    public: 
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceExpression *expr,
                                    const UniqueID op_id,
                                    const unsigned index);
      virtual ApEvent register_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const unsigned index,
                                    ApEvent term_event,
                                    RtEvent collect_event,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info,
                                    const AddressSpaceID source);
      virtual RtEvent find_copy_preconditions(bool reading,
                                    ReductionOpID redop,
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    CopyFillAggregator &aggregator,
                                    const bool trace_recording,
                                    const AddressSpaceID source);
      virtual void find_copy_preconditions_remote(bool reading,
                                    ReductionOpID redop,
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    EventFieldExprs &preconditions,
                                    const bool trace_recording,
                                    const AddressSpaceID source);
      virtual void add_copy_user(bool reading, ReductionOpID redop,
                                 ApEvent term_event, RtEvent collect_event,
                                 const FieldMask &copy_mask,
                                 IndexSpaceExpression *copy_expr,
                                 UniqueID op_id, unsigned index,
                                 std::set<RtEvent> &applied_events,
                                 const bool trace_recording,
                                 const AddressSpaceID source);
    protected: 
      void find_reducing_preconditions(const RegionUsage &usage,
                                       const FieldMask &user_mask,
                                       IndexSpaceExpression *user_expr,
                                       UniqueID op_id,
                                       std::set<ApEvent> &wait_on) const;
      void find_initializing_preconditions(const FieldMask &user_mask,
                                           IndexSpaceExpression *user_expr,
                                           UniqueID op_id,
                                           EventFieldExprs &preconditions);
      void find_reducing_preconditions(const FieldMask &user_mask,
                                       IndexSpaceExpression *user_expr,
                                       UniqueID op_id,
                                       EventFieldExprs &preconditions) const;
      void find_reading_preconditions(const FieldMask &user_mask,
                                      IndexSpaceExpression *user_expr,
                                      UniqueID op_id,
                                      EventFieldExprs &preconditions) const;
      bool add_user(const RegionUsage &usage,
                    IndexSpaceExpression *user_expr,
                    const FieldMask &user_mask,
                    ApEvent term_event, RtEvent collect_event,
                    UniqueID op_id, unsigned index,
                    bool copy_user, std::set<RtEvent> &applied_events,
                    const bool trace_recording);
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &src_fields);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void add_collectable_reference(ReferenceMutator *mutator);
      virtual bool remove_collectable_reference(ReferenceMutator *mutator);
      virtual void collect_users(const std::set<ApEvent> &term_events);
    public:
      virtual void send_view(AddressSpaceID target); 
    protected:
      void add_physical_user(PhysicalUser *user, bool reading,
                             ApEvent term_event, const FieldMask &user_mask);
      void filter_local_users(ApEvent term_event);
    public:
      static void handle_send_reduction_view(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
      static void handle_defer_reduction_view(const void *args, Runtime *rt);
      static void create_remote_view(Runtime *runtime, DistributedID did, 
                                     PhysicalManager *manager,
                                     AddressSpaceID owner_space, 
                                     AddressSpaceID logical_owner, 
                                     UniqueID context_uid);
    public:
      ReductionOpID get_redop(void) const;
    public:
      FillView *const fill_view; // fill view for this reduction value
    protected:
      EventFieldUsers initialization_users;
      EventFieldUsers reduction_users;
      EventFieldUsers reading_users;
      std::set<ApEvent> outstanding_gc_events;
    protected:
      std::set<ApEvent> initial_user_events; 
    };

    /**
     * \class DeferredView
     * A DeferredView class is an abstract class the complements
     * the MaterializedView class. While materialized views are 
     * actual views onto a real instance, deferred views are 
     * effectively place holders for non-physical isntances which
     * contain enough information to perform the necessary 
     * operations to bring a materialized view up to date for 
     * specific fields. There are several different flavors of
     * deferred views and this class is the base type.
     */
    class DeferredView : public LogicalView {
    public:
      DeferredView(RegionTreeForest *ctx, DistributedID did,
                   AddressSpaceID owner_space, bool register_now);
      virtual ~DeferredView(void);
    public:
      // Deferred views never have managers
      virtual bool has_manager(void) const { return false; }
      virtual PhysicalManager* get_manager(void) const
        { return NULL; }
      virtual bool has_space(const FieldMask &space_mask) const
        { return false; }
    public:
      virtual void notify_active(ReferenceMutator *mutator) = 0;
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      // Should never be called directly
      virtual InnerContext* get_context(void) const
        { assert(false); return NULL; }
    public:
      virtual void flatten(CopyFillAggregator &aggregator,
                           InstanceView *dst_view, const FieldMask &src_mask,
                           IndexSpaceExpression *expr, 
                           CopyAcrossHelper *helper) = 0;
    };

    /**
     * \class FillView
     * This is a deferred view that is used for filling in 
     * fields with a default value.
     */
    class FillView : public DeferredView,
                     public LegionHeapify<FillView> {
    public:
      static const AllocationType alloc_type = FILL_VIEW_ALLOC;
    public:
      class FillViewValue : public Collectable {
      public:
        FillViewValue(const void *v, size_t size)
          : value(v), value_size(size) { }
        FillViewValue(const FillViewValue &rhs)
          : value(NULL), value_size(0) { assert(false); }
        ~FillViewValue(void)
        { free(const_cast<void*>(value)); }
      public:
        FillViewValue& operator=(const FillViewValue &rhs)
        { assert(false); return *this; }
      public:
        inline bool matches(const void *other, const size_t size)
        {
          if (value_size != size)
            return false;
          // Compare the bytes
          return (memcmp(other, value, value_size) == 0);
        }
      public:
        const void *const value;
        const size_t value_size;
      };
    public:
      FillView(RegionTreeForest *ctx, DistributedID did,
               AddressSpaceID owner_proc,
               FillViewValue *value, bool register_now
#ifdef LEGION_SPY
               , UniqueID fill_op_uid
#endif
               );
      FillView(const FillView &rhs);
      virtual ~FillView(void);
    public:
      FillView& operator=(const FillView &rhs);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target); 
    public:
      virtual void flatten(CopyFillAggregator &aggregator,
                           InstanceView *dst_view, const FieldMask &src_mask,
                           IndexSpaceExpression *expr, 
                           CopyAcrossHelper *helper);
    public:
      static void handle_send_fill_view(Runtime *runtime, Deserializer &derez,
                                        AddressSpaceID source);
    public:
      FillViewValue *const value;
#ifdef LEGION_SPY
      const UniqueID fill_op_uid;
#endif
    };

    /**
     * \class PhiView
     * A phi view is exactly what it sounds like: a view to merge two
     * different views together from different control flow paths.
     * Specifically it is able to merge together different paths for
     * predication so that we can issue copies from both a true and
     * a false version of a predicate. This allows us to map past lazy
     * predicated operations such as fills and virtual mappings and
     * continue to get ahead of actual execution. It's not pretty
     * but it seems to work.
     */
    class PhiView : public DeferredView, 
                    public LegionHeapify<PhiView> {
    public:
      static const AllocationType alloc_type = PHI_VIEW_ALLOC;
    public:
      struct DeferPhiViewRefArgs : 
        public LgTaskArgs<DeferPhiViewRefArgs> {
      public:
        static const LgTaskID TASK_ID =
          LG_DEFER_PHI_VIEW_REF_TASK_ID;
      public:
        DeferPhiViewRefArgs(DistributedCollectable *d, DistributedID id)
          : LgTaskArgs<DeferPhiViewRefArgs>(implicit_provenance),
            dc(d), did(id) { }
      public:
        DistributedCollectable *const dc;
        const DistributedID did; 
      };
      struct DeferPhiViewRegistrationArgs : 
        public LgTaskArgs<DeferPhiViewRegistrationArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_DEFER_PHI_VIEW_REGISTRATION_TASK_ID;
      public:
        DeferPhiViewRegistrationArgs(PhiView *v)
          : LgTaskArgs<DeferPhiViewRegistrationArgs>(implicit_provenance),
            view(v) { }
      public:
        PhiView *const view;
      };
    public:
      PhiView(RegionTreeForest *ctx, DistributedID did,
              AddressSpaceID owner_proc, PredEvent true_guard,
              PredEvent false_guard, InnerContext *owner,
              bool register_now);
      PhiView(const PhiView &rhs);
      virtual ~PhiView(void);
    public:
      PhiView& operator=(const PhiView &rhs);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target);
      virtual InnerContext* get_context(void) const
        { return owner_context; }
    public:
      virtual void flatten(CopyFillAggregator &aggregator,
                           InstanceView *dst_view, const FieldMask &src_mask,
                           IndexSpaceExpression *expr, 
                           CopyAcrossHelper *helper);
    public:
      void record_true_view(LogicalView *view, const FieldMask &view_mask);
      void record_false_view(LogicalView *view, const FieldMask &view_mask);
    public:
      void pack_phi_view(Serializer &rez);
      void unpack_phi_view(Deserializer &derez,std::set<RtEvent> &ready_events);
      RtEvent defer_add_reference(DistributedCollectable *dc, 
                                  RtEvent precondition) const;
      static void handle_send_phi_view(Runtime *runtime, Deserializer &derez,
                                       AddressSpaceID source);
      static void handle_deferred_view_ref(const void *args);
      static void handle_deferred_view_registration(const void *args);
    public:
      const PredEvent true_guard;
      const PredEvent false_guard;
      InnerContext *const owner_context;
    protected:
      LegionMap<LogicalView*,FieldMask> true_views;
      LegionMap<LogicalView*,FieldMask> false_views;
    };

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_materialized_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, MATERIALIZED_VIEW_DC); 
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_reduction_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REDUCTION_VIEW_DC); 
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_fill_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, FILL_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_phi_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, PHI_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_materialized_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xFULL) == 
                                          MATERIALIZED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_reduction_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xFULL) == 
                                              REDUCTION_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_fill_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xFULL) == 
                                                    FILL_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_phi_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xFULL) == PHI_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      return (is_materialized_did(did) || is_reduction_did(did));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_deferred_view(void) const
    //--------------------------------------------------------------------------
    {
      return (is_fill_did(did) || is_phi_did(did));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_materialized_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_fill_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_fill_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_phi_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_phi_did(did);
    }

    //--------------------------------------------------------------------------
    inline InstanceView* LogicalView::as_instance_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_instance_view());
#endif
      return static_cast<InstanceView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline DeferredView* LogicalView::as_deferred_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_deferred_view());
#endif
      return static_cast<DeferredView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline MaterializedView* LogicalView::as_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_materialized_view());
#endif
      return static_cast<MaterializedView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline ReductionView* LogicalView::as_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_reduction_view());
#endif
      return static_cast<ReductionView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline FillView* LogicalView::as_fill_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_fill_view());
#endif
      return static_cast<FillView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline PhiView* LogicalView::as_phi_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_phi_view());
#endif
      return static_cast<PhiView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    template<bool COPY_USER>
    inline bool ExprView::has_local_precondition(PhysicalUser *user,
                                                 const RegionUsage &next_user,
                                                 IndexSpaceExpression *expr,
                                                 const UniqueID op_id,
                                                 const unsigned index,
                                                 const bool next_covers,
                                                 bool &dominates)
    //--------------------------------------------------------------------------
    {
      // We order these tests in a entirely based on cost

      // Different region requirements of the same operation 
      // Copies from different region requirements though still 
      // need to wait on each other correctly
      if ((op_id == user->op_id) && (index != user->index) && 
          (!COPY_USER || !user->copy_user))
        return false;
      // Now do a dependence test for privilege non-interference
      // Only reductions here are copy reductions which we know do not interfere
      DependenceType dt = check_dependence_type<false>(user->usage, next_user);
      switch (dt)
      {
        case LEGION_NO_DEPENDENCE:
        case LEGION_ATOMIC_DEPENDENCE:
        case LEGION_SIMULTANEOUS_DEPENDENCE:
          return false;
        case LEGION_TRUE_DEPENDENCE:
        case LEGION_ANTI_DEPENDENCE:
          break;
        default:
          assert(false); // should never get here
      }
      if (!next_covers)
      {
        if (!user->covers)
        {
          // Neither one covers so we actually need to do the
          // full intersection test and see if next covers
          IndexSpaceExpression *overlap = 
            context->intersect_index_spaces(expr, user->expr);
          if (overlap->is_empty())
            return false;
        }
        // We don't allow any user that doesn't fully cover the
        // expression to dominate anything. It's hard to guarantee
        // correctness without this. Think very carefully if you
        // plan to change this!
        dominates = false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    template<bool COPY_USER>
    inline bool ExprView::has_local_precondition(PhysicalUser *user,
                                                 const RegionUsage &next_user,
                                                 IndexSpaceExpression *expr,
                                                 const UniqueID op_id,
                                                 const unsigned index,
                                                 const bool next_covers)
    //--------------------------------------------------------------------------
    {
      // We order these tests in a entirely based on cost

      // Different region requirements of the same operation 
      // Copies from different region requirements though still 
      // need to wait on each other correctly
      if ((op_id == user->op_id) && (index != user->index) && 
          (!COPY_USER || !user->copy_user))
        return false;
      // Now do a dependence test for privilege non-interference
      // Only reductions here are copy reductions which we know do not interfere
      DependenceType dt = check_dependence_type<false>(user->usage, next_user);
      switch (dt)
      {
        case LEGION_NO_DEPENDENCE:
        case LEGION_ATOMIC_DEPENDENCE:
        case LEGION_SIMULTANEOUS_DEPENDENCE:
          return false;
        case LEGION_TRUE_DEPENDENCE:
        case LEGION_ANTI_DEPENDENCE:
          break;
        default:
          assert(false); // should never get here
      }
      // If the user doesn't cover the expression for this view then
      // we need to do an extra intersection test, this should only
      // happen with copy users at the moment
      if (!user->covers && !next_covers)
      {
        IndexSpaceExpression *overlap = 
          context->intersect_index_spaces(expr, user->expr);
        if (overlap->is_empty())
          return false;
      }
      return true;
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_VIEWS_H__
