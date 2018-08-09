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

#ifndef __LEGION_VIEWS_H__
#define __LEGION_VIEWS_H__

#include "legion/legion_types.h"
#include "legion/legion_analysis.h"
#include "legion/legion_utilities.h"
#include "legion/legion_instances.h"
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

namespace Legion {
  namespace Internal {

    // Use this macro to enable migration of instance view data
    // structures across the machine rather than having all analysis
    // proceed for each instance on the owner node
    // #define DISTRIBUTED_INSTANCE_VIEWS

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
                  AddressSpaceID owner_proc, 
                  RegionTreeNode *node, bool register_now);
      virtual ~LogicalView(void);
    public:
      inline bool is_instance_view(void) const;
      inline bool is_deferred_view(void) const;
      inline bool is_materialized_view(void) const;
      inline bool is_reduction_view(void) const;
      inline bool is_composite_view(void) const;
      inline bool is_fill_view(void) const;
      inline bool is_phi_view(void) const;
    public:
      inline InstanceView* as_instance_view(void) const;
      inline DeferredView* as_deferred_view(void) const;
      inline MaterializedView* as_materialized_view(void) const;
      inline ReductionView* as_reduction_view(void) const;
      inline CompositeView* as_composite_view(void) const;
      inline FillView* as_fill_view(void) const;
      inline PhiView *as_phi_view(void) const;
    public:
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const LegionColor c) = 0;
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
      void defer_collect_user(ApEvent term_event, ReferenceMutator *mutator);
      virtual void collect_users(const std::set<ApEvent> &term_events) = 0;
      virtual void update_gc_events(const std::set<ApEvent> &term_events) = 0;
      static void handle_deferred_collect(LogicalView *view,
                                          const std::set<ApEvent> &term_events);
    public:
      static inline DistributedID encode_materialized_did(DistributedID did,
                                                           bool top);
      static inline DistributedID encode_reduction_did(DistributedID did);
      static inline DistributedID encode_composite_did(DistributedID did);
      static inline DistributedID encode_fill_did(DistributedID did);
      static inline DistributedID encode_phi_did(DistributedID did);
      static inline bool is_materialized_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_composite_did(DistributedID did);
      static inline bool is_fill_did(DistributedID did);
      static inline bool is_phi_did(DistributedID did);
      static inline bool is_top_did(DistributedID did);
    public:
      RegionTreeForest *const context;
      RegionTreeNode *const logical_node;
    protected:
      mutable LocalLock view_lock;
    };

    /**
     * \class InstanceView 
     * The InstanceView class is used for managing the meta-data
     * for one or more physical instances which represent the
     * up-to-date version from a logical region's perspective.
     * The InstaceView class has two sub-classes: materialized
     * views which represent a single physical instance, or
     * composite views which contain multiple physical instances.
     */
    class InstanceView : public LogicalView {
    public:
      InstanceView(RegionTreeForest *ctx, DistributedID did,
                   AddressSpaceID owner_proc, AddressSpaceID logical_owner, 
                   RegionTreeNode *node, UniqueID owner_context, 
                   bool register_now); 
      virtual ~InstanceView(void);
    public:
      inline bool is_logical_owner(void) const
        { return (local_space == logical_owner); }
    public:
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const LegionColor c) = 0;
      virtual Memory get_location(void) const = 0;
      virtual bool has_space(const FieldMask &space_mask) const = 0;
    public:
      // Entry point functions for doing physical dependence analysis
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           bool single_copy/*only for writing*/,
                                           bool restrict_out,
                                           const FieldMask &copy_mask,
                                           IndexSpaceExpression *copy_expr,
                                           VersionTracker *version_tracker,
                                           const UniqueID creator_op_id,
                                           const unsigned index,
                                           const AddressSpaceID source,
                     LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info,
                                           bool can_filter = true) = 0;
      virtual void add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                 VersionTracker *version_tracker,
                                 IndexSpaceExpression *copy_expr,
                                 const UniqueID creator_op_id,
                                 const unsigned index, const FieldMask &mask, 
                                 bool reading, bool restrict_out,
                                 const AddressSpaceID source,
                                 std::set<RtEvent> &applied_events,
                                 const PhysicalTraceInfo &trace_info) = 0;
      virtual ApEvent find_user_precondition(const RegionUsage &user,
                                       ApEvent term_event,
                                       const FieldMask &user_mask,
                                       const UniqueID op_id,
                                       const unsigned index,
                                       VersionTracker *version_tracker,
                                       std::set<RtEvent> &applied_events,
                                       const PhysicalTraceInfo &trace_info) = 0;
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, 
                            Operation *op, const unsigned index, 
                            VersionTracker *version_tracker,
                            std::set<RtEvent> &applied_events,
                            const PhysicalTraceInfo &trace_info) = 0;
      // This is a fused version of the above two methods
      virtual ApEvent add_user_fused(const RegionUsage &user,
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
                                   VersionTracker *version_tracker,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info,
                                   bool update_versions = true) = 0;
      virtual void add_user_base(const RegionUsage &user, ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   const UniqueID op_id, const unsigned index, 
                                   const AddressSpaceID source,
                                   VersionTracker *version_tracker,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info) = 0;
      virtual ApEvent add_user_fused_base(const RegionUsage &user,
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   const UniqueID op_id, const unsigned index,
                                   VersionTracker *version_tracker,
                                   const AddressSpaceID source,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info,
                                   bool update_versions = true) = 0;
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    const UniqueID op_id,
                                    const unsigned index) = 0;
    public:
      // Reference counting state change functions
      virtual void notify_active(ReferenceMutator *mutator) = 0;
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
    public:
      // Instance recycling
      virtual void collect_users(const std::set<ApEvent> &term_events) = 0;
    public:
      // Getting field information for performing copies
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL) = 0;
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &src_fields) = 0;
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                     std::vector<CopySrcDstField> &src_fields,
                             CopyAcrossHelper *across_helper = NULL) = 0;
      virtual void reduce_from(ReductionOpID redop,
                               const FieldMask &reduce_mask, 
                       std::vector<CopySrcDstField> &src_fields) = 0;
    public:
      inline InstanceView* get_instance_subview(const LegionColor c) 
        { return get_subview(c)->as_instance_view(); }
      
#ifdef DISTRIBUTED_INSTANCE_VIEWS
    public:
      virtual void process_update_request(AddressSpaceID source,
                               RtUserEvent done_event, Deserializer &derez) = 0;
      virtual void process_update_response(Deserializer &derez,
                                           RtUserEvent done_event,
                                           AddressSpaceID source,
                                           RegionTreeForest *forest) = 0;
      virtual void process_remote_update(Deserializer &derez,
                                         AddressSpaceID source,
                                         RegionTreeForest *forest) = 0;
      virtual void process_remote_invalidate(const FieldMask &invalid_mask,
                                             RtUserEvent done_event) = 0;
    public:
      static void handle_view_update_request(Deserializer &derez, 
          Runtime *runtime, AddressSpaceID source); 
      static void handle_view_update_response(Deserializer &derez, 
          Runtime *runtime, AddressSpaceID source);
      static void handle_view_remote_update(Deserializer &derez, Runtime *rt,
                                            AddressSpaceID source);
      static void handle_view_remote_invalidate(Deserializer &derez,  
                                                Runtime *rt);
#else
    public:
      static void handle_view_copy_preconditions(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_view_add_copy(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_view_user_preconditions(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_view_add_user(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      static void handle_view_add_user_fused(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
#endif
    public:
      // The ID of the context that made this view
      // Note this view can escape this context inside a composite
      // instance made for a virtual mapping
      const UniqueID owner_context;
      // This is the owner space for the purpose of logical analysis
      const AddressSpaceID logical_owner;
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
      struct DeferMaterializedViewArgs : 
        public LgTaskArgs<DeferMaterializedViewArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MATERIALIZED_VIEW_TASK_ID;
      public:
        DeferMaterializedViewArgs(DistributedID id, AddressSpaceID own,
                                  AddressSpaceID log, RegionTreeNode *target,
                                  PhysicalManager *man, MaterializedView *par,
                                  UniqueID ctx_uid)
          : LgTaskArgs<DeferMaterializedViewArgs>(implicit_provenance),
            did(id), owner_space(own), logical_owner(log), target_node(target),
            manager(man), parent(par), context_uid(ctx_uid) { }
      public:
        const DistributedID did;
        const AddressSpaceID owner_space;
        const AddressSpaceID logical_owner;
        RegionTreeNode *const target_node;
        PhysicalManager *const manager;
        MaterializedView *const parent;
        const UniqueID context_uid;
      };
    public:
      struct EventUsers {
      public:
        EventUsers(void)
          : single(true) { users.single_user = NULL; }
      public:
        EventUsers& operator=(const EventUsers &rhs)
          { assert(false); return *this; }
      public:
        FieldMask user_mask;
        union {
          PhysicalUser *single_user;
          LegionMap<PhysicalUser*,FieldMask>::aligned *multi_users;
        } users;
        bool single;
      };
    public:
      MaterializedView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, 
                       AddressSpaceID logical_owner, RegionTreeNode *node, 
                       InstanceManager *manager, MaterializedView *parent, 
                       UniqueID owner_context, bool register_now);
      MaterializedView(const MaterializedView &rhs);
      virtual ~MaterializedView(void);
    public:
      MaterializedView& operator=(const MaterializedView &rhs);
    public:
      void add_remote_child(MaterializedView *child);
    public:
      inline const FieldMask& get_space_mask(void) const 
        { return manager->layout->allocated_fields; }
    public:
      const FieldMask& get_physical_mask(void) const;
    public:
      virtual bool has_space(const FieldMask &space_mask) const;
    public:
      MaterializedView* get_materialized_subview(const LegionColor c);
      static void handle_subview_did_request(Deserializer &derez,
                             Runtime *runtime, AddressSpaceID source);
      static void handle_subview_did_response(Deserializer &derez); 
      MaterializedView* get_materialized_parent_view(void) const;
    public:
      void copy_field(FieldID fid, std::vector<CopySrcDstField> &infos);
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &src_fields);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<CopySrcDstField> &dst_fields,
                             CopyAcrossHelper *across_helper = NULL);
      virtual void reduce_from(ReductionOpID redop,
                               const FieldMask &reduce_mask,
                          std::vector<CopySrcDstField> &src_fields);
    public:
      void accumulate_events(std::set<ApEvent> &all_events);
    public:
      virtual bool has_manager(void) const { return true; }
      virtual PhysicalManager* get_manager(void) const { return manager; }
      virtual bool has_parent(void) const { return (parent != NULL); }
      virtual LogicalView* get_parent(void) const { return parent; }
      virtual LogicalView* get_subview(const LegionColor c);
      virtual Memory get_location(void) const;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           bool single_copy/*only for writing*/,
                                           bool restrict_out,
                                           const FieldMask &copy_mask,
                                           IndexSpaceExpression *copy_expr,
                                           VersionTracker *version_tracker,
                                           const UniqueID creator_op_id,
                                           const unsigned index,
                                           const AddressSpaceID source,
                         LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info,
                                           bool can_filter = true);
    protected: 
      void find_copy_preconditions_above(ReductionOpID redop, bool reading,
                                         bool single_copy, bool restrict_out,
                                         const FieldMask &copy_mask,
                                         const LegionColor child_color,
                                         IndexSpaceExpression *user_expr,
                                         VersionTracker *version_tracker,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                         const AddressSpaceID source,
                       LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info);
      friend class CompositeView;
      // Give composite views special access here so they can filter
      // back just the users at the particular level
      void find_local_copy_preconditions(ReductionOpID redop, bool reading,
                                         bool single_copy, bool restrict_out,
                                         const FieldMask &copy_mask,
                                         const LegionColor child_color,
                                         IndexSpaceExpression *user_expr,
                                         VersionTracker *version_tracker,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                         const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info);
      void find_local_copy_preconditions_above(ReductionOpID redop,bool reading,
                                         bool single_copy, bool restrict_out,
                                         const FieldMask &copy_mask,
                                         const LegionColor child_color,
                                         IndexSpaceExpression *user_expr,
                                         VersionTracker *version_tracker,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                         const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info,
                                         const bool actually_above = true);
    public:
      virtual void add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                 VersionTracker *version_tracker,
                                 IndexSpaceExpression *copy_expr,
                                 const UniqueID creator_op_id,
                                 const unsigned index,
                                 const FieldMask &mask, 
                                 bool reading, bool restrict_out,
                                 const AddressSpaceID source,
                                 std::set<RtEvent> &applied_events,
                                 const PhysicalTraceInfo &trace_info);
    protected:
      void add_copy_user_above(const RegionUsage &usage, ApEvent copy_term,
                               const LegionColor child_color,
                               IndexSpaceExpression *user_expr,
                               VersionTracker *version_tracker,
                               const UniqueID creator_op_id,
                               const unsigned index, const bool restrict_out,
                               const FieldMask &copy_mask,
                               const AddressSpaceID source,
                               std::set<RtEvent> &applied_events,
                               const PhysicalTraceInfo &trace_info);
      void add_local_copy_user(const RegionUsage &usage, ApEvent copy_term,
                               bool base_user, bool restrict_out,
                               const LegionColor child_color,
                               IndexSpaceExpression *user_expr,
                               VersionTracker *version_tracker,
                               const UniqueID creator_op_id,
                               const unsigned index,
                               const FieldMask &copy_mask,
                               const AddressSpaceID source,
                               std::set<RtEvent> &applied_events,
                               const PhysicalTraceInfo &trace_info);
    public:
      virtual ApEvent find_user_precondition(const RegionUsage &user,
                                           ApEvent term_event,
                                           const FieldMask &user_mask,
                                           const UniqueID op_id,
                                           const unsigned index,
                                           VersionTracker *version_tracker,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info);
    protected:
      void find_user_preconditions_above(const RegionUsage &usage,
                                         ApEvent term_event,
                                         const LegionColor child_color,
                                         IndexSpaceExpression *user_expr,
                                         VersionTracker *version_tracker,
                                         const UniqueID op_id,
                                         const unsigned index,
                                         const FieldMask &user_mask,
                                         std::set<ApEvent> &preconditions,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info);
      void find_local_user_preconditions(const RegionUsage &usage,
                                         ApEvent term_event,
                                         const LegionColor child_color,
                                         IndexSpaceExpression *user_expr,
                                         VersionTracker *version_tracker,
                                         const UniqueID op_id,
                                         const unsigned index,
                                         const FieldMask &user_mask,
                                         std::set<ApEvent> &preconditions,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info);
      void find_local_user_preconditions_above(const RegionUsage &usage,
                                         ApEvent term_event,
                                         const LegionColor child_color,
                                         IndexSpaceExpression *user_expr,
                                         VersionTracker *version_tracker,
                                         const UniqueID op_id,
                                         const unsigned index,
                                         const FieldMask &user_mask,
                                         std::set<ApEvent> &preconditions,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info,
                                         const bool actually_above = true);
    public:
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, 
                            Operation *op, const unsigned index,
                            VersionTracker *version_tracker,
                            std::set<RtEvent> &applied_events,
                            const PhysicalTraceInfo &trace_info);
      virtual void add_user_base(const RegionUsage &user, ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   const UniqueID op_id, const unsigned index, 
                                   const AddressSpaceID source,
                                   VersionTracker *version_tracker,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info);
    protected:
      void add_user_above(const RegionUsage &usage, ApEvent term_event,
                          const LegionColor child_color, 
                          IndexSpaceExpression *user_expr,
                          VersionTracker *version_tracker,
                          const UniqueID op_id, const unsigned index,
                          const FieldMask &user_mask,
                          const bool need_version_update,
                          const AddressSpaceID source,
                          std::set<RtEvent> &applied_events,
                          const PhysicalTraceInfo &trace_info);
      bool add_local_user(const RegionUsage &usage, ApEvent term_event,
                          const LegionColor child_color, 
                          IndexSpaceExpression *user_expr,
                          const bool base_user,
                          VersionTracker *version_tracker,
                          const UniqueID op_id, const unsigned index,
                          const FieldMask &user_mask,
                          const AddressSpaceID source,
                          std::set<RtEvent> &applied_events,
                          const PhysicalTraceInfo &trace_info);
    public:
      // This is a fused version of the above two virtual methods
      virtual ApEvent add_user_fused(const RegionUsage &user, 
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
                                   VersionTracker *version_tracker,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info,
                                   bool update_versions = true);
      virtual ApEvent add_user_fused_base(const RegionUsage &user,
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   const UniqueID op_id, const unsigned index,
                                   VersionTracker *version_tracker,
                                   const AddressSpaceID source,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info,
                                   bool update_versions = true);
    protected:
      void add_user_above_fused(const RegionUsage &usage, ApEvent term_event,
                                const LegionColor child_color,
                                IndexSpaceExpression *user_expr,
                                VersionTracker *version_tracker,
                                const UniqueID op_id,
                                const unsigned index,
                                const FieldMask &user_mask,
                                const AddressSpaceID source,
                                std::set<ApEvent> &preconditions,
                                std::set<RtEvent> &applied_events,
                                const PhysicalTraceInfo &trace_info,
                                const bool need_version_update);
    public:
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    const UniqueID op_id,
                                    const unsigned index);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void collect_users(const std::set<ApEvent> &term_users);
      virtual void update_gc_events(const std::set<ApEvent> &term_events);
    public:
      virtual void send_view(AddressSpaceID target); 
      void update_gc_events(const std::deque<ApEvent> &gc_events);
    public:
      void filter_invalid_fields(FieldMask &to_filter,
                                 VersionInfo &version_info);
    protected:
      // Update the version numbers
      // These first two methods do two-phase updates for copies
      // These methods must be called while holding the lock
      // in non-exclusive and exclusive mode respectively
      void find_copy_version_updates(const FieldMask &copy_mask,
                                     VersionTracker *version_tracker,
                                     FieldMask &write_skip_mask,
                                     FieldMask &filter_mask,
                            LegionMap<VersionID,FieldMask>::aligned &advance,
                            LegionMap<VersionID,FieldMask>::aligned &add_only,
                              bool is_reducing, bool restrict_out, bool base);
      void apply_version_updates(FieldMask &filter_mask,
                      const LegionMap<VersionID,FieldMask>::aligned &advance,
                      const LegionMap<VersionID,FieldMask>::aligned &add_only,
                      AddressSpaceID source, std::set<RtEvent> &applied_events);
      // This method does one phase update and advance for users
      // This one will take it's own lock
      bool update_version_numbers(const FieldMask &user_mask,
                                  const FieldVersions &field_versions,
                                  const AddressSpaceID source,
                                  std::set<RtEvent> &applied_events);
    protected:
      void filter_and_add(FieldMask &filter_mask,
                const LegionMap<VersionID,FieldMask>::aligned &add_versions);
#ifdef DEBUG_LEGION
      void sanity_check_versions(void);
#endif
    protected:
      void add_current_user(PhysicalUser *user, ApEvent term_event,
                            const FieldMask &user_mask);
      void filter_local_users(ApEvent term_event);
      void filter_local_users(const FieldMask &filter_mask,
          LegionMap<ApEvent,EventUsers>::aligned &local_epoch_users);
      void filter_current_user(ApEvent user_event, 
                               const FieldMask &filter_mask);
      void filter_previous_user(ApEvent user_event, 
                                const FieldMask &filter_mask);
    protected:
      template<bool TRACK_DOM>
      void find_current_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const LegionColor child_color,
                                      IndexSpaceExpression *user_expr,
                                      ApEvent term_event,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      std::set<ApEvent> &preconditions,
                                      std::set<ApEvent> &dead_events,
                  LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                      FieldMask &observed, 
                                      FieldMask &non_dominated,
                                      const PhysicalTraceInfo &trace_info);
      void find_previous_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const LegionColor child_color,
                                      IndexSpaceExpression *user_expr,
                                      ApEvent term_event,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      std::set<ApEvent> &preconditions,
                                      std::set<ApEvent> &dead_events,
                                      const PhysicalTraceInfo &trace_info);
      // Overloaded versions for being precise about copy preconditions
      template<bool TRACK_DOM>
      void find_current_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const LegionColor child_color,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                  LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                      std::set<ApEvent> &dead_events,
                  LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                      FieldMask &observed, 
                                      FieldMask &non_dominated,
                                      const PhysicalTraceInfo &trace_info);
      void find_previous_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const LegionColor child_color,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                  LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                      std::set<ApEvent> &dead_events,
                                      const PhysicalTraceInfo &trace_info);
      void find_previous_filter_users(const FieldMask &dominated_mask,
                  LegionMap<ApEvent,FieldMask>::aligned &filter_events);
      inline bool has_local_precondition(PhysicalUser *prev_user,
                                     const RegionUsage &next_user,
                                     const LegionColor child_color,
                                     const UniqueID op_id,
                                     const unsigned index,
                                     IndexSpaceExpression *user_expr,
                                     ApEvent &precondition_event);
    public:
      void find_atomic_reservations(const FieldMask &mask, 
                                    Operation *op, bool exclusive);
    public:
      void find_field_reservations(const std::vector<FieldID> &needed_fields,
                                   std::vector<Reservation> &results);
      static void handle_send_atomic_reservation_request(Runtime *runtime,
                                  Deserializer &derez, AddressSpaceID source);
      void update_field_reservations(
                            const std::vector<FieldID> &fields,
                            const std::vector<Reservation> &reservations);
      static void handle_send_atomic_reservation_response(Runtime *runtime,
                                                          Deserializer &derez);
    public:
      static void handle_send_materialized_view(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
      static void handle_deferred_materialized_view(Runtime *runtime, 
                                                    const void *args);
      static void create_remote_materialized_view(Runtime *runtime,
                                                  DistributedID did,
                                                  AddressSpaceID owner_space,
                                                  AddressSpaceID logical_owner,
                                                  RegionTreeNode *target_node,
                                                  PhysicalManager *manager,
                                                  MaterializedView *parent,
                                                  UniqueID context_uid);
#ifdef DISTRIBUTED_INSTANCE_VIEWS
    public:
      void perform_remote_valid_check(const FieldMask &check_mask,
                                      VersionTracker *version_tracker,
                                      bool reading,
                                      std::set<RtEvent> *wait_on = NULL);
      void perform_read_invalidations(const FieldMask &check_mask,
                                      VersionTracker *version_tracker,
                                      const AddressSpaceID source,
                                      std::set<RtEvent> &applied_events);
      void send_invalidations(const FieldMask &invalidate_mask,
                              const AddressSpaceID can_skip,
                              std::set<RtEvent> &applied_events);
    public:
      virtual void process_update_request(AddressSpaceID source,
                               RtUserEvent done_event, Deserializer &derez);
      virtual void process_update_response(Deserializer &derez,
                                           RtUserEvent done_event,
                                           AddressSpaceID source,
                                           RegionTreeForest *forest);
      virtual void process_remote_update(Deserializer &derez,
                                         AddressSpaceID source,
                                         RegionTreeForest *forest);
      virtual void process_remote_invalidate(const FieldMask &invalid_mask,
                                             RtUserEvent done_event); 
#else
    public:
      static void handle_filter_invalid_fields_request(Deserializer &derez,
                                  Runtime *runtime, AddressSpaceID source);
      static void handle_filter_invalid_fields_response(Deserializer &derez);
#endif
    public:
      InstanceManager *const manager;
      MaterializedView *const parent;
      const bool disjoint_children;
    protected:
      // Keep track of the locks used for managing atomic coherence
      // on individual fields of this materialized view. Only the
      // top-level view for an instance needs to track this.
      std::map<FieldID,Reservation> atomic_reservations;
      // Keep track of the child views
      std::map<LegionColor,MaterializedView*> children;
      // There are three operations that are done on materialized views
      // 1. iterate over all the users for use analysis
      // 2. garbage collection to remove old users for an event
      // 3. send updates for a certain set of fields
      // The first and last both iterate over the current and previous
      // user sets, while the second one needs to find specific events.
      // Therefore we store the current and previous sets as maps to
      // users indexed by events. Iterating over the maps are no worse
      // that iterating over lists (for arbitrary insertion and deletion)
      // and will provide fast indexing for removing items. We used to
      // store users in current and previous epochs similar to logical
      // analysis, but have since switched over to storing readers and
      // writers that are not filtered as part of analysis. This let's
      // us perform more analysis in parallel since we'll only need to
      // hold locks in read-only mode prevent user fragmentation. It also
      // deals better with the common case which are higher views in
      // the view tree that less frequently filter their sub-users.
      LegionMap<ApEvent,EventUsers>::aligned current_epoch_users;
      LegionMap<ApEvent,EventUsers>::aligned previous_epoch_users;
      // Also keep a set of events for which we have outstanding
      // garbage collection meta-tasks so we don't launch more than one
      // We need this even though we have the data structures above because
      // an event might be filtered out for some fields, so we can't rely
      // on it to detect when we have outstanding gc meta-tasks
      std::set<ApEvent> outstanding_gc_events;
      // Keep track of the current version numbers for each field
      // This will allow us to detect when physical instances are no
      // longer valid from a particular view when doing rollbacks for
      // resilience or mis-speculation.
      LegionMap<VersionID,FieldMask,
                PHYSICAL_VERSION_ALLOC>::track_aligned current_versions;
#ifdef DISTRIBUTED_INSTANCE_VIEWS
    protected:
      // The scheme for tracking whether remote copies of the meta-data
      // are valid is as follows:
      //
      // For readers:
      //  - At the base view: must be at the current version number
      //  - At an above view: must be at the current version number
      //    minus the split mask
      // These two cases allow us to find Read-After-Write (true) dependences
      //
      // For writers:
      //  - They must have a valid lease tracked by 'remote_valid_mask'
      //
      // Mask invalidation is as follows:
      //  - Writes that advance the version number invalidate the lease so
      //    we can get the Write-After-Write (anti) dependences correct
      //  - Any read invalidates the lease so that we can get the
      //    Write-After-Read (anti) dependences correct
      //
      // Note that this scheme still permits as many reads to occur in
      // parallel on different nodes, and writes to disjoint sub-regions
      // to occur in parallel without unnecessary invalidations.

      // The logical owner node maintains a data structure to track which
      // remote copies of this view have valid field data, whenever the
      // set of users gets filtered for a field from current to previous
      // then the logical owner send invalidate messages
      LegionMap<AddressSpaceID,FieldMask>::aligned valid_remote_instances;
      // On remote nodes, this field mask tracks whether we have a
      // valid lease from the logical owner node. On the owner node it
      // tracks a summary of all the fields that have remote leases.
      FieldMask remote_valid_mask;
      // Remote nodes also have a data structure for deduplicating
      // requests to the logical owner for updates to particular fields
      LegionMap<RtEvent,FieldMask>::aligned remote_update_requests;
#endif
    protected:
      // Useful for pruning the initial users at cleanup time
      std::set<ApEvent> initial_user_events;
    };

    /**
     * \class ReductionView
     * The ReductionView class is used for providing a view
     * onto reduction physical instances from any logical perspective.
     */
    class ReductionView : public InstanceView,
                          public LegionHeapify<ReductionView> {
    public:
      static const AllocationType alloc_type = REDUCTION_VIEW_ALLOC;
    public:
      struct EventUsers {
      public:
        EventUsers(void)
          : single(true) { users.single_user = NULL; }
      public:
        EventUsers& operator=(const EventUsers &rhs)
          { assert(false); return *this; }
      public:
        FieldMask user_mask;
        union {
          PhysicalUser *single_user;
          LegionMap<PhysicalUser*,FieldMask>::aligned *multi_users;
        } users;
        bool single;
      };
    public:
      ReductionView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc,
                    AddressSpaceID logical_owner, RegionTreeNode *node, 
                    ReductionManager *manager, UniqueID owner_context,
                    bool register_now);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      void perform_reduction(InstanceView *target, const FieldMask &copy_mask, 
                             VersionTracker *version_tracker, 
                             Operation *op, unsigned index,
                             std::set<RtEvent> &map_applied_events,
                             PredEvent pred_guard,
                             const PhysicalTraceInfo &trace_info,
                             bool restrict_out = false);
      ApEvent perform_deferred_reduction(MaterializedView *target,
                                         const FieldMask &reduction_mask,
                                         VersionTracker *version_tracker,
                                         ApEvent dst_precondition,
                                         Operation *op, unsigned index,
                                         PredEvent predicate_guard,
                                         CopyAcrossHelper *helper,
                                         RegionTreeNode *intersect,
                                         IndexSpaceExpression *mask,
                                         std::set<RtEvent> &map_applied_events,
                                         const PhysicalTraceInfo &trace_info,
                                         IndexSpaceExpression *&reduce_expr);
    public:
      virtual bool has_manager(void) const { return true; } 
      virtual PhysicalManager* get_manager(void) const;
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; } 
      virtual LogicalView* get_subview(const LegionColor c);
      virtual Memory get_location(void) const;
      virtual bool has_space(const FieldMask &space_mask) const
        { return false; }
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           bool single_copy/*only for writing*/,
                                           bool restrict_out,
                                           const FieldMask &copy_mask,
                                           IndexSpaceExpression *copy_expr,
                                           VersionTracker *version_tracker,
                                           const UniqueID creator_op_id,
                                           const unsigned index,
                                           const AddressSpaceID source,
                         LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info,
                                           bool can_filter = true);
      virtual void add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                 VersionTracker *version_tracker,
                                 IndexSpaceExpression *copy_expr,
                                 const UniqueID creator_op_id,
                                 const unsigned index,
                                 const FieldMask &mask, 
                                 bool reading, bool restrict_out,
                                 const AddressSpaceID source,
                                 std::set<RtEvent> &applied_events,
                                 const PhysicalTraceInfo &trace_info);
      virtual ApEvent find_user_precondition(const RegionUsage &user,
                                           ApEvent term_event,
                                           const FieldMask &user_mask,
                                           const UniqueID op_id,
                                           const unsigned index,
                                           VersionTracker *version_tracker,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info);
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, 
                            Operation *op, const unsigned index,
                            VersionTracker *version_tracker,
                            std::set<RtEvent> &applied_events,
                            const PhysicalTraceInfo &trace_info);
      // This is a fused version of the above two methods
      virtual ApEvent add_user_fused(const RegionUsage &user,ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
                                   VersionTracker *version_tracker,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info,
                                   bool update_versions = true);
      virtual void add_user_base(const RegionUsage &user, ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   const UniqueID op_id, const unsigned index, 
                                   const AddressSpaceID source,
                                   VersionTracker *version_tracker,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info);
      virtual ApEvent add_user_fused_base(const RegionUsage &user,
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   const UniqueID op_id, const unsigned index,
                                   VersionTracker *version_tracker,
                                   const AddressSpaceID source,
                                   std::set<RtEvent> &applied_events,
                                   const PhysicalTraceInfo &trace_info,
                                   bool update_versions = true);
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    const UniqueID op_id,
                                    const unsigned index);
    protected:
      void find_reducing_preconditions(const FieldMask &user_mask,
                                       ApEvent term_event,
                                       std::set<ApEvent> &wait_on);
      void find_reading_preconditions(const FieldMask &user_mask,
                                      ApEvent term_event,
                                      std::set<ApEvent> &wait_on);
    public:
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<CopySrcDstField> &dst_fields,
                             CopyAcrossHelper *across_helper = NULL);
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<CopySrcDstField> &src_fields);
      virtual void reduce_from(ReductionOpID redop,
                               const FieldMask &reduce_mask,
                          std::vector<CopySrcDstField> &src_fields);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void collect_users(const std::set<ApEvent> &term_events);
      virtual void update_gc_events(const std::set<ApEvent> &term_events);
    public:
      virtual void send_view(AddressSpaceID target); 
    protected:
      void add_physical_user(PhysicalUser *user, bool reading,
                             ApEvent term_event, const FieldMask &user_mask);
      void filter_local_users(ApEvent term_event);
    public:
      static void handle_send_reduction_view(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
#ifdef DISTRIBUTED_INSTANCE_VIEWS
    public:
      void perform_remote_valid_check(void);
      virtual void process_update_request(AddressSpaceID source,
                               RtUserEvent done_event, Deserializer &derez);
      virtual void process_update_response(Deserializer &derez,
                                           RtUserEvent done_event,
                                           AddressSpaceID source,
                                           RegionTreeForest *forest);
      virtual void process_remote_update(Deserializer &derez,
                                         AddressSpaceID source,
                                         RegionTreeForest *forest);
      virtual void process_remote_invalidate(const FieldMask &invalid_mask,
                                             RtUserEvent done_event);
#endif
    public:
      ReductionOpID get_redop(void) const;
    public:
      ReductionManager *const manager;
    protected:
      LegionMap<ApEvent,EventUsers>::aligned reduction_users;
      LegionMap<ApEvent,EventUsers>::aligned reading_users;
      std::set<ApEvent> outstanding_gc_events;
    protected:
      std::set<ApEvent> initial_user_events; 
#ifdef DISTRIBUTED_INSTANCE_VIEWS
    protected:
      // the request event for reducers
      // only needed on remote views
      RtEvent remote_request_event; 
#endif
    };

    /**
     * \class ShardedWriteTracker
     * A sharded write tracker is used for tracking the write
     * sets of composite copy requests from remote shards for a
     * particular field. It actually does this by tracking the 
     * complement of the write set (set of things not written
     * because they were already valid) since the common case will
     * be that we actually do write to instances from remote shards.
     */
    class ShardedWriteTracker : public Collectable {
    public:
      struct ShardedWriteTrackerArgs : 
        public LgTaskArgs<ShardedWriteTrackerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_COMPUTE_SHARDED_WRITE_TASK_ID;
      public:
        ShardedWriteTrackerArgs(ShardedWriteTracker *t)
          : LgTaskArgs<ShardedWriteTrackerArgs>(implicit_provenance),
            tracker(t) { }
      public:
        ShardedWriteTracker *const tracker;
      };
    public:
      ShardedWriteTracker(unsigned field_index, RegionTreeForest *forest,
                          IndexSpaceExpression *upper_bound,
                          ShardedWriteTracker *remote_tracker = NULL,
                          RtUserEvent event = RtUserEvent::NO_RT_USER_EVENT,
                          AddressSpaceID remote_target = 0);
      ShardedWriteTracker(const ShardedWriteTracker &rhs);
      ~ShardedWriteTracker(void);
    public:
      ShardedWriteTracker& operator=(const ShardedWriteTracker &rhs);
    public:
      void pack_for_remote_shard(Serializer &rez);
      void record_valid_expression(IndexSpaceExpression *expr);
      void record_sub_expression(IndexSpaceExpression *expr);
      // Return true if we can delete the object
      bool arm(void);
    public:
      void evaluate(void);
    public:
      static void handle_evaluate(const void *args);
      static void unpack_tracker(Deserializer &derez,
                     ShardedWriteTracker *&tracker, RtUserEvent &event);
      static ShardedWriteTracker* unpack_tracker(unsigned field_index,
          AddressSpaceID source, Runtime *runtime, Deserializer &derez);
      static void send_shard_valid(Runtime *rt, ShardedWriteTracker *tracker,
                          AddressSpaceID target, IndexSpaceExpression *expr,
                          RtUserEvent done_event);
      static void send_shard_sub(Runtime *rt, ShardedWriteTracker *tracker,
                          AddressSpaceID target, IndexSpaceExpression *expr,
                          RtUserEvent done_event);
      static void process_shard_summary(Deserializer &derez, 
          RegionTreeForest *forest, AddressSpaceID source);
    public:
      const unsigned field_index;
      RegionTreeForest *const forest;
      IndexSpaceExpression *const upper_bound;
      PendingIndexSpaceExpression *const pending_expr;
      // In case we're a remote copy of a tracker
      ShardedWriteTracker *const remote_tracker;
      const RtUserEvent remote_event;
      const AddressSpaceID remote_target;
    protected:
      mutable LocalLock expr_lock;
      std::set<RtEvent> remote_events;
      std::set<IndexSpaceExpression*> valid_expressions;
      std::set<IndexSpaceExpression*> sub_expressions;
    };

    /**
     * \struct DeferredCopier 
     * This is a helper class for performing copies from a deferred 
     * instance. It stores all of the arguments that need to be passed 
     * through to all of the methods that are used to issue copies
     * and reductions from a deferred view.
     */
    struct DeferredCopier {
    public:
      struct PendingReduction {
      public:
        PendingReduction(void)
          : version_tracker(NULL), intersect(NULL), mask(NULL) { }
        PendingReduction(const FieldMask &m, VersionTracker *vt,
            PredEvent g, RegionTreeNode *i, IndexSpaceExpression *e)
          : reduction_mask(m), version_tracker(vt), pred_guard(g),
            intersect(i), mask(e) { }
      public:
        FieldMask reduction_mask;
        VersionTracker *version_tracker;
        PredEvent pred_guard;
        RegionTreeNode *intersect;
        IndexSpaceExpression *mask;
      };
      typedef std::map<ReductionView*,
                       LegionList<PendingReduction>::aligned> PendingReductions;
#ifndef DISABLE_CVOPT
      struct ShardInfo {
      public:
        ShardInfo(void) { }
        ShardInfo(ShardID s, ReplicationID r, RtEvent b)
          : shard(s), repl_id(r), shard_invalid_barrier(b) { }
      public:
        inline bool operator==(const ShardInfo &rhs) const
        { if (shard != rhs.shard) return false;
          if (repl_id != rhs.repl_id) return false;
          if (shard_invalid_barrier != rhs.shard_invalid_barrier) return false;
          return true; }
        inline bool operator!=(const ShardInfo &rhs) const
          { return !((*this) == rhs); }
        inline bool operator<(const ShardInfo &rhs) const
        { if (shard < rhs.shard) return true;
          if (shard > rhs.shard) return false;
          if (repl_id < rhs.repl_id) return true;
          if (repl_id > rhs.repl_id) return false;
          return (shard_invalid_barrier < rhs.shard_invalid_barrier); }
      public:
        ShardID shard;
        ReplicationID repl_id;
        RtEvent shard_invalid_barrier;
      };
      struct ReductionShard {
      public:
        ReductionShard(void)
          : mask(NULL) { }
        ReductionShard(const FieldMask &m, PredEvent g, IndexSpaceExpression *e)
          : reduction_mask(m), pred_guard(g), mask(e) { }
      public:
        FieldMask reduction_mask;
        PredEvent pred_guard;
        IndexSpaceExpression *mask;
      };
      typedef std::map<ShardInfo,
                   LegionDeque<ReductionShard>::aligned> PendingReductionShards;
#endif
    public:
      DeferredCopier(const TraversalInfo *info, 
                     InnerContext *context,
                     MaterializedView *dst, 
                     const FieldMask &copy_mask, 
                     const RestrictInfo &restrict_info,
                     bool restrict_out);
      // For handling deferred copies across
      DeferredCopier(const TraversalInfo *info, 
                     InnerContext *context,
                     MaterializedView *dst, 
                     const FieldMask &copy_mask,
                     ApEvent precondition,
                     CopyAcrossHelper *helper = NULL);
      DeferredCopier(const DeferredCopier &rhs);
      virtual ~DeferredCopier(void);
    public:
      DeferredCopier& operator=(const DeferredCopier &rhs);
    public:
      virtual bool is_remote(void) const { return false; }
    public:
      void merge_destination_preconditions(const FieldMask &copy_mask,
                LegionMap<ApEvent,FieldMask>::aligned &preconditions);
      void buffer_reductions(VersionTracker *tracker, PredEvent pred_guard, 
                             RegionTreeNode *intersect,
                             const WriteMasks &write_masks,
               LegionMap<ReductionView*,FieldMask>::aligned &source_reductions);
#ifndef DISABLE_CVOPT
      void buffer_reduction_shards(PredEvent pred_guard, 
              ReplicationID repl_id, RtEvent shard_invalid_barrier,
              const LegionMap<ShardID,WriteMasks>::aligned &reduction_shards);
#endif
      void begin_guard_protection(void);
      void end_guard_protection(void);
      void begin_reduction_epoch(void);
      void end_reduction_epoch(void);
      void record_previously_valid(IndexSpaceExpression *expr, 
                                   const FieldMask &mask);
      void finalize(DeferredView *src_view,
                    std::set<ApEvent> *postconditions = NULL);
#ifndef DISABLE_CVOPT
      void pack_copier(Serializer &rez, const FieldMask &copy_mask);
#endif
      void pack_sharded_write_tracker(unsigned field_index, Serializer &rez);
      inline bool has_reductions(void) const 
        { return !reduction_epochs.empty(); }
    protected:
      void uniquify_copy_postconditions(void);
      void compute_dst_preconditions(const FieldMask &mask);
      void apply_reduction_epochs(WriteSet &reduce_exprs);
      bool issue_reductions(const int epoch, ApEvent reduction_pre, 
                            const FieldMask &mask, WriteSet &reduce_exprs,
              LegionMap<ApEvent,FieldMask>::aligned &reduction_postconditions);
      void arm_write_trackers(WriteSet &reduce_exprs, bool add_reference);
      void compute_actual_dst_exprs(IndexSpaceExpression *dst_expr,
                                    WriteSet &reduce_exprs,
              std::vector<IndexSpaceExpression*> &actual_dst_exprs,
              LegionVector<FieldMask>::aligned &previously_valid_masks);
    public: // const fields
      const TraversalInfo *const info;
      InnerContext *const shard_context;
      MaterializedView *const dst;
      CopyAcrossHelper *const across_helper;
      const RestrictInfo *const restrict_info;
      const bool restrict_out;
    public: // visible mutable fields
      FieldMask deferred_copy_mask;
      LegionMap<ApEvent,FieldMask>::aligned copy_postconditions;
    protected: // internal members
      LegionMap<ApEvent,FieldMask>::aligned dst_preconditions;
      FieldMask dst_precondition_mask;
    protected: 
      // Reduction data 
      unsigned current_reduction_epoch;
      std::vector<PendingReductions> reduction_epochs;
    protected:
      // Keep track of expressions for data that was already valid
      // in the desintation instance, this will allow us to compute
      // an expression for the actual write set of the copy
      WriteSet dst_previously_valid;
      // For control replication computations of dst_previously_valid
      std::map<unsigned/*field index*/,ShardedWriteTracker*> write_trackers;
#ifndef DISABLE_CVOPT
      std::vector<PendingReductionShards> reduction_shards;
#endif
      LegionVector<FieldMask>::aligned reduction_epoch_masks;
    protected:
      // Handle protection of events for guarded operations
      std::vector<LegionMap<ApEvent,FieldMask>::aligned> protected_copy_posts;
#ifdef DEBUG_LEGION
      bool finalized;
#endif
    };

#ifndef DISABLE_CVOPT
    /**
     * \struct RemoteDeferredCopier
     * This is a version of the above copier that is used for
     * handling sharded copies on remote nodes
     */
    struct RemoteDeferredCopier : public DeferredCopier, 
      public LegionHeapify<RemoteDeferredCopier> {
    public:
      RemoteDeferredCopier(RemoteTraversalInfo *info, 
                           InnerContext *context,
                           MaterializedView *dst, 
                           const FieldMask &copy_mask,
                           CopyAcrossHelper *helper);
      RemoteDeferredCopier(const RemoteDeferredCopier &rhs);
      virtual ~RemoteDeferredCopier(void);
    public:
      RemoteDeferredCopier& operator=(const RemoteDeferredCopier &rhs);
    public:
      virtual bool is_remote(void) const { return true; }
    public:
      void unpack(Deserializer &derez, const FieldMask &copy_mask);
      void unpack_write_tracker(unsigned field_index, AddressSpaceID source,
                                Runtime *runtime, Deserializer &derez);
      void finalize(std::map<unsigned,ApUserEvent> &done_events);
    public:
      static RemoteDeferredCopier* unpack_copier(Deserializer &derez, 
            Runtime *runtime, const FieldMask &copy_mask, InnerContext *ctx);
    public:
      RemoteTraversalInfo *const remote_info;
    };
#endif

    /**
     * \struct DeferredSingleCopier
     * This is a specialized class for doing deferred copies
     * from a composite view for a single field.
     */
    struct DeferredSingleCopier {
    public:
      struct PendingReduction {
      public:
        PendingReduction(void)
          : version_tracker(NULL), intersect(NULL), mask(NULL) { }
        PendingReduction(VersionTracker *vt, PredEvent g, 
                         RegionTreeNode *i, IndexSpaceExpression *e)
          : version_tracker(vt), pred_guard(g), intersect(i), mask(e) { }
      public:
        VersionTracker *version_tracker;
        PredEvent pred_guard;
        RegionTreeNode *intersect;
        IndexSpaceExpression *mask;
      };
      typedef std::map<ReductionView*,PendingReduction> PendingReductions;
#ifndef DISABLE_CVOPT
      struct ShardInfo {
      public:
        ShardInfo(void) { }
        ShardInfo(ShardID s, ReplicationID r, RtEvent b)
          : shard(s), repl_id(r), shard_invalid_barrier(b) { }
      public:
        inline bool operator==(const ShardInfo &rhs) const
        { if (shard != rhs.shard) return false;
          if (repl_id != rhs.repl_id) return false;
          if (shard_invalid_barrier != rhs.shard_invalid_barrier) return false;
          return true; }
        inline bool operator!=(const ShardInfo &rhs) const
          { return !((*this) == rhs); }
        inline bool operator<(const ShardInfo &rhs) const
        { if (shard < rhs.shard) return true;
          if (shard > rhs.shard) return false;
          if (repl_id < rhs.repl_id) return true;
          if (repl_id > rhs.repl_id) return false;
          return (shard_invalid_barrier < rhs.shard_invalid_barrier); }
      public:
        ShardID shard;
        ReplicationID repl_id;
        RtEvent shard_invalid_barrier;
      };
      struct ReductionShard {
      public:
        ReductionShard(void) { }
        ReductionShard(PredEvent g, IndexSpaceExpression *e)
          : pred_guard(g), mask(e) { }
      public:
        PredEvent pred_guard;
        IndexSpaceExpression *mask;
      };
      typedef std::map<ShardInfo,ReductionShard> PendingReductionShards;
#endif
    public:
      DeferredSingleCopier(const TraversalInfo *info, 
                           InnerContext *context,
                           MaterializedView *dst, 
                           const FieldMask &copy_mask,
                           const RestrictInfo &restrict_info,
                           bool restrict_out);
      // For handling deferred copies across
      DeferredSingleCopier(const TraversalInfo *info, 
                           InnerContext *context,
                           MaterializedView *dst, 
                           const FieldMask &copy_mask,
                           ApEvent precondition,
                           CopyAcrossHelper *helper = NULL);
      DeferredSingleCopier(const DeferredSingleCopier &rhs);
      virtual ~DeferredSingleCopier(void);
    public:
      DeferredSingleCopier& operator=(const DeferredSingleCopier &rhs);
    public:
      virtual bool is_remote(void) const { return false; }
    public:
      void merge_destination_preconditions(std::set<ApEvent> &preconditions);
      void buffer_reductions(VersionTracker *tracker, PredEvent pred_guard,
                             RegionTreeNode *intersect, IndexSpaceExpression *mask,
                             std::vector<ReductionView*> &source_reductions);
#ifndef DISABLE_CVOPT
      void buffer_reduction_shards(PredEvent pred_guard, 
          ReplicationID repl_id, RtEvent shard_invalid_barrier,
          const std::map<ShardID,IndexSpaceExpression*> &source_reductions);
#endif
      void begin_guard_protection(void);
      void end_guard_protection(void);
      void begin_reduction_epoch(void);
      void end_reduction_epoch(void);
      void record_previously_valid(IndexSpaceExpression *expr);
      void finalize(DeferredView *src_view,
                    std::set<ApEvent> *postconditions = NULL);
      void arm_write_tracker(const std::set<IndexSpaceExpression*> &reduce_exps,
                             bool add_reference);
#ifndef DISABLE_CVOPT
      void pack_copier(Serializer &rez);
#endif
      void pack_sharded_write_tracker(Serializer &rez);
      inline void record_postcondition(ApEvent post)
        { copy_postconditions.insert(post); }
      inline bool has_reductions(void) const 
        { return !reduction_epochs.empty(); }
    protected:
      void compute_dst_preconditions(void);
      void apply_reduction_epochs(std::set<IndexSpaceExpression*> &red_exprs);
    public: // const fields
      const unsigned field_index;
      const FieldMask copy_mask;
      const TraversalInfo *const info;
      InnerContext *const shard_context;
      MaterializedView *const dst;
      CopyAcrossHelper *const across_helper;
      const RestrictInfo *const restrict_info;
      const bool restrict_out;
    public:
      std::set<ApEvent> copy_postconditions;
    protected: // internal members
      // Keep track of expressions for data that was already valid
      // in the desintation instance, this will allow us to compute
      // an expression for the actual write set of the copy
      std::set<IndexSpaceExpression*> dst_previously_valid;
      unsigned current_reduction_epoch;
      std::vector<PendingReductions> reduction_epochs;
      ShardedWriteTracker *write_tracker;
#ifndef DISABLE_CVOPT
      std::vector<PendingReductionShards> reduction_shards;
#endif
    protected:
      std::set<ApEvent> dst_preconditions;
      std::vector<std::set<ApEvent> > protected_copy_posts;
      bool has_dst_preconditions;
#ifdef DEBUG_LEGION
      bool finalized;
#endif
    };

#ifndef DISABLE_CVOPT
    /**
     * \struct RemoteDeferredSingleCopier
     * This is a version of the above copier that is used for
     * handling sharded copies on remote nodes
     */
    struct RemoteDeferredSingleCopier : public DeferredSingleCopier,
      public LegionHeapify<RemoteDeferredSingleCopier> {
    public:
      RemoteDeferredSingleCopier(RemoteTraversalInfo *info, 
                                 InnerContext *context,
                                 MaterializedView *dst, 
                                 const FieldMask &copy_mask,
                                 CopyAcrossHelper *helper);
      RemoteDeferredSingleCopier(const RemoteDeferredSingleCopier &rhs);
      virtual ~RemoteDeferredSingleCopier(void);
    public:
      RemoteDeferredSingleCopier& operator=(
                                 const RemoteDeferredSingleCopier &rhs);
    public:
      virtual bool is_remote(void) const { return true; }
    public:
      void unpack(Deserializer &derez);
      void unpack_write_tracker(AddressSpaceID source, Runtime *runtime,
                                Deserializer &derez);
      void finalize(ApUserEvent done_event);
    public:
      static RemoteDeferredSingleCopier* unpack_copier(Deserializer &derez,
               Runtime *runtime, const FieldMask &copy_mask, InnerContext *ctx);
    public:
      RemoteTraversalInfo *const remote_info;
    };
#endif

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
                   AddressSpaceID owner_space,
                   RegionTreeNode *node, bool register_now);
      virtual ~DeferredView(void);
    public:
      // Deferred views never have managers
      virtual bool has_manager(void) const { return false; }
      virtual PhysicalManager* get_manager(void) const
        { return NULL; }
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const LegionColor c) = 0;
      virtual bool has_space(const FieldMask &space_mask) const
        { return false; }
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_owner_inactive(ReferenceMutator *mutator) 
        { assert(false); }
    public:
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
      virtual void notify_owner_invalid(ReferenceMutator *mutator)
        { assert(false); }
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      // Should never be called directly
      virtual InnerContext* get_shard_context(void) const = 0;
    public:
      // Should never be called directly
      virtual void collect_users(const std::set<ApEvent> &term_events)
        { assert(false); }
      // Should never be called
      virtual void update_gc_events(const std::set<ApEvent> &term_events)
        { assert(false); }
    public:
      void issue_deferred_copies_across(const TraversalInfo &info,
                                        MaterializedView *dst,
                                  const std::vector<unsigned> &src_indexes,
                                  const std::vector<unsigned> &dst_indexes,
                                        ApEvent precondition, PredEvent guard,
                                        std::set<ApEvent> &postconditions);
    public:
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         FieldMask copy_mask,
                                         const RestrictInfo &restrict_info,
                                         bool restrict_out) = 0;
      virtual void issue_deferred_copies(DeferredCopier &copier,
                                         const FieldMask &local_copy_mask,
                                         const WriteMasks &write_masks,
                                         WriteSet &performed_writes,
                                         PredEvent pred_guard) = 0;
      virtual bool issue_deferred_copies_single(DeferredSingleCopier &copier,
                                         IndexSpaceExpression *write_mask,
                                         IndexSpaceExpression *&write_performed,
                                         PredEvent pred_guard) = 0;
#ifdef DEBUG_LEGION
    protected:
      bool currently_active;
      bool currently_valid;
#endif
    };

    /**
     * \class DeferredVersionInfo
     * This is a wrapper class for keeping track of the version
     * information for all the composite nodes in a composite instance.
     * TODO: do we need synchronization on computing field versions
     * because these objects can be shared between composite views
     */
    class DeferredVersionInfo : public VersionInfo, public Collectable {
    public:
      DeferredVersionInfo(void);
      DeferredVersionInfo(const DeferredVersionInfo &rhs);
      ~DeferredVersionInfo(void);
    public:
      DeferredVersionInfo& operator=(const DeferredVersionInfo &rhs);
    };

    /**
     * \struct CompositeReducer
     * A helper class for issuing reductions from a composite view
     */
    struct CompositeReducer {
    public:
      CompositeReducer(TraversalInfo *info, 
                       InnerContext *context,
                       MaterializedView *dst, 
                       const FieldMask &reduce_mask,
                       CopyAcrossHelper *helper);
      CompositeReducer(const CompositeReducer &rhs);
      ~CompositeReducer(void);
    public:
      CompositeReducer& operator=(const CompositeReducer &rhs);
    public:
      void unpack(Deserializer &derez);
      void unpack_write_tracker(unsigned field_index, Deserializer &derez);
      ApEvent find_precondition(const FieldMask &mask) const;
      void record_postcondition(ApEvent done, const FieldMask &mask);
      void record_expression(IndexSpaceExpression *expr, const FieldMask &mask);
      void finalize(std::map<unsigned,ApUserEvent> &done_events,
                    Runtime *runtime, AddressSpaceID target);
    public:
      TraversalInfo *const info;
      InnerContext *const context;
      MaterializedView *const dst;
      const FieldMask reduction_mask;
      CopyAcrossHelper *const across_helper;
    protected:
      LegionMap<ApEvent,FieldMask>::aligned reduce_preconditions;
      LegionMap<ApEvent,FieldMask>::aligned reduce_postconditions;
    protected:
      // For sending back to our sharded write trackers
      WriteSet reduce_expressions;
      std::map<unsigned/*fidx*/,
               std::pair<ShardedWriteTracker*,RtUserEvent> > remote_trackers;
    };

    /**
     * \struct CompositeSingleReducer
     * A helper class for issuing reductions from a composite view
     * for only a single field
     */
    struct CompositeSingleReducer {
    public:
      CompositeSingleReducer(TraversalInfo *info, 
                             InnerContext *context,
                             MaterializedView *dst, 
                             const FieldMask &reduce_mask,
                             ApEvent reduce_pre,
                             CopyAcrossHelper *helper);
      CompositeSingleReducer(const CompositeSingleReducer &rhs);
      ~CompositeSingleReducer(void);
    public:
      CompositeSingleReducer& operator=(const CompositeSingleReducer &rhs);
    public:
       inline void record_postcondition(ApEvent post)
        { reduce_postconditions.insert(post); }
      void unpack_write_tracker(Deserializer &derez);
      inline void record_expression(IndexSpaceExpression *expr)
        { if (remote_tracker != NULL) reduce_expressions.insert(expr); }
      void finalize(ApUserEvent done_event, Runtime *rt, AddressSpaceID target);
    public:
      TraversalInfo *const info;
      InnerContext *const context;
      MaterializedView *const dst;
      const FieldMask reduction_mask;
      const unsigned field_index;
      const ApEvent reduce_pre;
      CopyAcrossHelper *const across_helper;
    protected:
      std::set<ApEvent> reduce_postconditions;
    protected:
      // For sending back to our sharded write tracker
      std::set<IndexSpaceExpression*> reduce_expressions;
      ShardedWriteTracker *remote_tracker;
      RtUserEvent remote_event;
    };

    /**
     * \class CompositeBase
     * A small helper class that provides some base functionality
     * for both the CompositeView and CompositeNode classes
     */
    class CompositeBase {
    public:
      CompositeBase(LocalLock &base_lock, bool composite_shard);
      virtual ~CompositeBase(void);
    protected:
      void issue_composite_updates(DeferredCopier &copier,
                                   RegionTreeNode *logical_node,
                                   const FieldMask &copy_mask,
                                   VersionTracker *src_version_tracker,
                                   PredEvent pred_guard, 
                                   const WriteMasks &write_masks,
                                   WriteSet &performed_writes/*write-only*/,
                                   bool need_shard_check = true);
      // Single field version of the method above
      bool issue_composite_updates_single(DeferredSingleCopier &copier,
                                   RegionTreeNode *logical_node,
                                   VersionTracker *src_version_tracker,
                                   PredEvent pred_guard,
                                   IndexSpaceExpression *write_mask,
                                   // Only valid if !done
                                   IndexSpaceExpression *&performed,
                                   bool need_shard_check = true);
    protected:
      void issue_composite_reductions(CompositeReducer &reducer,
                                      const FieldMask &local_mask,
                                      const WriteMasks &needed_expressions,
                                      RegionTreeNode *logical_node,
                                      PredEvent pred_guard,
                                      VersionTracker *src_version_tracker);
      // Single field version of the method above
      void issue_composite_reductions_single(CompositeSingleReducer &reducer,
                                             IndexSpaceExpression *needed_expr,
                                             RegionTreeNode *logical_node,
                                             PredEvent pred_guard,
                                             VersionTracker *version_tracker);
    public:
      static void issue_update_copies(DeferredCopier &copier, 
                               RegionTreeNode *logical_node,FieldMask copy_mask,
                               VersionTracker *src_version_tracker,
                               PredEvent predcate_guard,
               const LegionMap<LogicalView*,FieldMask>::aligned &source_views,
                               // previous_writes Should be field unique
                               const WriteMasks &previous_writes,
                                     WriteSet &performed_writes);
      // Write combining unions together all index space expressions for
      // the same field so that we get one index expression for each field
      static void combine_writes(WriteMasks &write_masks,
                                 DeferredCopier &copier,
                                 bool prune_global = true);
    public:
      static IndexSpaceExpression* issue_update_copies_single(
                                    DeferredSingleCopier &copier,
                                    RegionTreeNode *logical_node,
                                    VersionTracker *version_tracker,
                                    PredEvent pred_guard,
                                    std::vector<LogicalView*> &source_views,
                                    IndexSpaceExpression *write_mask);
    public:
      static bool test_done(DeferredSingleCopier &copier,
                            IndexSpaceExpression *write1,
                            IndexSpaceExpression *write2 = NULL);
    public:
      virtual InnerContext* get_owner_context(void) const = 0;
      virtual DistributedID get_owner_did(void) const = 0;
#ifndef DISABLE_CVOPT
      virtual void perform_ready_check(FieldMask mask) = 0;
#else
      virtual void perform_ready_check(FieldMask mask,
                                       RegionTreeNode *target) = 0;
#endif
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views, 
                                    bool needs_lock = true) = 0;
      virtual void unpack_composite_view_response(Deserializer &derez,
                                                  Runtime *runtime) = 0;
    public:
      virtual void issue_shard_updates(DeferredCopier &copier,
                                       RegionTreeNode *logical_node,
                                       const FieldMask &local_copy_mask,
                                       PredEvent pred_guard,
                                       const WriteMasks &write_masks,
                                             WriteMasks &performed_writes)
        { assert(false); }
      virtual IndexSpaceExpression *issue_shard_updates_single(
                                              DeferredSingleCopier &copier,
                                              RegionTreeNode *logical_node,
                                              PredEvent pred_guard,
                                              IndexSpaceExpression *write_mask)
        { assert(false); return NULL; }
    public:
      virtual void print_view_state(const FieldMask &capture_mask,
                                    TreeStateLogger* logger,
                                    int current_nesting,
                                    int max_nesting);
    public:
      CompositeNode* find_child_node(RegionTreeNode *child);
    private:
      LocalLock &base_lock;
      const bool composite_shard;
    protected:
      FieldMask dirty_mask, reduction_mask;
      LegionMap<CompositeNode*,FieldMask>::aligned children;
      LegionMap<LogicalView*,FieldMask>::aligned valid_views;
      LegionMap<ReductionView*,FieldMask>::aligned reduction_views; 
    }; 

    /**
     * \class CompositeView
     * The CompositeView class is used for deferring close
     * operations by representing a valid version of a single
     * logical region with a bunch of different instances.
     */
    class CompositeView : public DeferredView, 
                          public VersionTracker, 
                          public CompositeBase, 
                          public LegionHeapify<CompositeView> {
    public:
      static const AllocationType alloc_type = COMPOSITE_VIEW_ALLOC; 
    public:
      struct DeferCompositeViewRefArgs : 
        public LgTaskArgs<DeferCompositeViewRefArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COMPOSITE_VIEW_REF_TASK_ID;
      public:
        DeferCompositeViewRefArgs(DistributedCollectable *d, DistributedID id)
          : LgTaskArgs<DeferCompositeViewRefArgs>(implicit_provenance),
            dc(d), did(id) { }
      public:
        DistributedCollectable *const dc;
        const DistributedID did;
      };
      struct DeferCompositeViewRegistrationArgs : 
        public LgTaskArgs<DeferCompositeViewRegistrationArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_DEFER_COMPOSITE_VIEW_REGISTRATION_TASK_ID;
      public:
        DeferCompositeViewRegistrationArgs(CompositeView *v)
          : LgTaskArgs<DeferCompositeViewRegistrationArgs>(implicit_provenance),
            view(v) { }
      public:
        CompositeView *const view;
      };
      struct DeferInvalidateArgs :
        public LgTaskArgs<DeferInvalidateArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_DEFER_COMPOSITE_VIEW_INVALIDATION_TASK_ID;
      public:
        DeferInvalidateArgs(CompositeView *v)
          : LgTaskArgs<DeferInvalidateArgs>(implicit_provenance), view(v) { }
      public:
        CompositeView *const view;
      };
    public:
      struct NodeVersionInfo {
      public:
        FieldVersions versions;
        FieldMask valid_fields;
      };
    public:
      // A custom comparator for making sure that all our nested
      // composite views are traversed in order for when we are
      // control replicated and we have to guaranteed that things
      // are done in the same order across all shards
      struct NestedComparator {
      public:
        inline bool operator()(CompositeView* left, CompositeView *right) const
        {
          if (left->shard_invalid_barrier < right->shard_invalid_barrier)
            return true;
          else if (left->shard_invalid_barrier != right->shard_invalid_barrier)
            return false;
          else
            return (left < right);
        }
      };
      // Note we use a special comparator here to ensure that we always
      // iterate over nested composite views in the same way across all
      // shards for when we are doing control replication
      typedef LegionMap<CompositeView*,FieldMask,
                        LAST_ALLOC,NestedComparator>::aligned NestedViewMap;
    public:
      CompositeView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, RegionTreeNode *node, 
                    DeferredVersionInfo *info, CompositeViewSummary &summary,
                    InnerContext *context, bool register_now, 
                    ReplicationID repl_id = 0,
                    RtBarrier shard_invalid_barrier = RtBarrier::NO_RT_BARRIER,
                    ShardID origin_shard = 0);
      CompositeView(const CompositeView &rhs);
      virtual ~CompositeView(void);
    public:
      CompositeView& operator=(const CompositeView &rhs);
    public:
      CompositeView* clone(const FieldMask &clone_mask,
          const NestedViewMap &replacements, 
          ReferenceMutator *mutator, InterCloseOp *op) const;
    public:
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; }
      virtual LogicalView* get_subview(const LegionColor c);
    public:
      virtual void notify_owner_inactive(ReferenceMutator *mutator);
      virtual void notify_owner_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target);
      virtual InnerContext* get_shard_context(void) const
        { return owner_context; }
    public:
      void prune(const WriteMasks &partial_write_masks, FieldMask &valid_mask,
                 NestedViewMap &replacements, unsigned prune_depth, 
                 ReferenceMutator *mutator, InterCloseOp *op);
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         FieldMask copy_mask,
                                         const RestrictInfo &restrict_info,
                                         bool restrict_out);
      virtual void issue_deferred_copies(DeferredCopier &copier,
                                         const FieldMask &local_copy_mask,
                                         const WriteMasks &write_masks,
                                         WriteSet &performed_writes,
                                         PredEvent pred_guard);
      virtual bool issue_deferred_copies_single(DeferredSingleCopier &copier,
                                         IndexSpaceExpression *write_mask,
                                         IndexSpaceExpression *&write_performed,
                                         PredEvent pred_guard);
    public:
      // From VersionTracker
      virtual bool is_upper_bound_node(RegionTreeNode *node) const;
      virtual void get_field_versions(RegionTreeNode *node, bool split_prev,
                                      const FieldMask &needed_fields,
                                      FieldVersions &field_versions);
      virtual void get_advance_versions(RegionTreeNode *node, bool base,
                                        const FieldMask &needed_fields,
                                        FieldVersions &field_versions);
      virtual void get_split_mask(RegionTreeNode *node, 
                                  const FieldMask &needed_fields,
                                  FieldMask &split);
      virtual void pack_writing_version_numbers(Serializer &rez) const;
      virtual void pack_upper_bound_node(Serializer &rez) const;
    protected:
#ifndef DISABLE_CVOPT
      CompositeNode* capture_above(RegionTreeNode *node,
                                   const FieldMask &needed_fields);
#else
      CompositeNode* capture_above(RegionTreeNode *node,
                                   const FieldMask &needed_fields,
                                   RegionTreeNode *target);
#endif
    public:
      // From CompositeBase
      virtual InnerContext* get_owner_context(void) const;
      virtual DistributedID get_owner_did(void) const { return did; }
#ifndef DISABLE_CVOPT
      virtual void perform_ready_check(FieldMask mask);
#else
      virtual void perform_ready_check(FieldMask mask,
                                       RegionTreeNode *target);
#endif
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views,
                                    bool need_lock = true);
      virtual void unpack_composite_view_response(Deserializer &derez,
                                                  Runtime *runtime);
    public:
      virtual void issue_shard_updates(DeferredCopier &copier,
                                       RegionTreeNode *logical_node,
                                       const FieldMask &local_copy_mask,
                                       PredEvent pred_guard,
                                       const WriteMasks &write_masks,
                                             WriteMasks &performed_writes);
      virtual IndexSpaceExpression *issue_shard_updates_single(
                                              DeferredSingleCopier &copier,
                                              RegionTreeNode *logical_node,
                                              PredEvent pred_guard,
                                              IndexSpaceExpression *write_mask); 
#ifndef DISABLE_CVOPT
      void find_needed_shards(const FieldMask &mask, ShardID origin_shard,
            IndexSpaceExpression *target, 
            const WriteMasks &write_masks,
            LegionMap<ShardID,WriteMasks>::aligned &needed_shards,
            LegionMap<ShardID,WriteMasks>::aligned &reduction_shards) const;
      void find_needed_shards_single(const unsigned field_index, 
          const ShardID origin_shard, IndexSpaceExpression *target_expr,
          std::map<ShardID,IndexSpaceExpression*> &needed_shards,
          std::map<ShardID,IndexSpaceExpression*> &reduction_shards) const;
    protected:
      void find_interfering_shards(FieldMask mask, 
          const ShardID origin_shard, IndexSpaceExpression *target_expr,
          const WriteMasks &write_masks,
          const FieldMaskSet<ShardingSummary> &projections,
          LegionMap<ShardID,WriteMasks>::aligned &needed_shards) const;
      void find_interfering_shards_single(const unsigned field_index, 
          const ShardID origin_shard, IndexSpaceExpression *target_expr,
          const FieldMaskSet<ShardingSummary> &projections,
          std::map<ShardID,IndexSpaceExpression*> &needed_shards) const;
#else
      void find_needed_shards(FieldMask mask, RegionTreeNode *target,
                              std::set<ShardID> &needed_shards) const;
#endif
    public:
      static void handle_send_composite_view(Runtime *runtime, 
                              Deserializer &derez, AddressSpaceID source);
      static void handle_deferred_view_registration(const void *args);
      static void handle_deferred_view_invalidation(const void *args);
    public:
      void record_dirty_fields(const FieldMask &dirty_mask);
      void record_valid_view(LogicalView *view, FieldMask mask,
                             ReferenceMutator *mutator);
      void record_reduction_fields(const FieldMask &reduction_fields);
      void record_reduction_view(ReductionView *view, const FieldMask &mask,
                                 ReferenceMutator *mutator);
      void record_child_version_state(const LegionColor child_color, 
         VersionState *state, const FieldMask &mask, ReferenceMutator *mutator);
      void finalize_capture(bool need_prune, 
                            ReferenceMutator *mutator, InterCloseOp *op);
    public:
      void pack_composite_view(Serializer &rez) const;
      void unpack_composite_view(Deserializer &derez,
                                 std::set<RtEvent> &preconditions);
      RtEvent defer_add_reference(DistributedCollectable *dc, 
                                  RtEvent precondition) const;
      static void handle_deferred_view_ref(const void *args);
    public:
      virtual void print_view_state(const FieldMask &capture_mask,
                                    TreeStateLogger* logger,
                                    int current_nesting,
                                    int max_nesting);
    public:
      // For control replication
#ifndef DISABLE_CVOPT
      void handle_sharding_copy_request(Deserializer &derez, Runtime *runtime,
                                  InnerContext *ctx, AddressSpaceID source);
      void handle_sharding_reduction_request(Deserializer &derez, 
                      Runtime *rt, InnerContext *ctx, AddressSpaceID source);
#else
      void handle_sharding_update_request(Deserializer &derez,
                                          Runtime *runtime);
      static void handle_composite_view_response(Deserializer &derez,
                                                 Runtime *runtime);
#endif
    public:
      // The path version info for this composite instance
      DeferredVersionInfo *const version_info;
      // A summary of our composite view information
      const CompositeViewSummary summary;
      // The translation context if any
      InnerContext *const owner_context;
      // Things used for control replication of composite views
      const ReplicationID repl_id;
      const RtBarrier shard_invalid_barrier;
      const ShardID origin_shard;
    protected:
      // Note that we never record any version state names here, we just
      // record the views and children we immediately depend on and that
      // is how we break the inifinite meta-data cycle 
      NestedViewMap nested_composite_views;
    protected:
      LegionMap<RegionTreeNode*,NodeVersionInfo>::aligned node_versions;
#ifdef DISABLE_CVOPT
    protected:
      // Keep track of a packed version of tree for this shard 
      // when we are running in a control replication setting
      // so we can share it with other shards when they ask for it
      Serializer *packed_shard;
    protected:
      // For control replication to determine which shard checks
      // we've performed for different sub-nodes
      LegionMap<RegionTreeNode*,FieldMask>::aligned shard_checks; 
      std::map<ShardID,RtEvent> requested_shards;
#endif
    };

    /**
     * \class CompositeNode
     * A composite node is a read-only snapshot of the final state of
     * one or more version state objects. It's used for issuing
     * copy operations from closed region tree.
     */
    class CompositeNode : public CompositeBase,
                          public LegionHeapify<CompositeNode> {
    public:
      static const AllocationType alloc_type = COMPOSITE_NODE_ALLOC;
    public:
      struct DeferCompositeNodeStateArgs : 
        public LgTaskArgs<DeferCompositeNodeStateArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COMPOSITE_NODE_STATE_TASK_ID;
      public:
#ifdef DISABLE_CVOPT
        DeferCompositeNodeStateArgs(CompositeNode *proxy, VersionState *s,
                                    DistributedID own, FieldMask *m, bool r)
          : LgTaskArgs<DeferCompositeNodeStateArgs>(implicit_provenance),
            proxy_this(proxy), state(s), owner_did(own), 
            mask(m), root_owner(r) { }
#else
        DeferCompositeNodeStateArgs(CompositeNode *proxy, VersionState *s,
                                    DistributedID own, bool r)
          : LgTaskArgs<DeferCompositeNodeStateArgs>(implicit_provenance),
            proxy_this(proxy), state(s), owner_did(own), root_owner(r) { }
#endif
      public:
        CompositeNode *const proxy_this;
        VersionState *const state;
        const DistributedID owner_did;
#ifdef DISABLE_CVOPT
        FieldMask *const mask;
#endif
        const bool root_owner;
      };
      struct DeferCaptureArgs : public LgTaskArgs<DeferCaptureArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COMPOSITE_NODE_CAPTURE_TASK_ID;
      public:
        DeferCaptureArgs(CompositeNode *proxy, VersionState *state,
                         const FieldMask *mask)
          : LgTaskArgs<DeferCaptureArgs>(implicit_provenance),
            proxy_this(proxy), version_state(state), capture_mask(mask) { }
      public:
        CompositeNode *const proxy_this;
        VersionState *const version_state;
        const FieldMask *const capture_mask;
      };
    public:
      CompositeNode(RegionTreeNode *node, CompositeBase *parent,
                    DistributedID owner_did, bool root_owner);
      CompositeNode(const CompositeNode &rhs);
      virtual ~CompositeNode(void);
    public:
      CompositeNode& operator=(const CompositeNode &rhs);
    public:
      // From CompositeBase
      virtual InnerContext* get_owner_context(void) const;
      virtual DistributedID get_owner_did(void) const { return owner_did; }
#ifndef DISABLE_CVOPT
      virtual void perform_ready_check(FieldMask mask);
#else
      virtual void perform_ready_check(FieldMask mask,
                                       RegionTreeNode *target);
#endif
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views, 
                                    bool needs_lock = true);
      virtual void unpack_composite_view_response(Deserializer &derez,
                                                  Runtime *runtime);
      void capture(VersionState *state, const FieldMask &capture_mask,
                   ReferenceMutator *mutator);
    public:
      void capture(RtUserEvent capture_event, ReferenceMutator *mutator);
      static void handle_deferred_capture(const void *args);
    public:
      void clone(CompositeView *target, const FieldMask &clone_mask,
                 ReferenceMutator *mutator) const;
      void pack_composite_node(Serializer &rez) const;
      static CompositeNode* unpack_composite_node(Deserializer &derez,
                     CompositeView *parent, Runtime *runtime, 
                     DistributedID owner_did, std::set<RtEvent> &preconditions);
      // For merging composite node children from different shards
      static CompositeNode* unpack_composite_node(Deserializer &derez,
                 CompositeBase *parent, Runtime *runtime,
                 DistributedID owner_did, std::set<RtEvent> &preconditions,
                 const LegionMap<CompositeNode*,FieldMask>::aligned &existing,
                 bool root_owner);
      void unpack_version_states(Deserializer &derez, Runtime *runtime,
                       std::set<RtEvent> &preconditions, bool need_lock);
#ifdef DISABLE_CVOPT
      void add_uncaptured_state(VersionState *state, const FieldMask &mask);
#endif
      static void handle_deferred_node_state(const void *args);
    public:
      void record_dirty_fields(const FieldMask &dirty_mask);
      void record_valid_view(LogicalView *view, const FieldMask &mask);
      void record_reduction_fields(const FieldMask &reduction_fields);
      void record_reduction_view(ReductionView *view, const FieldMask &mask);
      void record_child_version_state(const LegionColor child_color, 
                                VersionState *state, const FieldMask &mask);
      void record_version_state(VersionState *state, const FieldMask &mask, 
                                ReferenceMutator *mutator);
      void release_gc_references(ReferenceMutator *mutator);
      void release_valid_references(ReferenceMutator *mutator);
    public:
      void capture_field_versions(FieldVersions &versions,
                                  const FieldMask &capture_mask) const;
    public:
      RegionTreeNode *const logical_node;
      CompositeBase *const parent;
      const DistributedID owner_did;
      const bool root_owner;
    protected:
      mutable LocalLock node_lock;
      // No need to hold references in general, but we do have to hold
      // them if we are the root child of a composite view subtree
      LegionMap<VersionState*,FieldMask>::aligned version_states;
#ifdef DISABLE_CVOPT
      // Only used on the owner node to track the set of version 
      // states on which we hold valid references
      std::vector<VersionState*> *valid_version_states;
#endif
    protected:
#ifndef DISABLE_CVOPT
      // Keep track of the fields we have captured
      FieldMask captured_fields;
      LegionMap<RtUserEvent,FieldMask>::aligned pending_captures;
#else
      // Keep track of the fields for which we still need captures
      FieldMask uncaptured_fields;
      LegionMap<RtUserEvent,FieldMask>::aligned pending_captures;
      LegionMap<VersionState*,FieldMask>::aligned uncaptured_states;
#endif
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
        const void *const value;
        const size_t value_size;
      };
    public:
      FillView(RegionTreeForest *ctx, DistributedID did,
               AddressSpaceID owner_proc, RegionTreeNode *node, 
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
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; }
      virtual LogicalView* get_subview(const LegionColor c);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target); 
      virtual InnerContext* get_shard_context(void) const
        { return NULL; }
    public:
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         FieldMask copy_mask,
                                         const RestrictInfo &restrict_info,
                                         bool restrict_out);
      virtual void issue_deferred_copies(DeferredCopier &copier,
                                         const FieldMask &local_copy_mask,
                                         const WriteMasks &write_masks,
                                         WriteSet &performed_writes,
                                         PredEvent pred_guard);
      virtual bool issue_deferred_copies_single(DeferredSingleCopier &copier,
                                         IndexSpaceExpression *write_mask,
                                         IndexSpaceExpression *&write_performed,
                                         PredEvent pred_guard);
    protected:
      void issue_update_fills(DeferredCopier &copier,
                              const FieldMask &fill_mask,
                              IndexSpaceExpression *mask,
                              WriteSet &performed_writes,
                              PredEvent pred_guard) const;
      void issue_internal_fills(const TraversalInfo &info,
                                MaterializedView *dst,
                                const FieldMask &fill_mask,
            const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                  LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                PredEvent pred_guard,
                                CopyAcrossHelper *helper = NULL,
                                IndexSpaceExpression *mask = NULL,
                                WriteSet *perf = NULL) const;
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
                    public VersionTracker,
                    public CompositeBase,
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
              AddressSpaceID owner_proc,
              DeferredVersionInfo *version_info,
              RegionTreeNode *node, PredEvent true_guard,
              PredEvent false_guard, InnerContext *owner,
              bool register_now);
      PhiView(const PhiView &rhs);
      virtual ~PhiView(void);
    public:
      PhiView& operator=(const PhiView &rhs);
    public:
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; }
      virtual LogicalView* get_subview(const LegionColor c);
    public:
      virtual void notify_owner_inactive(ReferenceMutator *mutator);
      virtual void notify_owner_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target);
      virtual InnerContext* get_shard_context(void) const
        { return owner_context; }
    public:
      virtual bool is_upper_bound_node(RegionTreeNode *node) const;
      virtual void get_field_versions(RegionTreeNode *node, bool split_prev, 
                                      const FieldMask &needed_fields,
                                      FieldVersions &field_versions);
      virtual void get_advance_versions(RegionTreeNode *node, bool base,
                                        const FieldMask &needed_fields,
                                        FieldVersions &field_versions);
      virtual void get_split_mask(RegionTreeNode *node, 
                                  const FieldMask &needed_fields,
                                  FieldMask &split);
      virtual void pack_writing_version_numbers(Serializer &rez) const;
      virtual void pack_upper_bound_node(Serializer &rez) const;
    public:
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         FieldMask copy_mask,
                                         const RestrictInfo &restrict_info,
                                         bool restrict_out);
      virtual void issue_deferred_copies(DeferredCopier &copier,
                                         const FieldMask &local_copy_mask,
                                         const WriteMasks &write_masks,
                                         WriteSet &performed_writes,
                                         PredEvent pred_guard);
      virtual bool issue_deferred_copies_single(DeferredSingleCopier &copier,
                                         IndexSpaceExpression *write_mask,
                                         IndexSpaceExpression *&write_performed,
                                         PredEvent pred_guard);
    public:
      virtual InnerContext* get_owner_context(void) const
        { assert(false); return NULL; }
      virtual DistributedID get_owner_did(void) const
        { assert(false); return 0; }
#ifndef DISABLE_CVOPT
      virtual void perform_ready_check(FieldMask mask)
#else
      virtual void perform_ready_check(FieldMask mask, RegionTreeNode *target)
#endif
        { assert(false); }
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views, 
                                    bool needs_lock = true)
        { assert(false); }
      virtual void unpack_composite_view_response(Deserializer &derez,
                                                  Runtime *runtime)
        { assert(false); }
    public:
      void record_true_view(LogicalView *view, const FieldMask &view_mask,
                            ReferenceMutator *mutator);
      void record_false_view(LogicalView *view, const FieldMask &view_mask,
                             ReferenceMutator *mutator);
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
      DeferredVersionInfo *const version_info;
      InnerContext *const owner_context;
    protected:
      LegionMap<LogicalView*,FieldMask>::aligned true_views;
      LegionMap<LogicalView*,FieldMask>::aligned false_views;
    };

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_materialized_did(
                                                    DistributedID did, bool top)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      if (top)
        return LEGION_DISTRIBUTED_HELP_ENCODE(did, 
                MATERIALIZED_VIEW_DC | (1ULL << 7));
      else
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
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 
                REDUCTION_VIEW_DC | (1ULL << 7));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_composite_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 
                  COMPOSITE_VIEW_DC | (1ULL << 7));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_fill_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, FILL_VIEW_DC | (1ULL << 7));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_phi_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(DIST_TYPE_LAST_DC < (1U << 7));
#endif
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, PHI_VIEW_DC | (1ULL << 7));
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
    /*static*/ inline bool LogicalView::is_composite_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xFULL) == 
                                              COMPOSITE_VIEW_DC);
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
    /*static*/ inline bool LogicalView::is_top_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (((LEGION_DISTRIBUTED_HELP_DECODE(did) & (1ULL << 7)) >> 7) == 1);
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
      return (is_composite_did(did) || is_fill_did(did) || is_phi_did(did));
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
    inline bool LogicalView::is_composite_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_composite_did(did);
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
    inline CompositeView* LogicalView::as_composite_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_composite_view());
#endif
      return static_cast<CompositeView*>(const_cast<LogicalView*>(this));
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
    inline bool MaterializedView::has_local_precondition(PhysicalUser *user,
                                               const RegionUsage &next_user,
                                               const LegionColor child_color,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               IndexSpaceExpression *user_expr,
                                               ApEvent &precondition)
    //--------------------------------------------------------------------------
    {
      // Different region requirements of the same operation 
      // We just need to wait on any copies generated for this region
      // requirement, we'll implicitly wait for all other copies to 
      // finish anyway as the region requirements that generated those
      // copies will catch dependences
      // We order these tests in a slightly entirely based on cost which
      // doesn't make the for the most read-able code
      if ((op_id == user->op_id) && (index != user->index))
        return false;
      const bool has_child = (child_color != INVALID_COLOR);
      // Same child, already done the analysis
      // Or we have disjoint children from our knowledge of the partition
      if (has_child && ((child_color == user->child) || 
            ((user->child != INVALID_COLOR) && disjoint_children)))
        return false;
      // Now do a dependence test for privilege non-interference
      DependenceType dt = check_dependence_type(user->usage, next_user);
      switch (dt)
      {
        case NO_DEPENDENCE:
        case ATOMIC_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          return false;
        case TRUE_DEPENDENCE:
        case ANTI_DEPENDENCE:
          break;
        default:
          assert(false); // should never get here
      }
      // See if we have disjoint children the hard-way
      if (has_child && (user->child != INVALID_COLOR) && !disjoint_children &&
            logical_node->are_children_disjoint(child_color, user->child))
        return false;
      // This is the most expensive test so we do it last
      // See if the two user expressions intersect, note that we have to
      // do this with a special intersection test because either of these
      // could be pending index space expression and we don't want to block
      return user_expr->test_intersection_nonblocking(user->expr, 
                                            context, precondition);
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_VIEWS_H__
