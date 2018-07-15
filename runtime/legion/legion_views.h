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
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

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
                                       Operation *op, const unsigned index,
                                       VersionTracker *version_tracker,
                                       std::set<RtEvent> &applied_events,
                                       const PhysicalTraceInfo &trace_info) = 0;
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, Operation *op,
                            const unsigned index, AddressSpaceID source,
                            VersionTracker *version_tracker,
                            std::set<RtEvent> &applied_events,
                            const PhysicalTraceInfo &trace_info) = 0;
      // This is a fused version of the above two methods
      virtual ApEvent add_user_fused(const RegionUsage &user,
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
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
        DistributedID did;
        AddressSpaceID owner_space;
        AddressSpaceID logical_owner;
        RegionTreeNode *target_node;
        PhysicalManager *manager;
        MaterializedView *parent;
        UniqueID context_uid;
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
                                           Operation *op, const unsigned index,
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
                            const FieldMask &user_mask, Operation *op,
                            const unsigned index, AddressSpaceID source,
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
                                     IndexSpaceExpression *user_expr);
    public:
      //void update_versions(const FieldMask &update_mask);
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
                                         const PhysicalTraceInfo &trace_info);
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
                                           Operation *op, const unsigned index,
                                           VersionTracker *version_tracker,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info);
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, Operation *op,
                            const unsigned index, AddressSpaceID source,
                            VersionTracker *version_tracker,
                            std::set<RtEvent> &applied_events,
                            const PhysicalTraceInfo &trace_info);
      // This is a fused version of the above two methods
      virtual ApEvent add_user_fused(const RegionUsage &user,ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
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
    protected:
      // the request event for reducers
      // only needed on remote views
      RtEvent remote_request_event; 
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
    public:
      DeferredCopier(const TraversalInfo *info, 
                     MaterializedView *dst, 
                     const FieldMask &copy_mask, 
                     const RestrictInfo &restrict_info,
                     bool restrict_out);
      // For handling deferred copies across
      DeferredCopier(const TraversalInfo *info, 
                     MaterializedView *dst, 
                     const FieldMask &copy_mask,
                     ApEvent precondition,
                     CopyAcrossHelper *helper = NULL);
      DeferredCopier(const DeferredCopier &rhs);
      ~DeferredCopier(void);
    public:
      DeferredCopier& operator=(const DeferredCopier &rhs);
    public:
      void merge_destination_preconditions(const FieldMask &copy_mask,
                LegionMap<ApEvent,FieldMask>::aligned &preconditions);
      void buffer_reductions(VersionTracker *tracker, PredEvent pred_guard, 
                             RegionTreeNode *intersect,
                             const WriteMasks &write_masks,
               LegionMap<ReductionView*,FieldMask>::aligned &source_reductions);
      void begin_guard_protection(void);
      void end_guard_protection(void);
      void begin_reduction_epoch(void);
      void end_reduction_epoch(void);
      void finalize(std::set<ApEvent> *postconditions = NULL);
      inline bool has_reductions(void) const 
        { return !reduction_epochs.empty(); }
    protected:
      void uniquify_copy_postconditions(void);
      void compute_dst_preconditions(const FieldMask &mask);
      bool issue_reductions(const int epoch, ApEvent reduction_pre, 
                            const FieldMask &mask,
              LegionMap<ApEvent,FieldMask>::aligned &reduction_postconditions);
    public: // const fields
      const TraversalInfo *const info;
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
      LegionVector<FieldMask>::aligned reduction_epoch_masks;
    protected:
      // Handle protection of events for guarded operations
      std::vector<LegionMap<ApEvent,FieldMask>::aligned> protected_copy_posts;
      bool finalized;
    };

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
    public:
      DeferredSingleCopier(const TraversalInfo *info, 
                           MaterializedView *dst, 
                           const FieldMask &copy_mask,
                           const RestrictInfo &restrict_info,
                           bool restrict_out);
      // For handling deferred copies across
      DeferredSingleCopier(const TraversalInfo *info, 
                           MaterializedView *dst, 
                           const FieldMask &copy_mask,
                           ApEvent precondition,
                           CopyAcrossHelper *helper = NULL);
      DeferredSingleCopier(const DeferredSingleCopier &rhs);
      ~DeferredSingleCopier(void);
    public:
      DeferredSingleCopier& operator=(const DeferredSingleCopier &rhs);
    public:
      void merge_destination_preconditions(std::set<ApEvent> &preconditions);
      void buffer_reductions(VersionTracker *tracker, PredEvent pred_guard,
                             RegionTreeNode *intersect, IndexSpaceExpression *mask,
                             std::vector<ReductionView*> &source_reductions);
      void begin_guard_protection(void);
      void end_guard_protection(void);
      void begin_reduction_epoch(void);
      void end_reduction_epoch(void);
      void finalize(std::set<ApEvent> *postconditions = NULL);
      inline void record_postcondition(ApEvent post)
        { copy_postconditions.insert(post); }
      inline bool has_reductions(void) const 
        { return !reduction_epochs.empty(); }
    protected:
      void compute_dst_preconditions(void);
    public: // const fields
      const unsigned field_index;
      const FieldMask copy_mask;
      const TraversalInfo *const info;
      MaterializedView *const dst;
      CopyAcrossHelper *const across_helper;
      const RestrictInfo *const restrict_info;
      const bool restrict_out;
    public:
      std::set<ApEvent> copy_postconditions;
    protected: // internal members
      unsigned current_reduction_epoch;
      std::vector<PendingReductions> reduction_epochs;
    protected:
      std::set<ApEvent> dst_preconditions;
      std::vector<std::set<ApEvent> > protected_copy_posts;
      bool has_dst_preconditions;
      bool finalized;
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
     * \class CompositeBase
     * A small helper class that provides some base functionality
     * for both the CompositeView and CompositeNode classes
     */
    class CompositeBase {
    public:
      CompositeBase(LocalLock &base_lock);
      virtual ~CompositeBase(void);
    protected:
      void issue_composite_updates(DeferredCopier &copier,
                                   RegionTreeNode *logical_node,
                                   const FieldMask &copy_mask,
                                   VersionTracker *src_version_tracker,
                                   PredEvent pred_guard, 
                                   const WriteMasks &write_masks,
                                   WriteSet &performed_writes/*write-only*/);
      // Single field version of the method above
      bool issue_composite_updates_single(DeferredSingleCopier &copier,
                                   RegionTreeNode *logical_node,
                                   VersionTracker *src_version_tracker,
                                   PredEvent pred_guard,
                                   IndexSpaceExpression *write_mask,
                                   // Only valid if !done
                                   IndexSpaceExpression *&performed);
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
      virtual void perform_ready_check(FieldMask mask) = 0;
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views, 
                                    bool needs_lock = true) = 0;
    public:
      virtual void print_view_state(const FieldMask &capture_mask,
                                    TreeStateLogger* logger,
                                    int current_nesting,
                                    int max_nesting);
    public:
      CompositeNode* find_child_node(RegionTreeNode *child);
    private:
      LocalLock &base_lock;
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
        DistributedCollectable *dc;
        DistributedID did;
      };
      struct DeferCompositeViewRegistrationArgs : 
        public LgTaskArgs<DeferCompositeViewRegistrationArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_DEFER_COMPOSITE_VIEW_REGISTRATION_TASK_ID;
      public:
        CompositeView *view;
      };
    public:
      struct NodeVersionInfo {
      public:
        FieldVersions versions;
        FieldMask valid_fields;
      };
    public:
      CompositeView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, RegionTreeNode *node, 
                    DeferredVersionInfo *info, const FieldMask &complete_mask,
                    WriteSet &partial_writes, InnerContext *context,
                    bool register_now);
      CompositeView(const CompositeView &rhs);
      virtual ~CompositeView(void);
    public:
      CompositeView& operator=(const CompositeView &rhs);
    public:
      CompositeView* clone(const FieldMask &clone_mask,
        const LegionMap<CompositeView*,FieldMask>::aligned &replacements) const;
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
      virtual InnerContext* get_context(void) const
        { return owner_context; }
    public:
      void prune(const WriteMasks &partial_write_masks, FieldMask &valid_mask,
                 LegionMap<CompositeView*,FieldMask>::aligned &replacements,
                 unsigned prune_depth);
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
    protected:
      CompositeNode* capture_above(RegionTreeNode *node,
                                   const FieldMask &needed_fields);
    public:
      // From CompositeBase
      virtual InnerContext* get_owner_context(void) const;
      virtual void perform_ready_check(FieldMask mask);
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views, 
                                    bool needs_lock = true);
    public:
      static void handle_send_composite_view(Runtime *runtime, 
                              Deserializer &derez, AddressSpaceID source);
      static void handle_deferred_view_registration(const void *args);
    public:
      void record_dirty_fields(const FieldMask &dirty_mask);
      void record_valid_view(LogicalView *view, FieldMask mask);
      void record_reduction_fields(const FieldMask &reduction_fields);
      void record_reduction_view(ReductionView *view, const FieldMask &mask);
      void record_child_version_state(const LegionColor child_color, 
         VersionState *state, const FieldMask &mask, ReferenceMutator *mutator);
      void finalize_capture(bool need_prune);
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
      bool has_nested_views() { return nested_composite_views.size() > 0; }
    public:
      // The path version info for this composite instance
      DeferredVersionInfo *const version_info;
      // Fields for which this composite view represents a complete
      // write of the root logical region
      const FieldMask complete_fields;
      // If we only have partial writes then we record it in this
      // write set data structure, we also keep references to all
      // the index space expressions and remove them when we're deleted
      WriteSet partial_writes;
      // The translation context if any
      InnerContext *const owner_context;
    protected:
      // Note that we never record any version state names here, we just
      // record the views and children we immediately depend on and that
      // is how we break the inifinite meta-data cycle
      LegionMap<CompositeView*,FieldMask>::aligned nested_composite_views;
    protected:
      LegionMap<RegionTreeNode*,NodeVersionInfo>::aligned node_versions;
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
      struct DeferCompositeNodeRefArgs : 
        public LgTaskArgs<DeferCompositeNodeRefArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COMPOSITE_NODE_REF_TASK_ID;
      public:
        VersionState *state;
        DistributedID owner_did;
      };
      struct DeferCaptureArgs : public LgTaskArgs<DeferCaptureArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COMPOSITE_NODE_CAPTURE_TASK_ID;
      public:
        CompositeNode *proxy_this;
        RtUserEvent capture_event;
      };
    public:
      CompositeNode(RegionTreeNode *node, CompositeBase *parent,
                    DistributedID owner_did);
      CompositeNode(const CompositeNode &rhs);
      virtual ~CompositeNode(void);
    public:
      CompositeNode& operator=(const CompositeNode &rhs);
    public:
      // From CompositeBase
      virtual InnerContext* get_owner_context(void) const;
      virtual void perform_ready_check(FieldMask mask);
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views, 
                                    bool needs_lock = true);
    public:
      void capture(RtUserEvent capture_event, ReferenceMutator *mutator);
      static void handle_deferred_capture(const void *args);
    public:
      void clone(CompositeView *target, const FieldMask &clone_mask) const;
      void pack_composite_node(Serializer &rez) const;
      static CompositeNode* unpack_composite_node(Deserializer &derez,
                     CompositeView *parent, Runtime *runtime, 
                     DistributedID owner_did, std::set<RtEvent> &preconditions);
      static void handle_deferred_node_ref(const void *args);
    public:
      void notify_valid(ReferenceMutator *mutator, bool root);
      void notify_invalid(ReferenceMutator *mutator, bool root);
    public:
      void record_dirty_fields(const FieldMask &dirty_mask);
      void record_valid_view(LogicalView *view, const FieldMask &mask);
      void record_reduction_fields(const FieldMask &reduction_fields);
      void record_reduction_view(ReductionView *view, const FieldMask &mask);
      void record_child_version_state(const LegionColor child_color, 
         VersionState *state, const FieldMask &mask, ReferenceMutator *mutator);
      void record_version_state(VersionState *state, const FieldMask &mask, 
                                ReferenceMutator *mutator, bool root);
    public:
      void capture_field_versions(FieldVersions &versions,
                                  const FieldMask &capture_mask) const;
    public:
      RegionTreeNode *const logical_node;
      CompositeBase *const parent;
      const DistributedID owner_did;
    protected:
      mutable LocalLock node_lock;
      // No need to hold references in general, but we do have to hold
      // them if we are the root child of a composite view subtree
      LegionMap<VersionState*,FieldMask>::aligned version_states;
    protected:
      // Keep track of the fields that are valid because we've captured them
      FieldMask valid_fields;
      LegionMap<RtUserEvent,FieldMask>::aligned pending_captures;
    protected:
      // Track whether we are currently valid or not, we start off
      // currently valid so we can add as many views as we want before
      // we are first made valid, but then if we become no longer
      // valid (e.g. on a remote node) then we have to remove our
      // references and possibly add them again
      bool currently_valid;
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
        DistributedCollectable *dc;
        DistributedID did; 
      };
      struct DeferPhiViewRegistrationArgs : 
        public LgTaskArgs<DeferPhiViewRegistrationArgs> {
      public:
        static const LgTaskID TASK_ID = 
          LG_DEFER_PHI_VIEW_REGISTRATION_TASK_ID;
      public:
        PhiView *view;
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
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target);
      virtual InnerContext* get_context(void) const
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
      virtual void perform_ready_check(FieldMask mask)
        { assert(false); }
      virtual void find_valid_views(const FieldMask &update_mask,
                  LegionMap<LogicalView*,FieldMask>::aligned &valid_views, 
                                    bool needs_lock = true)
        { assert(false); }
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
                                               IndexSpaceExpression *user_expr)
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
      // See if the two user expressions intersect
      IndexSpaceExpression *overlap = 
        logical_node->context->intersect_index_spaces(user->expr, user_expr);
      // If they don't overlap then there is no dependence
      if (overlap->check_empty())
        return false;
      return true;
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_VIEWS_H__
