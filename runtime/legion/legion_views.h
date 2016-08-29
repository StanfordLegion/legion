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

#ifndef __LEGION_VIEWS_H__
#define __LEGION_VIEWS_H__

#include "legion_types.h"
#include "legion_utilities.h"
#include "legion_allocation.h"
#include "garbage_collection.h"

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
                  AddressSpaceID owner_proc, AddressSpaceID local_space,
                  RegionTreeNode *node, bool register_now);
      virtual ~LogicalView(void);
    public:
      static void delete_logical_view(LogicalView *view);
    public:
      inline bool is_instance_view(void) const;
      inline bool is_deferred_view(void) const;
      inline bool is_materialized_view(void) const;
      inline bool is_reduction_view(void) const;
      inline bool is_composite_view(void) const;
      inline bool is_fill_view(void) const;
    public:
      inline InstanceView* as_instance_view(void) const;
      inline DeferredView* as_deferred_view(void) const;
      inline MaterializedView* as_materialized_view(void) const;
      inline ReductionView* as_reduction_view(void) const;
      inline CompositeView* as_composite_view(void) const;
      inline FillView* as_fill_view(void) const;
    public:
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const ColorPoint &c) = 0;
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
      static void handle_deferred_collect(LogicalView *view,
                                          const std::set<ApEvent> &term_events);
    public:
      static inline DistributedID encode_materialized_did(DistributedID did,
                                                           bool top);
      static inline DistributedID encode_reduction_did(DistributedID did);
      static inline DistributedID encode_composite_did(DistributedID did);
      static inline DistributedID encode_fill_did(DistributedID did);
      static inline bool is_materialized_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_composite_did(DistributedID did);
      static inline bool is_fill_did(DistributedID did);
      static inline bool is_top_did(DistributedID did);
    public:
      RegionTreeForest *const context;
      RegionTreeNode *const logical_node;
    protected:
      Reservation view_lock;
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
                   AddressSpaceID owner_proc, AddressSpaceID local_space,
                   AddressSpaceID logical_owner, RegionTreeNode *node, 
                   UniqueID owner_context, bool register_now); 
      virtual ~InstanceView(void);
    public:
      inline bool is_logical_owner(void) const
        { return (local_space == logical_owner); }
    public:
      virtual bool has_manager(void) const = 0;
      virtual PhysicalManager* get_manager(void) const = 0;
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const ColorPoint &c) = 0;
      virtual Memory get_location(void) const = 0;
      virtual bool has_space(const FieldMask &space_mask) const = 0;
    public:
      // Entry point functions for doing physical dependence analysis
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const VersionInfo &version_info,
                                           const UniqueID creator_op_id,
                                           const unsigned index,
                                           const AddressSpaceID source,
                     LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                         std::set<RtEvent> &applied_events) = 0;
      virtual void add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                 const VersionInfo &version_info,
                                 const UniqueID creator_op_id,
                                 const unsigned index,
                                 const FieldMask &mask, bool reading,
                                 const AddressSpaceID source,
                                 std::set<RtEvent> &applied_events) = 0;
      virtual ApEvent find_user_precondition(const RegionUsage &user,
                                           ApEvent term_event,
                                           const FieldMask &user_mask,
                                           Operation *op, const unsigned index,
                                           const VersionInfo &version_info,
                                           std::set<RtEvent> &applied_events) = 0;
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, Operation *op,
                            const unsigned index, AddressSpaceID source,
                            const VersionInfo &version_info,
                            std::set<RtEvent> &applied_events) = 0;
      // This is a fused version of the above two methods
      virtual ApEvent add_user_fused(const RegionUsage &user,
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
                                   const VersionInfo &version_info,
                                   const AddressSpaceID source,
                                   std::set<RtEvent> &applied_events,
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
                   std::vector<Domain::CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL) = 0;
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                     std::vector<Domain::CopySrcDstField> &src_fields,
                             CopyAcrossHelper *across_helper = NULL) = 0;
      virtual void reduce_from(ReductionOpID redop,const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields) = 0;
    public:
      inline InstanceView* get_instance_subview(const ColorPoint &c) 
        { return get_subview(c)->as_instance_view(); }
    public:
      virtual void process_update_request(AddressSpaceID source,
                               RtUserEvent done_event, Deserializer &derez) = 0;
      virtual void process_update_response(Deserializer &derez,
                                           RtUserEvent done_event) = 0;
      virtual void process_remote_update(Deserializer &derez,
                                         AddressSpaceID source) = 0;
      virtual void process_remote_invalidate(const FieldMask &invalid_mask,
                                             RtUserEvent done_event) = 0;
    public:
      static void handle_view_update_request(Deserializer &derez, 
          Runtime *runtime, AddressSpaceID source); 
      static void handle_view_update_response(Deserializer &derez, Runtime *rt);
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
    class MaterializedView : public InstanceView {
    public:
      static const AllocationType alloc_type = MATERIALIZED_VIEW_ALLOC;
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
                       AddressSpaceID owner_proc, AddressSpaceID local_proc,
                       AddressSpaceID logical_owner, RegionTreeNode *node, 
                       InstanceManager *manager, MaterializedView *parent, 
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
      MaterializedView* get_materialized_subview(const ColorPoint &c);
      static void handle_subview_did_request(Deserializer &derez,
                             Runtime *runtime, AddressSpaceID source);
      static void handle_subview_did_response(Deserializer &derez); 
      MaterializedView* get_materialized_parent_view(void) const;
    public:
      void copy_field(FieldID fid, std::vector<Domain::CopySrcDstField> &infos);
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<Domain::CopySrcDstField> &dst_fields,
                             CopyAcrossHelper *across_helper = NULL);
      virtual void reduce_from(ReductionOpID redop,const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields);
    public:
      void accumulate_events(std::set<ApEvent> &all_events);
    public:
      virtual bool has_manager(void) const { return true; }
      virtual PhysicalManager* get_manager(void) const { return manager; }
      virtual bool has_parent(void) const { return (parent != NULL); }
      virtual LogicalView* get_parent(void) const { return parent; }
      virtual LogicalView* get_subview(const ColorPoint &c);
      virtual Memory get_location(void) const;
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const VersionInfo &version_info,
                                           const UniqueID creator_op_id,
                                           const unsigned index,
                                           const AddressSpaceID source,
                         LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                           std::set<RtEvent> &applied_events);
    protected: 
      void find_copy_preconditions_above(ReductionOpID redop, bool reading,
                                         const FieldMask &copy_mask,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                         const AddressSpaceID source,
                       LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                         std::set<RtEvent> &applied_events);
      void find_local_copy_preconditions(ReductionOpID redop, bool reading,
                                         const FieldMask &copy_mask,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                         const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                         std::set<RtEvent> &applied_events);
    public:
      virtual void add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                 const VersionInfo &version_info,
                                 const UniqueID creator_op_id,
                                 const unsigned index,
                                 const FieldMask &mask, bool reading,
                                 const AddressSpaceID source,
                                 std::set<RtEvent> &applied_events);
    protected:
      void add_copy_user_above(const RegionUsage &usage, ApEvent copy_term,
                               const ColorPoint &child_color,
                               const VersionInfo &version_info,
                               const UniqueID creator_op_id,
                               const unsigned index,
                               const FieldMask &copy_mask,
                               const AddressSpaceID source,
                               std::set<RtEvent> &applied_events);
      void add_local_copy_user(const RegionUsage &usage, 
                               ApEvent copy_term, bool base_user,
                               const ColorPoint &child_color,
                               const VersionInfo &version_info,
                               const UniqueID creator_op_id,
                               const unsigned index,
                               const FieldMask &copy_mask,
                               const AddressSpaceID source,
                               std::set<RtEvent> &applied_events);
    public:
      virtual ApEvent find_user_precondition(const RegionUsage &user,
                                           ApEvent term_event,
                                           const FieldMask &user_mask,
                                           Operation *op, const unsigned index,
                                           const VersionInfo &version_info,
                                           std::set<RtEvent> &applied_events);
    protected:
      void find_user_preconditions_above(const RegionUsage &usage,
                                         ApEvent term_event,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                                         const UniqueID op_id,
                                         const unsigned index,
                                         const FieldMask &user_mask,
                                         std::set<ApEvent> &preconditions,
                                         std::set<RtEvent> &applied_events);
      void find_local_user_preconditions(const RegionUsage &usage,
                                         ApEvent term_event,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                                         const UniqueID op_id,
                                         const unsigned index,
                                         const FieldMask &user_mask,
                                         std::set<ApEvent> &preconditions,
                                         std::set<RtEvent> &applied_events);
    public:
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, Operation *op,
                            const unsigned index, AddressSpaceID source,
                            const VersionInfo &version_info,
                            std::set<RtEvent> &applied_events);
    protected:
      void add_user_above(const RegionUsage &usage, ApEvent term_event,
                          const ColorPoint &child_color, 
                          const VersionInfo &version_info,
                          const UniqueID op_id, const unsigned index,
                          const FieldMask &user_mask,
                          const bool need_version_update,
                          const AddressSpaceID source,
                          std::set<RtEvent> &applied_events);
      bool add_local_user(const RegionUsage &usage, ApEvent term_event,
                          const ColorPoint &child_color, 
                          const VersionInfo &version_info,
                          const UniqueID op_id, const unsigned index,
                          const FieldMask &user_mask,
                          const AddressSpaceID source,
                          std::set<RtEvent> &applied_events);
    public:
      // This is a fused version of the above two virtual methods
      virtual ApEvent add_user_fused(const RegionUsage &user, 
                                   ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
                                   const VersionInfo &version_info,
                                   const AddressSpaceID source,
                                   std::set<RtEvent> &applied_events,
                                   bool update_versions = true);
    protected:
      void add_user_above_fused(const RegionUsage &usage, ApEvent term_event,
                                const ColorPoint &child_color,
                                const VersionInfo &version_info,
                                const UniqueID op_id,
                                const unsigned index,
                                const FieldMask &user_mask,
                                const AddressSpaceID source,
                                std::set<ApEvent> &preconditions,
                                std::set<RtEvent> &applied_events,
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
    public:
      virtual void send_view(AddressSpaceID target); 
      void update_gc_events(const std::deque<ApEvent> &gc_events);
    protected:
      // Update the version numbers
      // These first two methods do two-phase updates for copies
      // These methods must be called while holding the lock
      // in non-exclusive and exclusive mode respectively
      void find_version_updates(const FieldMask &user_mask,
                                const VersionInfo &version_info,
                                FieldMask &write_skip_mask,
                                FieldMask &filter_mask,
                            LegionMap<VersionID,FieldMask>::aligned &advance,
                            LegionMap<VersionID,FieldMask>::aligned &add_only,
                                bool is_reducing);
      void apply_version_updates(FieldMask &filter_mask,
                      const LegionMap<VersionID,FieldMask>::aligned &advance,
                      const LegionMap<VersionID,FieldMask>::aligned &add_only,
                      AddressSpaceID source, std::set<RtEvent> &applied_events);
      // This method does one phase update and advance for users
      // This one will take it's own lock
      bool update_version_numbers(const FieldMask &user_mask,
                                  const VersionInfo &version_info,
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
      void filter_current_user(ApEvent user_event, 
                               const FieldMask &filter_mask);
      void filter_previous_user(ApEvent user_event, 
                                const FieldMask &filter_mask);
    protected:
      void find_current_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const ColorPoint &child_color,
                                      ApEvent term_event,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      std::set<ApEvent> &preconditions,
                                      std::set<ApEvent> &dead_events,
                  LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                      FieldMask &observed, 
                                      FieldMask &non_dominated);
      void find_previous_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const ColorPoint &child_color,
                                      ApEvent term_event,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      std::set<ApEvent> &preconditions,
                                      std::set<ApEvent> &dead_events);
      // Overloaded versions for being precise about copy preconditions
      void find_current_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const ColorPoint &child_color,
                                      const UniqueID op_id,
                                      const unsigned index,
                  LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                      std::set<ApEvent> &dead_events,
                  LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                      FieldMask &observed, 
                                      FieldMask &non_dominated);
      void find_previous_preconditions(const FieldMask &user_mask,
                                      const RegionUsage &usage,
                                      const ColorPoint &child_color,
                                      const UniqueID op_id,
                                      const unsigned index,
                  LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                      std::set<ApEvent> &dead_events);
      void find_previous_filter_users(const FieldMask &dominated_mask,
                  LegionMap<ApEvent,FieldMask>::aligned &filter_events);
      inline bool has_local_precondition(PhysicalUser *prev_user,
                                     const RegionUsage &next_user,
                                     const ColorPoint &child_color,
                                     const UniqueID op_id,
                                     const unsigned index);
    public:
      //void update_versions(const FieldMask &update_mask);
      void find_atomic_reservations(const FieldMask &mask, 
                                    Operation *op, bool exclusive);
    public:
      void set_descriptor(FieldDataDescriptor &desc, FieldID field_id) const;
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
    public:
      void perform_remote_valid_check(const FieldMask &check_mask,
                                      const VersionInfo &version_info,
                                      bool reading,
                                      std::set<RtEvent> *wait_on = NULL);
      void perform_read_invalidations(const FieldMask &check_mask,
                                      const VersionInfo &version_info,
                                      const AddressSpaceID source,
                                      std::set<RtEvent> &applied_events);
      void send_invalidations(const FieldMask &invalidate_mask,
                              const AddressSpaceID can_skip,
                              std::set<RtEvent> &applied_events);
    public:
      virtual void process_update_request(AddressSpaceID source,
                               RtUserEvent done_event, Deserializer &derez);
      virtual void process_update_response(Deserializer &derez,
                                           RtUserEvent done_event);
      virtual void process_remote_update(Deserializer &derez,
                                         AddressSpaceID source);
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
      std::map<ColorPoint,MaterializedView*> children;
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
      // The logical owner node maintains a data structure to track which
      // remote copies of this view have valid field data, whenever the
      // set of users gets filtered for a field from current to previous
      // then the logical owner send invalidate messages
      LegionMap<AddressSpaceID,FieldMask>::aligned valid_remote_instances;
      // On remote nodes, this field mask tracks whether we have a
      // valid lease from the logical owner node. On the owner node it
      // tracks a summary of all the fields that have remote leases.
      FieldMask remote_valid_mask;
      // These masks track whether we have sent the remote read requests
      // for fields for the current and previous versions
      FieldMask current_remote_read_requests, previous_remote_read_requests;
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
    class ReductionView : public InstanceView {
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
                    AddressSpaceID owner_proc, AddressSpaceID local_proc,
                    AddressSpaceID logical_owner, RegionTreeNode *node, 
                    ReductionManager *manager, UniqueID owner_context,
                    bool register_now);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      void perform_reduction(InstanceView *target, const FieldMask &copy_mask, 
                             const VersionInfo &version_info, 
                             Operation *op, unsigned index,
                             std::set<RtEvent> &map_applied_events);
      ApEvent perform_deferred_reduction(MaterializedView *target,
                                        const FieldMask &copy_mask,
                                        const VersionInfo &version_info,
                                        const std::set<ApEvent> &preconditions,
                                        Operation *op, unsigned index,
                                        CopyAcrossHelper *helper,
                                        RegionTreeNode *intersect,
                                        std::set<RtEvent> &map_applied_events);
      ApEvent perform_deferred_across_reduction(MaterializedView *target,
                                              FieldID dst_field,
                                              FieldID src_field,
                                              unsigned src_index,
                                       const VersionInfo &version_info,
                                       const std::set<ApEvent> &preconditions,
                                       Operation *op, unsigned index,
                                       RegionTreeNode *intersect,
                                       std::set<RtEvent> &map_applied_events);
    public:
      virtual bool has_manager(void) const { return true; } 
      virtual PhysicalManager* get_manager(void) const;
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; } 
      virtual LogicalView* get_subview(const ColorPoint &c);
      virtual Memory get_location(void) const;
      virtual bool has_space(const FieldMask &space_mask) const
        { return false; }
    public:
      virtual void find_copy_preconditions(ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const VersionInfo &version_info,
                                           const UniqueID creator_op_id,
                                           const unsigned index,
                                           const AddressSpaceID source,
                         LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                           std::set<RtEvent> &applied_events);
      virtual void add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                 const VersionInfo &version_info,
                                 const UniqueID creator_op_id,
                                 const unsigned index,
                                 const FieldMask &mask, bool reading,
                                 const AddressSpaceID source,
                                 std::set<RtEvent> &applied_events);
      virtual ApEvent find_user_precondition(const RegionUsage &user,
                                           ApEvent term_event,
                                           const FieldMask &user_mask,
                                           Operation *op, const unsigned index,
                                           const VersionInfo &version_info,
                                           std::set<RtEvent> &applied_events);
      virtual void add_user(const RegionUsage &user, ApEvent term_event,
                            const FieldMask &user_mask, Operation *op,
                            const unsigned index, AddressSpaceID source,
                            const VersionInfo &version_info,
                            std::set<RtEvent> &applied_events);
      // This is a fused version of the above two methods
      virtual ApEvent add_user_fused(const RegionUsage &user,ApEvent term_event,
                                   const FieldMask &user_mask, 
                                   Operation *op, const unsigned index,
                                   const VersionInfo &version_info,
                                   const AddressSpaceID source,
                                   std::set<RtEvent> &applied_events,
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
                     std::vector<Domain::CopySrcDstField> &dst_fields,
                             CopyAcrossHelper *across_helper = NULL);
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields);
    public:
      void reduce_from(ReductionOpID redop, const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
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
    public:
      void perform_remote_valid_check(void);
      virtual void process_update_request(AddressSpaceID source,
                               RtUserEvent done_event, Deserializer &derez);
      virtual void process_update_response(Deserializer &derez,
                                           RtUserEvent done_event);
      virtual void process_remote_update(Deserializer &derez,
                                         AddressSpaceID source);
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
                   AddressSpaceID owner_space, AddressSpaceID local_space,
                   RegionTreeNode *node, bool register_now);
      virtual ~DeferredView(void);
    public:
      // Deferred views never have managers
      virtual bool has_manager(void) const { return false; }
      virtual PhysicalManager* get_manager(void) const
      { return NULL; }
      virtual bool has_parent(void) const = 0;
      virtual LogicalView* get_parent(void) const = 0;
      virtual LogicalView* get_subview(const ColorPoint &c) = 0;
      virtual bool has_space(const FieldMask &space_mask) const
        { return false; }
    public:
      virtual void notify_active(ReferenceMutator *mutator) = 0;
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
    public:
      // Should never be called
      virtual void collect_users(const std::set<ApEvent> &term_events)
        { assert(false); }
    public:
      virtual DeferredView* simplify(CompositeCloser &closer, 
                                      const FieldMask &capture_mask) = 0;
    public:
      void issue_deferred_copies(const TraversalInfo &info,
                                 MaterializedView *dst,
                                 const FieldMask &copy_mask);
      void issue_deferred_copies_across(const TraversalInfo &info,
                                        MaterializedView *dst,
                                  const std::vector<unsigned> &src_indexes,
                                  const std::vector<unsigned> &dst_indexes,
                                        ApEvent precondition,
                                        std::set<ApEvent> &postconditions);
      void find_field_descriptors(ApEvent term_event,
                                  const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  FieldID field_id, Operation *op,
                                  const unsigned index,
                          std::vector<FieldDataDescriptor> &field_data,
                                  std::set<ApEvent> &preconditions);
    public:
      virtual bool need_temporary_instance(MaterializedView *dst,
                                           const FieldMask &copy_mask,
                                           bool &already_valid) = 0;
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                         CopyAcrossHelper *helper = NULL) = 0; 
    };

    /**
     * \class CompositeVersionInfo
     * This is a wrapper class for keeping track of the version
     * information for all the composite nodes in a composite instance.
     */
    class CompositeVersionInfo : public Collectable {
    public:
      CompositeVersionInfo(void);
      CompositeVersionInfo(const CompositeVersionInfo &rhs);
      ~CompositeVersionInfo(void);
    public:
      CompositeVersionInfo& operator=(const CompositeVersionInfo &rhs);
    public:
      inline VersionInfo& get_version_info(void)
        { return version_info; }
    protected:
      VersionInfo version_info;
    };

    /**
     * \class CompositeView
     * The CompositeView class is used for deferring close
     * operations by representing a valid version of a single
     * logical region with a bunch of different instances.
     */
    class CompositeView : public DeferredView {
    public:
      static const AllocationType alloc_type = COMPOSITE_VIEW_ALLOC; 
    public:
      struct DeferCompositeNodeRefArgs {
      public:
        HLRTaskID hlr_id;
        LogicalView *view;
        unsigned refs;
      };
      struct DeferCompositeViewCreationArgs {
      public:
        HLRTaskID hlr_id;
        DistributedID did;
        AddressSpaceID owner;
        RegionTreeNode *target_node;
        CompositeNode *root;
        CompositeVersionInfo *version_info;
      };
    public:
      CompositeView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, RegionTreeNode *node, 
                    AddressSpaceID local_proc, CompositeNode *root,
                    CompositeVersionInfo *version_info, 
                    bool register_now);
      CompositeView(const CompositeView &rhs);
      virtual ~CompositeView(void);
    public:
      CompositeView& operator=(const CompositeView &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; }
      virtual LogicalView* get_subview(const ColorPoint &c);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target); 
    public:
      virtual DeferredView* simplify(CompositeCloser &closer, 
                                     const FieldMask &capture_mask);
    public:
      virtual bool need_temporary_instance(MaterializedView *dst,
                                           const FieldMask &copy_mask,
                                           bool &already_valid);
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                         CopyAcrossHelper *helper = NULL);
    public:
      static void handle_send_composite_view(Runtime *runtime, 
                              Deserializer &derez, AddressSpaceID source);
      static void handle_deferred_node_refs(const void *args);
      static void handle_deferred_view_creation(Runtime *runtime,
                                                const void *args);
    public:
      // The root node for this composite view
      CompositeNode *const root;
      CompositeVersionInfo *const version_info;
    };

    /**
     * \class CompositeNode
     * A helper class for representing the frozen state of a region
     * tree as part of one or more composite views.
     */
    class CompositeNode {
    public:
      static const AllocationType alloc_type = COMPOSITE_NODE_ALLOC;
    public:
      CompositeNode(RegionTreeNode *node, CompositeNode *parent);
      CompositeNode(const CompositeNode &rhs);
      ~CompositeNode(void);
    public:
      CompositeNode& operator=(const CompositeNode &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void add_child(CompositeNode *child);
      void update_child(CompositeNode *child, const FieldMask &mask);
      void finalize(FieldMask &final_mask);
      void set_owner_did(DistributedID own_did);
    public:
      void capture_physical_state(CompositeCloser &closer, 
                                  PhysicalState *state, 
                                  const FieldMask &close_mask,
                                  const FieldMask &capture_dirty,
                                  const FieldMask &capture_reduc);
      bool capture_instances(CompositeCloser &closer, 
                             const FieldMask &capture_mask,
                     const LegionMap<LogicalView*,FieldMask>::aligned *views);
      void capture_reductions(const FieldMask &capture_mask,
                     const LegionMap<ReductionView*,FieldMask>::aligned *views);
      bool simplify(CompositeCloser &closer, FieldMask &capture_mask,
                    CompositeNode *new_parent);
    public:
      bool need_temporary_instance(MaterializedView *dst,
                                   const FieldMask &copy_mask,
                                   bool &fully_valid,
                                   bool &partially_valid,
                                   bool check_root = true) const;
      void issue_deferred_copies(const TraversalInfo &info, 
                                 MaterializedView *dst,
                                 const FieldMask &copy_mask,
                                 const VersionInfo &src_version_info,
              const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                    LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                    LegionMap<ApEvent,FieldMask>::aligned &postreductions,
                    CopyAcrossHelper *helper, bool check_root = true) const;
      CompositeNode* find_next_root(RegionTreeNode *target,
                                    const FieldMask &mask) const;
      bool are_domination_tests_sound(const FieldMask &mask) const;
      void find_valid_views(const FieldMask &search_mask,
                      LegionMap<LogicalView*,FieldMask>::aligned &valid) const;
      void issue_update_copies(const TraversalInfo &info, MaterializedView *dst,
                       FieldMask copy_mask, const VersionInfo &src_version_info,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                    const LegionMap<LogicalView*,FieldMask>::aligned &views,
                          CopyAcrossHelper *helper) const;
      void issue_update_reductions(const TraversalInfo &info, 
                                   MaterializedView *dst, 
                                   const FieldMask &copy_mask,
                                   const VersionInfo &src_version_info,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                          CopyAcrossHelper *helper) const;
    public:
      void pack_composite_tree(Serializer &rez, AddressSpaceID target);
      void unpack_composite_tree(Deserializer &derez, AddressSpaceID source,
                               Runtime *runtime, std::map<LogicalView*,
                                std::pair<RtEvent,unsigned> > &pending_refs);
    public:
      void notify_active(ReferenceMutator *mutator);
      void notify_inactive(ReferenceMutator *mutator);
      void notify_valid(ReferenceMutator *mutator);
      void notify_invalid(ReferenceMutator *mutator);
    public:
      RegionTreeNode *const logical_node;
      CompositeNode *const parent;
    protected:
      DistributedID owner_did;
      FieldMask dirty_mask, reduction_mask;
      LegionMap<CompositeNode*,FieldMask/*valid fields*/>::aligned children;
      LegionMap<LogicalView*,FieldMask>::aligned valid_views;
      LegionMap<ReductionView*,FieldMask>::aligned reduction_views;
    };

    /**
     * \class FillView
     * This is a deferred view that is used for filling in 
     * fields with a default value.
     */
    class FillView : public DeferredView {
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
               AddressSpaceID owner_proc, AddressSpaceID local_proc,
               RegionTreeNode *node, FillViewValue *value,
               bool register_now);
      FillView(const FillView &rhs);
      virtual ~FillView(void);
    public:
      FillView& operator=(const FillView &rhs);
    public:
      virtual bool has_parent(void) const { return false; }
      virtual LogicalView* get_parent(void) const 
        { assert(false); return NULL; }
      virtual LogicalView* get_subview(const ColorPoint &c);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_view(AddressSpaceID target); 
    public:
      virtual DeferredView* simplify(CompositeCloser &closer, 
                                     const FieldMask &capture_mask);
    public:
      virtual bool need_temporary_instance(MaterializedView *dst,
                                           const FieldMask &copy_mask,
                                           bool &already_valid);
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                         CopyAcrossHelper *helper = NULL);
    public:
      static void handle_send_fill_view(Runtime *runtime, Deserializer &derez,
                                        AddressSpaceID source);
    public:
      FillViewValue *const value;
    };

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_materialized_did(
                                                    DistributedID did, bool top)
    //--------------------------------------------------------------------------
    {
      if (top)
        return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x0UL | (1UL << 2));
      else
        return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x0UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_reduction_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x1UL | (1UL << 2));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_composite_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x2UL | (1UL << 2));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_fill_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x3UL | (1UL << 2));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_materialized_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x3UL) == 0x0UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_reduction_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x3UL) == 0x1UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_composite_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x3UL) == 0x2UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_fill_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x3UL) == 0x3UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_top_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x8UL) == 0x8UL);
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
      return (is_composite_did(did) || is_fill_did(did));
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
    inline bool MaterializedView::has_local_precondition(PhysicalUser *user,
                                                 const RegionUsage &next_user,
                                                 const ColorPoint &child_color,
                                                 const UniqueID op_id,
                                                 const unsigned index)
    //--------------------------------------------------------------------------
    {
      // Different region requirements of the same operation 
      // We just need to wait on any copies generated for this region
      // requirement, we'll implicitly wait for all other copies to 
      // finish anyway as the region requirements that generated those
      // copies will catch dependences
      if ((op_id == user->op_id) && (index != user->index))
        return false;
      if (child_color.is_valid())
      {
        // Same child, already done the analysis
        if (child_color == user->child)
          return false;
        // Disjoint children means we can skip it
        if (user->child.is_valid() && (disjoint_children || 
              logical_node->are_children_disjoint(child_color, user->child)))
          return false;
      }
      // Now do a dependence test for coherence non-interference
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
      return true;
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_VIEWS_H__
