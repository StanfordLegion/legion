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
                  RegionTreeNode *node);
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
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
      static void handle_view_request(Deserializer &derez, Runtime *runtime,
                                      AddressSpaceID source);
    public:
      void defer_collect_user(Event term_event);
      virtual void collect_users(const std::set<Event> &term_events) = 0;
      static void handle_deferred_collect(LogicalView *view,
                                          const std::set<Event> &term_events);
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
                   RegionTreeNode *node, SingleTask *owner_context); 
      virtual ~InstanceView(void);
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
                     LegionMap<Event,FieldMask>::aligned &preconditions) = 0;
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const VersionInfo &version_info,
                                 const FieldMask &mask, bool reading) = 0;
      virtual Event add_user(const RegionUsage &user, Event term_event,
                             const FieldMask &user_mask, Operation *op,
                             const VersionInfo &version_info) = 0;
      virtual void add_initial_user(Event term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask) = 0;
    public:
      // Reference counting state change functions
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      // Instance recycling
      virtual void collect_users(const std::set<Event> &term_events) = 0;
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
      virtual bool has_war_dependence(const RegionUsage &usage, 
                                      const FieldMask &user_mask) = 0;
    public:
      inline InstanceView* get_instance_subview(const ColorPoint &c) 
        { return get_subview(c)->as_instance_view(); }
    public:
      SingleTask *const owner_context;
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
                       RegionTreeNode *node, InstanceManager *manager,
                       MaterializedView *parent, SingleTask *owner_context);
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
      virtual bool has_war_dependence(const RegionUsage &usage, 
                              const FieldMask &user_mask);
    public:
      void accumulate_events(std::set<Event> &all_events);
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
                         LegionMap<Event,FieldMask>::aligned &preconditions);
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const VersionInfo &version_info,
                                 const FieldMask &mask, bool reading);
      virtual Event add_user(const RegionUsage &user, Event term_event,
                             const FieldMask &user_mask, Operation *op,
                             const VersionInfo &version_info);
      virtual void add_initial_user(Event term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void collect_users(const std::set<Event> &term_users);
    public:
      virtual void send_view(AddressSpaceID target); 
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
      void process_update(Deserializer &derez, AddressSpaceID source);
      void update_gc_events(const std::deque<Event> &gc_events);
    protected:
      void add_user_above(const RegionUsage &usage, Event term_event,
                          const ColorPoint &child_color,
                          const VersionInfo &version_info,
                          const FieldMask &user_mask,
                          std::set<Event> &preconditions);
      bool add_local_user(const RegionUsage &usage, 
                          Event term_event, bool base_user,
                          const ColorPoint &child_color,
                          const VersionInfo &version_info,
                          const FieldMask &user_mask,
                          std::set<Event> &preconditions);
    protected:
      void add_copy_user_above(const RegionUsage &usage, Event copy_term,
                               const ColorPoint &child_color,
                               const VersionInfo &version_info,
                               const FieldMask &copy_mask);
      void add_local_copy_user(const RegionUsage &usage, 
                               Event copy_term, bool base_user,
                               const ColorPoint &child_color,
                               const VersionInfo &version_info,
                               const FieldMask &copy_mask);
      bool find_current_preconditions(Event test_event,
                                      const PhysicalUser *prev_user,
                                      const FieldMask &prev_mask,
                                      const RegionUsage &next_user,
                                      const FieldMask &next_mask,
                                      const ColorPoint &child_color,
                                      std::set<Event> &preconditions,
                                      FieldMask &observed,
                                      FieldMask &non_dominated);
      bool find_previous_preconditions(Event test_event,
                                       const PhysicalUser *prev_user,
                                       const FieldMask &prev_mask,
                                       const RegionUsage &next_user,
                                       const FieldMask &next_mask,
                                       const ColorPoint &child_color,
                                       std::set<Event> &preconditions);
    protected: 
      void find_copy_preconditions_above(ReductionOpID redop, bool reading,
                                         const FieldMask &copy_mask,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                       LegionMap<Event,FieldMask>::aligned &preconditions);
      void find_local_copy_preconditions(ReductionOpID redop, bool reading,
                                         const FieldMask &copy_mask,
                                         const ColorPoint &child_color,
                                         const VersionInfo &version_info,
                           LegionMap<Event,FieldMask>::aligned &preconditions);
      void find_current_copy_preconditions(Event test_event,
                                           const PhysicalUser *user, 
                                           const FieldMask &user_mask,
                                           ReductionOpID redop, bool reading,
                                           const FieldMask &copy_mask,
                                           const ColorPoint &child_color,
                                           const FieldVersions *versions,
                        LegionMap<Event,FieldMask>::aligned &preconditions,
                                           FieldMask &observed,
                                           FieldMask &non_dominated);
      void find_previous_copy_preconditions(Event test_event,
                                            const PhysicalUser *user,
                                            const FieldMask &user_Mask,
                                            ReductionOpID redop, bool reading,
                                            const FieldMask &copy_mask,
                                            const ColorPoint &child_dolor,
                                            const FieldVersions *versions,
                        LegionMap<Event,FieldMask>::aligned &preconditions);
    protected:
      void filter_previous_users(
                    const LegionMap<Event,FieldMask>::aligned &filter_previous);
      void filter_current_users(const FieldMask &dominated);
      void filter_local_users(Event term_event);
      void add_current_user(PhysicalUser *user, Event term_event,
                            const FieldMask &user_mask);
      void add_previous_user(PhysicalUser *user, Event term_event,
                             const FieldMask &user_mask);
    protected:
      bool has_war_dependence_above(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    const ColorPoint &child_color);
      bool has_local_war_dependence(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    const ColorPoint &child_color,
                                    const ColorPoint &local_color);
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
      static void handle_send_update(Runtime *runtime, Deserializer &derez,
                                     AddressSpaceID source);
    public:
      InstanceManager *const manager;
      MaterializedView *const parent;
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
      // and will provide fast indexing for removing items.
      LegionMap<Event,EventUsers>::aligned current_epoch_users;
      LegionMap<Event,EventUsers>::aligned previous_epoch_users;
      // Also keep a set of events for which we have outstanding
      // garbage collection meta-tasks so we don't launch more than one
      // We need this even though we have the data structures above because
      // an event might be filtered out for some fields, so we can't rely
      // on it to detect when we have outstanding gc meta-tasks
      std::set<Event> outstanding_gc_events;
      // TODO: Keep track of the current version numbers for each field
      // This will allow us to detect when physical instances are no
      // longer valid from a particular view when doing rollbacks for
      // resilience or mis-speculation.
      //LegionMap<VersionID,FieldMask,
      //          PHYSICAL_VERSION_ALLOC>::track_aligned current_versions;
    protected:
      // Useful for pruning the initial users at cleanup time
      std::set<Event> initial_user_events;
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
                    RegionTreeNode *node, ReductionManager *manager,
                    SingleTask *owner_context);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      void perform_reduction(InstanceView *target, const FieldMask &copy_mask, 
                             const VersionInfo &version_info, 
                             Operation *op, CopyTracker *tracker = NULL);
      Event perform_deferred_reduction(MaterializedView *target,
                                        const FieldMask &copy_mask,
                                        const VersionInfo &version_info,
                                        const std::set<Event> &preconditions,
                                        Operation *op, CopyAcrossHelper *helper,
                                        RegionTreeNode *intersect);
      Event perform_deferred_across_reduction(MaterializedView *target,
                                              FieldID dst_field,
                                              FieldID src_field,
                                              unsigned src_index,
                                       const VersionInfo &version_info,
                                       const std::set<Event> &preconditions,
                                       Operation *op,RegionTreeNode *intersect);
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
                         LegionMap<Event,FieldMask>::aligned &preconditions);
      virtual void add_copy_user(ReductionOpID redop, Event copy_term,
                                 const VersionInfo &version_info,
                                 const FieldMask &mask, bool reading);
      virtual Event add_user(const RegionUsage &user, Event term_event,
                             const FieldMask &user_mask, Operation *op,
                             const VersionInfo &version_info);
      virtual void add_initial_user(Event term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask);
    public:
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<Domain::CopySrcDstField> &dst_fields,
                             CopyAcrossHelper *across_helper = NULL);
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields,
                           CopyAcrossHelper *across_helper = NULL);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool has_war_dependence(const RegionUsage &usage, 
                                      const FieldMask &user_mask);
    public:
      void reduce_from(ReductionOpID redop, const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
      virtual void collect_users(const std::set<Event> &term_events);
    public:
      virtual void send_view(AddressSpaceID target); 
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
      void process_update(Deserializer &derez, AddressSpaceID source);
    protected:
      void add_physical_user(PhysicalUser *user, bool reading,
                             Event term_event, const FieldMask &user_mask);
      void filter_local_users(Event term_event);
    public:
      static void handle_send_reduction_view(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
      static void handle_send_update(Runtime *runtime,
                              Deserializer &derez, AddressSpaceID source);
    public:
      ReductionOpID get_redop(void) const;
    public:
      ReductionManager *const manager;
    protected:
      LegionMap<Event,EventUsers>::aligned reduction_users;
      LegionMap<Event,EventUsers>::aligned reading_users;
      std::set<Event> outstanding_gc_events;
    protected:
      std::set<Event> initial_user_events;
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
                   RegionTreeNode *node);
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
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      // Should never be called
      virtual void collect_users(const std::set<Event> &term_events)
        { assert(false); }
    public:
      virtual DeferredView* simplify(CompositeCloser &closer, 
                                      const FieldMask &capture_mask) = 0;
    public:
      void issue_deferred_copies(const TraversalInfo &info,
                                 MaterializedView *dst,
                                 const FieldMask &copy_mask,
                                 CopyTracker *tracker = NULL);
      void issue_deferred_copies_across(const TraversalInfo &info,
                                        MaterializedView *dst,
                                  const std::vector<unsigned> &src_indexes,
                                  const std::vector<unsigned> &dst_indexes,
                                        Event precondition,
                                        std::set<Event> &postconditions);
      void find_field_descriptors(Event term_event,
                                  const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  FieldID field_id, Operation *op,
                          std::vector<FieldDataDescriptor> &field_data,
                                  std::set<Event> &preconditions);
    public:
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                    const LegionMap<Event,FieldMask>::aligned &preconditions,
                          LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL,
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
      CompositeView(RegionTreeForest *ctx, DistributedID did,
                    AddressSpaceID owner_proc, RegionTreeNode *node, 
                    AddressSpaceID local_proc, CompositeNode *root,
                    CompositeVersionInfo *version_info, bool across);
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
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      virtual void send_view(AddressSpaceID target); 
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
      void make_local(std::set<Event> &preconditions);
    public:
      virtual DeferredView* simplify(CompositeCloser &closer, 
                                     const FieldMask &capture_mask);
    public:
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                    const LegionMap<Event,FieldMask>::aligned &preconditions,
                          LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL,
                                         CopyAcrossHelper *helper = NULL);
    public:
      static void handle_send_composite_view(Runtime *runtime, 
                              Deserializer &derez, AddressSpaceID source);
    public:
      // The root node for this composite view
      CompositeNode *const root;
      CompositeVersionInfo *const version_info;
      const bool across_contexts;
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
                                  const FieldMask &dirty_mask,
                                  const FieldMask &reduc_mask);
      bool capture_instances(CompositeCloser &closer, 
                             const FieldMask &capture_mask,
                     const LegionMap<LogicalView*,FieldMask>::aligned *views);
      void capture_reductions(const FieldMask &capture_mask,
                     const LegionMap<ReductionView*,FieldMask>::aligned *views);
      bool simplify(CompositeCloser &closer, FieldMask &capture_mask,
                    CompositeNode *new_parent);
    public:
      void issue_deferred_copies(const TraversalInfo &info, 
                                 MaterializedView *dst, bool across_contexts,
                                 const FieldMask &copy_mask,
                                 const VersionInfo &src_version_info,
              const LegionMap<Event,FieldMask>::aligned &preconditions,
                    LegionMap<Event,FieldMask>::aligned &postconditions,
                    LegionMap<Event,FieldMask>::aligned &postreductions,
                           CopyTracker *tracker, CopyAcrossHelper *helper,
                           bool check_root = true) const;
      CompositeNode* find_next_root(RegionTreeNode *target) const;
      void find_valid_views(const FieldMask &search_mask,
                      LegionMap<LogicalView*,FieldMask>::aligned &valid) const;
      void issue_update_copies(const TraversalInfo &info, MaterializedView *dst,
                       FieldMask copy_mask, const VersionInfo &src_version_info,
                      const LegionMap<Event,FieldMask>::aligned &preconditions,
                            LegionMap<Event,FieldMask>::aligned &postconditions,
                      const LegionMap<LogicalView*,FieldMask>::aligned &views,
                         CopyTracker *tracker, CopyAcrossHelper *helper,
                         bool across_contexts) const; 
      void issue_update_reductions(const TraversalInfo &info, 
                                   MaterializedView *dst, 
                                   const FieldMask &copy_mask,
                                   const VersionInfo &src_version_info,
                      const LegionMap<Event,FieldMask>::aligned &preconditions,
                            LegionMap<Event,FieldMask>::aligned &postconditions,
                         CopyTracker *tracker, CopyAcrossHelper *helper) const;
    public:
      void pack_composite_tree(Serializer &rez, AddressSpaceID target);
      void unpack_composite_tree(Deserializer &derez, AddressSpaceID source,
                                 Runtime *runtime,std::set<Event> &ready_events,
                                 std::map<LogicalView*,unsigned> &pending_refs);
      void make_local(std::set<Event> &preconditions, 
                      std::set<DistributedID> &checked_views);
    public:
      void notify_active(void);
      void notify_inactive(void);
      void notify_valid(void);
      void notify_invalid(void);
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
               RegionTreeNode *node, FillViewValue *value);
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
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      virtual void send_view(AddressSpaceID target); 
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
    public:
      virtual DeferredView* simplify(CompositeCloser &closer, 
                                     const FieldMask &capture_mask);
    public:
      virtual void issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                    const LegionMap<Event,FieldMask>::aligned &preconditions,
                          LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL,
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
#ifdef DEBUG_HIGH_LEVEL
      assert(is_instance_view());
#endif
      return static_cast<InstanceView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline DeferredView* LogicalView::as_deferred_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_deferred_view());
#endif
      return static_cast<DeferredView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline MaterializedView* LogicalView::as_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_materialized_view());
#endif
      return static_cast<MaterializedView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline ReductionView* LogicalView::as_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_reduction_view());
#endif
      return static_cast<ReductionView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline FillView* LogicalView::as_fill_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_fill_view());
#endif
      return static_cast<FillView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline CompositeView* LogicalView::as_composite_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_composite_view());
#endif
      return static_cast<CompositeView*>(const_cast<LogicalView*>(this));
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_VIEWS_H__
