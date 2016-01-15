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

namespace LegionRuntime {
  namespace HighLevel {

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
      virtual void send_remote_registration(void);
      static void handle_view_remote_registration(RegionTreeForest *forest,
                                                  Deserializer &derez,
                                                  AddressSpaceID source);
    public:
      virtual bool is_instance_view(void) const = 0;
      virtual bool is_deferred_view(void) const = 0;
      virtual InstanceView* as_instance_view(void) const = 0;
      virtual DeferredView* as_deferred_view(void) const = 0;
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
      DistributedID send_view(AddressSpaceID target, 
                              const FieldMask &update_mask);
      virtual DistributedID send_view_base(AddressSpaceID target) = 0;
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      void defer_collect_user(Event term_event);
      virtual void collect_users(const std::set<Event> &term_events) = 0;
      static void handle_deferred_collect(LogicalView *view,
                                          const std::set<Event> &term_events);
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
                   RegionTreeNode *node, bool register_now);
      virtual ~InstanceView(void);
    public:
      virtual bool is_instance_view(void) const;
      virtual bool is_deferred_view(void) const;
      virtual InstanceView* as_instance_view(void) const;
      virtual DeferredView* as_deferred_view(void) const;
      virtual bool is_materialized_view(void) const = 0;
      virtual bool is_reduction_view(void) const = 0;
      virtual MaterializedView* as_materialized_view(void) const = 0;
      virtual ReductionView* as_reduction_view(void) const = 0;
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
      virtual InstanceRef add_user(const RegionUsage &user, Event term_event,
                                   const FieldMask &user_mask,
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
      virtual DistributedID send_view_base(AddressSpaceID target) = 0;
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      // Instance recycling
      virtual void collect_users(const std::set<Event> &term_events) = 0;
    public:
      // Getting field information for performing copies
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields) = 0;
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual bool reduce_to(ReductionOpID redop, 
                             const FieldMask &reduce_mask,
                     std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual void reduce_from(ReductionOpID redop,const FieldMask &reduce_mask,
                       std::vector<Domain::CopySrcDstField> &src_fields) = 0;
      virtual bool has_war_dependence(const RegionUsage &usage, 
                                      const FieldMask &user_mask) = 0;
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
      template<bool MAKE>
      struct PersistenceFunctor {
      public:
        PersistenceFunctor(AddressSpaceID s, Internal *rt, 
                           SingleTask *p, LogicalRegion h,
                           LogicalRegion u,
                           DistributedID id, unsigned pidx, 
                           std::set<Event> &d)
          : source(s), runtime(rt), parent(p), handle(h),
            upper(u), did(id), parent_idx(pidx), done_events(d) { }
      public:
        void apply(AddressSpaceID target);
      protected:
        AddressSpaceID source;
        Internal *runtime;
        SingleTask *parent;
        LogicalRegion handle;
        LogicalRegion upper;
        DistributedID did;
        unsigned parent_idx;
        std::set<Event> &done_events;
      };
    public:
      MaterializedView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, AddressSpaceID local_proc,
                       RegionTreeNode *node, InstanceManager *manager,
                       MaterializedView *parent, unsigned depth,
                       bool register_now, bool persist = false);
      MaterializedView(const MaterializedView &rhs);
      virtual ~MaterializedView(void);
    public:
      MaterializedView& operator=(const MaterializedView &rhs);
    public:
      size_t get_blocking_factor(void) const;
      const FieldMask& get_physical_mask(void) const;
    public:
      virtual bool is_materialized_view(void) const;
      virtual bool is_reduction_view(void) const;
      virtual MaterializedView* as_materialized_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
      virtual bool has_space(const FieldMask &space_mask) const;
    public:
      MaterializedView* get_materialized_subview(const ColorPoint &c);
      MaterializedView* get_materialized_parent_view(void) const;
    public:
      bool is_persistent(void) const;
      void make_persistent(SingleTask *parent_ctx, unsigned parent_idx,
                           AddressSpaceID source, UserEvent to_trigger,
                           RegionTreeNode *top_node);
      void unmake_persistent(SingleTask *parent_ctx, unsigned parent_idx,
                             AddressSpaceID source, UserEvent to_trigger);
    public:
      void copy_field(FieldID fid, std::vector<Domain::CopySrcDstField> &infos);
    public:
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual void copy_from(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &src_fields);
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<Domain::CopySrcDstField> &dst_fields);
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
      virtual InstanceRef add_user(const RegionUsage &user, Event term_event,
                                   const FieldMask &user_mask,
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
      virtual DistributedID send_view_base(AddressSpaceID target);
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
      void find_atomic_reservations(InstanceRef &target, const FieldMask &mask);
    public:
      void set_descriptor(FieldDataDescriptor &desc, unsigned fid_idx) const;
    public:
      void send_back_atomic_reservations(
          const std::vector<std::pair<FieldID,Reservation> > &send_back);
      void process_atomic_reservations(Deserializer &derez);
      static void handle_send_back_atomic(RegionTreeForest *ctx,
                                          Deserializer &derez);
    public:
      static void handle_send_materialized_view(Internal *runtime,
                              Deserializer &derez, AddressSpaceID source);
      static void handle_send_update(Internal *runtime, Deserializer &derez,
                                     AddressSpaceID source);
      static void handle_make_persistent(Internal *runtime, Deserializer &derez,
                                         AddressSpaceID source);
      static void handle_unmake_persistent(Internal *runtime, 
                                   Deserializer &derez, AddressSpaceID source);
    public:
      InstanceManager *const manager;
      MaterializedView *const parent;
      const unsigned depth;
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
    protected:
      bool persistent_view; // only valid at the top-most node
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
                    bool register_now);
      ReductionView(const ReductionView &rhs);
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs);
    public:
      void perform_reduction(InstanceView *target, const FieldMask &copy_mask, 
                             const VersionInfo &version_info,
                             Processor local_proc, Operation *op,
                             CopyTracker *tracker = NULL);
      Event perform_deferred_reduction(MaterializedView *target,
                                        const FieldMask &copy_mask,
                                        const VersionInfo &version_info,
                                        const std::set<Event> &preconditions,
                                        const std::set<Domain> &reduce_domains,
                                        Event domain_precondition,
                                        Operation *op);
      Event perform_deferred_across_reduction(MaterializedView *target,
                                              FieldID dst_field,
                                              FieldID src_field,
                                              unsigned src_index,
                                       const VersionInfo &version_info,
                                       const std::set<Event> &preconditions,
                                       const std::set<Domain> &reduce_domains,
                                       Event domain_precondition,
                                       Operation *op);
    public:
      virtual bool is_materialized_view(void) const;
      virtual bool is_reduction_view(void) const;
      virtual MaterializedView* as_materialized_view(void) const;
      virtual ReductionView* as_reduction_view(void) const;
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
      virtual InstanceRef add_user(const RegionUsage &user, Event term_event,
                                   const FieldMask &user_mask,
                                   const VersionInfo &version_info);
      virtual void add_initial_user(Event term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask);
    public:
      virtual bool reduce_to(ReductionOpID redop, const FieldMask &copy_mask,
                     std::vector<Domain::CopySrcDstField> &dst_fields);
      virtual void copy_to(const FieldMask &copy_mask, 
                   std::vector<Domain::CopySrcDstField> &dst_fields);
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
      virtual DistributedID send_view_base(AddressSpaceID target);
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
      void process_update(Deserializer &derez, AddressSpaceID source);
    protected:
      void add_physical_user(PhysicalUser *user, bool reading,
                             Event term_event, const FieldMask &user_mask);
      void filter_local_users(Event term_event);
    public:
      static void handle_send_reduction_view(Internal *runtime,
                              Deserializer &derez, AddressSpaceID source);
      static void handle_send_update(Internal *runtime,
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
      struct ReductionEpoch {
      public:
        ReductionEpoch(void)
          : redop(0) { }
        ReductionEpoch(ReductionView *v, ReductionOpID r, const FieldMask &m)
          : valid_fields(m), redop(r) { views.insert(v); }
      public:
        FieldMask valid_fields;
        ReductionOpID redop;
        std::set<ReductionView*> views;
      };
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
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      virtual DistributedID send_view_base(AddressSpaceID target) = 0;
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask) = 0;
    public:
      // Should never be called
      virtual void collect_users(const std::set<Event> &term_events)
        { assert(false); }
    public:
      virtual bool is_instance_view(void) const { return false; }
      virtual bool is_deferred_view(void) const { return true; }
      virtual InstanceView* as_instance_view(void) const { return NULL; }
      virtual DeferredView* as_deferred_view(void) const
        { return const_cast<DeferredView*>(this); }
    public:
      virtual bool is_composite_view(void) const = 0;
      virtual bool is_fill_view(void) const = 0;
      virtual FillView* as_fill_view(void) const = 0;
      virtual CompositeView* as_composite_view(void) const = 0;
    public:
      void update_reduction_views(ReductionView *view, 
                                  const FieldMask &valid_mask,
                                  bool update_parent = true);
      void update_reduction_epochs(const ReductionEpoch &epoch);
    protected:
      void update_reduction_views_above(ReductionView *view,
                                        const FieldMask &valid_mask,
                                        DeferredView *from_child);
      void update_local_reduction_views(ReductionView *view,
                                        const FieldMask &valid_mask);
      virtual void update_child_reduction_views(ReductionView *view,
                                                const FieldMask &valid_mask,
                                                DeferredView *skip = NULL) = 0;
      void flush_reductions(const MappableInfo &info,
                            MaterializedView *dst,
                            const FieldMask &reduce_mask,
                            LegionMap<Event,FieldMask>::aligned &conditions);
      void flush_reductions_across(const MappableInfo &info,
                                   MaterializedView *dst,
                                   FieldID src_field, FieldID dst_field,
                                   Event dst_precondition,
                                   std::set<Event> &conditions);
      Event find_component_domains(ReductionView *reduction_view,
                                   MaterializedView *dst_view,
                                   std::set<Domain> &component_domains);
    protected:
      void activate_deferred(void);
      void deactivate_deferred(void);
      void validate_deferred(void);
      void invalidate_deferred(void);
    public:
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                                         CopyTracker *tracker = NULL) = 0;
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
             const LegionMap<Event,FieldMask>::aligned &preconditions,
                   LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL) = 0;
    public:
      virtual void issue_deferred_copies_across(const MappableInfo &info,
                                                MaterializedView *dst,
                                                FieldID src_field,
                                                FieldID dst_field,
                                                Event precondition,
                                          std::set<Event> &postconditions) = 0;
    public:
      virtual void find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions) = 0;
      virtual bool find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                          Realm::IndexSpace target,
                                          Event target_precondition,
                                  std::vector<FieldDataDescriptor> &field_data,
                                  std::set<Event> &preconditions,
                             std::vector<Realm::IndexSpace> &already_handled,
                             std::set<Event> &already_preconditions) = 0;
    protected:
      void send_deferred_view_updates(AddressSpaceID target, 
                                      const FieldMask &update_mask);
      void process_deferred_view_update(Deserializer &derez, 
                                        AddressSpaceID source);
    public:
      static void handle_deferred_update(Internal *rt, Deserializer &derez,
                                         AddressSpaceID source);
    protected:
      // Track the set of reduction views which need to be applied here
      FieldMask reduction_mask;
      // We need to keep these in order because there may be multiple
      // generations of reductions applied to this instance
      LegionDeque<ReductionEpoch>::aligned reduction_epochs;
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
                    AddressSpaceID local_proc, const FieldMask &mask,
                    bool register_now, CompositeView *parent = NULL);
      CompositeView(const CompositeView &rhs);
      virtual ~CompositeView(void);
    public:
      CompositeView& operator=(const CompositeView &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      virtual bool has_parent(void) const { return (parent != NULL); }
      virtual LogicalView* get_parent(void) const { return parent; }
      virtual LogicalView* get_subview(const ColorPoint &c);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      virtual DistributedID send_view_base(AddressSpaceID target);
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
      void unpack_composite_view(Deserializer &derez, AddressSpaceID source);
      void make_local(std::set<Event> &preconditions);
    public:
      virtual bool is_composite_view(void) const { return true; }
      virtual bool is_fill_view(void) const { return false; }
      virtual FillView* as_fill_view(void) const
        { return NULL; }
      virtual CompositeView* as_composite_view(void) const
        { return const_cast<CompositeView*>(this); }
    public:
      void update_valid_mask(const FieldMask &mask);
      void flatten_composite_view(FieldMask &global_dirt,
              const FieldMask &flatten_mask, CompositeCloser &closer,
              CompositeNode *target);
    public:
      virtual void find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions);
      virtual bool find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                          Realm::IndexSpace target,
                                          Event target_precondition,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions,
                             std::vector<Realm::IndexSpace> &already_handled,
                                       std::set<Event> &already_preconditions);
    public:
      void add_root(CompositeNode *root, const FieldMask &valid, bool top);
      virtual void update_child_reduction_views(ReductionView *view,
                                                const FieldMask &valid_mask,
                                                DeferredView *skip = NULL);
    public:
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                                         CopyTracker *tracker = NULL);
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
             const LegionMap<Event,FieldMask>::aligned &preconditions,
                   LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL);
    public:
      // Note that copy-across only works for a single field at a time
      virtual void issue_deferred_copies_across(const MappableInfo &info,
                                                MaterializedView *dst,
                                                FieldID src_field,
                                                FieldID dst_field,
                                                Event precondition,
                                         std::set<Event> &postconditions);
    public:
      static void handle_send_composite_view(Internal *runtime, 
                              Deserializer &derez, AddressSpaceID source);
    public:
      CompositeView *const parent;
    protected:
      // The set of fields represented by this composite view
      FieldMask valid_mask;
      // Keep track of the roots and their field masks
      // There is exactly one root for every field
      LegionMap<CompositeNode*,FieldMask>::aligned roots;
      // Keep track of all the child views
      std::map<ColorPoint,CompositeView*> children;
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
     * \class CompositeNode
     * A helper class for representing the frozen state of a region
     * tree as part of one or more composite views.
     */
    class CompositeNode : public Collectable {
    public:
      static const AllocationType alloc_type = COMPOSITE_NODE_ALLOC;
    public:
      struct ChildInfo {
      public:
        ChildInfo(void)
          : complete(false) { }
        ChildInfo(bool c, const FieldMask &m)
          : complete(c), open_fields(m) { }
      public:
        bool complete;
        FieldMask open_fields;
      };
    public:
      CompositeNode(RegionTreeNode *logical, CompositeNode *parent,
                    CompositeVersionInfo *version_info);
      CompositeNode(const CompositeNode &rhs);
      ~CompositeNode(void);
    public:
      CompositeNode& operator=(const CompositeNode &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void capture_physical_state(RegionTreeNode *tree_node,
                                  PhysicalState *state,
                                  const FieldMask &capture_mask,
                                  CompositeCloser &closer,
                                  FieldMask &global_dirty,
                                  const FieldMask &other_dirty_mask,
                const LegionMap<LogicalView*,FieldMask,
                        VALID_VIEW_ALLOC>::track_aligned &other_valid_views);
      CompositeNode* flatten(const FieldMask &flatten_mask, 
                             CompositeCloser &closer,
                             CompositeNode *parent,
                             FieldMask &global_dirt,
                             CompositeNode *target);
      CompositeNode* create_clone_node(CompositeNode *parent,
                                       CompositeCloser &closer);
      void update_parent_info(const FieldMask &mask);
      void update_child_info(CompositeNode *child, const FieldMask &mask);
      void update_instance_views(LogicalView *view,
                                 const FieldMask &valid_mask);
    public:
      void issue_update_copies(const MappableInfo &info,
                               MaterializedView *dst,
                               FieldMask traversal_mask,
                               const FieldMask &copy_mask,
                       const LegionMap<Event,FieldMask>::aligned &preconditions,
                           LegionMap<Event,FieldMask>::aligned &postconditions,
                               CopyTracker *tracker = NULL);
      void issue_across_copies(const MappableInfo &info,
                               MaterializedView *dst,
                               unsigned src_index,
                               FieldID  src_field,
                               FieldID  dst_field,
                               bool    need_field,
                               std::set<Event> &preconditions,
                               std::set<Event> &postconditions);
    public:
      bool intersects_with(RegionTreeNode *dst);
      const std::set<Domain>& find_intersection_domains(RegionTreeNode *dst);
    public:
      void find_bounding_roots(CompositeView *target, const FieldMask &mask);
      void set_owner_did(DistributedID owner_did);
    public:
      bool find_field_descriptors(Event term_event, const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  unsigned fid_idx, Processor local_proc, 
                                  Realm::IndexSpace target, Event target_pre,
                                  std::vector<FieldDataDescriptor> &field_data,
                                  std::set<Event> &preconditions,
                             std::vector<Realm::IndexSpace> &already_handled,
                                  std::set<Event> &already_preconditions);
    public:
      void add_gc_references(void);
      void remove_gc_references(void);
      void add_valid_references(void);
      void remove_valid_references(void);
    public:
      void pack_composite_tree(Serializer &rez, AddressSpaceID target);
      void unpack_composite_tree(Deserializer &derez, AddressSpaceID source);
      void make_local(std::set<Event> &preconditions,
                      std::set<DistributedID> &checked_views);
    protected:
      bool dominates(RegionTreeNode *dst);
      template<typename MAP_TYPE>
      void capture_instances(const FieldMask &capture_mask, 
                             FieldMask &need_flatten,
          LegionMap<CompositeView*,FieldMask>::aligned &to_flatten,
          const MAP_TYPE &instances);
    public:
      RegionTreeForest *const context;
      RegionTreeNode *const logical_node;
      CompositeNode *const parent;
      CompositeVersionInfo *const version_info;
    protected:
      DistributedID owner_did;
      FieldMask dirty_mask;
      LegionMap<CompositeNode*,ChildInfo>::aligned open_children;
      LegionMap<LogicalView*,FieldMask,
                VALID_VIEW_ALLOC>::track_aligned valid_views;
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
               RegionTreeNode *node, bool reg_now, FillViewValue *value,
               FillView *parent = NULL);
      FillView(const FillView &rhs);
      virtual ~FillView(void);
    public:
      FillView& operator=(const FillView &rhs);
    public:
      virtual bool has_parent(void) const { return (parent != NULL); }
      virtual LogicalView* get_parent(void) const { return parent; }
      virtual LogicalView* get_subview(const ColorPoint &c);
    public:
      virtual void notify_active(void);
      virtual void notify_inactive(void);
      virtual void notify_valid(void);
      virtual void notify_invalid(void);
    public:
      virtual DistributedID send_view_base(AddressSpaceID target);
      virtual void send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask);
    public:
      virtual bool is_composite_view(void) const { return false; }
      virtual bool is_fill_view(void) const { return true; }
      virtual FillView* as_fill_view(void) const
        { return const_cast<FillView*>(this); }
      virtual CompositeView* as_composite_view(void) const { return NULL; }
    public:
      virtual void update_child_reduction_views(ReductionView *view,
                                                const FieldMask &valid_mask,
                                                DeferredView *skip = NULL);
    public:
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                                         CopyTracker *tracker = NULL);
      virtual void issue_deferred_copies(const MappableInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
             const LegionMap<Event,FieldMask>::aligned &preconditions,
                   LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker = NULL);
    public:
      virtual void issue_deferred_copies_across(const MappableInfo &info,
                                                MaterializedView *dst,
                                                FieldID src_field,
                                                FieldID dst_field,
                                                Event precondition,
                                          std::set<Event> &postconditions);
    public:
      virtual void find_field_descriptors(Event term_event, 
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions);
      virtual bool find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx,
                                          Processor local_proc,
                                          Realm::IndexSpace target,
                                          Event target_precondition,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions,
                             std::vector<Realm::IndexSpace> &already_handled,
                                       std::set<Event> &already_preconditions);
    public:
      static void handle_send_fill_view(Internal *runtime, Deserializer &derez,
                                        AddressSpaceID source);
    public:
      FillView *const parent;
      FillViewValue *const value;
    protected:
      // Keep track of the child views
      std::map<ColorPoint,FillView*> children;
    };

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __LEGION_VIEWS_H__
