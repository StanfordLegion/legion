/* Copyright 2024 Stanford University, NVIDIA Corporation
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
      LogicalView(Runtime *runtime, DistributedID did,
                  bool register_now, CollectiveMapping *mapping);
      virtual ~LogicalView(void);
    public:
      inline bool deterministic_pointer_less(const LogicalView *rhs) const
        { return (did < rhs->did); }
    public:
      inline bool is_instance_view(void) const;
      inline bool is_deferred_view(void) const;
      inline bool is_individual_view(void) const;
      inline bool is_collective_view(void) const;
      inline bool is_materialized_view(void) const;
      inline bool is_reduction_view(void) const;
      inline bool is_replicated_view(void) const;
      inline bool is_allreduce_view(void) const;
      inline bool is_fill_view(void) const;
      inline bool is_phi_view(void) const;
      inline bool is_reduction_kind(void) const;
    public:
      inline InstanceView* as_instance_view(void) const;
      inline DeferredView* as_deferred_view(void) const;
      inline IndividualView* as_individual_view(void) const;
      inline CollectiveView* as_collective_view(void) const;
      inline MaterializedView* as_materialized_view(void) const;
      inline ReductionView* as_reduction_view(void) const;
      inline ReplicatedView* as_replicated_view(void) const;
      inline AllreduceView* as_allreduce_view(void) const;
      inline FillView* as_fill_view(void) const;
      inline PhiView *as_phi_view(void) const;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      static void handle_view_request(Deserializer &derez, Runtime *runtime);
    public:
      inline void add_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, int cnt = 1);
    public:
      virtual void pack_valid_ref(void) = 0;
      virtual void unpack_valid_ref(void) = 0;
    protected:
#ifndef DEBUG_LEGION_GC
      void add_valid_reference(int cnt);
      bool remove_valid_reference(int cnt);
#else
      void add_base_valid_ref_internal(ReferenceSource source, int cnt);
      void add_nested_valid_ref_internal(DistributedID source, int cnt);
      bool remove_base_valid_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_valid_ref_internal(DistributedID source, int cnt);
#endif
      virtual void notify_valid(void) = 0;
      virtual bool notify_invalid(void) = 0;
    public:
      static inline DistributedID encode_materialized_did(DistributedID did);
      static inline DistributedID encode_reduction_did(DistributedID did);
      static inline DistributedID encode_replicated_did(DistributedID did);
      static inline DistributedID encode_allreduce_did(DistributedID did);
      static inline DistributedID encode_fill_did(DistributedID did);
      static inline DistributedID encode_phi_did(DistributedID did);
      static inline bool is_materialized_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_replicated_did(DistributedID did);
      static inline bool is_allreduce_did(DistributedID did);
      static inline bool is_individual_did(DistributedID did);
      static inline bool is_collective_did(DistributedID did);
      static inline bool is_fill_did(DistributedID did);
      static inline bool is_phi_did(DistributedID did);
    protected:
      mutable LocalLock view_lock;
    protected:
#ifdef DEBUG_LEGION_GC
      int valid_references;
#else
      std::atomic<int> valid_references;
#endif
#ifdef DEBUG_LEGION_GC
    protected:
      std::map<ReferenceSource,int> detailed_base_valid_references;
      std::map<DistributedID,int> detailed_nested_valid_references;
#endif
    };

    /**
     * \class InstanceView 
     * The InstanceView class is used for performing the dependence
     * analysis for a single physical instance.
     * The InstaceView class has two sub-classes: materialized
     * views which represent a normal instance a reduction
     * view which is a specialized instance for storing reductions
     */
    class InstanceView : public LogicalView {
    public:
      // This structure acts as a key for performing rendezvous
      // between collective user registrations
      struct RendezvousKey {
      public:
        RendezvousKey(void)
          : op_context_index(0), match(0), index(0) { }
        RendezvousKey(size_t ctx, unsigned idx, IndexSpaceID m)
          : op_context_index(ctx), match(m), index(idx) { }
      public:
        inline bool operator<(const RendezvousKey &rhs) const
        {
          if (op_context_index < rhs.op_context_index) return true;
          if (op_context_index > rhs.op_context_index) return false;
          if (match < rhs.match) return true;
          if (match > rhs.match) return false;
          return (index < rhs.index);
        }
      public:
        size_t op_context_index; // unique name operation in context
        IndexSpaceID match; // index space of regions that should match
        unsigned index; // uniquely name analysis for op by region req index
      };
    public:
      typedef LegionMap<ApEvent,FieldMask> EventFieldMap;
      typedef LegionMap<ApEvent,FieldMaskSet<PhysicalUser> > EventFieldUsers;
      typedef FieldMaskSet<PhysicalUser> EventUsers;
    public:
      InstanceView(Runtime *runtime, DistributedID did,
                   bool register_now, CollectiveMapping *mapping);
      virtual ~InstanceView(void);  
    public:
      virtual ApEvent fill_from(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool fill_restricted,
                                const bool need_valid_return) = 0;
      virtual ApEvent copy_from(InstanceView *src_view, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &copy_mask,
                                PhysicalManager *src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool copy_restricted,
                                const bool need_valid_return) = 0;
      // Always want users to be full index space expressions
      virtual ApEvent register_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const size_t op_ctx_index,
                                    const unsigned index,
                                    const IndexSpaceID collective_match_space,
                                    ApEvent term_event,
                                    PhysicalManager *target,
                                    CollectiveMapping *collective_mapping,
                                    size_t local_collective_arrivals,
                                    std::vector<RtEvent> &registered_events,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info,
                                    const AddressSpaceID source,
                                    const bool symbolic = false) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
      virtual ReductionOpID get_redop(void) const { return 0; }
      virtual FillView* get_redop_fill_view(void) const 
        { assert(false); return NULL; }
      virtual AddressSpaceID get_analysis_space(PhysicalManager *man) const = 0;
      virtual bool aliases(InstanceView *other) const = 0;
    public:
      static void handle_view_register_user(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
    }; 

    /**
     * \class IndividualView
     * This class provides an abstract base class for any kind of view 
     * that only represents an individual physical instance.
     */
    class IndividualView : public InstanceView { 
    public:
      IndividualView(Runtime *runtime, DistributedID did,
                     PhysicalManager *man, AddressSpaceID logical_owner,
                     bool register_now, CollectiveMapping *mapping); 
      virtual ~IndividualView(void);
    public:
      inline bool is_logical_owner(void) const
        { return (local_space == logical_owner); } 
      inline PhysicalManager* get_manager(void) const { return manager; }
      void destroy_reservations(ApEvent all_done);
    public:
      virtual AddressSpaceID get_analysis_space(PhysicalManager *inst) const;
      virtual bool aliases(InstanceView *other) const;
    public:
      // Reference counting state change functions
      virtual void notify_local(void);
      virtual void notify_valid(void);
      virtual bool notify_invalid(void);
    public:
      virtual void pack_valid_ref(void);
      virtual void unpack_valid_ref(void);
    public:
      virtual ApEvent fill_from(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool fill_restricted,
                                const bool need_valid_return);
      virtual ApEvent copy_from(InstanceView *src_view, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &copy_mask,
                                PhysicalManager *src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool copy_restricted,
                                const bool need_valid_return);
    public:
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceExpression *expr,
                                    const UniqueID op_id,
                                    const unsigned index) = 0;
      virtual ApEvent find_copy_preconditions(bool reading,
                                    ReductionOpID redop,              
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info) = 0;
      virtual void add_copy_user(bool reading, ReductionOpID redop,
                                 ApEvent done_event,
                                 const FieldMask &copy_mask,
                                 IndexSpaceExpression *copy_expr,
                                 UniqueID op_id, unsigned index,
                                 std::set<RtEvent> &applied_events,
                                 const bool trace_recording,
                                 const AddressSpaceID source) = 0;
      virtual void find_last_users(PhysicalManager *target,
                                   std::set<ApEvent> &events,
                                   const RegionUsage &usage,
                                   const FieldMask &mask,
                                   IndexSpaceExpression *user_expr,
                                   std::vector<RtEvent> &applied) const = 0;
    public:
      void pack_fields(Serializer &rez,
                       const std::vector<CopySrcDstField> &fields) const;
      void find_atomic_reservations(const FieldMask &mask, Operation *op, 
                                    const unsigned index, bool exclusive);
      void find_field_reservations(const FieldMask &mask,
                                   std::vector<Reservation> &results);
    protected:
      RtEvent find_field_reservations(const FieldMask &mask,
                                      std::vector<Reservation> *results,
                                      AddressSpaceID source,
                                      RtUserEvent to_trigger =
                                        RtUserEvent::NO_RT_USER_EVENT);
      void update_field_reservations(const FieldMask &mask,
                                     const std::vector<Reservation> &rsrvs);
    public: 
      void register_collective_analysis(const CollectiveView *source,
                                        CollectiveAnalysis *analysis);
      CollectiveAnalysis* find_collective_analysis(size_t context_index,
                        unsigned region_index, IndexSpaceID match_space);
      void unregister_collective_analysis(const CollectiveView *source,
                                          size_t context_index,
                                          unsigned region_index,
                                          IndexSpaceID match_space);
    protected:
      ApEvent register_collective_user(const RegionUsage &usage,
                                       const FieldMask &user_mask,
                                       IndexSpaceNode *expr,
                                       const UniqueID op_id,
                                       const size_t op_ctx_index,
                                       const unsigned index,
                                       const IndexSpaceID match_space,
                                       ApEvent term_event,
                                       PhysicalManager *target,
                                       CollectiveMapping *analysis_mapping,
                                       size_t local_collective_arrivals,
                                       std::vector<RtEvent> &registered_events,
                                       std::set<RtEvent> &applied_events,
                                       const PhysicalTraceInfo &trace_info,
                                       const bool symbolic);
      void process_collective_user_registration(const size_t op_ctx_index,
                                            const unsigned index,
                                            const IndexSpaceID match_space,
                                            const AddressSpaceID origin,
                                            const PhysicalTraceInfo &trace_info,
                                            CollectiveMapping *analysis_mapping,
                                            ApEvent remote_term_event,
                                            ApUserEvent remote_ready_event,
                                            RtUserEvent remote_registered,
                                            std::set<RtEvent> &applied_events);
    public:
      static void handle_view_find_copy_pre_request(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_add_copy_user(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_find_last_users_request(Deserializer &derz,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_find_last_users_response(Deserializer &derez);
      static void handle_collective_user_registration(Runtime *runtime,
                                                      Deserializer &derez);
    public:
      static void handle_atomic_reservation_request(Runtime *runtime,
                                                    Deserializer &derez);
      static void handle_atomic_reservation_response(Runtime *runtime,
                                                     Deserializer &derez);
#ifdef ENABLE_VIEW_REPLICATION
    public:
      virtual void process_replication_request(AddressSpaceID source,
                                 const FieldMask &request_mask,
                                 RtUserEvent done_event) = 0;
      virtual void process_replication_response(RtUserEvent done_event,
                                 Deserializer &derez) = 0;
      virtual void process_replication_removal(AddressSpaceID source,
                                 const FieldMask &removal_mask) = 0;
      static void handle_view_replication_request(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
      static void handle_view_replication_response(Deserializer &derez,
                        Runtime *runtime);
      static void handle_view_replication_removal(Deserializer &derez,
                        Runtime *runtime, AddressSpaceID source);
#endif
    public:
      PhysicalManager *const manager; 
      // This is the owner space for the purpose of logical analysis
      // If you ever make this non-const then be sure to update the
      // code in register_collective_user
      const AddressSpaceID logical_owner;
    protected:
      std::map<unsigned,Reservation> view_reservations;
    protected:
      // This is an infrequently used data structure for handling collective
      // register user calls on individual managers that occurs with certain
      // operation in control replicated contexts
      struct UserRendezvous {
        UserRendezvous(void) 
          : remaining_local_arrivals(0), remaining_remote_arrivals(0),
            trace_info(NULL), analysis_mapping(NULL), mask(NULL),
            expr(NULL), op_id(0), symbolic(false), local_initialized(false) { }
        // event for when local instances can be used
        ApUserEvent ready_event; 
        // remote ready events to trigger
        std::map<ApUserEvent,PhysicalTraceInfo*> remote_ready_events;
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
        PhysicalTraceInfo *trace_info;
        CollectiveMapping *analysis_mapping;
        // Arguments for performing the local registration
        RegionUsage usage;
        FieldMask *mask;
        IndexSpaceNode *expr;
        UniqueID op_id;
        bool symbolic;
        bool local_initialized;
      };
      std::map<RendezvousKey,UserRendezvous> rendezvous_users;
    protected:
      // This is actually quite important!
      // Normally each collective analysis is associated with a specific
      // collective view. However the copies done by that analysis might
      // only be occurring on collective views that are a subset of the 
      // collective view for the analysis. Therefore we register the analyses
      // with the individual views so that they can be found by any copies
      struct RegisteredAnalysis {
      public:
        CollectiveAnalysis *analysis;
        RtUserEvent            ready;
        // We need to deduplicate across views that are performing
        // registrations on this instance. With multiple fields we
        // can get multiple different views using the same instance
        // and each doing their own registration
        std::set<DistributedID> views;
      };
      std::map<RendezvousKey,RegisteredAnalysis> collective_analyses;
    };

    /**
     * \class CollectiveView
     * This class provides an abstract base class for any kind of view
     * that represents a group of instances that need to be analyzed
     * cooperatively for physical analysis.
     */
    class CollectiveView : public InstanceView, 
                           public InstanceDeletionSubscriber {
    public:
      enum ValidState {
        FULL_VALID_STATE,
        PENDING_INVALID_STATE,
        NOT_VALID_STATE, 
      };
    public:
      CollectiveView(Runtime *runtime, DistributedID did,
                     DistributedID context_did,
                     const std::vector<IndividualView*> &views,
                     const std::vector<DistributedID> &instances,
                     bool register_now, CollectiveMapping *mapping); 
      virtual ~CollectiveView(void);
    public:
      virtual AddressSpaceID get_analysis_space(PhysicalManager *inst) const;
      virtual bool aliases(InstanceView *other) const;
    public:
      // Reference counting state change functions
      virtual void notify_local(void);
      virtual void notify_valid(void);
      virtual bool notify_invalid(void);
    public:
      virtual void pack_valid_ref(void);
      virtual void unpack_valid_ref(void);
    public:
      virtual ApEvent fill_from(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool fill_restricted,
                                const bool need_valid_return);
      virtual ApEvent copy_from(InstanceView *src_view, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &copy_mask,
                                PhysicalManager *src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool copy_restricted,
                                const bool need_valid_return);
      virtual ApEvent register_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const size_t op_ctx_index,
                                    const unsigned index,
                                    const IndexSpaceID collective_match_space,
                                    ApEvent term_event,
                                    PhysicalManager *target,
                                    CollectiveMapping *collective_mapping,
                                    size_t local_collective_arrivals,
                                    std::vector<RtEvent> &registered_events,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info,
                                    const AddressSpaceID source,
                                    const bool symbolic = false);
      // This is a special entry point variation copy_from only for
      // collective view (not it is not virtual) that will handle the 
      // special case where we have a bunch of individual views that
      // we'll be copying to this collective view, so we can do all
      // the individual copies to a local instance, and then fuse the
      // resulting broadcast or reduce out to everywhere
      ApEvent collective_fuse_gather(
                const std::map<IndividualView*,IndexSpaceExpression*> &sources,
                                ApEvent precondition, PredEvent predicate_guard,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                const bool copy_restricted,
                                const bool need_valid_return);
    public:
      void perform_collective_fill(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID match_space,
                                const size_t op_context_index,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent result, AddressSpaceID origin,
                                const bool fill_restricted);
      ApEvent perform_collective_point(
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const Memory location,
                                const UniqueInst &dst_inst,
                                const LgEvent dst_unique_event,
                                const DistributedID src_inst_did,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CollectiveKind collective = COLLECTIVE_NONE);
      void perform_collective_broadcast(
                                const std::vector<CopySrcDstField> &src_fields,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const UniqueInst &src_inst,
                                const LgEvent src_unique_event,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent copy_done, ApUserEvent all_done,
                                ApBarrier all_bar, ShardID owner_shard,
                                AddressSpaceID origin,
                                const bool copy_restricted,
                                const CollectiveKind collective_kind);
      void perform_collective_reducecast(ReductionView *source,
                                const std::vector<CopySrcDstField> &src_fields,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const UniqueInst &src_inst,
                                const LgEvent src_unique_event,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent copy_done,
                                ApBarrier all_bar, ShardID owner_shard,
                                AddressSpaceID origin,
                                const bool copy_restricted);
      void perform_collective_hourglass(AllreduceView *source,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const FieldMask &copy_mask,
                                const DistributedID src_inst_did,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent all_done,
                                AddressSpaceID target,
                                const bool copy_restricted);
      void perform_collective_pointwise(CollectiveView *source,
                                ApEvent precondition,
                                PredEvent predicate_guard, 
                                IndexSpaceExpression *copy_expression,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const DistributedID src_inst_did,
                                const UniqueID src_inst_op_id,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent all_done, ApBarrier all_bar,
                                ShardID owner_shard, AddressSpaceID origin,
                                const uint64_t allreduce_tag,
                                const bool copy_restricted);
    public:
      inline AddressSpaceID select_origin_space(void) const
        { return (collective_mapping->contains(local_space) ? local_space :
                  collective_mapping->find_nearest(local_space)); }
      bool contains(PhysicalManager *manager) const;
      bool meets_regions(const std::vector<LogicalRegion> &regions,
                         bool tight_bounds = false) const;
      void find_instances_in_memory(Memory memory,
                                    std::vector<PhysicalManager*> &instances);
      void find_instances_nearest_memory(Memory memory,
                                    std::vector<PhysicalManager*> &instances,
                                    bool bandwidth);
    protected:
      void process_remote_instances_response(AddressSpaceID source,
                          const std::vector<IndividualView*> &view);
      void record_remote_instances(const std::vector<IndividualView*> &view);
      RtEvent find_instances_nearest_memory(Memory memory, 
                                    AddressSpaceID source,
                                    std::vector<DistributedID> *instances,
                                    std::atomic<size_t> *target,
                                    AddressSpaceID origin, size_t best,
                                    bool bandwidth);
      void find_nearest_local_instances(Memory memory, size_t &best,
                                    std::vector<PhysicalManager*> &results,
                                    bool bandwidth) const;
    public:
      AddressSpaceID select_source_space(AddressSpaceID destination) const;
      void pack_fields(Serializer &rez,
                       const std::vector<CopySrcDstField> &fields) const;
      unsigned find_local_index(PhysicalManager *target) const;
      void register_collective_analysis(PhysicalManager *target,
                                        CollectiveAnalysis *analysis,
                                        std::set<RtEvent> &applied_events);
    public:
      void notify_instance_deletion(RegionTreeID tid);
      virtual void notify_instance_deletion(PhysicalManager *manager);
      virtual void add_subscriber_reference(PhysicalManager *manager);
      virtual bool remove_subscriber_reference(PhysicalManager *manager);
    protected:
      ApEvent register_collective_user(const RegionUsage &usage,
                                       const FieldMask &user_mask,
                                       IndexSpaceNode *expr,
                                       const UniqueID op_id,
                                       const size_t op_ctx_index,
                                       const unsigned index,
                                       const IndexSpaceID match_space,
                                       ApEvent term_event,
                                       PhysicalManager *target,
                                       size_t local_collective_arrivals,
                                       std::vector<RtEvent> &regsitered_events,
                                       std::set<RtEvent> &applied_events,
                                       const PhysicalTraceInfo &trace_info,
                                       const bool symbolic);
      void process_register_user_request(const size_t op_ctx_index,
                                       const unsigned index,
                                       const IndexSpaceID match_space,
                                       RtEvent registered, RtEvent applied);
      void process_register_user_response(const size_t op_ctx_index,
                                       const unsigned index,
                                       const IndexSpaceID match_space,
                                       const RtEvent registered,
                                       const RtEvent applied);
      void finalize_collective_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const size_t op_ctx_index,
                                    const unsigned index,
                                    const IndexSpaceID match_space,
                                    RtUserEvent local_registered,
                                    RtEvent global_registered,
                                    RtUserEvent local_applied,
                                    RtEvent global_applied,
                                    std::vector<ApUserEvent> &ready_events,
                                    std::vector<std::vector<ApEvent> > &terms,
                                    const PhysicalTraceInfo *trace_info,
                                    const bool symbolic);
      void perform_local_broadcast(IndividualView *local_view,
                                const std::vector<CopySrcDstField> &src_fields,
                                const std::vector<AddressSpaceID> &children,
                                CollectiveAnalysis *first_local_analysis,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const IndexSpaceID collective_match_space,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const UniqueInst &src_inst,
                                const LgEvent src_unique_event,
                                const PhysicalTraceInfo &local_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent all_done,
                                ApBarrier all_bar, ShardID owner_shard,
                                AddressSpaceID origin,
                                const bool copy_restricted,
                                const CollectiveKind collective_kind);
    protected:
      void broadcast_local(const PhysicalManager *src_manager,
                           const unsigned src_index, Operation *op,
                           const unsigned index,IndexSpaceExpression *copy_expr,
                           const FieldMask &copy_mask,
                           ApEvent precondition, PredEvent predicate_guard,
                           const std::vector<CopySrcDstField> &src_fields,
                           const UniqueInst &src_inst,
                           const PhysicalTraceInfo &trace_info,
                           const CollectiveKind collective_kind,
                           std::vector<ApEvent> &destination_events,
                           std::set<RtEvent> &recorded_events,
                           std::set<RtEvent> &applied_events,
                           const bool has_instance_events = false,
                           const bool first_local_analysis = false,
                           const size_t op_ctx_index = 0,
                           const IndexSpaceID match_space = 0); 
      const std::vector<std::pair<unsigned,unsigned> >&
                  find_spanning_broadcast_copies(unsigned root_index);
      bool construct_spanning_adjacency_matrix(unsigned root_index,
                  const std::map<Memory,unsigned> &first_in_memory,
                  std::vector<float> &adjacency_matrix) const;
      void compute_spanning_tree_same_bandwidth(unsigned root_index,
                  const std::vector<float> &adjacency_matrix,
                  std::vector<unsigned> &previous,
                  std::map<Memory,unsigned> &first_in_memory) const;
      void compute_spanning_tree_diff_bandwidth(unsigned root_index,
                  const std::vector<float> &adjacency_matrix,
                  std::vector<unsigned> &previous,
                  std::map<Memory,unsigned> &first_in_memory) const;
    protected:
      void make_valid(bool need_lock);
      bool make_invalid(bool need_lock);
      bool perform_invalidate_request(uint64_t generation, bool need_lock);
      bool perform_invalidate_response(uint64_t generation, uint64_t sent,
                          uint64_t received, bool failed, bool need_lock);
    public:
      static void handle_register_user_request(Runtime *runtime,
                                    Deserializer &derez);
      static void handle_register_user_response(Runtime *runtime,
                                    Deserializer &derez);
      static void handle_remote_instances_request(Runtime *runtime,
                                    Deserializer &derez, AddressSpaceID source);
      static void handle_remote_instances_response(Runtime *runtime,
                                    Deserializer &derez, AddressSpaceID source);
      static void handle_nearest_instances_request(Runtime *runtime,
                                                   Deserializer &derez);
      static void handle_nearest_instances_response(Deserializer &derez);
      static void process_nearest_instances(std::atomic<size_t> *target,
          std::vector<DistributedID> *instances, size_t best,
          const std::vector<DistributedID> &results, bool bandwidth);
      static void handle_remote_analysis_registration(Deserializer &derez,
                                                      Runtime *runtime);
      static void handle_collective_view_deletion(Deserializer &derez,
                                                  Runtime *runtime);
      static void unpack_fields(std::vector<CopySrcDstField> &fields,
          Deserializer &derez, std::set<RtEvent> &ready_events,
          CollectiveView *view, RtEvent view_ready, Runtime *runtime);
      static void handle_distribute_fill(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_point(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_broadcast(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_reducecast(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_hourglass(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_pointwise(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_fuse_gather(Runtime *runtime,
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_make_valid(Runtime *runtime, Deserializer &derez);
      static void handle_make_invalid(Runtime *runtime, Deserializer &derez);
      static void handle_invalidate_request(Runtime *runtime, 
                                            Deserializer &derez);
      static void handle_invalidate_response(Runtime *runtime,
                                             Deserializer &derez);
      static void handle_add_remote_reference(Runtime *runtime, 
                                              Deserializer &derez);
      static void handle_remove_remote_reference(Runtime *runtime,
                                                 Deserializer &derez);
      static bool has_multiple_local_memories(
                    const std::vector<IndividualView*> &local_views);
    public:
      const DistributedID context_did;
      const std::vector<DistributedID> instances;
      const std::vector<IndividualView*> local_views;
    protected:
      std::map<PhysicalManager*,IndividualView*> remote_instances;
      NodeSet remote_instance_responses;
    protected:
      struct UserRendezvous {
        UserRendezvous(void) 
          : remaining_local_arrivals(0), remaining_remote_arrivals(0),
            remaining_analyses(0), trace_info(NULL), mask(NULL), expr(NULL),
            op_id(0), symbolic(false), local_initialized(false) { }
        // event for when local instances can be used
        std::vector<ApUserEvent> ready_events;
        // all the local term events for each view
        std::vector<std::vector<ApEvent> > local_term_events;
        // events from remote nodes indicating they are registered
        std::vector<RtEvent> remote_registered;
        // events from remote nodes indicating they are applied
        std::vector<RtEvent> remote_applied;
        // event to trigger when local registration is done
        RtUserEvent local_registered; 
        // event that marks when all registrations are done
        RtUserEvent global_registered;
        // event to trigger when local effects are done
        RtUserEvent local_applied; 
        // event that marks when all effects are done
        RtUserEvent global_applied;
        // Counts of remaining notficiations before registration
        unsigned remaining_local_arrivals;
        unsigned remaining_remote_arrivals;
        unsigned remaining_analyses;
        // PhysicalTraceInfo that made the ready_event and should trigger it
        PhysicalTraceInfo *trace_info;
        // Arguments for performing the local registration
        RegionUsage usage;
        FieldMask *mask;
        IndexSpaceNode *expr;
        UniqueID op_id;
        bool symbolic;
        bool local_initialized;
      };
      std::map<RendezvousKey,UserRendezvous> rendezvous_users;
    private:
      // For valid state tracking
      ValidState valid_state;
      uint32_t remaining_invalidation_responses;
      uint64_t invalidation_generation;
      uint64_t total_valid_sent, total_valid_received;
      uint64_t sent_valid_references, received_valid_references;
      bool invalidation_failed;
    private:
      // Use this flag to deduplicate deletion notifications from our instances
      std::atomic<bool> deletion_notified;
    protected:
      // Whether our local views are contained in multiple local memories
      const bool multiple_local_memories;
      std::map<unsigned,
        std::vector<std::pair<unsigned,unsigned> > > spanning_copies;
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
    class ExprView : public LegionHeapify<ExprView>, public Collectable {
    public:
      typedef LegionMap<ApEvent,FieldMask> EventFieldMap;
      typedef LegionMap<ApEvent,FieldMaskSet<PhysicalUser> > EventFieldUsers;
      typedef FieldMaskSet<PhysicalUser> EventUsers;
    public:
      ExprView(RegionTreeForest *ctx, PhysicalManager *manager,
               MaterializedView *view, IndexSpaceExpression *expr); 
      ExprView(const ExprView &rhs) = delete;
      virtual ~ExprView(void);
    public:
      ExprView& operator=(const ExprView &rhs) = delete;
    public:
      inline bool deterministic_pointer_less(const ExprView *rhs) const
        { return view_expr->deterministic_pointer_less(rhs->view_expr); }
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
                                   std::set<ApEvent> &preconditions,
                                   const bool trace_recording);
      void find_last_users(const RegionUsage &usage,
                                   IndexSpaceExpression *expr,
                                   const bool expr_dominates,
                                   const FieldMask &mask,
                                   std::set<ApEvent> &last_events) const;
      // Check to see if there is any view with the same shape already
      // in the ExprView tree, if so return it
      ExprView* find_congruent_view(IndexSpaceExpression *expr);
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
                            IndexSpaceExpression *user_expr,
                            const size_t user_volume);
      void add_current_user(PhysicalUser *user, const ApEvent term_event,
                            const FieldMask &user_mask);
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
                                      std::set<ApEvent> &preconditions,
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
                                      std::set<ApEvent> &preconditions,
                                      std::set<ApEvent> &dead_events,
                                      const bool trace_recording);
      // Overloads for find_last_users
      void find_current_preconditions(const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      IndexSpaceExpression *expr,
                                      const bool expr_covers,
                                      std::set<ApEvent> &last_events,
                                      FieldMask &observed, 
                                      FieldMask &non_dominated) const;
      void find_previous_preconditions(const RegionUsage &usage,
                                      const FieldMask &user_mask,
                                      IndexSpaceExpression *expr,
                                      const bool expr_covers,
                                      std::set<ApEvent> &last_events) const;

      template<bool COPY_USER>
      inline bool has_local_precondition(PhysicalUser *prev_user,
                                      const RegionUsage &next_user,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers,
                                      bool &dominates) const;
      template<bool COPY_USER>
      inline bool has_local_precondition(PhysicalUser *prev_user,
                                      const RegionUsage &next_user,
                                      IndexSpaceExpression *user_expr,
                                      const UniqueID op_id,
                                      const unsigned index,
                                      const bool user_covers) const;
    public:
      size_t get_view_volume(void);
      void find_all_done_events(std::set<ApEvent> &all_done) const;
    protected:
      void filter_local_users(ApEvent term_event);
      void filter_current_users(const EventFieldUsers &to_filter);
      void filter_previous_users(const EventFieldUsers &to_filter);
      bool refine_users(void);
      static void verify_current_to_filter(const FieldMask &dominated,
                                  EventFieldUsers &current_to_filter);
    public:
      RegionTreeForest *const forest;
      PhysicalManager *const manager;
      MaterializedView *const inst_view;
      IndexSpaceExpression *const view_expr;
      std::atomic<size_t> view_volume;
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
                      const unsigned index, const ApEvent term_event);
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
    };

    class PendingCopyUser : public RemotePendingUser, 
                            public LegionHeapify<PendingCopyUser> {
    public:
      PendingCopyUser(const bool reading, const FieldMask &copy_mask,
                      IndexSpaceExpression *copy_expr, const UniqueID op_id,
                      const unsigned index, const ApEvent term_event);
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
    };

    /**
     * \class MaterializedView 
     * This class represents a view on to a single normal physical 
     * instance in a specific memory.
     */
    class MaterializedView : public IndividualView, 
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
                                  AddressSpaceID log)
          : LgTaskArgs<DeferMaterializedViewArgs>(implicit_provenance),
            did(d), manager(m), logical_owner(log) { }
      public:
        const DistributedID did;
        PhysicalManager *const manager;
        const AddressSpaceID logical_owner;
      };
    public:
      MaterializedView(Runtime *runtime, DistributedID did,
                       AddressSpaceID logical_owner, PhysicalManager *manager,
                       bool register_now, CollectiveMapping *mapping = NULL);
      MaterializedView(const MaterializedView &rhs) = delete;
      virtual ~MaterializedView(void);
    public:
      MaterializedView& operator=(const MaterializedView &rhs) = delete;
    public:
      inline const FieldMask& get_space_mask(void) const 
        { return manager->layout->allocated_fields; }
    public:
      const FieldMask& get_physical_mask(void) const;
    public:
      virtual bool has_space(const FieldMask &space_mask) const;
    public: // From InstanceView
      virtual void send_view(AddressSpaceID target);
      // Always want users to be full index space expressions
      virtual ApEvent register_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const size_t op_ctx_index,
                                    const unsigned index,
                                    const IndexSpaceID collective_match_space,
                                    ApEvent term_event,
                                    PhysicalManager *target,
                                    CollectiveMapping *collective_mapping,
                                    size_t local_collective_arrivals,
                                    std::vector<RtEvent> &registered_events,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info,
                                    const AddressSpaceID source,
                                    const bool symbolic = false);
    public: // From IndividualView
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceExpression *expr,
                                    const UniqueID op_id,
                                    const unsigned index);
      virtual ApEvent find_copy_preconditions(bool reading,
                                    ReductionOpID redop,              
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info);
      virtual void add_copy_user(bool reading, ReductionOpID redop,
                                 ApEvent term_event,
                                 const FieldMask &copy_mask,
                                 IndexSpaceExpression *copy_expr,
                                 UniqueID op_id, unsigned index,
                                 std::set<RtEvent> &applied_events,
                                 const bool trace_recording,
                                 const AddressSpaceID source);
      virtual void find_last_users(PhysicalManager *manager,
                                   std::set<ApEvent> &events,
                                   const RegionUsage &usage,
                                   const FieldMask &mask,
                                   IndexSpaceExpression *user_expr,
                                   std::vector<RtEvent> &applied) const;
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
    protected:
      friend class PendingTaskUser;
      friend class PendingCopyUser;
      void add_internal_task_user(const RegionUsage &usage,
                                  IndexSpaceExpression *user_expr,
                                  const FieldMask &user_mask,
                                  ApEvent term_event, 
                                  UniqueID op_id,
                                  const unsigned index);
      void add_internal_copy_user(const RegionUsage &usage,
                                  IndexSpaceExpression *user_expr,
                                  const FieldMask &user_mask,
                                  ApEvent term_event, 
                                  UniqueID op_id,
                                  const unsigned index);
      template<bool NEED_EXPR_LOCK>
      void clean_cache(void);
#ifdef ENABLE_VIEW_REPLICATION
      // Must be called while holding the replication lock
      void update_remote_replication_state(std::set<RtEvent> &applied_events);
#endif 
    public:
      static void handle_send_materialized_view(Runtime *runtime,
                                                Deserializer &derez);
      static void handle_defer_materialized_view(const void *args, Runtime *rt);
      static void create_remote_view(Runtime *runtime, DistributedID did, 
                                     PhysicalManager *manager,
                                     AddressSpaceID logical_owner); 
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
     * \class ReplicatedView
     * This class represents a group of normal instances which all
     * must contain the same copy of data.
     */
    class ReplicatedView : public CollectiveView,
                           public LegionHeapify<ReplicatedView> {
    public:
      static const AllocationType alloc_type = REPLICATED_VIEW_ALLOC;
    public:
      ReplicatedView(Runtime *runtime, DistributedID did, DistributedID ctx_did,
                     const std::vector<IndividualView*> &views,
                     const std::vector<DistributedID> &instances,
                     bool register_now, CollectiveMapping *mapping);
      ReplicatedView(const ReplicatedView &rhs) = delete;
      virtual ~ReplicatedView(void);
    public:
      ReplicatedView& operator=(const ReplicatedView &rhs) = delete;
    public: // From InstanceView
      virtual void send_view(AddressSpaceID target);
      static void handle_send_replicated_view(Runtime *runtime,
                                              Deserializer &derez);
    };

    /**
     * \class ReductionView
     * This class represents a single reduction physical instance
     * in a specific memory.
     */
    class ReductionView : public IndividualView,
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
                               AddressSpaceID log)
          : LgTaskArgs<DeferReductionViewArgs>(implicit_provenance),
            did(d), manager(m), logical_owner(log) { }
      public:
        const DistributedID did;
        PhysicalManager *const manager;
        const AddressSpaceID logical_owner;
      };
    public:
      ReductionView(Runtime *runtime, DistributedID did,
                    AddressSpaceID logical_owner, PhysicalManager *manager,
                    bool register_now, CollectiveMapping *mapping = NULL);
      ReductionView(const ReductionView &rhs) = delete;
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView&rhs) = delete;
    public: // From InstanceView
      virtual void send_view(AddressSpaceID target);
      virtual ReductionOpID get_redop(void) const; 
      virtual FillView* get_redop_fill_view(void) const { return fill_view; }
      // Always want users to be full index space expressions
      virtual ApEvent register_user(const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceNode *expr,
                                    const UniqueID op_id,
                                    const size_t op_ctx_index,
                                    const unsigned index,
                                    const IndexSpaceID collective_match_space,
                                    ApEvent term_event,
                                    PhysicalManager *target,
                                    CollectiveMapping *collective_mapping,
                                    size_t local_collective_arrivals,
                                    std::vector<RtEvent> &registered_events,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info,
                                    const AddressSpaceID source,
                                    const bool symbolic = false);
    public: // From IndividualView
      virtual void add_initial_user(ApEvent term_event,
                                    const RegionUsage &usage,
                                    const FieldMask &user_mask,
                                    IndexSpaceExpression *expr,
                                    const UniqueID op_id,
                                    const unsigned index);
      virtual ApEvent find_copy_preconditions(bool reading,
                                    ReductionOpID redop,              
                                    const FieldMask &copy_mask,
                                    IndexSpaceExpression *copy_expr,
                                    UniqueID op_id, unsigned index,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info);
      virtual void add_copy_user(bool reading, ReductionOpID redop,
                                 ApEvent term_event,
                                 const FieldMask &copy_mask,
                                 IndexSpaceExpression *copy_expr,
                                 UniqueID op_id, unsigned index,
                                 std::set<RtEvent> &applied_events,
                                 const bool trace_recording,
                                 const AddressSpaceID source);
      virtual void find_last_users(PhysicalManager *manager,
                                   std::set<ApEvent> &events,
                                   const RegionUsage &usage,
                                   const FieldMask &mask,
                                   IndexSpaceExpression *user_expr,
                                   std::vector<RtEvent> &applied) const;
    protected: 
      void find_reducing_preconditions(const RegionUsage &usage,
                                       const FieldMask &user_mask,
                                       IndexSpaceExpression *user_expr,
                                       std::set<ApEvent> &wait_on) const;
      void find_writing_preconditions(const FieldMask &user_mask,
                                      IndexSpaceExpression *user_expr,
                                      std::set<ApEvent> &preconditions,
                                      const bool trace_recording);
      void find_reading_preconditions(const FieldMask &user_mask,
                                      IndexSpaceExpression *user_expr,
                                      std::set<ApEvent> &preconditions) const;
      void find_initializing_last_users(const FieldMask &user_mask,
                                        IndexSpaceExpression *user_expr,
                                        std::set<ApEvent> &preconditions) const;
      void add_user(const RegionUsage &usage,
                    IndexSpaceExpression *user_expr,
                    const FieldMask &user_mask, ApEvent term_event,
                    UniqueID op_id, unsigned index, bool copy_user);
    protected:
      void add_physical_user(PhysicalUser *user, bool reading,
                             ApEvent term_event, const FieldMask &user_mask);
      void find_dependences(const EventFieldUsers &users,
                            IndexSpaceExpression *user_expr,
                            const FieldMask &user_mask,
                            std::set<ApEvent> &wait_on) const;
      void find_dependences_and_filter(EventFieldUsers &users,
                            IndexSpaceExpression *user_expr,
                            const FieldMask &user_mask,
                            std::set<ApEvent> &wait_on,
                            const bool trace_recording);
    public:
      static void handle_send_reduction_view(Runtime *runtime,
                                             Deserializer &derez);
      static void handle_defer_reduction_view(const void *args, Runtime *rt);
      static void create_remote_view(Runtime *runtime, DistributedID did, 
                                     PhysicalManager *manager,
                                     AddressSpaceID logical_owner); 
    public:
      FillView *const fill_view;
    protected:
      EventFieldUsers writing_users;
      EventFieldUsers reduction_users;
      EventFieldUsers reading_users;
    };

    /**
     * \class AllreduceView
     * This class represents a group of reduction instances that
     * all need to be reduced together to produce valid reduction data
     */
    class AllreduceView : public CollectiveView,
                          public LegionHeapify<AllreduceView> {
    public:
      static const AllocationType alloc_type = ALLREDUCE_VIEW_ALLOC;
    public:
      AllreduceView(Runtime *runtime, DistributedID did, DistributedID ctx_did,
                    const std::vector<IndividualView*> &views,
                    const std::vector<DistributedID> &instances,
                    bool register_now, CollectiveMapping *mapping,
                    ReductionOpID redop_id); 
      AllreduceView(const AllreduceView &rhs) = delete;
      virtual ~AllreduceView(void);
    public:
      AllreduceView& operator=(const AllreduceView &rhs) = delete;
    public: // From InstanceView
      virtual void send_view(AddressSpaceID target);
      virtual ReductionOpID get_redop(void) const { return redop; }
      virtual FillView* get_redop_fill_view(void) const { return fill_view; }
    public:
      void perform_collective_reduction(
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const DistributedID src_inst_did,
                                const UniqueInst &dst_inst,
                                const LgEvent dst_unique_event,
                                const PhysicalTraceInfo &trace_info,
                                const CollectiveKind collective_kind,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent result, AddressSpaceID origin);
      // Degenerate case
      ApEvent perform_hammer_reduction(
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const UniqueInst &dst_inst,
                                const LgEvent dst_unique_event,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                AddressSpaceID origin);
      void perform_collective_allreduce(ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                const uint64_t allreduce_tag);
      uint64_t generate_unique_allreduce_tag(void);
    protected:
      inline void set_redop(std::vector<CopySrcDstField> &fields) const
      {
#ifdef DEBUG_LEGION
        assert(redop > 0);
#endif
        for (std::vector<CopySrcDstField>::iterator it =
              fields.begin(); it != fields.end(); it++)
          it->set_redop(redop, true/*fold*/, true/*exclusive*/);
      }
      inline void clear_redop(std::vector<CopySrcDstField> &fields) const 
      {
        for (std::vector<CopySrcDstField>::iterator it =
              fields.begin(); it != fields.end(); it++)
          it->set_redop(0/*redop*/, false/*fold*/);
      }
      bool is_multi_instance(void);
      void perform_single_allreduce(const uint64_t allreduce_tag,
                                Operation *op, unsigned index,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events);
      void perform_multi_allreduce(const uint64_t allreduce_tag,
                                Operation *op, unsigned index,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events);
      ApEvent initialize_allreduce_with_reductions(
                                ApEvent precondition, PredEvent predicate_guard,
                                Operation *op, unsigned index,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &applied_events,
                                std::vector<ApEvent> &instance_events,
                    std::vector<std::vector<CopySrcDstField> > &local_fields,
                    std::vector<std::vector<Reservation> > &reservations);
      void complete_initialize_allreduce_with_reductions(
                                Operation *op, unsigned index,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                std::vector<ApEvent> &instance_events,
                    std::vector<std::vector<CopySrcDstField> > &local_fields,
                                std::vector<ApEvent> *reduced = NULL);
      void initialize_allreduce_without_reductions(
                                ApEvent precondition, PredEvent predicate_guard,
                                Operation *op, unsigned index,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                std::vector<ApEvent> &instance_events,
                    std::vector<std::vector<CopySrcDstField> > &local_fields,
                    std::vector<std::vector<Reservation> > &reservations);
      ApEvent finalize_allreduce_with_broadcasts(PredEvent predicate_guard,
                                Operation *op, unsigned index,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                std::vector<ApEvent> &instance_events,
              const std::vector<std::vector<CopySrcDstField> > &local_fields,
                                const unsigned final_index = 0);
      void complete_finalize_allreduce_with_broadcasts(
                                Operation *op, unsigned index,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                const std::vector<ApEvent> &instance_events,
                                std::vector<ApEvent> *broadcast = NULL,
                                const unsigned final_index = 0);
      void finalize_allreduce_without_broadcasts(PredEvent predicate_guard,
                                Operation *op, unsigned index,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                std::vector<ApEvent> &instance_events,
              const std::vector<std::vector<CopySrcDstField> > &local_fields,
                                const unsigned finalize_index = 0);
      void send_allreduce_stage(const uint64_t allreduce_tag, const int stage,
                                const int local_rank, ApEvent src_precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                const PhysicalTraceInfo &trace_info,
                                const std::vector<CopySrcDstField> &src_fields,
                                const unsigned src_index,
                                const AddressSpaceID *targets, size_t total,
                                std::vector<ApEvent> &read_events);
      void receive_allreduce_stage(const unsigned dst_index,
                                const uint64_t allreduce_tag,
                                const int stage, Operation *op,
                                ApEvent dst_precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &applied_events,
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                const int *expected_ranks, size_t total_ranks,
                                std::vector<ApEvent> &reduce_events);
      void process_distribute_allreduce(const uint64_t allreduce_tag,
                                const int src_rank, const int stage,
                                std::vector<CopySrcDstField> &src_fields,
                                const ApEvent src_precondition,
                                ApUserEvent src_postcondition,
                                ApBarrier src_barrier, ShardID bar_shard,
                                const UniqueInst &src_inst,
                                const LgEvent src_unique_event);
      void reduce_local(const PhysicalManager *dst_manager,
                        const unsigned dst_index, Operation *op,
                        const unsigned index, IndexSpaceExpression *copy_expr,
                        const FieldMask &copy_mask, ApEvent precondition,
                        PredEvent predicate_guard,
                        const std::vector<CopySrcDstField> &dst_fields,
                        const std::vector<Reservation> &dst_reservations,
                        const UniqueInst &dst_inst,
                        const PhysicalTraceInfo &trace_info,
                        const CollectiveKind collective_kind,
                        std::vector<ApEvent> &reduced_events,
                        std::set<RtEvent> &applied_events,
                        std::set<RtEvent> *recorded_events = NULL,
                        const bool prepare_allreduce = false,
                        std::vector<std::vector<
                              CopySrcDstField> > *src_fields = NULL);
    public:
      static void handle_send_allreduce_view(Runtime *runtime,
                                             Deserializer &derez);
      static void handle_distribute_reduction(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_hammer_reduction(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_allreduce(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
    public:
      const ReductionOpID redop;
      const ReductionOp *const reduction_op;
      FillView *const fill_view;
    protected:
      struct CopyKey {
      public:
        CopyKey(void) : tag(0), rank(0), stage(0) { }
        CopyKey(uint64_t t, int r, int s) : tag(t), rank(r), stage(s) { }
      public:
        inline bool operator==(const CopyKey &rhs) const
        { return (tag == rhs.tag) &&
            (rank == rhs.rank) && (stage == rhs.stage); }
        inline bool operator<(const CopyKey &rhs) const
        {
          if (tag < rhs.tag) return true;
          if (tag > rhs.tag) return false;
          if (rank < rhs.rank) return true;
          if (rank > rhs.rank) return false;
          return (stage < rhs.stage);
        }
      public:
        uint64_t tag;
        int rank, stage;
      };
      struct AllReduceCopy {
        std::vector<CopySrcDstField> src_fields;
        ApEvent src_precondition;
        ApUserEvent src_postcondition;
        ApBarrier barrier_postcondition;
        ShardID barrier_shard;
        UniqueInst src_inst;
        LgEvent src_unique_event;
      };
      std::map<CopyKey,AllReduceCopy> all_reduce_copies;
      struct AllReduceStage {
        unsigned dst_index;
        Operation *op;
        IndexSpaceExpression *copy_expression;
        FieldMask copy_mask;
        std::vector<CopySrcDstField> dst_fields;
        std::vector<Reservation> reservations;
        PhysicalTraceInfo *trace_info;
        ApEvent dst_precondition;
        PredEvent predicate_guard;
        std::vector<ApUserEvent> remaining_postconditions;
        std::set<RtEvent> applied_events;
        RtUserEvent applied_event;
      };
      LegionMap<std::pair<uint64_t,int>,AllReduceStage> remaining_stages;
    protected:
      std::atomic<uint64_t> unique_allreduce_tag;
      // A boolean flag that says whether this collective instance
      // has multiple instances on every node. This is primarily
      // useful for reduction instances where we want to pick an
      // algorithm for performing an in-place all-reduce
      std::atomic<bool> multi_instance;
      // Whether we've computed multi instance or not
      std::atomic<bool> evaluated_multi_instance;
    };

    /**
     * \class DeferredView
     * A DeferredView class is an abstract class for representing
     * lazy computation in an equivalence set. At the moment, the
     * types only allow deferred views to capture other kinds of
     * lazy evaluation. In particular this is either a fill view
     * or nested levels of predicated fill views. It could also
     * support other kinds of lazy evaluation as well. Importantly,
     * since it only captures lazy computation and not materialized
     * data there are no InstanceView managers captured in its
     * representations, which is an important invariant for the
     * equivalence sets. Think long and hard about what you're
     * doing if you ever decide that you want to break that 
     * invariant and capture the names of instance views inside
     * of a deferred view.
     */
    class DeferredView : public LogicalView {
    public:
      DeferredView(Runtime *runtime, DistributedID did,
                   bool register_now, CollectiveMapping *mapping = NULL);
      virtual ~DeferredView(void);
    public:
      virtual void send_view(AddressSpaceID target) = 0; 
    public:
      virtual void flatten(CopyFillAggregator &aggregator,
                           InstanceView *dst_view, const FieldMask &src_mask,
                           IndexSpaceExpression *expr, 
                           PredEvent pred_guard,
                           const PhysicalTraceInfo &trace_info,
                           EquivalenceSet *tracing_eq,
                           CopyAcrossHelper *helper) = 0;
    public:
      virtual void notify_valid(void);
      virtual bool notify_invalid(void);
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
      struct DeferIssueFill : public LgTaskArgs<DeferIssueFill> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_ISSUE_FILL_TASK_ID;
      public:
        DeferIssueFill(FillView *view, Operation *op, 
                       IndexSpaceExpression *fill_expr,
                       const PhysicalTraceInfo &trace_info,
                       const std::vector<CopySrcDstField> &dst_fields,
                       PhysicalManager *manager, ApEvent precondition,
                       PredEvent pred_guard, CollectiveKind collective);
      public:
        FillView *const view;
        Operation *const op;
        IndexSpaceExpression *const fill_expr;
        PhysicalTraceInfo *const trace_info;
        std::vector<CopySrcDstField> *const dst_fields;
        PhysicalManager *const manager;
        const ApEvent precondition;
        const PredEvent pred_guard;
        const CollectiveKind collective;
        const ApUserEvent done;
      };
    public:
      // Don't know the fill value yet, will be set later
      FillView(Runtime *runtime, DistributedID did,
#ifdef LEGION_SPY
               UniqueID fill_op_uid,
#endif
               bool register_now,
               CollectiveMapping *mapping = NULL);
      // Already know the fill value
      FillView(Runtime *runtime, DistributedID did,
#ifdef LEGION_SPY
               UniqueID fill_op_uid,
#endif
               const void *value, size_t size, bool register_now,
               CollectiveMapping *mapping = NULL);
      FillView(const FillView &rhs) = delete;
      virtual ~FillView(void);
    public:
      FillView& operator=(const FillView &rhs) = delete;
    public:
      virtual void notify_local(void) { /*nothing to do*/ }
      virtual void pack_valid_ref(void);
      virtual void unpack_valid_ref(void);
    public:
      virtual void send_view(AddressSpaceID target); 
    public:
      virtual void flatten(CopyFillAggregator &aggregator,
                           InstanceView *dst_view, const FieldMask &src_mask,
                           IndexSpaceExpression *expr, 
                           PredEvent pred_guard,
                           const PhysicalTraceInfo &trace_info,
                           EquivalenceSet *tracing_eq,
                           CopyAcrossHelper *helper); 
    public:
      bool matches(FillView *other);
      bool matches(const void *value, size_t size);
      bool set_value(const void *value, size_t size);
      ApEvent issue_fill(Operation *op, IndexSpaceExpression *fill_expr,
                         const PhysicalTraceInfo &trace_info,
                         const std::vector<CopySrcDstField> &dst_fields,
                         std::set<RtEvent> &applied_events,
                         PhysicalManager *manager,
                         ApEvent precondition, PredEvent pred_guard,
                         CollectiveKind collective = COLLECTIVE_NONE);
      static void handle_defer_issue_fill(const void *args);
    public:
      static void handle_send_fill_view(Runtime *runtime, Deserializer &derez);
      static void handle_send_fill_view_value(Runtime *runtime,
                                              Deserializer &derez);
#ifdef LEGION_SPY
    public:
      const UniqueID fill_op_uid;
#endif
    protected:
      std::atomic<void*> value;
      std::atomic<size_t> value_size;
      RtUserEvent value_ready;
      // To help with reference counting creation on collective fill views
      // we don't need to actually send the updates on our first active call
      // Note that this only works the fill view will eventually becomes
      // active on all the nodes of the collective mapping, which currently
      // it does, but that is a higher-level invariant maintained by the
      // fill view creation and not the fill view itself
      bool collective_first_active;
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
      PhiView(Runtime *runtime, DistributedID did,
              PredEvent true_guard, PredEvent false_guard,
              FieldMaskSet<DeferredView> &&true_views,
              FieldMaskSet<DeferredView> &&false_views,
              bool register_now = true);
      PhiView(const PhiView &rhs) = delete;
      virtual ~PhiView(void);
    public:
      PhiView& operator=(const PhiView &rhs) = delete;
    public:
      virtual void notify_local(void);
      virtual void pack_valid_ref(void);
      virtual void unpack_valid_ref(void);
    public:
      virtual void send_view(AddressSpaceID target);
    public:
      virtual void flatten(CopyFillAggregator &aggregator,
                           InstanceView *dst_view, const FieldMask &src_mask,
                           IndexSpaceExpression *expr, 
                           PredEvent pred_guard,
                           const PhysicalTraceInfo &trace_info,
                           EquivalenceSet *tracign_eq,
                           CopyAcrossHelper *helper);
    public:
      void add_initial_references(bool unpack_references);
      static void handle_send_phi_view(Runtime *runtime, Deserializer &derez);
      static void handle_deferred_view_registration(const void *args);
    public:
      const PredEvent true_guard;
      const PredEvent false_guard;
      const FieldMaskSet<DeferredView> true_views, false_views;
    };

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_materialized_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, MATERIALIZED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_reduction_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REDUCTION_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_replicated_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REPLICATED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_allreduce_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, ALLREDUCE_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_fill_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, FILL_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_phi_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, PHI_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_materialized_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC-1)) ==
                                          MATERIALIZED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_reduction_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC-1)) ==
                                              REDUCTION_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_replicated_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC-1)) ==
                                          REPLICATED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_allreduce_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC-1)) ==
                                              ALLREDUCE_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_individual_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return is_materialized_did(did) || is_reduction_did(did);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_collective_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return is_replicated_did(did) || is_allreduce_did(did);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_fill_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC-1)) ==
                                                    FILL_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_phi_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC-1)) ==
                                                    PHI_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      return (is_materialized_did(did) || is_reduction_did(did) ||
              is_replicated_did(did) || is_allreduce_did(did));
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
    inline bool LogicalView::is_replicated_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_replicated_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_allreduce_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_allreduce_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_individual_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_individual_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_collective_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_collective_did(did);
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
    inline bool LogicalView::is_reduction_kind(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_view() || is_allreduce_view();
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
    inline ReplicatedView* LogicalView::as_replicated_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_replicated_view());
#endif
      return static_cast<ReplicatedView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline AllreduceView* LogicalView::as_allreduce_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_allreduce_view());
#endif
      return static_cast<AllreduceView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline IndividualView* LogicalView::as_individual_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_individual_view());
#endif
      return static_cast<IndividualView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline CollectiveView* LogicalView::as_collective_view(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_collective_view());
#endif
      return static_cast<CollectiveView*>(const_cast<LogicalView*>(this));
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
                                                 bool &dominates) const
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
            forest->intersect_index_spaces(expr, user->expr);
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
                                                 const bool next_covers) const
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
          forest->intersect_index_spaces(expr, user->expr);
        if (overlap->is_empty())
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    inline void LogicalView::add_base_valid_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void LogicalView::add_nested_valid_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_nested_valid_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), 
                                    cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::remove_base_valid_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::remove_nested_valid_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_nested_valid_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = valid_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_VIEWS_H__
