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

#ifndef __LEGION_PHYSICAL_INSTANCE_H__
#define __LEGION_PHYSICAL_INSTANCE_H__

#include "legion/instances/instance.h"
#include "legion/managers/memory.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PhysicalManager
     * This is an abstract intermediate class for representing an allocation
     * of data; this includes both individual instances and collective instances
     */
    class PhysicalManager : public InstanceManager,
                            public Heapify<PhysicalManager, LONG_LIFETIME> {
    public:
      enum InstanceKind {
        // Normal Realm allocations
        INTERNAL_INSTANCE_KIND,
        // External allocations imported by attach operations
        EXTERNAL_ATTACHED_INSTANCE_KIND,
        // Instance not yet bound
        UNBOUND_INSTANCE_KIND,
      };
      enum GarbageCollectionState {
        VALID_GC_STATE,
        COLLECTABLE_GC_STATE,
        PENDING_COLLECTED_GC_STATE,
        COLLECTED_GC_STATE,
      };
    public:
      struct DeferPhysicalManagerArgs
        : public LgTaskArgs<DeferPhysicalManagerArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_PHYSICAL_MANAGER_TASK_ID;
      public:
        DeferPhysicalManagerArgs(void) = default;
        DeferPhysicalManagerArgs(
            DistributedID d, Memory m, PhysicalInstance i, size_t f,
            IndexSpaceExpression* lx, FieldSpace fs, RegionTreeID tid,
            LayoutConstraintID l, ApEvent use, LgEvent unique,
            InstanceKind kind, ReductionOpID redop, const void* piece_list,
            size_t piece_list_size, GarbageCollectionState state);
        void execute(void) const;
      public:
        DistributedID did;
        Memory mem;
        PhysicalInstance inst;
        size_t footprint;
        IndexSpaceExpression* local_expr;
        FieldSpace space;
        RegionTreeID tree_id;
        LayoutConstraintID layout_id;
        ApEvent use_event;
        LgEvent unique_event;
        InstanceKind kind;
        ReductionOpID redop;
        const void* piece_list;
        size_t piece_list_size;
        GarbageCollectionState state;
      };
    public:
      struct DeferDeletePhysicalManager
        : public LgTaskArgs<DeferDeletePhysicalManager> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_DEFER_DELETE_PHYSICAL_MANAGER_TASK_ID;
      public:
        DeferDeletePhysicalManager(void) = default;
        DeferDeletePhysicalManager(PhysicalManager* manager_);
        void execute(void) const;
      public:
        PhysicalManager* manager;
      };
      struct RemoteCreateViewArgs : public LgTaskArgs<RemoteCreateViewArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_REMOTE_VIEW_CREATION_TASK_ID;
      public:
        RemoteCreateViewArgs(void) = default;
        RemoteCreateViewArgs(
            PhysicalManager* man, InnerContext* ctx, AddressSpaceID log,
            CollectiveMapping* map, std::atomic<DistributedID>* tar,
            AddressSpaceID src, RtUserEvent done)
          : LgTaskArgs<RemoteCreateViewArgs>(false, false), manager(man),
            context(ctx), logical_owner(log), mapping(map), target(tar),
            source(src), done_event(done)
        { }
        void execute(void) const;
      public:
        PhysicalManager* manager;
        InnerContext* context;
        AddressSpaceID logical_owner;
        CollectiveMapping* mapping;
        std::atomic<DistributedID>* target;
        AddressSpaceID source;
        RtUserEvent done_event;
      };
    public:
      PhysicalManager(
          DistributedID did, MemoryManager* memory, PhysicalInstance inst,
          IndexSpaceExpression* instance_domain, const void* piece_list,
          size_t piece_list_size, FieldSpaceNode* node, RegionTreeID tree_id,
          LayoutDescription* desc, ReductionOpID redop, bool register_now,
          size_t footprint, ApEvent use_event, LgEvent unique_event,
          InstanceKind kind, const ReductionOp* op = nullptr,
          CollectiveMapping* collective_mapping = nullptr,
          ApEvent producer_event = ApEvent::NO_AP_EVENT,
          GarbageCollectionState init = COLLECTABLE_GC_STATE);
      PhysicalManager(const PhysicalManager& rhs) = delete;
      virtual ~PhysicalManager(void);
    public:
      PhysicalManager& operator=(const PhysicalManager& rhs) = delete;
    public:
      virtual PointerConstraint get_pointer_constraint(void) const override;
    public:
      void log_instance_creation(
          UniqueID creator_id, Processor proc,
          const std::vector<LogicalRegion>& regions) const;
    public:
      ApEvent get_use_event(ApEvent e = ApEvent::NO_AP_EVENT) const;
      inline LgEvent get_unique_event(void) const { return unique_event; }
      PhysicalInstance get_instance(void) const;
      inline Memory get_memory(void) const { return memory_manager->memory; }
      void compute_copy_offsets(
          const FieldMask& copy_mask, std::vector<CopySrcDstField>& fields);
    public:
      inline bool is_unbound(void) const
      {
        return (kind.load() == UNBOUND_INSTANCE_KIND);
      }
      inline void add_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, int cnt = 1);
      inline bool acquire_instance(ReferenceSource source);
      inline bool acquire_instance(DistributedID source);
      inline bool remove_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, int cnt = 1);
    public:
      void pack_valid_ref(Serializer& rez);
      void pack_valid_ref(LamportClock& lamport_clock);
      void unpack_valid_ref(Deserializer& derez);
      void unpack_valid_ref(LamportClock lamport_clock);
    protected:
      friend class GarbageCollectionAcquireResponse;
      // Internal valid reference counting
      void add_valid_reference(int cnt, bool need_check = true);
#ifdef LEGION_DEBUG_GC
      void add_base_valid_ref_internal(ReferenceSource source, int cnt);
      void add_nested_valid_ref_internal(DistributedID source, int cnt);
      bool remove_base_valid_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_valid_ref_internal(DistributedID source, int cnt);
      template<typename T>
      bool acquire_internal(T source, std::map<T, int>& valid_references);
#else
      bool acquire_internal(void);
      bool remove_valid_reference(int cnt);
#endif
      void notify_valid(bool need_check);
      bool notify_invalid(AutoLock& i_lock);
    public:
      virtual void send_manager(AddressSpaceID target);
    public:
      virtual void notify_local(void) override;
    public:
      bool is_collected(void) const;
      bool can_collect(bool& already_collected) const;
      bool acquire_collect(
          std::set<ApEvent>& gc_events, uint64_t& sent_valid,
          uint64_t& received_valid, LamportClock lamport_clock,
          LamportClock& collect_clock);
      bool collect(
          RtEvent& collected, PhysicalInstance* hole = nullptr,
          AutoLock* i_lock = nullptr);
      void notify_remote_deletion(void);
      RtEvent set_garbage_collection_priority(
          MapperID mapper_id, Processor p, GCPriority priority);
      RtEvent broadcast_garbage_collection_priority_update(GCPriority priority);
      RtEvent perform_deletion(
          AddressSpaceID source, PhysicalInstance* hole = nullptr,
          AutoLock* i_lock = nullptr);
      void force_deletion(void);
      RtEvent attach_external_instance(void);
      void detach_external_instance(void);
      bool has_visible_from(const std::set<Memory>& memories) const;
      uintptr_t get_instance_pointer(void) const;
      size_t get_instance_size(void) const;
    public:
      bool update_physical_instance(
          PhysicalInstance new_instance, RtEvent ready, size_t new_footprint);
      void broadcast_manager_update(void);
      void pack_fields(
          Serializer& rez, const std::vector<CopySrcDstField>& fields) const;
      void initialize_across_helper(
          CopyAcrossHelper* across_helper, const FieldMask& mask,
          const std::vector<unsigned>& src_indexes,
          const std::vector<unsigned>& dst_indexes);
    public:
      // Methods for creating/finding/destroying logical top views
      IndividualView* find_or_create_instance_top_view(
          InnerContext* context, AddressSpaceID logical_owner,
          CollectiveMapping* mapping);
      IndividualView* construct_top_view(
          AddressSpaceID logical_owner, DistributedID did, InnerContext* ctx,
          CollectiveMapping* mapping);
      bool register_deletion_subscriber(
          InstanceDeletionSubscriber* subscriber,
          bool allow_duplicates = false);
      void unregister_deletion_subscriber(
          InstanceDeletionSubscriber* subscriber);
      void unregister_active_context(InnerContext* context);
    public:
      PieceIteratorImpl* create_piece_iterator(IndexSpaceNode* privilege_node);
      void record_instance_user(ApEvent term_event, std::set<RtEvent>& applied);
      void process_remote_reference_mismatch(
          uint64_t sent, uint64_t received, LamportClock lamport_clock);
      void find_shutdown_preconditions(std::set<ApEvent>& preconditions);
    public:
      bool meets_regions(
          const std::vector<LogicalRegion>& regions,
          bool tight_region_bounds = false,
          const Domain* padding_delta = nullptr) const;
      bool meets_expression(
          IndexSpaceExpression* expr, bool tight_bounds = false,
          const Domain* padding_delta = nullptr) const;
    public:
      void find_padded_reservations(
          const FieldMask& mask, Operation* op, unsigned index);
      void find_field_reservations(
          const FieldMask& mask, std::vector<Reservation>& results);
      void update_field_reservations(
          const FieldMask& mask, const std::vector<Reservation>& reservations);
    protected:
      void pack_garbage_collection_state(
          Serializer& rez, AddressSpaceID target, bool need_lock);
    public:
      static void create_remote_manager(
          DistributedID did, Memory mem, PhysicalInstance inst,
          size_t inst_footprint, IndexSpaceExpression* inst_domain,
          const void* piece_list, size_t piece_list_size,
          FieldSpaceNode* space_node, RegionTreeID tree_id,
          LayoutConstraints* constraints, ApEvent use_event, LgEvent unique,
          InstanceKind kind, ReductionOpID redop, GarbageCollectionState state);
    public:
      ApEvent fetch_metadata(ApEvent use_event) const;
      static void process_top_view_request(
          PhysicalManager* manager, InnerContext* context,
          AddressSpaceID logical_owner, CollectiveMapping* mapping,
          std::atomic<DistributedID>* target, AddressSpaceID source,
          RtUserEvent done_event);
    public:
      MemoryManager* const memory_manager;
      // Unique identifier event that is common across nodes
      // Note this is just an LgEvent which suggests you shouldn't be using
      // it for anything other than logging
      const LgEvent unique_event;
      size_t instance_footprint;
      const ReductionOp* reduction_op;
      const ReductionOpID redop;
      const void* const piece_list;
      const size_t piece_list_size;
    public:
      PhysicalInstance instance;
      // Event that needs to trigger before we can start using
      // this physical instance.
      ApUserEvent use_event;
      // Event that signifies if the instance name is available
      RtUserEvent instance_ready;
      std::atomic<InstanceKind> kind;
      // Completion event of the task that sets a realm instance
      // to this manager. Valid only when the kind is UNBOUND
      // initially, otherwise NO_AP_EVENT.
      const ApEvent producer_event;
    protected:
      mutable LocalLock inst_lock;
      std::set<InstanceDeletionSubscriber*> subscribers;
      typedef std::pair<IndividualView*, unsigned> ViewEntry;
      std::map<DistributedID, ViewEntry> context_views;
      std::map<DistributedID, RtUserEvent> pending_views;
    protected:
      // Stuff for garbage collection
      std::atomic<GarbageCollectionState> gc_state;
      unsigned pending_changes;
      std::atomic<unsigned> failed_collection_count;
      RtEvent collection_ready;
      // Garbage collection priorities
      GCPriority min_gc_priority;
      RtEvent priority_update_done;
      std::map<std::pair<MapperID, Processor>, GCPriority> mapper_gc_priorities;
    protected:
      // Events for application users of this instance that must trigger
      // before we could possibly do a deferred deletion
      std::set<ApEvent> gc_events;
      // The number of events added since the last time we pruned the list
      unsigned added_gc_events;
    private:
#ifdef LEGION_DEBUG_GC
      int valid_references;
#else
      std::atomic<int> valid_references;
#endif
      uint64_t sent_valid_references, received_valid_references;
      // Lamport clock for the valid-reference instance-collection protocol
      // (analogous to DistributedCollectable::downgrade_lamport_clock, but for
      // this separate protocol and guarded by inst_lock). Stamped onto packed
      // valid references and merged on unpack so that a valid reference packed
      // after a node has committed its contribution to a collection round can
      // be detected and force the round to fail and retry.
      LamportClock collect_lamport_clock;
      // Snapshot of collect_lamport_clock for the in-progress collection round
      // (the round's cutoff). The round is only allowed to succeed if no node
      // observed a packed valid reference newer than this snapshot, i.e. if
      // collect_lamport_clock <= pending_collect_lamport_clock.
      LamportClock pending_collect_lamport_clock;
      // Whether the next packed valid reference must bump the clock (armed once
      // we accumulate/freeze our counts for a collection round).
      bool bump_collect_lamport_clock;
      std::map<unsigned, Reservation>* padded_reservations;
#ifdef LEGION_DEBUG_GC
    private:
      std::map<ReferenceSource, int> detailed_base_valid_references;
      std::map<DistributedID, int> detailed_nested_valid_references;
#endif
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/instances/physical.inl"

#endif  // __LEGION_PHYSICAL_INSTANCE_H__
