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

#ifndef __LEGION_INSTANCES_H__
#define __LEGION_INSTANCES_H__

#include "legion/runtime.h"
#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LayoutDescription
     * This class is for deduplicating the meta-data
     * associated with describing the layouts of physical
     * instances. Often times this meta data is rather 
     * large (~100K) and since we routinely create up
     * to 100K instances, it is important to deduplicate
     * the data.  Since many instances will have the
     * same layout then they can all share the same
     * description object.
     */
    class LayoutDescription : public Collectable,
                              public LegionHeapify<LayoutDescription> {
    public:
      LayoutDescription(FieldSpaceNode *owner,
                        const FieldMask &mask,
                        const unsigned total_dims,
                        LayoutConstraints *constraints,
                        const std::vector<unsigned> &mask_index_map,
                        const std::vector<FieldID> &fids,
                        const std::vector<size_t> &field_sizes,
                        const std::vector<CustomSerdezID> &serdez);
      // Used only by the virtual manager
      LayoutDescription(const FieldMask &mask, LayoutConstraints *constraints);
      LayoutDescription(const LayoutDescription &rhs);
      ~LayoutDescription(void);
    public:
      LayoutDescription& operator=(const LayoutDescription &rhs);
    public:
      void log_instance_layout(LgEvent inst_event) const;
    public:
      void compute_copy_offsets(const FieldMask &copy_mask, 
                                const PhysicalInstance instance,  
                                std::vector<CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                const PhysicalInstance instance,
                                std::vector<CopySrcDstField> &fields);
    public:
      void get_fields(std::set<FieldID> &fields) const;
      bool has_field(FieldID fid) const;
      void has_fields(std::map<FieldID,bool> &fields) const;
      void remove_space_fields(std::set<FieldID> &fields) const;
    public:
      const CopySrcDstField& find_field_info(FieldID fid) const;
      size_t get_total_field_size(void) const;
      void get_fields(std::vector<FieldID>& fields) const;
      void compute_destroyed_fields(
          std::vector<PhysicalInstance::DestroyedField> &serdez_fields) const;
    public:
      bool match_layout(const LayoutConstraintSet &constraints,
                        unsigned num_dims) const;
      bool match_layout(const LayoutDescription *layout,
                        unsigned num_dims) const;
    public:
      void pack_layout_description(Serializer &rez, AddressSpaceID target);
      static LayoutDescription* handle_unpack_layout_description(
                            LayoutConstraints *constraints,
                            FieldSpaceNode *field_space, size_t total_dims);
    public:
      const FieldMask allocated_fields;
      LayoutConstraints *const constraints;
      FieldSpaceNode *const owner;
      const unsigned total_dims;
    protected:
      // In order by index of bit mask
      std::vector<CopySrcDstField> field_infos;
      // A mapping from FieldIDs to indexes into our field_infos
      std::map<FieldID,unsigned/*index*/> field_indexes;
    protected:
      mutable LocalLock layout_lock; 
      std::map<LEGION_FIELD_MASK_FIELD_TYPE,
               LegionList<std::pair<FieldMask,FieldMask> > > comp_cache;
    }; 

    /**
     * \class CollectiveMapping
     * A collective mapping is an ordering of unique address spaces
     * and can be used to construct broadcast and reduction trees.
     * This is especialy useful for collective instances and for
     * parts of control replication.
     */
    class CollectiveMapping : public Collectable {
    public:
      CollectiveMapping(const std::vector<AddressSpaceID> &spaces,size_t radix);
      CollectiveMapping(const ShardMapping &shard_mapping, size_t radix);
      CollectiveMapping(Deserializer &derez, size_t total_spaces);
      CollectiveMapping(const CollectiveMapping &rhs);
    public:
      inline AddressSpaceID operator[](unsigned idx) const
#ifdef DEBUG_LEGION
        { assert(idx < size()); return unique_sorted_spaces.get_index(idx); }
#else
        { return unique_sorted_spaces.get_index(idx); }
#endif
      inline unsigned find_index(const AddressSpaceID space) const
        { return unique_sorted_spaces.find_index(space); }
      inline const NodeSet& get_unique_spaces(void) const 
        { return unique_sorted_spaces; }
      inline size_t size(void) const { return total_spaces; }
      inline AddressSpaceID get_origin(void) const 
#ifdef DEBUG_LEGION
        { assert(size() > 0); return unique_sorted_spaces.find_first_set(); }
#else
        { return unique_sorted_spaces.find_first_set(); }
#endif
      bool operator==(const CollectiveMapping &rhs) const;
      bool operator!=(const CollectiveMapping &rhs) const;
    public:
      AddressSpaceID get_parent(const AddressSpaceID origin, 
                                const AddressSpaceID local) const;
      size_t count_children(const AddressSpaceID origin,
                            const AddressSpaceID local) const;
      void get_children(const AddressSpaceID origin, const AddressSpaceID local,
                        std::vector<AddressSpaceID> &children) const;
      AddressSpaceID find_nearest(AddressSpaceID start) const;
      inline bool contains(const AddressSpaceID space) const
        { return unique_sorted_spaces.contains(space); }
      bool contains(const CollectiveMapping &rhs) const;
      CollectiveMapping* clone_with(AddressSpace space) const;
      void pack(Serializer &rez) const;
    protected:
      unsigned convert_to_offset(unsigned index, unsigned origin) const;
      unsigned convert_to_index(unsigned offset, unsigned origin) const;
    protected:
      NodeSet unique_sorted_spaces;
      size_t total_spaces;
      size_t radix;
    };

    /**
     * \class InstanceManager
     * This is the abstract base class for all instances of a physical
     * resource manager for memory.
     */
    class InstanceManager : public DistributedCollectable {
    public:
      enum {
        EXTERNAL_CODE = 0x20,
        REDUCTION_CODE = 0x40,
      };
    public:
      InstanceManager(RegionTreeForest *forest,
                      DistributedID did, LayoutDescription *layout,
                      FieldSpaceNode *node, IndexSpaceExpression *domain,
                      RegionTreeID tree_id, bool register_now,
                      CollectiveMapping *mapping = NULL);
      virtual ~InstanceManager(void);
    public:
      virtual PointerConstraint get_pointer_constraint(void) const = 0;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const = 0;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const = 0; 
    public:
      inline bool is_reduction_manager(void) const;
      inline bool is_physical_manager(void) const;
      inline bool is_virtual_manager(void) const;
      inline bool is_external_instance(void) const;
      inline PhysicalManager* as_physical_manager(void) const;
      inline VirtualManager* as_virtual_manager(void) const;
    public:
      static inline DistributedID encode_instance_did(DistributedID did,
                                          bool external, bool reduction);
      static inline bool is_physical_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_external_did(DistributedID did);
    public:
      // Interface to the mapper for layouts
      inline void get_fields(std::set<FieldID> &fields) const
        { if (layout != NULL) layout->get_fields(fields); }
      inline bool has_field(FieldID fid) const
        { if (layout != NULL) return layout->has_field(fid); return false; }
      inline void has_fields(std::map<FieldID,bool> &fields) const
        { if (layout != NULL) layout->has_fields(fields); 
          else for (std::map<FieldID,bool>::iterator it = fields.begin();
                    it != fields.end(); it++) it->second = false; } 
      inline void remove_space_fields(std::set<FieldID> &fields) const
        { if (layout != NULL) layout->remove_space_fields(fields);
          else fields.clear(); } 
    public:
      bool entails(LayoutConstraints *constraints,
                   const LayoutConstraint **failed_constraint) const;
      bool entails(const LayoutConstraintSet &constraints, 
                   const LayoutConstraint **failed_constraint) const;
      bool conflicts(LayoutConstraints *constraints,
                     const LayoutConstraint **conflict_constraint) const;
      bool conflicts(const LayoutConstraintSet &constraints,
                     const LayoutConstraint **conflict_constraint) const;
    public:
      RegionTreeForest *const context;
      LayoutDescription *const layout;
      FieldSpaceNode *const field_space_node;
      IndexSpaceExpression *instance_domain;
      const RegionTreeID tree_id;
    };

    /**
     * A small interface for subscribing to notifications for
     * when an instance is deleted
     */
    class InstanceDeletionSubscriber {
    public:
      virtual ~InstanceDeletionSubscriber(void) { }
      virtual void notify_instance_deletion(PhysicalManager *manager) = 0;
      virtual void add_subscriber_reference(PhysicalManager *manager) = 0;
      virtual bool remove_subscriber_reference(PhysicalManager *manager) = 0;
    };

    /**
     * \class PhysicalManager 
     * This is an abstract intermediate class for representing an allocation
     * of data; this includes both individual instances and collective instances
     */
    class PhysicalManager : public InstanceManager, 
                            public LegionHeapify<PhysicalManager> {
    public:
      static const AllocationType alloc_type = PHYSICAL_MANAGER_ALLOC;
    public:
      enum InstanceKind {
        // Normal Realm allocations
        INTERNAL_INSTANCE_KIND,
        // External allocations imported by attach operations
        EXTERNAL_ATTACHED_INSTANCE_KIND,
        // Allocations drawn from the eager pool
        EAGER_INSTANCE_KIND,
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
      struct DeferPhysicalManagerArgs : 
        public LgTaskArgs<DeferPhysicalManagerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PHYSICAL_MANAGER_TASK_ID;
      public:
        DeferPhysicalManagerArgs(DistributedID d,
            Memory m, PhysicalInstance i, size_t f, IndexSpaceExpression *lx,
            const PendingRemoteExpression &pending, FieldSpace h, 
            RegionTreeID tid, LayoutConstraintID l, ApEvent use,
            InstanceKind kind, ReductionOpID redop, const void *piece_list,
            size_t piece_list_size, GarbageCollectionState state);
      public:
        const DistributedID did;
        const Memory mem;
        const PhysicalInstance inst;
        const size_t footprint;
        const PendingRemoteExpression pending;
        IndexSpaceExpression *local_expr;
        const FieldSpace handle;
        const RegionTreeID tree_id;
        const LayoutConstraintID layout_id;
        const ApEvent use_event;
        const InstanceKind kind;
        const ReductionOpID redop;
        const void *const piece_list;
        const size_t piece_list_size;
        const GarbageCollectionState state;
      };
    public:
      struct DeferDeletePhysicalManager :
        public LgTaskArgs<DeferDeletePhysicalManager> {
      public:
        static const LgTaskID TASK_ID =
          LG_DEFER_DELETE_PHYSICAL_MANAGER_TASK_ID;
      public:
        DeferDeletePhysicalManager(PhysicalManager *manager_);
      public:
        PhysicalManager *manager;
        const RtUserEvent done;
      };
      struct RemoteCreateViewArgs : public LgTaskArgs<RemoteCreateViewArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REMOTE_VIEW_CREATION_TASK_ID;
      public:
        RemoteCreateViewArgs(PhysicalManager *man, InnerContext *ctx, 
                             AddressSpaceID log, CollectiveMapping *map,
                             std::atomic<DistributedID> *tar, 
                             AddressSpaceID src, RtUserEvent done)
          : LgTaskArgs<RemoteCreateViewArgs>(implicit_provenance),
            manager(man), context(ctx), logical_owner(log), mapping(map),
            target(tar), source(src), done_event(done) { }
      public:
        PhysicalManager *const manager;
        InnerContext *const context;
        const AddressSpaceID logical_owner;
        CollectiveMapping *const mapping;
        std::atomic<DistributedID> *const target;
        const AddressSpaceID source;
        const RtUserEvent done_event;
      }; 
    public:
      struct BroadcastFunctor {
        BroadcastFunctor(Runtime *rt, Serializer &r) : runtime(rt), rez(r) { }
        inline void apply(AddressSpaceID target)
          { runtime->send_manager_update(target, rez); }
        Runtime *runtime;
        Serializer &rez;
      };
    public:
      PhysicalManager(RegionTreeForest *ctx, DistributedID did,
                      MemoryManager *memory, PhysicalInstance inst, 
                      IndexSpaceExpression *instance_domain,
                      const void *piece_list, size_t piece_list_size,
                      FieldSpaceNode *node, RegionTreeID tree_id,
                      LayoutDescription *desc, ReductionOpID redop, 
                      bool register_now, size_t footprint,
                      ApEvent use_event, InstanceKind kind,
                      const ReductionOp *op = NULL,
                      CollectiveMapping *collective_mapping = NULL,
                      ApEvent producer_event = ApEvent::NO_AP_EVENT);
      PhysicalManager(const PhysicalManager &rhs) = delete;
      virtual ~PhysicalManager(void);
    public:
      PhysicalManager& operator=(const PhysicalManager &rhs) = delete;
    public:
      virtual PointerConstraint get_pointer_constraint(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      void log_instance_creation(UniqueID creator_id, Processor proc,
                                 const std::vector<LogicalRegion> &regions) const;
    public: 
      ApEvent get_use_event(ApEvent e = ApEvent::NO_AP_EVENT) const;
      inline LgEvent get_unique_event(void) const { return unique_event; }
      PhysicalInstance get_instance(void) const { return instance; }
      inline Memory get_memory(void) const { return memory_manager->memory; }
      void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &fields);
    public:
      inline void add_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, int cnt = 1);
      inline bool acquire_instance(ReferenceSource source);
      inline bool acquire_instance(DistributedID source);
      inline bool remove_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, int cnt = 1);
    public:
      void pack_valid_ref(void);
      void unpack_valid_ref(void);
    protected:
      // Internal valid reference counting 
      void add_valid_reference(int cnt, bool need_check = true);
#ifdef DEBUG_LEGION_GC
      void add_base_valid_ref_internal(ReferenceSource source, int cnt); 
      void add_nested_valid_ref_internal(DistributedID source, int cnt);
      bool remove_base_valid_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_valid_ref_internal(DistributedID source, int cnt);
      template<typename T>
      bool acquire_internal(T source, std::map<T,int> &valid_references);
#else
      bool acquire_internal(void);
      bool remove_valid_reference(int cnt);
#endif
      void notify_valid(bool need_check);
      bool notify_invalid(void);
    public:
      virtual void send_manager(AddressSpaceID target);
      static void handle_manager_request(Deserializer &derez, 
                          Runtime *runtime, AddressSpaceID source);
    public:
      virtual void notify_local(void);
    public:
      bool can_collect(bool &already_collected) const;
      bool acquire_collect(std::set<ApEvent> &gc_events);
      bool collect(RtEvent &collected);
      void notify_remote_deletion(void);
      RtEvent set_garbage_collection_priority(MapperID mapper_id, Processor p, 
                                  AddressSpaceID source, GCPriority priority);
      RtEvent perform_deletion(AddressSpaceID source, AutoLock *i_lock = NULL);
      void force_deletion(void);
      RtEvent update_garbage_collection_priority(AddressSpaceID source,
                                                 GCPriority priority);
      RtEvent attach_external_instance(void);
      RtEvent detach_external_instance(void);
      bool has_visible_from(const std::set<Memory> &memories) const;
      uintptr_t get_instance_pointer(void) const; 
      size_t get_instance_size(void) const;
      void update_instance_footprint(size_t footprint)
        { instance_footprint = footprint; }
    public:
      inline bool is_unbound() const 
        { return kind == UNBOUND_INSTANCE_KIND; }
      bool update_physical_instance(PhysicalInstance new_instance,
                                    InstanceKind new_kind,
                                    size_t new_footprint,
                                    uintptr_t new_pointer = 0);
      void broadcast_manager_update(void);
      static void handle_send_manager_update(Runtime *runtime,
                                             AddressSpaceID source,
                                             Deserializer &derez);
      void pack_fields(Serializer &rez, 
                       const std::vector<CopySrcDstField> &fields) const;
      void initialize_across_helper(CopyAcrossHelper *across_helper,
                                    const FieldMask &mask,
                                    const std::vector<unsigned> &src_indexes,
                                    const std::vector<unsigned> &dst_indexes);
    public:
      // Methods for creating/finding/destroying logical top views
      IndividualView* find_or_create_instance_top_view(InnerContext *context,
          AddressSpaceID logical_owner, CollectiveMapping *mapping);
      IndividualView* construct_top_view(AddressSpaceID logical_owner,
                                         DistributedID did, InnerContext *ctx,
                                         CollectiveMapping *mapping);
      void register_deletion_subscriber(InstanceDeletionSubscriber *subscriber);
      void unregister_deletion_subscriber(InstanceDeletionSubscriber *subscrib);
      void unregister_active_context(InnerContext *context); 
    public:
      PieceIteratorImpl* create_piece_iterator(IndexSpaceNode *privilege_node);
      void record_instance_user(ApEvent term_event, std::set<RtEvent> &applied);
      void find_shutdown_preconditions(std::set<ApEvent> &preconditions);
    public:
      bool meets_regions(const std::vector<LogicalRegion> &regions,
                         bool tight_region_bounds = false) const;
      bool meets_expression(IndexSpaceExpression *expr, 
                            bool tight_bounds = false) const;
    protected:
      void pack_garbage_collection_state(Serializer &rez,
                                         AddressSpaceID target, bool need_lock);
      void initialize_remote_gc_state(GarbageCollectionState state);
    public:
      static void handle_send_manager(Runtime *runtime, 
                                      AddressSpaceID source,
                                      Deserializer &derez); 
      static void handle_defer_manager(const void *args, Runtime *runtime);
      static void handle_defer_perform_deletion(const void *args,
                                                Runtime *runtime);
      static void create_remote_manager(Runtime *runtime, DistributedID did,
          Memory mem, PhysicalInstance inst,
          size_t inst_footprint, IndexSpaceExpression *inst_domain,
          const void *piece_list, size_t piece_list_size,
          FieldSpaceNode *space_node, RegionTreeID tree_id,
          LayoutConstraints *constraints, ApEvent use_event,
          InstanceKind kind, ReductionOpID redop, GarbageCollectionState state);
    public: 
      static ApEvent fetch_metadata(PhysicalInstance inst, ApEvent use_event);
      static void process_top_view_request(PhysicalManager *manager,
          InnerContext *context, AddressSpaceID logical_owner,
          CollectiveMapping *mapping, std::atomic<DistributedID> *target,
          AddressSpaceID source, RtUserEvent done_event, Runtime *runtime);
      static void handle_top_view_request(Deserializer &derez, Runtime *runtime,
                                          AddressSpaceID source);
      static void handle_top_view_response(Deserializer &derez);
      static void handle_top_view_creation(const void *args, Runtime *runtime);
      static void handle_acquire_request(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_acquire_response(Deserializer &derez, 
          AddressSpaceID source);
      static void handle_garbage_collection_request(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_garbage_collection_response(Deserializer &derez);
      static void handle_garbage_collection_acquire(Runtime *runtime,
          Deserializer &derez);
      static void handle_garbage_collection_failed(Deserializer &derez);
      static void handle_garbage_collection_notify(Runtime *runtime,
          Deserializer &derez);
      static void handle_garbage_collection_priority_update(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_garbage_collection_debug_request(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_garbage_collection_debug_response(Deserializer &derez); 
      static void handle_record_event(Runtime *runtime, Deserializer &derez);
    public:
      MemoryManager *const memory_manager;
      // Unique identifier event that is common across nodes
      // Note this is just an LgEvent which suggests you shouldn't be using
      // it for anything other than logging
      const LgEvent unique_event;
      size_t instance_footprint;
      const ReductionOp *reduction_op;
      const ReductionOpID redop; 
      const void *const piece_list;
      const size_t piece_list_size;
    public:
      PhysicalInstance instance;
      // Event that needs to trigger before we can start using
      // this physical instance.
      ApUserEvent use_event;
      // Event that signifies if the instance name is available
      RtUserEvent instance_ready;
      InstanceKind kind;
      // Keep the pointer for owned external instances
      uintptr_t external_pointer;
      // Completion event of the task that sets a realm instance
      // to this manager. Valid only when the kind is UNBOUND
      // initially, otherwise NO_AP_EVENT.
      const ApEvent producer_event;
    protected:
      mutable LocalLock inst_lock;
      std::set<InstanceDeletionSubscriber*> subscribers;
      typedef std::pair<IndividualView*,unsigned> ViewEntry;
      std::map<DistributedID,ViewEntry> context_views;
      std::map<DistributedID,RtUserEvent> pending_views;
    protected:
      // Stuff for garbage collection
      GarbageCollectionState gc_state; 
      unsigned pending_changes;
      std::atomic<unsigned> failed_collection_count;
      RtEvent collection_ready;
      // Garbage collection priorities
      GCPriority min_gc_priority;
      RtEvent priority_update_done;
      std::map<std::pair<MapperID,Processor>,GCPriority> mapper_gc_priorities;
    protected:
      // Events for application users of this instance that must trigger
      // before we could possibly do a deferred deletion
      std::set<ApEvent> gc_events;
      // The number of events added since the last time we pruned the list
      unsigned added_gc_events;
    private:
#ifdef DEBUG_LEGION_GC
      int valid_references;
#else
      std::atomic<int> valid_references;
#endif
      uint64_t sent_valid_references, received_valid_references;
#ifdef DEBUG_LEGION_GC
    private:
      std::map<ReferenceSource,int> detailed_base_valid_references;
      std::map<DistributedID,int> detailed_nested_valid_references;
#endif
    };

    /**
     * \class CopyAcrossHelper
     * A small helper class for performing copies between regions
     * from diferrent region trees
     */
    class CopyAcrossHelper {
    public:
      CopyAcrossHelper(const FieldMask &full,
                       const std::vector<unsigned> &src,
                       const std::vector<unsigned> &dst)
        : full_mask(full), src_indexes(src), dst_indexes(dst) { }
    public:
      const FieldMask &full_mask;
      const std::vector<unsigned> &src_indexes;
      const std::vector<unsigned> &dst_indexes;
      std::map<unsigned,unsigned> forward_map;
      std::map<unsigned,unsigned> backward_map;
    public:
      void compute_across_offsets(const FieldMask &src_mask,
                   std::vector<CopySrcDstField> &dst_fields);
      FieldMask convert_src_to_dst(const FieldMask &src_mask);
      FieldMask convert_dst_to_src(const FieldMask &dst_mask);
    public:
      unsigned convert_src_to_dst(unsigned index);
      unsigned convert_dst_to_src(unsigned index);
    public:
      std::vector<CopySrcDstField> offsets; 
      LegionDeque<std::pair<FieldMask,FieldMask> > compressed_cache;
    };

#ifdef NO_EXPLICIT_COLLECTIVES
    /**
     * \class IndividualManager 
     * The individual manager class represents a single physical instance
     * that lives in memory in a given location in the system. This is the
     * most common kind of instance that gets made.
     */
    class IndividualManager : public PhysicalManager,
                              public LegionHeapify<IndividualManager> {
    public:
      static const AllocationType alloc_type = INDIVIDUAL_INST_MANAGER_ALLOC;
    public:
      struct DeferIndividualManagerArgs : 
        public LgTaskArgs<DeferIndividualManagerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_INDIVIDUAL_MANAGER_TASK_ID;
      public:
        DeferIndividualManagerArgs(DistributedID d, AddressSpaceID own, 
            Memory m, PhysicalInstance i, size_t f, IndexSpaceExpression *lx,
            const PendingRemoteExpression &pending, FieldSpace h, 
            RegionTreeID tid, LayoutConstraintID l, ApEvent use,
            InstanceKind kind, ReductionOpID redop, const void *piece_list,
            size_t piece_list_size, GarbageCollectionState state);
      public:
        const DistributedID did;
        const AddressSpaceID owner;
        const Memory mem;
        const PhysicalInstance inst;
        const size_t footprint;
        const PendingRemoteExpression pending;
        IndexSpaceExpression *local_expr;
        const FieldSpace handle;
        const RegionTreeID tree_id;
        const LayoutConstraintID layout_id;
        const ApEvent use_event;
        const InstanceKind kind;
        const ReductionOpID redop;
        const void *const piece_list;
        const size_t piece_list_size;
        const GarbageCollectionState state;
      };
    public:
      struct DeferDeleteIndividualManager :
        public LgTaskArgs<DeferDeleteIndividualManager> {
      public:
        static const LgTaskID TASK_ID =
          LG_DEFER_DELETE_INDIVIDUAL_MANAGER_TASK_ID;
      public:
        DeferDeleteIndividualManager(IndividualManager *manager_);
      public:
        IndividualManager *manager;
        const RtUserEvent done;
      };
    private:
      
    public:
      IndividualManager(RegionTreeForest *ctx, DistributedID did,
                        AddressSpaceID owner_space,
                        MemoryManager *memory, PhysicalInstance inst, 
                        IndexSpaceExpression *instance_domain,
                        const void *piece_list, size_t piece_list_size,
                        FieldSpaceNode *node, RegionTreeID tree_id,
                        LayoutDescription *desc, ReductionOpID redop, 
                        bool register_now, size_t footprint,
                        ApEvent use_event, InstanceKind kind,
                        const ReductionOp *op = NULL,
                        CollectiveMapping *collective_mapping = NULL,
                        ApEvent producer_event = ApEvent::NO_AP_EVENT);
      IndividualManager(const IndividualManager &rhs) = delete;
      virtual ~IndividualManager(void);
    public:
      IndividualManager& operator=(const IndividualManager &rhs) = delete;
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      virtual PhysicalInstance get_instance(bool from_mapper = false) const 
                                                   { return instance; }
      virtual PointerConstraint get_pointer_constraint(void) const;
      virtual Memory get_memory(bool from_mapper = false) const
        { return memory_manager->memory; }
    public:
      virtual ApEvent fill_from(FillView *fill_view, InstanceView *dst_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool fill_restricted,
                                const bool need_valid_return);
      virtual ApEvent copy_from(InstanceView *src_view, InstanceView *dst_view,
                                PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const DomainPoint &src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool copy_restricted,
                                const bool need_valid_return);
      virtual void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &fields);
      virtual ApEvent register_collective_user(InstanceView *view, 
                                const RegionUsage &usage,
                                const FieldMask &user_mask,
                                IndexSpaceNode *expr,
                                const UniqueID op_id,
                                const size_t op_ctx_index,
                                const unsigned index,
                                ApEvent term_event,
                                RtEvent collect_event,
                                std::set<RtEvent> &applied_events,
                                const CollectiveMapping *mapping,
                                Operation *local_collective_op,
                                const PhysicalTraceInfo &trace_info,
                                const bool symbolic);
    public:
      virtual RtEvent find_field_reservations(const FieldMask &mask,
                                DistributedID view_did,
                                std::vector<Reservation> *reservations,
                                AddressSpaceID source,
                                RtUserEvent to_trigger);
      virtual void update_field_reservations(const FieldMask &mask,
                                DistributedID view_did,
                                const std::vector<Reservation> &rsrvs);
      virtual void reclaim_field_reservations(DistributedID view_did,
                                std::vector<Reservation> &to_delete);
    public:
      void process_collective_user_registration(const DistributedID view_did,
                                            const size_t op_ctx_index,
                                            const unsigned index,
                                            const AddressSpaceID origin,
                                            const CollectiveMapping *mapping,
                                            const PhysicalTraceInfo &trace_info,
                                            ApEvent remote_term_event,
                                            ApUserEvent remote_ready_event,
                                            RtUserEvent remote_registered);
      
    public:
      virtual void send_manager(AddressSpaceID target);
      
      static void handle_collective_user_registration(Runtime *runtime,
                                                      Deserializer &derez);
    public:
      virtual void get_instance_pointers(Memory memory, 
                                    std::vector<uintptr_t> &pointers) const;
      virtual RtEvent perform_deletion(AddressSpaceID source, 
                                       AutoLock *i_lock = NULL);
      virtual void force_deletion(void);
      virtual RtEvent update_garbage_collection_priority(AddressSpaceID source,
                                                         GCPriority priority);
      virtual RtEvent attach_external_instance(void);
      virtual RtEvent detach_external_instance(void);
      virtual bool has_visible_from(const std::set<Memory> &memories) const;
    };

    /**
     * \class CollectiveManager
     * The collective instance manager class supports the interface
     * of a single instance but is actually contains N distributed 
     * copies of the same data and will perform collective operations
     * as part of any reads, writes, or reductions performed to it.
     */
    class CollectiveManager : public PhysicalManager,
              public LegionHeapify<CollectiveManager> {
    public:
      static const AllocationType alloc_type = COLLECTIVE_INST_MANAGER_ALLOC;
    public:
      struct DeferCollectiveManagerArgs : 
        public LgTaskArgs<DeferCollectiveManagerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COLLECTIVE_MANAGER_TASK_ID;
      public:
        DeferCollectiveManagerArgs(DistributedID d, AddressSpaceID own, 
            const Domain &pts, size_t tp, CollectiveMapping *map, size_t f,
            IndexSpaceExpression *lx, const PendingRemoteExpression &pending,
            FieldSpace h, RegionTreeID tid, LayoutConstraintID l,
            ReductionOpID redop, const void *piece_list,size_t piece_list_size,
            const AddressSpaceID source, GarbageCollectionState state,
            bool multi_instace);
      public:
        const DistributedID did;
        const AddressSpaceID owner;
        const Domain dense_points;
        const size_t total_points;
        CollectiveMapping *const mapping;
        const size_t footprint;
        IndexSpaceExpression *const local_expr;
        const PendingRemoteExpression pending;
        const FieldSpace handle;
        const RegionTreeID tree_id;
        const LayoutConstraintID layout_id;
        const ReductionOpID redop;
        const void *const piece_list;
        const size_t piece_list_size;
        const AddressSpaceID source;
        const GarbageCollectionState state;
        const bool multi_instance;
      };
    protected:
      struct RemoteInstInfo {
        PhysicalInstance instance;
        ApEvent unique_event;
        unsigned index;
      public:
        inline bool operator==(const RemoteInstInfo &rhs) const
        {
          if (instance != rhs.instance) return false;
          if (unique_event != rhs.unique_event) return false;
          if (index != rhs.index) return false;
          return true;
        }
      };
    public:
      CollectiveManager(RegionTreeForest *ctx, DistributedID did,
                        AddressSpaceID owner_space, const Domain &dense_points,
                        size_t total_pts, CollectiveMapping *mapping,
                        IndexSpaceExpression *instance_domain,
                        const void *piece_list, size_t piece_list_size,
                        FieldSpaceNode *node, RegionTreeID tree_id,
                        LayoutDescription *desc, ReductionOpID redop, 
                        bool register_now, size_t footprint,
                        bool external_instance, bool multi_instance);
      CollectiveManager(const CollectiveManager &rhs) = delete;
      virtual ~CollectiveManager(void);
    public:
      CollectiveManager& operator=(const CollectiveManager &rh) = delete;
    public:
      // These methods can be slow in the case where there is not a point
      // space and the set of points are implicit so only use them for 
      // error checking code
      bool contains_point(const DomainPoint &point) const;
      bool contains_isomorphic_points(IndexSpaceNode *points) const;
    public:
      bool is_first_local_point(const DomainPoint &point) const;
    public:
      void record_point_instance(const DomainPoint &point,
                                 PhysicalInstance instance,
                                 ApEvent ready_event);
      bool finalize_point_instance(const DomainPoint &point,
                                   bool success, bool acquire, 
                                   bool remote = false);
    public:
      virtual ApEvent get_use_event(ApEvent user = ApEvent::NO_AP_EVENT) const;
      virtual ApEvent get_unique_event(const DomainPoint &point) const;
      virtual bool has_collective_point(const DomainPoint &point) const
        { return contains_point(point); }
      virtual PhysicalInstance get_instance(const DomainPoint &point, 
                                            bool from_mapper = false) const;
      virtual PointerConstraint
                     get_pointer_constraint(const DomainPoint &key) const; 
      virtual Memory get_memory(const DomainPoint &point,
                                bool from_mapper = false) const
        { return get_instance(point, from_mapper).get_location(); }
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      virtual void get_instance_pointers(Memory memory, 
                                    std::vector<uintptr_t> &pointers) const;
      virtual RtEvent perform_deletion(AddressSpaceID source,
                                       AutoLock *i_lock = NULL);
      virtual void force_deletion(void);
      virtual RtEvent update_garbage_collection_priority(AddressSpaceID source,
                                                         GCPriority priority);
      virtual RtEvent attach_external_instance(void);
      virtual RtEvent detach_external_instance(void);
      virtual bool has_visible_from(const std::set<Memory> &memories) const;
    protected:
      void collective_deletion(RtEvent deferred_event);
      void collective_force(void);
      void collective_detach(std::set<RtEvent> &detach_events);
      RtEvent broadcast_point_request(const DomainPoint &point) const;
      void find_or_forward_physical_instance(
            AddressSpaceID source, AddressSpaceID origin,
            std::set<DomainPoint> &points, RtUserEvent to_trigger);
      void record_remote_physical_instances(
            const std::map<DomainPoint,RemoteInstInfo> &instances);
    public:
      virtual ApEvent fill_from(FillView *fill_view, InstanceView *dst_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool fill_restricted,
                                const bool need_valid_return);
      virtual ApEvent copy_from(InstanceView *src_view, InstanceView *dst_view,
                                PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const DomainPoint &src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                CopyAcrossHelper *across_helper,
                                const bool manage_dst_events,
                                const bool copy_restricted,
                                const bool need_valid_return);
      virtual void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &fields);
      virtual ApEvent register_collective_user(InstanceView *view, 
                                const RegionUsage &usage,
                                const FieldMask &user_mask,
                                IndexSpaceNode *expr,
                                const UniqueID op_id,
                                const size_t op_ctx_index,
                                const unsigned index,
                                ApEvent term_event,
                                RtEvent collect_event,
                                std::set<RtEvent> &applied_events,
                                const CollectiveMapping *mapping,
                                Operation *local_collective_op,
                                const PhysicalTraceInfo &trace_info,
                                const bool symbolic);
    public:
      virtual RtEvent find_field_reservations(const FieldMask &mask,
                                DistributedID view_did,const DomainPoint &point,
                                std::vector<Reservation> *reservations,
                                AddressSpaceID source,
                                RtUserEvent to_trigger);
      virtual void update_field_reservations(const FieldMask &mask,
                                DistributedID view_did,const DomainPoint &point,
                                const std::vector<Reservation> &rsrvs);
      virtual void reclaim_field_reservations(DistributedID view_did,
                                std::vector<Reservation> &to_delete);
    public:
      void find_points_in_memory(Memory memory, 
                                 std::vector<DomainPoint> &point) const;
      void find_points_nearest_memory(Memory memory,
                                 std::map<DomainPoint,Memory> &points,
                                 bool bandwidth) const;
      RtEvent find_points_nearest_memory(Memory, AddressSpaceID source, 
                                 std::map<DomainPoint,Memory> *points, 
                                 std::atomic<size_t> *target,
                                 AddressSpaceID origin, size_t best,
                                 bool bandwidth) const;
      void find_nearest_local_points(Memory memory, size_t &best,
                                 std::map<DomainPoint,Memory> &results,
                                 bool bandwidth) const;
    public:
      inline AddressSpaceID select_origin_space(void) const
        { return (collective_mapping->contains(local_space) ? local_space :
                  collective_mapping->find_nearest(local_space)); }
      void register_collective_analysis(DistributedID view_did,
                                        CollectiveCopyFillAnalysis *analysis);
      RtEvent find_collective_analyses(DistributedID view_did,
                                       size_t context_index, unsigned index,
                     const std::vector<CollectiveCopyFillAnalysis*> *&analyses);
      void perform_collective_fill(FillView *fill_view, InstanceView *dst_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                Operation *op, const unsigned index,
                                const size_t op_context_index,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent result, AddressSpaceID origin,
                                const bool fill_restricted);
      ApEvent perform_collective_point(InstanceView *src_view,
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const Memory location,
                                const DistributedID dst_view_did,
                                const DomainPoint &dst_point,
                                const DomainPoint &src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events);
      void perform_collective_pointwise(CollectiveManager *source,
                                InstanceView *src_view,
                                InstanceView *dst_view,
                                ApEvent precondition,
                                PredEvent predicate_guard, 
                                IndexSpaceExpression *copy_expression,
                                Operation *op, const unsigned index,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const DomainPoint &origin_point,
                                const DomainPoint &origin_src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent all_done, ApBarrier all_bar,
                                ShardID owner_shard, AddressSpaceID origin,
                                const uint64_t allreduce_tag,
                                const bool copy_restricted); 
      void perform_collective_broadcast(InstanceView *dst_view,
                                const std::vector<CopySrcDstField> &src_fields,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const UniqueInst &src_inst,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent copy_done, ApUserEvent all_done,
                                ApBarrier all_bar, ShardID owner_shard,
                                AddressSpaceID origin,
                                const bool copy_restricted);
      void perform_collective_reducecast(IndividualManager *source,
                                InstanceView *dst_view,
                                const std::vector<CopySrcDstField> &src_fields,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const UniqueInst &src_inst,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent copy_done,
                                ApBarrier all_bar, ShardID owner_shard,
                                AddressSpaceID origin,
                                const bool copy_restricted);
      void perform_collective_hourglass(CollectiveManager *source,
                                InstanceView *src_view, InstanceView *dst_view,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const DomainPoint &src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent all_done,
                                AddressSpaceID target,
                                const bool copy_restricted);
      void perform_collective_allreduce(ReductionView *src_view,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                       const std::vector<CollectiveCopyFillAnalysis*> *analyses,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                const uint64_t allreduce_tag);
      // Degenerate case
      ApEvent perform_hammer_reduction(InstanceView *src_view,
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expresison,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const UniqueInst &dst_inst,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                AddressSpaceID origin);
    protected:
      void perform_single_allreduce(FillView *fill_view,
                                const DistributedID reduce_view_did,
                                const uint64_t allreduce_tag,
                                Operation *op, PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &applied_events,
                                std::vector<ApEvent> &instance_preconditions,
                    std::vector<std::vector<CopySrcDstField> > &local_fields,
              const std::vector<std::vector<Reservation> > &reservations,
                                std::vector<ApEvent> &local_init_events,
                                std::vector<ApEvent> &local_final_events);
      unsigned perform_multi_allreduce(FillView *fill_view,
                                const DistributedID reduce_view_did,
                                const uint64_t allreduce_tag,
                                Operation *op, PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                    const std::vector<CollectiveCopyFillAnalysis*> *analyses,
                                std::set<RtEvent> &applied_events,
                                std::vector<ApEvent> &instance_preconditions,
                    std::vector<std::vector<CopySrcDstField> > &local_fields,
              const std::vector<std::vector<Reservation> > &reservations,
                                std::vector<ApEvent> &local_init_events,
                                std::vector<ApEvent> &local_final_events); 
      void send_allreduce_stage(const uint64_t allreduce_tag, const int stage,
                                const int local_rank, ApEvent src_precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                const PhysicalTraceInfo &trace_info,
                                const std::vector<CopySrcDstField> &src_fields,
                                const DomainPoint &src_point,
                                const AddressSpaceID *targets, size_t total,
                                std::vector<ApEvent> &src_events);
      void receive_allreduce_stage(const UniqueInst dst_inst,
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
                                std::vector<ApEvent> &dst_events);
      
      void process_distribute_allreduce(const uint64_t allreduce_tag,
                                const int src_rank, const int stage,
                                std::vector<CopySrcDstField> &src_fields,
                                const ApEvent src_precondition,
                                ApUserEvent src_postcondition,
                                ApBarrier src_barrier, ShardID bar_shard,
                                const DomainPoint &src_point);
      void process_register_user_request(const DistributedID view_did,
                                const size_t op_ctx_index, const unsigned index,
                                const RtEvent registered);
      void process_register_user_response(const DistributedID view_did,
                                const size_t op_ctx_index, const unsigned index,
                                const RtEvent registered);
      void finalize_collective_user(InstanceView *view,
                                const RegionUsage &usage,
                                const FieldMask &user_mask,
                                IndexSpaceNode *expr,
                                const UniqueID op_id,
                                const size_t op_ctx_index,
                                const unsigned index,
                                RtEvent collect_event,
                                RtUserEvent local_registered,
                                RtEvent global_registered,
                                ApUserEvent ready_event,
                                ApEvent term_event,
                                const PhysicalTraceInfo *trace_info,
                                std::vector<CollectiveCopyFillAnalysis*> &ses,
                                const bool symbolic) const;
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
    public:
      virtual void send_manager(AddressSpaceID target);
    public:
      static void handle_send_manager(Runtime *runtime, 
                                      AddressSpaceID source,
                                      Deserializer &derez);
      static void handle_instance_creation(Runtime *runtime, 
                                           Deserializer &derez);
      static void handle_defer_manager(const void *args, Runtime *runtime);
      static void handle_distribute_fill(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_point(Runtime *runtime,
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_pointwise(Runtime *runtime,
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_reduction(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_broadcast(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_reducecast(Runtime *runtime,
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_hourglass(Runtime *runtime,
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_distribute_allreduce(Runtime *runtime,
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_hammer_reduction(Runtime *runtime, 
                                    AddressSpaceID source, Deserializer &derez);
      static void handle_register_user_request(Runtime *runtime,
                                    Deserializer &derez);
      static void handle_register_user_response(Runtime *runtime,
                                    Deserializer &derez);
      static void handle_point_request(Runtime *runtime, Deserializer &derez);
      static void handle_point_response(Runtime *runtime, Deserializer &derez);
      static void handle_find_points_request(Runtime *runtime,
                                    Deserializer &derez, AddressSpaceID source);
      static void handle_find_points_response(Deserializer &derez);
      static void handle_nearest_points_request(Runtime *runtime,
                                                Deserializer &derez);
      static void handle_nearest_points_response(Deserializer &derez);
      static void handle_remote_registration(Runtime *runtime,
                                             Deserializer &derez);
      static void handle_deletion(Runtime *runtime, Deserializer &derez);
      static void create_collective_manager(Runtime *runtime, DistributedID did,
          AddressSpaceID owner_space, const Domain &dense_points,
          size_t points, CollectiveMapping *collective_mapping,
          size_t inst_footprint, IndexSpaceExpression *inst_domain,
          const void *piece_list, size_t piece_list_size, 
          FieldSpaceNode *space_node, RegionTreeID tree_id, 
          LayoutConstraints *constraints, ReductionOpID redop, 
          GarbageCollectionState state, bool multi_instance);
      void pack_fields(Serializer &rez, 
                       const std::vector<CopySrcDstField> &fields) const;
      void log_remote_point_instances(
                       const std::vector<CopySrcDstField> &fields,
                       const std::vector<unsigned> &indexes,
                       const std::vector<DomainPoint> &points,
                       const std::vector<ApEvent> &events);
      static void unpack_fields(std::vector<CopySrcDstField> &fields,
          Deserializer &derez, std::set<RtEvent> &ready_events,
          CollectiveManager *manager, RtEvent man_ready, Runtime *runtime);
    public:
      const size_t total_points;
      // This domain should only be valid if it is a dense rectangle
      // No sparsity maps!
      const Domain dense_points;
      static constexpr size_t GUARD_SIZE = std::numeric_limits<size_t>::max();
    protected:
      // Note that there is a collective mapping from DistributedCollectable
      //CollectiveMapping *collective_mapping;
      std::vector<MemoryManager*> memories; // local memories
      std::vector<PhysicalInstance> instances; // local instances
      std::vector<DomainPoint> instance_points; // points for local instances
      std::vector<ApEvent> instance_events; // ready events for each instance 
      std::map<DomainPoint,RemoteInstInfo> remote_points;
    protected:
      struct UserRendezvous {
        UserRendezvous(void) 
          : remaining_local_arrivals(0), remaining_remote_arrivals(0),
            valid_analyses(0), trace_info(NULL), view(NULL), mask(NULL), 
            expr(NULL), op_id(0), symbolic(false), local_initialized(false) { }
        // event for when local instances can be used
        ApUserEvent ready_event; 
        // all the local term events
        std::vector<ApEvent> local_term_events;
        // events from remote nodes indicating they are registered
        std::vector<RtEvent> remote_registered;
        // the local set of analyses
        std::vector<CollectiveCopyFillAnalysis*> analyses;
        // event for when the analyses are all registered
        RtUserEvent analyses_ready;
        // event to trigger when local registration is done
        RtUserEvent local_registered; 
        // event that marks when all registrations are done
        RtUserEvent global_registered;
        // Counts of remaining notficiations before registration
        unsigned remaining_local_arrivals;
        unsigned remaining_remote_arrivals;
        unsigned valid_analyses;
        // PhysicalTraceInfo that made the ready_event and should trigger it
        PhysicalTraceInfo *trace_info;
        // Arguments for performing the local registration
        InstanceView *view;
        RegionUsage usage;
        FieldMask *mask;
        IndexSpaceNode *expr;
        UniqueID op_id;
        RtEvent collect_event;
        bool symbolic;
        bool local_initialized;
      };
      std::map<RendezvousKey,UserRendezvous> rendezvous_users;
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
        DomainPoint src_point;
      };
      std::map<CopyKey,AllReduceCopy> all_reduce_copies;
      struct AllReduceStage {
        UniqueInst dst_inst;
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
      std::map<std::pair<DistributedID,DomainPoint>,
                std::map<unsigned,Reservation> > view_reservations;
    protected:
      std::atomic<uint64_t> unique_allreduce_tag;
    public:
      // A boolean flag that says whether this collective instance
      // has multiple instances on every node. This is primarily
      // useful for reduction instances where we want to pick an
      // algorithm for performing an in-place all-reduce
      const bool multi_instance;
    };
#endif

    /**
     * \class VirtualManager
     * This is a singleton class of which there will be exactly one
     * on every node in the machine. The virtual manager class will
     * represent all the virtual instances.
     */
    class VirtualManager : public InstanceManager,
                           public LegionHeapify<VirtualManager> {
    public:
      VirtualManager(Runtime *runtime, DistributedID did, 
                     LayoutDescription *layout, CollectiveMapping *mapping);
      VirtualManager(const VirtualManager &rhs);
      virtual ~VirtualManager(void);
    public:
      VirtualManager& operator=(const VirtualManager &rhs);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      virtual void notify_local(void) { }
    public: 
      virtual PointerConstraint get_pointer_constraint(void) const;
      virtual void send_manager(AddressSpaceID target);
    };

#ifdef NO_EXPLICIT_COLLECTIVES
    /**
     * \class PendingCollectiveManager
     * This data structure stores the necessary meta-data required
     * for constructing a CollectiveManager by an InstanceBuilder
     * when creating a physical instance for a collective instance
     */
    class PendingCollectiveManager : public Collectable {
    public:
      PendingCollectiveManager(DistributedID did, size_t total_points,
                               const Domain &dense_points,
                               CollectiveMapping *mapping, bool multi_instance);
      PendingCollectiveManager(const PendingCollectiveManager &rhs) = delete;
      ~PendingCollectiveManager(void);
      PendingCollectiveManager& operator=(
          const PendingCollectiveManager&) = delete;
    public:
      const DistributedID did;
      const size_t total_points;
      const Domain dense_points;
      CollectiveMapping *const collective_mapping;
      const bool multi_instance;
    public:
      void pack(Serializer &rez) const;
      static PendingCollectiveManager* unpack(Deserializer &derez);
    };
#endif

    /**
     * \class InstanceBuilder 
     * A helper for building physical instances of logical regions
     */
    class InstanceBuilder : public ProfilingResponseHandler {
    public:
      InstanceBuilder(const std::vector<LogicalRegion> &regs,
                      const LayoutConstraintSet &cons, Runtime *rt,
                      MemoryManager *memory = NULL, UniqueID cid = 0)
        : regions(regs), constraints(cons), runtime(rt), memory_manager(memory),
          creator_id(cid), instance(PhysicalInstance::NO_INST), 
          field_space_node(NULL), instance_domain(NULL), tree_id(0),
          redop_id(0), reduction_op(NULL), realm_layout(NULL), piece_list(NULL),
          piece_list_size(0), valid(false) { }
      InstanceBuilder(const std::vector<LogicalRegion> &regs,
                      IndexSpaceExpression *expr, FieldSpaceNode *node,
                      RegionTreeID tree_id, const LayoutConstraintSet &cons, 
                      Runtime *rt, MemoryManager *memory, UniqueID cid,
                      const void *piece_list, size_t piece_list_size); 
      virtual ~InstanceBuilder(void);
    public:
      void initialize(RegionTreeForest *forest);
      PhysicalManager* create_physical_instance(RegionTreeForest *forest,
            LayoutConstraintKind *unsat_kind,
                        unsigned *unsat_index, size_t *footprint = NULL,
                        RtEvent collection_done = RtEvent::NO_RT_EVENT);
    public:
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &response,
                                      const void *orig, size_t orig_length);
    protected:
      void compute_space_and_domain(RegionTreeForest *forest);
    protected:
      void compute_layout_parameters(void);
    protected:
      const std::vector<LogicalRegion> &regions;
      LayoutConstraintSet constraints;
      Runtime *const runtime;
      MemoryManager *const memory_manager;
      const UniqueID creator_id;
    protected:
      PhysicalInstance instance;
      RtUserEvent profiling_ready;
    protected:
      FieldSpaceNode *field_space_node;
      IndexSpaceExpression *instance_domain;
      RegionTreeID tree_id;
      // Mapping from logical field order to layout order
      std::vector<unsigned> mask_index_map;
      std::vector<size_t> field_sizes;
      std::vector<CustomSerdezID> serdez;
      FieldMask instance_mask;
      ReductionOpID redop_id;
      const ReductionOp *reduction_op;
      Realm::InstanceLayoutGeneric *realm_layout;
      void *piece_list;
      size_t piece_list_size;
    public:
      bool valid;
    };

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID InstanceManager::encode_instance_did(
                               DistributedID did, bool external, bool reduction)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, PHYSICAL_MANAGER_DC | 
                                        (external ? EXTERNAL_CODE : 0) | 
                                        (reduction ? REDUCTION_CODE : 0));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_physical_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC-1)) ==
                                                        PHYSICAL_MANAGER_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_reduction_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      const unsigned decode = LEGION_DISTRIBUTED_HELP_DECODE(did);
      if ((decode & (DIST_TYPE_LAST_DC-1)) != PHYSICAL_MANAGER_DC)
        return false;
      return ((decode & REDUCTION_CODE) != 0);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_external_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      const unsigned decode = LEGION_DISTRIBUTED_HELP_DECODE(did);
      if ((decode & (DIST_TYPE_LAST_DC-1)) != PHYSICAL_MANAGER_DC)
        return false;
      return ((decode & EXTERNAL_CODE) != 0);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_physical_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_physical_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_virtual_manager(void) const
    //--------------------------------------------------------------------------
    {
      return (did == 0);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_external_instance(void) const
    //--------------------------------------------------------------------------
    {
      return is_external_did(did);
    }

    //--------------------------------------------------------------------------
    inline PhysicalManager* InstanceManager::as_physical_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_physical_manager());
#endif
      return static_cast<PhysicalManager*>(const_cast<InstanceManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline VirtualManager* InstanceManager::as_virtual_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_virtual_manager());
#endif
      return static_cast<VirtualManager*>(const_cast<InstanceManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline void PhysicalManager::add_base_valid_ref(
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
    inline void PhysicalManager::add_nested_valid_ref(
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
      add_nested_valid_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
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
    inline bool PhysicalManager::acquire_instance(ReferenceSource source) 
    //--------------------------------------------------------------------------
    {
#ifndef DEBUG_LEGION_GC
      // Note that we cannot do this for external instances as they might
      // have been detached while still holding valid references so they
      // have to go through the full path every time
      if (!is_external_instance())
      {
        // Check to see if we can do the add without the lock first
        int current = valid_references.load();
        while (current > 0)
        {
          int next = current + 1;
          if (valid_references.compare_exchange_weak(current, next))
          {
#ifdef LEGION_GC
            log_base_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
            return true;
          }
        }
      }
      bool result = acquire_internal();
#else
      bool result = acquire_internal(source, detailed_base_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_base_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::acquire_instance(DistributedID source) 
    //--------------------------------------------------------------------------
    {
#ifndef DEBUG_LEGION_GC
      // Note that we cannot do this for external instances as they might
      // have been detached while still holding valid references so they
      // have to go through the full path every time
      if (!is_external_instance())
      {
        // Check to see if we can do the add without the lock first
        int current = valid_references.load();
        while (current > 0)
        {
          int next = current + 1;
          if (valid_references.compare_exchange_weak(current, next))
          {
#ifdef LEGION_GC
            log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
            return true;
          }
        }
      }
      bool result = acquire_internal();
#else
      bool result = acquire_internal(LEGION_DISTRIBUTED_ID_FILTER(source),
                                     detailed_nested_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::remove_base_valid_ref(
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
    inline bool PhysicalManager::remove_nested_valid_ref(
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

#endif // __LEGION_INSTANCES_H__
