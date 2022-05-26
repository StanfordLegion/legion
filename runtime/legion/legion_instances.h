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
      void log_instance_layout(ApEvent inst_event) const;
    public:
      void compute_copy_offsets(const FieldMask &copy_mask, 
                                const PhysicalInstance instance,  
#ifdef LEGION_SPY
                                const ApEvent inst_event, 
#endif
                                std::vector<CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                const PhysicalInstance instance,
#ifdef LEGION_SPY
                                const ApEvent inst_event,
#endif
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
     * \class InstanceManager
     * This is the abstract base class for all instances of a physical
     * resource manager for memory.
     */
    class InstanceManager : public DistributedCollectable {
    public:
      enum {
        EXTERNAL_CODE = 0x10,
        REDUCTION_CODE = 0x20,
        COLLECTIVE_CODE = 0x40,
      };
    public:
      InstanceManager(RegionTreeForest *forest, AddressSpaceID owner, 
                      DistributedID did, LayoutDescription *layout,
                      FieldSpaceNode *node, IndexSpaceExpression *domain,
                      RegionTreeID tree_id, bool register_now);
      virtual ~InstanceManager(void);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const = 0;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const = 0;
    public: 
      virtual ApEvent get_use_event(void) const = 0;
      virtual ApEvent get_unique_event(void) const = 0;
      virtual PhysicalInstance get_instance(const DomainPoint &key) const = 0;
      virtual PointerConstraint 
                     get_pointer_constraint(const DomainPoint &key) const = 0;
      virtual InstanceView* create_instance_top_view(InnerContext *context,
                                            AddressSpaceID logical_owner) = 0;
    public:
      inline bool is_reduction_manager(void) const;
      inline bool is_physical_manager(void) const;
      inline bool is_virtual_manager(void) const;
      inline bool is_external_instance(void) const;
      inline bool is_collective_manager(void) const;
      inline PhysicalManager* as_physical_manager(void) const;
      inline VirtualManager* as_virtual_manager(void) const;
      inline IndividualManager* as_individual_manager(void) const;
      inline CollectiveManager* as_collective_manager(void) const;
    public:
      static inline DistributedID encode_instance_did(DistributedID did,
                        bool external, bool reduction, bool collective);
      static inline bool is_physical_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_external_did(DistributedID did);
      static inline bool is_collective_did(DistributedID did);
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
      bool meets_region_tree(const std::vector<LogicalRegion> &regions) const; 
      bool entails(LayoutConstraints *constraints, const DomainPoint &key,
                   const LayoutConstraint **failed_constraint) const;
      bool entails(const LayoutConstraintSet &constraints, 
                   const DomainPoint &key,
                   const LayoutConstraint **failed_constraint) const;
      bool conflicts(LayoutConstraints *constraints, const DomainPoint &key,
                     const LayoutConstraint **conflict_constraint) const;
      bool conflicts(const LayoutConstraintSet &constraints,
                     const DomainPoint &key,
                     const LayoutConstraint **conflict_constraint) const;
    public:
      RegionTreeForest *const context;
      LayoutDescription *const layout;
      FieldSpaceNode *const field_space_node;
      IndexSpaceExpression *instance_domain;
      const RegionTreeID tree_id;
    };

    /**
     * \class PhysicalManager 
     * This is an abstract intermediate class for representing an allocation
     * of data; this includes both individual instances and collective instances
     */
    class PhysicalManager : public InstanceManager {
    public:
      struct GarbageCollectionArgs : public LgTaskArgs<GarbageCollectionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COLLECT_ID;
      public:
        GarbageCollectionArgs(CollectableView *v, std::set<ApEvent> *collect)
          : LgTaskArgs<GarbageCollectionArgs>(implicit_provenance), 
            view(v), to_collect(collect) { }
      public:
        CollectableView *const view;
        std::set<ApEvent> *const to_collect;
      };
    public:
      struct CollectableInfo {
      public:
        CollectableInfo(void) : events_added(0) { }
      public:
        std::set<ApEvent> view_events;
        // This event tracks when tracing is completed and it is safe
        // to resume pruning of users from this view
        RtEvent collect_event;
        // Events added since the last collection of view events
        unsigned events_added;
      };
      enum GarbageCollectionState {
        VALID_GC_STATE,
        ACQUIRED_GC_STATE,
        COLLECTABLE_GC_STATE,
        PENDING_COLLECTED_GC_STATE,
        COLLECTED_GC_STATE,
      };
    public:
      PhysicalManager(RegionTreeForest *ctx, LayoutDescription *layout, 
                      DistributedID did, AddressSpaceID owner_space, 
                      const size_t footprint, ReductionOpID redop_id, 
                      const ReductionOp *rop, FieldSpaceNode *node,
                      IndexSpaceExpression *index_domain, 
                      const void *piece_list, size_t piece_list_size,
                      RegionTreeID tree_id, ApEvent unique, 
                      bool register_now);
      virtual ~PhysicalManager(void);
    public:
      virtual ApEvent get_unique_event(void) const { return unique_event; }
    public:
      void log_instance_creation(UniqueID creator_id, Processor proc,
                     const std::vector<LogicalRegion> &regions) const; 
    public:
      virtual ApEvent fill_from(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<FillView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper = NULL) = 0;
      virtual ApEvent copy_from(PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<InstanceView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper = NULL) = 0;
      virtual void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &fields) = 0;
    public:
      virtual void send_manager(AddressSpaceID target) = 0; 
      static void handle_manager_request(Deserializer &derez, 
                          Runtime *runtime, AddressSpaceID source);
    public:
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      bool acquire_instance(ReferenceSource source, ReferenceMutator *mutator);
      bool can_collect(AddressSpaceID source, bool &already_collected);
      bool collect(RtEvent &collected);
      RtEvent set_garbage_collection_priority(MapperID mapper_id,
                                              Processor p, GCPriority priority);
      virtual void get_instance_pointers(Memory memory, 
                                    std::vector<uintptr_t> &pointers) const = 0;
      virtual RtEvent perform_deletion(AddressSpaceID source, 
                                       AutoLock *i_lock = NULL) = 0;
      virtual void force_deletion(void) = 0;
      virtual RtEvent update_garbage_collection_priority(
                                                       GCPriority priority) = 0;
      virtual RtEvent attach_external_instance(void) = 0;
      virtual RtEvent detach_external_instance(void) = 0;
      virtual bool has_visible_from(const std::set<Memory> &memories) const = 0;
      virtual Memory get_memory(void) const = 0; 
      inline size_t get_instance_size(void) const { return instance_footprint; }
    public:
      // Methods for creating/finding/destroying logical top views
      virtual InstanceView* create_instance_top_view(InnerContext *context,
                                            AddressSpaceID logical_owner);
      void register_active_context(InnerContext *context);
      void unregister_active_context(InnerContext *context); 
    public:
      PieceIteratorImpl* create_piece_iterator(IndexSpaceNode *privilege_node);
      void defer_collect_user(CollectableView *view, ApEvent term_event,
                              RtEvent collect, std::set<ApEvent> &to_collect, 
                              bool &add_ref, bool &remove_ref);
      void find_shutdown_preconditions(std::set<ApEvent> &preconditions);
    public:
      bool meets_regions(const std::vector<LogicalRegion> &regions,
                         bool tight_region_bounds = false) const;
      bool meets_expression(IndexSpaceExpression *expr, 
                            bool tight_bounds = false) const;
    protected:
      void prune_gc_events(void);
      void pack_garbage_collection_state(Serializer &rez,AddressSpaceID target);
      void initialize_remote_gc_state(GarbageCollectionState state);
    public: 
      static ApEvent fetch_metadata(PhysicalInstance inst, ApEvent use_event);
      static void handle_acquire_request(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_acquire_response(Deserializer &derez, 
          AddressSpaceID source);
      static void handle_garbage_collection_request(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_garbage_collection_response(Deserializer &derez);
      static void handle_garbage_collection_acquire(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_garbage_collection_acquired(Deserializer &derez);
      static void handle_garbage_collection_priority_update(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_garbage_collection_debug_request(Runtime *runtime,
          Deserializer &derez, AddressSpaceID source);
      static void handle_garbage_collection_debug_response(Deserializer &derez);
    public:
      const size_t instance_footprint;
      const ReductionOp *reduction_op;
      const ReductionOpID redop;
      // Unique identifier event that is common across nodes
      const ApEvent unique_event;
      const void *const piece_list;
      const size_t piece_list_size;
    protected:
      mutable LocalLock inst_lock;
      std::set<InnerContext*> active_contexts;
    protected:
      // Stuff for garbage collection
      GarbageCollectionState gc_state; 
      unsigned pending_changes;
      std::atomic<unsigned> remaining_collection_guards;
      RtEvent collection_ready;
      RtUserEvent deferred_deletion;
      bool currently_active;
      // Garbage collection priorities
      GCPriority min_gc_priority;
      RtEvent priority_update_done;
      std::map<std::pair<MapperID,Processor>,GCPriority> mapper_gc_priorities;
    private:
      // Events that have to trigger before we can remove our GC reference
      std::map<CollectableView*,CollectableInfo> gc_events;
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
      std::vector<CopySrcDstField> offsets; 
      LegionDeque<std::pair<FieldMask,FieldMask> > compressed_cache;
    };

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
            ReductionOpID redop, const void *piece_list, size_t piece_list_size,
            AddressSpaceID src, GarbageCollectionState state);
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
        const ReductionOpID redop;
        const void *const piece_list;
        const size_t piece_list_size;
        const AddressSpaceID source;
        const GarbageCollectionState state;
      };
    public:
      IndividualManager(RegionTreeForest *ctx, DistributedID did,
                        AddressSpaceID owner_space,
                        MemoryManager *memory, PhysicalInstance inst, 
                        IndexSpaceExpression *instance_domain,
                        const void *piece_list, size_t piece_list_size,
                        FieldSpaceNode *node, RegionTreeID tree_id,
                        LayoutDescription *desc, ReductionOpID redop, 
                        bool register_now, size_t footprint,
                        ApEvent use_event, bool external_instance,
                        const ReductionOp *op = NULL);
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
      virtual ApEvent get_use_event(void) const { return use_event; }
      virtual PhysicalInstance get_instance(const DomainPoint &key) const 
                                                   { return instance; }
      virtual PointerConstraint
                     get_pointer_constraint(const DomainPoint &key) const;
    public:
      virtual ApEvent fill_from(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<FillView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper = NULL);
      virtual ApEvent copy_from(PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<InstanceView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper = NULL);
      virtual void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &fields);
    public:
      void initialize_across_helper(CopyAcrossHelper *across_helper,
                                    const FieldMask &mask,
                                    const std::vector<unsigned> &src_indexes,
                                    const std::vector<unsigned> &dst_indexes);
    public:
      virtual void send_manager(AddressSpaceID target);
      static void handle_send_manager(Runtime *runtime, 
                                      AddressSpaceID source,
                                      Deserializer &derez); 
      static void handle_defer_manager(const void *args, Runtime *runtime);
      static void create_remote_manager(Runtime *runtime, DistributedID did,
          AddressSpaceID owner_space, Memory mem, PhysicalInstance inst,
          size_t inst_footprint, IndexSpaceExpression *inst_domain,
          const void *piece_list, size_t piece_list_size,
          FieldSpaceNode *space_node, RegionTreeID tree_id,
          LayoutConstraints *constraints, ApEvent use_event,
          ReductionOpID redop, GarbageCollectionState state);
    public:
      virtual void get_instance_pointers(Memory memory, 
                                    std::vector<uintptr_t> &pointers) const;
      virtual RtEvent perform_deletion(AddressSpaceID source, 
                                       AutoLock *i_lock = NULL);
      virtual void force_deletion(void);
      virtual RtEvent update_garbage_collection_priority(GCPriority);
      virtual RtEvent attach_external_instance(void);
      virtual RtEvent detach_external_instance(void);
      virtual bool has_visible_from(const std::set<Memory> &memories) const;
      virtual Memory get_memory(void) const;
    public:
      MemoryManager *const memory_manager;
      const PhysicalInstance instance;
      // Event that needs to trigger before we can start using
      // this physical instance.
      const ApEvent use_event; 
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
      enum MessageKind {
        PERFORM_DELETE_MESSAGE,
        FORCE_DELETE_MESSAGE,
        DETACH_EXTERNAL_MESSAGE,
        FINALIZE_MESSAGE,
      };
    public:
      struct DeferCollectiveManagerArgs : 
        public LgTaskArgs<DeferCollectiveManagerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_COLLECTIVE_MANAGER_TASK_ID;
      public:
        DeferCollectiveManagerArgs(DistributedID d, AddressSpaceID own, 
            IndexSpace p, size_t f, IndexSpaceExpression *lx, 
            const PendingRemoteExpression &pending, FieldSpace h, 
            RegionTreeID tid, LayoutConstraintID l, ApEvent use, 
            ReductionOpID redop, const void *piece_list,
            size_t piece_list_size, AddressSpaceID source,
            GarbageCollectionState state);
      public:
        const DistributedID did;
        const AddressSpaceID owner;
        IndexSpace point_space;
        const size_t footprint;
        IndexSpaceExpression *const local_expr;
        const PendingRemoteExpression pending;
        const FieldSpace handle;
        const RegionTreeID tree_id;
        const LayoutConstraintID layout_id;
        const ApEvent use_event;
        const ReductionOpID redop;
        const void *const piece_list;
        const size_t piece_list_size;
        const AddressSpaceID source;
        const GarbageCollectionState state;
      };
    public:
      CollectiveManager(RegionTreeForest *ctx, DistributedID did,
                        AddressSpaceID owner_space, IndexSpaceNode *point_space,
                        IndexSpaceExpression *instance_domain,
                        const void *piece_list, size_t piece_list_size,
                        FieldSpaceNode *node, RegionTreeID tree_id,
                        LayoutDescription *desc, ReductionOpID redop, 
                        bool register_now, size_t footprint,
                        ApEvent unique_event, bool external_instance);
      CollectiveManager(const CollectiveManager &rhs) = delete;
      virtual ~CollectiveManager(void);
    public:
      CollectiveManager& operator=(const CollectiveManager &rh) = delete;
    public:
      void finalize_collective_instance(ApUserEvent instance_event);
    public:
      virtual ApEvent get_use_event(void) const;
      virtual PhysicalInstance get_instance(const DomainPoint &key) const;
      virtual PointerConstraint
                     get_pointer_constraint(const DomainPoint &key) const; 
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
      virtual RtEvent update_garbage_collection_priority(GCPriority);
      virtual RtEvent attach_external_instance(void);
      virtual RtEvent detach_external_instance(void);
      virtual bool has_visible_from(const std::set<Memory> &memories) const;
      virtual Memory get_memory(void) const;
    protected:
      void perform_delete(RtEvent deferred_event, bool left); 
      void force_delete(bool left);
      void detach_external(RtUserEvent to_trigger, bool left, 
                  RtEvent full_detach = RtEvent::NO_RT_EVENT);
      bool finalize_message(void);
    protected:
      void collective_deletion(RtEvent deferred_event);
      void collective_force(void);
      void collective_detach(std::set<RtEvent> &detach_events);
    public:
      virtual ApEvent fill_from(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<FillView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper = NULL);
      virtual ApEvent copy_from(PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<InstanceView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper = NULL);
      virtual void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &fields);
    public:
      virtual void send_manager(AddressSpaceID target);
    public:
      static void handle_send_manager(Runtime *runtime, 
                                      AddressSpaceID source,
                                      Deserializer &derez);
      static void handle_defer_manager(const void *args, Runtime *runtime);
      static void handle_collective_message(Deserializer &derez,
                                            Runtime *runtime);
      static void create_collective_manager(Runtime *runtime, DistributedID did,
          AddressSpaceID owner_space, IndexSpaceNode *point_space,
          size_t inst_footprint, IndexSpaceExpression *inst_domain, 
          const void *piece_list, size_t piece_list_size, 
          FieldSpaceNode *space_node, RegionTreeID tree_id, 
          LayoutConstraints *constraints, ApEvent use_event,
          ReductionOpID redop, GarbageCollectionState state);
    public:
      IndexSpaceNode *const point_space;
    protected:
      std::vector<MemoryManager*> memories; // local memories
      std::vector<PhysicalInstance> instances; // local instances
      std::vector<AddressSpaceID> right_spaces;
      AddressSpaceID left_space;
    protected:
      RtEvent detached;
      unsigned finalize_messages;
      bool deleted_or_detached;
    };

    /**
     * \class VirtualManager
     * This is a singleton class of which there will be exactly one
     * on every node in the machine. The virtual manager class will
     * represent all the virtual virtual/composite instances.
     */
    class VirtualManager : public InstanceManager,
                           public LegionHeapify<VirtualManager> {
    public:
      VirtualManager(Runtime *runtime, DistributedID did, 
                     LayoutDescription *layout);
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
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public: 
      virtual ApEvent get_use_event(void) const;
      virtual ApEvent get_unique_event(void) const;
      virtual PhysicalInstance get_instance(const DomainPoint &key) const;
      virtual PointerConstraint
                     get_pointer_constraint(const DomainPoint &key) const;
      virtual void send_manager(AddressSpaceID target);
      virtual InstanceView* create_instance_top_view(InnerContext *context,
                                            AddressSpaceID logical_owner);
    };

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
                        CollectiveManager *collective, DomainPoint *point,
                        LayoutConstraintKind *unsat_kind,
                        unsigned *unsat_index, size_t *footprint = NULL,
                        RtEvent precondition = RtEvent::NO_RT_EVENT);
      CollectiveManager* create_collective_instance(RegionTreeForest *forest,
                        Memory::Kind mem_kind, IndexSpaceNode *point_space,
                        LayoutConstraintKind *unsat_kind, unsigned *unsat_index,
                        ApEvent ready_event, size_t *footprint = NULL);
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
              DistributedID did, bool external, bool reduction, bool collective)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, PHYSICAL_MANAGER_DC | 
                                        (external ? EXTERNAL_CODE : 0) | 
                                        (reduction ? REDUCTION_CODE : 0) |
                                        (collective ? COLLECTIVE_CODE : 0));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_physical_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xF) == 
                                                        PHYSICAL_MANAGER_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_reduction_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      const unsigned decode = LEGION_DISTRIBUTED_HELP_DECODE(did);
      if ((decode & 0xF) != PHYSICAL_MANAGER_DC)
        return false;
      return ((decode & REDUCTION_CODE) != 0);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_external_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      const unsigned decode = LEGION_DISTRIBUTED_HELP_DECODE(did);
      if ((decode & 0xF) != PHYSICAL_MANAGER_DC)
        return false;
      return ((decode & EXTERNAL_CODE) != 0);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_collective_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      const unsigned decode = LEGION_DISTRIBUTED_HELP_DECODE(did);
      if ((decode & 0xF) != PHYSICAL_MANAGER_DC)
        return false;
      return ((decode & COLLECTIVE_CODE) != 0);
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
    inline bool InstanceManager::is_collective_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_collective_did(did);
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
    inline IndividualManager* InstanceManager::as_individual_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_collective_manager());
#endif
      return 
        static_cast<IndividualManager*>(const_cast<InstanceManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline CollectiveManager* InstanceManager::as_collective_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_collective_manager());
#endif
      return 
        static_cast<CollectiveManager*>(const_cast<InstanceManager*>(this));
    }

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_INSTANCES_H__
