/* Copyright 2020 Stanford University, NVIDIA Corporation
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
                                PhysicalManager *manager,
                                std::vector<CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                PhysicalManager *manager,
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
               LegionList<std::pair<FieldMask,FieldMask> >::aligned> comp_cache;
    }; 
 
    /**
     * \class PhysicalManager
     * This class abstracts a physical instance in memory
     * be it a normal instance or a reduction instance.
     */
    class PhysicalManager : public DistributedCollectable {
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
    public:
      PhysicalManager(RegionTreeForest *ctx, MemoryManager *memory_manager,
                      LayoutDescription *layout, const PointerConstraint &cons,
                      DistributedID did, AddressSpaceID owner_space, 
                      FieldSpaceNode *node, PhysicalInstance inst, 
                      const size_t footprint, 
                      IndexSpaceExpression *index_domain, 
                      RegionTreeID tree_id, bool register_now);
      virtual ~PhysicalManager(void);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const = 0;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const = 0;
    public:
      void log_instance_creation(UniqueID creator_id, Processor proc,
                     const std::vector<LogicalRegion> &regions) const;
    public:
      inline bool is_reduction_manager(void) const;
      inline bool is_instance_manager(void) const;
      inline bool is_fold_manager(void) const;
      inline bool is_list_manager(void) const;
      inline bool is_virtual_manager(void) const;
      inline bool is_external_instance(void) const;
      inline InstanceManager* as_instance_manager(void) const;
      inline ReductionManager* as_reduction_manager(void) const;
      inline FoldReductionManager* as_fold_manager(void) const;
      inline ListReductionManager* as_list_manager(void) const;
      inline VirtualManager* as_virtual_manager(void) const;
    public:
      virtual ApEvent get_use_event(void) const = 0;
      virtual ApEvent get_unique_event(void) const = 0;
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual ApEvent fill_from(FillView *fill_view, ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                CopyAcrossHelper *across_helper = NULL,
                                FieldMaskSet<FillView> *tracing_srcs=NULL,
                                FieldMaskSet<InstanceView> *tracing_dsts=NULL);
      virtual ApEvent copy_from(PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                CopyAcrossHelper *across_helper = NULL,
                                FieldMaskSet<InstanceView> *tracing_srcs=NULL,
                                FieldMaskSet<InstanceView> *tracing_dsts=NULL);
      virtual void compute_copy_offsets(const FieldMask &copy_mask,
                                        std::vector<CopySrcDstField> &fields);
    public:
      virtual void send_manager(AddressSpaceID target) = 0; 
      static void handle_manager_request(Deserializer &derez, 
                          Runtime *runtime, AddressSpaceID source);
    public:
      // Interface to the mapper PhysicalInstance
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
      inline size_t get_instance_size(void) const { return instance_footprint; }
    public:
      inline bool is_normal_instance(void) const 
        { return is_instance_manager(); }
      inline bool is_reduction_instance(void) const
        { return is_reduction_manager(); }
      inline bool is_virtual_instance(void) const
        { return is_virtual_manager(); }
    public:
      // Methods for creating/finding/destroying logical top views
      virtual InstanceView* create_instance_top_view(InnerContext *context,
                                            AddressSpaceID logical_owner) = 0;
      void register_active_context(InnerContext *context);
      void unregister_active_context(InnerContext *context);
    public:
      bool meets_region_tree(const std::vector<LogicalRegion> &regions) const;
      bool meets_regions(const std::vector<LogicalRegion> &regions,
                         bool tight_region_bounds = false) const;
      bool meets_expression(IndexSpaceExpression *expr, 
                            bool tight_bounds = false) const;
      bool entails(LayoutConstraints *constraints,
                   const LayoutConstraint **failed_constraint) const;
      bool entails(const LayoutConstraintSet &constraints,
                   const LayoutConstraint **failed_constraint) const;
      bool conflicts(LayoutConstraints *constraints,
                     const LayoutConstraint **conflict_constraint) const;
      bool conflicts(const LayoutConstraintSet &constraints,
                     const LayoutConstraint **conflict_constraint) const;
    public:
      inline PhysicalInstance get_instance(void) const
      {
#ifdef DEBUG_LEGION
        assert(instance.exists());
#endif
        return instance;
      }
      inline Memory get_memory(void) const { return memory_manager->memory; }
    public:
      bool acquire_instance(ReferenceSource source, ReferenceMutator *mutator);
      void perform_deletion(RtEvent deferred_event);
      void force_deletion(void);
      void set_garbage_collection_priority(MapperID mapper_id, Processor p,
                                           GCPriority priority); 
      RtEvent detach_external_instance(void);
    public:
      void defer_collect_user(CollectableView *view, ApEvent term_event,
                              RtEvent collect, std::set<ApEvent> &to_collect, 
                              bool &add_ref, bool &remove_ref);
      void find_shutdown_preconditions(std::set<ApEvent> &preconditions);
    public:
      static inline DistributedID encode_instance_did(DistributedID did,
                                                      bool external);
      static inline DistributedID encode_reduction_fold_did(DistributedID did);
      static inline DistributedID encode_reduction_list_did(DistributedID did);
      static inline bool is_instance_did(DistributedID did);
      static inline bool is_reduction_fold_did(DistributedID did);
      static inline bool is_reduction_list_did(DistributedID did);
      static inline bool is_external_did(DistributedID did);
      static ApEvent fetch_metadata(PhysicalInstance inst, ApEvent use_event);
    public:
      RegionTreeForest *const context;
      MemoryManager *const memory_manager;
      FieldSpaceNode *const field_space_node;
      LayoutDescription *const layout;
      const PhysicalInstance instance;
      const size_t instance_footprint;
      IndexSpaceExpression *instance_domain;
      const RegionTreeID tree_id;
      const PointerConstraint pointer_constraint;
    protected:
      mutable LocalLock inst_lock;
      std::set<InnerContext*> active_contexts;
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
      LegionDeque<std::pair<FieldMask,FieldMask> >::aligned compressed_cache;
    };

    /**
     * \class InstanceManager
     * A class for managing normal physical instances
     */
    class InstanceManager : public PhysicalManager,
                            public LegionHeapify<InstanceManager> {
    public:
      static const AllocationType alloc_type = INSTANCE_MANAGER_ALLOC;
    public:
      struct DeferInstanceManagerArgs : 
        public LgTaskArgs<DeferInstanceManagerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_INSTANCE_MANAGER_TASK_ID;
      public:
        DeferInstanceManagerArgs(DistributedID d, AddressSpaceID own, Memory m,
            PhysicalInstance i, size_t f, bool local, IndexSpaceExpression *lx,
            bool is, IndexSpace dh, IndexSpaceExprID dx, FieldSpace h, 
            RegionTreeID tid, LayoutConstraintID l, PointerConstraint &p, 
            ApEvent use);
      public:
        const DistributedID did;
        const AddressSpaceID owner;
        const Memory mem;
        const PhysicalInstance inst;
        const size_t footprint;
        const bool local_is;
        const bool domain_is;
        IndexSpaceExpression *local_expr;
        const IndexSpace domain_handle;
        const IndexSpaceExprID domain_expr;
        const FieldSpace handle;
        const RegionTreeID tree_id;
        const LayoutConstraintID layout_id;
        PointerConstraint *const pointer;
        const ApEvent use_event;
      };
    public:
      InstanceManager(RegionTreeForest *ctx, DistributedID did,
                      AddressSpaceID owner_space,
                      MemoryManager *memory, PhysicalInstance inst, 
                      IndexSpaceExpression *instance_domain,
                      FieldSpaceNode *node, RegionTreeID tree_id,
                      LayoutDescription *desc, 
                      const PointerConstraint &constraint,
                      bool register_now, size_t footprint,
                      ApEvent use_event, bool external_instance);
      InstanceManager(const InstanceManager &rhs);
      virtual ~InstanceManager(void);
    public:
      InstanceManager& operator=(const InstanceManager &rhs);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      virtual ApEvent get_use_event(void) const { return use_event; }
      virtual ApEvent get_unique_event(void) const { return unique_event; }
    public:
      virtual ApEvent fill_from(FillView *fill_view, ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                CopyAcrossHelper *across_helper = NULL,
                                FieldMaskSet<FillView> *tracing_srcs=NULL,
                                FieldMaskSet<InstanceView> *tracing_dsts=NULL);
      virtual ApEvent copy_from(PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                CopyAcrossHelper *across_helper = NULL,
                                FieldMaskSet<InstanceView> *tracing_srcs=NULL,
                                FieldMaskSet<InstanceView> *tracing_dsts=NULL);
    public:
      virtual InstanceView* create_instance_top_view(InnerContext *context,
                                            AddressSpaceID logical_owner); 
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
          FieldSpaceNode *space_node, RegionTreeID tree_id,
          LayoutConstraints *constraints, ApEvent use_event,
          PointerConstraint &pointer_constraint);
    public:
      // Event that needs to trigger before we can start using
      // this physical instance.
      const ApEvent use_event; 
      // Unique identifier event that is common across nodes
      const ApEvent unique_event;
    };

    /**
     * \class ReductionManager
     * An abstract class for managing reduction physical instances
     */
    class ReductionManager : public PhysicalManager {
    public:
      struct DeferReductionManagerArgs : 
        public LgTaskArgs<DeferReductionManagerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_REDUCTION_MANAGER_TASK_ID;
      public:
        DeferReductionManagerArgs(DistributedID d, AddressSpaceID own, Memory m,
            PhysicalInstance i, size_t f, bool local, IndexSpaceExpression *lx,
            bool is, IndexSpace dh, IndexSpaceExprID dx, FieldSpace h, 
            RegionTreeID tid, LayoutConstraintID l, PointerConstraint &p, 
            ApEvent use, bool fold, const Domain &ptr, ReductionOpID r);
      public:
        const DistributedID did;
        const AddressSpaceID owner;
        const Memory mem;
        const PhysicalInstance inst;
        const size_t footprint;
        const bool local_is;
        const bool domain_is;
        IndexSpaceExpression *const local_expr;
        const IndexSpace domain_handle;
        const IndexSpaceExprID domain_expr;
        const FieldSpace handle;
        const RegionTreeID tree_id;
        const LayoutConstraintID layout_id;
        PointerConstraint *const pointer;
        const ApEvent use_event;
        const bool foldable;
        const Domain ptr_space;
        const ReductionOpID redop;
      };
    public:
      ReductionManager(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_space,
                       MemoryManager *mem, PhysicalInstance inst, 
                       LayoutDescription *description,
                       const PointerConstraint &constraint,
                       IndexSpaceExpression *inst_domain,
                       FieldSpaceNode *field_node, 
                       RegionTreeID tree_id, ReductionOpID redop, 
                       const ReductionOp *op, ApEvent use_event,
                       size_t footprint, bool register_now);
      virtual ~ReductionManager(void);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const = 0;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const = 0;
    public:
      virtual bool is_foldable(void) const = 0;
      virtual Domain get_pointer_space(void) const = 0;
    public:
      virtual ApEvent get_use_event(void) const { return use_event; }
      virtual ApEvent get_unique_event(void) const { return unique_event; }
    public:
      virtual void send_manager(AddressSpaceID target);
    public:
      static void handle_send_manager(Runtime *runtime,
                                      AddressSpaceID source,
                                      Deserializer &derez);
      static void handle_defer_manager(const void *args, Runtime *runtime);
      static void create_remote_manager(Runtime *runtime, DistributedID did,
          AddressSpaceID owner_space, Memory mem, PhysicalInstance inst,
          size_t inst_footprint, IndexSpaceExpression *inst_domain,
          FieldSpaceNode *space_node, RegionTreeID tree_id,
          LayoutConstraints *constraints, ApEvent use_event,
          PointerConstraint &pointer_constraint, bool foldable,
          const Domain &ptr_space, ReductionOpID redop);
    public:
      virtual InstanceView* create_instance_top_view(InnerContext *context,
                                            AddressSpaceID logical_owner);
    public:
      const ReductionOp *const op;
      const ReductionOpID redop;
      const ApEvent use_event;
      const ApEvent unique_event;
    protected:
      mutable LocalLock manager_lock;
    };

    /**
     * \class ListReductionManager
     * A class for storing list reduction instances
     */
    class ListReductionManager : public ReductionManager,
                                 public LegionHeapify<ListReductionManager> {
    public:
      static const AllocationType alloc_type = LIST_MANAGER_ALLOC;
    public:
      ListReductionManager(RegionTreeForest *ctx, DistributedID did,
                           AddressSpaceID owner_space, 
                           MemoryManager *mem, PhysicalInstance inst, 
                           LayoutDescription *description,
                           const PointerConstraint &constraint,
                           IndexSpaceExpression *inst_domain,
                           FieldSpaceNode *node, 
                           RegionTreeID tree_id, ReductionOpID redop, 
                           const ReductionOp *op, Domain dom,
                           ApEvent use_event,size_t fooprint,bool register_now);
      ListReductionManager(const ListReductionManager &rhs);
      virtual ~ListReductionManager(void);
    public:
      ListReductionManager& operator=(const ListReductionManager &rhs);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      virtual bool is_foldable(void) const;
      virtual void compute_copy_offsets(const FieldMask &reduce_mask,
          std::vector<CopySrcDstField> &fields);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual ApEvent copy_from(PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                CopyAcrossHelper *across_helper = NULL,
                                FieldMaskSet<InstanceView> *tracing_srcs=NULL,
                                FieldMaskSet<InstanceView> *tracing_dsts=NULL);
    protected:
      const Domain ptr_space;
    };

    /**
     * \class FoldReductionManager
     * A class for representing fold reduction instances
     */
    class FoldReductionManager : public ReductionManager,
                                 public LegionHeapify<FoldReductionManager> {
    public:
      static const AllocationType alloc_type = FOLD_MANAGER_ALLOC;
    public:
      FoldReductionManager(RegionTreeForest *ctx, DistributedID did,
                           AddressSpaceID owner_space, 
                           MemoryManager *mem, PhysicalInstance inst, 
                           LayoutDescription *description,
                           const PointerConstraint &constraint,
                           IndexSpaceExpression *inst_dom,
                           FieldSpaceNode *node, 
                           RegionTreeID tree_id, ReductionOpID redop, 
                           const ReductionOp *op, ApEvent use_event,
                           size_t footprint, bool register_now);
      FoldReductionManager(const FoldReductionManager &rhs);
      virtual ~FoldReductionManager(void);
    public:
      FoldReductionManager& operator=(const FoldReductionManager &rhs);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const;
    public:
      virtual bool is_foldable(void) const;
      virtual Domain get_pointer_space(void) const;
    public:
      virtual ApEvent copy_from(PhysicalManager *manager, ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                CopyAcrossHelper *across_helper = NULL,
                                FieldMaskSet<InstanceView> *tracing_srcs=NULL,
                                FieldMaskSet<InstanceView> *tracing_dsts=NULL);
    public:
      const ApEvent use_event;
    };

    /**
     * \class VirtualManager
     * This is a singleton class of which there will be exactly one
     * on every node in the machine. The virtual manager class will
     * represent all the virtual virtual/composite instances.
     */
    class VirtualManager : public PhysicalManager {
    public:
      VirtualManager(RegionTreeForest *ctx, LayoutDescription *desc,
                     const PointerConstraint &constraint,
                     DistributedID did);
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
      virtual ApEvent get_use_event(void) const;
      virtual ApEvent get_unique_event(void) const;
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
                      MemoryManager *memory, UniqueID cid)
        : regions(regs), constraints(cons), runtime(rt), memory_manager(memory),
          creator_id(cid), instance(PhysicalInstance::NO_INST), 
          field_space_node(NULL), instance_domain(NULL), tree_id(0),
          redop_id(0), reduction_op(NULL), realm_layout(NULL), valid(false) { }
      virtual ~InstanceBuilder(void);
    public:
      void initialize(RegionTreeForest *forest);
      PhysicalManager* create_physical_instance(RegionTreeForest *forest,
                        LayoutConstraintKind *unsat_kind,
                        unsigned *unsat_index, size_t *footprint = NULL);
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
      size_t instance_volume;
      RegionTreeID tree_id;
      // Mapping from logical field order to layout order
      std::vector<unsigned> mask_index_map;
      std::vector<size_t> field_sizes;
      std::vector<CustomSerdezID> serdez;
      FieldMask instance_mask;
      ReductionOpID redop_id;
      const ReductionOp *reduction_op;
      Realm::InstanceLayoutGeneric *realm_layout;
    public:
      bool valid;
    };

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID PhysicalManager::encode_instance_did(
                                               DistributedID did, bool external)
    //--------------------------------------------------------------------------
    {
      if (external)
        return LEGION_DISTRIBUTED_HELP_ENCODE(did, INSTANCE_MANAGER_DC | 0x10);
      else
        return LEGION_DISTRIBUTED_HELP_ENCODE(did, INSTANCE_MANAGER_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID PhysicalManager::encode_reduction_fold_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REDUCTION_FOLD_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID PhysicalManager::encode_reduction_list_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REDUCTION_LIST_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool PhysicalManager::is_instance_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xF) == 
                                                        INSTANCE_MANAGER_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool PhysicalManager::is_reduction_fold_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xF) == 
                                                    REDUCTION_FOLD_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool PhysicalManager::is_reduction_list_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0xF) == 
                                                    REDUCTION_LIST_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool PhysicalManager::is_external_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x10) == 0x10);
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::is_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
      return (is_reduction_fold_did(did) || is_reduction_list_did(did));
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::is_instance_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_instance_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::is_fold_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_fold_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::is_list_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_list_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::is_virtual_manager(void) const
    //--------------------------------------------------------------------------
    {
      return (did == 0);
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::is_external_instance(void) const
    //--------------------------------------------------------------------------
    {
      return is_external_did(did);
    }

    //--------------------------------------------------------------------------
    inline InstanceManager* PhysicalManager::as_instance_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_instance_manager());
#endif
      return static_cast<InstanceManager*>(const_cast<PhysicalManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline ReductionManager* PhysicalManager::as_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_reduction_manager());
#endif
      return static_cast<ReductionManager*>(const_cast<PhysicalManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline VirtualManager* PhysicalManager::as_virtual_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_virtual_manager());
#endif
      return static_cast<VirtualManager*>(const_cast<PhysicalManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline ListReductionManager* PhysicalManager::as_list_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_list_manager());
#endif
      return static_cast<ListReductionManager*>(
              const_cast<PhysicalManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline FoldReductionManager* PhysicalManager::as_fold_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_fold_manager());
#endif
      return static_cast<FoldReductionManager*>(
              const_cast<PhysicalManager*>(this));
    }

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_INSTANCES_H__
