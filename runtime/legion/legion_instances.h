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

#ifndef __LEGION_INSTANCES_H__
#define __LEGION_INSTANCES_H__

#include "runtime.h"
#include "legion_types.h"
#include "legion_utilities.h"
#include "legion_allocation.h"
#include "garbage_collection.h"

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
    class LayoutDescription : public Collectable {
    public:
      LayoutDescription(FieldSpaceNode *owner,
                        const FieldMask &mask,
                        LayoutConstraints *constraints,
                        const std::vector<unsigned> &mask_index_map,
                        const std::vector<CustomSerdezID> &serdez,
          const std::vector<std::pair<FieldID,size_t> > &field_sizes);
      // Used only by the virtual manager
      LayoutDescription(const FieldMask &mask, LayoutConstraints *constraints);
      LayoutDescription(const LayoutDescription &rhs);
      ~LayoutDescription(void);
    public:
      LayoutDescription& operator=(const LayoutDescription &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void log_instance_layout(PhysicalInstance inst) const;
    public:
      void compute_copy_offsets(const FieldMask &copy_mask, 
                                PhysicalInstance inst,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(FieldID copy_field, PhysicalInstance inst,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                PhysicalInstance inst,
                                std::vector<Domain::CopySrcDstField> &fields);
    public:
      bool has_field(FieldID fid) const;
      void has_fields(std::map<FieldID,bool> &fields) const;
      void remove_space_fields(std::set<FieldID> &fields) const;
    public:
      const Domain::CopySrcDstField& find_field_info(FieldID fid) const;
      size_t get_total_field_size(void) const;
      void get_fields(std::vector<FieldID>& fields) const;
      void compute_destroyed_fields(
          std::vector<PhysicalInstance::DestroyedField> &serdez_fields) const;
    public:
      bool match_layout(const LayoutConstraintSet &constraints) const;
      bool match_layout(const LayoutDescription *layout) const;
    public:
      void set_descriptor(FieldDataDescriptor &desc, FieldID fid) const;
    public:
      void pack_layout_description(Serializer &rez, AddressSpaceID target);
      static LayoutDescription* handle_unpack_layout_description(
          Deserializer &derez, AddressSpaceID source, RegionNode *node);
    public:
      const FieldMask allocated_fields;
      LayoutConstraints *const constraints;
      FieldSpaceNode *const owner;
    protected:
      // In order by index of bit mask
#ifdef NEW_INSTANCE_CREATION
      std::vector<Domain::CopySrcDstFieldInfo> field_infos;
#else
      std::vector<Domain::CopySrcDstField> field_infos;
#endif
      // A mapping from FieldIDs to indexes into our field_infos
      std::map<FieldID,unsigned/*index*/> field_indexes;
    protected:
      Reservation layout_lock; 
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
      PhysicalManager(RegionTreeForest *ctx, MemoryManager *memory_manager,
                      LayoutDescription *layout, const PointerConstraint &cons,
                      DistributedID did, AddressSpaceID owner_space, 
                      AddressSpaceID local_space, RegionNode *node,
                      PhysicalInstance inst, const Domain &intance_domain,
                      bool own_domain, RtUserEvent dest, bool register_now);
      virtual ~PhysicalManager(void);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const = 0;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const = 0;
    public:
      inline bool is_reduction_manager(void) const;
      inline bool is_instance_manager(void) const;
      inline bool is_fold_manager(void) const;
      inline bool is_list_manager(void) const;
      inline bool is_virtual_manager(void) const;
      inline InstanceManager* as_instance_manager(void) const;
      inline ReductionManager* as_reduction_manager(void) const;
      inline FoldReductionManager* as_fold_manager(void) const;
      inline ListReductionManager* as_list_manager(void) const;
      inline VirtualManager* as_virtual_manager(void) const;
    public:
      virtual size_t get_instance_size(void) const = 0;
      virtual void notify_active(ReferenceMutator *mutator);
      virtual void notify_inactive(ReferenceMutator *mutator);
      virtual void notify_valid(ReferenceMutator *mutator);
      virtual void notify_invalid(ReferenceMutator *mutator);
    public:
      virtual void send_manager(AddressSpaceID target) = 0; 
      static void handle_manager_request(Deserializer &derez, 
                          Runtime *runtime, AddressSpaceID source);
    public:
      // Support for mapper queries
      virtual bool has_field(FieldID fid) const = 0;
      virtual void has_fields(std::map<FieldID,bool> &fields) const = 0;
      virtual void remove_space_fields(std::set<FieldID> &fields) const = 0;
      inline bool is_normal_instance(void) const 
        { return is_instance_manager(); }
      inline bool is_reduction_instance(void) const
        { return is_reduction_manager(); }
      inline bool is_virtual_instance(void) const
        { return is_virtual_manager(); }
    public:
      // Methods for creating/finding/destroying logical top views
      virtual InstanceView* create_instance_top_view(SingleTask *context,
                                            AddressSpaceID logical_owner) = 0;
      void register_active_context(SingleTask *context);
      void unregister_active_context(SingleTask *context);
    public:
      bool meets_region_tree(const std::vector<LogicalRegion> &regions) const;
      bool meets_regions(const std::vector<LogicalRegion> &regions,
                         bool tight_region_bounds = false) const;
      bool entails(LayoutConstraints *constraints) const;
      bool entails(const LayoutConstraintSet &constraints) const;
      bool conflicts(LayoutConstraints *constraints) const;
      bool conflicts(const LayoutConstraintSet &constraints) const;
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
      void perform_deletion(RtEvent deferred_event) const;
      void set_garbage_collection_priority(MapperID mapper_id, Processor p,
                                           GCPriority priority); 
      static void delete_physical_manager(PhysicalManager *manager);
    public:
      static inline DistributedID encode_instance_did(DistributedID did);
      static inline DistributedID encode_reduction_fold_did(DistributedID did);
      static inline DistributedID encode_reduction_list_did(DistributedID did);
      static inline bool is_instance_did(DistributedID did);
      static inline bool is_reduction_fold_did(DistributedID did);
      static inline bool is_reduction_list_did(DistributedID did);
    public:
      RegionTreeForest *const context;
      MemoryManager *const memory_manager;
      RegionNode *const region_node;
      LayoutDescription *const layout;
      const PhysicalInstance instance;
      const Domain instance_domain;
      const bool own_domain;
      const PointerConstraint pointer_constraint;
    protected:
      std::map<SingleTask*,GenerationID> active_contexts;
    };

    /**
     * \class CopyAcrossHelper
     * A small helper class for performing copies between regions
     * from diferrent region trees
     */
    class CopyAcrossHelper {
    public:
      CopyAcrossHelper(const FieldMask &full)
        : full_mask(full) { }
    public:
      const FieldMask &full_mask;
    public:
      void compute_across_offsets(const FieldMask &src_mask,
             std::vector<Domain::CopySrcDstField> &dst_fields);
    public:
      std::vector<Domain::CopySrcDstField> offsets; 
      LegionDeque<std::pair<FieldMask,FieldMask> >::aligned compressed_cache;
    };

    /**
     * \class InstanceManager
     * A class for managing normal physical instances
     */
    class InstanceManager : public PhysicalManager {
    public:
      static const AllocationType alloc_type = INSTANCE_MANAGER_ALLOC;
    public:
      enum InstanceFlag {
        NO_INSTANCE_FLAG = 0x00000000,
        ATTACH_FILE_FLAG = 0x00000001,
      };
    public:
      InstanceManager(RegionTreeForest *ctx, DistributedID did,
                      AddressSpaceID owner_space, AddressSpaceID local_space,
                      MemoryManager *memory, PhysicalInstance inst, 
                      const Domain &instance_domain, bool own_domain,
                      RegionNode *node, LayoutDescription *desc, 
                      const PointerConstraint &constraint,
                      RtUserEvent destruction_event,
                      bool register_now, ApEvent use_event, 
                      InstanceFlag flag = NO_INSTANCE_FLAG);
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
      virtual size_t get_instance_size(void) const;
    public:
      inline ApEvent get_use_event(void) const { return use_event; }
    public:
      virtual InstanceView* create_instance_top_view(SingleTask *context,
                                            AddressSpaceID logical_owner);
      void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(FieldID fid, 
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                std::vector<Domain::CopySrcDstField> &fields);
      void initialize_across_helper(CopyAcrossHelper *across_helper,
                                    const FieldMask &mask,
                                    const std::vector<unsigned> &src_indexes,
                                    const std::vector<unsigned> &dst_indexes);
    public:
      // Interface to the mapper PhysicalInstance
      virtual bool has_field(FieldID fid) const
        { return layout->has_field(fid); }
      virtual void has_fields(std::map<FieldID,bool> &fields) const
        { return layout->has_fields(fields); } 
      virtual void remove_space_fields(std::set<FieldID> &fields) const
        { return layout->remove_space_fields(fields); }
    public:
      void set_descriptor(FieldDataDescriptor &desc, unsigned fid_idx) const;
    public:
      virtual void send_manager(AddressSpaceID target);
      static void handle_send_manager(Runtime *runtime, 
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      bool is_attached_file(void) const;
    public:
      // Event that needs to trigger before we can start using
      // this physical instance.
      const ApEvent use_event;
    protected:
      // This is monotonic variable that once it becomes true
      // will remain true for the duration of the instance lifetime.
      // If set to true, it should prevent the instance from ever
      // being collected before the context in which it was created
      // is destroyed.
      InstanceFlag instance_flags;
    };

    /**
     * \class ReductionManager
     * An abstract class for managing reduction physical instances
     */
    class ReductionManager : public PhysicalManager {
    public:
      ReductionManager(RegionTreeForest *ctx, DistributedID did, FieldID fid,
                       AddressSpaceID owner_space, AddressSpaceID local_space,
                       MemoryManager *mem, PhysicalInstance inst, 
                       LayoutDescription *description,
                       const PointerConstraint &constraint,
                       const Domain &inst_domain, bool own_domain,
                       RegionNode *region_node, ReductionOpID redop, 
                       const ReductionOp *op, RtUserEvent destruction_event,
                       bool register_now);
      virtual ~ReductionManager(void);
    public:
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_accessor(void) const = 0;
      virtual LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          get_field_accessor(FieldID fid) const = 0;
    public:
      virtual size_t get_instance_size(void) const = 0;
    public:
      virtual bool is_foldable(void) const = 0;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields) = 0;
      virtual ApEvent issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          RegionTreeNode *dst, ApEvent precondition, bool reduction_fold,
          bool precise_domain, RegionTreeNode *intersect) = 0;
      virtual Domain get_pointer_space(void) const = 0;
    public:
      virtual ApEvent get_use_event(void) const = 0;
    public:
      // Support for mapper queries
      virtual bool has_field(FieldID fid) const;
      virtual void has_fields(std::map<FieldID,bool> &fields) const;
      virtual void remove_space_fields(std::set<FieldID> &fields) const;
    public:
      virtual void send_manager(AddressSpaceID target);
    public:
      static void handle_send_manager(Runtime *runtime,
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      virtual InstanceView* create_instance_top_view(SingleTask *context,
                                            AddressSpaceID logical_owner);
    public:
      const ReductionOp *const op;
      const ReductionOpID redop;
      const FieldID logical_field;
    };

    /**
     * \class ListReductionManager
     * A class for storing list reduction instances
     */
    class ListReductionManager : public ReductionManager {
    public:
      static const AllocationType alloc_type = LIST_MANAGER_ALLOC;
    public:
      ListReductionManager(RegionTreeForest *ctx, DistributedID did,FieldID fid,
                           AddressSpaceID owner_space, 
                           AddressSpaceID local_space,
                           MemoryManager *mem, PhysicalInstance inst, 
                           LayoutDescription *description,
                           const PointerConstraint &constraint,
                           const Domain &inst_domain, bool own_domain,
                           RegionNode *node, ReductionOpID redop, 
                           const ReductionOp *op, Domain dom,
                           RtUserEvent dest_event, bool register_now);
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
      virtual size_t get_instance_size(void) const;
    public:
      virtual bool is_foldable(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields);
      virtual ApEvent issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          RegionTreeNode *dst, ApEvent precondition, bool reduction_fold,
          bool precise_domain, RegionTreeNode *intersect);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual ApEvent get_use_event(void) const;
    protected:
      const Domain ptr_space;
    };

    /**
     * \class FoldReductionManager
     * A class for representing fold reduction instances
     */
    class FoldReductionManager : public ReductionManager {
    public:
      static const AllocationType alloc_type = FOLD_MANAGER_ALLOC;
    public:
      FoldReductionManager(RegionTreeForest *ctx, DistributedID did,FieldID fid,
                           AddressSpaceID owner_space, 
                           AddressSpaceID local_space,
                           MemoryManager *mem, PhysicalInstance inst, 
                           LayoutDescription *description,
                           const PointerConstraint &constraint,
                           const Domain &inst_dom, bool own_dom,
                           RegionNode *node, ReductionOpID redop, 
                           const ReductionOp *op, ApEvent use_event,
                           RtUserEvent dest_event, bool register_now);
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
      virtual size_t get_instance_size(void) const;
    public:
      virtual bool is_foldable(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields);
      virtual ApEvent issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          RegionTreeNode *dst, ApEvent precondition, bool reduction_fold,
          bool precise_domain, RegionTreeNode *intersect);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual ApEvent get_use_event(void) const;
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
                     DistributedID did, AddressSpaceID local_space);
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
      virtual size_t get_instance_size(void) const;
      virtual void send_manager(AddressSpaceID target);
      virtual bool has_field(FieldID fid) const;
      virtual void has_fields(std::map<FieldID,bool> &fields) const;
      virtual void remove_space_fields(std::set<FieldID> &fields) const;
      virtual InstanceView* create_instance_top_view(SingleTask *context,
                                            AddressSpaceID logical_owner);
    public:
      static inline VirtualManager* get_virtual_instance(void)
        { return get_singleton(); }
      static void initialize_virtual_instance(Runtime *runtime,
                                              DistributedID did);
    protected:
      static inline VirtualManager*& get_singleton(void)
      {
        static VirtualManager *singleton = NULL;
        return singleton;
      }
    };

    /**
     * \class InstanceBuilder 
     * A helper for building physical instances of logical regions
     */
    class InstanceBuilder {
    public:
      InstanceBuilder(const std::vector<LogicalRegion> &regs,
                      const LayoutConstraintSet &cons,
                      MemoryManager *memory, UniqueID cid)
        : regions(regs), constraints(cons), memory_manager(memory),
          creator_id(cid), ancestor(NULL), instance_domain(Domain::NO_DOMAIN), 
          own_domain(false), redop_id(0), reduction_op(NULL), valid(false) { }
    public:
      size_t compute_needed_size(RegionTreeForest *forest);
      PhysicalManager* create_physical_instance(RegionTreeForest *forest);
    protected:
      void initialize(RegionTreeForest *forest);
      void compute_ancestor_and_domain(RegionTreeForest *forest);
      RegionNode* find_common_ancestor(RegionNode *one, RegionNode *two) const;
    protected:
      void compute_new_parameters(void);
      void compute_old_parameters(void);
    protected:
      const std::vector<LogicalRegion> &regions;
      LayoutConstraintSet constraints;
      MemoryManager *const memory_manager;
      const UniqueID creator_id;
    protected:
      RegionNode *ancestor;
      Domain instance_domain;
      bool own_domain;
      std::vector<std::pair<FieldID,size_t> > field_sizes;
      std::vector<unsigned> mask_index_map;
      std::vector<CustomSerdezID> serdez;
      FieldMask instance_mask;
      ReductionOpID redop_id;
      const ReductionOp *reduction_op;
#ifndef NEW_INSTANCE_CREATION
      std::vector<size_t> sizes_only;
      size_t block_size;
#endif
    public:
      bool valid;
    };

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID PhysicalManager::encode_instance_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x0UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID PhysicalManager::encode_reduction_fold_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x1UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID PhysicalManager::encode_reduction_list_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, 0x2UL);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool PhysicalManager::is_instance_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x3) == 0x0);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool PhysicalManager::is_reduction_fold_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x3) == 0x1);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool PhysicalManager::is_reduction_list_did(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      return ((LEGION_DISTRIBUTED_HELP_DECODE(did) & 0x3) == 0x2);
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
