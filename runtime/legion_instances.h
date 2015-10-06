/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "legion_types.h"
#include "legion_utilities.h"
#include "legion_allocation.h"
#include "garbage_collection.h"

namespace LegionRuntime {
  namespace HighLevel {

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
      struct OffsetEntry {
      public:
        OffsetEntry(void) { }
        OffsetEntry(const FieldMask &m,
                    const std::vector<Domain::CopySrcDstField> &f)
          : offset_mask(m), offsets(f) { }
      public:
        FieldMask offset_mask;
        std::vector<Domain::CopySrcDstField> offsets;
      };
    public:
      LayoutDescription(const FieldMask &mask,
                        const Domain &domain,
                        size_t blocking_factor,
                        FieldSpaceNode *owner);
      LayoutDescription(const LayoutDescription &rhs);
      ~LayoutDescription(void);
    public:
      LayoutDescription& operator=(const LayoutDescription &rhs);
      void* operator new(size_t count);
      void operator delete(void *ptr);
    public:
      void compute_copy_offsets(const FieldMask &copy_mask, 
                                PhysicalInstance inst,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                PhysicalInstance inst,
                                std::vector<Domain::CopySrcDstField> &fields);
    public:
      void add_field_info(FieldID fid, unsigned index,
                          size_t offset, size_t field_size);
      const Domain::CopySrcDstField& find_field_info(FieldID fid) const;
      size_t get_layout_size(void) const;
    public:
      bool match_shape(const size_t field_size) const;
      bool match_shape(const std::vector<size_t> &field_sizes, 
                       const size_t bf) const;
    public:
      bool match_layout(const FieldMask &mask, 
                        const size_t vol, const size_t bf) const;
      bool match_layout(const FieldMask &mask,
                        const Domain &d, const size_t bf) const;
      bool match_layout(LayoutDescription *rhs) const;
    public:
      void set_descriptor(FieldDataDescriptor &desc, unsigned fid_idx) const;
    public:
      void pack_layout_description(Serializer &rez, AddressSpaceID target);
      void unpack_layout_description(Deserializer &derez);
      void update_known_nodes(AddressSpaceID target);
      static LayoutDescription* handle_unpack_layout_description(
          Deserializer &derez, AddressSpaceID source, RegionNode *node);
    public:
      static size_t compute_layout_volume(const Domain &d);
    public:
      const FieldMask allocated_fields;
      const size_t blocking_factor;
      const size_t volume;
      FieldSpaceNode *const owner;
    protected:
      std::map<FieldID,Domain::CopySrcDstField> field_infos;
      // Remember these indexes are only good on the local node and
      // have to be transformed when the manager is sent remotely
      std::map<unsigned/*index*/,FieldID> field_indexes;
    protected:
      // Memoized value for matching physical instances
      std::map<unsigned/*offset*/,unsigned/*size*/> offset_size_map;
    protected:
      Reservation layout_lock; 
      std::map<FIELD_TYPE,LegionVector<OffsetEntry>::aligned > 
                                                  memoized_offsets;
      NodeSet known_nodes;
    };
 
    /**
     * \class PhysicalManager
     * This class abstracts a physical instance in memory
     * be it a normal instance or a reduction instance.
     */
    class PhysicalManager : public DistributedCollectable {
    public:
      PhysicalManager(RegionTreeForest *ctx, DistributedID did,
                      AddressSpaceID owner_space, AddressSpaceID local_space,
                      Memory mem, RegionNode *node,
                      PhysicalInstance inst, bool register_now);
      virtual ~PhysicalManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const = 0;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const = 0;
      virtual bool is_reduction_manager(void) const = 0;
      virtual InstanceManager* as_instance_manager(void) const = 0;
      virtual ReductionManager* as_reduction_manager(void) const = 0;
      virtual size_t get_instance_size(void) const = 0;
      virtual void notify_active(void);
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void);
      virtual void notify_invalid(void) = 0;
      virtual DistributedID send_manager(AddressSpaceID target) = 0; 
    public:
      inline PhysicalInstance get_instance(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
#endif
        return instance;
      }
    public:
      RegionTreeForest *const context;
      const Memory memory;
      RegionNode *const region_node;
    protected:
      PhysicalInstance instance;
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
                      Memory mem, PhysicalInstance inst, RegionNode *node,
                      LayoutDescription *desc, Event use_event, 
                      unsigned depth, bool register_now,
                      InstanceFlag flag = NO_INSTANCE_FLAG);
      InstanceManager(const InstanceManager &rhs);
      virtual ~InstanceManager(void);
    public:
      InstanceManager& operator=(const InstanceManager &rhs);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      virtual bool is_reduction_manager(void) const;
      virtual InstanceManager* as_instance_manager(void) const;
      virtual ReductionManager* as_reduction_manager(void) const;
      virtual size_t get_instance_size(void) const;
      virtual void notify_inactive(void);
#ifdef DEBUG_HIGH_LEVEL
      virtual void notify_valid(void);
#endif
      virtual void notify_invalid(void);
    public:
      inline Event get_use_event(void) const { return use_event; }
    public:
      MaterializedView* create_top_view(unsigned depth);
      void compute_copy_offsets(const FieldMask &copy_mask,
                                std::vector<Domain::CopySrcDstField> &fields);
      void compute_copy_offsets(const std::vector<FieldID> &copy_fields,
                                std::vector<Domain::CopySrcDstField> &fields);
    public:
      void set_descriptor(FieldDataDescriptor &desc, unsigned fid_idx) const;
    public:
      virtual DistributedID send_manager(AddressSpaceID target); 
      static void handle_send_manager(Runtime *runtime, 
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      bool match_instance(size_t field_size, const Domain &dom) const;
      bool match_instance(const std::vector<size_t> &fields_sizes,
                          const Domain &dom, const size_t bf) const;
    public:
      bool is_attached_file(void) const;
    public:
      LayoutDescription *const layout;
      // Event that needs to trigger before we can start using
      // this physical instance.
      const Event use_event;
      const unsigned depth;
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
      ReductionManager(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_space, AddressSpaceID local_space,
                       Memory mem, PhysicalInstance inst, 
                       RegionNode *region_node, ReductionOpID redop, 
                       const ReductionOp *op, bool register_now);
      virtual ~ReductionManager(void);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const = 0;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const = 0;
      virtual bool is_reduction_manager(void) const;
      virtual InstanceManager* as_instance_manager(void) const;
      virtual ReductionManager* as_reduction_manager(void) const;
      virtual size_t get_instance_size(void) const = 0;
      virtual void notify_inactive(void);
      virtual void notify_invalid(void);
    public:
      virtual bool is_foldable(void) const = 0;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields) = 0;
      virtual Event issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain) = 0;
      virtual Domain get_pointer_space(void) const = 0;
    public:
      virtual bool is_list_manager(void) const = 0;
      virtual ListReductionManager* as_list_manager(void) const = 0;
      virtual FoldReductionManager* as_fold_manager(void) const = 0;
      virtual Event get_use_event(void) const = 0;
    public:
      virtual DistributedID send_manager(AddressSpaceID target); 
    public:
      static void handle_send_manager(Runtime *runtime,
                                      AddressSpaceID source,
                                      Deserializer &derez);
    public:
      ReductionView* create_view(void);
    public:
      const ReductionOp *const op;
      const ReductionOpID redop;
    };

    /**
     * \class ListReductionManager
     * A class for storing list reduction instances
     */
    class ListReductionManager : public ReductionManager {
    public:
      static const AllocationType alloc_type = LIST_MANAGER_ALLOC;
    public:
      ListReductionManager(RegionTreeForest *ctx, DistributedID did,
                           AddressSpaceID owner_space, 
                           AddressSpaceID local_space,
                           Memory mem, PhysicalInstance inst, 
                           RegionNode *node, ReductionOpID redop, 
                           const ReductionOp *op, Domain dom, bool reg_now);
      ListReductionManager(const ListReductionManager &rhs);
      virtual ~ListReductionManager(void);
    public:
      ListReductionManager& operator=(const ListReductionManager &rhs);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      virtual size_t get_instance_size(void) const;
    public:
      virtual bool is_foldable(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields);
      virtual Event issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual bool is_list_manager(void) const;
      virtual ListReductionManager* as_list_manager(void) const;
      virtual FoldReductionManager* as_fold_manager(void) const;
      virtual Event get_use_event(void) const;
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
      FoldReductionManager(RegionTreeForest *ctx, DistributedID did,
                           AddressSpaceID owner_space, 
                           AddressSpaceID local_space,
                           Memory mem, PhysicalInstance inst, 
                           RegionNode *node, ReductionOpID redop, 
                           const ReductionOp *op, Event use_event,
                           bool register_now);
      FoldReductionManager(const FoldReductionManager &rhs);
      virtual ~FoldReductionManager(void);
    public:
      FoldReductionManager& operator=(const FoldReductionManager &rhs);
    public:
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_accessor(void) const;
      virtual Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        get_field_accessor(FieldID fid) const;
      virtual size_t get_instance_size(void) const;
    public:
      virtual bool is_foldable(void) const;
      virtual void find_field_offsets(const FieldMask &reduce_mask,
          std::vector<Domain::CopySrcDstField> &fields);
      virtual Event issue_reduction(Operation *op,
          const std::vector<Domain::CopySrcDstField> &src_fields,
          const std::vector<Domain::CopySrcDstField> &dst_fields,
          Domain space, Event precondition, bool reduction_fold,
          bool precise_domain);
      virtual Domain get_pointer_space(void) const;
    public:
      virtual bool is_list_manager(void) const;
      virtual ListReductionManager* as_list_manager(void) const;
      virtual FoldReductionManager* as_fold_manager(void) const;
      virtual Event get_use_event(void) const;
    public:
      const Event use_event;
    };

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __LEGION_INSTANCES_H__
