/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_FIELD_SPACE_H__
#define __LEGION_FIELD_SPACE_H__

#include "legion/api/data.h"
#include "legion/contexts/context.h"
#include "legion/kernel/metatask.h"
#include "legion/utilities/buffers.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FieldSpaceNode
     * Represent a generic field space that can be
     * pointed at by nodes in the region trees.
     */
    class FieldSpaceNode : public Heapify<FieldSpaceNode, LONG_LIFETIME>,
                           public DistributedCollectable {
    public:
      enum FieldAllocationState {
        FIELD_ALLOC_INVALID,     // field_infos is invalid
        FIELD_ALLOC_READ_ONLY,   // field_infos is valid and read-only
        FIELD_ALLOC_PENDING,     // about to have allocation privileges
                                 // (owner-only)
        FIELD_ALLOC_EXCLUSIVE,   // field_infos is valid and can allocate
        FIELD_ALLOC_COLLECTIVE,  // same as above but exactly one total CR
                                 // context
      };
    public:
      struct FieldInfo {
      public:
        FieldInfo(void);
        FieldInfo(
            size_t size, unsigned id, CustomSerdezID sid, Provenance* prov,
            bool loc = false, bool collect = false);
        FieldInfo(
            ApEvent ready, unsigned id, CustomSerdezID sid, Provenance* prov,
            bool loc = false, bool collect = false);
        FieldInfo(const FieldInfo& rhs);
        FieldInfo(FieldInfo&& rhs) noexcept;
        ~FieldInfo(void);
      public:
        FieldInfo& operator=(const FieldInfo& rhs);
        FieldInfo& operator=(FieldInfo&& rhs) noexcept;
      public:
        void serialize(Serializer& rez) const;
        void deserialize(Deserializer& derez);
      public:
        size_t field_size;
        ApEvent size_ready;
        unsigned idx;
        CustomSerdezID serdez_id;
        Provenance* provenance;
        bool collective;
        bool local;
      };
      struct FindTargetsFunctor {
      public:
        FindTargetsFunctor(std::deque<AddressSpaceID>& t) : targets(t) { }
      public:
        void apply(AddressSpaceID target);
      private:
        std::deque<AddressSpaceID>& targets;
      };
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_FIELD_SPACE_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(void) = default;
        SemanticRequestArgs(
            FieldSpaceNode* proxy, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(false, false), proxy_this(proxy),
            tag(t), source(src)
        { }
        void execute(void) const;
      public:
        FieldSpaceNode* proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
      struct SemanticFieldRequestArgs
        : public LgTaskArgs<SemanticFieldRequestArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_FIELD_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticFieldRequestArgs(void) = default;
        SemanticFieldRequestArgs(
            FieldSpaceNode* proxy, FieldID f, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticFieldRequestArgs>(false, false),
            proxy_this(proxy), fid(f), tag(t), source(src)
        { }
        void execute(void) const;
      public:
        FieldSpaceNode* proxy_this;
        FieldID fid;
        SemanticTag tag;
        AddressSpaceID source;
      };
      struct DeferRequestFieldInfoArgs
        : public LgTaskArgs<DeferRequestFieldInfoArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_FIELD_INFOS_TASK_ID;
      public:
        DeferRequestFieldInfoArgs(void) = default;
        DeferRequestFieldInfoArgs(
            const FieldSpaceNode* n, std::map<FieldID, FieldInfo>* c,
            AddressSpaceID src, RtUserEvent t)
          : LgTaskArgs<DeferRequestFieldInfoArgs>(false, false), proxy_this(n),
            copy(c), source(src), to_trigger(t)
        { }
        void execute(void) const;
      public:
        const FieldSpaceNode* proxy_this;
        std::map<FieldID, FieldInfo>* copy;
        AddressSpaceID source;
        RtUserEvent to_trigger;
      };
    public:
      FieldSpaceNode(
          FieldSpace sp, RtEvent initialized, CollectiveMapping* mapping,
          Provenance* provenance);
      FieldSpaceNode(
          FieldSpace sp, RtEvent initialized, CollectiveMapping* mapping,
          Provenance* provenance, Deserializer& derez);
      FieldSpaceNode(const FieldSpaceNode& rhs) = delete;
      virtual ~FieldSpaceNode(void);
    public:
      FieldSpaceNode& operator=(const FieldSpaceNode& rhs) = delete;
      AddressSpaceID get_owner_space(void) const;
      static AddressSpaceID get_owner_space(FieldSpace handle);
    public:
      virtual void notify_local(void) override { }
    public:
      void attach_semantic_information(
          SemanticTag tag, AddressSpaceID source, const void* buffer,
          size_t size, bool is_mutable, bool local_only);
      void attach_semantic_information(
          FieldID fid, SemanticTag tag, AddressSpaceID source,
          const void* buffer, size_t size, bool is_mutable, bool local_only);
      bool retrieve_semantic_information(
          SemanticTag tag, const void*& result, size_t& size, bool can_fail,
          bool wait_until);
      bool retrieve_semantic_information(
          FieldID fid, SemanticTag tag, const void*& result, size_t& size,
          bool can_fail, bool wait_until);
      void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* result,
          size_t size, bool is_mutable, RtUserEvent ready);
      void send_semantic_field_info(
          AddressSpaceID target, FieldID fid, SemanticTag tag,
          const void* result, size_t size, bool is_mutable,
          RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT);
      void process_semantic_request(
          SemanticTag tag, AddressSpaceID source, bool can_fail,
          bool wait_until, RtUserEvent ready);
      void process_semantic_field_request(
          FieldID fid, SemanticTag tag, AddressSpaceID source, bool can_fail,
          bool wait_until, RtUserEvent ready);
    public:
      RtEvent create_allocator(
          AddressSpaceID source,
          RtUserEvent ready = RtUserEvent::NO_RT_USER_EVENT,
          bool sharded_owner_context = false, bool owner_shard = false);
      RtEvent destroy_allocator(
          AddressSpaceID source, bool sharded_owner_context = false,
          bool owner_shard = false);
    public:
      void initialize_fields(
          const std::vector<size_t>& sizes,
          const std::vector<FieldID>& resulting_fields,
          CustomSerdezID serdez_id, Provenance* provenance,
          bool collective = false);
      void initialize_fields(
          ApEvent sizes_ready, const std::vector<FieldID>& resulting_fields,
          CustomSerdezID serdez_id, Provenance* provenance,
          bool collective = false);
      RtEvent allocate_field(
          FieldID fid, size_t size, CustomSerdezID serdez_id,
          Provenance* provenance, bool sharded_non_owner = false);
      RtEvent allocate_field(
          FieldID fid, ApEvent size_ready, CustomSerdezID serdez_id,
          Provenance* provenance, bool sharded_non_owner = false);
      RtEvent allocate_fields(
          const std::vector<size_t>& sizes, const std::vector<FieldID>& fids,
          CustomSerdezID serdez_id, Provenance* provenance,
          bool sharded_non_owner = false);
      RtEvent allocate_fields(
          ApEvent sizes_ready, const std::vector<FieldID>& fids,
          CustomSerdezID serdez_id, Provenance* provenance,
          bool sharded_non_owner = false);
      void update_field_size(
          FieldID fid, size_t field_size, std::set<RtEvent>& update_events,
          AddressSpaceID source);
      void free_field(
          FieldID fid, AddressSpaceID source, std::set<RtEvent>& applied,
          bool sharded_non_owner = false);
      void free_fields(
          const std::vector<FieldID>& to_free, AddressSpaceID source,
          std::set<RtEvent>& applied, bool sharded_non_owner = false);
      void free_field_indexes(
          const std::vector<FieldID>& to_free, RtEvent freed_event,
          bool sharded_non_owner = false);
    public:
      bool allocate_local_fields(
          const std::vector<FieldID>& fields, const std::vector<size_t>& sizes,
          CustomSerdezID serdez_id, const std::set<unsigned>& indexes,
          std::vector<unsigned>& new_indexes, Provenance* provenance);
      void free_local_fields(
          const std::vector<FieldID>& to_free,
          const std::vector<unsigned>& indexes,
          const CollectiveMapping* mapping);
      void update_local_fields(
          const std::vector<FieldID>& fields, const std::vector<size_t>& sizes,
          const std::vector<CustomSerdezID>& serdez_ids,
          const std::vector<unsigned>& indexes, Provenance* provenance);
      void remove_local_fields(const std::vector<FieldID>& to_removes);
    public:
      bool has_field(FieldID fid);
      size_t get_field_size(FieldID fid);
      CustomSerdezID get_field_serdez(FieldID fid);
      void get_all_fields(std::vector<FieldID>& to_set);
      void get_all_regions(std::set<LogicalRegion>& regions);
      void get_field_set(
          const FieldMask& mask, TaskContext* context,
          std::set<FieldID>& to_set) const;
      void get_field_set(
          const FieldMask& mask, TaskContext* context,
          std::vector<FieldID>& to_set) const;
      void get_field_set(
          const FieldMask& mask, const std::set<FieldID>& basis,
          std::set<FieldID>& to_set) const;
    public:
      FieldMask get_field_mask(const std::set<FieldID>& fields) const;
      unsigned get_field_index(FieldID fid) const;
      void get_field_indexes(
          const std::vector<FieldID>& fields,
          std::vector<unsigned>& indexes) const;
    public:
      void compute_field_layout(
          const std::vector<FieldID>& create_fields,
          std::vector<size_t>& field_sizes,
          std::vector<unsigned>& mask_index_map,
          std::vector<CustomSerdezID>& serdez, FieldMask& instance_mask);
    public:
      InstanceRef create_external_instance(
          const std::set<FieldID>& priv_fields,
          const std::vector<FieldID>& fields, RegionNode* node, AttachOp* op);
      PhysicalManager* create_external_manager(
          PhysicalInstance inst, ApEvent ready_event, size_t instance_footprint,
          LayoutConstraintSet& constraints,
          const std::vector<FieldID>& field_set,
          const std::vector<size_t>& field_sizes, const FieldMask& file_mask,
          const std::vector<unsigned>& mask_index_map, LgEvent unique_event,
          RegionNode* node, const std::vector<CustomSerdezID>& serdez,
          DistributedID did, CollectiveMapping* collective_mapping = nullptr);
    public:
      LayoutDescription* find_layout_description(
          const FieldMask& field_mask, unsigned num_dims,
          const LayoutConstraintSet& constraints);
      LayoutDescription* find_layout_description(
          const FieldMask& field_mask, LayoutConstraints* constraints);
      LayoutDescription* create_layout_description(
          const FieldMask& layout_mask, const unsigned total_dims,
          LayoutConstraints* constraints, const std::vector<unsigned>& indexes,
          const std::vector<FieldID>& fids, const std::vector<size_t>& sizes,
          const std::vector<CustomSerdezID>& serdez);
      LayoutDescription* register_layout_description(LayoutDescription* desc);
    public:
      void send_node(AddressSpaceID target);
    public:
      // Help with debug printing
      char* to_string(const FieldMask& mask, TaskContext* ctx) const;
    protected:
      // Assume we are already holding the node lock
      // when calling these methods
      int allocate_index(RtEvent& ready_event, bool initializing = false);
      void free_index(unsigned index, RtEvent free_event);
    public:
      void invalidate_layouts(
          unsigned index, std::set<RtEvent>& applied, AddressSpaceID source,
          bool need_lock = true);
    public:
      RtEvent request_field_infos_copy(
          std::map<FieldID, FieldInfo>* copy, AddressSpaceID source,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT) const;
      void record_read_only_infos(const std::map<FieldID, FieldInfo>& infos);
      void process_allocator_response(Deserializer& derez);
      void process_allocator_invalidation(
          RtUserEvent done, bool flush, bool merge);
      bool process_allocator_flush(Deserializer& derez);
      void process_allocator_free(Deserializer& derez, AddressSpaceID source);
    protected:
      bool allocate_local_indexes(
          CustomSerdezID serdez, const std::vector<size_t>& sizes,
          const std::set<unsigned>& current_indexes,
          std::vector<unsigned>& new_indexes);
    public:
      const FieldSpace handle;
      Provenance* const provenance;
      RtEvent initialized;
    private:
      mutable LocalLock node_lock;
      std::map<FieldID, FieldInfo> field_infos;  // depends on allocation_state
      // Local field sizes
      std::vector<std::pair<size_t, CustomSerdezID> > local_index_infos;
    private:
      // Keep track of the layouts associated with this field space
      // Index them by their hash of their field mask to help
      // differentiate them.
      std::map<LEGION_FIELD_MASK_FIELD_TYPE, lng::list<LayoutDescription*> >
          layouts;
    private:
      lng::map<SemanticTag, SemanticInfo> semantic_info;
      lng::map<std::pair<FieldID, SemanticTag>, SemanticInfo>
          semantic_field_info;
    private:
      // Track which node is the owner for allocation privileges
      FieldAllocationState allocation_state;
      // For all normal (aka non-local) fields we track which indexes in the
      // field mask have not been allocated. Only valid on the allocation owner
      FieldMask unallocated_indexes;
      // Use a list here so that we cycle through all the indexes
      // that have been freed before we reuse to avoid false aliasing
      // We may pull things out from the middle though
      std::list<std::pair<unsigned, RtEvent> > available_indexes;
      // Keep track of the nodes with remote copies of field_infos
      mutable std::set<AddressSpaceID> remote_field_infos;
      // An event for recording when we are available for allocation
      // on the owner node in the case we had to send invalidations
      RtEvent pending_field_allocation;
      // Total number of outstanding allocators
      unsigned outstanding_allocators;
      // Total number of outstanding invalidations (owner node only)
      unsigned outstanding_invalidations;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FIELD_SPACE_H__
