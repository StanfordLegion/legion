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

#ifndef __LEGION_INDEX_SPACE_H__
#define __LEGION_INDEX_SPACE_H__

#include "legion/api/physical_region_impl.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/expression.h"
#include "legion/utilities/buffers.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct FieldDataDescriptor
     * A small helper class for performing dependent
     * partitioning operations
     */
    struct FieldDataDescriptor {
    public:
      inline bool operator<(const FieldDataDescriptor& rhs) const
      {
        return (color < rhs.color);
      }
      inline void serialize(Serializer& rez) const
      {
        rez.serialize(domain);
        rez.serialize(color);
        rez.serialize(inst);
        rez.serialize(unique);
        rez.serialize(space);
      }
      inline void deserialize(Deserializer& derez)
      {
        derez.deserialize(domain);
        derez.deserialize(color);
        derez.deserialize(inst);
        derez.deserialize(unique);
        derez.deserialize(space);
      }
    public:
      // Index space user events for these domains are already
      // added by the operation that populates these structs
      Domain domain;
      DomainPoint color;
      PhysicalInstance inst;
      // These are not necessary for functional correctness but
      // are used by the profiler to record the instances and
      // names of domains being used by the dependent partitions
      LgEvent unique;       // unique event for inst
      DistributedID space;  // index space for domain
    };

    struct DeppartResult {
    public:
      inline bool operator<(const DeppartResult& rhs) const
      {
        return (color < rhs.color);
      }
    public:
      Domain domain;
      LegionColor color;
    };

    /**
     * \class PieceIteratorImplT
     * This is the templated version of this class that is
     * instantiated for each cominbation of type and dimensoinality
     */
    template<int DIM, typename T>
    class PieceIteratorImplT : public PieceIteratorImpl {
    public:
      PieceIteratorImplT(
          const void* piece_list, size_t piece_list_size,
          IndexSpaceNodeT<DIM, T>* privilege_node);
      virtual ~PieceIteratorImplT(void) { }
      virtual int get_next(int index, Domain& next_piece);
    protected:
      std::vector<Rect<DIM, T> > pieces;
    };

    // This is a small helper class for converting realm points when the
    // types don't naturally align with the underling index space type
    template<int DIM, typename TYPELIST>
    struct RealmPointConverter {
      // Convert To
      static inline void convert_to(
          const DomainPoint& point, void* realm_point, const TypeTag type_tag,
          const char* context)
      {
        // Compute the type tag for this particular type with the same DIM
        const TypeTag tag = NT_TemplateHelper::template encode_tag<
            DIM, typename TYPELIST::HEAD>();
        if (tag == type_tag)
        {
          Realm::Point<DIM, typename TYPELIST::HEAD>* target =
              static_cast<Realm::Point<DIM, typename TYPELIST::HEAD>*>(
                  realm_point);
          *target = point;
        }
        else
          RealmPointConverter<DIM, typename TYPELIST::TAIL>::convert_to(
              point, realm_point, type_tag, context);
      }
      // Convert From
      static inline void convert_from(
          const void* realm_point, TypeTag type_tag, DomainPoint& point,
          const char* context)
      {
        // Compute the type tag for this particular type with the same DIM
        const TypeTag tag =
            NT_TemplateHelper::encode_tag<DIM, typename TYPELIST::HEAD>();
        if (tag == type_tag)
        {
          const Realm::Point<DIM, typename TYPELIST::HEAD>* source =
              static_cast<const Realm::Point<DIM, typename TYPELIST::HEAD>*>(
                  realm_point);
          point = *source;
        }
        else
          RealmPointConverter<DIM, typename TYPELIST::TAIL>::convert_from(
              realm_point, type_tag, point, context);
      }
    };

    // Specialization for the end-of-list cases
    template<int DIM>
    struct RealmPointConverter<DIM, Realm::DynamicTemplates::TypeListTerm> {
      static inline void convert_to(
          const DomainPoint& point, void* realm_point, const TypeTag type_tag,
          const char* context)
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Dynamic type mismatch in " << context << ".";
        error.raise();
      }
      static inline void convert_from(
          const void* realm_point, TypeTag type_tag, DomainPoint& point,
          const char* context)
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Dynamic type mismatch in " << context << ".";
        error.raise();
      }
    };

    // This is a small helper class for converting realm index spaces when
    // the types don't naturally align with the underlying index space type
    template<int DIM, typename TYPELIST>
    struct RealmSpaceConverter {
      static inline void convert_to(
          const Domain& domain, void* realm_is, const TypeTag type_tag,
          const char* context)
      {
        // Compute the type tag for this particular type with the same DIM
        const TypeTag tag =
            NT_TemplateHelper::encode_tag<DIM, typename TYPELIST::HEAD>();
        if (tag == type_tag)
        {
          Realm::IndexSpace<DIM, typename TYPELIST::HEAD>* target =
              static_cast<Realm::IndexSpace<DIM, typename TYPELIST::HEAD>*>(
                  realm_is);
          *target = domain;
        }
        else
          RealmSpaceConverter<DIM, typename TYPELIST::TAIL>::convert_to(
              domain, realm_is, type_tag, context);
      }
    };

    // Specialization for end-of-list cases
    template<int DIM>
    struct RealmSpaceConverter<DIM, Realm::DynamicTemplates::TypeListTerm> {
      static inline void convert_to(
          const Domain& domain, void* realm_is, const TypeTag type_tag,
          const char* context)
      {
        {
          Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          err << "Dynamic type mismatch in '" << context << "'";
          err.raise();
        }
      }
    };

    /**
     * \class IndexTreeNode
     * The abstract base class for nodes in the index space trees.
     */
    class IndexTreeNode : public ValidDistributedCollectable {
    public:
      IndexTreeNode(
          unsigned depth, LegionColor color, DistributedID did,
          RtEvent init_event, CollectiveMapping* mapping,
          Provenance* provenance, bool tree_valid);
      virtual ~IndexTreeNode(void);
    public:
      virtual IndexTreeNode* get_parent(void) const = 0;
      virtual LegionColor get_colors(std::vector<LegionColor>& colors) = 0;
    public:
      virtual bool is_index_space_node(void) const = 0;
#ifdef LEGION_DEBUG
      virtual IndexSpaceNode* as_index_space_node(void) = 0;
      virtual IndexPartNode* as_index_part_node(void) = 0;
#else
      inline IndexSpaceNode* as_index_space_node(void);
      inline IndexPartNode* as_index_part_node(void);
#endif
      virtual AddressSpaceID get_owner_space(void) const = 0;
    public:
      void attach_semantic_information(
          SemanticTag tag, AddressSpaceID source, const void* buffer,
          size_t size, bool is_mutable, bool local_only);
      bool retrieve_semantic_information(
          SemanticTag tag, const void*& result, size_t& size, bool can_fail,
          bool wait_until);
      virtual void send_semantic_request(
          AddressSpaceID target, SemanticTag tag, bool can_fail,
          bool wait_until, RtUserEvent ready) = 0;
      virtual void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable, RtUserEvent ready) = 0;
    public:
      const unsigned depth;
      const LegionColor color;
      Provenance* const provenance;
    public:
      RtEvent initialized;
      NodeSet<LONG_LIFETIME> child_creation;
    protected:
      mutable LocalLock node_lock;
    protected:
      std::map<IndexTreeNode*, bool> dominators;
    protected:
      lng::map<SemanticTag, SemanticInfo> semantic_info;
    };

    /**
     * \class IndexSpaceNode
     * A class for representing a generic index space node.
     */
    class IndexSpaceNode : public IndexTreeNode,
                           public IndexSpaceExpression {
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_INDEX_SPACE_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(void) = default;
        SemanticRequestArgs(
            IndexSpaceNode* proxy, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(false, false), proxy_this(proxy),
            tag(t), source(src)
        { }
        void execute(void) const;
      public:
        IndexSpaceNode* proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
      struct DeferChildArgs : public LgTaskArgs<DeferChildArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_INDEX_SPACE_DEFER_CHILD_TASK_ID;
      public:
        DeferChildArgs(void) = default;
        DeferChildArgs(
            IndexSpaceNode* proxy, LegionColor child, DistributedID* tar,
            RtUserEvent trig, AddressSpaceID src)
          : LgTaskArgs<DeferChildArgs>(false, false), proxy_this(proxy),
            child_color(child), target(tar), to_trigger(trig), source(src)
        { }
        void execute(void) const;
      public:
        IndexSpaceNode* proxy_this;
        LegionColor child_color;
        DistributedID* target;
        RtUserEvent to_trigger;
        AddressSpaceID source;
      };
      class IndexSpaceSetFunctor {
      public:
        IndexSpaceSetFunctor(AddressSpaceID src, IndexSpaceSet& r)
          : source(src), rez(r)
        { }
      public:
        void apply(AddressSpaceID target);
      public:
        const AddressSpaceID source;
        IndexSpaceSet& rez;
      };
    public:
      IndexSpaceNode(
          IndexSpace handle, IndexPartNode* parent, LegionColor color,
          IndexSpaceExprID expr_id, RtEvent initialized, unsigned depth,
          Provenance* provenance, CollectiveMapping* mapping, bool tree_valid);
      IndexSpaceNode(const IndexSpaceNode& rhs) = delete;
      virtual ~IndexSpaceNode(void);
    public:
      IndexSpaceNode& operator=(const IndexSpaceNode& rhs) = delete;
    public:
      virtual void notify_invalid(void) override;
      virtual void notify_local(void) override;
    public:
      virtual bool is_set(void) const override
      {
        return index_space_set.load();
      }
      virtual bool is_index_space_node(void) const override;
#ifdef LEGION_DEBUG
      virtual IndexSpaceNode* as_index_space_node(void) override;
      virtual IndexPartNode* as_index_part_node(void) override;
#endif
      virtual AddressSpaceID get_owner_space(void) const override;
      static AddressSpaceID get_owner_space(IndexSpace handle);
    public:
      virtual IndexTreeNode* get_parent(void) const override;
      virtual LegionColor get_colors(std::vector<LegionColor>& colors) override;
    public:
      virtual void send_semantic_request(
          AddressSpaceID target, SemanticTag tag, bool can_fail,
          bool wait_until, RtUserEvent ready) override;
      virtual void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable, RtUserEvent ready) override;
      void process_semantic_request(
          SemanticTag tag, AddressSpaceID source, bool can_fail,
          bool wait_until, RtUserEvent ready);
    public:
      bool has_color(const LegionColor color);
      LegionColor generate_color(LegionColor suggestion = INVALID_COLOR);
      void release_color(LegionColor color);
      // If you pass can_fail=true here then the node comes back with
      // a resource REGION_TREE_REF to keep it alive
      IndexPartNode* get_child(
          const LegionColor c, RtEvent* defer = nullptr, bool can_fail = false);
      void add_child(IndexPartNode* child);
      void remove_child(const LegionColor c);
      size_t get_num_children(void) const;
      RtEvent get_ready_event(void);
    public:
      bool are_disjoint(LegionColor c1, LegionColor c2);
      void record_remote_child(IndexPartition pid, LegionColor part_color);
    public:
      void send_node(AddressSpaceID target, bool recurse, bool valid = true);
      void pack_node(
          Serializer& rez, AddressSpaceID target, bool recurse, bool valid);
      bool invalidate_root(
          AddressSpaceID source, std::set<RtEvent>& applied,
          const CollectiveMapping* mapping);
    public:
      virtual Domain get_tight_domain(void) override = 0;
      [[nodiscard]] virtual ApEvent get_loose_domain(
          Domain& domain, ApUserEvent& done_event) override = 0;
      virtual RtEvent add_sparsity_map_references(
          const Domain& domain, unsigned references) = 0;
      virtual void record_index_space_user(ApEvent user) override = 0;
      virtual bool set_domain(
          const Domain& domain, ApEvent is_ready, bool take_ownership,
          bool broadcast = false, bool initializing = false) = 0;
      virtual bool set_output_union(
          const std::map<DomainPoint, DomainPoint>& sizes) = 0;
      virtual void tighten_index_space(void) override = 0;
      virtual bool check_empty(void) override = 0;
      virtual void pack_expression(
          Serializer& rez, AddressSpaceID target) override;
      virtual void skip_unpack_expression(Deserializer& derez) const override;
    public:
#ifdef LEGION_DEBUG
      virtual bool is_valid(void) override
      {
        return ValidDistributedCollectable::is_global();
      }
#endif
      virtual DistributedID get_distributed_id(void) const override
      {
        return did;
      }
      virtual void add_canonical_reference(DistributedID source) override;
      virtual bool remove_canonical_reference(DistributedID source) override;
      virtual bool try_add_live_reference(void) override;
      virtual void add_base_expression_reference(
          ReferenceSource source, unsigned count = 1) override;
      virtual void add_nested_expression_reference(
          DistributedID source, unsigned count = 1) override;
      virtual bool remove_base_expression_reference(
          ReferenceSource source, unsigned count = 1) override;
      virtual bool remove_nested_expression_reference(
          DistributedID source, unsigned count = 1) override;
      virtual void add_tree_expression_reference(
          DistributedID source, unsigned count = 1) override;
      virtual bool remove_tree_expression_reference(
          DistributedID source, unsigned count = 1) override;
    public:
      virtual IndexSpaceNode* create_node(
          IndexSpace handle, RtEvent initialized, Provenance* provenance,
          CollectiveMapping* mapping,
          IndexSpaceExprID expr_id = 0) override = 0;
      virtual IndexSpaceExpression* create_from_rectangles(
          const local::set<Domain>& rectangles) override = 0;
      virtual PieceIteratorImpl* create_piece_iterator(
          const void* piece_list, size_t piece_list_size,
          IndexSpaceNode* privilege_node) override = 0;
      virtual bool is_below_in_tree(
          IndexPartNode* p, LegionColor& child) const override;
    public:
      virtual ApEvent compute_pending_space(
          Operation* op, const std::vector<IndexSpace>& handles,
          bool is_union) = 0;
      virtual ApEvent compute_pending_space(
          Operation* op, IndexPartition handle, bool is_union) = 0;
      virtual ApEvent compute_pending_difference(
          Operation* op, IndexSpace initial,
          const std::vector<IndexSpace>& handles) = 0;
      virtual void get_index_space_domain(void* realm_is, TypeTag type_tag) = 0;
      virtual size_t get_volume(void) override = 0;
      virtual size_t get_num_dims(void) const = 0;
      virtual bool contains_point(
          const void* realm_point, TypeTag type_tag) = 0;
      virtual bool contains_point(const DomainPoint& point) = 0;
      virtual bool has_interfering_point(
          const std::vector<std::pair<DomainPoint, Domain> >& tests,
          DomainPoint& interfering_point, DomainPoint to_skip) = 0;
    public:
      virtual LegionColor get_max_linearized_color(void) = 0;
      virtual LegionColor linearize_color(const DomainPoint& point) = 0;
      virtual LegionColor linearize_color(
          const void* realm_color, TypeTag type_tag) = 0;
      virtual void delinearize_color(
          LegionColor color, void* realm_color, TypeTag type_tag) = 0;
      virtual bool contains_color(
          LegionColor color, bool report_error = false) = 0;
      virtual void instantiate_colors(std::vector<LegionColor>& colors) = 0;
      virtual Domain get_color_space_domain(void) = 0;
      virtual DomainPoint get_domain_point_color(void) const = 0;
      virtual DomainPoint delinearize_color_to_point(LegionColor c) = 0;
      virtual size_t compute_color_offset(LegionColor color) = 0;
    public:
      bool intersects_with(IndexSpaceNode* rhs, bool compute = true);
      bool intersects_with(IndexPartNode* rhs, bool compute = true);
      bool dominates(IndexSpaceNode* rhs);
    public:
      virtual void pack_index_space(
          Serializer& rez, unsigned references) const = 0;
      virtual bool unpack_index_space(
          Deserializer& derez, AddressSpaceID source) = 0;
    public:
      virtual ApEvent create_equal_children(
          Operation* op, IndexPartNode* partition, size_t granularity) = 0;
      virtual ApEvent create_by_union(
          Operation* op, IndexPartNode* partition, IndexPartNode* left,
          IndexPartNode* right) = 0;
      virtual ApEvent create_by_intersection(
          Operation* op, IndexPartNode* partition, IndexPartNode* left,
          IndexPartNode* right) = 0;
      virtual ApEvent create_by_intersection(
          Operation* op, IndexPartNode* partition,
          // Left is implicit "this"
          IndexPartNode* right, const bool dominates = false) = 0;
      virtual ApEvent create_by_difference(
          Operation* op, IndexPartNode* partition, IndexPartNode* left,
          IndexPartNode* right) = 0;
      // Called on color space and not parent
      virtual ApEvent create_by_restriction(
          IndexPartNode* partition, const void* transform, const void* extent,
          int partition_dim) = 0;
      virtual ApEvent create_by_domain(
          Operation* op, IndexPartNode* partition,
          const std::map<DomainPoint, FutureImpl*>& futures,
          const Domain& future_map_domain, bool perform_intersections) = 0;
      virtual ApEvent create_by_weights(
          Operation* op, IndexPartNode* partition,
          const std::map<DomainPoint, FutureImpl*>& weights,
          size_t granularity) = 0;
      virtual ApEvent create_by_field(
          Operation* op, FieldID fid, IndexPartNode* partition,
          const std::vector<FieldDataDescriptor>& instances,
          std::vector<DeppartResult>* results, ApEvent instances_ready) = 0;
      virtual ApEvent create_by_image(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          std::vector<FieldDataDescriptor>& instances,
          ApEvent instances_ready) = 0;
      virtual ApEvent create_by_image_range(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          std::vector<FieldDataDescriptor>& instances,
          ApEvent instances_ready) = 0;
      virtual ApEvent create_by_preimage(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          const std::vector<FieldDataDescriptor>& instances,
          const std::map<DomainPoint, Domain>* remote_targets,
          std::vector<DeppartResult>* results, ApEvent instances_ready) = 0;
      virtual ApEvent create_by_preimage_range(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          const std::vector<FieldDataDescriptor>& instances,
          const std::map<DomainPoint, Domain>* remote_targets,
          std::vector<DeppartResult>* results, ApEvent instances_ready) = 0;
      virtual ApEvent create_association(
          Operation* op, FieldID fid, IndexSpaceNode* range,
          const std::vector<FieldDataDescriptor>& instances,
          ApEvent instances_ready) = 0;
      virtual size_t get_coordinate_size(bool range) const = 0;
    public:
      virtual Realm::InstanceLayoutGeneric* create_hdf5_layout(
          const std::vector<FieldID>& field_ids,
          const std::vector<size_t>& field_sizes,
          const std::vector<std::string>& field_files,
          const OrderingConstraint& dimension_order) = 0;
    public:
      virtual void validate_slicing(
          const std::vector<IndexSpace>& slice_spaces, MultiTask* task,
          MapperManager* mapper) = 0;
      virtual void log_launch_space(UniqueID op_id) = 0;
      virtual IndexSpace create_shard_space(
          ShardingFunction* func, ShardID shard, IndexSpace shard_space,
          const Domain& shard_domain,
          const std::vector<DomainPoint>& shard_points,
          Provenance* provenance) = 0;
      virtual void compute_range_shards(
          ShardingFunction* func, IndexSpace shard_space,
          const std::vector<DomainPoint>& shard_points,
          const Domain& shard_domain, std::set<ShardID>& range_shards) = 0;
      virtual bool has_shard_participants(
          ShardingFunction* func, ShardID shard, IndexSpace shard_space,
          const std::vector<DomainPoint>& shard_points,
          const Domain& shard_domain) = 0;
    public:
      virtual EqKDTree* create_equivalence_set_kd_tree(
          size_t total_shards = 1) = 0;
      virtual void invalidate_equivalence_set_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          std::vector<RtEvent>& invalidated, bool move_to_previous) = 0;
      virtual void invalidate_shard_equivalence_set_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          std::vector<RtEvent>& invalidated,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard) = 0;
      virtual void find_trace_local_sets_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          unsigned req_index, ShardID local_shard,
          std::map<EquivalenceSet*, unsigned>& current_sets) = 0;
      virtual void find_shard_trace_local_sets_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          unsigned req_index, std::map<EquivalenceSet*, unsigned>& current_sets,
          local::map<ShardID, FieldMask>& remote_shards,
          ShardID local_shard) = 0;
    public:
      const IndexSpace handle;
      IndexPartNode* const parent;
    protected:
      // Must hold the node lock when accessing these data structures
      std::map<LegionColor, IndexPartNode*> color_map;
      std::map<LegionColor, IndexPartition> remote_colors;
      std::set<RegionNode*> logical_nodes;
      std::set<std::pair<LegionColor, LegionColor> > disjoint_subsets;
      std::set<std::pair<LegionColor, LegionColor> > aliased_subsets;
      std::deque<ApEvent> index_space_users;
    protected:
      static constexpr uintptr_t REMOVED_CHILD = 0xdead;
      Color next_uncollected_color;
      // Event for when the sparsity map is ready
      ApEvent index_space_valid;
      // Event to signal for anything waiting for the index space to be set
      RtUserEvent index_space_ready;
      std::atomic<bool> index_space_set;
      std::atomic<bool> index_space_tight;
    };

    /**
     * \class IndexSpaceNodeT
     * A templated class for handling any templated realm calls
     * associated with realm index spaces
     */
    template<int DIM, typename T>
    class IndexSpaceNodeT
      : public IndexSpaceNode,
        public Heapify<IndexSpaceNodeT<DIM, T>, LONG_LIFETIME> {
    public:
      IndexSpaceNodeT(
          IndexSpace handle, IndexPartNode* parent, LegionColor color,
          IndexSpaceExprID expr_id, RtEvent init, unsigned depth,
          Provenance* provenance, CollectiveMapping* mapping, bool tree_valid);
      IndexSpaceNodeT(const IndexSpaceNodeT& rhs) = delete;
      virtual ~IndexSpaceNodeT(void);
    public:
      IndexSpaceNodeT& operator=(const IndexSpaceNodeT& rhs) = delete;
    public:
      DomainT<DIM, T> get_tight_index_space(void);
      // Return event is when the result index space is safe to use
      // The done event must be triggered after the index space is
      // done being used if it is not a no-event
      [[nodiscard]] ApEvent get_loose_index_space(
          DomainT<DIM, T>& result, ApUserEvent& done_event);
      bool set_realm_index_space(
          const Realm::IndexSpace<DIM, T>& value, ApEvent valid,
          bool initialization = false, bool broadcast = false,
          AddressSpaceID source = std::numeric_limits<AddressSpaceID>::max());
      RtEvent get_realm_index_space_ready(bool need_tight_result);
    public:
      virtual bool is_sparse(void) override;
      virtual Domain get_tight_domain(void) override;
      [[nodiscard]] virtual ApEvent get_loose_domain(
          Domain& domain, ApUserEvent& done_event) override;
      virtual RtEvent add_sparsity_map_references(
          const Domain& domain, unsigned references) override;
      virtual void record_index_space_user(ApEvent user) override;
      virtual bool set_domain(
          const Domain& domain, ApEvent is_ready, bool take_ownership,
          bool broadcast = false, bool initializing = false) override;
      virtual bool set_output_union(
          const std::map<DomainPoint, DomainPoint>& sizes) override;
      virtual void tighten_index_space(void) override;
      virtual DistributedID record_profiler_expression(void) override;
      virtual bool check_empty(void) override;
      virtual IndexSpaceNode* create_node(
          IndexSpace handle, RtEvent initialized, Provenance* provenance,
          CollectiveMapping* mapping, IndexSpaceExprID expr_id = 0) override;
      virtual IndexSpaceExpression* create_from_rectangles(
          const local::set<Domain>& rectangles) override;
      virtual PieceIteratorImpl* create_piece_iterator(
          const void* piece_list, size_t piece_list_size,
          IndexSpaceNode* privilege_node) override;
    public:
      void log_index_space_points(const Realm::IndexSpace<DIM, T>& space) const;
    public:
      virtual ApEvent compute_pending_space(
          Operation* op, const std::vector<IndexSpace>& handles,
          bool is_union) override;
      virtual ApEvent compute_pending_space(
          Operation* op, IndexPartition handle, bool is_union) override;
      virtual ApEvent compute_pending_difference(
          Operation* op, IndexSpace initial,
          const std::vector<IndexSpace>& handles) override;
      virtual void get_index_space_domain(
          void* realm_is, TypeTag type_tag) override;
      virtual size_t get_volume(void) override;
      virtual size_t get_num_dims(void) const override;
      virtual bool contains_point(
          const void* realm_point, TypeTag type_tag) override;
      virtual bool contains_point(const DomainPoint& point) override;
      virtual bool has_interfering_point(
          const std::vector<std::pair<DomainPoint, Domain> >& tests,
          DomainPoint& interfering_point, DomainPoint to_skip) override;
    public:
      virtual LegionColor get_max_linearized_color(void) override;
      virtual LegionColor linearize_color(const DomainPoint& point) override;
      virtual LegionColor linearize_color(
          const void* realm_color, TypeTag type_tag) override;
      LegionColor linearize_color(const Point<DIM, T>& color);
      virtual void delinearize_color(
          LegionColor color, void* realm_color, TypeTag type_tag) override;
      void delinearize_color(LegionColor color, Point<DIM, T>& point);
      virtual bool contains_color(
          LegionColor color, bool report_error = false) override;
      virtual void instantiate_colors(
          std::vector<LegionColor>& colors) override;
      virtual Domain get_color_space_domain(void) override;
      virtual DomainPoint get_domain_point_color(void) const override;
      virtual DomainPoint delinearize_color_to_point(LegionColor c) override;
      virtual size_t compute_color_offset(LegionColor color) override;
    public:
      virtual void pack_index_space(
          Serializer& rez, unsigned references) const override;
      virtual bool unpack_index_space(
          Deserializer& derez, AddressSpaceID source) override;
    public:
      virtual ApEvent create_equal_children(
          Operation* op, IndexPartNode* partition, size_t granularity) override;
      virtual ApEvent create_by_union(
          Operation* op, IndexPartNode* partition, IndexPartNode* left,
          IndexPartNode* right) override;
      virtual ApEvent create_by_intersection(
          Operation* op, IndexPartNode* partition, IndexPartNode* left,
          IndexPartNode* right) override;
      virtual ApEvent create_by_intersection(
          Operation* op, IndexPartNode* partition,
          // Left is implicit "this"
          IndexPartNode* right, const bool dominates = false) override;
      virtual ApEvent create_by_difference(
          Operation* op, IndexPartNode* partition, IndexPartNode* left,
          IndexPartNode* right) override;
      // Called on color space and not parent
      virtual ApEvent create_by_restriction(
          IndexPartNode* partition, const void* transform, const void* extent,
          int partition_dim) override;
      template<int N>
      ApEvent create_by_restriction_helper(
          IndexPartNode* partition, const Realm::Matrix<N, DIM, T>& transform,
          const Realm::Rect<N, T>& extent);
      virtual ApEvent create_by_domain(
          Operation* op, IndexPartNode* partition,
          const std::map<DomainPoint, FutureImpl*>& futures,
          const Domain& future_map_domain, bool perform_intersections) override;
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_domain_helper(
          Operation* op, IndexPartNode* partition,
          const std::map<DomainPoint, FutureImpl*>& futures,
          const Domain& future_map_domain, bool perform_intersections);
      virtual ApEvent create_by_weights(
          Operation* op, IndexPartNode* partition,
          const std::map<DomainPoint, FutureImpl*>& weights,
          size_t granularity) override;
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_weight_helper(
          Operation* op, IndexPartNode* partition,
          const std::map<DomainPoint, FutureImpl*>& weights,
          size_t granularity);
      virtual ApEvent create_by_field(
          Operation* op, FieldID fid, IndexPartNode* partition,
          const std::vector<FieldDataDescriptor>& instances,
          std::vector<DeppartResult>* results,
          ApEvent instances_ready) override;
      template<int COLOR_DIM, typename COLOR_T>
      ApEvent create_by_field_helper(
          Operation* op, FieldID fid, IndexPartNode* partition,
          const std::vector<FieldDataDescriptor>& instances,
          std::vector<DeppartResult>* results, ApEvent instances_ready);
      virtual ApEvent create_by_image(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          std::vector<FieldDataDescriptor>& instances,
          ApEvent instances_ready) override;
      template<int DIM2, typename T2>
      ApEvent create_by_image_helper(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          std::vector<FieldDataDescriptor>& instances, ApEvent instances_ready);
      virtual ApEvent create_by_image_range(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          std::vector<FieldDataDescriptor>& instances,
          ApEvent instances_ready) override;
      template<int DIM2, typename T2>
      ApEvent create_by_image_range_helper(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          std::vector<FieldDataDescriptor>& instances, ApEvent instances_ready);
      virtual ApEvent create_by_preimage(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          const std::vector<FieldDataDescriptor>& instances,
          const std::map<DomainPoint, Domain>* remote_targets,
          std::vector<DeppartResult>* results,
          ApEvent instances_ready) override;
      template<int DIM2, typename T2>
      ApEvent create_by_preimage_helper(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          const std::vector<FieldDataDescriptor>& instances,
          const std::map<DomainPoint, Domain>* remote_targets,
          std::vector<DeppartResult>* results, ApEvent instances_ready);
      virtual ApEvent create_by_preimage_range(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          const std::vector<FieldDataDescriptor>& instances,
          const std::map<DomainPoint, Domain>* remote_targets,
          std::vector<DeppartResult>* results,
          ApEvent instances_ready) override;
      template<int DIM2, typename T2>
      ApEvent create_by_preimage_range_helper(
          Operation* op, FieldID fid, IndexPartNode* partition,
          IndexPartNode* projection,
          const std::vector<FieldDataDescriptor>& instances,
          const std::map<DomainPoint, Domain>* remote_targets,
          std::vector<DeppartResult>* results, ApEvent instances_ready);
      virtual ApEvent create_association(
          Operation* op, FieldID fid, IndexSpaceNode* range,
          const std::vector<FieldDataDescriptor>& instances,
          ApEvent instances_ready) override;
      template<int DIM2, typename T2>
      ApEvent create_association_helper(
          Operation* op, FieldID fid, IndexSpaceNode* range,
          const std::vector<FieldDataDescriptor>& instances,
          ApEvent instances_ready);
      void prepare_broadcast_results(
          IndexPartNode* partition, std::vector<DomainT<DIM, T> >& subspaces,
          std::vector<DeppartResult>& results, ApEvent& result);
      virtual size_t get_coordinate_size(bool range) const override;
    public:
      virtual Realm::InstanceLayoutGeneric* create_hdf5_layout(
          const std::vector<FieldID>& field_ids,
          const std::vector<size_t>& field_sizes,
          const std::vector<std::string>& field_files,
          const OrderingConstraint& dimension_order) override;
    public:
      virtual IndexSpaceExpression* inline_union(
          IndexSpaceExpression* rhs) override;
      virtual IndexSpaceExpression* inline_union(
          const SetView<IndexSpaceExpression*>& exprs) override;
      virtual IndexSpaceExpression* inline_intersection(
          IndexSpaceExpression* rhs) override;
      virtual IndexSpaceExpression* inline_intersection(
          const SetView<IndexSpaceExpression*>& exprs) override;
      virtual IndexSpaceExpression* inline_subtraction(
          IndexSpaceExpression* rhs) override;
      virtual uint64_t get_canonical_hash(void) override;
    public:
      virtual ApEvent issue_fill(
          Operation* op, const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& dst_fields,
          const void* fill_value, size_t fill_size, UniqueID fill_uid,
          FieldSpace handle, RegionTreeID tree_id, ApEvent precondition,
          PredEvent pred_guard, LgEvent unique_event, CollectiveKind collective,
          bool record_effect, int priority = 0, bool replay = false) override;
      virtual ApEvent issue_copy(
          Operation* op, const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent pred_guard, LgEvent src_unique,
          LgEvent dst_unique, CollectiveKind collective, bool record_effect,
          int priority = 0, bool replay = false) override;
      virtual CopyAcrossUnstructured* create_across_unstructured(
          const std::map<Reservation, bool>& reservations,
          const bool compute_preimages,
          const bool shadow_indirections) override;
      virtual Realm::InstanceLayoutGeneric* create_layout(
          const LayoutConstraintSet& constraints,
          const std::vector<FieldID>& field_ids,
          const std::vector<size_t>& field_sizes, bool compact,
          void** piece_list = nullptr, size_t* piece_list_size = nullptr,
          size_t* num_pieces = nullptr, size_t base_alignment = 32) override;
      virtual IndexSpaceExpression* create_layout_expression(
          const void* piece_list, size_t piece_list_size) override;
      virtual bool meets_layout_expression(
          IndexSpaceExpression* expr, bool tight_bounds, const void* piece_list,
          size_t piece_list_size, const Domain* padding_delta) override;
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
          SmallPointerVector<IndexSpaceExpression, true>& expressions) override;
      virtual KDTree* get_sparsity_map_kd_tree(void) override;
    public:
      virtual void validate_slicing(
          const std::vector<IndexSpace>& slice_spaces, MultiTask* task,
          MapperManager* mapper) override;
      virtual void log_launch_space(UniqueID op_id) override;
      virtual IndexSpace create_shard_space(
          ShardingFunction* func, ShardID shard, IndexSpace shard_space,
          const Domain& shard_domain,
          const std::vector<DomainPoint>& shard_points,
          Provenance* provenance) override;
      virtual void compute_range_shards(
          ShardingFunction* func, IndexSpace shard_space,
          const std::vector<DomainPoint>& shard_points,
          const Domain& shard_domain, std::set<ShardID>& range_shards) override;
      virtual bool has_shard_participants(
          ShardingFunction* func, ShardID shard, IndexSpace shard_space,
          const std::vector<DomainPoint>& shard_points,
          const Domain& shard_domain) override;
    public:
      virtual EqKDTree* create_equivalence_set_kd_tree(
          size_t total_shards = 1) override;
      virtual void initialize_equivalence_set_kd_tree(
          EqKDTree* tree, EquivalenceSet* set, const FieldMask& mask,
          ShardID local_shard, bool current) override;
      virtual void compute_equivalence_sets(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          const std::vector<EqSetTracker*>& trackers,
          const std::vector<AddressSpaceID>& tracker_spaces,
          std::vector<unsigned>& new_tracker_references,
          op::FieldMaskMap<EquivalenceSet>& eq_sets,
          std::vector<RtEvent>& pending_sets,
          op::FieldMaskMap<EqKDTree>& subscriptions,
          op::FieldMaskMap<EqKDTree>& to_create,
          op::map<EqKDTree*, Domain>& creation_rects,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0) override;
      virtual unsigned record_output_equivalence_set(
          EqKDTree* tree, LocalLock* tree_lock, EquivalenceSet* set,
          const FieldMask& mask, EqSetTracker* tracker,
          AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& subscriptions,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0) override;
      virtual void invalidate_equivalence_set_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          std::vector<RtEvent>& invalidated, bool move_to_previous) override;
      virtual void invalidate_shard_equivalence_set_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          std::vector<RtEvent>& invalidated,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard) override;
      virtual void find_trace_local_sets_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          unsigned req_index, ShardID local_shard,
          std::map<EquivalenceSet*, unsigned>& current_sets) override;
      virtual void find_shard_trace_local_sets_kd_tree(
          EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
          unsigned req_index, std::map<EquivalenceSet*, unsigned>& current_sets,
          local::map<ShardID, FieldMask>& remote_shards,
          ShardID local_shard) override;
    public:
      bool contains_point(const Point<DIM, T>& point);
    protected:
      ColorSpaceLinearizationT<DIM, T>* compute_linearization_metadata(void);
    protected:
      DomainT<DIM, T> realm_index_space;
    protected:
      std::atomic<ColorSpaceLinearizationT<DIM, T>*> linearization;
    public:
      struct CreateByDomainHelper {
      public:
        CreateByDomainHelper(
            IndexSpaceNodeT<DIM, T>* n, IndexPartNode* p, Operation* o,
            const std::map<DomainPoint, FutureImpl*>& fts, const Domain& domain,
            bool inter)
          : node(n), partition(p), op(o), futures(fts),
            future_map_domain(domain), intersect(inter)
        { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByDomainHelper* creator)
        {
          creator->result =
              creator->node
                  ->template create_by_domain_helper<COLOR_DIM::N, COLOR_T>(
                      creator->op, creator->partition, creator->futures,
                      creator->future_map_domain, creator->intersect);
        }
      public:
        IndexSpaceNodeT<DIM, T>* const node;
        IndexPartNode* const partition;
        Operation* const op;
        const std::map<DomainPoint, FutureImpl*>& futures;
        const Domain& future_map_domain;
        const bool intersect;
        ApEvent result;
      };
      struct CreateByWeightHelper {
      public:
        CreateByWeightHelper(
            IndexSpaceNodeT<DIM, T>* n, IndexPartNode* p, Operation* o,
            const std::map<DomainPoint, FutureImpl*>& wts, size_t g)
          : node(n), partition(p), op(o), weights(wts), granularity(g)
        { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByWeightHelper* creator)
        {
          creator->result =
              creator->node
                  ->template create_by_weight_helper<COLOR_DIM::N, COLOR_T>(
                      creator->op, creator->partition, creator->weights,
                      creator->granularity);
        }
      public:
        IndexSpaceNodeT<DIM, T>* const node;
        IndexPartNode* const partition;
        Operation* const op;
        const std::map<DomainPoint, FutureImpl*>& weights;
        const size_t granularity;
        ApEvent result;
      };
      struct CreateByFieldHelper {
      public:
        CreateByFieldHelper(
            IndexSpaceNodeT<DIM, T>* n, Operation* o, FieldID f,
            IndexPartNode* p, const std::vector<FieldDataDescriptor>& i,
            std::vector<DeppartResult>* res, ApEvent r)
          : node(n), op(o), partition(p), instances(i), results(res), ready(r),
            fid(f)
        { }
      public:
        template<typename COLOR_DIM, typename COLOR_T>
        static inline void demux(CreateByFieldHelper* creator)
        {
          creator->result =
              creator->node
                  ->template create_by_field_helper<COLOR_DIM::N, COLOR_T>(
                      creator->op, creator->fid, creator->partition,
                      creator->instances, creator->results, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM, T>* node;
        Operation* op;
        IndexPartNode* partition;
        const std::vector<FieldDataDescriptor>& instances;
        std::vector<DeppartResult>* const results;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByImageHelper {
      public:
        CreateByImageHelper(
            IndexSpaceNodeT<DIM, T>* n, Operation* o, FieldID f,
            IndexPartNode* p, IndexPartNode* j,
            std::vector<FieldDataDescriptor>& i, ApEvent r)
          : node(n), op(o), partition(p), projection(j), instances(i), ready(r),
            fid(f)
        { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageHelper* creator)
        {
          creator->result =
              creator->node->template create_by_image_helper<DIM2::N, T2>(
                  creator->op, creator->fid, creator->partition,
                  creator->projection, creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM, T>* node;
        Operation* op;
        IndexPartNode* partition;
        IndexPartNode* projection;
        std::vector<FieldDataDescriptor>& instances;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByImageRangeHelper {
      public:
        CreateByImageRangeHelper(
            IndexSpaceNodeT<DIM, T>* n, Operation* o, FieldID f,
            IndexPartNode* p, IndexPartNode* j,
            std::vector<FieldDataDescriptor>& i, ApEvent r)
          : node(n), op(o), partition(p), projection(j), instances(i), ready(r),
            fid(f)
        { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByImageRangeHelper* creator)
        {
          creator->result =
              creator->node->template create_by_image_range_helper<DIM2::N, T2>(
                  creator->op, creator->fid, creator->partition,
                  creator->projection, creator->instances, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM, T>* node;
        Operation* op;
        IndexPartNode* partition;
        IndexPartNode* projection;
        std::vector<FieldDataDescriptor>& instances;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByPreimageHelper {
      public:
        CreateByPreimageHelper(
            IndexSpaceNodeT<DIM, T>* n, Operation* o, FieldID f,
            IndexPartNode* p, IndexPartNode* j,
            const std::vector<FieldDataDescriptor>& i,
            const std::map<DomainPoint, Domain>* t,
            std::vector<DeppartResult>* res, ApEvent r)
          : node(n), op(o), partition(p), projection(j), instances(i),
            remote_targets(t), results(res), ready(r), fid(f)
        { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByPreimageHelper* creator)
        {
          creator->result =
              creator->node->template create_by_preimage_helper<DIM2::N, T2>(
                  creator->op, creator->fid, creator->partition,
                  creator->projection, creator->instances,
                  creator->remote_targets, creator->results, creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM, T>* node;
        Operation* op;
        IndexPartNode* partition;
        IndexPartNode* projection;
        const std::vector<FieldDataDescriptor>& instances;
        const std::map<DomainPoint, Domain>* const remote_targets;
        std::vector<DeppartResult>* const results;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateByPreimageRangeHelper {
      public:
        CreateByPreimageRangeHelper(
            IndexSpaceNodeT<DIM, T>* n, Operation* o, FieldID f,
            IndexPartNode* p, IndexPartNode* j,
            const std::vector<FieldDataDescriptor>& i,
            const std::map<DomainPoint, Domain>* t,
            std::vector<DeppartResult>* res, ApEvent r)
          : node(n), op(o), partition(p), projection(j), instances(i),
            remote_targets(t), results(res), ready(r), fid(f)
        { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateByPreimageRangeHelper* creator)
        {
          creator->result =
              creator->node
                  ->template create_by_preimage_range_helper<DIM2::N, T2>(
                      creator->op, creator->fid, creator->partition,
                      creator->projection, creator->instances,
                      creator->remote_targets, creator->results,
                      creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM, T>* node;
        Operation* op;
        IndexPartNode* partition;
        IndexPartNode* projection;
        const std::vector<FieldDataDescriptor>& instances;
        const std::map<DomainPoint, Domain>* const remote_targets;
        std::vector<DeppartResult>* const results;
        ApEvent ready, result;
        FieldID fid;
      };
      struct CreateAssociationHelper {
      public:
        CreateAssociationHelper(
            IndexSpaceNodeT<DIM, T>* n, Operation* o, FieldID f,
            IndexSpaceNode* g, const std::vector<FieldDataDescriptor>& i,
            ApEvent r)
          : node(n), op(o), range(g), instances(i), ready(r), fid(f)
        { }
      public:
        template<typename DIM2, typename T2>
        static inline void demux(CreateAssociationHelper* creator)
        {
          creator->result =
              creator->node->template create_association_helper<DIM2::N, T2>(
                  creator->op, creator->fid, creator->range, creator->instances,
                  creator->ready);
        }
      public:
        IndexSpaceNodeT<DIM, T>* node;
        Operation* op;
        IndexSpaceNode* range;
        const std::vector<FieldDataDescriptor>& instances;
        ApEvent ready, result;
        FieldID fid;
      };
    };

    /**
     * \class ColorSpaceLinearization
     * A color space linearation maps N-D color spaces to an
     * (almost) contiguous 1-D space that can be traversed by
     * the runtime with good locality between the points in
     * N dimensions. It does this using generalized N-D Morton
     * curves. There are some catches though that prevent us
     * from just using one big Morton curve in most cases.
     * The first problem is that Morton curves must be done
     * on hypercubes with powers of 2 dimensions, which means we
     * either need to under approximate most rectangles in
     * the color space. The second problem is in higher that
     * in higher dimenstions we might end up exceeding the
     * maximum number of bits we can use to represent the
     * Morton curve since we only have 64-bits in our the
     * LegionColor type. We therefore often will end up tiling
     * the color space to meet these constraints.
     */
    template<int DIM, typename T>
    class ColorSpaceLinearizationT {
    public:
      class MortonTile {
      public:
        MortonTile(
            const Rect<DIM, T>& b, unsigned count, const int dims[DIM],
            unsigned order)
          : bounds(b), interesting_count(count), morton_order(order), index(0)
        {
          for (unsigned idx = 0; idx < DIM; idx++)
            interesting_dims[idx] = dims[idx];
        }
      public:
        LegionColor get_max_linearized_color(void) const;
        LegionColor linearize(const Point<DIM, T>& point) const;
        void delinearize(LegionColor color, Point<DIM, T>& point) const;
        bool contains_color(LegionColor color) const;
        size_t compute_color_offset(LegionColor color) const;
      public:
        Rect<DIM, T> bounds;
        int interesting_dims[DIM];
        unsigned interesting_count;
        unsigned morton_order;
        unsigned index;
      };
    public:
      ColorSpaceLinearizationT(const DomainT<DIM, T>& domain);
      ColorSpaceLinearizationT(const ColorSpaceLinearizationT& rhs) = delete;
      ~ColorSpaceLinearizationT(void);
    public:
      ColorSpaceLinearizationT& operator=(const ColorSpaceLinearizationT& rhs) =
          delete;
    public:
      LegionColor get_max_linearized_color(void) const;
      LegionColor linearize(const Point<DIM, T>& point) const;
      void delinearize(LegionColor color, Point<DIM, T>& point) const;
      bool contains_color(LegionColor color) const;
      size_t compute_color_offset(LegionColor color) const;
    protected:
      // Bounds of a rectangle contained in the color space
      std::vector<MortonTile*> morton_tiles;
      // The starting color for each tile (sorted)
      std::vector<LegionColor> color_offsets;
      // KD-Tree for looking up the owner tile for points
      KDNode<DIM, T, MortonTile*>* kdtree;
    };

    // Specialization for the case of DIM==1 since that is easy
    // No need for any fancy Morton curves here, we can pack
    // all the points nice and densely
    template<typename T>
    class ColorSpaceLinearizationT<1, T> {
    public:
      ColorSpaceLinearizationT(const DomainT<1, T>& domain);
    public:
      LegionColor get_max_linearized_color(void) const;
      LegionColor linearize(const Point<1, T>& point) const;
      void delinearize(LegionColor color, Point<1, T>& point) const;
      bool contains_color(LegionColor color) const;
      size_t compute_color_offset(LegionColor color) const;
    protected:
      // The lo point for each tile (sorted)
      std::vector<T> tiles;
      // Extents of each tile
      std::vector<size_t> extents;
      // The starting color for each tile
      std::vector<LegionColor> color_offsets;
    };

    /**
     * \class ColorSpaceIterator
     * A color space iterator helps iterating over a (subset)
     * of the colors in a color space for a particular partition.
     * It abstracts the details of whether the colors space is
     * sparse or dense and can deal with chunking for sharding.
     */
    class ColorSpaceIterator {
    public:
      ColorSpaceIterator(IndexPartNode* partition, bool local_only = false);
      ColorSpaceIterator(
          IndexPartNode* partition, ShardID local_shard, size_t total_shards);
    public:
      operator bool(void) const;
      LegionColor operator*(void) const;
      ColorSpaceIterator& operator++(int /*postfix*/);
      void step(void);
      static LegionColor compute_chunk(
          LegionColor max_color, size_t total_spaces);
    private:
      IndexSpaceNode* color_space;
      LegionColor current, end;
      bool simple_step;
    };

    /**
     * \class IndexSpaceCreator
     * A small helper class for creating templated index spaces
     */
    class IndexSpaceCreator {
    public:
      IndexSpaceCreator(
          IndexSpace s, IndexPartNode* p, LegionColor c, IndexSpaceExprID e,
          RtEvent init, unsigned dp, Provenance* prov, CollectiveMapping* m,
          bool valid)
        : space(s), parent(p), color(c), expr_id(e), initialized(init),
          depth(dp), provenance(prov), mapping(m), tree_valid(valid),
          result(nullptr)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexSpaceCreator* creator)
      {
        creator->result = new IndexSpaceNodeT<N::N, T>(
            creator->space, creator->parent, creator->color, creator->expr_id,
            creator->initialized, creator->depth, creator->provenance,
            creator->mapping, creator->tree_valid);
      }
    public:
      const IndexSpace space;
      IndexPartNode* const parent;
      const LegionColor color;
      const IndexSpaceExprID expr_id;
      const RtEvent initialized;
      const unsigned depth;
      Provenance* const provenance;
      CollectiveMapping* const mapping;
      const bool tree_valid;
      IndexSpaceNode* result;
    };

    /**
     * \class PartitionTracker
     * This is a small helper class that is used for figuring out
     * when to remove references to LogicalPartition objects. We
     * want to remove the references as soon as either the index
     * partition is destroyed or the logical region is destroyed.
     * We use this class to detect which one occurs first.
     */
    class PartitionTracker : public Collectable {
    public:
      PartitionTracker(PartitionNode* part);
      PartitionTracker(const PartitionTracker& rhs) = delete;
      ~PartitionTracker(void) { }
    public:
      PartitionTracker& operator=(const PartitionTracker& rhs) = delete;
    public:
      bool can_prune(void);
      bool remove_partition_reference(void);
    private:
      PartitionNode* const partition;
    };

    /**
     * \class IndexPartNode
     * A node for representing a generic index partition.
     */
    class IndexPartNode : public IndexTreeNode {
    public:
      struct DisjointnessArgs : public LgTaskArgs<DisjointnessArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DISJOINTNESS_TASK_ID;
      public:
        DisjointnessArgs(void) = default;
        DisjointnessArgs(IndexPartNode* proxy)
          : LgTaskArgs<DisjointnessArgs>(true, true), proxy_this(proxy)
        { }
        void execute(void) const;
      public:
        IndexPartNode* proxy_this;
      };
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_INDEX_PART_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(void) = default;
        SemanticRequestArgs(
            IndexPartNode* proxy, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(false, false), proxy_this(proxy),
            tag(t), source(src)
        { }
        void execute(void) const;
      public:
        IndexPartNode* proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
      struct DeferChildArgs : public LgTaskArgs<DeferChildArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_INDEX_PART_DEFER_CHILD_TASK_ID;
      public:
        DeferChildArgs(void) = default;
        DeferChildArgs(
            IndexPartNode* proxy, LegionColor child, AddressSpaceID src)
          : LgTaskArgs<DeferChildArgs>(false, false), proxy_this(proxy),
            child_color(child), source(src)
        { }
        void execute(void) const;
      public:
        IndexPartNode* proxy_this;
        LegionColor child_color;
        AddressSpaceID source;
      };
      class DeferFindShardRects : public LgTaskArgs<DeferFindShardRects> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_INDEX_PART_DEFER_SHARD_RECTS_TASK_ID;
      public:
        DeferFindShardRects(void) = default;
        DeferFindShardRects(IndexPartNode* proxy)
          : LgTaskArgs<DeferFindShardRects>(false, false), proxy_this(proxy)
        { }
        void execute(void) const;
      public:
        IndexPartNode* proxy_this;
      };
      class RemoteDisjointnessFunctor {
      public:
        RemoteDisjointnessFunctor(IndexPartitionDisjointUpdate& r);
      public:
        void apply(AddressSpaceID target);
      public:
        IndexPartitionDisjointUpdate& rez;
      };
    protected:
      class InterferenceEntry {
      public:
        InterferenceEntry(void) : expr_id(0), older(nullptr), newer(nullptr) { }
      public:
        std::vector<LegionColor> colors;
        IndexSpaceExprID expr_id;
        InterferenceEntry* older;
        InterferenceEntry* newer;
      };
    public:
      class RemoteKDTracker {
      public:
        RemoteKDTracker(void);
      public:
        RtEvent find_remote_interfering(
            const std::set<AddressSpaceID>& targets, IndexPartition handle,
            IndexSpaceExpression* expr);
        void get_remote_interfering(std::set<LegionColor>& colors);
        void process_remote_interfering_response(Deserializer& derez);
      protected:
        mutable LocalLock tracker_lock;
        std::set<LegionColor> remote_colors;
        RtUserEvent done_event;
        std::atomic<unsigned> remaining;
      };
    public:
      IndexPartNode(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_space,
          LegionColor c, bool disjoint, int complete, RtEvent initialized,
          CollectiveMapping* mapping, Provenance* provenance);
      IndexPartNode(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_space,
          LegionColor c, int complete, RtEvent initialized,
          CollectiveMapping* mapping, Provenance* provenance);
      IndexPartNode(const IndexPartNode& rhs) = delete;
      virtual ~IndexPartNode(void);
    public:
      IndexPartNode& operator=(const IndexPartNode& rhs) = delete;
    public:
      virtual void notify_invalid(void) override;
      virtual void notify_local(void) override;
    public:
      virtual bool is_index_space_node(void) const override;
#ifdef LEGION_DEBUG
      virtual IndexSpaceNode* as_index_space_node(void) override;
      virtual IndexPartNode* as_index_part_node(void) override;
#endif
      virtual AddressSpaceID get_owner_space(void) const override;
      static AddressSpaceID get_owner_space(IndexPartition handle);
    public:
      virtual IndexTreeNode* get_parent(void) const override;
      virtual LegionColor get_colors(std::vector<LegionColor>& colors) override;
    public:
      virtual void send_semantic_request(
          AddressSpaceID target, SemanticTag tag, bool can_fail,
          bool wait_until, RtUserEvent ready) override;
      virtual void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable, RtUserEvent ready) override;
      void process_semantic_request(
          SemanticTag tag, AddressSpaceID source, bool can_fail,
          bool wait_until, RtUserEvent ready);
    public:
      bool has_color(const LegionColor c);
      AddressSpaceID find_color_creator_space(
          LegionColor color, CollectiveMapping*& child_mapping) const;
      IndexSpaceNode* get_child(const LegionColor c, RtEvent* defer = nullptr);
      void add_child(IndexSpaceNode* child);
      void set_child(IndexSpaceNode* child);
      void add_tracker(PartitionTracker* tracker);
      size_t get_num_children(void) const;
      bool needs_subspace_broadcast(void) const;
      bool compute_disjointness_and_completeness(void);
      bool update_disjoint_complete_result(
          uint64_t children_volume, uint64_t intersection_volume = 0);
      bool update_disjoint_complete_result(
          std::map<LegionColor, uint64_t>& children_volumes,
          std::map<std::pair<LegionColor, LegionColor>, uint64_t>*
              intersection_volumes = nullptr);
      bool finalize_disjoint_complete(void);
    public:
      void initialize_disjoint_complete_notifications(void);
      bool is_disjoint(bool from_app = false, bool false_if_not_ready = false);
      bool are_disjoint(
          LegionColor c1, LegionColor c2, bool force_compute = false);
      bool is_complete(bool from_app = false, bool false_if_not_ready = false);
      bool handle_disjointness_update(Deserializer& derez);
    public:
      ApEvent create_equal_children(Operation* op, size_t granularity);
      ApEvent create_by_weights(
          Operation* op, const std::map<DomainPoint, FutureImpl*>& weights,
          size_t granularity);
      ApEvent create_by_union(
          Operation* Op, IndexPartNode* left, IndexPartNode* right);
      ApEvent create_by_intersection(
          Operation* op, IndexPartNode* left, IndexPartNode* right);
      ApEvent create_by_intersection(
          Operation* op, IndexPartNode* original, const bool dominates);
      ApEvent create_by_difference(
          Operation* op, IndexPartNode* left, IndexPartNode* right);
      ApEvent create_by_restriction(const void* transform, const void* extent);
      ApEvent create_by_domain(
          const std::map<DomainPoint, FutureImpl*>& futures,
          const Domain& future_map_domain);
    public:
      bool intersects_with(IndexSpaceNode* other, bool compute = true);
      bool intersects_with(IndexPartNode* other, bool compute = true);
      void find_interfering_children(
          IndexSpaceExpression* expr, std::vector<LegionColor>& colors);
      virtual bool find_interfering_children_kd(
          IndexSpaceExpression* expr, std::vector<LegionColor>& colors,
          bool local_only = false) = 0;
    public:
      void send_node(AddressSpaceID target, bool recurse);
      void pack_node(Serializer& rez, AddressSpaceID target);
    public:
      RtEvent request_shard_rects(void);
      virtual void initialize_shard_rects(void) = 0;
      virtual bool find_local_shard_rects(void) = 0;
      virtual void pack_shard_rects(Serializer& rez, bool clear) = 0;
      virtual void unpack_shard_rects(Deserializer& derez) = 0;
      bool process_shard_rects_response(Deserializer& derez, AddressSpace src);
      bool perform_shard_rects_notification(void);
    public:
      const IndexPartition handle;
      IndexSpaceNode* const parent;
      IndexSpaceNode* const color_space;
      const LegionColor total_children;
      const LegionColor max_linearized_color;
    protected:
      // Must hold the node lock when accessing these data structures
      // the remaining data structures
      std::map<LegionColor, IndexSpaceNode*> color_map;
      std::map<LegionColor, RtUserEvent> pending_child_map;
      std::set<std::pair<LegionColor, LegionColor> > disjoint_subspaces;
      std::set<std::pair<LegionColor, LegionColor> > aliased_subspaces;
      std::list<PartitionTracker*> partition_trackers;
    protected:
      // Support for computing disjointness locally
      uint64_t total_children_volume, total_intersection_volume;
      std::map<LegionColor, uint64_t> total_children_volumes;
      std::map<std::pair<LegionColor, LegionColor>, uint64_t>
          total_intersection_volumes;
      unsigned remaining_local_disjoint_complete_notifications;
      unsigned remaining_global_disjoint_complete_notifications;
    protected:
      std::atomic<bool> has_disjoint, disjoint;
      std::atomic<bool> has_complete, complete;
      RtUserEvent disjoint_complete_ready;
    protected:
      // Members for the interference cache
      static constexpr size_t MAX_INTERFERENCE_CACHE_SIZE = 64;
      std::map<IndexSpaceExprID, InterferenceEntry> interference_cache;
      InterferenceEntry* first_entry;
    protected:
      // Help for building distributed kd-trees with shard mappings
      RtUserEvent shard_rects_ready;
      unsigned remaining_rect_notifications;
    };

    /**
     * \class IndexPartNodeT
     * A template class for handling any templated realm calls
     * associated with realm index spaces
     */
    template<int DIM, typename T>
    class IndexPartNodeT
      : public IndexPartNode,
        public Heapify<IndexPartNodeT<DIM, T>, LONG_LIFETIME> {
    public:
      IndexPartNodeT(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_space,
          LegionColor c, bool disjoint, int complete, RtEvent initialized,
          CollectiveMapping* mapping, Provenance* provenance);
      IndexPartNodeT(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_space,
          LegionColor c, int complete, RtEvent initialized,
          CollectiveMapping* mapping, Provenance* provenance);
      IndexPartNodeT(const IndexPartNodeT& rhs) = delete;
      virtual ~IndexPartNodeT(void);
    public:
      IndexPartNodeT& operator=(const IndexPartNodeT& rhs) = delete;
    public:
      virtual bool find_interfering_children_kd(
          IndexSpaceExpression* expr, std::vector<LegionColor>& colors,
          bool local_only = false);
    protected:
      virtual void initialize_shard_rects(void);
      virtual bool find_local_shard_rects(void);
      virtual void pack_shard_rects(Serializer& rez, bool clear);
      virtual void unpack_shard_rects(Deserializer& derez);
    protected:
      KDNode<DIM, T, LegionColor>* kd_root;
      KDNode<DIM, T, AddressSpaceID>* kd_remote;
      RtUserEvent kd_remote_ready;
    protected:
      // Each color appears exactly once in this data structure
      std::vector<std::pair<Rect<DIM, T>, LegionColor> >* dense_shard_rects;
      // There might be multiple rectangles for each color here
      // These rectangles are just an approximation of the actual
      // points in the children with sparsity maps
      std::vector<std::pair<Rect<DIM, T>, LegionColor> >* sparse_shard_rects;
    };

    /**
     * \class IndexPartCreator
     * A msall helper class for creating templated index partitions
     */
    class IndexPartCreator {
    public:
      IndexPartCreator(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* cs,
          LegionColor c, bool d, int k, RtEvent initialized,
          CollectiveMapping* m, Provenance* prov)
        : partition(p), parent(par), color_space(cs), color(c),
          has_disjoint(true), disjoint(d), complete(k), init(initialized),
          mapping(m), provenance(prov)
      { }
      IndexPartCreator(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* cs,
          LegionColor c, int k, RtEvent initialized, CollectiveMapping* m,
          Provenance* prov)
        : partition(p), parent(par), color_space(cs), color(c),
          has_disjoint(false), disjoint(false), complete(k), init(initialized),
          mapping(m), provenance(prov)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(IndexPartCreator* creator)
      {
        if (!creator->has_disjoint)
          creator->result = new IndexPartNodeT<N::N, T>(
              creator->partition, creator->parent, creator->color_space,
              creator->color, creator->complete, creator->init,
              creator->mapping, creator->provenance);
        else
          creator->result = new IndexPartNodeT<N::N, T>(
              creator->partition, creator->parent, creator->color_space,
              creator->color, creator->disjoint, creator->complete,
              creator->init, creator->mapping, creator->provenance);
      }
    public:
      const IndexPartition partition;
      IndexSpaceNode* const parent;
      IndexSpaceNode* const color_space;
      const LegionColor color;
      const bool has_disjoint;
      const bool disjoint;
      const int complete;
      const RtEvent init;
      CollectiveMapping* const mapping;
      Provenance* const provenance;
      IndexPartNode* result;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/nodes/index.inl"

#endif  // __LEGION_INDEX_SPACE_H__
