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

#ifndef __LEGION_EXPRESSION_H__
#define __LEGION_EXPRESSION_H__

#include "legion/kernel/metatask.h"
#include "legion/kernel/runtime.h"
#include "legion/operations/operation.h"
#include "legion/tools/spy.h"
#include "legion/utilities/serdez.h"
#include "legion/utilities/fieldmask_map.h"
#include "legion/utilities/hasher.h"
#include "legion/utilities/small_vector.h"

namespace Legion {
  namespace Internal {

    /**
     * \class IndexSpaceExpression
     * An IndexSpaceExpression represents a set computation
     * one on or more index spaces. IndexSpaceExpressions
     * currently are either IndexSpaceNodes at the leaves
     * or have intermeidate set operations that are either
     * set union, intersection, or difference.
     */
    class IndexSpaceExpression {
    public:
      struct TightenIndexSpaceArgs : public LgTaskArgs<TightenIndexSpaceArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_TIGHTEN_INDEX_SPACE_TASK_ID;
      public:
        TightenIndexSpaceArgs(void) = default;
        TightenIndexSpaceArgs(
            IndexSpaceExpression* proxy, DistributedCollectable* dc)
          : LgTaskArgs<TightenIndexSpaceArgs>(true, true), proxy_this(proxy),
            proxy_dc(dc)
        {
          proxy_dc->add_base_resource_ref(META_TASK_REF);
        }
        void execute(void) const;
      public:
        IndexSpaceExpression* proxy_this;
        DistributedCollectable* proxy_dc;
      };
    public:
      IndexSpaceExpression(LocalLock& lock);
      IndexSpaceExpression(TypeTag tag, LocalLock& lock);
      IndexSpaceExpression(TypeTag tag, IndexSpaceExprID id, LocalLock& lock);
      virtual ~IndexSpaceExpression(void);
    public:
      virtual bool is_sparse(void) = 0;
      virtual Domain get_tight_domain(void) = 0;
      [[nodiscard]] virtual ApEvent get_loose_domain(
          Domain& domain, ApUserEvent& done_event) = 0;
      virtual void record_index_space_user(ApEvent user) = 0;
      virtual void tighten_index_space(void) = 0;
      virtual bool is_set(void) const { return true; }
      virtual bool check_empty(void) = 0;
      virtual size_t get_volume(void) = 0;
      virtual void pack_expression(Serializer& rez, AddressSpaceID target) = 0;
      virtual void skip_unpack_expression(Deserializer& derez) const = 0;
    public:
      virtual IndexSpaceExpression* inline_union(IndexSpaceExpression* rhs) = 0;
      virtual IndexSpaceExpression* inline_union(
          const SetView<IndexSpaceExpression*>& exprs) = 0;
      virtual IndexSpaceExpression* inline_intersection(
          IndexSpaceExpression* rhs) = 0;
      virtual IndexSpaceExpression* inline_intersection(
          const SetView<IndexSpaceExpression*>& exprs) = 0;
      virtual IndexSpaceExpression* inline_subtraction(
          IndexSpaceExpression* rhs) = 0;
    public:
#ifdef LEGION_DEBUG
      virtual bool is_valid(void) = 0;
#endif
      virtual DistributedID get_distributed_id(void) const = 0;
      virtual void add_canonical_reference(DistributedID source) = 0;
      virtual bool remove_canonical_reference(DistributedID source) = 0;
      virtual bool try_add_live_reference(void) = 0;
      virtual void add_base_expression_reference(
          ReferenceSource source, unsigned count = 1) = 0;
      virtual void add_nested_expression_reference(
          DistributedID source, unsigned count = 1) = 0;
      virtual bool remove_base_expression_reference(
          ReferenceSource source, unsigned count = 1) = 0;
      virtual bool remove_nested_expression_reference(
          DistributedID source, unsigned count = 1) = 0;
      virtual void add_tree_expression_reference(
          DistributedID source, unsigned count = 1) = 0;
      virtual bool remove_tree_expression_reference(
          DistributedID source, unsigned count = 1) = 0;
      virtual bool test_intersection_nonblocking(
          IndexSpaceExpression* expr, ApEvent& precondition,
          bool second = false);
    public:
      virtual IndexSpaceNode* create_node(
          IndexSpace handle, RtEvent initialized, Provenance* provenance,
          CollectiveMapping* mapping, IndexSpaceExprID expr_id = 0) = 0;
      virtual IndexSpaceExpression* create_from_rectangles(
          const local::set<Domain>& rectangles) = 0;
      virtual PieceIteratorImpl* create_piece_iterator(
          const void* piece_list, size_t piece_list_size,
          IndexSpaceNode* privilege_node) = 0;
      virtual bool is_below_in_tree(IndexPartNode* p, LegionColor& child) const
      {
        return false;
      }
    public:
      virtual DistributedID record_profiler_expression(void) = 0;
      virtual ApEvent issue_fill(
          Operation* op, const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& dst_fields,
          const void* fill_value, size_t fill_size, UniqueID fill_uid,
          FieldSpace handle, RegionTreeID tree_id, ApEvent precondition,
          PredEvent pred_guard, LgEvent unique_event, CollectiveKind collective,
          bool record_effect, int priority = 0, bool replay = false) = 0;
      virtual ApEvent issue_copy(
          Operation* op, const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent pred_guard, LgEvent src_unique,
          LgEvent dst_unique, CollectiveKind collective, bool record_effect,
          int priority = 0, bool replay = false) = 0;
      virtual CopyAcrossUnstructured* create_across_unstructured(
          const std::map<Reservation, bool>& reservations,
          const bool compute_preimages, const bool shadow_indirections) = 0;
      virtual Realm::InstanceLayoutGeneric* create_layout(
          const LayoutConstraintSet& constraints,
          const std::vector<FieldID>& field_ids,
          const std::vector<size_t>& field_sizes, bool compact,
          void** piece_list = nullptr, size_t* piece_list_size = nullptr,
          size_t* num_pieces = nullptr, size_t base_alignment = 32) = 0;
      // Return the expression with a resource ref on the expression
      virtual IndexSpaceExpression* create_layout_expression(
          const void* piece_list, size_t piece_list_size) = 0;
      virtual bool meets_layout_expression(
          IndexSpaceExpression* expr, bool tight_bounds, const void* piece_list,
          size_t piece_list_size, const Domain* padding_delta) = 0;
    public:
      virtual IndexSpaceExpression* find_congruent_expression(
          SmallPointerVector<IndexSpaceExpression, true>& expressions) = 0;
      virtual KDTree* get_sparsity_map_kd_tree(void) = 0;
    public:
      virtual void initialize_equivalence_set_kd_tree(
          EqKDTree* tree, EquivalenceSet* set, const FieldMask& mask,
          ShardID local_shard, bool current) = 0;
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
          ShardID local_shard = 0) = 0;
      virtual unsigned record_output_equivalence_set(
          EqKDTree* tree, LocalLock* tree_lock, EquivalenceSet* set,
          const FieldMask& mask, EqSetTracker* tracker,
          AddressSpaceID tracker_space,
          local::FieldMaskMap<EqKDTree>& subscriptions,
          op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
          ShardID local_shard = 0) = 0;
    public:
      static AddressSpaceID get_owner_space(IndexSpaceExprID id);
    public:
      void add_derived_operation(IndexSpaceOperation* op);
      void remove_derived_operation(IndexSpaceOperation* op);
      void invalidate_derived_operations(void);
    public:
      inline bool is_empty(void)
      {
        if (!has_empty.load())
        {
          empty = check_empty();
          has_empty.store(true);
        }
        return empty;
      }
      inline size_t get_num_dims(void) const
      {
        return NT_TemplateHelper::get_dim(type_tag);
      }
    public:
      // Convert this index space expression to the canonical one that
      // represents all expressions that are all congruent
      IndexSpaceExpression* get_canonical_expression(void);
      virtual uint64_t get_canonical_hash(void) = 0;
    protected:
      template<int DIM, typename T>
      static IndexSpaceExpression* find_or_create_empty_expression(void);
      template<int DIM, typename T>
      IndexSpaceExpression* inline_union_internal(IndexSpaceExpression* rhs);
      template<int DIM, typename T>
      IndexSpaceExpression* inline_union_internal(
          const SetView<IndexSpaceExpression*>& exprs);
      template<int DIM, typename T>
      IndexSpaceExpression* inline_intersection_internal(
          IndexSpaceExpression* rhs);
      template<int DIM, typename T>
      IndexSpaceExpression* inline_intersection_internal(
          const SetView<IndexSpaceExpression*>& exprs);
      template<int DIM, typename T>
      IndexSpaceExpression* inline_subtraction_internal(
          IndexSpaceExpression* rhs);
      template<int DIM, typename T>
      uint64_t get_canonical_hash_internal(const DomainT<DIM, T>& domain) const;
      template<int DIM, typename T>
      void log_profiler_points(
          DistributedID did, const DomainT<DIM, T>& tight_space) const;
    protected:
      template<int DIM, typename T>
      inline ApEvent issue_fill_internal(
          Operation* op, const Realm::IndexSpace<DIM, T>& space,
          const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& dst_fields,
          const void* fill_value, size_t fill_size, UniqueID fill_uid,
          FieldSpace handle, RegionTreeID tree_id, ApEvent precondition,
          PredEvent pred_guard, LgEvent unique_event, CollectiveKind collective,
          bool record_effect, int priority, bool replay);
    public:
      // Make this one public so it can be accessed by CopyUnstructuredT
      // Be careful using this directly
      template<int DIM, typename T>
      inline ApEvent issue_copy_internal(
          Operation* op, const Realm::IndexSpace<DIM, T>& space,
          const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent pred_guard, LgEvent src_unique,
          LgEvent dst_unique, CollectiveKind collective, bool record_effect,
          int priority, bool replay);
    protected:
      template<int DIM, typename T>
      inline Realm::InstanceLayoutGeneric* create_layout_internal(
          const Realm::IndexSpace<DIM, T>& space,
          const LayoutConstraintSet& constraints,
          const std::vector<FieldID>& field_ids,
          const std::vector<size_t>& field_sizes, bool compact,
          void** piece_list = nullptr, size_t* piece_list_size = nullptr,
          size_t* num_pieces = nullptr, size_t base_alignment = 32) const;
      template<int DIM, typename T>
      inline IndexSpaceExpression* create_layout_expression_internal(
          const Realm::IndexSpace<DIM, T>& space, const Rect<DIM, T>* rects,
          size_t num_rects);
      template<int DIM, typename T>
      inline bool meets_layout_expression_internal(
          IndexSpaceExpression* space_expr, bool tight_bounds,
          const Rect<DIM, T>* piece_list, size_t piece_list_size,
          const Domain* padding_delta);
      template<int DIM, typename T>
      inline IndexSpaceExpression* create_from_rectangles_internal(
          const local::set<Domain>& rects);
    public:
      template<int DIM, typename T>
      inline IndexSpaceExpression* find_congruent_expression_internal(
          SmallPointerVector<IndexSpaceExpression, true>& expressions);
      template<int DIM, typename T>
      inline KDTree* get_sparsity_map_kd_tree_internal(void);
    public:
      static IndexSpaceExpression* unpack_expression(
          Deserializer& derez, AddressSpaceID source);
    public:
      const TypeTag type_tag;
      const IndexSpaceExprID expr_id;
    private:
      LocalLock& expr_lock;
    protected:
      std::set<IndexSpaceOperation*> derived_operations;
      std::atomic<IndexSpaceExpression*> canonical;
      KDTree* sparsity_map_kd_tree;
      size_t volume;
      std::atomic<bool> has_volume;
      bool empty;
      std::atomic<bool> has_empty;
      std::atomic<bool> profiler_logged = false;
    };

    class IndexSpaceOperation : public IndexSpaceExpression,
                                public DistributedCollectable {
    public:
      enum OperationKind {
        UNION_OP_KIND,
        INTERSECT_OP_KIND,
        DIFFERENCE_OP_KIND,
        REMOTE_EXPRESSION_KIND,
        INSTANCE_EXPRESSION_KIND,
      };
    public:
      IndexSpaceOperation(TypeTag tag, OperationKind kind);
      IndexSpaceOperation(
          TypeTag tag, IndexSpaceExprID eid, DistributedID did,
          IndexSpaceOperation* origin);
      virtual ~IndexSpaceOperation(void);
    public:
      virtual void notify_local(void) override;
    public:
      virtual Domain get_tight_domain(void) override = 0;
      [[nodiscard]] virtual ApEvent get_loose_domain(
          Domain& domain, ApUserEvent& done_event) override = 0;
      virtual void record_index_space_user(ApEvent user) override = 0;
      virtual void tighten_index_space(void) override = 0;
      virtual bool check_empty(void) override = 0;
      virtual size_t get_volume(void) override = 0;
      virtual void pack_expression(
          Serializer& rez, AddressSpaceID target) override = 0;
      virtual void skip_unpack_expression(
          Deserializer& derez) const override = 0;
    public:
#ifdef LEGION_DEBUG
      virtual bool is_valid(void) override
      {
        return DistributedCollectable::is_global();
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
      bool own_invalidation(void);
      virtual void invalidate_operation(IndexSpaceExpression* source) = 0;
      virtual void remove_operation(void) = 0;
      virtual IndexSpaceNode* create_node(
          IndexSpace handle, RtEvent initialized, Provenance* provenance,
          CollectiveMapping* mapping,
          IndexSpaceExprID expr_id = 0) override = 0;
    public:
      IndexSpaceOperation* const origin_expr;
      const OperationKind op_kind;
    protected:
      mutable LocalLock inter_lock;
      std::deque<ApEvent> index_space_users;
      std::atomic<bool> invalidated = false;
    };

    template<int DIM, typename T>
    class IndexSpaceOperationT : public IndexSpaceOperation {
    public:
      IndexSpaceOperationT(OperationKind kind);
      IndexSpaceOperationT(
          IndexSpaceExprID eid, DistributedID did, IndexSpaceOperation* op,
          TypeTag tag, Deserializer& derez);
      virtual ~IndexSpaceOperationT(void);
    public:
      virtual bool is_sparse(void) override;
      virtual Domain get_tight_domain(void) override;
      [[nodiscard]] virtual ApEvent get_loose_domain(
          Domain& domain, ApUserEvent& done_event) override;
      virtual void record_index_space_user(ApEvent user) override;
      virtual void tighten_index_space(void) override;
      virtual bool check_empty(void) override;
      virtual size_t get_volume(void) override;
      virtual void pack_expression(
          Serializer& rez, AddressSpaceID target) override;
      virtual void skip_unpack_expression(Deserializer& derez) const override;
      virtual void invalidate_operation(
          IndexSpaceExpression* source) override = 0;
      virtual void remove_operation(void) override = 0;
      virtual IndexSpaceNode* create_node(
          IndexSpace handle, RtEvent initialized, Provenance* provenance,
          CollectiveMapping* mapping, IndexSpaceExprID expr_id = 0) override;
      virtual IndexSpaceExpression* create_from_rectangles(
          const local::set<Domain>& rectangles) override;
      virtual PieceIteratorImpl* create_piece_iterator(
          const void* piece_list, size_t piece_list_size,
          IndexSpaceNode* privilege_node) override;
      virtual DistributedID record_profiler_expression(void) override;
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
    public:
      DomainT<DIM, T> get_tight_index_space(void);
      // Return event is when the result index space is safe to use
      // The done event must be triggered after the index space is
      // done being used if it is not a no-event
      [[nodiscard]] ApEvent get_loose_index_space(
          DomainT<DIM, T>& result, ApUserEvent& done_event);
    protected:
      Realm::IndexSpace<DIM, T> realm_index_space, tight_index_space;
      ApEvent realm_index_space_ready;
      RtEvent tight_index_space_ready;
      std::atomic<bool> is_index_space_tight;
    };

    template<int DIM, typename T>
    class IndexSpaceUnion
      : public IndexSpaceOperationT<DIM, T>,
        public Heapify<IndexSpaceUnion<DIM, T>, LONG_LIFETIME> {
    public:
      IndexSpaceUnion(const std::vector<IndexSpaceExpression*>& to_union);
      IndexSpaceUnion(const Rect<DIM, T>& bounds);
      IndexSpaceUnion(const IndexSpaceUnion<DIM, T>& rhs) = delete;
      virtual ~IndexSpaceUnion(void);
    public:
      IndexSpaceUnion& operator=(const IndexSpaceUnion& rhs) = delete;
    public:
      virtual void invalidate_operation(IndexSpaceExpression* source) override;
      virtual void remove_operation(void) override;
    protected:
      const std::vector<IndexSpaceExpression*> sub_expressions;
    };

    class UnionOpCreator : public OperationCreator {
    public:
      UnionOpCreator(TypeTag t, const std::vector<IndexSpaceExpression*>& e)
        : OperationCreator(), type_tag(t), exprs(e)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(UnionOpCreator* creator)
      {
        creator->produce(new IndexSpaceUnion<N::N, T>(creator->exprs));
      }
    public:
      virtual void create_operation(void) override
      {
        NT_TemplateHelper::demux<UnionOpCreator>(type_tag, this);
      }
    public:
      const TypeTag type_tag;
      const std::vector<IndexSpaceExpression*>& exprs;
    };

    template<int DIM, typename T>
    class IndexSpaceIntersection
      : public IndexSpaceOperationT<DIM, T>,
        public Heapify<IndexSpaceIntersection<DIM, T>, LONG_LIFETIME> {
    public:
      IndexSpaceIntersection(
          const std::vector<IndexSpaceExpression*>& to_inter);
      IndexSpaceIntersection(const Rect<DIM, T>& bounds);
      IndexSpaceIntersection(const IndexSpaceIntersection& rhs) = delete;
      virtual ~IndexSpaceIntersection(void);
    public:
      IndexSpaceIntersection& operator=(const IndexSpaceIntersection& rhs) =
          delete;
    public:
      virtual void invalidate_operation(IndexSpaceExpression* source) override;
      virtual void remove_operation(void) override;
    protected:
      const std::vector<IndexSpaceExpression*> sub_expressions;
    };

    class IntersectionOpCreator : public OperationCreator {
    public:
      IntersectionOpCreator(
          TypeTag t, const std::vector<IndexSpaceExpression*>& e)
        : OperationCreator(), type_tag(t), exprs(e)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(IntersectionOpCreator* creator)
      {
        creator->produce(new IndexSpaceIntersection<N::N, T>(creator->exprs));
      }
    public:
      virtual void create_operation(void) override
      {
        NT_TemplateHelper::demux<IntersectionOpCreator>(type_tag, this);
      }
    public:
      const TypeTag type_tag;
      const std::vector<IndexSpaceExpression*>& exprs;
    };

    template<int DIM, typename T>
    class IndexSpaceDifference
      : public IndexSpaceOperationT<DIM, T>,
        public Heapify<IndexSpaceDifference<DIM, T>, LONG_LIFETIME> {
    public:
      IndexSpaceDifference(
          IndexSpaceExpression* lhs, IndexSpaceExpression* rhs);
      IndexSpaceDifference(const Rect<DIM, T>& bounds);
      IndexSpaceDifference(const IndexSpaceDifference& rhs) = delete;
      virtual ~IndexSpaceDifference(void);
    public:
      IndexSpaceDifference& operator=(const IndexSpaceDifference& rhs) = delete;
    public:
      virtual void invalidate_operation(IndexSpaceExpression* source) override;
      virtual void remove_operation(void) override;
    protected:
      IndexSpaceExpression* const lhs;
      IndexSpaceExpression* const rhs;
    };

    class DifferenceOpCreator : public OperationCreator {
    public:
      DifferenceOpCreator(
          TypeTag t, IndexSpaceExpression* l, IndexSpaceExpression* r)
        : OperationCreator(), type_tag(t), lhs(l), rhs(r)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(DifferenceOpCreator* creator)
      {
        creator->produce(
            new IndexSpaceDifference<N::N, T>(creator->lhs, creator->rhs));
      }
    public:
      virtual void create_operation(void) override
      {
        NT_TemplateHelper::demux<DifferenceOpCreator>(type_tag, this);
      }
    public:
      const TypeTag type_tag;
      IndexSpaceExpression* const lhs;
      IndexSpaceExpression* const rhs;
    };

    /**
     * \class InternalExpression
     * This class stores an internal expression corresponding to a
     * group of rectangles that the runtime had to compute and not
     * derived from any other expressions. This can occur when creating
     * a custom sparse physical instance, but can also come from a
     * computing equivalence sets.
     */
    template<int DIM, typename T>
    class InternalExpression
      : public IndexSpaceOperationT<DIM, T>,
        public Heapify<InternalExpression<DIM, T>, LONG_LIFETIME> {
    public:
      InternalExpression(const Rect<DIM, T>* rects, size_t num_rects);
      InternalExpression(const InternalExpression<DIM, T>& rhs) = delete;
      virtual ~InternalExpression(void);
    public:
      InternalExpression& operator=(const InternalExpression& rhs) = delete;
    public:
      virtual void invalidate_operation(IndexSpaceExpression* source) override;
      virtual void remove_operation(void) override;
    };

    class InternalExpressionCreator {
    public:
      InternalExpressionCreator(TypeTag t, const Domain& d)
        : type_tag(t), domain(d)
      { }

      virtual void create_operation()
      {
        NT_TemplateHelper::demux<InternalExpressionCreator>(type_tag, this);
      }

      template<typename N, typename T>
      static inline void demux(InternalExpressionCreator* creator)
      {
        DomainT<N::N, T> domain = creator->domain;
        if (!domain.dense())
        {
          std::vector<Rect<N::N, T> > rects;
          for (Realm::IndexSpaceIterator<N::N, T> itr(domain); itr.valid;
               itr.step())
            rects.push_back(itr.rect);
          creator->result =
              new InternalExpression<N::N, T>(&rects.front(), rects.size());
        }
        else
          creator->result = new InternalExpression<N::N, T>(&domain.bounds, 1);
      }

      static IndexSpaceOperation* create_with_domain(
          TypeTag tag, const Domain& dom);
    public:
      const TypeTag type_tag;
      const Domain domain;
      IndexSpaceOperation* result;
    };

    /**
     * \class RemoteExpression
     * A copy of an expression that lives on a remote node.
     */
    template<int DIM, typename T>
    class RemoteExpression
      : public IndexSpaceOperationT<DIM, T>,
        public Heapify<RemoteExpression<DIM, T>, LONG_LIFETIME> {
    public:
      RemoteExpression(
          IndexSpaceExprID eid, DistributedID did, IndexSpaceOperation* op,
          TypeTag type_tag, Deserializer& derez);
      RemoteExpression(const RemoteExpression<DIM, T>& rhs) = delete;
      virtual ~RemoteExpression(void);
    public:
      RemoteExpression& operator=(const RemoteExpression& op) = delete;
    public:
      virtual void invalidate_operation(IndexSpaceExpression* source) override;
      virtual void remove_operation(void) override;
    };

    class RemoteExpressionCreator {
    public:
      RemoteExpressionCreator(IndexSpaceExprID e, TypeTag t, Deserializer& d)
        : expr_id(e), type_tag(t), derez(d), operation(nullptr)
      { }
    public:
      template<typename N, typename T>
      static inline void demux(RemoteExpressionCreator* creator)
      {
        IndexSpaceOperation* origin;
        creator->derez.deserialize(origin);
        DistributedID did;
        creator->derez.deserialize(did);
        legion_assert(creator->operation == nullptr);
        creator->operation = new RemoteExpression<N::N, T>(
            creator->expr_id, did, origin, creator->type_tag, creator->derez);
      }
    public:
      const IndexSpaceExprID expr_id;
      const TypeTag type_tag;
      Deserializer& derez;
      IndexSpaceOperation* operation;
    };

    /**
     * \class ExpressionTrieNode
     * This is a class for constructing a trie for index space
     * expressions so we can quickly detect commmon subexpression
     * in O(log N)^M time where N is the number of expressions
     * in total and M is the number of expression in the operation
     */
    class ExpressionTrieNode {
    public:
      ExpressionTrieNode(
          unsigned depth, IndexSpaceExprID expr_id,
          IndexSpaceExpression* op = nullptr);
      ExpressionTrieNode(const ExpressionTrieNode& rhs) = delete;
      ~ExpressionTrieNode(void);
    public:
      ExpressionTrieNode& operator=(const ExpressionTrieNode& rhs) = delete;
    public:
      bool find_operation(
          const std::vector<IndexSpaceExpression*>& expressions,
          IndexSpaceExpression*& result, ExpressionTrieNode*& last);
      IndexSpaceExpression* find_or_create_operation(
          const std::vector<IndexSpaceExpression*>& expressions,
          OperationCreator& creator);
      bool remove_operation(const std::vector<IndexSpaceExpression*>& exprs);
    public:
      const unsigned depth;
      const IndexSpaceExprID expr;
    protected:
      IndexSpaceExpression* local_operation;
      std::map<IndexSpaceExprID, IndexSpaceExpression*> operations;
      std::map<IndexSpaceExprID, ExpressionTrieNode*> nodes;
    protected:
      mutable LocalLock trie_lock;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/nodes/expression.inl"

#endif  // __LEGION_EXPRESSION_H__
