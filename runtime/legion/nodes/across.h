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

#include "legion/kernel/metatask.h"
#include "legion/api/geometry.h"
#include "legion/api/requirements.h"
#include "legion/tools/profiler.h"
#include "legion/utilities/bitmask.h"

#ifndef __LEGION_COPY_ACROSS_NODE_H__
#define __LEGION_COPY_ACROSS_NODE_H__

namespace Legion {
  namespace Internal {

    /**
     * \struct IndirectRecord
     * A small helper class for performing exchanges of
     * instances for indirection copies
     */
    struct IndirectRecord {
    public:
      IndirectRecord(void) { }
      IndirectRecord(
          const RegionRequirement& req, const InstanceSet& insts,
          size_t total_points);
    public:
      void serialize(Serializer& rez) const;
      void deserialize(Deserializer& derez);
    public:
      // In the same order as the fields for the actual copy
      std::vector<PhysicalInstance> instances;
      // Only valid if profiling is enabled
      std::vector<LgEvent> instance_events;
      IndexSpace index_space;
      Domain domain;
      ApEvent domain_ready;
    };

    /**
     * \class CopyAcrossHelper
     * A small helper class for performing copies between regions
     * from diferrent region trees
     */
    class CopyAcrossHelper {
    public:
      CopyAcrossHelper(
          const FieldMask& full, const std::vector<unsigned>& src,
          const std::vector<unsigned>& dst)
        : full_mask(full), src_indexes(src), dst_indexes(dst)
      {
        legion_assert(src_indexes.size() == dst_indexes.size());
        legion_assert(std::is_sorted(src_indexes.begin(), src_indexes.end()));
        offsets.resize(dst_indexes.size());
      }
    public:
      const FieldMask& full_mask;
      const std::vector<unsigned>& src_indexes;
      const std::vector<unsigned>& dst_indexes;
      std::map<unsigned, unsigned> forward_map;
      std::map<unsigned, unsigned> backward_map;
    public:
      void compute_across_offsets(
          const FieldMask& src_mask, std::vector<CopySrcDstField>& dst_fields);
      FieldMask convert_src_to_dst(const FieldMask& src_mask);
      FieldMask convert_dst_to_src(const FieldMask& dst_mask);
    public:
      unsigned convert_src_to_dst(unsigned index);
      unsigned convert_dst_to_src(unsigned index);
    public:
      std::vector<CopySrcDstField> offsets;
      op::deque<std::pair<FieldMask, FieldMask> > compressed_cache;
    };

    /**
     * \class CopyAcrossExecutor
     * This is a virtual interface for performing copies between
     * two different fields including with lots of different kinds
     * of indirections and transforms.
     */
    class CopyAcrossExecutor : public InstanceNameClosure {
    public:
      struct DeferCopyAcrossArgs : public LgTaskArgs<DeferCopyAcrossArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_COPY_ACROSS_TASK_ID;
      public:
        DeferCopyAcrossArgs(void) = default;
        DeferCopyAcrossArgs(
            CopyAcrossExecutor* e, Operation* o, PredEvent guard,
            ApEvent copy_pre, ApEvent src_pre, ApEvent dst_pre, bool replay,
            bool recurrent, unsigned stage);
        void execute(void) const;
      public:
        CopyAcrossExecutor* executor;
        Operation* op;
        PredEvent guard;
        ApEvent copy_precondition;
        ApEvent src_indirect_precondition;
        ApEvent dst_indirect_precondition;
        ApUserEvent done_event;
        unsigned stage;
        bool replay;
        bool recurrent_replay;
      };
    public:
      CopyAcrossExecutor(
          const bool preimages, const std::map<Reservation, bool>& rsrvs)
        : reservations(rsrvs), priority(0), compute_preimages(preimages)
      { }
      virtual ~CopyAcrossExecutor(void) { }
    public:
      // From InstanceNameClosure
      virtual LgEvent find_instance_name(PhysicalInstance inst) const = 0;
      virtual DistributedID find_instance_subspace(
          PhysicalInstance inst) const = 0;
      virtual DistributedID find_copy_expression(void) const = 0;
      virtual ReductionOpID find_redop(void) const = 0;
    public:
      virtual ApEvent execute(
          Operation* op, PredEvent pred_guard, ApEvent copy_precondition,
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          const PhysicalTraceInfo& trace_info, const bool replay = false,
          const bool recurrent_replay = false, const unsigned stage = 0) = 0;
      virtual void record_trace_immutable_indirection(bool source) = 0;
      virtual void release_shadow_instances(void) = 0;
    public:
      static void handle_deferred_copy_across(const void* args);
    public:
      // Reservations that must be acquired for performing this copy
      // across and whether they need to be acquired with exclusive
      // permissions or not
      const std::map<Reservation, bool> reservations;
      // Priority for this copy across
      int priority;
      // Say whether we should be computing preimages or not
      const bool compute_preimages;
    };

    /**
     * \class CopyAcrossUnstructured
     * Untyped base class for all unstructured copies between fields
     */
    class CopyAcrossUnstructured : public CopyAcrossExecutor {
    public:
      CopyAcrossUnstructured(
          IndexSpaceExpression* exp, const bool preimages,
          const std::map<Reservation, bool>& rsrvs)
        : CopyAcrossExecutor(preimages, rsrvs), expr(exp),
          src_indirect_field(0), dst_indirect_field(0),
          src_indirect_instance(PhysicalInstance::NO_INST),
          dst_indirect_instance(PhysicalInstance::NO_INST)
      { }
      virtual ~CopyAcrossUnstructured(void);
    public:
      // From InstanceNameClosure
      virtual LgEvent find_instance_name(PhysicalInstance inst) const override;
      virtual DistributedID find_instance_subspace(
          PhysicalInstance inst) const override;
      virtual DistributedID find_copy_expression(void) const override;
      virtual ReductionOpID find_redop(void) const override;
    public:
      virtual ApEvent execute(
          Operation* op, PredEvent pred_guard, ApEvent copy_precondition,
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          const PhysicalTraceInfo& trace_info, const bool replay = false,
          const bool recurrent_replay = false,
          const unsigned stage = 0) override = 0;
      virtual void record_trace_immutable_indirection(bool source) override = 0;
      virtual void release_shadow_instances(void) override = 0;
    public:
      void initialize_source_fields(
          const RegionRequirement& req, const InstanceSet& instances,
          const PhysicalTraceInfo& trace_info);
      void initialize_destination_fields(
          const RegionRequirement& req, const InstanceSet& instances,
          const PhysicalTraceInfo& trace_info, const bool exclusive_redop);
      void initialize_source_indirections(
          std::vector<IndirectRecord>& records,
          const RegionRequirement& src_req, const RegionRequirement& idx_req,
          const InstanceRef& indirect_instance, const bool is_range_indirection,
          const bool possible_out_of_range);
      void initialize_destination_indirections(
          std::vector<IndirectRecord>& records,
          const RegionRequirement& dst_req, const RegionRequirement& idx_req,
          const InstanceRef& indirect_instance, const bool is_range_indirection,
          const bool possible_out_of_range, const bool possible_aliasing,
          const bool exclusive_redop);
    public:
      IndexSpaceExpression* const expr;
    protected:
      mutable LocalLock preimage_lock;
    public:
      // All the entries in these data structures are ordered by the
      // order of the fields in the original region requirements
      std::vector<CopySrcDstField> src_fields, dst_fields;
      std::vector<LgEvent> src_unique_events, dst_unique_events;
      RegionTreeID src_tree_id, dst_tree_id;
      unsigned unique_indirections_identifier;
    public:
      // All the 'instances' in the entries in these data strctures are
      // ordered by the order of the fields in the origin region requirements
      std::vector<IndirectRecord> src_indirections, dst_indirections;
      FieldID src_indirect_field, dst_indirect_field;
      PhysicalInstance src_indirect_instance, dst_indirect_instance;
      LgEvent src_indirect_instance_event, dst_indirect_instance_event;
      TypeTag src_indirect_type, dst_indirect_type;
      std::vector<unsigned> nonempty_indexes;
      // Shadow indirection instances
      // Only one copy of this since we only do it in gather/scatter cases
      // so we can use the same data structure for either kind of indirection
      struct ShadowInstance {
        PhysicalInstance instance;
        ApEvent ready;
        LgEvent unique_event;
      };
      std::map<Memory, ShadowInstance> shadow_instances;
      // Only valid when profiling for looking up instance names
      std::map<PhysicalInstance, LgEvent> profiling_shadow_instances;
      // Zips with current_src/dst_preimages
      std::vector<ApEvent> indirection_preconditions;
    public:
      RtEvent prev_done;
      ApEvent last_copy;
    public:
      bool is_range_indirection;
      bool possible_src_out_of_range;
      bool possible_dst_out_of_range;
      bool possible_dst_aliasing;
    };

    /**
     * \class CopyAcrossExecutorT
     * This is the templated version of the copy-across executor. It is
     * templated on the dimensions and coordinate type of the copy space
     * for the copy operation.
     */
    template<int DIM, typename T>
    class CopyAcrossUnstructuredT : public CopyAcrossUnstructured {
    public:
      typedef typename Realm::CopyIndirection<DIM, T>::Base CopyIndirection;
    public:
      struct ComputePreimagesHelper {
      public:
        ComputePreimagesHelper(
            CopyAcrossUnstructuredT<DIM, T>* u, Operation* o, ApEvent p, bool s)
          : unstructured(u), op(o), precondition(p), source(s)
        { }
      public:
        template<typename N2, typename T2>
        static inline void demux(ComputePreimagesHelper* helper)
        {
          helper->result = helper->unstructured
                               ->template perform_compute_preimages<N2::N, T2>(
                                   helper->new_preimages, helper->op,
                                   helper->precondition, helper->source);
        }
      public:
        std::vector<DomainT<DIM, T> > new_preimages;
        CopyAcrossUnstructuredT<DIM, T>* const unstructured;
        Operation* const op;
        const ApEvent precondition;
        ApEvent result;
        const bool source;
      };
      struct RebuildIndirectionsHelper {
      public:
        RebuildIndirectionsHelper(
            CopyAcrossUnstructuredT<DIM, T>* u, Operation* o, ApEvent e, bool s)
          : unstructured(u), op(o), indirection_event(e), source(s), empty(true)
        { }
      public:
        template<typename N2, typename T2>
        static inline void demux(RebuildIndirectionsHelper* helper)
        {
          helper->empty =
              helper->unstructured->template rebuild_indirections<N2::N, T2>(
                  helper->op, helper->indirection_event, helper->source);
        }
      public:
        CopyAcrossUnstructuredT<DIM, T>* const unstructured;
        Operation* const op;
        const ApEvent indirection_event;
        const bool source;
        bool empty;
      };
    public:
      CopyAcrossUnstructuredT(
          IndexSpaceExpression* expr, const DomainT<DIM, T>& domain,
          ApEvent domain_ready, const std::map<Reservation, bool>& rsrvs,
          const bool compute_preimages, const bool shadow_indirections);
      virtual ~CopyAcrossUnstructuredT(void);
    public:
      virtual ApEvent execute(
          Operation* op, PredEvent pred_guard, ApEvent copy_precondition,
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          const PhysicalTraceInfo& trace_info, const bool replay = false,
          const bool recurrent_replay = false,
          const unsigned stage = 0) override;
      virtual void record_trace_immutable_indirection(bool source) override;
      virtual void release_shadow_instances(void) override;
    public:
      ApEvent issue_individual_copies(
          Operation* op, const ApEvent precondition,
          const Realm::ProfilingRequestSet& requests);
      template<int D2, typename T2>
      ApEvent perform_compute_preimages(
          std::vector<DomainT<DIM, T> >& preimages, Operation* op,
          ApEvent precondition, const bool source);
      template<int D2, typename T2>
      bool rebuild_indirections(
          Operation* op, ApEvent indirection_event, const bool source);
    protected:
      Realm::InstanceLayoutGeneric* select_shadow_layout(bool source) const;
      PhysicalInstance allocate_shadow_indirection(
          Memory memory, UniqueID creator_uid, bool source,
          LgEvent& unique_event);
      ApEvent update_shadow_indirection(
          PhysicalInstance shadow, LgEvent unique_event,
          ApEvent indirection_event, const DomainT<DIM, T>& update_domain,
          Operation* op, size_t field_size, bool source) const;
    public:
      const DomainT<DIM, T> copy_domain;
      const ApEvent copy_domain_ready;
      const bool shadow_indirections;
    protected:
      std::deque<std::vector<DomainT<DIM, T> > > src_preimages, dst_preimages;
      std::vector<DomainT<DIM, T> > current_src_preimages,
          current_dst_preimages;
      std::vector<const CopyIndirection*> indirections;
      // Realm performs better if you can issue a separate copy for each of the
      // preimages so it doesn't have to do address splitting. Therefore when
      // we compute preimages and we only have a gather or a scatter copy then
      // we will attempt to issue individual copies for such cases. Note that
      // we don't bother doing this for full-indirection copies though as then
      // we would need to do the full quadratic intersection between each of
      // the source and destination preimages.
      std::vector<std::vector<unsigned> > individual_field_indexes;
      // For help in creating shadow indirections
      Realm::InstanceLayoutGeneric* shadow_layout;
      std::deque<ApEvent> src_preimage_preconditions;
      std::deque<ApEvent> dst_preimage_preconditions;
      bool need_src_indirect_precondition, need_dst_indirect_precondition;
      bool src_indirect_immutable_for_tracing;
      bool dst_indirect_immutable_for_tracing;
      bool has_empty_preimages;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/nodes/across.inl"

#endif  // __LEGION_COPY_ACROSS_NODE_H__
