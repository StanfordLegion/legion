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

#ifndef __LEGION_TRACE_RECORDING_H__
#define __LEGION_TRACE_RECORDING_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/api/mapping.h"
#include "legion/api/types.h"
#include "legion/tools/types.h"
#include "legion/utilities/coordinates.h"
#include "legion/utilities/hasher.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct LogicalTraceInfo
     * Information about tracing needed for logical
     * dependence analysis.
     */
    struct LogicalTraceInfo {
    public:
      LogicalTraceInfo(
          Operation* op, unsigned idx, const RegionRequirement& r,
          const FieldMask& mask);
    public:
      LogicalTrace* const trace;
      const unsigned req_idx;
      const RegionRequirement& req;
      const bool skip_analysis;
    };

    /**
     * \interface TraceHashRecorder
     * An interface for recording hashes for traces
     */
    class TraceHashRecorder {
    public:
      virtual ~TraceHashRecorder(void) { }
    public:
      virtual bool record_operation_hash(
          Operation* op, Murmur3Hasher& hasher, uint64_t opidx) = 0;
      virtual bool record_operation_noop(Operation* op, uint64_t opidx) = 0;
      virtual bool record_operation_untraceable(
          Operation* op, uint64_t opidx) = 0;
    };

    /**
     * \struct UniqueInst
     * A small helper class for uniquely naming a physical
     * instance for the purposes of physical trace recording
     */
    struct UniqueInst {
    public:
      UniqueInst(void);
      UniqueInst(IndividualView* v);
    public:
      inline bool operator<(const UniqueInst& rhs) const
      {
        return (inst_did < rhs.inst_did);
      }
      inline bool operator==(const UniqueInst& rhs) const
      {
        return (inst_did == rhs.inst_did);
      }
      inline bool operator!=(const UniqueInst& rhs) const
      {
        return !this->operator==(rhs);
      }
    public:
      void serialize(Serializer& rez) const;
      void deserialize(Deserializer& derez);
      AddressSpaceID get_analysis_space(void) const;
    public:
      // Distributed ID for the physical manager
      DistributedID inst_did;
      // Distributed ID for the view to the instance
      DistributedID view_did;
      // Logical owner space for the view
      AddressSpaceID analysis_space;
      RegionTreeID tid;
    };

    /**
     * \interface PhysicalTraceRecorder
     * This interface describes all the methods that need to be
     * implemented for an object to act as the recorder of a
     * physical trace. They will be invoked by the PhysicalTraceInfo
     * object as part of trace capture.
     */
    class PhysicalTraceRecorder {
    public:
      virtual ~PhysicalTraceRecorder(void) { }
    public:
      virtual bool is_recording(void) const = 0;
      virtual void add_recorder_reference(void) = 0;
      virtual bool remove_recorder_reference(void) = 0;
      virtual void pack_recorder(Serializer& rez) = 0;
    public:
      virtual void record_replay_mapping(
          ApEvent lhs, unsigned op_kind, const TraceLocalID& tlid,
          std::set<RtEvent>& applied_events) = 0;
      virtual void request_term_event(ApUserEvent& term_event) = 0;
      virtual void record_create_ap_user_event(
          ApUserEvent& lhs, const TraceLocalID& tlid) = 0;
      virtual void record_trigger_event(
          ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid,
          std::set<RtEvent>& applied) = 0;
    public:
      virtual void record_merge_events(
          ApEvent& lhs, const ApEvent* rhs, size_t num_events,
          const TraceLocalID& tlid) = 0;
      virtual void record_merge_events(
          PredEvent& lhs, PredEvent e1, PredEvent e2,
          const TraceLocalID& tlid) = 0;
      // This collective barrier is managed by the operations and is auto
      // refreshed by the operations on each replay
      virtual void record_collective_barrier(
          ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
          size_t arrival_count) = 0;
      // This collective barrier is managed by the template and will be
      // refreshed as necessary when barrier generations are exhausted
      virtual ShardID record_barrier_creation(
          ApBarrier& bar, size_t total_arrivals) = 0;
      virtual void record_barrier_arrival(
          ApBarrier bar, ApEvent pre, size_t arrival_count,
          std::set<RtEvent>& applied, ShardID owner_shard) = 0;
    public:
      virtual void record_issue_copy(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent pred_guard, LgEvent src_unique,
          LgEvent dst_unique, int priority, CollectiveKind collective,
          bool record_effect) = 0;
      virtual void record_issue_across(
          const TraceLocalID& tlid, ApEvent& lhs,
          ApEvent collective_precondition, ApEvent copy_precondition,
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          CopyAcrossExecutor* executor) = 0;
      virtual void record_copy_insts(
          ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
          unsigned dst_idx, IndexSpaceExpression* expr,
          const UniqueInst& src_inst, const UniqueInst& dst_inst,
          const FieldMask& src_mask, const FieldMask& dst_mask,
          PrivilegeMode src_mode, PrivilegeMode dst_mode, ReductionOpID redop,
          std::set<RtEvent>& applied) = 0;
      typedef local::map<UniqueInst, FieldMask> AcrossInsts;
      virtual void record_across_insts(
          ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
          unsigned dst_idx, IndexSpaceExpression* expr,
          const AcrossInsts& src_insts, const AcrossInsts& dst_insts,
          PrivilegeMode src_mode, PrivilegeMode dst_mode, bool src_indirect,
          bool dst_indirect, std::set<RtEvent>& applied) = 0;
      virtual void record_indirect_insts(
          ApEvent indirect_done, ApEvent all_done, IndexSpaceExpression* expr,
          const AcrossInsts& insts, std::set<RtEvent>& applied,
          PrivilegeMode priv) = 0;
      virtual void record_issue_fill(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& fields, const void* fill_value,
          size_t fill_size, UniqueID fill_uid, FieldSpace handle,
          RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
          LgEvent unique_event, int priority, CollectiveKind collective,
          bool record_effect) = 0;
      virtual void record_fill_inst(
          ApEvent lhs, IndexSpaceExpression* expr, const UniqueInst& dst_inst,
          const FieldMask& fill_mask, std::set<RtEvent>& applied_events,
          const bool reduction_initialization) = 0;
    public:
      virtual void record_op_inst(
          const TraceLocalID& tlid, unsigned parent_req_index,
          const UniqueInst& inst, RegionNode* node, const RegionUsage& usage,
          const FieldMask& user_mask, bool update_validity,
          std::set<RtEvent>& applied) = 0;
      virtual void record_set_op_sync_event(
          ApEvent& lhs, const TraceLocalID& tlid) = 0;
      virtual void record_mapper_output(
          const TraceLocalID& tlid, const Mapper::MapTaskOutput& output,
          const std::deque<InstanceSet>& physical_instances, bool is_leaf,
          bool has_return_size, std::set<RtEvent>& applied_events) = 0;
      virtual void record_complete_replay(
          const TraceLocalID& tlid, ApEvent pre,
          std::set<RtEvent>& applied) = 0;
      virtual void record_reservations(
          const TraceLocalID& tlid, const std::map<Reservation, bool>& locks,
          std::set<RtEvent>& applied_events) = 0;
      virtual void record_future_allreduce(
          const TraceLocalID& tlid, const std::vector<Memory>& target_memories,
          size_t future_size) = 0;
    };

    /**
     * \class RemoteTraceRecorder
     * This class is used for handling tracing calls that are
     * performed on remote nodes from where the trace is being captured.
     */
    class RemoteTraceRecorder : public PhysicalTraceRecorder,
                                public Collectable {
    public:
      enum RemoteTraceKind {
        REMOTE_TRACE_RECORD_REPLAY_MAPPING,
        REMOTE_TRACE_REQUEST_TERM_EVENT,
        REMOTE_TRACE_CREATE_USER_EVENT,
        REMOTE_TRACE_TRIGGER_EVENT,
        REMOTE_TRACE_MERGE_EVENTS,
        REMOTE_TRACE_MERGE_PRED_EVENTS,
        REMOTE_TRACE_ISSUE_COPY,
        REMOTE_TRACE_COPY_INSTS,
        REMOTE_TRACE_ISSUE_FILL,
        REMOTE_TRACE_FILL_INST,
        REMOTE_TRACE_RECORD_OP_INST,
        REMOTE_TRACE_SET_OP_SYNC,
        REMOTE_TRACE_RECORD_MAPPER_OUTPUT,
        REMOTE_TRACE_COMPLETE_REPLAY,
        REMOTE_TRACE_ACQUIRE_RELEASE,
        REMOTE_TRACE_RECORD_BARRIER,
        REMOTE_TRACE_BARRIER_ARRIVAL,
      };
    public:
      RemoteTraceRecorder(
          AddressSpaceID origin, const TraceLocalID& tlid,
          PhysicalTemplate* tpl, DistributedID repl_did, TraceID tid);
      RemoteTraceRecorder(const RemoteTraceRecorder& rhs) = delete;
      virtual ~RemoteTraceRecorder(void);
    public:
      RemoteTraceRecorder& operator=(const RemoteTraceRecorder& rhs) = delete;
    public:
      virtual bool is_recording(void) const override { return true; }
      virtual void add_recorder_reference(void) override;
      virtual bool remove_recorder_reference(void) override;
      virtual void pack_recorder(Serializer& rez) override;
    public:
      virtual void record_replay_mapping(
          ApEvent lhs, unsigned op_kind, const TraceLocalID& tlid,
          std::set<RtEvent>& applied_events) override;
      virtual void request_term_event(ApUserEvent& term_event) override;
      virtual void record_create_ap_user_event(
          ApUserEvent& hs, const TraceLocalID& tlid) override;
      virtual void record_trigger_event(
          ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid,
          std::set<RtEvent>& applied) override;
    public:
      virtual void record_merge_events(
          ApEvent& lhs, const ApEvent* rhs, size_t num_rhs,
          const TraceLocalID& tlid) override;
      virtual void record_merge_events(
          PredEvent& lhs, PredEvent e1, PredEvent e2,
          const TraceLocalID& tlid) override;
      virtual void record_collective_barrier(
          ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
          size_t arrival_count) override;
      virtual ShardID record_barrier_creation(
          ApBarrier& bar, size_t total_arrivals) override;
      virtual void record_barrier_arrival(
          ApBarrier bar, ApEvent pre, size_t arrival_count,
          std::set<RtEvent>& applied, ShardID owner_shard) override;
    public:
      virtual void record_issue_copy(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent pred_guard, LgEvent src_unique,
          LgEvent dst_unique, int priority, CollectiveKind collective,
          bool record_effect) override;
      virtual void record_issue_across(
          const TraceLocalID& tlid, ApEvent& lhs,
          ApEvent collective_precondition, ApEvent copy_precondition,
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          CopyAcrossExecutor* executor) override;
      virtual void record_copy_insts(
          ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
          unsigned dst_idx, IndexSpaceExpression* expr,
          const UniqueInst& src_inst, const UniqueInst& dst_inst,
          const FieldMask& src_mask, const FieldMask& dst_mask,
          PrivilegeMode src_mode, PrivilegeMode dst_mode, ReductionOpID redop,
          std::set<RtEvent>& applied) override;
      virtual void record_across_insts(
          ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
          unsigned dst_idx, IndexSpaceExpression* expr,
          const AcrossInsts& src_insts, const AcrossInsts& dst_insts,
          PrivilegeMode src_mode, PrivilegeMode dst_mode, bool src_indirect,
          bool dst_indirect, std::set<RtEvent>& applied) override;
      virtual void record_indirect_insts(
          ApEvent indirect_done, ApEvent all_done, IndexSpaceExpression* expr,
          const AcrossInsts& insts, std::set<RtEvent>& applied,
          PrivilegeMode priv) override;
      virtual void record_issue_fill(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& fields, const void* fill_value,
          size_t fill_size, UniqueID fill_uid, FieldSpace handle,
          RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
          LgEvent unique_event, int priority, CollectiveKind collective,
          bool record_effect) override;
      virtual void record_fill_inst(
          ApEvent lhs, IndexSpaceExpression* expr, const UniqueInst& dst_inst,
          const FieldMask& fill_mask, std::set<RtEvent>& applied_events,
          const bool reduction_initialization) override;
    public:
      virtual void record_op_inst(
          const TraceLocalID& tlid, unsigned parent_req_index,
          const UniqueInst& inst, RegionNode* node, const RegionUsage& usage,
          const FieldMask& user_mask, bool update_validity,
          std::set<RtEvent>& applied) override;
      virtual void record_set_op_sync_event(
          ApEvent& lhs, const TraceLocalID& tlid) override;
      virtual void record_mapper_output(
          const TraceLocalID& tlid, const Mapper::MapTaskOutput& output,
          const std::deque<InstanceSet>& physical_instances, bool is_leaf,
          bool has_return_size, std::set<RtEvent>& applied_events) override;
      virtual void record_complete_replay(
          const TraceLocalID& tlid, ApEvent pre,
          std::set<RtEvent>& applied) override;
      virtual void record_reservations(
          const TraceLocalID& tlid, const std::map<Reservation, bool>& locks,
          std::set<RtEvent>& applied_events) override;
      virtual void record_future_allreduce(
          const TraceLocalID& tlid, const std::vector<Memory>& target_memories,
          size_t future_size) override;
    public:
      static PhysicalTraceRecorder* unpack_remote_recorder(
          Deserializer& derez, const TraceLocalID& tlid);
      static void handle_remote_update(
          Deserializer& derez, AddressSpaceID source);
      static void handle_remote_response(Deserializer& derez);
    protected:
      static void pack_src_dst_field(Serializer& rez, const CopySrcDstField& f);
      static void unpack_src_dst_field(Deserializer& derez, CopySrcDstField& f);
    public:
      const AddressSpaceID origin_space;
      PhysicalTemplate* const remote_tpl;
      const DistributedID repl_did;
      const TraceID trace_id;
    };

    /**
     * \struct TraceInfo
     * This provides a generic tracing struct for operations
     */
    struct TraceInfo {
    public:
      explicit TraceInfo(Operation* op);
      TraceInfo(SingleTask* task, RemoteTraceRecorder* rec);
      TraceInfo(const TraceInfo& info);
      ~TraceInfo(void);
    protected:
      TraceInfo(PhysicalTraceRecorder* rec, const TraceLocalID& tlid);
    public:
      inline void record_replay_mapping(
          ApEvent lhs, unsigned op_kind,
          std::set<RtEvent>& applied_events) const
      {
        base_sanity_check();
        rec->record_replay_mapping(lhs, op_kind, tlid, applied_events);
      }
      inline void request_term_event(ApUserEvent& term_event) const
      {
        base_sanity_check();
        rec->request_term_event(term_event);
      }
      inline void record_create_ap_user_event(ApUserEvent& result) const
      {
        base_sanity_check();
        rec->record_create_ap_user_event(result, tlid);
      }
      inline void record_trigger_event(
          ApUserEvent result, ApEvent rhs,
          std::set<RtEvent>& applied_events) const
      {
        base_sanity_check();
        rec->record_trigger_event(result, rhs, tlid, applied_events);
      }
      inline void record_merge_events(
          PredEvent& result, PredEvent e1, PredEvent e2) const
      {
        base_sanity_check();
        rec->record_merge_events(result, e1, e2, tlid);
      }
      inline void record_merge_events(
          ApEvent& result, const ApEvent* rhs, size_t num_rhs) const
      {
        base_sanity_check();
        rec->record_merge_events(result, rhs, num_rhs, tlid);
      }
      inline void record_collective_barrier(
          ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
          size_t arrival_count = 1) const
      {
        base_sanity_check();
        rec->record_collective_barrier(bar, pre, key, arrival_count);
      }
      inline ShardID record_barrier_creation(
          ApBarrier& bar, size_t total_arrivals) const
      {
        base_sanity_check();
        return rec->record_barrier_creation(bar, total_arrivals);
      }
      inline void record_barrier_arrival(
          ApBarrier bar, ApEvent pre, size_t arrival_count,
          std::set<RtEvent>& applied, ShardID owner) const
      {
        base_sanity_check();
        rec->record_barrier_arrival(bar, pre, arrival_count, applied, owner);
      }
      inline void record_op_sync_event(ApEvent& result) const
      {
        base_sanity_check();
        rec->record_set_op_sync_event(result, tlid);
      }
      inline void record_mapper_output(
          const TraceLocalID& tlid, const Mapper::MapTaskOutput& output,
          const std::deque<InstanceSet>& physical_instances, bool is_leaf,
          bool has_return_size, std::set<RtEvent>& applied)
      {
        base_sanity_check();
        rec->record_mapper_output(
            tlid, output, physical_instances, is_leaf, has_return_size,
            applied);
      }
      inline void record_complete_replay(
          std::set<RtEvent>& applied, ApEvent pre = ApEvent::NO_AP_EVENT) const
      {
        base_sanity_check();
        rec->record_complete_replay(tlid, pre, applied);
      }
      inline void record_reservations(
          const TraceLocalID& tlid,
          const std::map<Reservation, bool>& reservations,
          std::set<RtEvent>& applied) const
      {
        base_sanity_check();
        rec->record_reservations(tlid, reservations, applied);
      }
      inline void record_future_allreduce(
          const TraceLocalID& tlid, const std::vector<Memory>& target_memories,
          size_t future_size) const
      {
        base_sanity_check();
        rec->record_future_allreduce(tlid, target_memories, future_size);
      }
    protected:
      inline void base_sanity_check(void) const
      {
        legion_assert(recording);
        legion_assert(rec != nullptr);
        legion_assert(rec->is_recording());
      }
      static PhysicalTraceRecorder* init_recorder(Operation* op);
      static TraceLocalID init_tlid(Operation* op);
    protected:
      PhysicalTraceRecorder* const rec;
    public:
      const TraceLocalID tlid;
      const bool recording;
    };

    /**
     * \struct PhysicalTraceInfo
     * A Physical trace info is a TraceInfo but with special
     * information about the region requirement being traced
     */
    struct PhysicalTraceInfo : public TraceInfo {
    public:
      PhysicalTraceInfo(Operation* op, unsigned index);
      PhysicalTraceInfo(
          const TraceInfo& info, unsigned index, bool update_validity = true);
      // Weird argument order to help the compiler avoid ambiguity
      PhysicalTraceInfo(
          unsigned src_idx, const TraceInfo& info, unsigned dst_idx);
      PhysicalTraceInfo(const PhysicalTraceInfo& rhs);
    protected:
      PhysicalTraceInfo(
          const TraceLocalID& tlid, unsigned src_idx, unsigned dst_idx,
          bool update_validity, PhysicalTraceRecorder* rec);
    public:
      inline void record_issue_copy(
          ApEvent& result, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent pred_guard, LgEvent src_unique,
          LgEvent dst_unique, int priority, CollectiveKind collective,
          bool record_effect) const
      {
        sanity_check();
        rec->record_issue_copy(
            tlid, result, expr, src_fields, dst_fields, reservations,
            src_tree_id, dst_tree_id, precondition, pred_guard, src_unique,
            dst_unique, priority, collective, record_effect);
      }
      inline void record_issue_fill(
          ApEvent& result, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& fields, const void* fill_value,
          size_t fill_size, UniqueID fill_uid, FieldSpace handle,
          RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
          LgEvent unique_event, int priority, CollectiveKind collective,
          bool record_effect) const
      {
        sanity_check();
        rec->record_issue_fill(
            tlid, result, expr, fields, fill_value, fill_size, fill_uid, handle,
            tree_id, precondition, pred_guard, unique_event, priority,
            collective, record_effect);
      }
      inline void record_issue_across(
          ApEvent& result, ApEvent collective_precondition,
          ApEvent copy_precondition, ApEvent src_indirect_precondition,
          ApEvent dst_indirect_precondition, CopyAcrossExecutor* executor) const
      {
        sanity_check();
        rec->record_issue_across(
            tlid, result, collective_precondition, copy_precondition,
            src_indirect_precondition, dst_indirect_precondition, executor);
      }
      inline void record_fill_inst(
          ApEvent lhs, IndexSpaceExpression* expr, const UniqueInst& inst,
          const FieldMask& fill_mask, std::set<RtEvent>& applied,
          const bool reduction_initialization) const
      {
        sanity_check();
        rec->record_fill_inst(
            lhs, expr, inst, fill_mask, applied, reduction_initialization);
      }
      inline void record_copy_insts(
          ApEvent lhs, IndexSpaceExpression* expr, const UniqueInst& src_inst,
          const UniqueInst& dst_inst, const FieldMask& src_mask,
          const FieldMask& dst_mask, ReductionOpID redop,
          std::set<RtEvent>& applied) const
      {
        sanity_check();
        rec->record_copy_insts(
            lhs, tlid, index, dst_index, expr, src_inst, dst_inst, src_mask,
            dst_mask, LEGION_READ_PRIV,
            (redop > 0) ? LEGION_REDUCE_PRIV : LEGION_WRITE_PRIV, redop,
            applied);
      }
      typedef local::map<UniqueInst, FieldMask> AcrossInsts;
      inline void record_across_insts(
          ApEvent lhs, unsigned idx1, unsigned idx2, PrivilegeMode mode1,
          PrivilegeMode mode2, IndexSpaceExpression* expr,
          AcrossInsts& src_insts, AcrossInsts& dst_insts, bool src_indirect,
          bool dst_indirect, std::set<RtEvent>& applied) const
      {
        sanity_check();
        rec->record_across_insts(
            lhs, tlid, idx1, idx2, expr, src_insts, dst_insts, mode1, mode2,
            src_indirect, dst_indirect, applied);
      }
      inline void record_indirect_insts(
          ApEvent indirect_done, ApEvent all_done, IndexSpaceExpression* expr,
          AcrossInsts& insts, std::set<RtEvent>& applied,
          PrivilegeMode privilege) const
      {
        sanity_check();
        rec->record_indirect_insts(
            indirect_done, all_done, expr, insts, applied, privilege);
      }
      // Not inline because we need to call a method on Operation
      void record_op_inst(
          const RegionUsage& usage, const FieldMask& user_mask,
          const UniqueInst& inst, RegionNode* node, Operation* op,
          std::set<RtEvent>& applied) const;
    public:
      void pack_trace_info(Serializer& rez) const;
      static PhysicalTraceInfo unpack_trace_info(Deserializer& derez);
    private:
      inline void sanity_check(void) const
      {
#ifdef LEGION_DEBUG
        base_sanity_check();
#endif
        legion_assert(index != -1U);
        legion_assert(dst_index != -1U);
      }
    public:
      const unsigned index;
      const unsigned dst_index;
      const bool update_validity;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TRACE_RECORDING_H__
